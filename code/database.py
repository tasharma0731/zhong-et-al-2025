from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

import drive
from data_access import arrays, catalog, release, warehouse


class ZhongDB:
    def __init__(
        self,
        *,
        root: str | Path | None = None,
        cache: str | Path | None = None,
        database: str | Path | None = None,
        mount: bool = True,
    ) -> None:
        self.root, self.cache = drive.locate_release(root=root, cache=cache, mount=mount)

        release_metadata, file_records = release.read(self.root)
        files = catalog.files(file_records)
        imaging_index = self._read_imaging_index(files)
        experiment_rows = catalog.experiment_rows(imaging_index)

        release_metadata["root"] = str(self.root) if self.root is not None else None
        self._tables = catalog.tables(release_metadata, files, experiment_rows)
        self._catalog_tables = tuple(sorted(self._tables))

        self.database_path = warehouse.database_path(database, self.cache)
        if self.database_path.is_file():
            from behavior import validate_database

            self._connection = warehouse.open_database(self.database_path)
            validate_database(self._connection)
        else:
            self._connection = warehouse.memory(self._tables)

    @property
    def connected(self) -> bool:
        return self.root is not None

    @property
    def tables(self) -> tuple[str, ...]:
        return tuple(sorted(self._tables))

    @property
    def catalog_tables(self) -> tuple[str, ...]:
        return self._catalog_tables

    def table(self, name: str) -> pd.DataFrame:
        if name not in self._tables:
            raise KeyError(f"Unknown table {name!r}; choose from {self.tables}")
        return self._tables[name].copy()

    def query(
        self,
        statement: str,
        parameters: Iterable[Any] | Mapping[str, Any] | None = None,
    ) -> pd.DataFrame:
        return warehouse.query(self._connection, statement, parameters)

    def register(self, name: str, frame: pd.DataFrame) -> pd.DataFrame:
        stored = warehouse.registration(name, frame, self._database_tables())
        self._tables[name] = stored
        self._connection.register(name, stored)
        return stored

    def export(self, path: str | Path) -> Path:
        if not self.database_path.is_file():
            self.behavior().close()
        return warehouse.export(self._connection, self.database_path, path)

    def schema(self, table: str | None = None) -> pd.DataFrame:
        if table is None:
            return self.query(
                """
                SELECT table_name AS table,
                       estimated_size AS rows,
                       column_count AS columns
                FROM duckdb_tables()
                ORDER BY table_name
                """
            )
        available = self._database_tables()
        if table not in available:
            raise KeyError(f"Unknown table {table!r}; choose from {available}")
        return self.query(f'DESCRIBE SELECT * FROM "{table}"')

    def fetch_file(
        self,
        filename: str,
        *,
        max_gib: float = drive.DATASET["default_max_gib"],
    ) -> Path:
        return self._fetch(self._file(filename), max_gib)

    def load_file(
        self,
        filename: str,
        *,
        max_gib: float = drive.DATASET["default_max_gib"],
    ) -> Any:
        return self._load(self._file(filename), max_gib)

    def fetch(
        self,
        recording_id: str,
        layer: str,
        *,
        experiment: str | None = None,
        behavior_key: str | None = None,
        max_gib: float = drive.DATASET["default_max_gib"],
    ) -> Path:
        if layer == "behavior":
            row = self._behavior_session(
                recording_id=recording_id,
                experiment=experiment,
                behavior_key=behavior_key,
            )
            return self._fetch(row, max_gib)
        return self._fetch(self._recording_file(recording_id, layer, experiment), max_gib)

    def load(
        self,
        recording_id: str,
        layer: str,
        *,
        experiment: str | None = None,
        behavior_key: str | None = None,
        max_gib: float = drive.DATASET["default_max_gib"],
    ) -> Any:
        if layer == "behavior":
            row = self._behavior_session(
                recording_id=recording_id,
                experiment=experiment,
                behavior_key=behavior_key,
            )
            return self._load_behavior(row, max_gib)
        row = self._recording_file(recording_id, layer, experiment)
        return self._load(row, max_gib)

    def behavior_session(
        self,
        behavior_session_id: str | None = None,
        *,
        recording_id: str | None = None,
        experiment: str | None = None,
        behavior_key: str | None = None,
    ) -> dict[str, Any]:
        return self._behavior_session(
            behavior_session_id=behavior_session_id,
            recording_id=recording_id,
            experiment=experiment,
            behavior_key=behavior_key,
        )

    def load_behavior(
        self,
        behavior_session_id: str,
        *,
        max_gib: float = drive.DATASET["default_max_gib"],
    ) -> Any:
        return self._load_behavior(
            self._behavior_session(behavior_session_id=behavior_session_id),
            max_gib,
        )

    def behavior(
        self,
        *,
        database: str | Path | None = None,
        rebuild: bool = False,
        report: bool = False,
    ) -> Any:
        from behavior import BehaviorDB

        selected = warehouse.database_path(database, self.cache)
        replaces_connection = selected == self.database_path and (
            rebuild or not selected.is_file()
        )
        if replaces_connection:
            self._connection.close()
        try:
            result = BehaviorDB.open(
                self,
                database=selected,
                rebuild=rebuild,
                report=report,
            )
        finally:
            if replaces_connection:
                self._connection = (
                    warehouse.open_database(self.database_path)
                    if self.database_path.is_file()
                    else warehouse.memory(self._tables)
                )
        return result

    def integrity(self) -> pd.DataFrame:
        if not self.database_path.is_file():
            raise drive.DriveDataError("Build zhong.duckdb before checking integrity")
        from behavior import integrity

        return integrity(self._connection)

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> "ZhongDB":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        mode = "mounted" if self.connected else "metadata only"
        files = len(self._tables["files"])
        recordings = len(self._tables["recordings"])
        return f"ZhongDB({mode}; {files} files; {recordings} recordings; {self.database_path})"

    def _read_imaging_index(self, files: pd.DataFrame) -> Mapping[str, Any]:
        if self.root is None:
            return release.read_experiment_index()

        matches = files.loc[files["filename"] == "Imaging_Exp_info.npy"]
        if len(matches) != 1:
            raise drive.DriveDataError("The release must contain Imaging_Exp_info.npy")
        value = self._load(row_dict(matches.iloc[0]), max_gib=1.0)
        if not isinstance(value, Mapping):
            raise drive.DriveDataError("Imaging_Exp_info.npy should contain a dictionary")
        return value

    def _database_tables(self) -> tuple[str, ...]:
        result = self.query("SHOW TABLES")
        return tuple(sorted(result.iloc[:, 0].astype(str)))

    def _fetch(self, row: Mapping[str, Any], max_gib: float) -> Path:
        return drive.fetch_file(row, root=self.root, cache=self.cache, max_gib=max_gib)

    def _load(self, row: Mapping[str, Any], max_gib: float) -> Any:
        return arrays.load(row, root=self.root, cache=self.cache, max_gib=max_gib)

    def _load_behavior(self, row: Mapping[str, Any], max_gib: float) -> Any:
        value = self._load(row, max_gib)
        return arrays.session_behavior(
            value,
            str(row["filename"]),
            str(row["behavior_key"]),
        )

    def _file(self, filename: str) -> dict[str, Any]:
        if Path(filename).name != filename:
            raise drive.DriveDataError("Choose an exact filename, not a path")
        matches = self._tables["files"]
        matches = matches[matches["filename"] == filename]
        if len(matches) != 1:
            raise drive.DriveDataError(f"File is not in the pinned release: {filename!r}")
        return row_dict(matches.iloc[0])

    def _behavior_session(
        self,
        behavior_session_id: str | None = None,
        *,
        recording_id: str | None = None,
        experiment: str | None = None,
        behavior_key: str | None = None,
    ) -> dict[str, Any]:
        matches = self._tables["behavior_sessions"]
        if behavior_session_id is not None:
            matches = matches[matches["behavior_session_id"] == behavior_session_id]
        else:
            if recording_id is None:
                raise ValueError("Choose behavior_session_id or recording_id")
            matches = matches[matches["recording_id"] == recording_id]
            if experiment is not None:
                matches = matches[matches["experiment"] == experiment]
            if behavior_key is not None:
                matches = matches[matches["behavior_key"] == behavior_key]

        if matches.empty:
            raise drive.DriveDataError("No matching behavior session")
        if len(matches) != 1:
            choices = matches[
                ["behavior_session_id", "experiment", "behavior_key"]
            ].sort_values("behavior_session_id")
            raise drive.DriveDataError(
                "Choose one behavior session:\n" + choices.to_string(index=False)
            )
        return row_dict(matches.iloc[0])

    def _recording_file(
        self,
        recording_id: str,
        layer: str,
        experiment: str | None,
    ) -> dict[str, Any]:
        selected_layer = "reduced_neural" if layer == "svd" else layer
        if selected_layer not in set(catalog.LAYERS.values()):
            raise ValueError(
                "layer must be behavior, reduced_neural (or svd), full_neural, or retinotopy"
            )

        matches = self._tables["recording_files"]
        matches = matches[
            (matches["recording_id"] == recording_id)
            & (matches["layer"] == selected_layer)
        ]
        if selected_layer == "behavior" and experiment is not None:
            matches = matches[matches["experiment"] == experiment]

        if matches.empty:
            detail = f" and experiment {experiment!r}" if experiment else ""
            raise drive.DriveDataError(
                f"No {selected_layer!r} file for recording {recording_id!r}{detail}"
            )
        if len(matches) != 1:
            choices = matches["experiment"].dropna().sort_values().tolist()
            raise drive.DriveDataError(
                f"Choose experiment= for {recording_id!r}; behavior labels are {choices}"
            )
        return row_dict(matches.iloc[0])


def row_dict(series: pd.Series) -> dict[str, Any]:
    return {str(name): value for name, value in series.items()}


def main() -> None:
    parser = ArgumentParser(description="Build zhong.duckdb from the Zhong dataset")
    parser.add_argument("database", nargs="?", help="output DuckDB path")
    options = parser.parse_args()
    with ZhongDB(database=options.database) as database:
        database.behavior(database=options.database).close()
        print(database.database_path)


__all__ = ["ZhongDB"]


if __name__ == "__main__":
    main()
