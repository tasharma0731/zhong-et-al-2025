from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, cast, TYPE_CHECKING

import duckdb
import pandas as pd

import drive
from data_access import behavior as normalization
from data_access import warehouse

if TYPE_CHECKING:
    from database import ZhongDB


DATABASE_NAME = "zhong.duckdb"
BEHAVIOR_TABLES = (
    "behavior_sessions",
    "behavior_trials",
    "behavior_frames",
    "behavior_stimuli",
    "behavior_events",
    "behavior_fields",
)
EXPECTED_ROWS = {
    "release": 1,
    "files": 297,
    "experiment_rows": 142,
    "behavior_sessions": 142,
    "memberships": 133,
    "experiments": 23,
    "recording_files": 400,
    "recordings": 89,
    "mice": 19,
    "behavior_trials": 63_177,
    "behavior_frames": 3_222_952,
    "behavior_stimuli": 594,
    "behavior_events": 280_202,
    "behavior_fields": 59,
    "behavior_trial_counts": 142,
}
UNIQUE_KEYS = {
    "files": ("filename",),
    "experiment_rows": ("experiment", "source_row"),
    "behavior_sessions": ("behavior_session_id",),
    "memberships": ("experiment", "recording_id"),
    "experiments": ("experiment",),
    "recording_files": ("recording_id", "experiment", "layer", "filename"),
    "recordings": ("recording_id",),
    "mice": ("mouse",),
    "behavior_trials": ("behavior_session_id", "trial_id"),
    "behavior_frames": ("behavior_session_id", "frame_id"),
    "behavior_stimuli": ("behavior_session_id", "stimulus_id"),
    "behavior_events": ("behavior_session_id", "event_type", "event_id"),
    "behavior_fields": ("raw_field",),
    "behavior_trial_counts": ("behavior_session_id",),
}
JSON_COLUMNS = {
    "experiment_rows": ("depth_json", "stimulus_ids_json", "source_json"),
    "behavior_sessions": ("depth_json", "stimulus_ids_json", "source_json"),
    "recordings": ("experiments_json",),
    "mice": ("cohorts_json",),
}


class BehaviorDB:
    def __init__(self, path: str | Path) -> None:
        self.database_path = Path(path).expanduser().resolve()
        self._connection = duckdb.connect(str(self.database_path), read_only=True)

    @classmethod
    def open(
        cls,
        source: ZhongDB,
        *,
        database: str | Path | None = None,
        rebuild: bool = False,
        report: bool = False,
    ) -> BehaviorDB:
        path = (
            Path(database).expanduser().resolve()
            if database is not None
            else source.cache / DATABASE_NAME
        )
        if rebuild or not path.is_file():
            build(source, path, report=report)
        opened = cls(path)
        validate_database(opened._connection)
        return opened

    @property
    def tables(self) -> tuple[str, ...]:
        result = self.query("SHOW TABLES")
        return tuple(sorted(result.iloc[:, 0].astype(str)))

    def query(
        self,
        statement: str,
        parameters: Iterable[Any] | Mapping[str, Any] | None = None,
    ) -> pd.DataFrame:
        return warehouse.query(self._connection, statement, parameters)

    def table(self, name: str, *, limit: int | None = 1_000) -> pd.DataFrame:
        if name not in self.tables:
            raise KeyError(f"Unknown table {name!r}; choose from {self.tables}")
        suffix = "" if limit is None else f" LIMIT {int(limit)}"
        return self.query(f'SELECT * FROM "{name}"{suffix}')

    def schema(self, table: str | None = None) -> pd.DataFrame:
        if table is not None:
            if table not in self.tables:
                raise KeyError(f"Unknown table {table!r}; choose from {self.tables}")
            return self.query(f'DESCRIBE SELECT * FROM "{table}"')
        return self.query(
            """
            SELECT table_name AS table,
                   estimated_size AS rows,
                   column_count AS columns
            FROM duckdb_tables()
            ORDER BY table_name
            """
        )

    def integrity(self) -> pd.DataFrame:
        return integrity(self._connection)

    def export(self, path: str | Path) -> Path:
        return drive.atomic_copy(self.database_path, Path(path).expanduser().resolve())

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> BehaviorDB:
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"BehaviorDB({len(self.tables)} tables; {self.database_path})"


def build(source: ZhongDB, path: str | Path, *, report: bool = False) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f".{destination.name}.partial")
    partial.unlink(missing_ok=True)
    connection = duckdb.connect(str(partial))

    try:
        catalog_tables = {
            name: source.table(name)
            for name in source.catalog_tables
            if name != "behavior_sessions"
        }
        warehouse.write_tables(connection, catalog_tables)
        append_table(connection, "behavior_fields", normalization.field_surface())

        sessions = source.table("behavior_sessions")
        summaries = []
        completed = 0
        for filename, file_sessions in sessions.groupby("filename", sort=True):
            bundle = source.load_file(str(filename))
            validate_bundle(file_sessions, bundle, str(filename))
            for row in file_sessions.to_dict("records"):
                session = cast(dict[str, Any], row)
                normalized = normalization.tables(
                    session,
                    bundle[session["behavior_key"]],
                )
                summaries.append(normalized.pop("behavior_session_summary"))
                for name, frame in normalized.items():
                    append_table(connection, name, frame)
                completed += 1
                if report:
                    print(f"[{completed:03d}/{len(sessions):03d}] {row['behavior_session_id']}")

        summary = pd.concat(summaries, ignore_index=True)
        enriched_sessions = sessions.merge(
            summary,
            on="behavior_session_id",
            how="left",
            validate="one_to_one",
        )
        append_table(connection, "behavior_sessions", enriched_sessions)
        connection.execute(
            """
            CREATE TABLE behavior_trial_counts AS
            SELECT behavior_session_id,
                   experiment,
                   recording_id,
                   COUNT(*)::BIGINT AS trial_count
            FROM behavior_trials
            GROUP BY behavior_session_id, experiment, recording_id
            """
        )
        validate_database(connection, len(sessions))
        connection.execute("CHECKPOINT")
        connection.close()
        partial.replace(destination)
        return destination
    except BaseException:
        connection.close()
        partial.unlink(missing_ok=True)
        raise


def append_table(
    connection: duckdb.DuckDBPyConnection,
    name: str,
    frame: pd.DataFrame,
) -> None:
    connection.register("_frame", frame)
    exists = scalar_result(connection.execute(
        "SELECT COUNT(*) FROM duckdb_tables() WHERE table_name = ?", [name]
    ))
    statement = (
        f'INSERT INTO "{name}" SELECT * FROM _frame'
        if exists
        else f'CREATE TABLE "{name}" AS SELECT * FROM _frame'
    )
    try:
        connection.execute(statement)
    finally:
        connection.unregister("_frame")


def validate_bundle(
    sessions: pd.DataFrame,
    bundle: Any,
    filename: str,
) -> None:
    if not isinstance(bundle, Mapping):
        raise drive.DriveDataError(f"Behavior file is not a mapping: {filename}")
    expected = set(sessions["behavior_key"].astype(str))
    actual = {str(name) for name in bundle}
    if expected != actual:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise drive.DriveDataError(
            f"Behavior keys disagree for {filename}; missing={missing}, unexpected={unexpected}"
        )


def integrity(connection: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    checks: list[dict[str, Any]] = []

    def add(name: str, violations: int) -> None:
        checks.append({"check": name, "violations": violations, "passed": violations == 0})

    tables = {row[0] for row in connection.execute("SHOW TABLES").fetchall()}
    add("table set", len(set(EXPECTED_ROWS) ^ tables))

    for table, expected in EXPECTED_ROWS.items():
        actual = scalar_result(connection.execute(f'SELECT COUNT(*) FROM "{table}"'))
        add(f"{table} row count", abs(int(actual) - expected))

    for table, keys in UNIQUE_KEYS.items():
        key_sql = ", ".join(f'"{key}"' for key in keys)
        duplicates = scalar_result(connection.execute(
            f'SELECT COUNT(*) FROM ('
            f'SELECT {key_sql} FROM "{table}" '
            f'GROUP BY {key_sql} HAVING COUNT(*) > 1)'
        ))
        add(f"{table} unique key", int(duplicates))

    for table, columns in JSON_COLUMNS.items():
        for column in columns:
            invalid = scalar_result(connection.execute(
                f'SELECT COUNT(*) FROM "{table}" '
                f'WHERE "{column}" IS NOT NULL AND NOT json_valid("{column}")'
            ))
            add(f"{table}.{column} valid JSON", int(invalid))

    references = {
        "memberships recording": """
            SELECT COUNT(*) FROM memberships AS child
            LEFT JOIN recordings AS parent USING (recording_id)
            WHERE parent.recording_id IS NULL
        """,
        "memberships experiment": """
            SELECT COUNT(*) FROM memberships AS child
            LEFT JOIN experiments AS parent USING (experiment)
            WHERE parent.experiment IS NULL
        """,
        "recording files recording": """
            SELECT COUNT(*) FROM recording_files AS child
            LEFT JOIN recordings AS parent USING (recording_id)
            WHERE parent.recording_id IS NULL
        """,
        "recording files file": """
            SELECT COUNT(*) FROM recording_files AS child
            LEFT JOIN files AS parent USING (filename)
            WHERE parent.filename IS NULL
        """,
        "recordings mouse": """
            SELECT COUNT(*) FROM recordings AS child
            LEFT JOIN mice AS parent USING (mouse)
            WHERE parent.mouse IS NULL
        """,
        "behavior sessions recording": """
            SELECT COUNT(*) FROM behavior_sessions AS child
            LEFT JOIN recordings AS parent USING (recording_id)
            WHERE parent.recording_id IS NULL
        """,
        "behavior sessions experiment": """
            SELECT COUNT(*) FROM behavior_sessions AS child
            LEFT JOIN experiments AS parent USING (experiment)
            WHERE parent.experiment IS NULL
        """,
        "behavior trials session": """
            SELECT COUNT(*) FROM behavior_trials AS child
            LEFT JOIN behavior_sessions AS parent USING (behavior_session_id)
            WHERE parent.behavior_session_id IS NULL
        """,
        "behavior frames session": """
            SELECT COUNT(*) FROM behavior_frames AS child
            LEFT JOIN behavior_sessions AS parent USING (behavior_session_id)
            WHERE parent.behavior_session_id IS NULL
        """,
        "behavior stimuli session": """
            SELECT COUNT(*) FROM behavior_stimuli AS child
            LEFT JOIN behavior_sessions AS parent USING (behavior_session_id)
            WHERE parent.behavior_session_id IS NULL
        """,
        "behavior events session": """
            SELECT COUNT(*) FROM behavior_events AS child
            LEFT JOIN behavior_sessions AS parent USING (behavior_session_id)
            WHERE parent.behavior_session_id IS NULL
        """,
        "behavior frames trial": """
            SELECT COUNT(*) FROM behavior_frames AS child
            LEFT JOIN behavior_trials AS parent
              USING (behavior_session_id, trial_id)
            WHERE child.valid_trial AND parent.behavior_session_id IS NULL
        """,
        "behavior events trial": """
            SELECT COUNT(*) FROM behavior_events AS child
            LEFT JOIN behavior_trials AS parent
              USING (behavior_session_id, trial_id)
            WHERE child.trial_id IS NOT NULL AND parent.behavior_session_id IS NULL
        """,
        "session row counts": """
            SELECT COUNT(*)
            FROM behavior_sessions AS session
            LEFT JOIN behavior_trial_counts AS trials USING (behavior_session_id)
            LEFT JOIN (
                SELECT behavior_session_id, COUNT(*) AS frame_count
                FROM behavior_frames
                GROUP BY behavior_session_id
            ) AS frames USING (behavior_session_id)
            WHERE session.trial_count != trials.trial_count
               OR session.frame_count != frames.frame_count
        """,
    }
    for name, statement in references.items():
        add(name, int(scalar_result(connection.execute(statement))))

    return pd.DataFrame(checks)


def validate_database(connection: duckdb.DuckDBPyConnection, *_args: Any) -> None:
    failures = integrity(connection).query("not passed")
    if not failures.empty:
        raise drive.DriveDataError(
            "ZhongDB integrity checks failed:\n" + failures.to_string(index=False)
        )


def scalar_result(cursor: duckdb.DuckDBPyConnection) -> Any:
    row = cursor.fetchone()
    if row is None:
        raise drive.DriveDataError("The database returned no result")
    return row[0]


__all__ = ["BEHAVIOR_TABLES", "BehaviorDB", "build", "integrity"]
