from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import duckdb
import pandas as pd


class DataFrameSQL:
    def __init__(self, **tables: pd.DataFrame) -> None:
        self._connection = duckdb.connect(":memory:")
        self._tables: dict[str, pd.DataFrame] = {}
        for name, frame in tables.items():
            self.register(name, frame)

    @property
    def tables(self) -> tuple[str, ...]:
        return tuple(sorted(self._tables))

    def register(self, name: str, frame: pd.DataFrame) -> pd.DataFrame:
        if not name.isidentifier():
            raise ValueError("Table name must be a Python and SQL identifier")
        if name in self._tables:
            self._connection.unregister(name)
        stored = frame.copy()
        self._tables[name] = stored
        self._connection.register(name, stored)
        return stored

    def query(
        self,
        statement: str,
        parameters: Iterable[Any] | Mapping[str, Any] | None = None,
    ) -> pd.DataFrame:
        cursor = (
            self._connection.execute(statement, parameters)
            if parameters is not None
            else self._connection.execute(statement)
        )
        return cursor.fetchdf()

    def schema(self, table: str | None = None) -> pd.DataFrame:
        if table is None:
            return pd.DataFrame(
                [
                    {"table": name, "rows": len(frame), "columns": len(frame.columns)}
                    for name, frame in sorted(self._tables.items())
                ]
            )
        if table not in self._tables:
            raise KeyError(f"Unknown table {table!r}; choose from {self.tables}")
        return self.query(f'DESCRIBE SELECT * FROM "{table}"')

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> "DataFrameSQL":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()


__all__ = ["DataFrameSQL"]
