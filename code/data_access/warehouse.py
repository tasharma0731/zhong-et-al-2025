from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Iterable, Mapping

import duckdb
import pandas as pd

import drive


DATABASE_NAME = "zhong.duckdb"
IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def database_path(database: str | Path | None, cache: Path) -> Path:
    selected = Path(database).expanduser() if database is not None else cache / DATABASE_NAME
    return selected.resolve()


def create(
    path: Path,
    tables: Mapping[str, pd.DataFrame],
) -> duckdb.DuckDBPyConnection:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial")
    partial.unlink(missing_ok=True)

    connection = duckdb.connect(str(partial))
    try:
        write_tables(connection, tables)
        connection.execute("CHECKPOINT")
        connection.close()
        partial.replace(path)
    except BaseException:
        connection.close()
        partial.unlink(missing_ok=True)
        raise

    return duckdb.connect(str(path))


def memory(tables: Mapping[str, pd.DataFrame]) -> duckdb.DuckDBPyConnection:
    connection = duckdb.connect()
    write_tables(connection, tables)
    return connection


def open_database(path: Path) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(path), read_only=True)


def write_tables(
    connection: duckdb.DuckDBPyConnection,
    tables: Mapping[str, pd.DataFrame],
) -> None:
    for name, frame in tables.items():
        connection.register("_frame", frame)
        connection.execute(f'CREATE TABLE "{name}" AS SELECT * FROM _frame')
        connection.unregister("_frame")


def query(
    connection: duckdb.DuckDBPyConnection,
    statement: str,
    parameters: Iterable[Any] | Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    if parameters is None:
        return connection.execute(statement).fetchdf()
    return connection.execute(statement, parameters).fetchdf()


def registration(
    name: str,
    frame: pd.DataFrame,
    existing_tables: Iterable[str],
) -> pd.DataFrame:
    if not IDENTIFIER.fullmatch(name):
        raise ValueError("Table name must be a simple SQL identifier")
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas.DataFrame")
    if name in existing_tables:
        raise ValueError(f"Table already exists: {name}")
    return frame.copy()


def export(
    connection: duckdb.DuckDBPyConnection,
    source: Path,
    destination: str | Path,
) -> Path:
    target = Path(destination).expanduser().resolve()
    if target == source:
        return target
    return drive.atomic_copy(source, target)


__all__ = [
    "create",
    "database_path",
    "export",
    "memory",
    "open_database",
    "query",
    "registration",
]
