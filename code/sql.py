from __future__ import annotations

from pathlib import Path

from behavior import BehaviorDB
from database import ZhongDB
from joiner import Joiner


def connect(
    *,
    root: str | Path | None = None,
    cache: str | Path | None = None,
    database: str | Path | None = None,
    mount: bool = True,
) -> ZhongDB:
    return ZhongDB(root=root, cache=cache, database=database, mount=mount)


def setup(
    *,
    root: str | Path | None = None,
    cache: str | Path | None = None,
    database: str | Path | None = None,
    mount: bool = True,
    report: bool = True,
) -> ZhongDB:
    db = connect(root=root, cache=cache, database=database, mount=mount)
    if report:
        print(f"DuckDB: {db.database_path}")
        print(db.schema().to_string(index=False))
    return db


__all__ = ["BehaviorDB", "Joiner", "ZhongDB", "connect", "setup"]
