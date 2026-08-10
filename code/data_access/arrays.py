from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

import drive


def load(
    row: Mapping[str, Any],
    *,
    root: Path | None,
    cache: Path,
    max_gib: float,
) -> Any:
    filename = str(row["filename"])
    value = drive.load_numpy(
        row,
        root=root,
        cache=cache,
        max_gib=max_gib,
        allow_pickle=filename.endswith(".npy") or row["category"] == "area_outlines",
    )
    return reduced_neural(value, filename) if row["category"] == "reduced_neural" else value


def reduced_neural(value: Any, filename: str) -> dict[str, np.ndarray]:
    if not isinstance(value, Mapping):
        raise drive.DriveDataError(f"Reduced-neural file is not a mapping: {filename}")
    result = {
        str(name): array
        for name, array in value.items()
        if isinstance(array, np.ndarray)
    }
    if not {"U", "V"}.issubset(result):
        raise drive.DriveDataError(f"Reduced-neural file has no U/V arrays: {filename}")
    return result


def session_behavior(value: Any, filename: str, recording_id: str) -> Any:
    if not isinstance(value, Mapping) or recording_id not in value:
        raise drive.DriveDataError(
            f"Behavior file {filename} has no session {recording_id!r}"
        )
    return value[recording_id]


__all__ = ["load", "session_behavior"]
