from __future__ import annotations

import numpy as np
import pandas as pd


def paired_trial_windows(
    pairs: pd.DataFrame,
    *,
    size: int = 20,
    stride: int = 1,
) -> pd.DataFrame:
    if size < 1 or stride < 1:
        raise ValueError("size and stride must be positive")
    required = {"pair_id", "first_trial_id", "last_trial_id", "trial_midpoint"}
    missing = required - set(pairs.columns)
    if missing:
        raise ValueError(f"pairs is missing {sorted(missing)}")

    starts = np.arange(0, max(len(pairs) - size + 1, 0), stride, dtype=np.int64)
    rows = []
    for window_id, start in enumerate(starts):
        stop = int(start + size)
        block = pairs.iloc[start:stop]
        rows.append(
            {
                "window_id": window_id,
                "start_pair_id": int(block["pair_id"].iloc[0]),
                "stop_pair_id": int(block["pair_id"].iloc[-1]),
                "pair_count": len(block),
                "first_trial_id": int(block["first_trial_id"].min()),
                "last_trial_id": int(block["last_trial_id"].max()),
                "trial_midpoint": float(block["trial_midpoint"].mean()),
                "progress": float((start + (size / 2.0)) / len(pairs)),
            }
        )
    return pd.DataFrame.from_records(
        rows,
        columns=[
            "window_id",
            "start_pair_id",
            "stop_pair_id",
            "pair_count",
            "first_trial_id",
            "last_trial_id",
            "trial_midpoint",
            "progress",
        ],
    )


def equal_pair_windows(
    pairs: pd.DataFrame,
    *,
    bins: int = 5,
) -> pd.DataFrame:
    if bins < 1 or len(pairs) < bins:
        raise ValueError("bins must be positive and no greater than the number of pairs")
    required = {"pair_id", "first_trial_id", "last_trial_id", "trial_midpoint"}
    missing = required - set(pairs.columns)
    if missing:
        raise ValueError(f"pairs is missing {sorted(missing)}")

    rows = []
    for trial_bin, indices in enumerate(np.array_split(np.arange(len(pairs)), bins), start=1):
        block = pairs.iloc[indices]
        rows.append(
            {
                "window_id": trial_bin - 1,
                "trial_bin": trial_bin,
                "start_pair_id": int(block["pair_id"].iloc[0]),
                "stop_pair_id": int(block["pair_id"].iloc[-1]),
                "pair_count": len(block),
                "first_trial_id": int(block["first_trial_id"].min()),
                "last_trial_id": int(block["last_trial_id"].max()),
                "trial_midpoint": float(block["trial_midpoint"].mean()),
                "progress": float((indices.mean() + 0.5) / len(pairs)),
            }
        )
    return pd.DataFrame.from_records(rows)
