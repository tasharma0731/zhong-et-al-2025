from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


FRAME_COLUMNS = {
    "frame_id",
    "trial_id",
    "stimulus_role",
    "valid_trial",
    "is_moving",
    "in_texture",
}


def eligible_trial_frames(
    frames: pd.DataFrame,
    *,
    roles: Sequence[int] = (2, 0),
    min_frames: int = 3,
) -> pd.DataFrame:
    missing = FRAME_COLUMNS - set(frames.columns)
    if missing:
        raise ValueError(f"frames is missing {sorted(missing)}")
    if len(roles) != 2 or roles[0] == roles[1]:
        raise ValueError("roles must contain two distinct stimulus roles")
    if min_frames < 1:
        raise ValueError("min_frames must be positive")

    selected = frames.loc[
        frames["valid_trial"]
        & frames["is_moving"]
        & frames["in_texture"]
        & frames["stimulus_role"].isin(roles),
        ["frame_id", "trial_id", "stimulus_role"],
    ].dropna()
    selected = selected.astype(
        {"frame_id": "int64", "trial_id": "int64", "stimulus_role": "int64"}
    )
    counts = selected.groupby("trial_id")["frame_id"].transform("size")
    return selected.loc[counts >= min_frames].sort_values("frame_id").reset_index(drop=True)


def ordinal_trial_pairs(
    trial_frames: pd.DataFrame,
    *,
    role_a: int = 2,
    role_b: int = 0,
) -> pd.DataFrame:
    trials = (
        trial_frames[["trial_id", "stimulus_role"]]
        .drop_duplicates()
        .sort_values("trial_id")
    )
    roles_per_trial = trials.groupby("trial_id")["stimulus_role"].nunique()
    if (roles_per_trial > 1).any():
        raise ValueError("each trial must have exactly one stimulus role")

    trials_a = trials.loc[trials["stimulus_role"] == role_a, "trial_id"].to_numpy()
    trials_b = trials.loc[trials["stimulus_role"] == role_b, "trial_id"].to_numpy()
    pair_count = min(len(trials_a), len(trials_b))
    trials_a, trials_b = trials_a[:pair_count], trials_b[:pair_count]

    return pd.DataFrame(
        {
            "pair_id": np.arange(pair_count, dtype=np.int64),
            "trial_a_id": trials_a,
            "trial_b_id": trials_b,
            "first_trial_id": np.minimum(trials_a, trials_b),
            "last_trial_id": np.maximum(trials_a, trials_b),
            "trial_midpoint": (trials_a + trials_b) / 2.0,
            "role_a": role_a,
            "role_b": role_b,
        }
    )
