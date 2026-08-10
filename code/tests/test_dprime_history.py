from __future__ import annotations

import numpy as np
import pandas as pd

from dprime import (
    balanced_area_neurons,
    eligible_trial_frames,
    ordinal_trial_pairs,
    paired_trial_windows,
    summarize_dprime_history,
    windowed_svd_dprime,
)


def frame_table() -> pd.DataFrame:
    trial_id = np.repeat(np.arange(6), 2)
    role = np.repeat([2, 0, 2, 0, 2, 0], 2)
    return pd.DataFrame(
        {
            "frame_id": np.arange(12),
            "trial_id": trial_id,
            "stimulus_role": role,
            "valid_trial": True,
            "is_moving": True,
            "in_texture": True,
        }
    )


def test_pairing_and_windows_are_explicit() -> None:
    trial_frames = eligible_trial_frames(frame_table(), min_frames=2)
    pairs = ordinal_trial_pairs(trial_frames)
    windows = paired_trial_windows(pairs, size=2, stride=1)

    assert pairs[["trial_a_id", "trial_b_id"]].values.tolist() == [
        [0, 1],
        [2, 3],
        [4, 5],
    ]
    assert windows[["start_pair_id", "stop_pair_id"]].values.tolist() == [
        [0, 1],
        [1, 2],
    ]


def test_windowed_svd_dprime_matches_direct_reconstruction() -> None:
    rng = np.random.default_rng(4)
    u = rng.normal(size=(3, 6))
    v = rng.normal(size=(3, 12))
    trial_frames = eligible_trial_frames(frame_table(), min_frames=2)
    pairs = ordinal_trial_pairs(trial_frames)
    windows = paired_trial_windows(pairs, size=2)
    neuron_ids = np.arange(u.shape[1])

    actual = windowed_svd_dprime(
        u,
        v,
        trial_frames,
        pairs,
        windows,
        neuron_ids=neuron_ids,
    )

    activity = u.T @ v
    expected = []
    for window in windows.itertuples():
        pair_block = pairs.iloc[window.start_pair_id : window.stop_pair_id + 1]
        trials_a = pair_block["trial_a_id"].to_numpy()
        trials_b = pair_block["trial_b_id"].to_numpy()
        frames_a = trial_frames.loc[trial_frames["trial_id"].isin(trials_a), "frame_id"]
        frames_b = trial_frames.loc[trial_frames["trial_id"].isin(trials_b), "frame_id"]
        a = activity[:, frames_a]
        b = activity[:, frames_b]
        direct = 2.0 * (a.mean(1) - b.mean(1)) / (a.std(1) + b.std(1) + 1e-8)
        expected.append(np.nan_to_num(direct, nan=0.0, posinf=0.0, neginf=0.0))

    np.testing.assert_allclose(actual, np.stack(expected), rtol=1e-10, atol=1e-10)


def test_balanced_area_summary_preserves_equal_samples() -> None:
    neurons = pd.DataFrame(
        {
            "neuron_id": np.arange(8),
            "area_group": ["V1"] * 5 + ["mHV"] * 3,
        }
    )
    selected = balanced_area_neurons(
        neurons,
        areas=("V1", "mHV"),
        per_area=3,
        seed=2,
    )
    windows = pd.DataFrame(
        {
            "window_id": [0],
            "start_pair_id": [0],
            "stop_pair_id": [1],
            "pair_count": [2],
            "first_trial_id": [0],
            "last_trial_id": [3],
            "trial_midpoint": [1.5],
            "progress": [0.5],
        }
    )
    summary = summarize_dprime_history(
        np.array([[0.4, -0.4, 0.0, 0.6, 0.1, -0.2]]),
        windows,
        selected,
        include_all_areas=True,
    )

    assert selected.groupby("area_group").size().to_dict() == {"V1": 3, "mHV": 3}
    assert set(summary["area"]) == {"all", "V1", "mHV"}
    assert summary.set_index("area")["n_neurons"].to_dict() == {
        "all": 6,
        "V1": 3,
        "mHV": 3,
    }
    pooled = summary.set_index("area").loc["all"]
    assert pooled["median"] == 0.05
    assert pooled["frac_leaf_selective"] == 2 / 6
    assert pooled["frac_circle_selective"] == 1 / 6
