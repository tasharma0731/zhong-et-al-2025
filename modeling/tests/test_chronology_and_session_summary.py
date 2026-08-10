from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from modeling.chronology import chronological_trial_pairs_and_windows
from modeling.session_summary import (
    balanced_trial_covariate_mean,
    balanced_trial_ids,
    recording_neuron_seed,
    summarize_trial_balanced_dprime,
    trial_balanced_dprime,
)


def _trial_frames(trial_ids: list[int], roles: list[int]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "frame_id": np.arange(len(trial_ids), dtype=np.int64),
            "trial_id": trial_ids,
            "stimulus_role": roles,
        }
    )


def test_chronological_windows_discard_pre_boundary_role_imbalance() -> None:
    # Role 2 gets ahead early. Trial 2 cannot leak into the second window after
    # the first window closes at role-0 trial 4.
    frames = _trial_frames(
        list(range(12)),
        [2, 2, 2, 0, 0, 2, 0, 0, 2, 0, 2, 0],
    )
    pairs, windows = chronological_trial_pairs_and_windows(frames, size=2)

    assert windows[["first_trial_id", "last_trial_id"]].values.tolist() == [
        [0, 4],
        [5, 8],
    ]
    assert 2 not in pairs.loc[pairs["pair_id"] >= 2, "trial_a_id"].tolist()
    assert pairs[["trial_a_id", "trial_b_id"]].values.tolist() == [
        [0, 3],
        [1, 4],
        [5, 6],
        [8, 7],
    ]


def test_chronological_windows_are_strictly_disjoint_and_ordered() -> None:
    frames = _trial_frames(
        list(range(16)),
        [2, 0, 2, 0, 0, 2, 2, 0, 2, 0, 0, 2, 0, 2, 2, 0],
    )
    pairs, windows = chronological_trial_pairs_and_windows(frames, size=2)

    assert len(windows) == 4
    assert np.all(
        windows["last_trial_id"].to_numpy()[:-1]
        < windows["first_trial_id"].to_numpy()[1:]
    )
    selected_trials: list[set[int]] = []
    for window in windows.itertuples(index=False):
        block = pairs.loc[
            pairs["pair_id"].between(window.start_pair_id, window.stop_pair_id)
        ]
        trial_set = set(block["trial_a_id"]) | set(block["trial_b_id"])
        assert len(trial_set) == 4
        selected_trials.append(trial_set)
    assert all(
        earlier.isdisjoint(later)
        for index, earlier in enumerate(selected_trials)
        for later in selected_trials[index + 1 :]
    )


def test_chronology_returns_schema_for_insufficient_trials() -> None:
    frames = _trial_frames([0, 1, 2], [2, 0, 2])
    pairs, windows = chronological_trial_pairs_and_windows(frames, size=2)

    assert pairs.empty and windows.empty
    assert {"trial_a_id", "trial_b_id", "pair_id"}.issubset(pairs.columns)
    assert {"first_trial_id", "last_trial_id", "window_id"}.issubset(
        windows.columns
    )


def test_chronology_rejects_non_integer_window_size() -> None:
    frames = _trial_frames([0, 1, 2, 3], [2, 0, 2, 0])
    with pytest.raises(ValueError, match="positive integer"):
        chronological_trial_pairs_and_windows(frames, size=1.5)  # type: ignore[arg-type]


def test_balanced_trials_cover_larger_role_without_randomness() -> None:
    frames = _trial_frames(
        list(range(8)),
        [2, 2, 2, 2, 2, 0, 0, 0],
    )
    first, counts = balanced_trial_ids(frames)
    second, _ = balanced_trial_ids(frames.sample(frac=1.0, random_state=9))

    assert counts == {2: 5, 0: 3}
    assert first[2].tolist() == [0, 2, 4]
    assert first[0].tolist() == [5, 6, 7]
    np.testing.assert_array_equal(first[2], second[2])
    np.testing.assert_array_equal(first[0], second[0])


def test_trial_balanced_dprime_uses_trial_means_and_preserves_zero_denominator() -> None:
    # Two components are also two neurons. Each trial has two identical frames,
    # so the expected trial responses can be read directly below.
    u = np.eye(2)
    trial_responses = np.array(
        [
            [1.0, 3.0, 0.0, 0.0],
            [5.0, 5.0, 5.0, 5.0],
        ]
    )
    v = np.repeat(trial_responses, 2, axis=1)
    frames = pd.DataFrame(
        {
            "frame_id": np.arange(8),
            "trial_id": np.repeat(np.arange(4), 2),
            "stimulus_role": np.repeat([2, 2, 0, 0], 2),
        }
    )
    result = trial_balanced_dprime(u, v, frames, neuron_ids=np.array([0, 1]))

    # Neuron 0: 2 * (mean([1, 3]) - mean([0, 0])) / (1 + 0) = 4.
    assert result.values[0] == 4.0
    assert np.isnan(result.values[1])
    assert result.denominators.tolist() == [1.0, 0.0]
    assert result.valid_denominator.tolist() == [True, False]

    neurons = pd.DataFrame(
        {"neuron_id": [0, 1], "area_group": ["mHV", "mHV"]}
    )
    summary = summarize_trial_balanced_dprime(result, neurons, threshold=0.3).iloc[0]
    assert summary["n_neurons"] == 2
    assert summary["n_valid_dprime"] == 1
    assert summary["n_invalid_dprime"] == 1
    assert summary["n_invalid_denominator"] == 1
    assert summary["frac_invalid_dprime"] == 0.5
    assert summary["median"] == 4.0
    assert summary["frac_leaf_selective"] == 1.0
    assert summary["frac_circle_selective"] == 0.0


def test_trial_means_give_each_trial_equal_weight_when_frame_counts_differ() -> None:
    # Role-2 trial means are [0, 2], even though the second trial contributes
    # three times as many eligible frames. Role-0 trial means are [0, 0].
    v = np.array([[0.0, 2.0, 2.0, 2.0, 0.0, 0.0]])
    frames = pd.DataFrame(
        {
            "frame_id": np.arange(6),
            "trial_id": [0, 1, 1, 1, 2, 3],
            "stimulus_role": [2, 2, 2, 2, 0, 0],
        }
    )
    result = trial_balanced_dprime(
        np.ones((1, 1)), v, frames, neuron_ids=np.array([0])
    )

    # 2 * (1 - 0) / (1 + 0) = 2. Frame pooling would give a different answer.
    assert result.values.tolist() == [2.0]


def test_recording_seed_is_stable_and_changes_with_base_seed() -> None:
    assert recording_neuron_seed("recording", 2025) == recording_neuron_seed(
        "recording", 2025
    )
    assert recording_neuron_seed("recording", 2025) != recording_neuron_seed(
        "recording", 2026
    )


def test_balanced_covariate_averages_trials_instead_of_frames() -> None:
    trial_frames = pd.DataFrame(
        {
            "frame_id": np.arange(6),
            "trial_id": [0, 1, 1, 1, 2, 3],
            "stimulus_role": [2, 2, 2, 2, 0, 0],
        }
    )
    frame_covariate = pd.DataFrame(
        {"frame_id": np.arange(6), "run_speed": [0, 6, 6, 6, 2, 4]}
    )
    mean, valid_trials = balanced_trial_covariate_mean(
        frame_covariate,
        trial_frames,
        {2: np.array([0, 1]), 0: np.array([2, 3])},
        value_column="run_speed",
    )

    assert mean == 3.0  # mean of trial means [0, 6, 2, 4]
    assert valid_trials == 4
