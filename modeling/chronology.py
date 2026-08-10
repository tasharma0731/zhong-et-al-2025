"""Strictly chronological, balanced stimulus windows for forecasting."""

from __future__ import annotations

import gc

import numpy as np
import pandas as pd

from .session_summary import (
    BEHAVIOR_ESTIMATOR,
    DENOMINATOR_ESTIMATOR,
    NEURON_SEED_METHOD,
    TRIAL_RESPONSE_ESTIMATOR,
    dprime_for_balanced_trial_ids,
    recording_neuron_seed,
    summarize_trial_balanced_dprime,
)


WINDOW_ESTIMATOR = "strict_chronological_disjoint_trial_balanced_dprime"
WINDOW_ESTIMATOR_VERSION = "2.0.0"
WINDOW_CONSTRUCTION = "balanced_roles_after_previous_window_boundary"


PAIR_COLUMNS = (
    "pair_id",
    "trial_a_id",
    "trial_b_id",
    "first_trial_id",
    "last_trial_id",
    "trial_midpoint",
    "role_a",
    "role_b",
)
WINDOW_COLUMNS = (
    "window_id",
    "start_pair_id",
    "stop_pair_id",
    "pair_count",
    "first_trial_id",
    "last_trial_id",
    "trial_midpoint",
    "progress",
)


def chronological_trial_pairs_and_windows(
    trial_frames: pd.DataFrame,
    *,
    size: int,
    role_a: int = 2,
    role_b: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create disjoint blocks with ``size`` trials of each stimulus role.

    After a block closes, every trial in the next block must occur after the
    previous block's final trial. Trials from an overrepresented role that fall
    before the boundary are deliberately discarded. This costs some data but
    makes current→next prediction genuinely chronological.
    """

    if (
        isinstance(size, (bool, np.bool_))
        or not isinstance(size, (int, np.integer))
        or size < 1
        or role_a == role_b
    ):
        raise ValueError("size must be a positive integer and roles must differ")
    required = {"trial_id", "stimulus_role"}
    missing = required - set(trial_frames.columns)
    if missing:
        raise ValueError(f"trial_frames is missing {sorted(missing)}")
    identifiers = trial_frames[["trial_id", "stimulus_role"]].apply(
        pd.to_numeric, errors="coerce"
    )
    if not np.isfinite(identifiers.to_numpy(dtype=np.float64)).all():
        raise ValueError("trial identifiers and stimulus roles must be finite")
    if not np.equal(identifiers, np.floor(identifiers)).all().all():
        raise ValueError("trial identifiers and stimulus roles must be integers")
    trials = identifiers.astype("int64").drop_duplicates().sort_values("trial_id")
    roles_per_trial = trials.groupby("trial_id")["stimulus_role"].nunique()
    if (roles_per_trial > 1).any():
        raise ValueError("each trial must have exactly one stimulus role")
    by_role = {
        role: trials.loc[trials["stimulus_role"].eq(role), "trial_id"].to_numpy(
            dtype=np.int64
        )
        for role in (role_a, role_b)
    }

    pair_rows: list[dict[str, object]] = []
    window_rows: list[dict[str, object]] = []
    boundary = -np.inf
    pair_id = 0
    max_trial_id = int(trials["trial_id"].max()) if len(trials) else 0
    while True:
        selected = {
            role: values[values > boundary][:size]
            for role, values in by_role.items()
        }
        if any(len(values) < size for values in selected.values()):
            break
        trials_a, trials_b = selected[role_a], selected[role_b]
        first_trial = int(min(trials_a.min(), trials_b.min()))
        last_trial = int(max(trials_a.max(), trials_b.max()))
        start_pair = pair_id
        for trial_a, trial_b in zip(trials_a, trials_b):
            pair_rows.append(
                {
                    "pair_id": pair_id,
                    "trial_a_id": int(trial_a),
                    "trial_b_id": int(trial_b),
                    "first_trial_id": int(min(trial_a, trial_b)),
                    "last_trial_id": int(max(trial_a, trial_b)),
                    "trial_midpoint": float((trial_a + trial_b) / 2.0),
                    "role_a": role_a,
                    "role_b": role_b,
                }
            )
            pair_id += 1
        window_rows.append(
            {
                "window_id": len(window_rows),
                "start_pair_id": start_pair,
                "stop_pair_id": pair_id - 1,
                "pair_count": size,
                "first_trial_id": first_trial,
                "last_trial_id": last_trial,
                "trial_midpoint": float(np.mean(np.concatenate([trials_a, trials_b]))),
                "progress": float(
                    ((first_trial + last_trial) / 2.0) / max(max_trial_id, 1)
                ),
            }
        )
        boundary = last_trial

    pairs = pd.DataFrame.from_records(pair_rows, columns=PAIR_COLUMNS)
    windows = pd.DataFrame.from_records(window_rows, columns=WINDOW_COLUMNS)
    if len(windows) > 1:
        current_end = windows["last_trial_id"].to_numpy()[:-1]
        next_start = windows["first_trial_id"].to_numpy()[1:]
        if not np.all(current_end < next_start):
            raise RuntimeError("constructed windows are not strictly chronological")
    return pairs, windows


def _window_running_speed(joiner, frames, pairs, windows) -> np.ndarray:
    selected_frames = joiner.frames.loc[
        joiner.frames["frame_id"].isin(frames["frame_id"]),
        ["trial_id", "run_speed"],
    ]
    trial_speed = selected_frames.groupby("trial_id")["run_speed"].mean()
    values = []
    for window in windows.itertuples(index=False):
        block = pairs.loc[
            pairs["pair_id"].between(window.start_pair_id, window.stop_pair_id)
        ]
        trial_ids = np.concatenate(
            [block["trial_a_id"].to_numpy(), block["trial_b_id"].to_numpy()]
        )
        values.append(float(trial_speed.reindex(trial_ids).mean()))
    return np.asarray(values)


def chronological_session_history(
    db,
    session,
    *,
    window_pairs: int,
    areas: tuple[str, ...] = ("V1", "mHV", "lHV", "aHV"),
    neurons_per_area: int = 1000,
    threshold: float = 0.3,
    min_frames: int = 3,
    neuron_seed: int = 2025,
    verify: bool = False,
) -> pd.DataFrame:
    """Compute one strictly chronological d-prime history."""

    from dprime import (
        balanced_area_neurons,
        eligible_trial_frames,
    )
    from joiner import Joiner, session_metadata

    joiner = Joiner(
        db,
        session.recording_id,
        experiment=session.experiment,
        behavior_key=session.behavior_key,
    )
    frames = eligible_trial_frames(joiner.frames, roles=(2, 0), min_frames=min_frames)
    eligible_trials = frames[["trial_id", "stimulus_role"]].drop_duplicates()
    eligible_counts = eligible_trials.groupby("stimulus_role")["trial_id"].size()
    pairs, windows = chronological_trial_pairs_and_windows(
        frames, size=int(window_pairs), role_a=2, role_b=0
    )
    if windows.empty:
        raise ValueError(f"{session.behavior_session_id} has too few chronological trials")
    windows["mean_run_speed"] = _window_running_speed(joiner, frames, pairs, windows)
    seed = recording_neuron_seed(session.recording_id, neuron_seed)
    neurons = balanced_area_neurons(
        joiner.neurons,
        areas=areas,
        per_area=neurons_per_area,
        seed=seed,
    )
    window_summaries: list[pd.DataFrame] = []
    for window in windows.itertuples(index=False):
        block = pairs.loc[
            pairs["pair_id"].between(window.start_pair_id, window.stop_pair_id)
        ]
        trial_ids_by_role = {
            2: block["trial_a_id"].to_numpy(dtype=np.int64),
            0: block["trial_b_id"].to_numpy(dtype=np.int64),
        }
        dprime = dprime_for_balanced_trial_ids(
            joiner.U,
            joiner.V,
            frames,
            neuron_ids=neurons["neuron_id"].to_numpy(dtype=np.int64),
            trial_ids_by_role=trial_ids_by_role,
            roles=(2, 0),
        )
        summary = summarize_trial_balanced_dprime(
            dprime, neurons, threshold=threshold
        )
        for name in WINDOW_COLUMNS:
            summary[name] = getattr(window, name)
        summary["mean_run_speed"] = window.mean_run_speed
        window_summaries.append(summary)
    if verify:
        if len(windows) > 1:
            assert bool(
                (windows["last_trial_id"].to_numpy()[:-1]
                 < windows["first_trial_id"].to_numpy()[1:]).all()
            )
        if any(
            not summary["n_neurons"].eq(neurons_per_area).all()
            for summary in window_summaries
        ):
            raise RuntimeError("window summaries are not aligned to sampled neurons")
    result = pd.concat(window_summaries, ignore_index=True)
    available = joiner.neurons.groupby("area_group")["neuron_id"].size()
    result["available_neurons"] = result["area"].map(available).astype(int)
    result = result.assign(
        **session_metadata(session),
        total_pairs=len(pairs),
        eligible_trials_role_positive=int(eligible_counts.get(2, 0)),
        eligible_trials_role_negative=int(eligible_counts.get(0, 0)),
        selected_trials_per_role=len(pairs),
        discarded_eligible_trials_role_positive=int(
            eligible_counts.get(2, 0) - len(pairs)
        ),
        discarded_eligible_trials_role_negative=int(
            eligible_counts.get(0, 0) - len(pairs)
        ),
        trial_count=int(getattr(session, "trial_count", len(joiner.trials))),
        recorded_neurons=joiner.U.shape[1],
        chronological=True,
        estimator=WINDOW_ESTIMATOR,
        estimator_version=WINDOW_ESTIMATOR_VERSION,
        window_construction=WINDOW_CONSTRUCTION,
        role_positive=2,
        role_negative=0,
        minimum_eligible_frames_per_trial=int(min_frames),
        selectivity_threshold=float(threshold),
        requested_neurons_per_area=int(neurons_per_area),
        neuron_sampling_seed=int(seed),
        neuron_sampling_base_seed=int(neuron_seed),
        neuron_sampling_seed_method=NEURON_SEED_METHOD,
        trial_response_estimator=TRIAL_RESPONSE_ESTIMATOR,
        behavior_estimator=BEHAVIOR_ESTIMATOR,
        denominator_estimator=DENOMINATOR_ESTIMATOR,
        denominator_handling="invalid_left_missing_and_counted",
    )
    del joiner
    gc.collect()
    return result
