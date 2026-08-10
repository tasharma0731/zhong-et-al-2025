import gc
import zlib

import numpy as np

from joiner import Joiner, session_metadata

from .history import (
    balanced_area_neurons,
    summarize_dprime_history,
    windowed_svd_dprime,
)
from .pairing import eligible_trial_frames, ordinal_trial_pairs
from .trials import AREAS
from .windows import equal_pair_windows, paired_trial_windows


def _direct_window_dprime(joiner, neuron_ids, frames, pairs, window):
    block = pairs.iloc[int(window.start_pair_id) : int(window.stop_pair_id) + 1]
    trial_groups = [block["trial_a_id"], block["trial_b_id"]]
    activity = []
    for trials in trial_groups:
        frame_ids = frames.loc[
            frames["trial_id"].isin(trials), "frame_id"
        ].to_numpy(dtype=np.int64)
        activity.append(joiner.U[:, neuron_ids].T @ joiner.V[:, frame_ids])
    leaf, circle = activity
    return 2.0 * (leaf.mean(1) - circle.mean(1)) / (
        leaf.std(1) + circle.std(1) + 1e-8
    )


def _window_running_speed(joiner, frames, pairs, windows):
    joiner.register("analysis_trial_frames", frames)
    joiner.register("analysis_pairs", pairs)
    joiner.register("analysis_windows", windows)
    result = joiner.query("""
        WITH pair_trials AS (
            SELECT pair_id, trial_a_id AS trial_id FROM analysis_pairs
            UNION ALL
            SELECT pair_id, trial_b_id AS trial_id FROM analysis_pairs
        ),
        trial_speed AS (
            SELECT tf.trial_id, AVG(f.run_speed) AS mean_run_speed
            FROM analysis_trial_frames AS tf
            JOIN frames AS f USING (frame_id, trial_id)
            GROUP BY tf.trial_id
        )
        SELECT w.window_id, AVG(s.mean_run_speed) AS mean_run_speed
        FROM analysis_windows AS w
        JOIN pair_trials AS p
          ON p.pair_id BETWEEN w.start_pair_id AND w.stop_pair_id
        JOIN trial_speed AS s USING (trial_id)
        GROUP BY w.window_id
        ORDER BY w.window_id
    """)
    return result["mean_run_speed"].to_numpy()


def session_dprime_history(
    db,
    session,
    *,
    areas=AREAS,
    window_pairs=None,
    trial_bins=None,
    neurons_per_area=1000,
    threshold=0.3,
    include_all_areas=False,
    verify=False,
):
    if (window_pairs is None) == (trial_bins is None):
        raise ValueError("provide exactly one of window_pairs or trial_bins")
    joiner = Joiner(
        db,
        session.recording_id,
        experiment=session.experiment,
        behavior_key=session.behavior_key,
    )
    frames = eligible_trial_frames(joiner.frames, roles=(2, 0), min_frames=3)
    pairs = ordinal_trial_pairs(frames, role_a=2, role_b=0)
    windows = (
        paired_trial_windows(pairs, size=int(window_pairs), stride=1)
        if window_pairs is not None
        else equal_pair_windows(pairs, bins=int(trial_bins))
    )
    if windows.empty:
        raise ValueError(f"{session.behavior_session_id} has too few paired trials")
    windows["mean_run_speed"] = _window_running_speed(joiner, frames, pairs, windows)
    neurons = balanced_area_neurons(
        joiner.neurons,
        areas=areas,
        per_area=neurons_per_area,
        seed=zlib.crc32(session.recording_id.encode()),
    )
    dprime = windowed_svd_dprime(
        joiner.U,
        joiner.V,
        frames,
        pairs,
        windows,
        neuron_ids=neurons["neuron_id"],
    )
    if verify:
        ids = neurons["neuron_id"].to_numpy(dtype=np.int64)[:32]
        expected = _direct_window_dprime(joiner, ids, frames, pairs, windows.iloc[0])
        np.testing.assert_allclose(dprime[0, :32], expected, rtol=1e-6, atol=2e-7)
    result = summarize_dprime_history(
        dprime,
        windows,
        neurons,
        threshold=threshold,
        include_all_areas=include_all_areas,
    )
    available = joiner.neurons.groupby("area_group")["neuron_id"].size()
    if include_all_areas:
        available.loc["all"] = int(available.reindex(list(areas), fill_value=0).sum())
    result["available_neurons"] = result["area"].map(available).astype(int)
    result = result.assign(
        **session_metadata(session),
        total_pairs=len(pairs),
        trial_count=int(getattr(session, "trial_count", len(joiner.trials))),
        recorded_neurons=joiner.U.shape[1],
    )
    del joiner, dprime
    gc.collect()
    return result
