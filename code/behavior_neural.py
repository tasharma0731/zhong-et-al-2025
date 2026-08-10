import gc

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from dprime import svd_dprime
from joiner import Joiner, session_metadata
from position import component_trial_position, position_bin_indices


def safe_spearman(x, y, minimum=5):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    if keep.sum() < minimum or np.ptp(x[keep]) == 0 or np.ptp(y[keep]) == 0:
        return np.nan, int(keep.sum())
    return float(spearmanr(x[keep], y[keep]).statistic), int(keep.sum())


def finite_median(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return np.nan if len(values) == 0 else float(np.median(values))


def trial_behavior(db, behavior_session_id):
    result = db.query(
        """
        WITH trial_events AS (
            SELECT t.trial_id, t.stimulus_role, t.wall_name, t.is_rewarded,
                   t.reward_position / 10.0 AS reward_position_m,
                   t.sound_position / 10.0 AS sound_position_m,
                   (t.sound_delay_time - t.sound_time) * 86400.0 AS sound_delay_s,
                   (t.end_time - t.start_time) * 86400.0 AS duration_s,
                   COUNT(e.event_id) FILTER (
                       WHERE e.event_type = 'lick' AND e.occurred
                   ) AS lick_count,
                   COUNT(e.event_id) FILTER (
                       WHERE e.event_type = 'lick' AND e.occurred
                         AND (e.time - t.sound_time) * 86400.0 >= -2.0
                         AND (e.time - t.sound_time) * 86400.0 < 0.0
                   ) AS anticipatory_licks_2s,
                   MIN((e.time - t.sound_time) * 86400.0) FILTER (
                       WHERE e.event_type = 'lick' AND e.occurred
                   ) AS first_lick_latency_s
            FROM behavior_trials AS t
            LEFT JOIN behavior_events AS e
              ON e.behavior_session_id = t.behavior_session_id
             AND e.trial_id = t.trial_id
            WHERE t.behavior_session_id = ?
            GROUP BY ALL
        )
        SELECT *,
               lick_count / NULLIF(duration_s, 0) AS lick_rate_hz,
               CASE WHEN LAG(is_rewarded) OVER (ORDER BY trial_id)
                    THEN LAG(reward_position_m) OVER (ORDER BY trial_id)
               END AS previous_reward_position_m
        FROM trial_events
        ORDER BY trial_id
        """,
        [behavior_session_id],
    )
    return result


def frame_trial_metrics(joiner, position_edges):
    centers = (position_edges[:-1] + position_edges[1:]) / 2
    ids = joiner.neurons.loc[
        joiner.neurons["area_group"] == "mHV", "neuron_id"
    ].to_numpy(dtype=np.int64)
    activity = joiner.U[:, ids].mean(axis=1) @ joiner.V
    frames = joiner.query("""
        SELECT frame_id, trial_id, position_m, run_speed, is_moving
        FROM frames
        WHERE valid_trial AND in_texture AND position_m >= 0 AND position_m < 4
          AND trial_id IS NOT NULL AND position_m IS NOT NULL AND run_speed IS NOT NULL
        ORDER BY frame_id
    """)
    frame_ids = frames["frame_id"].to_numpy(dtype=np.int64)
    values = activity[frame_ids]
    bins = position_bin_indices(frames["position_m"], position_edges)
    joiner.register(
        "frame_values",
        frames.assign(mhv_z=(values - values.mean()) / (values.std() + 1e-12), position_bin=bins),
    )
    centers_sql = "CASE position_bin " + " ".join(
        f"WHEN {index} THEN {float(center)}" for index, center in enumerate(centers)
    ) + " END"
    return joiner.query(f"""
        WITH trial_metrics AS (
            SELECT trial_id, AVG(run_speed) AS mean_run_speed,
                   AVG(mhv_z) AS mean_mhv_z,
                   AVG(is_moving::INTEGER) AS moving_fraction,
                   COUNT(*) AS texture_frames
            FROM frame_values
            GROUP BY trial_id
        ),
        profiles AS (
            SELECT trial_id, position_bin, AVG(mhv_z) AS mhv_z
            FROM frame_values
            GROUP BY trial_id, position_bin
        ),
        peaks AS (
            SELECT trial_id, {centers_sql} AS mhv_peak_position_m
            FROM profiles
            QUALIFY ROW_NUMBER() OVER (
                PARTITION BY trial_id ORDER BY mhv_z DESC, position_bin
            ) = 1
        )
        SELECT t.*, p.mhv_peak_position_m
        FROM trial_metrics AS t
        LEFT JOIN peaks AS p USING (trial_id)
        ORDER BY trial_id
    """)


def position_discriminability(joiner, position_edges, threshold=0.3, minimum_trials=5):
    centers = (position_edges[:-1] + position_edges[1:]) / 2
    _, response, _, roles = component_trial_position(
        joiner, position_edges, stimulus_only=True
    )
    ids = joiner.neurons.loc[
        joiner.neurons["area_group"] == "mHV", "neuron_id"
    ].to_numpy(dtype=np.int64)
    u = joiner.U[:, ids]
    rows = []
    for position_bin, position_m in enumerate(centers):
        current = response[:, :, position_bin]
        finite = np.isfinite(current).all(axis=0)
        selected = current[:, finite]
        selected_roles = roles[finite]
        leaf = selected_roles == 2
        circle = selected_roles == 0
        if min(leaf.sum(), circle.sum()) < minimum_trials:
            continue
        dprime = svd_dprime(u, selected, leaf, circle)
        dprime = dprime[np.isfinite(dprime)]
        if len(dprime) == 0:
            continue
        rows.append(
            [
                position_bin,
                position_m,
                int(leaf.sum()),
                int(circle.sum()),
                np.median(dprime),
                np.mean(np.abs(dprime)),
                np.mean(np.abs(dprime) >= threshold),
                np.mean(dprime >= threshold),
                np.mean(dprime <= -threshold),
            ]
        )
    return pd.DataFrame(
        rows,
        columns=[
            "position_bin",
            "position_m",
            "n_leaf_trials",
            "n_circle_trials",
            "median_dprime",
            "mean_abs_dprime",
            "frac_selective",
            "frac_leaf_selective",
            "frac_circle_selective",
        ],
    )


def first_sustained_position(position, threshold=0.3):
    ordered = position.sort_values("position_bin")
    selective = ordered["frac_selective"].to_numpy() >= threshold
    crossing = np.flatnonzero(selective[:-1] & selective[1:])
    return np.nan if len(crossing) == 0 else float(ordered.iloc[crossing[0]]["position_m"])


def behavior_neural_coupling_session(db, session, position_edges):
    joiner = Joiner(
        db,
        session.recording_id,
        experiment=session.experiment,
        behavior_key=session.behavior_key,
    )
    trials = trial_behavior(db, session.behavior_session_id).merge(
        frame_trial_metrics(joiner, position_edges), on="trial_id", how="left"
    )
    position = position_discriminability(joiner, position_edges)
    correlations = {
        "run_mhv": safe_spearman(trials["mean_run_speed"], trials["mean_mhv_z"]),
        "lick_mhv": safe_spearman(trials["lick_rate_hz"], trials["mean_mhv_z"]),
        "previous_reward_peak": safe_spearman(
            trials["previous_reward_position_m"], trials["mhv_peak_position_m"]
        ),
        "previous_reward_mean_mhv": safe_spearman(
            trials["previous_reward_position_m"], trials["mean_mhv_z"]
        ),
    }
    info = session_metadata(session)
    trials = trials.assign(**info)
    position = position.assign(**info)
    summary = {
        **info,
        "trials": len(trials),
        **{f"rho_{name}": value[0] for name, value in correlations.items()},
        **{f"n_{name}": value[1] for name, value in correlations.items()},
        "anticipatory_trial_fraction": float((trials["anticipatory_licks_2s"] > 0).mean()),
        "median_first_lick_latency_s": finite_median(trials["first_lick_latency_s"]),
        "first_sustained_selective_position_m": first_sustained_position(position),
        "peak_fraction_selective": float(position["frac_selective"].max()),
    }
    del joiner
    gc.collect()
    return trials, position, summary

