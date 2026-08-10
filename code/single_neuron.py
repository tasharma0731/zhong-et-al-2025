import gc

import numpy as np
import pandas as pd

from dprime import AREAS
from joiner import Joiner, session_metadata
from position import component_trial_position


def projected_variance(u, covariance, chunk_size=8192):
    result = np.empty(u.shape[1], dtype=np.float64)
    for start in range(0, len(result), chunk_size):
        stop = min(start + chunk_size, len(result))
        block = u[:, start:stop].astype(np.float64, copy=False)
        result[start:stop] = np.einsum(
            "ki,kl,li->i", block, covariance, block, optimize=True
        )
    return np.maximum(result, 0.0)


def row_correlation(left, right):
    if left.shape != right.shape or left.shape[1] < 3:
        return np.full(left.shape[0], np.nan)
    left = left - left.mean(axis=1, keepdims=True)
    right = right - right.mean(axis=1, keepdims=True)
    denominator = np.sqrt(np.sum(left**2, axis=1) * np.sum(right**2, axis=1))
    return np.divide(
        np.sum(left * right, axis=1),
        denominator,
        out=np.full(left.shape[0], np.nan),
        where=denominator > 0,
    )


def nearest_frame_ids(frame_times, event_times):
    frame_times = np.asarray(frame_times, dtype=float)
    event_times = np.asarray(event_times, dtype=float)
    finite = np.flatnonzero(np.isfinite(frame_times))
    ordered = frame_times[finite]
    if len(ordered) < 2 or len(event_times) == 0:
        return np.array([], dtype=np.int64)
    insertion = np.searchsorted(ordered, event_times)
    upper = np.clip(insertion, 0, len(ordered) - 1)
    lower = np.clip(insertion - 1, 0, len(ordered) - 1)
    nearest = np.where(
        np.abs(ordered[upper] - event_times) < np.abs(ordered[lower] - event_times),
        upper,
        lower,
    )
    keep = np.abs(ordered[nearest] - event_times) <= 2.0 * np.median(np.diff(ordered))
    return np.unique(finite[nearest[keep]]).astype(np.int64)


def lick_triggered_components(db, joiner, lags):
    events = db.query(
        """
        SELECT time FROM behavior_events
        WHERE behavior_session_id = ? AND event_type = 'lick'
          AND occurred AND isfinite(time)
        ORDER BY time
        """,
        [joiner.behavior_session_id],
    )
    if "time" not in joiner.frames or events.empty:
        return None, np.array([], dtype=np.int64), np.nan
    times = joiner.frames["time"].to_numpy(dtype=float)
    frames = nearest_frame_ids(times, events["time"])
    frames = frames[(frames + lags[0] >= 0) & (frames + lags[-1] < len(times))]
    trial_ids = joiner.frames["trial_id"].to_numpy()
    frames = np.asarray(
        [
            frame
            for frame in frames
            if pd.notna(trial_ids[frame])
            and np.all(trial_ids[frame + lags] == trial_ids[frame])
        ],
        dtype=np.int64,
    )
    if len(frames) == 0:
        return None, frames, np.nan
    component_average = joiner.V[:, frames[:, None] + lags].mean(axis=1)
    return component_average, frames, float(np.nanmedian(np.diff(times)) * 86400.0)


def validate_tuning(joiner, frames, trials, response, neuron_ids, tuning, valid_bins):
    ids = neuron_ids[:8]
    direct = joiner.U[:, ids].T @ joiner.V[:, frames["frame_id"].to_numpy(dtype=np.int64)]
    index = {trial: position for position, trial in enumerate(trials)}
    expected = np.full((len(ids), len(trials), valid_bins.sum()), np.nan)
    bin_index = {position: index for index, position in enumerate(np.flatnonzero(valid_bins))}
    validation_frames = frames.reset_index(drop=True).assign(column_id=np.arange(len(frames)))
    joiner.register("validation_frames", validation_frames)
    groups = joiner.query("""
        SELECT trial_id, position_bin, LIST(column_id ORDER BY column_id) AS column_ids
        FROM validation_frames
        GROUP BY trial_id, position_bin
        ORDER BY trial_id, position_bin
    """)
    for group in groups.itertuples(index=False):
        if int(group.position_bin) in bin_index:
            rows = np.asarray(group.column_ids, dtype=np.int64)
            expected[:, index[int(group.trial_id)], bin_index[int(group.position_bin)]] = direct[:, rows].mean(1)
    np.testing.assert_allclose(tuning[:8], np.nanmean(expected, axis=1), rtol=1e-6, atol=1e-6)


def single_neuron_tuning_session(db, session, position_edges, lags, *, verify=False):
    centers = (position_edges[:-1] + position_edges[1:]) / 2
    joiner = Joiner(
        db,
        session.recording_id,
        experiment=session.experiment,
        behavior_key=session.behavior_key,
    )
    frames, response, trials, _ = component_trial_position(joiner, position_edges)
    valid_bins = np.isfinite(response).any(axis=1).all(axis=0)
    even = trials % 2 == 0
    component = np.nanmean(response[:, :, valid_bins], axis=1)
    component_even = np.nanmean(response[:, even][:, :, valid_bins], axis=1)
    component_odd = np.nanmean(response[:, ~even][:, :, valid_bins], axis=1)
    neuron_ids = joiner.neurons.loc[
        joiner.neurons["area_group"].isin(AREAS), "neuron_id"
    ].to_numpy(dtype=np.int64)
    u = joiner.U[:, neuron_ids]
    tuning = u.T @ component
    tuning_even = u.T @ component_even
    tuning_odd = u.T @ component_odd
    if verify:
        validate_tuning(joiner, frames, trials, response, neuron_ids, tuning, valid_bins)

    frame_ids = frames["frame_id"].to_numpy(dtype=np.int64)
    activity = joiner.V[:, frame_ids].astype(np.float64, copy=False)
    component_mean = activity.mean(axis=1)
    centered = activity - component_mean[:, None]
    covariance = centered @ centered.T / max(activity.shape[1] - 1, 1)
    mean_activity = u.T @ component_mean
    activity_sd = np.sqrt(projected_variance(u, covariance))
    speed = joiner.frames.loc[frame_ids, "run_speed"].to_numpy(dtype=float)
    speed_centered = speed - speed.mean()
    speed_sd = speed.std(ddof=1)
    run_covariance = u.T @ (centered @ speed_centered / max(len(speed) - 1, 1))
    run_correlation = np.divide(
        run_covariance,
        activity_sd * speed_sd,
        out=np.full(len(neuron_ids), np.nan),
        where=(activity_sd > 0) & (speed_sd > 0),
    )
    reliability = row_correlation(tuning_even, tuning_odd)
    spatial_strength = np.divide(
        tuning.std(axis=1),
        activity_sd,
        out=np.full(len(neuron_ids), np.nan),
        where=activity_sd > 0,
    )
    preferred_position = centers[valid_bins][np.argmax(tuning, axis=1)]
    component_lick, lick_frames, frame_dt_s = lick_triggered_components(db, joiner, lags)
    lick_average = np.full((len(neuron_ids), len(lags)), np.nan)
    lick_modulation = np.full(len(neuron_ids), np.nan)
    if component_lick is not None:
        lick_average = u.T @ component_lick
        baseline = lick_average[:, lags < -3].mean(axis=1)
        response_average = lick_average[:, (lags >= 0) & (lags <= 3)].mean(axis=1)
        lick_modulation = np.divide(
            response_average - baseline,
            activity_sd,
            out=np.full(len(neuron_ids), np.nan),
            where=activity_sd > 0,
        )

    neurons = joiner.neurons.set_index("neuron_id").loc[neuron_ids].reset_index()
    metrics = neurons.assign(
        **session_metadata(session),
        mean_activity=mean_activity,
        activity_sd=activity_sd,
        run_correlation=run_correlation,
        spatial_strength=spatial_strength,
        spatial_reliability=reliability,
        preferred_position_m=preferred_position,
        lick_modulation_z=lick_modulation,
        lick_frames=len(lick_frames),
    )
    tuning_rows = []
    lick_rows = []
    info = session_metadata(session)
    for area in AREAS:
        indices = np.flatnonzero(neurons["area_group"].to_numpy() == area)
        score = np.nan_to_num(reliability[indices], nan=-np.inf)
        if np.isneginf(score).all():
            score = np.nan_to_num(spatial_strength[indices], nan=-np.inf)
        exemplar = indices[int(np.argmax(score))]
        key = {**info, "area_group": area, "neuron_id": int(neuron_ids[exemplar])}
        tuning_rows.extend(
            {**key, "position_m": position, "activity": value}
            for position, value in zip(centers[valid_bins], tuning[exemplar])
        )
        if component_lick is not None:
            lick_rows.extend(
                {**key, "lag_s": lag, "activity": value}
                for lag, value in zip(lags * frame_dt_s, lick_average[exemplar])
            )
    del joiner, response, activity, centered
    gc.collect()
    return metrics, pd.DataFrame(tuning_rows), pd.DataFrame(lick_rows)

