from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray


def balanced_area_neurons(
    neurons: pd.DataFrame,
    *,
    areas: Sequence[str] = ("V1", "mHV", "lHV", "aHV"),
    per_area: int | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    selected = neurons.loc[neurons["area_group"].isin(areas)].copy()
    counts = selected.groupby("area_group")["neuron_id"].size().reindex(list(areas))
    if counts.isna().any() or (counts == 0).any():
        missing = counts[counts.isna() | counts.eq(0)].index.tolist()
        raise ValueError(f"no neurons found for {missing}")
    sample_size = int(counts.min()) if per_area is None else int(per_area)
    if sample_size < 1 or (counts < sample_size).any():
        raise ValueError(f"per_area must be between 1 and {int(counts.min())}")
    return (
        selected.groupby("area_group", group_keys=False, sort=False)
        .sample(n=sample_size, random_state=seed)
        .sort_values(["area_group", "neuron_id"])
        .reset_index(drop=True)
    )


def windowed_svd_dprime(
    u_components_by_neuron: ArrayLike,
    v_components_by_frame: ArrayLike,
    trial_frames: pd.DataFrame,
    pairs: pd.DataFrame,
    windows: pd.DataFrame,
    *,
    neuron_ids: ArrayLike,
    epsilon: float = 1e-8,
) -> NDArray[np.float64]:
    u = np.asarray(u_components_by_neuron)
    v = np.asarray(v_components_by_frame)
    neuron_ids = np.asarray(neuron_ids, dtype=np.int64)
    if u.ndim != 2 or v.ndim != 2 or u.shape[0] != v.shape[0]:
        raise ValueError("U and V must have aligned component axes")
    if not np.isfinite(epsilon) or epsilon < 0:
        raise ValueError("epsilon must be finite and non-negative")
    if len(neuron_ids) == 0:
        raise ValueError("neuron_ids must not be empty")
    if len(windows) == 0:
        return np.empty((0, len(neuron_ids)), dtype=np.float64)

    frames = trial_frames.sort_values("frame_id").reset_index(drop=True)
    frame_ids = frames["frame_id"].to_numpy(dtype=np.int64)
    if len(frame_ids) != len(np.unique(frame_ids)):
        raise ValueError("trial_frames must contain each frame once")
    activity = u[:, neuron_ids].T @ v[:, frame_ids]
    moments = _trial_moments(activity, frames)

    pair_a = _paired_moments(pairs["trial_a_id"], moments)
    pair_b = _paired_moments(pairs["trial_b_id"], moments)
    starts = windows["start_pair_id"].to_numpy(dtype=np.int64)
    stops = windows["stop_pair_id"].to_numpy(dtype=np.int64) + 1
    sum_a, square_a, count_a = _window_moments(pair_a, starts, stops)
    sum_b, square_b, count_b = _window_moments(pair_b, starts, stops)

    mean_a = sum_a / count_a[:, None]
    mean_b = sum_b / count_b[:, None]
    sd_a = np.sqrt(np.maximum(square_a / count_a[:, None] - mean_a**2, 0.0))
    sd_b = np.sqrt(np.maximum(square_b / count_b[:, None] - mean_b**2, 0.0))
    with np.errstate(all="ignore"):
        dprime = 2.0 * (mean_a - mean_b) / (sd_a + sd_b + epsilon)
    return np.nan_to_num(dprime, nan=0.0, posinf=0.0, neginf=0.0)


def summarize_dprime_history(
    dprime: ArrayLike,
    windows: pd.DataFrame,
    neurons: pd.DataFrame,
    *,
    threshold: float = 0.3,
    include_all_areas: bool = False,
) -> pd.DataFrame:
    values = np.asarray(dprime, dtype=np.float64)
    if values.shape != (len(windows), len(neurons)):
        raise ValueError("dprime must align as windows x neurons")
    rows = []
    areas = neurons["area_group"].to_numpy()
    area_masks = [(area, areas == area) for area in pd.unique(areas)]
    if include_all_areas:
        area_masks.insert(0, ("all", np.ones(len(areas), dtype=bool)))
    for window_index, window in enumerate(windows.to_dict("records")):
        for area, mask in area_masks:
            current = values[window_index, mask]
            finite = current[np.isfinite(current)]
            if len(finite) == 0:
                continue
            mean = float(np.mean(finite))
            standard_deviation = float(np.std(finite))
            normalized = (
                np.zeros_like(finite)
                if standard_deviation == 0
                else (finite - mean) / standard_deviation
            )
            rows.append(
                {
                    **window,
                    "area": area,
                    "n_neurons": len(current),
                    "n_finite": len(finite),
                    "mean_dprime": mean,
                    "mean_abs_dprime": float(np.mean(np.abs(finite))),
                    "sd_dprime": standard_deviation,
                    "skewness": float(np.mean(normalized**3)),
                    "excess_kurtosis": float(np.mean(normalized**4) - 3.0),
                    "frac_selective": float(np.mean(np.abs(finite) >= threshold)),
                    "frac_leaf_selective": float(np.mean(finite >= threshold)),
                    "frac_circle_selective": float(np.mean(finite <= -threshold)),
                    "q05": float(np.quantile(finite, 0.05)),
                    "median": float(np.median(finite)),
                    "q95": float(np.quantile(finite, 0.95)),
                }
            )
    return pd.DataFrame.from_records(rows)


def _trial_moments(
    activity: NDArray,
    frames: pd.DataFrame,
) -> dict[int, tuple[NDArray[np.float64], NDArray[np.float64], int]]:
    moments: dict[int, tuple[NDArray[np.float64], NDArray[np.float64], int]] = {}
    for trial_id, group in frames.groupby("trial_id", sort=False):
        indices = group.index.to_numpy(dtype=np.int64)
        block = activity[:, indices]
        trial_key = int(np.asarray(trial_id, dtype=np.int64).item())
        total = np.asarray(np.sum(block, axis=1, dtype=np.float64), dtype=np.float64)
        squared = np.asarray(
            np.sum(np.square(block, dtype=np.float64), axis=1),
            dtype=np.float64,
        )
        moments[trial_key] = total, squared, block.shape[1]
    return moments


def _paired_moments(
    trial_ids: pd.Series,
    moments: dict[int, tuple[NDArray[np.float64], NDArray[np.float64], int]],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]]:
    selected = [moments[int(trial_id)] for trial_id in trial_ids]
    return (
        np.stack([item[0] for item in selected]),
        np.stack([item[1] for item in selected]),
        np.asarray([item[2] for item in selected], dtype=np.int64),
    )


def _window_moments(
    moments: tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]],
    starts: NDArray[np.int64],
    stops: NDArray[np.int64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]]:
    values = []
    for item in moments:
        zero = np.zeros((1, *item.shape[1:]), dtype=item.dtype)
        cumulative = np.concatenate([zero, np.cumsum(item, axis=0)], axis=0)
        values.append(cumulative[stops] - cumulative[starts])
    return values[0], values[1], values[2]
