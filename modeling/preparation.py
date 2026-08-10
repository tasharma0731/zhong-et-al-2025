"""Validated transformations from d-prime histories to prediction tables."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .config import DEFAULT_METRICS


def required_history_columns(metrics: Sequence[str]) -> set[str]:
    return {
        "behavior_session_id",
        "mouse",
        "cohort",
        "moment",
        "area",
        "start_pair_id",
        "stop_pair_id",
        "pair_count",
        "first_trial_id",
        "last_trial_id",
        "progress",
        "mean_run_speed",
        "chronological",
        *metrics,
    }


def load_history(
    path: str | Path,
    *,
    metrics: Sequence[str] = DEFAULT_METRICS,
) -> tuple[pd.DataFrame, int]:
    path = Path(path)
    history = pd.read_csv(path)
    missing = required_history_columns(metrics) - set(history.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    sizes = history["pair_count"].dropna().astype(int).unique()
    if len(sizes) != 1:
        raise ValueError(f"expected one window size, found {sorted(sizes)}")
    window_pairs = int(sizes[0])
    if window_pairs < 2:
        raise ValueError("window size must contain at least two trial pairs")
    chronological = history["chronological"]
    if chronological.dtype == object:
        chronological = chronological.astype(str).str.lower().map(
            {"true": True, "false": False}
        )
    if chronological.isna().any() or not bool(chronological.all()):
        raise ValueError("forecast histories must be explicitly chronological")
    if (history["first_trial_id"] > history["last_trial_id"]).any():
        raise ValueError("history contains an invalid trial boundary")
    return history, window_pairs


def load_session_summaries(
    path: str | Path,
    *,
    metrics: Sequence[str] = DEFAULT_METRICS,
) -> pd.DataFrame:
    """Load one trial-balanced population summary per session and area."""

    path = Path(path)
    frame = pd.read_csv(path)
    required = {
        "behavior_session_id",
        "mouse",
        "cohort",
        "moment",
        "area",
        "mean_run_speed",
        *metrics,
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    duplicated = frame.duplicated(["behavior_session_id", "area"])
    if duplicated.any():
        raise ValueError("session summary contains duplicate session/area rows")
    if "estimator" in frame and not frame["estimator"].eq(
        "full_session_trial_balanced_dprime"
    ).all():
        values = sorted(frame["estimator"].dropna().astype(str).unique())
        raise ValueError(
            "expected full_session_trial_balanced_dprime estimator, "
            f"found {values}"
        )
    if not np.isfinite(frame[list(metrics)].to_numpy(dtype=float)).all():
        raise ValueError("session summaries contain non-finite target metrics")
    return frame.sort_values(["cohort", "mouse", "moment", "area"]).reset_index(
        drop=True
    )


def nonoverlapping_windows(history: pd.DataFrame, window_pairs: int) -> pd.DataFrame:
    """Select the predeclared disjoint subset from a stride-one history."""

    if "chronological" in history and history["chronological"].fillna(False).all():
        selected = history.copy()
    else:
        selected = history.loc[
            history["start_pair_id"].astype(int).mod(window_pairs).eq(0)
        ].copy()
    selected = selected.sort_values(
        ["behavior_session_id", "area", "start_pair_id"]
    ).reset_index(drop=True)
    duplicated = selected.duplicated(
        ["behavior_session_id", "area", "start_pair_id"]
    )
    if duplicated.any():
        raise ValueError("history contains duplicate session/area/window rows")
    return selected


def make_session_pairs(
    windows: pd.DataFrame,
    *,
    metrics: Sequence[str] = DEFAULT_METRICS,
) -> pd.DataFrame:
    """Return one complete pre/post row for every available mouse and area."""

    aggregations = {metric: (metric, "mean") for metric in metrics}
    session = windows.groupby(
        ["behavior_session_id", "mouse", "cohort", "moment", "area"],
        as_index=False,
    ).agg(**aggregations, mean_run_speed=("mean_run_speed", "mean"))

    counts = session.groupby(["mouse", "cohort", "area", "moment"]).size()
    if (counts > 1).any():
        bad = counts[counts > 1].head().to_dict()
        raise ValueError(f"multiple sessions for the same mouse/moment: {bad}")

    measured = ["behavior_session_id", *metrics, "mean_run_speed"]
    moments = {}
    for moment in ("before", "after"):
        frame = session.loc[session["moment"].eq(moment)].drop(columns="moment")
        moments[moment] = frame.rename(
            columns={column: f"{moment}_{column}" for column in measured}
        )
    pairs = moments["before"].merge(
        moments["after"],
        on=["mouse", "cohort", "area"],
        validate="one_to_one",
    )
    expected = session.groupby(["mouse", "cohort", "area"])["moment"].nunique()
    if len(pairs) != int(expected.eq(2).sum()):
        raise ValueError("paired table did not match complete mouse-area groups")
    if pairs.empty:
        raise ValueError("no complete before/after mouse-area pairs were available")
    return pairs.sort_values(["cohort", "mouse", "area"]).reset_index(drop=True)


def make_repeated_session_pairs(
    summaries: pd.DataFrame,
    *,
    metrics: Sequence[str] = DEFAULT_METRICS,
) -> pd.DataFrame:
    """Average repeated sessions within moment for a labeled control cohort.

    This is intentionally separate from the primary one-session-per-moment
    estimand. It is useful for the grating mice, two of which have replicate
    recordings in each moment.
    """

    aggregations = {metric: (metric, "mean") for metric in metrics}
    grouped = summaries.groupby(
        ["mouse", "cohort", "moment", "area"], as_index=False
    ).agg(
        **aggregations,
        mean_run_speed=("mean_run_speed", "mean"),
        replicate_sessions=("behavior_session_id", "nunique"),
    )
    moments: dict[str, pd.DataFrame] = {}
    measured = [*metrics, "mean_run_speed", "replicate_sessions"]
    for moment in ("before", "after"):
        current = grouped.loc[grouped["moment"].eq(moment)].drop(columns="moment")
        moments[moment] = current.rename(
            columns={column: f"{moment}_{column}" for column in measured}
        )
    pairs = moments["before"].merge(
        moments["after"], on=["mouse", "cohort", "area"], validate="one_to_one"
    )
    if pairs.empty:
        raise ValueError("no complete repeated-session control pairs")
    return pairs.sort_values(["cohort", "mouse", "area"]).reset_index(drop=True)


def make_window_transitions(
    windows: pd.DataFrame,
    window_pairs: int,
    *,
    metrics: Sequence[str] = DEFAULT_METRICS,
) -> pd.DataFrame:
    """Pair adjacent, non-overlapping windows within a recording and area."""

    records: list[dict[str, object]] = []
    for (_, area), group in windows.groupby(["behavior_session_id", "area"], sort=False):
        ordered = group.sort_values("start_pair_id").reset_index(drop=True)
        for index in range(len(ordered) - 1):
            current, future = ordered.iloc[index], ordered.iloc[index + 1]
            if int(future["start_pair_id"]) != int(current["start_pair_id"]) + window_pairs:
                continue
            if int(current["last_trial_id"]) >= int(future["first_trial_id"]):
                raise ValueError(
                    "next-window target is not strictly after the current window"
                )
            record: dict[str, object] = {
                "behavior_session_id": current["behavior_session_id"],
                "mouse": current["mouse"],
                "cohort": current["cohort"],
                "moment": current["moment"],
                "area": area,
                "current_start_pair_id": int(current["start_pair_id"]),
                "next_start_pair_id": int(future["start_pair_id"]),
                "current_last_trial_id": int(current["last_trial_id"]),
                "next_first_trial_id": int(future["first_trial_id"]),
                "current_progress": float(current["progress"]),
                "current_run_speed": float(current["mean_run_speed"]),
            }
            for metric in metrics:
                record[f"current_{metric}"] = float(current[metric])
                record[f"next_{metric}"] = float(future[metric])
            records.append(record)
    result = pd.DataFrame.from_records(records)
    if result.empty:
        raise ValueError("no adjacent non-overlapping windows were available")
    return result.sort_values(
        ["cohort", "mouse", "behavior_session_id", "area", "current_start_pair_id"]
    ).reset_index(drop=True)


def metric_matrix(
    frame: pd.DataFrame,
    prefix: str,
    metrics: Sequence[str] = DEFAULT_METRICS,
) -> np.ndarray:
    return frame[[f"{prefix}_{metric}" for metric in metrics]].to_numpy(dtype=float)


def safe_scale(values: np.ndarray) -> np.ndarray:
    scale = np.std(values, axis=0, ddof=1)
    return np.where(np.isfinite(scale) & (scale > 1e-8), scale, 1.0)


def mouse_balanced_scale(
    values: np.ndarray,
    mice: Sequence[object],
    units: Sequence[object] | None = None,
) -> np.ndarray:
    """Return a scale with equal mouse and, optionally, session influence."""

    values = np.asarray(values, dtype=float)
    mice = np.asarray(mice)
    if values.ndim != 2 or len(values) != len(mice) or len(values) == 0:
        raise ValueError("values and mice must be aligned, non-empty rows")
    mouse_series = pd.Series(mice)
    if units is None:
        counts = mouse_series.groupby(mouse_series).transform("size").to_numpy()
        weights = 1.0 / counts
    else:
        unit_series = pd.Series(np.asarray(units))
        if len(unit_series) != len(values):
            raise ValueError("units must align with values")
        keys = pd.DataFrame({"mouse": mouse_series, "unit": unit_series})
        rows_per_unit = keys.groupby(["mouse", "unit"])["mouse"].transform("size")
        units_per_mouse = keys.groupby("mouse")["unit"].transform("nunique")
        weights = 1.0 / (rows_per_unit.to_numpy() * units_per_mouse.to_numpy())
    weights /= weights.sum()
    center = np.sum(values * weights[:, None], axis=0)
    variance = np.sum(np.square(values - center) * weights[:, None], axis=0)
    scale = np.sqrt(variance)
    return np.where(np.isfinite(scale) & (scale > 1e-8), scale, 1.0)


def task_scale(
    train: pd.DataFrame,
    task: str,
    metrics: Sequence[str] = DEFAULT_METRICS,
) -> np.ndarray:
    units: np.ndarray | None = None
    if task == "cross_day":
        values = np.vstack(
            [
                metric_matrix(train, "before", metrics),
                metric_matrix(train, "after", metrics),
            ]
        )
        mice = np.concatenate([train["mouse"].to_numpy()] * 2)
    elif task == "next_window":
        values = metric_matrix(train, "next", metrics)
        mice = train["mouse"].to_numpy()
        units = train["behavior_session_id"].to_numpy()
    else:
        raise ValueError(f"unknown task: {task}")
    return mouse_balanced_scale(values, mice, units)


def row_nrmse(
    observed: np.ndarray,
    predicted: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    return np.sqrt(np.mean(np.square((observed - predicted) / scale), axis=1))
