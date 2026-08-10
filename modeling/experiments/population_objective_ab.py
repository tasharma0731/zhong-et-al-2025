"""Population-level bridge between stable Objective A and final Objective B.

This module is deliberately isolated from the stable release and from the
finalized same-neuron Objective B experiment.

The combined estimand is one six-number d-prime distribution summary for each
mouse, broad area, and post-learning horizon:

* horizon 0: the first W20 window in the after recording;
* horizon 1: the second W20 window in the after recording.

The architecture preserves the two established model families:

1. Objective A is the stable constraint-safe ``ridge_delta`` estimator.  It is
   refit on full-derived W20-grid session summaries so that its cross-day shift
   is expressed in the same representation as Objective B.
2. Objective B is the final dual-W20, full-derived-plus-SVD-derived 12-feature
   ridge.  Separate direct horizon models predict the no-boundary continuation
   one and two W20 windows after the final before-session window.

The B neuron forecasts are only an internal representation.  They are reduced
to q05, median, q95, SD, leaf-selective fraction, and circle-selective
fraction before the A shift is applied.  No neuron is matched or scored across
recordings.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Iterable, Literal, Sequence

import numpy as np
import pandas as pd

from modeling.config import DEFAULT_ALPHAS, DEFAULT_METRICS, StudyConfig
from modeling.estimators import CROSS_DAY_MODELS, FittedModel, fit_model
from modeling.experiments.direct_future_objective_b import (
    FittedDirectFutureRidge,
    fit_direct_future_ridge,
    select_minimum_cv,
    target_candidate,
)
from modeling.experiments.lag2_full_neural_objective_b import (
    FittedLag2Ridge,
    build_lag2_view,
)
from modeling.experiments.refined_neuron_validation import (
    CandidateSpec,
    OneSESelection,
    score_predictions,
    select_one_se,
    stable_seed,
    training_scale,
)
from modeling.preparation import make_session_pairs, metric_matrix
from modeling.targets import DistributionTargetTransform
from modeling.validation import choose_alpha


MODEL_VERSION = "1.0.0"
AREAS = ("mHV", "aHV")
METRICS = tuple(DEFAULT_METRICS)
ALPHAS = (100.0, 1_000.0, 10_000.0)
SELECTIVITY_THRESHOLD = 0.3
STABLE_A_NAME = "ridge_delta"
HORIZONS = (0, 1)
TargetParameterization = Literal["future_minus_current", "future_level"]
SelectionRule = Literal["one_se", "minimum_cv"]


class PopulationObjectiveABError(ValueError):
    """Raised when the population bridge contract is violated."""


@dataclass(frozen=True)
class BProcedure:
    """One target-parameterization and alpha-selection variant."""

    name: str
    target_parameterization: TargetParameterization
    selection_rule: SelectionRule
    display_name: str


B_PROCEDURES = (
    BProcedure(
        "delta__one_se",
        "future_minus_current",
        "one_se",
        "Delta target + one-SE",
    ),
    BProcedure(
        "delta__minimum_cv",
        "future_minus_current",
        "minimum_cv",
        "Delta target + minimum CV",
    ),
    BProcedure(
        "direct__one_se",
        "future_level",
        "one_se",
        "Direct future + one-SE",
    ),
    BProcedure(
        "direct__minimum_cv",
        "future_level",
        "minimum_cv",
        "Direct future + minimum CV",
    ),
)
B_PROCEDURE_BY_NAME = {item.name: item for item in B_PROCEDURES}


@dataclass
class PopulationABFit:
    """Frozen A model plus one fitted B model per procedure and horizon."""

    area: str
    a_model: FittedModel
    a_alpha: float
    b_models: dict[tuple[int, str], FittedLag2Ridge | FittedDirectFutureRidge]
    b_selections: pd.DataFrame
    training_mice: tuple[str, ...]


@dataclass
class PopulationABEvaluation:
    """Auditable nested-development and frozen-transfer outputs."""

    predictions: pd.DataFrame
    scores: pd.DataFrame
    summary: pd.DataFrame
    metric_errors: pd.DataFrame
    selections: pd.DataFrame
    session_summaries: pd.DataFrame
    session_pairs: pd.DataFrame
    actual_horizons: pd.DataFrame
    final_models: dict[str, PopulationABFit]


def summarize_dprime(
    values: np.ndarray,
    *,
    threshold: float = SELECTIVITY_THRESHOLD,
) -> np.ndarray:
    """Return the stable six-summary state for one neuron distribution."""

    finite = np.asarray(values, dtype=float).reshape(-1)
    finite = finite[np.isfinite(finite)]
    if len(finite) < 2:
        raise PopulationObjectiveABError(
            "a population summary requires at least two finite d-prime values"
        )
    q05, median, q95 = np.quantile(finite, (0.05, 0.50, 0.95))
    result = np.asarray(
        [
            q05,
            median,
            q95,
            np.std(finite, ddof=0),
            np.mean(finite >= threshold),
            np.mean(finite <= -threshold),
        ],
        dtype=float,
    )
    DistributionTargetTransform(METRICS).validate(result[None, :])
    return result


def _pool_equal_count_moments(
    means: np.ndarray,
    standard_deviations: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Pool equal-sized groups represented by ddof=0 means and SDs."""

    means = np.asarray(means, dtype=float)
    standard_deviations = np.asarray(standard_deviations, dtype=float)
    if means.ndim != 2 or standard_deviations.shape != means.shape:
        raise PopulationObjectiveABError(
            "window moments must be windows x neurons with aligned shapes"
        )
    pooled_mean = np.mean(means, axis=0)
    pooled_variance = np.mean(
        np.square(standard_deviations)
        + np.square(means - pooled_mean[None, :]),
        axis=0,
    )
    return pooled_mean, np.sqrt(np.maximum(pooled_variance, 0.0))


def _dprime_from_moments(
    leaf_mean: np.ndarray,
    leaf_sd: np.ndarray,
    circle_mean: np.ndarray,
    circle_sd: np.ndarray,
) -> np.ndarray:
    denominator = np.asarray(leaf_sd, dtype=float) + np.asarray(
        circle_sd,
        dtype=float,
    )
    numerator = 2.0 * (
        np.asarray(leaf_mean, dtype=float)
        - np.asarray(circle_mean, dtype=float)
    )
    result = np.full_like(denominator, np.nan, dtype=float)
    valid = (
        np.isfinite(numerator)
        & np.isfinite(denominator)
        & (denominator > 0)
    )
    result[valid] = numerator[valid] / denominator[valid]
    return result


def _ordered_transition_indices(
    metadata: pd.DataFrame,
    group_index: Iterable[int],
    *,
    window_pairs: int,
) -> np.ndarray:
    indices = np.asarray(tuple(group_index), dtype=int)
    ordered = indices[
        np.argsort(
            metadata.iloc[indices]["current_start_pair_id"].to_numpy(dtype=int)
        )
    ]
    rows = metadata.iloc[ordered]
    current = rows["current_start_pair_id"].to_numpy(dtype=int)
    following = rows["next_start_pair_id"].to_numpy(dtype=int)
    if not np.array_equal(following, current + int(window_pairs)):
        raise PopulationObjectiveABError("transition target is not the next W20")
    if len(ordered) > 1 and not np.array_equal(current[1:], following[:-1]):
        raise PopulationObjectiveABError(
            "session transitions are not a contiguous W20 chain"
        )
    return ordered


def full_window_grid_session_summaries(data: object) -> pd.DataFrame:
    """Reconstruct full-derived session summaries from all disjoint W20s.

    The compact full-neural cache retains per-role mean and ddof=0 SD for each
    W20.  Because all windows contain 20 trials per stimulus role, those
    moments can be pooled exactly across the W20 grid.  The last window is read
    from the final transition's ``next_*`` fields.
    """

    metadata = data.metadata.reset_index(drop=True)
    required = {
        "behavior_session_id",
        "recording_id",
        "mouse",
        "cohort",
        "moment",
        "area",
        "current_start_pair_id",
        "next_start_pair_id",
    }
    missing = required - set(metadata)
    if missing:
        raise PopulationObjectiveABError(
            f"full-neural metadata is missing {sorted(missing)}"
        )
    records: list[dict[str, object]] = []
    group_columns = [
        "behavior_session_id",
        "recording_id",
        "mouse",
        "cohort",
        "moment",
        "area",
    ]
    for key, group in metadata.groupby(group_columns, sort=True):
        ordered = _ordered_transition_indices(
            metadata,
            group.index,
            window_pairs=int(data.window_pairs),
        )
        first_to_penultimate = {
            role: (
                np.asarray(getattr(data, f"current_mean_{role}"), dtype=float)[
                    ordered
                ],
                np.asarray(getattr(data, f"current_sd_{role}"), dtype=float)[
                    ordered
                ],
            )
            for role in ("leaf", "circle")
        }
        final_index = int(ordered[-1])
        pooled: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for role in ("leaf", "circle"):
            means, sds = first_to_penultimate[role]
            final_mean = np.asarray(
                getattr(data, f"next_mean_{role}"),
                dtype=float,
            )[final_index][None, :]
            final_sd = np.asarray(
                getattr(data, f"next_sd_{role}"),
                dtype=float,
            )[final_index][None, :]
            pooled[role] = _pool_equal_count_moments(
                np.vstack((means, final_mean)),
                np.vstack((sds, final_sd)),
            )
        dprime = _dprime_from_moments(
            pooled["leaf"][0],
            pooled["leaf"][1],
            pooled["circle"][0],
            pooled["circle"][1],
        )
        summary = summarize_dprime(dprime)
        identity = dict(zip(group_columns, key, strict=True))
        records.append(
            {
                **identity,
                "n_valid_dprime": int(np.isfinite(dprime).sum()),
                "mean_run_speed": float(
                    np.nanmean(
                        metadata.iloc[ordered]["current_run_speed"].to_numpy(
                            dtype=float
                        )
                    )
                ),
                "w20_windows": int(len(ordered) + 1),
                **dict(zip(METRICS, summary, strict=True)),
            }
        )
    result = pd.DataFrame.from_records(records)
    if result.empty:
        raise PopulationObjectiveABError("no session summaries were reconstructed")
    return result.sort_values(
        ["cohort", "mouse", "moment", "area"]
    ).reset_index(drop=True)


def full_session_pairs(data: object) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return full-derived W20-grid session summaries and before/after pairs."""

    summaries = full_window_grid_session_summaries(data)
    pairs = make_session_pairs(summaries, metrics=METRICS)
    support = summaries[
        [
            "mouse",
            "cohort",
            "area",
            "moment",
            "n_valid_dprime",
            "w20_windows",
        ]
    ]
    for moment in ("before", "after"):
        current = support.loc[support["moment"].eq(moment)].drop(columns="moment")
        current = current.rename(
            columns={
                "n_valid_dprime": f"{moment}_n_valid_dprime",
                "w20_windows": f"{moment}_w20_windows",
            }
        )
        pairs = pairs.merge(
            current,
            on=["mouse", "cohort", "area"],
            how="left",
            validate="one_to_one",
        )
    return summaries, pairs


def actual_after_horizon_summaries(data: object) -> pd.DataFrame:
    """Return observed full-derived first and second after-W20 summaries."""

    metadata = data.metadata.reset_index(drop=True)
    records: list[dict[str, object]] = []
    after = metadata.loc[metadata["moment"].astype(str).eq("after")]
    grouping = [
        "behavior_session_id",
        "recording_id",
        "mouse",
        "cohort",
        "area",
    ]
    for key, group in after.groupby(grouping, sort=True):
        ordered = _ordered_transition_indices(
            metadata,
            group.index,
            window_pairs=int(data.window_pairs),
        )
        first = int(ordered[0])
        horizon_values = (
            np.asarray(data.current[first], dtype=float),
            np.asarray(data.future[first], dtype=float),
        )
        identity = dict(zip(grouping, key, strict=True))
        for horizon, values in enumerate(horizon_values):
            summary = summarize_dprime(values)
            records.append(
                {
                    **identity,
                    "horizon": int(horizon),
                    "window_start_pair_id": int(horizon * data.window_pairs),
                    "n_valid_dprime": int(np.isfinite(values).sum()),
                    **dict(zip(METRICS, summary, strict=True)),
                }
            )
    result = pd.DataFrame.from_records(records)
    if result.empty:
        raise PopulationObjectiveABError("no after-session horizons were found")
    return result.sort_values(
        ["cohort", "mouse", "area", "horizon"]
    ).reset_index(drop=True)


def after_window_summaries(data: object) -> pd.DataFrame:
    """Return every observed supervised after-learning W20 population state.

    Each transition contributes its current W20 once. The final transition
    additionally contributes its next W20, so the trajectory contains every
    disjoint W20 represented by the compact full-neural cache without
    duplicating shared transition boundaries.
    """

    metadata = data.metadata.reset_index(drop=True)
    required = {
        "behavior_session_id",
        "recording_id",
        "mouse",
        "cohort",
        "moment",
        "area",
        "current_start_pair_id",
        "next_start_pair_id",
    }
    missing = required - set(metadata)
    if missing:
        raise PopulationObjectiveABError(
            f"metadata is missing trajectory columns: {sorted(missing)}"
        )

    selected = metadata.loc[
        metadata["cohort"].astype(str).eq("supervised")
        & metadata["moment"].astype(str).eq("after")
        & metadata["area"].astype(str).isin(AREAS)
    ]
    records: list[dict[str, object]] = []
    grouping = [
        "behavior_session_id",
        "recording_id",
        "mouse",
        "cohort",
        "moment",
        "area",
    ]
    for key, group in selected.groupby(grouping, sort=True):
        ordered = _ordered_transition_indices(
            metadata,
            group.index,
            window_pairs=int(data.window_pairs),
        )
        starts = metadata.iloc[ordered][
            "current_start_pair_id"
        ].to_numpy(dtype=int)
        values = [
            np.asarray(data.current[int(index)], dtype=float)
            for index in ordered
        ]
        final_index = int(ordered[-1])
        starts = np.append(
            starts,
            int(metadata.iloc[final_index]["next_start_pair_id"]),
        )
        values.append(np.asarray(data.future[final_index], dtype=float))
        identity = dict(zip(grouping, key, strict=True))
        for window_index, (start, dprime) in enumerate(
            zip(starts, values, strict=True)
        ):
            summary = summarize_dprime(dprime)
            records.append(
                {
                    **identity,
                    "window_index": int(window_index),
                    "window_number": int(window_index + 1),
                    "window_start_pair_id": int(start),
                    "n_valid_dprime": int(np.isfinite(dprime).sum()),
                    **dict(zip(METRICS, summary, strict=True)),
                }
            )
    result = pd.DataFrame.from_records(records)
    if result.empty:
        raise PopulationObjectiveABError(
            "no supervised after-learning W20 trajectories are available"
        )
    return result.sort_values(
        ["mouse", "area", "window_index"]
    ).reset_index(drop=True)


def _subset_rows(data: object, indices: np.ndarray) -> object:
    """Shallow-copy an experimental cache while subsetting row-aligned values."""

    indices = np.asarray(indices, dtype=int)
    result = copy.copy(data)
    source_rows = len(data.metadata)
    for name, value in vars(data).items():
        if isinstance(value, pd.DataFrame) and len(value) == source_rows:
            setattr(result, name, value.iloc[indices].reset_index(drop=True))
        elif (
            isinstance(value, np.ndarray)
            and value.ndim >= 1
            and value.shape[0] == source_rows
        ):
            setattr(result, name, value[indices].copy())
    return result


def build_boundary_input_view(data: object) -> object:
    """Create dual-W20 inputs from the final two before-session windows."""

    metadata = data.metadata.reset_index(drop=True)
    before = metadata.loc[metadata["moment"].astype(str).eq("before")]
    chosen: list[int] = []
    for _, group in before.groupby(
        ["behavior_session_id", "area"],
        sort=True,
    ):
        ordered = _ordered_transition_indices(
            metadata,
            group.index,
            window_pairs=int(data.window_pairs),
        )
        chosen.append(int(ordered[-1]))
    selected = np.asarray(chosen, dtype=int)
    result = _subset_rows(data, selected)

    previous_full = np.asarray(result.current, dtype=float).copy()
    previous_full_denominator = np.asarray(
        result.current_denominator,
        dtype=float,
    ).copy()
    previous_svd = np.asarray(result.svd_current, dtype=float).copy()
    previous_svd_denominator = np.asarray(
        result.svd_current_denominator,
        dtype=float,
    ).copy()
    current_full = np.asarray(result.future, dtype=float).copy()
    current_full_denominator = np.asarray(
        result.next_denominator,
        dtype=float,
    ).copy()
    current_svd = np.asarray(result.svd_future, dtype=float).copy()
    current_svd_denominator = np.asarray(
        result.svd_next_denominator,
        dtype=float,
    ).copy()

    result.previous_full = previous_full
    result.previous_full_denominator = previous_full_denominator
    result.previous_svd = previous_svd
    result.previous_svd_denominator = previous_svd_denominator
    result.current = current_full
    result.current_denominator = current_full_denominator
    result.svd_current = current_svd
    result.svd_current_denominator = current_svd_denominator
    result.future = np.full_like(current_full, np.nan, dtype=float)
    result.metadata = result.metadata.copy()
    result.metadata["previous_start_pair_id"] = result.metadata[
        "current_start_pair_id"
    ].to_numpy(dtype=int)
    result.metadata["current_start_pair_id"] = result.metadata[
        "next_start_pair_id"
    ].to_numpy(dtype=int)
    result.metadata["next_start_pair_id"] = (
        result.metadata["current_start_pair_id"].to_numpy(dtype=int)
        + int(data.window_pairs)
    )
    return result


def build_horizon_view(data: object, horizon: int) -> object:
    """Return t-1,t inputs with target t+1 or t+2."""

    if int(horizon) not in HORIZONS:
        raise PopulationObjectiveABError(
            f"horizon must be one of {HORIZONS}, received {horizon}"
        )
    lag2 = build_lag2_view(data)
    if int(horizon) == 0:
        return lag2

    metadata = data.metadata.reset_index(drop=True)
    lookup = {
        (
            str(row.behavior_session_id),
            str(row.area),
            int(row.current_start_pair_id),
        ): int(index)
        for index, row in metadata.iterrows()
    }
    keep: list[int] = []
    next_source: list[int] = []
    for index, row in lag2.metadata.reset_index(drop=True).iterrows():
        key = (
            str(row["behavior_session_id"]),
            str(row["area"]),
            int(row["current_start_pair_id"]) + int(data.window_pairs),
        )
        if key in lookup:
            keep.append(int(index))
            next_source.append(int(lookup[key]))
    if not keep:
        raise PopulationObjectiveABError(
            "no exact t-1,t,t+1,t+2 sequences are available"
        )
    result = _subset_rows(lag2, np.asarray(keep, dtype=int))
    result.future = np.asarray(data.future, dtype=float)[
        np.asarray(next_source, dtype=int)
    ].copy()
    result.metadata = result.metadata.copy()
    result.metadata["target_start_pair_id"] = (
        result.metadata["current_start_pair_id"].to_numpy(dtype=int)
        + 2 * int(data.window_pairs)
    )
    result.metadata["horizon2_source_data_index"] = np.asarray(
        next_source,
        dtype=int,
    )
    return result


def _stable_a_spec():
    return next(spec for spec in CROSS_DAY_MODELS if spec.name == STABLE_A_NAME)


def fit_stable_a(
    training_pairs: pd.DataFrame,
    *,
    alphas: Sequence[float] = DEFAULT_ALPHAS,
) -> tuple[FittedModel, float]:
    """Fit the exact stable constraint-safe Objective A ridge estimator."""

    areas = tuple(sorted(training_pairs["area"].astype(str).unique()))
    if len(areas) != 1:
        raise PopulationObjectiveABError(
            "stable A must be fit separately for exactly one area"
        )
    config = StudyConfig(
        areas=areas,
        metrics=METRICS,
        alphas=tuple(float(alpha) for alpha in alphas),
        bootstrap_repeats=100,
    )
    spec = _stable_a_spec()
    alpha = choose_alpha(training_pairs, spec, config)
    return (
        fit_model(training_pairs, spec, metrics=METRICS, alpha=alpha),
        float(alpha),
    )


def a_shift(
    model: FittedModel,
    frame: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Return stable-A prediction and its transformed cross-day delta."""

    transform = DistributionTargetTransform(METRICS)
    before = metric_matrix(frame, "before", METRICS)
    prediction = model.predict(frame)
    delta = transform.transform(prediction) - transform.transform(before)
    return prediction, delta


def combine_a_shift_with_b(
    b_summary: np.ndarray,
    transformed_a_delta: np.ndarray,
) -> np.ndarray:
    """Apply the stable A shift to one B-predicted distribution summary."""

    transform = DistributionTargetTransform(METRICS)
    b_summary = np.asarray(b_summary, dtype=float).reshape(1, -1)
    transformed_a_delta = np.asarray(
        transformed_a_delta,
        dtype=float,
    ).reshape(1, -1)
    if b_summary.shape[1] != len(METRICS):
        raise PopulationObjectiveABError("B summary has the wrong metric count")
    prediction = transform.inverse_transform(
        transform.transform(b_summary) + transformed_a_delta
    )
    transform.validate(prediction)
    return prediction[0]


def candidate_grid(
    target_parameterization: TargetParameterization,
    *,
    alphas: Sequence[float] = ALPHAS,
) -> tuple[CandidateSpec, ...]:
    return tuple(
        target_candidate(target_parameterization, alpha=float(alpha))
        for alpha in alphas
    )


def candidate_lomo_scores(
    data: object,
    pool: np.ndarray,
    candidates: Sequence[CandidateSpec],
    *,
    area: str,
    seed: int,
    scope: str,
) -> pd.DataFrame:
    """Score fixed B candidates by leave-one-training-mouse-out validation."""

    pool = np.asarray(pool, dtype=int)
    rows = data.metadata.iloc[pool]
    mice = tuple(sorted(rows["mouse"].astype(str).unique()))
    if len(mice) < 3:
        raise PopulationObjectiveABError(
            "B alpha selection requires at least three training mice"
        )
    records: list[dict[str, object]] = []
    for candidate in candidates:
        for mouse in mice:
            validation = pool[
                rows["mouse"].astype(str).eq(mouse).to_numpy()
            ]
            train = pool[~np.isin(pool, validation)]
            fitted = candidate.fit(
                data,
                train,
                stable_seed(seed, scope, area, candidate.name, mouse),
            )
            prediction = candidate.predict(fitted, data, validation)
            _, mouse_scores = score_predictions(
                data,
                validation,
                prediction,
                scale=training_scale(data, train),
            )
            current = mouse_scores.loc[
                mouse_scores["area"].astype(str).eq(area)
                & mouse_scores["mouse"].astype(str).eq(mouse)
            ]
            if len(current) != 1:
                raise PopulationObjectiveABError(
                    f"candidate scoring did not return one row for {area}/{mouse}"
                )
            records.append(
                {
                    "candidate": candidate.name,
                    "validation_mouse": mouse,
                    "area": area,
                    "scope": scope,
                    "nrmse": float(current.iloc[0]["nrmse"]),
                }
            )
    return pd.DataFrame.from_records(records)


def select_b_candidate(
    scores: pd.DataFrame,
    candidates: Sequence[CandidateSpec],
    *,
    rule: SelectionRule,
) -> OneSESelection:
    if rule == "one_se":
        return select_one_se(scores, candidates)
    if rule == "minimum_cv":
        return select_minimum_cv(scores, candidates)
    raise PopulationObjectiveABError(f"unknown selection rule: {rule}")


def _pool_indices(
    data: object,
    *,
    area: str,
    cohort: str,
    moment: str,
    excluded_mouse: str | None = None,
) -> np.ndarray:
    metadata = data.metadata
    mask = (
        metadata["area"].astype(str).eq(area)
        & metadata["cohort"].astype(str).eq(cohort)
        & metadata["moment"].astype(str).eq(moment)
    )
    if excluded_mouse is not None:
        mask &= metadata["mouse"].astype(str).ne(excluded_mouse)
    result = np.flatnonzero(mask.to_numpy())
    if not len(result):
        raise PopulationObjectiveABError(
            f"no rows for {area}/{cohort}/{moment}"
        )
    return result


def _boundary_index(
    boundary: object,
    *,
    area: str,
    cohort: str,
    mouse: str,
) -> np.ndarray:
    metadata = boundary.metadata
    selected = np.flatnonzero(
        (
            metadata["area"].astype(str).eq(area)
            & metadata["cohort"].astype(str).eq(cohort)
            & metadata["mouse"].astype(str).eq(mouse)
        ).to_numpy()
    )
    if len(selected) != 1:
        raise PopulationObjectiveABError(
            f"expected one boundary row for {cohort}/{mouse}/{area}; "
            f"found {len(selected)}"
        )
    return selected


def _actual_row(
    actual: pd.DataFrame,
    *,
    area: str,
    cohort: str,
    mouse: str,
    horizon: int,
) -> pd.DataFrame:
    selected = actual.loc[
        actual["area"].astype(str).eq(area)
        & actual["cohort"].astype(str).eq(cohort)
        & actual["mouse"].astype(str).eq(mouse)
        & actual["horizon"].eq(int(horizon))
    ]
    if len(selected) != 1:
        raise PopulationObjectiveABError(
            f"expected one actual row for {cohort}/{mouse}/{area}/h{horizon}"
        )
    return selected


def _pair_row(
    pairs: pd.DataFrame,
    *,
    area: str,
    cohort: str,
    mouse: str,
) -> pd.DataFrame:
    selected = pairs.loc[
        pairs["area"].astype(str).eq(area)
        & pairs["cohort"].astype(str).eq(cohort)
        & pairs["mouse"].astype(str).eq(mouse)
    ]
    if len(selected) != 1:
        raise PopulationObjectiveABError(
            f"expected one session pair for {cohort}/{mouse}/{area}"
        )
    return selected


def _target_scale(
    actual: pd.DataFrame,
    *,
    area: str,
    training_mice: Sequence[str],
) -> np.ndarray:
    selected = actual.loc[
        actual["area"].astype(str).eq(area)
        & actual["cohort"].astype(str).eq("unsupervised")
        & actual["mouse"].astype(str).isin(tuple(training_mice))
    ]
    values = selected[list(METRICS)].to_numpy(dtype=float)
    scale = np.std(values, axis=0, ddof=1)
    return np.where(np.isfinite(scale) & (scale > 1e-8), scale, 1.0)


def _prediction_records(
    *,
    split: str,
    cohort: str,
    area: str,
    mouse: str,
    horizon: int,
    model: str,
    display_model: str,
    observed: np.ndarray,
    predicted: np.ndarray,
    scale: np.ndarray,
    procedure: str = "",
    a_alpha: float | None = None,
    b_alpha: float | None = None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    metric_records: list[dict[str, object]] = []
    squared: list[float] = []
    for index, metric in enumerate(METRICS):
        residual = float(observed[index] - predicted[index])
        scaled = residual / float(scale[index])
        squared.append(scaled**2)
        metric_records.append(
            {
                "split": split,
                "cohort": cohort,
                "area": area,
                "mouse": mouse,
                "horizon": int(horizon),
                "model": model,
                "display_model": display_model,
                "procedure": procedure,
                "metric": metric,
                "observed": float(observed[index]),
                "predicted": float(predicted[index]),
                "residual": residual,
                "absolute_error": abs(residual),
                "scale": float(scale[index]),
                "scaled_absolute_error": abs(scaled),
                "scaled_squared_error": scaled**2,
                "a_alpha": a_alpha,
                "b_alpha": b_alpha,
            }
        )
    score = {
        "split": split,
        "cohort": cohort,
        "area": area,
        "mouse": mouse,
        "horizon": int(horizon),
        "model": model,
        "display_model": display_model,
        "procedure": procedure,
        "nrmse": float(np.sqrt(np.mean(squared))),
        "a_alpha": a_alpha,
        "b_alpha": b_alpha,
    }
    return metric_records, score


def _selection_record(
    *,
    split: str,
    area: str,
    horizon: int,
    procedure: BProcedure,
    selection: OneSESelection,
    outer_mouse: str | None,
) -> dict[str, object]:
    candidate = selection.candidate.name
    alpha = float(candidate.rsplit("__alpha_", 1)[1])
    return {
        "split": split,
        "area": area,
        "horizon": int(horizon),
        "procedure": procedure.name,
        "target_parameterization": procedure.target_parameterization,
        "selection_rule": procedure.selection_rule,
        "outer_mouse": outer_mouse,
        "selected_candidate": candidate,
        "selected_alpha": alpha,
        "best_candidate": selection.best_candidate,
        "best_mean_nrmse": selection.best_mean_nrmse,
        "best_sem_nrmse": selection.best_sem_nrmse,
        "selection_threshold_nrmse": selection.threshold_nrmse,
    }


def _fit_selected_b_models(
    data_by_horizon: dict[int, object],
    *,
    area: str,
    excluded_mouse: str | None,
    alphas: Sequence[float],
    seed: int,
    split: str,
) -> tuple[
    dict[tuple[int, str], FittedLag2Ridge | FittedDirectFutureRidge],
    pd.DataFrame,
]:
    fitted: dict[
        tuple[int, str],
        FittedLag2Ridge | FittedDirectFutureRidge,
    ] = {}
    records: list[dict[str, object]] = []
    for horizon, data in data_by_horizon.items():
        training = _pool_indices(
            data,
            area=area,
            cohort="unsupervised",
            moment="before",
            excluded_mouse=excluded_mouse,
        )
        for parameterization in ("future_minus_current", "future_level"):
            candidates = candidate_grid(parameterization, alphas=alphas)
            inner_scores = candidate_lomo_scores(
                data,
                training,
                candidates,
                area=area,
                seed=seed,
                scope=(
                    f"{split}:{area}:h{horizon}:"
                    f"{excluded_mouse or 'all'}:{parameterization}"
                ),
            )
            for procedure in (
                item
                for item in B_PROCEDURES
                if item.target_parameterization == parameterization
            ):
                selection = select_b_candidate(
                    inner_scores,
                    candidates,
                    rule=procedure.selection_rule,
                )
                model = selection.candidate.fit(
                    data,
                    training,
                    stable_seed(
                        seed,
                        split,
                        area,
                        horizon,
                        procedure.name,
                        excluded_mouse or "all",
                    ),
                )
                fitted[(horizon, procedure.name)] = model
                records.append(
                    _selection_record(
                        split=split,
                        area=area,
                        horizon=horizon,
                        procedure=procedure,
                        selection=selection,
                        outer_mouse=excluded_mouse,
                    )
                )
    return fitted, pd.DataFrame.from_records(records)


def _evaluate_mouse(
    *,
    split: str,
    cohort: str,
    area: str,
    mouse: str,
    boundary: object,
    actual: pd.DataFrame,
    session_pair: pd.DataFrame,
    a_model: FittedModel,
    a_alpha: float,
    b_models: dict[
        tuple[int, str],
        FittedLag2Ridge | FittedDirectFutureRidge,
    ],
    scale: np.ndarray,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    metrics: list[dict[str, object]] = []
    scores: list[dict[str, object]] = []
    a_prediction, transformed_delta = a_shift(a_model, session_pair)
    boundary_index = _boundary_index(
        boundary,
        area=area,
        cohort=cohort,
        mouse=mouse,
    )
    last_before = summarize_dprime(
        np.asarray(boundary.current, dtype=float)[int(boundary_index[0])]
    )
    for horizon in HORIZONS:
        observed = _actual_row(
            actual,
            area=area,
            cohort=cohort,
            mouse=mouse,
            horizon=horizon,
        )[list(METRICS)].to_numpy(dtype=float)[0]

        baseline_states = (
            (
                "last_before_persistence",
                "Last before-W20 persistence",
                last_before,
            ),
            (
                "stable_a_ridge",
                "Stable A ridge",
                a_prediction[0],
            ),
        )
        for model_name, display_name, prediction in baseline_states:
            metric_rows, score = _prediction_records(
                split=split,
                cohort=cohort,
                area=area,
                mouse=mouse,
                horizon=horizon,
                model=model_name,
                display_model=display_name,
                observed=observed,
                predicted=prediction,
                scale=scale,
                a_alpha=a_alpha if model_name == "stable_a_ridge" else None,
            )
            metrics.extend(metric_rows)
            scores.append(score)

        for procedure in B_PROCEDURES:
            fitted = b_models[(horizon, procedure.name)]
            b_prediction = fitted.predict(boundary, boundary_index)
            b_summary = summarize_dprime(b_prediction[0])
            combined = combine_a_shift_with_b(
                b_summary,
                transformed_delta[0],
            )
            b_alpha = float(fitted.ridge.alpha)
            variants = (
                (
                    f"b_only__{procedure.name}",
                    f"B only: {procedure.display_name}",
                    b_summary,
                ),
                (
                    f"a_plus_b__{procedure.name}",
                    f"A + B: {procedure.display_name}",
                    combined,
                ),
            )
            for model_name, display_name, prediction in variants:
                metric_rows, score = _prediction_records(
                    split=split,
                    cohort=cohort,
                    area=area,
                    mouse=mouse,
                    horizon=horizon,
                    model=model_name,
                    display_model=display_name,
                    procedure=procedure.name,
                    observed=observed,
                    predicted=prediction,
                    scale=scale,
                    a_alpha=a_alpha if model_name.startswith("a_plus_b") else None,
                    b_alpha=b_alpha,
                )
                metrics.extend(metric_rows)
                scores.append(score)
    return metrics, scores


def summarize_evaluation(
    scores: pd.DataFrame,
) -> pd.DataFrame:
    """Mouse-balanced score summary with persistence-relative improvement."""

    summary = (
        scores.groupby(
            ["split", "area", "horizon", "model", "display_model"],
            as_index=False,
        )
        .agg(
            mice=("mouse", "nunique"),
            mean_nrmse=("nrmse", "mean"),
            sem_nrmse=("nrmse", "sem"),
        )
        .sort_values(["split", "area", "horizon", "mean_nrmse"])
    )
    persistence = summary.loc[
        summary["model"].eq("last_before_persistence"),
        ["split", "area", "horizon", "mean_nrmse"],
    ].rename(columns={"mean_nrmse": "persistence_nrmse"})
    a_only = summary.loc[
        summary["model"].eq("stable_a_ridge"),
        ["split", "area", "horizon", "mean_nrmse"],
    ].rename(columns={"mean_nrmse": "a_only_nrmse"})
    summary = summary.merge(
        persistence,
        on=["split", "area", "horizon"],
        how="left",
        validate="many_to_one",
    ).merge(
        a_only,
        on=["split", "area", "horizon"],
        how="left",
        validate="many_to_one",
    )
    summary["improvement_vs_persistence_pct"] = 100.0 * (
        summary["persistence_nrmse"] - summary["mean_nrmse"]
    ) / summary["persistence_nrmse"]
    summary["improvement_vs_a_only_pct"] = 100.0 * (
        summary["a_only_nrmse"] - summary["mean_nrmse"]
    ) / summary["a_only_nrmse"]
    return summary


def metric_error_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    """Return raw-unit MAE/RMSE for every output metric."""

    return (
        predictions.groupby(
            [
                "split",
                "area",
                "horizon",
                "model",
                "display_model",
                "metric",
            ],
            as_index=False,
        )
        .agg(
            mice=("mouse", "nunique"),
            mae=("absolute_error", "mean"),
            rmse=(
                "residual",
                lambda values: float(
                    np.sqrt(np.mean(np.square(values.to_numpy(dtype=float))))
                ),
            ),
            mean_observed=("observed", "mean"),
            mean_predicted=("predicted", "mean"),
        )
        .sort_values(["split", "area", "horizon", "model", "metric"])
    )


def evaluate_population_objective_ab(
    data: object,
    *,
    areas: Sequence[str] = AREAS,
    a_alphas: Sequence[float] = DEFAULT_ALPHAS,
    b_alphas: Sequence[float] = ALPHAS,
    seed: int = 2025,
) -> PopulationABEvaluation:
    """Run nested unsupervised development and frozen supervised transfer."""

    areas = tuple(str(area) for area in areas)
    if set(areas) - set(AREAS):
        raise PopulationObjectiveABError(
            f"full-neural cache supports only {AREAS}"
        )
    if int(data.window_pairs) != 20:
        raise PopulationObjectiveABError("combined A+B requires W20 data")
    session_summaries, pairs = full_session_pairs(data)
    actual = actual_after_horizon_summaries(data)
    boundary = build_boundary_input_view(data)
    horizon_data = {
        horizon: build_horizon_view(data, horizon)
        for horizon in HORIZONS
    }

    prediction_records: list[dict[str, object]] = []
    score_records: list[dict[str, object]] = []
    selection_frames: list[pd.DataFrame] = []
    final_models: dict[str, PopulationABFit] = {}

    for area in areas:
        unsupervised_pairs = pairs.loc[
            pairs["area"].astype(str).eq(area)
            & pairs["cohort"].astype(str).eq("unsupervised")
        ].reset_index(drop=True)
        training_mice = tuple(
            sorted(unsupervised_pairs["mouse"].astype(str).unique())
        )
        if len(training_mice) < 4:
            raise PopulationObjectiveABError(
                f"{area} has too few unsupervised paired mice"
            )

        for outer_mouse in training_mice:
            a_train = unsupervised_pairs.loc[
                unsupervised_pairs["mouse"].astype(str).ne(outer_mouse)
            ].reset_index(drop=True)
            a_test = _pair_row(
                pairs,
                area=area,
                cohort="unsupervised",
                mouse=outer_mouse,
            )
            a_model, a_alpha = fit_stable_a(a_train, alphas=a_alphas)
            b_models, selections = _fit_selected_b_models(
                horizon_data,
                area=area,
                excluded_mouse=outer_mouse,
                alphas=b_alphas,
                seed=seed,
                split="unsupervised_lomo",
            )
            selection_frames.append(selections)
            scale = _target_scale(
                actual,
                area=area,
                training_mice=tuple(
                    mouse for mouse in training_mice if mouse != outer_mouse
                ),
            )
            metric_rows, scores = _evaluate_mouse(
                split="unsupervised_lomo",
                cohort="unsupervised",
                area=area,
                mouse=outer_mouse,
                boundary=boundary,
                actual=actual,
                session_pair=a_test,
                a_model=a_model,
                a_alpha=a_alpha,
                b_models=b_models,
                scale=scale,
            )
            prediction_records.extend(metric_rows)
            score_records.extend(scores)

        a_model, a_alpha = fit_stable_a(
            unsupervised_pairs,
            alphas=a_alphas,
        )
        b_models, selections = _fit_selected_b_models(
            horizon_data,
            area=area,
            excluded_mouse=None,
            alphas=b_alphas,
            seed=seed,
            split="supervised_test",
        )
        selection_frames.append(selections)
        final_models[area] = PopulationABFit(
            area=area,
            a_model=a_model,
            a_alpha=a_alpha,
            b_models=b_models,
            b_selections=selections.copy(),
            training_mice=training_mice,
        )
        scale = _target_scale(
            actual,
            area=area,
            training_mice=training_mice,
        )
        supervised_mice = sorted(
            set(
                pairs.loc[
                    pairs["area"].astype(str).eq(area)
                    & pairs["cohort"].astype(str).eq("supervised"),
                    "mouse",
                ].astype(str)
            )
            & set(
                boundary.metadata.loc[
                    boundary.metadata["area"].astype(str).eq(area)
                    & boundary.metadata["cohort"].astype(str).eq("supervised"),
                    "mouse",
                ].astype(str)
            )
            & set(
                actual.loc[
                    actual["area"].astype(str).eq(area)
                    & actual["cohort"].astype(str).eq("supervised"),
                    "mouse",
                ].astype(str)
            )
        )
        if not supervised_mice:
            raise PopulationObjectiveABError(
                f"{area} has no complete supervised transfer mice"
            )
        for mouse in supervised_mice:
            a_test = _pair_row(
                pairs,
                area=area,
                cohort="supervised",
                mouse=mouse,
            )
            metric_rows, scores = _evaluate_mouse(
                split="supervised_test",
                cohort="supervised",
                area=area,
                mouse=mouse,
                boundary=boundary,
                actual=actual,
                session_pair=a_test,
                a_model=a_model,
                a_alpha=a_alpha,
                b_models=b_models,
                scale=scale,
            )
            prediction_records.extend(metric_rows)
            score_records.extend(scores)

    predictions = pd.DataFrame.from_records(prediction_records)
    scores = pd.DataFrame.from_records(score_records)
    selections = pd.concat(selection_frames, ignore_index=True)
    return PopulationABEvaluation(
        predictions=predictions,
        scores=scores,
        summary=summarize_evaluation(scores),
        metric_errors=metric_error_summary(predictions),
        selections=selections.sort_values(
            ["split", "area", "horizon", "procedure", "outer_mouse"],
            na_position="last",
        ).reset_index(drop=True),
        session_summaries=session_summaries,
        session_pairs=pairs,
        actual_horizons=actual,
        final_models=final_models,
    )


__all__ = [
    "ALPHAS",
    "AREAS",
    "B_PROCEDURES",
    "HORIZONS",
    "METRICS",
    "MODEL_VERSION",
    "PopulationABEvaluation",
    "PopulationABFit",
    "PopulationObjectiveABError",
    "a_shift",
    "after_window_summaries",
    "actual_after_horizon_summaries",
    "build_boundary_input_view",
    "build_horizon_view",
    "candidate_lomo_scores",
    "combine_a_shift_with_b",
    "evaluate_population_objective_ab",
    "fit_stable_a",
    "full_session_pairs",
    "full_window_grid_session_summaries",
    "metric_error_summary",
    "summarize_dprime",
    "summarize_evaluation",
]
