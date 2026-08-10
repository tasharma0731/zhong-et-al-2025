"""Direct population forecaster for the before/after learning boundary.

This module is an isolated alternative to composing the existing Objective A
and Objective B predictions.  It constructs one example per mouse and broad
area and evaluates the quantity that is ultimately reported: the six-number
population state in the first two W20 windows of the after-learning session.

Every learned prediction is anchored to persistence in the constraint-safe
space.  If ``p`` is the last before-session W20 population state, the model is

    g(y_h) = g(p) + r_h(x),

where ``h`` is the first or second after-learning W20 and ``g`` preserves
ordered quantiles, positive spread, and valid selectivity fractions.  Each
coordinate of ``r_h`` is an independent ridge head with a standard,
unpenalized intercept.  Exact persistence and the training-set mean shift are
also explicit candidates rather than limits that ridge must approximate.

Model selection is mouse nested:

* the outer unsupervised leave-one-mouse-out loop estimates performance;
* an inner leave-one-mouse-out loop chooses the candidate separately for each
  area and horizon, using final raw six-output population NRMSE (minimum-CV
  selection by default, with a one-standard-error sensitivity available);
* one final selection on all unsupervised mice is frozen before supervised
  transfer.

No raw traces are read.  "Full" and "SVD" below refer to the two d-prime
representations already present in the compact full-neural cache.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from modeling.config import DEFAULT_METRICS
from modeling.experiments.population_objective_ab import (
    actual_after_horizon_summaries,
    build_boundary_input_view,
    full_window_grid_session_summaries,
    summarize_dprime,
)
from modeling.targets import DistributionTargetTransform


METRICS = tuple(DEFAULT_METRICS)
HORIZONS = (0, 1)
TRANSFORMED_METRIC_NAMES = (
    "median",
    "log_lower_gap",
    "log_upper_gap",
    "log_sd_dprime",
    "leaf_vs_nonselective_log_ratio",
    "circle_vs_nonselective_log_ratio",
)
BEHAVIOR_NAMES = (
    "before_session_mean_run_speed",
    "penultimate_w20_run_speed",
)

FeatureVariant = Literal[
    "session_only",
    "dual_full",
    "dual_full_svd",
    "dual_full_svd_behavior",
]
SelectionRule = Literal["one_se", "minimum_cv"]

FEATURE_VARIANTS: tuple[FeatureVariant, ...] = (
    "session_only",
    "dual_full",
    "dual_full_svd",
    "dual_full_svd_behavior",
)

# The final values deliberately include penalties that are very large relative
# to the 7-9 training mice.  With an unpenalized intercept, these values approach
# the exact training mean-shift candidate.
DEFAULT_ALPHAS = (
    0.1,
    1.0,
    10.0,
    100.0,
    1_000.0,
    10_000.0,
)


class BoundaryPopulationError(ValueError):
    """Raised when the population-boundary contract is violated."""


@dataclass
class BoundaryPopulationData:
    """One aligned population example per mouse and broad area."""

    metadata: pd.DataFrame
    whole_before: np.ndarray
    previous_full: np.ndarray
    current_full: np.ndarray
    previous_svd: np.ndarray
    current_svd: np.ndarray
    behavior: np.ndarray
    targets: np.ndarray

    @property
    def persistence(self) -> np.ndarray:
        """Last before-session full-derived W20 population state."""

        return self.current_full

    def __len__(self) -> int:
        return len(self.metadata)

    def subset(
        self,
        indices: Sequence[int] | np.ndarray,
    ) -> "BoundaryPopulationData":
        selected = _validate_indices(self, indices, name="subset")
        return BoundaryPopulationData(
            metadata=self.metadata.iloc[selected].reset_index(drop=True),
            whole_before=self.whole_before[selected].copy(),
            previous_full=self.previous_full[selected].copy(),
            current_full=self.current_full[selected].copy(),
            previous_svd=self.previous_svd[selected].copy(),
            current_svd=self.current_svd[selected].copy(),
            behavior=self.behavior[selected].copy(),
            targets=self.targets[selected].copy(),
        )

    def feature_matrix(
        self,
        variant: FeatureVariant,
    ) -> tuple[np.ndarray, tuple[str, ...]]:
        """Return one future-blind feature matrix and auditable names."""

        if variant not in FEATURE_VARIANTS:
            raise BoundaryPopulationError(f"unknown feature variant: {variant}")
        transform = DistributionTargetTransform(METRICS)
        blocks: list[tuple[str, np.ndarray]] = [
            ("whole_before", transform.transform(self.whole_before)),
        ]
        if variant != "session_only":
            blocks.extend(
                (
                    (
                        "full_tminus1",
                        transform.transform(self.previous_full),
                    ),
                    ("full_t", transform.transform(self.current_full)),
                )
            )
        if variant in ("dual_full_svd", "dual_full_svd_behavior"):
            blocks.extend(
                (
                    (
                        "svd_tminus1",
                        transform.transform(self.previous_svd),
                    ),
                    ("svd_t", transform.transform(self.current_svd)),
                )
            )

        matrices = [values for _, values in blocks]
        names = tuple(
            f"{block}_{metric}"
            for block, _ in blocks
            for metric in TRANSFORMED_METRIC_NAMES
        )
        if variant == "dual_full_svd_behavior":
            matrices.append(np.asarray(self.behavior, dtype=float))
            names += BEHAVIOR_NAMES
        return np.column_stack(matrices), names


def _validate_population_data(data: BoundaryPopulationData) -> None:
    rows = len(data.metadata)
    if rows == 0:
        raise BoundaryPopulationError("boundary data must not be empty")
    required = {"mouse", "cohort", "area", "before_behavior_session_id"}
    missing = required - set(data.metadata)
    if missing:
        raise BoundaryPopulationError(
            f"boundary metadata is missing {sorted(missing)}"
        )
    if data.metadata.duplicated(["mouse", "cohort", "area"]).any():
        raise BoundaryPopulationError(
            "boundary data must contain one row per mouse/cohort/area"
        )
    state_shape = (rows, len(METRICS))
    for name in (
        "whole_before",
        "previous_full",
        "current_full",
        "previous_svd",
        "current_svd",
    ):
        values = np.asarray(getattr(data, name), dtype=float)
        if values.shape != state_shape:
            raise BoundaryPopulationError(
                f"{name} has shape {values.shape}; expected {state_shape}"
            )
    if np.asarray(data.behavior).shape != (rows, len(BEHAVIOR_NAMES)):
        raise BoundaryPopulationError(
            "behavior must align with rows and declared behavior features"
        )
    expected_targets = (rows, len(HORIZONS), len(METRICS))
    if np.asarray(data.targets).shape != expected_targets:
        raise BoundaryPopulationError(
            f"targets have shape {np.asarray(data.targets).shape}; "
            f"expected {expected_targets}"
        )

    transform = DistributionTargetTransform(METRICS)
    for name in (
        "whole_before",
        "previous_full",
        "current_full",
        "previous_svd",
        "current_svd",
    ):
        transform.validate(np.asarray(getattr(data, name), dtype=float))
    transform.validate(data.targets.reshape(-1, len(METRICS)))


def _validate_indices(
    data: BoundaryPopulationData,
    indices: Sequence[int] | np.ndarray,
    *,
    name: str,
) -> np.ndarray:
    selected = np.asarray(indices, dtype=int)
    if selected.ndim != 1 or len(selected) == 0:
        raise BoundaryPopulationError(f"{name} indices must be non-empty")
    if len(np.unique(selected)) != len(selected):
        raise BoundaryPopulationError(f"{name} indices must be unique")
    if selected.min() < 0 or selected.max() >= len(data):
        raise BoundaryPopulationError(f"{name} index lies outside the data")
    return selected


def _one_row(
    frame: pd.DataFrame,
    *,
    cohort: str,
    mouse: str,
    area: str,
    horizon: int | None = None,
) -> pd.Series:
    selected = frame.loc[
        frame["cohort"].astype(str).eq(cohort)
        & frame["mouse"].astype(str).eq(mouse)
        & frame["area"].astype(str).eq(area)
    ]
    if horizon is not None:
        selected = selected.loc[selected["horizon"].eq(int(horizon))]
    if len(selected) != 1:
        suffix = "" if horizon is None else f"/h{horizon}"
        raise BoundaryPopulationError(
            f"expected one row for {cohort}/{mouse}/{area}{suffix}; "
            f"found {len(selected)}"
        )
    return selected.iloc[0]


def build_boundary_population_data(data: object) -> BoundaryPopulationData:
    """Build complete before-only examples and two after-W20 targets.

    Sessions without all required pieces are excluded by an explicit key
    intersection.  In the current cache this excludes, for example, a
    before-learning recording that does not contain the two W20 windows needed
    by the dual-window input.
    """

    if int(data.window_pairs) != 20:
        raise BoundaryPopulationError("boundary forecaster requires W20 data")
    summaries = full_window_grid_session_summaries(data)
    before = summaries.loc[
        summaries["moment"].astype(str).eq("before")
    ].copy()
    actual = actual_after_horizon_summaries(data)
    boundary = build_boundary_input_view(data)
    boundary_metadata = boundary.metadata.reset_index(drop=True)

    before_keys = {
        (str(row.cohort), str(row.mouse), str(row.area))
        for row in before.itertuples(index=False)
    }
    actual_key_counts = (
        actual.groupby(["cohort", "mouse", "area"])["horizon"]
        .nunique()
        .to_dict()
    )
    actual_keys = {
        tuple(map(str, key))
        for key, count in actual_key_counts.items()
        if int(count) == len(HORIZONS)
    }
    boundary_keys = {
        (str(row.cohort), str(row.mouse), str(row.area))
        for row in boundary_metadata.itertuples(index=False)
    }
    complete = sorted(before_keys & actual_keys & boundary_keys)
    if not complete:
        raise BoundaryPopulationError(
            "no mouse/area has a complete before input and two after targets"
        )

    metadata_records: list[dict[str, object]] = []
    whole_before: list[np.ndarray] = []
    previous_full: list[np.ndarray] = []
    current_full: list[np.ndarray] = []
    previous_svd: list[np.ndarray] = []
    current_svd: list[np.ndarray] = []
    behavior: list[np.ndarray] = []
    targets: list[np.ndarray] = []

    for cohort, mouse, area in complete:
        before_row = _one_row(
            before,
            cohort=cohort,
            mouse=mouse,
            area=area,
        )
        boundary_positions = np.flatnonzero(
            (
                boundary_metadata["cohort"].astype(str).eq(cohort)
                & boundary_metadata["mouse"].astype(str).eq(mouse)
                & boundary_metadata["area"].astype(str).eq(area)
            ).to_numpy()
        )
        if len(boundary_positions) != 1:
            raise BoundaryPopulationError(
                f"expected one boundary input for {cohort}/{mouse}/{area}"
            )
        position = int(boundary_positions[0])
        horizon_targets = np.vstack(
            [
                _one_row(
                    actual,
                    cohort=cohort,
                    mouse=mouse,
                    area=area,
                    horizon=horizon,
                )[list(METRICS)].to_numpy(dtype=float)
                for horizon in HORIZONS
            ]
        )
        metadata_records.append(
            {
                "mouse": mouse,
                "cohort": cohort,
                "area": area,
                "before_behavior_session_id": str(
                    before_row["behavior_session_id"]
                ),
                "before_recording_id": str(before_row["recording_id"]),
                "before_w20_windows": int(before_row["w20_windows"]),
                "after_behavior_session_id": str(
                    _one_row(
                        actual,
                        cohort=cohort,
                        mouse=mouse,
                        area=area,
                        horizon=0,
                    )["behavior_session_id"]
                ),
            }
        )
        whole_before.append(
            before_row[list(METRICS)].to_numpy(dtype=float)
        )
        previous_full.append(
            summarize_dprime(
                np.asarray(boundary.previous_full[position], dtype=float)
            )
        )
        current_full.append(
            summarize_dprime(
                np.asarray(boundary.current[position], dtype=float)
            )
        )
        previous_svd.append(
            summarize_dprime(
                np.asarray(boundary.previous_svd[position], dtype=float)
            )
        )
        current_svd.append(
            summarize_dprime(
                np.asarray(boundary.svd_current[position], dtype=float)
            )
        )
        behavior.append(
            np.asarray(
                [
                    float(before_row["mean_run_speed"]),
                    float(
                        boundary_metadata.iloc[position]["current_run_speed"]
                    ),
                ],
                dtype=float,
            )
        )
        targets.append(horizon_targets)

    result = BoundaryPopulationData(
        metadata=pd.DataFrame.from_records(metadata_records),
        whole_before=np.vstack(whole_before),
        previous_full=np.vstack(previous_full),
        current_full=np.vstack(current_full),
        previous_svd=np.vstack(previous_svd),
        current_svd=np.vstack(current_svd),
        behavior=np.vstack(behavior),
        targets=np.stack(targets),
    )
    order = (
        result.metadata.assign(_position=np.arange(len(result)))
        .sort_values(["cohort", "mouse", "area"])["_position"]
        .to_numpy(dtype=int)
    )
    result = result.subset(order)
    _validate_population_data(result)
    return result


@dataclass(frozen=True)
class BoundaryCandidate:
    """One complete forecaster candidate."""

    feature_variant: FeatureVariant | Literal["persistence", "mean_shift"]
    alpha: float | None

    @property
    def name(self) -> str:
        if self.feature_variant in ("persistence", "mean_shift"):
            return self.feature_variant
        return f"{self.feature_variant}__alpha_{float(self.alpha):g}"

    @property
    def complexity(self) -> tuple[float, ...]:
        if self.feature_variant == "persistence":
            return (0.0, 0.0, 0.0)
        if self.feature_variant == "mean_shift":
            return (1.0, 0.0, 0.0)
        feature_count = {
            "session_only": 6,
            "dual_full": 18,
            "dual_full_svd": 30,
            "dual_full_svd_behavior": 32,
        }[self.feature_variant]
        return (
            2.0,
            float(feature_count),
            -math.log10(float(self.alpha)),
        )

    def __post_init__(self) -> None:
        if self.feature_variant in ("persistence", "mean_shift"):
            if self.alpha is not None:
                raise BoundaryPopulationError(
                    f"{self.feature_variant} candidate must not define alpha"
                )
            return
        if self.feature_variant not in FEATURE_VARIANTS:
            raise BoundaryPopulationError(
                f"unknown feature variant: {self.feature_variant}"
            )
        if (
            self.alpha is None
            or not np.isfinite(float(self.alpha))
            or float(self.alpha) <= 0
        ):
            raise BoundaryPopulationError(
                "ridge candidate alpha must be finite and positive"
            )


def candidate_grid(
    *,
    variants: Sequence[FeatureVariant] = FEATURE_VARIANTS,
    alphas: Sequence[float] = DEFAULT_ALPHAS,
    include_persistence: bool = True,
    include_mean_shift: bool = True,
) -> tuple[BoundaryCandidate, ...]:
    """Return deterministic baselines and ridge candidates."""

    variants = tuple(variants)
    alphas = tuple(float(alpha) for alpha in alphas)
    if not variants or len(set(variants)) != len(variants):
        raise BoundaryPopulationError(
            "feature variants must be non-empty and unique"
        )
    if not alphas or len(set(alphas)) != len(alphas):
        raise BoundaryPopulationError("alphas must be non-empty and unique")
    result: list[BoundaryCandidate] = []
    if include_persistence:
        result.append(BoundaryCandidate("persistence", None))
    if include_mean_shift:
        result.append(BoundaryCandidate("mean_shift", None))
    result.extend(
        BoundaryCandidate(variant, alpha)
        for variant in variants
        for alpha in alphas
    )
    if len({candidate.name for candidate in result}) != len(result):
        raise BoundaryPopulationError("candidate names are not unique")
    return tuple(result)


@dataclass
class FittedPersistence:
    """Exact horizon-specific persistence candidate."""

    candidate: BoundaryCandidate
    horizon: int
    feature_names: tuple[str, ...] = ()

    def predict(
        self,
        data: BoundaryPopulationData,
        indices: Sequence[int] | np.ndarray,
    ) -> np.ndarray:
        selected = _validate_indices(data, indices, name="prediction")
        return np.asarray(
            data.persistence[selected],
            dtype=float,
        )


@dataclass
class FittedMeanShift:
    """Exact mean transformed before-to-future shift for one horizon."""

    candidate: BoundaryCandidate
    horizon: int
    transformed_delta: np.ndarray
    feature_names: tuple[str, ...] = ()

    def predict(
        self,
        data: BoundaryPopulationData,
        indices: Sequence[int] | np.ndarray,
    ) -> np.ndarray:
        selected = _validate_indices(data, indices, name="prediction")
        transform = DistributionTargetTransform(METRICS)
        anchor = transform.transform(data.persistence[selected])
        prediction = transform.inverse_transform(
            anchor + self.transformed_delta[None, :]
        )
        transform.validate(prediction)
        return prediction


@dataclass
class FittedBoundaryPopulationForecaster:
    """Training-only preprocessing and six unpenalized-intercept ridge heads."""

    candidate: BoundaryCandidate
    horizon: int
    feature_names: tuple[str, ...]
    fill_values: np.ndarray
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    response_mean: np.ndarray
    response_scale: np.ndarray
    heads: tuple[Ridge, ...]

    def predict(
        self,
        data: BoundaryPopulationData,
        indices: Sequence[int] | np.ndarray,
    ) -> np.ndarray:
        selected = _validate_indices(data, indices, name="prediction")
        if self.candidate.feature_variant == "persistence":
            raise RuntimeError("ridge forecaster cannot have persistence variant")
        features, names = data.feature_matrix(self.candidate.feature_variant)
        if names != self.feature_names:
            raise BoundaryPopulationError(
                "prediction features differ from fitted feature schema"
            )
        values = _fill_features(features[selected], self.fill_values)
        standardized = (values - self.feature_mean) / self.feature_scale
        residual = np.column_stack(
            [
                head.predict(standardized)
                * self.response_scale[head_index]
                + self.response_mean[head_index]
                for head_index, head in enumerate(self.heads)
            ]
        )
        transform = DistributionTargetTransform(METRICS)
        anchor = transform.transform(data.persistence[selected])
        prediction = transform.inverse_transform(anchor + residual)
        transform.validate(prediction)
        return prediction


FittedBoundaryModel = (
    FittedPersistence
    | FittedMeanShift
    | FittedBoundaryPopulationForecaster
)


def _feature_fill(values: np.ndarray) -> np.ndarray:
    fill = np.zeros(values.shape[1], dtype=float)
    for column in range(values.shape[1]):
        finite = np.isfinite(values[:, column])
        if finite.any():
            fill[column] = float(np.mean(values[finite, column]))
    return fill


def _fill_features(values: np.ndarray, fill: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=float).copy()
    invalid = ~np.isfinite(result)
    if invalid.any():
        result[invalid] = np.broadcast_to(fill, result.shape)[invalid]
    return result


def fit_boundary_candidate(
    data: BoundaryPopulationData,
    indices: Sequence[int] | np.ndarray,
    candidate: BoundaryCandidate,
    *,
    horizon: int,
) -> FittedBoundaryModel:
    """Fit one horizon-specific candidate without reading held-out targets."""

    _validate_population_data(data)
    selected = _validate_indices(data, indices, name="training")
    if horizon not in HORIZONS:
        raise BoundaryPopulationError(f"unknown horizon: {horizon}")
    if candidate.feature_variant == "persistence":
        return FittedPersistence(candidate=candidate, horizon=horizon)

    transform = DistributionTargetTransform(METRICS)
    anchor = transform.transform(data.persistence[selected])
    target = transform.transform(data.targets[selected, horizon, :])
    response = target - anchor
    if candidate.feature_variant == "mean_shift":
        return FittedMeanShift(
            candidate=candidate,
            horizon=horizon,
            transformed_delta=np.mean(response, axis=0),
        )

    features, names = data.feature_matrix(candidate.feature_variant)
    training_features = features[selected]
    fill = _feature_fill(training_features)
    training_features = _fill_features(training_features, fill)
    feature_mean = np.mean(training_features, axis=0)
    feature_scale = np.std(training_features, axis=0, ddof=0)
    feature_scale = np.where(
        np.isfinite(feature_scale) & (feature_scale > 1e-10),
        feature_scale,
        1.0,
    )
    standardized = (training_features - feature_mean) / feature_scale
    response_mean = np.mean(response, axis=0)
    response_scale = np.std(response, axis=0, ddof=0)
    response_scale = np.where(
        np.isfinite(response_scale) & (response_scale > 1e-10),
        response_scale,
        1.0,
    )

    heads: list[Ridge] = []
    for head in range(len(METRICS)):
        standardized_response = (
            response[:, head] - response_mean[head]
        ) / response_scale[head]
        ridge = Ridge(
            alpha=float(candidate.alpha),
            fit_intercept=True,
        ).fit(
            standardized,
            standardized_response,
        )
        heads.append(ridge)
    return FittedBoundaryPopulationForecaster(
        candidate=candidate,
        horizon=horizon,
        feature_names=names,
        fill_values=fill,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        response_mean=response_mean,
        response_scale=response_scale,
        heads=tuple(heads),
    )


def raw_output_scale(
    data: BoundaryPopulationData,
    indices: Sequence[int] | np.ndarray,
    *,
    horizon: int,
) -> np.ndarray:
    """Training-only raw-unit scale for one horizon."""

    selected = _validate_indices(data, indices, name="scale")
    if horizon not in HORIZONS:
        raise BoundaryPopulationError(f"unknown horizon: {horizon}")
    values = data.targets[selected, horizon, :]
    scale = np.std(values, axis=0, ddof=1)
    return np.where(np.isfinite(scale) & (scale > 1e-8), scale, 1.0)


def raw_six_output_nrmse(
    observed: np.ndarray,
    predicted: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    """Return equal-weight raw six-output NRMSE for each mouse."""

    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    scale = np.asarray(scale, dtype=float)
    if observed.ndim == 1:
        observed = observed[None, ...]
    if predicted.ndim == 1:
        predicted = predicted[None, ...]
    if (
        observed.shape != predicted.shape
        or observed.ndim != 2
        or observed.shape[1] != len(METRICS)
        or scale.shape != (len(METRICS),)
    ):
        raise BoundaryPopulationError(
            "observed, predicted, or scale has an invalid score shape"
        )
    if not np.isfinite(scale).all() or np.any(scale <= 0):
        raise BoundaryPopulationError("score scale must be finite and positive")
    scaled = (predicted - observed) / scale[None, :]
    return np.sqrt(np.mean(np.square(scaled), axis=1))


@dataclass(frozen=True)
class BoundarySelection:
    """One inner-LOMO candidate selection."""

    candidate: BoundaryCandidate
    best_candidate: str
    best_mean_nrmse: float
    best_sem_nrmse: float
    threshold_nrmse: float
    candidate_summary: pd.DataFrame


def select_boundary_candidate(
    scores: pd.DataFrame,
    candidates: Sequence[BoundaryCandidate],
    *,
    rule: SelectionRule = "minimum_cv",
) -> BoundarySelection:
    """Select by final population NRMSE, with deterministic simplicity ties."""

    if scores.empty:
        raise BoundaryPopulationError("candidate scores must not be empty")
    by_name = {candidate.name: candidate for candidate in candidates}
    if len(by_name) != len(candidates):
        raise BoundaryPopulationError("candidate names must be unique")
    unknown = set(scores["candidate"].astype(str)) - set(by_name)
    if unknown:
        raise BoundaryPopulationError(
            f"scores contain unknown candidates: {sorted(unknown)}"
        )
    summary = (
        scores.groupby("candidate", as_index=False)
        .agg(
            mice=("validation_mouse", "nunique"),
            mean_nrmse=("nrmse", "mean"),
            sem_nrmse=("nrmse", "sem"),
        )
        .fillna({"sem_nrmse": 0.0})
    )
    summary["complexity"] = summary["candidate"].map(
        lambda name: by_name[str(name)].complexity
    )
    best = summary.sort_values(["mean_nrmse", "candidate"]).iloc[0]
    if rule == "one_se":
        threshold = float(best["mean_nrmse"] + best["sem_nrmse"])
        eligible = set(
            summary.loc[
                summary["mean_nrmse"].le(threshold + 1e-15),
                "candidate",
            ].astype(str)
        )
        selected = min(
            (by_name[name] for name in eligible),
            key=lambda candidate: (candidate.complexity, candidate.name),
        )
    elif rule == "minimum_cv":
        threshold = float(best["mean_nrmse"])
        best_mean = float(best["mean_nrmse"])
        eligible = set(
            summary.loc[
                np.isclose(
                    summary["mean_nrmse"].to_numpy(dtype=float),
                    best_mean,
                    rtol=0,
                    atol=1e-15,
                ),
                "candidate",
            ].astype(str)
        )
        selected = min(
            (by_name[name] for name in eligible),
            key=lambda candidate: (candidate.complexity, candidate.name),
        )
    else:
        raise BoundaryPopulationError(f"unknown selection rule: {rule}")
    summary["eligible"] = summary["candidate"].astype(str).isin(eligible)
    summary["selected"] = summary["candidate"].eq(selected.name)
    return BoundarySelection(
        candidate=selected,
        best_candidate=str(best["candidate"]),
        best_mean_nrmse=float(best["mean_nrmse"]),
        best_sem_nrmse=float(best["sem_nrmse"]),
        threshold_nrmse=threshold,
        candidate_summary=summary.sort_values(
            ["mean_nrmse", "candidate"]
        ).reset_index(drop=True),
    )


def _candidate_lomo_scores(
    data: BoundaryPopulationData,
    pool: np.ndarray,
    candidates: Sequence[BoundaryCandidate],
    *,
    area: str,
    horizon: int,
    scope: str,
) -> pd.DataFrame:
    pool = _validate_indices(data, pool, name="candidate pool")
    mice = data.metadata.iloc[pool]["mouse"].astype(str).to_numpy()
    if len(np.unique(mice)) != len(pool):
        raise BoundaryPopulationError(
            "candidate pool must have one example per mouse"
        )
    if len(pool) < 3:
        raise BoundaryPopulationError(
            "candidate LOMO requires at least three mice"
        )
    records: list[dict[str, object]] = []
    for validation_position in pool:
        train = pool[pool != validation_position]
        scale = raw_output_scale(data, train, horizon=horizon)
        validation_mouse = str(
            data.metadata.iloc[int(validation_position)]["mouse"]
        )
        for candidate in candidates:
            fitted = fit_boundary_candidate(
                data,
                train,
                candidate,
                horizon=horizon,
            )
            prediction = fitted.predict(data, [validation_position])
            nrmse = raw_six_output_nrmse(
                data.targets[[validation_position], horizon, :],
                prediction,
                scale,
            )
            records.append(
                {
                    "area": area,
                    "horizon": int(horizon),
                    "scope": scope,
                    "validation_mouse": validation_mouse,
                    "candidate": candidate.name,
                    "feature_variant": candidate.feature_variant,
                    "alpha": candidate.alpha,
                    "nrmse": float(nrmse[0]),
                }
            )
    return pd.DataFrame.from_records(records)


def _selection_record(
    selection: BoundarySelection,
    *,
    area: str,
    horizon: int,
    scope: str,
    outer_mouse: str | None,
    rule: SelectionRule,
) -> dict[str, object]:
    return {
        "area": area,
        "horizon": int(horizon),
        "scope": scope,
        "outer_mouse": outer_mouse,
        "selection_rule": rule,
        "selected_candidate": selection.candidate.name,
        "selected_feature_variant": selection.candidate.feature_variant,
        "selected_alpha": selection.candidate.alpha,
        "best_candidate": selection.best_candidate,
        "best_mean_nrmse": selection.best_mean_nrmse,
        "best_sem_nrmse": selection.best_sem_nrmse,
        "selection_threshold_nrmse": selection.threshold_nrmse,
    }


def _prediction_records(
    data: BoundaryPopulationData,
    indices: np.ndarray,
    prediction: np.ndarray,
    *,
    horizon: int,
    split: str,
    fold: str,
    model: str,
    candidate: BoundaryCandidate,
    scale: np.ndarray,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    prediction_records: list[dict[str, object]] = []
    score_records: list[dict[str, object]] = []
    observed_values = data.targets[indices, horizon, :]
    scores = raw_six_output_nrmse(
        observed_values,
        prediction,
        scale,
    )
    for position, data_index in enumerate(indices):
        row = data.metadata.iloc[int(data_index)]
        score_records.append(
            {
                "split": split,
                "fold": fold,
                "model": model,
                "candidate": candidate.name,
                "feature_variant": candidate.feature_variant,
                "alpha": candidate.alpha,
                "mouse": str(row["mouse"]),
                "cohort": str(row["cohort"]),
                "area": str(row["area"]),
                "horizon": int(horizon),
                "nrmse": float(scores[position]),
            }
        )
        for metric_position, metric in enumerate(METRICS):
            observed = float(observed_values[position, metric_position])
            predicted = float(prediction[position, metric_position])
            residual = observed - predicted
            scaled_residual = residual / float(scale[metric_position])
            prediction_records.append(
                {
                    "split": split,
                    "fold": fold,
                    "model": model,
                    "candidate": candidate.name,
                    "feature_variant": candidate.feature_variant,
                    "alpha": candidate.alpha,
                    "mouse": str(row["mouse"]),
                    "cohort": str(row["cohort"]),
                    "area": str(row["area"]),
                    "horizon": int(horizon),
                    "metric": metric,
                    "observed": observed,
                    "predicted": predicted,
                    "residual": residual,
                    "absolute_error": abs(residual),
                    "scale": float(scale[metric_position]),
                    "scaled_absolute_error": abs(scaled_residual),
                    "scaled_squared_error": scaled_residual**2,
                }
            )
    return prediction_records, score_records


@dataclass
class BoundaryPopulationEvaluation:
    """Auditable nested development and frozen transfer outputs."""

    candidate_scores: pd.DataFrame
    candidate_summary: pd.DataFrame
    selections: pd.DataFrame
    predictions: pd.DataFrame
    mouse_scores: pd.DataFrame
    final_models: dict[tuple[str, int], FittedBoundaryModel]


def evaluate_boundary_population_forecaster(
    data: BoundaryPopulationData,
    *,
    candidates: Sequence[BoundaryCandidate] | None = None,
    areas: Sequence[str] | None = None,
    training_cohort: str = "unsupervised",
    transfer_cohort: str = "supervised",
    selection_rule: SelectionRule = "minimum_cv",
) -> BoundaryPopulationEvaluation:
    """Run nested mouse LOMO and one frozen transfer evaluation."""

    _validate_population_data(data)
    if training_cohort == transfer_cohort:
        raise BoundaryPopulationError(
            "training and transfer cohorts must be different"
        )
    candidates = tuple(candidates or candidate_grid())
    if not candidates:
        raise BoundaryPopulationError("at least one candidate is required")
    if len({candidate.name for candidate in candidates}) != len(candidates):
        raise BoundaryPopulationError("candidate names must be unique")
    available_areas = tuple(sorted(data.metadata["area"].astype(str).unique()))
    areas = tuple(areas or available_areas)
    if set(areas) - set(available_areas):
        raise BoundaryPopulationError("requested area is absent from data")

    candidate_frames: list[pd.DataFrame] = []
    summary_frames: list[pd.DataFrame] = []
    selection_records: list[dict[str, object]] = []
    prediction_records: list[dict[str, object]] = []
    score_records: list[dict[str, object]] = []
    final_models: dict[tuple[str, int], FittedBoundaryModel] = {}
    persistence_candidate = BoundaryCandidate("persistence", None)

    for area in areas:
        training = np.flatnonzero(
            (
                data.metadata["area"].astype(str).eq(area)
                & data.metadata["cohort"].astype(str).eq(training_cohort)
            ).to_numpy()
        )
        transfer = np.flatnonzero(
            (
                data.metadata["area"].astype(str).eq(area)
                & data.metadata["cohort"].astype(str).eq(transfer_cohort)
            ).to_numpy()
        )
        if len(training) < 4:
            raise BoundaryPopulationError(
                f"{area} nested LOMO needs at least four training mice"
            )
        if len(transfer) == 0:
            raise BoundaryPopulationError(f"{area} has no transfer mice")
        if (
            data.metadata.iloc[training]["mouse"].astype(str).nunique()
            != len(training)
        ):
            raise BoundaryPopulationError(
                f"{area} training data is not one row per mouse"
            )

        for horizon in HORIZONS:
            for validation_index in training:
                outer_mouse = str(
                    data.metadata.iloc[int(validation_index)]["mouse"]
                )
                outer_train = training[training != validation_index]
                scope = f"outer:{outer_mouse}"
                inner = _candidate_lomo_scores(
                    data,
                    outer_train,
                    candidates,
                    area=area,
                    horizon=horizon,
                    scope=scope,
                )
                candidate_frames.append(inner)
                selection = select_boundary_candidate(
                    inner,
                    candidates,
                    rule=selection_rule,
                )
                current_summary = selection.candidate_summary.copy()
                current_summary.insert(0, "area", area)
                current_summary.insert(1, "horizon", int(horizon))
                current_summary.insert(2, "scope", scope)
                current_summary.insert(3, "outer_mouse", outer_mouse)
                summary_frames.append(current_summary)
                selection_records.append(
                    _selection_record(
                        selection,
                        area=area,
                        horizon=horizon,
                        scope=scope,
                        outer_mouse=outer_mouse,
                        rule=selection_rule,
                    )
                )
                fitted = fit_boundary_candidate(
                    data,
                    outer_train,
                    selection.candidate,
                    horizon=horizon,
                )
                scale = raw_output_scale(
                    data,
                    outer_train,
                    horizon=horizon,
                )
                validation = np.asarray([validation_index], dtype=int)
                learned = fitted.predict(data, validation)
                rows, scores = _prediction_records(
                    data,
                    validation,
                    learned,
                    horizon=horizon,
                    split="unsupervised_lomo",
                    fold=outer_mouse,
                    model="selected_nested",
                    candidate=selection.candidate,
                    scale=scale,
                )
                prediction_records.extend(rows)
                score_records.extend(scores)
                persistence = FittedPersistence(
                    candidate=persistence_candidate,
                    horizon=horizon,
                ).predict(data, validation)
                rows, scores = _prediction_records(
                    data,
                    validation,
                    persistence,
                    horizon=horizon,
                    split="unsupervised_lomo",
                    fold=outer_mouse,
                    model="persistence",
                    candidate=persistence_candidate,
                    scale=scale,
                )
                prediction_records.extend(rows)
                score_records.extend(scores)

            final_scope = "all_unsupervised"
            final_scores = _candidate_lomo_scores(
                data,
                training,
                candidates,
                area=area,
                horizon=horizon,
                scope=final_scope,
            )
            candidate_frames.append(final_scores)
            final_selection = select_boundary_candidate(
                final_scores,
                candidates,
                rule=selection_rule,
            )
            current_summary = final_selection.candidate_summary.copy()
            current_summary.insert(0, "area", area)
            current_summary.insert(1, "horizon", int(horizon))
            current_summary.insert(2, "scope", final_scope)
            current_summary.insert(3, "outer_mouse", None)
            summary_frames.append(current_summary)
            selection_records.append(
                _selection_record(
                    final_selection,
                    area=area,
                    horizon=horizon,
                    scope=final_scope,
                    outer_mouse=None,
                    rule=selection_rule,
                )
            )
            fitted = fit_boundary_candidate(
                data,
                training,
                final_selection.candidate,
                horizon=horizon,
            )
            final_models[(area, horizon)] = fitted
            scale = raw_output_scale(data, training, horizon=horizon)
            learned = fitted.predict(data, transfer)
            rows, scores = _prediction_records(
                data,
                transfer,
                learned,
                horizon=horizon,
                split="supervised_test",
                fold="frozen",
                model="selected_nested",
                candidate=final_selection.candidate,
                scale=scale,
            )
            prediction_records.extend(rows)
            score_records.extend(scores)
            persistence = FittedPersistence(
                candidate=persistence_candidate,
                horizon=horizon,
            ).predict(data, transfer)
            rows, scores = _prediction_records(
                data,
                transfer,
                persistence,
                horizon=horizon,
                split="supervised_test",
                fold="frozen",
                model="persistence",
                candidate=persistence_candidate,
                scale=scale,
            )
            prediction_records.extend(rows)
            score_records.extend(scores)

    return BoundaryPopulationEvaluation(
        candidate_scores=pd.concat(candidate_frames, ignore_index=True),
        candidate_summary=pd.concat(summary_frames, ignore_index=True),
        selections=pd.DataFrame.from_records(selection_records),
        predictions=pd.DataFrame.from_records(prediction_records),
        mouse_scores=pd.DataFrame.from_records(score_records),
        final_models=final_models,
    )


__all__ = [
    "BEHAVIOR_NAMES",
    "DEFAULT_ALPHAS",
    "FEATURE_VARIANTS",
    "HORIZONS",
    "METRICS",
    "TRANSFORMED_METRIC_NAMES",
    "BoundaryCandidate",
    "BoundaryPopulationData",
    "BoundaryPopulationError",
    "BoundaryPopulationEvaluation",
    "BoundarySelection",
    "FeatureVariant",
    "FittedBoundaryModel",
    "FittedBoundaryPopulationForecaster",
    "FittedMeanShift",
    "FittedPersistence",
    "SelectionRule",
    "build_boundary_population_data",
    "candidate_grid",
    "evaluate_boundary_population_forecaster",
    "fit_boundary_candidate",
    "raw_output_scale",
    "raw_six_output_nrmse",
    "select_boundary_candidate",
]
