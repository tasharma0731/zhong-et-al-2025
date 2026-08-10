"""Final Objective B evaluation utilities.

This module finalizes the supported next-window forecaster without changing
the stable models or the earlier experiment branches.  The architecture is
locked to the per-neuron, dual-W20 full-plus-SVD ridge.  Nested mouse-level
validation selects only the ridge penalty.

The phrase "full plus SVD" is intentionally precise here.  For one focal
neuron, each representation contributes three local response statistics at
W20(t-1) and W20(t): d-prime, the d-prime numerator, and log denominator.
The model therefore receives 12 scalar predictors, not a complete population
trace or a 1,000-neuron vector.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from modeling.experiments.lag2_full_neural_objective_b import (
    equal_full_average_candidate,
    lag2_candidate,
)
from modeling.experiments.refined_neuron_validation import (
    CandidateSpec,
    FROZEN_MODEL_NAME,
    K1_BASELINE_NAME,
    NESTED_MODEL_NAME,
    PERSISTENCE_NAME,
    RefinedNeuronEvaluation,
    stable_seed,
)


AREAS = ("mHV", "aHV")
ALPHAS = (100.0, 1_000.0, 10_000.0)
TWO_WINDOW_AVERAGE_NAME = "two_window_equal_average"
FINAL_MODEL_DISPLAY = "Dual-W20 full+SVD ridge"
MODEL_DISPLAY_NAMES = {
    NESTED_MODEL_NAME: FINAL_MODEL_DISPLAY,
    FROZEN_MODEL_NAME: FINAL_MODEL_DISPLAY,
    PERSISTENCE_NAME: "Persistence: d-prime(t)",
    K1_BASELINE_NAME: "Training mean shift",
    TWO_WINDOW_AVERAGE_NAME: "Two-window average",
}
FEATURE_NAMES = (
    "full_dprime_tminus1",
    "full_numerator_tminus1",
    "full_log_denominator_tminus1",
    "full_dprime_t",
    "full_numerator_t",
    "full_log_denominator_t",
    "svd_dprime_tminus1",
    "svd_numerator_tminus1",
    "svd_log_denominator_tminus1",
    "svd_dprime_t",
    "svd_numerator_t",
    "svd_log_denominator_t",
)
METRIC_COLUMNS = (
    "rmse_dprime",
    "mae_dprime",
    "bias_dprime",
    "pearson",
    "calibration_intercept",
    "calibration_slope",
    "absolute_error_q50",
    "absolute_error_q70",
    "absolute_error_q80",
    "fraction_within_0.3_dprime",
    "fraction_within_0.5_dprime",
    "mean_abs_actual_dprime",
    "actual_dprime_sd",
    "predicted_dprime_sd",
)


def final_candidate_grid(
    alphas: Sequence[float] = ALPHAS,
) -> tuple[CandidateSpec, ...]:
    """Return the locked architecture with a predeclared alpha grid."""

    values = tuple(float(alpha) for alpha in alphas)
    if not values or any(not np.isfinite(alpha) or alpha <= 0 for alpha in values):
        raise ValueError("alphas must be finite positive values")
    if len(set(values)) != len(values):
        raise ValueError("alphas must be unique")
    return tuple(
        lag2_candidate("hybrid_local", alpha=alpha)
        for alpha in values
    )


def selected_model_for_split(split: str) -> str:
    if split == "unsupervised_lomo":
        return NESTED_MODEL_NAME
    if split == "supervised_test":
        return FROZEN_MODEL_NAME
    raise ValueError(f"unknown evaluation split: {split}")


def display_split(split: str) -> str:
    labels = {
        "unsupervised_lomo": "Unsupervised nested LOMO",
        "supervised_test": "Frozen supervised transfer",
    }
    return labels.get(split, split)


def _safe_pearson(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    if valid.sum() < 2:
        return np.nan
    x = np.asarray(left[valid], dtype=float)
    y = np.asarray(right[valid], dtype=float)
    if np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def _calibration(
    predicted: np.ndarray,
    observed: np.ndarray,
) -> tuple[float, float]:
    """Fit observed = intercept + slope * predicted for diagnostics."""

    valid = np.isfinite(predicted) & np.isfinite(observed)
    if valid.sum() < 2:
        return np.nan, np.nan
    x = np.asarray(predicted[valid], dtype=float)
    y = np.asarray(observed[valid], dtype=float)
    variance = float(np.var(x))
    if variance <= 1e-12:
        return float(np.mean(y)), np.nan
    slope = float(np.mean((x - np.mean(x)) * (y - np.mean(y))) / variance)
    intercept = float(np.mean(y) - slope * np.mean(x))
    return intercept, slope


def _prediction_frame(
    data: object,
    indices: np.ndarray,
    prediction: np.ndarray,
    *,
    split: str,
    fold: str,
    model: str,
    selected_candidate: str = "",
) -> pd.DataFrame:
    """Create compact in-memory neuron rows for one prediction block."""

    frames: list[pd.DataFrame] = []
    current = np.asarray(data.current, dtype=float)
    future = np.asarray(data.future, dtype=float)
    for position, data_index in enumerate(np.asarray(indices, dtype=int)):
        predicted = np.asarray(prediction[position], dtype=float)
        valid = (
            np.isfinite(current[data_index])
            & np.isfinite(future[data_index])
            & np.isfinite(predicted)
        )
        neurons = np.flatnonzero(valid)
        if not len(neurons):
            continue
        metadata = data.metadata.iloc[int(data_index)]
        observed = future[data_index, neurons]
        values = predicted[neurons]
        frames.append(
            pd.DataFrame(
                {
                    "split": split,
                    "fold": fold,
                    "model": model,
                    "selected_candidate": selected_candidate,
                    "data_index": np.full(
                        len(neurons),
                        int(data_index),
                        dtype=np.int32,
                    ),
                    "neuron_position": neurons.astype(np.int16),
                    "mouse": str(metadata["mouse"]),
                    "cohort": str(metadata["cohort"]),
                    "area": str(metadata["area"]),
                    "behavior_session_id": str(
                        metadata["behavior_session_id"]
                    ),
                    "moment": str(metadata["moment"]),
                    "current": current[data_index, neurons].astype(np.float32),
                    "observed": observed.astype(np.float32),
                    "predicted": values.astype(np.float32),
                    "residual": (observed - values).astype(np.float32),
                }
            )
        )
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def two_window_average_predictions(data: object) -> pd.DataFrame:
    """Score the unlearned temporal average on the exact final folds."""

    metadata = data.metadata
    candidate = equal_full_average_candidate()
    fitted = candidate.fit(data, np.arange(len(metadata)), 0)
    frames: list[pd.DataFrame] = []
    for area in AREAS:
        unsupervised = np.flatnonzero(
            metadata["area"].astype(str).eq(area)
            & metadata["cohort"].astype(str).eq("unsupervised")
        )
        for mouse in sorted(
            metadata.iloc[unsupervised]["mouse"].astype(str).unique()
        ):
            validation = unsupervised[
                metadata.iloc[unsupervised]["mouse"]
                .astype(str)
                .eq(mouse)
                .to_numpy()
            ]
            prediction = candidate.predict(fitted, data, validation)
            frames.append(
                _prediction_frame(
                    data,
                    validation,
                    prediction,
                    split="unsupervised_lomo",
                    fold=mouse,
                    model=TWO_WINDOW_AVERAGE_NAME,
                )
            )

        supervised = np.flatnonzero(
            metadata["area"].astype(str).eq(area)
            & metadata["cohort"].astype(str).eq("supervised")
        )
        prediction = candidate.predict(fitted, data, supervised)
        frames.append(
            _prediction_frame(
                data,
                supervised,
                prediction,
                split="supervised_test",
                fold="frozen",
                model=TWO_WINDOW_AVERAGE_NAME,
            )
        )
    return pd.concat(frames, ignore_index=True)


def combine_evaluation_predictions(
    data: object,
    evaluation: RefinedNeuronEvaluation,
) -> pd.DataFrame:
    """Combine learned, standard-baseline, and temporal-average predictions."""

    nested = evaluation.nested_predictions.copy()
    transfer = evaluation.transfer_predictions.copy()
    average = two_window_average_predictions(data)
    result = pd.concat((nested, transfer, average), ignore_index=True)
    result["selected_candidate"] = (
        result["selected_candidate"].fillna("").astype(str)
    )
    result["display_model"] = result["model"].map(MODEL_DISPLAY_NAMES)
    if result["display_model"].isna().any():
        unknown = sorted(result.loc[result["display_model"].isna(), "model"].unique())
        raise ValueError(f"prediction table has unknown model names: {unknown}")
    return result


def transition_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Compute absolute metrics before any cross-transition averaging."""

    identity = [
        "split",
        "fold",
        "model",
        "display_model",
        "selected_candidate",
        "data_index",
        "mouse",
        "cohort",
        "area",
        "behavior_session_id",
        "moment",
    ]
    records: list[dict[str, object]] = []
    for key, group in predictions.groupby(identity, sort=False, dropna=False):
        observed = group["observed"].to_numpy(dtype=float)
        predicted = group["predicted"].to_numpy(dtype=float)
        error = predicted - observed
        absolute_error = np.abs(error)
        intercept, slope = _calibration(predicted, observed)
        record = dict(zip(identity, key, strict=True))
        record.update(
            {
                "n_predictions": int(len(group)),
                "rmse_dprime": float(np.sqrt(np.mean(error**2))),
                "mae_dprime": float(np.mean(absolute_error)),
                "bias_dprime": float(np.mean(error)),
                "pearson": _safe_pearson(predicted, observed),
                "calibration_intercept": intercept,
                "calibration_slope": slope,
                "absolute_error_q50": float(
                    np.quantile(absolute_error, 0.50)
                ),
                "absolute_error_q70": float(
                    np.quantile(absolute_error, 0.70)
                ),
                "absolute_error_q80": float(
                    np.quantile(absolute_error, 0.80)
                ),
                "fraction_within_0.3_dprime": float(
                    np.mean(absolute_error <= 0.3)
                ),
                "fraction_within_0.5_dprime": float(
                    np.mean(absolute_error <= 0.5)
                ),
                "mean_abs_actual_dprime": float(
                    np.mean(np.abs(observed))
                ),
                "actual_dprime_sd": float(np.std(observed, ddof=1)),
                "predicted_dprime_sd": float(np.std(predicted, ddof=1)),
            }
        )
        records.append(record)
    return pd.DataFrame.from_records(records)


def aggregate_metrics(
    transitions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Average transition metrics through session/moment and then mouse."""

    identity = [
        "split",
        "fold",
        "model",
        "display_model",
        "selected_candidate",
        "mouse",
        "cohort",
        "area",
    ]
    sessions = (
        transitions.groupby(
            identity + ["behavior_session_id", "moment"],
            as_index=False,
            dropna=False,
        )[list(METRIC_COLUMNS)]
        .mean()
    )
    mouse = (
        sessions.groupby(identity, as_index=False, dropna=False)[
            list(METRIC_COLUMNS)
        ]
        .mean()
        .sort_values(["split", "area", "model", "mouse"])
        .reset_index(drop=True)
    )
    summary = (
        mouse.groupby(
            ["split", "model", "display_model", "area"],
            as_index=False,
        )
        .agg(
            mice=("mouse", "nunique"),
            **{
                f"mean_{column}": (column, "mean")
                for column in METRIC_COLUMNS
            },
            **{
                f"sem_{column}": (column, "sem")
                for column in METRIC_COLUMNS
            },
        )
        .sort_values(["split", "area", "mean_rmse_dprime", "model"])
        .reset_index(drop=True)
    )
    return mouse, summary


def coverage_metrics(
    predictions: pd.DataFrame,
    *,
    thresholds: np.ndarray | None = None,
    included_models: Iterable[str] = (
        NESTED_MODEL_NAME,
        FROZEN_MODEL_NAME,
        PERSISTENCE_NAME,
        TWO_WINDOW_AVERAGE_NAME,
    ),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return mouse-balanced empirical absolute-error coverage curves."""

    if thresholds is None:
        thresholds = np.linspace(0.0, 1.5, 61)
    thresholds = np.asarray(thresholds, dtype=float)
    included = set(included_models)
    selected = predictions.loc[predictions["model"].isin(included)].copy()
    identity = [
        "split",
        "model",
        "display_model",
        "data_index",
        "mouse",
        "cohort",
        "area",
        "behavior_session_id",
        "moment",
    ]
    records: list[dict[str, object]] = []
    for key, group in selected.groupby(identity, sort=False):
        absolute_error = np.sort(
            np.abs(
                group["predicted"].to_numpy(dtype=float)
                - group["observed"].to_numpy(dtype=float)
            )
        )
        coverage = (
            np.searchsorted(absolute_error, thresholds, side="right")
            / float(len(absolute_error))
        )
        base = dict(zip(identity, key, strict=True))
        records.extend(
            {
                **base,
                "threshold_dprime": float(threshold),
                "fraction_within": float(fraction),
            }
            for threshold, fraction in zip(
                thresholds,
                coverage,
                strict=True,
            )
        )
    transition = pd.DataFrame.from_records(records)
    session = (
        transition.groupby(
            [
                "split",
                "model",
                "display_model",
                "mouse",
                "cohort",
                "area",
                "behavior_session_id",
                "moment",
                "threshold_dprime",
            ],
            as_index=False,
        )["fraction_within"]
        .mean()
    )
    mouse = (
        session.groupby(
            [
                "split",
                "model",
                "display_model",
                "mouse",
                "cohort",
                "area",
                "threshold_dprime",
            ],
            as_index=False,
        )["fraction_within"]
        .mean()
    )
    summary = (
        mouse.groupby(
            [
                "split",
                "model",
                "display_model",
                "area",
                "threshold_dprime",
            ],
            as_index=False,
        )
        .agg(
            mice=("mouse", "nunique"),
            mean_fraction_within=("fraction_within", "mean"),
            sem_fraction_within=("fraction_within", "sem"),
        )
    )
    return transition, mouse, summary


def _two_sided_sign_test(wins: int, losses: int) -> float:
    n = int(wins + losses)
    if n == 0:
        return np.nan
    tail = min(int(wins), int(losses))
    probability = 2.0 * sum(
        math.comb(n, k) for k in range(tail + 1)
    ) / (2.0**n)
    return float(min(1.0, probability))


def _bootstrap_mean_ci(
    values: np.ndarray,
    *,
    seed: int,
    repetitions: int = 20_000,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan
    generator = np.random.default_rng(seed)
    indices = generator.integers(
        0,
        len(values),
        size=(int(repetitions), len(values)),
    )
    means = values[indices].mean(axis=1)
    lower, upper = np.quantile(means, (0.025, 0.975))
    return float(lower), float(upper)


def paired_model_comparisons(
    mouse: pd.DataFrame,
    *,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare the fold-selected model with predeclared baselines by mouse."""

    paired_frames: list[pd.DataFrame] = []
    for split in ("unsupervised_lomo", "supervised_test"):
        challenger_name = selected_model_for_split(split)
        challenger = mouse.loc[
            mouse["split"].eq(split) & mouse["model"].eq(challenger_name)
        ]
        for baseline_name in (
            PERSISTENCE_NAME,
            TWO_WINDOW_AVERAGE_NAME,
            K1_BASELINE_NAME,
        ):
            baseline = mouse.loc[
                mouse["split"].eq(split) & mouse["model"].eq(baseline_name)
            ]
            keys = ["split", "mouse", "cohort", "area"]
            paired = challenger.merge(
                baseline,
                on=keys,
                how="inner",
                validate="one_to_one",
                suffixes=("_model", "_baseline"),
            )
            paired["baseline"] = baseline_name
            paired["baseline_display"] = MODEL_DISPLAY_NAMES[baseline_name]
            paired["rmse_difference"] = (
                paired["rmse_dprime_model"]
                - paired["rmse_dprime_baseline"]
            )
            paired["rmse_reduction_percent"] = 100.0 * (
                paired["rmse_dprime_baseline"]
                - paired["rmse_dprime_model"]
            ) / paired["rmse_dprime_baseline"]
            paired_frames.append(paired)
    paired_all = pd.concat(paired_frames, ignore_index=True)

    records: list[dict[str, object]] = []
    for key, group in paired_all.groupby(
        ["split", "area", "baseline", "baseline_display"],
        sort=True,
    ):
        split, area, baseline, baseline_display = key
        differences = group["rmse_difference"].to_numpy(dtype=float)
        reductions = group["rmse_reduction_percent"].to_numpy(dtype=float)
        wins = int(np.sum(differences < 0))
        losses = int(np.sum(differences > 0))
        model_rmse = float(group["rmse_dprime_model"].mean())
        baseline_rmse = float(group["rmse_dprime_baseline"].mean())
        lower, upper = _bootstrap_mean_ci(
            reductions,
            seed=stable_seed(seed, split, area, baseline),
        )
        records.append(
            {
                "split": split,
                "split_display": display_split(split),
                "area": area,
                "baseline": baseline,
                "baseline_display": baseline_display,
                "mice": int(group["mouse"].nunique()),
                "mean_model_rmse": model_rmse,
                "mean_baseline_rmse": baseline_rmse,
                "rmse_reduction_from_mean_percent": (
                    100.0
                    * (baseline_rmse - model_rmse)
                    / baseline_rmse
                ),
                "mean_paired_rmse_reduction_percent": float(
                    np.mean(reductions)
                ),
                "paired_reduction_ci95_lower": lower,
                "paired_reduction_ci95_upper": upper,
                "mice_model_better": wins,
                "mice_baseline_better": losses,
                "two_sided_sign_test_p": _two_sided_sign_test(wins, losses),
            }
        )
    return paired_all, pd.DataFrame.from_records(records)


def balanced_prediction_sample(
    predictions: pd.DataFrame,
    *,
    per_mouse_model: int,
    seed: int,
) -> pd.DataFrame:
    """Sample equally within mouse/model for plotting and human inspection."""

    frames: list[pd.DataFrame] = []
    keys = ["split", "area", "mouse", "model"]
    for key, group in predictions.groupby(keys, sort=True):
        count = min(int(per_mouse_model), len(group))
        if count == len(group):
            frames.append(group.copy())
            continue
        frames.append(
            group.sample(
                n=count,
                replace=False,
                random_state=stable_seed(seed, "plot-sample", *key),
            )
        )
    return pd.concat(frames, ignore_index=True)


def save_compact_predictions(
    predictions: pd.DataFrame,
    *,
    npz_path: Path,
    codes_path: Path,
) -> None:
    """Save long neuron predictions once, using categorical codes + float32."""

    category_columns = (
        "split",
        "fold",
        "model",
        "selected_candidate",
    )
    code_arrays: dict[str, np.ndarray] = {}
    mappings: dict[str, list[str]] = {}
    for column in category_columns:
        categorical = pd.Categorical(predictions[column].fillna("").astype(str))
        if (categorical.codes < 0).any():
            raise ValueError(f"{column} contains an unencodable category")
        code_arrays[f"{column}_code"] = categorical.codes.astype(np.int16)
        mappings[column] = [str(value) for value in categorical.categories]
    np.savez_compressed(
        npz_path,
        **code_arrays,
        data_index=predictions["data_index"].to_numpy(dtype=np.int16),
        neuron_position=predictions["neuron_position"].to_numpy(
            dtype=np.int16
        ),
        current=predictions["current"].to_numpy(dtype=np.float32),
        observed=predictions["observed"].to_numpy(dtype=np.float32),
        predicted=predictions["predicted"].to_numpy(dtype=np.float32),
        residual=predictions["residual"].to_numpy(dtype=np.float32),
    )
    codes_path.write_text(
        json.dumps(mappings, indent=2, sort_keys=True) + "\n"
    )


def parse_alpha(candidate_name: str) -> float:
    match = re.search(r"__alpha_([0-9.eE+-]+)$", str(candidate_name))
    if match is None:
        raise ValueError(f"candidate name has no alpha suffix: {candidate_name}")
    return float(match.group(1))


def selection_audit(selections: pd.DataFrame) -> pd.DataFrame:
    """Add numeric alpha columns to the nested-selection audit table."""

    result = selections.copy()
    result["selected_alpha"] = result["selected_candidate"].map(parse_alpha)
    result["best_alpha"] = result["best_candidate"].map(parse_alpha)
    return result


def frozen_coefficient_table(
    evaluation: RefinedNeuronEvaluation,
) -> pd.DataFrame:
    """Export fitted coefficients in both standardized and original units."""

    records: list[dict[str, object]] = []
    for (area, model_name), fitted in sorted(evaluation.final_models.items()):
        if model_name != FROZEN_MODEL_NAME:
            continue
        model = fitted.model
        coefficients = np.asarray(model.ridge.coef_, dtype=float).reshape(-1)
        if len(coefficients) != len(FEATURE_NAMES):
            raise ValueError(
                f"{area} frozen model has {len(coefficients)} coefficients; "
                f"expected {len(FEATURE_NAMES)}"
            )
        target_scale = float(model.target_scaler.scale_[0])
        feature_scale = np.asarray(
            model.feature_scaler.scale_,
            dtype=float,
        )
        original = target_scale * coefficients / feature_scale
        for position, feature in enumerate(FEATURE_NAMES):
            records.append(
                {
                    "area": area,
                    "selected_candidate": fitted.candidate.name,
                    "alpha": parse_alpha(fitted.candidate.name),
                    "feature_position": position,
                    "feature": feature,
                    "standardized_coefficient": float(
                        coefficients[position]
                    ),
                    "original_delta_dprime_coefficient": float(
                        original[position]
                    ),
                    "training_fill_value": float(
                        model.fill_values[position]
                    ),
                    "training_feature_mean": float(
                        model.feature_scaler.mean_[position]
                    ),
                    "training_feature_scale": float(feature_scale[position]),
                }
            )
    return pd.DataFrame.from_records(records)


@dataclass(frozen=True)
class FinalArtifacts:
    predictions: pd.DataFrame
    transition_metrics: pd.DataFrame
    mouse_metrics: pd.DataFrame
    metric_summary: pd.DataFrame
    coverage_transition: pd.DataFrame
    coverage_mouse: pd.DataFrame
    coverage_summary: pd.DataFrame
    paired_mouse: pd.DataFrame
    paired_summary: pd.DataFrame
    plot_sample: pd.DataFrame
    selections: pd.DataFrame
    coefficients: pd.DataFrame


def derive_final_artifacts(
    data: object,
    evaluation: RefinedNeuronEvaluation,
    *,
    seed: int,
    plot_sample_per_mouse_model: int = 1_000,
) -> FinalArtifacts:
    predictions = combine_evaluation_predictions(data, evaluation)
    transitions = transition_metrics(predictions)
    mouse, summary = aggregate_metrics(transitions)
    coverage_transition, coverage_mouse, coverage_summary = coverage_metrics(
        predictions
    )
    paired_mouse, paired_summary = paired_model_comparisons(mouse, seed=seed)
    plot_sample = balanced_prediction_sample(
        predictions,
        per_mouse_model=plot_sample_per_mouse_model,
        seed=seed,
    )
    return FinalArtifacts(
        predictions=predictions,
        transition_metrics=transitions,
        mouse_metrics=mouse,
        metric_summary=summary,
        coverage_transition=coverage_transition,
        coverage_mouse=coverage_mouse,
        coverage_summary=coverage_summary,
        paired_mouse=paired_mouse,
        paired_summary=paired_summary,
        plot_sample=plot_sample,
        selections=selection_audit(evaluation.selections),
        coefficients=frozen_coefficient_table(evaluation),
    )


__all__ = [
    "ALPHAS",
    "AREAS",
    "FEATURE_NAMES",
    "FINAL_MODEL_DISPLAY",
    "FinalArtifacts",
    "MODEL_DISPLAY_NAMES",
    "TWO_WINDOW_AVERAGE_NAME",
    "aggregate_metrics",
    "balanced_prediction_sample",
    "combine_evaluation_predictions",
    "coverage_metrics",
    "derive_final_artifacts",
    "display_split",
    "final_candidate_grid",
    "frozen_coefficient_table",
    "paired_model_comparisons",
    "parse_alpha",
    "save_compact_predictions",
    "selected_model_for_split",
    "selection_audit",
    "transition_metrics",
    "two_window_average_predictions",
]
