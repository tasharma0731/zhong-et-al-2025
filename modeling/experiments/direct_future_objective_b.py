"""Factorial Objective B experiment: target parameterization × alpha rule.

This module is deliberately separate from the finalized Objective B pipeline.
It compares four procedures while holding the dual-W20, full-derived-plus-SVD
12-feature architecture fixed:

``future_minus_current`` versus ``future_level``
    Predict either the next-minus-current d-prime change and add the current
    state back, or predict the next d-prime level directly.

``one_se`` versus ``minimum_cv``
    Select the strongest regularization within one standard error of the
    inner-LOMO winner, or select the alpha with the smallest inner-LOMO
    mouse-level NRMSE.

The current finalized model remains the locked ``delta__one_se`` reference.
The sole predeclared challenger is ``direct__minimum_cv``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Literal, Sequence
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from modeling.experiments.full_neural_objective_b import (
    _fill,
    _weighted_fill,
)
from modeling.experiments.lag2_full_neural_objective_b import (
    _lag_design,
    equal_full_average_candidate,
    lag2_candidate,
)
from modeling.experiments.objective_b_final import (
    _prediction_frame,
    aggregate_metrics,
    balanced_prediction_sample,
    coverage_metrics,
    transition_metrics,
)
from modeling.experiments.refined_neuron_objective_b import (
    hierarchical_row_weights,
)
from modeling.experiments.refined_neuron_validation import (
    CandidateSpec,
    FittedCandidate,
    OneSESelection,
    _candidate_lomo,
    _fit_predict,
    select_one_se,
    stable_seed,
)


TargetParameterization = Literal["future_minus_current", "future_level"]
SelectionRule = Literal["one_se", "minimum_cv"]

AREAS = ("mHV", "aHV")
ALPHAS = (100.0, 1_000.0, 10_000.0)
LOCKED_REFERENCE = "delta__one_se"
PRIMARY_CHALLENGER = "direct__minimum_cv"
PERSISTENCE = "persistence"
TWO_WINDOW_AVERAGE = "two_window_average"


@dataclass(frozen=True)
class Procedure:
    """One predeclared cell of the target × selection-rule experiment."""

    procedure: str
    target_parameterization: TargetParameterization
    selection_rule: SelectionRule
    display_name: str


PROCEDURES = (
    Procedure(
        procedure=LOCKED_REFERENCE,
        target_parameterization="future_minus_current",
        selection_rule="one_se",
        display_name="Delta target + one-SE",
    ),
    Procedure(
        procedure="delta__minimum_cv",
        target_parameterization="future_minus_current",
        selection_rule="minimum_cv",
        display_name="Delta target + minimum CV",
    ),
    Procedure(
        procedure="direct__one_se",
        target_parameterization="future_level",
        selection_rule="one_se",
        display_name="Direct future + one-SE",
    ),
    Procedure(
        procedure=PRIMARY_CHALLENGER,
        target_parameterization="future_level",
        selection_rule="minimum_cv",
        display_name="Direct future + minimum CV",
    ),
)
PROCEDURE_BY_ID = {item.procedure: item for item in PROCEDURES}
DISPLAY_NAMES = {
    **{item.procedure: item.display_name for item in PROCEDURES},
    PERSISTENCE: "Persistence: d-prime(t)",
    TWO_WINDOW_AVERAGE: "Two-window average",
}


@dataclass
class FittedDirectFutureRidge:
    """Training-fold preprocessing and direct future-level ridge."""

    fill_values: np.ndarray
    feature_scaler: StandardScaler
    target_scaler: StandardScaler
    ridge: Ridge

    def predict(
        self,
        data: object,
        indices: Sequence[int] | np.ndarray,
    ) -> np.ndarray:
        design = _lag_design(data, indices, representation="hybrid_local")
        features = _fill(design.features, self.fill_values)
        standardized = self.ridge.predict(
            self.feature_scaler.transform(features)
        )
        future_level = self.target_scaler.inverse_transform(
            np.asarray(standardized, dtype=float).reshape(-1, 1)
        )[:, 0]
        prediction = np.full(
            (len(design.transition_indices), data.neurons_per_transition),
            np.nan,
            dtype=float,
        )
        prediction[
            design.transition_positions,
            design.neuron_positions,
        ] = future_level
        return prediction


def fit_direct_future_ridge(
    data: object,
    indices: Sequence[int] | np.ndarray,
    *,
    alpha: float,
) -> FittedDirectFutureRidge:
    """Fit the direct ``d-prime(t+1)`` ridge using training rows only."""

    design = _lag_design(data, indices, representation="hybrid_local")
    global_indices = design.transition_indices[design.transition_positions]
    future = np.asarray(
        data.future[global_indices, design.neuron_positions],
        dtype=float,
    )
    valid = np.isfinite(future)
    weights = hierarchical_row_weights(
        data,
        design.transition_indices,
        design.transition_positions[valid],
    )
    features = design.features[valid]
    response = future[valid, None]
    fill_values = _weighted_fill(features, weights)
    features = _fill(features, fill_values)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="invalid value encountered in sqrt",
            category=RuntimeWarning,
        )
        feature_scaler = StandardScaler().fit(
            features,
            sample_weight=weights,
        )
        target_scaler = StandardScaler().fit(
            response,
            sample_weight=weights,
        )
    ridge = Ridge(alpha=float(alpha)).fit(
        feature_scaler.transform(features),
        target_scaler.transform(response),
        sample_weight=weights,
    )
    return FittedDirectFutureRidge(
        fill_values=fill_values,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        ridge=ridge,
    )


def target_candidate(
    target_parameterization: TargetParameterization,
    *,
    alpha: float,
) -> CandidateSpec:
    """Return one fixed-architecture candidate for the requested target."""

    alpha = float(alpha)
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be finite and positive")
    if target_parameterization == "future_minus_current":
        # Reuse the exact finalized implementation so the locked-reference arm
        # is numerically identical to the immutable final run.
        return lag2_candidate("hybrid_local", alpha=alpha)
    if target_parameterization != "future_level":
        raise ValueError(
            f"unknown target parameterization: {target_parameterization}"
        )

    name = f"lag2_hybrid_local__future_level__alpha_{alpha:g}"

    def fit(
        data: object,
        train: np.ndarray,
        seed: int,
    ) -> FittedDirectFutureRidge:
        del seed
        return fit_direct_future_ridge(data, train, alpha=alpha)

    def predict(
        model: FittedDirectFutureRidge,
        data: object,
        validation: np.ndarray,
    ) -> np.ndarray:
        return model.predict(data, validation)

    return CandidateSpec(
        name=name,
        complexity=(2.0, 12.0, -math.log10(alpha)),
        fit=fit,
        predict=predict,
    )


def candidate_grid(
    target_parameterization: TargetParameterization,
    alphas: Sequence[float] = ALPHAS,
) -> tuple[CandidateSpec, ...]:
    """Build the common alpha grid for one target parameterization."""

    values = tuple(float(alpha) for alpha in alphas)
    if not values or len(set(values)) != len(values):
        raise ValueError("alphas must be a non-empty unique sequence")
    return tuple(
        target_candidate(target_parameterization, alpha=alpha)
        for alpha in values
    )


def parse_alpha(candidate_name: str) -> float:
    match = re.search(r"__alpha_([0-9.eE+-]+)$", str(candidate_name))
    if match is None:
        raise ValueError(f"candidate name has no alpha suffix: {candidate_name}")
    return float(match.group(1))


def _candidate_summary(
    candidate_scores: pd.DataFrame,
    candidates: Sequence[CandidateSpec],
) -> pd.DataFrame:
    by_name = {candidate.name: candidate for candidate in candidates}
    if len(by_name) != len(candidates):
        raise ValueError("candidate names must be unique")
    unknown = set(candidate_scores["candidate"].astype(str)) - set(by_name)
    if unknown:
        raise ValueError(f"scores contain unknown candidates: {sorted(unknown)}")
    result = (
        candidate_scores.groupby("candidate", as_index=False)
        .agg(
            mice=("validation_mouse", "nunique"),
            mean_nrmse=("nrmse", "mean"),
            sem_nrmse=("nrmse", "sem"),
        )
        .fillna({"sem_nrmse": 0.0})
    )
    result["complexity"] = result["candidate"].map(
        lambda name: by_name[str(name)].complexity
    )
    result["alpha"] = result["candidate"].map(parse_alpha)
    return result


def select_minimum_cv(
    candidate_scores: pd.DataFrame,
    candidates: Sequence[CandidateSpec],
) -> OneSESelection:
    """Select minimum inner-LOMO mean NRMSE; prefer larger alpha on exact ties."""

    if candidate_scores.empty:
        raise ValueError("candidate scores must not be empty")
    summary = _candidate_summary(candidate_scores, candidates)
    by_name = {candidate.name: candidate for candidate in candidates}
    ranked = summary.sort_values(
        ["mean_nrmse", "alpha", "candidate"],
        ascending=[True, False, True],
    )
    best = ranked.iloc[0]
    selected_candidate = by_name[str(best["candidate"])]
    return OneSESelection(
        candidate=selected_candidate,
        best_candidate=str(best["candidate"]),
        best_mean_nrmse=float(best["mean_nrmse"]),
        best_sem_nrmse=float(best["sem_nrmse"]),
        threshold_nrmse=float(best["mean_nrmse"]),
        selected_mean_nrmse=float(best["mean_nrmse"]),
        selected_sem_nrmse=float(best["sem_nrmse"]),
        candidate_summary=summary.sort_values(
            ["mean_nrmse", "alpha"],
            ascending=[True, False],
        ).reset_index(drop=True),
    )


def select_candidate(
    candidate_scores: pd.DataFrame,
    candidates: Sequence[CandidateSpec],
    *,
    rule: SelectionRule,
) -> OneSESelection:
    if rule == "one_se":
        return select_one_se(candidate_scores, candidates)
    if rule == "minimum_cv":
        return select_minimum_cv(candidate_scores, candidates)
    raise ValueError(f"unknown selection rule: {rule}")


def _selection_record(
    selection: OneSESelection,
    *,
    procedure: Procedure,
    area: str,
    scope: str,
    outer_mouse: str | None,
) -> dict[str, object]:
    return {
        "procedure": procedure.procedure,
        "display_model": procedure.display_name,
        "target_parameterization": procedure.target_parameterization,
        "selection_rule": procedure.selection_rule,
        "area": area,
        "selection_scope": scope,
        "outer_mouse": outer_mouse,
        "selected_candidate": selection.candidate.name,
        "selected_alpha": parse_alpha(selection.candidate.name),
        "best_candidate": selection.best_candidate,
        "best_alpha": parse_alpha(selection.best_candidate),
        "best_mean_nrmse": selection.best_mean_nrmse,
        "best_sem_nrmse": selection.best_sem_nrmse,
        "selection_threshold_nrmse": selection.threshold_nrmse,
        "selected_mean_nrmse": selection.selected_mean_nrmse,
        "selected_sem_nrmse": selection.selected_sem_nrmse,
    }


def _selection_summary(
    selection: OneSESelection,
    *,
    procedure: Procedure,
    area: str,
    scope: str,
    outer_mouse: str | None,
) -> pd.DataFrame:
    result = selection.candidate_summary.copy()
    result.insert(0, "procedure", procedure.procedure)
    result.insert(1, "display_model", procedure.display_name)
    result.insert(
        2,
        "target_parameterization",
        procedure.target_parameterization,
    )
    result.insert(3, "selection_rule", procedure.selection_rule)
    result.insert(4, "area", area)
    result.insert(5, "selection_scope", scope)
    result.insert(6, "outer_mouse", outer_mouse)
    result["selected"] = result["candidate"].eq(selection.candidate.name)
    if procedure.selection_rule == "one_se":
        result["eligible"] = result["mean_nrmse"].le(
            selection.threshold_nrmse + 1e-15
        )
    else:
        result["eligible"] = result["selected"]
    return result


def _indices_for_mouse(
    data: object,
    pool: np.ndarray,
    mouse: str,
) -> np.ndarray:
    rows = data.metadata.iloc[pool]
    return pool[rows["mouse"].astype(str).eq(mouse).to_numpy()]


def _decorate_prediction(
    frame: pd.DataFrame,
    *,
    model: str,
) -> pd.DataFrame:
    result = frame.copy()
    result["display_model"] = DISPLAY_NAMES[model]
    if model in PROCEDURE_BY_ID:
        procedure = PROCEDURE_BY_ID[model]
        result["target_parameterization"] = procedure.target_parameterization
        result["selection_rule"] = procedure.selection_rule
    else:
        result["target_parameterization"] = "baseline"
        result["selection_rule"] = "baseline"
    result["procedure"] = model
    return result


@dataclass
class FactorialEvaluation:
    """Auditable outputs from the predeclared 2×2 experiment."""

    candidate_scores: pd.DataFrame
    candidate_summary: pd.DataFrame
    selections: pd.DataFrame
    predictions: pd.DataFrame
    frozen_models: dict[tuple[str, str], FittedCandidate]


def evaluate_factorial(
    data: object,
    *,
    alphas: Sequence[float] = ALPHAS,
    areas: Sequence[str] = AREAS,
    seed: int = 2025,
) -> FactorialEvaluation:
    """Evaluate both target parameterizations and both alpha rules.

    Candidate fits used for inner scoring are shared across the two selection
    rules. Every learned prediction remains outer-mouse held out.
    """

    metadata = data.metadata.reset_index(drop=True)
    areas = tuple(str(area) for area in areas)
    candidate_score_frames: list[pd.DataFrame] = []
    candidate_summary_frames: list[pd.DataFrame] = []
    selection_records: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    frozen_models: dict[tuple[str, str], FittedCandidate] = {}

    for target_parameterization in (
        "future_minus_current",
        "future_level",
    ):
        candidates = candidate_grid(target_parameterization, alphas)
        procedures = tuple(
            item
            for item in PROCEDURES
            if item.target_parameterization == target_parameterization
        )
        for area in areas:
            training_pool = np.flatnonzero(
                metadata["area"].astype(str).eq(area)
                & metadata["cohort"].astype(str).eq("unsupervised")
            )
            transfer = np.flatnonzero(
                metadata["area"].astype(str).eq(area)
                & metadata["cohort"].astype(str).eq("supervised")
            )
            training_mice = tuple(
                sorted(
                    metadata.iloc[training_pool]["mouse"]
                    .astype(str)
                    .unique()
                )
            )
            if len(training_mice) < 4 or not len(transfer):
                raise ValueError(
                    f"{area} lacks the required training or transfer rows"
                )

            for outer_mouse in training_mice:
                validation = _indices_for_mouse(
                    data,
                    training_pool,
                    outer_mouse,
                )
                train = training_pool[~np.isin(training_pool, validation)]
                scope = f"outer:{outer_mouse}"
                inner_scores = _candidate_lomo(
                    data,
                    train,
                    candidates,
                    area=area,
                    base_seed=seed,
                    scope=scope,
                    outer_mouse=outer_mouse,
                )
                inner_scores.insert(
                    0,
                    "target_parameterization",
                    target_parameterization,
                )
                candidate_score_frames.append(inner_scores)

                for procedure in procedures:
                    selection = select_candidate(
                        inner_scores,
                        candidates,
                        rule=procedure.selection_rule,
                    )
                    candidate_summary_frames.append(
                        _selection_summary(
                            selection,
                            procedure=procedure,
                            area=area,
                            scope=scope,
                            outer_mouse=outer_mouse,
                        )
                    )
                    selection_records.append(
                        _selection_record(
                            selection,
                            procedure=procedure,
                            area=area,
                            scope=scope,
                            outer_mouse=outer_mouse,
                        )
                    )
                    _, prediction = _fit_predict(
                        data,
                        selection.candidate,
                        train,
                        validation,
                        seed=stable_seed(
                            seed,
                            procedure.procedure,
                            area,
                            scope,
                        ),
                    )
                    frame = _prediction_frame(
                        data,
                        validation,
                        prediction,
                        split="unsupervised_lomo",
                        fold=outer_mouse,
                        model=procedure.procedure,
                        selected_candidate=selection.candidate.name,
                    )
                    prediction_frames.append(
                        _decorate_prediction(
                            frame,
                            model=procedure.procedure,
                        )
                    )

            final_scope = "all_unsupervised"
            final_scores = _candidate_lomo(
                data,
                training_pool,
                candidates,
                area=area,
                base_seed=seed,
                scope=final_scope,
                outer_mouse=None,
            )
            final_scores.insert(
                0,
                "target_parameterization",
                target_parameterization,
            )
            candidate_score_frames.append(final_scores)
            for procedure in procedures:
                selection = select_candidate(
                    final_scores,
                    candidates,
                    rule=procedure.selection_rule,
                )
                candidate_summary_frames.append(
                    _selection_summary(
                        selection,
                        procedure=procedure,
                        area=area,
                        scope=final_scope,
                        outer_mouse=None,
                    )
                )
                selection_records.append(
                    _selection_record(
                        selection,
                        procedure=procedure,
                        area=area,
                        scope=final_scope,
                        outer_mouse=None,
                    )
                )
                fitted, prediction = _fit_predict(
                    data,
                    selection.candidate,
                    training_pool,
                    transfer,
                    seed=stable_seed(
                        seed,
                        procedure.procedure,
                        area,
                        final_scope,
                    ),
                )
                frozen_models[(procedure.procedure, area)] = FittedCandidate(
                    selection.candidate,
                    fitted,
                )
                frame = _prediction_frame(
                    data,
                    transfer,
                    prediction,
                    split="supervised_test",
                    fold="frozen",
                    model=procedure.procedure,
                    selected_candidate=selection.candidate.name,
                )
                prediction_frames.append(
                    _decorate_prediction(
                        frame,
                        model=procedure.procedure,
                    )
                )

    # Add the two predeclared, unlearned baselines once on the exact same rows.
    average = equal_full_average_candidate()
    average_fit = average.fit(
        data,
        np.arange(len(metadata), dtype=int),
        seed,
    )
    for area in areas:
        unsupervised = np.flatnonzero(
            metadata["area"].astype(str).eq(area)
            & metadata["cohort"].astype(str).eq("unsupervised")
        )
        for mouse in sorted(
            metadata.iloc[unsupervised]["mouse"].astype(str).unique()
        ):
            validation = _indices_for_mouse(data, unsupervised, mouse)
            for model, prediction in (
                (PERSISTENCE, np.asarray(data.current)[validation].copy()),
                (
                    TWO_WINDOW_AVERAGE,
                    average.predict(average_fit, data, validation),
                ),
            ):
                frame = _prediction_frame(
                    data,
                    validation,
                    prediction,
                    split="unsupervised_lomo",
                    fold=mouse,
                    model=model,
                )
                prediction_frames.append(
                    _decorate_prediction(frame, model=model)
                )

        supervised = np.flatnonzero(
            metadata["area"].astype(str).eq(area)
            & metadata["cohort"].astype(str).eq("supervised")
        )
        for model, prediction in (
            (PERSISTENCE, np.asarray(data.current)[supervised].copy()),
            (
                TWO_WINDOW_AVERAGE,
                average.predict(average_fit, data, supervised),
            ),
        ):
            frame = _prediction_frame(
                data,
                supervised,
                prediction,
                split="supervised_test",
                fold="frozen",
                model=model,
            )
            prediction_frames.append(
                _decorate_prediction(frame, model=model)
            )

    predictions = pd.concat(prediction_frames, ignore_index=True)
    return FactorialEvaluation(
        candidate_scores=pd.concat(
            candidate_score_frames,
            ignore_index=True,
        ),
        candidate_summary=pd.concat(
            candidate_summary_frames,
            ignore_index=True,
        ),
        selections=pd.DataFrame.from_records(selection_records),
        predictions=predictions,
        frozen_models=frozen_models,
    )


@dataclass
class FactorialArtifacts:
    predictions: pd.DataFrame
    transition_metrics: pd.DataFrame
    mouse_metrics: pd.DataFrame
    metric_summary: pd.DataFrame
    coverage_mouse: pd.DataFrame
    coverage_summary: pd.DataFrame
    paired_vs_locked_mouse: pd.DataFrame
    paired_vs_locked_summary: pd.DataFrame
    paired_vs_persistence_mouse: pd.DataFrame
    paired_vs_persistence_summary: pd.DataFrame
    plot_sample: pd.DataFrame


def _bootstrap_ci(
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


def _sign_test(wins: int, losses: int) -> float:
    total = int(wins + losses)
    if total == 0:
        return np.nan
    tail = min(int(wins), int(losses))
    probability = 2.0 * sum(
        math.comb(total, k) for k in range(tail + 1)
    ) / (2.0**total)
    return float(min(1.0, probability))


def paired_comparison(
    mouse_metrics: pd.DataFrame,
    *,
    baseline: str,
    challengers: Sequence[str],
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare each challenger with one baseline using mice as pairs."""

    paired_frames: list[pd.DataFrame] = []
    for split in ("unsupervised_lomo", "supervised_test"):
        baseline_rows = mouse_metrics.loc[
            mouse_metrics["split"].eq(split)
            & mouse_metrics["model"].eq(baseline)
        ]
        for challenger in challengers:
            challenger_rows = mouse_metrics.loc[
                mouse_metrics["split"].eq(split)
                & mouse_metrics["model"].eq(challenger)
            ]
            keys = ["split", "mouse", "cohort", "area"]
            paired = challenger_rows.merge(
                baseline_rows,
                on=keys,
                how="inner",
                validate="one_to_one",
                suffixes=("_challenger", "_baseline"),
            )
            paired["challenger"] = challenger
            paired["challenger_display"] = DISPLAY_NAMES[challenger]
            paired["baseline"] = baseline
            paired["baseline_display"] = DISPLAY_NAMES[baseline]
            paired["rmse_difference"] = (
                paired["rmse_dprime_challenger"]
                - paired["rmse_dprime_baseline"]
            )
            paired["rmse_reduction_percent"] = 100.0 * (
                paired["rmse_dprime_baseline"]
                - paired["rmse_dprime_challenger"]
            ) / paired["rmse_dprime_baseline"]
            paired_frames.append(paired)
    paired_all = pd.concat(paired_frames, ignore_index=True)

    records: list[dict[str, object]] = []
    grouping = [
        "split",
        "area",
        "challenger",
        "challenger_display",
        "baseline",
        "baseline_display",
    ]
    for key, group in paired_all.groupby(grouping, sort=True):
        values = dict(zip(grouping, key, strict=True))
        differences = group["rmse_difference"].to_numpy(dtype=float)
        reductions = group["rmse_reduction_percent"].to_numpy(dtype=float)
        wins = int(np.sum(differences < 0))
        losses = int(np.sum(differences > 0))
        difference_low, difference_high = _bootstrap_ci(
            differences,
            seed=stable_seed(seed, "difference", *key),
        )
        reduction_low, reduction_high = _bootstrap_ci(
            reductions,
            seed=stable_seed(seed, "reduction", *key),
        )
        model_rmse = float(group["rmse_dprime_challenger"].mean())
        baseline_rmse = float(group["rmse_dprime_baseline"].mean())
        records.append(
            {
                **values,
                "mice": int(group["mouse"].nunique()),
                "mean_challenger_rmse": model_rmse,
                "mean_baseline_rmse": baseline_rmse,
                "rmse_reduction_from_mean_percent": (
                    100.0 * (baseline_rmse - model_rmse) / baseline_rmse
                ),
                "mean_paired_rmse_difference": float(
                    np.mean(differences)
                ),
                "paired_difference_ci95_lower": difference_low,
                "paired_difference_ci95_upper": difference_high,
                "mean_paired_rmse_reduction_percent": float(
                    np.mean(reductions)
                ),
                "paired_reduction_ci95_lower": reduction_low,
                "paired_reduction_ci95_upper": reduction_high,
                "mice_challenger_better": wins,
                "mice_baseline_better": losses,
                "two_sided_sign_test_p": _sign_test(wins, losses),
            }
        )
    return paired_all, pd.DataFrame.from_records(records)


def derive_artifacts(
    evaluation: FactorialEvaluation,
    *,
    seed: int = 2025,
    plot_sample_per_mouse_model: int = 2_000,
) -> FactorialArtifacts:
    transitions = transition_metrics(evaluation.predictions)
    mouse, summary = aggregate_metrics(transitions)
    model_factors = pd.DataFrame(
        [
            {
                "model": model,
                "procedure": model,
                "target_parameterization": (
                    PROCEDURE_BY_ID[model].target_parameterization
                    if model in PROCEDURE_BY_ID
                    else "baseline"
                ),
                "selection_rule": (
                    PROCEDURE_BY_ID[model].selection_rule
                    if model in PROCEDURE_BY_ID
                    else "baseline"
                ),
            }
            for model in DISPLAY_NAMES
        ]
    )
    mouse = mouse.merge(model_factors, on="model", validate="many_to_one")
    summary = summary.merge(model_factors, on="model", validate="many_to_one")
    _, coverage_mouse, coverage_summary = coverage_metrics(
        evaluation.predictions,
        included_models=tuple(DISPLAY_NAMES),
    )
    coverage_mouse = coverage_mouse.merge(
        model_factors,
        on="model",
        validate="many_to_one",
    )
    coverage_summary = coverage_summary.merge(
        model_factors,
        on="model",
        validate="many_to_one",
    )
    learned = tuple(item.procedure for item in PROCEDURES)
    locked_mouse, locked_summary = paired_comparison(
        mouse,
        baseline=LOCKED_REFERENCE,
        challengers=tuple(
            model for model in learned if model != LOCKED_REFERENCE
        ),
        seed=seed,
    )
    persistence_mouse, persistence_summary = paired_comparison(
        mouse,
        baseline=PERSISTENCE,
        challengers=learned,
        seed=seed,
    )
    return FactorialArtifacts(
        predictions=evaluation.predictions,
        transition_metrics=transitions,
        mouse_metrics=mouse,
        metric_summary=summary,
        coverage_mouse=coverage_mouse,
        coverage_summary=coverage_summary,
        paired_vs_locked_mouse=locked_mouse,
        paired_vs_locked_summary=locked_summary,
        paired_vs_persistence_mouse=persistence_mouse,
        paired_vs_persistence_summary=persistence_summary,
        plot_sample=balanced_prediction_sample(
            evaluation.predictions,
            per_mouse_model=plot_sample_per_mouse_model,
            seed=seed,
        ),
    )


__all__ = [
    "ALPHAS",
    "AREAS",
    "DISPLAY_NAMES",
    "FactorialArtifacts",
    "FactorialEvaluation",
    "FittedDirectFutureRidge",
    "LOCKED_REFERENCE",
    "PERSISTENCE",
    "PRIMARY_CHALLENGER",
    "PROCEDURES",
    "PROCEDURE_BY_ID",
    "Procedure",
    "SelectionRule",
    "TWO_WINDOW_AVERAGE",
    "TargetParameterization",
    "candidate_grid",
    "derive_artifacts",
    "evaluate_factorial",
    "fit_direct_future_ridge",
    "paired_comparison",
    "parse_alpha",
    "select_candidate",
    "select_minimum_cv",
    "target_candidate",
]
