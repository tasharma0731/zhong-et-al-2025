"""Mouse-grouped model selection and transfer evaluation."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .config import StudyConfig
from .estimators import (
    CROSS_DAY_MODELS,
    NEXT_WINDOW_MODELS,
    FittedModel,
    ModelSpec,
    fit_model,
    prefixes,
)
from .preparation import metric_matrix, row_nrmse, task_scale


def _mouse_score(
    frame: pd.DataFrame,
    prediction: np.ndarray,
    scale: np.ndarray,
    *,
    task: str,
    metrics: Sequence[str],
) -> float:
    _, target_prefix = prefixes(task)
    errors = row_nrmse(metric_matrix(frame, target_prefix, metrics), prediction, scale)
    if task == "next_window":
        units = frame[["behavior_session_id", "moment"]].copy()
        units["error"] = errors
        return float(
            units.groupby(["moment", "behavior_session_id"])["error"].mean().mean()
        )
    return float(np.mean(errors))


def choose_alpha(
    train: pd.DataFrame,
    spec: ModelSpec,
    config: StudyConfig,
) -> float:
    """Select alpha using leave-one-training-mouse-out CV only."""

    if not spec.tuned:
        raise ValueError(f"{spec.name} does not have a tunable alpha")
    mice = tuple(sorted(train["mouse"].unique()))
    if len(mice) < 3:
        raise ValueError("ridge selection requires at least three training mice")
    scores = []
    for alpha in config.alphas:
        fold_scores = []
        for mouse in mice:
            inner_train = train.loc[train["mouse"].ne(mouse)]
            validation = train.loc[train["mouse"].eq(mouse)]
            model = fit_model(
                inner_train,
                spec,
                metrics=config.metrics,
                alpha=alpha,
            )
            fold_scores.append(
                _mouse_score(
                    validation,
                    model.predict(validation),
                    task_scale(inner_train, spec.task, config.metrics),
                    task=spec.task,
                    metrics=config.metrics,
                )
            )
        scores.append(float(np.mean(fold_scores)))
    return float(config.alphas[int(np.argmin(scores))])


def _prediction_records(
    frame: pd.DataFrame,
    prediction: np.ndarray,
    scale: np.ndarray,
    *,
    spec: ModelSpec,
    split: str,
    area: str,
    alpha: float | None,
    metrics: Sequence[str],
) -> list[dict[str, object]]:
    current_prefix, target_prefix = prefixes(spec.task)
    target = metric_matrix(frame, target_prefix, metrics)
    current = metric_matrix(frame, current_prefix, metrics)
    records = []
    for row_index, row in frame.reset_index(drop=True).iterrows():
        identity: dict[str, object] = {
            "task": spec.task,
            "split": split,
            "model": spec.name,
            "area": area,
            "mouse": row["mouse"],
            "cohort": row["cohort"],
            "alpha": alpha,
            "prediction_id": str(row["mouse"]),
        }
        if spec.task == "next_window":
            prediction_id = (
                f"{row['behavior_session_id']}::{int(row['current_start_pair_id'])}"
            )
            identity.update(
                {
                    "prediction_id": prediction_id,
                    "behavior_session_id": row["behavior_session_id"],
                    "moment": row["moment"],
                    "current_start_pair_id": int(row["current_start_pair_id"]),
                    "next_start_pair_id": int(row["next_start_pair_id"]),
                    "current_last_trial_id": int(row["current_last_trial_id"]),
                    "next_first_trial_id": int(row["next_first_trial_id"]),
                }
            )
        for metric_index, metric in enumerate(metrics):
            observed = float(target[row_index, metric_index])
            predicted = float(prediction[row_index, metric_index])
            metric_scale = float(scale[metric_index])
            records.append(
                {
                    **identity,
                    "metric": metric,
                    "current": float(current[row_index, metric_index]),
                    "observed": observed,
                    "predicted": predicted,
                    "residual": observed - predicted,
                    "scale": metric_scale,
                    "scaled_absolute_error": abs(observed - predicted) / metric_scale,
                    "scaled_squared_error": ((observed - predicted) / metric_scale) ** 2,
                }
            )
    return records


def _score_records(
    frame: pd.DataFrame,
    prediction: np.ndarray,
    scale: np.ndarray,
    *,
    spec: ModelSpec,
    split: str,
    area: str,
    alpha: float | None,
    metrics: Sequence[str],
) -> list[dict[str, object]]:
    _, target_prefix = prefixes(spec.task)
    row_errors = row_nrmse(
        metric_matrix(frame, target_prefix, metrics), prediction, scale
    )
    identity_columns = ["mouse", "cohort"]
    if spec.task == "next_window":
        identity_columns.extend(["behavior_session_id", "moment"])
    scored = frame[identity_columns].copy()
    scored["row_nrmse"] = row_errors
    if spec.task == "next_window":
        # Prevent a long session (or a moment with more usable transitions)
        # from receiving more weight: transition -> session/moment -> mouse.
        units = scored.groupby(
            ["mouse", "cohort", "moment", "behavior_session_id"],
            as_index=False,
        ).agg(row_nrmse=("row_nrmse", "mean"), n_predictions=("row_nrmse", "size"))
    else:
        units = scored.assign(n_predictions=1)
    return [
        {
            "task": spec.task,
            "split": split,
            "model": spec.name,
            "area": area,
            "mouse": mouse,
            "cohort": cohort,
            "alpha": alpha,
            "n_predictions": int(group["n_predictions"].sum()),
            "n_sessions": int(len(group)),
            "nrmse": float(group["row_nrmse"].mean()),
        }
        for (mouse, cohort), group in units.groupby(["mouse", "cohort"], sort=False)
    ]


def evaluate_task(
    data: pd.DataFrame,
    *,
    specs: Sequence[ModelSpec],
    config: StudyConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[tuple[str, str, str], FittedModel]]:
    if not specs:
        raise ValueError("at least one model specification is required")
    task = specs[0].task
    if any(spec.task != task for spec in specs):
        raise ValueError("all model specifications must use the same task")

    predictions: list[dict[str, object]] = []
    scores: list[dict[str, object]] = []
    fitted: dict[tuple[str, str, str], FittedModel] = {}
    for area in config.areas:
        area_data = data.loc[data["area"].eq(area)]
        training = area_data.loc[
            area_data["cohort"].eq(config.training_cohort)
        ].reset_index(drop=True)
        transfer = area_data.loc[
            area_data["cohort"].eq(config.transfer_cohort)
        ].reset_index(drop=True)
        train_mice = tuple(sorted(training["mouse"].unique()))
        if len(train_mice) < 3 or transfer["mouse"].nunique() < 1:
            raise ValueError(f"insufficient mice for {task}/{area}")

        for spec in specs:
            for mouse in train_mice:
                outer_train = training.loc[training["mouse"].ne(mouse)]
                validation = training.loc[training["mouse"].eq(mouse)]
                alpha = choose_alpha(outer_train, spec, config) if spec.tuned else None
                model = fit_model(
                    outer_train,
                    spec,
                    metrics=config.metrics,
                    alpha=alpha,
                )
                scale = task_scale(outer_train, task, config.metrics)
                prediction = model.predict(validation)
                predictions.extend(
                    _prediction_records(
                        validation,
                        prediction,
                        scale,
                        spec=spec,
                        split="unsupervised_lomo",
                        area=area,
                        alpha=alpha,
                        metrics=config.metrics,
                    )
                )
                scores.extend(
                    _score_records(
                        validation,
                        prediction,
                        scale,
                        spec=spec,
                        split="unsupervised_lomo",
                        area=area,
                        alpha=alpha,
                        metrics=config.metrics,
                    )
                )

            final_alpha = choose_alpha(training, spec, config) if spec.tuned else None
            model = fit_model(
                training,
                spec,
                metrics=config.metrics,
                alpha=final_alpha,
            )
            fitted[(task, area, spec.name)] = model
            scale = task_scale(training, task, config.metrics)
            prediction = model.predict(transfer)
            predictions.extend(
                _prediction_records(
                    transfer,
                    prediction,
                    scale,
                    spec=spec,
                    split="supervised_test",
                    area=area,
                    alpha=final_alpha,
                    metrics=config.metrics,
                )
            )
            scores.extend(
                _score_records(
                    transfer,
                    prediction,
                    scale,
                    spec=spec,
                    split="supervised_test",
                    area=area,
                    alpha=final_alpha,
                    metrics=config.metrics,
                )
            )
    return pd.DataFrame(predictions), pd.DataFrame(scores), fitted


def evaluate_study(
    session_pairs: pd.DataFrame,
    transitions: pd.DataFrame,
    config: StudyConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[tuple[str, str, str], FittedModel]]:
    cross_predictions, cross_scores, cross_models = evaluate_task(
        session_pairs,
        specs=CROSS_DAY_MODELS,
        config=config,
    )
    window_predictions, window_scores, window_models = evaluate_task(
        transitions,
        specs=NEXT_WINDOW_MODELS,
        config=config,
    )
    return (
        pd.concat([cross_predictions, window_predictions], ignore_index=True),
        pd.concat([cross_scores, window_scores], ignore_index=True),
        {**cross_models, **window_models},
    )


def evaluate_frozen_cross_day(
    control_pairs: pd.DataFrame,
    training_pairs: pd.DataFrame,
    models: dict[tuple[str, str, str], FittedModel],
    config: StudyConfig,
    *,
    split: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply already-fitted cross-day models to a separate control cohort."""

    prediction_records: list[dict[str, object]] = []
    score_records: list[dict[str, object]] = []
    for area in config.areas:
        tested = control_pairs.loc[control_pairs["area"].eq(area)].reset_index(
            drop=True
        )
        training = training_pairs.loc[
            training_pairs["area"].eq(area)
            & training_pairs["cohort"].eq(config.training_cohort)
        ]
        scale = task_scale(training, "cross_day", config.metrics)
        for spec in CROSS_DAY_MODELS:
            model = models[("cross_day", area, spec.name)]
            prediction = model.predict(tested)
            prediction_records.extend(
                _prediction_records(
                    tested,
                    prediction,
                    scale,
                    spec=spec,
                    split=split,
                    area=area,
                    alpha=model.alpha,
                    metrics=config.metrics,
                )
            )
            score_records.extend(
                _score_records(
                    tested,
                    prediction,
                    scale,
                    spec=spec,
                    split=split,
                    area=area,
                    alpha=model.alpha,
                    metrics=config.metrics,
                )
            )
    return pd.DataFrame(prediction_records), pd.DataFrame(score_records)


def training_mouse_deletion_sensitivity(
    session_pairs: pd.DataFrame,
    transitions: pd.DataFrame,
    config: StudyConfig,
) -> pd.DataFrame:
    """Refit primary models after deleting each unsupervised training mouse."""

    selected_specs = {
        "cross_day": tuple(
            spec
            for spec in CROSS_DAY_MODELS
            if spec.name in {"exposure_shift", "ridge_delta"}
        ),
        "next_window": tuple(
            spec
            for spec in NEXT_WINDOW_MODELS
            if spec.name in {"transition_shift", "ridge_window_delta"}
        ),
    }
    records: list[dict[str, object]] = []
    for task, data in (("cross_day", session_pairs), ("next_window", transitions)):
        for area in config.areas:
            area_data = data.loc[data["area"].eq(area)]
            complete_train = area_data.loc[
                area_data["cohort"].eq(config.training_cohort)
            ]
            tested = area_data.loc[
                area_data["cohort"].eq(config.transfer_cohort)
            ]
            for omitted in sorted(complete_train["mouse"].unique()):
                train = complete_train.loc[complete_train["mouse"].ne(omitted)]
                scale = task_scale(train, task, config.metrics)
                for spec in selected_specs[task]:
                    alpha = choose_alpha(train, spec, config) if spec.tuned else None
                    model = fit_model(
                        train, spec, metrics=config.metrics, alpha=alpha
                    )
                    prediction = model.predict(tested)
                    for score in _score_records(
                        tested,
                        prediction,
                        scale,
                        spec=spec,
                        split="supervised_training_mouse_deleted",
                        area=area,
                        alpha=alpha,
                        metrics=config.metrics,
                    ):
                        records.append({**score, "omitted_training_mouse": omitted})
    return pd.DataFrame.from_records(records).sort_values(
        ["task", "area", "model", "omitted_training_mouse", "mouse"]
    )
