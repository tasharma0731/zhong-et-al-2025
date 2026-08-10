"""Small, explicit estimators used by both prediction tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from .config import DEFAULT_METRICS
from .preparation import metric_matrix
from .targets import DistributionTargetTransform


ModelKind = Literal["persistence", "training_mean", "mean_shift", "ridge"]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    task: Literal["cross_day", "next_window"]
    kind: ModelKind
    predict_delta: bool = False
    include_behavior: bool = False

    @property
    def tuned(self) -> bool:
        return self.kind == "ridge"


CROSS_DAY_MODELS = (
    ModelSpec("persistence", "cross_day", "persistence"),
    ModelSpec("training_mean", "cross_day", "training_mean"),
    ModelSpec("exposure_shift", "cross_day", "mean_shift", predict_delta=True),
    ModelSpec("ridge_delta", "cross_day", "ridge", predict_delta=True),
    ModelSpec(
        "ridge_delta_behavior",
        "cross_day",
        "ridge",
        predict_delta=True,
        include_behavior=True,
    ),
)

NEXT_WINDOW_MODELS = (
    ModelSpec("persistence", "next_window", "persistence"),
    ModelSpec("transition_shift", "next_window", "mean_shift", predict_delta=True),
    ModelSpec("ridge_level", "next_window", "ridge"),
    ModelSpec(
        "ridge_level_behavior",
        "next_window",
        "ridge",
        include_behavior=True,
    ),
    ModelSpec("ridge_window_delta", "next_window", "ridge", predict_delta=True),
    ModelSpec(
        "ridge_window_delta_behavior",
        "next_window",
        "ridge",
        predict_delta=True,
        include_behavior=True,
    ),
)


def prefixes(task: str) -> tuple[str, str]:
    if task == "cross_day":
        return "before", "after"
    if task == "next_window":
        return "current", "next"
    raise ValueError(f"unknown task: {task}")


def behavior_columns(task: str) -> tuple[str, ...]:
    if task == "cross_day":
        return ("before_mean_run_speed",)
    if task == "next_window":
        return ("current_run_speed", "current_progress")
    raise ValueError(f"unknown task: {task}")


def feature_matrix(
    frame: pd.DataFrame,
    spec: ModelSpec,
    metrics: Sequence[str],
) -> np.ndarray:
    current_prefix, _ = prefixes(spec.task)
    neural = metric_matrix(frame, current_prefix, metrics)
    if not spec.include_behavior:
        return neural
    return np.column_stack(
        [neural, frame[list(behavior_columns(spec.task))].to_numpy(dtype=float)]
    )


def mouse_sample_weights(frame: pd.DataFrame) -> np.ndarray:
    """Give every mouse, then every session within mouse, equal influence."""

    if "mouse" not in frame or frame.empty:
        raise ValueError("frame must contain at least one mouse")
    if "behavior_session_id" in frame:
        rows_per_session = frame.groupby(
            ["mouse", "behavior_session_id"]
        )["mouse"].transform("size").to_numpy(dtype=float)
        sessions_per_mouse = frame.groupby("mouse")[
            "behavior_session_id"
        ].transform("nunique").to_numpy(dtype=float)
        weights = 1.0 / (rows_per_session * sessions_per_mouse)
    else:
        counts = frame.groupby("mouse")["mouse"].transform("size").to_numpy(
            dtype=float
        )
        weights = 1.0 / counts
    return weights * (len(weights) / weights.sum())


@dataclass
class FittedModel:
    """Serializable fitted estimator with an intentionally small interface."""

    spec: ModelSpec
    metrics: tuple[str, ...]
    alpha: float | None = None
    constant: np.ndarray | None = None
    feature_scaler: StandardScaler | None = None
    target_scaler: StandardScaler | None = None
    ridge: Ridge | None = None
    target_transform: DistributionTargetTransform | None = None

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        current_prefix, _ = prefixes(self.spec.task)
        current = metric_matrix(frame, current_prefix, self.metrics)
        transform = self.target_transform or DistributionTargetTransform(self.metrics)
        current_transformed = transform.transform(current)
        if self.spec.kind == "persistence":
            return current.copy()
        if self.spec.kind == "training_mean":
            if self.constant is None:
                raise RuntimeError("training-mean model has no fitted constant")
            prediction = transform.inverse_transform(
                np.broadcast_to(self.constant, current.shape).copy()
            )
            transform.validate(prediction)
            return prediction
        if self.spec.kind == "mean_shift":
            if self.constant is None:
                raise RuntimeError("mean-shift model has no fitted constant")
            prediction = transform.inverse_transform(current_transformed + self.constant)
            transform.validate(prediction)
            return prediction
        if self.spec.kind != "ridge":
            raise RuntimeError(f"unsupported fitted model kind: {self.spec.kind}")
        if self.feature_scaler is None or self.target_scaler is None or self.ridge is None:
            raise RuntimeError("ridge model is incomplete")
        standardized = np.asarray(
            self.ridge.predict(
                self.feature_scaler.transform(feature_matrix(frame, self.spec, self.metrics))
            ),
            dtype=float,
        ).reshape(len(frame), -1)
        response = self.target_scaler.inverse_transform(standardized)
        transformed_prediction = (
            current_transformed + response if self.spec.predict_delta else response
        )
        prediction = transform.inverse_transform(transformed_prediction)
        transform.validate(prediction)
        return prediction


def fit_model(
    train: pd.DataFrame,
    spec: ModelSpec,
    *,
    metrics: Sequence[str] = DEFAULT_METRICS,
    alpha: float | None = None,
) -> FittedModel:
    metrics = tuple(metrics)
    current_prefix, target_prefix = prefixes(spec.task)
    current = metric_matrix(train, current_prefix, metrics)
    target = metric_matrix(train, target_prefix, metrics)
    transform = DistributionTargetTransform(metrics)
    transform.validate(current)
    transform.validate(target)
    current_transformed = transform.transform(current)
    target_transformed = transform.transform(target)
    weights = mouse_sample_weights(train)

    if spec.kind == "persistence":
        return FittedModel(spec, metrics, target_transform=transform)
    if spec.kind == "training_mean":
        return FittedModel(
            spec,
            metrics,
            constant=np.average(target_transformed, axis=0, weights=weights),
            target_transform=transform,
        )
    if spec.kind == "mean_shift":
        return FittedModel(
            spec,
            metrics,
            constant=np.average(
                target_transformed - current_transformed,
                axis=0,
                weights=weights,
            ),
            target_transform=transform,
        )
    if spec.kind != "ridge":
        raise ValueError(f"unknown model kind: {spec.kind}")
    if alpha is None or alpha <= 0:
        raise ValueError("ridge models require a positive alpha")

    response = (
        target_transformed - current_transformed
        if spec.predict_delta
        else target_transformed
    )
    features = feature_matrix(train, spec, metrics)
    feature_scaler = StandardScaler().fit(features, sample_weight=weights)
    target_scaler = StandardScaler().fit(response, sample_weight=weights)
    ridge = Ridge(alpha=float(alpha)).fit(
        feature_scaler.transform(features),
        target_scaler.transform(response),
        sample_weight=weights,
    )
    return FittedModel(
        spec=spec,
        metrics=metrics,
        alpha=float(alpha),
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        ridge=ridge,
        target_transform=transform,
    )
