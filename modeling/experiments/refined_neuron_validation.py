"""Leakage-resistant validation utilities for experimental neuron forecasts.

This module is intentionally independent of the established modeling
pipeline and of any particular experimental forecaster.  It provides the
validation contract needed by a refined same-neuron next-window experiment:

* hyperparameters are selected with inner leave-one-unsupervised-mouse-out
  (LOMO) validation;
* the complete selection procedure is evaluated with an untouched outer
  unsupervised mouse;
* the final choice is fit to all unsupervised mice and frozen before the
  primary supervised evaluation; any additional supervised diagnostics are
  explicitly labeled exploratory repeated inspection;
* target scales are estimated from the training fold only; and
* errors are averaged neuron -> transition -> session/moment -> mouse.

A candidate owns its preprocessing.  Its ``fit`` callback may inspect target
values only at the supplied training indices.  Its ``predict`` callback must
return one row per supplied validation index and one column per neuron.

The module also supplies persistence and a deliberately simple one-cluster
mean-delta baseline, plus deterministic negative-control data views.  It does
not import an experimental model implementation, which avoids circular
dependencies and keeps the validation code reusable.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np
import pandas as pd


PERSISTENCE_NAME = "neuron_persistence"
K1_BASELINE_NAME = "experimental_k1_mean_delta"
NESTED_MODEL_NAME = "experimental_nested_selected"
FROZEN_MODEL_NAME = "experimental_frozen_selected"


class NeuronForecastData(Protocol):
    """Minimum data interface accepted by this module.

    :class:`modeling.soft_cluster_forecast.Cache` satisfies this protocol.
    Additional attributes remain available to candidate callbacks.
    """

    metadata: pd.DataFrame
    current: np.ndarray
    future: np.ndarray


FitCallback = Callable[[NeuronForecastData, np.ndarray, int], Any]
PredictCallback = Callable[
    [Any, NeuronForecastData, np.ndarray],
    np.ndarray,
]


@dataclass(frozen=True)
class CandidateSpec:
    """A model candidate and its deterministic complexity ordering.

    ``complexity`` is compared lexicographically.  Put the most scientifically
    important simplicity dimension first (for example latent dimension, then
    number of clusters, then effective parameter count).  Candidate name is
    the final deterministic tie-breaker.
    """

    name: str
    complexity: tuple[float, ...]
    fit: FitCallback
    predict: PredictCallback

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("candidate name must not be empty")
        if not self.complexity:
            raise ValueError("candidate complexity must not be empty")
        if not np.isfinite(np.asarray(self.complexity, dtype=float)).all():
            raise ValueError("candidate complexity must be finite")


@dataclass(frozen=True)
class OneSESelection:
    """Result of applying the one-standard-error rule."""

    candidate: CandidateSpec
    best_candidate: str
    best_mean_nrmse: float
    best_sem_nrmse: float
    threshold_nrmse: float
    selected_mean_nrmse: float
    selected_sem_nrmse: float
    candidate_summary: pd.DataFrame


@dataclass
class FittedCandidate:
    """A frozen model bundled with the callback required to apply it."""

    candidate: CandidateSpec
    model: Any

    def predict(
        self,
        data: NeuronForecastData,
        indices: np.ndarray,
    ) -> np.ndarray:
        indices = _validate_indices(data, indices, name="prediction")
        prediction = np.asarray(
            self.candidate.predict(self.model, data, indices.copy()),
            dtype=float,
        )
        expected = (len(indices), np.asarray(data.current).shape[1])
        if prediction.shape != expected:
            raise ValueError(
                f"{self.candidate.name} returned {prediction.shape}; "
                f"expected {expected}"
            )
        return prediction


@dataclass
class RefinedNeuronEvaluation:
    """All auditable outputs from nested validation and frozen transfer."""

    candidate_scores: pd.DataFrame
    candidate_summary: pd.DataFrame
    nested_scores: pd.DataFrame
    transfer_scores: pd.DataFrame
    selections: pd.DataFrame
    nested_predictions: pd.DataFrame
    transfer_predictions: pd.DataFrame
    final_models: dict[tuple[str, str], FittedCandidate]


@dataclass
class ControlledNeuronData:
    """Non-mutating data view used by the negative-control helpers."""

    source: NeuronForecastData
    metadata: pd.DataFrame
    current: np.ndarray
    future: np.ndarray
    overrides: Mapping[str, Any]

    def __getattr__(self, name: str) -> Any:
        if name in self.overrides:
            return self.overrides[name]
        return getattr(self.source, name)


@dataclass(frozen=True)
class _K1MeanDelta:
    delta: float


def stable_seed(*parts: object) -> int:
    """Return an exact, process-stable unsigned 32-bit seed."""

    digest = hashlib.blake2b(
        "\x1f".join(map(str, parts)).encode("utf-8"),
        digest_size=8,
    ).digest()
    return int.from_bytes(digest, "little") % (2**32 - 1)


def _validate_data(data: NeuronForecastData) -> None:
    metadata = data.metadata
    required = {
        "mouse",
        "cohort",
        "area",
        "behavior_session_id",
        "moment",
    }
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"metadata is missing columns: {sorted(missing)}")
    current = np.asarray(data.current)
    future = np.asarray(data.future)
    if current.ndim != 2 or future.shape != current.shape:
        raise ValueError("current and future must be aligned 2-D arrays")
    if len(metadata) != len(current):
        raise ValueError("metadata and neuronal arrays have different row counts")
    if len(metadata) == 0 or current.shape[1] == 0:
        raise ValueError("forecast data must not be empty")


def _validate_indices(
    data: NeuronForecastData,
    indices: np.ndarray,
    *,
    name: str,
) -> np.ndarray:
    result = np.asarray(indices, dtype=int)
    if result.ndim != 1 or len(result) == 0:
        raise ValueError(f"{name} indices must be a non-empty vector")
    if len(np.unique(result)) != len(result):
        raise ValueError(f"{name} indices must be unique")
    if result.min() < 0 or result.max() >= len(data.metadata):
        raise IndexError(f"{name} index lies outside the data")
    return result


def _indices_for_mouse(
    data: NeuronForecastData,
    pool: np.ndarray,
    mouse: str,
) -> np.ndarray:
    rows = data.metadata.iloc[pool]
    return pool[rows["mouse"].astype(str).eq(mouse).to_numpy()]


def training_scale(data: NeuronForecastData, indices: np.ndarray) -> float:
    """Return a training-only scale with equal session and mouse influence.

    Within each behavior-session/moment, the scale is the sample standard
    deviation of all finite future-neuron observations.  Session scales are
    averaged within mouse, then mouse scales are averaged.  This matches the
    hierarchy used by the existing same-neuron analysis while preventing a
    long session or neuron-rich transition from dominating normalization.
    """

    _validate_data(data)
    indices = _validate_indices(data, indices, name="training")
    rows = data.metadata.iloc[indices].copy()
    rows["_position"] = indices
    mouse_scales: list[float] = []
    for _, mouse_rows in rows.groupby("mouse", sort=True):
        session_scales: list[float] = []
        for _, session_rows in mouse_rows.groupby(
            ["behavior_session_id", "moment"],
            sort=True,
        ):
            positions = session_rows["_position"].to_numpy(dtype=int)
            values = np.asarray(data.future)[positions].reshape(-1)
            values = values[np.isfinite(values)]
            if len(values) > 1:
                session_scales.append(float(np.std(values, ddof=1)))
        if session_scales:
            mouse_scales.append(float(np.mean(session_scales)))
    scale = float(np.mean(mouse_scales)) if mouse_scales else 1.0
    return scale if np.isfinite(scale) and scale > 1e-8 else 1.0


def score_predictions(
    data: NeuronForecastData,
    indices: np.ndarray,
    predictions: np.ndarray,
    *,
    scale: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score aligned predictions at transition and mouse levels.

    The returned transition table contains one RMSE per transition.  The mouse
    table first averages transitions inside each area/session/moment, then
    averages those units inside each area/mouse.
    """

    _validate_data(data)
    indices = _validate_indices(data, indices, name="validation")
    predictions = np.asarray(predictions, dtype=float)
    expected_shape = (len(indices), np.asarray(data.future).shape[1])
    if predictions.shape != expected_shape:
        raise ValueError(
            f"predictions have shape {predictions.shape}; expected {expected_shape}"
        )
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be finite and positive")

    records: list[dict[str, object]] = []
    future = np.asarray(data.future)
    metadata = data.metadata.iloc[indices].reset_index(drop=True)
    current = np.asarray(data.current)
    for position, data_index in enumerate(indices):
        eligible = np.isfinite(future[data_index]) & np.isfinite(
            current[data_index]
        )
        if not eligible.any():
            raise ValueError(f"transition {data_index} has no scorable neurons")
        missing_prediction = eligible & ~np.isfinite(predictions[position])
        if missing_prediction.any():
            raise ValueError(
                f"transition {data_index} is missing predictions for "
                f"{int(missing_prediction.sum())} of "
                f"{int(eligible.sum())} eligible neurons"
            )
        valid = eligible
        rmse = float(
            np.sqrt(
                np.mean(
                    np.square(
                        future[data_index, valid] - predictions[position, valid]
                    )
                )
            )
            / scale
        )
        row = metadata.iloc[position]
        records.append(
            {
                "data_index": int(data_index),
                "mouse": str(row["mouse"]),
                "cohort": str(row["cohort"]),
                "area": str(row["area"]),
                "behavior_session_id": str(row["behavior_session_id"]),
                "moment": str(row["moment"]),
                "n_valid_neurons": int(valid.sum()),
                "transition_nrmse": rmse,
            }
        )
    transition_scores = pd.DataFrame.from_records(records)
    session_scores = (
        transition_scores.groupby(
            [
                "mouse",
                "cohort",
                "area",
                "behavior_session_id",
                "moment",
            ],
            as_index=False,
        )
        .agg(
            n_transitions=("transition_nrmse", "size"),
            min_valid_neurons=("n_valid_neurons", "min"),
            session_nrmse=("transition_nrmse", "mean"),
        )
    )
    mouse_scores = (
        session_scores.groupby(
            ["mouse", "cohort", "area"],
            as_index=False,
        )
        .agg(
            n_sessions=("behavior_session_id", "size"),
            n_transitions=("n_transitions", "sum"),
            min_valid_neurons=("min_valid_neurons", "min"),
            nrmse=("session_nrmse", "mean"),
        )
        .sort_values(["area", "mouse"])
        .reset_index(drop=True)
    )
    return transition_scores, mouse_scores


def _prediction_records(
    data: NeuronForecastData,
    indices: np.ndarray,
    predictions: np.ndarray,
    *,
    split: str,
    model: str,
    selected_candidate: str | None,
    fold: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    current = np.asarray(data.current)
    future = np.asarray(data.future)
    for position, data_index in enumerate(indices):
        metadata = data.metadata.iloc[int(data_index)]
        valid = (
            np.isfinite(future[data_index])
            & np.isfinite(current[data_index])
            & np.isfinite(predictions[position])
        )
        for neuron_position in np.flatnonzero(valid):
            rows.append(
                {
                    "split": split,
                    "fold": fold,
                    "model": model,
                    "selected_candidate": selected_candidate,
                    "data_index": int(data_index),
                    "neuron_position": int(neuron_position),
                    "mouse": str(metadata["mouse"]),
                    "cohort": str(metadata["cohort"]),
                    "area": str(metadata["area"]),
                    "behavior_session_id": str(metadata["behavior_session_id"]),
                    "moment": str(metadata["moment"]),
                    "current": float(current[data_index, neuron_position]),
                    "observed": float(future[data_index, neuron_position]),
                    "predicted": float(predictions[position, neuron_position]),
                    "residual": float(
                        future[data_index, neuron_position]
                        - predictions[position, neuron_position]
                    ),
                }
            )
    return pd.DataFrame.from_records(rows)


def _decorate_scores(
    scores: pd.DataFrame,
    *,
    split: str,
    model: str,
    selected_candidate: str | None,
    fold: str,
    scale: float,
) -> pd.DataFrame:
    result = scores.copy()
    result.insert(0, "split", split)
    result.insert(1, "fold", fold)
    result.insert(2, "model", model)
    result.insert(3, "selected_candidate", selected_candidate)
    result["training_scale"] = float(scale)
    return result


def _fit_predict(
    data: NeuronForecastData,
    candidate: CandidateSpec,
    train: np.ndarray,
    validation: np.ndarray,
    *,
    seed: int,
) -> tuple[Any, np.ndarray]:
    train = _validate_indices(data, train, name="training").copy()
    validation = _validate_indices(data, validation, name="validation").copy()
    if np.intersect1d(train, validation).size:
        raise ValueError("training and validation indices overlap")
    model = candidate.fit(data, train, int(seed))
    prediction = np.asarray(
        candidate.predict(model, data, validation),
        dtype=float,
    )
    expected = (len(validation), np.asarray(data.current).shape[1])
    if prediction.shape != expected:
        raise ValueError(
            f"{candidate.name} returned {prediction.shape}; expected {expected}"
        )
    return model, prediction


def _fit_k1(
    data: NeuronForecastData,
    train: np.ndarray,
    seed: int,
) -> _K1MeanDelta:
    del seed
    current = np.asarray(data.current)
    future = np.asarray(data.future)
    rows = data.metadata.iloc[train].copy()
    rows["_position"] = train
    transition_delta: list[float] = []
    for position in train:
        valid = np.isfinite(current[position]) & np.isfinite(future[position])
        if not valid.any():
            raise ValueError(f"training transition {position} has no valid neurons")
        transition_delta.append(
            float(np.mean(future[position, valid] - current[position, valid]))
        )
    rows["_delta"] = transition_delta
    sessions = (
        rows.groupby(
            ["mouse", "behavior_session_id", "moment"],
            as_index=False,
        )["_delta"]
        .mean()
    )
    mouse_delta = sessions.groupby("mouse")["_delta"].mean()
    return _K1MeanDelta(float(mouse_delta.mean()))


def _predict_k1(
    model: _K1MeanDelta,
    data: NeuronForecastData,
    validation: np.ndarray,
) -> np.ndarray:
    return np.asarray(data.current)[validation] + model.delta


def k1_mean_delta_candidate() -> CandidateSpec:
    """Return the built-in one-cluster mean-delta baseline."""

    return CandidateSpec(
        name=K1_BASELINE_NAME,
        complexity=(1.0,),
        fit=_fit_k1,
        predict=_predict_k1,
    )


def _candidate_lomo(
    data: NeuronForecastData,
    pool: np.ndarray,
    candidates: Sequence[CandidateSpec],
    *,
    area: str,
    base_seed: int,
    scope: str,
    outer_mouse: str | None,
) -> pd.DataFrame:
    mice = tuple(sorted(data.metadata.iloc[pool]["mouse"].astype(str).unique()))
    if len(mice) < 2:
        raise ValueError("candidate LOMO needs at least two mice")
    records: list[dict[str, object]] = []
    for held_mouse in mice:
        validation = _indices_for_mouse(data, pool, held_mouse)
        train = pool[~np.isin(pool, validation)]
        scale = training_scale(data, train)
        for candidate in candidates:
            _, prediction = _fit_predict(
                data,
                candidate,
                train,
                validation,
                seed=stable_seed(
                    base_seed,
                    area,
                    scope,
                    outer_mouse,
                    held_mouse,
                    candidate.name,
                ),
            )
            _, mouse_scores = score_predictions(
                data,
                validation,
                prediction,
                scale=scale,
            )
            if len(mouse_scores) != 1:
                raise RuntimeError("one held-out mouse must produce one area score")
            score = mouse_scores.iloc[0]
            records.append(
                {
                    "area": area,
                    "selection_scope": scope,
                    "outer_mouse": outer_mouse,
                    "validation_mouse": held_mouse,
                    "candidate": candidate.name,
                    "complexity": candidate.complexity,
                    "training_scale": float(scale),
                    "n_sessions": int(score["n_sessions"]),
                    "n_transitions": int(score["n_transitions"]),
                    "min_valid_neurons": int(score["min_valid_neurons"]),
                    "nrmse": float(score["nrmse"]),
                }
            )
    return pd.DataFrame.from_records(records)


def select_one_se(
    candidate_scores: pd.DataFrame,
    candidates: Sequence[CandidateSpec],
) -> OneSESelection:
    """Select the simplest candidate within one SEM of the empirical best."""

    if candidate_scores.empty:
        raise ValueError("candidate scores must not be empty")
    by_name = {candidate.name: candidate for candidate in candidates}
    if len(by_name) != len(candidates):
        raise ValueError("candidate names must be unique")
    unknown = set(candidate_scores["candidate"].astype(str)) - set(by_name)
    if unknown:
        raise ValueError(f"scores contain unknown candidates: {sorted(unknown)}")
    summary = (
        candidate_scores.groupby("candidate", as_index=False)
        .agg(
            mice=("validation_mouse", "nunique"),
            mean_nrmse=("nrmse", "mean"),
            sem_nrmse=("nrmse", "sem"),
        )
    )
    summary["sem_nrmse"] = summary["sem_nrmse"].fillna(0.0)
    summary["complexity"] = summary["candidate"].map(
        lambda name: by_name[str(name)].complexity
    )
    best = summary.sort_values(["mean_nrmse", "candidate"]).iloc[0]
    threshold = float(best["mean_nrmse"] + best["sem_nrmse"])
    eligible_names = set(
        summary.loc[
            summary["mean_nrmse"].le(threshold + 1e-15),
            "candidate",
        ].astype(str)
    )
    selected_candidate = min(
        (by_name[name] for name in eligible_names),
        key=lambda candidate: (candidate.complexity, candidate.name),
    )
    selected = summary.loc[
        summary["candidate"].eq(selected_candidate.name)
    ].iloc[0]
    return OneSESelection(
        candidate=selected_candidate,
        best_candidate=str(best["candidate"]),
        best_mean_nrmse=float(best["mean_nrmse"]),
        best_sem_nrmse=float(best["sem_nrmse"]),
        threshold_nrmse=threshold,
        selected_mean_nrmse=float(selected["mean_nrmse"]),
        selected_sem_nrmse=float(selected["sem_nrmse"]),
        candidate_summary=summary.sort_values(
            ["mean_nrmse", "candidate"]
        ).reset_index(drop=True),
    )


def _selection_record(
    selection: OneSESelection,
    *,
    area: str,
    scope: str,
    outer_mouse: str | None,
) -> dict[str, object]:
    return {
        "area": area,
        "selection_scope": scope,
        "outer_mouse": outer_mouse,
        "selected_candidate": selection.candidate.name,
        "selected_complexity": selection.candidate.complexity,
        "best_candidate": selection.best_candidate,
        "best_mean_nrmse": selection.best_mean_nrmse,
        "best_sem_nrmse": selection.best_sem_nrmse,
        "one_se_threshold_nrmse": selection.threshold_nrmse,
        "selected_mean_nrmse": selection.selected_mean_nrmse,
        "selected_sem_nrmse": selection.selected_sem_nrmse,
    }


def _summary_records(
    selection: OneSESelection,
    *,
    area: str,
    scope: str,
    outer_mouse: str | None,
) -> pd.DataFrame:
    result = selection.candidate_summary.copy()
    result.insert(0, "area", area)
    result.insert(1, "selection_scope", scope)
    result.insert(2, "outer_mouse", outer_mouse)
    result["eligible_under_one_se"] = result["mean_nrmse"].le(
        selection.threshold_nrmse + 1e-15
    )
    result["selected"] = result["candidate"].eq(selection.candidate.name)
    return result


def _evaluate_one(
    data: NeuronForecastData,
    *,
    train: np.ndarray,
    validation: np.ndarray,
    candidate: CandidateSpec,
    split: str,
    fold: str,
    base_seed: int,
    collect_predictions: bool,
) -> tuple[Any, list[pd.DataFrame], list[pd.DataFrame]]:
    scale = training_scale(data, train)
    model, learned_prediction = _fit_predict(
        data,
        candidate,
        train,
        validation,
        seed=stable_seed(base_seed, split, fold, candidate.name),
    )
    k1 = k1_mean_delta_candidate()
    _, k1_prediction = _fit_predict(
        data,
        k1,
        train,
        validation,
        seed=stable_seed(base_seed, split, fold, k1.name),
    )
    prediction_sets = (
        (
            NESTED_MODEL_NAME if split == "unsupervised_lomo" else FROZEN_MODEL_NAME,
            candidate.name,
            learned_prediction,
        ),
        (PERSISTENCE_NAME, None, np.asarray(data.current)[validation].copy()),
        (K1_BASELINE_NAME, None, k1_prediction),
    )
    score_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []
    for model_name, selected_candidate, prediction in prediction_sets:
        _, scores = score_predictions(
            data,
            validation,
            prediction,
            scale=scale,
        )
        score_frames.append(
            _decorate_scores(
                scores,
                split=split,
                model=model_name,
                selected_candidate=selected_candidate,
                fold=fold,
                scale=scale,
            )
        )
        if collect_predictions:
            prediction_frames.append(
                _prediction_records(
                    data,
                    validation,
                    prediction,
                    split=split,
                    model=model_name,
                    selected_candidate=selected_candidate,
                    fold=fold,
                )
            )
    return model, score_frames, prediction_frames


def evaluate_nested_transfer(
    data: NeuronForecastData,
    candidates: Sequence[CandidateSpec],
    *,
    training_cohort: str = "unsupervised",
    transfer_cohort: str = "supervised",
    areas: Sequence[str] | None = None,
    seed: int = 2025,
    collect_predictions: bool = True,
) -> RefinedNeuronEvaluation:
    """Run nested unsupervised LOMO and one frozen supervised evaluation."""

    _validate_data(data)
    candidates = tuple(candidates)
    if not candidates:
        raise ValueError("at least one candidate is required")
    if len({candidate.name for candidate in candidates}) != len(candidates):
        raise ValueError("candidate names must be unique")
    if training_cohort == transfer_cohort:
        raise ValueError("training and transfer cohorts must differ")
    available_areas = tuple(sorted(data.metadata["area"].astype(str).unique()))
    areas = tuple(areas or available_areas)
    unknown_areas = sorted(set(areas) - set(available_areas))
    if unknown_areas:
        raise ValueError(f"data has no rows for areas: {unknown_areas}")

    candidate_frames: list[pd.DataFrame] = []
    candidate_summary_frames: list[pd.DataFrame] = []
    nested_score_frames: list[pd.DataFrame] = []
    transfer_score_frames: list[pd.DataFrame] = []
    nested_prediction_frames: list[pd.DataFrame] = []
    transfer_prediction_frames: list[pd.DataFrame] = []
    selections: list[dict[str, object]] = []
    final_models: dict[tuple[str, str], FittedCandidate] = {}

    metadata = data.metadata
    for area in areas:
        training_pool = np.flatnonzero(
            metadata["area"].astype(str).eq(area)
            & metadata["cohort"].astype(str).eq(training_cohort)
        )
        transfer = np.flatnonzero(
            metadata["area"].astype(str).eq(area)
            & metadata["cohort"].astype(str).eq(transfer_cohort)
        )
        training_mice = tuple(
            sorted(metadata.iloc[training_pool]["mouse"].astype(str).unique())
        )
        if len(training_mice) < 4:
            raise ValueError(
                f"nested LOMO for {area} needs at least four training mice"
            )
        if len(transfer) == 0:
            raise ValueError(f"{area} has no transfer rows")

        for outer_mouse in training_mice:
            outer_validation = _indices_for_mouse(
                data,
                training_pool,
                outer_mouse,
            )
            outer_train = training_pool[
                ~np.isin(training_pool, outer_validation)
            ]
            scope = f"outer:{outer_mouse}"
            inner_scores = _candidate_lomo(
                data,
                outer_train,
                candidates,
                area=area,
                base_seed=seed,
                scope=scope,
                outer_mouse=outer_mouse,
            )
            candidate_frames.append(inner_scores)
            selection = select_one_se(inner_scores, candidates)
            candidate_summary_frames.append(
                _summary_records(
                    selection,
                    area=area,
                    scope=scope,
                    outer_mouse=outer_mouse,
                )
            )
            selections.append(
                _selection_record(
                    selection,
                    area=area,
                    scope=scope,
                    outer_mouse=outer_mouse,
                )
            )
            _, scores, prediction_frames = _evaluate_one(
                data,
                train=outer_train,
                validation=outer_validation,
                candidate=selection.candidate,
                split="unsupervised_lomo",
                fold=outer_mouse,
                base_seed=stable_seed(seed, area, scope),
                collect_predictions=collect_predictions,
            )
            nested_score_frames.extend(scores)
            nested_prediction_frames.extend(
                frame for frame in prediction_frames if not frame.empty
            )

        final_scope = "all_unsupervised"
        final_candidate_scores = _candidate_lomo(
            data,
            training_pool,
            candidates,
            area=area,
            base_seed=seed,
            scope=final_scope,
            outer_mouse=None,
        )
        candidate_frames.append(final_candidate_scores)
        final_selection = select_one_se(final_candidate_scores, candidates)
        candidate_summary_frames.append(
            _summary_records(
                final_selection,
                area=area,
                scope=final_scope,
                outer_mouse=None,
            )
        )
        selections.append(
            _selection_record(
                final_selection,
                area=area,
                scope=final_scope,
                outer_mouse=None,
            )
        )
        final_model, scores, prediction_frames = _evaluate_one(
            data,
            train=training_pool,
            validation=transfer,
            candidate=final_selection.candidate,
            split="supervised_test",
            fold="frozen",
            base_seed=stable_seed(seed, area, final_scope),
            collect_predictions=collect_predictions,
        )
        final_models[(area, FROZEN_MODEL_NAME)] = FittedCandidate(
            final_selection.candidate,
            final_model,
        )
        transfer_score_frames.extend(scores)
        transfer_prediction_frames.extend(
            frame for frame in prediction_frames if not frame.empty
        )

    return RefinedNeuronEvaluation(
        candidate_scores=pd.concat(candidate_frames, ignore_index=True),
        candidate_summary=pd.concat(
            candidate_summary_frames,
            ignore_index=True,
        ),
        nested_scores=pd.concat(nested_score_frames, ignore_index=True),
        transfer_scores=pd.concat(transfer_score_frames, ignore_index=True),
        selections=pd.DataFrame.from_records(selections),
        nested_predictions=(
            pd.concat(nested_prediction_frames, ignore_index=True)
            if nested_prediction_frames
            else pd.DataFrame()
        ),
        transfer_predictions=(
            pd.concat(transfer_prediction_frames, ignore_index=True)
            if transfer_prediction_frames
            else pd.DataFrame()
        ),
        final_models=final_models,
    )


def fixed_coordinate_permutation(
    data: NeuronForecastData,
    *,
    seed: int,
) -> ControlledNeuronData:
    """Break neuron-coordinate alignment while preserving anatomical strata.

    Refined caches receive one deterministic permutation per
    recording/broad-area trajectory and fine-area ID.  Thus the control breaks
    the proposed ``(fine area, x, y)`` pseudo-identity without creating
    biologically impossible cross-area assignments.  Generic protocol objects
    without those audit fields fall back to one global column permutation.
    """

    _validate_data(data)
    try:
        cortical_x = np.asarray(getattr(data, "cortical_x"))
        cortical_y = np.asarray(getattr(data, "cortical_y"))
    except AttributeError as error:
        raise ValueError("coordinate permutation requires cortical_x/y") from error
    if cortical_x.shape != data.current.shape or cortical_y.shape != data.current.shape:
        raise ValueError("cortical_x/y must align with neuronal arrays")
    permutation = _stratified_neuron_permutations(
        data,
        seed=seed,
        label="fixed-coordinate-permutation",
    )
    overrides: dict[str, Any] = {
        "cortical_x": _apply_neuron_permutations(cortical_x, permutation),
        "cortical_y": _apply_neuron_permutations(cortical_y, permutation),
        "negative_control": "fixed_coordinate_permutation",
        "negative_control_permutation": permutation.copy(),
    }
    for name in ("cortical_x_raw", "cortical_y_raw"):
        value = getattr(data, name, None)
        if value is not None and np.asarray(value).shape == data.current.shape:
            overrides[name] = _apply_neuron_permutations(value, permutation)
    return ControlledNeuronData(
        source=data,
        metadata=data.metadata.copy(deep=True),
        current=np.asarray(data.current).copy(),
        future=np.asarray(data.future).copy(),
        overrides=overrides,
    )


def fixed_future_neuron_permutation(
    data: NeuronForecastData,
    *,
    seed: int,
) -> ControlledNeuronData:
    """Destroy same-neuron targets while preserving fine-area marginals."""

    _validate_data(data)
    permutation = _stratified_neuron_permutations(
        data,
        seed=seed,
        label="fixed-future-neuron-permutation",
    )
    future = _apply_neuron_permutations(data.future, permutation)
    overrides: dict[str, Any] = {
        "negative_control": "fixed_future_neuron_permutation",
        "negative_control_permutation": permutation.copy(),
    }
    for name in (
        "next_denominator",
        "next_valid_denominator",
        "next_mean_leaf",
        "next_mean_circle",
        "next_sd_leaf",
        "next_sd_circle",
    ):
        value = getattr(data, name, None)
        if value is not None and np.asarray(value).shape == data.current.shape:
            overrides[name] = _apply_neuron_permutations(value, permutation)
    return ControlledNeuronData(
        source=data,
        metadata=data.metadata.copy(deep=True),
        current=np.asarray(data.current).copy(),
        future=future,
        overrides=overrides,
    )


def _stratified_neuron_permutations(
    data: NeuronForecastData,
    *,
    seed: int,
    label: str,
) -> np.ndarray:
    """Return one source-column index for every transition/neuron cell."""

    shape = np.asarray(data.current).shape
    columns = np.broadcast_to(np.arange(shape[1]), shape).copy()
    area_id = getattr(data, "area_id", None)
    if (
        "recording_id" not in data.metadata
        or area_id is None
        or np.asarray(area_id).shape != shape
    ):
        permutation = np.random.default_rng(
            stable_seed(seed, label)
        ).permutation(shape[1])
        return np.broadcast_to(permutation, shape).copy()

    metadata = data.metadata.reset_index(drop=True)
    fine_area = np.asarray(area_id)
    for key, group in metadata.groupby(["recording_id", "area"], sort=True):
        rows = group.index.to_numpy(dtype=int)
        reference = fine_area[rows[0]]
        if not np.equal(fine_area[rows], reference).all():
            raise ValueError(
                "fine-area neuron order changes within a recording trajectory"
            )
        permutation = np.arange(shape[1])
        for fine_id in np.unique(reference):
            positions = np.flatnonzero(reference == fine_id)
            shuffled = np.random.default_rng(
                stable_seed(seed, label, *key, int(fine_id))
            ).permutation(positions)
            permutation[positions] = shuffled
        columns[rows] = permutation
    return columns


def _apply_neuron_permutations(
    values: Any,
    permutation: np.ndarray,
) -> np.ndarray:
    array = np.asarray(values)
    if array.shape != permutation.shape:
        raise ValueError("neuron permutation must align with its values")
    return np.take_along_axis(array, permutation, axis=1).copy()


def circular_target_shift(
    data: NeuronForecastData,
    *,
    seed: int,
    group_columns: Sequence[str] = (
        "mouse",
        "behavior_session_id",
        "moment",
        "area",
    ),
) -> ControlledNeuronData:
    """Break temporal pairing without ever reusing a current window as target.

    Sequences of at least three transitions use a seeded circular offset of
    two or more. A two-transition sequence has no valid circular permutation,
    so it receives seeded target rows from another session of the same mouse
    and area.  The audit table records which method was used; this control is
    therefore a temporal target reassignment, not always a within-session
    circular shift.
    """

    _validate_data(data)
    missing = set(group_columns) - set(data.metadata.columns)
    if missing:
        raise ValueError(f"shift grouping columns are missing: {sorted(missing)}")
    metadata = data.metadata.copy(deep=True)
    metadata["_control_position"] = np.arange(len(metadata), dtype=int)
    target_names = (
        "future",
        "next_denominator",
        "next_valid_denominator",
        "next_mean_leaf",
        "next_mean_circle",
        "next_sd_leaf",
        "next_sd_circle",
    )
    shifted_targets: dict[str, np.ndarray] = {
        name: np.asarray(getattr(data, name)).copy()
        for name in target_names
        if getattr(data, name, None) is not None
        and np.asarray(getattr(data, name)).shape == data.current.shape
    }
    shift_records: list[dict[str, object]] = []
    order_column = (
        "current_start_pair_id"
        if "current_start_pair_id" in metadata
        else "_control_position"
    )
    for key, group in metadata.groupby(list(group_columns), sort=True):
        ordered = group.sort_values(order_column)
        positions = ordered["_control_position"].to_numpy(dtype=int)
        key_parts = key if isinstance(key, tuple) else (key,)
        method = "within_session_circular_shift"
        source_session: str | None = None
        if len(positions) >= 3:
            # Offset one is forbidden: transition t would receive the target
            # from t-1, which is exactly transition t's current window.
            offset = 2 + stable_seed(
                seed,
                "circular-target-shift",
                *key_parts,
            ) % (len(positions) - 2)
            for name in shifted_targets:
                shifted_targets[name][positions] = np.roll(
                    np.asarray(getattr(data, name))[positions],
                    shift=int(offset),
                    axis=0,
                )
        else:
            # Two adjacent transitions cannot be circularly permuted without
            # using either the true target or a target equal to the current
            # window. Use a different session from the same mouse and area.
            row = ordered.iloc[0]
            alternatives = metadata.loc[
                metadata["mouse"].astype(str).eq(str(row["mouse"]))
                & metadata["area"].astype(str).eq(str(row["area"]))
                & ~metadata["behavior_session_id"].astype(str).eq(
                    str(row["behavior_session_id"])
                )
            ].copy()
            if len(alternatives) < len(positions):
                raise ValueError(
                    "short trajectory has no cross-session target fallback"
                )
            eligible_sessions = [
                (str(session), rows.sort_values(order_column))
                for session, rows in alternatives.groupby(
                    "behavior_session_id",
                    sort=True,
                )
                if len(rows) >= len(positions)
            ]
            fallback_seed = stable_seed(
                seed,
                "cross-session-target-fallback",
                *key_parts,
            )
            if eligible_sessions:
                session_position = fallback_seed % len(eligible_sessions)
                source_session, source_rows = eligible_sessions[
                    session_position
                ]
                source_pool = source_rows["_control_position"].to_numpy(
                    dtype=int
                )
            else:
                # This permits a fallback when no single alternative session
                # is long enough, while keeping every donor auditable.
                source_rows = alternatives.sort_values(
                    [order_column, "_control_position"]
                )
                source_pool = source_rows["_control_position"].to_numpy(
                    dtype=int
                )
                source_session = "multiple:" + ",".join(
                    sorted(
                        source_rows["behavior_session_id"]
                        .astype(str)
                        .unique()
                    )
                )
            source_positions = np.random.default_rng(
                stable_seed(fallback_seed, "source-rows")
            ).choice(
                source_pool,
                size=len(positions),
                replace=False,
            )
            for name in shifted_targets:
                shifted_targets[name][positions] = np.asarray(
                    getattr(data, name)
                )[source_positions]
            offset = 0
            method = "cross_session_fallback"
        shift_records.append(
            {
                **{
                    column: value
                    for column, value in zip(
                        group_columns,
                        key if isinstance(key, tuple) else (key,),
                        strict=False,
                    )
                },
                "n_transitions": int(len(positions)),
                "offset": int(offset),
                "method": method,
                "source_session": source_session,
            }
        )
    metadata = metadata.drop(columns="_control_position")
    return ControlledNeuronData(
        source=data,
        metadata=metadata,
        current=np.asarray(data.current).copy(),
        future=shifted_targets["future"],
        overrides={
            "negative_control": (
                "temporal_target_shift_with_cross_session_fallback"
            ),
            "negative_control_shifts": pd.DataFrame.from_records(shift_records),
            **{
                name: value
                for name, value in shifted_targets.items()
                if name != "future"
            },
        },
    )


# A concise alias for experiment runners.
evaluate = evaluate_nested_transfer
