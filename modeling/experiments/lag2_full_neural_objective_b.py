"""Two-window candidates for the isolated full-neural Objective B branch.

The active row remains ``current window t -> future window t+1``.  A lag-two
view adds the immediately preceding disjoint W20 window ``t-1`` for the same
session, broad area, and neuron axis.  Prediction never reads the future
window.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, fields, replace
import math
from typing import Literal, Sequence
import warnings

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from modeling.experiments.full_neural_objective_b import (
    _fill,
    _weighted_fill,
)
from modeling.experiments.refined_neuron_data import RefinedNeuronCache
from modeling.experiments.refined_neuron_objective_b import (
    hierarchical_row_weights,
)
from modeling.experiments.refined_neuron_validation import CandidateSpec


LagRepresentation = Literal[
    "full_dprime",
    "full_local",
    "svd_local",
    "hybrid_local",
]


def _subset_rows(value: object, indices: np.ndarray, source_rows: int) -> object:
    if hasattr(value, "iloc") and len(value) == source_rows:
        return value.iloc[indices].reset_index(drop=True)
    if (
        isinstance(value, np.ndarray)
        and value.ndim >= 1
        and value.shape[0] == source_rows
    ):
        return value[indices].copy()
    return value


def build_lag2_view(cache: RefinedNeuronCache) -> RefinedNeuronCache:
    """Keep rows with an exact preceding W20 transition and attach its state."""

    metadata = cache.metadata.reset_index(drop=True)
    lookup = {
        (
            str(row.behavior_session_id),
            str(row.area),
            int(row.current_start_pair_id),
        ): int(index)
        for index, row in metadata.iterrows()
    }
    current_indices: list[int] = []
    previous_indices: list[int] = []
    for index, row in metadata.iterrows():
        previous_key = (
            str(row["behavior_session_id"]),
            str(row["area"]),
            int(row["current_start_pair_id"]) - int(cache.window_pairs),
        )
        if previous_key in lookup:
            current_indices.append(int(index))
            previous_indices.append(lookup[previous_key])
    current = np.asarray(current_indices, dtype=int)
    previous = np.asarray(previous_indices, dtype=int)
    if not len(current):
        raise ValueError("cache has no strictly consecutive lag-two transitions")

    if cache.neuron_id is not None:
        if not np.array_equal(cache.neuron_id[current], cache.neuron_id[previous]):
            raise ValueError("neuron identity/order changed between lag-two windows")
    if cache.area_id is not None:
        if not np.array_equal(cache.area_id[current], cache.area_id[previous]):
            raise ValueError("fine-area identity changed between lag-two windows")

    source_rows = len(metadata)
    values = {
        field.name: _subset_rows(
            getattr(cache, field.name),
            current,
            source_rows,
        )
        for field in fields(RefinedNeuronCache)
    }
    lag_metadata = values["metadata"].copy()
    lag_metadata["source_data_index"] = current
    lag_metadata["previous_source_data_index"] = previous
    lag_metadata["previous_start_pair_id"] = (
        lag_metadata["current_start_pair_id"].to_numpy(dtype=int)
        - int(cache.window_pairs)
    )
    values["metadata"] = lag_metadata
    if cache.refined_provenance is not None:
        values["refined_provenance"] = replace(
            cache.refined_provenance,
            config={
                **cache.refined_provenance.config,
                "lag_windows": 2,
                "lag_semantics": "strict consecutive disjoint W20 windows",
            },
        )
    result = RefinedNeuronCache(**values)

    current_dynamic = (
        "svd_current",
        "svd_future",
        "svd_current_denominator",
        "svd_next_denominator",
        "svd_current_split_dprime_instability",
    )
    for name in current_dynamic:
        setattr(result, name, np.asarray(getattr(cache, name))[current].copy())
    result.previous_full = np.asarray(cache.current)[previous].copy()
    result.previous_full_denominator = np.asarray(
        cache.current_denominator
    )[previous].copy()
    result.previous_svd = np.asarray(cache.svd_current)[previous].copy()
    result.previous_svd_denominator = np.asarray(
        cache.svd_current_denominator
    )[previous].copy()
    result.lag2_current_source_indices = current
    result.lag2_previous_source_indices = previous
    return result


@dataclass(frozen=True)
class _LagDesign:
    transition_indices: np.ndarray
    transition_positions: np.ndarray
    neuron_positions: np.ndarray
    features: np.ndarray
    current_full: np.ndarray


def _local_features(dprime: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.column_stack(
            (dprime, dprime * denominator, np.log(denominator))
        )


def _lag_design(
    data: RefinedNeuronCache,
    indices: Sequence[int] | np.ndarray,
    *,
    representation: LagRepresentation,
) -> _LagDesign:
    selected = np.asarray(indices, dtype=int)
    if selected.ndim != 1 or len(selected) == 0:
        raise ValueError("indices must be a non-empty vector")
    rows: list[np.ndarray] = []
    transition_positions: list[np.ndarray] = []
    neuron_positions: list[np.ndarray] = []
    current_rows: list[np.ndarray] = []

    previous_full = np.asarray(data.previous_full, dtype=float)
    previous_full_denominator = np.asarray(
        data.previous_full_denominator,
        dtype=float,
    )
    previous_svd = np.asarray(data.previous_svd, dtype=float)
    previous_svd_denominator = np.asarray(
        data.previous_svd_denominator,
        dtype=float,
    )
    current_full_denominator = np.asarray(data.current_denominator, dtype=float)
    current_svd = np.asarray(data.svd_current, dtype=float)
    current_svd_denominator = np.asarray(
        data.svd_current_denominator,
        dtype=float,
    )

    for transition_position, transition_index in enumerate(selected):
        current_full = np.asarray(data.current[transition_index], dtype=float)
        current_denominator = current_full_denominator[transition_index]
        valid = (
            np.isfinite(current_full)
            & np.isfinite(current_denominator)
            & (current_denominator > 0)
        )
        neurons = np.flatnonzero(valid)
        if not len(neurons):
            continue

        prior_full = previous_full[transition_index, neurons]
        prior_full_den = previous_full_denominator[transition_index, neurons]
        now_full = current_full[neurons]
        now_full_den = current_denominator[neurons]
        if representation == "full_dprime":
            features = np.column_stack((prior_full, now_full))
        elif representation == "full_local":
            full_features = np.column_stack(
                (
                    _local_features(prior_full, prior_full_den),
                    _local_features(now_full, now_full_den),
                )
            )
            features = full_features
        else:
            prior_svd = previous_svd[transition_index, neurons]
            prior_svd_den = previous_svd_denominator[
                transition_index,
                neurons,
            ]
            now_svd = current_svd[transition_index, neurons]
            now_svd_den = current_svd_denominator[
                transition_index,
                neurons,
            ]
            svd_features = np.column_stack(
                (
                    _local_features(prior_svd, prior_svd_den),
                    _local_features(now_svd, now_svd_den),
                )
            )
            if representation == "svd_local":
                features = svd_features
            elif representation == "hybrid_local":
                full_features = np.column_stack(
                    (
                        _local_features(prior_full, prior_full_den),
                        _local_features(now_full, now_full_den),
                    )
                )
                features = np.column_stack(
                    (
                        full_features,
                        svd_features,
                    )
                )
            else:
                raise ValueError(f"unknown lag representation: {representation}")
        rows.append(features)
        transition_positions.append(
            np.full(len(neurons), transition_position, dtype=int)
        )
        neuron_positions.append(neurons)
        current_rows.append(now_full)
    if not rows:
        raise ValueError("selected transitions have no valid neurons")
    return _LagDesign(
        transition_indices=selected.copy(),
        transition_positions=np.concatenate(transition_positions),
        neuron_positions=np.concatenate(neuron_positions),
        features=np.vstack(rows),
        current_full=np.concatenate(current_rows),
    )


@dataclass
class FittedLag2Ridge:
    representation: LagRepresentation
    fill_values: np.ndarray
    feature_scaler: StandardScaler
    target_scaler: StandardScaler
    ridge: Ridge

    def predict(
        self,
        data: RefinedNeuronCache,
        indices: Sequence[int] | np.ndarray,
    ) -> np.ndarray:
        design = _lag_design(
            data,
            indices,
            representation=self.representation,
        )
        features = _fill(design.features, self.fill_values)
        standardized = self.ridge.predict(
            self.feature_scaler.transform(features)
        )
        delta = self.target_scaler.inverse_transform(
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
        ] = design.current_full + delta
        return prediction


def fit_lag2_ridge(
    data: RefinedNeuronCache,
    indices: Sequence[int] | np.ndarray,
    *,
    representation: LagRepresentation,
    alpha: float,
) -> FittedLag2Ridge:
    design = _lag_design(data, indices, representation=representation)
    global_indices = design.transition_indices[design.transition_positions]
    future = np.asarray(
        data.future[global_indices, design.neuron_positions],
        dtype=float,
    )
    response = future - design.current_full
    valid = np.isfinite(response)
    weights = hierarchical_row_weights(
        data,
        design.transition_indices,
        design.transition_positions[valid],
    )
    features = design.features[valid]
    response = response[valid, None]
    fill_values = _weighted_fill(features, weights)
    features = _fill(features, fill_values)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="invalid value encountered in sqrt",
            category=RuntimeWarning,
        )
        feature_scaler = StandardScaler().fit(features, sample_weight=weights)
        target_scaler = StandardScaler().fit(response, sample_weight=weights)
    ridge = Ridge(alpha=float(alpha)).fit(
        feature_scaler.transform(features),
        target_scaler.transform(response),
        sample_weight=weights,
    )
    return FittedLag2Ridge(
        representation=representation,
        fill_values=fill_values,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        ridge=ridge,
    )


def lag2_candidate(
    representation: LagRepresentation,
    *,
    alpha: float = 1_000.0,
) -> CandidateSpec:
    name = f"lag2_{representation}__alpha_{alpha:g}"
    feature_count = {
        "full_dprime": 2.0,
        "full_local": 6.0,
        "svd_local": 6.0,
        "hybrid_local": 12.0,
    }[representation]

    def fit(
        data: RefinedNeuronCache,
        train: np.ndarray,
        seed: int,
    ) -> FittedLag2Ridge:
        del seed
        return fit_lag2_ridge(
            data,
            train,
            representation=representation,
            alpha=alpha,
        )

    def predict(
        model: FittedLag2Ridge,
        data: RefinedNeuronCache,
        validation: np.ndarray,
    ) -> np.ndarray:
        return model.predict(data, validation)

    return CandidateSpec(
        name=name,
        complexity=(2.0, feature_count, -math.log10(float(alpha))),
        fit=fit,
        predict=predict,
    )


@dataclass(frozen=True)
class _EqualAverage:
    pass


def equal_full_average_candidate() -> CandidateSpec:
    """Predict with the unlearned mean of full d-prime at t-1 and t."""

    def fit(
        data: RefinedNeuronCache,
        train: np.ndarray,
        seed: int,
    ) -> _EqualAverage:
        del data, train, seed
        return _EqualAverage()

    def predict(
        model: _EqualAverage,
        data: RefinedNeuronCache,
        validation: np.ndarray,
    ) -> np.ndarray:
        del model
        current = np.asarray(data.current)[validation]
        previous = np.asarray(data.previous_full)[validation]
        usable = np.isfinite(previous)
        return np.where(usable, 0.5 * (previous + current), current)

    return CandidateSpec(
        name="equal_average_full_dprime_tminus1_t",
        complexity=(0.0, 2.0, 0.0),
        fit=fit,
        predict=predict,
    )


__all__ = [
    "FittedLag2Ridge",
    "LagRepresentation",
    "build_lag2_view",
    "equal_full_average_candidate",
    "fit_lag2_ridge",
    "lag2_candidate",
]
