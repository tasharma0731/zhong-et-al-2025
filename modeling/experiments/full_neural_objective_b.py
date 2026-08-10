"""Full-neural cache loading and model candidates for Objective B.

This module is isolated from both the stable Objective B pipeline and the
existing refined SVD experiment.  A compact full-neural cache contains
trial-derived arrays for the sampled neurons and W20 transitions in the broad
areas declared by that cache's recorded configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import hashlib
import json
import math
from pathlib import Path
from typing import Literal, Sequence
import warnings

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from modeling.experiments.build_full_neural_objective_b_cache import (
    NEURON_FIELDS,
    SCHEMA_VERSION,
    VALIDITY_FIELDS,
)
from modeling.experiments.refined_neuron_data import (
    RefinedCacheProvenance,
    RefinedNeuronCache,
    load_refined_cache,
)
from modeling.experiments.refined_neuron_objective_b import (
    DescriptorSpec,
    DescriptorStage,
    FittedRefinedNeuronModel,
    fit_refined_neuron_model,
    hierarchical_row_weights,
)
from modeling.experiments.refined_neuron_validation import CandidateSpec


ReferenceKind = Literal["svd", "hybrid"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _scalar(archive: np.lib.npyio.NpzFile, name: str) -> object:
    if name not in archive.files:
        raise ValueError(f"full-neural cache is missing {name!r}")
    return np.asarray(archive[name]).item()


def _subset_value(
    value: object,
    indices: np.ndarray,
    *,
    source_rows: int,
) -> object:
    if hasattr(value, "iloc") and len(value) == source_rows:
        return value.iloc[indices].reset_index(drop=True)
    if isinstance(value, np.ndarray) and value.ndim >= 1 and value.shape[0] == source_rows:
        return value[indices].copy()
    return value


def load_full_neural_cache(
    path: str | Path,
    *,
    refined_cache_path: str | Path,
    verify_builder: bool = True,
) -> RefinedNeuronCache:
    """Load compact full-neural arrays on a declared subset of the refined cache."""

    path = Path(path).resolve()
    refined_cache_path = Path(refined_cache_path).resolve()
    refined = load_refined_cache(refined_cache_path)
    source_rows = len(refined.metadata)
    with np.load(path, allow_pickle=False) as archive:
        schema = int(_scalar(archive, "full_neural_cache_schema_version"))
        if schema != SCHEMA_VERSION:
            raise ValueError(
                f"full-neural schema {schema} is stale; expected {SCHEMA_VERSION}"
            )
        expected_refined_sha = str(
            _scalar(archive, "source_refined_cache_sha256")
        )
        observed_refined_sha = _sha256(refined_cache_path)
        if expected_refined_sha != observed_refined_sha:
            raise ValueError(
                "full-neural cache was built from a different refined cache"
            )
        builder_sha = str(
            _scalar(archive, "full_neural_cache_builder_sha256")
        )
        if verify_builder:
            builder_path = Path(__file__).with_name(
                "build_full_neural_objective_b_cache.py"
            )
            if _sha256(builder_path) != builder_sha:
                raise ValueError(
                    "full-neural cache builder differs from the recorded version"
                )
        indices = archive["base_indices"].astype(np.int64)
        if (
            indices.ndim != 1
            or len(indices) == 0
            or len(np.unique(indices)) != len(indices)
            or indices.min() < 0
            or indices.max() >= source_rows
        ):
            raise ValueError("full-neural base indices are invalid")
        configuration = json.loads(
            str(_scalar(archive, "full_neural_cache_config_json"))
        )
        source_manifest = json.loads(
            str(_scalar(archive, "full_neural_source_manifest_json"))
        )
        full_arrays = {
            name: archive[name].copy()
            for name in (*NEURON_FIELDS, *VALIDITY_FIELDS)
        }

    values = {
        field.name: _subset_value(
            getattr(refined, field.name),
            indices,
            source_rows=source_rows,
        )
        for field in fields(RefinedNeuronCache)
    }
    svd_reference = {
        "svd_current": np.asarray(values["current"], dtype=np.float32).copy(),
        "svd_future": np.asarray(values["future"], dtype=np.float32).copy(),
        "svd_current_denominator": np.asarray(
            values["current_denominator"],
            dtype=np.float32,
        ).copy(),
        "svd_next_denominator": np.asarray(
            values["next_denominator"],
            dtype=np.float32,
        ).copy(),
        "svd_current_split_dprime_instability": np.asarray(
            values["current_split_dprime_instability"],
            dtype=np.float32,
        ).copy(),
    }
    for name in (
        "current",
        "future",
        "current_denominator",
        "next_denominator",
        "current_valid_denominator",
        "next_valid_denominator",
        "current_mean_leaf",
        "current_mean_circle",
        "current_sd_leaf",
        "current_sd_circle",
        "next_mean_leaf",
        "next_mean_circle",
        "next_sd_leaf",
        "next_sd_circle",
        "current_split_dprime_instability",
        "current_split_signal_instability",
        "current_split_log_denominator_instability",
        "current_split_instability_valid",
    ):
        values[name] = full_arrays[name]

    # Neural/run coupling was not rebuilt from the full representation.  Make
    # accidental use explicit rather than silently mixing SVD and full values.
    neuron_shape = full_arrays["current"].shape
    values["current_neuron_run_correlation"] = np.full(
        neuron_shape,
        np.nan,
        dtype=np.float32,
    )
    values["current_neuron_run_correlation_valid"] = np.zeros(
        neuron_shape,
        dtype=bool,
    )
    values["refined_provenance"] = RefinedCacheProvenance(
        schema_version=schema,
        config={
            **configuration,
            "representation": "full_deconvolved_neural",
            "full_neural_cache": path.name,
        },
        base_sha256=observed_refined_sha,
        builder_sha256=builder_sha,
    )
    result = RefinedNeuronCache(**values)
    for name, value in svd_reference.items():
        setattr(result, name, value)
    setattr(result, "full_neural_source_manifest", source_manifest)
    setattr(result, "full_neural_cache_sha256", _sha256(path))

    expected_shape = result.current.shape
    if expected_shape != (len(result.metadata), result.neurons_per_transition):
        raise ValueError("full-neural metadata and neuronal arrays are misaligned")
    for name in (*NEURON_FIELDS, *VALIDITY_FIELDS):
        if np.asarray(full_arrays[name]).shape != expected_shape:
            raise ValueError(f"{name} must have shape {expected_shape}")
    configured_areas = configuration.get("areas")
    if (
        not isinstance(configured_areas, list)
        or not configured_areas
        or not all(isinstance(area, str) and area for area in configured_areas)
    ):
        raise ValueError(
            "full-neural cache configuration must declare non-empty areas"
        )
    observed_areas = set(result.metadata["area"].astype(str))
    expected_areas = set(configured_areas)
    if observed_areas != expected_areas:
        raise ValueError(
            "full-neural cache areas differ from its recorded configuration: "
            f"observed {sorted(observed_areas)}, expected "
            f"{sorted(expected_areas)}"
        )
    return result


def native_candidate(
    *,
    name: str,
    target_kind: Literal["direct_delta", "two_head"],
    stage: DescriptorStage,
    alpha: float = 1_000.0,
    complexity_prefix: float = 0.0,
) -> CandidateSpec:
    """Create a candidate using the active cache's native representation."""

    descriptor = DescriptorSpec(stage=stage)
    stage_position = list(DescriptorStage).index(stage)
    heads = 1.0 if target_kind == "direct_delta" else 2.0

    def fit(
        data: RefinedNeuronCache,
        train: np.ndarray,
        seed: int,
    ) -> FittedRefinedNeuronModel:
        del seed
        return fit_refined_neuron_model(
            data,
            train,
            target_kind=target_kind,
            descriptor=descriptor,
            alpha=alpha,
        )

    def predict(
        model: FittedRefinedNeuronModel,
        data: RefinedNeuronCache,
        validation: np.ndarray,
    ) -> np.ndarray:
        return model.predict(data, validation)

    return CandidateSpec(
        name=name,
        complexity=(
            float(complexity_prefix),
            float(stage_position),
            heads,
            -math.log10(float(alpha)),
        ),
        fit=fit,
        predict=predict,
    )


@dataclass(frozen=True)
class _ReferenceDesign:
    transition_indices: np.ndarray
    transition_positions: np.ndarray
    neuron_positions: np.ndarray
    features: np.ndarray
    current_full: np.ndarray


def _reference_design(
    data: RefinedNeuronCache,
    indices: Sequence[int] | np.ndarray,
    *,
    kind: ReferenceKind,
) -> _ReferenceDesign:
    selected = np.asarray(indices, dtype=int)
    if selected.ndim != 1 or len(selected) == 0:
        raise ValueError("indices must be a non-empty vector")
    full_denominator = np.asarray(data.current_denominator, dtype=float)
    svd_current = np.asarray(getattr(data, "svd_current"), dtype=float)
    svd_denominator = np.asarray(
        getattr(data, "svd_current_denominator"),
        dtype=float,
    )
    feature_rows: list[np.ndarray] = []
    transition_positions: list[np.ndarray] = []
    neuron_positions: list[np.ndarray] = []
    current_rows: list[np.ndarray] = []
    for transition_position, transition_index in enumerate(selected):
        full = np.asarray(data.current[transition_index], dtype=float)
        full_den = full_denominator[transition_index]
        valid = np.isfinite(full) & np.isfinite(full_den) & (full_den > 0)
        neurons = np.flatnonzero(valid)
        if len(neurons) == 0:
            continue
        svd = svd_current[transition_index, neurons]
        svd_den = svd_denominator[transition_index, neurons]
        with np.errstate(divide="ignore", invalid="ignore"):
            svd_features = np.column_stack(
                [svd, svd * svd_den, np.log(svd_den)]
            )
        if kind == "svd":
            features = svd_features
        elif kind == "hybrid":
            full_values = full[neurons]
            full_den_values = full_den[neurons]
            full_features = np.column_stack(
                [
                    full_values,
                    full_values * full_den_values,
                    np.log(full_den_values),
                ]
            )
            features = np.column_stack([full_features, svd_features])
        else:
            raise ValueError(f"unknown reference kind: {kind}")
        feature_rows.append(features)
        transition_positions.append(
            np.full(len(neurons), transition_position, dtype=int)
        )
        neuron_positions.append(neurons)
        current_rows.append(full[neurons])
    if not feature_rows:
        raise ValueError("selected transitions have no valid full-neural rows")
    return _ReferenceDesign(
        transition_indices=selected.copy(),
        transition_positions=np.concatenate(transition_positions),
        neuron_positions=np.concatenate(neuron_positions),
        features=np.vstack(feature_rows),
        current_full=np.concatenate(current_rows),
    )


def _weighted_fill(
    values: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    fill = np.zeros(values.shape[1], dtype=float)
    for column in range(values.shape[1]):
        finite = np.isfinite(values[:, column])
        if finite.any():
            fill[column] = float(
                np.average(values[finite, column], weights=weights[finite])
            )
    return fill


def _fill(values: np.ndarray, fill: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=float).copy()
    bad = ~np.isfinite(result)
    if bad.any():
        result[bad] = np.broadcast_to(fill, result.shape)[bad]
    return result


@dataclass
class FittedReferenceFeatureModel:
    """Direct full-delta ridge using SVD-only or hybrid current features."""

    kind: ReferenceKind
    fill_values: np.ndarray
    feature_scaler: StandardScaler
    target_scaler: StandardScaler
    ridge: Ridge

    def predict(
        self,
        data: RefinedNeuronCache,
        indices: Sequence[int] | np.ndarray,
    ) -> np.ndarray:
        design = _reference_design(data, indices, kind=self.kind)
        standardized = self.ridge.predict(
            self.feature_scaler.transform(
                _fill(design.features, self.fill_values)
            )
        )
        delta = self.target_scaler.inverse_transform(
            np.asarray(standardized, dtype=float).reshape(-1, 1)
        )[:, 0]
        shape = (len(design.transition_indices), data.neurons_per_transition)
        prediction = np.full(shape, np.nan, dtype=float)
        prediction[
            design.transition_positions,
            design.neuron_positions,
        ] = design.current_full + delta
        return prediction


def fit_reference_feature_model(
    data: RefinedNeuronCache,
    indices: Sequence[int] | np.ndarray,
    *,
    kind: ReferenceKind,
    alpha: float,
) -> FittedReferenceFeatureModel:
    design = _reference_design(data, indices, kind=kind)
    global_indices = design.transition_indices[design.transition_positions]
    future = np.asarray(
        data.future[global_indices, design.neuron_positions],
        dtype=float,
    )
    response = future - design.current_full
    valid = np.isfinite(response)
    if not valid.any():
        raise ValueError("training rows have no finite full-neural targets")
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
    return FittedReferenceFeatureModel(
        kind=kind,
        fill_values=fill_values,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        ridge=ridge,
    )


def reference_candidate(
    *,
    kind: ReferenceKind,
    alpha: float = 1_000.0,
) -> CandidateSpec:
    name = f"{kind}_current__direct_full_delta__alpha_{alpha:g}"
    feature_count = 3.0 if kind == "svd" else 6.0

    def fit(
        data: RefinedNeuronCache,
        train: np.ndarray,
        seed: int,
    ) -> FittedReferenceFeatureModel:
        del seed
        return fit_reference_feature_model(
            data,
            train,
            kind=kind,
            alpha=alpha,
        )

    def predict(
        model: FittedReferenceFeatureModel,
        data: RefinedNeuronCache,
        validation: np.ndarray,
    ) -> np.ndarray:
        return model.predict(data, validation)

    return CandidateSpec(
        name=name,
        complexity=(
            1.0,
            feature_count,
            1.0,
            -math.log10(float(alpha)),
        ),
        fit=fit,
        predict=predict,
    )


def full_candidate_grid() -> tuple[CandidateSpec, ...]:
    """Return a small representation-focused candidate set."""

    return (
        native_candidate(
            name="full_direct_local__alpha_1000",
            target_kind="direct_delta",
            stage=DescriptorStage.LOCAL,
        ),
        native_candidate(
            name="full_direct_response__alpha_1000",
            target_kind="direct_delta",
            stage=DescriptorStage.RESPONSE,
        ),
        native_candidate(
            name="full_two_head_local__alpha_1000",
            target_kind="two_head",
            stage=DescriptorStage.LOCAL,
        ),
        native_candidate(
            name="full_two_head_response__alpha_1000",
            target_kind="two_head",
            stage=DescriptorStage.RESPONSE,
        ),
        reference_candidate(kind="svd"),
        reference_candidate(kind="hybrid"),
    )


__all__ = [
    "FittedReferenceFeatureModel",
    "ReferenceKind",
    "fit_reference_feature_model",
    "full_candidate_grid",
    "load_full_neural_cache",
    "native_candidate",
    "reference_candidate",
]
