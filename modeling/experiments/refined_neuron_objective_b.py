"""Future-blind per-neuron models for the experimental Objective B branch.

This module is deliberately isolated from the stable modeling pipeline.  It
provides the reusable core needed to compare two same-neuron, next-window
targets:

``direct_delta``
    Predict the change in d-prime directly.

``two_head``
    Predict the change in the response numerator and in the log denominator,
    then reconstruct d-prime exactly from those two predicted components.

Every prediction uses only the observed current window and static anatomy.
Future arrays are read only while constructing a training response.  Rows are
weighted through the full mouse -> session/moment -> transition -> neuron
hierarchy so that repeated neurons and long recordings do not masquerade as
additional mice.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
from typing import Literal, Sequence
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from modeling.soft_cluster_forecast import Cache


TargetKind = Literal["direct_delta", "two_head"]


class DescriptorStage(str, Enum):
    """Cumulative, preregistration-friendly feature stages."""

    LOCAL = "local"
    ANATOMY = "anatomy"
    SPATIAL = "spatial"
    POPULATION = "population"
    RESPONSE = "response"
    TEMPORAL = "temporal"
    BEHAVIOR = "behavior"
    COUPLING = "coupling"


DESCRIPTOR_STAGES = tuple(DescriptorStage)
_STAGE_POSITION = {stage: index for index, stage in enumerate(DESCRIPTOR_STAGES)}


@dataclass(frozen=True)
class DescriptorSpec:
    """One cumulative descriptor configuration.

    ``spatial_degree=1`` adds aligned x/y terms.  Degree two additionally adds
    x-squared, x-by-y, and y-squared terms, which form a small smooth spatial
    basis without learning knots from validation mice.
    """

    stage: DescriptorStage = DescriptorStage.LOCAL
    spatial_degree: int = 2
    selectivity_threshold: float = 0.3

    def __post_init__(self) -> None:
        if not isinstance(self.stage, DescriptorStage):
            object.__setattr__(self, "stage", DescriptorStage(self.stage))
        if self.spatial_degree not in (1, 2):
            raise ValueError("spatial_degree must be 1 or 2")
        if (
            not np.isfinite(self.selectivity_threshold)
            or self.selectivity_threshold < 0
        ):
            raise ValueError("selectivity_threshold must be finite and non-negative")

    def includes(self, stage: DescriptorStage) -> bool:
        return _STAGE_POSITION[self.stage] >= _STAGE_POSITION[stage]


@dataclass(frozen=True)
class FeatureSchema:
    """Categorical levels learned from a training-mouse fold."""

    broad_areas: tuple[str, ...] = ()
    fine_areas: tuple[int, ...] = ()


@dataclass(frozen=True)
class CurrentNeuronDesign:
    """Flattened current-only neuron rows and their cache identities."""

    transition_indices: np.ndarray
    transition_positions: np.ndarray
    neuron_positions: np.ndarray
    features: np.ndarray
    feature_names: tuple[str, ...]
    current_dprime: np.ndarray
    current_numerator: np.ndarray
    current_denominator: np.ndarray


@dataclass(frozen=True)
class NeuronPrediction:
    """Predictions aligned as requested transitions by cache neuron position."""

    transition_indices: np.ndarray
    dprime: np.ndarray
    numerator: np.ndarray | None = None
    denominator: np.ndarray | None = None


@dataclass(frozen=True)
class PredictionContract:
    """Cache semantics that a serialized estimator is allowed to accept."""

    window_pairs: int
    stable_history_sha256: str | None
    stable_build_config_json: str | None
    refined_schema_version: int | None
    refined_builder_sha256: str | None
    refined_config_json: str | None


@dataclass
class FittedRefinedNeuronModel:
    """Serializable weighted-ridge model with a future-blind prediction API."""

    target_kind: TargetKind
    descriptor: DescriptorSpec
    schema: FeatureSchema
    alpha: float
    feature_names: tuple[str, ...]
    feature_fill_values: np.ndarray
    feature_scaler: StandardScaler
    target_scaler: StandardScaler
    ridge: Ridge
    training_mice: tuple[str, ...]
    training_areas: tuple[str, ...]
    prediction_contract: PredictionContract

    def predict_components(
        self,
        cache: Cache,
        indices: Sequence[int] | np.ndarray | None = None,
    ) -> NeuronPrediction:
        """Predict without reading ``future`` or ``next_denominator`` arrays."""

        observed_contract = _prediction_contract(cache)
        if observed_contract != self.prediction_contract:
            raise ValueError(
                "prediction cache contract differs from the fitted model: "
                f"expected {self.prediction_contract}, "
                f"observed {observed_contract}"
            )
        selected = _validated_indices(cache, indices)
        observed_areas = tuple(
            sorted(
                cache.metadata.iloc[selected]["area"].astype(str).unique()
            )
        )
        unexpected_areas = sorted(set(observed_areas) - set(self.training_areas))
        if unexpected_areas:
            raise ValueError(
                "prediction contains brain areas outside the fitted model: "
                f"trained on {self.training_areas}, "
                f"received {tuple(unexpected_areas)}"
            )
        design = build_current_design(
            cache,
            selected,
            descriptor=self.descriptor,
            schema=self.schema,
        )
        if design.feature_names != self.feature_names:
            raise RuntimeError("prediction feature schema differs from fitted schema")

        feature_values = _fill_nonfinite(
            design.features, self.feature_fill_values
        )
        standardized = np.asarray(
            self.ridge.predict(self.feature_scaler.transform(feature_values)),
            dtype=float,
        ).reshape(len(feature_values), -1)
        response = self.target_scaler.inverse_transform(standardized)

        shape = (len(selected), cache.neurons_per_transition)
        predicted_dprime = np.full(shape, np.nan, dtype=float)
        predicted_numerator: np.ndarray | None = None
        predicted_denominator: np.ndarray | None = None

        if self.target_kind == "direct_delta":
            values = design.current_dprime + response[:, 0]
            finite = np.isfinite(values)
            predicted_dprime[
                design.transition_positions[finite],
                design.neuron_positions[finite],
            ] = values[finite]
        elif self.target_kind == "two_head":
            numerator = design.current_numerator + response[:, 0]
            with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                denominator = design.current_denominator * np.exp(response[:, 1])
                values = numerator / denominator
            finite = (
                np.isfinite(numerator)
                & np.isfinite(denominator)
                & (denominator > 0)
                & np.isfinite(values)
            )
            predicted_numerator = np.full(shape, np.nan, dtype=float)
            predicted_denominator = np.full(shape, np.nan, dtype=float)
            row = design.transition_positions[finite]
            column = design.neuron_positions[finite]
            predicted_numerator[row, column] = numerator[finite]
            predicted_denominator[row, column] = denominator[finite]
            predicted_dprime[row, column] = values[finite]
        else:  # pragma: no cover - guarded by fit validation
            raise RuntimeError(f"unsupported target kind: {self.target_kind}")

        return NeuronPrediction(
            transition_indices=selected.copy(),
            dprime=predicted_dprime,
            numerator=predicted_numerator,
            denominator=predicted_denominator,
        )

    def predict(
        self,
        cache: Cache,
        indices: Sequence[int] | np.ndarray | None = None,
    ) -> np.ndarray:
        """Return predicted next-window d-prime in cache-neuron layout."""

        return self.predict_components(cache, indices).dprime


def _validated_indices(
    cache: Cache,
    indices: Sequence[int] | np.ndarray | None,
) -> np.ndarray:
    selected = (
        np.arange(len(cache.metadata), dtype=int)
        if indices is None
        else np.asarray(indices, dtype=int)
    )
    if selected.ndim != 1 or len(selected) == 0:
        raise ValueError("indices must be a non-empty one-dimensional sequence")
    if len(np.unique(selected)) != len(selected):
        raise ValueError("indices must not contain duplicates")
    if selected.min() < 0 or selected.max() >= len(cache.metadata):
        raise IndexError("transition index is outside the cache")
    return selected


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _semantic_config(value: object) -> object:
    """Remove location-only fields before constructing a model contract."""

    if isinstance(value, dict):
        return {
            str(key): _semantic_config(item)
            for key, item in value.items()
            if not (
                str(key) == "base_cache"
                or str(key).endswith("_path")
            )
        }
    if isinstance(value, list):
        return [_semantic_config(item) for item in value]
    return value


def _prediction_contract(cache: Cache) -> PredictionContract:
    stable = getattr(cache, "provenance", None)
    refined = getattr(cache, "refined_provenance", None)
    return PredictionContract(
        window_pairs=int(cache.window_pairs),
        stable_history_sha256=(
            str(stable.history_sha256) if stable is not None else None
        ),
        stable_build_config_json=(
            _canonical_json(_semantic_config(stable.build_config))
            if stable is not None
            else None
        ),
        refined_schema_version=(
            int(refined.schema_version) if refined is not None else None
        ),
        refined_builder_sha256=(
            str(refined.builder_sha256) if refined is not None else None
        ),
        refined_config_json=(
            _canonical_json(_semantic_config(refined.config))
            if refined is not None
            else None
        ),
    )


def infer_feature_schema(
    cache: Cache,
    indices: Sequence[int] | np.ndarray,
    *,
    descriptor: DescriptorSpec,
) -> FeatureSchema:
    """Learn categorical levels from one training fold only."""

    selected = _validated_indices(cache, indices)
    if not descriptor.includes(DescriptorStage.ANATOMY):
        return FeatureSchema()
    if cache.area_id is None:
        raise ValueError("anatomical descriptors require cache.area_id")
    broad = tuple(
        sorted(cache.metadata.iloc[selected]["area"].astype(str).unique())
    )
    fine = tuple(
        sorted(
            int(value)
            for value in np.unique(np.asarray(cache.area_id)[selected])
        )
    )
    return FeatureSchema(broad_areas=broad, fine_areas=fine)


def _require_current_denominator(cache: Cache) -> np.ndarray:
    if cache.current_denominator is None:
        raise ValueError("cache.current_denominator is required")
    denominator = np.asarray(cache.current_denominator, dtype=float)
    if denominator.shape != cache.current.shape:
        raise ValueError("current_denominator must align with current d-prime")
    return denominator


def _current_valid_mask(
    cache: Cache,
    transition_index: int,
    denominator: np.ndarray,
) -> np.ndarray:
    current = np.asarray(cache.current[transition_index], dtype=float)
    valid = (
        np.isfinite(current)
        & np.isfinite(denominator[transition_index])
        & (denominator[transition_index] > 0)
    )
    if cache.current_valid_denominator is not None:
        flags = np.asarray(cache.current_valid_denominator[transition_index], dtype=bool)
        if flags.shape != valid.shape:
            raise ValueError("current_valid_denominator must align with current")
        valid &= flags
    return valid


def _leave_one_neuron_out_context(
    values: np.ndarray,
    valid: np.ndarray,
    *,
    threshold: float,
) -> np.ndarray:
    """Return exact LOO q05/q50/q95, SD, and two selective fractions."""

    result = np.full((len(values), 6), np.nan, dtype=float)
    positions = np.flatnonzero(valid)
    observed = np.asarray(values[positions], dtype=float)
    count = len(observed)
    if count < 2:
        return result

    total = float(observed.sum())
    total_squared = float(np.square(observed).sum())
    remaining_count = count - 1
    remaining_mean = (total - observed) / remaining_count
    remaining_second = (total_squared - np.square(observed)) / remaining_count
    remaining_variance = np.maximum(
        remaining_second - np.square(remaining_mean), 0.0
    )
    result[positions, 3] = np.sqrt(remaining_variance)
    result[positions, 4] = (
        np.sum(observed >= threshold) - (observed >= threshold)
    ) / remaining_count
    result[positions, 5] = (
        np.sum(observed <= -threshold) - (observed <= -threshold)
    ) / remaining_count

    order = np.argsort(observed, kind="stable")
    ordered = observed[order]
    removed_rank = np.arange(count)
    for output_column, quantile in enumerate((0.05, 0.5, 0.95)):
        location = quantile * (remaining_count - 1)
        lower_rank = int(np.floor(location))
        upper_rank = int(np.ceil(location))
        fraction = location - lower_rank
        lower_source = lower_rank + (lower_rank >= removed_rank)
        upper_source = upper_rank + (upper_rank >= removed_rank)
        by_removed_rank = (
            (1.0 - fraction) * ordered[lower_source]
            + fraction * ordered[upper_source]
        )
        result[positions[order], output_column] = by_removed_rank
    return result


def _spatial_terms(
    x: np.ndarray,
    y: np.ndarray,
    *,
    degree: int,
) -> tuple[list[np.ndarray], list[str]]:
    columns = [x, y]
    names = ["cortical_x", "cortical_y"]
    if degree == 2:
        columns.extend([np.square(x), x * y, np.square(y)])
        names.extend(
            ["cortical_x2", "cortical_x_by_y", "cortical_y2"]
        )
    return columns, names


def _refined_neuron_values(
    cache: Cache,
    name: str,
    transition_index: int,
    neurons: np.ndarray,
) -> np.ndarray:
    """Read one current-only enriched neuron field with a useful error."""

    value = getattr(cache, name, None)
    if value is None:
        raise ValueError(
            f"descriptor stage requires {name!r}; load the isolated refined "
            "cache with load_refined_cache()"
        )
    array = np.asarray(value)
    if array.shape != cache.current.shape:
        raise ValueError(f"{name} must align with current d-prime")
    return np.asarray(array[transition_index, neurons], dtype=float)


def _refined_transition_value(
    cache: Cache,
    name: str,
    transition_index: int,
) -> float:
    """Read one current-only enriched transition field."""

    value = getattr(cache, name, None)
    if value is None:
        raise ValueError(
            f"descriptor stage requires {name!r}; load the isolated refined "
            "cache with load_refined_cache()"
        )
    array = np.asarray(value)
    expected = (len(cache.metadata),)
    if array.shape != expected:
        raise ValueError(f"{name} must have shape {expected}")
    return float(array[transition_index])


def build_current_design(
    cache: Cache,
    indices: Sequence[int] | np.ndarray,
    *,
    descriptor: DescriptorSpec = DescriptorSpec(),
    schema: FeatureSchema | None = None,
) -> CurrentNeuronDesign:
    """Construct causal neuron rows without consulting any future target."""

    selected = _validated_indices(cache, indices)
    learned_schema = (
        infer_feature_schema(cache, selected, descriptor=descriptor)
        if schema is None
        else schema
    )
    denominator = _require_current_denominator(cache)

    if descriptor.includes(DescriptorStage.ANATOMY) and cache.area_id is None:
        raise ValueError("anatomical descriptors require cache.area_id")
    if descriptor.includes(DescriptorStage.SPATIAL) and (
        cache.cortical_x is None or cache.cortical_y is None
    ):
        raise ValueError("spatial descriptors require cortical x/y")

    feature_rows: list[np.ndarray] = []
    transition_positions: list[np.ndarray] = []
    neuron_positions: list[np.ndarray] = []
    current_rows: list[np.ndarray] = []
    numerator_rows: list[np.ndarray] = []
    denominator_rows: list[np.ndarray] = []
    expected_names: tuple[str, ...] | None = None

    for transition_position, transition_index in enumerate(selected):
        current = np.asarray(cache.current[transition_index], dtype=float)
        current_denominator = denominator[transition_index]
        valid = _current_valid_mask(cache, int(transition_index), denominator)
        neurons = np.flatnonzero(valid)
        if len(neurons) == 0:
            continue

        current_values = current[neurons]
        denominator_values = current_denominator[neurons]
        numerator_values = current_values * denominator_values
        columns: list[np.ndarray] = [
            current_values,
            numerator_values,
            np.log(denominator_values),
        ]
        names = ["current_dprime", "current_numerator", "current_log_denominator"]

        if descriptor.includes(DescriptorStage.ANATOMY):
            broad_area = str(cache.metadata.iloc[transition_index]["area"])
            for area in learned_schema.broad_areas:
                columns.append(
                    np.full(len(neurons), float(broad_area == area), dtype=float)
                )
                names.append(f"broad_area::{area}")
            columns.append(
                np.full(
                    len(neurons),
                    float(broad_area not in learned_schema.broad_areas),
                    dtype=float,
                )
            )
            names.append("broad_area::<unknown>")

            fine_values = np.asarray(cache.area_id[transition_index], dtype=int)[
                neurons
            ]
            for fine_area in learned_schema.fine_areas:
                columns.append((fine_values == fine_area).astype(float))
                names.append(f"fine_area::{fine_area}")
            columns.append(
                (~np.isin(fine_values, learned_schema.fine_areas)).astype(float)
            )
            names.append("fine_area::<unknown>")

        if descriptor.includes(DescriptorStage.SPATIAL):
            x = np.asarray(cache.cortical_x[transition_index], dtype=float)[neurons]
            y = np.asarray(cache.cortical_y[transition_index], dtype=float)[neurons]
            spatial_columns, spatial_names = _spatial_terms(
                x, y, degree=descriptor.spatial_degree
            )
            columns.extend(spatial_columns)
            names.extend(spatial_names)

        if descriptor.includes(DescriptorStage.POPULATION):
            context = _leave_one_neuron_out_context(
                current,
                valid,
                threshold=descriptor.selectivity_threshold,
            )[neurons]
            columns.extend(context[:, index] for index in range(context.shape[1]))
            names.extend(
                [
                    "loo_population_q05",
                    "loo_population_median",
                    "loo_population_q95",
                    "loo_population_sd",
                    "loo_population_frac_leaf",
                    "loo_population_frac_circle",
                ]
            )

        if descriptor.includes(DescriptorStage.RESPONSE):
            mean_leaf = _refined_neuron_values(
                cache, "current_mean_leaf", int(transition_index), neurons
            )
            mean_circle = _refined_neuron_values(
                cache, "current_mean_circle", int(transition_index), neurons
            )
            sd_leaf = _refined_neuron_values(
                cache, "current_sd_leaf", int(transition_index), neurons
            )
            sd_circle = _refined_neuron_values(
                cache, "current_sd_circle", int(transition_index), neurons
            )
            with np.errstate(divide="ignore", invalid="ignore"):
                log_sd_leaf = np.where(sd_leaf > 0, np.log(sd_leaf), np.nan)
                log_sd_circle = np.where(
                    sd_circle > 0, np.log(sd_circle), np.nan
                )
            split_dprime = _refined_neuron_values(
                cache,
                "current_split_dprime_instability",
                int(transition_index),
                neurons,
            )
            split_signal = _refined_neuron_values(
                cache,
                "current_split_signal_instability",
                int(transition_index),
                neurons,
            )
            split_log_denominator = _refined_neuron_values(
                cache,
                "current_split_log_denominator_instability",
                int(transition_index),
                neurons,
            )
            split_valid = _refined_neuron_values(
                cache,
                "current_split_instability_valid",
                int(transition_index),
                neurons,
            )
            columns.extend(
                [
                    mean_leaf,
                    mean_circle,
                    mean_leaf - mean_circle,
                    log_sd_leaf,
                    log_sd_circle,
                    split_dprime,
                    split_signal,
                    split_log_denominator,
                    split_valid,
                ]
            )
            names.extend(
                [
                    "current_mean_leaf",
                    "current_mean_circle",
                    "current_mean_leaf_minus_circle",
                    "current_log_sd_leaf",
                    "current_log_sd_circle",
                    "current_split_dprime_instability",
                    "current_split_signal_instability",
                    "current_split_log_denominator_instability",
                    "current_split_instability_valid",
                ]
            )

        metadata = cache.metadata.iloc[transition_index]
        if descriptor.includes(DescriptorStage.TEMPORAL):
            enriched_window_index = getattr(cache, "current_window_index", None)
            if enriched_window_index is None:
                window_index = (
                    float(metadata["current_start_pair_id"])
                    / float(cache.window_pairs)
                )
            else:
                window_index = _refined_transition_value(
                    cache, "current_window_index", int(transition_index)
                )
            columns.extend(
                [
                    np.full(
                        len(neurons),
                        window_index,
                        dtype=float,
                    ),
                    np.full(
                        len(neurons),
                        float(metadata["current_last_trial_id"]),
                        dtype=float,
                    ),
                ]
            )
            names.extend(["current_window_index", "current_last_trial_id"])

        if descriptor.includes(DescriptorStage.BEHAVIOR):
            behavior_values = [
                float(metadata["current_run_speed"]),
                *(
                    _refined_transition_value(
                        cache, name, int(transition_index)
                    )
                    for name in (
                        "current_run_speed_sd",
                        "current_moving_fraction",
                        "current_trial_duration",
                        "current_leaf_run_speed",
                        "current_circle_run_speed",
                        "current_role_run_speed_difference",
                        "current_elapsed_seconds",
                    )
                ),
            ]
            columns.extend(
                np.full(len(neurons), value, dtype=float)
                for value in behavior_values
            )
            names.extend(
                [
                    "current_run_speed",
                    "current_run_speed_sd",
                    "current_moving_fraction",
                    "current_trial_duration",
                    "current_leaf_run_speed",
                    "current_circle_run_speed",
                    "current_role_run_speed_difference",
                    "current_elapsed_seconds",
                ]
            )

        if descriptor.includes(DescriptorStage.COUPLING):
            columns.extend(
                [
                    _refined_neuron_values(
                        cache,
                        "current_neuron_run_correlation",
                        int(transition_index),
                        neurons,
                    ),
                    _refined_neuron_values(
                        cache,
                        "current_neuron_run_correlation_valid",
                        int(transition_index),
                        neurons,
                    ),
                ]
            )
            names.extend(
                [
                    "current_neuron_run_correlation",
                    "current_neuron_run_correlation_valid",
                ]
            )

        current_names = tuple(names)
        if expected_names is None:
            expected_names = current_names
        elif current_names != expected_names:
            raise RuntimeError("feature names changed between transitions")

        feature_rows.append(np.column_stack(columns))
        transition_positions.append(
            np.full(len(neurons), transition_position, dtype=int)
        )
        neuron_positions.append(neurons.astype(int))
        current_rows.append(current_values)
        numerator_rows.append(numerator_values)
        denominator_rows.append(denominator_values)

    if not feature_rows or expected_names is None:
        raise ValueError("selected transitions have no valid current neuron rows")

    return CurrentNeuronDesign(
        transition_indices=selected.copy(),
        transition_positions=np.concatenate(transition_positions),
        neuron_positions=np.concatenate(neuron_positions),
        features=np.vstack(feature_rows),
        feature_names=expected_names,
        current_dprime=np.concatenate(current_rows),
        current_numerator=np.concatenate(numerator_rows),
        current_denominator=np.concatenate(denominator_rows),
    )


def hierarchical_row_weights(
    cache: Cache,
    transition_indices: Sequence[int] | np.ndarray,
    row_transition_positions: Sequence[int] | np.ndarray,
) -> np.ndarray:
    """Equalize mouse, session/moment, transition, then neuron influence."""

    selected = _validated_indices(cache, transition_indices)
    positions = np.asarray(row_transition_positions, dtype=int)
    if positions.ndim != 1 or len(positions) == 0:
        raise ValueError("row_transition_positions must be a non-empty vector")
    if positions.min() < 0 or positions.max() >= len(selected):
        raise IndexError("a row transition position is outside selected indices")

    global_indices = selected[positions]
    metadata = cache.metadata.iloc[global_indices]
    keys = pd.DataFrame(
        {
            "mouse": metadata["mouse"].astype(str).to_numpy(),
            "session": metadata["behavior_session_id"].astype(str).to_numpy(),
            "moment": metadata["moment"].astype(str).to_numpy(),
            "transition": global_indices,
        }
    )
    rows_per_transition = keys.groupby("transition")["transition"].transform("size")
    transitions_per_session = keys.groupby(
        ["mouse", "session", "moment"]
    )["transition"].transform("nunique")
    sessions_per_mouse = keys.drop_duplicates(
        ["mouse", "session", "moment"]
    ).groupby("mouse").size()
    mouse_sessions = keys["mouse"].map(sessions_per_mouse)
    mouse_count = keys["mouse"].nunique()

    weights = 1.0 / (
        float(mouse_count)
        * mouse_sessions.to_numpy(dtype=float)
        * transitions_per_session.to_numpy(dtype=float)
        * rows_per_transition.to_numpy(dtype=float)
    )
    return weights * (len(weights) / weights.sum())


def _training_response(
    cache: Cache,
    design: CurrentNeuronDesign,
    *,
    target_kind: TargetKind,
) -> tuple[np.ndarray, np.ndarray]:
    if cache.next_denominator is None:
        raise ValueError("cache.next_denominator is required for training")
    next_denominator = np.asarray(cache.next_denominator, dtype=float)
    if next_denominator.shape != cache.future.shape:
        raise ValueError("next_denominator must align with future d-prime")

    global_indices = design.transition_indices[design.transition_positions]
    future = np.asarray(
        cache.future[global_indices, design.neuron_positions], dtype=float
    )
    future_denominator = next_denominator[
        global_indices, design.neuron_positions
    ]
    valid = (
        np.isfinite(future)
        & np.isfinite(future_denominator)
        & (future_denominator > 0)
    )
    if cache.next_valid_denominator is not None:
        flags = np.asarray(
            cache.next_valid_denominator[
                global_indices, design.neuron_positions
            ],
            dtype=bool,
        )
        valid &= flags

    if target_kind == "direct_delta":
        response = (future - design.current_dprime)[:, None]
    elif target_kind == "two_head":
        future_numerator = future * future_denominator
        response = np.column_stack(
            [
                future_numerator - design.current_numerator,
                np.log(future_denominator) - np.log(design.current_denominator),
            ]
        )
    else:
        raise ValueError(f"unknown target kind: {target_kind}")
    valid &= np.isfinite(response).all(axis=1)
    return response, valid


def _weighted_feature_fill(
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


def _fill_nonfinite(values: np.ndarray, fill: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=float).copy()
    if result.ndim != 2 or fill.shape != (result.shape[1],):
        raise ValueError("fill values must align with feature columns")
    bad = ~np.isfinite(result)
    if bad.any():
        result[bad] = np.broadcast_to(fill, result.shape)[bad]
    return result


def fit_refined_neuron_model(
    cache: Cache,
    indices: Sequence[int] | np.ndarray,
    *,
    target_kind: TargetKind,
    descriptor: DescriptorSpec = DescriptorSpec(),
    alpha: float,
) -> FittedRefinedNeuronModel:
    """Fit one weighted ridge using only the supplied training transitions."""

    if target_kind not in ("direct_delta", "two_head"):
        raise ValueError(f"unknown target kind: {target_kind}")
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be finite and positive")

    selected = _validated_indices(cache, indices)
    schema = infer_feature_schema(cache, selected, descriptor=descriptor)
    design = build_current_design(
        cache,
        selected,
        descriptor=descriptor,
        schema=schema,
    )
    response, valid_target = _training_response(
        cache, design, target_kind=target_kind
    )
    if not valid_target.any():
        raise ValueError("training transitions have no valid future neuron targets")

    row_positions = design.transition_positions[valid_target]
    weights = hierarchical_row_weights(cache, selected, row_positions)
    features = design.features[valid_target]
    response = response[valid_target]
    fill_values = _weighted_feature_fill(features, weights)
    features = _fill_nonfinite(features, fill_values)

    # Weighted one-hot columns can be exactly constant.  Older sklearn
    # releases may emit a harmless sqrt warning from round-off just below
    # zero while correctly replacing those scales with one.
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
    training_global_indices = selected[row_positions]
    training_mice = tuple(
        sorted(
            cache.metadata.iloc[np.unique(training_global_indices)]["mouse"]
            .astype(str)
            .unique()
        )
    )
    training_areas = tuple(
        sorted(
            cache.metadata.iloc[selected]["area"].astype(str).unique()
        )
    )
    return FittedRefinedNeuronModel(
        target_kind=target_kind,
        descriptor=descriptor,
        schema=schema,
        alpha=float(alpha),
        feature_names=design.feature_names,
        feature_fill_values=fill_values,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        ridge=ridge,
        training_mice=training_mice,
        training_areas=training_areas,
        prediction_contract=_prediction_contract(cache),
    )


__all__ = [
    "DESCRIPTOR_STAGES",
    "CurrentNeuronDesign",
    "DescriptorSpec",
    "DescriptorStage",
    "FeatureSchema",
    "FittedRefinedNeuronModel",
    "NeuronPrediction",
    "PredictionContract",
    "TargetKind",
    "build_current_design",
    "fit_refined_neuron_model",
    "hierarchical_row_weights",
    "infer_feature_schema",
]
