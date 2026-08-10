#!/usr/bin/env python3
"""Future-blind soft clusters for same-neuron next-window forecasting.

This is a secondary, within-session analysis.  Hyperparameters are selected
only with unsupervised mice.  A nested leave-one-mouse-out (LOMO) estimate
measures the complete selection procedure, after which one configuration per
area is fit to all unsupervised mice and transferred once to supervised mice.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable, Sequence

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "8")

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


AREAS = ("V1", "mHV", "lHV", "aHV")
CLUSTER_GRID = (1, 2, 4, 8)
ALPHA_GRID = (1.0, 10.0, 100.0, 1000.0)
EXPECTED_CACHE_SCHEMA = 3


@dataclass(frozen=True)
class CacheProvenance:
    schema_version: int
    history_sha256: str
    history_path: str
    chronology_algorithm: str
    build_config: dict[str, object]


@dataclass
class Cache:
    metadata: pd.DataFrame
    current: np.ndarray
    future: np.ndarray
    cortical_x: np.ndarray
    cortical_y: np.ndarray
    window_pairs: int
    neuron_id: np.ndarray | None = None
    area_id: np.ndarray | None = None
    cortical_x_raw: np.ndarray | None = None
    cortical_y_raw: np.ndarray | None = None
    current_denominator: np.ndarray | None = None
    next_denominator: np.ndarray | None = None
    current_valid_denominator: np.ndarray | None = None
    next_valid_denominator: np.ndarray | None = None
    provenance: CacheProvenance | None = None

    @property
    def neurons_per_transition(self) -> int:
        return int(self.current.shape[1])


@dataclass(frozen=True, order=True)
class HyperParameters:
    clusters: int
    alpha: float


@dataclass
class SoftRepresentation:
    scaler: StandardScaler
    mixture: GaussianMixture
    lower: np.ndarray
    upper: np.ndarray
    training_rows_per_mouse: int


@dataclass
class CurrentTransitionSummary:
    features: np.ndarray
    membership: np.ndarray
    current_cluster_mean: np.ndarray


@dataclass
class TransitionSummary:
    current: CurrentTransitionSummary
    target_delta: np.ndarray


@dataclass
class Forecaster:
    feature_scaler: StandardScaler
    target_scaler: StandardScaler
    ridge: Ridge
    alpha: float


@dataclass
class ClusterPipeline:
    representation: SoftRepresentation
    forecaster: Forecaster
    parameters: HyperParameters


@dataclass
class EvaluationResult:
    candidate_scores: pd.DataFrame
    nested_scores: pd.DataFrame
    transfer_scores: pd.DataFrame
    selections: pd.DataFrame
    scores: pd.DataFrame
    summary: pd.DataFrame
    paired_tests: pd.DataFrame


RepresentationKey = tuple[str, tuple[str, ...], int, int]
ForecasterKey = tuple[str, tuple[str, ...], int, int, float]


def stable_seed(*parts: object) -> int:
    """Return a deterministic seed; unlike ``hash()``, this is process-stable."""

    digest = hashlib.blake2b(
        "\x1f".join(map(str, parts)).encode("utf-8"), digest_size=8
    ).digest()
    return int.from_bytes(digest, "little") % (2**32 - 1)


def _archive_scalar(archive: np.lib.npyio.NpzFile, name: str) -> object:
    if name not in archive.files:
        raise ValueError(f"cache is missing {name!r}")
    return np.asarray(archive[name]).item()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_cache(cache: Cache) -> None:
    transition_count = len(cache.metadata)
    if cache.current.ndim != 2 or cache.future.shape != cache.current.shape:
        raise ValueError("current and future arrays must be aligned 2-D matrices")
    if cache.current.shape[0] != transition_count:
        raise ValueError("metadata and neuronal arrays have different row counts")
    for name in (
        "cortical_x",
        "cortical_y",
        "neuron_id",
        "area_id",
        "cortical_x_raw",
        "cortical_y_raw",
    ):
        value = getattr(cache, name)
        if value is None or value.shape != cache.current.shape:
            raise ValueError(f"{name} must align with current_dprime")
    for name in (
        "current_denominator",
        "next_denominator",
        "current_valid_denominator",
        "next_valid_denominator",
    ):
        value = getattr(cache, name)
        if value is not None and value.shape != cache.current.shape:
            raise ValueError(f"{name} must align with current_dprime")
    if cache.metadata.empty:
        raise ValueError("cache has no transitions")
    required = {
        "recording_id",
        "mouse",
        "cohort",
        "area",
        "current_progress",
        "current_run_speed",
        "current_last_trial_id",
        "next_first_trial_id",
    }
    missing = required - set(cache.metadata.columns)
    if missing:
        raise ValueError(f"cache metadata is missing {sorted(missing)}")
    chronological = (
        cache.metadata["current_last_trial_id"].to_numpy(dtype=np.int64)
        < cache.metadata["next_first_trial_id"].to_numpy(dtype=np.int64)
    )
    if not bool(chronological.all()):
        raise ValueError("cache contains non-chronological current/next windows")
    for row in np.asarray(cache.neuron_id):
        if len(np.unique(row)) != len(row):
            raise ValueError("neuron_id must be unique within every transition")


def load_cache(path: Path, *, verify_source_history: bool = True) -> Cache:
    """Load schema-v2 cache and validate chronology, alignment, and provenance."""

    metadata_columns = (
        "behavior_session_id",
        "recording_id",
        "mouse",
        "cohort",
        "moment",
        "area",
        "current_start_pair_id",
        "next_start_pair_id",
        "current_first_trial_id",
        "current_last_trial_id",
        "next_first_trial_id",
        "next_last_trial_id",
        "current_progress",
        "current_run_speed",
    )
    try:
        with np.load(path, allow_pickle=False) as archive:
            schema = int(_archive_scalar(archive, "cache_schema_version"))
            if schema != EXPECTED_CACHE_SCHEMA:
                raise ValueError(
                    f"cache schema {schema} is stale; expected {EXPECTED_CACHE_SCHEMA}"
                )
            metadata = pd.DataFrame(
                {column: archive[column] for column in metadata_columns}
            )
            provenance = CacheProvenance(
                schema_version=schema,
                history_sha256=str(_archive_scalar(archive, "source_history_sha256")),
                history_path=str(_archive_scalar(archive, "source_history_path")),
                chronology_algorithm=str(
                    _archive_scalar(archive, "chronology_algorithm")
                ),
                build_config=json.loads(
                    str(_archive_scalar(archive, "build_config_json"))
                ),
            )
            cache = Cache(
                metadata=metadata,
                current=archive["current_dprime"].astype(float),
                future=archive["next_dprime"].astype(float),
                cortical_x=archive["cortical_x"].astype(float),
                cortical_y=archive["cortical_y"].astype(float),
                window_pairs=int(_archive_scalar(archive, "window_pairs")),
                neuron_id=archive["neuron_id"].astype(np.int64),
                area_id=archive["area_id"].astype(np.int64),
                cortical_x_raw=archive["cortical_x_raw"].astype(float),
                cortical_y_raw=archive["cortical_y_raw"].astype(float),
                current_denominator=archive["current_denominator"].astype(float),
                next_denominator=archive["next_denominator"].astype(float),
                current_valid_denominator=archive[
                    "current_valid_denominator"
                ].astype(bool),
                next_valid_denominator=archive["next_valid_denominator"].astype(bool),
                provenance=provenance,
            )
    except KeyError as error:
        raise ValueError(f"cache {path} is incomplete: missing {error}") from error
    _validate_cache(cache)
    if verify_source_history:
        source = Path(cache.provenance.history_path)  # type: ignore[union-attr]
        if source.is_file():
            actual_sha = _file_sha256(source)
            expected_sha = cache.provenance.history_sha256  # type: ignore[union-attr]
            if actual_sha != expected_sha:
                raise ValueError(
                    f"cache {path} is stale: source history digest has changed"
                )
    return cache


def _features(cache: Cache, indices: np.ndarray) -> np.ndarray:
    return np.column_stack(
        [
            cache.current[indices].reshape(-1),
            cache.cortical_x[indices].reshape(-1),
            cache.cortical_y[indices].reshape(-1),
        ]
    )


def transition_mouse_weights(cache: Cache, indices: np.ndarray) -> np.ndarray:
    """Give mice, then behavior-session/moments, equal total influence."""

    rows = cache.metadata.iloc[indices][
        ["mouse", "behavior_session_id", "moment"]
    ].copy()
    if rows.empty:
        raise ValueError("training indices must not be empty")
    session_columns = ["mouse", "behavior_session_id", "moment"]
    transition_counts = rows.groupby(session_columns)["mouse"].transform("size")
    sessions_per_mouse = (
        rows.drop_duplicates(session_columns).groupby("mouse").size()
    )
    mouse_session_counts = rows["mouse"].map(sessions_per_mouse)
    weights = 1.0 / (
        transition_counts.to_numpy(dtype=float)
        * mouse_session_counts.to_numpy(dtype=float)
    )
    return weights * (len(weights) / weights.sum())


def fit_representation(
    cache: Cache,
    indices: np.ndarray,
    *,
    clusters: int,
    rows_per_mouse: int,
    seed: int,
) -> SoftRepresentation:
    """Fit current-window membership features with exact mouse balance."""

    if clusters < 1 or rows_per_mouse < 1:
        raise ValueError("clusters and rows_per_mouse must be positive")
    mice = cache.metadata.iloc[indices]["mouse"].astype(str).to_numpy()
    unique_mice = sorted(np.unique(mice))
    if not unique_mice:
        raise ValueError("representation needs at least one mouse")
    mouse_features: dict[str, np.ndarray] = {}
    for mouse in unique_mice:
        values = _features(cache, indices[mice == mouse])
        mouse_features[mouse] = values[np.isfinite(values).all(axis=1)]
    available = [len(mouse_features[mouse]) for mouse in unique_mice]
    common_rows = min(rows_per_mouse, *available)
    if common_rows < clusters:
        raise ValueError(
            f"only {common_rows} rows per mouse are available for K={clusters}"
        )
    sampled: list[np.ndarray] = []
    for mouse in unique_mice:
        values = mouse_features[mouse]
        rng = np.random.default_rng(stable_seed(seed, "gmm-rows", mouse))
        selected = rng.choice(len(values), size=common_rows, replace=False)
        sampled.append(values[selected])
    training = np.vstack(sampled)
    lower, upper = np.quantile(training, [0.005, 0.995], axis=0)
    clipped = np.clip(training, lower, upper)
    scaler = StandardScaler().fit(clipped)
    mixture = GaussianMixture(
        n_components=clusters,
        covariance_type="diag",
        reg_covar=1e-5,
        n_init=2,
        max_iter=150,
        random_state=stable_seed(seed, "gmm-fit", clusters),
    ).fit(scaler.transform(clipped))
    return SoftRepresentation(scaler, mixture, lower, upper, common_rows)


def memberships(
    cache: Cache,
    index: int,
    representation: SoftRepresentation,
) -> np.ndarray:
    """Return memberships using only current d-prime and static retinotopy."""

    values = np.column_stack(
        [cache.current[index], cache.cortical_x[index], cache.cortical_y[index]]
    )
    valid = np.isfinite(values).all(axis=1)
    probability = np.full(
        (len(values), representation.mixture.n_components), np.nan, dtype=float
    )
    clipped = np.clip(values[valid], representation.lower, representation.upper)
    if len(clipped):
        probability[valid] = representation.mixture.predict_proba(
            representation.scaler.transform(clipped)
        )
    return probability


def summarize_current_transition(
    cache: Cache,
    index: int,
    representation: SoftRepresentation,
) -> CurrentTransitionSummary:
    """Summarize a transition without reading its future response."""

    probability = memberships(cache, index, representation)
    valid = np.isfinite(probability).all(axis=1) & np.isfinite(cache.current[index])
    if not valid.any():
        raise ValueError(f"transition {index} has no valid current neurons")
    observed_probability = probability[valid]
    observed_current = cache.current[index, valid]
    weight = observed_probability.sum(axis=0)
    denominator = np.maximum(weight, 1e-8)
    current_mean = (observed_probability.T @ observed_current) / denominator
    composition = weight / np.maximum(weight.sum(), 1e-8)
    context = cache.metadata.iloc[index]
    features = np.concatenate(
        [
            composition,
            current_mean,
            np.asarray(
                [context["current_progress"], context["current_run_speed"]],
                dtype=float,
            ),
        ]
    )
    return CurrentTransitionSummary(features, probability, current_mean)


def summarize_transition(
    cache: Cache,
    index: int,
    representation: SoftRepresentation,
) -> TransitionSummary:
    """Add a training target to the strictly current-only summary."""

    current = summarize_current_transition(cache, index, representation)
    valid = (
        np.isfinite(current.membership).all(axis=1)
        & np.isfinite(cache.current[index])
        & np.isfinite(cache.future[index])
    )
    if not valid.any():
        raise ValueError(f"transition {index} has no jointly valid neurons")
    probability = current.membership[valid]
    weight = probability.sum(axis=0)
    denominator = np.maximum(weight, 1e-8)
    future_mean = (probability.T @ cache.future[index, valid]) / denominator
    aligned_current_mean = (probability.T @ cache.current[index, valid]) / denominator
    return TransitionSummary(current, future_mean - aligned_current_mean)


def fit_forecaster(
    cache: Cache,
    indices: np.ndarray,
    representation: SoftRepresentation,
    *,
    alpha: float,
) -> Forecaster:
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    summaries = [
        summarize_transition(cache, int(index), representation) for index in indices
    ]
    features = np.stack([item.current.features for item in summaries])
    target = np.stack([item.target_delta for item in summaries])
    weights = transition_mouse_weights(cache, indices)
    feature_scaler = StandardScaler().fit(features, sample_weight=weights)
    target_scaler = StandardScaler().fit(target, sample_weight=weights)
    ridge = Ridge(alpha=float(alpha)).fit(
        feature_scaler.transform(features),
        target_scaler.transform(target),
        sample_weight=weights,
    )
    return Forecaster(feature_scaler, target_scaler, ridge, float(alpha))


def predict_transition(
    cache: Cache,
    index: int,
    pipeline: ClusterPipeline,
) -> np.ndarray:
    summary = summarize_current_transition(cache, index, pipeline.representation)
    standardized = np.asarray(
        pipeline.forecaster.ridge.predict(
            pipeline.forecaster.feature_scaler.transform(summary.features[None, :])
        ),
        dtype=float,
    ).reshape(1, -1)
    cluster_delta = pipeline.forecaster.target_scaler.inverse_transform(standardized)[0]
    prediction = np.full(cache.neurons_per_transition, np.nan, dtype=float)
    valid = np.isfinite(summary.membership).all(axis=1) & np.isfinite(
        cache.current[index]
    )
    prediction[valid] = (
        cache.current[index, valid] + summary.membership[valid] @ cluster_delta
    )
    return prediction


def fit_pipeline(
    cache: Cache,
    indices: np.ndarray,
    parameters: HyperParameters,
    *,
    rows_per_mouse: int,
    seed_parts: Sequence[object],
) -> ClusterPipeline:
    seed = stable_seed("soft-cluster", *seed_parts, parameters.clusters)
    representation = fit_representation(
        cache,
        indices,
        clusters=parameters.clusters,
        rows_per_mouse=rows_per_mouse,
        seed=seed,
    )
    forecaster = fit_forecaster(
        cache, indices, representation, alpha=parameters.alpha
    )
    return ClusterPipeline(representation, forecaster, parameters)


def _cached_pipeline(
    cache: Cache,
    indices: np.ndarray,
    parameters: HyperParameters,
    *,
    area: str,
    rows_per_mouse: int,
    representation_cache: dict[RepresentationKey, SoftRepresentation],
    forecaster_cache: dict[ForecasterKey, Forecaster],
) -> ClusterPipeline:
    """Reuse fits for the same training-mouse set across nested folds."""

    training_mice = tuple(
        sorted(cache.metadata.iloc[indices]["mouse"].astype(str).unique())
    )
    representation_key: RepresentationKey = (
        area,
        training_mice,
        parameters.clusters,
        rows_per_mouse,
    )
    if representation_key not in representation_cache:
        representation_cache[representation_key] = fit_representation(
            cache,
            indices,
            clusters=parameters.clusters,
            rows_per_mouse=rows_per_mouse,
            seed=stable_seed("representation", *representation_key),
        )
    representation = representation_cache[representation_key]
    forecaster_key: ForecasterKey = (*representation_key, parameters.alpha)
    if forecaster_key not in forecaster_cache:
        forecaster_cache[forecaster_key] = fit_forecaster(
            cache, indices, representation, alpha=parameters.alpha
        )
    return ClusterPipeline(
        representation, forecaster_cache[forecaster_key], parameters
    )


def _training_scale(cache: Cache, indices: np.ndarray) -> float:
    """Equal-session, then equal-mouse future SD used for normalization."""

    rows = cache.metadata.iloc[indices]
    mouse_scales = []
    for mouse in sorted(rows["mouse"].astype(str).unique()):
        mouse_indices = indices[rows["mouse"].astype(str).eq(mouse).to_numpy()]
        mouse_rows = cache.metadata.iloc[mouse_indices]
        session_scales = []
        for _, group in mouse_rows.groupby(["behavior_session_id", "moment"]):
            session_indices = group.index.to_numpy(dtype=int)
            values = cache.future[session_indices].reshape(-1)
            values = values[np.isfinite(values)]
            if len(values) > 1:
                session_scales.append(float(np.std(values, ddof=1)))
        if session_scales:
            mouse_scales.append(float(np.mean(session_scales)))
    scale = float(np.mean(mouse_scales)) if mouse_scales else 1.0
    return scale if np.isfinite(scale) and scale > 1e-8 else 1.0


def _score_predictions(
    cache: Cache,
    indices: np.ndarray,
    predictions: Iterable[np.ndarray],
    *,
    scale: float,
) -> pd.DataFrame:
    predictions_array = np.stack(list(predictions))
    transition_errors = []
    valid_neuron_counts = []
    for position, index in enumerate(indices):
        valid = np.isfinite(cache.future[index]) & np.isfinite(
            predictions_array[position]
        )
        valid_neuron_counts.append(int(valid.sum()))
        transition_errors.append(
            float(
                np.sqrt(
                    np.mean(
                        np.square(
                            cache.future[index, valid]
                            - predictions_array[position, valid]
                        )
                    )
                )
                / scale
            )
            if valid.any()
            else np.nan
        )
    frame = cache.metadata.iloc[indices][
        ["mouse", "behavior_session_id", "moment"]
    ].reset_index(drop=True)
    frame["nrmse"] = transition_errors
    frame["n_valid_neurons"] = valid_neuron_counts
    # A session with more valid windows must not receive more weight.  First
    # average transitions inside behavior-session/moment, then average those
    # session values inside mouse.
    sessions = (
        frame.groupby(
            ["mouse", "behavior_session_id", "moment"], as_index=False
        )
        .agg(
            n_transitions=("nrmse", "size"),
            min_valid_neurons=("n_valid_neurons", "min"),
            session_nrmse=("nrmse", "mean"),
        )
    )
    return (
        sessions.groupby("mouse", as_index=False)
        .agg(
            n_sessions=("behavior_session_id", "size"),
            n_transitions=("n_transitions", "sum"),
            min_valid_neurons=("min_valid_neurons", "min"),
            nrmse=("session_nrmse", "mean"),
        )
        .sort_values("mouse")
    )


def _score_pipeline(
    cache: Cache,
    train: np.ndarray,
    validation: np.ndarray,
    parameters: HyperParameters,
    *,
    area: str,
    rows_per_mouse: int,
    representation_cache: dict[RepresentationKey, SoftRepresentation],
    forecaster_cache: dict[ForecasterKey, Forecaster],
) -> pd.DataFrame:
    pipeline = _cached_pipeline(
        cache,
        train,
        parameters,
        area=area,
        rows_per_mouse=rows_per_mouse,
        representation_cache=representation_cache,
        forecaster_cache=forecaster_cache,
    )
    return _score_predictions(
        cache,
        validation,
        (predict_transition(cache, int(index), pipeline) for index in validation),
        scale=_training_scale(cache, train),
    )


def _score_persistence(
    cache: Cache, train: np.ndarray, validation: np.ndarray
) -> pd.DataFrame:
    return _score_predictions(
        cache,
        validation,
        (cache.current[index] for index in validation),
        scale=_training_scale(cache, train),
    )


def _indices_for_mouse(cache: Cache, pool: np.ndarray, mouse: str) -> np.ndarray:
    return pool[
        cache.metadata.iloc[pool]["mouse"].astype(str).eq(mouse).to_numpy()
    ]


def _candidate_lomo(
    cache: Cache,
    pool: np.ndarray,
    *,
    area: str,
    cluster_grid: Sequence[int],
    alpha_grid: Sequence[float],
    rows_per_mouse: int,
    representation_cache: dict[RepresentationKey, SoftRepresentation],
    forecaster_cache: dict[ForecasterKey, Forecaster],
) -> pd.DataFrame:
    """Score every candidate using unsupervised mice only."""

    rows: list[dict[str, object]] = []
    mice = tuple(sorted(cache.metadata.iloc[pool]["mouse"].astype(str).unique()))
    if len(mice) < 2:
        raise ValueError("LOMO selection needs at least two mice")
    for held_mouse in mice:
        validation = _indices_for_mouse(cache, pool, held_mouse)
        train = pool[~np.isin(pool, validation)]
        scale = _training_scale(cache, train)
        persistence = _score_persistence(cache, train, validation).iloc[0]
        rows.append(
            {
                "area": area,
                "clusters": 0,
                "alpha": np.nan,
                "model": "neuron_persistence",
                "mouse": held_mouse,
                "n_sessions": int(persistence["n_sessions"]),
                "n_transitions": int(persistence["n_transitions"]),
                "min_valid_neurons": int(persistence["min_valid_neurons"]),
                "nrmse": float(persistence["nrmse"]),
            }
        )
        for clusters in cluster_grid:
            for alpha in alpha_grid:
                parameters = HyperParameters(int(clusters), float(alpha))
                pipeline = _cached_pipeline(
                    cache,
                    train,
                    parameters,
                    area=area,
                    rows_per_mouse=rows_per_mouse,
                    representation_cache=representation_cache,
                    forecaster_cache=forecaster_cache,
                )
                scored = _score_predictions(
                    cache,
                    validation,
                    (
                        predict_transition(cache, int(index), pipeline)
                        for index in validation
                    ),
                    scale=scale,
                ).iloc[0]
                rows.append(
                    {
                        "area": area,
                        "clusters": parameters.clusters,
                        "alpha": parameters.alpha,
                        "model": "soft_cluster_delta_ridge",
                        "mouse": held_mouse,
                        "n_sessions": int(scored["n_sessions"]),
                        "n_transitions": int(scored["n_transitions"]),
                        "min_valid_neurons": int(scored["min_valid_neurons"]),
                        "nrmse": float(scored["nrmse"]),
                    }
                )
    return pd.DataFrame.from_records(rows)


def select_configuration(
    candidate_scores: pd.DataFrame,
    *,
    tolerance: float = 0.02,
) -> tuple[HyperParameters, float, float]:
    """Choose the simplest candidate within ``tolerance`` of minimum error."""

    candidates = candidate_scores.loc[
        candidate_scores["model"].eq("soft_cluster_delta_ridge")
    ]
    aggregate = (
        candidates.groupby(["clusters", "alpha"], as_index=False)
        .agg(mean_nrmse=("nrmse", "mean"), sem_nrmse=("nrmse", "sem"))
        .sort_values(["mean_nrmse", "clusters", "alpha"])
    )
    if aggregate.empty:
        raise ValueError("no candidate scores are available")
    best = float(aggregate["mean_nrmse"].min())
    eligible = aggregate.loc[aggregate["mean_nrmse"] <= best * (1.0 + tolerance)]
    # Prefer fewer clusters, then stronger regularization, within the tolerance.
    selected = eligible.sort_values(
        ["clusters", "alpha"], ascending=[True, False]
    ).iloc[0]
    return (
        HyperParameters(int(selected["clusters"]), float(selected["alpha"])),
        float(selected["mean_nrmse"]),
        float(selected["sem_nrmse"]),
    )


def _decorate_scores(
    frame: pd.DataFrame,
    *,
    split: str,
    selection_scope: str,
) -> pd.DataFrame:
    result = frame.copy()
    result["split"] = split
    result["selection_scope"] = selection_scope
    return result


def _nested_lomo(
    cache: Cache,
    pool: np.ndarray,
    *,
    area: str,
    cluster_grid: Sequence[int],
    alpha_grid: Sequence[float],
    rows_per_mouse: int,
    tolerance: float,
    representation_cache: dict[RepresentationKey, SoftRepresentation],
    forecaster_cache: dict[ForecasterKey, Forecaster],
    verbose: bool = False,
) -> pd.DataFrame:
    """Estimate selection + refitting with an untouched outer mouse."""

    rows: list[dict[str, object]] = []
    mice = tuple(sorted(cache.metadata.iloc[pool]["mouse"].astype(str).unique()))
    if len(mice) < 3:
        raise ValueError("nested LOMO needs at least three unsupervised mice")
    for outer_mouse in mice:
        outer_validation = _indices_for_mouse(cache, pool, outer_mouse)
        outer_train = pool[~np.isin(pool, outer_validation)]
        inner_scores = _candidate_lomo(
            cache,
            outer_train,
            area=area,
            cluster_grid=cluster_grid,
            alpha_grid=alpha_grid,
            rows_per_mouse=rows_per_mouse,
            representation_cache=representation_cache,
            forecaster_cache=forecaster_cache,
        )
        selected, inner_error, _ = select_configuration(
            inner_scores, tolerance=tolerance
        )
        selected_score = _score_pipeline(
            cache,
            outer_train,
            outer_validation,
            selected,
            area=area,
            rows_per_mouse=rows_per_mouse,
            representation_cache=representation_cache,
            forecaster_cache=forecaster_cache,
        ).iloc[0]
        rows.append(
            {
                "area": area,
                "clusters": selected.clusters,
                "alpha": selected.alpha,
                "model": "nested_selected_soft_cluster",
                "mouse": outer_mouse,
                "n_sessions": int(selected_score["n_sessions"]),
                "n_transitions": int(selected_score["n_transitions"]),
                "min_valid_neurons": int(selected_score["min_valid_neurons"]),
                "nrmse": float(selected_score["nrmse"]),
                "inner_cv_nrmse": inner_error,
            }
        )
        persistence = _score_persistence(
            cache, outer_train, outer_validation
        ).iloc[0]
        rows.append(
            {
                "area": area,
                "clusters": 0,
                "alpha": np.nan,
                "model": "neuron_persistence",
                "mouse": outer_mouse,
                "n_sessions": int(persistence["n_sessions"]),
                "n_transitions": int(persistence["n_transitions"]),
                "min_valid_neurons": int(persistence["min_valid_neurons"]),
                "nrmse": float(persistence["nrmse"]),
                "inner_cv_nrmse": np.nan,
            }
        )
        if verbose:
            print(
                f"  outer {outer_mouse}: K={selected.clusters}, "
                f"alpha={selected.alpha:g}, nRMSE={float(selected_score['nrmse']):.4f}",
                flush=True,
            )
    return pd.DataFrame.from_records(rows)


def _summary(scores: pd.DataFrame) -> pd.DataFrame:
    grouped_scores = scores.copy()
    nested = grouped_scores["model"].eq("nested_selected_soft_cluster")
    # K/alpha legitimately vary across outer folds.  Aggregate the complete
    # nested procedure as one model; fold-specific choices remain in scores.
    grouped_scores.loc[nested, "clusters"] = -1
    grouped_scores.loc[nested, "alpha"] = np.nan
    summary = (
        grouped_scores.groupby(
            ["area", "split", "model", "clusters", "alpha"],
            as_index=False,
            dropna=False,
        )
        .agg(
            mice=("mouse", "nunique"),
            mean_nrmse=("nrmse", "mean"),
            sem_nrmse=("nrmse", "sem"),
        )
        .sort_values(["area", "split", "model", "clusters", "alpha"])
    )
    persistence = summary.loc[
        summary["model"].eq("neuron_persistence"),
        ["area", "split", "mean_nrmse"],
    ].rename(columns={"mean_nrmse": "persistence_nrmse"})
    summary = summary.merge(persistence, on=["area", "split"], how="left")
    summary["improvement_vs_persistence_pct"] = 100.0 * (
        summary["persistence_nrmse"] - summary["mean_nrmse"]
    ) / summary["persistence_nrmse"]
    return summary


def _paired_sign_flip(values: np.ndarray) -> tuple[float, float, int]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan, 0
    observed = abs(float(np.mean(values)))
    permutations = 1 << len(values)
    extreme = 0
    for mask in range(permutations):
        signs = np.asarray(
            [1.0 if mask & (1 << index) else -1.0 for index in range(len(values))]
        )
        if abs(float(np.mean(values * signs))) >= observed - 1e-15:
            extreme += 1
    return float(np.mean(values)), extreme / permutations, permutations


def paired_model_tests(scores: pd.DataFrame) -> pd.DataFrame:
    """Exact mouse-level learned-minus-persistence comparisons."""

    rows: list[dict[str, object]] = []
    comparisons = (
        ("unsupervised_nested_lomo", "nested_selected_soft_cluster"),
        ("supervised_transfer", "final_selected_soft_cluster"),
    )
    for split, learned_model in comparisons:
        selected = scores.loc[scores["split"].eq(split)]
        for area, group in selected.groupby("area"):
            paired = group.pivot(index="mouse", columns="model", values="nrmse")
            difference = (
                paired[learned_model] - paired["neuron_persistence"]
            ).dropna()
            effect, pvalue, permutations = _paired_sign_flip(difference.to_numpy())
            rows.append(
                {
                    "area": area,
                    "split": split,
                    "learned_model": learned_model,
                    "contrast": "learned_minus_persistence_nrmse",
                    "mice": len(difference),
                    "mean_difference": effect,
                    "exact_two_sided_pvalue": pvalue,
                    "permutations": permutations,
                }
            )
    return pd.DataFrame.from_records(rows).sort_values(["split", "area"])


def evaluate(
    cache: Cache,
    *,
    cluster_grid: Sequence[int] = CLUSTER_GRID,
    alpha_grid: Sequence[float] = ALPHA_GRID,
    rows_per_mouse: int = 1000,
    selection_tolerance: float = 0.02,
    areas: Sequence[str] = AREAS,
    verbose: bool = False,
) -> EvaluationResult:
    """Run unsupervised-only selection, nested validation, then transfer."""

    _validate_cache(cache)
    cluster_grid = tuple(sorted({int(value) for value in cluster_grid}))
    alpha_grid = tuple(sorted({float(value) for value in alpha_grid}))
    if not cluster_grid or min(cluster_grid) < 1:
        raise ValueError("cluster_grid must contain positive integers")
    if not alpha_grid or min(alpha_grid) <= 0:
        raise ValueError("alpha_grid must contain positive values")
    if selection_tolerance < 0:
        raise ValueError("selection_tolerance must be non-negative")

    candidate_parts: list[pd.DataFrame] = []
    nested_parts: list[pd.DataFrame] = []
    transfer_parts: list[pd.DataFrame] = []
    selections: list[dict[str, object]] = []
    metadata = cache.metadata

    for area in areas:
        if verbose:
            print(f"{area}: unsupervised candidate selection", flush=True)
        representation_cache: dict[RepresentationKey, SoftRepresentation] = {}
        forecaster_cache: dict[ForecasterKey, Forecaster] = {}
        area_indices = np.flatnonzero(metadata["area"].eq(area).to_numpy())
        unsupervised = area_indices[
            metadata.iloc[area_indices]["cohort"].eq("unsupervised").to_numpy()
        ]
        supervised = area_indices[
            metadata.iloc[area_indices]["cohort"].eq("supervised").to_numpy()
        ]
        if len(unsupervised) == 0 or len(supervised) == 0:
            raise ValueError(f"{area} needs both unsupervised and supervised transitions")

        candidates = _candidate_lomo(
            cache,
            unsupervised,
            area=area,
            cluster_grid=cluster_grid,
            alpha_grid=alpha_grid,
            rows_per_mouse=rows_per_mouse,
            representation_cache=representation_cache,
            forecaster_cache=forecaster_cache,
        )
        selected, cv_error, cv_sem = select_configuration(
            candidates, tolerance=selection_tolerance
        )
        candidate_parts.append(
            _decorate_scores(
                candidates,
                split="unsupervised_selection_lomo",
                selection_scope="candidate_comparison",
            )
        )
        selections.append(
            {
                "area": area,
                "clusters": selected.clusters,
                "alpha": selected.alpha,
                "unsupervised_cv_nrmse": cv_error,
                "unsupervised_cv_sem": cv_sem,
                "selection_tolerance": selection_tolerance,
            }
        )
        if verbose:
            print(
                f"{area}: locked K={selected.clusters}, alpha={selected.alpha:g}; "
                "nested LOMO",
                flush=True,
            )

        nested = _nested_lomo(
            cache,
            unsupervised,
            area=area,
            cluster_grid=cluster_grid,
            alpha_grid=alpha_grid,
            rows_per_mouse=rows_per_mouse,
            tolerance=selection_tolerance,
            representation_cache=representation_cache,
            forecaster_cache=forecaster_cache,
            verbose=verbose,
        )
        nested_parts.append(
            _decorate_scores(
                nested,
                split="unsupervised_nested_lomo",
                selection_scope="outer_fold_selection",
            )
        )

        final_pipeline = _cached_pipeline(
            cache,
            unsupervised,
            selected,
            area=area,
            rows_per_mouse=rows_per_mouse,
            representation_cache=representation_cache,
            forecaster_cache=forecaster_cache,
        )
        learned_transfer = _score_predictions(
            cache,
            supervised,
            (
                predict_transition(cache, int(index), final_pipeline)
                for index in supervised
            ),
            scale=_training_scale(cache, unsupervised),
        )
        learned_transfer = learned_transfer.assign(
            area=area,
            clusters=selected.clusters,
            alpha=selected.alpha,
            model="final_selected_soft_cluster",
        )
        persistence_transfer = _score_persistence(
            cache, unsupervised, supervised
        ).assign(
            area=area,
            clusters=0,
            alpha=np.nan,
            model="neuron_persistence",
        )
        transfer_parts.append(
            _decorate_scores(
                pd.concat(
                    [learned_transfer, persistence_transfer], ignore_index=True
                ),
                split="supervised_transfer",
                selection_scope="locked_unsupervised_choice",
            )
        )
        if verbose:
            print(f"{area}: supervised transfer complete", flush=True)

    candidate_scores = pd.concat(candidate_parts, ignore_index=True)
    nested_scores = pd.concat(nested_parts, ignore_index=True)
    transfer_scores = pd.concat(transfer_parts, ignore_index=True)
    scores = pd.concat(
        [candidate_scores, nested_scores, transfer_scores],
        ignore_index=True,
        sort=False,
    )
    selection_frame = pd.DataFrame.from_records(selections).sort_values("area")
    return EvaluationResult(
        candidate_scores=candidate_scores,
        nested_scores=nested_scores,
        transfer_scores=transfer_scores,
        selections=selection_frame,
        scores=scores,
        summary=_summary(scores),
        paired_tests=paired_model_tests(scores),
    )


def run(
    cache_path: Path,
    output: Path,
    *,
    cluster_grid: Sequence[int] = CLUSTER_GRID,
    alpha_grid: Sequence[float] = ALPHA_GRID,
    rows_per_mouse: int = 1000,
) -> tuple[Path, ...]:
    cache = load_cache(cache_path)
    result = evaluate(
        cache,
        cluster_grid=cluster_grid,
        alpha_grid=alpha_grid,
        rows_per_mouse=rows_per_mouse,
        verbose=True,
    )
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "scores": output / "soft_cluster_mouse_scores.csv",
        "summary": output / "soft_cluster_summary.csv",
        "candidates": output / "soft_cluster_candidate_scores.csv",
        "nested": output / "soft_cluster_nested_scores.csv",
        "transfer": output / "soft_cluster_transfer_scores.csv",
        "selections": output / "soft_cluster_selections.csv",
        "paired_tests": output / "soft_cluster_paired_tests.csv",
        "report": output / "SOFT_CLUSTER_REPORT.md",
    }
    result.scores.to_csv(paths["scores"], index=False)
    result.summary.to_csv(paths["summary"], index=False)
    result.candidate_scores.to_csv(paths["candidates"], index=False)
    result.nested_scores.to_csv(paths["nested"], index=False)
    result.transfer_scores.to_csv(paths["transfer"], index=False)
    result.selections.to_csv(paths["selections"], index=False)
    result.paired_tests.to_csv(paths["paired_tests"], index=False)

    nested_summary = result.summary.loc[
        result.summary["split"].eq("unsupervised_nested_lomo")
    ]
    transfer_summary = result.summary.loc[
        result.summary["split"].eq("supervised_transfer")
    ]
    transfer_learned = transfer_summary.loc[
        transfer_summary["model"].eq("final_selected_soft_cluster")
    ].set_index("area")
    medial_gain = float(
        transfer_learned.loc["mHV", "improvement_vs_persistence_pct"]
    )
    anterior_gain = float(
        transfer_learned.loc["aHV", "improvement_vs_persistence_pct"]
    )
    interpretation = (
        "The locked model was essentially tied with neuron persistence in mHV "
        f"(relative improvement {medial_gain:+.2f}%, effectively zero). "
        "Its largest supervised gain was in aHV "
        f"({anterior_gain:+.2f}%). This secondary result therefore does not "
        "support a medial-selective transfer claim."
    )
    provenance = cache.provenance
    history_sha = provenance.history_sha256 if provenance else "synthetic/unknown"
    stable_objective_b_link = Path(
        os.path.relpath(
            Path(__file__).resolve().with_name(
                "STABLE_OBJECTIVE_B_MODEL_AND_RESULTS.md"
            ),
            start=paths["report"].resolve().parent,
        )
    ).as_posix()
    paths["report"].write_text(
        "# Soft-cluster next-window analysis\n\n"
        "> For the complete human-readable Objective B explanation and "
        "equations, see\n"
        "> [`STABLE_OBJECTIVE_B_MODEL_AND_RESULTS.md`]"
        f"({stable_objective_b_link}).\n\n"
        f"Window: {cache.window_pairs} trial pairs. Source-history SHA-256: "
        f"`{history_sha}`.\n\n"
        "Membership uses only current-window d-prime and static cortical "
        "coordinates; future activity never enters the GMM. GMM sampling, "
        "feature scaling, target scaling, and ridge fitting give every mouse "
        "equal total influence. Current progress and running speed enter the "
        "forecaster, not the membership model.\n\n"
        "Candidate K and ridge alpha values were compared only among "
        "unsupervised mice. Nested LOMO below estimates the complete "
        "selection-and-refit procedure. One locked configuration per area was "
        "then fit to all unsupervised mice and evaluated once on supervised "
        "mice; non-selected candidates have no supervised scores.\n\n"
        "## Locked choices\n\n"
        + result.selections.to_csv(index=False)
        + "\n## Nested unsupervised estimate\n\n"
        + nested_summary.to_csv(index=False)
        + "\nHere `clusters=-1` denotes the complete fold-specific selection "
        "procedure; each outer fold's actual K and alpha remain in the "
        "mouse-score file.\n\n"
        + "\n## Supervised transfer\n\n"
        + transfer_summary.to_csv(index=False)
        + "\n## Exact paired mouse-level tests\n\n"
        + result.paired_tests.to_csv(index=False)
        + "\nThese exact tests are exploratory and unadjusted for the four "
        "areas. With four supervised mice, the smallest attainable two-sided "
        "sign-flip p-value is 0.125.\n\n"
        + "## Interpretation\n\n"
        + interpretation
        + "\n\n"
        + "\nThis analysis forecasts registered neurons within a session. It "
        "does not establish cross-day cell identity and is secondary to the "
        "area-level before-to-after exposure test.\n"
    )
    return tuple(paths.values())


def _number_grid(text: str, converter) -> tuple:
    values = tuple(converter(value.strip()) for value in text.split(",") if value.strip())
    if not values:
        raise argparse.ArgumentTypeError("grid must contain at least one value")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path("modeling/cluster_cache/neuron_transitions_w20.npz"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("modeling/runs/soft_clusters_w20"),
    )
    parser.add_argument(
        "--clusters",
        type=lambda value: _number_grid(value, int),
        default=CLUSTER_GRID,
        help="comma-separated candidate K values",
    )
    parser.add_argument(
        "--alphas",
        type=lambda value: _number_grid(value, float),
        default=ALPHA_GRID,
        help="comma-separated ridge alpha values",
    )
    parser.add_argument("--rows-per-mouse", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    for path in run(
        arguments.cache,
        arguments.output,
        cluster_grid=arguments.clusters,
        alpha_grid=arguments.alphas,
        rows_per_mouse=arguments.rows_per_mouse,
    ):
        print(path)


if __name__ == "__main__":
    main()
