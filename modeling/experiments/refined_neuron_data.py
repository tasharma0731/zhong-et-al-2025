"""Typed loader for the isolated enriched neuron-transition cache.

The stable :class:`modeling.soft_cluster_forecast.Cache` remains the source of
truth for chronology, neuron sampling, d-prime, and cortical position.  The refined
cache is a strict superset stored under ``modeling/experiments/cache``.  This
module loads that superset into a typed object without changing the stable
cache class or loader.

Only attributes whose names start with ``current_`` are legal prediction-time
features.  The ``next_`` response moments are retained solely for target
auditing and must never be read by a prediction method.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import hashlib
import json
from pathlib import Path

import numpy as np

from modeling.soft_cluster_forecast import Cache, load_cache


EXPECTED_REFINED_CACHE_SCHEMA = 1
WORKSPACE = Path(__file__).resolve().parents[2]

CURRENT_NEURON_FIELDS = (
    "current_mean_leaf",
    "current_mean_circle",
    "current_sd_leaf",
    "current_sd_circle",
    "current_split_dprime_instability",
    "current_split_signal_instability",
    "current_split_log_denominator_instability",
    "current_neuron_run_correlation",
)
CURRENT_TRANSITION_FIELDS = (
    "current_window_index",
    "current_run_speed_sd",
    "current_moving_fraction",
    "current_trial_duration",
    "current_leaf_run_speed",
    "current_circle_run_speed",
    "current_role_run_speed_difference",
    "current_elapsed_seconds",
)
CURRENT_VALIDITY_FIELDS = (
    "current_split_instability_valid",
    "current_neuron_run_correlation_valid",
)
NEXT_AUDIT_FIELDS = (
    "next_mean_leaf",
    "next_mean_circle",
    "next_sd_leaf",
    "next_sd_circle",
)


@dataclass(frozen=True)
class RefinedCacheProvenance:
    """Provenance unique to the experimental cache extension."""

    schema_version: int
    config: dict[str, object]
    base_sha256: str
    builder_sha256: str


@dataclass
class RefinedNeuronCache(Cache):
    """Stable transition cache plus current-window experimental descriptors."""

    current_mean_leaf: np.ndarray | None = None
    current_mean_circle: np.ndarray | None = None
    current_sd_leaf: np.ndarray | None = None
    current_sd_circle: np.ndarray | None = None
    current_split_dprime_instability: np.ndarray | None = None
    current_split_signal_instability: np.ndarray | None = None
    current_split_log_denominator_instability: np.ndarray | None = None
    current_neuron_run_correlation: np.ndarray | None = None
    current_split_instability_valid: np.ndarray | None = None
    current_neuron_run_correlation_valid: np.ndarray | None = None
    current_window_index: np.ndarray | None = None
    current_run_speed_sd: np.ndarray | None = None
    current_moving_fraction: np.ndarray | None = None
    current_trial_duration: np.ndarray | None = None
    current_leaf_run_speed: np.ndarray | None = None
    current_circle_run_speed: np.ndarray | None = None
    current_role_run_speed_difference: np.ndarray | None = None
    current_elapsed_seconds: np.ndarray | None = None
    next_mean_leaf: np.ndarray | None = None
    next_mean_circle: np.ndarray | None = None
    next_sd_leaf: np.ndarray | None = None
    next_sd_circle: np.ndarray | None = None
    refined_provenance: RefinedCacheProvenance | None = None


def _archive_scalar(archive: np.lib.npyio.NpzFile, name: str) -> object:
    if name not in archive.files:
        raise ValueError(f"refined cache is missing {name!r}")
    return np.asarray(archive[name]).item()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _base_values(cache: Cache) -> dict[str, object]:
    return {field.name: getattr(cache, field.name) for field in fields(Cache)}


def _validate_refined_cache(cache: RefinedNeuronCache) -> None:
    neuron_shape = cache.current.shape
    transition_shape = (len(cache.metadata),)
    for name in (*CURRENT_NEURON_FIELDS, *NEXT_AUDIT_FIELDS):
        value = getattr(cache, name)
        if value is None or np.asarray(value).shape != neuron_shape:
            raise ValueError(f"{name} must have shape {neuron_shape}")
    for name in CURRENT_VALIDITY_FIELDS:
        value = getattr(cache, name)
        if value is None or np.asarray(value).shape != neuron_shape:
            raise ValueError(f"{name} must have shape {neuron_shape}")
    for name in CURRENT_TRANSITION_FIELDS:
        value = getattr(cache, name)
        if value is None or np.asarray(value).shape != transition_shape:
            raise ValueError(f"{name} must have shape {transition_shape}")

    if cache.refined_provenance is None:
        raise ValueError("refined cache provenance is unavailable")
    configured_window = int(cache.refined_provenance.config["window_pairs"])
    if configured_window != cache.window_pairs:
        raise ValueError(
            "refined cache window size disagrees with its stable-cache payload"
        )
    metadata_window = (
        cache.metadata["current_start_pair_id"].to_numpy(dtype=float)
        / float(cache.window_pairs)
    )
    np.testing.assert_allclose(
        np.asarray(cache.current_window_index, dtype=float),
        metadata_window,
        rtol=0,
        atol=1e-12,
    )


def load_refined_cache(
    path: str | Path,
    *,
    verify_source_history: bool = True,
    verify_base_cache: bool = True,
    verify_builder: bool = True,
    base_cache_path: str | Path | None = None,
) -> RefinedNeuronCache:
    """Load and validate an enriched cache without mutating stable objects."""

    path = Path(path).resolve()
    stable = load_cache(path, verify_source_history=verify_source_history)
    try:
        with np.load(path, allow_pickle=False) as archive:
            schema = int(_archive_scalar(archive, "refined_cache_schema_version"))
            if schema != EXPECTED_REFINED_CACHE_SCHEMA:
                raise ValueError(
                    f"refined cache schema {schema} is stale; "
                    f"expected {EXPECTED_REFINED_CACHE_SCHEMA}"
                )
            provenance = RefinedCacheProvenance(
                schema_version=schema,
                config=json.loads(
                    str(_archive_scalar(archive, "refined_cache_config_json"))
                ),
                base_sha256=str(
                    _archive_scalar(archive, "refined_cache_base_sha256")
                ),
                builder_sha256=str(
                    _archive_scalar(archive, "refined_cache_builder_sha256")
                ),
            )
            arrays: dict[str, np.ndarray] = {}
            for name in (
                *CURRENT_NEURON_FIELDS,
                *CURRENT_TRANSITION_FIELDS,
                *NEXT_AUDIT_FIELDS,
            ):
                if name not in archive.files:
                    raise ValueError(f"refined cache is missing {name!r}")
                arrays[name] = archive[name].astype(float)
            for name in CURRENT_VALIDITY_FIELDS:
                if name not in archive.files:
                    raise ValueError(f"refined cache is missing {name!r}")
                arrays[name] = archive[name].astype(bool)
    except KeyError as error:
        raise ValueError(f"refined cache {path} is incomplete: {error}") from error

    if verify_base_cache:
        recorded = Path(str(provenance.config["base_cache"]))
        candidates = (
            [Path(base_cache_path).resolve()]
            if base_cache_path is not None
            else [
                (
                    recorded
                    if recorded.is_absolute()
                    else (WORKSPACE / recorded).resolve()
                ),
                (
                    WORKSPACE
                    / "modeling/cluster_cache"
                    / recorded.name
                ).resolve(),
            ]
        )
        unique_candidates = tuple(dict.fromkeys(candidates))
        matching_base = next(
            (
                candidate
                for candidate in unique_candidates
                if candidate.is_file()
                and _sha256(candidate) == provenance.base_sha256
            ),
            None,
        )
        if matching_base is None:
            attempted = ", ".join(map(str, unique_candidates))
            raise ValueError(
                "the stable base cache is unavailable or has changed; "
                f"checked: {attempted}. Pass base_cache_path explicitly after "
                "an intentional repository relocation, or set "
                "verify_base_cache=False only when provenance verification is "
                "deliberately disabled"
            )
    builder_path = Path(__file__).with_name("build_refined_neuron_cache.py")
    if verify_builder and not builder_path.is_file():
        raise ValueError(f"refined cache builder is unavailable: {builder_path}")
    if verify_builder and _sha256(builder_path) != provenance.builder_sha256:
        raise ValueError(
            "the refined cache builder differs from the version recorded "
            "inside the cache"
        )

    result = RefinedNeuronCache(
        **_base_values(stable),
        **arrays,
        refined_provenance=provenance,
    )
    _validate_refined_cache(result)
    return result


__all__ = [
    "CURRENT_NEURON_FIELDS",
    "CURRENT_TRANSITION_FIELDS",
    "CURRENT_VALIDITY_FIELDS",
    "EXPECTED_REFINED_CACHE_SCHEMA",
    "NEXT_AUDIT_FIELDS",
    "RefinedCacheProvenance",
    "RefinedNeuronCache",
    "load_refined_cache",
]
