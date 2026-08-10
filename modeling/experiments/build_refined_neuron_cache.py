#!/usr/bin/env python3
"""Build the isolated, leakage-safe cache for the refined neuron experiment.

The stable W20 cache deliberately stores only d-prime, its denominator, static
cortical position, and two coarse context variables.  This experimental builder reads
that immutable cache and appends current-window response primitives and
behavioral summaries.  It never modifies or replaces the stable cache.

The output stays row-aligned with
``modeling/cluster_cache/neuron_transitions_w20.npz``:

``area-transition x sampled-neuron``.

All arrays whose names start with ``current_`` are legal predictor inputs.
Arrays whose names start with ``next_`` are targets or audit values only.
The builder verifies that independently reconstructed d-prime and denominator
values agree with the stable cache before it writes anything.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any

import numpy as np
import pandas as pd


WORKSPACE = Path(__file__).resolve().parents[2]
CODE = WORKSPACE / "code"
for source_root in (WORKSPACE, CODE):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from database import ZhongDB  # noqa: E402
from dprime import eligible_trial_frames  # noqa: E402
from joiner import Joiner  # noqa: E402
from modeling.chronology import chronological_trial_pairs_and_windows  # noqa: E402
from modeling.soft_cluster_forecast import Cache, load_cache  # noqa: E402


SCHEMA_VERSION = 1
ROLE_LEAF = 2
ROLE_CIRCLE = 0
MINIMUM_FRAMES_PER_TRIAL = 3


@dataclass(frozen=True)
class RefinedCacheConfig:
    """Immutable choices that define one enriched cache."""

    # Repository-relative when the source is inside this checkout.  This is an
    # informational lookup hint; the adjacent SHA-256 is the source identity.
    base_cache: str
    window_pairs: int
    minimum_frames_per_trial: int = MINIMUM_FRAMES_PER_TRIAL
    role_leaf: int = ROLE_LEAF
    role_circle: int = ROLE_CIRCLE
    split_method: str = "interleaved_pairs"
    neural_behavior_coupling: str = "role_residualized_trial_pearson"

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _portable_path_hint(path: Path) -> str:
    """Return a checkout-independent hint for a provenance input path."""

    try:
        return path.resolve().relative_to(WORKSPACE).as_posix()
    except ValueError:
        return path.name


def _metadata_lookup(db: ZhongDB, behavior_session_id: str) -> dict[str, str]:
    rows = db.query(
        """
        SELECT behavior_session_id, recording_id, experiment, behavior_key
        FROM behavior_sessions
        WHERE behavior_session_id = ?
        """,
        [behavior_session_id],
    )
    if len(rows) != 1:
        raise ValueError(
            f"expected one behavior session for {behavior_session_id!r}, "
            f"found {len(rows)}"
        )
    return {name: str(rows.iloc[0][name]) for name in rows.columns}


def _constant_neuron_axis(cache: Cache, indices: np.ndarray) -> np.ndarray:
    """Return one stable neuron axis and validate every transition copy."""

    if cache.neuron_id is None:
        raise ValueError("base cache does not contain neuron_id")
    reference = np.asarray(cache.neuron_id[int(indices[0])], dtype=np.int64)
    for index in indices[1:]:
        if not np.array_equal(reference, cache.neuron_id[int(index)]):
            raise ValueError(
                "neuron order changes within one recording/area trajectory"
            )
    return reference


def _trial_component_means(
    components_by_frame: np.ndarray,
    frames: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Return sorted trial IDs and one component mean per eligible trial."""

    grouped = frames.groupby("trial_id", sort=True)["frame_id"].apply(
        lambda values: values.to_numpy(dtype=np.int64)
    )
    trial_ids = grouped.index.to_numpy(dtype=np.int64)
    means = np.empty(
        (len(trial_ids), components_by_frame.shape[0]),
        dtype=np.float64,
    )
    for offset, frame_ids in enumerate(grouped):
        block = np.asarray(components_by_frame[:, frame_ids], dtype=np.float64)
        means[offset] = np.mean(block, axis=1, dtype=np.float64)
    return trial_ids, means


def _trial_behavior(joiner: Joiner, frames: pd.DataFrame) -> pd.DataFrame:
    """Compute trial-balanced behavioral values on the neural eligibility mask."""

    eligible = frames[["frame_id", "trial_id", "stimulus_role"]].merge(
        joiner.frames[
            ["frame_id", "run_speed", "is_moving", "time"]
            if "time" in joiner.frames
            else ["frame_id", "run_speed", "is_moving"]
        ],
        on="frame_id",
        how="left",
        validate="one_to_one",
    )
    aggregations: dict[str, tuple[str, str]] = {
        "mean_run_speed": ("run_speed", "mean"),
        "moving_fraction": ("is_moving", "mean"),
        "eligible_frames": ("frame_id", "size"),
    }
    if "time" in eligible:
        aggregations.update(
            first_time=("time", "min"),
            last_time=("time", "max"),
        )
    result = (
        eligible.groupby(["trial_id", "stimulus_role"], as_index=False)
        .agg(**aggregations)
        .sort_values("trial_id")
    )
    if "time" in eligible:
        # MATLAB datenums are measured in days in the released behavior files.
        result["duration_seconds"] = (
            result["last_time"] - result["first_time"]
        ) * 86400.0
    else:
        result["duration_seconds"] = np.nan
    return result


def _window_moments(
    responses: np.ndarray,
    trial_index: dict[int, int],
    trial_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    rows = np.asarray([trial_index[int(value)] for value in trial_ids], dtype=int)
    selected = responses[rows]
    return (
        np.mean(selected, axis=0, dtype=np.float64),
        np.std(selected, axis=0, ddof=0, dtype=np.float64),
    )


def _dprime(
    mean_leaf: np.ndarray,
    sd_leaf: np.ndarray,
    mean_circle: np.ndarray,
    sd_circle: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    denominator = sd_leaf + sd_circle
    valid = (
        np.isfinite(denominator)
        & (denominator > 0)
        & np.isfinite(mean_leaf)
        & np.isfinite(mean_circle)
    )
    values = np.full(len(denominator), np.nan, dtype=np.float64)
    values[valid] = (
        2.0 * (mean_leaf[valid] - mean_circle[valid]) / denominator[valid]
    )
    return values, denominator, valid


def _split_instability(
    responses: np.ndarray,
    trial_index: dict[int, int],
    leaf_ids: np.ndarray,
    circle_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if min(len(leaf_ids), len(circle_ids)) < 4:
        shape = responses.shape[1]
        missing = np.full(shape, np.nan, dtype=np.float64)
        return missing, missing.copy(), missing.copy()
    split_values: list[np.ndarray] = []
    split_signal: list[np.ndarray] = []
    split_log_denominator: list[np.ndarray] = []
    for parity in (0, 1):
        mean_leaf, sd_leaf = _window_moments(
            responses, trial_index, leaf_ids[parity::2]
        )
        mean_circle, sd_circle = _window_moments(
            responses, trial_index, circle_ids[parity::2]
        )
        values, denominator, valid = _dprime(
            mean_leaf, sd_leaf, mean_circle, sd_circle
        )
        signal = 2.0 * (mean_leaf - mean_circle)
        log_denominator = np.full(len(denominator), np.nan, dtype=np.float64)
        log_denominator[valid] = np.log(denominator[valid])
        split_values.append(values)
        split_signal.append(signal)
        split_log_denominator.append(log_denominator)
    return (
        np.abs(split_values[0] - split_values[1]),
        np.abs(split_signal[0] - split_signal[1]),
        np.abs(split_log_denominator[0] - split_log_denominator[1]),
    )


def _role_residualized_correlation(
    responses: np.ndarray,
    trial_index: dict[int, int],
    trial_behavior: pd.DataFrame,
    leaf_ids: np.ndarray,
    circle_ids: np.ndarray,
) -> np.ndarray:
    """Return per-neuron trial correlation after removing role means."""

    trial_ids = np.concatenate([leaf_ids, circle_ids])
    roles = np.concatenate(
        [
            np.full(len(leaf_ids), ROLE_LEAF, dtype=np.int64),
            np.full(len(circle_ids), ROLE_CIRCLE, dtype=np.int64),
        ]
    )
    neural_rows = np.asarray([trial_index[int(value)] for value in trial_ids])
    neural = np.asarray(responses[neural_rows], dtype=np.float64)
    speed_lookup = trial_behavior.set_index("trial_id")["mean_run_speed"]
    speed = speed_lookup.reindex(trial_ids).to_numpy(dtype=np.float64)
    valid_speed = np.isfinite(speed)
    if valid_speed.sum() < 6:
        return np.full(neural.shape[1], np.nan, dtype=np.float64)
    neural = neural[valid_speed]
    speed = speed[valid_speed]
    roles = roles[valid_speed]
    for role in (ROLE_LEAF, ROLE_CIRCLE):
        mask = roles == role
        speed[mask] -= np.mean(speed[mask])
        neural[mask] -= np.mean(neural[mask], axis=0, dtype=np.float64)
    speed_norm = np.sqrt(np.sum(np.square(speed)))
    neural_norm = np.sqrt(np.sum(np.square(neural), axis=0))
    denominator = speed_norm * neural_norm
    result = np.full(neural.shape[1], np.nan, dtype=np.float64)
    valid = np.isfinite(denominator) & (denominator > 0)
    result[valid] = (speed @ neural[:, valid]) / denominator[valid]
    return np.clip(result, -1.0, 1.0)


def _behavior_summary(
    trial_behavior: pd.DataFrame,
    leaf_ids: np.ndarray,
    circle_ids: np.ndarray,
    *,
    session_start_time: float | None,
) -> dict[str, float]:
    selected = trial_behavior.loc[
        trial_behavior["trial_id"].isin(np.concatenate([leaf_ids, circle_ids]))
    ]
    leaf = selected.loc[selected["stimulus_role"].eq(ROLE_LEAF)]
    circle = selected.loc[selected["stimulus_role"].eq(ROLE_CIRCLE)]
    trial_speed = selected["mean_run_speed"].to_numpy(dtype=float)
    elapsed_seconds = np.nan
    if (
        session_start_time is not None
        and "last_time" in selected
        and np.isfinite(selected["last_time"]).any()
    ):
        elapsed_seconds = (
            float(np.nanmax(selected["last_time"])) - session_start_time
        ) * 86400.0
    return {
        "current_run_speed_sd": float(np.nanstd(trial_speed, ddof=0)),
        "current_moving_fraction": float(selected["moving_fraction"].mean()),
        "current_trial_duration": float(selected["duration_seconds"].mean()),
        "current_leaf_run_speed": float(leaf["mean_run_speed"].mean()),
        "current_circle_run_speed": float(circle["mean_run_speed"].mean()),
        "current_role_run_speed_difference": float(
            leaf["mean_run_speed"].mean() - circle["mean_run_speed"].mean()
        ),
        "current_elapsed_seconds": float(elapsed_seconds),
    }


def _empty_outputs(shape: tuple[int, int]) -> dict[str, np.ndarray]:
    neuron_arrays = (
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
        "current_neuron_run_correlation",
    )
    scalar_arrays = (
        "current_window_index",
        "current_run_speed_sd",
        "current_moving_fraction",
        "current_trial_duration",
        "current_leaf_run_speed",
        "current_circle_run_speed",
        "current_role_run_speed_difference",
        "current_elapsed_seconds",
    )
    output = {
        name: np.full(shape, np.nan, dtype=np.float32)
        for name in neuron_arrays
    }
    output.update(
        {
            name: np.full(shape[0], np.nan, dtype=np.float64)
            for name in scalar_arrays
        }
    )
    output.update(
        current_split_instability_valid=np.zeros(shape, dtype=bool),
        current_neuron_run_correlation_valid=np.zeros(shape, dtype=bool),
    )
    return output


def _verify_against_base(
    *,
    cache: Cache,
    index: int,
    prefix: str,
    values: np.ndarray,
    denominator: np.ndarray,
    valid: np.ndarray,
) -> None:
    expected_values = cache.current if prefix == "current" else cache.future
    expected_denominator = (
        cache.current_denominator
        if prefix == "current"
        else cache.next_denominator
    )
    expected_valid = (
        cache.current_valid_denominator
        if prefix == "current"
        else cache.next_valid_denominator
    )
    if expected_denominator is None or expected_valid is None:
        raise ValueError("base cache is missing denominator audit arrays")
    np.testing.assert_array_equal(valid, expected_valid[index])
    np.testing.assert_allclose(
        denominator,
        expected_denominator[index],
        rtol=2e-5,
        atol=2e-5,
    )
    np.testing.assert_allclose(
        values,
        expected_values[index],
        rtol=3e-5,
        atol=3e-5,
        equal_nan=True,
    )


def _recording_outputs(
    cache: Cache,
    recording_indices: np.ndarray,
    joiner: Joiner,
    outputs: dict[str, np.ndarray],
) -> None:
    frames = eligible_trial_frames(
        joiner.frames,
        roles=(ROLE_LEAF, ROLE_CIRCLE),
        min_frames=MINIMUM_FRAMES_PER_TRIAL,
    )
    pairs, windows = chronological_trial_pairs_and_windows(
        frames,
        size=cache.window_pairs,
        role_a=ROLE_LEAF,
        role_b=ROLE_CIRCLE,
    )
    window_by_start = windows.set_index("start_pair_id")
    trial_ids, component_means = _trial_component_means(joiner.V, frames)
    trial_index = {int(value): offset for offset, value in enumerate(trial_ids)}
    behavior = _trial_behavior(joiner, frames)
    session_start_time = (
        float(np.nanmin(behavior["first_time"]))
        if "first_time" in behavior and np.isfinite(behavior["first_time"]).any()
        else None
    )

    for area in sorted(
        cache.metadata.iloc[recording_indices]["area"].astype(str).unique()
    ):
        area_indices = recording_indices[
            cache.metadata.iloc[recording_indices]["area"].astype(str).to_numpy()
            == area
        ]
        neuron_ids = _constant_neuron_axis(cache, area_indices)
        weights = np.asarray(joiner.U[:, neuron_ids], dtype=np.float64)
        responses = component_means @ weights

        for index in area_indices:
            row = cache.metadata.iloc[int(index)]
            current_start = int(row["current_start_pair_id"])
            next_start = int(row["next_start_pair_id"])
            if current_start not in window_by_start.index:
                raise KeyError(f"missing current window {current_start}")
            if next_start not in window_by_start.index:
                raise KeyError(f"missing next window {next_start}")

            window_values: dict[str, tuple[np.ndarray, ...]] = {}
            trial_ids_by_prefix: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            for prefix, start in (("current", current_start), ("next", next_start)):
                window = window_by_start.loc[start]
                block = pairs.loc[
                    pairs["pair_id"].between(
                        int(start),
                        int(window["stop_pair_id"]),
                    )
                ]
                leaf_ids = block["trial_a_id"].to_numpy(dtype=np.int64)
                circle_ids = block["trial_b_id"].to_numpy(dtype=np.int64)
                mean_leaf, sd_leaf = _window_moments(
                    responses, trial_index, leaf_ids
                )
                mean_circle, sd_circle = _window_moments(
                    responses, trial_index, circle_ids
                )
                values, denominator, valid = _dprime(
                    mean_leaf,
                    sd_leaf,
                    mean_circle,
                    sd_circle,
                )
                _verify_against_base(
                    cache=cache,
                    index=int(index),
                    prefix=prefix,
                    values=values,
                    denominator=denominator,
                    valid=valid,
                )
                window_values[prefix] = (
                    mean_leaf,
                    mean_circle,
                    sd_leaf,
                    sd_circle,
                )
                trial_ids_by_prefix[prefix] = (leaf_ids, circle_ids)

            for prefix in ("current", "next"):
                mean_leaf, mean_circle, sd_leaf, sd_circle = window_values[prefix]
                outputs[f"{prefix}_mean_leaf"][index] = mean_leaf.astype(np.float32)
                outputs[f"{prefix}_mean_circle"][index] = mean_circle.astype(
                    np.float32
                )
                outputs[f"{prefix}_sd_leaf"][index] = sd_leaf.astype(np.float32)
                outputs[f"{prefix}_sd_circle"][index] = sd_circle.astype(np.float32)

            leaf_ids, circle_ids = trial_ids_by_prefix["current"]
            split = _split_instability(
                responses,
                trial_index,
                leaf_ids,
                circle_ids,
            )
            outputs["current_split_dprime_instability"][index] = split[0].astype(
                np.float32
            )
            outputs["current_split_signal_instability"][index] = split[1].astype(
                np.float32
            )
            outputs["current_split_log_denominator_instability"][index] = split[
                2
            ].astype(np.float32)
            outputs["current_split_instability_valid"][index] = np.isfinite(
                split[0]
            )
            run_correlation = _role_residualized_correlation(
                responses,
                trial_index,
                behavior,
                leaf_ids,
                circle_ids,
            )
            outputs["current_neuron_run_correlation"][index] = (
                run_correlation.astype(np.float32)
            )
            outputs["current_neuron_run_correlation_valid"][index] = np.isfinite(
                run_correlation
            )
            outputs["current_window_index"][index] = (
                current_start / cache.window_pairs
            )
            for name, value in _behavior_summary(
                behavior,
                leaf_ids,
                circle_ids,
                session_start_time=session_start_time,
            ).items():
                outputs[name][index] = value


def _atomic_npz(destination: Path, arrays: dict[str, Any]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{destination.name}.",
            suffix=".npz",
            dir=destination.parent,
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def build(
    base_cache_path: Path,
    destination: Path,
    *,
    data_cache: Path,
    database_path: Path,
) -> Path:
    """Build and atomically write one refined experimental cache."""

    base_cache_path = base_cache_path.resolve()
    destination = destination.resolve()
    if destination == base_cache_path:
        raise ValueError("refined cache must not replace the stable cache")
    cache = load_cache(base_cache_path)
    config = RefinedCacheConfig(
        base_cache=_portable_path_hint(base_cache_path),
        window_pairs=cache.window_pairs,
    )
    outputs = _empty_outputs(cache.current.shape)

    with ZhongDB(
        cache=data_cache,
        database=database_path,
        mount=False,
    ) as db:
        recordings = cache.metadata[
            ["behavior_session_id", "recording_id"]
        ].drop_duplicates()
        for offset, row in enumerate(recordings.itertuples(index=False), start=1):
            started = time.perf_counter()
            selection = _metadata_lookup(db, str(row.behavior_session_id))
            joiner = Joiner(
                db,
                selection["recording_id"],
                experiment=selection["experiment"],
                behavior_key=selection["behavior_key"],
            )
            indices = np.flatnonzero(
                cache.metadata["recording_id"].astype(str).eq(
                    str(row.recording_id)
                )
            )
            _recording_outputs(cache, indices, joiner, outputs)
            print(
                f"[{offset:02d}/{len(recordings):02d}] "
                f"{row.behavior_session_id} {len(indices)} area-transitions "
                f"{time.perf_counter() - started:.1f}s",
                flush=True,
            )
            del joiner
            gc.collect()

    optional_missing = {
        "current_split_dprime_instability",
        "current_split_signal_instability",
        "current_split_log_denominator_instability",
        "current_neuron_run_correlation",
        "current_trial_duration",
        "current_elapsed_seconds",
    }
    for name, values in outputs.items():
        if name not in optional_missing and not np.isfinite(values).all():
            missing = int(np.size(values) - np.isfinite(values).sum())
            raise ValueError(f"{name} contains {missing} non-finite values")

    with np.load(base_cache_path, allow_pickle=False) as archive:
        arrays: dict[str, Any] = {name: archive[name] for name in archive.files}
    overlap = set(arrays) & set(outputs)
    if overlap:
        raise ValueError(f"refined fields collide with base cache: {sorted(overlap)}")
    arrays.update(outputs)
    arrays.update(
        refined_cache_schema_version=np.asarray(SCHEMA_VERSION, dtype=np.int64),
        refined_cache_config_json=np.asarray(
            json.dumps(config.as_dict(), sort_keys=True)
        ),
        refined_cache_base_sha256=np.asarray(_sha256(base_cache_path)),
        refined_cache_builder_sha256=np.asarray(_sha256(Path(__file__))),
    )
    _atomic_npz(destination, arrays)
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-cache",
        type=Path,
        default=WORKSPACE / "modeling/cluster_cache/neuron_transitions_w20.npz",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            WORKSPACE
            / "modeling/experiments/cache/refined_neuron_transitions_w20.npz"
        ),
    )
    parser.add_argument(
        "--data-cache",
        type=Path,
        default=WORKSPACE / "data/cache",
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=WORKSPACE / "data/cache/zhong.duckdb",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    result = build(
        arguments.base_cache,
        arguments.output,
        data_cache=arguments.data_cache,
        database_path=arguments.database,
    )
    print(result)


if __name__ == "__main__":
    main()
