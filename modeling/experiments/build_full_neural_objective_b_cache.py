#!/usr/bin/env python3
"""Build an isolated full-neural cache for refined Objective B experiments.

The published full-neural ``*_neural_data.npy`` files are trusted pickled
dictionaries containing a list of ``neurons x frames`` float32 blocks.  They
cannot be memory mapped.  This builder therefore:

1. stages one catalogued recording at a time;
2. extracts only the neurons already selected by the stable W20 cache;
3. computes trial-balanced current/next W20 response moments;
4. writes a small, resumable per-recording shard; and
5. deletes the multi-GiB staged source before continuing.

Only mHV and aHV are included by default.  The stable and refined SVD caches
are read-only inputs and are never modified.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Iterable, Sequence

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
from modeling.experiments.build_refined_neuron_cache import (  # noqa: E402
    MINIMUM_FRAMES_PER_TRIAL,
    ROLE_CIRCLE,
    ROLE_LEAF,
    _dprime,
    _split_instability,
    _window_moments,
)
from modeling.experiments.refined_neuron_data import load_refined_cache  # noqa: E402


SCHEMA_VERSION = 1
DEFAULT_AREAS = ("mHV", "aHV")
DEFAULT_REMOTE_ROOT = (
    "gdrive_dataset:"
    "Janelia dataset - Zhong et al. 2025 (Figshare v2)/data/spk"
)
NEURON_FIELDS = (
    "current",
    "future",
    "current_denominator",
    "next_denominator",
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
)
VALIDITY_FIELDS = (
    "current_valid_denominator",
    "next_valid_denominator",
    "current_split_instability_valid",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _catalog_row(db: ZhongDB, recording_id: str) -> dict[str, Any]:
    rows = db.query(
        """
        SELECT filename, size_bytes, md5
        FROM files
        WHERE category = 'full_neural' AND recording_id = ?
        """,
        [recording_id],
    )
    if len(rows) != 1:
        raise ValueError(
            f"expected one full-neural file for {recording_id}, found {len(rows)}"
        )
    return {
        "filename": str(rows.iloc[0]["filename"]),
        "size_bytes": int(rows.iloc[0]["size_bytes"]),
        "md5": str(rows.iloc[0]["md5"]).lower(),
    }


def _remote_path(remote_root: str, filename: str) -> str:
    return f"{remote_root.rstrip('/')}/{filename}"


def _stage_file(
    catalog: dict[str, Any],
    *,
    remote_root: str,
    stage_dir: Path,
) -> Path:
    stage_dir.mkdir(parents=True, exist_ok=True)
    destination = stage_dir / str(catalog["filename"])
    expected_size = int(catalog["size_bytes"])
    expected_md5 = str(catalog["md5"])

    if destination.is_file() and destination.stat().st_size == expected_size:
        observed_md5 = _md5(destination)
        if observed_md5 == expected_md5:
            print(f"  reuse staged {destination.name}", flush=True)
            return destination
        destination.unlink()
    elif destination.exists():
        destination.unlink()

    rclone = shutil.which("rclone")
    if rclone is None:
        raise RuntimeError("rclone is required to stage full-neural files")
    command = [
        rclone,
        "copyto",
        _remote_path(remote_root, destination.name),
        str(destination),
        "--transfers",
        "1",
        "--checkers",
        "1",
        "--stats",
        "20s",
        "--stats-one-line",
    ]
    print(
        f"  stage {destination.name} "
        f"({expected_size / 2**30:.2f} GiB)",
        flush=True,
    )
    subprocess.run(command, check=True)
    if not destination.is_file() or destination.stat().st_size != expected_size:
        raise IOError(f"staged file size mismatch for {destination}")
    observed_md5 = _md5(destination)
    if observed_md5 != expected_md5:
        destination.unlink(missing_ok=True)
        raise IOError(
            f"staged file MD5 mismatch for {destination.name}: "
            f"{observed_md5} != {expected_md5}"
        )
    return destination


def _constant_neuron_axis(
    cache: Any,
    base_indices: np.ndarray,
) -> np.ndarray:
    if cache.neuron_id is None:
        raise ValueError("refined cache has no neuron IDs")
    reference = np.asarray(cache.neuron_id[int(base_indices[0])], dtype=np.int64)
    for index in base_indices[1:]:
        if not np.array_equal(reference, cache.neuron_id[int(index)]):
            raise ValueError("neuron order changed inside a recording/area")
    return reference


def _eligible_trial_groups(
    joiner: Joiner,
) -> tuple[pd.DataFrame, np.ndarray, tuple[np.ndarray, ...]]:
    frames = eligible_trial_frames(
        joiner.frames,
        roles=(ROLE_LEAF, ROLE_CIRCLE),
        min_frames=MINIMUM_FRAMES_PER_TRIAL,
    )
    grouped = frames.groupby("trial_id", sort=True)["frame_id"].apply(
        lambda values: values.to_numpy(dtype=np.int64)
    )
    return (
        frames,
        grouped.index.to_numpy(dtype=np.int64),
        tuple(grouped),
    )


def _selected_trial_means(
    spk_blocks: Sequence[np.ndarray],
    neuron_ids: np.ndarray,
    frame_groups: Sequence[np.ndarray],
    *,
    expected_frames: int,
) -> np.ndarray:
    """Return ``trials x selected neurons`` without concatenating all neurons."""

    selected = np.asarray(neuron_ids, dtype=np.int64)
    if selected.ndim != 1 or len(selected) == 0:
        raise ValueError("neuron_ids must be a non-empty vector")
    if len(np.unique(selected)) != len(selected):
        raise ValueError("neuron_ids must be unique")

    result = np.empty((len(frame_groups), len(selected)), dtype=np.float64)
    populated = np.zeros(len(selected), dtype=bool)
    offset = 0
    for block_index, value in enumerate(spk_blocks):
        block = np.asarray(value)
        if block.ndim != 2 or block.shape[1] != expected_frames:
            raise ValueError(
                f"full-neural block {block_index} has shape {block.shape}; "
                f"expected neurons x {expected_frames} frames"
            )
        in_block = (selected >= offset) & (selected < offset + block.shape[0])
        positions = np.flatnonzero(in_block)
        if len(positions):
            local_ids = selected[positions] - offset
            selected_block = np.asarray(block[local_ids], dtype=np.float64)
            for trial_position, frame_ids in enumerate(frame_groups):
                result[trial_position, positions] = np.mean(
                    selected_block[:, frame_ids],
                    axis=1,
                    dtype=np.float64,
                )
            populated[positions] = True
            del selected_block
        offset += block.shape[0]
    if selected.min() < 0 or selected.max() >= offset:
        raise IndexError(
            f"selected neuron IDs [{selected.min()}, {selected.max()}] "
            f"outside full-neural axis of {offset} neurons"
        )
    if not populated.all():
        raise RuntimeError(
            f"failed to populate {int((~populated).sum())} selected neurons"
        )
    return result


def _empty_recording_outputs(
    base_indices: np.ndarray,
    neurons_per_transition: int,
) -> dict[str, np.ndarray]:
    shape = (len(base_indices), neurons_per_transition)
    result = {
        name: np.full(shape, np.nan, dtype=np.float32)
        for name in NEURON_FIELDS
    }
    result.update(
        {
            name: np.zeros(shape, dtype=bool)
            for name in VALIDITY_FIELDS
        }
    )
    result["base_indices"] = np.asarray(base_indices, dtype=np.int64)
    return result


def _window_trials(
    pairs: pd.DataFrame,
    window_by_start: pd.DataFrame,
    start_pair_id: int,
) -> tuple[np.ndarray, np.ndarray]:
    if start_pair_id not in window_by_start.index:
        raise KeyError(f"missing W20 window beginning at pair {start_pair_id}")
    window = window_by_start.loc[start_pair_id]
    block = pairs.loc[
        pairs["pair_id"].between(
            int(start_pair_id),
            int(window["stop_pair_id"]),
        )
    ]
    return (
        block["trial_a_id"].to_numpy(dtype=np.int64),
        block["trial_b_id"].to_numpy(dtype=np.int64),
    )


def _build_recording_shard(
    *,
    cache: Any,
    recording_id: str,
    recording_base_indices: np.ndarray,
    joiner: Joiner,
    staged_path: Path,
    catalog: dict[str, Any],
    cache_sha256: str,
    builder_sha256: str,
    destination: Path,
) -> Path:
    started = time.perf_counter()
    local_lookup = {
        int(base_index): position
        for position, base_index in enumerate(recording_base_indices)
    }
    outputs = _empty_recording_outputs(
        recording_base_indices,
        cache.neurons_per_transition,
    )
    frames, trial_ids, frame_groups = _eligible_trial_groups(joiner)
    pairs, windows = chronological_trial_pairs_and_windows(
        frames,
        size=cache.window_pairs,
        role_a=ROLE_LEAF,
        role_b=ROLE_CIRCLE,
    )
    window_by_start = windows.set_index("start_pair_id")
    trial_index = {int(value): offset for offset, value in enumerate(trial_ids)}

    areas = tuple(
        sorted(
            cache.metadata.iloc[recording_base_indices]["area"]
            .astype(str)
            .unique()
        )
    )
    area_axes: dict[str, np.ndarray] = {}
    for area in areas:
        area_indices = recording_base_indices[
            cache.metadata.iloc[recording_base_indices]["area"]
            .astype(str)
            .eq(area)
            .to_numpy()
        ]
        area_axes[area] = _constant_neuron_axis(cache, area_indices)
    union_ids = np.unique(np.concatenate(tuple(area_axes.values())))

    print(f"  load trusted full-neural pickle", flush=True)
    payload = np.load(staged_path, allow_pickle=True).item()
    if not isinstance(payload, dict) or "spks" not in payload:
        raise ValueError(f"{staged_path.name} has no 'spks' list")
    blocks = payload["spks"]
    if not isinstance(blocks, (list, tuple)) or not blocks:
        raise ValueError(f"{staged_path.name} has an invalid 'spks' value")
    responses_union = _selected_trial_means(
        blocks,
        union_ids,
        frame_groups,
        expected_frames=joiner.V.shape[1],
    )
    del payload, blocks
    gc.collect()

    for area in areas:
        area_indices = recording_base_indices[
            cache.metadata.iloc[recording_base_indices]["area"]
            .astype(str)
            .eq(area)
            .to_numpy()
        ]
        neuron_ids = area_axes[area]
        union_positions = np.searchsorted(union_ids, neuron_ids)
        if not np.array_equal(union_ids[union_positions], neuron_ids):
            raise RuntimeError("area neuron IDs are missing from the union axis")
        responses = responses_union[:, union_positions]

        for base_index in area_indices:
            output_index = local_lookup[int(base_index)]
            metadata = cache.metadata.iloc[int(base_index)]
            values_by_prefix: dict[str, tuple[np.ndarray, ...]] = {}
            trials_by_prefix: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            for prefix, start in (
                ("current", int(metadata["current_start_pair_id"])),
                ("next", int(metadata["next_start_pair_id"])),
            ):
                leaf_ids, circle_ids = _window_trials(
                    pairs,
                    window_by_start,
                    start,
                )
                mean_leaf, sd_leaf = _window_moments(
                    responses,
                    trial_index,
                    leaf_ids,
                )
                mean_circle, sd_circle = _window_moments(
                    responses,
                    trial_index,
                    circle_ids,
                )
                dprime, denominator, valid = _dprime(
                    mean_leaf,
                    sd_leaf,
                    mean_circle,
                    sd_circle,
                )
                target_name = "current" if prefix == "current" else "future"
                denominator_name = (
                    "current_denominator"
                    if prefix == "current"
                    else "next_denominator"
                )
                validity_name = (
                    "current_valid_denominator"
                    if prefix == "current"
                    else "next_valid_denominator"
                )
                outputs[target_name][output_index] = dprime.astype(np.float32)
                outputs[denominator_name][output_index] = denominator.astype(
                    np.float32
                )
                outputs[validity_name][output_index] = valid
                values_by_prefix[prefix] = (
                    mean_leaf,
                    mean_circle,
                    sd_leaf,
                    sd_circle,
                )
                trials_by_prefix[prefix] = (leaf_ids, circle_ids)

            for prefix in ("current", "next"):
                mean_leaf, mean_circle, sd_leaf, sd_circle = values_by_prefix[
                    prefix
                ]
                outputs[f"{prefix}_mean_leaf"][output_index] = mean_leaf.astype(
                    np.float32
                )
                outputs[f"{prefix}_mean_circle"][output_index] = (
                    mean_circle.astype(np.float32)
                )
                outputs[f"{prefix}_sd_leaf"][output_index] = sd_leaf.astype(
                    np.float32
                )
                outputs[f"{prefix}_sd_circle"][output_index] = sd_circle.astype(
                    np.float32
                )

            leaf_ids, circle_ids = trials_by_prefix["current"]
            split = _split_instability(
                responses,
                trial_index,
                leaf_ids,
                circle_ids,
            )
            outputs["current_split_dprime_instability"][output_index] = (
                split[0].astype(np.float32)
            )
            outputs["current_split_signal_instability"][output_index] = (
                split[1].astype(np.float32)
            )
            outputs[
                "current_split_log_denominator_instability"
            ][output_index] = split[2].astype(np.float32)
            outputs["current_split_instability_valid"][output_index] = (
                np.isfinite(split[0])
            )

    arrays: dict[str, Any] = {
        **outputs,
        "full_neural_cache_schema_version": np.asarray(
            SCHEMA_VERSION,
            dtype=np.int64,
        ),
        "recording_id": np.asarray(recording_id),
        "source_filename": np.asarray(str(catalog["filename"])),
        "source_size_bytes": np.asarray(
            int(catalog["size_bytes"]),
            dtype=np.int64,
        ),
        "source_md5": np.asarray(str(catalog["md5"])),
        "source_frame_count": np.asarray(joiner.V.shape[1], dtype=np.int64),
        "source_cache_sha256": np.asarray(cache_sha256),
        "builder_sha256": np.asarray(builder_sha256),
    }
    _atomic_npz(destination, arrays)
    print(
        f"  wrote {destination.name} in "
        f"{time.perf_counter() - started:.1f}s",
        flush=True,
    )
    return destination


def _scalar(archive: np.lib.npyio.NpzFile, name: str) -> object:
    if name not in archive.files:
        raise ValueError(f"shard is missing {name!r}")
    return np.asarray(archive[name]).item()


def _valid_shard(
    path: Path,
    *,
    recording_id: str,
    base_indices: np.ndarray,
    cache_sha256: str,
    builder_sha256: str,
) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as archive:
            return bool(
                int(_scalar(archive, "full_neural_cache_schema_version"))
                == SCHEMA_VERSION
                and str(_scalar(archive, "recording_id")) == recording_id
                and str(_scalar(archive, "source_cache_sha256"))
                == cache_sha256
                and str(_scalar(archive, "builder_sha256")) == builder_sha256
                and np.array_equal(
                    archive["base_indices"].astype(np.int64),
                    base_indices,
                )
                and all(name in archive.files for name in (*NEURON_FIELDS, *VALIDITY_FIELDS))
            )
    except (OSError, KeyError, ValueError):
        return False


def _merge_shards(
    *,
    shard_paths: Sequence[Path],
    selected_base_indices: np.ndarray,
    cache_sha256: str,
    refined_cache_path: Path,
    builder_sha256: str,
    areas: Sequence[str],
    destination: Path,
) -> Path:
    by_index: dict[int, tuple[Path, int]] = {}
    source_manifest: list[dict[str, Any]] = []
    for path in shard_paths:
        with np.load(path, allow_pickle=False) as archive:
            base_indices = archive["base_indices"].astype(np.int64)
            for position, base_index in enumerate(base_indices):
                key = int(base_index)
                if key in by_index:
                    raise ValueError(f"duplicate full-neural row for base index {key}")
                by_index[key] = (path, position)
            source_manifest.append(
                {
                    "recording_id": str(_scalar(archive, "recording_id")),
                    "filename": str(_scalar(archive, "source_filename")),
                    "size_bytes": int(_scalar(archive, "source_size_bytes")),
                    "md5": str(_scalar(archive, "source_md5")),
                    "frame_count": int(_scalar(archive, "source_frame_count")),
                    "shard": path.name,
                }
            )
    missing = sorted(set(map(int, selected_base_indices)) - set(by_index))
    if missing:
        raise ValueError(
            f"full-neural shards are missing {len(missing)} selected rows"
        )

    first_path = shard_paths[0]
    with np.load(first_path, allow_pickle=False) as first:
        neurons_per_transition = int(first["current"].shape[1])
    shape = (len(selected_base_indices), neurons_per_transition)
    arrays: dict[str, Any] = {
        name: (
            np.zeros(shape, dtype=bool)
            if name in VALIDITY_FIELDS
            else np.full(shape, np.nan, dtype=np.float32)
        )
        for name in (*NEURON_FIELDS, *VALIDITY_FIELDS)
    }
    grouped: dict[Path, list[tuple[int, int]]] = {}
    for output_position, base_index in enumerate(selected_base_indices):
        path, shard_position = by_index[int(base_index)]
        grouped.setdefault(path, []).append((output_position, shard_position))
    for path, positions in grouped.items():
        output_positions = np.asarray([value[0] for value in positions], dtype=int)
        shard_positions = np.asarray([value[1] for value in positions], dtype=int)
        with np.load(path, allow_pickle=False) as archive:
            for name in (*NEURON_FIELDS, *VALIDITY_FIELDS):
                arrays[name][output_positions] = archive[name][shard_positions]

    configuration = {
        "schema_version": SCHEMA_VERSION,
        "areas": list(areas),
        "window_pairs": 20,
        "roles": [ROLE_LEAF, ROLE_CIRCLE],
        "minimum_frames_per_trial": MINIMUM_FRAMES_PER_TRIAL,
        "trial_response": "mean full deconvolved activity per eligible trial",
        "denominator": "leaf trial-response SD + circle trial-response SD",
        "source_cache": refined_cache_path.name,
    }
    arrays.update(
        base_indices=np.asarray(selected_base_indices, dtype=np.int64),
        full_neural_cache_schema_version=np.asarray(
            SCHEMA_VERSION,
            dtype=np.int64,
        ),
        full_neural_cache_config_json=np.asarray(
            json.dumps(configuration, sort_keys=True)
        ),
        source_refined_cache_sha256=np.asarray(cache_sha256),
        full_neural_cache_builder_sha256=np.asarray(builder_sha256),
        full_neural_source_manifest_json=np.asarray(
            json.dumps(
                sorted(source_manifest, key=lambda value: value["recording_id"]),
                sort_keys=True,
            )
        ),
    )
    _atomic_npz(destination, arrays)
    print(f"wrote merged full-neural cache {destination}", flush=True)
    return destination


def build(
    *,
    refined_cache_path: Path,
    destination: Path,
    shard_dir: Path,
    stage_dir: Path,
    data_cache: Path,
    database_path: Path,
    remote_root: str,
    areas: Sequence[str] = DEFAULT_AREAS,
    recordings: Iterable[str] | None = None,
    keep_staged: bool = False,
    allow_partial: bool = False,
) -> Path:
    refined_cache_path = refined_cache_path.resolve()
    cache = load_refined_cache(refined_cache_path)
    cache_sha256 = _sha256(refined_cache_path)
    builder_sha256 = _sha256(Path(__file__))
    area_set = set(map(str, areas))
    selected_base_indices = np.flatnonzero(
        cache.metadata["area"].astype(str).isin(area_set)
    )
    if len(selected_base_indices) == 0:
        raise ValueError(f"cache contains no rows for areas {sorted(area_set)}")
    metadata = cache.metadata.iloc[selected_base_indices]
    all_recordings = tuple(sorted(metadata["recording_id"].astype(str).unique()))
    requested = (
        set(all_recordings)
        if recordings is None
        else set(map(str, recordings))
    )
    unknown = sorted(requested - set(all_recordings))
    if unknown:
        raise ValueError(f"requested recordings are not in Objective B: {unknown}")
    processing = tuple(value for value in all_recordings if value in requested)

    shard_dir.mkdir(parents=True, exist_ok=True)
    stage_dir.mkdir(parents=True, exist_ok=True)
    shard_paths: list[Path] = []
    with ZhongDB(
        cache=data_cache,
        database=database_path,
        mount=False,
    ) as db:
        for recording_position, recording_id in enumerate(processing, start=1):
            recording_base_indices = selected_base_indices[
                metadata["recording_id"].astype(str).eq(recording_id).to_numpy()
            ]
            shard_path = shard_dir / f"{recording_id}.npz"
            print(
                f"{recording_position:02d}/{len(processing):02d} "
                f"{recording_id}: {len(recording_base_indices)} "
                f"area-transition rows",
                flush=True,
            )
            if _valid_shard(
                shard_path,
                recording_id=recording_id,
                base_indices=recording_base_indices,
                cache_sha256=cache_sha256,
                builder_sha256=builder_sha256,
            ):
                print(f"  reuse validated shard {shard_path.name}", flush=True)
                shard_paths.append(shard_path)
                continue

            catalog = _catalog_row(db, recording_id)
            staged = _stage_file(
                catalog,
                remote_root=remote_root,
                stage_dir=stage_dir,
            )
            session = cache.metadata.iloc[int(recording_base_indices[0])]
            joiner = Joiner(
                db,
                recording_id,
                experiment=str(session["behavior_session_id"]).split("::", 1)[0],
            )
            try:
                _build_recording_shard(
                    cache=cache,
                    recording_id=recording_id,
                    recording_base_indices=recording_base_indices,
                    joiner=joiner,
                    staged_path=staged,
                    catalog=catalog,
                    cache_sha256=cache_sha256,
                    builder_sha256=builder_sha256,
                    destination=shard_path,
                )
                shard_paths.append(shard_path)
            finally:
                del joiner
                gc.collect()
                if not keep_staged:
                    staged.unlink(missing_ok=True)

    final_indices = selected_base_indices
    if requested != set(all_recordings):
        if not allow_partial:
            raise ValueError(
                "a recording subset requires --allow-partial for a prototype cache"
            )
        final_indices = selected_base_indices[
            metadata["recording_id"].astype(str).isin(requested).to_numpy()
        ]
    return _merge_shards(
        shard_paths=shard_paths,
        selected_base_indices=final_indices,
        cache_sha256=cache_sha256,
        refined_cache_path=refined_cache_path,
        builder_sha256=builder_sha256,
        areas=areas,
        destination=destination.resolve(),
    )


def _parse_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refined-cache",
        type=Path,
        default=WORKSPACE
        / "modeling/experiments/cache/refined_neuron_transitions_w20.npz",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=WORKSPACE
        / "modeling/experiments/cache/full_neural_objective_b_w20.npz",
    )
    parser.add_argument(
        "--shard-dir",
        type=Path,
        default=WORKSPACE
        / "modeling/experiments/cache/full_neural_objective_b_shards",
    )
    parser.add_argument(
        "--stage-dir",
        type=Path,
        default=Path("/private/tmp/zhong-full-neural-stage"),
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
    parser.add_argument("--remote-root", default=DEFAULT_REMOTE_ROOT)
    parser.add_argument("--areas", default=",".join(DEFAULT_AREAS))
    parser.add_argument(
        "--recordings",
        default="",
        help="comma-separated prototype subset; empty means all recordings",
    )
    parser.add_argument("--keep-staged", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    arguments = parser.parse_args()
    build(
        refined_cache_path=arguments.refined_cache,
        destination=arguments.output,
        shard_dir=arguments.shard_dir,
        stage_dir=arguments.stage_dir,
        data_cache=arguments.data_cache,
        database_path=arguments.database,
        remote_root=arguments.remote_root,
        areas=_parse_csv(arguments.areas),
        recordings=(
            _parse_csv(arguments.recordings)
            if arguments.recordings.strip()
            else None
        ),
        keep_staged=arguments.keep_staged,
        allow_partial=arguments.allow_partial,
    )


if __name__ == "__main__":
    main()
