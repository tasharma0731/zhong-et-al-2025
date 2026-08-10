#!/usr/bin/env python3
"""Build an auditable cache of chronological same-neuron transitions.

The cache is deliberately stricter than a generic ``.npz`` export.  It keeps
the neuron identifiers and retinotopy values used for every transition, and it
records a digest of the summary history that defined the windows.  An existing
cache is reused only when that provenance exactly matches the requested build.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd


WORKSPACE = Path(__file__).resolve().parents[1]
CODE = WORKSPACE / "code"
for source_root in (WORKSPACE, CODE):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

import drive  # noqa: E402
from dprime import AREAS, balanced_area_neurons, eligible_trial_frames  # noqa: E402
from joiner import Joiner  # noqa: E402
from modeling.chronology import (  # noqa: E402
    WINDOW_CONSTRUCTION,
    WINDOW_ESTIMATOR,
    WINDOW_ESTIMATOR_VERSION,
    chronological_trial_pairs_and_windows,
)
from modeling.session_summary import (  # noqa: E402
    DENOMINATOR_ESTIMATOR,
    NEURON_SEED_METHOD,
    TRIAL_RESPONSE_ESTIMATOR,
    dprime_for_balanced_trial_ids,
    recording_neuron_seed,
)


CACHE_SCHEMA_VERSION = 3
CHRONOLOGY_ALGORITHM = f"{WINDOW_ESTIMATOR}@{WINDOW_ESTIMATOR_VERSION}"
NEURON_SAMPLING_BASE_SEED = 2025


def _manifest(db) -> pd.DataFrame:
    result = db.query(
        """
        SELECT b.behavior_session_id, b.behavior_key, b.recording_id,
               b.experiment, b.mouse, b.cohort, e.stage, e.moment,
               b.trial_count
        FROM behavior_sessions AS b
        JOIN recordings AS r USING (recording_id)
        JOIN experiments AS e USING (experiment)
        WHERE r.has_behavior AND r.has_reduced_neural AND r.has_retinotopy
          AND e.stage = 'train1' AND e.moment IN ('before', 'after')
          AND b.cohort IN ('supervised', 'unsupervised')
        ORDER BY b.cohort, b.mouse, e.moment, b.recording_id
        """
    )
    if len(result) != 26 or result["mouse"].nunique() != 13:
        raise ValueError("unexpected train1 manifest")
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _zscore_coordinates(values: np.ndarray) -> np.ndarray:
    center = np.median(values, axis=0)
    q25, q75 = np.quantile(values, [0.25, 0.75], axis=0)
    scale = np.where(q75 - q25 > 1e-8, q75 - q25, 1.0)
    return (values - center) / scale


def _chronological_history(history: pd.DataFrame) -> bool:
    if "chronological" not in history:
        return False
    values = history["chronological"]
    if pd.api.types.is_bool_dtype(values):
        return bool(values.all())
    normalized = values.astype(str).str.strip().str.lower()
    return bool(normalized.isin({"true", "1", "yes"}).all())


def _require_history_value(
    history: pd.DataFrame, column: str, expected: object
) -> None:
    """Validate estimator configuration when the provenance column is present."""

    if column not in history:
        return
    values = history[column].drop_duplicates()
    if len(values) != 1 or str(values.iloc[0]) != str(expected):
        raise ValueError(f"history requires {column}={expected!r}")


def _scalar(archive: np.lib.npyio.NpzFile, name: str) -> object:
    if name not in archive.files:
        raise ValueError(f"cache is missing provenance field {name!r}")
    return np.asarray(archive[name]).item()


def _validate_reusable_cache(destination: Path, expected: dict[str, object]) -> None:
    """Reject silent reuse of an old or differently configured cache."""

    try:
        with np.load(destination, allow_pickle=False) as archive:
            actual = {
                "cache_schema_version": int(_scalar(archive, "cache_schema_version")),
                "source_history_sha256": str(_scalar(archive, "source_history_sha256")),
                "window_pairs": int(_scalar(archive, "window_pairs")),
                "neurons_per_area": int(_scalar(archive, "neurons_per_area")),
                "chronology_algorithm": str(_scalar(archive, "chronology_algorithm")),
            }
            required = {
                "recording_id",
                "neuron_id",
                "area_id",
                "current_last_trial_id",
                "next_first_trial_id",
                "current_denominator",
                "next_denominator",
                "current_valid_denominator",
                "next_valid_denominator",
            }
            missing = required - set(archive.files)
            if missing:
                raise ValueError(f"cache is missing audit arrays {sorted(missing)}")
    except (OSError, KeyError, ValueError) as error:
        raise ValueError(
            f"cannot safely reuse {destination}: {error}; rerun with --force"
        ) from error
    mismatches = {
        name: (actual[name], value)
        for name, value in expected.items()
        if actual.get(name) != value
    }
    if mismatches:
        details = ", ".join(
            f"{name}={old!r} (requested {new!r})"
            for name, (old, new) in mismatches.items()
        )
        raise ValueError(
            f"refusing stale cache reuse for {destination}: {details}; rerun with --force"
        )


def build(
    history_path: Path,
    destination: Path,
    *,
    neurons_per_area: int = 1000,
    force: bool = False,
) -> Path:
    history_path = history_path.resolve()
    if not history_path.is_file():
        raise FileNotFoundError(history_path)
    history_sha = _sha256(history_path)
    history = pd.read_csv(history_path)
    required_history = {
        "behavior_session_id",
        "area",
        "start_pair_id",
        "pair_count",
        "first_trial_id",
        "last_trial_id",
        "progress",
        "mean_run_speed",
    }
    missing_history = required_history - set(history.columns)
    if missing_history:
        raise ValueError(f"history is missing {sorted(missing_history)}")
    if not _chronological_history(history):
        raise ValueError(
            "history must be produced by chronological_session_history; "
            "ordinally paired histories are unsafe for forecasting"
        )
    for column, expected in (
        ("estimator", WINDOW_ESTIMATOR),
        ("estimator_version", WINDOW_ESTIMATOR_VERSION),
        ("window_construction", WINDOW_CONSTRUCTION),
        ("role_positive", 2),
        ("role_negative", 0),
        ("minimum_eligible_frames_per_trial", 3),
        ("requested_neurons_per_area", int(neurons_per_area)),
        ("neuron_sampling_base_seed", NEURON_SAMPLING_BASE_SEED),
        ("neuron_sampling_seed_method", NEURON_SEED_METHOD),
        ("trial_response_estimator", TRIAL_RESPONSE_ESTIMATOR),
        ("denominator_estimator", DENOMINATOR_ESTIMATOR),
    ):
        _require_history_value(history, column, expected)
    window_sizes = history["pair_count"].astype(int).unique()
    if len(window_sizes) != 1:
        raise ValueError("history must contain exactly one window size")
    window_pairs = int(window_sizes[0])
    expected_provenance: dict[str, object] = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "source_history_sha256": history_sha,
        "window_pairs": window_pairs,
        "neurons_per_area": int(neurons_per_area),
        "chronology_algorithm": CHRONOLOGY_ALGORITHM,
    }
    if destination.exists() and not force:
        _validate_reusable_cache(destination, expected_provenance)
        print(f"reuse validated cache {destination}")
        return destination

    duplicated = history.duplicated(["behavior_session_id", "area", "start_pair_id"])
    if duplicated.any():
        raise ValueError("history contains duplicate session/area/window rows")
    lookup = history.set_index(["behavior_session_id", "area", "start_pair_id"])[
        ["progress", "mean_run_speed", "first_trial_id", "last_trial_id"]
    ]

    cache_root = WORKSPACE / "data" / "cache"
    db = drive.setup(
        cache=str(cache_root),
        database=str(cache_root / "zhong.duckdb"),
        mount=False,
        report=False,
    )

    metadata_names = (
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
    metadata: dict[str, list[object]] = {name: [] for name in metadata_names}
    current_values: list[np.ndarray] = []
    next_values: list[np.ndarray] = []
    current_denominators: list[np.ndarray] = []
    next_denominators: list[np.ndarray] = []
    current_valid: list[np.ndarray] = []
    next_valid: list[np.ndarray] = []
    neuron_ids: list[np.ndarray] = []
    fine_area_ids: list[np.ndarray] = []
    cortical_x: list[np.ndarray] = []
    cortical_y: list[np.ndarray] = []
    cortical_x_raw: list[np.ndarray] = []
    cortical_y_raw: list[np.ndarray] = []

    sessions = _manifest(db)
    for session_index, session in enumerate(sessions.itertuples(index=False), start=1):
        started = time.perf_counter()
        joiner = Joiner(
            db,
            session.recording_id,
            experiment=session.experiment,
            behavior_key=session.behavior_key,
        )
        frames = eligible_trial_frames(joiner.frames, roles=(2, 0), min_frames=3)
        pairs, windows = chronological_trial_pairs_and_windows(
            frames, size=window_pairs, role_a=2, role_b=0
        )
        if len(windows) < 2:
            print(f"skip {session.behavior_session_id}: fewer than two chronological windows")
            del joiner
            gc.collect()
            continue
        if not bool(
            (
                windows["last_trial_id"].to_numpy()[:-1]
                < windows["first_trial_id"].to_numpy()[1:]
            ).all()
        ):
            raise RuntimeError(f"non-chronological windows for {session.recording_id}")

        neurons = balanced_area_neurons(
            joiner.neurons,
            areas=AREAS,
            per_area=neurons_per_area,
            seed=recording_neuron_seed(
                session.recording_id, NEURON_SAMPLING_BASE_SEED
            ),
        )
        window_results = []
        for window in windows.itertuples(index=False):
            block = pairs.loc[
                pairs["pair_id"].between(window.start_pair_id, window.stop_pair_id)
            ]
            window_results.append(
                dprime_for_balanced_trial_ids(
                    joiner.U,
                    joiner.V,
                    frames,
                    neuron_ids=neurons["neuron_id"].to_numpy(dtype=np.int64),
                    trial_ids_by_role={
                        2: block["trial_a_id"].to_numpy(dtype=np.int64),
                        0: block["trial_b_id"].to_numpy(dtype=np.int64),
                    },
                    roles=(2, 0),
                )
            )
        dprime = np.stack([result.values for result in window_results])
        denominators = np.stack([result.denominators for result in window_results])
        valid_denominator = np.stack(
            [result.valid_denominator for result in window_results]
        )
        neuron_areas = neurons["area_group"].to_numpy()
        raw_coordinates = neurons[["cortical_x", "cortical_y"]].to_numpy(dtype=float)
        normalized_coordinates = _zscore_coordinates(raw_coordinates)

        for area in AREAS:
            area_mask = neuron_areas == area
            if int(area_mask.sum()) != neurons_per_area:
                raise ValueError(f"unexpected neuron count for {session.recording_id}/{area}")
            area_neurons = neurons.loc[area_mask]
            for index in range(len(windows) - 1):
                current_window = windows.iloc[index]
                next_window = windows.iloc[index + 1]
                current_last = int(current_window.last_trial_id)
                next_first = int(next_window.first_trial_id)
                if current_last >= next_first:
                    raise RuntimeError(
                        f"forecast leakage in {session.recording_id}: "
                        f"current ends {current_last}, next starts {next_first}"
                    )
                key = (
                    session.behavior_session_id,
                    area,
                    int(current_window.start_pair_id),
                )
                if key not in lookup.index:
                    raise KeyError(f"history has no row for {key}")
                row = lookup.loc[key]
                if (
                    int(row["first_trial_id"]) != int(current_window.first_trial_id)
                    or int(row["last_trial_id"]) != current_last
                ):
                    raise ValueError(f"history/raw window mismatch for {key}")

                values = {
                    "behavior_session_id": session.behavior_session_id,
                    "recording_id": session.recording_id,
                    "mouse": session.mouse,
                    "cohort": session.cohort,
                    "moment": session.moment,
                    "area": area,
                    "current_start_pair_id": int(current_window.start_pair_id),
                    "next_start_pair_id": int(next_window.start_pair_id),
                    "current_first_trial_id": int(current_window.first_trial_id),
                    "current_last_trial_id": current_last,
                    "next_first_trial_id": next_first,
                    "next_last_trial_id": int(next_window.last_trial_id),
                    "current_progress": float(row["progress"]),
                    "current_run_speed": float(row["mean_run_speed"]),
                }
                for name, value in values.items():
                    metadata[name].append(value)
                current_values.append(dprime[index, area_mask].astype(np.float32))
                next_values.append(dprime[index + 1, area_mask].astype(np.float32))
                current_denominators.append(
                    denominators[index, area_mask].astype(np.float32)
                )
                next_denominators.append(
                    denominators[index + 1, area_mask].astype(np.float32)
                )
                current_valid.append(valid_denominator[index, area_mask].astype(bool))
                next_valid.append(
                    valid_denominator[index + 1, area_mask].astype(bool)
                )
                neuron_ids.append(area_neurons["neuron_id"].to_numpy(np.int64))
                fine_area_ids.append(area_neurons["area_id"].to_numpy(np.int64))
                cortical_x.append(normalized_coordinates[area_mask, 0].astype(np.float32))
                cortical_y.append(normalized_coordinates[area_mask, 1].astype(np.float32))
                cortical_x_raw.append(raw_coordinates[area_mask, 0].astype(np.float32))
                cortical_y_raw.append(raw_coordinates[area_mask, 1].astype(np.float32))

        print(
            f"{session_index:02d}/{len(sessions)} {session.behavior_session_id} "
            f"{len(windows) - 1} transitions x {len(AREAS)} areas "
            f"{time.perf_counter() - started:.1f}s"
        )
        del joiner, dprime, denominators, valid_denominator, window_results
        gc.collect()

    if not current_values:
        raise ValueError("no chronological transitions were available")
    build_config = {
        **expected_provenance,
        "source_history_path": str(history_path),
        "areas": list(AREAS),
        "roles": [2, 0],
        "minimum_frames_per_trial": 3,
        "window_estimator": WINDOW_ESTIMATOR,
        "window_estimator_version": WINDOW_ESTIMATOR_VERSION,
        "window_construction": WINDOW_CONSTRUCTION,
        "trial_response_estimator": TRIAL_RESPONSE_ESTIMATOR,
        "denominator_estimator": DENOMINATOR_ESTIMATOR,
        "neuron_sampling_base_seed": NEURON_SAMPLING_BASE_SEED,
        "neuron_sampling_seed_method": NEURON_SEED_METHOD,
    }
    arrays: dict[str, np.ndarray] = {
        "current_dprime": np.stack(current_values),
        "next_dprime": np.stack(next_values),
        "current_denominator": np.stack(current_denominators),
        "next_denominator": np.stack(next_denominators),
        "current_valid_denominator": np.stack(current_valid),
        "next_valid_denominator": np.stack(next_valid),
        "neuron_id": np.stack(neuron_ids),
        "area_id": np.stack(fine_area_ids),
        "cortical_x": np.stack(cortical_x),
        "cortical_y": np.stack(cortical_y),
        "cortical_x_raw": np.stack(cortical_x_raw),
        "cortical_y_raw": np.stack(cortical_y_raw),
        "window_pairs": np.asarray(window_pairs, dtype=np.int64),
        "neurons_per_area": np.asarray(neurons_per_area, dtype=np.int64),
        "cache_schema_version": np.asarray(CACHE_SCHEMA_VERSION, dtype=np.int64),
        "chronology_algorithm": np.asarray(CHRONOLOGY_ALGORITHM),
        "source_history_sha256": np.asarray(history_sha),
        "source_history_path": np.asarray(str(history_path)),
        "build_config_json": np.asarray(json.dumps(build_config, sort_keys=True)),
    }
    for name, values in metadata.items():
        arrays[name] = np.asarray(values)
    if not bool((arrays["current_last_trial_id"] < arrays["next_first_trial_id"]).all()):
        raise RuntimeError("cache contains a non-chronological transition")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(destination)
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--history",
        type=Path,
        default=(
            WORKSPACE
            / "modeling"
            / "window_histories"
            / "window_dprime_chronological_w20.csv"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            WORKSPACE
            / "modeling"
            / "cluster_cache"
            / "neuron_transitions_w20.npz"
        ),
    )
    parser.add_argument("--neurons-per-area", type=int, default=1000)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    print(
        build(
            arguments.history,
            arguments.output,
            neurons_per_area=arguments.neurons_per_area,
            force=arguments.force,
        )
    )


if __name__ == "__main__":
    main()
