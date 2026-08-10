#!/usr/bin/env python3
"""Build causal window histories and window-independent session summaries.

The primary manifest contains only supervised and unsupervised train-1
before/after sessions.  Grating train-1 sessions are processed into a separate
control artifact and are never pooled with the primary cohorts.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Callable

import pandas as pd


WORKSPACE = Path(__file__).resolve().parents[1]
CODE = WORKSPACE / "code"
for root in (WORKSPACE, CODE):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import drive  # noqa: E402
from modeling.chronology import (  # noqa: E402
    WINDOW_ESTIMATOR,
    WINDOW_ESTIMATOR_VERSION,
    chronological_session_history,
)
from modeling.session_summary import (  # noqa: E402
    SESSION_ESTIMATOR,
    SESSION_ESTIMATOR_VERSION,
    session_trial_balanced_summary,
)


MANIFEST_COLUMNS = (
    "behavior_session_id",
    "behavior_key",
    "recording_id",
    "experiment",
    "mouse",
    "cohort",
    "stage",
    "moment",
    "trial_count",
)


def _manifest_query(cohort_sql: str) -> str:
    return f"""
        SELECT b.behavior_session_id, b.behavior_key, b.recording_id,
               b.experiment, b.mouse, b.cohort, e.stage, e.moment,
               b.trial_count
        FROM behavior_sessions AS b
        JOIN recordings AS r USING (recording_id)
        JOIN experiments AS e USING (experiment)
        WHERE r.has_behavior AND r.has_reduced_neural AND r.has_retinotopy
          AND e.stage = 'train1' AND e.moment IN ('before', 'after')
          AND {cohort_sql}
        ORDER BY b.cohort, b.mouse, e.moment, b.recording_id
    """


def manifest(db) -> pd.DataFrame:
    """Return the fixed 26-session supervised/unsupervised source manifest."""

    result = db.query(
        _manifest_query("b.cohort IN ('supervised', 'unsupervised')")
    )[list(MANIFEST_COLUMNS)]
    if len(result) != 26 or result["mouse"].nunique() != 13:
        raise ValueError(
            f"expected 26 train1 sessions from 13 mice, got {len(result)} "
            f"sessions from {result['mouse'].nunique()} mice"
        )
    cohort_counts = result.groupby("cohort")["mouse"].nunique().to_dict()
    if cohort_counts != {"supervised": 4, "unsupervised": 9}:
        raise ValueError(f"unexpected primary cohort membership: {cohort_counts}")
    return result


def grating_manifest(db) -> pd.DataFrame:
    """Return train-1 grating sessions for the separate control artifact."""

    result = db.query(_manifest_query("b.cohort = 'grating'"))[list(MANIFEST_COLUMNS)]
    if result.empty:
        # Retain the exact schema so the omissions artifact remains readable.
        return pd.DataFrame(columns=MANIFEST_COLUMNS)
    if not result["cohort"].eq("grating").all():
        raise RuntimeError("grating manifest contains a non-grating session")
    return result


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _manifest_sha256(sessions: pd.DataFrame) -> str:
    return _sha256_bytes(sessions.to_csv(index=False, lineterminator="\n").encode())


def _source_database_identity(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": int(stat.st_size),
        "modified_time_ns": int(stat.st_mtime_ns),
    }


def _code_identity() -> dict[str, str]:
    paths = (
        Path(__file__),
        Path(chronological_session_history.__code__.co_filename),
        Path(session_trial_balanced_summary.__code__.co_filename),
        CODE / "dprime" / "pairing.py",
        CODE / "joiner.py",
    )
    return {str(path.resolve()): _sha256_file(path) for path in sorted(set(paths))}


def _provenance_base(
    *,
    artifact: str,
    estimator: str,
    estimator_version: str,
    sessions: pd.DataFrame,
    database_path: Path,
    parameters: dict[str, object],
) -> dict[str, object]:
    reproducibility = {
        "artifact": artifact,
        "estimator": estimator,
        "estimator_version": estimator_version,
        "parameters": parameters,
        "source_manifest_sha256": _manifest_sha256(sessions),
        "source_database": _source_database_identity(database_path),
        "implementation_sha256": _code_identity(),
    }
    fingerprint = _sha256_bytes(
        json.dumps(reproducibility, sort_keys=True, separators=(",", ":")).encode()
    )
    return {**reproducibility, "reproducibility_fingerprint": fingerprint}


def _atomic_csv(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            frame.to_csv(stream, index=False, lineterminator="\n")
        os.replace(temporary, destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _atomic_json(content: dict[str, object], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(content, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _may_reuse(
    destination: Path,
    omissions_path: Path,
    provenance_path: Path,
    expected: dict[str, object],
    *,
    force: bool,
) -> bool:
    if (
        force
        or not destination.exists()
        or not omissions_path.exists()
        or not provenance_path.exists()
    ):
        return False
    try:
        observed = json.loads(provenance_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return (
        observed.get("reproducibility_fingerprint")
        == expected["reproducibility_fingerprint"]
        and observed.get("output_sha256") == _sha256_file(destination)
        and observed.get("omissions_sha256") == _sha256_file(omissions_path)
    )


def _write_artifact(
    frame: pd.DataFrame,
    omissions: pd.DataFrame,
    *,
    destination: Path,
    omissions_path: Path,
    provenance_path: Path,
    provenance: dict[str, object],
) -> None:
    _atomic_csv(frame, destination)
    _atomic_csv(omissions, omissions_path)
    completed = {
        **provenance,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "row_count": int(len(frame)),
        "column_count": int(len(frame.columns)),
        "columns": list(frame.columns),
        "omission_count": int(len(omissions)),
        "output_sha256": _sha256_file(destination),
        "omissions_sha256": _sha256_file(omissions_path),
    }
    _atomic_json(completed, provenance_path)


def _omission_record(session, error: Exception, *, artifact: str) -> dict[str, object]:
    return {
        "artifact": artifact,
        "behavior_session_id": session.behavior_session_id,
        "recording_id": session.recording_id,
        "mouse": session.mouse,
        "cohort": session.cohort,
        "moment": session.moment,
        "error_type": type(error).__name__,
        "reason": str(error),
    }


def _run_sessions(
    sessions: pd.DataFrame,
    compute: Callable[[Any, int], pd.DataFrame],
    *,
    artifact: str,
    omit_any_value_error: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    omissions: list[dict[str, object]] = []
    for index, session in enumerate(sessions.itertuples(index=False), start=1):
        started = time.perf_counter()
        try:
            result = compute(session, index)
        except ValueError as error:
            expected = "too few chronological trials" in str(error)
            if not (omit_any_value_error or expected):
                raise
            omissions.append(_omission_record(session, error, artifact=artifact))
            print(
                f"{artifact} {index:02d}/{len(sessions)} "
                f"OMIT {session.behavior_session_id}: {error}"
            )
            continue
        frames.append(result)
        print(
            f"{artifact} {index:02d}/{len(sessions)} "
            f"{session.behavior_session_id} {len(result):,} rows "
            f"{time.perf_counter() - started:.1f}s"
        )
        gc.collect()
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    omission_columns = (
        "artifact",
        "behavior_session_id",
        "recording_id",
        "mouse",
        "cohort",
        "moment",
        "error_type",
        "reason",
    )
    return combined, pd.DataFrame.from_records(omissions, columns=omission_columns)


def recompute_window_history(
    db,
    sessions: pd.DataFrame,
    window_pairs: int,
    output: Path,
    *,
    database_path: Path,
    force: bool,
    neurons_per_area: int,
    neuron_seed: int,
    threshold: float,
    min_frames: int,
) -> Path:
    artifact = f"window_dprime_chronological_w{window_pairs}"
    destination = output / f"{artifact}.csv"
    omissions_path = output / f"{artifact}_omissions.csv"
    provenance_path = output / f"{artifact}_provenance.json"
    parameters: dict[str, object] = {
        "window_pairs": int(window_pairs),
        "neurons_per_area": int(neurons_per_area),
        "neuron_sampling_base_seed": int(neuron_seed),
        "selectivity_threshold": float(threshold),
        "minimum_eligible_frames_per_trial": int(min_frames),
        "roles": [2, 0],
        "cohorts": ["supervised", "unsupervised"],
        "stage": "train1",
        "moments": ["before", "after"],
    }
    provenance = _provenance_base(
        artifact=artifact,
        estimator=WINDOW_ESTIMATOR,
        estimator_version=WINDOW_ESTIMATOR_VERSION,
        sessions=sessions,
        database_path=database_path,
        parameters=parameters,
    )
    if _may_reuse(
        destination, omissions_path, provenance_path, provenance, force=force
    ):
        print(f"reuse {destination}")
        return destination

    history, omissions = _run_sessions(
        sessions,
        lambda session, index: chronological_session_history(
            db,
            session,
            window_pairs=window_pairs,
            neurons_per_area=neurons_per_area,
            neuron_seed=neuron_seed,
            threshold=threshold,
            min_frames=min_frames,
            verify=index == 1,
        ),
        artifact=artifact,
        omit_any_value_error=False,
    )
    if history.empty:
        raise ValueError(f"{artifact} produced no session histories")
    _write_artifact(
        history,
        omissions,
        destination=destination,
        omissions_path=omissions_path,
        provenance_path=provenance_path,
        provenance=provenance,
    )
    return destination


def recompute_session_summary(
    db,
    sessions: pd.DataFrame,
    output: Path,
    *,
    database_path: Path,
    force: bool,
    neurons_per_area: int,
    neuron_seed: int,
    threshold: float,
    min_frames: int,
    grating_control: bool = False,
) -> Path:
    suffix = "_grating" if grating_control else ""
    artifact = f"session_dprime_trial_balanced{suffix}"
    destination = output / f"{artifact}.csv"
    omissions_path = output / f"{artifact}_omissions.csv"
    provenance_path = output / f"{artifact}_provenance.json"
    cohorts = ["grating"] if grating_control else ["supervised", "unsupervised"]
    parameters: dict[str, object] = {
        "neurons_per_area": int(neurons_per_area),
        "neuron_sampling_base_seed": int(neuron_seed),
        "selectivity_threshold": float(threshold),
        "minimum_eligible_frames_per_trial": int(min_frames),
        "roles": [2, 0],
        "cohorts": cohorts,
        "stage": "train1",
        "moments": ["before", "after"],
        "grating_control": bool(grating_control),
    }
    provenance = _provenance_base(
        artifact=artifact,
        estimator=SESSION_ESTIMATOR,
        estimator_version=SESSION_ESTIMATOR_VERSION,
        sessions=sessions,
        database_path=database_path,
        parameters=parameters,
    )
    if _may_reuse(
        destination, omissions_path, provenance_path, provenance, force=force
    ):
        print(f"reuse {destination}")
        return destination

    if sessions.empty:
        summary = pd.DataFrame()
        omissions = pd.DataFrame.from_records(
            [
                {
                    "artifact": artifact,
                    "behavior_session_id": pd.NA,
                    "recording_id": pd.NA,
                    "mouse": pd.NA,
                    "cohort": "grating" if grating_control else pd.NA,
                    "moment": pd.NA,
                    "error_type": "UnavailableControlData",
                    "reason": "no eligible train1 before/after grating sessions were found",
                }
            ]
        )
    else:
        summary, omissions = _run_sessions(
            sessions,
            lambda session, _: session_trial_balanced_summary(
                db,
                session,
                neurons_per_area=neurons_per_area,
                threshold=threshold,
                min_frames=min_frames,
                roles=(2, 0),
                neuron_seed=neuron_seed,
            ),
            artifact=artifact,
            # Grating is a control: unavailable roles/areas are recorded rather
            # than allowed to abort the primary recomputation.
            omit_any_value_error=grating_control,
        )
    _write_artifact(
        summary,
        omissions,
        destination=destination,
        omissions_path=omissions_path,
        provenance_path=provenance_path,
        provenance=provenance,
    )
    return destination


def _write_source_manifest(
    sessions: pd.DataFrame,
    destination: Path,
) -> None:
    current = sessions.to_csv(index=False, lineterminator="\n")
    if destination.exists() and destination.read_text() == current:
        return
    _atomic_csv(sessions, destination)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--windows",
        nargs="+",
        type=int,
        default=[10, 20, 40],
        help="strictly chronological trial-pairs per window (default: 10 20 40)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=WORKSPACE / "modeling" / "window_histories",
    )
    parser.add_argument("--neurons-per-area", type=int, default=1000)
    parser.add_argument(
        "--neuron-seed",
        type=int,
        default=2025,
        help="base seed combined with each recording id (default: 2025)",
    )
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--min-frames", type=int, default=3)
    parser.add_argument(
        "--skip-grating-control",
        action="store_true",
        help="do not build the separate grating session-summary control",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    windows = list(dict.fromkeys(arguments.windows))
    if any(window < 2 for window in windows):
        raise ValueError("all window sizes must be at least two pairs")
    if arguments.neurons_per_area < 1 or arguments.min_frames < 1:
        raise ValueError("neurons-per-area and min-frames must be positive")
    if not 0 <= arguments.neuron_seed <= 2**32 - 1:
        raise ValueError("neuron-seed must fit in an unsigned 32-bit integer")
    if not math.isfinite(arguments.threshold) or arguments.threshold < 0:
        raise ValueError("threshold must be finite and non-negative")

    database_path = WORKSPACE / "data" / "cache" / "zhong.duckdb"
    db = drive.setup(
        cache=str(database_path.parent),
        database=str(database_path),
        mount=False,
        report=False,
    )
    primary = manifest(db)
    grating = grating_manifest(db)
    arguments.output.mkdir(parents=True, exist_ok=True)
    _write_source_manifest(primary, arguments.output / "source_manifest.csv")
    _write_source_manifest(
        grating, arguments.output / "source_manifest_grating.csv"
    )

    outputs = [
        recompute_session_summary(
            db,
            primary,
            arguments.output,
            database_path=database_path,
            force=arguments.force,
            neurons_per_area=arguments.neurons_per_area,
            neuron_seed=arguments.neuron_seed,
            threshold=arguments.threshold,
            min_frames=arguments.min_frames,
        )
    ]
    if not arguments.skip_grating_control:
        outputs.append(
            recompute_session_summary(
                db,
                grating,
                arguments.output,
                database_path=database_path,
                force=arguments.force,
                neurons_per_area=arguments.neurons_per_area,
                neuron_seed=arguments.neuron_seed,
                threshold=arguments.threshold,
                min_frames=arguments.min_frames,
                grating_control=True,
            )
        )
    outputs.extend(
        recompute_window_history(
            db,
            primary,
            window,
            arguments.output,
            database_path=database_path,
            force=arguments.force,
            neurons_per_area=arguments.neurons_per_area,
            neuron_seed=arguments.neuron_seed,
            threshold=arguments.threshold,
            min_frames=arguments.min_frames,
        )
        for window in windows
    )
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
