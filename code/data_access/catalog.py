from __future__ import annotations

import json
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from drive import DriveDataError


LAYERS = {
    "imaging_behavior": "behavior",
    "reduced_neural": "reduced_neural",
    "full_neural": "full_neural",
    "retinotopy": "retinotopy",
}
COHORT_ORDER = ("supervised", "unsupervised", "grating", "naive")


def files(records: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    rows = []
    for record in records:
        name = str(record["name"])
        size = int(record["size_bytes"])
        rows.append(
            {
                "filename": name,
                "figshare_file_id": int(record["id"]),
                "category": str(record["category"]),
                "experiment": text(record.get("experiment")),
                "recording_id": text(record.get("recording_id")),
                "retinotopy_id": text(record.get("retinotopy_id")),
                "size_bytes": size,
                "size_mib": size / 2**20,
                "size_gib": size / 2**30,
                "md5": str(record["md5"]).lower(),
                "relative_path": str(record.get("relative_path") or f"data/{name}"),
            }
        )
    return pd.DataFrame(rows).sort_values("filename", ignore_index=True)


def experiment_rows(index: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for experiment, entries in index.items():
        normalized = plain(entries)
        normalized = [normalized] if isinstance(normalized, Mapping) else normalized
        if not isinstance(normalized, list):
            raise DriveDataError(f"Imaging index {experiment!r} should contain rows")
        rows.extend(
            experiment_row(str(experiment), source_row, entry)
            for source_row, entry in enumerate(normalized)
        )
    return rows


def experiment_row(
    experiment: str,
    source_row: int,
    entry: Any,
) -> dict[str, Any]:
    if not isinstance(entry, Mapping):
        raise DriveDataError(f"Imaging index {experiment!r}[{source_row}] is not a row")
    source = plain(entry.get("source", entry))
    if not isinstance(source, Mapping):
        raise DriveDataError(f"Imaging index {experiment!r}[{source_row}] has no source row")

    mouse = str(source["mname"])
    date = str(source["datexp"])
    recording_block = block(source["blk"])
    return {
        "experiment": experiment,
        "source_row": source_row,
        "recording_id": str(
            entry.get("recording_id", f"{mouse}_{date}_{recording_block}")
        ),
        "retinotopy_id": str(entry.get("retinotopy_id", f"{mouse}_{date}")),
        "mouse": mouse,
        "date": date,
        "block": recording_block,
        "session_number": plain(source.get("sess#")),
        "is_2p": bool(source.get("is2p")) if source.get("is2p") is not None else None,
        "gender": text(source.get("Gender")),
        "reward_type": text(source.get("rewType")),
        "stimulus": text(source.get("stim")),
        "stimulus_type": text(source.get("stimtype")),
        "depth_json": json.dumps(plain(source.get("depth", []))),
        "stimulus_ids_json": json.dumps(plain(source.get("stim_id", []))),
        "note": text(source.get("Note")),
        "source_json": json.dumps(plain(source), sort_keys=True),
    }


def tables(
    release: Mapping[str, Any],
    files: pd.DataFrame,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, pd.DataFrame]:
    experiment_rows = pd.DataFrame(rows).sort_values(
        ["experiment", "recording_id", "source_row"], ignore_index=True
    )
    behavior_sessions = build_behavior_sessions(experiment_rows, files)
    memberships = build_memberships(experiment_rows)
    experiments = build_experiments(memberships)
    recording_files = build_recording_files(files, memberships)
    recordings = build_recordings(memberships, recording_files)
    mice = build_mice(memberships, recordings)
    release_row = {str(key): plain(value) for key, value in release.items()}
    return {
        "release": pd.DataFrame([release_row]),
        "files": files,
        "experiment_rows": experiment_rows,
        "behavior_sessions": behavior_sessions,
        "memberships": memberships,
        "experiments": experiments,
        "recording_files": recording_files,
        "recordings": recordings,
        "mice": mice,
    }


def build_behavior_sessions(
    experiment_rows: pd.DataFrame,
    files: pd.DataFrame,
) -> pd.DataFrame:
    sessions = experiment_rows.copy()
    swap = sessions["stimulus_type"].isin(("swap1", "swap2"))
    sessions["behavior_key"] = sessions["recording_id"]
    sessions.loc[swap, "behavior_key"] += "_" + sessions.loc[swap, "stimulus_type"]
    sessions.insert(
        0,
        "behavior_session_id",
        sessions["experiment"] + "::" + sessions["behavior_key"],
    )
    sessions.insert(2, "cohort", sessions["experiment"].map(cohort))

    behavior_files = files.loc[
        files["category"] == "imaging_behavior",
        [
            "experiment",
            "filename",
            "figshare_file_id",
            "category",
            "size_bytes",
            "size_mib",
            "size_gib",
            "md5",
            "relative_path",
        ],
    ]
    sessions = sessions.merge(
        behavior_files,
        on="experiment",
        how="left",
        validate="many_to_one",
    )
    if sessions["filename"].isna().any():
        missing = sessions.loc[sessions["filename"].isna(), "experiment"].unique()
        raise DriveDataError(f"Behavior files are missing for {sorted(missing)}")
    if not sessions["behavior_session_id"].is_unique:
        raise DriveDataError("Behavior session IDs must be unique")
    return sessions.sort_values(
        ["experiment", "recording_id", "source_row"], ignore_index=True
    )


def build_memberships(experiment_rows: pd.DataFrame) -> pd.DataFrame:
    identity = ["experiment", "recording_id", "retinotopy_id", "mouse", "date", "block"]
    memberships = (
        experiment_rows.groupby(identity, dropna=False, sort=True)
        .size()
        .rename("source_row_count")
        .reset_index()
    )
    memberships.insert(1, "cohort", memberships["experiment"].map(cohort))
    return memberships


def build_experiments(memberships: pd.DataFrame) -> pd.DataFrame:
    experiments = (
        memberships.groupby("experiment", sort=True)
        .agg(
            recording_count=("recording_id", "nunique"),
            mouse_count=("mouse", "nunique"),
        )
        .reset_index()
    )
    experiments.insert(1, "cohort", experiments["experiment"].map(cohort))
    experiments.insert(2, "stage", experiments["experiment"].map(stage))
    experiments.insert(3, "moment", experiments["experiment"].map(moment))
    return experiments


def build_recording_files(
    files: pd.DataFrame,
    memberships: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "recording_id",
        "experiment",
        "layer",
        "filename",
        "figshare_file_id",
        "category",
        "size_bytes",
        "size_mib",
        "size_gib",
        "md5",
        "relative_path",
    ]
    neural = files[files["category"].isin(("reduced_neural", "full_neural"))].copy()
    neural["layer"] = neural["category"].map(LAYERS)
    neural["experiment"] = None

    behavior = memberships[["experiment", "recording_id"]].merge(
        files[files["category"] == "imaging_behavior"],
        on="experiment",
        how="inner",
        validate="many_to_one",
        suffixes=("", "_file"),
    )
    behavior["layer"] = "behavior"

    retinotopy = memberships[["recording_id", "retinotopy_id"]].drop_duplicates().merge(
        files[files["category"] == "retinotopy"],
        on="retinotopy_id",
        how="inner",
        validate="many_to_one",
        suffixes=("", "_file"),
    )
    retinotopy["layer"] = "retinotopy"
    retinotopy["experiment"] = None

    return (
        pd.concat(
            [neural[columns], behavior[columns], retinotopy[columns]],
            ignore_index=True,
        )
        .drop_duplicates(["recording_id", "experiment", "layer", "filename"])
        .sort_values(["recording_id", "layer", "experiment"], na_position="first")
        .reset_index(drop=True)
    )


def build_recordings(
    memberships: pd.DataFrame,
    recording_files: pd.DataFrame,
) -> pd.DataFrame:
    identity = memberships[
        ["recording_id", "retinotopy_id", "mouse", "date", "block"]
    ].drop_duplicates()
    recordings = identity.set_index("recording_id")
    recordings["experiment_count"] = memberships.groupby("recording_id")[
        "experiment"
    ].nunique()
    recordings["experiments_json"] = memberships.groupby("recording_id")[
        "experiment"
    ].agg(lambda values: json.dumps(sorted(set(values))))

    layers = recording_files.assign(present=True).pivot_table(
        index="recording_id",
        columns="layer",
        values="present",
        aggfunc="any",
        fill_value=False,
    )
    for layer in LAYERS.values():
        recordings[f"has_{layer}"] = layers[layer] if layer in layers else False

    linked_files = recording_files.groupby("recording_id").agg(
        linked_file_count=("filename", "nunique"),
        linked_bytes=("size_bytes", "sum"),
    )
    recordings = recordings.join(linked_files).reset_index()
    recordings["linked_gib"] = recordings["linked_bytes"] / 2**30
    return recordings.sort_values(["mouse", "date", "block"], ignore_index=True)


def build_mice(
    memberships: pd.DataFrame,
    recordings: pd.DataFrame,
) -> pd.DataFrame:
    cohorts = (
        memberships[["mouse", "cohort"]]
        .drop_duplicates()
        .groupby("mouse")["cohort"]
        .agg(
            primary_cohort=primary_cohort,
            cohorts_json=lambda values: json.dumps(sorted(set(values))),
        )
    )
    summary = recordings.groupby("mouse").agg(
        recording_count=("recording_id", "nunique"),
        has_full_neural=("has_full_neural", "any"),
        has_reduced_neural=("has_reduced_neural", "any"),
    )
    return cohorts.join(summary).reset_index().sort_values("mouse", ignore_index=True)


def plain(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return plain(value.item()) if value.ndim == 0 else [plain(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, Mapping):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    return value


def text(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(plain(value)).strip()
    return normalized or None


def block(value: Any) -> str:
    normalized = plain(value)
    if isinstance(normalized, float) and normalized.is_integer():
        return str(int(normalized))
    return str(normalized).strip()


def cohort(experiment: str) -> str:
    if experiment.startswith("sup_"):
        return "supervised"
    if experiment.startswith("unsup_"):
        return "unsupervised"
    if experiment.startswith("naive_"):
        return "naive"
    return "grating" if experiment.endswith("_grating") else "other"


def stage(experiment: str) -> str | None:
    return next(
        (name for name in ("train1", "test1", "train2", "test2", "test3") if name in experiment),
        None,
    )


def moment(experiment: str) -> str:
    if "_before_" in experiment:
        return "before"
    if "_after_" in experiment:
        return "after"
    return "snapshot"


def primary_cohort(values: pd.Series) -> str:
    present = set(values)
    return next((name for name in COHORT_ORDER if name in present), "other")


__all__ = ["LAYERS", "experiment_rows", "files", "plain", "tables"]
