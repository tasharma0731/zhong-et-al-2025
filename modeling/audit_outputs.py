#!/usr/bin/env python3
"""Audit final artifacts for chronology, constraints, and provenance alignment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit(repository: Path, destination: Path) -> dict[str, object]:
    result_root = repository / "modeling" / "results" / "primary_w20"
    configuration = json.loads((result_root / "configuration.json").read_text())
    history_path = Path(configuration["history"])
    session_path = Path(configuration["session_summary"])
    if _sha256(history_path) != configuration["history_sha256"]:
        raise ValueError("primary history checksum does not match configuration")
    if _sha256(session_path) != configuration["session_summary_sha256"]:
        raise ValueError("session-summary checksum does not match configuration")

    transitions = pd.read_csv(result_root / "window_transitions.csv")
    chronology_violations = int(
        (
            transitions["current_last_trial_id"]
            >= transitions["next_first_trial_id"]
        ).sum()
    )
    if chronology_violations:
        raise ValueError(f"found {chronology_violations} chronology violations")

    predictions = pd.read_csv(result_root / "predictions.csv")
    wide = predictions.pivot_table(
        index=["task", "split", "model", "area", "mouse", "prediction_id"],
        columns="metric",
        values="predicted",
    )
    quantile_violations = int(
        ((wide["q05"] > wide["median"]) | (wide["median"] > wide["q95"])).sum()
    )
    sd_violations = int((wide["sd_dprime"] <= 0).sum())
    fraction_violations = int(
        (
            (wide["frac_leaf_selective"] < 0)
            | (wide["frac_circle_selective"] < 0)
            | (
                wide["frac_leaf_selective"]
                + wide["frac_circle_selective"]
                > 1 + 1e-10
            )
        ).sum()
    )
    if quantile_violations + sd_violations + fraction_violations:
        raise ValueError("model outputs violate distribution constraints")

    selection = json.loads(
        (
            repository
            / "modeling"
            / "results"
            / "window_selection"
            / "selection.json"
        ).read_text()
    )
    if selection["supervised_rows_read"] is not False:
        raise ValueError("window selection was not unsupervised-only")
    if int(selection["final_window_pairs"]) != int(configuration["window_pairs"]):
        raise ValueError("primary run does not use the selected window")

    cache_path = (
        repository / "modeling" / "cluster_cache" / "neuron_transitions_w20.npz"
    )
    with np.load(cache_path, allow_pickle=False) as cache:
        cluster_history_sha = str(cache["source_history_sha256"].item())
        cache_chronology_violations = int(
            (cache["current_last_trial_id"] >= cache["next_first_trial_id"]).sum()
        )
        cache_shape = list(cache["current_dprime"].shape)
    if cluster_history_sha != configuration["history_sha256"]:
        raise ValueError("cluster cache was built from a different history")
    if cache_chronology_violations:
        raise ValueError("cluster cache contains chronology violations")

    selections = pd.read_csv(
        repository
        / "modeling"
        / "results"
        / "soft_clusters_w20"
        / "soft_cluster_selections.csv"
    )
    transfer = pd.read_csv(
        repository
        / "modeling"
        / "results"
        / "soft_clusters_w20"
        / "soft_cluster_transfer_scores.csv"
    )
    learned = transfer.loc[transfer["model"].eq("final_selected_soft_cluster")]
    observed = learned[["area", "clusters", "alpha"]].drop_duplicates()
    expected = selections[["area", "clusters", "alpha"]].drop_duplicates()
    if not observed.sort_values("area").reset_index(drop=True).equals(
        expected.sort_values("area").reset_index(drop=True)
    ):
        raise ValueError("cluster transfer contains a non-selected configuration")

    payload: dict[str, object] = {
        "status": "pass",
        "primary_history_sha256": configuration["history_sha256"],
        "session_summary_sha256": configuration["session_summary_sha256"],
        "selected_window_pairs": int(configuration["window_pairs"]),
        "chronology_violations": chronology_violations,
        "prediction_quantile_violations": quantile_violations,
        "prediction_sd_violations": sd_violations,
        "prediction_fraction_violations": fraction_violations,
        "primary_prediction_rows": int(len(predictions)),
        "primary_mouse_score_rows": int(
            len(pd.read_csv(result_root / "mouse_scores.csv"))
        ),
        "cluster_cache_shape": cache_shape,
        "cluster_cache_chronology_violations": cache_chronology_violations,
        "cluster_transfer_configurations": int(len(observed)),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output", type=Path, default=Path("modeling/runs/AUDIT.json")
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    print(json.dumps(audit(arguments.repository.resolve(), arguments.output), indent=2))


if __name__ == "__main__":
    main()
