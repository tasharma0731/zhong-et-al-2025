from __future__ import annotations

import numpy as np
import pandas as pd

from modeling.transfer_pipeline import (
    METRICS,
    exact_sign_flip,
    make_session_pairs,
    make_window_transitions,
    nonoverlapping_windows,
)


def _history() -> pd.DataFrame:
    rows = []
    for start in range(5):
        row = {
            "behavior_session_id": "session",
            "mouse": "mouse",
            "cohort": "unsupervised",
            "moment": "before",
            "area": "mHV",
            "start_pair_id": start,
            "stop_pair_id": start + 1,
            "pair_count": 2,
            "first_trial_id": start * 10,
            "last_trial_id": start * 10 + 9,
            "progress": start / 5,
            "mean_run_speed": 1.0 + start,
        }
        row.update({metric: float(start) for metric in METRICS})
        rows.append(row)
    return pd.DataFrame(rows)


def test_nonoverlapping_windows_and_transitions() -> None:
    selected = nonoverlapping_windows(_history(), 2)
    assert selected["start_pair_id"].tolist() == [0, 2, 4]
    transitions = make_window_transitions(selected, 2)
    assert transitions[["current_start_pair_id", "next_start_pair_id"]].values.tolist() == [
        [0, 2],
        [2, 4],
    ]


def test_session_pairs_use_prediction_prefixes() -> None:
    before = _history().iloc[[0]].copy()
    after = before.assign(
        behavior_session_id="after",
        moment="after",
        **{metric: before[metric] + 1.0 for metric in METRICS},
    )
    pairs = make_session_pairs(pd.concat([before, after], ignore_index=True))
    assert pairs["before_median"].iloc[0] == 0.0
    assert pairs["after_median"].iloc[0] == 1.0


def test_exact_sign_flip_four_consistent_mice() -> None:
    effect, pvalue, permutations = exact_sign_flip([1.0, 2.0, 3.0, 4.0])
    assert effect == 2.5
    assert pvalue == 0.125
    assert permutations == 16


def test_exact_sign_flip_ignores_nonfinite_values() -> None:
    effect, pvalue, permutations = exact_sign_flip([1.0, np.nan])
    assert effect == 1.0
    assert pvalue == 1.0
    assert permutations == 2
