import pandas as pd
import pytest

from dprime import equal_pair_windows, exact_paired_sign_flip


def test_equal_pair_bins_are_complete_nonoverlapping_and_balanced():
    pairs = pd.DataFrame(
        {
            "pair_id": range(13),
            "first_trial_id": range(0, 26, 2),
            "last_trial_id": range(1, 27, 2),
            "trial_midpoint": [value + 0.5 for value in range(0, 26, 2)],
        }
    )
    bins = equal_pair_windows(pairs, bins=5)
    assert bins["pair_count"].tolist() == [3, 3, 3, 2, 2]
    assert bins["start_pair_id"].tolist() == [0, 3, 6, 9, 11]
    assert bins["stop_pair_id"].tolist() == [2, 5, 8, 10, 12]
    assert bins["pair_count"].sum() == len(pairs)


def test_equal_pair_bins_reject_too_many_bins():
    with pytest.raises(ValueError):
        equal_pair_windows(pd.DataFrame({"pair_id": [0]}), bins=2)


def test_exact_paired_sign_flip_uses_every_sign_assignment():
    result = exact_paired_sign_flip([1.0, 2.0, 3.0])
    assert result["difference"] == 2.0
    assert result["permutations"] == 8
    assert result["n_pairs"] == 3
    assert result["pvalue"] == 0.25
