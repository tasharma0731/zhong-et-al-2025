from __future__ import annotations

import numpy as np

from data_access import behavior
from database import ZhongDB


def sample_behavior() -> dict[str, object]:
    trials = 2
    frames = 4
    values: dict[str, object] = {
        "Corridor_Length": 140,
        "Texture_Length": 100,
        "Gray_Space_length": 40,
        "Reward_Delay_ms": 100,
        "Reward_Mode": "fixed",
        "ntrials": trials,
        "UniqWalls": np.array(["rock", "brick"]),
        "stim_id": np.array([0, 2]),
        "trInd": np.arange(trials),
        "trInd_even": np.array([1, 0]),
        "trInd_odd": np.array([0, 1]),
        "WallName": np.array(["rock", "brick"]),
        "WallType": np.array(["texture", "texture"]),
        "WallIsProbe": np.array([0, 1]),
        "TrialStim": np.array([0, 2]),
        "isRew": np.array([1, 0]),
        "StartFr": np.array([0, 2]),
        "EndFr": np.array([1, 3]),
        "GrayFr": np.array([1, 3]),
        "Gray_space_time": np.array([0.1, 0.2]),
        "RewardFr": np.array([1, np.nan]),
        "RewPos": np.array([100, np.nan]),
        "RewTime": np.array([0.1, np.nan]),
        "SoundFr": np.array([0, 2]),
        "SoundPos": np.array([10, 20]),
        "SoundTime": np.array([0.01, 0.02]),
        "SoundDelayFr": np.array([1, 3]),
        "SoundDelPos": np.array([15, 25]),
        "SoundTimeDelay": np.array([0.02, 0.03]),
        "Trial_start_time": np.array([0.0, 0.2]),
        "Trial_end_time": np.array([0.1, 0.3]),
        "ft": np.arange(frames) / 10,
        "ft_trInd": np.array([0, 0, 1, np.nan]),
        "ft_trInd_even": np.array([1, 1, 0, 0]),
        "ft_trInd_odd": np.array([0, 0, 1, 0]),
        "ft_Pos": np.array([0, 50, 0, 50]),
        "ft_PosCum": np.array([0, 50, 100, 150]),
        "ft_RunCum": np.array([0, 1, 2, 3]),
        "ft_RunSpeed": np.array([0, 5, 6, 0]),
        "ft_WallID": np.array(["rock", "rock", "brick", "gray"]),
        "ft_isMoving": np.array([0, 1, 1, 0]),
        "ft_move": np.array([0, 1, 1, 0]),
        "ft_CorrSpc": np.array([1, 1, 1, 0]),
        "ft_GraySpc": np.array([0, 0, 0, 1]),
        "RunFr": np.array([0, 1, 1, 0]),
        "BefCueFr": np.array([1, 0, 1, 0]),
        "AftCueFr": np.array([0, 1, 0, 1]),
        "LickFr": np.array([1]),
        "LickPos": np.array([50]),
        "LickTime": np.array([0.1]),
        "LickTrind": np.array([0]),
        "Lick_wallName": np.array(["rock"]),
        "StimFrame": np.array([0, 1, 2, 3]),
        "StimTrial": np.array([0, 2]),
        "SubjMove": {"SubjMTime": np.array([0.0, 0.01])},
        "VRpos": np.array([0, 1, 2]),
        "VRposCum": np.array([0, 1, 2]),
        "VRposTime": np.array([0.0, 0.01, 0.02]),
        "run_pos": np.zeros((trials, 3)),
    }
    return values


def test_behavior_catalog_has_one_row_per_session(tmp_path) -> None:
    database = ZhongDB(cache=tmp_path, mount=False)
    sessions = database.table("behavior_sessions")

    assert database.database_path.name == "zhong.duckdb"
    assert not database.database_path.exists()
    assert len(sessions) == 142
    assert sessions["behavior_session_id"].is_unique
    for swap in ("swap1", "swap2"):
        expected = sessions["stimulus_type"].eq(swap)
        assert sessions.loc[expected, "behavior_key"].str.endswith(f"_{swap}").all()

    database.close()


def test_behavior_normalization_preserves_identity_and_grain() -> None:
    session = {
        "behavior_session_id": "experiment::mouse_date_block",
        "experiment": "experiment",
        "recording_id": "mouse_date_block",
    }
    tables = behavior.tables(session, sample_behavior())

    assert len(tables["behavior_trials"]) == 2
    assert len(tables["behavior_frames"]) == 4
    assert len(tables["behavior_events"]) == 7
    assert tables["behavior_frames"]["valid_trial"].tolist() == [True, True, True, False]
    assert tables["behavior_trials"]["stimulus_role"].tolist() == [0, 2]
    assert set(tables["behavior_events"]["event_type"]) == {
        "lick",
        "reward",
        "sound",
        "sound_delay",
    }


def test_behavior_field_surface_accounts_for_every_raw_field() -> None:
    fields = behavior.field_surface()

    assert len(fields) == 59
    assert fields["raw_field"].is_unique
    assert fields["representation"].value_counts().to_dict() == {
        "direct": 52,
        "numpy_only": 5,
        "derived": 2,
    }
