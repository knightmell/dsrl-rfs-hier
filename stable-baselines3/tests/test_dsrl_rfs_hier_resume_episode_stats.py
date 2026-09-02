"""Regression tests for episode-stat state restored through SB3 JSON data."""

from stable_baselines3.dsrl.hierarchical_replay_buffer import BranchMode
from stable_baselines3.dsrl.hierarchical_rfs_dsrl import (
    _normalize_closed_episode_stats,
)


def test_restored_episode_stat_string_keys_are_normalized_to_branch_integers():
    restored = {
        "0": {
            "return_sum": 12.0,
            "return_sq_sum": 144.0,
            "length_sum": 7.0,
            "count": 1.0,
            "early_fall_count": 0.0,
        },
        "1": {
            "return_sum": 5.0,
            "return_sq_sum": 25.0,
            "length_sum": 3.0,
            "count": 1.0,
            "early_fall_count": 1.0,
        },
    }

    normalized = _normalize_closed_episode_stats(restored)

    assert set(normalized) == {
        int(BranchMode.BASE),
        int(BranchMode.JOINT),
    }
    assert normalized[int(BranchMode.BASE)]["return_sum"] == 12.0
    assert normalized[int(BranchMode.JOINT)]["early_fall_count"] == 1.0


def test_missing_restored_branch_gets_a_fresh_zero_window():
    normalized = _normalize_closed_episode_stats({})

    for branch in (int(BranchMode.BASE), int(BranchMode.JOINT)):
        assert normalized[branch] == {
            "return_sum": 0.0,
            "return_sq_sum": 0.0,
            "length_sum": 0.0,
            "count": 0.0,
            "early_fall_count": 0.0,
        }
