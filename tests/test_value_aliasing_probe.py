from __future__ import annotations

import numpy as np

from diagnose_per_step_value_aliasing import (
    _fixed_horizon_returns,
    _remaining_plan,
    split_episode_ids,
)


def test_remaining_plan_is_left_aligned_and_masked():
    chunk = np.arange(12, dtype=np.float32).reshape(4, 3)
    remaining, mask = _remaining_plan(chunk, phase=2)
    np.testing.assert_array_equal(remaining[:2], chunk[2:])
    np.testing.assert_array_equal(remaining[2:], np.zeros((2, 3)))
    np.testing.assert_array_equal(mask, [1.0, 1.0, 0.0, 0.0])


def test_fixed_horizon_targets_exclude_incomplete_time_limit_tail():
    rewards = [1.0, 2.0, 3.0, 4.0]
    targets, valid = _fixed_horizon_returns(
        rewards,
        gamma=0.5,
        horizon=3,
        terminal=False,
    )
    np.testing.assert_allclose(targets[:2], [2.75, 4.5])
    np.testing.assert_array_equal(valid, [True, True, False, False])


def test_fixed_horizon_targets_keep_true_terminal_tail():
    rewards = [1.0, 2.0, 3.0, 4.0]
    targets, valid = _fixed_horizon_returns(
        rewards,
        gamma=0.5,
        horizon=3,
        terminal=True,
    )
    np.testing.assert_allclose(targets, [2.75, 4.5, 5.0, 4.0])
    np.testing.assert_array_equal(valid, np.ones(4, dtype=bool))


def test_episode_split_is_disjoint_and_stratifies_early_falls():
    episodes = [
        {"episode_id": index, "early_fall": index >= 10}
        for index in range(15)
    ]
    splits = split_episode_ids(episodes, seed=7)
    identifiers = [set(splits[name]) for name in ("train", "validation", "test")]
    assert not identifiers[0].intersection(identifiers[1])
    assert not identifiers[0].intersection(identifiers[2])
    assert not identifiers[1].intersection(identifiers[2])
    assert set.union(*identifiers) == set(range(15))
    for identifiers_for_split in identifiers:
        assert any(identifier >= 10 for identifier in identifiers_for_split)
