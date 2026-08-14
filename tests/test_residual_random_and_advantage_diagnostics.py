from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from diagnose_residual_random_and_advantage import (
    _episode_seed,
    _nested_bootstrap_interval,
    _rankdata,
    _seed_frozen_planner,
    _spearman,
    state_ranking_metrics,
)
from diagnose_residual_std_and_horizon import (
    _float_list,
    _int_list,
    aggregate_multi_horizon_ranking,
)


def test_episode_seed_is_deterministic_and_separates_families():
    assert _episode_seed(30_000, 7) == _episode_seed(30_000, 7)
    assert _episode_seed(30_000, 7) != _episode_seed(30_001, 7)
    assert _episode_seed(30_000, 7) != _episode_seed(30_000, 8)


def test_diagnostic_seeds_private_planner_rng_through_wrappers():
    received = []
    planner = SimpleNamespace(seed=lambda value: received.append(value))
    environment = SimpleNamespace(env=SimpleNamespace(planner=planner))
    _seed_frozen_planner(environment, 22_007)
    assert received == [22_007]


def test_nested_bootstrap_resamples_both_axes():
    matrix = np.asarray(
        [[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]],
        dtype=np.float64,
    )
    low, high = _nested_bootstrap_interval(
        matrix,
        rng=np.random.default_rng(4),
        samples=1000,
    )
    assert low < matrix.mean() < high


def test_rank_helpers_handle_ties_and_perfect_order():
    np.testing.assert_array_equal(
        _rankdata(np.asarray([3.0, 1.0, 1.0, 2.0])),
        np.asarray([4.0, 1.5, 1.5, 3.0]),
    )
    assert _spearman(
        np.asarray([1.0, 2.0, 3.0]),
        np.asarray([10.0, 20.0, 30.0]),
    ) == 1.0


def test_state_ranking_metrics_compare_zero_and_random_candidates():
    row = {
        "candidates": [
            {
                "td_advantage": 0.0,
                "counterfactual_discounted_return": 10.0,
            },
            {
                "td_advantage": 2.0,
                "counterfactual_discounted_return": 12.0,
            },
            {
                "td_advantage": -1.0,
                "counterfactual_discounted_return": 8.0,
            },
        ]
    }
    metrics = state_ranking_metrics(row)
    assert metrics["spearman"] == 1.0
    assert metrics["pairwise_correct"] == metrics["pairwise_comparisons"] == 3
    assert (
        metrics["zero_random_correct"]
        == metrics["zero_random_comparisons"]
        == 2
    )
    assert metrics["top1_correct"]


def test_diagnostic_list_parsers_reject_ambiguous_values():
    assert _float_list("0.02,0.05") == (0.02, 0.05)
    assert _int_list("1,4,16") == (1, 4, 16)
    with pytest.raises(Exception):
        _float_list("0.05,0.05")
    with pytest.raises(Exception):
        _int_list("4,1")


def test_multi_horizon_aggregation_uses_each_prediction_separately():
    rows = []
    for phase in range(4):
        rows.append(
            {
                "phase": phase,
                "candidates": [
                    {
                        "counterfactual_discounted_return": 0.0,
                        "predictions": {
                            "1": {"predicted_advantage": 1.0},
                            "4": {"predicted_advantage": 0.0},
                        },
                    },
                    {
                        "counterfactual_discounted_return": 1.0,
                        "predictions": {
                            "1": {"predicted_advantage": 0.0},
                            "4": {"predicted_advantage": 1.0},
                        },
                    },
                ],
            }
        )
    result = aggregate_multi_horizon_ranking(
        rows,
        horizons=(1, 4),
        bootstrap_samples=100,
        bootstrap_seed=7,
    )
    assert result["1"]["overall"]["spearman_mean"] == pytest.approx(-1.0)
    assert result["4"]["overall"]["spearman_mean"] == pytest.approx(1.0)
