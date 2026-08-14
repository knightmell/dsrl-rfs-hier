from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

from diagnose_qa_joint_delta_correlation import (
    bootstrap_ci95,
    compute_correlation_report,
    pairwise_preference_accuracy,
    probe_candidate,
    spearman_correlation,
    top1_agreement,
)


def _load_three_critic_test_utils():
    """Import the SB3 test model builder without colliding with the outer
    ``tests`` package name."""
    path = (
        Path(__file__).resolve().parents[1]
        / "stable-baselines3"
        / "tests"
        / "three_critic_test_utils.py"
    )
    spec = importlib.util.spec_from_file_location(
        "three_critic_test_utils", str(path)
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_spearman_perfect_positive_and_negative():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert spearman_correlation(x, 2.0 * x + 1.0) == pytest.approx(1.0)
    assert spearman_correlation(x, -2.0 * x + 1.0) == pytest.approx(-1.0)


def test_spearman_recovers_monotone_rank_only():
    x = np.array([10.0, 1.0, 5.0, 7.0, 3.0])
    y = np.array([1.0, 100.0, 50.0, 20.0, 80.0])  # anti-monotone in value, monotone in rank
    # ranks: x -> (1,0,2,3,4)... check with a known rank-reversed partner
    # y has the same rank order as x (larger y = smaller x).
    assert spearman_correlation(x, -y) == pytest.approx(1.0)


def test_spearman_degenerate_returns_nan():
    x = np.ones(5)
    y = np.arange(5.0)
    assert math.isnan(spearman_correlation(x, y))
    assert math.isnan(spearman_correlation(np.array([1.0, 2.0]), np.array([1.0, 2.0])))


def test_pairwise_preference_accuracy():
    predicted = np.array([1.0, 2.0, 3.0])
    realized = np.array([1.0, 3.0, 5.0])
    # all 3 pairs agree in sign -> 1.0
    assert pairwise_preference_accuracy(predicted, realized) == pytest.approx(1.0)
    reversed_pred = -predicted
    assert pairwise_preference_accuracy(reversed_pred, realized) == pytest.approx(0.0)


def test_top1_agreement():
    predicted = np.array([0.5, 1.0, 2.0])
    realized = np.array([1.0, 2.0, 3.0])
    assert top1_agreement(predicted, realized) == 1.0
    assert top1_agreement(predicted, np.array([3.0, 2.0, 1.0])) == 0.0


def test_compute_correlation_report_shape():
    report = compute_correlation_report([0.1, 0.2, 0.3], [1.0, 2.0, 3.0])
    assert report["count"] == 3
    assert report["spearman"] == pytest.approx(1.0)
    assert report["pairwise_preference_accuracy"] == pytest.approx(1.0)
    assert report["top1_agreement"] == 1.0


def _make_per_state():
    # Monotone predicted vs realized relation, so both spearman and pairwise
    # accuracy are 1.0 on any resample that draws both points of a state.
    return [
        {
            "realized": [1.0, -1.0],
            "predicted": [1.0, -1.0],
        },
        {
            "realized": [0.5, -0.5],
            "predicted": [0.4, -0.4],
        },
        {
            "realized": [2.0, -2.0],
            "predicted": [1.8, -1.8],
        },
        {
            "realized": [0.2, -0.2],
            "predicted": [0.1, -0.1],
        },
    ]


def test_bootstrap_ci95_perfect_monotone():
    result = bootstrap_ci95(_make_per_state(), samples=200, seed=1)
    assert result["samples"] == 200
    # Every resample pools perfect monotone pairs -> CI collapses to 1.0.
    assert result["spearman_ci95"] == [pytest.approx(1.0), pytest.approx(1.0)]
    assert result["pairwise_ci95"] == [pytest.approx(1.0), pytest.approx(1.0)]


def test_bootstrap_ci95_requires_samples():
    with pytest.raises(ValueError):
        bootstrap_ci95(_make_per_state(), samples=10, seed=1)


def test_bootstrap_ci95_insufficient_states_returns_nan():
    result = bootstrap_ci95(
        [{"realized": [1.0], "predicted": [1.0]}],
        samples=200,
        seed=1,
    )
    assert math.isnan(result["spearman_ci95"][0])
    assert math.isnan(result["pairwise_ci95"][0])


def test_bootstrap_ci95_reproducible():
    first = bootstrap_ci95(_make_per_state(), samples=200, seed=7)
    second = bootstrap_ci95(_make_per_state(), samples=200, seed=7)
    assert first == second


def test_probe_candidate_shapes_and_zero_probe():
    test_utils = _load_three_critic_test_utils()
    model, _ = test_utils.make_model(n_envs=1)
    observation = np.zeros(3, dtype=np.float32)
    base_noise = np.full(4, 0.1, dtype=np.float32)
    direction = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    probe = probe_candidate(
        model,
        observation,
        base_noise,
        direction,
        eps=0.05,
        beta=0.1,
    )
    zero = probe_candidate(
        model,
        observation,
        base_noise,
        np.zeros(4, dtype=np.float32),
        eps=0.05,
        beta=0.1,
    )
    assert probe["action_exec"].shape == (4,)
    assert probe["action_exec"].dtype == np.float32
    assert np.all(np.abs(probe["action_exec"]) <= 1.0 + 1e-6)
    # The zero probe must reproduce the base action and a zero delta.
    assert zero["delta_conservative"] == pytest.approx(0.0)
    assert zero["delta_head0"] == pytest.approx(0.0)
    assert zero["delta_head1"] == pytest.approx(0.0)
    # IdentityChunkDecoder(gain=1) decodes a constant 0.1 noise to 0.1.
    assert zero["action_exec"] == pytest.approx(np.full(4, 0.1, dtype=np.float32))
    # A nonzero probe must perturb the executed action along the direction.
    assert probe["action_exec"][0] != pytest.approx(zero["action_exec"][0])
    assert probe["action_exec"][1:] == pytest.approx(zero["action_exec"][1:])
    # Conservative delta is the min over the two heads.
    assert probe["delta_conservative"] == pytest.approx(
        min(probe["delta_head0"], probe["delta_head1"])
    )
