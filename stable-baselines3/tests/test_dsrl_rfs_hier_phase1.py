from unittest.mock import Mock

import numpy as np
import pytest
import torch as th

from stable_baselines3.dsrl.hierarchical_replay_buffer import BranchMode
from stable_baselines3.dsrl.hierarchical_rfs_dsrl import (
    ARCHITECTURE_VERSION,
    ResidualActor,
    compose_action,
)
from tests.three_critic_test_utils import make_model


def test_centered_composition_zero_parity_and_derivative():
    base = th.tensor([[-1.0, -0.3, 0.4, 1.0]], requires_grad=False)
    logits = th.zeros_like(base, requires_grad=True)
    result = compose_action(
        base, logits, 0.1, th.full((4,), -1.0), th.full((4,), 1.0)
    )
    th.testing.assert_close(result.action_residual_delta, th.zeros_like(base))
    th.testing.assert_close(result.action_exec, base, rtol=0, atol=0)
    result.action_residual_delta.sum().backward()
    th.testing.assert_close(logits.grad, th.full_like(base, 0.1), rtol=0, atol=1e-7)


def test_composition_training_gradient_full_at_bound_tie():
    # The residual loss trains through `action_exec` (the clamped value the
    # critics evaluate).  At a bound-touching base with zero logits the spec
    # requires the full centered subgradient beta*(high-low)/2; the emergency
    # clamp must never halve it (it is value-only roundoff protection).
    low = th.full((2,), -1.0)
    high = th.full((2,), 1.0)
    beta = 0.7
    expected = beta * (high - low)[0] / 2.0
    for base_value in (-1.0, 1.0):
        base = th.full((1, 2), base_value)
        logits = th.zeros(1, 2, requires_grad=True)
        result = compose_action(base, logits, beta, low, high)
        result.action_exec.sum().backward()
        th.testing.assert_close(
            logits.grad, th.full_like(base, expected), rtol=1e-5, atol=1e-5
        )
        # Zero logits still reproduce the base exactly (value contract).
        th.testing.assert_close(result.action_exec, base, rtol=0, atol=0)
        assert not bool(result.emergency_clamp_applied.any())


def test_composition_uses_directional_margin_and_stays_bounded():
    base = th.tensor([[0.8, -0.8]], dtype=th.float32)
    logits = th.tensor([[2.0, -2.0]], dtype=th.float32)
    result = compose_action(base, logits, 1.0, th.tensor([-1.0, -1.0]), th.tensor([1.0, 1.0]))
    expected = th.tensor([[0.8, -0.8]]) + th.tensor([[0.2, -0.2]]) * th.tanh(
        th.tensor([[2.0, 2.0]])
    )
    th.testing.assert_close(result.action_exec, expected)
    assert bool((result.action_exec <= 1).all() and (result.action_exec >= -1).all())
    assert not bool(result.emergency_clamp_applied.any())


@pytest.mark.parametrize("beta", [-0.1, 1.1, float("nan")])
def test_composition_rejects_invalid_beta(beta):
    with pytest.raises(ValueError, match="beta"):
        compose_action(th.zeros(1, 2), th.zeros(1, 2), beta, -th.ones(2), th.ones(2))


def test_composition_rejects_material_base_oob():
    with pytest.raises(ValueError, match="materially outside"):
        compose_action(
            th.tensor([[1.01, 0.0]]),
            th.zeros(1, 2),
            0.1,
            -th.ones(2),
            th.ones(2),
        )


def test_residual_actor_structure_and_exact_zero_output():
    actor = ResidualActor(3, 4, (128, 128), "silu")
    assert [type(module) for module in actor.hidden_net] == [
        th.nn.Linear,
        th.nn.LayerNorm,
        th.nn.SiLU,
        th.nn.Linear,
        th.nn.LayerNorm,
        th.nn.SiLU,
    ]
    logits, unit = actor.forward_with_pre_tanh(
        th.randn(7, 3), th.randn(7, 4), th.randn(7, 4)
    )
    th.testing.assert_close(logits, th.zeros_like(logits), rtol=0, atol=0)
    th.testing.assert_close(unit, th.zeros_like(unit), rtol=0, atol=0)


def test_construction_has_three_disjoint_value_meanings_and_frozen_targets():
    model, decoder = make_model()
    assert model.architecture_version == ARCHITECTURE_VERSION
    assert model.qa_base is model.critic
    assert model.qa_base_target is model.critic_target
    assert model.qw_base is model.critic_noise
    assert not hasattr(model, "critic_modulation")
    model._assert_parameter_ownership()
    assert not (model._storage_pointers(model.qa_base) & model._storage_pointers(model.qa_joint))
    for target in (
        model.qa_base_target,
        model.qa_joint_target,
        model.residual_actor_target,
        model.reference_noise_actor,
    ):
        assert target.training is False
        assert all(not parameter.requires_grad for parameter in target.parameters())
    assert decoder.training is False
    assert all(not parameter.requires_grad for parameter in decoder.parameters())


def test_share_features_and_non_min_backup_fail_fast():
    with pytest.raises(ValueError, match="share_features_extractor"):
        make_model(policy_kwargs={"net_arch": [8, 8], "share_features_extractor": True})
    with pytest.raises(ValueError, match="critic_backup_combine_type"):
        make_model(critic_backup_combine_type="mean")


def test_base_lane_behavior_structurally_never_calls_residual(monkeypatch):
    model, _ = make_model()
    exploding = Mock(side_effect=AssertionError("residual called in BASE"))
    monkeypatch.setattr(model.residual_actor, "forward_pre_tanh", exploding)
    generated = model._generate_mixed_lane_behavior(
        th.zeros(2, 3),
        th.zeros(2, 4),
        np.full(2, int(BranchMode.BASE), dtype=np.uint8),
        beta=0.1,
    )
    assert exploding.call_count == 0
    th.testing.assert_close(generated.action_exec, generated.action_base, rtol=0, atol=0)
    th.testing.assert_close(generated.residual_unit, th.zeros_like(generated.residual_unit), rtol=0, atol=0)


def test_warmup_joint_lane_stores_exact_zero_residual_without_calling_actor(monkeypatch):
    model, _ = make_model(n_envs=2, learning_starts=100)
    model.num_timesteps = model.hierarchy_schedule.phase_r_start
    model._last_obs = np.zeros((2, 3), dtype=np.float32)
    model._active_branch_mode[:] = int(BranchMode.JOINT)
    model._active_episode_id[:] = np.array([10, 11], dtype=np.int64)
    exploding = Mock(side_effect=AssertionError("residual called during warmup"))
    monkeypatch.setattr(model.residual_actor, "forward_pre_tanh", exploding)
    action, _ = model._sample_action(learning_starts=100, n_envs=2)
    metadata = model._pending_rollout_metadata
    assert exploding.call_count == 0
    assert metadata is not None
    assert not metadata["residual_applied"].any()
    assert np.all(metadata["residual_policy_version"] == -1)
    for name in (
        "residual_pre_tanh",
        "residual_unit",
        "action_residual_delta",
    ):
        np.testing.assert_array_equal(metadata[name], np.zeros_like(metadata[name]))
    np.testing.assert_array_equal(action, metadata["action_base"])


def test_torch_noise_unscale_matches_policy_affine_and_detaches():
    model, _ = make_model()
    model._noise_action_low_tensor = th.tensor([-2.0, -1.0, 3.0, 10.0])
    model._noise_action_high_tensor = th.tensor([2.0, 5.0, 7.0, 14.0])
    noise = th.tensor([[-1.0, -0.5, 0.25, 1.0]], requires_grad=True)
    actual = model._unscale_noise(noise).reshape(1, 4)
    expected = th.tensor([[-2.0, 0.5, 5.5, 14.0]])
    th.testing.assert_close(actual, expected)
    assert actual.requires_grad is False


def test_prediction_modes_preserve_zero_and_reference_semantics(monkeypatch):
    model, _ = make_model()
    with th.no_grad():
        model.residual_actor.output_layer.bias.fill_(0.5)
        for parameter in model.reference_noise_actor.parameters():
            parameter.zero_()
    residual_spy = Mock(wraps=model.residual_actor.forward_pre_tanh)
    monkeypatch.setattr(model.residual_actor, "forward_pre_tanh", residual_spy)
    observation = np.zeros(3, np.float32)
    full, _ = model.predict_with_components(observation, deterministic=True)
    assert residual_spy.call_count == 1
    base, _ = model.predict_with_components(
        observation, deterministic=True, mode="current_base_only"
    )
    reference, _ = model.predict_with_components(
        observation, deterministic=True, mode="reference_base"
    )
    assert residual_spy.call_count == 1
    np.testing.assert_array_equal(base["action_exec"], base["action_base"])
    np.testing.assert_array_equal(reference["action_exec"], reference["action_base"])
    assert full["action_exec"].shape == (4,)
