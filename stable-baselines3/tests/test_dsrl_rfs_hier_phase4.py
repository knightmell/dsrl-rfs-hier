from unittest.mock import Mock

import numpy as np
import torch as th

from stable_baselines3.dsrl.hierarchical_replay_buffer import BranchMode
from stable_baselines3.dsrl.hierarchical_rfs_dsrl import _freeze_module_parameters
from tests.three_critic_test_utils import make_model, populate_branch


def clone_state(module):
    return {key: value.detach().clone() for key, value in module.state_dict().items()}


def changed(before, module):
    return any(not th.equal(before[key], value) for key, value in module.state_dict().items())


def test_noise_actor_graph_structurally_excludes_joint_and_ddim(monkeypatch):
    model, decoder = make_model()
    populate_branch(model, BranchMode.BASE)
    data = model.replay_buffer.sample_branch(BranchMode.BASE, 2)
    forbidden = Mock(side_effect=AssertionError("forbidden noise-loss call"))
    monkeypatch.setattr(model.residual_actor, "forward_pre_tanh", forbidden)
    monkeypatch.setattr(model.qa_joint, "forward", forbidden)
    monkeypatch.setattr(model.qa_base, "forward", forbidden)
    monkeypatch.setattr(decoder, "forward", forbidden)
    qw_spy = Mock(wraps=model.qw_base.forward)
    monkeypatch.setattr(model.qw_base, "forward", qw_spy)
    before = clone_state(model.actor)
    loss, _, _ = model._noise_actor_loss(data)
    model.actor.optimizer.zero_grad(set_to_none=True)
    loss.backward()
    model.actor.optimizer.step()
    assert changed(before, model.actor)
    assert qw_spy.call_count == 1
    assert forbidden.call_count == 0
    assert model.log_ent_coef.grad is None


def test_residual_actor_uses_only_frozen_current_qa_joint_action_gradient():
    model, decoder = make_model()
    populate_branch(model, BranchMode.JOINT)
    data = model.replay_buffer.sample_branch(BranchMode.JOINT, 2)
    actor_before = clone_state(model.actor)
    qa_before = clone_state(model.qa_joint)
    ddim_before = clone_state(decoder)
    residual_before = clone_state(model.residual_actor)
    loss, generated = model._residual_actor_loss(data, beta=0.1)
    action_gradient = th.autograd.grad(
        loss, generated.action_exec, retain_graph=True, allow_unused=False
    )[0]
    assert th.isfinite(action_gradient).all()
    model.residual_actor_optimizer.zero_grad(set_to_none=True)
    with _freeze_module_parameters(model.qa_joint):
        loss, _ = model._residual_actor_loss(data, beta=0.1)
        loss.backward()
    model.residual_actor_optimizer.step()
    assert changed(residual_before, model.residual_actor)
    assert not changed(actor_before, model.actor)
    assert not changed(qa_before, model.qa_joint)
    assert not changed(ddim_before, decoder)


def test_alpha_is_owned_only_by_alpha_loss():
    model, _ = make_model()
    populate_branch(model, BranchMode.BASE)
    data = model.replay_buffer.sample_branch(BranchMode.BASE, 2)
    model.log_ent_coef.grad = None
    base_loss, _, _ = model._qa_base_loss(data)
    base_loss.backward()
    assert model.log_ent_coef.grad is None
    model._clear_all_gradients()
    noise_loss, _, _ = model._noise_actor_loss(data)
    noise_loss.backward()
    assert model.log_ent_coef.grad is None
    before = model.log_ent_coef.detach().clone()
    model._clear_all_gradients()
    _, alpha_loss, _, _ = model._update_alpha_and_noise_once(
        data, update_alpha=True
    )
    assert alpha_loss is not None
    assert not th.equal(before, model.log_ent_coef.detach())
    assert model.log_ent_coef.grad is None


def test_phase_b_then_r_update_counts_and_boundary_clone():
    model, _ = make_model()
    populate_branch(model, BranchMode.BASE, rows=2)
    model.num_timesteps = 4
    model._last_action_batch_start = 3
    model._last_action_beta = 0.0
    model._collected_phase_since_train = int(0)
    model.train(20, 2)
    assert model.qa_base_optimizer_steps == 20
    assert model.qw_base_optimizer_steps == 10
    assert model.noise_actor_optimizer_steps == 20
    assert model.alpha_optimizer_steps == 20
    assert model.qa_joint_optimizer_steps == 0
    assert model.residual_actor_optimizer_steps == 0
    assert model.qa_joint_generation == 1
    assert len(model.qa_joint_optimizer.state) == 0
    for base_value, joint_value in zip(
        model.qa_base.state_dict().values(), model.qa_joint.state_dict().values()
    ):
        th.testing.assert_close(base_value, joint_value, rtol=0, atol=0)

    populate_branch(model, BranchMode.JOINT, rows=2)
    model.num_timesteps = 5
    model._last_action_batch_start = 4
    model._last_action_beta = 0.0
    model._collected_phase_since_train = int(1)
    model.train(20, 2)
    assert model.qa_base_optimizer_steps == 30
    assert model.qa_joint_optimizer_steps == 10
    assert model.qw_base_optimizer_steps == 15
    assert model.noise_actor_optimizer_steps == 20
    assert model.alpha_optimizer_steps == 20
    assert model.residual_actor_optimizer_steps == 1
    assert model.qa_base_target_updates == 30
    assert model.qa_joint_target_updates == 10
    assert model.residual_target_updates == 1


def test_episode_lane_is_stable_across_phase_boundary_until_reset():
    model, _ = make_model(n_envs=2)
    model.num_timesteps = 0
    model._allocate_unassigned_lanes()
    np.testing.assert_array_equal(
        model._active_branch_mode,
        np.full(2, int(BranchMode.BASE), dtype=np.int8),
    )
    old_episode = model._active_episode_id.copy()
    model.num_timesteps = model.hierarchy_schedule.phase_r_start
    model._ensure_phase_activation()
    model._allocate_unassigned_lanes()
    np.testing.assert_array_equal(model._active_episode_id, old_episode)
    np.testing.assert_array_equal(
        model._active_branch_mode,
        np.full(2, int(BranchMode.BASE), dtype=np.int8),
    )
    model._active_branch_mode[0] = -1
    model._active_episode_id[0] = -1
    model._allocate_unassigned_lanes()
    assert model._active_episode_id[0] > old_episode.max()
    assert model._active_episode_id[1] == old_episode[1]


def test_missing_branch_skips_without_incrementing_optimizer_counter():
    model, _ = make_model()
    populate_branch(model, BranchMode.BASE)
    model.num_timesteps = 5
    model._last_action_batch_start = 4
    model._collected_phase_since_train = int(1)
    model.train(20, 2)
    assert model.joint_block_skips == 1
    assert model.qa_joint_optimizer_steps == 0
    assert model.residual_actor_optimizer_steps == 0
    assert model.qa_base_optimizer_steps == 10
