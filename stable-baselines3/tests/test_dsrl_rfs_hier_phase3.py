from unittest.mock import Mock

import pytest
import torch as th

from stable_baselines3.dsrl.hierarchical_replay_buffer import BranchMode
from tests.three_critic_test_utils import make_model, populate_branch


def branch_sample(model, branch):
    populate_branch(model, branch, rows=2)
    return model.replay_buffer.sample_branch(branch, 2)


def test_qa_base_target_has_permanently_base_only_continuation(monkeypatch):
    model, _ = make_model()
    data = branch_sample(model, BranchMode.BASE)
    exploding = Mock(side_effect=AssertionError("residual entered QA_base target"))
    monkeypatch.setattr(model.residual_actor, "forward_pre_tanh", exploding)
    monkeypatch.setattr(model.residual_actor_target, "forward_pre_tanh", exploding)
    target_spy = Mock(wraps=model.qa_base_target.forward)
    monkeypatch.setattr(model.qa_base_target, "forward", target_spy)
    actor_spy = Mock(wraps=model.actor.action_log_prob)
    monkeypatch.setattr(model.actor, "action_log_prob", actor_spy)

    loss, target, _ = model._qa_base_loss(data)
    model._clear_all_gradients()
    loss.backward()
    assert target_spy.call_count == 1
    assert actor_spy.call_count == 1
    assert exploding.call_count == 0
    assert all(parameter.grad is None for parameter in model.actor.parameters())
    assert model.log_ent_coef.grad is None
    assert all(parameter.grad is not None for parameter in model.qa_base.parameters())
    assert th.isfinite(target).all()


def test_qa_joint_target_uses_target_residual_not_online(monkeypatch):
    model, _ = make_model()
    data = branch_sample(model, BranchMode.JOINT)
    online_exploding = Mock(side_effect=AssertionError("online residual in target"))
    monkeypatch.setattr(model.residual_actor, "forward_pre_tanh", online_exploding)
    target_spy = Mock(wraps=model.residual_actor_target.forward_pre_tanh)
    monkeypatch.setattr(model.residual_actor_target, "forward_pre_tanh", target_spy)
    joint_target_spy = Mock(wraps=model.qa_joint_target.forward)
    monkeypatch.setattr(model.qa_joint_target, "forward", joint_target_spy)

    loss, target, _ = model._qa_joint_loss(data, beta=0.1)
    loss.backward()
    assert online_exploding.call_count == 0
    assert target_spy.call_count == 1
    assert joint_target_spy.call_count == 1
    assert model.log_ent_coef.grad is None
    assert th.isfinite(target).all()


def test_qw_teacher_is_target_qa_base_head_aligned_and_current_policy(monkeypatch):
    model, _ = make_model()
    data = branch_sample(model, BranchMode.BASE)
    actor_spy = Mock(wraps=model.actor.action_log_prob)
    monkeypatch.setattr(model.actor, "action_log_prob", actor_spy)
    target_spy = Mock(
        return_value=(th.ones(2, 1), th.full((2, 1), 3.0))
    )
    monkeypatch.setattr(model.qa_base_target, "forward", target_spy)
    monkeypatch.setattr(
        model.qw_base,
        "forward",
        Mock(return_value=(th.zeros(2, 1), th.zeros(2, 1))),
    )
    forbidden = Mock(side_effect=AssertionError("joint path entered QW teacher"))
    monkeypatch.setattr(model.qa_joint, "forward", forbidden)
    monkeypatch.setattr(model.residual_actor, "forward_pre_tanh", forbidden)

    loss, student, teacher = model._qw_base_loss(data)
    assert actor_spy.call_count == 1
    assert target_spy.call_count == 1
    assert forbidden.call_count == 0
    assert loss.item() == pytest.approx(5.0)
    assert teacher[0].mean().item() == 1.0
    assert teacher[1].mean().item() == 3.0
    assert all(not tensor.requires_grad for tensor in teacher)
    assert all(tensor.mean().item() == 0 for tensor in student)


@pytest.mark.parametrize("timeout_mask", [0.0, 1.0])
def test_both_bellman_losses_share_terminal_mask_semantics(timeout_mask):
    model, _ = make_model()
    base = branch_sample(model, BranchMode.BASE)
    populate_branch(model, BranchMode.JOINT, rows=2)
    joint = model.replay_buffer.sample_branch(BranchMode.JOINT, 2)
    done = th.full_like(base.dones, timeout_mask)
    base = base._replace(dones=done, rewards=th.full_like(base.rewards, 7.0))
    joint = joint._replace(dones=done, rewards=th.full_like(joint.rewards, 7.0))
    _, base_target, _ = model._qa_base_loss(base)
    _, joint_target, _ = model._qa_joint_loss(joint, beta=0.1)
    if timeout_mask == 1.0:
        th.testing.assert_close(base_target, th.full_like(base_target, 7.0))
        th.testing.assert_close(joint_target, th.full_like(joint_target, 7.0))
    else:
        assert not th.equal(base_target, th.full_like(base_target, 7.0))
        assert not th.equal(joint_target, th.full_like(joint_target, 7.0))
