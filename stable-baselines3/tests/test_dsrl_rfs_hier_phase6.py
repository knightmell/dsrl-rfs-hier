"""Co-training schedule (fresh_frozen_ddim_2p5m_cotrain) tests.

Covers the user-design co-training profile: short Phase B with QA_joint
shadow-trained on BASE transitions, long Phase R that co-trains the base
branch (QA_base->QW->noise/alpha) and the joint branch (QA_joint->residual)
simultaneously, an R-start beta hold (residual structurally OFF at beta=0),
cross-lane replay, and pre-tanh residual exploration.  The frozen profiles
share this machinery but keep their original behavior via defaults.
"""

import numpy as np
import pytest
import torch as th

from stable_baselines3.dsrl.hierarchy_schedule import (
    HierarchyPhase,
    make_hierarchy_schedule,
)
from stable_baselines3.dsrl.hierarchical_replay_buffer import BranchMode
from stable_baselines3.dsrl.hierarchical_rfs_dsrl import compose_action
from tests.three_critic_test_utils import make_model, metadata_row, populate_branch


def clone_state(module):
    return {key: value.detach().clone() for key, value in module.state_dict().items()}


def _make_cotrain_model(**overrides):
    kwargs = dict(
        schedule_profile="fresh_frozen_ddim_2p5m_cotrain",
        n_envs=1,
        phase_b_steps=4,
        phase_r_steps=8,
        phase_j_steps=0,
        phase_j_enabled=False,
        beta_ramp_steps=4,
        beta_hold_steps=4,
        beta_floor=0.02,
        qa_joint_shadow_in_b=True,
        cross_lane_ratio=0.0,
        qa_base_cross_lane=False,
        residual_exploration_std=0.0,
    )
    kwargs.update(overrides)
    return make_model(**kwargs)


def _drive_train(model, batch_start: int, num_timesteps: int) -> None:
    model.num_timesteps = num_timesteps
    model._last_action_batch_start = batch_start
    model._last_action_beta = model.hierarchy_schedule.beta_at(batch_start)
    model._collected_phase_since_train = int(0)
    model.train(20, 2)


class _RecordingChunkDecoder(th.nn.Module):
    """Identity decoder that exposes the teacher input without replacing it."""

    def __init__(self):
        super().__init__()
        self.gain = th.nn.Parameter(th.ones(()))
        self.last_noise_decoder_input = None

    def forward(self, observation, noise_decoder_input, return_numpy=False):
        del observation
        self.last_noise_decoder_input = noise_decoder_input.detach().clone()
        result = (noise_decoder_input * self.gain).clamp(-1.0, 1.0)
        return result.detach().cpu().numpy() if return_numpy else result


class _CountingChunkDecoder(th.nn.Module):
    """Decoder spy used to verify bounded multi-w teacher microbatches."""

    def __init__(self):
        super().__init__()
        self.gain = th.nn.Parameter(th.ones(()))
        self.batch_sizes = []

    def forward(self, observation, noise_decoder_input, return_numpy=False):
        assert observation.shape[0] == noise_decoder_input.shape[0]
        self.batch_sizes.append(int(noise_decoder_input.shape[0]))
        result = (noise_decoder_input * self.gain).clamp(-1.0, 1.0)
        return result.detach().cpu().numpy() if return_numpy else result


def _capture_qw_noise(model, sample):
    captured = {}

    def capture(_module, inputs):
        captured["noise_scaled"] = inputs[1].detach().clone()

    handle = model.qw_base.register_forward_pre_hook(capture)
    try:
        model._qw_base_loss(sample)
    finally:
        handle.remove()
    return captured["noise_scaled"]


# ---------------------------------------------------------------------------
# Schedule-level tests
# ---------------------------------------------------------------------------


def test_cotrain_profile_budget_beta_and_update_profiles():
    cot = make_hierarchy_schedule("fresh_frozen_ddim_2p5m_cotrain", n_envs=10)
    assert cot.phase_b_steps == 500_000
    assert cot.phase_r_steps == 2_000_000
    assert cot.total_steps == 2_500_000
    assert cot.phase_j_steps == 0
    assert cot.phase_j_enabled is False
    assert cot.beta_ramp_steps == 50_000
    assert cot.beta_hold_steps == 50_000
    assert cot.beta_floor == 0.02
    assert cot.beta_target == 0.1
    assert cot.base_lane_probability == 0.5
    # R-start hold stays at 0.
    assert cot.beta_at(cot.phase_r_start) == 0.0
    assert cot.beta_at(cot.phase_r_start + 40_000) == 0.0
    # Ramp starts at the floor and ends at the frozen target.
    assert cot.beta_at(cot.phase_r_start + 50_000) == 0.02
    assert cot.beta_at(cot.phase_r_start + 100_000) == 0.1
    assert cot.beta_at(cot.phase_r_start + 200_000) == 0.1
    # Per-profile update counts: B shadow includes QA_joint; R co-trains
    # noise/alpha alongside the residual.
    base_profile = cot.update_profile_at(cot.phase_r_start - 10)
    assert base_profile.as_dict() == {
        "qa_base": 20,
        "qa_joint": 10,
        "qw_base": 10,
        "noise_actor": 20,
        "alpha": 20,
        "residual_actor": 0,
    }
    residual_profile = cot.update_profile_at(cot.phase_r_start)
    assert residual_profile.as_dict() == {
        "qa_base": 5,
        "qa_joint": 5,
        "qw_base": 2,
        "noise_actor": 1,
        "alpha": 1,
        "residual_actor": 1,
    }
    # Phases the profile does not customize (JOINT) fall back to the frozen
    # global defaults, so update_profile_at and serialization are complete.
    assert cot.update_profiles[HierarchyPhase.JOINT].as_dict() == {
        "qa_base": 10,
        "qa_joint": 10,
        "qw_base": 5,
        "noise_actor": 1,
        "alpha": 1,
        "residual_actor": 1,
    }


def test_cotrain_schedule_rejects_invalid_hold_and_floor():
    with pytest.raises(ValueError, match=r"cannot exceed phase_r_steps"):
        make_hierarchy_schedule(
            "fresh_frozen_ddim_5m",
            n_envs=10,
            overrides={"beta_hold_steps": 3_000_000},
        )
    with pytest.raises(ValueError, match="beta_floor"):
        make_hierarchy_schedule(
            "fresh_frozen_ddim_5m",
            n_envs=10,
            overrides={"beta_floor": 0.9},
        )
    with pytest.raises(ValueError, match="beta_hold_steps must be divisible"):
        make_hierarchy_schedule(
            "fresh_frozen_ddim_5m",
            n_envs=10,
            overrides={"beta_hold_steps": 5},
        )


def test_frozen_profiles_keep_global_update_profiles_and_zero_hold():
    fresh = make_hierarchy_schedule("fresh_frozen_ddim_5m", n_envs=10)
    assert fresh.update_profiles is None
    assert fresh.beta_hold_steps == 0
    assert fresh.beta_floor == 0.0
    assert fresh.update_profile_at(fresh.phase_r_start).qa_joint == 10
    # R starts training the residual immediately (floor 0 => always on in R).
    assert fresh.beta_at(fresh.phase_r_start) == 0.0
    assert fresh.beta_at(fresh.phase_r_start + 20) == 0.1 * (20 // 10) / (50000 // 10 - 1)


def test_co_training_model_flags_default_off_for_frozen_profiles():
    model, _ = make_model()  # frozen fresh_frozen_ddim_5m, overrides for n_envs=1
    assert model.qa_joint_shadow_in_b is False
    assert model.cross_lane_ratio == 0.0
    assert model.qa_base_cross_lane is False
    assert model.residual_exploration_std == 0.0


# ---------------------------------------------------------------------------
# Shadow Phase B + boundary
# ---------------------------------------------------------------------------


def test_shadow_phase_b_trains_qa_joint_on_base_without_joint_rows():
    model, _ = _make_cotrain_model()
    populate_branch(model, BranchMode.BASE, rows=2)
    _drive_train(model, batch_start=1, num_timesteps=2)
    # B profile: QA_base 20 / QA_joint shadow 10 / QW 10 / noise 20 / alpha 20.
    assert model.qa_base_optimizer_steps == 20
    assert model.qa_joint_optimizer_steps == 10
    assert model.qw_base_optimizer_steps == 10
    assert model.noise_actor_optimizer_steps == 20
    assert model.alpha_optimizer_steps == 20
    assert model.residual_actor_optimizer_steps == 0
    # The shadow QA_joint block consumed BASE rows: JOINT-empty is not a skip.
    assert model.joint_block_skips == 0
    # No R boundary has fired yet.
    assert model._joint_phase_initialized is False
    assert model.qa_joint_generation == 0


def test_shadow_boundary_skips_clone_and_preserves_optimizer():
    model, _ = _make_cotrain_model()
    populate_branch(model, BranchMode.BASE, rows=2)
    _drive_train(model, batch_start=1, num_timesteps=2)
    joint_state = clone_state(model.qa_joint)
    optimizer_before = model.qa_joint_optimizer

    model.num_timesteps = 4  # B->R boundary
    model._ensure_phase_activation()

    assert model._joint_phase_initialized is True
    assert model.qa_joint_generation == 1
    # Shadow mode: no clone, no optimizer recreation.
    assert model.qa_joint_optimizer is optimizer_before
    for key, value in joint_state.items():
        assert th.equal(value, model.qa_joint.state_dict()[key]), (
            f"shadow QA_joint weight {key} was overwritten at the boundary"
        )


def test_non_shadow_boundary_still_clones_from_qa_base():
    # Regression: the frozen clone behavior is intact when shadow is off.
    model, _ = make_model()
    populate_branch(model, BranchMode.BASE, rows=2)
    model.num_timesteps = 4
    model._last_action_batch_start = 3
    model._collected_phase_since_train = int(0)
    model.train(20, 2)
    assert model.qa_joint_generation == 1
    for base_value, joint_value in zip(
        model.qa_base.state_dict().values(),
        model.qa_joint.state_dict().values(),
    ):
        th.testing.assert_close(base_value, joint_value, rtol=0, atol=0)


# ---------------------------------------------------------------------------
# Phase R co-training + residual beta-floor gating
# ---------------------------------------------------------------------------


def test_phase_r_co_trains_base_and_joint_with_residual_hold_gating():
    model, _ = _make_cotrain_model(cross_lane_ratio=0.25)
    populate_branch(model, BranchMode.BASE, rows=2)
    populate_branch(model, BranchMode.JOINT, rows=2)
    model.num_timesteps = 4
    model._ensure_phase_activation()

    # R-start hold: beta=0, residual structurally OFF (gradient proportional to
    # beta), base branch continues at low frequency.
    _drive_train(model, batch_start=4, num_timesteps=4)
    assert model.residual_actor_optimizer_steps == 0
    assert model.qa_base_optimizer_steps == 5
    assert model.qa_joint_optimizer_steps == 5
    assert model.qw_base_optimizer_steps == 2
    assert model.noise_actor_optimizer_steps == 1
    assert model.alpha_optimizer_steps == 1

    # Ramp: beta=0.02 at the first ramp batch >= floor, residual turns on.
    _drive_train(model, batch_start=8, num_timesteps=8)
    assert model.hierarchy_schedule.beta_at(8) == 0.02
    assert model.residual_actor_optimizer_steps == 1
    assert model.qa_base_optimizer_steps == 10
    assert model.qa_joint_optimizer_steps == 10
    assert model.qw_base_optimizer_steps == 4
    assert model.noise_actor_optimizer_steps == 2
    assert model.alpha_optimizer_steps == 2


def test_missing_joint_still_skips_joint_block_in_r_when_residual_gated_off():
    # In R hold, the QA_joint block still needs JOINT rows (it is no longer a
    # shadow), so an empty JOINT lane fires the skip counter exactly as frozen.
    model, _ = _make_cotrain_model()
    populate_branch(model, BranchMode.BASE, rows=2)
    model.num_timesteps = 4
    model._ensure_phase_activation()
    _drive_train(model, batch_start=4, num_timesteps=4)
    assert model.joint_block_skips == 1
    assert model.qa_joint_optimizer_steps == 0


# ---------------------------------------------------------------------------
# Cross-lane replay
# ---------------------------------------------------------------------------


def test_sample_mixed_branch_composition_and_fallback():
    model, _ = make_model(cross_lane_ratio=0.25)
    populate_branch(model, BranchMode.BASE, rows=4)
    populate_branch(model, BranchMode.JOINT, rows=4)
    mixed = model._sample_mixed_branch(
        BranchMode.BASE, BranchMode.JOINT, 0.25, 4
    )
    assert mixed.observations.shape[0] == 4
    base_count = int((mixed.branch_mode == int(BranchMode.BASE)).sum())
    joint_count = int((mixed.branch_mode == int(BranchMode.JOINT)).sum())
    assert base_count == 3  # 75%
    assert joint_count == 1  # 25%

    # Secondary lane too small: falls back to a pure primary batch.
    model2, _ = make_model(cross_lane_ratio=0.25)
    populate_branch(model2, BranchMode.BASE, rows=4)
    pure = model2._sample_mixed_branch(BranchMode.BASE, BranchMode.JOINT, 0.25, 4)
    assert int((pure.branch_mode == int(BranchMode.BASE)).sum()) == 4

    # ratio == 0 keeps a fresh pure-primary sample (frozen behavior).
    model3, _ = make_model()
    populate_branch(model3, BranchMode.BASE, rows=4)
    frozen = model3._sample_mixed_branch(BranchMode.BASE, BranchMode.JOINT, 0.0, 4)
    assert int((frozen.branch_mode == int(BranchMode.BASE)).sum()) == 4


def _add_joint_row_with_residual(model, row):
    # A JOINT row that looks like a real joint execution: the executed action
    # diverges from the base action (a_exec = a_base + beta*delta).
    metadata, dones, infos = metadata_row(model, BranchMode.JOINT, row=row)
    metadata["action_exec"] = metadata["action_base"] + 0.25
    metadata["residual_pre_tanh"][:] = 0.5
    metadata["residual_unit"][:] = 0.3
    metadata["action_residual_delta"][:] = 0.25
    model.replay_buffer.add_hierarchy(
        np.full((model.n_envs, 3), row / 10, np.float32),
        np.full((model.n_envs, 3), row / 10 + 0.01, np.float32),
        metadata["action_exec"],
        np.ones(model.n_envs, np.float32),
        dones,
        infos,
        metadata=metadata,
    )


def test_qa_base_update_rejects_inconsistent_executed_action_batch():
    # Core V1 data-use invariant: the action QA_base queries at must be the
    # transition's actually-executed action.  A batch whose action_exec claims
    # the *unexecuted* base action (while the transition truly executed
    # a_exec != a_base) is a Bellman-invalid tuple and must raise.
    model, _ = _make_cotrain_model(cross_lane_ratio=0.5)
    populate_branch(model, BranchMode.BASE, rows=2)
    for row in range(2):
        _add_joint_row_with_residual(model, row=1 + row)
    mixed = model._sample_mixed_branch(BranchMode.BASE, BranchMode.JOINT, 0.5, 2)
    # The 50/50 split contains a JOINT row with a genuine residual.
    assert not th.all(mixed.action_base == mixed.action_exec)
    # actions still reports the true executed action; corrupting action_exec to
    # the unexecuted base action is exactly the old (s, a_base, r, s') leak.
    corrupted = mixed._replace(action_exec=mixed.action_base)
    with pytest.raises(AssertionError, match="executed action"):
        model._update_qa_base_once(corrupted)


def test_qa_base_cross_lane_ablation_consumes_joint_rows_at_executed_action():
    # Version B ("shared transitions, separated continuation"): with
    # qa_base_cross_lane=True, QA_base may consume JOINT rows -- legal because
    # _qa_base_loss queries at the executed action (action_exec) and targets a
    # base continuation.  train() must complete without the executed-action
    # invariant firing and QA_base must get its full updates.
    model, _ = _make_cotrain_model(cross_lane_ratio=0.5, qa_base_cross_lane=True)
    populate_branch(model, BranchMode.BASE, rows=2)
    for row in range(2):
        _add_joint_row_with_residual(model, row=1 + row)
    model.num_timesteps = 4
    model._ensure_phase_activation()
    _drive_train(model, batch_start=4, num_timesteps=4)
    assert model.qa_base_optimizer_steps == 5
    assert model.qa_joint_optimizer_steps == 5
    assert model.joint_block_skips == 0


def test_cross_lane_gating_keeps_qa_base_pure_base():
    # Version A (default qa_base_cross_lane=False): even with cross-lane replay
    # enabled and JOINT rows available, the QA_base block samples pure-BASE
    # batches.  This is the strict lane; the valid cross-lane ablation above
    # opts in explicitly.
    model, _ = _make_cotrain_model(cross_lane_ratio=0.25)
    populate_branch(model, BranchMode.BASE, rows=2)
    for row in range(2):
        _add_joint_row_with_residual(model, row=1 + row)
    model.num_timesteps = 4
    model._ensure_phase_activation()
    _drive_train(model, batch_start=4, num_timesteps=4)
    # R-hold window: QA_base trained its full share on pure BASE rows, and the
    # QA_joint block consumed the legal JOINT-primary cross-lane direction.
    assert model.qa_base_optimizer_steps == 5
    assert model.qa_joint_optimizer_steps == 5
    assert model.joint_block_skips == 0


# ---------------------------------------------------------------------------
# Residual exploration
# ---------------------------------------------------------------------------


def test_residual_exploration_flows_into_executed_action_and_metadata():
    model, _ = make_model(
        residual_exploration_std=0.5,
        n_envs=1,
        phase_b_steps=4,
        phase_r_steps=8,
        beta_ramp_steps=4,
        beta_hold_steps=4,
        beta_floor=0.02,
    )
    obs = th.tensor([[0.1, 0.2, 0.3]], dtype=th.float32)
    noise_scaled = th.tensor([[0.1, -0.2, 0.3, -0.4]], dtype=th.float32)
    joint_branches = np.array([int(BranchMode.JOINT)], dtype=np.int8)

    # beta > 0 and std > 0: pre-tanh logits are perturbed, so the executed
    # action diverges from the deterministic base and two draws differ.
    out1 = model._generate_mixed_lane_behavior(
        obs, noise_scaled, joint_branches, beta=0.5
    )
    out2 = model._generate_mixed_lane_behavior(
        obs, noise_scaled, joint_branches, beta=0.5
    )
    assert not th.allclose(out1.residual_pre_tanh, th.zeros_like(out1.residual_pre_tanh))
    assert not th.allclose(out1.action_exec, out1.action_base)
    assert not th.allclose(out1.action_exec, out2.action_exec)
    # The perturbation flows into every stored residual metadata field.
    assert not th.allclose(out1.residual_unit, th.zeros_like(out1.residual_unit))
    assert not th.allclose(out1.action_residual_delta, th.zeros_like(out1.action_residual_delta))

    # beta == 0: exploration is gated off (action_exec must equal base exactly).
    out0 = model._generate_mixed_lane_behavior(
        obs, noise_scaled, joint_branches, beta=0.0
    )
    assert th.allclose(out0.action_exec, out0.action_base)
    assert th.allclose(out0.residual_pre_tanh, th.zeros_like(out0.residual_pre_tanh))

    # BASE rows stay exactly zero even with std > 0 and beta > 0.
    base_branches = np.array([int(BranchMode.BASE)], dtype=np.int8)
    out_base = model._generate_mixed_lane_behavior(
        obs, noise_scaled, base_branches, beta=0.5
    )
    assert th.allclose(out_base.residual_pre_tanh, th.zeros_like(out_base.residual_pre_tanh))
    assert th.allclose(out_base.action_exec, out_base.action_base)

    # std == 0 preserves the deterministic frozen compose (two draws equal).
    model_det, _ = make_model(residual_exploration_std=0.0, n_envs=1)
    d1 = model_det._generate_mixed_lane_behavior(
        obs, noise_scaled, joint_branches, beta=0.5
    )
    d2 = model_det._generate_mixed_lane_behavior(
        obs, noise_scaled, joint_branches, beta=0.5
    )
    assert th.allclose(d1.action_exec, d2.action_exec)
    assert th.allclose(d1.action_exec, d1.action_base)


# ---------------------------------------------------------------------------
# E4 joint-credit baseline: QW teacher switch in Phase R only
# ---------------------------------------------------------------------------

def _base_teacher(model, observations):
    with th.no_grad():
        noise, _ = model.actor.action_log_prob(observations)
        decoder = model._unscale_noise(noise)
        base = model._decode_noise_decoder_input(observations, decoder)
        return tuple(
            value.detach()
            for value in model.qa_base_target(observations, base)
        )


def _joint_teacher_at_current_composed(model, observations):
    with th.no_grad():
        noise, _ = model.actor.action_log_prob(observations)
        decoder = model._unscale_noise(noise)
        base = model._decode_noise_decoder_input(observations, decoder)
        logits = model.residual_actor.forward_pre_tanh(
            observations, noise, base
        )
        composition = compose_action(
            base,
            logits,
            model._last_action_beta,
            model._exec_action_low_tensor,
            model._exec_action_high_tensor,
            numerical_tolerance=model.numerical_bound_tolerance,
        )
        return tuple(
            value.detach()
            for value in model.qa_joint_target(
                observations, composition.action_exec
            )
        )


def test_qw_teacher_joint_credit_flag_default_off():
    model, _ = make_model()  # frozen profile: flag must default OFF
    assert model.qw_teacher_joint_credit is False


def test_qw_teacher_joint_credit_phase_b_keeps_base_teacher():
    """E4 flag ON must not change Phase B (bit-identical to VS-Hier)."""
    model, _ = _make_cotrain_model(qw_teacher_joint_credit=True)
    populate_branch(model, BranchMode.BASE, rows=2)
    model.num_timesteps = 2  # Phase B
    model._last_action_beta = model.hierarchy_schedule.beta_at(1)
    # The actor samples noise inside the loss; reseed so the expected
    # re-computation draws the identical noise.
    th.manual_seed(1234)
    np.random.seed(1234)
    sample = model.replay_buffer.sample_branch(BranchMode.BASE, 2, env=None)
    _, _, teacher = model._qw_base_loss(sample)
    th.manual_seed(1234)
    np.random.seed(1234)
    expected = _base_teacher(model, sample.observations)
    assert len(teacher) == len(expected)
    for got, want in zip(teacher, expected):
        th.testing.assert_close(got, want, rtol=0, atol=0)


def test_qw_teacher_joint_credit_phase_r_switches_to_joint_value():
    """E4 flag ON in Phase R: teacher = qa_joint_target at the CURRENT
    composed action (replay base + current residual at replay noise/base),
    fully detached, and it differs from the QA_base teacher."""
    model, _ = _make_cotrain_model(qw_teacher_joint_credit=True)
    populate_branch(model, BranchMode.BASE, rows=2)
    model.num_timesteps = 4
    model._ensure_phase_activation()
    model.num_timesteps = 12  # deep into R: beta ramp finished (0.1)
    model._last_action_beta = model.hierarchy_schedule.beta_at(12)
    # Train a few R-phase steps first: at fresh init the residual output layer
    # is zero (composed == base), which would make the joint teacher coincide
    # with the base teacher trivially.
    _drive_train(model, batch_start=12, num_timesteps=12)
    th.manual_seed(1234)
    np.random.seed(1234)
    sample = model.replay_buffer.sample_branch(BranchMode.BASE, 2, env=None)
    _, _, teacher = model._qw_base_loss(sample)
    th.manual_seed(1234)
    np.random.seed(1234)
    expected = _joint_teacher_at_current_composed(model, sample.observations)
    assert len(teacher) == len(expected)
    for got, want in zip(teacher, expected):
        th.testing.assert_close(got, want, rtol=0, atol=0)
    th.manual_seed(1234)
    np.random.seed(1234)
    base_teacher = _base_teacher(model, sample.observations)
    assert not th.allclose(teacher[0], base_teacher[0]), (
        "joint-credit teacher must differ from the QA_base teacher"
    )


def test_qw_teacher_joint_credit_off_in_r_keeps_base_teacher():
    """Flag OFF (VS-Hier) in Phase R: teacher stays QA_base_target."""
    model, _ = _make_cotrain_model()  # flag default False
    populate_branch(model, BranchMode.BASE, rows=2)
    model.num_timesteps = 4
    model._ensure_phase_activation()
    model.num_timesteps = 12
    model._last_action_beta = model.hierarchy_schedule.beta_at(12)
    th.manual_seed(1234)
    np.random.seed(1234)
    sample = model.replay_buffer.sample_branch(BranchMode.BASE, 2, env=None)
    _, _, teacher = model._qw_base_loss(sample)
    th.manual_seed(1234)
    np.random.seed(1234)
    expected = _base_teacher(model, sample.observations)
    for got, want in zip(teacher, expected):
        th.testing.assert_close(got, want, rtol=0, atol=0)


def test_qw_teacher_joint_credit_r_train_runs_with_switch():
    """End-to-end train() in Phase R with the E4 flag: QW still steps (2),
    residual still steps (1 post-ramp), no crash from the teacher compose."""
    model, _ = _make_cotrain_model(qw_teacher_joint_credit=True)
    populate_branch(model, BranchMode.BASE, rows=2)
    populate_branch(model, BranchMode.JOINT, rows=2)
    model.num_timesteps = 4
    model._ensure_phase_activation()
    # Post-ramp (beta=0.1): residual gated ON; QW teacher uses joint value.
    _drive_train(model, batch_start=12, num_timesteps=12)
    assert model.qw_base_optimizer_steps == 2
    assert model.residual_actor_optimizer_steps == 1
    assert model.qa_base_optimizer_steps == 5


# ---------------------------------------------------------------------------
# Base-diagnosis source-only intervention: Current-K1 vs Gaussian-K1
# ---------------------------------------------------------------------------


def test_qw_teacher_source_rejects_unknown_distribution():
    with pytest.raises(ValueError, match="qw_teacher_source"):
        make_model(qw_teacher_source="not-a-distribution")


def test_qw_teacher_current_k1_uses_actor_noise_for_each_replay_state():
    """The explicit Current-K1 arm must preserve the former actor-local path."""
    decoder = _RecordingChunkDecoder()
    model, _ = make_model(
        decoder=decoder,
        qw_teacher_source="current_actor",
    )
    populate_branch(model, BranchMode.BASE, rows=2)
    sample = model.replay_buffer.sample_branch(BranchMode.BASE, 2, env=None)

    th.manual_seed(31415)
    with th.no_grad():
        expected_scaled, _ = model.actor.action_log_prob(sample.observations)
        expected_decoder = model._unscale_noise(expected_scaled)

    th.manual_seed(31415)
    actual_scaled = _capture_qw_noise(model, sample)

    assert decoder.last_noise_decoder_input is not None
    assert decoder.last_noise_decoder_input.shape == (2, 2, 2)
    th.testing.assert_close(
        decoder.last_noise_decoder_input, expected_decoder, rtol=0, atol=0
    )
    th.testing.assert_close(actual_scaled, expected_scaled, rtol=0, atol=0)


def test_qw_teacher_gaussian_k1_uses_one_standard_normal_per_replay_state():
    """Gaussian-K1 must match DSRL's broad decoder-noise source exactly.

    Resetting the RNG before the real loss also catches an accidental actor
    sample before ``randn``: that extra draw would change this exact tensor.
    """
    decoder = _RecordingChunkDecoder()
    model, _ = make_model(
        decoder=decoder,
        qw_teacher_source="gaussian",
    )
    populate_branch(model, BranchMode.BASE, rows=2)
    sample = model.replay_buffer.sample_branch(BranchMode.BASE, 2, env=None)

    # Exercise the non-trivial DSRL coordinate transform: decoder Gaussian
    # lives in [-2.5, 2.5] action coordinates while QW consumes scaled noise.
    # Replace rather than mutate these tensors in place: some SB3 spaces are
    # backed by shared NumPy storage, and fill_ would leak bounds into later
    # tests that reuse TinyChunkEnv's class-level Box.
    model._noise_action_low_tensor = th.full_like(
        model._noise_action_low_tensor, -2.5
    )
    model._noise_action_high_tensor = th.full_like(
        model._noise_action_high_tensor, 2.5
    )
    th.manual_seed(27182)
    expected_decoder = th.randn(2, 2, 2)
    expected_scaled = expected_decoder.reshape(2, 4) / 2.5

    th.manual_seed(27182)
    actual_scaled = _capture_qw_noise(model, sample)

    assert decoder.last_noise_decoder_input is not None
    # K=1 means no state expansion: B replay states produce exactly B pairs.
    assert decoder.last_noise_decoder_input.shape[0] == sample.observations.shape[0]
    th.testing.assert_close(
        decoder.last_noise_decoder_input, expected_decoder, rtol=0, atol=0
    )
    th.testing.assert_close(actual_scaled, expected_scaled, rtol=0, atol=1e-7)


def test_qw_multiw_contract_rejects_invalid_or_actor_local_settings():
    with pytest.raises(ValueError, match="qw_candidates_per_state"):
        make_model(qw_candidates_per_state=0)
    with pytest.raises(ValueError, match="qw_state_batch_size"):
        make_model(qw_state_batch_size=True)
    with pytest.raises(ValueError, match="qw_teacher_microbatch_size"):
        make_model(qw_teacher_microbatch_size=0)
    with pytest.raises(ValueError, match="requires.*gaussian"):
        make_model(
            qw_teacher_source="current_actor",
            qw_candidates_per_state=4,
        )


def test_qw_gaussian_multiw_expands_each_state_and_microbatches_teacher():
    decoder = _CountingChunkDecoder()
    model, _ = make_model(
        decoder=decoder,
        qw_teacher_source="gaussian",
        qw_candidates_per_state=4,
        qw_state_batch_size=2,
        qw_teacher_microbatch_size=3,
    )
    populate_branch(model, BranchMode.BASE, rows=2)
    sample = model.replay_buffer.sample_branch(BranchMode.BASE, 2, env=None)

    before_steps = model.qw_base_optimizer_steps
    loss, mse = model._update_qw_once(sample)

    assert model.qw_teacher_queries_per_update == 8
    assert decoder.batch_sizes == [3, 3, 2]
    assert model.qw_base_optimizer_steps == before_steps + 1
    assert np.isfinite(loss)
    assert np.isfinite(mse)


def test_qw_k1_defaults_preserve_global_batch_contract():
    model, _ = make_model(qw_teacher_source="gaussian")
    assert model.qw_candidates_per_state == 1
    assert model.qw_state_batch_size == model.batch_size
    assert model.qw_teacher_microbatch_size == model.batch_size
    assert model.qw_teacher_queries_per_update == model.batch_size


def test_qw_multiw_microbatching_matches_unchunked_optimizer_step():
    common = dict(
        qw_teacher_source="gaussian",
        qw_candidates_per_state=4,
        qw_state_batch_size=2,
    )
    unchunked, _ = make_model(qw_teacher_microbatch_size=8, **common)
    chunked, _ = make_model(qw_teacher_microbatch_size=3, **common)
    chunked.qw_base.load_state_dict(unchunked.qw_base.state_dict())
    chunked.qa_base_target.load_state_dict(unchunked.qa_base_target.state_dict())
    chunked.qw_base_optimizer.load_state_dict(
        unchunked.qw_base_optimizer.state_dict()
    )
    populate_branch(unchunked, BranchMode.BASE, rows=2)
    sample = unchunked.replay_buffer.sample_branch(BranchMode.BASE, 2, env=None)

    th.manual_seed(4242)
    full_loss, full_mse = unchunked._update_qw_once(sample)
    th.manual_seed(4242)
    split_loss, split_mse = chunked._update_qw_once(sample)

    assert split_loss == pytest.approx(full_loss, rel=1e-6, abs=1e-6)
    assert split_mse == pytest.approx(full_mse, rel=1e-6, abs=1e-6)
    for full_parameter, split_parameter in zip(
        unchunked.qw_base.parameters(), chunked.qw_base.parameters()
    ):
        th.testing.assert_close(
            split_parameter,
            full_parameter,
            rtol=1e-6,
            atol=1e-7,
        )
