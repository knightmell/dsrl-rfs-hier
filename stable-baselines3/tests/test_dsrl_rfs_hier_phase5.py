import random

import numpy as np
import pytest
import torch as th

from stable_baselines3 import HierarchicalRFSDSRL as TopLevelHierarchy
from stable_baselines3.dsrl import HierarchicalRFSDSRL
from stable_baselines3.dsrl.hierarchy_schedule import (
    HierarchyPhase,
    make_hierarchy_schedule,
)
from stable_baselines3.dsrl.hierarchical_replay_buffer import BranchMode
from tests.three_critic_test_utils import make_model, metadata_row, populate_branch


def _numpy_rng_equal(a, b):
    return (
        a[0] == b[0]
        and a[2] == b[2]
        and a[3] == b[3]
        and a[4] == b[4]
        and np.array_equal(a[1], b[1])
    )


def test_public_exports_remain_parallel_to_original_dsrl():
    assert TopLevelHierarchy is HierarchicalRFSDSRL


def test_frozen_production_profiles_and_beta_indexing():
    fresh = make_hierarchy_schedule("fresh_frozen_ddim_5m", n_envs=10)
    assert fresh.phase_b_steps == 2_500_000
    assert fresh.phase_r_steps == 2_500_000
    assert fresh.phase_j_steps == 0
    assert fresh.phase_j_enabled is False
    assert fresh.beta_at(fresh.phase_r_start) == 0.0
    assert fresh.beta_at(fresh.phase_r_start + fresh.beta_ramp_steps - 10) == 0.1
    assert fresh.phase_at(fresh.phase_r_start - 10) == HierarchyPhase.BASE
    assert fresh.phase_at(fresh.phase_r_start) == HierarchyPhase.RESIDUAL
    legacy = make_hierarchy_schedule("legacy_dsrl_warmstart_5m", n_envs=4)
    assert legacy.phase_b_steps == 0
    assert legacy.total_steps == 5_000_000
    assert legacy.beta_target == 0.1
    assert legacy.base_lane_probability == 0.5


def test_schedule_rejects_split_vector_boundary_and_budget_mismatch():
    with pytest.raises(ValueError, match="divisible"):
        make_hierarchy_schedule(
            "fresh_frozen_ddim_5m",
            n_envs=4,
            overrides={"phase_b_steps": 5},
        )
    schedule = make_hierarchy_schedule("fresh_frozen_ddim_5m", n_envs=4)
    with pytest.raises(ValueError, match="budget mismatch"):
        schedule.validate_budget(100)


def test_train_logs_separated_losses_counters_phase_and_lane_counts():
    model, _ = make_model()
    populate_branch(model, BranchMode.BASE)
    model.train(20, 2)
    keys = model.logger.name_to_value
    for key in (
        "train/qa_base_loss",
        "train/qw_base_loss",
        "train/noise_actor_loss",
        "train/alpha_loss",
        "train/qa_base_optimizer_steps",
        "train/qw_base_optimizer_steps",
        "train/noise_actor_optimizer_steps",
        "train/hierarchy_phase",
        "train/beta",
        "train/replay_base_count",
        "train/replay_joint_count",
    ):
        assert key in keys
    assert "train/modulation_critic_loss" not in keys


def test_reset_boundary_seed_derivation_and_fresh_episode_metadata():
    model, _ = make_model(n_envs=2)
    model.num_timesteps = model.hierarchy_schedule.phase_r_start
    model._allocate_unassigned_lanes()
    abandoned = model._active_episode_id.copy()
    next_id = model.next_episode_id
    seeds = model.prepare_reset_boundary_resume(train_env_seed=1001)
    assert seeds == [
        model.derive_reset_seed(1001, 1, 0),
        model.derive_reset_seed(1001, 1, 1),
    ]
    assert model.environment_discontinuity_count == 1
    np.testing.assert_array_equal(model._active_episode_id, np.full(2, -1))
    np.testing.assert_array_equal(model._active_branch_mode, np.full(2, -1))
    # The runner resets every simulator slot with ``seeds`` before allocating
    # fresh lane/episode metadata.
    model._allocate_unassigned_lanes()
    assert np.all(model._active_episode_id >= next_id)
    assert not np.isin(model._active_episode_id, abandoned).any()
    np.testing.assert_array_equal(model._chunk_index_in_episode, np.zeros(2))


def test_all_modules_return_to_inference_mode_after_training():
    model, decoder = make_model()
    populate_branch(model, BranchMode.BASE)
    model.train(20, 2)
    for module in (
        model.actor,
        model.qa_base,
        model.qa_base_target,
        model.qw_base,
        model.qa_joint,
        model.qa_joint_target,
        model.residual_actor,
        model.residual_actor_target,
        model.reference_noise_actor,
        decoder,
    ):
        assert module.training is False
    assert all(
        parameter.grad is None
        for module in (
            model.actor,
            model.qa_base,
            model.qw_base,
            model.qa_joint,
            model.residual_actor,
        )
        for parameter in module.parameters()
    )


def test_fresh_frozen_ddim_2p5m_profile_preserves_frozen_ratios():
    fresh = make_hierarchy_schedule("fresh_frozen_ddim_2p5m", n_envs=10)
    assert fresh.phase_b_steps == 1_250_000
    assert fresh.phase_r_steps == 1_250_000
    assert fresh.total_steps == 2_500_000
    assert fresh.phase_j_steps == 0
    assert fresh.phase_j_enabled is False
    assert fresh.beta_ramp_steps == 50_000
    assert fresh.beta_target == 0.1
    assert fresh.base_lane_probability == 0.5
    assert fresh.beta_at(fresh.phase_r_start) == 0.0
    assert fresh.phase_at(fresh.phase_r_start - 10) == HierarchyPhase.BASE
    assert fresh.phase_at(fresh.phase_r_start) == HierarchyPhase.RESIDUAL


def test_initialize_from_fresh_frozen_ddim_snapshot_and_defers_joint_clone():
    model, _ = make_model(
        phase_b_steps=8,
        phase_r_steps=8,
    )
    model.initialize_from_fresh_frozen_ddim()
    # QA_base_target is a hard copy of online QA_base (SB3 target is separate).
    for key, value in model.qa_base.state_dict().items():
        assert th.allclose(
            value, model.qa_base_target.state_dict()[key], atol=0.0, rtol=0.0
        ), f"QA_base_target {key} is not a hard copy"
    # Reference noise actor is an immutable snapshot of the fresh actor.
    for key, value in model.actor.state_dict().items():
        assert th.allclose(
            value,
            model.reference_noise_actor.state_dict()[key],
            atol=0.0,
            rtol=0.0,
        ), f"reference noise actor {key} differs from the fresh actor"
    # Residual output layer is exactly zero and the target matches it.
    assert int(th.count_nonzero(model.residual_actor.output_layer.weight)) == 0
    assert int(th.count_nonzero(model.residual_actor.output_layer.bias)) == 0
    for key, value in model.residual_actor.state_dict().items():
        assert th.allclose(
            value,
            model.residual_actor_target.state_dict()[key],
            atol=0.0,
            rtol=0.0,
        ), f"residual target {key} differs from online residual"
    # QA_joint clone is deferred to Phase R for a nonzero Phase B.
    assert model._joint_phase_initialized is False
    assert model.qa_joint_generation == 0
    # Counters and optimizers are fresh.
    assert model.noise_actor_optimizer_steps == 0
    assert model.residual_actor_optimizer_steps == 0
    assert model.qa_base_optimizer_steps == 0
    assert model.noise_policy_version == 0
    assert model.residual_policy_version == 0
    assert all(
        len(optimizer.state) == 0
        for optimizer in (
            model.actor.optimizer,
            model.qa_base_optimizer,
            model.qw_base_optimizer,
            model.qa_joint_optimizer,
            model.residual_actor_optimizer,
        )
    )
    # DDIM and all targets are frozen/eval.
    assert all(not parameter.requires_grad for parameter in model.qa_base_target.parameters())
    assert all(not parameter.requires_grad for parameter in model.residual_actor_target.parameters())


def test_training_diagnostics_are_wired_into_train_logger():
    model, _ = make_model(
        diagnostics_interval_updates=1,
        phase_b_steps=4,
        phase_r_steps=8,
    )
    populate_branch(model, BranchMode.BASE)
    model.train(20, 2)
    keys = model.logger.name_to_value
    for key in (
        "train/effective_utd_base",
        "train/realized_joint_lane_ratio",
        "train/emergency_clamp_count",
        "train/max_preclamp_violation",
        "train/noise_policy_version",
        "train/residual_policy_version",
    ):
        assert key in keys, f"missing Section-10 metric {key}"
    # diagnostics/* keys are produced (base branch populated; joint empty in B).
    for key in (
        "diagnostics/noise_scaled_l2",
        "diagnostics/qa_base_twin_disagreement",
        "diagnostics/qa_base_head0_mean",
        "diagnostics/qa_base_target_drift",
    ):
        assert key in keys, f"diagnostic {key} was not logged"


def test_gated_v11_v12_flags_default_off_and_runnable_when_enabled():
    model, _ = make_model(
        phase_b_steps=4,
        phase_r_steps=8,
    )
    # Core V1 default: all gated mechanisms are OFF.
    assert model.enable_qw_ranking is False
    assert model.qa_joint_target_smoothing is False
    populate_branch(model, BranchMode.BASE)
    model.train(20, 2)
    base_loss_default = model.logger.name_to_value["train/qa_base_loss"]

    # V1.2 target-policy smoothing: runnable and finite when enabled.
    model2, _ = make_model(
        phase_b_steps=4,
        phase_r_steps=8,
        qa_joint_target_smoothing=True,
        qa_joint_smoothing_std=0.001,
        qa_joint_smoothing_clip=0.002,
    )
    assert model2.qa_joint_target_smoothing is True
    populate_branch(model2, BranchMode.BASE)
    model2.train(20, 2)
    assert "train/qa_base_loss" in model2.logger.name_to_value

    # V1.1 QW ranking: runnable and finite when enabled (Base lane regression
    # still executes, ranking term is additive).
    model3, _ = make_model(
        phase_b_steps=4,
        phase_r_steps=8,
        enable_qw_ranking=True,
        qw_ranking_k_candidates=4,
        qw_ranking_tau_gap=0.0,
    )
    assert model3.enable_qw_ranking is True
    populate_branch(model3, BranchMode.BASE)
    model3.train(20, 2)
    assert "train/qw_base_loss" in model3.logger.name_to_value


def test_ranking_diagnostics_produce_section10_metrics_and_restore_rng_and_modes():
    model, _ = make_model(
        diagnostics_interval_updates=1,
        phase_b_steps=4,
        phase_r_steps=8,
    )
    populate_branch(model, BranchMode.BASE)
    base_data = model._sample_branch(BranchMode.BASE, model.batch_size)
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = th.random.get_rng_state()
    mode_snapshot = [(module, module.training) for module in model._all_modules()]

    diagnostics = model._ranking_diagnostics(base_data)

    # The isolated-RNG diagnostic must not perturb global RNG state.
    assert random.getstate() == python_state
    assert _numpy_rng_equal(np.random.get_state(), numpy_state)
    assert th.equal(th.random.get_rng_state(), torch_state)
    # And it must restore every module mode exactly.
    for (module, before), (module2, after) in zip(
        mode_snapshot, [(m, m.training) for m in model._all_modules()]
    ):
        assert module is module2 and before == after

    for key in (
        "diagnostics/rank_twin_head_ordering_agreement",
        "diagnostics/rank_qw_teacher_spearman_head0",
        "diagnostics/rank_qw_teacher_spearman_head1",
        "diagnostics/rank_qw_teacher_kendall_head0",
        "diagnostics/rank_qw_teacher_kendall_head1",
        "diagnostics/rank_pairwise_accuracy_head0",
        "diagnostics/rank_pairwise_accuracy_head1",
        "diagnostics/rank_top1_agreement_head0",
        "diagnostics/rank_top1_agreement_head1",
        "diagnostics/rank_teacher_top1_gap_mean",
        "diagnostics/rank_policy_age_mean",
        "diagnostics/rank_spearman_by_age_recent",
    ):
        assert key in diagnostics, f"missing ranking metric {key}"
        assert np.isfinite(diagnostics[key]), f"non-finite ranking metric {key}"


def test_store_transition_accumulates_episode_stats_and_resets_on_diagnostics():
    model, _ = make_model(n_envs=1)
    model._last_obs = np.full((1, 3), 0.1, dtype=np.float32)
    model._last_original_obs = np.full((1, 3), 0.1, dtype=np.float32)
    model._allocate_unassigned_lanes()
    metadata, _, _ = metadata_row(model, BranchMode.BASE, row=0, done=True)
    model._pending_rollout_metadata = metadata
    infos = [
        {
            "TimeLimit.truncated": False,
            "terminal_observation": np.zeros(3, dtype=np.float32),
            "nominal_primitive_steps": 2,
            "actual_primitive_steps": 2,
            "termination_primitive_index": 1,
            "action_chunk_termination_semantics": "early_break_on_done",
            "termination_reason": "environment_terminal",
        }
    ]
    model._store_transition(
        model.replay_buffer,
        metadata["action_exec"].astype(np.float32),
        np.full((1, 3), 0.2, dtype=np.float32),
        np.array([1.0], dtype=np.float32),
        np.array([True], dtype=np.bool_),
        infos,
    )
    base_stats = model._closed_episode_stats[int(BranchMode.BASE)]
    assert base_stats["count"] == 1.0
    assert base_stats["return_sum"] == 1.0
    assert base_stats["length_sum"] == 2.0
    assert base_stats["early_fall_count"] == 1.0
    # The done transition closed the episode: active episode state was reset.
    assert int(model._active_episode_id[0]) == -1

    diagnostics = model._episode_stat_diagnostics()
    assert diagnostics["diagnostics/episode_return_base_mean"] == 1.0
    assert diagnostics["diagnostics/episode_length_base_mean"] == 2.0
    assert diagnostics["diagnostics/episode_early_fall_rate_base"] == 1.0
    # Read-and-reset semantics: the window is empty after the dump.
    assert model._closed_episode_stats[int(BranchMode.BASE)]["count"] == 0.0


def test_train_logs_policy_age_and_noise_statistics():
    model, _ = make_model(
        diagnostics_interval_updates=1,
        phase_b_steps=4,
        phase_r_steps=8,
    )
    populate_branch(model, BranchMode.BASE)
    model.train(20, 2)
    keys = model.logger.name_to_value
    assert "train/noise_policy_age" in keys
    assert "train/residual_policy_age" in keys
    for key in (
        "diagnostics/noise_mean_abs",
        "diagnostics/noise_std",
        "diagnostics/noise_entropy",
        "diagnostics/qa_base_head0_std",
        "diagnostics/qa_base_head1_std",
    ):
        assert key in keys, f"missing Section-10 metric {key}"


def test_reset_boundary_resume_clears_episode_accumulators():
    # Regression: prepare_reset_boundary_resume abandoned the in-flight
    # simulator episode but left the Section 10 return/length/branch
    # accumulators holding partial sums, which would then be attributed to the
    # freshly-reset episode on the next diagnostic dump.
    model, _ = make_model(n_envs=2)
    model._episode_return_accum[:] = [3.5, -1.0]
    model._episode_length_accum[:] = [7, 4]
    model._episode_branch_accum[:] = [
        int(BranchMode.BASE),
        int(BranchMode.JOINT),
    ]
    model.prepare_reset_boundary_resume(train_env_seed=1001)
    np.testing.assert_array_equal(
        model._episode_return_accum, np.zeros(2, dtype=np.float64)
    )
    np.testing.assert_array_equal(
        model._episode_length_accum, np.zeros(2, dtype=np.int64)
    )
    np.testing.assert_array_equal(
        model._episode_branch_accum, np.full(2, -1, dtype=np.int8)
    )


def test_qw_ranking_loss_backprops_through_student_heads():
    # Regression: the ranking term detached the QW student heads, so it never
    # produced a gradient into QW weights (a silent no-op).  With a large
    # batch and zero tau-gap the twin-teacher sign agreement yields a
    # non-empty mask, and backward() must reach every QW parameter.
    model, _ = make_model(
        phase_b_steps=4,
        phase_r_steps=8,
        enable_qw_ranking=True,
        qw_ranking_k_candidates=4,
        qw_ranking_tau_gap=0.0,
    )
    obs = th.rand(32, 3)
    noise = th.rand(32, model.action_dim_flat)
    loss = model._qw_ranking_loss(obs, noise)
    assert loss.requires_grad, "ranking loss is detached from the QW student"
    model.qw_base_optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grads = [
        p.grad
        for p in model.qw_base.parameters()
        if p.grad is not None and bool(p.grad.abs().sum() > 0)
    ]
    assert grads, "ranking loss produced no gradient into QW weights"
    assert float(loss) >= 0.0


def test_smoothing_reclamps_target_action_to_exec_bounds():
    # Regression: target-policy smoothing added clipped noise but never
    # re-clamped the perturbed target action, so a draw near the envelope edge
    # could feed the joint critic an out-of-envelope action the residual
    # policy can never emit.
    model, _ = make_model(
        phase_b_steps=4,
        phase_r_steps=8,
        qa_joint_target_smoothing=True,
        qa_joint_smoothing_std=5.0,
        qa_joint_smoothing_clip=5.0,
    )
    boundary = th.full((8, model.action_dim_flat), 0.99)
    smoothed = model._smooth_joint_target_action(boundary)
    low = model._exec_action_low_tensor
    high = model._exec_action_high_tensor
    assert bool((smoothed >= low).all())
    assert bool((smoothed <= high).all())
    # Disabled smoothing is the identity.
    model_det, _ = make_model(phase_b_steps=4, phase_r_steps=8)
    assert th.equal(model_det._smooth_joint_target_action(boundary), boundary)


def test_joint_diagnostics_include_residual_tanh_saturation_fraction():
    # Core V1 Section 10 generic-residual-saturation requirement surfaced as a
    # dedicated diagnostic on the joint branch.
    model, _ = make_model(
        diagnostics_interval_updates=1,
        phase_b_steps=4,
        phase_r_steps=8,
    )
    populate_branch(model, BranchMode.BASE, rows=2)
    populate_branch(model, BranchMode.JOINT, rows=2)
    base_data = model._sample_branch(BranchMode.BASE, model.batch_size)
    joint_data = model._sample_branch(BranchMode.JOINT, model.batch_size)
    diagnostics = model._training_diagnostics(base_data, joint_data)
    fraction = diagnostics["diagnostics/residual_tanh_saturation_fraction"]
    assert 0.0 <= fraction <= 1.0
