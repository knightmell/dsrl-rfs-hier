"""Production P6 runner for matched DSRL control and hierarchy experiments."""

from __future__ import annotations

import json
import math
import os
import random
import shutil
import sys
import traceback
from pathlib import Path
from typing import Any, Mapping

import gym
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.tensorboard import SummaryWriter
from hydra.core.hydra_config import HydraConfig

BASE_PATH = Path(__file__).resolve().parent
sys.path.append(str(BASE_PATH / "dppo"))

from env_utils import ACTION_CHUNK_EARLY_BREAK, ActionChunkWrapper, ObservationWrapperGym
from p6_checkpointing import (
    P6CheckpointManager,
    P6IntentionalInterruption,
    P6TrainingCallback,
    initial_runtime_state,
    load_resume_payload,
    run_binding_from_manifest,
    validate_loaded_resume,
    validate_optimizer_counter_invariants,
)
from p6_evaluation import evaluate_exact_episodes, persist_evaluation
from p6_preflight import (
    CONTROL_ALGORITHM,
    FRESH_PREFILL_ACTION_POLICY,
    FRESH_PREFILL_SOURCE,
    HIERARCHY_ALGORITHM,
    HIERARCHY_ALGORITHMS,
    HIERARCHY_PREFILL_RESIDUAL_MODE,
    PREFILL_ACTION_POLICY,
    PREFILL_SOURCE,
    finalize_loaded_model_preflight,
    resolve_seed_plan,
    run_preflight,
    sha256_file,
    static_preflight,
    validate_execution_bounds,
    validate_observation_dimension,
)
from p6_runtime import (
    atomic_write_json,
    canonical_module_state_hash,
    collect_or_load_tagged_matched_prefill,
    isolated_rng,
    populate_tagged_replay_buffer,
    populate_replay_buffer,
    reset_vec_env_with_explicit_seeds,
    restore_rng_state,
    seed_all,
    tagged_prefill_standard_projection,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.dsrl.hierarchical_rfs_dsrl import (
    HierarchicalRFSDSRL,
    QW_TEACHER_SOURCE_CURRENT_ACTOR,
    _LegacyLoadableDSRL,
)
from stable_baselines3.dsrl.hierarchical_replay_buffer import (
    HierarchyTaggedReplayBuffer,
)
from utils import load_base_policy


OmegaConf.register_new_resolver("eval", eval, replace=True)
OmegaConf.register_new_resolver("round_up", math.ceil, replace=True)
OmegaConf.register_new_resolver("round_down", math.floor, replace=True)


def _set_model_inference_mode(model: Any) -> None:
    policy = getattr(model, "policy", None)
    if policy is not None and hasattr(policy, "set_training_mode"):
        policy.set_training_mode(False)
    for name in (
        "actor",
        "critic",
        "critic_target",
        "critic_noise",
        "qa_base",
        "qa_base_target",
        "qw_base",
        "qa_joint",
        "qa_joint_target",
        "residual_actor",
        "residual_actor_target",
        "reference_noise_actor",
    ):
        module = getattr(model, name, None)
        if module is None:
            continue
        if hasattr(module, "set_training_mode"):
            module.set_training_mode(False)
        elif isinstance(module, torch.nn.Module):
            module.eval()
    diffusion_policy = getattr(model, "diffusion_policy", None)
    for candidate in (
        diffusion_policy,
        getattr(diffusion_policy, "base_policy", None),
    ):
        if isinstance(candidate, torch.nn.Module):
            candidate.eval()
            candidate.requires_grad_(False)


class P6ControlDSRL(_LegacyLoadableDSRL):
    """Legacy DSRL with P6-only persistent optimizer counters."""

    def _setup_model(self) -> None:
        super()._setup_model()
        for name in (
            "action_critic_optimizer_steps",
            "modulation_critic_optimizer_steps",
            "noise_actor_optimizer_steps",
            "residual_actor_optimizer_steps",
            "hierarchy_train_calls",
        ):
            if not hasattr(self, name):
                setattr(self, name, 0)
        _set_model_inference_mode(self)

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        super().train(gradient_steps=gradient_steps, batch_size=batch_size)
        actor_steps = (
            gradient_steps
            if self.actor_gradient_steps < 0
            else self.actor_gradient_steps
        )
        self.action_critic_optimizer_steps += int(gradient_steps)
        self.modulation_critic_optimizer_steps += int(
            self.noise_critic_grad_steps
        )
        self.noise_actor_optimizer_steps += int(actor_steps)
        self.hierarchy_train_calls += 1
        self.logger.record(
            "train/action_critic_optimizer_steps",
            self.action_critic_optimizer_steps,
        )
        self.logger.record(
            "train/modulation_critic_optimizer_steps",
            self.modulation_critic_optimizer_steps,
        )
        self.logger.record(
            "train/noise_actor_optimizer_steps",
            self.noise_actor_optimizer_steps,
        )
        self.logger.record(
            "train/residual_actor_optimizer_steps",
            self.residual_actor_optimizer_steps,
        )
        self.logger.record("train/hierarchy_train_calls", self.hierarchy_train_calls)
        _set_model_inference_mode(self)

    def _get_torch_save_params(self) -> tuple[list[str], list[str]]:
        state_dicts, pytorch_variables = super()._get_torch_save_params()
        if "critic_noise.optimizer" not in state_dicts:
            state_dicts.append("critic_noise.optimizer")
        return state_dicts, pytorch_variables


def _reset_control_optimizers(model: P6ControlDSRL) -> None:
    learning_rate = model.lr_schedule(1)
    model.actor.optimizer = model.policy.optimizer_class(
        model.actor.parameters(),
        lr=learning_rate,
        **model.policy.optimizer_kwargs,
    )
    if model.policy.share_features_extractor:
        critic_parameters = [
            parameter
            for name, parameter in model.critic.named_parameters()
            if "features_extractor" not in name
        ]
    else:
        critic_parameters = list(model.critic.parameters())
    model.critic.optimizer = model.policy.optimizer_class(
        critic_parameters,
        lr=learning_rate,
        **model.policy.optimizer_kwargs,
    )
    model.critic_noise.optimizer = model.policy.optimizer_class(
        model.critic_noise.parameters(),
        lr=learning_rate,
        **model.policy.optimizer_kwargs,
    )
    if model.log_ent_coef is not None:
        model.ent_coef_optimizer = torch.optim.Adam(
            [model.log_ent_coef],
            lr=learning_rate,
        )
    model.num_timesteps = 0
    model._n_updates = 0
    model._episode_num = 0
    model._last_obs = None
    for name in (
        "action_critic_optimizer_steps",
        "modulation_critic_optimizer_steps",
        "noise_actor_optimizer_steps",
        "residual_actor_optimizer_steps",
        "hierarchy_train_calls",
    ):
        setattr(model, name, 0)
    _set_model_inference_mode(model)


def _promote_legacy_control(model: _LegacyLoadableDSRL) -> P6ControlDSRL:
    """Attach P6 resume behavior after the official legacy loader succeeds."""

    if type(model) is not _LegacyLoadableDSRL:
        raise TypeError(
            "Only the private official-load shell may be promoted to P6 control"
        )
    model.__class__ = P6ControlDSRL
    for name in (
        "action_critic_optimizer_steps",
        "modulation_critic_optimizer_steps",
        "noise_actor_optimizer_steps",
        "residual_actor_optimizer_steps",
        "hierarchy_train_calls",
    ):
        setattr(model, name, 0)
    return model


def _make_policy_kwargs(cfg: Any) -> dict[str, Any]:
    hidden = [int(cfg.train.layer_size)] * int(cfg.train.num_layers)
    post_linear_modules = [torch.nn.LayerNorm] if cfg.train.use_layer_norm else None
    return {
        "net_arch": {"pi": hidden, "qf": hidden},
        "activation_fn": torch.nn.Tanh,
        "log_std_init": 0.0,
        "post_linear_modules": post_linear_modules,
        "n_critics": int(cfg.train.n_critics),
    }


def _make_locomotion_environment(cfg: Any, normalization_path: Path):
    # Import lazily so CPU-only unit tests can exercise runner state classes
    # without requiring the legacy D4RL/MuJoCo environment.
    import d4rl  # noqa: F401
    import d4rl.gym_mujoco  # noqa: F401

    raw_environment = gym.make(cfg.env_name)
    normalized_environment = ObservationWrapperGym(
        raw_environment,
        normalization_path,
    )
    return ActionChunkWrapper(
        normalized_environment,
        cfg,
        max_episode_steps=int(cfg.env.max_episode_steps),
        action_chunk_termination_semantics=ACTION_CHUNK_EARLY_BREAK,
    )


# Backward-compat alias: the migration renamed `_make_hopper_environment` to
# `_make_locomotion_environment` (both build a normalized, action-chunked
# environment).  Diagnostic/experiment scripts and their tests still import the
# old name through multiple transitive paths, so keep it importable to avoid
# breaking collection of the test suite.
_make_hopper_environment = _make_locomotion_environment


def _load_legacy_network(
    *,
    cfg: Any,
    environment: Any,
    diffusion_policy: Any,
    buffer_size: int,
) -> _LegacyLoadableDSRL:
    checkpoint_path = hydra.utils.to_absolute_path(
        str(cfg.rfs_hier_legacy_checkpoint_path)
    )
    model = _LegacyLoadableDSRL.load(
        checkpoint_path,
        env=environment,
        device=cfg.device,
        custom_objects={"diffusion_policy": diffusion_policy},
        buffer_size=buffer_size,
        tensorboard_log=cfg.logdir,
        seed=int(cfg.seed),
    )
    expected_steps = int(cfg.p6.expected_init_checkpoint_steps)
    if int(model.num_timesteps) != expected_steps:
        raise ValueError(
            f"Official checkpoint step mismatch: {model.num_timesteps} != "
            f"{expected_steps}"
        )
    if model.critic_backup_combine_type != "min":
        raise ValueError("P6 checkpoint must use critic_backup_combine_type='min'")
    _validate_legacy_training_contract(cfg, model)
    _set_model_inference_mode(model)
    return model


def _validate_legacy_training_contract(cfg: Any, model: Any) -> None:
    train_frequency = getattr(model, "train_freq", None)
    actual_frequency = int(getattr(train_frequency, "frequency", -1))
    actual_unit = str(getattr(train_frequency, "unit", "")).lower()
    expected = {
        "actor_learning_rate": float(cfg.train.actor_lr),
        "batch_size": int(cfg.train.batch_size),
        "gamma": float(cfg.train.discount),
        "gradient_steps": int(cfg.train.utd),
        "learning_starts": 1,
        "noise_critic_grad_steps": int(cfg.train.noise_critic_grad_steps),
        "target_entropy": float(cfg.train.target_ent),
        "target_update_interval": 1,
        "tau": float(cfg.train.tau),
        "train_freq": int(cfg.train.train_freq),
    }
    actual = {
        "actor_learning_rate": float(model.lr_schedule(1.0)),
        "batch_size": int(model.batch_size),
        "gamma": float(model.gamma),
        "gradient_steps": int(model.gradient_steps),
        "learning_starts": int(model.learning_starts),
        "noise_critic_grad_steps": int(model.noise_critic_grad_steps),
        "target_entropy": float(model.target_entropy),
        "target_update_interval": int(model.target_update_interval),
        "tau": float(model.tau),
        "train_freq": actual_frequency,
    }
    for key, expected_value in expected.items():
        if actual[key] != expected_value:
            raise ValueError(
                f"Legacy checkpoint training contract mismatch for {key}: "
                f"{actual[key]} != {expected_value}"
            )
    if "step" not in actual_unit:
        raise ValueError(
            f"Legacy checkpoint train-frequency unit must be step, got {actual_unit}"
        )
    if int(model.actor_gradient_steps) != -1:
        raise ValueError("P6 control requires checkpoint actor_gradient_steps=-1")
    learned_entropy_expected = float(cfg.train.ent_coef) == -1.0
    if learned_entropy_expected != (model.log_ent_coef is not None):
        raise ValueError("Legacy checkpoint entropy mode differs from P6 config")
    if not learned_entropy_expected:
        actual_entropy = float(model.ent_coef_tensor.detach().cpu().item())
        expected_entropy = float(cfg.train.ent_coef)
        if actual_entropy != expected_entropy:
            raise ValueError(
                "Legacy checkpoint fixed entropy coefficient mismatch: "
                f"{actual_entropy} != {expected_entropy}"
            )


def _cloned_state_dict(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in module.state_dict().items()
    }


def _state_dict_max_abs_error(
    actual: Mapping[str, torch.Tensor],
    expected: Mapping[str, torch.Tensor],
    *,
    module_name: str,
) -> float:
    if actual.keys() != expected.keys():
        raise ValueError(f"{module_name} state_dict keys differ")
    maximum = 0.0
    for key in actual:
        actual_value = actual[key].detach().cpu()
        expected_value = expected[key].detach().cpu()
        if actual_value.shape != expected_value.shape:
            raise ValueError(f"{module_name}.{key} shape differs")
        if actual_value.numel():
            maximum = max(
                maximum,
                float(torch.max(torch.abs(actual_value - expected_value)).item()),
            )
    if maximum != 0.0:
        raise ValueError(f"{module_name} migration is not exact: {maximum}")
    return maximum


def _capture_legacy_warmstart(model: Any) -> dict[str, Any]:
    return {
        "actor": _cloned_state_dict(model.actor),
        "critic": _cloned_state_dict(model.critic),
        "critic_target": _cloned_state_dict(model.critic_target),
        "critic_noise": _cloned_state_dict(model.critic_noise),
        "log_ent_coef": (
            None
            if model.log_ent_coef is None
            else model.log_ent_coef.detach().cpu().clone()
        ),
        "target_entropy": float(model.target_entropy),
    }


def _verify_fresh_control_warmstart(
    model: P6ControlDSRL,
    expected: Mapping[str, Any],
) -> dict[str, float]:
    errors = {
        "actor_state_max_abs_error": _state_dict_max_abs_error(
            model.actor.state_dict(),
            expected["actor"],
            module_name="control_actor",
        ),
        "action_critic_state_max_abs_error": _state_dict_max_abs_error(
            model.critic.state_dict(),
            expected["critic"],
            module_name="control_action_critic",
        ),
        "action_critic_target_state_max_abs_error": _state_dict_max_abs_error(
            model.critic_target.state_dict(),
            expected["critic_target"],
            module_name="control_action_critic_target",
        ),
        "noise_critic_state_max_abs_error": _state_dict_max_abs_error(
            model.critic_noise.state_dict(),
            expected["critic_noise"],
            module_name="control_noise_critic",
        ),
    }
    if float(model.target_entropy) != float(expected["target_entropy"]):
        raise ValueError("Control target_entropy changed during promotion")
    if expected["log_ent_coef"] is None or model.log_ent_coef is None:
        if expected["log_ent_coef"] is not None or model.log_ent_coef is not None:
            raise ValueError("Control entropy mode changed during promotion")
        entropy_error = 0.0
    else:
        entropy_error = float(
            torch.max(
                torch.abs(
                    model.log_ent_coef.detach().cpu()
                    - expected["log_ent_coef"]
                )
            ).item()
        )
        if entropy_error != 0.0:
            raise ValueError("Control entropy coefficient changed during promotion")
    errors["entropy_coefficient_max_abs_error"] = entropy_error

    optimizers = (
        model.actor.optimizer,
        model.critic.optimizer,
        model.critic_noise.optimizer,
        model.ent_coef_optimizer,
    )
    if any(
        optimizer is not None and len(optimizer.state) != 0
        for optimizer in optimizers
    ):
        raise ValueError("Control network warm-start must use fresh optimizers")
    if model.num_timesteps != 0 or model._n_updates != 0:
        raise ValueError("Control network warm-start counters must begin at zero")
    if any(
        value != 0
        for value in (
            model.action_critic_optimizer_steps,
            model.modulation_critic_optimizer_steps,
            model.noise_actor_optimizer_steps,
            model.residual_actor_optimizer_steps,
            model.hierarchy_train_calls,
        )
    ):
        raise ValueError("Control optimizer counters must begin at zero")
    _set_model_inference_mode(model)
    return errors


def _hierarchy_init_mode(cfg: Any) -> str:
    """'fresh' for Frozen-DDIM from-scratch profiles, 'warmstart' otherwise."""
    profile_name = str(cfg.train.rfs_hier_schedule_profile)
    if profile_name in (
        "fresh_frozen_ddim_5m",
        "fresh_frozen_ddim_2p5m",
        "fresh_frozen_ddim_2p5m_cotrain",
        "fresh_frozen_ddim_2p5m_additive_res",
        "fresh_frozen_ddim_2p5m_base_continue",
    ):
        return "fresh"
    return "warmstart"


def _control_is_fresh(cfg: Any) -> bool:
    """A matched flat-DSRL control starts fresh when it has no legacy
    checkpoint (rfs_hier_legacy_checkpoint_path=null).  It is initialized from
    the Frozen DDIM with the identical Gaussian-prior prefill and seed family
    as the fresh hierarchy, so it shares the exact prefill artifact and base
    branch init."""
    return cfg.rfs_hier_legacy_checkpoint_path is None


def _run_is_fresh(cfg: Any, algorithm: str) -> bool:
    """True when a P6 run starts from Frozen DDIM only (no legacy checkpoint):
    fresh hierarchy profiles, or a fresh matched flat-DSRL control."""
    if algorithm == HIERARCHY_ALGORITHM:
        return _hierarchy_init_mode(cfg) == "fresh"
    if algorithm == CONTROL_ALGORITHM:
        return _control_is_fresh(cfg)
    return False


def _construct_hierarchy(
    cfg: Any,
    environment: Any,
    diffusion_policy: Any,
    *,
    init_mode: str,
) -> HierarchicalRFSDSRL:
    if init_mode not in ("fresh", "warmstart"):
        raise ValueError(f"Unknown hierarchy init mode {init_mode!r}")
    execution_low = np.asarray(environment.action_space.low, dtype=np.float32)
    execution_high = np.asarray(environment.action_space.high, dtype=np.float32)
    model = HierarchicalRFSDSRL(
        "MlpPolicy",
        environment,
        learning_rate=float(cfg.train.actor_lr),
        buffer_size=int(cfg.train.buffer_size_na),
        learning_starts=1,
        batch_size=int(cfg.train.batch_size),
        tau=float(cfg.train.tau),
        gamma=float(cfg.train.discount),
        train_freq=int(cfg.train.rfs_hier_train_freq),
        gradient_steps=int(cfg.train.utd),
        action_noise=None,
        replay_buffer_class=HierarchyTaggedReplayBuffer,
        optimize_memory_usage=False,
        ent_coef="auto" if cfg.train.ent_coef == -1 else cfg.train.ent_coef,
        target_update_interval=1,
        target_entropy=(
            "auto" if cfg.train.target_ent == -1 else cfg.train.target_ent
        ),
        use_sde=False,
        sde_sample_freq=-1,
        tensorboard_log=cfg.logdir,
        verbose=1,
        device=cfg.device,
        policy_kwargs=_make_policy_kwargs(cfg),
        diffusion_policy=diffusion_policy,
        diffusion_act_dim=(int(cfg.act_steps), int(cfg.action_dim)),
        noise_critic_grad_steps=int(cfg.train.noise_critic_grad_steps),
        critic_backup_combine_type=str(cfg.train.critic_backup_combine_type),
        exec_action_low=execution_low,
        exec_action_high=execution_high,
        residual_net_arch=tuple(cfg.train.rfs_hier_residual_net_arch),
        residual_activation=str(cfg.train.rfs_hier_residual_activation),
        residual_lr=float(cfg.train.rfs_hier_residual_lr),
        qa_joint_lr=float(cfg.train.rfs_hier_qa_joint_lr),
        schedule_profile=str(cfg.train.rfs_hier_schedule_profile),
        phase_b_steps=int(cfg.train.rfs_hier_phase_b_steps),
        phase_r_steps=int(cfg.train.rfs_hier_phase_r_steps),
        phase_j_steps=int(cfg.train.rfs_hier_phase_j_steps),
        phase_j_enabled=bool(cfg.train.rfs_hier_phase_j_enabled),
        beta_ramp_steps=int(cfg.train.rfs_hier_beta_ramp_steps),
        beta_target=float(cfg.train.rfs_hier_beta_target),
        base_lane_probability=float(
            cfg.train.rfs_hier_base_lane_probability
        ),
        # Co-training schedule/model fields.  Missing keys default to the
        # profile's frozen values (None -> schedule default; False/0.0 for the
        # model-level flags), so every existing config is unchanged.
        beta_hold_steps=(
            None
            if cfg.train.get("rfs_hier_beta_hold_steps") is None
            else int(cfg.train.rfs_hier_beta_hold_steps)
        ),
        beta_floor=(
            None
            if cfg.train.get("rfs_hier_beta_floor") is None
            else float(cfg.train.rfs_hier_beta_floor)
        ),
        qa_joint_shadow_in_b=bool(
            cfg.train.get("rfs_hier_qa_joint_shadow_in_b", False)
        ),
        qw_teacher_joint_credit=bool(
            cfg.train.get("rfs_hier_qw_teacher_joint_credit", False)
        ),
        qw_teacher_source=str(
            cfg.train.get(
                "rfs_hier_qw_teacher_source",
                QW_TEACHER_SOURCE_CURRENT_ACTOR,
            )
        ),
        qw_candidates_per_state=int(
            cfg.train.get("rfs_hier_qw_candidates_per_state", 1)
        ),
        qw_state_batch_size=int(
            cfg.train.get("rfs_hier_qw_state_batch_size", cfg.train.batch_size)
        ),
        qw_teacher_microbatch_size=int(
            cfg.train.get(
                "rfs_hier_qw_teacher_microbatch_size",
                int(cfg.train.get("rfs_hier_qw_state_batch_size", cfg.train.batch_size))
                * int(cfg.train.get("rfs_hier_qw_candidates_per_state", 1)),
            )
        ),
        cross_lane_ratio=float(
            cfg.train.get("rfs_hier_cross_lane_ratio", 0.0)
        ),
        qa_base_cross_lane=bool(
            cfg.train.get("rfs_hier_qa_base_cross_lane", False)
        ),
        residual_exploration_std=float(
            cfg.train.get("rfs_hier_residual_exploration_std", 0.0)
        ),
        min_branch_replay_transitions=int(
            cfg.train.rfs_hier_min_branch_replay_transitions
        ),
        noise_gradient_max_norm=float(
            cfg.train.rfs_hier_noise_gradient_max_norm
        ),
        noise_actor_gradient_clipping=bool(
            cfg.train.get("rfs_hier_noise_actor_gradient_clipping", True)
        ),
        residual_gradient_max_norm=float(
            cfg.train.rfs_hier_residual_gradient_max_norm
        ),
        lane_seed=int(cfg.p6.train_env_seed) + 17_071,
        termination_semantics=str(
            cfg.p6.action_chunk_termination_semantics
        ),
        diagnostics_interval_updates=int(
            cfg.p6.diagnostics_interval_updates
        ),
        runtime_contract_checks=bool(
            cfg.train.get("rfs_hier_runtime_contract_checks", True)
        ),
        seed=int(cfg.seed),
    )
    if init_mode == "fresh":
        model.initialize_from_fresh_frozen_ddim()
    else:
        model.initialize_from_legacy_checkpoint(
            hydra.utils.to_absolute_path(str(cfg.rfs_hier_legacy_checkpoint_path))
        )
    return model


def _construct_fresh_control(
    cfg: Any,
    environment: Any,
    diffusion_policy: Any,
) -> P6ControlDSRL:
    """Matched flat-DSRL control initialized from Frozen DDIM only.

    Mirrors the fresh hierarchy's base branch exactly: the same seeded
    DSRL constructors (same seed, same policy kwargs) build the noise actor /
    QA_base / QW_base, so a fresh hierarchy and its fresh control start from
    bit-identical base weights.  The QA_base target is hard-copied from the
    online QA_base (SB3's target is not a copy at construction), optimizers are
    reset, and the Frozen DDIM is placed in inference mode -- the flat analogue
    of ``HierarchicalRFSDSRL.initialize_from_fresh_frozen_ddim`` with no
    residual and no QA_joint.
    """

    model = P6ControlDSRL(
        "MlpPolicy",
        environment,
        learning_rate=float(cfg.train.actor_lr),
        buffer_size=int(cfg.train.buffer_size_na),
        learning_starts=1,
        batch_size=int(cfg.train.batch_size),
        tau=float(cfg.train.tau),
        gamma=float(cfg.train.discount),
        train_freq=int(cfg.train.train_freq),
        gradient_steps=int(cfg.train.utd),
        action_noise=None,
        optimize_memory_usage=False,
        ent_coef="auto" if cfg.train.ent_coef == -1 else cfg.train.ent_coef,
        target_update_interval=1,
        target_entropy=(
            "auto" if cfg.train.target_ent == -1 else cfg.train.target_ent
        ),
        use_sde=False,
        sde_sample_freq=-1,
        tensorboard_log=cfg.logdir,
        verbose=1,
        device=cfg.device,
        policy_kwargs=_make_policy_kwargs(cfg),
        diffusion_policy=diffusion_policy,
        diffusion_act_dim=(int(cfg.act_steps), int(cfg.action_dim)),
        noise_critic_grad_steps=int(cfg.train.noise_critic_grad_steps),
        critic_backup_combine_type=str(cfg.train.critic_backup_combine_type),
        seed=int(cfg.seed),
    )
    # Flat DSRL names the QA_base target critic_target; hard-copy the online
    # critic (QA_base) into it, exactly as the hierarchy's fresh init copies
    # qa_base -> qa_base_target.
    model.critic_target.load_state_dict(
        model.critic.state_dict(), strict=True
    )
    _reset_control_optimizers(model)
    return model


def _verify_real_zero_residual_parity(
    *,
    hierarchy: HierarchicalRFSDSRL,
    legacy: _LegacyLoadableDSRL,
    environment: Any,
    seed: int,
) -> dict[str, float]:
    _set_model_inference_mode(legacy)
    hierarchy.set_inference_mode()
    migration_errors = {
        "actor_state_max_abs_error": _state_dict_max_abs_error(
            hierarchy.actor.state_dict(),
            legacy.actor.state_dict(),
            module_name="hierarchy_actor",
        ),
        "reference_actor_state_max_abs_error": _state_dict_max_abs_error(
            hierarchy.reference_noise_actor.state_dict(),
            legacy.actor.state_dict(),
            module_name="hierarchy_reference_actor",
        ),
        "qa_base_state_max_abs_error": _state_dict_max_abs_error(
            hierarchy.qa_base.state_dict(),
            legacy.critic.state_dict(),
            module_name="hierarchy_qa_base",
        ),
        "qa_base_target_state_max_abs_error": _state_dict_max_abs_error(
            hierarchy.qa_base_target.state_dict(),
            legacy.critic.state_dict(),
            module_name="hierarchy_qa_base_target",
        ),
        "qw_base_state_max_abs_error": _state_dict_max_abs_error(
            hierarchy.qw_base.state_dict(),
            legacy.critic_noise.state_dict(),
            module_name="hierarchy_qw_base",
        ),
        "qa_joint_from_base_max_abs_error": _state_dict_max_abs_error(
            hierarchy.qa_joint.state_dict(),
            hierarchy.qa_base.state_dict(),
            module_name="hierarchy_qa_joint",
        ),
        "qa_joint_target_max_abs_error": _state_dict_max_abs_error(
            hierarchy.qa_joint_target.state_dict(),
            hierarchy.qa_joint.state_dict(),
            module_name="hierarchy_qa_joint_target",
        ),
        "residual_target_state_max_abs_error": _state_dict_max_abs_error(
            hierarchy.residual_actor_target.state_dict(),
            hierarchy.residual_actor.state_dict(),
            module_name="hierarchy_residual_target",
        ),
    }
    if float(hierarchy.target_entropy) != float(legacy.target_entropy):
        raise ValueError("Hierarchy target_entropy differs from legacy checkpoint")
    if hierarchy.log_ent_coef is None or legacy.log_ent_coef is None:
        if hierarchy.log_ent_coef is not None or legacy.log_ent_coef is not None:
            raise ValueError("Hierarchy and legacy entropy modes differ")
        entropy_error = 0.0
    else:
        entropy_error = float(
            torch.max(
                torch.abs(
                    hierarchy.log_ent_coef.detach().cpu()
                    - legacy.log_ent_coef.detach().cpu()
                )
            ).item()
        )
        if entropy_error != 0.0:
            raise ValueError("Hierarchy entropy coefficient migration is not exact")
    migration_errors["entropy_coefficient_max_abs_error"] = entropy_error

    if (
        torch.count_nonzero(hierarchy.residual_actor.output_layer.weight).item()
        or torch.count_nonzero(hierarchy.residual_actor.output_layer.bias).item()
    ):
        raise ValueError("Hierarchy residual output layer is not exactly zero")
    hierarchy_optimizers = (
        hierarchy.actor.optimizer,
        hierarchy.qa_base_optimizer,
        hierarchy.qw_base_optimizer,
        hierarchy.qa_joint_optimizer,
        hierarchy.residual_actor_optimizer,
        hierarchy.ent_coef_optimizer,
    )
    if any(
        optimizer is not None and len(optimizer.state) != 0
        for optimizer in hierarchy_optimizers
    ):
        raise ValueError("Hierarchy network warm-start must use fresh optimizers")
    if hierarchy.num_timesteps != 0 or hierarchy._n_updates != 0:
        raise ValueError("Hierarchy network warm-start counters must begin at zero")
    if any(
        value != 0 for value in hierarchy.requested_optimizer_steps.values()
    ):
        raise ValueError("Hierarchy requested optimizer counters must begin at zero")
    for name in (
        "qa_base_optimizer_steps",
        "qa_joint_optimizer_steps",
        "qw_base_optimizer_steps",
        "noise_actor_optimizer_steps",
        "alpha_optimizer_steps",
        "residual_actor_optimizer_steps",
        "qa_base_target_updates",
        "qa_joint_target_updates",
        "residual_target_updates",
        "hierarchy_train_calls",
        "base_block_skips",
        "joint_block_skips",
        "noise_policy_version",
        "residual_policy_version",
        "qa_joint_optimizer_steps_since_clone",
    ):
        if int(getattr(hierarchy, name)) != 0:
            raise ValueError(f"Hierarchy counter {name} must begin at zero")
    if hierarchy.qa_joint_generation != 1:
        raise ValueError("Legacy warm-start must create exactly one QA_joint generation")

    actor_decoder_error = 0.0
    action_error = 0.0
    qw_error = 0.0
    residual_error = 0.0
    with isolated_rng(seed), torch.no_grad():
        environment.seed(seed)
        observations = environment.reset()
        base_observation = np.asarray(observations, dtype=np.float32)[:1]
        batch_sizes = sorted({1, len(observations), 17})
        for batch_size in batch_sizes:
            observation_batch = np.repeat(
                base_observation,
                batch_size,
                axis=0,
            )
            legacy_decoder_numpy, _ = legacy.policy.predict(
                observation_batch,
                deterministic=True,
            )
            hierarchy_decoder_numpy, _ = hierarchy.policy.predict(
                observation_batch,
                deterministic=True,
            )
            actor_decoder_error = max(
                actor_decoder_error,
                float(
                    np.max(
                        np.abs(
                            np.asarray(legacy_decoder_numpy)
                            - np.asarray(hierarchy_decoder_numpy)
                        )
                    )
                ),
            )
            noise_scaled_numpy = hierarchy.policy.scale_action(
                np.asarray(hierarchy_decoder_numpy, dtype=np.float32)
            )
            noise_scaled = torch.as_tensor(
                noise_scaled_numpy,
                device=hierarchy.device,
                dtype=torch.float32,
            )
            observations_tensor = torch.as_tensor(
                observation_batch,
                device=hierarchy.device,
                dtype=torch.float32,
            )
            decoder_input_numpy = legacy.policy.unscale_action(
                noise_scaled_numpy
            )
            decoder_input = torch.as_tensor(
                decoder_input_numpy,
                device=hierarchy.device,
                dtype=torch.float32,
            ).reshape(
                -1,
                hierarchy.diffusion_act_chunk,
                hierarchy.diffusion_act_dim,
            )
            legacy_action = legacy.diffusion_policy(
                observations_tensor,
                decoder_input,
                return_numpy=False,
            ).reshape(-1, hierarchy.action_dim_flat)
            generated = hierarchy._generate_hierarchical_action(
                observations_tensor,
                noise_scaled,
                zero_residual=True,
            )
            action_error = max(
                action_error,
                float(
                    torch.max(
                        torch.abs(generated.action_exec - legacy_action)
                    ).item()
                ),
            )
            residual_error = max(
                residual_error,
                float(torch.max(torch.abs(generated.residual_unit)).item()),
            )
            legacy_qw = legacy.critic_noise(observations_tensor, noise_scaled)
            hierarchy_qw = hierarchy.qw_base(observations_tensor, noise_scaled)
            qw_error = max(
                qw_error,
                *(
                    float(torch.max(torch.abs(actual - expected)).item())
                    for actual, expected in zip(hierarchy_qw, legacy_qw)
                ),
            )
    if action_error > 1e-6 or qw_error > 1e-6 or residual_error != 0.0:
        raise ValueError(
            "Real checkpoint migration parity failed: "
            f"action={action_error}, QW={qw_error}, residual={residual_error}"
        )
    if actor_decoder_error > 1e-6:
        raise ValueError(
            "Real checkpoint actor decoder-input parity failed: "
            f"{actor_decoder_error}"
        )
    hierarchy.set_inference_mode()
    return {
        **migration_errors,
        "actor_decoder_input_max_abs_error": actor_decoder_error,
        "zero_residual_action_max_abs_error": action_error,
        "qw_base_max_abs_error": qw_error,
        "residual_unit_max_abs": residual_error,
    }


def _read_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as input_file:
        return json.load(input_file)


def _update_manifest(path: Path, updates: Mapping[str, Any]) -> dict[str, Any]:
    manifest = _read_manifest(path)
    manifest.update(dict(updates))
    atomic_write_json(path, manifest)
    return manifest


_ACTIVE_MARKERS = ("RUNNING", "INTERRUPTED", "FAILED", "COMPLETE")


def _write_marker(path: Path, text: str) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as output:
        output.write(text)
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary_path, path)


def _clear_active_markers(run_directory: Path) -> None:
    for marker_name in _ACTIVE_MARKERS:
        marker = run_directory / marker_name
        if marker.exists():
            marker.unlink()


def _start_attempt(
    *,
    cfg: Any,
    manifest_path: Path,
    run_directory: Path,
    start_chunk: int,
    resume_source: Path | None,
) -> int:
    manifest = _read_manifest(manifest_path)
    attempts = list(manifest.get("attempts", []))
    attempt_id = len(attempts)
    attempts_directory = run_directory / "attempts"
    attempts_directory.mkdir(parents=True, exist_ok=True)
    attempt_directory = attempts_directory / f"{attempt_id:04d}"
    if attempt_directory.exists():
        raise FileExistsError(
            "Attempt directory exists without matching manifest history: "
            f"{attempt_directory}"
        )
    temporary_attempt_directory = (
        attempts_directory / f".tmp_{attempt_id:04d}_{os.getpid()}"
    )
    temporary_attempt_directory.mkdir(parents=False, exist_ok=False)
    temporary_config_path = (
        temporary_attempt_directory / "resolved_config.json"
    )
    atomic_write_json(
        temporary_config_path,
        OmegaConf.to_container(cfg, resolve=True),
    )
    overrides = (
        list(HydraConfig.get().overrides.task)
        if HydraConfig.initialized()
        else []
    )
    atomic_write_json(
        temporary_attempt_directory / "task_overrides.json",
        {"overrides": overrides},
    )
    try:
        os.replace(temporary_attempt_directory, attempt_directory)
    except BaseException:
        shutil.rmtree(temporary_attempt_directory, ignore_errors=True)
        raise
    config_path = attempt_directory / "resolved_config.json"
    requested_stop = (
        None
        if cfg.p6.stop_after_chunk_transitions is None
        else int(cfg.p6.stop_after_chunk_transitions)
    )
    attempt = {
        "attempt_id": attempt_id,
        "kind": "fresh" if resume_source is None else "resume",
        "start_chunk": int(start_chunk),
        "end_chunk": None,
        "requested_stop_after_chunk": requested_stop,
        "resume_source": None if resume_source is None else str(resume_source),
        "status": "running",
        "reason": None,
        "resolved_config_path": str(config_path.relative_to(run_directory)),
        "resolved_config_sha256": sha256_file(config_path),
        "task_overrides_path": str(
            (attempt_directory / "task_overrides.json").relative_to(run_directory)
        ),
    }
    if os.environ.get(_SOURCE_STATE_BOUNDARY_RESUME_ENV) == "1":
        attempt["source_state_boundary_resume"] = True
    attempts.append(attempt)
    for stale_key in (
        "interruption_reason",
        "stop_after_chunk_transitions",
        "failure_type",
        "failure_message",
        "failure_traceback",
    ):
        manifest.pop(stale_key, None)
    manifest.update(
        {
            "attempts": attempts,
            "current_attempt_id": attempt_id,
            "run_status": "running",
            "active_stop_after_chunk_transitions": requested_stop,
            "remaining_chunk_transitions": (
                int(cfg.total_timesteps) - int(start_chunk)
            ),
            "final_evaluation_status": manifest.get(
                "final_evaluation_status",
                "pending",
            ),
        }
    )
    atomic_write_json(manifest_path, manifest)
    _clear_active_markers(run_directory)
    _write_marker(run_directory / "RUNNING", f"attempt={attempt_id}\n")
    return attempt_id


def _finish_attempt(
    *,
    manifest_path: Path,
    run_directory: Path,
    status: str,
    end_chunk: int,
    reason: str | None = None,
    traceback_path: str | None = None,
    root_updates: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    manifest = _read_manifest(manifest_path)
    attempt_id = int(manifest["current_attempt_id"])
    attempts = list(manifest["attempts"])
    attempt = dict(attempts[attempt_id])
    attempt.update(
        {
            "end_chunk": int(end_chunk),
            "status": status,
            "reason": reason,
        }
    )
    if traceback_path is not None:
        attempt["traceback_path"] = str(traceback_path)
        attempt["traceback_sha256"] = sha256_file(
            run_directory / traceback_path
        )
    attempts[attempt_id] = attempt
    manifest["attempts"] = attempts
    if root_updates is not None:
        manifest.update(dict(root_updates))
    atomic_write_json(manifest_path, manifest)
    attempt_directory = run_directory / "attempts" / f"{attempt_id:04d}"
    _write_marker(
        attempt_directory / status.upper(),
        "" if reason is None else reason + "\n",
    )
    return manifest


def _write_attempt_traceback(
    *,
    manifest_path: Path,
    run_directory: Path,
    traceback_text: str,
) -> str:
    """Persist the full exception context in the active attempt directory."""

    manifest = _read_manifest(manifest_path)
    attempt_id = int(manifest["current_attempt_id"])
    traceback_path = (
        run_directory / "attempts" / f"{attempt_id:04d}" / "traceback.txt"
    )
    _write_marker(traceback_path, traceback_text)
    return str(traceback_path.relative_to(run_directory))


# Opt-in escape hatch for resuming a bundle across a source-state boundary
# (the recorded config_contract_sha256 / source_state_sha256 were computed by
# a different working tree).  Every other resume check still applies.  The
# boundary crossing is printed to the run log and marked on the attempt.
_SOURCE_STATE_BOUNDARY_RESUME_ENV = "P6_ALLOW_SOURCE_STATE_BOUNDARY_RESUME"


def _verify_existing_manifest(
    *,
    path: Path,
    static_manifest: Mapping[str, Any],
    cfg: Any,
    environment: Any,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError("Resume requested but run_manifest.json is missing")
    manifest = _read_manifest(path)
    if manifest.get("run_status") == "complete":
        raise ValueError("Completed P6 runs cannot be resumed")
    allow_source_state_boundary = (
        os.environ.get(_SOURCE_STATE_BOUNDARY_RESUME_ENV) == "1"
    )
    for key in ("config_contract_sha256", "source_state_sha256"):
        if manifest.get(key) == static_manifest.get(key):
            continue
        if not allow_source_state_boundary:
            raise ValueError(f"Resume run manifest mismatch for {key}")
        print(
            f"[p6] AUTHORIZED source-state boundary resume for {key}: "
            f"recorded={manifest.get(key)} current={static_manifest.get(key)} "
            f"(enabled by {_SOURCE_STATE_BOUNDARY_RESUME_ENV}=1)",
            flush=True,
        )
    for key in (
        "algorithm",
        "run_name",
        "init_checkpoint_sha256",
        "frozen_ddim_sha256",
        "normalization_sha256",
        "chunk_budget",
        "action_chunk_termination_semantics",
    ):
        if manifest.get(key) != static_manifest.get(key):
            raise ValueError(f"Resume run manifest mismatch for {key}")
    execution_contract = validate_execution_bounds(cfg, environment)
    for key, value in execution_contract.items():
        if manifest.get(key) != value:
            raise ValueError(f"Resume execution contract mismatch for {key}")
    # Observation-dimension contract.  Old manifests (written before this check
    # existed) do not carry the key, so only enforce when the run recorded it —
    # a resume must not fail on a legacy manifest that predates the check.
    obs_contract = validate_observation_dimension(cfg, environment)
    for key, value in obs_contract.items():
        if manifest.get(key) is not None and manifest.get(key) != value:
            raise ValueError(f"Resume observation contract mismatch for {key}")
    return manifest


def _require_latest_resume_bundle(
    manifest: Mapping[str, Any],
    requested_bundle: Path,
) -> None:
    latest_value = manifest.get("latest_resume_bundle")
    latest_chunk = manifest.get("latest_resume_bundle_chunk")
    if latest_value is None or latest_chunk is None:
        raise ValueError("Run manifest does not identify a latest resume bundle")
    latest_bundle = Path(str(latest_value)).expanduser().resolve()
    requested_bundle = requested_bundle.expanduser().resolve()
    if requested_bundle != latest_bundle:
        raise ValueError(
            "Rollback from a non-latest resume bundle is not allowed: "
            f"requested={requested_bundle}, latest={latest_bundle}"
        )


def _algorithm_label(cfg: Any) -> str:
    if cfg.algorithm == HIERARCHY_ALGORITHM:
        return HIERARCHY_ALGORITHM
    if cfg.algorithm == "dsrl_na":
        return CONTROL_ALGORITHM
    raise ValueError(
        "P6 runner only accepts algorithm=dsrl_na_rfs_hier or dsrl_na; "
        "the deprecated frozen-noise diagnostic graph is not Core V1"
    )


def _network_warmstart_label(cfg: Any, algorithm: str) -> bool:
    """The recorded network-warm-start label for a fresh launch.

    A run is network-warm-started only when it built its network from a
    legacy checkpoint; a fresh run (no legacy checkpoint) records False.  The
    label is purely informational -- the preflight step checks are driven by
    ``legacy_loaded`` (see ``finalize_loaded_model_preflight``), never by this
    flag.  This is the same decision that picks the init branch, so the label
    can never drift from the branch that actually ran.
    """
    return not _run_is_fresh(cfg, algorithm)


def _remaining_training_chunks(current_chunks: int, target_chunks: int) -> int:
    current_chunks = int(current_chunks)
    target_chunks = int(target_chunks)
    if current_chunks > target_chunks:
        raise ValueError(
            f"Training exceeded target budget: target={target_chunks}, "
            f"current={current_chunks}"
        )
    return target_chunks - current_chunks


def _reset_resumed_environment(
    *,
    model: Any,
    environment: Any,
    algorithm: str,
    train_env_seed: int,
    discontinuity_count: int,
) -> dict[str, Any]:
    """Apply the certified reset-boundary resume contract.

    Replay, learner state and policy RNG are restored, but an old simulator
    episode is never continued.  New lane/episode metadata is allocated only
    after every environment slot has consumed its SHA-derived reset seed.
    """

    abandoned_episode_ids: list[int] = []
    if algorithm == HIERARCHY_ALGORITHM:
        active = np.asarray(model._active_episode_id, dtype=np.int64)
        abandoned_episode_ids = sorted(
            {int(value) for value in active if int(value) >= 0}
        )
        expected_before = int(discontinuity_count) - 1
        if int(model.environment_discontinuity_count) != expected_before:
            raise ValueError(
                "Model/runtime environment discontinuity counters differ before reset"
            )
        reset_seeds = model.prepare_reset_boundary_resume(
            train_env_seed=int(train_env_seed)
        )
        if int(model.environment_discontinuity_count) != int(
            discontinuity_count
        ):
            raise ValueError("Hierarchy discontinuity counter did not advance once")
    else:
        reset_seeds = [
            HierarchicalRFSDSRL.derive_reset_seed(
                int(train_env_seed), int(discontinuity_count), env_id
            )
            for env_id in range(int(environment.num_envs))
        ]

    observation = reset_vec_env_with_explicit_seeds(environment, reset_seeds)
    model._last_obs = observation
    model._last_episode_starts = np.ones(
        int(environment.num_envs), dtype=np.bool_
    )
    if model._vec_normalize_env is not None:
        model._last_original_obs = model._vec_normalize_env.get_original_obs()
    else:
        model._last_original_obs = None

    report: dict[str, Any] = {
        "environment_resume_mode": "reset_boundary_discontinuous",
        "paired_trajectory_continuity": False,
        "reset_seeds": [int(value) for value in reset_seeds],
        "abandoned_episode_ids": abandoned_episode_ids,
        "abandoned_episode_count": len(abandoned_episode_ids),
    }
    if algorithm == HIERARCHY_ALGORITHM:
        model._allocate_unassigned_lanes()
        report.update(
            {
                "new_episode_ids": [
                    int(value) for value in model._active_episode_id
                ],
                "new_branch_modes": [
                    int(value) for value in model._active_branch_mode
                ],
                "new_chunk_indices": [
                    int(value) for value in model._chunk_index_in_episode
                ],
            }
        )
        if any(value != 0 for value in report["new_chunk_indices"]):
            raise RuntimeError("Reset-boundary resume did not restart at chunk zero")
    return report


def _close_resources_safely(
    *,
    writer: Any,
    training_environment: Any,
    wandb_run: Any | None,
    manifest_path: Path,
) -> list[str]:
    """Close all outputs without masking the training/finalization outcome."""

    actions = [
        ("tensorboard_flush", writer.flush),
        ("tensorboard_close", writer.close),
        ("training_environment_close", training_environment.close),
    ]
    if wandb_run is not None:
        actions.append(("wandb_finish", wandb_run.finish))
    errors: list[str] = []
    for name, action in actions:
        try:
            action()
        except BaseException as error:
            message = f"{name}: {type(error).__name__}: {error}"
            errors.append(message)
            print(f"P6 cleanup warning: {message}", file=sys.stderr)
    if errors and manifest_path.is_file():
        try:
            manifest = _read_manifest(manifest_path)
            prior_errors = list(manifest.get("cleanup_errors", []))
            manifest["cleanup_errors"] = prior_errors + errors
            atomic_write_json(manifest_path, manifest)
        except BaseException as error:
            print(
                "P6 cleanup warning: failed to persist cleanup_errors: "
                f"{type(error).__name__}: {error}",
                file=sys.stderr,
            )
    return errors


@hydra.main(
    config_path=str(BASE_PATH / "cfg" / "gym"),
    config_name="p6_hopper",
    version_base=None,
)
def main(cfg: Any) -> None:
    OmegaConf.resolve(cfg)
    algorithm = _algorithm_label(cfg)
    static_manifest = static_preflight(cfg, BASE_PATH, algorithm=algorithm)
    seed_plan = resolve_seed_plan(cfg)
    run_directory = Path(hydra.utils.to_absolute_path(str(cfg.logdir)))
    run_directory.mkdir(parents=True, exist_ok=True)
    manifest_path = run_directory / "run_manifest.json"
    resume_value = cfg.p6.resume_bundle_path
    resume_directory = (
        None
        if resume_value is None
        else Path(hydra.utils.to_absolute_path(str(resume_value)))
    )
    if resume_directory is None and manifest_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing P6 run {run_directory}"
        )

    with open_dict(cfg):
        cfg.model.network_path = static_manifest["frozen_ddim_path"]
        cfg.base_policy_path = static_manifest["frozen_ddim_path"]
        cfg.normalization_path = static_manifest["normalization_path"]
    diffusion_policy = load_base_policy(cfg)
    make_environment = lambda: _make_locomotion_environment(
        cfg,
        Path(static_manifest["normalization_path"]),
    )
    training_environment = make_vec_env(
        make_environment,
        n_envs=int(cfg.env.n_envs),
        vec_env_cls=SubprocVecEnv,
    )
    writer = SummaryWriter(log_dir=str(run_directory / "tensorboard" / "p6"))
    wandb_run = None
    model = None
    matched_dsrl_model = None
    manager = None
    try:
        if cfg.use_wandb:
            import wandb

            wandb_run = wandb.init(
                project=cfg.wandb.project,
                name=cfg.name,
                group=cfg.wandb.group,
                config=OmegaConf.to_container(cfg, resolve=True),
            )

        if resume_directory is None:
            manifest = run_preflight(
                cfg,
                training_environment,
                BASE_PATH,
                manifest_path,
                algorithm=algorithm,
                static_manifest=static_manifest,
            )
            _start_attempt(
                cfg=cfg,
                manifest_path=manifest_path,
                run_directory=run_directory,
                start_chunk=0,
                resume_source=None,
            )
            init_mode = (
                "fresh" if _run_is_fresh(cfg, algorithm) else "warmstart"
            )
            prefill_source = (
                FRESH_PREFILL_SOURCE if init_mode == "fresh" else PREFILL_SOURCE
            )
            warmstart = None
            if algorithm == HIERARCHY_ALGORITHM and init_mode == "fresh":
                # Frozen-DDIM from-scratch init (spec 8.1): no legacy DSRL
                # checkpoint.  The fresh hierarchy itself is the prefill model,
                # which decodes the independent Gaussian decoder prior (5.3).
                model = _construct_hierarchy(
                    cfg,
                    training_environment,
                    diffusion_policy,
                    init_mode="fresh",
                )
                parity = {
                    "fresh_frozen_ddim_init": True,
                    "residual_unit_max_abs": float(
                        torch.count_nonzero(
                            model.residual_actor.output_layer.weight
                        ).item()
                        + torch.count_nonzero(
                            model.residual_actor.output_layer.bias
                        ).item()
                    ),
                    # Artifact-level init proof: canonical state-dict hashes of
                    # the base branch modules.  The matched fresh control must
                    # record the identical hashes (under the same canonical
                    # keys) so a cross-manifest check can prove the two runs
                    # start from bit-identical base weights.
                    "fresh_init_state_hashes": {
                        "actor": canonical_module_state_hash(model.actor),
                        "qa_base": canonical_module_state_hash(model.qa_base),
                        "qa_base_target": canonical_module_state_hash(
                            model.qa_base_target
                        ),
                        "qw_base": canonical_module_state_hash(model.qw_base),
                    },
                }
                warmstart = model
                matched_dsrl_model = None
            elif algorithm == CONTROL_ALGORITHM and init_mode == "fresh":
                # Matched flat-DSRL control from Frozen DDIM only: the same
                # seeded base branch and the identical Gaussian-prior prefill
                # artifact as the fresh hierarchy, so the gate comparison
                # isolates exactly the residual/QA_joint machinery.
                model = _construct_fresh_control(
                    cfg,
                    training_environment,
                    diffusion_policy,
                )
                parity = {
                    "fresh_frozen_ddim_init": True,
                    "qa_base_target_copied": True,
                    "zero_residual_action_max_abs_error": 0.0,
                    "qw_base_max_abs_error": 0.0,
                    "residual_unit_max_abs": 0.0,
                    # Flat DSRL names the QA_base/QW_base modules critic /
                    # critic_noise; record them under the hierarchy's canonical
                    # keys so a fresh hierarchy manifest and a fresh control
                    # manifest are directly comparable.  The control carries no
                    # residual network, so the hash set is the same four base
                    # modules the hierarchy records.
                    "fresh_init_state_hashes": {
                        "actor": canonical_module_state_hash(model.actor),
                        "qa_base": canonical_module_state_hash(model.critic),
                        "qa_base_target": canonical_module_state_hash(
                            model.critic_target
                        ),
                        "qw_base": canonical_module_state_hash(
                            model.critic_noise
                        ),
                    },
                }
                warmstart = model
                matched_dsrl_model = None
            else:
                warmstart = _load_legacy_network(
                    cfg=cfg,
                    environment=training_environment,
                    diffusion_policy=diffusion_policy,
                    buffer_size=(
                        int(cfg.train.buffer_size_na)
                        if algorithm == CONTROL_ALGORITHM
                        else 1
                    ),
                )
                if algorithm == CONTROL_ALGORITHM:
                    legacy_snapshot = _capture_legacy_warmstart(warmstart)
                    model = _promote_legacy_control(warmstart)
                    _reset_control_optimizers(model)
                    control_migration = _verify_fresh_control_warmstart(
                        model,
                        legacy_snapshot,
                    )
                    parity = {
                        **control_migration,
                        "zero_residual_action_max_abs_error": 0.0,
                        "qw_base_max_abs_error": 0.0,
                        "residual_unit_max_abs": 0.0,
                    }
                    matched_dsrl_model = _load_legacy_network(
                        cfg=cfg,
                        environment=training_environment,
                        diffusion_policy=diffusion_policy,
                        buffer_size=1,
                    )
                else:
                    model = _construct_hierarchy(
                        cfg,
                        training_environment,
                        diffusion_policy,
                        init_mode="warmstart",
                    )
                    parity = _verify_real_zero_residual_parity(
                        hierarchy=model,
                        legacy=warmstart,
                        environment=training_environment,
                        seed=int(cfg.seed) + 50_000,
                    )
                    matched_dsrl_model = warmstart
            validate_optimizer_counter_invariants(model, algorithm)
            finalize_loaded_model_preflight(
                cfg,
                training_environment,
                model,
                manifest_path,
                # A pure recorded label: fresh (no-legacy-checkpoint) runs are
                # NOT network-warm-starts; only runs that built the network
                # from a legacy checkpoint are.  Every branch here constructs
                # the model with counters at zero, so legacy_loaded stays False
                # and the zero-step check applies.  The label shares the
                # init-mode decision (_run_is_fresh), so it cannot drift.
                network_warmstart=_network_warmstart_label(cfg, algorithm),
            )

            prefill_path = Path(manifest["prefill_artifact_path"])
            prefill_arrays, prefill_metadata, generated = (
                collect_or_load_tagged_matched_prefill(
                    artifact_path=prefill_path,
                    warmstart_model=warmstart,
                    env=training_environment,
                    vector_steps=int(cfg.train.init_rollout_steps),
                    environment_seed=int(seed_plan["prefill_env_seed"]),
                    policy_seed=int(seed_plan["prefill_policy_seed"]),
                    action_chunk=int(cfg.act_steps),
                    termination_semantics=str(
                        cfg.p6.action_chunk_termination_semantics
                    ),
                    provenance=static_manifest,
                    prefill_source=prefill_source,
                )
            )
            standard_projection = tagged_prefill_standard_projection(
                prefill_arrays
            )
            if algorithm == HIERARCHY_ALGORITHM:
                replay_hash = populate_tagged_replay_buffer(
                    model.replay_buffer,
                    prefill_arrays,
                    prefill_metadata,
                )
            else:
                replay_hash = populate_replay_buffer(
                    model.replay_buffer,
                    standard_projection,
                )
            expected_capacity = (
                int(prefill_metadata["chunk_transitions"])
                + int(cfg.total_timesteps)
            )
            if int(cfg.train.buffer_size_na) <= expected_capacity:
                raise ValueError(
                    "Replay buffer must exceed prefill + online chunk transitions: "
                    f"{cfg.train.buffer_size_na} <= {expected_capacity}"
                )
            manifest = _update_manifest(
                manifest_path,
                {
                    "prefill_status": "verified_and_loaded",
                    "prefill_generated_by_this_run": generated,
                    "prefill_hash": prefill_metadata["semantic_hash"],
                    "prefill_semantic_hash": prefill_metadata[
                        "semantic_hash"
                    ],
                    "prefill_per_array_hashes": prefill_metadata[
                        "per_array_hashes"
                    ],
                    "prefill_projection_semantic_hash": prefill_metadata[
                        "projection_semantic_hash"
                    ],
                    "prefill_projection_per_array_hashes": prefill_metadata[
                        "projection_per_array_hashes"
                    ],
                    "prefill_archive_sha256": prefill_metadata["archive_sha256"],
                    "prefill_action_policy": (
                        PREFILL_ACTION_POLICY
                        if prefill_source == PREFILL_SOURCE
                        else FRESH_PREFILL_ACTION_POLICY
                    ),
                    "hierarchy_prefill_residual_mode": (
                        HIERARCHY_PREFILL_RESIDUAL_MODE
                    ),
                    "prefill_primitive_counters": prefill_metadata[
                        "primitive_counters"
                    ],
                    "initial_replay_hash": replay_hash,
                    "zero_residual_ddim_parity": parity,
                    "training_status": "ready",
                },
            )
            runtime_state = initial_runtime_state(
                target_chunk_budget=int(cfg.total_timesteps),
                action_chunk=int(cfg.act_steps),
                prefill_metadata=prefill_metadata,
                replay_hash=replay_hash,
                online_eval_interval=int(
                    cfg.p6.online_eval_interval_chunk_transitions
                ),
                model_checkpoint_interval=int(
                    cfg.p6.model_checkpoint_interval_chunk_transitions
                ),
                replay_checkpoint_interval=int(
                    cfg.p6.replay_checkpoint_interval_chunk_transitions
                ),
                run_binding=run_binding_from_manifest(manifest),
            )
            seed_all(int(cfg.seed))
            training_environment.seed(int(seed_plan["train_env_seed"]))
            model._last_obs = None
        else:
            existing_manifest = _verify_existing_manifest(
                path=manifest_path,
                static_manifest=static_manifest,
                cfg=cfg,
                environment=training_environment,
            )
            _require_latest_resume_bundle(
                existing_manifest,
                resume_directory,
            )
            bundle_manifest, runtime_payload = load_resume_payload(
                resume_directory,
                algorithm=algorithm,
                expected_run_manifest=existing_manifest,
            )
            _start_attempt(
                cfg=cfg,
                manifest_path=manifest_path,
                run_directory=run_directory,
                start_chunk=int(bundle_manifest["chunk_transitions"]),
                resume_source=resume_directory,
            )
            model_path = resume_directory / "model.zip"
            if algorithm in HIERARCHY_ALGORITHMS:
                model = HierarchicalRFSDSRL.load(
                    model_path,
                    env=training_environment,
                    device=cfg.device,
                    diffusion_policy=diffusion_policy,
                )
            else:
                model = P6ControlDSRL.load(
                    model_path,
                    env=training_environment,
                    device=cfg.device,
                    custom_objects={"diffusion_policy": diffusion_policy},
                )
            if _run_is_fresh(cfg, algorithm):
                # Fresh runs (fresh hierarchy or fresh matched control) have
                # no matched legacy DSRL checkpoint.
                matched_dsrl_model = None
            else:
                matched_dsrl_model = _load_legacy_network(
                    cfg=cfg,
                    environment=training_environment,
                    diffusion_policy=diffusion_policy,
                    buffer_size=1,
                )
            _set_model_inference_mode(model)
            model.load_replay_buffer(resume_directory / "replay_buffer.pkl")
            validate_loaded_resume(
                model=model,
                bundle_directory=resume_directory,
                bundle_manifest=bundle_manifest,
                runtime_payload=runtime_payload,
            )
            runtime_state = runtime_payload["runtime_state"]
            runtime_state["resume_count"] = int(runtime_state["resume_count"]) + 1
            runtime_state["environment_reset_discontinuities"] = (
                int(runtime_state["environment_reset_discontinuities"]) + 1
            )
            restore_rng_state(runtime_payload["rng_state"])
            reset_report = _reset_resumed_environment(
                model=model,
                environment=training_environment,
                algorithm=algorithm,
                train_env_seed=int(seed_plan["train_env_seed"]),
                discontinuity_count=int(
                    runtime_state["environment_reset_discontinuities"]
                ),
            )
            _update_manifest(
                manifest_path,
                {
                    "training_status": "resuming",
                    "run_status": "running",
                    "resume_source": str(resume_directory),
                    "resume_count": runtime_state["resume_count"],
                    "environment_reset_discontinuities": runtime_state[
                        "environment_reset_discontinuities"
                    ],
                    "latest_reset_boundary": reset_report,
                },
            )

        if matched_dsrl_model is None and not _run_is_fresh(cfg, algorithm):
            raise RuntimeError("Authenticated matched DSRL evaluator is missing")

        def evaluate_modes(
            *,
            chunk: int,
            nominal_primitive_steps: int,
            actual_primitive_env_steps: int,
            episode_count: int,
            artifact_label: str,
        ) -> dict[str, Any]:
            modes = (
                (
                    (
                        "current_base_only",
                        "current_full_hierarchy",
                        "reference_base",
                    )
                    + (
                        ("matched_dsrl",)
                        if matched_dsrl_model is not None
                        else ()
                    )
                )
                if algorithm == HIERARCHY_ALGORITHM
                else ("current_base_only",)
            )
            seeds = seed_plan["eval_seed_set"][:episode_count]
            results: dict[str, Any] = {}
            for evaluation_mode in modes:
                result = evaluate_exact_episodes(
                    model=model,
                    make_environment=make_environment,
                    environment_seeds=seeds,
                    policy_seed_start=int(cfg.p6.eval_policy_seed_start),
                    deterministic=bool(cfg.deterministic_eval),
                    action_chunk=int(cfg.act_steps),
                    max_episode_primitive_steps=int(cfg.env.max_episode_steps),
                    batch_size=int(cfg.p6.evaluation_batch_size),
                    chunk_transitions=chunk,
                    nominal_primitive_steps=nominal_primitive_steps,
                    actual_primitive_env_steps=actual_primitive_env_steps,
                    evaluation_mode=evaluation_mode,
                    matched_dsrl_model=(
                        matched_dsrl_model
                        if evaluation_mode == "matched_dsrl"
                        else None
                    ),
                )
                persist_evaluation(
                    result,
                    output_prefix=(
                        run_directory
                        / "evaluations"
                        / f"{artifact_label}_{evaluation_mode}"
                    ),
                    tensorboard_writer=writer,
                    tensorboard_tag=f"eval/{artifact_label}/{evaluation_mode}",
                )
                results[evaluation_mode] = result
                if wandb_run is not None:
                    wandb_run.log(
                        {
                            f"eval/{artifact_label}/{evaluation_mode}/{key}": value
                            for key, value in result["summary"].items()
                        },
                        step=chunk,
                    )
            primary_mode = (
                "current_full_hierarchy"
                if algorithm == HIERARCHY_ALGORITHM
                else "current_base_only"
            )
            return {
                "summary": results[primary_mode]["summary"],
                "primary_mode": primary_mode,
                "mode_summaries": {
                    mode: value["summary"] for mode, value in results.items()
                },
                "results": results,
            }

        def evaluate_online(chunk: int, counters: Any) -> Mapping[str, Any]:
            return evaluate_modes(
                chunk=chunk,
                nominal_primitive_steps=counters.nominal_primitive_steps,
                actual_primitive_env_steps=counters.actual_primitive_env_steps,
                episode_count=int(cfg.p6.online_eval_episodes),
                artifact_label=f"online_{chunk:012d}",
            )

        manager = P6CheckpointManager(
            run_directory=run_directory,
            algorithm=algorithm,
            manifest_path=manifest_path,
            runtime_state=runtime_state,
            provenance=_read_manifest(manifest_path),
            evaluation_function=evaluate_online,
        )
        callback = P6TrainingCallback(
            manager=manager,
            action_chunk=int(cfg.act_steps),
            stop_after_chunk_transitions=(
                None
                if cfg.p6.stop_after_chunk_transitions is None
                else int(cfg.p6.stop_after_chunk_transitions)
            ),
        )
        current_chunks = int(model.num_timesteps)
        remaining_chunks = _remaining_training_chunks(
            current_chunks,
            int(cfg.total_timesteps),
        )
        _update_manifest(
            manifest_path,
            {
                "training_status": "running",
                "run_status": "running",
                "training_start_chunk_transitions": current_chunks,
                "remaining_chunk_transitions": remaining_chunks,
            },
        )
        if remaining_chunks > 0:
            model.learn(
                total_timesteps=remaining_chunks,
                callback=callback,
                reset_num_timesteps=False,
                tb_log_name="p6_train",
            )
        if int(model.num_timesteps) != int(cfg.total_timesteps):
            raise RuntimeError(
                f"Training ended at {model.num_timesteps}, expected "
                f"{cfg.total_timesteps}"
            )
        validate_optimizer_counter_invariants(model, algorithm)
        _set_model_inference_mode(model)
        _update_manifest(
            manifest_path,
            {
                "training_status": "complete",
                "final_evaluation_status": "running",
                "run_status": "finalizing",
                "remaining_chunk_transitions": 0,
                "active_stop_after_chunk_transitions": None,
            },
        )

        final_episode_count = int(cfg.p6.final_eval_episodes)
        final_counters = manager.runtime_state["training_counters"]
        final_result = evaluate_modes(
            chunk=int(model.num_timesteps),
            nominal_primitive_steps=int(final_counters["nominal_primitive_steps"]),
            actual_primitive_env_steps=int(
                final_counters["actual_primitive_env_steps"]
            ),
            episode_count=final_episode_count,
            artifact_label="final",
        )
        _finish_attempt(
            manifest_path=manifest_path,
            run_directory=run_directory,
            status="complete",
            end_chunk=int(model.num_timesteps),
            root_updates={
                "training_status": "complete",
                "final_evaluation_status": "complete",
                "run_status": "complete",
                "remaining_chunk_transitions": 0,
                "active_stop_after_chunk_transitions": None,
                "final_evaluation_summary": final_result["summary"],
                "final_evaluation_mode_summaries": final_result[
                    "mode_summaries"
                ],
                "final_evaluation_episode_count": final_episode_count,
            },
        )
        _clear_active_markers(run_directory)
        _write_marker(run_directory / "COMPLETE", "complete\n")
    except P6IntentionalInterruption as error:
        _finish_attempt(
            manifest_path=manifest_path,
            run_directory=run_directory,
            status="interrupted",
            end_chunk=int(model.num_timesteps),
            reason=str(error),
            root_updates={
                "training_status": "interrupted",
                "final_evaluation_status": "pending",
                "run_status": "interrupted",
                "remaining_chunk_transitions": (
                    int(cfg.total_timesteps) - int(model.num_timesteps)
                ),
                "active_stop_after_chunk_transitions": None,
            },
        )
        _clear_active_markers(run_directory)
        _write_marker(run_directory / "INTERRUPTED", str(error) + "\n")
        raise SystemExit(75) from error
    except BaseException as error:
        failure_traceback_path = None
        if manifest_path.exists():
            current_chunks = int(getattr(model, "num_timesteps", 0))
            training_complete = current_chunks == int(cfg.total_timesteps)
            failure_traceback_path = _write_attempt_traceback(
                manifest_path=manifest_path,
                run_directory=run_directory,
                traceback_text=traceback.format_exc(),
            )
            failure_updates = {
                "training_status": (
                    "complete" if training_complete else "failed"
                ),
                "final_evaluation_status": (
                    "failed" if training_complete else "pending"
                ),
                "run_status": (
                    "finalization_failed"
                    if training_complete
                    else "failed"
                ),
                "remaining_chunk_transitions": max(
                    0,
                    int(cfg.total_timesteps) - current_chunks,
                ),
                "active_stop_after_chunk_transitions": None,
            }
            if "current_attempt_id" in _read_manifest(manifest_path):
                _finish_attempt(
                    manifest_path=manifest_path,
                    run_directory=run_directory,
                    status="failed",
                    end_chunk=current_chunks,
                    reason=f"{type(error).__name__}: {error}",
                    traceback_path=failure_traceback_path,
                    root_updates=failure_updates,
                )
            else:
                _update_manifest(manifest_path, failure_updates)
        _clear_active_markers(run_directory)
        _write_marker(
            run_directory / "FAILED",
            f"{type(error).__name__}: {error}\n",
        )
        raise
    finally:
        _close_resources_safely(
            writer=writer,
            training_environment=training_environment,
            wandb_run=wandb_run,
            manifest_path=manifest_path,
        )


if __name__ == "__main__":
    main()
