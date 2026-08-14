"""Strictly value-separated three-critic DSRL hierarchy.

The only actor credit paths implemented here are::

    QA_base -> QW_base -> noise actor
    QA_joint          -> residual actor

The common DSRL implementation and common SB3 replay buffer are intentionally
untouched.  This algorithm owns a tagged replay subclass and rejects old joint
QA/QM hierarchy checkpoints instead of partially loading them.
"""

from __future__ import annotations

import copy
import hashlib
import math
import random
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping, NamedTuple, Optional, Sequence, Union

import numpy as np
import torch as th
from gymnasium import spaces
from torch import nn
from torch.nn import functional as F

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.policies import ContinuousCritic
from stable_baselines3.common.type_aliases import GymEnv, Schedule
from stable_baselines3.common.utils import get_parameters_by_name, polyak_update
from stable_baselines3.dsrl.dsrl import DSRL
from stable_baselines3.dsrl.hierarchical_replay_buffer import (
    BranchMode,
    HierarchyReplayBufferSamples,
    HierarchyTaggedReplayBuffer,
    NoiseSampleSource,
    SCHEMA_VERSION,
    TerminationSemantics,
    TransitionOrigin,
)
from stable_baselines3.dsrl.hierarchy_schedule import (
    HierarchyPhase,
    HierarchySchedule,
    UpdateProfile,
    make_hierarchy_schedule,
)
from stable_baselines3.sac.policies import SACPolicy


ARCHITECTURE_VERSION = "dsrl_na_rfs_hier_three_critic_v1"
DEPRECATED_CHECKPOINT_MESSAGE = (
    "This checkpoint uses the deprecated joint QA/QM hierarchy. "
    "Start three-critic Core V1 from the corresponding legacy DSRL-NA checkpoint."
)


class ActionComposition(NamedTuple):
    residual_unit: th.Tensor
    margin_positive: th.Tensor
    margin_negative: th.Tensor
    action_residual_delta: th.Tensor
    action_exec_unclamped: th.Tensor
    action_exec: th.Tensor
    emergency_clamp_applied: th.Tensor
    maximum_preclamp_violation: th.Tensor


class HierarchicalActionOutput(NamedTuple):
    noise_scaled: th.Tensor
    noise_decoder_input: th.Tensor
    action_base: th.Tensor
    residual_pre_tanh: th.Tensor
    residual_unit: th.Tensor
    margin_positive: th.Tensor
    margin_negative: th.Tensor
    action_residual_delta: th.Tensor
    action_exec_unclamped: th.Tensor
    action_exec: th.Tensor
    emergency_clamp_applied: th.Tensor
    maximum_preclamp_violation: th.Tensor


@contextmanager
def _freeze_module_parameters(module: nn.Module) -> Iterator[None]:
    """Freeze module parameters while retaining gradients to module inputs."""

    parameters = tuple(module.parameters())
    requires_grad = tuple(parameter.requires_grad for parameter in parameters)
    training = module.training
    try:
        for parameter in parameters:
            parameter.requires_grad_(False)
        module.eval()
        yield
    finally:
        for parameter, original in zip(parameters, requires_grad):
            parameter.requires_grad_(original)
        module.train(training)


def _freeze_target(module: nn.Module) -> None:
    module.eval()
    module.requires_grad_(False)


def _fresh_episode_stat_window() -> dict[str, float]:
    """Per-branch closed-episode accumulator (Section 10).

    The window is read-and-reset on each diagnostics dump, so only running
    sums/counts are kept (the mean return/std/length and the early-fall rate
    are derived at read time).
    """
    return {
        "return_sum": 0.0,
        "return_sq_sum": 0.0,
        "length_sum": 0.0,
        "count": 0.0,
        "early_fall_count": 0.0,
    }


def _mean_and_std(values: Sequence[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean = float(np.mean(values))
    variance = float(np.var(values))
    return mean, math.sqrt(max(0.0, variance))


def _rank_array(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = np.arange(values.size, dtype=np.float64)
    return ranks


def _spearman_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation (0.0 when undefined on tiny/tied inputs)."""
    if a.size < 2:
        return 0.0
    ra = _rank_array(a)
    rb = _rank_array(b)
    denominator = math.sqrt(
        float(np.sum((ra - ra.mean()) ** 2)) * float(np.sum((rb - rb.mean()) ** 2))
    )
    if denominator == 0.0:
        return 0.0
    return float(np.sum((ra - ra.mean()) * (rb - rb.mean())) / denominator)


def _kendall_tau(a: np.ndarray, b: np.ndarray) -> float:
    """Kendall rank correlation; ties in either series are skipped."""
    size = a.size
    if size < 2:
        return 0.0
    concordant = 0
    discordant = 0
    for i in range(size):
        for j in range(i + 1, size):
            da = a[i] - a[j]
            db = b[i] - b[j]
            if da == 0.0 or db == 0.0:
                continue
            if (da > 0.0) == (db > 0.0):
                concordant += 1
            else:
                discordant += 1
    denominator = concordant + discordant
    if denominator == 0:
        return 0.0
    return float((concordant - discordant) / denominator)


def _pairwise_preference_accuracy(q: np.ndarray, teacher: np.ndarray) -> float:
    """Fraction of candidate pairs where QW's preference matches the teacher.

    Teacher ties are excluded; a QW tie counts as a missed preference.
    """
    size = q.size
    if size < 2:
        return 0.0
    matches = 0
    total = 0
    for i in range(size):
        for j in range(i + 1, size):
            dt = teacher[i] - teacher[j]
            if dt == 0.0:
                continue
            dq = q[i] - q[j]
            total += 1
            matches += int((dq > 0.0) == (dt > 0.0))
    if total == 0:
        return 0.0
    return float(matches / total)


class _LegacyLoadableDSRL(DSRL):
    """Load-only DSRL shell preserving the inherited official loader."""

    def __init__(
        self,
        *args: Any,
        diffusion_act_dim: Optional[tuple[int, int]] = None,
        _init_setup_model: bool = True,
        buffer_size: int = 1,
        **kwargs: Any,
    ) -> None:
        if diffusion_act_dim is None:
            if _init_setup_model:
                raise ValueError("diffusion_act_dim is required outside the load shell")
            diffusion_act_dim = (1, 1)
        super().__init__(
            *args,
            diffusion_act_dim=diffusion_act_dim,
            _init_setup_model=_init_setup_model,
            buffer_size=buffer_size,
            **kwargs,
        )


def compose_action(
    action_base: th.Tensor,
    residual_pre_tanh: th.Tensor,
    beta: float | th.Tensor,
    exec_action_low: th.Tensor,
    exec_action_high: th.Tensor,
    *,
    numerical_tolerance: float = 1e-6,
) -> ActionComposition:
    """Centered, bound-preserving residual map from the frozen Stage-1 spec."""

    if action_base.shape != residual_pre_tanh.shape or action_base.ndim < 1:
        raise ValueError(
            "action_base and residual_pre_tanh must have the same non-scalar "
            f"shape, got {tuple(action_base.shape)} and "
            f"{tuple(residual_pre_tanh.shape)}"
        )
    if not np.isfinite(numerical_tolerance) or numerical_tolerance < 0:
        raise ValueError("numerical_tolerance must be finite and non-negative")
    action_dim = action_base.shape[-1]
    low = th.as_tensor(
        exec_action_low, device=action_base.device, dtype=action_base.dtype
    )
    high = th.as_tensor(
        exec_action_high, device=action_base.device, dtype=action_base.dtype
    )
    if low.shape != (action_dim,) or high.shape != (action_dim,):
        raise ValueError(
            f"Execution bounds must have shape ({action_dim},), got "
            f"{tuple(low.shape)} and {tuple(high.shape)}"
        )
    if not bool(th.isfinite(low).all() and th.isfinite(high).all()):
        raise ValueError("Execution bounds must be finite")
    if not bool(th.all(high > low)):
        raise ValueError("exec_action_high must exceed exec_action_low elementwise")
    beta_tensor = th.as_tensor(beta, device=action_base.device, dtype=action_base.dtype)
    if beta_tensor.numel() != 1 or not bool(th.isfinite(beta_tensor).all()):
        raise ValueError("beta must be one finite scalar")
    beta_value = float(beta_tensor.detach().cpu().item())
    if not 0 <= beta_value <= 1:
        raise ValueError("beta must lie in [0, 1]")

    base_violation = th.maximum(
        (low - action_base).clamp_min(0),
        (action_base - high).clamp_min(0),
    )
    maximum_base_violation = float(base_violation.detach().max().cpu().item())
    if maximum_base_violation > numerical_tolerance:
        raise ValueError(
            "action_base is materially outside audited execution bounds: "
            f"max violation={maximum_base_violation:.9g}"
        )

    residual_unit = th.tanh(residual_pre_tanh)
    margin_positive = high - action_base
    margin_negative = action_base - low
    action_residual_delta = beta_tensor * (
        0.5 * (margin_positive + margin_negative) * residual_unit
        + 0.5 * (margin_positive - margin_negative) * residual_unit.abs()
    )
    action_exec_unclamped = action_base + action_residual_delta
    violation = th.maximum(
        (low - action_exec_unclamped).clamp_min(0),
        (action_exec_unclamped - high).clamp_min(0),
    )
    maximum_violation = violation.detach().amax()
    if float(maximum_violation.cpu().item()) > numerical_tolerance:
        raise RuntimeError(
            "Bound-preserving composition produced a material violation: "
            f"{float(maximum_violation.cpu().item()):.9g}"
        )
    clamped = th.maximum(th.minimum(action_exec_unclamped, high), low)
    # The composition is algebraically bound-preserving, so the clamp above is
    # value-only roundoff protection (action_exec_unclamped already lies inside
    # [low, high] up to fp roundoff).  Pass gradients through the UNCLAMPED path
    # (detached correction) so the residual loss keeps the spec's full centered
    # subgradient beta*(high-low)/2 at bound-touching transitions; the returned
    # action_exec stays the clamped value that rollout stores and critics see.
    action_exec = action_exec_unclamped + (clamped - action_exec_unclamped).detach()
    emergency = (action_exec != action_exec_unclamped).reshape(
        action_exec.shape[0], -1
    ).any(dim=1)
    return ActionComposition(
        residual_unit=residual_unit,
        margin_positive=margin_positive,
        margin_negative=margin_negative,
        action_residual_delta=action_residual_delta,
        action_exec_unclamped=action_exec_unclamped,
        action_exec=action_exec,
        emergency_clamp_applied=emergency,
        maximum_preclamp_violation=maximum_violation.reshape(1),
    )


class ResidualActor(nn.Module):
    """Deterministic residual conditioned on observation, noise and base action."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        net_arch: Sequence[int] = (128, 128),
        activation: str = "silu",
    ) -> None:
        super().__init__()
        if observation_dim <= 0 or action_dim <= 0:
            raise ValueError("observation_dim and action_dim must be positive")
        if not net_arch or any(int(width) <= 0 for width in net_arch):
            raise ValueError("net_arch must contain positive hidden dimensions")
        if activation.lower() != "silu":
            raise ValueError("Core V1 supports residual_activation='silu' only")
        self.observation_dim = int(observation_dim)
        self.action_dim = int(action_dim)
        self.net_arch = tuple(int(width) for width in net_arch)
        input_dim = self.observation_dim + 2 * self.action_dim
        blocks: list[nn.Module] = []
        for width in self.net_arch:
            blocks.extend(
                [nn.Linear(input_dim, width), nn.LayerNorm(width), nn.SiLU()]
            )
            input_dim = width
        self.hidden_net = nn.Sequential(*blocks)
        self.output_layer = nn.Linear(input_dim, self.action_dim)
        nn.init.zeros_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward_pre_tanh(
        self,
        observation: th.Tensor,
        noise_scaled: th.Tensor,
        action_base: th.Tensor,
    ) -> th.Tensor:
        if observation.ndim != 2 or observation.shape[1] != self.observation_dim:
            raise ValueError("Residual observation has an incompatible shape")
        expected = (observation.shape[0], self.action_dim)
        if noise_scaled.shape != expected or action_base.shape != expected:
            raise ValueError("Residual noise/base tensors have incompatible shapes")
        features = th.cat((observation, noise_scaled, action_base), dim=1)
        return self.output_layer(self.hidden_net(features))

    def forward_with_pre_tanh(
        self,
        observation: th.Tensor,
        noise_scaled: th.Tensor,
        action_base: th.Tensor,
    ) -> tuple[th.Tensor, th.Tensor]:
        logits = self.forward_pre_tanh(observation, noise_scaled, action_base)
        return logits, th.tanh(logits)

    def forward(
        self,
        observation: th.Tensor,
        noise_scaled: th.Tensor,
        action_base: th.Tensor,
    ) -> th.Tensor:
        return th.tanh(self.forward_pre_tanh(observation, noise_scaled, action_base))


class HierarchicalRFSDSRL(DSRL):
    """Three-value-meaning Core V1 implementation."""

    architecture_version = ARCHITECTURE_VERSION
    replay_schema_version = SCHEMA_VERSION

    def __init__(
        self,
        policy: Union[str, type[SACPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        buffer_size: int = 1_000_000,
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, tuple[int, str]] = 1,
        gradient_steps: int = 20,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[type[ReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        ent_coef: Union[str, float] = "auto",
        target_update_interval: int = 1,
        target_entropy: Union[str, float] = "auto",
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        use_sde_at_warmup: bool = False,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        actor_gradient_steps: int = -1,
        diffusion_policy: Any = None,
        diffusion_act_dim: Optional[tuple[int, int]] = None,
        noise_critic_grad_steps: int = 10,
        critic_backup_combine_type: str = "min",
        *,
        exec_action_low: Optional[np.ndarray] = None,
        exec_action_high: Optional[np.ndarray] = None,
        residual_net_arch: Sequence[int] = (128, 128),
        residual_activation: str = "silu",
        residual_lr: float = 3e-4,
        qa_joint_lr: Optional[float] = None,
        schedule_profile: str = "fresh_frozen_ddim_5m",
        phase_b_steps: Optional[int] = None,
        phase_r_steps: Optional[int] = None,
        phase_j_steps: Optional[int] = None,
        phase_j_enabled: Optional[bool] = None,
        beta_ramp_steps: Optional[int] = None,
        beta_target: float = 0.1,
        base_lane_probability: float = 0.5,
        beta_hold_steps: Optional[int] = None,
        beta_floor: Optional[float] = None,
        min_branch_replay_transitions: int = 256,
        qa_joint_shadow_in_b: bool = False,
        cross_lane_ratio: float = 0.0,
        qa_base_cross_lane: bool = False,
        residual_exploration_std: float = 0.0,
        noise_gradient_max_norm: float = 1.0,
        residual_gradient_max_norm: float = 1.0,
        lane_seed: Optional[int] = None,
        termination_semantics: str = "early_break_on_done",
        numerical_bound_tolerance: float = 1e-6,
        diagnostics_interval_updates: int = 100,
        ranking_diag_n_obs: int = 128,
        ranking_diag_seed: Optional[int] = None,
        # Gated V1.1/V1.2 mechanisms.  All default OFF: Core V1 behavior is
        # identical when these are disabled.  The plan requires the gate
        # diagnostics to pass BEFORE these flags may be enabled.
        enable_qw_ranking: bool = False,
        qw_ranking_tau_gap: float = 5.0,
        qw_ranking_lambda: float = 0.1,
        qw_ranking_k_candidates: int = 8,
        qa_joint_target_smoothing: bool = False,
        qa_joint_smoothing_std: float = 0.005,
        qa_joint_smoothing_clip: float = 0.01,
        # Deprecated arguments are accepted only at their Core-V1 neutral value.
        residual_scale: Optional[float] = None,
        residual_penalty_coef: float = 0.0,
        noise_actor_gradient_steps: Optional[int] = None,
        residual_actor_gradient_steps: Optional[int] = None,
    ) -> None:
        if diffusion_act_dim is None:
            if _init_setup_model:
                raise ValueError("diffusion_act_dim is required")
            diffusion_act_dim = (1, 1)
        if len(diffusion_act_dim) != 2 or any(int(v) <= 0 for v in diffusion_act_dim):
            raise ValueError("diffusion_act_dim must contain two positive dimensions")
        action_dim = int(diffusion_act_dim[0]) * int(diffusion_act_dim[1])
        if policy_kwargs and bool(policy_kwargs.get("share_features_extractor", False)):
            raise ValueError(
                "share_features_extractor=True is forbidden: Core V1 requires "
                "disjoint actor/critic representations"
            )
        if action_noise is not None:
            raise ValueError(
                "Core V1 does not support external action_noise because behavior "
                "noise log-prob/version metadata would be ambiguous"
            )
        if optimize_memory_usage:
            raise ValueError("Core V1 tagged replay requires optimize_memory_usage=False")
        if replay_buffer_class is None:
            replay_buffer_class = HierarchyTaggedReplayBuffer
        if not issubclass(replay_buffer_class, HierarchyTaggedReplayBuffer):
            raise ValueError("Core V1 requires HierarchyTaggedReplayBuffer")
        if critic_backup_combine_type != "min":
            raise ValueError("critic_backup_combine_type must be 'min'")
        if actor_gradient_steps != -1:
            raise ValueError("actor_gradient_steps is replaced by the frozen phase profile")
        if residual_penalty_coef != 0:
            raise ValueError("Residual L2 is forbidden in Core V1")
        if noise_actor_gradient_steps is not None or residual_actor_gradient_steps is not None:
            raise ValueError(
                "Legacy fixed actor step arguments are incompatible with the B/R/J profile"
            )
        if residual_scale is not None:
            if not np.isclose(float(residual_scale), float(beta_target), rtol=0, atol=0):
                raise ValueError("deprecated residual_scale must exactly equal beta_target")
        if not np.isfinite(residual_lr) or residual_lr <= 0:
            raise ValueError("residual_lr must be positive")
        if qa_joint_lr is not None and (not np.isfinite(qa_joint_lr) or qa_joint_lr <= 0):
            raise ValueError("qa_joint_lr must be positive")
        if min_branch_replay_transitions < batch_size:
            raise ValueError("min_branch_replay_transitions must be >= batch_size")
        if not np.isfinite(cross_lane_ratio) or not 0 <= cross_lane_ratio < 1:
            raise ValueError("cross_lane_ratio must be finite and lie in [0, 1)")
        if not np.isfinite(residual_exploration_std) or residual_exploration_std < 0:
            raise ValueError("residual_exploration_std must be finite and non-negative")
        for name, value in (
            ("noise_gradient_max_norm", noise_gradient_max_norm),
            ("residual_gradient_max_norm", residual_gradient_max_norm),
        ):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if termination_semantics not in {
            "legacy_continue_after_done",
            "early_break_on_done",
        }:
            raise ValueError("Unknown ActionChunk termination semantics")

        self.exec_action_low = self._constructor_bound(
            "exec_action_low", exec_action_low, action_dim, _init_setup_model, 0.0
        )
        self.exec_action_high = self._constructor_bound(
            "exec_action_high", exec_action_high, action_dim, _init_setup_model, 1.0
        )
        if not np.all(self.exec_action_high > self.exec_action_low):
            raise ValueError("Execution upper bounds must exceed lower bounds")
        self.residual_net_arch = tuple(int(v) for v in residual_net_arch)
        self.residual_activation = residual_activation
        self.residual_lr = float(residual_lr)
        if qa_joint_lr is None:
            self.qa_joint_lr = float(
                3e-4 if callable(learning_rate) else learning_rate
            )
        else:
            self.qa_joint_lr = float(qa_joint_lr)
        self.schedule_profile = schedule_profile
        self._schedule_overrides = {
            key: value
            for key, value in {
                "phase_b_steps": phase_b_steps,
                "phase_r_steps": phase_r_steps,
                "phase_j_steps": phase_j_steps,
                "phase_j_enabled": phase_j_enabled,
                "beta_ramp_steps": beta_ramp_steps,
                "beta_target": beta_target,
                "base_lane_probability": base_lane_probability,
                "beta_hold_steps": beta_hold_steps,
                "beta_floor": beta_floor,
            }.items()
            if value is not None
        }
        self.beta_target = float(beta_target)
        self.residual_scale = float(beta_target)  # read-only compatibility datum
        self.min_branch_replay_transitions = int(min_branch_replay_transitions)
        # Co-training model-level flags (not schedule fields).  Defaults OFF/0
        # keep the frozen Core V1 behavior identical.
        self.qa_joint_shadow_in_b = bool(qa_joint_shadow_in_b)
        self.cross_lane_ratio = float(cross_lane_ratio)
        self.qa_base_cross_lane = bool(qa_base_cross_lane)
        self.residual_exploration_std = float(residual_exploration_std)
        self.noise_gradient_max_norm = float(noise_gradient_max_norm)
        self.residual_gradient_max_norm = float(residual_gradient_max_norm)
        self.lane_seed = int(lane_seed if lane_seed is not None else (seed or 0) + 17_071)
        self.termination_semantics = termination_semantics
        self.numerical_bound_tolerance = float(numerical_bound_tolerance)
        self.diagnostics_interval_updates = int(diagnostics_interval_updates)
        self.ranking_diag_n_obs = int(ranking_diag_n_obs)
        self.ranking_diag_seed = int(
            ranking_diag_seed
            if ranking_diag_seed is not None
            else (seed or 0) ^ 0x9E3779B9
        )
        # Gated V1.1/V1.2 flags (all default OFF; plan requires gate evidence
        # before enabling).
        self.enable_qw_ranking = bool(enable_qw_ranking)
        self.qw_ranking_tau_gap = float(qw_ranking_tau_gap)
        self.qw_ranking_lambda = float(qw_ranking_lambda)
        self.qw_ranking_k_candidates = int(qw_ranking_k_candidates)
        self.qa_joint_target_smoothing = bool(qa_joint_target_smoothing)
        self.qa_joint_smoothing_std = float(qa_joint_smoothing_std)
        self.qa_joint_smoothing_clip = float(qa_joint_smoothing_clip)
        self.architecture_version = ARCHITECTURE_VERSION
        self.replay_schema_version = SCHEMA_VERSION

        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            action_noise=None,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            optimize_memory_usage=False,
            ent_coef=ent_coef,
            target_update_interval=target_update_interval,
            target_entropy=target_entropy,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            use_sde_at_warmup=use_sde_at_warmup,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=_init_setup_model,
            actor_gradient_steps=actor_gradient_steps,
            diffusion_policy=diffusion_policy,
            diffusion_act_dim=diffusion_act_dim,
            noise_critic_grad_steps=noise_critic_grad_steps,
            critic_backup_combine_type=critic_backup_combine_type,
        )

    @staticmethod
    def _constructor_bound(
        name: str,
        value: Optional[np.ndarray],
        action_dim: int,
        setup: bool,
        shell_value: float,
    ) -> np.ndarray:
        if value is None:
            if setup:
                raise ValueError(f"{name} is required")
            return np.full(action_dim, shell_value, dtype=np.float32)
        array = np.asarray(value, dtype=np.float32)
        if array.shape != (action_dim,) or not np.isfinite(array).all():
            raise ValueError(f"{name} must be finite with shape ({action_dim},)")
        return array.copy()

    def _setup_model(self) -> None:
        if hasattr(self, "modulation_action_space") or hasattr(self, "critic_modulation"):
            raise ValueError(DEPRECATED_CHECKPOINT_MESSAGE)
        if getattr(self, "architecture_version", ARCHITECTURE_VERSION) != ARCHITECTURE_VERSION:
            raise ValueError(
                f"Unsupported hierarchy architecture version: {self.architecture_version!r}"
            )
        if self.diffusion_policy is None:
            raise ValueError("diffusion_policy is required")
        super()._setup_model()
        if not isinstance(self.replay_buffer, HierarchyTaggedReplayBuffer):
            raise TypeError("Hierarchy replay construction did not preserve tagged metadata")
        if self.policy.share_features_extractor:
            raise ValueError("share_features_extractor=True is forbidden")
        if not isinstance(self.observation_space, spaces.Box) or len(self.observation_space.shape) != 1:
            raise ValueError("Core V1 requires a flat Box observation")
        if not isinstance(self.action_space, spaces.Box) or len(self.action_space.shape) != 1:
            raise ValueError("Core V1 requires a flat Box action")

        self.action_dim_flat = self.diffusion_act_chunk * self.diffusion_act_dim
        self.observation_dim = int(np.prod(self.observation_space.shape))
        if self.action_space.shape != (self.action_dim_flat,):
            raise ValueError("Execution action dimension and DDIM output disagree")
        if not np.array_equal(self.action_space.low, self.exec_action_low) or not np.array_equal(
            self.action_space.high, self.exec_action_high
        ):
            raise ValueError("Explicit execution bounds conflict with environment bounds")

        self.noise_actor = self.actor
        self.qa_base = self.critic
        self.qa_base_target = self.critic_target
        self.qw_base = self.critic_noise
        self.noise_actor_optimizer = self.actor.optimizer
        self.qa_base_optimizer = self.critic.optimizer
        self.qw_base_optimizer = self.qw_base.optimizer

        self.qa_joint = self.policy.make_critic(features_extractor=None).to(self.device)
        self.qa_joint_target = self.policy.make_critic(features_extractor=None).to(self.device)
        self.qa_joint.load_state_dict(self.qa_base.state_dict(), strict=True)
        self.qa_joint_target.load_state_dict(self.qa_joint.state_dict(), strict=True)
        self.qa_joint_optimizer = self.policy.optimizer_class(
            self.qa_joint.parameters(),
            lr=self.qa_joint_lr,
            **self.policy.optimizer_kwargs,
        )
        self.qa_joint.optimizer = self.qa_joint_optimizer

        self.residual_actor = ResidualActor(
            self.observation_dim,
            self.action_dim_flat,
            self.residual_net_arch,
            self.residual_activation,
        ).to(self.device)
        self.residual_actor_target = copy.deepcopy(self.residual_actor).to(self.device)
        self.residual_actor_optimizer = th.optim.Adam(
            self.residual_actor.parameters(), lr=self.residual_lr
        )
        self.reference_noise_actor = self.policy.make_actor().to(self.device)
        self.reference_noise_actor.load_state_dict(self.actor.state_dict(), strict=True)

        self._noise_action_low_tensor = th.as_tensor(
            self.policy.action_space.low, device=self.device, dtype=th.float32
        ).reshape(self.action_dim_flat)
        self._noise_action_high_tensor = th.as_tensor(
            self.policy.action_space.high, device=self.device, dtype=th.float32
        ).reshape(self.action_dim_flat)
        self._exec_action_low_tensor = th.as_tensor(
            self.exec_action_low, device=self.device, dtype=th.float32
        )
        self._exec_action_high_tensor = th.as_tensor(
            self.exec_action_high, device=self.device, dtype=th.float32
        )

        self.hierarchy_schedule = make_hierarchy_schedule(
            self.schedule_profile,
            n_envs=self.n_envs,
            overrides=self._schedule_overrides,
        )
        if not hasattr(self, "_lane_rng"):
            self._lane_rng = np.random.default_rng(self.lane_seed)
        if not hasattr(self, "_active_branch_mode"):
            self._active_branch_mode = np.full(self.n_envs, -1, dtype=np.int8)
        if not hasattr(self, "_active_episode_id"):
            self._active_episode_id = np.full(self.n_envs, -1, dtype=np.int64)
        if not hasattr(self, "_chunk_index_in_episode"):
            self._chunk_index_in_episode = np.zeros(self.n_envs, dtype=np.int32)
        # Section 10: per-branch closed-episode statistics (returns, lengths,
        # early falls).  Diagnostic-window state: accumulated during rollout,
        # read-and-reset on each diagnostics dump.  Survives checkpoint/resume
        # so an in-flight episode keeps its accumulated return.
        if not hasattr(self, "_episode_return_accum"):
            self._episode_return_accum = np.zeros(self.n_envs, dtype=np.float64)
        if not hasattr(self, "_episode_length_accum"):
            self._episode_length_accum = np.zeros(self.n_envs, dtype=np.int64)
        if not hasattr(self, "_episode_branch_accum"):
            self._episode_branch_accum = np.full(self.n_envs, -1, dtype=np.int8)
        if not hasattr(self, "_closed_episode_stats"):
            self._closed_episode_stats = {
                int(BranchMode.BASE): _fresh_episode_stat_window(),
                int(BranchMode.JOINT): _fresh_episode_stat_window(),
            }
        for name in (
            "_active_branch_mode",
            "_active_episode_id",
            "_chunk_index_in_episode",
            "_episode_return_accum",
            "_episode_length_accum",
            "_episode_branch_accum",
        ):
            if np.asarray(getattr(self, name)).shape != (self.n_envs,):
                raise ValueError(f"Saved {name} does not match loaded n_envs")
        if not hasattr(self, "next_episode_id"):
            self.next_episode_id = 0
        self._pending_rollout_metadata: Optional[dict[str, np.ndarray]] = None
        if not hasattr(self, "_collected_phase_since_train"):
            self._collected_phase_since_train = None
        if not hasattr(self, "_last_action_phase"):
            self._last_action_phase = int(self.hierarchy_schedule.phase_at(0))
        if not hasattr(self, "_last_action_batch_start"):
            self._last_action_batch_start = 0
        if not hasattr(self, "_last_action_beta"):
            self._last_action_beta = float(self.hierarchy_schedule.beta_at(0))
        if not hasattr(self, "qa_joint_generation"):
            self.qa_joint_generation = 0
        if not hasattr(self, "qa_joint_optimizer_steps_since_clone"):
            self.qa_joint_optimizer_steps_since_clone = 0
        if not hasattr(self, "qa_joint_clone_online_step"):
            self.qa_joint_clone_online_step = -1
        if not hasattr(self, "_joint_phase_initialized"):
            self._joint_phase_initialized = False
        if not hasattr(self, "environment_discontinuity_count"):
            self.environment_discontinuity_count = 0
        if not hasattr(self, "noise_policy_version"):
            self.noise_policy_version = 0
        if not hasattr(self, "residual_policy_version"):
            self.residual_policy_version = 0
        # Section 10 "policy age": chunk transitions since the current noise /
        # residual policy version was created (set on every version increment).
        if not hasattr(self, "_noise_policy_birth_step"):
            self._noise_policy_birth_step = 0
        if not hasattr(self, "_residual_policy_birth_step"):
            self._residual_policy_birth_step = 0
        self._initialize_counters()

        self.qa_base_batch_norm_stats = get_parameters_by_name(self.qa_base, ["running_"])
        self.qa_base_target_batch_norm_stats = get_parameters_by_name(
            self.qa_base_target, ["running_"]
        )
        self.qa_joint_batch_norm_stats = get_parameters_by_name(self.qa_joint, ["running_"])
        self.qa_joint_target_batch_norm_stats = get_parameters_by_name(
            self.qa_joint_target, ["running_"]
        )
        self._freeze_all_targets_and_diffusion()
        if self.hierarchy_schedule.phase_b_steps == 0:
            self._activate_joint_phase(boundary_step=0)
        self._assert_parameter_ownership()
        self.set_inference_mode()

    def _initialize_counters(self) -> None:
        names = (
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
            "emergency_clamp_count",
        )
        for name in names:
            if not hasattr(self, name):
                setattr(self, name, 0)
        if not hasattr(self, "_max_preclamp_violation"):
            self._max_preclamp_violation = 0.0
        if not hasattr(self, "requested_optimizer_steps"):
            self.requested_optimizer_steps = {
                name: 0
                for name in (
                    "qa_base",
                    "qa_joint",
                    "qw_base",
                    "noise_actor",
                    "alpha",
                    "residual_actor",
                )
            }

    def _freeze_diffusion_policy(self) -> None:
        for candidate in (
            self.diffusion_policy,
            getattr(self.diffusion_policy, "base_policy", None),
        ):
            if isinstance(candidate, nn.Module):
                _freeze_target(candidate)

    def _freeze_all_targets_and_diffusion(self) -> None:
        for target in (
            self.qa_base_target,
            self.qa_joint_target,
            self.residual_actor_target,
            self.reference_noise_actor,
        ):
            _freeze_target(target)
        self._freeze_diffusion_policy()

    def set_inference_mode(self) -> None:
        self.policy.set_training_mode(False)
        self.actor.eval()
        for critic in (
            self.qa_base,
            self.qa_base_target,
            self.qw_base,
            self.qa_joint,
            self.qa_joint_target,
        ):
            critic.set_training_mode(False)
        self.residual_actor.eval()
        self._freeze_all_targets_and_diffusion()

    @staticmethod
    def _parameter_ids(module: nn.Module) -> set[int]:
        return {id(parameter) for parameter in module.parameters()}

    @staticmethod
    def _storage_pointers(module: nn.Module) -> set[int]:
        return {parameter.untyped_storage().data_ptr() for parameter in module.parameters()}

    def _assert_parameter_ownership(self) -> None:
        online = {
            "noise_actor": self.actor,
            "qa_base": self.qa_base,
            "qw_base": self.qw_base,
            "qa_joint": self.qa_joint,
            "residual_actor": self.residual_actor,
        }
        items = list(online.items())
        for index, (left_name, left) in enumerate(items):
            for right_name, right in items[index + 1 :]:
                if self._parameter_ids(left) & self._parameter_ids(right):
                    raise RuntimeError(f"Parameter ownership overlaps: {left_name}/{right_name}")
                if self._storage_pointers(left) & self._storage_pointers(right):
                    raise RuntimeError(f"Parameter storage overlaps: {left_name}/{right_name}")
        optimizer_sets = {
            "noise_actor": {
                id(p) for group in self.actor.optimizer.param_groups for p in group["params"]
            },
            "qa_base": {
                id(p) for group in self.qa_base.optimizer.param_groups for p in group["params"]
            },
            "qw_base": {
                id(p) for group in self.qw_base.optimizer.param_groups for p in group["params"]
            },
            "qa_joint": {
                id(p) for group in self.qa_joint_optimizer.param_groups for p in group["params"]
            },
            "residual_actor": {
                id(p) for group in self.residual_actor_optimizer.param_groups for p in group["params"]
            },
        }
        optimizer_items = list(optimizer_sets.items())
        for index, (left_name, left) in enumerate(optimizer_items):
            for right_name, right in optimizer_items[index + 1 :]:
                if left & right:
                    raise RuntimeError(f"Optimizer ownership overlaps: {left_name}/{right_name}")

    def _activate_joint_phase(self, *, boundary_step: int) -> None:
        if self._joint_phase_initialized:
            return
        if not self.qa_joint_shadow_in_b:
            # Frozen Core V1: the boundary hard-clones QA_base into QA_joint.
            self.qa_joint.load_state_dict(self.qa_base.state_dict(), strict=True)
            self.qa_joint_target.load_state_dict(
                self.qa_joint.state_dict(), strict=True
            )
            self.qa_joint_optimizer = self.policy.optimizer_class(
                self.qa_joint.parameters(),
                lr=self.qa_joint_lr,
                **self.policy.optimizer_kwargs,
            )
            self.qa_joint.optimizer = self.qa_joint_optimizer
        # Co-training: QA_joint was shadow-trained on BASE transitions during
        # Phase B (its valid action IS the base action at beta=0), so the
        # boundary keeps its weights and Adam state instead of wiping them.
        self.residual_actor_target.load_state_dict(
            self.residual_actor.state_dict(), strict=True
        )
        self.qa_joint_generation += 1
        self.qa_joint_optimizer_steps_since_clone = 0
        self.qa_joint_clone_online_step = int(boundary_step)
        self._joint_phase_initialized = True
        _freeze_target(self.qa_joint_target)
        _freeze_target(self.residual_actor_target)

    def _ensure_phase_activation(self) -> None:
        if self.num_timesteps >= self.hierarchy_schedule.phase_r_start:
            self._activate_joint_phase(
                boundary_step=self.hierarchy_schedule.phase_r_start
            )

    def _unscale_noise(self, noise_scaled: th.Tensor) -> th.Tensor:
        if noise_scaled.ndim != 2 or noise_scaled.shape[1] != self.action_dim_flat:
            raise ValueError("noise_scaled has an incompatible shape")
        scaled = noise_scaled.detach()
        low = self._noise_action_low_tensor.to(dtype=scaled.dtype)
        high = self._noise_action_high_tensor.to(dtype=scaled.dtype)
        decoder = low + 0.5 * (scaled + 1.0) * (high - low)
        return decoder.reshape(-1, self.diffusion_act_chunk, self.diffusion_act_dim).detach()

    @th.no_grad()
    def _decode_noise_decoder_input(
        self, observation: th.Tensor, noise_decoder_input: th.Tensor
    ) -> th.Tensor:
        action = self.diffusion_policy(
            observation, noise_decoder_input, return_numpy=False
        )
        if isinstance(action, np.ndarray):
            action = th.as_tensor(action, device=self.device, dtype=observation.dtype)
        if not isinstance(action, th.Tensor):
            raise TypeError("diffusion_policy must return Tensor or ndarray")
        expected = (
            observation.shape[0],
            self.diffusion_act_chunk,
            self.diffusion_act_dim,
        )
        if tuple(action.shape) != expected:
            raise ValueError(
                f"Diffusion output shape mismatch: expected {expected}, got {tuple(action.shape)}"
            )
        return action.to(self.device, observation.dtype).reshape(
            -1, self.action_dim_flat
        ).detach()

    def _generate_hierarchical_action(
        self,
        observation: th.Tensor,
        noise_scaled: th.Tensor,
        *,
        zero_residual: bool = False,
        beta: Optional[float] = None,
        residual_actor: Optional[ResidualActor] = None,
    ) -> HierarchicalActionOutput:
        if observation.ndim != 2 or observation.shape[1] != self.observation_dim:
            raise ValueError("observation has an incompatible shape")
        if noise_scaled.shape != (observation.shape[0], self.action_dim_flat):
            raise ValueError("noise_scaled does not match observation batch")
        decoder = self._unscale_noise(noise_scaled)
        base = self._decode_noise_decoder_input(observation, decoder).detach()
        if zero_residual:
            logits = th.zeros_like(base)
        else:
            actor = residual_actor or self.residual_actor
            logits = actor.forward_pre_tanh(
                observation, noise_scaled.detach(), base.detach()
            )
        composition = compose_action(
            base,
            logits,
            self.current_beta if beta is None else beta,
            self._exec_action_low_tensor,
            self._exec_action_high_tensor,
            numerical_tolerance=self.numerical_bound_tolerance,
        )
        return HierarchicalActionOutput(
            noise_scaled=noise_scaled,
            noise_decoder_input=decoder,
            action_base=base,
            residual_pre_tanh=logits,
            residual_unit=composition.residual_unit,
            margin_positive=composition.margin_positive,
            margin_negative=composition.margin_negative,
            action_residual_delta=composition.action_residual_delta,
            action_exec_unclamped=composition.action_exec_unclamped,
            action_exec=composition.action_exec,
            emergency_clamp_applied=composition.emergency_clamp_applied,
            maximum_preclamp_violation=composition.maximum_preclamp_violation,
        )

    @property
    def current_phase(self) -> HierarchyPhase:
        return self.hierarchy_schedule.phase_at(self.num_timesteps)

    @property
    def current_beta(self) -> float:
        return self.hierarchy_schedule.beta_at(self.num_timesteps)

    @staticmethod
    def _observation_batch(observation: np.ndarray, observation_dim: int) -> np.ndarray:
        value = np.asarray(observation, dtype=np.float32)
        if value.shape == (observation_dim,):
            value = value[None, :]
        if value.ndim != 2 or value.shape[1] != observation_dim:
            raise ValueError("observation has an incompatible array shape")
        return value

    def _allocate_unassigned_lanes(self) -> None:
        if not getattr(self, "_replay_episode_id_floor_seeded", False):
            # The immutable tagged prefill that seeds the replay already owns
            # episode IDs starting at 0.  Continue the online episode counter
            # ABOVE every ID already present in the replay so online tags never
            # collide with offline ones (episode_id is a monotonic tag; nothing
            # keys on its absolute value, but collisions would silently confuse
            # any future per-episode tooling).  Seeded once, lazily, after the
            # replay (and its prefill) are attached; monotonic on resume.
            self._replay_episode_id_floor_seeded = True
            replay = getattr(self, "replay_buffer", None)
            if (
                replay is not None
                and hasattr(replay, "episode_id")
                and replay.episode_id is not None
                and replay.episode_id.size
            ):
                floor = int(replay.episode_id.max()) + 1
                if self.next_episode_id < floor:
                    self.next_episode_id = floor
        probability = self.hierarchy_schedule.base_probability_at(self.num_timesteps)
        for environment_id in range(self.n_envs):
            if self._active_branch_mode[environment_id] >= 0:
                continue
            if probability >= 1.0:
                lane = BranchMode.BASE
            elif probability <= 0.0:
                lane = BranchMode.JOINT
            else:
                lane = (
                    BranchMode.BASE
                    if self._lane_rng.random() < probability
                    else BranchMode.JOINT
                )
            self._active_branch_mode[environment_id] = int(lane)
            self._active_episode_id[environment_id] = self.next_episode_id
            self.next_episode_id += 1
            self._chunk_index_in_episode[environment_id] = 0

    def _sample_behavior_noise(
        self,
        observation: th.Tensor,
        *,
        warmup: bool,
    ) -> tuple[th.Tensor, th.Tensor, np.ndarray, int]:
        if warmup:
            noise = th.empty(
                (observation.shape[0], self.action_dim_flat),
                device=self.device,
                dtype=observation.dtype,
            ).uniform_(-1.0, 1.0)
            log_prob = th.zeros(observation.shape[0], device=self.device)
            valid = np.zeros(observation.shape[0], dtype=np.bool_)
            source = int(NoiseSampleSource.UNIFORM_WARMUP)
        else:
            with th.no_grad():
                noise, log_prob = self.actor.action_log_prob(observation)
            valid = np.ones(observation.shape[0], dtype=np.bool_)
            source = int(NoiseSampleSource.CURRENT_ACTOR)
        return noise.detach(), log_prob.detach(), valid, source

    @th.no_grad()
    def _generate_mixed_lane_behavior(
        self,
        observation: th.Tensor,
        noise_scaled: th.Tensor,
        branches: np.ndarray,
        beta: float,
        *,
        residual_enabled: Optional[np.ndarray] = None,
    ) -> HierarchicalActionOutput:
        decoder = self._unscale_noise(noise_scaled)
        base = self._decode_noise_decoder_input(observation, decoder).detach()
        logits = th.zeros_like(base)
        unit = th.zeros_like(base)
        delta = th.zeros_like(base)
        action_exec_unclamped = base.clone()
        action_exec = base.clone()
        margin_positive = self._exec_action_high_tensor - base
        margin_negative = base - self._exec_action_low_tensor
        emergency = th.zeros(base.shape[0], device=self.device, dtype=th.bool)
        maximum_violation = th.zeros(1, device=self.device, dtype=base.dtype)

        joint_mask = branches == int(BranchMode.JOINT)
        if residual_enabled is not None:
            enabled = np.asarray(residual_enabled, dtype=np.bool_)
            if enabled.shape != (self.n_envs,):
                raise ValueError(
                    f"residual_enabled must have shape ({self.n_envs},)"
                )
            joint_mask &= enabled
        joint_indices_np = np.flatnonzero(joint_mask)
        if joint_indices_np.size:
            indices = th.as_tensor(joint_indices_np, device=self.device, dtype=th.long)
            joint_logits = self.residual_actor.forward_pre_tanh(
                observation.index_select(0, indices),
                noise_scaled.index_select(0, indices).detach(),
                base.index_select(0, indices).detach(),
            )
            if self.residual_exploration_std > 0.0 and beta > 0.0:
                # Behavior exploration on the JOINT lane only: a small pre-tanh
                # perturbation that flows through compose_action into the
                # executed action and every stored residual metadata field
                # (pre_tanh/unit/delta), so replay metadata stays exact.  At
                # beta=0 the perturbation would not change the executed action,
                # so it is gated off to keep stored metadata clean.
                joint_logits = joint_logits + self.residual_exploration_std * th.randn_like(
                    joint_logits
                )
            composed = compose_action(
                base.index_select(0, indices),
                joint_logits,
                beta,
                self._exec_action_low_tensor,
                self._exec_action_high_tensor,
                numerical_tolerance=self.numerical_bound_tolerance,
            )
            logits.index_copy_(0, indices, joint_logits)
            unit.index_copy_(0, indices, composed.residual_unit)
            delta.index_copy_(0, indices, composed.action_residual_delta)
            action_exec_unclamped.index_copy_(
                0, indices, composed.action_exec_unclamped
            )
            action_exec.index_copy_(0, indices, composed.action_exec)
            emergency.index_copy_(0, indices, composed.emergency_clamp_applied)
            maximum_violation = composed.maximum_preclamp_violation
        # Section 10: record emergency-clamp activations and the worst pre-clamp
        # violation magnitude encountered during this rollout step.
        if bool(th.any(emergency).cpu().item()):
            self.emergency_clamp_count += int(emergency.sum().cpu().item())
        violation_value = float(maximum_violation.detach().cpu().item())
        self._max_preclamp_violation = max(
            self._max_preclamp_violation, violation_value
        )
        return HierarchicalActionOutput(
            noise_scaled=noise_scaled,
            noise_decoder_input=decoder,
            action_base=base,
            residual_pre_tanh=logits,
            residual_unit=unit,
            margin_positive=margin_positive,
            margin_negative=margin_negative,
            action_residual_delta=delta,
            action_exec_unclamped=action_exec_unclamped,
            action_exec=action_exec,
            emergency_clamp_applied=emergency,
            maximum_preclamp_violation=maximum_violation,
        )

    def _sample_action(
        self,
        learning_starts: int,
        action_noise: Optional[ActionNoise] = None,
        n_envs: int = 1,
    ) -> tuple[np.ndarray, np.ndarray]:
        if action_noise is not None:
            raise ValueError("Core V1 forbids external action_noise")
        if n_envs != self.n_envs:
            raise ValueError("Rollout n_envs changed after hierarchy setup")
        if self._pending_rollout_metadata is not None:
            raise RuntimeError("Previous hierarchy action has not been stored")
        if self._last_obs is None or isinstance(self._last_obs, dict):
            raise RuntimeError("Flat _last_obs is required before action sampling")
        self._ensure_phase_activation()
        phase = self.current_phase
        if self._collected_phase_since_train is not None and int(phase) != int(
            self._collected_phase_since_train
        ):
            raise RuntimeError(
                "A rollout collection crossed a B/R/J boundary before training; "
                "use train_freq=1 vector step"
            )
        self._collected_phase_since_train = int(phase)
        self._last_action_phase = int(phase)
        self._last_action_batch_start = int(self.num_timesteps)
        self._allocate_unassigned_lanes()
        observation = th.as_tensor(
            self._observation_batch(self._last_obs, self.observation_dim),
            device=self.device,
            dtype=th.float32,
        )
        warmup = self.num_timesteps < learning_starts and not (
            self.use_sde and self.use_sde_at_warmup
        )
        noise, log_prob, log_prob_valid, noise_source = self._sample_behavior_noise(
            observation, warmup=warmup
        )
        branches = self._active_branch_mode.copy()
        beta = self.current_beta
        self._last_action_beta = float(beta)
        generated = self._generate_mixed_lane_behavior(
            observation,
            noise,
            branches,
            beta,
            residual_enabled=np.full(self.n_envs, not warmup, dtype=np.bool_),
        )
        residual_applied = (
            (branches == int(BranchMode.JOINT))
            & np.full(self.n_envs, not warmup, dtype=np.bool_)
        )
        noise_version = np.full(
            self.n_envs,
            -1 if warmup else self.noise_policy_version,
            dtype=np.int64,
        )
        residual_version = np.where(
            residual_applied, self.residual_policy_version, -1
        ).astype(np.int64)
        self._pending_rollout_metadata = {
            "branch_mode": branches.astype(np.uint8),
            "noise_scaled": generated.noise_scaled.cpu().numpy().astype(np.float32),
            "noise_log_prob": log_prob.cpu().numpy().astype(np.float32),
            "noise_log_prob_valid": log_prob_valid,
            "noise_sample_source": np.full(
                self.n_envs, noise_source, dtype=np.uint8
            ),
            "transition_origin": np.full(
                self.n_envs, int(TransitionOrigin.ONLINE), dtype=np.uint8
            ),
            "action_base": generated.action_base.cpu().numpy().astype(np.float32),
            "residual_pre_tanh": generated.residual_pre_tanh.cpu()
            .numpy()
            .astype(np.float32),
            "residual_unit": generated.residual_unit.cpu().numpy().astype(np.float32),
            "action_residual_delta": generated.action_residual_delta.cpu()
            .numpy()
            .astype(np.float32),
            "action_exec": generated.action_exec.cpu().numpy().astype(np.float32),
            "beta": np.full(self.n_envs, beta, dtype=np.float32),
            "residual_applied": residual_applied.astype(np.bool_),
            "emergency_clamp_applied": generated.emergency_clamp_applied.cpu()
            .numpy()
            .astype(np.bool_),
            "episode_id": self._active_episode_id.copy(),
            "environment_id": np.arange(self.n_envs, dtype=np.int32),
            "chunk_index_in_episode": self._chunk_index_in_episode.copy(),
            "nominal_primitive_steps": np.full(
                self.n_envs, self.diffusion_act_chunk, dtype=np.uint8
            ),
            "actual_primitive_steps": np.zeros(self.n_envs, dtype=np.uint8),
            "termination_primitive_index": np.full(
                self.n_envs, -1, dtype=np.int8
            ),
            "termination_semantics": np.zeros(self.n_envs, dtype=np.uint8),
            "noise_policy_version": noise_version,
            "residual_policy_version": residual_version,
        }
        action = generated.action_exec.cpu().numpy().astype(np.float32)
        return action, action.copy()

    def _store_transition(
        self,
        replay_buffer: ReplayBuffer,
        buffer_action: np.ndarray,
        new_obs: Union[np.ndarray, dict[str, np.ndarray]],
        reward: np.ndarray,
        dones: np.ndarray,
        infos: list[dict[str, Any]],
    ) -> None:
        if not isinstance(replay_buffer, HierarchyTaggedReplayBuffer):
            raise TypeError("Core V1 requires tagged replay during rollout")
        if self._pending_rollout_metadata is None:
            raise RuntimeError("No pending hierarchy behavior tuple to store")
        metadata = {key: value.copy() for key, value in self._pending_rollout_metadata.items()}
        semantics_values = {
            "legacy_continue_after_done": int(
                TerminationSemantics.LEGACY_CONTINUE_AFTER_DONE
            ),
            "early_break_on_done": int(TerminationSemantics.EARLY_BREAK_ON_DONE),
        }
        for index, (done, info) in enumerate(zip(dones, infos)):
            if bool(done) and info.get("terminal_observation") is None:
                raise ValueError(
                    "A done hierarchy transition requires terminal_observation "
                    "before replay insertion"
                )
            nominal = int(info.get("nominal_primitive_steps", -1))
            actual = int(info.get("actual_primitive_steps", -1))
            if nominal != self.diffusion_act_chunk or not 1 <= actual <= nominal:
                raise ValueError("ActionChunk primitive counters are missing or invalid")
            metadata["nominal_primitive_steps"][index] = nominal
            metadata["actual_primitive_steps"][index] = actual
            termination_index = info.get("termination_primitive_index")
            metadata["termination_primitive_index"][index] = (
                -1 if termination_index is None else int(termination_index)
            )
            semantic = info.get(
                "action_chunk_termination_semantics", self.termination_semantics
            )
            if semantic not in semantics_values:
                raise ValueError("Unknown transition ActionChunk semantics")
            metadata["termination_semantics"][index] = semantics_values[semantic]
        replay_buffer.stage_metadata(metadata)
        try:
            super()._store_transition(
                replay_buffer, buffer_action, new_obs, reward, dones, infos
            )
        except Exception:
            replay_buffer.clear_staged_metadata()
            raise
        finally:
            self._pending_rollout_metadata = None
        for environment_id, done in enumerate(dones):
            branch = int(metadata["branch_mode"][environment_id])
            self._episode_return_accum[environment_id] += float(
                reward[environment_id]
            )
            self._episode_length_accum[environment_id] += int(
                metadata["actual_primitive_steps"][environment_id]
            )
            self._episode_branch_accum[environment_id] = branch
            if bool(done):
                early_fall = bool(
                    infos[environment_id].get("termination_reason")
                    == "environment_terminal"
                )
                self._record_closed_episode(
                    branch,
                    float(self._episode_return_accum[environment_id]),
                    int(self._episode_length_accum[environment_id]),
                    early_fall,
                )
                self._episode_return_accum[environment_id] = 0.0
                self._episode_length_accum[environment_id] = 0
                self._episode_branch_accum[environment_id] = -1
                self._active_branch_mode[environment_id] = -1
                self._active_episode_id[environment_id] = -1
                self._chunk_index_in_episode[environment_id] = 0
            else:
                self._chunk_index_in_episode[environment_id] += 1

    def collect_rollouts(self, *args: Any, **kwargs: Any) -> Any:
        try:
            return super().collect_rollouts(*args, **kwargs)
        finally:
            # A callback may stop after environment.step and before storage.
            if self._pending_rollout_metadata is not None:
                self._pending_rollout_metadata = None
                if isinstance(self.replay_buffer, HierarchyTaggedReplayBuffer):
                    self.replay_buffer.clear_staged_metadata()

    def _predict_noise(
        self,
        observation: np.ndarray,
        *,
        actor: nn.Module,
        deterministic: bool,
    ) -> th.Tensor:
        obs = th.as_tensor(
            self._observation_batch(observation, self.observation_dim),
            device=self.device,
            dtype=th.float32,
        )
        with th.no_grad():
            return actor(obs, deterministic=deterministic).detach()

    def predict_with_components(
        self,
        observation: Union[np.ndarray, dict[str, np.ndarray]],
        state: Optional[tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
        *,
        mode: str = "current_full_hierarchy",
    ) -> tuple[dict[str, np.ndarray], Optional[tuple[np.ndarray, ...]]]:
        del episode_start
        if isinstance(observation, dict):
            raise TypeError("Core V1 supports flat observations only")
        modes = {
            "current_full_hierarchy",
            "current_base_only",
            "reference_base",
        }
        if mode not in modes:
            raise ValueError(f"Unknown hierarchy prediction mode {mode!r}")
        self.set_inference_mode()
        actor = self.reference_noise_actor if mode == "reference_base" else self.actor
        noise = self._predict_noise(
            np.asarray(observation), actor=actor, deterministic=deterministic
        )
        obs = th.as_tensor(
            self._observation_batch(np.asarray(observation), self.observation_dim),
            device=self.device,
            dtype=th.float32,
        )
        with th.no_grad():
            generated = self._generate_hierarchical_action(
                obs,
                noise,
                zero_residual=mode != "current_full_hierarchy",
                beta=self.current_beta,
            )
        single = np.asarray(observation).shape == (self.observation_dim,)
        components = {
            name: getattr(generated, name).detach().cpu().numpy()
            for name in generated._fields
        }
        if single:
            components = {name: value[0] for name, value in components.items()}
        return components, state

    def predict_diffused(
        self,
        observation: Union[np.ndarray, dict[str, np.ndarray]],
        state: Optional[tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, Optional[tuple[np.ndarray, ...]]]:
        components, state = self.predict_with_components(
            observation,
            state,
            episode_start,
            deterministic,
            mode="current_full_hierarchy",
        )
        return components["action_exec"], state

    def predict(
        self,
        observation: Union[np.ndarray, dict[str, np.ndarray]],
        state: Optional[tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, Optional[tuple[np.ndarray, ...]]]:
        return self.predict_diffused(
            observation, state, episode_start, deterministic
        )

    def predict_zero_residual(
        self,
        observation: Union[np.ndarray, dict[str, np.ndarray]],
        state: Optional[tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, Optional[tuple[np.ndarray, ...]]]:
        components, state = self.predict_with_components(
            observation,
            state,
            episode_start,
            deterministic,
            mode="current_base_only",
        )
        return components["action_exec"], state

    def predict_reference_base(
        self,
        observation: Union[np.ndarray, dict[str, np.ndarray]],
        state: Optional[tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, Optional[tuple[np.ndarray, ...]]]:
        components, state = self.predict_with_components(
            observation,
            state,
            episode_start,
            deterministic,
            mode="reference_base",
        )
        return components["action_exec"], state

    def _current_entropy_coefficient(self) -> th.Tensor:
        if self.log_ent_coef is not None:
            return th.exp(self.log_ent_coef.detach())
        return self.ent_coef_tensor.detach()

    @staticmethod
    def _minimum_q(values: tuple[th.Tensor, ...]) -> th.Tensor:
        if not values:
            raise ValueError("A twin critic must expose at least one head")
        return th.min(th.cat(values, dim=1), dim=1, keepdim=True).values

    def _qa_base_loss(
        self,
        replay_data: HierarchyReplayBufferSamples,
        entropy_coefficient: Optional[th.Tensor] = None,
    ) -> tuple[th.Tensor, th.Tensor, tuple[th.Tensor, ...]]:
        alpha = (
            self._current_entropy_coefficient()
            if entropy_coefficient is None
            else entropy_coefficient.detach()
        )
        with th.no_grad():
            next_noise, next_log_prob = self.actor.action_log_prob(
                replay_data.next_observations
            )
            decoder = self._unscale_noise(next_noise)
            next_base = self._decode_noise_decoder_input(
                replay_data.next_observations, decoder
            )
            next_q = self._minimum_q(
                self.qa_base_target(replay_data.next_observations, next_base)
            )
            target = replay_data.rewards + (1 - replay_data.dones) * self.gamma * (
                next_q - alpha * next_log_prob.reshape(-1, 1)
            )
        # Core V1 data-use principle: the current-state query must be the
        # transition's actually-executed action -- base and joint value objects
        # are separated by their CONTINUATION target, not by which transitions
        # they consume.  Querying the unexecuted base action of a JOINT row
        # would pair a_base with residual-influenced reward/next-state.
        current = self.qa_base(
            replay_data.observations, replay_data.action_exec
        )
        loss = 0.5 * sum(F.mse_loss(value, target) for value in current)
        return loss, target, current

    def _qa_joint_loss(
        self,
        replay_data: HierarchyReplayBufferSamples,
        entropy_coefficient: Optional[th.Tensor] = None,
        *,
        beta: Optional[float] = None,
    ) -> tuple[th.Tensor, th.Tensor, tuple[th.Tensor, ...]]:
        alpha = (
            self._current_entropy_coefficient()
            if entropy_coefficient is None
            else entropy_coefficient.detach()
        )
        target_beta = self._last_action_beta if beta is None else float(beta)
        with th.no_grad():
            next_noise, next_log_prob = self.actor.action_log_prob(
                replay_data.next_observations
            )
            generated = self._generate_hierarchical_action(
                replay_data.next_observations,
                next_noise,
                beta=target_beta,
                residual_actor=self.residual_actor_target,
            )
            next_action = generated.action_exec
            next_action = self._smooth_joint_target_action(next_action)
            next_q = self._minimum_q(
                self.qa_joint_target(replay_data.next_observations, next_action)
            )
            target = replay_data.rewards + (1 - replay_data.dones) * self.gamma * (
                next_q - alpha * next_log_prob.reshape(-1, 1)
            )
        current = self.qa_joint(
            replay_data.observations, replay_data.action_exec
        )
        loss = 0.5 * sum(F.mse_loss(value, target) for value in current)
        return loss, target, current

    def _smooth_joint_target_action(self, next_action: th.Tensor) -> th.Tensor:
        """V1.2 TD3-style target-policy smoothing for the joint critic.

        Default OFF.  Adds clipped noise to the target action so the joint
        critic's bootstrap is smooth near the target residual action, then
        re-clamps the perturbed action to the executable action box: without
        the re-clamp, a smoothing draw near the boundary could feed the joint
        critic an out-of-envelope action the residual policy can never emit
        (biased evaluated Q).
        """
        if not self.qa_joint_target_smoothing:
            return next_action
        smoothing = th.randn_like(next_action) * self.qa_joint_smoothing_std
        smoothing = smoothing.clamp(
            -self.qa_joint_smoothing_clip, self.qa_joint_smoothing_clip
        )
        return (next_action + smoothing).clamp(
            self._exec_action_low_tensor, self._exec_action_high_tensor
        )

    def _qw_base_loss(
        self, replay_data: HierarchyReplayBufferSamples
    ) -> tuple[th.Tensor, tuple[th.Tensor, ...], tuple[th.Tensor, ...]]:
        with th.no_grad():
            noise, _ = self.actor.action_log_prob(replay_data.observations)
            decoder = self._unscale_noise(noise)
            base = self._decode_noise_decoder_input(
                replay_data.observations, decoder
            )
            teacher = tuple(
                value.detach()
                for value in self.qa_base_target(
                    replay_data.observations, base
                )
            )
        student = self.qw_base(replay_data.observations, noise.detach())
        if len(student) != len(teacher):
            raise RuntimeError("QW and target QA_base head counts differ")
        losses = tuple(
            F.mse_loss(student_head, teacher_head)
            for student_head, teacher_head in zip(student, teacher)
        )
        if not losses:
            raise RuntimeError("QW must contain at least one head")
        loss = 0.5 * sum(losses)
        if self.enable_qw_ranking:
            # V1.1 confidence-masked pairwise ranking (default OFF).  Samples
            # K candidates per observation (current-actor draws + symmetric
            # local perturbations), decodes them to base actions, and applies a
            # softplus ranking loss only on pairs whose twin target-QA_base
            # heads AGREE on direction AND whose teacher gap exceeds tau_gap.
            # The temperature T_Q is normalized by the median teacher gap.
            loss = loss + self.qw_ranking_lambda * self._qw_ranking_loss(
                replay_data.observations, noise
            )
        return loss, student, teacher

    def _qw_ranking_loss(
        self, observations: th.Tensor, anchor_noise: th.Tensor
    ) -> th.Tensor:
        """V1.1 candidate ranking term.  See Codex plan msg 559."""
        k = max(4, int(self.qw_ranking_k_candidates))
        with th.no_grad():
            batch = observations.shape[0]
            # Candidates: k//2 current-actor draws + k//2 symmetric local
            # perturbations around the anchor noise (so candidates remain near
            # the actor's query region even when actor variance collapses).
            current_noise, _ = self.actor.action_log_prob(
                observations.repeat(k // 2, 1)
            )
            local_eps = th.randn_like(anchor_noise.repeat(k // 2, 1)) * 0.05
            candidates = th.cat((current_noise, anchor_noise.repeat(k // 2, 1) + local_eps), dim=0)
            expanded_obs = observations.repeat(k, 1)
            decoder = self._unscale_noise(candidates)
            base = self._decode_noise_decoder_input(expanded_obs, decoder)
            teacher_heads = tuple(
                value.detach()
                for value in self.qa_base_target(expanded_obs, base)
            )
        # Headwise teacher values reshaped to (k, batch).
        teachers = th.stack(teacher_heads, dim=0)  # (num_heads, k*batch)
        num_heads = teachers.shape[0]
        teachers = teachers.reshape(num_heads, k, batch)
        conservative = teachers.min(dim=0).values  # (k, batch)
        # Student QW evaluated over all k candidates for every observation.
        expanded_obs = observations.repeat(k, 1)  # (k*batch, obs_dim)
        student = tuple(
            head
            for head in self.qw_base(expanded_obs, candidates.detach())
        )
        student_min = (
            th.stack(student, dim=0).min(dim=0).values.reshape(k, batch)
        )  # (k, batch)
        # Pairwise masks: twin-teacher direction agreement AND gap > tau_gap.
        agreements = []
        gaps = []
        for i in range(k):
            for j in range(i + 1, k):
                direction = conservative[i] - conservative[j]
                head_agree = th.all(
                    th.sign(teachers[:, i] - teachers[:, j])
                    == th.sign(direction.unsqueeze(0)),
                    dim=0,
                )
                gap_ok = direction.abs() > self.qw_ranking_tau_gap
                agreements.append((head_agree & gap_ok).float())
                gaps.append(direction)
        mask = th.stack(agreements, dim=1)  # (batch, num_pairs)
        gap_matrix = th.stack(gaps, dim=1)  # (batch, num_pairs)
        # Temperature normalized by the median teacher gap over accepted pairs.
        median_gap = gap_matrix.abs().median().clamp_min(1e-3)
        temperature = median_gap.detach()
        pair_diff = th.stack(
            [
                student_min[i] - student_min[j]
                for i in range(k)
                for j in range(i + 1, k)
            ],
            dim=1,
        )  # (batch, num_pairs)
        signs = th.sign(gap_matrix)
        masked = mask * F.softplus(-signs * pair_diff / temperature)
        denom = mask.sum().clamp_min(1.0)
        return masked.sum() / denom

    def _noise_actor_loss_from_sample(
        self,
        observations: th.Tensor,
        noise_scaled: th.Tensor,
        log_prob_noise: th.Tensor,
        entropy_coefficient: th.Tensor,
    ) -> th.Tensor:
        # Structural invariant: this method calls only QW.  It never decodes
        # DDIM and never calls residual/QA_joint.
        q_values = self._minimum_q(self.qw_base(observations, noise_scaled))
        return (
            entropy_coefficient.detach() * log_prob_noise.reshape(-1, 1)
            - q_values
        ).mean()

    def _noise_actor_loss(
        self,
        replay_data: HierarchyReplayBufferSamples,
        entropy_coefficient: Optional[th.Tensor] = None,
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        noise, log_prob = self.actor.action_log_prob(replay_data.observations)
        alpha = (
            self._current_entropy_coefficient()
            if entropy_coefficient is None
            else entropy_coefficient.detach()
        )
        return (
            self._noise_actor_loss_from_sample(
                replay_data.observations, noise, log_prob, alpha
            ),
            noise,
            log_prob,
        )

    def _residual_actor_loss(
        self,
        replay_data: HierarchyReplayBufferSamples,
        *,
        beta: Optional[float] = None,
    ) -> tuple[th.Tensor, HierarchicalActionOutput]:
        with th.no_grad():
            noise, _ = self.actor.action_log_prob(replay_data.observations)
            decoder = self._unscale_noise(noise)
            base = self._decode_noise_decoder_input(
                replay_data.observations, decoder
            ).detach()
        logits = self.residual_actor.forward_pre_tanh(
            replay_data.observations, noise.detach(), base.detach()
        )
        composition = compose_action(
            base,
            logits,
            self._last_action_beta if beta is None else beta,
            self._exec_action_low_tensor,
            self._exec_action_high_tensor,
            numerical_tolerance=self.numerical_bound_tolerance,
        )
        # Section 10: fold the TRAINING-path clamp activations into the same
        # emergency-clamp counter and violation magnitude that the rollout path
        # updates, so the logged clamp rate reflects the residual critic's view
        # of the composed action, not only the stored rollout action.
        if bool(th.any(composition.emergency_clamp_applied).cpu().item()):
            self.emergency_clamp_count += int(
                composition.emergency_clamp_applied.sum().cpu().item()
            )
        training_violation = float(
            composition.maximum_preclamp_violation.detach().cpu().item()
        )
        self._max_preclamp_violation = max(
            self._max_preclamp_violation, training_violation
        )
        q_values = self._minimum_q(
            self.qa_joint(replay_data.observations, composition.action_exec)
        )
        generated = HierarchicalActionOutput(
            noise_scaled=noise.detach(),
            noise_decoder_input=decoder.detach(),
            action_base=base,
            residual_pre_tanh=logits,
            residual_unit=composition.residual_unit,
            margin_positive=composition.margin_positive,
            margin_negative=composition.margin_negative,
            action_residual_delta=composition.action_residual_delta,
            action_exec_unclamped=composition.action_exec_unclamped,
            action_exec=composition.action_exec,
            emergency_clamp_applied=composition.emergency_clamp_applied,
            maximum_preclamp_violation=composition.maximum_preclamp_violation,
        )
        return -q_values.mean(), generated

    @staticmethod
    def _gradient_norm(parameters: Sequence[th.Tensor]) -> float:
        squared = th.zeros((), dtype=th.float64)
        found = False
        for parameter in parameters:
            if parameter.grad is None:
                continue
            found = True
            squared += parameter.grad.detach().double().square().sum().cpu()
        return float(th.sqrt(squared).item()) if found else 0.0

    def _update_qa_base_once(
        self, replay_data: HierarchyReplayBufferSamples
    ) -> tuple[float, float]:
        # Core V1 data-use invariant: the executed action carried by every
        # batch row must be exactly the action the environment ran for that
        # transition (replay validates this at add-time; re-assert here so an
        # inconsistent batch can never reach the update path).  _qa_base_loss
        # queries at this executed action and its target uses base
        # continuation, so BASE rows (a_exec == a_base) and JOINT rows
        # (a_exec == a_base + beta*delta) are both legal -- they differ only
        # in the continuation.  What is never allowed is querying an action
        # that was not executed.
        if not th.all(replay_data.action_exec == replay_data.actions):
            raise AssertionError(
                "QA_base batch contains a transition whose executed action "
                "action_exec does not match the executed action of the "
                "transition (Bellman-invalid tuple)"
            )
        self.qa_base.set_training_mode(True)
        loss, target, current = self._qa_base_loss(replay_data)
        self.qa_base_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.qa_base_optimizer.step()
        self.qa_base_optimizer_steps += 1
        if (self.qa_base_optimizer_steps - 1) % self.target_update_interval == 0:
            polyak_update(
                self.qa_base.parameters(), self.qa_base_target.parameters(), self.tau
            )
            polyak_update(
                self.qa_base_batch_norm_stats,
                self.qa_base_target_batch_norm_stats,
                1.0,
            )
            self.qa_base_target_updates += 1
        self.qa_base.set_training_mode(False)
        td = th.stack([(value.detach() - target).abs().mean() for value in current]).mean()
        return float(loss.item()), float(td.item())

    def _update_qa_joint_once(
        self, replay_data: HierarchyReplayBufferSamples
    ) -> tuple[float, float]:
        self.qa_joint.set_training_mode(True)
        loss, target, current = self._qa_joint_loss(replay_data)
        self.qa_joint_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.qa_joint_optimizer.step()
        self.qa_joint_optimizer_steps += 1
        self.qa_joint_optimizer_steps_since_clone += 1
        if (self.qa_joint_optimizer_steps - 1) % self.target_update_interval == 0:
            polyak_update(
                self.qa_joint.parameters(), self.qa_joint_target.parameters(), self.tau
            )
            polyak_update(
                self.qa_joint_batch_norm_stats,
                self.qa_joint_target_batch_norm_stats,
                1.0,
            )
            self.qa_joint_target_updates += 1
        self.qa_joint.set_training_mode(False)
        td = th.stack([(value.detach() - target).abs().mean() for value in current]).mean()
        return float(loss.item()), float(td.item())

    def _update_qw_once(
        self, replay_data: HierarchyReplayBufferSamples
    ) -> tuple[float, float]:
        self.qw_base.set_training_mode(True)
        loss, student, teacher = self._qw_base_loss(replay_data)
        self.qw_base_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.qw_base_optimizer.step()
        self.qw_base_optimizer_steps += 1
        self.qw_base.set_training_mode(False)
        mse = th.stack(
            [F.mse_loss(a.detach(), b) for a, b in zip(student, teacher)]
        ).mean()
        return float(loss.item()), float(mse.item())

    def _update_alpha_and_noise_once(
        self, replay_data: HierarchyReplayBufferSamples, *, update_alpha: bool
    ) -> tuple[float, Optional[float], float, float]:
        self.actor.train(True)
        noise, log_prob = self.actor.action_log_prob(replay_data.observations)
        alpha_snapshot = self._current_entropy_coefficient().detach().clone()
        alpha_loss_value: Optional[float] = None
        if update_alpha and self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
            alpha_loss = -(
                self.log_ent_coef
                * (log_prob.detach() + self.target_entropy)
            ).mean()
            self.ent_coef_optimizer.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.ent_coef_optimizer.step()
            self.alpha_optimizer_steps += 1
            alpha_loss_value = float(alpha_loss.item())
            self.ent_coef_optimizer.zero_grad(set_to_none=True)
        # Spec 6.3 / 12.5: the noise loss must not add gradients to any
        # non-actor module.  QW_base parameters are frozen for the FORWARD AND
        # the BACKWARD (freezing only the forward would let the backward
        # accumulate gradients on QW parameters).  Snapshot grad VALUES on
        # unrelated modules before the backward and require them unchanged
        # after: earlier update blocks in this train() call already left grads
        # on qa_base/qw_base, so comparing only grad PRESENCE could not detect a
        # leak that accumulates on top of those existing grads.
        unrelated = {
            "qa_base": self.qa_base,
            "qw_base": self.qw_base,
            "qa_joint": self.qa_joint,
            "residual_actor": self.residual_actor,
        }
        before = {
            name: {
                id(parameter): (
                    parameter.grad.detach().clone()
                    if parameter.grad is not None
                    else None
                )
                for parameter in module.parameters()
            }
            for name, module in unrelated.items()
        }
        self.actor.optimizer.zero_grad(set_to_none=True)
        with _freeze_module_parameters(self.qw_base):
            noise_loss = self._noise_actor_loss_from_sample(
                replay_data.observations,
                noise,
                log_prob,
                alpha_snapshot,
            )
            noise_loss.backward()
        actor_parameters = tuple(self.actor.parameters())
        pre_norm = self._gradient_norm(actor_parameters)
        th.nn.utils.clip_grad_norm_(actor_parameters, self.noise_gradient_max_norm)
        post_norm = self._gradient_norm(actor_parameters)
        self.actor.optimizer.step()
        self.noise_actor_optimizer_steps += 1
        self.noise_policy_version += 1
        self._noise_policy_birth_step = int(self.num_timesteps)
        self.actor.eval()
        for name, module in unrelated.items():
            for parameter in module.parameters():
                pid = id(parameter)
                prior = before[name].get(pid)
                if prior is None and parameter.grad is None:
                    continue
                if prior is None or parameter.grad is None or not th.equal(
                    prior, parameter.grad
                ):
                    raise RuntimeError(
                        f"Noise/alpha update changed gradients on {name} parameters"
                    )
        return float(noise_loss.item()), alpha_loss_value, pre_norm, post_norm

    def _update_residual_once(
        self, replay_data: HierarchyReplayBufferSamples
    ) -> tuple[float, float, float]:
        self.residual_actor.train(True)
        with _freeze_module_parameters(self.qa_joint):
            loss, _ = self._residual_actor_loss(replay_data)
        self.residual_actor_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        parameters = tuple(self.residual_actor.parameters())
        pre_norm = self._gradient_norm(parameters)
        th.nn.utils.clip_grad_norm_(parameters, self.residual_gradient_max_norm)
        post_norm = self._gradient_norm(parameters)
        self.residual_actor_optimizer.step()
        self.residual_actor_optimizer_steps += 1
        self.residual_policy_version += 1
        self._residual_policy_birth_step = int(self.num_timesteps)
        polyak_update(
            self.residual_actor.parameters(),
            self.residual_actor_target.parameters(),
            self.tau,
        )
        self.residual_target_updates += 1
        self.residual_actor.eval()
        _freeze_target(self.residual_actor_target)
        return float(loss.item()), pre_norm, post_norm

    def _branch_ready(self, branch: BranchMode) -> bool:
        return self.replay_buffer.branch_count(branch) >= self.min_branch_replay_transitions

    def _sample_branch(self, branch: BranchMode, batch_size: int) -> HierarchyReplayBufferSamples:
        return self.replay_buffer.sample_branch(
            branch, batch_size, env=self._vec_normalize_env
        )

    def _sample_mixed_branch(
        self,
        primary: BranchMode,
        secondary: BranchMode,
        ratio: float,
        batch_size: int,
    ) -> HierarchyReplayBufferSamples:
        """Sample a batch shared between two lanes at a fixed secondary ratio.

        Cross-lane replay shares *transitions*: a row is valid input to any
        critic whose current-state query uses that row's executed action and
        whose continuation target follows the critic's own policy.  BASE rows
        execute exactly action_base, so they serve QA_base (base continuation)
        and QA_joint (joint continuation) alike.  JOINT rows execute
        action_exec = a_base + beta*delta, so they serve QA_joint directly and
        QA_base ONLY under the valid cross-lane ablation (qa_base_cross_lane),
        where the base critic queries at action_exec with a base continuation.
        The Bellman-invalid use -- querying a JOINT row at its *unexecuted*
        action_base -- is what the QA_base update invariant rejects.  Falls
        back to a pure primary batch when the secondary lane has too few rows.
        """
        if not 0 <= ratio < 1:
            raise ValueError("ratio must lie in [0, 1)")
        if ratio == 0.0 or not self._branch_ready(secondary):
            return self._sample_branch(primary, batch_size)
        primary_count = int(round((1.0 - ratio) * batch_size))
        secondary_count = batch_size - primary_count
        if primary_count <= 0 or secondary_count <= 0:
            return self._sample_branch(primary, batch_size)
        primary_data = self._sample_branch(primary, primary_count)
        secondary_data = self._sample_branch(secondary, secondary_count)
        return type(primary_data)(
            *(
                th.cat((a, b), dim=0)
                for a, b in zip(primary_data, secondary_data)
            )
        )

    def _clear_all_gradients(self) -> None:
        for optimizer in (
            self.actor.optimizer,
            self.qa_base_optimizer,
            self.qw_base_optimizer,
            self.qa_joint_optimizer,
            self.residual_actor_optimizer,
            self.ent_coef_optimizer,
        ):
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        """Execute the frozen B/R/J block profile on branch-filtered replay."""

        del gradient_steps  # The phase profile is the authoritative update count.
        self._freeze_all_targets_and_diffusion()
        batch_start = int(
            self._last_action_batch_start
            if self._collected_phase_since_train is not None
            else self.num_timesteps
        )
        phase = self.hierarchy_schedule.phase_at(batch_start)
        profile: UpdateProfile = self.hierarchy_schedule.update_profile_at(batch_start)
        self._last_action_beta = self.hierarchy_schedule.beta_at(batch_start)
        if phase != HierarchyPhase.BASE:
            self._ensure_phase_activation()
        for name, count in profile.as_dict().items():
            self.requested_optimizer_steps[name] += int(count)

        # The common schedule owns the inherited DSRL learning rates.  The
        # independent joint/residual branches have explicit frozen-spec rates
        # and must not be silently overwritten by the global schedule.
        scheduled_optimizers = [
            self.actor.optimizer,
            self.qa_base_optimizer,
            self.qw_base_optimizer,
        ]
        if self.ent_coef_optimizer is not None:
            scheduled_optimizers.append(self.ent_coef_optimizer)
        self._update_learning_rate(scheduled_optimizers)
        self._clear_all_gradients()

        metrics: dict[str, list[float]] = {
            "qa_base_loss": [],
            "qa_base_td": [],
            "qa_joint_loss": [],
            "qa_joint_td": [],
            "qw_base_loss": [],
            "qw_base_mse": [],
            "noise_actor_loss": [],
            "alpha_loss": [],
            "residual_actor_loss": [],
            "noise_grad_pre": [],
            "noise_grad_post": [],
            "residual_grad_pre": [],
            "residual_grad_post": [],
        }
        base_ready = self._branch_ready(BranchMode.BASE)
        joint_ready = self._branch_ready(BranchMode.JOINT)
        if not base_ready and (
            profile.qa_base or profile.qw_base or profile.noise_actor
        ):
            self.base_block_skips += 1
        # Co-training: during Phase B the QA_joint block is shadow-trained on
        # BASE transitions (at beta=0 its valid action IS the base action), so
        # it needs base rows, not joint rows.  Only the residual block and the
        # R-phase QA_joint block actually consume JOINT transitions.  The
        # residual block is additionally gated on the beta floor: its gradient
        # is proportional to beta, so during the R-start hold (beta=0) it is
        # structurally a no-op and is skipped.
        shadow_in_b = self.qa_joint_shadow_in_b and phase == HierarchyPhase.BASE
        residual_on = self._last_action_beta >= self.hierarchy_schedule.beta_floor
        residual_count = profile.residual_actor if residual_on else 0
        joint_block_requires_joint = (
            (profile.qa_joint > 0 and not shadow_in_b) or residual_count > 0
        )
        if not joint_ready and joint_block_requires_joint:
            self.joint_block_skips += 1

        if base_ready:
            for _ in range(profile.qa_base):
                # QA_base Bellman tuple (s, a, r, s') must query at the action
                # that actually produced (r, s'); _qa_base_loss uses the
                # transition's executed action and a base continuation.  Under
                # the default strict lane (qa_base_cross_lane=False) that is a
                # pure-BASE batch.  Under the valid cross-lane ablation
                # (qa_base_cross_lane=True) JOINT rows are additionally mixed
                # in, still queried at action_exec -- "shared transitions,
                # separated continuation".
                if self.qa_base_cross_lane:
                    source = self._sample_mixed_branch(
                        BranchMode.BASE,
                        BranchMode.JOINT,
                        self.cross_lane_ratio,
                        batch_size,
                    )
                else:
                    source = self._sample_branch(BranchMode.BASE, batch_size)
                loss, td = self._update_qa_base_once(source)
                metrics["qa_base_loss"].append(loss)
                metrics["qa_base_td"].append(td)
        qa_joint_ready = base_ready if shadow_in_b else joint_ready
        if qa_joint_ready:
            for _ in range(profile.qa_joint):
                if shadow_in_b:
                    source = self._sample_branch(BranchMode.BASE, batch_size)
                else:
                    source = self._sample_mixed_branch(
                        BranchMode.JOINT,
                        BranchMode.BASE,
                        self.cross_lane_ratio,
                        batch_size,
                    )
                loss, td = self._update_qa_joint_once(source)
                metrics["qa_joint_loss"].append(loss)
                metrics["qa_joint_td"].append(td)
        if base_ready:
            for _ in range(profile.qw_base):
                loss, mse = self._update_qw_once(
                    self._sample_branch(BranchMode.BASE, batch_size)
                )
                metrics["qw_base_loss"].append(loss)
                metrics["qw_base_mse"].append(mse)

        if profile.alpha not in (0, profile.noise_actor):
            raise RuntimeError("Core V1 requires paired alpha/noise update counts")
        if base_ready:
            for _ in range(profile.noise_actor):
                noise_loss, alpha_loss, pre, post = self._update_alpha_and_noise_once(
                    self._sample_branch(BranchMode.BASE, batch_size),
                    update_alpha=profile.alpha > 0,
                )
                metrics["noise_actor_loss"].append(noise_loss)
                metrics["noise_grad_pre"].append(pre)
                metrics["noise_grad_post"].append(post)
                if alpha_loss is not None:
                    metrics["alpha_loss"].append(alpha_loss)
        if joint_ready and residual_count > 0:
            for _ in range(residual_count):
                loss, pre, post = self._update_residual_once(
                    self._sample_branch(BranchMode.JOINT, batch_size)
                )
                metrics["residual_actor_loss"].append(loss)
                metrics["residual_grad_pre"].append(pre)
                metrics["residual_grad_post"].append(post)

        self.hierarchy_train_calls += 1
        self._n_updates += sum(profile.as_dict().values())
        self._collected_phase_since_train = None
        self._clear_all_gradients()
        self.set_inference_mode()
        # Complete every B update before cloning and before the first R action.
        if self.num_timesteps >= self.hierarchy_schedule.phase_r_start:
            self._activate_joint_phase(
                boundary_step=self.hierarchy_schedule.phase_r_start
            )

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/hierarchy_phase", int(phase))
        self.logger.record("train/beta", self._last_action_beta)
        base_count = self.replay_buffer.branch_count(BranchMode.BASE)
        joint_count = self.replay_buffer.branch_count(BranchMode.JOINT)
        self.logger.record("train/replay_base_count", base_count)
        self.logger.record("train/replay_joint_count", joint_count)
        self.logger.record("train/ent_coef", self._current_entropy_coefficient().item())
        # Section 10 required metrics: realized lane ratio and effective UTD.
        total_transitions = max(1, int(base_count) + int(joint_count))
        self.logger.record(
            "train/realized_joint_lane_ratio",
            float(joint_count) / float(total_transitions),
        )
        self.logger.record(
            "train/effective_utd_base",
            float(self.qa_base_optimizer_steps) / max(1, int(base_count)),
        )
        self.logger.record(
            "train/effective_utd_joint",
            float(self.qa_joint_optimizer_steps) / max(1, int(joint_count)),
        )
        counter_names = (
            "qa_base_optimizer_steps",
            "qa_joint_optimizer_steps",
            "qw_base_optimizer_steps",
            "noise_actor_optimizer_steps",
            "alpha_optimizer_steps",
            "residual_actor_optimizer_steps",
            "qa_base_target_updates",
            "qa_joint_target_updates",
            "residual_target_updates",
            "noise_policy_version",
            "residual_policy_version",
            "qa_joint_generation",
            "base_block_skips",
            "joint_block_skips",
            "emergency_clamp_count",
        )
        for name in counter_names:
            self.logger.record(f"train/{name}", getattr(self, name))
        self.logger.record(
            "train/max_preclamp_violation", self._max_preclamp_violation
        )
        self._max_preclamp_violation = 0.0
        # Section 10 "policy age": chunk transitions since the current noise /
        # residual policy version was created.
        self.logger.record(
            "train/noise_policy_age",
            max(0, int(self.num_timesteps) - int(self._noise_policy_birth_step)),
        )
        self.logger.record(
            "train/residual_policy_age",
            max(0, int(self.num_timesteps) - int(self._residual_policy_birth_step)),
        )
        for name, values in metrics.items():
            if values:
                self.logger.record(f"train/{name}", float(np.mean(values)))
        # Wire the previously-dead Section 10 diagnostics (headwise Q means,
        # twin disagreements, headwise same-QA_joint delta-Q, residual
        # logits/unit/delta norms and saturation, margins, emergency clamp).
        if (
            self.diagnostics_interval_updates > 0
            and self.hierarchy_train_calls % self.diagnostics_interval_updates == 0
        ):
            diag_base = self._sample_branch(BranchMode.BASE, self.batch_size) if base_ready else None
            diag_joint = self._sample_branch(BranchMode.JOINT, self.batch_size) if joint_ready else None
            for name, value in self._training_diagnostics(diag_base, diag_joint).items():
                self.logger.record(name, value)
            # Spec 10 QW ranking diagnostic: held-out BASE observations, an
            # isolated RNG stream, and current/reference/perturbed candidates.
            # No ranking term enters any loss; RNG and module modes are
            # restored exactly inside _ranking_diagnostics.
            for name, value in self._ranking_diagnostics(diag_base).items():
                self.logger.record(name, value)

    @th.no_grad()
    def _training_diagnostics(
        self,
        base_data: Optional[HierarchyReplayBufferSamples],
        joint_data: Optional[HierarchyReplayBufferSamples],
    ) -> dict[str, float]:
        diagnostics: dict[str, float] = {}
        if base_data is not None:
            noise, log_prob = self.actor.action_log_prob(base_data.observations)
            decoder = self._unscale_noise(noise)
            base = self._decode_noise_decoder_input(base_data.observations, decoder)
            qa_teacher = self.qa_base_target(base_data.observations, base)
            qw = self.qw_base(base_data.observations, noise)
            diagnostics.update(
                {
                    "diagnostics/noise_scaled_l2": float(
                        th.linalg.vector_norm(noise, dim=1).mean().item()
                    ),
                    "diagnostics/noise_mean_abs": float(
                        noise.abs().mean().item()
                    ),
                    "diagnostics/noise_std": float(noise.std().item()),
                    "diagnostics/noise_log_prob": float(log_prob.mean().item()),
                    "diagnostics/noise_entropy": float(
                        (-log_prob).mean().item()
                    ),
                    "diagnostics/qw_teacher_disagreement": float(
                        th.stack(
                            [F.mse_loss(a, b) for a, b in zip(qw, qa_teacher)]
                        ).mean().item()
                    ),
                    "diagnostics/qa_base_twin_disagreement": float(
                        (qa_teacher[0] - qa_teacher[1]).abs().mean().item()
                    ),
                    "diagnostics/qa_base_head0_mean": float(
                        qa_teacher[0].mean().item()
                    ),
                    "diagnostics/qa_base_head1_mean": float(
                        qa_teacher[1].mean().item()
                    ),
                    "diagnostics/qa_base_head0_std": float(
                        qa_teacher[0].std().item()
                    ),
                    "diagnostics/qa_base_head1_std": float(
                        qa_teacher[1].std().item()
                    ),
                    "diagnostics/qa_base_target_drift": float(
                        th.stack(
                            [
                                (a - b).abs().mean()
                                for a, b in zip(
                                    self.qa_base.parameters(),
                                    self.qa_base_target.parameters(),
                                )
                            ]
                        ).mean().item()
                    ),
                }
            )
        if joint_data is not None:
            noise, _ = self.actor.action_log_prob(joint_data.observations)
            generated = self._generate_hierarchical_action(
                joint_data.observations, noise, beta=self._last_action_beta
            )
            q_exec = self.qa_joint(
                joint_data.observations, generated.action_exec
            )
            q_base = self.qa_joint(
                joint_data.observations, generated.action_base
            )
            diagnostics.update(
                {
                    "diagnostics/residual_pre_tanh_l2": float(
                        th.linalg.vector_norm(
                            generated.residual_pre_tanh, dim=1
                        ).mean().item()
                    ),
                    "diagnostics/residual_unit_mean_abs": float(
                        generated.residual_unit.abs().mean().item()
                    ),
                    "diagnostics/residual_tanh_saturation_fraction": float(
                        (generated.residual_unit.abs() >= 0.99).float().mean().item()
                    ),
                    "diagnostics/action_residual_delta_l2": float(
                        th.linalg.vector_norm(
                            generated.action_residual_delta, dim=1
                        ).mean().item()
                    ),
                    "diagnostics/margin_positive_min": float(
                        generated.margin_positive.min().item()
                    ),
                    "diagnostics/margin_negative_min": float(
                        generated.margin_negative.min().item()
                    ),
                    "diagnostics/emergency_clamp_fraction": float(
                        generated.emergency_clamp_applied.float().mean().item()
                    ),
                    "diagnostics/qa_joint_delta_head0": float(
                        (q_exec[0] - q_base[0]).mean().item()
                    ),
                    "diagnostics/qa_joint_delta_head1": float(
                        (q_exec[1] - q_base[1]).mean().item()
                    ),
                    "diagnostics/qa_joint_twin_disagreement": float(
                        (q_exec[0] - q_exec[1]).abs().mean().item()
                    ),
                    "diagnostics/qa_joint_head0_mean": float(
                        q_exec[0].mean().item()
                    ),
                    "diagnostics/qa_joint_head1_mean": float(
                        q_exec[1].mean().item()
                    ),
                    "diagnostics/qa_joint_head0_std": float(
                        q_exec[0].std().item()
                    ),
                    "diagnostics/qa_joint_head1_std": float(
                        q_exec[1].std().item()
                    ),
                    "diagnostics/qa_joint_target_drift": float(
                        th.stack(
                            [
                                (a - b).abs().mean()
                                for a, b in zip(
                                    self.qa_joint.parameters(),
                                    self.qa_joint_target.parameters(),
                                )
                            ]
                        ).mean().item()
                    ),
                }
            )
        # Section 10: BASE and JOINT episode returns, lengths, early-fall
        # rates.  Read-and-reset the closed-episode windows accumulated since
        # the previous diagnostics dump.
        diagnostics.update(self._episode_stat_diagnostics())
        return diagnostics

    def _record_closed_episode(
        self,
        branch: int,
        episode_return: float,
        episode_length: int,
        early_fall: bool,
    ) -> None:
        stats = self._closed_episode_stats[int(branch)]
        stats["return_sum"] += float(episode_return)
        stats["return_sq_sum"] += float(episode_return) ** 2.0
        stats["length_sum"] += float(episode_length)
        stats["count"] += 1.0
        if early_fall:
            stats["early_fall_count"] += 1.0

    def _episode_stat_diagnostics(self) -> dict[str, float]:
        """Read and reset the per-branch closed-episode windows (Section 10)."""
        diagnostics: dict[str, float] = {}
        for branch_label, branch in (
            ("base", int(BranchMode.BASE)),
            ("joint", int(BranchMode.JOINT)),
        ):
            stats = self._closed_episode_stats[branch]
            count = stats["count"]
            if count > 0.0:
                mean_return = stats["return_sum"] / count
                variance = max(
                    0.0,
                    stats["return_sq_sum"] / count - mean_return * mean_return,
                )
                diagnostics[
                    f"diagnostics/episode_return_{branch_label}_mean"
                ] = float(mean_return)
                diagnostics[
                    f"diagnostics/episode_return_{branch_label}_std"
                ] = float(math.sqrt(variance))
                diagnostics[
                    f"diagnostics/episode_length_{branch_label}_mean"
                ] = float(stats["length_sum"] / count)
                diagnostics[
                    f"diagnostics/episode_early_fall_rate_{branch_label}"
                ] = float(stats["early_fall_count"] / count)
                diagnostics[
                    f"diagnostics/episode_count_{branch_label}"
                ] = float(count)
            for key in stats:
                stats[key] = 0.0
        return diagnostics

    def _capture_rng_state(self) -> dict[str, Any]:
        state: dict[str, Any] = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": th.random.get_rng_state(),
        }
        if th.cuda.is_available():
            state["torch_cuda"] = th.cuda.get_rng_state_all()
        else:
            state["torch_cuda"] = None
        return state

    def _restore_rng_state(self, state: Mapping[str, Any]) -> None:
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        th.random.set_rng_state(state["torch_cpu"])
        cuda_state = state.get("torch_cuda")
        if cuda_state is not None:
            if not th.cuda.is_available():
                raise RuntimeError(
                    "Saved CUDA RNG state cannot be restored without CUDA"
                )
            th.cuda.set_rng_state_all(cuda_state)

    def _all_modules(self) -> list[nn.Module]:
        modules: list[nn.Module] = []
        seen: set[int] = set()
        for root in (
            self.policy,
            self.qa_base,
            self.qa_base_target,
            self.qw_base,
            self.qa_joint,
            self.qa_joint_target,
            self.residual_actor,
            self.residual_actor_target,
            self.reference_noise_actor,
            self.diffusion_policy,
        ):
            if root is None or not isinstance(root, nn.Module):
                continue
            for module in root.modules():
                if id(module) not in seen:
                    seen.add(id(module))
                    modules.append(module)
        return modules

    def _snapshot_module_modes(self) -> list[tuple[nn.Module, bool]]:
        return [(module, module.training) for module in self._all_modules()]

    def _restore_module_modes(
        self, snapshot: Sequence[tuple[nn.Module, bool]]
    ) -> None:
        # Set the flag directly (not module.train()) so restoring one module
        # cannot clobber another already-restored module's mode.
        for module, training in snapshot:
            module.training = bool(training)

    @th.no_grad()
    def _ranking_diagnostics(
        self,
        base_data: Optional[HierarchyReplayBufferSamples],
    ) -> dict[str, float]:
        """Spec 10 "QW ranking diagnostic only".

        Held-out BASE observations, an isolated RNG stream, and candidates
        from the current noise actor, the reference noise actor, and small
        local perturbations.  Reports teacher-twin head ordering agreement,
        QW-versus-teacher Spearman/Kendall per head, pairwise preference
        accuracy, top-1 agreement, the teacher top-1 gap distribution, and
        ranking metrics bucketed by policy age.

        No ranking term enters any loss.  Global Python/NumPy/Torch CPU/CUDA
        RNG and all module modes are restored exactly before returning.
        """
        if base_data is None or int(base_data.observations.shape[0]) == 0:
            return {}
        rng_state = self._capture_rng_state()
        mode_snapshot = self._snapshot_module_modes()
        try:
            return self._ranking_diagnostics_inner(base_data)
        finally:
            self._restore_module_modes(mode_snapshot)
            self._restore_rng_state(rng_state)

    @th.no_grad()
    def _ranking_diagnostics_inner(
        self,
        base_data: HierarchyReplayBufferSamples,
    ) -> dict[str, float]:
        n_obs = min(
            int(self.ranking_diag_n_obs), int(base_data.observations.shape[0])
        )
        obs = base_data.observations[:n_obs].to(self.device)
        n_perturb = 2
        generator = th.Generator(device=obs.device)
        generator.manual_seed(self.ranking_diag_seed)

        current_mean, current_log_std, current_kwargs = (
            self.actor.get_action_dist_params(obs)
        )
        reference_mean, reference_log_std, reference_kwargs = (
            self.reference_noise_actor.get_action_dist_params(obs)
        )
        if current_kwargs or reference_kwargs:
            raise RuntimeError(
                "Ranking diagnostic does not support SDE noise actors"
            )
        current_std = current_log_std.exp()
        reference_std = reference_log_std.exp()

        z_current = th.randn(
            n_obs,
            1,
            self.action_dim_flat,
            generator=generator,
            device=obs.device,
            dtype=obs.dtype,
        )
        z_reference = th.randn(
            n_obs,
            1,
            self.action_dim_flat,
            generator=generator,
            device=obs.device,
            dtype=obs.dtype,
        )
        z_perturbations = th.randn(
            n_obs,
            n_perturb,
            self.action_dim_flat,
            generator=generator,
            device=obs.device,
            dtype=obs.dtype,
        )
        gaussians = [
            current_mean.unsqueeze(1),
            current_mean.unsqueeze(1) + current_std.unsqueeze(1) * z_current,
            reference_mean.unsqueeze(1),
            reference_mean.unsqueeze(1) + reference_std.unsqueeze(1) * z_reference,
            current_mean.unsqueeze(1) + current_std.unsqueeze(1) * z_perturbations,
        ]
        candidate_noise = th.tanh(th.cat(gaussians, dim=1))
        n_candidates = int(candidate_noise.shape[1])
        flat_noise = candidate_noise.reshape(
            n_obs * n_candidates, self.action_dim_flat
        )
        flat_obs = obs.repeat_interleave(n_candidates, dim=0)
        decoder = self._unscale_noise(flat_noise)
        base = self._decode_noise_decoder_input(flat_obs, decoder)
        teacher_heads = self.qa_base_target(flat_obs, base)
        qw_heads = self.qw_base(flat_obs, flat_noise)
        n_heads = int(len(teacher_heads))
        if len(qw_heads) != n_heads:
            raise RuntimeError("QA_base_target and QW_base disagree on head count")
        teacher = (
            th.stack([head.reshape(-1) for head in teacher_heads], dim=0)
            .reshape(n_heads, n_obs, n_candidates)
            .cpu()
            .numpy()
        )
        qw = (
            th.stack([head.reshape(-1) for head in qw_heads], dim=0)
            .reshape(n_heads, n_obs, n_candidates)
            .cpu()
            .numpy()
        )
        stored_versions = (
            base_data.noise_policy_version[:n_obs]
            .reshape(-1)
            .cpu()
            .numpy()
            .astype(np.int64)
        )
        ages = int(self.noise_policy_version) - stored_versions

        diagnostics: dict[str, float] = {}
        # Teacher-twin head ordering agreement: fraction of candidate pairs
        # where both target-QA_base heads prefer the same candidate.
        twin_agreement: list[float] = []
        for index in range(n_obs):
            t0 = teacher[0, index]
            t1 = teacher[1, index]
            agreements = 0
            pairs = 0
            for j in range(n_candidates):
                for k in range(j + 1, n_candidates):
                    d0 = t0[j] - t0[k]
                    d1 = t1[j] - t1[k]
                    if d0 == 0.0 or d1 == 0.0:
                        continue
                    pairs += 1
                    agreements += int((d0 > 0.0) == (d1 > 0.0))
            if pairs:
                twin_agreement.append(agreements / pairs)
        diagnostics[
            "diagnostics/rank_twin_head_ordering_agreement"
        ] = float(np.mean(twin_agreement) if twin_agreement else 0.0)

        per_head_spearman: dict[int, list[float]] = {}
        per_head_kendall: dict[int, list[float]] = {}
        per_head_pairwise: dict[int, list[float]] = {}
        per_head_top1: dict[int, list[float]] = {}
        for head in range(n_heads):
            per_head_spearman[head] = []
            per_head_kendall[head] = []
            per_head_pairwise[head] = []
            per_head_top1[head] = []
        for index in range(n_obs):
            for head in range(n_heads):
                q_head = qw[head, index]
                teacher_head = teacher[head, index]
                per_head_spearman[head].append(
                    _spearman_correlation(q_head, teacher_head)
                )
                per_head_kendall[head].append(
                    _kendall_tau(q_head, teacher_head)
                )
                per_head_pairwise[head].append(
                    _pairwise_preference_accuracy(q_head, teacher_head)
                )
                per_head_top1[head].append(
                    int(np.argmax(q_head) == np.argmax(teacher_head))
                )
        for head in range(n_heads):
            diagnostics[
                f"diagnostics/rank_qw_teacher_spearman_head{head}"
            ] = float(np.mean(per_head_spearman[head]))
            diagnostics[
                f"diagnostics/rank_qw_teacher_kendall_head{head}"
            ] = float(np.mean(per_head_kendall[head]))
            diagnostics[
                f"diagnostics/rank_pairwise_accuracy_head{head}"
            ] = float(np.mean(per_head_pairwise[head]))
            diagnostics[
                f"diagnostics/rank_top1_agreement_head{head}"
            ] = float(np.mean(per_head_top1[head]))

        # Teacher top-1 gap distribution: mean-over-heads gap between the best
        # and second-best candidates per the target QA_base.
        teacher_gaps: list[float] = []
        for index in range(n_obs):
            teacher_mean = teacher[:, index, :].mean(axis=0)
            sorted_scores = np.sort(teacher_mean)
            gap = (
                sorted_scores[-1] - sorted_scores[-2]
                if n_candidates >= 2
                else 0.0
            )
            teacher_gaps.append(float(gap))
        if teacher_gaps:
            gap_mean, gap_std = _mean_and_std(teacher_gaps)
            diagnostics[
                "diagnostics/rank_teacher_top1_gap_mean"
            ] = gap_mean
            diagnostics[
                "diagnostics/rank_teacher_top1_gap_std"
            ] = gap_std
            diagnostics[
                "diagnostics/rank_teacher_top1_gap_min"
            ] = float(np.min(teacher_gaps))
            diagnostics[
                "diagnostics/rank_teacher_top1_gap_max"
            ] = float(np.max(teacher_gaps))

        # Ranking metrics by policy age: bucket held-out observations by how
        # many noise-policy updates happened since the data was collected.
        diagnostics["diagnostics/rank_policy_age_mean"] = float(
            np.mean(ages)
        )
        diagnostics["diagnostics/rank_policy_age_max"] = float(
            np.max(ages)
        )
        age_buckets = {
            "recent": ages <= 0,
            "stale": (ages > 0) & (ages <= 10),
            "old": ages > 10,
        }
        for label, mask in age_buckets.items():
            if not bool(np.any(mask)):
                continue
            head0_spearman = [
                _spearman_correlation(
                    qw[0, index], teacher[0, index]
                )
                for index in np.flatnonzero(mask)
            ]
            diagnostics[
                f"diagnostics/rank_spearman_by_age_{label}"
            ] = float(np.mean(head0_spearman))
        return diagnostics

    def _reset_optimizers(self) -> None:
        learning_rate = self.lr_schedule(1)
        self.actor.optimizer = self.policy.optimizer_class(
            self.actor.parameters(),
            lr=learning_rate,
            **self.policy.optimizer_kwargs,
        )
        self.noise_actor_optimizer = self.actor.optimizer
        self.qa_base.optimizer = self.policy.optimizer_class(
            self.qa_base.parameters(),
            lr=learning_rate,
            **self.policy.optimizer_kwargs,
        )
        self.qa_base_optimizer = self.qa_base.optimizer
        self.qw_base.optimizer = self.policy.optimizer_class(
            self.qw_base.parameters(),
            lr=learning_rate,
            **self.policy.optimizer_kwargs,
        )
        self.qw_base_optimizer = self.qw_base.optimizer
        self.qa_joint_optimizer = self.policy.optimizer_class(
            self.qa_joint.parameters(),
            lr=self.qa_joint_lr,
            **self.policy.optimizer_kwargs,
        )
        self.qa_joint.optimizer = self.qa_joint_optimizer
        self.residual_actor_optimizer = th.optim.Adam(
            self.residual_actor.parameters(), lr=self.residual_lr
        )
        if self.log_ent_coef is not None:
            self.ent_coef_optimizer = th.optim.Adam(
                [self.log_ent_coef], lr=learning_rate
            )

    def _validate_legacy_model(self, legacy_model: DSRL) -> None:
        if legacy_model.critic_backup_combine_type != "min":
            raise ValueError("Legacy DSRL checkpoint must use min critic backup")
        if (
            legacy_model.diffusion_act_chunk,
            legacy_model.diffusion_act_dim,
        ) != (self.diffusion_act_chunk, self.diffusion_act_dim):
            raise ValueError("Legacy diffusion dimensions do not match")
        if legacy_model.observation_space != self.observation_space:
            raise ValueError("Legacy observation space does not match")
        if legacy_model.action_space != self.action_space:
            raise ValueError("Legacy action space does not match")
        if len(legacy_model.critic.q_networks) != len(self.qa_base.q_networks):
            raise ValueError("Legacy QA head count does not match")
        if len(legacy_model.critic_noise.q_networks) != len(self.qw_base.q_networks):
            raise ValueError("Legacy QW head count does not match")
        # The target is authenticated for structure but is deliberately not copied.
        if set(legacy_model.critic_target.state_dict()) != set(
            legacy_model.critic.state_dict()
        ):
            raise ValueError("Legacy QA target structure does not match online QA")

    def _initialize_from_legacy_model(self, legacy_model: DSRL) -> None:
        self._validate_legacy_model(legacy_model)
        self.actor.load_state_dict(legacy_model.actor.state_dict(), strict=True)
        self.reference_noise_actor.load_state_dict(
            legacy_model.actor.state_dict(), strict=True
        )
        self.qa_base.load_state_dict(legacy_model.critic.state_dict(), strict=True)
        self.qa_base_target.load_state_dict(
            self.qa_base.state_dict(), strict=True
        )
        self.qw_base.load_state_dict(
            legacy_model.critic_noise.state_dict(), strict=True
        )
        self.qa_joint.load_state_dict(self.qa_base.state_dict(), strict=True)
        self.qa_joint_target.load_state_dict(
            self.qa_joint.state_dict(), strict=True
        )
        nn.init.zeros_(self.residual_actor.output_layer.weight)
        nn.init.zeros_(self.residual_actor.output_layer.bias)
        self.residual_actor_target.load_state_dict(
            self.residual_actor.state_dict(), strict=True
        )
        self.target_entropy = float(legacy_model.target_entropy)
        if legacy_model.log_ent_coef is not None:
            if self.log_ent_coef is None:
                raise ValueError("Legacy and hierarchy alpha modes differ")
            with th.no_grad():
                self.log_ent_coef.copy_(legacy_model.log_ent_coef.to(self.device))
        else:
            if self.log_ent_coef is not None:
                raise ValueError("Legacy and hierarchy alpha modes differ")
            with th.no_grad():
                self.ent_coef_tensor.copy_(
                    legacy_model.ent_coef_tensor.to(self.device)
                )
        self._reset_optimizers()
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
        ):
            setattr(self, name, 0)
        self.requested_optimizer_steps = {
            key: 0 for key in self.requested_optimizer_steps
        }
        self.noise_policy_version = 0
        self.residual_policy_version = 0
        self._noise_policy_birth_step = int(self.num_timesteps)
        self._residual_policy_birth_step = int(self.num_timesteps)
        self.qa_joint_generation = 1
        self.qa_joint_optimizer_steps_since_clone = 0
        self.qa_joint_clone_online_step = 0
        self._joint_phase_initialized = True
        self._freeze_all_targets_and_diffusion()
        self._assert_parameter_ownership()
        self.set_inference_mode()

    def initialize_from_legacy_checkpoint(
        self, checkpoint_path: Union[str, Path]
    ) -> "HierarchicalRFSDSRL":
        legacy_model = _LegacyLoadableDSRL.load(
            checkpoint_path,
            env=self.get_env(),
            device=self.device,
            custom_objects={"diffusion_policy": self.diffusion_policy},
            buffer_size=1,
        )
        if legacy_model.buffer_size != 1:
            raise RuntimeError("Legacy load shell allocated a non-minimal replay")
        self._initialize_from_legacy_model(legacy_model)
        return self

    def initialize_from_fresh_frozen_ddim(
        self,
    ) -> "HierarchicalRFSDSRL":
        """Spec 8.1: fresh run from Frozen DDIM only (no legacy DSRL checkpoint).

        The noise actor, QA_base, and QW_base were already built from the seeded
        DSRL constructors by ``_setup_model``.  This method:
          - snapshots the fresh noise actor as the immutable reference;
          - hard-copies online QA_base into QA_base_target (SB3's target is a
            separately initialised network and is NOT a copy at construction);
          - re-asserts the exact-zero residual output layer and re-copies the
            residual target;
          - resets optimizers/counters to zero;
          - defers the QA_base -> QA_joint clone to the Phase R activation
            boundary (spec 8.1) by leaving ``_joint_phase_initialized`` False.
        """

        self.reference_noise_actor.load_state_dict(
            self.actor.state_dict(), strict=True
        )
        self.qa_base_target.load_state_dict(
            self.qa_base.state_dict(), strict=True
        )
        nn.init.zeros_(self.residual_actor.output_layer.weight)
        nn.init.zeros_(self.residual_actor.output_layer.bias)
        self.residual_actor_target.load_state_dict(
            self.residual_actor.state_dict(), strict=True
        )
        self._reset_optimizers()
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
        ):
            setattr(self, name, 0)
        self.requested_optimizer_steps = {
            key: 0 for key in self.requested_optimizer_steps
        }
        self.noise_policy_version = 0
        self.residual_policy_version = 0
        self._noise_policy_birth_step = int(self.num_timesteps)
        self._residual_policy_birth_step = int(self.num_timesteps)
        self.qa_joint_optimizer_steps_since_clone = 0
        # Phase state is intentionally NOT reset here: if this schedule has a
        # nonzero Phase B, _joint_phase_initialized stays False so the clone is
        # deferred to the B-to-R boundary; if Phase B is zero, _setup_model has
        # already activated the joint phase at boundary 0.
        self._freeze_all_targets_and_diffusion()
        self._assert_parameter_ownership()
        self.set_inference_mode()
        return self

    @staticmethod
    def _module_state_hash(module: nn.Module) -> str:
        digest = hashlib.sha256()
        for name, tensor in sorted(module.state_dict().items()):
            digest.update(name.encode())
            digest.update(np.asarray(tensor.detach().cpu()).tobytes())
        return digest.hexdigest()

    def diffusion_state_hash(self) -> str:
        module = (
            self.diffusion_policy
            if isinstance(self.diffusion_policy, nn.Module)
            else getattr(self.diffusion_policy, "base_policy", None)
        )
        if not isinstance(module, nn.Module):
            return "non_module_frozen_wrapper"
        return self._module_state_hash(module)

    def _excluded_save_params(self) -> list[str]:
        return super()._excluded_save_params() + [
            "noise_actor",
            "qa_base",
            "qa_base_target",
            "qw_base",
            "critic_noise",
            "qa_joint",
            "qa_joint_target",
            "residual_actor",
            "residual_actor_target",
            "reference_noise_actor",
            "noise_actor_optimizer",
            "qa_base_optimizer",
            "qw_base_optimizer",
            "qa_joint_optimizer",
            "residual_actor_optimizer",
            "diffusion_policy",
            "_noise_action_low_tensor",
            "_noise_action_high_tensor",
            "_exec_action_low_tensor",
            "_exec_action_high_tensor",
            "_pending_rollout_metadata",
            "qa_base_batch_norm_stats",
            "qa_base_target_batch_norm_stats",
            "qa_joint_batch_norm_stats",
            "qa_joint_target_batch_norm_stats",
        ]

    def _get_torch_save_params(self) -> tuple[list[str], list[str]]:
        state_dicts = [
            "policy",
            "actor.optimizer",
            "critic.optimizer",
            "qw_base",
            "qw_base.optimizer",
            "qa_joint",
            "qa_joint.optimizer",
            "qa_joint_target",
            "residual_actor",
            "residual_actor_optimizer",
            "residual_actor_target",
            "reference_noise_actor",
        ]
        if self.ent_coef_optimizer is not None:
            state_dicts.append("ent_coef_optimizer")
            variables = ["log_ent_coef"]
        else:
            variables = ["ent_coef_tensor"]
        return state_dicts, variables

    def save(self, *args: Any, **kwargs: Any) -> None:
        if self._pending_rollout_metadata is not None:
            raise RuntimeError("Model save is only certified at a transition boundary")
        super().save(*args, **kwargs)

    def save_replay_buffer(self, path: Union[str, Path]) -> None:
        if self._pending_rollout_metadata is not None:
            raise RuntimeError("Replay save is only certified at a transition boundary")
        if not isinstance(self.replay_buffer, HierarchyTaggedReplayBuffer):
            raise TypeError("Cannot save a non-tagged hierarchy replay")
        super().save_replay_buffer(path)

    def load_replay_buffer(
        self,
        path: Union[str, Path],
        truncate_last_traj: bool = True,
    ) -> None:
        super().load_replay_buffer(path, truncate_last_traj=truncate_last_traj)
        if not isinstance(self.replay_buffer, HierarchyTaggedReplayBuffer):
            raise ValueError("Loaded replay is not hierarchy_tagged_replay_v1")
        if self.replay_buffer.schema_version != SCHEMA_VERSION:
            raise ValueError("Loaded hierarchy replay schema does not match")
        self.replay_buffer.rebuild_branch_counts()

    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        *args: Any,
        custom_objects: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> "HierarchicalRFSDSRL":
        # The authenticated DDIM is deliberately excluded from the archive.
        # Preserve the project's established custom_objects loading call while
        # forwarding that external artifact into the new-model shell.
        if "diffusion_policy" not in kwargs and custom_objects is not None:
            if "diffusion_policy" in custom_objects:
                kwargs["diffusion_policy"] = custom_objects["diffusion_policy"]
        if kwargs.get("diffusion_policy") is None:
            raise ValueError(
                "Loading three-critic Core V1 requires the authenticated "
                "diffusion_policy artifact"
            )
        return super().load(
            path,
            *args,
            custom_objects=custom_objects,
            **kwargs,
        )

    @staticmethod
    def derive_reset_seed(
        train_env_seed: int,
        environment_discontinuity_count: int,
        environment_id: int,
    ) -> int:
        text = (
            "three_critic_reset_v1:"
            f"{int(train_env_seed)}:{int(environment_discontinuity_count)}:"
            f"{int(environment_id)}"
        )
        return int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "big")

    def prepare_reset_boundary_resume(self, *, train_env_seed: int) -> list[int]:
        """Discard abandoned simulator episode metadata after learner resume."""

        self._pending_rollout_metadata = None
        if isinstance(self.replay_buffer, HierarchyTaggedReplayBuffer):
            self.replay_buffer.clear_staged_metadata()
        self.environment_discontinuity_count += 1
        seeds = [
            self.derive_reset_seed(
                train_env_seed, self.environment_discontinuity_count, env_id
            )
            for env_id in range(self.n_envs)
        ]
        self._active_branch_mode.fill(-1)
        self._active_episode_id.fill(-1)
        self._chunk_index_in_episode.fill(0)
        self._last_obs = None
        self._last_original_obs = None
        # Abandoned mid-episode Section 10 accumulators are equally invalid
        # after the environment discontinuity: the in-flight episode's
        # return/length/branch never complete, and reusing their partial sums
        # as if they belonged to the freshly-reset episode would corrupt the
        # per-branch episode statistics.
        self._episode_return_accum[:] = 0.0
        self._episode_length_accum[:] = 0
        self._episode_branch_accum[:] = -1
        return seeds

    def learn(
        self,
        total_timesteps: int,
        *args: Any,
        reset_num_timesteps: bool = True,
        **kwargs: Any,
    ) -> "HierarchicalRFSDSRL":
        if reset_num_timesteps:
            self.hierarchy_schedule.validate_budget(total_timesteps)
        else:
            remaining = self.hierarchy_schedule.total_steps - self.num_timesteps
            if int(total_timesteps) != int(remaining):
                raise ValueError(
                    "Resume total_timesteps must equal the remaining hierarchy budget"
                )
        return super().learn(
            total_timesteps,
            *args,
            reset_num_timesteps=reset_num_timesteps,
            **kwargs,
        )
