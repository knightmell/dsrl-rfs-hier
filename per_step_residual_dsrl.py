"""Primitive-step residual correction over a frozen DSRL action-chunk planner.

This module deliberately contains no diffusion/noise actor.  The frozen DSRL
planner is part of the environment-side state generator.  The learner receives
the current normalized locomotion observation, current primitive base action,
chunk phase and cached base-noise log probability, and learns only a
deterministic residual plus a twin action critic.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, ClassVar, Iterator, NamedTuple, Optional, Sequence, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces
from torch import nn
from torch.nn import functional as F

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy, ContinuousCritic
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import get_parameters_by_name, polyak_update
from stable_baselines3.sac.policies import CnnPolicy, MlpPolicy, MultiInputPolicy, SACPolicy


SelfPerStepResidualDSRL = TypeVar(
    "SelfPerStepResidualDSRL",
    bound="PerStepResidualDSRL",
)


class PrimitiveResidualOutput(NamedTuple):
    action_base: th.Tensor
    residual_pre_tanh: th.Tensor
    residual_unit: th.Tensor
    action_half_range: th.Tensor
    action_residual_delta: th.Tensor
    action_pre_clip: th.Tensor
    action_exec: th.Tensor
    action_exec_scaled: th.Tensor


@contextmanager
def freeze_parameters(module: nn.Module) -> Iterator[None]:
    states = [parameter.requires_grad for parameter in module.parameters()]
    try:
        for parameter in module.parameters():
            parameter.requires_grad_(False)
        yield
    finally:
        for parameter, requires_grad in zip(module.parameters(), states):
            parameter.requires_grad_(requires_grad)


class PrimitiveResidualActor(nn.Module):
    """Zero-initialized deterministic residual MLP."""

    def __init__(
        self,
        *,
        input_dim: int,
        action_dim: int,
        net_arch: Sequence[int] = (128, 128),
    ) -> None:
        super().__init__()
        if input_dim <= 0 or action_dim <= 0:
            raise ValueError("Residual actor dimensions must be positive")
        if not net_arch or any(int(width) <= 0 for width in net_arch):
            raise ValueError("Residual actor net_arch must contain positive widths")
        layers: list[nn.Module] = []
        previous = int(input_dim)
        for width_value in net_arch:
            width = int(width_value)
            layers.extend(
                (
                    nn.Linear(previous, width),
                    nn.LayerNorm(width),
                    nn.SiLU(),
                )
            )
            previous = width
        self.hidden = nn.Sequential(*layers)
        self.output_layer = nn.Linear(previous, int(action_dim))
        nn.init.zeros_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward_with_pre_tanh(
        self,
        residual_observation: th.Tensor,
    ) -> tuple[th.Tensor, th.Tensor]:
        residual_pre_tanh = self.output_layer(self.hidden(residual_observation))
        return residual_pre_tanh, th.tanh(residual_pre_tanh)

    def forward(self, residual_observation: th.Tensor) -> th.Tensor:
        return self.forward_with_pre_tanh(residual_observation)[1]


class PerStepResidualDSRL(OffPolicyAlgorithm):
    """Frozen-base primitive residual with a deterministic actor and twin QA.

    The environment observation layout is explicit:

    ``[observation | action_base | phase_one_hot | base_noise_log_prob]``.

    The residual actor only consumes the first three fields.  The final scalar
    is critic metadata used to apply the frozen noise entropy exactly once when
    a transition enters a newly planned chunk.
    """

    policy_aliases: ClassVar[dict[str, type[BasePolicy]]] = {
        "MlpPolicy": MlpPolicy,
        "CnnPolicy": CnnPolicy,
        "MultiInputPolicy": MultiInputPolicy,
    }
    policy: SACPolicy
    critic: ContinuousCritic
    critic_target: ContinuousCritic
    residual_actor: PrimitiveResidualActor

    def __init__(
        self,
        policy: Union[str, type[SACPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        buffer_size: int = 1_000_000,
        learning_starts: int = 0,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99 ** 0.25,
        train_freq: Union[int, tuple[int, str]] = 8,
        gradient_steps: int = 20,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[type[ReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        *,
        raw_observation_dim: Optional[int] = None,
        base_action_start: Optional[int] = None,
        phase_start: Optional[int] = None,
        phase_count: Optional[int] = None,
        noise_log_prob_index: Optional[int] = None,
        exec_action_low: Optional[np.ndarray] = None,
        exec_action_high: Optional[np.ndarray] = None,
        noise_entropy_coefficient: Optional[float] = None,
        residual_scale: float = 0.1,
        residual_net_arch: Sequence[int] = (128, 128),
        residual_lr: float = 3e-4,
        residual_penalty_coef: float = 0.0,
        residual_actor_gradient_steps: int = 5,
        diagnostics_interval_train_calls: int = 25,
    ) -> None:
        if not _init_setup_model:
            # SB3 load() constructs a shell, then restores saved constructor
            # data before calling _setup_model().
            raw_observation_dim = 1 if raw_observation_dim is None else raw_observation_dim
            base_action_start = 1 if base_action_start is None else base_action_start
            phase_start = 2 if phase_start is None else phase_start
            phase_count = 1 if phase_count is None else phase_count
            noise_log_prob_index = (
                3 if noise_log_prob_index is None else noise_log_prob_index
            )
            exec_action_low = (
                np.asarray([-1.0], dtype=np.float32)
                if exec_action_low is None
                else exec_action_low
            )
            exec_action_high = (
                np.asarray([1.0], dtype=np.float32)
                if exec_action_high is None
                else exec_action_high
            )
            noise_entropy_coefficient = (
                0.0
                if noise_entropy_coefficient is None
                else noise_entropy_coefficient
            )
        missing = [
            name
            for name, value in (
                ("raw_observation_dim", raw_observation_dim),
                ("base_action_start", base_action_start),
                ("phase_start", phase_start),
                ("phase_count", phase_count),
                ("noise_log_prob_index", noise_log_prob_index),
                ("exec_action_low", exec_action_low),
                ("exec_action_high", exec_action_high),
                ("noise_entropy_coefficient", noise_entropy_coefficient),
            )
            if value is None
        ]
        if missing:
            raise ValueError(f"Missing required constructor values: {missing}")
        self.raw_observation_dim = self._positive_int(
            "raw_observation_dim",
            raw_observation_dim,
        )
        self.base_action_start = self._nonnegative_int(
            "base_action_start",
            base_action_start,
        )
        self.phase_start = self._nonnegative_int("phase_start", phase_start)
        self.phase_count = self._positive_int("phase_count", phase_count)
        self.noise_log_prob_index = self._nonnegative_int(
            "noise_log_prob_index",
            noise_log_prob_index,
        )
        self.exec_action_low = np.asarray(exec_action_low, dtype=np.float32).copy()
        self.exec_action_high = np.asarray(exec_action_high, dtype=np.float32).copy()
        if self.exec_action_low.ndim != 1:
            raise ValueError("Execution bounds must be one-dimensional")
        if self.exec_action_high.shape != self.exec_action_low.shape:
            raise ValueError("Execution low/high bounds must have identical shape")
        if not np.all(np.isfinite(self.exec_action_low)) or not np.all(
            np.isfinite(self.exec_action_high)
        ):
            raise ValueError("Execution bounds must be finite")
        if not np.all(self.exec_action_high > self.exec_action_low):
            raise ValueError("Execution high bounds must exceed low bounds")
        if not np.isfinite(noise_entropy_coefficient) or (
            noise_entropy_coefficient < 0
        ):
            raise ValueError("noise_entropy_coefficient must be finite and non-negative")
        if not np.isfinite(residual_scale) or residual_scale < 0:
            raise ValueError("residual_scale must be finite and non-negative")
        if not np.isfinite(residual_lr) or residual_lr <= 0:
            raise ValueError("residual_lr must be finite and positive")
        if not np.isfinite(residual_penalty_coef) or residual_penalty_coef < 0:
            raise ValueError("residual_penalty_coef must be finite and non-negative")
        self.noise_entropy_coefficient = float(noise_entropy_coefficient)
        self.residual_scale = float(residual_scale)
        self.residual_net_arch = tuple(int(width) for width in residual_net_arch)
        self.residual_lr = float(residual_lr)
        self.residual_penalty_coef = float(residual_penalty_coef)
        self.residual_actor_gradient_steps = self._nonnegative_int(
            "residual_actor_gradient_steps",
            residual_actor_gradient_steps,
        )
        self.diagnostics_interval_train_calls = self._positive_int(
            "diagnostics_interval_train_calls",
            diagnostics_interval_train_calls,
        )

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
            action_noise=action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            optimize_memory_usage=optimize_memory_usage,
            policy_kwargs=policy_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            support_multi_env=True,
            monitor_wrapper=True,
            seed=seed,
            use_sde=False,
            sde_sample_freq=-1,
            use_sde_at_warmup=False,
            sde_support=False,
            supported_action_spaces=(spaces.Box,),
        )
        if _init_setup_model:
            self._setup_model()

    @staticmethod
    def _positive_int(name: str, value: int) -> int:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"{name} must be an integer")
        if int(value) <= 0:
            raise ValueError(f"{name} must be positive")
        return int(value)

    @staticmethod
    def _nonnegative_int(name: str, value: int) -> int:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"{name} must be an integer")
        if int(value) < 0:
            raise ValueError(f"{name} must be non-negative")
        return int(value)

    @property
    def action_dimension(self) -> int:
        return int(self.exec_action_low.shape[0])

    @property
    def base_action_slice(self) -> slice:
        return slice(
            self.base_action_start,
            self.base_action_start + self.action_dimension,
        )

    @property
    def phase_slice(self) -> slice:
        return slice(self.phase_start, self.phase_start + self.phase_count)

    def _setup_model(self) -> None:
        super()._setup_model()
        if not isinstance(self.observation_space, spaces.Box) or (
            len(self.observation_space.shape) != 1
        ):
            raise ValueError("Per-step residual requires a flat Box observation")
        if not isinstance(self.action_space, spaces.Box) or (
            self.action_space.shape != (self.action_dimension,)
        ):
            raise ValueError(
                "Environment action dimension must match execution bounds"
            )
        if not np.array_equal(self.action_space.low, self.exec_action_low) or (
            not np.array_equal(self.action_space.high, self.exec_action_high)
        ):
            raise ValueError("Explicit execution bounds conflict with environment bounds")
        observation_dimension = int(np.prod(self.observation_space.shape))
        if self.base_action_slice.stop > observation_dimension:
            raise ValueError("Base-action slice exceeds observation dimension")
        if self.phase_slice.stop > observation_dimension:
            raise ValueError("Phase slice exceeds observation dimension")
        if self.noise_log_prob_index >= observation_dimension:
            raise ValueError("Noise-log-prob index exceeds observation dimension")
        expected_residual_input = (
            self.raw_observation_dim + self.action_dimension + self.phase_count
        )
        if (
            self.base_action_start != self.raw_observation_dim
            or self.phase_start != self.base_action_slice.stop
            or self.noise_log_prob_index != self.phase_slice.stop
            or observation_dimension != self.noise_log_prob_index + 1
        ):
            raise ValueError(
                "Observation layout must be contiguous: "
                "[observation|action_base|phase_one_hot|noise_log_prob]"
            )
        self.critic = self.policy.critic
        self.critic_target = self.policy.critic_target
        self.batch_norm_stats = get_parameters_by_name(
            self.critic,
            ["running_"],
        )
        self.batch_norm_stats_target = get_parameters_by_name(
            self.critic_target,
            ["running_"],
        )
        # The SAC actor is an implementation shell only; it must never update.
        self.policy.actor.eval()
        self.policy.actor.requires_grad_(False)
        self.residual_actor = PrimitiveResidualActor(
            input_dim=expected_residual_input,
            action_dim=self.action_dimension,
            net_arch=self.residual_net_arch,
        ).to(self.device)
        self.residual_actor_optimizer = th.optim.Adam(
            self.residual_actor.parameters(),
            lr=self.residual_lr,
        )
        self.actor = self.residual_actor
        self._exec_action_low_tensor = th.as_tensor(
            self.exec_action_low,
            device=self.device,
            dtype=th.float32,
        )
        self._exec_action_high_tensor = th.as_tensor(
            self.exec_action_high,
            device=self.device,
            dtype=th.float32,
        )
        for counter_name in (
            "action_critic_optimizer_steps",
            "critic_pretrain_optimizer_steps",
            "residual_actor_optimizer_steps",
            "per_step_train_calls",
        ):
            if not hasattr(self, counter_name):
                setattr(self, counter_name, 0)
        if not hasattr(self, "_last_diagnostics_train_call"):
            self._last_diagnostics_train_call = -self.diagnostics_interval_train_calls
        self.set_inference_mode()

    def pretrain_action_critic(
        self,
        *,
        gradient_steps: int,
        batch_size: Optional[int] = None,
    ) -> list[float]:
        """Fit primitive QA on zero-residual prefill before actor updates.

        The method is only valid before any online/residual update and requires
        the residual output layer to remain exactly zero.  It uses the same
        Bellman loss and target-update cadence as online QA.
        """

        steps = self._positive_int("gradient_steps", gradient_steps)
        effective_batch_size = (
            self.batch_size if batch_size is None else self._positive_int(
                "batch_size",
                batch_size,
            )
        )
        if self.residual_actor_optimizer_steps != 0 or self.per_step_train_calls != 0:
            raise RuntimeError("Critic pretraining must precede residual/online updates")
        if (
            th.count_nonzero(self.residual_actor.output_layer.weight).item() != 0
            or th.count_nonzero(self.residual_actor.output_layer.bias).item() != 0
        ):
            raise RuntimeError("Critic pretraining requires exact zero residual")
        if self.replay_buffer is None or self.replay_buffer.size() == 0:
            raise RuntimeError("Critic pretraining requires populated replay")
        self.policy.actor.eval()
        self.critic.set_training_mode(True)
        self.critic_target.set_training_mode(False)
        self.residual_actor.eval()
        losses: list[float] = []
        for _ in range(steps):
            replay_data = self.replay_buffer.sample(
                effective_batch_size,
                env=self._vec_normalize_env,
            )
            loss = self._action_critic_loss(replay_data)
            losses.append(float(loss.item()))
            self.critic.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.critic.optimizer.step()
            self.action_critic_optimizer_steps += 1
            self.critic_pretrain_optimizer_steps += 1
            polyak_update(
                self.critic.parameters(),
                self.critic_target.parameters(),
                self.tau,
            )
            polyak_update(
                self.batch_norm_stats,
                self.batch_norm_stats_target,
                1.0,
            )
        self.critic.optimizer.zero_grad(set_to_none=True)
        self.residual_actor_optimizer.zero_grad(set_to_none=True)
        self.set_inference_mode()
        return losses

    def _residual_observation(self, observation: th.Tensor) -> th.Tensor:
        if observation.ndim != 2:
            raise ValueError("Residual observation must be batched")
        return th.cat(
            (
                observation[:, : self.raw_observation_dim],
                observation[:, self.base_action_slice],
                observation[:, self.phase_slice],
            ),
            dim=1,
        )

    def _scale_exec_action(self, action_exec: th.Tensor) -> th.Tensor:
        return 2.0 * (
            (action_exec - self._exec_action_low_tensor)
            / (self._exec_action_high_tensor - self._exec_action_low_tensor)
        ) - 1.0

    def _compose(
        self,
        observation: th.Tensor,
        *,
        zero_residual: bool = False,
    ) -> PrimitiveResidualOutput:
        action_base = observation[:, self.base_action_slice].detach()
        if zero_residual:
            residual_pre_tanh = th.zeros_like(action_base)
            residual_unit = th.zeros_like(action_base)
        else:
            residual_pre_tanh, residual_unit = (
                self.residual_actor.forward_with_pre_tanh(
                    self._residual_observation(observation)
                )
            )
        action_half_range = (
            self._exec_action_high_tensor - self._exec_action_low_tensor
        ) / 2.0
        action_residual_delta = (
            self.residual_scale * action_half_range * residual_unit
        )
        action_pre_clip = action_base + action_residual_delta
        action_exec = th.maximum(
            th.minimum(action_pre_clip, self._exec_action_high_tensor),
            self._exec_action_low_tensor,
        )
        return PrimitiveResidualOutput(
            action_base=action_base,
            residual_pre_tanh=residual_pre_tanh,
            residual_unit=residual_unit,
            action_half_range=action_half_range,
            action_residual_delta=action_residual_delta,
            action_pre_clip=action_pre_clip,
            action_exec=action_exec,
            action_exec_scaled=self._scale_exec_action(action_exec),
        )

    def _sample_action(
        self,
        learning_starts: int,
        action_noise: Optional[ActionNoise] = None,
        n_envs: int = 1,
    ) -> tuple[np.ndarray, np.ndarray]:
        if action_noise is not None:
            raise ValueError("Per-step frozen residual does not use action noise")
        if self._last_obs is None or isinstance(self._last_obs, dict):
            raise RuntimeError("Flat _last_obs is required before action sampling")
        observation = th.as_tensor(
            self._last_obs,
            device=self.device,
            dtype=th.float32,
        )
        warmup = self.num_timesteps < learning_starts
        with th.no_grad():
            generated = self._compose(observation, zero_residual=warmup)
        action_exec = generated.action_exec.cpu().numpy()
        buffer_action = generated.action_exec_scaled.cpu().numpy()
        return action_exec, buffer_action

    def predict_with_components(
        self,
        observation: np.ndarray,
        state: Optional[tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = True,
    ) -> tuple[dict[str, np.ndarray], Optional[tuple[np.ndarray, ...]]]:
        del episode_start, deterministic
        observation_array = np.asarray(observation, dtype=np.float32)
        single = observation_array.ndim == 1
        if single:
            observation_array = observation_array[None, :]
        observation_tensor = th.as_tensor(
            observation_array,
            device=self.device,
            dtype=th.float32,
        )
        residual_training = self.residual_actor.training
        self.residual_actor.eval()
        try:
            with th.no_grad():
                generated = self._compose(observation_tensor)
            result = {
                field_name: getattr(generated, field_name).cpu().numpy()
                for field_name in generated._fields
            }
            if single:
                result = {key: value[0] for key, value in result.items()}
            return result, state
        finally:
            self.residual_actor.train(residual_training)

    def predict(
        self,
        observation: np.ndarray,
        state: Optional[tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = True,
    ) -> tuple[np.ndarray, Optional[tuple[np.ndarray, ...]]]:
        components, next_state = self.predict_with_components(
            observation,
            state=state,
            episode_start=episode_start,
            deterministic=deterministic,
        )
        return components["action_exec"], next_state

    @staticmethod
    def _minimum_q(q_values: tuple[th.Tensor, ...]) -> th.Tensor:
        if not q_values:
            raise ValueError("At least one Q head is required")
        return th.min(th.cat(q_values, dim=1), dim=1, keepdim=True).values

    def _next_entropy_term(self, next_observations: th.Tensor) -> th.Tensor:
        next_phase_zero = next_observations[:, self.phase_start : self.phase_start + 1]
        next_noise_log_prob = next_observations[
            :,
            self.noise_log_prob_index : self.noise_log_prob_index + 1,
        ]
        return (
            self.noise_entropy_coefficient
            * next_phase_zero
            * next_noise_log_prob
        )

    def _action_critic_loss(self, replay_data: Any) -> th.Tensor:
        with th.no_grad():
            next_generated = self._compose(replay_data.next_observations)
            next_q = self._minimum_q(
                self.critic_target(
                    replay_data.next_observations,
                    next_generated.action_exec_scaled,
                )
            )
            next_q = next_q - self._next_entropy_term(
                replay_data.next_observations
            )
            target_q = replay_data.rewards + (
                1.0 - replay_data.dones
            ) * self.gamma * next_q
        current_q = self.critic(
            replay_data.observations,
            replay_data.actions,
        )
        return 0.5 * sum(
            F.mse_loss(head, target_q)
            for head in current_q
        )

    def _residual_actor_loss(
        self,
        replay_data: Any,
    ) -> tuple[th.Tensor, PrimitiveResidualOutput]:
        generated = self._compose(replay_data.observations)
        action_q = self._minimum_q(
            self.critic(
                replay_data.observations,
                generated.action_exec_scaled,
            )
        )
        penalty = generated.action_residual_delta.square().mean()
        return (
            -action_q.mean() + self.residual_penalty_coef * penalty,
            generated,
        )

    @th.no_grad()
    def _diagnostics(self, replay_data: Any) -> dict[str, float]:
        generated = self._compose(replay_data.observations)
        base_scaled = self._scale_exec_action(generated.action_base)
        q_exec = self._minimum_q(
            self.critic(replay_data.observations, generated.action_exec_scaled)
        )
        q_base = self._minimum_q(
            self.critic(replay_data.observations, base_scaled)
        )
        delta_norm = th.linalg.vector_norm(
            generated.action_residual_delta,
            dim=1,
        )
        base_norm = th.linalg.vector_norm(generated.action_base, dim=1)
        effective = generated.action_exec - generated.action_base
        return {
            "action_base_l2": base_norm.mean().item(),
            "residual_unit_mean_abs": generated.residual_unit.abs().mean().item(),
            "residual_unit_l2": th.linalg.vector_norm(
                generated.residual_unit,
                dim=1,
            ).mean().item(),
            "residual_tanh_saturation_fraction": (
                generated.residual_unit.abs() >= 0.99
            ).float().mean().item(),
            "action_residual_delta_l2": delta_norm.mean().item(),
            "action_pre_clip_l2": th.linalg.vector_norm(
                generated.action_pre_clip,
                dim=1,
            ).mean().item(),
            "action_exec_l2": th.linalg.vector_norm(
                generated.action_exec,
                dim=1,
            ).mean().item(),
            "residual_to_base_ratio": (
                delta_norm / (base_norm + 1e-6)
            ).mean().item(),
            "clip_fraction": (
                generated.action_pre_clip != generated.action_exec
            ).float().mean().item(),
            "effective_residual_l2": th.linalg.vector_norm(
                effective,
                dim=1,
            ).mean().item(),
            "Q_action(action_exec)-Q_action(action_base)": (
                q_exec - q_base
            ).mean().item(),
        }

    def set_inference_mode(self) -> None:
        self.policy.set_training_mode(False)
        self.policy.actor.eval()
        self.critic.set_training_mode(False)
        self.critic_target.set_training_mode(False)
        self.residual_actor.eval()

    def train(self, gradient_steps: int, batch_size: int = 256) -> None:
        if gradient_steps <= 0:
            raise ValueError("gradient_steps must be positive")
        self.policy.actor.eval()
        self.critic.set_training_mode(True)
        self.critic_target.set_training_mode(False)
        self.residual_actor.eval()
        self._update_learning_rate(
            [self.critic.optimizer, self.residual_actor_optimizer]
        )
        critic_losses: list[float] = []
        residual_losses: list[float] = []
        diagnostic_data = None

        self.critic.optimizer.zero_grad(set_to_none=True)
        self.residual_actor_optimizer.zero_grad(set_to_none=True)
        for gradient_step in range(gradient_steps):
            replay_data = self.replay_buffer.sample(
                batch_size,
                env=self._vec_normalize_env,
            )
            diagnostic_data = replay_data
            critic_loss = self._action_critic_loss(replay_data)
            critic_losses.append(float(critic_loss.item()))
            self.critic.optimizer.zero_grad(set_to_none=True)
            critic_loss.backward()
            self.critic.optimizer.step()
            self.action_critic_optimizer_steps += 1
            if gradient_step % 1 == 0:
                polyak_update(
                    self.critic.parameters(),
                    self.critic_target.parameters(),
                    self.tau,
                )
                polyak_update(
                    self.batch_norm_stats,
                    self.batch_norm_stats_target,
                    1.0,
                )

        self.critic.optimizer.zero_grad(set_to_none=True)
        self.critic.set_training_mode(False)
        self.residual_actor.train(True)
        for _ in range(self.residual_actor_gradient_steps):
            replay_data = self.replay_buffer.sample(
                batch_size,
                env=self._vec_normalize_env,
            )
            diagnostic_data = replay_data
            with freeze_parameters(self.critic):
                residual_loss, _ = self._residual_actor_loss(replay_data)
            residual_losses.append(float(residual_loss.item()))
            self.residual_actor_optimizer.zero_grad(set_to_none=True)
            residual_loss.backward()
            self.residual_actor_optimizer.step()
            self.residual_actor_optimizer_steps += 1

        self.critic.optimizer.zero_grad(set_to_none=True)
        self.residual_actor_optimizer.zero_grad(set_to_none=True)
        self._n_updates += int(gradient_steps)
        self.per_step_train_calls += 1
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record(
            "train/action_critic_loss",
            float(np.mean(critic_losses)),
        )
        if residual_losses:
            self.logger.record(
                "train/residual_actor_loss",
                float(np.mean(residual_losses)),
            )
        self.logger.record(
            "train/action_critic_optimizer_steps",
            self.action_critic_optimizer_steps,
        )
        self.logger.record(
            "train/residual_actor_optimizer_steps",
            self.residual_actor_optimizer_steps,
        )
        self.logger.record("train/per_step_train_calls", self.per_step_train_calls)
        if (
            diagnostic_data is not None
            and self.per_step_train_calls - self._last_diagnostics_train_call
            >= self.diagnostics_interval_train_calls
        ):
            for name, value in self._diagnostics(diagnostic_data).items():
                self.logger.record(name, value)
            self._last_diagnostics_train_call = self.per_step_train_calls
        self.set_inference_mode()

    def _excluded_save_params(self) -> list[str]:
        return super()._excluded_save_params() + [
            "actor",
            "critic",
            "critic_target",
        ]

    def _get_torch_save_params(self) -> tuple[list[str], list[str]]:
        return (
            [
                "policy",
                "critic.optimizer",
                "residual_actor",
                "residual_actor_optimizer",
            ],
            [],
        )

    def learn(
        self: SelfPerStepResidualDSRL,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "per_step_residual",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfPerStepResidualDSRL:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )
