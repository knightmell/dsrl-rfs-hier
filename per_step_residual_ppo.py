"""Per-primitive residual action wrapper and PPO helpers.

The frozen DSRL planner remains responsible for the H=4 base action chunk.
PPO observes the current normalized state, current primitive base action, and
chunk phase, and emits only a unit residual.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3 import PPO
from stable_baselines3.common.utils import explained_variance
from stable_baselines3.common.vec_env import VecEnv, VecEnvWrapper

from per_step_residual_env import FrozenDSRLPrimitiveResidualEnv


@dataclass(frozen=True)
class PPOResidualComposition:
    action_base: np.ndarray
    residual_unit: np.ndarray
    action_half_range: np.ndarray
    action_residual_delta: np.ndarray
    action_pre_clip: np.ndarray
    action_exec: np.ndarray


class PPOResidualUnitEnv(gym.Wrapper):
    """Expose residual-unit actions while executing base + scaled residual."""

    def __init__(
        self,
        environment: FrozenDSRLPrimitiveResidualEnv,
        *,
        residual_scale: float = 0.1,
    ) -> None:
        if not isinstance(environment, FrozenDSRLPrimitiveResidualEnv):
            raise TypeError("PPO residual wrapper requires the audited primitive env")
        super().__init__(environment)
        if not np.isfinite(residual_scale) or residual_scale < 0:
            raise ValueError("residual_scale must be finite and non-negative")
        self.residual_scale = float(residual_scale)
        self.exec_action_low = np.asarray(
            environment.action_space.low,
            dtype=np.float32,
        ).copy()
        self.exec_action_high = np.asarray(
            environment.action_space.high,
            dtype=np.float32,
        ).copy()
        self.action_space = spaces.Box(
            low=-np.ones_like(self.exec_action_low),
            high=np.ones_like(self.exec_action_high),
            dtype=np.float32,
        )

    @property
    def phase(self) -> int:
        return self.env.phase

    @property
    def current_base_action(self) -> np.ndarray:
        return self.env.current_base_action

    def compose(self, residual_unit: np.ndarray) -> PPOResidualComposition:
        residual = np.asarray(residual_unit, dtype=np.float32)
        if residual.shape != self.action_space.shape:
            raise ValueError(
                f"Residual shape {residual.shape} != {self.action_space.shape}"
            )
        residual = np.clip(
            residual,
            self.action_space.low,
            self.action_space.high,
        )
        action_base = self.current_base_action
        action_half_range = (
            self.exec_action_high - self.exec_action_low
        ) / 2.0
        action_residual_delta = (
            self.residual_scale * action_half_range * residual
        )
        action_pre_clip = action_base + action_residual_delta
        action_exec = np.clip(
            action_pre_clip,
            self.exec_action_low,
            self.exec_action_high,
        )
        return PPOResidualComposition(
            action_base=action_base.copy(),
            residual_unit=residual.copy(),
            action_half_range=action_half_range.copy(),
            action_residual_delta=action_residual_delta.copy(),
            action_pre_clip=action_pre_clip.copy(),
            action_exec=action_exec.copy(),
        )

    def step(self, residual_unit: np.ndarray):
        composition = self.compose(residual_unit)
        observation, reward, terminated, truncated, info = self.env.step(
            composition.action_exec
        )
        effective_residual = (
            composition.action_exec - composition.action_base
        )
        result_info = dict(info)
        result_info.update(
            {
                "residual_unit": composition.residual_unit,
                "action_residual_delta": composition.action_residual_delta,
                "action_pre_clip": composition.action_pre_clip,
                "effective_residual": effective_residual,
                "residual_unit_l2": float(
                    np.linalg.norm(composition.residual_unit)
                ),
                "action_residual_delta_l2": float(
                    np.linalg.norm(composition.action_residual_delta)
                ),
                "effective_residual_l2": float(
                    np.linalg.norm(effective_residual)
                ),
                "residual_action_clip_fraction": float(
                    np.mean(
                        np.asarray(residual_unit)
                        != composition.residual_unit
                    )
                ),
                "execution_clip_fraction": float(
                    np.mean(
                        composition.action_pre_clip
                        != composition.action_exec
                    )
                ),
            }
        )
        return observation, reward, terminated, truncated, result_info

    def capture_state(self) -> dict[str, Any]:
        return self.env.capture_state()

    def restore_state(self, state: dict[str, Any]) -> None:
        self.env.restore_state(state)


class PPOTrainingRewardScaleEnv(gym.Wrapper):
    """Scale only rewards placed in PPO's rollout buffer."""

    def __init__(
        self,
        environment: PPOResidualUnitEnv,
        *,
        reward_scale: float,
    ) -> None:
        if not isinstance(environment, PPOResidualUnitEnv):
            raise TypeError("Reward scaling must wrap PPOResidualUnitEnv")
        super().__init__(environment)
        if not np.isfinite(reward_scale) or reward_scale <= 0:
            raise ValueError("reward_scale must be finite and positive")
        self.reward_scale = float(reward_scale)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        result_info = dict(info)
        result_info["unscaled_training_reward"] = float(reward)
        result_info["training_reward_scale"] = self.reward_scale
        return (
            observation,
            float(reward) * self.reward_scale,
            terminated,
            truncated,
            result_info,
        )


class ResiPRewardNormalize(VecEnvWrapper):
    """Match ResiP's immediate-reward variance normalization.

    Unlike SB3 VecNormalize, ResiP divides the current reward by the running
    standard deviation of immediate rewards; it does not normalize discounted
    returns and does not subtract the running mean.
    """

    def __init__(
        self,
        venv: VecEnv,
        *,
        clip_reward: float = 5.0,
        epsilon: float = 1e-4,
    ) -> None:
        super().__init__(venv)
        if clip_reward <= 0 or epsilon <= 0:
            raise ValueError("Reward clip and epsilon must be positive")
        self.clip_reward = float(clip_reward)
        self.reward_mean = 0.0
        self.reward_var = 1.0
        self.reward_count = float(epsilon)

    def reset(self) -> np.ndarray:
        return self.venv.reset()

    def step_wait(self):
        observations, rewards, dones, infos = self.venv.step_wait()
        values = np.asarray(rewards, dtype=np.float64)
        batch_mean = float(values.mean())
        batch_var = float(values.var())
        batch_count = int(values.size)
        delta = batch_mean - self.reward_mean
        total = self.reward_count + batch_count
        updated_mean = self.reward_mean + delta * batch_count / total
        moment_a = self.reward_var * self.reward_count
        moment_b = batch_var * batch_count
        moment_2 = (
            moment_a
            + moment_b
            + delta * delta * self.reward_count * batch_count / total
        )
        self.reward_mean = updated_mean
        self.reward_var = moment_2 / total
        self.reward_count = total
        normalized = np.clip(
            values / np.sqrt(self.reward_var + 1e-8),
            -self.clip_reward,
            self.clip_reward,
        ).astype(np.float32)
        for raw_reward, normalized_reward, info in zip(
            values,
            normalized,
            infos,
        ):
            info["unscaled_training_reward"] = float(raw_reward)
            info["normalized_training_reward"] = float(normalized_reward)
            info["training_reward_running_variance"] = float(self.reward_var)
        return observations, normalized, dones, infos

    def state_dict(self) -> dict[str, float]:
        return {
            "clip_reward": self.clip_reward,
            "reward_mean": self.reward_mean,
            "reward_var": self.reward_var,
            "reward_count": self.reward_count,
        }

    def load_state_dict(self, state: dict[str, float]) -> None:
        if float(state["clip_reward"]) != self.clip_reward:
            raise ValueError("Reward-normalizer clip mismatch")
        self.reward_mean = float(state["reward_mean"])
        self.reward_var = float(state["reward_var"])
        self.reward_count = float(state["reward_count"])


class ResidualObservationExtractor(BaseFeaturesExtractor):
    """Drop cached base-noise log-probability from the PPO state.

    The remaining input is exactly
    [normalized observation | current base action | phase one-hot].
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        *,
        noise_log_prob_index: int,
        clamp_value: float | None = None,
    ) -> None:
        flat_dimension = int(np.prod(observation_space.shape))
        if noise_log_prob_index != flat_dimension - 1:
            raise ValueError("Noise log-probability must be the final observation field")
        self.noise_log_prob_index = int(noise_log_prob_index)
        if clamp_value is not None and clamp_value <= 0:
            raise ValueError("Observation clamp must be positive")
        self.clamp_value = (
            float(clamp_value) if clamp_value is not None else None
        )
        super().__init__(
            observation_space,
            features_dim=flat_dimension - 1,
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        features = observations[:, : self.noise_log_prob_index]
        if self.clamp_value is not None:
            features = torch.clamp(
                features,
                -self.clamp_value,
                self.clamp_value,
            )
        return features


def zero_initialize_ppo_residual_mean(policy: Any) -> None:
    """Make deterministic initial PPO execution exactly zero residual."""

    with torch.no_grad():
        policy.action_net.weight.zero_()
        if policy.action_net.bias is not None:
            policy.action_net.bias.zero_()


class CountingPPO(PPO):
    """PPO with a persistent count of actual optimizer steps."""

    def _setup_model(self) -> None:
        super()._setup_model()
        if not hasattr(self, "ppo_optimizer_steps"):
            self.ppo_optimizer_steps = 0
        optimizer = self.policy.optimizer
        original_step = optimizer.step

        def counted_step(*args: Any, **kwargs: Any):
            result = original_step(*args, **kwargs)
            self.ppo_optimizer_steps += 1
            return result

        optimizer.step = counted_step


class SeparatedClipPPO(CountingPPO):
    """Clip residual-actor and value gradients independently."""

    def __init__(
        self,
        *args: Any,
        actor_max_grad_norm: float = 0.5,
        value_max_grad_norm: float = 0.5,
        value_loss_multiplier: float = 1.0,
        target_kl_multiplier: float = 1.5,
        **kwargs: Any,
    ) -> None:
        if actor_max_grad_norm <= 0 or value_max_grad_norm <= 0:
            raise ValueError("Separate gradient limits must be positive")
        self.actor_max_grad_norm = float(actor_max_grad_norm)
        self.value_max_grad_norm = float(value_max_grad_norm)
        if value_loss_multiplier <= 0 or target_kl_multiplier <= 0:
            raise ValueError("Loss and KL multipliers must be positive")
        self.value_loss_multiplier = float(value_loss_multiplier)
        self.target_kl_multiplier = float(target_kl_multiplier)
        super().__init__(*args, **kwargs)

    def _separated_parameter_groups(
        self,
    ) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
        actor: list[torch.nn.Parameter] = []
        value: list[torch.nn.Parameter] = []
        unknown: list[str] = []
        for name, parameter in self.policy.named_parameters():
            if not parameter.requires_grad:
                continue
            if (
                name == "log_std"
                or "mlp_extractor.policy_net" in name
                or name.startswith("action_net.")
            ):
                actor.append(parameter)
            elif (
                "mlp_extractor.value_net" in name
                or name.startswith("value_net.")
            ):
                value.append(parameter)
            else:
                unknown.append(name)
        if unknown:
            raise RuntimeError(
                "Separate clipping does not support shared/unclassified policy "
                f"parameters: {unknown}"
            )
        if not actor or not value:
            raise RuntimeError("Actor and value parameter groups must be non-empty")
        if {id(parameter) for parameter in actor}.intersection(
            id(parameter) for parameter in value
        ):
            raise RuntimeError("Actor and value parameter groups overlap")
        return actor, value

    def _update_training_learning_rates(self) -> None:
        self._update_learning_rate(self.policy.optimizer)

    def _zero_training_gradients(self) -> None:
        self.policy.optimizer.zero_grad(set_to_none=True)

    def _step_training_optimizers(self) -> None:
        self.policy.optimizer.step()

    def train(self) -> None:
        self.policy.set_training_mode(True)
        self._update_training_learning_rates()
        clip_range = self.clip_range(self._current_progress_remaining)
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(
                self._current_progress_remaining
            )

        entropy_losses: list[float] = []
        policy_losses: list[float] = []
        value_losses: list[float] = []
        clip_fractions: list[float] = []
        actor_gradient_norms: list[float] = []
        value_gradient_norms: list[float] = []
        actor_clip_scales: list[float] = []
        value_clip_scales: list[float] = []
        actor_parameters, value_parameters = self._separated_parameter_groups()
        continue_training = True
        final_loss: torch.Tensor | None = None
        final_approx_kl_divs: list[float] = []

        for _ in range(self.n_epochs):
            approx_kl_divs: list[float] = []
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = actions.long().flatten()
                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations,
                    actions,
                )
                values = values.flatten()
                advantages = rollout_data.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (
                        advantages - advantages.mean()
                    ) / (advantages.std() + 1e-8)

                ratio = torch.exp(log_prob - rollout_data.old_log_prob)
                policy_loss = -torch.min(
                    advantages * ratio,
                    advantages
                    * torch.clamp(ratio, 1 - clip_range, 1 + clip_range),
                ).mean()
                policy_losses.append(float(policy_loss.item()))
                clip_fractions.append(
                    float(
                        torch.mean(
                            (torch.abs(ratio - 1) > clip_range).float()
                        ).item()
                    )
                )

                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + torch.clamp(
                        values - rollout_data.old_values,
                        -clip_range_vf,
                        clip_range_vf,
                    )
                value_loss = self.value_loss_multiplier * F.mse_loss(
                    rollout_data.returns,
                    values_pred,
                )
                value_losses.append(float(value_loss.item()))

                if entropy is None:
                    entropy_loss = -torch.mean(-log_prob)
                else:
                    entropy_loss = -torch.mean(entropy)
                entropy_losses.append(float(entropy_loss.item()))
                final_loss = (
                    policy_loss
                    + self.ent_coef * entropy_loss
                    + self.vf_coef * value_loss
                )

                with torch.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl = torch.mean(
                        (torch.exp(log_ratio) - 1) - log_ratio
                    ).cpu().item()
                    approx_kl_divs.append(float(approx_kl))
                if (
                    self.target_kl is not None
                    and approx_kl
                    > self.target_kl_multiplier * self.target_kl
                ):
                    continue_training = False
                    if self.verbose >= 1:
                        print(
                            "Early stopping due to max KL: "
                            f"{approx_kl:.6f}"
                        )
                    break

                self._zero_training_gradients()
                final_loss.backward()
                actor_norm = float(
                    torch.nn.utils.clip_grad_norm_(
                        actor_parameters,
                        self.actor_max_grad_norm,
                    ).item()
                )
                value_norm = float(
                    torch.nn.utils.clip_grad_norm_(
                        value_parameters,
                        self.value_max_grad_norm,
                    ).item()
                )
                actor_gradient_norms.append(actor_norm)
                value_gradient_norms.append(value_norm)
                actor_clip_scales.append(
                    min(
                        1.0,
                        self.actor_max_grad_norm / (actor_norm + 1e-12),
                    )
                )
                value_clip_scales.append(
                    min(
                        1.0,
                        self.value_max_grad_norm / (value_norm + 1e-12),
                    )
                )
                self._step_training_optimizers()
            self._n_updates += 1
            final_approx_kl_divs = approx_kl_divs
            if not continue_training:
                break

        if final_loss is None or not actor_gradient_norms:
            raise RuntimeError("PPO train received an empty rollout buffer")
        variance = explained_variance(
            self.rollout_buffer.values.flatten(),
            self.rollout_buffer.returns.flatten(),
        )
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record(
            "train/policy_gradient_loss",
            np.mean(policy_losses),
        )
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record(
            "train/approx_kl",
            np.mean(final_approx_kl_divs),
        )
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", float(final_loss.item()))
        self.logger.record("train/explained_variance", variance)
        self.logger.record(
            "train/actor_gradient_norm",
            np.mean(actor_gradient_norms),
        )
        self.logger.record(
            "train/value_gradient_norm",
            np.mean(value_gradient_norms),
        )
        self.logger.record(
            "train/actor_gradient_clip_scale",
            np.mean(actor_clip_scales),
        )
        self.logger.record(
            "train/value_gradient_clip_scale",
            np.mean(value_clip_scales),
        )
        self.logger.record(
            "train/value_to_actor_gradient_norm_ratio",
            np.mean(value_gradient_norms)
            / max(np.mean(actor_gradient_norms), 1e-12),
        )
        if hasattr(self.policy, "log_std"):
            self.logger.record(
                "train/std",
                torch.exp(self.policy.log_std).mean().item(),
            )
        self.logger.record(
            "train/n_updates",
            self._n_updates,
            exclude="tensorboard",
        )
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)


class ResiPAlignedPPO(SeparatedClipPPO):
    """ResiP-style PPO optimization with independent actor/value AdamW."""

    def __init__(
        self,
        *args: Any,
        actor_learning_rate: float = 3e-4,
        value_learning_rate: float = 5e-3,
        fixed_log_std: bool = True,
        schedule_total_iterations: int = 1_000,
        actor_warmup_iterations: int = 5,
        **kwargs: Any,
    ) -> None:
        if actor_learning_rate <= 0 or value_learning_rate <= 0:
            raise ValueError("ResiP actor/value learning rates must be positive")
        self.actor_learning_rate = float(actor_learning_rate)
        self.value_learning_rate = float(value_learning_rate)
        self.fixed_log_std = bool(fixed_log_std)
        if schedule_total_iterations <= 0 or actor_warmup_iterations < 0:
            raise ValueError("Invalid ResiP scheduler lengths")
        self.schedule_total_iterations = int(schedule_total_iterations)
        self.actor_warmup_iterations = int(actor_warmup_iterations)
        super().__init__(*args, **kwargs)

    def _setup_model(self) -> None:
        super()._setup_model()
        previous_head = self.policy.action_net
        self.policy.action_net = torch.nn.Linear(
            previous_head.in_features,
            previous_head.out_features,
            bias=False,
            device=previous_head.weight.device,
            dtype=previous_head.weight.dtype,
        )
        for network in (
            self.policy.mlp_extractor.policy_net,
            self.policy.mlp_extractor.value_net,
        ):
            for module in network.modules():
                if isinstance(module, torch.nn.Linear):
                    torch.nn.init.kaiming_normal_(
                        module.weight,
                        mode="fan_in",
                        nonlinearity="relu",
                    )
                    if module.bias is not None:
                        torch.nn.init.zeros_(module.bias)
        torch.nn.init.zeros_(self.policy.action_net.weight)
        torch.nn.init.orthogonal_(self.policy.value_net.weight, gain=0.25)
        torch.nn.init.constant_(self.policy.value_net.bias, 0.25)
        if self.fixed_log_std and hasattr(self.policy, "log_std"):
            self.policy.log_std.requires_grad_(False)
        actor_parameters, value_parameters = self._separated_parameter_groups()
        self.actor_optimizer = torch.optim.AdamW(
            actor_parameters,
            lr=self.actor_learning_rate,
            betas=(0.9, 0.999),
            eps=1e-5,
            weight_decay=1e-6,
        )
        self.value_optimizer = torch.optim.AdamW(
            value_parameters,
            lr=self.value_learning_rate,
            betas=(0.9, 0.999),
            eps=1e-5,
            weight_decay=1e-6,
        )
        if not hasattr(self, "actor_optimizer_steps"):
            self.actor_optimizer_steps = 0
        if not hasattr(self, "value_optimizer_steps"):
            self.value_optimizer_steps = 0
        if not hasattr(self, "resip_schedule_iterations"):
            self.resip_schedule_iterations = 0

    def _get_torch_save_params(self) -> tuple[list[str], list[str]]:
        return ["policy", "actor_optimizer", "value_optimizer"], []

    def _update_training_learning_rates(self) -> None:
        step = int(self.resip_schedule_iterations)
        if step < self.actor_warmup_iterations:
            actor_factor = step / max(1, self.actor_warmup_iterations)
        else:
            actor_progress = (
                (step - self.actor_warmup_iterations)
                / max(
                    1,
                    self.schedule_total_iterations
                    - self.actor_warmup_iterations,
                )
            )
            actor_factor = 0.5 * (
                1.0 + np.cos(np.pi * min(1.0, actor_progress))
            )
        critic_progress = step / max(1, self.schedule_total_iterations)
        value_factor = 0.5 * (
            1.0 + np.cos(np.pi * min(1.0, critic_progress))
        )
        actor_rate = self.actor_learning_rate * actor_factor
        value_rate = self.value_learning_rate * value_factor
        for group in self.actor_optimizer.param_groups:
            group["lr"] = actor_rate
        for group in self.value_optimizer.param_groups:
            group["lr"] = value_rate
        self.logger.record("train/actor_learning_rate", actor_rate)
        self.logger.record("train/value_learning_rate", value_rate)
        self.logger.record("train/resip_schedule_iteration", step)

    def _zero_training_gradients(self) -> None:
        self.actor_optimizer.zero_grad(set_to_none=True)
        self.value_optimizer.zero_grad(set_to_none=True)

    def _step_training_optimizers(self) -> None:
        self.actor_optimizer.step()
        self.value_optimizer.step()
        self.ppo_optimizer_steps += 1
        self.actor_optimizer_steps += 1
        self.value_optimizer_steps += 1

    def train(self) -> None:
        super().train()
        self.resip_schedule_iterations += 1
