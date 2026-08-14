from typing import Any, Optional, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces
from torch import nn
from torch.nn import functional as F

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.type_aliases import GymEnv, Schedule
from stable_baselines3.common.utils import polyak_update
from stable_baselines3.dsrl.dsrl import DSRL
from stable_baselines3.sac.policies import SACPolicy


SelfRFSDSRL = TypeVar("SelfRFSDSRL", bound="RFSDSRL")


def compose_rfs_action(
    base_action: th.Tensor,
    residual_action: th.Tensor,
    residual_scale: float,
    action_low: th.Tensor,
    action_high: th.Tensor,
) -> th.Tensor:
    """Apply the RFS output modulation in the environment action space."""
    if base_action.shape != residual_action.shape:
        raise ValueError(
            "base_action and residual_action must have the same shape, "
            f"got {tuple(base_action.shape)} and {tuple(residual_action.shape)}"
        )
    return th.maximum(
        th.minimum(base_action + residual_scale * residual_action, action_high),
        action_low,
    )


class RFSDSRL(DSRL):
    """
    RFS-style joint input/output modulation on top of DSRL-NA.

    This is a separate algorithm from :class:`DSRL`. The frozen diffusion
    sampler maps the first half of the joint actor output (latent noise) to a
    base action. The second half is an additive residual:

        (w, a_r) ~ pi(. | s)
        a_b = G(s, w)
        a_exec = clip(a_b + beta * a_r)

    The ordinary action critic is trained on ``a_exec``. A second, joint
    modulation critic Q(s, w, a_r) distills the action critic and supplies the
    actor gradient, so no gradient is propagated through the diffusion sampler.

    The online optimizer remains the SAC/DSRL-NA optimizer used by this
    repository. RFS's policy modulation is added without replacing the original
    ``DSRL`` implementation or changing its execution path.

    :param residual_scale: beta in the affine RFS output correction.
    :param residual_log_std_init: Initial log standard deviation for the
        residual half of the joint actor. Its mean is initialized to zero.
    """

    actor: nn.Module

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
        gradient_steps: int = 1,
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
        diffusion_policy=None,
        diffusion_act_dim=None,
        noise_critic_grad_steps: int = 1,
        critic_backup_combine_type: str = "min",
        residual_scale: float = 1.0,
        residual_log_std_init: float = -5.0,
    ):
        if not np.isfinite(residual_scale) or residual_scale < 0:
            raise ValueError(f"residual_scale must be finite and non-negative, got {residual_scale}")
        if not np.isfinite(residual_log_std_init):
            raise ValueError(
                "residual_log_std_init must be finite, "
                f"got {residual_log_std_init}"
            )

        self.residual_scale = float(residual_scale)
        self.residual_log_std_init = float(residual_log_std_init)
        self._joint_target_entropy_auto = target_entropy == "auto"
        if diffusion_act_dim is None:
            if _init_setup_model:
                raise ValueError("diffusion_act_dim is required when creating RFSDSRL")
            # SB3 load() first constructs an uninitialized shell and restores
            # the saved diffusion dimensions before calling _setup_model().
            diffusion_act_dim = (1, 1)

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

    def _setup_model(self) -> None:
        if self.use_sde:
            raise ValueError("RFSDSRL currently supports diagonal Gaussian actors only (use_sde=False)")

        # Build the untouched DSRL action critic/replay machinery first.
        super()._setup_model()

        self.action_dim_flat = int(np.prod(self.action_space.shape))
        expected_dim = self.diffusion_act_chunk * self.diffusion_act_dim
        if self.action_dim_flat != expected_dim:
            raise ValueError(
                "Environment action dimension must equal diffusion chunk dimension: "
                f"{self.action_dim_flat} != {self.diffusion_act_chunk} * "
                f"{self.diffusion_act_dim}"
            )

        # A separate policy keeps the original DSRL policy/action critic intact.
        # Its actor and critic both operate on the joint modulation (w, a_r).
        self.modulation_action_space = spaces.Box(
            low=-np.ones(2 * self.action_dim_flat, dtype=np.float32),
            high=np.ones(2 * self.action_dim_flat, dtype=np.float32),
            dtype=np.float32,
        )
        self.modulation_policy = self.policy_class(
            self.observation_space,
            self.modulation_action_space,
            self.lr_schedule,
            **self.policy_kwargs,
        ).to(self.device)
        self.actor = self.modulation_policy.actor
        self.critic_modulation = self.modulation_policy.critic

        # Compatibility alias for existing logging/checkpoint utilities. It is
        # the joint critic here, not the original noise-only critic.
        self.critic_noise = self.critic_modulation

        self._initialize_residual_head()
        if self._joint_target_entropy_auto:
            self.target_entropy = float(-2 * self.action_dim_flat)

    def _initialize_residual_head(self) -> None:
        if not isinstance(self.actor.mu, nn.Linear) or not isinstance(self.actor.log_std, nn.Linear):
            raise TypeError("RFSDSRL requires linear Gaussian mean/log-std actor heads")

        residual_slice = slice(self.action_dim_flat, 2 * self.action_dim_flat)
        with th.no_grad():
            self.actor.mu.weight[residual_slice].zero_()
            self.actor.mu.bias[residual_slice].zero_()
            self.actor.log_std.weight[residual_slice].zero_()
            self.actor.log_std.bias[residual_slice].fill_(self.residual_log_std_init)

    def _split_modulation(self, modulation: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        if modulation.ndim != 2 or modulation.shape[1] != 2 * self.action_dim_flat:
            raise ValueError(
                "Expected flattened joint modulation with shape "
                f"(batch, {2 * self.action_dim_flat}), got {tuple(modulation.shape)}"
            )
        return modulation[:, : self.action_dim_flat], modulation[:, self.action_dim_flat :]

    def _action_bounds(self, reference: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        low = th.as_tensor(self.action_space.low, device=reference.device, dtype=reference.dtype)
        high = th.as_tensor(self.action_space.high, device=reference.device, dtype=reference.dtype)
        return low.reshape(1, -1), high.reshape(1, -1)

    def _compose_action(self, base_action: th.Tensor, residual_action: th.Tensor) -> th.Tensor:
        low, high = self._action_bounds(base_action)
        return compose_rfs_action(
            base_action,
            residual_action.to(device=base_action.device, dtype=base_action.dtype),
            self.residual_scale,
            low,
            high,
        )

    @th.no_grad()
    def _decode_noise(self, observations: th.Tensor, noise: th.Tensor) -> th.Tensor:
        decoded = self.diffusion_policy(
            observations,
            noise.reshape(-1, self.diffusion_act_chunk, self.diffusion_act_dim),
            return_numpy=False,
        )
        if isinstance(decoded, np.ndarray):
            decoded = th.as_tensor(decoded, device=self.device, dtype=th.float32)
        return decoded.to(self.device).reshape(-1, self.action_dim_flat)

    @th.no_grad()
    def _decode_scaled_noise(self, observations: th.Tensor, scaled_noise: th.Tensor) -> th.Tensor:
        noise_numpy = self.policy.unscale_action(scaled_noise.detach().cpu().numpy())
        noise = th.as_tensor(noise_numpy, device=self.device, dtype=th.float32)
        return self._decode_noise(observations, noise)

    def _reduce_critics(self, q_values: tuple[th.Tensor, ...]) -> th.Tensor:
        values = th.cat(q_values, dim=1)
        if self.critic_backup_combine_type == "min":
            return th.min(values, dim=1, keepdim=True).values
        if self.critic_backup_combine_type == "mean":
            return th.mean(values, dim=1, keepdim=True)
        raise ValueError(
            "critic_backup_combine_type must be 'min' or 'mean', "
            f"got {self.critic_backup_combine_type!r}"
        )

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        self.policy.set_training_mode(True)
        self.modulation_policy.set_training_mode(True)

        optimizers = [
            self.actor.optimizer,
            self.critic.optimizer,
            self.critic_modulation.optimizer,
        ]
        if self.ent_coef_optimizer is not None:
            optimizers.append(self.ent_coef_optimizer)
        self._update_learning_rate(optimizers)

        ent_coef_losses: list[float] = []
        ent_coefs: list[float] = []
        actor_losses: list[float] = []
        critic_losses: list[float] = []
        modulation_critic_losses: list[float] = []

        if self.actor_gradient_steps < 0:
            actor_gradient_idx = set(range(gradient_steps))
        elif self.actor_gradient_steps == 0:
            actor_gradient_idx = set()
        else:
            actor_gradient_idx = set(
                np.linspace(
                    max(int(gradient_steps / self.actor_gradient_steps) - 1, 0),
                    gradient_steps - 1,
                    min(self.actor_gradient_steps, gradient_steps),
                    dtype=int,
                ).tolist()
            )

        for gradient_step in range(gradient_steps):
            replay_data = self.replay_buffer.sample(  # type: ignore[union-attr]
                batch_size,
                env=self._vec_normalize_env,
            )

            modulation_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                ent_coef = th.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(
                    self.log_ent_coef * (log_prob + self.target_entropy).detach()
                ).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor
            ent_coefs.append(ent_coef.item())

            if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            with th.no_grad():
                next_modulation, next_log_prob = self.actor.action_log_prob(
                    replay_data.next_observations
                )
                next_noise, next_residual = self._split_modulation(next_modulation)
                next_base_action = self._decode_scaled_noise(
                    replay_data.next_observations,
                    next_noise,
                )
                next_action = self._compose_action(next_base_action, next_residual)
                next_q_values = self._reduce_critics(
                    self.critic_target(replay_data.next_observations, next_action)
                )
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                target_q_values = (
                    replay_data.rewards
                    + (1 - replay_data.dones) * self.gamma * next_q_values
                )

            current_q_values = self.critic(replay_data.observations, replay_data.actions)
            critic_loss = 0.5 * sum(
                F.mse_loss(current_q, target_q_values)
                for current_q in current_q_values
            )
            critic_losses.append(critic_loss.item())
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            if gradient_step in actor_gradient_idx:
                q_values_pi = self._reduce_critics(
                    self.critic_modulation(replay_data.observations, modulation_pi)
                )
                actor_loss = (ent_coef * log_prob - q_values_pi).mean()
                actor_losses.append(actor_loss.item())
                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()

            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        for _ in range(self.noise_critic_grad_steps):
            replay_data = self.replay_buffer.sample(  # type: ignore[union-attr]
                batch_size,
                env=self._vec_normalize_env,
            )
            modulation_critic_loss = self.update_modulation_critic(replay_data)
            modulation_critic_losses.append(modulation_critic_loss.item())
            self.critic_modulation.optimizer.zero_grad()
            modulation_critic_loss.backward()
            self.critic_modulation.optimizer.step()

        self.critic_modulation.set_training_mode(False)
        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        self.logger.record(
            "train/modulation_critic_loss",
            np.mean(modulation_critic_losses),
        )
        # Retain the old key so existing DSRL experiment dashboards work.
        self.logger.record(
            "train/noise_critic_loss",
            np.mean(modulation_critic_losses),
        )
        if ent_coef_losses:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))

    def update_modulation_critic(self, replay_data) -> th.Tensor:
        with th.no_grad():
            batch_size = replay_data.actions.shape[0]
            prior_noise = th.randn(
                batch_size,
                self.diffusion_act_chunk,
                self.diffusion_act_dim,
                device=self.device,
            )
            base_action = self._decode_noise(replay_data.observations, prior_noise)

            sampled_modulation, _ = self.actor.action_log_prob(replay_data.observations)
            _, residual_action = self._split_modulation(sampled_modulation)
            executed_action = self._compose_action(base_action, residual_action)
            action_q_values = self.critic(replay_data.observations, executed_action)

            prior_noise_flat = prior_noise.reshape(batch_size, self.action_dim_flat)
            scaled_prior = self.policy.scale_action(prior_noise_flat.cpu().numpy())
            scaled_prior = th.as_tensor(scaled_prior, device=self.device, dtype=th.float32)
            critic_modulation_input = th.cat((scaled_prior, residual_action), dim=1)

        modulation_q_values = self.critic_modulation(
            replay_data.observations,
            critic_modulation_input,
        )
        if len(action_q_values) != len(modulation_q_values):
            raise RuntimeError(
                "Action and modulation critics must have the same number of Q heads"
            )
        return 0.5 * sum(
            F.mse_loss(modulation_q, action_q.detach())
            for modulation_q, action_q in zip(modulation_q_values, action_q_values)
        )

    def update_noise_critic(self, replay_data) -> th.Tensor:
        """Compatibility alias for utilities written for the original DSRL-NA."""
        return self.update_modulation_critic(replay_data)

    def _sample_action(
        self,
        learning_starts: int,
        action_noise: Optional[ActionNoise] = None,
        n_envs: int = 1,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.num_timesteps < learning_starts and not (
            self.use_sde and self.use_sde_at_warmup
        ):
            modulation = np.array(
                [self.modulation_action_space.sample() for _ in range(n_envs)]
            )
        else:
            assert self._last_obs is not None, "self._last_obs was not set"
            modulation, _ = self.modulation_policy.predict(
                self._last_obs,
                deterministic=False,
            )

        modulation = np.asarray(modulation, dtype=np.float32).reshape(n_envs, -1)
        noise = modulation[:, : self.action_dim_flat]
        residual = modulation[:, self.action_dim_flat :]
        if action_noise is not None:
            noise = np.clip(noise + action_noise(), -1, 1)

        noise_unscaled = self.policy.unscale_action(noise)
        noise_tensor = th.as_tensor(noise_unscaled, device=self.device, dtype=th.float32)
        residual_tensor = th.as_tensor(residual, device=self.device, dtype=th.float32)
        observation_tensor = th.as_tensor(
            self._last_obs,
            device=self.device,
            dtype=th.float32,
        )
        base_action = self._decode_noise(observation_tensor, noise_tensor)
        action = self._compose_action(base_action, residual_tensor).cpu().numpy()

        # As in DSRL-NA, replay stores the actual decoded/executed action.
        return action, action.copy()

    def predict_diffused(
        self,
        observation: Union[np.ndarray, dict[str, np.ndarray]],
        state: Optional[tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, Optional[tuple[np.ndarray, ...]]]:
        if isinstance(observation, dict):
            raise TypeError("RFSDSRL currently supports array observations only")

        modulation, next_state = self.modulation_policy.predict(
            observation,
            state,
            episode_start,
            deterministic,
        )
        single_observation = np.asarray(modulation).ndim == 1
        modulation = np.asarray(modulation, dtype=np.float32).reshape(-1, 2 * self.action_dim_flat)
        noise = modulation[:, : self.action_dim_flat]
        residual = modulation[:, self.action_dim_flat :]

        observation_array = np.asarray(observation, dtype=np.float32)
        if observation_array.ndim == len(self.observation_space.shape):
            observation_array = observation_array[None, ...]
        observation_tensor = th.as_tensor(
            observation_array,
            device=self.device,
            dtype=th.float32,
        )
        noise_unscaled = self.policy.unscale_action(noise)
        noise_tensor = th.as_tensor(noise_unscaled, device=self.device, dtype=th.float32)
        residual_tensor = th.as_tensor(residual, device=self.device, dtype=th.float32)
        base_action = self._decode_noise(observation_tensor, noise_tensor)
        action = self._compose_action(base_action, residual_tensor).cpu().numpy()
        if single_observation:
            action = action[0]
        return action, next_state

    def _excluded_save_params(self) -> list[str]:
        return super()._excluded_save_params() + [
            "critic_modulation",
            "critic_noise",
        ]

    def _get_torch_save_params(self) -> tuple[list[str], list[str]]:
        state_dicts = [
            "policy",
            "modulation_policy",
            "actor.optimizer",
            "critic.optimizer",
            "critic_modulation.optimizer",
        ]
        if self.ent_coef_optimizer is not None:
            saved_pytorch_variables = ["log_ent_coef"]
            state_dicts.append("ent_coef_optimizer")
        else:
            saved_pytorch_variables = ["ent_coef_tensor"]
        return state_dicts, saved_pytorch_variables
