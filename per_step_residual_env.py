"""Primitive Hopper wrapper driven by a frozen DSRL action-chunk planner."""

from __future__ import annotations

import copy
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Mapping, Optional

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from p6_runtime import capture_rng_state, isolated_rng, restore_rng_state


@dataclass(frozen=True)
class FrozenPlan:
    action_base_chunk: np.ndarray
    noise_scaled: np.ndarray
    noise_log_prob: float
    base_version: int = 0


class _PrivateTorchRNG:
    """Torch RNG stream that cannot be advanced by the residual policy."""

    def __init__(self, device: torch.device, seed: int = 0) -> None:
        self.device = torch.device(device)
        self.cpu_state: torch.Tensor
        self.device_state: Optional[torch.Tensor]
        self.seed(seed)

    def seed(self, seed: int) -> None:
        outer_cpu = torch.random.get_rng_state()
        outer_device = (
            torch.cuda.get_rng_state(self.device)
            if self.device.type == "cuda"
            else None
        )
        try:
            torch.manual_seed(int(seed))
            if self.device.type == "cuda":
                torch.cuda.manual_seed(int(seed))
            self.cpu_state = torch.random.get_rng_state().clone()
            self.device_state = (
                torch.cuda.get_rng_state(self.device).clone()
                if self.device.type == "cuda"
                else None
            )
        finally:
            torch.random.set_rng_state(outer_cpu)
            if outer_device is not None:
                torch.cuda.set_rng_state(outer_device, self.device)

    def get_state(self) -> dict[str, Optional[torch.Tensor]]:
        return {
            "cpu": self.cpu_state.clone(),
            "device": (
                self.device_state.clone()
                if self.device_state is not None
                else None
            ),
        }

    def set_state(self, state: Mapping[str, Optional[torch.Tensor]]) -> None:
        cpu = state.get("cpu")
        if cpu is None:
            raise ValueError("Private planner RNG is missing its CPU state")
        self.cpu_state = cpu.clone()
        device_state = state.get("device")
        self.device_state = (
            device_state.clone() if device_state is not None else None
        )

    @contextmanager
    def activate(self):
        outer_cpu = torch.random.get_rng_state()
        outer_device = (
            torch.cuda.get_rng_state(self.device)
            if self.device.type == "cuda"
            else None
        )
        torch.random.set_rng_state(self.cpu_state)
        if self.device_state is not None:
            torch.cuda.set_rng_state(self.device_state, self.device)
        try:
            yield
        finally:
            self.cpu_state = torch.random.get_rng_state().clone()
            if self.device.type == "cuda":
                self.device_state = torch.cuda.get_rng_state(
                    self.device
                ).clone()
            torch.random.set_rng_state(outer_cpu)
            if outer_device is not None:
                torch.cuda.set_rng_state(outer_device, self.device)


class FrozenDSRLChunkPlanner:
    """Inference-only adapter exposing a DSRL noise sample and decoded chunk."""

    def __init__(
        self,
        model: Any,
        *,
        action_chunk: int,
        action_dimension: int,
        policy_seed: int = 0,
    ) -> None:
        self.model = model
        self.action_chunk = int(action_chunk)
        self.action_dimension = int(action_dimension)
        if self.action_chunk <= 0 or self.action_dimension <= 0:
            raise ValueError("Planner action dimensions must be positive")
        self.flat_action_dimension = self.action_chunk * self.action_dimension
        self.base_source = "dsrl_checkpoint"
        self._rng = _PrivateTorchRNG(torch.device(self.model.device), policy_seed)
        self._freeze()

    def seed(self, seed: int) -> None:
        self._rng.seed(seed)

    def get_rng_state(self) -> dict[str, Optional[torch.Tensor]]:
        return self._rng.get_state()

    def set_rng_state(
        self,
        state: Mapping[str, Optional[torch.Tensor]],
    ) -> None:
        self._rng.set_state(state)

    def fork(self, seed: int) -> "FrozenDSRLChunkPlanner":
        return type(self)(
            self.model,
            action_chunk=self.action_chunk,
            action_dimension=self.action_dimension,
            policy_seed=seed,
        )

    def _freeze(self) -> None:
        policy = self.model.policy
        policy.set_training_mode(False)
        for name in ("actor", "critic", "critic_target", "critic_noise"):
            module = getattr(self.model, name, None)
            if isinstance(module, torch.nn.Module):
                module.eval()
                module.requires_grad_(False)
            elif hasattr(module, "set_training_mode"):
                module.set_training_mode(False)
        diffusion = getattr(self.model, "diffusion_policy", None)
        for candidate in (diffusion, getattr(diffusion, "base_policy", None)):
            if isinstance(candidate, torch.nn.Module):
                candidate.eval()
                candidate.requires_grad_(False)

    def assert_frozen(self) -> None:
        for name in ("actor", "critic", "critic_target", "critic_noise"):
            module = getattr(self.model, name, None)
            if isinstance(module, torch.nn.Module) and any(
                parameter.requires_grad for parameter in module.parameters()
            ):
                raise RuntimeError(f"Frozen planner module {name} has trainable parameters")
        diffusion = getattr(self.model, "diffusion_policy", None)
        for name, candidate in (
            ("diffusion_policy", diffusion),
            ("diffusion_policy.base_policy", getattr(diffusion, "base_policy", None)),
        ):
            if isinstance(candidate, torch.nn.Module) and any(
                parameter.requires_grad for parameter in candidate.parameters()
            ):
                raise RuntimeError(f"Frozen planner module {name} has trainable parameters")

    def _decode(
        self,
        observation: torch.Tensor,
        noise_decoder_input_flat: torch.Tensor,
    ) -> torch.Tensor:
        decoded = self.model.diffusion_policy(
            observation,
            noise_decoder_input_flat.reshape(
                -1,
                self.action_chunk,
                self.action_dimension,
            ),
            return_numpy=False,
        )
        if not isinstance(decoded, torch.Tensor):
            raise TypeError("Frozen diffusion planner must return a torch.Tensor")
        expected = (
            observation.shape[0],
            self.action_chunk,
            self.action_dimension,
        )
        if tuple(decoded.shape) != expected:
            raise ValueError(
                f"Frozen diffusion output shape {tuple(decoded.shape)} != {expected}"
            )
        return decoded

    def plan(
        self,
        observation: np.ndarray,
        *,
        deterministic: bool,
    ) -> FrozenPlan:
        self._freeze()
        self.assert_frozen()
        observation_array = np.asarray(observation, dtype=np.float32)
        if observation_array.shape != (
            int(np.prod(self.model.observation_space.shape)),
        ):
            raise ValueError(
                "Planner expects one flat observation, got "
                f"{observation_array.shape}"
            )
        observation_tensor = torch.as_tensor(
            observation_array[None],
            device=self.model.device,
            dtype=torch.float32,
        )
        with torch.no_grad(), self._rng.activate():
            if deterministic:
                decoder_numpy, _ = self.model.policy.predict(
                    observation_array[None],
                    deterministic=True,
                )
                noise_decoder_input = torch.as_tensor(
                    np.asarray(decoder_numpy, dtype=np.float32).reshape(
                        1,
                        self.flat_action_dimension,
                    ),
                    device=self.model.device,
                    dtype=torch.float32,
                )
                low = torch.as_tensor(
                    self.model.policy.action_space.low,
                    device=self.model.device,
                    dtype=torch.float32,
                )
                high = torch.as_tensor(
                    self.model.policy.action_space.high,
                    device=self.model.device,
                    dtype=torch.float32,
                )
                noise_scaled = (
                    2.0 * (noise_decoder_input - low) / (high - low) - 1.0
                )
                noise_log_prob = torch.zeros(
                    1,
                    device=self.model.device,
                    dtype=torch.float32,
                )
            else:
                noise_scaled, noise_log_prob = self.model.actor.action_log_prob(
                    observation_tensor
                )
                low = torch.as_tensor(
                    self.model.policy.action_space.low,
                    device=self.model.device,
                    dtype=torch.float32,
                )
                high = torch.as_tensor(
                    self.model.policy.action_space.high,
                    device=self.model.device,
                    dtype=torch.float32,
                )
                noise_decoder_input = low + 0.5 * (
                    noise_scaled + 1.0
                ) * (high - low)
            action_base = self._decode(
                observation_tensor,
                noise_decoder_input,
            )
        return FrozenPlan(
            action_base_chunk=action_base[0].detach().cpu().numpy().copy(),
            noise_scaled=noise_scaled[0].detach().cpu().numpy().copy(),
            noise_log_prob=float(noise_log_prob.reshape(-1)[0].item()),
            base_version=int(getattr(self, "base_version", 0)),
        )


class FrozenDiffusionChunkPlanner:
    """Inference-only DDIM planner driven by a standard Gaussian prior."""

    def __init__(
        self,
        diffusion_policy: Any,
        *,
        device: torch.device | str,
        observation_dimension: int,
        action_chunk: int,
        action_dimension: int,
        policy_seed: int = 0,
    ) -> None:
        self.diffusion_policy = diffusion_policy
        self.device = torch.device(device)
        self.observation_dimension = int(observation_dimension)
        self.action_chunk = int(action_chunk)
        self.action_dimension = int(action_dimension)
        self.flat_action_dimension = self.action_chunk * self.action_dimension
        self.base_source = "diffusion_prior"
        if min(
            self.observation_dimension,
            self.action_chunk,
            self.action_dimension,
        ) <= 0:
            raise ValueError("Planner dimensions must be positive")
        self._rng = _PrivateTorchRNG(self.device, policy_seed)
        self._freeze()

    def _freeze(self) -> None:
        for candidate in (
            self.diffusion_policy,
            getattr(self.diffusion_policy, "base_policy", None),
        ):
            if isinstance(candidate, torch.nn.Module):
                candidate.eval()
                candidate.requires_grad_(False)

    def assert_frozen(self) -> None:
        for name, candidate in (
            ("diffusion_policy", self.diffusion_policy),
            (
                "diffusion_policy.base_policy",
                getattr(self.diffusion_policy, "base_policy", None),
            ),
        ):
            if isinstance(candidate, torch.nn.Module) and any(
                parameter.requires_grad for parameter in candidate.parameters()
            ):
                raise RuntimeError(f"Frozen planner module {name} is trainable")

    def seed(self, seed: int) -> None:
        self._rng.seed(seed)

    def get_rng_state(self) -> dict[str, Optional[torch.Tensor]]:
        return self._rng.get_state()

    def set_rng_state(
        self,
        state: Mapping[str, Optional[torch.Tensor]],
    ) -> None:
        self._rng.set_state(state)

    def fork(self, seed: int) -> "FrozenDiffusionChunkPlanner":
        return type(self)(
            self.diffusion_policy,
            device=self.device,
            observation_dimension=self.observation_dimension,
            action_chunk=self.action_chunk,
            action_dimension=self.action_dimension,
            policy_seed=seed,
        )

    def plan(
        self,
        observation: np.ndarray,
        *,
        deterministic: bool,
    ) -> FrozenPlan:
        self._freeze()
        self.assert_frozen()
        observation_array = np.asarray(observation, dtype=np.float32)
        if observation_array.shape != (self.observation_dimension,):
            raise ValueError(
                "Planner expects one flat observation, got "
                f"{observation_array.shape}"
            )
        observation_tensor = torch.as_tensor(
            observation_array[None],
            device=self.device,
            dtype=torch.float32,
        )
        with torch.no_grad(), self._rng.activate():
            prior = (
                torch.zeros(
                    1,
                    self.action_chunk,
                    self.action_dimension,
                    device=self.device,
                )
                if deterministic
                else torch.randn(
                    1,
                    self.action_chunk,
                    self.action_dimension,
                    device=self.device,
                )
            )
            decoded = self.diffusion_policy(
                observation_tensor,
                prior,
                return_numpy=False,
            )
        if not isinstance(decoded, torch.Tensor):
            raise TypeError("Frozen diffusion planner must return a tensor")
        expected = (1, self.action_chunk, self.action_dimension)
        if tuple(decoded.shape) != expected:
            raise ValueError(
                f"Frozen diffusion output shape {tuple(decoded.shape)} != {expected}"
            )
        flat_prior = prior.reshape(1, -1)
        log_prob = -0.5 * (
            flat_prior.square() + np.log(2.0 * np.pi)
        ).sum(dim=1)
        return FrozenPlan(
            action_base_chunk=decoded[0].detach().cpu().numpy().copy(),
            noise_scaled=flat_prior[0].detach().cpu().numpy().copy(),
            noise_log_prob=float(log_prob.item()),
            base_version=0,
        )


def _copy_np_random_state(generator: Any) -> Any:
    if generator is None:
        return None
    if hasattr(generator, "get_state"):
        return ("random_state", copy.deepcopy(generator.get_state()))
    bit_generator = getattr(generator, "bit_generator", None)
    if bit_generator is not None:
        return ("generator", copy.deepcopy(bit_generator.state))
    raise TypeError(f"Unsupported numpy RNG type {type(generator).__name__}")


def _restore_np_random_state(generator: Any, state: Any) -> None:
    if state is None:
        return
    kind, payload = state
    if kind == "random_state":
        generator.set_state(payload)
    elif kind == "generator":
        generator.bit_generator.state = copy.deepcopy(payload)
    else:
        raise ValueError(f"Unknown numpy RNG state kind {kind!r}")


def _wrapper_elapsed_steps(environment: Any) -> list[tuple[Any, int]]:
    values: list[tuple[Any, int]] = []
    current = environment
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        if hasattr(current, "_elapsed_steps"):
            values.append((current, int(current._elapsed_steps)))
        current = getattr(current, "env", None)
    return values


@dataclass
class PlannerEnvironmentSnapshot:
    simulator_state: Any
    simulator_rng_state: Any
    action_space_rng_state: Any
    wrapper_elapsed_steps: list[tuple[Any, int]]
    primitive_count: int
    raw_observation: np.ndarray
    plan: FrozenPlan
    phase: int
    global_rng_state: Mapping[str, Any]
    planner_rng_state: Optional[Mapping[str, Optional[torch.Tensor]]]


class FrozenDSRLPrimitiveResidualEnv(gym.Env):
    """Expose primitive execution while retaining a cached frozen DSRL chunk."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        primitive_environment: Any,
        planner: FrozenDSRLChunkPlanner,
        *,
        raw_observation_dim: int,
        max_episode_steps: int,
        deterministic_base: bool = False,
    ) -> None:
        super().__init__()
        self.env = primitive_environment
        self.planner = planner
        self.raw_observation_dim = int(raw_observation_dim)
        self.max_episode_steps = int(max_episode_steps)
        self.deterministic_base = bool(deterministic_base)
        if self.raw_observation_dim <= 0 or self.max_episode_steps <= 0:
            raise ValueError("Observation and episode dimensions must be positive")
        primitive_low = np.asarray(self.env.action_space.low, dtype=np.float32)
        primitive_high = np.asarray(self.env.action_space.high, dtype=np.float32)
        if primitive_low.shape != (self.planner.action_dimension,):
            raise ValueError("Primitive action bounds conflict with planner dimension")
        self.action_space = spaces.Box(
            low=primitive_low,
            high=primitive_high,
            dtype=np.float32,
        )
        self.base_action_start = self.raw_observation_dim
        self.phase_start = self.base_action_start + self.planner.action_dimension
        self.noise_log_prob_index = self.phase_start + self.planner.action_chunk
        self.augmented_observation_dim = self.noise_log_prob_index + 1
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.augmented_observation_dim,),
            dtype=np.float32,
        )
        self._plan: Optional[FrozenPlan] = None
        self._phase = 0
        self._primitive_count = 0
        self._last_raw_observation: Optional[np.ndarray] = None

    @property
    def phase(self) -> int:
        return self._phase

    @property
    def current_base_action(self) -> np.ndarray:
        if self._plan is None:
            raise RuntimeError("Environment must be reset before reading base action")
        return self._plan.action_base_chunk[self._phase].copy()

    @property
    def current_base_action_chunk(self) -> np.ndarray:
        """Return the complete cached base plan without exposing mutable state."""

        if self._plan is None:
            raise RuntimeError("Environment must be reset before reading base plan")
        return self._plan.action_base_chunk.copy()

    @property
    def current_noise_scaled(self) -> np.ndarray:
        """Return the frozen DSRL latent associated with the cached base plan."""

        if self._plan is None:
            raise RuntimeError("Environment must be reset before reading base noise")
        return self._plan.noise_scaled.copy()

    def _augment(
        self,
        observation: np.ndarray,
        plan: FrozenPlan,
        phase: int,
    ) -> np.ndarray:
        observation_array = np.asarray(observation, dtype=np.float32)
        if observation_array.shape != (self.raw_observation_dim,):
            raise ValueError(
                f"Raw observation shape {observation_array.shape} is incompatible"
            )
        phase_one_hot = np.zeros(self.planner.action_chunk, dtype=np.float32)
        phase_one_hot[int(phase)] = 1.0
        return np.concatenate(
            (
                observation_array,
                np.asarray(plan.action_base_chunk[phase], dtype=np.float32),
                phase_one_hot,
                np.asarray([plan.noise_log_prob], dtype=np.float32),
            )
        ).astype(np.float32, copy=False)

    def _new_plan(self, observation: np.ndarray) -> FrozenPlan:
        return self.planner.plan(
            np.asarray(observation, dtype=np.float32),
            deterministic=self.deterministic_base,
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        reset_kwargs: dict[str, Any] = {}
        if seed is not None:
            reset_kwargs["seed"] = int(seed)
        if options is not None:
            reset_kwargs["options"] = options
        try:
            reset_result = self.env.reset(**reset_kwargs)
        except TypeError:
            reset_kwargs.pop("options", None)
            reset_result = self.env.reset(**reset_kwargs)
        observation = (
            reset_result[0] if isinstance(reset_result, tuple) else reset_result
        )
        observation_array = np.asarray(observation, dtype=np.float32)
        self._primitive_count = 0
        self._phase = 0
        self._last_raw_observation = observation_array.copy()
        self._plan = self._new_plan(observation_array)
        return self._augment(observation_array, self._plan, self._phase), {}

    def refresh_plan(self) -> np.ndarray:
        """Atomically replace a cached chunk after the base policy changes.

        This does not step or reset the simulator.  The next PPO rollout starts
        at phase zero with a chunk sampled from the current base policy and the
        observation returned here must replace the learner's cached `_last_obs`.
        """

        if self._last_raw_observation is None:
            raise RuntimeError("Environment must be reset before refreshing a plan")
        self._plan = self._new_plan(self._last_raw_observation)
        self._phase = 0
        return self._augment(
            self._last_raw_observation,
            self._plan,
            self._phase,
        )

    def _raw_hopper(self) -> Any:
        current = self.env
        visited: set[int] = set()
        while current is not None and id(current) not in visited:
            visited.add(id(current))
            unwrapped = getattr(current, "unwrapped", None)
            if unwrapped is not None and hasattr(unwrapped, "sim"):
                return unwrapped
            if hasattr(current, "sim"):
                return current
            current = getattr(current, "env", None)
        raise RuntimeError("Could not locate the Hopper simulator")

    def _raw_action(self, normalized_action: np.ndarray) -> np.ndarray:
        unnormalize = getattr(self.env, "unnormalize_action", None)
        if callable(unnormalize):
            return np.asarray(unnormalize(normalized_action), dtype=np.float64)
        return np.asarray(normalized_action, dtype=np.float64)

    def _terminal_context(
        self,
        terminal_observation: np.ndarray,
    ) -> tuple[FrozenPlan, int]:
        if self._plan is None:
            raise RuntimeError("Missing current plan")
        next_phase = self._phase + 1
        if next_phase < self.planner.action_chunk:
            return self._plan, next_phase
        # Time-limit bootstrapping needs a valid next planner state, but this
        # diagnostic construction must not perturb the reset policy stream.
        with isolated_rng():
            return self._new_plan(terminal_observation), 0

    def step(
        self,
        action_exec: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._plan is None:
            raise RuntimeError("Environment must be reset before stepping")
        action_array = np.asarray(action_exec, dtype=np.float32)
        if action_array.shape != self.action_space.shape:
            raise ValueError(
                f"Executed action shape {action_array.shape} != {self.action_space.shape}"
            )
        if np.any(action_array < self.action_space.low) or np.any(
            action_array > self.action_space.high
        ):
            raise ValueError("Executed primitive action is outside legal bounds")
        action_base = self.current_base_action
        executed_phase = self._phase
        raw_hopper = self._raw_hopper()
        x_before = float(raw_hopper.sim.data.qpos[0])
        raw_action = self._raw_action(action_array)
        step_result = self.env.step(action_array)
        if len(step_result) == 4:
            observation, reward, legacy_done, info = step_result
            env_timeout = bool(info.get("TimeLimit.truncated", False))
            terminated = bool(legacy_done and not env_timeout)
            truncated = bool(env_timeout)
        elif len(step_result) == 5:
            observation, reward, terminated, truncated, info = step_result
            terminated = bool(terminated)
            truncated = bool(truncated)
        else:
            raise ValueError("Unexpected primitive environment step result")
        self._primitive_count += 1
        if self._primitive_count >= self.max_episode_steps and not terminated:
            truncated = True
        done = bool(terminated or truncated)
        observation_array = np.asarray(observation, dtype=np.float32)
        self._last_raw_observation = observation_array.copy()
        x_after = float(raw_hopper.sim.data.qpos[0])
        height = float(raw_hopper.sim.data.qpos[1])
        pitch = float(raw_hopper.sim.data.qpos[2])
        reward_forward = (x_after - x_before) / float(raw_hopper.dt)
        reward_healthy = 1.0
        reward_control = -1e-3 * float(np.square(raw_action).sum())
        decomposition_error = float(
            reward - (reward_forward + reward_healthy + reward_control)
        )
        if abs(decomposition_error) > 1e-5:
            raise RuntimeError(
                "Hopper reward decomposition mismatch: "
                f"{decomposition_error}"
            )

        if done:
            next_plan, next_phase = self._terminal_context(observation_array)
        else:
            next_phase = self._phase + 1
            if next_phase == self.planner.action_chunk:
                next_plan = self._new_plan(observation_array)
                next_phase = 0
            else:
                next_plan = self._plan
        next_observation = self._augment(
            observation_array,
            next_plan,
            next_phase,
        )
        if not done:
            self._plan = next_plan
            self._phase = next_phase

        result_info = dict(info)
        result_info.update(
            {
                "nominal_primitive_steps": 1,
                "actual_primitive_steps": 1,
                "early_termination_within_chunk": False,
                "action_chunk_phase": int(executed_phase),
                "base_snapshot_version": int(self._plan.base_version),
                "action_base": action_base.copy(),
                "action_exec": action_array.copy(),
                "effective_residual": (action_array - action_base).copy(),
                "reward_forward": float(reward_forward),
                "reward_healthy": float(reward_healthy),
                "reward_control": float(reward_control),
                "reward_decomposition_error": decomposition_error,
                "x_velocity": float(reward_forward),
                "height": height,
                "pitch": pitch,
                "TimeLimit.truncated": bool(truncated and not terminated),
                "termination_reason": (
                    "environment_terminal"
                    if terminated
                    else "time_limit"
                    if truncated
                    else None
                ),
            }
        )
        return (
            next_observation,
            float(reward),
            bool(terminated),
            bool(truncated),
            result_info,
        )

    def _rng_owner(self) -> Any:
        return self._raw_hopper()

    def capture_state(self) -> PlannerEnvironmentSnapshot:
        if self._plan is None:
            raise RuntimeError("Environment must be reset before snapshot")
        raw = self._raw_hopper()
        action_rng = getattr(self.action_space, "np_random", None)
        return PlannerEnvironmentSnapshot(
            simulator_state=copy.deepcopy(raw.sim.get_state()),
            simulator_rng_state=_copy_np_random_state(
                getattr(raw, "np_random", None)
            ),
            action_space_rng_state=_copy_np_random_state(action_rng),
            wrapper_elapsed_steps=_wrapper_elapsed_steps(self.env),
            primitive_count=self._primitive_count,
            raw_observation=self._last_raw_observation.copy(),
            plan=FrozenPlan(
                action_base_chunk=self._plan.action_base_chunk.copy(),
                noise_scaled=self._plan.noise_scaled.copy(),
                noise_log_prob=self._plan.noise_log_prob,
                base_version=self._plan.base_version,
            ),
            phase=self._phase,
            global_rng_state=capture_rng_state(),
            planner_rng_state=(
                self.planner.get_rng_state()
                if hasattr(self.planner, "get_rng_state")
                else None
            ),
        )

    def restore_state(self, snapshot: PlannerEnvironmentSnapshot) -> None:
        raw = self._raw_hopper()
        raw.sim.set_state(copy.deepcopy(snapshot.simulator_state))
        raw.sim.forward()
        _restore_np_random_state(
            getattr(raw, "np_random", None),
            snapshot.simulator_rng_state,
        )
        _restore_np_random_state(
            getattr(self.action_space, "np_random", None),
            snapshot.action_space_rng_state,
        )
        for wrapper, elapsed_steps in snapshot.wrapper_elapsed_steps:
            wrapper._elapsed_steps = int(elapsed_steps)
        self._primitive_count = int(snapshot.primitive_count)
        self._last_raw_observation = snapshot.raw_observation.copy()
        self._plan = FrozenPlan(
            action_base_chunk=snapshot.plan.action_base_chunk.copy(),
            noise_scaled=snapshot.plan.noise_scaled.copy(),
            noise_log_prob=float(snapshot.plan.noise_log_prob),
            base_version=int(snapshot.plan.base_version),
        )
        self._phase = int(snapshot.phase)
        if snapshot.planner_rng_state is not None:
            self.planner.set_rng_state(snapshot.planner_rng_state)
        restore_rng_state(snapshot.global_rng_state)

    def get_normalized_score(self, raw_return: float) -> float:
        current = self.env
        visited: set[int] = set()
        while current is not None and id(current) not in visited:
            visited.add(id(current))
            scorer = getattr(current, "get_normalized_score", None)
            if callable(scorer):
                return float(scorer(raw_return))
            current = getattr(current, "env", None)
        raise AttributeError("Wrapped environment does not expose get_normalized_score")

    def render(self) -> Any:
        return self.env.render()

    def close(self) -> None:
        self.env.close()
