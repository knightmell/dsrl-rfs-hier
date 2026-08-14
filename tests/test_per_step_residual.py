from __future__ import annotations

import copy
import inspect
from types import MethodType, SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch
from gymnasium import spaces

from per_step_residual_dsrl import PerStepResidualDSRL
from per_step_residual_env import (
    FrozenDiffusionChunkPlanner,
    FrozenDSRLPrimitiveResidualEnv,
    FrozenPlan,
)
from per_step_residual_ppo import (
    CountingPPO,
    PPOResidualUnitEnv,
    PPOTrainingRewardScaleEnv,
    ResiPRewardNormalize,
    ResidualObservationExtractor,
    ResiPAlignedPPO,
    SeparatedClipPPO,
    zero_initialize_ppo_residual_mean,
)
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv
from train_per_step_residual import (
    ACTION_CHUNK,
    counterfactual_critic_diagnostics,
)
from train_per_step_residual_ppo import resolve_training_reward_scale


def test_counterfactual_sampling_cannot_alias_to_one_chunk_phase():
    spacing = inspect.signature(counterfactual_critic_diagnostics).parameters[
        "sample_spacing"
    ].default
    assert np.gcd(spacing, ACTION_CHUNK) == 1


class TinySim:
    def __init__(self) -> None:
        self.data = SimpleNamespace(
            qpos=np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
        )

    def get_state(self):
        return self.data.qpos.copy()

    def set_state(self, state):
        self.data.qpos[...] = state

    def forward(self):
        pass


class TinyPrimitiveEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.action_space = spaces.Box(-np.ones(1), np.ones(1), dtype=np.float32)
        self.observation_space = spaces.Box(
            -np.inf,
            np.inf,
            shape=(2,),
            dtype=np.float32,
        )
        self.sim = TinySim()
        self.dt = 0.1
        self.np_random = np.random.RandomState(0)
        self._elapsed_steps = 0

    @property
    def unwrapped(self):
        return self

    def unnormalize_action(self, action):
        return np.asarray(action)

    def reset(self, *, seed=None, options=None):
        del options
        if seed is not None:
            self.np_random.seed(seed)
        self._elapsed_steps = 0
        self.sim.data.qpos[...] = [0.0, 1.0, 0.0]
        return np.asarray([0.0, 0.0], dtype=np.float32), {}

    def step(self, action):
        raw = float(np.asarray(action)[0])
        self._elapsed_steps += 1
        x_before = self.sim.data.qpos[0]
        self.sim.data.qpos[0] += 0.1 * (0.5 + raw)
        self.sim.data.qpos[1] = 1.0
        self.sim.data.qpos[2] = 0.0
        forward = (self.sim.data.qpos[0] - x_before) / self.dt
        reward = forward + 1.0 - 1e-3 * raw * raw
        observation = np.asarray(
            [self.sim.data.qpos[0], float(self._elapsed_steps)],
            dtype=np.float32,
        )
        return observation, reward, False, False, {}

    def get_normalized_score(self, raw_return):
        return raw_return / 10.0


class TinyPlanner:
    action_chunk = 4
    action_dimension = 1

    def __init__(self):
        self.calls = 0

    def plan(self, observation, *, deterministic):
        del deterministic
        self.calls += 1
        offset = float(np.asarray(observation)[0])
        return FrozenPlan(
            action_base_chunk=np.asarray(
                [[np.clip(0.1 + offset, -0.9, 0.9)], [0.2], [0.3], [0.4]],
                dtype=np.float32,
            ),
            noise_scaled=np.arange(4, dtype=np.float32),
            noise_log_prob=-2.0,
        )


def make_wrapped():
    planner = TinyPlanner()
    environment = FrozenDSRLPrimitiveResidualEnv(
        TinyPrimitiveEnv(),
        planner,
        raw_observation_dim=2,
        max_episode_steps=20,
    )
    return environment, planner


def make_model(environment, **kwargs):
    policy_kwargs = {
        "net_arch": {"pi": [16], "qf": [16, 16]},
        "activation_fn": torch.nn.Tanh,
        "n_critics": 2,
    }
    return PerStepResidualDSRL(
        "MlpPolicy",
        environment,
        learning_rate=3e-4,
        buffer_size=1024,
        learning_starts=0,
        batch_size=16,
        train_freq=8,
        gradient_steps=20,
        gamma=0.99 ** 0.25,
        policy_kwargs=policy_kwargs,
        raw_observation_dim=2,
        base_action_start=2,
        phase_start=3,
        phase_count=4,
        noise_log_prob_index=7,
        exec_action_low=np.asarray([-1.0], dtype=np.float32),
        exec_action_high=np.asarray([1.0], dtype=np.float32),
        noise_entropy_coefficient=0.2,
        residual_scale=0.1,
        residual_net_arch=(16, 16),
        residual_lr=3e-4,
        residual_actor_gradient_steps=5,
        diagnostics_interval_train_calls=1,
        device="cpu",
        seed=7,
        **kwargs,
    )


def test_phase_advances_and_replans_only_at_chunk_boundary():
    environment, planner = make_wrapped()
    observation, _ = environment.reset(seed=11)
    assert planner.calls == 1
    assert np.argmax(observation[3:7]) == 0
    expected_bases = [0.1, 0.2, 0.3, 0.4]
    for phase, expected_base in enumerate(expected_bases):
        assert environment.phase == phase
        assert environment.current_base_action[0] == pytest.approx(expected_base)
        observation, _, terminated, truncated, info = environment.step(
            environment.current_base_action
        )
        assert not terminated and not truncated
        assert info["action_chunk_phase"] == phase
        assert info["reward_decomposition_error"] == pytest.approx(0.0, abs=1e-6)
        assert info["nominal_primitive_steps"] == 1
        assert info["actual_primitive_steps"] == 1
        if phase < 3:
            assert planner.calls == 1
            assert np.argmax(observation[3:7]) == phase + 1
    assert planner.calls == 2
    assert environment.phase == 0
    assert np.argmax(observation[3:7]) == 0


def test_phase_three_returns_the_new_plan_before_value_bootstrap():
    environment, planner = make_wrapped()
    observation, _ = environment.reset(seed=11)
    first_chunk = environment.current_base_action_chunk
    first_noise = environment.current_noise_scaled
    for _ in range(4):
        observation, _, terminated, truncated, _ = environment.step(
            environment.current_base_action
        )
        assert not terminated and not truncated
    assert planner.calls == 2
    assert environment.phase == 0
    assert np.argmax(observation[3:7]) == 0
    np.testing.assert_array_equal(
        observation[2:3],
        environment.current_base_action_chunk[0],
    )
    assert not np.array_equal(
        first_chunk,
        environment.current_base_action_chunk,
    )
    # The tiny planner deliberately reuses its diagnostic latent.  Reading the
    # public property must still return a copy rather than mutable planner state.
    np.testing.assert_array_equal(first_noise, environment.current_noise_scaled)
    copied = environment.current_base_action_chunk
    copied.fill(42.0)
    assert not np.array_equal(copied, environment.current_base_action_chunk)


def test_ppo_rollout_end_on_phase_three_bootstraps_the_new_plan():
    primitive, planner = make_wrapped()
    environment = PPOResidualUnitEnv(primitive, residual_scale=0.1)
    vector_environment = DummyVecEnv([lambda: environment])
    model = CountingPPO(
        "MlpPolicy",
        vector_environment,
        n_steps=4,
        batch_size=4,
        n_epochs=1,
        gamma=0.99 ** 0.25,
        gae_lambda=0.95 ** 0.25,
        learning_rate=1e-4,
        policy_kwargs={
            "net_arch": {"pi": [8], "vf": [8]},
            "activation_fn": torch.nn.SiLU,
            "log_std_init": np.log(0.05),
            "features_extractor_class": ResidualObservationExtractor,
            "features_extractor_kwargs": {"noise_log_prob_index": 7},
        },
        seed=5,
        device="cpu",
    )
    zero_initialize_ppo_residual_mean(model.policy)
    bootstrap_observations = []
    original_predict_values = model.policy.predict_values

    def record_predict_values(policy, observation):
        bootstrap_observations.append(observation.detach().cpu().numpy().copy())
        return original_predict_values(observation)

    model.policy.predict_values = MethodType(
        record_predict_values,
        model.policy,
    )
    model.learn(total_timesteps=4)
    assert planner.calls == 2
    assert len(bootstrap_observations) == 1
    bootstrap = bootstrap_observations[0][0]
    assert np.argmax(bootstrap[3:7]) == 0
    np.testing.assert_array_equal(
        bootstrap[2:3],
        primitive.current_base_action_chunk[0],
    )


def test_time_limit_terminal_observation_contains_a_new_phase_zero_plan():
    planner = TinyPlanner()
    primitive = FrozenDSRLPrimitiveResidualEnv(
        TinyPrimitiveEnv(),
        planner,
        raw_observation_dim=2,
        max_episode_steps=4,
    )
    environment = PPOResidualUnitEnv(primitive, residual_scale=0.1)
    vector_environment = DummyVecEnv([lambda: environment])
    observation = vector_environment.reset()
    info = None
    for _ in range(4):
        observation, _, done, infos = vector_environment.step(
            np.zeros((1, 1), dtype=np.float32)
        )
        info = infos[0]
    assert done[0]
    assert info is not None
    assert info["TimeLimit.truncated"]
    terminal = info["terminal_observation"]
    assert np.argmax(terminal[3:7]) == 0
    # One initial plan, one terminal bootstrap plan, and one auto-reset plan.
    assert planner.calls == 3
    assert np.argmax(observation[0, 3:7]) == 0


def test_environment_snapshot_restores_simulator_planner_and_rng():
    environment, _ = make_wrapped()
    environment.reset(seed=3)
    environment.step(environment.current_base_action)
    snapshot = environment.capture_state()
    action = np.asarray([0.25], dtype=np.float32)
    first = environment.step(action)
    after_first_rng = np.random.rand()
    environment.restore_state(snapshot)
    second = environment.step(action)
    after_second_rng = np.random.rand()
    np.testing.assert_array_equal(first[0], second[0])
    assert first[1:4] == second[1:4]
    assert first[4]["reward_forward"] == second[4]["reward_forward"]
    assert after_first_rng == after_second_rng


class TinyDiffusion(torch.nn.Module):
    def forward(self, observation, prior, *, return_numpy=False):
        assert not return_numpy
        offset = observation[:, :1, None]
        return torch.tanh(prior + offset)


def test_frozen_diffusion_planner_has_private_reproducible_rng():
    diffusion = TinyDiffusion()
    first = FrozenDiffusionChunkPlanner(
        diffusion,
        device="cpu",
        observation_dimension=2,
        action_chunk=4,
        action_dimension=1,
        policy_seed=123,
    )
    second = first.fork(123)
    observation = np.asarray([0.2, -0.1], dtype=np.float32)

    torch.manual_seed(77)
    global_state = torch.random.get_rng_state().clone()
    first_a = first.plan(observation, deterministic=False)
    torch.testing.assert_close(torch.random.get_rng_state(), global_state)
    _ = torch.randn(1000)
    first_b = first.plan(observation, deterministic=False)

    second_a = second.plan(observation, deterministic=False)
    second_b = second.plan(observation, deterministic=False)
    np.testing.assert_array_equal(
        first_a.action_base_chunk,
        second_a.action_base_chunk,
    )
    np.testing.assert_array_equal(
        first_b.action_base_chunk,
        second_b.action_base_chunk,
    )
    assert all(not parameter.requires_grad for parameter in diffusion.parameters())


def test_frozen_diffusion_deterministic_prior_is_exact_zero():
    planner = FrozenDiffusionChunkPlanner(
        TinyDiffusion(),
        device="cpu",
        observation_dimension=2,
        action_chunk=4,
        action_dimension=1,
        policy_seed=9,
    )
    plan = planner.plan(
        np.asarray([0.2, -0.1], dtype=np.float32),
        deterministic=True,
    )
    np.testing.assert_array_equal(plan.noise_scaled, np.zeros(4, dtype=np.float32))


def test_zero_residual_exactly_reproduces_current_base_action():
    environment, _ = make_wrapped()
    model = make_model(environment)
    observation, _ = environment.reset(seed=1)
    components, _ = model.predict_with_components(observation)
    np.testing.assert_array_equal(
        components["residual_unit"],
        np.zeros(1, dtype=np.float32),
    )
    np.testing.assert_array_equal(
        components["action_exec"],
        observation[2:3],
    )
    assert torch.count_nonzero(model.residual_actor.output_layer.weight) == 0
    assert torch.count_nonzero(model.residual_actor.output_layer.bias) == 0


def test_ppo_residual_wrapper_composes_in_execution_coordinates():
    primitive, _ = make_wrapped()
    environment = PPOResidualUnitEnv(primitive, residual_scale=0.1)
    observation, _ = environment.reset(seed=1)
    zero = environment.compose(np.asarray([0.0], dtype=np.float32))
    np.testing.assert_array_equal(zero.action_exec, observation[2:3])
    positive = environment.compose(np.asarray([0.5], dtype=np.float32))
    np.testing.assert_allclose(
        positive.action_residual_delta,
        [0.05],
        atol=1e-7,
    )
    _, _, _, _, info = environment.step(np.asarray([0.5], dtype=np.float32))
    np.testing.assert_allclose(info["residual_unit"], [0.5])
    np.testing.assert_allclose(info["action_exec"], [0.15], atol=1e-7)


def test_training_reward_scale_preserves_raw_reward_diagnostics():
    primitive, _ = make_wrapped()
    residual = PPOResidualUnitEnv(primitive, residual_scale=0.1)
    environment = PPOTrainingRewardScaleEnv(
        residual,
        reward_scale=0.01,
    )
    environment.reset(seed=1)
    _, scaled_reward, _, _, info = environment.step(
        np.asarray([0.0], dtype=np.float32)
    )
    assert scaled_reward == pytest.approx(
        0.01 * info["unscaled_training_reward"]
    )
    assert info["training_reward_scale"] == 0.01
    assert info["unscaled_training_reward"] == pytest.approx(
        info["reward_forward"]
        + info["reward_healthy"]
        + info["reward_control"]
    )


def test_resip_hopper_reward_profile_is_fixed_without_changing_resip_v1():
    assert resolve_training_reward_scale("resip_hopper_v1", None) == 0.01
    assert resolve_training_reward_scale("resip_hopper_v1", 0.01) == 0.01
    with pytest.raises(ValueError, match="fixes training reward scale"):
        resolve_training_reward_scale("resip_hopper_v1", 0.02)
    assert (
        resolve_training_reward_scale("resip_hopper_long_gae_v1", None)
        == 0.01
    )
    assert resolve_training_reward_scale("resip_v1", None) == 1.0
    assert resolve_training_reward_scale("stable_v1", None) == 0.01
    assert resolve_training_reward_scale("legacy", None) == 1.0


def test_resip_reward_normalization_matches_immediate_variance_formula():
    first, _ = make_wrapped()
    second, _ = make_wrapped()
    vector = DummyVecEnv(
        [
            lambda: PPOResidualUnitEnv(first, residual_scale=0.1),
            lambda: PPOResidualUnitEnv(second, residual_scale=0.1),
        ]
    )
    environment = ResiPRewardNormalize(vector, clip_reward=5.0)
    environment.reset()
    _, normalized, _, infos = environment.step(
        np.zeros((2, 1), dtype=np.float32)
    )
    raw = np.asarray(
        [info["unscaled_training_reward"] for info in infos],
        dtype=np.float64,
    )
    expected = np.clip(
        raw / np.sqrt(environment.reward_var + 1e-8),
        -5.0,
        5.0,
    )
    np.testing.assert_allclose(normalized, expected, rtol=0, atol=1e-7)
    assert environment.reward_count == pytest.approx(2.0001)


def test_separate_gradient_clipping_updates_actor_and_value_independently(
    monkeypatch,
):
    primitive, _ = make_wrapped()
    residual = PPOResidualUnitEnv(primitive, residual_scale=0.1)
    scaled = PPOTrainingRewardScaleEnv(residual, reward_scale=0.01)
    vector_environment = DummyVecEnv([lambda: scaled])
    model = SeparatedClipPPO(
        "MlpPolicy",
        vector_environment,
        n_steps=8,
        batch_size=8,
        n_epochs=2,
        gamma=0.99 ** 0.25,
        gae_lambda=0.95 ** 0.25,
        learning_rate=1e-4,
        clip_range=0.1,
        target_kl=None,
        max_grad_norm=0.5,
        actor_max_grad_norm=0.5,
        value_max_grad_norm=0.5,
        policy_kwargs={
            "net_arch": {"pi": [16], "vf": [16]},
            "activation_fn": torch.nn.SiLU,
            "log_std_init": np.log(0.05),
            "features_extractor_class": ResidualObservationExtractor,
            "features_extractor_kwargs": {"noise_log_prob_index": 7},
        },
        seed=5,
        device="cpu",
    )
    zero_initialize_ppo_residual_mean(model.policy)
    actor_parameters, value_parameters = model._separated_parameter_groups()
    actor_identifiers = {id(parameter) for parameter in actor_parameters}
    value_identifiers = {id(parameter) for parameter in value_parameters}
    assert actor_identifiers.isdisjoint(value_identifiers)
    assert actor_identifiers | value_identifiers == {
        id(parameter)
        for parameter in model.policy.parameters()
        if parameter.requires_grad
    }
    actor_before = [parameter.detach().clone() for parameter in actor_parameters]
    value_before = [parameter.detach().clone() for parameter in value_parameters]

    original_clip = torch.nn.utils.clip_grad_norm_
    clipped_groups = []

    def record_clip(parameters, max_norm, *args, **kwargs):
        parameters = list(parameters)
        clipped_groups.append(
            ({id(parameter) for parameter in parameters}, float(max_norm))
        )
        return original_clip(parameters, max_norm, *args, **kwargs)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", record_clip)
    model.learn(total_timesteps=16)
    assert model.ppo_optimizer_steps == 4
    assert len(clipped_groups) == 8
    for index, (identifiers, max_norm) in enumerate(clipped_groups):
        assert identifiers == (
            actor_identifiers if index % 2 == 0 else value_identifiers
        )
        assert max_norm == 0.5
    assert any(
        not torch.equal(before, after)
        for before, after in zip(actor_before, actor_parameters)
    )
    assert any(
        not torch.equal(before, after)
        for before, after in zip(value_before, value_parameters)
    )
    assert np.max(np.abs(model.rollout_buffer.rewards)) < 0.1


def test_resip_ppo_uses_fixed_std_and_independent_adamw(tmp_path):
    primitive, _ = make_wrapped()
    residual = PPOResidualUnitEnv(primitive, residual_scale=0.1)
    vector_environment = DummyVecEnv([lambda: residual])
    model = ResiPAlignedPPO(
        "MlpPolicy",
        vector_environment,
        n_steps=8,
        batch_size=8,
        n_epochs=2,
        gamma=0.999,
        gae_lambda=0.95,
        learning_rate=3e-4,
        clip_range=0.2,
        target_kl=0.1,
        max_grad_norm=1.0,
        actor_max_grad_norm=1.0,
        value_max_grad_norm=1.0,
        actor_learning_rate=3e-4,
        value_learning_rate=5e-3,
        policy_kwargs={
            "net_arch": {"pi": [16], "vf": [16]},
            "activation_fn": torch.nn.ReLU,
            "log_std_init": -1.0,
            "features_extractor_class": ResidualObservationExtractor,
            "features_extractor_kwargs": {"noise_log_prob_index": 7},
        },
        seed=5,
        device="cpu",
        schedule_total_iterations=2,
    )
    zero_initialize_ppo_residual_mean(model.policy)
    assert isinstance(model.actor_optimizer, torch.optim.AdamW)
    assert isinstance(model.value_optimizer, torch.optim.AdamW)
    assert not model.policy.log_std.requires_grad
    assert model.policy.action_net.bias is None
    assert torch.count_nonzero(model.policy.action_net.weight) == 0
    assert model.policy.value_net.bias.item() == pytest.approx(0.25)
    log_std_before = model.policy.log_std.detach().clone()
    actor_parameters, value_parameters = model._separated_parameter_groups()
    actor_before = [parameter.detach().clone() for parameter in actor_parameters]
    value_before = [parameter.detach().clone() for parameter in value_parameters]
    model.learn(total_timesteps=16)
    torch.testing.assert_close(model.policy.log_std, log_std_before, rtol=0, atol=0)
    assert model.ppo_optimizer_steps == 4
    assert model.actor_optimizer_steps == 4
    assert model.value_optimizer_steps == 4
    assert any(
        not torch.equal(before, after)
        for before, after in zip(actor_before, actor_parameters)
    )
    assert any(
        not torch.equal(before, after)
        for before, after in zip(value_before, value_parameters)
    )
    model_path = tmp_path / "resip_model"
    model.save(model_path)
    restored = ResiPAlignedPPO.load(
        model_path,
        env=vector_environment,
        device="cpu",
    )
    assert restored.actor_optimizer_steps == 4
    assert restored.value_optimizer_steps == 4
    assert restored.actor_optimizer.state_dict()["state"]
    assert restored.value_optimizer.state_dict()["state"]
    for left, right in zip(
        model.policy.parameters(),
        restored.policy.parameters(),
    ):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


def test_counting_ppo_zero_mean_and_optimizer_count():
    primitive, _ = make_wrapped()
    environment = PPOResidualUnitEnv(primitive, residual_scale=0.1)
    vector_environment = DummyVecEnv([lambda: environment])
    model = CountingPPO(
        "MlpPolicy",
        vector_environment,
        n_steps=8,
        batch_size=8,
        n_epochs=2,
        gamma=0.99 ** 0.25,
        gae_lambda=0.95 ** 0.25,
        learning_rate=1e-4,
        clip_range=0.1,
        target_kl=0.01,
        max_grad_norm=0.5,
        policy_kwargs={
            "net_arch": {"pi": [16], "vf": [16]},
            "activation_fn": torch.nn.SiLU,
            "log_std_init": np.log(0.05),
            "features_extractor_class": ResidualObservationExtractor,
            "features_extractor_kwargs": {"noise_log_prob_index": 7},
        },
        seed=5,
        device="cpu",
    )
    zero_initialize_ppo_residual_mean(model.policy)
    observation = vector_environment.reset()
    residual, _ = model.predict(observation, deterministic=True)
    np.testing.assert_array_equal(residual, np.zeros((1, 1), dtype=np.float32))
    model.learn(total_timesteps=16)
    assert model.ppo_optimizer_steps == 4
    assert model.num_timesteps == 16


def test_noise_entropy_is_applied_once_on_entering_phase_zero():
    environment, _ = make_wrapped()
    model = make_model(environment)
    observation, _ = environment.reset(seed=1)
    batch = np.repeat(observation[None], 2, axis=0)
    batch[0, 3:7] = [1.0, 0.0, 0.0, 0.0]
    batch[1, 3:7] = [0.0, 1.0, 0.0, 0.0]
    batch[:, 7] = -2.0
    entropy = model._next_entropy_term(torch.as_tensor(batch))
    assert entropy[0].item() == pytest.approx(-0.4)
    assert entropy[1].item() == pytest.approx(0.0)


def test_training_updates_only_critic_and_residual_actor():
    environment, _ = make_wrapped()
    vector_environment = DummyVecEnv([lambda: environment])
    model = make_model(vector_environment)
    model.set_logger(configure(folder=None, format_strings=[]))
    observation = vector_environment.reset()
    for _ in range(80):
        base_action = observation[:, 2:3]
        next_observation, reward, done, infos = vector_environment.step(base_action)
        model.replay_buffer.add(
            observation,
            next_observation,
            base_action,
            reward,
            done,
            infos,
        )
        observation = next_observation
    shell_actor_before = copy.deepcopy(model.policy.actor.state_dict())
    critic_before = copy.deepcopy(model.critic.state_dict())
    residual_before = copy.deepcopy(model.residual_actor.state_dict())
    model.train(gradient_steps=20, batch_size=16)
    for name, value in model.policy.actor.state_dict().items():
        torch.testing.assert_close(value, shell_actor_before[name], rtol=0, atol=0)
    assert any(
        not torch.equal(value, critic_before[name])
        for name, value in model.critic.state_dict().items()
    )
    assert any(
        not torch.equal(value, residual_before[name])
        for name, value in model.residual_actor.state_dict().items()
    )
    assert model.action_critic_optimizer_steps == 20
    assert model.residual_actor_optimizer_steps == 5
    assert all(parameter.grad is None for parameter in model.critic.parameters())
    assert all(
        parameter.grad is None for parameter in model.residual_actor.parameters()
    )
    assert all(
        parameter.grad is None for parameter in model.policy.actor.parameters()
    )


def test_critic_pretraining_keeps_residual_exactly_zero():
    environment, _ = make_wrapped()
    vector_environment = DummyVecEnv([lambda: environment])
    model = make_model(vector_environment)
    observation = vector_environment.reset()
    for _ in range(80):
        base_action = observation[:, 2:3]
        next_observation, reward, done, infos = vector_environment.step(base_action)
        model.replay_buffer.add(
            observation,
            next_observation,
            base_action,
            reward,
            done,
            infos,
        )
        observation = next_observation
    residual_before = copy.deepcopy(model.residual_actor.state_dict())
    losses = model.pretrain_action_critic(gradient_steps=25, batch_size=16)
    assert len(losses) == 25
    assert model.critic_pretrain_optimizer_steps == 25
    assert model.action_critic_optimizer_steps == 25
    assert model.residual_actor_optimizer_steps == 0
    assert model.per_step_train_calls == 0
    for name, value in model.residual_actor.state_dict().items():
        torch.testing.assert_close(value, residual_before[name], rtol=0, atol=0)


def test_save_load_preserves_deterministic_execution(tmp_path):
    environment, _ = make_wrapped()
    model = make_model(environment)
    observation, _ = environment.reset(seed=9)
    with torch.no_grad():
        model.residual_actor.output_layer.bias.fill_(0.2)
    before, _ = model.predict_with_components(observation)
    path = tmp_path / "model"
    model.save(path)
    restored = PerStepResidualDSRL.load(
        path,
        env=environment,
        device="cpu",
    )
    after, _ = restored.predict_with_components(observation)
    for key in before:
        np.testing.assert_array_equal(before[key], after[key])


def test_safe_boundary_segmented_resume_preserves_update_counts(tmp_path):
    environment, _ = make_wrapped()
    vector_environment = DummyVecEnv([lambda: environment])
    model = make_model(vector_environment)
    observation = vector_environment.reset()
    for _ in range(80):
        base_action = observation[:, 2:3]
        next_observation, reward, done, infos = vector_environment.step(base_action)
        model.replay_buffer.add(
            observation,
            next_observation,
            base_action,
            reward,
            done,
            infos,
        )
        observation = next_observation
    model.pretrain_action_critic(gradient_steps=25, batch_size=16)
    model.learn(total_timesteps=16, reset_num_timesteps=False)
    assert model.num_timesteps == 16
    assert model.per_step_train_calls == 2
    assert model.action_critic_optimizer_steps == 65
    assert model.residual_actor_optimizer_steps == 10
    model_path = tmp_path / "segment_model"
    replay_path = tmp_path / "segment_replay.pkl"
    model.save(model_path)
    model.save_replay_buffer(replay_path)

    resumed_environment, _ = make_wrapped()
    resumed_vector = DummyVecEnv([lambda: resumed_environment])
    resumed = PerStepResidualDSRL.load(
        model_path,
        env=resumed_vector,
        device="cpu",
    )
    resumed.load_replay_buffer(replay_path)
    resumed.learn(total_timesteps=16, reset_num_timesteps=False)
    assert resumed.num_timesteps == 32
    assert resumed.per_step_train_calls == 4
    assert resumed.action_critic_optimizer_steps == 105
    assert resumed.critic_pretrain_optimizer_steps == 25
    assert resumed.residual_actor_optimizer_steps == 20
