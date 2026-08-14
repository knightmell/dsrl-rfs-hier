from __future__ import annotations

from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch
from gymnasium import spaces

from asymmetric_dsrl_residual_ppo import (
    ActorSnapshotState,
    InteractionCounters,
    StreamIsolationGuard,
    assert_parameter_storage_disjoint,
    build_three_seven_cycles,
    module_state_hash,
)
from evaluate_asymmetric_checkpoints import paired_summary
from per_step_residual_env import (
    FrozenDSRLPrimitiveResidualEnv,
    FrozenPlan,
)


def test_three_seven_10k_budget_and_expected_update_counts():
    cycles = build_three_seven_cycles(
        total_equivalent_chunks=10_000,
        cycle_equivalent_chunks=1_000,
        action_chunk=4,
        base_fraction=0.3,
    )
    assert len(cycles) == 10
    assert {cycle.base_chunk_transitions for cycle in cycles} == {300}
    assert {cycle.joint_primitive_transitions for cycle in cycles} == {2_800}
    assert sum(cycle.base_chunk_transitions for cycle in cycles) == 3_000
    assert sum(cycle.joint_equivalent_chunks for cycle in cycles) == 7_000
    assert 3_000 // (10 * 2) * 20 == 3_000
    assert 3_000 // (10 * 2) * 10 == 1_500
    assert 10 * (2_800 // 400) * 5 == 350


def test_interaction_counters_keep_nominal_and_actual_primitive_steps():
    cycles = build_three_seven_cycles(
        total_equivalent_chunks=10_000,
        cycle_equivalent_chunks=1_000,
        action_chunk=4,
        base_fraction=0.3,
    )
    counters = InteractionCounters(action_chunk=4)
    for index, cycle in enumerate(cycles):
        counters.record_cycle(
            cycle=cycle,
            base_actual_primitive_steps=1_200 - index,
        )
    values = counters.to_dict()
    assert values["base_chunk_transitions"] == 3_000
    assert values["joint_primitive_transitions"] == 28_000
    assert values["total_equivalent_chunks"] == 10_000
    assert values["nominal_primitive_steps"] == 40_000
    assert values["actual_primitive_env_steps"] == 39_955
    assert values["skipped_primitive_steps_due_to_base_termination"] == 45


def test_paired_summary_reports_directional_fall_transitions():
    def result(returns, falls):
        return {
            "episodes": [
                {
                    "environment_seed": 100 + index,
                    "raw_return": value,
                    "early_fall": fall,
                }
                for index, (value, fall) in enumerate(zip(returns, falls))
            ]
        }

    summary = paired_summary(
        base=result([10.0, 10.0, 4.0, 4.0], [False, False, True, True]),
        treatment=result(
            [11.0, 3.0, 9.0, 2.0],
            [False, True, False, True],
        ),
        bootstrap_samples=100,
    )
    assert summary["paired_return_difference_mean"] == pytest.approx(-0.75)
    assert summary["base_success_treatment_fall_count"] == 1
    assert summary["base_fall_treatment_success_count"] == 1
    assert summary["both_success_count"] == 1
    assert summary["both_fall_count"] == 1


def test_actor_snapshot_is_exact_without_parameter_aliasing():
    torch.manual_seed(1)
    live = torch.nn.Sequential(
        torch.nn.Linear(3, 4),
        torch.nn.SiLU(),
        torch.nn.Linear(4, 2),
    )
    snapshot = torch.nn.Sequential(
        torch.nn.Linear(3, 4),
        torch.nn.SiLU(),
        torch.nn.Linear(4, 2),
    )
    assert_parameter_storage_disjoint(live, snapshot)
    state = ActorSnapshotState()
    copied_hash = state.commit(
        live_actor=live,
        snapshot_actor=snapshot,
    )
    assert state.version == 1
    assert copied_hash == module_state_hash({"actor": live})
    state.assert_unchanged(snapshot)
    with torch.no_grad():
        next(live.parameters()).add_(1)
    assert module_state_hash({"actor": live}) != copied_hash
    state.assert_unchanged(snapshot)
    assert all(not parameter.requires_grad for parameter in snapshot.parameters())


def test_stream_guard_enforces_one_way_parameter_ownership():
    live = {"live": torch.nn.Linear(2, 2)}
    snapshot = {"snapshot": torch.nn.Linear(2, 2)}
    residual = torch.nn.Linear(2, 1)
    guard = StreamIsolationGuard(
        live_modules=live,
        snapshot_modules=snapshot,
        residual_policy=residual,
    )

    base_token = guard.before_base()
    with torch.no_grad():
        live["live"].weight.add_(0.25)
    guard.after_base(base_token)

    joint_token = guard.before_joint()
    with torch.no_grad():
        residual.bias.add_(0.5)
    guard.after_joint(joint_token)

    bad_joint_token = guard.before_joint()
    with torch.no_grad():
        live["live"].bias.add_(0.1)
    with pytest.raises(RuntimeError, match="changed live DSRL"):
        guard.after_joint(bad_joint_token)


class _TinySim:
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


class _TinyPrimitive(gym.Env):
    def __init__(self) -> None:
        super().__init__()
        self.action_space = spaces.Box(-np.ones(1), np.ones(1), dtype=np.float32)
        self.observation_space = spaces.Box(
            -np.inf,
            np.inf,
            shape=(2,),
            dtype=np.float32,
        )
        self.sim = _TinySim()
        self.dt = 0.1
        self.np_random = np.random.RandomState(0)
        self._elapsed_steps = 0

    @property
    def unwrapped(self):
        return self

    def reset(self, *, seed=None, options=None):
        del options
        if seed is not None:
            self.np_random.seed(seed)
        self._elapsed_steps = 0
        self.sim.data.qpos[...] = [0.0, 1.0, 0.0]
        return np.asarray([0.0, 0.0], dtype=np.float32), {}

    def step(self, action):
        self._elapsed_steps += 1
        raw = float(np.asarray(action)[0])
        before = float(self.sim.data.qpos[0])
        self.sim.data.qpos[0] += 0.1 * (0.5 + raw)
        reward_forward = (float(self.sim.data.qpos[0]) - before) / self.dt
        reward = reward_forward + 1.0 - 1e-3 * raw * raw
        observation = np.asarray(
            [self.sim.data.qpos[0], float(self._elapsed_steps)],
            dtype=np.float32,
        )
        return observation, reward, False, False, {}


class _VersionedPlanner:
    action_chunk = 4
    action_dimension = 1

    def __init__(self) -> None:
        self.base_version = 0

    def plan(self, observation, *, deterministic):
        del observation, deterministic
        action = 0.1 * (self.base_version + 1)
        return FrozenPlan(
            action_base_chunk=np.full((4, 1), action, dtype=np.float32),
            noise_scaled=np.zeros(4, dtype=np.float32),
            noise_log_prob=0.0,
            base_version=self.base_version,
        )


def test_base_commit_refreshes_cached_plan_without_stepping_simulator():
    planner = _VersionedPlanner()
    environment = FrozenDSRLPrimitiveResidualEnv(
        _TinyPrimitive(),
        planner,
        raw_observation_dim=2,
        max_episode_steps=20,
    )
    observation, _ = environment.reset(seed=3)
    np.testing.assert_allclose(observation[2:3], [0.1], atol=1e-7)
    simulator_before = environment.env.sim.get_state()
    primitive_count_before = environment._primitive_count

    planner.base_version = 1
    refreshed = environment.refresh_plan()
    np.testing.assert_allclose(refreshed[2:3], [0.2], atol=1e-7)
    assert environment.phase == 0
    np.testing.assert_array_equal(
        environment.env.sim.get_state(),
        simulator_before,
    )
    assert environment._primitive_count == primitive_count_before

    _, _, _, _, info = environment.step(np.asarray([0.2], dtype=np.float32))
    assert info["base_snapshot_version"] == 1


def test_environment_snapshot_restores_refresh_observation_and_version():
    planner = _VersionedPlanner()
    environment = FrozenDSRLPrimitiveResidualEnv(
        _TinyPrimitive(),
        planner,
        raw_observation_dim=2,
        max_episode_steps=20,
    )
    environment.reset(seed=4)
    environment.step(np.asarray([0.1], dtype=np.float32))
    snapshot = environment.capture_state()
    planner.base_version = 7
    environment.refresh_plan()
    environment.restore_state(snapshot)
    restored = environment.refresh_plan()
    assert np.argmax(restored[3:7]) == 0
    assert restored[0] == pytest.approx(snapshot.raw_observation[0])
