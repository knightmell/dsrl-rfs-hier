from __future__ import annotations

import random
from types import SimpleNamespace

import numpy as np
import torch
from gymnasium import spaces

from env_utils import (
    ACTION_CHUNK_EARLY_BREAK,
    ACTION_CHUNK_LEGACY_CONTINUE,
    ActionChunkWrapper,
)
from p6_runtime import (
    capture_rng_state,
    collect_matched_prefill,
    hash_replay_prefix,
    load_prefill_artifact,
    populate_replay_buffer,
    restore_rng_state,
    save_prefill_artifact,
    seed_all,
    terminal_corrected_next_observations,
)
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.vec_env import DummyVecEnv


class PrimitiveTerminateEnv:
    observation_space = spaces.Box(-100.0, 100.0, shape=(1,), dtype=np.float32)
    action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)

    def __init__(self, terminate_at=2):
        self.terminate_at = terminate_at
        self.primitive_step = 0
        self.total_step_calls = 0

    def reset(self, seed=None, options=None):
        del seed, options
        self.primitive_step = 0
        return np.array([-100.0], dtype=np.float32)

    def step(self, action):
        del action
        self.primitive_step += 1
        self.total_step_calls += 1
        done = self.primitive_step == self.terminate_at
        return (
            np.array([float(self.primitive_step)], dtype=np.float32),
            1.0,
            done,
            {},
        )

    def close(self):
        return None


def chunk_cfg():
    return SimpleNamespace(act_steps=4, obs_dim=1)


def test_p6_action_chunk_breaks_at_terminal_and_explains_counter_gap():
    primitive_env = PrimitiveTerminateEnv(terminate_at=2)
    env = ActionChunkWrapper(
        primitive_env,
        chunk_cfg(),
        max_episode_steps=100,
        action_chunk_termination_semantics=ACTION_CHUNK_EARLY_BREAK,
    )
    env.reset()

    observation, reward, terminated, truncated, info = env.step(
        np.zeros(4, dtype=np.float32)
    )

    np.testing.assert_array_equal(observation, np.array([2.0], dtype=np.float32))
    assert reward == 2.0
    assert terminated is True
    assert truncated is False
    assert primitive_env.total_step_calls == 2
    assert info["nominal_primitive_steps"] == 4
    assert info["actual_primitive_steps"] == 2
    assert info["early_termination_within_chunk"] is True
    assert info["termination_primitive_index"] == 1
    assert info["termination_reason"] == "environment_terminal"


def test_legacy_action_chunk_default_retains_continue_after_done_behavior():
    primitive_env = PrimitiveTerminateEnv(terminate_at=2)
    env = ActionChunkWrapper(primitive_env, chunk_cfg(), max_episode_steps=100)
    assert env.action_chunk_termination_semantics == ACTION_CHUNK_LEGACY_CONTINUE
    env.reset()

    _, reward, terminated, truncated, info = env.step(
        np.zeros(4, dtype=np.float32)
    )

    assert reward == 4.0
    assert terminated is True
    assert truncated is False
    assert primitive_env.total_step_calls == 4
    assert info["actual_primitive_steps"] == 4
    assert info["early_termination_within_chunk"] is False


def test_terminal_observation_replaces_vec_env_autoreset_observation():
    reset_observation = np.array([[-100.0], [-200.0]], dtype=np.float32)
    terminal_observation = np.array([[2.0], [3.0]], dtype=np.float32)
    dones = np.array([True, False])
    infos = [
        {"terminal_observation": terminal_observation[0]},
        {},
    ]

    corrected = terminal_corrected_next_observations(
        reset_observation,
        dones,
        infos,
    )

    np.testing.assert_array_equal(corrected[0], terminal_observation[0])
    np.testing.assert_array_equal(corrected[1], reset_observation[1])


class StochasticWarmstart:
    def __init__(self, action_dimension):
        self.action_dimension = action_dimension

    def predict_diffused(self, observation, deterministic=False):
        assert deterministic is False
        action = torch.tanh(
            torch.randn(len(observation), self.action_dimension)
        ).numpy()
        return action, None


def _make_vector_env(n_envs=2):
    return DummyVecEnv(
        [
            lambda: ActionChunkWrapper(
                PrimitiveTerminateEnv(terminate_at=2),
                chunk_cfg(),
                max_episode_steps=100,
                action_chunk_termination_semantics=ACTION_CHUNK_EARLY_BREAK,
            )
            for _ in range(n_envs)
        ]
    )


def _provenance():
    return {
        "init_checkpoint_sha256": "a" * 64,
        "frozen_ddim_sha256": "b" * 64,
        "normalization_sha256": "c" * 64,
    }


def test_matched_prefill_artifact_replay_hash_and_rng_isolation(tmp_path):
    env = _make_vector_env()
    model = StochasticWarmstart(action_dimension=4)
    seed_all(91)
    before = capture_rng_state()
    expected = (
        random.random(),
        np.random.random(),
        torch.rand(1),
    )
    restore_rng_state(before)

    arrays, metadata = collect_matched_prefill(
        warmstart_model=model,
        env=env,
        vector_steps=3,
        environment_seed=123,
        policy_seed=456,
        action_chunk=4,
        termination_semantics=ACTION_CHUNK_EARLY_BREAK,
        provenance=_provenance(),
    )
    actual = (
        random.random(),
        np.random.random(),
        torch.rand(1),
    )
    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    torch.testing.assert_close(actual[2], expected[2], rtol=0.0, atol=0.0)
    assert metadata["chunk_transitions"] == 6
    assert metadata["primitive_counters"] == {
        "chunk_transitions": 6,
        "nominal_primitive_steps": 24,
        "actual_primitive_env_steps": 12,
        "skipped_primitive_steps_due_to_termination": 12,
        "early_termination_chunks": 6,
        "terminal_chunks": 6,
        "timeout_chunks": 0,
    }
    # The VecEnv return is an auto-reset observation; replay stores terminal 2.
    np.testing.assert_array_equal(
        arrays["next_observations"],
        np.full((3, 2, 1), 2.0, dtype=np.float32),
    )

    artifact = tmp_path / "matched_prefill.npz"
    saved_metadata = save_prefill_artifact(artifact, arrays, metadata)
    loaded_arrays, loaded_metadata = load_prefill_artifact(
        artifact,
        expected={
            key: saved_metadata[key]
            for key in (
                "format_version",
                "vector_steps",
                "chunk_transitions",
                "n_envs",
                "action_chunk",
                "environment_seed",
                "policy_seed",
                "termination_semantics",
                "init_checkpoint_sha256",
                "frozen_ddim_sha256",
                "normalization_sha256",
            )
        },
    )
    assert loaded_metadata["semantic_hash"] == saved_metadata["semantic_hash"]
    for name, expected_array in arrays.items():
        np.testing.assert_array_equal(loaded_arrays[name], expected_array)

    replay_hashes = []
    for _ in range(2):
        replay = ReplayBuffer(
            buffer_size=20,
            observation_space=spaces.Box(
                -100.0,
                100.0,
                shape=(1,),
                dtype=np.float32,
            ),
            action_space=spaces.Box(
                -1.0,
                1.0,
                shape=(4,),
                dtype=np.float32,
            ),
            device="cpu",
            n_envs=2,
        )
        populated_hash = populate_replay_buffer(replay, loaded_arrays)
        assert populated_hash == hash_replay_prefix(replay)
        replay_hashes.append(populated_hash)
    assert replay_hashes[0] == replay_hashes[1]
    env.close()
