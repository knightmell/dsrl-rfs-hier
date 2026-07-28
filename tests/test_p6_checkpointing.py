from __future__ import annotations

import json
import pickle

import numpy as np
import pytest
from gymnasium import spaces

from p6_checkpointing import (
    P6CheckpointManager,
    P6IntentionalInterruption,
    P6TrainingCallback,
    initial_runtime_state,
    load_resume_payload,
    validate_loaded_resume,
)
from stable_baselines3.common.buffers import ReplayBuffer


def make_replay():
    replay = ReplayBuffer(
        buffer_size=20,
        observation_space=spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32),
        action_space=spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32),
        device="cpu",
        n_envs=2,
    )
    replay.add(
        obs=np.zeros((2, 2), dtype=np.float32),
        next_obs=np.ones((2, 2), dtype=np.float32),
        action=np.zeros((2, 2), dtype=np.float32),
        reward=np.zeros(2, dtype=np.float32),
        done=np.zeros(2, dtype=np.bool_),
        infos=[{}, {}],
    )
    replay.final_offline_step()
    return replay


def add_online_replay_step(replay):
    replay.add(
        obs=np.full((2, 2), 0.25, dtype=np.float32),
        next_obs=np.full((2, 2), 0.5, dtype=np.float32),
        action=np.zeros((2, 2), dtype=np.float32),
        reward=np.ones(2, dtype=np.float32),
        done=np.zeros(2, dtype=np.bool_),
        infos=[{}, {}],
    )


class SerializableModel:
    def __init__(self):
        self.num_timesteps = 0
        self.replay_buffer = make_replay()
        self.action_critic_optimizer_steps = 0
        self.modulation_critic_optimizer_steps = 0
        self.noise_actor_optimizer_steps = 0
        self.residual_actor_optimizer_steps = 0
        self.hierarchy_train_calls = 0

    def save(self, path):
        path.write_bytes(
            json.dumps(
                {
                    "num_timesteps": self.num_timesteps,
                    "action_critic_optimizer_steps": (
                        self.action_critic_optimizer_steps
                    ),
                },
                sort_keys=True,
            ).encode()
        )

    def save_replay_buffer(self, path):
        with path.open("wb") as output:
            pickle.dump(self.replay_buffer, output)


def provenance():
    return {
        "init_checkpoint_sha256": "a" * 64,
        "frozen_ddim_sha256": "b" * 64,
        "normalization_sha256": "c" * 64,
    }


def runtime_state():
    return initial_runtime_state(
        target_chunk_budget=4,
        action_chunk=4,
        prefill_metadata={
            "primitive_counters": {
                "chunk_transitions": 2,
                "nominal_primitive_steps": 8,
                "actual_primitive_env_steps": 8,
                "skipped_primitive_steps_due_to_termination": 0,
                "early_termination_chunks": 0,
                "terminal_chunks": 0,
                "timeout_chunks": 0,
            },
            "semantic_hash": "prefill-hash",
        },
        replay_hash={
            "semantic_hash": "initial-replay-hash",
        },
        online_eval_interval=2,
        model_checkpoint_interval=2,
        replay_checkpoint_interval=2,
    )


def make_manager(tmp_path, state=None):
    run_directory = tmp_path / "run"
    run_directory.mkdir()
    manifest_path = run_directory / "run_manifest.json"
    manifest_path.write_text("{}\n")
    evaluations = []

    def evaluate(chunk, counters):
        result = {
            "summary": {"raw_return_mean": float(chunk)},
            "counter_chunks": counters.chunk_transitions,
        }
        evaluations.append((chunk, counters.chunk_transitions))
        return result

    manager = P6CheckpointManager(
        run_directory=run_directory,
        algorithm="dsrl_na_rfs_hier",
        manifest_path=manifest_path,
        runtime_state=state or runtime_state(),
        provenance=provenance(),
        evaluation_function=evaluate,
    )
    return manager, evaluations


def two_training_infos():
    return [
        {
            "nominal_primitive_steps": 4,
            "actual_primitive_steps": 4,
            "early_termination_within_chunk": False,
            "TimeLimit.truncated": False,
        },
        {
            "nominal_primitive_steps": 4,
            "actual_primitive_steps": 2,
            "early_termination_within_chunk": True,
            "TimeLimit.truncated": False,
        },
    ]


def test_safe_boundary_writes_consistent_bundle_and_validates_resume(tmp_path):
    manager, evaluations = make_manager(tmp_path)
    model = SerializableModel()
    manager.service_safe_boundary(model)
    assert evaluations == [(0, 0)]

    counters = manager.runtime_state["training_counters"]
    counters.update(
        {
            "chunk_transitions": 2,
            "nominal_primitive_steps": 8,
            "actual_primitive_env_steps": 6,
            "skipped_primitive_steps_due_to_termination": 2,
            "early_termination_chunks": 1,
            "terminal_chunks": 1,
            "timeout_chunks": 0,
        }
    )
    model.num_timesteps = 2
    model.action_critic_optimizer_steps = 20
    add_online_replay_step(model.replay_buffer)
    manager.service_safe_boundary(model)

    bundle = manager.run_directory / "resume" / "chunk_000000000002"
    assert (bundle / "COMPLETE").is_file()
    assert (manager.run_directory / "checkpoints" / "model_000000000002.zip").is_file()
    assert evaluations == [(0, 0), (2, 2)]
    bundle_manifest, payload = load_resume_payload(
        bundle,
        algorithm="dsrl_na_rfs_hier",
        expected_provenance=provenance(),
    )
    assert payload["runtime_state"]["last_replay_checkpoint_chunk"] == 2
    validate_loaded_resume(
        model=model,
        bundle_directory=bundle,
        bundle_manifest=bundle_manifest,
        runtime_payload=payload,
    )

    with (bundle / "replay_buffer.pkl").open("ab") as output:
        output.write(b"corruption")
    with pytest.raises(ValueError, match="replay_sha256 mismatch"):
        load_resume_payload(
            bundle,
            algorithm="dsrl_na_rfs_hier",
            expected_provenance=provenance(),
        )


def test_callback_counts_actual_primitives_and_interrupts_only_after_bundle(tmp_path):
    manager, _ = make_manager(tmp_path)
    model = SerializableModel()
    callback = P6TrainingCallback(
        manager=manager,
        action_chunk=4,
        stop_after_chunk_transitions=2,
    )
    callback.model = model
    callback._on_rollout_start()

    model.num_timesteps = 2
    callback.locals = {
        "infos": two_training_infos(),
        "dones": np.array([False, True]),
    }
    assert callback._on_step() is True
    assert manager.runtime_state["training_counters"] == {
        "chunk_transitions": 2,
        "nominal_primitive_steps": 8,
        "actual_primitive_env_steps": 6,
        "skipped_primitive_steps_due_to_termination": 2,
        "early_termination_chunks": 1,
        "terminal_chunks": 1,
        "timeout_chunks": 0,
    }

    add_online_replay_step(model.replay_buffer)
    with pytest.raises(P6IntentionalInterruption):
        callback._on_rollout_start()
    assert (
        manager.run_directory
        / "resume"
        / "chunk_000000000002"
        / "COMPLETE"
    ).is_file()
