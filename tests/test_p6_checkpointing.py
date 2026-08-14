from __future__ import annotations

import json
import pickle
from types import SimpleNamespace

import numpy as np
import pytest
from gymnasium import spaces

from p6_checkpointing import (
    P6CheckpointManager,
    P6IntentionalInterruption,
    P6TrainingCallback,
    initial_runtime_state,
    load_resume_payload,
    run_binding_from_manifest,
    validate_loaded_resume,
    validate_optimizer_counter_invariants,
    _replay_storage_bytes,
)
from p6_runtime import hash_replay_prefix
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
    """Mock carrying the three-critic dsrl_na_rfs_hier Core V1 schema."""

    def __init__(self):
        self.num_timesteps = 0
        self.replay_buffer = make_replay()
        self.architecture_version = "dsrl_na_rfs_hier_three_critic_v1"
        self.replay_schema_version = "dsrl_na_rfs_hier_replay_v1"
        self.hierarchy_schedule = SimpleNamespace(
            profile_name="residual",
            phase_b_steps=1_000_000,
            phase_r_steps=1_000_000,
            phase_j_steps=1_000_000,
            phase_j_enabled=True,
            beta_ramp_steps=50_000,
            beta_target=1.0,
            base_lane_probability=0.5,
        )
        self.environment_discontinuity_count = 0
        self.target_update_interval = 1
        self.gradient_steps = 20
        self.noise_critic_grad_steps = 10
        self.actor_gradient_steps = -1
        self.n_envs = 2
        self.train_freq = SimpleNamespace(frequency=1, unit="step")
        self._pending_rollout_metadata = None
        # Retained for the dsrl_na_control legacy-schema tests.
        self.action_critic_optimizer_steps = 0
        self.modulation_critic_optimizer_steps = 0
        self.set_residual_counters(0)

    def set_residual_counters(self, calls):
        """Set the RESIDUAL-phase per-call counter formula for N train calls.

        After N calls (train_freq=1, n_envs=2): qa_base=10N, qa_joint=10N,
        qw_base=5N, noise=0, alpha=0, residual=N; targets 10N/10N/N;
        versions 0/N; _n_updates=26N; num_timesteps=2N.
        """
        calls = int(calls)
        self.num_timesteps = 2 * calls
        self.hierarchy_train_calls = calls
        self.qa_base_optimizer_steps = 10 * calls
        self.qa_joint_optimizer_steps = 10 * calls
        self.qw_base_optimizer_steps = 5 * calls
        self.noise_actor_optimizer_steps = 0
        self.alpha_optimizer_steps = 0
        self.residual_actor_optimizer_steps = calls
        self.qa_base_target_updates = 10 * calls
        self.qa_joint_target_updates = 10 * calls
        self.residual_target_updates = calls
        self.noise_policy_version = 0
        self.residual_policy_version = calls
        self.qa_joint_generation = calls
        self.qa_joint_optimizer_steps_since_clone = calls
        self.base_block_skips = 0
        self.joint_block_skips = 0
        self._n_updates = 26 * calls
        self.requested_optimizer_steps = {
            "qa_base": 10 * calls,
            "qa_joint": 10 * calls,
            "qw_base": 5 * calls,
            "noise_actor": 0,
            "alpha": 0,
            "residual_actor": calls,
        }

    def save(self, path):
        path.write_bytes(
            json.dumps(
                {
                    "num_timesteps": self.num_timesteps,
                    "hierarchy_train_calls": self.hierarchy_train_calls,
                },
                sort_keys=True,
            ).encode()
        )

    def save_replay_buffer(self, path):
        with path.open("wb") as output:
            pickle.dump(self.replay_buffer, output)


class RecordingLogger:
    def __init__(self):
        self.values = {}
        self.dumps = []

    def record(self, name, value):
        self.values[name] = value

    def dump(self, step):
        self.dumps.append((step, dict(self.values)))


def provenance():
    initial_hash = hash_replay_prefix(make_replay())
    return {
        "init_checkpoint_sha256": "a" * 64,
        "frozen_ddim_sha256": "b" * 64,
        "normalization_sha256": "c" * 64,
        "run_id": "p6-test-run",
        "config_contract_sha256": "d" * 64,
        "source_state_sha256": "e" * 64,
        "prefill_semantic_hash": "prefill-hash",
        "initial_replay_hash": initial_hash,
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
            "semantic_hash": provenance()["initial_replay_hash"][
                "semantic_hash"
            ],
        },
        online_eval_interval=2,
        model_checkpoint_interval=2,
        replay_checkpoint_interval=2,
        run_binding=run_binding_from_manifest(provenance()),
    )


def make_manager(tmp_path, state=None):
    run_directory = tmp_path / "run"
    run_directory.mkdir()
    manifest_path = run_directory / "run_manifest.json"
    manifest_path.write_text(json.dumps(provenance()) + "\n")
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
    model.set_residual_counters(1)
    add_online_replay_step(model.replay_buffer)
    manager.service_safe_boundary(model)

    bundle = manager.run_directory / "resume" / "chunk_000000000002"
    assert (bundle / "COMPLETE").is_file()
    assert (manager.run_directory / "checkpoints" / "model_000000000002.zip").is_file()
    assert evaluations == [(0, 0), (2, 2)]
    bundle_manifest, payload = load_resume_payload(
        bundle,
        algorithm="dsrl_na_rfs_hier",
        expected_run_manifest=provenance(),
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
            expected_run_manifest=provenance(),
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

    model.set_residual_counters(1)
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


@pytest.mark.parametrize(
    "field",
    (
        "run_id",
        "config_contract_sha256",
        "source_state_sha256",
        "prefill_semantic_hash",
        "initial_replay_semantic_hash",
    ),
)
def test_resume_bundle_rejects_cross_run_binding(tmp_path, field):
    manager, _ = make_manager(tmp_path)
    model = SerializableModel()
    model.set_residual_counters(1)
    manager.runtime_state["training_counters"].update(
        {
            "chunk_transitions": 2,
            "nominal_primitive_steps": 8,
            "actual_primitive_env_steps": 8,
        }
    )
    add_online_replay_step(model.replay_buffer)
    bundle = manager.save_resume_bundle(model, 2)
    mismatched = provenance()
    if field == "initial_replay_semantic_hash":
        mismatched["initial_replay_hash"]["semantic_hash"] = "other"
    else:
        mismatched[field] = "other"

    with pytest.raises(ValueError, match=field):
        load_resume_payload(
            bundle,
            algorithm="dsrl_na_rfs_hier",
            expected_run_manifest=mismatched,
        )


def test_optimizer_counter_formula_rejects_self_consistent_wrong_metadata():
    model = SerializableModel()
    model.set_residual_counters(1)
    # Everything is valid for a RESIDUAL 1-call boundary, then corrupt exactly
    # one counter so the three-critic cadence invariant is violated.
    model.qa_base_target_updates = 9  # expected 10 for qa_base_optimizer_steps=10

    with pytest.raises(ValueError, match="target cadence differs"):
        validate_optimizer_counter_invariants(model, "dsrl_na_rfs_hier")


def test_control_optimizer_counter_formula_is_independent_of_hierarchy():
    model = SerializableModel()
    # The control graph uses the legacy flat counter schema, not three-critic.
    model.architecture_version = None
    model.num_timesteps = 4
    model.hierarchy_train_calls = 2
    model._n_updates = 40
    model.action_critic_optimizer_steps = 40
    model.modulation_critic_optimizer_steps = 20
    model.noise_actor_optimizer_steps = 40
    model.residual_actor_optimizer_steps = 0

    validate_optimizer_counter_invariants(model, "dsrl_na_control")
    model.residual_actor_optimizer_steps = 1
    with pytest.raises(ValueError, match="counter invariant"):
        validate_optimizer_counter_invariants(model, "dsrl_na_control")


def test_frozen_noise_optimizer_counter_formula_requires_zero_actor_steps():
    model = SerializableModel()
    with pytest.raises(ValueError, match="deprecated"):
        validate_optimizer_counter_invariants(
            model,
            "dsrl_na_rfs_hier_frozen_noise",
        )


def test_optimizer_counters_must_match_online_chunk_boundaries():
    model = SerializableModel()
    # RESIDUAL counters for 1 call (num_timesteps=2) are valid; bumping
    # num_timesteps to 4 without a second call violates the chunk boundary.
    model.set_residual_counters(1)
    model.num_timesteps = 4

    with pytest.raises(ValueError, match="train-call/chunk invariant"):
        validate_optimizer_counter_invariants(model, "dsrl_na_rfs_hier")

    model.set_residual_counters(2)
    validate_optimizer_counter_invariants(model, "dsrl_na_rfs_hier")


def test_replayed_milestone_preserves_prior_attempt_artifacts(tmp_path):
    manager, _ = make_manager(tmp_path)
    model = SerializableModel()
    model.set_residual_counters(1)
    manager.runtime_state["training_counters"].update(
        {
            "chunk_transitions": 2,
            "nominal_primitive_steps": 8,
            "actual_primitive_env_steps": 8,
        }
    )
    add_online_replay_step(model.replay_buffer)

    first_model = manager.save_model_snapshot(model, 2)
    first_bundle = manager.save_resume_bundle(model, 2)
    first_model_bytes = first_model.read_bytes()
    first_bundle_manifest = (first_bundle / "bundle_manifest.json").read_bytes()

    manifest = json.loads(manager.manifest_path.read_text())
    manifest["current_attempt_id"] = 1
    manager.manifest_path.write_text(json.dumps(manifest) + "\n")
    second_model = manager.save_model_snapshot(model, 2)
    second_bundle = manager.save_resume_bundle(model, 2)

    assert second_model.name == "model_000000000002_attempt0001.zip"
    assert second_bundle.name == "chunk_000000000002_attempt0001"
    assert first_model.read_bytes() == first_model_bytes
    assert (first_bundle / "bundle_manifest.json").read_bytes() == (
        first_bundle_manifest
    )


def test_flush_final_rejects_preexisting_unauthenticated_model(tmp_path):
    manager, _ = make_manager(tmp_path)
    model = SerializableModel()
    garbage = manager.run_directory / "checkpoints" / "final_model.zip"
    garbage.write_bytes(b"not-a-certified-model")

    with pytest.raises(FileExistsError, match="unauthenticated final model"):
        manager.flush_final(model)
    assert garbage.read_bytes() == b"not-a-certified-model"


def test_replay_storage_estimate_covers_every_numpy_array():
    replay = make_replay()
    replay.noise_actions = np.zeros((7, 3), dtype=np.float64)
    expected = sum(
        int(value.nbytes)
        for value in vars(replay).values()
        if isinstance(value, np.ndarray)
    )

    assert _replay_storage_bytes(replay) >= expected


def test_safe_boundary_flushes_counters_before_checkpoint_service(tmp_path):
    manager, _ = make_manager(tmp_path)
    model = SerializableModel()
    model.logger = RecordingLogger()

    manager.service_safe_boundary(model)

    assert len(model.logger.dumps) == 1
    step, values = model.logger.dumps[0]
    assert step == 0
    assert values["p6/chunk_transitions"] == 0
    assert values["p6/nominal_primitive_steps"] == 0
    assert values["p6/actual_primitive_env_steps"] == 0
    assert values["train/residual_actor_optimizer_steps"] == 0
