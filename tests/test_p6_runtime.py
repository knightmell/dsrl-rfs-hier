from __future__ import annotations

import copy
import json
import multiprocessing
import os
import pickle
import queue
import random
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import p6_runtime
import pytest
import torch
from gymnasium import spaces

from env_utils import (
    ACTION_CHUNK_EARLY_BREAK,
    ACTION_CHUNK_LEGACY_CONTINUE,
    ActionChunkWrapper,
)
from p6_runtime import (
    PREFILL_SOURCE_FRESH,
    TAGGED_PREFILL_ARRAY_NAMES,
    TAGGED_PREFILL_FORMAT_VERSION,
    _process_unique_temporary_path,
    assert_matched_fresh_init_state_hashes,
    atomic_write_json,
    canonical_module_state_hash,
    capture_rng_state,
    collect_matched_prefill,
    collect_or_load_tagged_matched_prefill,
    collect_tagged_matched_prefill,
    collect_tagged_matched_prefill_and_save,
    hash_replay_for_resume,
    hash_replay_offline_for_resume,
    hash_replay_prefix,
    load_prefill_artifact,
    populate_replay_buffer,
    populate_tagged_replay_buffer,
    restore_rng_state,
    save_prefill_artifact,
    seed_all,
    terminal_corrected_next_observations,
    validate_prefill_semantics,
)
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.dsrl.hierarchical_replay_buffer import (
    BranchMode,
    HierarchyTaggedReplayBuffer,
    NoiseSampleSource,
    TerminationSemantics,
    TransitionOrigin,
)
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


def _timeout_prefill():
    arrays = {
        "observations": np.array([[[-100.0]]], dtype=np.float32),
        "next_observations": np.array([[[2.0]]], dtype=np.float32),
        "actions": np.zeros((1, 1, 4), dtype=np.float32),
        "rewards": np.ones((1, 1), dtype=np.float32),
        "dones": np.ones((1, 1), dtype=np.bool_),
        "timeouts": np.ones((1, 1), dtype=np.bool_),
        "actual_primitive_steps": np.array([[2]], dtype=np.int16),
        "early_termination_within_chunk": np.ones((1, 1), dtype=np.bool_),
    }
    metadata = {
        "format_version": 2,
        "vector_steps": 1,
        "chunk_transitions": 1,
        "n_envs": 1,
        "action_chunk": 4,
        "primitive_counters": {
            "chunk_transitions": 1,
            "nominal_primitive_steps": 4,
            "actual_primitive_env_steps": 2,
            "skipped_primitive_steps_due_to_termination": 2,
            "early_termination_chunks": 1,
            "terminal_chunks": 0,
            "timeout_chunks": 1,
        },
    }
    return arrays, metadata


def test_timeout_prefill_uses_terminal_observation_and_standard_sb3_mask():
    arrays, metadata = _timeout_prefill()
    counters = validate_prefill_semantics(arrays, metadata)
    assert counters.timeout_chunks == 1

    replay = ReplayBuffer(
        buffer_size=4,
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
        n_envs=1,
    )
    populate_replay_buffer(replay, arrays)

    assert replay.dones[0, 0] == 1
    assert replay.timeouts[0, 0] == 1
    np.testing.assert_array_equal(
        replay.next_observations[0, 0],
        np.array([2.0], dtype=np.float32),
    )
    sampled = replay._get_samples(np.array([0]))
    assert sampled.dones.item() == 0.0


@pytest.mark.parametrize(
    "mutation,match",
    (
        ("timeout_without_done", "timeout_without_done"),
        ("actual_zero", "actual_primitive_steps"),
        ("actual_too_large", "actual_primitive_steps"),
        ("early_mismatch", "early_termination"),
        ("counter_mismatch", "primitive_counters"),
    ),
)
def test_prefill_semantic_validation_rejects_tampering(mutation, match):
    arrays, metadata = _timeout_prefill()
    arrays = {name: value.copy() for name, value in arrays.items()}
    metadata = copy.deepcopy(metadata)
    if mutation == "timeout_without_done":
        arrays["dones"][0, 0] = False
    elif mutation == "actual_zero":
        arrays["actual_primitive_steps"][0, 0] = 0
    elif mutation == "actual_too_large":
        arrays["actual_primitive_steps"][0, 0] = 5
    elif mutation == "early_mismatch":
        arrays["early_termination_within_chunk"][0, 0] = False
    elif mutation == "counter_mismatch":
        metadata["primitive_counters"]["actual_primitive_env_steps"] += 1
    else:
        raise AssertionError(mutation)

    with pytest.raises(ValueError, match=match):
        validate_prefill_semantics(arrays, metadata)


class FreshGaussianPriorStub:
    """Minimal stand-in for the Frozen-DDIM hierarchy model on the fresh path:
    a Gaussian decoder prior that maps a standard-normal ``decoder`` straight to
    the action (identity DDIM), with an identity ``scale_action``."""

    device = torch.device("cpu")
    diffusion_act_chunk = 4
    diffusion_act_dim = 1

    class policy:
        @staticmethod
        def scale_action(noise):
            return np.asarray(noise, dtype=np.float32)

    def diffusion_policy(self, observation, decoder, return_numpy=False):
        action = decoder.reshape(observation.shape[0], -1)
        return action.numpy() if return_numpy else action


def _hold_prefill_lock_in_child(
    artifact_path: str,
    events: multiprocessing.Queue,
    hold_seconds: float,
) -> None:
    with p6_runtime._prefill_artifact_lock(Path(artifact_path)):
        events.put(("entered", os.getpid()))
        time.sleep(hold_seconds)
        events.put(("exited", os.getpid()))


def test_tagged_prefill_collect_and_save_round_trip(tmp_path):
    # Regression: collect_tagged_matched_prefill_and_save passed TAGGED arrays
    # (keyed by action_exec, not actions) to save_prefill_artifact, whose
    # standard validator raised "Prefill artifact is missing 'actions'".
    env = _make_vector_env()
    model = FreshGaussianPriorStub()
    artifact = tmp_path / "tagged_prefill.npz"
    provenance = _provenance()
    kwargs = dict(
        warmstart_model=model,
        env=env,
        vector_steps=3,
        environment_seed=123,
        policy_seed=456,
        action_chunk=4,
        termination_semantics=ACTION_CHUNK_EARLY_BREAK,
        provenance=provenance,
        prefill_source=PREFILL_SOURCE_FRESH,
    )

    arrays, metadata, generated = collect_tagged_matched_prefill_and_save(
        artifact_path=artifact,
        metadata_path=artifact.with_suffix(artifact.suffix + ".json"),
        **kwargs,
    )
    assert generated is True
    assert "action_exec" in arrays
    assert "actions" not in arrays
    assert metadata["format_version"] == TAGGED_PREFILL_FORMAT_VERSION
    assert artifact.is_file()

    # Reloading the persisted artifact validates tagged semantics and matches.
    loaded_arrays, loaded_metadata, generated = collect_or_load_tagged_matched_prefill(
        artifact_path=artifact,
        **kwargs,
    )
    assert generated is False
    for name in TAGGED_PREFILL_ARRAY_NAMES:
        np.testing.assert_array_equal(loaded_arrays[name], arrays[name])
    assert loaded_metadata["semantic_hash"] == metadata["semantic_hash"]

    # Regression: populate_tagged_replay_buffer validated with the WARMSTART
    # default, which rejects a FRESH prefill ("noise policy version 0"); it
    # must derive the source from the metadata instead.
    replay = HierarchyTaggedReplayBuffer(
        buffer_size=20,
        observation_space=spaces.Box(
            -100.0, 100.0, shape=(1,), dtype=np.float32
        ),
        action_space=spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32),
        device="cpu",
        n_envs=2,
    )
    result = populate_tagged_replay_buffer(replay, loaded_arrays, loaded_metadata)
    assert result["vector_steps"] == 3
    assert result["chunk_transitions"] == 6
    assert result["branch_counts"][0] == 6
    env.close()


def _base_online_metadata(
    replay: HierarchyTaggedReplayBuffer, row: int
) -> dict[str, np.ndarray]:
    """Build a valid BASE-branch metadata row for add_hierarchy (online fill)."""
    n_envs = replay.n_envs
    action = np.full((n_envs, replay.action_dim), 0.25, np.float32)
    return {
        "branch_mode": np.full(n_envs, int(BranchMode.BASE), np.uint8),
        "noise_scaled": np.full((n_envs, replay.action_dim), 0.5, np.float32),
        "noise_log_prob": np.full(n_envs, -0.5, np.float32),
        "noise_log_prob_valid": np.ones(n_envs, np.bool_),
        "noise_sample_source": np.full(
            n_envs, int(NoiseSampleSource.CURRENT_ACTOR), np.uint8
        ),
        "transition_origin": np.full(
            n_envs, int(TransitionOrigin.ONLINE), np.uint8
        ),
        "action_base": action.copy(),
        "residual_pre_tanh": np.zeros((n_envs, replay.action_dim), np.float32),
        "residual_unit": np.zeros((n_envs, replay.action_dim), np.float32),
        "action_residual_delta": np.zeros(
            (n_envs, replay.action_dim), np.float32
        ),
        "action_exec": action.copy(),
        "beta": np.zeros(n_envs, np.float32),
        "residual_applied": np.zeros(n_envs, np.bool_),
        "emergency_clamp_applied": np.zeros(n_envs, np.bool_),
        "episode_id": np.arange(n_envs, dtype=np.int64) + row * 100,
        "environment_id": np.arange(n_envs, dtype=np.int32),
        "chunk_index_in_episode": np.full(n_envs, row, np.int32),
        "nominal_primitive_steps": np.full(n_envs, 4, np.uint8),
        "actual_primitive_steps": np.full(n_envs, 4, np.uint8),
        "termination_primitive_index": np.full(n_envs, -1, np.int8),
        "termination_semantics": np.full(
            n_envs, int(TerminationSemantics.EARLY_BREAK_ON_DONE), np.uint8
        ),
        "noise_policy_version": np.zeros(n_envs, np.int64),
        "residual_policy_version": np.full(n_envs, -1, np.int64),
    }


def test_tagged_resume_hashes_agree_across_pickle_restore():
    # C5 integration: validate_loaded_resume compares the whole-buffer digest
    # (hash_replay_for_resume) and the offline-prefix digest
    # (hash_replay_offline_for_resume) across the save/restore boundary on the
    # TAGGERED path (the existing resume test uses a flat buffer).  Two C5
    # guarantees must hold on this path: the whole-buffer digest survives a
    # pickle restore (pos/full are preserved), and the offline-prefix digest is
    # rotation-invariant while the whole-buffer digest still tracks pos.
    env = _make_vector_env()
    model = FreshGaussianPriorStub()
    arrays, metadata = collect_tagged_matched_prefill(
        warmstart_model=model,
        env=env,
        vector_steps=3,
        environment_seed=123,
        policy_seed=456,
        action_chunk=4,
        termination_semantics=ACTION_CHUNK_EARLY_BREAK,
        provenance=_provenance(),
        prefill_source=PREFILL_SOURCE_FRESH,
    )
    # The offline prefix occupies vector_steps rows, one per env-step: each
    # add_hierarchy call writes a single row holding n_envs transitions and
    # increments pos by 1, so after the prefill pos == vector_steps.  Production
    # requires capacity beyond the prefix (buffer_size > vector_steps), so the
    # buffer is NOT full at prefill; online training fills the rest.
    replay = HierarchyTaggedReplayBuffer(
        buffer_size=20,
        observation_space=spaces.Box(
            -100.0, 100.0, shape=(1,), dtype=np.float32
        ),
        action_space=spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32),
        device="cpu",
        n_envs=2,
    )
    populate_tagged_replay_buffer(replay, arrays, metadata)
    assert not replay.full
    assert replay.pos == 3
    assert replay.offline_steps == 3

    save_whole = hash_replay_for_resume(replay)
    save_offline = hash_replay_offline_for_resume(replay)
    assert save_whole["vector_steps"] == 3
    assert save_offline["vector_steps"] == 3

    # Pickle restore (what save_resume_bundle's replay pickling round-trips)
    # must preserve pos/offline_steps so both digests agree at the save state.
    restored = pickle.loads(pickle.dumps(replay))
    assert (
        hash_replay_for_resume(restored)["semantic_hash"]
        == save_whole["semantic_hash"]
    )
    assert (
        hash_replay_offline_for_resume(restored)["semantic_hash"]
        == save_offline["semantic_hash"]
    )
    assert hash_replay_for_resume(restored)["offline_steps"] == 3

    # Online training continues from pos=3: fill the rest of the circular
    # buffer (7 more rows -> pos=10, full).  The prefill prefix rows are never
    # overwritten, so the offline-prefix digest is preserved while the
    # whole-buffer digest now covers all 10 rows with full=True.
    for row in range(3, replay.buffer_size):
        online = _base_online_metadata(replay, row)
        replay.add_hierarchy(
            obs=np.full((replay.n_envs, 1), 10.0 + row, np.float32),
            next_obs=np.full((replay.n_envs, 1), 10.0 + row + 1, np.float32),
            action=online["action_exec"],
            reward=np.full(replay.n_envs, 1.0, np.float32),
            done=np.zeros(replay.n_envs, np.bool_),
            infos=[{"TimeLimit.truncated": False}] * replay.n_envs,
            metadata=online,
        )
    assert replay.full
    assert replay.offline_steps == 3
    filled_whole = hash_replay_for_resume(replay)
    assert filled_whole["vector_steps"] == replay.buffer_size
    assert filled_whole["semantic_hash"] != save_whole["semantic_hash"]
    assert (
        hash_replay_offline_for_resume(replay)["semantic_hash"]
        == save_offline["semantic_hash"]
    )

    # A wrapped next-write index must not change the immutable prefill-prefix
    # digest -- the exact cross-state comparison C5 protects -- while the
    # whole-buffer digest still tracks pos.
    replay.pos = 2
    assert (
        hash_replay_offline_for_resume(replay)["semantic_hash"]
        == save_offline["semantic_hash"]
    )
    assert (
        hash_replay_for_resume(replay)["semantic_hash"]
        != filled_whole["semantic_hash"]
    )
    env.close()


def test_process_unique_temp_suffix_is_per_writer(tmp_path):
    # Regression (concurrent-write race): the shared prefill artifact is written
    # by A/B/control processes with the same seed, so the temp sibling name must
    # be private to each writer process.  A fixed ``path.tmp`` would have A
    # truncate B's half-written temp and B's os.replace fail after A renamed.
    target = tmp_path / "shared.json"
    real_getpid = os.getpid
    try:
        os.getpid = lambda: 1111
        writer_a = _process_unique_temporary_path(target)
        os.getpid = lambda: 2222
        writer_b = _process_unique_temporary_path(target)
    finally:
        os.getpid = real_getpid
    assert writer_a != writer_b
    assert writer_a.name == "shared.json.1111.tmp"
    assert writer_b.name == "shared.json.2222.tmp"


def test_prefill_bundle_lock_serializes_independent_processes(tmp_path):
    # The npz/json pair has a real inter-file publication window.  Unique temp
    # names alone do not stop a loader from deleting the first half while the
    # writer is about to publish the second.  Verify the production flock holds
    # that entire transaction across independent processes.
    context = multiprocessing.get_context("fork")
    events = context.Queue()
    artifact = tmp_path / "shared_tagged_prefill.npz"
    first = context.Process(
        target=_hold_prefill_lock_in_child,
        args=(str(artifact), events, 0.35),
    )
    second = context.Process(
        target=_hold_prefill_lock_in_child,
        args=(str(artifact), events, 0.0),
    )
    first.start()
    first_event = events.get(timeout=2.0)
    assert first_event[0] == "entered"
    second.start()
    with pytest.raises(queue.Empty):
        events.get(timeout=0.15)
    assert events.get(timeout=2.0) == ("exited", first_event[1])
    second_enter = events.get(timeout=2.0)
    assert second_enter[0] == "entered"
    assert events.get(timeout=2.0) == ("exited", second_enter[1])
    first.join(timeout=2.0)
    second.join(timeout=2.0)
    assert first.exitcode == 0
    assert second.exitcode == 0
    assert artifact.with_suffix(artifact.suffix + ".lock").is_file()


def test_generated_prefill_is_reloaded_from_published_bundle(tmp_path, monkeypatch):
    env = _make_vector_env()
    model = FreshGaussianPriorStub()
    artifact = tmp_path / "reload_after_generate.npz"
    real_collect_and_save = p6_runtime.collect_tagged_matched_prefill_and_save

    def poison_private_return(**kwargs):
        arrays, metadata, generated = real_collect_and_save(**kwargs)
        arrays["observations"].fill(12345.0)
        return arrays, metadata, generated

    monkeypatch.setattr(
        p6_runtime,
        "collect_tagged_matched_prefill_and_save",
        poison_private_return,
    )
    arrays, metadata, generated = collect_or_load_tagged_matched_prefill(
        artifact_path=artifact,
        warmstart_model=model,
        env=env,
        vector_steps=3,
        environment_seed=123,
        policy_seed=456,
        action_chunk=4,
        termination_semantics=ACTION_CHUNK_EARLY_BREAK,
        provenance=_provenance(),
        prefill_source=PREFILL_SOURCE_FRESH,
    )
    assert generated is True
    assert not np.any(arrays["observations"] == 12345.0)
    assert metadata["archive_sha256"] == p6_runtime.sha256_file(artifact)
    env.close()


def test_interleaved_writers_both_complete_and_last_writer_wins(tmp_path):
    # Reproduce the interleaving the fixed ``.tmp`` name used to break: A opens
    # its temp, B opens its temp, A renames, then B renames.  With a shared
    # temp name B's os.replace raised FileNotFoundError after A renamed; with
    # process-unique names both renames succeed and the complete content wins.
    target = tmp_path / "shared_prefill.json"
    real_getpid = os.getpid
    try:
        os.getpid = lambda: 1111
        temp_a = _process_unique_temporary_path(target)
        temp_a.write_text(json.dumps({"writer": "A", "seed": 1}), encoding="utf-8")
        os.getpid = lambda: 2222
        temp_b = _process_unique_temporary_path(target)
        temp_b.write_text(json.dumps({"writer": "B", "seed": 1}), encoding="utf-8")
        os.replace(temp_a, target)
        os.replace(temp_b, target)
    finally:
        os.getpid = real_getpid
    assert json.loads(target.read_text(encoding="utf-8")) == {
        "writer": "B",
        "seed": 1,
    }
    # No temp siblings may survive a completed write.
    assert [path.name for path in tmp_path.iterdir()] == ["shared_prefill.json"]


def test_atomic_write_json_through_public_path_leaves_no_temp(tmp_path):
    # The public helper must route through the process-unique temp name so the
    # concurrent-write fix applies everywhere the manifest is persisted.
    target = tmp_path / "run_manifest.json"
    atomic_write_json(target, {"legacy_loaded": True, "init_checkpoint_path": None})
    assert json.loads(target.read_text(encoding="utf-8")) == {
        "legacy_loaded": True,
        "init_checkpoint_path": None,
    }
    assert [path.name for path in tmp_path.iterdir()] == ["run_manifest.json"]


def test_failed_metadata_write_never_unlinks_peer_archive(tmp_path, monkeypatch):
    # Regression (orphan-cleanup race): save_prefill_artifact's except-handler
    # used to unlink the archive unconditionally when its metadata write failed.
    # Under the matched A/B/control design a peer can rename its (identical,
    # deterministic) archive onto the same path in the archive-rename → metadata
    # rename window, and the failed writer's cleanup must not delete that
    # completed peer artifact.  The unlink is now inode-guarded against it.
    env = _make_vector_env()
    model = StochasticWarmstart(action_dimension=4)
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
    artifact = tmp_path / "matched_prefill.npz"
    real_save_npz = p6_runtime._atomic_save_npz

    def failing_metadata_write(metadata_path, payload):
        # A concurrent peer completes its (identical) archive over ours, then
        # our own metadata write fails.
        del metadata_path, payload
        real_save_npz(artifact, arrays)
        raise RuntimeError("simulated metadata failure")

    monkeypatch.setattr(p6_runtime, "atomic_write_json", failing_metadata_write)
    with pytest.raises(RuntimeError, match="simulated metadata failure"):
        save_prefill_artifact(artifact, arrays, metadata)
    # The peer's archive survives the failed writer's cleanup; the metadata
    # write failed before publishing, so only the archive is present.
    assert artifact.is_file()
    assert not artifact.with_suffix(artifact.suffix + ".json").exists()


def test_archive_write_failure_reprops_original_error_and_keeps_peer_archive(
    tmp_path, monkeypatch
):
    # Regression (unbound local): save_prefill_artifact's except-handler used
    # to evaluate `path.stat().st_ino == archive_inode` even when
    # _atomic_save_npz raised before its rename, so `archive_inode` was never
    # assigned and the handler replaced the real error (disk-full / EACCES)
    # with UnboundLocalError.  When the archive write itself fails the cleanup
    # must do nothing and the original exception must propagate unchanged.
    env = _make_vector_env()
    model = StochasticWarmstart(action_dimension=4)
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
    artifact = tmp_path / "matched_prefill.npz"

    def raising_save_npz(path, arrays):
        # A peer's completed archive is already at the path.
        del path, arrays
        artifact.write_bytes(b"peer archive")
        raise OSError("simulated disk-full during npz write")

    monkeypatch.setattr(p6_runtime, "_atomic_save_npz", raising_save_npz)
    with pytest.raises(OSError, match="simulated disk-full"):
        save_prefill_artifact(artifact, arrays, metadata)
    # The peer's archive survives untouched (nothing of ours reached the path).
    assert artifact.read_bytes() == b"peer archive"
    assert not artifact.with_suffix(artifact.suffix + ".json").exists()


def test_metadata_published_then_fsync_failure_keeps_complete_artifact(
    tmp_path, monkeypatch
):
    # Regression (stranded metadata-only pair): atomic_write_json publishes the
    # metadata with os.replace and only then fsyncs the directory.  If that
    # directory fsync raises (EINVAL on filesystems without directory-fsync
    # support), the metadata IS already on disk.  The old cleanup then unlinked
    # the archive anyway, leaving a metadata-only pair that every retry strands
    # behind FileExistsError.  Once the metadata has landed the artifact is
    # complete and must be left intact.
    env = _make_vector_env()
    model = StochasticWarmstart(action_dimension=4)
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
    artifact = tmp_path / "matched_prefill.npz"
    metadata_path = artifact.with_suffix(artifact.suffix + ".json")
    real_save_npz = p6_runtime._atomic_save_npz

    def publish_then_fsync_fails(metadata_path_arg, payload):
        # os.replace lands the metadata, then the durability fsync fails.
        metadata_path_arg.parent.mkdir(parents=True, exist_ok=True)
        temporary = metadata_path_arg.with_name(
            f"{metadata_path_arg.name}.{os.getpid()}.tmp"
        )
        temporary.write_text("{}")
        os.replace(temporary, metadata_path_arg)
        raise OSError("EINVAL: directory fsync not supported")

    monkeypatch.setattr(p6_runtime, "atomic_write_json", publish_then_fsync_fails)
    with pytest.raises(OSError, match="EINVAL"):
        save_prefill_artifact(artifact, arrays, metadata)
    # The complete artifact (archive + published metadata) survives intact.
    assert artifact.is_file()
    assert metadata_path.is_file()
    assert real_save_npz is not None


def test_canonical_module_state_hash_is_deterministic_and_order_independent():
    # Two structurally identical modules whose parameters are registered in
    # different insertion orders (same names, same values) must hash equally:
    # the digest iterates sorted state-dict keys so dict insertion order is
    # irrelevant.
    def make(reverse_registration):
        module = torch.nn.Module()
        # Fixed values so every constructed module holds identical weights.
        weight = torch.nn.Parameter(
            torch.arange(12, dtype=torch.float32).reshape(3, 4)
        )
        bias = torch.nn.Parameter(torch.arange(4, dtype=torch.float32))
        if reverse_registration:
            module.bias = bias
            module.weight = weight
        else:
            module.weight = weight
            module.bias = bias
        return module

    first = canonical_module_state_hash(make(reverse_registration=False))
    second = canonical_module_state_hash(make(reverse_registration=False))
    assert first == second
    assert canonical_module_state_hash(make(reverse_registration=True)) == first
    # Perturbing any weight changes the digest.
    module = make(reverse_registration=False)
    module.weight.data.add_(1.0)
    assert canonical_module_state_hash(module) != first


def test_canonical_module_state_hash_changes_on_weight_and_shape():
    module = torch.nn.Linear(2, 2)
    original = canonical_module_state_hash(module)
    module.weight.data.add_(0.5)
    assert canonical_module_state_hash(module) != original


def test_assert_matched_fresh_init_state_hashes_gate(tmp_path):
    base = {
        "actor": "a" * 64,
        "qa_base": "b" * 64,
        "qa_base_target": "c" * 64,
        "qw_base": "d" * 64,
    }
    hierarchy_manifest = {
        "zero_residual_ddim_parity": {"fresh_init_state_hashes": base}
    }

    # Matching manifests pass.
    assert_matched_fresh_init_state_hashes(
        {"zero_residual_ddim_parity": {"fresh_init_state_hashes": dict(base)}},
        hierarchy_manifest,
    )

    # Missing hashes on either side is a hard error, not a silent pass.
    with pytest.raises(ValueError, match="missing from one or both manifests"):
        assert_matched_fresh_init_state_hashes({}, hierarchy_manifest)
    with pytest.raises(ValueError, match="missing from one or both manifests"):
        assert_matched_fresh_init_state_hashes(
            {
                "zero_residual_ddim_parity": {
                    "fresh_init_state_hashes": None
                }
            },
            hierarchy_manifest,
        )
    # An empty hash mapping records no modules at all; reported as a key-set
    # difference against the hierarchy's four canonical modules.
    with pytest.raises(ValueError, match="keys differ"):
        assert_matched_fresh_init_state_hashes(
            {"zero_residual_ddim_parity": {"fresh_init_state_hashes": {}}},
            hierarchy_manifest,
        )

    # A key-set difference is reported.
    control_extra = dict(base)
    control_extra["critic_target"] = "e" * 64
    with pytest.raises(ValueError, match="keys differ"):
        assert_matched_fresh_init_state_hashes(
            {
                "zero_residual_ddim_parity": {
                    "fresh_init_state_hashes": control_extra
                }
            },
            hierarchy_manifest,
        )

    # A diverged module is reported by name.
    control_diverged = dict(base)
    control_diverged["qa_base"] = "x" * 64
    with pytest.raises(ValueError, match="diverged on qa_base"):
        assert_matched_fresh_init_state_hashes(
            {
                "zero_residual_ddim_parity": {
                    "fresh_init_state_hashes": control_diverged
                }
            },
            hierarchy_manifest,
        )
