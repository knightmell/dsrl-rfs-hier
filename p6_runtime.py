"""P6 matched-prefill, replay-integrity, RNG, and counter utilities.

This module deliberately uses the project's standard SB3 ReplayBuffer.  The
portable prefill artifact is only a compact dataset used to populate two
otherwise independent standard buffers with bitwise-identical transitions.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np
import torch

from p6_preflight import sha256_file


PREFILL_ARRAY_NAMES = (
    "observations",
    "next_observations",
    "actions",
    "rewards",
    "dones",
    "timeouts",
)
PREFILL_COUNTER_ARRAY_NAMES = (
    "actual_primitive_steps",
    "early_termination_within_chunk",
)


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as output:
        json.dump(dict(payload), output, indent=2, sort_keys=True)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary_path, path)


def _canonical_json(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(payload),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _update_array_hash(
    digest: "hashlib._Hash",
    name: str,
    value: np.ndarray,
) -> str:
    array = np.ascontiguousarray(value)
    component = hashlib.sha256()
    for hasher in (digest, component):
        hasher.update(name.encode("utf-8"))
        hasher.update(str(array.dtype).encode("ascii"))
        hasher.update(_canonical_json({"shape": list(array.shape)}))
        hasher.update(array.tobytes(order="C"))
    return component.hexdigest()


def hash_semantic_arrays(
    arrays: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
    *,
    array_names: tuple[str, ...] = PREFILL_ARRAY_NAMES,
) -> dict[str, Any]:
    digest = hashlib.sha256()
    per_array: dict[str, str] = {}
    for name in array_names:
        if name not in arrays:
            raise KeyError(f"Missing semantic array {name!r}")
        per_array[name] = _update_array_hash(digest, name, arrays[name])
    digest.update(_canonical_json(metadata))
    return {
        "semantic_hash": digest.hexdigest(),
        "per_array_hashes": per_array,
    }


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    else:
        state["torch_cuda"] = None
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    cuda_state = state.get("torch_cuda")
    if cuda_state is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("Saved CUDA RNG state cannot be restored without CUDA")
        torch.cuda.set_rng_state_all(cuda_state)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@contextmanager
def isolated_rng(seed: int | None = None) -> Iterator[None]:
    state = capture_rng_state()
    try:
        if seed is not None:
            seed_all(seed)
        yield
    finally:
        restore_rng_state(state)


@dataclass
class PrimitiveCounters:
    chunk_transitions: int = 0
    nominal_primitive_steps: int = 0
    actual_primitive_env_steps: int = 0
    skipped_primitive_steps_due_to_termination: int = 0
    early_termination_chunks: int = 0
    terminal_chunks: int = 0
    timeout_chunks: int = 0

    def update(
        self,
        infos: list[dict[str, Any]],
        dones: np.ndarray,
        *,
        action_chunk: int,
    ) -> None:
        if len(infos) != len(dones):
            raise ValueError("infos and dones must have the same vector-env length")
        for info, done in zip(infos, dones):
            nominal = int(info.get("nominal_primitive_steps", -1))
            actual = int(info.get("actual_primitive_steps", -1))
            if nominal != action_chunk:
                raise ValueError(
                    "Missing or inconsistent nominal primitive counter: "
                    f"expected {action_chunk}, got {nominal}"
                )
            if actual < 1 or actual > nominal:
                raise ValueError(
                    f"Invalid actual primitive count {actual} for nominal {nominal}"
                )
            early = bool(info.get("early_termination_within_chunk", False))
            if early != (bool(done) and actual < nominal):
                raise ValueError("Early-termination record does not explain counter gap")

            self.chunk_transitions += 1
            self.nominal_primitive_steps += nominal
            self.actual_primitive_env_steps += actual
            self.skipped_primitive_steps_due_to_termination += nominal - actual
            self.early_termination_chunks += int(early)
            self.terminal_chunks += int(bool(done) and not info.get("TimeLimit.truncated", False))
            self.timeout_chunks += int(bool(info.get("TimeLimit.truncated", False)))
        self.validate(action_chunk=action_chunk)

    def validate(self, *, action_chunk: int) -> None:
        expected_nominal = self.chunk_transitions * action_chunk
        if self.nominal_primitive_steps != expected_nominal:
            raise ValueError(
                "Nominal primitive counter mismatch: "
                f"{self.nominal_primitive_steps} != {expected_nominal}"
            )
        if self.actual_primitive_env_steps > self.nominal_primitive_steps:
            raise ValueError("Actual primitive steps exceed nominal primitive steps")
        if (
            self.nominal_primitive_steps - self.actual_primitive_env_steps
            != self.skipped_primitive_steps_due_to_termination
        ):
            raise ValueError("Termination records do not explain primitive-step gap")

    def to_dict(self) -> dict[str, int]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PrimitiveCounters":
        return cls(**{field: int(payload[field]) for field in cls.__dataclass_fields__})


def terminal_corrected_next_observations(
    next_observations: np.ndarray,
    dones: np.ndarray,
    infos: list[dict[str, Any]],
) -> np.ndarray:
    corrected = np.array(next_observations, copy=True)
    for index, done in enumerate(dones):
        if not done:
            continue
        terminal_observation = infos[index].get("terminal_observation")
        if terminal_observation is None:
            raise ValueError(
                f"Done transition {index} is missing terminal_observation"
            )
        corrected[index] = np.asarray(
            terminal_observation,
            dtype=corrected.dtype,
        )
    return corrected


def _prefill_metadata_for_hash(metadata: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: metadata[key]
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
    }


def _validate_prefill_arrays(
    arrays: Mapping[str, np.ndarray],
    *,
    vector_steps: int,
    n_envs: int,
) -> None:
    prefix = (vector_steps, n_envs)
    for name in PREFILL_ARRAY_NAMES + PREFILL_COUNTER_ARRAY_NAMES:
        if name not in arrays:
            raise ValueError(f"Prefill artifact is missing {name!r}")
        if arrays[name].shape[:2] != prefix:
            raise ValueError(
                f"Prefill {name} prefix shape {arrays[name].shape[:2]} != {prefix}"
            )
    if not np.isfinite(arrays["observations"]).all():
        raise ValueError("Prefill observations contain non-finite values")
    if not np.isfinite(arrays["next_observations"]).all():
        raise ValueError("Prefill next_observations contain non-finite values")
    if not np.isfinite(arrays["actions"]).all():
        raise ValueError("Prefill actions contain non-finite values")
    if not np.isfinite(arrays["rewards"]).all():
        raise ValueError("Prefill rewards contain non-finite values")


def _atomic_save_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("wb") as output:
        np.savez_compressed(output, **arrays)
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary_path, path)


def collect_matched_prefill(
    *,
    warmstart_model: Any,
    env: Any,
    vector_steps: int,
    environment_seed: int,
    policy_seed: int,
    action_chunk: int,
    termination_semantics: str,
    provenance: Mapping[str, str],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if vector_steps <= 0:
        raise ValueError("P6 matched prefill requires vector_steps > 0")
    n_envs = int(env.num_envs)
    counters = PrimitiveCounters()
    observations: list[np.ndarray] = []
    next_observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    rewards: list[np.ndarray] = []
    dones_values: list[np.ndarray] = []
    timeouts: list[np.ndarray] = []
    actual_steps: list[np.ndarray] = []
    early_terminations: list[np.ndarray] = []

    with isolated_rng(policy_seed):
        env.seed(environment_seed)
        observation = env.reset()
        for _ in range(vector_steps):
            action_exec, _ = warmstart_model.predict_diffused(
                observation,
                deterministic=False,
            )
            action_exec = np.asarray(action_exec, dtype=np.float32)
            new_observation, reward, done, infos = env.step(action_exec)
            corrected_next = terminal_corrected_next_observations(
                new_observation,
                done,
                infos,
            )
            counters.update(infos, done, action_chunk=action_chunk)

            observations.append(np.asarray(observation).copy())
            next_observations.append(corrected_next)
            actions.append(action_exec.copy())
            rewards.append(np.asarray(reward, dtype=np.float32).copy())
            dones_values.append(np.asarray(done, dtype=np.bool_).copy())
            timeouts.append(
                np.asarray(
                    [
                        bool(info.get("TimeLimit.truncated", False))
                        for info in infos
                    ],
                    dtype=np.bool_,
                )
            )
            actual_steps.append(
                np.asarray(
                    [int(info["actual_primitive_steps"]) for info in infos],
                    dtype=np.int16,
                )
            )
            early_terminations.append(
                np.asarray(
                    [
                        bool(info["early_termination_within_chunk"])
                        for info in infos
                    ],
                    dtype=np.bool_,
                )
            )
            observation = new_observation

    arrays = {
        "observations": np.asarray(observations),
        "next_observations": np.asarray(next_observations),
        "actions": np.asarray(actions, dtype=np.float32),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "dones": np.asarray(dones_values, dtype=np.bool_),
        "timeouts": np.asarray(timeouts, dtype=np.bool_),
        "actual_primitive_steps": np.asarray(actual_steps, dtype=np.int16),
        "early_termination_within_chunk": np.asarray(
            early_terminations,
            dtype=np.bool_,
        ),
    }
    _validate_prefill_arrays(arrays, vector_steps=vector_steps, n_envs=n_envs)
    metadata: dict[str, Any] = {
        "format_version": 1,
        "vector_steps": vector_steps,
        "chunk_transitions": vector_steps * n_envs,
        "n_envs": n_envs,
        "action_chunk": action_chunk,
        "environment_seed": environment_seed,
        "policy_seed": policy_seed,
        "termination_semantics": termination_semantics,
        "init_checkpoint_sha256": provenance["init_checkpoint_sha256"],
        "frozen_ddim_sha256": provenance["frozen_ddim_sha256"],
        "normalization_sha256": provenance["normalization_sha256"],
        "primitive_counters": counters.to_dict(),
    }
    hashes = hash_semantic_arrays(
        arrays,
        _prefill_metadata_for_hash(metadata),
    )
    metadata.update(hashes)
    return arrays, metadata


def save_prefill_artifact(
    path: Path,
    arrays: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    metadata_path = path.with_suffix(path.suffix + ".json")
    if path.exists() or metadata_path.exists():
        raise FileExistsError(f"Refusing to overwrite prefill artifact {path}")
    _atomic_save_npz(path, arrays)
    completed_metadata = dict(metadata)
    completed_metadata["archive_sha256"] = sha256_file(path)
    atomic_write_json(metadata_path, completed_metadata)
    return completed_metadata


def load_prefill_artifact(
    path: Path,
    *,
    expected: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    metadata_path = path.with_suffix(path.suffix + ".json")
    if not path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(
            f"Prefill artifact requires both {path} and {metadata_path}"
        )
    with metadata_path.open("r", encoding="utf-8") as input_file:
        metadata = json.load(input_file)
    if sha256_file(path) != metadata.get("archive_sha256"):
        raise ValueError("Prefill archive SHA-256 mismatch")
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(
                f"Prefill metadata mismatch for {key}: "
                f"expected {value!r}, got {metadata.get(key)!r}"
            )
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    _validate_prefill_arrays(
        arrays,
        vector_steps=int(metadata["vector_steps"]),
        n_envs=int(metadata["n_envs"]),
    )
    actual_hashes = hash_semantic_arrays(
        arrays,
        _prefill_metadata_for_hash(metadata),
    )
    if actual_hashes != {
        "semantic_hash": metadata.get("semantic_hash"),
        "per_array_hashes": metadata.get("per_array_hashes"),
    }:
        raise ValueError("Prefill semantic hash mismatch")
    return arrays, metadata


def collect_or_load_matched_prefill(
    *,
    artifact_path: Path,
    warmstart_model: Any,
    env: Any,
    vector_steps: int,
    environment_seed: int,
    policy_seed: int,
    action_chunk: int,
    termination_semantics: str,
    provenance: Mapping[str, str],
) -> tuple[dict[str, np.ndarray], dict[str, Any], bool]:
    expected = {
        "format_version": 1,
        "vector_steps": vector_steps,
        "chunk_transitions": vector_steps * int(env.num_envs),
        "n_envs": int(env.num_envs),
        "action_chunk": action_chunk,
        "environment_seed": environment_seed,
        "policy_seed": policy_seed,
        "termination_semantics": termination_semantics,
        "init_checkpoint_sha256": provenance["init_checkpoint_sha256"],
        "frozen_ddim_sha256": provenance["frozen_ddim_sha256"],
        "normalization_sha256": provenance["normalization_sha256"],
    }
    if artifact_path.exists() or artifact_path.with_suffix(
        artifact_path.suffix + ".json"
    ).exists():
        arrays, metadata = load_prefill_artifact(
            artifact_path,
            expected=expected,
        )
        return arrays, metadata, False

    arrays, metadata = collect_matched_prefill(
        warmstart_model=warmstart_model,
        env=env,
        vector_steps=vector_steps,
        environment_seed=environment_seed,
        policy_seed=policy_seed,
        action_chunk=action_chunk,
        termination_semantics=termination_semantics,
        provenance=provenance,
    )
    metadata = save_prefill_artifact(artifact_path, arrays, metadata)
    return arrays, metadata, True


def populate_replay_buffer(
    replay_buffer: Any,
    arrays: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    if replay_buffer.pos != 0 or replay_buffer.full:
        raise RuntimeError("Matched prefill requires an empty ReplayBuffer")
    vector_steps, n_envs = arrays["rewards"].shape
    if replay_buffer.n_envs != n_envs:
        raise ValueError(
            f"Replay n_envs mismatch: {replay_buffer.n_envs} != {n_envs}"
        )
    if replay_buffer.buffer_size <= vector_steps:
        raise ValueError(
            "ReplayBuffer must have capacity beyond the prefill prefix to leave "
            "room for online training"
        )
    for step in range(vector_steps):
        infos = [
            {"TimeLimit.truncated": bool(arrays["timeouts"][step, index])}
            for index in range(n_envs)
        ]
        replay_buffer.add(
            obs=arrays["observations"][step],
            next_obs=arrays["next_observations"][step],
            action=arrays["actions"][step],
            reward=arrays["rewards"][step],
            done=arrays["dones"][step],
            infos=infos,
        )
    replay_buffer.final_offline_step()
    return hash_replay_prefix(replay_buffer)


def hash_replay_prefix(replay_buffer: Any) -> dict[str, Any]:
    if replay_buffer.full:
        raise ValueError("P6 semantic hashing currently requires a non-wrapped buffer")
    vector_steps = int(replay_buffer.pos)
    arrays = {
        "observations": replay_buffer.observations[:vector_steps],
        "next_observations": replay_buffer.next_observations[:vector_steps],
        "actions": replay_buffer.actions[:vector_steps],
        "rewards": replay_buffer.rewards[:vector_steps],
        "dones": replay_buffer.dones[:vector_steps].astype(np.bool_),
        "timeouts": replay_buffer.timeouts[:vector_steps].astype(np.bool_),
    }
    metadata = {
        "vector_steps": vector_steps,
        "chunk_transitions": vector_steps * int(replay_buffer.n_envs),
        "n_envs": int(replay_buffer.n_envs),
        "offline_steps": int(replay_buffer.offline_steps),
    }
    return {
        **metadata,
        **hash_semantic_arrays(arrays, metadata),
    }
