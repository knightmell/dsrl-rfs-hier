"""P6 matched-prefill, replay-integrity, RNG, and counter utilities.

The hierarchy keeps exact behavior tags in its private replay subclass, while
the matched control receives the standard six-field projection of the same
immutable prefill artifact.  The common SB3 ReplayBuffer remains unchanged.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import random
import warnings
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

import numpy as np
import torch

from p6_preflight import sha256_file
from stable_baselines3.dsrl.hierarchical_replay_buffer import (
    ACTION_METADATA_FIELDS,
    BranchMode,
    HierarchyTaggedReplayBuffer,
    NoiseSampleSource,
    SCALAR_METADATA_DTYPES,
    SCHEMA_VERSION as TAGGED_PREFILL_FORMAT_VERSION,
    TerminationSemantics,
    TransitionOrigin,
)


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
PREFILL_SEMANTIC_ARRAY_NAMES = (
    PREFILL_ARRAY_NAMES + PREFILL_COUNTER_ARRAY_NAMES
)
PREFILL_FORMAT_VERSION = 2
# Mirrors p6_preflight PREFILL_SOURCE / FRESH_PREFILL_SOURCE without importing
# p6_preflight into this module's constant table (kept here for validation).
PREFILL_SOURCE_WARMSTART = "warmstart_dsrl"
PREFILL_SOURCE_FRESH = "fresh_frozen_ddim"

TAGGED_PREFILL_STANDARD_PROJECTION_NAMES = (
    "observations",
    "next_observations",
    "action_exec",
    "rewards",
    "dones",
    "timeouts",
)
TAGGED_PREFILL_ARRAY_NAMES = (
    "observations",
    "next_observations",
    "rewards",
    "dones",
    "timeouts",
    "terminated",
    "truncated",
    *ACTION_METADATA_FIELDS,
    *SCALAR_METADATA_DTYPES.keys(),
)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _process_unique_temporary_path(path: Path) -> Path:
    """A sibling temp path unique to this writer process.

    A fixed ``path.tmp`` sibling is unsafe when two processes can write the
    same target concurrently: the matched A/B/control design shares one
    prefill artifact path for the same seed, so a control run and its
    hierarchy run may both persist the identical artifact.  With a shared
    ``.tmp`` name they would (a) truncate each other's temp, (b) have one
    ``os.replace`` fail after the peer already renamed, and (c) let orphan
    recovery delete the peer's half-written file.  A process-unique suffix
    keeps each writer's temp private; ``os.replace`` then makes the last
    complete writer win, which is safe because the content is deterministic
    for a given seed.
    """
    return path.with_name(f"{path.name}.{os.getpid()}.tmp")


@contextmanager
def _prefill_artifact_lock(artifact_path: Path) -> Iterator[Path]:
    """Serialize every read/repair/write of one shared prefill bundle.

    The tagged prefill is a two-file transaction (``.npz`` plus ``.json``).
    Atomic rename protects each file individually, but cannot protect the gap
    between the two renames.  A stable advisory lock file closes that gap:
    only the lock owner may classify a half-pair as an orphan, regenerate it,
    or load it.  ``flock`` is released by the kernel if the owner exits, so the
    lock file itself intentionally remains on disk and cannot become stale.
    """

    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = artifact_path.with_suffix(artifact_path.suffix + ".lock")
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield lock_path
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = _process_unique_temporary_path(path)
    with temporary_path.open("w", encoding="utf-8") as output:
        json.dump(dict(payload), output, indent=2, sort_keys=True)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary_path, path)
    _fsync_directory(path.parent)


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


def canonical_module_state_hash(module: torch.nn.Module) -> str:
    """Canonical SHA-256 over a module's state dict.

    Sorted keys and explicit dtype/shape/bytes make the digest independent of
    insertion order and of the host's torch/numpy memory layout.  Used to
    prove at artifact level that the fresh control and the fresh hierarchy
    start from bit-identical base modules (noise actor / QA_base / QW_base),
    which the manifest can only claim if it records the hashes themselves.
    """
    digest = hashlib.sha256()
    for key in sorted(module.state_dict()):
        tensor = module.state_dict()[key].detach().cpu()
        digest.update(key.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(_canonical_json({"shape": list(tensor.shape)}))
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def assert_matched_fresh_init_state_hashes(
    control_manifest: Mapping[str, Any],
    hierarchy_manifest: Mapping[str, Any],
) -> None:
    """Prove the fresh control and fresh hierarchy started bit-identical.

    Both runs record ``zero_residual_ddim_parity.fresh_init_state_hashes``
    under the same canonical keys (actor, qa_base, qa_base_target, qw_base).
    The gate comparison is only meaningful if the two runs began from the same
    base-branch weights, so this raises unless every hash agrees.
    """
    # `or {}` (not just `.get(..., {})`): the manifest may carry the key with
    # an explicit JSON null (e.g. legacy runs omit the parity dict as None),
    # in which case `.get` returns None and the next `.get` would raise
    # AttributeError.  A null parity is the same as an absent one here.
    control_parity = control_manifest.get("zero_residual_ddim_parity") or {}
    hierarchy_parity = hierarchy_manifest.get("zero_residual_ddim_parity") or {}
    control_hashes = control_parity.get("fresh_init_state_hashes")
    hierarchy_hashes = hierarchy_parity.get("fresh_init_state_hashes")
    if not isinstance(control_hashes, Mapping) or not isinstance(
        hierarchy_hashes, Mapping
    ):
        raise ValueError(
            "Matched fresh-init state hashes missing from one or both manifests"
        )
    if control_hashes.keys() != hierarchy_hashes.keys():
        raise ValueError(
            "Matched fresh-init state-hash keys differ: "
            f"{sorted(control_hashes)} != {sorted(hierarchy_hashes)}"
        )
    for name, control_hash in control_hashes.items():
        hierarchy_hash = hierarchy_hashes[name]
        if control_hash != hierarchy_hash:
            raise ValueError(
                f"Matched fresh init diverged on {name}: "
                f"control {control_hash} != hierarchy {hierarchy_hash}"
            )


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


def reset_vec_env_with_explicit_seeds(
    env: Any, seeds: Sequence[int]
) -> np.ndarray:
    """Reset every VecEnv slot with the exact audited seed list.

    SB3's public ``VecEnv.seed(base)`` only supports ``base + env_id``.  P6
    reset-boundary resume deliberately uses independently SHA-derived seeds, so
    this local adapter fills the existing one-shot seed queue and immediately
    consumes it through the normal ``VecEnv.reset()`` implementation.
    """

    values = [int(seed) for seed in seeds]
    if len(values) != int(env.num_envs):
        raise ValueError("Explicit reset seed count must equal env.num_envs")
    if any(seed < 0 or seed > np.iinfo(np.uint32).max for seed in values):
        raise ValueError("Explicit reset seeds must be uint32-compatible")
    if not hasattr(env, "_seeds"):
        raise TypeError("VecEnv does not expose the audited one-shot seed queue")
    env._seeds = values.copy()
    observation = np.asarray(env.reset())
    if list(getattr(env, "_seeds", [])) != [None] * int(env.num_envs):
        raise RuntimeError("VecEnv did not consume and clear explicit reset seeds")
    return observation


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
            "primitive_counters",
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
    if arrays["next_observations"].shape != arrays["observations"].shape:
        raise ValueError(
            "Prefill next_observations shape must equal observations shape"
        )
    for name in (
        "rewards",
        "dones",
        "timeouts",
        "actual_primitive_steps",
        "early_termination_within_chunk",
    ):
        if arrays[name].shape != prefix:
            raise ValueError(
                f"Prefill {name} shape {arrays[name].shape} != {prefix}"
            )
    if not np.isfinite(arrays["observations"]).all():
        raise ValueError("Prefill observations contain non-finite values")
    if not np.isfinite(arrays["next_observations"]).all():
        raise ValueError("Prefill next_observations contain non-finite values")
    if not np.isfinite(arrays["actions"]).all():
        raise ValueError("Prefill actions contain non-finite values")
    if not np.isfinite(arrays["rewards"]).all():
        raise ValueError("Prefill rewards contain non-finite values")


def validate_prefill_semantics(
    arrays: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
) -> PrimitiveCounters:
    """Validate and recompute every transition/counter invariant in a prefill."""

    vector_steps = int(metadata["vector_steps"])
    n_envs = int(metadata["n_envs"])
    action_chunk = int(metadata["action_chunk"])
    if vector_steps <= 0 or n_envs <= 0 or action_chunk <= 0:
        raise ValueError("Prefill dimensions and action_chunk must be positive")
    expected_chunk_transitions = vector_steps * n_envs
    if int(metadata["chunk_transitions"]) != expected_chunk_transitions:
        raise ValueError("Prefill chunk_transitions disagrees with array shape")

    _validate_prefill_arrays(
        arrays,
        vector_steps=vector_steps,
        n_envs=n_envs,
    )
    dones = np.asarray(arrays["dones"], dtype=np.bool_)
    timeouts = np.asarray(arrays["timeouts"], dtype=np.bool_)
    actual = np.asarray(arrays["actual_primitive_steps"], dtype=np.int64)
    early = np.asarray(
        arrays["early_termination_within_chunk"],
        dtype=np.bool_,
    )
    if np.any(timeouts & ~dones):
        raise ValueError("Prefill contains timeout_without_done")
    if np.any((actual < 1) | (actual > action_chunk)):
        raise ValueError("Prefill actual_primitive_steps are outside legal bounds")
    expected_early = dones & (actual < action_chunk)
    if not np.array_equal(early, expected_early):
        raise ValueError("Prefill early_termination semantics are inconsistent")

    nominal = expected_chunk_transitions * action_chunk
    actual_total = int(actual.sum())
    counters = PrimitiveCounters(
        chunk_transitions=expected_chunk_transitions,
        nominal_primitive_steps=nominal,
        actual_primitive_env_steps=actual_total,
        skipped_primitive_steps_due_to_termination=nominal - actual_total,
        early_termination_chunks=int(early.sum()),
        terminal_chunks=int((dones & ~timeouts).sum()),
        timeout_chunks=int(timeouts.sum()),
    )
    counters.validate(action_chunk=action_chunk)
    if dict(metadata["primitive_counters"]) != counters.to_dict():
        raise ValueError("Prefill primitive_counters do not match transition arrays")
    return counters


def _atomic_save_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = _process_unique_temporary_path(path)
    with temporary_path.open("wb") as output:
        np.savez_compressed(output, **arrays)
        output.flush()
        os.fsync(output.fileno())
    # os.replace preserves the temp's inode at the destination, so capturing it
    # here (before any peer can replace the destination) lets a later cleanup
    # tell "the file I wrote" apart from "a peer's completed file".
    inode = temporary_path.stat().st_ino
    os.replace(temporary_path, path)
    _fsync_directory(path.parent)
    return inode


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
    metadata: dict[str, Any] = {
        "format_version": PREFILL_FORMAT_VERSION,
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
    validate_prefill_semantics(arrays, metadata)
    hashes = hash_semantic_arrays(
        arrays,
        _prefill_metadata_for_hash(metadata),
        array_names=PREFILL_SEMANTIC_ARRAY_NAMES,
    )
    metadata.update(hashes)
    return arrays, metadata


def save_prefill_artifact(
    path: Path,
    arrays: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
    *,
    validator: Callable[..., PrimitiveCounters] = validate_prefill_semantics,
) -> dict[str, Any]:
    """Atomically persist a prefill artifact.

    The default ``validator`` enforces the standard six-field format (used by
    the matched control).  The tagged hierarchy artifact is saved through the
    same atomic two-rename write, but validated with
    :func:`validate_tagged_prefill_semantics` because its arrays are keyed by
    ``TAGGED_PREFILL_ARRAY_NAMES`` (``action_exec``, not ``actions``).
    """
    metadata_path = path.with_suffix(path.suffix + ".json")
    if path.exists() or metadata_path.exists():
        raise FileExistsError(f"Refusing to overwrite prefill artifact {path}")
    validator(arrays, metadata)
    completed_metadata = dict(metadata)
    archive_inode: int | None = None
    try:
        archive_inode = _atomic_save_npz(path, arrays)
        completed_metadata["archive_sha256"] = sha256_file(path)
        atomic_write_json(metadata_path, completed_metadata)
    except BaseException:
        # The archive and its metadata are written in two separate renames.
        # Cleanup is warranted only when the archive is OURS (inode guard: a
        # peer may rename its identical, deterministic archive onto this path
        # between our archive rename and our metadata write) AND the metadata
        # never landed.  Two failure cases must not trigger an unlink:
        #  - `archive_inode` is unbound when _atomic_save_npz raised before
        #    its rename: nothing of ours is at `path`, so touching it could
        #    only ever delete a peer's file.
        #  - the metadata was already published (its os.replace succeeded; a
        #    later directory fsync raised): the artifact is then complete, so
        #    deleting the archive would leave a metadata-only pair that every
        #    retry strands behind a FileExistsError.
        if archive_inode is not None and not metadata_path.exists():
            try:
                if path.stat().st_ino == archive_inode:
                    path.unlink()
            except OSError:
                pass
        raise
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
    validate_prefill_semantics(arrays, metadata)
    actual_hashes = hash_semantic_arrays(
        arrays,
        _prefill_metadata_for_hash(metadata),
        array_names=PREFILL_SEMANTIC_ARRAY_NAMES,
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
        "format_version": PREFILL_FORMAT_VERSION,
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


def tagged_prefill_standard_projection(
    arrays: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Exact standard ReplayBuffer view of one immutable tagged artifact."""

    return {
        "observations": np.asarray(arrays["observations"]),
        "next_observations": np.asarray(arrays["next_observations"]),
        "actions": np.asarray(arrays["action_exec"]),
        "rewards": np.asarray(arrays["rewards"]),
        "dones": np.asarray(arrays["dones"]),
        "timeouts": np.asarray(arrays["timeouts"]),
    }


def _tagged_prefill_hash_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
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
        "primitive_counters",
    )
    return {key: metadata[key] for key in keys}


def validate_tagged_prefill_semantics(
    arrays: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
    *,
    prefill_source: str = PREFILL_SOURCE_WARMSTART,
) -> PrimitiveCounters:
    if metadata.get("format_version") != TAGGED_PREFILL_FORMAT_VERSION:
        raise ValueError(
            "Hierarchy requires hierarchy_tagged_replay_v1; legacy prefill v2 "
            "cannot be used because behavior metadata is missing"
        )
    vector_steps = int(metadata["vector_steps"])
    n_envs = int(metadata["n_envs"])
    action_chunk = int(metadata["action_chunk"])
    prefix = (vector_steps, n_envs)
    missing = set(TAGGED_PREFILL_ARRAY_NAMES) - set(arrays)
    if missing:
        raise ValueError(f"Tagged prefill is missing arrays: {sorted(missing)}")
    for name in TAGGED_PREFILL_ARRAY_NAMES:
        if np.asarray(arrays[name]).shape[:2] != prefix:
            raise ValueError(f"Tagged prefill {name} has an invalid prefix shape")
    for name in (
        "observations",
        "next_observations",
        "rewards",
        "noise_scaled",
        "noise_log_prob",
        "action_base",
        "residual_pre_tanh",
        "residual_unit",
        "action_residual_delta",
        "action_exec",
        "beta",
    ):
        if not np.isfinite(np.asarray(arrays[name])).all():
            raise ValueError(f"Tagged prefill {name} contains non-finite values")
    branches = np.asarray(arrays["branch_mode"])
    if np.any(branches != int(BranchMode.BASE)):
        raise ValueError("Matched prefill must contain BASE transitions only")
    for name in (
        "residual_pre_tanh",
        "residual_unit",
        "action_residual_delta",
    ):
        if np.any(np.asarray(arrays[name]) != 0):
            raise ValueError(f"Matched BASE prefill requires exact-zero {name}")
    if np.any(np.asarray(arrays["residual_applied"], dtype=np.bool_)):
        raise ValueError("Matched BASE prefill cannot apply residual")
    if np.any(np.asarray(arrays["residual_policy_version"]) != -1):
        raise ValueError("Matched BASE prefill requires residual policy version -1")
    if prefill_source == PREFILL_SOURCE_WARMSTART:
        if np.any(np.asarray(arrays["noise_policy_version"]) != 0):
            raise ValueError(
                "Matched prefill requires immutable noise policy version 0"
            )
        if not np.all(
            np.asarray(arrays["noise_log_prob_valid"], dtype=np.bool_)
        ):
            raise ValueError(
                "Warm-start matched prefill requires valid actor log-probs"
            )
        if np.any(
            np.asarray(arrays["noise_sample_source"])
            != int(NoiseSampleSource.CURRENT_ACTOR)
        ):
            raise ValueError(
                "Matched prefill must be sampled by the warm-start actor"
            )
    elif prefill_source == PREFILL_SOURCE_FRESH:
        if np.any(np.asarray(arrays["noise_policy_version"]) != -1):
            raise ValueError(
                "Fresh Frozen-DDIM prefill requires noise policy version -1"
            )
        if np.any(
            np.asarray(arrays["noise_log_prob_valid"], dtype=np.bool_)
        ):
            raise ValueError(
                "Fresh Frozen-DDIM prefill requires invalid actor log-probs"
            )
        if np.any(
            np.asarray(arrays["noise_sample_source"])
            != int(NoiseSampleSource.GAUSSIAN_PRIOR)
        ):
            raise ValueError(
                "Fresh Frozen-DDIM prefill must use the Gaussian decoder prior"
            )
    else:
        raise ValueError(f"Unknown tagged prefill source {prefill_source!r}")
    if np.any(
        np.asarray(arrays["transition_origin"]) != int(TransitionOrigin.PREFILL)
    ):
        raise ValueError("Matched prefill must use PREFILL transition origin")
    if np.any(np.asarray(arrays["beta"]) != 0):
        raise ValueError("Matched BASE prefill requires beta=0")
    if not np.array_equal(arrays["action_base"], arrays["action_exec"]):
        raise ValueError("Matched BASE prefill action_exec must equal action_base")
    if not np.array_equal(arrays["truncated"], arrays["timeouts"]):
        raise ValueError("Tagged truncated and timeout arrays differ")
    expected_terminated = np.asarray(arrays["dones"], dtype=np.bool_) & ~np.asarray(
        arrays["timeouts"], dtype=np.bool_
    )
    if not np.array_equal(arrays["terminated"], expected_terminated):
        raise ValueError("Tagged terminated truth is inconsistent")
    standard = tagged_prefill_standard_projection(arrays)
    compatibility = {
        **standard,
        "actual_primitive_steps": np.asarray(arrays["actual_primitive_steps"]),
        "early_termination_within_chunk": np.asarray(arrays["dones"], dtype=np.bool_)
        & (np.asarray(arrays["actual_primitive_steps"]) < action_chunk),
    }
    compatibility_metadata = dict(metadata)
    compatibility_metadata["format_version"] = PREFILL_FORMAT_VERSION
    return validate_prefill_semantics(compatibility, compatibility_metadata)


def _legacy_noise_and_base(
    warmstart_model: Any, observation: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    observation_tensor = torch.as_tensor(
        observation,
        device=warmstart_model.device,
        dtype=torch.float32,
    )
    with torch.no_grad():
        noise_scaled, log_prob = warmstart_model.actor.action_log_prob(
            observation_tensor
        )
        decoder_numpy = warmstart_model.policy.unscale_action(
            noise_scaled.detach().cpu().numpy()
        )
        decoder = torch.as_tensor(
            decoder_numpy,
            device=warmstart_model.device,
            dtype=torch.float32,
        ).reshape(
            -1,
            warmstart_model.diffusion_act_chunk,
            warmstart_model.diffusion_act_dim,
        )
        action_base = warmstart_model.diffusion_policy(
            observation_tensor, decoder, return_numpy=False
        ).reshape(observation_tensor.shape[0], -1)
    return (
        noise_scaled.detach().cpu().numpy().astype(np.float32),
        log_prob.detach().cpu().numpy().astype(np.float32),
        action_base.detach().cpu().numpy().astype(np.float32),
    )


def _gaussian_prior_noise_and_base(
    model: Any, observation: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Spec 5.3: sample an independent standard-Gaussian decoder prior.

    The standard-normal prior is the exact decoder input (matching the base
    DSRL QW distillation's ``th.randn`` source).  The stored ``noise_scaled``
    is the audited affine ``scale_action`` image of that prior; the inverse
    ``_unscale_noise``/``unscale_action`` reproduces the decoder input within
    ``1e-6``.  Log-probs are placeholder zero because the prior was not drawn
    by the SAC actor.
    """

    observation_tensor = torch.as_tensor(
        observation,
        device=model.device,
        dtype=torch.float32,
    )
    batch = observation_tensor.shape[0]
    noise_prior = torch.randn(
        batch,
        int(model.diffusion_act_chunk) * int(model.diffusion_act_dim),
        device=model.device,
        dtype=torch.float32,
    )
    decoder = noise_prior.reshape(
        batch,
        int(model.diffusion_act_chunk),
        int(model.diffusion_act_dim),
    )
    with torch.no_grad():
        action_base = model.diffusion_policy(
            observation_tensor, decoder, return_numpy=False
        ).reshape(batch, -1)
    noise_scaled = model.policy.scale_action(
        noise_prior.detach().cpu().numpy()
    ).astype(np.float32)
    return (
        noise_scaled,
        np.zeros(batch, dtype=np.float32),
        action_base.detach().cpu().numpy().astype(np.float32),
    )


def collect_tagged_matched_prefill(
    *,
    warmstart_model: Any,
    env: Any,
    vector_steps: int,
    environment_seed: int,
    policy_seed: int,
    action_chunk: int,
    termination_semantics: str,
    provenance: Mapping[str, str],
    prefill_source: str = PREFILL_SOURCE_WARMSTART,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if vector_steps <= 0:
        raise ValueError("Tagged matched prefill requires vector_steps > 0")
    if prefill_source not in (PREFILL_SOURCE_WARMSTART, PREFILL_SOURCE_FRESH):
        raise ValueError(f"Unknown tagged prefill source {prefill_source!r}")
    n_envs = int(env.num_envs)
    action_dim = int(np.prod(env.action_space.shape))
    counters = PrimitiveCounters()
    values: dict[str, list[np.ndarray]] = {
        name: [] for name in TAGGED_PREFILL_ARRAY_NAMES
    }
    episode_ids = np.arange(n_envs, dtype=np.int64)
    next_episode_id = n_envs
    chunk_indices = np.zeros(n_envs, dtype=np.int32)
    semantic_value = (
        int(TerminationSemantics.EARLY_BREAK_ON_DONE)
        if termination_semantics == "early_break_on_done"
        else int(TerminationSemantics.LEGACY_CONTINUE_AFTER_DONE)
    )
    fresh = prefill_source == PREFILL_SOURCE_FRESH
    with isolated_rng(policy_seed):
        env.seed(environment_seed)
        observation = np.asarray(env.reset())
        for _ in range(vector_steps):
            if fresh:
                noise_scaled, log_prob, action_base = (
                    _gaussian_prior_noise_and_base(warmstart_model, observation)
                )
            else:
                noise_scaled, log_prob, action_base = _legacy_noise_and_base(
                    warmstart_model, observation
                )
            new_observation, reward, done, infos = env.step(action_base)
            corrected_next = terminal_corrected_next_observations(
                new_observation, done, infos
            )
            counters.update(infos, done, action_chunk=action_chunk)
            timeout = np.asarray(
                [
                    bool(done[index])
                    and bool(info.get("TimeLimit.truncated", False))
                    for index, info in enumerate(infos)
                ],
                dtype=np.bool_,
            )
            done_array = np.asarray(done, dtype=np.bool_)
            zeros_action = np.zeros((n_envs, action_dim), dtype=np.float32)
            values["observations"].append(np.asarray(observation).copy())
            values["next_observations"].append(corrected_next.copy())
            values["rewards"].append(np.asarray(reward, dtype=np.float32))
            values["dones"].append(done_array.copy())
            values["timeouts"].append(timeout.copy())
            values["truncated"].append(timeout.copy())
            values["terminated"].append(done_array & ~timeout)
            values["branch_mode"].append(
                np.full(n_envs, int(BranchMode.BASE), dtype=np.uint8)
            )
            values["noise_scaled"].append(noise_scaled.copy())
            values["noise_log_prob"].append(log_prob.copy())
            values["noise_log_prob_valid"].append(
                np.zeros(n_envs, np.bool_) if fresh else np.ones(n_envs, np.bool_)
            )
            values["noise_sample_source"].append(
                np.full(
                    n_envs,
                    (
                        int(NoiseSampleSource.GAUSSIAN_PRIOR)
                        if fresh
                        else int(NoiseSampleSource.CURRENT_ACTOR)
                    ),
                    np.uint8,
                )
            )
            values["transition_origin"].append(
                np.full(n_envs, int(TransitionOrigin.PREFILL), np.uint8)
            )
            values["action_base"].append(action_base.copy())
            values["residual_pre_tanh"].append(zeros_action.copy())
            values["residual_unit"].append(zeros_action.copy())
            values["action_residual_delta"].append(zeros_action.copy())
            values["action_exec"].append(action_base.copy())
            values["beta"].append(np.zeros(n_envs, np.float32))
            values["residual_applied"].append(np.zeros(n_envs, np.bool_))
            values["emergency_clamp_applied"].append(
                np.zeros(n_envs, np.bool_)
            )
            values["episode_id"].append(episode_ids.copy())
            values["environment_id"].append(np.arange(n_envs, dtype=np.int32))
            values["chunk_index_in_episode"].append(chunk_indices.copy())
            values["nominal_primitive_steps"].append(
                np.full(n_envs, action_chunk, np.uint8)
            )
            values["actual_primitive_steps"].append(
                np.asarray(
                    [int(info["actual_primitive_steps"]) for info in infos],
                    dtype=np.uint8,
                )
            )
            values["termination_primitive_index"].append(
                np.asarray(
                    [
                        -1
                        if info.get("termination_primitive_index") is None
                        else int(info["termination_primitive_index"])
                        for info in infos
                    ],
                    dtype=np.int8,
                )
            )
            values["termination_semantics"].append(
                np.full(n_envs, semantic_value, np.uint8)
            )
            values["noise_policy_version"].append(
                np.full(n_envs, -1, np.int64) if fresh else np.zeros(n_envs, np.int64)
            )
            values["residual_policy_version"].append(
                np.full(n_envs, -1, np.int64)
            )
            for environment_id, terminal in enumerate(done_array):
                if terminal:
                    episode_ids[environment_id] = next_episode_id
                    next_episode_id += 1
                    chunk_indices[environment_id] = 0
                else:
                    chunk_indices[environment_id] += 1
            observation = np.asarray(new_observation)
    arrays = {name: np.asarray(entries) for name, entries in values.items()}
    metadata: dict[str, Any] = {
        "format_version": TAGGED_PREFILL_FORMAT_VERSION,
        "vector_steps": vector_steps,
        "chunk_transitions": vector_steps * n_envs,
        "n_envs": n_envs,
        "action_chunk": action_chunk,
        "environment_seed": environment_seed,
        "policy_seed": policy_seed,
        "termination_semantics": termination_semantics,
        "prefill_source": prefill_source,
        "init_checkpoint_sha256": provenance["init_checkpoint_sha256"],
        "frozen_ddim_sha256": provenance["frozen_ddim_sha256"],
        "normalization_sha256": provenance["normalization_sha256"],
        "primitive_counters": counters.to_dict(),
    }
    validate_tagged_prefill_semantics(
        arrays, metadata, prefill_source=prefill_source
    )
    full_hashes = hash_semantic_arrays(
        arrays,
        _tagged_prefill_hash_metadata(metadata),
        array_names=TAGGED_PREFILL_ARRAY_NAMES,
    )
    projection = tagged_prefill_standard_projection(arrays)
    projection_hashes = hash_semantic_arrays(
        projection,
        _tagged_prefill_hash_metadata(metadata),
        array_names=(
            "observations",
            "next_observations",
            "actions",
            "rewards",
            "dones",
            "timeouts",
        ),
    )
    metadata.update(
        {
            "semantic_hash": full_hashes["semantic_hash"],
            "per_array_hashes": full_hashes["per_array_hashes"],
            "projection_semantic_hash": projection_hashes["semantic_hash"],
            "projection_per_array_hashes": projection_hashes["per_array_hashes"],
        }
    )
    return arrays, metadata


def _load_tagged_matched_prefill_unlocked(
    *,
    artifact_path: Path,
    metadata_path: Path,
    expected: Mapping[str, Any],
    prefill_source: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load and fully validate a complete tagged bundle while its lock is held."""

    with metadata_path.open("r", encoding="utf-8") as source:
        metadata = json.load(source)
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(
                f"Tagged prefill metadata mismatch for {key}: "
                f"expected {value!r}, got {metadata.get(key)!r}"
            )
    if sha256_file(artifact_path) != metadata.get("archive_sha256"):
        raise ValueError("Tagged prefill archive SHA-256 mismatch")
    with np.load(artifact_path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    validate_tagged_prefill_semantics(
        arrays, metadata, prefill_source=prefill_source
    )
    actual = hash_semantic_arrays(
        arrays,
        _tagged_prefill_hash_metadata(metadata),
        array_names=TAGGED_PREFILL_ARRAY_NAMES,
    )
    if actual != {
        "semantic_hash": metadata.get("semantic_hash"),
        "per_array_hashes": metadata.get("per_array_hashes"),
    }:
        raise ValueError("Tagged prefill semantic hash mismatch")
    projection = tagged_prefill_standard_projection(arrays)
    projection_actual = hash_semantic_arrays(
        projection,
        _tagged_prefill_hash_metadata(metadata),
        array_names=(
            "observations",
            "next_observations",
            "actions",
            "rewards",
            "dones",
            "timeouts",
        ),
    )
    if projection_actual != {
        "semantic_hash": metadata.get("projection_semantic_hash"),
        "per_array_hashes": metadata.get("projection_per_array_hashes"),
    }:
        raise ValueError("Tagged prefill projection hash mismatch")
    return arrays, metadata


def collect_or_load_tagged_matched_prefill(
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
    prefill_source: str = PREFILL_SOURCE_WARMSTART,
) -> tuple[dict[str, np.ndarray], dict[str, Any], bool]:
    if prefill_source not in (PREFILL_SOURCE_WARMSTART, PREFILL_SOURCE_FRESH):
        raise ValueError(f"Unknown tagged prefill source {prefill_source!r}")
    metadata_path = artifact_path.with_suffix(artifact_path.suffix + ".json")
    expected = {
        "format_version": TAGGED_PREFILL_FORMAT_VERSION,
        "vector_steps": int(vector_steps),
        "chunk_transitions": int(vector_steps) * int(env.num_envs),
        "n_envs": int(env.num_envs),
        "action_chunk": int(action_chunk),
        "environment_seed": int(environment_seed),
        "policy_seed": int(policy_seed),
        "termination_semantics": termination_semantics,
        "prefill_source": prefill_source,
        "init_checkpoint_sha256": provenance["init_checkpoint_sha256"],
        "frozen_ddim_sha256": provenance["frozen_ddim_sha256"],
        "normalization_sha256": provenance["normalization_sha256"],
    }
    with _prefill_artifact_lock(artifact_path):
        generated = False
        if artifact_path.exists() or metadata_path.exists():
            if artifact_path.is_file() and not metadata_path.exists():
                # Only the lock owner may repair a crash window.  A live writer
                # can no longer have the archive renamed while a peer deletes it.
                artifact_path.unlink()
                warnings.warn(
                    "Removed orphaned prefill archive without metadata: "
                    f"{artifact_path}"
                )
            elif metadata_path.is_file() and not artifact_path.exists():
                metadata_path.unlink()
                warnings.warn(
                    "Removed orphaned prefill metadata without archive: "
                    f"{metadata_path}"
                )
            elif not artifact_path.is_file() or not metadata_path.is_file():
                raise FileNotFoundError(
                    "Tagged prefill path exists but is not a regular file pair: "
                    f"{artifact_path} / {metadata_path}"
                )

        if not artifact_path.exists() and not metadata_path.exists():
            collect_tagged_matched_prefill_and_save(
                artifact_path=artifact_path,
                metadata_path=metadata_path,
                warmstart_model=warmstart_model,
                env=env,
                vector_steps=vector_steps,
                environment_seed=environment_seed,
                policy_seed=policy_seed,
                action_chunk=action_chunk,
                termination_semantics=termination_semantics,
                provenance=provenance,
                prefill_source=prefill_source,
            )
            generated = True
        elif not artifact_path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError(
                "Tagged prefill recovery did not produce a complete regular-file pair: "
                f"{artifact_path} / {metadata_path}"
            )

        # Even the generating process reloads from the final two-file bundle.
        # Consequently every consumer populates replay from the same validated
        # bytes, never from private arrays retained before the atomic renames.
        arrays, metadata = _load_tagged_matched_prefill_unlocked(
            artifact_path=artifact_path,
            metadata_path=metadata_path,
            expected=expected,
            prefill_source=prefill_source,
        )
        return arrays, metadata, generated


def collect_tagged_matched_prefill_and_save(
    *,
    artifact_path: Path,
    metadata_path: Path,
    warmstart_model: Any,
    env: Any,
    vector_steps: int,
    environment_seed: int,
    policy_seed: int,
    action_chunk: int,
    termination_semantics: str,
    provenance: Mapping[str, str],
    prefill_source: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any], bool]:
    """Collect (or re-collect after orphan recovery) and atomically persist a
    tagged prefill, returning (arrays, metadata, generated=True)."""
    arrays, metadata = collect_tagged_matched_prefill(
        warmstart_model=warmstart_model,
        env=env,
        vector_steps=vector_steps,
        environment_seed=environment_seed,
        policy_seed=policy_seed,
        action_chunk=action_chunk,
        termination_semantics=termination_semantics,
        provenance=provenance,
        prefill_source=prefill_source,
    )
    metadata = save_prefill_artifact(
        artifact_path,
        arrays,
        metadata,
        validator=partial(
            validate_tagged_prefill_semantics,
            prefill_source=prefill_source,
        ),
    )
    return arrays, metadata, True


def populate_tagged_replay_buffer(
    replay_buffer: HierarchyTaggedReplayBuffer,
    arrays: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(replay_buffer, HierarchyTaggedReplayBuffer):
        raise TypeError("Hierarchy prefill requires HierarchyTaggedReplayBuffer")
    if replay_buffer.pos != 0 or replay_buffer.full:
        raise RuntimeError("Matched prefill requires an empty replay")
    validate_tagged_prefill_semantics(
        arrays,
        metadata,
        # The tagged collector stamps the source into the metadata; a FRESH
        # prefill must be validated against the FRESH rules (noise policy
        # version -1, invalid actor log-probs), not the WARMSTART defaults.
        prefill_source=metadata.get(
            "prefill_source", PREFILL_SOURCE_WARMSTART
        ),
    )
    vector_steps, n_envs = np.asarray(arrays["rewards"]).shape
    if replay_buffer.n_envs != n_envs:
        raise ValueError(
            f"Replay n_envs mismatch: {replay_buffer.n_envs} != {n_envs}"
        )
    if replay_buffer.buffer_size <= vector_steps:
        raise ValueError(
            "Hierarchy replay must have capacity beyond the prefill prefix"
        )
    for step in range(vector_steps):
        infos = [
            {"TimeLimit.truncated": bool(arrays["timeouts"][step, env_id])}
            for env_id in range(n_envs)
        ]
        metadata = {
            name: np.asarray(arrays[name][step])
            for name in (*ACTION_METADATA_FIELDS, *SCALAR_METADATA_DTYPES.keys())
        }
        replay_buffer.add_hierarchy(
            obs=arrays["observations"][step],
            next_obs=arrays["next_observations"][step],
            action=arrays["action_exec"][step],
            reward=arrays["rewards"][step],
            done=arrays["dones"][step],
            infos=infos,
            metadata=metadata,
        )
    replay_buffer.final_offline_step()
    return {
        "semantic_hash": replay_buffer.semantic_hash(vector_rows=vector_steps),
        "schema_version": replay_buffer.schema_version,
        "vector_steps": vector_steps,
        "chunk_transitions": vector_steps * n_envs,
        "offline_steps": int(replay_buffer.offline_steps),
        "branch_counts": dict(replay_buffer.branch_counts),
    }


def hash_replay_for_resume(replay_buffer: Any) -> dict[str, Any]:
    """Return a schema-aware resume hash without projecting away hierarchy tags."""

    if isinstance(replay_buffer, HierarchyTaggedReplayBuffer):
        rows = replay_buffer.buffer_size if replay_buffer.full else replay_buffer.pos
        return {
            "semantic_hash": replay_buffer.semantic_hash(),
            "schema_version": replay_buffer.schema_version,
            "vector_steps": int(rows),
            "chunk_transitions": int(rows) * int(replay_buffer.n_envs),
            "n_envs": int(replay_buffer.n_envs),
            "offline_steps": int(replay_buffer.offline_steps),
            "branch_counts": dict(replay_buffer.branch_counts),
        }
    return hash_replay_prefix(replay_buffer)


def hash_replay_offline_for_resume(replay_buffer: Any) -> dict[str, Any]:
    """Hash exactly the immutable prefill prefix with schema-aware semantics."""

    if isinstance(replay_buffer, HierarchyTaggedReplayBuffer):
        rows = int(replay_buffer.offline_steps)
        return {
            "semantic_hash": replay_buffer.semantic_hash(vector_rows=rows),
            "schema_version": replay_buffer.schema_version,
            "vector_steps": rows,
            "chunk_transitions": rows * int(replay_buffer.n_envs),
            "n_envs": int(replay_buffer.n_envs),
            "offline_steps": rows,
        }
    return hash_replay_offline_prefix(replay_buffer)


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


def hash_replay_prefix(
    replay_buffer: Any,
    *,
    vector_steps: int | None = None,
    offline_steps: int | None = None,
) -> dict[str, Any]:
    if replay_buffer.full:
        raise ValueError("P6 semantic hashing currently requires a non-wrapped buffer")
    available_steps = int(replay_buffer.pos)
    if vector_steps is None:
        vector_steps = available_steps
    vector_steps = int(vector_steps)
    if vector_steps < 0 or vector_steps > available_steps:
        raise ValueError(
            f"Replay hash prefix {vector_steps} exceeds available {available_steps}"
        )
    if offline_steps is None:
        offline_steps = int(replay_buffer.offline_steps)
    offline_steps = int(offline_steps)
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
        "offline_steps": offline_steps,
    }
    return {
        **metadata,
        **hash_semantic_arrays(arrays, metadata),
    }


def hash_replay_offline_prefix(replay_buffer: Any) -> dict[str, Any]:
    offline_steps = int(replay_buffer.offline_steps)
    return hash_replay_prefix(
        replay_buffer,
        vector_steps=offline_steps,
        offline_steps=offline_steps,
    )
