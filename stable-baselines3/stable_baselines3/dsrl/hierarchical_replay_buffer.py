"""Tagged replay storage for the three-critic DSRL hierarchy.

The common SB3 replay buffer deliberately remains unchanged.  This subclass
stores the exact behavior tuple required to keep BASE and JOINT continuation
values separate and exposes branch-filtered sampling over flattened
``(row, environment)`` slots.
"""

from __future__ import annotations

import hashlib
import json
from enum import IntEnum
from typing import Any, Mapping, NamedTuple, Optional

import numpy as np
import torch as th

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.vec_env import VecNormalize


class BranchMode(IntEnum):
    BASE = 0
    JOINT = 1


class NoiseSampleSource(IntEnum):
    CURRENT_ACTOR = 0
    REFERENCE_ACTOR = 1
    GAUSSIAN_PRIOR = 2
    UNIFORM_WARMUP = 3


class TransitionOrigin(IntEnum):
    ONLINE = 0
    PREFILL = 1


class TerminationSemantics(IntEnum):
    LEGACY_CONTINUE_AFTER_DONE = 0
    EARLY_BREAK_ON_DONE = 1


SCHEMA_VERSION = "hierarchy_tagged_replay_v1"


class HierarchyReplayBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    next_observations: th.Tensor
    dones: th.Tensor
    rewards: th.Tensor
    branch_mode: th.Tensor
    noise_scaled: th.Tensor
    noise_log_prob: th.Tensor
    noise_log_prob_valid: th.Tensor
    noise_sample_source: th.Tensor
    transition_origin: th.Tensor
    action_base: th.Tensor
    residual_pre_tanh: th.Tensor
    residual_unit: th.Tensor
    action_residual_delta: th.Tensor
    action_exec: th.Tensor
    beta: th.Tensor
    residual_applied: th.Tensor
    emergency_clamp_applied: th.Tensor
    episode_id: th.Tensor
    environment_id: th.Tensor
    chunk_index_in_episode: th.Tensor
    nominal_primitive_steps: th.Tensor
    actual_primitive_steps: th.Tensor
    termination_primitive_index: th.Tensor
    termination_semantics: th.Tensor
    noise_policy_version: th.Tensor
    residual_policy_version: th.Tensor
    terminated: th.Tensor
    truncated: th.Tensor


ACTION_METADATA_FIELDS = (
    "noise_scaled",
    "action_base",
    "residual_pre_tanh",
    "residual_unit",
    "action_residual_delta",
    "action_exec",
)

SCALAR_METADATA_DTYPES: dict[str, np.dtype[Any]] = {
    "branch_mode": np.dtype(np.uint8),
    "noise_log_prob": np.dtype(np.float32),
    "noise_log_prob_valid": np.dtype(np.bool_),
    "noise_sample_source": np.dtype(np.uint8),
    "transition_origin": np.dtype(np.uint8),
    "beta": np.dtype(np.float32),
    "residual_applied": np.dtype(np.bool_),
    "emergency_clamp_applied": np.dtype(np.bool_),
    "episode_id": np.dtype(np.int64),
    "environment_id": np.dtype(np.int32),
    "chunk_index_in_episode": np.dtype(np.int32),
    "nominal_primitive_steps": np.dtype(np.uint8),
    "actual_primitive_steps": np.dtype(np.uint8),
    "termination_primitive_index": np.dtype(np.int8),
    "termination_semantics": np.dtype(np.uint8),
    "noise_policy_version": np.dtype(np.int64),
    "residual_policy_version": np.dtype(np.int64),
}

HIERARCHY_SEMANTIC_ARRAY_NAMES = (
    "observations",
    "next_observations",
    "actions",
    "rewards",
    "dones",
    "timeouts",
    "terminated",
    "truncated",
    *SCALAR_METADATA_DTYPES.keys(),
    *ACTION_METADATA_FIELDS,
)


class HierarchyTaggedReplayBuffer(ReplayBuffer):
    """One physical replay with exact BASE/JOINT behavior metadata."""

    schema_version = SCHEMA_VERSION

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if bool(kwargs.get("optimize_memory_usage", False)):
            raise ValueError(
                "HierarchyTaggedReplayBuffer does not support "
                "optimize_memory_usage=True; exact tagged next observations are "
                "required for branch-filtered sampling and semantic hashing"
            )
        super().__init__(*args, **kwargs)
        shape = (self.buffer_size, self.n_envs, self.action_dim)
        for name in ACTION_METADATA_FIELDS:
            setattr(self, name, np.zeros(shape, dtype=np.float32))
        for name, dtype in SCALAR_METADATA_DTYPES.items():
            setattr(
                self,
                name,
                np.zeros((self.buffer_size, self.n_envs), dtype=dtype),
            )
        self.terminated = np.zeros(
            (self.buffer_size, self.n_envs), dtype=np.bool_
        )
        self.truncated = np.zeros(
            (self.buffer_size, self.n_envs), dtype=np.bool_
        )
        self._row_valid = np.zeros(self.buffer_size, dtype=np.bool_)
        self.branch_counts = {
            int(BranchMode.BASE): 0,
            int(BranchMode.JOINT): 0,
        }
        self._staged_metadata: Optional[Mapping[str, Any]] = None

    def stage_metadata(self, metadata: Mapping[str, Any]) -> None:
        """Stage one vector row for the audited SB3 ``_store_transition`` path.

        ``OffPolicyAlgorithm._store_transition`` owns terminal-observation and
        VecNormalize correction.  The hierarchy stages its behavior tuple,
        invokes that implementation, and this method lets the ensuing standard
        ``add`` call consume the tuple atomically.  Calling standard ``add``
        without a staged tuple remains a hard error.
        """

        if self._staged_metadata is not None:
            raise RuntimeError("A hierarchy metadata row is already staged")
        self._staged_metadata = metadata

    def clear_staged_metadata(self) -> None:
        self._staged_metadata = None

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: list[dict[str, Any]],
        **kwargs: Any,
    ) -> None:
        if kwargs:
            raise TypeError(
                "HierarchyTaggedReplayBuffer.add received unsupported keyword "
                f"arguments: {sorted(kwargs)}"
            )
        if self._staged_metadata is None:
            raise RuntimeError(
                "HierarchyTaggedReplayBuffer requires add_hierarchy() or a "
                "stage_metadata()/SB3 _store_transition transaction"
            )
        metadata = self._staged_metadata
        self._staged_metadata = None
        self.add_hierarchy(
            obs,
            next_obs,
            action,
            reward,
            done,
            infos,
            metadata=metadata,
        )

    @staticmethod
    def _as_scalar_vector(
        name: str,
        value: Any,
        n_envs: int,
        dtype: np.dtype[Any],
    ) -> np.ndarray:
        result = np.asarray(value, dtype=dtype)
        if result.shape == (n_envs, 1):
            result = result[:, 0]
        if result.shape != (n_envs,):
            raise ValueError(
                f"{name} must have shape ({n_envs},) or ({n_envs}, 1), "
                f"got {result.shape}"
            )
        return result

    def _validated_metadata(
        self,
        metadata: Mapping[str, Any],
        action: np.ndarray,
        done: np.ndarray,
        infos: list[dict[str, Any]],
    ) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
        missing = set(ACTION_METADATA_FIELDS) | set(SCALAR_METADATA_DTYPES)
        missing -= set(metadata)
        if missing:
            raise ValueError(
                "Hierarchy transition is missing metadata fields: "
                + ", ".join(sorted(missing))
            )

        validated: dict[str, np.ndarray] = {}
        for name in ACTION_METADATA_FIELDS:
            value = np.asarray(metadata[name], dtype=np.float32)
            if value.shape != (self.n_envs, self.action_dim):
                raise ValueError(
                    f"{name} must have shape "
                    f"({self.n_envs}, {self.action_dim}), got {value.shape}"
                )
            if not np.isfinite(value).all():
                raise ValueError(f"{name} contains non-finite values")
            validated[name] = value

        for name, dtype in SCALAR_METADATA_DTYPES.items():
            validated[name] = self._as_scalar_vector(
                name, metadata[name], self.n_envs, dtype
            )
        for name in ("noise_log_prob", "beta"):
            if not np.isfinite(validated[name]).all():
                raise ValueError(f"{name} must contain only finite values")

        action_reshaped = np.asarray(action, dtype=np.float32).reshape(
            self.n_envs, self.action_dim
        )
        if not np.array_equal(action_reshaped, validated["action_exec"]):
            raise ValueError("Replay action must equal metadata action_exec exactly")
        if not np.array_equal(
            validated["environment_id"], np.arange(self.n_envs, dtype=np.int32)
        ):
            raise ValueError("environment_id must enumerate vector slots in order")

        branches = validated["branch_mode"]
        if np.any(
            (branches != int(BranchMode.BASE))
            & (branches != int(BranchMode.JOINT))
        ):
            raise ValueError("branch_mode contains an unknown lane")
        base = branches == int(BranchMode.BASE)
        joint = branches == int(BranchMode.JOINT)
        for name in (
            "residual_pre_tanh",
            "residual_unit",
            "action_residual_delta",
        ):
            if np.any(validated[name][base] != 0):
                raise ValueError(f"BASE transitions require exact-zero {name}")
        if np.any(validated["residual_applied"][base]):
            raise ValueError("BASE transitions cannot mark residual_applied")
        if np.any(validated["residual_policy_version"][base] != -1):
            raise ValueError(
                "BASE transitions require residual_policy_version=-1"
            )
        if not np.array_equal(
            validated["action_base"][base], validated["action_exec"][base]
        ):
            raise ValueError("BASE action_exec must equal action_base exactly")
        if np.any(validated["beta"] < 0) or np.any(validated["beta"] > 1):
            raise ValueError("beta must be finite and lie in [0, 1]")
        if np.any(validated["episode_id"] < 0):
            raise ValueError("episode_id must be non-negative")
        if np.any(validated["chunk_index_in_episode"] < 0):
            raise ValueError("chunk_index_in_episode must be non-negative")
        if np.any(validated["nominal_primitive_steps"] <= 0):
            raise ValueError("nominal_primitive_steps must be positive")
        if np.any(
            validated["actual_primitive_steps"]
            > validated["nominal_primitive_steps"]
        ):
            raise ValueError(
                "actual_primitive_steps cannot exceed nominal_primitive_steps"
            )
        if np.any(
            validated["residual_applied"][joint]
            & (validated["residual_policy_version"][joint] < 0)
        ):
            raise ValueError(
                "Applied JOINT residuals require a non-negative policy version"
            )

        prior_or_warmup = np.isin(
            validated["noise_sample_source"],
            [
                int(NoiseSampleSource.GAUSSIAN_PRIOR),
                int(NoiseSampleSource.UNIFORM_WARMUP),
            ],
        )
        if np.any(validated["noise_log_prob_valid"][prior_or_warmup]):
            raise ValueError("External/warmup noise cannot have a valid actor log-prob")
        if np.any(validated["noise_policy_version"][prior_or_warmup] != -1):
            raise ValueError("External/warmup noise requires policy version -1")

        done_bool = np.asarray(done, dtype=np.bool_).reshape(self.n_envs)
        # truncated carries the wrapper's TimeLimit signal INDEPENDENTLY of
        # done_bool: the `truncated & ~done_bool` invariant below is only
        # reachable if truncated is not derived from done.  (The previous code
        # ANDed the two together first, which made the guard vacuous -- a
        # timeout whose done bit was unset was silently accepted and mis-stored
        # as a terminal transition.)  Under ActionChunkWrapper EARLY_BREAK
        # semantics a wrapper-limit timeout always also sets done, so this
        # guard is expected to stay silent in production.
        truncated = np.asarray(
            [
                bool(info.get("TimeLimit.truncated", False))
                for info in infos
            ],
            dtype=np.bool_,
        )
        if np.any(truncated & ~done_bool):
            raise ValueError("timeout_without_done")
        terminated = done_bool & ~truncated
        return validated, terminated, truncated

    def add_hierarchy(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: list[dict[str, Any]],
        *,
        metadata: Mapping[str, Any],
    ) -> None:
        validated, terminated, truncated = self._validated_metadata(
            metadata, action, done, infos
        )
        row = self.pos
        if self._row_valid[row]:
            old = self.branch_mode[row]
            for branch in (BranchMode.BASE, BranchMode.JOINT):
                self.branch_counts[int(branch)] -= int(
                    np.count_nonzero(old == int(branch))
                )

        # Bypass this class's fail-fast add(), but retain the audited common
        # storage layout and timeout handling.
        ReplayBuffer.add(self, obs, next_obs, action, reward, done, infos)

        for name, value in validated.items():
            getattr(self, name)[row] = value
        self.terminated[row] = terminated
        self.truncated[row] = truncated
        self._row_valid[row] = True
        for branch in (BranchMode.BASE, BranchMode.JOINT):
            self.branch_counts[int(branch)] += int(
                np.count_nonzero(validated["branch_mode"] == int(branch))
            )

    def reset(self) -> None:
        super().reset()
        self._row_valid.fill(False)
        self.branch_counts = {
            int(BranchMode.BASE): 0,
            int(BranchMode.JOINT): 0,
        }
        self._staged_metadata = None

    def rebuild_branch_counts(self) -> dict[int, int]:
        counts = {int(BranchMode.BASE): 0, int(BranchMode.JOINT): 0}
        valid_rows = np.flatnonzero(self._row_valid)
        if valid_rows.size:
            values = self.branch_mode[valid_rows].reshape(-1)
            for branch in (BranchMode.BASE, BranchMode.JOINT):
                counts[int(branch)] = int(
                    np.count_nonzero(values == int(branch))
                )
        self.branch_counts = counts
        return counts.copy()

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        if getattr(self, "schema_version", None) != SCHEMA_VERSION:
            raise ValueError(
                "Tagged replay schema mismatch; regenerate the hierarchy replay"
            )
        self._staged_metadata = None
        recorded = dict(self.branch_counts)
        rebuilt = self.rebuild_branch_counts()
        if recorded != rebuilt:
            raise ValueError(
                "Tagged replay branch counts do not match stored metadata: "
                f"recorded={recorded}, rebuilt={rebuilt}"
            )

    def branch_count(self, branch: BranchMode | int) -> int:
        return int(self.branch_counts[int(branch)])

    def _eligible_pairs(self, branch: Optional[BranchMode]) -> np.ndarray:
        valid = np.broadcast_to(self._row_valid[:, None], self.branch_mode.shape)
        if branch is not None:
            valid = valid & (self.branch_mode == int(branch))
        pairs = np.argwhere(valid)
        if pairs.size == 0:
            label = "ANY" if branch is None else branch.name
            raise ValueError(f"No valid {label} transitions are available")
        return pairs

    def _get_hierarchy_samples(
        self,
        rows: np.ndarray,
        env_indices: np.ndarray,
        env: Optional[VecNormalize],
    ) -> HierarchyReplayBufferSamples:
        if self.optimize_memory_usage:
            next_obs = self.observations[
                (rows + 1) % self.buffer_size, env_indices
            ]
        else:
            next_obs = self.next_observations[rows, env_indices]

        data = (
            self._normalize_obs(self.observations[rows, env_indices], env),
            self.actions[rows, env_indices],
            self._normalize_obs(next_obs, env),
            (
                self.dones[rows, env_indices]
                * (1 - self.timeouts[rows, env_indices])
            ).reshape(-1, 1),
            self._normalize_reward(
                self.rewards[rows, env_indices].reshape(-1, 1), env
            ),
            self.branch_mode[rows, env_indices].reshape(-1, 1),
            self.noise_scaled[rows, env_indices],
            self.noise_log_prob[rows, env_indices].reshape(-1, 1),
            self.noise_log_prob_valid[rows, env_indices].reshape(-1, 1),
            self.noise_sample_source[rows, env_indices].reshape(-1, 1),
            self.transition_origin[rows, env_indices].reshape(-1, 1),
            self.action_base[rows, env_indices],
            self.residual_pre_tanh[rows, env_indices],
            self.residual_unit[rows, env_indices],
            self.action_residual_delta[rows, env_indices],
            self.action_exec[rows, env_indices],
            self.beta[rows, env_indices].reshape(-1, 1),
            self.residual_applied[rows, env_indices].reshape(-1, 1),
            self.emergency_clamp_applied[rows, env_indices].reshape(-1, 1),
            self.episode_id[rows, env_indices].reshape(-1, 1),
            self.environment_id[rows, env_indices].reshape(-1, 1),
            self.chunk_index_in_episode[rows, env_indices].reshape(-1, 1),
            self.nominal_primitive_steps[rows, env_indices].reshape(-1, 1),
            self.actual_primitive_steps[rows, env_indices].reshape(-1, 1),
            self.termination_primitive_index[rows, env_indices].reshape(-1, 1),
            self.termination_semantics[rows, env_indices].reshape(-1, 1),
            self.noise_policy_version[rows, env_indices].reshape(-1, 1),
            self.residual_policy_version[rows, env_indices].reshape(-1, 1),
            self.terminated[rows, env_indices].reshape(-1, 1),
            self.truncated[rows, env_indices].reshape(-1, 1),
        )
        return HierarchyReplayBufferSamples(*tuple(map(self.to_torch, data)))

    def sample_branch(
        self,
        branch: BranchMode | int,
        batch_size: int,
        env: Optional[VecNormalize] = None,
    ) -> HierarchyReplayBufferSamples:
        branch_enum = BranchMode(int(branch))
        pairs = self._eligible_pairs(branch_enum)
        selected = pairs[np.random.randint(0, len(pairs), size=batch_size)]
        return self._get_hierarchy_samples(selected[:, 0], selected[:, 1], env)

    def sample_any(
        self,
        batch_size: int,
        env: Optional[VecNormalize] = None,
    ) -> HierarchyReplayBufferSamples:
        pairs = self._eligible_pairs(None)
        selected = pairs[np.random.randint(0, len(pairs), size=batch_size)]
        return self._get_hierarchy_samples(selected[:, 0], selected[:, 1], env)

    def sample(
        self,
        batch_size: int,
        env: Optional[VecNormalize] = None,
    ) -> HierarchyReplayBufferSamples:
        return self.sample_any(batch_size, env=env)

    def semantic_hash(self, *, vector_rows: Optional[int] = None) -> str:
        """Hash every behavior-semantic array and the valid-row boundary.

        The whole-buffer digest (``vector_rows`` omitted) includes the
        circular-buffer position and fullness so that two full buffers that
        share the same physical bytes but were filled starting at different
        circular offsets are not treated as the same replay state: ``pos`` is
        part of the replay's semantics, and the digest must track it.  The
        prefix digest (``vector_rows`` given) deliberately omits ``pos``/``full``
        so the immutable prefill prefix hashes the same at prefill time
        (non-full) and at resume time (full, wrapped): the prefix rows
        themselves are rotation-invariant, and P6 compares that hash across the
        two buffer states.
        """

        if self._staged_metadata is not None:
            raise RuntimeError("Cannot hash replay while metadata is staged")
        if self.full:
            rows = self.buffer_size
        else:
            rows = self.pos
        if vector_rows is not None:
            rows = int(vector_rows)
            if rows < 0 or rows > (self.buffer_size if self.full else self.pos):
                raise ValueError("semantic hash row prefix is outside valid replay")
        digest = hashlib.sha256()
        header = {
            "schema_version": self.schema_version,
            "rows": rows,
            "n_envs": self.n_envs,
            "action_dim": self.action_dim,
            "offline_steps": int(self.offline_steps),
        }
        if vector_rows is None:
            header["pos"] = int(self.pos)
            header["full"] = bool(self.full)
        digest.update(
            json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
        )
        for name in HIERARCHY_SEMANTIC_ARRAY_NAMES:
            array = np.ascontiguousarray(getattr(self, name)[:rows])
            digest.update(name.encode())
            digest.update(str(array.dtype).encode())
            digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
            digest.update(array.tobytes(order="C"))
        # The valid-row mask is part of replay semantics: two buffers holding
        # identical arrays but valid on different rows yield different samples.
        # Hash it as a dense byte mask over the same row prefix so slot-vs-
        # compact layout never leaks into the digest.
        valid = np.ascontiguousarray(self._row_valid[:rows])
        digest.update(b"_row_valid")
        digest.update(str(valid.dtype).encode())
        digest.update(np.asarray(valid.shape, dtype=np.int64).tobytes())
        digest.update(valid.tobytes(order="C"))
        return digest.hexdigest()
