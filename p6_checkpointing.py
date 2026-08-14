"""Safe-boundary P6 checkpoints, replay bundles, counters, and callbacks."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback

from p6_preflight import sha256_file
from p6_runtime import (
    PrimitiveCounters,
    atomic_write_json,
    capture_rng_state,
    hash_replay_offline_for_resume,
    hash_replay_for_resume,
)

RUN_BINDING_KEYS = (
    "run_id",
    "config_contract_sha256",
    "source_state_sha256",
    "prefill_semantic_hash",
    "initial_replay_semantic_hash",
)


class P6IntentionalInterruption(RuntimeError):
    """Raised only after a certified safe resume bundle has been written."""


def _fsync_file(path: Path) -> None:
    with path.open("rb") as input_file:
        os.fsync(input_file.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _replay_storage_bytes(replay_buffer: Any) -> int:
    total = 0
    seen: set[int] = set()
    for value in vars(replay_buffer).values():
        if not isinstance(value, np.ndarray) or id(value) in seen:
            continue
        seen.add(id(value))
        total += int(value.nbytes)
    return total


def _touch_fsynced(path: Path, text: str = "") -> None:
    with path.open("w", encoding="utf-8") as output:
        output.write(text)
        output.flush()
        os.fsync(output.fileno())


def _preserving_artifact_path(base_path: Path, attempt_id: int) -> Path:
    """Return a new path without deleting or overwriting a prior attempt."""

    if not base_path.exists():
        return base_path
    suffix = base_path.suffix
    stem = base_path.name[: -len(suffix)] if suffix else base_path.name
    for retry_index in range(1_000):
        retry_suffix = (
            ""
            if retry_index == 0
            else f"_retry{retry_index:03d}"
        )
        candidate = base_path.with_name(
            f"{stem}_attempt{int(attempt_id):04d}{retry_suffix}{suffix}"
        )
        if not candidate.exists():
            return candidate
    raise FileExistsError(
        f"Unable to allocate a preserving artifact path for {base_path}"
    )


def _load_torch_payload(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def model_optimizer_counters(model: Any) -> dict[str, int]:
    if getattr(model, "architecture_version", None) == (
        "dsrl_na_rfs_hier_three_critic_v1"
    ):
        names = (
            "qa_base_optimizer_steps",
            "qa_joint_optimizer_steps",
            "qw_base_optimizer_steps",
            "noise_actor_optimizer_steps",
            "alpha_optimizer_steps",
            "residual_actor_optimizer_steps",
            "qa_base_target_updates",
            "qa_joint_target_updates",
            "residual_target_updates",
            "noise_policy_version",
            "residual_policy_version",
            "qa_joint_generation",
            "qa_joint_optimizer_steps_since_clone",
            "hierarchy_train_calls",
            "base_block_skips",
            "joint_block_skips",
        )
    else:
        names = (
            "action_critic_optimizer_steps",
            "modulation_critic_optimizer_steps",
            "noise_actor_optimizer_steps",
            "residual_actor_optimizer_steps",
            "hierarchy_train_calls",
        )
    return {
        name: int(getattr(model, name, 0))
        for name in names
    }


def model_hierarchy_state(model: Any) -> dict[str, Any] | None:
    if getattr(model, "architecture_version", None) != (
        "dsrl_na_rfs_hier_three_critic_v1"
    ):
        return None
    schedule = model.hierarchy_schedule
    return {
        "architecture_version": model.architecture_version,
        "replay_schema_version": model.replay_schema_version,
        "schedule_profile": schedule.profile_name,
        "phase_b_steps": int(schedule.phase_b_steps),
        "phase_r_steps": int(schedule.phase_r_steps),
        "phase_j_steps": int(schedule.phase_j_steps),
        "phase_j_enabled": bool(schedule.phase_j_enabled),
        "beta_ramp_steps": int(schedule.beta_ramp_steps),
        "beta_target": float(schedule.beta_target),
        "base_lane_probability": float(schedule.base_lane_probability),
        "requested_optimizer_steps": {
            key: int(value)
            for key, value in model.requested_optimizer_steps.items()
        },
        "environment_discontinuity_count": int(
            model.environment_discontinuity_count
        ),
    }


def run_binding_from_manifest(manifest: Mapping[str, Any]) -> dict[str, str]:
    initial_replay = manifest.get("initial_replay_hash")
    if not isinstance(initial_replay, Mapping):
        raise ValueError("Run manifest is missing initial_replay_hash")
    binding = {
        "run_id": str(manifest.get("run_id", "")),
        "config_contract_sha256": str(
            manifest.get("config_contract_sha256", "")
        ),
        "source_state_sha256": str(
            manifest.get("source_state_sha256", "")
        ),
        "prefill_semantic_hash": str(
            manifest.get("prefill_semantic_hash", "")
        ),
        "initial_replay_semantic_hash": str(
            initial_replay.get("semantic_hash", "")
        ),
    }
    for key, value in binding.items():
        if not value:
            raise ValueError(f"Run manifest is missing resume binding {key}")
    return binding


def validate_optimizer_counter_invariants(model: Any, algorithm: str) -> None:
    counters = model_optimizer_counters(model)
    for name, value in counters.items():
        if type(getattr(model, name, None)) is not int or value < 0:
            raise ValueError(f"Invalid optimizer counter {name}={value}")
    calls = counters["hierarchy_train_calls"]
    train_frequency = getattr(model, "train_freq", None)
    frequency = int(getattr(train_frequency, "frequency", -1))
    unit = str(getattr(train_frequency, "unit", "")).lower()
    n_envs = int(getattr(model, "n_envs", -1))
    if frequency <= 0 or n_envs <= 0 or "step" not in unit:
        raise ValueError(
            "P6 counter validation requires a positive step-based train_freq "
            f"and n_envs, got frequency={frequency}, unit={unit!r}, "
            f"n_envs={n_envs}"
        )
    safe_boundary_chunks = frequency * n_envs
    num_timesteps = int(getattr(model, "num_timesteps", -1))
    if num_timesteps < 0 or num_timesteps % safe_boundary_chunks != 0:
        raise ValueError(
            "P6 num_timesteps is not an exact training boundary: "
            f"{num_timesteps} % {safe_boundary_chunks}"
        )
    expected_calls = num_timesteps // safe_boundary_chunks
    if calls != expected_calls:
        raise ValueError(
            "P6 train-call/chunk invariant failed: "
            f"{calls} != {num_timesteps}/{safe_boundary_chunks}"
        )
    updates = getattr(model, "_n_updates", None)
    if type(updates) is not int or updates < 0:
        raise ValueError(f"Invalid _n_updates={updates}")
    if int(getattr(model, "gradient_steps", -1)) != 20:
        raise ValueError("P6 gradient_steps invariant requires 20")
    if int(getattr(model, "noise_critic_grad_steps", -1)) != 10:
        raise ValueError("P6 noise_critic_grad_steps invariant requires 10")

    if algorithm == "dsrl_na_rfs_hier":
        if getattr(model, "architecture_version", None) != (
            "dsrl_na_rfs_hier_three_critic_v1"
        ):
            raise ValueError("Hierarchy architecture version mismatch")
        if int(getattr(model, "actor_gradient_steps", -2)) != -1:
            raise ValueError("Hierarchy actor_gradient_steps must remain -1")
        required = {
            "qa_base_optimizer_steps",
            "qa_joint_optimizer_steps",
            "qw_base_optimizer_steps",
            "noise_actor_optimizer_steps",
            "alpha_optimizer_steps",
            "residual_actor_optimizer_steps",
            "qa_base_target_updates",
            "qa_joint_target_updates",
            "residual_target_updates",
            "noise_policy_version",
            "residual_policy_version",
            "qa_joint_generation",
            "qa_joint_optimizer_steps_since_clone",
            "hierarchy_train_calls",
            "base_block_skips",
            "joint_block_skips",
        }
        if set(counters) != required:
            raise ValueError("Three-critic counter schema is incomplete")
        interval = int(getattr(model, "target_update_interval", 0))
        if interval <= 0:
            raise ValueError("Hierarchy target_update_interval must be positive")
        expected_base_targets = (
            0
            if counters["qa_base_optimizer_steps"] == 0
            else (counters["qa_base_optimizer_steps"] - 1) // interval + 1
        )
        expected_joint_targets = (
            0
            if counters["qa_joint_optimizer_steps"] == 0
            else (counters["qa_joint_optimizer_steps"] - 1) // interval + 1
        )
        if counters["qa_base_target_updates"] != expected_base_targets:
            raise ValueError("QA_base target cadence differs from optimizer steps")
        if counters["qa_joint_target_updates"] != expected_joint_targets:
            raise ValueError("QA_joint target cadence differs from optimizer steps")
        if counters["residual_target_updates"] != counters["residual_actor_optimizer_steps"]:
            raise ValueError("Residual target cadence differs from residual optimizer")
        if counters["noise_policy_version"] != counters["noise_actor_optimizer_steps"]:
            raise ValueError("Noise policy version differs from successful updates")
        if counters["residual_policy_version"] != counters["residual_actor_optimizer_steps"]:
            raise ValueError("Residual policy version differs from successful updates")
        if counters["qa_joint_optimizer_steps_since_clone"] > counters["qa_joint_optimizer_steps"]:
            raise ValueError("QA_joint since-clone counter exceeds lifetime count")
        requested = getattr(model, "requested_optimizer_steps", None)
        if not isinstance(requested, Mapping):
            raise ValueError("Hierarchy requested optimizer counters are missing")
        realized = {
            "qa_base": counters["qa_base_optimizer_steps"],
            "qa_joint": counters["qa_joint_optimizer_steps"],
            "qw_base": counters["qw_base_optimizer_steps"],
            "noise_actor": counters["noise_actor_optimizer_steps"],
            "alpha": counters["alpha_optimizer_steps"],
            "residual_actor": counters["residual_actor_optimizer_steps"],
        }
        for name, value in realized.items():
            if int(requested.get(name, -1)) < value:
                raise ValueError(f"Realized {name} steps exceed requested steps")
        if updates != sum(int(value) for value in requested.values()):
            raise ValueError("Hierarchy _n_updates differs from requested updates")
        if getattr(model, "_pending_rollout_metadata", None) is not None:
            raise ValueError("Checkpoint boundary contains pending hierarchy metadata")
        return
    if algorithm == "dsrl_na_rfs_hier_frozen_noise":
        raise ValueError(
            "dsrl_na_rfs_hier_frozen_noise is a deprecated diagnostic graph, "
            "not three-critic Core V1"
        )
    elif algorithm == "dsrl_na_control":
        if int(getattr(model, "actor_gradient_steps", -2)) != -1:
            raise ValueError("Control actor_gradient_steps must remain -1")
        expected = {
            "action_critic_optimizer_steps": 20 * calls,
            "modulation_critic_optimizer_steps": 10 * calls,
            "noise_actor_optimizer_steps": 20 * calls,
            "residual_actor_optimizer_steps": 0,
            "hierarchy_train_calls": calls,
        }
    else:
        raise ValueError(f"Unsupported counter-invariant algorithm {algorithm!r}")
    if counters != expected:
        raise ValueError(
            f"P6 optimizer counter invariant failed: {counters} != {expected}"
        )
    if updates != 20 * calls:
        raise ValueError(
            f"P6 _n_updates invariant failed: {updates} != {20 * calls}"
        )


def initial_runtime_state(
    *,
    target_chunk_budget: int,
    action_chunk: int,
    prefill_metadata: Mapping[str, Any],
    replay_hash: Mapping[str, Any],
    online_eval_interval: int,
    model_checkpoint_interval: int,
    replay_checkpoint_interval: int,
    run_binding: Mapping[str, str],
) -> dict[str, Any]:
    if min(
        target_chunk_budget,
        action_chunk,
        online_eval_interval,
        model_checkpoint_interval,
        replay_checkpoint_interval,
    ) <= 0:
        raise ValueError("P6 budgets and cadences must be positive")
    binding = {key: str(run_binding.get(key, "")) for key in RUN_BINDING_KEYS}
    for key, value in binding.items():
        if not value:
            raise ValueError(f"Missing runtime resume binding {key}")
    if str(prefill_metadata["semantic_hash"]) != binding["prefill_semantic_hash"]:
        raise ValueError("Runtime prefill semantic hash differs from run binding")
    if (
        str(replay_hash["semantic_hash"])
        != binding["initial_replay_semantic_hash"]
    ):
        raise ValueError("Runtime initial replay hash differs from run binding")
    return {
        "format_version": 1,
        "run_binding": binding,
        "target_chunk_budget": int(target_chunk_budget),
        "action_chunk": int(action_chunk),
        "training_counters": PrimitiveCounters().to_dict(),
        "prefill_counters": dict(prefill_metadata["primitive_counters"]),
        "prefill_semantic_hash": str(prefill_metadata["semantic_hash"]),
        "initial_replay_hash": dict(replay_hash),
        "next_online_eval_chunk": 0,
        "next_model_checkpoint_chunk": int(model_checkpoint_interval),
        "next_replay_checkpoint_chunk": int(replay_checkpoint_interval),
        "online_eval_interval": int(online_eval_interval),
        "model_checkpoint_interval": int(model_checkpoint_interval),
        "replay_checkpoint_interval": int(replay_checkpoint_interval),
        "last_online_eval_chunk": None,
        "last_model_checkpoint_chunk": None,
        "last_replay_checkpoint_chunk": None,
        "resume_count": 0,
        "environment_reset_discontinuities": 0,
    }


def validate_runtime_state(state: Mapping[str, Any]) -> PrimitiveCounters:
    counters = PrimitiveCounters.from_dict(state["training_counters"])
    counters.validate(action_chunk=int(state["action_chunk"]))
    if counters.chunk_transitions > int(state["target_chunk_budget"]):
        raise ValueError("Runtime counters exceed the target chunk budget")
    for cadence_name in (
        "online_eval_interval",
        "model_checkpoint_interval",
        "replay_checkpoint_interval",
    ):
        if int(state[cadence_name]) <= 0:
            raise ValueError(f"{cadence_name} must be positive")
    binding = state.get("run_binding")
    if not isinstance(binding, Mapping):
        raise ValueError("Runtime state is missing run_binding")
    for key in RUN_BINDING_KEYS:
        if not str(binding.get(key, "")):
            raise ValueError(f"Runtime state is missing run binding {key}")
    return counters


class P6CheckpointManager:
    def __init__(
        self,
        *,
        run_directory: Path,
        algorithm: str,
        manifest_path: Path,
        runtime_state: dict[str, Any],
        provenance: Mapping[str, Any],
        evaluation_function: Callable[[int, PrimitiveCounters], Mapping[str, Any]],
    ) -> None:
        self.run_directory = run_directory
        self.algorithm = algorithm
        self.manifest_path = manifest_path
        self.runtime_state = runtime_state
        self.provenance = {
            key: provenance[key]
            for key in (
                "init_checkpoint_sha256",
                "frozen_ddim_sha256",
                "normalization_sha256",
            )
        }
        self.run_binding = run_binding_from_manifest(provenance)
        self.evaluation_function = evaluation_function
        self.run_directory.mkdir(parents=True, exist_ok=True)
        (self.run_directory / "checkpoints").mkdir(exist_ok=True)
        (self.run_directory / "resume").mkdir(exist_ok=True)
        (self.run_directory / "evaluations").mkdir(exist_ok=True)
        validate_runtime_state(self.runtime_state)

    def _update_run_manifest(self, updates: Mapping[str, Any]) -> None:
        with self.manifest_path.open("r", encoding="utf-8") as input_file:
            manifest = json.load(input_file)
        manifest.update(dict(updates))
        atomic_write_json(self.manifest_path, manifest)

    def _current_attempt_id(self) -> int:
        with self.manifest_path.open("r", encoding="utf-8") as input_file:
            manifest = json.load(input_file)
        attempt_id = int(manifest.get("current_attempt_id", 0))
        if attempt_id < 0:
            raise ValueError(f"Invalid current_attempt_id={attempt_id}")
        return attempt_id

    def save_model_snapshot(self, model: Any, chunk: int, *, final: bool = False) -> Path:
        validate_optimizer_counter_invariants(model, self.algorithm)
        filename = "final_model.zip" if final else f"model_{chunk:012d}.zip"
        canonical_path = self.run_directory / "checkpoints" / filename
        if final and canonical_path.exists():
            raise FileExistsError(
                f"Refusing to overwrite model checkpoint {canonical_path}"
            )
        path = (
            canonical_path
            if final
            else _preserving_artifact_path(
                canonical_path,
                self._current_attempt_id(),
            )
        )
        temporary_path = path.with_suffix(".tmp.zip")
        model.save(temporary_path)
        if not temporary_path.is_file():
            raise RuntimeError(f"SB3 did not create checkpoint {temporary_path}")
        _fsync_file(temporary_path)
        os.replace(temporary_path, path)
        _fsync_directory(path.parent)
        checkpoint_hash = sha256_file(path)
        self.runtime_state["last_model_checkpoint_chunk"] = int(chunk)
        self._update_run_manifest(
            {
                "latest_model_checkpoint": str(path),
                "latest_model_checkpoint_sha256": checkpoint_hash,
                "latest_model_checkpoint_chunk": int(chunk),
            }
        )
        return path

    def _bundle_manifest(
        self,
        *,
        model: Any,
        model_path: Path,
        replay_path: Path,
        runtime_path: Path,
        replay_hash: Mapping[str, Any],
        chunk: int,
    ) -> dict[str, Any]:
        return {
            "format_version": 1,
            "algorithm": self.algorithm,
            "chunk_transitions": int(chunk),
            "model_num_timesteps": int(model.num_timesteps),
            "model_sha256": sha256_file(model_path),
            "replay_sha256": sha256_file(replay_path),
            "runtime_state_sha256": sha256_file(runtime_path),
            "replay_semantic_hash": replay_hash["semantic_hash"],
            "replay_vector_steps": replay_hash["vector_steps"],
            "replay_offline_steps": replay_hash["offline_steps"],
            "optimizer_counters": model_optimizer_counters(model),
            "hierarchy_state": model_hierarchy_state(model),
            **self.run_binding,
            **self.provenance,
        }

    def save_resume_bundle(self, model: Any, chunk: int) -> Path:
        validate_optimizer_counter_invariants(model, self.algorithm)
        if int(model.num_timesteps) != int(chunk):
            raise ValueError(
                f"Model timestep {model.num_timesteps} != safe boundary {chunk}"
            )
        counters = validate_runtime_state(self.runtime_state)
        if counters.chunk_transitions != chunk:
            raise ValueError(
                "Runtime/model chunk mismatch: "
                f"{counters.chunk_transitions} != {chunk}"
            )
        canonical_directory = (
            self.run_directory / "resume" / f"chunk_{chunk:012d}"
        )
        final_directory = _preserving_artifact_path(
            canonical_directory,
            self._current_attempt_id(),
        )
        replay_storage_bytes = _replay_storage_bytes(model.replay_buffer)
        disk_free_bytes = shutil.disk_usage(self.run_directory).free
        minimum_free_bytes = (
            int(np.ceil(1.25 * replay_storage_bytes))
            + 512 * 1024 * 1024
        )
        if disk_free_bytes < minimum_free_bytes:
            raise OSError(
                "Insufficient free space for an atomic replay bundle: "
                f"free={disk_free_bytes}, required={minimum_free_bytes}"
            )
        temporary_directory = (
            self.run_directory
            / "resume"
            / f".tmp_{final_directory.name}_{os.getpid()}"
        )
        if temporary_directory.exists():
            raise FileExistsError(f"Temporary bundle already exists: {temporary_directory}")
        temporary_directory.mkdir(parents=False)
        previous_last_replay_chunk = self.runtime_state[
            "last_replay_checkpoint_chunk"
        ]
        self.runtime_state["last_replay_checkpoint_chunk"] = int(chunk)
        try:
            model_path = temporary_directory / "model.zip"
            replay_path = temporary_directory / "replay_buffer.pkl"
            runtime_path = temporary_directory / "runtime_state.pt"
            model.save(model_path)
            model.save_replay_buffer(replay_path)
            replay_hash = hash_replay_for_resume(model.replay_buffer)
            expected_replay_chunks = (
                int(self.runtime_state["prefill_counters"]["chunk_transitions"])
                + counters.chunk_transitions
            )
            actual_replay_chunks = (
                int(replay_hash["vector_steps"])
                * int(model.replay_buffer.n_envs)
            )
            if actual_replay_chunks != expected_replay_chunks:
                raise ValueError(
                    "Replay/runtime transition mismatch: "
                    f"{actual_replay_chunks} != {expected_replay_chunks}"
                )
            runtime_payload = {
                "runtime_state": dict(self.runtime_state),
                "run_binding": dict(self.run_binding),
                "rng_state": capture_rng_state(),
                "optimizer_counters": model_optimizer_counters(model),
                "hierarchy_state": model_hierarchy_state(model),
            }
            torch.save(runtime_payload, runtime_path)
            for path in (model_path, replay_path, runtime_path):
                if not path.is_file():
                    raise RuntimeError(f"Missing resume payload {path}")
                _fsync_file(path)
            bundle_manifest = self._bundle_manifest(
                model=model,
                model_path=model_path,
                replay_path=replay_path,
                runtime_path=runtime_path,
                replay_hash=replay_hash,
                chunk=chunk,
            )
            atomic_write_json(
                temporary_directory / "bundle_manifest.json",
                {
                    **bundle_manifest,
                    "replay_storage_bytes": replay_storage_bytes,
                    "disk_free_bytes_before_bundle": disk_free_bytes,
                },
            )
            _touch_fsynced(temporary_directory / "COMPLETE", "complete\n")
            os.replace(temporary_directory, final_directory)
            _fsync_directory(final_directory.parent)
        except BaseException:
            self.runtime_state["last_replay_checkpoint_chunk"] = (
                previous_last_replay_chunk
            )
            shutil.rmtree(temporary_directory, ignore_errors=True)
            raise

        latest_payload = {
            "bundle_path": str(final_directory),
            "chunk_transitions": int(chunk),
            "bundle_manifest_sha256": sha256_file(
                final_directory / "bundle_manifest.json"
            ),
        }
        atomic_write_json(
            self.run_directory / "resume" / "latest.json",
            latest_payload,
        )
        self._update_run_manifest(
            {
                "latest_resume_bundle": str(final_directory),
                "latest_resume_bundle_chunk": int(chunk),
                "latest_resume_bundle_manifest_sha256": (
                    latest_payload["bundle_manifest_sha256"]
                ),
            }
        )
        return final_directory

    def run_evaluation(self, model: Any, chunk: int) -> Mapping[str, Any]:
        counters = validate_runtime_state(self.runtime_state)
        result = self.evaluation_function(chunk, counters)
        self.runtime_state["last_online_eval_chunk"] = int(chunk)
        updates = {
            "latest_online_evaluation_chunk": int(chunk),
            "latest_online_evaluation_summary": dict(result["summary"]),
        }
        if "mode_summaries" in result:
            updates["latest_online_evaluation_mode_summaries"] = dict(
                result["mode_summaries"]
            )
        self._update_run_manifest(updates)
        return result

    def _flush_training_logger(self, model: Any, chunk: int) -> None:
        logger = getattr(model, "logger", None)
        if logger is None:
            return
        counters = validate_runtime_state(self.runtime_state)
        values = {
            "p6/chunk_transitions": counters.chunk_transitions,
            "p6/nominal_primitive_steps": counters.nominal_primitive_steps,
            "p6/actual_primitive_env_steps": counters.actual_primitive_env_steps,
            **{
                f"train/{name}": value
                for name, value in model_optimizer_counters(model).items()
            },
        }
        for name, value in values.items():
            logger.record(name, value)
        logger.dump(step=int(chunk))

    def service_safe_boundary(
        self,
        model: Any,
        *,
        interrupt_after_service: bool = False,
        skip_online_eval: bool = False,
    ) -> None:
        current = int(model.num_timesteps)
        counters = validate_runtime_state(self.runtime_state)
        if counters.chunk_transitions != current:
            raise ValueError(
                f"Safe-boundary counter mismatch: {counters.chunk_transitions} != {current}"
            )
        service_due = interrupt_after_service or any(
            current >= int(self.runtime_state[field])
            for field in (
                "next_online_eval_chunk",
                "next_model_checkpoint_chunk",
                "next_replay_checkpoint_chunk",
            )
        )
        if service_due:
            self._flush_training_logger(model, current)

        if not skip_online_eval:
            while current >= int(self.runtime_state["next_online_eval_chunk"]):
                milestone = int(self.runtime_state["next_online_eval_chunk"])
                if current != milestone:
                    raise ValueError(
                        f"Online evaluation cadence skipped {milestone} at {current}"
                    )
                self.runtime_state["next_online_eval_chunk"] = (
                    milestone + int(self.runtime_state["online_eval_interval"])
                )
                self.run_evaluation(model, milestone)

        while current >= int(self.runtime_state["next_model_checkpoint_chunk"]):
            milestone = int(self.runtime_state["next_model_checkpoint_chunk"])
            if current != milestone:
                raise ValueError(
                    f"Model checkpoint cadence skipped {milestone} at {current}"
                )
            self.runtime_state["next_model_checkpoint_chunk"] = (
                milestone + int(self.runtime_state["model_checkpoint_interval"])
            )
            self.save_model_snapshot(model, milestone)

        while current >= int(self.runtime_state["next_replay_checkpoint_chunk"]):
            milestone = int(self.runtime_state["next_replay_checkpoint_chunk"])
            if current != milestone:
                raise ValueError(
                    f"Replay checkpoint cadence skipped {milestone} at {current}"
                )
            self.runtime_state["next_replay_checkpoint_chunk"] = (
                milestone + int(self.runtime_state["replay_checkpoint_interval"])
            )
            self.save_resume_bundle(model, milestone)

        if interrupt_after_service:
            if self.runtime_state["last_replay_checkpoint_chunk"] != current:
                self.save_resume_bundle(model, current)
            raise P6IntentionalInterruption(
                f"Intentional interruption after safe bundle at {current} chunks"
            )

    def flush_final(self, model: Any) -> Path:
        validate_optimizer_counter_invariants(model, self.algorithm)
        current = int(model.num_timesteps)
        self.service_safe_boundary(model, skip_online_eval=True)
        final_model = self.run_directory / "checkpoints" / "final_model.zip"
        if final_model.exists():
            raise FileExistsError(
                "Refusing to trust a pre-existing unauthenticated final model: "
                f"{final_model}"
            )
        self.save_model_snapshot(model, current, final=True)
        if self.runtime_state["last_replay_checkpoint_chunk"] != current:
            self.save_resume_bundle(model, current)
        self._update_run_manifest(
            {
                "training_status": "complete",
                "final_evaluation_status": "pending",
                "run_status": "finalizing",
                "remaining_chunk_transitions": 0,
                "final_chunk_transitions": current,
                "final_nominal_primitive_steps": (
                    self.runtime_state["training_counters"][
                        "nominal_primitive_steps"
                    ]
                ),
                "final_actual_primitive_env_steps": (
                    self.runtime_state["training_counters"][
                        "actual_primitive_env_steps"
                    ]
                ),
                "final_optimizer_counters": model_optimizer_counters(model),
            }
        )
        return final_model


class P6TrainingCallback(BaseCallback):
    def __init__(
        self,
        *,
        manager: P6CheckpointManager,
        action_chunk: int,
        stop_after_chunk_transitions: int | None = None,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        self.manager = manager
        self.action_chunk = int(action_chunk)
        self.stop_after_chunk_transitions = (
            None
            if stop_after_chunk_transitions is None
            else int(stop_after_chunk_transitions)
        )
        self._interrupt_pending = False

    def _on_step(self) -> bool:
        infos = self.locals["infos"]
        dones = self.locals["dones"]
        counters = PrimitiveCounters.from_dict(
            self.manager.runtime_state["training_counters"]
        )
        counters.update(infos, dones, action_chunk=self.action_chunk)
        self.manager.runtime_state["training_counters"] = counters.to_dict()
        logger = getattr(self.model, "logger", None)
        if logger is not None:
            logger.record("p6/chunk_transitions", counters.chunk_transitions)
            logger.record(
                "p6/nominal_primitive_steps",
                counters.nominal_primitive_steps,
            )
            logger.record(
                "p6/actual_primitive_env_steps",
                counters.actual_primitive_env_steps,
            )
        if counters.chunk_transitions != int(self.model.num_timesteps):
            raise ValueError(
                "Training counter/model timestep mismatch: "
                f"{counters.chunk_transitions} != {self.model.num_timesteps}"
            )
        if (
            self.stop_after_chunk_transitions is not None
            and counters.chunk_transitions >= self.stop_after_chunk_transitions
        ):
            if counters.chunk_transitions != self.stop_after_chunk_transitions:
                raise ValueError("Training overshot the intentional interruption boundary")
            self._interrupt_pending = True
        return True

    def _on_rollout_start(self) -> None:
        self.manager.service_safe_boundary(
            self.model,
            interrupt_after_service=self._interrupt_pending,
        )

    def _on_training_end(self) -> None:
        self.manager.flush_final(self.model)


def load_resume_payload(
    bundle_directory: Path,
    *,
    algorithm: str,
    expected_run_manifest: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not (bundle_directory / "COMPLETE").is_file():
        raise ValueError("Resume bundle is missing COMPLETE")
    manifest_path = bundle_directory / "bundle_manifest.json"
    with manifest_path.open("r", encoding="utf-8") as input_file:
        manifest = json.load(input_file)
    if manifest.get("algorithm") != algorithm:
        raise ValueError("Resume bundle algorithm mismatch")
    for key in (
        "init_checkpoint_sha256",
        "frozen_ddim_sha256",
        "normalization_sha256",
    ):
        if manifest.get(key) != expected_run_manifest[key]:
            raise ValueError(f"Resume bundle provenance mismatch for {key}")
    expected_binding = run_binding_from_manifest(expected_run_manifest)
    for key, value in expected_binding.items():
        if manifest.get(key) != value:
            raise ValueError(f"Resume bundle binding mismatch for {key}")
    file_hashes = {
        "model_sha256": sha256_file(bundle_directory / "model.zip"),
        "replay_sha256": sha256_file(bundle_directory / "replay_buffer.pkl"),
        "runtime_state_sha256": sha256_file(
            bundle_directory / "runtime_state.pt"
        ),
    }
    for key, value in file_hashes.items():
        if manifest.get(key) != value:
            raise ValueError(f"Resume bundle {key} mismatch")
    payload = _load_torch_payload(bundle_directory / "runtime_state.pt")
    validate_runtime_state(payload["runtime_state"])
    if payload.get("run_binding") != expected_binding:
        raise ValueError("Resume runtime payload run_binding mismatch")
    if payload["runtime_state"].get("run_binding") != expected_binding:
        raise ValueError("Resume runtime-state run_binding mismatch")
    if payload["optimizer_counters"] != manifest["optimizer_counters"]:
        raise ValueError("Resume optimizer-counter metadata mismatch")
    if payload.get("hierarchy_state") != manifest.get("hierarchy_state"):
        raise ValueError("Resume hierarchy-state metadata mismatch")
    return manifest, payload


def validate_loaded_resume(
    *,
    model: Any,
    bundle_directory: Path,
    bundle_manifest: Mapping[str, Any],
    runtime_payload: Mapping[str, Any],
) -> None:
    validate_optimizer_counter_invariants(
        model,
        str(bundle_manifest["algorithm"]),
    )
    if int(model.num_timesteps) != int(bundle_manifest["model_num_timesteps"]):
        raise ValueError("Loaded model timestep differs from bundle")
    replay_hash = hash_replay_for_resume(model.replay_buffer)
    if replay_hash["semantic_hash"] != bundle_manifest["replay_semantic_hash"]:
        raise ValueError("Loaded replay semantic hash differs from bundle")
    if replay_hash["vector_steps"] != int(bundle_manifest["replay_vector_steps"]):
        raise ValueError("Loaded replay position differs from bundle")
    if model_optimizer_counters(model) != bundle_manifest["optimizer_counters"]:
        raise ValueError("Loaded model optimizer counters differ from bundle")
    if model_hierarchy_state(model) != bundle_manifest.get("hierarchy_state"):
        raise ValueError("Loaded hierarchy state differs from bundle")
    offline_hash = hash_replay_offline_for_resume(model.replay_buffer)
    expected_offline_hash = bundle_manifest["initial_replay_semantic_hash"]
    if offline_hash["semantic_hash"] != expected_offline_hash:
        raise ValueError("Loaded replay offline prefix differs from initial replay")
    if (
        runtime_payload["runtime_state"]["training_counters"][
            "chunk_transitions"
        ]
        != model.num_timesteps
    ):
        raise ValueError("Loaded runtime counter differs from model timestep")
