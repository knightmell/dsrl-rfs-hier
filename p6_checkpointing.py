"""Safe-boundary P6 checkpoints, replay bundles, counters, and callbacks."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Callable, Mapping

import torch
from stable_baselines3.common.callbacks import BaseCallback

from p6_preflight import sha256_file
from p6_runtime import (
    PrimitiveCounters,
    atomic_write_json,
    capture_rng_state,
    hash_replay_prefix,
)


class P6IntentionalInterruption(RuntimeError):
    """Raised only after a certified safe resume bundle has been written."""


def _fsync_file(path: Path) -> None:
    with path.open("rb") as input_file:
        os.fsync(input_file.fileno())


def _touch_fsynced(path: Path, text: str = "") -> None:
    with path.open("w", encoding="utf-8") as output:
        output.write(text)
        output.flush()
        os.fsync(output.fileno())


def _load_torch_payload(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def model_optimizer_counters(model: Any) -> dict[str, int]:
    return {
        name: int(getattr(model, name, 0))
        for name in (
            "action_critic_optimizer_steps",
            "modulation_critic_optimizer_steps",
            "noise_actor_optimizer_steps",
            "residual_actor_optimizer_steps",
            "hierarchy_train_calls",
        )
    }


def initial_runtime_state(
    *,
    target_chunk_budget: int,
    action_chunk: int,
    prefill_metadata: Mapping[str, Any],
    replay_hash: Mapping[str, Any],
    online_eval_interval: int,
    model_checkpoint_interval: int,
    replay_checkpoint_interval: int,
) -> dict[str, Any]:
    if min(
        target_chunk_budget,
        action_chunk,
        online_eval_interval,
        model_checkpoint_interval,
        replay_checkpoint_interval,
    ) <= 0:
        raise ValueError("P6 budgets and cadences must be positive")
    return {
        "format_version": 1,
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

    def save_model_snapshot(self, model: Any, chunk: int, *, final: bool = False) -> Path:
        filename = "final_model.zip" if final else f"model_{chunk:012d}.zip"
        path = self.run_directory / "checkpoints" / filename
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite model checkpoint {path}")
        temporary_path = path.with_suffix(".tmp.zip")
        model.save(temporary_path)
        if not temporary_path.is_file():
            raise RuntimeError(f"SB3 did not create checkpoint {temporary_path}")
        _fsync_file(temporary_path)
        os.replace(temporary_path, path)
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
            **self.provenance,
        }

    def save_resume_bundle(self, model: Any, chunk: int) -> Path:
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
        final_directory = self.run_directory / "resume" / f"chunk_{chunk:012d}"
        if final_directory.exists():
            raise FileExistsError(f"Refusing to overwrite resume bundle {final_directory}")
        temporary_directory = (
            self.run_directory
            / "resume"
            / f".tmp_chunk_{chunk:012d}_{os.getpid()}"
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
            replay_hash = hash_replay_prefix(model.replay_buffer)
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
                "rng_state": capture_rng_state(),
                "optimizer_counters": model_optimizer_counters(model),
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
                bundle_manifest,
            )
            _touch_fsynced(temporary_directory / "COMPLETE", "complete\n")
            os.replace(temporary_directory, final_directory)
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
        self._update_run_manifest(
            {
                "latest_online_evaluation_chunk": int(chunk),
                "latest_online_evaluation_summary": dict(result["summary"]),
            }
        )
        return result

    def service_safe_boundary(
        self,
        model: Any,
        *,
        interrupt_after_service: bool = False,
    ) -> None:
        current = int(model.num_timesteps)
        counters = validate_runtime_state(self.runtime_state)
        if counters.chunk_transitions != current:
            raise ValueError(
                f"Safe-boundary counter mismatch: {counters.chunk_transitions} != {current}"
            )

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
        current = int(model.num_timesteps)
        self.service_safe_boundary(model)
        final_model = self.run_directory / "checkpoints" / "final_model.zip"
        if not final_model.exists():
            self.save_model_snapshot(model, current, final=True)
        if self.runtime_state["last_replay_checkpoint_chunk"] != current:
            self.save_resume_bundle(model, current)
        self._update_run_manifest(
            {
                "training_status": "complete",
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
    expected_provenance: Mapping[str, Any],
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
        if manifest.get(key) != expected_provenance[key]:
            raise ValueError(f"Resume bundle provenance mismatch for {key}")
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
    if payload["optimizer_counters"] != manifest["optimizer_counters"]:
        raise ValueError("Resume optimizer-counter metadata mismatch")
    return manifest, payload


def validate_loaded_resume(
    *,
    model: Any,
    bundle_directory: Path,
    bundle_manifest: Mapping[str, Any],
    runtime_payload: Mapping[str, Any],
) -> None:
    if int(model.num_timesteps) != int(bundle_manifest["model_num_timesteps"]):
        raise ValueError("Loaded model timestep differs from bundle")
    replay_hash = hash_replay_prefix(model.replay_buffer)
    if replay_hash["semantic_hash"] != bundle_manifest["replay_semantic_hash"]:
        raise ValueError("Loaded replay semantic hash differs from bundle")
    if replay_hash["vector_steps"] != int(bundle_manifest["replay_vector_steps"]):
        raise ValueError("Loaded replay position differs from bundle")
    if model_optimizer_counters(model) != bundle_manifest["optimizer_counters"]:
        raise ValueError("Loaded model optimizer counters differ from bundle")
    if (
        runtime_payload["runtime_state"]["training_counters"][
            "chunk_transitions"
        ]
        != model.num_timesteps
    ):
        raise ValueError("Loaded runtime counter differs from model timestep")
