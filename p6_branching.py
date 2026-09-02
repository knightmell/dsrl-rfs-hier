"""Audited helpers for schedule-only P6 forks at safe bundle boundaries."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping

from stable_baselines3.common.save_util import load_from_zip_file, save_to_zip_file
from stable_baselines3.dsrl.hierarchy_schedule import HierarchySchedule


MODEL_SCHEDULE_KEYS = (
    "_schedule_overrides",
    "hierarchy_schedule",
    "schedule_profile",
    "tensorboard_log",
)

ALLOWED_CONFIG_CONTRACT_CHANGE_PREFIXES = (
    "training.hierarchy.",
)
ALLOWED_CONFIG_CONTRACT_CHANGE_PATHS = {
    "replay_checkpoint_interval",
    "run_name",
    "source_state_sha256",
}


def schedule_overrides(schedule: HierarchySchedule) -> dict[str, Any]:
    return {
        "phase_b_steps": int(schedule.phase_b_steps),
        "phase_r_steps": int(schedule.phase_r_steps),
        "phase_j_steps": int(schedule.phase_j_steps),
        "phase_j_enabled": bool(schedule.phase_j_enabled),
        "beta_ramp_steps": int(schedule.beta_ramp_steps),
        "beta_target": float(schedule.beta_target),
        "base_lane_probability": float(schedule.base_lane_probability),
        "beta_hold_steps": int(schedule.beta_hold_steps),
        "beta_floor": float(schedule.beta_floor),
    }


def rewrite_model_schedule_archive(
    source_path: Path,
    destination_path: Path,
    schedule: HierarchySchedule,
    *,
    tensorboard_log: str,
) -> dict[str, Any]:
    """Rewrite only serialized schedule data; preserve all learned tensors."""

    source_path = Path(source_path)
    destination_path = Path(destination_path)
    if destination_path.exists():
        raise FileExistsError(destination_path)
    data, params, pytorch_variables = load_from_zip_file(source_path, device="cpu")
    if data is None or params is None:
        raise ValueError("Parent model archive is missing data or parameters")
    missing = [key for key in MODEL_SCHEDULE_KEYS if key not in data]
    if missing:
        raise ValueError(f"Parent model archive is missing schedule keys: {missing}")
    if int(data.get("num_timesteps", -1)) < 0:
        raise ValueError("Parent model archive has no valid num_timesteps")

    branched_data = copy.deepcopy(data)
    branched_data["schedule_profile"] = schedule.profile_name
    branched_data["_schedule_overrides"] = schedule_overrides(schedule)
    branched_data["hierarchy_schedule"] = schedule
    branched_data["tensorboard_log"] = str(tensorboard_log)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    save_to_zip_file(
        destination_path,
        branched_data,
        params,
        pytorch_variables,
    )
    return {
        "changed_model_data_keys": sorted(MODEL_SCHEDULE_KEYS),
        "model_num_timesteps": int(data["num_timesteps"]),
        "source_schedule_profile": str(data["schedule_profile"]),
        "destination_schedule_profile": schedule.profile_name,
    }


def _next_boundary(current: int, interval: int) -> int:
    if interval <= 0:
        raise ValueError("Checkpoint/evaluation intervals must be positive")
    return ((int(current) // int(interval)) + 1) * int(interval)


def branch_runtime_payload(
    parent_payload: Mapping[str, Any],
    *,
    new_binding: Mapping[str, str],
    new_hierarchy_state: Mapping[str, Any],
    online_eval_interval: int,
    model_checkpoint_interval: int,
    replay_checkpoint_interval: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Rebind runtime metadata without changing learner, optimizer, or RNG state."""

    child = copy.deepcopy(dict(parent_payload))
    runtime = child.get("runtime_state")
    if not isinstance(runtime, dict):
        raise ValueError("Parent payload is missing runtime_state")
    counters = runtime.get("training_counters")
    if not isinstance(counters, Mapping):
        raise ValueError("Parent runtime is missing training_counters")
    current = int(counters.get("chunk_transitions", -1))
    if current < 0:
        raise ValueError("Parent runtime has invalid chunk_transitions")

    binding = dict(new_binding)
    child["run_binding"] = copy.deepcopy(binding)
    runtime["run_binding"] = copy.deepcopy(binding)
    child["hierarchy_state"] = copy.deepcopy(dict(new_hierarchy_state))

    cadence = {
        "online_eval": int(online_eval_interval),
        "model_checkpoint": int(model_checkpoint_interval),
        "replay_checkpoint": int(replay_checkpoint_interval),
    }
    for name, interval in cadence.items():
        runtime[f"{name}_interval"] = interval
        runtime[f"next_{name}_chunk"] = _next_boundary(current, interval)

    report = {
        "branch_chunk_transitions": current,
        "next_online_eval_chunk": runtime["next_online_eval_chunk"],
        "next_model_checkpoint_chunk": runtime["next_model_checkpoint_chunk"],
        "next_replay_checkpoint_chunk": runtime["next_replay_checkpoint_chunk"],
        "runtime_learning_state_modified": False,
        "rng_state_modified": False,
        "optimizer_state_modified": False,
    }
    return child, report


def _mapping_differences(
    left: Any,
    right: Any,
    prefix: str = "",
) -> list[str]:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        paths: list[str] = []
        for key in sorted(set(left) | set(right)):
            path = f"{prefix}.{key}" if prefix else str(key)
            paths.extend(
                _mapping_differences(
                    left.get(key, object()),
                    right.get(key, object()),
                    path,
                )
            )
        return paths
    return [] if left == right else [prefix]


def validate_schedule_branch_contract(
    parent_contract: Mapping[str, Any],
    child_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Reject any child-config change unrelated to schedule or artifact cadence."""

    changed = _mapping_differences(parent_contract, child_contract)
    illegal = [
        path
        for path in changed
        if path not in ALLOWED_CONFIG_CONTRACT_CHANGE_PATHS
        and not any(
            path.startswith(prefix)
            for prefix in ALLOWED_CONFIG_CONTRACT_CHANGE_PREFIXES
        )
    ]
    if illegal:
        raise ValueError(
            "Schedule branch changes non-branch config fields: "
            + ", ".join(illegal)
        )
    return {
        "changed_config_contract_paths": changed,
        "unchanged_contract_fields_verified": True,
    }


def build_branch_manifest(
    parent_manifest: Mapping[str, Any],
    child_static_manifest: Mapping[str, Any],
    *,
    parent_bundle_path: Path,
    child_bundle_path: Path,
    branch_chunk_transitions: int,
    bundle_manifest_sha256: str,
) -> dict[str, Any]:
    """Build a child manifest that inherits data provenance, not attempt history."""

    parent = copy.deepcopy(dict(parent_manifest))
    child = copy.deepcopy(parent)
    parent_prefill = {
        key: copy.deepcopy(value)
        for key, value in parent.items()
        if key.startswith("prefill_")
    }
    child.update(copy.deepcopy(dict(child_static_manifest)))
    child.update(parent_prefill)
    for key in (
        "current_attempt_id",
        "active_stop_after_chunk_transitions",
        "interruption_reason",
        "failure_type",
        "failure_message",
        "failure_traceback",
        "resume_source",
        "resume_count",
        "environment_reset_discontinuities",
        "latest_reset_boundary",
        "cleanup_errors",
    ):
        child.pop(key, None)
    branch_chunk = int(branch_chunk_transitions)
    chunk_budget = int(child["chunk_budget"])
    if branch_chunk < 0 or branch_chunk >= chunk_budget:
        raise ValueError("Branch chunk must lie inside the configured budget")
    child_bundle = Path(child_bundle_path).resolve()
    child.update(
        {
            "attempts": [],
            "run_status": "interrupted",
            "training_status": "branch_bundle_ready",
            "final_evaluation_status": "pending",
            "latest_resume_bundle": str(child_bundle),
            "latest_resume_bundle_chunk": branch_chunk,
            "latest_resume_bundle_manifest_sha256": str(
                bundle_manifest_sha256
            ),
            "remaining_chunk_transitions": chunk_budget - branch_chunk,
            "branch_lineage": {
                "branch_chunk_transitions": branch_chunk,
                "parent_bundle_path": str(Path(parent_bundle_path).resolve()),
                "parent_run_id": str(parent["run_id"]),
                "parent_config_contract_sha256": str(
                    parent["config_contract_sha256"]
                ),
                "parent_source_state_sha256": str(
                    parent["source_state_sha256"]
                ),
                "child_source_state_sha256": str(
                    child["source_state_sha256"]
                ),
                "learning_state_inherited": True,
                "replay_inherited": True,
                "optimizer_state_inherited": True,
                "rng_state_inherited": True,
            },
            "preflight_status": "branched_bundle_verified",
        }
    )
    return child
