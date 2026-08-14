"""P6 run provenance, seed-plan, artifact, and execution-bound validation."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from stable_baselines3.dsrl.hierarchical_replay_buffer import (
    SCHEMA_VERSION as HIERARCHY_REPLAY_SCHEMA_VERSION,
)
from stable_baselines3.dsrl.hierarchical_rfs_dsrl import ARCHITECTURE_VERSION
from stable_baselines3.dsrl.hierarchy_schedule import (
    DEFAULT_UPDATE_PROFILES,
    make_hierarchy_schedule,
)


HIERARCHY_ALGORITHM = "dsrl_na_rfs_hier"
FROZEN_NOISE_ALGORITHM = "dsrl_na_rfs_hier_frozen_noise"
CONTROL_ALGORITHM = "dsrl_na_control"
HIERARCHY_ALGORITHMS = {HIERARCHY_ALGORITHM}
ACTION_CHUNK_TERMINATION = "early_break_on_done"
# Prefill sources.  warmstart_dsrl: the explicitly supplied legacy DSRL actor
# samples behavior (noise_log_prob_valid=True, CURRENT_ACTOR, version 0).
# fresh_frozen_ddim: an independent standard-Gaussian decoder prior is decoded
# through the Frozen DDIM (noise_log_prob_valid=False, GAUSSIAN_PRIOR,
# version -1).  Spec 5.3.
LEGACY_WARMSTART_PROFILE = "legacy_dsrl_warmstart_5m"
FRESH_FROZEN_PROFILES = (
    "fresh_frozen_ddim_5m",
    "fresh_frozen_ddim_2p5m",
    "fresh_frozen_ddim_2p5m_cotrain",
)
SUPPORTED_HIERARCHY_PROFILES = (LEGACY_WARMSTART_PROFILE, *FRESH_FROZEN_PROFILES)
PREFILL_SOURCE = "warmstart_dsrl"
FRESH_PREFILL_SOURCE = "fresh_frozen_ddim"
PREFILL_ACTION_POLICY = "shared_tagged_warmstart_dsrl_behavior"
FRESH_PREFILL_ACTION_POLICY = "fresh_frozen_ddim_gaussian_prior_behavior"
PREFILL_SOURCES = (PREFILL_SOURCE, FRESH_PREFILL_SOURCE)
HIERARCHY_PREFILL_RESIDUAL_MODE = (
    "exact_zero_residual_tagged_base_projection"
)
SAFE_RESUME_SEMANTICS = "reset_boundary_discontinuous"


def _required_string(value: Any, field_name: str) -> str:
    if value is None:
        raise ValueError(f"{field_name} must be explicitly configured")
    result = str(value).strip()
    if not result or result == "???":
        raise ValueError(f"{field_name} must be explicitly configured")
    return result


def _required_int(value: Any, field_name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be an integer, got bool")
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field_name} must be an integer") from error
    if result < minimum:
        raise ValueError(f"{field_name} must be >= {minimum}, got {result}")
    return result


def _optional_int(value: Any, field_name: str, *, minimum: int = 0) -> int | None:
    if value is None:
        return None
    return _required_int(value, field_name, minimum=minimum)


def _required_bool(value: Any, field_name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{field_name} must be an explicit boolean")
    return value


def _validate_sha256(value: Any, field_name: str) -> str:
    result = _required_string(value, field_name).lower()
    if re.fullmatch(r"[0-9a-f]{64}", result) is None:
        raise ValueError(f"{field_name} must be a lowercase 64-character SHA-256")
    return result


def _resolve_artifact_path(repo_root: Path, configured_path: Any, field_name: str) -> Path:
    path_string = _required_string(configured_path, field_name)
    path = Path(path_string).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{field_name} does not exist: {path}")
    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as artifact:
        for block in iter(lambda: artifact.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_mapping_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _verify_artifact(
    repo_root: Path,
    configured_path: Any,
    expected_sha256: Any,
    field_name: str,
) -> tuple[Path, str]:
    path = _resolve_artifact_path(repo_root, configured_path, field_name)
    expected = _validate_sha256(expected_sha256, f"{field_name}_sha256")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(
            f"{field_name} SHA-256 mismatch: expected {expected}, got {actual}"
        )
    return path, actual


def _git_output(repo_path: Path, *arguments: str) -> str:
    if not repo_path.is_dir():
        raise FileNotFoundError(f"Git repository path does not exist: {repo_path}")
    completed = subprocess.run(
        ["git", "-C", str(repo_path), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.rstrip("\n")


def git_state(repo_path: Path) -> dict[str, Any]:
    status_output = _git_output(
        repo_path,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    )
    return {
        "commit": _git_output(repo_path, "rev-parse", "HEAD"),
        "dirty": bool(status_output),
        "dirty_entries": status_output.splitlines() if status_output else [],
    }


@lru_cache(maxsize=None)
def git_source_fingerprint(repo_path: Path) -> dict[str, Any]:
    """Hash the exact tracked diff and every untracked file in a checkout."""

    tracked_diff = subprocess.run(
        ["git", "-C", str(repo_path), "diff", "--binary", "HEAD", "--"],
        check=True,
        capture_output=True,
    ).stdout
    untracked_output = subprocess.run(
        [
            "git",
            "-C",
            str(repo_path),
            "ls-files",
            "--others",
            "--exclude-standard",
            "-z",
        ],
        check=True,
        capture_output=True,
    ).stdout
    untracked_files: list[dict[str, str]] = []
    for encoded_path in filter(None, untracked_output.split(b"\0")):
        relative_path = encoded_path.decode("utf-8", errors="strict")
        absolute_path = repo_path / relative_path
        if absolute_path.is_symlink():
            content_hash = hashlib.sha256(
                os.readlink(absolute_path).encode("utf-8")
            ).hexdigest()
        elif absolute_path.is_file():
            content_hash = sha256_file(absolute_path)
        else:
            raise ValueError(
                f"Unsupported untracked source entry {absolute_path}"
            )
        untracked_files.append(
            {"path": relative_path, "sha256": content_hash}
        )
    payload = {
        "commit": _git_output(repo_path, "rev-parse", "HEAD"),
        "tracked_diff_sha256": hashlib.sha256(tracked_diff).hexdigest(),
        "untracked_files": untracked_files,
    }
    payload["source_state_sha256"] = canonical_mapping_sha256(payload)
    return payload


def _recorded_submodule_commit(repo_root: Path, submodule_path: str) -> str:
    output = _git_output(repo_root, "ls-tree", "HEAD", submodule_path)
    if not output:
        raise ValueError(
            f"Outer repository does not record submodule {submodule_path!r}"
        )
    metadata, recorded_path = output.split("\t", maxsplit=1)
    mode, object_type, commit = metadata.split()
    if (
        mode != "160000"
        or object_type != "commit"
        or recorded_path != submodule_path
    ):
        raise ValueError(f"Invalid gitlink for submodule {submodule_path!r}")
    return commit


def verified_submodule_state(
    repo_root: Path,
    submodule_path: str,
    checkout_path: Path,
) -> dict[str, Any]:
    state = git_state(checkout_path)
    recorded_commit = _recorded_submodule_commit(repo_root, submodule_path)
    if state["commit"] != recorded_commit:
        raise ValueError(
            f"Submodule {submodule_path} commit mismatch: outer records "
            f"{recorded_commit}, checkout is {state['commit']}"
        )
    state["recorded_commit"] = recorded_commit
    return state


def resolve_seed_plan(cfg: Any) -> dict[str, Any]:
    p6 = cfg.p6
    seed_plan = {
        "train_env_seed": _required_int(
            p6.train_env_seed,
            "p6.train_env_seed",
        ),
        "eval_env_seed": _required_int(
            p6.eval_env_seed,
            "p6.eval_env_seed",
        ),
        "prefill_env_seed": _required_int(
            p6.prefill_env_seed,
            "p6.prefill_env_seed",
        ),
        "prefill_policy_seed": _required_int(
            p6.prefill_policy_seed,
            "p6.prefill_policy_seed",
        ),
    }
    seed_values = list(seed_plan.values())
    if len(set(seed_values)) != len(seed_values):
        raise ValueError("P6 train/eval/prefill seeds must be pairwise distinct")

    eval_seed_start = _required_int(
        p6.eval_seed_start,
        "p6.eval_seed_start",
    )
    eval_seed_count = _required_int(
        p6.eval_seed_count,
        "p6.eval_seed_count",
        minimum=1,
    )
    seed_plan["eval_seed_set"] = list(
        range(eval_seed_start, eval_seed_start + eval_seed_count)
    )
    return seed_plan


def _validate_run_name(
    run_name: Any,
    *,
    init_checkpoint_id: str,
    algorithm: str,
    seed: int,
    chunk_budget: int,
) -> str:
    result = _required_string(run_name, "name")
    required_tokens = (
        init_checkpoint_id,
        algorithm,
        f"seed{seed}",
        f"{chunk_budget}chunks",
    )
    missing = [token for token in required_tokens if token not in result]
    if missing:
        raise ValueError(
            f"P6 run name is missing required tokens {missing}: {result!r}"
        )
    return result


def static_preflight(
    cfg: Any,
    repo_root: Path,
    *,
    algorithm: str,
) -> dict[str, Any]:
    """Validate everything that does not require a constructed environment."""

    repo_root = repo_root.resolve()
    p6 = cfg.p6
    if algorithm not in {*HIERARCHY_ALGORITHMS, CONTROL_ALGORITHM}:
        raise ValueError(f"Unsupported P6 algorithm label: {algorithm!r}")

    seed = _required_int(cfg.seed, "seed")
    chunk_budget = _required_int(
        cfg.total_timesteps,
        "total_timesteps",
        minimum=1,
    )
    action_chunk = _required_int(cfg.act_steps, "act_steps", minimum=1)
    action_dimension = _required_int(cfg.action_dim, "action_dim", minimum=1)
    n_envs = _required_int(cfg.env.n_envs, "env.n_envs", minimum=1)
    required_n_envs = _required_int(
        p6.pilot_n_envs,
        "p6.pilot_n_envs",
        minimum=1,
    )
    if n_envs != required_n_envs:
        raise ValueError(
            "P6 pilot n_envs must be supplied as the explicit audited override: "
            f"expected {required_n_envs}, got {n_envs}"
        )
    train_freq_field = (
        "train.rfs_hier_train_freq"
        if algorithm == HIERARCHY_ALGORITHM
        else "train.train_freq"
    )
    train_freq_value = (
        cfg.train.rfs_hier_train_freq
        if algorithm == HIERARCHY_ALGORITHM
        else cfg.train.train_freq
    )
    train_freq = _required_int(train_freq_value, train_freq_field, minimum=1)
    critic_backup_combine_type = _required_string(
        cfg.train.critic_backup_combine_type,
        "train.critic_backup_combine_type",
    ).lower()
    if critic_backup_combine_type != "min":
        raise ValueError("P6 requires critic_backup_combine_type='min'")
    gradient_steps = _required_int(cfg.train.utd, "train.utd", minimum=1)
    noise_critic_steps = _required_int(
        cfg.train.noise_critic_grad_steps,
        "train.noise_critic_grad_steps",
        minimum=1,
    )
    if (gradient_steps, noise_critic_steps) != (20, 10):
        raise ValueError(
            "P6 inherited DSRL critic constructor contract must remain 20/10"
        )
    hierarchy_schedule = None
    hierarchy_contract = None
    if algorithm == HIERARCHY_ALGORITHM:
        if _required_string(
            p6.architecture_version, "p6.architecture_version"
        ) != ARCHITECTURE_VERSION:
            raise ValueError("P6 hierarchy architecture version mismatch")
        if _required_string(
            p6.replay_schema_version, "p6.replay_schema_version"
        ) != HIERARCHY_REPLAY_SCHEMA_VERSION:
            raise ValueError("P6 hierarchy replay schema version mismatch")
        profile_name = _required_string(
            cfg.train.rfs_hier_schedule_profile,
            "train.rfs_hier_schedule_profile",
        )
        if profile_name not in SUPPORTED_HIERARCHY_PROFILES:
            raise ValueError(
                "Unsupported hierarchy schedule profile "
                f"{profile_name!r}; supported: {SUPPORTED_HIERARCHY_PROFILES}"
            )
        prefill_source = _required_string(
            p6.prefill_source, "p6.prefill_source"
        )
        if prefill_source not in PREFILL_SOURCES:
            raise ValueError(
                f"P6 prefill source must be one of {PREFILL_SOURCES}, "
                f"got {prefill_source!r}"
            )
        if (
            profile_name == LEGACY_WARMSTART_PROFILE
            and prefill_source != PREFILL_SOURCE
        ):
            raise ValueError(
                "Legacy warm-start profile requires prefill source "
                f"{PREFILL_SOURCE!r}, got {prefill_source!r}"
            )
        if (
            profile_name in FRESH_FROZEN_PROFILES
            and prefill_source != FRESH_PREFILL_SOURCE
        ):
            raise ValueError(
                "Fresh Frozen-DDIM profile requires prefill source "
                f"{FRESH_PREFILL_SOURCE!r}, got {prefill_source!r}"
            )
        schedule_overrides = {
            "phase_b_steps": _required_int(
                cfg.train.rfs_hier_phase_b_steps,
                "train.rfs_hier_phase_b_steps",
            ),
            "phase_r_steps": _required_int(
                cfg.train.rfs_hier_phase_r_steps,
                "train.rfs_hier_phase_r_steps",
                minimum=1,
            ),
            "phase_j_steps": _required_int(
                cfg.train.rfs_hier_phase_j_steps,
                "train.rfs_hier_phase_j_steps",
            ),
            "phase_j_enabled": _required_bool(
                cfg.train.rfs_hier_phase_j_enabled,
                "train.rfs_hier_phase_j_enabled",
            ),
            "beta_ramp_steps": _required_int(
                cfg.train.rfs_hier_beta_ramp_steps,
                "train.rfs_hier_beta_ramp_steps",
            ),
            "beta_target": float(cfg.train.rfs_hier_beta_target),
            "base_lane_probability": float(
                cfg.train.rfs_hier_base_lane_probability
            ),
        }
        # Co-training schedule fields are optional: absent -> the profile's
        # frozen default (0 hold / ramp from 0), so every existing config is
        # unchanged.
        if cfg.train.get("rfs_hier_beta_hold_steps") is not None:
            schedule_overrides["beta_hold_steps"] = _required_int(
                cfg.train.rfs_hier_beta_hold_steps,
                "train.rfs_hier_beta_hold_steps",
            )
        if cfg.train.get("rfs_hier_beta_floor") is not None:
            schedule_overrides["beta_floor"] = float(
                cfg.train.rfs_hier_beta_floor
            )
        hierarchy_schedule = make_hierarchy_schedule(
            profile_name,
            n_envs=n_envs,
            overrides=schedule_overrides,
        )
        hierarchy_schedule.validate_budget(chunk_budget)
        if (
            profile_name == LEGACY_WARMSTART_PROFILE
            and hierarchy_schedule.phase_b_steps != 0
        ):
            raise ValueError("Legacy warm-start P6 requires Phase B length zero")
        if hierarchy_schedule.phase_j_enabled:
            raise ValueError("Core V1 production default must keep Phase J disabled")
        if hierarchy_schedule.beta_target != 0.1:
            raise ValueError("Core V1 beta_target is frozen at 0.1")
        if hierarchy_schedule.base_lane_probability != 0.5:
            raise ValueError("Core V1 BASE/JOINT lane probability is frozen at 0.5")
        batch_size = _required_int(
            cfg.train.batch_size, "train.batch_size", minimum=1
        )
        minimum_branch = _required_int(
            cfg.train.rfs_hier_min_branch_replay_transitions,
            "train.rfs_hier_min_branch_replay_transitions",
            minimum=1,
        )
        if minimum_branch < batch_size:
            raise ValueError(
                "Hierarchy minimum branch replay transitions must be >= batch size"
            )
        hierarchy_contract = {
            "architecture_version": ARCHITECTURE_VERSION,
            "replay_schema_version": HIERARCHY_REPLAY_SCHEMA_VERSION,
            "schedule_profile": profile_name,
            **schedule_overrides,
            "min_branch_replay_transitions": minimum_branch,
            "prefill_source": prefill_source,
            "prefill_action_policy": (
                PREFILL_ACTION_POLICY
                if prefill_source == PREFILL_SOURCE
                else FRESH_PREFILL_ACTION_POLICY
            ),
            # Per-profile update counts: a profile that defines its own
            # update_profiles serializes those; every frozen profile resolves
            # to DEFAULT_UPDATE_PROFILES exactly as before.
            "update_profiles": {
                phase.name.lower(): profile.as_dict()
                for phase, profile in (
                    hierarchy_schedule.update_profiles
                    or DEFAULT_UPDATE_PROFILES
                ).items()
            },
            # Co-training model-level flags (defaults preserve frozen behavior).
            "qa_joint_shadow_in_b": bool(
                cfg.train.get("rfs_hier_qa_joint_shadow_in_b", False)
            ),
            "cross_lane_ratio": float(
                cfg.train.get("rfs_hier_cross_lane_ratio", 0.0)
            ),
            "qa_base_cross_lane": bool(
                cfg.train.get("rfs_hier_qa_base_cross_lane", False)
            ),
            "residual_exploration_std": float(
                cfg.train.get("rfs_hier_residual_exploration_std", 0.0)
            ),
        }
    safe_boundary_chunks = n_envs * train_freq
    if chunk_budget % safe_boundary_chunks != 0:
        raise ValueError(
            "Chunk budget must be divisible by n_envs * train_freq: "
            f"{chunk_budget} % {safe_boundary_chunks} != 0"
        )
    expected_action_chunk = _required_int(
        p6.expected_action_chunk,
        "p6.expected_action_chunk",
        minimum=1,
    )
    expected_action_dimension = _required_int(
        p6.expected_action_dimension,
        "p6.expected_action_dimension",
        minimum=1,
    )
    if action_chunk != expected_action_chunk:
        raise ValueError(
            f"Action chunk mismatch: expected {expected_action_chunk}, got {action_chunk}"
        )
    if action_dimension != expected_action_dimension:
        raise ValueError(
            "Action dimension mismatch: expected "
            f"{expected_action_dimension}, got {action_dimension}"
        )
    if _required_int(
        cfg.env.max_episode_steps,
        "env.max_episode_steps",
        minimum=1,
    ) % action_chunk != 0:
        raise ValueError("max_episode_steps must be divisible by action_chunk")

    init_checkpoint_id = _required_string(
        p6.init_checkpoint_id,
        "p6.init_checkpoint_id",
    )
    # A run starts fresh (no legacy DSRL checkpoint) when the hierarchy uses a
    # Fresh Frozen-DDIM profile, or when the matched flat-DSRL control is given
    # no legacy checkpoint (rfs_hier_legacy_checkpoint_path=null).  Fresh runs
    # require init_checkpoint_id='fresh_frozen_ddim' and the Gaussian-prior
    # prefill source below.
    fresh_init = (
        algorithm == HIERARCHY_ALGORITHM
        and profile_name in FRESH_FROZEN_PROFILES
    ) or (
        algorithm == CONTROL_ALGORITHM
        and cfg.rfs_hier_legacy_checkpoint_path is None
    )
    if fresh_init:
        if init_checkpoint_id != "fresh_frozen_ddim":
            raise ValueError(
                "Fresh Frozen-DDIM profiles require p6.init_checkpoint_id="
                f"'fresh_frozen_ddim', got {init_checkpoint_id!r}"
            )
        init_checkpoint_path = None
        init_checkpoint_sha256 = None
    else:
        if init_checkpoint_id != "init_5m":
            raise ValueError(
                f"P6.1 only accepts init_5m, got {init_checkpoint_id!r}"
            )
        init_checkpoint_path, init_checkpoint_sha256 = _verify_artifact(
            repo_root,
            cfg.rfs_hier_legacy_checkpoint_path,
            p6.init_checkpoint_sha256,
            "init_checkpoint",
        )
    frozen_ddim_path, frozen_ddim_sha256 = _verify_artifact(
        repo_root,
        cfg.base_policy_path,
        p6.frozen_ddim_sha256,
        "frozen_ddim",
    )
    normalization_path, normalization_sha256 = _verify_artifact(
        repo_root,
        cfg.normalization_path,
        p6.normalization_sha256,
        "normalization",
    )

    ddim_steps = _required_int(cfg.model.ddim_steps, "model.ddim_steps", minimum=1)
    expected_ddim_steps = _required_int(
        p6.expected_ddim_steps,
        "p6.expected_ddim_steps",
        minimum=1,
    )
    if ddim_steps != expected_ddim_steps:
        raise ValueError(
            f"DDIM step mismatch: expected {expected_ddim_steps}, got {ddim_steps}"
        )
    denoised_clip_value = float(cfg.model.denoised_clip_value)
    expected_denoised_clip_value = float(p6.expected_denoised_clip_value)
    if not np.isfinite(denoised_clip_value) or (
        denoised_clip_value != expected_denoised_clip_value
    ):
        raise ValueError(
            "Denoised clip mismatch: expected "
            f"{expected_denoised_clip_value}, got {denoised_clip_value}"
        )

    prefill_source = _required_string(p6.prefill_source, "p6.prefill_source")
    if prefill_source not in PREFILL_SOURCES:
        raise ValueError(
            f"P6 prefill source must be one of {PREFILL_SOURCES}, "
            f"got {prefill_source!r}"
        )
    # Non-hierarchy algorithms (control) derive their prefill at runtime.  A
    # legacy-checkpoint control always uses the warmstart CURRENT_ACTOR source;
    # a fresh matched-DSRL control (no legacy checkpoint) uses the same
    # Gaussian-prior source as the fresh hierarchy, so its prefill artifact is
    # the identical seeded decode.  A config claiming the wrong source would
    # mislabel the manifest, so gate it explicitly.
    if algorithm != HIERARCHY_ALGORITHM:
        expected_prefill_source = (
            FRESH_PREFILL_SOURCE if fresh_init else PREFILL_SOURCE
        )
        if prefill_source != expected_prefill_source:
            raise ValueError(
                f"Algorithm {algorithm!r} requires prefill source "
                f"{expected_prefill_source!r}, got {prefill_source!r}"
            )
    termination_semantics = _required_string(
        p6.action_chunk_termination_semantics,
        "p6.action_chunk_termination_semantics",
    )
    if termination_semantics != ACTION_CHUNK_TERMINATION:
        raise ValueError(
            "P6 requires early-break ActionChunk termination semantics"
        )
    resume_semantics = _required_string(
        p6.environment_resume_mode,
        "p6.environment_resume_mode",
    )
    if resume_semantics != SAFE_RESUME_SEMANTICS:
        raise ValueError(
            "P6 supports only reset-boundary discontinuous environment resume"
        )
    expected_algorithm_label = (
        algorithm if algorithm in HIERARCHY_ALGORITHMS else CONTROL_ALGORITHM
    )
    algorithm_label = _required_string(
        p6.algorithm_label,
        "p6.algorithm_label",
    )
    if algorithm_label != expected_algorithm_label:
        raise ValueError(
            f"P6 algorithm label mismatch: {algorithm_label!r} != "
            f"{expected_algorithm_label!r}"
        )

    online_eval_interval = _required_int(
        p6.online_eval_interval_chunk_transitions,
        "p6.online_eval_interval_chunk_transitions",
        minimum=1,
    )
    model_checkpoint_interval = _required_int(
        p6.model_checkpoint_interval_chunk_transitions,
        "p6.model_checkpoint_interval_chunk_transitions",
        minimum=1,
    )
    replay_checkpoint_interval = _required_int(
        p6.replay_checkpoint_interval_chunk_transitions,
        "p6.replay_checkpoint_interval_chunk_transitions",
        minimum=1,
    )
    for field_name, interval in (
        ("online evaluation", online_eval_interval),
        ("model checkpoint", model_checkpoint_interval),
        ("replay checkpoint", replay_checkpoint_interval),
    ):
        if interval % safe_boundary_chunks != 0:
            raise ValueError(
                f"{field_name} interval must be divisible by the safe boundary "
                f"{safe_boundary_chunks}"
            )
    stop_after_chunk_transitions = _optional_int(
        p6.stop_after_chunk_transitions,
        "p6.stop_after_chunk_transitions",
        minimum=1,
    )
    if stop_after_chunk_transitions is not None:
        if stop_after_chunk_transitions >= chunk_budget:
            raise ValueError(
                "Intentional interruption must occur before the target budget"
            )
        if stop_after_chunk_transitions % safe_boundary_chunks != 0:
            raise ValueError(
                "Intentional interruption must align to a safe training boundary"
            )
        if stop_after_chunk_transitions % replay_checkpoint_interval != 0:
            raise ValueError(
                "Intentional interruption must align to a replay checkpoint"
            )
    test_cadence_override = bool(p6.test_cadence_override)
    if test_cadence_override and chunk_budget > 10_000:
        raise ValueError("Shortened test cadence is only allowed up to 10k chunks")
    eval_policy_seed_start = _required_int(
        p6.eval_policy_seed_start,
        "p6.eval_policy_seed_start",
    )
    online_eval_episodes = _required_int(
        p6.online_eval_episodes,
        "p6.online_eval_episodes",
        minimum=1,
    )
    final_eval_episodes = _required_int(
        p6.final_eval_episodes,
        "p6.final_eval_episodes",
        minimum=1,
    )
    evaluation_batch_size = _required_int(
        p6.evaluation_batch_size,
        "p6.evaluation_batch_size",
        minimum=1,
    )
    eval_seed_count = _required_int(
        p6.eval_seed_count,
        "p6.eval_seed_count",
        minimum=1,
    )
    if online_eval_episodes > eval_seed_count or final_eval_episodes > eval_seed_count:
        raise ValueError(
            "Online/final evaluation episode counts cannot exceed eval_seed_count"
        )
    prefill_artifact_path = Path(
        _required_string(
            p6.prefill_artifact_path,
            "p6.prefill_artifact_path",
        )
    ).expanduser()
    if not prefill_artifact_path.is_absolute():
        prefill_artifact_path = (repo_root / prefill_artifact_path).resolve()

    run_name = _validate_run_name(
        cfg.name,
        init_checkpoint_id=init_checkpoint_id,
        algorithm=algorithm,
        seed=seed,
        chunk_budget=chunk_budget,
    )
    seed_plan = resolve_seed_plan(cfg)

    stable_repo = (repo_root / "stable-baselines3").resolve()
    dppo_repo = Path(str(cfg.dppo_path)).expanduser()
    if not dppo_repo.is_absolute():
        dppo_repo = repo_root / dppo_repo
    dppo_repo = dppo_repo.resolve()
    source_provenance = {
        "outer": git_source_fingerprint(repo_root),
        "stable_baselines3": git_source_fingerprint(stable_repo),
        "dppo": git_source_fingerprint(dppo_repo),
    }
    source_state_sha256 = canonical_mapping_sha256(source_provenance)

    flat_action_dimension = action_chunk * action_dimension
    expected_execution_low = _expanded_bound(
        p6.expected_exec_action_low,
        flat_action_dimension,
        "p6.expected_exec_action_low",
    ).tolist()
    expected_execution_high = _expanded_bound(
        p6.expected_exec_action_high,
        flat_action_dimension,
        "p6.expected_exec_action_high",
    ).tolist()
    training_contract = {
        "actor_learning_rate": float(cfg.train.actor_lr),
        "batch_size": _required_int(
            cfg.train.batch_size,
            "train.batch_size",
            minimum=1,
        ),
        "buffer_size": _required_int(
            cfg.train.buffer_size_na,
            "train.buffer_size_na",
            minimum=1,
        ),
        "critic_backup_combine_type": critic_backup_combine_type,
        "ent_coef": (
            float(cfg.train.ent_coef)
            if not isinstance(cfg.train.ent_coef, str)
            else str(cfg.train.ent_coef)
        ),
        "gamma": float(cfg.train.discount),
        "gradient_steps": gradient_steps,
        "noise_critic_gradient_steps": noise_critic_steps,
        "hierarchy": hierarchy_contract,
        "qa_joint_learning_rate": float(cfg.train.rfs_hier_qa_joint_lr),
        "residual_activation": str(cfg.train.rfs_hier_residual_activation),
        "residual_learning_rate": float(cfg.train.rfs_hier_residual_lr),
        "residual_net_arch": [
            int(value) for value in cfg.train.rfs_hier_residual_net_arch
        ],
        "noise_gradient_max_norm": float(
            cfg.train.rfs_hier_noise_gradient_max_norm
        ),
        "residual_gradient_max_norm": float(
            cfg.train.rfs_hier_residual_gradient_max_norm
        ),
        "target_entropy": float(cfg.train.target_ent),
        "tau": float(cfg.train.tau),
        "train_freq": train_freq,
    }
    config_contract = {
        "action_chunk": action_chunk,
        "action_chunk_termination_semantics": termination_semantics,
        "action_dimension": action_dimension,
        "algorithm": algorithm,
        "architecture_version": (
            ARCHITECTURE_VERSION if algorithm == HIERARCHY_ALGORITHM else None
        ),
        "chunk_budget": chunk_budget,
        "ddim_steps": ddim_steps,
        "denoised_clip_value": denoised_clip_value,
        "deterministic_eval": bool(cfg.deterministic_eval),
        "env_name": str(cfg.env_name),
        "eval_policy_seed_start": eval_policy_seed_start,
        "eval_seed_set": seed_plan["eval_seed_set"],
        "execution_action_high": expected_execution_high,
        "execution_action_low": expected_execution_low,
        "final_eval_episodes": final_eval_episodes,
        "frozen_ddim_sha256": frozen_ddim_sha256,
        "init_checkpoint_sha256": init_checkpoint_sha256,
        "max_episode_primitive_steps": _required_int(
            cfg.env.max_episode_steps,
            "env.max_episode_steps",
            minimum=1,
        ),
        "model_checkpoint_interval": model_checkpoint_interval,
        "n_envs": n_envs,
        "normalization_sha256": normalization_sha256,
        "online_eval_episodes": online_eval_episodes,
        "online_eval_interval": online_eval_interval,
        "prefill_action_policy": (
            PREFILL_ACTION_POLICY
            if prefill_source == PREFILL_SOURCE
            else FRESH_PREFILL_ACTION_POLICY
        ),
        "prefill_env_seed": seed_plan["prefill_env_seed"],
        "hierarchy_prefill_residual_mode": (
            HIERARCHY_PREFILL_RESIDUAL_MODE
        ),
        "prefill_policy_seed": seed_plan["prefill_policy_seed"],
        "prefill_vector_steps": _required_int(
            cfg.train.init_rollout_steps,
            "train.init_rollout_steps",
            minimum=1,
        ),
        "replay_checkpoint_interval": replay_checkpoint_interval,
        "replay_schema_version": (
            HIERARCHY_REPLAY_SCHEMA_VERSION
            if algorithm == HIERARCHY_ALGORITHM
            else None
        ),
        "resume_semantics": resume_semantics,
        "run_name": run_name,
        "seed": seed,
        "source_state_sha256": source_state_sha256,
        "training": training_contract,
    }
    config_contract_sha256 = canonical_mapping_sha256(config_contract)

    return {
        "manifest_version": 3,
        "preflight_status": "artifacts_verified",
        # The contract hash identifies a configuration.  A separate random ID
        # prevents bundles from two physically distinct runs with the same
        # config/seed from becoming interchangeable.
        "run_id": f"p6-{uuid.uuid4().hex}",
        "config_contract": config_contract,
        "config_contract_sha256": config_contract_sha256,
        "algorithm": algorithm,
        "architecture_version": (
            ARCHITECTURE_VERSION if algorithm == HIERARCHY_ALGORITHM else None
        ),
        "run_name": run_name,
        "seed": seed,
        **seed_plan,
        "init_checkpoint_id": init_checkpoint_id,
        "init_checkpoint_path": (
            None if init_checkpoint_path is None else str(init_checkpoint_path)
        ),
        "init_checkpoint_sha256": init_checkpoint_sha256,
        "frozen_ddim_path": str(frozen_ddim_path),
        "frozen_ddim_sha256": frozen_ddim_sha256,
        "normalization_path": str(normalization_path),
        "normalization_sha256": normalization_sha256,
        "source_provenance": source_provenance,
        "source_state_sha256": source_state_sha256,
        "outer_repository": git_state(repo_root),
        "stable_baselines3_submodule": verified_submodule_state(
            repo_root,
            "stable-baselines3",
            stable_repo,
        ),
        "dppo_submodule": verified_submodule_state(
            repo_root,
            "dppo",
            dppo_repo,
        ),
        "ddim_steps": ddim_steps,
        "denoised_clip_value": denoised_clip_value,
        "action_chunk": action_chunk,
        "action_dimension": action_dimension,
        "n_envs": n_envs,
        "train_freq": train_freq,
        "training_contract": training_contract,
        "hierarchy_schedule": hierarchy_contract,
        "replay_schema_version": (
            HIERARCHY_REPLAY_SCHEMA_VERSION
            if algorithm == HIERARCHY_ALGORITHM
            else None
        ),
        "critic_backup_combine_type": critic_backup_combine_type,
        "safe_boundary_chunk_transitions": safe_boundary_chunks,
        "chunk_budget": chunk_budget,
        "nominal_primitive_budget": chunk_budget * action_chunk,
        "actual_primitive_budget_upper_bound": chunk_budget * action_chunk,
        "prefill_source": prefill_source,
        "prefill_action_policy": (
            PREFILL_ACTION_POLICY
            if prefill_source == PREFILL_SOURCE
            else FRESH_PREFILL_ACTION_POLICY
        ),
        "hierarchy_prefill_residual_mode": (
            HIERARCHY_PREFILL_RESIDUAL_MODE
        ),
        "prefill_transition_count": (
            _required_int(
                cfg.train.init_rollout_steps,
                "train.init_rollout_steps",
            )
            * n_envs
        ),
        "prefill_hash": None,
        "prefill_status": "pending_tagged_prefill",
        "prefill_artifact_path": str(prefill_artifact_path),
        "action_chunk_termination_semantics": termination_semantics,
        "online_eval_interval_chunk_transitions": online_eval_interval,
        "model_checkpoint_interval_chunk_transitions": model_checkpoint_interval,
        "replay_checkpoint_interval_chunk_transitions": (
            replay_checkpoint_interval
        ),
        "online_eval_episodes": online_eval_episodes,
        "final_eval_episodes": final_eval_episodes,
        "evaluation_batch_size": evaluation_batch_size,
        "eval_policy_seed_start": eval_policy_seed_start,
        "test_cadence_override": test_cadence_override,
        "stop_after_chunk_transitions": stop_after_chunk_transitions,
        "resume_semantics": resume_semantics,
    }


def _expanded_bound(value: Any, flat_dimension: int, field_name: str) -> np.ndarray:
    configured = np.asarray(value, dtype=np.float32)
    if configured.ndim == 0:
        configured = np.full(flat_dimension, configured.item(), dtype=np.float32)
    configured = configured.reshape(-1)
    if configured.shape != (flat_dimension,):
        raise ValueError(
            f"{field_name} must be scalar or shape ({flat_dimension},), "
            f"got {configured.shape}"
        )
    if not np.isfinite(configured).all():
        raise ValueError(f"{field_name} contains non-finite values")
    return configured


def validate_execution_bounds(cfg: Any, env: Any) -> dict[str, Any]:
    flat_dimension = int(cfg.act_steps) * int(cfg.action_dim)
    actual_low = np.asarray(env.action_space.low, dtype=np.float32).reshape(-1)
    actual_high = np.asarray(env.action_space.high, dtype=np.float32).reshape(-1)
    if actual_low.shape != (flat_dimension,) or actual_high.shape != (
        flat_dimension,
    ):
        raise ValueError(
            "Execution action shape mismatch: expected "
            f"({flat_dimension},), got low={actual_low.shape}, high={actual_high.shape}"
        )
    if not np.isfinite(actual_low).all() or not np.isfinite(actual_high).all():
        raise ValueError("Execution bounds contain non-finite values")
    if np.any(actual_low >= actual_high):
        raise ValueError("Execution action bounds require low < high in every dimension")

    expected_low = _expanded_bound(
        cfg.p6.expected_exec_action_low,
        flat_dimension,
        "p6.expected_exec_action_low",
    )
    expected_high = _expanded_bound(
        cfg.p6.expected_exec_action_high,
        flat_dimension,
        "p6.expected_exec_action_high",
    )
    if not np.array_equal(actual_low, expected_low):
        raise ValueError(
            f"Execution low bounds mismatch: expected {expected_low}, got {actual_low}"
        )
    if not np.array_equal(actual_high, expected_high):
        raise ValueError(
            f"Execution high bounds mismatch: expected {expected_high}, got {actual_high}"
        )
    return {
        "execution_action_shape": list(env.action_space.shape),
        "execution_action_low": actual_low.tolist(),
        "execution_action_high": actual_high.tolist(),
    }


def validate_observation_dimension(cfg: Any, env: Any) -> dict[str, Any]:
    # ActionChunkWrapper derives its observation_space from cfg.obs_dim
    # (env_utils.py:182-186) and a VecEnv propagates that wrapped space, so any
    # check against the wrapped/vectorized env space is circular: it is built
    # from cfg.obs_dim and can never disagree with it.  The per-env normalization
    # file is the independent ground truth — obs_min/obs_max are indexed by the
    # RAW per-step observation (they are applied before chunking), so their
    # length is the real observation dimension regardless of cfg.obs_dim.  A
    # stale obs_dim (e.g. hopper 11 vs HC/WK 17) otherwise surfaces only as an
    # opaque torch size mismatch at model load, long after preflight passed.
    #
    # `env` is intentionally unused here: the runner sets cfg.normalization_path
    # to an absolute, sha256-verified path before preflight (p6_train.py copies
    # static_manifest["normalization_path"], which _verify_artifact resolved).
    # If no normalization file is reachable we skip the check rather than fail —
    # this is a defensive cross-check, not an execution gate.
    normalization_path = getattr(cfg, "normalization_path", None)
    actual = None
    if normalization_path is not None:
        resolved = Path(str(normalization_path))
        if resolved.is_file():
            normalization = np.load(resolved)
            obs_key = next(
                (k for k in ("obs_min", "obs_max", "mean") if k in normalization.files),
                None,
            )
            if obs_key is not None:
                actual = int(np.prod(normalization[obs_key].shape))
    if actual is not None and actual != int(cfg.obs_dim):
        raise ValueError(
            "Observation dimension mismatch: cfg.obs_dim="
            f"{cfg.obs_dim}, but normalization file {normalization_path} "
            f"records {actual} per-step observation dims"
        )
    return {"observation_dimension": actual}


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as output:
        json.dump(payload, output, indent=2, sort_keys=True)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary_path, path)


def run_preflight(
    cfg: Any,
    env: Any,
    repo_root: Path,
    manifest_path: Path,
    *,
    algorithm: str,
    static_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    manifest = (
        dict(static_manifest)
        if static_manifest is not None
        else static_preflight(cfg, repo_root, algorithm=algorithm)
    )
    if manifest.get("algorithm") != algorithm:
        raise ValueError("Static preflight algorithm does not match the run algorithm")
    manifest.update(validate_execution_bounds(cfg, env))
    manifest.update(validate_observation_dimension(cfg, env))
    manifest["preflight_status"] = "execution_contract_verified"
    _atomic_write_json(manifest_path, manifest)
    return manifest


def finalize_loaded_model_preflight(
    cfg: Any,
    env: Any,
    model: Any,
    manifest_path: Path,
    *,
    network_warmstart: bool = False,
    legacy_loaded: bool = False,
) -> dict[str, Any]:
    """Validate the officially loaded checkpoint without parsing its zip.

    ``network_warmstart`` is a pure recorded label (the manifest flag naming
    how the network was initialized); the step checks are driven by
    ``legacy_loaded``, which is true only when a real checkpoint was loaded
    and its interaction-step count is preserved.  A fresh (no-legacy-checkpoint)
    run and a network-warm-start from a legacy checkpoint both construct the
    model with counters at zero, so ``expected_init_checkpoint_steps`` is
    informational for them and zero is legal; a genuinely loaded checkpoint
    must match its own step count exactly.
    """

    expected_steps = _required_int(
        cfg.p6.expected_init_checkpoint_steps,
        "p6.expected_init_checkpoint_steps",
        minimum=(0 if not legacy_loaded else 1),
    )
    actual_steps = _required_int(
        model.num_timesteps,
        "loaded_model.num_timesteps",
    )
    if legacy_loaded and actual_steps != expected_steps:
        raise ValueError(
            f"Loaded checkpoint step mismatch: expected {expected_steps}, got {actual_steps}"
        )
    if not legacy_loaded and actual_steps != 0:
        raise ValueError(
            "A model constructed at zero interaction steps (fresh init, or a "
            "network warm-start whose counters were reset) must not carry a "
            f"nonzero step count, got {actual_steps}"
        )

    expected_diffusion_dims = (int(cfg.act_steps), int(cfg.action_dim))
    actual_diffusion_dims = (
        int(model.diffusion_act_chunk),
        int(model.diffusion_act_dim),
    )
    if actual_diffusion_dims != expected_diffusion_dims:
        raise ValueError(
            "Loaded checkpoint diffusion shape mismatch: expected "
            f"{expected_diffusion_dims}, got {actual_diffusion_dims}"
        )
    execution_contract = validate_execution_bounds(cfg, env)
    model_low = np.asarray(model.action_space.low, dtype=np.float32).reshape(-1)
    model_high = np.asarray(model.action_space.high, dtype=np.float32).reshape(-1)
    env_low = np.asarray(env.action_space.low, dtype=np.float32).reshape(-1)
    env_high = np.asarray(env.action_space.high, dtype=np.float32).reshape(-1)
    if not np.array_equal(model_low, env_low) or not np.array_equal(
        model_high,
        env_high,
    ):
        raise ValueError("Loaded checkpoint and environment execution bounds differ")
    if str(cfg.algorithm) == HIERARCHY_ALGORITHM:
        if getattr(model, "architecture_version", None) != ARCHITECTURE_VERSION:
            raise ValueError("Loaded hierarchy architecture version differs")
        if (
            getattr(model, "replay_schema_version", None)
            != HIERARCHY_REPLAY_SCHEMA_VERSION
        ):
            raise ValueError("Loaded hierarchy replay schema version differs")

    with manifest_path.open("r", encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)
    manifest.update(execution_contract)
    manifest.update(
        {
            "init_checkpoint_num_timesteps": expected_steps,
            "training_start_num_timesteps": actual_steps,
            "network_warmstart": network_warmstart,
            "legacy_loaded": legacy_loaded,
            "loaded_diffusion_shape": list(actual_diffusion_dims),
            "preflight_status": "loaded_model_verified_prefill_pending",
        }
    )
    _atomic_write_json(manifest_path, manifest)
    return manifest
