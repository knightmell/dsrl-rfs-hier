"""P6 run provenance, seed-plan, artifact, and execution-bound validation."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

import numpy as np


HIERARCHY_ALGORITHM = "dsrl_na_rfs_hier"
CONTROL_ALGORITHM = "dsrl_na_control"
LEGACY_CHUNK_TERMINATION = "legacy_continue_after_done"
PREFILL_SOURCE = "warmstart_dsrl"


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
    if algorithm not in {HIERARCHY_ALGORITHM, CONTROL_ALGORITHM}:
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

    init_checkpoint_id = _required_string(
        p6.init_checkpoint_id,
        "p6.init_checkpoint_id",
    )
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
    if prefill_source != PREFILL_SOURCE:
        raise ValueError(
            f"P6 prefill source must be {PREFILL_SOURCE!r}, got {prefill_source!r}"
        )
    termination_semantics = _required_string(
        p6.action_chunk_termination_semantics,
        "p6.action_chunk_termination_semantics",
    )
    if termination_semantics != LEGACY_CHUNK_TERMINATION:
        raise ValueError(
            "P6 must preserve the audited legacy ActionChunk termination semantics"
        )

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

    return {
        "manifest_version": 1,
        "preflight_status": "artifacts_verified",
        "algorithm": algorithm,
        "run_name": run_name,
        "seed": seed,
        **seed_plan,
        "init_checkpoint_id": init_checkpoint_id,
        "init_checkpoint_path": str(init_checkpoint_path),
        "init_checkpoint_sha256": init_checkpoint_sha256,
        "frozen_ddim_path": str(frozen_ddim_path),
        "frozen_ddim_sha256": frozen_ddim_sha256,
        "normalization_path": str(normalization_path),
        "normalization_sha256": normalization_sha256,
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
        "chunk_budget": chunk_budget,
        "primitive_budget": chunk_budget * action_chunk,
        "prefill_source": prefill_source,
        "prefill_transition_count": (
            _required_int(
                cfg.train.init_rollout_steps,
                "train.init_rollout_steps",
            )
            * n_envs
        ),
        "prefill_hash": None,
        "prefill_status": "pending_p6_2",
        "action_chunk_termination_semantics": termination_semantics,
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
) -> dict[str, Any]:
    """Validate the officially loaded checkpoint without parsing its zip."""

    expected_steps = _required_int(
        cfg.p6.expected_init_checkpoint_steps,
        "p6.expected_init_checkpoint_steps",
        minimum=1,
    )
    actual_steps = _required_int(
        model.num_timesteps,
        "loaded_model.num_timesteps",
    )
    if not network_warmstart and actual_steps != expected_steps:
        raise ValueError(
            f"Loaded checkpoint step mismatch: expected {expected_steps}, got {actual_steps}"
        )
    if network_warmstart and actual_steps != 0:
        raise ValueError(
            "A fresh hierarchy network-warm-start must begin at zero new "
            f"interaction steps, got {actual_steps}"
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

    with manifest_path.open("r", encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)
    manifest.update(execution_contract)
    manifest.update(
        {
            "init_checkpoint_num_timesteps": expected_steps,
            "training_start_num_timesteps": actual_steps,
            "network_warmstart": network_warmstart,
            "loaded_diffusion_shape": list(actual_diffusion_dims),
            "preflight_status": "loaded_model_verified_prefill_pending",
        }
    )
    _atomic_write_json(manifest_path, manifest)
    return manifest
