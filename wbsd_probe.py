"""Read-only Walker Base Support Diagnosis (WBSD) probe harness.

G0 deliberately contains no training loop and no environment interaction.  It
loads one saved hierarchy checkpoint, samples a tiny candidate batch, evaluates
the existing QA/QW modules, and writes an auditable artifact.  The reusable
state/particle helpers are kept independent of the production algorithm so the
G1 sampling sweep can be added without changing the training path.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import random
import subprocess
import sys
import time
import warnings
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn


ROOT = Path(__file__).resolve().parent


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(json.dumps(list(contiguous.shape)).encode("ascii"))
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def capture_global_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
    }


def restore_global_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda") is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def global_rng_states_equal(left: dict[str, Any], right: dict[str, Any]) -> bool:
    """Compare RNG states semantically, including NumPy's ndarray payload."""

    if left["python"] != right["python"]:
        return False
    left_numpy = left["numpy"]
    right_numpy = right["numpy"]
    if left_numpy[0] != right_numpy[0] or left_numpy[2:] != right_numpy[2:]:
        return False
    if not np.array_equal(left_numpy[1], right_numpy[1]):
        return False
    if not torch.equal(left["torch_cpu"], right["torch_cpu"]):
        return False
    left_cuda = left.get("torch_cuda")
    right_cuda = right.get("torch_cuda")
    if left_cuda is None or right_cuda is None:
        return left_cuda is None and right_cuda is None
    return len(left_cuda) == len(right_cuda) and all(
        torch.equal(first, second) for first, second in zip(left_cuda, right_cuda)
    )


@contextmanager
def isolated_global_rng():
    """Keep probe/model-loading randomness out of the caller's RNG streams."""

    state = capture_global_rng_state()
    try:
        yield
    finally:
        restore_global_rng_state(state)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_module_state_hash(module: nn.Module) -> str:
    """Hash a module state without depending on state-dict insertion order."""

    digest = hashlib.sha256()
    state = module.state_dict()
    for key in sorted(state):
        tensor = state[key].detach().cpu()
        digest.update(key.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _model_modules(model: Any) -> list[tuple[str, nn.Module]]:
    names = (
        "policy",
        "actor",
        "critic",
        "critic_target",
        "critic_noise",
        "qa_base",
        "qa_base_target",
        "qw_base",
        "qa_joint",
        "qa_joint_target",
        "residual_actor",
        "residual_actor_target",
        "reference_noise_actor",
        "diffusion_policy",
    )
    modules: list[tuple[str, nn.Module]] = []
    seen: set[int] = set()
    for name in names:
        candidate = getattr(model, name, None)
        if isinstance(candidate, nn.Module) and id(candidate) not in seen:
            seen.add(id(candidate))
            modules.append((name, candidate))
    diffusion = getattr(model, "diffusion_policy", None)
    base_policy = getattr(diffusion, "base_policy", None)
    if isinstance(base_policy, nn.Module) and id(base_policy) not in seen:
        modules.append(("diffusion_policy.base_policy", base_policy))
    return modules


def model_state_hashes(model: Any) -> dict[str, str]:
    return {
        name: canonical_module_state_hash(module)
        for name, module in _model_modules(model)
    }


def flatten_state_particles(
    observations: torch.Tensor, particles: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Flatten `[B,K,D]` while preserving state-particle alignment."""

    if observations.ndim != 2:
        raise ValueError("observations must have shape [B, observation_dim]")
    if particles.ndim != 3 or particles.shape[0] != observations.shape[0]:
        raise ValueError("particles must have shape [B,K,latent_dim]")
    batch, candidates, latent_dim = particles.shape
    flat_observations = observations.repeat_interleave(candidates, dim=0)
    flat_particles = particles.reshape(batch * candidates, latent_dim)
    return flat_observations, flat_particles


def nested_prefix(pool: torch.Tensor, candidates: int) -> torch.Tensor:
    """Return a candidate prefix and validate its nested-pool contract."""

    if pool.ndim != 3:
        raise ValueError("candidate pool must have shape [B,K,D]")
    if not 1 <= int(candidates) <= pool.shape[1]:
        raise ValueError("prefix size must lie in [1, pool_size]")
    return pool[:, : int(candidates), :]


def sample_current_actor(
    model: Any,
    observations: torch.Tensor,
    candidates: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """Sample actor latents using the actor's Gaussian before tanh."""

    mean, log_std, kwargs = model.actor.get_action_dist_params(observations)
    if kwargs:
        raise RuntimeError("WBSD G0 does not support SDE noise actors")
    standard_normal = torch.randn(
        observations.shape[0],
        int(candidates),
        mean.shape[-1],
        generator=generator,
        device=observations.device,
        dtype=observations.dtype,
    )
    pre_tanh = mean.unsqueeze(1) + log_std.exp().unsqueeze(1) * standard_normal
    return torch.tanh(pre_tanh)


def current_actor_from_standard_normal(
    model: Any,
    observations: torch.Tensor,
    standard_normal: torch.Tensor,
) -> torch.Tensor:
    """Transform a shared CRN draw through the current actor."""

    if standard_normal.ndim != 3:
        raise ValueError("standard_normal must have shape [B,K,latent_dim]")
    mean, log_std, kwargs = model.actor.get_action_dist_params(observations)
    if kwargs:
        raise RuntimeError("WBSD G0 does not support SDE noise actors")
    if standard_normal.shape[0] != observations.shape[0]:
        raise ValueError("CRN draw and observation batch sizes differ")
    if standard_normal.shape[2] != mean.shape[1]:
        raise ValueError("CRN latent dimension differs from actor dimension")
    pre_tanh = mean.unsqueeze(1) + log_std.exp().unsqueeze(1) * standard_normal
    return torch.tanh(pre_tanh)


def sample_gaussian_prior(
    model: Any,
    observations: torch.Tensor,
    candidates: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return exact DSRL-style decoder noise and its QW coordinate."""

    latent_dim = int(getattr(model, "action_dim_flat"))
    decoder_noise = torch.randn(
        observations.shape[0],
        int(candidates),
        latent_dim,
        generator=generator,
        device=observations.device,
        dtype=observations.dtype,
    )
    flat = decoder_noise.reshape(-1, latent_dim)
    scaled_np = model.policy.scale_action(flat.detach().cpu().numpy())
    scaled = torch.as_tensor(
        np.asarray(scaled_np),
        device=observations.device,
        dtype=observations.dtype,
    ).reshape_as(decoder_noise)
    return decoder_noise, scaled


def decoder_to_scaled(model: Any, decoder_noise: torch.Tensor) -> torch.Tensor:
    """Map decoder-space noise to the QW action coordinate exactly."""

    if decoder_noise.ndim != 2:
        raise ValueError("decoder_noise must have shape [N, latent_dim]")
    scaled_np = model.policy.scale_action(decoder_noise.detach().cpu().numpy())
    return torch.as_tensor(
        np.asarray(scaled_np),
        device=decoder_noise.device,
        dtype=decoder_noise.dtype,
    )


def tensor_sha256(tensor: torch.Tensor) -> str:
    return sha256_array(tensor.detach().cpu().numpy())


def prefix_index_hashes(pool_size: int) -> dict[str, str]:
    """Hash prefix indices only, independent of the method's values."""

    if pool_size <= 0:
        raise ValueError("pool_size must be positive")
    values = {}
    for candidates in (1, 2, 4, 8, 16, 32, 64):
        if candidates > pool_size:
            continue
        values[str(candidates)] = sha256_array(
            np.arange(candidates, dtype=np.int64)
        )
    return values


def evaluate_candidate_batch(
    model: Any,
    observations: torch.Tensor,
    noise_scaled: torch.Tensor,
) -> dict[str, Any]:
    """Run one counted-free DDIM -> QA/QW candidate evaluation."""

    flat_observations, flat_noise = flatten_state_particles(
        observations, noise_scaled
    )
    decoder = model._unscale_noise(flat_noise)
    base_actions = model._decode_noise_decoder_input(flat_observations, decoder)
    target_heads = model.qa_base_target(flat_observations, base_actions)
    student_heads = model.qw_base(flat_observations, flat_noise)
    return {
        "flat_observations": flat_observations,
        "flat_noise": flat_noise,
        "decoder": decoder,
        "base_actions": base_actions,
        "target_heads": target_heads,
        "student_heads": student_heads,
    }


@contextmanager
def counted_candidate_calls(model: Any):
    """Count actual decode, QA, and QW invocations for audit evidence."""

    counts: dict[str, int] = {
        "ddim_decode_calls": 0,
        "ddim_decode_candidates": 0,
        "qa_target_calls": 0,
        "qa_target_candidates": 0,
        "qw_student_calls": 0,
        "qw_student_candidates": 0,
    }
    original_decode = model._decode_noise_decoder_input

    def decode_spy(observations, decoder):
        counts["ddim_decode_calls"] += 1
        counts["ddim_decode_candidates"] += int(decoder.shape[0])
        return original_decode(observations, decoder)

    model._decode_noise_decoder_input = decode_spy
    original_forwards: list[tuple[nn.Module, Any]] = []
    for module, call_key, candidate_key in (
        (model.qa_base_target, "qa_target_calls", "qa_target_candidates"),
        (model.qw_base, "qw_student_calls", "qw_student_candidates"),
    ):
        original_forward = module.forward
        original_forwards.append((module, original_forward))

        def forward_spy(*args, _original=original_forward, _call_key=call_key, _candidate_key=candidate_key, **kwargs):
            counts[_call_key] += 1
            if args:
                counts[_candidate_key] += int(args[0].shape[0])
            elif "observations" in kwargs:
                counts[_candidate_key] += int(kwargs["observations"].shape[0])
            return _original(*args, **kwargs)

        module.forward = forward_spy
    try:
        yield counts
    finally:
        model._decode_noise_decoder_input = original_decode
        for module, original_forward in original_forwards:
            module.forward = original_forward


def load_state_bank(path: Path, count: int) -> tuple[np.ndarray, dict[str, Any]]:
    """Load a deterministic, evenly stratified subset of prefill observations."""

    if count <= 0:
        raise ValueError("state-bank count must be positive")
    with np.load(path, allow_pickle=False) as archive:
        if "observations" not in archive:
            raise KeyError("state bank does not contain observations")
        observations = np.asarray(archive["observations"])
    if observations.ndim < 2:
        raise ValueError("prefill observations must have at least two dimensions")
    flattened = observations.reshape(-1, observations.shape[-1]).astype(
        np.float32, copy=False
    )
    selected_count = min(int(count), int(flattened.shape[0]))
    positions = (
        (np.arange(selected_count, dtype=np.float64) + 0.5)
        * flattened.shape[0]
        / selected_count
    ).astype(np.int64)
    selected = np.ascontiguousarray(flattened[positions])
    metadata = {
        "source_shape": list(observations.shape),
        "available_states": int(flattened.shape[0]),
        "requested_states": int(count),
        "selected_states": int(selected.shape[0]),
        "selection": "evenly_stratified_flattened_prefill_observations",
        "selected_sha256": sha256_array(selected),
    }
    return selected, metadata


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    materialized = list(rows)
    if not materialized:
        raise ValueError("cannot write an empty metrics table")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(materialized[0])
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(materialized)
    temporary.replace(path)


def _git_metadata() -> dict[str, Any]:
    def run(*args: str) -> str:
        try:
            result = subprocess.run(
                args,
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError) as error:
            return f"unavailable:{type(error).__name__}"
        return result.stdout.strip()

    diff = run("git", "diff", "--binary")
    return {
        "commit": run("git", "rev-parse", "HEAD"),
        "status_porcelain": run("git", "status", "--short"),
        "tracked_diff_sha256": hashlib.sha256(diff.encode("utf-8")).hexdigest(),
    }


def _source_hashes() -> dict[str, str]:
    paths = (
        ROOT / "wbsd_probe.py",
        ROOT / "tests/test_wbsd_probe.py",
        ROOT / "docs/rfs_hier_v1/WALKER_BASE_SUPPORT_GATED_PLAN.md",
    )
    return {
        str(path.relative_to(ROOT)): sha256_file(path)
        for path in paths
        if path.is_file()
    }


def _load_hierarchy_checkpoint(
    checkpoint: Path,
    config_path: Path,
    device: str,
) -> Any:
    """Load a hierarchy checkpoint without constructing an environment."""

    sys.path[:0] = [str(ROOT / "dppo"), str(ROOT / "stable-baselines3")]
    from omegaconf import OmegaConf  # pylint: disable=import-outside-toplevel

    OmegaConf.register_new_resolver("eval", eval, replace=True)
    # The historical Hydra snapshot contains a few logging-only `${now:...}`
    # fields.  WBSD never uses them, but resolving the model config requires a
    # deterministic placeholder rather than the original launch timestamp.
    OmegaConf.register_new_resolver(
        "now", lambda _format: "1970-01-01_00-00-00", replace=True
    )
    from utils import load_base_policy  # pylint: disable=import-outside-toplevel
    from stable_baselines3.dsrl.hierarchical_rfs_dsrl import (  # pylint: disable=import-outside-toplevel
        HierarchicalRFSDSRL,
    )

    config = OmegaConf.load(config_path)
    base_policy = (ROOT / str(config.base_policy_path)).resolve()
    if not base_policy.is_file():
        raise FileNotFoundError(f"Frozen DDIM checkpoint not found: {base_policy}")
    OmegaConf.update(config, "device", device, merge=False)
    OmegaConf.update(config, "model.device", device, merge=False)
    OmegaConf.update(config, "base_policy_path", str(base_policy), merge=False)
    OmegaConf.update(config, "model.network_path", str(base_policy), merge=False)
    OmegaConf.resolve(config)
    diffusion_policy = load_base_policy(config)
    model = HierarchicalRFSDSRL.load(
        str(checkpoint),
        env=None,
        device=device,
        diffusion_policy=diffusion_policy,
    )
    return model


def run_smoke(
    *,
    checkpoint: Path,
    config_path: Path,
    state_bank_path: Path,
    output_dir: Path,
    device: str,
    state_count: int,
    candidates: int,
    seed: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    started_at_utc = utc_now()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "plots").mkdir(exist_ok=True)
    state_bank, state_metadata = load_state_bank(state_bank_path, state_count)
    warnings_seen: list[dict[str, str]] = []
    with isolated_global_rng():
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = _load_hierarchy_checkpoint(checkpoint, config_path, device)
            modules = _model_modules(model)
            mode_snapshot = [(module, module.training) for _, module in modules]
            before = model_state_hashes(model)
            try:
                for _, module in modules:
                    module.eval()
                if state_bank.shape[1] != int(model.observation_dim):
                    raise ValueError(
                        f"state dimension {state_bank.shape[1]} != model observation "
                        f"dimension {model.observation_dim}"
                    )
                if int(model.action_dim_flat) != 24:
                    raise ValueError(
                        f"Walker G0 expects latent dimension 24, got {model.action_dim_flat}"
                    )
                observations = torch.as_tensor(
                    state_bank[:1], device=device, dtype=torch.float32
                )
                generator = torch.Generator(device=observations.device)
                generator.manual_seed(int(seed))
                with torch.no_grad():
                    current_pool = sample_current_actor(
                        model, observations, candidates, generator
                    )
                    with counted_candidate_calls(model) as call_counts:
                        evaluated = evaluate_candidate_batch(
                            model, observations, current_pool
                        )
                target_values = [
                    head.detach().reshape(-1).cpu().tolist()
                    for head in evaluated["target_heads"]
                ]
                student_values = [
                    head.detach().reshape(-1).cpu().tolist()
                    for head in evaluated["student_heads"]
                ]
                rows = []
                for index in range(int(candidates)):
                    row: dict[str, Any] = {"candidate_index": index}
                    for head, values in enumerate(target_values):
                        row[f"qa_target_head{head}"] = float(values[index])
                    for head, values in enumerate(student_values):
                        row[f"qw_student_head{head}"] = float(values[index])
                    rows.append(row)
            finally:
                for module, training in mode_snapshot:
                    module.train(training)
            warnings_seen = [
                {
                    "category": type(item.message).__name__,
                    "message": str(item.message),
                }
                for item in caught
            ]
        after = model_state_hashes(model)
    finite = all(
        np.isfinite(value)
        for row in rows
        for value in row.values()
        if isinstance(value, (float, int))
    )
    elapsed = time.perf_counter() - started
    finished_at_utc = utc_now()
    summary: dict[str, Any] = {
        "material_passport": {
            "origin_skill": "academic-research-suite/experiment-agent",
            "origin_mode": "run",
            "verification_status": "EXECUTED_SMOKE",
            "version_label": "wbsd_g0_smoke_v1",
        },
        "gate": "G0",
        "status": "COMPLETE",
        "mode": "real_checkpoint_cpu_smoke",
        "started_at_utc": started_at_utc,
        "finished_at_utc": finished_at_utc,
        "seed": int(seed),
        "state_count": int(state_count),
        "candidates": int(candidates),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "config": str(config_path.resolve()),
        "config_sha256": sha256_file(config_path),
        "state_bank": str(state_bank_path.resolve()),
        "state_bank_sha256": sha256_file(state_bank_path),
        "state_bank_metadata": state_metadata,
        "model_num_timesteps": int(model.num_timesteps),
        "model_observation_dim": int(model.observation_dim),
        "model_latent_dim": int(model.action_dim_flat),
        "candidate_shape": [1, int(candidates), int(model.action_dim_flat)],
        "flat_observation_shape": [int(candidates), int(model.observation_dim)],
        "flat_latent_shape": [int(candidates), int(model.action_dim_flat)],
        "query_counts": {
            **call_counts,
            "environment_steps": 0,
            "optimizer_steps": 0,
        },
        "finite_metrics": bool(finite),
        "module_hashes_before": before,
        "module_hashes_after": after,
        "module_hashes_equal": before == after,
        "rows": rows,
        "elapsed_seconds": float(elapsed),
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "device": device,
        },
        "warnings": warnings_seen,
        "source_hashes": _source_hashes(),
        "git": _git_metadata(),
        "command": sys.argv,
    }
    _write_json(output_dir / "manifest.json", summary)
    _write_json(output_dir / "summary.json", summary)
    _write_csv(output_dir / "metrics.csv", rows)
    (output_dir / "EXECUTION_COMPLETE").write_text(
        "WBSD G0 smoke completed; no optimizer or environment steps.\n",
        encoding="utf-8",
    )
    return summary


def _prefix_ids(seed: int, pool_size: int) -> dict[str, str]:
    index_hashes = prefix_index_hashes(pool_size)
    return {
        candidates: hashlib.sha256(
            json.dumps(
                {
                    "proposal_seed": int(seed),
                    "candidates": int(candidates),
                    "prefix_index_sha256": index_hash,
                },
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        for candidates, index_hash in index_hashes.items()
    }


def _pool_prefix_hashes(pool: torch.Tensor) -> dict[str, str]:
    return {
        candidates: tensor_sha256(nested_prefix(pool, int(candidates)))
        for candidates in prefix_index_hashes(pool.shape[1])
    }


def run_matrix(
    *,
    checkpoints: list[Path],
    config_path: Path,
    state_bank_path: Path,
    output_dir: Path,
    device: str,
    state_count: int,
    pool_size: int,
    proposal_seeds: list[int],
) -> dict[str, Any]:
    """Build the complete G0 checkpoint/CRN matrix without training."""

    if not checkpoints:
        raise ValueError("at least one checkpoint is required")
    if pool_size < 64:
        raise ValueError("G0 matrix pool_size must be at least 64")
    if not proposal_seeds:
        raise ValueError("at least one proposal seed is required")
    started_at_utc = utc_now()
    started = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "pools").mkdir(exist_ok=True)
    (output_dir / "plots").mkdir(exist_ok=True)
    state_bank, state_metadata = load_state_bank(state_bank_path, state_count)
    state_ids = np.arange(state_bank.shape[0], dtype=np.int64)
    np.save(output_dir / "state_bank.npy", state_bank)
    np.save(output_dir / "state_ids.npy", state_ids)
    proposal_draws: dict[str, np.ndarray] = {}
    for proposal_seed in proposal_seeds:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(proposal_seed))
        proposal_draws[str(proposal_seed)] = torch.randn(
            state_bank.shape[0],
            int(pool_size),
            24,
            generator=generator,
            device="cpu",
            dtype=torch.float32,
        ).numpy()
    np.savez_compressed(
        output_dir / "crn_draws.npz",
        **{
            f"seed_{proposal_seed}": proposal_draws[str(proposal_seed)]
            for proposal_seed in proposal_seeds
        },
    )
    source_hashes = _source_hashes()
    state_bank_saved_hash = sha256_file(output_dir / "state_bank.npy")
    state_ids_hash = sha256_file(output_dir / "state_ids.npy")
    crn_draws_hash = sha256_file(output_dir / "crn_draws.npz")
    prefix_contract = {
        "pool_size": int(pool_size),
        "prefix_candidates": [
            int(value)
            for value in (1, 2, 4, 8, 16, 32, 64)
            if value <= pool_size
        ],
        "prefix_index_sha256": prefix_index_hashes(pool_size),
        "state_ids_sha256": state_ids_hash,
    }
    checkpoint_records: list[dict[str, Any]] = []
    with isolated_global_rng():
        for checkpoint in checkpoints:
            checkpoint_started = time.perf_counter()
            checkpoint_warnings: list[dict[str, str]] = []
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                model = _load_hierarchy_checkpoint(
                    checkpoint, config_path, device
                )
                checkpoint_warnings = [
                    {
                        "category": type(item.message).__name__,
                        "message": str(item.message),
                    }
                    for item in caught
                ]
            modules = _model_modules(model)
            mode_snapshot = [(module, module.training) for _, module in modules]
            before = model_state_hashes(model)
            seed_records: list[dict[str, Any]] = []
            try:
                for _, module in modules:
                    module.eval()
                if state_bank.shape[1] != int(model.observation_dim):
                    raise ValueError(
                        f"state dimension {state_bank.shape[1]} != model observation "
                        f"dimension {model.observation_dim}"
                    )
                if int(model.action_dim_flat) != 24:
                    raise ValueError(
                        f"Walker G0 expects latent dimension 24, got {model.action_dim_flat}"
                    )
                observations = torch.as_tensor(
                    state_bank, device=device, dtype=torch.float32
                )
                with torch.no_grad():
                    mean, log_std, kwargs = model.actor.get_action_dist_params(
                        observations
                    )
                if kwargs:
                    raise RuntimeError("WBSD G0 does not support SDE noise actors")
                smoke_call_counts: dict[str, int] | None = None
                for proposal_seed in proposal_seeds:
                    standard_normal = torch.as_tensor(
                        proposal_draws[str(proposal_seed)],
                        device=device,
                        dtype=observations.dtype,
                    )
                    with torch.no_grad():
                        current_pool = current_actor_from_standard_normal(
                            model, observations, standard_normal
                        )
                        prior_pool = decoder_to_scaled(
                            model, standard_normal.reshape(-1, 24)
                        ).reshape_as(standard_normal)
                    checkpoint_tag = f"{int(model.num_timesteps):012d}"
                    pool_path = (
                        output_dir
                        / "pools"
                        / f"checkpoint_{checkpoint_tag}_proposal_{int(proposal_seed)}.npz"
                    )
                    np.savez_compressed(
                        pool_path,
                        state_ids=state_ids,
                        standard_normal=standard_normal.detach().cpu().numpy(),
                        g_current=current_pool.detach().cpu().numpy(),
                        g_prior_exact=prior_pool.detach().cpu().numpy(),
                    )
                    if smoke_call_counts is None:
                        with torch.no_grad():
                            with counted_candidate_calls(model) as counts:
                                evaluated = evaluate_candidate_batch(
                                    model,
                                    observations[:1],
                                    current_pool[:1, :2],
                                )
                        smoke_call_counts = dict(counts)
                        smoke_target_finite = all(
                            bool(torch.isfinite(head).all())
                            for head in evaluated["target_heads"]
                        )
                        smoke_student_finite = all(
                            bool(torch.isfinite(head).all())
                            for head in evaluated["student_heads"]
                        )
                    pool_hashes = {
                        "g_current": tensor_sha256(current_pool),
                        "g_prior_exact": tensor_sha256(prior_pool),
                    }
                    seed_records.append(
                        {
                            "proposal_seed": int(proposal_seed),
                            "standard_normal_sha256": tensor_sha256(
                                standard_normal
                            ),
                            "pool_file": str(pool_path.relative_to(ROOT)),
                            "pool_file_sha256": sha256_file(pool_path),
                            "pool_hashes": pool_hashes,
                            "pool_prefix_hashes": {
                                "g_current": _pool_prefix_hashes(current_pool),
                                "g_prior_exact": _pool_prefix_hashes(prior_pool),
                            },
                            "shared_prefix_index_sha256": prefix_index_hashes(
                                pool_size
                            ),
                            "shared_prefix_ids": _prefix_ids(
                                int(proposal_seed), pool_size
                            ),
                        }
                    )
            finally:
                for module, training in mode_snapshot:
                    module.train(training)
            after = model_state_hashes(model)
            checkpoint_records.append(
                {
                    "checkpoint": str(checkpoint.resolve()),
                    "checkpoint_sha256": sha256_file(checkpoint),
                    "model_num_timesteps": int(model.num_timesteps),
                    "module_hashes_before": before,
                    "module_hashes_after": after,
                    "module_hashes_equal": before == after,
                    "warnings": checkpoint_warnings,
                    "smoke_query_counts": smoke_call_counts,
                    "smoke_metrics_finite": bool(
                        smoke_target_finite and smoke_student_finite
                    ),
                    "proposal_records": seed_records,
                    "elapsed_seconds": float(
                        time.perf_counter() - checkpoint_started
                    ),
                }
            )
    finished_at_utc = utc_now()
    summary: dict[str, Any] = {
        "material_passport": {
            "origin_skill": "academic-research-suite/experiment-agent",
            "origin_mode": "run",
            "verification_status": "EXECUTED_MATRIX",
            "version_label": "wbsd_g0_matrix_v1",
        },
        "gate": "G0",
        "status": "COMPLETE",
        "mode": "checkpoint_crn_matrix_read_only",
        "started_at_utc": started_at_utc,
        "finished_at_utc": finished_at_utc,
        "device": device,
        "state_count": int(state_bank.shape[0]),
        "requested_state_count": int(state_count),
        "pool_size": int(pool_size),
        "proposal_seeds": [int(value) for value in proposal_seeds],
        "state_bank": str(state_bank_path.resolve()),
        "state_bank_source_sha256": sha256_file(state_bank_path),
        "state_bank_saved_sha256": state_bank_saved_hash,
        "state_bank_metadata": state_metadata,
        "state_ids": state_ids.tolist(),
        "state_ids_sha256": state_ids_hash,
        "crn_draws": str((output_dir / "crn_draws.npz").resolve()),
        "crn_draws_sha256": crn_draws_hash,
        "prefix_contract": prefix_contract,
        "config": str(config_path.resolve()),
        "config_sha256": sha256_file(config_path),
        "checkpoints": checkpoint_records,
        "query_contract": {
            "environment_steps": 0,
            "optimizer_steps": 0,
            "matrix_pool_generation_ddim_steps": 0,
            "per_checkpoint_smoke_candidates": 2,
        },
        "source_hashes": source_hashes,
        "git": _git_metadata(),
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
        },
        "elapsed_seconds": float(time.perf_counter() - started),
        "command": sys.argv,
    }
    _write_json(output_dir / "manifest.json", summary)
    _write_json(output_dir / "summary.json", summary)
    (output_dir / "EXECUTION_COMPLETE").write_text(
        "WBSD G0 matrix completed; no optimizer or environment steps.\n",
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    default_run = (
        ROOT
        / "logs/p6/diagnostic_base_gpu_fresh_frozen_ddim_dsrl_na_rfs_hier_2500000chunks_walker_seed1_500k"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matrix",
        action="store_true",
        help="Run the full G0 checkpoint/CRN matrix instead of one smoke batch",
    )
    parser.add_argument("--checkpoint", type=Path, default=default_run / "checkpoints/model_000000100000.zip")
    parser.add_argument(
        "--checkpoints",
        type=Path,
        nargs="+",
        default=[
            default_run / "checkpoints/model_000000100000.zip",
            default_run / "checkpoints/model_000000300000.zip",
            default_run / "checkpoints/model_000000500000.zip",
        ],
    )
    parser.add_argument("--config", type=Path, default=default_run / ".hydra/config.yaml")
    parser.add_argument(
        "--state-bank",
        type=Path,
        default=ROOT / "logs/p6-prefill/walker2d-medium-v2_fresh_frozen_ddim_env3001_policy4001_nenv10_tagged_v1.npz",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/wbsd/G0/g0_smoke_seed1_100k",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--state-count", type=int, default=1)
    parser.add_argument("--candidates", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--pool-size", type=int, default=64)
    parser.add_argument(
        "--proposal-seeds",
        type=int,
        nargs="+",
        default=[1101, 2202, 3303, 4404],
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    if arguments.matrix:
        summary = run_matrix(
            checkpoints=[path.resolve() for path in arguments.checkpoints],
            config_path=arguments.config.resolve(),
            state_bank_path=arguments.state_bank.resolve(),
            output_dir=arguments.output.resolve(),
            device=str(arguments.device),
            state_count=max(512, int(arguments.state_count)),
            pool_size=int(arguments.pool_size),
            proposal_seeds=[int(value) for value in arguments.proposal_seeds],
        )
    else:
        summary = run_smoke(
            checkpoint=arguments.checkpoint.resolve(),
            config_path=arguments.config.resolve(),
            state_bank_path=arguments.state_bank.resolve(),
            output_dir=arguments.output.resolve(),
            device=str(arguments.device),
            state_count=int(arguments.state_count),
            candidates=int(arguments.candidates),
            seed=int(arguments.seed),
        )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
