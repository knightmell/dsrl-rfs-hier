"""Read-only matched DSRL-NA QW/ranking mirror for Walker diagnosis.

The mirror evaluates the QW learned by the matched flat DSRL control against
both its native online-QA teacher and the target QA.  It never constructs an
environment, calls backward, or steps an optimizer.  Four proposal-seed
workers are intended to run concurrently and are aggregated with state as the
independent unit.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn

from wbsd_g1 import paired_state_bootstrap_ci, state_metrics
from wbsd_probe import (
    ROOT,
    _model_modules,
    _write_csv,
    _write_json,
    isolated_global_rng,
    load_state_bank,
    model_state_hashes,
    sha256_array,
    sha256_file,
    utc_now,
)


METHOD_ACTOR_LOCAL = "actor_local"
METHOD_GAUSSIAN_NATIVE = "gaussian_native"
METHODS = (METHOD_ACTOR_LOCAL, METHOD_GAUSSIAN_NATIVE)
TEACHER_NATIVE_ONLINE = "native_online_qa"
TEACHER_TARGET = "target_qa"
TEACHERS = (TEACHER_NATIVE_ONLINE, TEACHER_TARGET)
PREFIXES = (1, 4, 16, 64)
MODEL_FAMILY = "matched_dsrl_na"
VIEW = "fixed_state_mirror"


class DSRLMirrorAdapter:
    """Expose flat DSRL modules with explicit mirror semantics."""

    def __init__(self, model: Any):
        self.model = model
        self.actor = model.actor
        self.policy = model.policy
        self.qa_native = model.critic
        self.qa_target = model.critic_target
        self.qw = model.critic_noise
        self.diffusion_policy = model.diffusion_policy
        self.diffusion_act_chunk = int(model.diffusion_act_chunk)
        self.diffusion_act_dim = int(model.diffusion_act_dim)
        self.action_dim_flat = self.diffusion_act_chunk * self.diffusion_act_dim

    def decoder_from_scaled(self, noise_scaled: torch.Tensor) -> torch.Tensor:
        values = self.policy.unscale_action(
            noise_scaled.detach().cpu().numpy()
        )
        return torch.as_tensor(
            np.asarray(values),
            device=noise_scaled.device,
            dtype=noise_scaled.dtype,
        ).reshape(-1, self.diffusion_act_chunk, self.diffusion_act_dim)

    @torch.no_grad()
    def decode(self, observations: torch.Tensor, noise_scaled: torch.Tensor) -> torch.Tensor:
        decoder = self.decoder_from_scaled(noise_scaled)
        actions = self.diffusion_policy(
            observations, decoder, return_numpy=False
        )
        if isinstance(actions, np.ndarray):
            actions = torch.as_tensor(
                actions, device=observations.device, dtype=observations.dtype
            )
        if not isinstance(actions, torch.Tensor):
            raise TypeError("diffusion_policy must return Tensor or ndarray")
        return actions.reshape(observations.shape[0], -1).detach()


def _module_device(adapter: DSRLMirrorAdapter) -> torch.device:
    for module in (adapter.qa_native, adapter.qw):
        parameter = next(module.parameters(), None)
        if parameter is not None:
            return parameter.device
    return torch.device("cpu")


def build_candidate_pools(
    adapter: DSRLMirrorAdapter,
    observations: np.ndarray,
    *,
    pool_size: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Build deterministic actor-local and exact Gaussian-native pools."""

    observations = np.asarray(observations, dtype=np.float32)
    if observations.ndim != 2 or pool_size <= 0:
        raise ValueError("observations must be [B,D] and pool_size positive")
    device = _module_device(adapter)
    obs = torch.as_tensor(observations, device=device, dtype=torch.float32)
    generator = torch.Generator(device=device).manual_seed(int(seed))
    with torch.no_grad():
        mean, log_std, kwargs = adapter.actor.get_action_dist_params(obs)
        if kwargs:
            raise RuntimeError("DSRL mirror does not support SDE actors")
        actor_z = torch.randn(
            obs.shape[0],
            int(pool_size),
            adapter.action_dim_flat,
            generator=generator,
            device=device,
            dtype=obs.dtype,
        )
        actor_pool = torch.tanh(
            mean.unsqueeze(1) + log_std.exp().unsqueeze(1) * actor_z
        )
        gaussian_decoder = torch.randn(
            obs.shape[0],
            int(pool_size),
            adapter.action_dim_flat,
            generator=generator,
            device=device,
            dtype=obs.dtype,
        )
        scaled = adapter.policy.scale_action(
            gaussian_decoder.reshape(-1, adapter.action_dim_flat)
            .cpu()
            .numpy()
        )
        gaussian_pool = torch.as_tensor(
            np.asarray(scaled), device=device, dtype=obs.dtype
        ).reshape_as(gaussian_decoder)
    return {
        METHOD_ACTOR_LOCAL: np.ascontiguousarray(actor_pool.cpu().numpy()),
        METHOD_GAUSSIAN_NATIVE: np.ascontiguousarray(
            gaussian_pool.cpu().numpy()
        ),
    }


def _stack_heads(
    heads: Iterable[torch.Tensor], states: int, candidates: int
) -> np.ndarray:
    values = tuple(heads)
    if len(values) < 2:
        raise ValueError("mirror requires twin critics")
    return (
        torch.stack([head.reshape(-1) for head in values], dim=0)
        .reshape(len(values), states, candidates)
        .cpu()
        .numpy()
    )


def evaluate_mirror_pool(
    adapter: DSRLMirrorAdapter,
    observations: np.ndarray,
    pool: np.ndarray,
    *,
    device: str,
    batch_candidates: int,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """Decode and evaluate one source pool against both QA teachers and QW."""

    observations = np.asarray(observations, dtype=np.float32)
    pool = np.asarray(pool, dtype=np.float32)
    states, candidates, latent_dim = pool.shape
    if latent_dim != adapter.action_dim_flat or observations.shape[0] != states:
        raise ValueError("state/pool dimensions are incompatible with DSRL")
    states_per_batch = max(1, int(batch_candidates) // candidates)
    parts: dict[str, list[np.ndarray]] = defaultdict(list)
    counts = {
        "ddim_decode_candidates": 0,
        "qa_online_candidates": 0,
        "qa_target_candidates": 0,
        "qw_candidates": 0,
    }
    with torch.no_grad():
        for start in range(0, states, states_per_batch):
            stop = min(states, start + states_per_batch)
            local_states = stop - start
            obs = torch.as_tensor(
                observations[start:stop], device=device, dtype=torch.float32
            )
            latent = torch.as_tensor(
                pool[start:stop], device=device, dtype=torch.float32
            )
            flat_obs = obs.repeat_interleave(candidates, dim=0)
            flat_latent = latent.reshape(-1, latent_dim)
            actions = adapter.decode(flat_obs, flat_latent)
            online = adapter.qa_native(flat_obs, actions)
            target = adapter.qa_target(flat_obs, actions)
            qw = adapter.qw(flat_obs, flat_latent)
            parts["actions"].append(
                actions.reshape(local_states, candidates, -1).cpu().numpy()
            )
            parts[TEACHER_NATIVE_ONLINE].append(
                _stack_heads(online, local_states, candidates)
            )
            parts[TEACHER_TARGET].append(
                _stack_heads(target, local_states, candidates)
            )
            parts["qw"].append(_stack_heads(qw, local_states, candidates))
            queried = local_states * candidates
            for key in counts:
                counts[key] += queried
    return {
        "actions": np.ascontiguousarray(np.concatenate(parts["actions"], axis=0)),
        TEACHER_NATIVE_ONLINE: np.ascontiguousarray(
            np.concatenate(parts[TEACHER_NATIVE_ONLINE], axis=1)
        ),
        TEACHER_TARGET: np.ascontiguousarray(
            np.concatenate(parts[TEACHER_TARGET], axis=1)
        ),
        "qw": np.ascontiguousarray(np.concatenate(parts["qw"], axis=1)),
    }, counts


def mirror_state_metrics(
    *,
    checkpoint_step: int,
    proposal_seed: int,
    method: str,
    teacher_kind: str,
    state_ids: np.ndarray,
    noise: np.ndarray,
    actions: np.ndarray,
    teacher_heads: np.ndarray,
    student_heads: np.ndarray,
    prefixes: Iterable[int],
) -> list[dict[str, Any]]:
    if method not in METHODS or teacher_kind not in TEACHERS:
        raise ValueError("unknown mirror method or teacher")
    tanh_generated = np.full(
        noise.shape[1], method == METHOD_ACTOR_LOCAL, dtype=bool
    )
    rows = state_metrics(
        checkpoint_step=checkpoint_step,
        proposal_seed=proposal_seed,
        method=method,
        view=VIEW,
        state_ids=state_ids,
        noise=noise,
        actions=actions,
        teacher_heads=teacher_heads,
        student_heads=student_heads,
        prefixes=prefixes,
        tanh_generated=tanh_generated,
    )
    for row in rows:
        row["model_family"] = MODEL_FAMILY
        row["teacher_kind"] = teacher_kind
        row["direct_selection_eligible"] = int(method == METHOD_ACTOR_LOCAL)
    return rows


def load_dsrl_checkpoint(
    checkpoint: Path, config_path: Path, device: str
) -> DSRLMirrorAdapter:
    """Load one P6 matched-control checkpoint without an environment."""

    sys.path[:0] = [str(ROOT / "dppo"), str(ROOT / "stable-baselines3")]
    from omegaconf import OmegaConf  # pylint: disable=import-outside-toplevel
    from utils import load_base_policy  # pylint: disable=import-outside-toplevel
    from p6_train import P6ControlDSRL  # pylint: disable=import-outside-toplevel

    config = OmegaConf.load(config_path)
    base_policy = Path(str(config.base_policy_path)).resolve()
    if not base_policy.is_file():
        raise FileNotFoundError(f"Frozen DDIM checkpoint not found: {base_policy}")
    OmegaConf.update(config, "device", device, merge=False)
    OmegaConf.update(config, "model.device", device, merge=False)
    OmegaConf.update(config, "base_policy_path", str(base_policy), merge=False)
    OmegaConf.update(config, "model.network_path", str(base_policy), merge=False)
    diffusion_policy = load_base_policy(config)
    model = P6ControlDSRL.load(
        str(checkpoint),
        env=None,
        device=device,
        custom_objects={"diffusion_policy": diffusion_policy},
    )
    return DSRLMirrorAdapter(model)


def _counter_snapshot(model: Any) -> dict[str, int]:
    names = (
        "num_timesteps",
        "_n_updates",
        "action_critic_optimizer_steps",
        "modulation_critic_optimizer_steps",
        "noise_actor_optimizer_steps",
        "residual_actor_optimizer_steps",
        "hierarchy_train_calls",
    )
    return {name: int(getattr(model, name, 0)) for name in names}


def run_worker(
    *,
    checkpoints: list[Path],
    config_path: Path,
    state_bank_path: Path,
    output_dir: Path,
    device: str,
    proposal_seed: int,
    state_count: int = 512,
    pool_size: int = 64,
    batch_candidates: int = 32768,
) -> dict[str, Any]:
    """Run one auditable proposal-seed worker across all checkpoints."""

    if pool_size < max(PREFIXES):
        raise ValueError(f"pool_size must be at least {max(PREFIXES)}")
    started_at = utc_now()
    output_dir.mkdir(parents=True, exist_ok=True)
    state_bank, state_meta = load_state_bank(state_bank_path, state_count)
    state_ids = np.arange(len(state_bank), dtype=np.int64)
    rows: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    global_started = time.perf_counter()
    with isolated_global_rng():
        for checkpoint in checkpoints:
            checkpoint_started = time.perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                adapter = load_dsrl_checkpoint(checkpoint, config_path, device)
            model = adapter.model
            modules = _model_modules(model)
            modes = [(module, module.training) for _, module in modules]
            for _, module in modules:
                module.eval()
            hashes_before = model_state_hashes(model)
            counters_before = _counter_snapshot(model)
            checkpoint_rows: list[dict[str, Any]] = []
            pools = build_candidate_pools(
                adapter,
                state_bank,
                pool_size=pool_size,
                seed=proposal_seed,
            )
            pool_records: dict[str, Any] = {}
            try:
                for method, pool in pools.items():
                    result, counts = evaluate_mirror_pool(
                        adapter,
                        state_bank,
                        pool,
                        device=device,
                        batch_candidates=batch_candidates,
                    )
                    for teacher_kind in TEACHERS:
                        checkpoint_rows.extend(
                            mirror_state_metrics(
                                checkpoint_step=int(model.num_timesteps),
                                proposal_seed=proposal_seed,
                                method=method,
                                teacher_kind=teacher_kind,
                                state_ids=state_ids,
                                noise=pool,
                                actions=result["actions"],
                                teacher_heads=result[teacher_kind],
                                student_heads=result["qw"],
                                prefixes=PREFIXES,
                            )
                        )
                    pool_records[method] = {
                        "pool_sha256": sha256_array(pool),
                        "query_counts": counts,
                    }
            finally:
                for module, training in modes:
                    module.train(training)
            hashes_after = model_state_hashes(model)
            counters_after = _counter_snapshot(model)
            if hashes_before != hashes_after or counters_before != counters_after:
                raise RuntimeError("read-only mirror changed model state or counters")
            if not all(int(row["finite"]) == 1 for row in checkpoint_rows):
                raise FloatingPointError("non-finite DSRL mirror metrics")
            rows.extend(checkpoint_rows)
            records.append(
                {
                    "checkpoint_step": int(model.num_timesteps),
                    "checkpoint": str(checkpoint.resolve()),
                    "checkpoint_sha256": sha256_file(checkpoint),
                    "pool_records": pool_records,
                    "module_hashes_equal": True,
                    "training_counters_equal": True,
                    "training_counters": counters_before,
                    "warnings": [
                        {
                            "category": type(item.message).__name__,
                            "message": str(item.message),
                        }
                        for item in caught
                    ],
                    "elapsed_seconds": time.perf_counter() - checkpoint_started,
                }
            )
            del adapter, model
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
    _write_csv(output_dir / "state_metrics.csv", rows)
    summary = {
        "mode": "matched_dsrl_na_fixed_state_qw_ranking_mirror",
        "status": "complete",
        "started_at_utc": started_at,
        "completed_at_utc": utc_now(),
        "proposal_seed": int(proposal_seed),
        "device": device,
        "state_bank": str(state_bank_path.resolve()),
        "state_bank_sha256": sha256_file(state_bank_path),
        "state_bank_selected_sha256": state_meta["selected_sha256"],
        "state_count": len(state_bank),
        "pool_size": int(pool_size),
        "prefixes": list(PREFIXES),
        "methods": list(METHODS),
        "teachers": list(TEACHERS),
        "rows": len(rows),
        "checkpoints": records,
        "mutations": {
            "environment_steps": 0,
            "optimizer_steps": 0,
            "backward_calls": 0,
        },
        "elapsed_seconds": time.perf_counter() - global_started,
    }
    _write_json(output_dir / "manifest.json", summary)
    _write_json(output_dir / "summary.json", summary)
    (output_dir / "EXECUTION_COMPLETE").write_text(
        "Matched DSRL-NA mirror completed read-only.\n", encoding="utf-8"
    )
    return summary


def _read_rows(path: Path) -> list[dict[str, Any]]:
    integer_keys = {
        "checkpoint_step",
        "proposal_seed",
        "state_id",
        "candidate_count",
        "finite",
    }
    text_keys = {"method", "view", "model_family", "teacher_kind"}
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for key, value in tuple(row.items()):
            if key in text_keys:
                continue
            if key in integer_keys:
                row[key] = int(value)
            else:
                row[key] = float(value)
    return rows


def aggregate_workers(
    worker_dirs: list[Path], output_dir: Path, *, bootstrap_seed: int = 20260830
) -> dict[str, Any]:
    """Aggregate repeated proposal seeds with state as the sampling unit."""

    output_dir.mkdir(parents=True, exist_ok=True)
    manifests = [json.loads((path / "manifest.json").read_text()) for path in worker_dirs]
    selected_hashes = {item["state_bank_selected_sha256"] for item in manifests}
    if len(selected_hashes) != 1:
        raise ValueError("worker state-bank hashes differ")
    all_rows = [row for path in worker_dirs for row in _read_rows(path / "state_metrics.csv")]
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in all_rows:
        key = (
            row["checkpoint_step"],
            row["method"],
            row["teacher_kind"],
            row["candidate_count"],
        )
        grouped[key].append(row)
    metrics = (
        "oracle_lift",
        "selected_lift",
        "selector_capture_raw",
        "qw_qa_exploitation_gap",
        "qw_pairwise_accuracy",
        "qw_spearman",
        "qw_top1_agreement",
        "teacher_twin_disagreement_median",
    )
    aggregate_rows: list[dict[str, Any]] = []
    for key, rows in sorted(grouped.items()):
        state_ids = sorted({int(row["state_id"]) for row in rows})
        proposal_seeds = sorted({int(row["proposal_seed"]) for row in rows})
        record: dict[str, Any] = {
            "checkpoint_step": key[0],
            "method": key[1],
            "teacher_kind": key[2],
            "candidate_count": key[3],
            "state_count": len(state_ids),
            "proposal_seed_count": len(proposal_seeds),
        }
        for metric in metrics:
            lookup = {
                (int(row["state_id"]), int(row["proposal_seed"])): float(row[metric])
                for row in rows
            }
            matrix = np.asarray(
                [[lookup[(state, seed)] for seed in proposal_seeds] for state in state_ids],
                dtype=np.float64,
            )
            state_means = matrix.mean(axis=1)
            lower, upper = paired_state_bootstrap_ci(
                matrix,
                repetitions=2000,
                seed=int(bootstrap_seed + len(aggregate_rows) * 97),
                statistic="mean",
            )
            record[f"{metric}_mean"] = float(state_means.mean())
            record[f"{metric}_median"] = float(np.median(state_means))
            record[f"{metric}_ci95_low"] = float(lower)
            record[f"{metric}_ci95_high"] = float(upper)
        aggregate_rows.append(record)
    _write_csv(output_dir / "aggregate_metrics.csv", aggregate_rows)
    summary = {
        "mode": "matched_dsrl_na_mirror_aggregate",
        "status": "complete",
        "worker_count": len(worker_dirs),
        "proposal_seeds": sorted(item["proposal_seed"] for item in manifests),
        "state_bank_selected_sha256": next(iter(selected_hashes)),
        "aggregate_groups": len(aggregate_rows),
        "worker_manifests": [str((path / "manifest.json").resolve()) for path in worker_dirs],
    }
    _write_json(output_dir / "manifest.json", summary)
    _write_json(output_dir / "summary.json", summary)
    (output_dir / "EXECUTION_COMPLETE").write_text(
        "Matched DSRL-NA mirror aggregation completed.\n", encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    run = ROOT / "logs/p6/fresh_frozen_ddim_control_walker2d-medium-v2_dsrl_na_seed2_2500000chunks"
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    worker = subparsers.add_parser("worker")
    worker.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    worker.add_argument(
        "--config", type=Path, default=run / "attempts/0000/resolved_config.json"
    )
    worker.add_argument(
        "--state-bank",
        type=Path,
        default=ROOT / "logs/p6-prefill/walker2d-medium-v2_fresh_frozen_ddim_env3002_policy4002_nenv10_tagged_v1.npz",
    )
    worker.add_argument("--output", type=Path, required=True)
    worker.add_argument("--device", required=True)
    worker.add_argument("--proposal-seed", type=int, required=True)
    worker.add_argument("--state-count", type=int, default=512)
    worker.add_argument("--pool-size", type=int, default=64)
    worker.add_argument("--batch-candidates", type=int, default=32768)
    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--workers", type=Path, nargs="+", required=True)
    aggregate.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "worker":
        result = run_worker(
            checkpoints=[path.resolve() for path in args.checkpoints],
            config_path=args.config.resolve(),
            state_bank_path=args.state_bank.resolve(),
            output_dir=args.output.resolve(),
            device=str(args.device),
            proposal_seed=int(args.proposal_seed),
            state_count=int(args.state_count),
            pool_size=int(args.pool_size),
            batch_candidates=int(args.batch_candidates),
        )
    else:
        result = aggregate_workers(
            [path.resolve() for path in args.workers], args.output.resolve()
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
