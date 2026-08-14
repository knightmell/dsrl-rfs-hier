"""Exact fixed-seed evaluation for saved per-step residual PPO checkpoints."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / "dppo"), str(ROOT / "stable-baselines3")]

from p6_preflight import sha256_file
from p6_train import _load_legacy_network
from per_step_residual_env import (
    FrozenDSRLChunkPlanner,
    FrozenDiffusionChunkPlanner,
)
from per_step_residual_ppo import CountingPPO, ResiPAlignedPPO, SeparatedClipPPO
from train_per_step_residual import (
    ACTION_CHUNK,
    ACTION_DIMENSION,
    compose_config,
    make_chunk_contract_environment,
    module_state_hash,
    planner_modules,
)
from train_per_step_residual_ppo import evaluate_exact, make_ppo_environment, write_evaluation
from utils import load_base_policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=10_000)
    parser.add_argument("--policy-seed-start", type=int, default=20_000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--stochastic-residual", action="store_true")
    return parser.parse_args()


def _model_class(training_variant: str) -> type[CountingPPO]:
    if training_variant in (
        "resip_v1",
        "resip_hopper_v1",
        "resip_hopper_long_gae_v1",
    ):
        return ResiPAlignedPPO
    if training_variant == "stable_v1":
        return SeparatedClipPPO
    if training_variant == "legacy":
        return CountingPPO
    raise ValueError(f"Unknown training variant {training_variant!r}")


def _planner_from_manifest(
    manifest: dict[str, Any],
    *,
    cfg: Any,
    diffusion: Any,
    contract: Any,
) -> tuple[Any, dict[str, torch.nn.Module]]:
    source = manifest["base_source"]
    if source == "dsrl_checkpoint":
        checkpoint = Path(manifest["checkpoint_path"]).resolve()
        if sha256_file(checkpoint) != manifest["checkpoint_sha256"]:
            raise ValueError("DSRL checkpoint hash differs from the run manifest")
        cfg.p6.init_checkpoint_sha256 = manifest["checkpoint_sha256"]
        cfg.p6.expected_init_checkpoint_steps = int(manifest["checkpoint_steps"])
        legacy = _load_legacy_network(
            cfg=cfg,
            environment=contract,
            diffusion_policy=diffusion,
            buffer_size=1,
        )
        planner = FrozenDSRLChunkPlanner(
            legacy,
            action_chunk=ACTION_CHUNK,
            action_dimension=ACTION_DIMENSION,
        )
        return planner, planner_modules(legacy)
    if source == "diffusion_prior":
        planner = FrozenDiffusionChunkPlanner(
            diffusion,
            device=cfg.device,
            observation_dimension=int(np.prod(contract.observation_space.shape)),
            action_chunk=ACTION_CHUNK,
            action_dimension=ACTION_DIMENSION,
        )
        modules = {
            name: module
            for name, module in (
                ("diffusion_policy", diffusion),
                ("diffusion_base_policy", getattr(diffusion, "base_policy", None)),
            )
            if isinstance(module, torch.nn.Module)
        }
        return planner, modules
    raise ValueError(f"Unknown base source {source!r}")


def main() -> None:
    arguments = parse_args()
    if arguments.episodes <= 0:
        raise ValueError("episodes must be positive")
    run_directory = arguments.run_dir.resolve()
    manifest = json.loads((run_directory / "run_manifest.json").read_text())
    checkpoint = (
        Path(manifest["checkpoint_path"])
        if manifest.get("checkpoint_path") is not None
        else None
    )
    config_arguments = SimpleNamespace(
        checkpoint=checkpoint,
        equivalent_chunk_budget=int(manifest["equivalent_chunk_budget"]),
        seed=int(manifest["seed"]),
        n_envs=int(manifest["n_envs"]),
        device=arguments.device,
    )
    cfg = compose_config(config_arguments)
    cfg.logdir = str(run_directory)
    if sha256_file(Path(cfg.base_policy_path)) != manifest["ddim_sha256"]:
        raise ValueError("DDIM hash differs from the run manifest")
    if sha256_file(Path(cfg.normalization_path)) != manifest["normalization_sha256"]:
        raise ValueError("Normalization hash differs from the run manifest")

    diffusion = load_base_policy(cfg)
    contract = make_chunk_contract_environment(cfg)
    try:
        planner, modules = _planner_from_manifest(
            manifest,
            cfg=cfg,
            diffusion=diffusion,
            contract=contract,
        )
        planner.assert_frozen()
        before_hash = module_state_hash(modules)
        model = _model_class(manifest["training_variant"]).load(
            arguments.model_checkpoint.resolve(),
            device=arguments.device,
        )
        equivalent_chunks = int(model.num_timesteps) // ACTION_CHUNK
        make_environment = lambda: make_ppo_environment(
            cfg,
            planner,
            residual_scale=float(manifest["residual_scale"]),
            deterministic_base=bool(manifest["deterministic_base"]),
        )
        result = evaluate_exact(
            model=model,
            make_environment=make_environment,
            seeds=list(
                range(
                    int(arguments.seed_start),
                    int(arguments.seed_start) + int(arguments.episodes),
                )
            ),
            policy_seed_start=int(arguments.policy_seed_start),
            equivalent_chunks=equivalent_chunks,
            deterministic=not arguments.stochastic_residual,
        )
        result.update(
            {
                "source_run_manifest": str(run_directory / "run_manifest.json"),
                "model_checkpoint": str(arguments.model_checkpoint.resolve()),
                "base_source": manifest["base_source"],
                "base_deterministic": bool(manifest["deterministic_base"]),
                "residual_deterministic": not arguments.stochastic_residual,
                "planner_state_unchanged": module_state_hash(modules) == before_hash,
            }
        )
        output = arguments.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        write_evaluation(output, result)
        print(json.dumps(result["summary"], indent=2, sort_keys=True))
    finally:
        contract.close()


if __name__ == "__main__":
    main()
