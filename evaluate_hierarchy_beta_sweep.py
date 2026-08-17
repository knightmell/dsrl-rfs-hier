"""Inference-only exact-seeded beta sweep for a trained VS-Hier checkpoint."""

from __future__ import annotations

import argparse
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from p6_evaluation import evaluate_exact_episodes, persist_evaluation
from p6_preflight import sha256_file
from p6_train import _make_locomotion_environment
from stable_baselines3.dsrl.hierarchical_rfs_dsrl import HierarchicalRFSDSRL
from utils import load_base_policy


ROOT = Path(__file__).resolve().parent
SUPPORTED_CONFIGS = (
    "p6_hopper_fresh_2p5m_cotrain",
    "p6_halfcheetah_fresh_2p5m_cotrain",
    "p6_walker_fresh_2p5m_cotrain",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", choices=SUPPORTED_CONFIGS, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-steps", type=int, default=2_500_000)
    parser.add_argument(
        "--beta-values",
        type=float,
        nargs="+",
        default=(0.0, 0.025, 0.05, 0.1),
    )
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--env-seed-start", type=int, default=10_000)
    parser.add_argument("--policy-seed-start", type=int, default=20_000)
    parser.add_argument("--model-seed", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _load_config(args: argparse.Namespace) -> Any:
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    OmegaConf.register_new_resolver("round_up", math.ceil, replace=True)
    OmegaConf.register_new_resolver("round_down", math.floor, replace=True)
    with initialize_config_dir(config_dir=str(ROOT / "cfg" / "gym"), version_base=None):
        cfg = compose(
            config_name=args.config_name,
            overrides=[f"seed={args.model_seed}", f"device={args.device}"],
        )
    OmegaConf.resolve(cfg)
    return cfg


def _beta_label(beta: float) -> str:
    return f"beta_{beta:.6f}".rstrip("0").rstrip(".").replace(".", "p")


def main() -> None:
    args = parse_args()
    if args.expected_steps <= 0 or args.episodes <= 0 or args.batch_size <= 0:
        raise ValueError("expected-steps, episodes, and batch-size must be positive")
    if len(set(args.beta_values)) != len(args.beta_values):
        raise ValueError("beta-values must be unique")
    if any(not 0.0 <= beta <= 1.0 for beta in args.beta_values):
        raise ValueError("every beta value must lie in [0, 1]")
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
    output_directory = args.output_dir.expanduser().resolve()
    if output_directory.exists() and any(output_directory.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty {output_directory}")
    output_directory.mkdir(parents=True, exist_ok=True)

    cfg = _load_config(args)
    normalization_path = Path(str(cfg.normalization_path))
    if not normalization_path.is_absolute():
        normalization_path = (ROOT / normalization_path).resolve()
    diffusion_policy = load_base_policy(cfg)
    model = HierarchicalRFSDSRL.load(
        str(checkpoint),
        device=args.device,
        diffusion_policy=diffusion_policy,
    )
    if int(model.num_timesteps) != args.expected_steps:
        raise ValueError(
            f"Checkpoint step mismatch: {model.num_timesteps} != {args.expected_steps}"
        )
    make_environment = lambda: _make_locomotion_environment(cfg, normalization_path)
    original_schedule = model.hierarchy_schedule
    try:
        for beta in args.beta_values:
            model.hierarchy_schedule = replace(
                original_schedule,
                beta_target=float(beta),
                beta_floor=min(float(original_schedule.beta_floor), float(beta)),
            )
            if model.current_beta != float(beta):
                raise RuntimeError(
                    f"Inference beta override failed: {model.current_beta} != {beta}"
                )
            result = evaluate_exact_episodes(
                model=model,
                make_environment=make_environment,
                environment_seeds=list(
                    range(args.env_seed_start, args.env_seed_start + args.episodes)
                ),
                policy_seed_start=args.policy_seed_start,
                deterministic=args.deterministic,
                action_chunk=int(cfg.act_steps),
                max_episode_primitive_steps=int(cfg.env.max_episode_steps),
                batch_size=args.batch_size,
                chunk_transitions=args.expected_steps,
                nominal_primitive_steps=args.expected_steps * int(cfg.act_steps),
                actual_primitive_env_steps=0,
                evaluation_mode="current_full_hierarchy",
            )
            result["beta_override"] = float(beta)
            result["checkpoint_provenance"] = {
                "path": str(checkpoint),
                "sha256": sha256_file(checkpoint),
                "num_timesteps": int(model.num_timesteps),
                "config_name": args.config_name,
                "environment": str(cfg.env_name),
                "frozen_ddim_path": str(
                    Path(str(cfg.model.network_path)).expanduser().resolve()
                ),
                "frozen_ddim_sha256": sha256_file(
                    Path(str(cfg.model.network_path)).expanduser().resolve()
                ),
                "normalization_path": str(normalization_path),
                "normalization_sha256": sha256_file(normalization_path),
                "training_actual_primitive_env_steps": None,
                "training_actual_primitive_env_steps_status": (
                    "available_in_original_run_manifest_not_rewritten_here"
                ),
            }
            prefix = output_directory / _beta_label(float(beta))
            persist_evaluation(result, output_prefix=prefix)
            summary = result["summary"]
            print(
                f"{cfg.env_name} beta={beta:g}: "
                f"raw={summary['raw_return_mean']:.6f}, "
                f"D4RL={summary['d4rl_score_mean']:.6f}, "
                f"early_fall={summary['early_fall_rate']:.6f}"
            )
    finally:
        model.hierarchy_schedule = original_schedule


if __name__ == "__main__":
    main()
