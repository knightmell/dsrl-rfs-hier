"""Exact-seeded evaluation of an independent legacy DSRL-NA checkpoint.

This is intentionally a thin CLI over the production P6 loader, locomotion
environment constructor, and exact-N evaluator.  It never trains or mutates a
checkpoint.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import hydra
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict

from p6_evaluation import evaluate_exact_episodes, persist_evaluation
from p6_preflight import sha256_file
from p6_train import _load_legacy_network, _make_locomotion_environment
from stable_baselines3.common.env_util import make_vec_env
from utils import load_base_policy


ROOT = Path(__file__).resolve().parent
SUPPORTED_CONFIGS = {
    "p6_hopper": "dsrl_hopper",
    "p6_halfcheetah": "dsrl_halfcheetah",
    "p6_walker": "dsrl_walker",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", choices=tuple(SUPPORTED_CONFIGS), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-steps", type=int, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--env-seed-start", type=int, default=10_000)
    parser.add_argument("--policy-seed-start", type=int, default=20_000)
    parser.add_argument("--model-seed", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--output-prefix", type=Path, required=True)
    return parser.parse_args()


def _load_config(args: argparse.Namespace) -> Any:
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    OmegaConf.register_new_resolver("round_up", math.ceil, replace=True)
    OmegaConf.register_new_resolver("round_down", math.floor, replace=True)
    with initialize_config_dir(config_dir=str(ROOT / "cfg" / "gym"), version_base=None):
        cfg = compose(
            config_name=SUPPORTED_CONFIGS[args.config_name],
            overrides=[
                "algorithm=dsrl_na",
                f"device={args.device}",
                "use_wandb=false",
                f"seed={args.model_seed}",
            ],
        )
    OmegaConf.resolve(cfg)
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
    if args.expected_steps <= 0 or args.episodes <= 0 or args.batch_size <= 0:
        raise ValueError("expected-steps, episodes, and batch-size must be positive")
    with open_dict(cfg):
        cfg.rfs_hier_legacy_checkpoint_path = str(checkpoint)
        cfg.logdir = str(args.output_prefix.expanduser().resolve().parent)
        cfg.p6 = {"expected_init_checkpoint_steps": int(args.expected_steps)}
    return cfg


def main() -> None:
    args = parse_args()
    cfg = _load_config(args)
    normalization_path = Path(hydra.utils.to_absolute_path(str(cfg.normalization_path)))
    diffusion_policy = load_base_policy(cfg)
    make_environment = lambda: _make_locomotion_environment(cfg, normalization_path)
    load_environment = make_vec_env(make_environment, n_envs=1)
    try:
        model = _load_legacy_network(
            cfg=cfg,
            environment=load_environment,
            diffusion_policy=diffusion_policy,
            buffer_size=1,
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
            evaluation_mode="matched_dsrl",
            matched_dsrl_model=model,
        )
        checkpoint = args.checkpoint.expanduser().resolve()
        result["checkpoint_provenance"] = {
            "path": str(checkpoint),
            "sha256": sha256_file(checkpoint),
            "num_timesteps": int(model.num_timesteps),
            "config_name": args.config_name,
            "environment": str(cfg.env_name),
            "frozen_ddim_path": str(
                Path(hydra.utils.to_absolute_path(str(cfg.model.network_path))).resolve()
            ),
            "frozen_ddim_sha256": sha256_file(
                Path(hydra.utils.to_absolute_path(str(cfg.model.network_path))).resolve()
            ),
            "normalization_path": str(normalization_path.resolve()),
            "normalization_sha256": sha256_file(normalization_path.resolve()),
            "training_actual_primitive_env_steps": None,
            "training_actual_primitive_env_steps_status": "not_recorded_by_legacy_run",
        }
        persist_evaluation(result, output_prefix=args.output_prefix.expanduser().resolve())
        summary = result["summary"]
        print(
            f"{cfg.env_name}: raw={summary['raw_return_mean']:.6f}, "
            f"D4RL={summary['d4rl_score_mean']:.6f}, "
            f"early_fall={summary['early_fall_rate']:.6f}"
        )
    finally:
        load_environment.close()


if __name__ == "__main__":
    main()
