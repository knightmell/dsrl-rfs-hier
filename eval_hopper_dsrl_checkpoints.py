"""Evaluate legacy Hopper DSRL checkpoints under fixed initial states."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import sys
from pathlib import Path

import gym
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "dppo"))

import d4rl  # noqa: E402
import d4rl.gym_mujoco  # noqa: E402,F401

from env_utils import ActionChunkWrapper, ObservationWrapperGym  # noqa: E402
from stable_baselines3.dsrl.hierarchical_rfs_dsrl import (  # noqa: E402
    _LegacyLoadableDSRL,
)
from utils import load_base_policy  # noqa: E402


DEFAULT_CHECKPOINTS = (
    ROOT
    / "logs/gym-dsrl/gym_hopper_dsrl_2026-07-23_21-21-02_1"
    / "2026-07-23_21-21-02_1/checkpoint/ft_policy_2500000_steps.zip",
    ROOT
    / "logs/gym-dsrl/gym_hopper_dsrl_2026-07-23_21-21-02_1"
    / "2026-07-23_21-21-02_1/checkpoint/ft_policy_5000000_steps.zip",
    ROOT
    / "logs/gym-dsrl/gym_hopper_dsrl_2026-07-23_21-21-02_1"
    / "2026-07-23_21-21-02_1/checkpoint/ft_policy_7500000_steps.zip",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        type=Path,
        default=list(DEFAULT_CHECKPOINTS),
    )
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--seed-start", type=int, default=10_000)
    parser.add_argument("--policy-seed", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("deterministic", "stochastic"),
        default=("deterministic", "stochastic"),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_config(device: str):
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    OmegaConf.register_new_resolver("round_up", math.ceil, replace=True)
    OmegaConf.register_new_resolver("round_down", math.floor, replace=True)
    with initialize_config_dir(
        config_dir=str(ROOT / "cfg" / "gym"),
        version_base=None,
    ):
        cfg = compose(
            config_name="dsrl_hopper",
            overrides=[
                "algorithm=dsrl_na",
                f"device={device}",
                "use_wandb=false",
            ],
        )
    OmegaConf.resolve(cfg)
    if not cfg.model.use_ddim or cfg.model.ddim_steps != 5:
        raise ValueError("Checkpoint evaluation requires the audited DDIM5 decoder")
    return cfg


def make_env(cfg):
    raw_env = gym.make(cfg.env_name)
    normalized_env = ObservationWrapperGym(
        raw_env,
        str((ROOT / cfg.normalization_path).resolve()),
    )
    env = ActionChunkWrapper(
        normalized_env,
        cfg,
        max_episode_steps=cfg.env.max_episode_steps,
    )
    return env, raw_env


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as checkpoint_file:
        for block in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_step(path: Path) -> int:
    match = re.search(r"ft_policy_(\d+)_steps\.zip$", path.name)
    if match is None:
        raise ValueError(f"Cannot infer checkpoint step from {path}")
    return int(match.group(1))


def reset_with_explicit_seed(env, raw_env, seed: int) -> np.ndarray:
    # ObservationWrapperGym currently ignores reset(seed=seed), so seed the
    # underlying legacy Gym environment explicitly for reproducible evaluation.
    raw_env.seed(seed)
    observation, _ = env.reset()
    return np.asarray(observation, dtype=np.float32)


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=0)),
        "median": float(np.median(array)),
        "min": float(array.min()),
        "max": float(array.max()),
        "stderr": float(array.std(ddof=1) / np.sqrt(array.size))
        if array.size > 1
        else 0.0,
    }


def evaluate_mode(
    model,
    cfg,
    seeds: list[int],
    batch_size: int,
    deterministic: bool,
    policy_seed: int,
) -> dict:
    random.seed(policy_seed)
    np.random.seed(policy_seed)
    torch.manual_seed(policy_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(policy_seed)

    returns: list[float] = []
    normalized_scores: list[float] = []
    lengths: list[int] = []
    max_chunk_steps = int(cfg.env.max_episode_steps / cfg.act_steps)

    for batch_start in range(0, len(seeds), batch_size):
        batch_seeds = seeds[batch_start : batch_start + batch_size]
        env_pairs = [make_env(cfg) for _ in batch_seeds]
        try:
            observations = np.stack(
                [
                    reset_with_explicit_seed(env, raw_env, seed)
                    for (env, raw_env), seed in zip(env_pairs, batch_seeds)
                ]
            )
            active = np.ones(len(batch_seeds), dtype=bool)
            batch_returns = np.zeros(len(batch_seeds), dtype=np.float64)
            batch_lengths = np.zeros(len(batch_seeds), dtype=np.int64)

            for _ in range(max_chunk_steps):
                action_exec, _ = model.predict_diffused(
                    observations,
                    deterministic=deterministic,
                )
                for index, ((env, _), action) in enumerate(
                    zip(env_pairs, action_exec)
                ):
                    if not active[index]:
                        continue
                    (
                        next_observation,
                        reward,
                        terminated,
                        truncated,
                        _,
                    ) = env.step(action)
                    observations[index] = next_observation
                    batch_returns[index] += float(reward)
                    batch_lengths[index] += cfg.act_steps
                    if terminated or truncated:
                        active[index] = False
                if not active.any():
                    break

            returns.extend(batch_returns.tolist())
            lengths.extend(batch_lengths.tolist())
            normalized_scores.extend(
                (
                    100.0
                    * np.asarray(
                        [
                            d4rl.get_normalized_score(cfg.env_name, score)
                            for score in batch_returns
                        ],
                        dtype=np.float64,
                    )
                ).tolist()
            )
        finally:
            for env, _ in env_pairs:
                env.close()

    return {
        "deterministic": deterministic,
        "episode_count": len(returns),
        "seeds": seeds,
        "return": summarize(returns),
        "normalized_score": summarize(normalized_scores),
        "episode_length": summarize([float(length) for length in lengths]),
        "raw_returns": returns,
        "raw_normalized_scores": normalized_scores,
        "raw_episode_lengths": lengths,
    }


def write_results(output_path: Path, results: dict) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    temporary_path.replace(output_path)

    csv_path = output_path.with_suffix(".csv")
    with csv_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=(
                "checkpoint_step",
                "mode",
                "episodes",
                "return_mean",
                "return_std",
                "return_stderr",
                "normalized_score_mean",
                "normalized_score_std",
                "normalized_score_stderr",
                "episode_length_mean",
            ),
        )
        writer.writeheader()
        for checkpoint in results["checkpoints"]:
            for mode, evaluation in checkpoint["evaluations"].items():
                writer.writerow(
                    {
                        "checkpoint_step": checkpoint["checkpoint_step"],
                        "mode": mode,
                        "episodes": evaluation["episode_count"],
                        "return_mean": evaluation["return"]["mean"],
                        "return_std": evaluation["return"]["std"],
                        "return_stderr": evaluation["return"]["stderr"],
                        "normalized_score_mean": evaluation[
                            "normalized_score"
                        ]["mean"],
                        "normalized_score_std": evaluation[
                            "normalized_score"
                        ]["std"],
                        "normalized_score_stderr": evaluation[
                            "normalized_score"
                        ]["stderr"],
                        "episode_length_mean": evaluation[
                            "episode_length"
                        ]["mean"],
                    }
                )


def main() -> None:
    args = parse_args()
    if args.episodes <= 0 or args.batch_size <= 0:
        raise ValueError("episodes and batch-size must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device is unavailable: {args.device}")

    checkpoints = [path.expanduser().resolve() for path in args.checkpoints]
    for checkpoint in checkpoints:
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
    cfg = load_config(args.device)
    base_policy = load_base_policy(cfg)
    seeds = list(range(args.seed_start, args.seed_start + args.episodes))
    results = {
        "environment": cfg.env_name,
        "diffusion_sampler": "DDIM",
        "ddim_steps": int(cfg.model.ddim_steps),
        "episodes_per_mode": args.episodes,
        "seed_start": args.seed_start,
        "policy_seed": args.policy_seed,
        "batch_size": args.batch_size,
        "checkpoints": [],
    }

    load_env, _ = make_env(cfg)
    try:
        for checkpoint in checkpoints:
            model = _LegacyLoadableDSRL.load(
                checkpoint,
                env=load_env,
                device=args.device,
                custom_objects={"diffusion_policy": base_policy},
                buffer_size=1,
            )
            checkpoint_result = {
                "path": str(checkpoint),
                "sha256": sha256(checkpoint),
                "checkpoint_step": checkpoint_step(checkpoint),
                "saved_num_timesteps": int(model.num_timesteps),
                "evaluations": {},
            }
            for mode in args.modes:
                checkpoint_result["evaluations"][mode] = evaluate_mode(
                    model=model,
                    cfg=cfg,
                    seeds=seeds,
                    batch_size=args.batch_size,
                    deterministic=mode == "deterministic",
                    policy_seed=args.policy_seed,
                )
                print(
                    f"{checkpoint.name} {mode}: "
                    f"return={checkpoint_result['evaluations'][mode]['return']['mean']:.3f}, "
                    "normalized_score="
                    f"{checkpoint_result['evaluations'][mode]['normalized_score']['mean']:.3f}"
                )
            results["checkpoints"].append(checkpoint_result)
            write_results(args.output.resolve(), results)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        load_env.close()

    print(f"Wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
