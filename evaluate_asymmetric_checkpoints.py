"""Exact fixed-seed evaluation for asymmetric DSRL/PPO boundary bundles."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / "dppo"), str(ROOT / "stable-baselines3")]

from p6_runtime import atomic_write_json  # noqa: E402
from p6_train import P6ControlDSRL, _make_hopper_environment  # noqa: E402
from per_step_residual_env import FrozenDSRLChunkPlanner  # noqa: E402
from per_step_residual_ppo import CountingPPO  # noqa: E402
from stable_baselines3.common.env_util import make_vec_env  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv  # noqa: E402
from train_per_step_residual import (  # noqa: E402
    ACTION_CHUNK,
    ACTION_DIMENSION,
    compose_config,
)
from train_per_step_residual_ppo import (  # noqa: E402
    evaluate_exact,
    make_ppo_environment,
    make_training_environment,
    write_evaluation,
)
from utils import load_base_policy  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-chunks",
        type=int,
        nargs="+",
        default=[2_000, 4_000, 6_000, 8_000, 10_000],
    )
    parser.add_argument(
        "--residual-multipliers",
        type=float,
        nargs="+",
        default=[0.0, 1.0],
    )
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--deterministic-residual", action="store_true")
    return parser.parse_args()


def paired_summary(
    *,
    base: dict[str, Any],
    treatment: dict[str, Any],
    bootstrap_seed: int = 20260731,
    bootstrap_samples: int = 20_000,
) -> dict[str, Any]:
    base_rows = base["episodes"]
    treatment_rows = treatment["episodes"]
    if len(base_rows) != len(treatment_rows) or not base_rows:
        raise ValueError("Paired evaluations must contain equal non-empty episodes")
    base_seeds = [row["environment_seed"] for row in base_rows]
    treatment_seeds = [row["environment_seed"] for row in treatment_rows]
    if base_seeds != treatment_seeds:
        raise ValueError("Paired evaluations use different environment seeds")
    base_returns = np.asarray(
        [row["raw_return"] for row in base_rows],
        dtype=np.float64,
    )
    treatment_returns = np.asarray(
        [row["raw_return"] for row in treatment_rows],
        dtype=np.float64,
    )
    differences = treatment_returns - base_returns
    base_falls = np.asarray(
        [row["early_fall"] for row in base_rows],
        dtype=bool,
    )
    treatment_falls = np.asarray(
        [row["early_fall"] for row in treatment_rows],
        dtype=bool,
    )
    rng = np.random.default_rng(int(bootstrap_seed))
    indices = rng.integers(
        0,
        len(differences),
        size=(int(bootstrap_samples), len(differences)),
    )
    bootstrap_means = differences[indices].mean(axis=1)
    both_success = ~base_falls & ~treatment_falls
    return {
        "episode_count": len(differences),
        "paired_return_difference_mean": float(differences.mean()),
        "paired_return_difference_median": float(np.median(differences)),
        "paired_return_difference_q10": float(np.quantile(differences, 0.1)),
        "paired_return_difference_ci95": [
            float(np.quantile(bootstrap_means, 0.025)),
            float(np.quantile(bootstrap_means, 0.975)),
        ],
        "positive_difference_fraction": float(np.mean(differences > 0)),
        "base_early_fall_rate": float(base_falls.mean()),
        "treatment_early_fall_rate": float(treatment_falls.mean()),
        "both_success_count": int(both_success.sum()),
        "base_success_treatment_fall_count": int(
            (~base_falls & treatment_falls).sum()
        ),
        "base_fall_treatment_success_count": int(
            (base_falls & ~treatment_falls).sum()
        ),
        "both_fall_count": int((base_falls & treatment_falls).sum()),
        "both_success_difference_mean": (
            float(differences[both_success].mean())
            if np.any(both_success)
            else None
        ),
    }


def _bundle_path(run_directory: Path, chunk: int) -> Path:
    candidate = (
        run_directory
        / "checkpoints"
        / f"boundary_{int(chunk):012d}"
    )
    if not (candidate / "bundle.json").is_file():
        raise FileNotFoundError(f"Missing boundary bundle {candidate}")
    return candidate


def _multiplier_label(value: float) -> str:
    return str(float(value)).replace("-", "m").replace(".", "p")


def main() -> None:
    arguments = parse_args()
    if arguments.episodes <= 0:
        raise ValueError("episodes must be positive")
    multipliers = [float(value) for value in arguments.residual_multipliers]
    if 0.0 not in multipliers:
        raise ValueError("A zero-residual multiplier is required for pairing")
    if any(not np.isfinite(value) or value < 0 for value in multipliers):
        raise ValueError("Residual multipliers must be finite and non-negative")

    run_directory = arguments.run_dir.resolve()
    manifest = json.loads((run_directory / "run_manifest.json").read_text())
    checkpoint_path = Path(manifest["checkpoint_path"])
    config_arguments = argparse.Namespace(
        checkpoint=checkpoint_path,
        equivalent_chunk_budget=int(
            manifest["online_equivalent_chunk_budget"]
        ),
        seed=int(manifest["seed"]),
        n_envs=int(manifest.get("n_envs", 10)),
        device=arguments.device,
    )
    cfg = compose_config(config_arguments)
    cfg.logdir = str(run_directory)
    diffusion_policy = load_base_policy(cfg)
    contract = make_vec_env(
        lambda: _make_hopper_environment(
            cfg,
            Path(cfg.normalization_path),
        ),
        n_envs=1,
        vec_env_cls=DummyVecEnv,
    )
    output_directory = run_directory / "evaluations" / "checkpoint_sweep"
    output_directory.mkdir(parents=True, exist_ok=True)
    seeds = list(range(10_000, 10_000 + int(arguments.episodes)))
    aggregate_path = output_directory / "aggregate.json"
    if aggregate_path.is_file():
        aggregate = json.loads(aggregate_path.read_text())
        if (
            aggregate["deterministic_residual"]
            != bool(arguments.deterministic_residual)
            or aggregate["environment_seeds"] != seeds
        ):
            raise ValueError(
                "Existing checkpoint sweep uses a different evaluation protocol"
            )
        aggregate["checkpoint_chunks"] = sorted(
            {
                *map(int, aggregate["checkpoint_chunks"]),
                *map(int, arguments.checkpoint_chunks),
            }
        )
        aggregate["residual_multipliers"] = sorted(
            {
                *map(float, aggregate["residual_multipliers"]),
                *multipliers,
            }
        )
    else:
        aggregate = {
            "protocol": (
                "exact fixed-seed paired evaluation; current boundary base actor; "
                "same residual PPO checkpoint; residual scale multiplier sweep"
            ),
            "deterministic_residual": bool(
                arguments.deterministic_residual
            ),
            "environment_seeds": seeds,
            "checkpoint_chunks": list(
                map(int, arguments.checkpoint_chunks)
            ),
            "residual_multipliers": multipliers,
            "results": {},
        }
    try:
        for chunk in arguments.checkpoint_chunks:
            bundle = _bundle_path(run_directory, int(chunk))
            bundle_data = json.loads((bundle / "bundle.json").read_text())
            base_model = P6ControlDSRL.load(
                bundle / "base_model.zip",
                env=contract,
                device=arguments.device,
                custom_objects={"diffusion_policy": diffusion_policy},
            )
            planner = FrozenDSRLChunkPlanner(
                base_model,
                action_chunk=ACTION_CHUNK,
                action_dimension=ACTION_DIMENSION,
            )
            planner.base_version = int(bundle_data["snapshot_version"])
            ppo_environment = make_training_environment(
                cfg,
                planner,
                n_envs=1,
                seed=int(manifest["joint_train_env_seed"]),
                residual_scale=float(manifest["residual_scale"]),
                deterministic_base=bool(
                    manifest.get("deterministic_base", False)
                ),
            )
            try:
                residual_model = CountingPPO.load(
                    bundle / "residual_ppo.zip",
                    env=ppo_environment,
                    device=arguments.device,
                )
                evaluations: dict[float, dict[str, Any]] = {}
                for multiplier in multipliers:
                    output_prefix = (
                        output_directory
                        / (
                            f"chunk_{int(chunk):012d}"
                            f"_eta_{_multiplier_label(multiplier)}"
                        )
                    )
                    json_path = output_prefix.with_suffix(".json")
                    if json_path.is_file():
                        result = json.loads(json_path.read_text())
                        if (
                            result["environment_seeds"] != seeds
                            or result["deterministic"]
                            != bool(arguments.deterministic_residual)
                        ):
                            raise ValueError(
                                f"Cached evaluation protocol mismatch: {json_path}"
                            )
                    else:
                        result = evaluate_exact(
                            model=residual_model,
                            make_environment=lambda multiplier=multiplier: (
                                make_ppo_environment(
                                    cfg,
                                    planner,
                                    residual_scale=(
                                        float(manifest["residual_scale"])
                                        * multiplier
                                    ),
                                    deterministic_base=bool(
                                        manifest.get(
                                            "deterministic_base",
                                            False,
                                        )
                                    ),
                                )
                            ),
                            seeds=seeds,
                            policy_seed_start=20_000,
                            equivalent_chunks=int(chunk),
                            deterministic=bool(
                                arguments.deterministic_residual
                            ),
                        )
                        write_evaluation(output_prefix, result)
                    evaluations[multiplier] = result
                base = evaluations[0.0]
                chunk_result = {
                    "base_summary": base["summary"],
                    "multipliers": {},
                }
                for multiplier, treatment in evaluations.items():
                    chunk_result["multipliers"][str(multiplier)] = {
                        "summary": treatment["summary"],
                        "paired_vs_zero": paired_summary(
                            base=base,
                            treatment=treatment,
                            bootstrap_seed=20260731 + int(chunk),
                        ),
                    }
                aggregate["results"][str(int(chunk))] = chunk_result
                atomic_write_json(aggregate_path, aggregate)
            finally:
                ppo_environment.close()
    finally:
        contract.close()


if __name__ == "__main__":
    main()
