"""Screen residual exploration scale and multi-step advantage ranking.

This diagnostic never updates a policy.  It keeps the frozen DSRL RNG stream
independent from residual RNG, uses validation-only seeds for scale selection,
and evaluates whether longer n-step bootstrap targets improve local residual
action ordering, especially across action-chunk boundaries.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / "dppo"), str(ROOT / "stable-baselines3")]

from diagnose_residual_random_and_advantage import (  # noqa: E402
    _aggregate_ranking_rows,
    _predict_value,
    _ranking_bootstrap,
    _seed_frozen_planner,
    evaluate_random_families,
)
from evaluate_per_step_residual_ppo import (  # noqa: E402
    _model_class,
    _planner_from_manifest,
)
from p6_preflight import sha256_file  # noqa: E402
from p6_runtime import atomic_write_json, isolated_rng, seed_all  # noqa: E402
from per_step_residual_ppo import PPOResidualUnitEnv, SeparatedClipPPO  # noqa: E402
from train_per_step_residual import (  # noqa: E402
    ACTION_CHUNK,
    compose_config,
    make_chunk_contract_environment,
    module_state_hash,
)
from train_per_step_residual_ppo import make_ppo_environment  # noqa: E402
from utils import load_base_policy  # noqa: E402


DEFAULT_STDS = (0.02, 0.05, 0.1, 0.2, math.exp(-1.0))
DEFAULT_HORIZONS = (1, 4, 16, 32, 64)


def _float_list(value: str) -> tuple[float, ...]:
    result = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not result or any(not np.isfinite(item) or item <= 0 for item in result):
        raise argparse.ArgumentTypeError("Expected comma-separated positive floats")
    if len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("Values must be unique")
    return result


def _int_list(value: str) -> tuple[int, ...]:
    result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not result or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError("Expected comma-separated positive integers")
    if tuple(sorted(set(result))) != result:
        raise argparse.ArgumentTypeError("Horizons must be unique and increasing")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--residual-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--stds", type=_float_list, default=DEFAULT_STDS)
    parser.add_argument("--horizons", type=_int_list, default=DEFAULT_HORIZONS)
    parser.add_argument("--validation-episodes", type=int, default=25)
    parser.add_argument("--random-families", type=int, default=5)
    parser.add_argument("--environment-seed-start", type=int, default=12_000)
    parser.add_argument("--base-policy-seed-start", type=int, default=22_000)
    parser.add_argument("--residual-seed-start", type=int, default=32_000)
    parser.add_argument("--counterfactual-environment-seed-start", type=int, default=14_000)
    parser.add_argument("--counterfactual-base-seed-start", type=int, default=24_000)
    parser.add_argument("--counterfactual-states", type=int, default=128)
    parser.add_argument("--random-candidates", type=int, default=8)
    parser.add_argument("--ranking-residual-std", type=float, default=0.05)
    parser.add_argument("--sample-spacing", type=int, default=31)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def _counterfactual_multi_horizon(
    *,
    environment: PPOResidualUnitEnv,
    observation: np.ndarray,
    snapshot: Any,
    residual: np.ndarray,
    model: SeparatedClipPPO,
    gamma: float,
    reward_scale: float,
    horizons: Sequence[int],
) -> dict[str, Any]:
    """Evaluate one residual intervention with common zero continuation."""

    environment.restore_state(snapshot)
    current_value = _predict_value(model, observation)
    max_horizon = int(max(horizons))
    horizon_set = set(map(int, horizons))
    predictions: dict[str, dict[str, Any]] = {}
    discounted_raw_return = 0.0
    discount = 1.0
    terminated = truncated = False
    first_info: dict[str, Any] | None = None
    next_observation = np.asarray(observation, dtype=np.float32)
    steps = 0
    while steps < max_horizon and not (terminated or truncated):
        action = (
            residual
            if steps == 0
            else np.zeros(environment.action_space.shape, dtype=np.float32)
        )
        (
            next_observation,
            reward,
            terminated,
            truncated,
            info,
        ) = environment.step(action)
        if first_info is None:
            first_info = dict(info)
        discounted_raw_return += discount * float(reward)
        discount *= float(gamma)
        steps += 1
        if steps in horizon_set:
            bootstrap = (
                0.0
                if terminated or truncated
                else _predict_value(model, next_observation)
            )
            predictions[str(steps)] = {
                "predicted_advantage": float(
                    reward_scale * discounted_raw_return
                    + discount * bootstrap
                    - current_value
                ),
                "discounted_raw_prefix_return": float(discounted_raw_return),
                "bootstrap_value": float(bootstrap),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "steps": int(steps),
            }
    for horizon in horizons:
        key = str(int(horizon))
        if key not in predictions:
            # The episode ended before this requested horizon.  The terminal
            # return and zero bootstrap are the correct value for every later
            # horizon.
            predictions[key] = {
                "predicted_advantage": float(
                    reward_scale * discounted_raw_return - current_value
                ),
                "discounted_raw_prefix_return": float(discounted_raw_return),
                "bootstrap_value": 0.0,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "steps": int(steps),
            }
    if first_info is None:
        raise RuntimeError("Counterfactual intervention executed no step")
    return {
        "predictions": predictions,
        "counterfactual_discounted_return": float(discounted_raw_return),
        "actual_steps": int(steps),
        "action_residual_delta_l2": float(
            first_info["action_residual_delta_l2"]
        ),
    }


def _rows_for_horizon(
    rows: Sequence[dict[str, Any]],
    horizon: int,
) -> list[dict[str, Any]]:
    key = str(int(horizon))
    return [
        {
            **row,
            "candidates": [
                {
                    **candidate,
                    "td_advantage": candidate["predictions"][key][
                        "predicted_advantage"
                    ],
                }
                for candidate in row["candidates"]
            ],
        }
        for row in rows
    ]


def aggregate_multi_horizon_ranking(
    rows: Sequence[dict[str, Any]],
    *,
    horizons: Sequence[int],
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for horizon in horizons:
        horizon_rows = _rows_for_horizon(rows, int(horizon))
        overall = _aggregate_ranking_rows(horizon_rows)
        confidence = _ranking_bootstrap(
            horizon_rows,
            samples=int(bootstrap_samples),
            seed=int(bootstrap_seed) + int(horizon),
        )
        by_phase: dict[str, Any] = {}
        for phase in range(ACTION_CHUNK):
            phase_rows = [
                row for row in horizon_rows if int(row["phase"]) == phase
            ]
            by_phase[str(phase)] = {
                **_aggregate_ranking_rows(phase_rows),
                "confidence_interval_95": _ranking_bootstrap(
                    phase_rows,
                    samples=int(bootstrap_samples),
                    seed=int(bootstrap_seed) + 1000 * (phase + 1) + int(horizon),
                ),
            }
        result[str(int(horizon))] = {
            "overall": overall,
            "confidence_interval_95": confidence,
            "by_phase": by_phase,
            "screening_gate": {
                "spearman_ci95_lower_gt_0": bool(
                    confidence["spearman_mean"][0] > 0.0
                ),
                "pairwise_ci95_lower_gt_0p55": bool(
                    confidence["pairwise_preference_accuracy"][0] > 0.55
                ),
                "zero_random_ci95_lower_gt_0p55": bool(
                    confidence["zero_random_preference_accuracy"][0] > 0.55
                ),
            },
        }
    return result


def collect_multi_horizon_ranking(
    *,
    model: SeparatedClipPPO,
    make_environment: Callable[[], PPOResidualUnitEnv],
    environment_seed_start: int,
    base_policy_seed_start: int,
    residual_seed: int,
    state_count: int,
    random_candidates: int,
    residual_std: float,
    gamma: float,
    reward_scale: float,
    horizons: Sequence[int],
    sample_spacing: int,
    bootstrap_samples: int,
) -> dict[str, Any]:
    if math.gcd(int(sample_spacing), ACTION_CHUNK) != 1:
        raise ValueError("Sample spacing must be coprime to action chunk")
    residual_rng = np.random.default_rng(int(residual_seed))
    rows: list[dict[str, Any]] = []
    episode_index = 0
    global_step = 0
    model.policy.set_training_mode(False)
    while len(rows) < int(state_count):
        environment = make_environment()
        try:
            environment_seed = int(environment_seed_start) + episode_index
            base_policy_seed = int(base_policy_seed_start) + episode_index
            seed_all(base_policy_seed)
            _seed_frozen_planner(environment, base_policy_seed)
            observation, _ = environment.reset(seed=environment_seed)
            terminated = truncated = False
            while not (terminated or truncated) and len(rows) < int(state_count):
                if global_step % int(sample_spacing) == 0:
                    snapshot = environment.capture_state()
                    phase = int(environment.phase)
                    random_actions = residual_rng.normal(
                        0.0,
                        float(residual_std),
                        size=(int(random_candidates), *environment.action_space.shape),
                    ).astype(np.float32)
                    random_actions = np.clip(
                        random_actions,
                        environment.action_space.low,
                        environment.action_space.high,
                    )
                    actions = [
                        np.zeros(environment.action_space.shape, dtype=np.float32),
                        *list(random_actions),
                    ]
                    candidates: list[dict[str, Any]] = []
                    for candidate_index, action in enumerate(actions):
                        candidate = _counterfactual_multi_horizon(
                            environment=environment,
                            observation=observation,
                            snapshot=snapshot,
                            residual=action,
                            model=model,
                            gamma=float(gamma),
                            reward_scale=float(reward_scale),
                            horizons=horizons,
                        )
                        candidate.update(
                            {
                                "candidate_index": int(candidate_index),
                                "kind": "zero" if candidate_index == 0 else "random",
                                "residual_unit": action.tolist(),
                            }
                        )
                        candidates.append(candidate)
                    environment.restore_state(snapshot)
                    rows.append(
                        {
                            "state_index": len(rows),
                            "episode_index": int(episode_index),
                            "environment_seed": environment_seed,
                            "base_policy_seed": base_policy_seed,
                            "primitive_step": int(global_step),
                            "phase": phase,
                            "candidates": candidates,
                        }
                    )
                zero = np.zeros(environment.action_space.shape, dtype=np.float32)
                observation, _, terminated, truncated, _ = environment.step(zero)
                global_step += 1
        finally:
            environment.close()
        episode_index += 1
    return {
        "protocol": (
            "matched simulator snapshot and frozen-planner RNG; zero/random first "
            "residual; common zero-residual continuation; n-step scaled-return "
            "plus value bootstrap ranked against max-horizon raw return"
        ),
        "gamma": float(gamma),
        "reward_scale": float(reward_scale),
        "residual_std": float(residual_std),
        "horizons": list(map(int, horizons)),
        "max_horizon_target": int(max(horizons)),
        "state_count": int(len(rows)),
        "random_candidates_per_state": int(random_candidates),
        "sample_spacing": int(sample_spacing),
        "metrics": aggregate_multi_horizon_ranking(
            rows,
            horizons=horizons,
            bootstrap_samples=int(bootstrap_samples),
            bootstrap_seed=int(residual_seed) + 1,
        ),
        "states": rows,
    }


def main() -> None:
    arguments = parse_args()
    if arguments.validation_episodes <= 0 or arguments.random_families < 2:
        raise ValueError("Scale screening requires episodes and >=2 RNG families")
    if arguments.counterfactual_states < ACTION_CHUNK:
        raise ValueError("Counterfactual states must cover every chunk phase")
    if arguments.random_candidates < 2:
        raise ValueError("At least two random candidates are required")
    if arguments.bootstrap_samples < 100:
        raise ValueError("At least 100 bootstrap samples are required")
    output_directory = arguments.output_dir.resolve()
    if output_directory.exists():
        raise FileExistsError(f"Refusing to overwrite {output_directory}")
    output_directory.mkdir(parents=True)

    manifest_path = arguments.run_manifest.resolve()
    manifest = json.loads(manifest_path.read_text())
    if manifest["training_variant"] not in ("stable_v1", "resip_hopper_v1"):
        raise ValueError("Diagnostic requires stable_v1 or resip_hopper_v1")
    model_path = arguments.residual_checkpoint.resolve()
    config_arguments = argparse.Namespace(
        checkpoint=(
            Path(manifest["checkpoint_path"])
            if manifest.get("checkpoint_path") is not None
            else None
        ),
        equivalent_chunk_budget=int(manifest["equivalent_chunk_budget"]),
        seed=int(arguments.seed),
        n_envs=1,
        device=arguments.device,
    )
    cfg = compose_config(config_arguments)
    cfg.logdir = str(output_directory)
    if sha256_file(Path(cfg.base_policy_path)) != manifest["ddim_sha256"]:
        raise ValueError("Frozen DDIM hash differs from source manifest")
    if sha256_file(Path(cfg.normalization_path)) != manifest["normalization_sha256"]:
        raise ValueError("Normalization hash differs from source manifest")

    diffusion = load_base_policy(cfg)
    contract = make_chunk_contract_environment(cfg)
    planner = None
    try:
        planner, modules = _planner_from_manifest(
            manifest,
            cfg=cfg,
            diffusion=diffusion,
            contract=contract,
        )
        planner.assert_frozen()
        planner_hash_before = module_state_hash(modules)
        model = _model_class(manifest["training_variant"]).load(
            model_path,
            device=arguments.device,
        )
        make_environment = lambda: make_ppo_environment(
            cfg,
            planner,
            residual_scale=float(manifest["residual_scale"]),
            deterministic_base=bool(manifest["deterministic_base"]),
        )
        environment_seeds = list(
            range(
                int(arguments.environment_seed_start),
                int(arguments.environment_seed_start) + int(arguments.validation_episodes),
            )
        )
        base_policy_seeds = list(
            range(
                int(arguments.base_policy_seed_start),
                int(arguments.base_policy_seed_start) + int(arguments.validation_episodes),
            )
        )
        residual_family_seeds = list(
            range(
                int(arguments.residual_seed_start),
                int(arguments.residual_seed_start) + int(arguments.random_families),
            )
        )
        scale_results: dict[str, Any] = {}
        with isolated_rng():
            shared_zero_rows = None
            for index, residual_std in enumerate(arguments.stds):
                print(f"std sweep {index + 1}/{len(arguments.stds)}: {residual_std}", flush=True)
                result = evaluate_random_families(
                    make_environment=make_environment,
                    environment_seeds=environment_seeds,
                    base_policy_seeds=base_policy_seeds,
                    residual_family_seeds=residual_family_seeds,
                    residual_std=float(residual_std),
                    bootstrap_samples=int(arguments.bootstrap_samples),
                    bootstrap_seed=int(arguments.seed) + 100 + index,
                    zero_rows=shared_zero_rows,
                )
                if shared_zero_rows is None:
                    shared_zero_rows = result["zero"]["episodes"]
                scale_results[str(float(residual_std))] = result
            print("multi-horizon counterfactual ranking", flush=True)
            ranking_result = collect_multi_horizon_ranking(
                model=model,
                make_environment=make_environment,
                environment_seed_start=int(
                    arguments.counterfactual_environment_seed_start
                ),
                base_policy_seed_start=int(arguments.counterfactual_base_seed_start),
                residual_seed=int(arguments.residual_seed_start) + 1000,
                state_count=int(arguments.counterfactual_states),
                random_candidates=int(arguments.random_candidates),
                residual_std=float(arguments.ranking_residual_std),
                gamma=float(model.gamma),
                reward_scale=float(manifest["training_reward_scale"]),
                horizons=arguments.horizons,
                sample_spacing=int(arguments.sample_spacing),
                bootstrap_samples=int(arguments.bootstrap_samples),
            )
        planner_hash_after = module_state_hash(modules)
        if planner_hash_after != planner_hash_before:
            raise RuntimeError("Frozen planner changed during diagnostics")
        scale_payload = {
            "split": "validation",
            "selection_prohibited_on_test_seeds": True,
            "stds": list(map(float, arguments.stds)),
            "results": scale_results,
        }
        atomic_write_json(output_directory / "std_sweep_validation.json", scale_payload)
        atomic_write_json(output_directory / "multi_horizon_ranking.json", ranking_result)
        diagnostic_manifest = {
            "source_run_manifest": str(manifest_path),
            "source_model_checkpoint": str(model_path),
            "training_variant": manifest["training_variant"],
            "base_source": manifest["base_source"],
            "checkpoint_sha256": manifest.get("checkpoint_sha256"),
            "ddim_sha256": manifest["ddim_sha256"],
            "normalization_sha256": manifest["normalization_sha256"],
            "planner_hash_before": planner_hash_before,
            "planner_hash_after": planner_hash_after,
            "validation_environment_seeds": environment_seeds,
            "validation_base_policy_seeds": base_policy_seeds,
            "residual_family_seeds": residual_family_seeds,
            "counterfactual_environment_seed_start": int(
                arguments.counterfactual_environment_seed_start
            ),
            "counterfactual_base_seed_start": int(
                arguments.counterfactual_base_seed_start
            ),
            "bootstrap_samples": int(arguments.bootstrap_samples),
        }
        atomic_write_json(output_directory / "diagnostic_manifest.json", diagnostic_manifest)
        summary = {
            "std_sweep": {
                key: value["aggregate"] for key, value in scale_results.items()
            },
            "multi_horizon": {
                key: value["overall"]
                for key, value in ranking_result["metrics"].items()
            },
        }
        atomic_write_json(output_directory / "summary.json", summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
    finally:
        contract.close()


if __name__ == "__main__":
    main()
