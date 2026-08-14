"""Diagnose stochastic residual robustness and PPO local-action ranking.

This script deliberately makes no training updates.  It answers two gates:

1. Does the zero-mean N(0, sigma_0) residual baseline remain beneficial when
   only its RNG family changes and the frozen DSRL RNG stream is held fixed?
2. Does the trained PPO value function rank zero/random first-action
   interventions in the same order as matched simulator counterfactual return?

Only if both gates pass is a variance-only PPO follow-up justified.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / "dppo"), str(ROOT / "stable-baselines3")]

from p6_preflight import sha256_file  # noqa: E402
from p6_runtime import atomic_write_json, isolated_rng, seed_all  # noqa: E402
from p6_train import _load_legacy_network  # noqa: E402
from per_step_residual_env import FrozenDSRLChunkPlanner  # noqa: E402
from per_step_residual_ppo import PPOResidualUnitEnv, SeparatedClipPPO  # noqa: E402
from train_per_step_residual import (  # noqa: E402
    ACTION_CHUNK,
    ACTION_DIMENSION,
    compose_config,
    make_chunk_contract_environment,
    module_state_hash,
    planner_modules,
)
from train_per_step_residual_ppo import make_ppo_environment  # noqa: E402
from utils import load_base_policy  # noqa: E402


DEFAULT_RANDOM_FAMILIES = 10
DEFAULT_EVALUATION_EPISODES = 100
DEFAULT_COUNTERFACTUAL_STATES = 128
DEFAULT_RANDOM_CANDIDATES = 8
DEFAULT_COUNTERFACTUAL_HORIZON = 64
DEFAULT_SAMPLE_SPACING = 31


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--residual-checkpoint", type=Path, required=True)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--environment-seed-start", type=int, default=10_000)
    parser.add_argument("--base-policy-seed-start", type=int, default=20_000)
    parser.add_argument("--residual-seed-start", type=int, default=30_000)
    parser.add_argument(
        "--random-families",
        type=int,
        default=DEFAULT_RANDOM_FAMILIES,
    )
    parser.add_argument(
        "--evaluation-episodes",
        type=int,
        default=DEFAULT_EVALUATION_EPISODES,
    )
    parser.add_argument(
        "--counterfactual-states",
        type=int,
        default=DEFAULT_COUNTERFACTUAL_STATES,
    )
    parser.add_argument(
        "--random-candidates",
        type=int,
        default=DEFAULT_RANDOM_CANDIDATES,
    )
    parser.add_argument(
        "--counterfactual-horizon",
        type=int,
        default=DEFAULT_COUNTERFACTUAL_HORIZON,
    )
    parser.add_argument(
        "--sample-spacing",
        type=int,
        default=DEFAULT_SAMPLE_SPACING,
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    return parser.parse_args()


def _episode_seed(family_seed: int, episode_index: int) -> int:
    sequence = np.random.SeedSequence([int(family_seed), int(episode_index)])
    return int(sequence.generate_state(1, dtype=np.uint64)[0])


def _seed_frozen_planner(environment: Any, seed: int) -> None:
    """Seed the planner's private RNG, not only Torch's global stream."""

    current = environment
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        planner = getattr(current, "planner", None)
        if planner is not None:
            seed_method = getattr(planner, "seed", None)
            if not callable(seed_method):
                raise TypeError("Frozen planner does not expose seed()")
            seed_method(int(seed))
            return
        current = getattr(current, "env", None)
    raise TypeError("Could not locate frozen planner in diagnostic environment")


def _bootstrap_interval(
    values: np.ndarray,
    *,
    rng: np.random.Generator,
    samples: int,
) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("Bootstrap input must be a non-empty vector")
    indices = rng.integers(0, array.size, size=(samples, array.size))
    estimates = array[indices].mean(axis=1)
    return (
        float(np.quantile(estimates, 0.025)),
        float(np.quantile(estimates, 0.975)),
    )


def _nested_bootstrap_interval(
    matrix: np.ndarray,
    *,
    rng: np.random.Generator,
    samples: int,
) -> tuple[float, float]:
    """Bootstrap residual RNG families and environment seeds independently."""

    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or min(values.shape) == 0:
        raise ValueError("Nested bootstrap input must be a non-empty matrix")
    estimates = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        family_indices = rng.integers(0, values.shape[0], values.shape[0])
        episode_indices = rng.integers(0, values.shape[1], values.shape[1])
        estimates[index] = values[
            family_indices[:, None],
            episode_indices[None, :],
        ].mean()
    return (
        float(np.quantile(estimates, 0.025)),
        float(np.quantile(estimates, 0.975)),
    )


def _episode_rollout(
    *,
    make_environment: Callable[[], PPOResidualUnitEnv],
    environment_seed: int,
    base_policy_seed: int,
    residual_seed: int | None,
    residual_std: float,
) -> dict[str, Any]:
    environment = make_environment()
    try:
        # The frozen DSRL actor uses global Torch RNG.  Residual actions use an
        # independent NumPy Generator and therefore cannot perturb that stream.
        seed_all(int(base_policy_seed))
        _seed_frozen_planner(environment, int(base_policy_seed))
        residual_rng = (
            None
            if residual_seed is None
            else np.random.default_rng(int(residual_seed))
        )
        _, _ = environment.reset(seed=int(environment_seed))
        terminated = truncated = False
        total_reward = 0.0
        primitive_length = 0
        residual_norms: list[float] = []
        while not (terminated or truncated):
            if residual_rng is None:
                residual = np.zeros(
                    environment.action_space.shape,
                    dtype=np.float32,
                )
            else:
                residual = residual_rng.normal(
                    loc=0.0,
                    scale=float(residual_std),
                    size=environment.action_space.shape,
                ).astype(np.float32)
                residual = np.clip(
                    residual,
                    environment.action_space.low,
                    environment.action_space.high,
                )
            _, reward, terminated, truncated, info = environment.step(residual)
            total_reward += float(reward)
            primitive_length += 1
            residual_norms.append(float(info["action_residual_delta_l2"]))
        return {
            "environment_seed": int(environment_seed),
            "base_policy_seed": int(base_policy_seed),
            "residual_seed": (
                None if residual_seed is None else int(residual_seed)
            ),
            "raw_return": float(total_reward),
            "primitive_length": int(primitive_length),
            "early_fall": bool(
                primitive_length < int(environment.env.max_episode_steps)
            ),
            "residual_delta_l2": float(np.mean(residual_norms)),
        }
    finally:
        environment.close()


def evaluate_random_families(
    *,
    make_environment: Callable[[], PPOResidualUnitEnv],
    environment_seeds: Sequence[int],
    base_policy_seeds: Sequence[int],
    residual_family_seeds: Sequence[int],
    residual_std: float,
    bootstrap_samples: int,
    bootstrap_seed: int,
    zero_rows: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if len(environment_seeds) != len(base_policy_seeds):
        raise ValueError("Environment/base seed lists must have equal length")
    if len(residual_family_seeds) < 2:
        raise ValueError("At least two residual RNG families are required")
    if zero_rows is None:
        zero_rows = [
            _episode_rollout(
                make_environment=make_environment,
                environment_seed=environment_seed,
                base_policy_seed=base_policy_seed,
                residual_seed=None,
                residual_std=residual_std,
            )
            for environment_seed, base_policy_seed in zip(
                environment_seeds,
                base_policy_seeds,
            )
        ]
    else:
        zero_rows = list(zero_rows)
        expected = list(zip(map(int, environment_seeds), map(int, base_policy_seeds)))
        observed = [
            (int(row["environment_seed"]), int(row["base_policy_seed"]))
            for row in zero_rows
        ]
        if observed != expected:
            raise ValueError("Shared zero rows do not match requested seed pairs")
    zero_returns = np.asarray(
        [row["raw_return"] for row in zero_rows],
        dtype=np.float64,
    )
    zero_falls = np.asarray(
        [row["early_fall"] for row in zero_rows],
        dtype=np.float64,
    )
    family_rows: list[dict[str, Any]] = []
    delta_matrix: list[np.ndarray] = []
    fall_delta_matrix: list[np.ndarray] = []
    for family_seed in residual_family_seeds:
        rows = [
            _episode_rollout(
                make_environment=make_environment,
                environment_seed=environment_seed,
                base_policy_seed=base_policy_seed,
                residual_seed=_episode_seed(family_seed, episode_index),
                residual_std=residual_std,
            )
            for episode_index, (
                environment_seed,
                base_policy_seed,
            ) in enumerate(zip(environment_seeds, base_policy_seeds))
        ]
        returns = np.asarray(
            [row["raw_return"] for row in rows],
            dtype=np.float64,
        )
        falls = np.asarray(
            [row["early_fall"] for row in rows],
            dtype=np.float64,
        )
        deltas = returns - zero_returns
        fall_deltas = falls - zero_falls
        delta_matrix.append(deltas)
        fall_delta_matrix.append(fall_deltas)
        family_rows.append(
            {
                "residual_family_seed": int(family_seed),
                "raw_return_mean": float(returns.mean()),
                "paired_delta_mean": float(deltas.mean()),
                "paired_delta_median": float(np.median(deltas)),
                "positive_fraction": float(np.mean(deltas > 0)),
                "early_fall_rate": float(falls.mean()),
                "early_fall_rate_delta": float(fall_deltas.mean()),
                "residual_delta_l2": float(
                    np.mean([row["residual_delta_l2"] for row in rows])
                ),
                "episodes": rows,
            }
        )
    delta_array = np.stack(delta_matrix)
    fall_delta_array = np.stack(fall_delta_matrix)
    bootstrap_rng = np.random.default_rng(int(bootstrap_seed))
    nested_ci = _nested_bootstrap_interval(
        delta_array,
        rng=bootstrap_rng,
        samples=int(bootstrap_samples),
    )
    family_means = delta_array.mean(axis=1)
    family_positive_fraction = float(np.mean(family_means > 0))
    aggregate_fall_delta = float(fall_delta_array.mean())
    robust = bool(
        family_positive_fraction >= 0.8
        and nested_ci[0] > 0.0
        and aggregate_fall_delta <= 0.0
    )
    return {
        "protocol": (
            "paired exact episodes; fixed environment and frozen-base RNG per "
            "episode; independent externally sampled residual RNG families"
        ),
        "residual_std": float(residual_std),
        "environment_seeds": list(map(int, environment_seeds)),
        "base_policy_seeds": list(map(int, base_policy_seeds)),
        "residual_family_seeds": list(map(int, residual_family_seeds)),
        "zero": {
            "raw_return_mean": float(zero_returns.mean()),
            "raw_return_median": float(np.median(zero_returns)),
            "early_fall_rate": float(zero_falls.mean()),
            "episodes": zero_rows,
        },
        "families": family_rows,
        "aggregate": {
            "raw_return_mean": float(
                zero_returns.mean() + delta_array.mean()
            ),
            "paired_delta_mean": float(delta_array.mean()),
            "paired_delta_family_std": float(family_means.std()),
            "paired_delta_nested_bootstrap_ci95": list(nested_ci),
            "positive_family_fraction": family_positive_fraction,
            "early_fall_rate_delta": aggregate_fall_delta,
            "random_baseline_robust": robust,
        },
        "predeclared_gate": {
            "positive_family_fraction_min": 0.8,
            "paired_delta_nested_ci95_lower_strictly_positive": True,
            "aggregate_early_fall_rate_delta_max": 0.0,
        },
    }


def _predict_value(model: SeparatedClipPPO, observation: np.ndarray) -> float:
    tensor = torch.as_tensor(
        np.asarray(observation, dtype=np.float32)[None],
        device=model.device,
    )
    with torch.no_grad():
        value = model.policy.predict_values(tensor)
    return float(value.reshape(-1)[0].item())


def _counterfactual_candidate(
    *,
    environment: PPOResidualUnitEnv,
    observation: np.ndarray,
    snapshot: Any,
    residual: np.ndarray,
    model: SeparatedClipPPO,
    gamma: float,
    reward_scale: float,
    horizon: int,
) -> dict[str, Any]:
    environment.restore_state(snapshot)
    current_value = _predict_value(model, observation)
    (
        next_observation,
        reward,
        terminated,
        truncated,
        info,
    ) = environment.step(residual)
    bootstrap = (
        0.0
        if terminated
        else _predict_value(model, next_observation)
    )
    td_advantage = (
        reward_scale * float(reward)
        + gamma * bootstrap
        - current_value
    )
    discounted_return = float(reward)
    discount = float(gamma)
    steps = 1
    while (
        steps < int(horizon)
        and not terminated
        and not truncated
    ):
        zero = np.zeros(environment.action_space.shape, dtype=np.float32)
        (
            _,
            continuation_reward,
            terminated,
            truncated,
            _,
        ) = environment.step(zero)
        discounted_return += discount * float(continuation_reward)
        discount *= float(gamma)
        steps += 1
    return {
        "td_advantage": float(td_advantage),
        "counterfactual_discounted_return": float(discounted_return),
        "first_reward": float(reward),
        "steps": int(steps),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "action_residual_delta_l2": float(
            info["action_residual_delta_l2"]
        ),
    }


def _rankdata(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(array.size, dtype=np.float64)
    start = 0
    while start < array.size:
        end = start + 1
        while end < array.size and array[order[end]] == array[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def _spearman(predicted: np.ndarray, actual: np.ndarray) -> float | None:
    predicted_rank = _rankdata(predicted)
    actual_rank = _rankdata(actual)
    if predicted_rank.std() == 0 or actual_rank.std() == 0:
        return None
    return float(np.corrcoef(predicted_rank, actual_rank)[0, 1])


def state_ranking_metrics(row: dict[str, Any]) -> dict[str, Any]:
    candidates = row["candidates"]
    predicted = np.asarray(
        [candidate["td_advantage"] for candidate in candidates],
        dtype=np.float64,
    )
    actual = np.asarray(
        [
            candidate["counterfactual_discounted_return"]
            for candidate in candidates
        ],
        dtype=np.float64,
    )
    correct = 0
    comparisons = 0
    for left in range(len(candidates)):
        for right in range(left + 1, len(candidates)):
            true_delta = actual[left] - actual[right]
            predicted_delta = predicted[left] - predicted[right]
            if abs(true_delta) <= 1e-9 or abs(predicted_delta) <= 1e-12:
                continue
            comparisons += 1
            correct += int(np.sign(true_delta) == np.sign(predicted_delta))
    zero_correct = 0
    zero_comparisons = 0
    for index in range(1, len(candidates)):
        true_delta = actual[index] - actual[0]
        predicted_delta = predicted[index] - predicted[0]
        if abs(true_delta) <= 1e-9 or abs(predicted_delta) <= 1e-12:
            continue
        zero_comparisons += 1
        zero_correct += int(np.sign(true_delta) == np.sign(predicted_delta))
    true_best = np.flatnonzero(
        np.isclose(actual, actual.max(), rtol=0.0, atol=1e-9)
    )
    predicted_best = int(np.argmax(predicted))
    return {
        "spearman": _spearman(predicted, actual),
        "pairwise_correct": int(correct),
        "pairwise_comparisons": int(comparisons),
        "zero_random_correct": int(zero_correct),
        "zero_random_comparisons": int(zero_comparisons),
        "top1_correct": bool(predicted_best in true_best),
    }


def _aggregate_ranking_rows(
    rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    metrics = [state_ranking_metrics(row) for row in rows]
    spearman = np.asarray(
        [
            metric["spearman"]
            for metric in metrics
            if metric["spearman"] is not None
        ],
        dtype=np.float64,
    )
    pair_correct = sum(metric["pairwise_correct"] for metric in metrics)
    pair_count = sum(metric["pairwise_comparisons"] for metric in metrics)
    zero_correct = sum(metric["zero_random_correct"] for metric in metrics)
    zero_count = sum(metric["zero_random_comparisons"] for metric in metrics)
    return {
        "state_count": len(rows),
        "valid_spearman_state_count": int(spearman.size),
        "spearman_mean": (
            float(spearman.mean()) if spearman.size else None
        ),
        "pairwise_preference_accuracy": (
            float(pair_correct / pair_count) if pair_count else None
        ),
        "pairwise_correct": int(pair_correct),
        "pairwise_comparisons": int(pair_count),
        "zero_random_preference_accuracy": (
            float(zero_correct / zero_count) if zero_count else None
        ),
        "zero_random_correct": int(zero_correct),
        "zero_random_comparisons": int(zero_count),
        "top1_agreement": float(
            np.mean([metric["top1_correct"] for metric in metrics])
        ),
    }


def _ranking_bootstrap(
    rows: Sequence[dict[str, Any]],
    *,
    samples: int,
    seed: int,
) -> dict[str, list[float]]:
    rng = np.random.default_rng(int(seed))
    values = {
        "spearman_mean": [],
        "pairwise_preference_accuracy": [],
        "zero_random_preference_accuracy": [],
        "top1_agreement": [],
    }
    for _ in range(int(samples)):
        indices = rng.integers(0, len(rows), len(rows))
        aggregate = _aggregate_ranking_rows([rows[index] for index in indices])
        for key in values:
            value = aggregate[key]
            if value is not None:
                values[key].append(float(value))
    return {
        key: [
            float(np.quantile(metric_values, 0.025)),
            float(np.quantile(metric_values, 0.975)),
        ]
        for key, metric_values in values.items()
    }


def collect_advantage_ranking(
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
    horizon: int,
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
            while (
                not (terminated or truncated)
                and len(rows) < int(state_count)
            ):
                should_sample = global_step % int(sample_spacing) == 0
                if should_sample:
                    phase = int(environment.phase)
                    snapshot = environment.capture_state()
                    random_actions = residual_rng.normal(
                        0.0,
                        float(residual_std),
                        size=(
                            int(random_candidates),
                            *environment.action_space.shape,
                        ),
                    ).astype(np.float32)
                    random_actions = np.clip(
                        random_actions,
                        environment.action_space.low,
                        environment.action_space.high,
                    )
                    actions = [
                        np.zeros(
                            environment.action_space.shape,
                            dtype=np.float32,
                        ),
                        *list(random_actions),
                    ]
                    candidate_rows = []
                    for candidate_index, action in enumerate(actions):
                        result = _counterfactual_candidate(
                            environment=environment,
                            observation=observation,
                            snapshot=snapshot,
                            residual=action,
                            model=model,
                            gamma=float(gamma),
                            reward_scale=float(reward_scale),
                            horizon=int(horizon),
                        )
                        result.update(
                            {
                                "candidate_index": int(candidate_index),
                                "kind": (
                                    "zero"
                                    if candidate_index == 0
                                    else "random"
                                ),
                                "residual_unit": action.tolist(),
                            }
                        )
                        candidate_rows.append(result)
                    environment.restore_state(snapshot)
                    rows.append(
                        {
                            "state_index": len(rows),
                            "episode_index": int(episode_index),
                            "environment_seed": environment_seed,
                            "base_policy_seed": base_policy_seed,
                            "primitive_step": int(global_step),
                            "phase": phase,
                            "candidates": candidate_rows,
                        }
                    )
                zero = np.zeros(environment.action_space.shape, dtype=np.float32)
                (
                    observation,
                    _,
                    terminated,
                    truncated,
                    _,
                ) = environment.step(zero)
                global_step += 1
        finally:
            environment.close()
        episode_index += 1
    overall = _aggregate_ranking_rows(rows)
    confidence = _ranking_bootstrap(
        rows,
        samples=int(bootstrap_samples),
        seed=int(residual_seed) + 1,
    )
    by_phase = {
        str(phase): _aggregate_ranking_rows(
            [row for row in rows if int(row["phase"]) == phase]
        )
        for phase in range(ACTION_CHUNK)
    }
    phase_positive_count = sum(
        metrics["spearman_mean"] is not None
        and metrics["spearman_mean"] > 0.0
        for metrics in by_phase.values()
    )
    credible = bool(
        confidence["spearman_mean"][0] > 0.0
        and confidence["pairwise_preference_accuracy"][0] > 0.5
        and confidence["zero_random_preference_accuracy"][0] > 0.5
        and phase_positive_count >= 3
    )
    return {
        "protocol": (
            "matched simulator snapshot; zero plus random first-action "
            "interventions; identical restored frozen-base RNG; zero-residual "
            f"continuation for {int(horizon)} primitive steps"
        ),
        "prediction": (
            "training-scale one-step TD advantage "
            "r*scale + gamma*V(next_observation) - V(observation)"
        ),
        "target": "raw fixed-horizon discounted counterfactual return",
        "residual_std": float(residual_std),
        "reward_scale": float(reward_scale),
        "gamma": float(gamma),
        "horizon": int(horizon),
        "random_candidates_per_state": int(random_candidates),
        "sample_spacing": int(sample_spacing),
        "overall": overall,
        "confidence_interval_95": confidence,
        "by_phase": by_phase,
        "phase_positive_spearman_count": int(phase_positive_count),
        "advantage_ranking_credible": credible,
        "predeclared_gate": {
            "spearman_ci95_lower_strictly_positive": True,
            "pairwise_accuracy_ci95_lower_gt": 0.5,
            "zero_random_accuracy_ci95_lower_gt": 0.5,
            "positive_phase_spearman_count_min": 3,
        },
        "states": rows,
    }


def main() -> None:
    arguments = parse_args()
    if arguments.random_families < 2:
        raise ValueError("At least two random families are required")
    if arguments.evaluation_episodes <= 0:
        raise ValueError("Evaluation episode count must be positive")
    if arguments.counterfactual_states < ACTION_CHUNK:
        raise ValueError("Counterfactual states must cover all phases")
    if arguments.random_candidates < 2:
        raise ValueError("At least two random candidates are required")
    if arguments.counterfactual_horizon <= 0:
        raise ValueError("Counterfactual horizon must be positive")
    if arguments.bootstrap_samples < 100:
        raise ValueError("At least 100 bootstrap samples are required")
    output_directory = arguments.output_dir.resolve()
    if output_directory.exists():
        raise FileExistsError(f"Refusing to overwrite {output_directory}")
    output_directory.mkdir(parents=True)

    manifest = json.loads(arguments.run_manifest.resolve().read_text())
    if manifest["algorithm"] != "dsrl_na_residual_per_step_ppo_stable_v1":
        raise ValueError("Diagnostic requires the stable_v1 source run")
    checkpoint = arguments.checkpoint.resolve()
    if sha256_file(checkpoint) != str(manifest["checkpoint_sha256"]):
        raise ValueError("Frozen DSRL checkpoint does not match run manifest")
    residual_std = float(manifest["initial_residual_std"])
    reward_scale = float(manifest["training_reward_scale"])
    config_arguments = argparse.Namespace(
        checkpoint=checkpoint,
        equivalent_chunk_budget=10_000,
        seed=int(arguments.seed),
        n_envs=1,
        device=arguments.device,
    )
    cfg = compose_config(config_arguments)
    cfg.logdir = str(output_directory)
    if sha256_file(Path(cfg.base_policy_path)) != str(manifest["ddim_sha256"]):
        raise ValueError("Frozen DDIM does not match run manifest")
    if sha256_file(Path(cfg.normalization_path)) != str(
        manifest["normalization_sha256"]
    ):
        raise ValueError("Normalization does not match run manifest")
    diffusion = load_base_policy(cfg)
    contract = make_chunk_contract_environment(cfg)
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
    planner_hash_before = module_state_hash(planner_modules(legacy))
    residual_model = SeparatedClipPPO.load(
        arguments.residual_checkpoint.resolve(),
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
            int(arguments.environment_seed_start)
            + int(arguments.evaluation_episodes),
        )
    )
    base_policy_seeds = list(
        range(
            int(arguments.base_policy_seed_start),
            int(arguments.base_policy_seed_start)
            + int(arguments.evaluation_episodes),
        )
    )
    residual_family_seeds = list(
        range(
            int(arguments.residual_seed_start),
            int(arguments.residual_seed_start)
            + int(arguments.random_families),
        )
    )
    try:
        with isolated_rng():
            random_result = evaluate_random_families(
                make_environment=make_environment,
                environment_seeds=environment_seeds,
                base_policy_seeds=base_policy_seeds,
                residual_family_seeds=residual_family_seeds,
                residual_std=residual_std,
                bootstrap_samples=int(arguments.bootstrap_samples),
                bootstrap_seed=int(arguments.seed) + 100,
            )
            atomic_write_json(
                output_directory / "random_rng_robustness.json",
                random_result,
            )
            ranking_result = collect_advantage_ranking(
                model=residual_model,
                make_environment=make_environment,
                environment_seed_start=int(arguments.environment_seed_start),
                base_policy_seed_start=int(arguments.base_policy_seed_start),
                residual_seed=int(arguments.residual_seed_start) + 1000,
                state_count=int(arguments.counterfactual_states),
                random_candidates=int(arguments.random_candidates),
                residual_std=residual_std,
                gamma=float(cfg.train.discount) ** (1.0 / ACTION_CHUNK),
                reward_scale=reward_scale,
                horizon=int(arguments.counterfactual_horizon),
                sample_spacing=int(arguments.sample_spacing),
                bootstrap_samples=int(arguments.bootstrap_samples),
            )
            atomic_write_json(
                output_directory / "advantage_ranking.json",
                ranking_result,
            )
    finally:
        contract.close()
    planner_hash_after = module_state_hash(planner_modules(legacy))
    if planner_hash_after != planner_hash_before:
        raise RuntimeError("Frozen DSRL changed during diagnostics")
    implement_v3 = bool(
        random_result["aggregate"]["random_baseline_robust"]
        and ranking_result["advantage_ranking_credible"]
    )
    decision = {
        "random_baseline_robust": bool(
            random_result["aggregate"]["random_baseline_robust"]
        ),
        "advantage_ranking_credible": bool(
            ranking_result["advantage_ranking_credible"]
        ),
        "implement_variance_only_reference_kl_v3": implement_v3,
        "fallback_if_false": (
            None
            if implement_v3
            else "do not use PPO; evaluate risk gate or fixed random smoothing"
        ),
        "frozen_planner_hash_before": planner_hash_before,
        "frozen_planner_hash_after": planner_hash_after,
    }
    atomic_write_json(output_directory / "decision.json", decision)
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
