"""Offline value probe for hidden future-plan aliasing in residual PPO.

This diagnostic never updates the frozen DSRL planner or the residual PPO
policy.  It collects complete episodes, constructs two views of every state,
and predicts a fixed-horizon discounted return from identical transitions:

old:
    [observation | current base action | phase]
plan:
    [observation | left-aligned remaining base plan | mask | phase]
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / "dppo"), str(ROOT / "stable-baselines3")]

from p6_preflight import sha256_file  # noqa: E402
from p6_runtime import atomic_write_json, isolated_rng, seed_all  # noqa: E402
from p6_train import _load_legacy_network  # noqa: E402
from per_step_residual_env import FrozenDSRLChunkPlanner  # noqa: E402
from per_step_residual_ppo import CountingPPO  # noqa: E402
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


@dataclass(frozen=True)
class ProbeDataset:
    old_features: np.ndarray
    plan_features: np.ndarray
    targets: np.ndarray
    episode_ids: np.ndarray
    phases: np.ndarray
    early_falls: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--residual-checkpoint", type=Path, required=True)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episode-count", type=int, default=100)
    parser.add_argument("--environment-seed-start", type=int, default=10_000)
    parser.add_argument("--policy-seed-start", type=int, default=20_000)
    parser.add_argument("--return-horizon", type=int, default=64)
    parser.add_argument("--probe-updates", type=int, default=1_500)
    parser.add_argument("--probe-batch-size", type=int, default=1_024)
    parser.add_argument("--probe-learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--probe-device", default="cuda:0")
    parser.add_argument("--deterministic-base", action="store_true")
    parser.add_argument("--deterministic-residual", action="store_true")
    return parser.parse_args()


def _remaining_plan(
    action_base_chunk: np.ndarray,
    phase: int,
) -> tuple[np.ndarray, np.ndarray]:
    chunk = np.asarray(action_base_chunk, dtype=np.float32)
    if chunk.shape != (ACTION_CHUNK, ACTION_DIMENSION):
        raise ValueError(f"Unexpected base chunk shape {chunk.shape}")
    if not 0 <= phase < ACTION_CHUNK:
        raise ValueError(f"Invalid chunk phase {phase}")
    remaining = np.zeros_like(chunk)
    mask = np.zeros(ACTION_CHUNK, dtype=np.float32)
    count = ACTION_CHUNK - phase
    remaining[:count] = chunk[phase:]
    mask[:count] = 1.0
    return remaining, mask


def _fixed_horizon_returns(
    rewards: Sequence[float],
    *,
    gamma: float,
    horizon: int,
    terminal: bool,
) -> tuple[np.ndarray, np.ndarray]:
    reward_array = np.asarray(rewards, dtype=np.float64)
    targets = np.zeros(len(reward_array), dtype=np.float64)
    valid = np.zeros(len(reward_array), dtype=bool)
    for start in range(len(reward_array)):
        available = len(reward_array) - start
        if available < horizon and not terminal:
            continue
        count = min(horizon, available)
        discounts = np.power(gamma, np.arange(count, dtype=np.float64))
        targets[start] = float(
            np.dot(discounts, reward_array[start : start + count])
        )
        valid[start] = True
    return targets.astype(np.float32), valid


def collect_dataset(
    *,
    model: CountingPPO,
    make_environment: Any,
    episode_count: int,
    environment_seed_start: int,
    policy_seed_start: int,
    gamma: float,
    return_horizon: int,
    deterministic_residual: bool,
) -> tuple[ProbeDataset, list[dict[str, Any]]]:
    old_rows: list[np.ndarray] = []
    plan_rows: list[np.ndarray] = []
    target_rows: list[float] = []
    episode_rows: list[int] = []
    phase_rows: list[int] = []
    early_rows: list[bool] = []
    episodes: list[dict[str, Any]] = []
    model.policy.set_training_mode(False)
    with isolated_rng():
        for episode_id in range(episode_count):
            environment = make_environment()
            primitive = environment.env
            try:
                environment_seed = environment_seed_start + episode_id
                policy_seed = policy_seed_start + episode_id
                seed_all(policy_seed)
                observation, _ = environment.reset(seed=environment_seed)
                episode_old: list[np.ndarray] = []
                episode_plan: list[np.ndarray] = []
                episode_phases: list[int] = []
                rewards: list[float] = []
                terminated = truncated = False
                while not (terminated or truncated):
                    phase = int(primitive.phase)
                    phase_one_hot = np.zeros(ACTION_CHUNK, dtype=np.float32)
                    phase_one_hot[phase] = 1.0
                    remaining, mask = _remaining_plan(
                        primitive.current_base_action_chunk,
                        phase,
                    )
                    observation_array = np.asarray(
                        observation,
                        dtype=np.float32,
                    )
                    old = observation_array[
                        : primitive.noise_log_prob_index
                    ].copy()
                    state = observation_array[: primitive.raw_observation_dim]
                    plan = np.concatenate(
                        (
                            state,
                            remaining.reshape(-1),
                            mask,
                            phase_one_hot,
                        )
                    ).astype(np.float32, copy=False)
                    residual, _ = model.predict(
                        observation_array,
                        deterministic=deterministic_residual,
                    )
                    (
                        observation,
                        reward,
                        terminated,
                        truncated,
                        _,
                    ) = environment.step(residual)
                    episode_old.append(old)
                    episode_plan.append(plan)
                    episode_phases.append(phase)
                    rewards.append(float(reward))

                early_fall = bool(
                    terminated
                    and not truncated
                    and len(rewards) < primitive.max_episode_steps
                )
                targets, valid = _fixed_horizon_returns(
                    rewards,
                    gamma=gamma,
                    horizon=return_horizon,
                    terminal=bool(terminated and not truncated),
                )
                for transition_index in np.flatnonzero(valid):
                    old_rows.append(episode_old[transition_index])
                    plan_rows.append(episode_plan[transition_index])
                    target_rows.append(float(targets[transition_index]))
                    episode_rows.append(episode_id)
                    phase_rows.append(episode_phases[transition_index])
                    early_rows.append(early_fall)
                episodes.append(
                    {
                        "episode_id": episode_id,
                        "environment_seed": environment_seed,
                        "policy_seed": policy_seed,
                        "primitive_length": len(rewards),
                        "early_fall": early_fall,
                        "valid_probe_transitions": int(valid.sum()),
                        "raw_return": float(sum(rewards)),
                    }
                )
            finally:
                environment.close()
    return (
        ProbeDataset(
            old_features=np.stack(old_rows).astype(np.float32),
            plan_features=np.stack(plan_rows).astype(np.float32),
            targets=np.asarray(target_rows, dtype=np.float32),
            episode_ids=np.asarray(episode_rows, dtype=np.int64),
            phases=np.asarray(phase_rows, dtype=np.int64),
            early_falls=np.asarray(early_rows, dtype=bool),
        ),
        episodes,
    )


def split_episode_ids(
    episodes: Sequence[dict[str, Any]],
    *,
    seed: int,
) -> dict[str, list[int]]:
    random = np.random.RandomState(seed)
    groups = {
        False: [
            int(row["episode_id"])
            for row in episodes
            if not row["early_fall"]
        ],
        True: [
            int(row["episode_id"])
            for row in episodes
            if row["early_fall"]
        ],
    }
    splits: dict[str, list[int]] = {"train": [], "validation": [], "test": []}
    for identifiers in groups.values():
        identifiers = list(identifiers)
        random.shuffle(identifiers)
        train_end = int(round(0.6 * len(identifiers)))
        validation_end = train_end + int(round(0.2 * len(identifiers)))
        splits["train"].extend(identifiers[:train_end])
        splits["validation"].extend(identifiers[train_end:validation_end])
        splits["test"].extend(identifiers[validation_end:])
    for identifiers in splits.values():
        identifiers.sort()
    if any(not identifiers for identifiers in splits.values()):
        raise RuntimeError("Every probe split must contain complete episodes")
    return splits


class ValueProbe(nn.Module):
    def __init__(self, input_dimension: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dimension, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


def _explained_variance(target: np.ndarray, prediction: np.ndarray) -> float:
    variance = float(np.var(target))
    if variance <= 1e-12:
        return float("nan")
    return float(1.0 - np.var(target - prediction) / variance)


def _metrics(
    target: np.ndarray,
    prediction: np.ndarray,
    *,
    phases: np.ndarray,
    early_falls: np.ndarray,
) -> dict[str, Any]:
    def group(mask: np.ndarray) -> dict[str, Any] | None:
        if int(mask.sum()) == 0:
            return None
        difference = target[mask] - prediction[mask]
        return {
            "count": int(mask.sum()),
            "mse": float(np.mean(np.square(difference))),
            "mae": float(np.mean(np.abs(difference))),
            "explained_variance": _explained_variance(
                target[mask],
                prediction[mask],
            ),
        }

    return {
        "overall": group(np.ones(len(target), dtype=bool)),
        "by_phase": {
            str(phase): group(phases == phase)
            for phase in range(ACTION_CHUNK)
        },
        "early_fall": group(early_falls),
        "nonfall": group(~early_falls),
    }


def train_probe(
    *,
    features: np.ndarray,
    dataset: ProbeDataset,
    splits: dict[str, list[int]],
    updates: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    masks = {
        name: np.isin(dataset.episode_ids, identifiers)
        for name, identifiers in splits.items()
    }
    train_features = features[masks["train"]]
    train_targets = dataset.targets[masks["train"]]
    feature_mean = train_features.mean(axis=0)
    feature_std = train_features.std(axis=0)
    feature_std = np.maximum(feature_std, 1e-6)
    target_mean = float(train_targets.mean())
    target_std = max(float(train_targets.std()), 1e-6)

    normalized_features = (features - feature_mean) / feature_std
    normalized_targets = (dataset.targets - target_mean) / target_std
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = ValueProbe(features.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_indices = np.flatnonzero(masks["train"])
    validation_indices = np.flatnonzero(masks["validation"])
    random = np.random.RandomState(seed + 1_000)
    best_validation = float("inf")
    best_update = 0
    best_state = copy.deepcopy(model.state_dict())
    features_tensor = torch.as_tensor(
        normalized_features,
        dtype=torch.float32,
        device=device,
    )
    targets_tensor = torch.as_tensor(
        normalized_targets,
        dtype=torch.float32,
        device=device,
    )
    validation_tensor = torch.as_tensor(
        validation_indices,
        dtype=torch.long,
        device=device,
    )
    for update in range(1, updates + 1):
        sampled = random.choice(
            train_indices,
            size=min(batch_size, len(train_indices)),
            replace=False,
        )
        indices = torch.as_tensor(sampled, dtype=torch.long, device=device)
        prediction = model(features_tensor[indices])
        loss = torch.mean(torch.square(prediction - targets_tensor[indices]))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if update == 1 or update % 25 == 0 or update == updates:
            model.eval()
            with torch.no_grad():
                validation_loss = float(
                    torch.mean(
                        torch.square(
                            model(features_tensor[validation_tensor])
                            - targets_tensor[validation_tensor]
                        )
                    ).item()
                )
            model.train()
            if validation_loss < best_validation:
                best_validation = validation_loss
                best_update = update
                best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    model.eval()
    predictions: dict[str, np.ndarray] = {}
    metrics: dict[str, Any] = {
        "input_dimension": int(features.shape[1]),
        "best_update": int(best_update),
        "best_validation_normalized_mse": best_validation,
    }
    with torch.no_grad():
        all_prediction = (
            model(features_tensor).cpu().numpy() * target_std + target_mean
        )
    for name, mask in masks.items():
        predictions[name] = all_prediction[mask]
        metrics[name] = _metrics(
            dataset.targets[mask],
            all_prediction[mask],
            phases=dataset.phases[mask],
            early_falls=dataset.early_falls[mask],
        )
    normalization = {
        "feature_mean": feature_mean.astype(np.float32),
        "feature_std": feature_std.astype(np.float32),
        "target_mean": np.asarray([target_mean], dtype=np.float32),
        "target_std": np.asarray([target_std], dtype=np.float32),
    }
    return metrics, normalization


def probe_decision(
    old_metrics: dict[str, Any],
    plan_metrics: dict[str, Any],
) -> dict[str, Any]:
    old_test = old_metrics["test"]["overall"]
    plan_test = plan_metrics["test"]["overall"]
    ev_gain = (
        plan_test["explained_variance"] - old_test["explained_variance"]
    )
    mse_reduction = 1.0 - plan_test["mse"] / old_test["mse"]
    improved_phases = sum(
        plan_metrics["test"]["by_phase"][str(phase)]["mse"]
        < old_metrics["test"]["by_phase"][str(phase)]["mse"]
        for phase in range(ACTION_CHUNK)
    )
    old_early = old_metrics["test"]["early_fall"]
    plan_early = plan_metrics["test"]["early_fall"]
    early_not_worse = (
        old_early is None
        or plan_early is None
        or plan_early["mse"] <= 1.1 * old_early["mse"]
    )
    supported = bool(
        ev_gain >= 0.05
        and mse_reduction >= 0.10
        and improved_phases >= 3
        and early_not_worse
    )
    return {
        "probe_supports_plan_context": supported,
        "test_explained_variance_gain": float(ev_gain),
        "test_mse_reduction_fraction": float(mse_reduction),
        "test_phase_mse_improvement_count": int(improved_phases),
        "test_early_fall_mse_not_worse_by_more_than_10pct": bool(
            early_not_worse
        ),
        "predeclared_thresholds": {
            "explained_variance_gain_min": 0.05,
            "mse_reduction_fraction_min": 0.10,
            "phase_mse_improvement_count_min": 3,
            "early_fall_mse_max_ratio": 1.10,
        },
    }


def main() -> None:
    arguments = parse_args()
    if arguments.episode_count < 15:
        raise ValueError("At least 15 complete episodes are required")
    if arguments.return_horizon <= 0 or arguments.probe_updates <= 0:
        raise ValueError("Probe horizon and updates must be positive")
    output_directory = arguments.output_dir.resolve()
    if output_directory.exists():
        raise FileExistsError(f"Refusing to overwrite {output_directory}")
    output_directory.mkdir(parents=True)

    manifest = json.loads(arguments.run_manifest.resolve().read_text())
    checkpoint = arguments.checkpoint.resolve()
    residual_checkpoint = arguments.residual_checkpoint.resolve()
    checkpoint_hash = sha256_file(checkpoint)
    if checkpoint_hash != str(manifest["checkpoint_sha256"]):
        raise ValueError("DSRL checkpoint does not match the source run")
    residual_scale = float(manifest["residual_scale"])
    config_arguments = argparse.Namespace(
        checkpoint=checkpoint,
        equivalent_chunk_budget=10_000,
        seed=int(arguments.seed),
        n_envs=1,
        device=arguments.device,
    )
    cfg = compose_config(config_arguments)
    cfg.logdir = str(output_directory)
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
    residual_model = CountingPPO.load(
        residual_checkpoint,
        device=arguments.device,
    )
    make_environment = lambda: make_ppo_environment(
        cfg,
        planner,
        residual_scale=residual_scale,
        deterministic_base=bool(arguments.deterministic_base),
    )
    try:
        dataset, episodes = collect_dataset(
            model=residual_model,
            make_environment=make_environment,
            episode_count=int(arguments.episode_count),
            environment_seed_start=int(arguments.environment_seed_start),
            policy_seed_start=int(arguments.policy_seed_start),
            gamma=float(cfg.train.discount) ** (1.0 / ACTION_CHUNK),
            return_horizon=int(arguments.return_horizon),
            deterministic_residual=bool(arguments.deterministic_residual),
        )
    finally:
        contract.close()
    planner_hash_after = module_state_hash(planner_modules(legacy))
    if planner_hash_after != planner_hash_before:
        raise RuntimeError("Frozen DSRL changed during diagnostic collection")

    splits = split_episode_ids(episodes, seed=int(arguments.seed))
    probe_device = torch.device(arguments.probe_device)
    old_metrics, old_normalization = train_probe(
        features=dataset.old_features,
        dataset=dataset,
        splits=splits,
        updates=int(arguments.probe_updates),
        batch_size=int(arguments.probe_batch_size),
        learning_rate=float(arguments.probe_learning_rate),
        seed=int(arguments.seed) + 100,
        device=probe_device,
    )
    plan_metrics, plan_normalization = train_probe(
        features=dataset.plan_features,
        dataset=dataset,
        splits=splits,
        updates=int(arguments.probe_updates),
        batch_size=int(arguments.probe_batch_size),
        learning_rate=float(arguments.probe_learning_rate),
        seed=int(arguments.seed) + 100,
        device=probe_device,
    )
    decision = probe_decision(old_metrics, plan_metrics)
    np.savez_compressed(
        output_directory / "probe_dataset.npz",
        old_features=dataset.old_features,
        plan_features=dataset.plan_features,
        targets=dataset.targets,
        episode_ids=dataset.episode_ids,
        phases=dataset.phases,
        early_falls=dataset.early_falls,
        old_feature_mean=old_normalization["feature_mean"],
        old_feature_std=old_normalization["feature_std"],
        plan_feature_mean=plan_normalization["feature_mean"],
        plan_feature_std=plan_normalization["feature_std"],
        target_mean=old_normalization["target_mean"],
        target_std=old_normalization["target_std"],
    )
    result = {
        "protocol": {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_hash,
            "residual_checkpoint": str(residual_checkpoint),
            "residual_checkpoint_sha256": sha256_file(residual_checkpoint),
            "source_manifest": str(arguments.run_manifest.resolve()),
            "episode_count": int(arguments.episode_count),
            "environment_seed_start": int(arguments.environment_seed_start),
            "policy_seed_start": int(arguments.policy_seed_start),
            "deterministic_base": bool(arguments.deterministic_base),
            "deterministic_residual": bool(arguments.deterministic_residual),
            "residual_scale": residual_scale,
            "return_target": "fixed-horizon discounted primitive reward",
            "return_horizon": int(arguments.return_horizon),
            "gamma_primitive": float(cfg.train.discount)
            ** (1.0 / ACTION_CHUNK),
            "probe_updates": int(arguments.probe_updates),
            "probe_batch_size": int(arguments.probe_batch_size),
            "probe_learning_rate": float(arguments.probe_learning_rate),
            "planner_hash_before": planner_hash_before,
            "planner_hash_after": planner_hash_after,
            "old_features": (
                "normalized observation | current base action | phase one-hot"
            ),
            "plan_features": (
                "normalized observation | left-aligned remaining base plan | "
                "remaining mask | phase one-hot"
            ),
        },
        "splits": splits,
        "episode_summary": {
            "early_fall_count": int(
                sum(row["early_fall"] for row in episodes)
            ),
            "mean_raw_return": float(
                np.mean([row["raw_return"] for row in episodes])
            ),
            "mean_primitive_length": float(
                np.mean([row["primitive_length"] for row in episodes])
            ),
            "transition_count": int(len(dataset.targets)),
        },
        "episodes": episodes,
        "old_probe": old_metrics,
        "plan_probe": plan_metrics,
        "decision": decision,
    }
    atomic_write_json(output_directory / "probe_results.json", result)
    print(json.dumps(result["episode_summary"], indent=2, sort_keys=True))
    print(json.dumps(decision, indent=2, sort_keys=True))
    print(f"artifact={output_directory / 'probe_results.json'}")


if __name__ == "__main__":
    main()
