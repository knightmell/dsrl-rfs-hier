"""Exact-N, RNG-isolated P6 locomotion evaluation and persistent logging."""

from __future__ import annotations

import csv
import json
import math
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from p6_runtime import atomic_write_json, isolated_rng, seed_all


def _model_modules(model: Any) -> list[nn.Module]:
    candidates = [
        getattr(model, name, None)
        for name in (
            "policy",
            "actor",
            "critic",
            "critic_target",
            "critic_noise",
            "critic_modulation",
            "residual_actor",
            "diffusion_policy",
        )
    ]
    diffusion_policy = getattr(model, "diffusion_policy", None)
    candidates.append(getattr(diffusion_policy, "base_policy", None))
    modules: list[nn.Module] = []
    seen: set[int] = set()
    for candidate in candidates:
        if isinstance(candidate, nn.Module) and id(candidate) not in seen:
            seen.add(id(candidate))
            modules.append(candidate)
    return modules


@contextmanager
def isolated_model_evaluation(model: Any) -> Iterator[None]:
    modules = _model_modules(model)
    modes = [module.training for module in modules]
    try:
        for module in modules:
            module.eval()
        yield
    finally:
        for module, training in zip(modules, modes):
            module.train(training)


def _normalized_d4rl_score(environment: Any, raw_return: float) -> float:
    current = environment
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        scorer = getattr(current, "get_normalized_score", None)
        if callable(scorer):
            return float(scorer(raw_return)) * 100.0
        current = getattr(current, "env", None)
    return float("nan")


def _predict_executed_action(
    model: Any,
    observation: np.ndarray,
    *,
    deterministic: bool,
) -> tuple[np.ndarray, Mapping[str, np.ndarray] | None]:
    if hasattr(model, "predict_with_components"):
        components, _ = model.predict_with_components(
            observation,
            deterministic=deterministic,
        )
        return np.asarray(components["action_exec"]), components
    if not hasattr(model, "predict_diffused"):
        raise TypeError("P6 evaluator requires an executed-action prediction API")
    action_exec, _ = model.predict_diffused(
        observation,
        deterministic=deterministic,
    )
    return np.asarray(action_exec), None


def _mean_or_nan(values: Sequence[float]) -> float:
    if not values:
        return float("nan")
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _atomic_write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty evaluation CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    field_names = list(rows[0])
    with temporary_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(rows)
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary_path, path)


def persist_evaluation(
    result: Mapping[str, Any],
    *,
    output_prefix: Path,
    tensorboard_writer: Any | None = None,
    tensorboard_tag: str = "eval",
) -> None:
    atomic_write_json(output_prefix.with_suffix(".json"), result)
    _atomic_write_csv(
        output_prefix.with_suffix(".csv"),
        list(result["episodes"]),
    )
    if tensorboard_writer is None:
        return
    chunk_step = int(result["chunk_transitions"])
    for metric in (
        "raw_return_mean",
        "d4rl_score_mean",
        "actual_primitive_length_mean",
        "early_fall_rate",
    ):
        tensorboard_writer.add_scalar(
            f"{tensorboard_tag}/{metric}",
            float(result["summary"][metric]),
            chunk_step,
        )
    tensorboard_writer.add_scalar(
        f"{tensorboard_tag}/chunk_transitions",
        chunk_step,
        chunk_step,
    )
    tensorboard_writer.add_scalar(
        f"{tensorboard_tag}/nominal_primitive_steps",
        int(result["nominal_primitive_steps"]),
        chunk_step,
    )
    tensorboard_writer.add_scalar(
        f"{tensorboard_tag}/actual_primitive_env_steps",
        int(result["actual_primitive_env_steps"]),
        chunk_step,
    )
    tensorboard_writer.flush()


def evaluate_exact_episodes(
    *,
    model: Any,
    make_environment: Callable[[], Any],
    environment_seeds: Sequence[int],
    policy_seed_start: int,
    deterministic: bool,
    action_chunk: int,
    max_episode_primitive_steps: int,
    batch_size: int = 1,
    chunk_transitions: int = 0,
    nominal_primitive_steps: int = 0,
    actual_primitive_env_steps: int = 0,
) -> dict[str, Any]:
    """Evaluate exactly one complete episode for every supplied seed.

    Episodes are intentionally run with independent policy RNG streams.  The
    ``batch_size`` argument controls grouping only; execution remains
    per-episode so results cannot depend on other episodes ending early.
    """

    if not environment_seeds:
        raise ValueError("At least one evaluation seed is required")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if action_chunk <= 0 or max_episode_primitive_steps <= 0:
        raise ValueError("Action chunk and episode limit must be positive")

    episodes: list[dict[str, Any]] = []
    max_chunk_steps = math.ceil(max_episode_primitive_steps / action_chunk) + 1
    with isolated_rng(), isolated_model_evaluation(model):
        for batch_start in range(0, len(environment_seeds), batch_size):
            seed_batch = environment_seeds[
                batch_start : batch_start + batch_size
            ]
            for local_index, environment_seed in enumerate(seed_batch):
                episode_index = batch_start + local_index
                policy_seed = int(policy_seed_start) + episode_index
                environment = make_environment()
                try:
                    seed_all(policy_seed)
                    reset_result = environment.reset(seed=int(environment_seed))
                    observation = (
                        reset_result[0]
                        if isinstance(reset_result, tuple)
                        else reset_result
                    )
                    raw_return = 0.0
                    episode_chunks = 0
                    episode_nominal = 0
                    episode_actual = 0
                    early_termination_chunks = 0
                    terminated = False
                    truncated = False
                    decomposition: dict[str, list[float]] = {
                        "noise_scaled_l2": [],
                        "action_base_l2": [],
                        "action_residual_delta_l2": [],
                        "effective_residual_l2": [],
                        "clip_fraction": [],
                    }
                    while not (terminated or truncated):
                        if episode_chunks >= max_chunk_steps:
                            raise RuntimeError(
                                "Evaluation episode exceeded its audited "
                                "primitive-step limit"
                            )
                        action_exec, components = _predict_executed_action(
                            model,
                            np.asarray(observation),
                            deterministic=deterministic,
                        )
                        (
                            observation,
                            reward,
                            terminated,
                            truncated,
                            info,
                        ) = environment.step(action_exec)
                        nominal = int(info["nominal_primitive_steps"])
                        actual = int(info["actual_primitive_steps"])
                        if nominal != action_chunk or not (1 <= actual <= nominal):
                            raise ValueError("Invalid evaluator primitive counters")
                        episode_chunks += 1
                        episode_nominal += nominal
                        episode_actual += actual
                        early_termination_chunks += int(
                            info["early_termination_within_chunk"]
                        )
                        raw_return += float(reward)

                        if components is not None:
                            action_base = np.asarray(components["action_base"])
                            residual_delta = np.asarray(
                                components["action_residual_delta"]
                            )
                            executed = np.asarray(components["action_exec"])
                            pre_clip = np.asarray(components["action_pre_clip"])
                            decomposition["noise_scaled_l2"].append(
                                float(np.linalg.norm(components["noise_scaled"]))
                            )
                            decomposition["action_base_l2"].append(
                                float(np.linalg.norm(action_base))
                            )
                            decomposition["action_residual_delta_l2"].append(
                                float(np.linalg.norm(residual_delta))
                            )
                            decomposition["effective_residual_l2"].append(
                                float(np.linalg.norm(executed - action_base))
                            )
                            decomposition["clip_fraction"].append(
                                float(np.mean(np.not_equal(pre_clip, executed)))
                            )

                    early_fall = bool(
                        terminated
                        and not truncated
                        and episode_actual < max_episode_primitive_steps
                    )
                    row = {
                        "episode_index": episode_index,
                        "environment_seed": int(environment_seed),
                        "policy_seed": policy_seed,
                        "raw_return": raw_return,
                        "d4rl_score": _normalized_d4rl_score(
                            environment,
                            raw_return,
                        ),
                        "chunk_length": episode_chunks,
                        "nominal_primitive_length": episode_nominal,
                        "actual_primitive_length": episode_actual,
                        "early_termination_chunk_count": early_termination_chunks,
                        "terminated": bool(terminated),
                        "truncated": bool(truncated),
                        "early_fall": early_fall,
                        **{
                            key: _mean_or_nan(values) if values else None
                            for key, values in decomposition.items()
                        },
                    }
                    episodes.append(row)
                finally:
                    environment.close()

    if len(episodes) != len(environment_seeds):
        raise RuntimeError("Exact-N evaluator did not collect the requested episodes")
    summary = {
        "episode_count": len(episodes),
        "raw_return_mean": _mean_or_nan(
            [float(episode["raw_return"]) for episode in episodes]
        ),
        "raw_return_std": float(
            np.std([float(episode["raw_return"]) for episode in episodes])
        ),
        "d4rl_score_mean": _mean_or_nan(
            [float(episode["d4rl_score"]) for episode in episodes]
        ),
        "actual_primitive_length_mean": _mean_or_nan(
            [float(episode["actual_primitive_length"]) for episode in episodes]
        ),
        "early_fall_rate": _mean_or_nan(
            [float(episode["early_fall"]) for episode in episodes]
        ),
    }
    return {
        "protocol_version": 1,
        "exact_episode_count": len(episodes),
        "deterministic": bool(deterministic),
        "policy_seed_start": int(policy_seed_start),
        "environment_seeds": [int(seed) for seed in environment_seeds],
        "requested_batch_size": int(batch_size),
        "chunk_transitions": int(chunk_transitions),
        "nominal_primitive_steps": int(nominal_primitive_steps),
        "actual_primitive_env_steps": int(actual_primitive_env_steps),
        "summary": summary,
        "episodes": episodes,
    }
