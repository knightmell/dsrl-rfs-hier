"""Exact-N, RNG-isolated P6 locomotion evaluation and persistent logging."""

from __future__ import annotations

import csv
import json
import math
import os
from contextlib import contextmanager, nullcontext
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
            "qa_base",
            "qa_base_target",
            "qw_base",
            "qa_joint",
            "qa_joint_target",
            "residual_actor",
            "residual_actor_target",
            "reference_noise_actor",
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
    evaluation_mode: str,
    matched_dsrl_model: Any | None,
) -> tuple[np.ndarray, Mapping[str, np.ndarray] | None]:
    if evaluation_mode == "matched_dsrl":
        if matched_dsrl_model is None:
            raise ValueError("matched_dsrl evaluation requires matched_dsrl_model")
        return _predict_legacy_action(
            matched_dsrl_model, observation, deterministic=deterministic
        )
    mode_map = {
        "current_base_only": "current_base_only",
        "current_full_hierarchy": "current_full_hierarchy",
        "reference_base": "reference_base",
    }
    if evaluation_mode not in mode_map:
        raise ValueError(f"Unknown evaluation_mode {evaluation_mode!r}")
    if hasattr(model, "predict_with_components"):
        components, _ = model.predict_with_components(
            observation,
            deterministic=deterministic,
            mode=mode_map[evaluation_mode],
        )
        return np.asarray(components["action_exec"]), components
    if evaluation_mode != "current_base_only":
        raise ValueError(
            "Legacy models support only current_base_only unless supplied as "
            "matched_dsrl_model"
        )
    return _predict_legacy_action(model, observation, deterministic=deterministic)


def _predict_legacy_action(
    model: Any,
    observation: np.ndarray,
    *,
    deterministic: bool,
) -> tuple[np.ndarray, None]:
    if not hasattr(model, "predict_diffused"):
        raise TypeError("P6 evaluator requires an executed-action prediction API")
    observation_array = np.asarray(observation)
    if observation_array.ndim == 0:
        raise ValueError("Legacy DSRL evaluation observation must not be scalar")
    # Legacy DSRL.predict_diffused() does not preserve the single-observation
    # batch convention used by BasePolicy.predict(): its diffusion wrapper
    # consumes ``obs`` directly while the sampled noise is always batched.
    # Batch explicitly here, then remove only the evaluator-owned batch axis.
    action_exec, _ = model.predict_diffused(
        observation_array[None, ...],
        deterministic=deterministic,
    )
    action_exec_array = np.asarray(action_exec)
    if action_exec_array.ndim < 2 or action_exec_array.shape[0] != 1:
        raise ValueError(
            "Legacy DSRL predict_diffused() must return exactly one batched action"
        )
    return action_exec_array[0], None


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
    evaluation_mode: str = "current_full_hierarchy",
    matched_dsrl_model: Any | None = None,
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
    max_chunk_steps = math.ceil(max_episode_primitive_steps / action_chunk)
    matched_context = (
        isolated_model_evaluation(matched_dsrl_model)
        if matched_dsrl_model is not None
        else nullcontext()
    )
    with isolated_rng(), isolated_model_evaluation(model), matched_context:
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
                        "residual_tanh_saturation_fraction": [],
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
                            evaluation_mode=evaluation_mode,
                            matched_dsrl_model=matched_dsrl_model,
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
                            unclamped = np.asarray(
                                components["action_exec_unclamped"]
                            )
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
                                float(np.mean(np.not_equal(unclamped, executed)))
                            )
                            # Post-tanh residual saturation, matching the
                            # training-side diagnostics fraction: the share of
                            # residual dimensions pinned near the |tanh| bounds
                            # (>= 0.99).  Saturation is the bang-bang rescue
                            # signature that healthy vs falling episodes should
                            # differ on.
                            decomposition[
                                "residual_tanh_saturation_fraction"
                            ].append(
                                float(
                                    np.mean(
                                        np.abs(
                                            np.asarray(
                                                components["residual_unit"]
                                            )
                                        )
                                        >= 0.99
                                    )
                                )
                            )

                    if episode_actual > max_episode_primitive_steps:
                        raise RuntimeError(
                            "Evaluation exceeded the audited primitive-step limit"
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

    def _outcome_slice_metrics(early_fall: bool) -> dict[str, float | None]:
        group = [episode for episode in episodes if episode["early_fall"] == early_fall]
        empty: dict[str, float | None] = {
            "episode_count": 0.0,
            "raw_return_mean": None,
            "d4rl_score_mean": None,
            "residual_tanh_saturation_fraction_mean": None,
            "effective_residual_l2_mean": None,
            "action_residual_delta_l2_mean": None,
        }
        if not group:
            return empty

        # None (not NaN) for absent decomposition data, so two identical runs
        # produce byte-comparable summaries (NaN != NaN would break it).
        def _mean(field: str) -> float | None:
            values = [
                float(episode[field])
                for episode in group
                if episode[field] is not None
            ]
            return _mean_or_nan(values) if values else None

        return {
            "episode_count": float(len(group)),
            "raw_return_mean": _mean("raw_return"),
            "d4rl_score_mean": _mean("d4rl_score"),
            "residual_tanh_saturation_fraction_mean": _mean(
                "residual_tanh_saturation_fraction"
            ),
            "effective_residual_l2_mean": _mean("effective_residual_l2"),
            "action_residual_delta_l2_mean": _mean(
                "action_residual_delta_l2"
            ),
        }

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
        # Residual saturation and effect split by episode outcome: the
        # bang-bang rescue signal is that falling episodes carry a large
        # saturating residual while healthy ones keep it near zero.
        "by_outcome": {
            "healthy": _outcome_slice_metrics(False),
            "fall": _outcome_slice_metrics(True),
        },
    }
    return {
        "protocol_version": 1,
        "evaluation_mode": evaluation_mode,
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
