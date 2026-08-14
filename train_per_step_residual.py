"""Matched primitive-step frozen-DSRL residual experiments.

Budgets on the CLI are expressed in equivalent action-chunk transitions.  With
H=4 the learner receives four primitive transitions per equivalent chunk.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import gym
import hydra
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / "dppo"), str(ROOT / "stable-baselines3")]

import d4rl  # noqa: E402,F401
import d4rl.gym_mujoco  # noqa: E402,F401
from env_utils import ObservationWrapperGym  # noqa: E402
from p6_preflight import sha256_file  # noqa: E402
from p6_runtime import (  # noqa: E402
    atomic_write_json,
    capture_rng_state,
    hash_semantic_arrays,
    isolated_rng,
    restore_rng_state,
    seed_all,
    terminal_corrected_next_observations,
)
from p6_train import (  # noqa: E402
    _load_legacy_network,
    _make_hopper_environment,
    _make_policy_kwargs,
)
from per_step_residual_dsrl import PerStepResidualDSRL  # noqa: E402
from per_step_residual_env import (  # noqa: E402
    FrozenDSRLChunkPlanner,
    FrozenDSRLPrimitiveResidualEnv,
)
from stable_baselines3.common.callbacks import BaseCallback  # noqa: E402
from stable_baselines3.common.env_util import make_vec_env  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv  # noqa: E402
from utils import load_base_policy  # noqa: E402


ALGORITHM = "dsrl_na_residual_per_step_qa"
ACTION_CHUNK = 4
ACTION_DIMENSION = 3
RAW_OBSERVATION_DIMENSION = 11
PREFILL_ARRAY_NAMES = (
    "observations",
    "next_observations",
    "actions",
    "rewards",
    "dones",
    "timeouts",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--equivalent-chunk-budget", type=int, required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--n-envs", type=int, default=10)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prefill-vector-chunks", type=int, default=2001)
    parser.add_argument("--critic-pretrain-steps", type=int, default=10000)
    parser.add_argument("--final-eval-episodes", type=int, default=100)
    parser.add_argument("--online-eval-episodes", type=int, default=10)
    parser.add_argument("--checkpoint-interval-chunks", type=int, default=100000)
    parser.add_argument("--replay-interval-chunks", type=int, default=500000)
    parser.add_argument("--online-eval-interval-chunks", type=int, default=100000)
    parser.add_argument("--stop-at-equivalent-chunks", type=int)
    parser.add_argument("--resume-bundle", type=Path)
    parser.add_argument("--wiring-cadence", action="store_true")
    parser.add_argument("--deterministic-base", action="store_true")
    return parser.parse_args()


def git_state(path: Path) -> dict[str, Any]:
    def run(*arguments: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(path), *arguments],
            text=True,
        ).strip()

    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "status": run("status", "--short"),
    }


def module_state_hash(modules: Mapping[str, torch.nn.Module]) -> str:
    digest = hashlib.sha256()
    for module_name in sorted(modules):
        digest.update(module_name.encode())
        state = modules[module_name].state_dict()
        for name in sorted(state):
            value = state[name].detach().cpu().contiguous()
            digest.update(name.encode())
            digest.update(str(value.dtype).encode())
            digest.update(json.dumps(list(value.shape)).encode())
            digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def planner_modules(legacy: Any) -> dict[str, torch.nn.Module]:
    modules = {
        "actor": legacy.actor,
        "critic": legacy.critic,
        "critic_target": legacy.critic_target,
        "critic_noise": legacy.critic_noise,
    }
    diffusion = legacy.diffusion_policy
    if isinstance(diffusion, torch.nn.Module):
        modules["diffusion_policy"] = diffusion
    base = getattr(diffusion, "base_policy", None)
    if isinstance(base, torch.nn.Module):
        modules["diffusion_base_policy"] = base
    return modules


def compose_config(arguments: argparse.Namespace) -> Any:
    checkpoint_override = (
        str(arguments.checkpoint.resolve())
        if arguments.checkpoint is not None
        else "null"
    )
    with initialize_config_dir(
        config_dir=str(ROOT / "cfg/gym"),
        version_base=None,
    ):
        cfg = compose(
            config_name="p6_hopper",
            overrides=[
                f"rfs_hier_legacy_checkpoint_path={checkpoint_override}",
                f"total_timesteps={arguments.equivalent_chunk_budget}",
                f"seed={arguments.seed}",
                f"env.n_envs={arguments.n_envs}",
                f"p6.pilot_n_envs={arguments.n_envs}",
                f"device={arguments.device}",
                "use_wandb=false",
            ],
        )
    OmegaConf.resolve(cfg)
    with open_dict(cfg):
        cfg.model.network_path = str((ROOT / str(cfg.base_policy_path)).resolve())
        cfg.base_policy_path = cfg.model.network_path
        cfg.normalization_path = str(
            (ROOT / str(cfg.normalization_path)).resolve()
        )
    return cfg


def make_chunk_contract_environment(cfg: Any):
    return make_vec_env(
        lambda: _make_hopper_environment(
            cfg,
            Path(cfg.normalization_path),
        ),
        n_envs=1,
        vec_env_cls=DummyVecEnv,
    )


def make_primitive_environment(
    cfg: Any,
    planner: FrozenDSRLChunkPlanner,
    *,
    deterministic_base: bool,
) -> FrozenDSRLPrimitiveResidualEnv:
    primitive = ObservationWrapperGym(
        gym.make(cfg.env_name),
        cfg.normalization_path,
    )
    return FrozenDSRLPrimitiveResidualEnv(
        primitive,
        planner,
        raw_observation_dim=RAW_OBSERVATION_DIMENSION,
        max_episode_steps=int(cfg.env.max_episode_steps),
        deterministic_base=deterministic_base,
    )


def make_training_environment(
    cfg: Any,
    planner: FrozenDSRLChunkPlanner,
    *,
    n_envs: int,
    seed: int,
    deterministic_base: bool,
):
    return make_vec_env(
        lambda: make_primitive_environment(
            cfg,
            planner,
            deterministic_base=deterministic_base,
        ),
        n_envs=n_envs,
        seed=seed,
        vec_env_cls=DummyVecEnv,
    )


def build_model(
    cfg: Any,
    environment: Any,
    *,
    alpha: float,
    buffer_size: int,
) -> PerStepResidualDSRL:
    if int(cfg.act_steps) != ACTION_CHUNK or int(cfg.action_dim) != ACTION_DIMENSION:
        raise ValueError("Per-step experiment requires audited Hopper H=4, A=3")
    wrapped = environment.envs[0].unwrapped
    if not isinstance(wrapped, FrozenDSRLPrimitiveResidualEnv):
        raise TypeError("Unexpected primitive training environment")
    gamma_primitive = float(cfg.train.discount) ** (1.0 / ACTION_CHUNK)
    train_frequency_primitive = int(cfg.train.train_freq) * ACTION_CHUNK
    return PerStepResidualDSRL(
        "MlpPolicy",
        environment,
        learning_rate=float(cfg.train.actor_lr),
        buffer_size=int(buffer_size),
        learning_starts=0,
        batch_size=int(cfg.train.batch_size),
        tau=float(cfg.train.tau),
        gamma=gamma_primitive,
        train_freq=train_frequency_primitive,
        gradient_steps=int(cfg.train.utd),
        action_noise=None,
        optimize_memory_usage=False,
        tensorboard_log=str(Path(cfg.logdir) / "tensorboard"),
        policy_kwargs=_make_policy_kwargs(cfg),
        verbose=1,
        seed=int(cfg.seed),
        device=cfg.device,
        raw_observation_dim=RAW_OBSERVATION_DIMENSION,
        base_action_start=wrapped.base_action_start,
        phase_start=wrapped.phase_start,
        phase_count=ACTION_CHUNK,
        noise_log_prob_index=wrapped.noise_log_prob_index,
        exec_action_low=np.asarray(environment.action_space.low, dtype=np.float32),
        exec_action_high=np.asarray(environment.action_space.high, dtype=np.float32),
        noise_entropy_coefficient=alpha,
        residual_scale=float(cfg.train.rfs_hier_residual_scale),
        residual_net_arch=tuple(cfg.train.rfs_hier_residual_net_arch),
        residual_lr=float(cfg.train.rfs_hier_residual_lr),
        residual_penalty_coef=float(cfg.train.rfs_hier_residual_penalty_coef),
        residual_actor_gradient_steps=int(
            cfg.train.rfs_hier_residual_actor_gradient_steps
        ),
        diagnostics_interval_train_calls=int(
            cfg.p6.diagnostics_interval_updates
        ),
    )


def prefill_metadata(
    *,
    arguments: argparse.Namespace,
    cfg: Any,
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "algorithm": ALGORITHM,
        "checkpoint_sha256": sha256_file(arguments.checkpoint.resolve()),
        "ddim_sha256": sha256_file(Path(cfg.base_policy_path)),
        "normalization_sha256": sha256_file(Path(cfg.normalization_path)),
        "n_envs": int(arguments.n_envs),
        "vector_primitive_steps": int(arguments.prefill_vector_chunks)
        * ACTION_CHUNK,
        "primitive_transitions": int(arguments.prefill_vector_chunks)
        * ACTION_CHUNK
        * int(arguments.n_envs),
        "equivalent_chunk_transitions": int(arguments.prefill_vector_chunks)
        * int(arguments.n_envs),
        "environment_seed": 3000 + int(arguments.seed),
        "policy_seed": 4000 + int(arguments.seed),
        "residual_mode": "strict_zero",
        "base_deterministic": bool(arguments.deterministic_base),
        "action_chunk": ACTION_CHUNK,
        "action_dimension": ACTION_DIMENSION,
    }


def collect_prefill(
    *,
    environment: Any,
    model: PerStepResidualDSRL,
    metadata: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    vector_steps = int(metadata["vector_primitive_steps"])
    seed_all(int(metadata["policy_seed"]))
    environment.seed(int(metadata["environment_seed"]))
    observation = np.asarray(environment.reset(), dtype=np.float32)
    collected: dict[str, list[np.ndarray]] = {
        name: [] for name in PREFILL_ARRAY_NAMES
    }
    for _ in range(vector_steps):
        action_exec = observation[:, model.base_action_slice].copy()
        action_scaled = model.policy.scale_action(action_exec)
        new_observation, reward, done, infos = environment.step(action_exec)
        corrected_next = terminal_corrected_next_observations(
            np.asarray(new_observation),
            np.asarray(done),
            infos,
        )
        collected["observations"].append(observation.copy())
        collected["next_observations"].append(corrected_next.copy())
        collected["actions"].append(np.asarray(action_scaled).copy())
        collected["rewards"].append(np.asarray(reward).copy())
        collected["dones"].append(np.asarray(done).copy())
        collected["timeouts"].append(
            np.asarray(
                [
                    bool(info.get("TimeLimit.truncated", False))
                    for info in infos
                ],
                dtype=bool,
            )
        )
        observation = np.asarray(new_observation, dtype=np.float32)
    arrays = {
        name: np.asarray(values)
        for name, values in collected.items()
    }
    prefill_scale_error = float(
        np.max(
            np.abs(
                arrays["actions"]
                - arrays["observations"][..., model.base_action_slice]
            )
        )
    )
    if prefill_scale_error > 1e-6:
        raise RuntimeError(
            "Zero-residual prefill action differs from cached base action: "
            f"{prefill_scale_error}"
        )
    return arrays


def save_prefill(
    path: Path,
    arrays: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    hashes = hash_semantic_arrays(
        arrays,
        metadata,
        array_names=PREFILL_ARRAY_NAMES,
    )
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        **arrays,
        metadata_json=np.asarray(json.dumps(dict(metadata), sort_keys=True)),
        semantic_hash=np.asarray(hashes["semantic_hash"]),
    )
    os.replace(temporary, path)
    return hashes


def load_prefill(
    path: Path,
    expected_metadata: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"].item()))
        if metadata != dict(expected_metadata):
            raise ValueError("Existing primitive prefill metadata does not match")
        arrays = {
            name: np.asarray(archive[name]).copy()
            for name in PREFILL_ARRAY_NAMES
        }
        expected_hash = str(archive["semantic_hash"].item())
    hashes = hash_semantic_arrays(
        arrays,
        metadata,
        array_names=PREFILL_ARRAY_NAMES,
    )
    if hashes["semantic_hash"] != expected_hash:
        raise ValueError("Existing primitive prefill semantic hash mismatch")
    return arrays, hashes


def populate_replay(
    model: PerStepResidualDSRL,
    arrays: Mapping[str, np.ndarray],
) -> None:
    vector_steps = arrays["observations"].shape[0]
    for index in range(vector_steps):
        infos = [
            {
                "TimeLimit.truncated": bool(value),
                "terminal_observation": arrays["next_observations"][index, env_index],
            }
            if arrays["dones"][index, env_index]
            else {"TimeLimit.truncated": False}
            for env_index, value in enumerate(arrays["timeouts"][index])
        ]
        model.replay_buffer.add(
            arrays["observations"][index],
            arrays["next_observations"][index],
            arrays["actions"][index],
            arrays["rewards"][index],
            arrays["dones"][index],
            infos,
        )


@contextmanager
def evaluation_modes(
    model: PerStepResidualDSRL,
) -> Iterator[None]:
    modules = (model.policy, model.critic, model.critic_target, model.residual_actor)
    modes = [module.training for module in modules]
    try:
        model.set_inference_mode()
        yield
    finally:
        for module, mode in zip(modules, modes):
            module.train(mode)


def evaluate_exact(
    *,
    model: PerStepResidualDSRL,
    make_environment: Any,
    seeds: Sequence[int],
    policy_seed_start: int,
    equivalent_chunks: int,
) -> dict[str, Any]:
    episodes: list[dict[str, Any]] = []
    slot_deltas: list[list[float]] = [[] for _ in range(ACTION_CHUNK)]
    slot_clips: list[list[float]] = [[] for _ in range(ACTION_CHUNK)]
    with isolated_rng(), evaluation_modes(model):
        for episode_index, environment_seed in enumerate(seeds):
            environment = make_environment()
            try:
                policy_seed = int(policy_seed_start) + episode_index
                seed_all(policy_seed)
                observation, _ = environment.reset(seed=int(environment_seed))
                raw_return = 0.0
                length = 0
                reward_sums = {
                    "reward_forward": 0.0,
                    "reward_healthy": 0.0,
                    "reward_control": 0.0,
                }
                terminated = truncated = False
                residual_norms: list[float] = []
                clip_values: list[float] = []
                while not (terminated or truncated):
                    components, _ = model.predict_with_components(observation)
                    action_exec = np.asarray(components["action_exec"])
                    (
                        observation,
                        reward,
                        terminated,
                        truncated,
                        info,
                    ) = environment.step(action_exec)
                    raw_return += float(reward)
                    length += 1
                    for key in reward_sums:
                        reward_sums[key] += float(info[key])
                    delta = np.asarray(components["action_residual_delta"])
                    pre_clip = np.asarray(components["action_pre_clip"])
                    clip = float(np.mean(pre_clip != action_exec))
                    residual_norm = float(np.linalg.norm(delta))
                    phase = int(info["action_chunk_phase"])
                    residual_norms.append(residual_norm)
                    clip_values.append(clip)
                    slot_deltas[phase].append(residual_norm)
                    slot_clips[phase].append(clip)
                    if length > 1000:
                        raise RuntimeError("Evaluation exceeded Hopper horizon")
                early_fall = bool(terminated and length < 1000)
                episodes.append(
                    {
                        "episode_index": episode_index,
                        "environment_seed": int(environment_seed),
                        "policy_seed": policy_seed,
                        "raw_return": raw_return,
                        "d4rl_score": environment.get_normalized_score(raw_return)
                        * 100.0,
                        "primitive_length": length,
                        "early_fall": early_fall,
                        "residual_delta_l2": float(np.mean(residual_norms)),
                        "clip_fraction": float(np.mean(clip_values)),
                        **reward_sums,
                    }
                )
            finally:
                environment.close()
    returns = np.asarray([row["raw_return"] for row in episodes], dtype=np.float64)
    nonfall = np.asarray(
        [row["raw_return"] for row in episodes if not row["early_fall"]],
        dtype=np.float64,
    )
    summary = {
        "episode_count": len(episodes),
        "raw_return_mean": float(returns.mean()),
        "raw_return_std": float(returns.std()),
        "raw_return_q10": float(np.quantile(returns, 0.1)),
        "nonfall_raw_return_mean": (
            float(nonfall.mean()) if len(nonfall) else None
        ),
        "early_fall_rate": float(
            np.mean([row["early_fall"] for row in episodes])
        ),
        "primitive_length_mean": float(
            np.mean([row["primitive_length"] for row in episodes])
        ),
        "d4rl_score_mean": float(
            np.mean([row["d4rl_score"] for row in episodes])
        ),
        **{
            f"slot_{phase}_residual_delta_l2": float(np.mean(slot_deltas[phase]))
            for phase in range(ACTION_CHUNK)
        },
        **{
            f"slot_{phase}_clip_fraction": float(np.mean(slot_clips[phase]))
            for phase in range(ACTION_CHUNK)
        },
    }
    return {
        "protocol": "per_step_exact_n_v1",
        "equivalent_chunk_transitions": int(equivalent_chunks),
        "primitive_transitions": int(equivalent_chunks) * ACTION_CHUNK,
        "environment_seeds": [int(seed) for seed in seeds],
        "policy_seed_start": int(policy_seed_start),
        "summary": summary,
        "episodes": episodes,
    }


def persist_evaluation(path: Path, result: Mapping[str, Any]) -> None:
    atomic_write_json(path.with_suffix(".json"), result)
    rows = list(result["episodes"])
    temporary = path.with_suffix(".csv.tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path.with_suffix(".csv"))


def _predicted_q_advantage(
    model: PerStepResidualDSRL,
    observation: np.ndarray,
    *,
    action_base: np.ndarray,
    action_exec: np.ndarray,
) -> float:
    observation_tensor = torch.as_tensor(
        np.asarray(observation, dtype=np.float32)[None],
        device=model.device,
        dtype=torch.float32,
    )
    base_tensor = torch.as_tensor(
        np.asarray(action_base, dtype=np.float32)[None],
        device=model.device,
        dtype=torch.float32,
    )
    exec_tensor = torch.as_tensor(
        np.asarray(action_exec, dtype=np.float32)[None],
        device=model.device,
        dtype=torch.float32,
    )
    with torch.no_grad():
        q_base = model._minimum_q(
            model.critic(
                observation_tensor,
                model._scale_exec_action(base_tensor),
            )
        )
        q_exec = model._minimum_q(
            model.critic(
                observation_tensor,
                model._scale_exec_action(exec_tensor),
            )
        )
    return float((q_exec - q_base).item())


def _rank_correlation(first: np.ndarray, second: np.ndarray) -> float | None:
    if len(first) < 2 or np.all(first == first[0]) or np.all(second == second[0]):
        return None
    first_rank = np.empty(len(first), dtype=np.float64)
    first_rank[np.argsort(first, kind="mergesort")] = np.arange(len(first))
    second_rank = np.empty(len(second), dtype=np.float64)
    second_rank[np.argsort(second, kind="mergesort")] = np.arange(len(second))
    return float(np.corrcoef(first_rank, second_rank)[0, 1])


def counterfactual_critic_diagnostics(
    *,
    model: PerStepResidualDSRL,
    make_environment: Any,
    environment_seeds: Sequence[int],
    policy_seed_start: int,
    samples_per_episode: int = 5,
    rollout_horizon: int = 50,
    sample_spacing: int = 101,
) -> dict[str, Any]:
    """Compare QA preference with state/RNG-matched Monte Carlo branches.

    Each branch changes only the first primitive action.  Both branches then
    follow the same learned residual policy, matching QA's action-value
    semantics.
    """

    if sample_spacing <= 0:
        raise ValueError("sample_spacing must be positive")
    if math.gcd(sample_spacing, ACTION_CHUNK) != 1:
        raise ValueError(
            "sample_spacing must be coprime with the action chunk so the "
            "diagnostic does not alias onto one phase"
        )
    rows: list[dict[str, Any]] = []
    with isolated_rng(), evaluation_modes(model):
        for episode_index, environment_seed in enumerate(environment_seeds):
            environment = make_environment()
            try:
                policy_seed = int(policy_seed_start) + episode_index
                seed_all(policy_seed)
                observation, _ = environment.reset(seed=int(environment_seed))
                terminated = truncated = False
                primitive_step = 0
                episode_samples = 0
                while (
                    not (terminated or truncated)
                    and episode_samples < samples_per_episode
                ):
                    components, _ = model.predict_with_components(observation)
                    action_base = np.asarray(components["action_base"]).copy()
                    action_exec = np.asarray(components["action_exec"]).copy()
                    if primitive_step % sample_spacing == 0:
                        snapshot = environment.capture_state()
                        sample_phase = int(environment.phase)
                        predicted = _predicted_q_advantage(
                            model,
                            observation,
                            action_base=action_base,
                            action_exec=action_exec,
                        )

                        def branch(first_action: np.ndarray) -> tuple[float, int]:
                            environment.restore_state(snapshot)
                            branch_observation = np.asarray(observation).copy()
                            total = 0.0
                            steps = 0
                            branch_terminated = branch_truncated = False
                            action = first_action
                            for horizon_index in range(rollout_horizon):
                                (
                                    branch_observation,
                                    reward,
                                    branch_terminated,
                                    branch_truncated,
                                    _,
                                ) = environment.step(action)
                                total += (model.gamma ** horizon_index) * float(reward)
                                steps += 1
                                if branch_terminated or branch_truncated:
                                    break
                                next_components, _ = model.predict_with_components(
                                    branch_observation
                                )
                                action = np.asarray(
                                    next_components["action_exec"]
                                )
                            return total, steps

                        base_return, base_steps = branch(action_base)
                        exec_return, exec_steps = branch(action_exec)
                        actual = exec_return - base_return
                        rows.append(
                            {
                                "episode_index": episode_index,
                                "environment_seed": int(environment_seed),
                                "policy_seed": policy_seed,
                                "primitive_step": primitive_step,
                                "phase": sample_phase,
                                "predicted_q_advantage": predicted,
                                "monte_carlo_advantage": actual,
                                "base_branch_return": base_return,
                                "residual_branch_return": exec_return,
                                "base_branch_steps": base_steps,
                                "residual_branch_steps": exec_steps,
                            }
                        )
                        episode_samples += 1
                        environment.restore_state(snapshot)
                    (
                        observation,
                        _,
                        terminated,
                        truncated,
                        _,
                    ) = environment.step(action_exec)
                    primitive_step += 1
            finally:
                environment.close()
    predicted = np.asarray(
        [row["predicted_q_advantage"] for row in rows],
        dtype=np.float64,
    )
    actual = np.asarray(
        [row["monte_carlo_advantage"] for row in rows],
        dtype=np.float64,
    )
    non_ties = (predicted != 0.0) & (actual != 0.0)
    preference_accuracy = (
        float(np.mean(np.sign(predicted[non_ties]) == np.sign(actual[non_ties])))
        if np.any(non_ties)
        else None
    )
    false_positive = (predicted > 0.0) & (actual < 0.0)
    positive_predictions = predicted > 0.0
    summary = {
        "pair_count": len(rows),
        "non_tie_pair_count": int(non_ties.sum()),
        "pairwise_preference_accuracy": preference_accuracy,
        "rank_correlation": _rank_correlation(predicted, actual),
        "predicted_advantage_mean": float(predicted.mean()),
        "monte_carlo_advantage_mean": float(actual.mean()),
        "false_positive_rate_all": float(false_positive.mean()),
        "false_positive_rate_given_predicted_positive": (
            float(false_positive.sum() / positive_predictions.sum())
            if positive_predictions.any()
            else None
        ),
        "rollout_horizon": int(rollout_horizon),
        "samples_per_episode": int(samples_per_episode),
        "sample_spacing": int(sample_spacing),
        "phase_sample_counts": {
            str(phase): sum(int(row["phase"] == phase) for row in rows)
            for phase in range(ACTION_CHUNK)
        },
    }
    return {
        "protocol": "first_action_intervention_then_common_residual_policy_v1",
        "summary": summary,
        "pairs": rows,
    }


def zero_residual_parity(
    model: PerStepResidualDSRL,
    environment: Any,
    *,
    seed: int,
) -> dict[str, float]:
    with isolated_rng(seed):
        observation, _ = environment.reset(seed=seed)
        components, _ = model.predict_with_components(observation)
    return {
        "residual_unit_max_abs": float(
            np.max(np.abs(components["residual_unit"]))
        ),
        "action_exec_base_max_abs_error": float(
            np.max(
                np.abs(
                    components["action_exec"]
                    - components["action_base"]
                )
            )
        ),
    }


def save_resume_bundle(
    *,
    path: Path,
    model: PerStepResidualDSRL,
    manifest: Mapping[str, Any],
) -> None:
    if path.exists():
        existing = json.loads((path / "bundle.json").read_text())
        if int(existing["primitive_transitions"]) != int(model.num_timesteps):
            raise FileExistsError(
                f"Existing resume bundle conflicts with current counters: {path}"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{path.name}.tmp-",
            dir=path.parent,
        )
    )
    try:
        model.save(temporary / "model")
        model.save_replay_buffer(temporary / "replay.pkl")
        torch.save(capture_rng_state(), temporary / "rng.pt")
        atomic_write_json(
            temporary / "bundle.json",
            {
                "algorithm": ALGORITHM,
                "primitive_transitions": int(model.num_timesteps),
                "equivalent_chunk_transitions": int(model.num_timesteps)
                // ACTION_CHUNK,
                "manifest_binding": {
                    key: manifest[key]
                    for key in (
                        "checkpoint_sha256",
                        "ddim_sha256",
                        "normalization_sha256",
                        "prefill_hash",
                        "seed",
                        "n_envs",
                    )
                },
            },
        )
        os.replace(temporary, path)
    except BaseException:
        # Keep the temporary directory as recoverable failure evidence.
        raise


class ExperimentCallback(BaseCallback):
    def __init__(
        self,
        *,
        run_directory: Path,
        manifest_path: Path,
        manifest: dict[str, Any],
        make_eval_environment: Any,
        eval_seeds: Sequence[int],
        online_eval_episodes: int,
        online_interval_chunks: int,
        checkpoint_interval_chunks: int,
        replay_interval_chunks: int,
        stop_at_chunks: int | None,
    ) -> None:
        super().__init__(verbose=0)
        self.run_directory = run_directory
        self.manifest_path = manifest_path
        self.manifest = manifest
        self.make_eval_environment = make_eval_environment
        self.eval_seeds = list(eval_seeds)
        self.online_eval_episodes = int(online_eval_episodes)
        self.online_interval_chunks = int(online_interval_chunks)
        self.checkpoint_interval_chunks = int(checkpoint_interval_chunks)
        self.replay_interval_chunks = int(replay_interval_chunks)
        self.stop_at_chunks = stop_at_chunks
        self.next_online = self.online_interval_chunks
        self.next_checkpoint = self.checkpoint_interval_chunks
        self.next_replay = self.replay_interval_chunks
        self.reward_totals = {
            "reward_forward": 0.0,
            "reward_healthy": 0.0,
            "reward_control": 0.0,
        }
        self.primitive_info_count = 0

    def _current_chunks(self) -> int:
        primitive = int(self.model.num_timesteps)
        return primitive // ACTION_CHUNK

    def _next_boundary(self, interval: int) -> int:
        current = int(getattr(self.model, "num_timesteps", 0)) // ACTION_CHUNK
        return ((current // interval) + 1) * interval

    def _on_training_start(self) -> None:
        self.next_online = self._next_boundary(self.online_interval_chunks)
        self.next_checkpoint = self._next_boundary(
            self.checkpoint_interval_chunks
        )
        self.next_replay = self._next_boundary(self.replay_interval_chunks)

    def _update_manifest(self, **updates: Any) -> None:
        self.manifest.update(updates)
        atomic_write_json(self.manifest_path, self.manifest)

    def _on_step(self) -> bool:
        infos = self.locals["infos"]
        for info in infos:
            for key in self.reward_totals:
                self.reward_totals[key] += float(info[key])
            self.primitive_info_count += 1
        return True


def main() -> None:
    arguments = parse_args()
    if arguments.equivalent_chunk_budget <= 0 or arguments.n_envs <= 0:
        raise ValueError("Budgets and environment count must be positive")
    primitive_budget = arguments.equivalent_chunk_budget * ACTION_CHUNK
    train_frequency_primitive = 2 * ACTION_CHUNK
    alignment = arguments.n_envs * train_frequency_primitive
    if primitive_budget % alignment != 0:
        raise ValueError(
            f"Primitive budget {primitive_budget} must divide update alignment {alignment}"
        )
    if arguments.wiring_cadence:
        if arguments.equivalent_chunk_budget > 10_000:
            raise ValueError("Wiring cadence is restricted to at most 10k chunks")
        online_interval = checkpoint_interval = 2_000
        replay_interval = 5_000
    else:
        online_interval = int(arguments.online_eval_interval_chunks)
        checkpoint_interval = int(arguments.checkpoint_interval_chunks)
        replay_interval = int(arguments.replay_interval_chunks)
    run_directory = arguments.run_dir.resolve()
    manifest_path = run_directory / "run_manifest.json"
    if arguments.resume_bundle is None:
        if run_directory.exists():
            raise FileExistsError(f"Refusing to overwrite {run_directory}")
        run_directory.mkdir(parents=True)
    elif not manifest_path.is_file():
        raise FileNotFoundError("Resume run manifest is missing")

    cfg = compose_config(arguments)
    with open_dict(cfg):
        cfg.logdir = str(run_directory)
    checkpoint = arguments.checkpoint.resolve()
    expected_checkpoint_hash = str(cfg.p6.init_checkpoint_sha256)
    actual_checkpoint_hash = sha256_file(checkpoint)
    if actual_checkpoint_hash != expected_checkpoint_hash:
        raise ValueError("Explicit 5M checkpoint hash mismatch")
    artifact_hashes = {
        "checkpoint_sha256": actual_checkpoint_hash,
        "ddim_sha256": sha256_file(Path(cfg.base_policy_path)),
        "normalization_sha256": sha256_file(Path(cfg.normalization_path)),
    }
    if artifact_hashes["ddim_sha256"] != str(cfg.p6.frozen_ddim_sha256):
        raise ValueError("DDIM artifact hash mismatch")
    if artifact_hashes["normalization_sha256"] != str(
        cfg.p6.normalization_sha256
    ):
        raise ValueError("Normalization artifact hash mismatch")

    diffusion = load_base_policy(cfg)
    contract_environment = make_chunk_contract_environment(cfg)
    legacy = _load_legacy_network(
        cfg=cfg,
        environment=contract_environment,
        diffusion_policy=diffusion,
        buffer_size=1,
    )
    planner = FrozenDSRLChunkPlanner(
        legacy,
        action_chunk=ACTION_CHUNK,
        action_dimension=ACTION_DIMENSION,
    )
    alpha = float(torch.exp(legacy.log_ent_coef.detach()).item())
    initial_planner_hash = module_state_hash(planner_modules(legacy))
    train_seed = 1000 + arguments.seed
    training_environment = make_training_environment(
        cfg,
        planner,
        n_envs=arguments.n_envs,
        seed=train_seed,
        deterministic_base=arguments.deterministic_base,
    )
    make_eval_environment = lambda: make_primitive_environment(
        cfg,
        planner,
        deterministic_base=arguments.deterministic_base,
    )

    try:
        if arguments.resume_bundle is None:
            prefill_transition_count = (
                int(arguments.prefill_vector_chunks)
                * ACTION_CHUNK
                * int(arguments.n_envs)
            )
            replay_capacity = (
                prefill_transition_count + primitive_budget + arguments.n_envs
            )
            model = build_model(
                cfg,
                training_environment,
                alpha=alpha,
                buffer_size=replay_capacity,
            )
            parity_environment = make_eval_environment()
            try:
                parity = zero_residual_parity(
                    model,
                    parity_environment,
                    seed=50_000 + arguments.seed,
                )
            finally:
                parity_environment.close()
            if max(parity.values()) > 1e-6:
                raise RuntimeError(f"Zero residual parity failed: {parity}")
            metadata = prefill_metadata(arguments=arguments, cfg=cfg)
            prefill_path = (
                ROOT
                / "logs/per_step_prefill"
                / (
                    f"init5m_seed{arguments.seed}_nenv{arguments.n_envs}_"
                    f"vchunks{arguments.prefill_vector_chunks}.npz"
                )
            )
            if prefill_path.is_file():
                arrays, prefill_hashes = load_prefill(prefill_path, metadata)
                prefill_generated = False
            else:
                prefill_environment = make_training_environment(
                    cfg,
                    planner,
                    n_envs=arguments.n_envs,
                    seed=int(metadata["environment_seed"]),
                    deterministic_base=arguments.deterministic_base,
                )
                try:
                    arrays = collect_prefill(
                        environment=prefill_environment,
                        model=model,
                        metadata=metadata,
                    )
                finally:
                    prefill_environment.close()
                prefill_hashes = save_prefill(
                    prefill_path,
                    arrays,
                    metadata,
                )
                prefill_generated = True
            populate_replay(model, arrays)
            if arguments.critic_pretrain_steps <= 0:
                raise ValueError("critic-pretrain-steps must be positive")
            critic_pretrain_losses = model.pretrain_action_critic(
                gradient_steps=int(arguments.critic_pretrain_steps),
                batch_size=int(cfg.train.batch_size),
            )
            post_pretrain_environment = make_eval_environment()
            try:
                post_pretrain_parity = zero_residual_parity(
                    model,
                    post_pretrain_environment,
                    seed=50_000 + arguments.seed,
                )
            finally:
                post_pretrain_environment.close()
            if max(post_pretrain_parity.values()) > 1e-6:
                raise RuntimeError(
                    "Critic pretraining changed zero residual behavior: "
                    f"{post_pretrain_parity}"
                )
            manifest = {
                "algorithm": ALGORITHM,
                "status": "ready",
                "seed": int(arguments.seed),
                "train_env_seed": train_seed,
                "eval_seed_set": list(range(10_000, 10_100)),
                "prefill_env_seed": metadata["environment_seed"],
                "prefill_policy_seed": metadata["policy_seed"],
                "checkpoint_path": str(checkpoint),
                "ddim_path": str(Path(cfg.base_policy_path)),
                "normalization_path": str(Path(cfg.normalization_path)),
                **artifact_hashes,
                "outer_repository": git_state(ROOT),
                "sb3_repository": git_state(ROOT / "stable-baselines3"),
                "dppo_repository": git_state(ROOT / "dppo"),
                "action_chunk": ACTION_CHUNK,
                "action_dimension": ACTION_DIMENSION,
                "equivalent_chunk_budget": int(
                    arguments.equivalent_chunk_budget
                ),
                "primitive_budget": primitive_budget,
                "n_envs": int(arguments.n_envs),
                "train_frequency_primitive_vector_steps": train_frequency_primitive,
                "gamma_chunk": float(cfg.train.discount),
                "gamma_primitive": float(model.gamma),
                "updates_per_train_call": {
                    "QA": int(cfg.train.utd),
                    "residual": int(
                        cfg.train.rfs_hier_residual_actor_gradient_steps
                    ),
                },
                "noise_entropy_application": "once_when_next_phase_is_zero",
                "noise_entropy_coefficient": alpha,
                "base_deterministic": bool(arguments.deterministic_base),
                "prefill_path": str(prefill_path),
                "prefill_hash": prefill_hashes["semantic_hash"],
                "prefill_per_array_hashes": prefill_hashes["per_array_hashes"],
                "prefill_generated_by_this_run": prefill_generated,
                "prefill_transition_count": int(
                    metadata["primitive_transitions"]
                ),
                "prefill_action_base_max_abs_error": float(
                    np.max(
                        np.abs(
                            arrays["actions"]
                            - arrays["observations"][
                                ...,
                                model.base_action_slice,
                            ]
                        )
                    )
                ),
                "replay_buffer_capacity": replay_capacity,
                "zero_residual_parity": parity,
                "post_critic_pretrain_zero_residual_parity": (
                    post_pretrain_parity
                ),
                "critic_pretrain_steps": int(arguments.critic_pretrain_steps),
                "critic_pretrain_loss_initial": float(
                    critic_pretrain_losses[0]
                ),
                "critic_pretrain_loss_final": float(
                    critic_pretrain_losses[-1]
                ),
                "critic_pretrain_loss_last_100_mean": float(
                    np.mean(critic_pretrain_losses[-100:])
                ),
                "initial_planner_state_hash": initial_planner_hash,
                "action_chunk_semantics": (
                    "frozen plan sampled at phase0; current observation "
                    "read before every primitive residual correction"
                ),
                "wiring_cadence": bool(arguments.wiring_cadence),
                "checkpoint_interval_chunks": checkpoint_interval,
                "replay_interval_chunks": replay_interval,
                "online_eval_interval_chunks": online_interval,
            }
            atomic_write_json(manifest_path, manifest)
            init_result = evaluate_exact(
                model=model,
                make_environment=make_eval_environment,
                seeds=manifest["eval_seed_set"][
                    : int(arguments.online_eval_episodes)
                ],
                policy_seed_start=20_000,
                equivalent_chunks=0,
            )
            persist_evaluation(
                run_directory / "evaluations" / "initial",
                init_result,
            )
        else:
            manifest = json.loads(manifest_path.read_text())
            for key, value in artifact_hashes.items():
                if manifest[key] != value:
                    raise ValueError(f"Resume {key} mismatch")
            bundle = arguments.resume_bundle.resolve()
            bundle_manifest = json.loads(
                (bundle / "bundle.json").read_text()
            )
            if bundle_manifest["manifest_binding"] != {
                key: manifest[key]
                for key in (
                    "checkpoint_sha256",
                    "ddim_sha256",
                    "normalization_sha256",
                    "prefill_hash",
                    "seed",
                    "n_envs",
                )
            }:
                raise ValueError("Resume bundle binding mismatch")
            model = PerStepResidualDSRL.load(
                bundle / "model.zip",
                env=training_environment,
                device=cfg.device,
            )
            model.load_replay_buffer(bundle / "replay.pkl")
            restore_rng_state(torch.load(bundle / "rng.pt"))
            if int(model.num_timesteps) != int(
                bundle_manifest["primitive_transitions"]
            ):
                raise ValueError("Resume primitive counter mismatch")
            manifest["status"] = "resuming"
            manifest["resume_bundle"] = str(bundle)
            manifest["resume_environment_discontinuity"] = True
            atomic_write_json(manifest_path, manifest)

        eval_seeds = manifest["eval_seed_set"]
        callback = ExperimentCallback(
            run_directory=run_directory,
            manifest_path=manifest_path,
            manifest=manifest,
            make_eval_environment=make_eval_environment,
            eval_seeds=eval_seeds,
            online_eval_episodes=int(arguments.online_eval_episodes),
            online_interval_chunks=online_interval,
            checkpoint_interval_chunks=checkpoint_interval,
            replay_interval_chunks=replay_interval,
            stop_at_chunks=None,
        )
        current_primitive = int(model.num_timesteps)
        remaining_primitive = primitive_budget - current_primitive
        if remaining_primitive < 0:
            raise ValueError("Resume model exceeds target primitive budget")
        current_chunks = current_primitive // ACTION_CHUNK
        safe_boundary_chunks = alignment // ACTION_CHUNK
        target_chunks = int(arguments.equivalent_chunk_budget)
        if arguments.stop_at_equivalent_chunks is not None:
            stop_chunks = int(arguments.stop_at_equivalent_chunks)
            if stop_chunks <= current_chunks or stop_chunks >= target_chunks:
                raise ValueError(
                    "stop-at-equivalent-chunks must be strictly between "
                    "current and target counters"
                )
            if stop_chunks % safe_boundary_chunks != 0:
                raise ValueError(
                    "Interruption boundary must preserve a complete rollout/"
                    "update block"
                )
            segment_target_chunks = stop_chunks
        else:
            segment_target_chunks = target_chunks
        manifest["status"] = "running"
        manifest["current_primitive_transitions"] = current_primitive
        manifest["remaining_primitive_transitions"] = remaining_primitive
        atomic_write_json(manifest_path, manifest)

        def next_boundary(current: int, interval: int) -> int:
            return ((current // interval) + 1) * interval

        while current_chunks < segment_target_chunks:
            event_chunks = min(
                segment_target_chunks,
                next_boundary(current_chunks, online_interval),
                next_boundary(current_chunks, checkpoint_interval),
                next_boundary(current_chunks, replay_interval),
            )
            event_primitive = event_chunks * ACTION_CHUNK
            segment_primitive = event_primitive - int(model.num_timesteps)
            if segment_primitive <= 0 or segment_primitive % alignment != 0:
                raise RuntimeError("Training segment is not update-boundary aligned")
            model.learn(
                total_timesteps=segment_primitive,
                callback=callback,
                reset_num_timesteps=False,
                tb_log_name="per_step_residual",
            )
            if int(model.num_timesteps) != event_primitive:
                raise RuntimeError("Segmented training counter mismatch")
            current_chunks = event_chunks
            if current_chunks % online_interval == 0:
                online_result = evaluate_exact(
                    model=model,
                    make_environment=make_eval_environment,
                    seeds=eval_seeds[: int(arguments.online_eval_episodes)],
                    policy_seed_start=20_000,
                    equivalent_chunks=current_chunks,
                )
                persist_evaluation(
                    run_directory
                    / "evaluations"
                    / f"online_{current_chunks:012d}",
                    online_result,
                )
            if current_chunks % checkpoint_interval == 0:
                destination = (
                    run_directory
                    / "checkpoints"
                    / f"model_{current_chunks:012d}"
                )
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.with_suffix(".zip").exists():
                    raise FileExistsError(
                        f"Refusing to overwrite checkpoint {destination}.zip"
                    )
                model.save(destination)
            if current_chunks % replay_interval == 0:
                save_resume_bundle(
                    path=(
                        run_directory
                        / "resume"
                        / f"chunk_{current_chunks:012d}"
                    ),
                    model=model,
                    manifest=manifest,
                )
                manifest["latest_resume_chunks"] = current_chunks
            manifest.update(
                {
                    "current_primitive_transitions": int(model.num_timesteps),
                    "current_equivalent_chunk_transitions": current_chunks,
                    "remaining_primitive_transitions": (
                        primitive_budget - int(model.num_timesteps)
                    ),
                    "action_critic_optimizer_steps": int(
                        model.action_critic_optimizer_steps
                    ),
                    "critic_pretrain_optimizer_steps": int(
                        model.critic_pretrain_optimizer_steps
                    ),
                    "residual_actor_optimizer_steps": int(
                        model.residual_actor_optimizer_steps
                    ),
                    "train_calls": int(model.per_step_train_calls),
                }
            )
            atomic_write_json(manifest_path, manifest)

        if segment_target_chunks < target_chunks:
            save_resume_bundle(
                path=(
                    run_directory
                    / "resume"
                    / f"chunk_{segment_target_chunks:012d}"
                ),
                model=model,
                manifest=manifest,
            )
            manifest.update(
                {
                    "status": "interrupted",
                    "interrupted_at_equivalent_chunks": segment_target_chunks,
                    "latest_resume_chunks": segment_target_chunks,
                    "current_primitive_transitions": int(model.num_timesteps),
                    "current_equivalent_chunk_transitions": (
                        segment_target_chunks
                    ),
                    "remaining_primitive_transitions": (
                        primitive_budget - int(model.num_timesteps)
                    ),
                }
            )
            atomic_write_json(manifest_path, manifest)
            return
        if int(model.num_timesteps) != primitive_budget:
            raise RuntimeError(
                f"Training ended at {model.num_timesteps}, expected {primitive_budget}"
            )
        final_planner_hash = module_state_hash(planner_modules(legacy))
        if final_planner_hash != initial_planner_hash:
            raise RuntimeError("Frozen DSRL/DDIM planner parameters changed")
        final_result = evaluate_exact(
            model=model,
            make_environment=make_eval_environment,
            seeds=eval_seeds[: int(arguments.final_eval_episodes)],
            policy_seed_start=20_000,
            equivalent_chunks=arguments.equivalent_chunk_budget,
        )
        persist_evaluation(
            run_directory / "evaluations" / "final",
            final_result,
        )
        counterfactual_result = counterfactual_critic_diagnostics(
            model=model,
            make_environment=make_eval_environment,
            environment_seeds=eval_seeds[:3],
            policy_seed_start=30_000,
        )
        atomic_write_json(
            run_directory / "evaluations" / "counterfactual_critic.json",
            counterfactual_result,
        )
        (run_directory / "checkpoints").mkdir(parents=True, exist_ok=True)
        model.save(run_directory / "checkpoints" / "final_model")
        save_resume_bundle(
            path=(
                run_directory
                / "resume"
                / f"chunk_{arguments.equivalent_chunk_budget:012d}"
            ),
            model=model,
            manifest=manifest,
        )
        manifest.update(
            {
                "status": "complete",
                "current_primitive_transitions": int(model.num_timesteps),
                "current_equivalent_chunk_transitions": int(model.num_timesteps)
                // ACTION_CHUNK,
                "action_critic_optimizer_steps": int(
                    model.action_critic_optimizer_steps
                ),
                "residual_actor_optimizer_steps": int(
                    model.residual_actor_optimizer_steps
                ),
                "train_calls": int(model.per_step_train_calls),
                "final_planner_state_hash": final_planner_hash,
                "planner_state_unchanged": True,
                "final_evaluation_summary": final_result["summary"],
                "counterfactual_critic_summary": counterfactual_result[
                    "summary"
                ],
            }
        )
        atomic_write_json(manifest_path, manifest)
        (run_directory / "COMPLETE").write_text("complete\n")
    finally:
        training_environment.close()
        contract_environment.close()


if __name__ == "__main__":
    main()
