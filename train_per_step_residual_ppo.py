"""Frozen DSRL + per-primitive residual PPO experiment runner."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / "dppo"), str(ROOT / "stable-baselines3")]

from p6_preflight import sha256_file  # noqa: E402
from p6_runtime import (  # noqa: E402
    atomic_write_json,
    capture_rng_state,
    isolated_rng,
    restore_rng_state,
    seed_all,
)
from p6_train import _load_legacy_network  # noqa: E402
from per_step_residual_env import FrozenDSRLChunkPlanner  # noqa: E402
from per_step_residual_env import FrozenDiffusionChunkPlanner  # noqa: E402
from per_step_residual_ppo import (  # noqa: E402
    CountingPPO,
    PPOResidualUnitEnv,
    PPOTrainingRewardScaleEnv,
    ResiPRewardNormalize,
    ResidualObservationExtractor,
    ResiPAlignedPPO,
    SeparatedClipPPO,
    zero_initialize_ppo_residual_mean,
)
from stable_baselines3.common.callbacks import BaseCallback  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv  # noqa: E402
from stable_baselines3.dsrl.hierarchical_rfs_dsrl import (  # noqa: E402
    _LegacyLoadableDSRL,
)
from train_per_step_residual import (  # noqa: E402
    ACTION_CHUNK,
    ACTION_DIMENSION,
    compose_config,
    git_state,
    make_chunk_contract_environment,
    make_primitive_environment,
    module_state_hash,
    planner_modules,
)
from utils import load_base_policy  # noqa: E402


ALGORITHM = "dsrl_na_residual_per_step_ppo"
STABLE_ALGORITHM = "dsrl_na_residual_per_step_ppo_stable_v1"
RESIP_ALGORITHM = "dsrl_na_residual_per_step_ppo_resip_v1"
RESIP_HOPPER_ALGORITHM = "dsrl_na_residual_per_step_ppo_resip_hopper_v1"
RESIP_HOPPER_LONG_GAE_ALGORITHM = (
    "dsrl_na_residual_per_step_ppo_resip_hopper_long_gae_v1"
)
RESIP_TRAINING_VARIANTS = (
    "resip_v1",
    "resip_hopper_v1",
    "resip_hopper_long_gae_v1",
)
HOPPER_RESIP_VARIANTS = (
    "resip_hopper_v1",
    "resip_hopper_long_gae_v1",
)
LONG_GAE_LAMBDA = 0.985
DEFAULT_RESIDUAL_SCALE = 0.1
DEFAULT_RESIDUAL_STD = 0.05
DEFAULT_ROLLOUT_STEPS = 400
DEFAULT_BATCH_SIZE = 400


def resolve_training_reward_scale(
    training_variant: str,
    requested_scale: float | None,
) -> float:
    """Resolve reward scale without silently changing a named profile."""

    if training_variant in HOPPER_RESIP_VARIANTS:
        if requested_scale is not None and not math.isclose(
            float(requested_scale),
            0.01,
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise ValueError("Hopper ResiP profile fixes training reward scale at 0.01")
        return 0.01
    if requested_scale is not None:
        return float(requested_scale)
    if training_variant == "stable_v1":
        return 0.01
    return 1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--base-source",
        choices=("dsrl_checkpoint", "diffusion_prior"),
        default="dsrl_checkpoint",
    )
    parser.add_argument("--expected-checkpoint-sha256")
    parser.add_argument("--expected-checkpoint-steps", type=int)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--equivalent-chunk-budget", type=int, required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--n-envs", type=int, default=10)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--final-eval-episodes", type=int, default=100)
    parser.add_argument("--online-eval-episodes", type=int, default=10)
    parser.add_argument("--rollout-steps", type=int, default=DEFAULT_ROLLOUT_STEPS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--n-epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--residual-std", type=float, default=DEFAULT_RESIDUAL_STD)
    parser.add_argument("--residual-scale", type=float, default=DEFAULT_RESIDUAL_SCALE)
    parser.add_argument("--clip-range", type=float, default=0.1)
    parser.add_argument("--target-kl", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--online-eval-interval-chunks", type=int, default=10_000)
    parser.add_argument("--checkpoint-interval-chunks", type=int, default=100_000)
    parser.add_argument("--stop-at-equivalent-chunks", type=int)
    parser.add_argument("--resume-bundle", type=Path)
    parser.add_argument("--wiring-cadence", action="store_true")
    parser.add_argument("--deterministic-base", action="store_true")
    parser.add_argument(
        "--training-variant",
        choices=("legacy", "stable_v1", *RESIP_TRAINING_VARIANTS),
        default="legacy",
    )
    parser.add_argument("--training-reward-scale", type=float)
    parser.add_argument("--actor-max-grad-norm", type=float, default=0.5)
    parser.add_argument("--value-max-grad-norm", type=float, default=0.5)
    return parser.parse_args()


def make_ppo_environment(
    cfg: Any,
    planner: Any,
    *,
    residual_scale: float,
    deterministic_base: bool,
) -> PPOResidualUnitEnv:
    return PPOResidualUnitEnv(
        make_primitive_environment(
            cfg,
            planner,
            deterministic_base=deterministic_base,
        ),
        residual_scale=residual_scale,
    )


def make_training_environment(
    cfg: Any,
    planner: Any,
    *,
    n_envs: int,
    seed: int,
    residual_scale: float,
    deterministic_base: bool,
    training_reward_scale: float = 1.0,
):
    if not np.isfinite(training_reward_scale) or training_reward_scale <= 0:
        raise ValueError("training_reward_scale must be finite and positive")

    def make_one(rank: int):
        environment_planner = planner.fork(seed + 100_000 + rank)
        environment = make_ppo_environment(
            cfg,
            environment_planner,
            residual_scale=residual_scale,
            deterministic_base=deterministic_base,
        )
        if training_reward_scale != 1.0:
            return PPOTrainingRewardScaleEnv(
                environment,
                reward_scale=training_reward_scale,
            )
        return environment

    vector_environment = DummyVecEnv(
        [lambda rank=rank: make_one(rank) for rank in range(n_envs)]
    )
    vector_environment.seed(seed)
    return vector_environment


def _first_primitive_wrapper(environment: Any) -> PPOResidualUnitEnv:
    vector = environment
    visited: set[int] = set()
    while not hasattr(vector, "envs"):
        if id(vector) in visited or not hasattr(vector, "venv"):
            raise TypeError("Could not locate PPO vector environment")
        visited.add(id(vector))
        vector = vector.venv
    wrapped = vector.envs[0]
    visited.clear()
    while not isinstance(wrapped, PPOResidualUnitEnv):
        if id(wrapped) in visited or not hasattr(wrapped, "env"):
            raise TypeError("Could not locate PPO residual environment")
        visited.add(id(wrapped))
        wrapped = wrapped.env
    return wrapped


def build_model(
    cfg: Any,
    environment: Any,
    *,
    arguments: argparse.Namespace,
) -> CountingPPO:
    resip_training = arguments.training_variant in RESIP_TRAINING_VARIANTS
    long_gae_training = arguments.training_variant == "resip_hopper_long_gae_v1"
    primitive_gamma = (
        0.999
        if resip_training
        else float(cfg.train.discount) ** (1.0 / ACTION_CHUNK)
    )
    primitive_gae_lambda = (
        LONG_GAE_LAMBDA
        if long_gae_training
        else 0.95
        if resip_training
        else 0.95 ** (1.0 / ACTION_CHUNK)
    )
    wrapped = _first_primitive_wrapper(environment)
    model_class = (
        ResiPAlignedPPO
        if resip_training
        else SeparatedClipPPO
        if arguments.training_variant == "stable_v1"
        else CountingPPO
    )
    separate_arguments = (
        {
            "actor_max_grad_norm": float(arguments.actor_max_grad_norm),
            "value_max_grad_norm": float(arguments.value_max_grad_norm),
        }
        if model_class in (SeparatedClipPPO, ResiPAlignedPPO)
        else {}
    )
    if resip_training:
        separate_arguments.update(
            {
                "actor_learning_rate": 3e-4,
                "value_learning_rate": 5e-3,
                "fixed_log_std": True,
                "schedule_total_iterations": (
                    int(arguments.equivalent_chunk_budget)
                    * ACTION_CHUNK
                    // (int(arguments.rollout_steps) * int(arguments.n_envs))
                ),
                "actor_warmup_iterations": 5,
                "value_loss_multiplier": 0.5,
                "target_kl_multiplier": 1.0,
            }
        )
    model = model_class(
        "MlpPolicy",
        environment,
        learning_rate=float(arguments.learning_rate),
        n_steps=int(arguments.rollout_steps),
        batch_size=int(arguments.batch_size),
        n_epochs=int(arguments.n_epochs),
        gamma=primitive_gamma,
        gae_lambda=primitive_gae_lambda,
        clip_range=float(arguments.clip_range),
        clip_range_vf=None,
        normalize_advantage=True,
        ent_coef=0.0,
        vf_coef=1.0 if resip_training else 0.5,
        max_grad_norm=float(arguments.max_grad_norm),
        use_sde=False,
        target_kl=float(arguments.target_kl),
        tensorboard_log=str(Path(cfg.logdir) / "tensorboard"),
        policy_kwargs={
            "activation_fn": nn.ReLU if resip_training else nn.SiLU,
            "net_arch": {
                "pi": [256, 256] if resip_training else [128, 128],
                "vf": [256, 256] if resip_training else [128, 128],
            },
            "ortho_init": True,
            "log_std_init": math.log(float(arguments.residual_std)),
            "features_extractor_class": ResidualObservationExtractor,
            "features_extractor_kwargs": {
                "noise_log_prob_index": (
                    wrapped.env.noise_log_prob_index
                ),
                "clamp_value": 3.0 if resip_training else None,
            },
        },
        verbose=1,
        seed=int(arguments.seed),
        device=arguments.device,
        **separate_arguments,
    )
    zero_initialize_ppo_residual_mean(model.policy)
    return model


class ResidualMetricsCallback(BaseCallback):
    def __init__(self) -> None:
        super().__init__(verbose=0)
        self.values: dict[str, list[float]] = {
            "residual_unit_l2": [],
            "action_residual_delta_l2": [],
            "effective_residual_l2": [],
            "residual_action_clip_fraction": [],
            "execution_clip_fraction": [],
            "reward_forward": [],
            "reward_healthy": [],
            "reward_control": [],
        }

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            for key in self.values:
                self.values[key].append(float(info[key]))
        return True

    def means(self) -> dict[str, float]:
        return {
            key: float(np.mean(values)) if values else 0.0
            for key, values in self.values.items()
        }


def evaluate_exact(
    *,
    model: CountingPPO,
    make_environment: Any,
    seeds: Sequence[int],
    policy_seed_start: int,
    equivalent_chunks: int,
    deterministic: bool,
) -> dict[str, Any]:
    episodes: list[dict[str, Any]] = []
    with isolated_rng():
        model.policy.set_training_mode(False)
        for episode_index, environment_seed in enumerate(seeds):
            environment = make_environment()
            try:
                policy_seed = int(policy_seed_start) + episode_index
                seed_all(policy_seed)
                environment.env.planner.seed(policy_seed)
                observation, _ = environment.reset(seed=int(environment_seed))
                terminated = truncated = False
                total_reward = 0.0
                primitive_length = 0
                reward_components = {
                    "reward_forward": 0.0,
                    "reward_healthy": 0.0,
                    "reward_control": 0.0,
                }
                delta_l2: list[float] = []
                action_clips: list[float] = []
                execution_clips: list[float] = []
                while not (terminated or truncated):
                    residual_unit, _ = model.predict(
                        observation,
                        deterministic=deterministic,
                    )
                    (
                        observation,
                        reward,
                        terminated,
                        truncated,
                        info,
                    ) = environment.step(residual_unit)
                    total_reward += float(reward)
                    primitive_length += 1
                    for key in reward_components:
                        reward_components[key] += float(info[key])
                    delta_l2.append(float(info["action_residual_delta_l2"]))
                    action_clips.append(
                        float(info["residual_action_clip_fraction"])
                    )
                    execution_clips.append(
                        float(info["execution_clip_fraction"])
                    )
                episodes.append(
                    {
                        "episode_index": episode_index,
                        "environment_seed": int(environment_seed),
                        "policy_seed": policy_seed,
                        "raw_return": total_reward,
                        "primitive_length": primitive_length,
                        "early_fall": primitive_length < int(
                            environment.env.max_episode_steps
                        ),
                        "residual_delta_l2": float(np.mean(delta_l2)),
                        "residual_action_clip_fraction": float(
                            np.mean(action_clips)
                        ),
                        "execution_clip_fraction": float(
                            np.mean(execution_clips)
                        ),
                        **reward_components,
                    }
                )
            finally:
                environment.close()
    returns = np.asarray([row["raw_return"] for row in episodes])
    early = np.asarray([row["early_fall"] for row in episodes], dtype=bool)
    summary = {
        "episode_count": len(episodes),
        "raw_return_mean": float(returns.mean()),
        "raw_return_std": float(returns.std()),
        "raw_return_q10": float(np.quantile(returns, 0.1)),
        "nonfall_raw_return_mean": (
            float(returns[~early].mean()) if np.any(~early) else None
        ),
        "early_fall_rate": float(early.mean()),
        "primitive_length_mean": float(
            np.mean([row["primitive_length"] for row in episodes])
        ),
        "residual_delta_l2": float(
            np.mean([row["residual_delta_l2"] for row in episodes])
        ),
        "residual_action_clip_fraction": float(
            np.mean(
                [row["residual_action_clip_fraction"] for row in episodes]
            )
        ),
        "execution_clip_fraction": float(
            np.mean([row["execution_clip_fraction"] for row in episodes])
        ),
        "reward_forward_mean": float(
            np.mean([row["reward_forward"] for row in episodes])
        ),
        "reward_healthy_mean": float(
            np.mean([row["reward_healthy"] for row in episodes])
        ),
        "reward_control_mean": float(
            np.mean([row["reward_control"] for row in episodes])
        ),
    }
    return {
        "protocol": (
            "exact sequential episodes; independent environment and policy "
            "seeds; deterministic residual mean"
            if deterministic
            else "exact sequential episodes; independent environment and policy seeds; stochastic residual"
        ),
        "deterministic": bool(deterministic),
        "equivalent_chunk_transitions": int(equivalent_chunks),
        "primitive_transitions": int(equivalent_chunks) * ACTION_CHUNK,
        "environment_seeds": list(map(int, seeds)),
        "policy_seed_start": int(policy_seed_start),
        "summary": summary,
        "episodes": episodes,
    }


def write_evaluation(path: Path, result: dict[str, Any]) -> None:
    atomic_write_json(path.with_suffix(".json"), result)
    rows = result["episodes"]
    temporary = path.with_suffix(".tmp.csv")
    with temporary.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path.with_suffix(".csv"))


def save_bundle(
    path: Path,
    *,
    model: CountingPPO,
    environment: Any,
    manifest: dict[str, Any],
    equivalent_chunks: int,
) -> None:
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{path.name}.", dir=str(path.parent))
    )
    try:
        model.save(temporary / "model")
        if isinstance(model, ResiPAlignedPPO):
            optimizer_payload = {
                "format": "resip_independent_adamw_v1",
                "actor_optimizer": model.actor_optimizer.state_dict(),
                "value_optimizer": model.value_optimizer.state_dict(),
                "actor_optimizer_steps": int(model.actor_optimizer_steps),
                "value_optimizer_steps": int(model.value_optimizer_steps),
            }
        else:
            optimizer_payload = {
                "format": "sb3_policy_optimizer_v1",
                "policy_optimizer": model.policy.optimizer.state_dict(),
            }
        torch.save(optimizer_payload, temporary / "optimizer.pt")
        if isinstance(environment, ResiPRewardNormalize):
            torch.save(
                environment.state_dict(),
                temporary / "resip_reward_normalizer.pt",
            )
        torch.save(capture_rng_state(), temporary / "rng.pt")
        atomic_write_json(
            temporary / "bundle.json",
            {
                "equivalent_chunk_transitions": int(equivalent_chunks),
                "primitive_transitions": int(model.num_timesteps),
                "ppo_optimizer_steps": int(model.ppo_optimizer_steps),
                "ppo_epochs_completed": int(model._n_updates),
                "manifest_algorithm": manifest["algorithm"],
                "optimizer_state_saved": True,
                "reward_normalizer_state_saved": isinstance(
                    environment,
                    ResiPRewardNormalize,
                ),
            },
        )
        os.replace(temporary, path)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> None:
    arguments = parse_args()
    stable_training = arguments.training_variant == "stable_v1"
    resip_training = arguments.training_variant in RESIP_TRAINING_VARIANTS
    long_gae_training = arguments.training_variant == "resip_hopper_long_gae_v1"
    resip_reward_normalization = arguments.training_variant == "resip_v1"
    if resip_training:
        # Fixed ResiP-aligned profile.  These assignments deliberately happen
        # before validation and manifest creation so the recorded experiment
        # cannot silently inherit the earlier Hopper PPO defaults.
        arguments.learning_rate = 3e-4
        arguments.n_epochs = 50
        arguments.residual_std = 0.05 if long_gae_training else math.exp(-1.0)
        arguments.clip_range = 0.2
        arguments.target_kl = 0.1
        arguments.max_grad_norm = 1.0
        arguments.actor_max_grad_norm = 1.0
        arguments.value_max_grad_norm = 1.0
        arguments.batch_size = arguments.rollout_steps * arguments.n_envs
    training_reward_scale = resolve_training_reward_scale(
        arguments.training_variant,
        arguments.training_reward_scale,
    )
    algorithm = (
        RESIP_HOPPER_LONG_GAE_ALGORITHM
        if long_gae_training
        else RESIP_HOPPER_ALGORITHM
        if arguments.training_variant == "resip_hopper_v1"
        else RESIP_ALGORITHM
        if resip_training
        else STABLE_ALGORITHM
        if stable_training
        else ALGORITHM
    )
    if not np.isfinite(training_reward_scale) or training_reward_scale <= 0:
        raise ValueError("training-reward-scale must be finite and positive")
    if not stable_training and not resip_training and training_reward_scale != 1.0:
        raise ValueError("Legacy PPO must preserve unscaled training rewards")
    if arguments.equivalent_chunk_budget <= 0 or arguments.n_envs <= 0:
        raise ValueError("Budgets and n_envs must be positive")
    if not (0 < arguments.residual_std < 1):
        raise ValueError("residual-std must be in (0,1)")
    if arguments.residual_scale != DEFAULT_RESIDUAL_SCALE:
        raise ValueError("C fixes maximum residual_scale at 0.1")
    rollout_primitives = arguments.rollout_steps * arguments.n_envs
    if rollout_primitives % ACTION_CHUNK != 0:
        raise ValueError("Rollout batch must map to integral equivalent chunks")
    rollout_chunks = rollout_primitives // ACTION_CHUNK
    if arguments.equivalent_chunk_budget % rollout_chunks != 0:
        raise ValueError("Budget must be divisible by one PPO rollout")
    if arguments.batch_size > rollout_primitives or (
        rollout_primitives % arguments.batch_size != 0
    ):
        raise ValueError("batch-size must exactly divide the PPO rollout buffer")
    if arguments.wiring_cadence:
        if arguments.equivalent_chunk_budget > 10_000:
            raise ValueError("Wiring cadence is restricted to <=10k")
        eval_interval = checkpoint_interval = 2_000
    else:
        eval_interval = int(arguments.online_eval_interval_chunks)
        checkpoint_interval = int(arguments.checkpoint_interval_chunks)
    for name, interval in (
        ("evaluation", eval_interval),
        ("checkpoint", checkpoint_interval),
    ):
        if interval % rollout_chunks != 0:
            raise ValueError(f"{name} interval must align to PPO rollouts")

    run_directory = arguments.run_dir.resolve()
    manifest_path = run_directory / "run_manifest.json"
    if arguments.resume_bundle is None:
        if run_directory.exists():
            raise FileExistsError(f"Refusing to overwrite {run_directory}")
        run_directory.mkdir(parents=True)
    elif not manifest_path.is_file():
        raise FileNotFoundError("Resume manifest is missing")
    for directory in ("evaluations", "checkpoints", "resume"):
        (run_directory / directory).mkdir(exist_ok=True)

    cfg = compose_config(arguments)
    cfg.logdir = str(run_directory)
    if arguments.base_source == "dsrl_checkpoint":
        if arguments.checkpoint is None:
            raise ValueError("dsrl_checkpoint base requires --checkpoint")
        if arguments.expected_checkpoint_sha256 is None:
            raise ValueError(
                "dsrl_checkpoint base requires --expected-checkpoint-sha256"
            )
        if arguments.expected_checkpoint_steps is None:
            raise ValueError(
                "dsrl_checkpoint base requires --expected-checkpoint-steps"
            )
        checkpoint = arguments.checkpoint.resolve()
        checkpoint_hash = sha256_file(checkpoint)
        if checkpoint_hash != arguments.expected_checkpoint_sha256:
            raise ValueError("Explicit DSRL checkpoint hash mismatch")
        cfg.p6.init_checkpoint_sha256 = checkpoint_hash
        cfg.p6.expected_init_checkpoint_steps = int(
            arguments.expected_checkpoint_steps
        )
        checkpoint_steps: int | None = int(arguments.expected_checkpoint_steps)
    else:
        if arguments.checkpoint is not None:
            raise ValueError("diffusion_prior base must not receive --checkpoint")
        if arguments.expected_checkpoint_sha256 is not None:
            raise ValueError(
                "diffusion_prior base must not receive a checkpoint SHA"
            )
        if arguments.expected_checkpoint_steps is not None:
            raise ValueError(
                "diffusion_prior base must not receive checkpoint steps"
            )
        checkpoint = None
        checkpoint_hash = None
        checkpoint_steps = None
    ddim_hash = sha256_file(Path(cfg.base_policy_path))
    normalization_hash = sha256_file(Path(cfg.normalization_path))
    if ddim_hash != str(cfg.p6.frozen_ddim_sha256):
        raise ValueError("DDIM artifact hash mismatch")
    if normalization_hash != str(cfg.p6.normalization_sha256):
        raise ValueError("Normalization artifact hash mismatch")

    diffusion = load_base_policy(cfg)
    contract = make_chunk_contract_environment(cfg)
    legacy = None
    if arguments.base_source == "dsrl_checkpoint":
        legacy = _load_legacy_network(
            cfg=cfg,
            environment=contract,
            diffusion_policy=diffusion,
            buffer_size=1,
        )
        planner: Any = FrozenDSRLChunkPlanner(
            legacy,
            action_chunk=ACTION_CHUNK,
            action_dimension=ACTION_DIMENSION,
            policy_seed=4000 + int(arguments.seed),
        )
        planner_state_modules = planner_modules(legacy)
    else:
        planner = FrozenDiffusionChunkPlanner(
            diffusion,
            device=cfg.device,
            observation_dimension=int(np.prod(contract.observation_space.shape)),
            action_chunk=ACTION_CHUNK,
            action_dimension=ACTION_DIMENSION,
            policy_seed=4000 + int(arguments.seed),
        )
        planner_state_modules = {
            name: module
            for name, module in (
                ("diffusion_policy", diffusion),
                (
                    "diffusion_base_policy",
                    getattr(diffusion, "base_policy", None),
                ),
            )
            if isinstance(module, torch.nn.Module)
        }
    planner.assert_frozen()
    initial_planner_hash = module_state_hash(planner_state_modules)
    raw_train_environment = make_training_environment(
        cfg,
        planner,
        n_envs=arguments.n_envs,
        seed=1000 + arguments.seed,
        residual_scale=arguments.residual_scale,
        deterministic_base=arguments.deterministic_base,
        training_reward_scale=training_reward_scale,
    )
    if resip_reward_normalization:
        if arguments.resume_bundle is None:
            train_environment = ResiPRewardNormalize(
                raw_train_environment,
                clip_reward=5.0,
            )
        else:
            statistics_path = (
                arguments.resume_bundle.resolve()
                / "resip_reward_normalizer.pt"
            )
            if not statistics_path.is_file():
                raise FileNotFoundError(
                    "ResiP resume is missing reward-normalizer state"
                )
            train_environment = ResiPRewardNormalize(
                raw_train_environment,
                clip_reward=5.0,
            )
            train_environment.load_state_dict(
                torch.load(statistics_path, map_location="cpu")
            )
    else:
        train_environment = raw_train_environment
    make_eval_environment = lambda: make_ppo_environment(
        cfg,
        planner,
        residual_scale=arguments.residual_scale,
        deterministic_base=arguments.deterministic_base,
    )

    try:
        if arguments.resume_bundle is None:
            model = build_model(cfg, train_environment, arguments=arguments)
            with torch.no_grad():
                zero_mean_error = float(
                    model.policy.action_net.weight.abs().max().item()
                    + (
                        model.policy.action_net.bias.abs().max().item()
                        if model.policy.action_net.bias is not None
                        else 0.0
                    )
                )
            if zero_mean_error != 0.0:
                raise RuntimeError("PPO deterministic residual mean is not exact zero")
            manifest = {
                "algorithm": algorithm,
                "training_variant": arguments.training_variant,
                "base_source": arguments.base_source,
                "training_reward_scale": training_reward_scale,
                "actor_max_grad_norm": (
                    float(arguments.actor_max_grad_norm)
                    if stable_training or resip_training
                    else None
                ),
                "value_max_grad_norm": (
                    float(arguments.value_max_grad_norm)
                    if stable_training or resip_training
                    else None
                ),
                "status": "ready",
                "seed": int(arguments.seed),
                "train_env_seed": 1000 + int(arguments.seed),
                "eval_seed_set": list(range(10_000, 10_100)),
                "checkpoint_path": (
                    str(checkpoint) if checkpoint is not None else None
                ),
                "checkpoint_sha256": checkpoint_hash,
                "checkpoint_steps": checkpoint_steps,
                "ddim_path": str(Path(cfg.base_policy_path)),
                "ddim_sha256": ddim_hash,
                "normalization_path": str(Path(cfg.normalization_path)),
                "normalization_sha256": normalization_hash,
                "outer_repository": git_state(ROOT),
                "sb3_repository": git_state(ROOT / "stable-baselines3"),
                "dppo_repository": git_state(ROOT / "dppo"),
                "action_chunk": ACTION_CHUNK,
                "action_dimension": ACTION_DIMENSION,
                "equivalent_chunk_budget": int(
                    arguments.equivalent_chunk_budget
                ),
                "primitive_budget": int(
                    arguments.equivalent_chunk_budget * ACTION_CHUNK
                ),
                "n_envs": int(arguments.n_envs),
                "rollout_steps_per_env": int(arguments.rollout_steps),
                "rollout_equivalent_chunks": int(rollout_chunks),
                "batch_size": int(arguments.batch_size),
                "n_epochs": int(arguments.n_epochs),
                "learning_rate": float(arguments.learning_rate),
                "residual_scale": float(arguments.residual_scale),
                "initial_residual_std": float(arguments.residual_std),
                "residual_log_std_trainable": not resip_training,
                "actor_optimizer": "AdamW" if resip_training else "Adam",
                "value_optimizer": "AdamW" if resip_training else "Adam",
                "actor_learning_rate": 3e-4 if resip_training else None,
                "value_learning_rate": 5e-3 if resip_training else None,
                "reward_normalization": bool(resip_reward_normalization),
                "reward_normalization_semantics": (
                    "immediate_reward_running_variance_no_mean_subtraction"
                    if resip_reward_normalization
                    else "fixed_multiplicative_scale"
                    if arguments.training_variant in HOPPER_RESIP_VARIANTS
                    else None
                ),
                "reward_clip": 5.0 if resip_reward_normalization else None,
                "learning_rate_schedule": (
                    "cosine_actor_warmup5_critic_warmup0"
                    if resip_training
                    else None
                ),
                "resip_schedule_total_iterations": (
                    int(model.schedule_total_iterations)
                    if resip_training
                    else None
                ),
                "actor_activation": "ReLU" if resip_training else "SiLU",
                "critic_activation": "ReLU" if resip_training else "SiLU",
                "critic_output_gain": 0.25 if resip_training else None,
                "critic_output_bias": 0.25 if resip_training else None,
                "observation_clamp": 3.0 if resip_training else None,
                "value_loss_multiplier": 0.5 if resip_training else 1.0,
                "target_kl_multiplier": 1.0 if resip_training else 1.5,
                "zero_mean_initialization_error": zero_mean_error,
                "gamma_chunk": float(cfg.train.discount),
                "gamma_primitive": float(model.gamma),
                "gae_lambda_chunk": 0.95,
                "gae_lambda_primitive": float(model.gae_lambda),
                "gae_effective_horizon_approx": float(
                    1.0 / (1.0 - float(model.gamma) * float(model.gae_lambda))
                ),
                "clip_range": float(arguments.clip_range),
                "target_kl": float(arguments.target_kl),
                "max_grad_norm": float(arguments.max_grad_norm),
                "deterministic_base": bool(arguments.deterministic_base),
                "prefill_source": None,
                "on_policy": True,
                "initial_planner_state_hash": initial_planner_hash,
                "online_eval_interval_chunks": eval_interval,
                "checkpoint_interval_chunks": checkpoint_interval,
                "wiring_cadence": bool(arguments.wiring_cadence),
            }
            deterministic_initial = evaluate_exact(
                model=model,
                make_environment=make_eval_environment,
                seeds=list(range(10_000, 10_000 + arguments.online_eval_episodes)),
                policy_seed_start=20_000,
                equivalent_chunks=0,
                deterministic=True,
            )
            stochastic_initial = evaluate_exact(
                model=model,
                make_environment=make_eval_environment,
                seeds=list(range(10_000, 10_000 + arguments.online_eval_episodes)),
                policy_seed_start=20_000,
                equivalent_chunks=0,
                deterministic=False,
            )
            write_evaluation(
                run_directory / "evaluations/initial_deterministic",
                deterministic_initial,
            )
            write_evaluation(
                run_directory / "evaluations/initial_stochastic",
                stochastic_initial,
            )
            manifest["initial_deterministic_summary"] = deterministic_initial[
                "summary"
            ]
            manifest["initial_stochastic_summary"] = stochastic_initial["summary"]
            atomic_write_json(manifest_path, manifest)
        else:
            manifest = json.loads(manifest_path.read_text())
            if manifest["algorithm"] != algorithm:
                raise ValueError("Resume training variant conflicts with manifest")
            if float(manifest.get("training_reward_scale", 1.0)) != (
                training_reward_scale
            ):
                raise ValueError("Resume reward scale conflicts with manifest")
            bundle = arguments.resume_bundle.resolve()
            bundle_data = json.loads((bundle / "bundle.json").read_text())
            model_class = (
                ResiPAlignedPPO
                if resip_training
                else SeparatedClipPPO
                if stable_training
                else CountingPPO
            )
            model = model_class.load(
                bundle / "model.zip",
                env=train_environment,
                device=arguments.device,
            )
            model.ppo_optimizer_steps = int(
                bundle_data["ppo_optimizer_steps"]
            )
            optimizer_payload = torch.load(
                bundle / "optimizer.pt",
                map_location=arguments.device,
            )
            if resip_training:
                if optimizer_payload.get("format") != "resip_independent_adamw_v1":
                    raise ValueError("Unexpected ResiP optimizer bundle format")
                model.actor_optimizer.load_state_dict(
                    optimizer_payload["actor_optimizer"]
                )
                model.value_optimizer.load_state_dict(
                    optimizer_payload["value_optimizer"]
                )
                model.actor_optimizer_steps = int(
                    optimizer_payload["actor_optimizer_steps"]
                )
                model.value_optimizer_steps = int(
                    optimizer_payload["value_optimizer_steps"]
                )
            else:
                if optimizer_payload.get("format") != "sb3_policy_optimizer_v1":
                    raise ValueError("Unexpected PPO optimizer bundle format")
                model.policy.optimizer.load_state_dict(
                    optimizer_payload["policy_optimizer"]
                )
            restore_rng_state(
                torch.load(bundle / "rng.pt", map_location="cpu")
            )
            manifest.update(
                {
                    "status": "resumed",
                    "resume_bundle": str(bundle),
                    "resume_environment_discontinuity": True,
                }
            )
            atomic_write_json(manifest_path, manifest)

        current_chunks = int(model.num_timesteps) // ACTION_CHUNK
        stop_chunks = arguments.stop_at_equivalent_chunks
        target_chunks = int(arguments.equivalent_chunk_budget)
        segment_target = stop_chunks if stop_chunks is not None else target_chunks
        if segment_target > target_chunks or segment_target <= current_chunks:
            raise ValueError("Invalid stop/target boundary")
        if segment_target % rollout_chunks != 0:
            raise ValueError("Stop boundary must align to PPO rollouts")

        callback = ResidualMetricsCallback()
        while current_chunks < segment_target:
            next_event = min(
                segment_target,
                ((current_chunks // eval_interval) + 1) * eval_interval,
                ((current_chunks // checkpoint_interval) + 1)
                * checkpoint_interval,
            )
            if next_event <= current_chunks:
                next_event = current_chunks + rollout_chunks
            primitive_segment = (next_event - current_chunks) * ACTION_CHUNK
            model.learn(
                total_timesteps=primitive_segment,
                callback=callback,
                reset_num_timesteps=False,
                tb_log_name="per_step_residual_ppo",
            )
            latest_train_metrics = {
                key: float(value)
                for key, value in model.logger.name_to_value.items()
                if key.startswith("train/")
                and isinstance(value, (int, float, np.number))
            }
            # SB3 records the final PPO update after its last periodic dump.
            # Flush explicitly so a one-rollout smoke and the final production
            # rollout both persist their actual update diagnostics.
            model.logger.dump(step=int(model.num_timesteps))
            current_chunks = int(model.num_timesteps) // ACTION_CHUNK
            if current_chunks % eval_interval == 0:
                evaluation = evaluate_exact(
                    model=model,
                    make_environment=make_eval_environment,
                    seeds=list(
                        range(
                            10_000,
                            10_000 + arguments.online_eval_episodes,
                        )
                    ),
                    policy_seed_start=20_000,
                    equivalent_chunks=current_chunks,
                    deterministic=True,
                )
                write_evaluation(
                    run_directory
                    / "evaluations"
                    / f"online_{current_chunks:012d}_deterministic",
                    evaluation,
                )
            if current_chunks % checkpoint_interval == 0:
                model.save(
                    run_directory
                    / "checkpoints"
                    / f"model_{current_chunks:012d}"
                )
            manifest.update(
                {
                    "status": "training",
                    "current_equivalent_chunk_transitions": current_chunks,
                    "current_primitive_transitions": int(model.num_timesteps),
                    "nominal_primitive_steps": int(model.num_timesteps),
                    "actual_primitive_env_steps": int(model.num_timesteps),
                    "ppo_optimizer_steps": int(model.ppo_optimizer_steps),
                    "ppo_epochs_completed": int(model._n_updates),
                    "training_diagnostics": callback.means(),
                    "latest_ppo_train_metrics": latest_train_metrics,
                }
            )
            atomic_write_json(manifest_path, manifest)

        bundle_path = (
            run_directory / "resume" / f"chunk_{current_chunks:012d}"
        )
        save_bundle(
            bundle_path,
            model=model,
            environment=train_environment,
            manifest=manifest,
            equivalent_chunks=current_chunks,
        )
        manifest["latest_resume_chunks"] = current_chunks
        if current_chunks < target_chunks:
            manifest["status"] = "interrupted"
            manifest["interrupted_at_equivalent_chunks"] = current_chunks
            atomic_write_json(manifest_path, manifest)
            return

        final_deterministic = evaluate_exact(
            model=model,
            make_environment=make_eval_environment,
            seeds=list(range(10_000, 10_000 + arguments.final_eval_episodes)),
            policy_seed_start=20_000,
            equivalent_chunks=current_chunks,
            deterministic=True,
        )
        final_stochastic = evaluate_exact(
            model=model,
            make_environment=make_eval_environment,
            seeds=list(range(10_000, 10_000 + arguments.final_eval_episodes)),
            policy_seed_start=20_000,
            equivalent_chunks=current_chunks,
            deterministic=False,
        )
        write_evaluation(
            run_directory / "evaluations/final_deterministic",
            final_deterministic,
        )
        write_evaluation(
            run_directory / "evaluations/final_stochastic",
            final_stochastic,
        )
        model.save(run_directory / "checkpoints/final_model")
        final_planner_hash = module_state_hash(planner_state_modules)
        manifest.update(
            {
                "status": "complete",
                "current_equivalent_chunk_transitions": current_chunks,
                "current_primitive_transitions": int(model.num_timesteps),
                "nominal_primitive_steps": int(model.num_timesteps),
                "actual_primitive_env_steps": int(model.num_timesteps),
                "ppo_optimizer_steps": int(model.ppo_optimizer_steps),
                "ppo_epochs_completed": int(model._n_updates),
                "final_deterministic_summary": final_deterministic["summary"],
                "final_stochastic_summary": final_stochastic["summary"],
                "final_planner_state_hash": final_planner_hash,
                "planner_state_unchanged": (
                    final_planner_hash == initial_planner_hash
                ),
                "training_diagnostics": callback.means(),
            }
        )
        atomic_write_json(manifest_path, manifest)
        (run_directory / "COMPLETE").touch()
    finally:
        train_environment.close()
        contract.close()


if __name__ == "__main__":
    main()
