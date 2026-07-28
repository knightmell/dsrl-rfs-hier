"""Production P6 runner for matched DSRL control and hierarchy experiments."""

from __future__ import annotations

import json
import math
import os
import random
import sys
import traceback
from pathlib import Path
from typing import Any, Mapping

import gym
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.tensorboard import SummaryWriter

BASE_PATH = Path(__file__).resolve().parent
sys.path.append(str(BASE_PATH / "dppo"))

from env_utils import ACTION_CHUNK_EARLY_BREAK, ActionChunkWrapper, ObservationWrapperGym
from p6_checkpointing import (
    P6CheckpointManager,
    P6IntentionalInterruption,
    P6TrainingCallback,
    initial_runtime_state,
    load_resume_payload,
    validate_loaded_resume,
)
from p6_evaluation import evaluate_exact_episodes, persist_evaluation
from p6_preflight import (
    CONTROL_ALGORITHM,
    HIERARCHY_ALGORITHM,
    finalize_loaded_model_preflight,
    resolve_seed_plan,
    run_preflight,
    static_preflight,
    validate_execution_bounds,
)
from p6_runtime import (
    atomic_write_json,
    collect_or_load_matched_prefill,
    isolated_rng,
    populate_replay_buffer,
    restore_rng_state,
    seed_all,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.dsrl.hierarchical_rfs_dsrl import (
    HierarchicalRFSDSRL,
    _LegacyLoadableDSRL,
)
from utils import load_base_policy


OmegaConf.register_new_resolver("eval", eval, replace=True)
OmegaConf.register_new_resolver("round_up", math.ceil, replace=True)
OmegaConf.register_new_resolver("round_down", math.floor, replace=True)


class P6ControlDSRL(_LegacyLoadableDSRL):
    """Legacy DSRL with P6-only persistent optimizer counters."""

    def _setup_model(self) -> None:
        super()._setup_model()
        for name in (
            "action_critic_optimizer_steps",
            "modulation_critic_optimizer_steps",
            "noise_actor_optimizer_steps",
            "residual_actor_optimizer_steps",
            "hierarchy_train_calls",
        ):
            if not hasattr(self, name):
                setattr(self, name, 0)

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        super().train(gradient_steps=gradient_steps, batch_size=batch_size)
        actor_steps = (
            gradient_steps
            if self.actor_gradient_steps < 0
            else self.actor_gradient_steps
        )
        self.action_critic_optimizer_steps += int(gradient_steps)
        self.modulation_critic_optimizer_steps += int(
            self.noise_critic_grad_steps
        )
        self.noise_actor_optimizer_steps += int(actor_steps)
        self.hierarchy_train_calls += 1

    def _get_torch_save_params(self) -> tuple[list[str], list[str]]:
        state_dicts, pytorch_variables = super()._get_torch_save_params()
        if "critic_noise.optimizer" not in state_dicts:
            state_dicts.append("critic_noise.optimizer")
        return state_dicts, pytorch_variables


def _reset_control_optimizers(model: P6ControlDSRL) -> None:
    learning_rate = model.lr_schedule(1)
    model.actor.optimizer = model.policy.optimizer_class(
        model.actor.parameters(),
        lr=learning_rate,
        **model.policy.optimizer_kwargs,
    )
    if model.policy.share_features_extractor:
        critic_parameters = [
            parameter
            for name, parameter in model.critic.named_parameters()
            if "features_extractor" not in name
        ]
    else:
        critic_parameters = list(model.critic.parameters())
    model.critic.optimizer = model.policy.optimizer_class(
        critic_parameters,
        lr=learning_rate,
        **model.policy.optimizer_kwargs,
    )
    model.critic_noise.optimizer = model.policy.optimizer_class(
        model.critic_noise.parameters(),
        lr=learning_rate,
        **model.policy.optimizer_kwargs,
    )
    if model.log_ent_coef is not None:
        model.ent_coef_optimizer = torch.optim.Adam(
            [model.log_ent_coef],
            lr=learning_rate,
        )
    model.num_timesteps = 0
    model._n_updates = 0
    model._episode_num = 0
    model._last_obs = None
    for name in (
        "action_critic_optimizer_steps",
        "modulation_critic_optimizer_steps",
        "noise_actor_optimizer_steps",
        "residual_actor_optimizer_steps",
        "hierarchy_train_calls",
    ):
        setattr(model, name, 0)


def _make_policy_kwargs(cfg: Any) -> dict[str, Any]:
    hidden = [int(cfg.train.layer_size)] * int(cfg.train.num_layers)
    post_linear_modules = [torch.nn.LayerNorm] if cfg.train.use_layer_norm else None
    return {
        "net_arch": {"pi": hidden, "qf": hidden},
        "activation_fn": torch.nn.Tanh,
        "log_std_init": 0.0,
        "post_linear_modules": post_linear_modules,
        "n_critics": int(cfg.train.n_critics),
    }


def _make_hopper_environment(cfg: Any, normalization_path: Path):
    # Import lazily so CPU-only unit tests can exercise runner state classes
    # without requiring the legacy D4RL/MuJoCo environment.
    import d4rl  # noqa: F401
    import d4rl.gym_mujoco  # noqa: F401

    raw_environment = gym.make(cfg.env_name)
    normalized_environment = ObservationWrapperGym(
        raw_environment,
        normalization_path,
    )
    return ActionChunkWrapper(
        normalized_environment,
        cfg,
        max_episode_steps=int(cfg.env.max_episode_steps),
        action_chunk_termination_semantics=ACTION_CHUNK_EARLY_BREAK,
    )


def _load_legacy_network(
    *,
    cfg: Any,
    environment: Any,
    diffusion_policy: Any,
    buffer_size: int,
) -> P6ControlDSRL:
    checkpoint_path = hydra.utils.to_absolute_path(
        str(cfg.rfs_hier_legacy_checkpoint_path)
    )
    model = P6ControlDSRL.load(
        checkpoint_path,
        env=environment,
        device=cfg.device,
        custom_objects={"diffusion_policy": diffusion_policy},
        buffer_size=buffer_size,
        tensorboard_log=cfg.logdir,
        seed=int(cfg.seed),
    )
    expected_steps = int(cfg.p6.expected_init_checkpoint_steps)
    if int(model.num_timesteps) != expected_steps:
        raise ValueError(
            f"Official checkpoint step mismatch: {model.num_timesteps} != "
            f"{expected_steps}"
        )
    if model.critic_backup_combine_type != "min":
        raise ValueError("P6 checkpoint must use critic_backup_combine_type='min'")
    return model


def _construct_hierarchy(
    cfg: Any,
    environment: Any,
    diffusion_policy: Any,
) -> HierarchicalRFSDSRL:
    execution_low = np.asarray(environment.action_space.low, dtype=np.float32)
    execution_high = np.asarray(environment.action_space.high, dtype=np.float32)
    model = HierarchicalRFSDSRL(
        "MlpPolicy",
        environment,
        learning_rate=float(cfg.train.actor_lr),
        buffer_size=int(cfg.train.buffer_size_na),
        learning_starts=1,
        batch_size=int(cfg.train.batch_size),
        tau=float(cfg.train.tau),
        gamma=float(cfg.train.discount),
        train_freq=int(cfg.train.train_freq),
        gradient_steps=int(cfg.train.utd),
        action_noise=None,
        optimize_memory_usage=False,
        ent_coef="auto" if cfg.train.ent_coef == -1 else cfg.train.ent_coef,
        target_update_interval=1,
        target_entropy=(
            "auto" if cfg.train.target_ent == -1 else cfg.train.target_ent
        ),
        use_sde=False,
        sde_sample_freq=-1,
        tensorboard_log=cfg.logdir,
        verbose=1,
        device=cfg.device,
        policy_kwargs=_make_policy_kwargs(cfg),
        diffusion_policy=diffusion_policy,
        diffusion_act_dim=(int(cfg.act_steps), int(cfg.action_dim)),
        noise_critic_grad_steps=int(cfg.train.noise_critic_grad_steps),
        critic_backup_combine_type=str(cfg.train.critic_backup_combine_type),
        exec_action_low=execution_low,
        exec_action_high=execution_high,
        residual_scale=float(cfg.train.rfs_hier_residual_scale),
        residual_penalty_coef=float(cfg.train.rfs_hier_residual_penalty_coef),
        residual_net_arch=tuple(cfg.train.rfs_hier_residual_net_arch),
        residual_activation=str(cfg.train.rfs_hier_residual_activation),
        residual_lr=float(cfg.train.rfs_hier_residual_lr),
        noise_actor_gradient_steps=int(
            cfg.train.rfs_hier_noise_actor_gradient_steps
        ),
        residual_actor_gradient_steps=int(
            cfg.train.rfs_hier_residual_actor_gradient_steps
        ),
        diagnostics_interval_updates=int(
            cfg.p6.diagnostics_interval_updates
        ),
        seed=int(cfg.seed),
    )
    model.initialize_from_legacy_checkpoint(
        hydra.utils.to_absolute_path(str(cfg.rfs_hier_legacy_checkpoint_path))
    )
    return model


def _verify_real_zero_residual_parity(
    *,
    hierarchy: HierarchicalRFSDSRL,
    legacy: P6ControlDSRL,
    environment: Any,
    seed: int,
) -> dict[str, float]:
    with isolated_rng(seed), torch.no_grad():
        environment.seed(seed)
        observations = environment.reset()
        observations_tensor = torch.as_tensor(
            observations,
            device=hierarchy.device,
            dtype=torch.float32,
        )
        noise_scaled = torch.tanh(
            torch.randn(
                len(observations),
                hierarchy.action_dim_flat,
                device=hierarchy.device,
            )
        )
        decoder_input_numpy = legacy.policy.unscale_action(
            noise_scaled.detach().cpu().numpy()
        )
        decoder_input = torch.as_tensor(
            decoder_input_numpy,
            device=hierarchy.device,
            dtype=torch.float32,
        ).reshape(
            -1,
            hierarchy.diffusion_act_chunk,
            hierarchy.diffusion_act_dim,
        )
        legacy_action = legacy.diffusion_policy(
            observations_tensor,
            decoder_input,
            return_numpy=False,
        ).reshape(-1, hierarchy.action_dim_flat)
        generated = hierarchy._generate_hierarchical_action(
            observations_tensor,
            noise_scaled,
            zero_residual=True,
        )
        action_error = float(
            torch.max(torch.abs(generated.action_exec - legacy_action)).item()
        )
        residual_error = float(torch.max(torch.abs(generated.residual_unit)).item())
        legacy_q = legacy.critic_noise(observations_tensor, noise_scaled)
        modulation_q = hierarchy.critic_modulation(
            observations_tensor,
            noise_scaled,
            torch.zeros_like(noise_scaled),
        )
        q_error = max(
            float(torch.max(torch.abs(actual - expected)).item())
            for actual, expected in zip(modulation_q, legacy_q)
        )
    if action_error > 1e-6 or q_error > 1e-6 or residual_error != 0.0:
        raise ValueError(
            "Real checkpoint migration parity failed: "
            f"action={action_error}, QM={q_error}, residual={residual_error}"
        )
    return {
        "zero_residual_action_max_abs_error": action_error,
        "qm_zero_residual_max_abs_error": q_error,
        "residual_unit_max_abs": residual_error,
    }


def _read_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as input_file:
        return json.load(input_file)


def _update_manifest(path: Path, updates: Mapping[str, Any]) -> dict[str, Any]:
    manifest = _read_manifest(path)
    manifest.update(dict(updates))
    atomic_write_json(path, manifest)
    return manifest


def _verify_existing_manifest(
    *,
    path: Path,
    static_manifest: Mapping[str, Any],
    cfg: Any,
    environment: Any,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError("Resume requested but run_manifest.json is missing")
    manifest = _read_manifest(path)
    for key in (
        "algorithm",
        "run_name",
        "init_checkpoint_sha256",
        "frozen_ddim_sha256",
        "normalization_sha256",
        "chunk_budget",
        "action_chunk_termination_semantics",
    ):
        if manifest.get(key) != static_manifest.get(key):
            raise ValueError(f"Resume run manifest mismatch for {key}")
    execution_contract = validate_execution_bounds(cfg, environment)
    for key, value in execution_contract.items():
        if manifest.get(key) != value:
            raise ValueError(f"Resume execution contract mismatch for {key}")
    return manifest


def _algorithm_label(cfg: Any) -> str:
    if cfg.algorithm == HIERARCHY_ALGORITHM:
        return HIERARCHY_ALGORITHM
    if cfg.algorithm == "dsrl_na":
        return CONTROL_ALGORITHM
    raise ValueError(
        "P6 runner only accepts algorithm=dsrl_na_rfs_hier or dsrl_na"
    )


@hydra.main(
    config_path=str(BASE_PATH / "cfg" / "gym"),
    config_name="p6_hopper",
    version_base=None,
)
def main(cfg: Any) -> None:
    OmegaConf.resolve(cfg)
    algorithm = _algorithm_label(cfg)
    static_manifest = static_preflight(cfg, BASE_PATH, algorithm=algorithm)
    seed_plan = resolve_seed_plan(cfg)
    run_directory = Path(hydra.utils.to_absolute_path(str(cfg.logdir)))
    run_directory.mkdir(parents=True, exist_ok=True)
    manifest_path = run_directory / "run_manifest.json"
    resume_value = cfg.p6.resume_bundle_path
    resume_directory = (
        None
        if resume_value is None
        else Path(hydra.utils.to_absolute_path(str(resume_value)))
    )
    if resume_directory is None and manifest_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing P6 run {run_directory}"
        )

    with open_dict(cfg):
        cfg.model.network_path = static_manifest["frozen_ddim_path"]
        cfg.base_policy_path = static_manifest["frozen_ddim_path"]
        cfg.normalization_path = static_manifest["normalization_path"]
    diffusion_policy = load_base_policy(cfg)
    make_environment = lambda: _make_hopper_environment(
        cfg,
        Path(static_manifest["normalization_path"]),
    )
    training_environment = make_vec_env(
        make_environment,
        n_envs=int(cfg.env.n_envs),
        vec_env_cls=SubprocVecEnv,
    )
    writer = SummaryWriter(log_dir=str(run_directory / "tensorboard" / "p6"))
    wandb_run = None
    model = None
    manager = None
    try:
        if cfg.use_wandb:
            import wandb

            wandb_run = wandb.init(
                project=cfg.wandb.project,
                name=cfg.name,
                group=cfg.wandb.group,
                config=OmegaConf.to_container(cfg, resolve=True),
            )

        if resume_directory is None:
            manifest = run_preflight(
                cfg,
                training_environment,
                BASE_PATH,
                manifest_path,
                algorithm=algorithm,
                static_manifest=static_manifest,
            )
            warmstart = _load_legacy_network(
                cfg=cfg,
                environment=training_environment,
                diffusion_policy=diffusion_policy,
                buffer_size=(
                    int(cfg.train.buffer_size_na)
                    if algorithm == CONTROL_ALGORITHM
                    else 1
                ),
            )
            if algorithm == CONTROL_ALGORITHM:
                model = warmstart
                _reset_control_optimizers(model)
                parity = {
                    "zero_residual_action_max_abs_error": 0.0,
                    "qm_zero_residual_max_abs_error": 0.0,
                    "residual_unit_max_abs": 0.0,
                }
            else:
                model = _construct_hierarchy(
                    cfg,
                    training_environment,
                    diffusion_policy,
                )
                parity = _verify_real_zero_residual_parity(
                    hierarchy=model,
                    legacy=warmstart,
                    environment=training_environment,
                    seed=int(cfg.seed) + 50_000,
                )
            finalize_loaded_model_preflight(
                cfg,
                training_environment,
                model,
                manifest_path,
                network_warmstart=True,
            )

            prefill_path = Path(manifest["prefill_artifact_path"])
            prefill_arrays, prefill_metadata, generated = (
                collect_or_load_matched_prefill(
                    artifact_path=prefill_path,
                    warmstart_model=warmstart,
                    env=training_environment,
                    vector_steps=int(cfg.train.init_rollout_steps),
                    environment_seed=int(seed_plan["prefill_env_seed"]),
                    policy_seed=int(seed_plan["prefill_policy_seed"]),
                    action_chunk=int(cfg.act_steps),
                    termination_semantics=str(
                        cfg.p6.action_chunk_termination_semantics
                    ),
                    provenance=static_manifest,
                )
            )
            replay_hash = populate_replay_buffer(
                model.replay_buffer,
                prefill_arrays,
            )
            expected_capacity = (
                int(prefill_metadata["chunk_transitions"])
                + int(cfg.total_timesteps)
            )
            if int(cfg.train.buffer_size_na) <= expected_capacity:
                raise ValueError(
                    "Replay buffer must exceed prefill + online chunk transitions: "
                    f"{cfg.train.buffer_size_na} <= {expected_capacity}"
                )
            runtime_state = initial_runtime_state(
                target_chunk_budget=int(cfg.total_timesteps),
                action_chunk=int(cfg.act_steps),
                prefill_metadata=prefill_metadata,
                replay_hash=replay_hash,
                online_eval_interval=int(
                    cfg.p6.online_eval_interval_chunk_transitions
                ),
                model_checkpoint_interval=int(
                    cfg.p6.model_checkpoint_interval_chunk_transitions
                ),
                replay_checkpoint_interval=int(
                    cfg.p6.replay_checkpoint_interval_chunk_transitions
                ),
            )
            _update_manifest(
                manifest_path,
                {
                    "prefill_status": "verified_and_loaded",
                    "prefill_generated_by_this_run": generated,
                    "prefill_hash": prefill_metadata["semantic_hash"],
                    "prefill_archive_sha256": prefill_metadata["archive_sha256"],
                    "prefill_primitive_counters": prefill_metadata[
                        "primitive_counters"
                    ],
                    "initial_replay_hash": replay_hash,
                    "zero_residual_ddim_parity": parity,
                    "training_status": "ready",
                },
            )
            seed_all(int(cfg.seed))
            training_environment.seed(int(seed_plan["train_env_seed"]))
            model._last_obs = None
        else:
            _verify_existing_manifest(
                path=manifest_path,
                static_manifest=static_manifest,
                cfg=cfg,
                environment=training_environment,
            )
            bundle_manifest, runtime_payload = load_resume_payload(
                resume_directory,
                algorithm=algorithm,
                expected_provenance=static_manifest,
            )
            model_path = resume_directory / "model.zip"
            if algorithm == HIERARCHY_ALGORITHM:
                model = HierarchicalRFSDSRL.load(
                    model_path,
                    env=training_environment,
                    device=cfg.device,
                    diffusion_policy=diffusion_policy,
                )
            else:
                model = P6ControlDSRL.load(
                    model_path,
                    env=training_environment,
                    device=cfg.device,
                    custom_objects={"diffusion_policy": diffusion_policy},
                )
            model.load_replay_buffer(resume_directory / "replay_buffer.pkl")
            validate_loaded_resume(
                model=model,
                bundle_directory=resume_directory,
                bundle_manifest=bundle_manifest,
                runtime_payload=runtime_payload,
            )
            runtime_state = runtime_payload["runtime_state"]
            runtime_state["resume_count"] = int(runtime_state["resume_count"]) + 1
            runtime_state["environment_reset_discontinuities"] = (
                int(runtime_state["environment_reset_discontinuities"]) + 1
            )
            restore_rng_state(runtime_payload["rng_state"])
            training_environment.seed(
                int(seed_plan["train_env_seed"])
                + 100_000 * int(runtime_state["resume_count"])
            )
            model._last_obs = None
            _update_manifest(
                manifest_path,
                {
                    "training_status": "resuming",
                    "resume_source": str(resume_directory),
                    "resume_count": runtime_state["resume_count"],
                    "environment_reset_discontinuities": runtime_state[
                        "environment_reset_discontinuities"
                    ],
                },
            )

        def evaluate_online(chunk: int, counters: Any) -> Mapping[str, Any]:
            seeds = seed_plan["eval_seed_set"][
                : int(cfg.p6.online_eval_episodes)
            ]
            result = evaluate_exact_episodes(
                model=model,
                make_environment=make_environment,
                environment_seeds=seeds,
                policy_seed_start=int(cfg.p6.eval_policy_seed_start),
                deterministic=bool(cfg.deterministic_eval),
                action_chunk=int(cfg.act_steps),
                max_episode_primitive_steps=int(cfg.env.max_episode_steps),
                batch_size=int(cfg.p6.evaluation_batch_size),
                chunk_transitions=chunk,
                nominal_primitive_steps=counters.nominal_primitive_steps,
                actual_primitive_env_steps=counters.actual_primitive_env_steps,
            )
            persist_evaluation(
                result,
                output_prefix=(
                    run_directory
                    / "evaluations"
                    / f"online_{chunk:012d}"
                ),
                tensorboard_writer=writer,
                tensorboard_tag="eval/online",
            )
            if wandb_run is not None:
                wandb_run.log(
                    {
                        f"eval/{key}": value
                        for key, value in result["summary"].items()
                    },
                    step=chunk,
                )
            return result

        manager = P6CheckpointManager(
            run_directory=run_directory,
            algorithm=algorithm,
            manifest_path=manifest_path,
            runtime_state=runtime_state,
            provenance=static_manifest,
            evaluation_function=evaluate_online,
        )
        callback = P6TrainingCallback(
            manager=manager,
            action_chunk=int(cfg.act_steps),
            stop_after_chunk_transitions=(
                None
                if cfg.p6.stop_after_chunk_transitions is None
                else int(cfg.p6.stop_after_chunk_transitions)
            ),
        )
        current_chunks = int(model.num_timesteps)
        remaining_chunks = int(cfg.total_timesteps) - current_chunks
        if remaining_chunks <= 0:
            raise ValueError(
                f"No remaining training budget: target={cfg.total_timesteps}, "
                f"current={current_chunks}"
            )
        _update_manifest(
            manifest_path,
            {
                "training_status": "running",
                "training_start_chunk_transitions": current_chunks,
                "remaining_chunk_transitions": remaining_chunks,
            },
        )
        model.learn(
            total_timesteps=remaining_chunks,
            callback=callback,
            reset_num_timesteps=False,
            tb_log_name="p6_train",
        )
        if int(model.num_timesteps) != int(cfg.total_timesteps):
            raise RuntimeError(
                f"Training ended at {model.num_timesteps}, expected "
                f"{cfg.total_timesteps}"
            )

        final_episode_count = int(cfg.p6.final_eval_episodes)
        final_seeds = seed_plan["eval_seed_set"][:final_episode_count]
        final_counters = manager.runtime_state["training_counters"]
        final_result = evaluate_exact_episodes(
            model=model,
            make_environment=make_environment,
            environment_seeds=final_seeds,
            policy_seed_start=int(cfg.p6.eval_policy_seed_start),
            deterministic=bool(cfg.deterministic_eval),
            action_chunk=int(cfg.act_steps),
            max_episode_primitive_steps=int(cfg.env.max_episode_steps),
            batch_size=int(cfg.p6.evaluation_batch_size),
            chunk_transitions=int(model.num_timesteps),
            nominal_primitive_steps=int(
                final_counters["nominal_primitive_steps"]
            ),
            actual_primitive_env_steps=int(
                final_counters["actual_primitive_env_steps"]
            ),
        )
        persist_evaluation(
            final_result,
            output_prefix=run_directory / "evaluations" / "final",
            tensorboard_writer=writer,
            tensorboard_tag="eval/final",
        )
        _update_manifest(
            manifest_path,
            {
                "training_status": "complete",
                "final_evaluation_summary": final_result["summary"],
                "final_evaluation_episode_count": final_episode_count,
            },
        )
        (run_directory / "COMPLETE").write_text("complete\n")
    except P6IntentionalInterruption as error:
        _update_manifest(
            manifest_path,
            {
                "training_status": "interrupted",
                "interruption_reason": str(error),
            },
        )
        (run_directory / "INTERRUPTED").write_text(str(error) + "\n")
        raise SystemExit(75) from error
    except BaseException as error:
        if manifest_path.exists():
            _update_manifest(
                manifest_path,
                {
                    "training_status": "failed",
                    "failure_type": type(error).__name__,
                    "failure_message": str(error),
                    "failure_traceback": traceback.format_exc(),
                },
            )
        (run_directory / "FAILED").write_text(
            f"{type(error).__name__}: {error}\n"
        )
        raise
    finally:
        writer.flush()
        writer.close()
        training_environment.close()
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
