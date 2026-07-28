"""Matched DSRL-NA network-warm-start control for Phase 6 pilots.

This is a separate experiment runner so the existing ``algorithm=dsrl_na``
construction and training path in ``train_dsrl.py`` remain untouched.
"""

import math
import os
import random
import sys

import d4rl
import d4rl.gym_mujoco
import gym
import hydra
import numpy as np
import torch
import wandb
from omegaconf import OmegaConf

sys.path.append("./dppo")

from env_utils import ActionChunkWrapper, ObservationWrapperGym
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.dsrl.hierarchical_rfs_dsrl import _LegacyLoadableDSRL
from utils import LoggingCallback, collect_rollouts, load_base_policy


OmegaConf.register_new_resolver("eval", eval, replace=True)
OmegaConf.register_new_resolver("round_up", math.ceil, replace=True)
OmegaConf.register_new_resolver("round_down", math.floor, replace=True)

BASE_PATH = os.path.dirname(os.path.abspath(__file__))


def reset_dsrl_optimizers(model) -> None:
    """Discard checkpoint optimizer state while preserving loaded networks."""

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


def assert_network_warmstart_contract(model, cfg, env) -> None:
    expected_dims = (int(cfg.act_steps), int(cfg.action_dim))
    actual_dims = (model.diffusion_act_chunk, model.diffusion_act_dim)
    if actual_dims != expected_dims:
        raise ValueError(
            f"Checkpoint diffusion dimensions differ: {actual_dims} != {expected_dims}"
        )
    if model.observation_space != env.observation_space:
        raise ValueError("Checkpoint and control observation spaces differ")
    if not np.array_equal(model.action_space.low, env.action_space.low) or not np.array_equal(
        model.action_space.high,
        env.action_space.high,
    ):
        raise ValueError("Checkpoint and control execution bounds differ")
    if model.replay_buffer.pos != 0 or model.replay_buffer.full:
        raise RuntimeError("Matched control must start with an empty replay buffer")

    optimizer_state_sizes = [
        len(model.actor.optimizer.state),
        len(model.critic.optimizer.state),
        len(model.critic_noise.optimizer.state),
    ]
    if model.ent_coef_optimizer is not None:
        optimizer_state_sizes.append(len(model.ent_coef_optimizer.state))
    if any(optimizer_state_sizes):
        raise RuntimeError(
            f"Matched control optimizers are not fresh: {optimizer_state_sizes}"
        )


@hydra.main(
    config_path=os.path.join(BASE_PATH, "cfg/gym"),
    config_name="dsrl_hopper.yaml",
    version_base=None,
)
def main(cfg: OmegaConf) -> None:
    OmegaConf.resolve(cfg)
    if cfg.algorithm != "dsrl_na":
        raise ValueError(
            "The matched-control runner only accepts algorithm=dsrl_na, got "
            f"{cfg.algorithm!r}"
        )

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    if cfg.use_wandb:
        wandb.init(
            project=cfg.wandb.project,
            name=cfg.name,
            group=cfg.wandb.group,
            monitor_gym=True,
            save_code=True,
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    def make_env():
        env = gym.make(cfg.env_name)
        env = ObservationWrapperGym(env, cfg.normalization_path)
        return ActionChunkWrapper(
            env,
            cfg,
            max_episode_steps=cfg.env.max_episode_steps,
        )

    base_policy = load_base_policy(cfg)
    env = make_vec_env(
        make_env,
        n_envs=cfg.env.n_envs,
        vec_env_cls=SubprocVecEnv,
    )
    eval_env = None
    try:
        env.seed(cfg.seed + 1)
        checkpoint_path = hydra.utils.to_absolute_path(
            cfg.rfs_hier_legacy_checkpoint_path
        )
        model = _LegacyLoadableDSRL.load(
            checkpoint_path,
            env=env,
            device=cfg.device,
            custom_objects={"diffusion_policy": base_policy},
            buffer_size=cfg.train.buffer_size_na,
            tensorboard_log=cfg.logdir,
        )
        reset_dsrl_optimizers(model)
        assert_network_warmstart_contract(model, cfg, env)
        if cfg.get("phase6_warmstart_only", False):
            print(
                "PHASE6_DSRL_CONTROL_WARMSTART: PASS "
                f"checkpoint={checkpoint_path} "
                f"alpha={float(torch.exp(model.log_ent_coef).detach().cpu())}"
            )
            return

        checkpoint_callback = CheckpointCallback(
            save_freq=cfg.save_model_interval,
            save_path=cfg.logdir + "/checkpoint/",
            name_prefix="ft_policy",
            save_replay_buffer=cfg.save_replay_buffer,
            save_vecnormalize=True,
        )
        eval_env = make_vec_env(
            make_env,
            n_envs=cfg.env.n_eval_envs,
            vec_env_cls=SubprocVecEnv,
        )
        eval_env.seed(cfg.seed + cfg.env.n_envs + 1)
        max_steps = int(cfg.env.max_episode_steps / cfg.act_steps)
        logging_callback = LoggingCallback(
            action_chunk=cfg.act_steps,
            eval_episodes=int(cfg.num_evals / cfg.env.n_eval_envs),
            log_freq=max_steps,
            use_wandb=cfg.use_wandb,
            eval_env=eval_env,
            eval_freq=cfg.eval_interval,
            num_train_env=cfg.env.n_envs,
            num_eval_env=cfg.env.n_eval_envs,
            rew_offset=cfg.env.reward_offset,
            algorithm="dsrl_na",
            max_steps=max_steps,
            deterministic_eval=cfg.deterministic_eval,
        )

        logging_callback.evaluate(model, deterministic=False)
        if cfg.deterministic_eval:
            logging_callback.evaluate(model, deterministic=True)
        logging_callback.log_count += 1

        if cfg.load_offline_data:
            raise ValueError(
                "Phase 6 matched control requires an empty replay buffer; "
                "load_offline_data must remain false"
            )
        if cfg.train.init_rollout_steps > 0:
            collect_rollouts(
                model,
                env,
                cfg.train.init_rollout_steps,
                base_policy,
                cfg,
            )
            logging_callback.set_timesteps(
                cfg.train.init_rollout_steps * cfg.env.n_envs
            )

        model.learn(
            total_timesteps=cfg.total_timesteps,
            callback=[checkpoint_callback, logging_callback],
        )
        if cfg.name:
            model.save(cfg.logdir + "/checkpoint/final")
    finally:
        env.close()
        if eval_env is not None:
            eval_env.close()
        if cfg.use_wandb:
            wandb.finish()


if __name__ == "__main__":
    main()
