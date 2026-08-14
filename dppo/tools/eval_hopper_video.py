#!/usr/bin/env python3

import os
from pathlib import Path

# 必须在导入 MuJoCo/Gym 之前设置
os.environ.setdefault("DPPO_LOG_DIR", str(Path.cwd() / "log"))
os.environ.setdefault("DPPO_DATA_DIR", str(Path.cwd() / "log"))

import d4rl  # noqa: F401
import gym
import d4rl.gym_mujoco
import hydra
import imageio.v2 as imageio
import numpy as np
import torch
from omegaconf import OmegaConf


def main() -> None:
    root = Path.cwd()

    OmegaConf.register_new_resolver("eval", eval, replace=True)

    with hydra.initialize_config_dir(
        config_dir=str(root / "cfg/gym/eval/hopper-v2"),
        version_base=None,
    ):
        cfg = hydra.compose(
            config_name="eval_diffusion_mlp",
            overrides=[
                "ft_denoising_steps=0",
                "+model.use_ddim=True",
                "+model.ddim_steps=5",
            ],
        )

    OmegaConf.resolve(cfg)

    print("Checkpoint:", cfg.base_policy_path)
    print("Normalization:", cfg.normalization_path)

    model = hydra.utils.instantiate(cfg.model)
    model.eval()

    normalization = np.load(cfg.normalization_path)
    obs_min = normalization["obs_min"]
    obs_max = normalization["obs_max"]
    action_min = normalization["action_min"]
    action_max = normalization["action_max"]

    def normalize_obs(obs: np.ndarray) -> np.ndarray:
        return 2.0 * (
            (obs - obs_min) / (obs_max - obs_min + 1e-6) - 0.5
        )

    def unnormalize_action(action: np.ndarray) -> np.ndarray:
        action = np.clip(action, -1.0, 1.0)
        action = (action + 1.0) / 2.0
        return action * (action_max - action_min) + action_min

    env = gym.make(cfg.env_name)
    obs = env.reset()

    frames = []
    episode_return = 0.0
    primitive_steps = 0
    done = False

    while not done and primitive_steps < cfg.env.max_episode_steps:
        frame = env.render(
            mode="rgb_array",
            width=640,
            height=480,
        )
        frames.append(frame)

        obs_norm = normalize_obs(obs)
        state = torch.as_tensor(
            obs_norm,
            dtype=torch.float32,
            device=cfg.device,
        ).reshape(1, 1, cfg.obs_dim)

        with torch.no_grad():
            samples = model(
                cond={"state": state},
                deterministic=True,
            )

        action_chunk = samples.trajectories[0, : cfg.act_steps]
        action_chunk = action_chunk.detach().cpu().numpy()

        for normalized_action in action_chunk:
            raw_action = unnormalize_action(normalized_action)
            obs, reward, done, info = env.step(raw_action)

            episode_return += float(reward)
            primitive_steps += 1

            frame = env.render(
                mode="rgb_array",
                width=640,
                height=480,
            )
            frames.append(frame)

            if done:
                break

    env.close()

    output = root / "hopper_frozen_ddim5.mp4"
    imageio.mimsave(
        output,
        frames,
        fps=60,
        codec="libx264",
        quality=8,
    )

    print(f"Episode return: {episode_return:.3f}")
    print(f"Primitive steps: {primitive_steps}")
    print(f"Saved video: {output}")


if __name__ == "__main__":
    main()

