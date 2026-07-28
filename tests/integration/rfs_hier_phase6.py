"""Real Hopper/DDIM5 gates for hierarchical DSRL-NA Phase 6.

Run from the repository root with the project ``dsrl`` Python environment.
This is intentionally a standalone integration script rather than a default
pytest test: it requires MuJoCo assets, the frozen Hopper checkpoint, and CUDA.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import gym
import hydra
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "dppo"))

import d4rl  # noqa: E402,F401
import d4rl.gym_mujoco  # noqa: E402,F401

from env_utils import ActionChunkWrapper, ObservationWrapperGym  # noqa: E402
from stable_baselines3.dsrl.hierarchical_rfs_dsrl import (  # noqa: E402
    HierarchicalRFSDSRL,
    _LegacyLoadableDSRL,
)
from utils import load_base_policy  # noqa: E402


EXPECTED_LEGACY_SHA256 = (
    "e75686d06f7297b870ee8d286fc36db6ecb6cb62e9c6a1783b9466b3f0691fb6"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate", choices=("smoke", "migration"), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def load_config(device: str, seed: int):
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    OmegaConf.register_new_resolver("round_up", math.ceil, replace=True)
    OmegaConf.register_new_resolver("round_down", math.floor, replace=True)
    with initialize_config_dir(
        config_dir=str(ROOT / "cfg" / "gym"),
        version_base=None,
    ):
        cfg = compose(
            config_name="dsrl_hopper",
            overrides=[
                "algorithm=dsrl_na_rfs_hier",
                f"device={device}",
                f"seed={seed}",
                "use_wandb=false",
                "env.n_envs=1",
                "env.n_eval_envs=1",
            ],
        )
    OmegaConf.resolve(cfg)
    return cfg


def make_hopper_env(cfg):
    def factory():
        env = gym.make(cfg.env_name)
        env = ObservationWrapperGym(
            env,
            str((ROOT / cfg.normalization_path).resolve()),
        )
        return ActionChunkWrapper(
            env,
            cfg,
            max_episode_steps=cfg.env.max_episode_steps,
        )

    env = make_vec_env(factory, n_envs=1, vec_env_cls=DummyVecEnv)
    env.seed(cfg.seed + 1)
    return env


def make_policy_kwargs(cfg) -> dict[str, Any]:
    post_linear_modules = [torch.nn.LayerNorm] if cfg.train.use_layer_norm else None
    net_arch = [cfg.train.layer_size] * cfg.train.num_layers
    return {
        "net_arch": {"pi": net_arch, "qf": net_arch},
        "activation_fn": torch.nn.Tanh,
        "log_std_init": 0.0,
        "post_linear_modules": post_linear_modules,
        "n_critics": cfg.train.n_critics,
    }


def make_hierarchical_model(cfg, env, base_policy) -> HierarchicalRFSDSRL:
    exec_action_low = np.asarray(env.action_space.low, dtype=np.float32).reshape(-1)
    exec_action_high = np.asarray(env.action_space.high, dtype=np.float32).reshape(-1)
    return HierarchicalRFSDSRL(
        "MlpPolicy",
        env,
        learning_rate=cfg.train.actor_lr,
        buffer_size=32,
        learning_starts=1,
        batch_size=4,
        tau=cfg.train.tau,
        gamma=cfg.train.discount,
        train_freq=cfg.train.train_freq,
        gradient_steps=cfg.train.utd,
        ent_coef="auto" if cfg.train.ent_coef == -1 else cfg.train.ent_coef,
        target_update_interval=1,
        target_entropy=(
            "auto" if cfg.train.target_ent == -1 else cfg.train.target_ent
        ),
        device=cfg.device,
        policy_kwargs=make_policy_kwargs(cfg),
        diffusion_policy=base_policy,
        diffusion_act_dim=(cfg.act_steps, cfg.action_dim),
        noise_critic_grad_steps=cfg.train.noise_critic_grad_steps,
        critic_backup_combine_type=cfg.train.critic_backup_combine_type,
        exec_action_low=exec_action_low,
        exec_action_high=exec_action_high,
        residual_scale=cfg.train.rfs_hier_residual_scale,
        residual_penalty_coef=cfg.train.rfs_hier_residual_penalty_coef,
        residual_net_arch=cfg.train.rfs_hier_residual_net_arch,
        residual_activation=cfg.train.rfs_hier_residual_activation,
        residual_lr=cfg.train.rfs_hier_residual_lr,
        noise_actor_gradient_steps=cfg.train.rfs_hier_noise_actor_gradient_steps,
        residual_actor_gradient_steps=cfg.train.rfs_hier_residual_actor_gradient_steps,
    )


def tensor_max_abs(value: torch.Tensor) -> float:
    return float(value.detach().abs().max().cpu())


def state_dict_max_abs(
    actual: dict[str, torch.Tensor],
    expected: dict[str, torch.Tensor],
) -> float:
    if actual.keys() != expected.keys():
        raise AssertionError("State-dict keys differ")
    return max(
        tensor_max_abs(actual[key] - expected[key])
        for key in actual
    )


def fixed_noise(batch_size: int, action_dim: int, device: torch.device) -> torch.Tensor:
    rows = [
        torch.linspace(-0.9 + 0.1 * index, 0.9 - 0.1 * index, action_dim)
        for index in range(batch_size)
    ]
    return torch.stack(rows).to(device=device, dtype=torch.float32)


def assert_finite(name: str, value: torch.Tensor) -> None:
    if not bool(torch.isfinite(value).all()):
        raise AssertionError(f"{name} contains non-finite values")


def run_smoke(cfg, env, base_policy, model) -> dict[str, Any]:
    if not cfg.model.use_ddim or cfg.model.ddim_steps != 5:
        raise AssertionError("Smoke gate requires the audited DDIM5 decoder")
    observation_numpy = env.reset()
    observation = torch.as_tensor(
        observation_numpy,
        device=model.device,
        dtype=torch.float32,
    )
    noise_scaled = fixed_noise(1, model.action_dim_flat, model.device)
    with torch.no_grad():
        generated = model._generate_hierarchical_action(
            observation,
            noise_scaled,
        )

    for field in generated._fields:
        assert_finite(field, getattr(generated, field))
    if tensor_max_abs(generated.residual_unit) != 0.0:
        raise AssertionError("Zero-initialized residual actor produced a non-zero residual")
    action_parity_max_abs = tensor_max_abs(
        generated.action_exec - generated.action_base
    )
    if action_parity_max_abs > 1e-6:
        raise AssertionError(
            f"Initial action parity exceeded 1e-6: {action_parity_max_abs}"
        )
    exec_action_low = torch.as_tensor(
        model.exec_action_low,
        device=model.device,
    )
    exec_action_high = torch.as_tensor(
        model.exec_action_high,
        device=model.device,
    )
    if not bool(
        (generated.action_exec >= exec_action_low).all()
        and (generated.action_exec <= exec_action_high).all()
    ):
        raise AssertionError("Generated action_exec violates execution bounds")

    next_observation, reward, done, infos = env.step(
        generated.action_exec.cpu().numpy()
    )
    if not np.isfinite(next_observation).all() or not np.isfinite(reward).all():
        raise AssertionError("Hopper step returned non-finite observation/reward")

    decoder = base_policy.base_policy
    decoder_parameters = tuple(decoder.parameters())
    return {
        "gate": "smoke",
        "status": "PASS",
        "device": str(model.device),
        "ddim_steps": int(cfg.model.ddim_steps),
        "observation_shape": list(observation.shape),
        "noise_scaled_shape": list(generated.noise_scaled.shape),
        "noise_decoder_input_shape": list(generated.noise_decoder_input.shape),
        "action_base_shape": list(generated.action_base.shape),
        "action_exec_shape": list(generated.action_exec.shape),
        "action_base_min": float(generated.action_base.min().cpu()),
        "action_base_max": float(generated.action_base.max().cpu()),
        "action_parity_max_abs": action_parity_max_abs,
        "residual_unit_max_abs": tensor_max_abs(generated.residual_unit),
        "action_exec_min": float(generated.action_exec.min().cpu()),
        "action_exec_max": float(generated.action_exec.max().cpu()),
        "decoder_training": bool(decoder.training),
        "decoder_parameter_count": sum(p.numel() for p in decoder_parameters),
        "step_reward": float(reward[0]),
        "step_done": bool(done[0]),
        "step_info_count": len(infos),
    }


def run_migration(cfg, env, base_policy, model) -> dict[str, Any]:
    checkpoint_path = (ROOT / cfg.rfs_hier_legacy_checkpoint_path).resolve()
    checkpoint_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    if checkpoint_hash != EXPECTED_LEGACY_SHA256:
        raise AssertionError(
            f"Legacy checkpoint SHA-256 mismatch: {checkpoint_hash}"
        )

    legacy = _LegacyLoadableDSRL.load(
        checkpoint_path,
        env=env,
        device=cfg.device,
        custom_objects={"diffusion_policy": base_policy},
        buffer_size=1,
    )
    model.initialize_from_legacy_checkpoint(checkpoint_path)

    parameter_differences = {
        "actor": state_dict_max_abs(
            model.actor.state_dict(),
            legacy.actor.state_dict(),
        ),
        "action_critic": state_dict_max_abs(
            model.critic.state_dict(),
            legacy.critic.state_dict(),
        ),
        "action_critic_target": state_dict_max_abs(
            model.critic_target.state_dict(),
            legacy.critic_target.state_dict(),
        ),
    }
    if any(value != 0.0 for value in parameter_differences.values()):
        raise AssertionError(f"Legacy parameter migration was not exact: {parameter_differences}")

    observation_numpy = env.reset()
    observation = torch.as_tensor(
        np.repeat(observation_numpy, 3, axis=0),
        device=model.device,
        dtype=torch.float32,
    )
    noise_scaled = fixed_noise(3, model.action_dim_flat, model.device)
    with torch.no_grad():
        generated = model._generate_hierarchical_action(
            observation,
            noise_scaled,
        )
        legacy_noise_decoder_input = torch.as_tensor(
            legacy.policy.unscale_action(noise_scaled.cpu().numpy()),
            device=model.device,
            dtype=torch.float32,
        ).reshape(3, cfg.act_steps, cfg.action_dim)
        legacy_action = base_policy(
            observation,
            legacy_noise_decoder_input,
            return_numpy=False,
        ).reshape(3, model.action_dim_flat)
        legacy_qw = legacy.critic_noise(observation, noise_scaled)
        migrated_qm = model.critic_modulation(
            observation,
            noise_scaled,
            torch.zeros_like(noise_scaled),
        )

    residual_unit_max_abs = tensor_max_abs(generated.residual_unit)
    action_parity_max_abs = tensor_max_abs(generated.action_exec - legacy_action)
    qm_parity_max_abs = max(
        tensor_max_abs(qm - qw)
        for qm, qw in zip(migrated_qm, legacy_qw)
    )
    alpha_max_abs = tensor_max_abs(model.log_ent_coef - legacy.log_ent_coef)
    if residual_unit_max_abs != 0.0:
        raise AssertionError("Residual is non-zero immediately after migration")
    if action_parity_max_abs > 1e-6:
        raise AssertionError(
            f"Legacy action parity exceeded 1e-6: {action_parity_max_abs}"
        )
    if qm_parity_max_abs > 1e-6:
        raise AssertionError(f"QW/QM parity exceeded 1e-6: {qm_parity_max_abs}")
    if alpha_max_abs != 0.0:
        raise AssertionError(f"Entropy coefficient migration was not exact: {alpha_max_abs}")
    if model.target_entropy != legacy.target_entropy:
        raise AssertionError("Legacy target_entropy semantics were not restored")

    optimizer_state_sizes = {
        "noise_actor": len(model.actor.optimizer.state),
        "action_critic": len(model.critic.optimizer.state),
        "modulation_critic": len(model.critic_modulation.optimizer.state),
        "residual_actor": len(model.residual_actor_optimizer.state),
        "entropy": len(model.ent_coef_optimizer.state),
    }
    if any(optimizer_state_sizes.values()):
        raise AssertionError(f"Migrated optimizers are not fresh: {optimizer_state_sizes}")

    return {
        "gate": "migration",
        "status": "PASS",
        "device": str(model.device),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_hash,
        "parameter_max_abs": parameter_differences,
        "residual_unit_max_abs": residual_unit_max_abs,
        "action_parity_max_abs": action_parity_max_abs,
        "qm_parity_max_abs": qm_parity_max_abs,
        "alpha_max_abs": alpha_max_abs,
        "target_entropy": float(model.target_entropy),
        "optimizer_state_sizes": optimizer_state_sizes,
    }


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {args.device}")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    cfg = load_config(args.device, args.seed)
    env = make_hopper_env(cfg)
    try:
        base_policy = load_base_policy(cfg)
        model = make_hierarchical_model(cfg, env, base_policy)
        if args.gate == "smoke":
            result = run_smoke(cfg, env, base_policy, model)
        else:
            result = run_migration(cfg, env, base_policy, model)
    finally:
        env.close()

    output = json.dumps(result, indent=2, sort_keys=True)
    print(output)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n")


if __name__ == "__main__":
    main()
