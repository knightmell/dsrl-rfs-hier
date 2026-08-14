from __future__ import annotations

from typing import Any

import numpy as np
import torch as th
from gymnasium import Env, spaces

from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.dsrl.hierarchical_replay_buffer import (
    BranchMode,
    NoiseSampleSource,
    TerminationSemantics,
    TransitionOrigin,
)
from stable_baselines3.dsrl.hierarchical_rfs_dsrl import HierarchicalRFSDSRL


class TinyChunkEnv(Env):
    observation_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
    action_space = spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)

    def __init__(self, episode_chunks: int = 3):
        super().__init__()
        self.episode_chunks = episode_chunks
        self.chunk = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.chunk = 0
        return np.zeros(3, dtype=np.float32), {}

    def step(self, action_exec):
        del action_exec
        self.chunk += 1
        terminated = self.chunk >= self.episode_chunks
        info = {
            "nominal_primitive_steps": 2,
            "actual_primitive_steps": 2,
            "termination_primitive_index": 1 if terminated else None,
            "action_chunk_termination_semantics": "early_break_on_done",
        }
        return (
            np.full(3, self.chunk / 10, dtype=np.float32),
            1.0,
            terminated,
            False,
            info,
        )


class IdentityChunkDecoder(th.nn.Module):
    def __init__(self):
        super().__init__()
        self.gain = th.nn.Parameter(th.ones(()))
        self.calls = 0

    def forward(self, observation, noise_decoder_input, return_numpy=False):
        del observation
        self.calls += 1
        result = (noise_decoder_input * self.gain).clamp(-1.0, 1.0)
        return result.detach().cpu().numpy() if return_numpy else result


POLICY_KWARGS = {
    "net_arch": [16, 16],
    "activation_fn": th.nn.Tanh,
    "post_linear_modules": [th.nn.LayerNorm],
}


def make_model(
    *,
    n_envs: int = 1,
    decoder: Any | None = None,
    **overrides: Any,
) -> tuple[HierarchicalRFSDSRL, Any]:
    decoder = decoder or IdentityChunkDecoder()
    environment = (
        TinyChunkEnv()
        if n_envs == 1
        else DummyVecEnv([lambda: TinyChunkEnv() for _ in range(n_envs)])
    )
    kwargs: dict[str, Any] = {
        "learning_rate": 3e-4,
        "buffer_size": 64,
        "learning_starts": 1,
        "batch_size": 2,
        "min_branch_replay_transitions": 2,
        "ent_coef": "auto_0.2",
        "target_entropy": 0.0,
        "target_update_interval": 1,
        "device": "cpu",
        "policy_kwargs": POLICY_KWARGS,
        "diffusion_policy": decoder,
        "diffusion_act_dim": (2, 2),
        "exec_action_low": np.full(4, -1.0, dtype=np.float32),
        "exec_action_high": np.full(4, 1.0, dtype=np.float32),
        "schedule_profile": "fresh_frozen_ddim_5m",
        "phase_b_steps": 4 * n_envs,
        "phase_r_steps": 8 * n_envs,
        "phase_j_steps": 0,
        "phase_j_enabled": False,
        "beta_ramp_steps": 4 * n_envs,
        "base_lane_probability": 0.5,
        "train_freq": 1,
        "seed": 7,
        "lane_seed": 29,
    }
    kwargs.update(overrides)
    model = HierarchicalRFSDSRL("MlpPolicy", environment, **kwargs)
    model.set_logger(configure(folder=None, format_strings=[]))
    return model, decoder


def metadata_row(
    model: HierarchicalRFSDSRL,
    branch: BranchMode,
    *,
    row: int = 0,
    done: bool = False,
) -> tuple[dict[str, np.ndarray], np.ndarray, list[dict[str, Any]]]:
    n_envs = model.n_envs
    action = np.full((n_envs, model.action_dim_flat), row / 100, np.float32)
    residual = np.zeros_like(action)
    applied = branch == BranchMode.JOINT
    metadata = {
        "branch_mode": np.full(n_envs, int(branch), np.uint8),
        "noise_scaled": np.full_like(action, row / 200),
        "noise_log_prob": np.full(n_envs, -0.5, np.float32),
        "noise_log_prob_valid": np.ones(n_envs, np.bool_),
        "noise_sample_source": np.full(
            n_envs, int(NoiseSampleSource.CURRENT_ACTOR), np.uint8
        ),
        "transition_origin": np.full(
            n_envs, int(TransitionOrigin.ONLINE), np.uint8
        ),
        "action_base": action.copy(),
        "residual_pre_tanh": residual.copy(),
        "residual_unit": residual.copy(),
        "action_residual_delta": residual.copy(),
        "action_exec": action.copy(),
        "beta": np.full(n_envs, 0.1 if applied else 0.0, np.float32),
        "residual_applied": np.full(n_envs, applied, np.bool_),
        "emergency_clamp_applied": np.zeros(n_envs, np.bool_),
        "episode_id": np.arange(n_envs, dtype=np.int64) + row * 100,
        "environment_id": np.arange(n_envs, dtype=np.int32),
        "chunk_index_in_episode": np.full(n_envs, row, np.int32),
        "nominal_primitive_steps": np.full(n_envs, 2, np.uint8),
        "actual_primitive_steps": np.full(n_envs, 2, np.uint8),
        "termination_primitive_index": np.full(
            n_envs, 1 if done else -1, np.int8
        ),
        "termination_semantics": np.full(
            n_envs, int(TerminationSemantics.EARLY_BREAK_ON_DONE), np.uint8
        ),
        "noise_policy_version": np.zeros(n_envs, np.int64),
        "residual_policy_version": np.full(n_envs, 0 if applied else -1, np.int64),
    }
    dones = np.full(n_envs, done, np.bool_)
    infos = [
        {"TimeLimit.truncated": False}
        for _ in range(n_envs)
    ]
    return metadata, dones, infos


def populate_branch(
    model: HierarchicalRFSDSRL,
    branch: BranchMode,
    rows: int = 2,
) -> None:
    for row in range(rows):
        metadata, dones, infos = metadata_row(model, branch, row=row)
        observations = np.full((model.n_envs, 3), row / 10, np.float32)
        model.replay_buffer.add_hierarchy(
            observations,
            observations + 0.01,
            metadata["action_exec"],
            np.ones(model.n_envs, np.float32),
            dones,
            infos,
            metadata=metadata,
        )
