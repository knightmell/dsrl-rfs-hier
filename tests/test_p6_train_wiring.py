from __future__ import annotations

import numpy as np
import torch
from gymnasium import Env, spaces

from p6_train import P6ControlDSRL
from stable_baselines3.common.logger import configure


class TinyEnvironment(Env):
    observation_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
    action_space = spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(3, dtype=np.float32), {}

    def step(self, action):
        del action
        return np.zeros(3, dtype=np.float32), 0.0, False, False, {}


class IdentityDecoder(torch.nn.Module):
    def forward(self, observation, noise, return_numpy=False):
        del observation
        result = noise.clamp(-1.0, 1.0)
        return result.detach().cpu().numpy() if return_numpy else result


def make_control(decoder=None):
    model = P6ControlDSRL(
        "MlpPolicy",
        TinyEnvironment(),
        learning_rate=3e-4,
        buffer_size=64,
        learning_starts=1,
        batch_size=2,
        ent_coef="auto_0.2",
        target_entropy=0.0,
        device="cpu",
        policy_kwargs={
            "net_arch": [8, 8],
            "activation_fn": torch.nn.Tanh,
            "post_linear_modules": [torch.nn.LayerNorm],
        },
        diffusion_policy=decoder or IdentityDecoder(),
        diffusion_act_dim=(2, 2),
        noise_critic_grad_steps=1,
        critic_backup_combine_type="min",
        seed=4,
    )
    model.set_logger(configure(folder=None, format_strings=[]))
    return model


def fill_replay(model):
    for index in range(12):
        observation = np.full((1, 3), index / 12, dtype=np.float32)
        model.replay_buffer.add(
            observation,
            observation + 0.01,
            np.zeros((1, 4), dtype=np.float32),
            np.array([0.0], dtype=np.float32),
            np.array([False]),
            [{}],
        )


def test_control_update_counters_and_qw_optimizer_survive_resume_save(tmp_path):
    model = make_control()
    fill_replay(model)
    model.train(gradient_steps=2, batch_size=2)

    assert model.action_critic_optimizer_steps == 2
    assert model.modulation_critic_optimizer_steps == 1
    assert model.noise_actor_optimizer_steps == 2
    assert model.residual_actor_optimizer_steps == 0
    assert model.hierarchy_train_calls == 1
    assert len(model.critic_noise.optimizer.state) > 0

    observation = np.zeros((2, 3), dtype=np.float32)
    expected_action, _ = model.predict_diffused(observation, deterministic=True)
    checkpoint = tmp_path / "p6_control.zip"
    model.save(checkpoint)
    restored = P6ControlDSRL.load(
        checkpoint,
        env=TinyEnvironment(),
        device="cpu",
        custom_objects={"diffusion_policy": IdentityDecoder()},
    )
    actual_action, _ = restored.predict_diffused(observation, deterministic=True)

    np.testing.assert_allclose(actual_action, expected_action, atol=1e-6, rtol=0.0)
    assert restored.action_critic_optimizer_steps == 2
    assert restored.modulation_critic_optimizer_steps == 1
    assert restored.noise_actor_optimizer_steps == 2
    assert restored.residual_actor_optimizer_steps == 0
    assert restored.hierarchy_train_calls == 1
    assert len(restored.critic_noise.optimizer.state) > 0
