import numpy as np
import pytest
import torch as th
from gymnasium import Env, spaces

from stable_baselines3 import DSRL, RFSDSRL
from stable_baselines3.common.logger import configure
from stable_baselines3.dsrl.rfs_dsrl import compose_rfs_action


class TinyChunkEnv(Env):
    observation_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
    action_space = spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(3, dtype=np.float32), {}

    def step(self, action):
        return (
            np.zeros(3, dtype=np.float32),
            float(np.asarray(action).sum()),
            False,
            False,
            {},
        )


class IdentityChunkDecoder(th.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0
        self.gain = th.nn.Parameter(th.ones(()))

    def __call__(self, observations, noise, return_numpy=False):
        del observations
        self.calls += 1
        decoded = (noise * self.gain).clamp(-1.0, 1.0)
        return decoded.cpu().numpy() if return_numpy else decoded


def make_model(**kwargs):
    decoder = kwargs.pop("diffusion_policy", IdentityChunkDecoder())
    residual_scale = kwargs.pop("residual_scale", 0.25)
    ent_coef = kwargs.pop("ent_coef", 0.01)
    model = RFSDSRL(
        "MlpPolicy",
        TinyChunkEnv(),
        learning_rate=3e-4,
        buffer_size=64,
        learning_starts=0,
        batch_size=4,
        train_freq=1,
        gradient_steps=1,
        ent_coef=ent_coef,
        target_entropy=0.0,
        device="cpu",
        policy_kwargs={"net_arch": [16, 16]},
        diffusion_policy=decoder,
        diffusion_act_dim=(2, 2),
        noise_critic_grad_steps=1,
        residual_scale=residual_scale,
        **kwargs,
    )
    model.set_logger(configure(folder=None, format_strings=[]))
    return model, decoder


def populate_replay(model, count=8):
    for index in range(count):
        observation = np.full((1, 3), index / count, dtype=np.float32)
        next_observation = np.full((1, 3), (index + 1) / count, dtype=np.float32)
        action = np.zeros((1, 4), dtype=np.float32)
        model.replay_buffer.add(
            observation,
            next_observation,
            action,
            np.ones(1, dtype=np.float32),
            np.zeros(1, dtype=np.float32),
            [{}],
        )


def test_rfs_is_a_parallel_algorithm_and_does_not_replace_dsrl():
    decoder = IdentityChunkDecoder()
    dsrl = DSRL(
        "MlpPolicy",
        TinyChunkEnv(),
        device="cpu",
        policy_kwargs={"net_arch": [8]},
        diffusion_policy=decoder,
        diffusion_act_dim=(2, 2),
    )
    rfs, _ = make_model()

    assert type(dsrl) is DSRL
    assert not hasattr(dsrl, "modulation_policy")
    assert rfs.modulation_action_space.shape == (8,)
    assert rfs.actor.action_space.shape == (8,)
    assert rfs.critic.action_space.shape == (4,)
    assert rfs.critic_modulation.action_space.shape == (8,)


def test_original_dsrl_training_path_still_runs():
    decoder = IdentityChunkDecoder()
    model = DSRL(
        "MlpPolicy",
        TinyChunkEnv(),
        learning_rate=3e-4,
        buffer_size=64,
        batch_size=4,
        ent_coef=0.01,
        target_entropy=0.0,
        device="cpu",
        policy_kwargs={"net_arch": [16, 16]},
        diffusion_policy=decoder,
        diffusion_act_dim=(2, 2),
        noise_critic_grad_steps=1,
    )
    model.set_logger(configure(folder=None, format_strings=[]))
    populate_replay(model)

    model.train(gradient_steps=1, batch_size=4)

    assert decoder.calls == 2
    assert not hasattr(model, "modulation_policy")


def test_rfs_composes_after_decoder_and_clips():
    base = th.tensor([[0.9, -0.9]])
    residual = th.tensor([[1.0, -1.0]])
    result = compose_rfs_action(
        base,
        residual,
        residual_scale=0.25,
        action_low=th.tensor([[-1.0, -1.0]]),
        action_high=th.tensor([[1.0, 1.0]]),
    )
    th.testing.assert_close(result, th.tensor([[1.0, -1.0]]))


def test_residual_head_starts_near_zero_without_changing_joint_shape():
    model, _ = make_model(residual_log_std_init=-6.0)
    observations = th.zeros(5, 3)
    mean, log_std, _ = model.actor.get_action_dist_params(observations)

    th.testing.assert_close(mean[:, 4:], th.zeros(5, 4))
    th.testing.assert_close(log_std[:, 4:], th.full((5, 4), -6.0))
    assert mean[:, :4].shape == (5, 4)


def test_one_training_update_is_finite_and_uses_two_sampler_calls():
    model, decoder = make_model()
    populate_replay(model)
    decoder.calls = 0
    decoder_gain = decoder.gain.detach().clone()
    actor_before = [parameter.detach().clone() for parameter in model.actor.parameters()]
    critic_before = [parameter.detach().clone() for parameter in model.critic.parameters()]
    modulation_before = [
        parameter.detach().clone() for parameter in model.critic_modulation.parameters()
    ]

    model.train(gradient_steps=1, batch_size=4)

    assert decoder.calls == 2  # TD target and modulation-critic distillation.
    assert any(
        not th.equal(before, after)
        for before, after in zip(actor_before, model.actor.parameters())
    )
    assert any(
        not th.equal(before, after)
        for before, after in zip(critic_before, model.critic.parameters())
    )
    assert any(
        not th.equal(before, after)
        for before, after in zip(modulation_before, model.critic_modulation.parameters())
    )
    th.testing.assert_close(decoder.gain, decoder_gain)
    assert decoder.gain.grad is None
    for module in (model.actor, model.critic, model.critic_modulation):
        assert all(th.isfinite(parameter).all() for parameter in module.parameters())


def test_predict_executes_base_plus_residual_in_action_space():
    model, decoder = make_model(residual_scale=0.5)
    observations = np.zeros((3, 3), dtype=np.float32)
    actions, state = model.predict_diffused(observations, deterministic=True)

    assert state is None
    assert actions.shape == (3, 4)
    assert decoder.calls == 1
    assert np.all(actions >= -1.0)
    assert np.all(actions <= 1.0)


def test_checkpoint_round_trip_preserves_joint_policy(tmp_path):
    model, _ = make_model(ent_coef="auto_0.1")
    populate_replay(model)
    model.train(gradient_steps=1, batch_size=4)
    checkpoint = tmp_path / "rfs_dsrl.zip"
    model.save(checkpoint)

    restored = RFSDSRL.load(checkpoint, env=TinyChunkEnv(), device="cpu")
    assert restored.modulation_action_space.shape == (8,)
    assert restored.residual_scale == pytest.approx(model.residual_scale)
    th.testing.assert_close(restored.log_ent_coef, model.log_ent_coef)
    assert len(restored.actor.optimizer.state) == len(model.actor.optimizer.state) > 0
    assert (
        len(restored.critic_modulation.optimizer.state)
        == len(model.critic_modulation.optimizer.state)
        > 0
    )

    observation = np.zeros((2, 3), dtype=np.float32)
    expected, _ = model.predict_diffused(observation, deterministic=True)
    actual, _ = restored.predict_diffused(observation, deterministic=True)
    np.testing.assert_allclose(actual, expected, atol=1e-6)


def test_rollout_buffer_stores_the_composed_executed_action():
    model, _ = make_model()
    model._last_obs = np.zeros((1, 3), dtype=np.float32)
    model.num_timesteps = 1

    action, buffer_action = model._sample_action(learning_starts=0, n_envs=1)

    assert action.shape == (1, 4)
    np.testing.assert_allclose(buffer_action, action)
