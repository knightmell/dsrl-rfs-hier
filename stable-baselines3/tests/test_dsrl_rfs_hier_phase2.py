import copy

import numpy as np
import pytest
import torch as th

from stable_baselines3.dsrl.dsrl import DSRL
from stable_baselines3.dsrl.hierarchical_rfs_dsrl import (
    DEPRECATED_CHECKPOINT_MESSAGE,
    HierarchicalRFSDSRL,
    _LegacyLoadableDSRL,
)
from tests.three_critic_test_utils import (
    POLICY_KWARGS,
    IdentityChunkDecoder,
    TinyChunkEnv,
    make_model,
)


def make_legacy(decoder=None):
    return DSRL(
        "MlpPolicy",
        TinyChunkEnv(),
        learning_rate=3e-4,
        buffer_size=32,
        batch_size=2,
        ent_coef="auto_0.2",
        target_entropy=0.0,
        device="cpu",
        policy_kwargs=POLICY_KWARGS,
        diffusion_policy=decoder or IdentityChunkDecoder(),
        diffusion_act_dim=(2, 2),
        noise_critic_grad_steps=1,
        critic_backup_combine_type="min",
    )


def assert_state_equal(left, right):
    assert set(left) == set(right)
    for key in left:
        th.testing.assert_close(left[key], right[key], rtol=0, atol=0)


def test_legacy_load_shell_requires_dimensions_outside_formal_load():
    with pytest.raises(ValueError, match="diffusion_act_dim"):
        _LegacyLoadableDSRL("MlpPolicy", TinyChunkEnv())


def test_legacy_migration_maps_three_critics_and_ignores_old_target():
    decoder = IdentityChunkDecoder()
    legacy = make_legacy(decoder)
    with th.no_grad():
        for parameter in legacy.actor.parameters():
            parameter.add_(0.01)
        for parameter in legacy.critic.parameters():
            parameter.add_(0.02)
        for parameter in legacy.critic_noise.parameters():
            parameter.add_(0.03)
        for parameter in legacy.critic_target.parameters():
            parameter.add_(9.0)
        legacy.log_ent_coef.fill_(-1.7)
    model, _ = make_model(
        decoder=decoder,
        schedule_profile="legacy_dsrl_warmstart_5m",
        phase_b_steps=0,
        phase_r_steps=8,
        beta_ramp_steps=4,
    )
    model._initialize_from_legacy_model(legacy)

    assert_state_equal(model.actor.state_dict(), legacy.actor.state_dict())
    assert_state_equal(model.reference_noise_actor.state_dict(), legacy.actor.state_dict())
    assert_state_equal(model.qa_base.state_dict(), legacy.critic.state_dict())
    assert_state_equal(model.qa_base_target.state_dict(), legacy.critic.state_dict())
    assert_state_equal(model.qw_base.state_dict(), legacy.critic_noise.state_dict())
    assert_state_equal(model.qa_joint.state_dict(), model.qa_base.state_dict())
    assert_state_equal(model.qa_joint_target.state_dict(), model.qa_joint.state_dict())
    assert_state_equal(model.residual_actor_target.state_dict(), model.residual_actor.state_dict())
    assert th.count_nonzero(model.residual_actor.output_layer.weight) == 0
    assert th.count_nonzero(model.residual_actor.output_layer.bias) == 0
    th.testing.assert_close(model.log_ent_coef, legacy.log_ent_coef, rtol=0, atol=0)
    optimizers = (
        model.actor.optimizer,
        model.qa_base_optimizer,
        model.qw_base_optimizer,
        model.qa_joint_optimizer,
        model.residual_actor_optimizer,
        model.ent_coef_optimizer,
    )
    assert all(len(optimizer.state) == 0 for optimizer in optimizers if optimizer is not None)


def test_zero_residual_migration_action_and_qw_parity():
    decoder = IdentityChunkDecoder()
    legacy = make_legacy(decoder)
    model, _ = make_model(
        decoder=decoder,
        schedule_profile="legacy_dsrl_warmstart_5m",
        phase_b_steps=0,
        phase_r_steps=8,
        beta_ramp_steps=4,
    )
    model._initialize_from_legacy_model(legacy)
    observation = th.randn(5, 3)
    noise = th.empty(5, 4).uniform_(-0.9, 0.9)
    with th.no_grad():
        generated = model._generate_hierarchical_action(
            observation, noise, zero_residual=True, beta=0.1
        )
        decoder_input = th.as_tensor(
            legacy.policy.unscale_action(noise.numpy())
        ).reshape(5, 2, 2)
        expected_action = decoder(observation, decoder_input).reshape(5, 4)
        legacy_qw = legacy.critic_noise(observation, noise)
        actual_qw = model.qw_base(observation, noise)
    th.testing.assert_close(generated.action_exec, expected_action, atol=1e-6, rtol=0)
    for actual, expected in zip(actual_qw, legacy_qw):
        th.testing.assert_close(actual, expected, atol=1e-6, rtol=0)


def test_save_load_roundtrip_restores_all_online_targets_and_counters(tmp_path):
    model, decoder = make_model()
    with th.no_grad():
        model.residual_actor.output_layer.bias.fill_(0.2)
        model.residual_actor_target.load_state_dict(model.residual_actor.state_dict())
    model.qa_base_optimizer_steps = 17
    model.qa_joint_optimizer_steps = 11
    model.qw_base_optimizer_steps = 9
    model.noise_policy_version = 4
    model.residual_policy_version = 3
    model.next_episode_id = 123
    model._allocate_unassigned_lanes()
    lane_state = copy.deepcopy(model._lane_rng.bit_generator.state)
    path = tmp_path / "three_critic.zip"
    model.save(path)
    restored = HierarchicalRFSDSRL.load(
        path,
        env=TinyChunkEnv(),
        device="cpu",
        custom_objects={"diffusion_policy": decoder},
    )
    for name in (
        "qa_base",
        "qa_base_target",
        "qw_base",
        "qa_joint",
        "qa_joint_target",
        "residual_actor",
        "residual_actor_target",
        "reference_noise_actor",
    ):
        assert_state_equal(
            getattr(restored, name).state_dict(), getattr(model, name).state_dict()
        )
    assert restored.qa_base_optimizer_steps == 17
    assert restored.qa_joint_optimizer_steps == 11
    assert restored.qw_base_optimizer_steps == 9
    assert restored.noise_policy_version == 4
    assert restored.residual_policy_version == 3
    assert restored.next_episode_id == model.next_episode_id
    assert restored._lane_rng.bit_generator.state == lane_state
    observation = np.zeros(3, np.float32)
    expected, _ = model.predict(observation, deterministic=True)
    actual, _ = restored.predict(observation, deterministic=True)
    np.testing.assert_allclose(actual, expected, atol=1e-7, rtol=0)


def test_deprecated_joint_qm_checkpoint_is_rejected(tmp_path):
    model, decoder = make_model()
    model.modulation_action_space = "deprecated-marker"
    path = tmp_path / "old_qm.zip"
    model.save(path)
    with pytest.raises(ValueError, match="deprecated joint QA/QM"):
        HierarchicalRFSDSRL.load(
            path,
            env=TinyChunkEnv(),
            device="cpu",
            custom_objects={"diffusion_policy": decoder},
        )
    assert "deprecated joint QA/QM" in DEPRECATED_CHECKPOINT_MESSAGE
