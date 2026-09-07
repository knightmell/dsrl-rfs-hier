"""Contracts for the current 600k full/control pair."""

from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from p6_preflight import HIERARCHY_ALGORITHM, static_preflight
from p6_train import _hierarchy_init_mode


CONFIG_DIR = Path(__file__).resolve().parents[1] / "cfg" / "gym"
PROJECT_ROOT = CONFIG_DIR.parents[1]
OmegaConf.register_new_resolver("eval", eval, replace=True)


def _compose(config_name):
    with initialize_config_dir(
        config_dir=str(CONFIG_DIR),
        job_name="test_current_600k_control_configs",
        version_base=None,
    ):
        config = compose(config_name=config_name, overrides=["seed=1"])
    OmegaConf.resolve(config)
    return config


def test_current_600k_full_is_300k_base_plus_300k_residual():
    config = _compose("p6_halfcheetah_fresh_600k_cotrain_k4_noclip")

    assert config.total_timesteps == 600_000
    assert config.train.rfs_hier_phase_b_steps == 300_000
    assert config.train.rfs_hier_phase_r_steps == 300_000
    assert config.train.rfs_hier_qw_teacher_source == "gaussian"
    assert config.train.rfs_hier_qw_candidates_per_state == 4
    assert config.train.rfs_hier_qw_state_batch_size == 256
    assert config.train.rfs_hier_noise_actor_gradient_clipping is False


def test_current_600k_base_control_stays_in_base_phase():
    config = _compose("p6_halfcheetah_fresh_600k_base_control_k4_noclip")

    assert config.total_timesteps == 2_500_000
    assert config.p6.stop_after_chunk_transitions == 600_000
    assert config.train.rfs_hier_schedule_profile == (
        "fresh_frozen_ddim_2p5m_base_continue"
    )
    assert config.train.rfs_hier_phase_b_steps == 2_000_000
    assert config.train.rfs_hier_phase_r_steps == 500_000
    assert config.train.rfs_hier_qw_teacher_source == "gaussian"
    assert config.train.rfs_hier_qw_candidates_per_state == 4
    assert config.train.rfs_hier_qw_state_batch_size == 256
    assert config.train.rfs_hier_noise_actor_gradient_clipping is False

    manifest = static_preflight(
        config,
        PROJECT_ROOT,
        algorithm=HIERARCHY_ALGORITHM,
    )
    assert manifest["hierarchy_schedule"]["phase_b_steps"] == 2_000_000
    assert _hierarchy_init_mode(config) == "fresh"


def test_hopper_base_control_uses_the_same_contract():
    config = _compose("p6_hopper_fresh_600k_base_control_k4_noclip")

    assert config.p6.stop_after_chunk_transitions == 600_000
    assert config.train.rfs_hier_qw_teacher_source == "gaussian"
    assert config.train.rfs_hier_qw_candidates_per_state == 4
    assert config.train.rfs_hier_qw_state_batch_size == 256
    assert config.train.rfs_hier_noise_actor_gradient_clipping is False
