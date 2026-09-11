"""Production-config contracts for the paired Walker K4 R diagnosis."""

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
        job_name="test_walker_k4_additive_res_configs",
        version_base=None,
    ):
        config = compose(config_name=config_name, overrides=["seed=1"])
    OmegaConf.resolve(config)
    return config


def test_additive_res_config_keeps_k4_noclip_and_stops_at_750k():
    config = _compose("p6_walker_k4_additive_res_750k")

    assert config.train.rfs_hier_schedule_profile == (
        "fresh_frozen_ddim_2p5m_additive_res"
    )
    assert config.train.rfs_hier_phase_b_steps == 500_000
    assert config.train.rfs_hier_phase_r_steps == 2_000_000
    assert config.train.rfs_hier_qw_teacher_source == "gaussian"
    assert config.train.rfs_hier_qw_candidates_per_state == 4
    assert config.train.rfs_hier_qw_state_batch_size == 256
    assert config.train.rfs_hier_noise_actor_gradient_clipping is False
    assert config.p6.stop_after_chunk_transitions == 750_000


def test_base_continue_config_stays_in_base_and_stops_at_750k():
    config = _compose("p6_walker_k4_base_continue_750k")

    assert config.train.rfs_hier_schedule_profile == (
        "fresh_frozen_ddim_2p5m_base_continue"
    )
    assert config.train.rfs_hier_phase_b_steps == 2_000_000
    assert config.train.rfs_hier_phase_r_steps == 500_000
    assert config.train.rfs_hier_qw_teacher_source == "gaussian"
    assert config.train.rfs_hier_qw_candidates_per_state == 4
    assert config.train.rfs_hier_qw_state_batch_size == 256
    assert config.train.rfs_hier_noise_actor_gradient_clipping is False
    assert config.p6.stop_after_chunk_transitions == 750_000


def test_new_k4_profiles_pass_static_preflight_and_use_fresh_init():
    for config_name in (
        "p6_walker_k4_additive_res_750k",
        "p6_walker_k4_base_continue_750k",
    ):
        config = _compose(config_name)
        manifest = static_preflight(
            config,
            PROJECT_ROOT,
            algorithm=HIERARCHY_ALGORITHM,
        )
        assert manifest["hierarchy_schedule"]["schedule_profile"] == (
            config.train.rfs_hier_schedule_profile
        )
        assert _hierarchy_init_mode(config) == "fresh"


def test_fast_runtime_overlay_preserves_k4_query_budget_and_reduces_io_cadence():
    with initialize_config_dir(
        config_dir=str(CONFIG_DIR),
        job_name="test_fast_runtime_overlay",
        version_base=None,
    ):
        config = compose(
            config_name="p6_walker_300k_800k_additive_res_k4_noclip",
            overrides=["seed=1", "+runtime=fast"],
        )
    OmegaConf.resolve(config)

    assert config.train.rfs_hier_qw_candidates_per_state == 4
    assert config.train.rfs_hier_qw_state_batch_size == 256
    assert config.train.rfs_hier_qw_teacher_microbatch_size == 1024
    assert config.train.rfs_hier_runtime_contract_checks is False
    assert config.p6.online_eval_interval_chunk_transitions == 100_000
    assert config.p6.model_checkpoint_interval_chunk_transitions == 100_000
    assert config.p6.replay_checkpoint_interval_chunk_transitions == 200_000
