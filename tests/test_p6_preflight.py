import ast
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import gym
import numpy as np
import pytest
from gymnasium import spaces
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from env_utils import ActionChunkWrapper, ObservationWrapperGym
from p6_preflight import (
    ARCHITECTURE_VERSION,
    CONTROL_ALGORITHM,
    FROZEN_NOISE_ALGORITHM,
    HIERARCHY_ALGORITHM,
    HIERARCHY_REPLAY_SCHEMA_VERSION,
    finalize_loaded_model_preflight,
    resolve_seed_plan,
    run_preflight,
    static_preflight,
    validate_observation_dimension,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "cfg" / "gym"
FIVE_M_CHECKPOINT = (
    "./logs/gym-dsrl/"
    "gym_hopper_dsrl_2026-07-23_21-21-02_1/"
    "2026-07-23_21-21-02_1/checkpoint/ft_policy_5000000_steps.zip"
)

OmegaConf.register_new_resolver("eval", eval, replace=True)


def compose_hopper(overrides=None):
    with initialize_config_dir(
        config_dir=str(CONFIG_DIR),
        job_name="test_p6_preflight",
        version_base=None,
    ):
        cfg = compose(
            config_name="dsrl_hopper",
            overrides=overrides or [],
        )
    OmegaConf.resolve(cfg)
    return cfg


def compose_p6(overrides):
    with initialize_config_dir(
        config_dir=str(CONFIG_DIR),
        job_name="test_p6_preflight",
        version_base=None,
    ):
        cfg = compose(
            config_name="p6_hopper",
            overrides=overrides,
        )
    OmegaConf.resolve(cfg)
    return cfg


def p6_config(
    *,
    algorithm=HIERARCHY_ALGORITHM,
    name_algorithm=None,
    extra_overrides=None,
):
    name_algorithm = name_algorithm or algorithm
    overrides = [
        f"algorithm={'dsrl_na' if algorithm == CONTROL_ALGORITHM else algorithm}",
        f"p6.algorithm_label={name_algorithm}",
        "total_timesteps=100000",
        "train.rfs_hier_phase_r_steps=100000",
        "train.rfs_hier_beta_ramp_steps=10000",
        f"rfs_hier_legacy_checkpoint_path={FIVE_M_CHECKPOINT}",
        f"name=init_5m_{name_algorithm}_seed1_100000chunks",
    ]
    overrides.extend(extra_overrides or [])
    return compose_p6(overrides)


def compose_fresh_hopper():
    """Compose the real fresh 2.5M config, shrunk to a 100k test budget while
    preserving the fresh profile and its 50:50 Phase B:R ratio."""
    with initialize_config_dir(
        config_dir=str(CONFIG_DIR),
        job_name="test_p6_preflight",
        version_base=None,
    ):
        cfg = compose(
            config_name="p6_hopper_fresh_2p5m",
            overrides=[
                "total_timesteps=100000",
                "train.rfs_hier_phase_b_steps=50000",
                "train.rfs_hier_phase_r_steps=50000",
            ],
        )
    OmegaConf.resolve(cfg)
    return cfg


def compose_cotrain_hopper(config_name="p6_hopper_fresh_2p5m_cotrain"):
    """Compose a real fresh co-training 2.5M config (version A or B), shrunk to
    a 100k test budget.  The profile's hold+ramp (50k each) must shrink with R
    so the schedule stays valid (hold + ramp <= R)."""
    with initialize_config_dir(
        config_dir=str(CONFIG_DIR),
        job_name="test_p6_preflight",
        version_base=None,
    ):
        cfg = compose(
            config_name=config_name,
            overrides=[
                "total_timesteps=100000",
                "train.rfs_hier_phase_b_steps=20000",
                "train.rfs_hier_phase_r_steps=80000",
                "train.rfs_hier_beta_ramp_steps=20000",
                "train.rfs_hier_beta_hold_steps=20000",
            ],
        )
    OmegaConf.resolve(cfg)
    return cfg


def compose_cotrain_locomotion(config_name):
    """Compose an unshortened migrated locomotion production config."""
    with initialize_config_dir(
        config_dir=str(CONFIG_DIR),
        job_name="test_p6_locomotion_migration",
        version_base=None,
    ):
        cfg = compose(config_name=config_name, overrides=["seed=1"])
    OmegaConf.resolve(cfg)
    return cfg


def make_execution_env(low=-1.0, high=1.0, shape=(12,)):
    return SimpleNamespace(
        action_space=spaces.Box(
            low=low,
            high=high,
            shape=shape,
            dtype=np.float32,
        )
    )


def test_real_hopper_artifacts_seed_plan_and_manifest_pass(tmp_path):
    cfg = p6_config()
    seed_plan = resolve_seed_plan(cfg)

    assert seed_plan == {
        "train_env_seed": 1001,
        "eval_env_seed": 2001,
        "prefill_env_seed": 3001,
        "prefill_policy_seed": 4001,
        "eval_seed_set": list(range(10000, 10100)),
    }

    static_manifest = static_preflight(
        cfg,
        PROJECT_ROOT,
        algorithm=HIERARCHY_ALGORITHM,
    )
    manifest_path = tmp_path / "run_manifest.json"
    manifest = run_preflight(
        cfg,
        make_execution_env(),
        PROJECT_ROOT,
        manifest_path,
        algorithm=HIERARCHY_ALGORITHM,
        static_manifest=static_manifest,
    )

    assert manifest["init_checkpoint_id"] == "init_5m"
    assert manifest["init_checkpoint_sha256"] == (
        "f6deae068822cd9bc29405e493600c23a"
        "ed0d0273b9068f8e16b015c7406dc8b"
    )
    assert manifest["frozen_ddim_sha256"] == (
        "9a5839d3d172d1e24b5bed0831d49bb3"
        "116a7e62fafa223c57c322f4bc7e9121"
    )
    assert manifest["normalization_sha256"] == (
        "d05b2943bc39772f7f770dfc0a4df2a"
        "9f205cb1e05ae74b585cd0dd382a0742e"
    )
    assert manifest["execution_action_shape"] == [12]
    assert manifest["execution_action_low"] == [-1.0] * 12
    assert manifest["execution_action_high"] == [1.0] * 12
    assert manifest["chunk_budget"] == 100_000
    assert manifest["nominal_primitive_budget"] == 400_000
    assert manifest["actual_primitive_budget_upper_bound"] == 400_000
    assert manifest["n_envs"] == 10
    assert manifest["prefill_source"] == "warmstart_dsrl"
    assert manifest["prefill_action_policy"] == (
        "shared_tagged_warmstart_dsrl_behavior"
    )
    assert (
        manifest["prefill_action_policy"]
        == manifest["hierarchy_schedule"]["prefill_action_policy"]
        == manifest["config_contract"]["prefill_action_policy"]
    )


def test_fresh_manifest_top_level_prefill_action_policy_is_gaussian_prior(tmp_path):
    cfg = compose_fresh_hopper()
    static_manifest = static_preflight(
        cfg,
        PROJECT_ROOT,
        algorithm=HIERARCHY_ALGORITHM,
    )
    manifest_path = tmp_path / "run_manifest.json"
    manifest = run_preflight(
        cfg,
        make_execution_env(),
        PROJECT_ROOT,
        manifest_path,
        algorithm=HIERARCHY_ALGORITHM,
        static_manifest=static_manifest,
    )

    assert manifest["init_checkpoint_id"] == "fresh_frozen_ddim"
    assert manifest["init_checkpoint_path"] is None
    assert manifest["init_checkpoint_sha256"] is None
    assert manifest["prefill_source"] == "fresh_frozen_ddim"
    expected_fresh = "fresh_frozen_ddim_gaussian_prior_behavior"
    assert manifest["prefill_action_policy"] == expected_fresh
    # All three manifest copies must agree (regression for the hard-coded
    # warm-start label at the top level).
    assert (
        manifest["prefill_action_policy"]
        == manifest["hierarchy_schedule"]["prefill_action_policy"]
        == manifest["config_contract"]["prefill_action_policy"]
        == expected_fresh
    )
    assert manifest["hierarchy_prefill_residual_mode"] == (
        "exact_zero_residual_tagged_base_projection"
    )
    assert manifest["prefill_transition_count"] == 20_010
    assert manifest["prefill_hash"] is None
    assert manifest["prefill_status"] == "pending_tagged_prefill"
    assert manifest["prefill_artifact_path"].endswith("_tagged_v1.npz")
    assert manifest["run_id"].startswith("p6-")
    assert len(manifest["run_id"]) == len("p6-") + 32
    assert manifest["config_contract"]["training"] == manifest[
        "training_contract"
    ]
    assert (
        manifest["config_contract"]["source_state_sha256"]
        == manifest["source_state_sha256"]
    )
    assert set(manifest["source_provenance"]) == {
        "outer",
        "stable_baselines3",
        "dppo",
    }
    assert manifest["critic_backup_combine_type"] == "min"
    assert manifest["action_chunk_termination_semantics"] == (
        "early_break_on_done"
    )
    assert manifest["stable_baselines3_submodule"]["commit"] == (
        manifest["stable_baselines3_submodule"]["recorded_commit"]
    )
    assert manifest["dppo_submodule"]["commit"] == (
        manifest["dppo_submodule"]["recorded_commit"]
    )
    expected_dirty_entries = subprocess.run(
        [
            "git",
            "-C",
            str(PROJECT_ROOT),
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.rstrip("\n").splitlines()
    assert manifest["outer_repository"]["dirty_entries"] == expected_dirty_entries
    assert json.loads(manifest_path.read_text()) == manifest


@pytest.mark.parametrize(
    (
        "config_name",
        "environment_name",
        "frozen_ddim_sha256",
        "normalization_sha256",
        "expected_observation_dimension",
    ),
    (
        (
            "p6_halfcheetah_fresh_2p5m_cotrain",
            "halfcheetah-medium-v2",
            "ec1169d29e2d0e006f0c2477ee8bc976b2f9b04b2c209df5bce9f36dd42596b8",
            "1bc71a75b5cde71bb699cdbb4d169de9309a714b87c08bf88bd9c6c10a963e2a",
            17,
        ),
        (
            "p6_walker_fresh_2p5m_cotrain",
            "walker2d-medium-v2",
            "1a650d5d882c78cfef1d753c532c0bdbced054738b0076efe6ce29d088681fa0",
            "7d103b793b591081947fca807ae13c2752bd6964621374101f5a89a19346ae16",
            17,
        ),
    ),
)
def test_migrated_locomotion_configs_bind_artifacts_bounds_and_schedule(
    tmp_path,
    config_name,
    environment_name,
    frozen_ddim_sha256,
    normalization_sha256,
    expected_observation_dimension,
):
    cfg = compose_cotrain_locomotion(config_name)
    static_manifest = static_preflight(
        cfg,
        PROJECT_ROOT,
        algorithm=HIERARCHY_ALGORITHM,
    )
    manifest = run_preflight(
        cfg,
        make_execution_env(shape=(24,)),
        PROJECT_ROOT,
        tmp_path / f"{environment_name}_manifest.json",
        algorithm=HIERARCHY_ALGORITHM,
        static_manifest=static_manifest,
    )

    assert manifest["config_contract"]["env_name"] == environment_name
    assert manifest["action_dimension"] == 6
    assert manifest["action_chunk"] == 4
    assert manifest["execution_action_shape"] == [24]
    assert manifest["execution_action_low"] == [-1.0] * 24
    assert manifest["execution_action_high"] == [1.0] * 24
    assert manifest["frozen_ddim_sha256"] == frozen_ddim_sha256
    assert manifest["normalization_sha256"] == normalization_sha256
    assert manifest["chunk_budget"] == 2_500_000
    assert manifest["hierarchy_schedule"]["phase_b_steps"] == 500_000
    assert manifest["hierarchy_schedule"]["phase_r_steps"] == 2_000_000
    assert manifest["hierarchy_schedule"]["phase_j_enabled"] is False
    assert environment_name in manifest["prefill_artifact_path"]
    # Round-2 C4 fix: run_preflight records the ground-truth observation
    # dimension (from the per-env normalization file), independent of the
    # cfg-derived wrapped obs space.
    assert manifest["observation_dimension"] == expected_observation_dimension


@pytest.mark.parametrize(
    "config_name, expected_observation_dimension",
    (
        ("p6_halfcheetah_fresh_2p5m_cotrain", 17),
        ("p6_walker_fresh_2p5m_cotrain", 17),
        ("p6_hopper_fresh_2p5m_cotrain", 11),
    ),
)
def test_validate_observation_dimension_uses_normalization_ground_truth(
    config_name,
    expected_observation_dimension,
):
    # The wrapped/vectorized obs space is derived from cfg.obs_dim and can never
    # disagree with it, so the only non-circular source of truth is the per-env
    # normalization file (obs_min indexed by the raw per-step observation).
    cfg = compose_cotrain_locomotion(config_name)

    result = validate_observation_dimension(cfg, env=None)

    assert result == {
        "observation_dimension": expected_observation_dimension,
    }


def test_validate_observation_dimension_rejects_stale_obs_dim():
    # A stale cfg.obs_dim (e.g. hopper's 11 shipped with an HC config) must be
    # caught against the real normalization file, not silently recorded.
    cfg = compose_cotrain_locomotion("p6_halfcheetah_fresh_2p5m_cotrain")
    cfg.obs_dim = 11

    with pytest.raises(ValueError, match="Observation dimension mismatch"):
        validate_observation_dimension(cfg, env=None)


def test_validate_observation_dimension_skips_when_normalization_unavailable():
    # Defensive degradation: no reachable normalization file -> no assertion,
    # no crash, and the manifest key records None.
    cfg = compose_cotrain_locomotion("p6_halfcheetah_fresh_2p5m_cotrain")
    cfg.normalization_path = "./logs/does-not-exist-normalization.npz"

    assert validate_observation_dimension(cfg, env=None) == {
        "observation_dimension": None,
    }


def test_migrated_prefill_artifacts_are_environment_disjoint():
    halfcheetah = compose_cotrain_locomotion(
        "p6_halfcheetah_fresh_2p5m_cotrain"
    )
    walker = compose_cotrain_locomotion("p6_walker_fresh_2p5m_cotrain")

    halfcheetah_path = str(halfcheetah.p6.prefill_artifact_path)
    walker_path = str(walker.p6.prefill_artifact_path)
    assert halfcheetah_path != walker_path
    assert "halfcheetah-medium-v2" in halfcheetah_path
    assert "walker2d-medium-v2" in walker_path


def test_cotrain_manifest_update_profiles_and_flags(tmp_path):
    cfg = compose_cotrain_hopper()
    static_manifest = static_preflight(
        cfg,
        PROJECT_ROOT,
        algorithm=HIERARCHY_ALGORITHM,
    )
    manifest_path = tmp_path / "run_manifest.json"
    manifest = run_preflight(
        cfg,
        make_execution_env(),
        PROJECT_ROOT,
        manifest_path,
        algorithm=HIERARCHY_ALGORITHM,
        static_manifest=static_manifest,
    )

    schedule = manifest["hierarchy_schedule"]
    assert schedule["schedule_profile"] == "fresh_frozen_ddim_2p5m_cotrain"
    assert schedule["phase_b_steps"] == 20_000
    assert schedule["phase_r_steps"] == 80_000
    assert schedule["phase_j_steps"] == 0
    assert schedule["phase_j_enabled"] is False
    # Frozen constraint values are untouched by the co-training deviations.
    assert schedule["beta_target"] == 0.1
    assert schedule["base_lane_probability"] == 0.5
    # Co-training overrides reach the contract.
    assert schedule["beta_ramp_steps"] == 20_000
    assert schedule["beta_hold_steps"] == 20_000
    assert schedule["beta_floor"] == 0.02

    # Fresh gate still recognizes the co-training profile.
    assert manifest["init_checkpoint_id"] == "fresh_frozen_ddim"
    assert manifest["init_checkpoint_path"] is None
    assert manifest["prefill_source"] == "fresh_frozen_ddim"
    assert manifest["prefill_action_policy"] == (
        "fresh_frozen_ddim_gaussian_prior_behavior"
    )

    # Per-profile update counts: B shadow includes QA_joint; R co-trains
    # noise/alpha alongside the residual.  JOINT stays the frozen default.
    profiles = schedule["update_profiles"]
    assert profiles["base"] == {
        "qa_base": 20,
        "qa_joint": 10,
        "qw_base": 10,
        "noise_actor": 20,
        "alpha": 20,
        "residual_actor": 0,
    }
    assert profiles["residual"] == {
        "qa_base": 5,
        "qa_joint": 5,
        "qw_base": 2,
        "noise_actor": 1,
        "alpha": 1,
        "residual_actor": 1,
    }
    assert profiles["joint"] == {
        "qa_base": 10,
        "qa_joint": 10,
        "qw_base": 5,
        "noise_actor": 1,
        "alpha": 1,
        "residual_actor": 1,
    }

    # Co-training model-level flags reach the contract.
    assert schedule["qa_joint_shadow_in_b"] is True
    assert schedule["cross_lane_ratio"] == 0.25
    assert schedule["qa_base_cross_lane"] is False
    assert schedule["residual_exploration_std"] == 0.02
    assert json.loads(manifest_path.read_text()) == manifest


def test_cotrain_b_contract_marks_qa_base_cross_lane_true(tmp_path):
    cfg = compose_cotrain_hopper(
        config_name="p6_hopper_fresh_2p5m_cotrain_b",
    )
    static_manifest = static_preflight(
        cfg,
        PROJECT_ROOT,
        algorithm=HIERARCHY_ALGORITHM,
    )
    manifest_path = tmp_path / "run_manifest.json"
    manifest = run_preflight(
        cfg,
        make_execution_env(),
        PROJECT_ROOT,
        manifest_path,
        algorithm=HIERARCHY_ALGORITHM,
        static_manifest=static_manifest,
    )

    schedule = manifest["hierarchy_schedule"]
    assert schedule["schedule_profile"] == "fresh_frozen_ddim_2p5m_cotrain"
    assert schedule["qa_base_cross_lane"] is True
    # Everything else identical to version A.
    assert schedule["qa_joint_shadow_in_b"] is True
    assert schedule["cross_lane_ratio"] == 0.25
    assert schedule["residual_exploration_std"] == 0.02


@pytest.mark.parametrize(
    ("config_name", "expected_algorithm"),
    (
        ("p6_hopper_fresh_gate_100k", HIERARCHY_ALGORITHM),
        ("p6_hopper_fresh_gate_b_100k", HIERARCHY_ALGORITHM),
        ("p6_hopper_fresh_control_100k", CONTROL_ALGORITHM),
    ),
)
def test_gate_configs_pass_static_preflight(config_name, expected_algorithm):
    with initialize_config_dir(
        config_dir=str(CONFIG_DIR),
        job_name="test_p6_gate_configs",
        version_base=None,
    ):
        cfg = compose(
            config_name=config_name,
            overrides=["seed=1"],
        )
    OmegaConf.resolve(cfg)
    manifest = static_preflight(
        cfg,
        PROJECT_ROOT,
        algorithm=expected_algorithm,
    )
    assert manifest["chunk_budget"] == 100_000
    assert manifest["prefill_source"] == "fresh_frozen_ddim"
    assert manifest["prefill_action_policy"] == (
        "fresh_frozen_ddim_gaussian_prior_behavior"
    )
    assert manifest["init_checkpoint_id"] == "fresh_frozen_ddim"


def test_preflight_rejects_deprecated_frozen_noise_algorithm_label():
    cfg = p6_config()
    with pytest.raises(ValueError, match="Unsupported P6 algorithm label"):
        static_preflight(
            cfg,
            PROJECT_ROOT,
            algorithm=FROZEN_NOISE_ALGORITHM,
        )


def test_preflight_rejects_missing_checkpoint_and_non_explicit_pilot_env_count():
    missing_checkpoint_cfg = compose_p6(
        [
            "algorithm=dsrl_na_rfs_hier",
            "total_timesteps=100000",
            "train.rfs_hier_phase_r_steps=100000",
            "train.rfs_hier_beta_ramp_steps=10000",
            "rfs_hier_legacy_checkpoint_path=null",
            "name=init_5m_dsrl_na_rfs_hier_seed1_100000chunks",
        ]
    )
    with pytest.raises(ValueError, match="init_checkpoint must be explicitly"):
        static_preflight(
            missing_checkpoint_cfg,
            PROJECT_ROOT,
            algorithm=HIERARCHY_ALGORITHM,
        )

    default_env_cfg = p6_config(extra_overrides=["env.n_envs=4"])
    with pytest.raises(ValueError, match="explicit audited override"):
        static_preflight(
            default_env_cfg,
            PROJECT_ROOT,
            algorithm=HIERARCHY_ALGORITHM,
        )


def test_preflight_rejects_artifact_hash_run_name_and_seed_conflicts():
    wrong_hash_cfg = p6_config(
        extra_overrides=[f"p6.init_checkpoint_sha256={'0' * 64}"]
    )
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        static_preflight(
            wrong_hash_cfg,
            PROJECT_ROOT,
            algorithm=HIERARCHY_ALGORITHM,
        )

    wrong_name_cfg = p6_config(extra_overrides=["name=ambiguous_run"])
    with pytest.raises(ValueError, match="missing required tokens"):
        static_preflight(
            wrong_name_cfg,
            PROJECT_ROOT,
            algorithm=HIERARCHY_ALGORITHM,
        )

    duplicate_seed_cfg = p6_config(
        extra_overrides=["p6.eval_env_seed=1001"]
    )
    with pytest.raises(ValueError, match="pairwise distinct"):
        resolve_seed_plan(duplicate_seed_cfg)


@pytest.mark.parametrize(
    "override,match",
    (
        ("model.ddim_steps=4", "DDIM step mismatch"),
        ("model.denoised_clip_value=0.9", "Denoised clip mismatch"),
        ("act_steps=3", "Action chunk mismatch"),
        ("action_dim=2", "Action dimension mismatch"),
    ),
)
def test_preflight_rejects_sampler_or_action_layout_conflicts(override, match):
    cfg = p6_config(extra_overrides=[override])
    with pytest.raises(ValueError, match=match):
        static_preflight(
            cfg,
            PROJECT_ROOT,
            algorithm=HIERARCHY_ALGORITHM,
        )


@pytest.mark.parametrize(
    "override,match",
    (
        (
            "train.critic_backup_combine_type=mean",
            "critic_backup_combine_type='min'",
        ),
        ("train.utd=19", "20/10"),
        ("train.noise_critic_grad_steps=9", "20/10"),
    ),
)
def test_preflight_rejects_training_contract_drift(override, match):
    cfg = p6_config(extra_overrides=[override])
    with pytest.raises(ValueError, match=match):
        static_preflight(
            cfg,
            PROJECT_ROOT,
            algorithm=HIERARCHY_ALGORITHM,
        )


def test_config_contract_hash_is_deterministic_and_seed_bound():
    first = static_preflight(
        p6_config(),
        PROJECT_ROOT,
        algorithm=HIERARCHY_ALGORITHM,
    )
    second = static_preflight(
        p6_config(),
        PROJECT_ROOT,
        algorithm=HIERARCHY_ALGORITHM,
    )
    different_seed = static_preflight(
        p6_config(
            extra_overrides=[
                "seed=2",
                "name=init_5m_dsrl_na_rfs_hier_seed2_100000chunks",
            ]
        ),
        PROJECT_ROOT,
        algorithm=HIERARCHY_ALGORITHM,
    )

    assert first["config_contract_sha256"] == second["config_contract_sha256"]
    assert first["run_id"] != second["run_id"]
    assert (
        first["config_contract_sha256"]
        != different_seed["config_contract_sha256"]
    )


@pytest.mark.parametrize(
    "env,match",
    (
        (make_execution_env(shape=(3,)), "Execution action shape mismatch"),
        (make_execution_env(low=-2.0), "Execution low bounds mismatch"),
        (make_execution_env(high=2.0), "Execution high bounds mismatch"),
    ),
)
def test_preflight_rejects_execution_shape_or_bound_conflicts(tmp_path, env, match):
    cfg = p6_config()
    static_manifest = static_preflight(
        cfg,
        PROJECT_ROOT,
        algorithm=HIERARCHY_ALGORITHM,
    )
    with pytest.raises(ValueError, match=match):
        run_preflight(
            cfg,
            env,
            PROJECT_ROOT,
            tmp_path / "manifest.json",
            algorithm=HIERARCHY_ALGORITHM,
            static_manifest=static_manifest,
        )


def test_officially_loaded_model_metadata_finalizes_manifest(tmp_path):
    cfg = p6_config()
    env = make_execution_env()
    manifest_path = tmp_path / "run_manifest.json"
    run_preflight(
        cfg,
        env,
        PROJECT_ROOT,
        manifest_path,
        algorithm=HIERARCHY_ALGORITHM,
    )
    model = SimpleNamespace(
        num_timesteps=5_000_000,
        diffusion_act_chunk=4,
        diffusion_act_dim=3,
        action_space=env.action_space,
        architecture_version=ARCHITECTURE_VERSION,
        replay_schema_version=HIERARCHY_REPLAY_SCHEMA_VERSION,
    )

    manifest = finalize_loaded_model_preflight(
        cfg,
        env,
        model,
        manifest_path,
        # A real checkpoint was loaded, so its interaction-step count is
        # preserved and must match the manifest's expected steps.
        legacy_loaded=True,
    )

    assert manifest["preflight_status"] == "loaded_model_verified_prefill_pending"
    assert manifest["init_checkpoint_num_timesteps"] == 5_000_000
    assert manifest["training_start_num_timesteps"] == 5_000_000
    assert manifest["network_warmstart"] is False
    assert manifest["legacy_loaded"] is True
    assert manifest["loaded_diffusion_shape"] == [4, 3]

    model.num_timesteps = 7_500_000
    with pytest.raises(ValueError, match="step mismatch"):
        finalize_loaded_model_preflight(
            cfg,
            env,
            model,
            manifest_path,
            legacy_loaded=True,
        )

    model.num_timesteps = 0
    manifest = finalize_loaded_model_preflight(
        cfg,
        env,
        model,
        manifest_path,
        network_warmstart=True,
    )
    assert manifest["init_checkpoint_num_timesteps"] == 5_000_000
    assert manifest["training_start_num_timesteps"] == 0
    assert manifest["network_warmstart"] is True

    model.num_timesteps = 1
    with pytest.raises(ValueError, match="constructed at zero interaction steps"):
        finalize_loaded_model_preflight(
            cfg,
            env,
            model,
            manifest_path,
            network_warmstart=True,
        )


def test_finalize_fresh_accepts_zero_expected_steps(tmp_path):
    # Regression: the fresh (no-legacy-checkpoint) run configures
    # expected_init_checkpoint_steps=0, which is the semantically correct
    # manifest value.  The step validation must not reject it with a >= 1
    # requirement — the >= 1 enforcement belongs to the checkpoint-comparison
    # path that only runs when a real checkpoint was loaded (legacy_loaded).
    # A fresh run records network_warmstart=False and legacy_loaded=False.
    cfg = compose_cotrain_hopper()
    env = make_execution_env()
    manifest_path = tmp_path / "run_manifest.json"
    run_preflight(
        cfg,
        env,
        PROJECT_ROOT,
        manifest_path,
        algorithm=HIERARCHY_ALGORITHM,
    )
    model = SimpleNamespace(
        num_timesteps=0,
        diffusion_act_chunk=4,
        diffusion_act_dim=3,
        action_space=env.action_space,
        architecture_version=ARCHITECTURE_VERSION,
        replay_schema_version=HIERARCHY_REPLAY_SCHEMA_VERSION,
    )

    manifest = finalize_loaded_model_preflight(
        cfg,
        env,
        model,
        manifest_path,
    )
    assert manifest["init_checkpoint_num_timesteps"] == 0
    assert manifest["training_start_num_timesteps"] == 0
    assert manifest["network_warmstart"] is False
    assert manifest["legacy_loaded"] is False
    assert manifest["preflight_status"] == "loaded_model_verified_prefill_pending"

    # A fresh model must still begin at zero interaction steps.
    model.num_timesteps = 1
    with pytest.raises(ValueError, match="constructed at zero interaction steps"):
        finalize_loaded_model_preflight(
            cfg,
            env,
            model,
            manifest_path,
        )


def test_control_uses_the_same_audited_checkpoint_and_seed_plan():
    cfg = p6_config(
        algorithm=CONTROL_ALGORITHM,
        name_algorithm=CONTROL_ALGORITHM,
    )
    manifest = static_preflight(
        cfg,
        PROJECT_ROOT,
        algorithm=CONTROL_ALGORITHM,
    )

    assert manifest["algorithm"] == CONTROL_ALGORITHM
    assert manifest["init_checkpoint_id"] == "init_5m"
    assert manifest["train_env_seed"] == 1001
    assert manifest["prefill_policy_seed"] == 4001


def test_control_preflight_rejects_fresh_prefill_source_mislabel():
    # Control/frozen-noise always derive the warmstart prefill at runtime, so a
    # config claiming the fresh source would mislabel the manifest (prefill
    # action policy would say Gaussian prior while the actual prefill is the
    # warmstart CURRENT_ACTOR).  The gate must fail preflight.
    cfg = p6_config(
        algorithm=CONTROL_ALGORITHM,
        name_algorithm=CONTROL_ALGORITHM,
        extra_overrides=["p6.prefill_source=fresh_frozen_ddim"],
    )
    with pytest.raises(ValueError, match="requires prefill source"):
        static_preflight(cfg, PROJECT_ROOT, algorithm=CONTROL_ALGORITHM)


def test_fresh_control_preflight_uses_fresh_init_and_gaussian_prefill(tmp_path):
    # A matched flat-DSRL control with rfs_hier_legacy_checkpoint_path=null is
    # a FRESH run: init_checkpoint_id=fresh_frozen_ddim and the same
    # Gaussian-prior prefill source as the fresh hierarchy, so the prefill
    # artifact (same frozen DDIM, normalization, prefill seeds, n_envs) is the
    # identical seeded decode shared with the hierarchy gate runs.
    cfg = p6_config(
        algorithm=CONTROL_ALGORITHM,
        name_algorithm=CONTROL_ALGORITHM,
        extra_overrides=[
            "rfs_hier_legacy_checkpoint_path=null",
            "p6.prefill_source=fresh_frozen_ddim",
            "p6.init_checkpoint_id=fresh_frozen_ddim",
            "p6.expected_init_checkpoint_steps=0",
            "name=fresh_frozen_ddim_dsrl_na_control_seed1_100000chunks",
        ],
    )
    static_manifest = static_preflight(
        cfg,
        PROJECT_ROOT,
        algorithm=CONTROL_ALGORITHM,
    )
    manifest_path = tmp_path / "run_manifest.json"
    manifest = run_preflight(
        cfg,
        make_execution_env(),
        PROJECT_ROOT,
        manifest_path,
        algorithm=CONTROL_ALGORITHM,
        static_manifest=static_manifest,
    )

    assert manifest["init_checkpoint_id"] == "fresh_frozen_ddim"
    assert manifest["init_checkpoint_path"] is None
    assert manifest["prefill_source"] == "fresh_frozen_ddim"
    assert manifest["prefill_action_policy"] == (
        "fresh_frozen_ddim_gaussian_prior_behavior"
    )
    assert manifest["algorithm"] == CONTROL_ALGORITHM


def test_fresh_control_preflight_rejects_warmstart_prefill_source():
    # The fresh matched control derives the Gaussian-prior prefill at runtime;
    # a config claiming the warmstart source would mislabel the manifest.
    cfg = p6_config(
        algorithm=CONTROL_ALGORITHM,
        name_algorithm=CONTROL_ALGORITHM,
        extra_overrides=[
            "rfs_hier_legacy_checkpoint_path=null",
            "p6.prefill_source=warmstart_dsrl",
            "p6.init_checkpoint_id=fresh_frozen_ddim",
            "p6.expected_init_checkpoint_steps=0",
            "name=fresh_frozen_ddim_dsrl_na_control_seed1_100000chunks",
        ],
    )
    with pytest.raises(ValueError, match="requires prefill source"):
        static_preflight(cfg, PROJECT_ROOT, algorithm=CONTROL_ALGORITHM)


def test_legacy_control_preflight_rejects_fresh_init_checkpoint_id():
    # A legacy-checkpoint control cannot claim the fresh init id: init_5m is
    # the only accepted init for warmstart control.
    cfg = p6_config(
        algorithm=CONTROL_ALGORITHM,
        name_algorithm=CONTROL_ALGORITHM,
        extra_overrides=[
            "p6.init_checkpoint_id=fresh_frozen_ddim",
            "p6.expected_init_checkpoint_steps=0",
            "name=fresh_frozen_ddim_dsrl_na_control_seed1_100000chunks",
        ],
    )
    with pytest.raises(ValueError, match="init_5m"):
        static_preflight(cfg, PROJECT_ROOT, algorithm=CONTROL_ALGORITHM)


def test_p6_production_cadence_defaults_and_10k_override_are_explicit():
    production = compose_p6(
        [
            "total_timesteps=5000000",
            f"rfs_hier_legacy_checkpoint_path={FIVE_M_CHECKPOINT}",
        ]
    )
    assert production.p6.online_eval_interval_chunk_transitions == 100_000
    assert production.p6.model_checkpoint_interval_chunk_transitions == 100_000
    assert production.p6.replay_checkpoint_interval_chunk_transitions == 500_000
    assert production.p6.test_cadence_override is False

    wiring = p6_config(
        extra_overrides=[
            "total_timesteps=10000",
            "train.rfs_hier_phase_r_steps=10000",
            "train.rfs_hier_beta_ramp_steps=1000",
            "name=init_5m_dsrl_na_rfs_hier_seed1_10000chunks",
            "p6.online_eval_interval_chunk_transitions=2000",
            "p6.model_checkpoint_interval_chunk_transitions=2000",
            "p6.replay_checkpoint_interval_chunk_transitions=5000",
            "p6.test_cadence_override=true",
        ]
    )
    manifest = static_preflight(
        wiring,
        PROJECT_ROOT,
        algorithm=HIERARCHY_ALGORITHM,
    )
    assert manifest["test_cadence_override"] is True
    assert manifest["safe_boundary_chunk_transitions"] == 10
    assert manifest["online_eval_interval_chunk_transitions"] == 2_000
    assert manifest["model_checkpoint_interval_chunk_transitions"] == 2_000
    assert manifest["replay_checkpoint_interval_chunk_transitions"] == 5_000


def test_same_config_has_stable_contract_but_distinct_physical_run_ids():
    cfg = p6_config()
    first = static_preflight(cfg, PROJECT_ROOT, algorithm=HIERARCHY_ALGORITHM)
    second = static_preflight(cfg, PROJECT_ROOT, algorithm=HIERARCHY_ALGORITHM)

    assert first["config_contract_sha256"] == second["config_contract_sha256"]
    assert first["run_id"] != second["run_id"]
    assert first["run_id"].startswith("p6-")
    assert second["run_id"].startswith("p6-")


def test_raw_hopper_seed_reaches_the_legacy_environment():
    pytest.importorskip("d4rl")
    pytest.importorskip("d4rl.gym_mujoco")
    cfg = compose_hopper()
    normalization_path = PROJECT_ROOT / str(cfg.normalization_path)

    raw_env_a = gym.make(cfg.env_name)
    raw_env_b = gym.make(cfg.env_name)
    try:
        env_a = ActionChunkWrapper(
            ObservationWrapperGym(raw_env_a, normalization_path),
            cfg,
            max_episode_steps=cfg.env.max_episode_steps,
        )
        env_b = ActionChunkWrapper(
            ObservationWrapperGym(raw_env_b, normalization_path),
            cfg,
            max_episode_steps=cfg.env.max_episode_steps,
        )

        observation_a, _ = env_a.reset(seed=12345)
        observation_b, _ = env_b.reset(seed=12345)
        np.testing.assert_array_equal(observation_a, observation_b)

        different_observation, _ = env_b.reset(seed=54321)
        assert not np.array_equal(observation_a, different_observation)
    finally:
        raw_env_a.close()
        raw_env_b.close()


@pytest.mark.parametrize(
    "config_name",
    (
        "p6_halfcheetah_fresh_2p5m_cotrain",
        "p6_walker_fresh_2p5m_cotrain",
    ),
)
def test_migrated_locomotion_environment_has_audited_chunk_bounds_and_seed(
    config_name,
):
    pytest.importorskip("d4rl")
    pytest.importorskip("d4rl.gym_mujoco")
    cfg = compose_cotrain_locomotion(config_name)
    normalization_path = PROJECT_ROOT / str(cfg.normalization_path)

    raw_env_a = gym.make(cfg.env_name)
    raw_env_b = gym.make(cfg.env_name)
    try:
        env_a = ActionChunkWrapper(
            ObservationWrapperGym(raw_env_a, normalization_path),
            cfg,
            max_episode_steps=cfg.env.max_episode_steps,
            action_chunk_termination_semantics="early_break_on_done",
        )
        env_b = ActionChunkWrapper(
            ObservationWrapperGym(raw_env_b, normalization_path),
            cfg,
            max_episode_steps=cfg.env.max_episode_steps,
            action_chunk_termination_semantics="early_break_on_done",
        )

        assert env_a.action_space.shape == (24,)
        np.testing.assert_array_equal(env_a.action_space.low, -np.ones(24))
        np.testing.assert_array_equal(env_a.action_space.high, np.ones(24))
        observation_a, _ = env_a.reset(seed=12345)
        observation_b, _ = env_b.reset(seed=12345)
        np.testing.assert_array_equal(observation_a, observation_b)
        assert observation_a.shape == (17,)
    finally:
        raw_env_a.close()
        raw_env_b.close()


def test_all_model_constructors_and_legacy_load_receive_the_explicit_seed():
    source = (PROJECT_ROOT / "train_dsrl.py").read_text()
    tree = ast.parse(source)
    # train_dsrl.py intentionally constructs only the flat legacy algorithms
    # (SAC / DSRL / RFSDSRL).  The hierarchy path is gated to p6_train.py: its
    # tagged-prefill, four-mode evaluation and reset-boundary resume do not
    # belong in the generic entry point (see the guard at the top of main()).
    required_constructors = {
        "SAC",
        "DSRL",
    }
    optional_constructors = {"RFSDSRL"}
    seen = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        if node.func.id not in required_constructors | optional_constructors:
            continue
        seen.add(node.func.id)
        assert any(keyword.arg == "seed" for keyword in node.keywords)

    assert required_constructors <= seen
    control_source = (PROJECT_ROOT / "train_dsrl_warmstart_control.py").read_text()
    assert "seed=cfg.seed" in control_source


@pytest.mark.parametrize(
    "runner",
    ("train_dsrl.py", "train_dsrl_warmstart_control.py"),
)
def test_p6_runners_restore_the_training_env_seed_after_prefill(runner):
    source = (PROJECT_ROOT / runner).read_text()
    prefill_index = source.index("collect_rollouts(")
    restore_index = source.index(
        'env.seed(p6_seed_plan["train_env_seed"])',
        prefill_index,
    )
    learn_index = source.index("model.learn(", restore_index)

    assert prefill_index < restore_index < learn_index
