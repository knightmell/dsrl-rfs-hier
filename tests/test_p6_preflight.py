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
    CONTROL_ALGORITHM,
    HIERARCHY_ALGORITHM,
    finalize_loaded_model_preflight,
    resolve_seed_plan,
    run_preflight,
    static_preflight,
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


def p6_config(
    *,
    algorithm=HIERARCHY_ALGORITHM,
    name_algorithm=None,
    extra_overrides=None,
):
    name_algorithm = name_algorithm or algorithm
    overrides = [
        f"algorithm={'dsrl_na' if algorithm == CONTROL_ALGORITHM else algorithm}",
        "env.n_envs=10",
        "total_timesteps=100000",
        f"rfs_hier_legacy_checkpoint_path={FIVE_M_CHECKPOINT}",
        f"name=init_5m_{name_algorithm}_seed1_100000chunks",
    ]
    overrides.extend(extra_overrides or [])
    return compose_hopper(overrides)


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
    assert manifest["primitive_budget"] == 400_000
    assert manifest["n_envs"] == 10
    assert manifest["prefill_source"] == "warmstart_dsrl"
    assert manifest["prefill_transition_count"] == 20_010
    assert manifest["prefill_hash"] is None
    assert manifest["prefill_status"] == "pending_p6_2"
    assert manifest["action_chunk_termination_semantics"] == (
        "legacy_continue_after_done"
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


def test_preflight_rejects_missing_checkpoint_and_non_explicit_pilot_env_count():
    missing_checkpoint_cfg = compose_hopper(
        [
            "algorithm=dsrl_na_rfs_hier",
            "env.n_envs=10",
            "total_timesteps=100000",
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
    )

    manifest = finalize_loaded_model_preflight(
        cfg,
        env,
        model,
        manifest_path,
    )

    assert manifest["preflight_status"] == "loaded_model_verified_prefill_pending"
    assert manifest["init_checkpoint_num_timesteps"] == 5_000_000
    assert manifest["training_start_num_timesteps"] == 5_000_000
    assert manifest["network_warmstart"] is False
    assert manifest["loaded_diffusion_shape"] == [4, 3]

    model.num_timesteps = 7_500_000
    with pytest.raises(ValueError, match="step mismatch"):
        finalize_loaded_model_preflight(
            cfg,
            env,
            model,
            manifest_path,
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
    with pytest.raises(ValueError, match="zero new interaction steps"):
        finalize_loaded_model_preflight(
            cfg,
            env,
            model,
            manifest_path,
            network_warmstart=True,
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


def test_all_model_constructors_and_legacy_load_receive_the_explicit_seed():
    source = (PROJECT_ROOT / "train_dsrl.py").read_text()
    tree = ast.parse(source)
    required_constructors = {
        "SAC",
        "DSRL",
        "HierarchicalRFSDSRL",
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
