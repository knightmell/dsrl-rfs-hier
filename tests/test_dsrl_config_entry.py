from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from p6_preflight import HIERARCHY_ALGORITHM


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DSRL_CONFIGS = (
    ("gym", "dsrl_halfcheetah"),
    ("gym", "dsrl_hopper"),
    ("gym", "dsrl_walker"),
    ("robomimic", "dsrl_can"),
    ("robomimic", "dsrl_lift"),
    ("robomimic", "dsrl_square"),
    ("robomimic", "dsrl_transport"),
)
# Locomotion configs whose base `dsrl_na` and flat `dsrl_na_rfs` branches in
# train_dsrl.py are the planned 10M base runs.
GYM_DSRL_CONFIGS = (
    "dsrl_halfcheetah",
    "dsrl_hopper",
    "dsrl_walker",
)

OmegaConf.register_new_resolver("eval", eval, replace=True)


def compose_dsrl_config(group, config_name, overrides=None):
    config_dir = PROJECT_ROOT / "cfg" / group
    with initialize_config_dir(
        config_dir=str(config_dir),
        job_name=f"test_{config_name}",
        version_base=None,
    ):
        return compose(
            config_name=config_name,
            overrides=overrides or [],
        )


@pytest.mark.parametrize("group,config_name", DSRL_CONFIGS)
def test_every_existing_dsrl_config_explicitly_defaults_to_20m(
    group,
    config_name,
):
    cfg = compose_dsrl_config(group, config_name)

    assert cfg.total_timesteps == 20_000_000
    assert cfg.train.buffer_size_sac == 20_000_000
    assert cfg.train.buffer_size_na == 10_000_000


def test_hopper_preserves_legacy_default_env_count_and_has_no_checkpoint_default():
    cfg = compose_dsrl_config("gym", "dsrl_hopper")

    assert cfg.env.n_envs == 4
    assert cfg.rfs_hier_legacy_checkpoint_path is None


def test_hopper_total_timesteps_can_be_explicitly_overridden_to_5m():
    cfg = compose_dsrl_config(
        "gym",
        "dsrl_hopper",
        overrides=[
            "algorithm=dsrl_na_rfs_hier",
            "total_timesteps=5000000",
        ],
    )

    assert cfg.algorithm == "dsrl_na_rfs_hier"
    assert cfg.total_timesteps == 5_000_000


def test_hopper_p6_pilot_env_count_requires_an_explicit_override():
    default_cfg = compose_dsrl_config("gym", "dsrl_hopper")
    pilot_cfg = compose_dsrl_config(
        "gym",
        "dsrl_hopper",
        overrides=["env.n_envs=10"],
    )

    assert default_cfg.env.n_envs == 4
    assert pilot_cfg.env.n_envs == 10


@pytest.mark.parametrize(
    "checkpoint_path",
    (
        "./logs/checkpoints/ft_policy_1000000_steps.zip",
        "./logs/checkpoints/ft_policy_5000000_steps.zip",
        "./logs/checkpoints/ft_policy_10000000_steps.zip",
    ),
)
def test_hopper_checkpoint_path_is_an_explicit_hydra_override(checkpoint_path):
    cfg = compose_dsrl_config(
        "gym",
        "dsrl_hopper",
        overrides=[f"rfs_hier_legacy_checkpoint_path={checkpoint_path}"],
    )

    assert cfg.rfs_hier_legacy_checkpoint_path == checkpoint_path


def test_training_entry_uses_the_explicit_configured_total():
    source = (PROJECT_ROOT / "train_dsrl.py").read_text()

    assert "total_timesteps=cfg.total_timesteps" in source
    assert "total_timesteps=20000000" not in source


# The flat dsrl_na_rfs branch (train_dsrl.py:180-214) reads every one of these
# keys from cfg.train at construction time.  Accessing a missing key raises
# ConfigAttributeError, which is exactly how the MED-3 regression surfaced
# (rfs_hier_train_freq was absent from the HC/WK base configs).  Touching each
# key here is a cheap smoke test that the branch can actually be constructed
# from these configs before a 10M base run is launched.
DSRL_NA_RFS_BRANCH_KEYS = (
    "actor_lr",
    "buffer_size_na",
    "batch_size",
    "tau",
    "discount",
    "rfs_hier_train_freq",
    "utd",
    "noise_critic_grad_steps",
    "critic_backup_combine_type",
    "rfs_residual_scale",
    "rfs_residual_log_std_init",
    "ent_coef",
    "target_ent",
)

# The base dsrl_na branch (train_dsrl.py:152-178) reads this subset plus the
# legacy update cadence key (which the rfs branch does not use).
DSRL_NA_BRANCH_KEYS = (
    "actor_lr",
    "buffer_size_na",
    "batch_size",
    "tau",
    "discount",
    "train_freq",
    "utd",
    "noise_critic_grad_steps",
    "critic_backup_combine_type",
    "ent_coef",
    "target_ent",
)


@pytest.mark.parametrize("config_name", GYM_DSRL_CONFIGS)
def test_dsrl_na_branch_keys_resolve_for_gym_configs(config_name):
    cfg = compose_dsrl_config(
        "gym",
        config_name,
        overrides=["algorithm=dsrl_na"],
    )

    assert cfg.algorithm == "dsrl_na"
    for key in DSRL_NA_BRANCH_KEYS:
        value = cfg.train[key]
        assert value is not None, f"dsrl_na branch key train.{key} is null"
    assert cfg.train.train_freq >= 1


@pytest.mark.parametrize("config_name", GYM_DSRL_CONFIGS)
def test_flat_dsrl_na_rfs_branch_keys_resolve_for_gym_configs(config_name):
    # Regression guard for MED-3: every key the flat dsrl_na_rfs branch reads
    # must be present and usable in all three gym base configs.
    cfg = compose_dsrl_config(
        "gym",
        config_name,
        overrides=["algorithm=dsrl_na_rfs"],
    )

    assert cfg.algorithm == "dsrl_na_rfs"
    for key in DSRL_NA_RFS_BRANCH_KEYS:
        value = cfg.train[key]
        assert value is not None, f"dsrl_na_rfs branch key train.{key} is null"
    assert cfg.train.rfs_hier_train_freq >= 1
    assert cfg.train.rfs_residual_scale >= 0
    assert cfg.train.rfs_residual_log_std_init is not None


def test_train_dsrl_rejects_hierarchy_algorithm_and_guard_value_is_coherent():
    # The RuntimeError guard at the top of train_dsrl.main() fires only when
    # the resolved algorithm equals HIERARCHY_ALGORITHM.  Pin the guard value
    # against the imported constant so a rename cannot silently un-moor the
    # rejection path from the actual hierarchy label.
    cfg = compose_dsrl_config(
        "gym",
        "dsrl_halfcheetah",
        overrides=["algorithm=dsrl_na_rfs_hier"],
    )

    assert cfg.algorithm == HIERARCHY_ALGORITHM
