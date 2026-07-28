from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


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
