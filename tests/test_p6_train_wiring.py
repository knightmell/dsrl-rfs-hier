from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from gymnasium import Env, spaces
from omegaconf import OmegaConf

from p6_train import (
    P6ControlDSRL,
    _algorithm_label,
    _capture_legacy_warmstart,
    _close_resources_safely,
    _construct_fresh_control,
    _control_is_fresh,
    _construct_hierarchy,
    _finish_attempt,
    _make_policy_kwargs,
    _network_warmstart_label,
    _promote_legacy_control,
    _require_latest_resume_bundle,
    _remaining_training_chunks,
    _reset_control_optimizers,
    _run_is_fresh,
    _set_model_inference_mode,
    _start_attempt,
    _validate_legacy_training_contract,
    _verify_fresh_control_warmstart,
    _write_attempt_traceback,
)
from p6_runtime import (
    assert_matched_fresh_init_state_hashes,
    atomic_write_json,
    canonical_module_state_hash,
)
from p6_preflight import CONTROL_ALGORITHM, HIERARCHY_ALGORITHM, sha256_file
from stable_baselines3.common.logger import configure
from stable_baselines3.dsrl.hierarchical_rfs_dsrl import (
    HierarchicalRFSDSRL,
    _LegacyLoadableDSRL,
)


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
    def __init__(self):
        super().__init__()
        self.gain = torch.nn.Parameter(torch.ones(()))

    def forward(self, observation, noise, return_numpy=False):
        del observation
        result = (noise * self.gain).clamp(-1.0, 1.0)
        return result.detach().cpu().numpy() if return_numpy else result


@pytest.mark.parametrize(
    ("configured", "expected"),
    (
        ("dsrl_na", "dsrl_na_control"),
        ("dsrl_na_rfs_hier", "dsrl_na_rfs_hier"),
    ),
)
def test_p6_algorithm_label_maps_core_v1_labels(configured, expected):
    assert _algorithm_label(OmegaConf.create({"algorithm": configured})) == expected


def test_p6_algorithm_label_rejects_deprecated_frozen_noise():
    with pytest.raises(ValueError, match="frozen-noise diagnostic graph"):
        _algorithm_label(
            OmegaConf.create({"algorithm": "dsrl_na_rfs_hier_frozen_noise"})
        )


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


def make_legacy(decoder=None):
    model = _LegacyLoadableDSRL(
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


def make_control_cfg(*, seed=4):
    return OmegaConf.create(
        {
            "seed": seed,
            "device": "cpu",
            "logdir": "/tmp/p6-test-logdir",
            "act_steps": 2,
            "action_dim": 2,
            "train": {
                "actor_lr": 3e-4,
                "buffer_size_na": 64,
                "batch_size": 2,
                "tau": 0.005,
                "discount": 0.99,
                "train_freq": 1,
                "utd": 20,
                "ent_coef": -1,
                "target_ent": 0.0,
                "noise_critic_grad_steps": 10,
                "critic_backup_combine_type": "min",
                "layer_size": 8,
                "num_layers": 2,
                "use_layer_norm": True,
                "n_critics": 2,
            },
        }
    )


def make_hierarchy_source_cfg(source="gaussian"):
    return OmegaConf.merge(
        make_control_cfg(),
        {
            "rfs_hier_legacy_checkpoint_path": None,
            "p6": {
                "train_env_seed": 1004,
                "action_chunk_termination_semantics": "early_break_on_done",
                "diagnostics_interval_updates": 100,
            },
            "train": {
                "rfs_hier_train_freq": 1,
                "rfs_hier_residual_net_arch": [8, 8],
                "rfs_hier_residual_activation": "silu",
                "rfs_hier_residual_lr": 3e-4,
                "rfs_hier_qa_joint_lr": 3e-4,
                "rfs_hier_schedule_profile": "fresh_frozen_ddim_2p5m_cotrain",
                "rfs_hier_phase_b_steps": 4,
                "rfs_hier_phase_r_steps": 8,
                "rfs_hier_phase_j_steps": 0,
                "rfs_hier_phase_j_enabled": False,
                "rfs_hier_beta_ramp_steps": 4,
                "rfs_hier_beta_target": 0.1,
                "rfs_hier_base_lane_probability": 0.5,
                "rfs_hier_beta_hold_steps": 4,
                "rfs_hier_beta_floor": 0.02,
                "rfs_hier_qa_joint_shadow_in_b": True,
                "rfs_hier_qw_teacher_joint_credit": False,
                "rfs_hier_qw_teacher_source": source,
                "rfs_hier_qw_candidates_per_state": 8,
                "rfs_hier_qw_state_batch_size": 32,
                "rfs_hier_qw_teacher_microbatch_size": 64,
                "rfs_hier_cross_lane_ratio": 0.25,
                "rfs_hier_qa_base_cross_lane": False,
                "rfs_hier_residual_exploration_std": 0.02,
                "rfs_hier_min_branch_replay_transitions": 2,
                "rfs_hier_noise_gradient_max_norm": 1.0,
                "rfs_hier_residual_gradient_max_norm": 1.0,
            },
        },
    )


def test_construct_hierarchy_forwards_qw_teacher_source(monkeypatch):
    captured = {}

    class CapturingHierarchy:
        def __init__(self, *args, **kwargs):
            del args
            captured.update(kwargs)
            self.initialized = False

        def initialize_from_fresh_frozen_ddim(self):
            self.initialized = True

    monkeypatch.setattr("p6_train.HierarchicalRFSDSRL", CapturingHierarchy)
    model = _construct_hierarchy(
        make_hierarchy_source_cfg("gaussian"),
        TinyEnvironment(),
        IdentityDecoder(),
        init_mode="fresh",
    )

    assert model.initialized is True
    assert captured["qw_teacher_source"] == "gaussian"
    assert captured["qw_candidates_per_state"] == 8
    assert captured["qw_state_batch_size"] == 32
    assert captured["qw_teacher_microbatch_size"] == 64


def test_construct_hierarchy_forwards_noise_actor_no_clip_switch(monkeypatch):
    captured = {}

    class CapturingHierarchy:
        def __init__(self, *args, **kwargs):
            del args
            captured.update(kwargs)
            self.initialized = False

        def initialize_from_fresh_frozen_ddim(self):
            self.initialized = True

    cfg = make_hierarchy_source_cfg("current_actor")
    cfg.train.rfs_hier_noise_actor_gradient_clipping = False
    monkeypatch.setattr("p6_train.HierarchicalRFSDSRL", CapturingHierarchy)
    model = _construct_hierarchy(
        cfg,
        TinyEnvironment(),
        IdentityDecoder(),
        init_mode="fresh",
    )

    assert model.initialized is True
    assert captured["noise_actor_gradient_clipping"] is False


def test_construct_hierarchy_forwards_runtime_contract_check_switch(monkeypatch):
    """Fast runtime must opt out only when the resolved config says so."""
    captured = {}

    class CapturingHierarchy:
        def __init__(self, *args, **kwargs):
            del args
            captured.update(kwargs)
            self.initialized = False

        def initialize_from_fresh_frozen_ddim(self):
            self.initialized = True

    cfg = make_hierarchy_source_cfg("gaussian")
    cfg.train.rfs_hier_runtime_contract_checks = False
    monkeypatch.setattr("p6_train.HierarchicalRFSDSRL", CapturingHierarchy)
    model = _construct_hierarchy(
        cfg,
        TinyEnvironment(),
        IdentityDecoder(),
        init_mode="fresh",
    )

    assert model.initialized is True
    assert captured["runtime_contract_checks"] is False


def make_hierarchy_fresh(decoder=None, *, cfg=None):
    """Fresh cotrain hierarchy built from the same cfg/policy-kwargs/seed as a
    matched fresh control, so the base-branch init is directly comparable."""
    cfg = cfg or make_control_cfg()
    model = HierarchicalRFSDSRL(
        "MlpPolicy",
        TinyEnvironment(),
        learning_rate=float(cfg.train.actor_lr),
        buffer_size=int(cfg.train.buffer_size_na),
        learning_starts=1,
        batch_size=int(cfg.train.batch_size),
        tau=float(cfg.train.tau),
        gamma=float(cfg.train.discount),
        train_freq=int(cfg.train.train_freq),
        gradient_steps=int(cfg.train.utd),
        min_branch_replay_transitions=2,
        ent_coef="auto" if cfg.train.ent_coef == -1 else cfg.train.ent_coef,
        target_update_interval=1,
        target_entropy=(
            "auto" if cfg.train.target_ent == -1 else cfg.train.target_ent
        ),
        device="cpu",
        policy_kwargs=_make_policy_kwargs(cfg),
        diffusion_policy=decoder or IdentityDecoder(),
        diffusion_act_dim=(int(cfg.act_steps), int(cfg.action_dim)),
        exec_action_low=np.full(4, -1.0, dtype=np.float32),
        exec_action_high=np.full(4, 1.0, dtype=np.float32),
        schedule_profile="fresh_frozen_ddim_2p5m_cotrain",
        phase_b_steps=4,
        phase_r_steps=8,
        phase_j_steps=0,
        phase_j_enabled=False,
        beta_ramp_steps=4,
        beta_hold_steps=4,
        beta_floor=0.02,
        qa_joint_shadow_in_b=True,
        noise_critic_grad_steps=10,
        critic_backup_combine_type="min",
        lane_seed=int(cfg.seed) + 100,
        seed=int(cfg.seed),
    )
    model.set_logger(configure(folder=None, format_strings=[]))
    model.initialize_from_fresh_frozen_ddim()
    return model


def test_run_is_fresh_distinguishes_fresh_and_legacy_paths():
    fresh_hierarchy = make_control_cfg()
    OmegaConf.update(
        fresh_hierarchy,
        "train.rfs_hier_schedule_profile",
        "fresh_frozen_ddim_2p5m_cotrain",
    )
    legacy_hierarchy = make_control_cfg()
    OmegaConf.update(
        legacy_hierarchy,
        "train.rfs_hier_schedule_profile",
        "legacy_dsrl_warmstart_5m",
    )
    fresh_control = make_control_cfg()
    OmegaConf.update(fresh_control, "rfs_hier_legacy_checkpoint_path", None)
    legacy_control = make_control_cfg()
    OmegaConf.update(
        legacy_control, "rfs_hier_legacy_checkpoint_path", "./logs/init_5m.zip"
    )

    assert _run_is_fresh(fresh_hierarchy, HIERARCHY_ALGORITHM) is True
    assert _run_is_fresh(legacy_hierarchy, HIERARCHY_ALGORITHM) is False
    assert _run_is_fresh(fresh_control, CONTROL_ALGORITHM) is True
    assert _run_is_fresh(legacy_control, CONTROL_ALGORITHM) is False
    assert _control_is_fresh(fresh_control) is True
    assert _control_is_fresh(legacy_control) is False


def test_network_warmstart_label_matches_freshness_for_call_site():
    # Regression (call-site wiring): p6_train used to pass
    # network_warmstart=True unconditionally to finalize_loaded_model_preflight,
    # so fresh runs were mislabeled as network-warm-starts.  The caller now
    # derives the label from the same _run_is_fresh decision that picks the
    # init branch; the label must be True exactly when the run is legacy-driven.
    fresh_hierarchy = make_control_cfg()
    OmegaConf.update(
        fresh_hierarchy,
        "train.rfs_hier_schedule_profile",
        "fresh_frozen_ddim_2p5m_cotrain",
    )
    legacy_hierarchy = make_control_cfg()
    OmegaConf.update(
        legacy_hierarchy,
        "train.rfs_hier_schedule_profile",
        "legacy_dsrl_warmstart_5m",
    )
    fresh_control = make_control_cfg()
    OmegaConf.update(fresh_control, "rfs_hier_legacy_checkpoint_path", None)
    legacy_control = make_control_cfg()
    OmegaConf.update(
        legacy_control, "rfs_hier_legacy_checkpoint_path", "./logs/init_5m.zip"
    )

    assert _network_warmstart_label(fresh_hierarchy, HIERARCHY_ALGORITHM) is False
    assert _network_warmstart_label(legacy_hierarchy, HIERARCHY_ALGORITHM) is True
    assert _network_warmstart_label(fresh_control, CONTROL_ALGORITHM) is False
    assert _network_warmstart_label(legacy_control, CONTROL_ALGORITHM) is True
    # The label must be the exact complement of the freshness decision the
    # init branch is keyed on (init_mode = "fresh" iff fresh).
    for cfg, algorithm in (
        (fresh_hierarchy, HIERARCHY_ALGORITHM),
        (legacy_hierarchy, HIERARCHY_ALGORITHM),
        (fresh_control, CONTROL_ALGORITHM),
        (legacy_control, CONTROL_ALGORITHM),
    ):
        assert _network_warmstart_label(cfg, algorithm) is not _run_is_fresh(
            cfg, algorithm
        )


def test_fresh_control_matches_hierarchy_fresh_base_branch():
    # The matched flat-DSRL control is built from the same seeded constructors
    # and policy kwargs as the fresh hierarchy, so the base branch (noise
    # actor / QA_base / QW_base) must start bit-identical.  Any drift here
    # would confound the gate comparison with an init difference.
    decoder = IdentityDecoder()
    cfg = make_control_cfg(seed=13)
    hierarchy = make_hierarchy_fresh(decoder=decoder, cfg=cfg)
    control = _construct_fresh_control(cfg, TinyEnvironment(), decoder)

    assert type(control) is P6ControlDSRL
    assert control.num_timesteps == 0
    # Flat DSRL names the QA_base/QW_base modules critic / critic_noise; the
    # hierarchy aliases them qa_base / qw_base.
    for control_name, hierarchy_name in (
        ("actor", "actor"),
        ("critic", "qa_base"),
        ("critic_noise", "qw_base"),
    ):
        assert_nested_exact(
            getattr(control, control_name).state_dict(),
            getattr(hierarchy, hierarchy_name).state_dict(),
        )
    # Fresh control mirrors the hierarchy's fresh init for the target copy.
    assert_nested_exact(
        control.critic_target.state_dict(),
        control.critic.state_dict(),
    )
    # Frozen DDIM is placed in inference mode, matching the hierarchy.
    assert control.diffusion_policy.training is False
    assert all(
        parameter.requires_grad is False
        for parameter in control.diffusion_policy.parameters()
    )


def test_fresh_control_and_hierarchy_record_identical_init_state_hashes():
    # Regression (artifact-level init proof): the fresh control and fresh
    # hierarchy must record canonical state hashes of the SAME base-branch
    # modules under the SAME keys, so a cross-manifest gate can prove both runs
    # started bit-identical.  p6_train writes these dicts into each run's
    # manifest; this test builds the same dicts in memory from the same
    # constructors and asserts they agree, exactly as the manifest check would.
    decoder = IdentityDecoder()
    cfg = make_control_cfg(seed=13)
    hierarchy = make_hierarchy_fresh(decoder=decoder, cfg=cfg)
    control = _construct_fresh_control(cfg, TinyEnvironment(), decoder)

    # Flat DSRL names QA_base/QW_base as critic/critic_noise; the manifest
    # projection maps them onto the hierarchy's canonical keys.  Each parity
    # dict is stored in the run manifest under "zero_residual_ddim_parity",
    # mirroring exactly what p6_train records.
    hierarchy_parity = {
        "zero_residual_ddim_parity": {
            "fresh_init_state_hashes": {
                "actor": canonical_module_state_hash(hierarchy.actor),
                "qa_base": canonical_module_state_hash(hierarchy.qa_base),
                "qa_base_target": canonical_module_state_hash(
                    hierarchy.qa_base_target
                ),
                "qw_base": canonical_module_state_hash(hierarchy.qw_base),
            },
        },
    }
    control_parity = {
        "zero_residual_ddim_parity": {
            "fresh_init_state_hashes": {
                "actor": canonical_module_state_hash(control.actor),
                "qa_base": canonical_module_state_hash(control.critic),
                "qa_base_target": canonical_module_state_hash(
                    control.critic_target
                ),
                "qw_base": canonical_module_state_hash(control.critic_noise),
            },
        },
    }
    assert_matched_fresh_init_state_hashes(control_parity, hierarchy_parity)


def test_matched_fresh_init_gate_rejects_missing_or_null_parity():
    # Regression (helper robustness): the gate must reject a manifest pair that
    # does not carry the artifact-level init proof with a clean ValueError, not
    # a crash.  Two cases: (a) a legacy/warmstart run omits
    # zero_residual_ddim_parity entirely (the hashes are recorded only on the
    # fresh path), and (b) the key is present but explicitly null in JSON.
    decoder = IdentityDecoder()
    cfg = make_control_cfg(seed=13)
    hierarchy = make_hierarchy_fresh(decoder=decoder, cfg=cfg)
    control = _construct_fresh_control(cfg, TinyEnvironment(), decoder)

    hierarchy_parity = {
        "zero_residual_ddim_parity": {
            "fresh_init_state_hashes": {
                "actor": canonical_module_state_hash(hierarchy.actor),
                "qa_base": canonical_module_state_hash(hierarchy.qa_base),
                "qa_base_target": canonical_module_state_hash(
                    hierarchy.qa_base_target
                ),
                "qw_base": canonical_module_state_hash(hierarchy.qw_base),
            },
        },
    }
    # Legacy/warmstart runs record no parity dict at all.
    legacy_manifest = {}
    with pytest.raises(ValueError, match="missing from one or both"):
        assert_matched_fresh_init_state_hashes(legacy_manifest, hierarchy_parity)
    # The key present-but-null (JSON null) must behave like missing, not raise
    # AttributeError on `None.get`.
    null_manifest = {"zero_residual_ddim_parity": None}
    with pytest.raises(ValueError, match="missing from one or both"):
        assert_matched_fresh_init_state_hashes(null_manifest, hierarchy_parity)
    # A complete fresh pair still passes.
    control_parity = {
        "zero_residual_ddim_parity": {
            "fresh_init_state_hashes": {
                "actor": canonical_module_state_hash(control.actor),
                "qa_base": canonical_module_state_hash(control.critic),
                "qa_base_target": canonical_module_state_hash(
                    control.critic_target
                ),
                "qw_base": canonical_module_state_hash(control.critic_noise),
            },
        },
    }
    assert_matched_fresh_init_state_hashes(control_parity, hierarchy_parity)


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


def assert_nested_exact(actual, expected):
    assert type(actual) is type(expected)
    if isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            assert_nested_exact(actual[key], expected[key])
    elif isinstance(actual, (list, tuple)):
        assert len(actual) == len(expected)
        for actual_value, expected_value in zip(actual, expected):
            assert_nested_exact(actual_value, expected_value)
    elif isinstance(actual, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    else:
        assert actual == expected


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
    assert model.policy.training is False
    assert model.actor.training is False
    assert model.critic.training is False
    assert model.critic_target.training is False
    assert model.critic_noise.training is False
    assert model.diffusion_policy.training is False
    assert all(
        parameter.requires_grad is False
        for parameter in model.diffusion_policy.parameters()
    )

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
    for restored_optimizer, expected_optimizer in (
        (restored.actor.optimizer, model.actor.optimizer),
        (restored.critic.optimizer, model.critic.optimizer),
        (restored.critic_noise.optimizer, model.critic_noise.optimizer),
        (restored.ent_coef_optimizer, model.ent_coef_optimizer),
    ):
        assert_nested_exact(
            restored_optimizer.state_dict(),
            expected_optimizer.state_dict(),
        )


def test_official_legacy_checkpoint_loads_before_p6_control_promotion(tmp_path):
    legacy = make_legacy()
    legacy.num_timesteps = 5_000_000
    checkpoint = tmp_path / "legacy_without_qw_optimizer.zip"
    legacy.save(checkpoint)

    loaded = _LegacyLoadableDSRL.load(
        checkpoint,
        env=TinyEnvironment(),
        device="cpu",
        custom_objects={"diffusion_policy": IdentityDecoder()},
        buffer_size=64,
    )
    assert type(loaded) is _LegacyLoadableDSRL
    assert loaded.num_timesteps == 5_000_000

    expected_network = _capture_legacy_warmstart(loaded)
    promoted = _promote_legacy_control(loaded)
    _reset_control_optimizers(promoted)
    parity = _verify_fresh_control_warmstart(promoted, expected_network)
    assert type(promoted) is P6ControlDSRL
    assert promoted.num_timesteps == 0
    assert len(promoted.critic_noise.optimizer.state) == 0
    _set_model_inference_mode(promoted)
    assert promoted.policy.training is False
    assert promoted.actor.training is False
    assert promoted.critic.training is False
    assert promoted.critic_target.training is False
    assert promoted.critic_noise.training is False
    assert promoted.diffusion_policy.training is False
    assert all(
        parameter.requires_grad is False
        for parameter in promoted.diffusion_policy.parameters()
    )
    assert set(parity.values()) == {0.0}


def test_attempt_history_preserves_interruption_and_resolved_configs(tmp_path):
    run_directory = tmp_path / "run"
    run_directory.mkdir()
    manifest_path = run_directory / "run_manifest.json"
    atomic_write_json(
        manifest_path,
        {
            "run_status": "preflight",
            "final_evaluation_status": "pending",
        },
    )
    fresh_cfg = OmegaConf.create(
        {
            "total_timesteps": 40,
            "p6": {
                "stop_after_chunk_transitions": 20,
            },
        }
    )
    first_id = _start_attempt(
        cfg=fresh_cfg,
        manifest_path=manifest_path,
        run_directory=run_directory,
        start_chunk=0,
        resume_source=None,
    )
    _finish_attempt(
        manifest_path=manifest_path,
        run_directory=run_directory,
        status="interrupted",
        end_chunk=20,
        reason="intentional",
    )

    resume_cfg = OmegaConf.create(
        {
            "total_timesteps": 40,
            "p6": {
                "stop_after_chunk_transitions": None,
            },
        }
    )
    second_id = _start_attempt(
        cfg=resume_cfg,
        manifest_path=manifest_path,
        run_directory=run_directory,
        start_chunk=20,
        resume_source=run_directory / "resume" / "chunk_000000000020",
    )
    _finish_attempt(
        manifest_path=manifest_path,
        run_directory=run_directory,
        status="complete",
        end_chunk=40,
    )

    manifest = json.loads(manifest_path.read_text())
    assert (first_id, second_id) == (0, 1)
    assert manifest["attempts"][0]["status"] == "interrupted"
    assert manifest["attempts"][0]["requested_stop_after_chunk"] == 20
    assert manifest["attempts"][0]["reason"] == "intentional"
    assert manifest["attempts"][1]["status"] == "complete"
    assert manifest["attempts"][1]["requested_stop_after_chunk"] is None
    assert manifest["attempts"][1]["start_chunk"] == 20
    assert manifest["attempts"][1]["end_chunk"] == 40
    for attempt in manifest["attempts"]:
        config_path = run_directory / attempt["resolved_config_path"]
        assert config_path.is_file()
        assert sha256_file(config_path) == attempt["resolved_config_sha256"]


def test_completed_training_budget_skips_learn_but_overshoot_fails():
    assert _remaining_training_chunks(20, 40) == 20
    assert _remaining_training_chunks(40, 40) == 0

    with pytest.raises(ValueError, match="exceeded target"):
        _remaining_training_chunks(41, 40)


def test_attempt_failure_persists_full_traceback_without_top_level_stale_state(
    tmp_path,
):
    run_directory = tmp_path / "run"
    run_directory.mkdir()
    manifest_path = run_directory / "run_manifest.json"
    atomic_write_json(
        manifest_path,
        {
            "run_status": "preflight",
            "final_evaluation_status": "pending",
        },
    )
    cfg = OmegaConf.create(
        {
            "total_timesteps": 40,
            "p6": {"stop_after_chunk_transitions": None},
        }
    )
    _start_attempt(
        cfg=cfg,
        manifest_path=manifest_path,
        run_directory=run_directory,
        start_chunk=40,
        resume_source=run_directory / "resume" / "chunk_000000000040",
    )
    traceback_path = _write_attempt_traceback(
        manifest_path=manifest_path,
        run_directory=run_directory,
        traceback_text="Traceback (most recent call last):\nRuntimeError: final eval\n",
    )
    _finish_attempt(
        manifest_path=manifest_path,
        run_directory=run_directory,
        status="failed",
        end_chunk=40,
        reason="RuntimeError: final eval",
        traceback_path=traceback_path,
    )

    manifest = json.loads(manifest_path.read_text())
    attempt = manifest["attempts"][0]
    assert attempt["traceback_path"] == "attempts/0000/traceback.txt"
    assert (run_directory / attempt["traceback_path"]).read_text().endswith(
        "RuntimeError: final eval\n"
    )
    assert attempt["traceback_sha256"] == sha256_file(
        run_directory / attempt["traceback_path"]
    )
    assert "failure_traceback" not in manifest


def test_resume_must_use_manifest_latest_bundle(tmp_path):
    run_directory = tmp_path / "run"
    latest = run_directory / "resume" / "chunk_000000000040"
    stale = run_directory / "resume" / "chunk_000000000020"
    manifest = {
        "latest_resume_bundle": str(latest),
        "latest_resume_bundle_chunk": 40,
    }

    _require_latest_resume_bundle(manifest, latest)
    with pytest.raises(ValueError, match="non-latest"):
        _require_latest_resume_bundle(manifest, stale)


@pytest.mark.parametrize(
    ("mutation", "match"),
    (
        ("learning_rate", "actor_learning_rate"),
        ("learning_starts", "learning_starts"),
        ("target_update_interval", "target_update_interval"),
        ("entropy_mode", "entropy mode"),
    ),
)
def test_legacy_control_contract_rejects_update_semantic_drift(
    mutation,
    match,
):
    model = make_legacy()
    cfg = OmegaConf.create(
        {
            "train": {
                "actor_lr": 3e-4,
                "batch_size": 2,
                "discount": model.gamma,
                "utd": model.gradient_steps,
                "noise_critic_grad_steps": model.noise_critic_grad_steps,
                "target_ent": model.target_entropy,
                "tau": model.tau,
                "train_freq": model.train_freq.frequency,
                "ent_coef": -1,
            }
        }
    )
    _validate_legacy_training_contract(cfg, model)

    if mutation == "learning_rate":
        model.lr_schedule = lambda _: 1e-4
    elif mutation == "learning_starts":
        model.learning_starts += 1
    elif mutation == "target_update_interval":
        model.target_update_interval += 1
    elif mutation == "entropy_mode":
        model.log_ent_coef = None
    else:
        raise AssertionError(mutation)

    with pytest.raises(ValueError, match=match):
        _validate_legacy_training_contract(cfg, model)


def test_cleanup_failures_are_persisted_without_masking_run_completion(
    tmp_path,
):
    class FailingWriter:
        def flush(self):
            raise RuntimeError("flush failed")

        def close(self):
            raise RuntimeError("close failed")

    class FailingEnvironment:
        def close(self):
            raise RuntimeError("env close failed")

    manifest_path = tmp_path / "run_manifest.json"
    atomic_write_json(manifest_path, {"run_status": "complete"})

    errors = _close_resources_safely(
        writer=FailingWriter(),
        training_environment=FailingEnvironment(),
        wandb_run=None,
        manifest_path=manifest_path,
    )

    assert len(errors) == 3
    manifest = json.loads(manifest_path.read_text())
    assert manifest["run_status"] == "complete"
    assert manifest["cleanup_errors"] == errors


def test_train_dsrl_flat_path_never_wires_tagged_replay():
    # Regression: the legacy flat dsrl_na_rfs entry point accidentally wired
    # the tagged HierarchyTaggedReplayBuffer, which rejects untagged flat
    # transitions on its first add().  The certified tagged-prefill hierarchy
    # path lives only in p6_train.py; train_dsrl.py must not re-introduce it.
    source = (Path(__file__).resolve().parent.parent / "train_dsrl.py").read_text()
    assert "replay_buffer_class=" not in source
    # And the stale dead hier branch (which mixed generic collect_rollouts
    # with the tagged replay) must not come back as a constructible path.
    assert "elif cfg.algorithm == 'dsrl_na_rfs_hier'" not in source
