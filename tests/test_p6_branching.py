"""Certified schedule-only forks from a completed P6 boundary bundle."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import torch
from stable_baselines3.common.save_util import load_from_zip_file, save_to_zip_file
from stable_baselines3.dsrl.hierarchy_schedule import make_hierarchy_schedule

from p6_branching import (
    branch_runtime_payload,
    build_branch_manifest,
    rewrite_model_schedule_archive,
    validate_schedule_branch_contract,
)


def _load_zip(path: Path):
    return load_from_zip_file(path, device="cpu")


def _assert_nested_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert isinstance(right, dict)
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right)
        assert len(left) == len(right)
        for lhs, rhs in zip(left, right):
            _assert_nested_equal(lhs, rhs)
    else:
        assert left == right


def test_model_schedule_rewrite_preserves_every_parameter_and_optimizer_tensor(
    tmp_path: Path,
):
    parent = tmp_path / "parent.zip"
    child = tmp_path / "child.zip"
    parent_schedule = make_hierarchy_schedule(
        "fresh_frozen_ddim_2p5m_cotrain", n_envs=10
    )
    child_schedule = make_hierarchy_schedule(
        "fresh_frozen_ddim_2p5m_additive_res", n_envs=10
    )
    parent_data = {
        "num_timesteps": 500_000,
        "schedule_profile": parent_schedule.profile_name,
        "_schedule_overrides": {
            "phase_b_steps": 500_000,
            "phase_r_steps": 2_000_000,
            "phase_j_steps": 0,
            "phase_j_enabled": False,
            "beta_ramp_steps": 50_000,
            "beta_target": 0.1,
            "base_lane_probability": 0.5,
            "beta_hold_steps": 50_000,
            "beta_floor": 0.02,
        },
        "hierarchy_schedule": parent_schedule,
        "tensorboard_log": "/parent/tensorboard",
        "unrelated_state": {"counter": 17},
    }
    parent_params = {
        "policy": {"weight": torch.tensor([[1.0, 2.0]])},
        "actor.optimizer": {"state": {0: {"exp_avg": torch.tensor([3.0])}}},
    }
    parent_pytorch = {"log_ent_coef": torch.tensor([0.25])}
    save_to_zip_file(parent, parent_data, parent_params, parent_pytorch)

    report = rewrite_model_schedule_archive(
        parent,
        child,
        child_schedule,
        tensorboard_log="/child/tensorboard",
    )

    old_data, old_params, old_pytorch = _load_zip(parent)
    new_data, new_params, new_pytorch = _load_zip(child)
    _assert_nested_equal(new_params, old_params)
    _assert_nested_equal(new_pytorch, old_pytorch)
    assert new_data["unrelated_state"] == {"counter": 17}
    assert new_data["num_timesteps"] == 500_000
    assert new_data["tensorboard_log"] == "/child/tensorboard"
    assert new_data["schedule_profile"] == (
        "fresh_frozen_ddim_2p5m_additive_res"
    )
    assert new_data["hierarchy_schedule"] == child_schedule
    assert new_data["_schedule_overrides"] == {
        "phase_b_steps": 500_000,
        "phase_r_steps": 2_000_000,
        "phase_j_steps": 0,
        "phase_j_enabled": False,
        "beta_ramp_steps": 50_000,
        "beta_target": 0.1,
        "base_lane_probability": 0.5,
        "beta_hold_steps": 50_000,
        "beta_floor": 0.02,
    }
    assert report["changed_model_data_keys"] == [
        "_schedule_overrides",
        "hierarchy_schedule",
        "schedule_profile",
        "tensorboard_log",
    ]


def test_runtime_branch_changes_only_binding_schedule_metadata_and_cadence():
    parent_binding = {
        "run_id": "parent",
        "config_contract_sha256": "parent-contract",
        "source_state_sha256": "source",
        "prefill_semantic_hash": "prefill",
        "initial_replay_semantic_hash": "replay-prefix",
    }
    child_binding = {**parent_binding, "run_id": "child", "config_contract_sha256": "child-contract"}
    parent_hierarchy = {
        "schedule_profile": "fresh_frozen_ddim_2p5m_cotrain",
        "phase_b_steps": 500_000,
        "phase_r_steps": 2_000_000,
        "requested_optimizer_steps": {"qa_base": 1_000_000, "residual_actor": 0},
    }
    child_hierarchy = {
        **parent_hierarchy,
        "schedule_profile": "fresh_frozen_ddim_2p5m_additive_res",
    }
    payload = {
        "run_binding": copy.deepcopy(parent_binding),
        "runtime_state": {
            "run_binding": copy.deepcopy(parent_binding),
            "training_counters": {"chunk_transitions": 500_000},
            "next_online_eval_chunk": 550_000,
            "next_model_checkpoint_chunk": 550_000,
            "next_replay_checkpoint_chunk": 600_000,
            "online_eval_interval": 50_000,
            "model_checkpoint_interval": 50_000,
            "replay_checkpoint_interval": 100_000,
            "last_online_eval_chunk": 500_000,
            "last_model_checkpoint_chunk": 500_000,
            "last_replay_checkpoint_chunk": 500_000,
        },
        "rng_state": {"torch": torch.tensor([11, 12], dtype=torch.uint8)},
        "optimizer_counters": {"qa_base_optimizer_steps": 1_000_000, "residual_actor_optimizer_steps": 0},
        "hierarchy_state": copy.deepcopy(parent_hierarchy),
    }

    child, report = branch_runtime_payload(
        payload,
        new_binding=child_binding,
        new_hierarchy_state=child_hierarchy,
        online_eval_interval=50_000,
        model_checkpoint_interval=50_000,
        replay_checkpoint_interval=250_000,
    )

    assert torch.equal(child["rng_state"]["torch"], payload["rng_state"]["torch"])
    assert child["optimizer_counters"] == payload["optimizer_counters"]
    assert child["runtime_state"]["training_counters"] == {"chunk_transitions": 500_000}
    assert child["run_binding"] == child_binding
    assert child["runtime_state"]["run_binding"] == child_binding
    assert child["hierarchy_state"] == child_hierarchy
    assert child["runtime_state"]["next_online_eval_chunk"] == 550_000
    assert child["runtime_state"]["next_model_checkpoint_chunk"] == 550_000
    assert child["runtime_state"]["next_replay_checkpoint_chunk"] == 750_000
    assert child["runtime_state"]["replay_checkpoint_interval"] == 250_000
    assert report == {
        "branch_chunk_transitions": 500_000,
        "next_online_eval_chunk": 550_000,
        "next_model_checkpoint_chunk": 550_000,
        "next_replay_checkpoint_chunk": 750_000,
        "runtime_learning_state_modified": False,
        "rng_state_modified": False,
        "optimizer_state_modified": False,
    }


def test_branch_contract_rejects_changes_outside_schedule_name_and_replay_cadence():
    parent = {
        "run_name": "parent",
        "replay_checkpoint_interval": 100_000,
        "seed": 1,
        "source_state_sha256": "parent-source",
        "training": {
            "batch_size": 256,
            "hierarchy": {"schedule_profile": "parent", "phase_b_steps": 500_000},
        },
    }
    child = copy.deepcopy(parent)
    child["run_name"] = "child"
    child["replay_checkpoint_interval"] = 250_000
    child["source_state_sha256"] = "child-source-with-branch-tool-only"
    child["training"]["hierarchy"] = {
        "schedule_profile": "additive",
        "phase_b_steps": 500_000,
    }

    report = validate_schedule_branch_contract(parent, child)
    assert report["unchanged_contract_fields_verified"] is True

    child["training"]["batch_size"] = 128
    try:
        validate_schedule_branch_contract(parent, child)
    except ValueError as error:
        assert "training.batch_size" in str(error)
    else:
        raise AssertionError("A schedule fork accepted a batch-size change")


def test_child_manifest_inherits_parent_data_lineage_and_resets_attempt_history(tmp_path: Path):
    parent = {
        "run_id": "parent-id",
        "run_name": "parent-name",
        "config_contract": {"run_name": "parent-name"},
        "config_contract_sha256": "parent-contract",
        "source_state_sha256": "same-source",
        "prefill_status": "verified_and_loaded",
        "prefill_hash": "prefill",
        "prefill_semantic_hash": "prefill",
        "initial_replay_hash": {"semantic_hash": "initial-replay"},
        "attempts": [{"attempt_id": 0, "status": "interrupted"}],
        "current_attempt_id": 0,
        "run_status": "interrupted",
        "training_status": "interrupted",
        "latest_resume_bundle": "/parent/bundle",
        "latest_resume_bundle_chunk": 500_000,
        "failure_type": "old-failure",
    }
    static = {
        "run_id": "child-id",
        "run_name": "child-name",
        "config_contract": {"run_name": "child-name"},
        "config_contract_sha256": "child-contract",
        "source_state_sha256": "same-source",
        "prefill_status": "pending_tagged_prefill",
        "prefill_hash": None,
        "stop_after_chunk_transitions": 750_000,
        "chunk_budget": 2_500_000,
    }
    child_bundle = tmp_path / "child" / "resume" / "chunk_000000500000"

    child = build_branch_manifest(
        parent,
        static,
        parent_bundle_path=Path("/parent/bundle"),
        child_bundle_path=child_bundle,
        branch_chunk_transitions=500_000,
        bundle_manifest_sha256="bundle-hash",
    )

    assert child["run_id"] == "child-id"
    assert child["config_contract_sha256"] == "child-contract"
    assert child["prefill_status"] == "verified_and_loaded"
    assert child["prefill_hash"] == "prefill"
    assert child["prefill_semantic_hash"] == "prefill"
    assert child["initial_replay_hash"] == {"semantic_hash": "initial-replay"}
    assert child["attempts"] == []
    assert "current_attempt_id" not in child
    assert "failure_type" not in child
    assert child["run_status"] == "interrupted"
    assert child["training_status"] == "branch_bundle_ready"
    assert child["latest_resume_bundle"] == str(child_bundle.resolve())
    assert child["latest_resume_bundle_chunk"] == 500_000
    assert child["remaining_chunk_transitions"] == 2_000_000
    assert child["branch_lineage"] == {
        "branch_chunk_transitions": 500_000,
        "parent_bundle_path": "/parent/bundle",
        "parent_run_id": "parent-id",
        "parent_config_contract_sha256": "parent-contract",
        "parent_source_state_sha256": "same-source",
        "child_source_state_sha256": "same-source",
        "learning_state_inherited": True,
        "replay_inherited": True,
        "optimizer_state_inherited": True,
        "rng_state_inherited": True,
    }
