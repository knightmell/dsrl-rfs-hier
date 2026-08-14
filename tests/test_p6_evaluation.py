from __future__ import annotations

import csv
import json
import random

import numpy as np
import torch
from torch import nn

from p6_evaluation import evaluate_exact_episodes, persist_evaluation
from p6_runtime import capture_rng_state, restore_rng_state, seed_all


class VariableLengthEnvironment:
    def __init__(self):
        self.seed = None
        self.limit = None
        self.steps = 0

    def reset(self, seed=None):
        self.seed = int(seed)
        # Even seeds hit a short 2-chunk limit (early fall); odd seeds run the
        # full 3-chunk length.  The saturating policy keys off observation[0],
        # which is pinned to the seed for the whole episode, so the residual
        # signature lines up exactly with the early-fall outcome split.
        self.limit = 2 if self.seed % 2 == 0 else 3
        self.steps = 0
        return np.array([float(self.seed), 0.0], dtype=np.float32), {}

    def step(self, action):
        self.steps += 1
        terminated = self.steps >= self.limit
        reward = float(np.asarray(action).sum()) + 1.0
        actual = 2 if terminated else 4
        return (
            np.array([float(self.seed), float(self.steps)], dtype=np.float32),
            reward,
            terminated,
            False,
            {
                "nominal_primitive_steps": 4,
                "actual_primitive_steps": actual,
                "early_termination_within_chunk": bool(actual < 4),
            },
        )

    def get_normalized_score(self, raw_return):
        return raw_return / 10.0

    def close(self):
        return None


class StochasticExecutedPolicy:
    def __init__(self):
        self.policy = nn.Sequential(nn.Linear(1, 2), nn.Dropout(0.2))
        self.critic = nn.Linear(3, 1)
        self.policy.train(True)
        self.critic.train(False)

    def predict_diffused(self, observation, deterministic=False):
        del deterministic
        observation = np.asarray(observation)
        assert observation.ndim == 2
        return torch.rand((observation.shape[0], 4)).numpy(), None


class ComponentPolicy(StochasticExecutedPolicy):
    def predict_with_components(
        self,
        observation,
        deterministic=False,
        mode="current_full_hierarchy",
    ):
        del observation, deterministic, mode
        action_base = torch.rand(4).numpy()
        residual_delta = np.full(4, 0.05, dtype=np.float32)
        action_pre_clip = action_base + residual_delta
        action_exec = np.clip(action_pre_clip, -1.0, 1.0)
        return {
            "noise_scaled": action_base.copy(),
            "noise_decoder_input": action_base.reshape(2, 2).copy(),
            "action_base": action_base,
            "residual_pre_tanh": residual_delta.copy(),
            "residual_unit": residual_delta.copy(),
            "action_half_range": np.ones(4, dtype=np.float32),
            "action_residual_delta": residual_delta,
            "action_pre_clip": action_pre_clip,
            "action_exec_unclamped": action_pre_clip,
            "action_exec": action_exec,
        }, None


class SaturatingComponentPolicy(ComponentPolicy):
    def predict_with_components(
        self,
        observation,
        deterministic=False,
        mode="current_full_hierarchy",
    ):
        del deterministic, mode
        observation = np.asarray(observation)
        action_base = torch.rand(4).numpy()
        # Saturate the residual exactly on falling episodes (even env seed ->
        # shorter limit -> early fall): the bang-bang rescue signature.
        saturated = bool(int(observation[0]) % 2 == 0)
        residual_value = np.full(4, 0.995 if saturated else 0.05, np.float32)
        action_pre_clip = action_base + residual_value
        action_exec = np.clip(action_pre_clip, -1.0, 1.0)
        return {
            "noise_scaled": action_base.copy(),
            "noise_decoder_input": action_base.reshape(2, 2).copy(),
            "action_base": action_base,
            "residual_pre_tanh": residual_value.copy(),
            "residual_unit": residual_value.copy(),
            "action_half_range": np.ones(4, dtype=np.float32),
            "action_residual_delta": residual_value,
            "action_pre_clip": action_pre_clip,
            "action_exec_unclamped": action_pre_clip,
            "action_exec": action_exec,
        }, None


def _rng_sample():
    return (
        random.random(),
        np.random.random(),
        torch.rand(3),
    )


def test_eval_reports_residual_saturation_split_by_outcome():
    # Even env seeds (10, 12, 14) hit their short 2-chunk limit early (actual
    # 6 primitives) and saturate the residual; odd seeds (11, 13) run the full
    # 3-chunk length (actual 10 primitives, == the audit limit) with a
    # near-zero residual.  The by_outcome summary must split exactly on that
    # signature.
    result = evaluate_exact_episodes(
        model=SaturatingComponentPolicy(),
        make_environment=VariableLengthEnvironment,
        environment_seeds=[10, 11, 12, 13, 14],
        policy_seed_start=900,
        deterministic=False,
        action_chunk=4,
        max_episode_primitive_steps=10,
        batch_size=1,
        evaluation_mode="current_full_hierarchy",
    )

    assert [row["early_fall"] for row in result["episodes"]] == [
        True,
        False,
        True,
        False,
        True,
    ]
    assert [
        row["residual_tanh_saturation_fraction"] for row in result["episodes"]
    ] == [1.0, 0.0, 1.0, 0.0, 1.0]

    by_outcome = result["summary"]["by_outcome"]
    assert by_outcome["fall"]["episode_count"] == 3.0
    assert by_outcome["fall"]["residual_tanh_saturation_fraction_mean"] == 1.0
    assert by_outcome["healthy"]["episode_count"] == 2.0
    assert by_outcome["healthy"]["residual_tanh_saturation_fraction_mean"] == 0.0
    assert by_outcome["healthy"]["effective_residual_l2_mean"] > 0.0


def test_exact_n_is_batch_size_invariant_and_restores_rng_and_modes():
    model = StochasticExecutedPolicy()
    original_modes = (model.policy.training, model.critic.training)
    seed_all(731)
    rng_state = capture_rng_state()
    expected_rng = _rng_sample()
    restore_rng_state(rng_state)

    result_batch_one = evaluate_exact_episodes(
        model=model,
        make_environment=VariableLengthEnvironment,
        environment_seeds=[10, 11, 12, 13, 14],
        policy_seed_start=900,
        deterministic=False,
        action_chunk=4,
        max_episode_primitive_steps=20,
        batch_size=1,
        chunk_transitions=100,
        nominal_primitive_steps=400,
        actual_primitive_env_steps=390,
        evaluation_mode="current_base_only",
    )
    actual_rng = _rng_sample()
    assert actual_rng[0] == expected_rng[0]
    assert actual_rng[1] == expected_rng[1]
    torch.testing.assert_close(actual_rng[2], expected_rng[2], rtol=0.0, atol=0.0)
    assert (model.policy.training, model.critic.training) == original_modes

    result_batch_four = evaluate_exact_episodes(
        model=model,
        make_environment=VariableLengthEnvironment,
        environment_seeds=[10, 11, 12, 13, 14],
        policy_seed_start=900,
        deterministic=False,
        action_chunk=4,
        max_episode_primitive_steps=20,
        batch_size=4,
        chunk_transitions=100,
        nominal_primitive_steps=400,
        actual_primitive_env_steps=390,
        evaluation_mode="current_base_only",
    )

    assert result_batch_one["exact_episode_count"] == 5
    assert result_batch_four["exact_episode_count"] == 5
    assert result_batch_one["episodes"] == result_batch_four["episodes"]
    assert result_batch_one["summary"] == result_batch_four["summary"]
    assert [row["environment_seed"] for row in result_batch_one["episodes"]] == [
        10,
        11,
        12,
        13,
        14,
    ]
    assert [row["policy_seed"] for row in result_batch_one["episodes"]] == [
        900,
        901,
        902,
        903,
        904,
    ]
    assert all(row["early_fall"] for row in result_batch_one["episodes"])
    assert result_batch_one["summary"]["d4rl_score_mean"] == (
        result_batch_one["summary"]["raw_return_mean"] * 10.0
    )


class RecordingWriter:
    def __init__(self):
        self.scalars = []
        self.flushed = False

    def add_scalar(self, name, value, step):
        self.scalars.append((name, value, step))

    def flush(self):
        self.flushed = True


def test_evaluation_persists_atomic_json_csv_tensorboard_and_components(tmp_path):
    result = evaluate_exact_episodes(
        model=ComponentPolicy(),
        make_environment=VariableLengthEnvironment,
        environment_seeds=[21, 22],
        policy_seed_start=1200,
        deterministic=True,
        action_chunk=4,
        max_episode_primitive_steps=20,
        batch_size=2,
        chunk_transitions=50,
        nominal_primitive_steps=200,
        actual_primitive_env_steps=196,
    )
    writer = RecordingWriter()
    output_prefix = tmp_path / "eval" / "milestone_50"

    persist_evaluation(
        result,
        output_prefix=output_prefix,
        tensorboard_writer=writer,
        tensorboard_tag="eval/online",
    )

    loaded_json = json.loads(output_prefix.with_suffix(".json").read_text())
    assert loaded_json["exact_episode_count"] == 2
    with output_prefix.with_suffix(".csv").open(newline="") as input_file:
        rows = list(csv.DictReader(input_file))
    assert len(rows) == 2
    assert float(rows[0]["action_residual_delta_l2"]) > 0
    assert writer.flushed is True
    assert {
        name for name, _, _ in writer.scalars
    } >= {
        "eval/online/raw_return_mean",
        "eval/online/d4rl_score_mean",
        "eval/online/chunk_transitions",
        "eval/online/nominal_primitive_steps",
        "eval/online/actual_primitive_env_steps",
    }
    assert all(step == 50 for _, _, step in writer.scalars)
    assert not list(output_prefix.parent.glob("*.tmp"))
