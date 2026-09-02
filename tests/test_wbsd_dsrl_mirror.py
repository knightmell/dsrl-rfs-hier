from __future__ import annotations

import numpy as np
import torch
from torch import nn

from wbsd_dsrl_mirror import (
    METHOD_ACTOR_LOCAL,
    METHOD_GAUSSIAN_NATIVE,
    TEACHER_NATIVE_ONLINE,
    TEACHER_TARGET,
    DSRLMirrorAdapter,
    build_candidate_pools,
    evaluate_mirror_pool,
    mirror_state_metrics,
)


class _ToyActor:
    def get_action_dist_params(self, observations):
        mean = observations[:, :2] * 0.1
        log_std = torch.zeros_like(mean)
        return mean, log_std, {}


class _ToyPolicy:
    def scale_action(self, values):
        return np.asarray(values, dtype=np.float32) * 0.5

    def unscale_action(self, values):
        return np.asarray(values, dtype=np.float32) * 2.0


class _ToyTwinQ(nn.Module):
    def __init__(self, offset: float):
        super().__init__()
        self.offset = nn.Parameter(torch.tensor(float(offset)))

    def forward(self, observations, actions):
        base = observations.sum(dim=1, keepdim=True) + actions.sum(
            dim=1, keepdim=True
        )
        return base + self.offset, base + self.offset + 0.25

    def set_training_mode(self, mode: bool):
        self.train(mode)


class _ToyDiffusion(nn.Module):
    def forward(self, observations, decoder, return_numpy=False):
        del observations, return_numpy
        return decoder


class _ToyDSRL:
    def __init__(self):
        self.actor = _ToyActor()
        self.policy = _ToyPolicy()
        self.critic = _ToyTwinQ(1.0)
        self.critic_target = _ToyTwinQ(10.0)
        self.critic_noise = _ToyTwinQ(2.0)
        self.diffusion_policy = _ToyDiffusion()
        self.diffusion_act_chunk = 1
        self.diffusion_act_dim = 2
        self.num_timesteps = 200_000


def test_adapter_exposes_dsrl_native_online_teacher_and_target_separately():
    adapter = DSRLMirrorAdapter(_ToyDSRL())
    assert adapter.qa_native is adapter.model.critic
    assert adapter.qa_target is adapter.model.critic_target
    assert adapter.qw is adapter.model.critic_noise
    assert adapter.action_dim_flat == 2


def test_candidate_pools_are_reproducible_and_keep_exact_gaussian_unbounded():
    adapter = DSRLMirrorAdapter(_ToyDSRL())
    observations = np.ones((3, 2), dtype=np.float32)
    first = build_candidate_pools(adapter, observations, pool_size=16, seed=17)
    second = build_candidate_pools(adapter, observations, pool_size=16, seed=17)
    assert set(first) == {METHOD_ACTOR_LOCAL, METHOD_GAUSSIAN_NATIVE}
    assert np.array_equal(first[METHOD_ACTOR_LOCAL], second[METHOD_ACTOR_LOCAL])
    assert np.array_equal(
        first[METHOD_GAUSSIAN_NATIVE], second[METHOD_GAUSSIAN_NATIVE]
    )
    assert np.max(np.abs(first[METHOD_ACTOR_LOCAL])) < 1.0
    assert np.max(np.abs(first[METHOD_GAUSSIAN_NATIVE])) > 1.0


def test_pool_evaluation_reports_online_target_and_qw_heads_without_mutation():
    adapter = DSRLMirrorAdapter(_ToyDSRL())
    observations = np.ones((2, 2), dtype=np.float32)
    pool = np.zeros((2, 4, 2), dtype=np.float32)
    result, counts = evaluate_mirror_pool(
        adapter,
        observations,
        pool,
        device="cpu",
        batch_candidates=32,
    )
    assert result[TEACHER_NATIVE_ONLINE].shape == (2, 2, 4)
    assert result[TEACHER_TARGET].shape == (2, 2, 4)
    assert result["qw"].shape == (2, 2, 4)
    assert np.all(result[TEACHER_TARGET] > result[TEACHER_NATIVE_ONLINE])
    assert counts == {
        "ddim_decode_candidates": 8,
        "qa_online_candidates": 8,
        "qa_target_candidates": 8,
        "qw_candidates": 8,
    }


def test_mirror_rows_keep_teacher_kind_as_an_aggregation_dimension():
    noise = np.array([[[0.0], [0.5]]], dtype=np.float32)
    actions = noise.copy()
    teacher = np.array([[[1.0, 3.0]], [[1.5, 2.5]]], dtype=np.float32)
    student = np.array([[[1.0, 2.0]], [[1.0, 2.0]]], dtype=np.float32)
    rows = mirror_state_metrics(
        checkpoint_step=200_000,
        proposal_seed=1101,
        method=METHOD_GAUSSIAN_NATIVE,
        teacher_kind=TEACHER_NATIVE_ONLINE,
        state_ids=np.array([7]),
        noise=noise,
        actions=actions,
        teacher_heads=teacher,
        student_heads=student,
        prefixes=[1, 2],
    )
    assert len(rows) == 2
    assert {row["teacher_kind"] for row in rows} == {TEACHER_NATIVE_ONLINE}
    assert {row["model_family"] for row in rows} == {"matched_dsrl_na"}
