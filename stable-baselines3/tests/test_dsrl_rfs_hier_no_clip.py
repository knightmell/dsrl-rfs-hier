from __future__ import annotations

import pytest
import torch

from stable_baselines3.dsrl.hierarchical_replay_buffer import BranchMode
from tests.three_critic_test_utils import make_model, populate_branch


def _base_sample(model):
    populate_branch(model, BranchMode.BASE, rows=2)
    return model._sample_branch(BranchMode.BASE, 2)


def test_noise_actor_clipping_remains_enabled_by_default(monkeypatch):
    model, _ = make_model()
    sample = _base_sample(model)
    calls = []
    original = torch.nn.utils.clip_grad_norm_

    def record(parameters, max_norm, *args, **kwargs):
        calls.append(float(max_norm))
        return original(parameters, max_norm, *args, **kwargs)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", record)
    model._update_alpha_and_noise_once(sample, update_alpha=False)
    assert model.noise_actor_gradient_clipping is True
    assert calls == [1.0]


def test_noise_actor_no_clip_skips_clip_and_reports_raw_post_norm(monkeypatch):
    model, _ = make_model(noise_actor_gradient_clipping=False)
    sample = _base_sample(model)

    def forbidden(*_args, **_kwargs):
        pytest.fail("clip_grad_norm_ must not be called in no-clip mode")

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", forbidden)
    _loss, _alpha_loss, pre_norm, post_norm = model._update_alpha_and_noise_once(
        sample, update_alpha=False
    )
    assert model.noise_actor_gradient_clipping is False
    assert pre_norm > 0.0
    assert post_norm == pytest.approx(pre_norm, rel=0, abs=0)
