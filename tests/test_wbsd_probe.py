from __future__ import annotations

import random

import numpy as np
import pytest
import torch
from torch import nn

from wbsd_probe import (
    capture_global_rng_state,
    canonical_module_state_hash,
    counted_candidate_calls,
    decoder_to_scaled,
    evaluate_candidate_batch,
    flatten_state_particles,
    global_rng_states_equal,
    isolated_global_rng,
    load_state_bank,
    nested_prefix,
    prefix_index_hashes,
    sample_current_actor,
)


class _ToyActor:
    def get_action_dist_params(self, observations):
        batch = observations.shape[0]
        mean = torch.zeros(batch, 3, dtype=observations.dtype)
        log_std = torch.zeros_like(mean)
        return mean, log_std, {}


class _ToyPolicy:
    def scale_action(self, values):
        return np.asarray(values) * 0.5


class _ToyModel:
    actor = _ToyActor()
    policy = _ToyPolicy()
    action_dim_flat = 3


class _CountingQ(nn.Module):
    def __init__(self, offset: float):
        super().__init__()
        self.offset = float(offset)

    def forward(self, observations, actions):
        return (
            observations.sum(dim=1, keepdim=True)
            + actions.sum(dim=1, keepdim=True)
            + self.offset
        )


class _CountingProbeModel:
    def __init__(self):
        self.qa_base_target = _CountingQ(1.0)
        self.qw_base = _CountingQ(2.0)
        self.decode_calls = 0
        self.decode_candidates = 0

    def _unscale_noise(self, noise):
        return noise

    def _decode_noise_decoder_input(self, observations, decoder):
        self.decode_calls += 1
        self.decode_candidates += int(decoder.shape[0])
        return decoder


def test_flatten_state_particles_preserves_state_grouping():
    observations = torch.tensor([[10.0], [20.0]])
    particles = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ]
    )
    flat_obs, flat_particles = flatten_state_particles(observations, particles)
    assert flat_obs.tolist() == [[10.0], [10.0], [20.0], [20.0]]
    assert flat_particles.tolist() == [
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
        [7.0, 8.0],
    ]


def test_nested_prefix_is_prefix_of_the_same_pool():
    pool = torch.arange(2 * 5 * 3, dtype=torch.float32).reshape(2, 5, 3)
    prefix = nested_prefix(pool, 3)
    assert prefix.shape == (2, 3, 3)
    assert torch.equal(prefix, pool[:, :3])
    with pytest.raises(ValueError):
        nested_prefix(pool, 0)


def test_current_actor_sampling_is_deterministic_for_a_seed_and_bounded():
    observations = torch.zeros(2, 4)
    first_generator = torch.Generator(device="cpu").manual_seed(7)
    second_generator = torch.Generator(device="cpu").manual_seed(7)
    first = sample_current_actor(_ToyModel(), observations, 4, first_generator)
    second = sample_current_actor(_ToyModel(), observations, 4, second_generator)
    assert torch.equal(first, second)
    assert first.shape == (2, 4, 3)
    assert torch.all(first.abs() < 1.0)


def test_decoder_to_scaled_uses_policy_scale_action():
    decoder = torch.tensor([[2.0, -4.0, 0.0]])
    scaled = decoder_to_scaled(_ToyModel(), decoder)
    assert scaled.tolist() == [[1.0, -2.0, 0.0]]


def test_candidate_evaluation_counts_actual_decode_qa_and_qw_calls():
    model = _CountingProbeModel()
    observations = torch.zeros(2, 4)
    particles = torch.zeros(2, 2, 3)
    with counted_candidate_calls(model) as counts:
        result = evaluate_candidate_batch(model, observations, particles)
    assert result["base_actions"].shape == (4, 3)
    assert counts == {
        "ddim_decode_calls": 1,
        "ddim_decode_candidates": 4,
        "qa_target_calls": 1,
        "qa_target_candidates": 4,
        "qw_student_calls": 1,
        "qw_student_candidates": 4,
    }
    assert model.decode_calls == 1
    assert model.decode_candidates == 4


def test_isolated_global_rng_restores_python_numpy_and_torch():
    random.seed(19)
    np.random.seed(19)
    torch.manual_seed(19)
    before = capture_global_rng_state()
    with isolated_global_rng():
        random.random()
        np.random.random()
        torch.rand(5)
    after = capture_global_rng_state()
    assert global_rng_states_equal(before, after)


def test_prefix_index_hashes_are_method_independent_and_nested():
    current = prefix_index_hashes(64)
    prior = prefix_index_hashes(64)
    assert current == prior
    assert list(current) == ["1", "2", "4", "8", "16", "32", "64"]
    assert current["2"] != current["4"]


def test_load_state_bank_is_evenly_stratified_and_hashed(tmp_path):
    observations = np.arange(4 * 2 * 3, dtype=np.float64).reshape(4, 2, 3)
    source = tmp_path / "prefill.npz"
    np.savez(source, observations=observations)
    selected, metadata = load_state_bank(source, 4)
    assert selected.dtype == np.float32
    assert selected.shape == (4, 3)
    assert metadata["available_states"] == 8
    assert metadata["selected_states"] == 4
    # The production hash includes JSON shape; verify the recorded hash is
    # stable by comparing two independent loads instead of duplicating the
    # implementation here.
    selected_again, metadata_again = load_state_bank(source, 4)
    assert np.array_equal(selected, selected_again)
    assert metadata["selected_sha256"] == metadata_again["selected_sha256"]


def test_canonical_module_state_hash_changes_when_state_changes():
    module = nn.Linear(2, 2)
    first = canonical_module_state_hash(module)
    with torch.no_grad():
        module.weight[0, 0] += 1.0
    second = canonical_module_state_hash(module)
    assert first != second
