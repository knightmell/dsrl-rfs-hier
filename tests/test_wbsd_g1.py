from __future__ import annotations

import numpy as np

from wbsd_g1 import (
    METHOD_CURRENT,
    PRIMARY_VIEW,
    _pairwise_sign_agreement,
    build_reachable_mix,
    paired_state_bootstrap_ci,
    pairwise_rms,
    replace_anchor,
    state_metrics,
)


def test_reachable_mix_has_shared_anchor_nested_alternating_sources():
    current = np.arange(1 * 6 * 2, dtype=np.float32).reshape(1, 6, 2) / 20
    prior = np.arange(100, 112, dtype=np.float32).reshape(1, 6, 2) / 20
    mixed, source = build_reachable_mix(current, prior)
    assert np.array_equal(mixed[:, 0], current[:, 0])
    assert np.allclose(mixed[:, 1], np.tanh(prior[:, 1]))
    assert np.array_equal(mixed[:, 2], current[:, 1])
    assert np.allclose(mixed[:, 3], np.tanh(prior[:, 2]))
    assert source.tolist() == [0, 1, 0, 1, 0, 1]
    assert np.max(np.abs(mixed)) < 1.0


def test_replace_anchor_supports_pool_and_head_arrays():
    primary = np.ones((2, 3, 4), dtype=np.float32)
    anchor = np.full((2, 3, 4), 7.0, dtype=np.float32)
    replaced = replace_anchor(primary, anchor)
    assert np.all(replaced[:, 0] == 7.0)
    assert np.all(replaced[:, 1:] == 1.0)
    primary_heads = np.ones((2, 2, 3), dtype=np.float32)
    anchor_heads = np.full((2, 2, 3), 9.0, dtype=np.float32)
    replaced_heads = replace_anchor(primary_heads, anchor_heads, candidate_axis=2)
    assert np.all(replaced_heads[:, :, 0] == 9.0)
    assert np.all(replaced_heads[:, :, 1:] == 1.0)


def test_pairwise_rms_matches_two_point_l2_distance():
    values = np.array([[[0.0, 0.0], [3.0, 4.0]]])
    assert np.allclose(pairwise_rms(values), [5.0])
    assert np.array_equal(pairwise_rms(values[:, :1]), [0.0])


def test_pairwise_sign_agreement_excludes_ties():
    first = np.array([[3.0, 2.0, 1.0], [1.0, 1.0, 0.0]])
    second = np.array([[4.0, 2.0, 0.0], [0.0, 0.0, 1.0]])
    agreement, counts = _pairwise_sign_agreement(first, second)
    assert agreement[0] == 1.0
    assert counts.tolist() == [3, 2]
    assert agreement[1] == 0.0


def test_state_metrics_uses_conservative_teacher_and_student_selection():
    noise = np.array([[[0.0], [0.2], [0.4]]], dtype=np.float32)
    actions = noise.copy()
    teacher = np.array(
        [
            [[1.0, 4.0, 3.0]],
            [[1.5, 3.5, 2.5]],
        ],
        dtype=np.float32,
    )
    student = np.array(
        [
            [[1.0, 2.0, 5.0]],
            [[1.2, 1.8, 4.0]],
        ],
        dtype=np.float32,
    )
    rows = state_metrics(
        checkpoint_step=100000,
        proposal_seed=1101,
        method=METHOD_CURRENT,
        view=PRIMARY_VIEW,
        state_ids=np.array([17]),
        noise=noise,
        actions=actions,
        teacher_heads=teacher,
        student_heads=student,
        prefixes=[1, 2, 3],
    )
    final = rows[-1]
    assert final["oracle_index"] == 1
    assert final["selected_index"] == 2
    assert final["oracle_lift"] == 2.5
    assert final["selected_lift"] == 1.5
    assert final["predicted_lift"] == 3.0
    assert final["selector_capture_raw"] == 0.6
    assert final["direct_selection_eligible"] == 1


def test_state_metrics_k1_is_finite_and_has_zero_lifts():
    pool = np.zeros((2, 1, 2), dtype=np.float32)
    heads = np.ones((2, 2, 1), dtype=np.float32)
    rows = state_metrics(
        checkpoint_step=1,
        proposal_seed=1,
        method=METHOD_CURRENT,
        view=PRIMARY_VIEW,
        state_ids=np.array([0, 1]),
        noise=pool,
        actions=pool,
        teacher_heads=heads,
        student_heads=heads,
        prefixes=[1],
    )
    assert all(row["oracle_lift"] == 0.0 for row in rows)
    assert all(row["selected_lift"] == 0.0 for row in rows)
    assert all(row["finite"] == 1 for row in rows)


def test_exact_prior_boundary_is_not_mislabelled_as_tanh_saturation():
    noise = np.array([[[0.99], [2.0], [-3.0]]], dtype=np.float32)
    heads = np.ones((2, 1, 3), dtype=np.float32)
    rows = state_metrics(
        checkpoint_step=1,
        proposal_seed=1,
        method="g_prior_exact",
        view=PRIMARY_VIEW,
        state_ids=np.array([0]),
        noise=noise,
        actions=np.zeros_like(noise),
        teacher_heads=heads,
        student_heads=heads,
        prefixes=[3],
        tanh_generated=np.array([True, False, False]),
    )
    assert rows[0]["tanh_saturation_fraction"] == 1.0
    assert rows[0]["tanh_generated_candidate_fraction"] == 1 / 3
    assert rows[0]["scaled_boundary_or_outside_fraction"] == 1.0
    assert rows[0]["out_of_envelope_fraction"] == 2 / 3


def test_cluster_bootstrap_averages_seed_repeats_within_state():
    matrix = np.array([[1.0, 3.0], [5.0, 7.0], [9.0, 11.0]])
    lower, upper = paired_state_bootstrap_ci(
        matrix, repetitions=500, seed=4, statistic="mean"
    )
    # State averages are 2, 6, 10; every resampled mean remains in that range.
    assert 2.0 <= lower <= upper <= 10.0


def test_mix_is_deterministic_and_prefix_stable():
    rng = np.random.default_rng(12)
    current = rng.normal(size=(3, 64, 2)).astype(np.float32)
    prior = rng.normal(size=(3, 64, 2)).astype(np.float32)
    first, first_source = build_reachable_mix(current, prior)
    second, second_source = build_reachable_mix(current, prior)
    assert np.array_equal(first, second)
    assert np.array_equal(first_source, second_source)
    assert np.array_equal(first[:, :16], second[:, :16])
