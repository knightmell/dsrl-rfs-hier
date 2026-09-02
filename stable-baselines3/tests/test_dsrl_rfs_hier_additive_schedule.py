"""Schedule contracts for the Walker K4 additive-residual diagnosis."""

from stable_baselines3.dsrl.hierarchy_schedule import (
    HierarchyPhase,
    make_hierarchy_schedule,
)


BASE_PROFILE = {
    "qa_base": 20,
    "qa_joint": 10,
    "qw_base": 10,
    "noise_actor": 20,
    "alpha": 20,
    "residual_actor": 0,
}


def test_additive_res_profile_preserves_base_budget_and_adds_residual_four():
    schedule = make_hierarchy_schedule(
        "fresh_frozen_ddim_2p5m_additive_res",
        n_envs=10,
    )

    assert schedule.phase_b_steps == 500_000
    assert schedule.phase_r_steps == 2_000_000
    assert schedule.total_steps == 2_500_000
    assert schedule.beta_hold_steps == 50_000
    assert schedule.beta_ramp_steps == 50_000
    assert schedule.beta_floor == 0.02
    assert schedule.beta_target == 0.1
    assert schedule.base_lane_probability == 0.5
    assert schedule.update_profiles[HierarchyPhase.BASE].as_dict() == BASE_PROFILE
    assert schedule.update_profiles[HierarchyPhase.RESIDUAL].as_dict() == {
        **BASE_PROFILE,
        "residual_actor": 4,
    }


def test_base_continue_profile_remains_base_through_one_million_chunks():
    schedule = make_hierarchy_schedule(
        "fresh_frozen_ddim_2p5m_base_continue",
        n_envs=10,
    )

    assert schedule.total_steps == 2_500_000
    assert schedule.phase_b_steps == 2_000_000
    assert schedule.phase_at(500_000) == HierarchyPhase.BASE
    assert schedule.phase_at(750_000) == HierarchyPhase.BASE
    assert schedule.phase_at(1_000_000) == HierarchyPhase.BASE
    assert schedule.update_profile_at(750_000).as_dict() == BASE_PROFILE
