"""Frozen Core-V1 phase and beta schedule for three-critic DSRL."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Mapping

import numpy as np


class HierarchyPhase(IntEnum):
    BASE = 0
    RESIDUAL = 1
    JOINT = 2


@dataclass(frozen=True)
class UpdateProfile:
    qa_base: int
    qa_joint: int
    qw_base: int
    noise_actor: int
    alpha: int
    residual_actor: int

    def as_dict(self) -> dict[str, int]:
        return {
            "qa_base": self.qa_base,
            "qa_joint": self.qa_joint,
            "qw_base": self.qw_base,
            "noise_actor": self.noise_actor,
            "alpha": self.alpha,
            "residual_actor": self.residual_actor,
        }


DEFAULT_UPDATE_PROFILES: Mapping[HierarchyPhase, UpdateProfile] = {
    HierarchyPhase.BASE: UpdateProfile(20, 0, 10, 20, 20, 0),
    HierarchyPhase.RESIDUAL: UpdateProfile(10, 10, 5, 0, 0, 1),
    HierarchyPhase.JOINT: UpdateProfile(10, 10, 5, 1, 1, 1),
}


@dataclass(frozen=True)
class HierarchySchedule:
    """Schedule measured exclusively in online action-chunk transitions."""

    profile_name: str
    phase_b_steps: int
    phase_r_steps: int
    phase_j_steps: int
    phase_j_enabled: bool
    beta_ramp_steps: int
    beta_target: float
    base_lane_probability: float
    n_envs: int
    # Co-training extension.  Defaults keep the frozen schedules bit-identical:
    # no hold, ramp from 0, and the global DEFAULT_UPDATE_PROFILES.
    beta_hold_steps: int = 0
    beta_floor: float = 0.0
    update_profiles: Mapping[HierarchyPhase, UpdateProfile] | None = None

    def __post_init__(self) -> None:
        integral = {
            "phase_b_steps": self.phase_b_steps,
            "phase_r_steps": self.phase_r_steps,
            "phase_j_steps": self.phase_j_steps,
            "beta_ramp_steps": self.beta_ramp_steps,
            "beta_hold_steps": self.beta_hold_steps,
            "n_envs": self.n_envs,
        }
        for name, value in integral.items():
            if isinstance(value, bool) or int(value) != value:
                raise TypeError(f"{name} must be an integer")
            if name == "n_envs" and int(value) <= 0:
                raise ValueError("n_envs must be positive")
            if name != "n_envs" and int(value) < 0:
                raise ValueError(f"{name} must be non-negative")
        for name in (
            "phase_b_steps",
            "phase_r_steps",
            "phase_j_steps",
            "beta_ramp_steps",
            "beta_hold_steps",
        ):
            if int(getattr(self, name)) % int(self.n_envs) != 0:
                raise ValueError(
                    f"{name} must be divisible by n_envs so a vector batch "
                    "never straddles a phase boundary"
                )
        if self.phase_j_enabled != (self.phase_j_steps > 0):
            raise ValueError(
                "phase_j_enabled must be true iff phase_j_steps is positive"
            )
        if self.phase_r_steps <= 0:
            raise ValueError("phase_r_steps must be positive")
        if self.beta_ramp_steps > self.phase_r_steps:
            raise ValueError("beta_ramp_steps cannot exceed phase_r_steps")
        ramp_batches = self.beta_ramp_steps // self.n_envs
        if ramp_batches == 1:
            raise ValueError(
                "beta_ramp_steps must be zero or span at least two vector batches"
            )
        if self.beta_hold_steps + self.beta_ramp_steps > self.phase_r_steps:
            raise ValueError(
                "beta_hold_steps + beta_ramp_steps cannot exceed phase_r_steps"
            )
        hold_batches = self.beta_hold_steps // self.n_envs
        if hold_batches == 1:
            raise ValueError(
                "beta_hold_steps must be zero or span at least two vector batches"
            )
        if not np.isfinite(self.beta_floor) or not 0 <= self.beta_floor <= self.beta_target:
            raise ValueError("beta_floor must be finite and lie in [0, beta_target]")
        if not np.isfinite(self.beta_target) or not 0 <= self.beta_target <= 1:
            raise ValueError("beta_target must be finite and lie in [0, 1]")
        if not np.isfinite(self.base_lane_probability) or not (
            0 < self.base_lane_probability < 1
        ):
            raise ValueError(
                "base_lane_probability must be finite and lie strictly in (0, 1)"
            )
        if self.update_profiles is not None:
            # A profile may define only the phases it actively customizes (the
            # cotrain profile defines BASE + RESIDUAL); every other phase falls
            # back to the frozen global defaults so update_profile_at always
            # resolves and serialization is complete for all three phases.
            missing = set(DEFAULT_UPDATE_PROFILES) - set(self.update_profiles)
            if missing:
                object.__setattr__(
                    self,
                    "update_profiles",
                    {
                        **self.update_profiles,
                        **{
                            phase: DEFAULT_UPDATE_PROFILES[phase]
                            for phase in missing
                        },
                    },
                )

    @property
    def phase_r_start(self) -> int:
        return int(self.phase_b_steps)

    @property
    def phase_j_start(self) -> int:
        return int(self.phase_b_steps + self.phase_r_steps)

    @property
    def total_steps(self) -> int:
        return int(self.phase_b_steps + self.phase_r_steps + self.phase_j_steps)

    def validate_budget(self, total_timesteps: int) -> None:
        if int(total_timesteps) != self.total_steps:
            raise ValueError(
                "Hierarchy schedule budget mismatch: "
                f"B+R+J={self.total_steps}, requested={int(total_timesteps)}"
            )

    def phase_at(self, completed_online_steps: int) -> HierarchyPhase:
        completed = int(completed_online_steps)
        if completed < 0 or completed % self.n_envs != 0:
            raise ValueError(
                "completed_online_steps must be a non-negative vector-batch boundary"
            )
        if completed < self.phase_r_start:
            return HierarchyPhase.BASE
        if completed < self.phase_j_start or not self.phase_j_enabled:
            return HierarchyPhase.RESIDUAL
        return HierarchyPhase.JOINT

    def beta_at(self, completed_online_steps: int) -> float:
        phase = self.phase_at(completed_online_steps)
        if phase == HierarchyPhase.BASE:
            return 0.0
        if phase == HierarchyPhase.JOINT or self.beta_ramp_steps == 0:
            return float(self.beta_target)
        ramp_batches = self.beta_ramp_steps // self.n_envs
        hold_batches = self.beta_hold_steps // self.n_envs
        batch_index = max(
            0,
            (int(completed_online_steps) - self.phase_r_start) // self.n_envs,
        )
        if batch_index >= hold_batches + ramp_batches - 1:
            return float(self.beta_target)
        if batch_index < hold_batches:
            # R-start hold: β stays at 0 so joint data is collected with the
            # base action while QA_joint warms up (residual gradient ∝ β).
            return 0.0
        progress = (batch_index - hold_batches) / (ramp_batches - 1)
        return float(self.beta_floor + (self.beta_target - self.beta_floor) * progress)

    def base_probability_at(self, completed_online_steps: int) -> float:
        if self.phase_at(completed_online_steps) == HierarchyPhase.BASE:
            return 1.0
        return float(self.base_lane_probability)

    def update_profile_at(self, completed_online_steps: int) -> UpdateProfile:
        profiles = (
            self.update_profiles
            if self.update_profiles is not None
            else DEFAULT_UPDATE_PROFILES
        )
        return profiles[self.phase_at(completed_online_steps)]


def make_hierarchy_schedule(
    profile_name: str,
    *,
    n_envs: int,
    overrides: Mapping[str, object] | None = None,
) -> HierarchySchedule:
    profiles: dict[str, dict[str, object]] = {
        "fresh_frozen_ddim_5m": {
            "phase_b_steps": 2_500_000,
            "phase_r_steps": 2_500_000,
            "phase_j_steps": 0,
            "phase_j_enabled": False,
            "beta_ramp_steps": 50_000,
            "beta_target": 0.1,
            "base_lane_probability": 0.5,
        },
        # 2.5M-total from-scratch profile.  Derived from the frozen
        # fresh_frozen_ddim_5m by proportional scaling that preserves its
        # 50:50 Phase B:R ratio (B=1.25M, R=1.25M).  Every other frozen
        # parameter (beta_ramp, beta_target, lane probability) is unchanged.
        "fresh_frozen_ddim_2p5m": {
            "phase_b_steps": 1_250_000,
            "phase_r_steps": 1_250_000,
            "phase_j_steps": 0,
            "phase_j_enabled": False,
            "beta_ramp_steps": 50_000,
            "beta_target": 0.1,
            "base_lane_probability": 0.5,
        },
        "legacy_dsrl_warmstart_5m": {
            "phase_b_steps": 0,
            "phase_r_steps": 5_000_000,
            "phase_j_steps": 0,
            "phase_j_enabled": False,
            "beta_ramp_steps": 50_000,
            "beta_target": 0.1,
            "base_lane_probability": 0.5,
        },
        # 2.5M-total from-scratch co-training profile (user design decision).
        # Short Phase B (0.5M) builds the base branch AND shadows QA_joint on
        # the same BASE transitions (β=0, so QA_joint's valid action is the
        # base action); the long Phase R (2.0M) co-trains both branches —
        # QA_base→QW→noise/alpha continue at low frequency while QA_joint→
        # residual trains simultaneously, β held at 0 for the first 50k then
        # ramped 0.02→0.1 over the next 50k.  Residual exploration and
        # cross-lane replay are model-level flags (set by the cotrain config),
        # not schedule fields.  Frozen beta_target/lane probability unchanged.
        "fresh_frozen_ddim_2p5m_cotrain": {
            "phase_b_steps": 500_000,
            "phase_r_steps": 2_000_000,
            "phase_j_steps": 0,
            "phase_j_enabled": False,
            "beta_ramp_steps": 50_000,
            "beta_hold_steps": 50_000,
            "beta_floor": 0.02,
            "beta_target": 0.1,
            "base_lane_probability": 0.5,
            "update_profiles": {
                HierarchyPhase.BASE: UpdateProfile(20, 10, 10, 20, 20, 0),
                HierarchyPhase.RESIDUAL: UpdateProfile(5, 5, 2, 1, 1, 1),
            },
        },
        # Walker K4 diagnosis: preserve every Phase-B optimizer budget in R
        # and add residual learning instead of reallocating base updates.
        "fresh_frozen_ddim_2p5m_additive_res": {
            "phase_b_steps": 500_000,
            "phase_r_steps": 2_000_000,
            "phase_j_steps": 0,
            "phase_j_enabled": False,
            "beta_ramp_steps": 50_000,
            "beta_hold_steps": 50_000,
            "beta_floor": 0.02,
            "beta_target": 0.1,
            "base_lane_probability": 0.5,
            "update_profiles": {
                HierarchyPhase.BASE: UpdateProfile(20, 10, 10, 20, 20, 0),
                HierarchyPhase.RESIDUAL: UpdateProfile(20, 10, 10, 20, 20, 4),
            },
        },
        # Matched development control.  Its Phase-B boundary is deliberately
        # beyond the 750k/1M gates so an exact 500k resume keeps collecting and
        # updating only the base branch throughout the paired comparison.
        "fresh_frozen_ddim_2p5m_base_continue": {
            "phase_b_steps": 2_000_000,
            "phase_r_steps": 500_000,
            "phase_j_steps": 0,
            "phase_j_enabled": False,
            "beta_ramp_steps": 50_000,
            "beta_hold_steps": 50_000,
            "beta_floor": 0.02,
            "beta_target": 0.1,
            "base_lane_probability": 0.5,
            "update_profiles": {
                HierarchyPhase.BASE: UpdateProfile(20, 10, 10, 20, 20, 0),
            },
        },
    }
    if profile_name not in profiles:
        raise ValueError(
            f"Unknown hierarchy schedule profile {profile_name!r}; "
            f"expected one of {sorted(profiles)}"
        )
    values = dict(profiles[profile_name])
    # The co-training fields are accepted as overrides on every profile: their
    # defaults (0 / 0.0) reproduce the frozen beta ramp exactly, so a frozen
    # profile remains bit-identical unless hold/floor are explicitly set.
    # update_profiles itself stays profile-defined and is never overridable.
    override_keys = set(values) | {"beta_hold_steps", "beta_floor"}
    if overrides:
        unknown = set(overrides) - override_keys
        if unknown:
            raise ValueError(f"Unknown hierarchy schedule overrides: {sorted(unknown)}")
        values.update(overrides)
    return HierarchySchedule(
        profile_name=profile_name,
        n_envs=int(n_envs),
        **values,
    )
