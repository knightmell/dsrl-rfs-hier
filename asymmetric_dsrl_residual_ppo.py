"""Invariants for asymmetric DSRL-NA + per-step residual PPO.

The live DSRL learner and the residual PPO learner never share returns,
replay, parameters, or gradients.  A separate inference shell receives a
one-way actor snapshot only at a completed PPO-update boundary.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch


def _unique_parameters(modules: Mapping[str, torch.nn.Module]):
    seen: set[int] = set()
    for module in modules.values():
        for parameter in module.parameters():
            identifier = id(parameter)
            if identifier not in seen:
                seen.add(identifier)
                yield parameter


def module_state_hash(modules: Mapping[str, torch.nn.Module]) -> str:
    digest = hashlib.sha256()
    for module_name in sorted(modules):
        digest.update(module_name.encode())
        for name, value in sorted(modules[module_name].state_dict().items()):
            tensor = value.detach().cpu().contiguous()
            digest.update(name.encode())
            digest.update(str(tensor.dtype).encode())
            digest.update(json.dumps(list(tensor.shape)).encode())
            digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def clear_module_gradients(modules: Mapping[str, torch.nn.Module]) -> None:
    for parameter in _unique_parameters(modules):
        parameter.grad = None


def assert_no_module_gradients(
    modules: Mapping[str, torch.nn.Module],
    *,
    owner: str,
) -> None:
    offenders = [
        name
        for name, module in modules.items()
        if any(parameter.grad is not None for parameter in module.parameters())
    ]
    if offenders:
        raise RuntimeError(f"{owner} unexpectedly received gradients: {offenders}")


def assert_parameter_storage_disjoint(
    left: torch.nn.Module,
    right: torch.nn.Module,
) -> None:
    left_storage = {
        parameter.untyped_storage().data_ptr()
        for parameter in left.parameters()
    }
    right_storage = {
        parameter.untyped_storage().data_ptr()
        for parameter in right.parameters()
    }
    overlap = left_storage & right_storage
    if overlap:
        raise RuntimeError("Live and snapshot actors share parameter storage")


def copy_actor_snapshot(
    *,
    live_actor: torch.nn.Module,
    snapshot_actor: torch.nn.Module,
) -> str:
    assert_parameter_storage_disjoint(live_actor, snapshot_actor)
    snapshot_actor.load_state_dict(live_actor.state_dict(), strict=True)
    snapshot_actor.eval()
    snapshot_actor.requires_grad_(False)
    live_hash = module_state_hash({"actor": live_actor})
    snapshot_hash = module_state_hash({"actor": snapshot_actor})
    if snapshot_hash != live_hash:
        raise RuntimeError("Actor snapshot copy is not exact")
    return snapshot_hash


@dataclass
class ActorSnapshotState:
    version: int = 0
    actor_hash: str = ""

    def commit(
        self,
        *,
        live_actor: torch.nn.Module,
        snapshot_actor: torch.nn.Module,
    ) -> str:
        self.actor_hash = copy_actor_snapshot(
            live_actor=live_actor,
            snapshot_actor=snapshot_actor,
        )
        self.version += 1
        return self.actor_hash

    def assert_unchanged(self, snapshot_actor: torch.nn.Module) -> None:
        actual = module_state_hash({"actor": snapshot_actor})
        if actual != self.actor_hash:
            raise RuntimeError(
                f"Snapshot actor changed within version {self.version}"
            )


@dataclass(frozen=True)
class AsymmetricCycle:
    index: int
    base_chunk_transitions: int
    joint_primitive_transitions: int

    @property
    def joint_equivalent_chunks(self) -> int:
        if self.joint_primitive_transitions % 4:
            raise RuntimeError("Joint primitive budget is not H=4 aligned")
        return self.joint_primitive_transitions // 4

    @property
    def total_equivalent_chunks(self) -> int:
        return self.base_chunk_transitions + self.joint_equivalent_chunks


def build_three_seven_cycles(
    *,
    total_equivalent_chunks: int,
    cycle_equivalent_chunks: int,
    action_chunk: int,
    base_fraction: float,
) -> list[AsymmetricCycle]:
    """Build equal cycles with exact base/joint interaction accounting."""

    total = int(total_equivalent_chunks)
    per_cycle = int(cycle_equivalent_chunks)
    horizon = int(action_chunk)
    if total <= 0 or per_cycle <= 0 or total % per_cycle:
        raise ValueError("Total budget must be divisible by cycle budget")
    if horizon <= 0 or not (0.0 < base_fraction < 1.0):
        raise ValueError("Invalid action chunk or base fraction")
    base_chunks_float = per_cycle * float(base_fraction)
    base_chunks = int(round(base_chunks_float))
    if not np.isclose(base_chunks_float, base_chunks):
        raise ValueError("Each cycle must contain an integral base budget")
    joint_chunks = per_cycle - base_chunks
    if base_chunks <= 0 or joint_chunks <= 0:
        raise ValueError("Each cycle must exercise both streams")
    return [
        AsymmetricCycle(
            index=index,
            base_chunk_transitions=base_chunks,
            joint_primitive_transitions=joint_chunks * horizon,
        )
        for index in range(total // per_cycle)
    ]


@dataclass
class InteractionCounters:
    action_chunk: int
    base_chunk_transitions: int = 0
    base_actual_primitive_steps: int = 0
    joint_primitive_transitions: int = 0
    completed_cycles: int = 0

    def record_cycle(
        self,
        *,
        cycle: AsymmetricCycle,
        base_actual_primitive_steps: int,
    ) -> None:
        actual = int(base_actual_primitive_steps)
        base_nominal = cycle.base_chunk_transitions * int(self.action_chunk)
        if actual < 0 or actual > base_nominal:
            raise ValueError("Invalid base-stream primitive counter")
        self.base_chunk_transitions += cycle.base_chunk_transitions
        self.base_actual_primitive_steps += actual
        self.joint_primitive_transitions += cycle.joint_primitive_transitions
        self.completed_cycles += 1

    @property
    def joint_equivalent_chunks(self) -> int:
        horizon = int(self.action_chunk)
        if self.joint_primitive_transitions % horizon:
            raise RuntimeError("Joint primitive counter is not chunk aligned")
        return self.joint_primitive_transitions // horizon

    @property
    def total_equivalent_chunks(self) -> int:
        return self.base_chunk_transitions + self.joint_equivalent_chunks

    @property
    def nominal_primitive_steps(self) -> int:
        return self.total_equivalent_chunks * int(self.action_chunk)

    @property
    def actual_primitive_env_steps(self) -> int:
        return self.base_actual_primitive_steps + self.joint_primitive_transitions

    def to_dict(self) -> dict[str, int]:
        nominal = self.nominal_primitive_steps
        actual = self.actual_primitive_env_steps
        if actual > nominal:
            raise RuntimeError("Actual primitive steps exceed nominal budget")
        return {
            "completed_cycles": self.completed_cycles,
            "base_chunk_transitions": self.base_chunk_transitions,
            "base_nominal_primitive_steps": (
                self.base_chunk_transitions * int(self.action_chunk)
            ),
            "base_actual_primitive_steps": self.base_actual_primitive_steps,
            "joint_primitive_transitions": self.joint_primitive_transitions,
            "joint_equivalent_chunks": self.joint_equivalent_chunks,
            "total_equivalent_chunks": self.total_equivalent_chunks,
            "nominal_primitive_steps": nominal,
            "actual_primitive_env_steps": actual,
            "skipped_primitive_steps_due_to_base_termination": nominal - actual,
        }


class StreamIsolationGuard:
    """Hash and gradient checks around the two sequential update blocks."""

    def __init__(
        self,
        *,
        live_modules: Mapping[str, torch.nn.Module],
        snapshot_modules: Mapping[str, torch.nn.Module],
        residual_policy: torch.nn.Module,
    ) -> None:
        self.live_modules = dict(live_modules)
        self.snapshot_modules = dict(snapshot_modules)
        self.residual_modules = {"residual_policy": residual_policy}

    def before_base(self) -> dict[str, str]:
        clear_module_gradients(self.residual_modules)
        clear_module_gradients(self.snapshot_modules)
        return {
            "residual": module_state_hash(self.residual_modules),
            "snapshot": module_state_hash(self.snapshot_modules),
        }

    def after_base(self, before: Mapping[str, str]) -> None:
        if module_state_hash(self.residual_modules) != before["residual"]:
            raise RuntimeError("Base-only update changed residual PPO")
        if module_state_hash(self.snapshot_modules) != before["snapshot"]:
            raise RuntimeError("Base-only update changed the frozen actor snapshot")
        assert_no_module_gradients(
            self.residual_modules,
            owner="residual PPO during base-only update",
        )
        assert_no_module_gradients(
            self.snapshot_modules,
            owner="actor snapshot during base-only update",
        )

    def before_joint(self) -> dict[str, str]:
        clear_module_gradients(self.live_modules)
        clear_module_gradients(self.snapshot_modules)
        return {
            "live": module_state_hash(self.live_modules),
            "snapshot": module_state_hash(self.snapshot_modules),
        }

    def after_joint(self, before: Mapping[str, str]) -> None:
        if module_state_hash(self.live_modules) != before["live"]:
            raise RuntimeError("Residual PPO update changed live DSRL")
        if module_state_hash(self.snapshot_modules) != before["snapshot"]:
            raise RuntimeError("Residual PPO update changed the actor snapshot")
        assert_no_module_gradients(
            self.live_modules,
            owner="live DSRL during residual PPO update",
        )
        assert_no_module_gradients(
            self.snapshot_modules,
            owner="actor snapshot during residual PPO update",
        )
