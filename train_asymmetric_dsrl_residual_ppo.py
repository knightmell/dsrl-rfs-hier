"""10k wiring canary for asymmetric DSRL-NA + per-step residual PPO.

This is intentionally a new algorithm line.  The base stream is the original
DSRL-NA learner and receives only zero-residual action-chunk transitions.  The
joint stream is on-policy primitive-step PPO and reads an immutable actor
snapshot.  The 10k online budget is split 30/70 in ten safe-boundary cycles.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / "dppo"), str(ROOT / "stable-baselines3")]

from asymmetric_dsrl_residual_ppo import (  # noqa: E402
    ActorSnapshotState,
    InteractionCounters,
    StreamIsolationGuard,
    assert_parameter_storage_disjoint,
    build_three_seven_cycles,
    module_state_hash,
)
from env_utils import ACTION_CHUNK_EARLY_BREAK  # noqa: E402
from p6_preflight import sha256_file  # noqa: E402
from p6_runtime import (  # noqa: E402
    atomic_write_json,
    capture_rng_state,
    collect_or_load_matched_prefill,
    populate_replay_buffer,
    restore_rng_state,
    seed_all,
)
from p6_train import (  # noqa: E402
    P6ControlDSRL,
    _capture_legacy_warmstart,
    _load_legacy_network,
    _make_hopper_environment,
    _promote_legacy_control,
    _reset_control_optimizers,
    _set_model_inference_mode,
    _verify_fresh_control_warmstart,
)
from per_step_residual_env import FrozenDSRLChunkPlanner  # noqa: E402
from per_step_residual_ppo import CountingPPO  # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback  # noqa: E402
from stable_baselines3.common.env_util import make_vec_env  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv  # noqa: E402
from train_per_step_residual import (  # noqa: E402
    ACTION_CHUNK,
    ACTION_DIMENSION,
    compose_config,
    git_state,
    planner_modules,
)
from train_per_step_residual_ppo import (  # noqa: E402
    ResidualMetricsCallback,
    build_model as build_ppo_model,
    evaluate_exact,
    make_ppo_environment,
    make_training_environment as make_ppo_training_environment,
    write_evaluation,
)
from utils import load_base_policy  # noqa: E402


ALGORITHM = "dsrl_na_asym_per_step_ppo"
CANARY_EQUIVALENT_CHUNKS = 10_000
CYCLE_EQUIVALENT_CHUNKS = 1_000
BASE_FRACTION = 0.3
PPO_ROLLOUT_STEPS = 280
PPO_BATCH_SIZE = 400
PPO_N_EPOCHS = 5
CHECKPOINT_INTERVAL_CYCLES = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--equivalent-chunk-budget",
        type=int,
        default=CANARY_EQUIVALENT_CHUNKS,
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--n-envs", type=int, default=10)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--online-eval-episodes", type=int, default=10)
    parser.add_argument("--final-eval-episodes", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--residual-std", type=float, default=0.05)
    parser.add_argument("--residual-scale", type=float, default=0.1)
    parser.add_argument("--clip-range", type=float, default=0.1)
    parser.add_argument("--target-kl", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--stop-after-cycles", type=int)
    parser.add_argument("--resume-bundle", type=Path)
    parser.add_argument("--deterministic-base", action="store_true")
    return parser.parse_args()


class BasePrimitiveCounterCallback(BaseCallback):
    def __init__(self) -> None:
        super().__init__(verbose=0)
        self.actual_primitive_steps = 0
        self.early_termination_chunks = 0

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            self.actual_primitive_steps += int(info["actual_primitive_steps"])
            self.early_termination_chunks += int(
                bool(info["early_termination_within_chunk"])
            )
        return True


class SnapshotResidualMetricsCallback(ResidualMetricsCallback):
    def __init__(self) -> None:
        super().__init__()
        self.snapshot_versions: set[int] = set()

    def _on_step(self) -> bool:
        if not super()._on_step():
            return False
        self.snapshot_versions.update(
            int(info["base_snapshot_version"])
            for info in self.locals["infos"]
        )
        return True


def _make_base_environment(cfg: Any, *, seed: int):
    return make_vec_env(
        lambda: _make_hopper_environment(
            cfg,
            Path(cfg.normalization_path),
        ),
        n_envs=int(cfg.env.n_envs),
        seed=int(seed),
        vec_env_cls=DummyVecEnv,
    )


def _create_stream_rng(seed: int) -> Mapping[str, Any]:
    seed_all(int(seed))
    return capture_rng_state()


def _validate_arguments(arguments: argparse.Namespace) -> None:
    if arguments.equivalent_chunk_budget != CANARY_EQUIVALENT_CHUNKS:
        raise ValueError(
            "This runner is restricted to the approved 10k wiring canary"
        )
    if arguments.n_envs != 10:
        raise ValueError("The audited 10k asymmetric canary requires n_envs=10")
    if arguments.residual_scale != 0.1:
        raise ValueError("The canary fixes residual_scale=0.1")
    if arguments.stop_after_cycles is not None and not (
        1 <= arguments.stop_after_cycles <= 10
    ):
        raise ValueError("stop-after-cycles must be in [1,10]")
    rollout_primitives = PPO_ROLLOUT_STEPS * arguments.n_envs
    if rollout_primitives != 2_800:
        raise RuntimeError("PPO rollout must contain exactly 2,800 primitives")
    if rollout_primitives % PPO_BATCH_SIZE:
        raise RuntimeError("PPO batch size must divide one rollout exactly")


def _prefill_artifact_path(cfg: Any) -> Path:
    path = Path(str(cfg.p6.prefill_artifact_path)).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def _refresh_joint_observations(
    *,
    model: CountingPPO,
    environment: Any,
) -> None:
    if model._last_obs is None:
        return
    refreshed = np.asarray(
        environment.env_method("refresh_plan"),
        dtype=np.float32,
    )
    expected = (int(environment.num_envs),) + environment.observation_space.shape
    if refreshed.shape != expected:
        raise RuntimeError(
            f"Refreshed PPO observation shape {refreshed.shape} != {expected}"
        )
    model._last_obs = refreshed


def _evaluate_pair(
    *,
    model: CountingPPO,
    cfg: Any,
    planner: FrozenDSRLChunkPlanner,
    seeds: Sequence[int],
    policy_seed_start: int,
    equivalent_chunks: int,
    deterministic: bool,
    residual_scale: float,
) -> dict[str, Any]:
    return evaluate_exact(
        model=model,
        make_environment=lambda: make_ppo_environment(
            cfg,
            planner,
            residual_scale=residual_scale,
            deterministic_base=False,
        ),
        seeds=seeds,
        policy_seed_start=policy_seed_start,
        equivalent_chunks=equivalent_chunks,
        deterministic=deterministic,
    )


def _write_boundary_evaluations(
    *,
    run_directory: Path,
    prefix: str,
    model: CountingPPO,
    cfg: Any,
    planner: FrozenDSRLChunkPlanner,
    seeds: Sequence[int],
    equivalent_chunks: int,
    deterministic: bool,
) -> dict[str, Any]:
    base = _evaluate_pair(
        model=model,
        cfg=cfg,
        planner=planner,
        seeds=seeds,
        policy_seed_start=20_000,
        equivalent_chunks=equivalent_chunks,
        deterministic=deterministic,
        residual_scale=0.0,
    )
    joint = _evaluate_pair(
        model=model,
        cfg=cfg,
        planner=planner,
        seeds=seeds,
        policy_seed_start=20_000,
        equivalent_chunks=equivalent_chunks,
        deterministic=deterministic,
        residual_scale=0.1,
    )
    mode = "deterministic" if deterministic else "stochastic"
    write_evaluation(
        run_directory / "evaluations" / f"{prefix}_base_only_{mode}",
        base,
    )
    write_evaluation(
        run_directory / "evaluations" / f"{prefix}_joint_{mode}",
        joint,
    )
    return {
        "base_only": base["summary"],
        "joint": joint["summary"],
    }


def _save_bundle(
    path: Path,
    *,
    base_model: P6ControlDSRL,
    residual_model: CountingPPO,
    counters: InteractionCounters,
    snapshot_state: ActorSnapshotState,
    base_rng_state: Mapping[str, Any],
    joint_rng_state: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite bundle {path}")
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{path.name}.", dir=str(path.parent))
    )
    try:
        base_model.save(temporary / "base_model")
        base_model.save_replay_buffer(temporary / "base_replay.pkl")
        residual_model.save(temporary / "residual_ppo")
        torch.save(
            {
                "base_rng_state": base_rng_state,
                "joint_rng_state": joint_rng_state,
            },
            temporary / "rng.pt",
        )
        atomic_write_json(
            temporary / "bundle.json",
            {
                "algorithm": ALGORITHM,
                "counters": counters.to_dict(),
                "snapshot_version": int(snapshot_state.version),
                "snapshot_actor_hash": snapshot_state.actor_hash,
                "base_model_num_timesteps": int(base_model.num_timesteps),
                "base_replay_position": int(base_model.replay_buffer.pos),
                "base_replay_offline_steps": int(
                    base_model.replay_buffer.offline_steps
                ),
                "residual_model_num_timesteps": int(
                    residual_model.num_timesteps
                ),
                "ppo_optimizer_steps": int(
                    residual_model.ppo_optimizer_steps
                ),
                "manifest_checkpoint_sha256": manifest[
                    "checkpoint_sha256"
                ],
            },
        )
        os.replace(temporary, path)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _restore_counters(payload: Mapping[str, Any]) -> InteractionCounters:
    values = payload["counters"]
    counters = InteractionCounters(
        action_chunk=ACTION_CHUNK,
        base_chunk_transitions=int(values["base_chunk_transitions"]),
        base_actual_primitive_steps=int(
            values["base_actual_primitive_steps"]
        ),
        joint_primitive_transitions=int(
            values["joint_primitive_transitions"]
        ),
        completed_cycles=int(values["completed_cycles"]),
    )
    if counters.to_dict() != dict(values):
        raise RuntimeError("Restored interaction counters are inconsistent")
    return counters


def main() -> None:
    arguments = parse_args()
    _validate_arguments(arguments)
    cycles = build_three_seven_cycles(
        total_equivalent_chunks=arguments.equivalent_chunk_budget,
        cycle_equivalent_chunks=CYCLE_EQUIVALENT_CHUNKS,
        action_chunk=ACTION_CHUNK,
        base_fraction=BASE_FRACTION,
    )

    run_directory = arguments.run_dir.resolve()
    manifest_path = run_directory / "run_manifest.json"
    if arguments.resume_bundle is None:
        if run_directory.exists():
            raise FileExistsError(f"Refusing to overwrite {run_directory}")
        run_directory.mkdir(parents=True)
    elif not manifest_path.is_file():
        raise FileNotFoundError("Resume requires the original run manifest")
    for directory in ("evaluations", "checkpoints", "resume"):
        (run_directory / directory).mkdir(exist_ok=True)

    config_arguments = argparse.Namespace(
        checkpoint=arguments.checkpoint,
        equivalent_chunk_budget=arguments.equivalent_chunk_budget,
        seed=arguments.seed,
        n_envs=arguments.n_envs,
        device=arguments.device,
    )
    cfg = compose_config(config_arguments)
    cfg.logdir = str(run_directory)
    checkpoint_path = arguments.checkpoint.resolve()
    checkpoint_hash = sha256_file(checkpoint_path)
    ddim_path = Path(cfg.base_policy_path).resolve()
    normalization_path = Path(cfg.normalization_path).resolve()
    ddim_hash = sha256_file(ddim_path)
    normalization_hash = sha256_file(normalization_path)
    if checkpoint_hash != str(cfg.p6.init_checkpoint_sha256):
        raise ValueError("Explicit 5M checkpoint SHA-256 mismatch")
    if ddim_hash != str(cfg.p6.frozen_ddim_sha256):
        raise ValueError("Frozen DDIM SHA-256 mismatch")
    if normalization_hash != str(cfg.p6.normalization_sha256):
        raise ValueError("Normalization SHA-256 mismatch")

    base_environment = _make_base_environment(
        cfg,
        seed=int(cfg.p6.train_env_seed),
    )
    snapshot_contract = make_vec_env(
        lambda: _make_hopper_environment(cfg, normalization_path),
        n_envs=1,
        seed=int(cfg.p6.train_env_seed) + 500_000,
        vec_env_cls=DummyVecEnv,
    )
    diffusion_policy = load_base_policy(cfg)
    prefill_chunks = int(cfg.train.init_rollout_steps) * int(cfg.env.n_envs)
    base_online_chunks = sum(cycle.base_chunk_transitions for cycle in cycles)
    replay_capacity = prefill_chunks + base_online_chunks + int(cfg.env.n_envs)

    live_base: P6ControlDSRL | None = None
    snapshot_shell = None
    joint_environment = None
    residual_model: CountingPPO | None = None
    try:
        snapshot_shell = _load_legacy_network(
            cfg=cfg,
            environment=snapshot_contract,
            diffusion_policy=diffusion_policy,
            buffer_size=1,
        )
        planner = FrozenDSRLChunkPlanner(
            snapshot_shell,
            action_chunk=ACTION_CHUNK,
            action_dimension=ACTION_DIMENSION,
        )

        if arguments.resume_bundle is None:
            loaded_live = _load_legacy_network(
                cfg=cfg,
                environment=base_environment,
                diffusion_policy=diffusion_policy,
                buffer_size=replay_capacity,
            )
            warmstart_snapshot = _capture_legacy_warmstart(loaded_live)
            live_base = _promote_legacy_control(loaded_live)
            _reset_control_optimizers(live_base)
            warmstart_parity = _verify_fresh_control_warmstart(
                live_base,
                warmstart_snapshot,
            )

            artifact_path = _prefill_artifact_path(cfg)
            metadata_path = artifact_path.with_suffix(
                artifact_path.suffix + ".json"
            )
            if not artifact_path.is_file() or not metadata_path.is_file():
                raise FileNotFoundError(
                    "The authenticated shared 5M prefill must already exist; "
                    "the 10k canary may not generate unbudgeted interactions"
                )
            provenance = {
                "init_checkpoint_sha256": checkpoint_hash,
                "frozen_ddim_sha256": ddim_hash,
                "normalization_sha256": normalization_hash,
            }
            prefill_arrays, prefill_metadata, generated = (
                collect_or_load_matched_prefill(
                    artifact_path=artifact_path,
                    warmstart_model=live_base,
                    env=base_environment,
                    vector_steps=int(cfg.train.init_rollout_steps),
                    environment_seed=int(cfg.p6.prefill_env_seed),
                    policy_seed=int(cfg.p6.prefill_policy_seed),
                    action_chunk=ACTION_CHUNK,
                    termination_semantics=ACTION_CHUNK_EARLY_BREAK,
                    provenance=provenance,
                )
            )
            if generated:
                raise RuntimeError("10k canary unexpectedly generated prefill")
            replay_hash = populate_replay_buffer(
                live_base.replay_buffer,
                prefill_arrays,
            )
            snapshot_state = ActorSnapshotState()
            snapshot_state.commit(
                live_actor=live_base.actor,
                snapshot_actor=snapshot_shell.actor,
            )
            planner.base_version = snapshot_state.version
            assert_parameter_storage_disjoint(
                live_base.actor,
                snapshot_shell.actor,
            )

            ppo_arguments = argparse.Namespace(
                learning_rate=arguments.learning_rate,
                rollout_steps=PPO_ROLLOUT_STEPS,
                batch_size=PPO_BATCH_SIZE,
                n_epochs=PPO_N_EPOCHS,
                clip_range=arguments.clip_range,
                target_kl=arguments.target_kl,
                max_grad_norm=arguments.max_grad_norm,
                residual_std=arguments.residual_std,
                seed=arguments.seed,
                device=arguments.device,
            )
            joint_environment = make_ppo_training_environment(
                cfg,
                planner,
                n_envs=arguments.n_envs,
                seed=int(cfg.p6.train_env_seed) + 100_000,
                residual_scale=arguments.residual_scale,
                deterministic_base=arguments.deterministic_base,
            )
            residual_model = build_ppo_model(
                cfg,
                joint_environment,
                arguments=ppo_arguments,
            )
            counters = InteractionCounters(action_chunk=ACTION_CHUNK)
            base_rng_state = _create_stream_rng(
                int(arguments.seed) + 100_000
            )
            joint_rng_state = _create_stream_rng(
                int(arguments.seed) + 200_000
            )
            manifest: dict[str, Any] = {
                "algorithm": ALGORITHM,
                "status": "ready",
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_sha256": checkpoint_hash,
                "ddim_path": str(ddim_path),
                "ddim_sha256": ddim_hash,
                "normalization_path": str(normalization_path),
                "normalization_sha256": normalization_hash,
                "outer_repository": git_state(ROOT),
                "sb3_repository": git_state(ROOT / "stable-baselines3"),
                "dppo_repository": git_state(ROOT / "dppo"),
                "seed": int(arguments.seed),
                "n_envs": int(arguments.n_envs),
                "deterministic_base": bool(arguments.deterministic_base),
                "base_train_env_seed": int(cfg.p6.train_env_seed),
                "joint_train_env_seed": int(cfg.p6.train_env_seed) + 100_000,
                "eval_seed_set": list(range(10_000, 10_100)),
                "action_chunk": ACTION_CHUNK,
                "action_dimension": ACTION_DIMENSION,
                "online_equivalent_chunk_budget": CANARY_EQUIVALENT_CHUNKS,
                "base_equivalent_chunk_budget": base_online_chunks,
                "joint_equivalent_chunk_budget": sum(
                    cycle.joint_equivalent_chunks for cycle in cycles
                ),
                "base_fraction": BASE_FRACTION,
                "cycle_equivalent_chunks": CYCLE_EQUIVALENT_CHUNKS,
                "cycle_count": len(cycles),
                "base_chunks_per_cycle": cycles[0].base_chunk_transitions,
                "joint_primitives_per_cycle": (
                    cycles[0].joint_primitive_transitions
                ),
                "ppo_rollout_steps_per_env": PPO_ROLLOUT_STEPS,
                "ppo_batch_size": PPO_BATCH_SIZE,
                "ppo_n_epochs": PPO_N_EPOCHS,
                "residual_scale": float(arguments.residual_scale),
                "initial_residual_std": float(arguments.residual_std),
                "prefill_source": "authenticated_shared_warmstart_dsrl",
                "prefill_counted_in_online_budget": False,
                "prefill_semantic_hash": prefill_metadata["semantic_hash"],
                "prefill_archive_sha256": prefill_metadata["archive_sha256"],
                "prefill_chunk_transitions": prefill_metadata[
                    "chunk_transitions"
                ],
                "initial_replay_hash": replay_hash,
                "warmstart_parity": warmstart_parity,
                "snapshot_version": snapshot_state.version,
                "snapshot_actor_hash": snapshot_state.actor_hash,
                "action_chunk_termination_semantics": (
                    ACTION_CHUNK_EARLY_BREAK
                ),
                "stream_protocol": {
                    "base": "original DSRL-NA; base-only chunk replay",
                    "joint": "per-step on-policy residual PPO",
                    "coupling": "one-way non-aliasing actor snapshot at boundary",
                },
                "history": [],
                "resume_environment_discontinuities": 0,
            }
            initial = _write_boundary_evaluations(
                run_directory=run_directory,
                prefix="initial",
                model=residual_model,
                cfg=cfg,
                planner=planner,
                seeds=list(
                    range(10_000, 10_000 + arguments.online_eval_episodes)
                ),
                equivalent_chunks=0,
                deterministic=True,
            )
            parity_error = abs(
                initial["base_only"]["raw_return_mean"]
                - initial["joint"]["raw_return_mean"]
            )
            if parity_error != 0.0:
                raise RuntimeError(
                    "Zero-residual initial evaluation is not exactly base-only: "
                    f"{parity_error}"
                )
            manifest["initial_evaluation"] = initial
            manifest["zero_residual_return_mean_abs_error"] = parity_error
            atomic_write_json(manifest_path, manifest)
        else:
            manifest = json.loads(manifest_path.read_text())
            if manifest.get("algorithm") != ALGORITHM:
                raise ValueError("Resume manifest algorithm mismatch")
            if manifest.get("checkpoint_sha256") != checkpoint_hash:
                raise ValueError("Resume checkpoint provenance mismatch")
            bundle = arguments.resume_bundle.resolve()
            bundle_data = json.loads((bundle / "bundle.json").read_text())
            if bundle_data.get("algorithm") != ALGORITHM:
                raise ValueError("Resume bundle algorithm mismatch")
            live_base = P6ControlDSRL.load(
                bundle / "base_model.zip",
                env=base_environment,
                device=arguments.device,
                custom_objects={"diffusion_policy": diffusion_policy},
            )
            live_base.load_replay_buffer(bundle / "base_replay.pkl")
            _set_model_inference_mode(live_base)
            snapshot_state = ActorSnapshotState(
                version=int(bundle_data["snapshot_version"]),
            )
            snapshot_state.actor_hash = module_state_hash(
                {"actor": live_base.actor}
            )
            if (
                snapshot_state.actor_hash
                != bundle_data["snapshot_actor_hash"]
            ):
                raise ValueError("Resume snapshot/live actor hash mismatch")
            copied = snapshot_state.actor_hash
            snapshot_shell.actor.load_state_dict(
                live_base.actor.state_dict(),
                strict=True,
            )
            snapshot_shell.actor.eval()
            snapshot_shell.actor.requires_grad_(False)
            if module_state_hash({"actor": snapshot_shell.actor}) != copied:
                raise RuntimeError("Resume actor snapshot copy failed")
            planner.base_version = snapshot_state.version

            joint_environment = make_ppo_training_environment(
                cfg,
                planner,
                n_envs=arguments.n_envs,
                seed=int(cfg.p6.train_env_seed) + 100_000,
                residual_scale=arguments.residual_scale,
                deterministic_base=arguments.deterministic_base,
            )
            residual_model = CountingPPO.load(
                bundle / "residual_ppo.zip",
                env=joint_environment,
                device=arguments.device,
            )
            residual_model.ppo_optimizer_steps = int(
                bundle_data["ppo_optimizer_steps"]
            )
            counters = _restore_counters(bundle_data)
            rng_payload = torch.load(bundle / "rng.pt")
            base_rng_state = rng_payload["base_rng_state"]
            joint_rng_state = rng_payload["joint_rng_state"]
            if int(live_base.num_timesteps) != counters.base_chunk_transitions:
                raise ValueError("Resume base timestep counter mismatch")
            if (
                int(residual_model.num_timesteps)
                != counters.joint_primitive_transitions
            ):
                raise ValueError("Resume PPO timestep counter mismatch")
            manifest["status"] = "resumed"
            manifest["resume_bundle"] = str(bundle)
            manifest["resume_environment_discontinuities"] = int(
                manifest.get("resume_environment_discontinuities", 0)
            ) + 1
            atomic_write_json(manifest_path, manifest)

        assert live_base is not None
        assert residual_model is not None
        assert joint_environment is not None
        guard = StreamIsolationGuard(
            live_modules=planner_modules(live_base),
            snapshot_modules=planner_modules(snapshot_shell),
            residual_policy=residual_model.policy,
        )

        stop_cycle = (
            len(cycles)
            if arguments.stop_after_cycles is None
            else int(arguments.stop_after_cycles)
        )
        if stop_cycle <= counters.completed_cycles:
            raise ValueError("Stop cycle must exceed restored progress")

        for cycle in cycles[counters.completed_cycles : stop_cycle]:
            history_entry: dict[str, Any] = {
                "cycle": cycle.index,
                "base_snapshot_version_before": snapshot_state.version,
            }

            restore_rng_state(base_rng_state)
            base_guard = guard.before_base()
            base_callback = BasePrimitiveCounterCallback()
            base_actor_before = module_state_hash({"actor": live_base.actor})
            live_base.learn(
                total_timesteps=cycle.base_chunk_transitions,
                callback=base_callback,
                reset_num_timesteps=False,
                tb_log_name="asymmetric_base_dsrl",
            )
            base_rng_state = capture_rng_state()
            guard.after_base(base_guard)
            base_actor_after = module_state_hash({"actor": live_base.actor})
            if base_actor_after == base_actor_before:
                raise RuntimeError("Base DSRL block did not update the noise actor")

            snapshot_state.commit(
                live_actor=live_base.actor,
                snapshot_actor=snapshot_shell.actor,
            )
            planner.base_version = snapshot_state.version

            restore_rng_state(joint_rng_state)
            _refresh_joint_observations(
                model=residual_model,
                environment=joint_environment,
            )
            joint_guard = guard.before_joint()
            residual_callback = SnapshotResidualMetricsCallback()
            residual_steps_before = int(residual_model.num_timesteps)
            residual_model.learn(
                total_timesteps=cycle.joint_primitive_transitions,
                callback=residual_callback,
                reset_num_timesteps=False,
                tb_log_name="asymmetric_residual_ppo",
            )
            joint_rng_state = capture_rng_state()
            guard.after_joint(joint_guard)
            residual_steps = int(residual_model.num_timesteps) - (
                residual_steps_before
            )
            if residual_steps != cycle.joint_primitive_transitions:
                raise RuntimeError(
                    f"PPO rollout consumed {residual_steps} primitives, expected "
                    f"{cycle.joint_primitive_transitions}"
                )
            if residual_callback.snapshot_versions != {
                snapshot_state.version
            }:
                raise RuntimeError(
                    "A PPO rollout mixed actor snapshot versions: "
                    f"{residual_callback.snapshot_versions}"
                )

            counters.record_cycle(
                cycle=cycle,
                base_actual_primitive_steps=(
                    base_callback.actual_primitive_steps
                ),
            )
            history_entry.update(
                {
                    "base_snapshot_version_after": snapshot_state.version,
                    "base_actor_hash_before": base_actor_before,
                    "base_actor_hash_after": base_actor_after,
                    "snapshot_actor_hash": snapshot_state.actor_hash,
                    "rollout_snapshot_versions": sorted(
                        residual_callback.snapshot_versions
                    ),
                    "base_actual_primitive_steps": (
                        base_callback.actual_primitive_steps
                    ),
                    "base_early_termination_chunks": (
                        base_callback.early_termination_chunks
                    ),
                    "residual_metrics": residual_callback.means(),
                    "action_critic_optimizer_steps": int(
                        live_base.action_critic_optimizer_steps
                    ),
                    "noise_critic_optimizer_steps": int(
                        live_base.modulation_critic_optimizer_steps
                    ),
                    "noise_actor_optimizer_steps": int(
                        live_base.noise_actor_optimizer_steps
                    ),
                    "ppo_optimizer_steps": int(
                        residual_model.ppo_optimizer_steps
                    ),
                    "counters": counters.to_dict(),
                }
            )
            manifest["history"].append(history_entry)
            manifest.update(
                {
                    "status": "training",
                    "snapshot_version": snapshot_state.version,
                    "snapshot_actor_hash": snapshot_state.actor_hash,
                    "counters": counters.to_dict(),
                }
            )

            if counters.completed_cycles % CHECKPOINT_INTERVAL_CYCLES == 0:
                online = _write_boundary_evaluations(
                    run_directory=run_directory,
                    prefix=f"online_{counters.total_equivalent_chunks:012d}",
                    model=residual_model,
                    cfg=cfg,
                    planner=planner,
                    seeds=list(
                        range(
                            10_000,
                            10_000 + arguments.online_eval_episodes,
                        )
                    ),
                    equivalent_chunks=counters.total_equivalent_chunks,
                    deterministic=True,
                )
                history_entry["online_evaluation"] = online
                bundle_path = (
                    run_directory
                    / "checkpoints"
                    / f"boundary_{counters.total_equivalent_chunks:012d}"
                )
                _save_bundle(
                    bundle_path,
                    base_model=live_base,
                    residual_model=residual_model,
                    counters=counters,
                    snapshot_state=snapshot_state,
                    base_rng_state=base_rng_state,
                    joint_rng_state=joint_rng_state,
                    manifest=manifest,
                )
                manifest["latest_boundary_bundle"] = str(bundle_path)
            atomic_write_json(manifest_path, manifest)

        resume_path = (
            run_directory
            / "resume"
            / f"boundary_{counters.total_equivalent_chunks:012d}"
        )
        _save_bundle(
            resume_path,
            base_model=live_base,
            residual_model=residual_model,
            counters=counters,
            snapshot_state=snapshot_state,
            base_rng_state=base_rng_state,
            joint_rng_state=joint_rng_state,
            manifest=manifest,
        )
        manifest["latest_resume_bundle"] = str(resume_path)

        if counters.completed_cycles < len(cycles):
            manifest["status"] = "interrupted"
            manifest["counters"] = counters.to_dict()
            atomic_write_json(manifest_path, manifest)
            (run_directory / "INTERRUPTED").touch()
            return

        final_seeds = list(
            range(10_000, 10_000 + arguments.final_eval_episodes)
        )
        final_deterministic = _write_boundary_evaluations(
            run_directory=run_directory,
            prefix="final",
            model=residual_model,
            cfg=cfg,
            planner=planner,
            seeds=final_seeds,
            equivalent_chunks=counters.total_equivalent_chunks,
            deterministic=True,
        )
        final_stochastic = _write_boundary_evaluations(
            run_directory=run_directory,
            prefix="final",
            model=residual_model,
            cfg=cfg,
            planner=planner,
            seeds=final_seeds,
            equivalent_chunks=counters.total_equivalent_chunks,
            deterministic=False,
        )
        expected_counts = {
            "action_critic_optimizer_steps": 3_000,
            "noise_critic_optimizer_steps": 1_500,
            "noise_actor_optimizer_steps": 3_000,
        }
        actual_counts = {
            "action_critic_optimizer_steps": int(
                live_base.action_critic_optimizer_steps
            ),
            "noise_critic_optimizer_steps": int(
                live_base.modulation_critic_optimizer_steps
            ),
            "noise_actor_optimizer_steps": int(
                live_base.noise_actor_optimizer_steps
            ),
        }
        if actual_counts != expected_counts:
            raise RuntimeError(
                f"Base optimizer counts {actual_counts} != {expected_counts}"
            )
        if counters.total_equivalent_chunks != CANARY_EQUIVALENT_CHUNKS:
            raise RuntimeError("10k combined budget was not consumed exactly")
        manifest.update(
            {
                "status": "complete",
                "counters": counters.to_dict(),
                "final_deterministic": final_deterministic,
                "final_stochastic": final_stochastic,
                "base_optimizer_counts": actual_counts,
                "ppo_optimizer_steps": int(
                    residual_model.ppo_optimizer_steps
                ),
                "snapshot_version": snapshot_state.version,
                "snapshot_actor_hash": snapshot_state.actor_hash,
            }
        )
        atomic_write_json(manifest_path, manifest)
        interrupted_marker = run_directory / "INTERRUPTED"
        if interrupted_marker.exists():
            interrupted_marker.unlink()
        (run_directory / "COMPLETE").touch()
    finally:
        if joint_environment is not None:
            joint_environment.close()
        base_environment.close()
        snapshot_contract.close()


if __name__ == "__main__":
    main()
