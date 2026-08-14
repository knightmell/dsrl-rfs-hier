"""V1.3 gate diagnostic: does headwise QA_joint delta-Q predict real returns?

The plan (Codex 2026-08-05) forbids using the headwise delta-Q
    deltaQ_i = QA_joint,i(s, a_exec) - QA_joint,i(s, a_base)
as a residual gate until it positively correlates with real paired counterfactual
returns.  This diagnostic measures that correlation on a saved three-critic
checkpoint.

Method (deterministic re-roll protocol):
  1. Load a three-critic HierarchicalRFSDSRL checkpoint (model.zip) and the
     authenticated Frozen DDIM artifact (from the p6 config).
  2. Roll out the CURRENT BASE policy (residual disabled, deterministic) to
     collect chunk-boundary states at a configurable spacing.
  3. For every collected state, re-roll from the same (policy, env) seeds to the
     exact same chunk boundary (a parity check guards RNG drift), then branch:
       a_exec^0  = base action (residual off)                -> realized return R0
       a_exec^+  = compose(base, +eps*d, beta)               -> realized return R+
       a_exec^-  = compose(base, -eps*d, beta)               -> realized return R-
     where d is a per-state seeded unit direction in residual-logit space.  The
     branch first-chunk action is executed, then the base policy continues for
     horizon_chunks - 1 further chunks (or until the episode ends).
  4. Predict headwise conservative delta-Q for each candidate using QA_joint at
     the SAME base/noise as executed (probe composition is shared between the
     prediction and the executed action).
  5. Report Spearman / pairwise-preference / top-1 of the conservative delta-Q
     against realized paired return deltas, with a state-bootstrap 95% CI.

NOTE: this diagnostic never trains.  It requires MuJoCo/D4RL environments and
an actual checkpoint, so it is exercised on real checkpoints only; the
statistical helpers below are unit-tested independently.  Because Phase B runs
at beta=0, a Phase-B checkpoint must be probed with --beta (default 0.1, the
frozen target) or every branch collapses to the base action.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / "dppo"), str(ROOT / "stable-baselines3")]

from env_utils import ACTION_CHUNK_EARLY_BREAK, ActionChunkWrapper, ObservationWrapperGym  # noqa: E402
from p6_evaluation import isolated_model_evaluation  # noqa: E402
from p6_runtime import atomic_write_json, isolated_rng, seed_all  # noqa: E402
from stable_baselines3.dsrl.hierarchical_rfs_dsrl import (  # noqa: E402
    HierarchicalRFSDSRL,
    compose_action,
)
from stable_baselines3.dsrl.hierarchical_replay_buffer import BranchMode  # noqa: E402
from utils import load_base_policy  # noqa: E402


# --------------------------------------------------------------------------- #
# Statistical helpers (unit-testable, no env/model dependency)
# --------------------------------------------------------------------------- #


def spearman_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation between two equal-length vectors."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape or x.size < 3:
        return float("nan")
    if np.all(x == x[0]) or np.all(y == y[0]):
        return float("nan")
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    rx = rx.astype(np.float64)
    ry = ry.astype(np.float64)
    mu = (x.size - 1.0) / 2.0
    denom = np.sqrt(np.sum((rx - mu) ** 2) * np.sum((ry - mu) ** 2))
    if denom == 0.0:
        return float("nan")
    return float(np.sum((rx - mu) * (ry - mu)) / denom)


def pairwise_preference_accuracy(
    predicted: np.ndarray, realized: np.ndarray
) -> float:
    """Fraction of comparable pairs where sign(predicted delta) matches
    sign(realized delta).  Ties are excluded."""
    n = len(predicted)
    agree = 0
    total = 0
    for i in range(n):
        for j in range(i + 1, n):
            dp = predicted[i] - predicted[j]
            dr = realized[i] - realized[j]
            if dr == 0.0:
                continue
            total += 1
            agree += 1 if np.sign(dp) == np.sign(dr) else 0
    return float(agree / total) if total else float("nan")


def top1_agreement(predicted: np.ndarray, realized: np.ndarray) -> float:
    """1.0 if the argmax of predicted matches argmax of realized."""
    if len(predicted) < 2:
        return float("nan")
    return float(int(np.argmax(predicted) == np.argmax(realized)))


def compute_correlation_report(
    predicted_deltas: Sequence[float],
    realized_deltas: Sequence[float],
) -> dict[str, float]:
    p = np.asarray(predicted_deltas, dtype=np.float64)
    r = np.asarray(realized_deltas, dtype=np.float64)
    return {
        "count": int(len(p)),
        "spearman": spearman_correlation(p, r),
        "pairwise_preference_accuracy": pairwise_preference_accuracy(p, r),
        "top1_agreement": top1_agreement(p, r),
    }


def bootstrap_ci95(
    per_state: Sequence[dict[str, Any]],
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    """Percentile 95% CI of spearman / pairwise accuracy over state resamples.

    Every resample draws ``len(per_state)`` states with replacement and pools
    the +eps / -eps (predicted, realized) points of the drawn states before
    computing the correlation statistics.
    """
    if samples < 100:
        raise ValueError("At least 100 bootstrap samples are required")
    rng = np.random.default_rng(int(seed))
    spearmans: list[float] = []
    pairwise: list[float] = []
    count = len(per_state)
    if count < 2:
        return {"samples": 0, "spearman_ci95": [float("nan"), float("nan")], "pairwise_ci95": [float("nan"), float("nan")]}
    for _ in range(int(samples)):
        draw = [
            per_state[index]
            for index in rng.integers(0, count, size=count)
        ]
        predicted = [point for state in draw for point in state["predicted"]]
        realized = [point for state in draw for point in state["realized"]]
        spearmans.append(spearman_correlation(np.asarray(predicted), np.asarray(realized)))
        pairwise.append(pairwise_preference_accuracy(np.asarray(predicted), np.asarray(realized)))
    spearman_array = np.asarray([value for value in spearmans if not math.isnan(value)], dtype=np.float64)
    pairwise_array = np.asarray([value for value in pairwise if not math.isnan(value)], dtype=np.float64)
    return {
        "samples": int(samples),
        "spearman_ci95": _percentile_ci(spearman_array),
        "pairwise_ci95": _percentile_ci(pairwise_array),
    }


def _percentile_ci(values: np.ndarray) -> list[float]:
    if values.size == 0:
        return [float("nan"), float("nan")]
    low, high = np.percentile(values, [5.0, 95.0])
    return [float(low), float(high)]


# --------------------------------------------------------------------------- #
# Prediction side (uses the model's own forward path, no env)
# --------------------------------------------------------------------------- #


def probe_candidate(
    model: HierarchicalRFSDSRL,
    observation: np.ndarray,
    base_noise: np.ndarray,
    direction: np.ndarray,
    *,
    eps: float,
    beta: float,
) -> dict[str, Any]:
    """Headwise conservative QA_joint delta-Q and the executed action for a
    ``eps * direction`` residual-logit probe.

    The executed action is the bound-preserving composition of the Frozen-DDIM
    base action with ``eps * direction`` at the given beta, and is exactly what
    the diagnostic feeds to the environment, so prediction and execution share
    one composition.

    Returns dict with:
      delta_conservative: min over heads of Q(candidate) - Q(base)
      delta_head0 / delta_head1: per-head deltas
      action_exec: np.float32 (action_dim_flat,) composed chunk action
    """
    obs = torch.as_tensor(observation, device=model.device, dtype=torch.float32)[None]
    noise = torch.as_tensor(base_noise, device=model.device, dtype=torch.float32)[None]
    probe_logits = eps * torch.as_tensor(
        direction, device=model.device, dtype=torch.float32
    )[None]
    with torch.no_grad():
        generated = model._generate_hierarchical_action(
            obs,
            noise,
            beta=beta,
        )
        base = generated.action_base
        zero_logits = torch.zeros_like(base)
        candidate = compose_action(
            base,
            probe_logits,
            beta,
            model._exec_action_low_tensor,
            model._exec_action_high_tensor,
            numerical_tolerance=model.numerical_bound_tolerance,
        )
        baseline = compose_action(
            base,
            zero_logits,
            beta,
            model._exec_action_low_tensor,
            model._exec_action_high_tensor,
            numerical_tolerance=model.numerical_bound_tolerance,
        )
        candidate_heads = model.qa_joint(obs, candidate.action_exec)
        baseline_heads = model.qa_joint(obs, baseline.action_exec)
        delta_0 = candidate_heads[0] - baseline_heads[0]
        delta_1 = candidate_heads[1] - baseline_heads[1]
        conservative = float(torch.min(delta_0, delta_1).detach().cpu().item())
    return {
        "delta_conservative": conservative,
        "delta_head0": float(delta_0.detach().cpu().item()),
        "delta_head1": float(delta_1.detach().cpu().item()),
        "action_exec": candidate.action_exec[0].detach().cpu().numpy().astype(np.float32),
    }


def predict_delta_q(
    model: HierarchicalRFSDSRL,
    observation: np.ndarray,
    base_noise: np.ndarray,
    residual_direction: np.ndarray,
    *,
    beta: float,
    eps: float,
    device: str,
) -> tuple[float, float]:
    """Headwise conservative QA_joint delta-Q for a +eps residual perturbation.

    Returns (delta_q, delta_q_head1) where delta_q is the min-over-heads
    predicted gain of composing +eps*d over the base action.  Kept for API
    compatibility; ``probe_candidate`` is the full interface used by main().
    """
    del device
    result = probe_candidate(
        model,
        observation,
        base_noise,
        residual_direction,
        eps=eps,
        beta=beta,
    )
    return result["delta_conservative"], result["delta_head1"]


# --------------------------------------------------------------------------- #
# Environment + config
# --------------------------------------------------------------------------- #


def _resolve_path(value: str, repo_root: Path) -> str:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return str(path.resolve())


def compose_p6_config(
    config_path: Path,
    *,
    repo_root: Path,
    seed: int,
) -> Any:
    """Compose the p6 config yaml the same way p6_train does (resolvers
    registered, seed overridden).  The config must fully resolve on its own;
    the fresh 2.5M config does."""
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    OmegaConf.register_new_resolver("round_up", math.ceil, replace=True)
    OmegaConf.register_new_resolver("round_down", math.floor, replace=True)
    config_path = config_path.resolve()
    config_dir = config_path.parent
    config_name = config_path.stem
    with initialize_config_dir(
        config_dir=str(config_dir),
        job_name="diagnose_qa_joint_delta_correlation",
        version_base=None,
    ):
        cfg = compose(config_name=config_name, overrides=[f"seed={int(seed)}"])
    OmegaConf.resolve(cfg)
    return cfg


def make_environment_factory(
    cfg: Any,
    *,
    repo_root: Path,
) -> Callable[[], Any]:
    normalization_path = _resolve_path(cfg.normalization_path, repo_root)

    def _factory() -> Any:
        import d4rl  # noqa: F401
        import d4rl.gym_mujoco  # noqa: F401
        import gym  # noqa: F401

        raw_environment = gym.make(cfg.env_name)
        normalized_environment = ObservationWrapperGym(
            raw_environment,
            normalization_path,
        )
        return ActionChunkWrapper(
            normalized_environment,
            cfg,
            max_episode_steps=int(cfg.env.max_episode_steps),
            action_chunk_termination_semantics=ACTION_CHUNK_EARLY_BREAK,
        )

    return _factory


def load_checkpoint(
    cfg: Any,
    model_path: Path,
    *,
    device: str,
) -> HierarchicalRFSDSRL:
    diffusion_policy = load_base_policy(cfg)
    model = HierarchicalRFSDSRL.load(
        str(model_path.resolve()),
        device=device,
        diffusion_policy=diffusion_policy,
    )
    return model


# --------------------------------------------------------------------------- #
# Rollout + counterfactual protocol
# --------------------------------------------------------------------------- #


def _rollout_to_branch(
    *,
    make_environment: Callable[[], Any],
    model: HierarchicalRFSDSRL,
    environment_seed: int,
    policy_seed: int,
    chunk_index: int,
    branch_action: np.ndarray,
    horizon_chunks: int,
) -> tuple[float, np.ndarray]:
    """Re-roll the base policy to ``chunk_index``, inject ``branch_action`` as
    the next chunk, then continue the base policy for horizon-1 more chunks.

    Returns (realized_chunk_return, branch_state_observation).  The branch
    observation parity check is the caller's responsibility.
    """
    environment = make_environment()
    try:
        seed_all(int(policy_seed))
        observation = environment.reset(seed=int(environment_seed))[0]
        terminated = truncated = False
        index = 0
        while index < int(chunk_index) and not (terminated or truncated):
            components, _ = model.predict_with_components(
                np.asarray(observation),
                deterministic=True,
                mode="current_base_only",
            )
            (
                observation,
                _,
                terminated,
                truncated,
                _,
            ) = environment.step(components["action_exec"])
            index += 1
        if terminated or truncated:
            raise RuntimeError(
                "Re-rolled episode terminated before the branch chunk; "
                "collected state is not reachable"
            )
        branch_observation = np.asarray(observation, dtype=np.float32)
        (
            observation,
            reward,
            terminated,
            truncated,
            _,
        ) = environment.step(branch_action)
        total = float(reward)
        remaining = int(horizon_chunks) - 1
        for _ in range(remaining):
            if terminated or truncated:
                break
            components, _ = model.predict_with_components(
                np.asarray(observation),
                deterministic=True,
                mode="current_base_only",
            )
            (
                observation,
                reward,
                terminated,
                truncated,
                _,
            ) = environment.step(components["action_exec"])
            total += float(reward)
        return total, branch_observation
    finally:
        environment.close()


def collect_states(
    *,
    model: HierarchicalRFSDSRL,
    make_environment: Callable[[], Any],
    environment_seeds: Sequence[int],
    policy_seed_start: int,
    states_per_episode: int,
    max_chunk_steps: int,
) -> list[dict[str, Any]]:
    """Deterministic base-policy rollouts collecting chunk-boundary states."""
    if int(states_per_episode) < 1:
        raise ValueError("states_per_episode must be positive")
    sample_step = max(1, int(max_chunk_steps) // (int(states_per_episode) + 1))
    states: list[dict[str, Any]] = []
    with isolated_rng(), isolated_model_evaluation(model):
        for episode_index, environment_seed in enumerate(environment_seeds):
            policy_seed = int(policy_seed_start) + episode_index
            environment = make_environment()
            try:
                seed_all(policy_seed)
                observation = environment.reset(seed=int(environment_seed))[0]
                chunk_index = 0
                terminated = truncated = False
                while not (terminated or truncated):
                    if chunk_index > 0 and chunk_index % sample_step == 0:
                        components, _ = model.predict_with_components(
                            np.asarray(observation),
                            deterministic=True,
                            mode="current_base_only",
                        )
                        states.append(
                            {
                                "episode_index": int(episode_index),
                                "environment_seed": int(environment_seed),
                                "policy_seed": policy_seed,
                                "chunk_index": int(chunk_index),
                                "observation": np.asarray(
                                    observation, dtype=np.float32
                                ),
                                "noise_scaled": np.asarray(
                                    components["noise_scaled"], dtype=np.float32
                                ),
                                "base_action": np.asarray(
                                    components["action_base"], dtype=np.float32
                                ),
                            }
                        )
                        if len(states) >= (
                            int(states_per_episode) * (episode_index + 1)
                        ):
                            break
                    components, _ = model.predict_with_components(
                        np.asarray(observation),
                        deterministic=True,
                        mode="current_base_only",
                    )
                    (
                        observation,
                        _,
                        terminated,
                        truncated,
                        _,
                    ) = environment.step(components["action_exec"])
                    chunk_index += 1
                    if chunk_index >= int(max_chunk_steps):
                        break
            finally:
                environment.close()
    return states


def run_diagnostic(
    *,
    model: HierarchicalRFSDSRL,
    make_environment: Callable[[], Any],
    environment_seeds: Sequence[int],
    policy_seed_start: int,
    states_per_episode: int,
    horizon_chunks: int,
    eps: float,
    beta: float,
    residual_seed_start: int,
    bootstrap_samples: int,
    action_dim_flat: int,
    max_chunk_steps: int,
) -> dict[str, Any]:
    residual_rng = np.random.default_rng(int(residual_seed_start))
    states = collect_states(
        model=model,
        make_environment=make_environment,
        environment_seeds=environment_seeds,
        policy_seed_start=policy_seed_start,
        states_per_episode=states_per_episode,
        max_chunk_steps=max_chunk_steps,
    )
    if len(states) < 2:
        raise RuntimeError(
            "Collected fewer than 2 counterfactual states; the base policy "
            "falls too early to sample paired returns.  The gate needs a "
            "trained base (see the runbook)."
        )
    rows: list[dict[str, Any]] = []
    per_state: list[dict[str, Any]] = []
    for state_index, state in enumerate(states):
        direction = residual_rng.standard_normal(size=(int(action_dim_flat),))
        direction = direction / (float(np.linalg.norm(direction)) + 1e-12)
        candidates = [
            ("zero", np.zeros_like(direction)),
            ("plus", direction),
            ("minus", -direction),
        ]
        candidate_rows: list[dict[str, Any]] = []
        for name, unit_direction in candidates:
            probe = probe_candidate(
                model,
                state["observation"],
                state["noise_scaled"],
                unit_direction,
                eps=eps,
                beta=beta,
            )
            realized, branch_observation = _rollout_to_branch(
                make_environment=make_environment,
                model=model,
                environment_seed=state["environment_seed"],
                policy_seed=state["policy_seed"],
                chunk_index=state["chunk_index"],
                branch_action=probe["action_exec"],
                horizon_chunks=horizon_chunks,
            )
            if not np.allclose(
                branch_observation,
                state["observation"],
                rtol=1e-4,
                atol=1e-4,
            ):
                raise RuntimeError(
                    "Re-rolled branch state diverged from the collected state; "
                    "the rollout is not deterministic for "
                    f"state {state_index}, candidate {name}"
                )
            candidate_rows.append(
                {
                    "candidate": name,
                    "predicted_delta": float(probe["delta_conservative"]),
                    "predicted_delta_head0": float(probe["delta_head0"]),
                    "predicted_delta_head1": float(probe["delta_head1"]),
                    "realized_return": float(realized),
                }
            )
            rows.append(
                {
                    "state_index": int(state_index),
                    "episode_index": int(state["episode_index"]),
                    "environment_seed": int(state["environment_seed"]),
                    "policy_seed": int(state["policy_seed"]),
                    "chunk_index": int(state["chunk_index"]),
                    "candidate": name,
                    "predicted_delta": float(probe["delta_conservative"]),
                    "realized_return": float(realized),
                }
            )
        zero_row = next(row for row in candidate_rows if row["candidate"] == "zero")
        plus_row = next(row for row in candidate_rows if row["candidate"] == "plus")
        minus_row = next(row for row in candidate_rows if row["candidate"] == "minus")
        per_state.append(
            {
                "state_index": int(state_index),
                "realized": [
                    float(plus_row["realized_return"] - zero_row["realized_return"]),
                    float(minus_row["realized_return"] - zero_row["realized_return"]),
                ],
                "predicted": [
                    float(plus_row["predicted_delta"]),
                    float(minus_row["predicted_delta"]),
                ],
            }
        )
    predicted = [point for state in per_state for point in state["predicted"]]
    realized = [point for state in per_state for point in state["realized"]]
    report = compute_correlation_report(predicted, realized)
    confidence = bootstrap_ci95(
        per_state,
        samples=int(bootstrap_samples),
        seed=int(residual_seed_start) + 1,
    )
    spearman_low = confidence["spearman_ci95"][0]
    pairwise_low = confidence["pairwise_ci95"][0]
    return {
        "protocol": (
            "deterministic seeded re-roll to the branch chunk; zero/+eps/-eps "
            "paired counterfactual first chunks composed at the probe beta; "
            "base zero-residual continuation; realized return over "
            "horizon_chunks chunks or episode end; conservative QA_joint "
            "delta-Q predicted at the same base/noise as executed"
        ),
        "state_count": int(len(states)),
        "points": int(len(predicted)),
        "horizon_chunks": int(horizon_chunks),
        "horizon_primitive_steps": int(horizon_chunks) * int(
            model.diffusion_act_chunk
        ),
        "eps": float(eps),
        "beta_probe": float(beta),
        "aggregate": report,
        "confidence_interval_95": confidence,
        "screening_gate": {
            "spearman_ci95_lower_gt_0": bool(spearman_low > 0.0),
            "pairwise_ci95_lower_gt_0p55": bool(pairwise_low > 0.55),
        },
        "states": per_state,
        "episodes": rows,
    }


# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="three-critic Core V1 model.zip checkpoint (Phase-B for the gate)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="p6 config yaml that fully resolves (e.g. cfg/gym/p6_hopper_fresh_2p5m.yaml)",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--episodes", type=int, default=25)
    parser.add_argument("--states-per-episode", type=int, default=4)
    parser.add_argument("--horizon-chunks", type=int, default=4)
    parser.add_argument("--eps", type=float, default=0.05)
    parser.add_argument(
        "--beta",
        type=float,
        default=0.1,
        help="probe composition beta; Phase-B checkpoints must probe at 0.1",
    )
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument("--residual-seed-start", type=int, default=32_000)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("logs/qa_joint_delta_correlation.json"),
        help=(
            "Output JSON path.  Keep it under the gitignored logs/ directory: "
            "the certified-resume source fingerprint hashes every untracked "
            "non-ignored file, so writing to the repo root between launch and "
            "resume would block the resume with a manifest mismatch."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.episodes <= 0 or args.states_per_episode <= 0:
        raise ValueError("episodes and states_per_episode must be positive")
    if not 0 < args.horizon_chunks:
        raise ValueError("horizon_chunks must be positive")
    if not (0 < args.eps < 1.0):
        raise ValueError("eps must lie in (0, 1)")
    if not (0.0 <= args.beta <= 1.0):
        raise ValueError("beta must lie in [0, 1]")
    if args.bootstrap_samples < 100:
        raise ValueError("At least 100 bootstrap samples are required")

    cfg = compose_p6_config(
        args.config.resolve(),
        repo_root=ROOT,
        seed=int(args.seed),
    )
    cfg.device = args.device
    model = load_checkpoint(cfg, args.model, device=args.device)
    action_dim_flat = int(model.diffusion_act_chunk) * int(model.diffusion_act_dim)
    make_environment = make_environment_factory(cfg, repo_root=ROOT)

    environment_seeds = [
        int(args.seed) + 10_000 + index for index in range(int(args.episodes))
    ]
    result = run_diagnostic(
        model=model,
        make_environment=make_environment,
        environment_seeds=environment_seeds,
        policy_seed_start=int(args.seed) + 20_000,
        states_per_episode=int(args.states_per_episode),
        horizon_chunks=int(args.horizon_chunks),
        eps=float(args.eps),
        beta=float(args.beta),
        residual_seed_start=int(args.residual_seed_start),
        bootstrap_samples=int(args.bootstrap_samples),
        action_dim_flat=action_dim_flat,
        max_chunk_steps=math.ceil(
            int(cfg.env.max_episode_steps) / int(cfg.act_steps)
        ),
    )
    result.update(
        {
            "model_checkpoint": str(args.model.resolve()),
            "config": str(args.config.resolve()),
            "device": args.device,
            "seed": int(args.seed),
            "residual_seed_start": int(args.residual_seed_start),
        }
    )
    output_path = Path(args.output).resolve()
    atomic_write_json(output_path, result)
    summary = {
        "horizon_chunks": result["horizon_chunks"],
        "horizon_primitive_steps": result["horizon_primitive_steps"],
        "state_count": result["state_count"],
        "points": result["points"],
        "aggregate": result["aggregate"],
        "confidence_interval_95": result["confidence_interval_95"],
        "screening_gate": result["screening_gate"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"\nWrote {output_path}")


if __name__ == "__main__":
    main()
