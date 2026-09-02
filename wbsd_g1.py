"""WBSD G1: read-only Gaussian support and QW-ranking checkpoint sweep.

The probe consumes the frozen G0 state bank and CRN pools.  It never creates an
environment, calls backward, or steps an optimizer.  Four workers are split by
proposal RNG seed; aggregation treats those seeds as repeated measurements of
the same 512 states rather than as independent samples.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from wbsd_probe import (
    ROOT,
    _git_metadata,
    _load_hierarchy_checkpoint,
    _model_modules,
    _write_csv,
    _write_json,
    capture_global_rng_state,
    counted_candidate_calls,
    global_rng_states_equal,
    isolated_global_rng,
    model_state_hashes,
    sha256_array,
    sha256_file,
    utc_now,
)


PREFIXES = (1, 2, 4, 8, 16, 32, 64)
PRIMARY_VIEW = "augmentation"
NATIVE_VIEW = "native_source"
METHOD_CURRENT = "g_current"
METHOD_PRIOR = "g_prior_exact"
METHOD_MIX = "g_mix"
METHODS = (METHOD_CURRENT, METHOD_PRIOR, METHOD_MIX)


def build_reachable_mix(
    current: np.ndarray, prior_exact: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Build a nested 50/50 deployable pool with one shared current anchor.

    Slot 0 is ``current[:, 0]``.  Odd subsequent slots use ``tanh``-restricted
    standard-Gaussian prior draws and even slots use current-actor draws.  The
    returned source map records 0=current and 1=reachable prior.
    """

    current = np.asarray(current, dtype=np.float32)
    prior_exact = np.asarray(prior_exact, dtype=np.float32)
    if current.shape != prior_exact.shape or current.ndim != 3:
        raise ValueError("current and prior pools must share shape [B,K,D]")
    if current.shape[1] < 1:
        raise ValueError("candidate pools cannot be empty")
    mixed = np.empty_like(current)
    source = np.empty(current.shape[1], dtype=np.int8)
    mixed[:, 0] = current[:, 0]
    source[0] = 0
    current_index = 1
    prior_index = 1
    for slot in range(1, current.shape[1]):
        if slot % 2 == 1:
            mixed[:, slot] = np.tanh(prior_exact[:, prior_index])
            source[slot] = 1
            prior_index += 1
        else:
            mixed[:, slot] = current[:, current_index]
            source[slot] = 0
            current_index += 1
    return np.ascontiguousarray(mixed), source


def replace_anchor(
    primary: np.ndarray,
    anchor_source: np.ndarray,
    *,
    candidate_axis: int = 1,
) -> np.ndarray:
    """Return ``primary`` with candidate zero replaced from ``anchor_source``."""

    primary = np.asarray(primary)
    anchor_source = np.asarray(anchor_source)
    if primary.ndim < 2 or primary.ndim != anchor_source.ndim:
        raise ValueError("anchor arrays must have the same rank of at least two")
    candidate_axis = int(candidate_axis) % primary.ndim
    if primary.shape[:candidate_axis] != anchor_source.shape[:candidate_axis]:
        raise ValueError("anchor and primary leading dimensions differ")
    if primary.shape[candidate_axis + 1 :] != anchor_source.shape[candidate_axis + 1 :]:
        raise ValueError("anchor and primary trailing dimensions differ")
    result = np.array(primary, copy=True)
    destination = [slice(None)] * primary.ndim
    destination[candidate_axis] = 0
    result[tuple(destination)] = anchor_source[tuple(destination)]
    return np.ascontiguousarray(result)


def pairwise_rms(values: np.ndarray) -> np.ndarray:
    """Exact per-state RMS L2 distance over all unordered candidate pairs."""

    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError("values must have shape [B,K,D]")
    candidates = int(values.shape[1])
    if candidates == 1:
        return np.zeros(values.shape[0], dtype=np.float64)
    centered = values - values.mean(axis=1, keepdims=True)
    summed_squared = np.square(centered).sum(axis=(1, 2))
    return np.sqrt(np.maximum(0.0, 2.0 * summed_squared / (candidates - 1)))


def _pairwise_sign_agreement(
    left: np.ndarray, right: np.ndarray, *, tolerance: float = 1e-12
) -> tuple[np.ndarray, np.ndarray]:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("pairwise inputs must share shape [B,K]")
    if left.shape[1] == 1:
        return (
            np.ones(left.shape[0], dtype=np.float64),
            np.zeros(left.shape[0], dtype=np.int64),
        )
    first, second = np.triu_indices(left.shape[1], k=1)
    delta_left = left[:, first] - left[:, second]
    delta_right = right[:, first] - right[:, second]
    valid = (np.abs(delta_left) > tolerance) & (np.abs(delta_right) > tolerance)
    agreement = (delta_left > 0.0) == (delta_right > 0.0)
    counts = valid.sum(axis=1)
    matched = (agreement & valid).sum(axis=1)
    scores = np.divide(
        matched,
        counts,
        out=np.ones_like(matched, dtype=np.float64),
        where=counts > 0,
    )
    return scores, counts


def _rank_rows(values: np.ndarray) -> np.ndarray:
    """Tie-aware average ranks, implemented without a SciPy dependency."""

    values = np.asarray(values, dtype=np.float64)
    ranks = np.empty_like(values, dtype=np.float64)
    for row_index, row in enumerate(values):
        order = np.argsort(row, kind="mergesort")
        sorted_values = row[order]
        start = 0
        while start < len(row):
            stop = start + 1
            while stop < len(row) and sorted_values[stop] == sorted_values[start]:
                stop += 1
            ranks[row_index, order[start:stop]] = 0.5 * (start + stop - 1)
            start = stop
    return ranks


def _row_spearman(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if left.shape[1] == 1:
        return np.ones(left.shape[0], dtype=np.float64)
    rank_left = _rank_rows(left)
    rank_right = _rank_rows(right)
    rank_left -= rank_left.mean(axis=1, keepdims=True)
    rank_right -= rank_right.mean(axis=1, keepdims=True)
    numerator = (rank_left * rank_right).sum(axis=1)
    denominator = np.sqrt(
        np.square(rank_left).sum(axis=1) * np.square(rank_right).sum(axis=1)
    )
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 0.0,
    )


def state_metrics(
    *,
    checkpoint_step: int,
    proposal_seed: int,
    method: str,
    view: str,
    state_ids: np.ndarray,
    noise: np.ndarray,
    actions: np.ndarray,
    teacher_heads: np.ndarray,
    student_heads: np.ndarray,
    prefixes: Iterable[int],
    tanh_generated: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """Compute all G1 metrics with state, not candidate, as the row unit."""

    noise = np.asarray(noise, dtype=np.float64)
    actions = np.asarray(actions, dtype=np.float64)
    teacher_heads = np.asarray(teacher_heads, dtype=np.float64)
    student_heads = np.asarray(student_heads, dtype=np.float64)
    state_ids = np.asarray(state_ids, dtype=np.int64)
    if noise.ndim != 3 or actions.ndim != 3:
        raise ValueError("noise/actions must have shape [B,K,D]")
    if teacher_heads.ndim != 3 or student_heads.ndim != 3:
        raise ValueError("Q arrays must have shape [H,B,K]")
    if teacher_heads.shape != student_heads.shape:
        raise ValueError("teacher and student head shapes differ")
    if teacher_heads.shape[1:3] != noise.shape[:2]:
        raise ValueError("Q arrays and candidate pools are misaligned")
    if len(state_ids) != noise.shape[0] or teacher_heads.shape[0] < 2:
        raise ValueError("invalid state IDs or missing twin heads")
    if tanh_generated is None:
        tanh_generated = np.ones(noise.shape[1], dtype=bool)
    tanh_generated = np.asarray(tanh_generated, dtype=bool)
    if tanh_generated.shape != (noise.shape[1],):
        raise ValueError("tanh-generated mask must have shape [K]")

    rows: list[dict[str, Any]] = []
    for candidates in prefixes:
        candidates = int(candidates)
        if not 1 <= candidates <= noise.shape[1]:
            raise ValueError("prefix lies outside the candidate pool")
        local_noise = noise[:, :candidates]
        local_actions = actions[:, :candidates]
        local_teacher_heads = teacher_heads[:, :, :candidates]
        local_student_heads = student_heads[:, :, :candidates]
        teacher = local_teacher_heads.min(axis=0)
        student = local_student_heads.min(axis=0)
        batch_indices = np.arange(noise.shape[0])
        oracle_indices = np.argmax(teacher, axis=1)
        selected_indices = np.argmax(student, axis=1)
        anchor_teacher = teacher[:, 0]
        anchor_student = student[:, 0]
        oracle_teacher = teacher[batch_indices, oracle_indices]
        selected_teacher = teacher[batch_indices, selected_indices]
        selected_student = student[batch_indices, selected_indices]
        oracle_lift = oracle_teacher - anchor_teacher
        selected_lift = selected_teacher - anchor_teacher
        predicted_lift = selected_student - anchor_student
        selector_capture = np.divide(
            selected_lift,
            oracle_lift,
            out=np.zeros_like(selected_lift),
            where=oracle_lift > 1e-8,
        )
        twin_disagreement = np.abs(
            local_teacher_heads[0] - local_teacher_heads[1]
        )
        selected_disagreement = twin_disagreement[batch_indices, selected_indices]
        oracle_disagreement = twin_disagreement[batch_indices, oracle_indices]
        twin_directional, twin_pairs = _pairwise_sign_agreement(
            local_teacher_heads[0], local_teacher_heads[1]
        )
        ranking_pairwise, ranking_pairs = _pairwise_sign_agreement(student, teacher)
        spearman = _row_spearman(student, teacher)
        top1 = (selected_indices == oracle_indices).astype(np.float64)
        latent_diversity = pairwise_rms(local_noise)
        action_diversity = pairwise_rms(local_actions)
        local_tanh_generated = tanh_generated[:candidates]
        if bool(local_tanh_generated.any()):
            saturation = (
                np.abs(local_noise[:, local_tanh_generated]) >= 0.98
            ).mean(axis=(1, 2))
        else:
            saturation = np.zeros(noise.shape[0], dtype=np.float64)
        boundary_or_outside = (np.abs(local_noise) >= 0.98).mean(axis=(1, 2))
        out_of_envelope = (np.abs(local_noise) > 1.0 + 1e-6).mean(axis=(1, 2))
        finite = (
            np.isfinite(local_noise).all(axis=(1, 2))
            & np.isfinite(local_actions).all(axis=(1, 2))
            & np.isfinite(local_teacher_heads).all(axis=(0, 2))
            & np.isfinite(local_student_heads).all(axis=(0, 2))
        )
        for index, state_id in enumerate(state_ids):
            rows.append(
                {
                    "checkpoint_step": int(checkpoint_step),
                    "proposal_seed": int(proposal_seed),
                    "method": method,
                    "view": view,
                    "state_id": int(state_id),
                    "candidate_count": candidates,
                    "direct_selection_eligible": int(
                        view == PRIMARY_VIEW and method in (METHOD_CURRENT, METHOD_MIX)
                    ),
                    "anchor_teacher": float(anchor_teacher[index]),
                    "anchor_student": float(anchor_student[index]),
                    "oracle_index": int(oracle_indices[index]),
                    "selected_index": int(selected_indices[index]),
                    "oracle_lift": float(oracle_lift[index]),
                    "selected_lift": float(selected_lift[index]),
                    "predicted_lift": float(predicted_lift[index]),
                    "selector_capture_raw": float(selector_capture[index]),
                    "selector_capture_display": float(
                        np.clip(selector_capture[index], 0.0, 1.0)
                    ),
                    "qw_qa_exploitation_gap": float(
                        predicted_lift[index] - selected_lift[index]
                    ),
                    "teacher_twin_directional_agreement": float(
                        twin_directional[index]
                    ),
                    "teacher_twin_pair_count": int(twin_pairs[index]),
                    "teacher_twin_disagreement_median": float(
                        np.median(twin_disagreement[index])
                    ),
                    "selected_twin_disagreement": float(
                        selected_disagreement[index]
                    ),
                    "oracle_twin_disagreement": float(oracle_disagreement[index]),
                    "random_candidate_twin_disagreement": float(
                        twin_disagreement[index].mean()
                    ),
                    "qw_pairwise_accuracy": float(ranking_pairwise[index]),
                    "qw_pair_count": int(ranking_pairs[index]),
                    "qw_spearman": float(spearman[index]),
                    "qw_top1_agreement": float(top1[index]),
                    "latent_pairwise_rms": float(latent_diversity[index]),
                    "decoded_action_pairwise_rms": float(action_diversity[index]),
                    "tanh_saturation_fraction": float(saturation[index]),
                    "tanh_saturation_applicable": int(local_tanh_generated.any()),
                    "tanh_generated_candidate_fraction": float(
                        local_tanh_generated.mean()
                    ),
                    "scaled_boundary_or_outside_fraction": float(
                        boundary_or_outside[index]
                    ),
                    "out_of_envelope_fraction": float(out_of_envelope[index]),
                    "finite": int(finite[index]),
                }
            )
    return rows


def _evaluate_pool(
    model: Any,
    observations: np.ndarray,
    pool: np.ndarray,
    *,
    device: str,
    batch_candidates: int,
) -> tuple[dict[str, np.ndarray], dict[str, int], float]:
    """Decode and label one complete source pool in state-aligned chunks."""

    observations = np.asarray(observations, dtype=np.float32)
    pool = np.asarray(pool, dtype=np.float32)
    states, candidates, _ = pool.shape
    states_per_batch = max(1, int(batch_candidates) // candidates)
    action_parts: list[np.ndarray] = []
    teacher_parts: list[np.ndarray] = []
    student_parts: list[np.ndarray] = []
    started = time.perf_counter()
    with torch.no_grad(), counted_candidate_calls(model) as counts:
        for start in range(0, states, states_per_batch):
            stop = min(states, start + states_per_batch)
            observation_tensor = torch.as_tensor(
                observations[start:stop], device=device, dtype=torch.float32
            )
            pool_tensor = torch.as_tensor(
                pool[start:stop], device=device, dtype=torch.float32
            )
            flat_observations = observation_tensor.repeat_interleave(candidates, dim=0)
            flat_noise = pool_tensor.reshape(-1, pool.shape[-1])
            decoder = model._unscale_noise(flat_noise)
            actions = model._decode_noise_decoder_input(flat_observations, decoder)
            teacher = model.qa_base_target(flat_observations, actions)
            student = model.qw_base(flat_observations, flat_noise)
            action_parts.append(
                actions.reshape(stop - start, candidates, -1).cpu().numpy()
            )
            teacher_parts.append(
                torch.stack([head.reshape(-1) for head in teacher], dim=0)
                .reshape(len(teacher), stop - start, candidates)
                .cpu()
                .numpy()
            )
            student_parts.append(
                torch.stack([head.reshape(-1) for head in student], dim=0)
                .reshape(len(student), stop - start, candidates)
                .cpu()
                .numpy()
            )
    result = {
        "actions": np.ascontiguousarray(np.concatenate(action_parts, axis=0)),
        "teacher": np.ascontiguousarray(np.concatenate(teacher_parts, axis=1)),
        "student": np.ascontiguousarray(np.concatenate(student_parts, axis=1)),
    }
    return result, dict(counts), float(time.perf_counter() - started)


def _counter_snapshot(model: Any) -> dict[str, int]:
    names = (
        "num_timesteps",
        "_n_updates",
        "qa_base_optimizer_steps",
        "qw_base_optimizer_steps",
        "noise_actor_optimizer_steps",
        "qa_joint_optimizer_steps",
        "residual_actor_optimizer_steps",
        "ent_coef_optimizer_steps",
    )
    return {
        name: int(getattr(model, name))
        for name in names
        if hasattr(model, name) and isinstance(getattr(model, name), (int, np.integer))
    }


def _g1_source_hashes() -> dict[str, str]:
    paths = (
        ROOT / "wbsd_g1.py",
        ROOT / "tests/test_wbsd_g1.py",
        ROOT / "wbsd_probe.py",
        ROOT / "tests/test_wbsd_probe.py",
        ROOT / "docs/rfs_hier_v1/WALKER_BASE_SUPPORT_GATED_PLAN.md",
    )
    return {
        str(path.relative_to(ROOT)): sha256_file(path)
        for path in paths
        if path.is_file()
    }


def _find_pool_record(
    checkpoint_record: dict[str, Any], proposal_seed: int
) -> dict[str, Any]:
    matches = [
        record
        for record in checkpoint_record["proposal_records"]
        if int(record["proposal_seed"]) == int(proposal_seed)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one G0 pool for proposal seed {proposal_seed}, found {len(matches)}"
        )
    return matches[0]


def _sum_counts(counts: Iterable[dict[str, int]]) -> dict[str, int]:
    total: defaultdict[str, int] = defaultdict(int)
    for item in counts:
        for key, value in item.items():
            total[key] += int(value)
    return dict(total)


def run_worker(
    *,
    g0_dir: Path,
    output_dir: Path,
    proposal_seed: int,
    device: str,
    batch_candidates: int,
    state_limit: int | None,
    pool_limit: int,
    checkpoint_steps: list[int] | None,
    save_evaluations: bool,
) -> dict[str, Any]:
    """Run one proposal-seed worker across the requested checkpoints."""

    started_at = utc_now()
    started = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "evaluations").mkdir(exist_ok=True)
    g0_summary_path = g0_dir / "summary.json"
    g0_summary = json.loads(g0_summary_path.read_text(encoding="utf-8"))
    if g0_summary.get("gate") != "G0" or g0_summary.get("status") != "COMPLETE":
        raise ValueError("G1 requires a complete G0 matrix")
    if int(proposal_seed) not in [int(value) for value in g0_summary["proposal_seeds"]]:
        raise ValueError("proposal seed is absent from the G0 contract")
    if not 1 <= int(pool_limit) <= int(g0_summary["pool_size"]):
        raise ValueError("pool limit lies outside the G0 pool")
    prefixes = [value for value in PREFIXES if value <= int(pool_limit)]
    if prefixes[-1] != int(pool_limit):
        raise ValueError("pool limit must be one of the declared nested K values")

    state_bank_path = g0_dir / "state_bank.npy"
    state_ids_path = g0_dir / "state_ids.npy"
    if sha256_file(state_bank_path) != g0_summary["state_bank_saved_sha256"]:
        raise ValueError("G0 state-bank hash mismatch")
    if sha256_file(state_ids_path) != g0_summary["state_ids_sha256"]:
        raise ValueError("G0 state-ID hash mismatch")
    state_bank = np.load(state_bank_path, allow_pickle=False)
    state_ids = np.load(state_ids_path, allow_pickle=False)
    if state_limit is not None:
        if not 1 <= int(state_limit) <= len(state_ids):
            raise ValueError("state limit lies outside the G0 state bank")
        state_bank = state_bank[: int(state_limit)]
        state_ids = state_ids[: int(state_limit)]

    checkpoint_records = [
        record
        for record in g0_summary["checkpoints"]
        if checkpoint_steps is None
        or int(record["model_num_timesteps"]) in set(checkpoint_steps)
    ]
    if not checkpoint_records:
        raise ValueError("no requested checkpoints found in G0")
    checkpoint_records.sort(key=lambda item: int(item["model_num_timesteps"]))
    config_path = Path(g0_summary["config"])
    all_rows: list[dict[str, Any]] = []
    completed: list[dict[str, Any]] = []
    global_before = capture_global_rng_state()
    print(
        f"[G1 seed={proposal_seed}] start checkpoints={len(checkpoint_records)} "
        f"states={len(state_ids)} K={pool_limit} device={device}",
        flush=True,
    )
    with isolated_global_rng():
        for checkpoint_record in checkpoint_records:
            step = int(checkpoint_record["model_num_timesteps"])
            checkpoint_started = time.perf_counter()
            checkpoint_path = Path(checkpoint_record["checkpoint"])
            if sha256_file(checkpoint_path) != checkpoint_record["checkpoint_sha256"]:
                raise ValueError(f"checkpoint hash mismatch at {step}")
            pool_record = _find_pool_record(checkpoint_record, proposal_seed)
            pool_path = ROOT / pool_record["pool_file"]
            if sha256_file(pool_path) != pool_record["pool_file_sha256"]:
                raise ValueError(f"G0 pool hash mismatch at {step}")
            with np.load(pool_path, allow_pickle=False) as archive:
                loaded_ids = np.asarray(archive["state_ids"][: len(state_ids)])
                if not np.array_equal(loaded_ids, state_ids):
                    raise ValueError("pool state IDs differ from the shared state bank")
                current_pool = np.ascontiguousarray(
                    archive["g_current"][: len(state_ids), :pool_limit], dtype=np.float32
                )
                prior_pool = np.ascontiguousarray(
                    archive["g_prior_exact"][: len(state_ids), :pool_limit], dtype=np.float32
                )
            if (
                len(state_ids) == int(g0_summary["state_count"])
                and pool_limit == int(g0_summary["pool_size"])
            ):
                if sha256_array(current_pool) != pool_record["pool_hashes"][METHOD_CURRENT]:
                    raise ValueError(f"G0 current-pool content hash mismatch at {step}")
                if sha256_array(prior_pool) != pool_record["pool_hashes"][METHOD_PRIOR]:
                    raise ValueError(f"G0 prior-pool content hash mismatch at {step}")
            mix_pool, mix_source = build_reachable_mix(current_pool, prior_pool)
            load_warnings: list[dict[str, str]] = []
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                model = _load_hierarchy_checkpoint(checkpoint_path, config_path, device)
                load_warnings = [
                    {"category": type(item.message).__name__, "message": str(item.message)}
                    for item in caught
                ]
            modules = _model_modules(model)
            modes = [(module, module.training) for _, module in modules]
            for _, module in modules:
                module.eval()
            hashes_before = model_state_hashes(model)
            counters_before = _counter_snapshot(model)
            if device.startswith("cuda"):
                torch.cuda.reset_peak_memory_stats(torch.device(device))
            source_results: dict[str, dict[str, np.ndarray]] = {}
            source_counts: dict[str, dict[str, int]] = {}
            source_seconds: dict[str, float] = {}
            try:
                for method, pool in (
                    (METHOD_CURRENT, current_pool),
                    (METHOD_PRIOR, prior_pool),
                    (METHOD_MIX, mix_pool),
                ):
                    result, counts, elapsed = _evaluate_pool(
                        model,
                        state_bank,
                        pool,
                        device=device,
                        batch_candidates=batch_candidates,
                    )
                    source_results[method] = result
                    source_counts[method] = counts
                    source_seconds[method] = elapsed
                    print(
                        f"[G1 seed={proposal_seed} step={step}] {method} "
                        f"{len(state_ids) * pool_limit} candidates in {elapsed:.1f}s",
                        flush=True,
                    )
            finally:
                for module, training in modes:
                    module.train(training)
            hashes_after = model_state_hashes(model)
            counters_after = _counter_snapshot(model)
            if hashes_before != hashes_after:
                raise RuntimeError(f"model state changed during read-only G1 at {step}")
            if counters_before != counters_after:
                raise RuntimeError(f"training counters changed during read-only G1 at {step}")

            current = source_results[METHOD_CURRENT]
            prior_native = source_results[METHOD_PRIOR]
            mix = source_results[METHOD_MIX]
            prior_augmentation = {
                "actions": replace_anchor(prior_native["actions"], current["actions"]),
                "teacher": replace_anchor(
                    prior_native["teacher"], current["teacher"], candidate_axis=2
                ),
                "student": replace_anchor(
                    prior_native["student"], current["student"], candidate_axis=2
                ),
            }
            all_tanh = np.ones(pool_limit, dtype=bool)
            anchor_only_tanh = np.zeros(pool_limit, dtype=bool)
            anchor_only_tanh[0] = True
            no_tanh = np.zeros(pool_limit, dtype=bool)
            views = (
                (
                    METHOD_CURRENT,
                    PRIMARY_VIEW,
                    current_pool,
                    current,
                    all_tanh,
                ),
                (
                    METHOD_PRIOR,
                    PRIMARY_VIEW,
                    replace_anchor(prior_pool, current_pool),
                    prior_augmentation,
                    anchor_only_tanh,
                ),
                (METHOD_MIX, PRIMARY_VIEW, mix_pool, mix, all_tanh),
                (METHOD_PRIOR, NATIVE_VIEW, prior_pool, prior_native, no_tanh),
            )
            checkpoint_rows: list[dict[str, Any]] = []
            for method, view, noise, result, tanh_mask in views:
                checkpoint_rows.extend(
                    state_metrics(
                        checkpoint_step=step,
                        proposal_seed=proposal_seed,
                        method=method,
                        view=view,
                        state_ids=state_ids,
                        noise=noise,
                        actions=result["actions"],
                        teacher_heads=result["teacher"],
                        student_heads=result["student"],
                        prefixes=prefixes,
                        tanh_generated=tanh_mask,
                    )
                )
            all_rows.extend(checkpoint_rows)
            actual_counts = _sum_counts(source_counts.values())
            expected_candidates = len(state_ids) * pool_limit * len(METHODS)
            for key in (
                "ddim_decode_candidates",
                "qa_target_candidates",
                "qw_student_candidates",
            ):
                if actual_counts.get(key) != expected_candidates:
                    raise RuntimeError(
                        f"query-count mismatch for {key}: "
                        f"{actual_counts.get(key)} != {expected_candidates}"
                    )
            finite = all(int(row["finite"]) == 1 for row in checkpoint_rows)
            if not finite:
                raise FloatingPointError(f"non-finite G1 metrics at checkpoint {step}")
            evaluation_file: str | None = None
            evaluation_sha256: str | None = None
            if save_evaluations:
                evaluation_path = (
                    output_dir / "evaluations" / f"checkpoint_{step:012d}.npz"
                )
                np.savez_compressed(
                    evaluation_path,
                    state_ids=state_ids,
                    mix_source=mix_source,
                    mix_noise=mix_pool,
                    current_actions=current["actions"],
                    current_teacher=current["teacher"],
                    current_student=current["student"],
                    prior_actions=prior_native["actions"],
                    prior_teacher=prior_native["teacher"],
                    prior_student=prior_native["student"],
                    mix_actions=mix["actions"],
                    mix_teacher=mix["teacher"],
                    mix_student=mix["student"],
                )
                evaluation_file = str(evaluation_path.resolve())
                evaluation_sha256 = sha256_file(evaluation_path)
            peak_memory = (
                int(torch.cuda.max_memory_allocated(torch.device(device)))
                if device.startswith("cuda")
                else 0
            )
            completed.append(
                {
                    "checkpoint_step": step,
                    "checkpoint": str(checkpoint_path.resolve()),
                    "checkpoint_sha256": checkpoint_record["checkpoint_sha256"],
                    "g0_pool": str(pool_path.resolve()),
                    "g0_pool_sha256": pool_record["pool_file_sha256"],
                    "current_pool_sha256": sha256_array(current_pool),
                    "prior_pool_sha256": sha256_array(prior_pool),
                    "mix_pool_sha256": sha256_array(mix_pool),
                    "mix_source_schedule": mix_source.tolist(),
                    "source_query_counts": source_counts,
                    "actual_query_counts": {
                        **actual_counts,
                        "qw_backward_calls": 0,
                        "environment_steps": 0,
                        "optimizer_steps": 0,
                    },
                    "logical_queries_per_primary_method": len(state_ids) * pool_limit,
                    "source_seconds": source_seconds,
                    "elapsed_seconds": float(time.perf_counter() - checkpoint_started),
                    "peak_cuda_memory_bytes": peak_memory,
                    "module_hashes_before": hashes_before,
                    "module_hashes_after": hashes_after,
                    "module_hashes_equal": True,
                    "training_counters_before": counters_before,
                    "training_counters_after": counters_after,
                    "training_counters_equal": True,
                    "finite_metrics": finite,
                    "warnings": load_warnings,
                    "evaluation_file": evaluation_file,
                    "evaluation_file_sha256": evaluation_sha256,
                }
            )
            del model, source_results
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
            print(
                f"[G1 seed={proposal_seed} step={step}] complete "
                f"elapsed={time.perf_counter() - checkpoint_started:.1f}s",
                flush=True,
            )
    global_after = capture_global_rng_state()
    if not global_rng_states_equal(global_before, global_after):
        raise RuntimeError("G1 failed to restore the process RNG state")

    metrics_path = output_dir / "state_metrics.csv"
    _write_csv(metrics_path, all_rows)
    full_contract = (
        len(state_ids) == 512
        and pool_limit == 64
        and [int(item["checkpoint_step"]) for item in completed]
        == [100000, 300000, 500000]
    )
    summary: dict[str, Any] = {
        "material_passport": {
            "origin_skill": "academic-research-suite/experiment-agent",
            "origin_mode": "run",
            "verification_status": (
                "EXECUTED_G1_WORKER" if full_contract else "EXECUTED_G1_SMOKE"
            ),
            "version_label": "wbsd_g1_worker_v1",
        },
        "gate": "G1",
        "status": "COMPLETE",
        "full_contract": full_contract,
        "started_at_utc": started_at,
        "finished_at_utc": utc_now(),
        "proposal_seed": int(proposal_seed),
        "device": device,
        "state_count": int(len(state_ids)),
        "pool_size": int(pool_limit),
        "prefixes": prefixes,
        "batch_candidates": int(batch_candidates),
        "g0_summary": str(g0_summary_path.resolve()),
        "g0_summary_sha256": sha256_file(g0_summary_path),
        "g0_state_bank_sha256": sha256_file(state_bank_path),
        "g0_state_ids_sha256": sha256_file(state_ids_path),
        "state_ids_sha256": sha256_array(state_ids),
        "sampling_contract": {
            "primary_view": "shared current-policy slot-0 anchor plus K-1 named-family candidates",
            "g_current": "current actor pre-tanh Gaussian transformed by its normal tanh",
            "g_prior_exact": "unbounded standard Gaussian exactly mapped by DSRL policy.scale_action",
            "g_mix": "slot 0 current; odd slots tanh(exact prior); even slots current",
            "mix_prior_envelope_transform": "tanh",
            "tanh_saturation_accounting": (
                "only coordinates produced by a tanh transform; exact-prior coordinates "
                "are tracked by boundary/out-of-envelope metrics instead"
            ),
            "nested_prefixes": True,
            "common_random_numbers_from_g0": True,
        },
        "statistical_contract": {
            "primary_unit": "state",
            "proposal_seeds": "repeated measurements, not independent states",
        },
        "query_contract": {
            "methods": list(METHODS),
            "environment_steps": 0,
            "optimizer_steps": 0,
            "backward_calls": 0,
        },
        "global_rng_restored": True,
        "checkpoints": completed,
        "metrics_file": str(metrics_path.resolve()),
        "metrics_file_sha256": sha256_file(metrics_path),
        "metric_rows": len(all_rows),
        "source_hashes": _g1_source_hashes(),
        "git": _git_metadata(),
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
        },
        "elapsed_seconds": float(time.perf_counter() - started),
        "command": sys.argv,
    }
    _write_json(output_dir / "manifest.json", summary)
    _write_json(output_dir / "summary.json", summary)
    (output_dir / "EXECUTION_COMPLETE").write_text(
        "WBSD G1 worker completed; no optimizer, backward, or environment steps.\n",
        encoding="utf-8",
    )
    print(
        f"[G1 seed={proposal_seed}] worker complete elapsed={summary['elapsed_seconds']:.1f}s",
        flush=True,
    )
    return summary


def _read_metric_rows(path: Path) -> list[dict[str, Any]]:
    integer_fields = {
        "checkpoint_step",
        "proposal_seed",
        "state_id",
        "candidate_count",
        "direct_selection_eligible",
        "oracle_index",
        "selected_index",
        "teacher_twin_pair_count",
        "qw_pair_count",
        "finite",
    }
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as stream:
        for raw in csv.DictReader(stream):
            row: dict[str, Any] = {}
            for key, value in raw.items():
                if key in ("method", "view"):
                    row[key] = value
                elif key in integer_fields:
                    row[key] = int(value)
                else:
                    row[key] = float(value)
            rows.append(row)
    return rows


def _state_seed_matrix(rows: list[dict[str, Any]], metric: str) -> np.ndarray:
    states = sorted({int(row["state_id"]) for row in rows})
    seeds = sorted({int(row["proposal_seed"]) for row in rows})
    state_index = {value: index for index, value in enumerate(states)}
    seed_index = {value: index for index, value in enumerate(seeds)}
    matrix = np.full((len(states), len(seeds)), np.nan, dtype=np.float64)
    for row in rows:
        matrix[state_index[int(row["state_id"])], seed_index[int(row["proposal_seed"])]] = float(row[metric])
    if not np.isfinite(matrix).all():
        raise ValueError(f"incomplete or non-finite state/seed matrix for {metric}")
    return matrix


def paired_state_bootstrap_ci(
    matrix: np.ndarray,
    *,
    repetitions: int,
    seed: int,
    statistic: str = "mean",
) -> tuple[float, float]:
    """Cluster bootstrap state IDs while retaining all proposal-seed repeats."""

    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError("bootstrap matrix must be finite [state, proposal_seed]")
    rng = np.random.default_rng(int(seed))
    state_values = matrix.mean(axis=1)
    indices = rng.integers(0, len(state_values), size=(int(repetitions), len(state_values)))
    sampled = state_values[indices]
    if statistic == "mean":
        estimates = sampled.mean(axis=1)
    elif statistic == "median":
        estimates = np.median(sampled, axis=1)
    else:
        raise ValueError("statistic must be mean or median")
    lower, upper = np.quantile(estimates, [0.025, 0.975])
    return float(lower), float(upper)


def _group_seed(key: tuple[Any, ...], base_seed: int) -> int:
    digest = hashlib.sha256(repr(key).encode("utf-8")).digest()
    return int(base_seed) + int.from_bytes(digest[:4], "little")


def aggregate_group(
    rows: list[dict[str, Any]], *, bootstrap_repetitions: int, bootstrap_seed: int
) -> dict[str, Any]:
    first = rows[0]
    key = (
        int(first["checkpoint_step"]),
        str(first["method"]),
        str(first["view"]),
        int(first["candidate_count"]),
    )
    output: dict[str, Any] = {
        "checkpoint_step": key[0],
        "method": key[1],
        "view": key[2],
        "candidate_count": key[3],
        "state_count": len({int(row["state_id"]) for row in rows}),
        "proposal_seed_count": len({int(row["proposal_seed"]) for row in rows}),
        "direct_selection_eligible": int(first["direct_selection_eligible"]),
    }
    metrics = (
        "oracle_lift",
        "selected_lift",
        "predicted_lift",
        "selector_capture_raw",
        "selector_capture_display",
        "qw_qa_exploitation_gap",
        "teacher_twin_directional_agreement",
        "teacher_twin_disagreement_median",
        "selected_twin_disagreement",
        "oracle_twin_disagreement",
        "random_candidate_twin_disagreement",
        "qw_pairwise_accuracy",
        "qw_spearman",
        "qw_top1_agreement",
        "latent_pairwise_rms",
        "decoded_action_pairwise_rms",
        "tanh_saturation_fraction",
        "tanh_saturation_applicable",
        "tanh_generated_candidate_fraction",
        "scaled_boundary_or_outside_fraction",
        "out_of_envelope_fraction",
    )
    matrices: dict[str, np.ndarray] = {}
    for metric in metrics:
        matrix = _state_seed_matrix(rows, metric)
        matrices[metric] = matrix
        state_average = matrix.mean(axis=1)
        output[f"{metric}_mean"] = float(state_average.mean())
        output[f"{metric}_median"] = float(np.median(state_average))
    for metric in ("oracle_lift", "selected_lift"):
        lower, upper = paired_state_bootstrap_ci(
            matrices[metric],
            repetitions=bootstrap_repetitions,
            seed=_group_seed(key + (metric,), bootstrap_seed),
        )
        output[f"{metric}_ci95_low"] = lower
        output[f"{metric}_ci95_high"] = upper
    selected_denominator = output["random_candidate_twin_disagreement_median"]
    output["selected_vs_random_twin_disagreement_ratio"] = float(
        output["selected_twin_disagreement_median"]
        / max(selected_denominator, 1e-12)
    )
    output["oracle_lift_to_twin_disagreement_ratio"] = float(
        output["oracle_lift_median"]
        / max(output["teacher_twin_disagreement_median_median"], 1e-12)
    )
    return output


def evaluate_gates(aggregate_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply the predeclared G1 support and direct-selection gates."""

    primary = [row for row in aggregate_rows if row["view"] == PRIMARY_VIEW]
    lookup = {
        (int(row["checkpoint_step"]), str(row["method"]), int(row["candidate_count"])): row
        for row in primary
    }
    checkpoints = sorted({int(row["checkpoint_step"]) for row in primary})
    support: dict[str, Any] = {}
    for method in METHODS:
        checkpoint_details: dict[str, Any] = {}
        passed_checkpoints = 0
        for checkpoint in checkpoints:
            k64 = lookup[(checkpoint, method, 64)]
            candidate_details: dict[str, Any] = {}
            passing_k: list[int] = []
            for candidates in (2, 4, 8, 16):
                row = lookup[(checkpoint, method, candidates)]
                k64_gain = float(k64["oracle_lift_mean"])
                gain_capture = (
                    float(row["oracle_lift_mean"]) / k64_gain
                    if k64_gain > 1e-12
                    else 0.0
                )
                k64_diversity = float(k64["decoded_action_pairwise_rms_median"])
                diversity_retention = (
                    float(row["decoded_action_pairwise_rms_median"])
                    / k64_diversity
                    if k64_diversity > 1e-12
                    else 0.0
                )
                k1 = lookup[(checkpoint, method, 1)]
                saturation_delta = float(row["tanh_saturation_fraction_mean"]) - float(
                    k1["tanh_saturation_fraction_mean"]
                )
                conditions = {
                    "oracle_ci_above_zero": float(row["oracle_lift_ci95_low"]) > 0.0,
                    "oracle_lift_over_twin_disagreement_ge_1": float(
                        row["oracle_lift_to_twin_disagreement_ratio"]
                    )
                    >= 1.0,
                    "captures_80pct_k64_gain": gain_capture >= 0.80,
                    "decoded_diversity_no_collapse": (
                        float(row["decoded_action_pairwise_rms_median"]) > 1e-6
                        and diversity_retention >= 0.50
                    ),
                    "saturation_delta_le_10pp": saturation_delta <= 0.10,
                }
                passed = all(conditions.values())
                if passed:
                    passing_k.append(candidates)
                candidate_details[str(candidates)] = {
                    "pass": passed,
                    "conditions": conditions,
                    "k64_oracle_gain_capture": gain_capture,
                    "k64_decoded_diversity_retention": diversity_retention,
                    "saturation_delta": saturation_delta,
                }
            checkpoint_pass = bool(passing_k)
            passed_checkpoints += int(checkpoint_pass)
            checkpoint_details[str(checkpoint)] = {
                "pass": checkpoint_pass,
                "passing_k": passing_k,
                "smallest_passing_k": min(passing_k) if passing_k else None,
                "candidates": candidate_details,
            }
        support[method] = {
            "pass": passed_checkpoints >= 2,
            "passed_checkpoints": passed_checkpoints,
            "required_checkpoints": 2,
            "checkpoints": checkpoint_details,
        }

    direct: dict[str, Any] = {}
    for method in (METHOD_CURRENT, METHOD_MIX):
        checkpoint_details = {}
        passed_checkpoints = 0
        for checkpoint in checkpoints:
            candidate_details = {}
            passing_k = []
            for candidates in (2, 4, 8, 16):
                row = lookup[(checkpoint, method, candidates)]
                conditions = {
                    "median_selector_capture_ge_0p50": float(
                        row["selector_capture_raw_median"]
                    )
                    >= 0.50,
                    "selected_lift_ci_above_zero": float(
                        row["selected_lift_ci95_low"]
                    )
                    > 0.0,
                    "selected_twin_disagreement_le_1p25_random": float(
                        row["selected_vs_random_twin_disagreement_ratio"]
                    )
                    <= 1.25,
                }
                passed = all(conditions.values())
                if passed:
                    passing_k.append(candidates)
                candidate_details[str(candidates)] = {
                    "pass": passed,
                    "conditions": conditions,
                }
            checkpoint_pass = bool(passing_k)
            passed_checkpoints += int(checkpoint_pass)
            checkpoint_details[str(checkpoint)] = {
                "pass": checkpoint_pass,
                "passing_k": passing_k,
                "smallest_passing_k": min(passing_k) if passing_k else None,
                "candidates": candidate_details,
            }
        direct[method] = {
            "pass": passed_checkpoints >= 2,
            "passed_checkpoints": passed_checkpoints,
            "required_checkpoints": 2,
            "checkpoints": checkpoint_details,
        }
    direct[METHOD_PRIOR] = {
        "pass": False,
        "eligible": False,
        "reason": "teacher-query control is outside the current tanh actor envelope",
    }
    support_pass = any(item["pass"] for item in support.values())
    direct_pass = any(
        direct[method]["pass"] for method in (METHOD_CURRENT, METHOD_MIX)
    )
    return {
        "support_pass": support_pass,
        "direct_selection_pass": direct_pass,
        "support": support,
        "direct_selection": direct,
        "predeclared_operational_definitions": {
            "twin_safe_oracle": "max over conservative min-of-twin QA target values",
            "diversity_no_collapse": "median decoded pairwise RMS >1e-6 and >=50% of K64",
            "saturation": "among genuinely tanh-generated coordinates only, fraction with abs(w)>=0.98",
            "boundary_or_outside": "all proposal coordinates with abs(w)>=0.98; reported separately from tanh saturation",
            "bootstrap": "cluster state IDs; average four proposal-seed repeats within state",
            "gate_K": [2, 4, 8, 16],
        },
    }


def _write_plots(aggregate_rows: list[dict[str, Any]], output_dir: Path) -> list[str]:
    try:
        import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel
    except ImportError:
        return []
    primary = [row for row in aggregate_rows if row["view"] == PRIMARY_VIEW]
    paths: list[str] = []
    for metric, ylabel, filename in (
        ("oracle_lift_mean", "Conservative oracle lift", "oracle_lift_vs_k.png"),
        ("selected_lift_mean", "Teacher-verified selected lift", "selected_lift_vs_k.png"),
        ("qw_pairwise_accuracy_mean", "QW pairwise accuracy", "ranking_vs_k.png"),
    ):
        figure, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=False)
        for axis, checkpoint in zip(axes, (100000, 300000, 500000)):
            for method in METHODS:
                rows = sorted(
                    [
                        row
                        for row in primary
                        if int(row["checkpoint_step"]) == checkpoint
                        and row["method"] == method
                    ],
                    key=lambda item: int(item["candidate_count"]),
                )
                axis.plot(
                    [row["candidate_count"] for row in rows],
                    [row[metric] for row in rows],
                    marker="o",
                    label=method,
                )
            axis.set_xscale("log", base=2)
            axis.set_xticks(PREFIXES, labels=[str(value) for value in PREFIXES])
            axis.set_title(f"Walker checkpoint {checkpoint // 1000}k")
            axis.set_xlabel("Nested candidates K")
            axis.grid(alpha=0.25)
        axes[0].set_ylabel(ylabel)
        axes[-1].legend(fontsize=8)
        figure.tight_layout()
        path = output_dir / "plots" / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=180)
        plt.close(figure)
        paths.append(str(path.resolve()))
    return paths


def run_aggregate(
    *,
    worker_dirs: list[Path],
    output_dir: Path,
    bootstrap_repetitions: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    started_at = utc_now()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifests: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    worker_inputs: list[dict[str, Any]] = []
    for worker_dir in worker_dirs:
        manifest_path = worker_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not manifest.get("full_contract") or manifest.get("status") != "COMPLETE":
            raise ValueError(f"worker is not a complete G1 contract: {worker_dir}")
        if not manifest.get("global_rng_restored"):
            raise ValueError(f"worker did not restore RNG state: {worker_dir}")
        if not all(item["module_hashes_equal"] for item in manifest["checkpoints"]):
            raise ValueError(f"worker model hash changed: {worker_dir}")
        metrics_path = Path(manifest["metrics_file"])
        if sha256_file(metrics_path) != manifest["metrics_file_sha256"]:
            raise ValueError(f"worker metrics hash mismatch: {worker_dir}")
        manifests.append(manifest)
        rows.extend(_read_metric_rows(metrics_path))
        worker_inputs.append(
            {
                "proposal_seed": int(manifest["proposal_seed"]),
                "manifest": str(manifest_path.resolve()),
                "manifest_sha256": sha256_file(manifest_path),
                "metrics": str(metrics_path.resolve()),
                "metrics_sha256": manifest["metrics_file_sha256"],
            }
        )
    seeds = sorted(int(item["proposal_seed"]) for item in manifests)
    if seeds != [1101, 2202, 3303, 4404]:
        raise ValueError(f"aggregate requires the four declared seeds, got {seeds}")
    reference_contract = manifests[0]["sampling_contract"]
    if any(item["sampling_contract"] != reference_contract for item in manifests[1:]):
        raise ValueError("worker sampling contracts differ")

    grouped: defaultdict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                int(row["checkpoint_step"]),
                str(row["method"]),
                str(row["view"]),
                int(row["candidate_count"]),
            )
        ].append(row)
    aggregate_rows = [
        aggregate_group(
            grouped[key],
            bootstrap_repetitions=bootstrap_repetitions,
            bootstrap_seed=bootstrap_seed,
        )
        for key in sorted(grouped)
    ]
    expected_groups = 3 * 4 * len(PREFIXES)
    if len(aggregate_rows) != expected_groups:
        raise ValueError(
            f"expected {expected_groups} checkpoint/method/view/K groups, "
            f"found {len(aggregate_rows)}"
        )
    gates = evaluate_gates(aggregate_rows)
    aggregate_csv = output_dir / "aggregate_metrics.csv"
    _write_csv(aggregate_csv, aggregate_rows)
    plots = _write_plots(aggregate_rows, output_dir)
    status = (
        "G1_PASS"
        if gates["support_pass"]
        else "G1_FAIL"
    )
    if gates["support_pass"] and not gates["direct_selection_pass"]:
        status = "G1_SUPPORT_PASS_DIRECT_SELECTION_BLOCKED"
    summary: dict[str, Any] = {
        "material_passport": {
            "origin_skill": "academic-research-suite/experiment-agent",
            "origin_mode": "analysis",
            "verification_status": "EXECUTED_G1_AGGREGATE",
            "version_label": "wbsd_g1_aggregate_v1",
        },
        "gate": "G1",
        "status": status,
        "started_at_utc": started_at,
        "finished_at_utc": utc_now(),
        "worker_inputs": worker_inputs,
        "proposal_seeds": seeds,
        "state_count": 512,
        "checkpoint_steps": [100000, 300000, 500000],
        "prefixes": list(PREFIXES),
        "sampling_contract": reference_contract,
        "statistical_contract": {
            "primary_unit": "state",
            "proposal_seed_handling": "average repeats within each state before bootstrap",
            "bootstrap_repetitions": int(bootstrap_repetitions),
            "bootstrap_seed": int(bootstrap_seed),
            "interval": "paired percentile 95% cluster bootstrap over state IDs",
        },
        "gate_results": gates,
        "aggregate_metrics": str(aggregate_csv.resolve()),
        "aggregate_metrics_sha256": sha256_file(aggregate_csv),
        "plots": plots,
        "source_hashes": _g1_source_hashes(),
        "git": _git_metadata(),
        "elapsed_seconds": float(time.perf_counter() - started),
        "command": sys.argv,
    }
    _write_json(output_dir / "manifest.json", summary)
    _write_json(output_dir / "summary.json", summary)
    result_lines = [
        "# WBSD G1 execution result",
        "",
        f"- Status: **{status}**",
        f"- Support gate: **{'PASS' if gates['support_pass'] else 'FAIL'}**",
        f"- Direct-selection gate: **{'PASS' if gates['direct_selection_pass'] else 'BLOCKED'}**",
        "- Scope: Walker seed-1 checkpoints 100k/300k/500k, 512 fixed states, four CRN proposal seeds.",
        "- Mutations: zero optimizer steps, zero backward calls, zero environment steps; all module hashes unchanged.",
        "",
        "## Family decisions",
        "",
        "| Family | Support | Passing checkpoints | Direct selection |",
        "|---|---:|---:|---:|",
    ]
    for method in METHODS:
        support_item = gates["support"][method]
        direct_item = gates["direct_selection"][method]
        result_lines.append(
            f"| {method} | {'PASS' if support_item['pass'] else 'FAIL'} | "
            f"{support_item['passed_checkpoints']}/3 | "
            f"{'PASS' if direct_item.get('pass') else 'BLOCKED'} |"
        )
    result_lines.extend(
        [
            "",
            "The exact prior is a teacher-query control and is never marked behavior-executable.",
            "Gate calculations and every per-K condition are stored in `summary.json`.",
            "",
        ]
    )
    (output_dir / "execution_result.md").write_text(
        "\n".join(result_lines), encoding="utf-8"
    )
    (output_dir / "EXECUTION_COMPLETE").write_text(
        f"WBSD G1 aggregate completed with status {status}.\n",
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command_name", required=True)
    default_g0 = ROOT / "artifacts/wbsd/G0/g0_matrix_walker_seed1_100_300_500k"
    worker = subparsers.add_parser("worker", help="run one proposal-seed worker")
    worker.add_argument("--g0-dir", type=Path, default=default_g0)
    worker.add_argument("--output", type=Path, required=True)
    worker.add_argument("--proposal-seed", type=int, required=True)
    worker.add_argument("--device", default="cuda:0")
    worker.add_argument("--batch-candidates", type=int, default=1024)
    worker.add_argument("--state-limit", type=int)
    worker.add_argument("--pool-limit", type=int, default=64)
    worker.add_argument("--checkpoint-steps", type=int, nargs="+")
    worker.add_argument("--no-save-evaluations", action="store_true")
    worker.add_argument("--print-json", action="store_true")

    aggregate = subparsers.add_parser("aggregate", help="aggregate four full workers")
    aggregate.add_argument("--workers", type=Path, nargs="+", required=True)
    aggregate.add_argument("--output", type=Path, required=True)
    aggregate.add_argument("--bootstrap-repetitions", type=int, default=2000)
    aggregate.add_argument("--bootstrap-seed", type=int, default=20260829)
    aggregate.add_argument("--print-json", action="store_true")
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    if arguments.command_name == "worker":
        summary = run_worker(
            g0_dir=arguments.g0_dir.resolve(),
            output_dir=arguments.output.resolve(),
            proposal_seed=int(arguments.proposal_seed),
            device=str(arguments.device),
            batch_candidates=int(arguments.batch_candidates),
            state_limit=arguments.state_limit,
            pool_limit=int(arguments.pool_limit),
            checkpoint_steps=arguments.checkpoint_steps,
            save_evaluations=not arguments.no_save_evaluations,
        )
    else:
        summary = run_aggregate(
            worker_dirs=[path.resolve() for path in arguments.workers],
            output_dir=arguments.output.resolve(),
            bootstrap_repetitions=int(arguments.bootstrap_repetitions),
            bootstrap_seed=int(arguments.bootstrap_seed),
        )
    if arguments.print_json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(
            json.dumps(
                {
                    "gate": summary["gate"],
                    "status": summary["status"],
                    "summary": str((arguments.output.resolve() / "summary.json")),
                    "elapsed_seconds": summary["elapsed_seconds"],
                },
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
