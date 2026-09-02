# Walker Gaussian Multi-w Budget Sweep Design

**Date:** 2026-08-31

## Objective

Find a high-performing same-state Gaussian multi-`w` configuration for the
Walker BASE phase quickly, after Gaussian-K1-NoClip reduced the observed BASE
gap to the matched DSRL-NA learning range.  This is a development sweep, not a
fully compute-normalized benchmark.  It must preserve the three-critic credit
semantics and must not enter the residual phase.

## Frozen Algorithm Semantics

Every arm keeps the following settings unchanged:

- QW proposals are standard Gaussian decoder inputs.
- QW labels come from target `QA_base`; online-QA teaching is excluded.
- Noise-actor global-norm clipping is disabled.
- The existing hierarchy-block update order remains unchanged.
- `QA_joint` shadow training remains enabled during BASE.
- The residual actor performs zero optimizer steps.
- QW continues to regress twin-head values with the existing mean-squared loss.
- Actor, QA, learning rates, target updates, replay, prefill, seed and evaluator
  settings match the completed Gaussian-K1-NoClip seed-1 run.

## Multi-w Construction

For each QW optimizer step, sample `B_s` replay states and independently draw
`K` Gaussian decoder inputs for each state.  Repeat each state contiguously in
candidate-major shape `[B_s, K, ...]`, flatten to `B_s*K` rows for DDIM,
target-`QA_base`, and QW execution, and average the twin-head MSE over all
state-candidate rows.  Changing `K` must not multiply the optimizer-step
magnitude.

Teacher and student execution is chunked by an explicit microbatch size so the
largest arm does not require all candidates to reside in GPU memory at once.
Gradient accumulation must be weighted by each microbatch row count so it is
mathematically equivalent to one mean loss over all `B_s*K` rows.

`K=1`, `B_s=batch_size`, and a microbatch at least as large as `B_s` must
preserve the existing RNG order, tensors, loss and optimizer path exactly.

## Approved Screening Matrix

The first four arms run concurrently to 100k chunk transitions with 50k and
100k checkpoints:

| Arm | Distinct states `B_s` | `K` | Teacher rows/QW update | Relative to K1 |
|---|---:|---:|---:|---:|
| K4-B256 | 256 | 4 | 1,024 | 4x |
| K8-B256 | 256 | 8 | 2,048 | 8x |
| K8-B32 | 32 | 8 | 256 | 1x |
| K16-B128 | 128 | 16 | 2,048 | 8x |

The second batch contains the upper-bound arm:

| Arm | Distinct states `B_s` | `K` | Teacher rows/QW update | Relative to K1 |
|---|---:|---:|---:|---:|
| K64-B256 | 256 | 64 | 16,384 | 64x |

K64-B256 starts after the first batch releases GPU capacity.  It is a formal
50k/100k candidate, not a smoke-only run.  Microbatching controls memory but
does not hide its 64x teacher workload; wall time and query count must be
reported.

## Configuration Contract

Add three explicit positive integer fields:

- `rfs_hier_qw_candidates_per_state`
- `rfs_hier_qw_state_batch_size`
- `rfs_hier_qw_teacher_microbatch_size`

The defaults are `1`, the global training batch size, and the total QW teacher
row count respectively.  Preflight records all three values plus
`qw_teacher_queries_per_update = K * B_s`.  Unknown, boolean, zero or negative
values are rejected.  Screening configs must stop at exactly 100k, before the
500k BASE-to-RESIDUAL boundary.

## Gates

### M0 — Static and Unit Gate

- K1 reproduces the existing path and RNG order.
- Observation repetition and Gaussian candidate grouping are correct.
- DDIM, target QA and QW receive exactly `B_s*K` rows.
- Microbatched and unchunked losses/gradients agree within numerical tolerance.
- Loss normalization is invariant to duplicated candidate rows.
- Manifest and resolved configs contain the declared K/B/microbatch/query
  contract.
- All existing source/no-clip tests remain green.

### M1 — 1k Smoke Gate

- All four first-batch configs produce finite losses and a checkpoint.
- Initial source, prefill and replay hashes match within seed.
- `residual_actor_optimizer_steps == 0`.
- Measured peak GPU memory leaves at least 8 GB free before four-way admission.
- No OOM, NaN, dead process or counter drift is present.

### M2 — 50k/100k Screening Gate

At both checkpoints report base-only mean, healthy-only mean, early-fall rate,
minimum return, QW pairwise/top-1 diagnostics, throughput and cumulative
teacher queries.  At 100k, evaluate every arm on the same 100 episode seeds.

Select the smallest arm within one D4RL point of the best stable result.  If
the 50k and 100k leaders disagree by more than two points, continue only the
top two to 150k.  QW ranking is explanatory and is not a hard selection gate.

## Follow-on R-stage Direction

Continue the selected BASE arm to 500k.  The next experiment forks that exact
model/replay/RNG bundle to test BASE-budget-preserving residual co-training.
It keeps the target-`QA_base` teacher and separates update-count allocation
from update-order changes.  The current evidence directly supports an R-stage
optimizer-budget deficit; it does not yet support replacing target QA or
copying native DSRL batch semantics.

## Reporting Scope

This sweep is performance-first.  It need not match DSRL's internal compute,
because DSRL has no same-state conditional multi-`w` construction.  Any paper
claim must nevertheless disclose teacher rows/update, cumulative teacher
queries, wall time and the selected K/state batch.  K8-B32, K8-B256 and
K16-B128 provide the minimum internal controls for depth, state breadth and
teacher budget.
