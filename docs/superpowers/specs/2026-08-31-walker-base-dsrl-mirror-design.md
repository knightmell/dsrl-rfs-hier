# Walker BASE DSRL-Mirror Diagnostic Design

**Date:** 2026-08-31

## Objective

Diagnose and close the Walker Phase-B learning gap without changing the
VS-Hier residual architecture or entering Phase R.  The diagnostic must first
separate four mechanisms that currently differ from matched DSRL-NA:

1. QW proposal distribution: current-actor-local versus standard Gaussian.
2. QW teacher network: target QA versus online QA.
3. Update organization: hierarchy blocks versus DSRL-interleaved order and
   batch reuse.
4. Replay sampling/RNG organization: tagged pair sampling versus flat replay
   sampling.

Same-state multi-w is a later method experiment, not part of the DSRL mirror.
It is admitted only after a K=1 mirror explains or closes the BASE gap.

## Existing Evidence and Interpretation

The seed1 source/clip factorial is already running to 250k.  At 50k,
Gaussian-K1-NoClip was stronger than Gaussian-K1-Clip, while Current-K1 was
largely unchanged by clipping.  Earlier Gaussian-K1-Clip reached matched DSRL
near 200k but showed evaluation-tail instability at 250k.  These observations
support Gaussian proposal breadth as a major factor, but they do not establish
that broad QW supervision reliably improves the final actor.

The code audit found these residual differences after enabling Gaussian and
disabling actor clipping:

- VS-Hier labels QW with `qa_base_target`; DSRL labels it with the online QA.
- VS-Hier executes QA, shadow QA-joint, QW, then actor/alpha in blocks; DSRL
  interleaves alpha, QA and actor on one batch, then performs QW updates.
- VS-Hier therefore lets the actor consume QW updated in the same train call;
  DSRL's actor consumes the QW state from the preceding call.
- Phase-B VS-Hier performs QA-joint shadow updates; DSRL has no such module.
- Tagged replay and flat replay are distributionally close in Phase B but use
  different sampling code and RNG sequences.
- VS-Hier freezes QW parameters during actor backward.  DSRL does not, although
  its unused QW gradients are cleared before its optimizer step.  This is an
  implementation difference, not a proposed primary cause.

## Non-goals

- No DSRL actor-clipping arm: clipping DSRL would answer an artificial
  question and is excluded.
- No residual actor, QA-joint learning, beta ramp, or Phase-R transition.
- No simultaneous change of teacher network, update order and replay sampler
  in the first causal comparison.
- No K=16 250k production run before K=4 passes its compute and memory gates.
- No claim based only on one 10-episode evaluation mean.

## Frozen Comparison Contract

All causal arms use Walker2d-medium-v2, seed1 first, the same Frozen-DDIM
checkpoint, normalization artifact, prefill artifact, initial actor/QA/QW
module hashes, total transition budget, batch size, UTD, QW update count,
learning rates, network widths, tau, gamma, entropy settings, target hard-copy
initialization, train frequency, number of environments, checkpoint cadence and
evaluation seed list.  Phase B ends at 500k, while pilots intentionally stop at
250k.  Every manifest must record all new switches and optimizer counters.

`residual_actor_optimizer_steps` must remain zero in every run covered by this
design.

## Nested K=1 Mirror Ladder

The ladder changes one semantic layer at a time:

| Arm | Proposal | QW teacher | Update organization | Phase-B shadow | Replay sampler |
|---|---|---|---|---|---|
| H0 | Gaussian K1 | target QA | hierarchy blocks | on | tagged pairs |
| H1 | Gaussian K1 | online QA | hierarchy blocks | on | tagged pairs |
| H2 | Gaussian K1 | online QA | DSRL-interleaved | off | tagged pairs |
| H3 | Gaussian K1 | online QA | DSRL-interleaved | off | flat-equivalent |
| D | Gaussian K1 | online QA | native DSRL | absent | native flat replay |

H0 is the running `Gaussian-K1-NoClip` arm.  D is the existing matched DSRL-NA
control.  H1 identifies teacher lag.  H1 versus H2 identifies update timing,
batch coupling and shadow-update effects as one organization layer.  H2 versus
H3 identifies the final replay/RNG implementation gap.  If H2 closes the gap,
H3 is optional and serves only as a strict implementation mirror.

The QW-to-actor gradient-isolation difference is retained in H0-H3 because
changing it has no forward/optimizer effect under the audited DSRL code path
and would weaken VS-Hier's safety invariant.  It can be probed read-only if
gradient accumulation is later suspected.

## Configuration Semantics

New switches must be explicit enums or booleans with backward-compatible
defaults:

- `rfs_hier_qw_teacher_network`: `target` or `online`.
- `rfs_hier_phase_b_update_organization`: `hierarchy_block` or
  `dsrl_interleaved`.
- `rfs_hier_phase_b_replay_sampling`: `tagged_pair` or `flat_equivalent`.
- Existing `rfs_hier_qa_joint_shadow_in_b` is set false in DSRL-order arms.
- Existing `rfs_hier_qw_teacher_source` remains `gaussian` for H0-H3.
- Existing `rfs_hier_noise_actor_gradient_clipping` remains false for H0-H3.

The DSRL-interleaved mode must reproduce this Phase-B train-call contract:

1. For each QA/actor gradient step, draw one BASE batch, construct one actor
   sample/log-prob before any optimizer step, and reuse that batch and actor
   sample for alpha, QA-base and noise-actor updates in DSRL order.
2. Apply the QA target update after the actor step, at the same relative
   location and interval as matched DSRL.
3. After all QA/actor steps, draw independent BASE batches for QW updates.
4. Disable QA-joint shadow updates.
5. Preserve the same optimizer-step counts as matched DSRL.

This mode is strictly Phase-B-only.  Preflight must reject it if a pilot can
cross into Phase R or if residual/joint updates are enabled.

## Same-State Multi-w Stage

After the K=1 mirror gate, test depth versus breadth using standard Gaussian
teacher proposals and the winning mirror semantics:

| Arm | Distinct states | w per state | Teacher queries/update | Purpose |
|---|---:|---:|---:|---|
| K1 | 256 | 1 | 256 | mirror reference |
| K4-QM | 64 | 4 | 256 | same compute, more latent depth |
| K4-XC | 256 | 4 | 1024 | depth plus extra teacher compute |

K4 candidates for a state share exactly the same observation and are flattened
only for decode/QA/QW execution.  The loss is the mean over all state-candidate
pairs, so changing K does not multiply the optimizer-step magnitude.  K4-QM
and K1 must have equal decode, QA and QW query counts.  K16 is limited to a
50k stress run unless K4-QM shows an independent gain over K1.

## Gates

### Gate D0 — Static and Unit Contract

- New fields resolve into the manifest and default to current behavior.
- Unknown enum values and Phase-R-incompatible mirror configs are rejected.
- Spy tests verify online versus target QA calls, exact DDIM/QA/QW query counts,
  update order, batch object reuse and optimizer counters.
- Existing focused P6 tests remain green.

### Gate D1 — 1k Smoke

- Correct source, teacher, organization and replay labels in the manifest.
- Initial actor/QA/QW, prefill and normalization hashes match the control.
- All losses/gradients/evaluation values are finite.
- No residual or QA-joint optimizer steps in DSRL-order mode.
- Peak GPU allocation is recorded; at least 8 GB physical free memory remains
  before another production process is admitted.

### Gate D2 — 50k Viability

- No OOM, NaN, dead process, counter drift or checkpoint corruption.
- Exact expected optimizer-step ratios are observed.
- Evaluation uses the frozen episode seed list and reports mean, median,
  standard deviation, minimum and early-fall rate.
- H1/H2 can stop for clear regression, but a single outlier episode alone is
  not sufficient to reject an arm.

### Gate D3 — 250k Attribution

Compare 100k, 150k, 200k and 250k learning curves and paired episode returns.
Use these operational decisions:

- If H0 is within 2 D4RL points of D at both 200k and 250k without worse
  tail-risk, declare the BASE gap provisionally closed and skip directly to
  K4-QM; H1-H3 are optional mechanism checks.
- If the gap is 3-5 points or tail-risk is materially worse, run H1 then H2.
- If H2 remains more than 2 points below D, run H3.
- A mirror is accepted only when return, tail-risk, update counters and initial
  hashes agree; matching a noisy mean is insufficient.

### Gate K — Multi-w Admission

K4-QM has independent value only if it beats K1 under equal teacher-query
count across at least two seeds in learning AUC or tail-risk, not merely one
checkpoint mean.  K4-XC is run only after K4-QM, and distinguishes useful
extra compute from same-state structure.

## Resource Policy

At most four K1 Walker processes run concurrently on the RTX 4090.  Admission
requires a live `nvidia-smi` check, no existing OOM evidence, temperature below
80°C, and at least 8 GB free after initialization.  K4-XC and K16 initially run
alone because their teacher batches are 4x and 16x larger.  Processes are never
killed or restarted automatically; a failed/stalled run is diagnosed
read-only before user-directed recovery.

## Deliverables

- Collision-free configs for H1-H3 and K1/K4 arms.
- Manifest-bound semantic labels, hashes, query counts and optimizer counters.
- Focused tests for call targets, update order, batch reuse and multi-w shapes.
- A 50k gate report and a 250k attribution table against matched DSRL-NA.
- No production claim until at least two seeds support the selected mechanism.
