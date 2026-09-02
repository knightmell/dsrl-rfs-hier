# Walker Base Support Diagnosis — Gated Plan

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-08-29
- Verification Status: CODE-VERIFIED, EXPERIMENT-UNVERIFIED
- Version Label: walker_base_support_gated_plan_v1
- Current Gate: G0 (probe contract and harness)

## 1. Objective

This plan diagnoses and repairs the early Walker base-stage gap without changing
the residual branch or the three-critic role assignment. It answers four ordered
questions:

1. For a fixed replay state `s`, is the reachable high-value latent support
   already present under Gaussian multi-sampling?
2. If support is present, can the current `QW_base` rank and select it reliably?
3. At matched expensive-query budget, does SVGD add value beyond Gaussian
   sampling and independent gradient ascent?
4. Does the selected mechanism improve early base learning and time-to-threshold,
   rather than only an offline critic score?

The experiment family is named **Walker Base Support Diagnosis (WBSD)**. Gate
names `G0`–`G5` are deliberately separate from the existing algorithmic
`BASE/RESIDUAL/JOINT` phases and P6 naming.

## 2. Code-grounded starting point

The current implementation contains a distributional difference that must be
separated from the number of samples:

- Current VS-Hier: `_qw_base_loss()` calls the current conditional noise actor
  once for each replay state and trains `QW_base` on that actor sample.
  See `stable-baselines3/stable_baselines3/dsrl/hierarchical_rfs_dsrl.py`.
- Original DSRL-NA: `update_noise_critic()` samples fresh standard Gaussian
  diffusion noise independently of the replay state, decodes it, and distils the
  action critic into the noise critic. See
  `stable-baselines3/stable_baselines3/dsrl/dsrl.py`.
- Walker latent dimension is 24 (`4` action steps × `6` action dimensions).
- The existing ranking diagnostic has only six candidates. At the current 500k
  Walker checkpoint, QW pairwise accuracy is about `0.69`, but top-1 agreement is
  only about `0.21`. This is enough signal for proposal diagnostics, but not
  enough evidence to execute the QW top-1 candidate directly.

Therefore, a gain from `K > 1` could have three different causes:

1. restoring the DSRL Gaussian teacher distribution;
2. adding more total QA/DDIM pseudo-label queries;
3. specifically sampling multiple `w` values for the same `s`.

The plan contains controls that identify these causes separately.

## 3. What “same sample budget” means

Four budgets must be reported separately. They are not interchangeable.

| Budget | Unit | Effect of using K latents for one state |
|---|---:|---|
| Environment budget | online action-chunk transitions | unchanged |
| Distinct-state budget | unique or sampled replay states | unchanged in state-deep sampling |
| Teacher-query budget | DDIM decodes + `QA_base_target` labels | approximately K× |
| Optimizer budget | optimizer steps | unchanged if the BK loss is averaged in one update |

Consequences:

- At the same **environment** budget, multi-sampling can improve conditional
  coverage because QA supplies counterfactual pseudo-labels without new
  environment interaction. It is not free: decoder/teacher compute increases.
- At the same total **teacher-query** budget, multi-sampling is not guaranteed to
  help. `B states × K latents` trades state coverage against
  `BK states × 1 latent`.
- QA-generated labels do not add new reward evidence. Dense queries can amplify
  QA error, so twin agreement and real rollouts are mandatory later gates.

## 4. Sampling families

All checkpoint probes use pre-tanh variables where applicable. They report two
views from the same draws:

- **augmentation view (primary):** preserve one shared current-policy anchor and
  add `K-1` candidates from the named proposal family;
- **native-source view (secondary):** use K candidates entirely from the named
  family, so the exact DSRL prior and current conditional distribution can be
  compared without pretending their K=1 draws are identical.

### 4.1 Current conditional Gaussian (`G-current`)

For each fixed state, independently sample from the current actor's Gaussian,
then apply its normal tanh transform. This tests whether actor-local support is
merely under-sampled.

### 4.2 Exact DSRL Gaussian teacher (`G-prior-exact`)

Sample decoder noise `epsilon ~ N(0, I)` exactly as original DSRL-NA, decode it,
and map the same noise to the QW coordinate with the original scaling rule. This
is first a **teacher-query distribution**, not an executable actor output.

The exact prior can map some coordinates outside the tanh actor's scaled range.
That is allowed for reproducing the original DSRL teacher distribution, but such
candidates must not be directly selected for behavior before a later reachability
and rollout gate.

### 4.3 Reachable prior / mixture (`G-mix`)

Use one current-policy anchor, then split the remaining candidates between the
current actor and a Gaussian prior restricted to the declared decoder envelope.
This tests a deployable compromise between actor locality and prior coverage.

### 4.4 SVGD (`S-prior-SVGD`)

Initialize the same Gaussian/mixed particles, freeze QW during transport, and run
at most three trust-region SVGD steps in pre-tanh space. Keep particles grouped
as `[state, particle, latent_dim]`; kernels may never mix particles belonging to
different states. SVGD is compared against both unchanged Gaussian particles and
independent QW gradient ascent.

## 5. Core probe metrics

For candidate `w_i` at state `s`, define the conservative teacher and student:

```text
T_i = min_h QA_base_target_h(s, decode(s, w_i))
S_i = min_h QW_base_h(s, w_i)
```

Every method reports:

- `oracle_lift(K)`: `max_i T_i - T_anchor`;
- `selected_lift(K)`: `T_argmax(S_i) - T_anchor`;
- `selector_capture`: `selected_lift / max(oracle_lift, eps)`, clipped only for
  summary display, with raw values retained;
- QW/QA exploitation gap: predicted lift minus teacher-verified lift;
- twin QA directional agreement and absolute disagreement;
- QW pairwise accuracy, Spearman correlation, and top-1 agreement;
- latent and decoded-action pairwise diversity;
- tanh saturation and out-of-envelope fraction;
- DDIM/QA query count, QW forward/backward count, wall time, and peak memory.

The primary statistical unit is a state, not an individual candidate. Confidence
intervals are paired bootstrap intervals over states, stratified by checkpoint
and proposal RNG seed.

## 6. Choosing K instead of guessing K

Use nested candidate pools with:

```text
K in {1, 2, 4, 8, 16, 32, 64}
```

Draw the pool of 64 once and evaluate prefixes, so the oracle curve is paired and
monotone by construction. `K*` is the smallest K that:

1. captures at least 80% of the K=64 twin-safe oracle gain;
2. has a median oracle lift at least as large as median teacher-twin
   disagreement;
3. satisfies the diversity and saturation checks in G1.

For training, use `K_train = min(K*, 8)` initially. Larger K is diagnostic-only
until its marginal return justifies the compute.

Why there is no defensible K before the probe: if one independent draw hits a
meaningfully better region with probability `p`, then

```text
P(at least one hit in K draws) = 1 - (1 - p)^K.
```

Approximate K requirements are:

| Per-draw hit probability p | 50% hit | 80% hit | 90% hit | 95% hit |
|---:|---:|---:|---:|---:|
| 10% | 7 | 16 | 22 | 29 |
| 5% | 14 | 32 | 45 | 59 |
| 2% | 35 | 80 | 114 | 149 |
| 1% | 69 | 161 | 230 | 299 |

Thus `K=8` is only likely to look dramatic when useful regions are fairly common.
`K=16` is the first serious practical screen, `K=32` tests diminishing returns,
and `K=64` estimates the brute-force ceiling. The probe estimates `p` using a
meaningful-improvement threshold equal to the larger of zero and the local twin
uncertainty, then reports the implied K for 80% and 90% hit probability.

## 7. Two-track operating rule

Each gate has two independent tracks.

### Execution track

The execution agent may implement and run only the current gate. It writes:

```text
artifacts/wbsd/<gate>/<run_id>/
  manifest.json
  metrics.csv or metrics.parquet
  summary.json
  plots/
  EXECUTION_COMPLETE
```

The manifest records code commit/dirty diff hash, checkpoint hashes, state-bank
hash, resolved parameters, random seeds, candidate/query counts, commands, start
and end times, and hardware identity. The executor stops after producing the
current gate artifact and does not start the next gate.

### Audit track

The audit agent is read-only and checks at most five items:

1. artifact and hash completeness;
2. frozen-model / no-leak invariant;
3. intended one-factor contrast and budget accounting;
4. primary gate statistic and confidence interval;
5. one degeneracy check (NaN, collapse, saturation, or critic exploitation).

It may spot-recompute a deterministic 5% sample, but it does not perform a full
line-by-line audit unless a discrepancy appears. It writes one verdict:

- `PASS`: the next gate may begin;
- `CONDITIONAL`: no automatic progression; the user decides after reading the
  named limitation;
- `FAIL`: stop and repair or change direction.

Only `PASS` permits automatic progression.

## 8. Gates

### G0 — Probe contract and harness

**Purpose:** make the checkpoint probe causal and read-only before spending GPU
time.

**Execution:**

- Add a standalone WBSD probe; do not put experimental transport into the
  training path yet.
- Load Walker seed-1 checkpoints at 100k, 300k, and 500k from
  `logs/p6/diagnostic_base_gpu_fresh_frozen_ddim_dsrl_na_rfs_hier_2500000chunks_walker_seed1_500k/checkpoints/`.
- Build a fixed, stratified state bank of 512 states from
  `logs/p6-prefill/walker2d-medium-v2_fresh_frozen_ddim_env3001_policy4001_nenv10_tagged_v1.npz`.
- Use four proposal RNG seeds and common random numbers across methods.
- Hash every trainable module before and after; prohibit optimizer steps and
  environment steps.
- Add focused tests for `[B,K,24]` state-particle alignment, nested-prefix
  sampling, exact query counts, RNG restoration, and unchanged model hashes.

**G0 PASS:** all focused tests pass; model hashes are identical before/after;
the same states and candidate prefixes are reused across methods; a small CPU or
single-batch smoke artifact is complete.

### G1 — Gaussian support and ranking sweep

**Purpose:** decide whether brute-force same-state support is enough and estimate
the smallest useful K.

**Execution:** run `G-current`, `G-prior-exact`, and `G-mix` on all three
checkpoints and all four proposal seeds. Four GPUs may split by proposal seed.
No training or behavior selection occurs.

**G1 support PASS:** at least one proposal family, at at least two of the three
checkpoints, satisfies all of:

1. paired 95% CI for K≤16 conservative oracle lift is above zero;
2. median oracle lift / median twin disagreement is at least 1;
3. some K≤16 captures at least 80% of its K=64 twin-safe oracle gain;
4. decoded-action diversity does not collapse and saturation rises by no more
   than 10 percentage points relative to K=1.

**Separate direct-selection gate:** QW-based best-of-K behavior remains blocked
unless, for `G-current` or in-envelope `G-mix`, at at least two checkpoints:

- median selector capture is at least 0.50;
- paired 95% CI for teacher-verified selected lift is above zero; and
- selected-candidate twin disagreement is no more than 1.25× random-candidate
  disagreement.

`G-prior-exact` is a teacher-query control and cannot pass the behavior-selection
gate merely because its QW metrics are good; out-of-envelope particles remain
non-executable by the current tanh actor.

**Decision after G1:**

| Observation | Diagnosis | Next action |
|---|---|---|
| Prior/mix passes; current fails | actor-local support/distribution collapse | test Gaussian teacher augmentation |
| Oracle passes; QW selection fails | support exists, ranking is the bottleneck | augment QW training; do not execute top-1 |
| Native prior is strong and G3 prior-1 closes the learning gap | distribution source, not multi-sampling, is primary | prefer the simpler prior-1 fix |
| Fixed-state K and state-wide control are similar later | extra teacher compute, not s-conditional depth | do not claim multi-w mechanism |
| No Gaussian family passes by K=64 | not a simple sampling-count problem | inspect decoder reachability, QA validity, or state coverage |

### G2 — Matched-budget SVGD checkpoint probe

**Entry condition:** G1 is PASS, and either Gaussian needs an impractical K or
there is evidence that a directed proposal could improve over it. Before SVGD,
a one-step score sanity check must show teacher-verified improvement on more than
55% of states and QW-twin gradient-direction agreement of at least 70%.

**Execution:** fixed `K=16`, the same initial particles, at most three transport
steps, and the same 16 final DDIM/QA labels per state:

1. unchanged Gaussian particles;
2. independent QW gradient ascent;
3. conditional SVGD attraction + repulsion;
4. repulsion-only control.

QW backward compute is reported separately; expensive decoder/QA query counts
must match. Particles are detached before QA labelling.

**G2 PASS:** compared with the best simpler method, SVGD has a positive paired
95% CI for additional teacher-verified lift, the gain is at least 10% of the
Gaussian K=1→16 gain, decoded diversity falls by less than 10%, and the
QW-versus-QA exploitation gap does not increase materially. Otherwise SVGD is
dropped from the training plan.

### G3 — Four-arm 100k causal training pilot

**Purpose:** distinguish sampling source, extra query compute, and same-state
multi-sampling. Walker seed 1 only; stop at 100k for audit.

| Arm | QW teacher query construction | What it identifies |
|---|---|---|
| A | current actor, `K=1` | exact VS-Hier baseline |
| B | exact DSRL Gaussian, `K=1` | sampling-source effect |
| C | selected Gaussian source, `B states × K_train w/state` | source + conditional-depth effect |
| D | same source and same total BK labels, `BK states × 1 w/state` | query-matched state-wide control |

All arms keep environment steps, QA/actor schedules, QW optimizer-step count,
learning rates, network sizes, and evaluation seeds fixed. For C and D, average
the loss over all BK pairs; never sum it. Use micro-batching only as a numerical
implementation detail and verify that it matches the unchunked loss on a toy
batch.

**G3-100k PASS:**

1. no NaN, module-role leak, or query-count mismatch;
2. the proposed arm improves at least two of three held-out QW metrics relative
   to A: Spearman by 0.05, pairwise accuracy by 0.03, or top-1 agreement by 0.05;
3. 50-episode paired evaluation is non-inferior to A within 1.0 D4RL normalized
   score point;
4. tanh saturation rises by no more than 10 points and actor pre-clip gradient
   norm is no more than 2× A.

After audit PASS, resume the surviving arms to 300k and 500k; do not restart them
from newly initialized models.

**G3-500k PASS (early-rescue gate):** the selected arm must meet either of:

- reduce time-to-D4RL-score-85 by at least 20%; or
- improve 0–500k learning-curve AUC by at least 2 normalized-score points;

and it must be non-inferior at 500k within 1 point, with no higher early-fall
rate. Use fixed evaluation environment and policy seeds.

**Attribution rule:**

- B > A, but C ≈ B: restoring the Gaussian source is the result; multi-sampling
  is unnecessary.
- C > B and C > D: evidence supports same-state multi-w depth.
- C ≈ D and both > B: additional teacher-query compute helps, not specifically
  fixed-s multi-sampling.
- Offline QW metrics improve but return does not: do not call the base problem
  solved; investigate actor optimization/clipping next.

### G4 — Robust confirmation and task expansion

Only the single mechanism selected by G3 proceeds.

**First four-GPU wave:** Walker method seeds 1–3 plus HalfCheetah method seed 1.
Existing baselines may be reused only if an audit confirms matching code/config,
prefill, evaluation seeds, and budget; otherwise matched baselines must be rerun.

**G4 PASS:**

- Walker improves the early-rescue metric in at least two of three seeds and has
  a positive aggregate paired effect;
- no Walker seed has a catastrophic final-score regression greater than 2
  normalized points;
- HalfCheetah seed 1 is non-inferior within 2 points and shows no new instability.

If PASS, run HalfCheetah seeds 2–3 for a paper-level result. A third task is not
needed for diagnosis; add Hopper only after the mechanism is stable or when a
benchmark table requires broader coverage.

### G5 — Full hierarchy integration

Continue the selected base mechanism into the unchanged residual schedule. Keep
the residual actor, `QA_joint`, beta schedule, replay lanes, and evaluation
protocol fixed.

**G5 PASS:** base-only early rescue persists, full-hierarchy return is not worse
than the matched VS-Hier control, residual gains are not obtained by erasing the
base improvement, and all compute/environment budgets remain disclosed.

## 9. Agent handoff templates

### Executor prompt

```text
Read docs/rfs_hier_v1/WALKER_BASE_SUPPORT_GATED_PLAN.md completely.
Act only as the execution owner for gate <GX>. Do not start any later gate.
Preserve unrelated dirty-worktree changes. Implement/run only the items listed
under <GX>, write the required artifacts and manifest, run the focused tests,
then stop with EXECUTION_COMPLETE and a concise anomaly list. Do not decide PASS.
```

### Auditor prompt

```text
Read docs/rfs_hier_v1/WALKER_BASE_SUPPORT_GATED_PLAN.md and the artifact directory
for gate <GX>. Perform a read-only audit using only the five audit-card checks.
Spot-recompute 5% only if needed. Do not edit code, resume training, or start the
next gate. Output PASS, CONDITIONAL, or FAIL; list the exact evidence for each
gate criterion and at most three required corrections.
```

## 10. Immediate recommendation

Start with G0 and G1 only. Do not add SVGD to training yet. The simplest plausible
repair is not necessarily “more samples”; it may be restoring the original DSRL
Gaussian teacher source at `K=1`. The four-arm G3 design is what determines
whether same-state multi-sampling itself deserves to become part of the method
and paper narrative.
