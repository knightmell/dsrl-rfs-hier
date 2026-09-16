# VS-Hier Project State

Last updated: 2026-09-16

This is the first document to read for ongoing work. It records the current interpretation, not a history of every phase.

## Current method

The active locomotion line is VS-Hier with a frozen DDIM base policy, a latent noise actor, `QA_base`, supervised `QW_base`, `QA_joint`, and an additive residual actor. The current investigated BASE recipe uses a Gaussian QW teacher source, same-state multi-w with explicit `K`, and an explicit actor clipping setting. Recent experiments commonly use `K=4` and `noclip`, but future runs must still record both values explicitly.

Training boundaries are experimental variables. There is no project-wide default BASE/RES split or final transition budget.

## Corrections now in force

- A full checkpoint evaluated with residual disabled is a `checkpoint base view`, not an independently trained `basecontrol`.
- The known Walker full/control continuation pair branched at 500k. It is a legacy local residual diagnostic and cannot represent the 300k-to-600k paired protocol.
- The known Hopper and HalfCheetah 600k full runs branched from 300k. Independent matching base-control artifacts were not found in the latest audit; do not report their checkpoint base views as base-control runs.
- A config file or launch command is not proof that training ran or completed.

## Current evidence policy

Experiment claims require machine artifacts listed in `docs/EXPERIMENT_REGISTRY.md`. Algorithm and code decisions follow `docs/CURRENT_ALGORITHM.md`, `docs/CURRENT_PROTOCOL.md`, and current source code. Archived documents are not authoritative.

## Current experiment state

The local host currently has four live seed-1 learners: paired VS-Hier and
matched DSRL runs for Robomimic Can and D3IL Avoid-M1. Both pairs loaded their
task-specific verified shared prefills and demonstrated transition and
optimizer progress. The four local Square seed-1/seed-2 learners were stopped
at the user's request on 2026-09-16 so Square can be rerun on another machine;
none had reached its first resumable 100k boundary and none is result evidence.
Their exact paths and terminal qualifications are registered below.
Historical `running` fields remain non-evidence unless a live process and the
latest launcher state agree.

### Robomimic Can extension

Can is the first manipulation benchmark added to the locomotion suite. The
initial matched comparison uses action chunks of four and a 300k-chunk budget
(1.2M nominal primitive actions) for both methods. VS-Hier uses 100k BASE plus
200k additive RES; matched DSRL trains its base pathway for the same interaction
budget from the same frozen diffusion policy and shared seeded prefill.

The 300k boundary is a predeclared convergence check, not an assumption that
training must stop there. Fixed-seed evaluations at 200k and 300k determine
whether success has plateaued. If success is still materially improving, both
methods continue from their original state to the same 400k boundary. An
unfinished rising endpoint is not used as the final paper comparison.

Square uses a 500k-chunk initial endpoint (2.0M nominal primitive actions),
with 200k BASE plus 300k RES for VS-Hier. Avoid-M1 (`d56_r12`) uses a 100k-
chunk initial endpoint (0.4M nominal primitive actions), with 25k BASE plus
75k RES. Each task's matched DSRL run has the same frozen policy, seeds,
prefill, environment count, and interaction endpoint as its VS-Hier run.

### Current VS-Hier paired evidence

Two independent 300k-branch pairs reached their intended 800k boundary and
ended through the planned safe interruption (`exit 75`):

| Pair | Full: 800k 10ep D4RL / early-fall | Independent base-control: 800k 10ep D4RL / early-fall | Evidence state |
|---|---:|---:|---|
| Hopper seed 3 | 98.03 / 0% | 95.84 / 10% | Terminal model and resume bundle present; full used the documented 300k schedule repair after a discarded wrong-schedule attempt. |
| Walker2d seed 1 | 105.79 / 0% | 98.31 / 0% | Terminal model and resume bundle present; both preserve the declared historical 300k branch lineage, although their final continuation resumed from available 600k state. |

These are 10-episode checkpoint diagnostics, not multi-seed final claims.
The Hopper full checkpoint's residual-disabled base view is 97.69 / 0%; the
Walker full checkpoint's base view is 98.06 / 0%. They must not be substituted
for the independent controls above. All exact paths and historical-attempt
qualifications are in `docs/EXPERIMENT_REGISTRY.md`.

### Architecture and credit diagnostics

Two distinct diagnostic families now exist and must remain separate in writing:

- **Old dual-head, 800k re-evaluations:** Hopper s3 has full/base-view D4RL
  94.96/66.62 with 97% base-view early fall, while HalfCheetah s3 is
  49.55/47.69 and Walker s3 is 97.28/95.92. Thus the pronounced residual-
  masking observation is presently Hopper-specific in that old architecture.
- **Current three-critic Joint-Credit variant:** Hopper and Walker reached
  800k with full/base-view 97.16/92.82 and 105.82/96.23 under 100 episodes.
  HalfCheetah remains in progress; its 600k 20-episode process point is
  49.80/47.55 and is not a final cross-task result.

Both families use same-checkpoint residual-off base views, not independent
base-control runs. They are useful mechanism evidence but cannot alone assign
the old Hopper degradation to shared Joint-Credit, nor prove a causal
advantage of separated critics until matched-start/state/config checks are
recorded. The registry contains the exact protocol and evidence qualifiers.

### External baseline state

- **Policy Decorator (matched):** Hopper, HalfCheetah, and Walker2d seed-1
  runs all reached 800k chunks / 3.2M primitive actions. Their full 100k--800k
  10-episode curves are recorded in the registry; final 100-episode checks
  remain pending. Ordinary Residual-SAC / classical Residual RL is cancelled.
- **DPPO:** older 800k artifacts and fresh seed-1 3.2M-primitive-action runs
  exist for all three tasks. The 3.2M runs are training-complete; final policy
  evaluations are still pending.
- **DIPO:** seed-1 3.2M-primitive-action training artifacts exist for all
  three tasks, with finite logged metrics; final policy evaluations are
  pending.

The local bottleneck observed during prior multi-run waves was host memory from
replay states, not GPU memory. Treat every future concurrency choice as a
fresh resource decision rather than relying on historical process counts.

## Maintenance rule

Before launching or summarizing an experiment, update its registry row with the intended canonical name and actual artifact path. After termination, record its terminal state and evaluation evidence in the same row.
