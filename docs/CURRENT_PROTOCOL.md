# Current Experiment Protocol

Last updated: 2026-09-16

This is the only current operational protocol. Old P6 runbooks and gate documents are archived.

## Run naming

Canonical format:

```text
{task}_s{seed}_{clip|noclip}_k{K}_{branch_k}k_{final_k}k_{full|basecontrol}
```

Examples:

```text
hopper_s1_noclip_k4_300k_600k_full
hopper_s1_noclip_k4_300k_600k_basecontrol
walker_s3_clip_k8_175k_925k_full
```

Use lowercase `noclip`; `npclip` is treated as a typo, not a second status. `branch_k` is the actual shared checkpoint where the pair diverges. `final_k` is the intended final number of chunk transitions. Neither value has a default. Use `project_run_naming.py` to build or parse names.

Historical directories are not renamed automatically because their paths may be embedded in manifests and resume chains. Give them a canonical alias in the experiment registry.

## Paired branch semantics

At the declared branch checkpoint, full and base-control must share the same model, optimizer, replay, RNG, source hash, and BASE update schedule.

- `full`: preserves the agreed BASE updates and adds the configured residual or joint updates.
- `basecontrol`: preserves the same BASE updates and performs no residual actor optimization. Existing profiles may retain the shared BASE-phase QA_joint shadow updates; distinguish these from residual-driven joint training rather than claiming all joint-critic counters are zero.

Evaluation-time residual disabling produces only a checkpoint base view.

## Launch and reporting

A launch is blocked only by a condition that can invalidate or corrupt the run: missing or incomplete resume source, incompatible state, output-path collision, invalid resolved config, unavailable requested CUDA device, or unsafe resources likely to cause OOM. Names are validated for parseability but are not coupled to a historical schedule or fixed budget.

Every launch immediately records `RUNNING`. Every process termination records one of `COMPLETE`, `INTERRUPTED`, or `FAILED`, including the latest transition and concise reason. A launched run may never disappear without a terminal report.

The completed local Walker seed-1 pair is the reference example: both
canonical `300k_800k` child runs record the same complete 300k source bundle
in their branch lineage. Full enters RES with UTD=4 while preserving BASE
updates; base-control stays in BASE with RES UTD=0. Their final continuation
used available 600k resume state, which does not rewrite their 300k historical
branch. A prior Walker pair with a 500k branch remains a legacy diagnostic and
cannot be relabeled as this protocol.

Monitoring checks process liveness, transition progress, new checkpoints, terminal status, and material resource failure. Routine monitoring does not re-audit old gates or historical provenance.

Before launch, present the resolved experiment-defining parameters: task, seed, clip status, K, role, branch/final transitions, QW source/query budget, BASE/RES UTD, beta/lane settings, environment count, resume source, and output path. Parameter correctness is the first gate.

For a schedule-changing branch, check the **serialized model schedule** as well as the config: normal resume loads the saved schedule, so CLI overrides alone do not change the branch point. This single parameter check would have caught the Hopper s3 500k/300k mismatch. Keep the original bundle, explicitly record any authorized schedule-only repair, and exclude wrong-attempt metrics. A later 600k resume does not change an experiment's historical 300k branch name.

Use only focused checks needed for that run. Broad test suites, repeated hash audits, and historical P6 gates are not default launch requirements. If another constraint blocks training, report the exact constraint, its source and purpose, and the options to keep, relax, or remove it; the user decides before further work.

## Throughput overlay for new runs

The optional Hydra override `+runtime=fast` is the current throughput-oriented
configuration for **new** VS-Hier launches. It is explicit in the resolved
config and manifest. It preserves model architecture, K/query source,
optimizer update counts, UTD, replay/RNG semantics, and residual composition;
it changes only runtime overhead:

- skips redundant per-update gradient-isolation and executed-action rechecks;
- uses a QW teacher microbatch ceiling of 1024 rows (one batch for K=4, B=256);
- leaves checkpoint and evaluation cadence unchanged from the selected task
  configuration.

Strict mode remains the default and must be used for diagnosis whose purpose is
to validate those per-update checks. Do not apply this overlay to a process
already running or silently to a resume chain with a different resolved config.

## Documentation

Update the living documents rather than creating another phase or handoff document. Do not consult `docs/archive/` unless current run artifacts and living documents cannot resolve a historical path, budget, or decision. Archived content must be confirmed by current artifacts or by the user before use.

## Checkpoint evaluation schedule

The default evaluation policy is intentionally lightweight:

- checkpoints before 500k are not evaluated by default;
- from 500k onward, evaluate each 100k checkpoint with 10 episodes;
- evaluate the final 800k checkpoint with 100 episodes, replacing its 10-
  episode check;
- use the current run manifest, resolved config, and existing evaluator only;
- distinguish an independently trained `basecontrol` from a full checkpoint’s
  residual-disabled `checkpoint base view`.

The detailed operational instructions live in
`/home/mrf/.agents/skills/running-research-evaluation/SKILL.md`. A different
final budget or an evaluation before 500k requires an explicit user request.

### Manipulation benchmark learning curves

For new Robomimic/D3IL runs, each saved curve checkpoint is evaluated on 100
fixed-seed episodes and has a matching model snapshot and resume bundle. The
cadence is task-specific and belongs in the task YAML, not in a runtime overlay:

| Task | Curve milestones (chunk transitions) | Rationale |
|---|---|---|
| Can | 0, 50k, 100k, 150k, 200k, 250k, 300k | Seven points across the 300k initial budget. |
| Square | 0, 100k, 200k, 300k, 400k, 500k | Square's 100-step DDIM evaluation makes a denser cadence disproportionately costly. |
| Avoid-M1 | 0, 25k, 50k, 75k, 100k | Five points expose fast saturation in the short 100k budget. |

This policy applies to future launches only. A completed run cannot recover an
unwritten intermediate model; its existing short evaluation remains historical
evidence and must not be relabeled as a 100-episode checkpoint.

At intermediate VS-Hier curve nodes, evaluate `current_full_hierarchy` only;
at matched DSRL nodes, evaluate `current_base_only` only. This makes each
plotted method point exactly 100 episodes rather than silently spending 300
episodes on full/base/reference diagnostic views. The final checkpoint retains
all views for mechanism analysis.

## Budget accounting across methods

Use **nominal primitive environment actions** for cross-method interaction
plots. With `act_steps=4`, the conversion is:

| Method family | Native logged counter | Conversion to nominal primitive actions | Meaning of 800k native / 3.2M native |
|---|---|---|---|
| VS-Hier / P6 matched DSRL | chunk transitions (one decoded action chunk per environment) | `chunks × 4` | 800k chunks = 3.2M nominal primitive actions |
| Policy Decorator (matched) | chunk decisions | `chunk decisions × 4` | 800k decisions = 3.2M nominal primitive actions |
| DPPO / DIPO | `total env step` | already incremented by `n_envs × act_steps` per vector action | 3.2M env steps = 800k action-chunk decisions; 800k env steps is only 200k chunk decisions |

P6 additionally records `actual_primitive_env_steps`: an action chunk stops at
termination, so this can be lower than its nominal `chunks × 4` count. The
current DPPO/DIPO loop increments `total env step` by a full `act_steps` even
when a trajectory terminates inside that chunk. Therefore the completed 3.2M
DPPO/DIPO runs are matched to P6 at **nominal interaction budget**, not at an
exact post-termination action count. Do not claim exact executed-action parity
unless the external baseline records equivalent termination-aware counters.

Optimizer updates, denoising-step updates, teacher queries, and wall-clock time
are method-specific and are not made equal by the interaction conversion.

### Convergence-based benchmark endpoints

A declared final budget is a common matched checkpoint, not automatic evidence
of convergence. For a new benchmark, compare identical fixed-seed evaluations
at the final two checkpoints. If both learning curves have not reached a
credible plateau, continue every method in that comparison from its saved
state to the same next boundary; do not select different endpoints after seeing
which method benefits. For Robomimic Can, 300k chunks equals 1.2M nominal
primitive actions. It is the initial common endpoint because the published DSRL
Can curve spans about 1M primitive timesteps. Evaluate both methods for the
same 100 fixed-seed episodes at 200k and 300k. If either method improves by at
least five success-rate percentage points from 200k to 300k and its online
success trace has not flattened, continue both methods equally to 400k;
otherwise 300k is the shared endpoint. This benchmark-specific request
overrides the usual rule that skips evaluations before 500k.

For Robomimic Square, use fixed-seed 100-episode success evaluations at 400k
and 500k; if either method gains at least five percentage points and its online
trace has not flattened, continue both equally to 600k. For D3IL Avoid-M1, use
the same rule at 50k and 100k, with an equal continuation to 150k. Square
counts 500k chunks as 2.0M nominal primitive actions; Avoid counts 100k chunks
as 0.4M nominal primitive actions.
