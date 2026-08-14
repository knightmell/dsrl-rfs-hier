# P6_REPAIR_AND_WIRING unified handoff

## Scope and repository state

- Repair-series outer base: `d16c4f98ab733add5beb6a3de7286428e79f7aaf`
- Current outer HEAD: `18d5be67c78694527b52599c4aaeafaf6e432590`
- Repair-series SB3 base: `2d998b65f2985e483d4e52629e2d841c60c1b991`
- Current SB3 HEAD: `076e5563ceb51ff91c6c7743501d18f0456b13af`
- No new commit was created after the user requested that commits be deferred.
- The current P6 repair changes on top of those HEADs remain uncommitted.
- No matched 100k or 5M run was started.

Existing outer commits in the repair series:

- `bc63887` P6 preflight and seed integrity
- `0b8fa1d` hierarchy invariant pointer
- `70936ad` hierarchy decomposition pointer
- `d18b7c0` matched prefill and transition protocol
- `26eae47` exact fixed-seed evaluator
- `18d5be6` certified resume and production runner

Existing SB3 commits in the repair series:

- `b61a0a7` hierarchy diagnostics
- `a000a5a` hierarchy training invariants
- `076e556` exact action decomposition

Uncommitted owned files:

- `cfg/gym/p6_hopper.yaml`
- `p6_checkpointing.py`
- `p6_evaluation.py`
- `p6_launcher.py`
- `p6_preflight.py`
- `p6_runtime.py`
- `p6_train.py`
- `tests/test_p6_checkpointing.py`
- `tests/test_p6_evaluation.py`
- `tests/test_p6_launcher.py`
- `tests/test_p6_preflight.py`
- `tests/test_p6_runtime.py`
- `tests/test_p6_train_wiring.py`
- `stable-baselines3/stable_baselines3/dsrl/hierarchical_rfs_dsrl.py`
- `stable-baselines3/tests/test_dsrl_rfs_hier_phase2.py`

Explicitly excluded flat-RFS files remain present but are not part of this
implementation:

- `stable-baselines3/stable_baselines3/dsrl/rfs_dsrl.py`
- `stable-baselines3/tests/test_dsrl_rfs.py`
- their export hunks in `stable_baselines3/__init__.py` and
  `stable_baselines3/dsrl/__init__.py`

## Implemented requirements

- Explicit 5M DSRL-NA checkpoint and artifact hash preflight.
- Separate train, evaluation, prefill-environment and prefill-policy seeds.
- Hopper legacy default `n_envs=4`; P6 pilot explicitly uses `n_envs=10`.
- Safe composition bounds and action-space contract.
- Shared warm-start DSRL prefill with terminal-observation correction.
- Identical semantic replay hash contract for control and hierarchy.
- Hierarchy prefill represented by the shared legacy executed action and zero
  residual.
- Torch affine noise unscale and detached decoder input.
- Frozen DDIM parameters and explicit module inference modes.
- Minimum-only critic backup.
- Executed-action `predict()` plus compatible `predict_diffused()`.
- Gradient/update-count invariants for QA/QM/noise/residual.
- Exact-N evaluator with per-episode policy seeds and RNG restoration.
- Chunk, nominal primitive and actual primitive counters.
- Authenticated model/replay/runtime resume bundles.
- Source/config/run/prefill/replay binding on resume.
- Atomic attempt records, manifests, JSON/CSV evaluation and finalization.
- Short test cadence is limited to runs of at most 10k chunks; production
  defaults remain eval/model/replay = `100k/100k/500k`.

## Test commands and exact results

Full tracked hierarchy/config/P6 regression set, excluding flat RFS:

```bash
env PYTHONDONTWRITEBYTECODE=1 \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  PYTHONPATH=/home/mrf/dsrl/stable-baselines3:/home/mrf/dsrl \
  /home/mrf/miniconda3/envs/dsrl/bin/python -m pytest -q \
  tests/test_dsrl_config_entry.py \
  tests/test_p6_runtime.py \
  tests/test_p6_preflight.py \
  tests/test_p6_checkpointing.py \
  tests/test_p6_evaluation.py \
  tests/test_p6_launcher.py \
  tests/test_p6_train_wiring.py \
  stable-baselines3/tests/test_dsrl_rfs_hier_phase1.py \
  stable-baselines3/tests/test_dsrl_rfs_hier_phase2.py \
  stable-baselines3/tests/test_dsrl_rfs_hier_phase3.py \
  stable-baselines3/tests/test_dsrl_rfs_hier_phase4.py \
  stable-baselines3/tests/test_dsrl_rfs_hier_phase5.py
```

Result: `116 passed, 17 warnings in 10.87s`.

No test was skipped in the project environment. The earlier single skip/failure
was caused by running outside the project environment or under a sandbox that
could not create MuJoCo's `mujocopy-buildlock`; the same raw-Hopper seed test
passes with the real MuJoCo execution permissions.

`git diff --check` also passed.

## Matched prefill and zero-residual evidence

- Prefill artifact:
  `/home/mrf/dsrl/logs/p6-prefill/init_5m_env3001_policy4001_nenv10_v2.npz`
- Archive SHA-256:
  `733842856bc37e9f2eff27251e7b4a77a151d626f04458dd23b731631dc3e957`
- Prefill semantic hash:
  `06ee7e775b2157c066c654a3e8fa63b885df85c455f37c2f4fb6220ad0ed64f0`
- Initial replay semantic hash:
  `1933ab11f7d76327cee30fbb67d76a761db393665285d5d27a188b900b6cf9be`
- Prefill chunks/nominal/actual:
  `20,010 / 80,040 / 80,032`
- The 8-step gap is explained by 4 early-termination chunks.

Real 5M checkpoint migration parity:

- actor state error: `0`
- QA state error: `0`
- QA target state error: `0`
- entropy coefficient error: `0`
- decoder-input error: `0`
- zero-residual executed-action error: `0`
- `QM(observation, noise_scaled, 0)` error: `0`
- residual unit maximum: `0`

## Gradient and parameter-change review matrix

| Update block | Parameters allowed to change | Parameters required unchanged | Required input gradient |
|---|---|---|---|
| QA Bellman | QA; QA target only at original Polyak cadence | noise actor, QM, residual, DDIM | none outside QA |
| QM distillation | QM only | QA, QA target, noise actor, residual, DDIM | none into sampled policy modules |
| entropy | `log_ent_coef` only | all policy/critic/DDIM parameters | entropy coefficient |
| noise actor | noise actor only | QM parameters, residual parameters, QA, DDIM | `noise_scaled -> residual(noise_scaled) -> QM -> noise actor` must be nonzero |
| residual actor | residual actor only | noise actor, QA parameters, QM, DDIM | `action_exec -> QA` back to residual must be nonzero |
| train exit | none | DDIM always frozen | no stale non-owned gradients |

The automated suite checks these routes, parameter isolation, module modes,
optimizer persistence and deterministic save/load equivalence. Reviewer should
also inspect that no `no_grad()` blocks the required input-gradient forwards.

## Real 10k production-wiring run

Run directory:

`/home/mrf/dsrl/logs/p6/20260729_hier_init5m_seed1_10k_wiring`

Core manifest evidence:

- run ID: `p6-a2f3d3621b61451baeeafe836a9eb4e5`
- run/training/final-evaluation status: `complete/complete/complete`
- algorithm: `dsrl_na_rfs_hier`
- init checkpoint: explicit official 5M DSRL-NA
- init checkpoint SHA:
  `f6deae068822cd9bc29405e493600c23aed0d0273b9068f8e16b015c7406dc8b`
- DDIM SHA:
  `9a5839d3d172d1e24b5bed0831d49bb3116a7e62fafa223c57c322f4bc7e9121`
- normalization SHA:
  `d05b2943bc39772f7f770dfc0a4df2a9f205cb1e05ae74b585cd0dd382a0742e`
- source-state SHA:
  `e4f35fabdf3a5d07a2b6e02a0a872eea58f5d322fc91ea85dbc4517de7c81383`
- config-contract SHA:
  `019ad7bda52e2e1c7eff223445a2f095080172ef4c7ee5fe24680de22bcc734b`
- train/prefill-env/prefill-policy seeds: `1001/3001/4001`
- `n_envs=10`
- action bounds: `[-1,1]^12`, verified by artifact and wrapper preflight
- critic combine: `min`

The fresh attempt intentionally stopped at 5k and exited with code 75. The
resume attempt loaded:

- `/resume/chunk_000000005000/model.zip`
- `/resume/chunk_000000005000/replay_buffer.pkl`
- `/resume/chunk_000000005000/runtime_state.pt`

It then finished at 10k with:

- `/resume/chunk_000000010000/`
- `/checkpoints/final_model.zip`
- final evaluation JSON/CSV
- six online milestone evaluations at 0/2k/4k/6k/8k/10k
- model checkpoints at 2k/4k/6k/8k/10k
- replay resume bundles at 5k and 10k

Final counters:

- chunk transitions: `10,000`
- nominal primitive steps: `40,000`
- actual primitive environment steps: `39,998`
- skipped primitive steps: `2`
- early-termination chunks: `1`
- terminal chunks: `1`
- timeout chunks: `39`
- QA/QM/noise/residual optimizer steps:
  `10,000/5,000/2,500/2,500`
- hierarchy train calls: `500`
- resume count: `1`
- environment-reset discontinuities: `1`

The primitive gap is therefore exactly explained:
`40,000 - 39,998 = 2 skipped steps`.

Final resume replay semantic hash:
`7d63fc859ec63533f14dbe8ad09daad69f479e90002cea97b2d9ba86591ddd05`.

The final exact-N wiring evaluation used only 2 episodes and is not a
performance claim:

- raw return `3077.2183 +/- 48.1333`
- D4RL score `95.1735`
- primitive episode length `1000`
- early-fall rate `0`

## Current algorithm diagnosis

The wiring and algorithm invariants are functioning, but the 10k trace contains
an early algorithmic warning:

| Metric | First logged value (2k) | Last value (10k) | Interpretation |
|---|---:|---:|---|
| residual unit mean absolute | 0.650 | 0.754 | residual branch rapidly approaches its tanh limits |
| tanh saturation fraction | 1.0% | 17.0% | not catastrophic yet, but rising quickly |
| residual/base L2 ratio | 12.8% | 15.3% | still residual-sized, but no longer near zero |
| action clip fraction | 7.1% | 8.3% | QA gradients are lost on a material fraction of dimensions |
| effective residual L2 | 0.235 | 0.268 | clipping removes part of requested correction |
| QA(exec)-QA(base) | 2.88 | 2.07 | QA predicts a small positive residual benefit |
| QM(w,r)-QM(w,0) | 2.49 | 0.68 | modulation critic's estimated residual benefit is shrinking |
| policy-sample distill MSE | 47.76 | 43.93 | stable in raw units; needs scale-normalized logging |

For the same single online evaluation seed, raw return was `3167.55` at init
and `3029.08` at 10k, a change of `-138.47` (`-4.37%`). This is a real warning,
but cannot be attributed to the hierarchy until an identically warm-started
DSRL control is trained with the same prefill, RNG plan and interaction budget.
One evaluation episode per milestone and two final episodes are intentionally
insufficient for an algorithm conclusion.

The present evidence is consistent with either:

1. ordinary degradation from continued off-policy DSRL training;
2. residual actor exploitation of optimistic QA gradients;
3. QM lag while noise/residual actors move;
4. evaluation noise.

The matched control is what separates (1) from (2)/(3).

## Outstanding review findings

### Blocking a matched 100k

1. **Ambiguous launcher terminal markers.** After successful resume the run
   directory contains both `LAUNCHER_INTERRUPTED` and `LAUNCHER_COMPLETE`.
   `launcher_status.json` is correct, but marker-only automation can
   misclassify the run. A new launcher attempt must archive/remove only the
   previous launcher terminal marker, and tests must prove exactly one current
   launcher terminal marker.

2. **The run is not reconstructible from commits alone.** The manifest
   correctly fingerprints the dirty source, but current P6 repairs are
   uncommitted and the SB3 fingerprint also includes the explicitly excluded
   flat-RFS untracked files. Before a statistical run, commit only the owned
   hierarchy/P6 changes, leave excluded files out, and require a clean
   hierarchy source state. The 10k wiring run remains valid as wiring evidence,
   not as a publishable result.

3. **No matched control result exists yet.** The current hierarchy-only 10k run
   cannot distinguish residual-specific degradation from continued-DSRL
   degradation.

### Required before 100k but not algorithm changes

4. Copy all final training counters into the root manifest, not only
   chunk/nominal/actual. The current full record is authenticated inside
   `runtime_state.pt`, but root-level audit should expose skipped, early,
   terminal and timeout counts directly.

5. Add scale-aware critic diagnostics:
   `distill_rmse/(mean_abs_QA+eps)`, QA/QM mean and standard deviation, and
   policy-sample QA/QM correlation. Raw MSE around 40 cannot be interpreted
   without Q scale. These are logs only; no loss change.

6. Persist a compact final diagnostics summary in JSON so review does not
   depend exclusively on TensorBoard event parsing.

### Non-blocking known constraints

7. Resume is safe-boundary recovery with an explicit environment reset, not an
   exact uninterrupted trajectory. This is correctly recorded as one
   discontinuity.

8. The 10k run uses short cadence and N=1/N=2 evaluation solely to exercise
   wiring. Production defaults remain `100k/100k/500k`.

## Reviewer acceptance criteria

### Protocol and provenance

- Explicit 5M checkpoint/DDIM/normalization SHA values match the manifest.
- Bounds and dimensions match the action-space audit exactly.
- Control and hierarchy share an identical prefill semantic hash and every
  replay-array hash.
- Initial hierarchy action and QM parity errors are each `<=1e-6`.
- Statistical-run source is clean and reproducible from recorded commits.
- Excluded flat-RFS files do not appear in the implementation commit.

### Gradient and module invariants

- The parameter-change matrix above passes for every update block.
- Required input gradients are finite and nonzero.
- DDIM gradients and parameter changes are exactly zero.
- No unexpected gradient remains after `train()`.
- QA/QM/residual modules use training mode only in their own update blocks and
  all inference/evaluation paths restore evaluation mode.
- Save/load executed-action error is `<=1e-6`; counters and optimizer states
  are exact.

### Replay, counters and recovery

- `nominal = chunks * 4`.
- `actual <= nominal`.
- `nominal - actual = skipped_due_to_termination`.
- early-termination records agree with done and per-chunk actual counts.
- Model/replay/runtime/config/source/run hashes agree at every resume bundle.
- Resume rejects stale, non-latest, mismatched or incomplete bundles.
- One resumed run has exactly one current launcher terminal marker.
- Existing run/checkpoint paths are never overwritten.

### Evaluator

- Exactly N complete episodes are returned for arbitrary vector batch sizes.
- Same episode/policy seeds produce return differences `<=1e-6` across batch
  sizes.
- Python, NumPy and Torch CPU/CUDA RNG states are bitwise restored.
- Raw return, D4RL score, primitive length and early-fall rate are present in
  atomic JSON/CSV and TensorBoard.
- Initial, milestone and final evaluations use the approved fixed seed set.

## Required next experiment sequence

1. Fix the two operational blockers and the manifest/log observability items.
2. Commit only owned P6/hierarchy changes and rerun all 116 tests.
3. Rerun a matched 10k canary:
   - DSRL control versus hierarchy;
   - same official 5M checkpoint;
   - same shared prefill and hashes;
   - same training/evaluation seed plan;
   - one training seed;
   - paired 100-episode final evaluation.
4. If wiring and diagnostics are healthy, run matched 100k with three training
   seeds and fixed evaluations at init/10k/50k/100k.
5. Do not start 5M until the matched 100k gate passes and the user sends an
   explicit approval.

Suggested 100k go/no-go conditions:

- zero integrity/test/provenance failures;
- no NaN/Inf and no action-bound violation;
- hierarchy-control final paired raw-return point estimate is positive;
- paired 95% bootstrap lower bound is no worse than `-2%` of control;
- at least two of three training seeds are non-negative;
- no hierarchy-only early-fall increase above 2 percentage points;
- clip fraction remains below 15%;
- tanh saturation remains below 35% and is not monotonically exploding;
- residual/base L2 ratio remains below 20%;
- scale-normalized policy-sample distillation RMSE remains below 2%;
- QA/QM agreement does not systematically deteriorate at later milestones.

These thresholds are screening gates, not claims of statistical significance.
If only one training seed is run, the result remains a canary and cannot approve
a 5M claim.

## Recommendation

Do not start matched 100k yet. First fix launcher marker uniqueness, create a
clean reproducible hierarchy/P6 commit, expose complete termination counters
and add scale-aware critic diagnostics. Then run the matched 10k
control/hierarchy canary. The current 10k result passes production wiring but
does not yet establish that the hierarchy improves DSRL.
