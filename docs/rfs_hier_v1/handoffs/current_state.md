# Hierarchical DSRL-NA Residual Modulation V1 — Paused State

## Pause status

- Execution was paused on 2026-07-27 at the Phase 2 independent-review gate.
- Phase 0 is complete and has an independent Reviewer `PASS`.
- Phase 1 is complete and has an independent Reviewer `PASS`.
- Phase 2 implementation and its read-only checkpoint/QM specialist inspection are complete.
- The fresh Phase 2 independent Reviewer was interrupted immediately after the pause request. It produced no verdict. Therefore Phase 2 is **not accepted** and Phase 3 has not started.
- No new phase may start until the user sends the exact approval string required for that phase, for example `APPROVE_PHASE_2` to resume the Phase 2 review gate.

## Repository and commit state

| Repository | Branch/state | Original base | Head before this WIP snapshot | Implemented head |
|---|---|---|---|---|
| Outer `/home/mrf/dsrl` | `exp/locomotion-stage0` | `8d21b9cf55459f022c0e769bca1343edd9d30e5a` | `4f94ac1784fe2fdd25180b7f8bfa08b6849dd355` | this document's WIP commit; resolve with `git log -1 --format=%H` |
| SB3 `/home/mrf/dsrl/stable-baselines3` | detached submodule HEAD | `10e5d311b5c42fe571ff3eeb72f622ca616a149d` | `0e3d4e62f7cec247d4b43be3f7d191094ac17ba9` | `0e3d4e62f7cec247d4b43be3f7d191094ac17ba9` |
| DPPO `/home/mrf/dsrl/dppo` | unchanged by this work | `86ce51834055c02f9013e60dd4c4275606d82df7` | unchanged | unchanged |

The outer WIP commit records the SB3 pointer at the Phase 2 implementation head. It is made on `exp/locomotion-stage0`; it is not merged into `main`.

Phase commits already created:

- Outer Phase 0 audit: `30ac9ebecb6f3217450c7595c2bdafd0b9d10861`.
- Outer Phase 0 Reviewer record: `1609e8e587b29ae07b1bf35d1bc05272a823a4d9`.
- SB3 Phase 1 implementation: `70511336eb663a31921c25bef35c23d6759fe748`.
- Outer Phase 1 specialist record: `23fb867`.
- Outer Phase 1 Reviewer record: `d295365bdffc4da6ba9a5bf9c17c4984bec5652c`.
- SB3 Phase 2 implementation: `0e3d4e62f7cec247d4b43be3f7d191094ac17ba9`.
- Outer Phase 2 specialist record: `4f94ac1784fe2fdd25180b7f8bfa08b6849dd355`.
- There is no Phase 2 Reviewer record because that review was interrupted before a verdict.

## Files modified by this staged implementation

Outer documentation:

- `docs/rfs_hier_v1/action_space_audit.md`
- `docs/rfs_hier_v1/handoffs/phase0_dataflow_mapper.md`
- `docs/rfs_hier_v1/handoffs/phase0_checkpoint_inspector.md`
- `docs/rfs_hier_v1/handoffs/phase0_reviewer.md`
- `docs/rfs_hier_v1/handoffs/phase1_bounds_validator.md`
- `docs/rfs_hier_v1/handoffs/phase1_reviewer.md`
- `docs/rfs_hier_v1/handoffs/phase2_checkpoint_inspector.md`
- `docs/rfs_hier_v1/handoffs/current_state.md`
- Outer `stable-baselines3` gitlink only, in this WIP snapshot.

SB3 implementation and tests:

- `stable-baselines3/stable_baselines3/dsrl/hierarchical_rfs_dsrl.py`
- `stable-baselines3/tests/test_dsrl_rfs_hier_phase1.py`
- `stable-baselines3/tests/test_dsrl_rfs_hier_phase2.py`

## Files and scopes not modified by this staged implementation

- `stable-baselines3/stable_baselines3/dsrl/dsrl.py` is unchanged from the SB3 base; SHA-256 is `41dc8cfb86b8640284eb29f9b5b585e9906e93c30e7dfe684dd7da8adcbf6bc8`.
- Common ReplayBuffer code is unchanged; no custom replay exists.
- SAC actor/policy implementations are unchanged.
- Existing flat/shared-output `RFSDSRL` work is unchanged by this implementation.
- No hierarchy export, `algorithm=dsrl_na_rfs_hier` entry, Hydra configuration, or training-entry integration has been added.
- Phase 3 QA/QM training losses have not been implemented.
- Phase 4 noise/residual actor losses and `20/10/5/5` scheduling have not been implemented.
- Phase 5 logging/configuration/entry work has not been implemented.
- Phase 6 smoke, 10k, 100k, and 5M runs have not been started.
- VLA-JEPA has not been modified.

The following working-tree items predated or are outside this staged implementation and remain deliberately uncommitted:

Outer repository:

- `cfg/gym/dsrl_halfcheetah.yaml`
- `cfg/gym/dsrl_hopper.yaml`
- `cfg/gym/dsrl_walker.yaml`
- `train_dsrl.py`
- `utils.py`
- dirty DPPO submodule worktree

SB3 worktree:

- `stable_baselines3/__init__.py`
- `stable_baselines3/dsrl/__init__.py`
- untracked `stable_baselines3/dsrl/rfs_dsrl.py`
- untracked `tests/test_dsrl_rfs.py`

None of those paths is staged in the WIP commit. The outer gitlink records only SB3 commit `0e3d4e62f7cec247d4b43be3f7d191094ac17ba9`; it does not include nested dirty files.

## Sub-agent ledger

| Agent | Narrow task | Status | Conclusion used by the project |
|---|---|---|---|
| `/root/audit_rfs_dsrl_temp` | Phase 0 action-dataflow mapping; later reused for Phase 1 bounds/API validation | completed, read-only | Phase 1 specialist `PASS`; verified explicit execution bounds, real scale/unscale use, frozen decoder, strict shapes, zero residual warmup, and unchanged `dsrl.py`. Specialist result is not a Reviewer verdict. |
| `/root/design_joint_dsrl` | Phase 0 checkpoint-structure inspection; later reused for Phase 2 checkpoint/QM inspection | completed, read-only | Phase 2 specialist `PASS`; verified official legacy loader path, block-linear QM parity, exact migration, fresh optimizers, and sampler-injected save/load. Specialist result is not a Reviewer verdict. |
| `/root/design_rfs_dsrl_tests` | Historical pre-implementation loss/test-design consultation | closed/not present in the current live-agent registry | No Phase 0–2 gate relies on a verdict from this agent, and no main-code write is attributed to it. |
| `/root/phase0_reviewer` | Fresh independent review of Phase 0 audit commit | completed, read-only | `VERDICT: PASS`; no blocking/high/medium finding. Recorded in `phase0_reviewer.md`. |
| `/root/phase1_reviewer` | Fresh independent review of exact SB3 Phase 1 commit | completed, read-only | `VERDICT: PASS`; no critical/high/medium/low finding. Recorded in `phase1_reviewer.md`. |
| `/root/phase2_reviewer` | Fresh independent review of exact SB3 Phase 2 commit | **interrupted on pause request** | **No verdict and no accepted test report. Phase 2 remains unreviewed.** |

No implementation sub-agent modified main code. The primary Agent is the only writer.

## Tests executed before the pause

No additional tests were started after the pause request.

Phase 0 audit/review checks:

- `git diff-tree --check 8d21b9cf55459f022c0e769bca1343edd9d30e5a 30ac9ebecb6f3217450c7595c2bdafd0b9d10861`: exit 0, no output.
- All three recorded artifact SHA-256 values matched.
- Runtime Hopper primitive/chunk bounds, normalization arrays, DDIM clipping path, official checkpoint layout, and old-vs-block-linear GEMM diagnostics matched the audit.
- Independent outcome: `VERDICT: PASS`.

Phase 1 focused and regression tests, from the SB3 repository:

```bash
env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 MPLCONFIGDIR=/tmp/dsrl-mpl \
  /home/mrf/miniconda3/bin/python -m pytest -p no:cacheprovider \
  tests/test_dsrl_rfs_hier_phase1.py -q
```

Result: `11 passed, 1 warning in 1.73s`.

```bash
env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 MPLCONFIGDIR=/tmp/dsrl-mpl \
  /home/mrf/miniconda3/bin/python -m pytest -p no:cacheprovider \
  tests/test_dsrl_rfs.py::test_original_dsrl_training_path_still_runs -q
```

Result: `1 passed, 1 warning in 1.71s`.

Combined Phase 1 plus original-DSRL regression result: `12 passed, 1 warning in 2.56s`.

Official `/home/mrf/miniconda3/envs/dsrl/bin/python` direct assertions: `DSRL_ENV_ASSERTIONS: PASS`; maximum warmup scale/unscale round-trip error `4.470348358154297e-08`; decoder eval was true, all decoder parameters had `requires_grad=False`, and decoder call count was 1.

Phase 2 focused tests:

```bash
env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 MPLCONFIGDIR=/tmp/dsrl-mpl \
  /home/mrf/miniconda3/bin/python -m pytest -p no:cacheprovider \
  tests/test_dsrl_rfs_hier_phase2.py -q
```

Result: `11 passed, 1 warning in 1.96s`.

Phase 1 + Phase 2 + original-DSRL regression:

```bash
env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 MPLCONFIGDIR=/tmp/dsrl-mpl \
  /home/mrf/miniconda3/bin/python -m pytest -p no:cacheprovider \
  tests/test_dsrl_rfs_hier_phase2.py \
  tests/test_dsrl_rfs_hier_phase1.py \
  tests/test_dsrl_rfs.py::test_original_dsrl_training_path_still_runs -q
```

Result: `23 passed, 1 warning in 2.09s`.

The pytest warning in these runs is the repository's unknown `env` configuration when plugin auto-loading is deliberately disabled; it does not reflect a test failure.

Additional Phase 2 acceptance diagnostics already run:

- Official Python 3.9 synthetic new-format save/load: `DSRL_PY39_SAVE_LOAD: PASS`; explicit sampler injection was required and module state was exact.
- Real Hopper 7.5M checkpoint plus DDIM5 migration: `REAL_7P5M_MIGRATION: PASS`.
  - Legacy temporary buffer size: 1.
  - Restored diffusion layout: `(4,3)`.
  - Actor, QA, and QA-target maximum parameter differences: `0.0`.
  - Alpha: `0.16040660440921783`.
  - QM-vs-QW maximum differences for batches 1, 2, 17, 256, and 1024: `0.0`.
  - Zero-residual action maximum difference: `0.0`.
  - Residual maximum absolute value: `0.0`.
  - Five rebuilt optimizer state sizes: `[0,0,0,0,0]`.
- Post sampler-exclusion real-checkpoint rerun: `REAL_7P5M_POST_SAVE_FIX: PASS`; QM maximum difference `0.0`, legacy buffer 1, alpha unchanged.
- The first real-run attempt only lacked Hydra's `now` resolver; registering the standard resolver allowed the diagnostic to run. This was not an implementation failure.

## Current unverified risks and blockers

1. Phase 2 has no independent Reviewer verdict. Its implementation and specialist `PASS` cannot substitute for the mandatory independent review.
2. `critic_noise` currently aliases the three-input `ModulationCritic`, while inherited `DSRL.train()` expects the old two-input QW. The hierarchy class must not call `learn()`/`train()` or be exported before Phase 3 supplies the new training path.
3. Phase 3 gradient isolation is unimplemented and unverified: QA target `no_grad`, policy-only strictly paired QM samples, and QM-only parameter updates remain future work.
4. Phase 4 gradient routing is unimplemented and unverified: residual-conditional noise gradients, frozen-QM parameter routing, direct QA action gradients, and residual-only updates remain future work.
5. New hierarchy checkpoints deliberately exclude the diffusion sampler. Every load must reconstruct and inject `diffusion_policy=`; this contract has tests but no independent Phase 2 Reviewer acceptance yet.
6. Real Hopper smoke/training has not occurred. Only checkpoint migration and action/QM parity diagnostics have used the real 7.5M checkpoint and DDIM5 sampler.
7. Original Phase 0 non-blocking risks remain: actor noise is tanh-bounded while legacy prefill/QW distillation used an unbounded Gaussian prior; normalized observations are not clipped and may exceed their declared box.
8. Existing unrelated dirty export, flat-RFS, config, and entry files could be accidentally included later; future commits must continue path/hunk-specific staging.

## Resume gate

The next permitted action is only a fresh/restarted independent Phase 2 review, and only after the exact user string `APPROVE_PHASE_2`. Phase 3 requires Phase 2 Reviewer `PASS` plus the user's later exact string `APPROVE_PHASE_3`.
