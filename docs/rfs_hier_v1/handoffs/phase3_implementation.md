# Phase 3 QA/QM Loss Implementation Handoff

This is an implementation handoff for external review. It is not a Reviewer verdict.

## Modified files

- `stable-baselines3/stable_baselines3/dsrl/hierarchical_rfs_dsrl.py`
- `stable-baselines3/tests/test_dsrl_rfs_hier_phase3.py`

No other file is part of the Phase 3 SB3 commit. In particular,
`stable_baselines3/dsrl/dsrl.py` is unchanged and has SHA-256
`41dc8cfb86b8640284eb29f9b5b585e9906e93c30e7dfe684dd7da8adcbf6bc8`.

## Implemented behavior

### Q_action block

- `train(gradient_steps, batch_size)` uses the existing DSRL `gradient_steps`
  count for the QA block. The audited Hopper configuration supplies 20.
- Every QA iteration samples replay data and passes `replay_data.actions`
  directly to the online action critic. Replay therefore remains the source of
  the actually executed, clipped `action_exec`; no modulation is reconstructed
  from replay.
- The complete next-state target path is inside one `torch.no_grad()` block:
  current noise actor sample, existing unscale, frozen diffusion decode,
  deterministic residual, action composition/clamp, target QA, entropy term,
  and Bellman target.
- The target uses the minimum over twin target QA heads and noise entropy only.
- QA target Polyak updates retain the original DSRL local-block cadence:
  `gradient_step % target_update_interval == 0`. Running statistics use the
  same cadence.

### Q_modulation block

- The existing `noise_critic_grad_steps` count controls the subsequent QM
  block. The audited Hopper configuration supplies 10.
- Each QM sample calls the current noise actor exactly once.
- That exact `noise_scaled` is passed through the existing unscale and frozen
  diffusion path, then through the deterministic residual actor; the resulting
  paired `noise_scaled`, `residual_unit`, and `action_exec` are used for both
  the QA target and QM input.
- There is no Gaussian-prior sample, prior/policy mixture, second residual
  sample, or replayed historical modulation.
- Actor, residual actor, diffusion, and QA target generation is inside
  `torch.no_grad()`. The QA values are detached. Only the QM forward is outside
  `no_grad`, so only QM receives gradients in this block.
- QA gradients are cleared before QM begins. Stale actor, residual, entropy,
  and QM gradients are also explicitly cleared at the Phase 3 block boundary.
- The recorded policy-sample distillation metric is the mean per-head MSE.

### Explicit Phase 3 boundary

- Phase 3 does not optimize the entropy coefficient, noise actor, or residual
  actor.
- Phase 3 does not implement Phase 4 gradient routing or the final
  `20/10/5/5` four-block update.
- No algorithm export, Hydra entry, configuration change, replay extension, or
  additional loss was added.

## Tests and exact results

Focused Phase 3 CPU tests:

```bash
env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 MPLCONFIGDIR=/tmp/dsrl-mpl \
  /home/mrf/miniconda3/bin/python -m pytest -p no:cacheprovider \
  tests/test_dsrl_rfs_hier_phase3.py -q
```

Result:

```text
3 passed, 1 warning in 1.79s
```

The three tests cover:

- complete QA target `no_grad`, composed next action, and replayed current
  `action_exec`;
- policy-only strictly paired QM samples, absence of Gaussian-prior sampling,
  and QM-only gradients;
- separate QA/QM update counts, no actor/residual/entropy optimizer step, and
  original Polyak cadence.

Phase 1–3 plus original DSRL regression:

```bash
env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 MPLCONFIGDIR=/tmp/dsrl-mpl \
  /home/mrf/miniconda3/bin/python -m pytest -p no:cacheprovider \
  tests/test_dsrl_rfs_hier_phase1.py \
  tests/test_dsrl_rfs_hier_phase2.py \
  tests/test_dsrl_rfs_hier_phase3.py \
  tests/test_dsrl_rfs.py::test_original_dsrl_training_path_still_runs -q
```

Result:

```text
26 passed, 1 warning in 2.04s
```

The warning is the known unknown `env` pytest option when plugin auto-loading
is disabled; it is not a test failure.

Official project Python 3.9 import check:

```text
DSRL_PY39_PHASE3_IMPORT: PASS
```

`git diff --cached --check` passed before the Phase 3 commit.

## Unresolved risks and external-review focus

- Phase 3 has not been externally reviewed. No PASS is claimed here.
- This intermediate class is critic-only: it must not be treated as the final
  trainable algorithm until Phase 4 adds entropy/noise/residual actor blocks.
- The 20/10 values are currently supplied through the existing
  `gradient_steps` and `noise_critic_grad_steps` interfaces. Final hierarchy
  entry/config naming belongs to Phase 5.
- Tests use a synthetic CPU chunk environment. Real Hopper DDIM5 smoke and
  longer runs remain Phase 6 work.
- External review should verify the entire QA target is under `no_grad`, QM
  uses exactly one current-policy noise sample, the same residual/action pair
  reaches QA and QM, only QM obtains gradients during distillation, and Polyak
  updates occur only inside the QA block.

## Commit/base information

- Outer base at Phase 3 start:
  `39b8acfc4facf4f44485aac02cad129c34c1d17c`
- SB3 Phase 3 base:
  `0e3d4e62f7cec247d4b43be3f7d191094ac17ba9`
- SB3 Phase 3 head:
  `b1f44fb032f6e648327debf166ad27c1762f8a1b`
- DPPO remains unchanged by Phase 3:
  `86ce51834055c02f9013e60dd4c4275606d82df7`
- The outer submodule pointer is deliberately not advanced by this handoff
  commit; Phase 1–4 code commits remain independently reviewable in SB3.
