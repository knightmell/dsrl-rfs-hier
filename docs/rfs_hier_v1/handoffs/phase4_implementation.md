# Phase 4 Noise/Residual Actor Implementation Handoff

This is an implementation handoff for external review. It is not a Reviewer verdict.

## Modified files

- `stable-baselines3/stable_baselines3/dsrl/hierarchical_rfs_dsrl.py`
- `stable-baselines3/tests/test_dsrl_rfs_hier_phase3.py`
- `stable-baselines3/tests/test_dsrl_rfs_hier_phase4.py`

The Phase 3 test change only sets both new actor-step counts to zero in the
critic-only block test, preserving its original Phase 3 scope after the full
`train()` method gained Phase 4 blocks.

No existing DSRL implementation file changed. In particular,
`stable_baselines3/dsrl/dsrl.py` retains SHA-256
`41dc8cfb86b8640284eb29f9b5b585e9906e93c30e7dfe684dd7da8adcbf6bc8`.

## Implemented behavior

### Configuration owned by Phase 4

- `noise_actor_gradient_steps=5`
- `residual_actor_gradient_steps=5`
- `residual_penalty_coef=0.0`

Step counts require non-negative integers. The residual penalty coefficient
must be finite and non-negative. Existing `gradient_steps` and
`noise_critic_grad_steps` continue to supply the QA and QM counts respectively;
the audited Hopper values are 20 and 10.

### Noise actor and entropy block

For each noise actor step:

1. Sample one `noise_scaled` and `log_prob_noise` from the current SAC actor.
2. Reuse the audited unscale and frozen diffusion path to obtain detached
   `action_base`.
3. Evaluate the deterministic residual actor on
   `(observation, noise_scaled, action_base.detach())`.
4. Evaluate `min_i Q_modulation(observation, noise_scaled, residual_unit)`.
5. Optimize
   `mean(alpha * log_prob_noise - min_i Q_modulation_i)`.

QM and residual actor parameters are temporarily marked non-trainable during
this forward/backward, but the forward is not wrapped in `no_grad`. Gradients
therefore reach the noise actor both directly through QM's noise input and
through the explicit
`noise_scaled -> residual_actor -> residual_unit -> QM` path. Original
`requires_grad` states are restored after each update. Diffusion and
`action_base` remain detached.

The entropy coefficient uses the same noise sample's detached log probability,
inherits the existing target-entropy semantics, and is updated once per noise
actor step. No entropy term belongs to the deterministic residual actor.

### Residual actor block

For each residual step:

1. Sample current `noise_scaled` under `no_grad` and detach it.
2. Decode a detached `action_base` through the frozen diffusion policy.
3. Produce `residual_unit` and compose the bounded `action_exec` in execution
   coordinates.
4. Freeze QA parameters without using `no_grad` for its forward.
5. Optimize
   `-mean(min_i Q_action_i(observation, action_exec)) +
   residual_penalty_coef * mean(action_residual_delta ** 2)`.

This preserves `dQ_action / d action_exec` while preventing QA parameter
updates. The optional penalty uses the physically applied
`action_residual_delta`, not raw `residual_unit`. Its V1 default is zero.

### Update order

The complete update is four non-interleaved blocks:

1. QA: `gradient_steps` (Hopper: 20).
2. QM: `noise_critic_grad_steps` (Hopper: 10).
3. Entropy coefficient and noise actor: 5.
4. Deterministic residual actor: 5.

Actor, QM, QA, residual, and entropy gradients are explicitly cleared at block
boundaries. QA Polyak updates remain inside the QA block with the original
DSRL cadence.

### Explicit Phase 4 boundary

- No export or `algorithm=dsrl_na_rfs_hier` entry was added.
- No Hydra configuration was changed.
- No custom ReplayBuffer, prior mixture, stochastic residual, residual entropy,
  residual BC, temporal loss, uncertainty, or extra algorithm was added.
- Phase 5's full diagnostics and training entry are not implemented here.

## Tests and exact results

Focused Phase 4 CPU tests:

```bash
env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 MPLCONFIGDIR=/tmp/dsrl-mpl \
  /home/mrf/miniconda3/bin/python -m pytest -p no:cacheprovider \
  tests/test_dsrl_rfs_hier_phase4.py -q
```

Result:

```text
5 passed, 1 warning in 1.90s
```

The tests verify:

- V1 defaults and invalid count/penalty rejection;
- a non-zero noise gradient through a frozen residual actor when QM is made to
  depend only on `residual_unit`, proving the explicit conditional path;
- detached noise/base and direct QA action gradient for residual learning, with
  actor/QA/diffusion isolation;
- residual penalty units use `action_residual_delta`;
- an actual update executes optimizer steps in exact `20/10/5/5` block order,
  with entropy paired to each noise actor step and Polyak confined to QA.

Phase 1–4 plus original DSRL regression:

```bash
env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 MPLCONFIGDIR=/tmp/dsrl-mpl \
  /home/mrf/miniconda3/bin/python -m pytest -p no:cacheprovider \
  tests/test_dsrl_rfs_hier_phase1.py \
  tests/test_dsrl_rfs_hier_phase2.py \
  tests/test_dsrl_rfs_hier_phase3.py \
  tests/test_dsrl_rfs_hier_phase4.py \
  tests/test_dsrl_rfs.py::test_original_dsrl_training_path_still_runs -q
```

Result:

```text
31 passed, 1 warning in 2.18s
```

The warning is the known unknown `env` pytest option when plugin auto-loading
is disabled; it is not a test failure.

Official project Python 3.9 import check:

```text
DSRL_PY39_PHASE4_IMPORT: PASS
```

`git diff --cached --check` passed before the Phase 4 commit.

## Unresolved risks and external-review focus

- Phase 4 has not been externally reviewed. No PASS is claimed here.
- The hierarchy is deliberately not exported or reachable from the training
  entry until Phase 5.
- Full residual/clip/Q-difference diagnostics are Phase 5 work and are not yet
  available during training.
- Hard clipping can still zero residual gradients outside execution bounds;
  Phase 5 must add the required clip and tanh-saturation diagnostics, not a new
  loss.
- Tests use a synthetic CPU chunk environment. Real Hopper DDIM5 smoke and
  training remain Phase 6.
- Final hierarchy-specific public naming for QA/QM step counts belongs to the
  Phase 5 entry/config integration; current behavior is already tested at
  `20/10/5/5`.
- External review should inspect parameter freezing/restoration, retained input
  gradients, detached diffusion/base paths, entropy ownership, direct residual
  QA gradient, delta-based penalty, and exact non-interleaved block order.

## Commit/base information

- Outer base at Phase 4 start:
  `e3bcd3c91581a5359cc196afa70ef6ee19a61825`
- SB3 Phase 4 base:
  `b1f44fb032f6e648327debf166ad27c1762f8a1b`
- SB3 Phase 4 head:
  `2d998b65f2985e483d4e52629e2d841c60c1b991`
- DPPO remains unchanged by Phase 4:
  `86ce51834055c02f9013e60dd4c4275606d82df7`
- The outer submodule pointer is deliberately not advanced by this handoff
  commit; Phase 1–4 SB3 commits remain independently reviewable.
