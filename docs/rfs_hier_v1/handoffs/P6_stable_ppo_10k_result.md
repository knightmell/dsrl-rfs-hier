# Stable per-step residual PPO: D0 diagnosis and 10k result

## Scope

This iteration did not modify the legacy `dsrl_na_residual_per_step_ppo`
default. It added the explicit algorithm identity:

`dsrl_na_residual_per_step_ppo_stable_v1`

The stable variant uses a fixed training reward scale of `0.01` and clips the
residual-actor and value parameter groups independently. Evaluation continues
to use raw Hopper rewards.

## Cross-chunk bootstrap audit

Tests verify:

- phase 3 returns a phase-0 observation containing the newly decoded chunk;
- a PPO rollout ending exactly on phase 3 bootstraps from that new chunk;
- time-limit `terminal_observation` also contains a newly decoded phase-0
  chunk.

The cross-chunk bootstrap was already correct. It was not the observed PPO
failure.

## Offline value aliasing probe

Artifact:

`logs/per_step_residual_ppo/20260730_ppo_init5m_seed1_10k_wiring/diagnostics/value_aliasing_h64_seed1/probe_results.json`

Protocol:

- 100 exact complete episodes;
- 93,694 valid transitions;
- fixed 64-primitive-step discounted target;
- identical episode split, `[128,128]` capacity, batches and update count.

Held-out test:

| Probe | Explained variance | MSE |
|---|---:|---:|
| state + current base + phase | 0.93258 | 30.5342 |
| state + remaining plan + mask + phase | 0.92905 | 32.1268 |

Remaining-plan input reduced EV by `0.00352`, increased MSE by `5.22%`, and
failed to improve any of the four phase groups. The predeclared plan-context
gate failed. Plan-conditioned V2 was therefore not implemented.

## Confirmed gradient-coupling failure

Artifact:

`logs/per_step_residual_ppo/20260730_ppo_init5m_seed1_10k_wiring/diagnostics/ppo_rollout_value_and_gradient_diagnostics.json`

On fresh rollouts from existing checkpoints:

| Checkpoint | Actor grad | Value grad after vf coef | Global clip scale | Effective actor grad |
|---|---:|---:|---:|---:|
| legacy 10k | 0.602 | 1,933 | 2.586e-4 | 1.558e-4 |
| legacy 50k | 0.930 | 20,311 | 2.462e-5 | 2.291e-5 |

Raw-scale value gradients were 3,209x and 21,829x larger than the actor
gradient. SB3's single global `max_grad_norm=0.5` therefore suppressed the
residual actor by roughly four to five orders of magnitude.

## Implemented repair

- explicit `stable_v1` variant;
- training-only reward scale `0.01`;
- raw reward retained in `info` and raw evaluation unchanged;
- independent actor/value parameter-group clipping;
- fail-fast on unknown/shared trainable policy parameters;
- persistent actor/value gradient norms and clip scales;
- explicit final logger flush so the last PPO update is stored;
- manifest records algorithm identity, reward scale, base mode and gradient
  limits;
- legacy variant retains reward scale `1.0` and the original `CountingPPO`.

The real one-rollout smoke changed:

- value loss: approximately `28,000` legacy to `2.29` stable;
- value/actor gradient ratio: `3,209` legacy to `0.86` stable;
- actor clip scale: `2.586e-4` legacy to `0.209` stable.

The frozen DSRL planner hash remained exact.

## Tests

Targeted:

`27 passed`

Full hierarchy/P6/legacy DSRL regression:

`150 passed, 23 warnings`

## Uninterrupted 10k result

Run:

`logs/per_step_residual_ppo/20260731_ppo_stable_init5m_seed1_10k`

Invariant results:

- status: complete;
- optimizer steps: 500;
- planner hash unchanged;
- no interruption/resume discontinuity;
- final value loss: `0.6660`;
- final actor/value gradient ratio: `1.242`;
- final actor clip scale: `0.227`;
- final residual std: `0.05028`.

Evaluation:

| Mode | Raw return | Early fall | Residual delta L2 |
|---|---:|---:|---:|
| trained residual mean | 3102.80 | 8% | 0.000950 |
| trained sampled residual | 3091.40 | 11% | 0.008092 |
| untrained sampled residual reference | 3124.24 | 8% | 0.007993 |

Paired trained-sampled minus untrained-sampled:

- mean: `-32.8458`;
- median: `-4.0292`;
- 95% bootstrap CI: `[-78.0019, 9.5876]`;
- positive fraction: `45%`;
- early fall: `8% -> 11%`;
- both-success mean difference: `-2.2477`;
- reference-fall to trained-success: 8;
- reference-success to trained-fall: 11.

The final policy remained close to the initial random reference:

- mean KL: `0.02111`;
- median KL: `0.01891`;
- KL q90: `0.03994`;
- KL q99: `0.05889`;
- mean residual-unit mean L2: `0.00957`.

Even this small learned mean drift was harmful on the matched seed set.

## Decision

No 100k or 5M run is justified.

The plan-context hypothesis was rejected. The gradient-coupling failure was
real and has been repaired, but stronger PPO updates moved the residual policy
away from a better untrained stochastic reference.

The next method change, if approved, should be a separate algorithm:

- residual mean fixed exactly at zero;
- learn only bounded state-dependent standard deviation;
- initialize exactly at `N(0, 0.05)`;
- constrain average KL to that stochastic reference;
- keep base DSRL and DDIM frozen;
- keep the current-state/current-base/phase observation because the plan probe
  did not justify a larger input.

This tests stochastic robustness modulation directly and prevents the harmful
directional mean correction observed here.
