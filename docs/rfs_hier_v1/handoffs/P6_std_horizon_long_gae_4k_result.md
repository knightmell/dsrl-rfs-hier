# Residual std, multi-horizon credit, and long-GAE 4k screen

## Scope

This work did not modify `dsrl_na`, the joint hierarchy loss, replay, or any
existing PPO profile.  It first ran no-training diagnostics on the explicit
2.5M DSRL-NA base, then added a separate
`resip_hopper_long_gae_v1` screening profile and stopped at 4k equivalent
action-chunk transitions for three training seeds.  No 10k, 100k, or 5M run
was launched.

## Diagnostic RNG correction

The first std/horizon artifact is invalid for scale selection:

`logs/per_step_residual_ppo/20260803_std_horizon_validation_2p5m_seed1`

The old diagnostic called `seed_all(base_policy_seed)` but did not call
`planner.seed(base_policy_seed)`.  Since the frozen planner now owns a private
Torch RNG, the five nominally identical zero-residual references differed.
The diagnostic now finds the planner through wrappers and explicitly seeds its
private stream at every episode.  The effective artifact is:

`logs/per_step_residual_ppo/20260803_std_horizon_validation_2p5m_seed1_rngfixed`

In the corrected artifact all five std conditions share the same 25 zero rows;
environment seed, base-policy seed, return, length, and fall flag are identical
row by row.  The frozen planner module hash is also identical before/after.

## Corrected validation std sweep

Protocol: 25 previously unused environment/base-policy seed pairs, five
independent residual RNG families, stochastic 2.5M DSRL base, zero-mean
residual, physical residual scale 0.1.

| Unit std | Mean paired delta | Nested 95% CI | Positive families | Early-fall delta |
|---:|---:|---:|---:|---:|
| 0.02 | +28.99 | [-90.01, 165.18] | 3/5 | -2.4pp |
| 0.05 | +32.88 | [-84.42, 172.60] | 4/5 | +0.8pp |
| 0.10 | +2.33 | [-119.50, 150.26] | 3/5 | +8.8pp |
| 0.20 | -13.41 | [-131.96, 117.90] | 0/5 | +8.0pp |
| 0.367879 | -141.78 | [-280.98, 8.49] | 0/5 | +35.2pp |

No std passes the predeclared random-baseline robustness gate.  The official
ResiP std is clearly unsafe in this Hopper action coordinate, while 0.02/0.05
are only safe candidates, not confirmed improvements.  Their mean gain is
again dominated by rescued/new-fall flips; both-complete trajectories change
by only a few raw return points on average.

## Multi-horizon ranking

Protocol: 128 matched simulator states, exactly 32 per phase; zero plus eight
random `std=0.05` first-residual candidates; restored simulator and private
planner RNG; common zero-residual continuation; predictions use scaled n-step
return plus value bootstrap and are ranked against the same 64-step raw return.

| Horizon | Spearman [95% CI] | Pairwise [95% CI] | Zero/random [95% CI] |
|---:|---:|---:|---:|
| 1 | -0.180 [-0.295,-0.064] | 41.58% [36.52,46.70] | 41.41% [36.13,46.78] |
| 4 | +0.005 [-0.118,0.126] | 51.58% [46.09,57.10] | 52.15% [46.19,57.91] |
| 16 | +0.172 [0.037,0.306] | 58.12% [51.82,64.30] | 58.40% [51.76,64.94] |
| 32 | +0.247 [0.105,0.386] | 62.11% [55.49,68.58] | 62.11% [55.37,68.75] |
| 64 | +0.341 [0.201,0.478] | 66.45% [59.79,73.03] | 65.92% [59.08,72.75] |

At horizon 64, phase 2 and phase 3 have strictly positive Spearman intervals;
phase 1 has a marginally positive lower bound, while phase 0 remains
uncertain.  Thus the earlier one-step ranking failure was partly a temporal
credit-window failure, especially around later chunk phases.  It was not fixed
by a larger observation alone.

## Separate long-GAE profile

`resip_hopper_long_gae_v1` changes only the two diagnostic-mandated safety and
credit settings relative to `resip_hopper_v1`:

- fixed unit residual std: `exp(-1) -> 0.05`;
- primitive GAE lambda: `0.95 -> 0.985`.

With gamma 0.999 this gives an approximate effective GAE window of 62.56
primitive steps.  Reward scale remains 0.01; residual scale remains 0.1;
network, initialization, actor/value AdamW learning rates, 50 PPO epochs,
clip 0.2, target KL 0.1, and schedules remain unchanged.  Existing named
profiles remain unchanged.

## Matched 4k results

Common stochastic-base, zero-residual reference: mean 3077.12, q10 2786.26,
early fall 22%.  Every learned policy uses the same 100 environment and private
base-policy seeds and deterministic residual mean evaluation.

| Training seed | Mean | Paired delta | q10 | Early fall | Rescued/new falls | Residual delta L2 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3070.79 | -6.33 | 2775.73 | 19% | 20 / 17 | 0.00149 |
| 2 | 3016.78 | -60.34 | 2549.68 | 30% | 15 / 23 | 0.00135 |
| 3 | 3046.98 | -30.14 | 2650.75 | 21% | 16 / 15 | 0.00148 |

Cross-training-seed mean delta is -32.27, between-seed standard deviation is
27.07, and the hierarchical bootstrap 95% interval is [-76.95, 10.79].  This
is less harmful than the reward-only profile's -61.34 average, but it is not a
positive result and does not pass the continuation gate.

Actor gradients remain clipped heavily (final raw norms 5.23--6.51, clip
scales 0.187--0.235).  Final per-rollout KL is 0.0141--0.0178 while target KL
is 0.1, so the target never constrains the 50 repeated epochs.  This is the
remaining concrete instability candidate.

## Tests

Targeted diagnostic and residual PPO tests after the planner-RNG fix and new
profile:

```text
28 passed
```

The tests cover list/protocol parsing, per-horizon prediction separation,
private planner seeding through wrappers, reward-profile isolation, save/load,
resume, action composition, and existing ResiP/stable PPO behavior.

## Decision

No-go for 10k/100k/5M.  Long-horizon credit improves the local ranking proxy
and reduces average harm, but the current actor update still fails across
training seeds.  The next single-factor screen, if approved, should retain
`std=0.05`, `lambda=0.985`, and reward scale 0.01 while tightening actor drift
control only.  Prefer lowering per-rollout target KL from 0.1 to 0.01 (or,
separately, reducing actor learning rate); do not change both simultaneously.
