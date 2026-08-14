# Random-residual RNG and PPO advantage diagnosis

## Erratum (2026-08-03)

The random-family result and the original gate decision in this document are
superseded.  The diagnostic seeded global Torch RNG but did not seed the frozen
planner's newer private Torch RNG.  Therefore nominally matched base-policy
seeds did not reproduce the same base latent stream across episodes/families.

The within-state counterfactual candidates did restore one common planner RNG
snapshot and remain informative as a local paired probe, but the declared
episode-level base seed was not enforced.  The corrected implementation calls
`planner.seed(base_policy_seed)` explicitly.  Corrected scale and multi-horizon
results are documented in:

`P6_std_horizon_long_gae_4k_result.md`

and stored under:

`logs/per_step_residual_ppo/20260803_std_horizon_validation_2p5m_seed1_rngfixed`

Do not cite the old `random_rng_robustness.json` values or its V3 decision as a
valid matched-RNG result.

## Decision

- `random_baseline_robust`: **false**
- `advantage_ranking_credible`: **false**
- `implement_variance_only_reference_kl_v3`: **false**

The predeclared gate therefore rejects variance-only PPO V3.  No 100k/5M
training and no V3 implementation were started.

## Protocol invariants

- Frozen DSRL checkpoint: explicit 5M checkpoint from the stable-v1 manifest.
- Frozen DDIM and normalization hashes were checked before rollout.
- Frozen planner hash before/after:
  `950c521a8a7aa798693913db737c5be08483d7ef3defea8e5b0648ba100fb2b8`.
- Environment seeds and frozen-base policy seeds were paired across every
  residual RNG family.
- Residual actions were sampled from an independent NumPy generator, so they
  could not consume or shift the Torch RNG stream used by the frozen DSRL
  actor.
- Zero-residual parity was checked against the old evaluator: the 10 common
  initial deterministic episodes were bitwise equal in return and length.

## Multi-RNG random baseline

Protocol: 100 fixed environment/base-policy seed pairs, 10 independent
residual RNG families, zero-mean Gaussian unit residual with `std=0.05` and
the existing execution scale `0.1`.

Zero residual:

- mean raw return: `3093.8000`
- early fall: `12%`

Across random families:

- aggregate paired mean: `+16.8531`
- family-mean standard deviation: `19.8375`
- nested family/environment bootstrap 95% CI: `[-24.1631, 63.6299]`
- positive family fraction: `9/10`
- aggregate early-fall change: `-2.7pp`

One family produced `-32.2000` and increased early fall by `5pp`; therefore
the predeclared robustness gate failed despite the positive aggregate mean.

The gain is a rare-event effect:

| Paired outcome | Count | Mean delta | Contribution to aggregate mean |
|---|---:|---:|---:|
| both complete | 796 | -0.5249 | -0.4178 |
| zero falls, random completes | 111 | +471.5371 | +52.3406 |
| zero completes, random falls | 84 | -417.7774 | -35.0933 |
| both fall | 9 | +2.6282 | +0.0237 |

Thus full-episode random smoothing is nominally neutral/slightly harmful and
changes return mainly by flipping fall outcomes in both directions.

## Previous stochastic-result confound

The old evaluator seeded global Torch RNG once and then interleaved:

1. frozen DSRL latent sampling at chunk boundaries;
2. PPO residual sampling at every primitive step.

Changing or removing residual sampling therefore shifted all later base-noise
draws.  The earlier `+35.75` result was a combined residual-action plus
base-RNG-stream intervention, not a pure residual counterfactual.  The new
diagnostic removes this confound.

## Advantage ranking

For 128 matched simulator snapshots (32 per chunk phase), each state compared
zero residual with 8 independently sampled random residuals.  Every candidate
restored the same simulator and frozen-planner RNG state, changed only the
first primitive residual, and then used a common zero-residual continuation
for 64 primitive steps.

Prediction:

`reward * 0.01 + gamma * V(next_observation) - V(observation)`

Ground truth: raw 64-step discounted counterfactual return.

Overall:

- mean per-state Spearman: `0.0241`, CI `[-0.0990, 0.1479]`
- pairwise preference accuracy: `51.89%`, CI `[46.31%, 57.49%]`
- random-vs-zero preference accuracy: `50.88%`, CI `[45.12%, 56.64%]`
- top-1 agreement: `27.34%`, CI `[20.31%, 35.16%]`
- positive-Spearman phases: `2/4`

By phase:

| Phase | Spearman | Pairwise | Random vs zero |
|---:|---:|---:|---:|
| 0 | +0.2151 | 61.37% | 60.94% |
| 1 | -0.0573 | 47.92% | 49.22% |
| 2 | +0.1562 | 58.25% | 55.47% |
| 3 | -0.2177 | 40.02% | 37.89% |

The critic's next-value difference dominates the TD ranking.  Median
candidate true-return spreads are only `0.0053` to `0.0147` raw reward across
phases, while the critic produces much larger and frequently wrong local
ordering, especially across phase 3 -> phase 0 plan boundaries.  This is not
a reliable PPO policy-improvement signal.

## Tests

Targeted diagnostic/per-step regression:

`20 passed, 48 warnings`

## Recommended next direction

Do not implement PPO V3.  Do not claim fixed random smoothing as a robust
improvement yet.  The data supports a risk-selective stochastic intervention
diagnostic:

1. learn or construct a held-out risk score for near-future fall, without an
   action-gradient objective;
2. test whether random residual is beneficial specifically in high-risk
   states and disabled in nominal states;
3. keep residual RNG isolated from frozen-base RNG;
4. validate thresholds on separate validation seeds and report once on held-
   out test seeds.

If risk-conditioned interventions do not improve paired tail outcomes without
inducing new falls, retain random smoothing only as a negative/diagnostic
ablation rather than a method component.

## Artifacts

- `logs/per_step_residual_ppo/20260802_random_rng_and_advantage_diagnosis/random_rng_robustness.json`
- `logs/per_step_residual_ppo/20260802_random_rng_and_advantage_diagnosis/advantage_ranking.json`
- `logs/per_step_residual_ppo/20260802_random_rng_and_advantage_diagnosis/decision.json`
