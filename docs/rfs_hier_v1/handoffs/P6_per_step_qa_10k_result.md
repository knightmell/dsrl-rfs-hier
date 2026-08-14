# Per-step frozen-DSRL residual + QA: 10k wiring result

## Scope

This run isolates the temporal-structure question while keeping a deterministic
off-policy residual update:

- frozen 5M DSRL-NA noise actor;
- frozen DDIM5;
- a new observation is read before every primitive action;
- the H=4 base chunk is retained until its normal boundary;
- residual input is normalized observation, current base action, and phase;
- primitive QA uses `gamma_primitive = 0.99 ** (1/4)`;
- QA receives 20 updates and the residual actor 5 updates per two equivalent
  vectorized chunk decisions;
- no residual entropy, BC, L2 penalty, joint modulation, or noise update.

This is experiment B in the A/B/C isolation:

- A: chunk residual + QA;
- B: per-step residual + QA;
- C: per-step residual + on-policy trust-region optimization (not implemented
  or run here).

## Artifacts

- Run:
  `logs/per_step_residual/20260729_per_step_qa_init5m_seed1_10k_wiring`
- Explicit 5M checkpoint SHA-256:
  `f6deae068822cd9bc29405e493600c23aed0d0273b9068f8e16b015c7406dc8b`
- DDIM SHA-256:
  `9a5839d3d172d1e24b5bed0831d49bb3116a7e62fafa223c57c322f4bc7e9121`
- Normalization SHA-256:
  `d05b2943bc39772f7f770dfc0a4df2a9f205cb1e05ae74b585cd0dd382a0742e`
- Matched zero-residual prefill:
  80,040 primitive transitions, semantic hash
  `99841813d12f230b792a1670b7dfa57333bcc685820c20b048c0297cf05ba2cd`.

## Wiring invariants

- Zero residual action parity max error: `0.0`.
- Prefill cached-base/action max error: `5.960464477539063e-08`.
- Zero residual remained exact after 10,000 QA-only pretraining updates.
- Frozen planner module hash was unchanged across the run.
- Safe interruption at 5k and resume to 10k completed.
- Final counters:
  - QA optimizer: 20,000, of which 10,000 were pretraining;
  - residual optimizer: 2,500;
  - train calls: 500;
  - primitive transitions: 40,000;
  - equivalent chunk transitions: 10,000.
- Model checkpoints fired every 2k, replay/resume bundles every 5k.

## Fixed-seed results

The strict comparison uses the same 100 environment seeds (10000--10099) and
policy seeds (20000--20099).

| Policy | Raw mean | Raw std | Raw q10 | Non-fall mean | Early fall |
|---|---:|---:|---:|---:|---:|
| Frozen 5M base, residual scale forced to zero | 3093.80 | 202.05 | 3098.87 | 3150.34 | 12% |
| Per-step QA residual after 10k | 2331.93 | 413.56 | 1889.38 | 3275.56 | 96% |

The residual policy loses `761.87` raw return and adds 84 percentage points of
early falls. Only four residual episodes finish the full horizon, so its
apparently higher non-fall mean is selection bias and not an improvement.

Online 10-episode health evaluations:

| Equivalent chunks | Raw mean | Early fall |
|---:|---:|---:|
| 0 | 3098.94 | 10% |
| 2k | 558.76 | 100% |
| 4k | 2447.19 | 70% |
| 6k | 2226.09 | 80% |
| 8k | 1816.38 | 100% |
| 10k | 2304.71 | 100% |

At final evaluation the residual delta L2 is approximately `0.137--0.142`
across slots, and clipping is approximately `8.1%--9.7%`.

## Critic diagnosis

The corrected phase-balanced state/RNG-matched counterfactual diagnostic uses
15 pairs with phase counts 6/3/3/3 and a 50-step horizon:

- QA/Monte-Carlo rank correlation: `-0.3143`;
- pairwise sign accuracy: `66.7%`;
- mean predicted QA advantage: `+0.1860`;
- mean realized 50-step advantage: `+0.0588`;
- false-positive rate among predicted-positive actions: `33.3%`.

There is no evidence of an action-space, frozen-policy, terminal, phase, or
resume wiring violation. The dominant failure is consistent with unsupported
off-policy action gradients: zero-residual prefill constrains QA values at the
base action but does not identify the local slope around it. The deterministic
actor exploits this slope, rapidly saturates, and causes falls. A small Bellman
MSE or plausible absolute Q magnitude is therefore insufficient.

## Decision

Do not run B for 100k. It meets the screening plan's obvious-failure rejection
condition already at wiring scale.

The next discriminating experiment is C: retain the identical frozen DSRL,
DDIM, per-step wrapper, composition scale, seeds, and evaluation protocol, but
replace deterministic replay-Q action gradients with an on-policy
trust-region/clipped objective. It must first pass a separate 10k wiring run.
No PPO implementation or C result is included in this handoff.

## Tests

Unified regression:

```text
131 passed, 23 warnings in 11.20s
```

The diagnostic phase-alias regression is included in
`tests/test_per_step_residual.py`; its targeted suite reports:

```text
9 passed, 34 warnings in 2.21s
```
