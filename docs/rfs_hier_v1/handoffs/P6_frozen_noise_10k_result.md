# Frozen-noise residual matched 10k result

## Correct invariant and comparison

For the frozen-noise experiment, the exact implementation invariant is:

`base(frozen-noise residual after 10k) == base(initial 5M DSRL)`.

The performance comparator is:

`5M checkpoint + 10k frozen-noise residual training`

versus:

`5M checkpoint + 10k continued DSRL control training`.

The base action is not expected to equal the continued-training control action,
because the control updates its noise actor.

## Implementation

New P6 experiment label:

`dsrl_na_rfs_hier_frozen_noise`

Update schedule:

- QA: 20 per train call
- QM: 10 per train call, diagnosis only
- noise actor: 0
- entropy coefficient: 0
- residual actor: 5

The frozen experiment keeps the original QA Bellman target semantics, including
the fixed checkpoint alpha and fixed noise-policy entropy term. The noise actor,
DDIM and alpha are not updated.

Regression result after implementation:

`122 passed, 17 warnings in 10.42s`

Exact 10k freeze evidence:

- actor state maximum absolute error relative to official 5M: `0`
- log-entropy-coefficient maximum absolute error: `0`
- noise actor optimizer steps: `0`
- QA/QM/noise/residual optimizer steps:
  `10000/5000/0/2500`

## Matched protocol

Control:

`/home/mrf/dsrl/logs/p6/20260729_control_init5m_seed1_10k_frozen_ablation`

Frozen-noise residual:

`/home/mrf/dsrl/logs/p6/20260729_frozen_noise_init5m_seed1_10k_ablation`

The following match exactly:

- official 5M checkpoint SHA
- DDIM SHA
- normalization SHA
- source-state SHA:
  `b3fbec7298f27ab6785f54621ccf95879def34f77edc2d6e02c0005ce9aee63a`
- prefill semantic hash:
  `06ee7e775b2157c066c654a3e8fa63b885df85c455f37c2f4fb6220ad0ed64f0`
- every initial replay-array hash
- train/prefill/evaluation seeds
- `n_envs=10`
- 10k action-chunk budget
- exact 100 final evaluation episodes

## Final 100-episode results

| Policy | Raw mean | Raw std | Early fall | Mean primitive length |
|---|---:|---:|---:|---:|
| 5M base / frozen zero-residual | 3093.80 | 202.05 | 12% | 979.52 |
| 5M + 10k DSRL control | 3095.80 | 218.26 | 12% | 976.53 |
| 5M + 10k frozen-noise residual | 3052.36 | 17.94 | 0% | 1000 |

### Frozen residual versus its unchanged 5M base

- paired mean difference: `-41.44 raw`
- paired median difference: `-94.61 raw`
- frozen better on `10%` of seeds
- paired bootstrap 95% CI: `[-78.18, +2.21]`
- early-fall change: `12% -> 0%`
- raw-return standard deviation: `202.05 -> 17.94`

### Frozen residual versus continued DSRL control

- paired mean difference: `-43.44 raw`
- paired median difference: `-105.25 raw`
- frozen better on `10%` of seeds
- paired bootstrap 95% CI: `[-83.33, +2.44]`
- early-fall change: `12% -> 0%`
- raw-return standard deviation: `218.26 -> 17.94`

The confidence interval includes zero, so one training seed does not establish
a statistically reliable raw-return loss or gain. However, the direction is
consistent on 90% of evaluation seeds: the frozen residual produces slightly
slower but much more stable locomotion.

Control non-fall episodes average `3162.64 raw`; frozen residual episodes
average `3052.36 raw`. The residual therefore appears to trade forward reward
for balance rather than improve the high-performing trajectories.

## Frozen residual diagnostics at 10k

- residual unit mean absolute: `0.738`
- residual tanh saturation: `13.4%`
- residual/base L2 ratio: `14.6%`
- clip fraction: `9.28%`
- effective residual L2: `0.261`
- QA(exec) - QA(base): `+1.71`
- QM(noise,residual) - QM(noise,0): `+1.29`
- policy-sample distillation MSE: `26.38`
- alpha: fixed at `0.1610965`

QA predicts a small positive action-value advantage for residual execution,
while real paired evaluation shows a lower typical raw return but elimination
of falls. This can arise because replay Bellman targets value survival and
long-horizon state distribution differently from the fixed finite evaluation
sample, and because a few control falls strongly affect the mean.

## Conclusion

The experiment answers the three requested questions:

1. Adding a residual to the frozen 5M DSRL does not currently demonstrate a
   raw-return improvement.
2. Continuing to update the noise actor is not required to eliminate early
   falls; the residual alone is sufficient for stability.
3. The present residual learns a conservative stabilizer, not a performance
   enhancer. Joint noise/residual training is responsible for a more complex
   reconstruction, but it also does not yet show a clear gain over the 5M
   starting policy.

Under the proposed decision rule, this result is closest to “mean performance
roughly holds, stability improves”. It is not evidence for adopting the frozen
variant as the main return-maximizing method yet.

## Recommended next step

Do not open noise-actor updates or start 100k/5M yet. First run short matched
residual-only diagnostics that preserve the frozen base:

1. lower residual learning rate;
2. fewer residual updates per train call;
3. residual-scale warmup/ramp;
4. log forward-progress and control/survival reward components if exposed by
   the environment;
5. select by paired raw return and early-fall rate, not QA advantage alone.

The immediate target is to retain the observed `0%` early-fall rate without
losing roughly 100 raw return on the control's non-fall trajectories.
