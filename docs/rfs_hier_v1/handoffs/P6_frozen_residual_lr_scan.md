# P6 frozen-noise residual learning-rate scan

## Scope

This is a single-training-seed, matched 10k diagnostic. It does not authorize
or constitute a matched 100k or 5M experiment.

All hierarchy runs use:

- algorithm: `dsrl_na_rfs_hier_frozen_noise`
- official DSRL-NA 5M network warm-start
- frozen noise actor, frozen entropy coefficient and frozen DDIM
- QA/QM/noise/residual update counts: `20/10/0/5`
- residual scale: `0.1`
- no residual penalty or additional loss
- 10k action-chunk transitions / 40k nominal primitive actions
- exact 100-episode stochastic final evaluation

The only training-contract difference among the three scan runs is
`residual_learning_rate`.

## Run artifacts

| Residual LR | Run directory | Status |
|---:|---|---|
| `3e-4` | `logs/p6/20260729_frozen_lr3e4_seed1_10k_scan_v2` | complete |
| `1e-4` | `logs/p6/20260729_frozen_lr1e4_seed1_10k_scan` | complete |
| `3e-5` | `logs/p6/20260729_frozen_lr3e5_seed1_10k_scan` | complete |

The first attempted `3e-4` run,
`logs/p6/20260729_frozen_lr3e4_seed1_10k_scan`, failed before manifest
creation because its run name did not contain the complete algorithm token.
It was not resumed, reused or deleted. The corrected `_v2` run is the only
`3e-4` result used below.

## Matched-protocol evidence

All three completed runs have identical:

- checkpoint SHA-256:
  `f6deae068822cd9bc29405e493600c23aed0d0273b9068f8e16b015c7406dc8b`
- frozen DDIM SHA-256:
  `9a5839d3d172d1e24b5bed0831d49bb3116a7e62fafa223c57c322f4bc7e9121`
- normalization SHA-256:
  `d05b2943bc39772f7f770dfc0a4df2a9f205cb1e05ae74b585cd0dd382a0742e`
- source-state SHA-256:
  `c0dbdcf85b525dd6b1b5a671ea77cc3e059fe18296e141dabae5a9593156f1f6`
- prefill artifact hash:
  `06ee7e775b2157c066c654a3e8fa63b885df85c455f37c2f4fb6220ad0ed64f0`
- initial replay semantic hash:
  `1933ab11f7d76327cee30fbb67d76a761db393665285d5d27a188b900b6cf9be`
- every initial replay-array hash
- training, prefill and evaluation seeds
- `n_envs=10`, train frequency and batch size
- exact evaluation environment and policy seed lists

The earlier frozen-noise baseline and matched control have source-state
SHA `b3fbec...`, while this scan has `c0dbdc...`. Manifest comparison confirms
the only dirty-entry difference is the newly added result document
`P6_frozen_noise_10k_result.md`; the outer commit, SB3 state, DPPO state,
algorithm files, checkpoint, DDIM, normalization and prefill all match.

For every scan run, the initialization parity record reports exact zero error:

- actor state
- action critic and target action critic states
- entropy coefficient
- actor decoder input
- zero-residual action
- QM at zero residual
- initial residual output

Final optimizer counters are identical:

- QA: `10000`
- QM: `5000`
- noise actor: `0`
- residual actor: `2500`
- hierarchy train calls: `500`

All runs completed exactly `10000` chunk transitions, `40000` nominal
primitive steps and `40000` actual primitive environment steps.

## Final 100-episode results

| Policy | Raw mean | Raw std | D4RL score | Early fall | Mean primitive length |
|---|---:|---:|---:|---:|---:|
| 5M + 10k DSRL control | 3095.80 | 218.26 | 95.74 | 12% | 976.53 |
| Frozen residual, LR `3e-4` | **3052.36** | **17.94** | **94.41** | **0%** | 1000.00 |
| Frozen residual, LR `1e-4` | 3044.78 | 19.21 | 94.18 | 0% | 1000.00 |
| Frozen residual, LR `3e-5` | 3037.46 | 70.85 | 93.95 | 1% | 997.41 |

The corrected `3e-4` run reproduces the earlier frozen-noise result exactly:
raw mean `3052.357307607326`, raw standard deviation
`17.939816315709294`, and zero early falls.

### Paired fixed-seed differences

Bootstrap intervals use 100,000 paired resamples with RNG seed `20260729`.

| Comparison | Mean | Median | Positive fraction | Paired 95% CI |
|---|---:|---:|---:|---:|
| LR `3e-4` minus control | -43.44 | -105.25 | 10% | [-83.48, +2.16] |
| LR `1e-4` minus control | -51.02 | -110.85 | 10% | [-91.04, -5.08] |
| LR `3e-5` minus control | -58.34 | -111.17 | 10% | [-101.81, -10.35] |
| LR `1e-4` minus LR `3e-4` | -7.58 | -8.55 | 38% | [-12.47, -2.57] |
| LR `3e-5` minus LR `3e-4` | -14.90 | -10.29 | 36% | [-32.29, -3.75] |

Control non-fall episodes average `3162.64` raw. The corresponding non-fall
means are `3052.36`, `3044.78`, and `3044.29` for `3e-4`, `1e-4`, and
`3e-5`. Therefore the hierarchy's lower typical return is not explained only
by a different number of falls. It is a lower-return locomotion mode among
full-length episodes.

## Action and critic diagnostics at 10k

Training diagnostics are the last logged value. Final-action diagnostics are
means over the exact 100 final episodes.

| Metric | `3e-4` | `1e-4` | `3e-5` |
|---|---:|---:|---:|
| residual unit mean abs | 0.7384 | 0.6382 | 0.4246 |
| residual tanh saturation | 13.41% | 1.14% | 0% |
| residual/base L2 ratio | 14.59% | 13.25% | 9.17% |
| training clip fraction | 9.28% | 8.30% | 4.98% |
| effective residual L2 | 0.2611 | 0.2312 | 0.1630 |
| QA(exec) - QA(base) | +1.7105 | +1.5424 | +0.8092 |
| QM(noise,residual) - QM(noise,0) | +1.2853 | +1.0991 | +1.0634 |
| policy-sample distillation MSE | 26.30 | 23.67 | 34.35 |
| final-eval residual delta L2 | 0.2737 | 0.2396 | 0.1667 |
| final-eval effective residual L2 | 0.2655 | 0.2337 | 0.1643 |
| final-eval clip fraction | 7.31% | 6.22% | 3.21% |

The learning-rate intervention behaves as intended mechanically: lower
learning rate produces smaller residuals, lower saturation and less clipping.
However, return monotonically decreases rather than recovers. The residual
at `3e-4` is the strongest stabilizer and the best of the tested residual
learning rates.

The action critic predicts a positive residual advantage for all three runs,
but the fixed-seed real return remains below the control's non-fall return.
Moreover, the predicted advantage is only approximately `0.8--1.7`, while
policy-sample QM distillation MSE is `23--34`. QM is diagnostic in this frozen
experiment and does not update either actor, but these values reinforce that
the critic advantages should not be treated as reliable evidence of a real
performance gain.

## Diagnosis

This scan rejects the narrow hypothesis that the frozen residual loses return
because its optimizer learning rate is simply too high.

The evidence instead supports:

1. The residual objective rapidly finds a conservative gait that removes most
   or all early falls.
2. This gait has lower typical forward return even when episodes reach the
   full horizon.
3. Reducing residual learning rate only weakens the correction at the fixed
   10k budget; it does not preserve the original high-return gait.
4. The highest tested rate is not desirable as a final algorithm setting
   merely because it wins this scan: it still saturates 13.4% of residual
   outputs, clips 7--9% of action dimensions and does not beat the matched
   control in raw return.

Likely remaining causes to distinguish are residual update count/trajectory
coverage and objective-induced survival bias. A scale schedule is a different
intervention from lowering optimizer LR because it constrains the executed
perturbation while allowing the residual network to learn promptly.

## Decision

Do not start matched 100k or 5M from these results.

Among the tested learning rates, keep `3e-4` as the diagnostic reference; do
not change the default to `1e-4` or `3e-5`.

The next bounded experiment should change one factor while keeping the noise
actor frozen and the matched protocol intact. Preferred order:

1. residual-scale warmup/ramp to `0.1`, or a fixed lower execution scale;
2. fewer residual optimizer updates per train call;
3. only if reward components can be observed without changing the objective,
   log Hopper forward and healthy/survival reward components to directly
   identify the approximately 100-raw non-fall deficit.

No new loss, entropy term, BC term, critic, 100k run or 5M run is justified by
this scan.
