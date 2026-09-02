# Walker Base Support Diagnosis — G1 report

## Decision

**G1 = FAIL.** The support gate passes at zero of three checkpoints for every
proposal family, and the independent QW direct-selection gate is blocked. G2
and behavior execution remain blocked pending audit; this run launches no
training and makes no production-code change.

The failure is not “there are no better candidates.” Every family has a
strictly positive conservative-oracle lift. It is the conjunction required by
the gate that fails: at `K<=16`, the lift remains smaller than QA twin
disagreement and captures less than 80% of the still-growing K=64 oracle gain.
QW then captures only a small fraction of that available gain.

## Canonical scope and audit

- Walker method seed 1, checkpoints 100k/300k/500k.
- One shared 512-state prefill bank; proposal seeds 1101/2202/3303/4404 are
  repeated measurements of those states, not 2,048 independent states.
- Nested `K={1,2,4,8,16,32,64}` and a shared current-policy K=1 anchor.
- 1,179,648 actual DDIM decodes, the same number of QA-target labels and QW
  forwards; zero backward calls, optimizer steps, and environment steps.
- 172,032 state-level metric rows; all finite. All module hashes and saved
  training counters are unchanged.
- Two independent 2,000-replicate aggregations are byte-identical:
  `aggregate_metrics.csv` SHA-256
  `6351efa3bbaa30fc87fad80eeedc5c3e9d9b6871742bcc8a200d1925b599c16d`.
- All 24,576 state/seed/method/view prefix sequences have nondecreasing oracle
  lift, and all 6,144 primary K=1 anchors are exactly equal across methods.
- Focused tests: 18 passed with third-party pytest plugin autoload disabled.

An initial v1 aggregate incorrectly called the exact prior's unbounded
coordinates “tanh saturation.” The canonical v2 rerun fixes this: tanh
saturation is computed only for coordinates actually produced by tanh, while
the exact prior is reported separately by boundary/out-of-envelope fraction.
The corrected full rerun gives the same G1 decision.

## K=16 gate evidence

`Oracle/K64` is the mean conservative-oracle gain captured by K=16. `Lift/twin`
is median oracle lift divided by median QA-twin disagreement. Both must be at
least 0.80 and 1.00 respectively. `Capture` is the gate's raw median QW selector
capture; it must be at least 0.50 for executable families.

| Checkpoint | Family | Oracle lift | Oracle/K64 | Lift/twin | QW pairwise | Raw median capture |
|---:|---|---:|---:|---:|---:|---:|
| 100k | current | 4.234 | 0.762 | 0.865 | 0.530 | -0.065 |
| 100k | prior-exact | 12.124 | 0.676 | 0.850 | 0.579 | 0.000 |
| 100k | mix | 6.872 | 0.689 | 0.746 | 0.646 | -0.019 |
| 300k | current | 3.883 | 0.761 | 0.722 | 0.526 | -0.060 |
| 300k | prior-exact | 11.553 | 0.625 | 0.809 | 0.577 | 0.000 |
| 300k | mix | 6.479 | 0.655 | 0.676 | 0.648 | 0.024 |
| 500k | current | 4.224 | 0.772 | 0.679 | 0.532 | -0.051 |
| 500k | prior-exact | 11.529 | 0.653 | 0.756 | 0.593 | 0.000 |
| 500k | mix | 6.981 | 0.685 | 0.692 | 0.676 | 0.022 |

All nine K=16 oracle-lift confidence intervals are strictly above zero, and
decoded-action diversity does not collapse. Thus the negative gate decision is
not caused by missing candidate diversity or a null oracle effect.

## What the curves say

1. **Current-policy support is real but shallow.** K=16 gives about 3.9–4.2 Q
   units of conservative oracle lift, but only 76–77% of K=64. Its pairwise QW
   accuracy remains 0.526–0.532, close to chance, from 100k through 500k.

2. **The DSRL prior exposes much larger teacher opportunities but is not
   executable.** At K=16 its oracle lift is about 11.5–12.1, roughly 2.7–3.0x
   current. About 29.8% of its coordinates are outside the actor envelope. At
   K=32, prior-exact reaches 84.6%/81.9%/83.5% of K=64 and lift/twin ratios
   1.013/1.049/0.927; therefore it would satisfy the support criteria at 100k
   and 300k only if the predeclared `K<=16` efficiency requirement were removed.
   This is useful diagnostic evidence, not a post-hoc PASS.

3. **Reachable mix improves coverage without saturation, but not enough at
   practical K.** At K=16 it gives 6.5–7.0 oracle lift and zero out-of-envelope
   mass, yet captures only 65.5–68.9% of K=64 and remains below the twin-SNR
   threshold. At K=64 it reaches lift/twin 1.124 and 1.025 at 100k/300k, but
   0.995 at 500k.

4. **QW ranking is the actionable bottleneck.** At K=16, current-policy top-1
   agreement is only 0.065/0.074/0.081 (random is 0.0625). Mix improves
   pairwise accuracy and Spearman correlation, but top-1 is still only
   0.097/0.090/0.095. QW predicts gains much larger than QA verifies: the mean
   exploitation gap is 3.735/3.150/3.191 for current and
   3.117/2.604/2.751 for mix. The selected-lift CI is positive at K=16 for both
   executable families, so QW is not completely useless; it simply captures
   far less than the required half of the oracle opportunity.

5. **More base training did not repair this mechanism.** Current K=16 QW
   pairwise accuracy is essentially flat from 100k to 500k, while its
   lift/twin ratio falls from 0.865 to 0.679 as QA twin disagreement grows.
   This is consistent with a persistent teacher-distribution/ranking problem,
   not merely too few samples from the already-trained current actor.

## Consequence for the next gate

- Do not execute QW top-1 candidates online.
- Do not start SVGD/G2 on this QW: directed optimization would currently target
  a critic with a large QW–QA exploitation gap.
- The high-K prior/mix oracle curves justify investigating proposal-source and
  QW-teacher-distribution repair, but that requires a newly audited plan (for
  example the predeclared prior-1/source-control training arm), not relabelling
  this failed G1 after seeing the result.

The Python/cloudpickle `lr_schedule` load warning remains non-blocking because
the probe never constructs or steps an optimizer. The deprecated NumPy namespace
warning is also recorded in every worker manifest.
