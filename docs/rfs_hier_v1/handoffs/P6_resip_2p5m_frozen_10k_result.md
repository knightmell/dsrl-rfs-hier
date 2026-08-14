# ResiP-aligned per-step residual: 2.5M and frozen-base 10k screen

## Scope

This screen does not modify the joint `dsrl_na_rfs_hier` loss.  It tests the
independent per-primitive residual PPO fallback on two bases with more nominal
headroom than the 5M DSRL checkpoint:

- explicit 2.5M DSRL-NA network checkpoint;
- frozen DDIM5 driven by an independent standard-Gaussian prior.

Both use `residual_scale=0.1`, H=4, ten training environments, 10,000
action-chunk-equivalent transitions (40,000 primitive transitions), and one
training seed.  These are screening experiments, not statistical confirmation.

## ResiP reference and implementation alignment

Reference repository commit:
`cf80d12d4acae9e4751a49407a96f85f07b342e3`.

Aligned details:

- residual observation contains normalized state and the current primitive
  base action; Hopper additionally requires the H=4 phase one-hot;
- actor and critic MLPs are `[256, 256]` with ReLU;
- residual mean head is exactly zero and bias-free;
- fixed `log_std=-1` and physical residual scale 0.1;
- critic output gain and bias are both 0.25;
- actor/value AdamW learning rates are 3e-4/5e-3;
- actor warm-up is five PPO iterations, followed by cosine decay; critic uses
  cosine decay without warm-up;
- gamma 0.999, GAE lambda 0.95, PPO clip 0.2, target KL 0.1;
- one full-rollout minibatch and 50 PPO epochs;
- immediate rewards are divided by their running standard deviation without
  mean subtraction and clipped to `[-5, 5]`;
- observations are clamped to `[-3, 3]` before the residual network.

Intentional locomotion adaptations:

- phase is included because the frozen policy caches an H=4 open-loop chunk;
- the final execution action is clipped to the audited Hopper bounds;
- PPO rollouts bootstrap across rollout boundaries instead of resetting every
  400 primitive steps, because Hopper episodes last up to 1,000 steps;
- actor and value gradients retain the previously approved separate clipping.

## Engineering fixes

- removed the hard-coded 5M checkpoint assumption;
- checkpoint source, SHA-256, and expected network step are mandatory;
- added a checkpoint-free `diffusion_prior` base source;
- gave every training/evaluation planner an independent Torch RNG stream so
  residual sampling cannot advance the frozen base-policy stream;
- saved/restored the independent actor/value optimizers, scheduler counters,
  reward-normalizer state, and global RNG state;
- fixed RNG restore to deserialize CPU RNG tensors on CPU;
- kept the original `stable_v1`, legacy PPO, DSRL, and P6 paths intact.

## Artifacts

2.5M checkpoint:

- path: `/home/mrf/dsrl/logs/gym-dsrl/gym_hopper_dsrl_2026-07-23_21-21-02_1/2026-07-23_21-21-02_1/checkpoint/ft_policy_2500000_steps.zip`
- SHA-256: `6deedd21f0fd6e86f96d9a6c5b54ccd6a29a993f6303a4e3d1a8a798a9c0d793`

Frozen DDIM5:

- SHA-256: `9a5839d3d172d1e24b5bed0831d49bb3116a7e62fafa223c57c322f4bc7e9121`

Normalization:

- SHA-256: `d05b2943bc39772f7f770dfc0a4df2a9f205cb1e05ae74b585cd0dd382a0742e`

Run directories:

- `logs/resip-aligned-10k/dsrl_2p5m_seed1`
- `logs/resip-aligned-4k-confirm/dsrl_2p5m_seed2`
- `logs/resip-aligned-4k-confirm/dsrl_2p5m_seed3`
- `logs/resip-aligned-10k/frozen_diffusion_seed1`
- exact frozen zero-residual reference:
  `logs/resip-aligned-10k/frozen_diffusion_zero_reference`

## Evaluation protocol correction

All comparable results below use environment seeds 10000--10099 and a
**stochastic DSRL base** with the same private base-policy RNG stream.  Residual
evaluation is deterministic (the residual Gaussian mean is used).

The earlier 2.5M table compared a stochastic-base trained policy against a
`deterministic_base=True` zero-residual reference (3098.70).  That comparison is
not matched and must not be used.  The corrected stochastic-base zero-residual
reference is 3077.12.  The new standalone evaluator records both
`base_deterministic` and `residual_deterministic` explicitly and verifies all
artifact hashes from the run manifest.

## Corrected results

### 2.5M DSRL base: selected 4k milestone, three training seeds

The zero-residual reference has mean 3077.12, q10 2786.26, and 22% early falls.
Each learned policy was evaluated on the same 100 environment and base-policy
seeds.

| Training seed | Learned mean | Paired delta | q10 | Early fall | Win rate | Rescued falls | New falls |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3083.32 | +6.20 | 2988.95 | 11% | 51% | 20 | 9 |
| 2 | 3019.90 | -57.22 | 2469.34 | 24% | 39% | 16 | 18 |
| 3 | 2988.00 | -89.12 | 2410.61 | 38% | 45% | 11 | 27 |

Across training seeds, the mean paired delta is -46.71 raw return
(between-seed standard deviation 48.52).  A hierarchical bootstrap that
resamples both training seeds and paired evaluation seeds gives a 95% interval
of [-105.04, 12.91].  Aggregate early-fall rate is 24.33%, versus 22% for the
common zero-residual reference.  Seed 1's apparent tail-risk improvement does
not reproduce.

The corrected zero reference is stored at:
`logs/resip-aligned-10k/dsrl_2p5m_seed1/evaluations/zero_reference_stochastic_base_100ep.json`.
The three milestone evaluations are stored under each run's `evaluations/`
directory as `milestone_004000_deterministic_residual_100ep.json`.

### Frozen DDIM prior: selected 6k milestone

The 100-episode re-evaluation rejects the ten-episode online peak:

| Policy | Mean | q10 | Early fall | Residual delta L2 |
|---|---:|---:|---:|---:|
| Zero residual | 1432.01 | 1219.56 | 100% | 0 |
| Seed 1, 6k | 1411.86 | 1212.33 | 100% | 0.01159 |

Because the selected frozen milestone is already negative and every episode
falls early, no additional frozen training seeds were launched.

The original 10k entries (frozen +29.62 and 2.5M -46.07) remain useful only as
single-seed diagnostics.  The corrected milestone/multi-seed results supersede
them for the continuation decision.

## Tests

Targeted:

```text
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ... -m pytest -q \
  tests/test_per_step_residual.py \
  tests/test_residual_random_and_advantage_diagnostics.py \
  tests/test_value_aliasing_probe.py
28 passed
```

All repository unit/config/P6/DSRL regression tests:

```text
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ... -m pytest -q tests/test_*.py
118 passed
```

Real smoke coverage includes both base sources and an interrupted 1k -> resumed
2k 2.5M run.  The planner parameter hash remained unchanged, the zero-residual
reference reported exactly zero residual, and the resume bundle contains model,
actor/value optimizers, reward normalizer, RNG, and counters.

The standalone evaluator was additionally checked with `py_compile` and a real
one-episode manifest/hash/load/evaluation probe.

## Remaining risks and decision

- The 10k cosine schedule is compressed into only ten PPO iterations; it is a
  screen, not a faithful proxy for a long ResiP training horizon.
- Resume is optimizer/RNG exact at a safe rollout boundary, but simulator states
  are not serialized; the manifest correctly labels this environment
  discontinuity.
- Residual samples are clipped to the unit Box before the audited 0.1 physical
  scaling.  Official ResiP does not impose this exact unit clip; it is retained
  here because the approved locomotion action contract defines 0.1 as a maximum
  residual scale.
- Official ResiP trains with a vastly larger parallel rollout batch.  Here each
  PPO iteration contains only 4,000 primitive transitions but still performs 50
  full-batch epochs, making policy updates much more sensitive to trajectory and
  advantage noise.
- The official immediate-reward variance normalization was designed for its
  original task distribution.  Hopper's dense, low-variance positive reward
  produces very large value gradients; logs show that the value optimizer is
  repeatedly controlled mainly by gradient clipping.  This is a concrete
  transport mismatch, not evidence that PPO residual learning is categorically
  invalid.
- A pure frozen diffusion prior is too weak for a 0.1-bounded residual to repair
  within this budget; all evaluated episodes terminate early.

**No-go:** neither condition passes the screen.  Do not launch 100k or 5M and do
not select seed 1 post hoc.

## Single-variable Hopper reward repair

The proposed reward repair was implemented as a separate
`resip_hopper_v1` profile so the reference-aligned `resip_v1` remains
unchanged.  It retains the ResiP actor, critic, initialization, fixed
exploration standard deviation, PPO loss, 50 epochs, learning rates, schedules,
and residual scale.  Its only training change is:

```text
immediate-reward variance normalization -> raw reward * 0.01
```

The named profile fixes the scale at 0.01 and rejects conflicting overrides.
Its manifest records `reward_normalization=false` and
`reward_normalization_semantics=fixed_multiplicative_scale`.  The exact
evaluator also recognizes the new profile.

Targeted regression result after this change:

```text
tests/test_per_step_residual.py: 21 passed
```

Three matched runs used the same 2.5M checkpoint, declared 10k schedule, 4k
safe-boundary stop, and 100 paired evaluation seeds:

| Training seed | Learned mean | Paired delta | q10 | Early fall | Rescued | New falls |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3065.86 | -11.26 | 2683.23 | 18% | 19 | 15 |
| 2 | 2961.72 | -115.40 | 2425.95 | 32% | 15 | 25 |
| 3 | 3019.76 | -57.36 | 2595.23 | 31% | 13 | 22 |

The common zero-residual reference remains 3077.12 mean, 2786.26 q10, and 22%
early falls.  Cross-seed mean delta is -61.34, between-seed standard deviation
is 52.18, and the hierarchical bootstrap 95% interval is [-126.72, -1.60].

The repair did solve the value-scale pathology: final value-gradient norms fell
from roughly 4,365--15,167 to 1.07--1.57, value losses fell from roughly
600--1,861 to 0.021--0.026, and the value gradient is no longer reduced almost
entirely by clipping.  It did **not** make the learned residual reliably useful.
This separates two findings:

1. immediate-reward variance normalization was an invalid Hopper transport;
2. after fixing it, the 50-epoch small-rollout actor update remains unstable
   across training seeds.

Artifacts are under:
`logs/resip-hopper-reward-scale-4k/`.  The first seed-3 process was initially
misread as stopped because its inner execution session had not yet been polled;
it subsequently completed normally.  A redundant `seed3_retry1` was launched
before that was discovered.  The two saved policies have 12/12 parameter
tensors bitwise equal (maximum absolute difference 0), 16,000 primitive
timesteps, and 200 optimizer steps.  They are one duplicated run, not two
independent seeds; the table counts seed 3 once.

**Current decision:** still no-go for 100k/5M.  The next permitted ablation must
change only update intensity (prefer lower actor learning rate first, while
holding reward scale and all other settings fixed).  It should again pass a
three-seed 4k screen before any larger budget.  This is a new experimental
choice and was not launched as part of the reward-only repair.
