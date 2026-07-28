# Hopper DSRL Checkpoint Evaluation

## Scope

The interrupted Hopper DSRL-NA run contains three usable main checkpoints:

| Label | Saved timestep | SHA-256 |
|---|---:|---|
| 2.5M | 2,500,000 | `6deedd21f0fd6e86f96d9a6c5b54ccd6a29a993f6303a4e3d1a8a798a9c0d793` |
| 5M | 5,000,000 | `f6deae068822cd9bc29405e493600c23aed0d0273b9068f8e16b015c7406dc8b` |
| 7.5M | 7,500,000 | `e75686d06f7297b870ee8d286fc36db6ecb6cb62e9c6a1783b9466b3f0691fb6` |

There is no 2M checkpoint and no checkpoint between 7.5M and 10M. Therefore
2.5M replaces the requested approximate 2M point, and 7.5M is the closest
available checkpoint to 10M.

## Protocol

- Environment: `hopper-medium-v2`.
- Frozen sampler: audited DDIM5 diffusion policy.
- Episodes: 100 per checkpoint and evaluation mode.
- Environment initial-state seeds: exactly `10000..10099` for every condition.
- Policy RNG seed: 1.
- Modes: deterministic noise actor and stochastic noise actor.
- Execution uses four-action chunks and at most 1,000 primitive environment
  steps per episode.
- The evaluator explicitly seeds the underlying legacy Gym Hopper environment.
  This is necessary because the current `ObservationWrapperGym.reset` ignores
  a top-level `seed` argument.
- Reported D4RL scores are normalized scores multiplied by 100.

Command:

```bash
/home/mrf/miniconda3/envs/dsrl/bin/python \
  eval_hopper_dsrl_checkpoints.py \
  --episodes 100 \
  --batch-size 25 \
  --seed-start 10000 \
  --policy-seed 1 \
  --device cuda:0 \
  --modes deterministic stochastic \
  --output \
  logs/rfs_hier_v1/dsrl_checkpoint_eval_100ep_seed10000/results.json
```

## Results

Values are mean ± population standard deviation across 100 episodes. The
parenthesized value is the standard error of the mean.

| Checkpoint | Mode | Return | D4RL normalized score | Mean primitive length |
|---|---|---:|---:|---:|
| 2.5M | deterministic | 3098.70 ± 194.82 (19.58) | 95.834 ± 5.986 (0.602) | 979.44 |
| 2.5M | stochastic | 3080.04 ± 244.10 (24.53) | 95.260 ± 7.500 (0.754) | 970.32 |
| 5M | deterministic | 3139.43 ± 30.70 (3.09) | 97.085 ± 0.943 (0.095) | 999.08 |
| 5M | stochastic | 3114.88 ± 148.00 (14.87) | 96.331 ± 4.548 (0.457) | 986.84 |
| 7.5M | deterministic | 3136.13 ± 74.86 (7.52) | 96.984 ± 2.300 (0.231) | 994.72 |
| 7.5M | stochastic | 3087.56 ± 208.92 (21.00) | 95.491 ± 6.419 (0.645) | 976.56 |

The 5M checkpoint has the highest deterministic and stochastic mean and the
lowest variance. The deterministic mean difference between 5M and 7.5M is only
3.30 raw return (0.101 normalized points), so this evaluation does not support
claiming that one is meaningfully better without paired statistical analysis.
It does show that later training did not produce a clear monotonic gain.

## Required hierarchy/control matrix

The existing hierarchy run initialized only from 7.5M. It is a
single-initialization pilot and cannot establish checkpoint-robust performance.
The final experiment must run the following matched pairs:

| Initialization | DSRL-NA control | Hierarchical DSRL-NA |
|---|---|---|
| 2.5M | +5M new interactions | +5M new interactions |
| 5M | +5M new interactions | +5M new interactions |
| 7.5M | +5M new interactions | +5M new interactions |

Within every row, both methods must use:

- the exact same checkpoint;
- empty replay and fresh optimizers;
- the same new-interaction count;
- the same training and evaluation seeds;
- the same evaluation initial states;
- local TensorBoard logging with W&B disabled.

This matrix measures sensitivity to network initialization and the improvement
from residual modulation relative to matched additional DSRL-NA training.
Because 2.5M/5M/7.5M are correlated snapshots from one training trajectory,
they are not independent random seeds. Final uncertainty claims still require
independent training seeds; the checkpoint sweep is an initialization
sensitivity study, not a substitute for seed replication.

Using the same `+5M` adaptation budget also means the total historical
interaction budgets differ across rows (7.5M, 10M, and 12.5M respectively).
Results must therefore be reported by initialization checkpoint and additional
interaction budget, not collapsed into one pooled score.
