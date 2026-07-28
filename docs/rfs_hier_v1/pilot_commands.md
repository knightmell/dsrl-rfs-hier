# Hierarchical DSRL-NA Residual Modulation V1 Pilot Commands

Run commands from `/home/mrf/dsrl`.

## Configuration-only validation

This composes and prints the resolved Hopper job without constructing an
environment or starting training:

```bash
/home/mrf/miniconda3/envs/dsrl/bin/python train_dsrl.py \
  --config-path cfg/gym \
  --config-name dsrl_hopper \
  --cfg job \
  algorithm=dsrl_na_rfs_hier \
  total_timesteps=5000000 \
  env.n_envs=10 \
  seed=1
```

The resolved V1 update counts must be `20/10/5/5`, and the checkpoint must be:

```text
./logs/gym-dsrl/gym_hopper_dsrl_2026-07-23_21-21-02_1/2026-07-23_21-21-02_1/checkpoint/ft_policy_7500000_steps.zip
```

Its audited SHA-256 is:

```text
e75686d06f7297b870ee8d286fc36db6ecb6cb62e9c6a1783b9466b3f0691fb6
```

## Phase 6 pre-pilot gates

The real Hopper/DDIM5 and migration gates are standalone because they require
CUDA, MuJoCo, the frozen diffusion checkpoint, and the local 7.5M DSRL-NA
checkpoint:

```bash
/home/mrf/miniconda3/envs/dsrl/bin/python \
  tests/integration/rfs_hier_phase6.py \
  --gate smoke --device cuda:0 --seed 1

/home/mrf/miniconda3/envs/dsrl/bin/python \
  tests/integration/rfs_hier_phase6.py \
  --gate migration --device cuda:0 --seed 1
```

The migration gate requires exact actor, QA, QA-target, alpha, action, and
QW-to-QM parity within the declared `1e-6` tolerance, plus empty optimizer
states.

## 5M hierarchy pilot

```bash
/home/mrf/miniconda3/envs/dsrl/bin/python train_dsrl.py \
  --config-path cfg/gym \
  --config-name dsrl_hopper \
  algorithm=dsrl_na_rfs_hier \
  total_timesteps=5000000 \
  env.n_envs=10 \
  seed=1 \
  use_wandb=false \
  name=phase6_hier_5m_seed1 \
  logdir=./logs/rfs_hier_v1/phase6_hier_5m_seed1 \
  hydra.run.dir=./logs/rfs_hier_v1/phase6_hier_5m_seed1
```

The entry constructs a new hierarchy model with an empty replay buffer and new
optimizers, then performs a network warm-start through the formal legacy loader.
This is not an exact training resume. The configured `init_rollout_steps` are
new interactions collected after construction.

## Matched 5M DSRL-NA control

The control uses a separate Phase 6 runner so the existing `algorithm=dsrl_na`
branch in `train_dsrl.py` remains unchanged:

```bash
/home/mrf/miniconda3/envs/dsrl/bin/python \
  train_dsrl_warmstart_control.py \
  algorithm=dsrl_na \
  total_timesteps=5000000 \
  env.n_envs=10 \
  seed=1 \
  use_wandb=false \
  name=phase6_dsrl_na_control_5m_seed1 \
  logdir=./logs/rfs_hier_v1/phase6_dsrl_na_control_5m_seed1 \
  hydra.run.dir=./logs/rfs_hier_v1/phase6_dsrl_na_control_5m_seed1
```

Both jobs use the same frozen DDIM5 decoder, 7.5M network checkpoint, seed,
new-interaction count, evaluation seeds, initial evaluation protocol, and
configured base-policy prefill. Both start with empty replay and freshly
created optimizers. Both disable W&B and retain local TensorBoard logs. The
comparison is therefore a matched network warm-start, not an exact resume.

Do not start either 5M job until Phase 5 external review and the Phase 6
CPU/smoke/migration/10k/100k gates pass.
