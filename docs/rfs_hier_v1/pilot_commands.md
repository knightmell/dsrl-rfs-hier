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

## 5M hierarchy pilot

```bash
/home/mrf/miniconda3/envs/dsrl/bin/python train_dsrl.py \
  --config-path cfg/gym \
  --config-name dsrl_hopper \
  algorithm=dsrl_na_rfs_hier \
  total_timesteps=5000000 \
  seed=1
```

The entry constructs a new hierarchy model with an empty replay buffer and new
optimizers, then performs a network warm-start through the formal legacy loader.
This is not an exact training resume. The configured `init_rollout_steps` are
new interactions collected after construction.

Do not start the pilot until Phase 5 external review and the preceding Phase 6
CPU/smoke/migration/10k/100k gates pass.
