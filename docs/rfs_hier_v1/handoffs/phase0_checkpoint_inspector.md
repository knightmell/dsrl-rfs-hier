# Phase 0 Checkpoint Inspector Handoff

## Inspected files

- `stable-baselines3/stable_baselines3/dsrl/dsrl.py`
- `stable-baselines3/stable_baselines3/common/base_class.py`
- `stable-baselines3/stable_baselines3/common/save_util.py`
- `stable-baselines3/stable_baselines3/sac/policies.py`
- `train_dsrl.py`
- `utils.py`
- `logs/gym-dsrl/gym_hopper_dsrl_2026-07-23_21-21-02_1/2026-07-23_21-21-02_1/checkpoint/ft_policy_7500000_steps.zip`
- Reconstructed frozen diffusion policy selected by Hopper config

## Verified facts

- Checkpoint: 7.5M timesteps, observation dimension 11, noise/execution dimension 12, diffusion layout `4x3`.
- Noise actor output heads have shape `(12,128)`.
- Twin QA and QA-target first layers have shape `(128,23)`.
- Twin legacy QW first layers have shape `(128,23)` and there is no QW target.
- `target_entropy=0.0`, `log_ent_coef` is approximately `-1.8300434`, and alpha is approximately `0.1604066`.
- The archive saves policy, actor optimizer, critic optimizer, QW, entropy optimizer, and `log_ent_coef`; it does not save replay or a QW optimizer.
- Direct `DSRL.load` fails because its temporary shell indexes `diffusion_act_dim=None` before checkpoint data restoration.
- A private load-only subclass can provide a placeholder only for the `_init_setup_model=False` shell, inherit official `.load()`, restore real `4x3` dimensions, and validate them before setup. Zip parsing is unnecessary and forbidden.
- CPU loading requires `custom_objects={"diffusion_policy": reconstructed_frozen_policy}` because the embedded sampler references CUDA storage.
- The temporary legacy model can use a minimal buffer to avoid allocating 10M replay entries.
- New optimizers must be fresh. Only noise actor, QA, QA target, QW-derived QM weights, and entropy scalar are inherited.
- A conventional widened QW first layer produced errors up to about `1.22e-4` despite zero residual columns.
- A block-linear first affine preserving the old 23-input GEMM and adding a zero residual projection was bitwise equal for batch sizes 1, 2, 17, 256, and 1024.

## Unresolved conflicts

- None blocking Phase 1.
- Phase 2 must prove official loading restores real action dimensions and rejects mismatched checkpoints.
- Phase 2 must prove exact QA/QA-target/alpha transfer and QM parity at most `1e-6`.

## Recommended tests

- Official legacy-load test with the 7.5M checkpoint, reconstructed policy, and minimal temporary replay capacity.
- Assert restored `(diffusion_act_chunk,diffusion_act_dim)==(4,3)` and actor output 12 before copying.
- Parameter-by-parameter exact actor, QA, and QA-target equality.
- Exact entropy scalar/target-entropy checks; assert fresh optimizers with no inherited state.
- QM parity over multiple batch sizes at `residual_unit==0`, maximum error `<=1e-6`.
- New-algorithm save/load round-trip without invoking legacy loader.

## Commit/base information

- Outer base: `8d21b9cf55459f022c0e769bca1343edd9d30e5a`
- SB3 base (unchanged): `10e5d311b5c42fe571ff3eeb72f622ca616a149d`
- DPPO base (unchanged): `86ce51834055c02f9013e60dd4c4275606d82df7`
- Phase 0 head: outer commit containing this handoff; resolve with `git log -1 --format='%H' -- docs/rfs_hier_v1`.
