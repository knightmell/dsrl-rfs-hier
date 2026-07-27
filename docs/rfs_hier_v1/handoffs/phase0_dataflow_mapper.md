# Phase 0 Dataflow Mapper Handoff

## Inspected files

- `env_utils.py`
- `train_dsrl.py`
- `cfg/gym/dsrl_hopper.yaml`
- `stable-baselines3/stable_baselines3/common/policies.py`
- `stable-baselines3/stable_baselines3/dsrl/dsrl.py`
- `dppo/model/diffusion/diffusion.py`
- `dppo/model/diffusion/diffusion_eval.py`
- `dppo/cfg/gym/pretrain/hopper-medium-v2/pre_diffusion_mlp.yaml`
- `dppo/log/gym/hopper-medium-v2/normalization.npz`
- Configured Hopper/D4RL wrapper action spaces
- Frozen DDIM and learned DSRL artifacts listed in `action_space_audit.md`

## Verified facts

- `observation` is 11-dimensional and dataset min/max normalized without clip.
- `noise_scaled` is 12-dimensional and passes through the existing SB3 `unscale_action` before diffusion decoding.
- Hopper makes that unscale numerically identity, but the transform is semantically required and must not be bypassed.
- `noise_decoder_input` reshapes to `(B,4,3)`.
- Frozen DDIM5 returns `action_base` in the normalized execution coordinate and uses `denoised_clip_value=1.0`.
- `ActionChunkWrapper` tiles primitive `[-1,1]^3` bounds four times and reshapes `(12,)` to four complete `(3,)` actions in matching row-major order.
- `ObservationWrapperGym` maps normalized actions to dataset extrema, all within raw Hopper `[-1,1]^3` bounds.
- The correct Hopper constructor values are explicit flattened `ActionChunkWrapper.action_space.low/high`, each shape `(12,)`.
- Learned-policy diagnostics found no `action_base` violations; observed samples are not the legal-bounds source.

## Unresolved conflicts

- None for the `action_base` coordinate or selected execution bounds.
- Non-blocking: normalized observations may exceed the declared box.
- Non-blocking: legacy Gaussian-prior noise is unbounded while actor noise is tanh-bounded.
- Non-blocking/out of scope: the existing chunk wrapper does not break the primitive loop immediately after done.

## Recommended tests

- Assert explicit constructor bounds are finite, ordered, and match flattened diffusion dimension without hard-coded Hopper dimensions.
- Assert the existing `unscale_action` is called and shape-preserving before decoder reshape.
- Assert per-dimension half-range residual scaling and explicit clamp.
- Test flatten/reshape order with nonuniform synthetic per-dimension bounds.
- Assert warmup uses decoded base actions with an exactly zero residual.
- Re-run frozen-policy range diagnostics in Hopper DDIM5 smoke validation.

## Commit/base information

- Outer base: `8d21b9cf55459f022c0e769bca1343edd9d30e5a`
- SB3 base (unchanged): `10e5d311b5c42fe571ff3eeb72f622ca616a149d`
- DPPO base (unchanged): `86ce51834055c02f9013e60dd4c4275606d82df7`
- Phase 0 head: outer commit containing this handoff; resolve with `git log -1 --format='%H' -- docs/rfs_hier_v1`.
