# Hierarchical DSRL-NA Residual Modulation V1: Action-Space Audit

## Scope and verdict

This is the Phase 0 read-only audit for Hopper (`hopper-medium-v2`) with the frozen DDIM5 policy and the 7.5M-step DSRL-NA network checkpoint. It traces each coordinate from the SAC noise actor to MuJoCo. Observed samples are cross-checks only and do not define legal bounds.

**Phase 0 bounds verdict: PASS.** `action_base` is in the normalized execution coordinate consumed by `ActionChunkWrapper`, whose audited flattened bounds are `[-1, 1]^12`. Phase 1 may therefore receive `exec_action_low` and `exec_action_high` explicitly from `ActionChunkWrapper.action_space.low/high`. The composition helper must remain general and must not infer or hard-code these values.

## Provenance

| Item | Path | Revision or SHA-256 |
|---|---|---|
| Outer repository base | `/home/mrf/dsrl` | `8d21b9cf55459f022c0e769bca1343edd9d30e5a` |
| SB3 submodule base | `stable-baselines3` | `10e5d311b5c42fe571ff3eeb72f622ca616a149d` |
| DPPO submodule base | `dppo` | `86ce51834055c02f9013e60dd4c4275606d82df7` |
| DSRL-NA checkpoint | `logs/gym-dsrl/gym_hopper_dsrl_2026-07-23_21-21-02_1/2026-07-23_21-21-02_1/checkpoint/ft_policy_7500000_steps.zip` | `e75686d06f7297b870ee8d286fc36db6ecb6cb62e9c6a1783b9466b3f0691fb6` |
| Frozen diffusion checkpoint | `dppo/log/gym-pretrain/hopper-medium-v2_pre_diffusion_mlp_ta4_td20/2024-06-12_23-10-05/checkpoint/state_3000.pt` | `9a5839d3d172d1e24b5bed0831d49bb3116a7e62fafa223c57c322f4bc7e9121` |
| Hopper normalization | `dppo/log/gym/hopper-medium-v2/normalization.npz` | `d05b2943bc39772f7f770dfc0a4df2a9f205cb1e05ae74b585cd0dd382a0742e` |

The outer repository and both submodules already contained unrelated dirty or untracked changes. Phase 0 treated them as user-owned and did not modify them.

## Complete dataflow

| Variable | Producer | Consumer | Shape | Coordinate system | Legal bounds source | Observed range | Transformation |
|---|---|---|---:|---|---|---|---|
| `observation` | `ObservationWrapperGym.reset/step` | Noise actor and diffusion condition | `(B,11)` | Dataset min/max normalized | Wrapper declares `[-1,1]^11`, but does not clip | May exceed declaration | `2*((raw-observation_min)/(observation_max-observation_min+1e-6)-0.5)` |
| `noise_scaled` | Squashed SAC actor | Existing DSRL unscale | `(B,12)` | SAC scaled/squashed noise | Noise actor / action-chunk action space | `[-0.9986892343, 0.9990059733]` | Existing SB3 actor scaling |
| `noise_decoder_input` | `BasePolicy.unscale_action` | Frozen diffusion decoder | `(B,4,3)` | Existing DSRL decoder input | Actual `unscale_action` implementation and actor action space | Same min/max; round-trip max error `5.9604645e-08` | `low+0.5*(noise_scaled+1)*(high-low)`, reshape `(B,12)->(B,4,3)` |
| `action_base` | Frozen DDIM5 policy | Residual composition | `(B,12)` | Normalized execution | Diffusion normalization, `denoised_clip_value=1.0`, wrapper contract | `[-0.9992717505,1.0]` | DDIM decode, flatten `(B,4,3)->(B,12)` |
| `residual_unit` | Phase 1 residual actor | Residual scaling | `(B,12)` | Unit residual | Architectural `tanh` contract | N/A in Phase 0 | `tanh(residual_pre_tanh)` |
| `action_residual_delta` | Phase 1 composition | Addition | `(B,12)` | Normalized execution | Explicit execution bounds and scale | N/A in Phase 0 | `residual_scale*((high-low)/2)*residual_unit` |
| `action_pre_clip` | Phase 1 composition | Clamp | `(B,12)` | Normalized execution | Same explicit bounds | N/A in Phase 0 | `action_base+action_residual_delta` |
| `action_exec` | Phase 1 composition | `ActionChunkWrapper.step` | `(B,12)` | Normalized execution | `ActionChunkWrapper.action_space.low/high` | Must be within explicit bounds | Per-dimension clamp |
| Primitive normalized action | `ActionChunkWrapper.step` | `ObservationWrapperGym.step` | `(3,)` | Normalized execution | Wrapped D4RL action space | `[-1,1]^3` interface | Row-major `(12,)->(4,3)`, one row per step |
| Dataset-scale action | `ObservationWrapperGym.unnormalize_action` | D4RL normalized wrapper | `(3,)` | Dataset action coordinate | `normalization.npz` | Extrema below | `(action+1)/2*(action_max-action_min)+action_min` |
| Raw Hopper action | D4RL `NormalizedBoxEnv` | MuJoCo actuators | `(3,)` | MuJoCo control | Hopper actuator/action space | `[-1,1]^3` interface | D4RL normalized action mapping |

For Hopper, the noise policy space and normalized execution space happen to share numeric bounds `[-1,1]^12`. They remain different semantic variables. The implementation must call the existing `unscale_action` transform and retain the names `noise_scaled` and `noise_decoder_input`.

## Boundary evidence

### Observation normalization

`ObservationWrapperGym.normalize_obs` applies dataset min/max normalization without clipping. `ActionChunkWrapper` declares `[-1,1]^11`; this is a nominal coordinate declaration, not a runtime guarantee.

### SAC noise and DSRL unscale

The DSRL rollout and prediction paths use SB3 scaling/unscaling before diffusion reshape. `BasePolicy.unscale_action` implements:

```text
low + 0.5 * (scaled_action + 1.0) * (high - low)
```

Because the audited Hopper action space is `[-1,1]^12`, this is numerically identity up to rounding here. It remains a required transform and cannot be replaced by an identity assumption.

### Frozen diffusion output

The base model has 20 training diffusion steps, DDIM with five evaluation steps, and `denoised_clip_value=1.0`. Its implementation clips reconstructed `x_recon` to `[-1,1]`. Source inspection and sampling agree that the flattened returned sample is valid in the normalized execution coordinate.

### Chunk and environment mapping

`ActionChunkWrapper.action_space` tiles the wrapped primitive bounds four times. `step` row-major reshapes `(12,)` to `(4,3)` and executes four complete three-dimensional actions. This matches diffusion flattening order.

`ObservationWrapperGym` maps each normalized primitive action to:

```text
action_min = [-0.9999679, -0.9999835, -0.99996823]
action_max = [ 0.9998843,  0.9999483,  0.9999945]
```

Every dataset extremum lies within Hopper's raw `[-1,1]^3` bounds. No legal-bounds sources conflict.

## Frozen-policy diagnostics

The following are diagnostics, not definitions of legal bounds.

### 128 learned-DSRL decisions

```text
noise_scaled min/max:               -0.9986892342567444 / 0.9990059733390808
noise_decoder_input min/max:        -0.9986892342567444 / 0.9990059733390808
scale/unscale max absolute error:    5.960464477539063e-08
action_base min/max:                 -0.9992717504501343 / 1.0
action_base per primitive dim min:   [-0.9742333, -0.9853786, -0.99927175]
action_base per primitive dim max:   [0.9803126, 0.96570915, 1.0]
out-of-bounds scalar fraction:       0.0
abs(action_base) >= 0.999 fraction:  0.0013020833333333333
```

### 128 Gaussian-prior chunks through the frozen decoder

```text
action_base min/max:                 -1.0 / 1.0
out-of-bounds scalar fraction:       0.0
abs(action_base) >= 0.999 fraction:  0.005208333333333333
```

## Commands and exact artifact checks

Run from `/home/mrf/dsrl`:

```bash
git status --short
git log -1 --format='%H'
git -C stable-baselines3 log -1 --format='%H'
git -C dppo log -1 --format='%H'
rg -n "class (ActionChunkWrapper|ObservationWrapperGym)|def (scale_action|unscale_action)|denoised_clip_value|class DSRL|diffusion_act_dim" train_dsrl.py env_utils.py stable-baselines3/stable_baselines3 dppo -g '*.py' -g '*.yaml'
sha256sum logs/gym-dsrl/gym_hopper_dsrl_2026-07-23_21-21-02_1/2026-07-23_21-21-02_1/checkpoint/ft_policy_7500000_steps.zip dppo/log/gym-pretrain/hopper-medium-v2_pre_diffusion_mlp_ta4_td20/2024-06-12_23-10-05/checkpoint/state_3000.pt dppo/log/gym/hopper-medium-v2/normalization.npz
/home/mrf/miniconda3/envs/dsrl/bin/python <read-only action-chain diagnostic>
/home/mrf/miniconda3/envs/dsrl/bin/python <official-load checkpoint inspector>
/home/mrf/miniconda3/envs/dsrl/bin/python <QW-to-QM parity diagnostic>
```

The Python diagnostics rebuilt the configured environment/frozen policy and printed the statistics above. They did not write training code or alter artifacts.

## Checkpoint-interface findings for later phases

The legacy checkpoint has actor output dimension 12; twin QA/Q-action-target first layers `(128,23)`; twin QW first layers `(128,23)`; `target_entropy=0.0`; `log_ent_coef` approximately `-1.8300434`; and alpha approximately `0.1604066`. Observation dimension is 11 and diffusion layout is `4x3`.

Direct `DSRL.load` constructs an `_init_setup_model=False` shell that indexes `diffusion_act_dim=None` before saved data restoration. Phase 2 must not change `dsrl.py`: it must use the planned private load-only `LegacyLoadableDSRL`, inherited official `.load()`, a minimal temporary buffer, and `custom_objects={"diffusion_policy": reconstructed_frozen_policy}` to replace the checkpoint's CUDA-referencing sampler. It must then validate the restored real dimensions. No zip parsing is permitted.

A conventional dense first-layer expansion from 23 to 35 inputs changed the GEMM kernel and produced differences up to about `1.22e-4` with zero residual columns. Phase 2 therefore requires a block-linear first affine preserving the old `(observation,noise_scaled)` GEMM and adding a separate zero-initialized residual projection. The dry-run was bitwise equal to QW for batch sizes 1, 2, 17, 256, and 1024.

## Unresolved non-blocking risks

1. Online actor noise is tanh-bounded, while the original prefill/QW path samples an unbounded Gaussian prior. V1 does not change legacy behavior; QM uses policy samples only.
2. Normalized observations can exceed declared `[-1,1]` because there is no clip. V1 does not change observation normalization.
3. `ActionChunkWrapper` keeps iterating remaining primitive actions after a primitive done. This is existing, outside the bounds decision, and outside V1 scope.

None conflicts with the selected execution bounds or the coordinate of `action_base`.

## Minimal later modification set

- New hierarchical module under `stable-baselines3/stable_baselines3/dsrl/`.
- New focused tests under `stable-baselines3/tests/`.
- SB3 export only when the new class is ready.
- `train_dsrl.py` and Hopper config only in Phase 5.
- Documents/handoffs under `docs/rfs_hier_v1/`.

The existing `dsrl.py`, common ReplayBuffer, SAC policy, flat `RFSDSRL`, and VLA-JEPA are outside the modification set.
