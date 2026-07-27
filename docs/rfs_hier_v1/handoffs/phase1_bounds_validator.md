# Phase 1 Bounds/API Specialist Handoff

## Inspected files

- `stable-baselines3/stable_baselines3/dsrl/hierarchical_rfs_dsrl.py`
- `stable-baselines3/tests/test_dsrl_rfs_hier_phase1.py`
- `stable-baselines3/stable_baselines3/dsrl/dsrl.py`
- `stable-baselines3/stable_baselines3/common/policies.py`
- `utils.py`
- `docs/rfs_hier_v1/action_space_audit.md`
- `docs/rfs_hier_v1/handoffs/phase0_dataflow_mapper.md`
- Relevant SB3 exports and repository status/diff

## Verified facts

- Specialist verdict: **PASS**. This is a bounds/API specialist result, not the independent Phase 1 Reviewer verdict.
- Execution bounds are explicit, finite, ordered, shaped by flattened diffusion dimension, and checked against the environment wrapper action space.
- `compose_action` has no Hopper-specific numeric bound assumption and implements `residual_scale * (high-low)/2 * residual_unit` followed by explicit clamp.
- The implementation calls the existing `policy.scale_action` and `policy.unscale_action`; numeric identity on Hopper is not used to bypass either interface.
- The Hopper shape contract is supported: observation `(B,11)`, `noise_scaled` `(B,12)`, `noise_decoder_input` `(B,4,3)`, and all base/residual/execution actions `(B,12)`.
- The residual actor explicitly concatenates `(observation, noise_scaled, action_base)`; hidden blocks are Linear, LayerNorm, SiLU; the final layer is exactly zero initialized.
- The decoder is put in evaluation mode, all module parameters are marked non-trainable, decode executes under `torch.no_grad`, and `action_base` is detached before residual conditioning.
- Decoder output layout is checked exactly as `(B, diffusion_act_chunk, diffusion_act_dim)` before flattening.
- Warmup sets `zero_residual=True` and skips residual actor evaluation, even when residual output bias is intentionally nonzero.
- Rollout `_sample_action` and evaluation `predict_diffused` use the same `_generate_hierarchical_action` helper.
- Replay receives a copy of actual clipped `action_exec`.
- `dsrl.py` is unchanged from SB3 base and has SHA-256 `41dc8cfb86b8640284eb29f9b5b585e9906e93c30e7dfe684dd7da8adcbf6bc8`.
- The new module does not override `train`, add losses/QM/checkpoint logic, or enter exports/config/training entrypoints.
- Focused tests after specialist feedback: `12 passed, 1 warning in 1.81s`; the warning is only the system pytest environment not loading the repository's `pytest-env` plugin.
- Official `dsrl` Python 3.9 direct assertions: `DSRL_ENV_ASSERTIONS: PASS`; warmup max absolute unscale round-trip error `4.470348358154297e-08`; decoder eval `True`; decoder requires-grad `False`; decoder calls `1`.

## Unresolved conflicts

- None blocking independent Phase 1 review.
- The `dsrl` Conda environment lacks pytest, so its validation used direct executable assertions; focused pytest ran under the available base interpreter with plugin auto-load disabled.
- The class is intentionally not yet a complete trainable/saveable algorithm: inherited legacy training does not update the residual actor, and Phase 2–4 have not implemented migration/QM/losses.
- Real Hopper `(11,4x3)` smoke is reserved for Phase 6; Phase 1 uses synthetic `(3,2x2)` unit fixtures.

## Recommended tests

- Independent Reviewer should inspect only SB3 commit `70511336eb663a31921c25bef35c23d6759fe748` relative to the Phase 1 SB3 base.
- Re-run the focused Phase 1 tests and original DSRL regression test.
- Verify required explicit bounds with a nonuniform synthetic composition case.
- Verify zero initialization, explicit three-input residual conditioning, common rollout/predict helper, and warmup residual bypass.
- Verify exact decoder output shape rejection and frozen decoder state.
- Defer real Hopper smoke, checkpoint save/load, QM, and losses to their specified later phases.

## Commit/base information

- Outer base for Phase 1: `1609e8e587b29ae07b1bf35d1bc05272a823a4d9`
- SB3 Phase 1 base: `10e5d311b5c42fe571ff3eeb72f622ca616a149d`
- SB3 Phase 1 head: `70511336eb663a31921c25bef35c23d6759fe748`
- DPPO unchanged: `86ce51834055c02f9013e60dd4c4275606d82df7`
- Outer submodule pointer is deliberately not committed until the planned later outer integration phase.
