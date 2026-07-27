# Phase 2 Checkpoint/QM Specialist Handoff

## Inspected files

- `stable-baselines3/stable_baselines3/dsrl/hierarchical_rfs_dsrl.py`
- `stable-baselines3/tests/test_dsrl_rfs_hier_phase2.py`
- `stable-baselines3/tests/test_dsrl_rfs_hier_phase1.py`
- `stable-baselines3/stable_baselines3/dsrl/dsrl.py`
- `stable-baselines3/stable_baselines3/common/base_class.py`
- `stable-baselines3/stable_baselines3/common/save_util.py`
- `stable-baselines3/stable_baselines3/common/policies.py`
- `stable-baselines3/stable_baselines3/common/torch_layers.py`
- `stable-baselines3/stable_baselines3/sac/policies.py`
- Real Hopper 7.5M DSRL checkpoint and frozen DDIM5 policy

## Verified facts

- Specialist verdict: **PASS**. This is not the independent Phase 2 Reviewer verdict.
- Legacy Hopper QW has twin heads with first `Linear(23,128)`, three complete `Linear -> LayerNorm -> Tanh` hidden blocks, and final `Linear(128,1)`.
- `BlockLinear` preserves the old 23-input observation/noise GEMM and adds a separate zero-initialized 12-input residual projection. All downstream module types, indices, parameters, and buffers copy exactly.
- QM is called as `Q_modulation(observation, noise_scaled, residual_unit)` and registers independent `qf0/qf1` heads.
- Real 7.5M checkpoint parity at zero residual is bitwise exact for batch sizes 1, 2, 17, 256, and 1024; maximum error is `0.0`.
- The inherited SB3 load order was verified: archive load/custom replacement, load shell construction, saved-data update, load-kwargs update, `_setup_model`, exact state loading, then PyTorch scalar restoration.
- Legacy archives contain `diffusion_policy`, so `_LegacyLoadableDSRL` replaces it through `custom_objects` and applies `buffer_size=1` through load kwargs before setup.
- The load-only `(1,1)` diffusion placeholder is never used by a configured model; real dimensions are restored and validated before setup. Normal construction still rejects missing dimensions.
- New hierarchy archives exclude `diffusion_policy`; loading requires `diffusion_policy=reconstructed_sampler`. Bounds remain in saved data and overwrite load-shell sentinels before full validation.
- Missing sampler injection fails explicitly before network setup.
- Migration copies only actor, QA, QA target, QW-derived QM, `target_entropy`, and entropy scalar. It rebuilds actor, QA, QM, residual, and entropy optimizers.
- Real checkpoint values: timesteps `7500000`, layout `4x3`, observation/action `(11,)/(12,)`, `target_entropy=0.0`, `log_ent_coef=-1.8300434350967407`, alpha `0.16040660440921783`.
- Real actor/QA/QA-target migration max differences are all `0.0`; residual-unit max absolute value is `0.0`; action parity max error is `0.0`; all five migrated optimizer state sizes are `0`.
- New-format save groups include policy, actor/QA optimizers, QM and optimizer, residual actor and optimizer, and entropy optimizer/scalar. A populated-optimizer round-trip restores nonempty Adam state for all five optimizers.
- Official Python 3.9 synthetic save/load returned `DSRL_PY39_SAVE_LOAD: PASS`, required explicit sampler injection, and restored module states exactly.
- Phase 2 focused tests: `11 passed, 1 warning in 1.96s`.
- Phase 1 + Phase 2 + original DSRL regression: `23 passed, 1 warning in 2.09s`.
- The only warning is the unavailable `pytest-env` plugin under disabled plugin auto-load.

## Unresolved conflicts

- None blocking independent Phase 2 review.
- The hierarchy is deliberately not exported yet.
- `critic_noise` is now a compatibility alias to the three-input QM, while inherited legacy `DSRL.train` expects a two-input QW. Phase 3 must replace training before the algorithm is exposed or learned.
- Existing flat-RFS/export dirty files remain outside the Phase 2 commit.

## Recommended tests

- Independent Reviewer should inspect only SB3 commit `0e3d4e62f7cec247d4b43be3f7d191094ac17ba9` relative to `70511336eb663a31921c25bef35c23d6759fe748`.
- Re-run Phase 2 focused tests, Phase 1 regression, and original DSRL regression.
- Verify no manual zip parsing appears in production migration code.
- Verify new archives require load-kwarg sampler injection and legacy archives use `custom_objects` replacement.
- Verify real/synthetic zero-residual action and QM parity no worse than `1e-6`, exact QA/QA-target/alpha transfer, and fresh migration optimizers.
- Phase 3 must add only QA/QM losses and must not yet add actor losses.

## Commit/base information

- Outer base for Phase 2: `d295365bdffc4da6ba9a5bf9c17c4984bec5652c`
- SB3 Phase 2 base: `70511336eb663a31921c25bef35c23d6759fe748`
- SB3 Phase 2 head: `0e3d4e62f7cec247d4b43be3f7d191094ac17ba9`
- DPPO unchanged: `86ce51834055c02f9013e60dd4c4275606d82df7`
- Original `dsrl.py` SHA-256: `41dc8cfb86b8640284eb29f9b5b585e9906e93c30e7dfe684dd7da8adcbf6bc8`
- Outer submodule pointer remains deliberately uncommitted until the later outer integration phase.
