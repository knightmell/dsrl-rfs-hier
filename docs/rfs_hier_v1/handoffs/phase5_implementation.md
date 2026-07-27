# Phase 5 Entry, Configuration, and Diagnostics Handoff

This is an implementation handoff for external review. It is not a Reviewer verdict.

## Modified files

SB3 Phase 5 commit:

- `stable-baselines3/stable_baselines3/__init__.py`
- `stable-baselines3/stable_baselines3/dsrl/__init__.py`
- `stable-baselines3/stable_baselines3/dsrl/hierarchical_rfs_dsrl.py`
- `stable-baselines3/tests/test_dsrl_rfs_hier_phase5.py`

Outer Phase 5 integration commit:

- `cfg/gym/dsrl_hopper.yaml`
- `train_dsrl.py`
- `utils.py`
- `docs/rfs_hier_v1/pilot_commands.md`
- `stable-baselines3` gitlink

The original `stable_baselines3/dsrl/dsrl.py` remains unchanged with SHA-256
`41dc8cfb86b8640284eb29f9b5b585e9906e93c30e7dfe684dd7da8adcbf6bc8`.

Hunk-level index staging excluded the existing flat-RFS export, entry, callback,
and configuration changes. The exact commits contain only the hierarchy option;
the user's flat-RFS work remains in the working tree.

## Implemented behavior

### Public algorithm entry

- `HierarchicalRFSDSRL` is exported from `stable_baselines3.dsrl` and the SB3
  top level.
- `train_dsrl.py` accepts the unique new option
  `algorithm=dsrl_na_rfs_hier`.
- Existing `algorithm=dsrl_na` construction remains unchanged, except that the
  global `model.learn()` call now reads the Hydra `total_timesteps` field. Its
  Hopper default remains 20M.

The hierarchy branch:

1. Reconstructs the frozen diffusion policy through the existing loader.
2. Reads `exec_action_low/high` directly from the constructed vectorized
   environment action space, converts to float32, and flattens without assuming
   numeric values or Hopper dimensions.
3. Constructs `HierarchicalRFSDSRL` with the audited diffusion layout and V1
   parameters.
4. Resolves the configured legacy checkpoint through
   `hydra.utils.to_absolute_path`.
5. Calls `initialize_from_legacy_checkpoint`, which uses the official legacy
   load path and rebuilds optimizers.

The configured checkpoint is the audited Hopper 7.5M network checkpoint:

```text
./logs/gym-dsrl/gym_hopper_dsrl_2026-07-23_21-21-02_1/2026-07-23_21-21-02_1/checkpoint/ft_policy_7500000_steps.zip
```

SHA-256:

```text
e75686d06f7297b870ee8d286fc36db6ecb6cb62e9c6a1783b9466b3f0691fb6
```

### Hopper V1 configuration

- QA: `utd=20`.
- QM: `noise_critic_grad_steps=10`.
- Noise actor: 5.
- Residual actor: 5.
- `residual_scale=0.1`.
- `residual_penalty_coef=0.0`.
- Residual MLP: `[128,128]`, SiLU, LR `3e-4`.
- `target_ent=0.0` remains unchanged.
- `total_timesteps` is Hydra-configurable.

### Phase 5 diagnostics

The last replay batch already sampled by the current update is reused under
`torch.no_grad`; diagnostics do not add a replay sample or replay field.

Recorded losses:

- `train/noise_actor_loss`
- `train/residual_actor_loss`
- `train/action_critic_loss`
- `train/modulation_critic_loss`
- `distill_mse/policy_samples`

Recorded action/noise diagnostics:

- `noise_scaled_l2`
- `noise_decoder_input_l2`
- `action_base_l2`
- `residual_unit_mean_abs`
- `residual_unit_l2`
- `residual_tanh_saturation_fraction`
- `action_residual_delta_l2`
- `action_pre_clip_l2`
- `action_exec_l2`
- `residual_to_base_ratio`
- `clip_fraction`
- `effective_residual_l2`

Recorded Q differences:

- `Q_action(action_exec)-Q_action(action_base)`
- `Q_modulation(noise_scaled,residual_unit)-Q_modulation(noise_scaled,0)`

All L2 values are the mean of per-sample flattened vector norms.
`clip_fraction` is the scalar-dimension fraction for which `action_pre_clip` is
strictly outside explicit execution bounds. Tanh saturation uses
`abs(residual_unit) >= 0.99`. The residual/base ratio uses the norm of actual
`action_residual_delta`, divided by `norm(action_base)+1e-6`. Both Q differences
use the minimum over twin critics.

The existing callback forwards these metrics to W&B only for the hierarchy
algorithm and uses `predict_diffused` for hierarchy evaluation. Base-policy
prefill stores executed diffusion actions for both DSRL-NA variants.

### Pilot command

`docs/rfs_hier_v1/pilot_commands.md` contains a Hydra configuration-only check
and the 5M hierarchy pilot command. The command explicitly sets
`env.n_envs=10`, so it does not depend on an unrelated unstaged local config
hunk. It describes the run as a network warm-start, not an exact resume.

No 5M training was launched in Phase 5.

## Tests and exact results

Focused Phase 5 SB3 tests:

```text
3 passed, 1 warning in 1.72s
```

These verify hierarchy exports, every diagnostic key, execution-delta and clip
definitions, saturation, no diagnostic gradients, and twin-min Q differences.

Phase 1–5 plus the existing original-DSRL regression in the shared working tree:

```text
34 passed, 1 warning in 2.18s
```

Exact SB3 Phase 5 commit in a clean temporary worktree with no flat-RFS files:

```text
33 passed, 1 warning in 2.27s
```

The temporary worktree was removed after the test.

Hydra configuration-only command resolved successfully without starting
training and produced:

```text
algorithm: dsrl_na_rfs_hier
total_timesteps: 5000000
QA/QM/noise/residual: 20/10/5/5
```

Project Python 3.9 checks:

```text
PHASE5_ENTRY_CALLBACK_ASSERTIONS: PASS
STAGED_TRAIN_ENTRY: PASS
STAGED_CALLBACK: PASS
STAGED_HOPPER_CONFIG: PASS
```

The first staged-entry boundary assertion used an overly broad substring check
that matched the suffix of `HierarchicalRFSDSRL`; the corrected exact-token
assertion passed. Python compilation had already passed in the first attempt.

`git diff --cached --check` passed for both repositories before commits.

## Unresolved risks and external-review focus

- Phase 5 has not been externally reviewed. No PASS is claimed here.
- No real Hopper environment, DDIM5 smoke, 10k, 100k, or 5M training run was
  executed. These remain Phase 6 gates.
- The hierarchy 5M command is present. A matched DSRL-NA control launch from the
  same network weights must be finalized before the Phase 6 A/B 5M comparison;
  it must not be mislabeled as an exact resume.
- The callback retains the repository's existing policy of emitting training
  metrics when completed episode data is available.
- The checkpoint path is local to this project layout. Runtime migration still
  validates structure/dimensions, but the entry does not recompute its SHA-256.
- Existing unrelated dirty config, flat-RFS, DPPO, and callback changes remain
  outside the exact Phase 5 commits and require continued hunk-specific staging.
- External review should inspect explicit wrapper bounds, official warm-start,
  exact algorithm isolation, all diagnostic definitions, callback key safety,
  Hydra timesteps, and the clean-commit test evidence.

## Commit/base information

- Outer Phase 5 base:
  `8a01d86835b211d182dbebd24bf5503299f83012`
- SB3 Phase 5 base:
  `2d998b65f2985e483d4e52629e2d841c60c1b991`
- SB3 Phase 5 head:
  `b61a0a702a65cac2104d12655824f9c0432c803a`
- Outer Phase 5 integration head:
  `480ed1023f3047f3b1a67d8be0d9466a066c3eab`
- DPPO remains unchanged by Phase 5:
  `86ce51834055c02f9013e60dd4c4275606d82df7`
