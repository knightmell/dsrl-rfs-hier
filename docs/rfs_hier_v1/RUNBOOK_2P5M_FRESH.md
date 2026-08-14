# Runbook — Three-Critic DSRL-NA from Frozen DDIM, 2.5M Fresh Run

Status: ACTIVE (stage S6 deliverable, 2026-08-06)
Anchor: `STAGED_DELIVERY_PLAN.md`, `THREE_CRITIC_IMPLEMENTATION_SPEC.md`, Codex plan (2026-08-05).
Environment: `dsrl` conda env (`/home/mrf/miniconda3/envs/dsrl/bin/python`), GPU required for training.

## 0. What this run is

A from-scratch three-critic Core V1 run: the Frozen DDIM decoder is permanently
frozen, and the noise actor, QA_base, QW_base, QA_joint, and residual actor are
all trained from their seeded constructors.

**Primary config — co-training schedule**
`fresh_frozen_ddim_2p5m_cotrain` (user design decision, 2026-08-08): a short
Phase B of 0.5M chunk transitions establishes the base branch AND shadow-trains
QA_joint on the same BASE transitions (β=0, so QA_joint's valid action IS the
base action); the long Phase R of 2.0M co-trains both branches at the same
time — QA_base→QW→noise/alpha continue at low frequency while QA_joint→residual
trains. This replaces the frozen 50:50 split (`fresh_frozen_ddim_2p5m`:
B=1.25M / R=1.25M with the noise actor + alpha frozen in R), which the user
rejected because freezing the noise actor at 1.25M caps residual-phase
performance and wastes the three-critic value of updating QA_base→QW→π_w and
QA_joint→π_r simultaneously. The frozen 2.5M profile is kept as a reference
schedule, not the primary run.

In R, β is held at 0 for the first 50k (joint data collection, QA_joint warm-up,
residual structurally OFF), then ramps 0.02→0.1 over the next 50k. Residual
exploration (pre-tanh structured perturbation on JOINT lanes, std 0.02) and
cross-lane replay 75/25 are enabled (see §7a deviations in
`STAGED_DELIVERY_PLAN.md`).

The primary goal is to beat the raw Frozen-DDIM Gaussian-prior checkpoint
(~1432 raw / ~100% early fall per `handoffs/P6_resip_2p5m_frozen_10k_result.md`)
by a wide margin.

Fidelity constraints (from `STAGED_DELIVERY_PLAN.md`):
- No QM_joint, no ranking loss (V1.1 is default OFF), no delta-Q hard gate
  (V1.3 is diagnostic-only), no residual dropout, no shared trunks, no PPO.
- Frozen parameters (untouched by the co-training deviations): beta_target=0.1,
  beta_ramp=50k, base_lane=0.5, min_branch=256, Phase J disabled.
- Co-training deviations (B=0.5M/R=2.0M, shadow QA_joint, unfrozen noise/alpha
  in R, β hold/floor, exploration, cross-lane replay) are registered in
  `STAGED_DELIVERY_PLAN.md` §7a with cited basis = user design decision.
- Do not modify `dsrl.py`, SB3-common, `env_utils.py`, `p6_launcher.py`, DPPO,
  flat-RFS, or per-step experiments.

## 1. Fresh 2.5M co-training launch

Config: `cfg/gym/p6_hopper_fresh_2p5m_cotrain.yaml` (inherits `p6_hopper`,
schedule `fresh_frozen_ddim_2p5m_cotrain`, B=0.5M / R=2.0M, prefill source
`fresh_frozen_ddim`, legacy checkpoint cleared).

```bash
cd /home/mrf/dsrl
env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/mrf/miniconda3/envs/dsrl/bin/python p6_train.py \
  --config-name p6_hopper_fresh_2p5m_cotrain \
  seed=1 \
  total_timesteps=2500000
```

What happens on launch (fresh branch in `p6_train.main`):
1. `static_preflight` verifies DDIM/normalization artifacts and the fresh
   schedule budget (B+R+J == 2.5M) plus the co-training contract
   (per-profile `update_profiles`, β hold/floor, model flags).
2. `_construct_hierarchy(..., init_mode="fresh")` builds the three critics +
   residual and calls `initialize_from_fresh_frozen_ddim()`: reference actor
   snapshot, hard-copied QA_base_target, zero residual, fresh optimizers.
   Because shadow mode is on, the QA_joint soft-clone at the B→R boundary is
   skipped — QA_joint keeps the shadow weights trained during B (its optimizer
   and Adam state are preserved).
3. `collect_or_load_tagged_matched_prefill(prefill_source="fresh_frozen_ddim")`
   generates the immutable Gaussian-prior BASE prefill (20,010 chunk
   transitions by default) and seeds the tagged replay.
4. **Phase B (0.5M)** is the base curriculum: trains the noise actor through
   QW_base→QA_base on BASE-lane data (residual structurally off, β=0) at the
   original DSRL ratio, AND shadow-trains QA_joint on the same BASE transitions
   (BASE profile 20/10/10/20/20/0). Cross-lane replay shares 25% of the other
   lane's transitions between QA_base and QA_joint.
5. **Phase R (2.0M)** co-trains both branches simultaneously: QA_base 5 /
   QA_joint 5 / QW 2 / noise 1 / alpha 1 / residual 1, with β held at 0 for the
   first 50k (residual OFF), then ramped 0.02→0.1 over 50k. Cross-lane replay
   and residual exploration stay active on JOINT lanes.

Reference (frozen schedule, not the primary run):
`cfg/gym/p6_hopper_fresh_2p5m.yaml` — B=1.25M / R=1.25M, residual-only R with
noise/alpha frozen, QA_base→QA_joint clone at the boundary, no hold/floor.

## 2. "Pre-training" (base curriculum)

There is no separate pre-training step. Phase B IS the base curriculum: it
trains the noise actor through QW_base→QA_base on Frozen-DDIM base-lane data
with residual structurally off. Because the cotrain schedule shadow-trains
QA_joint in B (BASE profile 20/10/10/20/20/0), QA_joint learns the base-action
value at the same scale as QA_base before R begins — so R does not start with a
sudden clone. The prefill (Gaussian prior through DDIM) seeds replay so Phase B
starts from the raw-checkpoint state distribution. Do not run a separate DSRL
pre-training — the plan defines the base curriculum as part of the algorithm
(Codex 2026-08-05, msg 557).

## 3. Interface check tools (run before / during the run)

### 3.1 Preflight (before launch, fast)
```bash
cd /home/mrf/dsrl
/home/mrf/miniconda3/envs/dsrl/bin/python -c "
from omegaconf import OmegaConf
from hydra import compose, initialize
import math
OmegaConf.register_new_resolver('eval', eval, replace=True)
OmegaConf.register_new_resolver('round_up', math.ceil, replace=True)
OmegaConf.register_new_resolver('round_down', math.floor, replace=True)
with initialize(version_base=None, config_path='cfg/gym'):
    cfg = compose(config_name='p6_hopper_fresh_2p5m_cotrain', overrides=['seed=1','total_timesteps=2500000'])
    OmegaConf.resolve(cfg)
    from p6_preflight import static_preflight
    from pathlib import Path
    m = static_preflight(cfg, Path('/home/mrf/dsrl'), algorithm='dsrl_na_rfs_hier')
    s = m['hierarchy_schedule']
    print('PREFLIGHT OK; schedule:', s['schedule_profile'])
    print('B/R/J:', s['phase_b_steps'], s['phase_r_steps'], s['phase_j_steps'])
    print('beta hold/floor:', s['beta_hold_steps'], s['beta_floor'])
    print('flags shadow/cross/explore:', s['qa_joint_shadow_in_b'], s['cross_lane_ratio'], s['residual_exploration_std'])
    print('R profile:', s['update_profiles']['residual'])
"
```
This exercises the real artifact verification (DDIM SHA-256, normalization,
seed plan) without launching training. Expected: `PREFLIGHT OK; schedule:
fresh_frozen_ddim_2p5m_cotrain`, B/R/J = 500000 / 2000000 / 0, hold/floor =
50000 / 0.02, flags True / 0.25 / 0.02, R profile `{'qa_base': 5, 'qa_joint': 5,
'qw_base': 2, 'noise_actor': 1, 'alpha': 1, 'residual_actor': 1}`.

### 3.2 Test suites (after any code change)
```bash
cd /home/mrf/dsrl/stable-baselines3
env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 MPLCONFIGDIR=/tmp/dsrl-mpl \
  /home/mrf/miniconda3/envs/dsrl/bin/python -m pytest -p no:cacheprovider \
  tests/test_dsrl_rfs_hier_phase1.py tests/test_dsrl_rfs_hier_phase2.py \
  tests/test_dsrl_rfs_hier_phase3.py tests/test_dsrl_rfs_hier_phase4.py \
  tests/test_dsrl_rfs_hier_phase5.py tests/test_dsrl_rfs_hier_phase6.py \
  tests/test_hierarchical_replay_buffer.py -q
cd /home/mrf/dsrl
env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 MPLCONFIGDIR=/tmp/dsrl-mpl \
  /home/mrf/miniconda3/envs/dsrl/bin/python -m pytest -p no:cacheprovider \
  tests/test_p6_runtime.py tests/test_p6_preflight.py tests/test_p6_checkpointing.py \
  tests/test_p6_evaluation.py tests/test_p6_train_wiring.py -q
```

### 3.3 Fresh-init construction sanity (unit test)
`tests/test_dsrl_rfs_hier_phase5.py::test_initialize_from_fresh_frozen_ddim_snapshot_and_defers_joint_clone`
verifies the target hard-copy, zero residual, fresh counters, and deferred
QA_joint clone (frozen path). Run it directly:
```bash
cd /home/mrf/dsrl/stable-baselines3
env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 MPLCONFIGDIR=/tmp/dsrl-mpl \
  /home/mrf/miniconda3/envs/dsrl/bin/python -m pytest -p no:cacheprovider \
  tests/test_dsrl_rfs_hier_phase5.py::test_initialize_from_fresh_frozen_ddim_snapshot_and_defers_joint_clone -q
```

### 3.4 Co-training schedule sanity (unit tests)
`tests/test_dsrl_rfs_hier_phase6.py` covers the cotrain schedule end-to-end at
the model level: shadow Phase B trains QA_joint on BASE rows without a boundary
clone, Phase R co-trains noise/alpha alongside the residual with the β-hold
gating the residual off, cross-lane replay composition/fallback, and residual
exploration flowing into the stored metadata (JOINT rows only, gated at β=0).
```bash
cd /home/mrf/dsrl/stable-baselines3
env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 MPLCONFIGDIR=/tmp/dsrl-mpl \
  /home/mrf/miniconda3/envs/dsrl/bin/python -m pytest -p no:cacheprovider \
  tests/test_dsrl_rfs_hier_phase6.py -q
```
`tests/test_p6_preflight.py::test_cotrain_manifest_update_profiles_and_flags`
checks the P6 contract for the cotrain config (fresh gate, per-profile
update_profiles, β hold/floor, model flags).

## 4. Diagnostic files to watch

The trainer logs `train/*` every call and `diagnostics/*` every
`diagnostics_interval_updates` (default 100) train calls. Key signals:

| Metric | Meaning | Watch for |
|---|---|---|
| `train/qa_base_loss`, `qa_joint_td` | Critic TD errors | Divergence/NaN |
| `train/effective_utd_base/joint` | Realized per-branch UTD | Excessive refitting |
| `train/realized_joint_lane_ratio` | Actual joint-lane share | ≈ 0.5 in R (before cross-lane replay sampling) |
| `train/emergency_clamp_count` | Bound-preservation safety | Should stay ~0 |
| `diagnostics/qa_joint_delta_head0/head1` | Headwise same-QA_joint ΔQ | Signal magnitude vs twin disagreement |
| `diagnostics/qa_base_twin_disagreement` | Twin critic spread | Growth = critic instability |
| `diagnostics/residual_unit_mean_abs` | Residual saturation | 1.0 = saturated (bad) |
| `diagnostics/qa_joint_target_drift` | Target-online drift | Stable small value |
| `diagnostics/rank_qw_teacher_spearman_head0/head1` | QW-vs-teacher ranking correlation | Growth = QW alias bridge learns the teacher's preference ordering (V1.1 gate evidence) |
| `diagnostics/rank_twin_head_ordering_agreement` | Teacher twin-head ranking agreement | High = teacher ranking is self-consistent |
| `diagnostics/rank_teacher_top1_gap_mean` | Teacher's decisive margin between top candidates | Small relative to value scale = teacher is not discriminative |
| `diagnostics/episode_return_base_mean` / `_joint_mean` | Closed-episode return per branch (since last dump) | BASE should improve through Phase B; JOINT should not degrade in R |
| `diagnostics/episode_early_fall_rate_base` / `_joint` | Early-fall rate per branch (since last dump) | Watch the raw ~100% baseline drop |

Online evaluation every 100k chunks produces `current_base_only`,
`current_full_hierarchy`, and `reference_base` exact-N reports under
`logs/p6/<run>/evaluations/`. **The honest comparison for "beat the raw
checkpoint" is `reference_base` (init actor ≈ raw prior) vs `current_base_only`
(trained base).** `current_full_hierarchy` adds the residual; do not let it mask
the base's own improvement.

## 5. The V1.3 delta-Q correlation gate (run after Phase B, before enabling residual)

The plan forbids using headwise ΔQ as a gate until it positively correlates with
real paired returns. Use `diagnose_qa_joint_delta_correlation.py` on a saved
Phase-B checkpoint (the checkpoint saved at the B→R boundary, `model.zip`):

```bash
cd /home/mrf/dsrl
/home/mrf/miniconda3/envs/dsrl/bin/python diagnose_qa_joint_delta_correlation.py \
  --model <path-to-phase-b-model.zip> \
  --config cfg/gym/p6_hopper_fresh_2p5m_cotrain.yaml \
  --device cuda:0 \
  --episodes 25 \
  --states-per-episode 4 \
  --horizon-chunks 16 \
  --eps 0.05 \
  --beta 0.1 \
  --seed 10000 \
  --output logs/qa_joint_delta_correlation.json
```

The diagnostic deterministically re-rolls the base policy from the (policy, env)
seeds to every collected chunk boundary, then branches with zero / +eps / −eps
residual-logit probes composed at `--beta`, continues the base policy for
`horizon_chunks − 1` more chunks, and compares conservative headwise QA_joint
ΔQ against the realized paired returns (Spearman, pairwise accuracy, top-1,
state-bootstrap 95% CI).

Notes:
- **`--beta` must be a probe value (default 0.1, the frozen target).** A
  Phase-B checkpoint sits at β≈0, where every residual probe collapses to the
  base action and the diagnostic measures nothing.
- **`--horizon-chunks 16` = 64 primitive steps** (act_steps=4), matching the
  horizon where the empirical delta-Q signal was positive
  (`handoffs/P6_std_horizon_long_gae_4k_result.md`). The default 4 chunks = 16
  primitive steps is a cheap first pass.
- Requires GPU, MuJoCo/D4RL, and the real checkpoint; it never trains.

The output `screening_gate` flags fire only if the bootstrap 95% lower bound of
Spearman > 0 (and pairwise accuracy > 0.55). Only then may V1.1/V1.2 flags be
considered — and then only one at a time, each gated on its own evidence.

## 6. Resume

The run saves resume bundles every 500k chunks and a model checkpoint every
100k chunks. Resume with:
```bash
/home/mrf/miniconda3/envs/dsrl/bin/python p6_train.py \
  --config-name p6_hopper_fresh_2p5m_cotrain \
  seed=1 total_timesteps=2500000 \
  p6.resume_bundle_path=<latest-bundle-dir>
```
Resume is `reset_boundary_discontinuous` (the certified default): learner/replay/
RNG state is restored exactly, environments are reset with derived seeds.

> **Source fingerprint warning.** Resume re-verifies `source_state_sha256`, which
> hashes every untracked, non-gitignored file in the outer repo, the
> `stable-baselines3` submodule, and `dppo`. Creating or editing any such file
> between launch and resume (e.g. a new diagnostic script or an output JSON at
> the repo root) fails the resume with "Resume run manifest mismatch for
> source_state_sha256". Keep run outputs under the gitignored `logs/` directory
> (the §5 diagnostic default now writes there) and commit or gitignore any new
> source file before resuming.
