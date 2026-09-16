# Experiment Registry

Last updated: 2026-09-16

This registry separates intended runs from demonstrated runs. A row is not `complete` without a terminal marker, checkpoint or bundle, and evaluation evidence.

## Evidence levels

| Status | Meaning |
|---|---|
| planned | Name or config exists; no training claim |
| running | Live process and `RUNNING` status observed |
| interrupted | Process ended intentionally with a resumable boundary |
| stopped | Process ended by explicit user request before a resumable boundary; not result evidence |
| complete | Final boundary, terminal marker, checkpoint or bundle, and evaluation verified |
| training-complete | Training reached its declared boundary; evaluation evidence is still pending |
| finalization_failed | Training reached a saved boundary, but the run failed during finalization; record checkpoint and evaluation evidence separately |
| legacy-diagnostic | Real artifacts exist but the protocol does not match the current comparison |
| unknown | Current artifacts are insufficient; archive may be consulted only as a lead |

## Audited locomotion entries

| Canonical identity | Actual artifact path | Status | Interpretation |
|---|---|---|---|
| `hopper_s1_noclip_k4_300k_600k_full` | `logs/p6/fresh_frozen_ddim_600k_k4_additive_res_noclip_hopper-medium-v2_dsrl_na_rfs_hier_seed1_600000chunks` | complete | Full run branched from the 300k BASE state |
| `hopper_s1_noclip_k4_300k_600k_basecontrol` | — | planned | No independent control artifact found; checkpoint base view is not this run |
| `halfcheetah_s1_noclip_k4_300k_600k_full` | `logs/p6/fresh_frozen_ddim_600k_k4_additive_res_noclip_halfcheetah-medium-v2_dsrl_na_rfs_hier_seed1_600000chunks` | complete | Full run branched from the 300k BASE state |
| `halfcheetah_s1_noclip_k4_300k_600k_basecontrol` | — | planned | No independent control artifact found; checkpoint base view is not this run |
| `walker_s1_noclip_k4_500k_600k_full` | `logs/p6/walker_k4_additive_res_750k_fresh_frozen_ddim_walker2d-medium-v2_dsrl_na_rfs_hier_seed1_2500000chunks_tbfix_v1` | legacy-diagnostic | Common branch is 500k, not the intended 300k main comparison |
| `walker_s1_noclip_k4_500k_600k_basecontrol` | `logs/p6/walker_k4_base_continue_750k_fresh_frozen_ddim_walker2d-medium-v2_dsrl_na_rfs_hier_seed1_2500000chunks_tbfix_v1` | legacy-diagnostic | Independent control exists, but uses the legacy 500k boundary |

Before quoting a score, add the exact evaluation path, episode count, checkpoint transition, and seed to the relevant row or a linked results table. Do not reconstruct missing evidence from archived prose.

## 800k campaign

`full` and `basecontrol` branches must use the same declared BASE source and
BASE update schedule where they are paired; disabling residuals only at
evaluation time is not sufficient to create a `basecontrol` result. `noclip`
is the canonical spelling. For VS-Hier full runs, the agreed continuation uses
RES UTD=4; for base-control runs, RES UTD=0. Shared BASE-phase joint-critic shadow updates may remain; do not confuse them with residual updates.

**9月15日状态更新：** 本机未发现活动训练进程。目录内旧 attempt 的 `running` 字段不是活动证据；以下状态以 launcher 终态、最新完成 attempt 与 checkpoint/evaluation 产物为准。当前 resume 点不改变历史分岔点。

| Canonical identity | Exact resume source | Role | Target | Status |
|---|---|---|---:|---|
| `walker_s1_noclip_k4_300k_800k_full` | `logs/p6/base_diag_qw_gaussian_k4_b256_noclip_100k_fresh_frozen_ddim_walker2d-medium-v2_dsrl_na_rfs_hier_seed1_2500000chunks/resume/chunk_000000300000_source_compat_v1` | VS-Hier full, RES UTD=4 | 800k | interrupted as planned at 800k (exit 75); 10-episode online eval D4RL 105.79, early-fall 0%; 100-episode final eval pending |
| `walker_s1_noclip_k4_300k_800k_basecontrol` | `logs/p6/walker_s1_noclip_k4_300k_800k_basecontrol/resume/chunk_000000600000` (branch lineage records the shared Walker 300k source) | independent BASE-only control, RES UTD=0 | 800k | interrupted as planned at 800k (exit 75); 10ep `current_base_only`: D4RL 98.31, early-fall 0%; 100ep final eval pending |
| `hopper_s1_noclip_k4_300k_800k_full` | `logs/p6/hopper_s1_noclip_k4_300k_800k_full` | existing 300k-branch full continuation from 600k, RES UTD=4 | 800k | training-complete (launcher exit 0; 800k model present); final evaluation evidence not yet registered |
| `hopper_s1_noclip_k4_300k_800k_basecontrol` | `logs/p6/hopper_s1_noclip_k4_300k_800k_basecontrol` | intended independent BASE-only control from 300k, RES UTD=0 | 800k | unknown: stale `RUNNING` marker but no live process and no terminal marker; do not quote as a result |
| `halfcheetah_s1_noclip_k4_300k_800k_full` | `logs/p6/halfcheetah_s1_noclip_k4_300k_800k_full` | existing 300k-branch full continuation from 600k, RES UTD=4 | 800k | training-complete (launcher exit 0; 800k model present); final evaluation evidence not yet registered |
| `halfcheetah_s1_noclip_k4_300k_800k_basecontrol` | `logs/p6/fresh_frozen_ddim_600k_k4_noclip_halfcheetah-medium-v2_dsrl_na_rfs_hier_seed1_600000chunks_v2/resume/chunk_000000300000` | independent BASE-only control, RES UTD=0 | 800k | planned |
| `hopper_s3_noclip_k4_300k_800k_full` | `logs/p6/hopper_s3_noclip_k4_300k_800k_full/resume/chunk_000000300000_schedule300k_v1` | independent full continuation, Gaussian teacher, K=4, NoClip, RES UTD=4 | 800k | interrupted as planned at 800k (exit 75); 10ep full D4RL 98.03, early-fall 0%; 10ep checkpoint base view 97.69, early-fall 0%; 100ep final eval pending. Discard attempt 2 because its serialized BASE boundary was 500k. |
| `hopper_s3_noclip_k4_300k_800k_basecontrol` | `logs/p6/hopper_s3_noclip_k4_300k_800k_basecontrol/resume/chunk_000000300000` | independent BASE-only control, Gaussian teacher, K=4, NoClip, RES UTD=0 | 800k | interrupted as planned at 800k (exit 75); 10ep D4RL 95.84, early-fall 10%; 100ep final eval pending |
| `hopper_s1_matched_dsrl_600k_800k_control` | `logs/p6/fresh_frozen_ddim_matched_dsrl_hopper-medium-v2_seed1_600000chunks/resume/chunk_000000800000` | attempt 2 reached 800k; finalization failed on pre-existing `checkpoints/final_model.zip`; 100ep eval D4RL 92.81, early-fall 28%; `evaluations/final800k_100ep_attempt2_current_base_only.json` | 800k | finalization_failed |
| `hopper_s2_matched_dsrl_600k_800k_control` | `logs/p6/fresh_frozen_ddim_matched_dsrl_hopper-medium-v2_seed2_600000chunks/resume/chunk_000000800000` | attempt 2 reached 800k; finalization failed on pre-existing `checkpoints/final_model.zip`; 100ep eval D4RL 96.34, early-fall 15%; `evaluations/final800k_100ep_attempt2_current_base_only.json` | 800k | finalization_failed |

### DPPO baseline

| Canonical identity | Artifact path | Configuration / budget | Status |
|---|---|---|---|
| `hopper_s1_dppo_800k` | `logs/dppo/hopper_s1_dppo_800k` | Original DPPO PPO diffusion MLP; Hopper-medium-v2; action chunk 4; DDIM 5; frozen pretrained diffusion source; 10 envs; 800k training environment-action steps; seed 1; final `checkpoint/state_39.pt`; `result.pkl` last step 800000 | training-complete |
| `hopper_s2_dppo_800k` | `logs/dppo/hopper_s2_dppo_800k` | Same DPPO protocol; Hopper-medium-v2; 10 envs; seed 2; 40 iterations / 800k environment-action steps; final `checkpoint/state_39.pt`; 40 result records, last step 800000 | training-complete |
| `halfcheetah_s1_dppo_800k` | `logs/dppo/halfcheetah_s1_dppo_800k` | Same DPPO protocol; HalfCheetah-medium-v2; 10 envs; seed 1; 40 iterations / 800k environment-action steps; final `checkpoint/state_39.pt`; 40 result records, last step 800000 | training-complete |
| `walker_s1_dppo_800k` | `logs/dppo/walker_s1_dppo_800k` | Same DPPO protocol; Walker2d-medium-v2; 10 envs; seed 1; 40 iterations / 800k environment-action steps; final `checkpoint/state_39.pt`; 40 result records, last step 800000 | training-complete |
| `hopper_s1_dppo_3200k` | `logs/dppo/hopper_s1_dppo_3200k` | Fresh original DPPO PPO diffusion MLP; Hopper-medium-v2; action chunk 4; DDIM 5; frozen pretrained diffusion source; 10 envs; 160 iterations = 3.2M primitive environment-action steps; seed 1 | training-complete; 160 records, final step 3.2M, `checkpoint/state_159.pt`; all recorded numeric metrics finite. Attempt 2 is valid; attempts 0/1 exited before training because of the wrong Python environment and relative Hydra config path |
| `halfcheetah_s1_dppo_3200k` | `logs/dppo/halfcheetah_s1_dppo_3200k` | Same strict 3.2M-primitive-step DPPO protocol; HalfCheetah-medium-v2; seed 1 | training-complete; 160 records, final step 3.2M, `checkpoint/state_159.pt`; finite metrics |
| `walker_s1_dppo_3200k` | `logs/dppo/walker_s1_dppo_3200k` | Same strict 3.2M-primitive-step DPPO protocol; Walker2d-medium-v2; seed 1 | training-complete; 160 records, final step 3.2M, `checkpoint/state_159.pt`; finite metrics |

The DPPO launch used the repository's Hopper fine-tuning config with two
necessary legacy fixes: a fixed `EtaFixed` module for DDIM sampling (the YAML
omits it although the model requires it), and removal of an explicit
`pdb.set_trace()` left in the PPO training loop. The run uses `wandb=null` and
writes its resolved Hydra config under the artifact directory. Its step budget
is not directly interchangeable with P6 chunk transitions: DPPO counts
`n_envs * act_steps` for each training rollout.

#### DPPO seed-1 trajectory evaluation (2026-09-14)

Each saved DPPO state was evaluated with the stock DPPO deterministic evaluator
using 100 parallel environments for 250 action-chunk steps. `0` is the frozen
pretrained policy; the saved `state_0`, `state_10`, `state_20`, `state_30`, and
`state_39` checkpoints correspond to 20k, 220k, 420k, 620k, and 800k
environment-action steps respectively. Hopper and Walker naturally terminate
and reset within the collection window, so their completed-episode counts are
shown rather than falsely calling every point “100 episodes”. Scores use the
environment's D4RL normalization. This evaluator does not emit P6 early-fall
fields.

| Task, seed 1 | 0 | 20k | 220k | 420k | 620k | 800k |
|---|---:|---:|---:|---:|---:|---:|
| Hopper D4RL (episodes) | 44.68 (170) | 43.64 (179) | 43.87 (174) | 43.86 (170) | 42.82 (177) | 44.06 (180) |
| HalfCheetah D4RL (episodes) | 36.81 (100) | 37.30 (100) | 36.94 (100) | 36.83 (100) | 37.30 (100) | 37.82 (100) |
| Walker2d D4RL (episodes) | 57.62 (106) | 59.10 (102) | 59.44 (105) | 61.86 (104) | 58.24 (104) | 63.98 (103) |

Evaluation outputs are under `logs/dppo/evaluations/{run_name}/step_*`; each
directory contains the stock evaluator's `result.npz` with raw return and
completed-episode count. Hopper's initial result predates the unified suffix
and is `step_000000_100ep`; all other entries use `step_{transition}_n100`.

### DIPO baseline

These runs use the DIPO implementation and locomotion configs distributed in
the official DPPO repository. They share the frozen pretrained diffusion
checkpoint, D4RL environment, action chunk 4, DDIM-5 sampler, seed, environment
count, and 3.2M primitive interaction budget with the strict DPPO comparison.
DIPO-specific replay, critic warm-up, action-gradient, and update-ratio settings
remain at the method's supplied defaults; compute is therefore not artificially
equalized across algorithms.

| Canonical identity | Artifact path | Configuration / budget | Status |
|---|---|---|---|
| `hopper_s1_dipo_3200k` | `logs/dipo/hopper_s1_dipo_3200k` | Hopper-medium-v2; seed 1; 10 envs; 160 iterations = 3.2M primitive environment-action steps | training-complete; 160 records, final step 3.2M, `checkpoint/state_159.pt`; finite metrics |
| `halfcheetah_s1_dipo_3200k` | `logs/dipo/halfcheetah_s1_dipo_3200k` | HalfCheetah-medium-v2; same matched protocol | training-complete; 160 records, final step 3.2M, `checkpoint/state_159.pt`; finite metrics |
| `walker_s1_dipo_3200k` | `logs/dipo/walker_s1_dipo_3200k` | Walker2d-medium-v2; same matched protocol | training-complete; 160 records, final step 3.2M, `checkpoint/state_159.pt`; finite metrics |

Policy Decorator's official code is present at `/home/mrf/policy_decorator`
and tracks `tongzhoumu/policy_decorator`, but its released environments are
ManiSkill and Adroit rather than D4RL locomotion. A locomotion result therefore
requires an explicitly labeled matched adaptation.

### Policy Decorator (matched) baseline

This is an explicit locomotion adaptation, not an official task configuration.
It freezes the same task-specific pretrained diffusion policy, uses action
chunks of four, and trains a sequence-valued SAC residual for 800k chunk
decisions (3.2M primitive actions). The residual bound is fixed at
`alpha=0.1` to match VS-Hier's maximum correction magnitude. Progressive
exploration follows Policy Decorator semantics: during the first 100k chunk
decisions, `epsilon=t/100k` is the Bernoulli probability of applying the full
bounded residual; it is **not** an amplitude ramp. Residual-SAC / classical
Residual RL was explicitly cancelled and is not part of this campaign.

| Canonical identity | Artifact path | Configuration / budget | Status |
|---|---|---|---|
| `hopper_s1_policy_decorator_matched_800kchunks_3200kprimitive` | `logs/policy_decorator/hopper_s1_policy_decorator_matched_800kchunks_3200kprimitive` | Hopper-medium-v2; seed 1; frozen DDIM-5 base; 10 envs; fixed alpha 0.1; epsilon ramp over 100k chunks; SAC UTD 0.25 | training-complete; 10ep preliminary evaluation at 800k: raw 2178.21 ± 564.65, D4RL 67.55, early-fall 90%; `evaluations/chunk_000000800000_n10_full.json`; 100ep final evaluation pending |
| `halfcheetah_s1_policy_decorator_matched_800kchunks_3200kprimitive` | `logs/policy_decorator/halfcheetah_s1_policy_decorator_matched_800kchunks_3200kprimitive` | HalfCheetah-medium-v2; otherwise identical matched protocol | training-complete; 10ep preliminary evaluation at 800k: raw 4504.81 ± 319.13, D4RL 38.54, early-fall 0%; `evaluations/chunk_000000800000_n10_full.json`; 100ep final evaluation pending |
| `walker_s1_policy_decorator_matched_800kchunks_3200kprimitive` | `logs/policy_decorator/walker_s1_policy_decorator_matched_800kchunks_3200kprimitive` | Walker2d-medium-v2; otherwise identical matched protocol | training-complete; 10ep preliminary evaluation at 800k: raw 3727.47 ± 414.11, D4RL 81.16, early-fall 30%; `evaluations/chunk_000000800000_n10_full.json`; 100ep final evaluation pending |

The following fixed-seed 10-episode intermediate evaluations were requested
explicitly (so they intentionally include checkpoints below the normal 500k
evaluation threshold). Each cell is `D4RL / early-fall`; all 27 outputs have
exactly 10 episodes and finite metrics. These are learning-curve diagnostics,
not final-score estimates.

| Task | 0 | 100k | 200k | 300k | 400k | 500k | 600k | 700k | 800k |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Hopper | 46.73 / 100% | 39.67 / 100% | 42.47 / 100% | 46.29 / 100% | 64.33 / 80% | 62.09 / 90% | 74.84 / 80% | 65.34 / 90% | 67.55 / 90% |
| HalfCheetah | 33.71 / 0% | 32.61 / 0% | 35.30 / 0% | 35.78 / 0% | 37.45 / 0% | 38.63 / 0% | 39.94 / 0% | 40.27 / 0% | 38.54 / 0% |
| Walker2d | 62.41 / 70% | 56.09 / 80% | 58.60 / 90% | 70.85 / 20% | 64.12 / 50% | 68.27 / 50% | 81.56 / 10% | 69.30 / 40% | 81.16 / 30% |

Artifact pattern for the intermediate files is
`logs/policy_decorator/{task}_s1_policy_decorator_matched_800kchunks_3200kprimitive/evaluations/chunk_000000{transition}_n10_full.json`.
The zero-step frozen-base evaluations use
`evaluations/chunk_000000000000_n10_initial_base.json`; their residual-use
probability is exactly zero and the evaluator records no residual checkpoint.

#### Unified shared frozen-base starting point (2026-09-15)

The main locomotion learning-curve figure uses one canonical zero-step
evaluation per task for all five methods. Each evaluation runs the shared
task-specific frozen diffusion checkpoint with deterministic DDIM-5 for 100
episodes, environment seeds 10000--10099, and policy seeds 20000--20099. No
residual checkpoint is loaded and the residual-use probability is zero. This
shared point replaces method-specific zero-step estimates only; every positive-
step historical evaluation remains unchanged.

| Task | Raw reward mean ± std | D4RL | Early fall |
|---|---:|---:|---:|
| Hopper | 1424.40 ± 273.70 | 44.389 | 100% |
| Walker2d | 2622.26 ± 945.52 | 57.086 | 76% |
| HalfCheetah | 4138.67 ± 544.72 | 35.592 | 0% |

The machine-readable artifacts are
`logs/policy_decorator/{task}_s1_policy_decorator_matched_800kchunks_3200kprimitive/evaluations/chunk_000000000000_n100_shared_frozen_base.json`,
where `{task}` is `hopper`, `walker`, or `halfcheetah`.

For the current main-figure rendering, the VS-Hier Hopper 50k and
HalfCheetah 250k single-seed diagnostics are omitted from the displayed
trajectory at the user's request. Every other VS-Hier checkpoint, including
all Walker2d checkpoints, retains its original plotting cadence. The omitted
measurements remain in their source records and must not be reported as
missing evaluations.

Implementation entry points are `train_policy_decorator_matched.py` and
`policy_decorator_matched.py`; focused contract tests are in
`tests/test_policy_decorator_matched.py`. A real Hopper smoke completed 200
chunk decisions including SAC updates before the formal runs were launched.

### Resource observation

On 2026-09-10, two seed-3 BASE runs plus the new Walker full/control pair
were simultaneously loaded. GPU use was about 7.1/24.6 GiB at 55 C, but host
available RAM fell to about 8.6 GiB and the 2 GiB swap filled. The local
constraint is replay-host memory, not VRAM. Four active runs are an observed
configuration, not evidence that six are safe. Before adding a fifth process,
take one focused host-memory/GPU reading; do not co-schedule matched DSRL
continuations with this four-run wave.

## Robomimic Can matched extension (2026-09-16)

Both seed-1 runs use the same official frozen Can diffusion checkpoint, the
same normalization file, four parallel environments, action chunk 4, and the
same seeded shared prefill. Their common initial target is 300k chunk decisions
= 1.2M nominal primitive actions. Robomimic `success_rate` is the primary
metric; D4RL score and locomotion early-fall are not applicable.

| Canonical identity | Artifact path | Configuration / budget | Status |
|---|---|---|---|
| `can_s1_noclip_k4_100k_300k_full` | `logs/p6/can_s1_noclip_k4_100k_300k_full` | VS-Hier; Gaussian QW teacher; K=4; NoClip; 100k BASE + 200k additive RES; BASE updates preserved during RES and residual UTD=4; beta 0.1; seed 1 | running; launcher and learner live, shared prefill `verified_and_loaded`, optimizer counters advancing |
| `can_s1_matched_dsrl_0k_300k_control` | `logs/p6/fresh_frozen_ddim_can_dsrl_na_control_seed1_300000chunks` | matched DSRL base pathway; UTD=20; no residual/QA_joint; same seed, environment, frozen policy, prefill, and 300k interaction budget | running; launcher and learner live, shared prefill `verified_and_loaded`, optimizer counters advancing |

Frozen source:
`dppo/log/robomimic-pretrain/can/can_pre_diffusion_mlp_ta4_td20/2024-06-28_13-29-54/checkpoint/state_5000.pt`
(SHA-256 `61851045e6b516807826e3bda4270c9e4a086023f2dd85af00eadb59d9b98a1b`).
Normalization source:
`dppo/log/robomimic/can/normalization.npz`
(SHA-256 `a4bb04c498625bfad0ee6faae21674c0a06879cd9f9614bbd2da41a7d0dc1c1a`).
The pair shares
`logs/p6-prefill/can_s1_env3001_policy4001_nenv4_tagged_v1.npz`.

The 300k boundary is a predeclared convergence check rather than a forced
publication endpoint. Evaluate both 200k and 300k with the same 100 fixed-seed
episodes. If either method gains at least five success-rate percentage points
from 200k to 300k and its online success trace has not flattened, resume both
methods without reset to the same 400k boundary; otherwise accept 300k as the
shared endpoint. Never extend only the currently weaker or stronger method.

### Robomimic Square pair

| Canonical identity | Artifact path | Configuration / budget | Status |
|---|---|---|---|
| `square_s1_noclip_k4_200k_500k_full` | `logs/p6/square_s1_noclip_k4_200k_500k_full` | VS-Hier; Gaussian QW teacher; K=4; NoClip; 200k BASE + 300k additive RES; BASE preserved and residual UTD=4; 500k chunks = 2.0M nominal primitive actions | stopped by user for remote migration at last observed 16.4k; launcher exit -15; no 100k resume bundle and not result evidence |
| `square_s1_matched_dsrl_0k_500k_control` | `logs/p6/fresh_frozen_ddim_square_dsrl_na_control_seed1_500000chunks` | matched DSRL; UTD=20; same seed, environment, frozen policy, prefill, and 500k interaction budget | stopped by user for remote migration at last observed 44.8k; launcher exit -15; no 100k resume bundle and not result evidence |
| `square_s2_noclip_k4_200k_500k_full` | `logs/p6/square_s2_noclip_k4_200k_500k_full` | Same VS-Hier Square protocol; independent seed 2 environment/policy seeds and shared seed-2 prefill | stopped by user for remote migration at last observed 2.0k; launcher exit -15; no 100k resume bundle and not result evidence |
| `square_s2_matched_dsrl_0k_500k_control` | `logs/p6/fresh_frozen_ddim_square_dsrl_na_control_seed2_500000chunks` | Same matched DSRL Square protocol; paired with seed-2 VS-Hier through identical frozen source, interaction budget, environment seeds, and prefill | stopped by user for remote migration at last observed 6.8k; launcher exit -15; no 100k resume bundle and not result evidence |

Square frozen source:
`dppo/log/robomimic-pretrain/square/square_pre_diffusion_mlp_ta4_td100_ddim-100steps/2025-04-11_19-13-26_44/checkpoint/state_3000.pt`
(SHA-256 `e4b391aedc33e8a94bb5cbe25fc737159b64963f51bec22c5e3e104dc1e32373`).
Normalization SHA-256 is
`68ec0abfd989d5f0121f0e9a1dd074b49f859c167ff941c70d2eff8006074ff3`.
The pair shares
`logs/p6-prefill/square_s1_env3001_policy4001_nenv4_tagged_v1.npz`.
Use 400k/500k fixed-seed 100-episode success evaluations for convergence; an
unflattened gain of at least five points extends both methods equally to 600k.

### D3IL Avoid-M1 pair

Avoid-M1 is mode `d56_r12`; M2 and M3 are different target-mode variants, not
additional seeds. The environment dependency is the official
`allenzren/d3il` fork at commit
`139dbf9b114d0f6192e5433ebcffeb0fc17098f4`. Primary evaluation reports
success rate (reward 2 means desired mode plus finish) and return.

| Canonical identity | Artifact path | Configuration / budget | Status |
|---|---|---|---|
| `avoidm1_s1_noclip_k4_25k_100k_full` | `logs/p6/avoidm1_s1_noclip_k4_25k_100k_full` | VS-Hier; Gaussian QW teacher; K=4; NoClip; 25k BASE + 75k additive RES; BASE preserved and residual UTD=4; 100k chunks = 0.4M nominal primitive actions | running; shared prefill verified/loaded; transition and optimizer progress observed |
| `avoidm1_s1_matched_dsrl_0k_100k_control` | `logs/p6/fresh_frozen_ddim_avoidm1_dsrl_na_control_seed1_100000chunks` | matched DSRL; UTD=20; same seed, Avoid-M1 environment, frozen policy, prefill, and 100k interaction budget | running; shared prefill verified/loaded; transition and optimizer progress observed |

Avoid-M1 frozen source:
`dppo/log/d3il-pretrain/m1/avoid_d56_r12_pre_diffusion_mlp_ta4_td20/2024-07-06_22-50-07/checkpoint/state_10000.pt`
(SHA-256 `7a420985fd213f79ac03b13f62c3e75c5ae33e6f5c82343e75ad8d4ad0ba230b`).
Normalization SHA-256 is
`24d0c2b650fe26832e0de0474c06d2a989c8a989b0233d039ff1e6b1a52cdc68`.
The pair shares
`logs/p6-prefill/avoidm1_s1_env3001_policy4001_nenv4_tagged_v1.npz`.
Use 50k/100k fixed-seed 100-episode success evaluations for convergence; an
unflattened gain of at least five points extends both methods equally to 150k.

## Seed-3 shared BASE sources

These runs trained the task-specific seed-3 BASE state from scratch to the
300k branch boundary. They are not final base-control results. Their completed
300k bundles are the shared source recorded by the corresponding child runs.

| Canonical identity | Method | Target | Status |
|---|---|---:|---|
| `hopper_s3_noclip_k4_0k_300k_basecontrol` | Gaussian teacher, K=4, B=256, NoClip, BASE only | 300k | interrupted as planned at exact 300k (exit 75); source used by the completed Hopper s3 pair |
| `walker_s3_noclip_k4_0k_300k_basecontrol` | Gaussian teacher, K=4, B=256, NoClip, BASE only | 300k | interrupted as planned at exact 300k (exit 75); no child pair is registered as complete here |

## Verified evaluation evidence

### 实验章节与消融规划（2026-09-11 更新，非启动指令）

#### 本次修复与证据隔离

Hopper s3 full 的 attempt2 虽配置300k分岔，但300k bundle序列化日程仍为500k BASE/2M RES。该次300k后的进度不能计为正确full训练。9月11日停止该进程，从原300k生成 `resume/chunk_000000300000_schedule300k_v1`，仅纠正日程为300k BASE/2.2M RES（实际stop800k）；参数和optimizer张量逐项相等，replay硬链接、RNG继承，原bundle及旧日志保留。修复报告：`logs/p6/hopper_s3_noclip_k4_300k_800k_full/schedule300k_repair_report.json`。后续取最新attempt，不混入错误尝试的同step TensorBoard曲线。RES profile为每train call4次residual更新，前50k beta hold沿用原配置，不是额外重新warm-up。

#### 论文主线：按问题推进，不按实验发生顺序堆表

| 顺序／图表 | 要回答的问题 | 对照与读数 | 证据要求／补充优先级 |
|---|---|---|---|
| 1 主图：三任务三列 | 最终方法有没有更强？ | matched DSRL、独立basecontrol、VS-Hier full；统一800k，500–800k学习曲线；最终100ep | P0：补齐当前版本3seed缺项；远端结果先收evaluation/config/manifest，不重复训练 |
| 2 机制图：分离信用 | full变好是否掩盖base变差？分离是否缓解？ | 同300k源、同Gaussian/K4/NoClip/BASE更新/RES4的 separated vs joint-credit；同时画各自base view和full | P1：先Hopper当前版本代表seed，再扩展另外2seed；旧H4不能直接声称严格因果 |
| 3 组件表：互补性 | 只有residual是否足够？ | 已有basecontrol代表无residual；新增固定300k noise actor的residual-only与full比较 | P2：固定noise/alpha，明确critic是否继续训练；继承同起点、同residual预算，不能与旧两critic冻结实验混用 |
| 4 效率表／曲线 | 有限交互预算下何时收益出现，代价多少？ | 已有500/600/700/800k；elapsed active time、env/chunk、更新次数、queries；必要时单卡吞吐 | P0数据整理；不由相同chunks宣称相同算力，也不把并发墙钟当独占吞吐 |
| 附录参数 | 配方是否稳健？ | source、gradient clip、K/B、UTD、microbatch | 优先复用P1历史产物，标预算、版本、混杂；不再扫大网格 |

主图full−独立control是整个新增residual训练分支的效果（也包含采集分布变化）；full−同checkpoint base view仅是推理时启用residual的效果。两者必须分开。训练seed才是重复实验单位；100episodes不替代3seed。报告训练seed均值/标准差、每seed结果，并固定early-fall定义；Hopper/Walker报告失败率，HC主要看return，不强造摔倒叙事。

#### 对当前abstract的修订约束

- “all nine runs”及“Hopper58–68%→0–4%”对应下面H3的旧2.5M cotrain、同checkpoint residual-off对照，不是当前800k独立control结果。不能在当前方法摘要中无版本限定沿用。
- “base下降10分、full几乎不变”来自H4与H3横向比较；500k起点评估不同，尚不足以写“directly证明分离的因果效果”。可写为旧设置的masking observation，待P1同起点实验支撑机制主张。
- 分离改变的是优化信用来源，不保证base性能单调不降；不要把实现上的梯度/teacher分离写成性能保证。
- 用户新找到旧两critic100k点：暂登记“用户确认存在，路径待提供”。重新评估可扩充H1动机图，但不能替代当前joint-credit受控消融，也不能与10k不同初始化结果直接拼曲线。

#### 双4090服务器安排（计划，未启动）

两张卡优先跑独立作业，不把48GB总显存当单进程共享池，也不为这些小网络上DDP。第一批补主结果缺失seed／control，并回收远端已完成结果；每卡先一条，按实测吞吐再增加并发。第二批做Hopper同起点joint-credit，优先复用匹配full；第三批在机制证据不足时做residual-only。256GB RAM缓解多replay内存压力，但并发上限仍由CPU环境步进和吞吐决定。各机器独占run目录，以Git提交/配置、manifest和evaluation文件同步，不共享正在写的replay或checkpoint。当前机器保留三条在跑任务及Walkercontrol后继队列。

VLA若仍主张少数据／少rollout：必须有VLA固定预训练起点的交互预算曲线及至少一个同起点机制对照；locomotion支持算法动机，不能替代VLA样本效率证据。时间紧则先收窄论文主张，优先主结果+信用分离，不以补几十个调参实验代替核心验证。

正文围绕三个问题组织：最终方法是否优于 matched DSRL；在同一 BASE 起点上加入 residual 是否优于独立 base-control；分离 BASE/full 价值信用是否有独立作用。参数筛选主要放附录。旧版本结果保留版本标记，不直接充当当前方法的受控消融。

| 消融或诊断 | 本轮找到的证据／待核实项 | 建议位置与可支持结论 |
|---|---|---|
| QW source：current-actor vs Gaussian，K1 | `logs/p6/base_diag_qw_{current,gaussian}_k1_pilot250k_*` 均有 evaluation；NoClip 对应 seed1/2 目录也有 evaluation | 附录完整表；正文可简述 teacher 支持范围的作用。逐项核对配置后才视为 source 单变量比较 |
| Clip vs NoClip | 上述 K1 同 source 目录形成比较候选 | 附录；这里需明确是 noise-actor gradient clipping，不能表述为移除全部 action/decoder bounds |
| Gaussian multi-w | K4-B256、K8-B256、K8-B32、K16-B128 正式目录均有 evaluation | 附录参数敏感性；query/update 分别1024、2048、256、2048。K8-B256 与 K16-B128 是现有等 query 候选；尚不能据此声称同状态多 w 优于等 query 的 K1 |
| K64-B256 | 会话有提议，本轮未发现正式运行目录 | 未确认执行，不能列作已完成 |
| Full vs 独立 BASE continuation | 当前 Walker 300k→800k pair 与旧500k pair分开记录 | 正文核心；同起点、同基础更新设置，full 增加 residual 学习。旧500k结果作附录诊断 |
| full checkpoint 关闭 residual 评估 | 已有 `current_base_only` 和 full 评估 | 附录部署诊断；只能回答此模型当下 residual 的作用，不能代替独立 base-control |
| joint-credit vs separated-credit | `logs/p6/fresh_frozen_ddim_joint_credit_dsrl_na_rfs_hier_seed1_2500000chunks` 目录存在；配置 `p6_hopper_fresh_2p5m_joint_credit.yaml` 把 RES 阶段 QW teacher 切到 QA_joint | 正文机制候选；需核实完成预算与版本。该配置保留 QA_base，不能叫严格“删除一个 critic” |
| RES UTD=1 vs4、BASE 更新保持 | 有旧 cotrain/additive 系列与当前配置；本轮未做逐对匹配 | 附录训练组织；若其他设置不同，只能当迭代证据，不能单独归因 UTD |
| 冻结 latent/base 后只学 residual、residual LR | `20260729_frozen_noise_init5m_seed1_10k_ablation`、`20260729_control_init5m_seed1_10k_frozen_ablation`、`20260729_frozen_lr*` 目录存在 | 附录历史设计证据，待核验冻结对象、架构和评估；不等同当前300k起点消融 |
| 旧两 critic 的 BASE/RES 组件诊断 | 已定位 July29 Hopper 10k run、配置、模型与历史诊断报告，详见下方 H1；25-episode诊断原始JSON仍未定位 | 历史动机实验；不能直接充当当前三critic的训练消融 |
| fixed-state gradient / matched mirror / ranking probe | 会话提出过，当前登记未提供原始结果路径 | 待定位；属于机制诊断，排名与梯度相关性不能证明组件必要或不必要 |

正文主图保持三任务三列，每列仅 matched DSRL、独立 base-control、VS-Hier full 三条曲线。报告训练 seed 均值及变异；每 checkpoint 的100 episodes用于评估策略，不能当作100个训练 seed。所有方法横轴口径一致，并附实际 primitive steps、optimizer updates、teacher queries 和训练耗时，不能由相同 chunks 推导相同算力。

最小补充优先级：先完成当前主结果与独立 control；其次在同一当前 BASE checkpoint 上比较 full 与 joint-credit，以检验信用分离（不是强行改成 online QA）；若要主张 BASE 与 RES 互补，再增加冻结 latent actor 的 residual-only 分支，明确 QA/QW/alpha 是否冻结。先使用一个代表任务检验实现和效果，再按论文主张决定扩展。不要为附录参数表重跑全部网格。

低数据／少 rollout 的论文主张需要 VLA 自身的证据：固定预训练起点、固定示范数据量，绘制在线交互预算曲线，比较 full、独立 base-control 及外部 baseline；若主张数据效率，再增加少量示范数据档位。机制结果应至少在代表性 VLA 设置验证一次。Locomotion 消融本身不能证明 VLA 少 rollout 的原因。

现阶段不新增 K64、SVGD、强行 online-QA 或全量超参数网格。若论文不声称 same-state multi-w 的独立理论优势，则无需新增昂贵的 K1-B2048 等 query 消融；若保留该主张，则必须补充相应比较。

### 历史实验与消融证据台账（论文引用入口）

本节汇总已检索的历史材料，不改变训练计划。原始 evaluation/config 是结果依据；归档报告与对话缓存只用于定位、保留历史记载，不能自动升级为已复核实验事实。以下“报告记载”数字保留供写作检索，正式引用前需原始产物或用户确认。各实验的 raw return、D4RL score、训练 seed 和评估 episodes 不可混用。

| 编号 | 实验性质 | 版本／用途 | 证据与限制 |
|---|---|---|---|
| H1 | 历史正式训练 + inference-time组件诊断 | 旧 QA/QM joint-credit，Hopper 5M初始化后10k；动机图候选 | run/config/model存在；25ep和分段诊断数字来自历史报告，原始诊断JSON待定位 |
| H2 | 历史训练组件消融 | 冻结 noise actor，仅训练 residual；附录 | frozen/control的100ep final JSON存在；不是当前300k分岔实验 |
| H3 | 历史正式主实验 + residual-off评估 | 旧三critic cotrain，三任务×3seed，2.5M；附录 | 九组final JSON已核对；base列均不是独立base-control |
| H4 | 信用来源训练消融候选 | 旧三critic joint-credit Hopper s1；机制候选 | final JSON已核对，但与H3的500k评估不同，尚不能称严格配对单变量消融 |
| H5 | 历史架构诊断 | 旧双头，三任务800k、同checkpoint residual-off评估 | 用户提供的P6口径100ep重评估；Hopper显示强掩盖现象，不能据此归因 shared joint credit |
| M1 | 当前三critic joint-credit 机制对照 | Hopper / Walker 800k完成；HalfCheetah仍在运行 | 用户提供的P6口径结果；在核对共同起点、其余配置与独立control前，不称严格单变量因果对照 |
| P1 | 超参数／实现选择 | source、clip、K/B、UTD；附录 | 见上方规划表及下方路径索引；不同版本不能合并归因 |
| D1 | 历史替代方案／工程诊断 | per-step QA、PPO、ResiP、GAE、RNG | 归档入口已定位；不是当前方法消融，也不是新增主结果 |

#### H1：旧两价值角色 Hopper——base退化被residual补偿

实验目录：`logs/p6/20260729_hier_init5m_seed1_10k_wiring`。身份依据为 `run_manifest.json`、`attempts/0001/resolved_config.json` 与 `checkpoints/final_model.zip`；5k处恢复过训练，不能描述为无中断运行。初始模型为 `logs/gym-dsrl/gym_hopper_dsrl_2026-07-23_21-21-02_1/2026-07-23_21-21-02_1/checkpoint/ft_policy_5000000_steps.zip`。

历史说明：[P6_10k_actor_residual_diagnosis.md](archive/rfs_hier_v1_legacy/handoffs/P6_10k_actor_residual_diagnosis.md)。旧结构为 joint action QA 与 QM(s,w,residual)，不是当前 `QA_base → QW_base` 加 `QA_joint` 的分离结构。配置中的 twin-head数量也不能直接当作语义critic角色数量。

下表是报告记载的25个固定评估seed（stochastic），单位为 **raw return**：

| 策略 | Return | Early fall |
|---|---:|---:|
| 初始5M DSRL | 3065.42 | 12% |
| 10k模型关闭residual | 2339.84 | 92% |
| 初始noise actor + 训练后residual（hybrid） | 3006.04 | 0% |
| 10k full | 3074.23 | 0% |

报告内同为25ep的deterministic结果依次为3143.35、2493.19、2995.00、3034.61，early fall依次0%、96%、0%、0%。另一个**10ep deterministic**分段诊断如下，不能拼接成25ep曲线：

| 训练chunks | Base return | Base early fall | Full return | Full early fall |
|---|---:|---:|---:|---:|
| 2k | 3021.53 | 20% | 2958.43 | 0% |
| 4k | 2976.32 | 30% | 2970.62 | 0% |
| 6k | 3086.30 | 30% | 2994.91 | 0% |
| 8k | 2399.85 | 90% | 3008.63 | 0% |
| 10k | 2453.43 | 90% | 3035.97 | 0% |

可写的观察：该历史诊断中base下降725.58，开启residual增加734.39，full相对初始化仅增加8.81；full稳定并不意味着base独立能力稳定。hybrid支持组件适配的解释，但不能单独证明退化由residual造成，也不能证明三个critic必要。报告中的QM误差不能单凭整体RMSE推导局部ranking错误。

证据缺口：目录自动 `evaluations/final.json` 只有2 episodes，**不是上述25ep表的来源**。缓存线索为 `/home/mrf/.codex/sessions/2026/07/30/rollout-2026-07-30T22-41-46-019fb379-6a15-7ac3-bb12-f527791c9cd0.jsonl`；缓存转述不得当作新的评估。正文若使用，标为旧架构motivating example，不放入当前方法主比较。

#### H2：冻结noise actor的历史训练消融

说明文档：[P6_frozen_noise_10k_result.md](archive/rfs_hier_v1_legacy/handoffs/P6_frozen_noise_10k_result.md)。两条训练结果各自的 `evaluations/final.json` 为100ep stochastic；下表单位raw return。

| 设置 | Return | Early fall | 来源 |
|---|---:|---:|---|
| 初始5M／zero-residual参考 | 3093.80 | 12% | 历史报告记载；参考评估文件待定位 |
| 继续DSRL 10k | 3095.80 | 12% | `logs/p6/20260729_control_init5m_seed1_10k_frozen_ablation/evaluations/final.json` |
| 冻结noise + residual训练10k | 3052.36 | 0% | `logs/p6/20260729_frozen_noise_init5m_seed1_10k_ablation/evaluations/final.json` |

报告记载冻结actor哈希不变、actor optimizer steps=0。它支持“该设置减少early fall”，不支持平均回报显著提升：报告给出的residual−continued DSRL配对差为−43.44，区间[-83.33,2.44]。LR扫描目录 `logs/p6/20260729_frozen_lr1e4_seed1_10k_scan`、`logs/p6/20260729_frozen_lr3e5_seed1_10k_scan` 有评估，尚未在本台账提取完整表，不声明优劣。

#### H3：旧三critic cotrain三任务九组结果

历史说明：[COTRAIN_2P5M_RESULTS.md](archive/rfs_hier_v1_legacy/COTRAIN_2P5M_RESULTS.md)。其旧baseline完成状态等叙述已过时，不覆盖本台账。该系列为旧500k BASE + 2M RES安排，不是当前Gaussian-K4-NoClip/UTDres4。

下表从各run的 `evaluations/final_current_base_only.json` 与 `evaluations/final_current_full_hierarchy.json` 核对：100ep stochastic，**D4RL normalized score**。base是同一full checkpoint关闭residual，不是单独训练。Hopper括号为early fall。

| 任务／训练seed | Base view | Full | Full−base |
|---|---:|---:|---:|
| Hopper s1 | 87.54（58%） | 96.12（4%） | +8.58 |
| Hopper s2 | 84.22（68%） | 95.60（0%） | +11.38 |
| Hopper s3 | 85.84（61%） | 94.30（0%） | +8.46 |
| HalfCheetah s1 | 52.63 | 57.47 | +4.84 |
| HalfCheetah s2 | 49.48 | 54.82 | +5.34 |
| HalfCheetah s3 | 55.44 | 58.93 | +3.49 |
| Walker s1 | 88.65 | 91.12 | +2.47 |
| Walker s2 | 88.98 | 91.70 | +2.72 |
| Walker s3 | 87.90 | 91.41 | +3.50 |

完整run路径模板（`{s}`分别取1、2、3）：

- Hopper：`logs/p6/fresh_frozen_ddim_cotrain_dsrl_na_rfs_hier_seed{s}_2500000chunks`
- HalfCheetah：`logs/p6/fresh_frozen_ddim_cotrain_halfcheetah-medium-v2_dsrl_na_rfs_hier_seed{s}_2500000chunks`
- Walker：`logs/p6/fresh_frozen_ddim_cotrain_walker2d-medium-v2_dsrl_na_rfs_hier_seed{s}_2500000chunks`

这里证明的是旧模型的residual开启收益，不是九组base collapse。500k时base记录分别为Hopper82.21/80.54/81.66、HC49.02/47.46/50.03、Walker86.84/87.23/86.02，均低于最终base；但500k只有10ep，最终100ep，不能据此做精确配对改善检验。s1目录中的 `evaluations/validation_beta_sweep_v1/` 是推理期β扫描入口，不能称为不同β的重新训练消融。

#### H4：旧JointCredit信用来源比较

Run：`logs/p6/fresh_frozen_ddim_joint_credit_dsrl_na_rfs_hier_seed1_2500000chunks`；配置：[p6_hopper_fresh_2p5m_joint_credit.yaml](../cfg/gym/p6_hopper_fresh_2p5m_joint_credit.yaml)。RES阶段 `qw_teacher_joint_credit: true` 将teacher切到QA_joint，仍保留QA_base；因此不是“删去一个critic”，也不是H1的旧QM结构。

| Hopper s1 @2.5M，100ep | Base view | Full | Base early fall | Full early fall |
|---|---:|---:|---:|---:|
| JointCredit | 77.55 | 95.93 | 85% | 1% |
| H3 separated cotrain | 87.54 | 96.12 | 58% | 4% |

数值来源是两个run各自的 `evaluations/final_current_base_only.json`、`final_current_full_hierarchy.json`。JointCredit的full−base为18.38；相对H3 base低9.99，full低约0.19。结果符合“full掩盖base能力差异”的解释，但**目前不能声称严格单变量因果证明**：两run在500k的base评估分别87.80与82.21，尚未审计共同模型/RNG/其余配置一致性；500k10ep与最终100ep也不是同协议纵向配对。历史对话中的“实锤”“BASE完全相同”等强表述不沿用。

#### H5：旧双头架构的三任务800k重评估

这是旧双头架构的诊断，不是当前 VS-Hier 主结果，也不是独立
`basecontrol`。用户提供的重评估使用 P6 口径：每个 policy view 完整
100 episodes、随机策略、环境种子10000--10099、独立策略种子20000--20099，
action chunk 在终止时立即结束。待将机器上的具体 evaluation 路径登记后，以下
表可作为可追溯的历史诊断；在此之前不得把它升级为当前主实验的机器证据。

| 任务，旧双头 @800k | Full D4RL | 同 checkpoint residual-off base view | Full−base | Base-view early fall |
|---|---:|---:|---:|---:|
| Hopper s3 | 94.96 | 66.62 | +28.34 | 97% |
| HalfCheetah s3 | 49.55 | 47.69 | +1.87 | 0% |
| Walker s3 | 97.28 | 95.92 | +1.37 | 0% |

可写的结论仅限于：在这套旧双头架构和该评估协议下，Hopper 明显呈现
“full 被 residual 支撑、base view 已退化”的现象；HalfCheetah 与 Walker 没有
同等幅度的差距。它**不能**单独把 Hopper 的退化归因于 shared joint credit，
因为架构、训练路径和其他旧配置因素并未在该表中隔离。

论文中的临时机制图另采用用户于 2026-09-16 提供的确定性 100-episode
重评估：环境种子固定为 10000--10099，动作采用确定性执行。该协议与上表的
随机策略评估不同，因此单独登记、不得混合求均值；具体 evaluation 产物路径
仍待补录。

| 任务，旧双头 @800k（确定性） | Full D4RL | 同 checkpoint residual-off base view | Full−base | Residual-only D4RL |
|---|---:|---:|---:|---:|
| Hopper | 94.41 ± 0.28 | 67.27 ± 14.55 | +27.14 | 1.51 ± 0.00 |
| HalfCheetah | 49.82 ± 5.07 | 48.34 ± 0.86 | +1.48 | 2.23 ± 0.01 |
| Walker | 97.84 ± 0.43 | 97.03 ± 0.32 | +0.81 | 4.01 ± 0.72 |

`IEEE-conference-VSHS/base_residual_decomposition_data.csv` 使用该确定性旧双头
结果与当前 Table 1 的 VS-Hier base/full 聚合值作端点分解。图中仅展示 base view
与 full−base residual gain；由于两代架构和评估聚合语义不同，不绘制误差棒，
也不把该图表述为严格的 value-separation 单变量消融。论文图使用与主学习曲线
一致的 D4RL-v2 reference scores 逆归一化为 raw episodic reward，并为三个任务
分别设置从零开始的纵轴范围；CSV 同时保留 D4RL 与 raw reward 两种读数。

#### M1：当前三critic Joint-Credit 机制对照

下表与 H4 分开：它属于三critic实现中的 Joint-Credit 变体，而不是旧双头。
同样，`base view` 是从 full checkpoint 在推理时关闭 residual 得到的视图，不是
独立 base-control。Hopper 与 Walker 的800k点为100-episode评估；HalfCheetah
仍在训练，其600k点仅为20-episode过程诊断，不能与前两行作最终比较。

| 任务 / 状态 | Full D4RL | 同 checkpoint base view | Full−base | 评估协议 |
|---|---:|---:|---:|---|
| Hopper 三critic Joint-Credit，完成800k | 97.16 | 92.82 | +4.34 | 100 episodes |
| Walker 三critic Joint-Credit，完成800k | 105.82 | 96.23 | +9.59 | 100 episodes |
| HalfCheetah 三critic Joint-Credit，约645k/800k运行中 | 49.80 | 47.55 | +2.25 | 600k过程点，20 episodes |

这些数值由用户提供，具体 artifact 路径待登记。它们说明 Joint-Credit 变体在
Hopper/Walker 的 full 与 base view 之间仍有差距，但尚不支持 “separated critic
导致该差距更小/更大” 的因果结论：仍需核验它们与 separated 运行是否具有相同
起点、budget、BASE更新、随机状态和其余训练超参数。HalfCheetah 完成800k并做
100-episode评估前，不应进入跨任务最终表。

#### P1 / D1：参数与其他历史材料检索入口

| 项目 | Run或文档入口 | 写作分类／尚缺信息 |
|---|---|---|
| QW Current/Gaussian与NoClip | `logs/p6/base_diag_qw_{current,gaussian}_k1_{noclip_}pilot250k_*`（花括号为路径检索提示） | 参数筛选；逐run核对source/clip/seed和相同预算后提取分数 |
| Gaussian K/B | `logs/p6/base_diag_qw_gaussian_k{4_b256,8_b256,8_b32,16_b128}_noclip_100k_*` | 参数筛选；目录名100k不代表实际停止点，以evaluation chunks为准 |
| BASE支持诊断背景 | [WALKER_BASE_SUPPORT_GATED_PLAN.md](archive/rfs_hier_v1_legacy/WALKER_BASE_SUPPORT_GATED_PLAN.md) | 历史意图，不是完成证据；旧gate不恢复为现行约束 |
| per-step QA | [P6_per_step_qa_10k_result.md](archive/rfs_hier_v1_legacy/handoffs/P6_per_step_qa_10k_result.md) | 历史替代训练方案，需原始结果复核后才能引用数字 |
| stable PPO | [P6_stable_ppo_10k_result.md](archive/rfs_hier_v1_legacy/handoffs/P6_stable_ppo_10k_result.md)；`logs/per_step_residual_ppo/20260731_ppo_stable_init5m_seed1_10k` | 冻结base的替代方案，不是base-collapse证据 |
| ResiP | [P6_resip_2p5m_frozen_10k_result.md](archive/rfs_hier_v1_legacy/handoffs/P6_resip_2p5m_frozen_10k_result.md)；`logs/resip-aligned-10k/dsrl_2p5m_seed1`、`logs/resip-aligned-4k-confirm/dsrl_2p5m_seed{2,3}` | 历史冻结base方案；报告标题与实际4k/10k子实验须区分 |
| horizon/std/GAE | [P6_std_horizon_long_gae_4k_result.md](archive/rfs_hier_v1_legacy/handoffs/P6_std_horizon_long_gae_4k_result.md)；`logs/resip-hopper-long-gae-4k/dsrl_2p5m_seed{1,2,3}` | 历史参数诊断，不并入当前方法消融 |
| RNG与advantage | [P6_random_rng_and_advantage_diagnosis.md](archive/rfs_hier_v1_legacy/handoffs/P6_random_rng_and_advantage_diagnosis.md) | 评估/实现诊断，用于解释协议，不能当算法收益 |

论文使用顺序：正文主结果仅当前版本三任务的matched DSRL、独立base-control、full；H1可作注明旧架构的动机图；H4仅在完成配对审计后考虑机制主图，否则与H2/H3一起放历史诊断附录。source/clip/K/B/UTD放参数附录，D1通常仅保留内部研究记录。尚缺原始证据的表不当作已完成消融；本节没有启动任何补实验。

### 当前及旧Walker边界的100-episode评估

All values below are stochastic, fixed-seed, 100-episode re-evaluations. A
`checkpoint base view` is deliberately distinct from an independently trained
base-control.

| Task / boundary | Artifact | Mode | D4RL normalized score | Early fall | Evidence status |
|---|---|---|---:|---:|---|
| Hopper / 600k | `fresh_frozen_ddim_matched_dsrl_hopper-medium-v2_seed1_600000chunks` | matched DSRL, seed 1 | 89.66 | 46% | local final evaluation |
| Hopper / 600k | `fresh_frozen_ddim_matched_dsrl_hopper-medium-v2_seed2_600000chunks` | matched DSRL, seed 2 | 94.23 | 25% | local final evaluation |
| Walker / 600k | `walker_k4_base_continue_750k...tbfix_v1` | independent base-control | 96.78 | 0% | local re-evaluation; legacy 500k branch |
| Walker / 600k | `walker_k4_additive_res_750k...tbfix_v1` | checkpoint base view | 96.46 | 1% | local re-evaluation; legacy 500k branch |
| Walker / 600k | `walker_k4_additive_res_750k...tbfix_v1` | full hierarchy | 94.16 | 36% | local re-evaluation; legacy 500k branch |
| Hopper / 800k | remote result, artifact not present locally | full, seed 1 | 97.86 | 5% | user-reported; not local-machine evidence |
| Hopper / 800k | remote result, artifact not present locally | independent base-control, seed 1 | 92.88 | 26% | user-reported; not local-machine evidence |

The legacy Walker rows are diagnostic only: their full/control pair branches
at 500k, not the active 300k paired protocol. Do not use them as the current
main comparison.

Local score evidence paths are:

- `logs/p6/fresh_frozen_ddim_matched_dsrl_hopper-medium-v2_seed1_600000chunks/evaluations/final_current_base_only.json`
- `logs/p6/fresh_frozen_ddim_matched_dsrl_hopper-medium-v2_seed2_600000chunks/evaluations/final_current_base_only.json`
- `logs/p6/eval_100ep_walker_600k_20260909/base_control_checkpoint_600000_current_base_only.json`
- `logs/p6/eval_100ep_walker_600k_20260909/vs_hier_full_checkpoint_600000_current_base_only.json`
- `logs/p6/eval_100ep_walker_600k_20260909/vs_hier_full_checkpoint_600000_current_full_hierarchy.json`
