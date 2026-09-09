# Experiment Registry

Last updated: 2026-09-09

This registry separates intended runs from demonstrated runs. A row is not `complete` without a terminal marker, checkpoint or bundle, and evaluation evidence.

## Evidence levels

| Status | Meaning |
|---|---|
| planned | Name or config exists; no training claim |
| running | Live process and `RUNNING` status observed |
| interrupted | Process ended intentionally with a resumable boundary |
| complete | Final boundary, terminal marker, checkpoint or bundle, and evaluation verified |
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

## Planned 800k campaign

The following eight runs are planned but have not been launched. `full` and
`basecontrol` branches must use the same declared BASE source and BASE update
schedule where they are paired; disabling residuals only at evaluation time is
not sufficient to create a `basecontrol` result. `noclip` is the canonical
spelling. For VS-Hier full runs, the agreed continuation uses RES UTD=4; for
base-control runs, RES UTD=0 and no residual/joint optimization.

| Canonical identity | Exact resume source | Role | Target | Status |
|---|---|---|---:|---|
| `walker_s1_noclip_k4_300k_800k_full` | `logs/p6/base_diag_qw_gaussian_k4_b256_noclip_100k_fresh_frozen_ddim_walker2d-medium-v2_dsrl_na_rfs_hier_seed1_2500000chunks/resume/chunk_000000300000_source_compat_v1` | VS-Hier full, RES UTD=4 | 800k | planned |
| `walker_s1_noclip_k4_300k_800k_basecontrol` | same Walker 300k source as above | independent BASE-only control, RES UTD=0 | 800k | planned |
| `hopper_s1_noclip_k4_300k_800k_full` | `logs/p6/hopper_s1_noclip_k4_300k_800k_full` | existing 300k-branch full continuation from 600k, RES UTD=4 | 800k | running |
| `hopper_s1_noclip_k4_300k_800k_basecontrol` | `logs/p6/hopper_s1_noclip_k4_300k_800k_basecontrol` | independent BASE-only control from 300k, RES UTD=0 | 800k | running |
| `halfcheetah_s1_noclip_k4_300k_800k_full` | `logs/p6/halfcheetah_s1_noclip_k4_300k_800k_full` | existing 300k-branch full continuation from 600k, RES UTD=4 | 800k | running |
| `halfcheetah_s1_noclip_k4_300k_800k_basecontrol` | `logs/p6/fresh_frozen_ddim_600k_k4_noclip_halfcheetah-medium-v2_dsrl_na_rfs_hier_seed1_600000chunks_v2/resume/chunk_000000300000` | independent BASE-only control, RES UTD=0 | 800k | planned |
| `hopper_s1_matched_dsrl_600k_800k_control` | `logs/p6/fresh_frozen_ddim_matched_dsrl_hopper-medium-v2_seed1_600000chunks/resume/chunk_000000600000` | matched DSRL control | 800k | planned |
| `hopper_s2_matched_dsrl_600k_800k_control` | `logs/p6/fresh_frozen_ddim_matched_dsrl_hopper-medium-v2_seed2_600000chunks/resume/chunk_000000600000` | matched DSRL control | 800k | planned |

### Proposed resource waves

This is a scheduling proposal, not permission to launch. The two matched DSRL
runs should be run together in an otherwise empty GPU wave because they have
the highest observed GPU pressure. The six VS-Hier runs can be attempted in
two three-process waves:

1. Hopper full + Hopper basecontrol + HalfCheetah full.
2. HalfCheetah basecontrol + Walker full + Walker basecontrol.

Four concurrent processes are not an approved default. A historical attempt to
restore four 100k replay states concurrently produced `HostMemoryOOM` on the
64 GiB host with 2 GiB swap. The replay payloads alone are approximately 5.36
GiB for Hopper and 9.68 GiB for HalfCheetah/Walker, before Python, model,
allocator, and environment overhead. Three processes are therefore a
candidate, not a proven safe limit; before each wave, perform one focused
`nvidia-smi` plus host-memory check and reduce to two if either GPU or host
memory is not clearly available. Do not run the two matched DSRL processes
concurrently with a VS-Hier wave.
