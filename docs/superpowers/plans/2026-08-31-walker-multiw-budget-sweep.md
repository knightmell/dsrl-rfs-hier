# Walker Gaussian Multi-w Budget Sweep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add auditable, memory-safe same-state Gaussian multi-`w` QW training and launch the approved Walker 50k/100k performance sweep.

**Architecture:** Extend only QW teacher construction with explicit state-count, candidate-count and teacher-microbatch contracts.  Keep K1 backward-compatible and preserve target-`QA_base`, Gaussian source, NoClip and Phase-B-only semantics.  Resolve each experiment through Hydra/preflight and bind the effective teacher workload into run manifests.

**Tech Stack:** Python 3, PyTorch, Stable-Baselines3 fork, Hydra/OmegaConf, pytest, MuJoCo/Gym.

**Spec:** `docs/superpowers/specs/2026-08-31-walker-multiw-budget-sweep-design.md`

## Global Constraints

- No online-QA teacher, ranking loss, SVGD, residual update or update-order change.
- Stop every screening run at exactly 100,000 chunk transitions.
- Default K1 behavior and existing configs remain unchanged.
- Use `apply_patch` for source/config/document edits and preserve unrelated dirty-worktree changes.
- Do not launch production before focused tests, 1k smoke and live GPU admission pass.

---

### Task 1: Add the Multi-w Constructor and Wiring Contract

**Files:**
- Modify: `stable-baselines3/stable_baselines3/dsrl/hierarchical_rfs_dsrl.py`
- Modify: `p6_train.py`
- Modify: `p6_preflight.py`
- Test: `stable-baselines3/tests/test_dsrl_rfs_hier_phase6.py`
- Test: `tests/test_p6_train_wiring.py`
- Test: `tests/test_p6_preflight.py`

**Interfaces:**
- Consumes: existing `qw_teacher_source`, global `batch_size`, `_qw_base_loss()` and `_update_qw_once()`.
- Produces: constructor fields `qw_candidates_per_state: int`, `qw_state_batch_size: int`, and `qw_teacher_microbatch_size: int`; manifest key `qw_teacher_queries_per_update`.

- [ ] **Step 1: Write failing constructor and preflight tests**

Add tests that instantiate K/B/microbatch values, reject booleans and non-positive integers, assert K1 defaults, and require the four screening configs to record exact query counts 1024, 2048, 256 and 2048.

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
/home/mrf/miniconda3/envs/dsrl/bin/python -m pytest stable-baselines3/tests/test_dsrl_rfs_hier_phase6.py tests/test_p6_train_wiring.py tests/test_p6_preflight.py -q
```

Expected: failures for missing constructor/config/manifest fields.

- [ ] **Step 3: Implement validation and wiring**

Validate positive non-boolean integers in `HierarchicalRFSDSRL.__init__`, forward them from `_construct_hierarchy`, and serialize K/B/microbatch/query count under the hierarchy training contract.  When fields are absent, resolve K=1, B=`batch_size`, and microbatch=B.

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run the Step 2 command.  Expected: all selected tests pass.

### Task 2: Implement Vectorized Multi-w and Exact Mean-Loss Microbatching

**Files:**
- Modify: `stable-baselines3/stable_baselines3/dsrl/hierarchical_rfs_dsrl.py`
- Test: `stable-baselines3/tests/test_dsrl_rfs_hier_phase6.py`

**Interfaces:**
- Consumes: constructor fields from Task 1 and `HierarchyReplayBufferSamples` sampled with `qw_state_batch_size`.
- Produces: `_iter_qw_teacher_microbatches(observations)` yielding contiguous `(observation_rows, noise_rows, teacher_heads)` blocks and `_update_qw_once()` performing one normalized optimizer step.

- [ ] **Step 1: Write failing shape, call-count and equivalence tests**

Use small B/K values to assert contiguous state repetition, independent Gaussian candidates, exact DDIM/QA/QW row counts, one optimizer step, zero residual calls, and equality between unchunked and microbatched losses/parameter gradients.  Add a K1 RNG regression that compares the existing one-draw path exactly.

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
/home/mrf/miniconda3/envs/dsrl/bin/python -m pytest stable-baselines3/tests/test_dsrl_rfs_hier_phase6.py -q
```

Expected: failures because multi-w construction and microbatch accumulation do not exist.

- [ ] **Step 3: Implement the minimal multi-w path**

For K>1, repeat observations with `repeat_interleave(K, dim=0)`, draw Gaussian decoder inputs shaped `[B_s,K,chunk,dim]`, flatten candidates, decode and label under `no_grad`, and accumulate the per-head sum of squared errors divided by the global row count.  Preserve the original `_qw_base_loss()` branch unchanged for K1.  Sample QW replay with `qw_state_batch_size`; leave QA/actor replay batches at the global batch size.

- [ ] **Step 4: Run focused and existing no-clip tests**

Run:

```bash
/home/mrf/miniconda3/envs/dsrl/bin/python -m pytest stable-baselines3/tests/test_dsrl_rfs_hier_phase6.py stable-baselines3/tests/test_dsrl_rfs_hier_no_clip.py -q
```

Expected: all tests pass.

### Task 3: Add Collision-free 100k Screening Configs

**Files:**
- Create: `cfg/gym/p6_walker_base_diag_gaussian_k4_b256_noclip_100k.yaml`
- Create: `cfg/gym/p6_walker_base_diag_gaussian_k8_b256_noclip_100k.yaml`
- Create: `cfg/gym/p6_walker_base_diag_gaussian_k8_b32_noclip_100k.yaml`
- Create: `cfg/gym/p6_walker_base_diag_gaussian_k16_b128_noclip_100k.yaml`
- Create: `cfg/gym/p6_walker_base_diag_gaussian_k64_b256_noclip_100k.yaml`
- Test: `tests/test_p6_preflight.py`

**Interfaces:**
- Consumes: `p6_walker_base_diag_gaussian_k1_noclip_250k` and Task 1 fields.
- Produces: five unique run names, 50k checkpoint/eval cadence, exact 100k intentional stop, and teacher microbatch size 256.

- [ ] **Step 1: Write failing resolved-config matrix tests**

Assert exact `(B,K,queries)` tuples, Gaussian/NoClip labels, seed-1 inheritance, 100k stop, 50k cadence, 500k Phase-B boundary and pairwise equality after removing name/K/B/query fields.

- [ ] **Step 2: Run preflight tests and verify RED**

Run:

```bash
/home/mrf/miniconda3/envs/dsrl/bin/python -m pytest tests/test_p6_preflight.py -q
```

Expected: missing-config failures.

- [ ] **Step 3: Create the five configs**

Inherit the Gaussian-K1-NoClip config, override stop/cadence and K/B/microbatch fields, and include K/B in every run directory name.  Do not modify the completed K1 run path.

- [ ] **Step 4: Resolve and test all configs**

Run the Step 2 command.  Expected: all preflight tests pass and no run-directory collisions occur.

### Task 4: Rewrite the Chinese Method and Evidence Documents

**Files:**
- Modify: `IEEE-conference-VSHS/算法说明_CN.md`
- Modify: `IEEE-conference-VSHS/实验总览_CN.md`

**Interfaces:**
- Consumes: final manifests for main VS-Hier, matched DSRL-NA, G1 diagnostics, Gaussian-K1-NoClip results and this approved sweep spec.
- Produces: a stable method document and timestamped evidence ledger with fact/inference/pending labels.

- [ ] **Step 1: Rewrite the method document**

Correct the actor/DDIM graph, QA-base replay semantics, cross-lane scope and phase-update counts.  Explain target-`QA_base` QW distillation, Gaussian proposal repair, multi-w construction, and the planned BASE-budget-preserving R scheduler without presenting pending work as a result.

- [ ] **Step 2: Rewrite the evidence overview**

Retain verified 2.5M tables, correct HalfCheetah matched differences, explain D4RL scores above 100, add completed NoClip 250k results for both seeds, record G1 ranking limitations, and add the approved K matrix and gates.  Mark K>1 and R-UTD results as pending.

- [ ] **Step 3: Verify document consistency**

Run:

```bash
rg -n '未运行|正在运行|online QA|K>1|102\.62|107\.78|4\.6' IEEE-conference-VSHS/*_CN.md
git diff --check -- IEEE-conference-VSHS docs/superpowers
```

Expected: no stale run-status claims, no incorrect HC difference and no whitespace errors.

### Task 5: Run the 1k Smoke Gate

**Files:**
- Output: `artifacts/wbsd/K_multiw_100k/smoke/`

**Interfaces:**
- Consumes: tested code and five resolved configs.
- Produces: machine-readable config/hash/counter/finite/peak-memory audit for first-batch admission.

- [ ] **Step 1: Run static preflight for all five configs**

Resolve each config and confirm unique paths, exact source/K/B/query labels and stop boundaries.

- [ ] **Step 2: Run first-batch 1k smokes sequentially**

Use collision-free smoke overrides, verify finite losses/checkpoints, `residual_actor_optimizer_steps=0`, matching seed-1 source/prefill/replay hashes and measured CUDA peak allocation.

- [ ] **Step 3: Apply the GPU admission gate**

Require a live `nvidia-smi`, temperature below 80 C, no OOM evidence and at least 8 GB free after initialization.  Do not auto-kill or restart any process.

### Task 6: Launch and Monitor the Formal Sweep

**Files:**
- Output: `logs/p6/base_diag_qw_gaussian_multiw_*`
- Output: `artifacts/wbsd/K_multiw_100k/`

**Interfaces:**
- Consumes: M0/M1-passing first-batch configs.
- Produces: four concurrent 100k runs, then queued K64-B256; 50k/100k checkpoints and a fixed 100-episode comparison.

- [ ] **Step 1: Launch the first four production runs**

Start K4-B256, K8-B256, K8-B32 and K16-B128 only after GPU admission.  Record PIDs, resolved config hashes, source/prefill/initial-replay hashes and start times.

- [ ] **Step 2: Monitor material events**

Report only 50k/100k checkpoints, failure, three-check stalls or unsafe GPU state.  Never kill or restart automatically.

- [ ] **Step 3: Queue K64-B256**

Launch K64 only after first-batch GPU capacity is released.  Preserve its full 50k/100k contract and report its 64x teacher workload.

- [ ] **Step 4: Produce the screening table**

Evaluate all completed checkpoints on the frozen 100-episode seed list and report mean, healthy mean, minimum, early-fall, AUC, QW diagnostics, cumulative teacher queries and wall time.  Apply the selection gate from the spec.

## Self-review

- Spec coverage: constructor, multi-w math, microbatching, configs, docs, smoke, launch and selection each have a task.
- Placeholder scan: no TBD/TODO/"similar to" instructions are present.
- Type consistency: the same three constructor/config names and query-count definition are used throughout.
