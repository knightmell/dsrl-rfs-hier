# Walker BASE DSRL-Mirror Implementation Plan

> **For agent:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add auditable Phase-B-only switches that progressively mirror matched DSRL-NA, then add a query-matched same-state K4 experiment without altering the residual architecture.

**Architecture:** Keep `HierarchicalRFSDSRL` as the single algorithm and introduce narrow Phase-B strategy switches for QW teacher network, update organization and replay sampling.  Existing defaults remain bit-compatible.  Multi-w is implemented inside QW teacher construction with explicit state-count and candidate-count contracts; it is not mixed into the DSRL-mirror attribution ladder.

**Tech Stack:** Python 3, PyTorch, Stable-Baselines3 fork, Hydra/OmegaConf YAML, pytest, MuJoCo/Gym.

**Design spec:** `docs/superpowers/specs/2026-08-31-walker-base-dsrl-mirror-design.md`

---

## Task 1: Define and Gate the New Semantic Switches

**Files:**
- Modify: `stable-baselines3/stable_baselines3/dsrl/hierarchical_rfs_dsrl.py`
- Modify: `p6_preflight.py`
- Modify: `p6_train.py`
- Test: `stable-baselines3/tests/test_dsrl_rfs_hier_phase6.py`
- Test: `tests/test_p6_preflight.py`
- Test: `tests/test_p6_train_wiring.py`

**Step 1: Write failing enum/default tests**

Add tests that require:

- teacher network values `target` and `online`;
- update organization values `hierarchy_block` and `dsrl_interleaved`;
- replay sampling values `tagged_pair` and `flat_equivalent`;
- absent fields resolve to the current target/block/tagged behavior;
- invalid values fail preflight;
- DSRL-interleaved is rejected when `stop_after_chunk_transitions` exceeds the
  Phase-B boundary, QA-joint shadow is enabled, or Phase-B residual/joint
  optimizer counts are nonzero.

Run:

```bash
/home/mrf/miniconda3/envs/dsrl/bin/python -m pytest tests/test_p6_preflight.py tests/test_p6_train_wiring.py stable-baselines3/tests/test_dsrl_rfs_hier_phase6.py -q
```

Expected: FAIL on missing fields/constants/wiring.

**Step 2: Add constants and constructor validation**

Define the enums next to `QW_TEACHER_SOURCES`, validate them in
`HierarchicalRFSDSRL.__init__`, and store normalized string values.  Defaults
must reproduce the current implementation.

**Step 3: Wire config, preflight and manifest**

Read the fields from Hydra in `p6_train.py`, validate cross-field Phase-B
constraints in `p6_preflight.py`, and serialize them under
`hierarchy_contract`.  Do not mutate existing configs.

**Step 4: Run focused tests**

Run the command from Step 1.  Expected: PASS.

## Task 2: Select Online or Target QA for QW Labels

**Files:**
- Modify: `stable-baselines3/stable_baselines3/dsrl/hierarchical_rfs_dsrl.py`
- Test: `stable-baselines3/tests/test_dsrl_rfs_hier_phase6.py`
- Test: `tests/test_p6_train_wiring.py`

**Step 1: Write a failing call-target test**

Use distinct spy QA modules for `qa_base` and `qa_base_target`.  For Gaussian
K1, assert exactly one selected teacher is called, the other receives zero
calls, teacher outputs are detached, QW receives one candidate per state, and
no actor or residual module is queried.

**Step 2: Implement the selector in `_qw_base_loss`**

Choose `self.qa_base` for `online` and `self.qa_base_target` for `target`.
Preserve the existing joint-credit branch and no-grad boundary.  Do not change
proposal generation, loss weighting or optimizer counts.

**Step 3: Run the focused tests**

```bash
/home/mrf/miniconda3/envs/dsrl/bin/python -m pytest stable-baselines3/tests/test_dsrl_rfs_hier_phase6.py tests/test_p6_train_wiring.py -q
```

Expected: PASS.

## Task 3: Add Phase-B DSRL-Interleaved Update Organization

**Files:**
- Modify: `stable-baselines3/stable_baselines3/dsrl/hierarchical_rfs_dsrl.py`
- Test: `stable-baselines3/tests/test_dsrl_rfs_hier_phase6.py`
- Test: `tests/test_p6_train_wiring.py`

**Step 1: Write failing update-trace tests**

Instrument sampling and update methods.  For a small profile, require the
trace:

```text
sample(base,A) -> alpha(A) -> qa_base(A) -> actor(A) -> target
sample(base,B) -> alpha(B) -> qa_base(B) -> actor(B) -> target
sample(base,C) -> qw(C)
```

Assert the same Python batch object is reused within each A/B update, QW uses
independent later batches, QA-joint/residual update counts remain zero, and all
optimizer/target counters equal the matched-DSRL formula.

**Step 2: Extract a Phase-B interleaved helper**

Add a private helper called only when the new organization enum is selected.
For each shared batch, construct the actor action/log-prob once before any
optimizer step, then apply alpha, QA-base and actor updates from those tensors.
Refactor `_update_qa_base_once` so target updating can be deferred; in this mode
perform Polyak/batch-norm target updates only after the actor step, exactly as
native DSRL does.  Reuse `_update_qw_once` and existing loss primitives rather
than duplicating formulas, while retaining the actor-to-QW gradient isolation
invariant.

**Step 3: Preserve hierarchy-block behavior**

Keep the existing block path byte-for-byte where practical.  Add a regression
trace proving the default still executes QA-base, optional QA-joint, QW, then
actor blocks with independently sampled batches.

**Step 4: Run focused and regression tests**

```bash
/home/mrf/miniconda3/envs/dsrl/bin/python -m pytest stable-baselines3/tests/test_dsrl_rfs_hier_phase6.py tests/test_p6_train_wiring.py tests/test_p6_checkpointing.py -q
```

Expected: PASS.

## Task 4: Add Flat-Equivalent Phase-B Replay Sampling

**Files:**
- Modify: `stable-baselines3/stable_baselines3/dsrl/hierarchical_replay_buffer.py`
- Modify: `stable-baselines3/stable_baselines3/dsrl/hierarchical_rfs_dsrl.py`
- Test: `stable-baselines3/tests/test_dsrl_rfs_hier_phase6.py`

**Step 1: Write failing sampler tests**

Build a small all-BASE tagged buffer with unequal valid rows near the ring
boundary.  Verify:

- `tagged_pair` samples uniformly from valid `(row, env)` pairs;
- `flat_equivalent` follows SB3's row-then-environment sampling and never
  samples invalid storage;
- returned transition tensors and metadata remain aligned;
- a fixed NumPy RNG state makes each mode reproducible.

**Step 2: Implement an explicit sampler method**

Add a Phase-B sampler that reproduces native flat replay's row/environment draw
order while returning `HierarchyReplayBufferSamples`.  Do not change the
default `sample_branch` implementation.

**Step 3: Route only H3 through the sampler**

Select the sampler through the new enum in the Phase-B helper.  Record sampler
mode and sample-call counts in diagnostics/manifest.

**Step 4: Run focused tests**

```bash
/home/mrf/miniconda3/envs/dsrl/bin/python -m pytest stable-baselines3/tests/test_dsrl_rfs_hier_phase6.py -q
```

Expected: PASS.

## Task 5: Create H1-H3 Pilot Configs and Static Audit

**Files:**
- Create: `cfg/gym/p6_walker_base_diag_gaussian_k1_noclip_online_250k.yaml`
- Create: `cfg/gym/p6_walker_base_diag_gaussian_k1_noclip_online_dsrl_order_250k.yaml`
- Create: `cfg/gym/p6_walker_base_diag_full_mirror_k1_250k.yaml`
- Modify: `tests/test_p6_preflight.py`

**Step 1: Write failing resolved-config comparisons**

Resolve H0-H3 and matched DSRL.  Assert H0/H1 differ only in teacher network,
H1/H2 only in organization plus required shadow disablement, and H2/H3 only in
replay sampler/name.  Assert all frozen hyperparameters and artifact hashes
match.

**Step 2: Add collision-free configs**

Each config inherits the existing Gaussian-K1-NoClip 250k config and overrides
only its declared layer.  Names must include `online`, `dsrl_order`, or
`full_mirror` and seed.

**Step 3: Run static audit**

```bash
/home/mrf/miniconda3/envs/dsrl/bin/python -m pytest tests/test_p6_preflight.py -q
```

Expected: PASS with a printed/resolved comparison artifact generated by the
existing preflight path.

## Task 6: Run D1 Smoke Gates Before Production

**Files:**
- Create: `cfg/gym/p6_walker_base_diag_gaussian_k1_noclip_online_smoke1k.yaml`
- Create: `cfg/gym/p6_walker_base_diag_gaussian_k1_noclip_online_dsrl_order_smoke1k.yaml`
- Create: `cfg/gym/p6_walker_base_diag_full_mirror_k1_smoke1k.yaml`
- Output: `artifacts/wbsd/D1_mirror_smoke/`

**Step 1: Launch H1 only and audit**

Run H1 to 1k, verify completion marker, finite values, hashes, call labels,
optimizer counters and zero residual steps.  Record peak allocated/reserved GPU
memory.

**Step 2: Admit H2 and H3 sequentially**

Only launch the next smoke after the preceding manifest/counter audit passes.
Do not run production arms yet.

**Step 3: Produce a machine-readable gate summary**

Write one JSON/CSV summary containing config/source hashes, initial module
hashes, counters, finite checks, process exit status and GPU peak for all arms.

Expected: D1 PASS for every arm proposed for 50k.

## Task 7: Add Same-State Multi-w Teacher Construction

**Files:**
- Modify: `stable-baselines3/stable_baselines3/dsrl/hierarchical_rfs_dsrl.py`
- Modify: `p6_preflight.py`
- Modify: `p6_train.py`
- Test: `stable-baselines3/tests/test_dsrl_rfs_hier_phase6.py`
- Test: `tests/test_p6_preflight.py`

**Step 1: Write failing shape/query-count tests**

For small B/K values, assert observations are repeated by candidate within
state, Gaussian candidates are independent, decode/teacher/student receive
exactly `B*K` rows, the mean loss is invariant to a duplicated K dimension,
and K=1 reproduces the existing loss/RNG path.

**Step 2: Add explicit K and state-batch fields**

Introduce `rfs_hier_qw_candidates_per_state` and
`rfs_hier_qw_state_batch_size`.  Default to K=1 and the normal batch size.
Preflight must record total teacher queries and reject inconsistent
query-matched declarations.

**Step 3: Vectorize teacher construction**

Repeat observations, draw `[B,K,chunk,dim]` Gaussian decoder inputs, flatten
for DDIM/QA/QW, and average over all pairs.  Keep all teacher tensors detached.
Do not add ranking or SVGD.

**Step 4: Run focused tests**

```bash
/home/mrf/miniconda3/envs/dsrl/bin/python -m pytest stable-baselines3/tests/test_dsrl_rfs_hier_phase6.py tests/test_p6_preflight.py tests/test_p6_train_wiring.py -q
```

Expected: PASS.

## Task 8: Create and Gate K4 Configs

**Files:**
- Create: `cfg/gym/p6_walker_base_diag_mirror_k4_query_matched_250k.yaml`
- Create: `cfg/gym/p6_walker_base_diag_mirror_k4_extra_compute_250k.yaml`
- Create: `cfg/gym/p6_walker_base_diag_mirror_k16_stress50k.yaml`
- Modify: `tests/test_p6_preflight.py`

**Step 1: Encode the matrix**

- K1: 256 states x 1.
- K4-QM: 64 states x 4, 256 teacher queries.
- K4-XC: 256 states x 4, 1024 teacher queries.
- K16 stress: 50k maximum unless explicitly promoted later.

**Step 2: Assert all non-K dimensions match**

Resolved configs must match the accepted K1 mirror in source, teacher network,
organization, sampler, optimizer counts and artifacts.

**Step 3: Run 1k memory smokes sequentially**

Record CUDA peak allocation/reservation and wall time.  Run K4-XC alone for its
first smoke.  Require at least 8 GB physical free memory after initialization.

Expected: Gate K smoke PASS before any 250k launch.

## Task 9: Production Scheduling and Analysis

**Files:**
- Output: `artifacts/wbsd/D3_mirror_250k/`
- Output: `artifacts/wbsd/K_multiw_250k/`
- Create: `docs/rfs_hier_v1/WALKER_BASE_MIRROR_RESULTS.md`

**Step 1: Apply the D3 branch rule**

When H0 seed1/seed2 finish, compare 200k and 250k to matched DSRL.  If both are
within 2 points with comparable tail-risk, prioritize K4-QM.  Otherwise launch
H1, audit at 50k, then H2, and H3 only if H2 remains outside tolerance.

**Step 2: Limit concurrency by measured memory**

Use at most four K1 processes.  Before every admission, query GPU memory,
utilization and temperature.  K4-XC/K16 start alone until their measured peaks
are known.  Never auto-kill or auto-restart.

**Step 3: Report robust metrics**

At 50k intervals report D4RL mean, median, standard deviation, minimum,
early-fall rate, learning AUC, QW diagnostics, optimizer counters, actor
pre/post-clipping gradient norms and wall-clock throughput.  Compare paired
episode seeds where available.

**Step 4: Apply the multi-w claim gate**

Claim same-state depth only if K4-QM improves over K1 at equal query count on at
least two seeds in AUC or tail-risk.  Otherwise report multi-w as extra compute
without independent structural benefit and stop K16 promotion.

## Task 10: Final Verification

**Files:**
- Verify all modified Python/YAML/test files above.

**Step 1: Run the full focused suite**

```bash
/home/mrf/miniconda3/envs/dsrl/bin/python -m pytest tests/test_p6_preflight.py tests/test_p6_train_wiring.py tests/test_p6_checkpointing.py stable-baselines3/tests/test_dsrl_rfs_hier_phase6.py stable-baselines3/tests/test_dsrl_rfs_hier_no_clip.py tests/test_wbsd_dsrl_mirror.py -q
```

Expected: PASS.

**Step 2: Audit the diff**

Confirm defaults are unchanged, no unrelated dirty-worktree files were staged,
all configs are collision-free, and every new semantic field appears in the
manifest and result tables.

**Step 3: Request code review**

Use `superpowers:requesting-code-review` on the implementation diff before any
production H1-H3/K4 run.
