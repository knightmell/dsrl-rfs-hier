# Stage 1 Independent Technical Re-review

Review date: 2026-08-05  
Repository: `/home/mrf/dsrl`  
Outer HEAD: `18d5be67c78694527b52599c4aaeafaf6e432590`  
SB3 HEAD: `076e5563ceb51ff91c6c7743501d18f0456b13af`  
DPPO HEAD: `86ce51834055c02f9013e60dd4c4275606d82df7`

## 1. Review conclusion

**PASS**

The five Stage 1 `REVISE` findings are resolved by document-only changes. The current-code audit is accurate, the reference review is adequate for the design decisions it supports, and the implementation specification now defines one mechanically testable Core V1 contract without requiring the Stage 2 implementation Agent to invent phase, replay, resume, entropy, or initialization semantics.

This `PASS` authorizes **Stage 2 Core V1 implementation and automated tests only**. It does not authorize Stage 3 smoke, 100k, 500k, or 5M training.

## 2. Evidence inspected

- `docs/rfs_hier_v1/THREE_CRITIC_CODE_AUDIT.md`, SHA-256 `222bb4044086df11e2f39e881a1b850e9328b4c7be735bc376ceddaf5b06ec69`
- `docs/rfs_hier_v1/MULTI_CRITIC_REFERENCE_REVIEW.md`, SHA-256 `8f5e06d2865c0970b73676f6e42a6e5ba9f506d2b77d356692cc9c1c22c6d51f`
- `docs/rfs_hier_v1/THREE_CRITIC_IMPLEMENTATION_SPEC.md`, SHA-256 `0389de6543a4a03bf4cf1befe3f669d0f6a12dc826dd38344a43a156f91922c6`
- the current hierarchy, legacy DSRL, replay/storage, ActionChunk, P6 checkpoint/runtime/evaluation code named by the audit
- the complete file-level working-tree fingerprint in the revised code audit
- the primary/official reference sources and artifact-recovery subsection

No training code was changed or executed for this document re-review. The reviewer independently recomputed every listed source hash: all 29 inspected source/test entries and both explicitly excluded flat-RFS files match the revised manifest.

## 3. Re-review of previous findings

### S1-01 — Schedule: resolved

The specification now freezes two named, manifest-visible profiles:

| Profile | Phase B | Phase R | Phase J | Beta ramp | Beta target | BASE lane probability |
|---|---:|---:|---:|---:|---:|---:|
| `fresh_frozen_ddim_5m` | 2,500,000 | 2,500,000 | disabled / 0 | 50,000 | 0.1 | 0.5 |
| `legacy_dsrl_warmstart_5m` | 0 | 5,000,000 | disabled / 0 | 50,000 | 0.1 | 0.5 |

`min_branch_replay_transitions=256` is frozen and must be at least the batch size. The spec defines:

- online chunk-transition units and exclusion of prefill from the scheduler;
- divisibility by `n_envs` and rejection of split vector batches;
- exact B/R/J interval inequalities;
- zero-length B initialization;
- exact beta first/final ramp values;
- phase changes for action batches while preserving the lane of an active episode;
- B-to-R QA-joint clone, target copy, optimizer recreation, generation and counter semantics;
- default-disabled Phase J and explicit manifest-visible override behavior.

The production values are divisible for the audited Hopper `n_envs=4` and the explicit pilot `n_envs=10`. No schedule choice is left implicit.

### S1-02 — Resume: resolved

The specification now separates:

1. required `reset_boundary` resume, which restores learner/replay/optimizers/counters/software RNG but discards stale active simulator episode state; and
2. optional `simulator_exact` resume, which is legal only behind an adapter that serializes and restores the complete simulator, VecEnv, wrapper, action-chunk, RNG, observation, lane, episode and partial-statistic state.

For reset-boundary resume, reset seed derivation, environment order, fresh episode/lane allocation, chunk reset, discontinuity counters and first-transition assertions are defined. Old lane/chunk metadata cannot be attached to a reset simulator. Requesting simulator-exact mode without a complete adapter must fail preflight.

### S1-03 — Alpha ownership: resolved

The revised spec explicitly defines:

\[
\alpha_{sg}=\exp(\operatorname{sg}(\log\alpha))
\]

for every Bellman and noise-actor loss. Only `L_alpha` owns `log_alpha`. It freezes alpha-first/noise-second ordering, uses the same actor sample, snapshots the pre-alpha-step detached coefficient, and requires unrelated gradients to be cleared and absent. The loss definitions now agree with the gradient/parameter-change matrix.

### S1-04 — Matched prefill and terminal truth: resolved

The specification now mandates one immutable tagged BASE artifact for every matched control/hierarchy pair. Hierarchy reads the full schema; control reads the exact standard projection without regeneration, casting, reordering, or re-sampling. Per-array and aggregate hashes must be bitwise identical.

Terminal insertion is uniquely defined:

```text
done       = bool(dones[i])
timeout    = done and TimeLimit.truncated
terminated = done and not timeout
truncated  = timeout
```

Autoreset output is replaced with `terminal_observation` before artifact or replay hashing. Timeout without done and done without terminal observation fail. The Bellman mask remains `done * (1-timeout)`.

### S1-05 — Research recovery and source provenance: resolved

The reference review now distinguishes network warm-start, reset-boundary resume and simulator-exact resume, and documents the recovery limits of official TD3, SB3, REDQ, Policy Decorator, ResFiT, Recovery RL, TQC and HIRO sources. It correctly avoids treating ordinary model save/load as exact replay/RNG/simulator continuation.

The code audit now fingerprints the exact dirty working-tree bytes instead of relying on committed SHAs. The reviewer recomputed the hashes and found no mismatch. The two untracked flat-RFS files are separately fingerprinted and remain explicitly excluded from Stage 2.

## 4. Frozen design approved for Stage 2

Stage 2 must implement exactly these credit paths:

```text
QA_base -> QW_base -> noise actor
QA_joint          -> residual actor
```

The approved contract includes:

- independent twin `QA_base`, twin `QW_base`, and twin `QA_joint` with disjoint parameters and optimizers;
- target `QA_base`, target `QA_joint`, and target residual; no target QW or target noise actor;
- current stochastic noise and matching same-sample log-prob in both Bellman targets;
- residual permanently absent from the base target;
- target residual present only in the joint Bellman target;
- target-QA-base, head-aligned QW regression on fresh current-policy candidates from BASE observations;
- noise loss that does not call DDIM, residual or QA-joint;
- residual update through current QA-joint action gradient with noise/base detached;
- one hierarchy-specific tagged replay with episode-fixed lanes and strict Core V1 branch sampling;
- centered bound-preserving residual composition with emergency clamp only for numerical roundoff;
- the two frozen 5M profiles and their exact beta/update/boundary semantics;
- legacy DSRL formal loading, hard-copy initialization, zero residual output, and explicit rejection of old joint-QM hierarchy checkpoints;
- complete online/target/optimizer/replay/counter/RNG/version persistence under the frozen resume labels;
- four same-seed exact-N evaluation modes;
- ranking, delta-Q, disagreement and policy-age diagnostics only.

Core V1 must not add QM, ranking loss, hard residual gates, residual dropout, shared QA trunks, target noise actor, REDQ/TQC ensembles, PPO residual, conservative offline losses, CVaR, residual entropy, BC or L2.

## 5. Stage 2 reviewer requirements

The later Stage 2 review must inspect the real code diff and exact test output. At minimum it must verify:

- parameter and optimizer ownership by identity/storage pointer;
- target source and module call graphs for both QA losses and QW;
- gradient and parameter-change matrix for every loss;
- DDIM hash/freeze and module train/eval modes;
- action composition parity, zero-point gradient and bounds;
- vector-env episode-fixed lanes and terminal-transition ownership;
- tagged replay insertion, wraparound, branch counts, filtered sampling, hashing and pickle restoration;
- matched prefill projection equality and terminal/timeout truth;
- schedule boundaries, beta indexing, B-to-R regeneration, counters and both resume modes;
- deterministic save/load equivalence;
- all four evaluation modes;
- complete regression coverage proving `algorithm=dsrl_na` remains unchanged.

## 6. Non-blocking note

`THREE_CRITIC_IMPLEMENTATION_SPEC.md` contains a duplicated `Copy:` immediately before the legacy mapping block. This is a cosmetic documentation typo and does not create ambiguity. It may be removed during a documentation-only cleanup but is not a condition of this `PASS`.

## 7. Final authorization

**PASS — Stage 2 Core V1 implementation and automated tests may begin.**

Stage 2 must stop after its code/test/report delivery and await a separate `REVIEW_STAGE_2.md` decision. No smoke or long training is authorized by this review.
