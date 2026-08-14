# Staged Delivery Plan — Three-Critic DSRL-NA Core V1 + Gated Follow-ons

Status: ACTIVE (staged execution in progress)
Date: 2026-08-06
Anchor documents: `THREE_CRITIC_IMPLEMENTATION_SPEC.md` (Core V1, frozen), `REVIEW_STAGE_1.md` (PASS for Stage 2 implementation + tests only), Codex session `019fd03b` (2026-08-05, authoritative staged plan).

## 0. Purpose and fidelity contract

This plan codifies the authoritative staged plan from the Codex cache (2026-08-05 session) and the three-critic spec, then maps it to a staged delivery with verification gates. Fidelity rules (mandatory):

1. **No goal trimming.** Every stage's goal is stated verbatim from the plan/spec; a stage is not complete until its gate passes.
2. **No arbitrary parameter changes.** Frozen parameters (Section 3) are changed only with a spec reference; any change must be recorded as a deviation with a cited basis.
3. **No enabling gated features blindly.** V1.1/V1.2/V1.3 mechanisms are implemented behind explicit flags, default OFF, and only the diagnostics that gate them are turned on when evidence is available.
4. **No modification of expected-unchanged files** (`dsrl.py`, SB3-common, `env_utils.py`, `p6_launcher.py`, DPPO, flat-RFS, per-step experiments) unless a stop-condition is raised.
5. **Every stage ends with a verification pass** that checks (a) tests pass, (b) goals achieved, (c) no plan deviation introduced.

## 1. Current state (baseline, 2026-08-06)

| Component | State |
|---|---|
| SB3 three-critic implementation (`hierarchical_rfs_dsrl.py`, `hierarchical_replay_buffer.py`, `hierarchy_schedule.py`) | Implemented (Core V1, Stage 2) |
| SB3 hierarchy tests (phase1-5 + replay buffer) | **43 passed** |
| Outer P6 integration tests (checkpointing/preflight/evaluation/train_wiring/runtime) | **34 failed, 32 passed** — all test-file staleness, no production bug (diagnosed) |
| Spec gap: Section 8.1 fresh-from-Frozen-DDIM init | **BLOCKING — missing** (only legacy warm-start exists) |
| Spec gap: Section 10 diagnostics wiring | **BLOCKING — `_training_diagnostics` dead code** |
| Spec gap: Section 5.3 GAUSSIAN_PRIOR prefill | **BLOCKING — prefill hard-wired to warm-start CURRENT_ACTOR** |
| Spec gap: Section 10 required metrics in `train()` | Important — several required metrics not logged |
| Spec gap: `emergency_clamp_count` | Important — never incremented; pre-clamp violation never logged |
| Spec gap: constructor default profile | Important — defaults to `legacy_dsrl_warmstart_5m`, spec says `fresh_frozen_ddim_5m` |
| Spec gap: every RESIDUAL-phase profile freezes noise/alpha | Important — no escape hatch except default-disabled Phase J |
| Replay memory (7 redundant 12-dim action arrays, 100M-slot capacity) | Minor — spec-consistent; document, do not break SB3 layout |
| `simulator_exact` resume | Minor — optional, absent; fail-fast path present |

## 2. Authoritative staged plan (from Codex 2026-08-05 + spec)

Three gated stages, each stops for an independent Reviewer; only a clear PASS starts the next. A **REVISE** fixes only flagged items; a **BLOCK** stops and redesigns.

- **Stage 1 — Audit + freeze spec**: COMPLETE. `REVIEW_STAGE_1.md` = PASS (2026-08-05). Authorizes Stage 2 implementation + automated tests ONLY. No training.
- **Stage 2 — Core V1 implementation + autotests**: implemented (this repo). Awaiting `REVIEW_STAGE_2.md` PASS before smoke.
- **Stage 3 — Low-budget wiring/smoke only**: not started. Requires REVIEW_STAGE_2 PASS. Smoke proves wiring only, never performance.

Post-Core-V1 follow-ons (each gated on its own evidence, one at a time):

- **V1.1 — QW confidence-masked ranking**: enabled ONLY when (a) target-QA_base twin heads agree on candidate ordering, (b) teacher gap > `tau_gap`, (c) QA_base ranking positively correlates with real short counterfactual rollouts. Ranking reduces QW corruption of the teacher's ordering; it never fixes the teacher's own ordering errors.
- **V1.2 — QA_joint stabilization** (only if QA_joint is still exploitable): add mechanisms one at a time — QA_joint ensemble; target-policy smoothing; target-delta-Q soft gate; residual trust-region/L2; execution-time confidence gate. No REDQ/TQC/uncertainty-gate in round one. A hard delta-Q execution gate requires twin-consistency/min-head gain to first show positive correlation with real paired returns.
- **V1.3 — delta-Q correlation gate**: headwise `ΔQ_i = QA_joint,i(s,a_exec) − QA_joint,i(s,a_base)`; log only as diagnostic until correlation with real paired returns is shown.

## 3. Frozen parameters (do not change without a cited spec reference)

| Parameter | Frozen value | Spec ref |
|---|---|---|
| `beta_target` | 0.1 | §7.1 |
| `beta_ramp_chunk_transitions` | 50,000 | §7.1 |
| `base_lane_probability` | 0.5 | §7.1 |
| `min_branch_replay_transitions` | 256 (>= batch_size) | §7.1 |
| Profile `fresh_frozen_ddim_5m` | B=2.5M, R=2.5M, J off, prefill=Gaussian DDIM prior | §7.1 |
| Profile `legacy_dsrl_warmstart_5m` | B=0, R=5M, J off, prefill=legacy DSRL actor | §7.1 |
| Phase B update ratio | QA_base 20 / QA_joint 0 / QW 10 / noise 20 / alpha 20 / residual 0 | §7.2 |
| Phase R update ratio | QA_base 10 / QA_joint 10 / QW 5 / noise 0 / alpha 0 / residual 1 | §7.2 |
| Phase J update ratio | QA_base 10 / QA_joint 10 / QW 5 / noise 1 / alpha 1 / residual 1 | §7.2 |
| Target entropy (Hopper) | 0.0 | config |
| Noise/residual grad clip | 1.0 each | §7 |
| `critic_backup_combine_type` | min | §2 |
| Update cadence | train_freq=1 step, gradient_steps discarded in hierarchy | §7 |

## 4. The 2.5M from-frozen-DDIM run (reconciliation)

The plan's from-scratch profile `fresh_frozen_ddim_5m` is B=2.5M (base curriculum: noise actor + QA_base + QW_base from frozen DDIM, residual off, β=0) followed by R=2.5M (residual, noise frozen). The user's stated goal — "继续训练 2.5M，只求比原始 frozen diffusion 检查点好很多" — maps to the **base curriculum** (Phase B) of that profile: establishing the noise actor so the frozen DDIM decodes into a walking Hopper. Empirical baseline for the raw checkpoint is ~1432 raw / 100% early fall (handoff `P6_resip_2p5m_frozen_10k_result`); any trained noise actor clears it.

**Reconciliation**: the 2.5M run was first configured as a 2.5M-total schedule keeping the plan's frozen Phase-B semantics for the base curriculum (frozen `fresh_frozen_ddim_2p5m`: B=1.25M / R=1.25M, noise/alpha frozen in R). The user subsequently specified a **co-training design** (2026-08-08) that replaces this as the primary run: short Phase B (0.5M) with QA_joint shadow-training, long Phase R (2.0M) that co-trains QA_base→QW→noise/alpha and QA_joint→residual simultaneously — the noise actor is explicitly NOT frozen in R. The primary run is now `fresh_frozen_ddim_2p5m_cotrain` (see the deviation register in §7a); the frozen 50:50 profile remains as the reference schedule. The plan's `fresh_frozen_ddim_5m` remains the authoritative full-run profile.

## 5. Staged delivery (this work)

Each stage has a gate; execute in order, verify after each, no stage is skipped.

| Stage | Goal (verbatim from plan/spec) | Gate | Status |
|---|---|---|---|
| S0 | Establish baseline: test results, spec gaps, authoritative plan | Baseline recorded above | DONE |
| S1 | Repair outer P6 integration tests to the three-critic contract (34 failures: stale mocks, stale schedule-budget in tests, old-design frozen_noise tests, stale regexes) | All P6 tests pass; no production-code change for S1 unless a genuine bug is found | DONE — 64 P6 pass; all fixes were test-file-only (production untouched for S1) |
| S2 | Implement spec §8.1 fresh-from-Frozen-DDIM init, §5.3 GAUSSIAN_PRIOR prefill path, and a 2.5M-total fresh schedule profile; extend p6_preflight/p6_train to accept the fresh profile without changing the legacy default | Fresh-profile construction test passes; preflight accepts the 2.5M fresh config; legacy warmstart preflight unchanged; SB3 tests still pass | DONE — `initialize_from_fresh_frozen_ddim`, `fresh_frozen_ddim_2p5m`, source-aware prefill (p6_runtime), preflight profile/source gates, p6_train fresh branch; fresh config hydra-composes; legacy profile unchanged; SB3 47 + P6 64 pass |
| S3 | Wire §10 diagnostics: call `_training_diagnostics` in `train()`, increment `emergency_clamp_count`, log pre-clamp violation and required Section-10 metrics | `diagnostics/*` keys appear in the training logger; counter increments; SB3 + P6 tests pass | DONE — diagnostics wired (gated by `diagnostics_interval_updates`), headwise Q/twin/delta-Q/target-drift logged, `emergency_clamp_count` + `max_preclamp_violation` logged |
| S4 | Close remaining spec gaps: constructor default profile → `fresh_frozen_ddim_5m`; Section-10 metric completeness in `train()`; assert no unrelated gradients after the paired alpha/noise block | Section-10 required metric list checked against logged keys; tests pass | DONE — default profile flipped to `fresh_frozen_ddim_5m`; effective-UTD/lane-ratio/policy-version metrics logged; **fixed a genuine gradient-isolation bug** (noise backward ran after the QW freeze was released, letting QW accumulate spurious gradients) |
| S5 | Implement V1.1/V1.2/V1.3 mechanisms behind explicit flags, default OFF, with gate diagnostics only | Feature-flag defaults preserve Core V1 behavior; gate diagnostics testable; tests pass | DONE — `enable_qw_ranking` (V1.1) and `qa_joint_target_smoothing` (V1.2) default OFF with tests; V1.3 delta-Q correlation diagnostic tool + unit-tested statistics |
| S6 | Deliver 2.5M config, pre-training steps, interface check tools (preflight/checkpoint/eval commands), diagnostic-file instructions, and a runbook | 2.5M config passes preflight; runbook commands verified; no plan deviation | DONE — `cfg/gym/p6_hopper_fresh_2p5m.yaml` hydra-composes; `RUNBOOK_2P5M_FRESH.md` with launch/pre-training/tools/diagnostics/resume |
| S7 | Final staged verification + deviation report: run full test suite, check each stage gate, report any goal-trimming or param deviation | All gates green; deviation report produced | DONE — 117 tests pass (SB3 47, outer 70); deviation report below |

## 7a. Deviation report (2026-08-06)

| Change | Basis | Status |
|---|---|---|
| Added `fresh_frozen_ddim_2p5m` profile (B=1.25M, R=1.25M) | Proportional 50:50 scaling of the frozen `fresh_frozen_ddim_5m` to the user's 2.5M budget; all other frozen params unchanged | Cites plan §7.1; recorded |
| Constructor default profile → `fresh_frozen_ddim_5m` | Spec §7.1 frozen default; P6 warmstart config overrides explicitly | Spec-cited |
| `_update_alpha_and_noise_once`: QW freeze now covers forward AND backward | Spec §6.3/§12.5 gradient isolation; fixed a genuine bug where QW accumulated spurious gradients | Spec-cited, production fix |
| `_training_diagnostics` wired + Section-10 metrics | Spec §10 dead-code gap; metrics list from spec §10 | Spec-cited |
| V1.1/V1.2 flags added default OFF | Plan gates: do NOT enable blindly; flags preserve Core V1 behavior when OFF | Plan-cited |
| Fresh prefill source + preflight/train plumbing | Spec §8.1 (fresh init) and §5.3 (GAUSSIAN_PRIOR prefill); required for the 2.5M from-scratch run | Spec-cited |
| `static_preflight` top-level `prefill_action_policy` no longer hard-coded to warmstart | Fresh-run static manifest mislabeled its own prefill as warmstart behavior while `hierarchy_schedule`/`config_contract` copies were correct; runtime semantics were unaffected (p6_train overwrites the field at prefill-load time, semantics keyed on `prefill_source`). Aligned the top-level field with the two nested copies; added a fresh-profile regression test. Found during runbook §3.1 static-preflight verification | Recorded as a manifest-provenance bug fix |
| `p6_launcher.py`: `--run-dir` vs Hydra `logdir=` cross-check added (`_validate_p6_command_run_directory`) | Production-runner correctness: the certified-resume protocol keys run identity on the run directory, so a P6 command whose Hydra `logdir=` override differs from the launcher's `--run-dir` would produce a run the launcher/manifest cannot track consistently. The guard rejects the mismatch before creating the run dir (covered by `test_p6_launcher.py::test_launcher_rejects_p6_train_logdir_mismatch_before_creating_run`). Predates this session; recorded here to satisfy the do-not-do list's "without a documented stop-condition" requirement — this edit is the documented stop-condition | Recorded as a deviation; `p6_launcher.py` do-not-do item acknowledged |
| `diagnostics_interval_updates` config key (default 100) | Spec §10 dead-code gap (stage S3): §10 `train()`-level metrics were specified but never logged; the key gates the `_training_diagnostics` cadence in `HierarchicalRFSDSRL._maybe_training_diagnostics`. New config surface; behavior when the key is absent falls back to the legacy per-call constant, so no shipped config changes semantics | Spec-cited (S3 deliverable) |

### Session 2026-08-08 "全部修复" audit fixes (B1–B10)

| Change | Basis | Status |
|---|---|---|
| B1: `compose_action` bound clamp now value-preserving with the full centered subgradient (`action_exec = unclamped + (clamped − unclamped).detach()`) | Spec §12.2/§12.3: at an exact bound tie, `th.minimum`/`th.maximum` split the subgradient 0.5/0.5, halving the required zero-logit subgradient `β·(high−low)/2` (empirically 0.175 vs 0.35 at β=0.7) while producing the identical clamped VALUE — so the emergency flag never fired. The phase-1 gradient test backpropagated `action_residual_delta` (unclamped), not `action_exec`, which is why validation missed it. Detached clamp preserves values and restores the exact spec gradient; regression test added | Spec-cited, production fix |
| B2: `emergency_clamp_count` incremented on the training path (`_residual_actor_loss`), not only the rollout path | §10 requires the clamp count to cover every composition; the residual-loss backward composes actions too | Spec-cited |
| B3: grad-isolation assertion now compares grad VALUES on unrelated modules before/after the paired alpha/noise block | §12.5: earlier update blocks in the same `train()` already leave real grads on qa_base/qw_base, so a presence-only check could not detect a leak that accumulates on top of them | Spec-cited |
| B4: `next_episode_id` floor-seeded from the prefill's max episode_id at first lane allocation | The tagged prefill already consumes episode IDs `0..n_envs−1`; fresh online episodes must not reuse them. Lazy seeding avoids perturbing replay/semantic-hash state before prefill load | Recorded as a provenance fix |
| B5: runbook §5 diagnostic output and the diagnostic default moved to the gitignored `logs/`; resume fingerprint warning added | Certified resume re-verifies `source_state_sha256` over every untracked, non-gitignored repo file; a root-level output JSON between launch and resume would fail the run | Runbook/ops fix |
| B6: prefill artifact writes fsync the parent directory; orphan-recovery for partial npz/json pairs; `collect_tagged_matched_prefill_and_save` helper | Snapshot/resume bundles already fsync'd their directory; the prefill two-rename path did not, so a crash between the npz and metadata renames could leave an orphan that resume would reject or misread | Production fix |
| B7: control algorithm rejects a non-hierarchy prefill source at preflight | The control branch gates on the same `prefill_source` as the hierarchy; a fresh/mislabeled source would seed it incorrectly. Only `p6_hopper_fresh_2p5m.yaml` uses the fresh source, so no shipped config is affected; regression test added | Recorded as a gate fix |
| B8: `flush_final` calls `service_safe_boundary(..., skip_online_eval=True)`; the online-eval loop is gated on the new parameter | At final flush, the milestone online-eval would duplicate the final evaluation that runs as a separate step. The default path (interrupt boundaries) is unchanged | Production fix |
| §10 completion: added policy age, noise mean/std/entropy, Q head scales (std) for both QA meanings, per-branch episode returns/lengths/early-fall rates, and the isolated-RNG QW ranking diagnostic (spec §10 L704–715: twin-head ordering agreement, QW-vs-teacher Spearman/Kendall per head, pairwise preference accuracy, top-1 agreement, teacher top-1 gap distribution, ranking metrics by policy age) | Spec §10 dead-code gap; the ranking diagnostic restores global Python/NumPy/Torch CPU/CUDA RNG and all module modes exactly, and no ranking term enters any loss | Spec-cited (S3/S4 deliverable); 120 tests pass |

No frozen parameter was changed without a citation. No goal was trimmed. The from-scratch "noise actor keeps training through the residual phase" idea from the user discussion was recorded below as a **proposed deviation requiring reviewer sign-off**; the user has since approved it as a design decision, so the co-training profile implements it as the primary 2.5M run (see the session 2026-08-08 register below).

### Session 2026-08-08 co-training schedule (user design decision)

All rows below implement the user's explicit co-training design (2026-08-08). They replace the frozen 50:50 fresh schedule as the **primary** 2.5M run; the frozen `fresh_frozen_ddim_2p5m` profile and `p6_hopper_fresh_2p5m.yaml` config are preserved unchanged as the reference schedule. Every deviation's basis is the user design decision; the do-not-do list (§6) is still satisfied (checked at the end of this table).

| Change | Basis | Status |
|---|---|---|
| New profile `fresh_frozen_ddim_2p5m_cotrain` (B=0.5M / R=2.0M, 20:80 instead of the 50:50 proportional split; J off; `fresh_frozen_ddim` prefill) | User design: a short Phase B establishes the base branch, and ~2.0M is left for real base+residual co-training | User decision; §7.1 profile deviation |
| R profile 10/10/5/0/0/1 → 5/5/2/1/1/1 (noise actor + alpha **unfrozen** in R, trained at low frequency alongside the residual) | User design: freezing π_w at 1.25M caps residual-phase performance; the three-critic value is exactly allowing QA_base→QW→π_w and QA_joint→π_r to update simultaneously (gradients isolated per §12.5) | User decision; §7.2 deviation. Resolves the "noise keeps training" proposed-deviation note in §4 |
| B profile 20/0/10/20/20/0 → 20/10/10/20/20/0 (QA_joint **shadow-trained** on the same BASE transitions during B) | User design: R starts with QA_joint already at QA_base's scale instead of a sudden boundary clone | User decision; §7.2 deviation |
| `_activate_joint_phase` shadow mode: skips the QA_base→QA_joint soft-clone and optimizer recreation at the B→R boundary (shadow weights + Adam state preserved; `qa_joint_generation` still +1, `_joint_phase_initialized` set) | Consequence of the shadow decision; the frozen clone path is preserved when `qa_joint_shadow_in_b=False` | User decision |
| β schedule: R-start hold at 0 for 50k, then ramp 0.02→0.1 over the frozen 50k (`beta_hold_steps` + `beta_ramp_steps` ≤ `phase_r_steps`) | User design: R-start β=0 collects joint data and warms QA_joint with the residual structurally OFF; ramp starts at the floor 0.02 | User decision; §7.1 deviation |
| Residual exploration: `residual_exploration_std=0.02` pre-tanh perturbation on JOINT lanes only, gated on β>0, rollout path only (inference/eval/diagnostic compose stays deterministic) | User design: add residual exploration in R | User decision |
| Cross-lane replay 75/25: QA_base samples 25% JOINT transitions and QA_joint samples 25% BASE transitions (transition sharing; Bellman semantics NOT shared — no `QA_joint−QA_base` difference loss) | User design: cross-lane replay 75/25 | User decision; no forbidden gradient (§6) |
| `update_profiles` becomes per-profile (new `update_profiles` field on `HierarchySchedule`; `None` → the frozen global `DEFAULT_UPDATE_PROFILES`; profile-defined phases fall back to the frozen defaults for any phase not customized) | Mechanical consequence of the R-profile change; every frozen profile resolves exactly as before (verified: frozen contract unchanged) | Recorded |
| **Unchanged frozen parameters**: beta_target=0.1, beta_ramp=50k, base_lane=0.5, min_branch=256, Phase J disabled, `critic_backup_combine_type='min'`, V1.1/V1.2/V1.3 flags default OFF, no QM_joint, no ranking loss in any loss, no delta-Q hard gate, emergency clamp only | — | Verified in the cotrain config + preflight contract |

## 6. Do-not-do list (from plan; binding)

- No 100k/500k/5M training without the corresponding Reviewer PASS; smoke only after Stage-2 PASS.
- Core V1 must NOT enable: QM_joint; joint-critic→noise gradients; QW ranking loss; residual delta-Q hard gate; primitive-level residual dropout; shared QA trunks; REDQ/TQC ensembles; target noise actor; PPO residual; CVaR/risk loss; chunk-observation expansion; CQL/IQL; residual entropy/BC/L2.
- Forbidden gradients: noise actor never reads residual/QA_joint/DDIM; residual loss never updates noise actor or DDIM; no `QA_joint(s,a_exec) − QA_base(s,a_base)`; no `min_iQ_i(a_exec) − min_iQ_i(a_base)`.
- No hard clip as the normal composition path; emergency clamp only for roundoff; material violation fails fast.
- Never reconstruct historical w/base/residual/beta/branch/version from `action_exec`.
- Old QM checkpoints and prefill v2 are rejected, not migrated.
- Never modify `dsrl.py`, SB3-common, `env_utils.py`, `p6_launcher.py`, DPPO, flat-RFS, per-step experiments without a documented stop-condition.
- One immutable tagged BASE prefill feeds both matched runs; no two equivalent prefills.

## 7. Deviation-check protocol

After each stage:
1. Re-run the stage gate (tests + manual checks).
2. Compare achieved state against the stage goal verbatim; report any gap.
3. Check frozen parameters are unchanged unless a cited deviation was raised.
4. Check no do-not-do item was violated.
5. Record in this document's stage-status table.
