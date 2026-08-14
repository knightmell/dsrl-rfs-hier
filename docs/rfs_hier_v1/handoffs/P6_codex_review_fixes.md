# P6 — Codex review fixes (C1–C5) + full-buffer sample fix

Status: RESOLVED (2026-08-11). All five codex-review findings were adversarially
verified CONFIRMED and fixed; a final verification pass surfaced one additional
pre-existing bug in the base replay-buffer `sample()` (also fixed). Full test
evidence at the bottom.

Anchor: `RUNBOOK_2P5M_FRESH.md`, `STAGED_DELIVERY_PLAN.md` §7a.

## C1 — Shared prefill artifact concurrent-write race (`p6_runtime.py`)

Two new bugs were introduced by the original fix and caught by the audit; both
are fixed:

1. **Unbound `archive_inode`** — when `_atomic_save_npz` raised, `archive_inode`
   was unbound and the except-handler raised `UnboundLocalError`, masking the
   real error. Fixed with `archive_inode: int | None = None` + `is not None`
   guard. Regression: `test_archive_write_failure_reprops_original_error_and_keeps_peer_archive`.
2. **Metadata-published stranding** — `atomic_write_json` landed the metadata
   JSON, then `_fsync_directory` raised; the old cleanup unlinked the archive,
   leaving a metadata-only path that permanently raised `FileExistsError` on
   the next write. Fixed with the `not metadata_path.exists()` guard so the
   archive is only cleaned when the metadata write never landed. Regression:
   `test_metadata_published_then_fsync_failure_keeps_complete_artifact`.

The atomic write itself uses process-unique temp names (`{name}.{pid}.tmp`),
`os.replace`, directory fsync, and inode-guarded orphan cleanup. The audit also
flagged the load-path temp clean-up race and the tmp-sibling accumulation on
exception as residuals; both are accepted (below).

## C2 — Launcher lock/status ordering race (`p6_job_wrapper.py`)

The wrapper published `status=complete` before removing `.launch.lock`, so a
resumer could observe "running + no lock". The original fix changed the catch
from `FileNotFoundError` to `OSError` so a non-ENOENT unlink failure can never
suppress the terminal status publish. This is strictly stronger than the
baseline and than the first version (which silently swallowed the unlink error).
The wrapper writes a `LAUNCHER_COMPLETE` marker BEFORE the finally block and the
launcher checks `(COMPLETE or LAUNCHER_COMPLETE).exists()` for resume +
`mkdir(exist_ok=False)` for fresh, which neutralize the no-lock+running
micro-window. Regression: `test_wrapper_lock_unlink_failure_still_publishes_terminal_status`.

## C3 — Matched control lacked artifact-level init proof (`p6_runtime.py`)

Control parity fields were hardcoded constants. Fixed: `assert_matched_fresh_init_state_hashes`
now reads `zero_residual_ddim_parity` from the control and hierarchy manifests
via `control_manifest.get("zero_residual_ddim_parity") or {}` (a missing or null
parity block no longer crashes with `TypeError: 'NoneType' not subscriptable`).
Recording remains gated to `init_mode == "fresh"`; the control↔hierarchy module
mapping still produces bit-identical state_dicts (existing test). Regression:
`test_matched_fresh_init_gate_rejects_missing_or_null_parity` in
`tests/test_p6_train_wiring.py`.

## C4 — Fresh manifest mislabeled warmstart (`p6_preflight.py`, `p6_train.py`)

The fresh manifest recorded `network_warmstart=True` and `init_checkpoint_path`
as the string `"None"`. Fix: `network_warmstart` is now a pure recorded label
derived by `_network_warmstart_label(cfg, algorithm) = not _run_is_fresh(...)`;
the step-checks are driven by a separate `legacy_loaded` flag (true only when a
real checkpoint's step count is preserved). Error message rewritten:
"A model constructed at zero interaction steps (fresh init, or a network
warm-start whose counters were reset) must not carry a nonzero step count".
Regression tests: `test_network_warmstart_label_matches_freshness_for_call_site`
(wiring), the two updated regexes in `tests/test_p6_preflight.py`, and the
handoff `P6_cotrain_launch_blockers_fixed.md` §1 was rewritten to the
legacy_loaded-decoupled description.

## C5 — Replay semantic_hash missing pos/full ordering (`hierarchical_replay_buffer.py`)

`semantic_hash(vector_rows=None)` (whole buffer) includes `pos`/`full` in the
digest header; `semantic_hash(vector_rows=X)` (prefix) omits them so the
immutable prefill prefix hashes the same at prefill (non-full) and at resume
(full, wrapped). The docstring no longer overclaims "samples differently" — it
states that `pos` is part of replay semantics and the digest tracks it.

- `test_prefix_semantic_hash_is_rotation_invariant_across_pos_flip` now compares
  whole-buffer at pos=4 vs pos=6 (both non-full) instead of the trivially-true
  whole-vs-prefix comparison.
- New `test_tagged_resume_hashes_agree_across_pickle_restore` (the tagged-path
  resume comparison C5 touches had no integration test). Key corrected fact:
  `pos` counts vector-rows (one per env-step, each holding n_envs transitions),
  and production requires `buffer_size > vector_steps`, so the buffer is never
  full at prefill. The test populates 3 offline rows, verifies pickle
  round-trip preserves both digests, fills the remaining 7 rows online, and
  verifies the offline-prefix digest is preserved (rows [0:3] untouched) while
  the whole-buffer digest tracks full/wrapped pos.

## Bonus — base `ReplayBuffer.sample()` crashed on wrapped-full buffers

Surfaced during final verification: `sample()` sized its probability vector by
`self.pos`, but a wrapped-full buffer resets `pos` to 0 while the number of
valid rows is `buffer_size` → `np.random.choice` raised
`"'a' and 'p' must have same size"` on every sample. This is pre-existing
(committed at HEAD, from the offline-mix feature) and NOT a codex finding, but
it is reachable on the flat `DSRL`/`RFSDSRL` training path (`dsrl.py:243/330`,
`train_dsrl.py:153/181`) once the replay wraps. Fix: size `prob` by
`upper_bound` instead of `self.pos`; identical to prior behavior when non-full,
crash → correct sampling when full. The hierarchy cotrain path is unaffected
(`HierarchyTaggedReplayBuffer.sample` routes to `sample_any`). Regression:
`test_replay_buffer_sample_on_wrapped_full_buffer`; the 8 previously-failing
`test_buffers.py` tests now pass.

## Accepted residuals (with rationale)

- C1 load-path temp-cleanup race and tmp-sibling accumulation on exception —
  bounded, self-healing on next launch, no data corruption.
- C2 reverse-window (status≠complete while unlink in progress) — only visible
  in a failed run; a fresh launch is refused by `mkdir(exist_ok=False)`.
- C2 goal-conditional `complete` — by design, status reflects the declared goal.
- C3 no production caller for the control-parity assert at runtime — the assert
  is cross-run tooling; the module-level mapping is covered by a test.
- Full-SB3 suite: 8 failures are optional-dependency gaps (`tqdm`/`rich` for the
  progress-bar callback, `pygame` for render) — environment, not code.

## Verification

- P6 suite (`tests/`): 162 passed.
- SB3 DSRL suite (phase1–6, `test_dsrl_rfs`, `test_hierarchical_replay_buffer`):
  81 passed.
- SB3 `test_buffers.py`: 18 passed (8 pre-existing failures fixed + new test).
- Full SB3 run (minus the three pre-existing broken-collection files):
  732 passed / 8 optional-dep failures / 7 skipped.
