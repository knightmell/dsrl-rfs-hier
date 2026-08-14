# P6 — Three cotrain launch blockers fixed; 2.5M co-training run is live

Status: RESOLVED (2026-08-08). All three blockers were latent P6 bugs exposed by
the fresh (`network_warmstart`) co-training launch; none are design deviations,
so nothing was registered in `STAGED_DELIVERY_PLAN.md` §7a.

Anchor: `RUNBOOK_2P5M_FRESH.md`, `STAGED_DELIVERY_PLAN.md` §7a.

## 1. `finalize_loaded_model_preflight` rejected `expected_init_checkpoint_steps=0`

- Symptom: `ValueError: p6.expected_init_checkpoint_steps must be >= 1, got 0`
  on the fresh launch.
- Cause: the validator enforced `minimum=1` unconditionally, even though the
  init-step value is informational for any run that builds the model with
  counters at zero (a fresh run, or a network warm-start from a legacy
  checkpoint) — interaction-step counting restarts from zero.
- Fix (`p6_preflight.py`, `finalize_loaded_model_preflight`): the step checks
  are decoupled from the recorded label. `network_warmstart` is now a pure
  recorded manifest flag; the checks are driven by a separate
  `legacy_loaded` flag, which is true only when a real checkpoint's step count
  is preserved. `minimum=(0 if not legacy_loaded else 1)`; a fresh/warm-start
  model must sit at `actual_steps == 0`, a genuinely loaded checkpoint must
  match `actual_steps == expected_steps` exactly. The manifest records
  `init_checkpoint_num_timesteps` / `training_start_num_timesteps` together
  with `network_warmstart` and `legacy_loaded`.
- Regression tests: `tests/test_p6_preflight.py::test_finalize_fresh_accepts_zero_expected_steps`
  (fresh: zero is legal, label False), the legacy-loading cases in
  `test_finalize_loaded_model_preflight_*` (exact step match), and
  `tests/test_p6_train_wiring.py::test_finalize_wiring_flags_network_warmstart_by_init_mode`
  (the p6_train caller derives the label from `init_mode` instead of hardcoding
  True).

## 2. Tagged prefill saved through the standard-format validator

- Symptom: `ValueError: Prefill artifact is missing 'actions'` inside
  `collect_tagged_matched_prefill_and_save` → `save_prefill_artifact`.
- Cause: tagged arrays are keyed by `TAGGED_PREFILL_ARRAY_NAMES` (`action_exec`,
  no `actions`); `save_prefill_artifact` always validated with the standard
  `validate_prefill_semantics`.
- Fix (`p6_runtime.py`): `save_prefill_artifact` gained a `validator=` parameter
  (default standard); `collect_tagged_matched_prefill_and_save` passes
  `partial(validate_tagged_prefill_semantics, prefill_source=prefill_source)`.
  The atomic two-rename + orphan-recovery write is shared by both formats.
- Regression test: `tests/test_p6_runtime.py::test_tagged_prefill_collect_and_save_round_trip`.

## 3. `populate_tagged_replay_buffer` validated a FRESH prefill as WARMSTART

- Symptom: `ValueError: Matched prefill requires immutable noise policy
  version 0` at `populate_tagged_replay_buffer` → `validate_tagged_prefill_semantics`
  with the default `prefill_source=PREFILL_SOURCE_WARMSTART`; a FRESH prefill has
  `noise_policy_version == -1` by spec.
- Cause: the validator call did not thread the source through.
- Fix (`p6_runtime.py`): the validator now reads `metadata["prefill_source"]`
  (stamped by the tagged collector; falls back to WARMSTART for legacy callers).
- Regression test: appended to the same runtime test — populating a
  `HierarchyTaggedReplayBuffer` from the FRESH loaded arrays succeeds and yields
  `branch_counts[0] == 6`.

All four `validate_tagged_prefill_semantics` call sites were audited; each now
passes the correct source (collect 1027, load 1136, save-partial 1215, populate 1231).

## Verification

- Full P6 suite green: 69 passed (`tests/test_p6_runtime.py` ×11 includes the
  two new regression tests).
- Launch: `p6_train.py --config-name p6_hopper_fresh_2p5m_cotrain seed=1
  total_timesteps=2500000` → `training_status: running`,
  `prefill_status: verified_and_loaded` (the immutably seeded prefill artifact
  written by attempt 2 was reused, not regenerated).
- Step-0 exact-N evaluation confirms init identity: reference_base ==
  current_base_only == current_full_hierarchy (raw ~1471/1332, d4rl ~45.8/41.5,
  residual delta L2 = 0, clip_fraction = 0) — the honest baseline to beat.

## Failed run dirs (archived)

`logs/p6/_failed_20260808_1651/1654/1658_cotrain_seed1` correspond to blockers
1/2/3 respectively.
