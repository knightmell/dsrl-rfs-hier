# Walker Phase-B QW source diagnosis — Material Passport

Status: **LAUNCH GATE PASS; 250k RESULT GATE PENDING**  
Started: 2026-08-29 16:53 CST  
Task: isolate whether broad Gaussian QW-teacher proposals recover BASE learning.

## Causal contrast

| Arm | `qw_teacher_source` | Per-state K | Formal stop |
|---|---|---:|---:|
| Current-K1 | `current_actor` | 1 | 250,000 chunks |
| Gaussian-K1 | `gaussian` | 1 | 250,000 chunks |

The Gaussian arm samples `randn(B, action_chunk, action_dim)` and then applies
the same action-coordinate scaling used by DSRL before querying `QW(s, w)`.
There is no state expansion, multi-w, ranking loss, SVGD, actor/critic change,
clip change, update-count change, or residual training in this contrast.

Resolved-config comparison is enforced by a test: after removing the source
field and source-derived run paths (`name`, `logdir`, `wandb.run`), the two
resolved configs are exactly equal.

## Frozen formal contract

- Environment: `walker2d-medium-v2`
- Seed: 1
- Declared schedule: B=500k, R=2.0M, J=0
- Intentional stop: 250k, therefore the whole pilot is Phase B
- `n_envs=10`, hierarchy `train_freq=1`, UTD=20
- QA-base/QW/noise/alpha updates unchanged
- QA-joint Phase-B shadow unchanged
- Residual actor expected optimizer steps: exactly 0
- Evaluation/model cadence: 50k
- Replay checkpoint cadence: 250k
- Shared prefill archive:
  `walker2d-medium-v2_fresh_frozen_ddim_env3001_policy4001_nenv10_tagged_v1.npz`

## Provenance hashes

- Source state: `1d6f950e0a02aa171c383625a3ccad4faefb6ba60b04c9497b6176481bfeac80`
- Prefill archive: `369ed05f4d893bac0bc825ac7aec738dcb69f8f044a0e88af57b2195e9f81de2`
- Prefill semantic: `0b23fff6ebe680baa698593ae18b9b423d0b2859c2db07a5fe37d0b6cdb15316`
- Initial replay semantic: `0cfad2aac7563ab0181e51309f5f93bc40ad937fe7ca46d5e59958d01d528c78`
- Initial actor: `be8ec34e3f5c88ee7f46c7baf58e7969edcc8c6c431c979d42671216557b9c7c`
- Initial QA-base/target: `c9a2e465c6c4d5f411e5a7cd48177d528437f0d303bf4a4d660df696b5d7c946`
- Initial QW-base: `1eedfbc59cfbf50a39b976f81d65ba0fc4f5c894af6d5700170c9b9883752443`
- QW implementation: `89a3095d9ec4782779afc45a800ec36d282185d63d7ed356880d628bda266226`
- Runner: `14345e73c57e4483a77b1fada2b66cbb7dafb627e27d247d05bdd7e9896c2ead`
- Preflight: `bf0e7cc7b8d96b45b7e3f38e0f00bd58cbb79294d806240031507a22851a0f79`

The source, prefill, initial replay, actor, QA-base/target, and QW hashes are
identical across both formal manifests.

## Verification before launch

- TDD source/config/wiring tests: `8 passed`
- Relevant regression suite: `91 passed, 3 deselected`
- The three deselections are host-environment tests whose `mujoco_py` build
  lock cannot be written inside the read-only test sandbox; the real host
  MuJoCo smoke below covers runtime loading.
- `git diff --check`: pass
- Gaussian real GPU/MuJoCo smoke: 100 chunks, intentional exit code 75
- Smoke counters: QW=100, QA-base=200, noise actor=200, residual actor=0
- Smoke model and replay resume bundle: complete

## Formal runs

- Current-K1:
  `/home/mrf/dsrl/logs/p6/base_diag_qw_current_k1_pilot250k_fresh_frozen_ddim_walker2d-medium-v2_dsrl_na_rfs_hier_seed1_2500000chunks`
- Gaussian-K1:
  `/home/mrf/dsrl/logs/p6/base_diag_qw_gaussian_k1_pilot250k_fresh_frozen_ddim_walker2d-medium-v2_dsrl_na_rfs_hier_seed1_2500000chunks`
- Launch sessions: Current `81965`; Gaussian `29858`
- Initial GPU check: both active, about 1.88 GiB each

## 250k result gate

Integrity requirements (all mandatory):

1. Both arms reach exactly 250k and have an intentional-interruption marker,
   complete model/replay bundle, and finite diagnostics.
2. Both remain in Phase B with residual actor optimizer steps exactly zero.
3. Update counters and checkpoint/evaluation cadence match; source labels and
   all frozen provenance hashes remain correct.
4. Run held-out fixed-state QW probes on matched states/candidate pools at the
   available checkpoints; probes must not mutate RNG or model state.

Mechanism-support rule:

- Gaussian-K1 improves at least two of three broad-support QW diagnostics:
  Spearman by at least 0.05, pairwise accuracy by at least 0.03, or top-1 by
  at least 0.05 versus Current-K1; and
- Gaussian-K1 improves normalized 0–250k BASE-return AUC by at least 2 D4RL
  points while its 250k endpoint is not worse by more than 1 point.

Interpretation:

- QW and BASE both improve: broad teacher support is a supported root cause;
  only then test whether same-state multi-w adds value.
- QW improves but BASE does not: support repair is real but not sufficient;
  inspect QW-to-actor transfer/clipping before multi-w.
- Neither improves: reject sampling source as the dominant cause and return to
  QW ranking/support diagnostics, actor clipping, and update organization.

This is a one-seed diagnostic gate, not publication-level evidence. No 500k
extension or multi-w experiment is authorized before the 250k audit.
