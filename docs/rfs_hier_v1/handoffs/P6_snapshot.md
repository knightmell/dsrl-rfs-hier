# P6 snapshot handoff

## Snapshot identity

- `ORIGINAL_OUTER_HEAD`: `d16c4f98ab733add5beb6a3de7286428e79f7aaf`
- `P6_BASE_SNAPSHOT`: the commit containing this handoff; resolve via
  `git rev-parse P6_BASE_SNAPSHOT` after the snapshot tag is created.
- Outer source branch: `exp/locomotion-stage0`
- Outer snapshot branch: `wip/rfs-hier-p6-base-snapshot`
- Implementation worktree branch after finalization: `rfs-hier-p6-impl`
- Target worktree: `/home/mrf/dsrl-rfs-hier-impl`

## Submodule commits

- `stable-baselines3`: `b61a0a702a65cac2104d12655824f9c0432c803a`
  - Local preservation branch:
    `wip/rfs-hier-p6-snapshot-sb3`
  - This commit already contains the P1-P5 hierarchy implementation.
  - No uncommitted hierarchy changes existed in the submodule.
- `dppo`: `86ce51834055c02f9013e60dd4c4275606d82df7`
  - Local preservation branch:
    `wip/rfs-hier-p6-snapshot-dppo`
  - No uncommitted hierarchy changes existed in the submodule.

The SB3 commit is local-only and was absent from the configured remote. The
new worktree submodule was therefore populated by fetching the exact object
from `/home/mrf/dsrl/stable-baselines3`, not by copying its working directory.

## Dirty-state classification

The committed outer HEAD and SB3 commit already contain the hierarchy Phase
0-P5 implementation, tests, audit artifacts, training entry, and Phase 6
diagnostic scaffolding.

The following original outer tracked modifications were classified as
unrelated to the hierarchy P1-P5 snapshot and were left uncommitted:

- `cfg/gym/dsrl_halfcheetah.yaml`: flat `RFSDSRL` configuration and buffer
  parameterization.
- `cfg/gym/dsrl_hopper.yaml`: flat `RFSDSRL` configuration, buffer
  parameterization, and the uncommitted `n_envs=10` experiment override.
- `cfg/gym/dsrl_walker.yaml`: flat `RFSDSRL` configuration and buffer
  parameterization.
- `eval_hopper_dsrl_checkpoints.py`: frozen-DDIM checkpoint evaluator
  extension.
- `train_dsrl.py`: flat `RFSDSRL` entry and buffer parameterization.
- `utils.py`: flat `RFSDSRL` logging/prefill branches.

The following stable-baselines3 changes were classified as the separate flat
RFS baseline and left uncommitted:

- `stable_baselines3/__init__.py`
- `stable_baselines3/dsrl/__init__.py`
- `stable_baselines3/dsrl/rfs_dsrl.py` (untracked)
- `tests/test_dsrl_rfs.py` (untracked)

The following DPPO files were classified as frozen-policy video diagnostics
and left uncommitted:

- `hopper_frozen_ddim5.mp4` (untracked)
- `tools/eval_hopper_video.py` (untracked)

No dirty file was deleted, reset, overwritten, or moved. All unrelated state
remains in `/home/mrf/dsrl` and was additionally backed up.

## Backup artifacts

Backup root:
`/home/mrf/dsrl-p6-snapshot-backups/20260728T085216Z`

- Outer tracked binary patch:
  `outer/tracked-working-tree.patch`
  - SHA-256:
    `885cf432ac935ce08958a6a8dd58a26a30b79c9d73e45f23b91cad618d31ddae`
- Outer untracked archive (empty by inventory):
  `outer/untracked-files.tar`
  - SHA-256:
    `84ff92691f909a05b224e1c56abb4864f01b4f8e3c854e4bb4c7baf1d3f6d652`
- SB3 tracked binary patch:
  `stable-baselines3/tracked-working-tree.patch`
  - SHA-256:
    `9c963f3b9abe7bb024c8eb8b54cc99147a214af1e9e5235cc6ebe7fafe9e3593`
- SB3 untracked archive:
  `stable-baselines3/untracked-files.tar`
  - SHA-256:
    `1c933633f20cffeddcf7a0ae5ad466c96baa7d64a85bbf8a93c13f8c0d9dd97c`
- DPPO tracked binary patch (empty because its dirty state was untracked):
  `dppo/tracked-working-tree.patch`
  - SHA-256:
    `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- DPPO untracked archive:
  `dppo/untracked-files.tar`
  - SHA-256:
    `0f78634173a776af7062644302b35c1238cd2d23c74adf933e3044ed5fc8801c`

## Exact tests and results

Clean target hierarchy suite:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
MPLCONFIGDIR=/tmp/dsrl-rfs-hier-snapshot-mpl \
PYTHONPATH=/home/mrf/dsrl-rfs-hier-impl/stable-baselines3:/home/mrf/dsrl-rfs-hier-impl/dppo \
/home/mrf/miniconda3/bin/python -m pytest -p no:cacheprovider \
  stable-baselines3/tests/test_dsrl_rfs_hier_phase1.py \
  stable-baselines3/tests/test_dsrl_rfs_hier_phase2.py \
  stable-baselines3/tests/test_dsrl_rfs_hier_phase3.py \
  stable-baselines3/tests/test_dsrl_rfs_hier_phase4.py \
  stable-baselines3/tests/test_dsrl_rfs_hier_phase5.py \
  tests/test_dsrl_config_entry.py -q
```

Result: `45 passed, 1 warning in 3.27s`.

The previously reported additional two tests live in the excluded, untracked
flat-RFS file `stable-baselines3/tests/test_dsrl_rfs.py`. They cannot exist in
the clean hierarchy-only snapshot without also committing the prohibited flat
baseline. They were re-run in the preserved original dirty workspace:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
MPLCONFIGDIR=/tmp/dsrl-rfs-flat-backup-mpl \
PYTHONPATH=/home/mrf/dsrl/stable-baselines3:/home/mrf/dsrl/dppo \
/home/mrf/miniconda3/bin/python -m pytest -p no:cacheprovider \
  stable-baselines3/tests/test_dsrl_rfs.py::test_rfs_is_a_parallel_algorithm_and_does_not_replace_dsrl \
  stable-baselines3/tests/test_dsrl_rfs.py::test_original_dsrl_training_path_still_runs \
  -q
```

Result: `2 passed, 1 warning in 2.45s`.

This is intentionally reported as `45 clean-snapshot tests + 2 preserved
unrelated-flat tests`, not as 47 tests contained by the target snapshot.

## Verification and unresolved risks

- Target outer worktree was clean after exact submodule checkout.
- Target submodule SHAs match the original committed hierarchy state.
- Core hierarchy implementation and all tracked Phase 0-P6 files match the
  original repository's committed HEAD.
- Original dirty files and untracked files remain present and are recoverable
  from the recorded backup artifacts.
- The clean target excludes the separate flat-RFS baseline by design.
- Reviewer must decide whether the original “47 tests” wording intended to
  require the unrelated flat-RFS baseline. Including it would violate the
  requested hierarchy-only separation; no such inclusion was guessed.
- P6.1 has not started.
