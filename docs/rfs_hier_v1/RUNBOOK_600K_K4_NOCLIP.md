# 600k K4 Gaussian/NoClip Screening Runbook

This branch packages the short screening profile for the current VS-Hier
implementation. It keeps the existing `p6_train.py`, `p6_launcher.py`, and
`p6_job_wrapper.py` entry points unchanged; only the task-specific 600k
configuration files are added.

## Contract

- Tasks: `halfcheetah-medium-v2` and `hopper-medium-v2`
- Seed for the first screening pass: `1`
- Budget: `300000` BASE transitions followed by `300000` RES transitions
- Joint phase: disabled
- QW teacher: Gaussian source, `K=4`, state batch `B=256`, `1024` queries/update
- Actor gradient clipping: disabled
- `n_envs=10`, hierarchy train frequency `1`, UTD `20`
- Evaluation and model checkpoint cadence: every `50000` chunk transitions
- Replay checkpoint cadence: every `300000` chunk transitions

The 600k run intentionally does not set
`p6.stop_after_chunk_transitions=600000`. The preflight contract rejects an
intentional stop at the exact total budget; the natural `total_timesteps`
completion is the valid end condition.

## Configurations

- `cfg/gym/p6_halfcheetah_fresh_600k_cotrain_k4_noclip.yaml`
- `cfg/gym/p6_hopper_fresh_600k_cotrain_k4_noclip.yaml`

## Durable launch

Run from the repository root, with the project environment active:

```bash
PY=/home/mrf/miniconda3/envs/dsrl/bin/python
ROOT=/home/mrf/dsrl

$PY p6_launcher.py \
  --run-dir "$ROOT/logs/p6/fresh_frozen_ddim_600k_k4_noclip_halfcheetah-medium-v2_dsrl_na_rfs_hier_seed1_600000chunks" \
  -- $PY p6_train.py \
  --config-name p6_halfcheetah_fresh_600k_cotrain_k4_noclip \
  seed=1 total_timesteps=600000 use_wandb=false \
  logdir="$ROOT/logs/p6/fresh_frozen_ddim_600k_k4_noclip_halfcheetah-medium-v2_dsrl_na_rfs_hier_seed1_600000chunks"
```

After HalfCheetah reaches its natural `600000` completion and the process has
exited, launch Hopper with the analogous command:

```bash
$PY p6_launcher.py \
  --run-dir "$ROOT/logs/p6/fresh_frozen_ddim_600k_k4_noclip_hopper-medium-v2_dsrl_na_rfs_hier_seed1_600000chunks" \
  -- $PY p6_train.py \
  --config-name p6_hopper_fresh_600k_cotrain_k4_noclip \
  seed=1 total_timesteps=600000 use_wandb=false \
  logdir="$ROOT/logs/p6/fresh_frozen_ddim_600k_k4_noclip_hopper-medium-v2_dsrl_na_rfs_hier_seed1_600000chunks"
```

Do not run the two jobs concurrently on a single host unless host-memory
capacity has been checked. Replay is CPU-resident and a serialized 100k replay
snapshot is roughly 9.7 GB; extra GPU VRAM does not remove this host-RAM
constraint.

## Transfer to another machine

Clone or fetch this branch, then verify the exact code and configuration before
launching:

```bash
git fetch origin
git checkout codex/exp-600k-k4-noclip-hc-hopper
git rev-parse HEAD
sha256sum cfg/gym/p6_halfcheetah_fresh_600k_cotrain_k4_noclip.yaml \
          cfg/gym/p6_hopper_fresh_600k_cotrain_k4_noclip.yaml
```

The frozen DDIM checkpoints, normalization files, and any prefill artifacts
must also exist at the paths resolved by the configs. Copy these immutable
inputs once with checksums; keep replay, checkpoints, and TensorBoard logs on
the local disk of the machine doing the run.

## Cross-machine synchronization

Do not share a live optimizer, replay file, or active run directory over NFS or
rsync. Run independent seeds on independent machines and compare the common
50k checkpoint boundaries. Each run should retain its own manifest with the
Git commit, resolved-config hash, frozen-checkpoint hash, prefill hash, seed,
and host name.

Only synchronize completed small artifacts such as manifests, evaluation CSV/
JSON files, and selected model checkpoints. For a checkpoint handoff, wait for
the `COMPLETE` marker, copy to a temporary destination, verify the manifest and
file hashes, then atomically rename it. Never resume from a partially copied
replay bundle.

True lock-step distributed training is not part of the P6 runtime and is not
recommended here: replay and environment trajectories would still diverge,
while network synchronization would slow the experiment. Independent seed
runs with the same immutable inputs provide the cleaner comparison.
