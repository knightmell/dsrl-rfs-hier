## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: run
- Experiment ID: wbsd_g0_smoke_seed1_100k
- Verification Status: EXECUTED_SMOKE
- Scope: G0 only; no training and no environment interaction

## Result

- Command: `PYTHONPATH=/home/mrf/dsrl:/home/mrf/dsrl/dppo:/home/mrf/dsrl/stable-baselines3 python wbsd_probe.py --device cpu --state-count 1 --candidates 2 --output artifacts/wbsd/G0/g0_smoke_seed1_100k`
- Checkpoint: Walker seed-1 100k
- Model load: successful on CPU
- Observation shape: `17`
- Candidate shape: `[1, 2, 24]`
- DDIM/QA/QW candidate queries: `2`
- Environment steps: `0`
- Optimizer steps: `0`
- Metrics finite: yes
- Module state hashes before/after: identical
- Focused tests: `6 passed in 0.70s`

## Anomalies

- The first smoke attempt exposed an unregistered historical Hydra `${now:...}`
  resolver. The standalone harness registered a deterministic placeholder and
  the rerun completed.
- `ruff` is not installed in the current environment; syntax compilation and
  pytest passed. No training process was started.

## Artifact

See `manifest.json`, `summary.json`, and `metrics.csv` in this directory. This
is an execution artifact only; G0 PASS/FAIL remains an independent audit
decision.
