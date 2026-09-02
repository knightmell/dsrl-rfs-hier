## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: run
- Experiment ID: wbsd_g0_matrix_walker_seed1_100_300_500k
- Verification Status: EXECUTED_MATRIX
- Scope: G0 only; no training and no environment interaction

## Result

- Checkpoints: 100k, 300k, 500k
- State bank: one shared, evenly stratified 512-state bank from the immutable
  Walker prefill artifact
- Proposal seeds: `1101, 2202, 3303, 4404`
- Candidate pool: 64 standard-normal draws per state/seed, with prefixes
  `1, 2, 4, 8, 16, 32, 64`
- Common random numbers: identical standard-normal draw hashes and prefix-index
  hashes are recorded across checkpoint/method views
- Per-checkpoint executable smoke: 2 candidates through DDIM, QA target, and QW
- Environment steps: `0`
- Optimizer steps: `0`
- All three checkpoint module hashes: unchanged before/after
- Metrics: finite
- Focused tests after remediation: `9 passed in 0.71s`

## Recorded Contract

The manifest records explicit state IDs, state-bank hashes, proposal seeds, pool
and prefix hashes, probe/test/plan source hashes, start/end UTC timestamps, code
status, runtime, and all load warnings. The Python 3.13/cloudpickle
`lr_schedule` warning is retained as a nonblocking load warning.

## Artifact

- `manifest.json`: complete matrix contract and checkpoint records
- `summary.json`: same immutable execution summary
- `state_bank.npy` and `state_ids.npy`: exact shared state inputs
- `crn_draws.npz`: exact shared proposal draws
- `pools/*.npz`: checkpoint/seed candidate views and hashes
- `EXECUTION_COMPLETE`: completion marker

This is an execution artifact only; G0 PASS/FAIL remains an independent audit
decision. G1 has not been started.
