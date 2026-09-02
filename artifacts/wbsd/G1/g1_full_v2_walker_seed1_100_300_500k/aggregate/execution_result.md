# WBSD G1 execution result

- Status: **G1_FAIL**
- Support gate: **FAIL**
- Direct-selection gate: **BLOCKED**
- Scope: Walker seed-1 checkpoints 100k/300k/500k, 512 fixed states, four CRN proposal seeds.
- Mutations: zero optimizer steps, zero backward calls, zero environment steps; all module hashes unchanged.

## Family decisions

| Family | Support | Passing checkpoints | Direct selection |
|---|---:|---:|---:|
| g_current | FAIL | 0/3 | BLOCKED |
| g_prior_exact | FAIL | 0/3 | BLOCKED |
| g_mix | FAIL | 0/3 | BLOCKED |

The exact prior is a teacher-query control and is never marked behavior-executable.
Gate calculations and every per-K condition are stored in `summary.json`.
The corrected interpretation and independent reproducibility audit are in
`G1_DIAGNOSIS.md`; this v2 directory is the canonical G1 result.
