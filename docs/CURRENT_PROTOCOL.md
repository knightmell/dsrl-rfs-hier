# Current Experiment Protocol

Last updated: 2026-09-09

This is the only current operational protocol. Old P6 runbooks and gate documents are archived.

## Run naming

Canonical format:

```text
{task}_s{seed}_{clip|noclip}_k{K}_{branch_k}k_{final_k}k_{full|basecontrol}
```

Examples:

```text
hopper_s1_noclip_k4_300k_600k_full
hopper_s1_noclip_k4_300k_600k_basecontrol
walker_s3_clip_k8_175k_925k_full
```

Use lowercase `noclip`; `npclip` is treated as a typo, not a second status. `branch_k` is the actual shared checkpoint where the pair diverges. `final_k` is the intended final number of chunk transitions. Neither value has a default. Use `project_run_naming.py` to build or parse names.

Historical directories are not renamed automatically because their paths may be embedded in manifests and resume chains. Give them a canonical alias in the experiment registry.

## Paired branch semantics

At the declared branch checkpoint, full and base-control must share the same model, optimizer, replay, RNG, source hash, and BASE update schedule.

- `full`: preserves the agreed BASE updates and adds the configured residual or joint updates.
- `basecontrol`: preserves the same BASE updates and performs no residual actor or joint critic optimization.

Evaluation-time residual disabling produces only a checkpoint base view.

## Launch and reporting

A launch is blocked only by a condition that can invalidate or corrupt the run: missing or incomplete resume source, incompatible state, output-path collision, invalid resolved config, unavailable requested CUDA device, or unsafe resources likely to cause OOM. Names are validated for parseability but are not coupled to a historical schedule or fixed budget.

Every launch immediately records `RUNNING`. Every process termination records one of `COMPLETE`, `INTERRUPTED`, or `FAILED`, including the latest transition and concise reason. A launched run may never disappear without a terminal report.

Monitoring checks process liveness, transition progress, new checkpoints, terminal status, and material resource failure. Routine monitoring does not re-audit old gates or historical provenance.

Before launch, present the resolved experiment-defining parameters: task, seed, clip status, K, role, branch/final transitions, QW source/query budget, BASE/RES UTD, beta/lane settings, environment count, resume source, and output path. Parameter correctness is the first gate.

Use only focused checks needed for that run. Broad test suites, repeated hash audits, and historical P6 gates are not default launch requirements. If another constraint blocks training, report the exact constraint, its source and purpose, and the options to keep, relax, or remove it; the user decides before further work.

## Documentation

Update the living documents rather than creating another phase or handoff document. Do not consult `docs/archive/` unless current run artifacts and living documents cannot resolve a historical path, budget, or decision. Archived content must be confirmed by current artifacts or by the user before use.
