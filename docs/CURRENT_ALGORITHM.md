# Current VS-Hier Algorithm Contract

Last updated: 2026-09-15

This document is the current human-readable algorithm contract. Current source and tests provide executable evidence; archived design documents do not override it.

## Components and credit paths

For observation `s`, the latent actor samples `w`, and the frozen DDIM decoder produces the base action:

```text
w ~ pi_w(. | s)
a_base = G_DDIM(s, stop_gradient(w))
```

The value roles are separate:

```text
QA_base(s, a)  -> execution-space BASE value
QW_base(s, w)  -> supervised bridge from QA_base for the latent actor
QA_joint(s, a) -> value of the full base-plus-residual behavior
```

The intended actor credit paths are:

```text
QA_base -> QW_base -> latent actor
QA_joint          -> residual actor
```

`QW_base` is trained from `QA_base` labels. The teacher latent source, same-state multiplicity `K`, state batch, query count, clipping, and every UTD value are explicit run parameters rather than global defaults.

## Operational objective and phase semantics

The equations below state the credit-assignment contract; they do not imply a
single fixed schedule for every experiment. With frozen DDIM decoder `D` and
chunk-terminal indicator `d`:

```text
w ~ pi_w(. | s)
a_base = D(s, stop_gradient(w))
delta = beta(t) * tanh(pi_res(s, stop_gradient(w), stop_gradient(a_base)))
a_exec = clip(a_base + delta, action_low, action_high)
```

`QA_base` is fitted on the base-execution pathway and supplies the labels used
to fit `QW_base`; conceptually its TD target is
`r + gamma_chunk (1-d) QA_base_target(s', a_base')`. The latent actor receives
credit through `QW_base(s,w)`, not through `QA_joint`. `QA_joint` evaluates
the executed action and has conceptual target
`r + gamma_chunk (1-d) QA_joint_target(s', a_exec')`; its actor-side credit
goes only to the residual policy. Twin critics, target networks, entropy
terms, and exact update counts are implementation details recorded per run.

```text
BASE credit:  QA_base -> QW_base -> pi_w -> frozen DDIM -> a_base
RES credit:   QA_joint              -> pi_res          -> delta -> a_exec
```

In a run's BASE portion, `beta=0`, so the executed and base actions coincide.
In its RES portion, the configured BASE updates continue and the residual lane
adds its own updates. The branch transition, beta hold/ramp, BASE/RES UTD,
lane ratio, and any BASE-phase QA_joint shadow updates are explicit run-level
parameters; they must never be inferred from the directory name.

## Residual composition

The residual receives the observation plus detached base quantities and produces a bounded correction:

```text
delta = bounded_residual(s, stop_gradient(w), stop_gradient(a_base), beta)
a_exec = a_base + delta
```

The composition must preserve environment bounds. BASE learning must not silently lose update volume merely because residual learning is enabled; actual optimizer counters and effective per-lane UTD are the evidence.

## Current empirical choices

Gaussian QW teacher sampling, `K=4`, and `noclip` are the current locomotion choices supported by the recent diagnostic direction. They are not baked into the naming helper or imposed on future tasks. The branch point, final budget, residual UTD, beta schedule, and lane allocation remain experiment-specific and must be recorded in config and manifest.

## Comparison boundaries

`VS-Hier full` means the complete three-critic method above. An independently
trained `basecontrol` shares the declared branch state and BASE schedule, but
does not optimize the residual actor. A full checkpoint evaluated with its
residual disabled is a **checkpoint base view**—a deployment diagnostic, not
an independent base-control result.

The external baselines deliberately have different optimization rules:

- **matched DSRL:** a flat DSRL-style baseline with the same frozen diffusion
  source and explicitly recorded interaction accounting;
- **DPPO / DIPO:** diffusion-policy fine-tuning baselines whose counters are
  primitive environment actions rather than P6 chunk decisions;
- **Policy Decorator (matched):** frozen DDIM plus bounded SAC residual. Its
  residual-use probability rises from 0 to 1 in the first 100k chunk
  decisions, while the correction bound remains fixed at 0.1.

Equal chunk interaction budgets do not prove equal optimizer compute,
teacher-query count, or wall-clock cost. Every result must state the budget
being matched and retain the method-specific update settings.

## Authoritative implementation surfaces

- training and model wiring: `p6_train.py`
- runtime and resume state: `p6_runtime.py`, `p6_checkpointing.py`
- branch construction: `p6_branching.py`
- evaluation: `p6_evaluation.py`
- behavior tests: `tests/test_p6_*.py` and focused experiment-contract tests
