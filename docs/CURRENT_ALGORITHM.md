# Current VS-Hier Algorithm Contract

Last updated: 2026-09-09

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

## Residual composition

The residual receives the observation plus detached base quantities and produces a bounded correction:

```text
delta = bounded_residual(s, stop_gradient(w), stop_gradient(a_base), beta)
a_exec = a_base + delta
```

The composition must preserve environment bounds. BASE learning must not silently lose update volume merely because residual learning is enabled; actual optimizer counters and effective per-lane UTD are the evidence.

## Current empirical choices

Gaussian QW teacher sampling, `K=4`, and `noclip` are the current locomotion choices supported by the recent diagnostic direction. They are not baked into the naming helper or imposed on future tasks. The branch point, final budget, residual UTD, beta schedule, and lane allocation remain experiment-specific and must be recorded in config and manifest.

## Authoritative implementation surfaces

- training and model wiring: `p6_train.py`
- runtime and resume state: `p6_runtime.py`, `p6_checkpointing.py`
- branch construction: `p6_branching.py`
- evaluation: `p6_evaluation.py`
- behavior tests: `tests/test_p6_*.py` and focused experiment-contract tests
