# VS-Hier Project State

Last updated: 2026-09-09

This is the first document to read for ongoing work. It records the current interpretation, not a history of every phase.

## Current method

The active locomotion line is VS-Hier with a frozen DDIM base policy, a latent noise actor, `QA_base`, supervised `QW_base`, `QA_joint`, and an additive residual actor. The current investigated BASE recipe uses a Gaussian QW teacher source, same-state multi-w with explicit `K`, and an explicit actor clipping setting. Recent experiments commonly use `K=4` and `noclip`, but future runs must still record both values explicitly.

Training boundaries are experimental variables. There is no project-wide default BASE/RES split or final transition budget.

## Corrections now in force

- A full checkpoint evaluated with residual disabled is a `checkpoint base view`, not an independently trained `basecontrol`.
- The known Walker full/control continuation pair branched at 500k. It is a legacy local residual diagnostic and cannot represent the 300k-to-600k paired protocol.
- The known Hopper and HalfCheetah 600k full runs branched from 300k. Independent matching base-control artifacts were not found in the latest audit; do not report their checkpoint base views as base-control runs.
- A config file or launch command is not proof that training ran or completed.

## Current evidence policy

Experiment claims require machine artifacts listed in `docs/EXPERIMENT_REGISTRY.md`. Algorithm and code decisions follow `docs/CURRENT_ALGORITHM.md`, `docs/CURRENT_PROTOCOL.md`, and current source code. Archived documents are not authoritative.

## Maintenance rule

Before launching or summarizing an experiment, update its registry row with the intended canonical name and actual artifact path. After termination, record its terminal state and evaluation evidence in the same row.
