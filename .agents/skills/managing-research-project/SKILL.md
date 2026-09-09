---
name: managing-research-project
description: Use when working in a research project on experiments, training, evaluation, algorithm changes, results, project status, run naming, documentation, or historical evidence.
---

# Managing Research Project

Treat each project's living documents and machine artifacts as project memory. Apply this skill across repositories; project-local instructions and current artifacts define the concrete filenames and conventions.

## Read first

First discover the project root and its current instructions. When the project uses the standard living documents below, always read `docs/PROJECT_STATE.md`, then only what the task needs:

- training, continuation, evaluation, or result claims: `docs/EXPERIMENT_REGISTRY.md` and `docs/CURRENT_PROTOCOL.md`;
- algorithm or code changes: `docs/CURRENT_ALGORITHM.md` and `docs/CURRENT_PROTOCOL.md`;
- environment setup: `README.md` and `docs/ENVIRONMENT_SETUP.md`.

If those files do not exist, use the project's documented equivalents. Do not invent a parallel documentation hierarchy during ordinary work. Ask before bootstrapping new project-management files when no canonical documents exist.

Do not read archive or legacy-plan directories during ordinary work. Consult them only when current documents and run artifacts cannot identify a path, budget, or historical decision. Archived text is a lead, never direct evidence. Report the uncertainty and ask the user to confirm before treating it as fact.

## Evidence

For experiment facts, prefer the actual `run_manifest`, config, checkpoint or bundle, evaluation, and terminal marker. Distinguish an independently trained `basecontrol` from a full checkpoint evaluated with residual disabled. Never infer a completed run from a config or command alone.

For code and algorithm constraints, the current living documents and current source override archived plans, handoffs, and old P6 gates.

## Maintain instead of multiplying

Update the existing living documents after material changes. Do not create a new phase, handoff, status, or plan document unless the user explicitly requests a separate artifact. Git history preserves earlier versions.

Use the project's canonical naming tool when one exists. In VS-Hier repositories, use `project_run_naming.py`; branch and final transitions must be explicit, with no assumed default boundary or total budget.

For any training launch, resume, monitoring, or stop diagnosis, **REQUIRED SUB-SKILL:** use `running-research-training`.
