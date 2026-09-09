---
name: running-research-training
description: Use when configuring, launching, resuming, monitoring, stopping, or diagnosing a machine-learning training run in a research project.
---

# Running Research Training

Prioritize correct parameters and a real running process. Keep checks proportional to the training risk.

**REQUIRED COMPANION SKILL:** Use `managing-research-project` for current state, evidence, naming, and registry updates.

## Authorization routing

An explicit user request to launch, resume, monitor, stop, or diagnose training authorizes the ordinary in-scope operations needed for that request. Proceed without a second conversational confirmation when the action is reversible and confined to the selected experiment, including creating its output directory, invoking the configured launcher, writing logs/checkpoints, and starting its requested monitor.

If the runtime permission system requires approval for an in-scope training operation, immediately submit the narrowest accurate approval request. Explain the exact training need and target; do not first ask the user to repeat their intent. The agent cannot approve its own request, claim that approval was granted, or bypass the permission system.

Training intent does not authorize destructive or materially broader actions. Request confirmation and alert the user before deleting or overwriting artifacts, killing unrelated processes, changing drivers or system services, installing system packages, publishing externally, incurring new cost, or weakening security boundaries. Never use broad approval prefixes when a narrower training command is sufficient.

When approval is denied or a permission boundary blocks execution, report the exact blocked command and impact once. Do not loop, silently substitute another environment, or engineer around the boundary.

## Before launch

Read `docs/PROJECT_STATE.md`, `docs/CURRENT_PROTOCOL.md`, and the relevant registry row. Resolve and show the effective values that determine the experiment:

- task, seed, `clip|noclip`, K, role, explicit branch transition, and explicit final transition;
- QW source, state batch, query count, BASE/RES UTD, beta/lane settings, environment count, checkpoint/evaluation cadence;
- fresh or resume mode; for resume, exact source path and source transition;
- canonical run name from `project_run_naming.py` and output directory.

For a paired full/base-control experiment, verify that both branches preserve the intended identical BASE configuration and differ only in the declared residual/joint training behavior.

Do not assume default branch or final transitions. Do not restart warm-up, reset optimizer/replay/RNG, or silently migrate a resume bundle unless explicitly requested.

## Necessary checks only

Run the smallest checks that establish parameter correctness and launchability. Prefer resolved-config inspection and one focused contract test. Do not run broad test suites, repeated hash audits, historical gates, or unrelated smoke tests by default.

Hard blockers are limited to conditions that would make the requested run wrong or unsafe: missing/invalid parameters, incomplete or incompatible resume state, output collision, unavailable requested CUDA device, or measured resources likely to OOM.

If any other rule or preflight constraint blocks training, stop and report immediately:

1. the exact constraint and where it comes from;
2. why it blocked this run and whether it protects correctness or is legacy policy;
3. the smallest options: keep, relax, or remove it.

The user decides whether the constraint remains. Do not spend time engineering around it or repeatedly retrying before that decision.

CUDA availability must be checked in the same execution environment that will launch training. A host `nvidia-smi` result does not prove that a container or sandbox has `/dev/nvidia*`, and a sandbox failure does not prove the host driver is broken. Report the boundary precisely and launch only from an environment that can access the requested device.

## Launch proof and reporting

After launch, verify promptly that the durable process exists and transition progress has begun. A returned launcher command alone is not proof of training.

Every launched run must remain accounted for. Report immediately if it fails to start, exits unexpectedly, stalls, hits OOM, or is blocked. Record one terminal state: `COMPLETE`, `INTERRUPTED`, or `FAILED`, with latest observed transition and reason. Never leave a started run without a status report.

Monitoring should read process state, latest transition/checkpoint, concise logs, and necessary GPU/host resources. Do not turn routine monitoring into a fresh provenance audit.

Update `docs/EXPERIMENT_REGISTRY.md` when the run starts and when its status materially changes.
