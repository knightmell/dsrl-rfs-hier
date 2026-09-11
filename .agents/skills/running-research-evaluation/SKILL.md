---
name: running-research-evaluation
description: Run concise checkpoint evaluations for VS-Hier research experiments using the current manifest/config and the project milestone and final evaluation policy.
---

# Running Research Evaluation

Use this skill for saved-checkpoint evaluation. Keep the workflow focused on
the requested checkpoint and parameter correctness; do not repeat a full
training or provenance audit.

## Default schedule

- Do not evaluate checkpoints before 500k unless explicitly requested.
- From 500k onward, evaluate every 100k checkpoint with 10 episodes.
- Evaluate the final 800k checkpoint with 100 episodes instead of 10.
- Do not invent a 100-episode final evaluation for another final budget.
- If a run starts after a milestone, use the next available 100k checkpoint.

## Minimal checks

Read only the relevant registry row, `run_manifest.json`, resolved config, and
the requested checkpoint/evaluation path. Confirm task, seed, clip status, K,
role, transition, completion marker, and evaluator settings. Use the existing
evaluator and manifest evaluation seeds; change only the episode count required
by the schedule. Keep `use_wandb=false` unless explicitly requested.

Distinguish an independently trained `basecontrol` from a full checkpoint
evaluated with residual disabled. The latter must be labelled `checkpoint base
view`, never independent control.

Record canonical name, exact transition, role, episode count, seed set,
raw/normalized return, early-fall rate, finite/non-finite status, output path,
and command/config provenance. Do not modify checkpoints or manifests, rerun
prefill, rebind bundles, or consult archived documents merely to evaluate a
current artifact. If a required artifact is missing or inconsistent, report
the exact blocker and stop.
