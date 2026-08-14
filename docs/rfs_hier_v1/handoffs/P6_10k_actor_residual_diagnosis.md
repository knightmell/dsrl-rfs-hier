# P6 10k noise-actor / residual diagnosis

## Question

Determine whether the 10k hierarchy run has the previously suspected pattern:

1. the trained noise actor produces a worse diffusion base policy;
2. the residual actor recovers part of that loss;
3. identify why the noise actor is allowed to degrade.

No training or source-code modification was performed for this diagnosis.

## Evaluation protocol

- Initial model: official 5M DSRL-NA checkpoint.
- Final model:
  `/home/mrf/dsrl/logs/p6/20260729_hier_init5m_seed1_10k_wiring/checkpoints/final_model.zip`
- Frozen decoder: audited DDIM5 artifact.
- Environment seeds: `10000..10024`.
- Per-episode policy seeds: `20000..20024`.
- Exact 25 complete episodes per policy.
- Same action-chunk termination and normalization as P6 training.
- Both stochastic and deterministic policy evaluation were run.

Policies:

- `init_5m`: original 5M DSRL noise actor and frozen DDIM.
- `base_only_10k`: trained 10k noise actor, frozen DDIM, residual forced to zero.
- `full_10k`: trained 10k noise actor plus trained residual.
- `hybrid`: original 5M noise actor plus the trained 10k residual.

## Main result

### Stochastic SAC policy

| Policy | Raw return | Early-fall rate | Mean primitive length |
|---|---:|---:|---:|
| initial 5M DSRL | 3065.42 | 12% | 974.16 |
| 10k base-only | 2339.84 | 92% | 722.64 |
| initial actor + 10k residual | 3006.04 | 0% | 1000 |
| 10k full hierarchy | 3074.23 | 0% | 1000 |

Decomposition:

- noise-actor/base degradation:
  `2339.84 - 3065.42 = -725.58`
- residual recovery relative to the degraded base:
  `3074.23 - 2339.84 = +734.39`
- final full hierarchy relative to initialization:
  `3074.23 - 3065.42 = +8.81`
- trained noise actor's joint-coordination benefit relative to the hybrid:
  `3074.23 - 3006.04 = +68.19`

### Deterministic actor mean

| Policy | Raw return | Early-fall rate | Mean primitive length |
|---|---:|---:|---:|
| initial 5M DSRL | 3143.35 | 0% | 1000 |
| 10k base-only | 2493.19 | 96% | 766.96 |
| initial actor + 10k residual | 2995.00 | 0% | 1000 |
| 10k full hierarchy | 3034.61 | 0% | 1000 |

The deterministic result proves that the collapse is not only a change in
`log_std` or stochastic evaluation variance. The actor mean itself has moved
to a base policy that is unstable without the residual.

## When the collapse appears

Ten fixed deterministic evaluation seeds were used at every saved milestone:

| Chunk | Base-only raw | Base early fall | Full raw | Full early fall |
|---:|---:|---:|---:|---:|
| 2k | 3021.53 | 20% | 2958.43 | 0% |
| 4k | 2976.32 | 30% | 2970.62 | 0% |
| 6k | 3086.30 | 30% | 2994.91 | 0% |
| 8k | 2399.85 | 90% | 3008.63 | 0% |
| 10k | 2453.43 | 90% | 3035.97 | 0% |

The large return collapse occurs between 6k and 8k. Actor drift is smooth
across this interval; Hopper locomotion crosses a nonlinear stability boundary
rather than showing a single discontinuous parameter jump.

## Common-state actor and critic evidence

On 1,024–2,048 identical warm-start observations:

- total actor parameter relative L2 drift: `4.15%`
- deterministic initial/final noise cosine similarity: `0.23–0.28`
- deterministic noise output delta L2: `2.36–2.47`
- decoded base-action delta L2: `0.45–0.46`
- residual unit mean absolute value: `0.77`
- residual tanh saturation: `17–18%`

The largest relative actor changes are:

- `log_std.weight`: `34.5%`
- `mu.weight`: `15.3%`
- `mu.bias`: `6.7%`

The mean-policy collapse is therefore compatible with a relatively small
global parameter change because the multi-layer policy output and Hopper
dynamics amplify it.

At 10k, on common states:

- QA(final base) - QA(initial base): approximately `-0.24` to `-0.55`
- QA(final executed) - QA(final base): approximately `+1.3` to `+1.8`
- QM(final noise, zero) - QM(initial noise, zero): approximately `+2.3` to
  `+3.1`
- QM(final joint) - QM(final zero): approximately `+0.2` to `+1.1`
- policy-sample QM distillation MSE: approximately `44`
- corresponding RMSE: `sqrt(44) ~= 6.63`

The QM preference that moves the noise actor is smaller than the measured
distillation error scale. Raw relative error is small compared with
`Q ~= 975`, but the error is large compared with the value differences between
candidate latents. This is the relevant signal-to-error comparison.

## Root-cause diagnosis

### 1. The loss optimizes the joint policy, not base-policy quality

The noise actor minimizes:

`alpha * log_prob_noise - QM(observation, noise_scaled, residual_unit)`.

There is no term that preserves:

- the initial DSRL noise actor;
- the quality of `action_base`;
- the quality of the zero-residual policy.

Therefore, base-only degradation is permitted by the current objective whenever
the residual can restore joint executed-action value.

### 2. Noise becomes a coordination latent

The residual is explicitly conditioned on `noise_scaled` and `action_base`.
The hybrid experiment is decisive:

- original noise + trained residual: `3006.04`
- trained noise + trained residual: `3074.23`

The trained noise actor improves compatibility with the trained residual by
about 68 raw return, despite producing a much worse base policy. The system has
learned co-adaptation rather than an independently strong base plus a small
correction.

### 3. QM error is large relative to the actor's optimization margin

QM is a sampler-gradient-free surrogate. It is distilled only at current joint
policy samples. Actor-relevant Q differences are around 1–3, while policy
sample distillation RMSE is around 6.6. The actor can therefore follow ranking
errors even when the normalized critic MSE looks small.

### 4. Replay contains executed actions, not base-only actions

QA is trained correctly on replayed `action_exec`. It does not receive Bellman
supervision for the current policy's counterfactual `action_base`. Likewise, QM
is distilled at `(noise_scaled, residual_unit)`, not at
`(noise_scaled, zero)`. Base-only QA/QM values are consequently diagnostic
extrapolations, not trained objectives.

### 5. Residual optimization rapidly creates dependence

With no residual penalty:

- residual mean absolute value reaches about `0.75`;
- tanh saturation reaches about `17%`;
- action clipping reaches about `8–9%`;
- residual/base L2 ratio reaches about `15%`.

The action delta is numerically residual-sized, but Hopper is sufficiently
sensitive that this delta becomes essential for balance.

## Interpretation

The residual does not merely recover "part" of the actor loss. Under stochastic
evaluation it recovers essentially all of it; under deterministic evaluation
it recovers about 83%. It also eliminates early falls.

This is not automatically an implementation bug. It is a valid solution of the
current joint objective. It is, however, inconsistent with a strong claim that
the method preserves a high-quality diffusion base policy and performs only
local optional correction around it. The trained residual has become a required
stabilizer.

## Required next diagnostics

Before changing losses, run a matched 10k study from the same 5M checkpoint and
prefill:

1. legacy DSRL control;
2. current joint hierarchy;
3. frozen-noise residual-only diagnostic;
4. noise-only/control diagnostic already supplied by legacy DSRL.

For all variants record at init/2k/4k/6k/8k/10k:

- full executed return;
- zero-residual/base-only return;
- hybrid return where applicable;
- early-fall rate and episode length;
- actor KL or pre-tanh Gaussian KL to the initialization;
- deterministic noise cosine/delta to initialization;
- base-action delta to initialization;
- normalized QM distillation RMSE;
- QA/QM rank correlation on held-out current-policy samples;
- critic preference divided by distillation RMSE;
- residual magnitude, saturation and clipping.

Use paired 100-episode final evaluation. The present 25-episode diagnostic is
strong enough to identify the mechanism, but not to approve an algorithm
change.

## Candidate repairs, in experiment order

These are algorithm experiments and should not be mixed into the completed P6
wiring repair.

1. **Freeze the noise actor for a 10k residual-only ablation.**
   This determines whether updating the noise actor is necessary for gains.

2. **Reduce noise-actor update speed.**
   Test one update per train call or a lower noise-actor learning rate while
   retaining the 20/10 QA/QM updates.

3. **Use a staged schedule.**
   First train the residual against a fixed DSRL actor, then allow slow joint
   noise updates after QM achieves an approved signal-to-error ratio.

4. **If preservation is a method requirement, add an explicit trust region.**
   A KL/mean-action anchor to the initialized DSRL actor is more defensible than
   optimizing `QM(noise, 0)`, because the zero-residual QM slice is not
   currently distilled.

5. **Only after the above, consider residual regularization or scale ramping.**
   These address residual dependence but do not directly repair QM-driven
   noise drift.

## Proposed acceptance criteria

If the paper claims base preservation:

- base-only return degradation at 10k must be less than 5%;
- base-only early-fall increase must be less than 5 percentage points;
- residual/base L2 ratio remains below 20%;
- tanh saturation remains below 20%;
- clip fraction remains below 10%;
- full policy must outperform matched continued-DSRL control.

For permitting noise-actor updates:

- QM/QA held-out rank correlation must remain positive and stable;
- actor preference magnitude divided by QM distillation RMSE must exceed 1;
- actor KL/delta to initialization must not grow while held-out full return
  declines;
- the trained joint policy must outperform the frozen-noise residual ablation,
  not only its own degraded base.

The current model fails the base-preservation return and early-fall criteria,
and its QM preference-to-error ratio is below 1.
