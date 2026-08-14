# Three-Critic Core V1 Implementation Specification

Status: **Stage 1 REVISE addressed; implementation remains unauthorized until Reviewer PASS**  
Date: 2026-08-05  
Algorithm token: `dsrl_na_rfs_hier`  
Architecture version: `dsrl_na_rfs_hier_three_critic_v1`  
Replay schema version: `hierarchy_tagged_replay_v1`

## 1. Purpose and non-goals

Core V1 replaces the current joint `QA/QM` hierarchy with three logically distinct twin critics:

1. `QA_base(observation, action)` evaluates the supplied current action followed forever by the current noise actor plus Frozen DDIM with residual disabled.
2. `QW_base(observation, noise_scaled)` is a supervised alias bridge to `QA_base(observation, G(observation, noise_scaled))`; it never bootstraps.
3. `QA_joint(observation, action)` evaluates the supplied current action followed by the full current noise actor, Frozen DDIM, and residual hierarchy.

The only actor credit paths are:

```text
QA_base -> QW_base -> noise actor
QA_joint          -> residual actor
```

Core V1 is intended to remove continuation-value contamination and residual rescue credit from the noise actor. It does not claim to solve teacher ranking error, local action-gradient error, lane data inefficiency, or non-stationarity from a moving base.

Core V1 does not enable `QM_joint`, ranking loss, cross-critic delta-Q, hard residual gates, residual dropout, target noise actor, PPO, CQL/IQL/TQC/REDQ, CVaR, shared QA trunks, or extra chunk observations.

## 2. Names and parameter ownership

The implementation must use explicit names. Compatibility aliases may exist only if documented and must point to the same object rather than silently introducing a fourth critic.

| Module | Inputs | Outputs | Optimizer | Target | Actor consumer |
|---|---|---|---|---|---|
| `noise_actor` / inherited `actor` | observation | `noise_scaled`, `log_prob_noise` | `noise_actor_optimizer` | none | — |
| `qa_base` | observation, execution-space action | twin scalar Q | `qa_base_optimizer` | `qa_base_target` | none directly |
| `qw_base` | observation, `noise_scaled` | twin scalar Q | `qw_base_optimizer` | none | noise actor only |
| `qa_joint` | observation, execution-space action | twin scalar Q | `qa_joint_optimizer` | `qa_joint_target` | residual actor only |
| `residual_actor` | observation, detached `noise_scaled`, detached `action_base` | residual logits | `residual_actor_optimizer` | `residual_actor_target` | — |
| `log_alpha` | current noise log-prob | scalar alpha | `alpha_optimizer` | none | noise entropy only |
| Frozen DDIM | observation, decoder noise | `action_base` | none | none | no gradient consumer |

Hard invariants:

- `qa_base`, `qa_joint`, and `qw_base` parameter IDs and storage pointers are disjoint.
- `qa_base` and `qa_joint` do not share feature extractors or trunks.
- `policy_kwargs.share_features_extractor=True` fails fast for this algorithm.
- Every optimizer owns exactly its declared parameters; optimizer parameter sets do not overlap.
- `log_alpha` is owned exclusively by `L_alpha`. Bellman and actor losses use only `alpha_detached = exp(log_alpha.detach())` (or a detached fixed-alpha tensor).
- All target parameters have `requires_grad=False` and remain in evaluation mode.
- DDIM wrapper and every reachable PyTorch base-policy module remain in evaluation mode with `requires_grad=False` and unchanged state hash.

Although each logical critic has twin heads, this must be described as three value meanings, not as a six-member uncertainty ensemble.

## 3. Public forward and episode lanes

For every chunk-level observation:

\[
w\sim\pi_w(\cdot\mid s),
\qquad
a_b=G_{DDIM}(s,\operatorname{sg}(w)).
\]

`noise_scaled` is the SAC actor’s squashed/scaled coordinate. The existing audited Torch affine unscale converts it to decoder input. Both the decoder input and decoder result are detached.

### Base lane

For the complete episode:

\[
a_{exec}=a_b.
\]

The residual actor is not called to decide behavior. Stored residual logits, unit, and delta are exact zero tensors; `residual_applied=false`.

### Joint lane

For the complete episode:

\[
z_r=f_r(s,\operatorname{sg}(w),\operatorname{sg}(a_b)),
\qquad
a_{exec}=C(a_b,z_r,\beta).
\]

### Lane allocator

- `branch_mode` is `BASE=0` or `JOINT=1`.
- A lane is assigned with a dedicated, persisted lane RNG only at an episode reset.
- The lane recorded on a terminal transition is the lane of the episode that produced the transition. Reassignment affects only the next action after reset.
- Default post-base-stage probability is `base_lane_probability=0.5`; it is configurable and recorded.
- Core V1 uses episode-random assignment. No primitive- or chunk-level switching is allowed within an episode.
- Training is skipped with a clear metric until the requested branch contains the configured minimum number of samples.
- Actual branch transition counts and effective per-branch UTD are authoritative; configured probability is not treated as the realized ratio.

The two lanes share model parameters but are different simulator trajectories. The implementation must never describe them as two actions executed from the same physical state.

## 4. Bound-preserving action composition

Let:

\[
u=\tanh(z_r),\quad
m_+=a_{high}-a_b,\quad
m_-=a_b-a_{low}.
\]

The desired forward map is:

\[
\delta a=
\begin{cases}
\beta m_+u,&u\ge0,\\
\beta m_-u,&u<0,
\end{cases}
\qquad
a_{exec}=a_b+\delta a.
\]

Direct `where(z_r >= 0, m_+, m_-)` chooses an arbitrary upper-margin derivative at exactly zero and gives zero autograd derivative when DDIM lands exactly on the upper bound. Core V1 must use the forward-equivalent expression:

```python
residual_unit = torch.tanh(residual_pre_tanh)
margin_positive = exec_action_high - action_base
margin_negative = action_base - exec_action_low
action_residual_delta = beta * (
    0.5 * (margin_positive + margin_negative) * residual_unit
    + 0.5 * (margin_positive - margin_negative) * residual_unit.abs()
)
action_exec_unclamped = action_base + action_residual_delta
```

For nonzero `residual_unit`, this is exactly the piecewise margin formula. At zero, PyTorch’s zero subgradient for `abs` yields:

\[
\frac{\partial\delta a}{\partial z_r}\Big|_{z_r=0}
=\beta\frac{a_{high}-a_{low}}{2},
\]

which is centered and nonzero for non-degenerate bounds whenever `beta > 0`, including when `action_base` equals one bound.

Required checks:

- Constructor and schedule enforce finite `0 <= beta <= 1`.
- Execution bounds have the exact flattened action shape and match the audited environment wrapper bounds.
- `action_base` outside bounds by more than numeric tolerance is a pre-training/runtime error, not silently repaired.
- An emergency final clamp is permitted only for floating-point roundoff; its activation count and maximum pre-clamp violation are logged. A material violation fails fast.
- `z_r=0` produces bitwise/equality-tolerance `action_exec==action_base` and `delta==0`.
- `beta=0` intentionally has zero residual gradient during the base-only phase.

## 5. Replay schema

One hierarchy-specific physical replay buffer is used. Common SB3 `ReplayBuffer` remains unchanged.

### 5.1 Stored arrays

| Field | Dtype / shape per transition | Meaning |
|---|---|---|
| `observation` | environment dtype, observation shape | observation used by behavior |
| `next_observation` | same | true terminal observation when the VecEnv autoresets |
| `reward` | `float32[1]` | current chunk’s undiscounted primitive-reward sum |
| `done` | `bool[1]` | terminated or truncated |
| `terminated` | `bool[1]` | true MDP terminal |
| `truncated` / `timeout` | `bool[1]` | time-limit transition, eligible for bootstrap |
| `branch_mode` | `uint8[1]` | `BASE` or `JOINT` behavior lane |
| `noise_scaled` | `float32[action_dim_flat]` | exact SAC-coordinate behavior sample |
| `noise_log_prob` | `float32[1]` | exact behavior log-prob when defined |
| `noise_log_prob_valid` | `bool[1]` | false for uniform/warmup/external samples |
| `noise_sample_source` | `uint8[1]` | current actor, reference actor, Gaussian prior, or uniform warmup |
| `transition_origin` | `uint8[1]` | online interaction or tagged prefill; separate from action source |
| `action_base` | `float32[action_dim_flat]` | exact Frozen-DDIM behavior output |
| `residual_pre_tanh` | `float32[action_dim_flat]` | exact behavior residual logits; zero in BASE |
| `residual_unit` | `float32[action_dim_flat]` | exact `tanh(logits)`; zero in BASE |
| `action_residual_delta` | `float32[action_dim_flat]` | exact execution-coordinate delta; zero in BASE |
| `action_exec` | `float32[action_dim_flat]` | exact action sent to ActionChunkWrapper |
| `beta` | `float32[1]` | behavior beta |
| `residual_applied` | `bool[1]` | false in BASE, true only when the joint residual was evaluated/applied |
| `emergency_clamp_applied` | `bool[1]` | numeric safety path indicator |
| `episode_id` | `int64[1]` | unique within run and environment stream |
| `environment_id` | `int32[1]` | vector environment slot |
| `chunk_index_in_episode` | `int32[1]` | zero-based chunk decision index |
| `nominal_primitive_steps` | `uint8[1]` | configured chunk length, Hopper=4 |
| `actual_primitive_steps` | `uint8[1]` | primitives actually executed |
| `termination_primitive_index` | `int8[1]` | `-1` if none, otherwise zero-based slot |
| `termination_semantics` | `uint8[1]` | audited legacy-continue or early-break enum |
| `noise_policy_version` | `int64[1]` | version that sampled behavior noise |
| `residual_policy_version` | `int64[1]` | version that produced behavior residual |

All values describe what actually happened. No training code may reconstruct a historical `w`, base action, residual, beta, branch, or policy version from `action_exec`.

External prior/warmup samples use `noise_policy_version=-1`; BASE transitions, where the residual actor is not called, use `residual_policy_version=-1`. Those sentinels are legal only with the corresponding source/applied flags and are validated on insertion.

The decoder input is not stored because it is a deterministic audited affine transform of stored `noise_scaled`; parity is tested at insertion and in diagnostics. Margins are deterministic from stored base and immutable bounds.

### 5.2 Branch sampling

The buffer stores branch arrays alongside standard SB3 arrays and exposes:

```text
sample_branch(BASE, batch_size)
sample_branch(JOINT, batch_size)
sample_any(batch_size)       # diagnostics only in Core V1
```

Sampling is over flattened `(time_index, environment_index)` slots. The implementation may use vectorized rejection sampling plus exact per-branch counts; it must:

- update counts correctly when the circular buffer overwrites a row;
- sample only currently valid slots;
- bound rejection attempts and fail informatively if a branch is unavailable;
- preserve sampling-with-replacement semantics;
- reconstruct/validate counts after pickle load;
- include every metadata array in semantic replay hashing.

Core V1 data use is strict:

| Update | Replay observations/transitions |
|---|---|
| `QA_base` | BASE only |
| `QW_base` regression states | BASE only |
| noise actor / alpha states | BASE only |
| `QA_joint` | JOINT only |
| residual actor states | JOINT only |

Cross-lane reuse is an interface placeholder only and defaults off. In particular, the reward from a joint transition cannot be paired with its unexecuted base action to train `QA_base`.

### 5.3 Prefill compatibility

- Existing P6 prefill v2 is incompatible because it lacks exact hierarchy metadata.
- Every matched control/hierarchy pair must generate or load **one and the same** immutable `hierarchy_tagged_replay_v1` BASE-prefill artifact. Generating two nominally equivalent prefills is forbidden.
- The hierarchy consumes the complete tagged schema. The control consumes the standard projection of that exact artifact:

  ```text
  observation
  corrected_next_observation
  action_exec
  reward
  done
  timeout
  ```

  These six projected arrays are not regenerated, cast, reordered, or re-sampled. Their per-array SHA-256 values and their ordered aggregate semantic hash must be bitwise identical in the control and hierarchy manifests.
- The profile `legacy_dsrl_warmstart_5m` records the warm-start DSRL behavior as BASE with actual `noise_scaled`, valid behavior log-prob, `action_base==action_exec`, and exact-zero residual fields.
- The profile `fresh_frozen_ddim_5m` records Frozen-DDIM Gaussian-prior behavior as BASE. Its exact decoder sample is mapped back through the audited affine `scale_action` to the stored `noise_scaled`; the inverse must reproduce the decoder input within `1e-6`. `noise_log_prob_valid=false` because this sample was not drawn by the SAC actor.
- Prefill transitions seed replay but do not advance the B/R/J online-schedule counter. Prefill interaction and primitive-action counts remain separate manifest fields and are identical for the matched pair.
- Loading v2 into the hierarchy fails with a migration message; fields are never synthesized.

### 5.4 Terminal and timeout insertion truth

For environment slot `i`, insertion is derived only from the synchronous VecEnv result and its `info`:

```text
done_i       = bool(dones[i])
timeout_i    = done_i and bool(infos[i].get("TimeLimit.truncated", False))
terminated_i = done_i and not timeout_i
truncated_i  = timeout_i
```

`timeout_i` without `done_i` is invalid; when both names are physically stored, `truncated` and `timeout` must be bitwise identical boolean arrays. `done`, `terminated`, `truncated`, and `timeout` are canonicalized to `np.bool_` once at artifact creation, before projection hashing. Before either the tagged artifact or replay semantic hash is computed:

```text
if done_i:
    next_observation_i = infos[i]["terminal_observation"]
else:
    next_observation_i = vecenv_next_observations[i]
```

A done transition without `terminal_observation` fails rather than hashing an autoreset observation. Bellman sampling exposes:

\[
d_{train}=done\,(1-timeout)=terminated.
\]

The terminal transition retains the producing episode's ID, lane, chunk index, policy versions, beta, and termination metadata. A new episode ID/lane is attached only to the first action after reset.

## 6. Mathematical losses

Entropy ownership is explicit. At the start of every loss block:

\[
\alpha_{sg}=\exp(\operatorname{sg}(\log\alpha))
\]

for learned temperature, or a detached constant tensor for fixed temperature. Both Bellman targets and `L_noise` use only `alpha_sg`; they must not create or leave a gradient on `log_alpha`. Only `L_alpha` owns `log_alpha`.

Define the replay training terminal mask:

\[
d_{train}=done\,(1-timeout).
\]

This preserves SB3 time-limit bootstrapping. Reward is the current code’s chunk reward and `gamma` appears once per chunk.

### 6.1 Base action critic

Sample a BASE transition `(s,a_b,r,s',d_train)`. Under one `torch.no_grad()` target branch:

\[
w'\sim\pi_w(\cdot\mid s'),
\qquad
(w',\log\pi_w(w'\mid s'))\text{ are the same sample},
\]

\[
a_b'=G_{DDIM}(s',\operatorname{sg}(w')),
\]

\[
y_b=r+\gamma(1-d_{train})
\left[
\min_i\bar Q^{base}_{A,i}(s',a_b')
-\alpha_{sg}\log\pi_w(w'\mid s')
\right].
\]

Each online head regresses to the shared pessimistic target:

\[
L_{QA\_base}=\frac12\sum_i
\mathbb E_{D_{base}}
\left(Q^{base}_{A,i}(s,a_b)-y_b\right)^2.
\]

Future residual is structurally absent: no residual module is called in this target.

### 6.2 Base latent critic

Core V1 samples fresh current-policy noise on BASE-lane observations under `no_grad`:

\[
w\sim\pi_w(\cdot\mid s),
\qquad a_b=G_{DDIM}(s,w).
\]

This source matches actor queries and avoids training QW only on stale policy versions. Stored behavior tuples are used for exact behavior diagnostics, not relabelled as current samples. Gaussian-prior and behavior/current mixtures are disabled in Core V1.

The target teacher is head-aligned target `QA_base`:

\[
q_{teacher,i}=\operatorname{sg}
\bar Q^{base}_{A,i}(s,a_b),
\]

\[
L_{QW\_base}=\frac12\sum_i
\mathbb E_{s\sim D_{base},w\sim\pi_w}
\left(Q^{base}_{W,i}(s,w)-q_{teacher,i}\right)^2.
\]

`QW_base` has no Bellman target and no target network. Regression is per head; both heads must not be collapsed onto the minimum teacher.

The original DSRL code instead uses standard-Gaussian prior noise and the online QA teacher. This intentional Core V1 difference must appear in manifest/config and be covered by a sample-source test. Ranking candidates from current policy, reference policy, and local perturbations are diagnostic-only.

### 6.3 Noise actor and entropy coefficient

On BASE-lane observations:

\[
(w,\log\pi_w)=\pi_w(s),
\]

\[
L_{noise}=\mathbb E
\left[\alpha_{sg}\log\pi_w(w\mid s)
-\min_iQ^{base}_{W,i}(s,w)\right].
\]

During this forward, `QW_base` parameters are frozen but its input gradient remains enabled. The loss graph contains no DDIM, residual actor, `QA_base`, or `QA_joint` forward.

Automatic entropy tuning retains the legacy target-entropy meaning, including Hopper’s explicit `target_entropy=0.0`:

\[
L_{\alpha}=-\mathbb E
\left[\log\alpha\;\operatorname{sg}
(\log\pi_w+H_{target})\right].
\]

There is no residual entropy.

When alpha and noise are both enabled, Core V1 preserves the legacy ordering within each paired update:

1. draw one current noise sample and compute its log-probability;
2. snapshot `alpha_sg = exp(log_alpha.detach())`;
3. zero alpha gradients, backpropagate `L_alpha` using detached log-probability, and step `alpha_optimizer`;
4. zero noise-actor gradients, backpropagate `L_noise` using the **pre-alpha-step** `alpha_sg` snapshot and the same actor sample, then step the noise optimizer;
5. clear both owners' gradients before leaving the block and assert that unrelated parameter gradients are absent.

When alpha is frozen, steps 3 and its counter increment are skipped. Bellman targets independently snapshot detached alpha when their no-grad target branch begins.

### 6.4 Joint action critic

Sample a JOINT transition `(s,a_exec,r,s',d_train)`. Under one `torch.no_grad()` target branch:

\[
(w',\log\pi_w(w'\mid s'))\sim\pi_w(\cdot\mid s'),
\]

\[
a_b'=G_{DDIM}(s',w'),
\]

\[
z_r'=\bar f_r(s',\operatorname{sg}(w'),\operatorname{sg}(a_b')),
\qquad
a_{exec}'=C(a_b',z_r',\beta_{current}),
\]

\[
y_j=r+\gamma(1-d_{train})
\left[
\min_i\bar Q^{joint}_{A,i}(s',a_{exec}')
-\alpha_{sg}\log\pi_w(w'\mid s')
\right].
\]

Then:

\[
L_{QA\_joint}=\frac12\sum_i
\mathbb E_{D_{joint}}
\left(Q^{joint}_{A,i}(s,a_{exec})-y_j\right)^2.
\]

No target noise actor is created. `beta_current` is the persisted schedule value at update time; behavior beta remains stored for diagnostics.

### 6.5 Residual actor

On JOINT-lane observations, sample current noise without a noise-actor gradient and decode a detached current base:

\[
w\sim\pi_w(\cdot\mid s),\quad
w=\operatorname{sg}(w),\quad
a_b=\operatorname{sg}(G_{DDIM}(s,w)).
\]

The online residual produces `z_r`, and composition produces `a_exec`. Freeze online `QA_joint` parameters while retaining its action-input gradient:

\[
L_{residual}=-\mathbb E
\min_iQ^{joint}_{A,i}(s,C(a_b,f_r(s,w,a_b),\beta)).
\]

Core V1 has no residual entropy, BC, or L2 penalty. Noise actor, both base critics, DDIM, and all targets must receive zero gradients.

Headwise diagnostic only:

\[
\Delta Q_i=Q^{joint}_{A,i}(s,a_{exec})
-Q^{joint}_{A,i}(s,a_b).
\]

No `QA_joint-QA_base` subtraction is defined or logged as a gain.

## 7. Curriculum, beta, and update cadence

Core V1 uses a persisted chunk-transition scheduler. Prefill does not advance this scheduler. Let `c` be the number of completed **online** chunk transitions before the next vectorized environment step. All phase boundaries and the run budget must be divisible by `n_envs`; preflight rejects a split vector batch.

### 7.1 Frozen scheduler keys and profiles

| Key | Type / unit | Frozen default | Validation |
|---|---|---:|---|
| `schedule_profile` | enum | `fresh_frozen_ddim_5m` | explicit in manifest; P6 warm-start config must override to `legacy_dsrl_warmstart_5m` |
| `phase_b_chunk_transitions` | non-negative int, online chunks | profile value | divisible by `n_envs` |
| `phase_r_chunk_transitions` | positive int, online chunks | profile value | divisible by `n_envs` |
| `phase_j_enabled` | bool | `false` | Core V1 default is disabled |
| `phase_j_chunk_transitions` | non-negative int, online chunks | `0` | exactly zero when disabled; positive and divisible by `n_envs` when enabled |
| `beta_target` | finite float | `0.1` | `0 <= beta_target <= 1` |
| `beta_ramp_chunk_transitions` | non-negative int, online chunks | profile value | zero or at least `2*n_envs`; divisible by `n_envs`; no greater than Phase R length |
| `base_lane_probability` | float | `0.5` | strictly between zero and one in R/J |
| `min_branch_replay_transitions` | positive int | `256` | must be `>= batch_size`; changing batch size above 256 requires an explicit matching override |

The two named profiles are part of the algorithm contract, not launcher suggestions:

| Profile | B | R | J enabled / length | beta ramp | beta target | Post-B BASE probability | Prefill source |
|---|---:|---:|---:|---:|---:|---:|---|
| `fresh_frozen_ddim_5m` | 2,500,000 | 2,500,000 | false / 0 | 50,000 | 0.1 | 0.5 | Frozen DDIM + recorded Gaussian decoder prior |
| `legacy_dsrl_warmstart_5m` | 0 | 5,000,000 | false / 0 | 50,000 | 0.1 | 0.5 | explicitly supplied legacy DSRL actor |

Thus Core V1 does **not** enter Phase J by default. An experiment that enables J is a manifest-visible schedule override and must set a positive J length while preserving:

```text
declared_online_chunk_budget = B + R + J
```

Smoke tests may use a shorter named `test_override` only when every scheduler field is supplied explicitly and the manifest marks `test_cadence_override=true`; production defaults must be revalidated after the smoke.

### 7.2 Exact boundaries and beta indexing

The phase for the next vector batch is:

```text
B if 0 <= c < B_end
R if B_end <= c < R_end = B + R
J if phase_j_enabled and R_end <= c < B + R + J
```

For a zero-length B profile, B-to-R initialization occurs once during model setup before the first online action. Otherwise it occurs after exactly `B` completed online transitions and before generating the first R action.

Within R, define vector-batch index and ramp length:

```text
k = (c - B) / n_envs
K = beta_ramp_chunk_transitions / n_envs
```

Then:

```text
beta = beta_target                                  if K == 0
beta = beta_target * k / (K - 1)                    if K >= 2 and 0 <= k < K
beta = beta_target                                  if k >= K
```

Consequently the first ramp batch uses exactly `beta=0`, the final ramp batch uses exactly `beta_target`, and later R/J batches retain `beta_target`. One beta value is shared by all slots in a vector step and stored per transition. Phase B always stores `beta=0`.

Global phase is assigned per transition batch, not per episode. If an episode crosses B-to-R or R-to-J, its transition uses the phase/beta active when that action was generated, but its existing `branch_mode` remains unchanged. At B-to-R, all pre-existing episodes are BASE and remain BASE until their next reset; only reset environments draw from the post-B lane distribution. At R-to-J, existing BASE/JOINT lanes also remain unchanged. The phase, lane and beta used by each action are stored independently.

### Phase B: establish base

```text
branch assignment: BASE only
beta: 0
QA_base: 20
QA_joint: 0
QW_base: 10
noise actor: 20
alpha: 20
residual actor: 0
```

This profile preserves the original DSRL optimizer-step counts when `gradient_steps=20` and legacy `actor_gradient_steps=-1`. It is still not bitwise the original algorithm because Core V1 deliberately uses a target-`QA_base` teacher and current-policy QW candidates instead of the legacy online-QA/Gaussian-prior distillation. That difference is explicit in config and manifest.

### Phase R: residual ramp, noise frozen

At the B-to-R boundary:

- copy current `QA_base` head-for-head into `QA_joint`;
- hard-copy `QA_joint` into `QA_joint_target`;
- hard-copy online residual into target residual;
- discard and recreate `qa_joint_optimizer` with an empty state using the configured joint-critic learning rate;
- preserve lifetime counters but set `qa_joint_optimizer_steps_since_clone=0`, increment `qa_joint_generation`, and record the boundary counter and generation in checkpoint/manifest;
- activate episode-level BASE/JOINT lanes;
- freeze noise actor and alpha updates;
- ramp beta using the exact indexing above.

No QA-joint or residual optimizer step occurs until the JOINT replay population reaches `min_branch_replay_transitions`. The corresponding BASE updates independently require the BASE population threshold. A skipped block increments a skip metric, not an optimizer counter.

Default update profile:

```text
QA_base: 10
QA_joint: 10
QW_base: 5
noise actor: 0
alpha: 0
residual actor: 1
```

### Phase J: disabled by default; explicit separated fine-tuning only

Only when `phase_j_enabled=true`, beginning exactly at `R_end`; it does not restore the old coupled graph:

```text
QA_base: 10
QA_joint: 10
QW_base: 5
noise actor: 1
alpha: 1
residual actor: 1
```

Noise still reads only `QW_base`, and residual still reads only `QA_joint`.

The update counts above are the frozen defaults for their phase. Overrides are configuration- and manifest-visible. Manifests record requested, skipped and realized optimizer steps. The runner logs:

\[
effective\_UTD_{base}=\frac{QA\_base\ optimizer\ steps}{new\ BASE\ transitions},
\]

\[
effective\_UTD_{joint}=\frac{QA\_joint\ optimizer\ steps}{new\ JOINT\ transitions}.
\]

### Target updates

- Each `QA_base` optimizer step is followed, at the configured target interval, by one Polyak update of `QA_base_target`.
- Each `QA_joint` optimizer step is followed similarly by one Polyak update of `QA_joint_target`.
- Each residual optimizer step is followed by one Polyak update of `residual_actor_target`.
- QW and the noise actor have no targets.

The Polyak unit is therefore an optimizer step, matching original DSRL. Logs include target update counts and effective cumulative retention `(1-tau)^N`; a block-level single update must not silently replace the legacy cadence.

Noise and residual actors use distinct configurable gradient clipping, default `max_norm=1.0`. Pre- and post-clip norms are logged separately. Critics use separate optimizers; no shared global clipping call is allowed.

## 8. Initialization and migration

### 8.1 Fresh run from Frozen DDIM

- Load and authenticate the same Frozen DDIM and normalization artifacts as the matched DSRL run.
- Initialize noise actor, `QA_base`, and `QW_base` from the same seeded constructors as DSRL.
- Hard-copy online `QA_base` to `QA_base_target`.
- Defer the current `QA_base -> QA_joint` clone until Phase R activation if Phase B is nonzero.
- Zero the residual output layer exactly and hard-copy it to target residual.
- Create all optimizers fresh.
- Save an immutable reference noise-actor snapshot for evaluation.

### 8.2 Network warm-start from legacy DSRL-NA

Use `_LegacyLoadableDSRL.load`; never parse the zip manually.

Copy:

```text
legacy actor       -> noise actor and immutable reference actor
legacy online QA   -> QA_base
legacy QW          -> QW_base
legacy alpha       -> alpha
QA_base            -> QA_base_target (hard copy)
QA_base            -> QA_joint
QA_joint           -> QA_joint_target (hard copy)
zero residual      -> target residual (hard copy)
```

The legacy QA target is validated for shape/artifact identity but is not inherited; the latest architecture contract requires target QA to be a hard copy of online QA. Optimizer state is not inherited. All new optimizer states begin empty.

Acceptance at step zero:

- residual logits/unit/delta are exactly zero;
- `action_exec` equals legacy DSRL decoded action for identical observation and noise within `1e-6`;
- `QA_base` equals legacy online QA headwise;
- `QW_base` equals legacy QW headwise within `1e-6`;
- `QA_joint` equals `QA_base` exactly;
- both QA targets equal their online networks exactly;
- alpha equals legacy alpha;
- DDIM state hash is unchanged.

### 8.3 Old hierarchy checkpoints

Old `critic_modulation`/QM hierarchy checkpoints are not semantically migratable. Loading them must raise a versioned error that explains:

```text
This checkpoint uses the deprecated joint QA/QM hierarchy.
Start three-critic Core V1 from the corresponding legacy DSRL-NA checkpoint.
```

No partial `strict=False` load is allowed.

## 9. Save/load and resume contract

Model checkpoint state must include:

- noise actor and immutable reference actor;
- online/target `QA_base`;
- `QW_base`;
- online/target `QA_joint`;
- online/target residual actor;
- all five network optimizers plus alpha optimizer when learned;
- alpha/target entropy;
- architecture and replay schema versions;
- optimizer, target, policy-version, phase, and train-call counters;
- current phase and beta schedule state;
- lane RNG, monotonic `next_episode_id`, and (only for simulator-exact mode) active branch/episode/chunk state for every env.

Replay checkpoint must include all arrays, circular state, branch counts, and offline/prefill boundary. Runtime bundle must retain Python, NumPy, Torch CPU/CUDA RNG and bind to DDIM, normalization, execution bounds, config, outer/submodule commits plus a sorted source fingerprint, prefill full/projection hashes, and replay semantic hash.

Policy versions increment only after their corresponding optimizer step succeeds. Stored transitions record the version used before any later update.

The certified save boundary is after a complete vectorized chunk step has returned, terminal observations and all tagged transitions have been stored, the scheduled optimizer/target blocks and counters have completed, and before any next action/noise/lane sample is drawn. No mid-gradient, partially inserted vector batch, or partially executed ActionChunk checkpoint is valid.

Two resume modes are frozen:

### 9.1 `reset_boundary` resume — required default

This mode is used whenever the simulator plus every vector/wrapper state is not serialized.

- Restore online/target networks, optimizers, alpha, replay, schedule and policy versions, completed chunk/primitive counters, global RNGs, the dedicated lane RNG at its post-last-draw state, and monotonic `next_episode_id`.
- Do **not** restore `_last_obs`, active lane, active episode ID, chunk index, partial episode return/length, pending action chunk, or any other state belonging to the abandoned simulator episodes.
- Increment persisted `environment_discontinuity_count` once. For each slot in ascending `environment_id`, derive a reset seed as the first 32 bits of `SHA256("three_critic_reset_v1:{train_env_seed}:{environment_discontinuity_count}:{environment_id}")`; record the complete seed list and reset every environment with it. This derivation consumes no training RNG.
- In the same ascending slot order, allocate a fresh never-before-used episode ID and draw a new lane from the restored lane RNG; set `chunk_index_in_episode=0`. The lane distribution is the one active at the restored global phase. No synthetic terminal transition is inserted and completed interaction counters are not changed.
- The manifest records `environment_resume_mode=reset_boundary`, resume chunk, reset seeds, abandoned episode IDs/count, `environment_discontinuity_count`, and that paired trajectory continuity is not claimed.
- The first post-resume transition for every slot must carry the new episode ID, chunk index zero, and the lane predicted by replaying the restored lane-RNG draw order.

### 9.2 `simulator_exact` resume — optional capability, fail-fast otherwise

This mode is legal only if the environment adapter serializes and restores the complete simulator, VecEnv, wrapper, action-chunk, environment/action-space RNG, last observation, active lane/episode/chunk, and partial episode statistics. In that case all active state is restored and the next transition must be numerically identical under a deterministic resume test. Requesting this mode without a complete adapter fails preflight.

`reset_boundary` is an exact learner/replay/RNG resume with an explicitly reset environment; it must never be labelled bitwise trajectory or simulator-exact resume.

## 10. Logging and diagnostics

### Required training metrics

- losses: `qa_base`, `qa_joint`, `qw_base`, noise actor, residual actor, alpha;
- headwise/current/target Q means, scales, TD errors, and disagreements for both QA meanings;
- headwise QW regression MSE and target-teacher lag;
- target-online drift and target update counts;
- all optimizer counters, policy versions, phase, beta, branch counts, realized lane ratio, policy age, and effective UTD;
- noise mean/std/log-prob/entropy/alpha;
- residual logits/unit/delta norms and saturation;
- positive/negative action margins, emergency clamp count, bound violation magnitude;
- actor pre/post-clip gradient norms;
- headwise same-`QA_joint` delta-Q;
- BASE and JOINT episode returns, lengths, early falls, and primitive/chunk counters.

### QW ranking diagnostic only

On held-out BASE observations, use an isolated RNG stream to generate candidates from current noise actor, reference noise actor, and small local perturbations. Report:

- target `QA_base` twin head ordering agreement;
- QW-versus-teacher Spearman/Kendall correlation per head;
- pairwise preference accuracy;
- top-1 agreement;
- teacher gap distribution;
- ranking metrics by policy age.

No ranking term enters any loss. Diagnostics must restore global Python/NumPy/Torch CPU/CUDA RNG and all module modes exactly.

## 11. Evaluation modes

All modes use the same exact-N environment seeds and independent policy seeds:

| Mode | Noise actor | Residual | Purpose |
|---|---|---|---|
| `current_base_only` | current | forced zero | exposes whether base itself improves/collapses |
| `current_full_hierarchy` | current | current | actual hierarchy behavior |
| `reference_base` | immutable initialization snapshot | zero | fixed within-run reference |
| `matched_dsrl` | matched baseline checkpoint/model | absent | external control |

Each result reports raw return, D4RL score, complete-episode return, length, early-fall rate, chunk transitions, nominal primitive steps, and actual primitive steps. Full-hierarchy performance must never substitute for current-base reporting.

## 12. Automated test plan

### 12.1 Graph and construction

- Exactly three logical twin critics exist; QM does not.
- Critic/actor/target parameter sets and optimizer sets match the ownership table.
- `QA_base` and `QA_joint` storage pointers do not overlap.
- `share_features_extractor=True` fails fast.
- All targets and DDIM are frozen/eval.
- `critic_backup_combine_type` accepts only `min`.

### 12.2 Composition

- Zero logits give exact base action and zero delta.
- Autograd derivative at zero equals `beta*(high-low)/2` when `beta>0`.
- Positive/negative finite differences match the piecewise margins.
- Random base/logit/beta tensors remain within per-dimension bounds.
- Exact lower/upper-bound bases have a centered nonzero zero-logit subgradient.
- Invalid beta, bounds, shapes, material base OOB, and material emergency clamp fail.

### 12.3 Lane and replay

- Lane never changes inside an episode; terminal record keeps old lane; next action may use new lane.
- BASE never calls/applies residual and stores exact zeros.
- Joint metadata is the exact behavior tuple.
- Multi-env insertion, circular overwrite, branch counts, filtered sampling, pickle round-trip, and count reconstruction are correct.
- Every metadata field is included in semantic hash; one-field tampering fails resume.
- Legacy prefill v2 is rejected; one tagged BASE artifact feeds both matched runs; every standard projected array and aggregate projection hash is bitwise identical.
- Done insertion replaces autoreset output with `terminal_observation` before hashing; `timeout=done and TimeLimit.truncated`, `terminated=done and not timeout`, and timeout-without-done fails.
- Policy versions increment and restore correctly.

### 12.4 Bellman semantics

- `QA_base` target never calls residual and uses target base QA.
- `QA_joint` target calls target residual, not online residual.
- Current next noise and log-prob are the same sample in each target.
- Both QA losses use identical reward scaling, gamma, entropy count, terminal mask, timeout bootstrap, and chunk semantics.
- Bellman and noise-actor backward leave `log_alpha.grad is None`; only `L_alpha` changes alpha, and the paired alpha-first/noise-second update uses the pre-alpha-step detached coefficient.
- QW teacher is target `QA_base`, head-aligned, current-policy sampled, and no-grad.
- QW has no Bellman or target network.

### 12.5 Gradient/parameter-change matrix

After backward/step, expected parameter changes are:

| Loss/update | noise | alpha | QA base | QW base | QA joint | residual | DDIM | targets |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `L_QA_base` | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| `L_QW_base` | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 |
| `L_QA_joint` | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 |
| `L_alpha` | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| `L_noise` | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `L_residual` | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
| QA Polyak | 0 | 0 | 0 | 0 | 0 | 0 | 0 | matching QA target only |
| residual Polyak | 0 | 0 | 0 | 0 | 0 | 0 | 0 | residual target only |

Tests inspect both gradients and before/after state dictionaries. Noise loss must not even call residual/QA_joint/DDIM. Residual loss must retain a nonzero `dQ_joint/da_exec` while critic parameters remain unchanged.

### 12.6 Initialization, mode, and checkpoint

- Fresh and legacy mappings satisfy all step-zero equivalences.
- B-to-R re-clone occurs once at the recorded boundary.
- Online modules enter train mode only during their own block; targets/DDIM remain eval.
- Save/load produces deterministic-equivalent actions, all Q heads, targets, alpha, beta, versions, counters, lanes, optimizer states, and replay samples.
- Deprecated hierarchy checkpoints give the explicit version error.
- DDIM hash remains unchanged across a complete train/update/save/load cycle.
- Both frozen schedule profiles pass exact boundary tests: first/last B and R batches, zero-length B initialization, first/final ramp beta, default-disabled J, vector-boundary divisibility, and `B+R+J` budget equality.
- B-to-R clone recreates an empty QA-joint optimizer, increments generation once, preserves lifetime counters, and resets only the since-clone counter.
- A phase boundary never changes an active episode's lane; the next reset uses the new phase's lane distribution.
- `reset_boundary` resume restores learner/replay/RNG/counters but discards active environment metadata; reset seeds match the frozen SHA-256 derivation and its first post-resume row has a fresh episode ID, chunk zero, and the expected lane RNG draw.
- A mock complete-state adapter validates `simulator_exact`; requesting exact mode without such an adapter fails preflight.

### 12.7 Environment, evaluator, and baseline regression

- Terminal observation correction matches SB3.
- Timeout bootstraps; true terminal does not.
- H=4 reward/gamma/entropy convention matches original DSRL.
- All four evaluation modes collect exact N complete episodes, preserve RNG/modes, and are batch-size independent.
- Original `algorithm=dsrl_na` configs, construction, save/load, rollout, and regression tests remain unchanged.
- Direct hierarchy entry cannot use missing-metadata legacy prefill.
- P6 preflight/counter/resume/hash tests use the new versioned contract.

## 13. Explicit Stage 2 file list

### Required code/config files

- `stable-baselines3/stable_baselines3/dsrl/hierarchical_rfs_dsrl.py`
- new `stable-baselines3/stable_baselines3/dsrl/hierarchical_replay_buffer.py`
- `train_dsrl.py`
- `utils.py` only for hierarchy logging/prefill fail-fast
- `cfg/gym/dsrl_hopper.yaml`
- `cfg/gym/p6_hopper.yaml`
- `p6_train.py`
- `p6_runtime.py`
- `p6_preflight.py`
- `p6_checkpointing.py`
- `p6_evaluation.py`

### Required tests

- all five existing `stable-baselines3/tests/test_dsrl_rfs_hier_phase*.py` files
- `tests/test_dsrl_config_entry.py`
- `tests/test_p6_runtime.py`
- `tests/test_p6_preflight.py`
- `tests/test_p6_checkpointing.py`
- `tests/test_p6_evaluation.py`
- `tests/test_p6_train_wiring.py`
- optionally a new focused `stable-baselines3/tests/test_hierarchical_replay_buffer.py` rather than overloading phase tests

### Must remain unchanged unless implementation proves an unavoidable conflict

- `stable-baselines3/stable_baselines3/dsrl/dsrl.py`
- common SB3 replay/off-policy/SAC code
- `env_utils.py`
- `p6_launcher.py` and job wrapper
- DPPO
- flat RFS files
- per-step PPO/QA experiments

Any need to modify an expected-unchanged file is a stop condition for Stage 2 and requires reviewer-visible justification.

## 14. Known risks retained after Core V1

- Target `QA_base` can still rank candidate decoded actions incorrectly; QW may faithfully copy that error.
- `QA_joint` can still provide a wrong local action gradient to the residual actor.
- Target residual stabilizes bootstrap but adds lag.
- Current-policy-only QW candidates trade legacy Gaussian-prior coverage for on-policy relevance.
- Episode lanes divide interaction data; one branch may learn more slowly.
- During Phase J, the residual’s conditioning distribution moves as the noise actor changes, even though credit is separated.
- Margin composition removes ordinary hard-clip aliasing but does not guarantee useful residual directions.
- Twin disagreement is not calibrated uncertainty and delta-Q remains diagnostic.
- A successful smoke proves wiring only, not return improvement.

## 15. Stage 1 REVISE resolution index

| Review item | Frozen resolution |
|---|---|
| S1-01 schedule | Sections 7.1–7.2 define both 5M profiles, all keys/defaults, vector boundaries, beta indexing, episode straddling, B-to-R optimizer reset, and default-disabled J |
| S1-02 resume | Sections 9.1–9.2 separate reset-boundary from simulator-exact resume and forbid stale active metadata after reset |
| S1-03 alpha | Sections 2 and 6 define detached alpha for every non-alpha loss and freeze alpha-first/noise-second ownership/order |
| S1-04 prefill/terminal | Sections 5.3–5.4 mandate one tagged BASE artifact, bitwise control projection, terminal-observation correction, and unique done/timeout truth |
| S1-05 provenance | `MULTI_CRITIC_REFERENCE_REVIEW.md` Section 3.7 documents recovery limits; `THREE_CRITIC_CODE_AUDIT.md` Section 1.1 fingerprints the dirty source bytes |

Reviewer PASS on this document authorizes Stage 2 implementation only. It does not authorize 100k, 500k, or 5M training.
