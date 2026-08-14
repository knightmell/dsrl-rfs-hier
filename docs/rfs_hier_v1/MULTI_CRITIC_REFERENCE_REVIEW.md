# Multi-Critic Reference Review

Status: Stage 1 design evidence, revised after independent review  
Date: 2026-08-05  
Target: `algorithm=dsrl_na_rfs_hier` Core V1

## 1. Executive conclusion

No reviewed paper proves the exact proposed combination of a frozen DDIM, a latent noise actor, and a deterministic residual actor with three value semantics. The evidence supports the components, not a claim that “more critics must perform better.”

The most defensible design is nevertheless the strict three-logical-critic graph:

```text
QA_base -> QW_base -> noise actor
QA_joint          -> residual actor
```

This is preferable to the two-critic direct-`QW` alternative for this project because it preserves the useful DSRL-NA decomposition: Bellman learning stays in execution-action space, and the latent critic remains a supervised alias bridge around the non-differentiable DDIM. The third logical critic does not form a REDQ-style ensemble; it represents a different continuation policy.

The literature supports five implementation choices particularly strongly:

1. Twin heads and target critics for each Bellman value, with pessimistic target aggregation.
2. Current stochastic noise actor plus target critic for SAC-style targets; no target noise actor.
3. A target deterministic residual actor only in the joint Bellman target, with delayed residual updates.
4. Independent networks and optimizers when the value meanings differ.
5. Replay metadata sufficient to identify the behavior policy and hierarchy branch.

It does **not** justify enabling ranking loss, residual gates, REDQ/TQC ensembles, conservative offline losses, PPO, or shared trunks in Core V1.

## 2. Comparison matrix

| Method | Critic role(s) | Parameter sharing | Targets and actor credit | Replay / update pattern | Transfer to this project | Do not transfer |
|---|---|---|---|---|---|---|
| TD3 | Two estimates of the **same** action value | Usually one module with two independent Q networks; no semantic sharing with actor | Target actor and target twin critics; target minimum; current actor updated less often | Shared off-policy replay; paper recommends delayed policy updates | Twin minimum, target residual actor, delayed residual update, action-gradient preservation while critic parameters are frozen | Target noise actor: the noise actor is stochastic SAC-style, not deterministic TD3 |
| SAC | Two estimates of the same soft action value | Actor and critics normally have separate parameters | Current stochastic actor supplies next action and matching log-prob; target Q supplies value; actor uses current Q | Shared replay; Polyak Q targets; entropy temperature may be learned | Current `pi_w` plus target QA in both Bellman targets; one noise entropy term per current chunk decision | Residual entropy or a second alpha; residual is deterministic in Core V1 |
| REDQ | Ensemble of same-semantic Q values | Ensemble members may share batching, not value meaning | Random-subset minimum in target; high UTD paired with ensemble control of bias | High UTD, common replay, delayed/controlled policy updates | Log branch-specific effective UTD and critic disagreement | A 10-member ensemble or REDQ UTD assumptions; three semantic critics are not a REDQ ensemble |
| TQC | Ensemble of return-distribution quantile critics | Multiple quantile networks | Highest target atoms are dropped to control overestimation | SAC-like replay and target updates | Later option only if measured overestimation remains | Quantile critics and truncation in Core V1 |
| CPO / constrained actor-critic | Reward and cost values have different meanings | Implementations commonly keep reward and cost critics distinct | Actor combines reward improvement and constraint information explicitly | Same transitions can carry reward and cost labels | A critic must have one declared meaning; separate logging and optimizer ownership | Combining `QA_base` and `QA_joint` as if they were reward/cost scalars; their difference is continuation, not objective |
| Recovery RL | Task critic/policy plus a learned safety critic and recovery policy | Separate role-specific modules | Safety critic decides when a recovery policy is used | Shared environment, explicitly routed behavior | Explicit route/lane identity and no hidden credit crossing | A hard residual gate before its score correlates with real paired outcomes |
| Multi-objective / MultiCriticAL | Separate values for distinct tasks or objectives | Separate critics can reduce negative interference; one actor may consume several values | Actor aggregation must be explicit | Often shared observations, task-labelled data | Semantic separation and branch labels | Implicitly mixing incompatible targets in one critic |
| HIRO | Manager and worker have different temporal/action abstractions | Separate policies and critics | Each critic trains its own level; off-policy correction addresses a moving lower policy | Hierarchy-aware replay metadata | Store behavior `w`, policy versions, episode/lane identity; measure policy age | Re-label historical hierarchy actions by guessing them from current policies |
| Residual RL / Residual Policy Learning | Commonly one critic evaluates the composed action | Base is fixed or treated as an external controller; residual actor is separate | Residual actor learns on value of `a_base + delta` | Standard on/off-policy data depending on implementation | Residual conditions on detached base context; evaluate true composed action | It does not solve base-policy value contamination when the base actor also learns |
| Policy Decorator | Residual policy refines a frozen large policy | Frozen base and separate residual | Bounded residual and progressive exploration protect the base behavior | Online data from composed policy | Zero residual init, bounded composition, beta schedule | Claiming its frozen-base manipulation results validate a moving noise actor in locomotion |
| ResFiT | Off-policy residual fine-tuning of a behavior-cloned policy | Separate base/residual; critic warm-up and conservative actor learning rates | Delayed residual learning over combined action | Uses offline/online data and careful warm-up | Critic warm-up, zero initialization, actor delay, checkpoint hygiene | Its exact hyperparameters without reproducing its control and chunk semantics |
| CQL / IQL / uncertainty-weighted offline RL | Conservative or support-aware value estimates | Extra value/uncertainty components depend on method | Actor is prevented from exploiting unsupported Q regions | Designed primarily for static/offline distribution shift | A warning that branch-sparse or stale replay can make actor queries OOD | Conservative losses in Core V1 before an OOD failure is measured |
| PCGrad / multi-task gradient research | Multiple task losses can conflict on shared parameters | Explicitly studies shared parameters | Projects or balances conflicting gradients | Multi-task batches | Prefer disjoint QA trunks and optimizers first | Add gradient surgery to compensate for avoidable sharing |
| PPG | Separates policy and auxiliary/value phases to limit interference | Carefully controlled sharing plus distillation | Policy objective is protected from auxiliary updates | On-policy phased buffers | Supports the general principle of protecting the actor’s credit path | Its on-policy algorithm or auxiliary phase in this off-policy Core V1 |

## 3. Primary-source findings

### 3.1 TD3 and SAC

TD3 attributes deterministic actor-critic instability to function-approximation error and combines twin minimum targets, delayed actor updates, and a target actor. Its official explanation also describes target-policy smoothing as protection against narrow erroneous Q peaks. These ideas support twin `QA_joint`, delayed residual updates, and a target residual actor, but not a target noise actor. The noise actor follows SAC semantics instead. Sources: [TD3 paper](https://arxiv.org/abs/1802.09477), [official TD3 code](https://github.com/sfujim/TD3), [Spinning Up TD3 explanation](https://spinningup.openai.com/en/latest/algorithms/td3.html).

Modern SAC samples a next action from the current stochastic actor, evaluates it with target Q networks, and uses the log-probability of that same sample. This directly supports both proposed Bellman targets:

```text
w_next, logp_next = current_noise_actor(next_observation)
target_value = target_Q(next_observation, action_from_that_same_w_next)
               - alpha * logp_next
```

It rules out sampling from a target noise actor while evaluating entropy under the current actor. Sources: [SAC algorithms and applications](https://arxiv.org/abs/1812.05905), [Spinning Up SAC explanation](https://spinningup.openai.com/en/latest/algorithms/sac.html), [SB3 SAC source](https://stable-baselines3.readthedocs.io/en/master/_modules/stable_baselines3/sac/sac.html).

### 3.2 REDQ and TQC

REDQ makes high UTD viable by combining a large same-semantic Q ensemble with random-subset target minimization. TQC combines distributional critics, ensembles, and truncation of high quantiles. Neither result says that three critics with different continuation policies should be aggregated. Their useful lesson here is operational: report effective updates per new branch transition and disagreement, and do not increase actor updates merely because value updates are numerous. Sources: [REDQ paper](https://arxiv.org/abs/2101.05982), [official REDQ repository](https://github.com/watchernyu/REDQ), [TQC project and official code links](https://bayesgroup.github.io/tqc/).

### 3.3 Constrained and multi-objective RL

Constrained RL routinely distinguishes reward and cost values. OmniSafe’s constrained off-policy actor-critic exposes separate reward/cost critics and targets. Recovery RL similarly learns a safety critic with a distinct decision role. These examples support naming, optimizer, target, and log isolation when values mean different things. They do not support subtracting `QA_base` from `QA_joint`: those critics condition on different future policies, so a cross-critic difference is not an advantage. Sources: [CPO](https://proceedings.mlr.press/v70/achiam17a.html), [OmniSafe constrained actor-Q-critic](https://omnisafe.readthedocs.io/en/stable/model/actor_critic.html), [Recovery RL paper](https://arxiv.org/abs/2010.15920), [Recovery RL project](https://sites.google.com/berkeley.edu/recovery-rl/home).

Multi-objective and multi-task work shows that shared representations can introduce negative interference when targets conflict. MultiCriticAL reports gains from task-specific critics, while PCGrad explicitly diagnoses and modifies conflicting gradients. Core V1 should take the simpler preventative step: do not share the `QA_base` and `QA_joint` trunks or optimizers. Sources: [MultiCriticAL](https://openreview.net/pdf?id=rJvY_5OzoI), [multi-task learning as multi-objective optimization](https://proceedings.neurips.cc/paper_files/paper/2018/hash/432aca3a1e345e339f35a30c8f65edce-Abstract.html), [PCGrad](https://proceedings.neurips.cc/paper_files/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html).

### 3.4 Hierarchical RL

HIRO’s off-policy correction exists because a changing lower-level policy changes the effective action meaning seen by the manager. Frozen DDIM makes this project’s decoder stable, but both `pi_w` and the residual policy move. Saving behavior noise, residual, policy versions, and episode-fixed branch identity is therefore warranted. It is not acceptable to reconstruct a historical hierarchy tuple by running the current policies. Source: [HIRO paper](https://arxiv.org/abs/1805.08296).

### 3.5 Residual RL

Classical residual RL and residual policy learning demonstrate that a fixed conventional or non-differentiable policy can be improved by learning an additive correction. Policy Decorator adds bounded residual actions and a progressive exploration schedule; its project page specifically warns that random early residuals can destroy base success signals and that directly fine-tuning a base policy with a random critic can cause unlearning. Sources: [Residual RL](https://arxiv.org/abs/1812.03201), [Residual Policy Learning](https://arxiv.org/abs/1812.06298), [Policy Decorator paper/project](https://policydecorator.github.io/), [official Policy Decorator repository](https://github.com/tongzhoumu/policy_decorator).

ResFiT’s open implementation is useful for engineering comparisons, but its public issues also illustrate why chunk control and episode reset semantics must be audited rather than copied: one issue questions residuals over absolute action chunks, and another reports a missing base-policy reset during offline-buffer construction. Sources: [ResFiT repository](https://github.com/amazon-far/residual-offpolicy-rl), [absolute-action-chunk issue](https://github.com/amazon-far/residual-offpolicy-rl/issues/3), [base-policy reset issue](https://github.com/amazon-far/residual-offpolicy-rl/issues/5).

These residual results do not isolate a learned base actor from a residual rescue path. That is the specific role of `QA_base -> QW_base` in this project.

### 3.6 Offline conservatism and uncertainty

CQL addresses value overestimation under static-data distribution shift; IQL avoids directly querying unseen actions during offline learning. These are relevant warnings for stale policy versions and branch-imbalanced replay. They are not appropriate default losses for an online Core V1 because they would change the algorithmic question before the separated graph is validated. Sources: [CQL](https://proceedings.neurips.cc/paper/2020/hash/0d2b2061826a5df3221116a5085a6052-Abstract.html), [IQL](https://arxiv.org/abs/2110.06169), [official IQL code](https://github.com/ikostrikov/implicit_q_learning).

### 3.7 Checkpoint, replay, and RNG recovery

Artifact recovery is not an algorithmic detail that can be inferred from a paper. The official or primary implementations were inspected separately:

| Reference implementation | What it saves or exposes | What it does not provide | Transfer decision |
|---|---|---|---|
| [Official TD3](https://github.com/sfujim/TD3/blob/master/TD3.py) | Actor, critic, and their optimizer states; targets are reconstructed from online networks on load | Replay, `total_it` delayed-update phase, Python/NumPy/Torch RNG, and environment state | Network warm-start reference only. Core V1 must save targets and update counters rather than reconstruct them during resume |
| [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) | Model zip plus explicit, separate [`save_replay_buffer/load_replay_buffer`](https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/off_policy_algorithm.py); VecNormalize has a separate artifact | Default model save excludes replay/environment/VecNormalize and does not capture global RNG or simulator state; load defaults to resetting stale last observations | Reuse separate model/replay/normalization artifacts and device/schema handling, but do not call standard SB3 save/load an exact resume |
| [Official REDQ](https://github.com/watchernyu/REDQ) | Agent holds policy, Q ensemble, targets, and optimizers for training | The official path has no complete learner+replay+RNG+environment checkpoint; replay sampling RNG position is not restored | Borrow ensemble/UTD evidence only, not recovery semantics |
| [Policy Decorator](https://github.com/tongzhoumu/policy_decorator) | Its online scripts produce selected residual/value/temperature artifacts around a separately supplied base checkpoint | They do not form a consistent snapshot of all online Qs, optimizers, replay, counters, RNG, and simulator state | Evidence for separating base and residual artifacts, not for exact training resume |
| [ResFiT](https://github.com/amazon-far/residual-offpolicy-rl) | Version/config-aware replay caches and explicit random seeding | Public W&B/run continuation is not a single atomic restoration of learner, optimizer, scheduler, replay position, RNG, and simulator | Borrow provenance/version metadata; replay cache plus run continuation must not be labelled exact resume |
| [Recovery RL](https://github.com/abalakrishna123/recovery-rl) | Role-specific task/safety replay in memory | No complete replay/RNG persistence contract | Supports semantic replay separation only |
| [TQC project](https://bayesgroup.github.io/tqc/) | Project page identifies the author implementation | The linked code's recovery behavior was not reliably accessible during this audit | Draw no checkpoint conclusion from an unavailable implementation |
| [HIRO](https://arxiv.org/abs/1805.08296) | Paper defines hierarchy-aware replay/off-policy correction | It does not define an atomic learner/replay/RNG/simulator recovery protocol | Supports replay semantics only |

No reviewed implementation supplies one atomic snapshot containing all online/target networks, optimizers, replay arrays and cursor, every software RNG, delayed-update counters, vector-wrapper episode state, and simulator physics state. Core V1 therefore distinguishes:

1. **Network warm-start:** migrate selected network/alpha state; recreate optimizers, replay, and RNG streams.
2. **Reset-boundary resume:** restore learner, replay, optimizer, counters, and software RNG, then reset environments and record trajectory discontinuity.
3. **Simulator-exact resume:** additionally restore the simulator plus every environment/wrapper/RNG state before claiming the next transition is reproducible.

Core V1 requires the second mode and permits the third only behind a complete environment adapter. Saving only an initial seed is insufficient: stream position must be captured through [Python `random.getstate`](https://docs.python.org/3/library/random.html#random.getstate), NumPy RNG or [BitGenerator state](https://numpy.org/doc/stable/reference/random/bit_generators/generated/numpy.random.BitGenerator.state.html), [Torch CPU RNG state](https://docs.pytorch.org/docs/stable/generated/torch.get_rng_state.html), [all CUDA RNG states](https://pytorch.org/docs/stable/generated/torch.cuda.get_rng_state_all.html), and each project-owned generator such as the lane RNG.

## 4. Decisions frozen from this review

### Adopt in Core V1

- Twin, independent `QA_base` and `QA_joint`, each with its own target and optimizer.
- Twin `QW_base`, trained only by head-aligned regression to target `QA_base`; no Bellman target and no target `QW`.
- Current `pi_w` for next noise and matching log-prob in both Bellman targets.
- Target residual actor only in the joint Bellman target.
- Current `QA_joint`, with frozen parameters but retained action input gradient, for the residual actor.
- Delayed, low-frequency actor updates and per-actor gradient clipping.
- One tagged physical replay with exact behavior metadata and episode-fixed lanes.
- Branch-specific update counts, effective UTD, policy age, disagreement, and target drift logs.
- Zero residual initialization and a bounded beta schedule.
- Atomic project-owned model/replay/manifest generations and an explicitly labelled reset-boundary resume; no borrowed implementation is treated as simulator-exact.

### Keep diagnostic-only

- Headwise `QA_joint(s, action_exec) - QA_joint(s, action_base)`.
- QW candidate rank correlation, head agreement, top-1 agreement, and pairwise accuracy.
- Historical-versus-current policy age and value error.
- Base/joint critic disagreement and target lag.

### Explicitly exclude from Core V1

- `QM_joint` or any residual-conditioned latent critic used by the noise actor.
- Ranking loss, delta-Q gates, residual dropout, target noise actor, PPO residual, CVaR, CQL/IQL losses, REDQ/TQC ensembles, and shared QA trunks.
- Claims that target critics, delayed updates, or residual bounds solve teacher mis-ranking.

## 5. Evidence limitations

- The three-critic graph is a project-specific synthesis. Its correctness follows from value semantics and gradient isolation; its performance remains empirical.
- Target-`QA_base` teaching of `QW_base` is a stability-motivated inference, not a published DSRL result. Teacher lag and candidate ranking must be measured.
- Independent critics avoid direct gradient interference but split data and compute. Branch-specific learning efficiency must be reported.
- Twin minimum is a pessimistic heuristic, not calibrated uncertainty.
- A bounded residual can still exploit an incorrect local `QA_joint` gradient.
- Exact learner/replay/RNG recovery is a project-specific engineering contract, not a performance claim or a feature established by the cited algorithms.

The Stage 2 implementation should therefore optimize for falsifiability: each value definition, behavior branch, target source, and optimizer owner must be mechanically testable.
