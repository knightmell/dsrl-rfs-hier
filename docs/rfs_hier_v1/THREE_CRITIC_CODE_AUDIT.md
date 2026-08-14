# Three-Critic Code Audit

Status: Stage 1 read-only audit, source fingerprint revised after independent review  
Date: 2026-08-05  
Repository: `/home/mrf/dsrl`

## 1. Audited source state

| Repository | Branch | HEAD | State |
|---|---|---|---|
| outer `/home/mrf/dsrl` | `exp/locomotion-stage0` | `18d5be67c78694527b52599c4aaeafaf6e432590` | dirty |
| `stable-baselines3` | `feature/rfs-hier-p6` | `076e5563ceb51ff91c6c7743501d18f0456b13af` | dirty |
| `dppo` | `main` | `86ce51834055c02f9013e60dd4c4275606d82df7` | dirty |

This audit describes the files as they existed on that dirty working tree, not only the committed SHAs. No training code was changed during Stage 1.

Hierarchy/P6-related tracked changes are present in `train_dsrl.py`, `utils.py`, the `p6_*.py` runner stack, Hopper/P6 configuration, P6 tests, and the SB3 hierarchy implementation/tests. Unrelated or excluded state includes untracked flat-RFS files in SB3 and untracked DPPO media/tools. The three-critic implementation must not import, stage, delete, or otherwise depend on those flat-RFS or DPPO changes.

The previously recorded test result `116 passed, 17 warnings, 0 skipped` belongs to an earlier P6 snapshot. It is useful regression history but is not evidence that the future three-critic implementation passes.

### 1.1 Audited working-tree fingerprint

The committed SHAs above are insufficient because the inspected source is dirty. The following manifest is sorted by logical repository and UTF-8 path and records SHA-256 over the exact worktree bytes inspected on 2026-08-05. `M` means tracked-modified, `C` tracked-clean at the recorded HEAD. Stage 1 documentation files are excluded to avoid a self-referential fingerprint.

| Repo | Status | Path | Worktree SHA-256 |
|---|---:|---|---|
| outer | M | `cfg/gym/dsrl_hopper.yaml` | `2f302e2e71cb5b76ed7f29c256d4810138d779bd649cca7f15b697aa94596596` |
| outer | M | `cfg/gym/p6_hopper.yaml` | `ed4a9e0e82fc373f6964444b4566c9077f59be2af781565337a219afc1115eeb` |
| outer | C | `env_utils.py` | `5746a6137ff4c6407b82542d8a46fa8e519442442e73d704b74954677307d1bd` |
| outer | M | `p6_checkpointing.py` | `a4633861ae3f90ad8777ed868c25ee93ccff1dc410305ec48ab9db12809528fb` |
| outer | M | `p6_evaluation.py` | `84287815b25f91db4d415fc84fbd005f022e532330e01e37e1b0f0f6b15761fb` |
| outer | M | `p6_launcher.py` | `eacb873232616c9d1d216935a896492a575ce62c8a9680756c229e8057c77a47` |
| outer | M | `p6_preflight.py` | `ab7ff4a28be6d18fe9f0a73a0b2ae65bc6d0989596f778d9ba9fa11a811ae26d` |
| outer | M | `p6_runtime.py` | `03cbc873628d908e30fcedd5f173eeedd7512fb5a4359493729dbeb925c813f7` |
| outer | M | `p6_train.py` | `a0c9323884f02a806de05016d521ec53c28ed17ae1dea7962840715fca51b4b0` |
| outer | C | `tests/test_dsrl_config_entry.py` | `4cf1da62cd25b6b9575742ac376565332afcbfc8a558c66ed549cec8202f5908` |
| outer | M | `tests/test_p6_checkpointing.py` | `5f2a500248a55c2bfc2cf4f3cc51b62e697425eed2634dc35732016926c84309` |
| outer | M | `tests/test_p6_evaluation.py` | `ad91ce033028ada2c5b280ada177d67f81b17e634b8372a5fac2149f5c167d79` |
| outer | M | `tests/test_p6_launcher.py` | `572a319cee36cff482155b5ff29345d30dc79b23eb15799d7f3f88fefa26dc9f` |
| outer | M | `tests/test_p6_preflight.py` | `76544049fc49c772df65d31e1dd7776ae97c8e0f1e3449cabad8a2c239a6492c` |
| outer | M | `tests/test_p6_runtime.py` | `d3895d2b3ede4c29349b635e9fb26b8e0b33a8af37bd9a58e7b498d2c980f4ee` |
| outer | M | `tests/test_p6_train_wiring.py` | `1c48b2b7b228b576a94d16e8659f2d02f08a40966bf8ab24bdce42f810962500` |
| outer | M | `train_dsrl.py` | `f1163cd6acd0f0b20988b075efbcdf2f1c612bfc00d2fc0da62885d72e000c14` |
| outer | M | `utils.py` | `93f971ab684cd59e456745d7c524c6ae7a6f0e29cf256fcf76ef189c6fbde9ea` |
| SB3 | M | `stable_baselines3/__init__.py` | `bd5bd6145d1b35287e16ffc899f93621bf888fbc59f077d058fba32888b28040` |
| SB3 | C | `stable_baselines3/common/buffers.py` | `5b5832cee4747f326633af48defcbc8331b4faed8988333d0a23d4bbd9d79c7a` |
| SB3 | C | `stable_baselines3/common/off_policy_algorithm.py` | `faf445bb31e27d8621e688ae56cb76a4430f38ddd8422508e486be432d1fcb0e` |
| SB3 | M | `stable_baselines3/dsrl/__init__.py` | `cb3f577a894cde38c7bf1c3c9cb585d43694f6081afdcd40d72ea52afccd5ac9` |
| SB3 | C | `stable_baselines3/dsrl/dsrl.py` | `41dc8cfb86b8640284eb29f9b5b585e9906e93c30e7dfe684dd7da8adcbf6bc8` |
| SB3 | M | `stable_baselines3/dsrl/hierarchical_rfs_dsrl.py` | `1d3b9e5c119397cf50c2c0269e1b6cf3496008b634bd9d5849941d243d92f8e6` |
| SB3 | C | `tests/test_dsrl_rfs_hier_phase1.py` | `57fabe3756161cfd7da8337539d846c8afa9f13846e646bc879c1ace1b0e673c` |
| SB3 | M | `tests/test_dsrl_rfs_hier_phase2.py` | `56d5b9634135f69bda48198b8ee30c4d40852ba17b2f67bfd69885f853155043` |
| SB3 | C | `tests/test_dsrl_rfs_hier_phase3.py` | `5400496099da7f61f6eeed830c4ab60b2187f63390f7d42eaff0086cbb3ca6d8` |
| SB3 | M | `tests/test_dsrl_rfs_hier_phase4.py` | `f600444840faa282c2b6674156cc4d960870087cb8950e2558e8a30fb7d6b788` |
| SB3 | C | `tests/test_dsrl_rfs_hier_phase5.py` | `d9b951b33630274b51b4ac5bdb1a4ba531159a6fdb3982e4891aed13335542a5` |

Submodule binding at fingerprint time:

| Submodule | Outer HEAD gitlink | Actual HEAD | Dirty cause |
|---|---|---|---|
| `stable-baselines3` | `076e5563ceb51ff91c6c7743501d18f0456b13af` | same | five tracked modifications plus two untracked flat-RFS files |
| `dppo` | `86ce51834055c02f9013e60dd4c4275606d82df7` | same | untracked media/tools only; not inspected for this design |

Excluded but explicitly fingerprinted flat-RFS files are `stable_baselines3/dsrl/rfs_dsrl.py` (`545358fdb0ae425827594880dd7f25c9654cfef96cc2d1c161f96300833a52f6`) and `tests/test_dsrl_rfs.py` (`9b32ca967679c60baa7206e116e41fd19d4cd50227e0a21e22481657ac4a3996`). They are unrelated/untracked and must not enter Stage 2. DPPO untracked media/tools and ignored/log artifacts were not evidence for this audit and are outside the manifest.

Reproduction command shape is:

```bash
git rev-parse HEAD
git status --porcelain=v1
sha256sum <the sorted paths above>
git -C stable-baselines3 rev-parse HEAD
git -C stable-baselines3 status --porcelain=v1
git -C dppo rev-parse HEAD
git -C dppo status --porcelain=v1
```

Any code-byte change after this fingerprint invalidates the audit until the affected entry is rechecked. Documentation-only revisions do not.

## 2. Current hierarchy implementation

Primary file: `stable-baselines3/stable_baselines3/dsrl/hierarchical_rfs_dsrl.py`

### 2.1 Current modules

| Current name | Meaning | Online/target | Source |
|---|---|---|---|
| `actor` | SAC noise actor | online only | inherited from `DSRL` |
| `critic`, `critic_target` | one twin action critic for executed hierarchy action | online + target | inherited SAC policy |
| `critic_modulation` | twin `QM(observation, noise_scaled, residual_unit)` | online only | `ModulationCritic`, lines 97–197 |
| `critic_noise` | compatibility alias to `critic_modulation` | same object | setup code |
| `residual_actor` | deterministic residual conditioned on observation/noise/base | online only | lines 287–368 |
| `diffusion_policy` | DDIM decoder wrapper | frozen | lines 572–593, 720–753 |

There is no separate `QA_base`, `QA_joint`, or target residual actor.

### 2.2 Current public forward

```text
observation
  -> noise actor
  -> noise_scaled
  -> Torch affine unscale (noise detached)
  -> Frozen DDIM under no_grad
  -> action_base detached
  -> residual_actor(observation, noise_scaled, action_base)
  -> residual_unit
  -> hard-clipped composition
  -> action_exec
```

The residual network has the requested two hidden blocks with LayerNorm and SiLU and a zero-initialized output layer. It is genuinely conditioned on `(observation, noise_scaled, action_base)`. The decoder path is detached. These parts can be retained.

The current composition at lines 224–284 is:

\[
\delta a=\beta\frac{a_{high}-a_{low}}{2}a_r,
\qquad
a_{exec}=\operatorname{clip}(a_b+\delta a,a_{low},a_{high}).
\]

This has ordinary hard-clamp saturation and is not the proposed margin-preserving composition.

Rollout returns `action_exec` for both the environment action and replay action. Standard `predict()` also returns the full executed action, while `predict_zero_residual()` exists for base-only evaluation.

### 2.3 Current losses and prohibited credit paths

The only action critic has joint continuation semantics:

```text
next current noise actor
  -> Frozen DDIM
  -> current residual actor
  -> next_action_exec
  -> target action critic
```

The complete target is under `no_grad` (lines 1025–1064), but it uses the online residual actor, not a target residual actor.

`QM` is trained on a fresh current-policy tuple and the current online action critic:

```text
(observation, current w, current residual)
  -> current QA(observation, current action_exec)
  -> QM regression
```

This is not a replayed historical modulation tuple and not a target/base-only teacher (lines 1066–1106).

The current noise loss explicitly retains the path:

```text
noise actor
  -> noise_scaled
  -> residual_actor(observation, noise_scaled, detached action_base)
  -> QM(observation, noise_scaled, residual_unit)
```

Residual parameters are temporarily frozen, but the residual forward retains input gradient, so residual-conditioned joint value does update the noise actor (lines 1108–1133). An existing Phase 4 test intentionally verifies this path. That test must be replaced because the new architecture forbids it.

The residual actor uses the current action critic with its parameters frozen and its action-input gradient retained (lines 1135–1159). Noise and base are detached. This gradient-isolation pattern is suitable for the future `QA_joint` path.

### 2.4 Current update cadence

With Hopper P6 defaults, each train call performs:

```text
current action QA       20
current joint QM        10
noise actor + alpha      5
residual actor           5
```

The QA target is Polyak-updated inside the 20-step QA loop. With `target_update_interval=1`, it is updated 20 times per call, not once. There is no actor gradient clipping. Separate optimizer blocks clear stale gradients and restore inference modes after training.

## 3. Original DSRL-NA behavior that must remain unchanged

Primary file: `stable-baselines3/stable_baselines3/dsrl/dsrl.py`.

Original DSRL has:

```text
QA(observation, decoded_action)      online + target, Bellman
QW(observation, noise_scaled)        online only, supervised distillation
noise actor                          optimized through QW
alpha                               noise entropy only
Frozen diffusion decoder
```

The QA target (lines 273–295) uses the current noise actor, CPU/NumPy `unscale_action`, Frozen diffusion output, target QA minimum, and one entropy term. The actor is updated from `QW`, never through the diffusion sampler.

The original QW update (lines 327–363) differs from both the current hierarchy and the proposed Core V1:

- it samples unbounded standard-Gaussian decoder noise;
- it scales that noise for `QW` input;
- it uses the online QA as teacher;
- it aligns the two heads;
- QW has no target network and no Bellman bootstrap.

Original alpha is updated on each QA gradient step. With `actor_gradient_steps=-1`, the noise actor is also updated on each QA step. QA target Polyak occurs at the configured gradient-step cadence.

Legacy DSRL saves `policy`, actor optimizer, QA optimizer, the QW module, and alpha state, but not QW optimizer state. P6 control adds QW optimizer persistence without changing the legacy class.

The new hierarchy must not modify `dsrl.py`, common SAC policy behavior, or legacy configs to obtain its three-critic semantics.

## 4. Action chunk, reward, discount, and termination

Primary wrapper: `env_utils.py::ActionChunkWrapper`.

One SB3 transition contains an action chunk of four primitive 3-D Hopper actions. The wrapper:

- executes up to four primitive actions;
- sums primitive rewards without within-chunk discount;
- returns one chunk transition;
- records nominal and actual primitive counts;
- records the first terminal observation and termination index;
- can use legacy “continue after done” or P6 “early break on done” semantics.

The critic then applies `gamma` once per chunk, and the SAC entropy term appears once for the full 12-D noise decision. Core V1 must preserve exactly:

```text
chunk reward       = sum of primitive rewards
Bellman discount   = gamma once per replay transition
noise entropy      = once per replay transition
true terminal      = no bootstrap
time-limit timeout = bootstrap
```

It must not silently use `gamma ** actual_primitive_steps` or per-primitive entropy.

SB3 `_store_transition()` replaces a VecEnv autoreset observation with `info["terminal_observation"]`. Standard replay sampling exposes the training done mask as `done * (1 - timeout)`. A hierarchy-specific storage override must reuse that logic before adding hierarchy metadata.

The direct `train_dsrl.py` helper path uses an older prefill function in `utils.py` that does not perform the same terminal-observation correction. It cannot populate the new tagged replay. Fixing the common legacy path would change baseline behavior; the hierarchy instead needs a dedicated tagged-prefill path or a fail-fast restriction.

“Chunk phase” is not currently a state variable: one record is the whole chunk. The new schema must name concrete chunk metadata (chunk index, nominal/actual primitive count, termination primitive index) rather than borrowing a primitive-step PPO phase with a different meaning.

## 5. Current replay and the required delta

The standard SB3 replay physically stores:

```text
observation, next_observation, action, reward, done, timeout
```

It also has an optional `noise_action` array, but hierarchy rollout does not use it. The current hierarchy stores only `action_exec`; it cannot recover behavior `noise_scaled`, `action_base`, residual, beta, lane, or policy versions.

The required one-physical-buffer design therefore needs a hierarchy-only `ReplayBuffer` subclass. Common `ReplayBuffer` must not be modified. The subclass must support filtering by the flattened `(time_index, environment_index)` branch tag, including after circular overwrite and after save/load.

The current P6 prefill v2 artifact contains only standard transitions and primitive counters. It cannot be migrated by decoding `action_exec` back to a historical noise. Core V1 requires a versioned tagged prefill artifact generated while the behavior policy is running. Legacy v2 input must fail with an explicit message.

Memory is a real constraint. For a 10M-transition capacity, five additional 12-D `float32` arrays alone add approximately 2.4 GB, before IDs, versions, branch fields, and the existing replay. Stage 2 must report allocated replay bytes and perform disk-space preflight; unit tests must use tiny capacities.

## 6. Checkpoint, resume, evaluation, and launcher

### 6.1 Legacy initialization

`_LegacyLoadableDSRL` is a private load-only subclass that invokes the official SB3 load path while supplying a temporary dimension and small replay allocation. P6 control imports it. It must be preserved.

The current hierarchy warm-start copies the noise actor, online QA, target QA, QW into expanded QM, and alpha; it then creates fresh optimizers and zeroes the residual output. The future migration must instead map legacy QA/QW to `QA_base/QW_base`, clone `QA_joint`, and build all targets according to the frozen initialization rule.

### 6.2 Current hierarchy save/load

The current save list includes the policy (and therefore actor/current QA/QA target), actor optimizer, QA optimizer, QM and optimizer, residual and optimizer, and alpha state/optimizer. It has no joint target pair, target residual, branch allocator, beta schedule state, or three-critic counters.

Old hierarchy checkpoints contain `critic_modulation` and a different graph. They must not partially load into Core V1. A saved architecture/replay schema version must produce a clear instruction to restart from a legacy DSRL checkpoint.

### 6.3 P6 resume

P6 currently saves model, replay, manifest, RNG/counters, and validates artifact bindings. Its counter formulas assume the old `20/10/5/5` hierarchy. Its semantic replay hash covers only the standard arrays. Both must be updated for branch fields, three critics, target residual, beta, lane state, and policy versions.

The generic replay-storage byte estimator already includes every NumPy array and can be reused.

### 6.4 Evaluation

`p6_evaluation.py` already provides exact-N complete episodes, independent episode policy seeds, RNG save/restore, model-mode save/restore, and atomic JSON/CSV output. It currently dispatches only to the full hierarchy and recognizes old module names.

Core V1 needs four explicitly named, same-seed modes:

1. current noise actor with residual forced to zero;
2. current full hierarchy;
3. frozen reference noise actor plus DDIM, residual zero;
4. matched DSRL baseline.

Random training diagnostics currently consume the global policy RNG. Candidate-ranking diagnostics must use an isolated RNG context or dedicated generator so logging cadence cannot change training.

### 6.5 Launcher

`p6_launcher.py` is largely algorithm-independent and already protects run directories, logs process state, and supports resume. It should not need an algorithmic rewrite. `p6_train.py`, `p6_preflight.py`, `p6_checkpointing.py`, manifests, and their tests do need new schema/counter/model semantics.

## 7. Direct conflicts with the target graph

| Required invariant | Current behavior |
|---|---|
| Three logical critic meanings | One joint action critic plus joint modulation critic |
| Base future permanently disables residual | Only action critic bootstraps full hierarchy |
| QW teacher is target `QA_base` | QM teacher is current online joint QA |
| Noise actor reads only `QW_base` | Noise gradient passes through residual and QM |
| Joint target uses target residual | Uses online residual |
| Episode-fixed base/joint lane | No lane exists |
| Replay stores actual hierarchy tuple | Stores only executed action |
| Bound-preserving normal composition | Normal path uses hard clip |
| Separate QA trunks and optimizers | Only one QA exists |
| Separate actor clipping | No actor gradient clipping |
| Three/four evaluation modes | Only full hierarchy dispatch exists |
| Versioned exact resume | Old graph/counters/hash only |

## 8. Conflicts resolved by the frozen implementation spec

The accompanying implementation spec resolves several ambiguities that would otherwise produce different algorithms:

- Target initialization distinguishes fresh initialization, legacy DSRL warm-start, and the B-to-R joint activation point.
- Polyak updates remain per critic optimizer step, matching the unit used by original DSRL.
- The margin composition uses a forward-equivalent `u`/`abs(u)` expression so zero initialization receives a centered nonzero subgradient, including at an exact DDIM bound.
- Behavior log-prob is saved for audit only; Bellman entropy always uses the current actor and the same newly sampled next noise.
- `share_features_extractor=True` is rejected for the hierarchy.
- QW Core V1 trains on fresh current-actor candidates over base-lane observations; stored behavior tuples remain exact and are used for behavior diagnostics. No historical tuple is reconstructed.
- Fresh and legacy-warm-start 5M schedules now have fixed B/R/J lengths, exact vector-step boundaries, beta ramps, lane probability, minimum branch population, and a default-disabled Phase J.
- `log_alpha` has one owner: targets and the noise actor use detached alpha, while the alpha-first paired update preserves legacy ordering.
- One immutable tagged BASE prefill is mandatory for a matched pair; control reads its bitwise standard projection and hierarchy reads the full schema after terminal-observation correction.
- Reset-boundary and simulator-exact resume are distinct: stale active episode/lane/chunk state is never restored onto reset environments.
- Cross-lane training, ranking loss, residual gates, and joint residual dropout remain disabled.

## 9. Minimum Stage 2 file scope

### Must modify

SB3:

- `stable-baselines3/stable_baselines3/dsrl/hierarchical_rfs_dsrl.py`
- new `stable-baselines3/stable_baselines3/dsrl/hierarchical_replay_buffer.py`
- `stable-baselines3/tests/test_dsrl_rfs_hier_phase1.py`
- `stable-baselines3/tests/test_dsrl_rfs_hier_phase2.py`
- `stable-baselines3/tests/test_dsrl_rfs_hier_phase3.py`
- `stable-baselines3/tests/test_dsrl_rfs_hier_phase4.py`
- `stable-baselines3/tests/test_dsrl_rfs_hier_phase5.py`

Outer runner/config:

- `train_dsrl.py`
- `utils.py` only where hierarchy logging/legacy-prefill fail-fast is required
- `cfg/gym/dsrl_hopper.yaml`
- `cfg/gym/p6_hopper.yaml`
- `p6_train.py`
- `p6_runtime.py`
- `p6_preflight.py`
- `p6_checkpointing.py`
- `p6_evaluation.py`

Outer tests:

- `tests/test_dsrl_config_entry.py`
- `tests/test_p6_runtime.py`
- `tests/test_p6_preflight.py`
- `tests/test_p6_checkpointing.py`
- `tests/test_p6_evaluation.py`
- `tests/test_p6_train_wiring.py`

### Expected unchanged

- `stable-baselines3/stable_baselines3/dsrl/dsrl.py`
- common replay and off-policy algorithm
- SAC actor/policy
- `env_utils.py`
- `p6_launcher.py` and job wrapper unless a schema-only test proves otherwise
- DPPO
- flat RFS
- per-step PPO/QA diagnostic experiments

If Stage 2 discovers that any expected-unchanged file must change, implementation must stop and return the specific invariant that cannot be met within this scope.
