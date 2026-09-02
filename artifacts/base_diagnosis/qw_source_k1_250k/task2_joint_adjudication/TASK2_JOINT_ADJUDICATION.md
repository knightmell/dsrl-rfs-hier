# Task 2 — Walker 250k 三方联合裁决

## 结论

**BLOCKED**（前置 gate 未通过；未进入三方裁决）。

执行时间：2026-08-30 02:00:33 CST（北京时间，UTC+08:00）。要求步数为 exactly 250000 chunks；检查时 Current-K1 与 Gaussian-K1 的实际最新步数均为 200000。两者仍处于运行状态，缺失 250k 完整 model/replay checkpoint bundle、`COMPLETE` 标记和 intentional-interruption 证据。因此依照预注册规则，后续 matched DSRL-NA、固定 evaluator AUC、WBSD probe、机制曲线和 A–D 决策树均未执行。

## 前置 gate

| 输入 | requested_step | actual_step | run_status | training_status | final evaluation | 250k model/replay bundle | COMPLETE | intentional interruption | gate |
|---|---:|---:|---|---|---|---|---|---|---|
| Current-K1 | 250000 | 200000 | running | running | pending | missing | missing | missing | FAIL |
| Gaussian-K1 | 250000 | 200000 | running | running | pending | missing | missing | missing | FAIL |

`actual_step` 取自各自 `run_manifest.json` 的 `latest_model_checkpoint_chunk`。两方 manifest 均声明 `active_stop_after_chunk_transitions: 250000`、`requested_stop_after_chunk: 250000`、model interval 50000 和 replay interval 250000，但检查时最新正式 model checkpoint 仅到 200000；这不能视为 exactly 250000，也不能视为正常到达有界早停。

## 三方输入清单

### Current-K1

- 运行目录：`/home/mrf/dsrl/logs/p6/base_diag_qw_current_k1_pilot250k_fresh_frozen_ddim_walker2d-medium-v2_dsrl_na_rfs_hier_seed1_2500000chunks`
- manifest：`/home/mrf/dsrl/logs/p6/base_diag_qw_current_k1_pilot250k_fresh_frozen_ddim_walker2d-medium-v2_dsrl_na_rfs_hier_seed1_2500000chunks/run_manifest.json`
- manifest SHA-256：`804ea2e9a5fff3109f286e100138ac28a4650ba98bd576c0fcaf6bd0ad5ce0f0`
- 最新 model checkpoint：`/home/mrf/dsrl/logs/p6/base_diag_qw_current_k1_pilot250k_fresh_frozen_ddim_walker2d-medium-v2_dsrl_na_rfs_hier_seed1_2500000chunks/checkpoints/model_000000200000.zip`
- 最新 model checkpoint SHA-256（manifest 记录）：`43b14e42055ad7fbdc3e99cd6c09c4bd3440afb3783d75791eedfb6fd47a7f9d`
- `RUNNING`：存在；SHA-256 `80240996f0afd01bbc37522842f7776718689df92cc86ae99186fb162cfc2444`

### Gaussian-K1

- 运行目录：`/home/mrf/dsrl/logs/p6/base_diag_qw_gaussian_k1_pilot250k_fresh_frozen_ddim_walker2d-medium-v2_dsrl_na_rfs_hier_seed1_2500000chunks`
- manifest：`/home/mrf/dsrl/logs/p6/base_diag_qw_gaussian_k1_pilot250k_fresh_frozen_ddim_walker2d-medium-v2_dsrl_na_rfs_hier_seed1_2500000chunks/run_manifest.json`
- manifest SHA-256：`f83329ce3d01364511d65112f36e3209a718226fad0edf0fcfb93f3f86ec3668`
- 最新 model checkpoint：`/home/mrf/dsrl/logs/p6/base_diag_qw_gaussian_k1_pilot250k_fresh_frozen_ddim_walker2d-medium-v2_dsrl_na_rfs_hier_seed1_2500000chunks/checkpoints/model_000000200000.zip`
- 最新 model checkpoint SHA-256（manifest 记录）：`6ac955e606a5e894f48694fdf80e963b4b12b7f886386946bd0e66ae7675e8a5`
- `RUNNING`：存在；SHA-256 `80240996f0afd01bbc37522842f7776718689df92cc86ae99186fb162cfc2444`

### matched DSRL-NA

- 状态：`not inspected due to failed prerequisite gate`
- requested_step、actual_step、路径与 checkpoint hash：unavailable（按规则在前置失败后禁止进入后续分析；未以 300k/500k 代替 250k）。

## 指标与机制证据

- 固定 evaluator 的 0–250k AUC：unavailable（250k 前置未完成，未计算部分 AUC）。
- 50k/100k/150k/200k/250k paired differences：unavailable（未进入裁决，且 250k 点缺失）。
- within-state QW pairwise、QW Spearman、directional positive-gain rate、normalized regret、QW/QA exploitation gap：unavailable。
- actor parameter-update norm、alpha、entropy、noise mean/std、QA twin disagreement、TD error：unavailable。
- state IDs、state-bank hash、candidate IDs、candidate-prefix hash、proposal seeds、CRN：unavailable；未运行 probe，故未生成或声称匹配证据。

## Warnings

- 两个运行均尚未达到 requested_step；任何基于 200k 的裁决都会违反 exactly-250k 前置条件。
- `RUNNING`、`run_status=running`、`training_status=running` 与 `final_evaluation_status=pending` 不构成正常有界早停证据。
- 缺失 250k replay checkpoint（其配置 interval 为 250000）意味着完整 checkpoint bundle 不存在。
- 没有把部分曲线外推为 250k，没有补造缺失数值，也没有把其他步数 checkpoint 冒充 250k。

## Read-only invariants

- 仅对正式运行目录进行了目录列举、文本读取、存在性检查和 SHA-256 读取/计算。
- 未启动、暂停、杀死或重启任何训练；未修改训练代码、配置、replay 或 checkpoint。
- 未调用 `env.step`、`optimizer.step`，未加载或修改模型参数，未执行 probe。
- 因未运行 probe，模块 hash、optimizer/env counters、CPU/CUDA RNG 前后状态均为 `not applicable`，没有虚构 invariant 记录。

## 唯一下一 gate 决策

**无实验 gate；BLOCKED。** A–D 分支均未选择。唯一允许的后续动作是等待 Current-K1 与 Gaussian-K1 各自自然、正常到达 exactly 250000，并在完整 model/replay bundle、`COMPLETE`、intentional-interruption 及正常有界早停状态证据全部出现后，另行重新执行本联合裁决。不得因本报告自动开展任何实验。

