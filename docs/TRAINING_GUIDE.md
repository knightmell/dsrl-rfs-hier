# 当前 VS-Hier 训练指南

本页只描述当前分支 `exp` 的 K=4 Gaussian/NoClip 训练。主图需要的三类
方法是 matched DSRL、base control、VS-Hier full；这里新增的 base-control
配置是内部强对照，不等同于外部 matched DSRL。

## 1. 运行约定

```bash
conda activate dsrl
cd /path/to/dsrl-rfs-hier
export PY="$CONDA_PREFIX/bin/python"
export ROOT="$PWD"
export CUDA_VISIBLE_DEVICES=0
```

也可以用仓库自带的一站式入口（参数依次为 `运行类型 seed GPU`）：

```bash
scripts/run_current_600k.sh hc-full 1 0
scripts/run_current_600k.sh hc-base 1 0
scripts/run_current_600k.sh hopper-full 1 0
scripts/run_current_600k.sh hopper-base 1 0
```

脚本会先检查 CUDA，并拒绝覆盖已存在的 run directory。

不要手工运行 `p6_train.py` 而绕过 launcher。下面的命令使用持久化
`p6_launcher.py`，会写入 `launcher_status.json`、stdout/stderr 和锁文件。

## 2. 第一优先：HalfCheetah VS-Hier full

这一路严格是 300k BASE + 300k RES，总预算 600k：

```bash
RUN="$ROOT/logs/p6/fresh_frozen_ddim_600k_k4_noclip_halfcheetah-medium-v2_dsrl_na_rfs_hier_seed1_600000chunks"
$PY p6_launcher.py \
  --run-dir "$RUN" -- \
  "$PY" p6_train.py \
  --config-name p6_halfcheetah_fresh_600k_cotrain_k4_noclip \
  seed=1 use_wandb=false logdir="$RUN"
```

## 3. 第二优先：HalfCheetah base control

这一路使用同一套 Gaussian K=4、NoClip、网络和环境配置，但在观察预算
600k 内始终停留在 BASE：

```bash
RUN="$ROOT/logs/p6/fresh_frozen_ddim_600k_base_control_k4_noclip_halfcheetah-medium-v2_dsrl_na_rfs_hier_seed1_2500000chunks"
$PY p6_launcher.py \
  --run-dir "$RUN" -- \
  "$PY" p6_train.py \
  --config-name p6_halfcheetah_fresh_600k_base_control_k4_noclip \
  seed=1 use_wandb=false logdir="$RUN"
```

这里配置中的 `total_timesteps=2500000` 是完整 schedule 的声明；真正的
观察停止点是 `stop_after_chunk_transitions=600000`。因此该运行计划内以
`exit 75` 和 `INTERRUPTED` 结束，不能把它误判为训练失败。

## 4. 第三优先：Hopper VS-Hier full

```bash
RUN="$ROOT/logs/p6/fresh_frozen_ddim_600k_k4_noclip_hopper-medium-v2_dsrl_na_rfs_hier_seed1_600000chunks"
$PY p6_launcher.py \
  --run-dir "$RUN" -- \
  "$PY" p6_train.py \
  --config-name p6_hopper_fresh_600k_cotrain_k4_noclip \
  seed=1 use_wandb=false logdir="$RUN"
```

## 5. 第四优先：Hopper base control

```bash
RUN="$ROOT/logs/p6/fresh_frozen_ddim_600k_base_control_k4_noclip_hopper-medium-v2_dsrl_na_rfs_hier_seed1_2500000chunks"
$PY p6_launcher.py \
  --run-dir "$RUN" -- \
  "$PY" p6_train.py \
  --config-name p6_hopper_fresh_600k_base_control_k4_noclip \
  seed=1 use_wandb=false logdir="$RUN"
```

## 6. 多 seed

先完成 seed 1 的四路结果与审计，再用同一配置把 `seed=1` 改成 `seed=2`
或 `seed=3`，并把 run directory 中的 seed 同步修改。不要用相同的
`RUN` 覆盖已有目录。

## 7. 查看进度与结果

```bash
RUN="$ROOT/logs/p6/<run-name>"
cat "$RUN/launcher_status.json"
tail -n 80 "$RUN"/stdout_*.log
find "$RUN" -maxdepth 2 -type f \( -name '*evaluation*' -o -name '*checkpoint*' \) | sort
```

TensorBoard：

```bash
$PY -m tensorboard.main --logdir "$ROOT/logs/p6" --port 6006
```

关键检查点是每 50k chunk transitions。比较时至少记录 D4RL reward、early-fall、
base/full reward，以及 base 和 residual 的 optimizer counters；不要只看某一
次早停分数。

## 8. 不要做的事

1. 不要把当前 VS-Hier full 替换成旧的 original VS-Hier 数值。
2. 不要把 additive-R base-only 当成主图的 VS-Hier full。
3. 不要为了让目录显示 `600000chunks` 而给 base-control 额外覆盖
   `total_timesteps=600000`；那会破坏完整 schedule 的 preflight 合同。
4. 不要两个进程共享同一个 replay/checkpoint 目录，也不要从尚未出现
   `COMPLETE`/`INTERRUPTED` 的 bundle 续训。
5. 不要在当前四个 600k 结果出来前擅自改 K、UTD、clip、QW source 或 lane
   ratio；否则就不再是主图的配对实验。

## 9. 结果文件归档

每个 run 至少保留：

- `run_manifest.json`
- `launcher_status.json`
- 50k 边界的 model checkpoint 与 evaluation
- 计划内的 `COMPLETE` 或 `INTERRUPTED` 标记
- TensorBoard `tensorboard/p6`

训练结束后只同步小型 manifest、evaluation 和选定 checkpoint；不要在训练中
用 rsync 覆盖另一个机器正在写的 run directory。
