<div align="center">

# Steering Your Diffusion Policy with Latent Space Reinforcement Learning (DSRL)

## [[website](https://diffusion-steering.github.io)]      [[pdf](https://arxiv.org/pdf/2506.15799)]

</div>


<p align="center">
  <a href="https://colinqiyangli.github.io/qc/">
    <img alt="teaser figure" src="./assets/teaser.png" width="90%">
  </a>
</p>

## Overview
Diffusion steering via reinforcement learning (DSRL) is a lightweight and efficient method for RL finetuning of diffusion and flow policies. Rather than modifying the weights of the diffusion/flow policy, DSRL instead modifies the noise distribution sampled from to begin the denoising process.

## 当前项目入口

所有持续迭代都更新同一组 living documents，不再新增阶段计划或 handoff：

- [当前项目状态](docs/PROJECT_STATE.md)
- [当前算法约束](docs/CURRENT_ALGORITHM.md)
- [当前实验协议](docs/CURRENT_PROTOCOL.md)
- [实验事实账本](docs/EXPERIMENT_REGISTRY.md)
- [环境安装与检查](docs/ENVIRONMENT_SETUP.md)

训练分岔点和最终步数没有默认值。每次运行都必须显式记录，并使用
`task_s{seed}_{clip|noclip}_k{K}_{branch}k_{final}k_{full|basecontrol}` 命名。
旧 P6 计划、gate、runbook 和 handoff 已移动到 `docs/archive/`，不作为当前
算法或实验事实的直接证据。

## 最短阅读路径（新服务器 / 新协作者）

为了快速了解当前项目，不需要从旧 handoff 或全部代码开始。按下面顺序阅读即可：

1. `docs/PROJECT_STATE.md`：当前算法、当前工作、已知问题和最近结论。
2. `docs/CURRENT_PROTOCOL.md`：命名、分岔、full/base-control 语义、检查点和评估规则。
3. `docs/EXPERIMENT_REGISTRY.md`：已完成实验的事实账本、结果和实际路径。
4. `docs/ENVIRONMENT_SETUP.md`：环境版本、模型文件位置和 GPU 检查。
5. 只有在需要启动或诊断时才看对应的 `cfg/gym/*.yaml`、`p6_train.py`、
   `p6_launcher.py`；不要默认读取 `docs/archive/`。

当前代码和文档的优先级是：living documents（`docs/`）和当前分支代码 >
`docs/archive/`。如果日志、manifest 或检查点与历史文档冲突，以实际运行产物为准；
如果实际产物缺失，先报告缺口，不要用 archive 内容补写成“已完成”。

## 在另一台服务器上获取当前 `exp` 分支

在新机器上不要复制正在写入的 run directory，也不要让两台机器共同写同一个
TensorBoard、replay 或 checkpoint 路径。代码、配置和训练脚本通过 Git 同步；模型
权重、normalization 文件和实验产物通过独立的文件复制方式同步。

```bash
git clone --recurse-submodules git@github.com:knightmell/dsrl-rfs-hier.git
cd dsrl-rfs-hier
git fetch origin exp
git switch --track -c exp origin/exp
git submodule update --init --recursive
```

已有 clone 则使用：

```bash
git fetch origin exp
git switch exp
git pull --ff-only origin exp
git submodule update --init --recursive
```

确认当前版本和工作区：

```bash
git branch --show-current
git rev-parse --short HEAD
git status --short
```

训练前必须确认工作区没有未预期的代码改动；实验结果目录不要提交到 Git。

## 一站式环境准备

完整版本表和不可随 Git 分发的 Frozen-DDIM / normalization 文件位置见
[`docs/ENVIRONMENT_SETUP.md`](docs/ENVIRONMENT_SETUP.md)。最小流程如下：

```bash
conda create -n dsrl python=3.9.23 -y
conda activate dsrl
python -m pip install --upgrade pip
python -m pip install torch==2.4.0 torchvision==0.19.0 \
  --index-url https://download.pytorch.org/whl/cu121
python -m pip install -e './dppo[gym]'
python -m pip install -e './stable-baselines3'
python -m pip install numpy==1.26.4 gym==0.22.0 gymnasium==1.1.1 \
  mujoco==3.1.6 hydra-core==1.3.2 tensorboard==2.20.0 tensorboardX==2.6.5
```

先做环境检查，再启动训练；`nvidia-smi` 正常但 PyTorch 的
`torch.cuda.is_available()` 为 `False` 时不要反复重试训练：

```bash
conda activate dsrl
nvidia-smi
python - <<'PY'
import torch
print(torch.__version__, torch.version.cuda)
print('cuda ok =', torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit('CUDA unavailable; fix the environment first')
print(torch.cuda.get_device_name(0))
PY
python -m pip check
```

## 加速训练：如何使用 `runtime/fast`

当前加速实现已经包含在 `exp` 分支中：

- `cfg/gym/runtime/fast.yaml`：显式的吞吐 overlay；
- `p6_train.py`：QW teacher 的设备内批处理和运行时开关；
- `stable-baselines3/stable_baselines3/dsrl/hierarchical_rfs_dsrl.py`：对应的
  QW 更新实现；
- `p6_preflight.py`：启动前参数和路径检查。

加速的目标是缓解 CPU / host-memory 瓶颈，不是改变算法。`+runtime=fast` 只适合
**新的 VS-Hier run**，其当前效果是：关闭重复的热路径 contract 检查，将 K=4、
B=256 的 QW teacher microbatch 合并到最多 1024 行，并降低在线评估、模型检查点和
replay bundle 的写入频率。K、query 数、UTD、优化器步数、replay/RNG 和 residual
组合语义不变。严格诊断时仍使用默认 strict 模式。

例如，新建一个明确命名的 Hopper full run：

```bash
conda activate dsrl
ROOT="$PWD"
PY="$CONDA_PREFIX/bin/python"
RUN="$ROOT/logs/p6/hopper_s3_noclip_k4_300k_600k_full"
export CUDA_VISIBLE_DEVICES=0

"$PY" -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
"$PY" "$ROOT/p6_launcher.py" --run-dir "$RUN" -- \
  "$PY" "$ROOT/p6_train.py" \
  --config-name p6_hopper_fresh_600k_additive_res_k4_noclip \
  seed=3 use_wandb=false "+runtime=fast" "logdir=$RUN"
```

实际运行时，把 config、seed、clip 状态、K、分岔点、最终步数和 role 改成实验
计划中的值；不要只改目录名。启动前应能在 resolved config / manifest 中核对：
`resume source`、`training_start_chunk_transitions`、`stop_after_chunk_transitions`、
QW source/query、BASE/RES UTD、beta/lane 设置以及输出路径。

### 继续已有训练

续训必须使用原检查点的 resolved config 和原始状态链。不要为了“加速”在 resume
时偷偷加入 `+runtime=fast`，也不要重置 optimizer、replay、RNG 或重新 warm-up。
示意命令如下，`RUN` 必须是一个未完成且确实允许原地续训的 run directory：

```bash
"$PY" "$ROOT/p6_launcher.py" --resume --run-dir "$RUN" -- \
  "$PY" "$ROOT/p6_train.py" \
  --config-name <原始config> \
  seed=<原seed> use_wandb=false "logdir=$RUN" \
  "p6.resume_bundle_path=<原resume bundle或manifest要求的路径>"
```

如果原 run 的 resolved runtime 与 `fast` 不同，应新建一个明确的新实验，而不是
混合两条训练链。原 run directory、manifest、checkpoint 和评估日志必须保留；
同一目录不可被两台机器同时写入。

## 评估、监控和失败处理

评估遵循当前协议：500k 以前默认不评估；从 500k 起每个 100k 检查点做 10 episodes；
最终 800k 检查点做 100 episodes，并区分独立训练的 `basecontrol` 与 full checkpoint
的 residual-disabled base view。完整评估流程见
[`running-research-evaluation` skill](.agents/skills/running-research-evaluation/SKILL.md)。

监控只报告有意义的状态变化：新检查点、终止（`COMPLETE` / `INTERRUPTED` /
`FAILED`）、无进展、OOM 或资源越界。训练意外停止时先查看 launcher status、最新
manifest、stdout/stderr 和进程状态，确定最后 transition 和终止原因；不要自动杀死、
重启、改 bundle 或重绑 provenance。参数错误是硬阻塞；非必要的旧 gate / 历史 preflight
若阻塞训练，应把具体约束报告给用户，由用户决定是否放宽。


## Installation
本分支的 VS-Hier 训练请先按上面的最小流程和
[ENVIRONMENT_SETUP.md](docs/ENVIRONMENT_SETUP.md) 完成当前 `dsrl` 环境。

如果只想使用原始 DSRL/Robomimic 代码，原始安装流程仍然是：

1. Clone repository
```
git clone --recurse-submodules git@github.com:knightmell/dsrl-rfs-hier.git
cd dsrl-rfs-hier
```
2. Create conda environment
```
conda create -n dsrl python=3.9 -y
conda activate dsrl
```
3. Install our fork of DPPO 
```
cd dppo
pip install -e .
pip install -e .[robomimic]
pip install -e .[gym]
cd ..
```
4. Install our fork of Stable Baselines3
```
cd stable-baselines3
pip install -e .
cd ..
```
The diffusion policy checkpoints for the Robomimic and Gym experiments can be found [here](https://drive.google.com/drive/folders/1kzC49RRFOE7aTnJh_7OvJ1K5XaDmtuh1?usp=share_link). Download the contents of this folder and place in `./dppo/log`.

## Running DSRL
To run DSRL on Robomimic, call
```
python train_dsrl.py --config-path=cfg/robomimic --config-name=dsrl_can.yaml
```
where `dsrl_can.yaml` is set to the config file for the desired task. Similarly, for Gym, call
```
python train_dsrl.py --config-path=cfg/gym --config-name=dsrl_hopper.yaml
```
where `dsrl_hopper.yaml` is set to the config file for the desired task.

## Applying DSRL to new settings
It is straightforward to apply DSRL to new settings. Doing this typically requires:
- Access to a diffusion or flow policy with the ability to control the noise initializing the denoising process. Note that if using a diffusion policy it must be sampled from with DDIM sampling.
-  In the case of `DSRL-NA`, the diffusion/flow policy is passed to the `SACDiffusionNoise` algorithm, and then this algorithm is simply run on a standard gym environment. 
- In the case of `DSRL-SAC`, it is recommended that you write a wrapper around your environment which transforms the action space from the original action space to the noise space of the diffusion/noise policy. Here, the noise action given to the environment wrapper is then denoised through the diffusion policy, and this denoised action is played on the original environment, all of which is performed within the environment wrapper. See the `DiffusionPolicyEnvWrapper` in `env_utils.py` for an example of this. 



### Tips for hyperparameter tuning
The following may be helpful in tuning DSRL on new settings:
- Typically the key hyperparameters to tune are `action_magnitude` and `utd`. `action_magnitude` controls how large a noise value can be played in the noise action space, and `utd` is the number of gradient steps taken per update. Typically setting `action_magnitude` around 1.5 and `utd` around 20 performs effectively, but for best performance these should be tuned on new environments.
- As described in the paper, there are two primary variants of the algorithm: `DSRL-NA` and `DSRL-SAC`. `DSRL-SAC` simply runs `SAC` with the action space the noise space of the diffusion policy, while `DSRL-NA` distills a Q-function learned on the original action space (see the paper for further details). In general `DSRL-NA` is more sample efficient and should be preferred to `DSRL-SAC`, however `DSRL-SAC` is somewhat more computationally efficient in settings where speed is critical.
- DSRL typically performs best when using relatively large actor and critic networks. A reasonable value here is typically using a 3-layer MLP of width 2048. Tuning the size can sometimes lead to further gains. 

## Acknowledgements
Our implementation of DSRL is built on top of [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3). For our diffusion policy implementation, we utilize the implementation given in the [DPPO](https://github.com/irom-princeton/dppo) codebase.

## Citation
```
@article{wagenmaker2025steering,
  author    = {Wagenmaker, Andrew and Nakamoto, Mitsuhiko and Zhang, Yunchu and Park, Seohong and Yagoub, Waleed and Nagabandi, Anusha and Gupta, Abhishek and Levine, Sergey},
  title     = {Steering Your Diffusion Policy with Latent Space Reinforcement Learning},
  journal   = {Conference on Robot Learning (CoRL)},
  year      = {2025},
}
```
