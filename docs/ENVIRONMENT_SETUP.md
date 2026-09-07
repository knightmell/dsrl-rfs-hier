# VS-Hier 训练环境：一站式安装与检查

本文记录当前训练机上实际使用的 `dsrl` 环境，不自行升级 CUDA、PyTorch
或 Gym 版本。另一台机器优先复刻这一组版本，再开始训练。

## 1. 获取代码

```bash
git clone --recurse-submodules git@github.com:knightmell/dsrl-rfs-hier.git
cd dsrl-rfs-hier
git fetch origin exp
git checkout exp
```

如果 SSH 不可用，使用同一仓库的 HTTPS 地址，并确保最终检出的分支仍然是
`exp`。

## 2. 创建与安装环境

当前训练环境的关键版本是：

```text
Python       3.9.23
PyTorch      2.4.0 + CUDA 12.1
torchvision  0.19.0
NumPy        1.26.4
Gym          0.22.0
Gymnasium    1.1.1
MuJoCo       3.1.6
Hydra        1.3.2
TensorBoard  2.20.0
```

在新机器上：

```bash
conda create -n dsrl python=3.9.23 -y
conda activate dsrl
python -m pip install --upgrade pip

# 当前训练使用 CUDA 12.1 wheel；不要安装 CPU-only torch。
python -m pip install \
  torch==2.4.0 torchvision==0.19.0 \
  --index-url https://download.pytorch.org/whl/cu121

# 项目本地包和 Gym 依赖。
python -m pip install -e './dppo[gym]'
python -m pip install -e './stable-baselines3'

# 当前任务实际会用到的固定运行依赖。
python -m pip install \
  'numpy==1.26.4' \
  'gym==0.22.0' \
  'gymnasium==1.1.1' \
  'hydra-core==1.3.2' \
  'mujoco==3.1.6' \
  'dm-control==1.0.16' \
  'matplotlib==3.7.5' \
  'tensorboard==2.20.0' \
  'tensorboardX==2.6.5' \
  'cloudpickle==3.1.2' \
  'opencv-python==4.11.0.86' \
  'pytest==8.4.2'
```

`dppo[gym]` 会安装 `d4rl`、`cython<3` 和 `patchelf`。不需要为这四个
MuJoCo 训练任务额外安装 ROS、robomimic 或 VLA 环境；那些是本机其他项目的
依赖，不应混入本实验环境。

## 3. 放置不可随 Git 分发的模型与归一化文件

训练配置引用的 Frozen-DDIM 文件不在 Git 中。将已有的模型文件和归一化文件
放到下面的相对路径：

```text
dppo/log/gym-pretrain/halfcheetah-medium-v2_pre_diffusion_mlp_ta4_td20/
  2024-06-12_23-04-42/checkpoint/state_3000.pt
dppo/log/gym-pretrain/hopper-medium-v2_pre_diffusion_mlp_ta4_td20/
  2024-06-12_23-10-05/checkpoint/state_3000.pt
dppo/log/gym/halfcheetah-medium-v2/normalization.npz
dppo/log/gym/hopper-medium-v2/normalization.npz
```

启动前不要凭文件名猜模型是否正确。配置的 preflight 会检查 Frozen-DDIM
哈希和 normalization 哈希；哈希不匹配就停止，先复制正确的 artifact。

## 4. 30 秒环境检查

```bash
conda activate dsrl
which python
python --version
nvidia-smi

python - <<'PY'
import torch, gym, gymnasium, mujoco, hydra
print('torch      =', torch.__version__)
print('torch cuda =', torch.version.cuda)
print('cuda ok    =', torch.cuda.is_available())
print('gym        =', gym.__version__)
print('gymnasium  =', gymnasium.__version__)
print('mujoco     =', mujoco.__version__)
print('hydra      =', hydra.__version__)
if not torch.cuda.is_available():
    raise SystemExit('CUDA is unavailable: do not start a training run')
print('gpu        =', torch.cuda.get_device_name(0))
PY

python -m pip check
```

必须看到 `cuda ok = True`，并且 `pip check` 没有依赖冲突。`nvidia-smi`
正常但 PyTorch 仍显示 CUDA 不可用时，先处理驱动/环境问题，不要重启训练脚本
碰运气。

## 5. 资源建议

- 一张 48GB RTX 4090 优先一次跑一个训练进程；两个 MuJoCo P6 进程会同时
  占用大量 CPU replay 内存。
- 训练前保留至少 24GB 主机可用内存，并确认 GPU 空闲显存至少 16GB。
- `n_envs=10`、`train_freq=1`、`UTD=20`、`batch_size=256`、`QW K=4`
  是当前配置的一部分，不要在复现实验时顺手改掉。
- 实验目录、TensorBoard、replay 和 checkpoint 放在本机磁盘，不要让两个
  机器共享一个正在写入的 run directory。
