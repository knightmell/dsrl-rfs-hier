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

# 项目本地包和当前 Gym / Robomimic / D3IL 依赖。
python -m pip install -e './dppo[gym,robomimic,d3il]'
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

`dppo[robomimic]` 固定使用 `robomimic==0.3.0` 与 Robosuite v1.4.1。
Avoid-M1 还需要当前实测的 D3IL fork：

```bash
git clone https://github.com/allenzren/d3il ../d3il
git -C ../d3il checkout 139dbf9b114d0f6192e5433ebcffeb0fc17098f4
python -m pip install -e '../d3il/environments/d3il'
python -m pip install -e '../d3il/environments/d3il/envs/gym_avoiding_env'
```

远端无显示器时，在训练 shell 中设置：

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
```

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

# Robomimic Can
dppo/log/robomimic-pretrain/can/can_pre_diffusion_mlp_ta4_td20/
  2024-06-28_13-29-54/checkpoint/state_5000.pt
dppo/log/robomimic/can/normalization.npz

# Robomimic Square
dppo/log/robomimic-pretrain/square/
  square_pre_diffusion_mlp_ta4_td100_ddim-100steps/
  2025-04-11_19-13-26_44/checkpoint/state_3000.pt
dppo/log/robomimic/square/normalization.npz

# D3IL Avoid-M1
dppo/log/d3il-pretrain/m1/avoid_d56_r12_pre_diffusion_mlp_ta4_td20/
  2024-07-06_22-50-07/checkpoint/state_10000.pt
dppo/data/d3il/avoid_m1/normalization.npz
```

Can 与 Square 文件可从公开镜像直接放入配置期望的 `dppo/log` 布局：

```bash
python -m pip install 'huggingface_hub[cli]'
hf download knightnemo/vam-robomimic-assets \
  robomimic-pretrain/can/can_pre_diffusion_mlp_ta4_td20/2024-06-28_13-29-54/checkpoint/state_5000.pt \
  robomimic-pretrain/square/square_pre_diffusion_mlp_ta4_td100_ddim-100steps/2025-04-11_19-13-26_44/checkpoint/state_3000.pt \
  robomimic/can/normalization.npz \
  robomimic/square/normalization.npz \
  --local-dir dppo/log
```

Avoid-M1 的两个文件从已有训练机或 DSRL 发布资产复制到上述路径。全部资产在
启动前用下面的固定哈希核对：

```bash
sha256sum \
  dppo/log/robomimic-pretrain/can/can_pre_diffusion_mlp_ta4_td20/2024-06-28_13-29-54/checkpoint/state_5000.pt \
  dppo/log/robomimic/can/normalization.npz \
  dppo/log/robomimic-pretrain/square/square_pre_diffusion_mlp_ta4_td100_ddim-100steps/2025-04-11_19-13-26_44/checkpoint/state_3000.pt \
  dppo/log/robomimic/square/normalization.npz \
  dppo/log/d3il-pretrain/m1/avoid_d56_r12_pre_diffusion_mlp_ta4_td20/2024-07-06_22-50-07/checkpoint/state_10000.pt \
  dppo/data/d3il/avoid_m1/normalization.npz
```

期望依次为：

```text
61851045e6b516807826e3bda4270c9e4a086023f2dd85af00eadb59d9b98a1b
a4bb04c498625bfad0ee6faae21674c0a06879cd9f9614bbd2da41a7d0dc1c1a
e4b391aedc33e8a94bb5cbe25fc737159b64963f51bec22c5e3e104dc1e32373
68ec0abfd989d5f0121f0e9a1dd074b49f859c167ff941c70d2eff8006074ff3
7a420985fd213f79ac03b13f62c3e75c5ae33e6f5c82343e75ad8d4ad0ba230b
24d0c2b650fe26832e0de0474c06d2a989c8a989b0233d039ff1e6b1a52cdc68
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
import robomimic, robosuite, gym_avoiding
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
- 新 Can/Square/Avoid 配置使用 `n_envs=4`、`train_freq=1`、`batch_size=256`；
  matched DSRL 使用 UTD=20，VS-Hier 显式记录 Gaussian QW、K=4、NoClip
  和各任务 BASE/RES 边界。不要为提速顺手修改这些算法参数。
- 实验目录、TensorBoard、replay 和 checkpoint 放在本机磁盘，不要让两个
  机器共享一个正在写入的 run directory。
