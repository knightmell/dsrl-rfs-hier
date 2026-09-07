#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-}"
SEED="${2:-1}"
GPU="${3:-0}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-${CONDA_PREFIX:-}/bin/python}"

if [[ -z "${CONDA_PREFIX:-}" || ! -x "$PY" ]]; then
  echo "请先执行: conda activate dsrl" >&2
  exit 2
fi

case "$MODE" in
  hc-full)
    CONFIG="p6_halfcheetah_fresh_600k_cotrain_k4_noclip"
    RUN="${ROOT}/logs/p6/fresh_frozen_ddim_600k_k4_noclip_halfcheetah-medium-v2_dsrl_na_rfs_hier_seed${SEED}_600000chunks"
    ;;
  hc-base)
    CONFIG="p6_halfcheetah_fresh_600k_base_control_k4_noclip"
    RUN="${ROOT}/logs/p6/fresh_frozen_ddim_600k_base_control_k4_noclip_halfcheetah-medium-v2_dsrl_na_rfs_hier_seed${SEED}_2500000chunks"
    ;;
  hopper-full)
    CONFIG="p6_hopper_fresh_600k_cotrain_k4_noclip"
    RUN="${ROOT}/logs/p6/fresh_frozen_ddim_600k_k4_noclip_hopper-medium-v2_dsrl_na_rfs_hier_seed${SEED}_600000chunks"
    ;;
  hopper-base)
    CONFIG="p6_hopper_fresh_600k_base_control_k4_noclip"
    RUN="${ROOT}/logs/p6/fresh_frozen_ddim_600k_base_control_k4_noclip_hopper-medium-v2_dsrl_na_rfs_hier_seed${SEED}_2500000chunks"
    ;;
  *)
    cat >&2 <<'USAGE'
用法: scripts/run_current_600k.sh {hc-full|hc-base|hopper-full|hopper-base} [seed] [gpu]

示例:
  scripts/run_current_600k.sh hc-full 1 0
  scripts/run_current_600k.sh hopper-base 2 0
USAGE
    exit 2
    ;;
esac

if [[ -e "$RUN" ]]; then
  echo "拒绝覆盖已有 run directory: $RUN" >&2
  exit 3
fi

export CUDA_VISIBLE_DEVICES="$GPU"

"$PY" - <<'PY'
import sys
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA 不可用；请先修复 GPU 环境，不要启动训练")
print("GPU:", torch.cuda.get_device_name(0))
PY

exec "$PY" "$ROOT/p6_launcher.py" \
  --run-dir "$RUN" -- \
  "$PY" "$ROOT/p6_train.py" \
  --config-name "$CONFIG" \
  "seed=$SEED" use_wandb=false "logdir=$RUN"
