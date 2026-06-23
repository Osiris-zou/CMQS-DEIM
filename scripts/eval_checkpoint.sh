#!/usr/bin/env bash
set -euo pipefail
CONFIG=${1:?"Usage: bash scripts/eval_checkpoint.sh <config.yml> <checkpoint.pth>"}
CHECKPOINT=${2:?"Usage: bash scripts/eval_checkpoint.sh <config.yml> <checkpoint.pth>"}
GPUS=${GPUS:-0,1,2,3}
NPROC=${NPROC:-4}
CUDA_VISIBLE_DEVICES="${GPUS}" torchrun --nproc_per_node="${NPROC}" train.py \
  -c "${CONFIG}" --resume "${CHECKPOINT}" --test-only
