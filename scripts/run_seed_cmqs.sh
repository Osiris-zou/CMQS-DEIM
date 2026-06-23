#!/usr/bin/env bash
set -euo pipefail

SEED=${1:?"Usage: bash scripts/run_seed_cmqs.sh <seed>"}
GPUS=${GPUS:-0,1,2,3}
NPROC=${NPROC:-4}
PRETRAIN=${PRETRAIN:-/path/to/deim_dfine_hgnetv2_l_coco_50e.pth}
OUT=outputs/seed_stability/deim_l_cmqs_seed${SEED}

CUDA_VISIBLE_DEVICES=${GPUS} torchrun --nproc_per_node=${NPROC} train.py \
  -c configs/deim_dfine/deim-l-cmqs.yml \
  --tuning "${PRETRAIN}" \
  --seed "${SEED}" \
  --use-amp \
  --output-dir "${OUT}"
