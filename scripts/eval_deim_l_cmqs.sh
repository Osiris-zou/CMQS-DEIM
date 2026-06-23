#!/usr/bin/env bash
set -euo pipefail

GPUS=${GPUS:-0,1,2,3}
NPROC=${NPROC:-4}
CKPT=${CKPT:-outputs/deim_l_cmqs_seed42/best_stg2.pth}

CUDA_VISIBLE_DEVICES=${GPUS} torchrun --nproc_per_node=${NPROC} train.py \
  -c configs/deim_dfine/deim-l-cmqs.yml \
  --resume "${CKPT}" \
  --test-only
