#!/usr/bin/env bash
set -euo pipefail
GPUS=${GPUS:-0,1,2,3}
NPROC=${NPROC:-4}
PRETRAIN=${PRETRAIN:-}
require_pretrain() {
  if [[ -z "${PRETRAIN}" ]]; then
    echo "Set PRETRAIN=/path/to/upstream_DEIM_checkpoint.pth" >&2
    exit 2
  fi
}
run_training() {
  local config=$1
  local seed=$2
  local out=$3
  require_pretrain
  mkdir -p "${out}"
  set -o pipefail
  CUDA_VISIBLE_DEVICES="${GPUS}" torchrun --nproc_per_node="${NPROC}" train.py \
    -c "${config}" --tuning "${PRETRAIN}" --seed "${seed}" --use-amp \
    --output-dir "${out}" 2>&1 | tee -a "${out}/console.log"
  local status=${PIPESTATUS[0]}
  echo "exit_code=${status}" | tee -a "${out}/console.log"
  return "${status}"
}
