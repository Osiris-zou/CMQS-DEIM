#!/usr/bin/env bash
set -euo pipefail

CONFIG=${CONFIG:-configs/deim_dfine/deim-l-cmqs.yml}
CHECKPOINT=${CHECKPOINT:-outputs/deim_l_cmqs_seed42/best_stg2.pth}
OUT=${OUT:-outputs/table7_profile/table7_profile.csv}

python tools/profile_table7_full.py \
  --config "${CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --out-csv "${OUT}"
