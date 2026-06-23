#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_common.sh"
SEED=${1:-42}
OUT=${OUT:-outputs/deim_s_cmqs_seed${SEED}}
run_training "configs/deim_dfine/deim-s-cmqs.yml" "${SEED}" "${OUT}"
