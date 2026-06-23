#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_common.sh"
NAME=${1:?"Usage: bash scripts/run_ablation.sh <config-name-without-yml> [seed]"}
SEED=${2:-42}
CONFIG="configs/ablations/${NAME}.yml"
[[ -f "${CONFIG}" ]] || { echo "Missing ${CONFIG}" >&2; exit 2; }
OUT=${OUT:-"outputs/ablations/${NAME}_seed${SEED}"}
run_training "${CONFIG}" "${SEED}" "${OUT}"
