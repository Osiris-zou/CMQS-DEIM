#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_common.sh"
SEED=${1:?"Usage: bash scripts/run_seed_pair.sh <seed>"}
run_training configs/deim_dfine/deim-l-baseline.yml "${SEED}" "outputs/seed_stability/deim_l_baseline_seed${SEED}"
run_training configs/deim_dfine/deim-l-cmqs.yml "${SEED}" "outputs/seed_stability/deim_l_cmqs_seed${SEED}"
