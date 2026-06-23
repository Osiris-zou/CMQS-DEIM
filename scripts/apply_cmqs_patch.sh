#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 1 ]]; then
  echo "Usage: bash scripts/apply_cmqs_patch.sh /path/to/DEIM" >&2
  exit 2
fi
DEIM_ROOT=$(cd "$1" && pwd)
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
required=(
  "engine/deim"
  "engine/solver"
  "configs/deim_dfine/dfine_hgnetv2_l_coco.yml"
  "configs/deim_dfine/dfine_hgnetv2_s_coco.yml"
  "configs/base/deim.yml"
  "train.py"
)
for rel in "${required[@]}"; do
  [[ -e "${DEIM_ROOT}/${rel}" ]] || { echo "Missing upstream DEIM path: ${rel}" >&2; exit 3; }
done
stamp=$(date +%Y%m%d_%H%M%S)
backup="${DEIM_ROOT}/.cmqs_backup_${stamp}"
mkdir -p "${backup}/engine/deim" "${backup}/engine/solver"
for rel in engine/deim/dfine_decoder.py engine/deim/deim.py engine/solver/det_engine.py; do
  [[ -f "${DEIM_ROOT}/${rel}" ]] && cp "${DEIM_ROOT}/${rel}" "${backup}/${rel}"
done
mkdir -p "${DEIM_ROOT}/engine/deim" "${DEIM_ROOT}/engine/solver" \
  "${DEIM_ROOT}/configs/deim_dfine" "${DEIM_ROOT}/configs/ablations" \
  "${DEIM_ROOT}/tools" "${DEIM_ROOT}/scripts" "${DEIM_ROOT}/results"
cp "${REPO_ROOT}/engine/deim/dfine_decoder.py" "${DEIM_ROOT}/engine/deim/dfine_decoder.py"
cp "${REPO_ROOT}/engine/deim/deim.py" "${DEIM_ROOT}/engine/deim/deim.py"
cp "${REPO_ROOT}/engine/solver/det_engine.py" "${DEIM_ROOT}/engine/solver/det_engine.py"
cp "${REPO_ROOT}/configs/deim_dfine/"*.yml "${DEIM_ROOT}/configs/deim_dfine/"
cp "${REPO_ROOT}/configs/ablations/"*.yml "${DEIM_ROOT}/configs/ablations/"
cp "${REPO_ROOT}/tools/"*.py "${DEIM_ROOT}/tools/"
cp "${REPO_ROOT}/scripts/"*.sh "${DEIM_ROOT}/scripts/"
cp "${REPO_ROOT}/results/"*.csv "${DEIM_ROOT}/results/"
chmod +x "${DEIM_ROOT}/scripts/"*.sh
echo "CMQS files installed into: ${DEIM_ROOT}"
echo "Backup of replaced core files: ${backup}"
echo "Next: cd ${DEIM_ROOT} && bash scripts/preflight_check.sh"
