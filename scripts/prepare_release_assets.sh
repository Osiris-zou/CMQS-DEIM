#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "Usage: $0 S_CHECKPOINT S_LOG_OR_RUN_DIR L_CHECKPOINT L_LOG_OR_RUN_DIR" >&2
  exit 2
fi

S_CKPT=$1
S_LOG=$2
L_CKPT=$3
L_LOG=$4
OUT=${OUT:-release_assets/v1.0.0}
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)

for f in "$S_CKPT" "$L_CKPT"; do
  [[ -f "$f" ]] || { echo "Checkpoint not found: $f" >&2; exit 1; }
done

cp -f "$S_CKPT" "$OUT/cmqs_deim_s_best.pth"
cp -f "$L_CKPT" "$OUT/cmqs_deim_l_best.pth"

pack_log() {
  local src=$1
  local dst=$2
  rm -f "$dst"
  if [[ -d "$src" ]]; then
    (cd "$(dirname "$src")" && zip -qr "$dst" "$(basename "$src")")
  elif [[ -f "$src" ]]; then
    zip -qj "$dst" "$src"
  else
    echo "Log file or run directory not found: $src" >&2
    exit 1
  fi
}

pack_log "$S_LOG" "$OUT/cmqs_deim_s_logs.zip"
pack_log "$L_LOG" "$OUT/cmqs_deim_l_logs.zip"

(
  cd "$OUT"
  sha256sum cmqs_deim_s_best.pth cmqs_deim_s_logs.zip \
            cmqs_deim_l_best.pth cmqs_deim_l_logs.zip > SHA256SUMS.txt
)

echo "Prepared release assets in $OUT"
cat "$OUT/SHA256SUMS.txt"
