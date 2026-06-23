#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "Usage: $0 S_CHECKPOINT S_LOG_TXT L_CHECKPOINT L_LOG_TXT" >&2
  exit 2
fi

S_CKPT=$1
S_LOG=$2
L_CKPT=$3
L_LOG=$4

OUT=${OUT:-release_assets/v1.0.0}
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)

for file in "$S_CKPT" "$S_LOG" "$L_CKPT" "$L_LOG"; do
  if [[ ! -f "$file" ]]; then
    echo "File not found: $file" >&2
    exit 1
  fi

  if [[ ! -s "$file" ]]; then
    echo "File is empty: $file" >&2
    exit 1
  fi
done

cp -f "$S_CKPT" "$OUT/cmqs_deim_s_best.pth"
cp -f "$S_LOG"  "$OUT/cmqs_deim_s_logs.txt"
cp -f "$L_CKPT" "$OUT/cmqs_deim_l_best.pth"
cp -f "$L_LOG"  "$OUT/cmqs_deim_l_logs.txt"

(
  cd "$OUT"
  sha256sum \
    cmqs_deim_s_best.pth \
    cmqs_deim_s_logs.txt \
    cmqs_deim_l_best.pth \
    cmqs_deim_l_logs.txt \
    > SHA256SUMS.txt
)

echo "Prepared release assets in: $OUT"
cat "$OUT/SHA256SUMS.txt"
