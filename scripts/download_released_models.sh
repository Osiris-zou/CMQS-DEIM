#!/usr/bin/env bash
set -euo pipefail

TAG=${TAG:-v1.0.0}
BASE="https://github.com/Osiris-zou/CMQS-DEIM/releases/download/${TAG}"
OUT=${OUT:-checkpoints}
LOG_OUT=${LOG_OUT:-logs/released}
mkdir -p "$OUT" "$LOG_OUT"

fetch() {
  local url=$1
  local dest=$2
  echo "Downloading $url"
  curl -fL --retry 3 --retry-delay 2 "$url" -o "$dest"
}

fetch "$BASE/cmqs_deim_s_best.pth" "$OUT/cmqs_deim_s_best.pth"
fetch "$BASE/cmqs_deim_l_best.pth" "$OUT/cmqs_deim_l_best.pth"
fetch "$BASE/cmqs_deim_s_logs.zip" "$LOG_OUT/cmqs_deim_s_logs.zip"
fetch "$BASE/cmqs_deim_l_logs.zip" "$LOG_OUT/cmqs_deim_l_logs.zip"

echo "Downloaded released CMQS assets."
