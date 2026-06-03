#!/usr/bin/env bash
set -euo pipefail

if [ $# -ne 1 ]; then
  echo "Usage: bash scripts/apply_cmqs_patch.sh /path/to/DEIM"
  exit 1
fi

DEIM_ROOT="$1"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ ! -d "$DEIM_ROOT" ]; then
  echo "Target DEIM directory does not exist: $DEIM_ROOT"
  exit 1
fi

mkdir -p "$DEIM_ROOT/configs/deim_dfine"
mkdir -p "$DEIM_ROOT/engine/deim"
mkdir -p "$DEIM_ROOT/tools"
mkdir -p "$DEIM_ROOT/docs"
mkdir -p "$DEIM_ROOT/scripts"

cp "$REPO_ROOT/configs/deim_dfine/"*.yml "$DEIM_ROOT/configs/deim_dfine/"
cp "$REPO_ROOT/engine/deim/dfine_decoder.py" "$DEIM_ROOT/engine/deim/dfine_decoder.py"
cp "$REPO_ROOT/train.py" "$DEIM_ROOT/train.py"
cp "$REPO_ROOT/tools/"*.py "$DEIM_ROOT/tools/" 2>/dev/null || true
cp "$REPO_ROOT/docs/"*.md "$DEIM_ROOT/docs/" 2>/dev/null || true
cp "$REPO_ROOT/scripts/"*.sh "$DEIM_ROOT/scripts/" 2>/dev/null || true

echo "CMQS patch files have been copied to: $DEIM_ROOT"
echo "Please check dataset paths and pretrained checkpoint paths before training."
