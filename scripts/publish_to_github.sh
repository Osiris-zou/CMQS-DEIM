#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/publish_to_github.sh https://github.com/Osiris-zou/CCAQS_main.git
# Run this script from the repository root.

REMOTE=${1:-https://github.com/Osiris-zou/CCAQS_main.git}

git init
git add .
git commit -m "Initial CMQS-DEIM reproducibility release"
git branch -M main
git remote remove origin 2>/dev/null || true
git remote add origin "${REMOTE}"
git push -u origin main --force

echo "Pushed to ${REMOTE}"
