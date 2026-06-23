#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
NAME=${NAME:-CMQS-DEIM-v1.0.0}
OUT=${OUT:-"${ROOT}/dist/${NAME}.zip"}
mkdir -p "$(dirname "${OUT}")"
python -c "from pathlib import Path; import zipfile; root=Path(r'${ROOT}'); out=Path(r'${OUT}'); exclude={'.git','.idea','__pycache__','outputs','dist'}; z=zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED); [z.write(p, Path(root.name)/p.relative_to(root)) for p in root.rglob('*') if p.is_file() and not any(part in exclude for part in p.parts)]; z.close(); print(out)"
