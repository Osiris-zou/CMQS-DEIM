#!/usr/bin/env bash
set -euo pipefail
python -m py_compile engine/deim/dfine_decoder.py engine/deim/deim.py engine/solver/det_engine.py tools/*.py
python tools/verify_cmqs_runtime.py
python -c "from pathlib import Path; import yaml; [yaml.safe_load(p.read_text(encoding='utf-8')) for p in Path('configs').rglob('*.yml')]; print('PASS: YAML files parsed successfully.')"
