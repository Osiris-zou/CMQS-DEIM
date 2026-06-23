#!/usr/bin/env python3
"""Static verification of the CMQS epoch-propagation chain and main configs."""
import ast
from pathlib import Path
import yaml
ROOT = Path(__file__).resolve().parents[1]

def calls_with_epoch(path: Path, called_name: str):
    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    found = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = func.id if isinstance(func, ast.Name) else (func.attr if isinstance(func, ast.Attribute) else None)
            if name == called_name and any(kw.arg == 'epoch' for kw in node.keywords):
                found += 1
    return found

def main():
    decoder = ROOT / 'engine/deim/dfine_decoder.py'
    deim = ROOT / 'engine/deim/deim.py'
    engine = ROOT / 'engine/solver/det_engine.py'
    for path in (decoder, deim, engine):
        if not path.exists():
            raise SystemExit(f'MISSING: {path}')
    if calls_with_epoch(deim, 'decoder') < 1:
        raise SystemExit('FAIL: deim.py does not forward epoch to decoder')
    if calls_with_epoch(engine, 'model') < 2:
        raise SystemExit('FAIL: det_engine.py must pass epoch in AMP and non-AMP branches')
    text = decoder.read_text(encoding='utf-8')
    for item in [
        "def forward(self, feats, targets=None, epoch=None)",
        "query_select_cost_mode='sum'",
        "box_noise_scale=self.box_noise_scale",
        "CMQS training requires the current epoch",
    ]:
        if item not in text:
            raise SystemExit(f'FAIL: decoder missing expected text: {item}')
    for rel, stop in {
        'configs/deim_dfine/deim-l-cmqs.yml': 10,
        'configs/deim_dfine/deim-s-cmqs.yml': 24,
    }.items():
        cfg = yaml.safe_load((ROOT / rel).read_text(encoding='utf-8'))
        actual = cfg['DFINETransformer']['query_select_gt_stop_epoch']
        if actual != stop:
            raise SystemExit(f'FAIL: {rel} has T_exit={actual}, expected {stop}')
        if cfg['DFINETransformer']['query_select_cost_mode'] != 'sum':
            raise SystemExit(f'FAIL: {rel} must use query_select_cost_mode=sum')
    print('PASS: explicit epoch propagation and main CMQS configurations are consistent.')

if __name__ == '__main__':
    main()
