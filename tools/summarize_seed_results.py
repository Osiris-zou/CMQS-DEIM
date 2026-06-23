#!/usr/bin/env python3
"""Summarize paired seed results from DEIM JSON-lines training logs."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
METRICS = ['AP', 'AP50', 'AP75', 'APs', 'APm', 'APl']

def best_record(path: Path):
    best = None
    with path.open('r', encoding='utf-8') as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f'Invalid JSON in {path}:{line_no}') from exc
            values = record.get('test_coco_eval_bbox')
            if not values or len(values) < 6:
                continue
            row = [float(v) * 100.0 for v in values[:6]]
            if best is None or row[0] > best[0]:
                best = row
    if best is None:
        raise ValueError(f'No COCO bbox metrics found in {path}')
    return best

def parse_run(spec: str):
    parts = spec.split(':', 2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError('Run must be METHOD:SEED:PATH')
    method, seed, path = parts
    return method, int(seed), Path(path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', action='append', type=parse_run, required=True,
                        help='METHOD:SEED:PATH; provide baseline and Ours for each seed')
    parser.add_argument('--output', type=Path, default=Path('results/seed_summary_from_logs.csv'))
    args = parser.parse_args()
    rows, by_method, by_seed = [], {}, {}
    for method, seed, path in args.run:
        values = best_record(path)
        rows.append([method, seed, *values])
        by_method.setdefault(method, []).append(values)
        by_seed.setdefault(seed, {})[method] = values
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['method', 'seed', *METRICS])
        writer.writerows(rows)
    print(f'Saved per-run results: {args.output}')
    for method, values in by_method.items():
        array = np.asarray(values, dtype=float)
        means = array.mean(axis=0)
        sds = array.std(axis=0, ddof=1) if len(array) > 1 else np.zeros(array.shape[1])
        print(method)
        for metric, mean, sd in zip(METRICS, means, sds):
            print(f'  {metric}: {mean:.4f} ± {sd:.4f}')
    baseline_names = [n for n in by_method if n.lower() in {'baseline', 'deim-l', 'deim'}]
    ours_names = [n for n in by_method if n.lower() in {'ours', 'cmqs'}]
    if baseline_names and ours_names:
        bname, oname = baseline_names[0], ours_names[0]
        deltas = []
        for seed in sorted(by_seed):
            pair = by_seed[seed]
            if bname in pair and oname in pair:
                delta = np.asarray(pair[oname]) - np.asarray(pair[bname])
                deltas.append(delta)
                print(f'seed {seed} paired delta AP: {delta[0]:+.4f}')
        if deltas:
            deltas = np.asarray(deltas)
            sd = deltas[:, 0].std(ddof=1) if len(deltas) > 1 else 0.0
            print(f'paired AP: {deltas[:, 0].mean():+.4f} ± {sd:.4f}')

if __name__ == '__main__':
    main()
