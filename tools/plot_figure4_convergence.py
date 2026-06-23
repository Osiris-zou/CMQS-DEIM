#!/usr/bin/env python3
"""Plot manuscript Figure 4 from DEIM-S and DEIM-L JSON-lines logs."""
import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt

def extract_ap(path: Path):
    values = []
    with path.open('r', encoding='utf-8') as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f'Invalid JSON in {path}:{line_no}') from exc
            metrics = record.get('test_coco_eval_bbox')
            if metrics:
                values.append(float(metrics[0]) * 100.0)
    if not values:
        raise ValueError(f'No test_coco_eval_bbox values found in {path}')
    return values

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--deim-s-log', type=Path, required=True)
    parser.add_argument('--cmqs-s-log', type=Path, required=True)
    parser.add_argument('--deim-l-log', type=Path, required=True)
    parser.add_argument('--cmqs-l-log', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('figures/figure4_convergence.png'))
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()
    baseline_s = extract_ap(args.deim_s_log)
    cmqs_s = extract_ap(args.cmqs_s_log)
    baseline_l = extract_ap(args.deim_l_log)
    cmqs_l = extract_ap(args.cmqs_l_log)
    plt.rcParams['font.weight'] = 'bold'
    fig, (ax_s, ax_l) = plt.subplots(1, 2, figsize=(14, 6), sharey=False)
    ax_s.plot(range(len(baseline_s)), baseline_s, label='DEIM-S Baseline', color='blue', linewidth=2.5)
    ax_s.plot(range(len(cmqs_s)), cmqs_s, label=r'Ours ($\beta_0$=0.2, $T_{\mathrm{exit}}$=24)', color='red', linewidth=2.5)
    ax_s.set_title('DEIM-S on COCO val2017', fontsize=16, fontweight='bold')
    ax_l.plot(range(len(baseline_l)), baseline_l, label='DEIM-L Baseline', color='blue', linewidth=2.5)
    ax_l.plot(range(len(cmqs_l)), cmqs_l, label=r'Ours ($\beta_0$=0.2, $T_{\mathrm{exit}}$=10)', color='red', linewidth=2.5)
    ax_l.set_title('DEIM-L on COCO val2017', fontsize=16, fontweight='bold')
    for ax in (ax_s, ax_l):
        ax.set_xlabel('Epoch', fontsize=16, fontweight='bold')
        ax.set_ylabel('AP (%)', fontsize=16, fontweight='bold')
        ax.tick_params(axis='both', labelsize=14)
        ax.legend(fontsize=12, prop={'weight': 'bold'})
        ax.grid(True, linestyle='--', alpha=0.6)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.output, dpi=300, bbox_inches='tight')
    if args.show:
        plt.show()
    plt.close(fig)
    print(f'DEIM-S last logged AP: baseline={baseline_s[-1]:.2f}, CMQS={cmqs_s[-1]:.2f}')
    print(f'DEIM-L last logged AP: baseline={baseline_l[-1]:.2f}, CMQS={cmqs_l[-1]:.2f}')
    print(f'Saved: {args.output}')

if __name__ == '__main__':
    main()
