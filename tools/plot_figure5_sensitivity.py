#!/usr/bin/env python3
"""Plot manuscript Figure 5 from the reported sensitivity results."""
import argparse
from pathlib import Path
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('figures/figure5_sensitivity.png'))
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()
    beta_values = [0.1, 0.2, 0.3]
    beta_ap = [54.41, 54.54, 54.17]
    exit_labels = ['0', '3', '5', '10', '58']
    exit_positions = list(range(len(exit_labels)))
    exit_ap = [54.37, 54.41, 54.54, 54.58, 54.39]
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.titleweight'] = 'bold'
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(beta_values, beta_ap, marker='o', linewidth=2.5, markersize=10, color='darkorange')
    ax1.set_xlabel(r'$\beta_0$', fontsize=18)
    ax1.set_ylabel('AP (%)', fontsize=18)
    ax1.set_title(r'(a) Sensitivity to $\beta_0$ ($T_{\mathrm{exit}}$=5)', fontsize=18)
    ax1.set_xticks(beta_values)
    ax1.axhline(beta_ap[1], color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax2.plot(exit_positions, exit_ap, marker='s', linewidth=2.5, markersize=10, color='steelblue')
    ax2.set_xlabel(r'$T_{\mathrm{exit}}$', fontsize=18)
    ax2.set_ylabel('AP (%)', fontsize=18)
    ax2.set_title(r'(b) Sensitivity to exit epoch ($\beta_0$=0.2)', fontsize=18)
    ax2.set_xticks(exit_positions)
    ax2.set_xticklabels(exit_labels)
    ax2.axhline(54.58, color='red', linestyle='--', linewidth=2, alpha=0.7)
    for ax in (ax1, ax2):
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.tick_params(axis='both', labelsize=18)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight('bold')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.output, dpi=300, bbox_inches='tight')
    if args.show:
        plt.show()
    plt.close(fig)
    print(f'Saved: {args.output}')

if __name__ == '__main__':
    main()
