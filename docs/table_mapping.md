# Manuscript Table and Figure Mapping

| Manuscript item | Configuration / script | Reported values |
|---|---|---|
| Tables 1-2, DEIM-L | `configs/deim_dfine/deim-l-*.yml` | `results/table1_accuracy.csv`, `table2_recall.csv` |
| Tables 1-2, DEIM-S | `configs/deim_dfine/deim-s-*.yml` | same files |
| Table 4 | `configs/ablations/table4_*.yml` | `results/table4_ablation.csv` |
| Table 5 | `configs/ablations/table5_beta_*.yml` | `results/table5_beta_sensitivity.csv` |
| Table 6 | `configs/ablations/table6_exit_*.yml` | `results/table6_exit_sensitivity.csv` |
| Table 7 | `tools/profile_table7_full.py` | `results/table7_profile.csv` |
| Table 8 | query dump + analysis tools | `results/table8_query_analysis.csv` |
| Table 9 | `scripts/run_seed_pair.sh`, `tools/summarize_seed_results.py` | `results/table9_seed_*.csv` |
| Figure 2 | framework asset | `assets/figures/figure2_cmqs_framework.png` |
| Figure 4 | `tools/plot_figure4_convergence.py` | four training logs |
| Figure 5 | `tools/plot_figure5_sensitivity.py` | Tables 5-6 values |
| Figures 6-7 | query visualization tools | checkpoint + COCO val2017 |

## Table 4 semantics

- `table4_classification_only`: original classification-dominated DEIM ranking.
- `table4_geometry_only`: geometric-stability score only.
- `table4_cost_only`: cost-dominated diagnostic with a tiny classification fallback.
- `table4_classification_geometry`: classification and geometry, no GT cost.
- `table4_classification_cost`: classification and GT cost active through the full schedule.
- `table4_classification_cost_curriculum`: classification and cost, exiting at epoch 10.
- `table4_classification_geometry_full_stage_cost`: all three cues with GT cost active through the full schedule.
- `table4_ours_full`: full CMQS setting with exit epoch 10.
