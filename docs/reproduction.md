# Reproduction Guide

## Runtime wiring

The reported curriculum schedule requires explicit epoch propagation:

```text
engine/solver/det_engine.py
  -> model(samples, targets=targets, epoch=epoch)
engine/deim/deim.py
  -> decoder(features, targets, epoch=epoch)
engine/deim/dfine_decoder.py
  -> beta(t) and T_exit
```

Run `bash scripts/preflight_check.sh` after applying the overlay. The decoder raises a runtime error during CMQS training when the current epoch is missing.

## Main configurations

- `configs/deim_dfine/deim-l-baseline.yml`
- `configs/deim_dfine/deim-l-cmqs.yml`
- `configs/deim_dfine/deim-s-baseline.yml`
- `configs/deim_dfine/deim-s-cmqs.yml`

DEIM-L uses $T_{\mathrm{exit}}=10$ and DEIM-S uses $T_{\mathrm{exit}}=24$. Both use fixed Hungarian-style cost weights `(2.0, 5.0, 2.0)` through `query_select_cost_mode: sum`.

## Fair-comparison requirements

Keep the upstream revision, initialization checkpoint, GPUs, optimizer, global batch size, augmentation schedule, random seed and evaluation protocol fixed within each baseline-CMQS pair. Use unique output directories.

## Seed summary command

```bash
python tools/summarize_seed_results.py \
  --run baseline:42:outputs/seed_stability/deim_l_baseline_seed42/log.txt \
  --run Ours:42:outputs/seed_stability/deim_l_cmqs_seed42/log.txt \
  --run baseline:3407:outputs/seed_stability/deim_l_baseline_seed3407/log.txt \
  --run Ours:3407:outputs/seed_stability/deim_l_cmqs_seed3407/log.txt \
  --run baseline:2024:outputs/seed_stability/deim_l_baseline_seed2024/log.txt \
  --run Ours:2024:outputs/seed_stability/deim_l_cmqs_seed2024/log.txt
```

## Ablations

Dedicated configurations for Tables 4-6 are under `configs/ablations/`. The mapping is documented in `docs/table_mapping.md`. “Full-stage cost” uses `query_select_gt_stop_epoch: 58`, so the GT-cost term remains active throughout the 58-epoch schedule.

## Computational profiling

```bash
CONFIG=configs/deim_dfine/deim-l-cmqs.yml \
CHECKPOINT=/path/to/checkpoint.pth \
OUT=outputs/table7_profile/table7_profile.csv \
bash scripts/profile_table7.sh
```

## Query analysis

The decoder supports an optional `dump_selected_queries` flag. It is disabled by default and does not change selected query indices. The analysis tools consume the exported fields for Table 8 and Figures 6-7.

## Result provenance

`results/` contains the numeric values reported in the manuscript. Table 9 stores full-precision AP values where available, and the summary uses sample standard deviation (`ddof=1`). The package does not claim that all full logs or checkpoints are bundled.


## Released model verification

After publishing or downloading the `v1.0.0` assets:

```bash
bash scripts/download_released_models.sh
bash scripts/eval_checkpoint.sh configs/deim_dfine/deim-s-cmqs.yml checkpoints/cmqs_deim_s_best.pth
bash scripts/eval_checkpoint.sh configs/deim_dfine/deim-l-cmqs.yml checkpoints/cmqs_deim_l_best.pth
```

The expected best-recorded validation AP values are 49.27 for DEIM-S + CMQS and 54.58 for DEIM-L + CMQS. Preserve the full terminal COCO summary and compare the downloaded asset checksum with the release metadata.
