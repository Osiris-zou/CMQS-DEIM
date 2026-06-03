# Reproduction Guide

This guide explains how to reproduce the main DEIM-L experiments with CMQS.

## 1. Prepare DEIM Base Code

Use a clean upstream DEIM codebase and apply the CMQS files:

```bash
bash scripts/apply_cmqs_patch.sh /path/to/DEIM
cd /path/to/DEIM
```

## 2. Prepare COCO 2017

Expected structure:

```text
datasets/coco/
├── train2017/
├── val2017/
└── annotations/
    ├── instances_train2017.json
    └── instances_val2017.json
```

## 3. Train DEIM-L CMQS

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train.py \
  -c configs/deim_dfine/deim-l-cmqs.yml \
  --tuning /path/to/deim_dfine_hgnetv2_l_coco_50e.pth \
  --seed 42 \
  --use-amp \
  --output-dir outputs/deim_l_cmqs_seed42
```

## 4. Train Local DEIM-L Baseline

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train.py \
  -c configs/deim_dfine/deim-l-baseline.yml \
  --tuning /path/to/deim_dfine_hgnetv2_l_coco_50e.pth \
  --seed 42 \
  --use-amp \
  --output-dir outputs/deim_l_baseline_seed42
```

## 5. Evaluate

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train.py \
  -c configs/deim_dfine/deim-l-cmqs.yml \
  --resume outputs/deim_l_cmqs_seed42/best_stg2.pth \
  --test-only
```

## 6. Notes

- Use the same number of GPUs and the same evaluation protocol for baseline and CMQS.
- Do not mix `--resume` and `--tuning` in the same command.
- Use separate output directories for different seeds.
- In the paper, \(T_{exit}\) corresponds to `query_select_gt_stop_epoch` in YAML.
