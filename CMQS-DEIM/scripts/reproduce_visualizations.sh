#!/usr/bin/env bash
set -euo pipefail

# Update paths before running.
RAW_DIR=${RAW_DIR:-outputs/query_vis_auto/raw}
COCO_ANN=${COCO_ANN:-datasets/coco/annotations/instances_val2017.json}
COCO_IMG_DIR=${COCO_IMG_DIR:-datasets/coco/val2017}

python tools/plot_figure6_hybrid_final_coco_gt.py \
  --raw-dir "${RAW_DIR}" \
  --coco-ann "${COCO_ANN}" \
  --coco-img-dir "${COCO_IMG_DIR}" \
  --crowded-id 166918 \
  --small-id 498919 \
  --failure-id 547383 \
  --topn 30 \
  --failure-topn 80 \
  --common-center-thr 0.025 \
  --out-dir figures/figure6_hybrid_final \
  --debug

python tools/make_figure7_difference_gallery_coco_gt_v2.py \
  --raw-dir "${RAW_DIR}" \
  --coco-ann "${COCO_ANN}" \
  --coco-img-dir "${COCO_IMG_DIR}" \
  --out-dir figures/figure7_difference_final \
  --topn 40 \
  --common-center-thr 0.025 \
  --debug
