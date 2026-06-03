#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Automatically select objective visualization cases for Fig. 6.

This script compares DEIM-L baseline and CMQS on COCO val2017 and ranks images
using selected-query statistics before decoder refinement. It avoids subjective
manual image picking.

Outputs:
  - per_image_query_metrics.csv
  - figure6_selected_cases.json
  - raw query dump files for plotting Figure 6

Before running, make sure your dfine_decoder.py exports:
  outputs['selected_query_boxes']   # [B, K, 4], normalized cxcywh
  outputs['selected_query_scores']  # [B, K]
  outputs['selected_query_indices'] # [B, K]
  outputs['selected_query_logits']  # [B, K, C]  (recommended for cost metric)
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torchvision

# Make project root importable when script is placed in project root or tools/.
PROJECT_ROOT = Path(__file__).resolve().parent
if (PROJECT_ROOT / "engine").exists():
    sys.path.insert(0, str(PROJECT_ROOT))
else:
    sys.path.insert(0, str(PROJECT_ROOT.parent))

from engine.core import YAMLConfig
from engine.solver import TASKS

# Standalone box utilities. Do not depend on src.zoo.dfine import paths.
def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack(
        [cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h],
        dim=-1,
    )


def _box_area(boxes: torch.Tensor) -> torch.Tensor:
    return (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)


def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Generalized IoU for boxes in xyxy format."""
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((boxes1.shape[0], boxes2.shape[0]), device=boxes1.device, dtype=boxes1.dtype)

    area1 = _box_area(boxes1)
    area2 = _box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    union = area1[:, None] + area2[None, :] - inter
    iou = inter / union.clamp(min=1e-7)

    lt_c = torch.min(boxes1[:, None, :2], boxes2[None, :, :2])
    rb_c = torch.max(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh_c = (rb_c - lt_c).clamp(min=0)
    area_c = wh_c[:, :, 0] * wh_c[:, :, 1]

    return iou - (area_c - union) / area_c.clamp(min=1e-7)


# -----------------------------------------------------------------------------
# Default paths. Modify these paths directly if you prefer not to use CLI args.
# -----------------------------------------------------------------------------
BASELINE_CONFIG = r"/media/hanyong/zgw/zp/deim 2/configs/deim_dfine/deim_hgnetv2_l_coco.yml"
BASELINE_CKPT = r"/media/hanyong/zgw/zp/deim 2/outputs/deim_hgnetv2_l_coco/best_stg2.pth"
CMQS_CONFIG = r"/media/hanyong/zgw/zp/deim 2/configs/deim_dfine/deim-l.yml"
CMQS_CKPT = r"/media/hanyong/zgw/zp/deim 2/outputs/deim_hgnetv2_l_coco_a7_stop_10/best_stg2.pth"
OUT_DIR = r"/media/hanyong/zgw/zp/deim 2/outputs/query_vis_auto"


def init_single_process_distributed(port: str = "29571") -> None:
    if not dist.is_available() or dist.is_initialized():
        return
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", port)
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend, init_method="env://")
    if torch.cuda.is_available():
        torch.cuda.set_device(0)


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        try:
            dist.barrier()
            dist.destroy_process_group()
        except Exception:
            pass


def robust_to_device(targets, device):
    return [{k: (v.to(device) if hasattr(v, "to") else v) for k, v in t.items()} for t in targets]


def get_image_id(target: dict) -> int:
    for key in ["image_id", "img_id", "id"]:
        if key in target:
            v = target[key]
            if torch.is_tensor(v):
                return int(v.item())
            return int(v)
    raise KeyError("Cannot find image_id/img_id/id in target.")


def get_sample_image(samples, b: int) -> torch.Tensor:
    # In this codebase samples is normally a tensor [B, 3, H, W].
    if torch.is_tensor(samples):
        return samples[b].detach().cpu().clamp(0, 1)
    if hasattr(samples, "tensors"):
        return samples.tensors[b].detach().cpu().clamp(0, 1)
    raise TypeError(f"Unsupported samples type: {type(samples)}")


def box_xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    x0, y0, x1, y1 = boxes.unbind(-1)
    return torch.stack([(x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0], dim=-1)


def normalize_gt_boxes(target: dict) -> torch.Tensor:
    boxes = target["boxes"].float()
    if boxes.numel() == 0:
        return boxes.reshape(0, 4)
    if boxes.max() <= 1.5:
        return boxes.clamp(0, 1)

    if "orig_size" in target:
        h, w = target["orig_size"].float()
    elif "size" in target:
        h, w = target["size"].float()
    else:
        raise KeyError("target must contain orig_size or size for absolute GT boxes.")
    cxcywh = box_xyxy_to_cxcywh(boxes)
    scale = torch.tensor([w, h, w, h], dtype=boxes.dtype, device=boxes.device)
    return (cxcywh / scale).clamp(0, 1)


def get_gt_scale_labels(gt_cxcywh: torch.Tensor, target: dict) -> torch.Tensor:
    if gt_cxcywh.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=gt_cxcywh.device)
    if "orig_size" in target:
        h, w = target["orig_size"].float()
    elif "size" in target:
        h, w = target["size"].float()
    else:
        raise KeyError("target must contain orig_size or size.")
    pixel_area = gt_cxcywh[:, 2] * w * gt_cxcywh[:, 3] * h
    scale = torch.empty_like(pixel_area, dtype=torch.long)
    scale[pixel_area < 32 ** 2] = 0
    scale[(pixel_area >= 32 ** 2) & (pixel_area < 96 ** 2)] = 1
    scale[pixel_area >= 96 ** 2] = 2
    return scale


def associate_queries_to_gt(query_cxcywh, gt_cxcywh, iou_thr=0.1):
    device = query_cxcywh.device
    k = query_cxcywh.shape[0]
    m = gt_cxcywh.shape[0]
    if k == 0 or m == 0:
        return torch.zeros((m,), dtype=torch.bool, device=device), torch.full((k,), -1, dtype=torch.long, device=device)

    q_xyxy = box_cxcywh_to_xyxy(query_cxcywh).clamp(0, 1)
    g_xyxy = box_cxcywh_to_xyxy(gt_cxcywh).clamp(0, 1)
    iou = torchvision.ops.box_iou(q_xyxy, g_xyxy)
    center = query_cxcywh[:, :2]
    inside = (
        (center[:, None, 0] >= g_xyxy[None, :, 0])
        & (center[:, None, 0] <= g_xyxy[None, :, 2])
        & (center[:, None, 1] >= g_xyxy[None, :, 1])
        & (center[:, None, 1] <= g_xyxy[None, :, 3])
    )
    assoc = inside | (iou > iou_thr)
    gt_covered = assoc.any(dim=0)
    assigned = torch.full((k,), -1, dtype=torch.long, device=device)
    has_assoc = assoc.any(dim=1)
    if has_assoc.any():
        masked_iou = iou.clone()
        masked_iou[~assoc] = -1.0
        best_gt = masked_iou.argmax(dim=1)
        assigned[has_assoc] = best_gt[has_assoc]
    return gt_covered, assigned


def compute_min_cost_to_small_gt(selected_logits, selected_boxes, gt_boxes, gt_labels, gt_scales):
    small_mask = gt_scales == 0
    if selected_logits is None or selected_logits.numel() == 0 or small_mask.sum() == 0:
        return None
    small_boxes = gt_boxes[small_mask]
    small_labels = gt_labels[small_mask].long()
    prob = selected_logits.sigmoid()
    cls_prob = prob[:, small_labels]
    alpha, gamma = 0.25, 2.0
    neg = (1 - alpha) * (cls_prob ** gamma) * (-(1 - cls_prob + 1e-8).log())
    pos = alpha * ((1 - cls_prob) ** gamma) * (-(cls_prob + 1e-8).log())
    cost_class = pos - neg
    cost_bbox = torch.cdist(selected_boxes, small_boxes, p=1)
    cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(selected_boxes), box_cxcywh_to_xyxy(small_boxes))
    cost = 2.0 * cost_class + 5.0 * cost_bbox + 2.0 * cost_giou
    return cost.min(dim=1).values


def safe_ratio(num, den):
    den = float(den)
    return 0.0 if den == 0 else float(num) / den * 100.0


def set_dump_selected_queries(model):
    raw_model = model.module if hasattr(model, "module") else model
    for module in raw_model.modules():
        if hasattr(module, "dump_selected_queries"):
            module.dump_selected_queries = True


def build_solver(config_path: str, checkpoint_path: str):
    cfg = YAMLConfig(config_path, resume=checkpoint_path)
    if checkpoint_path is not None and "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    solver = TASKS[cfg.yaml_cfg["task"]](cfg)
    solver.eval()
    return solver


@torch.no_grad()
def collect_method(method, config_path, ckpt_path, device, out_dir, max_images=None, iou_thr=0.1, topn_save=80):
    print(f"\nCollecting method: {method}")
    solver = build_solver(config_path, ckpt_path)
    model = solver.ema.module if getattr(solver, "ema", None) is not None else solver.model
    model.to(device)
    model.eval()
    set_dump_selected_queries(model)

    raw_dir = Path(out_dir) / "raw" / method / "best"
    raw_dir.mkdir(parents=True, exist_ok=True)

    rows: Dict[int, dict] = {}
    image_count = 0
    for samples, targets in solver.val_dataloader:
        samples = samples.to(device) if hasattr(samples, "to") else samples
        targets = robust_to_device(targets, device)
        outputs = model(samples)
        if "selected_query_boxes" not in outputs:
            raise RuntimeError("selected_query_boxes not found. Please use the query-dump decoder.")
        selected_boxes_b = outputs["selected_query_boxes"]
        selected_scores_b = outputs.get("selected_query_scores", None)
        selected_indices_b = outputs.get("selected_query_indices", None)
        selected_logits_b = outputs.get("selected_query_logits", None)

        for b, target in enumerate(targets):
            image_id = get_image_id(target)
            selected_boxes = selected_boxes_b[b].detach()
            selected_scores = selected_scores_b[b].detach() if selected_scores_b is not None else torch.arange(selected_boxes.shape[0], device=device).float()
            selected_indices = selected_indices_b[b].detach() if selected_indices_b is not None else torch.arange(selected_boxes.shape[0], device=device)
            selected_logits = selected_logits_b[b].detach() if selected_logits_b is not None else None

            gt_boxes = normalize_gt_boxes(target).to(device)
            gt_labels = target.get("labels", torch.empty((0,), dtype=torch.long, device=device)).long().to(device)
            gt_scales = get_gt_scale_labels(gt_boxes, target).to(device)

            gt_covered, assigned = associate_queries_to_gt(selected_boxes, gt_boxes, iou_thr=iou_thr)
            valid_q = assigned >= 0
            assigned_scales = gt_scales[assigned[valid_q]] if valid_q.any() else torch.empty((0,), dtype=torch.long, device=device)

            scale_counts = [(gt_scales == i).sum().item() for i in range(3)]
            covered_counts = [((gt_scales == i) & gt_covered).sum().item() for i in range(3)]
            query_assoc_counts = [(assigned_scales == i).sum().item() for i in range(3)]
            total_gt = int(gt_boxes.shape[0])
            total_queries = int(selected_boxes.shape[0])

            small_q_mask = valid_q.clone()
            if valid_q.any():
                small_q_mask[valid_q] = gt_scales[assigned[valid_q]] == 0
            small_ranks = torch.arange(1, selected_boxes.shape[0] + 1, device=device).float()[small_q_mask]
            small_mean_rank = float(small_ranks.mean().item()) if small_ranks.numel() else float("nan")
            small_min_cost_all = compute_min_cost_to_small_gt(selected_logits, selected_boxes, gt_boxes, gt_labels, gt_scales)
            if small_min_cost_all is not None and small_q_mask.any():
                small_mean_min_cost = float(small_min_cost_all[small_q_mask].mean().item())
            else:
                small_mean_min_cost = float("nan")

            row = {
                "image_id": image_id,
                "method": method,
                "num_gt": total_gt,
                "num_small_gt": scale_counts[0],
                "num_medium_gt": scale_counts[1],
                "num_large_gt": scale_counts[2],
                "small_gt_coverage": safe_ratio(covered_counts[0], scale_counts[0]),
                "medium_gt_coverage": safe_ratio(covered_counts[1], scale_counts[1]),
                "large_gt_coverage": safe_ratio(covered_counts[2], scale_counts[2]),
                "small_query_ratio": safe_ratio(query_assoc_counts[0], total_queries),
                "medium_query_ratio": safe_ratio(query_assoc_counts[1], total_queries),
                "large_query_ratio": safe_ratio(query_assoc_counts[2], total_queries),
                "small_mean_rank": small_mean_rank,
                "small_mean_min_cost": small_mean_min_cost,
            }
            rows[image_id] = row

            # Save enough information for Figure 6 plotting.
            torch.save(
                {
                    "image_id": image_id,
                    "image": get_sample_image(samples, b),
                    "gt_boxes": gt_boxes.detach().cpu(),
                    "gt_labels": gt_labels.detach().cpu(),
                    "gt_scales": gt_scales.detach().cpu(),
                    "selected_query_boxes": selected_boxes[:topn_save].detach().cpu(),
                    "selected_query_scores": selected_scores[:topn_save].detach().cpu(),
                    "selected_query_indices": selected_indices[:topn_save].detach().cpu(),
                    "assigned_gt": assigned[:topn_save].detach().cpu(),
                    "metrics": row,
                },
                raw_dir / f"{image_id}.pt",
            )

            image_count += 1
            if max_images is not None and image_count >= max_images:
                return rows
    return rows


def is_finite(x):
    return x == x and abs(x) < 1e20


def merge_and_rank(baseline_rows, cmqs_rows):
    merged = []
    for image_id, b in baseline_rows.items():
        c = cmqs_rows.get(image_id)
        if c is None:
            continue
        small_cost_gain = 0.0
        if is_finite(b["small_mean_min_cost"]) and is_finite(c["small_mean_min_cost"]):
            small_cost_gain = b["small_mean_min_cost"] - c["small_mean_min_cost"]
        small_rank_gain = 0.0
        if is_finite(b["small_mean_rank"]) and is_finite(c["small_mean_rank"]):
            small_rank_gain = b["small_mean_rank"] - c["small_mean_rank"]
        row = {
            "image_id": image_id,
            "num_gt": b["num_gt"],
            "num_small_gt": b["num_small_gt"],
            "num_medium_gt": b["num_medium_gt"],
            "num_large_gt": b["num_large_gt"],
            "baseline_small_cost": b["small_mean_min_cost"],
            "cmqs_small_cost": c["small_mean_min_cost"],
            "small_cost_gain": small_cost_gain,
            "baseline_small_rank": b["small_mean_rank"],
            "cmqs_small_rank": c["small_mean_rank"],
            "small_rank_gain": small_rank_gain,
            "small_cov_gain": c["small_gt_coverage"] - b["small_gt_coverage"],
            "small_qratio_gain": c["small_query_ratio"] - b["small_query_ratio"],
            "large_qratio_drop": b["large_query_ratio"] - c["large_query_ratio"],
            "baseline_large_qratio": b["large_query_ratio"],
            "cmqs_large_qratio": c["large_query_ratio"],
        }
        # Objective selection scores.
        row["small_case_score"] = (
            5.0 * row["small_cost_gain"]
            + 0.5 * row["small_rank_gain"]
            + 0.2 * row["small_cov_gain"]
            + 0.2 * row["small_qratio_gain"]
        ) if row["num_small_gt"] >= 2 else -1e9
        row["crowded_score"] = (
            0.5 * row["num_gt"] + 2.0 * row["small_cost_gain"] + 0.1 * row["large_qratio_drop"]
        ) if row["num_gt"] >= 8 else -1e9
        row["failure_score"] = (
            row["large_qratio_drop"] - 2.0 * row["small_cost_gain"]
        ) if row["num_large_gt"] >= 1 else -1e9
        merged.append(row)
    return merged


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-config", default=BASELINE_CONFIG)
    parser.add_argument("--baseline-ckpt", default=BASELINE_CKPT)
    parser.add_argument("--cmqs-config", default=CMQS_CONFIG)
    parser.add_argument("--cmqs-ckpt", default=CMQS_CKPT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-dir", default=OUT_DIR)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--iou-thr", type=float, default=0.1)
    parser.add_argument("--topn-save", type=int, default=80)
    args = parser.parse_args()

    init_single_process_distributed()
    device = torch.device(args.device)
    print("Using device:", device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        baseline = collect_method("baseline", args.baseline_config, args.baseline_ckpt, device, out_dir,
                                  max_images=args.max_images, iou_thr=args.iou_thr, topn_save=args.topn_save)
        cmqs = collect_method("cmqs", args.cmqs_config, args.cmqs_ckpt, device, out_dir,
                              max_images=args.max_images, iou_thr=args.iou_thr, topn_save=args.topn_save)
        merged = merge_and_rank(baseline, cmqs)
        write_csv(out_dir / "per_image_query_metrics.csv", merged)

        small_sorted = sorted(merged, key=lambda x: x["small_case_score"], reverse=True)
        crowded_sorted = sorted(merged, key=lambda x: x["crowded_score"], reverse=True)
        failure_sorted = sorted(merged, key=lambda x: x["failure_score"], reverse=True)
        selected = {
            "small_object_case": small_sorted[0] if small_sorted else None,
            "crowded_scene": crowded_sorted[0] if crowded_sorted else None,
            "failure_case": failure_sorted[0] if failure_sorted else None,
        }
        with (out_dir / "figure6_selected_cases.json").open("w") as f:
            json.dump(selected, f, indent=2)
        print("\nRecommended Figure 6 cases:")
        for k, v in selected.items():
            print(k, "=>", None if v is None else v["image_id"], v)
        print("\nSaved:", out_dir / "per_image_query_metrics.csv")
        print("Saved:", out_dir / "figure6_selected_cases.json")
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
