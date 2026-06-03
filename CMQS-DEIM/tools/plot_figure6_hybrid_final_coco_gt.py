#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Final hybrid Figure 6 visualization for DEIM-L baseline vs CMQS.

This script creates a 3-row x 3-column Figure 6:

Row 1: crowded scene, using difference visualization
    Image + GT | Full difference | GT-associated / local

Row 2: small-object scene, using difference visualization
    Image + GT | Full difference | GT-associated / local

Row 3: failure case, using the old direct overlay visualization
    Image + GT | DEIM-L baseline | CMQS

Default image ids follow the current recommended setting:
    crowded: 166918
    small-object: 498919
    failure: 547383  # sheep case from the earlier old-style Figure 6

If image_id=547383 is not available in your raw dumps, use --failure-id 514979
or any other failure image id you prefer.

Expected raw query dumps:
    <raw_dir>/baseline/best/<image_id>.pt
    <raw_dir>/cmqs/best/<image_id>.pt

The .pt files should contain one of:
    selected_query_boxes, query_boxes, boxes
in normalized cxcywh format.

Example command:
CUDA_VISIBLE_DEVICES=1 python plot_figure6_hybrid_final_coco_gt.py \
  --raw-dir "outputs/query_vis_auto/raw" \
  --coco-ann "/media/hanyong/zgw/zp/deim 2/annotations/instances_val2017.json" \
  --coco-img-dir "/media/hanyong/zgw/zp/val2017" \
  --crowded-id 166918 \
  --small-id 498919 \
  --failure-id 547383 \
  --topn 30 \
  --failure-topn 80 \
  --common-center-thr 0.025 \
  --out-dir "figures/figure6_hybrid_final" \
  --debug
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from PIL import Image
import torch


# -----------------------------------------------------------------------------
# Box utilities
# -----------------------------------------------------------------------------

def cxcywh_to_xyxy_norm(boxes: torch.Tensor) -> torch.Tensor:
    boxes = boxes.float()
    if boxes.numel() == 0:
        return boxes.reshape(0, 4)
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1).clamp(0, 1)


def xywh_abs_to_xyxy_norm(bbox: Sequence[float], img_w: float, img_h: float) -> List[float]:
    x, y, w, h = bbox
    return [
        max(0.0, float(x)) / img_w,
        max(0.0, float(y)) / img_h,
        min(float(img_w), float(x) + float(w)) / img_w,
        min(float(img_h), float(y) + float(h)) / img_h,
    ]


def box_iou_xyxy_norm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.numel() == 0 or b.numel() == 0:
        return torch.zeros((a.shape[0], b.shape[0]), dtype=torch.float32)
    lt = torch.maximum(a[:, None, :2], b[None, :, :2])
    rb = torch.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    area_a = (a[:, 2] - a[:, 0]).clamp(min=0) * (a[:, 3] - a[:, 1]).clamp(min=0)
    area_b = (b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0)
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / union.clamp(min=1e-9)


def box_centers_xyxy(q_xyxy: torch.Tensor) -> torch.Tensor:
    if q_xyxy.numel() == 0:
        return torch.zeros((0, 2), dtype=torch.float32)
    return 0.5 * (q_xyxy[:, :2] + q_xyxy[:, 2:])


def intersects_xyxy(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.numel() == 0:
        return torch.zeros((0,), dtype=torch.bool)
    return (a[:, 0] < b[2]) & (a[:, 2] > b[0]) & (a[:, 1] < b[3]) & (a[:, 3] > b[1])


# -----------------------------------------------------------------------------
# COCO GT loader
# -----------------------------------------------------------------------------

class CocoGT:
    def __init__(self, ann_path: str, img_dir: str):
        self.img_dir = Path(img_dir)
        with open(ann_path, "r", encoding="utf-8") as f:
            coco = json.load(f)
        self.images: Dict[int, dict] = {int(im["id"]): im for im in coco["images"]}
        self.anns: Dict[int, list] = defaultdict(list)
        for ann in coco["annotations"]:
            if ann.get("ignore", 0) or ann.get("iscrowd", 0):
                continue
            self.anns[int(ann["image_id"])].append(ann)

    def load_image(self, image_id: int) -> Tuple[Image.Image, dict]:
        info = self.images[int(image_id)]
        img_path = self.img_dir / info["file_name"]
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")
        return Image.open(img_path).convert("RGB"), info

    def gt_boxes_norm_and_scales(self, image_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        info = self.images[int(image_id)]
        img_w, img_h = float(info["width"]), float(info["height"])
        boxes, scales = [], []
        for ann in self.anns[int(image_id)]:
            boxes.append(xywh_abs_to_xyxy_norm(ann["bbox"], img_w, img_h))
            area = float(ann.get("area", ann["bbox"][2] * ann["bbox"][3]))
            if area < 32 ** 2:
                scales.append(0)       # small
            elif area < 96 ** 2:
                scales.append(1)       # medium
            else:
                scales.append(2)       # large
        if not boxes:
            return torch.zeros((0, 4), dtype=torch.float32), torch.zeros((0,), dtype=torch.long)
        return torch.tensor(boxes, dtype=torch.float32), torch.tensor(scales, dtype=torch.long)


# -----------------------------------------------------------------------------
# Query loading and association
# -----------------------------------------------------------------------------

def load_query_xyxy(pt_path: Path) -> torch.Tensor:
    if not pt_path.exists():
        raise FileNotFoundError(f"Raw query dump not found: {pt_path}")
    data = torch.load(pt_path, map_location="cpu")
    if isinstance(data, torch.Tensor):
        boxes = data
    elif "selected_query_boxes" in data:
        boxes = data["selected_query_boxes"]
    elif "query_boxes" in data:
        boxes = data["query_boxes"]
    elif "boxes" in data:
        boxes = data["boxes"]
    else:
        raise KeyError(f"Cannot find selected_query_boxes/query_boxes/boxes in {pt_path}. Keys={list(data.keys())}")
    return cxcywh_to_xyxy_norm(boxes.float())


def associate_query_to_gt(q_xyxy: torch.Tensor, gt_xyxy: torch.Tensor, iou_thr: float = 0.1) -> torch.Tensor:
    """Assign each query to one GT index. -1 means no association."""
    assigned = torch.full((q_xyxy.shape[0],), -1, dtype=torch.long)
    if q_xyxy.numel() == 0 or gt_xyxy.numel() == 0:
        return assigned

    centers = box_centers_xyxy(q_xyxy)
    inside = (
        (centers[:, None, 0] >= gt_xyxy[None, :, 0])
        & (centers[:, None, 0] <= gt_xyxy[None, :, 2])
        & (centers[:, None, 1] >= gt_xyxy[None, :, 1])
        & (centers[:, None, 1] <= gt_xyxy[None, :, 3])
    )
    iou = box_iou_xyxy_norm(q_xyxy, gt_xyxy)
    assoc = inside | (iou >= iou_thr)
    if not assoc.any():
        return assigned

    score = iou.clone()
    score[inside & (score <= 0)] = 1e-6
    score[~assoc] = -1.0
    best = score.argmax(dim=1)
    has = assoc.any(dim=1)
    assigned[has] = best[has]
    return assigned


def filter_queries(
    q_xyxy: torch.Tensor,
    assigned: torch.Tensor,
    gt_scales: torch.Tensor,
    query_filter: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return filtered boxes and original ranking indices."""
    ranks = torch.arange(q_xyxy.shape[0], dtype=torch.long)
    if query_filter == "all":
        return q_xyxy, ranks
    if assigned.numel() == 0:
        return q_xyxy[:0], ranks[:0]
    if query_filter == "associated":
        mask = assigned >= 0
    elif query_filter == "small_associated":
        mask = (assigned >= 0) & (gt_scales[assigned.clamp(min=0)] == 0)
    elif query_filter == "medium_large_associated":
        mask = (assigned >= 0) & (gt_scales[assigned.clamp(min=0)] > 0)
    else:
        raise ValueError(f"Unknown query_filter: {query_filter}")
    return q_xyxy[mask], ranks[mask]


# -----------------------------------------------------------------------------
# Difference matching
# -----------------------------------------------------------------------------

def match_common_by_center(
    baseline_q: torch.Tensor,
    cmqs_q: torch.Tensor,
    baseline_ranks: torch.Tensor,
    cmqs_ranks: torch.Tensor,
    center_thr: float,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """Greedy one-to-one matching of query centers by normalized center distance."""
    if baseline_q.numel() == 0:
        return [], [], list(range(cmqs_q.shape[0]))
    if cmqs_q.numel() == 0:
        return [], list(range(baseline_q.shape[0])), []

    b_centers = box_centers_xyxy(baseline_q)
    c_centers = box_centers_xyxy(cmqs_q)
    dist_mat = torch.cdist(b_centers, c_centers, p=2)

    candidates = []
    nb, nc = dist_mat.shape
    for i in range(nb):
        for j in range(nc):
            d = float(dist_mat[i, j])
            if d <= center_thr:
                rank_gap = abs(int(baseline_ranks[i]) - int(cmqs_ranks[j]))
                candidates.append((d, rank_gap, i, j))
    candidates.sort(key=lambda x: (x[0], x[1]))

    used_b, used_c = set(), set()
    pairs: List[Tuple[int, int]] = []
    for _, _, i, j in candidates:
        if i not in used_b and j not in used_c:
            used_b.add(i)
            used_c.add(j)
            pairs.append((i, j))

    b_only = [i for i in range(nb) if i not in used_b]
    c_only = [j for j in range(nc) if j not in used_c]
    return pairs, b_only, c_only


def rank_marker_size(rank: int) -> float:
    """Larger, publication-friendly point sizes."""
    if rank < 10:
        return 78.0
    if rank < 30:
        return 58.0
    return 38.0


# -----------------------------------------------------------------------------
# Crop utilities
# -----------------------------------------------------------------------------

def choose_focus_gt(
    gt_xyxy: torch.Tensor,
    gt_scales: torch.Tensor,
    base_assign: torch.Tensor,
    cmqs_assign: torch.Tensor,
    row_label: str,
) -> Optional[int]:
    if gt_xyxy.numel() == 0:
        return None
    lower = row_label.lower()
    prefer_small = ("small" in lower) or ("crowded" in lower)
    best_i, best_score = 0, -1e9
    for i in range(gt_xyxy.shape[0]):
        cb = int((base_assign == i).sum().item())
        cc = int((cmqs_assign == i).sum().item())
        diff = abs(cc - cb)
        total = cb + cc
        scale = int(gt_scales[i].item())
        if prefer_small:
            score = 2.2 * diff + 0.35 * total + (3.0 if scale == 0 else 0.0)
        else:
            score = 2.0 * diff + 0.40 * total + (2.0 if scale > 0 else 0.0)
        if score > best_score:
            best_i, best_score = i, score
    return best_i


def make_crop_box(
    gt_xyxy: torch.Tensor,
    gt_scales: torch.Tensor,
    base_q_all: torch.Tensor,
    cmqs_q_all: torch.Tensor,
    base_assign: torch.Tensor,
    cmqs_assign: torch.Tensor,
    row_label: str,
    margin: float = 0.18,
    min_size: float = 0.38,
) -> torch.Tensor:
    focus = choose_focus_gt(gt_xyxy, gt_scales, base_assign, cmqs_assign, row_label)
    if focus is None:
        return torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=torch.float32)

    boxes = [gt_xyxy[focus].reshape(1, 4)]
    if base_q_all.numel() > 0:
        boxes.append(base_q_all[base_assign == focus])
    if cmqs_q_all.numel() > 0:
        boxes.append(cmqs_q_all[cmqs_assign == focus])
    boxes = [b for b in boxes if b.numel() > 0]
    union = torch.cat(boxes, dim=0)

    x0, y0 = union[:, 0].min().item(), union[:, 1].min().item()
    x1, y1 = union[:, 2].max().item(), union[:, 3].max().item()
    w, h = max(x1 - x0, 0.03), max(y1 - y0, 0.03)
    x0 -= margin * w
    x1 += margin * w
    y0 -= margin * h
    y1 += margin * h

    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    w, h = max(x1 - x0, min_size), max(y1 - y0, min_size)
    x0, x1 = cx - w / 2, cx + w / 2
    y0, y1 = cy - h / 2, cy + h / 2

    if x0 < 0:
        x1 -= x0
        x0 = 0
    if y0 < 0:
        y1 -= y0
        y0 = 0
    if x1 > 1:
        x0 -= x1 - 1
        x1 = 1
    if y1 > 1:
        y0 -= y1 - 1
        y1 = 1
    return torch.tensor([max(0, x0), max(0, y0), min(1, x1), min(1, y1)], dtype=torch.float32)


def crop_image(image: Image.Image, crop_norm: Optional[torch.Tensor]) -> Tuple[Image.Image, Optional[torch.Tensor]]:
    if crop_norm is None:
        return image, None
    W, H = image.size
    x0, y0, x1, y1 = crop_norm.tolist()
    px0, py0 = int(round(x0 * W)), int(round(y0 * H))
    px1, py1 = int(round(x1 * W)), int(round(y1 * H))
    px0, py0 = max(0, px0), max(0, py0)
    px1, py1 = min(W, px1), min(H, py1)
    if px1 <= px0 or py1 <= py0:
        return image, None
    return image.crop((px0, py0, px1, py1)), crop_norm


def transform_box_for_crop(box: torch.Tensor, crop_norm: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if crop_norm is None:
        return box.clone()
    c = crop_norm
    if not bool(intersects_xyxy(box.reshape(1, 4), c)[0].item()):
        return None
    denom_w = max(float(c[2] - c[0]), 1e-9)
    denom_h = max(float(c[3] - c[1]), 1e-9)
    x0 = (max(float(box[0]), float(c[0])) - float(c[0])) / denom_w
    y0 = (max(float(box[1]), float(c[1])) - float(c[1])) / denom_h
    x1 = (min(float(box[2]), float(c[2])) - float(c[0])) / denom_w
    y1 = (min(float(box[3]), float(c[3])) - float(c[1])) / denom_h
    return torch.tensor([x0, y0, x1, y1], dtype=torch.float32).clamp(0, 1)


def transform_center_for_crop(center: torch.Tensor, crop_norm: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if crop_norm is None:
        return center.clone()
    c = crop_norm
    x, y = float(center[0]), float(center[1])
    if not (float(c[0]) <= x <= float(c[2]) and float(c[1]) <= y <= float(c[3])):
        return None
    denom_w = max(float(c[2] - c[0]), 1e-9)
    denom_h = max(float(c[3] - c[1]), 1e-9)
    return torch.tensor([(x - float(c[0])) / denom_w, (y - float(c[1])) / denom_h], dtype=torch.float32)


# -----------------------------------------------------------------------------
# Drawing helpers
# -----------------------------------------------------------------------------

def prepare_axis_image(ax, image: Image.Image, title: str) -> Tuple[int, int]:
    """Show image without padding/letterbox and force axes to image extent."""
    ax.imshow(image, aspect="auto")
    ax.set_title(title, fontsize=16, fontweight="bold", pad=6)
    ax.set_xticks([])
    ax.set_yticks([])
    W, H = image.size
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_aspect("auto")
    return W, H


def draw_gt_boxes(ax, gt_xyxy: torch.Tensor, gt_scales: torch.Tensor, W: int, H: int, crop_used: Optional[torch.Tensor]):
    for i, b in enumerate(gt_xyxy):
        bb = transform_box_for_crop(b, crop_used)
        if bb is None:
            continue
        x0, y0, x1, y1 = bb.tolist()
        edge = "red" if int(gt_scales[i]) == 0 else "yellow"
        ax.add_patch(patches.Rectangle(
            (x0 * W, y0 * H), (x1 - x0) * W, (y1 - y0) * H,
            linewidth=1.6, edgecolor=edge, facecolor="none", alpha=0.95, zorder=2,
        ))


def draw_image_gt_panel(ax, image: Image.Image, gt_xyxy: torch.Tensor, gt_scales: torch.Tensor, title: str, crop_norm=None):
    img_show, crop_used = crop_image(image, crop_norm)
    W, H = prepare_axis_image(ax, img_show, title)
    draw_gt_boxes(ax, gt_xyxy, gt_scales, W, H, crop_used)


def draw_difference_panel(
    ax,
    image: Image.Image,
    gt_xyxy: torch.Tensor,
    gt_scales: torch.Tensor,
    baseline_q: torch.Tensor,
    cmqs_q: torch.Tensor,
    baseline_ranks: torch.Tensor,
    cmqs_ranks: torch.Tensor,
    title: str,
    common_center_thr: float,
    crop_norm: Optional[torch.Tensor] = None,
    show_common: bool = True,
    show_counts: bool = True,
):
    img_show, crop_used = crop_image(image, crop_norm)
    W, H = prepare_axis_image(ax, img_show, title)
    draw_gt_boxes(ax, gt_xyxy, gt_scales, W, H, crop_used)

    pairs, b_only, c_only = match_common_by_center(
        baseline_q, cmqs_q, baseline_ranks, cmqs_ranks, common_center_thr
    )

    # Common points: larger, with clear black outline.
    if show_common and pairs:
        xs, ys, ss = [], [], []
        for i, j in pairs:
            p = 0.5 * (box_centers_xyxy(baseline_q[i:i + 1])[0] + box_centers_xyxy(cmqs_q[j:j + 1])[0])
            pp = transform_center_for_crop(p, crop_used)
            if pp is None:
                continue
            xs.append(float(pp[0]) * W)
            ys.append(float(pp[1]) * H)
            ss.append(max(rank_marker_size(int(baseline_ranks[i])), rank_marker_size(int(cmqs_ranks[j]))))
        if xs:
            ax.scatter(xs, ys, s=ss, c="lightgray", edgecolors="black", linewidths=0.85,
                       alpha=0.92, zorder=4)

    # DEIM-L only.
    if b_only:
        xs, ys, ss = [], [], []
        centers = box_centers_xyxy(baseline_q)
        for i in b_only:
            pp = transform_center_for_crop(centers[i], crop_used)
            if pp is None:
                continue
            xs.append(float(pp[0]) * W)
            ys.append(float(pp[1]) * H)
            ss.append(rank_marker_size(int(baseline_ranks[i])) * 1.08)
        if xs:
            ax.scatter(xs, ys, s=ss, c="orange", edgecolors="black", linewidths=0.9,
                       alpha=0.98, zorder=5)

    # CMQS only.
    if c_only:
        xs, ys, ss = [], [], []
        centers = box_centers_xyxy(cmqs_q)
        for j in c_only:
            pp = transform_center_for_crop(centers[j], crop_used)
            if pp is None:
                continue
            xs.append(float(pp[0]) * W)
            ys.append(float(pp[1]) * H)
            ss.append(rank_marker_size(int(cmqs_ranks[j])) * 1.08)
        if xs:
            ax.scatter(xs, ys, s=ss, c="cyan", edgecolors="black", linewidths=0.9,
                       alpha=0.98, zorder=6)

    if show_counts:
        txt = f"common {len(pairs)} | base-only {len(b_only)} | CMQS-only {len(c_only)}"
        ax.text(
            0.012, 0.985, txt,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9.8,
            fontweight="bold",
            color="black",
            bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="black", alpha=0.90, linewidth=0.85),
            zorder=10,
        )

    return len(pairs), len(b_only), len(c_only)


def draw_query_boxes_overlay(
    ax,
    image: Image.Image,
    gt_xyxy: torch.Tensor,
    gt_scales: torch.Tensor,
    q_xyxy: torch.Tensor,
    title: str,
    max_boxes: int,
):
    """Old-style direct overlay for the failure case."""
    W, H = prepare_axis_image(ax, image, title)
    draw_gt_boxes(ax, gt_xyxy, gt_scales, W, H, crop_used=None)

    q_xyxy = q_xyxy[:max_boxes]
    for rank, b in enumerate(q_xyxy):
        x0, y0, x1, y1 = b.tolist()
        lw = 1.8 if rank < 20 else (1.25 if rank < 50 else 0.85)
        alpha = 0.90 if rank < 30 else 0.62
        ax.add_patch(patches.Rectangle(
            (x0 * W, y0 * H), (x1 - x0) * W, (y1 - y0) * H,
            linewidth=lw, edgecolor="cyan", facecolor="none", alpha=alpha, zorder=3,
        ))

    centers = box_centers_xyxy(q_xyxy)
    if centers.numel() > 0:
        xs = (centers[:, 0] * W).tolist()
        ys = (centers[:, 1] * H).tolist()
        sizes = [34 if i < 20 else 20 for i in range(len(xs))]
        ax.scatter(xs, ys, s=sizes, c="white", edgecolors="black", linewidths=0.45,
                   alpha=0.92, zorder=5)


def add_hybrid_legend(fig):
    handles = [
        Line2D([0], [0], marker="o", color="none", label="Common", markerfacecolor="lightgray",
               markeredgecolor="black", markersize=7),
        Line2D([0], [0], marker="o", color="none", label="DEIM-L only", markerfacecolor="orange",
               markeredgecolor="black", markersize=7),
        Line2D([0], [0], marker="o", color="none", label="CMQS only", markerfacecolor="cyan",
               markeredgecolor="black", markersize=7),
        Line2D([0], [0], color="cyan", label="Selected query box", linewidth=1.8),
        Line2D([0], [0], color="red", label="Small GT", linewidth=1.6),
        Line2D([0], [0], color="yellow", label="Medium/Large GT", linewidth=1.6),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=10.0,
               frameon=False, bbox_to_anchor=(0.5, 0.006))


# -----------------------------------------------------------------------------
# Main plotting
# -----------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Create final hybrid Figure 6 visualization.")
    parser.add_argument("--raw-dir", required=True, help="outputs/query_vis_auto/raw")
    parser.add_argument("--coco-ann", required=True)
    parser.add_argument("--coco-img-dir", required=True)

    parser.add_argument("--crowded-id", type=int, default=166918)
    parser.add_argument("--small-id", type=int, default=498919)
    parser.add_argument("--failure-id", type=int, default=547383)
    parser.add_argument("--row-labels", default="Crowded scene;Small-object case;Failure case")

    parser.add_argument("--topn", type=int, default=30, help="Top-N selected queries for difference rows.")
    parser.add_argument("--failure-topn", type=int, default=80, help="Top-N query boxes for old-style failure overlay.")
    parser.add_argument("--common-center-thr", type=float, default=0.025)
    parser.add_argument("--assoc-iou-thr", type=float, default=0.1)
    parser.add_argument("--full-query-filter", choices=["all", "associated", "small_associated", "medium_large_associated"], default="all")
    parser.add_argument("--local-query-filter", choices=["all", "associated", "small_associated", "medium_large_associated"], default="associated")
    parser.add_argument("--crop-margin", type=float, default=0.18)
    parser.add_argument("--min-crop-size", type=float, default=0.38)
    parser.add_argument("--hide-common", action="store_true")
    parser.add_argument("--hide-counts", action="store_true")
    parser.add_argument("--dpi", type=int, default=280)
    parser.add_argument("--out-dir", default="figures/figure6_hybrid_final")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def load_pair_queries(raw_dir: Path, image_id: int, topn: int) -> Tuple[torch.Tensor, torch.Tensor]:
    base = load_query_xyxy(raw_dir / "baseline" / "best" / f"{image_id}.pt")[:topn]
    cmqs = load_query_xyxy(raw_dir / "cmqs" / "best" / f"{image_id}.pt")[:topn]
    return base, cmqs


def main():
    args = parse_args()
    labels = [x.strip() for x in args.row_labels.split(";") if x.strip()]
    if len(labels) != 3:
        raise ValueError("--row-labels must contain exactly three labels separated by ';'.")

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    coco = CocoGT(args.coco_ann, args.coco_img_dir)

    row_specs = [
        {"kind": "difference", "image_id": args.crowded_id, "label": labels[0]},
        {"kind": "difference", "image_id": args.small_id, "label": labels[1]},
        {"kind": "old_failure", "image_id": args.failure_id, "label": labels[2]},
    ]

    fig, axes = plt.subplots(3, 3, figsize=(15.4, 10.7), dpi=args.dpi)
    index_rows = []

    for r, spec in enumerate(row_specs):
        image_id = int(spec["image_id"])
        label = str(spec["label"])
        image, info = coco.load_image(image_id)
        gt_xyxy, gt_scales = coco.gt_boxes_norm_and_scales(image_id)

        if args.debug:
            print(f"[DEBUG] row={r + 1}, id={image_id}, label={label}, file={info.get('file_name')}, "
                  f"GT={len(gt_scales)}, small={int((gt_scales == 0).sum())}")

        if spec["kind"] == "difference":
            base_all, cmqs_all = load_pair_queries(raw_dir, image_id, args.topn)
            base_assign = associate_query_to_gt(base_all, gt_xyxy, iou_thr=args.assoc_iou_thr)
            cmqs_assign = associate_query_to_gt(cmqs_all, gt_xyxy, iou_thr=args.assoc_iou_thr)

            base_full, base_full_ranks = filter_queries(base_all, base_assign, gt_scales, args.full_query_filter)
            cmqs_full, cmqs_full_ranks = filter_queries(cmqs_all, cmqs_assign, gt_scales, args.full_query_filter)
            base_local, base_local_ranks = filter_queries(base_all, base_assign, gt_scales, args.local_query_filter)
            cmqs_local, cmqs_local_ranks = filter_queries(cmqs_all, cmqs_assign, gt_scales, args.local_query_filter)

            crop_norm = make_crop_box(
                gt_xyxy, gt_scales, base_all, cmqs_all, base_assign, cmqs_assign,
                row_label=label, margin=args.crop_margin, min_size=args.min_crop_size,
            )

            draw_image_gt_panel(axes[r, 0], image, gt_xyxy, gt_scales, "Image + GT")
            full_counts = draw_difference_panel(
                axes[r, 1], image, gt_xyxy, gt_scales,
                base_full, cmqs_full, base_full_ranks, cmqs_full_ranks,
                "Full difference",
                common_center_thr=args.common_center_thr,
                crop_norm=None,
                show_common=not args.hide_common,
                show_counts=not args.hide_counts,
            )
            local_counts = draw_difference_panel(
                axes[r, 2], image, gt_xyxy, gt_scales,
                base_local, cmqs_local, base_local_ranks, cmqs_local_ranks,
                "GT-associated / local",
                common_center_thr=args.common_center_thr,
                crop_norm=crop_norm,
                show_common=not args.hide_common,
                show_counts=not args.hide_counts,
            )

            index_rows.append({
                "row": r + 1,
                "image_id": image_id,
                "label": label,
                "visualization": "difference",
                "file_name": info.get("file_name", ""),
                "num_gt": int(len(gt_scales)),
                "num_small_gt": int((gt_scales == 0).sum()),
                "topn": args.topn,
                "full_common": full_counts[0],
                "full_baseline_only": full_counts[1],
                "full_cmqs_only": full_counts[2],
                "local_common": local_counts[0],
                "local_baseline_only": local_counts[1],
                "local_cmqs_only": local_counts[2],
            })

        else:
            # Old-style direct overlay for the failure row.
            base_fail, cmqs_fail = load_pair_queries(raw_dir, image_id, args.failure_topn)
            draw_image_gt_panel(axes[r, 0], image, gt_xyxy, gt_scales, "Image + GT")
            draw_query_boxes_overlay(axes[r, 1], image, gt_xyxy, gt_scales, base_fail,
                                     "DEIM-L baseline", max_boxes=args.failure_topn)
            draw_query_boxes_overlay(axes[r, 2], image, gt_xyxy, gt_scales, cmqs_fail,
                                     "CMQS", max_boxes=args.failure_topn)
            index_rows.append({
                "row": r + 1,
                "image_id": image_id,
                "label": label,
                "visualization": "old_failure_overlay",
                "file_name": info.get("file_name", ""),
                "num_gt": int(len(gt_scales)),
                "num_small_gt": int((gt_scales == 0).sum()),
                "topn": args.failure_topn,
                "full_common": "-",
                "full_baseline_only": "-",
                "full_cmqs_only": "-",
                "local_common": "-",
                "local_baseline_only": "-",
                "local_cmqs_only": "-",
            })

        ylabel = label
        axes[r, 0].set_ylabel(ylabel, fontsize=16, fontweight="bold", rotation=90, labelpad=12)
        axes[r, 0].yaxis.set_label_coords(-0.075, 0.5)

    add_hybrid_legend(fig)
    plt.tight_layout(rect=(0.016, 0.06, 1.0, 1.0), pad=0.55, w_pad=0.42, h_pad=0.65)

    out_png = out_dir / "figure6_hybrid_final.png"
    out_pdf = out_dir / "figure6_hybrid_final.pdf"
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    out_csv = out_dir / "figure6_hybrid_final_index.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(index_rows[0].keys()))
        writer.writeheader()
        writer.writerows(index_rows)

    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")
    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()
