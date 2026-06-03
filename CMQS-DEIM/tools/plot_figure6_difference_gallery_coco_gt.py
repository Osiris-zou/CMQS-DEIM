#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 6 final-stage difference visualization for DEIM-L vs CMQS.

Purpose
-------
The old Figure 6 style, "Image+GT | DEIM-L | CMQS" with many cyan query boxes,
can be too cluttered in crowded or small-object scenes. This script visualizes
query-selection differences directly at the final/best checkpoint:

  Column 1: Image + COCO GT boxes
  Column 2: Full-image difference map of selected query centers
  Column 3: Local/GT-associated difference map, optionally cropped around the
            most informative GT region

Point meanings:
  light gray / white = query centers selected by both DEIM-L and CMQS
  orange             = query centers selected only by DEIM-L
  cyan               = query centers selected only by CMQS

GT boxes:
  red    = small objects (COCO area < 32^2)
  yellow = medium / large objects

Expected raw query dumps
------------------------
The script reads raw dumps produced by auto_select_query_vis_cases.py:

  <raw_dir>/baseline/best/<image_id>.pt
  <raw_dir>/cmqs/best/<image_id>.pt

Each .pt file should contain:

  selected_query_boxes: Tensor[K, 4] in normalized cxcywh format

Example: final Figure 6 from manually selected images
-----------------------------------------------------
python plot_figure6_difference_gallery_coco_gt.py \
  --raw-dir outputs/query_vis_auto/raw \
  --coco-ann "/media/hanyong/zgw/zp/deim 2/annotations/instances_val2017.json" \
  --coco-img-dir "/media/hanyong/zgw/zp/val2017" \
  --image-ids "91406,498919,264335,514979" \
  --row-labels "Crowded subtle case;Small-object subtle case;Limitation case 1;Limitation case 2" \
  --topn 30 \
  --common-center-thr 0.025 \
  --full-query-filter all \
  --local-query-filter associated \
  --out-dir figures/figure6_difference_final \
  --debug

Example: candidate gallery from auto-ranked pools
------------------------------------------------
python plot_figure6_difference_gallery_coco_gt.py \
  --candidate-json outputs/query_vis_auto/figure_candidate_pool.json \
  --raw-dir outputs/query_vis_auto/raw \
  --coco-ann "/media/hanyong/zgw/zp/deim 2/annotations/instances_val2017.json" \
  --coco-img-dir "/media/hanyong/zgw/zp/val2017" \
  --categories "crowded_positive,small_positive,balanced_small_scene,failure_case" \
  --max-per-category 6 \
  --rows-per-page 4 \
  --topn 30 \
  --out-dir figures/figure6_difference_gallery \
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
# Basic box utilities
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
# Query loading and filtering
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
    """Greedy one-to-one matching of query centers by normalized center distance.

    Returned indices refer to the filtered baseline_q/cmqs_q arrays.
    """
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
                # Tie-break by similar rank if possible.
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
    if rank < 10:
        return 34.0
    if rank < 30:
        return 22.0
    return 12.0


# -----------------------------------------------------------------------------
# Crop utilities
# -----------------------------------------------------------------------------

def choose_focus_gt(
    gt_xyxy: torch.Tensor,
    gt_scales: torch.Tensor,
    base_assign: torch.Tensor,
    cmqs_assign: torch.Tensor,
    category: str,
) -> Optional[int]:
    if gt_xyxy.numel() == 0:
        return None
    is_failure = "failure" in category.lower() or "limitation" in category.lower()
    best_i, best_score = 0, -1e9
    for i in range(gt_xyxy.shape[0]):
        cb = int((base_assign == i).sum().item())
        cc = int((cmqs_assign == i).sum().item())
        diff = abs(cc - cb)
        total = cb + cc
        scale = int(gt_scales[i].item())
        if is_failure:
            # For limitation cases, prefer medium/large objects with redistribution.
            score = 2.0 * diff + 0.4 * total + (2.0 if scale > 0 else 0.0)
        else:
            # For crowded/small-object examples, prefer small objects with local differences.
            score = 2.2 * diff + 0.35 * total + (3.0 if scale == 0 else 0.0)
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
    category: str,
    margin: float = 0.18,
    min_size: float = 0.38,
) -> torch.Tensor:
    focus = choose_focus_gt(gt_xyxy, gt_scales, base_assign, cmqs_assign, category)
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
# Drawing
# -----------------------------------------------------------------------------

def draw_gt_boxes(ax, gt_xyxy: torch.Tensor, gt_scales: torch.Tensor, W: int, H: int, crop_used: Optional[torch.Tensor]):
    for i, b in enumerate(gt_xyxy):
        bb = transform_box_for_crop(b, crop_used)
        if bb is None:
            continue
        x0, y0, x1, y1 = bb.tolist()
        edge = "red" if int(gt_scales[i]) == 0 else "yellow"
        ax.add_patch(patches.Rectangle(
            (x0 * W, y0 * H), (x1 - x0) * W, (y1 - y0) * H,
            linewidth=1.35, edgecolor=edge, facecolor="none", alpha=0.90, zorder=2,
        ))


def draw_image_gt_panel(
    ax,
    image: Image.Image,
    gt_xyxy: torch.Tensor,
    gt_scales: torch.Tensor,
    title: str,
    crop_norm: Optional[torch.Tensor] = None,
):
    img_show, crop_used = crop_image(image, crop_norm)
    ax.imshow(img_show, aspect="auto")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=6)
    ax.set_xticks([])
    ax.set_yticks([])
    W, H = img_show.size
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
    ax.imshow(img_show, aspect="auto")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=6)
    ax.set_xticks([])
    ax.set_yticks([])
    W, H = img_show.size

    draw_gt_boxes(ax, gt_xyxy, gt_scales, W, H, crop_used)

    pairs, b_only, c_only = match_common_by_center(
        baseline_q, cmqs_q, baseline_ranks, cmqs_ranks, common_center_thr
    )

    # Common points.
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
            ax.scatter(xs, ys, s=ss, c="lightgray", edgecolors="black", linewidths=0.25,
                       alpha=0.74, zorder=4)

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
            ss.append(rank_marker_size(int(baseline_ranks[i])))
        if xs:
            ax.scatter(xs, ys, s=ss, c="orange", edgecolors="black", linewidths=0.22,
                       alpha=0.88, zorder=5)

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
            ss.append(rank_marker_size(int(cmqs_ranks[j])))
        if xs:
            ax.scatter(xs, ys, s=ss, c="cyan", edgecolors="black", linewidths=0.22,
                       alpha=0.88, zorder=6)

    if show_counts:
        txt = f"common {len(pairs)} | base-only {len(b_only)} | CMQS-only {len(c_only)}"
        ax.text(
            0.012, 0.985, txt,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9.0,
            fontweight="bold",
            color="black",
            bbox=dict(boxstyle="round,pad=0.20", facecolor="white", edgecolor="black", alpha=0.88, linewidth=0.75),
            zorder=10,
        )

    return len(pairs), len(b_only), len(c_only)


def add_legend(fig, show_common: bool = True):
    handles = []
    if show_common:
        handles.append(Line2D([0], [0], marker="o", color="none", label="Common", markerfacecolor="lightgray",
                              markeredgecolor="black", markersize=6))
    handles.extend([
        Line2D([0], [0], marker="o", color="none", label="DEIM-L only", markerfacecolor="orange",
               markeredgecolor="black", markersize=6),
        Line2D([0], [0], marker="o", color="none", label="CMQS only", markerfacecolor="cyan",
               markeredgecolor="black", markersize=6),
        Line2D([0], [0], color="red", label="Small GT", linewidth=1.5),
        Line2D([0], [0], color="yellow", label="Medium/Large GT", linewidth=1.5),
    ])
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=10.0, frameon=False,
               bbox_to_anchor=(0.5, 0.006))


# -----------------------------------------------------------------------------
# Candidate loading
# -----------------------------------------------------------------------------

def load_items_from_candidate_json(path: str, categories: List[str], max_per_category: int) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        pool = json.load(f)
    items = []
    for cat in categories:
        if cat not in pool:
            print(f"[WARN] category not found in candidate json: {cat}")
            continue
        for rank, r in enumerate(pool[cat][:max_per_category], start=1):
            if isinstance(r, dict):
                item = dict(r)
            else:
                item = {"image_id": int(r)}
            item["category"] = cat
            item["rank_in_category"] = rank
            items.append(item)
    return items


def load_items(args) -> List[dict]:
    if args.image_ids:
        ids = [int(x.strip()) for x in args.image_ids.split(",") if x.strip()]
        labels = [x.strip() for x in args.row_labels.split(";")] if args.row_labels else []
        if labels and len(labels) != len(ids):
            raise ValueError(f"--row-labels must contain {len(ids)} labels separated by ';', got {len(labels)}")
        items = []
        for i, image_id in enumerate(ids):
            items.append({
                "image_id": image_id,
                "category": labels[i] if labels else f"Example {i + 1}",
                "rank_in_category": i + 1,
            })
        return items

    if not args.candidate_json:
        raise ValueError("Provide either --image-ids or --candidate-json")
    categories = [x.strip() for x in args.categories.split(",") if x.strip()]
    return load_items_from_candidate_json(args.candidate_json, categories, args.max_per_category)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Create Figure-6 final-stage query difference visualization.")
    parser.add_argument("--candidate-json", default=None)
    parser.add_argument("--categories", default="crowded_positive,small_positive,balanced_small_scene,failure_case")
    parser.add_argument("--max-per-category", type=int, default=6)
    parser.add_argument("--image-ids", default=None, help="Comma-separated image ids. Overrides candidate-json.")
    parser.add_argument("--row-labels", default=None, help="Optional row labels separated by ';' when using --image-ids.")
    parser.add_argument("--raw-dir", required=True, help="outputs/query_vis_auto/raw")
    parser.add_argument("--coco-ann", required=True)
    parser.add_argument("--coco-img-dir", required=True)
    parser.add_argument("--topn", type=int, default=30, help="Use top-N selected queries from each method.")
    parser.add_argument("--common-center-thr", type=float, default=0.025)
    parser.add_argument("--assoc-iou-thr", type=float, default=0.1)
    parser.add_argument("--full-query-filter", choices=["all", "associated", "small_associated", "medium_large_associated"], default="all")
    parser.add_argument("--local-query-filter", choices=["all", "associated", "small_associated", "medium_large_associated"], default="associated")
    parser.add_argument("--local-view", choices=["crop", "full"], default="crop")
    parser.add_argument("--crop-margin", type=float, default=0.18)
    parser.add_argument("--min-crop-size", type=float, default=0.38)
    parser.add_argument("--rows-per-page", type=int, default=4)
    parser.add_argument("--hide-common", action="store_true")
    parser.add_argument("--hide-counts", action="store_true")
    parser.add_argument("--dpi", type=int, default=260)
    parser.add_argument("--out-dir", default="figures/figure6_difference_gallery")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    items = load_items(args)
    if not items:
        raise RuntimeError("No Figure 6 items loaded.")

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    coco = CocoGT(args.coco_ann, args.coco_img_dir)

    index_rows = []
    page_id = 0

    for start in range(0, len(items), args.rows_per_page):
        page_items = items[start:start + args.rows_per_page]
        page_id += 1
        fig_h = 3.55 * len(page_items) + 0.45
        fig, axes = plt.subplots(len(page_items), 3, figsize=(14.2, fig_h), dpi=args.dpi)
        if len(page_items) == 1:
            axes = axes.reshape(1, 3)

        col_titles = ["Image + GT", "Full difference", "GT-associated / local difference"]
        for c, title in enumerate(col_titles):
            axes[0, c].set_title(title, fontsize=15, fontweight="bold", pad=8)

        for r, item in enumerate(page_items):
            image_id = int(item["image_id"])
            row_label = str(item.get("category", f"Example {r + 1}"))
            try:
                image, info = coco.load_image(image_id)
                gt_xyxy, gt_scales = coco.gt_boxes_norm_and_scales(image_id)
                base_all = load_query_xyxy(raw_dir / "baseline" / "best" / f"{image_id}.pt")[:args.topn]
                cmqs_all = load_query_xyxy(raw_dir / "cmqs" / "best" / f"{image_id}.pt")[:args.topn]

                base_assign = associate_query_to_gt(base_all, gt_xyxy, iou_thr=args.assoc_iou_thr)
                cmqs_assign = associate_query_to_gt(cmqs_all, gt_xyxy, iou_thr=args.assoc_iou_thr)

                base_full, base_full_ranks = filter_queries(base_all, base_assign, gt_scales, args.full_query_filter)
                cmqs_full, cmqs_full_ranks = filter_queries(cmqs_all, cmqs_assign, gt_scales, args.full_query_filter)
                base_local, base_local_ranks = filter_queries(base_all, base_assign, gt_scales, args.local_query_filter)
                cmqs_local, cmqs_local_ranks = filter_queries(cmqs_all, cmqs_assign, gt_scales, args.local_query_filter)

                crop_norm = None
                if args.local_view == "crop":
                    crop_norm = make_crop_box(
                        gt_xyxy, gt_scales, base_all, cmqs_all, base_assign, cmqs_assign,
                        category=row_label, margin=args.crop_margin, min_size=args.min_crop_size,
                    )

            except Exception as e:
                print(f"[WARN] skip image_id={image_id}: {e}")
                for c in range(3):
                    axes[r, c].axis("off")
                continue

            if args.debug:
                print(
                    f"[DEBUG] page={page_id}, row={r}, id={image_id}, label={row_label}, file={info.get('file_name')}, "
                    f"GT={len(gt_scales)}, small={int((gt_scales == 0).sum())}, "
                    f"full B/C={len(base_full)}/{len(cmqs_full)}, local B/C={len(base_local)}/{len(cmqs_local)}"
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

            small_gt = int((gt_scales == 0).sum().item())
            ylabel = f"{row_label}\nid={image_id} | GT={len(gt_scales)}, small={small_gt}"
            axes[r, 0].set_ylabel(ylabel, fontsize=12.5, fontweight="bold", rotation=90, labelpad=12)
            axes[r, 0].yaxis.set_label_coords(-0.065, 0.5)

            index_rows.append({
                "page": page_id,
                "row": r + 1,
                "image_id": image_id,
                "row_label": row_label,
                "file_name": info.get("file_name", ""),
                "num_gt": int(len(gt_scales)),
                "num_small_gt": small_gt,
                "topn": args.topn,
                "full_query_filter": args.full_query_filter,
                "local_query_filter": args.local_query_filter,
                "full_baseline_plotted": int(len(base_full)),
                "full_cmqs_plotted": int(len(cmqs_full)),
                "full_common": full_counts[0],
                "full_baseline_only": full_counts[1],
                "full_cmqs_only": full_counts[2],
                "local_baseline_plotted": int(len(base_local)),
                "local_cmqs_plotted": int(len(cmqs_local)),
                "local_common": local_counts[0],
                "local_baseline_only": local_counts[1],
                "local_cmqs_only": local_counts[2],
            })

        add_legend(fig, show_common=not args.hide_common)
        plt.tight_layout(rect=(0.02, 0.06, 1.0, 1.0), pad=0.65, w_pad=0.45, h_pad=0.82)
        out_png = out_dir / f"figure6_difference_page_{page_id:02d}.png"
        fig.savefig(out_png, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_png}")

    out_csv = out_dir / "figure6_difference_index.csv"
    if index_rows:
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(index_rows[0].keys()))
            writer.writeheader()
            writer.writerows(index_rows)
        print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()
