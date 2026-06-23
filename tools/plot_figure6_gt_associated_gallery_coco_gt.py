#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate Figure-6 candidate galleries with GT-associated selected queries.

Purpose
-------
The old full-image top-K visualization can be too cluttered. This script only
shows selected queries that are associated with COCO ground-truth objects by:

    query center inside GT box OR IoU(query, GT) > threshold

This makes the qualitative visualization consistent with Table 8 and prevents
background-only query differences from being treated as positive evidence.

Expected raw query dumps
------------------------
The script reads raw dumps produced by auto_select_query_vis_cases.py:

    <raw_dir>/baseline/best/<image_id>.pt
    <raw_dir>/cmqs/best/<image_id>.pt

Each .pt file must contain:

    selected_query_boxes: Tensor[K, 4] in normalized cxcywh format

Example
-------
python plot_figure6_gt_associated_gallery_coco_gt.py \
  --candidate-json outputs/query_vis_auto/figure_candidate_pool.json \
  --raw-dir outputs/query_vis_auto/raw \
  --coco-ann /path/to/annotations/instances_val2017.json \
  --coco-img-dir /path/to/val2017 \
  --categories crowded_positive,small_positive,balanced_small_scene \
  --view both \
  --query-filter associated \
  --max-per-category 8 \
  --topn 40 \
  --out-dir figures/query_vis_gallery_fig6_gt_assoc
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import torch


# -----------------------------
# Basic box utilities
# -----------------------------

def cxcywh_to_xyxy_norm(boxes: torch.Tensor) -> torch.Tensor:
    boxes = boxes.float()
    if boxes.numel() == 0:
        return boxes.reshape(0, 4)
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1).clamp(0, 1)


def xywh_abs_to_xyxy_norm(bbox: Sequence[float], img_w: float, img_h: float) -> List[float]:
    x, y, w, h = bbox
    return [
        max(0.0, x) / img_w,
        max(0.0, y) / img_h,
        min(float(img_w), x + w) / img_w,
        min(float(img_h), y + h) / img_h,
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


def intersects_xyxy(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Return intersection mask between boxes a[N,4] and one box b[4]."""
    if a.numel() == 0:
        return torch.zeros((0,), dtype=torch.bool)
    return (a[:, 0] < b[2]) & (a[:, 2] > b[0]) & (a[:, 1] < b[3]) & (a[:, 3] > b[1])


# -----------------------------
# COCO GT loader
# -----------------------------

class CocoGT:
    def __init__(self, ann_path: str, img_dir: str):
        self.img_dir = Path(img_dir)
        with open(ann_path, "r", encoding="utf-8") as f:
            coco = json.load(f)
        self.images: Dict[int, dict] = {int(im["id"]): im for im in coco["images"]}
        self.anns: Dict[int, list] = defaultdict(list)
        for ann in coco["annotations"]:
            if ann.get("ignore", 0):
                continue
            if ann.get("iscrowd", 0):
                # Ignore crowd boxes for this visualization because they are not
                # standard instance boxes.
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
                scales.append(0)   # small
            elif area < 96 ** 2:
                scales.append(1)   # medium
            else:
                scales.append(2)   # large
        if not boxes:
            return torch.zeros((0, 4), dtype=torch.float32), torch.zeros((0,), dtype=torch.long)
        return torch.tensor(boxes, dtype=torch.float32), torch.tensor(scales, dtype=torch.long)


# -----------------------------
# Query loading and association
# -----------------------------

def load_query_xyxy(pt_path: Path) -> torch.Tensor:
    if not pt_path.exists():
        raise FileNotFoundError(f"Raw query dump not found: {pt_path}")
    data = torch.load(pt_path, map_location="cpu")
    if "selected_query_boxes" not in data:
        raise KeyError(f"selected_query_boxes not found in {pt_path}")
    return cxcywh_to_xyxy_norm(data["selected_query_boxes"].float())


def associate_query_to_gt(q_xyxy: torch.Tensor, gt_xyxy: torch.Tensor, iou_thr: float = 0.1) -> torch.Tensor:
    """Assign each query to one GT index; -1 means no association."""
    assigned = torch.full((q_xyxy.shape[0],), -1, dtype=torch.long)
    if q_xyxy.numel() == 0 or gt_xyxy.numel() == 0:
        return assigned

    centers = (q_xyxy[:, :2] + q_xyxy[:, 2:]) / 2
    inside = (
        (centers[:, None, 0] >= gt_xyxy[None, :, 0])
        & (centers[:, None, 0] <= gt_xyxy[None, :, 2])
        & (centers[:, None, 1] >= gt_xyxy[None, :, 1])
        & (centers[:, None, 1] <= gt_xyxy[None, :, 3])
    )
    iou = box_iou_xyxy_norm(q_xyxy, gt_xyxy)
    assoc = inside | (iou > iou_thr)
    if not assoc.any():
        return assigned

    # Prefer higher IoU; if center-inside but IoU is zero, use a tiny positive value.
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
    """Return filtered query boxes and their original rank indices."""
    ranks = torch.arange(q_xyxy.shape[0], dtype=torch.long)
    if query_filter == "all":
        return q_xyxy, ranks
    if query_filter == "associated":
        mask = assigned >= 0
    elif query_filter == "small_associated":
        mask = (assigned >= 0) & (gt_scales[assigned.clamp(min=0)] == 0)
    elif query_filter == "medium_large_associated":
        mask = (assigned >= 0) & (gt_scales[assigned.clamp(min=0)] > 0)
    else:
        raise ValueError(f"Unknown query_filter: {query_filter}")
    return q_xyxy[mask], ranks[mask]


# -----------------------------
# Crop selection
# -----------------------------

def choose_focus_gt(
    gt_xyxy: torch.Tensor,
    gt_scales: torch.Tensor,
    base_assign: torch.Tensor,
    cmqs_assign: torch.Tensor,
    category: str,
) -> Optional[int]:
    """Choose one GT object to center the crop around."""
    if gt_xyxy.numel() == 0:
        return None
    n = gt_xyxy.shape[0]
    best_i, best_score = 0, -1e9
    is_failure = "failure" in category.lower()
    for i in range(n):
        cb = int((base_assign == i).sum().item())
        cc = int((cmqs_assign == i).sum().item())
        diff = abs(cc - cb)
        total = cb + cc
        scale = int(gt_scales[i].item())
        if is_failure:
            # For failure cases, prefer medium/large objects with a clear drop or redistribution.
            score = 3.0 * max(cb - cc, 0) + 1.5 * diff + 0.3 * total + (2.0 if scale > 0 else 0.0)
        else:
            # For positive cases, prefer small objects with local query changes.
            score = 2.0 * diff + 0.35 * total + (3.0 if scale == 0 else 0.0)
        if score > best_score:
            best_i, best_score = i, score
    return best_i


def make_crop_box(
    gt_xyxy: torch.Tensor,
    gt_scales: torch.Tensor,
    base_q: torch.Tensor,
    cmqs_q: torch.Tensor,
    base_assign: torch.Tensor,
    cmqs_assign: torch.Tensor,
    category: str,
    margin: float = 0.12,
    min_size: float = 0.35,
) -> torch.Tensor:
    """Create a normalized crop box around one informative GT and its associated queries."""
    focus = choose_focus_gt(gt_xyxy, gt_scales, base_assign, cmqs_assign, category)
    if focus is None:
        return torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=torch.float32)

    boxes = [gt_xyxy[focus].reshape(1, 4)]
    if base_q.numel() > 0:
        boxes.append(base_q[base_assign == focus])
    if cmqs_q.numel() > 0:
        boxes.append(cmqs_q[cmqs_assign == focus])

    boxes = [b for b in boxes if b.numel() > 0]
    union = torch.cat(boxes, dim=0)
    x0, y0 = union[:, 0].min().item(), union[:, 1].min().item()
    x1, y1 = union[:, 2].max().item(), union[:, 3].max().item()

    # Add margin.
    w, h = x1 - x0, y1 - y0
    x0 -= margin * max(w, 0.05)
    x1 += margin * max(w, 0.05)
    y0 -= margin * max(h, 0.05)
    y1 += margin * max(h, 0.05)

    # Enforce minimum crop size for readability.
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    w, h = max(x1 - x0, min_size), max(y1 - y0, min_size)
    x0, x1 = cx - w / 2, cx + w / 2
    y0, y1 = cy - h / 2, cy + h / 2

    # Clamp while keeping size as much as possible.
    if x0 < 0:
        x1 -= x0
        x0 = 0
    if y0 < 0:
        y1 -= y0
        y0 = 0
    if x1 > 1:
        x0 -= (x1 - 1)
        x1 = 1
    if y1 > 1:
        y0 -= (y1 - 1)
        y1 = 1
    return torch.tensor([max(0, x0), max(0, y0), min(1, x1), min(1, y1)], dtype=torch.float32)


# -----------------------------
# Drawing
# -----------------------------

def _crop_image(image: Image.Image, crop_norm: Optional[torch.Tensor]) -> Tuple[Image.Image, Optional[torch.Tensor]]:
    if crop_norm is None:
        return image, None
    W, H = image.size
    x0, y0, x1, y1 = crop_norm.tolist()
    px0 = int(round(x0 * W))
    py0 = int(round(y0 * H))
    px1 = int(round(x1 * W))
    py1 = int(round(y1 * H))
    px0, py0 = max(0, px0), max(0, py0)
    px1, py1 = min(W, px1), min(H, py1)
    if px1 <= px0 or py1 <= py0:
        return image, None
    return image.crop((px0, py0, px1, py1)), crop_norm


def _transform_box_for_crop(box: torch.Tensor, crop_norm: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if crop_norm is None:
        return box.clone()
    c = crop_norm
    # skip if no intersection
    if not bool(intersects_xyxy(box.reshape(1, 4), c)[0].item()):
        return None
    x0 = (max(float(box[0]), float(c[0])) - float(c[0])) / max(float(c[2] - c[0]), 1e-9)
    y0 = (max(float(box[1]), float(c[1])) - float(c[1])) / max(float(c[3] - c[1]), 1e-9)
    x1 = (min(float(box[2]), float(c[2])) - float(c[0])) / max(float(c[2] - c[0]), 1e-9)
    y1 = (min(float(box[3]), float(c[3])) - float(c[1])) / max(float(c[3] - c[1]), 1e-9)
    return torch.tensor([x0, y0, x1, y1], dtype=torch.float32).clamp(0, 1)


def draw_panel(
    ax,
    image: Image.Image,
    gt_xyxy: torch.Tensor,
    gt_scales: torch.Tensor,
    query_xyxy: Optional[torch.Tensor],
    query_ranks: Optional[torch.Tensor],
    title: str,
    crop_norm: Optional[torch.Tensor] = None,
    show_query: bool = True,
    show_gt: bool = True,
):
    img_show, crop_used = _crop_image(image, crop_norm)
    ax.imshow(img_show, aspect="auto")
    ax.set_title(title, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    W, H = img_show.size

    if show_gt:
        for i, b in enumerate(gt_xyxy):
            bb = _transform_box_for_crop(b, crop_used)
            if bb is None:
                continue
            x0, y0, x1, y1 = bb.tolist()
            edge = "red" if int(gt_scales[i]) == 0 else "yellow"
            ax.add_patch(patches.Rectangle(
                (x0 * W, y0 * H), (x1 - x0) * W, (y1 - y0) * H,
                linewidth=1.4, edgecolor=edge, facecolor="none", alpha=0.95))

    if show_query and query_xyxy is not None and query_xyxy.numel() > 0:
        for k, b in enumerate(query_xyxy):
            bb = _transform_box_for_crop(b, crop_used)
            if bb is None:
                continue
            rank = int(query_ranks[k].item()) if query_ranks is not None else k
            x0, y0, x1, y1 = bb.tolist()
            lw = 1.8 if rank < 10 else 1.0
            alpha = 0.90 if rank < 10 else 0.55
            ax.add_patch(patches.Rectangle(
                (x0 * W, y0 * H), (x1 - x0) * W, (y1 - y0) * H,
                linewidth=lw, edgecolor="cyan", facecolor="none", alpha=alpha))
            cx, cy = (x0 + x1) * 0.5 * W, (y0 + y1) * 0.5 * H
            ax.plot(cx, cy, marker="o", markersize=2.4 if rank < 10 else 1.8, color="white", alpha=0.9)


def load_candidates(path: str, categories: List[str], max_per_category: int) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        pool = json.load(f)
    items = []
    for cat in categories:
        if cat not in pool:
            print(f"[WARN] category not found: {cat}")
            continue
        for rank, r in enumerate(pool[cat][:max_per_category], start=1):
            rr = dict(r)
            rr["category"] = cat
            rr["rank_in_category"] = rank
            items.append(rr)
    return items


def fmt_metric(x) -> str:
    try:
        return f"{float(x):.2f}"
    except Exception:
        return "nan"


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-json", required=True)
    parser.add_argument("--raw-dir", required=True, help="outputs/query_vis_auto/raw")
    parser.add_argument("--coco-ann", required=True)
    parser.add_argument("--coco-img-dir", required=True)
    parser.add_argument("--categories", default="crowded_positive,small_positive,balanced_small_scene")
    parser.add_argument("--max-per-category", type=int, default=8)
    parser.add_argument("--rows-per-page", type=int, default=4)
    parser.add_argument("--topn", type=int, default=40, help="Only consider the first top-N selected queries before filtering.")
    parser.add_argument("--iou-thr", type=float, default=0.1)
    parser.add_argument("--query-filter", default="associated", choices=["all", "associated", "small_associated", "medium_large_associated"])
    parser.add_argument("--view", default="both", choices=["full", "crop", "both"])
    parser.add_argument("--crop-margin", type=float, default=0.15)
    parser.add_argument("--min-crop-size", type=float, default=0.35)
    parser.add_argument("--out-dir", default="figures/query_vis_gallery_fig6_gt_assoc")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    categories = [x.strip() for x in args.categories.split(",") if x.strip()]
    items = load_candidates(args.candidate_json, categories, args.max_per_category)
    if not items:
        raise RuntimeError("No candidate items loaded.")

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    coco = CocoGT(args.coco_ann, args.coco_img_dir)

    views = ["full", "crop"] if args.view == "both" else [args.view]
    index_rows = []

    for view in views:
        page_id = 0
        for start in range(0, len(items), args.rows_per_page):
            page_items = items[start:start + args.rows_per_page]
            page_id += 1
            fig, axes = plt.subplots(len(page_items), 3, figsize=(13.5, 3.7 * len(page_items)), dpi=220)
            if len(page_items) == 1:
                axes = axes.reshape(1, 3)
            col_titles = ["Image + GT", "DEIM-L associated queries", "CMQS associated queries"]
            for c, title in enumerate(col_titles):
                axes[0, c].set_title(title, fontsize=12)

            for r, item in enumerate(page_items):
                image_id = int(item["image_id"])
                category = str(item.get("category", "candidate"))
                try:
                    image, info = coco.load_image(image_id)
                    gt_xyxy, gt_scales = coco.gt_boxes_norm_and_scales(image_id)
                    base_all = load_query_xyxy(raw_dir / "baseline" / "best" / f"{image_id}.pt")[: args.topn]
                    cmqs_all = load_query_xyxy(raw_dir / "cmqs" / "best" / f"{image_id}.pt")[: args.topn]
                    base_assign = associate_query_to_gt(base_all, gt_xyxy, iou_thr=args.iou_thr)
                    cmqs_assign = associate_query_to_gt(cmqs_all, gt_xyxy, iou_thr=args.iou_thr)
                    base_q, base_ranks = filter_queries(base_all, base_assign, gt_scales, args.query_filter)
                    cmqs_q, cmqs_ranks = filter_queries(cmqs_all, cmqs_assign, gt_scales, args.query_filter)
                    crop_norm = None
                    if view == "crop":
                        crop_norm = make_crop_box(
                            gt_xyxy, gt_scales, base_all, cmqs_all, base_assign, cmqs_assign,
                            category=category, margin=args.crop_margin, min_size=args.min_crop_size)
                except Exception as e:
                    print(f"[WARN] skip image_id={image_id}: {e}")
                    for c in range(3):
                        axes[r, c].axis("off")
                    continue

                if args.debug:
                    print(
                        f"[DEBUG] view={view}, page={page_id}, id={image_id}, category={category}, "
                        f"file={info['file_name']}, base_assoc={len(base_q)}, cmqs_assoc={len(cmqs_q)}"
                    )

                draw_panel(axes[r, 0], image, gt_xyxy, gt_scales, None, None, "Image + GT", crop_norm=crop_norm, show_query=False)
                draw_panel(axes[r, 1], image, gt_xyxy, gt_scales, base_q, base_ranks, "DEIM-L", crop_norm=crop_norm)
                draw_panel(axes[r, 2], image, gt_xyxy, gt_scales, cmqs_q, cmqs_ranks, "CMQS", crop_norm=crop_norm)

                small_gt = int((gt_scales == 0).sum().item())
                label = (
                    f"{category} #{int(item.get('rank_in_category', 0))} | id={image_id}\n"
                    f"GT={len(gt_scales)}, small={small_gt}, shown B/C={len(base_q)}/{len(cmqs_q)}\n"
                    f"Δrank={fmt_metric(item.get('small_rank_gain'))}, "
                    f"Δcost={fmt_metric(item.get('small_cost_gain'))}, "
                    f"ΔlargeQ={fmt_metric(item.get('large_qratio_drop'))}"
                )
                axes[r, 0].set_ylabel(label, fontsize=8)

                index_rows.append({
                    "view": view,
                    "page": page_id,
                    "category": category,
                    "rank_in_category": int(item.get("rank_in_category", 0)),
                    "image_id": image_id,
                    "num_gt": len(gt_scales),
                    "num_small_gt": small_gt,
                    "shown_baseline_queries": len(base_q),
                    "shown_cmqs_queries": len(cmqs_q),
                    "small_rank_gain": item.get("small_rank_gain"),
                    "small_cost_gain": item.get("small_cost_gain"),
                    "large_qratio_drop": item.get("large_qratio_drop"),
                    "file_name": info.get("file_name"),
                })

            plt.tight_layout(pad=0.8)
            out = out_dir / f"figure6_gt_assoc_{view}_page_{page_id:02d}.png"
            fig.savefig(out, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved: {out}")

    csv_path = out_dir / "figure6_gt_assoc_gallery_index.csv"
    if index_rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(index_rows[0].keys()))
            writer.writeheader()
            writer.writerows(index_rows)
        print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
