#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 7 difference visualization for DEIM-L vs CMQS.

This script creates a 3 x 3 style figure:
  rows    = different validation images
  columns = Early / Middle / Late checkpoints

Each subplot overlays the selected query centers of two methods on the same image:
  gray   = queries selected by both DEIM-L and CMQS (matched by center distance)
  orange = DEIM-L-only selected query centers
  cyan   = CMQS-only selected query centers

GT boxes are overlaid for spatial reference:
  yellow = medium/large GT boxes
  red    = small GT boxes (COCO area < 32^2)

Requirements:
  1) Your decoder must support query dumping and return outputs['selected_query_boxes'].
  2) The selected_query_boxes are expected to be normalized cxcywh boxes, sorted by ranking.

Example:
CUDA_VISIBLE_DEVICES=1 python make_figure7_difference_gallery_coco_gt.py \
  --baseline-config "configs/deim_dfine/deim_hgnetv2_l_coco.yml" \
  --cmqs-config "configs/deim_dfine/deim-l.yml" \
  --baseline-early "outputs/deim_hgnetv2_l_coco/checkpoint0007.pth" \
  --baseline-middle "outputs/deim_hgnetv2_l_coco/checkpoint0027.pth" \
  --baseline-late "outputs/deim_hgnetv2_l_coco/best_stg2.pth" \
  --cmqs-early "outputs/deim_hgnetv2_l_coco_a7_stop_10/checkpoint0007.pth" \
  --cmqs-middle "outputs/deim_hgnetv2_l_coco_a7_stop_10/checkpoint0027.pth" \
  --cmqs-late "outputs/deim_hgnetv2_l_coco_a7_stop_10/best_stg2.pth" \
  --image-ids "68387,201148,209829" \
  --coco-ann "/media/hanyong/zgw/zp/deim 2/annotations/instances_val2017.json" \
  --coco-img-dir "/media/hanyong/zgw/zp/val2017" \
  --topn 40 \
  --common-center-thr 0.025 \
  --out-dir "figures/figure7_difference_final" \
  --debug
"""

import argparse
import json
import os
import sys
import gc
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from PIL import Image
import torch
import torch.distributed as dist

# -----------------------------------------------------------------------------
# Project imports
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
if (PROJECT_ROOT / "engine").exists():
    sys.path.insert(0, str(PROJECT_ROOT))
elif (PROJECT_ROOT.parent / "engine").exists():
    sys.path.insert(0, str(PROJECT_ROOT.parent))
else:
    sys.path.insert(0, str(Path.cwd()))

from engine.core import YAMLConfig
from engine.solver import TASKS


# -----------------------------------------------------------------------------
# Box utilities
# -----------------------------------------------------------------------------
def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    boxes = boxes.float()
    if boxes.numel() == 0:
        return boxes.reshape(0, 4)
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([
        cx - 0.5 * w,
        cy - 0.5 * h,
        cx + 0.5 * w,
        cy + 0.5 * h,
    ], dim=-1).clamp(0, 1)


def xywh_abs_to_xyxy_norm(bbox: Sequence[float], img_w: float, img_h: float) -> List[float]:
    x, y, w, h = bbox
    return [
        max(0.0, float(x)) / img_w,
        max(0.0, float(y)) / img_h,
        min(float(img_w), float(x) + float(w)) / img_w,
        min(float(img_h), float(y) + float(h)) / img_h,
    ]


def box_iou_xyxy(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """IoU for normalized xyxy boxes."""
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((boxes1.shape[0], boxes2.shape[0]), dtype=torch.float32)
    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
    union = area1[:, None] + area2[None, :] - inter
    return inter / (union + 1e-6)


def center_inside_boxes(centers: torch.Tensor, gt_xyxy: torch.Tensor) -> torch.Tensor:
    if centers.numel() == 0 or gt_xyxy.numel() == 0:
        return torch.zeros((centers.shape[0],), dtype=torch.bool)
    x, y = centers[:, 0], centers[:, 1]
    inside = (
        (x[:, None] >= gt_xyxy[None, :, 0]) &
        (y[:, None] >= gt_xyxy[None, :, 1]) &
        (x[:, None] <= gt_xyxy[None, :, 2]) &
        (y[:, None] <= gt_xyxy[None, :, 3])
    )
    return inside.any(dim=1)


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
                scales.append(0)      # small
            elif area < 96 ** 2:
                scales.append(1)      # medium
            else:
                scales.append(2)      # large
        if not boxes:
            return torch.zeros((0, 4), dtype=torch.float32), torch.zeros((0,), dtype=torch.long)
        return torch.tensor(boxes, dtype=torch.float32), torch.tensor(scales, dtype=torch.long)


# -----------------------------------------------------------------------------
# DEIM model loading and dumping
# -----------------------------------------------------------------------------
def init_single_process_distributed(port: str = "29597") -> None:
    """Some HGNetV2/DEIM code calls torch.distributed.get_rank(). Initialize a 1-rank group."""
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
            dist.destroy_process_group()
        except Exception:
            pass


def robust_to_device(targets, device):
    return [{k: (v.to(device) if hasattr(v, "to") else v) for k, v in t.items()} for t in targets]


def get_image_id(target: dict) -> int:
    for key in ["image_id", "img_id", "id"]:
        if key in target:
            v = target[key]
            return int(v.item()) if torch.is_tensor(v) else int(v)
    raise KeyError("Cannot find image_id/img_id/id in target.")


def set_dump_selected_queries(model) -> None:
    raw = model.module if hasattr(model, "module") else model
    found = False
    for m in raw.modules():
        if hasattr(m, "dump_selected_queries"):
            m.dump_selected_queries = True
            found = True
    if not found:
        print("[WARN] No module has dump_selected_queries. Make sure you are using the query-dump decoder.")


def build_solver(config_path: str, ckpt_path: str):
    cfg = YAMLConfig(config_path, resume=ckpt_path)
    # Avoid trying to download backbone weights when loading checkpoint.
    if ckpt_path is not None and "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    solver = TASKS[cfg.yaml_cfg["task"]](cfg)
    solver.eval()
    return solver


@torch.no_grad()
def dump_many(
    config_path: str,
    ckpt_path: str,
    image_ids: List[int],
    device: torch.device,
    debug: bool = False,
) -> Dict[int, torch.Tensor]:
    """Return {image_id: selected_query_boxes[cxcywh_norm, sorted]} for one checkpoint."""
    wanted = set(int(x) for x in image_ids)
    found: Dict[int, torch.Tensor] = {}

    solver = build_solver(config_path, ckpt_path)
    model = solver.ema.module if getattr(solver, "ema", None) is not None else solver.model
    model.to(device)
    model.eval()
    set_dump_selected_queries(model)

    for samples, targets in solver.val_dataloader:
        samples = samples.to(device) if hasattr(samples, "to") else samples
        targets = robust_to_device(targets, device)
        ids = [get_image_id(t) for t in targets]

        if not (wanted - set(found.keys())):
            break
        if not any((i in wanted and i not in found) for i in ids):
            continue

        outputs = model(samples)
        if "selected_query_boxes" not in outputs:
            raise RuntimeError(
                "Model output does not contain outputs['selected_query_boxes']. "
                "Please use the query-dump decoder and set dump_selected_queries=True."
            )
        for b, image_id in enumerate(ids):
            if image_id in wanted and image_id not in found:
                found[image_id] = outputs["selected_query_boxes"][b].detach().cpu().float()
                if debug:
                    print(f"[DEBUG] found image_id={image_id}")
        if not (wanted - set(found.keys())):
            break

    missing = sorted(wanted - set(found.keys()))
    if missing:
        print(f"[WARN] image ids not found for ckpt={ckpt_path}: {missing}")

    # Explicit cleanup to reduce GPU memory when loading six checkpoints in sequence.
    try:
        del model, solver
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    return found


# -----------------------------------------------------------------------------
# Candidate id loading
# -----------------------------------------------------------------------------
def load_image_ids(args) -> List[int]:
    if args.image_ids:
        return [int(x.strip()) for x in args.image_ids.split(",") if x.strip()]
    if not args.candidate_json:
        raise ValueError("Provide either --image-ids or --candidate-json")
    with open(args.candidate_json, "r", encoding="utf-8") as f:
        pool = json.load(f)
    if args.category not in pool:
        raise KeyError(f"Category {args.category} not found in {args.candidate_json}")
    ids = []
    for x in pool[args.category][: args.max_images]:
        if isinstance(x, dict):
            ids.append(int(x["image_id"]))
        else:
            ids.append(int(x))
    return ids


def parse_row_labels(s: Optional[str], n: int) -> Optional[List[str]]:
    if not s:
        return None
    labels = [x.strip() for x in s.split(";")]
    if len(labels) != n:
        raise ValueError(f"--row-labels must contain {n} labels separated by ';', got {len(labels)}")
    return labels


# -----------------------------------------------------------------------------
# Filtering and matching
# -----------------------------------------------------------------------------
def filter_queries(
    q_cxcywh: torch.Tensor,
    gt_xyxy: torch.Tensor,
    gt_scales: torch.Tensor,
    topn: int,
    query_filter: str,
    assoc_iou_thr: float,
) -> torch.Tensor:
    """Filter sorted selected query boxes. Returned boxes keep original ranking order."""
    q = q_cxcywh[:topn].float().clamp(0, 1)
    if query_filter == "all":
        return q
    if q.numel() == 0 or gt_xyxy.numel() == 0:
        return q[:0]

    q_xyxy = box_cxcywh_to_xyxy(q)
    centers = q[:, :2]

    if query_filter == "associated":
        target_gt = gt_xyxy
    elif query_filter == "small_associated":
        target_gt = gt_xyxy[gt_scales == 0]
    else:
        raise ValueError(f"Unsupported query_filter: {query_filter}")

    if target_gt.numel() == 0:
        return q[:0]

    inside = center_inside_boxes(centers, target_gt)
    iou = box_iou_xyxy(q_xyxy, target_gt).max(dim=1).values
    keep = inside | (iou >= assoc_iou_thr)
    return q[keep]


def match_common_by_center(
    baseline_q: torch.Tensor,
    cmqs_q: torch.Tensor,
    center_thr: float,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """Greedy one-to-one matching of query centers by normalized center distance."""
    if baseline_q.numel() == 0:
        return [], [], list(range(cmqs_q.shape[0]))
    if cmqs_q.numel() == 0:
        return [], list(range(baseline_q.shape[0])), []

    b_centers = baseline_q[:, :2]
    c_centers = cmqs_q[:, :2]
    dist_mat = torch.cdist(b_centers, c_centers, p=2)
    candidates = []
    nb, nc = dist_mat.shape
    for i in range(nb):
        for j in range(nc):
            d = float(dist_mat[i, j])
            if d <= center_thr:
                candidates.append((d, i, j))
    candidates.sort(key=lambda x: x[0])

    used_b, used_c = set(), set()
    pairs: List[Tuple[int, int]] = []
    for _, i, j in candidates:
        if i not in used_b and j not in used_c:
            used_b.add(i)
            used_c.add(j)
            pairs.append((i, j))

    b_only = [i for i in range(nb) if i not in used_b]
    c_only = [j for j in range(nc) if j not in used_c]
    return pairs, b_only, c_only


def rank_marker_size(rank: int) -> float:
    """Point size based on rank. Matplotlib scatter uses area-like sizes."""
    if rank < 10:
        return 28.0
    if rank < 30:
        return 16.0
    return 8.0


def draw_difference_panel(
    ax,
    image: Image.Image,
    gt_xyxy: torch.Tensor,
    gt_scales: torch.Tensor,
    baseline_q: torch.Tensor,
    cmqs_q: torch.Tensor,
    title: str,
    common_center_thr: float,
    show_gt: bool = True,
    show_common: bool = True,
    show_counts: bool = True,
):
    ax.imshow(image, aspect="auto")
    ax.set_title(title, fontsize=16, fontweight="bold", pad=8)
    ax.set_xticks([])
    ax.set_yticks([])
    W, H = image.size

    # Draw GT boxes first.
    if show_gt:
        for i, b in enumerate(gt_xyxy):
            x0, y0, x1, y1 = b.tolist()
            edge = "red" if int(gt_scales[i]) == 0 else "yellow"
            ax.add_patch(patches.Rectangle(
                (x0 * W, y0 * H),
                (x1 - x0) * W,
                (y1 - y0) * H,
                linewidth=1.05,
                edgecolor=edge,
                facecolor="none",
                alpha=0.78,
                zorder=2,
            ))

    pairs, b_only, c_only = match_common_by_center(baseline_q, cmqs_q, common_center_thr)

    # Common points: use midpoint of matched centers.
    if show_common and pairs:
        xs, ys, ss = [], [], []
        for i, j in pairs:
            p = 0.5 * (baseline_q[i, :2] + cmqs_q[j, :2])
            xs.append(float(p[0]) * W)
            ys.append(float(p[1]) * H)
            ss.append(max(rank_marker_size(i), rank_marker_size(j)))
        ax.scatter(xs, ys, s=ss, c="lightgray", edgecolors="black", linewidths=0.25,
                   alpha=0.72, zorder=4)

    # Baseline-only points.
    if b_only:
        xs = [float(baseline_q[i, 0]) * W for i in b_only]
        ys = [float(baseline_q[i, 1]) * H for i in b_only]
        ss = [rank_marker_size(i) for i in b_only]
        ax.scatter(xs, ys, s=ss, c="orange", edgecolors="black", linewidths=0.20,
                   alpha=0.86, zorder=5)

    # CMQS-only points.
    if c_only:
        xs = [float(cmqs_q[j, 0]) * W for j in c_only]
        ys = [float(cmqs_q[j, 1]) * H for j in c_only]
        ss = [rank_marker_size(j) for j in c_only]
        ax.scatter(xs, ys, s=ss, c="cyan", edgecolors="black", linewidths=0.20,
                   alpha=0.86, zorder=6)

    if show_counts:
        txt = f"common {len(pairs)} | base-only {len(b_only)} | CMQS-only {len(c_only)}"
        ax.text(
            0.01, 0.98, txt,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10.2,
            fontweight="bold",
            color="black",
            bbox=dict(boxstyle="round,pad=0.20", facecolor="white", edgecolor="black", alpha=0.88, linewidth=0.8),
            zorder=10,
        )


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
        Line2D([0], [0], color="red", label="Small GT", linewidth=1.4),
        Line2D([0], [0], color="yellow", label="Medium/Large GT", linewidth=1.4),
    ])
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=10.5, frameon=False,
               bbox_to_anchor=(0.5, 0.005))


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Create Figure-7 difference visualization gallery.")
    parser.add_argument("--baseline-config", required=True)
    parser.add_argument("--cmqs-config", required=True)
    parser.add_argument("--baseline-early", required=True)
    parser.add_argument("--baseline-middle", required=True)
    parser.add_argument("--baseline-late", required=True)
    parser.add_argument("--cmqs-early", required=True)
    parser.add_argument("--cmqs-middle", required=True)
    parser.add_argument("--cmqs-late", required=True)
    parser.add_argument("--candidate-json", default=None)
    parser.add_argument("--category", default="figure7_candidates")
    parser.add_argument("--image-ids", default=None, help="Comma-separated image ids; overrides candidate-json.")
    parser.add_argument("--max-images", type=int, default=9)
    parser.add_argument("--row-labels", default=None,
                        help="Optional row labels separated by ';'. Example: 'Structured scene;Small-object scene;Limitation case'")
    parser.add_argument("--coco-ann", required=True)
    parser.add_argument("--coco-img-dir", required=True)
    parser.add_argument("--topn", type=int, default=40, help="Use top-N selected queries from each method.")
    parser.add_argument("--common-center-thr", type=float, default=0.025,
                        help="Normalized center-distance threshold for common selected queries. Try 0.02-0.04.")
    parser.add_argument("--query-filter", choices=["all", "associated", "small_associated"], default="all",
                        help="Filter queries before comparing. Use 'all' for ranking trajectory; 'associated' for GT-related queries.")
    parser.add_argument("--assoc-iou-thr", type=float, default=0.1,
                        help="IoU threshold for associated/small_associated filtering.")
    parser.add_argument("--hide-common", action="store_true", help="Do not draw common points; only draw differences.")
    parser.add_argument("--hide-counts", action="store_true", help="Do not display common/base-only/CMQS-only counts.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rows-per-page", type=int, default=3)
    parser.add_argument("--dpi", type=int, default=260)
    parser.add_argument("--out-dir", default="figures/figure7_difference_gallery")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    image_ids = load_image_ids(args)[: args.max_images]
    if not image_ids:
        raise RuntimeError("No image ids selected.")
    row_labels_all = parse_row_labels(args.row_labels, len(image_ids))
    print("Selected Figure 7 difference ids:", image_ids)

    init_single_process_distributed()
    device = torch.device(args.device if (torch.cuda.is_available() or args.device == "cpu") else "cpu")
    coco = CocoGT(args.coco_ann, args.coco_img_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        specs = [
            ("baseline", "early", args.baseline_config, args.baseline_early),
            ("baseline", "middle", args.baseline_config, args.baseline_middle),
            ("baseline", "late", args.baseline_config, args.baseline_late),
            ("cmqs", "early", args.cmqs_config, args.cmqs_early),
            ("cmqs", "middle", args.cmqs_config, args.cmqs_middle),
            ("cmqs", "late", args.cmqs_config, args.cmqs_late),
        ]
        dumps: Dict[Tuple[str, str], Dict[int, torch.Tensor]] = {}
        for method, stage, cfg, ckpt in specs:
            print(f"Dumping {method}/{stage}: {ckpt}")
            dumps[(method, stage)] = dump_many(cfg, ckpt, image_ids, device, debug=args.debug)

        stages = ["early", "middle", "late"]
        stage_titles = ["Early", "Middle", "Late"]
        page_id = 0
        index_rows = []

        for start in range(0, len(image_ids), args.rows_per_page):
            ids_page = image_ids[start:start + args.rows_per_page]
            labels_page = row_labels_all[start:start + args.rows_per_page] if row_labels_all else None
            page_id += 1

            # 3 columns; one row per image. Width intentionally large for journal readability.
            fig_h = 3.35 * len(ids_page) + 0.35
            fig, axes = plt.subplots(len(ids_page), 3, figsize=(14.2, fig_h), dpi=args.dpi)
            if len(ids_page) == 1:
                axes = axes.reshape(1, 3)

            for r, image_id in enumerate(ids_page):
                image, info = coco.load_image(image_id)
                gt_xyxy, gt_scales = coco.gt_boxes_norm_and_scales(image_id)

                row_label = labels_page[r] if labels_page else f"id={image_id}"
                for c, stage in enumerate(stages):
                    ax = axes[r, c]
                    if image_id not in dumps[("baseline", stage)] or image_id not in dumps[("cmqs", stage)]:
                        ax.axis("off")
                        continue

                    bq_raw = dumps[("baseline", stage)][image_id]
                    cq_raw = dumps[("cmqs", stage)][image_id]
                    bq = filter_queries(
                        bq_raw, gt_xyxy, gt_scales, args.topn,
                        args.query_filter, args.assoc_iou_thr,
                    )
                    cq = filter_queries(
                        cq_raw, gt_xyxy, gt_scales, args.topn,
                        args.query_filter, args.assoc_iou_thr,
                    )

                    draw_difference_panel(
                        ax, image, gt_xyxy, gt_scales, bq, cq,
                        stage_titles[c],
                        common_center_thr=args.common_center_thr,
                        show_gt=True,
                        show_common=not args.hide_common,
                        show_counts=not args.hide_counts,
                    )

                    pairs, b_only, c_only = match_common_by_center(bq, cq, args.common_center_thr)
                    index_rows.append({
                        "image_id": image_id,
                        "file_name": info.get("file_name", ""),
                        "stage": stage,
                        "topn": args.topn,
                        "query_filter": args.query_filter,
                        "baseline_plotted": int(bq.shape[0]),
                        "cmqs_plotted": int(cq.shape[0]),
                        "common": len(pairs),
                        "baseline_only": len(b_only),
                        "cmqs_only": len(c_only),
                    })

                    if args.debug:
                        print(
                            f"[DEBUG] image={image_id}, stage={stage}, "
                            f"baseline={bq.shape[0]}, cmqs={cq.shape[0]}, "
                            f"common={len(pairs)}, base_only={len(b_only)}, cmqs_only={len(c_only)}"
                        )

                axes[r, 0].set_ylabel(row_label, fontsize=16, fontweight="bold", rotation=90, labelpad=10)
                axes[r, 0].yaxis.set_label_coords(-0.055, 0.5)

            add_legend(fig, show_common=not args.hide_common)
            plt.tight_layout(rect=(0.015, 0.06, 1.0, 1.0), pad=0.65, w_pad=0.45, h_pad=0.8)
            out_png = out_dir / f"figure7_difference_page_{page_id:02d}.png"
            fig.savefig(out_png, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved: {out_png}")

        # Save simple CSV index without pandas dependency.
        out_csv = out_dir / "figure7_difference_index.csv"
        if index_rows:
            keys = list(index_rows[0].keys())
            with open(out_csv, "w", encoding="utf-8") as f:
                f.write(",".join(keys) + "\n")
                for row in index_rows:
                    f.write(",".join(str(row[k]) for k in keys) + "\n")
            print(f"Saved: {out_csv}")

    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
