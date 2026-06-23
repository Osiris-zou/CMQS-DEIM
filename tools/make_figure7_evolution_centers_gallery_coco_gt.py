#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create Figure-7 candidate galleries with query centers instead of dense boxes.

Why this script?
----------------
The original Figure-7 visualization draws many full query boxes, which can hide
subtle differences between DEIM-L and CMQS. This script draws query centers
(and optionally only top few boxes) across early / middle / late checkpoints.
It is intended for human screening of clearer evolution examples.

Example
-------
CUDA_VISIBLE_DEVICES=1 python make_figure7_evolution_centers_gallery_coco_gt.py \
  --baseline-config configs/deim_dfine/deim_hgnetv2_l_coco.yml \
  --cmqs-config configs/deim_dfine/deim-l.yml \
  --baseline-early outputs/deim_hgnetv2_l_coco/checkpoint0007.pth \
  --baseline-middle outputs/deim_hgnetv2_l_coco/checkpoint0027.pth \
  --baseline-late outputs/deim_hgnetv2_l_coco/best_stg2.pth \
  --cmqs-early outputs/deim_hgnetv2_l_coco_a7_stop_10/checkpoint0007.pth \
  --cmqs-middle outputs/deim_hgnetv2_l_coco_a7_stop_10/checkpoint0027.pth \
  --cmqs-late outputs/deim_hgnetv2_l_coco_a7_stop_10/best_stg2.pth \
  --candidate-json outputs/query_vis_auto/figure_candidate_pool.json \
  --category figure7_candidates \
  --max-images 6 \
  --coco-ann /path/to/annotations/instances_val2017.json \
  --coco-img-dir /path/to/val2017 \
  --topn 80 \
  --show-boxes-topk 8 \
  --out-dir figures/query_vis_gallery_fig7_centers
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import torch
import torch.distributed as dist

PROJECT_ROOT = Path(__file__).resolve().parent
if (PROJECT_ROOT / "engine").exists():
    sys.path.insert(0, str(PROJECT_ROOT))
elif (PROJECT_ROOT.parent / "engine").exists():
    sys.path.insert(0, str(PROJECT_ROOT.parent))
else:
    sys.path.insert(0, str(Path.cwd()))

from engine.core import YAMLConfig
from engine.solver import TASKS


def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    boxes = boxes.float()
    if boxes.numel() == 0:
        return boxes.reshape(0, 4)
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], dim=-1).clamp(0, 1)


def xywh_abs_to_xyxy_norm(bbox: Sequence[float], img_w: float, img_h: float) -> List[float]:
    x, y, w, h = bbox
    return [
        max(0.0, x) / img_w,
        max(0.0, y) / img_h,
        min(float(img_w), x + w) / img_w,
        min(float(img_h), y + h) / img_h,
    ]


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
                scales.append(0)
            elif area < 96 ** 2:
                scales.append(1)
            else:
                scales.append(2)
        if not boxes:
            return torch.zeros((0, 4), dtype=torch.float32), torch.zeros((0,), dtype=torch.long)
        return torch.tensor(boxes, dtype=torch.float32), torch.tensor(scales, dtype=torch.long)


def init_single_process_distributed(port: str = "29591") -> None:
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
    if ckpt_path is not None and "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    solver = TASKS[cfg.yaml_cfg["task"]](cfg)
    solver.eval()
    return solver


@torch.no_grad()
def dump_many(config_path: str, ckpt_path: str, image_ids: List[int], device: torch.device, debug: bool = False) -> Dict[int, torch.Tensor]:
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
            raise RuntimeError("Model output does not contain selected_query_boxes. Please use the query-dump decoder.")
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
    return found


def load_image_ids(args) -> List[int]:
    if args.image_ids:
        return [int(x.strip()) for x in args.image_ids.split(",") if x.strip()]
    if not args.candidate_json:
        raise ValueError("Provide either --image-ids or --candidate-json")
    with open(args.candidate_json, "r", encoding="utf-8") as f:
        pool = json.load(f)
    if args.category not in pool:
        raise KeyError(f"Category {args.category} not found in {args.candidate_json}")
    return [int(x["image_id"]) for x in pool[args.category][: args.max_images]]


def draw_evolution_panel(
    ax,
    image: Image.Image,
    gt_xyxy: torch.Tensor,
    gt_scales: torch.Tensor,
    query_cxcywh: torch.Tensor,
    title: str,
    topn: int = 80,
    show_boxes_topk: int = 8,
    show_gt: bool = True,
):
    ax.imshow(image, aspect="auto")
    ax.set_title(title, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    W, H = image.size

    if show_gt:
        for i, b in enumerate(gt_xyxy):
            x0, y0, x1, y1 = b.tolist()
            edge = "red" if int(gt_scales[i]) == 0 else "yellow"
            ax.add_patch(patches.Rectangle(
                (x0 * W, y0 * H), (x1 - x0) * W, (y1 - y0) * H,
                linewidth=1.1, edgecolor=edge, facecolor="none", alpha=0.75))

    q = query_cxcywh[:topn].float().clamp(0, 1)
    if q.numel() == 0:
        return
    centers = q[:, :2]
    for j, c in enumerate(centers):
        # Larger dots for higher-ranked queries.
        size = 4.8 if j < 10 else (3.2 if j < 30 else 2.0)
        alpha = 0.95 if j < 10 else (0.70 if j < 30 else 0.45)
        ax.plot(float(c[0]) * W, float(c[1]) * H, marker="o", markersize=size, color="cyan", alpha=alpha)

    if show_boxes_topk > 0:
        q_xyxy = box_cxcywh_to_xyxy(q[:show_boxes_topk])
        for j, b in enumerate(q_xyxy):
            x0, y0, x1, y1 = b.tolist()
            ax.add_patch(patches.Rectangle(
                (x0 * W, y0 * H), (x1 - x0) * W, (y1 - y0) * H,
                linewidth=1.0, edgecolor="cyan", facecolor="none", alpha=0.45))


def main():
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--image-ids", default=None, help="comma-separated image ids; overrides candidate-json")
    parser.add_argument("--max-images", type=int, default=6)
    parser.add_argument("--coco-ann", required=True)
    parser.add_argument("--coco-img-dir", required=True)
    parser.add_argument("--topn", type=int, default=80)
    parser.add_argument("--show-boxes-topk", type=int, default=8, help="Draw boxes only for the top-k queries. Use 0 for centers only.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rows-per-page", type=int, default=1)
    parser.add_argument("--out-dir", default="figures/query_vis_gallery_fig7_centers")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    image_ids = load_image_ids(args)[: args.max_images]
    if not image_ids:
        raise RuntimeError("No image ids selected.")
    print("Selected Figure 7 candidate ids:", image_ids)

    init_single_process_distributed()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
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
        for start in range(0, len(image_ids), args.rows_per_page):
            ids_page = image_ids[start:start + args.rows_per_page]
            page_id += 1
            fig, axes = plt.subplots(2 * len(ids_page), 3, figsize=(13.2, 4.2 * len(ids_page)), dpi=220)
            if len(ids_page) == 1:
                axes = axes.reshape(2, 3)
            for idx, image_id in enumerate(ids_page):
                image, info = coco.load_image(image_id)
                gt_xyxy, gt_scales = coco.gt_boxes_norm_and_scales(image_id)
                row0 = 2 * idx
                for c, stage in enumerate(stages):
                    if image_id in dumps[("baseline", stage)]:
                        draw_evolution_panel(
                            axes[row0, c], image, gt_xyxy, gt_scales,
                            dumps[("baseline", stage)][image_id],
                            stage_titles[c], topn=args.topn, show_boxes_topk=args.show_boxes_topk)
                    else:
                        axes[row0, c].axis("off")
                    if image_id in dumps[("cmqs", stage)]:
                        draw_evolution_panel(
                            axes[row0 + 1, c], image, gt_xyxy, gt_scales,
                            dumps[("cmqs", stage)][image_id],
                            stage_titles[c], topn=args.topn, show_boxes_topk=args.show_boxes_topk)
                    else:
                        axes[row0 + 1, c].axis("off")
                axes[row0, 0].set_ylabel(f"id={image_id}\nDEIM-L", fontsize=10)
                axes[row0 + 1, 0].set_ylabel(f"id={image_id}\nCMQS", fontsize=10)
                if args.debug:
                    print(f"[DEBUG] plotted image_id={image_id}, file={info.get('file_name')}")

            plt.tight_layout(pad=0.8)
            out = out_dir / f"figure7_centers_gallery_page_{page_id:02d}.png"
            fig.savefig(out, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved: {out}")
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
