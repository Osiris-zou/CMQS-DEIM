#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quickly compute Params and model-level GFLOPs for two DEIM/CMQS configs.

This script uses the DEIM project's own `engine.misc.stats` utility, so the
reported FLOPs should be consistent with the FLOPs printed at training startup.
Run it from the project root, i.e. the same directory as train.py.

Example:
  CUDA_VISIBLE_DEVICES=0 python compute_params_flops_quick.py \
    --baseline-config configs/deim_dfine/deim_hgnetv2_l_coco.yml \
    --cmqs-config configs/deim_dfine/deim-l.yml \
    --out outputs/table7_profile/params_flops.csv
"""

import argparse
import csv
import gc
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.distributed as dist


def init_single_process_distributed() -> torch.device:
    """Initialize distributed because HGNetv2 may call torch.distributed.get_rank()."""
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29731")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"

    if dist.is_available() and not dist.is_initialized():
        dist.init_process_group(backend=backend, init_method="env://")
    return device


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        try:
            dist.barrier()
        except Exception:
            pass
        try:
            dist.destroy_process_group()
        except Exception:
            pass


def is_main_process() -> bool:
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0


def disable_backbone_pretrained(cfg: Any) -> None:
    """Avoid downloading/loading HGNetv2 pretrained weights during complexity profiling."""
    if hasattr(cfg, "yaml_cfg") and isinstance(cfg.yaml_cfg, dict):
        if "HGNetv2" in cfg.yaml_cfg:
            cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
        # Some configs nest backbone under model/backbone; keep this defensive.
        for key in ("backbone", "Backbone"):
            if key in cfg.yaml_cfg and isinstance(cfg.yaml_cfg[key], dict):
                if cfg.yaml_cfg[key].get("type", "") == "HGNetv2" or cfg.yaml_cfg[key].get("_name", "") == "HGNetv2":
                    cfg.yaml_cfg[key]["pretrained"] = False


def count_params(model: torch.nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def parse_stats_text(text: str) -> Dict[str, Optional[float]]:
    """Parse strings like: {'Model FLOPs:93.2879 GFLOPS   MACs:46.5209 GMACs   Params:30769891'}"""
    out: Dict[str, Optional[float]] = {
        "gflops": None,
        "gmacs": None,
        "stats_params": None,
    }

    m = re.search(r"Model\s+FLOPs\s*:\s*([0-9.]+)\s*GFLOPS", text, flags=re.I)
    if m:
        out["gflops"] = float(m.group(1))

    m = re.search(r"MACs\s*:\s*([0-9.]+)\s*GMACs", text, flags=re.I)
    if m:
        out["gmacs"] = float(m.group(1))

    m = re.search(r"Params\s*:\s*([0-9]+)", text, flags=re.I)
    if m:
        out["stats_params"] = float(m.group(1))

    # Fallback parse for startup block lines.
    if out["gflops"] is None:
        m = re.search(r"fwd\s+FLOPs\s*:\s*([0-9.]+)\s*GFLOPS", text, flags=re.I)
        if m:
            out["gflops"] = float(m.group(1))
    if out["gmacs"] is None:
        m = re.search(r"fwd\s+MACs\s*:\s*([0-9.]+)\s*GMACs", text, flags=re.I)
        if m:
            out["gmacs"] = float(m.group(1))
    if out["stats_params"] is None:
        m = re.search(r"Total\s+Training\s+Params\s*:\s*([0-9.]+)\s*M", text, flags=re.I)
        if m:
            out["stats_params"] = float(m.group(1)) * 1e6

    return out


def compute_one(name: str, config_path: str, device: torch.device) -> Dict[str, Any]:
    from engine.core import YAMLConfig
    from engine.misc import stats as deim_stats

    cfg = YAMLConfig(config_path, device=str(device))
    disable_backbone_pretrained(cfg)

    # Build the model once for direct parameter counting.
    model = cfg.model
    model.eval()
    total_params, trainable_params = count_params(model)

    stats_text = ""
    n_parameters_from_stats = None
    gflops = None
    gmacs = None
    stats_params = None
    err = ""

    try:
        # DEIM's own complexity utility. It usually prints a block and returns
        # (n_parameters, model_stats_string_or_dict).
        ret = deim_stats(cfg)
        if isinstance(ret, tuple) and len(ret) >= 2:
            n_parameters_from_stats = ret[0]
            stats_text = str(ret[1])
        else:
            stats_text = str(ret)

        parsed = parse_stats_text(stats_text)
        gflops = parsed["gflops"]
        gmacs = parsed["gmacs"]
        stats_params = parsed["stats_params"]
    except Exception as exc:
        err = repr(exc)

    row = {
        "name": name,
        "config": config_path,
        "params_total": total_params,
        "params_total_M": f"{total_params / 1e6:.4f}",
        "params_trainable": trainable_params,
        "params_trainable_M": f"{trainable_params / 1e6:.4f}",
        "stats_n_parameters": n_parameters_from_stats if n_parameters_from_stats is not None else "",
        "stats_params": int(stats_params) if stats_params is not None else "",
        "stats_params_M": f"{stats_params / 1e6:.4f}" if stats_params is not None else "",
        "gmacs": f"{gmacs:.4f}" if gmacs is not None else "",
        "gflops": f"{gflops:.4f}" if gflops is not None else "",
        "error": err,
    }

    # Release memory before the next config.
    del model, cfg
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-config", required=True, type=str)
    parser.add_argument("--cmqs-config", required=True, type=str)
    parser.add_argument("--out", default="outputs/table7_profile/params_flops.csv", type=str)
    args = parser.parse_args()

    device = init_single_process_distributed()

    rows = []
    try:
        rows.append(compute_one("DEIM-L baseline", args.baseline_config, device))
        rows.append(compute_one("CMQS", args.cmqs_config, device))

        if is_main_process():
            out_path = Path(args.out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with out_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)

            print("\n===== Params / FLOPs summary =====")
            for r in rows:
                print(
                    f"{r['name']}: "
                    f"params_total={r['params_total_M']}M, "
                    f"params_trainable={r['params_trainable_M']}M, "
                    f"GFLOPs={r['gflops']}, GMACs={r['gmacs']}, "
                    f"stats_params={r['stats_params_M']}M"
                )
                if r["error"]:
                    print(f"  [stats error] {r['error']}")
            print(f"Saved: {out_path}")
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
