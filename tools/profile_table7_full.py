#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Profile Table 7 for DEIM-L baseline and CMQS.

This script measures:
  - Params
  - GFLOPs (reported from command-line values, because most DEIM repos already
    print FLOPs with their own stats utility; defaults follow the manuscript)
  - Inference latency / FPS / peak inference memory on a single GPU
  - Training time per iteration / peak training memory under the launched setting
  - Component-wise query-selection overhead:
      matching-cost time
      normalization + ranking time

Recommended usage:
  1) Single-GPU inference profiling
  2) 4-GPU training profiling with torchrun
Both commands can write/append to the same CSV.
"""

import argparse
import csv
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.distributed as dist


def init_distributed_if_needed() -> Tuple[int, int, int, torch.device]:
    """Initialize a default process group even for single-process profiling.

    Some DEIM/HGNetv2 code paths call torch.distributed.get_rank() during model
    construction. Therefore, a default process group is required even when using
    ordinary `python profile_table7_full.py`.
    """
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29671")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
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

    return rank, world_size, local_rank, device


def cleanup_distributed():
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


def reduce_scalar(value: float, device: torch.device, mode: str = "mean") -> float:
    if not (dist.is_available() and dist.is_initialized()):
        return float(value)
    t = torch.tensor(float(value), device=device)
    if mode == "max":
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        return float(t.item())
    if mode == "sum":
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        return float(t.item())
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    t /= dist.get_world_size()
    return float(t.item())


def safe_to_device(x: Any, device: torch.device) -> Any:
    if hasattr(x, "to"):
        return x.to(device)
    return x


def move_targets_to_device(targets: List[Dict[str, Any]], device: torch.device) -> List[Dict[str, Any]]:
    return [{k: safe_to_device(v, device) for k, v in t.items()} for t in targets]


def batch_size_of(samples: Any) -> int:
    if hasattr(samples, "tensors"):
        return int(samples.tensors.shape[0])
    if torch.is_tensor(samples):
        return int(samples.shape[0])
    if hasattr(samples, "shape"):
        return int(samples.shape[0])
    raise RuntimeError(f"Cannot infer batch size from samples of type {type(samples)}")


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    while hasattr(model, "module"):
        model = model.module
    return model


class QuerySelectionTimer:
    """CUDA-event based component timer for query selection methods."""

    def __init__(self, device: torch.device):
        self.device = device
        self.enabled = device.type == "cuda"
        self.events: List[Tuple[str, torch.cuda.Event, torch.cuda.Event]] = []
        self.cpu_records: List[Tuple[str, float]] = []

    def reset(self):
        self.events.clear()
        self.cpu_records.clear()

    def time_call(self, name: str, fn, *args, **kwargs):
        if self.enabled:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            out = fn(*args, **kwargs)
            end.record()
            self.events.append((name, start, end))
            return out
        st = time.perf_counter()
        out = fn(*args, **kwargs)
        ed = time.perf_counter()
        self.cpu_records.append((name, (ed - st) * 1000.0))
        return out

    def collect_ms(self) -> Dict[str, float]:
        values: Dict[str, float] = {}
        if self.enabled:
            torch.cuda.synchronize(self.device)
            for name, start, end in self.events:
                values[name] = values.get(name, 0.0) + float(start.elapsed_time(end))
        else:
            for name, ms in self.cpu_records:
                values[name] = values.get(name, 0.0) + float(ms)
        return values


def instrument_query_selection(model: torch.nn.Module, timer: QuerySelectionTimer):
    """Wrap decoder query-selection methods to record component times.

    Expected DEIM/D-FINE methods:
      - _select_topk: whole top-k query selection path
      - _compute_min_cost_score: GT-dependent matching-cost construction

    We report:
      matching-cost time      = time(_compute_min_cost_score)
      norm. + ranking time    = time(_select_topk) - time(_compute_min_cost_score)
    """
    raw = unwrap_model(model)
    for module in raw.modules():
        if hasattr(module, "_select_topk") and not getattr(module, "_table7_select_wrapped", False):
            orig_select = module._select_topk

            def wrapped_select(*args, __orig=orig_select, **kwargs):
                return timer.time_call("select_topk", __orig, *args, **kwargs)

            module._select_topk = wrapped_select
            module._table7_select_wrapped = True

        if hasattr(module, "_compute_min_cost_score") and not getattr(module, "_table7_cost_wrapped", False):
            orig_cost = module._compute_min_cost_score

            def wrapped_cost(*args, __orig=orig_cost, **kwargs):
                return timer.time_call("matching_cost", __orig, *args, **kwargs)

            module._compute_min_cost_score = wrapped_cost
            module._table7_cost_wrapped = True


def set_cmqs_gt_cost_branch(model: torch.nn.Module, enabled: bool):
    """Force GT-cost branch on/off for early-training or after-exit profiling."""
    raw = unwrap_model(model)
    for m in raw.modules():
        if hasattr(m, "query_select_use_gt"):
            m.query_select_use_gt = bool(enabled)
        # Some implementations store current epoch inside decoder.
        if hasattr(m, "_current_epoch"):
            m._current_epoch = 0 if enabled else 10**9


def count_params_m(model: torch.nn.Module) -> float:
    raw = unwrap_model(model)
    return sum(p.numel() for p in raw.parameters()) / 1e6


def get_orig_target_sizes(targets: List[Dict[str, Any]], device: torch.device) -> Optional[torch.Tensor]:
    if len(targets) == 0:
        return None
    key = "orig_size" if "orig_size" in targets[0] else ("size" if "size" in targets[0] else None)
    if key is None:
        return None
    return torch.stack([t[key].to(device) for t in targets], dim=0)


def next_batch(data_iter: Iterable, dataloader: Iterable):
    try:
        return next(data_iter), data_iter
    except StopIteration:
        data_iter = iter(dataloader)
        return next(data_iter), data_iter


def build_solver(config_path: str, checkpoint_path: str, for_train: bool, device: torch.device):
    # Imports are delayed so that this script can be imported without the project path.
    from engine.core import YAMLConfig
    from engine.solver import TASKS

    cfg = YAMLConfig(config_path, resume=checkpoint_path, device=str(device))

    # When loading a full checkpoint, do not download/load backbone pretrained weights.
    if checkpoint_path is not None and checkpoint_path != "":
        if "HGNetv2" in cfg.yaml_cfg:
            cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    solver = TASKS[cfg.yaml_cfg["task"]](cfg)
    if for_train:
        solver.train()
    else:
        solver.eval()
    return solver


def measure_inference_one(
    name: str,
    config_path: str,
    checkpoint_path: str,
    gflops: float,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, Any]:
    solver = build_solver(config_path, checkpoint_path, for_train=False, device=device)
    model = solver.ema.module if getattr(solver, "ema", None) is not None else solver.model
    model.eval()
    set_cmqs_gt_cost_branch(model, enabled=False)

    postprocessor = solver.postprocessor
    dataloader = solver.val_dataloader
    data_iter = iter(dataloader)

    timer = QuerySelectionTimer(device)
    instrument_query_selection(model, timer)

    use_amp = getattr(solver, "scaler", None) is not None or getattr(args, "use_amp", False)

    # Warm-up
    with torch.no_grad():
        for _ in range(args.infer_warmup):
            (samples, targets), data_iter = next_batch(data_iter, dataloader)
            samples = samples.to(device)
            targets = move_targets_to_device(targets, device)
            with torch.autocast(device_type=device.type, enabled=(device.type == "cuda" and use_amp)):
                outputs = model(samples)
                orig_sizes = get_orig_target_sizes(targets, device)
                if orig_sizes is not None and postprocessor is not None:
                    _ = postprocessor(outputs, orig_sizes)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)

        total_ms = 0.0
        total_images = 0
        sum_matching_ms = 0.0
        sum_norm_rank_ms = 0.0

        for _ in range(args.infer_iters):
            (samples, targets), data_iter = next_batch(data_iter, dataloader)
            samples = samples.to(device)
            targets = move_targets_to_device(targets, device)
            bs = batch_size_of(samples)
            timer.reset()

            if device.type == "cuda":
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                with torch.autocast(device_type="cuda", enabled=use_amp):
                    outputs = model(samples)
                    orig_sizes = get_orig_target_sizes(targets, device)
                    if orig_sizes is not None and postprocessor is not None:
                        _ = postprocessor(outputs, orig_sizes)
                end.record()
                torch.cuda.synchronize(device)
                elapsed_ms = float(start.elapsed_time(end))
            else:
                st = time.perf_counter()
                outputs = model(samples)
                orig_sizes = get_orig_target_sizes(targets, device)
                if orig_sizes is not None and postprocessor is not None:
                    _ = postprocessor(outputs, orig_sizes)
                elapsed_ms = (time.perf_counter() - st) * 1000.0

            comp = timer.collect_ms()
            matching = comp.get("matching_cost", 0.0)
            select = comp.get("select_topk", 0.0)
            norm_rank = max(select - matching, 0.0)

            total_ms += elapsed_ms
            total_images += bs
            sum_matching_ms += matching
            sum_norm_rank_ms += norm_rank

    latency_ms = total_ms / max(total_images, 1)
    fps = 1000.0 / latency_ms if latency_ms > 0 else 0.0
    peak_mem_gb = torch.cuda.max_memory_allocated(device) / (1024**3) if device.type == "cuda" else 0.0

    row = {
        "Model / stage": name,
        "GT cost branch": "off" if "Ours" in name else "-",
        "Params (M)": f"{count_params_m(model):.2f}",
        "GFLOPs": f"{gflops:.2f}",
        "Inference latency (ms/img)": f"{latency_ms:.3f}",
        "FPS": f"{fps:.2f}",
        "Train time / iter (s)": "-",
        "Peak GPU memory (GB)": f"{peak_mem_gb:.3f}",
        "Matching-cost time (ms/iter)": "0.000",
        "Norm. + ranking time (ms/iter)": f"{sum_norm_rank_ms / max(args.infer_iters, 1):.3f}",
    }
    return row


def run_one_train_iter(model, criterion, optimizer, scaler, samples, targets, device, use_amp: bool, epoch_value: int):
    optimizer.zero_grad(set_to_none=True)
    metas = dict(epoch=epoch_value, step=0, global_step=0, epoch_step=1)

    if scaler is not None:
        with torch.autocast(device_type=device.type, enabled=(device.type == "cuda" and use_amp)):
            outputs = model(samples, targets=targets)
        with torch.autocast(device_type=device.type, enabled=False):
            loss_dict = criterion(outputs, targets, **metas)
        loss = sum(loss_dict.values())
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        with torch.autocast(device_type=device.type, enabled=(device.type == "cuda" and use_amp)):
            outputs = model(samples, targets=targets)
            loss_dict = criterion(outputs, targets, **metas)
            loss = sum(loss_dict.values())
        loss.backward()
        optimizer.step()
    return float(loss.detach().item())


def measure_training_one(
    name: str,
    config_path: str,
    checkpoint_path: str,
    gflops: float,
    gt_cost_enabled: bool,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, Any]:
    solver = build_solver(config_path, checkpoint_path, for_train=True, device=device)
    model = solver.model
    criterion = solver.criterion
    optimizer = solver.optimizer
    scaler = getattr(solver, "scaler", None)

    model.train()
    criterion.train()
    set_cmqs_gt_cost_branch(model, enabled=gt_cost_enabled)

    if hasattr(solver.train_dataloader, "set_epoch"):
        try:
            solver.train_dataloader.set_epoch(0)
        except Exception:
            pass
    if hasattr(getattr(solver.train_dataloader, "sampler", None), "set_epoch"):
        try:
            solver.train_dataloader.sampler.set_epoch(0)
        except Exception:
            pass

    dataloader = solver.train_dataloader
    data_iter = iter(dataloader)
    timer = QuerySelectionTimer(device)
    instrument_query_selection(model, timer)
    use_amp = scaler is not None or getattr(args, "use_amp", False)
    epoch_value = 0 if gt_cost_enabled else int(args.after_exit_epoch)

    # Warm-up iterations, data loading and H2D transfer are excluded from timing.
    for _ in range(args.train_warmup):
        (samples, targets), data_iter = next_batch(data_iter, dataloader)
        samples = samples.to(device)
        targets = move_targets_to_device(targets, device)
        _ = run_one_train_iter(model, criterion, optimizer, scaler, samples, targets, device, use_amp, epoch_value)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    total_s = 0.0
    sum_matching_ms = 0.0
    sum_norm_rank_ms = 0.0

    for _ in range(args.train_iters):
        (samples, targets), data_iter = next_batch(data_iter, dataloader)
        samples = samples.to(device)
        targets = move_targets_to_device(targets, device)
        timer.reset()

        if device.type == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            _ = run_one_train_iter(model, criterion, optimizer, scaler, samples, targets, device, use_amp, epoch_value)
            end.record()
            torch.cuda.synchronize(device)
            elapsed_s = float(start.elapsed_time(end)) / 1000.0
        else:
            st = time.perf_counter()
            _ = run_one_train_iter(model, criterion, optimizer, scaler, samples, targets, device, use_amp, epoch_value)
            elapsed_s = time.perf_counter() - st

        comp = timer.collect_ms()
        matching = comp.get("matching_cost", 0.0)
        select = comp.get("select_topk", 0.0)
        norm_rank = max(select - matching, 0.0)

        total_s += elapsed_s
        sum_matching_ms += matching
        sum_norm_rank_ms += norm_rank

    local_time = total_s / max(args.train_iters, 1)
    local_matching = sum_matching_ms / max(args.train_iters, 1)
    local_norm_rank = sum_norm_rank_ms / max(args.train_iters, 1)
    local_mem = torch.cuda.max_memory_allocated(device) / (1024**3) if device.type == "cuda" else 0.0

    # In distributed training, wall-clock speed is determined by the slowest rank.
    avg_time = reduce_scalar(local_time, device, mode="max")
    avg_matching = reduce_scalar(local_matching, device, mode="max")
    avg_norm_rank = reduce_scalar(local_norm_rank, device, mode="max")
    peak_mem = reduce_scalar(local_mem, device, mode="max")
    params_m = reduce_scalar(count_params_m(model), device, mode="max")

    row = {
        "Model / stage": name,
        "GT cost branch": "on" if gt_cost_enabled else ("off" if "Ours" in name else "-"),
        "Params (M)": f"{params_m:.2f}",
        "GFLOPs": f"{gflops:.2f}",
        "Inference latency (ms/img)": "-",
        "FPS": "-",
        "Train time / iter (s)": f"{avg_time:.4f}",
        "Peak GPU memory (GB)": f"{peak_mem:.3f}",
        "Matching-cost time (ms/iter)": f"{avg_matching:.3f}",
        "Norm. + ranking time (ms/iter)": f"{avg_norm_rank:.3f}",
    }
    return row


def write_rows_csv(rows: List[Dict[str, Any]], out_path: str, append: bool):
    if not is_main_process():
        return
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "Model / stage",
        "GT cost branch",
        "Params (M)",
        "GFLOPs",
        "Inference latency (ms/img)",
        "FPS",
        "Train time / iter (s)",
        "Peak GPU memory (GB)",
        "Matching-cost time (ms/iter)",
        "Norm. + ranking time (ms/iter)",
    ]
    file_exists = out.exists()
    mode = "a" if append and file_exists else "w"
    with out.open(mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if mode == "w":
            writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"Saved Table 7 profiling rows to: {out}")
    for row in rows:
        print(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile Table 7 metrics for DEIM-L baseline and CMQS")

    parser.add_argument("--profile", choices=["inference", "training", "all"], default="all")
    parser.add_argument("--baseline-config", type=str, required=True)
    parser.add_argument("--baseline-ckpt", type=str, required=True)
    parser.add_argument("--cmqs-config", type=str, required=True)
    parser.add_argument("--cmqs-ckpt", type=str, required=True)
    parser.add_argument("--out", type=str, default="outputs/table7_profile/table7_profile.csv")
    parser.add_argument("--append", action="store_true", help="Append rows to an existing CSV")

    parser.add_argument("--baseline-gflops", type=float, default=91.0)
    parser.add_argument("--cmqs-gflops", type=float, default=93.0)

    parser.add_argument("--infer-warmup", type=int, default=50)
    parser.add_argument("--infer-iters", type=int, default=300)
    parser.add_argument("--train-warmup", type=int, default=20)
    parser.add_argument("--train-iters", type=int, default=200)
    parser.add_argument("--after-exit-epoch", type=int, default=999)
    parser.add_argument("--use-amp", action="store_true", help="Force autocast if your cfg does not create a scaler")

    return parser.parse_args()


def main():
    rank, world_size, local_rank, device = init_distributed_if_needed()
    args = parse_args()

    if is_main_process():
        print(f"Device: {device}, world_size={world_size}")
        print("Important: for final paper numbers, run inference on a clean single GPU and training profiling with the same 4-GPU setting as the main experiments.")

    rows: List[Dict[str, Any]] = []

    try:
        if args.profile in ("inference", "all"):
            # For paper-quality inference metrics, launch this script with one visible GPU.
            if world_size > 1 and is_main_process():
                print("[WARNING] Inference profiling is being run under distributed launch. For final latency/FPS, prefer single-GPU python launch.")
            if world_size == 1 or rank == 0:
                rows.append(measure_inference_one(
                    "DEIM-L baseline, inference",
                    args.baseline_config,
                    args.baseline_ckpt,
                    args.baseline_gflops,
                    args,
                    device,
                ))
                rows.append(measure_inference_one(
                    "Ours, inference",
                    args.cmqs_config,
                    args.cmqs_ckpt,
                    args.cmqs_gflops,
                    args,
                    device,
                ))
            if dist.is_available() and dist.is_initialized():
                dist.barrier()

        if args.profile in ("training", "all"):
            rows.append(measure_training_one(
                "DEIM-L baseline, training",
                args.baseline_config,
                args.baseline_ckpt,
                args.baseline_gflops,
                gt_cost_enabled=False,
                args=args,
                device=device,
            ))
            rows.append(measure_training_one(
                "Ours, early training",
                args.cmqs_config,
                args.cmqs_ckpt,
                args.cmqs_gflops,
                gt_cost_enabled=True,
                args=args,
                device=device,
            ))
            rows.append(measure_training_one(
                "Ours, after exit",
                args.cmqs_config,
                args.cmqs_ckpt,
                args.cmqs_gflops,
                gt_cost_enabled=False,
                args=args,
                device=device,
            ))

        write_rows_csv(rows, args.out, append=args.append)

    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
