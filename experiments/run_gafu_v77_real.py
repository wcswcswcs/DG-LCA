"""DG-KAN v7.7 real-only TimeAUC/S1/profiler runner.

v7.7 keeps the v7.6 M12 task/grad/efficiency implementation intact and adds
measured system diagnostics for the remaining blockers:

* step-vs-time AUC decomposition from the real task traces;
* real time accounting for online/offline teacher paths;
* torch.profiler availability/op diagnostics, with CUDA kernel metrics only
  reported when actually available;
* S1 memory source attribution copied from measured efficiency rows.

No loss, sampler, classwise, focal, margin, or class-weight tuning is added.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn.functional as F

import run_gafu_v76_real as v76
import run_gafu_v73_real as v73
import run_gafu_v72_real as v72
from dgkan_core import get_device, parse_int_list, parse_str_list, save_json, set_seed, write_csv
from run_gafu_v63 import V63Params
from run_gafu_v66_real import _git_commit, _git_status


PLAN_PATH = "docs/DG-KAN_v7.7_TimeAUC_S1_TrueKernelProfiler_DensePreservingFused_完整实验计划.md"
SCRIPT_PATH = "experiments/run_gafu_v77_real.py"
METRIC_UNAVAILABLE = v76.METRIC_UNAVAILABLE


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def _float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _mean(values: Iterable[Any], default: float = float("nan")) -> float:
    vals = [float(v) for v in values if _finite(v)]
    return sum(vals) / len(vals) if vals else default


def _read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _hash_file(path: Path) -> str:
    return v72._hash_file(path) if path.exists() else ""


def _sync_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _measure_block(device: torch.device, fn: Any) -> Tuple[Any, float]:
    _sync_if_needed(device)
    started = time.perf_counter()
    result = fn()
    _sync_if_needed(device)
    return result, time.perf_counter() - started


def _first_items(values: Sequence[int], limit: int) -> List[int]:
    return list(values[: max(0, int(limit))])


def _make_m12_student(args: argparse.Namespace, dataset: str, seed: int, bundle: Any, device: torch.device) -> Tuple[Any, Any, Any, float, float]:
    spec = v73._spec_map()["M12"]
    set_seed(v72._stable_seed(
        "v73-distill-student",
        dataset,
        seed,
        v73._distill_student_init_id(spec),
        spec.head_kind,
        spec.group_count,
        spec.shuffle,
    ))
    stack, head = v73._make_manual_candidate(
        spec,
        bundle.input_dim,
        bundle.num_classes,
        args.hidden_dim,
        args.basis_count,
        device,
    )
    params = V63Params(train_size=args.train_size, val_size=args.val_size, test_size=args.test_size, batch_size=args.batch_size)
    opt = v72.v71.FastAdamW([stack, head], lr=params.lr_manual * spec.lr_mult, weight_decay=args.weight_decay)
    return stack, head, opt, v73._distill_alpha_for_spec(args, spec), v73._distill_temperature_for_spec(args, spec)


def _make_mlp(args: argparse.Namespace, dataset: str, seed: int, bundle: Any, device: torch.device, tag: str) -> Tuple[torch.nn.Module, torch.optim.Optimizer]:
    params = V63Params(train_size=args.train_size, val_size=args.val_size, test_size=args.test_size, batch_size=args.batch_size)
    set_seed(v72._stable_seed(tag, dataset, seed, "mlp"))
    model = v72.v71._make_mlp(bundle.input_dim, bundle.num_classes, args.hidden_dim, 2).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=params.lr_mlp, weight_decay=args.weight_decay)
    return model, opt


def _time_accounting_m12(
    args: argparse.Namespace,
    dataset: str,
    seed: int,
    reps: int,
    teacher_mode: str,
) -> Dict[str, Any]:
    device = get_device(args.device)
    bundle = v72.v71.load_vision_bundle(
        dataset,
        data_root=Path(args.data_root),
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=seed,
        allow_fake_data=False,
    )
    x_train = bundle.x_train.to(device)
    y_train = bundle.y_train.to(device)
    x_val = bundle.x_val.to(device)
    y_val = bundle.y_val.to(device)
    teacher_stack, teacher_head, _teacher_meta = v73._train_c3_teacher(args, dataset, seed)
    stack, head, opt, alpha, temperature = _make_m12_student(args, dataset, seed, bundle, device)
    offline_logits: Dict[int, torch.Tensor] = {}
    if teacher_mode == "offline_logits":
        for rep in range(1, reps + 1):
            xb, _yb = v72.v71._select_batch(x_train, y_train, args.batch_size, rep)
            offline_logits[rep] = v73._teacher_logits(teacher_stack, teacher_head, xb)

    rows: List[Dict[str, float]] = []
    for rep in range(1, reps + 1):
        outer_start = time.perf_counter()
        (xb, yb), data_sec = _measure_block(device, lambda: v72.v71._select_batch(x_train, y_train, args.batch_size, rep))
        if teacher_mode == "online_teacher":
            tlogits, teacher_sec = _measure_block(device, lambda: v73._teacher_logits(teacher_stack, teacher_head, xb))
        elif teacher_mode == "offline_logits":
            tlogits = offline_logits[rep]
            teacher_sec = 0.0
        else:
            tlogits = torch.zeros((xb.shape[0], bundle.num_classes), device=device)
            teacher_sec = 0.0
        (h, caches), fwd_stack_sec = _measure_block(device, lambda: stack.forward_manual(xb))
        (logits, head_cache), fwd_head_sec = _measure_block(device, lambda: head.forward_manual(h))
        if teacher_mode == "no_teacher":
            (loss, grad_logits), loss_sec = _measure_block(
                device,
                lambda: v72._weighted_smooth_ce_and_grad(
                    logits,
                    yb,
                    0.0,
                    torch.ones(bundle.num_classes, device=device),
                ),
            )
        else:
            (loss, grad_logits, _kl), loss_sec = _measure_block(
                device,
                lambda: v73._distill_loss_and_grad(
                    logits,
                    yb,
                    tlogits,
                    0.0,
                    alpha=alpha,
                    temperature=temperature,
                ),
            )
        dh, head_bwd_sec = _measure_block(device, lambda: head.backward_manual(grad_logits, head_cache))
        _dx, stack_bwd_sec = _measure_block(device, lambda: stack.backward_manual(dh, caches))
        _update_norm, update_sec = _measure_block(device, lambda: opt.step(rep, reps, warmup_cosine=True))
        _sync_if_needed(device)
        outer_sec = time.perf_counter() - outer_start
        rows.append({
            "outer": outer_sec,
            "teacher": teacher_sec,
            "forward": fwd_stack_sec + fwd_head_sec,
            "forward_stack": fwd_stack_sec,
            "forward_head": fwd_head_sec,
            "loss": loss_sec,
            "backward": stack_bwd_sec + head_bwd_sec,
            "backward_stack": stack_bwd_sec,
            "backward_head": head_bwd_sec,
            "update": update_sec,
            "data_loading": data_sec,
        })

    _eval, validation_sec = _measure_block(
        device,
        lambda: v72.v71._manual_eval(stack, head, x_val, y_val, args.eval_batch_size, bundle.num_classes),
    )
    _log, logging_sec = _measure_block(device, lambda: {"candidate_id": "M12", "dataset": dataset, "seed": seed, "step": reps})
    return _summarize_time_rows("M12", dataset, seed, reps, teacher_mode, rows, validation_sec, logging_sec, int(args.trace_every), device)


def _time_accounting_mlp(
    args: argparse.Namespace,
    dataset: str,
    seed: int,
    reps: int,
    candidate_id: str,
    teacher_mode: str,
) -> Dict[str, Any]:
    device = get_device(args.device)
    bundle = v72.v71.load_vision_bundle(
        dataset,
        data_root=Path(args.data_root),
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=seed,
        allow_fake_data=False,
    )
    x_train = bundle.x_train.to(device)
    y_train = bundle.y_train.to(device)
    x_val = bundle.x_val.to(device)
    y_val = bundle.y_val.to(device)
    model, opt = _make_mlp(args, dataset, seed, bundle, device, f"v77-{candidate_id}-{teacher_mode}")
    teacher_stack = teacher_head = None
    offline_logits: Dict[int, torch.Tensor] = {}
    if candidate_id == "B2":
        teacher_stack, teacher_head, _teacher_meta = v73._train_c3_teacher(args, dataset, seed)
        if teacher_mode == "offline_logits":
            for rep in range(1, reps + 1):
                xb, _yb = v72.v71._select_batch(x_train, y_train, args.batch_size, rep)
                offline_logits[rep] = v73._teacher_logits(teacher_stack, teacher_head, xb)

    rows: List[Dict[str, float]] = []
    for rep in range(1, reps + 1):
        outer_start = time.perf_counter()
        (xb, yb), data_sec = _measure_block(device, lambda: v72.v71._select_batch(x_train, y_train, args.batch_size, rep))
        teacher_sec = 0.0
        if candidate_id == "B2" and teacher_mode == "online_teacher":
            assert teacher_stack is not None and teacher_head is not None
            tlogits, teacher_sec = _measure_block(device, lambda: v73._teacher_logits(teacher_stack, teacher_head, xb))
        elif candidate_id == "B2" and teacher_mode == "offline_logits":
            tlogits = offline_logits[rep]
        else:
            tlogits = None
        _zero, zero_sec = _measure_block(device, lambda: opt.zero_grad(set_to_none=True))
        logits, forward_sec = _measure_block(device, lambda: model(xb))
        if candidate_id == "B2":
            assert tlogits is not None
            loss, loss_sec = _measure_block(
                device,
                lambda: v73._distill_loss_autograd(logits, yb, tlogits, 0.0, alpha=0.25, temperature=4.0),
            )
        else:
            loss, loss_sec = _measure_block(device, lambda: F.cross_entropy(logits, yb))
        _bw, backward_sec = _measure_block(device, lambda: loss.backward())
        _step, step_sec = _measure_block(device, lambda: opt.step())
        _sync_if_needed(device)
        outer_sec = time.perf_counter() - outer_start
        rows.append({
            "outer": outer_sec,
            "teacher": teacher_sec,
            "forward": forward_sec,
            "forward_stack": METRIC_UNAVAILABLE,  # type: ignore[typeddict-item]
            "forward_head": METRIC_UNAVAILABLE,  # type: ignore[typeddict-item]
            "loss": loss_sec,
            "backward": backward_sec,
            "backward_stack": METRIC_UNAVAILABLE,  # type: ignore[typeddict-item]
            "backward_head": METRIC_UNAVAILABLE,  # type: ignore[typeddict-item]
            "update": zero_sec + step_sec,
            "data_loading": data_sec,
        })

    _eval, validation_sec = _measure_block(
        device,
        lambda: v72.v71._mlp_eval(model, x_val, y_val, args.eval_batch_size, bundle.num_classes),
    )
    _log, logging_sec = _measure_block(device, lambda: {"candidate_id": candidate_id, "dataset": dataset, "seed": seed, "step": reps})
    return _summarize_time_rows(candidate_id, dataset, seed, reps, teacher_mode, rows, validation_sec, logging_sec, int(args.trace_every), device)


def _summarize_time_rows(
    candidate_id: str,
    dataset: str,
    seed: int,
    reps: int,
    teacher_mode: str,
    rows: Sequence[Dict[str, Any]],
    validation_sec: float,
    logging_sec: float,
    trace_every: int,
    device: torch.device,
) -> Dict[str, Any]:
    outer = _mean(row.get("outer") for row in rows)
    teacher = _mean(row.get("teacher") for row in rows)
    forward = _mean(row.get("forward") for row in rows)
    backward = _mean(row.get("backward") for row in rows)
    update = _mean(row.get("update") for row in rows)
    loss = _mean(row.get("loss") for row in rows)
    data_loading = _mean(row.get("data_loading") for row in rows)
    fstack = _mean(row.get("forward_stack") for row in rows)
    fhead = _mean(row.get("forward_head") for row in rows)
    bstack = _mean(row.get("backward_stack") for row in rows)
    bhead = _mean(row.get("backward_head") for row in rows)
    known = sum(x for x in [teacher, forward, backward, update, loss, data_loading] if _finite(x))
    unknown = max(0.0, outer - known) if _finite(outer) and _finite(known) else float("nan")
    unknown_fraction = unknown / outer if _finite(unknown) and _finite(outer) and outer > 0 else float("nan")
    validation_amortized = validation_sec / max(1, trace_every)
    logging_amortized = logging_sec / max(1, trace_every)
    total_amortized = outer + validation_amortized + logging_amortized
    teacher_val_log_fraction = (teacher + validation_amortized + logging_amortized) / total_amortized if total_amortized > 0 else float("nan")
    cuda_sync_separated = int(device.type != "cuda")
    return {
        "stage": "P1",
        "candidate_id": candidate_id,
        "dataset": dataset,
        "seed": seed,
        "teacher_mode": teacher_mode,
        "measured_reps": reps,
        "device": str(device),
        "wall_clock_time_sec_total_per_step_mean": outer,
        "train_step_time_sec": outer,
        "train_only_time_sec": outer - teacher if _finite(outer) and _finite(teacher) else METRIC_UNAVAILABLE,
        "forward_time_sec": forward,
        "forward_stack_time_sec": fstack if _finite(fstack) else METRIC_UNAVAILABLE,
        "forward_head_time_sec": fhead if _finite(fhead) else METRIC_UNAVAILABLE,
        "backward_time_sec": backward,
        "backward_stack_time_sec": bstack if _finite(bstack) else METRIC_UNAVAILABLE,
        "backward_head_time_sec": bhead if _finite(bhead) else METRIC_UNAVAILABLE,
        "update_time_sec": update,
        "teacher_forward_time_sec": teacher,
        "distill_loss_time_sec": loss,
        "validation_time_sec": validation_sec,
        "validation_amortized_per_step_sec": validation_amortized,
        "logging_time_sec": logging_sec,
        "logging_amortized_per_step_sec": logging_amortized,
        "cuda_sync_time_sec": 0.0 if device.type != "cuda" else METRIC_UNAVAILABLE,
        "cuda_sync_time_status": "not_applicable_cpu" if device.type != "cuda" else "included_in_block_timing_not_separately_measured",
        "data_loading_time_sec": data_loading if _finite(data_loading) else METRIC_UNAVAILABLE,
        "compile_warmup_time_sec": METRIC_UNAVAILABLE,
        "profiler_overhead_sec": METRIC_UNAVAILABLE,
        "unknown_time_sec": unknown if _finite(unknown) else METRIC_UNAVAILABLE,
        "unknown_time_fraction": unknown_fraction if _finite(unknown_fraction) else METRIC_UNAVAILABLE,
        "teacher_validation_logging_fraction": teacher_val_log_fraction if _finite(teacher_val_log_fraction) else METRIC_UNAVAILABLE,
        "samples_per_second_train_only": (128 / (outer - teacher)) if _finite(outer) and _finite(teacher) and outer > teacher else METRIC_UNAVAILABLE,
        "samples_per_second_end_to_end": 128 / total_amortized if total_amortized > 0 else METRIC_UNAVAILABLE,
        "time_accounting_pass_row": int(_finite(unknown_fraction) and unknown_fraction <= 0.10 and cuda_sync_separated),
        "fake_data_used": 0,
        "proxy_row_used": 0,
    }


def _write_time_accounting(out_dir: Path, args: argparse.Namespace) -> List[Dict[str, Any]]:
    datasets = parse_str_list(args.datasets)
    seeds = _first_items(parse_int_list(args.seeds), 3)
    reps = max(3, min(20, int(args.bench_reps)))
    rows: List[Dict[str, Any]] = []
    for dataset in datasets:
        for seed in seeds:
            rows.append(_time_accounting_mlp(args, dataset, seed, reps, "B0", "no_teacher"))
            rows.append(_time_accounting_mlp(args, dataset, seed, reps, "B2", "online_teacher"))
            rows.append(_time_accounting_mlp(args, dataset, seed, reps, "B2", "offline_logits"))
            rows.append(_time_accounting_m12(args, dataset, seed, reps, "online_teacher"))
            rows.append(_time_accounting_m12(args, dataset, seed, reps, "offline_logits"))
            rows.append(_time_accounting_m12(args, dataset, seed, reps, "no_teacher"))
    write_csv(out_dir / "p1_time_accounting_v77.csv", rows)
    return rows


def _write_auc_decomposition(out_dir: Path) -> List[Dict[str, Any]]:
    p7 = _read_csv_rows(out_dir / "p7_time_auc_v76.csv")
    candidates = sorted({str(row.get("candidate_id")) for row in p7})
    rows: List[Dict[str, Any]] = []
    b0_step = _mean(row.get("val_loss_auc_step") for row in p7 if str(row.get("candidate_id")) == "B0")
    b0_time = _mean(row.get("val_loss_auc_time") for row in p7 if str(row.get("candidate_id")) == "B0")
    b0_acc_step = _mean(row.get("val_acc_auc_step") for row in p7 if str(row.get("candidate_id")) == "B0")
    b0_acc_time = _mean(row.get("val_acc_auc_time") for row in p7 if str(row.get("candidate_id")) == "B0")
    for cid in candidates:
        step = _mean(row.get("val_loss_auc_step") for row in p7 if str(row.get("candidate_id")) == cid)
        time_auc = _mean(row.get("val_loss_auc_time") for row in p7 if str(row.get("candidate_id")) == cid)
        acc_step = _mean(row.get("val_acc_auc_step") for row in p7 if str(row.get("candidate_id")) == cid)
        acc_time = _mean(row.get("val_acc_auc_time") for row in p7 if str(row.get("candidate_id")) == cid)
        wall = _mean(row.get("wall_clock_time_sec") for row in p7 if str(row.get("candidate_id")) == cid)
        rows.append({
            "stage": "P2",
            "candidate_id": cid,
            "val_loss_auc_step_mean": step if _finite(step) else METRIC_UNAVAILABLE,
            "val_loss_auc_time_mean": time_auc if _finite(time_auc) else METRIC_UNAVAILABLE,
            "val_acc_auc_step_mean": acc_step if _finite(acc_step) else METRIC_UNAVAILABLE,
            "val_acc_auc_time_mean": acc_time if _finite(acc_time) else METRIC_UNAVAILABLE,
            "wall_clock_time_sec_mean": wall if _finite(wall) else METRIC_UNAVAILABLE,
            "val_loss_auc_step_ratio_vs_B0": step / b0_step if _finite(step) and _finite(b0_step) and b0_step != 0 else METRIC_UNAVAILABLE,
            "val_loss_auc_time_ratio_vs_B0": time_auc / b0_time if _finite(time_auc) and _finite(b0_time) and b0_time != 0 else METRIC_UNAVAILABLE,
            "val_acc_auc_step_delta_vs_B0": acc_step - b0_acc_step if _finite(acc_step) and _finite(b0_acc_step) else METRIC_UNAVAILABLE,
            "val_acc_auc_time_delta_vs_B0": acc_time - b0_acc_time if _finite(acc_time) and _finite(b0_acc_time) else METRIC_UNAVAILABLE,
            "optimization_dynamics_problem": int(_finite(step) and _finite(b0_step) and b0_step != 0 and step / b0_step > 1.10),
            "runtime_problem": int(_finite(step) and _finite(b0_step) and _finite(time_auc) and _finite(b0_time) and step <= b0_step and time_auc > b0_time),
            "time_auc_pass": int(_finite(time_auc) and _finite(b0_time) and time_auc <= b0_time),
            "fake_data_used": 0,
            "proxy_row_used": 0,
        })
    write_csv(out_dir / "p2_auc_dynamics_v77.csv", rows)
    return rows


def _event_device_us(event: Any) -> float:
    for attr in ("device_time_total", "self_device_time_total", "cuda_time_total", "self_cuda_time_total"):
        try:
            value = getattr(event, attr)
            if value is not None:
                return float(value)
        except Exception:
            continue
    return 0.0


def _looks_like_real_kernel_name(name: str) -> bool:
    text = str(name)
    if text in {"teacher_forward", "student_forward", "distill_loss", "manual_backward", "manual_update"}:
        return False
    if text.startswith(("aten::", "cuda", "ProfilerStep", "Memcpy", "Memset")):
        return False
    # Real CUDA kernel names usually expose C++/mangled symbols, templates, or
    # backend names. Phase ranges / ATen op aggregates are kept as op-level data
    # and are not counted as true kernel names.
    kernel_markers = ("void ", "::", "<", ">", "kernel", "gemm", "cutlass", "cudnn", "triton")
    return any(marker in text.lower() for marker in kernel_markers)


def _profile_m12_step(args: argparse.Namespace, dataset: str = "MNIST", seed: int = 0, batch_size: int | None = None) -> Dict[str, Any]:
    device = get_device(args.device)
    batch = int(batch_size or args.batch_size)
    bundle = v72.v71.load_vision_bundle(
        dataset,
        data_root=Path(args.data_root),
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=seed,
        allow_fake_data=False,
    )
    x_train = bundle.x_train.to(device)
    y_train = bundle.y_train.to(device)
    teacher_stack, teacher_head, _teacher_meta = v73._train_c3_teacher(args, dataset, seed)
    stack, head, opt, alpha, temperature = _make_m12_student(args, dataset, seed, bundle, device)
    xb, yb = v72.v71._select_batch(x_train, y_train, batch, 1)

    def one_step() -> None:
        with torch.profiler.record_function("teacher_forward"):
            tlogits = v73._teacher_logits(teacher_stack, teacher_head, xb)
        with torch.profiler.record_function("student_forward"):
            h, caches = stack.forward_manual(xb)
            logits, head_cache = head.forward_manual(h)
        with torch.profiler.record_function("distill_loss"):
            _loss, grad_logits, _kl = v73._distill_loss_and_grad(logits, yb, tlogits, 0.0, alpha=alpha, temperature=temperature)
        with torch.profiler.record_function("manual_backward"):
            dh = head.backward_manual(grad_logits, head_cache)
            stack.backward_manual(dh, caches)
        with torch.profiler.record_function("manual_update"):
            opt.step(1, 1, warmup_cosine=True)

    cuda_available = bool(torch.cuda.is_available() and device.type == "cuda")
    activities = [torch.profiler.ProfilerActivity.CPU]
    if cuda_available:
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    profiler_status = "measured_cpu_ops"
    try:
        with torch.profiler.profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
        ) as prof:
            one_step()
        events = prof.key_averages()
        raw_events = list(prof.events())
        cpu_events = sorted(events, key=lambda e: float(getattr(e, "self_cpu_time_total", 0.0)), reverse=True)
        total_cpu_us = sum(float(getattr(e, "self_cpu_time_total", 0.0)) for e in cpu_events)
        top = cpu_events[:3]
        raw_cuda_events = [
            e for e in raw_events
            if "cuda" in str(getattr(e, "device_type", "")).lower()
            and _event_device_us(e) > 0.0
            and _looks_like_real_kernel_name(str(getattr(e, "key", "")))
        ]
        raw_cuda_events = sorted(
            raw_cuda_events,
            key=_event_device_us,
            reverse=True,
        )
        op_cuda_events = [
            e for e in events
            if _event_device_us(e) > 0.0
        ]
        op_cuda_events = sorted(
            op_cuda_events,
            key=_event_device_us,
            reverse=True,
        )
        cuda_events = raw_cuda_events
        total_cuda_us = sum(_event_device_us(e) for e in cuda_events)
        op_cuda_total_us = sum(_event_device_us(e) for e in op_cuda_events)
        if cuda_available and cuda_events:
            profiler_status = "measured_cuda_kernel_events"
        elif cuda_available and op_cuda_events:
            profiler_status = "measured_cuda_op_time_without_kernel_names"
        elif cuda_available:
            profiler_status = "cuda_available_no_cuda_events"
        top3_kernel_fraction = (
            sum(_event_device_us(e) for e in cuda_events[:3]) / total_cuda_us
            if total_cuda_us > 0 else float("nan")
        )
        phase_to_kernel_mapping = 0
        profiler_pass = int(
            bool(cuda_events)
            and total_cuda_us > 0.0
            and phase_to_kernel_mapping
            and _finite(top3_kernel_fraction)
            and top3_kernel_fraction >= 0.70
        )
        return {
            "stage": "P3",
            "candidate_id": "M12",
            "dataset": dataset,
            "seed": seed,
            "batch_size": batch,
            "profiler_backend": "torch.profiler",
            "torch_profiler_available": 1,
            "nsys_available": METRIC_UNAVAILABLE,
            "ncu_available": METRIC_UNAVAILABLE,
            "cuda_activities_available": int(cuda_available),
            "profile_memory_available": 1,
            "with_stack_available": 0,
            "nvtx_ranges_available": 0,
            "profiler_status": profiler_status,
            "kernel_count_total": len(cuda_events) if cuda_events else METRIC_UNAVAILABLE,
            "unique_kernel_names_total": len({e.key for e in cuda_events}) if cuda_events else METRIC_UNAVAILABLE,
            "kernel_name_available": int(bool(cuda_events)),
            "kernel_duration_available": int(bool(cuda_events)),
            "phase_to_kernel_mapping_available": phase_to_kernel_mapping,
            "torch_op_count_total": len(cpu_events),
            "top_cpu_op_name_1": top[0].key if len(top) > 0 else METRIC_UNAVAILABLE,
            "top_cpu_op_time_ms_1": float(getattr(top[0], "self_cpu_time_total", 0.0)) / 1000.0 if len(top) > 0 else METRIC_UNAVAILABLE,
            "top_cpu_op_calls_1": int(getattr(top[0], "count", 0)) if len(top) > 0 else METRIC_UNAVAILABLE,
            "top_cpu_op_name_2": top[1].key if len(top) > 1 else METRIC_UNAVAILABLE,
            "top_cpu_op_time_ms_2": float(getattr(top[1], "self_cpu_time_total", 0.0)) / 1000.0 if len(top) > 1 else METRIC_UNAVAILABLE,
            "top_cpu_op_calls_2": int(getattr(top[1], "count", 0)) if len(top) > 1 else METRIC_UNAVAILABLE,
            "top_cpu_op_name_3": top[2].key if len(top) > 2 else METRIC_UNAVAILABLE,
            "top_cpu_op_time_ms_3": float(getattr(top[2], "self_cpu_time_total", 0.0)) / 1000.0 if len(top) > 2 else METRIC_UNAVAILABLE,
            "top_cpu_op_calls_3": int(getattr(top[2], "count", 0)) if len(top) > 2 else METRIC_UNAVAILABLE,
            "top_kernel_name_1": cuda_events[0].key if len(cuda_events) > 0 else METRIC_UNAVAILABLE,
            "top_kernel_time_ms_1": _event_device_us(cuda_events[0]) / 1000.0 if len(cuda_events) > 0 else METRIC_UNAVAILABLE,
            "top_kernel_calls_1": int(getattr(cuda_events[0], "count", 0)) if len(cuda_events) > 0 else METRIC_UNAVAILABLE,
            "top_kernel_name_2": cuda_events[1].key if len(cuda_events) > 1 else METRIC_UNAVAILABLE,
            "top_kernel_time_ms_2": _event_device_us(cuda_events[1]) / 1000.0 if len(cuda_events) > 1 else METRIC_UNAVAILABLE,
            "top_kernel_calls_2": int(getattr(cuda_events[1], "count", 0)) if len(cuda_events) > 1 else METRIC_UNAVAILABLE,
            "top_kernel_name_3": cuda_events[2].key if len(cuda_events) > 2 else METRIC_UNAVAILABLE,
            "top_kernel_time_ms_3": _event_device_us(cuda_events[2]) / 1000.0 if len(cuda_events) > 2 else METRIC_UNAVAILABLE,
            "top_kernel_calls_3": int(getattr(cuda_events[2], "count", 0)) if len(cuda_events) > 2 else METRIC_UNAVAILABLE,
            "cpu_self_time_total_ms": total_cpu_us / 1000.0,
            "cuda_kernel_time_total_ms": total_cuda_us / 1000.0 if cuda_events else METRIC_UNAVAILABLE,
            "cuda_op_time_total_ms": op_cuda_total_us / 1000.0 if op_cuda_events else METRIC_UNAVAILABLE,
            "top3_cpu_time_explained_fraction": (
                sum(float(getattr(e, "self_cpu_time_total", 0.0)) for e in top) / total_cpu_us
                if total_cpu_us > 0 else METRIC_UNAVAILABLE
            ),
            "top3_kernel_time_explained_fraction": top3_kernel_fraction if _finite(top3_kernel_fraction) else METRIC_UNAVAILABLE,
            "unknown_time_fraction": 0.0 if cuda_events else METRIC_UNAVAILABLE,
            "profiler_pass": profiler_pass,
            "fake_data_used": 0,
            "proxy_row_used": 0,
        }
    except Exception as exc:
        return {
            "stage": "P3",
            "candidate_id": "M12",
            "dataset": dataset,
            "seed": seed,
            "batch_size": batch,
            "profiler_backend": "torch.profiler",
            "torch_profiler_available": 0,
            "cuda_activities_available": int(cuda_available),
            "profiler_status": f"profiler_error:{type(exc).__name__}",
            "kernel_count_total": METRIC_UNAVAILABLE,
            "kernel_name_available": 0,
            "kernel_duration_available": 0,
            "profiler_pass": 0,
            "fake_data_used": 0,
            "proxy_row_used": 0,
        }


def _write_profiler(out_dir: Path, args: argparse.Namespace) -> List[Dict[str, Any]]:
    datasets = [ds for ds in parse_str_list(args.datasets) if ds in {"MNIST", "KMNIST"}] or ["MNIST"]
    batches = [b for b in parse_int_list(args.bench_batch_sizes) if b in {128, 512}] or [int(args.batch_size)]
    rows = [_profile_m12_step(args, dataset=ds, seed=0, batch_size=bs) for ds in datasets[:2] for bs in batches[:2]]
    write_csv(out_dir / "p3_true_kernel_profiler_v77.csv", rows)
    return rows


def _write_s1_memory_attribution(out_dir: Path) -> List[Dict[str, Any]]:
    eff_rows = _read_csv_rows(out_dir / "p10_efficiency_profiler.csv")
    rows: List[Dict[str, Any]] = []
    for row in eff_rows:
        cid = str(row.get("candidate_id"))
        if cid not in {"B0", "B2", "M12", "M14", "M16"}:
            continue
        mem = _float(row.get("memory_ratio"))
        peak = _float(row.get("peak_allocated_MB"))
        cache = _float(row.get("cache_total_MB_measured", row.get("cache_total_MB")))
        opt = _float(row.get("optimizer_state_memory_MB", row.get("optimizer_state_MB")))
        gap = mem - 1.0 if _finite(mem) else float("nan")
        actionable_cache = cache / peak if _finite(cache) and _finite(peak) and peak > 0 else float("nan")
        actionable_opt = opt / peak if _finite(opt) and _finite(peak) and peak > 0 else float("nan")
        rows.append({
            "stage": "P4",
            "candidate_id": cid,
            "dataset": row.get("dataset"),
            "batch_size": row.get("batch_size"),
            "peak_allocated_MB": row.get("peak_allocated_MB"),
            "peak_reserved_MB": row.get("peak_reserved_MB"),
            "memory_ratio": row.get("memory_ratio"),
            "step_ratio": row.get("step_ratio"),
            "forward_ratio": row.get("forward_ratio"),
            "backward_ratio": row.get("backward_ratio"),
            "update_ratio": row.get("update_ratio"),
            "root_input_cache_MB": row.get("cache_x_MB", row.get("cache_x_MB_measured", METRIC_UNAVAILABLE)),
            "hidden_y_cache_MB": row.get("hidden_y_cache_MB", row.get("cache_hidden_y_MB", METRIC_UNAVAILABLE)),
            "head_temp_MB": row.get("head_temp_MB", METRIC_UNAVAILABLE),
            "stack_temp_MB": row.get("linear_body_temp_MB", METRIC_UNAVAILABLE),
            "manual_cache_MB": row.get("manual_cache_MB", row.get("manual_cache_MB_measured", METRIC_UNAVAILABLE)),
            "optimizer_state_MB": row.get("optimizer_state_MB", row.get("optimizer_state_memory_MB", METRIC_UNAVAILABLE)),
            "packed_owner_state_MB": row.get("packed_owner_state_MB", METRIC_UNAVAILABLE),
            "allocator_padding_MB": METRIC_UNAVAILABLE,
            "reserved_unallocated_MB": (
                _float(row.get("peak_reserved_MB")) - _float(row.get("peak_allocated_MB"))
                if _finite(row.get("peak_reserved_MB")) and _finite(row.get("peak_allocated_MB")) else METRIC_UNAVAILABLE
            ),
            "largest_live_tensor_MB": row.get("largest_live_tensor_MB", row.get("largest_live_tensor_MB_measured", METRIC_UNAVAILABLE)),
            "materialized_tensor_count": row.get("materialized_tensor_count", row.get("materialized_tensor_count_measured", METRIC_UNAVAILABLE)),
            "kernel_count_total": row.get("kernel_count_total", METRIC_UNAVAILABLE),
            "top1_memory_source": row.get("top1_memory_source", row.get("top1_memory_source_measured", METRIC_UNAVAILABLE)),
            "top2_memory_source": row.get("top2_memory_source", row.get("top2_memory_source_measured", METRIC_UNAVAILABLE)),
            "top3_memory_source": row.get("top3_memory_source", row.get("top3_memory_source_measured", METRIC_UNAVAILABLE)),
            "unknown_memory_fraction": row.get("unknown_memory_fraction", row.get("unknown_memory_fraction_measured", METRIC_UNAVAILABLE)),
            "s1_memory_gap": gap if _finite(gap) else METRIC_UNAVAILABLE,
            "cache_fraction_of_peak": actionable_cache if _finite(actionable_cache) else METRIC_UNAVAILABLE,
            "optimizer_state_fraction_of_peak": actionable_opt if _finite(actionable_opt) else METRIC_UNAVAILABLE,
            "actionable_cache_source": int(_finite(actionable_cache) and actionable_cache >= 0.02),
            "actionable_optimizer_source": int(_finite(actionable_opt) and actionable_opt >= 0.02),
            "fake_data_used": 0,
            "proxy_row_used": 0,
        })
    write_csv(out_dir / "p4_s1_memory_attribution_v77.csv", rows)
    return rows


def _write_not_implemented_rows(out_dir: Path) -> None:
    rows = [
        {
            "stage": "P5",
            "microkernel": "dense_preserving_fused_backward",
            "implementation_status": "not_implemented",
            "stage_status": "not_run",
            "reason": "v77 profiler did not provide real CUDA kernel attribution in this environment",
            "fake_data_used": 0,
            "proxy_row_used": 0,
        },
        {
            "stage": "P6",
            "candidate_id": "F9-M12-fused-streaming-S1-combo",
            "implementation_status": "not_implemented",
            "stage_status": "not_run",
            "reason": "requires P3 real kernel bottleneck and P5 validated microkernel first",
            "fake_data_used": 0,
            "proxy_row_used": 0,
        },
    ]
    write_csv(out_dir / "p5_p6_fused_package_status_v77.csv", rows)


def _write_v77_route(out_dir: Path, args: argparse.Namespace, time_rows: Sequence[Dict[str, Any]], auc_rows: Sequence[Dict[str, Any]], profiler_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    v76_route = json.loads((out_dir / "v76_route_decision.json").read_text(encoding="utf-8")) if (out_dir / "v76_route_decision.json").exists() else {}
    m12_auc = next((r for r in auc_rows if str(r.get("candidate_id")) == "M12"), {})
    time_unknowns = [_float(r.get("unknown_time_fraction")) for r in time_rows if _finite(r.get("unknown_time_fraction"))]
    time_accounting_pass = int(bool(time_rows) and all(str(r.get("time_accounting_pass_row")) in {"1", "1.0", "true", "True"} for r in time_rows))
    profiler_pass = int(bool(profiler_rows) and all(str(r.get("profiler_pass")) in {"1", "1.0", "true", "True"} for r in profiler_rows))
    p0_repro = int(
        _finite(v76_route.get("macro_gap"))
        and abs(float(v76_route["macro_gap"]) - 0.020572916666666666) <= 0.004
        and _finite(v76_route.get("memory_ratio_mean"))
        and abs(float(v76_route["memory_ratio_mean"]) - 1.0143359939345469) <= 0.02
        and _finite(v76_route.get("step_ratio_mean"))
        and abs(float(v76_route["step_ratio_mean"]) - 1.1683173565223521) <= 0.10
    )
    fullgrid_s2 = int(str(v76_route.get("fullgrid_s2_pass")) in {"1", "1.0", "true", "True"})
    fullgrid_s1 = int(str(v76_route.get("fullgrid_s1_pass")) in {"1", "1.0", "true", "True"})
    time_auc_pass = int(str(m12_auc.get("time_auc_pass")) in {"1", "1.0", "true", "True"})
    formal = int(p0_repro and fullgrid_s1 and time_auc_pass and profiler_pass)
    minimum = int(p0_repro and fullgrid_s2 and time_accounting_pass and profiler_pass)
    if formal:
        route = "S1-TimeAUC-S1-ProfilerSystemSuccess"
        blocker = "none"
    elif minimum:
        route = "S2-v77-MinimumProfilerAccountingSuccess"
        blocker = "s1_or_time_auc_not_closed"
    elif not profiler_pass:
        route = "R1-TrueKernelProfilerNotClosed"
        blocker = "true_cuda_kernel_profiler_unavailable_or_incomplete"
    elif not time_auc_pass:
        route = "R2-TimeAUCNotClosed"
        blocker = "time_auc"
    elif not fullgrid_s1:
        route = "R3-S1MemoryNotClosed"
        blocker = "s1_memory"
    else:
        route = "R4-SystemEvidenceIncomplete"
        blocker = "missing_required_evidence"
    decision = {
        "route": route,
        "best_candidate_id": "M12",
        "p0_reproduction_pass": p0_repro,
        "strict_pass": v76_route.get("strict_pass", METRIC_UNAVAILABLE),
        "grad_pass": v76_route.get("grad_pass", METRIC_UNAVAILABLE),
        "macro_significant_pass_10seed": v76_route.get("macro_significant_pass_10seed", METRIC_UNAVAILABLE),
        "fairness_pass": v76_route.get("fairness_pass", METRIC_UNAVAILABLE),
        "fullgrid_s2_pass": fullgrid_s2,
        "fullgrid_s1_pass": fullgrid_s1,
        "time_accounting_pass": time_accounting_pass,
        "profiler_pass": profiler_pass,
        "time_auc_pass": time_auc_pass,
        "success_v77_minimum": minimum,
        "success_v77_formal": formal,
        "macro_gap": v76_route.get("macro_gap", METRIC_UNAVAILABLE),
        "ci95_low": v76_route.get("ci95_low", METRIC_UNAVAILABLE),
        "holm_p": v76_route.get("holm_p", METRIC_UNAVAILABLE),
        "test_gap": v76_route.get("test_gap", METRIC_UNAVAILABLE),
        "ECE_delta": v76_route.get("ECE_delta", METRIC_UNAVAILABLE),
        "NLL_delta": v76_route.get("NLL_delta", METRIC_UNAVAILABLE),
        "memory_ratio_mean": v76_route.get("memory_ratio_mean", METRIC_UNAVAILABLE),
        "memory_ratio_max": v76_route.get("memory_ratio_max", METRIC_UNAVAILABLE),
        "step_ratio_mean": v76_route.get("step_ratio_mean", METRIC_UNAVAILABLE),
        "step_ratio_max": v76_route.get("step_ratio_max", METRIC_UNAVAILABLE),
        "s2_shape_count": v76_route.get("s2_shape_count", METRIC_UNAVAILABLE),
        "s1_shape_count": v76_route.get("s1_shape_count", METRIC_UNAVAILABLE),
        "val_loss_auc_step_ratio_vs_B0": m12_auc.get("val_loss_auc_step_ratio_vs_B0", METRIC_UNAVAILABLE),
        "val_loss_auc_time_ratio_vs_B0": m12_auc.get("val_loss_auc_time_ratio_vs_B0", METRIC_UNAVAILABLE),
        "mean_time_accounting_unknown_fraction": _mean(time_unknowns) if time_unknowns else METRIC_UNAVAILABLE,
        "max_time_accounting_unknown_fraction": max(time_unknowns) if time_unknowns else METRIC_UNAVAILABLE,
        "primary_blocker": blocker,
        "classwise_is_hard_gate": False,
        "no_fake": True,
        "no_proxy": True,
    }
    save_json(out_dir / "v77_route_decision.json", decision)
    write_csv(out_dir / "p13_candidate_selection_v77.csv", [{**decision, "stage": "P13", "fake_data_used": 0, "proxy_row_used": 0}])
    return decision


def _write_v77_audit_and_manifest(out_dir: Path, args: argparse.Namespace, route: Dict[str, Any]) -> None:
    artifact_paths = [
        out_dir / "candidate_registry.csv",
        out_dir / "p9_task_gate.csv",
        out_dir / "p9_task_trace.csv",
        out_dir / "p9_task_summary.csv",
        out_dir / "p1_significance_audit.csv",
        out_dir / "p2_full_gradient_correctness.csv",
        out_dir / "p10_efficiency_profiler.csv",
        out_dir / "p10_efficiency_summary.csv",
        out_dir / "p7_time_auc_v76.csv",
        out_dir / "p1_time_accounting_v77.csv",
        out_dir / "p2_auc_dynamics_v77.csv",
        out_dir / "p3_true_kernel_profiler_v77.csv",
        out_dir / "p4_s1_memory_attribution_v77.csv",
        out_dir / "p5_p6_fused_package_status_v77.csv",
        out_dir / "p13_candidate_selection_v77.csv",
    ]
    audit = v72._audit_fake_proxy(artifact_paths)
    write_csv(out_dir / "v77_provenance_audit.csv", [{
        **audit,
        "plan_path": PLAN_PATH,
        "script_path": SCRIPT_PATH,
        "fake_data_used": 0,
        "proxy_row_used": 0,
    }])
    save_json(out_dir / "v77_manifest.json", {
        "plan_path": PLAN_PATH,
        "script_path": SCRIPT_PATH,
        "runner_reuse": "run_gafu_v76_real.py",
        "out_dir": str(out_dir),
        "source_commit": _git_commit(),
        "git_status": _git_status(),
        "datasets": parse_str_list(args.datasets),
        "seeds": parse_int_list(args.seeds),
        "candidates": parse_str_list(args.candidates),
        "bench_batch_sizes": parse_int_list(args.bench_batch_sizes),
        "grad_batch_sizes": parse_int_list(args.grad_batch_sizes),
        "postprocess_artifact_hashes": {p.name: _hash_file(p) for p in artifact_paths if p.exists()},
        "v77_route": route,
        "v77_audit": audit,
    })


def _write_v77_postprocess(out_dir: Path, args: argparse.Namespace) -> None:
    time_rows = _write_time_accounting(out_dir, args)
    auc_rows = _write_auc_decomposition(out_dir)
    profiler_rows = _write_profiler(out_dir, args)
    _write_s1_memory_attribution(out_dir)
    _write_not_implemented_rows(out_dir)
    route = _write_v77_route(out_dir, args, time_rows, auc_rows, profiler_rows)
    _write_v77_audit_and_manifest(out_dir, args, route)


def run(args: argparse.Namespace) -> None:
    v76.PLAN_PATH = PLAN_PATH
    v76.SCRIPT_PATH = SCRIPT_PATH
    v76.run(args)
    out_dir = Path(args.out_dir)
    _write_v77_postprocess(out_dir, args)
    for src_name, dst_name in {
        "v76_manifest.json": "v77_reused_v76_manifest.json",
        "v76_route_decision.json": "v77_reused_v76_route_decision.json",
        "v76_provenance_audit.csv": "v77_reused_v76_provenance_audit.csv",
    }.items():
        src = out_dir / src_name
        if src.exists():
            shutil.copyfile(src, out_dir / dst_name)


def parse_args() -> argparse.Namespace:
    return v76.parse_args()


if __name__ == "__main__":
    run(parse_args())
