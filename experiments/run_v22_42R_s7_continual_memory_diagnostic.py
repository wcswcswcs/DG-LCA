#!/usr/bin/env python3
"""v22.42R S7 continual memory diagnostic.

Diagnostic only.  This runner checks whether a continuous memory velocity from
an old task reduces forgetting on a later task.  It does not promote official
routes and does not use candidate-action runtime selection.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import math
import os
from pathlib import Path
import random
import shlex
import statistics
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import run_v22_37_causal_instrumented_functional_optimizer as core
from experiments import run_v22_40_continuous_functional_flow_fu as v2240
from experiments.run_v22_42R_support_first_residual_kan_carrier import append_exec, now_sg, safe_fragment


PYTHON = os.environ.get("KAN_PYTHON", "/home/chengshun.wang/miniconda3/envs/kan/bin/python")
OUT_ROOT = ROOT / "results/v22_42R"
S7_ROOT = OUT_ROOT / "s7"
LOG_ROOT = OUT_ROOT / "logs"

VARIANTS = {
    "optimizer_alone",
    "memory_signal_flow",
    "memory_random_control",
    "memory_signflip_control",
}


def ensure_out() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    S7_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        rows = [{"status": "empty"}]
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def finite(value: Any, default: float | None = None) -> float | None:
    try:
        if value in {"", None}:
            return default
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def json_clean(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: json_clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_clean(v) for v in value]
    return value


def seed_all(seed: int) -> None:
    import torch

    random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def mnist_split_loaders(train_size: int, held_size: int, batch_size: int, seed: int) -> tuple[Any, Any, Any, Any]:
    from torch.utils.data import DataLoader, Subset
    from torchvision import datasets, transforms

    transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
    train_ds = datasets.MNIST(root=str(ROOT / "data"), train=True, transform=transform, download=False)
    test_ds = datasets.MNIST(root=str(ROOT / "data"), train=False, transform=transform, download=False)
    rng = random.Random(int(seed) + 7042)
    train_targets = train_ds.targets.tolist()
    test_targets = test_ds.targets.tolist()
    old_train = [i for i, y in enumerate(train_targets) if int(y) < 5]
    new_train = [i for i, y in enumerate(train_targets) if int(y) >= 5]
    old_test = [i for i, y in enumerate(test_targets) if int(y) < 5]
    new_test = [i for i, y in enumerate(test_targets) if int(y) >= 5]
    for idxs in [old_train, new_train, old_test, new_test]:
        rng.shuffle(idxs)
    old_train = old_train[: int(train_size)]
    new_train = new_train[: int(train_size)]
    old_test = old_test[: int(held_size)]
    new_test = new_test[: int(held_size)]
    import torch

    gen_old = torch.Generator().manual_seed(int(seed) + 7111)
    gen_new = torch.Generator().manual_seed(int(seed) + 7222)
    old_loader = DataLoader(Subset(train_ds, old_train), batch_size=int(batch_size), shuffle=True, generator=gen_old)
    new_loader = DataLoader(Subset(train_ds, new_train), batch_size=int(batch_size), shuffle=True, generator=gen_new)
    old_eval = DataLoader(Subset(test_ds, old_test), batch_size=512, shuffle=False)
    new_eval = DataLoader(Subset(test_ds, new_test), batch_size=512, shuffle=False)
    return old_loader, new_loader, old_eval, new_eval


def evaluate_accuracy(model: Any, loader: Any, device: Any) -> float:
    import torch

    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device).float()
            yb = yb.to(device).long()
            pred = model(xb).float().argmax(dim=-1)
            correct += int((pred == yb).sum().item())
            total += int(yb.numel())
    model.train()
    return correct / max(1, total)


def named_trainable(model: Any) -> list[tuple[str, Any]]:
    return [(n, p) for n, p in model.named_parameters() if p.requires_grad]


def grad_dict(named: list[tuple[str, Any]]) -> dict[str, Any]:
    return {n: (p.grad.detach().float().clone() if p.grad is not None else p.detach().float() * 0.0) for n, p in named}


def zeros_like(named: list[tuple[str, Any]]) -> dict[str, Any]:
    return {n: p.detach().float() * 0.0 for n, p in named}


def tensor_norm(values: dict[str, Any]) -> float:
    import torch

    if not values:
        return 0.0
    total = torch.zeros((), device=next(iter(values.values())).device)
    for v in values.values():
        total = total + v.detach().float().pow(2.0).sum()
    return float(total.sqrt().item())


def normalize(values: dict[str, Any], target: float = 1.0) -> dict[str, Any]:
    norm = tensor_norm(values)
    if norm <= 1.0e-12:
        return {k: v * 0.0 for k, v in values.items()}
    return {k: v.detach().float() * (float(target) / norm) for k, v in values.items()}


def random_like(values: dict[str, Any], seed: int) -> dict[str, Any]:
    import torch

    gen = torch.Generator(device=next(iter(values.values())).device).manual_seed(int(seed))
    return {k: torch.randn(v.shape, device=v.device, generator=gen, dtype=torch.float32) for k, v in values.items()}


def apply_velocity(named: list[tuple[str, Any]], velocity: dict[str, Any], scale: float) -> float:
    import torch

    total = 0.0
    with torch.no_grad():
        for name, p in named:
            v = velocity.get(name)
            if v is None:
                continue
            delta = -float(scale) * v.to(device=p.device, dtype=p.dtype)
            p.add_(delta)
            total += float(delta.norm().item()) ** 2
    return math.sqrt(total)


def train_one_batch(model: Any, opt: Any, optimizer_name: str, batch: Any, device: Any, avg_state: dict[str, Any], step: int) -> dict[str, Any]:
    import torch.nn.functional as F

    xb, yb = batch
    xb = xb.to(device).float()
    yb = yb.to(device).long()
    opt.zero_grad(set_to_none=True)
    loss = F.cross_entropy(model(xb).float(), yb)
    loss.backward()
    named = named_trainable(model)
    grad = grad_dict(named)
    v2240.optimizer_step(model, opt, optimizer_name, step, avg_state)
    return grad


def run_collect(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    ensure_out()
    seed_all(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or not str(args.device).startswith("cuda") else "cpu")
    old_loader, new_loader, old_eval, new_eval = mnist_split_loaders(args.train_size, args.held_size, args.batch_size, args.seed)
    x_stats = next(iter(old_loader))[0].to(device).float()
    model_seed = int(args.seed) + 742000 + (0 if args.architecture == "MLP" else 1000)
    model = core.make_model_for_arch(args.architecture, 784, 10, args.hidden, model_seed, device, x_stats)
    opt = core.optimizer_for(args.optimizer, model.parameters(), args.lr, args.weight_decay)
    old_iter = core.cycle_batches(old_loader)
    new_iter = core.cycle_batches(new_loader)
    avg_state: dict[str, Any] = {}
    memory_state: dict[str, Any] = {}
    old_grad_norms: list[float] = []
    new_grad_norms: list[float] = []
    runtime_rows: list[dict[str, Any]] = []
    trace: list[dict[str, Any]] = []
    start = time.time()
    global_step = 0
    for step in range(1, int(args.steps_per_task) + 1):
        global_step += 1
        grad = train_one_batch(model, opt, args.optimizer, next(old_iter), device, avg_state, global_step)
        for name, g in grad.items():
            old = memory_state.get(name)
            memory_state[name] = g if old is None else (1.0 - args.beta_memory) * old.to(device=g.device).float() + args.beta_memory * g
        mem_norm = tensor_norm(memory_state)
        old_grad_norms.append(tensor_norm(grad))
        runtime_rows.append(runtime_row(args, "old", global_step, 1, 1, 0.0, mem_norm))
        if step in {1, max(1, int(args.steps_per_task) // 2), int(args.steps_per_task)}:
            trace.append(trace_row(args, "old", global_step, evaluate_accuracy(model, old_eval, device), evaluate_accuracy(model, new_eval, device), mem_norm, 0.0))
    old_before = evaluate_accuracy(model, old_eval, device)
    new_before = evaluate_accuracy(model, new_eval, device)
    for step in range(1, int(args.steps_per_task) + 1):
        global_step += 1
        grad = train_one_batch(model, opt, args.optimizer, next(new_iter), device, avg_state, global_step)
        new_grad_norms.append(tensor_norm(grad))
        named = named_trainable(model)
        mem_norm = tensor_norm(memory_state)
        if args.variant == "optimizer_alone":
            velocity = zeros_like(named)
        elif args.variant == "memory_signal_flow":
            velocity = memory_state
        elif args.variant == "memory_random_control":
            velocity = normalize(random_like(memory_state, int(args.seed) * 100000 + step), mem_norm)
        elif args.variant == "memory_signflip_control":
            velocity = {k: -v for k, v in memory_state.items()}
        else:
            raise ValueError(f"unknown variant {args.variant}")
        velocity = normalize(velocity, 1.0)
        fu_norm = 0.0
        if args.variant != "optimizer_alone":
            fu_norm = apply_velocity(named, velocity, args.lr * args.rho * args.velocity_scale)
        runtime_rows.append(runtime_row(args, "new", global_step, 1, 1, fu_norm, mem_norm))
        if step in {1, max(1, int(args.steps_per_task) // 2), int(args.steps_per_task)}:
            trace.append(trace_row(args, "new", global_step, evaluate_accuracy(model, old_eval, device), evaluate_accuracy(model, new_eval, device), mem_norm, fu_norm))
    v2240.final_schedule_free_swap(model, args.optimizer, avg_state)
    old_after = evaluate_accuracy(model, old_eval, device)
    new_after = evaluate_accuracy(model, new_eval, device)
    forgetting = old_before - old_after
    final_avg = 0.5 * (old_after + new_after)
    memory_norm = tensor_norm(memory_state)
    memory_snr = memory_norm / max(1.0e-12, statistics.fmean(new_grad_norms) if new_grad_norms else 0.0)
    summary = {
        "run_label": args.label,
        "dataset": "Class_MNIST_0_4_to_5_9",
        "seed": args.seed,
        "architecture": args.architecture,
        "optimizer_family": args.optimizer,
        "variant": args.variant,
        "steps_per_task": args.steps_per_task,
        "hidden": args.hidden,
        "train_size": args.train_size,
        "held_size": args.held_size,
        "old_task_accuracy_before_new_task": old_before,
        "new_task_accuracy_before_new_task": new_before,
        "old_task_accuracy_after_new_task": old_after,
        "new_task_accuracy": new_after,
        "final_average_accuracy": final_avg,
        "average_forgetting": forgetting,
        "relative_forgetting_reduction": "",
        "average_accuracy_delta_vs_base": "",
        "memory_state_norm": memory_norm,
        "memory_state_SNR": memory_snr,
        "old_grad_norm_mean": statistics.fmean(old_grad_norms) if old_grad_norms else 0.0,
        "new_grad_norm_mean": statistics.fmean(new_grad_norms) if new_grad_norms else 0.0,
        "continuous_fu_state_updated_every_step": 1,
        "fu_velocity_emitted_every_step": 1,
        "candidate_action_selection_used_for_runtime": 0,
        "runtime_argmax_candidate_used": 0,
        "runtime_topk_candidate_used": 0,
        "candidate_value_model_used_as_runtime_policy": 0,
        "micro_rct_winner_used_as_runtime_action": 0,
        "path_mpc_discrete_action_sequence_used": 0,
        "elapsed_sec": time.time() - start,
        "status": "completed_v22_42R_s7_row",
    }
    prefix = S7_ROOT / f"v22_42R_s7_{safe_fragment(args.label)}"
    write_rows(Path(str(prefix) + "_summary.csv"), [summary])
    write_rows(Path(str(prefix) + "_trace.csv"), trace)
    write_rows(Path(str(prefix) + "_runtime_trace.csv"), runtime_rows)
    append_exec(
        "train_s7_variant",
        task_id=f"s7_collect_{safe_fragment(args.label)}",
        status="pass",
        gpu=str(args.device),
        files=f"{Path(str(prefix) + '_summary.csv').relative_to(ROOT)}, {Path(str(prefix) + '_trace.csv').relative_to(ROOT)}, {Path(str(prefix) + '_runtime_trace.csv').relative_to(ROOT)}",
        note=f"variant={args.variant}; arch={args.architecture}; seed={args.seed}; steps_per_task={args.steps_per_task}",
    )
    return summary


def runtime_row(args: argparse.Namespace, phase: str, step: int, state_updated: int, emitted: int, fu_norm: float, memory_norm: float) -> dict[str, Any]:
    return {
        "run_label": args.label,
        "dataset": "Class_MNIST_0_4_to_5_9",
        "seed": args.seed,
        "architecture": args.architecture,
        "variant": args.variant,
        "phase": phase,
        "step": step,
        "runtime_policy_type": "continuous_continual_memory_velocity_field",
        "continuous_fu_state_updated": state_updated,
        "fu_velocity_emitted": emitted,
        "runtime_argmax_candidate_used": 0,
        "runtime_topk_candidate_used": 0,
        "candidate_action_selection_used_for_runtime": 0,
        "candidate_value_model_used_as_runtime_policy": 0,
        "micro_rct_winner_used_as_runtime_action": 0,
        "path_mpc_discrete_action_sequence_used": 0,
        "fu_velocity_norm": fu_norm,
        "memory_state_norm": memory_norm,
    }


def trace_row(args: argparse.Namespace, phase: str, step: int, old_acc: float, new_acc: float, memory_norm: float, fu_norm: float) -> dict[str, Any]:
    return {
        "run_label": args.label,
        "dataset": "Class_MNIST_0_4_to_5_9",
        "seed": args.seed,
        "architecture": args.architecture,
        "variant": args.variant,
        "phase": phase,
        "step": step,
        "old_task_accuracy": old_acc,
        "new_task_accuracy": new_acc,
        "memory_state_norm": memory_norm,
        "fu_velocity_norm": fu_norm,
    }


def run_logged(cmd: list[str], task_id: str, gpu: str, timeout: int) -> subprocess.CompletedProcess[str]:
    ensure_out()
    env = os.environ.copy()
    if str(gpu).startswith("cuda:"):
        env["CUDA_VISIBLE_DEVICES"] = str(gpu).split(":", 1)[1]
    started = time.time()
    proc = subprocess.run(cmd, cwd=str(ROOT), env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout, check=False)
    stdout = LOG_ROOT / f"{safe_fragment(task_id)}_stdout.log"
    stderr = LOG_ROOT / f"{safe_fragment(task_id)}_stderr.log"
    stdout.write_text(proc.stdout or "", encoding="utf-8", errors="replace")
    stderr.write_text(proc.stderr or "", encoding="utf-8", errors="replace")
    append_exec(
        " ".join(shlex.quote(x) for x in cmd),
        task_id=task_id,
        status="pass" if proc.returncode == 0 else "fail",
        gpu=gpu,
        files=f"{stdout.relative_to(ROOT)}, {stderr.relative_to(ROOT)}",
        note=f"elapsed_sec={time.time() - started:.3f}",
        exit_code=proc.returncode,
    )
    return proc


def run_full(args: argparse.Namespace) -> None:
    ensure_out()
    specs = []
    gpus = [x.strip() for x in str(args.gpus).split(",") if x.strip()]
    for seed in [int(x) for x in str(args.seeds).split(",") if x.strip()]:
        for arch in [x.strip() for x in str(args.architectures).split(",") if x.strip()]:
            for variant in [x.strip() for x in str(args.variants).split(",") if x.strip()]:
                label = f"{safe_fragment(args.label)}_{safe_fragment(arch)}_s{seed}_{safe_fragment(variant)}"
                specs.append((seed, arch, variant, label, f"cuda:{gpus[len(specs) % max(1, len(gpus))]}"))
    commands = []
    for seed, arch, variant, label, gpu in specs:
        child_device = "cuda:0" if str(gpu).startswith("cuda") else gpu
        cmd = [
            PYTHON,
            str(Path(__file__).relative_to(ROOT)),
            "--stage",
            "collect",
            "--label",
            label,
            "--seed",
            str(seed),
            "--architecture",
            arch,
            "--variant",
            variant,
            "--device",
            child_device,
            "--steps-per-task",
            str(args.steps_per_task),
            "--hidden",
            str(args.hidden),
            "--batch-size",
            str(args.batch_size),
            "--train-size",
            str(args.train_size),
            "--held-size",
            str(args.held_size),
            "--optimizer",
            args.optimizer,
            "--lr",
            str(args.lr),
            "--weight-decay",
            str(args.weight_decay),
            "--rho",
            str(args.rho),
            "--velocity-scale",
            str(args.velocity_scale),
            "--beta-memory",
            str(args.beta_memory),
        ]
        commands.append((cmd, f"s7_full_{label}", gpu))
    append_exec(
        "stage_s7_full",
        task_id=f"s7_full_started_{safe_fragment(args.label)}",
        status="pass",
        gpu=",".join(gpus),
        note=f"rows={len(commands)}; variants={args.variants}; seeds={args.seeds}; arch={args.architectures}",
    )
    failures = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=int(args.workers)) as ex:
        futs = [ex.submit(run_logged, cmd, task_id, gpu, int(args.row_timeout)) for cmd, task_id, gpu in commands]
        for fut in concurrent.futures.as_completed(futs):
            proc = fut.result()
            failures += int(proc.returncode != 0)
    merge(args.label)
    append_exec(
        "stage_s7_full_completed",
        task_id=f"s7_full_completed_{safe_fragment(args.label)}",
        status="pass" if failures == 0 else "fail",
        gpu=",".join(gpus),
        files="results/v22_42R/v22_42R_s7_continual_memory_matrix.csv, results/v22_42R/v22_42R_s7_continual_memory_summary.json",
        note=f"rows={len(commands)}; failures={failures}",
        exit_code=failures,
    )


def merge(label_prefix: str) -> dict[str, Any]:
    rows: list[dict[str, str]] = []
    traces: list[dict[str, str]] = []
    runtimes: list[dict[str, str]] = []
    for path in sorted(S7_ROOT.glob("v22_42R_s7_*_summary.csv")):
        rows.extend([r for r in read_rows(path) if str(r.get("run_label", "")).startswith(label_prefix)])
    for path in sorted(S7_ROOT.glob("v22_42R_s7_*_trace.csv")):
        if path.name.endswith("_runtime_trace.csv"):
            continue
        traces.extend([r for r in read_rows(path) if str(r.get("run_label", "")).startswith(label_prefix)])
    for path in sorted(S7_ROOT.glob("v22_42R_s7_*_runtime_trace.csv")):
        runtimes.extend([r for r in read_rows(path) if str(r.get("run_label", "")).startswith(label_prefix)])
    rows = annotate_relative(rows)
    label_safe = safe_fragment(label_prefix)
    matrix_path = OUT_ROOT / "v22_42R_s7_continual_memory_matrix.csv"
    trace_path = OUT_ROOT / "v22_42R_s7_continual_memory_trace.csv"
    runtime_path = OUT_ROOT / "v22_42R_s7_continual_memory_runtime_trace.csv"
    label_matrix_path = OUT_ROOT / f"v22_42R_s7_continual_memory_{label_safe}_matrix.csv"
    label_trace_path = OUT_ROOT / f"v22_42R_s7_continual_memory_{label_safe}_trace.csv"
    label_runtime_path = OUT_ROOT / f"v22_42R_s7_continual_memory_{label_safe}_runtime_trace.csv"
    write_rows(matrix_path, rows)
    write_rows(trace_path, traces)
    write_rows(runtime_path, runtimes)
    write_rows(label_matrix_path, rows)
    write_rows(label_trace_path, traces)
    write_rows(label_runtime_path, runtimes)
    summary = summarize(rows, runtimes)
    summary["label_prefix"] = label_prefix
    summary_clean = json_clean(summary)
    summary_path = OUT_ROOT / "v22_42R_s7_continual_memory_summary.json"
    label_summary_path = OUT_ROOT / f"v22_42R_s7_continual_memory_{label_safe}_summary.json"
    summary_text = json.dumps(summary_clean, indent=2, ensure_ascii=False, allow_nan=False)
    summary_path.write_text(summary_text, encoding="utf-8")
    label_summary_path.write_text(summary_text, encoding="utf-8")
    append_exec(
        "merge_s7",
        task_id=f"s7_merge_{safe_fragment(label_prefix)}",
        status="pass",
        gpu="n/a",
        files=f"{matrix_path.relative_to(ROOT)}, {summary_path.relative_to(ROOT)}, {label_matrix_path.relative_to(ROOT)}, {label_summary_path.relative_to(ROOT)}",
        note=f"rows={len(rows)}; traces={len(traces)}; runtimes={len(runtimes)}; route={summary.get('route')}",
    )
    return summary_clean


def annotate_relative(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    groups: dict[tuple[str, str], dict[str, dict[str, str]]] = {}
    for r in rows:
        groups.setdefault((r.get("architecture", ""), r.get("seed", "")), {})[r.get("variant", "")] = r
    for vals in groups.values():
        base = vals.get("optimizer_alone")
        base_forget = finite(base.get("average_forgetting")) if base else None
        base_avg = finite(base.get("final_average_accuracy")) if base else None
        for row in vals.values():
            f = finite(row.get("average_forgetting"))
            avg = finite(row.get("final_average_accuracy"))
            row["relative_forgetting_reduction"] = "" if base_forget in {None, 0.0} or f is None else (base_forget - f) / abs(base_forget)
            row["average_accuracy_delta_vs_base"] = "" if base_avg is None or avg is None else avg - base_avg
    return rows


def summarize(rows: list[dict[str, str]], runtimes: list[dict[str, str]]) -> dict[str, Any]:
    groups: dict[tuple[str, str], dict[str, dict[str, str]]] = {}
    for r in rows:
        groups.setdefault((r.get("architecture", ""), r.get("seed", "")), {})[r.get("variant", "")] = r
    rels = []
    signal_beats_base = 0
    signal_beats_controls = 0
    avg_nonworse = 0
    for vals in groups.values():
        base = vals.get("optimizer_alone")
        sig = vals.get("memory_signal_flow")
        rand = vals.get("memory_random_control")
        sign = vals.get("memory_signflip_control")
        if not base or not sig:
            continue
        rel = finite(sig.get("relative_forgetting_reduction"))
        if rel is not None:
            rels.append(rel)
            signal_beats_base += int(rel > 0.0)
        sig_forget = finite(sig.get("average_forgetting"))
        control_forgets = [finite(r.get("average_forgetting")) for r in [rand, sign] if r]
        if sig_forget is not None and len(control_forgets) == 2:
            signal_beats_controls += int(all(c is not None and sig_forget < c for c in control_forgets))
        avg_delta = finite(sig.get("average_accuracy_delta_vs_base"))
        avg_nonworse += int(avg_delta is not None and avg_delta >= -1.0e-12)
    bad_runtime = 0
    for r in runtimes:
        for key in [
            "runtime_argmax_candidate_used",
            "runtime_topk_candidate_used",
            "candidate_action_selection_used_for_runtime",
            "candidate_value_model_used_as_runtime_policy",
            "micro_rct_winner_used_as_runtime_action",
            "path_mpc_discrete_action_sequence_used",
        ]:
            bad_runtime += int(str(r.get(key, "0")) not in {"0", ""})
        bad_runtime += int(str(r.get("continuous_fu_state_updated", "1")) != "1")
        bad_runtime += int(str(r.get("fu_velocity_emitted", "1")) != "1")
    mean_rel = statistics.fmean(rels) if rels else math.nan
    route = "S7DiagnosticOnly"
    if rels and mean_rel >= 0.05 and signal_beats_controls >= max(1, len(groups) // 2) and avg_nonworse >= max(1, len(groups) // 2):
        route = "R12-ContinualMemoryOpened_DiagnosticOnly"
    return {
        "generated_at": now_sg(),
        "rows": len(rows),
        "groups": len(groups),
        "memory_signal_beats_base_groups": signal_beats_base,
        "memory_signal_beats_controls_groups": signal_beats_controls,
        "final_average_accuracy_nonworse_groups": avg_nonworse,
        "relative_forgetting_reduction_mean": mean_rel,
        "runtime_bad_flag_count": bad_runtime,
        "candidate_action_regression_pass": int(bad_runtime == 0),
        "route": route,
        "promotion_allowed": 0,
        "note": "S7 is diagnostic/pressure only in v22.42R and cannot bypass S4/S1/S6 official gates.",
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", default="full", choices=["collect", "full", "merge"])
    p.add_argument("--label", default="s7_class_mnist")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--architecture", default="MLP", choices=["MLP", "DGKAN_DCHE"])
    p.add_argument("--architectures", default="MLP,DGKAN_DCHE")
    p.add_argument("--variant", default="memory_signal_flow", choices=sorted(VARIANTS))
    p.add_argument("--variants", default="optimizer_alone,memory_signal_flow,memory_random_control,memory_signflip_control")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--gpus", default="0,1,2,3")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--row-timeout", type=int, default=1800)
    p.add_argument("--steps-per-task", type=int, default=200)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--train-size", type=int, default=512)
    p.add_argument("--held-size", type=int, default=512)
    p.add_argument("--optimizer", default="AdamW")
    p.add_argument("--lr", type=float, default=1.0e-3)
    p.add_argument("--weight-decay", type=float, default=1.0e-4)
    p.add_argument("--rho", type=float, default=0.05)
    p.add_argument("--velocity-scale", type=float, default=0.02)
    p.add_argument("--beta-memory", type=float, default=0.02)
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.stage == "collect":
        run_collect(args)
    elif args.stage == "full":
        run_full(args)
    elif args.stage == "merge":
        merge(args.label)


if __name__ == "__main__":
    main()
