#!/usr/bin/env python3
"""v22.42R S5 temporal slow-signal diagnostic.

Diagnostic only.  This runner checks whether a delayed modular-addition regime
exists for continuous slow-signal FU.  It does not promote official routes and
does not use candidate-action runtime selection.
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
S5_ROOT = OUT_ROOT / "s5"
LOG_ROOT = OUT_ROOT / "logs"

VARIANTS = {
    "optimizer_alone",
    "slow_signal_flow",
    "slow_random_control",
    "slow_signflip_control",
}


def ensure_out() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    S5_ROOT.mkdir(parents=True, exist_ok=True)
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


def modular_loaders(p: int, train_fraction: float, batch_size: int, seed: int, device: Any) -> tuple[Any, Any, Any, int, int, Any]:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    xs = []
    ys = []
    for a in range(int(p)):
        for b in range(int(p)):
            x = torch.zeros(2 * int(p), dtype=torch.float32)
            x[a] = 1.0
            x[int(p) + b] = 1.0
            xs.append(x)
            ys.append((a + b) % int(p))
    x_all = torch.stack(xs)
    y_all = torch.tensor(ys, dtype=torch.long)
    gen = torch.Generator().manual_seed(int(seed) + 4205)
    perm = torch.randperm(len(y_all), generator=gen)
    n_train = max(1, int(round(float(train_fraction) * len(y_all))))
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]
    train_x = x_all[train_idx]
    train_y = y_all[train_idx]
    test_x = x_all[test_idx]
    test_y = y_all[test_idx]
    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=int(batch_size), shuffle=True, generator=gen)
    eval_train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=512, shuffle=False)
    test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=512, shuffle=False)
    return train_loader, eval_train_loader, test_loader, 2 * int(p), int(p), train_x.to(device)


def evaluate(model: Any, loader: Any, device: Any) -> dict[str, float]:
    import torch
    import torch.nn.functional as F

    model.eval()
    total = 0
    correct = 0
    nll_sum = 0.0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device).float()
            y = y.to(device).long()
            logits = model(x).float()
            loss = F.cross_entropy(logits, y, reduction="sum")
            nll_sum += float(loss.item())
            pred = logits.argmax(dim=-1)
            correct += int((pred == y).sum().item())
            total += int(y.numel())
    model.train()
    return {"NLL": nll_sum / max(1, total), "accuracy": correct / max(1, total)}


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
    scale = float(target) / norm
    return {k: v.detach().float() * scale for k, v in values.items()}


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


def run_collect(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    ensure_out()
    seed_all(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or not str(args.device).startswith("cuda") else "cpu")
    train_loader, eval_train_loader, test_loader, input_dim, output_dim, x_stats = modular_loaders(
        args.modulus, args.train_fraction, args.batch_size, args.seed, device
    )
    model_seed = int(args.seed) + 425000 + (0 if args.architecture == "MLP" else 1000)
    model = core.make_model_for_arch(args.architecture, input_dim, output_dim, args.hidden, model_seed, device, x_stats)
    opt = core.optimizer_for(args.optimizer, model.parameters(), args.lr, args.weight_decay)
    train_iter = core.cycle_batches(train_loader)
    avg_state: dict[str, Any] = {}
    slow_state: dict[str, Any] = {}
    fast_energy: list[float] = []
    slow_energy: list[float] = []
    trace: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    best_train_acc = 0.0
    test_at_train_cross = ""
    train_cross_step = ""
    test_cross_step = ""
    start = time.time()
    eval_steps = {1, 20, 60, 200, 400, 800, 1200, 1600, int(args.steps)}
    for step in range(1, int(args.steps) + 1):
        x, y = next(train_iter)
        x = x.to(device).float()
        y = y.to(device).long()
        opt.zero_grad(set_to_none=True)
        logits = model(x).float()
        loss = F.cross_entropy(logits, y)
        loss.backward()
        named = named_trainable(model)
        grad = grad_dict(named)
        for name, g in grad.items():
            old = slow_state.get(name)
            slow_state[name] = g if old is None else (1.0 - args.beta_slow) * old.to(device=g.device).float() + args.beta_slow * g
        v2240.optimizer_step(model, opt, args.optimizer, step, avg_state)
        slow_norm = tensor_norm(slow_state)
        grad_norm = tensor_norm(grad)
        if args.variant == "optimizer_alone":
            velocity = zeros_like(named)
        elif args.variant == "slow_signal_flow":
            velocity = slow_state
        elif args.variant == "slow_random_control":
            velocity = normalize(random_like(slow_state, int(args.seed) * 100000 + step), slow_norm)
        elif args.variant == "slow_signflip_control":
            velocity = {k: -v for k, v in slow_state.items()}
        else:
            raise ValueError(f"unknown variant {args.variant}")
        velocity = normalize(velocity, 1.0)
        fu_norm = 0.0
        if args.variant != "optimizer_alone":
            fu_norm = apply_velocity(named, velocity, args.lr * args.rho * args.velocity_scale)
        fast_energy.append(grad_norm)
        slow_energy.append(slow_norm)
        runtime_rows.append(
            {
                "run_label": args.label,
                "dataset": "modular_addition",
                "seed": args.seed,
                "architecture": args.architecture,
                "variant": args.variant,
                "step": step,
                "runtime_policy_type": "continuous_temporal_slow_signal_velocity_field",
                "continuous_fu_state_updated": 1,
                "fu_velocity_emitted": 1,
                "runtime_argmax_candidate_used": 0,
                "runtime_topk_candidate_used": 0,
                "candidate_action_selection_used_for_runtime": 0,
                "candidate_value_model_used_as_runtime_policy": 0,
                "micro_rct_winner_used_as_runtime_action": 0,
                "path_mpc_discrete_action_sequence_used": 0,
                "fu_velocity_norm": fu_norm,
            }
        )
        if step in eval_steps:
            tr = evaluate(model, eval_train_loader, device)
            te = evaluate(model, test_loader, device)
            best_train_acc = max(best_train_acc, tr["accuracy"])
            if train_cross_step == "" and tr["accuracy"] >= args.train_threshold:
                train_cross_step = step
                test_at_train_cross = te["accuracy"]
            if test_cross_step == "" and te["accuracy"] >= args.test_threshold:
                test_cross_step = step
            trace.append(
                {
                    "run_label": args.label,
                    "dataset": "modular_addition",
                    "seed": args.seed,
                    "architecture": args.architecture,
                    "variant": args.variant,
                    "step": step,
                    "train_NLL": tr["NLL"],
                    "train_accuracy": tr["accuracy"],
                    "test_NLL": te["NLL"],
                    "test_accuracy": te["accuracy"],
                    "slow_signal_energy": slow_norm,
                    "fast_signal_energy": grad_norm,
                    "slow_fast_ratio": slow_norm / max(1.0e-12, grad_norm),
                }
            )
    v2240.final_schedule_free_swap(model, args.optimizer, avg_state)
    final_train = evaluate(model, eval_train_loader, device)
    final_test = evaluate(model, test_loader, device)
    if test_at_train_cross == "":
        test_at_train_cross = final_test["accuracy"]
    grokking_delay = ""
    if train_cross_step != "" and test_cross_step != "":
        grokking_delay = max(0, int(test_cross_step) - int(train_cross_step))
    migration = final_test["accuracy"] - float(test_at_train_cross)
    summary = {
        "run_label": args.label,
        "dataset": "modular_addition",
        "modulus": args.modulus,
        "train_fraction": args.train_fraction,
        "seed": args.seed,
        "architecture": args.architecture,
        "optimizer_family": args.optimizer,
        "variant": args.variant,
        "steps": args.steps,
        "hidden": args.hidden,
        "final_train_accuracy": final_train["accuracy"],
        "final_test_accuracy": final_test["accuracy"],
        "final_train_NLL": final_train["NLL"],
        "final_test_NLL": final_test["NLL"],
        "train_cross_step": train_cross_step,
        "test_cross_step": test_cross_step,
        "grokking_delay": grokking_delay,
        "test_accuracy_lift_after_delay": migration,
        "slow_signal_energy": statistics.fmean(slow_energy) if slow_energy else 0.0,
        "fast_signal_energy": statistics.fmean(fast_energy) if fast_energy else 0.0,
        "slow_fast_ratio": (statistics.fmean(slow_energy) / max(1.0e-12, statistics.fmean(fast_energy))) if slow_energy and fast_energy else 0.0,
        "reservoir_to_signal_migration_index": migration,
        "continuous_fu_state_updated_every_step": 1,
        "fu_velocity_emitted_every_step": 1,
        "candidate_action_selection_used_for_runtime": 0,
        "runtime_argmax_candidate_used": 0,
        "runtime_topk_candidate_used": 0,
        "candidate_value_model_used_as_runtime_policy": 0,
        "micro_rct_winner_used_as_runtime_action": 0,
        "path_mpc_discrete_action_sequence_used": 0,
        "elapsed_sec": time.time() - start,
        "status": "completed_v22_42R_s5_row",
    }
    prefix = S5_ROOT / f"v22_42R_s5_{safe_fragment(args.label)}"
    write_rows(Path(str(prefix) + "_summary.csv"), [summary])
    write_rows(Path(str(prefix) + "_trace.csv"), trace)
    write_rows(Path(str(prefix) + "_runtime_trace.csv"), runtime_rows)
    append_exec(
        "train_s5_variant",
        task_id=f"s5_collect_{safe_fragment(args.label)}",
        status="pass",
        gpu=str(args.device),
        files=f"{Path(str(prefix) + '_summary.csv').relative_to(ROOT)}, {Path(str(prefix) + '_trace.csv').relative_to(ROOT)}, {Path(str(prefix) + '_runtime_trace.csv').relative_to(ROOT)}",
        note=f"variant={args.variant}; arch={args.architecture}; seed={args.seed}; steps={args.steps}",
    )
    return summary


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
            "--modulus",
            str(args.modulus),
            "--train-fraction",
            str(args.train_fraction),
            "--steps",
            str(args.steps),
            "--hidden",
            str(args.hidden),
            "--batch-size",
            str(args.batch_size),
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
            "--beta-slow",
            str(args.beta_slow),
            "--train-threshold",
            str(args.train_threshold),
            "--test-threshold",
            str(args.test_threshold),
        ]
        commands.append((cmd, f"s5_full_{label}", gpu))
    append_exec(
        "stage_s5_full",
        task_id=f"s5_full_started_{safe_fragment(args.label)}",
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
        "stage_s5_full_completed",
        task_id=f"s5_full_completed_{safe_fragment(args.label)}",
        status="pass" if failures == 0 else "fail",
        gpu=",".join(gpus),
        files="results/v22_42R/v22_42R_s5_temporal_slow_signal_matrix.csv, results/v22_42R/v22_42R_s5_temporal_slow_signal_summary.json",
        note=f"rows={len(commands)}; failures={failures}",
        exit_code=failures,
    )


def merge(label_prefix: str) -> dict[str, Any]:
    rows: list[dict[str, str]] = []
    traces: list[dict[str, str]] = []
    runtimes: list[dict[str, str]] = []
    for path in sorted(S5_ROOT.glob("v22_42R_s5_*_summary.csv")):
        rows.extend([r for r in read_rows(path) if str(r.get("run_label", "")).startswith(label_prefix)])
    for path in sorted(S5_ROOT.glob("v22_42R_s5_*_trace.csv")):
        if path.name.endswith("_runtime_trace.csv"):
            continue
        traces.extend([r for r in read_rows(path) if str(r.get("run_label", "")).startswith(label_prefix)])
    for path in sorted(S5_ROOT.glob("v22_42R_s5_*_runtime_trace.csv")):
        runtimes.extend([r for r in read_rows(path) if str(r.get("run_label", "")).startswith(label_prefix)])
    label_safe = safe_fragment(label_prefix)
    matrix_path = OUT_ROOT / "v22_42R_s5_temporal_slow_signal_matrix.csv"
    trace_path = OUT_ROOT / "v22_42R_s5_temporal_slow_signal_trace.csv"
    runtime_path = OUT_ROOT / "v22_42R_s5_temporal_runtime_trace.csv"
    label_matrix_path = OUT_ROOT / f"v22_42R_s5_temporal_slow_signal_{label_safe}_matrix.csv"
    label_trace_path = OUT_ROOT / f"v22_42R_s5_temporal_slow_signal_{label_safe}_trace.csv"
    label_runtime_path = OUT_ROOT / f"v22_42R_s5_temporal_slow_signal_{label_safe}_runtime_trace.csv"
    write_rows(matrix_path, rows)
    write_rows(trace_path, traces)
    write_rows(runtime_path, runtimes)
    write_rows(label_matrix_path, rows)
    write_rows(label_trace_path, traces)
    write_rows(label_runtime_path, runtimes)
    summary = summarize(rows, runtimes)
    summary["label_prefix"] = label_prefix
    summary_clean = json_clean(summary)
    summary_path = OUT_ROOT / "v22_42R_s5_temporal_slow_signal_summary.json"
    label_summary_path = OUT_ROOT / f"v22_42R_s5_temporal_slow_signal_{label_safe}_summary.json"
    summary_text = json.dumps(summary_clean, indent=2, ensure_ascii=False, allow_nan=False)
    summary_path.write_text(summary_text, encoding="utf-8")
    label_summary_path.write_text(summary_text, encoding="utf-8")
    append_exec(
        "merge_s5",
        task_id=f"s5_merge_{safe_fragment(label_prefix)}",
        status="pass",
        gpu="n/a",
        files=(
            f"{matrix_path.relative_to(ROOT)}, {summary_path.relative_to(ROOT)}, "
            f"{label_matrix_path.relative_to(ROOT)}, {label_summary_path.relative_to(ROOT)}"
        ),
        note=f"rows={len(rows)}; traces={len(traces)}; runtimes={len(runtimes)}; route={summary.get('route')}",
    )
    return summary_clean


def summarize(rows: list[dict[str, str]], runtimes: list[dict[str, str]]) -> dict[str, Any]:
    groups: dict[tuple[str, str], dict[str, dict[str, str]]] = {}
    for r in rows:
        groups.setdefault((r.get("architecture", ""), r.get("seed", "")), {})[r.get("variant", "")] = r
    slow_wins = 0
    delay_reductions = []
    matched_fail = 0
    invalid_no_delay = 0
    train_cross_no_test_cross = 0
    any_test_cross = 0
    for key, vals in groups.items():
        base = vals.get("optimizer_alone")
        slow = vals.get("slow_signal_flow")
        rand = vals.get("slow_random_control")
        sign = vals.get("slow_signflip_control")
        if not base or not slow:
            continue
        bdelay = finite(base.get("grokking_delay"))
        sdelay = finite(slow.get("grokking_delay"))
        if bdelay is None or sdelay is None:
            invalid_no_delay += 1
        elif bdelay > 0:
            delay_reductions.append((bdelay - sdelay) / bdelay)
        train_cross_no_test_cross += int(
            finite(base.get("train_cross_step")) is not None
            and finite(base.get("test_cross_step")) is None
            and finite(slow.get("train_cross_step")) is not None
            and finite(slow.get("test_cross_step")) is None
        )
        any_test_cross += int(finite(base.get("test_cross_step")) is not None or finite(slow.get("test_cross_step")) is not None)
        slow_acc = finite(slow.get("final_test_accuracy"), 0.0) or 0.0
        slow_wins += int(slow_acc > (finite(base.get("final_test_accuracy"), 0.0) or 0.0))
        controls = [rand, sign]
        matched_fail += int(all((finite(c.get("final_test_accuracy"), -1.0) if c else -1.0) < slow_acc for c in controls))
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
    mean_reduction = statistics.fmean(delay_reductions) if delay_reductions else math.nan
    route = "S5DiagnosticOnly"
    if invalid_no_delay >= max(1, len(groups) // 2):
        route = "GrokkingTaskInvalidNoDelay"
    elif delay_reductions and mean_reduction >= 0.25 and matched_fail >= max(1, len(groups) // 2):
        route = "R13-GrokkingReservoirMigrationOpened_DiagnosticOnly"
    return {
        "generated_at": now_sg(),
        "rows": len(rows),
        "groups": len(groups),
        "slow_signal_final_test_acc_beats_base_groups": slow_wins,
        "matched_slow_controls_fail_groups": matched_fail,
        "invalid_no_delay_groups": invalid_no_delay,
        "train_cross_no_test_cross_groups": train_cross_no_test_cross,
        "any_test_cross_groups": any_test_cross,
        "grokking_delay_reduction_mean": mean_reduction,
        "runtime_bad_flag_count": bad_runtime,
        "candidate_action_regression_pass": int(bad_runtime == 0),
        "route": route,
        "promotion_allowed": 0,
        "note": "S5 is diagnostic/pressure only in v22.42R and cannot bypass S4/S1/S6 official gates.",
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", default="full", choices=["collect", "full", "merge"])
    p.add_argument("--label", default="s5_modadd")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--architecture", default="MLP", choices=["MLP", "DGKAN_DCHE"])
    p.add_argument("--architectures", default="MLP,DGKAN_DCHE")
    p.add_argument("--variant", default="slow_signal_flow", choices=sorted(VARIANTS))
    p.add_argument("--variants", default="optimizer_alone,slow_signal_flow,slow_random_control,slow_signflip_control")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--gpus", default="0,1,2,3")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--row-timeout", type=int, default=1800)
    p.add_argument("--modulus", type=int, default=17)
    p.add_argument("--train-fraction", type=float, default=0.40)
    p.add_argument("--steps", type=int, default=1600)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--optimizer", default="Schedule-Free AdamW")
    p.add_argument("--lr", type=float, default=1.0e-3)
    p.add_argument("--weight-decay", type=float, default=1.0e-3)
    p.add_argument("--rho", type=float, default=0.05)
    p.add_argument("--velocity-scale", type=float, default=0.02)
    p.add_argument("--beta-slow", type=float, default=0.01)
    p.add_argument("--train-threshold", type=float, default=0.90)
    p.add_argument("--test-threshold", type=float, default=0.80)
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
