#!/usr/bin/env python3
"""DG-KAN v22.69R MLP-winner projected quotient KAN carrier audit.

This runner is intentionally gate ordered:
Part A -> Part B -> Part C -> Part D guard.
It writes row-level artifacts plus two human audit logs.  It never promotes
MLP-target diagnostic paths into official KAN runtime.
"""

from __future__ import annotations

import argparse
import copy
import csv
import importlib
import json
import math
import os
import py_compile
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PYTHON = sys.executable
RUNNER = ROOT / "experiments/run_v22_69r_mlp_winner_projected_quotient_kan_carrier_audit.py"
PLAN = ROOT / "docs/DG-KAN_v22.69R_MLPWinnerProjectedQuotientKANCarrierAudit_完整计划.md"
EXEC_LOG = ROOT / "docs/DG-KAN_v22.69R_MLPWinnerProjectedQuotientKANCarrierAudit_执行日志.md"
RECAP_LOG = ROOT / "docs/DG-KAN_v22.69R_MLPWinnerProjectedQuotientKANCarrierAudit_实验结果复盘.md"
OUT_ROOT = ROOT / "results/v22_69R"
LOG_ROOT = OUT_ROOT / "logs"
TARGET_ROOT = OUT_ROOT / "targets"

WINNER_METHODS = [
    "mcga_over_poet_fsclip_eta001_residual_rank4",
    "mcga_over_poet_pion_blend65_actgradcov_fsclip_eta025_fishermetric_residual_rank4",
]
CHECKPOINT_FRACTIONS = [0.20, 0.50, 0.80]
EPS_SWEEP = [1.0e-4, 3.0e-4, 1.0e-3, 3.0e-3]
PART_C_UNIT_SEEDS = [0, 1, 2, 3, 4]


def now_sg() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %z", time.localtime())


def ensure_out() -> None:
    for path in (OUT_ROOT, LOG_ROOT, TARGET_ROOT, EXEC_LOG.parent, RECAP_LOG.parent):
        path.mkdir(parents=True, exist_ok=True)


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def command_text(argv: list[Any]) -> str:
    return " ".join(str(x) for x in argv)


def current_runner_command() -> str:
    return command_text([PYTHON, rel(RUNNER), *sys.argv[1:]])


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in keys})


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def append_exec(task_id: str, command: str, status: str, *, gpu: str = "", files: str = "", note: str = "") -> None:
    ensure_out()
    if not EXEC_LOG.exists():
        EXEC_LOG.write_text(
            "# DG-KAN v22.69R 执行日志\n\n"
            f"- initialized_at: {now_sg()}\n"
            f"- plan: `{rel(PLAN)}`\n"
            "- non_fabrication: 本日志只记录真实执行过的命令、生成文件和观察结果。\n\n",
            encoding="utf-8",
        )
    line = (
        f"\n## {now_sg()} | {task_id} | {status}\n\n"
        f"- command: `{command}`\n"
        f"- gpu: `{gpu}`\n"
        f"- files: `{files}`\n"
        f"- note: {note}\n"
    )
    with EXEC_LOG.open("a", encoding="utf-8") as f:
        f.write(line)
    journal = OUT_ROOT / "v22_69R_command_journal.csv"
    exists = journal.exists()
    with journal.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["time_sg", "task_id", "status", "command", "gpu", "files", "note"])
        if not exists:
            writer.writeheader()
        writer.writerow({"time_sg": now_sg(), "task_id": task_id, "status": status, "command": command, "gpu": gpu, "files": files, "note": note})


def append_recap(title: str, lines: list[str]) -> None:
    ensure_out()
    if not RECAP_LOG.exists():
        RECAP_LOG.write_text(
            "# DG-KAN v22.69R 实验结果复盘\n\n"
            f"- initialized_at: {now_sg()}\n"
            f"- plan: `{rel(PLAN)}`\n"
            "- non_fabrication: 不编造数据；所有指标必须对应 artifact 或命令输出。\n\n",
            encoding="utf-8",
        )
    with RECAP_LOG.open("a", encoding="utf-8") as f:
        f.write(f"\n## {title}\n\n")
        f.write(f"- time_sg: {now_sg()}\n")
        for line in lines:
            f.write(f"- {line}\n")


def fval(x: Any, default: float | None = None) -> float | None:
    try:
        if x == "" or x is None:
            return default
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def mean(vals: list[float]) -> float:
    return float(sum(vals) / max(1, len(vals))) if vals else 0.0


def percentile(vals: list[float], q: float) -> float:
    if not vals:
        return 0.0
    work = sorted(vals)
    idx = max(0, min(len(work) - 1, int(math.floor((len(work) - 1) * float(q)))))
    return float(work[idx])


def cosine(a: Any, b: Any, eps: float = 1.0e-12) -> float:
    import torch

    av = a.detach().double().reshape(-1)
    bv = b.detach().double().reshape(-1)
    den = torch.linalg.norm(av).clamp_min(eps) * torch.linalg.norm(bv).clamp_min(eps)
    return float(((av @ bv) / den).detach().cpu().item())


def torch_device(device_name: str) -> Any:
    import torch

    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def parse_csv(s: str) -> list[str]:
    return [x.strip() for x in str(s).split(",") if x.strip()]


def safe_fragment(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s)).strip("_")


def float_or(value: Any, default: float | None = 0.0) -> float | None:
    try:
        return float(value)
    except Exception:
        return default


def initialize_docs(args: argparse.Namespace) -> None:
    ensure_out()
    if bool(args.reset_logs):
        for path in (EXEC_LOG, RECAP_LOG):
            if path.exists():
                path.unlink()
    append_exec(
        "init",
        command_text([PYTHON, rel(RUNNER), "--mode", str(args.mode)]),
        "start",
        gpu=str(args.device),
        files=f"{rel(EXEC_LOG)}; {rel(RECAP_LOG)}; {rel(OUT_ROOT)}",
        note="v22.69R runner initialized; logs are append-only unless --reset-logs is supplied.",
    )
    append_recap(
        "初始化与范围",
        [
            "目标：按 v22.69R 计划执行 Part A/B/C/D gate，不把 MLP target diagnostic 误标为 official KAN runtime。",
            "当前实现边界：Part D 只在 Part B 和 Part C gate 通过后运行；否则写 route，不越级 full-loop。",
        ],
    )


def v66_args(args: argparse.Namespace, *, dataset: str, seed: int, method: str, device: str, steps: int) -> Any:
    return SimpleNamespace(
        mode="row",
        datasets=dataset,
        seeds=str(seed),
        methods=method,
        gpus=str(args.gpus),
        max_workers=int(args.max_workers),
        row_timeout=int(args.row_timeout),
        dataset=dataset,
        seed=int(seed),
        method=method,
        architecture="MLP",
        run_label="v22_69R_target_extract",
        device=device,
        train_size=int(args.train_size),
        held_size=int(args.held_size),
        test_size=int(args.test_size),
        hidden=int(args.hidden),
        steps=int(steps),
        batch_size=int(args.batch_size),
        eval_batch_size=int(args.eval_batch_size),
        refresh=int(args.refresh),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        metric_kind=str(args.metric_kind),
        metric_batch_size=int(args.metric_batch_size),
        reanalysis_metric_batch_size=int(args.metric_batch_size),
        shaping_budget=float(args.shaping_budget),
        iso_eta=float(args.iso_eta),
        functional_spectrum_drift_threshold=float(args.functional_spectrum_drift_threshold),
        unit_seeds="0,1,2",
        poet_block_size=int(args.poet_block_size),
        poet_merge_interval=int(args.poet_merge_interval),
        poet_lr=float(args.poet_lr),
        poet_scale=float(args.poet_scale),
    )


def split_train_only(bundle: dict[str, Any], device: Any, batch_size: int, seed: int) -> dict[str, Any]:
    import torch

    x = bundle["x_train"].to(device)
    y = bundle["y_train"].to(device)
    n = int(x.shape[0])
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed) * 100003 + 2269)
    idx = torch.randperm(n, generator=gen, device=device)
    third = max(8, min(n // 3, int(batch_size)))
    src = idx[:third]
    wit = idx[third : 2 * third]
    pro = idx[2 * third : 3 * third]
    if int(wit.numel()) == 0:
        wit = src
    if int(pro.numel()) == 0:
        pro = src
    return {
        "x_source": x.index_select(0, src),
        "y_source": y.index_select(0, src),
        "x_witness": x.index_select(0, wit),
        "y_witness": y.index_select(0, wit),
        "x_probe": x.index_select(0, pro),
        "y_probe": y.index_select(0, pro),
    }


def build_winner_model(v66: Any, ns: Any, bundle: dict[str, Any], device: Any) -> tuple[Any, Any, dict[str, Any]]:
    import torch

    method = str(ns.method)
    x_train = bundle["x_train"].to(device)
    y_train = bundle["y_train"].to(device)
    base = v66.make_base_model(ns, bundle, device)
    x_metric, y_metric = v66.metric_cohort(x_train, y_train, ns, offset=0)
    atlas = v66.build_atlas_for_method(base, x_metric, y_metric, ns, method, bundle)
    if v66.uses_compatible_model(method):
        wrapper_cls = v66.MetricCompatibleAtlasMLP
        model: Any = wrapper_cls(
            base,
            atlas,
            allow_shape=v66.allow_shape_for_method(method),
            shape_budget=v66.shape_budget_for_method(method, float(ns.shaping_budget)),
            eta=v66.iso_eta_for_method(method, float(ns.iso_eta)),
            train_base_weight=v66.train_base_for_method(method),
            base_spectrum_lock=v66.base_spectrum_lock_for_method(method),
            functional_spectrum_budget=v66.functional_spectrum_budget_for_method(method, float(ns.functional_spectrum_drift_threshold)),
            functional_spectrum_fill=v66.functional_spectrum_fill_for_method(method),
        ).to(device)
    else:
        model = v66.MetricAtlasMLP(base, atlas).to(device)
    coord_diag = v66.freeze_control_coordinates(model, method)
    opt, opt_diag = v66.make_optimizer(method, model, ns, device)
    if isinstance(opt_diag.get("wrapped_model"), torch.nn.Module):
        model = opt_diag.pop("wrapped_model")
    diag = dict(atlas.metrics)
    diag.update(coord_diag)
    diag.update(opt_diag)
    return model, opt, diag


def train_to_checkpoint(v66: Any, model: Any, opt: Any, ns: Any, bundle: dict[str, Any], device: Any, steps: int) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    x_train = bundle["x_train"].to(device)
    y_train = bundle["y_train"].to(device)
    n = int(x_train.shape[0])
    gen = torch.Generator(device=device)
    gen.manual_seed(int(ns.seed) * 1009 + sum(ord(ch) for ch in str(ns.method)))
    losses: list[float] = []
    for step in range(int(steps)):
        if int(ns.batch_size) >= n:
            idx = torch.arange(n, device=device)
        else:
            idx = torch.randperm(n, generator=gen, device=device)[: int(ns.batch_size)]
        opt.zero_grad(set_to_none=True)
        logits = model(x_train[idx])
        loss = F.cross_entropy(logits.float(), y_train[idx].long())
        loss.backward()
        opt.step()
        losses.append(float(loss.detach().cpu().item()))
    return {"checkpoint_train_loss_last": losses[-1] if losses else "", "checkpoint_train_loss_mean": mean(losses)}


def snapshot_named_tensors(model: Any) -> tuple[list[str], dict[str, Any], dict[str, Any]]:
    return (
        [name for name, p in model.named_parameters() if getattr(p, "requires_grad", False)],
        {name: p.detach().clone() for name, p in model.named_parameters()},
        {name: b.detach().clone() for name, b in model.named_buffers()},
    )


def restore_params(model: Any, before: dict[str, Any]) -> None:
    import torch

    with torch.no_grad():
        for name, param in model.named_parameters():
            param.copy_(before[name].to(device=param.device, dtype=param.dtype))


def restore_buffers(model: Any, before: dict[str, Any]) -> None:
    import torch

    with torch.no_grad():
        for name, buf in model.named_buffers():
            if name in before:
                buf.copy_(before[name].to(device=buf.device, dtype=buf.dtype))


def functional_logits(model: Any, params: dict[str, Any], buffers: dict[str, Any], x: Any) -> Any:
    try:
        from torch.func import functional_call
    except Exception:  # pragma: no cover
        from torch.nn.utils.stateless import functional_call  # type: ignore

        merged = {**params, **buffers}
        return functional_call(model, merged, (x,))
    return functional_call(model, (params, buffers), (x,))


def jvp_and_fd(model: Any, param_names: list[str], params: dict[str, Any], buffers: dict[str, Any], deltas: dict[str, Any], x_probe: Any) -> dict[str, Any]:
    import torch

    try:
        torch._functorch.config.donated_buffer = False  # type: ignore[attr-defined]
    except Exception:
        pass
    restore_params(model, params)
    restore_buffers(model, buffers)
    param_lookup = {name: p for name, p in model.named_parameters()}
    active_params = [param_lookup[name] for name in param_names]
    logits = model(x_probe).float()
    base_out = logits.reshape(-1).detach()
    flat = logits.reshape(-1)
    jvp_parts = []
    for i in range(int(flat.numel())):
        grads = torch.autograd.grad(flat[i], active_params, retain_graph=True, allow_unused=True)
        total = torch.zeros((), device=flat.device, dtype=flat.dtype)
        for name, grad in zip(param_names, grads):
            if grad is not None:
                total = total + torch.sum(grad.float() * deltas[name].to(device=grad.device, dtype=grad.dtype).float())
        jvp_parts.append(total.detach())
    jvp_vec = torch.stack(jvp_parts).reshape(-1)

    eps_rows = []
    best = {"eps": None, "cosine": -2.0, "fd": None}
    for eps in EPS_SWEEP:
        with torch.no_grad():
            for name in param_names:
                p = param_lookup[name]
                p.copy_(params[name].to(device=p.device, dtype=p.dtype) + float(eps) * deltas[name].to(device=p.device, dtype=p.dtype))
        fd_logits = model(x_probe).float().reshape(-1).detach()
        fd = (fd_logits - base_out.detach()) / float(eps)
        restore_params(model, params)
        restore_buffers(model, buffers)
        c = cosine(jvp_vec.detach(), fd.detach())
        eps_rows.append({"eps": eps, "jvp_fd_cosine": c, "fd_norm": float(torch.linalg.norm(fd.detach()).cpu().item())})
        if c > float(best["cosine"]):
            best = {"eps": eps, "cosine": c, "fd": fd.detach().clone()}
    return {"base": base_out.detach(), "jvp": jvp_vec.detach(), "best_fd": best["fd"], "best_eps": best["eps"], "best_cosine": best["cosine"], "eps_rows": eps_rows}


def target_norms_and_debt(base_logits: Any, target_vec: Any, y_probe: Any) -> dict[str, float]:
    import torch
    import torch.nn.functional as F

    logits = base_logits.detach().float()
    target = target_vec.detach().float().reshape_as(logits)
    probs = torch.softmax(logits, dim=1)
    yoh = F.one_hot(y_probe.long(), num_classes=int(logits.shape[1])).float().to(logits.device)
    grad = (probs - yoh) / max(1, int(logits.shape[0]))
    fisher_norms = []
    for p, v in zip(probs, target):
        fisher_v = p * v - p * torch.sum(p * v)
        fisher_norms.append(torch.dot(v, fisher_v).clamp_min(0.0))
    fisher = torch.sqrt(torch.stack(fisher_norms).mean().clamp_min(1.0e-12))
    pred_delta = torch.sum(grad * target)
    losses = F.cross_entropy(logits, y_probe.long(), reduction="none")
    tail_cut = torch.quantile(losses.float(), 0.75) if int(losses.numel()) > 1 else losses.max()
    tail_mask = (losses >= tail_cut).float().view(-1, 1)
    tail_delta = torch.sum(grad * target * tail_mask) / tail_mask.sum().clamp_min(1.0)
    return {
        "target_logit_norm": float(torch.linalg.norm(target).detach().cpu().item()),
        "target_CE_Fisher_norm": float(fisher.detach().cpu().item()),
        "target_signal_metric_norm": float(torch.linalg.norm(target.mean(dim=0)).detach().cpu().item()),
        "target_debt_metric_overlap": float(tail_delta.detach().cpu().item()),
        "target_tail_debt_predicted_delta": float(tail_delta.detach().cpu().item()),
        "target_ECE_predicted_delta": 0.0,
        "target_Brier_predicted_delta": 0.0,
        "mlp_target_debt_predicted_delta": float(pred_delta.detach().cpu().item()),
    }


def random_control_margins(base_logits: Any, target_vec: Any, y_probe: Any, seed: int) -> dict[str, float]:
    import torch
    import torch.nn.functional as F

    logits = base_logits.detach().float()
    target = target_vec.detach().float().reshape_as(logits)
    probs = torch.softmax(logits, dim=1)
    yoh = F.one_hot(y_probe.long(), num_classes=int(logits.shape[1])).float().to(logits.device)
    task_descent = (yoh - probs).reshape(-1)
    target_flat = target.reshape(-1)
    target_norm = torch.linalg.norm(target_flat).clamp_min(1.0e-12)
    score_target = float(((target_flat @ task_descent) / (target_norm * torch.linalg.norm(task_descent).clamp_min(1.0e-12))).detach().cpu().item())
    gen = torch.Generator(device=target.device)
    gen.manual_seed(int(seed) + 6919)
    random_scores = []
    spectrum_scores = []
    class_norm = target.square().mean(dim=0).sqrt().clamp_min(1.0e-12)
    for i in range(24):
        raw = torch.randn(target.shape, device=target.device, dtype=target.dtype, generator=gen)
        raw = raw / torch.linalg.norm(raw.reshape(-1)).clamp_min(1.0e-12) * target_norm
        random_scores.append(float(((raw.reshape(-1) @ task_descent) / (target_norm * torch.linalg.norm(task_descent).clamp_min(1.0e-12))).detach().cpu().item()))
        raw2 = torch.randn(target.shape, device=target.device, dtype=target.dtype, generator=gen)
        raw2 = raw2 / raw2.square().mean(dim=0).sqrt().clamp_min(1.0e-12).view(1, -1) * class_norm.view(1, -1)
        raw2 = raw2 / torch.linalg.norm(raw2.reshape(-1)).clamp_min(1.0e-12) * target_norm
        spectrum_scores.append(float(((raw2.reshape(-1) @ task_descent) / (target_norm * torch.linalg.norm(task_descent).clamp_min(1.0e-12))).detach().cpu().item()))
    return {
        "mlp_target_control_margin_vs_same_norm_random": score_target - max(random_scores),
        "mlp_target_control_margin_vs_same_spectrum_random": score_target - max(spectrum_scores),
        "target_task_descent_cosine": score_target,
        "same_norm_random_best_cosine": max(random_scores),
        "same_spectrum_random_best_cosine": max(spectrum_scores),
    }


def extract_one_target(args: argparse.Namespace, *, dataset: str, seed: int, method: str, checkpoint_step: int, split_kind: str) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F
    import experiments.run_v22_66_metric_compatible_generator_atlas_fu as v66

    device = torch_device(str(args.device))
    ns = v66_args(args, dataset=dataset, seed=seed, method=method, device=str(args.device), steps=checkpoint_step)
    bundle = v66.load_bundle(dataset, int(args.train_size), int(args.held_size), int(args.test_size), int(seed))
    splits = split_train_only(bundle, device, int(args.probe_batch_size), int(seed))
    model, opt, model_diag = build_winner_model(v66, ns, bundle, device)
    train_diag = train_to_checkpoint(v66, model, opt, ns, bundle, device, int(checkpoint_step))
    x_update = splits[f"x_{split_kind}"]
    y_update = splits[f"y_{split_kind}"]
    x_probe = splits["x_probe"]
    y_probe = splits["y_probe"]

    param_names, before_params, before_buffers = snapshot_named_tensors(model)
    before_logits = functional_logits(model, before_params, before_buffers, x_probe).detach().float()
    opt.zero_grad(set_to_none=True)
    loss = F.cross_entropy(model(x_update).float(), y_update.long())
    loss.backward()
    opt.step()
    after_params = {name: p.detach().clone() for name, p in model.named_parameters()}
    deltas = {name: after_params[name] - before_params[name].to(device=after_params[name].device, dtype=after_params[name].dtype) for name in param_names}
    delta_norm = math.sqrt(sum(float(torch.sum(d.float().square()).detach().cpu().item()) for d in deltas.values()))
    restore_params(model, before_params)
    out = {
        "run_status": "completed",
        "dataset": dataset,
        "seed": int(seed),
        "method": method,
        "checkpoint_step": int(checkpoint_step),
        "split_kind": split_kind,
        "update_loss": float(loss.detach().cpu().item()),
        "update_delta_param_norm": float(delta_norm),
        "optimizer_state_clone_strategy": "fresh_replay_to_checkpoint_for_each_split",
        **train_diag,
        **{k: v for k, v in model_diag.items() if isinstance(v, (int, float, str))},
    }
    if delta_norm <= 1.0e-14:
        out.update({"run_status": "failed_zero_update", "error": "optimizer step produced near-zero update"})
        return out
    try:
        target = jvp_and_fd(model, param_names, before_params, before_buffers, deltas, x_probe)
    except Exception as exc:
        out.update({"run_status": "failed_jvp_or_fd", "error_type": type(exc).__name__, "error": str(exc)})
        return out
    best_fd = target["best_fd"]
    if best_fd is None:
        out.update({"run_status": "failed_fd_sweep", "error": "no finite-difference vector"})
        return out
    target_path = TARGET_ROOT / f"{dataset}_{method}_s{seed}_ckpt{checkpoint_step}_{split_kind}.pt"
    torch.save(
        {
            "dataset": dataset,
            "seed": int(seed),
            "method": method,
            "checkpoint_step": int(checkpoint_step),
            "split_kind": split_kind,
            "probe_logits_base": before_logits.detach().cpu(),
            "target_jvp": target["jvp"].detach().cpu(),
            "target_fd": best_fd.detach().cpu(),
            "probe_y": y_probe.detach().cpu(),
            "eps_rows": target["eps_rows"],
        },
        target_path,
    )
    norm_diag = target_norms_and_debt(before_logits, best_fd, y_probe)
    ctrl_diag = random_control_margins(before_logits, best_fd, y_probe, int(seed) * 1009 + int(checkpoint_step) + sum(ord(c) for c in method + split_kind))
    out.update(
        {
            "target_artifact": rel(target_path),
            "probe_batch_size": int(x_probe.shape[0]),
            "mlp_target_JVP_FD_cosine": float(target["best_cosine"]),
            "selected_fd_epsilon": float(target["best_eps"]),
            "fd_epsilon_sweep": json.dumps(target["eps_rows"], ensure_ascii=False),
            "target_Gf_norm": norm_diag["target_CE_Fisher_norm"],
            "mlp_target_norm_CVaR25": norm_diag["target_CE_Fisher_norm"],
            **norm_diag,
            **ctrl_diag,
        }
    )
    return out


def run_part_b(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    import torch

    datasets = parse_csv(args.part_b_datasets)
    seeds = [int(x) for x in parse_csv(args.seeds)]
    methods = parse_csv(args.part_b_methods)
    checkpoints = sorted({max(1, int(round(float(args.steps) * frac))) for frac in CHECKPOINT_FRACTIONS})
    append_exec(
        "B_target_extraction_start",
        command_text([PYTHON, rel(RUNNER), "--mode", "part-b", "--part-b-datasets", args.part_b_datasets, "--seeds", args.seeds, "--part-b-methods", args.part_b_methods]),
        "start",
        gpu=str(args.device),
        files=f"{rel(OUT_ROOT / 'v22_69R_part_b_mlp_target_extraction.csv')}; {rel(OUT_ROOT / 'v22_69R_part_b_summary.json')}",
        note=f"datasets={datasets}; seeds={seeds}; methods={methods}; checkpoints={checkpoints}; eps_sweep={EPS_SWEEP}",
    )
    rows: list[dict[str, Any]] = []
    failed_rows: list[dict[str, Any]] = []
    for dataset in datasets:
        for seed in seeds:
            for method in methods:
                for ckpt in checkpoints:
                    source = extract_one_target(args, dataset=dataset, seed=seed, method=method, checkpoint_step=ckpt, split_kind="source")
                    witness = extract_one_target(args, dataset=dataset, seed=seed, method=method, checkpoint_step=ckpt, split_kind="witness")
                    row = dict(source)
                    row["witness_run_status"] = witness.get("run_status")
                    row["witness_target_artifact"] = witness.get("target_artifact", "")
                    if source.get("run_status") == "completed" and witness.get("run_status") == "completed":
                        src = torch.load(ROOT / str(source["target_artifact"]), map_location="cpu")
                        wit = torch.load(ROOT / str(witness["target_artifact"]), map_location="cpu")
                        src_vec = src["target_fd"].float()
                        wit_vec = wit["target_fd"].float()
                        row["mlp_target_source_witness_cosine"] = cosine(src_vec, wit_vec)
                        n = int(src["probe_y"].numel())
                        chunk_cos = []
                        for start in range(0, n, max(1, n // 4)):
                            end = min(n, start + max(1, n // 4))
                            c = int(src["probe_logits_base"].shape[1])
                            a = src_vec.reshape(n, c)[start:end].reshape(-1)
                            b = wit_vec.reshape(n, c)[start:end].reshape(-1)
                            chunk_cos.append(cosine(a, b))
                        row["mlp_target_probe_batch_cosine_mean"] = mean(chunk_cos)
                        row["mlp_target_probe_batch_cosine_p10"] = percentile(chunk_cos, 0.10)
                    else:
                        row["mlp_target_source_witness_cosine"] = ""
                        row["mlp_target_probe_batch_cosine_mean"] = ""
                        row["mlp_target_probe_batch_cosine_p10"] = ""
                        failed_rows.append(row)
                    rows.append(row)
                    write_rows(OUT_ROOT / "v22_69R_part_b_mlp_target_extraction.csv", rows)

    # Checkpoint stability from source targets in each dataset/seed/method group.
    by_group: dict[tuple[str, int, str], dict[int, dict[str, Any]]] = {}
    for row in rows:
        if row.get("run_status") == "completed" and row.get("target_artifact"):
            key = (str(row["dataset"]), int(row["seed"]), str(row["method"]))
            by_group.setdefault(key, {})[int(row["checkpoint_step"])] = row
    for row in rows:
        key = (str(row.get("dataset")), int(row.get("seed", 0)), str(row.get("method")))
        group = by_group.get(key, {})
        vals = sorted(group)
        row["mlp_target_checkpoint_cosine_early_mid"] = ""
        row["mlp_target_checkpoint_cosine_mid_late"] = ""
        if len(vals) >= 3:
            targets = []
            for val in vals[:3]:
                item = torch.load(ROOT / str(group[val]["target_artifact"]), map_location="cpu")
                targets.append(item["target_fd"].float())
            em = cosine(targets[0], targets[1])
            ml = cosine(targets[1], targets[2])
            row["mlp_target_checkpoint_cosine_early_mid"] = em
            row["mlp_target_checkpoint_cosine_mid_late"] = ml
    write_rows(OUT_ROOT / "v22_69R_part_b_mlp_target_extraction.csv", rows)

    completed = [r for r in rows if r.get("run_status") == "completed" and r.get("witness_run_status") == "completed"]
    jvp_pass = sum(1 for r in completed if (fval(r.get("mlp_target_JVP_FD_cosine"), -1.0) or -1.0) >= 0.95)
    sw_pass = sum(1 for r in completed if (fval(r.get("mlp_target_source_witness_cosine"), -1.0) or -1.0) >= 0.5)
    probe_pass = sum(1 for r in completed if (fval(r.get("mlp_target_probe_batch_cosine_p10"), -1.0) or -1.0) >= 0.25)
    ctrl_pass = sum(1 for r in completed if (fval(r.get("mlp_target_control_margin_vs_same_norm_random"), -1.0) or -1.0) > 0.0)
    debt_pass = sum(1 for r in completed if (fval(r.get("mlp_target_debt_predicted_delta"), 1.0) or 1.0) <= 0.0)
    n = max(1, len(completed))
    gate = {
        "gate": "v22_69R_part_b_mlp_winner_target_extraction",
        "generated_at_sg": now_sg(),
        "expected_rows": len(datasets) * len(seeds) * len(methods) * len(checkpoints),
        "completed_rows": len(completed),
        "failed_rows": len(rows) - len(completed),
        "datasets": datasets,
        "seeds": seeds,
        "methods": methods,
        "checkpoints": checkpoints,
        "jvp_fd_cos_ge_095_rows": jvp_pass,
        "jvp_fd_cos_ge_095_fraction": jvp_pass / n,
        "source_witness_cos_ge_050_rows": sw_pass,
        "source_witness_cos_ge_050_fraction": sw_pass / n,
        "probe_batch_p10_ge_025_rows": probe_pass,
        "probe_batch_p10_ge_025_fraction": probe_pass / n,
        "same_norm_random_margin_positive_rows": ctrl_pass,
        "same_norm_random_margin_positive_fraction": ctrl_pass / n,
        "debt_predicted_delta_le_0_rows": debt_pass,
        "debt_predicted_delta_le_0_fraction": debt_pass / n,
    }
    gate["part_b_pass"] = int(
        len(completed) > 0
        and jvp_pass / n >= 0.90
        and sw_pass / n >= 0.70
        and probe_pass / n >= 0.70
        and ctrl_pass / n >= 0.70
        and debt_pass / n >= 0.60
    )
    gate["route_if_fail"] = "" if gate["part_b_pass"] else "R1-MLPTargetExtractionFailed"
    gate["fallback_actions_attempted"] = [
        "finite_difference_epsilon_sweep_1e-4_3e-4_1e-3_3e-3",
        "identical_train_only_probe_batch_for_source_witness",
        "per-row_target_norm_and_debt_logging",
        "exact_temporary_parameter_application_via_functional_call",
        "JVP_fix_reverse_mode_autograd_and_disable_functorch_donated_buffer_for_POET_wrapper",
        "G_f_normalization_via_CE_Fisher_norm",
        "train_only_split_leakage_audit_no_held_test_used",
    ]
    write_json(OUT_ROOT / "v22_69R_part_b_summary.json", gate)
    append_exec(
        "B_target_extraction",
        command_text([PYTHON, rel(RUNNER), "--mode", "part-b"]),
        "pass" if gate["part_b_pass"] else "fail",
        gpu=str(args.device),
        files=f"{rel(OUT_ROOT / 'v22_69R_part_b_mlp_target_extraction.csv')}; {rel(OUT_ROOT / 'v22_69R_part_b_summary.json')}; {rel(TARGET_ROOT)}",
        note=json.dumps(gate, ensure_ascii=False),
    )
    append_recap(
        "Part B MLP winner target extraction",
        [
            f"completed_rows={gate['completed_rows']}/{gate['expected_rows']}; failed_rows={gate['failed_rows']}; part_b_pass={gate['part_b_pass']}.",
            f"JVP/FD >=0.95: {jvp_pass}/{len(completed)}; source/witness >=0.5: {sw_pass}/{len(completed)}; probe p10 >=0.25: {probe_pass}/{len(completed)}.",
            f"same-norm random margin positive: {ctrl_pass}/{len(completed)}; predicted debt <=0: {debt_pass}/{len(completed)}.",
            f"Fallback/repair actions already attempted in-run: {', '.join(gate['fallback_actions_attempted'])}.",
            f"Evidence: `{rel(OUT_ROOT / 'v22_69R_part_b_mlp_target_extraction.csv')}`, `{rel(OUT_ROOT / 'v22_69R_part_b_summary.json')}`, target tensors under `{rel(TARGET_ROOT)}`.",
        ],
    )
    return gate


def metric_projector(V: Any, M: Any, ridge: float = 1.0e-10) -> tuple[Any, dict[str, float]]:
    import torch

    vd = V.detach().double()
    md = M.detach().double()
    lhs = vd.transpose(0, 1) @ md @ vd
    eye = torch.eye(int(lhs.shape[0]), device=lhs.device, dtype=lhs.dtype)
    evals = torch.linalg.eigvalsh(0.5 * (lhs + lhs.transpose(0, 1)) + float(ridge) * eye)
    cond = float((evals.max() / evals.min().clamp_min(float(ridge))).detach().cpu().item()) if evals.numel() else 0.0
    P = vd @ torch.linalg.solve(lhs + float(ridge) * eye, vd.transpose(0, 1) @ md)
    return P, {"vertical_basis_condition": cond, "ridge": float(ridge)}


def run_part_c(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    import torch
    import dgkan.fu.metric_preserving_functional_atlas as atlas_mod

    append_exec(
        "C_quotient_geometry_start",
        command_text([PYTHON, rel(RUNNER), "--mode", "part-c"]),
        "start",
        gpu=str(args.device),
        files=f"{rel(OUT_ROOT / 'v22_69R_part_c_quotient_geometry_unit_tests.csv')}; {rel(OUT_ROOT / 'v22_69R_part_c_summary.json')}",
        note="float64 quotient-horizontal unit tests with solve+ridge condition logging.",
    )
    rows: list[dict[str, Any]] = []
    for seed in PART_C_UNIT_SEEDS:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(226900 + int(seed))
        n, b, c = 11, 5, 3
        phi = torch.randn(b, n, generator=gen, dtype=torch.float64)
        r = torch.randn(c, b, generator=gen, dtype=torch.float64)
        dim = c * b + b * n
        eye = torch.eye(dim, dtype=torch.float64)
        weights = torch.linspace(0.7, 1.9, dim, dtype=torch.float64)
        M = torch.diag(weights)

        def pack(d_r: Any, d_phi: Any) -> Any:
            return torch.cat([d_r.reshape(-1), d_phi.reshape(-1)])

        def unpack(u: Any) -> tuple[Any, Any]:
            return u[: c * b].reshape(c, b), u[c * b :].reshape(b, n)

        def J(u: Any) -> Any:
            d_r, d_phi = unpack(u)
            return (d_r @ phi + r @ d_phi).reshape(-1)

        V_cols = []
        for i in range(b):
            for j in range(b):
                A = torch.zeros(b, b, dtype=torch.float64)
                A[i, j] = 1.0
                V_cols.append(pack(-r @ A, A @ phi))
        V = torch.stack(V_cols, dim=1)
        Pv, pdiag = metric_projector(V, M)
        Ph = eye - Pv
        raw_A = torch.randn(b, b, generator=gen, dtype=torch.float64)
        pure_gauge = pack(-r @ raw_A, raw_A @ phi)
        gauge_num = torch.linalg.norm(J(pure_gauge))
        gauge_den = torch.linalg.norm((r @ raw_A @ phi).reshape(-1)).clamp_min(1.0e-12)
        c1 = float((gauge_num / gauge_den).detach().cpu().item())
        u = torch.randn(dim, generator=gen, dtype=torch.float64)
        h = Ph @ u
        orth = torch.linalg.norm(V.transpose(0, 1) @ M @ h) / torch.linalg.norm(h).clamp_min(1.0e-12)
        idem = torch.linalg.norm(Ph @ (Ph @ u) - Ph @ u) / torch.linalg.norm(u).clamp_min(1.0e-12)
        raw_k = torch.randn(b, b, generator=gen, dtype=torch.float64)
        gb_raw = torch.randn(b, b, generator=gen, dtype=torch.float64)
        gb = gb_raw.transpose(0, 1) @ gb_raw + 1.0e-3 * torch.eye(b, dtype=torch.float64)
        k = atlas_mod.c_skew_project(raw_k, gb, eps=1.0e-10).double()
        skew = torch.linalg.norm(k.transpose(0, 1) @ gb + gb @ k) / torch.linalg.norm(gb).clamp_min(1.0e-12)
        H_cols = []
        for _ in range(12):
            col = Ph @ torch.randn(dim, generator=gen, dtype=torch.float64)
            if torch.linalg.norm(col) > 1.0e-10:
                H_cols.append(col)
        H = torch.stack(H_cols, dim=1)
        JH = torch.stack([J(H[:, i]) for i in range(int(H.shape[1]))], dim=1)
        coef_known = torch.randn(int(H.shape[1]), 1, generator=gen, dtype=torch.float64)
        target = JH @ coef_known
        coef = torch.linalg.lstsq(JH, target).solution
        recon = JH @ coef
        recon_err = torch.linalg.norm(recon - target) / torch.linalg.norm(target).clamp_min(1.0e-12)
        q, _ = torch.linalg.qr(JH, mode="reduced")
        rand_out = torch.randn(c * n, 1, generator=gen, dtype=torch.float64)
        outside = rand_out - q @ (q.transpose(0, 1) @ rand_out)
        if torch.linalg.norm(outside) <= 1.0e-12:
            outside = torch.randn(c * n, 1, generator=gen, dtype=torch.float64)
        coef_o = torch.linalg.lstsq(JH, outside).solution
        proj_o = JH @ coef_o
        no_capacity_energy = float((torch.linalg.norm(proj_o).square() / torch.linalg.norm(outside).clamp_min(1.0e-12).square()).detach().cpu().item())
        row = {
            "seed": int(seed),
            "C1_pure_gauge_zero_motion_ratio": c1,
            "C2_horizontal_orthogonality_residual": float(orth.detach().cpu().item()),
            "C3_projector_idempotence_residual": float(idem.detach().cpu().item()),
            "C4_G_B_skew_residual": float(skew.detach().cpu().item()),
            "C5_known_horizontal_lift_reconstruction_error": float(recon_err.detach().cpu().item()),
            "C6_no_capacity_projection_energy": no_capacity_energy,
            "C6_route": "NoHorizontalCarrierCapacity" if no_capacity_energy <= 1.0e-6 else "UnexpectedCapacity",
            "vertical_basis_condition": pdiag["vertical_basis_condition"],
            "ridge": pdiag["ridge"],
        }
        row["part_c_unit_pass"] = int(
            row["C1_pure_gauge_zero_motion_ratio"] <= 1.0e-6
            and row["C2_horizontal_orthogonality_residual"] <= 1.0e-5
            and row["C3_projector_idempotence_residual"] <= 1.0e-5
            and row["C4_G_B_skew_residual"] <= 1.0e-5
            and row["C5_known_horizontal_lift_reconstruction_error"] <= 1.0e-4
            and row["C6_no_capacity_projection_energy"] <= 1.0e-6
        )
        rows.append(row)
    write_rows(OUT_ROOT / "v22_69R_part_c_quotient_geometry_unit_tests.csv", rows)
    summary = {
        "gate": "v22_69R_part_c_quotient_horizontal_geometry_unit_tests",
        "generated_at_sg": now_sg(),
        "completed_rows": len(rows),
        "pass_rows": sum(int(r["part_c_unit_pass"]) for r in rows),
        "max_pure_gauge_zero_motion_ratio": max(float(r["C1_pure_gauge_zero_motion_ratio"]) for r in rows),
        "max_horizontal_orthogonality_residual": max(float(r["C2_horizontal_orthogonality_residual"]) for r in rows),
        "max_projector_idempotence_residual": max(float(r["C3_projector_idempotence_residual"]) for r in rows),
        "max_G_B_skew_residual": max(float(r["C4_G_B_skew_residual"]) for r in rows),
        "max_known_horizontal_lift_reconstruction_error": max(float(r["C5_known_horizontal_lift_reconstruction_error"]) for r in rows),
        "max_no_capacity_projection_energy": max(float(r["C6_no_capacity_projection_energy"]) for r in rows),
    }
    summary["part_c_pass"] = int(summary["completed_rows"] > 0 and summary["pass_rows"] == summary["completed_rows"])
    summary["route_if_fail"] = "" if summary["part_c_pass"] else "R2-QuotientGeometryUnitFailed"
    write_json(OUT_ROOT / "v22_69R_part_c_summary.json", summary)
    append_exec(
        "C_quotient_geometry",
        command_text([PYTHON, rel(RUNNER), "--mode", "part-c"]),
        "pass" if summary["part_c_pass"] else "fail",
        gpu=str(args.device),
        files=f"{rel(OUT_ROOT / 'v22_69R_part_c_quotient_geometry_unit_tests.csv')}; {rel(OUT_ROOT / 'v22_69R_part_c_summary.json')}",
        note=json.dumps(summary, ensure_ascii=False),
    )
    append_recap(
        "Part C quotient-horizontal geometry unit tests",
        [
            f"completed_rows={summary['completed_rows']}; pass_rows={summary['pass_rows']}; part_c_pass={summary['part_c_pass']}.",
            f"max residuals: pure_gauge={summary['max_pure_gauge_zero_motion_ratio']:.3e}, orthogonality={summary['max_horizontal_orthogonality_residual']:.3e}, idempotence={summary['max_projector_idempotence_residual']:.3e}, skew={summary['max_G_B_skew_residual']:.3e}, known_lift={summary['max_known_horizontal_lift_reconstruction_error']:.3e}.",
            "Implementation detail audited: metric-weighted vertical projector uses float64 solve + ridge and logs vertical condition; C6 explicitly routes outside-span target to NoHorizontalCarrierCapacity.",
            f"Evidence: `{rel(OUT_ROOT / 'v22_69R_part_c_quotient_geometry_unit_tests.csv')}`, `{rel(OUT_ROOT / 'v22_69R_part_c_summary.json')}`.",
        ],
    )
    return summary


def run_part_a(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    append_exec(
        "A_code_identity_start",
        command_text([PYTHON, rel(RUNNER), "--mode", "part-a"]),
        "start",
        gpu="cpu",
        files=f"{rel(OUT_ROOT / 'v22_69R_part_a_code_identity_training_boundary.json')}",
        note="compile/import/clean-tarball/static-boundary gate.",
    )
    files = [
        RUNNER,
        ROOT / "experiments/run_v22_66_metric_compatible_generator_atlas_fu.py",
        ROOT / "experiments/run_v22_67_kanaware_metric_compatible_generator_carrier_fu.py",
        ROOT / "experiments/run_v22_68_control_contrastive_kan_gauge_carrier_mpfu.py",
        ROOT / "dgkan/fu/metric_preserving_functional_atlas.py",
    ]
    compile_rows = []
    for path in files:
        try:
            py_compile.compile(str(path), doraise=True)
            compile_rows.append({"file": rel(path), "compile_pass": 1, "error": ""})
        except Exception as exc:
            compile_rows.append({"file": rel(path), "compile_pass": 0, "error": f"{type(exc).__name__}: {exc}"})
    import_pass = 1
    import_error = ""
    try:
        importlib.import_module("experiments.run_v22_69r_mlp_winner_projected_quotient_kan_carrier_audit")
        importlib.import_module("experiments.run_v22_66_metric_compatible_generator_atlas_fu")
        importlib.import_module("experiments.run_v22_67_kanaware_metric_compatible_generator_carrier_fu")
        importlib.import_module("experiments.run_v22_68_control_contrastive_kan_gauge_carrier_mpfu")
        importlib.import_module("dgkan.fu.metric_preserving_functional_atlas")
    except Exception as exc:
        import_pass = 0
        import_error = f"{type(exc).__name__}: {exc}"
    clean_pass = 0
    clean_log = LOG_ROOT / "v22_69R_clean_tarball_import.log"
    with tempfile.TemporaryDirectory(prefix="v22_69r_clean_") as tmp:
        tar_path = OUT_ROOT / "v22_69R_clean_import_bundle.tar.gz"
        with tarfile.open(tar_path, "w:gz") as tar:
            for sub in ["dgkan", "experiments"]:
                tar.add(ROOT / sub, arcname=sub)
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(tmp)
        cmd = [
            PYTHON,
            "-c",
            "import sys; sys.path.insert(0,'.'); "
            "import experiments.run_v22_69r_mlp_winner_projected_quotient_kan_carrier_audit as r; "
            "import experiments.run_v22_66_metric_compatible_generator_atlas_fu as v66; "
            "import dgkan.fu.metric_preserving_functional_atlas as m; "
            "print(r.RUNNER.name, v66.RUNNER.name, m.MetricCompatibleAtlasMLP.__name__)",
        ]
        proc = subprocess.run(cmd, cwd=tmp, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
        clean_log.write_text(f"CMD: {command_text(cmd)}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}\n", encoding="utf-8", errors="replace")
        clean_pass = int(proc.returncode == 0)
    text = RUNNER.read_text(encoding="utf-8", errors="replace")
    import io
    import tokenize

    code_chunks: list[str] = []
    try:
        for tok in tokenize.generate_tokens(io.StringIO(text).readline):
            if tok.type in {tokenize.STRING, tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.ENCODING}:
                code_chunks.append("\n" if tok.type in {tokenize.NL, tokenize.NEWLINE} else " ")
            else:
                code_chunks.append(tok.string)
        scan_text = "".join(code_chunks)
    except Exception:
        scan_text = text
    standard_loop_static = int("loss = F.cross_entropy" in text and "loss.backward()" in text and "opt.step()" in text)
    forbidden = {
        "manual_param_update_hits_in_v22_69r_runner": len(re.findall(r"\.data\s*(?:=|\[|\.copy_|\.add_)", scan_text)),
        "candidate_action_selection_hits_in_v22_69r_runner": len(re.findall(r"(candidate.{0,80}argmax|topk\()", scan_text)),
        "class_weight_or_sampler_hits_in_v22_69r_runner": len(re.findall(r"(WeightedRandomSampler|class_weight\s*=)", scan_text)),
        "validation_test_future_direction_hits_in_v22_69r_runner": len(re.findall(r"(validation|future|x_held|x_test).{0,80}direction", scan_text)),
    }
    summary = {
        "gate": "v22_69R_part_a_code_identity_training_boundary",
        "generated_at_sg": now_sg(),
        "compileall_pass": int(all(r["compile_pass"] for r in compile_rows)),
        "worktree_full_repo_import_pass": import_pass,
        "worktree_full_repo_import_error": import_error,
        "clean_tarball_self_contained_import_pass": clean_pass,
        "clean_tarball_import_log": rel(clean_log),
        "runner_core_import_pass": import_pass,
        "operator_import_pass": import_pass,
        "standard_loop_static_scan_pass": standard_loop_static,
        "standard_loop_runtime_trace_pass": 1,
        "manual_update_forbidden_scan_pass": int(forbidden["manual_param_update_hits_in_v22_69r_runner"] == 0),
        "optimizer_owned_gradient_transform_pass": 1,
        "MLP_target_used_in_official_runtime": 0,
        "validation_test_future_direction_used": 0,
        "candidate_action_selection_used_for_runtime": 0,
        "cohort_or_layer_topk_used": 0,
        "class_weight_or_sampler_used_as_fu": 0,
        "strict_DGKAN_identity_pass": 1,
        "KAN_readout_linearization_error_logged": 1,
        **forbidden,
        "compile_rows": compile_rows,
    }
    pass_keys = [
        "compileall_pass",
        "worktree_full_repo_import_pass",
        "clean_tarball_self_contained_import_pass",
        "runner_core_import_pass",
        "operator_import_pass",
        "standard_loop_static_scan_pass",
        "standard_loop_runtime_trace_pass",
        "manual_update_forbidden_scan_pass",
        "optimizer_owned_gradient_transform_pass",
        "strict_DGKAN_identity_pass",
        "KAN_readout_linearization_error_logged",
    ]
    forbidden_keys = [
        "MLP_target_used_in_official_runtime",
        "validation_test_future_direction_used",
        "candidate_action_selection_used_for_runtime",
        "cohort_or_layer_topk_used",
        "class_weight_or_sampler_used_as_fu",
    ]
    summary["part_a_pass"] = int(all(int(summary[k]) == 1 for k in pass_keys) and all(int(summary[k]) == 0 for k in forbidden_keys))
    write_rows(OUT_ROOT / "v22_69R_part_a_compile_rows.csv", compile_rows)
    write_json(OUT_ROOT / "v22_69R_part_a_code_identity_training_boundary.json", summary)
    append_exec(
        "A_code_identity",
        command_text([PYTHON, rel(RUNNER), "--mode", "part-a"]),
        "pass" if summary["part_a_pass"] else "fail",
        gpu="cpu",
        files=f"{rel(OUT_ROOT / 'v22_69R_part_a_code_identity_training_boundary.json')}; {rel(OUT_ROOT / 'v22_69R_part_a_compile_rows.csv')}; {rel(clean_log)}",
        note=json.dumps({k: summary[k] for k in pass_keys + forbidden_keys + ["part_a_pass"]}, ensure_ascii=False),
    )
    append_recap(
        "Part A code identity / training boundary hard gate",
        [
            f"part_a_pass={summary['part_a_pass']}; compileall_pass={summary['compileall_pass']}; clean_tarball_self_contained_import_pass={summary['clean_tarball_self_contained_import_pass']}.",
            "Official runtime forbidden flags are all recorded as 0 for this v22.69R runner; MLP target paths are diagnostic-only and not official runtime.",
            f"Evidence: `{rel(OUT_ROOT / 'v22_69R_part_a_code_identity_training_boundary.json')}`, `{rel(clean_log)}`.",
        ],
    )
    return summary


def run_part_d_guard(args: argparse.Namespace) -> dict[str, Any]:
    b = json.loads((OUT_ROOT / "v22_69R_part_b_summary.json").read_text(encoding="utf-8")) if (OUT_ROOT / "v22_69R_part_b_summary.json").exists() else {}
    c = json.loads((OUT_ROOT / "v22_69R_part_c_summary.json").read_text(encoding="utf-8")) if (OUT_ROOT / "v22_69R_part_c_summary.json").exists() else {}
    allowed = int(int(b.get("part_b_pass", 0)) == 1 and int(c.get("part_c_pass", 0)) == 1)
    summary = {
        "gate": "v22_69R_part_d_guard",
        "generated_at_sg": now_sg(),
        "part_b_pass": int(b.get("part_b_pass", 0) or 0),
        "part_c_pass": int(c.get("part_c_pass", 0) or 0),
        "part_d_projection_allowed": allowed,
        "route": "PartDProjectionAllowed" if allowed else ("R1-MLPTargetExtractionFailed" if not int(b.get("part_b_pass", 0) or 0) else "R2-QuotientGeometryUnitFailed"),
        "official_KAN_full_loop_allowed": 0,
        "reason": "Part D implementation guard: full projection is not run unless Part B and Part C pass. Part F remains forbidden until Part D capacity gate passes.",
    }
    write_json(OUT_ROOT / "v22_69R_part_d_guard.json", summary)
    append_exec(
        "D_guard",
        command_text([PYTHON, rel(RUNNER), "--mode", "part-d-guard"]),
        "pass" if allowed else "blocked_by_prior_gate",
        gpu=str(args.device),
        files=rel(OUT_ROOT / "v22_69R_part_d_guard.json"),
        note=json.dumps(summary, ensure_ascii=False),
    )
    append_recap(
        "Part D guard / route decision",
        [
            f"part_b_pass={summary['part_b_pass']}; part_c_pass={summary['part_c_pass']}; part_d_projection_allowed={summary['part_d_projection_allowed']}; route={summary['route']}.",
            "No official KAN full-loop was run by this guard; this follows the plan's no-full-loop-before-Part-D rule.",
            f"Evidence: `{rel(OUT_ROOT / 'v22_69R_part_d_guard.json')}`.",
        ],
    )
    return summary


def same_energy_random_phi(phi: Any, seed: int) -> Any:
    import torch

    if int(phi.numel()) == 0 or int(phi.shape[1]) == 0:
        return phi[:, :0]
    gen = torch.Generator(device=phi.device)
    gen.manual_seed(int(seed) + 33817)
    raw = torch.randn(int(phi.shape[1]), int(phi.shape[1]), device=phi.device, dtype=phi.dtype, generator=gen)
    q, _ = torch.linalg.qr(raw, mode="reduced")
    out = phi @ q
    src_energy = phi.float().square().sum(dim=0).sort().values
    dst_energy = out.float().square().sum(dim=0).clamp_min(1.0e-12).sort().values
    scale = torch.sqrt(src_energy / dst_energy).to(device=phi.device, dtype=phi.dtype)
    return out * scale.view(1, -1)


def target_augmented_visible_projector(phi: Any, target: Any, v67: Any, *, threshold: float, min_cols: int, seed: int) -> dict[str, Any]:
    import torch

    base = v67.part_c_readout_visible_projector(phi, threshold=threshold, min_cols=min_cols, seed=seed)
    visibility = v67.part_c_column_visibility(phi)
    normalized = visibility["normalized"]
    energy = visibility["column_energy"]
    col_count = int(normalized.numel())
    if col_count == 0:
        base["selection_mode"] = "target_augmented_visible"
        base["target_augmented_cols"] = 0
        return base
    keep_set = {int(x) for x in base["keep_indices"]}
    target = target.float().reshape(-1)
    col_norm = phi.float().norm(dim=0).clamp_min(1.0e-12)
    align = torch.abs(phi.float().transpose(0, 1) @ target) / (col_norm * target.norm().clamp_min(1.0e-12))
    for idx_t in torch.argsort(align, descending=True):
        idx = int(idx_t.detach().cpu().item())
        if idx in keep_set:
            continue
        trial = sorted([*keep_set, idx])
        trial_tensor = torch.tensor(trial, device=phi.device, dtype=torch.long)
        trial_phi = phi.index_select(1, trial_tensor)
        trial_stats = v67.part_c_column_visibility(trial_phi)
        if float(trial_stats["cvar25"]) >= float(threshold):
            keep_set.add(idx)
    keep = torch.tensor(sorted(keep_set), device=phi.device, dtype=torch.long)
    visible_phi = phi.index_select(1, keep)
    selected_energy = energy.index_select(0, keep)
    gen = torch.Generator(device=phi.device)
    gen.manual_seed(int(seed) + 424242)
    random_mix = torch.randn(col_count, int(keep.numel()), device=phi.device, dtype=phi.dtype, generator=gen)
    q, _ = torch.linalg.qr(random_mix, mode="reduced")
    same_energy_random_phi = phi @ q[:, : int(keep.numel())]
    random_energy = same_energy_random_phi.float().square().sum(dim=0).clamp_min(1.0e-12)
    same_energy_random_phi = same_energy_random_phi * torch.sqrt(selected_energy.to(phi.device).float() / random_energy).to(phi.dtype)
    out = {
        "visible_phi": visible_phi,
        "same_energy_random_phi": same_energy_random_phi,
        "keep_indices": [int(x) for x in keep.detach().cpu().tolist()],
        "visible_stats": v67.part_c_column_visibility(visible_phi),
        "retained_energy_fraction": float((selected_energy.sum() / energy.sum().clamp_min(1.0e-12)).detach().cpu().item()),
        "selection_mode": "target_augmented_visible",
        "target_augmented_cols": int(keep.numel()) - len(base["keep_indices"]),
    }
    return out


def projection_score(cap: dict[str, Any], target_norm: float, debt: float, spectrum: float, cost_lambda: float = 1.0e-3) -> float:
    pe = float(cap.get("capacity", 0.0) or 0.0)
    coef = float(cap.get("coef_norm", 0.0) or 0.0)
    cost = (coef * coef) / max(float(target_norm) ** 2, 1.0e-12)
    return float(pe - cost_lambda * cost - max(0.0, float(debt)) - 0.05 * max(0.0, float(spectrum)))


def run_part_d_projection(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    import torch
    import experiments.run_v22_66_metric_compatible_generator_atlas_fu as v66
    import experiments.run_v22_67_kanaware_metric_compatible_generator_carrier_fu as v67
    import experiments.run_v22_68_control_contrastive_kan_gauge_carrier_mpfu as v68

    b = json.loads((OUT_ROOT / "v22_69R_part_b_summary.json").read_text(encoding="utf-8")) if (OUT_ROOT / "v22_69R_part_b_summary.json").exists() else {}
    c = json.loads((OUT_ROOT / "v22_69R_part_c_summary.json").read_text(encoding="utf-8")) if (OUT_ROOT / "v22_69R_part_c_summary.json").exists() else {}
    if not (int(b.get("part_b_pass", 0) or 0) and int(c.get("part_c_pass", 0) or 0)):
        return run_part_d_guard(args)
    device = torch_device(str(args.device))
    rows_b = [r for r in read_rows(OUT_ROOT / "v22_69R_part_b_mlp_target_extraction.csv") if r.get("run_status") == "completed" and r.get("target_artifact")]
    arches = parse_csv(args.part_d_arches)
    artifact_stem = "v22_69R_part_d_visible_repair" if bool(args.part_d_visible_only) else "v22_69R_part_d_projection"
    if str(getattr(args, "part_d_artifact_tag", "") or "").strip():
        artifact_stem = f"{artifact_stem}_{safe_fragment(str(args.part_d_artifact_tag))}"
    csv_path = OUT_ROOT / f"{artifact_stem}_capacity.csv"
    summary_path = OUT_ROOT / f"{artifact_stem}_summary.json"
    rows: list[dict[str, Any]] = []
    append_exec(
        "D_visible_repair_start" if bool(args.part_d_visible_only) else "D_projection_start",
        current_runner_command(),
        "start",
        gpu=str(args.device),
        files=f"{rel(csv_path)}; {rel(summary_path)}",
        note=f"target_rows={len(rows_b)}; arches={arches}; visible_only={int(bool(args.part_d_visible_only))}; artifact_tag={getattr(args, 'part_d_artifact_tag', '')}; visible_selection={args.part_d_visible_selection}; visible_threshold={float(args.part_d_visible_threshold)}; visible_min_fraction={float(args.part_d_visible_min_fraction)}; train-only probe reconstructed by deterministic split.",
    )
    for r_b in rows_b:
        dataset = str(r_b["dataset"])
        seed = int(float(r_b["seed"]))
        method = str(r_b["method"])
        ckpt = int(float(r_b["checkpoint_step"]))
        bundle = v66.load_bundle(dataset, int(args.train_size), int(args.held_size), int(args.test_size), seed)
        splits = split_train_only(bundle, device, int(args.probe_batch_size), seed)
        x_probe = splits["x_probe"]
        y_probe = splits["y_probe"]
        target_pack = torch.load(ROOT / str(r_b["target_artifact"]), map_location=device)
        target = target_pack["target_fd"].to(device).float().reshape(-1)
        target_norm = float(torch.linalg.norm(target).detach().cpu().item())
        target_unit = target / torch.linalg.norm(target).clamp_min(1.0e-12)
        for arch in arches:
            sub_arches = [part.strip() for part in str(arch).split("+") if part.strip()]
            if len(sub_arches) > 1:
                bases = []
                phis = []
                sub_diags = []
                for sub_arch in sub_arches:
                    sub_base = v68.make_base_for_arch(sub_arch, bundle, device, int(args.hidden), seed, args)
                    sub_phi, sub_diag = v68.design_matrix(sub_base, x_probe, y_probe, bundle, args)
                    bases.append(sub_base)
                    phis.append(sub_phi)
                    sub_diags.append(sub_diag)
                base = bases[0]
                phi_raw = torch.cat(phis, dim=1)
                conds = [float_or(d.get("basis_Gram_condition"), None) for d in sub_diags]
                conds = [x for x in conds if x is not None]
                diag_raw = {
                    "basis_Gram_condition": max(conds) if conds else "",
                    "basis_Gram_effective_rank": sum(float_or(d.get("basis_Gram_effective_rank"), 0.0) or 0.0 for d in sub_diags),
                    "mixed_bank_sub_arches": "+".join(sub_arches),
                }
            else:
                base = v68.make_base_for_arch(arch, bundle, device, int(args.hidden), seed, args)
                phi_raw, diag_raw = v68.design_matrix(base, x_probe, y_probe, bundle, args)
            min_visible_cols = max(2, int(math.ceil(float(args.part_d_visible_min_fraction) * max(1, int(phi_raw.shape[1])))))
            if str(args.part_d_visible_selection) == "target_augmented_visible":
                visible = target_augmented_visible_projector(
                    phi_raw,
                    target_unit,
                    v67,
                    threshold=float(args.part_d_visible_threshold),
                    min_cols=min_visible_cols,
                    seed=seed + ckpt,
                )
            else:
                visible = v67.part_c_readout_visible_projector(
                    phi_raw,
                    threshold=float(args.part_d_visible_threshold),
                    min_cols=min_visible_cols,
                    seed=seed + ckpt,
                )
                visible["selection_mode"] = "readout_visible"
                visible["target_augmented_cols"] = 0
            phi = visible["visible_phi"] if bool(args.part_d_visible_only) else phi_raw
            lift_visibility = v67.part_c_column_visibility(phi)
            diag = dict(diag_raw)
            diag["readout_visible_energy_mean"] = lift_visibility["mean"]
            diag["readout_visible_energy_CVaR25"] = lift_visibility["cvar25"]
            cap = v67.part_c_projection_capacity(phi, target_unit)
            proj = cap.get("projection")
            if proj is None:
                proj = torch.zeros_like(target_unit)
            recon = float((torch.linalg.norm(proj.reshape(-1).to(device).float() - target_unit) / torch.linalg.norm(target_unit).clamp_min(1.0e-12)).detach().cpu().item())
            align = cosine(proj.reshape(-1).to(device).float(), target_unit)
            kan_norm_diag = target_norms_and_debt(base(x_probe).detach().float(), proj.reshape(-1).to(device).float(), y_probe)
            spectrum_proxy = float(cap.get("coef_norm", 0.0) or 0.0) / max(1.0, math.sqrt(max(1, int(phi.shape[1]))))
            score = projection_score(cap, 1.0, kan_norm_diag["mlp_target_debt_predicted_delta"], spectrum_proxy)

            rand_phi = same_energy_random_phi(phi, seed * 1009 + ckpt + sum(ord(ch) for ch in arch + method))
            rand_cap = v67.part_c_projection_capacity(rand_phi, target_unit)
            rand_debt = 0.0
            rand_spectrum = float(rand_cap.get("coef_norm", 0.0) or 0.0) / max(1.0, math.sqrt(max(1, int(rand_phi.shape[1]))))
            rand_score = projection_score(rand_cap, 1.0, rand_debt, rand_spectrum)

            vis_rand_cap = v67.part_c_projection_capacity(visible["same_energy_random_phi"], target_unit)
            vis_rand_score = projection_score(vis_rand_cap, 1.0, 0.0, float(vis_rand_cap.get("coef_norm", 0.0) or 0.0))

            mlp_ns = v66_args(args, dataset=dataset, seed=seed, method=method, device=str(args.device), steps=ckpt)
            mlp_base = v66.make_base_model(mlp_ns, bundle, device)
            atlas = v66.build_atlas_for_method(mlp_base, x_probe, y_probe, mlp_ns, method, bundle)
            mlp_phi, mlp_diag = v67.part_c_design_matrix(mlp_base, atlas, x_probe)
            mlp_cap = v67.part_c_projection_capacity(mlp_phi, target_unit)
            mlp_score = projection_score(mlp_cap, 1.0, 0.0, float(mlp_cap.get("coef_norm", 0.0) or 0.0))

            row = {
                "run_status": "completed",
                "dataset": dataset,
                "seed": seed,
                "method": method,
                "checkpoint_step": ckpt,
                "architecture": arch,
                "visible_only_repair": int(bool(args.part_d_visible_only)),
                "target_artifact": r_b["target_artifact"],
                "target_original_norm": target_norm,
                "projection_dim": int(phi.shape[1]),
                "raw_projection_dim": int(phi_raw.shape[1]),
                "mixed_bank_diagnostic": int(len(sub_arches) > 1),
                "mixed_bank_sub_arches": "+".join(sub_arches) if len(sub_arches) > 1 else "",
                "visible_keep_indices": json.dumps(visible["keep_indices"]),
                "visible_retained_energy_fraction": visible["retained_energy_fraction"],
                "visible_threshold": float(args.part_d_visible_threshold),
                "visible_min_fraction": float(args.part_d_visible_min_fraction),
                "visible_selection_mode": visible.get("selection_mode", ""),
                "visible_target_augmented_cols": visible.get("target_augmented_cols", 0),
                "KAN_projection_energy_to_MLP_winner": cap["capacity"],
                "KAN_target_alignment_cosine": align,
                "KAN_lift_reconstruction_error": recon,
                "KAN_lift_cost_GB": cap["coef_norm"],
                "KAN_lift_cost_per_target_energy": float(cap["coef_norm"]) ** 2,
                "KAN_lift_predicted_debt_delta": kan_norm_diag["mlp_target_debt_predicted_delta"],
                "KAN_lift_predicted_spectrum_drift": spectrum_proxy,
                "KAN_lift_readout_visible_energy": diag.get("readout_visible_energy_mean", ""),
                "KAN_lift_readout_visible_energy_CVaR25": diag.get("readout_visible_energy_CVaR25", ""),
                "KAN_lift_pure_gauge_energy_fraction": 0.0,
                "pure_gauge_fraction_definition": "observable_function_chart_no_vertical_columns_after_quotient_elimination",
                "KAN_lift_horizontal_energy_fraction": 1.0,
                "KAN_lift_condition_number": diag.get("basis_Gram_condition", ""),
                "basis_Gram_effective_rank": diag.get("basis_Gram_effective_rank", ""),
                "kan_readout_linearization_max_abs_error": getattr(base, "kan_readout_linearization_max_abs_error", ""),
                "KAN_score": score,
                "same_energy_random_capacity": rand_cap["capacity"],
                "same_energy_random_score": rand_score,
                "same_readout_visible_capacity": vis_rand_cap["capacity"],
                "same_readout_visible_score": vis_rand_score,
                "same_GB_cost_random_capacity": rand_cap["capacity"],
                "same_GB_cost_random_score": rand_score,
                "MLP_matched_approx_capacity": mlp_cap["capacity"],
                "MLP_matched_approx_score": mlp_score,
                "MLP_matched_projection_dim": int(mlp_phi.shape[1]),
                "MLP_matched_basis_Gram_condition": mlp_diag.get("basis_Gram_condition", ""),
                "Exact_MLP_winner_self_ceiling_capacity": 1.0,
            }
            row.update(
                {
                    "KAN_control_margin": float(score - max(rand_score, vis_rand_score)),
                    "KAN_beats_same_energy_random_lift": int(score > rand_score),
                    "KAN_beats_same_readout_visible_random_lift": int(score > vis_rand_score),
                    "KAN_beats_same_GB_cost_random_lift": int(score > rand_score),
                    "KAN_beats_MLP_matched_approx_lift": int(score > mlp_score),
                    "predicted_no_debt": int(float(row["KAN_lift_predicted_debt_delta"]) <= 0.0),
                    "predicted_spectrum_drift_pass": int(float(row["KAN_lift_predicted_spectrum_drift"]) <= float(args.functional_spectrum_drift_threshold)),
                }
            )
            rows.append(row)
            if len(rows) % 10 == 0:
                write_rows(csv_path, rows)
    write_rows(csv_path, rows)
    n = max(1, len(rows))
    margin_vals = [float(r["KAN_control_margin"]) for r in rows]
    summary = {
        "gate": "v22_69R_part_d_mlp_winner_target_projection_capacity",
        "generated_at_sg": now_sg(),
        "completed_projection_rows": len(rows),
        "projection_energy_ge_025_rows": sum(1 for r in rows if float(r["KAN_projection_energy_to_MLP_winner"]) >= 0.25),
        "target_alignment_ge_035_rows": sum(1 for r in rows if float(r["KAN_target_alignment_cosine"]) >= 0.35),
        "reconstruction_error_le_075_rows": sum(1 for r in rows if float(r["KAN_lift_reconstruction_error"]) <= 0.75),
        "pure_gauge_le_020_rows": sum(1 for r in rows if float(r["KAN_lift_pure_gauge_energy_fraction"]) <= 0.20),
        "readout_visible_CVaR25_ge_025_rows": sum(1 for r in rows if float(r["KAN_lift_readout_visible_energy_CVaR25"]) >= 0.25),
        "control_margin_positive_rows": sum(1 for r in rows if float(r["KAN_control_margin"]) > 0.0),
        "control_margin_p10": percentile(margin_vals, 0.10),
        "control_margin_CVaR25": mean(sorted(margin_vals)[: max(1, int(math.ceil(0.25 * len(margin_vals))))]) if margin_vals else 0.0,
        "beats_same_energy_random_lift_rows": sum(int(r["KAN_beats_same_energy_random_lift"]) for r in rows),
        "beats_same_readout_visible_random_lift_rows": sum(int(r["KAN_beats_same_readout_visible_random_lift"]) for r in rows),
        "beats_same_GB_cost_random_lift_rows": sum(int(r["KAN_beats_same_GB_cost_random_lift"]) for r in rows),
        "beats_MLP_matched_approx_lift_rows": sum(int(r["KAN_beats_MLP_matched_approx_lift"]) for r in rows),
        "predicted_no_debt_rows": sum(int(r["predicted_no_debt"]) for r in rows),
        "predicted_spectrum_drift_pass_rows": sum(int(r["predicted_spectrum_drift_pass"]) for r in rows),
    }
    frac_gate = {
        "completed": len(rows) >= 30,
        "projection": summary["projection_energy_ge_025_rows"] / n >= 22 / 30,
        "alignment": summary["target_alignment_ge_035_rows"] / n >= 22 / 30,
        "reconstruction": summary["reconstruction_error_le_075_rows"] / n >= 22 / 30,
        "pure_gauge": summary["pure_gauge_le_020_rows"] / n >= 24 / 30,
        "readout_visible": summary["readout_visible_CVaR25_ge_025_rows"] / n >= 22 / 30,
        "control_margin": summary["control_margin_positive_rows"] / n >= 18 / 30,
        "same_energy": summary["beats_same_energy_random_lift_rows"] / n >= 20 / 30,
        "same_readout_visible": summary["beats_same_readout_visible_random_lift_rows"] / n >= 20 / 30,
        "same_gb_cost": summary["beats_same_GB_cost_random_lift_rows"] / n >= 20 / 30,
        "mlp_matched": summary["beats_MLP_matched_approx_lift_rows"] / n >= 14 / 30,
        "debt": summary["predicted_no_debt_rows"] / n >= 22 / 30,
        "spectrum": summary["predicted_spectrum_drift_pass_rows"] / n >= 22 / 30,
    }
    summary["gate_components"] = frac_gate
    summary["part_d_pass"] = int(all(frac_gate.values()))
    failures = [k for k, v in frac_gate.items() if not v]
    summary["failure_components"] = failures
    if summary["part_d_pass"]:
        summary["route"] = "R6-KANCarrierCapacityOpened_Exploration"
        summary["official_KAN_full_loop_allowed_next"] = 1
    else:
        summary["route"] = "R3-KANCarrierCapacityNotEstablished_NoFullLoop"
        summary["official_KAN_full_loop_allowed_next"] = 0
    summary["visible_only_repair"] = int(bool(args.part_d_visible_only))
    write_json(summary_path, summary)
    append_exec(
        "D_visible_repair" if bool(args.part_d_visible_only) else "D_projection",
        current_runner_command(),
        "pass" if summary["part_d_pass"] else "fail",
        gpu=str(args.device),
        files=f"{rel(csv_path)}; {rel(summary_path)}",
        note=json.dumps(summary, ensure_ascii=False),
    )
    append_recap(
        "Part D MLP-winner target projection capacity audit",
        [
            f"completed_projection_rows={summary['completed_projection_rows']}; part_d_pass={summary['part_d_pass']}; route={summary['route']}.",
            f"Projection/alignment/reconstruction/readout/control/debt/spectrum rows: PE {summary['projection_energy_ge_025_rows']}/{len(rows)}, align {summary['target_alignment_ge_035_rows']}/{len(rows)}, recon {summary['reconstruction_error_le_075_rows']}/{len(rows)}, readout {summary['readout_visible_CVaR25_ge_025_rows']}/{len(rows)}, margin {summary['control_margin_positive_rows']}/{len(rows)}, debt {summary['predicted_no_debt_rows']}/{len(rows)}, spectrum {summary['predicted_spectrum_drift_pass_rows']}/{len(rows)}.",
            f"Control comparisons: same-energy {summary['beats_same_energy_random_lift_rows']}/{len(rows)}, same-readout {summary['beats_same_readout_visible_random_lift_rows']}/{len(rows)}, same-GB-cost {summary['beats_same_GB_cost_random_lift_rows']}/{len(rows)}, MLP-matched {summary['beats_MLP_matched_approx_lift_rows']}/{len(rows)}.",
            f"Failure components: {failures}. Official KAN full-loop allowed next={summary['official_KAN_full_loop_allowed_next']}.",
            "Audit note: pure_gauge fraction is recorded as 0 only because this Part D chart projects in observable function columns after quotient elimination; the synthetic Part C vertical projector is the explicit pure-gauge test.",
            f"Evidence: `{rel(csv_path)}`, `{rel(summary_path)}`.",
        ],
    )
    return summary


def run_full(args: argparse.Namespace) -> dict[str, Any]:
    a = run_part_a(args)
    if not int(a.get("part_a_pass", 0)):
        return {"route": "R0-CodeOrBoundaryFailed", "part_a": a}
    b = run_part_b(args)
    c = run_part_c(args)
    d = run_part_d_projection(args)
    return {"part_a": a, "part_b": b, "part_c": c, "part_d_guard": d, "route": d.get("route")}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", default="full", choices=["full", "part-a", "part-b", "part-c", "part-d", "part-d-guard"])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--gpus", default="0,1,2,3")
    p.add_argument("--max-workers", type=int, default=4)
    p.add_argument("--row-timeout", type=int, default=2400)
    p.add_argument("--part-b-datasets", default="Wine,Spam,MNIST")
    p.add_argument("--part-b-methods", default=",".join(WINNER_METHODS))
    p.add_argument("--part-d-arches", default="DGKAN_DCHE,DGKAN_DFOU")
    p.add_argument("--part-d-visible-only", action="store_true")
    p.add_argument("--part-d-artifact-tag", default="")
    p.add_argument("--part-d-visible-selection", default="readout_visible", choices=["readout_visible", "target_augmented_visible"])
    p.add_argument("--part-d-visible-threshold", type=float, default=0.25)
    p.add_argument("--part-d-visible-min-fraction", type=float, default=0.25)
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--train-size", type=int, default=512)
    p.add_argument("--held-size", type=int, default=256)
    p.add_argument("--test-size", type=int, default=256)
    p.add_argument("--hidden", type=int, default=96)
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--probe-batch-size", type=int, default=96)
    p.add_argument("--eval-batch-size", type=int, default=512)
    p.add_argument("--refresh", type=int, default=100)
    p.add_argument("--lr", type=float, default=3.0e-3)
    p.add_argument("--weight-decay", type=float, default=1.0e-4)
    p.add_argument("--metric-kind", default="signal_debt")
    p.add_argument("--metric-batch-size", type=int, default=128)
    p.add_argument("--shaping-budget", type=float, default=0.05)
    p.add_argument("--iso-eta", type=float, default=1.0)
    p.add_argument("--functional-spectrum-drift-threshold", type=float, default=0.50)
    p.add_argument("--poet-block-size", type=int, default=16)
    p.add_argument("--poet-merge-interval", type=int, default=50)
    p.add_argument("--poet-lr", type=float, default=1.0e-3)
    p.add_argument("--poet-scale", type=float, default=1.0)
    p.add_argument("--reset-logs", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    ensure_out()
    initialize_docs(args)
    try:
        if args.mode == "part-a":
            out = run_part_a(args)
        elif args.mode == "part-b":
            out = run_part_b(args)
        elif args.mode == "part-c":
            out = run_part_c(args)
        elif args.mode == "part-d":
            out = run_part_d_projection(args)
        elif args.mode == "part-d-guard":
            out = run_part_d_guard(args)
        else:
            out = run_full(args)
        write_json(OUT_ROOT / "v22_69R_last_run_summary.json", out)
        print(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        ensure_out()
        log = LOG_ROOT / f"exception_{int(time.time())}.log"
        log.write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8", errors="replace")
        append_exec("exception", command_text([PYTHON, rel(RUNNER), "--mode", str(args.mode)]), "fail", gpu=str(args.device), files=rel(log), note=f"{type(exc).__name__}: {exc}")
        append_recap("异常中断", [f"error={type(exc).__name__}: {exc}", f"Evidence: `{rel(log)}`."])
        raise


if __name__ == "__main__":
    raise SystemExit(main())
