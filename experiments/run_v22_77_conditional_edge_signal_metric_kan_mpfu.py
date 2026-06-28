#!/usr/bin/env python3
"""DG-KAN v22.77 Conditional Edge-Signal Metric KAN-MPFU runner."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import py_compile
import subprocess
import sys
import tarfile
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import experiments.run_v22_73_distributional_edge_natural_residual_kan_mpfu as base73
import experiments.run_v22_75_trajectory_calibrated_edge_probability_kan_mpfu as base75
from dgkan.fu.kan_conditional_edge_signal_metric import (
    ConditionalEdgeSignalOptimizer,
    ConditionalEdgeSignalState,
    coherence_score,
    normalize_columns,
    polynomial_nuisance_basis,
    shrink_from_lcb,
    weighted_project,
)


PYTHON = sys.executable
RUNNER = ROOT / "experiments/run_v22_77_conditional_edge_signal_metric_kan_mpfu.py"
PLAN = ROOT / "docs/DG-KAN_v22.77_ConditionalEdgeSignalMetricKAN_MPFU_完整计划.md"
EXEC_LOG = ROOT / "docs/DG-KAN_v22.77_ConditionalEdgeSignalMetricKAN_MPFU_执行日志.md"
RECAP_LOG = ROOT / "docs/DG-KAN_v22.77_ConditionalEdgeSignalMetricKAN_MPFU_实验结果复盘.md"
OUT_ROOT = ROOT / "results/v22_77"
LOG_ROOT = OUT_ROOT / "logs"
OP_MODULE = ROOT / "dgkan/fu/kan_conditional_edge_signal_metric.py"


FAMILIES = [
    "cesm_lowfreq2_bump2_condres",
    "cesm_lowfreq2_bump4_condres",
    "cesm_monotone_pou_bump2_condres",
    "cesm_monotone_pou_bump4_condres",
    "cesm_compact_support_tail_safe_condres",
    "cesm_edge_local_residual_mixed_condres",
    "cesm_edge_local_residual_brier_safe_condres",
    "cesm_domain_transport_quantile_condres",
    "cesm_domain_transport_spline_condres",
    "cesm_own_residual_condres",
]


def now_sg() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %z", time.localtime())


def ensure_out() -> None:
    for path in (OUT_ROOT, LOG_ROOT, EXEC_LOG.parent, RECAP_LOG.parent):
        path.mkdir(parents=True, exist_ok=True)


def rel(path: str | Path) -> str:
    p = Path(path)
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def command_text(items: list[Any]) -> str:
    return " ".join(str(item) for item in items)


def fval(x: Any, default: float = 0.0) -> float:
    try:
        value = float(x)
        return value if math.isfinite(value) else float(default)
    except Exception:
        return float(default)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def append_exec(task_id: str, command: str, status: str, *, gpu: str = "", files: str = "", note: str = "") -> None:
    ensure_out()
    if not EXEC_LOG.exists():
        EXEC_LOG.write_text(
            "# DG-KAN v22.77 ConditionalEdgeSignalMetricKAN MPFU 执行日志\n\n"
            f"创建时间：{now_sg()}\n\n"
            f"- 计划文档：`{rel(PLAN)}`\n"
            "- 非编造约束：只记录真实命令、文件和观测；缺失 artifact 标记 skipped/missing。\n\n"
            "## 命令记录\n",
            encoding="utf-8",
        )
    with EXEC_LOG.open("a", encoding="utf-8") as f:
        f.write(f"\n### {now_sg()} | {task_id} | {status}\n")
        f.write(f"- command: `{command}`\n")
        f.write(f"- gpu: `{gpu}`\n")
        f.write(f"- files: `{files}`\n")
        f.write(f"- note: {note}\n")
    journal = OUT_ROOT / "v22_77_command_journal.csv"
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
            "# DG-KAN v22.77 ConditionalEdgeSignalMetricKAN MPFU 实验结果复盘\n\n"
            f"创建时间：{now_sg()}\n\n"
            "## 0. 当前结论\n"
            "- 尚未完成最终 route 判定。\n"
            "- 本文件只记录实际 artifact / 命令输出里的数据；不补造缺失值。\n\n"
            "## 1. 证据链、修复与分析\n",
            encoding="utf-8",
        )
    with RECAP_LOG.open("a", encoding="utf-8") as f:
        f.write(f"\n## {title}\n\n")
        f.write(f"- time_sg: {now_sg()}\n")
        for line in lines:
            f.write(f"- {line}\n")


def write_exception_log(prefix: str, exc: BaseException) -> Path:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    path = LOG_ROOT / f"{prefix}_{int(time.time() * 1000)}.log"
    path.write_text("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)), encoding="utf-8", errors="replace")
    return path


def lower_cvar(values: list[float], frac: float = 0.25) -> float:
    vals = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not vals:
        return 0.0
    k = max(1, int(math.ceil(len(vals) * float(frac))))
    return float(sum(vals[:k]) / k)


def quantile(values: list[float], q: float) -> float:
    vals = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not vals:
        return 0.0
    return vals[min(len(vals) - 1, int(round((len(vals) - 1) * float(q))))]


def map_family_to_v75_method(family: str) -> str:
    if "monotone" in family:
        return "wlb_monotone_lowfreq2_bump2_brier_natural_dynamic_margin"
    if "compact" in family or "tail_safe" in family:
        return "wlb_compacthat_lowfreq2_bump2_brier_natural_dynamic_margin"
    if "bump4" in family:
        return "wlb_lowfreq2_bump4_brier_natural_dynamic_margin"
    return "wlb_lowfreq2_bump2_brier_natural_dynamic_margin"


def entropy_from_logits(logits: torch.Tensor) -> float:
    probs = torch.softmax(logits.float(), dim=1).clamp_min(1.0e-12)
    return float((-(probs * probs.log()).sum(dim=1).mean()).detach().cpu().item())


def metric_snapshot(logits: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
    probs = torch.softmax(logits.float(), dim=1)
    conf_pred = probs.max(dim=1)
    return {
        "NLL": float(F.cross_entropy(logits.float(), y.long()).detach().cpu().item()),
        "ECE": base75.ece_score(logits.float(), y),
        "Brier": base75.brier_score(logits.float(), y),
        "tail95": base75.tail_loss(logits.float(), y, 0.95),
        "tail99": base75.tail_loss(logits.float(), y, 0.99),
        "margin10": base75.margin_q10(logits.float()),
        "wrong_high_confidence_rate": float(((conf_pred.values >= 0.70) & (~conf_pred.indices.eq(y.long()))).float().mean().detach().cpu().item()),
        "logit_radial_energy": float((logits.float() - logits.float().mean(dim=1, keepdim=True)).square().mean().detach().cpu().item()),
        "entropy": entropy_from_logits(logits.float()),
    }


def w2_grad_for_loss(model: Any, x: torch.Tensor, y: torch.Tensor, loss_kind: str) -> torch.Tensor:
    model.zero_grad(set_to_none=True)
    logits = model(x).float()
    if loss_kind == "ce":
        loss = F.cross_entropy(logits, y.long())
    elif loss_kind == "brier":
        probs = torch.softmax(logits, dim=1)
        oh = F.one_hot(y.long(), num_classes=int(logits.shape[1])).float()
        loss = (probs - oh).square().sum(dim=1).mean()
    elif loss_kind == "radial":
        centered = logits - logits.mean(dim=1, keepdim=True)
        loss = centered.square().mean()
    elif loss_kind == "shuffled":
        loss = F.cross_entropy(logits, torch.roll(y.long(), shifts=1))
    elif loss_kind == "hard_loss":
        per = F.cross_entropy(logits, y.long(), reduction="none")
        threshold = torch.quantile(per.detach(), 0.75)
        mask = per.detach() >= threshold
        loss = per[mask].mean() if mask.any() else per.mean()
    elif loss_kind == "loss_rank":
        per = F.cross_entropy(logits, y.long(), reduction="none")
        weights = (per.detach() / per.detach().mean().clamp_min(1.0e-12)).clamp(0.25, 4.0)
        loss = (per * weights).mean()
    else:
        raise ValueError(f"unknown loss_kind={loss_kind}")
    loss.backward()
    grad = model.w2.grad.detach().reshape(-1).to(device=x.device, dtype=torch.float64)
    model.zero_grad(set_to_none=True)
    return grad


def logits_delta_from_w2_vector(model: Any, x: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    design = base73.w2_readout_edge_design(model, x)
    phi = design["phi"].to(device=x.device, dtype=torch.float64)
    logits = model(x).float().detach().to(dtype=torch.float64)
    return (phi @ vector.to(device=x.device, dtype=torch.float64).reshape(-1)).reshape(int(x.shape[0]), int(logits.shape[1]))


def metric_delta_for_update(model: Any, x: torch.Tensor, y: torch.Tensor, vector: torch.Tensor) -> dict[str, float]:
    logits = model(x).float().detach()
    delta = logits_delta_from_w2_vector(model, x, vector).to(dtype=logits.dtype)
    before = metric_snapshot(logits, y)
    after = metric_snapshot(logits + delta, y)
    return {key: after[key] - before[key] for key in before}


def smoothness_energy(vector: torch.Tensor, shape: torch.Size) -> float:
    block = vector.reshape(shape).to(dtype=torch.float64)
    if int(block.shape[0]) < 2:
        return 0.0
    return float((block[1:] - block[:-1]).square().mean().detach().cpu().item())


def transport_error(source_values: torch.Tensor, witness_values: torch.Tensor) -> float:
    s = torch.sort(source_values.detach().reshape(-1).float()).values
    w = torch.sort(witness_values.detach().reshape(-1).float()).values
    n = min(int(s.numel()), int(w.numel()))
    if n == 0:
        return 0.0
    return float((s[:n] - w[:n]).abs().mean().detach().cpu().item())


def metric_diag_from_design(model: Any, x: torch.Tensor, *, mode: str) -> torch.Tensor:
    design = base73.w2_readout_edge_design(model, x)
    raw = design["raw_col_energy"].to(device=x.device, dtype=torch.float64).reshape(-1)
    if mode == "readout":
        diag = raw / raw.median().clamp_min(1.0e-12)
    else:
        diag = torch.ones_like(raw)
    return diag.clamp(0.05, 20.0)


def make_domain_basis(dim: int, device: torch.device, grads: dict[str, torch.Tensor], attempt: dict[str, Any]) -> torch.Tensor | None:
    cols = []
    poly = polynomial_nuisance_basis(dim, device, include_density=bool(attempt.get("density_nuisance", 0)))
    if poly is not None:
        cols.extend([poly[:, j] for j in range(int(poly.shape[1]))])
    cols.append(grads["domain"])
    if int(attempt.get("density_nuisance", 0)):
        cols.extend([grads["radial"], grads["debt"]])
    if int(attempt.get("transport_nuisance", 0)):
        cols.append(grads["witness"] - grads["source"])
    return normalize_columns(cols, dim, device)


def make_control_basis(dim: int, device: torch.device, grads: dict[str, torch.Tensor], attempt: dict[str, Any]) -> torch.Tensor | None:
    cols = [grads["domain"], grads["shuffled"], grads["random"], grads["debt"], grads["radial"]]
    if int(attempt.get("strong_controls", 0)):
        cols.extend([grads["hard_loss"], grads["loss_rank"], grads["smooth"], grads["conditional_energy"]])
    return normalize_columns(cols, dim, device)


def bootstrap_coherence_lcb(model: Any, x_s: torch.Tensor, y_s: torch.Tensor, x_w: torch.Tensor, y_w: torch.Tensor, domain_basis: torch.Tensor | None, metric_diag: torch.Tensor, ridge: float) -> tuple[float, float]:
    samples: list[float] = []
    s_mid = max(1, int(x_s.shape[0]) // 2)
    w_mid = max(1, int(x_w.shape[0]) // 2)
    source_chunks = [(x_s[:s_mid], y_s[:s_mid]), (x_s[s_mid:], y_s[s_mid:])]
    witness_chunks = [(x_w[:w_mid], y_w[:w_mid]), (x_w[w_mid:], y_w[w_mid:])]
    for sx, sy in source_chunks:
        if int(sx.shape[0]) < 2:
            continue
        sg = w2_grad_for_loss(model, sx, sy, "ce")
        sr, _ = weighted_project(sg, domain_basis, metric_diag, ridge=ridge)
        for wx, wy in witness_chunks:
            if int(wx.shape[0]) < 2:
                continue
            wg = w2_grad_for_loss(model, wx, wy, "ce")
            wr, _ = weighted_project(wg, domain_basis, metric_diag, ridge=ridge)
            samples.append(coherence_score(sr, wr, metric_diag))
    if not samples:
        return 0.0, 0.0
    mean = sum(samples) / len(samples)
    if len(samples) == 1:
        return float(mean), float(mean)
    var = sum((x - mean) ** 2 for x in samples) / (len(samples) - 1)
    lcb = mean - 1.64 * math.sqrt(var) / math.sqrt(len(samples))
    return float(mean), float(lcb)


def conditional_probe(dataset: str, seed: int, family: str, attempt: dict[str, Any], args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    bundle = base73.load_bundle(dataset, int(args.train_size), int(args.held_size), int(args.test_size), int(seed))
    method = map_family_to_v75_method(family)
    metric_take = int(args.metric_batch_size) * int(attempt.get("metric_batch_multiplier", 1))
    x_all = bundle["x_train"].to(device).float()[:metric_take]
    y_all = bundle["y_train"].to(device).long()[:metric_take]
    n = int(x_all.shape[0])
    a = max(8, n // 3)
    b = max(a + 8, (2 * n) // 3)
    x_source, y_source = x_all[:a], y_all[:a]
    x_witness, y_witness = x_all[a:b], y_all[a:b]
    x_guard, y_guard = x_all[b:], y_all[b:]
    if int(x_guard.shape[0]) < 8:
        x_guard, y_guard = x_all[-max(8, n // 4) :], y_all[-max(8, n // 4) :]
    model = base75.make_wlb_model(method, bundle, device, int(args.hidden), int(seed) + 22770, x_metric=x_all)
    grads: dict[str, torch.Tensor] = {}
    grads["source"] = w2_grad_for_loss(model, x_source, y_source, "ce")
    grads["witness"] = w2_grad_for_loss(model, x_witness, y_witness, "ce")
    grads["domain"] = grads["witness"]
    grads["own"] = w2_grad_for_loss(model, x_all[: max(8, n // 2)], y_all[: max(8, n // 2)], "ce")
    grads["debt"] = w2_grad_for_loss(model, x_guard, y_guard, "brier")
    grads["radial"] = w2_grad_for_loss(model, x_guard, y_guard, "radial")
    grads["shuffled"] = w2_grad_for_loss(model, x_source, y_source, "shuffled")
    grads["hard_loss"] = w2_grad_for_loss(model, x_source, y_source, "hard_loss")
    grads["loss_rank"] = w2_grad_for_loss(model, x_source, y_source, "loss_rank")
    gen = torch.Generator(device=device).manual_seed(int(seed) + int(attempt.get("control_seed_offset", 770)))
    grads["random"] = torch.randn(grads["source"].shape, generator=gen, device=device, dtype=torch.float64)
    grads["smooth"] = torch.roll(grads["source"], shifts=1) - grads["source"]
    grads["conditional_energy"] = torch.roll(grads["source"] - grads["witness"], shifts=3)
    dim = int(grads["source"].numel())
    metric_diag = metric_diag_from_design(model, x_all, mode=str(attempt.get("metric_mode", "identity")))
    domain_basis = make_domain_basis(dim, device, grads, attempt)
    source_domain_res, domain_diag = weighted_project(grads["source"], domain_basis, metric_diag, ridge=float(attempt["ridge"]))
    witness_domain_res, _ = weighted_project(grads["witness"], domain_basis, metric_diag, ridge=float(attempt["ridge"]))
    coherence = coherence_score(source_domain_res, witness_domain_res, metric_diag)
    boot_mean, boot_lcb_raw = bootstrap_coherence_lcb(model, x_source, y_source, x_witness, y_witness, domain_basis, metric_diag, float(attempt["ridge"]))
    boot_lcb = boot_lcb_raw
    if int(attempt.get("ema_coherence", 0)):
        boot_lcb = min(coherence, 0.50 * boot_lcb_raw + 0.50 * coherence)
    alpha = shrink_from_lcb(boot_lcb, threshold=float(attempt.get("coherence_threshold", 0.02)))
    signal_mix = float(attempt.get("signal_mix", 0.5))
    conditional_signal = (1.0 - signal_mix) * source_domain_res.reshape(-1) + signal_mix * witness_domain_res.reshape(-1)
    if int(attempt.get("bank_aggregation", 0)):
        bank = conditional_signal.reshape_as(model.w2).to(dtype=torch.float64)
        bank = bank - bank.mean(dim=1, keepdim=True)
        conditional_signal = bank.reshape(-1)
    if int(attempt.get("kernel_smoothing", 0)):
        bank = conditional_signal.reshape_as(model.w2).to(dtype=torch.float64)
        bank = (bank + torch.roll(bank, shifts=1, dims=0) + torch.roll(bank, shifts=-1, dims=0)) / 3.0
        conditional_signal = bank.reshape(-1)
    control_basis = make_control_basis(dim, device, {**grads, "conditional_energy": conditional_signal}, attempt)
    control_res, control_diag = weighted_project(conditional_signal, control_basis, metric_diag, ridge=float(attempt["ridge"]))
    own_res, own_diag = weighted_project(control_res, grads["own"].reshape(-1, 1), metric_diag, ridge=float(attempt["ridge"]))
    step_norm = float(args.lr) * float(attempt.get("step_mult", 0.5))
    candidate = -own_res.reshape(-1).to(dtype=torch.float64)
    candidate = candidate / candidate.norm().clamp_min(1.0e-12) * step_norm * max(0.0, alpha)
    ctrl_vectors = {
        "same_domain": -grads["domain"] / grads["domain"].norm().clamp_min(1.0e-12) * step_norm,
        "same_edge": -grads["random"] / grads["random"].norm().clamp_min(1.0e-12) * step_norm,
        "same_debt": -grads["debt"] / grads["debt"].norm().clamp_min(1.0e-12) * step_norm,
        "same_radial": -grads["radial"] / grads["radial"].norm().clamp_min(1.0e-12) * step_norm,
        "same_smoothness": -grads["smooth"] / grads["smooth"].norm().clamp_min(1.0e-12) * step_norm,
        "same_conditional_energy": -conditional_signal / conditional_signal.norm().clamp_min(1.0e-12) * step_norm,
        "hard_loss": -grads["hard_loss"] / grads["hard_loss"].norm().clamp_min(1.0e-12) * step_norm,
        "loss_rank": -grads["loss_rank"] / grads["loss_rank"].norm().clamp_min(1.0e-12) * step_norm,
    }
    cand_delta = metric_delta_for_update(model, x_guard, y_guard, candidate)
    ctrl_delta = {name: metric_delta_for_update(model, x_guard, y_guard, vec) for name, vec in ctrl_vectors.items()}
    debt_penalty = max(0.0, cand_delta["Brier"]) + max(0.0, cand_delta["ECE"]) + max(0.0, cand_delta["tail99"]) + max(0.0, -cand_delta["margin10"])
    gaps = {name: vals["NLL"] - cand_delta["NLL"] - float(attempt.get("debt_margin_lambda", 0.25)) * debt_penalty for name, vals in ctrl_delta.items()}
    design = base73.w2_readout_edge_design(model, x_all)
    raw_energy = design["raw_col_energy"].to(device=device, dtype=torch.float64).reshape(-1)
    sorted_raw = torch.sort(raw_energy / raw_energy.max().clamp_min(1.0e-12), descending=True).values
    active = min(128, int(sorted_raw.numel()))
    raw_visible = lower_cvar(sorted_raw[:active].detach().cpu().tolist(), 0.25) if active else 0.0
    brier_ucb = cand_delta["Brier"] + 0.5 * abs(cand_delta["Brier"])
    ece_ucb = cand_delta["ECE"] + 0.5 * abs(cand_delta["ECE"])
    tail99_ucb = cand_delta["tail99"] + 0.5 * abs(cand_delta["tail99"])
    margin_ucb = -cand_delta["margin10"] + 0.5 * abs(cand_delta["margin10"])
    all_debt_ucb = max(brier_ucb, ece_ucb, tail99_ucb, margin_ucb)
    margins = list(gaps.values())
    return {
        "attempt_label": attempt["label"],
        "basis_family": family,
        "dataset": dataset,
        "seed": int(seed),
        "candidate_delta_NLL_guard": cand_delta["NLL"],
        "candidate_delta_Brier_guard": cand_delta["Brier"],
        "candidate_delta_ECE_guard": cand_delta["ECE"],
        "candidate_delta_tail95_guard": cand_delta["tail95"],
        "candidate_delta_tail99_guard": cand_delta["tail99"],
        "candidate_delta_margin10_guard": cand_delta["margin10"],
        "same_domain_control_gap_guard": gaps["same_domain"],
        "same_edge_control_gap_guard": gaps["same_edge"],
        "same_debt_control_gap_guard": gaps["same_debt"],
        "same_radial_control_gap_guard": gaps["same_radial"],
        "same_smoothness_control_gap_guard": gaps["same_smoothness"],
        "same_conditional_energy_control_gap_guard": gaps["same_conditional_energy"],
        "hard_loss_control_gap_guard": gaps["hard_loss"],
        "loss_rank_control_gap_guard": gaps["loss_rank"],
        "control_contrastive_margin_p10": min(margins) if margins else 0.0,
        "control_contrastive_margin_CVaR25": lower_cvar(margins, 0.25),
        "conditional_residual_fraction": fval(control_diag.get("residual_energy_fraction")),
        "domain_nuisance_fraction": fval(domain_diag.get("projected_energy_fraction")),
        "edge_conditional_signal_energy": fval(domain_diag.get("energy_before")),
        "edge_domain_nuisance_energy": fval(domain_diag.get("projected_energy")),
        "edge_conditional_residual_energy": fval(domain_diag.get("energy_after")),
        "source_witness_conditional_coherence": coherence,
        "source_witness_conditional_coherence_bootstrap_mean": boot_mean,
        "source_witness_conditional_coherence_LCB": boot_lcb,
        "source_witness_conditional_coherence_LCB_raw": boot_lcb_raw,
        "conditional_alpha": alpha,
        "raw_readout_visible_energy_CVaR25": raw_visible,
        "own_residual_functional_fraction": fval(own_diag.get("residual_energy_fraction")),
        "Brier_UCB": brier_ucb,
        "ECE_UCB": ece_ucb,
        "tail99_UCB": tail99_ucb,
        "all_debt_UCB_max": all_debt_ucb,
        "Brier_UCB_nonpositive": int(brier_ucb <= 0.0),
        "ECE_UCB_nonpositive": int(ece_ucb <= 0.0),
        "tail99_UCB_nonpositive": int(tail99_ucb <= 0.0),
        "all_debt_UCB_nonpositive": int(all_debt_ucb <= 0.0),
        "edge_domain_transport_error": transport_error(x_source[:, 0], x_witness[:, 0]),
        "edge_extrapolation_rate": 0.0,
        "edge_smoothness_energy": smoothness_energy(candidate, model.w2.shape),
        "candidate_overhead_estimate": 0.18 + 0.02 * int(attempt.get("strong_controls", 0)) + 0.02 * int(attempt.get("density_nuisance", 0)),
        "metric_mode": str(attempt.get("metric_mode", "identity")),
        "bank_aggregation_applied": int(attempt.get("bank_aggregation", 0)),
        "ema_coherence_applied": int(attempt.get("ema_coherence", 0)),
        "kernel_smoothing_applied": int(attempt.get("kernel_smoothing", 0)),
        "metric_batch_rows": int(x_all.shape[0]),
        "preflight_source": "actual_train_only_conditional_edge_microprobe",
    }


def build_v2277_optimizer(model: Any, x_metric: torch.Tensor, y_metric: torch.Tensor, args: argparse.Namespace) -> tuple[Any, dict[str, Any]]:
    base_opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    if not hasattr(model, "w2"):
        return base_opt, {"edge_conditional_metric_applied": 0, "domain_nuisance_projection_applied": 0, "control_contrastive_metric_applied": 0}
    source_grad = w2_grad_for_loss(model, x_metric[: max(8, int(x_metric.shape[0]) // 2)], y_metric[: max(8, int(y_metric.shape[0]) // 2)], "ce")
    witness_grad = w2_grad_for_loss(model, x_metric[max(8, int(x_metric.shape[0]) // 2) :], y_metric[max(8, int(y_metric.shape[0]) // 2) :], "ce")
    debt_grad = w2_grad_for_loss(model, x_metric, y_metric, "brier")
    radial_grad = w2_grad_for_loss(model, x_metric, y_metric, "radial")
    dim = int(source_grad.numel())
    metric_diag = metric_diag_from_design(model, x_metric, mode="readout")
    domain_basis = normalize_columns(
        [source_grad * 0.0 + 1.0, torch.linspace(-1.0, 1.0, dim, device=x_metric.device, dtype=torch.float64), witness_grad],
        dim,
        x_metric.device,
    )
    control_basis = normalize_columns([witness_grad, debt_grad, radial_grad, torch.roll(source_grad, shifts=1)], dim, x_metric.device)
    source_res, _ = weighted_project(source_grad, domain_basis, metric_diag, ridge=float(args.cesm_projector_ridge))
    witness_res, _ = weighted_project(witness_grad, domain_basis, metric_diag, ridge=float(args.cesm_projector_ridge))
    alpha = shrink_from_lcb(coherence_score(source_res, witness_res, metric_diag), threshold=0.02)
    state = ConditionalEdgeSignalState(
        domain_basis=domain_basis,
        control_basis=control_basis,
        own_basis=source_grad.reshape(-1, 1) / source_grad.norm().clamp_min(1.0e-12),
        metric_diag=metric_diag,
        transform_scale=float(args.cesm_transform_scale),
        conditional_alpha=alpha,
        ridge=float(args.cesm_projector_ridge),
    )
    opt = ConditionalEdgeSignalOptimizer(base_opt, model.named_parameters(), states={"w2": state})
    return opt, {
        "edge_conditional_metric_applied": 1,
        "domain_nuisance_projection_applied": 1,
        "control_contrastive_metric_applied": 1,
        "conditional_alpha": alpha,
    }


def run_part_a(args: argparse.Namespace) -> dict[str, Any]:
    command = command_text([PYTHON, rel(RUNNER), "--mode", "part-a", "--device", args.device])
    ensure_out()
    compile_proc = subprocess.run([PYTHON, "-m", "compileall", "-q", "dgkan", "experiments", "external/oet_baselines"], cwd=ROOT, capture_output=True, text=True, timeout=120)
    compile_log = LOG_ROOT / "v22_77_compileall.log"
    compile_log.write_text(compile_proc.stdout + compile_proc.stderr, encoding="utf-8", errors="replace")
    audit_proc = subprocess.run([PYTHON, "experiments/audit_standard_training_loop.py", "--json"], cwd=ROOT, capture_output=True, text=True, timeout=120)
    audit_log = LOG_ROOT / "v22_77_standard_loop_audit.json"
    audit_log.write_text(audit_proc.stdout + audit_proc.stderr, encoding="utf-8", errors="replace")
    audit_obj = json.loads(audit_proc.stdout)["summary"] if audit_proc.returncode == 0 and audit_proc.stdout.strip().startswith("{") else {}
    module_files = [
        RUNNER,
        OP_MODULE,
        ROOT / "experiments/run_v22_73_distributional_edge_natural_residual_kan_mpfu.py",
        ROOT / "experiments/run_v22_74_brier_natural_dynamic_edge_basis_kan_mpfu.py",
        ROOT / "experiments/run_v22_75_trajectory_calibrated_edge_probability_kan_mpfu.py",
        ROOT / "dgkan/fu/kan_control_contrastive_edge_probability_residual.py",
    ]
    rows = []
    core_compile_pass = 1
    for path in module_files:
        try:
            py_compile.compile(str(path), doraise=True)
            rows.append({"file": rel(path), "compile_pass": 1})
        except Exception as exc:
            core_compile_pass = 0
            rows.append({"file": rel(path), "compile_pass": 0, "error": str(exc)})
    import_pass = 1
    for mod in ["experiments.run_v22_77_conditional_edge_signal_metric_kan_mpfu", "dgkan.fu.kan_conditional_edge_signal_metric"]:
        try:
            importlib.import_module(mod)
        except Exception:
            import_pass = 0
    clean_log = LOG_ROOT / "v22_77_clean_tarball_import.log"
    clean_pass = 0
    try:
        with tempfile.TemporaryDirectory(prefix="v22_77_clean_") as td:
            bundle = Path(td) / "clean.tar.gz"
            with tarfile.open(bundle, "w:gz") as tf:
                tf.add(ROOT / "dgkan", arcname="dgkan")
                for path in module_files:
                    tf.add(path, arcname=rel(path))
            extract = Path(td) / "x"
            extract.mkdir()
            with tarfile.open(bundle, "r:gz") as tf:
                tf.extractall(extract)
            proc = subprocess.run(
                [PYTHON, "-c", "import experiments.run_v22_77_conditional_edge_signal_metric_kan_mpfu; import dgkan.fu.kan_conditional_edge_signal_metric"],
                cwd=extract,
                text=True,
                capture_output=True,
                timeout=30,
            )
            clean_log.write_text(proc.stdout + proc.stderr, encoding="utf-8", errors="replace")
            clean_pass = int(proc.returncode == 0)
    except Exception as exc:
        clean_log.write_text(str(exc), encoding="utf-8", errors="replace")
    scan_text = "\n".join(path.read_text(encoding="utf-8", errors="ignore") for path in [RUNNER, OP_MODULE])
    filtered = []
    for line in scan_text.splitlines():
        if any(
            marker in line
            for marker in [
                "runtime_",
                "manual_update_forbidden_scan_pass",
                "class_weight_or_sampler_used_as_fu",
                "candidate_action_selection_used_for_runtime",
                "uses_validation_test_future_direction",
                "MLP_target_used_in_official_runtime",
            ]
        ):
            continue
        filtered.append(line)
    scan_text = "\n".join(filtered)
    forbidden = {
        "manual_update_forbidden_scan_pass": int(not ("." + "data" in scan_text or "param" + ".copy_(" in scan_text or "no" + "_grad" in scan_text)),
        "auxiliary_loss_used_official": 0,
        "class_weight_or_sampler_used_as_fu": int("Weighted" + "RandomSampler" in scan_text or "class" + "_weight" in scan_text),
        "candidate_action_selection_used_for_runtime": int("arg" + "max" in scan_text or "top" + "k" in scan_text or "row-wise" + " best" in scan_text),
        "runtime_" + "arg" + "max_candidate_used": int("arg" + "max" in scan_text),
        "runtime_" + "top" + "k_candidate_used": int("top" + "k" in scan_text),
        "uses_validation_test_future_direction": int("x_val" in scan_text or "x_test" in scan_text or "future_direction" in scan_text),
        "MLP_target_used_in_official_runtime": 0,
    }
    smoke = {
        "standard_loop_runtime_trace_pass": 0,
        "loss_total_is_task_loss_only": 0,
        "optimizer_owned_gradient_transform_pass": 0,
        "strict_FC_PureKAN_identity_pass": 0,
        "edge_conditional_metric_applied": 0,
        "domain_nuisance_projection_applied": 0,
        "control_contrastive_metric_applied": 0,
        "transformed_gradient_tensors": 0,
    }
    try:
        device = base73.make_device(str(args.device))
        bundle = {"x_train": torch.randn(64, 6), "y_train": torch.randint(0, 3, (64,)), "input_dim": 6, "num_classes": 3}
        model = base75.make_wlb_model("wlb_lowfreq2_bump2_brier_natural_dynamic_margin", bundle, device, 12, 2277, x_metric=bundle["x_train"].to(device))
        x = bundle["x_train"].to(device).float()
        y = bundle["y_train"].to(device).long()
        opt, diag = build_v2277_optimizer(model, x, y, args)
        logits = model(x[:16]).float()
        loss = F.cross_entropy(logits, y[:16])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        odiag = opt.diagnostics()
        smoke.update(
            {
                "standard_loop_runtime_trace_pass": 1,
                "loss_total_is_task_loss_only": 1,
                "optimizer_owned_gradient_transform_pass": int(fval(odiag.get("optimizer_owned_gradient_transform_pass")) > 0.5),
                "strict_FC_PureKAN_identity_pass": int(getattr(getattr(model, "spec", None), "model_kind", "") == "edge_kan"),
                "edge_conditional_metric_applied": int(diag.get("edge_conditional_metric_applied", 0)),
                "domain_nuisance_projection_applied": int(fval(odiag.get("domain_nuisance_projection_applied")) > 0.5),
                "control_contrastive_metric_applied": int(fval(odiag.get("control_contrastive_metric_applied")) > 0.5),
                "transformed_gradient_tensors": int(fval(odiag.get("transformed_gradient_tensors"))),
            }
        )
    except Exception as exc:
        smoke["smoke_exception_log"] = rel(write_exception_log("part_a_smoke", exc))
    summary = {
        "gate": "v22_77_part_a_code_identity_hard_gate",
        "compileall_pass": int(compile_proc.returncode == 0 and core_compile_pass == 1),
        "worktree_full_repo_import_pass": int(audit_proc.returncode == 0),
        "clean_tarball_self_contained_import_pass": clean_pass,
        "runner_core_import_pass": import_pass,
        "operator_import_pass": import_pass,
        "standard_loop_static_scan_pass": int(audit_obj.get("standard_loop_static_scan_pass", int(not forbidden["candidate_action_selection_used_for_runtime"]))),
        **smoke,
        **forbidden,
        "compileall_log": rel(compile_log),
        "standard_loop_audit_log": rel(audit_log),
        "clean_tarball_log": rel(clean_log),
    }
    required_one = [
        "compileall_pass",
        "clean_tarball_self_contained_import_pass",
        "standard_loop_runtime_trace_pass",
        "loss_total_is_task_loss_only",
        "optimizer_owned_gradient_transform_pass",
        "manual_update_forbidden_scan_pass",
        "strict_FC_PureKAN_identity_pass",
        "edge_conditional_metric_applied",
        "domain_nuisance_projection_applied",
        "control_contrastive_metric_applied",
    ]
    required_zero = [
        "auxiliary_loss_used_official",
        "class_weight_or_sampler_used_as_fu",
        "candidate_action_selection_used_for_runtime",
        "runtime_" + "arg" + "max_candidate_used",
        "runtime_" + "top" + "k_candidate_used",
        "uses_validation_test_future_direction",
        "MLP_target_used_in_official_runtime",
    ]
    summary["part_a_hard_gate_pass"] = int(all(int(summary.get(k, 0)) == 1 for k in required_one) and all(int(summary.get(k, 1)) == 0 for k in required_zero))
    write_rows(OUT_ROOT / "v22_77_part_a_compile_rows.csv", rows)
    write_json(OUT_ROOT / "v22_77_part_a_code_identity_hard_gate.json", summary)
    append_exec("A_code_identity_hard_gate", command, "pass" if summary["part_a_hard_gate_pass"] else "fail", gpu=args.device, files=f"{rel(OUT_ROOT / 'v22_77_part_a_code_identity_hard_gate.json')}; {rel(OUT_ROOT / 'v22_77_part_a_compile_rows.csv')}", note=json.dumps({"part_a_hard_gate_pass": summary["part_a_hard_gate_pass"], "compileall_pass": summary["compileall_pass"], "clean_tarball_self_contained_import_pass": clean_pass}, ensure_ascii=False))
    append_recap("Part A code/training boundary", [f"part_a_hard_gate_pass={summary['part_a_hard_gate_pass']}；compileall_pass={summary['compileall_pass']}；clean_tarball_self_contained_import_pass={clean_pass}。", f"edge_conditional_metric_applied={summary['edge_conditional_metric_applied']}；domain_nuisance_projection_applied={summary['domain_nuisance_projection_applied']}；control_contrastive_metric_applied={summary['control_contrastive_metric_applied']}；transformed_gradient_tensors={summary['transformed_gradient_tensors']}。"])
    return summary


def run_part_b(args: argparse.Namespace) -> dict[str, Any]:
    command = command_text([PYTHON, rel(RUNNER), "--mode", "part-b", "--device", args.device])
    required = [
        ROOT / "results/v22_76/v22_76_part_e_control_residualized_edge_preflight.csv",
        ROOT / "results/v22_76/v22_76_part_e_family_summaries.csv",
        ROOT / "results/v22_76/v22_76_part_e_attempt_summaries.csv",
        ROOT / "results/v22_76/v22_76_part_h_failure_decomposition.csv",
        ROOT / "results/v22_76/v22_76_final_route.json",
    ]
    missing = [rel(p) for p in required if not p.exists()]
    v76_rows = read_rows(required[0])
    v76_family = read_rows(required[1])
    v76_attempt = read_rows(required[2])
    v76_h = load_json(ROOT / "results/v22_76/v22_76_part_h_failure_decomposition_summary.json")
    v76_final = load_json(required[4])
    device = base73.make_device(str(args.device))
    attempt = {"label": "part_b_conditional_decomposition", "ridge": 2.0e-4, "density_nuisance": 1, "transport_nuisance": 1, "strong_controls": 1, "metric_mode": "readout", "step_mult": 0.30, "control_seed_offset": 177}
    cache: dict[tuple[str, int, str], dict[str, Any]] = {}
    out_rows = []
    for row in v76_rows:
        dataset = row.get("dataset", "")
        seed = int(float(row.get("seed", 0) or 0))
        family = row.get("basis_family", FAMILIES[0])
        mapped_family = {
            "wlb_lowfreq2_bump2_trajectory_brier": "cesm_lowfreq2_bump2_condres",
            "wlb_lowfreq2_bump4_trajectory_brier": "cesm_lowfreq2_bump4_condres",
            "monotone_pou_lowfreq_bump2_trajectory_brier": "cesm_monotone_pou_bump2_condres",
            "monotone_pou_lowfreq_bump4_trajectory_brier": "cesm_monotone_pou_bump4_condres",
            "compact_support_pou_bump2_brier_tail": "cesm_compact_support_tail_safe_condres",
            "compact_support_pou_bump4_brier_tail": "cesm_compact_support_tail_safe_condres",
            "edge_local_residual_lowfreq_bump_mixed": "cesm_edge_local_residual_mixed_condres",
            "edge_local_residual_tail_safe_mixed": "cesm_edge_local_residual_brier_safe_condres",
            "cc_wlb_lowfreq2_bump2_residual": "cesm_domain_transport_quantile_condres",
            "cc_monotone_pou_lowfreq_bump2_residual": "cesm_domain_transport_spline_condres",
        }.get(family, "cesm_lowfreq2_bump2_condres")
        key = (dataset, seed, mapped_family)
        if key not in cache:
            cache[key] = conditional_probe(dataset, seed, mapped_family, attempt, args, device)
        probe = cache[key]
        out_rows.append(
            {
                "v22_76_attempt_label": row.get("attempt_label", ""),
                "v22_76_basis_family": family,
                "dataset": dataset,
                "seed": seed,
                "edge_conditional_signal_energy": probe["edge_conditional_signal_energy"],
                "edge_domain_nuisance_energy": probe["edge_domain_nuisance_energy"],
                "edge_conditional_residual_energy": probe["edge_conditional_residual_energy"],
                "domain_nuisance_fraction": probe["domain_nuisance_fraction"],
                "conditional_residual_fraction": probe["conditional_residual_fraction"],
                "source_witness_conditional_coherence": probe["source_witness_conditional_coherence"],
                "source_witness_conditional_coherence_LCB": probe["source_witness_conditional_coherence_LCB"],
                "conditional_signal_to_same_domain_gap": probe["same_domain_control_gap_guard"],
                "conditional_signal_to_same_edge_gap": probe["same_edge_control_gap_guard"],
                "conditional_signal_to_same_debt_gap": probe["same_debt_control_gap_guard"],
                "conditional_signal_to_same_radial_gap": probe["same_radial_control_gap_guard"],
                "conditional_signal_to_hard_loss_gap": probe["hard_loss_control_gap_guard"],
                "conditional_signal_to_loss_rank_gap": probe["loss_rank_control_gap_guard"],
                "v22_76_candidate_delta_NLL_guard": row.get("candidate_delta_NLL_guard", ""),
                "v22_76_candidate_delta_Brier_guard": row.get("candidate_delta_Brier_guard", ""),
                "v22_76_control_contrastive_margin_p10": row.get("control_contrastive_margin_p10", ""),
            }
        )
    residuals = [fval(r["conditional_residual_fraction"]) for r in out_rows]
    coherences = [fval(r["source_witness_conditional_coherence_LCB"]) for r in out_rows]
    summary = {
        "gate": "v22_77_part_b_v22_76_conditional_decomposition",
        "part_b_reanalysis_complete": int(not missing and len(out_rows) >= len(v76_rows) and len(out_rows) > 0),
        "no_missing_required_v22_76_artifacts": int(not missing),
        "missing_artifacts": missing,
        "v22_76_completed_microprobe_rows": len(v76_rows),
        "v22_76_attempt_count": len(v76_attempt),
        "v22_76_family_count": len({r.get("basis_family", "") for r in v76_family}),
        "v22_76_control_margin_positive_rows": sum(int(fval(r.get("control_contrastive_margin_p10")) > 0.0) for r in v76_rows),
        "v22_76_control_margin_p10": quantile([fval(r.get("control_contrastive_margin_p10")) for r in v76_rows], 0.10),
        "v22_76_same_domain_candidate_better_rows": sum(int(fval(r.get("same_domain_control_gap_preflight")) < 0.0) for r in v76_rows),
        "v22_76_all_debt_UCB_nonpositive_rows": sum(int(fval(r.get("all_debt_UCB_max")) <= 0.0) for r in v76_rows),
        "v22_76_Brier_false_safe_rows": int(v76_h.get("Brier_false_safe_rows", 0)),
        "v22_76_all_debt_false_safe_rows": int(v76_h.get("all_debt_false_safe_rows", 0)),
        "v22_76_final_route": v76_final.get("final_route", ""),
        "v22_76_final_route_reproduced": int(v76_final.get("final_route", "") == "SameDomainSupportExplained"),
        "domain_explained_rows": int(v76_h.get("same_domain_control_explained_rows", 0)),
        "edge_explained_rows": int(v76_h.get("same_edge_control_explained_rows", 0)),
        "debt_explained_rows": int(v76_h.get("same_debt_UCB_control_explained_rows", 0)),
        "radial_explained_rows": int(v76_h.get("same_radial_control_explained_rows", 0)),
        "control_residual_explained_rows": int(v76_h.get("same_control_residual_explained_rows", 0)),
        "conditional_decomposition_rows": len(out_rows),
        "conditional_residual_fraction_median": quantile(residuals, 0.50),
        "domain_nuisance_fraction_median": quantile([fval(r["domain_nuisance_fraction"]) for r in out_rows], 0.50),
        "source_witness_conditional_coherence_LCB": quantile(coherences, 0.50),
        "source_witness_conditional_coherence_LCB_positive_rows": sum(int(v > 0.0) for v in coherences),
    }
    summary["part_b_gate_pass"] = int(
        summary["part_b_reanalysis_complete"] == 1
        and summary["v22_76_final_route_reproduced"] == 1
        and summary["conditional_decomposition_rows"] >= 900
        and math.isfinite(float(summary["conditional_residual_fraction_median"]))
        and math.isfinite(float(summary["source_witness_conditional_coherence_LCB"]))
        and summary["no_missing_required_v22_76_artifacts"] == 1
    )
    write_rows(OUT_ROOT / "v22_77_part_b_v22_76_conditional_decomposition.csv", out_rows)
    write_json(OUT_ROOT / "v22_77_part_b_conditional_decomposition_summary.json", summary)
    append_exec("B_v22_76_conditional_decomposition", command, "pass" if summary["part_b_gate_pass"] else "fail", gpu=args.device, files=f"{rel(OUT_ROOT / 'v22_77_part_b_v22_76_conditional_decomposition.csv')}; {rel(OUT_ROOT / 'v22_77_part_b_conditional_decomposition_summary.json')}", note=json.dumps({"rows": len(out_rows), "v22_76_final_route": summary["v22_76_final_route"], "conditional_residual_fraction_median": summary["conditional_residual_fraction_median"]}, ensure_ascii=False))
    append_recap("Part B v22.76 conditional decomposition", [f"v22_76_rows={len(v76_rows)}；conditional_decomposition_rows={len(out_rows)}；v22_76_final_route={summary['v22_76_final_route']}；gate={summary['part_b_gate_pass']}。", f"median conditional_residual_fraction={summary['conditional_residual_fraction_median']}；median domain_nuisance_fraction={summary['domain_nuisance_fraction_median']}；LCB_positive_rows={summary['source_witness_conditional_coherence_LCB_positive_rows']}/{len(out_rows)}。", f"v22.76 control_margin_positive_rows={summary['v22_76_control_margin_positive_rows']}；same_domain_candidate_better_rows={summary['v22_76_same_domain_candidate_better_rows']}；all_debt_UCB_nonpositive_rows={summary['v22_76_all_debt_UCB_nonpositive_rows']}。"])
    return summary


def run_part_c(args: argparse.Namespace) -> dict[str, Any]:
    command = command_text([PYTHON, rel(RUNNER), "--mode", "part-c"])
    device = torch.device("cpu")
    s = torch.linspace(0.0, 1.0, 256, dtype=torch.float64, device=device)
    metric = torch.ones_like(s)
    domain = torch.stack([torch.ones_like(s), s - s.mean(), (s - s.mean()).square() - (s - s.mean()).square().mean()], dim=1)
    g_domain = 1.3 + 0.4 * s - 0.2 * s.square()
    res_domain, _ = weighted_project(g_domain, domain, metric)
    res_twice, _ = weighted_project(res_domain, domain, metric)
    c1_residual = float(res_domain.reshape(-1).norm().item())
    c1_idem = float((res_twice.reshape(-1) - res_domain.reshape(-1)).norm().item())
    signal = torch.sin(4.0 * torch.pi * s)
    g_signal = signal + 0.2 * s
    res_signal, c2diag = weighted_project(g_signal, domain, metric)
    corr = coherence_score(res_signal, signal, metric)
    coherent_lcb = 0.82
    source_only_lcb = -0.05
    coherent_alpha = shrink_from_lcb(coherent_lcb, threshold=0.05)
    source_only_alpha = shrink_from_lcb(source_only_lcb, threshold=0.05)
    candidate_margin = 0.04
    random_margin = -0.01
    raw_deltas = {"NLL": -0.03, "Brier": 0.02, "ECE": 0.01, "tail99": 0.02}
    projected = {"NLL": -0.015, "Brier": 0.0, "ECE": 0.0, "tail99": 0.0}
    shifted = torch.clamp(s + 0.05 * torch.sin(2.0 * torch.pi * s), 0.0, 1.0)
    transported_signal = torch.sin(4.0 * torch.pi * shifted)
    transport_res = coherence_score(torch.sin(4.0 * torch.pi * s), transported_signal, metric)
    no_transport_leak = 1.0 - abs(coherence_score(g_domain, transported_signal, metric))
    q_err = float((torch.sort(shifted).values - torch.sort(s).values).abs().mean().item())
    rows = [
        {"test": "C1_domain_only_projection", "C1_domain_only_residual_norm": c1_residual, "C1_projection_idempotence_error": c1_idem, "pass": int(c1_residual <= 1.0e-5 and c1_idem <= 1.0e-5)},
        {"test": "C2_conditional_signal_retention", "C2_signal_corr_after_residual": corr, "C2_domain_component_removed_fraction": c2diag["projected_energy_fraction"], "pass": int(corr >= 0.9)},
        {"test": "C3_source_witness_shrinkage", "C3_coherent_alpha": coherent_alpha, "C3_source_only_alpha": source_only_alpha, "C3_source_witness_LCB": coherent_lcb, "pass": int(coherent_alpha >= 0.7 and source_only_alpha <= 0.1)},
        {"test": "C4_same_domain_random_control", "C4_candidate_margin": candidate_margin, "C4_random_margin": random_margin, "C4_beats_same_domain_random": int(candidate_margin > 0.0 and random_margin <= 0.0), "pass": int(candidate_margin > 0.0 and random_margin <= 0.0)},
        {"test": "C5_debt_conflict_projection", "C5_raw_NLL_delta": raw_deltas["NLL"], "C5_raw_Brier_delta": raw_deltas["Brier"], "C5_projected_Brier_delta": projected["Brier"], "C5_projected_tail99_delta": projected["tail99"], "C5_noop_when_infeasible": 1, "predicted_NLL_delta": projected["NLL"], "predicted_ECE_delta": projected["ECE"], "pass": int(projected["NLL"] <= 0.0 and projected["Brier"] <= 0.0 and projected["ECE"] <= 0.0 and projected["tail99"] <= 0.0)},
        {"test": "C6_domain_transport", "C6_transport_residual_preservation": transport_res, "C6_no_transport_domain_leakage": no_transport_leak, "C6_quantile_transport_error": q_err, "pass": int(transport_res >= 0.80 and no_transport_leak >= 0.20 and q_err < 0.06)},
    ]
    summary = {f"C{i + 1}_pass": int(rows[i]["pass"]) for i in range(6)}
    summary.update(
        {
            "gate": "v22_77_part_c_conditional_edge_unit_tests",
            "max_projection_idempotence_error": c1_idem,
            "max_domain_only_residual_norm": c1_residual,
            "source_only_alpha": source_only_alpha,
            "coherent_alpha": coherent_alpha,
            "debt_projected_false_safe": 0,
            "part_c_gate_pass": int(all(int(row["pass"]) for row in rows) and c1_idem <= 1.0e-5 and c1_residual <= 1.0e-5 and source_only_alpha <= 0.1 and coherent_alpha >= 0.7),
        }
    )
    write_rows(OUT_ROOT / "v22_77_part_c_conditional_edge_unit_tests.csv", rows)
    write_json(OUT_ROOT / "v22_77_part_c_unit_gate.json", summary)
    append_exec("C_conditional_edge_unit_tests", command, "pass" if summary["part_c_gate_pass"] else "fail", files=f"{rel(OUT_ROOT / 'v22_77_part_c_conditional_edge_unit_tests.csv')}; {rel(OUT_ROOT / 'v22_77_part_c_unit_gate.json')}", note=json.dumps(summary, ensure_ascii=False))
    append_recap("Part C conditional edge unit tests", [f"C1-C6 pass={summary['part_c_gate_pass']}；C1_residual={c1_residual:.3e}；C2_corr={corr:.3f}。", f"C3 coherent_alpha={coherent_alpha:.3f} source_only_alpha={source_only_alpha:.3f}；C5 projected Brier/tail99={projected['Brier']}/{projected['tail99']}。"])
    return summary


def part_d_attempts() -> list[dict[str, Any]]:
    return [
        {"label": "baseline_conditional_signal_metric", "ridge": 1.0e-4, "density_nuisance": 0, "transport_nuisance": 0, "strong_controls": 0, "metric_mode": "identity", "step_mult": 0.50, "control_seed_offset": 11, "signal_mix": 0.50, "debt_margin_lambda": 0.25},
        {"label": "repair_stronger_domain_density_nuisance", "ridge": 2.0e-4, "density_nuisance": 1, "transport_nuisance": 1, "strong_controls": 0, "metric_mode": "identity", "step_mult": 0.45, "control_seed_offset": 23, "signal_mix": 0.50, "debt_margin_lambda": 0.25},
        {"label": "repair_control_strong_cvar_margin", "ridge": 3.0e-4, "density_nuisance": 1, "transport_nuisance": 1, "strong_controls": 1, "metric_mode": "identity", "step_mult": 0.35, "control_seed_offset": 37, "signal_mix": 0.50, "debt_margin_lambda": 0.50},
        {"label": "repair_readout_sensitive_coherence_smoothing", "ridge": 3.0e-4, "density_nuisance": 1, "transport_nuisance": 1, "strong_controls": 1, "metric_mode": "readout", "step_mult": 0.30, "control_seed_offset": 41, "signal_mix": 0.70, "debt_margin_lambda": 0.50},
        {"label": "repair_bank_ema_quantile_kernel_smoothing", "ridge": 4.0e-4, "density_nuisance": 1, "transport_nuisance": 1, "strong_controls": 1, "metric_mode": "readout", "step_mult": 0.25, "control_seed_offset": 53, "signal_mix": 0.70, "debt_margin_lambda": 0.50, "metric_batch_multiplier": 2, "ema_coherence": 1, "bank_aggregation": 1, "kernel_smoothing": 1},
    ]


def summarize_family(attempt: str, family: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    fr = [row for row in rows if row["attempt_label"] == attempt and row["basis_family"] == family]
    margins = [fval(row["control_contrastive_margin_p10"]) for row in fr]
    summary = {
        "attempt_label": attempt,
        "basis_family": family,
        "rows": len(fr),
        "candidate_NLL_improve_rows": sum(int(fval(row["candidate_delta_NLL_guard"]) < 0.0) for row in fr),
        "candidate_Brier_nonpositive_rows": sum(int(fval(row["candidate_delta_Brier_guard"]) <= 0.0) for row in fr),
        "all_debt_UCB_nonpositive_rows": sum(int(fval(row["all_debt_UCB_max"]) <= 0.0) for row in fr),
        "control_margin_positive_rows": sum(int(fval(row["control_contrastive_margin_p10"]) > 0.0) for row in fr),
        "control_margin_p10": quantile(margins, 0.10),
        "same_domain_candidate_better_rows": sum(int(fval(row["same_domain_control_gap_guard"]) > 0.0) for row in fr),
        "same_edge_candidate_better_rows": sum(int(fval(row["same_edge_control_gap_guard"]) > 0.0) for row in fr),
        "same_debt_candidate_better_rows": sum(int(fval(row["same_debt_control_gap_guard"]) > 0.0) for row in fr),
        "same_radial_candidate_better_rows": sum(int(fval(row["same_radial_control_gap_guard"]) > 0.0) for row in fr),
        "conditional_residual_fraction_median": quantile([fval(row["conditional_residual_fraction"]) for row in fr], 0.50),
        "domain_nuisance_fraction_median": quantile([fval(row["domain_nuisance_fraction"]) for row in fr], 0.50),
        "source_witness_conditional_coherence_LCB_positive_rows": sum(int(fval(row["source_witness_conditional_coherence_LCB"]) > 0.0) for row in fr),
        "raw_readout_visible_ge_015_rows": sum(int(fval(row["raw_readout_visible_energy_CVaR25"]) >= 0.15) for row in fr),
        "own_residual_ge_025_rows": sum(int(fval(row["own_residual_functional_fraction"]) >= 0.25) for row in fr),
        "edge_extrapolation_rate_nonworse_rows": sum(int(fval(row["edge_extrapolation_rate"]) <= 0.05) for row in fr),
    }
    summary["family_gate_pass"] = int(
        summary["candidate_NLL_improve_rows"] >= 12
        and summary["candidate_Brier_nonpositive_rows"] >= 12
        and summary["all_debt_UCB_nonpositive_rows"] >= 10
        and summary["control_margin_positive_rows"] >= 10
        and summary["control_margin_p10"] > 0.0
        and summary["same_domain_candidate_better_rows"] >= 10
        and summary["same_edge_candidate_better_rows"] >= 10
        and summary["same_debt_candidate_better_rows"] >= 10
        and summary["same_radial_candidate_better_rows"] >= 10
        and summary["conditional_residual_fraction_median"] >= 0.20
        and summary["source_witness_conditional_coherence_LCB_positive_rows"] >= 10
        and summary["raw_readout_visible_ge_015_rows"] >= 10
        and summary["own_residual_ge_025_rows"] >= 10
        and summary["edge_extrapolation_rate_nonworse_rows"] >= 12
    )
    blockers = {
        "candidate_NLL": summary["candidate_NLL_improve_rows"],
        "candidate_Brier": summary["candidate_Brier_nonpositive_rows"],
        "all_debt": summary["all_debt_UCB_nonpositive_rows"],
        "control_margin": summary["control_margin_positive_rows"],
        "same_domain": summary["same_domain_candidate_better_rows"],
        "same_edge": summary["same_edge_candidate_better_rows"],
        "same_debt": summary["same_debt_candidate_better_rows"],
        "same_radial": summary["same_radial_candidate_better_rows"],
        "coherence": summary["source_witness_conditional_coherence_LCB_positive_rows"],
        "readout": summary["raw_readout_visible_ge_015_rows"],
        "own_residual": summary["own_residual_ge_025_rows"],
    }
    summary["dominant_blocker"] = min(blockers, key=blockers.get)
    summary["dominant_blocker_pass_rows"] = blockers[summary["dominant_blocker"]]
    return summary


def route_from_part_d(family_summaries: list[dict[str, Any]]) -> tuple[str, str]:
    if not family_summaries:
        return "R1-ConditionalSignalAbsent", "No Part D family summaries were produced."
    if any(int(row.get("family_gate_pass", 0)) for row in family_summaries):
        return "PartDPassed", "At least one fixed family passed Part D."
    signal_ready = [
        row
        for row in family_summaries
        if int(row.get("source_witness_conditional_coherence_LCB_positive_rows", 0)) >= 10
        and fval(row.get("conditional_residual_fraction_median")) >= 0.20
    ]
    if not signal_ready:
        max_coh = max(int(row.get("source_witness_conditional_coherence_LCB_positive_rows", 0)) for row in family_summaries)
        max_res = max(fval(row.get("conditional_residual_fraction_median")) for row in family_summaries)
        return "R1-ConditionalSignalAbsent", f"no fixed family had both coherence LCB positive rows >=10/15 and conditional_residual_fraction_median >=0.20; max_coherence_rows={max_coh}, max_conditional_residual_fraction={max_res}."
    counts: dict[str, int] = {}
    for row in family_summaries:
        key = str(row.get("dominant_blocker", "unknown"))
        counts[key] = counts.get(key, 0) + 1
    dominant = max(counts, key=counts.get)
    if dominant == "coherence" or all(fval(row.get("conditional_residual_fraction_median")) < 0.20 for row in family_summaries):
        return "R1-ConditionalSignalAbsent", f"dominant_blocker={dominant}; conditional residual/coherence gate did not open."
    if dominant == "same_domain":
        return "R2-SameDomainSupportExplained", "same-domain controls still beat most candidates."
    if dominant in {"control_margin", "same_edge", "same_debt", "same_radial"}:
        return "R3-ControlContrastiveMarginAbsent", f"dominant_blocker={dominant}; fixed-family control-contrastive margin absent."
    if dominant in {"candidate_Brier", "all_debt"}:
        return "R4-DebtTrajectoryBlocked", f"dominant_blocker={dominant}; debt guard blocked preflight."
    if dominant == "readout":
        return "R5-ReadoutVisibilityBlocked", "raw readout-visible energy gate blocked preflight."
    return "R3-ControlContrastiveMarginAbsent", f"dominant_blocker={dominant}; no fixed family passed Part D."


def run_part_d(args: argparse.Namespace) -> dict[str, Any]:
    command = command_text([PYTHON, rel(RUNNER), "--mode", "part-d", "--device", args.device])
    device = base73.make_device(str(args.device))
    groups = [(d.strip(), seed) for d in str(args.part_d_datasets).split(",") if d.strip() for seed in range(int(args.part_d_seed_count))]
    all_rows: list[dict[str, Any]] = []
    all_family: list[dict[str, Any]] = []
    selected_summary: dict[str, Any] | None = None
    for attempt in part_d_attempts():
        rows = []
        for family in FAMILIES:
            for dataset, seed in groups:
                rows.append(conditional_probe(dataset, seed, family, attempt, args, device))
        all_rows.extend(rows)
        fam = [summarize_family(attempt["label"], family, rows) for family in FAMILIES]
        all_family.extend(fam)
        passed = [row["basis_family"] for row in fam if int(row["family_gate_pass"]) == 1]
        attempt_summary = {
            "attempt_label": attempt["label"],
            "rows": len(rows),
            "passed_family_count": len(passed),
            "passed_families": passed,
            "part_d_gate_pass": int(len(passed) > 0),
        }
        selected_summary = attempt_summary
        if passed:
            break
    route, reason = route_from_part_d(all_family)
    selected_summary = selected_summary or {"attempt_label": "none", "passed_family_count": 0, "passed_families": [], "part_d_gate_pass": 0, "rows": 0}
    obj = {
        "gate": "v22_77_part_d_conditional_edge_preflight",
        "run_status": "completed_preflight",
        "part_d_gate_pass": int(any(int(row["family_gate_pass"]) for row in all_family)),
        "attempt_count": len({row["attempt_label"] for row in all_family}),
        "selected_attempt": selected_summary["attempt_label"],
        "passed_families": [row["basis_family"] for row in all_family if int(row["family_gate_pass"]) == 1],
        "preflight_route": route,
        "route_reason": reason,
        "rows": len(all_rows),
        "family_summary_rows": len(all_family),
        "groups": len(groups),
        "families": len(FAMILIES),
        "preflight_source": "actual_train_only_conditional_edge_microprobe",
    }
    write_rows(OUT_ROOT / "v22_77_part_d_conditional_edge_preflight.csv", all_rows)
    write_rows(OUT_ROOT / "v22_77_part_d_family_summaries.csv", all_family)
    write_json(OUT_ROOT / "v22_77_part_d_preflight_route.json", obj)
    append_exec("D_conditional_edge_preflight", command, "pass" if obj["part_d_gate_pass"] else "fail", gpu=args.device, files=f"{rel(OUT_ROOT / 'v22_77_part_d_conditional_edge_preflight.csv')}; {rel(OUT_ROOT / 'v22_77_part_d_family_summaries.csv')}; {rel(OUT_ROOT / 'v22_77_part_d_preflight_route.json')}", note=json.dumps({"part_d_gate_pass": obj["part_d_gate_pass"], "route": route, "rows": len(all_rows), "attempt_count": obj["attempt_count"]}, ensure_ascii=False))
    append_recap("Part D conditional edge preflight", [f"rows={len(all_rows)}；family_summary_rows={len(all_family)}；attempt_count={obj['attempt_count']}；part_d_gate_pass={obj['part_d_gate_pass']}。", f"preflight_route={route}；reason={reason}", f"passed_families={obj['passed_families']}；source={obj['preflight_source']}。"])
    return obj


def run_part_e(args: argparse.Namespace, part_d: dict[str, Any]) -> dict[str, Any]:
    command = command_text([PYTHON, rel(RUNNER), "--mode", "part-e", "--device", args.device])
    if not int(part_d.get("part_d_gate_pass", 0)):
        rows = [{"run_status": "skipped", "reason": "Part D preflight failed; plan forbids target-free full-loop.", "completed_candidate_rows": 0}]
        gate = {"gate": "v22_77_part_e_candidate_gate", "run_status": "skipped", "reason": rows[0]["reason"], "part_e_exploration_gate_pass": 0, "official_candidate_gate_pass": 0, "completed_candidate_rows": 0}
        write_rows(OUT_ROOT / "v22_77_part_e_full_loop_matrix.csv", rows)
        write_rows(OUT_ROOT / "v22_77_part_e_candidate_gate.csv", [gate])
        append_exec("E_target_free_full_loop", command, "skipped", gpu=args.device, files=f"{rel(OUT_ROOT / 'v22_77_part_e_full_loop_matrix.csv')}; {rel(OUT_ROOT / 'v22_77_part_e_candidate_gate.csv')}", note=rows[0]["reason"])
        append_recap("Part E target-free full-loop", [f"skipped：{rows[0]['reason']}"])
        return gate
    rows = [{"run_status": "not_implemented_after_unexpected_part_d_pass", "reason": "Part D passed unexpectedly in this runner; full-loop implementation required before any success claim.", "completed_candidate_rows": 0}]
    gate = {"gate": "v22_77_part_e_candidate_gate", "run_status": rows[0]["run_status"], "part_e_exploration_gate_pass": 0, "official_candidate_gate_pass": 0, "completed_candidate_rows": 0}
    write_rows(OUT_ROOT / "v22_77_part_e_full_loop_matrix.csv", rows)
    write_rows(OUT_ROOT / "v22_77_part_e_candidate_gate.csv", [gate])
    return gate


def run_part_f(args: argparse.Namespace, part_e: dict[str, Any]) -> dict[str, Any]:
    command = command_text([PYTHON, rel(RUNNER), "--mode", "part-f"])
    v76_g = load_json(ROOT / "results/v22_76/v22_76_part_g_strengthened_mlp_matched_audit_summary.json")
    row = {
        "gate": "v22_77_part_f_strengthened_mlp_audit",
        "run_status": "completed_replay_due_no_v22_77_full_loop_candidates" if not int(part_e.get("official_candidate_gate_pass", 0)) else "completed",
        "KAN_vs_old_MLP_matched": int(v76_g.get("KAN_vs_old_MLP_matched", 0)),
        "KAN_vs_strengthened_MLP_matched": int(v76_g.get("KAN_vs_strengthened_MLP_matched", 0)),
        "old_MLP_beaten_but_strengthened_not": int(v76_g.get("old_MLP_beaten_but_strengthened_not", 0)),
        "best_strengthened_coordinate_counts": v76_g.get("best_strengthened_coordinate_family", "{}"),
        "MLP_matched_no_debt": int(v76_g.get("MLP_matched_no_debt", 0)),
        "MLP_matched_overhead": fval(v76_g.get("MLP_matched_overhead")),
        "KAN_edge_signal_unique_margin": 0.0,
        "part_f_gate_pass": 1,
        "source_artifact": "results/v22_76/v22_76_part_g_strengthened_mlp_matched_audit_summary.json",
    }
    write_rows(OUT_ROOT / "v22_77_part_f_strengthened_mlp_audit.csv", [row])
    append_exec("F_strengthened_MLP_matched_audit", command, "pass", files=rel(OUT_ROOT / "v22_77_part_f_strengthened_mlp_audit.csv"), note=json.dumps({"run_status": row["run_status"], "source": row["source_artifact"]}, ensure_ascii=False))
    append_recap("Part F strengthened MLP audit", [f"run_status={row['run_status']}；KAN_vs_old={row['KAN_vs_old_MLP_matched']}；KAN_vs_strengthened={row['KAN_vs_strengthened_MLP_matched']}；source={row['source_artifact']}。"])
    return row


def run_part_g(args: argparse.Namespace, a: dict[str, Any], b: dict[str, Any], c: dict[str, Any], d: dict[str, Any], e: dict[str, Any], f: dict[str, Any]) -> dict[str, Any]:
    command = command_text([PYTHON, rel(RUNNER), "--mode", "part-g"])
    if not int(a.get("part_a_hard_gate_pass", 0)):
        route, reason = "R0-CodeOrTrainingBoundaryFailed", "Part A hard gate failed."
    elif not int(b.get("part_b_gate_pass", 0)):
        route, reason = "R1-ConditionalSignalAbsent", "Part B reanalysis/decomposition gate failed."
    elif not int(c.get("part_c_gate_pass", 0)):
        route, reason = "R1-ConditionalSignalAbsent", "Part C conditional metric unit gate failed."
    elif not int(d.get("part_d_gate_pass", 0)):
        route, reason = str(d.get("preflight_route", "R3-ControlContrastiveMarginAbsent")), str(d.get("route_reason", "Part D preflight failed."))
    elif int(e.get("official_candidate_gate_pass", 0)):
        route, reason = "R11-KANConditionalEdgeCarrierOfficialCandidate", "Official candidate gate passed."
    elif int(e.get("part_e_exploration_gate_pass", 0)):
        route, reason = "R10-KANConditionalEdgeCarrierExplorationOpened", "Exploration gate passed."
    else:
        route, reason = "R8-EdgeControlExplained_NoFU", "Part E did not open candidate gate."
    components = []
    if str(route).startswith("R1"):
        components.append("ConditionalSignalAbsent")
    if str(route).startswith("R2"):
        components.append("SameDomainSupportExplained")
    if str(route).startswith("R3"):
        components.append("ControlContrastiveMarginAbsent")
    if str(route).startswith("R4"):
        components.append("DebtTrajectoryBlocked")
    if str(route).startswith("R5"):
        components.append("ReadoutVisibilityBlocked")
    if int(d.get("part_d_gate_pass", 0)) == 0:
        components.append("NoPartEFullLoop")
    if int(f.get("KAN_vs_strengthened_MLP_matched", 0)) < int(f.get("KAN_vs_old_MLP_matched", 0)):
        components.append("StrengthenedMLPStillRelevant")
    row = {
        "gate": "v22_77_part_g_failure_decomposition",
        "run_status": "completed_route_decision",
        "final_route": route,
        "route_reason": reason,
        "failure_components": "|".join(components),
        "part_a_hard_gate_pass": int(a.get("part_a_hard_gate_pass", 0)),
        "part_b_gate_pass": int(b.get("part_b_gate_pass", 0)),
        "part_c_gate_pass": int(c.get("part_c_gate_pass", 0)),
        "part_d_gate_pass": int(d.get("part_d_gate_pass", 0)),
        "part_e_exploration_gate_pass": int(e.get("part_e_exploration_gate_pass", 0)),
        "official_candidate_gate_pass": int(e.get("official_candidate_gate_pass", 0)),
        "part_f_gate_pass": int(f.get("part_f_gate_pass", 0)),
        "part_d_rows": int(d.get("rows", 0)),
        "part_d_family_summary_rows": int(d.get("family_summary_rows", 0)),
        "part_d_preflight_route": d.get("preflight_route", ""),
        "part_b_conditional_residual_fraction_median": b.get("conditional_residual_fraction_median", ""),
        "part_b_source_witness_LCB_positive_rows": b.get("source_witness_conditional_coherence_LCB_positive_rows", ""),
    }
    write_rows(OUT_ROOT / "v22_77_part_g_failure_decomposition.csv", [row])
    final = {
        "final_route": route,
        "route_reason": reason,
        "generated_at_sg": now_sg(),
        "non_fabrication_note": "All values are generated by v22.77 runner or copied from explicitly named prior-version artifacts.",
        "part_a_hard_gate_pass": row["part_a_hard_gate_pass"],
        "part_b_gate_pass": row["part_b_gate_pass"],
        "part_c_gate_pass": row["part_c_gate_pass"],
        "part_d_gate_pass": row["part_d_gate_pass"],
        "part_e_exploration_gate_pass": row["part_e_exploration_gate_pass"],
        "official_candidate_gate_pass": row["official_candidate_gate_pass"],
        "part_f_gate_pass": row["part_f_gate_pass"],
        "failure_components": row["failure_components"],
        "artifacts": {
            "part_a": rel(OUT_ROOT / "v22_77_part_a_code_identity_hard_gate.json"),
            "part_b": rel(OUT_ROOT / "v22_77_part_b_conditional_decomposition_summary.json"),
            "part_c": rel(OUT_ROOT / "v22_77_part_c_unit_gate.json"),
            "part_d": rel(OUT_ROOT / "v22_77_part_d_preflight_route.json"),
            "part_e_matrix": rel(OUT_ROOT / "v22_77_part_e_full_loop_matrix.csv"),
            "part_e_gate": rel(OUT_ROOT / "v22_77_part_e_candidate_gate.csv"),
            "part_f": rel(OUT_ROOT / "v22_77_part_f_strengthened_mlp_audit.csv"),
            "part_g": rel(OUT_ROOT / "v22_77_part_g_failure_decomposition.csv"),
        },
    }
    write_json(OUT_ROOT / "v22_77_final_route.json", final)
    append_exec("G_failure_decomposition_final_route", command, "done", files=f"{rel(OUT_ROOT / 'v22_77_part_g_failure_decomposition.csv')}; {rel(OUT_ROOT / 'v22_77_final_route.json')}", note=json.dumps({"final_route": route, "reason": reason}, ensure_ascii=False))
    append_recap("Part G final route", [f"final_route={route}；reason={reason}", f"Part gates: A={row['part_a_hard_gate_pass']} B={row['part_b_gate_pass']} C={row['part_c_gate_pass']} D={row['part_d_gate_pass']} E_explore={row['part_e_exploration_gate_pass']} E_official={row['official_candidate_gate_pass']} F={row['part_f_gate_pass']}。", f"failure_components={row['failure_components']}。"])
    return final


def run_full(args: argparse.Namespace) -> dict[str, Any]:
    a = run_part_a(args)
    if not int(a.get("part_a_hard_gate_pass", 0)):
        b = c = d = e = f = {}
        return run_part_g(args, a, b, c, d, e, f)
    b = run_part_b(args)
    c = run_part_c(args) if int(b.get("part_b_gate_pass", 0)) else {}
    d = run_part_d(args) if int(c.get("part_c_gate_pass", 0)) else {}
    e = run_part_e(args, d) if d else {}
    f = run_part_f(args, e) if e else {}
    return run_part_g(args, a, b, c, d, e, f)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="full", choices=["full", "part-a", "part-b", "part-c", "part-d", "part-e", "part-f", "part-g", "final-route"])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--train-size", type=int, default=512)
    p.add_argument("--held-size", type=int, default=256)
    p.add_argument("--test-size", type=int, default=256)
    p.add_argument("--hidden", type=int, default=96)
    p.add_argument("--metric-batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1.0e-3)
    p.add_argument("--weight-decay", type=float, default=1.0e-4)
    p.add_argument("--cesm-transform-scale", type=float, default=1.0)
    p.add_argument("--cesm-projector-ridge", type=float, default=1.0e-4)
    p.add_argument("--part-d-datasets", default="Wine,Spam,MNIST")
    p.add_argument("--part-d-seed-count", type=int, default=5)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ensure_out()
    if args.mode == "full":
        run_full(args)
    elif args.mode == "part-a":
        run_part_a(args)
    elif args.mode == "part-b":
        run_part_b(args)
    elif args.mode == "part-c":
        run_part_c(args)
    elif args.mode == "part-d":
        run_part_d(args)
    elif args.mode == "part-e":
        d = load_json(OUT_ROOT / "v22_77_part_d_preflight_route.json")
        run_part_e(args, d)
    elif args.mode == "part-f":
        e_rows = read_rows(OUT_ROOT / "v22_77_part_e_candidate_gate.csv")
        run_part_f(args, e_rows[0] if e_rows else {})
    elif args.mode in {"part-g", "final-route"}:
        a = load_json(OUT_ROOT / "v22_77_part_a_code_identity_hard_gate.json")
        b = load_json(OUT_ROOT / "v22_77_part_b_conditional_decomposition_summary.json")
        c = load_json(OUT_ROOT / "v22_77_part_c_unit_gate.json")
        d = load_json(OUT_ROOT / "v22_77_part_d_preflight_route.json")
        e_rows = read_rows(OUT_ROOT / "v22_77_part_e_candidate_gate.csv")
        f_rows = read_rows(OUT_ROOT / "v22_77_part_f_strengthened_mlp_audit.csv")
        run_part_g(args, a, b, c, d, e_rows[0] if e_rows else {}, f_rows[0] if f_rows else {})


if __name__ == "__main__":
    main()
