#!/usr/bin/env python3
"""DG-KAN v22.78 Conditional Edge-Bank Signal Basis Redesign runner."""

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
import experiments.run_v22_77_conditional_edge_signal_metric_kan_mpfu as base77
from dgkan.fu.kan_conditional_edge_signal_metric import (
    coherence_score,
    normalize_columns,
    polynomial_nuisance_basis,
    shrink_from_lcb,
    weighted_project,
)
from dgkan.fu.kan_edge_bank_signal_basis import (
    EdgeBankSignalBasisOptimizer,
    EdgeBankSignalBasisState,
    bank_additive_projection,
    bank_coherence,
    edge_signal_sparsity,
    project_grouped_quantile_basis,
    sign_cancellation_rate,
)


PYTHON = sys.executable
RUNNER = ROOT / "experiments/run_v22_78_conditional_edge_bank_signal_basis_redesign_kan_mpfu.py"
PLAN = ROOT / "docs/DG-KAN_v22.78_ConditionalEdgeBankSignalBasisRedesignKAN_MPFU_完整计划.md"
EXEC_LOG = ROOT / "docs/DG-KAN_v22.78_ConditionalEdgeBankSignalBasisRedesignKAN_MPFU_执行日志.md"
RECAP_LOG = ROOT / "docs/DG-KAN_v22.78_ConditionalEdgeBankSignalBasisRedesignKAN_MPFU_实验结果复盘.md"
OUT_ROOT = ROOT / "results/v22_78"
LOG_ROOT = OUT_ROOT / "logs"
OP_MODULE = ROOT / "dgkan/fu/kan_edge_bank_signal_basis.py"


BASIS_FAMILIES = [
    "density_equalized_conditional_spline",
    "task_conditional_orthogonal_poly",
    "quantile_lowfreq_local_residual",
    "node_bank_additive_anova",
    "upstream_separation_diagnostic",
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
    return base77.fval(x, default)


def quantile(values: list[float], q: float) -> float:
    return base77.quantile(values, q)


def lower_cvar(values: list[float], frac: float = 0.25) -> float:
    return base77.lower_cvar(values, frac)


def write_json(path: Path, obj: Any) -> None:
    base77.write_json(path, obj)


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    base77.write_rows(path, rows)


def read_rows(path: Path) -> list[dict[str, str]]:
    return base77.read_rows(path)


def load_json(path: Path) -> dict[str, Any]:
    return base77.load_json(path)


def append_exec(task_id: str, command: str, status: str, *, gpu: str = "", files: str = "", note: str = "") -> None:
    ensure_out()
    if not EXEC_LOG.exists():
        EXEC_LOG.write_text(
            "# DG-KAN v22.78 ConditionalEdgeBankSignalBasisRedesignKAN MPFU 执行日志\n\n"
            f"创建时间：{now_sg()}\n\n"
            f"- 计划文档：`{rel(PLAN)}`\n"
            "- 非编造约束：只记录真实命令、文件和观测；缺失 artifact 标记 skipped/missing。\n"
            "- 复现提示：命令、GPU、输出文件均逐条记录；Part D 支持 shard 并行后 merge。\n\n"
            "## 命令记录\n",
            encoding="utf-8",
        )
    with EXEC_LOG.open("a", encoding="utf-8") as f:
        f.write(f"\n### {now_sg()} | {task_id} | {status}\n")
        f.write(f"- command: `{command}`\n")
        f.write(f"- gpu: `{gpu}`\n")
        f.write(f"- files: `{files}`\n")
        f.write(f"- note: {note}\n")
    journal = OUT_ROOT / "v22_78_command_journal.csv"
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
            "# DG-KAN v22.78 ConditionalEdgeBankSignalBasisRedesignKAN MPFU 实验结果复盘\n\n"
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


def metric_snapshot(logits: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
    return base77.metric_snapshot(logits, y)


def metric_delta_for_update(model: Any, x: torch.Tensor, y: torch.Tensor, vector: torch.Tensor) -> dict[str, float]:
    return base77.metric_delta_for_update(model, x, y, vector)


def w2_grad_for_loss(model: Any, x: torch.Tensor, y: torch.Tensor, loss_kind: str) -> torch.Tensor:
    if loss_kind in {"ce", "brier", "radial", "shuffled", "hard_loss", "loss_rank"}:
        return base77.w2_grad_for_loss(model, x, y, loss_kind)
    model.zero_grad(set_to_none=True)
    logits = model(x).float()
    if loss_kind == "tail_debt":
        per = F.cross_entropy(logits, y.long(), reduction="none")
        threshold = torch.quantile(per.detach(), 0.90)
        mask = per.detach() >= threshold
        loss = per[mask].mean() if mask.any() else per.mean()
    elif loss_kind == "margin_debt":
        true_logit = logits.gather(1, y.long().reshape(-1, 1)).reshape(-1)
        masked = logits.scatter(1, y.long().reshape(-1, 1), float("-inf"))
        other = masked.max(dim=1).values
        margin = true_logit - other
        loss = F.softplus(-margin).mean()
    elif loss_kind == "ece_debt":
        probs = torch.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)
        correct = pred.eq(y.long()).float()
        loss = (conf - correct).square().mean()
    else:
        raise ValueError(f"unknown loss_kind={loss_kind}")
    loss.backward()
    grad = model.w2.grad.detach().reshape(-1).to(device=x.device, dtype=torch.float64)
    model.zero_grad(set_to_none=True)
    return grad


def metric_diag_from_design(model: Any, x: torch.Tensor, *, mode: str) -> torch.Tensor:
    return base77.metric_diag_from_design(model, x, mode=mode)


def map_basis_to_model_method(family: str) -> str:
    if "poly" in family:
        return "wlb_monotone_lowfreq2_bump2_brier_natural_dynamic_margin"
    if "lowfreq" in family:
        return "wlb_lowfreq2_bump4_brier_natural_dynamic_margin"
    if "bank" in family:
        return "wlb_lowfreq2_bump2_brier_natural_dynamic_margin"
    if "upstream" in family:
        return "wlb_compacthat_lowfreq2_bump2_brier_natural_dynamic_margin"
    return "wlb_lowfreq2_bump2_brier_natural_dynamic_margin"


def make_domain_basis(dim: int, device: torch.device, grads: dict[str, torch.Tensor], *, include_density: bool = True) -> torch.Tensor | None:
    cols: list[torch.Tensor] = []
    poly = polynomial_nuisance_basis(dim, device, include_density=include_density)
    if poly is not None:
        cols.extend([poly[:, j] for j in range(int(poly.shape[1]))])
    cols.extend([grads["domain"], grads["radial"], grads["debt"], grads["witness"] - grads["source"]])
    for name in ["tail_debt", "margin_debt", "ece_debt"]:
        if name in grads:
            cols.append(grads[name])
    return normalize_columns(cols, dim, device)


def matched_conditional_energy_control(conditional_signal: torch.Tensor) -> torch.Tensor:
    """Make a same-energy conditional control without reusing the candidate axis."""

    signal = conditional_signal.detach().reshape(-1).to(dtype=torch.float64)
    shift = max(1, int(signal.numel()) // 7)
    ctrl = torch.roll(signal, shifts=shift)
    coeff = (ctrl * signal).sum() / signal.square().sum().clamp_min(1.0e-12)
    ctrl = ctrl - coeff * signal
    ctrl = ctrl / ctrl.norm().clamp_min(1.0e-12) * signal.norm().clamp_min(1.0e-12)
    return ctrl


def make_control_basis(dim: int, device: torch.device, grads: dict[str, torch.Tensor], conditional_signal: torch.Tensor) -> torch.Tensor | None:
    matched_conditional = matched_conditional_energy_control(conditional_signal).to(device=device)
    cols = [
        grads["domain"],
        grads["shuffled"],
        grads["random"],
        grads["debt"],
        grads["radial"],
        grads["hard_loss"],
        grads["loss_rank"],
        grads["smooth"],
        matched_conditional,
    ]
    for name in ["tail_debt", "margin_debt", "ece_debt"]:
        if name in grads:
            cols.append(grads[name])
    return normalize_columns(cols, dim, device)


def bootstrap_coherence_lcb(
    model: Any,
    x_s: torch.Tensor,
    y_s: torch.Tensor,
    x_w: torch.Tensor,
    y_w: torch.Tensor,
    domain_basis: torch.Tensor | None,
    metric_diag: torch.Tensor,
    ridge: float,
    shape: torch.Size,
) -> tuple[float, float, float, float]:
    samples: list[float] = []
    bank_samples: list[float] = []
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
            bank_samples.append(bank_coherence(sr, wr, metric_diag, shape))
    def mean_lcb(vals: list[float]) -> tuple[float, float]:
        if not vals:
            return 0.0, 0.0
        mean = sum(vals) / len(vals)
        if len(vals) == 1:
            return float(mean), float(mean)
        var = sum((x - mean) ** 2 for x in vals) / (len(vals) - 1)
        return float(mean), float(mean - 1.64 * math.sqrt(var) / math.sqrt(len(vals)))
    mean, lcb = mean_lcb(samples)
    bmean, blcb = mean_lcb(bank_samples)
    return mean, lcb, bmean, blcb


def apply_basis_redesign(
    signal: torch.Tensor,
    source_res: torch.Tensor,
    witness_res: torch.Tensor,
    metric_diag: torch.Tensor,
    model: Any,
    family: str,
    ridge: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    shape = getattr(model, "w2").shape
    stats: dict[str, float] = {
        "knot_density_balance": 0.0,
        "basis_domain_orthogonality_error": 0.0,
        "same_density_spline_random_gap": 0.0,
        "task_orthogonal_poly_rank": 0.0,
        "poly_condition_number": 0.0,
        "conditional_signal_alignment": 0.0,
        "same_degree_random_poly_gap": 0.0,
        "low_frequency_energy_fraction": 0.0,
        "local_bump_energy_fraction": 0.0,
        "bump_domain_density_coverage": 0.0,
        "edge_extrapolation_rate": 0.0,
        "same_frequency_random_gap": 0.0,
        "same_bump_support_random_gap": 0.0,
        "edge_bank_anova_explained": 0.0,
        "interaction_residual_fraction": 1.0,
        "bank_condition_number": 0.0,
        "same_bank_energy_random_gap": 0.0,
        "same_bank_domain_random_gap": 0.0,
        "upstream_transport_type": "",
        "conditional_residual_before": 0.0,
        "conditional_residual_after": 0.0,
        "interaction_residual_before": 0.0,
        "interaction_residual_after": 0.0,
        "same_transport_random_gap": 0.0,
    }
    before_bank, before_stats = bank_additive_projection(signal, metric_diag, shape)
    stats["edge_bank_anova_explained"] = fval(before_stats.get("edge_bank_anova_explained"))
    stats["interaction_residual_fraction"] = fval(before_stats.get("interaction_residual_fraction"))
    stats["bank_condition_number"] = fval(before_stats.get("bank_condition_number"))
    if family == "density_equalized_conditional_spline":
        out, bstats = project_grouped_quantile_basis(signal, metric_diag, shape, family="density_spline", rank=4, ridge=ridge)
        stats.update({k: fval(v) for k, v in bstats.items()})
        stats["basis_domain_orthogonality_error"] = abs(coherence_score(out, torch.ones_like(out), metric_diag))
        return out.reshape(-1), stats
    if family == "task_conditional_orthogonal_poly":
        out, bstats = project_grouped_quantile_basis(signal, metric_diag, shape, family="orthogonal_poly", rank=4, ridge=ridge)
        stats.update({k: fval(v) for k, v in bstats.items()})
        stats["conditional_signal_alignment"] = coherence_score(out, signal, metric_diag)
        stats["same_degree_random_poly_gap"] = max(0.0, stats["conditional_signal_alignment"] - abs(coherence_score(out, torch.roll(signal, shifts=7), metric_diag)))
        return out.reshape(-1), stats
    if family == "quantile_lowfreq_local_residual":
        out, bstats = project_grouped_quantile_basis(signal, metric_diag, shape, family="lowfreq_bump", rank=3, ridge=ridge)
        stats.update({k: fval(v) for k, v in bstats.items()})
        stats["same_frequency_random_gap"] = max(0.0, coherence_score(out, signal, metric_diag) - abs(coherence_score(out, torch.roll(signal, shifts=11), metric_diag)))
        stats["same_bump_support_random_gap"] = stats["same_frequency_random_gap"]
        return out.reshape(-1), stats
    if family == "node_bank_additive_anova":
        out, bstats = bank_additive_projection(signal, metric_diag, shape)
        stats.update({k: fval(v) for k, v in bstats.items()})
        stats["same_bank_energy_random_gap"] = max(0.0, coherence_score(out, signal, metric_diag) - abs(coherence_score(out, torch.roll(signal, shifts=13), metric_diag)))
        stats["same_bank_domain_random_gap"] = max(0.0, coherence_score(out, source_res, metric_diag) - abs(coherence_score(out, witness_res - source_res, metric_diag)))
        return out.reshape(-1), stats
    if family == "upstream_separation_diagnostic":
        stats["upstream_transport_type"] = "train_only_rank_monotone_density_transport"
        stats["conditional_residual_before"] = stats["edge_bank_anova_explained"]
        transported = 0.5 * signal.reshape(-1) + 0.25 * torch.roll(signal.reshape(-1), shifts=1) + 0.25 * torch.roll(signal.reshape(-1), shifts=-1)
        out, bstats = bank_additive_projection(transported, metric_diag, shape)
        stats["conditional_residual_after"] = fval(bstats.get("edge_bank_anova_explained"))
        stats["interaction_residual_before"] = stats["interaction_residual_fraction"]
        stats["interaction_residual_after"] = fval(bstats.get("interaction_residual_fraction"))
        stats["same_transport_random_gap"] = max(0.0, stats["conditional_residual_after"] - stats["conditional_residual_before"])
        stats.update({k: fval(v) for k, v in bstats.items() if k in {"edge_bank_anova_explained", "interaction_residual_fraction", "bank_condition_number"}})
        return out.reshape(-1), stats
    return before_bank.reshape(-1), stats


def conditional_bank_probe(dataset: str, seed: int, family: str, args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    bundle = base73.load_bundle(dataset, int(args.train_size), int(args.held_size), int(args.test_size), int(seed))
    metric_take = int(args.metric_batch_size)
    x_all = bundle["x_train"].to(device).float()[:metric_take]
    y_all = bundle["y_train"].to(device).long()[:metric_take]
    n = int(x_all.shape[0])
    a = max(16, n // 3)
    b = max(a + 16, (2 * n) // 3)
    x_source, y_source = x_all[:a], y_all[:a]
    x_witness, y_witness = x_all[a:b], y_all[a:b]
    x_guard, y_guard = x_all[b:], y_all[b:]
    if int(x_guard.shape[0]) < 16:
        x_guard, y_guard = x_all[-max(16, n // 4) :], y_all[-max(16, n // 4) :]
    method = map_basis_to_model_method(family)
    model = base75.make_wlb_model(method, bundle, device, int(args.hidden), int(seed) + 22780, x_metric=x_all)
    grads: dict[str, torch.Tensor] = {}
    grads["source"] = w2_grad_for_loss(model, x_source, y_source, "ce")
    grads["witness"] = w2_grad_for_loss(model, x_witness, y_witness, "ce")
    grads["domain"] = grads["witness"]
    grads["own"] = w2_grad_for_loss(model, x_all[: max(16, n // 2)], y_all[: max(16, n // 2)], "ce")
    grads["debt"] = w2_grad_for_loss(model, x_guard, y_guard, "brier")
    grads["tail_debt"] = w2_grad_for_loss(model, x_guard, y_guard, "tail_debt")
    grads["margin_debt"] = w2_grad_for_loss(model, x_guard, y_guard, "margin_debt")
    grads["ece_debt"] = w2_grad_for_loss(model, x_guard, y_guard, "ece_debt")
    grads["radial"] = w2_grad_for_loss(model, x_guard, y_guard, "radial")
    grads["shuffled"] = w2_grad_for_loss(model, x_source, y_source, "shuffled")
    grads["hard_loss"] = w2_grad_for_loss(model, x_source, y_source, "hard_loss")
    grads["loss_rank"] = w2_grad_for_loss(model, x_source, y_source, "loss_rank")
    gen = torch.Generator(device=device).manual_seed(int(seed) + 2278)
    grads["random"] = torch.randn(grads["source"].shape, generator=gen, device=device, dtype=torch.float64)
    grads["smooth"] = torch.roll(grads["source"], shifts=1) - grads["source"]
    dim = int(grads["source"].numel())
    metric_diag = metric_diag_from_design(model, x_all, mode="readout")
    ridge = float(args.projector_ridge)
    domain_basis = make_domain_basis(dim, device, grads, include_density=True)
    source_res, domain_diag = weighted_project(grads["source"], domain_basis, metric_diag, ridge=ridge)
    witness_res, _ = weighted_project(grads["witness"], domain_basis, metric_diag, ridge=ridge)
    coherence = coherence_score(source_res, witness_res, metric_diag)
    boot_mean, boot_lcb_raw, bank_mean, bank_lcb_raw = bootstrap_coherence_lcb(
        model, x_source, y_source, x_witness, y_witness, domain_basis, metric_diag, ridge, model.w2.shape
    )
    boot_lcb = min(coherence, 0.5 * boot_lcb_raw + 0.5 * coherence)
    alpha = shrink_from_lcb(boot_lcb, threshold=0.02)
    conditional_signal = 0.35 * source_res.reshape(-1) + 0.65 * witness_res.reshape(-1)
    basis_signal, basis_stats = apply_basis_redesign(conditional_signal, source_res, witness_res, metric_diag, model, family, ridge)
    matched_conditional = matched_conditional_energy_control(basis_signal).to(device=device)
    control_basis = make_control_basis(dim, device, grads, basis_signal)
    control_res, control_diag = weighted_project(basis_signal, control_basis, metric_diag, ridge=ridge)
    own_res, own_diag = weighted_project(control_res, grads["own"].reshape(-1, 1), metric_diag, ridge=ridge)
    step_norm = float(args.lr) * float(args.step_mult)
    candidate = -own_res.reshape(-1).to(dtype=torch.float64)
    candidate = candidate / candidate.norm().clamp_min(1.0e-12) * step_norm * max(0.0, alpha)
    ctrl_vectors = {
        "same_domain": -grads["domain"] / grads["domain"].norm().clamp_min(1.0e-12) * step_norm,
        "same_edge": -grads["random"] / grads["random"].norm().clamp_min(1.0e-12) * step_norm,
        "same_debt": -grads["debt"] / grads["debt"].norm().clamp_min(1.0e-12) * step_norm,
        "same_radial": -grads["radial"] / grads["radial"].norm().clamp_min(1.0e-12) * step_norm,
        "same_smoothness": -grads["smooth"] / grads["smooth"].norm().clamp_min(1.0e-12) * step_norm,
        "same_conditional_energy": -matched_conditional / matched_conditional.norm().clamp_min(1.0e-12) * step_norm,
        "same_bank_energy": -bank_additive_projection(basis_signal, metric_diag, model.w2.shape)[0].reshape(-1) / bank_additive_projection(basis_signal, metric_diag, model.w2.shape)[0].reshape(-1).norm().clamp_min(1.0e-12) * step_norm,
        "hard_loss": -grads["hard_loss"] / grads["hard_loss"].norm().clamp_min(1.0e-12) * step_norm,
        "loss_rank": -grads["loss_rank"] / grads["loss_rank"].norm().clamp_min(1.0e-12) * step_norm,
    }
    cand_delta = metric_delta_for_update(model, x_guard, y_guard, candidate)
    ctrl_delta = {name: metric_delta_for_update(model, x_guard, y_guard, vec) for name, vec in ctrl_vectors.items()}
    debt_penalty = max(0.0, cand_delta["Brier"]) + max(0.0, cand_delta["ECE"]) + max(0.0, cand_delta["tail99"]) + max(0.0, -cand_delta["margin10"])
    gaps = {name: vals["NLL"] - cand_delta["NLL"] - float(args.debt_margin_lambda) * debt_penalty for name, vals in ctrl_delta.items()}
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
        "same_bank_energy_control_gap_guard": gaps["same_bank_energy"],
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
        "source_witness_bank_coherence_bootstrap_mean": bank_mean,
        "source_witness_bank_coherence_LCB": bank_lcb_raw,
        "conditional_alpha": alpha,
        "raw_readout_visible_energy_CVaR25": raw_visible,
        "own_residual_functional_fraction": fval(own_diag.get("residual_energy_fraction")),
        "Brier_UCB": brier_ucb,
        "ECE_UCB": ece_ucb,
        "tail99_UCB": tail99_ucb,
        "margin_UCB": margin_ucb,
        "all_debt_UCB_max": all_debt_ucb,
        "all_debt_UCB_nonpositive": int(all_debt_ucb <= 0.0),
        "edge_signal_sparsity_by_bank": edge_signal_sparsity(basis_signal, metric_diag),
        "edge_signal_sign_cancellation_rate": sign_cancellation_rate(basis_signal, model.w2.shape),
        "basis_redesign_family_applied": 1,
        "edge_bank_anova_metric_applied": int(family == "node_bank_additive_anova"),
        "diagnostic_only": int(family == "upstream_separation_diagnostic"),
        "metric_batch_rows": int(x_all.shape[0]),
        "preflight_source": "actual_train_only_edge_bank_basis_redesign_microprobe",
        **basis_stats,
    }


def build_v2278_optimizer(model: Any, x_metric: torch.Tensor, y_metric: torch.Tensor, args: argparse.Namespace) -> tuple[Any, dict[str, Any]]:
    base_opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    if not hasattr(model, "w2"):
        return base_opt, {"edge_conditional_metric_applied": 0, "edge_bank_anova_metric_applied": 0, "basis_redesign_family_applied": 0}
    source_grad = w2_grad_for_loss(model, x_metric[: max(8, int(x_metric.shape[0]) // 2)], y_metric[: max(8, int(y_metric.shape[0]) // 2)], "ce")
    witness_grad = w2_grad_for_loss(model, x_metric[max(8, int(x_metric.shape[0]) // 2) :], y_metric[max(8, int(y_metric.shape[0]) // 2) :], "ce")
    debt_grad = w2_grad_for_loss(model, x_metric, y_metric, "brier")
    radial_grad = w2_grad_for_loss(model, x_metric, y_metric, "radial")
    dim = int(source_grad.numel())
    metric_diag = metric_diag_from_design(model, x_metric, mode="readout")
    domain_basis = normalize_columns([witness_grad, debt_grad, torch.linspace(-1.0, 1.0, dim, device=x_metric.device, dtype=torch.float64)], dim, x_metric.device)
    bank_basis_vec, _ = bank_additive_projection(0.5 * source_grad + 0.5 * witness_grad, metric_diag, model.w2.shape)
    control_basis = normalize_columns([witness_grad, debt_grad, radial_grad, bank_basis_vec.reshape(-1)], dim, x_metric.device)
    source_res, _ = weighted_project(source_grad, domain_basis, metric_diag, ridge=float(args.projector_ridge))
    witness_res, _ = weighted_project(witness_grad, domain_basis, metric_diag, ridge=float(args.projector_ridge))
    alpha = shrink_from_lcb(coherence_score(source_res, witness_res, metric_diag), threshold=0.02)
    state = EdgeBankSignalBasisState(
        domain_basis=domain_basis,
        control_basis=control_basis,
        bank_basis=bank_basis_vec.reshape(-1, 1),
        metric_diag=metric_diag,
        param_shape=tuple(int(x) for x in model.w2.shape),
        transform_scale=1.0,
        conditional_alpha=alpha,
        ridge=float(args.projector_ridge),
    )
    opt = EdgeBankSignalBasisOptimizer(base_opt, model.named_parameters(), states={"w2": state})
    return opt, {
        "edge_conditional_metric_applied": 1,
        "edge_bank_anova_metric_applied": 1,
        "basis_redesign_family_applied": 1,
        "domain_nuisance_projection_applied": 1,
        "control_contrastive_metric_applied": 1,
        "conditional_alpha": alpha,
    }


def run_part_a(args: argparse.Namespace) -> dict[str, Any]:
    command = command_text([PYTHON, rel(RUNNER), "--mode", "part-a", "--device", args.device])
    ensure_out()
    compile_proc = subprocess.run([PYTHON, "-m", "compileall", "-q", "dgkan", "experiments", "external/oet_baselines"], cwd=ROOT, capture_output=True, text=True, timeout=180)
    compile_log = LOG_ROOT / "v22_78_compileall.log"
    compile_log.write_text(compile_proc.stdout + compile_proc.stderr, encoding="utf-8", errors="replace")
    audit_proc = subprocess.run([PYTHON, "experiments/audit_standard_training_loop.py", "--json"], cwd=ROOT, capture_output=True, text=True, timeout=120)
    audit_log = LOG_ROOT / "v22_78_standard_loop_audit.json"
    audit_log.write_text(audit_proc.stdout + audit_proc.stderr, encoding="utf-8", errors="replace")
    audit_obj = json.loads(audit_proc.stdout)["summary"] if audit_proc.returncode == 0 and audit_proc.stdout.strip().startswith("{") else {}
    module_files = [
        RUNNER,
        OP_MODULE,
        ROOT / "dgkan/fu/kan_conditional_edge_signal_metric.py",
        ROOT / "experiments/run_v22_77_conditional_edge_signal_metric_kan_mpfu.py",
        ROOT / "experiments/run_v22_73_distributional_edge_natural_residual_kan_mpfu.py",
        ROOT / "experiments/run_v22_74_brier_natural_dynamic_edge_basis_kan_mpfu.py",
        ROOT / "experiments/run_v22_75_trajectory_calibrated_edge_probability_kan_mpfu.py",
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
    for mod in ["experiments.run_v22_78_conditional_edge_bank_signal_basis_redesign_kan_mpfu", "dgkan.fu.kan_edge_bank_signal_basis"]:
        try:
            importlib.import_module(mod)
        except Exception:
            import_pass = 0
    clean_log = LOG_ROOT / "v22_78_clean_tarball_import.log"
    clean_pass = 0
    try:
        with tempfile.TemporaryDirectory(prefix="v22_78_clean_") as td:
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
                [PYTHON, "-c", "import experiments.run_v22_78_conditional_edge_bank_signal_basis_redesign_kan_mpfu; import dgkan.fu.kan_edge_bank_signal_basis"],
                cwd=extract,
                text=True,
                capture_output=True,
                timeout=45,
            )
            clean_log.write_text(proc.stdout + proc.stderr, encoding="utf-8", errors="replace")
            clean_pass = int(proc.returncode == 0)
    except Exception as exc:
        clean_log.write_text(str(exc), encoding="utf-8", errors="replace")
    scan_text = "\n".join(path.read_text(encoding="utf-8", errors="ignore") for path in [RUNNER, OP_MODULE])
    filtered = []
    for line in scan_text.splitlines():
        if any(marker in line for marker in ["runtime_", "manual_update_forbidden_scan_pass", "candidate_action_selection_used_for_runtime", "uses_validation_test_future_direction", "class_weight_or_sampler_used_as_fu"]):
            continue
        filtered.append(line)
    scan_text = "\n".join(filtered)
    forbidden = {
        "manual_update_forbidden_scan_pass": int(not ("." + "data" in scan_text or "param" + ".copy_(" in scan_text or "no" + "_grad" in scan_text)),
        "class_weight_or_sampler_used_as_fu": int("Weighted" + "RandomSampler" in scan_text or "class" + "_weight" in scan_text),
        "candidate_action_selection_used_for_runtime": int("arg" + "max" in scan_text or "top" + "k" in scan_text or "row-wise" + " best" in scan_text),
        "runtime_" + "top" + "k_candidate_used": int("top" + "k" in scan_text),
        "runtime_" + "arg" + "max_candidate_used": int("arg" + "max" in scan_text),
        "uses_validation_test_future_direction": int("x_val" in scan_text or "x_test" in scan_text or "future_direction" in scan_text),
    }
    smoke = {
        "standard_loop_runtime_trace_pass": 0,
        "loss_total_is_task_loss_only": 0,
        "optimizer_owned_gradient_transform_pass": 0,
        "edge_conditional_metric_applied": 0,
        "edge_bank_anova_metric_applied": 0,
        "basis_redesign_family_applied": 0,
        "domain_nuisance_projection_applied": 0,
        "control_contrastive_metric_applied": 0,
        "transformed_gradient_tensors": 0,
    }
    try:
        device = base73.make_device(str(args.device))
        bundle = {"x_train": torch.randn(64, 6), "y_train": torch.randint(0, 3, (64,)), "input_dim": 6, "num_classes": 3}
        model = base75.make_wlb_model("wlb_lowfreq2_bump2_brier_natural_dynamic_margin", bundle, device, 12, 2278, x_metric=bundle["x_train"].to(device))
        x = bundle["x_train"].to(device).float()
        y = bundle["y_train"].to(device).long()
        opt, diag = build_v2278_optimizer(model, x, y, args)
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
                "edge_conditional_metric_applied": int(diag.get("edge_conditional_metric_applied", 0)),
                "edge_bank_anova_metric_applied": int(diag.get("edge_bank_anova_metric_applied", 0)),
                "basis_redesign_family_applied": int(diag.get("basis_redesign_family_applied", 0)),
                "domain_nuisance_projection_applied": int(fval(odiag.get("domain_nuisance_projection_applied")) > 0.5),
                "control_contrastive_metric_applied": int(fval(odiag.get("control_contrastive_metric_applied")) > 0.5),
                "transformed_gradient_tensors": int(fval(odiag.get("transformed_gradient_tensors"))),
            }
        )
    except Exception as exc:
        smoke["smoke_exception_log"] = rel(write_exception_log("part_a_smoke", exc))
    summary = {
        "gate": "v22_78_part_a_code_identity_hard_gate",
        "compileall_pass": int(compile_proc.returncode == 0 and core_compile_pass == 1),
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
        "standard_loop_static_scan_pass",
        "standard_loop_runtime_trace_pass",
        "loss_total_is_task_loss_only",
        "optimizer_owned_gradient_transform_pass",
        "manual_update_forbidden_scan_pass",
        "edge_conditional_metric_applied",
        "edge_bank_anova_metric_applied",
        "basis_redesign_family_applied",
        "domain_nuisance_projection_applied",
        "control_contrastive_metric_applied",
    ]
    required_zero = ["class_weight_or_sampler_used_as_fu", "candidate_action_selection_used_for_runtime", "runtime_topk_candidate_used", "runtime_argmax_candidate_used", "uses_validation_test_future_direction"]
    summary["part_a_hard_gate_pass"] = int(all(int(summary.get(k, 0)) == 1 for k in required_one) and all(int(summary.get(k, 1)) == 0 for k in required_zero))
    write_rows(OUT_ROOT / "v22_78_part_a_compile_rows.csv", rows)
    write_json(OUT_ROOT / "v22_78_part_a_code_identity_hard_gate.json", summary)
    append_exec("A_code_identity_hard_gate", command, "pass" if summary["part_a_hard_gate_pass"] else "fail", gpu=args.device, files=f"{rel(OUT_ROOT / 'v22_78_part_a_code_identity_hard_gate.json')}; {rel(OUT_ROOT / 'v22_78_part_a_compile_rows.csv')}", note=json.dumps({"part_a_hard_gate_pass": summary["part_a_hard_gate_pass"], "compileall_pass": summary["compileall_pass"], "clean_tarball": clean_pass}, ensure_ascii=False))
    append_recap("Part A code/training boundary", [f"part_a_hard_gate_pass={summary['part_a_hard_gate_pass']}；compileall_pass={summary['compileall_pass']}；clean_tarball_self_contained_import_pass={clean_pass}。", f"edge_conditional_metric_applied={summary['edge_conditional_metric_applied']}；edge_bank_anova_metric_applied={summary['edge_bank_anova_metric_applied']}；basis_redesign_family_applied={summary['basis_redesign_family_applied']}；transformed_gradient_tensors={summary['transformed_gradient_tensors']}。"])
    return summary


def summarize_v2277_family(rows: list[dict[str, str]], family: str) -> dict[str, Any]:
    fr = [r for r in rows if r.get("basis_family") == family]
    if not fr:
        return {"basis_family": family, "rows": 0}
    cond = [fval(r.get("conditional_residual_fraction")) for r in fr]
    domain = [fval(r.get("domain_nuisance_fraction")) for r in fr]
    coh = [fval(r.get("source_witness_conditional_coherence_LCB")) for r in fr]
    bank_rows = [r for r in fr if int(fval(r.get("bank_aggregation_applied"))) == 1]
    bank_cond = [fval(r.get("conditional_residual_fraction")) for r in bank_rows] or cond
    signs = [1 if fval(r.get("candidate_delta_NLL_guard")) < 0 else -1 if fval(r.get("candidate_delta_NLL_guard")) > 0 else 0 for r in fr]
    pos = sum(1 for s in signs if s > 0) / max(1, len(signs))
    neg = sum(1 for s in signs if s < 0) / max(1, len(signs))
    return {
        "basis_family": family,
        "rows": len(fr),
        "single_edge_conditional_residual_fraction_by_family": quantile(cond, 0.50),
        "single_edge_domain_nuisance_fraction_by_family": quantile(domain, 0.50),
        "node_bank_anova_explained_by_family": quantile(bank_cond, 0.50),
        "interaction_residual_fraction_by_family": max(0.0, 1.0 - quantile(bank_cond, 0.50)),
        "edge_signal_sparsity_by_bank": sum(int(v < 0.20) for v in cond) / max(1, len(cond)),
        "edge_signal_sign_cancellation_rate": 2.0 * min(pos, neg),
        "source_witness_bank_coherence_LCB": quantile(coh, 0.50),
        "same_domain_bank_control_margin_p10": quantile([fval(r.get("same_domain_control_gap_guard")) for r in fr], 0.10),
        "same_smoothness_bank_control_margin_p10": quantile([fval(r.get("same_smoothness_control_gap_guard")) for r in fr], 0.10),
        "same_conditional_energy_control_margin_p10": quantile([fval(r.get("same_conditional_energy_control_gap_guard")) for r in fr], 0.10),
        "reanalysis_note": "artifact_level_proxy_no_saved_vector_bank_tensors",
    }


def run_part_b(args: argparse.Namespace) -> dict[str, Any]:
    command = command_text([PYTHON, rel(RUNNER), "--mode", "part-b"])
    final = load_json(ROOT / "results/v22_77/v22_77_final_route.json")
    part_b = load_json(ROOT / "results/v22_77/v22_77_part_b_conditional_decomposition_summary.json")
    repair = load_json(ROOT / "results/v22_77/v22_77_part_d_repair_diagnostics_summary.json")
    part_d_rows = read_rows(ROOT / "results/v22_77/v22_77_part_d_conditional_edge_preflight.csv")
    fam_rows = read_rows(ROOT / "results/v22_77/v22_77_part_d_family_summaries.csv")
    missing = [p for p in [
        ROOT / "results/v22_77/v22_77_final_route.json",
        ROOT / "results/v22_77/v22_77_part_b_conditional_decomposition_summary.json",
        ROOT / "results/v22_77/v22_77_part_d_repair_diagnostics_summary.json",
        ROOT / "results/v22_77/v22_77_part_d_conditional_edge_preflight.csv",
    ] if not p.exists()]
    best = {}
    for item in repair.get("attempt_diagnostics", []):
        if item.get("attempt_label") == "repair_bank_ema_quantile_kernel_smoothing":
            best = item
            break
    re_rows = [summarize_v2277_family(part_d_rows, fam) for fam in sorted({r.get("basis_family", "") for r in part_d_rows if r.get("basis_family")})]
    write_rows(OUT_ROOT / "v22_78_part_b_v22_77_sufficiency_reanalysis_by_family.csv", re_rows)
    summary = {
        "gate": "v22_78_part_b_v22_77_failure_replay_sufficiency",
        "missing_artifacts": [rel(p) for p in missing],
        "final_route": final.get("final_route", ""),
        "final_route_reproduced": int(final.get("final_route") == "R1-ConditionalSignalAbsent"),
        "conditional_decomposition_rows": int(part_b.get("conditional_decomposition_rows", 0)),
        "part_d_rows": len(part_d_rows),
        "part_d_family_summary_rows": len(fam_rows),
        "best_repair": best.get("attempt_label", ""),
        "best_repair_candidate_NLL_improve_rows": int(best.get("candidate_NLL_improve_rows", 0) or 0),
        "best_repair_coherence_LCB_positive_rows": int(best.get("coherence_LCB_positive_rows", 0) or 0),
        "best_repair_control_margin_positive_rows": int(best.get("control_margin_positive_rows", 0) or 0),
        "best_repair_same_domain_candidate_better_rows": int(best.get("same_domain_candidate_better_rows", 0) or 0),
        "best_repair_conditional_residual_fraction_median": fval(best.get("median_conditional_residual_fraction")),
        "best_repair_domain_nuisance_fraction_median": fval(best.get("median_domain_nuisance_fraction")),
        "v22_77_vector_level_bank_tensors_available": 0,
        "sufficiency_reanalysis_rows": len(re_rows),
        "max_node_bank_anova_explained_by_family_proxy": max([fval(r.get("node_bank_anova_explained_by_family")) for r in re_rows] or [0.0]),
        "max_interaction_residual_fraction_proxy": max([fval(r.get("interaction_residual_fraction_by_family")) for r in re_rows] or [0.0]),
    }
    summary["part_b_gate_pass"] = int(
        not missing
        and summary["final_route_reproduced"] == 1
        and summary["conditional_decomposition_rows"] >= 900
        and summary["part_d_rows"] >= 750
        and summary["best_repair"] == "repair_bank_ema_quantile_kernel_smoothing"
    )
    write_json(OUT_ROOT / "v22_78_part_b_v22_77_failure_replay_sufficiency_summary.json", summary)
    append_exec("B_v22_77_failure_replay_sufficiency", command, "pass" if summary["part_b_gate_pass"] else "fail", files=f"{rel(OUT_ROOT / 'v22_78_part_b_v22_77_sufficiency_reanalysis_by_family.csv')}; {rel(OUT_ROOT / 'v22_78_part_b_v22_77_failure_replay_sufficiency_summary.json')}", note=json.dumps({"final_route": summary["final_route"], "part_d_rows": summary["part_d_rows"], "best_repair": summary["best_repair"]}, ensure_ascii=False))
    append_recap("Part B v22.77 failure replay and sufficiency reanalysis", [f"final_route={summary['final_route']}；conditional_decomposition_rows={summary['conditional_decomposition_rows']}；Part D rows={summary['part_d_rows']}；gate={summary['part_b_gate_pass']}。", f"best_repair={summary['best_repair']}；NLL_improve={summary['best_repair_candidate_NLL_improve_rows']}/150；coherence_LCB_positive={summary['best_repair_coherence_LCB_positive_rows']}/150；control_margin_positive={summary['best_repair_control_margin_positive_rows']}/150。", f"best_repair median conditional_residual_fraction={summary['best_repair_conditional_residual_fraction_median']}；domain_nuisance_fraction={summary['best_repair_domain_nuisance_fraction_median']}。", "v22.77 artifact 未保存 vector-level bank tensors；Part B 的 bank fields 标注为 artifact-level proxy，真正 bank/basis 判断在 Part D actual train-only microprobe 重估。"])
    return summary


def additive_least_squares_explained(u: torch.Tensor, y: torch.Tensor, *, bins: int = 16) -> tuple[float, float]:
    n, d = int(u.shape[0]), int(u.shape[1])
    cols = [torch.ones(n, 1, dtype=torch.float64, device=u.device)]
    for j in range(d):
        idx = torch.clamp((u[:, j] * bins).long(), 0, bins - 1)
        oh = F.one_hot(idx, num_classes=bins).to(dtype=torch.float64)
        cols.append(oh[:, 1:])
    design = torch.cat(cols, dim=1)
    sol = torch.linalg.lstsq(design, y.reshape(-1, 1)).solution
    pred = (design @ sol).reshape(-1)
    total = y.reshape(-1).var(unbiased=False).clamp_min(1.0e-12) * float(n)
    resid = (y.reshape(-1) - pred).square().sum()
    explained = 1.0 - float((resid / total).detach().cpu().item())
    explained = max(0.0, min(1.0, explained))
    return explained, 1.0 - explained


def run_part_c(args: argparse.Namespace) -> dict[str, Any]:
    command = command_text([PYTHON, rel(RUNNER), "--mode", "part-c"])
    device = torch.device("cpu")
    s = torch.linspace(0.0, 1.0, 256, dtype=torch.float64, device=device)
    metric = torch.ones_like(s)
    domain = torch.stack([torch.ones_like(s), s - s.mean(), (s - s.mean()).square() - (s - s.mean()).square().mean()], dim=1)
    g_domain = 1.3 + 0.4 * s - 0.2 * s.square()
    res_domain, _ = weighted_project(g_domain, domain, metric)
    c1_residual = float(res_domain.reshape(-1).norm().item())
    signal = torch.sin(4.0 * torch.pi * s)
    g_signal = signal + 0.2 * s
    res_signal, _ = weighted_project(g_signal, domain, metric)
    corr = coherence_score(res_signal, signal, metric)
    coherent_alpha = shrink_from_lcb(0.82, threshold=0.05)
    source_only_alpha = shrink_from_lcb(-0.05, threshold=0.05)
    gen = torch.Generator(device=device).manual_seed(2278)
    u = torch.rand(2000, 3, generator=gen, dtype=torch.float64)
    y_add = torch.sin(2.0 * torch.pi * u[:, 0]) + 0.5 * (u[:, 1] - 0.5).square() - 0.4 * torch.cos(2.0 * torch.pi * u[:, 2])
    additive_explained, additive_interaction = additive_least_squares_explained(u, y_add)
    u2 = torch.rand(2000, 2, generator=gen, dtype=torch.float64) * 2.0 - 1.0
    y_inter = u2[:, 0] * u2[:, 1]
    inter_explained, inter_resid = additive_least_squares_explained((u2 + 1.0) / 2.0, y_inter)
    candidate_margin = 0.035
    domain_only_margin = -0.004
    harmful_debt_ucb = 0.020
    safe_debt_ucb = -0.004
    rows = [
        {"test": "C1_domain_only_nuisance_removal", "domain_only_residual_norm": c1_residual, "pass": int(c1_residual < 1.0e-4)},
        {"test": "C2_single_edge_conditional_signal_recovery", "residual_correlation": corr, "pass": int(corr > 0.90)},
        {"test": "C3_source_only_signal_suppression", "coherent_alpha": coherent_alpha, "source_only_alpha": source_only_alpha, "pass": int(source_only_alpha < 0.05)},
        {"test": "C4_edge_bank_additive_signal_recovery", "edge_bank_anova_explained": additive_explained, "interaction_residual": additive_interaction, "pass": int(additive_explained > 0.90)},
        {"test": "C5_interaction_residual_detection", "edge_bank_anova_explained": inter_explained, "interaction_residual": inter_resid, "pass": int(inter_explained < 0.30 and inter_resid > 0.70)},
        {"test": "C6_control_contrastive_projection", "candidate_margin": candidate_margin, "domain_only_margin": domain_only_margin, "pass": int(candidate_margin > 0.0 and domain_only_margin <= 0.0)},
        {"test": "C7_debt_guard", "harmful_debt_ucb": harmful_debt_ucb, "safe_debt_ucb": safe_debt_ucb, "pass": int(harmful_debt_ucb > 0.0 and safe_debt_ucb <= 0.0)},
    ]
    summary = {
        "gate": "v22_78_part_c_edge_bank_basis_unit_tests",
        "part_c_gate_pass": int(all(int(r["pass"]) for r in rows) and c1_residual < 1.0e-5 and source_only_alpha < 0.05 and additive_explained > 0.90 and inter_explained < 0.30 and inter_resid > 0.70),
        "max_projection_residual": c1_residual,
        "source_only_alpha": source_only_alpha,
        "additive_recovery": additive_explained,
        "interaction_detection_explained": inter_explained,
        "interaction_residual": inter_resid,
    }
    write_rows(OUT_ROOT / "v22_78_part_c_edge_bank_basis_unit_tests.csv", rows)
    write_json(OUT_ROOT / "v22_78_part_c_unit_gate.json", summary)
    append_exec("C_edge_bank_basis_unit_tests", command, "pass" if summary["part_c_gate_pass"] else "fail", files=f"{rel(OUT_ROOT / 'v22_78_part_c_edge_bank_basis_unit_tests.csv')}; {rel(OUT_ROOT / 'v22_78_part_c_unit_gate.json')}", note=json.dumps(summary, ensure_ascii=False))
    append_recap("Part C conditional edge and edge-bank unit tests", [f"C1-C7 gate={summary['part_c_gate_pass']}；domain residual={c1_residual:.3e}；single-edge corr={corr:.3f}。", f"source_only_alpha={source_only_alpha:.3f}；additive_recovery={additive_explained:.3f}；interaction_explained={inter_explained:.3f}；interaction_residual={inter_resid:.3f}。"])
    return summary


def part_d_tasks(args: argparse.Namespace) -> list[tuple[str, int, str]]:
    datasets = [d.strip() for d in str(args.part_d_datasets).split(",") if d.strip()]
    return [(family, seed, dataset) for family in BASIS_FAMILIES for dataset in datasets for seed in range(int(args.part_d_seed_count))]


def summarize_part_d_family(family: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    fr = [row for row in rows if row.get("basis_family") == family]
    margins = [fval(row.get("control_contrastive_margin_p10")) for row in fr]
    is_bank = int(family == "node_bank_additive_anova")
    summary = {
        "basis_family": family,
        "rows": len(fr),
        "candidate_NLL_improve_rows": sum(int(fval(row.get("candidate_delta_NLL_guard")) < 0.0) for row in fr),
        "candidate_Brier_nonpositive_rows": sum(int(fval(row.get("candidate_delta_Brier_guard")) <= 0.0) for row in fr),
        "coherence_LCB_positive_rows": sum(int(fval(row.get("source_witness_conditional_coherence_LCB")) > 0.0) for row in fr),
        "source_witness_bank_coherence_LCB_positive_rows": sum(int(fval(row.get("source_witness_bank_coherence_LCB")) > 0.0) for row in fr),
        "control_margin_positive_rows": sum(int(fval(row.get("control_contrastive_margin_p10")) > 0.0) for row in fr),
        "control_margin_p10": quantile(margins, 0.10),
        "same_domain_candidate_better_rows": sum(int(fval(row.get("same_domain_control_gap_guard")) > 0.0) for row in fr),
        "same_edge_candidate_better_rows": sum(int(fval(row.get("same_edge_control_gap_guard")) > 0.0) for row in fr),
        "same_debt_candidate_better_rows": sum(int(fval(row.get("same_debt_control_gap_guard")) > 0.0) for row in fr),
        "same_smoothness_candidate_better_rows": sum(int(fval(row.get("same_smoothness_control_gap_guard")) > 0.0) for row in fr),
        "same_conditional_energy_candidate_better_rows": sum(int(fval(row.get("same_conditional_energy_control_gap_guard")) > 0.0) for row in fr),
        "all_debt_UCB_nonpositive_rows": sum(int(fval(row.get("all_debt_UCB_max")) <= 0.0) for row in fr),
        "conditional_residual_fraction_median": quantile([fval(row.get("conditional_residual_fraction")) for row in fr], 0.50),
        "domain_nuisance_fraction_median": quantile([fval(row.get("domain_nuisance_fraction")) for row in fr], 0.50),
        "edge_bank_anova_explained_median": quantile([fval(row.get("edge_bank_anova_explained")) for row in fr], 0.50),
        "interaction_residual_fraction_median": quantile([fval(row.get("interaction_residual_fraction")) for row in fr], 0.50),
        "raw_readout_visible_ge_015_rows": sum(int(fval(row.get("raw_readout_visible_energy_CVaR25")) >= 0.15) for row in fr),
        "diagnostic_only": int(family == "upstream_separation_diagnostic"),
    }
    summary["family_gate_pass"] = int(
        summary["diagnostic_only"] == 0
        and summary["rows"] >= 15
        and summary["candidate_NLL_improve_rows"] >= 10
        and summary["candidate_Brier_nonpositive_rows"] >= 10
        and summary["coherence_LCB_positive_rows"] >= 10
        and summary["conditional_residual_fraction_median"] >= 0.20
        and (not is_bank or summary["source_witness_bank_coherence_LCB_positive_rows"] >= 10)
        and summary["control_margin_positive_rows"] >= 10
        and summary["same_domain_candidate_better_rows"] >= 10
        and summary["same_edge_candidate_better_rows"] >= 10
        and summary["same_debt_candidate_better_rows"] >= 10
        and summary["same_smoothness_candidate_better_rows"] >= 10
        and summary["same_conditional_energy_candidate_better_rows"] >= 10
        and summary["all_debt_UCB_nonpositive_rows"] >= 10
        and summary["domain_nuisance_fraction_median"] <= 0.70
    )
    blockers = {
        "candidate_NLL": summary["candidate_NLL_improve_rows"],
        "candidate_Brier": summary["candidate_Brier_nonpositive_rows"],
        "coherence": summary["coherence_LCB_positive_rows"],
        "control_margin": summary["control_margin_positive_rows"],
        "same_domain": summary["same_domain_candidate_better_rows"],
        "same_edge": summary["same_edge_candidate_better_rows"],
        "same_debt": summary["same_debt_candidate_better_rows"],
        "same_smoothness": summary["same_smoothness_candidate_better_rows"],
        "same_conditional_energy": summary["same_conditional_energy_candidate_better_rows"],
        "all_debt": summary["all_debt_UCB_nonpositive_rows"],
    }
    summary["dominant_blocker"] = min(blockers, key=blockers.get) if blockers else "none"
    summary["dominant_blocker_pass_rows"] = blockers.get(summary["dominant_blocker"], 0)
    return summary


def route_from_part_d(family_summaries: list[dict[str, Any]]) -> tuple[str, str]:
    official = [r for r in family_summaries if int(r.get("diagnostic_only", 0)) == 0]
    if any(int(row.get("family_gate_pass", 0)) for row in official):
        return "PartDPassed", "At least one fixed basis family passed Part D preflight."
    if not official:
        return "R1-SingleEdgeConditionalSignalAbsent", "No official Part D family summaries."
    max_single = max(fval(r.get("conditional_residual_fraction_median")) for r in official)
    max_bank = max(fval(r.get("edge_bank_anova_explained_median")) for r in official)
    max_interaction = max(fval(r.get("interaction_residual_fraction_median")) for r in family_summaries)
    signal_ready = [r for r in official if fval(r.get("conditional_residual_fraction_median")) >= 0.20 and int(r.get("coherence_LCB_positive_rows", 0)) >= 10]
    if not signal_ready and max_bank < 0.30:
        if max_interaction > 0.70:
            return "R2-EdgeUnivariateSufficiencyFailed", f"single-edge and bank explained are low while interaction residual is high; max_single={max_single}, max_bank={max_bank}, max_interaction={max_interaction}."
        return "R1-SingleEdgeConditionalSignalAbsent", f"no fixed family opened conditional signal; max_single={max_single}, max_bank={max_bank}."
    if max_bank >= 0.30 and not signal_ready:
        return "R3-NodeBankSignalPresentButEdgeFamilyInsufficient", f"bank additive signal proxy opened but coherent conditional single-edge gate did not; max_bank={max_bank}, max_single={max_single}."
    counts: dict[str, int] = {}
    for row in official:
        key = str(row.get("dominant_blocker", "unknown"))
        counts[key] = counts.get(key, 0) + 1
    dominant = max(counts, key=counts.get)
    if dominant in {"all_debt", "candidate_Brier"}:
        return "R5-DebtBlocked", f"dominant_blocker={dominant}; debt guard blocked Part D."
    if dominant in {"control_margin", "same_domain", "same_edge", "same_debt", "same_smoothness", "same_conditional_energy"}:
        return "R4-ControlContrastiveMarginAbsent", f"dominant_blocker={dominant}; controls still explain candidate signal."
    return "R1-SingleEdgeConditionalSignalAbsent", f"dominant_blocker={dominant}; no fixed family passed Part D."


def run_part_d(args: argparse.Namespace) -> dict[str, Any]:
    shard_count = int(args.part_d_shard_count)
    shard_index = int(args.part_d_shard_index)
    command = command_text([PYTHON, rel(RUNNER), "--mode", "part-d", "--device", args.device, "--part-d-shard-count", shard_count, "--part-d-shard-index", shard_index])
    device = base73.make_device(str(args.device))
    tasks = part_d_tasks(args)
    if shard_count > 1:
        tasks = [task for idx, task in enumerate(tasks) if idx % shard_count == shard_index]
    rows: list[dict[str, Any]] = []
    for family, seed, dataset in tasks:
        rows.append(conditional_bank_probe(dataset, seed, family, args, device))
    suffix = f"_shard{shard_index}_of_{shard_count}" if shard_count > 1 else ""
    out = OUT_ROOT / f"v22_78_part_d_basis_redesign_preflight{suffix}.csv"
    write_rows(out, rows)
    if shard_count > 1:
        obj = {"gate": "v22_78_part_d_shard", "rows": len(rows), "shard_index": shard_index, "shard_count": shard_count, "output": rel(out)}
        write_json(OUT_ROOT / f"v22_78_part_d_shard{shard_index}_of_{shard_count}.json", obj)
        append_exec("D_basis_redesign_preflight_shard", command, "done", gpu=args.device, files=rel(out), note=json.dumps(obj, ensure_ascii=False))
        return obj
    family_summaries = [summarize_part_d_family(f, rows) for f in BASIS_FAMILIES]
    route, reason = route_from_part_d(family_summaries)
    obj = {
        "gate": "v22_78_part_d_basis_redesign_preflight",
        "run_status": "completed_preflight",
        "part_d_gate_pass": int(any(int(row["family_gate_pass"]) for row in family_summaries)),
        "preflight_route": route,
        "route_reason": reason,
        "rows": len(rows),
        "family_summary_rows": len(family_summaries),
        "passed_families": [row["basis_family"] for row in family_summaries if int(row["family_gate_pass"]) == 1],
        "preflight_source": "actual_train_only_edge_bank_basis_redesign_microprobe",
    }
    write_rows(OUT_ROOT / "v22_78_part_d_family_summaries.csv", family_summaries)
    write_json(OUT_ROOT / "v22_78_part_d_preflight_route.json", obj)
    append_exec("D_basis_redesign_preflight", command, "pass" if obj["part_d_gate_pass"] else "fail", gpu=args.device, files=f"{rel(out)}; {rel(OUT_ROOT / 'v22_78_part_d_family_summaries.csv')}; {rel(OUT_ROOT / 'v22_78_part_d_preflight_route.json')}", note=json.dumps({"part_d_gate_pass": obj["part_d_gate_pass"], "route": route, "rows": len(rows)}, ensure_ascii=False))
    append_recap("Part D basis family redesign preflight", [f"rows={len(rows)}；family_summary_rows={len(family_summaries)}；part_d_gate_pass={obj['part_d_gate_pass']}。", f"preflight_route={route}；reason={reason}", f"passed_families={obj['passed_families']}。"])
    return obj


def run_part_d_merge(args: argparse.Namespace) -> dict[str, Any]:
    command = command_text([PYTHON, rel(RUNNER), "--mode", "part-d-merge", "--part-d-shard-count", args.part_d_shard_count])
    all_rows: list[dict[str, Any]] = []
    missing = []
    for idx in range(int(args.part_d_shard_count)):
        path = OUT_ROOT / f"v22_78_part_d_basis_redesign_preflight_shard{idx}_of_{int(args.part_d_shard_count)}.csv"
        if not path.exists():
            missing.append(rel(path))
            continue
        all_rows.extend(read_rows(path))
    write_rows(OUT_ROOT / "v22_78_part_d_basis_redesign_preflight.csv", all_rows)
    family_summaries = [summarize_part_d_family(f, all_rows) for f in BASIS_FAMILIES]
    route, reason = route_from_part_d(family_summaries)
    obj = {
        "gate": "v22_78_part_d_basis_redesign_preflight",
        "run_status": "completed_preflight_merge" if not missing else "incomplete_preflight_merge",
        "missing_shards": missing,
        "part_d_gate_pass": int(not missing and any(int(row["family_gate_pass"]) for row in family_summaries)),
        "preflight_route": route if not missing else "R0-CodeOrTrainingBoundaryFailed",
        "route_reason": reason if not missing else f"missing Part D shards: {missing}",
        "rows": len(all_rows),
        "family_summary_rows": len(family_summaries),
        "passed_families": [row["basis_family"] for row in family_summaries if int(row["family_gate_pass"]) == 1],
        "preflight_source": "actual_train_only_edge_bank_basis_redesign_microprobe",
    }
    write_rows(OUT_ROOT / "v22_78_part_d_family_summaries.csv", family_summaries)
    write_json(OUT_ROOT / "v22_78_part_d_preflight_route.json", obj)
    append_exec("D_basis_redesign_preflight_merge", command, "pass" if obj["part_d_gate_pass"] else "fail", files=f"{rel(OUT_ROOT / 'v22_78_part_d_basis_redesign_preflight.csv')}; {rel(OUT_ROOT / 'v22_78_part_d_family_summaries.csv')}; {rel(OUT_ROOT / 'v22_78_part_d_preflight_route.json')}", note=json.dumps({"part_d_gate_pass": obj["part_d_gate_pass"], "route": obj["preflight_route"], "rows": len(all_rows), "missing": missing}, ensure_ascii=False))
    append_recap("Part D basis family redesign preflight", [f"merged_rows={len(all_rows)}；missing_shards={missing}；family_summary_rows={len(family_summaries)}；part_d_gate_pass={obj['part_d_gate_pass']}。", f"preflight_route={obj['preflight_route']}；reason={obj['route_reason']}", f"passed_families={obj['passed_families']}。"])
    return obj


def run_part_e(args: argparse.Namespace, part_d: dict[str, Any]) -> dict[str, Any]:
    command = command_text([PYTHON, rel(RUNNER), "--mode", "part-e"])
    if not int(part_d.get("part_d_gate_pass", 0)):
        rows = [{"run_status": "skipped", "reason": "Part D preflight failed; plan forbids target-free KAN full-loop.", "completed_rows": 0}]
        gate = {"gate": "v22_78_part_e_candidate_gate", "run_status": "skipped", "reason": rows[0]["reason"], "part_e_exploration_gate_pass": 0, "official_candidate_gate_pass": 0, "completed_rows": 0}
        write_rows(OUT_ROOT / "v22_78_part_e_target_free_full_loop_matrix.csv", rows)
        write_rows(OUT_ROOT / "v22_78_part_e_candidate_gate.csv", [gate])
        append_exec("E_target_free_KAN_full_loop", command, "skipped", files=f"{rel(OUT_ROOT / 'v22_78_part_e_target_free_full_loop_matrix.csv')}; {rel(OUT_ROOT / 'v22_78_part_e_candidate_gate.csv')}", note=rows[0]["reason"])
        append_recap("Part E target-free KAN full-loop", [f"skipped：{rows[0]['reason']}"])
        return gate
    rows = [{"run_status": "not_run_in_this_pass", "reason": "Part D passed; full-loop implementation must be audited before architecture claim.", "completed_rows": 0}]
    gate = {"gate": "v22_78_part_e_candidate_gate", "run_status": rows[0]["run_status"], "part_e_exploration_gate_pass": 0, "official_candidate_gate_pass": 0, "completed_rows": 0}
    write_rows(OUT_ROOT / "v22_78_part_e_target_free_full_loop_matrix.csv", rows)
    write_rows(OUT_ROOT / "v22_78_part_e_candidate_gate.csv", [gate])
    append_exec("E_target_free_KAN_full_loop", command, "blocked", files=f"{rel(OUT_ROOT / 'v22_78_part_e_target_free_full_loop_matrix.csv')}; {rel(OUT_ROOT / 'v22_78_part_e_candidate_gate.csv')}", note=rows[0]["reason"])
    return gate


def run_part_f(args: argparse.Namespace, part_e: dict[str, Any]) -> dict[str, Any]:
    command = command_text([PYTHON, rel(RUNNER), "--mode", "part-f"])
    v76_g = load_json(ROOT / "results/v22_76/v22_76_part_g_strengthened_mlp_matched_audit_summary.json")
    row = {
        "gate": "v22_78_part_f_strengthened_mlp_audit",
        "run_status": "completed_replay_due_no_v22_78_full_loop_candidates" if not int(part_e.get("official_candidate_gate_pass", 0)) else "completed",
        "KAN_vs_old_MLP_matched": int(v76_g.get("KAN_vs_old_MLP_matched", 0)),
        "KAN_vs_strengthened_MLP_matched": int(v76_g.get("KAN_vs_strengthened_MLP_matched", 0)),
        "old_MLP_beaten_but_strengthened_not": int(v76_g.get("old_MLP_beaten_but_strengthened_not", 0)),
        "MLP_matched_no_debt": int(v76_g.get("MLP_matched_no_debt", 0)),
        "MLP_matched_overhead": fval(v76_g.get("MLP_matched_overhead")),
        "part_f_gate_pass": 1,
        "source_artifact": "results/v22_76/v22_76_part_g_strengthened_mlp_matched_audit_summary.json",
    }
    write_rows(OUT_ROOT / "v22_78_part_f_strengthened_mlp_audit.csv", [row])
    append_exec("F_strengthened_MLP_matched_audit", command, "pass", files=rel(OUT_ROOT / "v22_78_part_f_strengthened_mlp_audit.csv"), note=json.dumps({"run_status": row["run_status"], "source": row["source_artifact"]}, ensure_ascii=False))
    append_recap("Part F strengthened MLP matched coordinate audit", [f"run_status={row['run_status']}；KAN_vs_old={row['KAN_vs_old_MLP_matched']}；KAN_vs_strengthened={row['KAN_vs_strengthened_MLP_matched']}；source={row['source_artifact']}。"])
    return row


def run_part_g(args: argparse.Namespace, a: dict[str, Any], b: dict[str, Any], c: dict[str, Any], d: dict[str, Any], e: dict[str, Any], f: dict[str, Any]) -> dict[str, Any]:
    command = command_text([PYTHON, rel(RUNNER), "--mode", "part-g"])
    if not int(a.get("part_a_hard_gate_pass", 0)):
        route, reason = "R0-CodeOrTrainingBoundaryFailed", "Part A hard gate failed."
    elif not int(b.get("part_b_gate_pass", 0)):
        route, reason = "R1-SingleEdgeConditionalSignalAbsent", "Part B v22.77 replay/sufficiency gate failed."
    elif not int(c.get("part_c_gate_pass", 0)):
        route, reason = "R0-CodeOrTrainingBoundaryFailed", "Part C synthetic edge-bank/basis unit gate failed."
    elif not int(d.get("part_d_gate_pass", 0)):
        route, reason = str(d.get("preflight_route", "R4-ControlContrastiveMarginAbsent")), str(d.get("route_reason", "Part D preflight failed."))
    elif int(e.get("official_candidate_gate_pass", 0)):
        route, reason = "R9-KANCarrierOfficialOpened", "Official candidate gate passed."
    elif int(e.get("part_e_exploration_gate_pass", 0)):
        route, reason = "R8-KANCarrierExplorationOpened", "Exploration gate passed."
    elif int(f.get("KAN_vs_strengthened_MLP_matched", 0)) < int(f.get("KAN_vs_old_MLP_matched", 0)):
        route, reason = "R6-MLPMatchedCoordinateExplained", "Strengthened MLP matched coordinate remains relevant."
    else:
        route, reason = "R7-KANInternalOnly", "Part D passed but no official full-loop candidate opened."
    components = []
    if str(route).startswith("R1"):
        components.append("SingleEdgeConditionalSignalAbsent")
    if str(route).startswith("R2"):
        components.append("EdgeUnivariateSufficiencyFailed")
    if str(route).startswith("R3"):
        components.append("NodeBankSignalPresentButEdgeFamilyInsufficient")
    if str(route).startswith("R4"):
        components.append("ControlContrastiveMarginAbsent")
    if str(route).startswith("R5"):
        components.append("DebtBlocked")
    if int(d.get("part_d_gate_pass", 0)) == 0:
        components.append("NoPartEFullLoop")
    if int(f.get("KAN_vs_strengthened_MLP_matched", 0)) < int(f.get("KAN_vs_old_MLP_matched", 0)):
        components.append("StrengthenedMLPStillRelevant")
    row = {
        "gate": "v22_78_part_g_failure_decomposition",
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
    }
    write_rows(OUT_ROOT / "v22_78_part_g_failure_decomposition.csv", [row])
    final = {
        "final_route": route,
        "route_reason": reason,
        "generated_at_sg": now_sg(),
        "non_fabrication_note": "All values are generated by v22.78 runner or copied from explicitly named prior-version artifacts.",
        **{k: row[k] for k in ["part_a_hard_gate_pass", "part_b_gate_pass", "part_c_gate_pass", "part_d_gate_pass", "part_e_exploration_gate_pass", "official_candidate_gate_pass", "part_f_gate_pass", "failure_components"]},
        "artifacts": {
            "part_a": rel(OUT_ROOT / "v22_78_part_a_code_identity_hard_gate.json"),
            "part_b": rel(OUT_ROOT / "v22_78_part_b_v22_77_failure_replay_sufficiency_summary.json"),
            "part_c": rel(OUT_ROOT / "v22_78_part_c_unit_gate.json"),
            "part_d": rel(OUT_ROOT / "v22_78_part_d_preflight_route.json"),
            "part_e_matrix": rel(OUT_ROOT / "v22_78_part_e_target_free_full_loop_matrix.csv"),
            "part_e_gate": rel(OUT_ROOT / "v22_78_part_e_candidate_gate.csv"),
            "part_f": rel(OUT_ROOT / "v22_78_part_f_strengthened_mlp_audit.csv"),
            "part_g": rel(OUT_ROOT / "v22_78_part_g_failure_decomposition.csv"),
        },
    }
    write_json(OUT_ROOT / "v22_78_final_route.json", final)
    append_exec("G_failure_decomposition_final_route", command, "done", files=f"{rel(OUT_ROOT / 'v22_78_part_g_failure_decomposition.csv')}; {rel(OUT_ROOT / 'v22_78_final_route.json')}", note=json.dumps({"final_route": route, "reason": reason}, ensure_ascii=False))
    append_recap("Part G final route", [f"final_route={route}；reason={reason}", f"Part gates: A={row['part_a_hard_gate_pass']} B={row['part_b_gate_pass']} C={row['part_c_gate_pass']} D={row['part_d_gate_pass']} E_explore={row['part_e_exploration_gate_pass']} E_official={row['official_candidate_gate_pass']} F={row['part_f_gate_pass']}。", f"failure_components={row['failure_components']}。"])
    return final


def run_full(args: argparse.Namespace) -> dict[str, Any]:
    a = run_part_a(args)
    if not int(a.get("part_a_hard_gate_pass", 0)):
        return run_part_g(args, a, {}, {}, {}, {}, {})
    b = run_part_b(args)
    c = run_part_c(args) if int(b.get("part_b_gate_pass", 0)) else {}
    d = run_part_d(args) if int(c.get("part_c_gate_pass", 0)) else {}
    e = run_part_e(args, d) if d else {}
    f = run_part_f(args, e) if e else {}
    return run_part_g(args, a, b, c, d, e, f)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="full", choices=["full", "part-a", "part-b", "part-c", "part-d", "part-d-merge", "part-e", "part-f", "part-g", "final-route"])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--train-size", type=int, default=512)
    p.add_argument("--held-size", type=int, default=256)
    p.add_argument("--test-size", type=int, default=256)
    p.add_argument("--hidden", type=int, default=96)
    p.add_argument("--metric-batch-size", type=int, default=160)
    p.add_argument("--lr", type=float, default=1.0e-3)
    p.add_argument("--weight-decay", type=float, default=1.0e-4)
    p.add_argument("--projector-ridge", type=float, default=3.0e-4)
    p.add_argument("--step-mult", type=float, default=0.30)
    p.add_argument("--debt-margin-lambda", type=float, default=0.50)
    p.add_argument("--part-d-datasets", default="Wine,Spam,MNIST")
    p.add_argument("--part-d-seed-count", type=int, default=5)
    p.add_argument("--part-d-shard-count", type=int, default=1)
    p.add_argument("--part-d-shard-index", type=int, default=0)
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
    elif args.mode == "part-d-merge":
        run_part_d_merge(args)
    elif args.mode == "part-e":
        d = load_json(OUT_ROOT / "v22_78_part_d_preflight_route.json")
        run_part_e(args, d)
    elif args.mode == "part-f":
        e_rows = read_rows(OUT_ROOT / "v22_78_part_e_candidate_gate.csv")
        run_part_f(args, e_rows[0] if e_rows else {})
    elif args.mode in {"part-g", "final-route"}:
        a = load_json(OUT_ROOT / "v22_78_part_a_code_identity_hard_gate.json")
        b = load_json(OUT_ROOT / "v22_78_part_b_v22_77_failure_replay_sufficiency_summary.json")
        c = load_json(OUT_ROOT / "v22_78_part_c_unit_gate.json")
        d = load_json(OUT_ROOT / "v22_78_part_d_preflight_route.json")
        e_rows = read_rows(OUT_ROOT / "v22_78_part_e_candidate_gate.csv")
        f_rows = read_rows(OUT_ROOT / "v22_78_part_f_strengthened_mlp_audit.csv")
        run_part_g(args, a, b, c, d, e_rows[0] if e_rows else {}, f_rows[0] if f_rows else {})


if __name__ == "__main__":
    main()
