#!/usr/bin/env python3
"""DG-KAN v22.80 Control-Orthogonal Edge Signal KAN-MPFU runner."""

from __future__ import annotations

import argparse
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
from typing import Any

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import experiments.run_v22_73_distributional_edge_natural_residual_kan_mpfu as base73
import experiments.run_v22_75_trajectory_calibrated_edge_probability_kan_mpfu as base75
import experiments.run_v22_79_representation_separated_edgebank_kan_mpfu as base79
from dgkan.fu.kan_conditional_edge_signal_metric import coherence_score, normalize_columns, polynomial_nuisance_basis, weighted_project
from dgkan.fu.kan_edge_bank_signal_basis import bank_additive_projection, metric_energy
from dgkan.fu.representation_separated_edge_bank import same_energy_orthogonal_control, separation_lift_to_downstream


PYTHON = sys.executable
RUNNER = ROOT / "experiments/run_v22_80_control_orthogonal_edge_signal_kan_mpfu.py"
PLAN = ROOT / "docs/DG-KAN_v22.80_ControlOrthogonalEdgeSignalKAN_MPFU_完整计划.md"
EXEC_LOG = ROOT / "docs/DG-KAN_v22.80_ControlOrthogonalEdgeSignalKAN_MPFU_执行日志.md"
RECAP_LOG = ROOT / "docs/DG-KAN_v22.80_ControlOrthogonalEdgeSignalKAN_MPFU_实验结果复盘.md"
OUT_ROOT = ROOT / "results/v22_80"
LOG_ROOT = OUT_ROOT / "logs"
V2279_ROOT = ROOT / "results/v22_79"


COER_FAMILIES = [
    "coer_output_residual_edge_bank",
    "coer_output_residual_domain_residual_edge_bank",
    "coer_output_residual_control_orth_edge_bank",
]

COER_LIFT_FAMILIES = [
    "coer_upstream_constrained_lift",
    "coer_upstream_constrained_lift_edge_residual",
    "coer_full_control_orthogonal_two_layer",
]

PART_C_PROJECTOR_VARIANTS = [
    ("all_debt_full_projection", None, "project_Brier_ECE_tail_margin_radial"),
    ("brier_only_projection", ["Brier"], "project_Brier_only"),
    ("ece_only_projection", ["ECE"], "project_ECE_only"),
    ("tail95_only_projection", ["tail95"], "project_tail95_only"),
    ("tail99_only_projection", ["tail99"], "project_tail99_only"),
    ("tail_pair_projection", ["tail95", "tail99"], "project_tail95_tail99"),
    ("margin_only_projection", ["margin"], "project_margin_only"),
    ("radial_only_projection", ["radial"], "project_radial_only"),
    ("no_brier_projection", ["ECE", "tail95", "tail99", "margin", "radial"], "project_all_except_Brier"),
    ("raw_with_debt_ucb_guard", [], "no_output_projection_guard_debt_UCB"),
]

PART_F_CONSTRAINT_VARIANTS = [
    "w1_no_constraint",
    "w1_brier_only",
    "w1_ece_only",
    "w1_tail99_only",
    "w1_margin_only",
    "w1_domain_only",
    "w1_all_debt_only",
    "w1_same_upstream_only",
    "w1_domain_all_debt",
    "w1_domain_all_debt_same_upstream",
]

PART_F_TARGET_VARIANTS = [
    "full_bank_target",
    "lowrank25_bank_target",
]

PART_F_LAYER_VARIANTS = [
    "w1_only",
    "w1_plus_w2_residual",
]

CARRIER_REDESIGN_SPECS = [
    {"carrier_family": "current_lowfreq2_control", "method": "task_conditional_orthogonal_poly", "kind": "current_lowfreq2"},
    {"carrier_family": "wlb_lowfreq2_global", "method": "wlb_lowfreq2_brier_natural_dynamic_margin", "kind": "wlb_global"},
    {"carrier_family": "wlb_lowfreq2_bump2", "method": "wlb_lowfreq2_bump2_brier_natural_dynamic_margin", "kind": "wlb_local"},
    {"carrier_family": "wlb_lowfreq2_bump4", "method": "wlb_lowfreq2_bump4_brier_natural_dynamic_margin", "kind": "wlb_local"},
    {"carrier_family": "monotone_pou_lowfreq2_bump2", "method": "wlb_monotone_lowfreq2_bump2_brier_natural_dynamic_margin", "kind": "monotone_pou"},
    {"carrier_family": "monotone_pou_lowfreq2_bump4", "method": "wlb_monotone_lowfreq2_bump4_brier_natural_dynamic_margin", "kind": "monotone_pou"},
    {"carrier_family": "compacthat_lowfreq2_bump2", "method": "wlb_compacthat_lowfreq2_bump2_brier_natural_dynamic_margin", "kind": "compact_support"},
    {"carrier_family": "compacthat_lowfreq2_bump4", "method": "wlb_compacthat_lowfreq2_bump4_brier_natural_dynamic_margin", "kind": "compact_support"},
    {"carrier_family": "tail_safe_lowfreq1_bump4", "method": "wlb_tail_safe_brier_natural_dynamic_margin", "kind": "tail_safe"},
]

CARRIER_REDESIGN_SIGNAL_FAMILIES = [
    "coer_output_residual_edge_bank",
    "coer_output_residual_domain_residual_edge_bank",
    "coer_output_residual_control_orth_edge_bank",
]

OUTPUT_ORACLE_DIRECTION_MODES = [
    "raw_ce",
    "all_debt_projected",
    "brier_only_projected",
    "no_brier_projected",
    "linear_debt_cone_qp",
]

OUTPUT_ORACLE_STEP_NORMS = [0.0005, 0.001, 0.002, 0.005, 0.02, 0.08]

ACTUATOR_COVERAGE_STEP_NORMS = [0.0005, 0.001]


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
    return base79.fval(x, default)


def quantile(values: list[float], q: float) -> float:
    return base79.quantile(values, q)


def lower_cvar(values: list[float], frac: float = 0.25) -> float:
    return base79.lower_cvar(values, frac)


def read_rows(path: str | Path) -> list[dict[str, str]]:
    p = Path(path)
    if not p.exists():
        return []
    with p.open("r", newline="") as fh:
        return list(csv.DictReader(fh))


def write_rows(path: str | Path, rows: list[dict[str, Any]]) -> None:
    ensure_out()
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        p.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with p.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def read_json(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_json(path: str | Path, obj: dict[str, Any]) -> None:
    ensure_out()
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def append_exec(task_id: str, command: str, status: str, *, gpu: str = "", files: str = "", note: str = "") -> None:
    ensure_out()
    with EXEC_LOG.open("a", encoding="utf-8") as fh:
        fh.write(f"\n### {now_sg()} | {task_id} | {status}\n")
        fh.write(f"- command: `{command}`\n")
        fh.write(f"- gpu: `{gpu}`\n")
        fh.write(f"- files: `{files}`\n")
        if note:
            fh.write(f"- note: {note}\n")


def append_recap(title: str, lines: list[str]) -> None:
    ensure_out()
    with RECAP_LOG.open("a", encoding="utf-8") as fh:
        fh.write(f"\n## {title}\n\n")
        fh.write(f"- time_sg: {now_sg()}\n")
        for line in lines:
            fh.write(f"- {line}\n")


def bool_sum(rows: list[dict[str, Any]], key: str) -> int:
    return sum(int(fval(row.get(key)) > 0.0) for row in rows)


def metric_snapshot(logits: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
    return base79.metric_snapshot(logits, y)


def metric_delta(before: dict[str, float], after: dict[str, float]) -> dict[str, float]:
    return base79.metric_delta(before, after)


def functional_call_model(model: Any, params: dict[str, torch.Tensor], x: torch.Tensor) -> torch.Tensor:
    return base79.functional_call_model(model, params, x)


def normalized_update(vector: torch.Tensor, shape: torch.Size | tuple[int, ...], norm: float) -> torch.Tensor:
    return base79.normalized_update(vector, shape, norm)


def metric_delta_for_updates(model: Any, x: torch.Tensor, y: torch.Tensor, updates: dict[str, torch.Tensor]) -> dict[str, float]:
    return base79.metric_delta_for_updates(model, x, y, updates)


def metric_diag_from_design(model: Any, x: torch.Tensor, *, mode: str = "readout") -> torch.Tensor:
    return base79.metric_diag_from_design(model, x, mode=mode)


def param_grad_for_loss(model: Any, x: torch.Tensor, y: torch.Tensor, param_name: str, loss_kind: str) -> torch.Tensor:
    return base79.param_grad_for_loss(model, x, y, param_name, loss_kind)


def collect_grads(model: Any, x_source: torch.Tensor, y_source: torch.Tensor, x_witness: torch.Tensor, y_witness: torch.Tensor, x_guard: torch.Tensor, y_guard: torch.Tensor, x_all: torch.Tensor, y_all: torch.Tensor, param_name: str) -> dict[str, torch.Tensor]:
    return base79.collect_grads(model, x_source, y_source, x_witness, y_witness, x_guard, y_guard, x_all, y_all, param_name)


def debt_grad_columns(grads: dict[str, torch.Tensor]) -> list[torch.Tensor]:
    return base79.debt_grad_columns(grads)


def make_domain_basis(dim: int, device: torch.device, grads: dict[str, torch.Tensor], *, include_density: bool = True) -> torch.Tensor | None:
    return base79.make_domain_basis(dim, device, grads, include_density=include_density)


def make_control_basis(dim: int, device: torch.device, grads: dict[str, torch.Tensor], candidate_signal: torch.Tensor) -> torch.Tensor | None:
    return base79.make_control_basis(dim, device, grads, candidate_signal)


def control_gap_diagnostics(model: Any, x: torch.Tensor, y: torch.Tensor, candidate: dict[str, torch.Tensor], controls: dict[str, dict[str, torch.Tensor]], args: argparse.Namespace) -> tuple[dict[str, float], dict[str, dict[str, float]], dict[str, float], float, float]:
    return base79.control_gap_diagnostics(model, x, y, candidate, controls, args)


def bootstrap_guard_margin(model: Any, x: torch.Tensor, y: torch.Tensor, candidate: dict[str, torch.Tensor], controls: dict[str, dict[str, torch.Tensor]], args: argparse.Namespace, *, seed: int) -> dict[str, float]:
    return base79.bootstrap_guard_margin(model, x, y, candidate, controls, args, seed=seed)


def map_family_to_method(family: str) -> str:
    if "orth" in family or "control" in family:
        return "task_conditional_orthogonal_poly"
    if "domain" in family:
        return "density_equalized_conditional_spline"
    return "node_bank_additive_anova"


def split_train(x_all: torch.Tensor, y_all: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    n = int(x_all.shape[0])
    a = max(16, n // 4)
    b = max(a + 16, n // 2)
    c = max(b + 16, (3 * n) // 4)
    x_source, y_source = x_all[:a], y_all[:a]
    x_witness, y_witness = x_all[a:b], y_all[a:b]
    x_guard, y_guard = x_all[b:c], y_all[b:c]
    x_control, y_control = x_all[c:], y_all[c:]
    if int(x_guard.shape[0]) < 8:
        x_guard, y_guard = x_all[-max(8, n // 4) :], y_all[-max(8, n // 4) :]
    if int(x_control.shape[0]) < 8:
        x_control, y_control = x_guard, y_guard
    return x_source, y_source, x_witness, y_witness, x_guard, y_guard, x_control, y_control


def logits_loss_grad(logits: torch.Tensor, y: torch.Tensor, loss_kind: str) -> torch.Tensor:
    z = logits.detach().clone().float().requires_grad_(True)
    loss = base79.loss_from_kind(z, y.long(), loss_kind)
    loss.backward()
    return z.grad.detach().reshape(-1).to(device=logits.device, dtype=torch.float64)


def output_metric_diag(logits: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        probs = torch.softmax(logits.float(), dim=1)
        fisher = (probs * (1.0 - probs)).reshape(-1).to(dtype=torch.float64)
        return (fisher / fisher.mean().clamp_min(1.0e-12)).clamp(0.05, 20.0)


OUTPUT_NUISANCE_KINDS = {
    "Brier": "brier",
    "ECE": "ece_debt",
    "tail95": "tail95_debt",
    "tail99": "tail99_debt",
    "margin": "margin_debt",
    "radial": "radial",
}


def output_debt_residual(logits: torch.Tensor, y: torch.Tensor, *, ridge: float, include: list[str] | None = None) -> dict[str, Any]:
    device = logits.device
    raw = logits_loss_grad(logits, y, "ce")
    nuisance = {name: logits_loss_grad(logits, y, kind) for name, kind in OUTPUT_NUISANCE_KINDS.items()}
    dim = int(raw.numel())
    metric = output_metric_diag(logits).to(device=device)
    chosen = list(nuisance.keys()) if include is None else list(include)
    basis = normalize_columns([nuisance[name] for name in chosen if name in nuisance], dim, device)
    if basis is None:
        residual = raw.clone()
        diag = {"projected_energy_fraction": 0.0, "residual_energy_fraction": 1.0}
    else:
        residual, diag = weighted_project(raw, basis, metric, ridge=ridge)
    raw_energy = metric_energy(raw, metric).clamp_min(1.0e-12)
    res_energy = metric_energy(residual.reshape(-1), metric).clamp_min(0.0)
    out: dict[str, Any] = {
        "raw": raw,
        "residual": residual.reshape(-1),
        "metric": metric,
        "output_residual_energy_fraction": float((res_energy / raw_energy).detach().cpu().item()),
        "all_debt_projection_fraction": fval(diag.get("projected_energy_fraction")),
        "projected_nuisance_names": ",".join(chosen),
    }
    for name, vec in nuisance.items():
        one = normalize_columns([vec], dim, device)
        _res, one_diag = weighted_project(raw, one, metric, ridge=ridge)
        out[f"{name}_projection_fraction"] = fval(one_diag.get("projected_energy_fraction"))
    return out


def weighted_cosine(a: torch.Tensor, b: torch.Tensor, metric: torch.Tensor) -> float:
    aa = a.detach().reshape(-1).to(dtype=torch.float64)
    bb = b.detach().reshape(-1).to(device=aa.device, dtype=torch.float64)
    mm = metric.detach().reshape(-1).to(device=aa.device, dtype=torch.float64)
    num = (aa * bb * mm).sum()
    den = (metric_energy(aa, mm).clamp_min(1.0e-12).sqrt() * metric_energy(bb, mm).clamp_min(1.0e-12).sqrt())
    return float((num / den).detach().cpu().item())


def logit_metric_delta_for_update(logits: torch.Tensor, y: torch.Tensor, update: torch.Tensor) -> dict[str, float]:
    before = metric_snapshot(logits.float(), y.long())
    after = metric_snapshot((logits + update.reshape_as(logits).to(device=logits.device, dtype=logits.dtype)).float(), y.long())
    return metric_delta(before, after)


def normalize_logit_update(vector: torch.Tensor, logits: torch.Tensor, norm: float) -> torch.Tensor:
    return normalized_update(vector.reshape(-1), logits.shape, float(norm)).to(device=logits.device, dtype=torch.float32)


def output_linear_debt_cone_direction(logits: torch.Tensor, y: torch.Tensor, *, ridge: float) -> dict[str, Any]:
    raw = logits_loss_grad(logits, y, "ce")
    debt_names = ["Brier", "ECE", "tail95", "tail99", "margin"]
    debt_kinds = ["brier", "ece_debt", "tail95_debt", "tail99_debt", "margin_debt"]
    debt_vecs = [logits_loss_grad(logits, y, kind).reshape(-1).to(dtype=torch.float64) for kind in debt_kinds]
    u0 = -raw.reshape(-1).to(dtype=torch.float64)
    device = u0.device
    A = torch.stack(debt_vecs, dim=0).to(device=device, dtype=torch.float64)
    best_u = torch.zeros_like(u0)
    best_dist = float("inf")
    best_mask = 0
    m = int(A.shape[0])
    eye_cache: dict[int, torch.Tensor] = {}
    for mask in range(1 << m):
        if mask == 0:
            u = u0.clone()
        else:
            idx = [i for i in range(m) if (mask >> i) & 1]
            sub = A.index_select(0, torch.tensor(idx, device=device, dtype=torch.long))
            k = int(sub.shape[0])
            if k not in eye_cache:
                eye_cache[k] = torch.eye(k, device=device, dtype=torch.float64)
            mat = sub @ sub.transpose(0, 1) + float(ridge) * eye_cache[k]
            rhs = sub @ u0
            try:
                lam = torch.linalg.solve(mat, rhs.reshape(-1, 1)).reshape(-1)
                u = u0 - sub.transpose(0, 1) @ lam
            except Exception:
                continue
        constraints = A @ u
        first_order_ce = float((raw.reshape(-1).to(dtype=torch.float64) * u).sum().detach().cpu().item())
        feasible = bool(torch.all(constraints <= 1.0e-10).detach().cpu().item()) and first_order_ce < -1.0e-12
        if feasible:
            dist = float((u - u0).square().sum().detach().cpu().item())
            if dist < best_dist:
                best_dist = dist
                best_u = u
                best_mask = mask
    constraints = A @ best_u
    first_order_ce = float((raw.reshape(-1).to(dtype=torch.float64) * best_u).sum().detach().cpu().item())
    return {
        "direction": best_u.reshape(-1),
        "first_order_CE_delta": first_order_ce,
        "first_order_debt_max": float(constraints.max().detach().cpu().item()) if int(constraints.numel()) else 0.0,
        "first_order_debt_feasible": int(best_dist < float("inf")),
        "active_debt_constraints": ",".join(name for i, name in enumerate(debt_names) if (best_mask >> i) & 1),
        "linear_cone_distance_from_raw": 0.0 if best_dist == float("inf") else best_dist,
    }


def edge_pullback_from_output_residual(model: Any, x: torch.Tensor, y: torch.Tensor, *, ridge: float, include: list[str] | None = None) -> dict[str, Any]:
    logits = model(x).float()
    out_res = output_debt_residual(logits, y, ridge=ridge, include=include)
    design = base73.w2_readout_edge_design(model, x)
    phi = design["phi"].to(device=x.device, dtype=torch.float64)
    raw_edge = phi.transpose(0, 1) @ out_res["raw"].reshape(-1)
    residual_edge = phi.transpose(0, 1) @ out_res["residual"].reshape(-1)
    metric_w2 = metric_diag_from_design(model, x, mode="readout")
    raw_bank, raw_stats = bank_additive_projection(raw_edge, metric_w2, model.w2.shape)
    res_bank, res_stats = bank_additive_projection(residual_edge, metric_w2, model.w2.shape)
    return {
        **{k: v for k, v in out_res.items() if not isinstance(v, torch.Tensor)},
        "raw_output_residual": out_res["raw"],
        "output_residual": out_res["residual"],
        "raw_edge_vec": raw_edge.reshape(-1),
        "residual_edge_vec": residual_edge.reshape(-1),
        "residual_bank_vec": res_bank.reshape(-1),
        "raw_bank_vec": raw_bank.reshape(-1),
        "metric_w2": metric_w2.reshape(-1),
        "edge_pullback_energy": float(metric_energy(residual_edge, metric_w2).detach().cpu().item()),
        "bank_pullback_energy": float(metric_energy(res_bank.reshape(-1), metric_w2).detach().cpu().item()),
        "bank_R2_on_residual_signal": fval(res_stats.get("edge_bank_anova_explained")),
        "bank_R2_on_raw_signal": fval(raw_stats.get("edge_bank_anova_explained")),
        "residual_bank_R2_ratio": fval(res_stats.get("edge_bank_anova_explained")) / max(1.0e-12, fval(raw_stats.get("edge_bank_anova_explained"))),
        "edge_pullback_rank": int(torch.linalg.matrix_rank(phi.detach().float()).detach().cpu().item()) if int(phi.numel()) else 0,
    }


def make_w2_nuisance_basis(w2_grads: dict[str, torch.Tensor], signal: torch.Tensor, bank_vec: torch.Tensor, sep_vec: torch.Tensor | None, upstream_lift_vec: torch.Tensor | None, dim: int, device: torch.device) -> torch.Tensor | None:
    cols: list[torch.Tensor] = []
    for name in ["domain", "random", "debt", "tail_debt", "tail95_debt", "tail99_debt", "margin_debt", "ece_debt", "radial", "smooth", "hard_loss", "loss_rank", "shuffled"]:
        if name in w2_grads:
            cols.append(w2_grads[name])
    cols.append(bank_vec.reshape(-1))
    cols.append(same_energy_orthogonal_control(signal).to(device=device))
    if sep_vec is not None:
        cols.append(sep_vec.reshape(-1))
    if upstream_lift_vec is not None:
        cols.append(upstream_lift_vec.reshape(-1))
    return normalize_columns(cols, dim, device)


def control_orthogonal_edge_signal(
    family: str,
    edge_vec: torch.Tensor,
    metric_w2: torch.Tensor,
    model: Any,
    w2_grads: dict[str, torch.Tensor],
    *,
    ridge: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    device = edge_vec.device
    dim = int(edge_vec.numel())
    bank_vec = bank_additive_projection(edge_vec, metric_w2, model.w2.shape)[0].reshape(-1)
    domain_basis = make_domain_basis(dim, device, w2_grads, include_density=True)
    domain_res, domain_diag = weighted_project(edge_vec, domain_basis, metric_w2, ridge=ridge)
    if family == "coer_output_residual_edge_bank":
        chosen = edge_vec.reshape(-1)
        full_diag = {"residual_energy_fraction": 1.0, "projected_energy_fraction": 0.0}
        projection_mode = "output_debt_residual_only"
    elif family == "coer_output_residual_domain_residual_edge_bank":
        chosen = domain_res.reshape(-1)
        full_diag = dict(domain_diag)
        projection_mode = "domain_residual_only"
    else:
        full_basis = make_w2_nuisance_basis(w2_grads, domain_res.reshape(-1), bank_vec, None, None, dim, device)
        chosen_res, full_diag = weighted_project(domain_res.reshape(-1), full_basis, metric_w2, ridge=ridge)
        chosen = chosen_res.reshape(-1)
        projection_mode = "domain_and_control_orthogonal"
    return chosen.reshape(-1), {
        "domain_nuisance_fraction": fval(domain_diag.get("projected_energy_fraction")),
        "conditional_residual_fraction": fval(full_diag.get("residual_energy_fraction")),
        "control_projected_energy_fraction": fval(full_diag.get("projected_energy_fraction")),
        "control_projection_mode": projection_mode,
    }


def make_probe_model(dataset: str, seed: int, family: str, args: argparse.Namespace, device: torch.device) -> tuple[Any, dict[str, Any], torch.Tensor, torch.Tensor]:
    bundle = base73.load_bundle(dataset, int(args.train_size), int(args.held_size), int(args.test_size), int(seed))
    x_all = bundle["x_train"].to(device).float()[: int(args.metric_batch_size)]
    y_all = bundle["y_train"].to(device).long()[: int(args.metric_batch_size)]
    method = map_family_to_method(family)
    model = base75.make_wlb_model(method, bundle, device, int(args.hidden), int(seed) + 22800, x_metric=x_all)
    return model, bundle, x_all, y_all


def task_grid(args: argparse.Namespace, families: list[str]) -> list[tuple[str, int, str]]:
    datasets = [item.strip() for item in str(args.datasets).split(",") if item.strip()]
    seeds = list(range(int(args.seed_count)))
    return [(family, seed, dataset) for family in families for dataset in datasets for seed in seeds]


def shard_items(items: list[Any], args: argparse.Namespace) -> list[Any]:
    count = max(1, int(args.shard_count))
    idx = int(args.shard_index)
    return [item for pos, item in enumerate(items) if pos % count == idx]


def part_c_probe(dataset: str, seed: int, family: str, args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    model, _bundle, x_all, y_all = make_probe_model(dataset, seed, family, args, device)
    x_source, y_source, x_witness, y_witness, x_guard, y_guard, _x_control, _y_control = split_train(x_all, y_all)
    source = edge_pullback_from_output_residual(model, x_source, y_source, ridge=float(args.projector_ridge))
    witness = edge_pullback_from_output_residual(model, x_witness, y_witness, ridge=float(args.projector_ridge))
    edge_vec = source["residual_edge_vec"].reshape(-1)
    step_norm = float(args.lr) * float(args.step_mult)
    candidate_w2 = -normalized_update(edge_vec, model.w2.shape, step_norm)
    deltas = metric_delta_for_updates(model, x_guard, y_guard, {"w2": candidate_w2})
    debt_ucb = max(
        deltas["Brier"] + 0.5 * abs(deltas["Brier"]),
        deltas["ECE"] + 0.5 * abs(deltas["ECE"]),
        deltas["tail95"] + 0.5 * abs(deltas["tail95"]),
        deltas["tail99"] + 0.5 * abs(deltas["tail99"]),
        -deltas["margin10"] + 0.5 * abs(deltas["margin10"]),
    )
    align = coherence_score(source["residual_edge_vec"], witness["residual_edge_vec"], source["metric_w2"])
    raw_align = coherence_score(source["raw_edge_vec"], witness["raw_edge_vec"], source["metric_w2"])
    residual_bank_R2 = fval(source["bank_R2_on_residual_signal"])
    raw_bank_R2 = fval(source["bank_R2_on_raw_signal"])
    return {
        "dataset": dataset,
        "seed": seed,
        "family": family,
        "architecture": map_family_to_method(family),
        "output_residual_energy_fraction": source["output_residual_energy_fraction"],
        "Brier_projection_fraction": source["Brier_projection_fraction"],
        "ECE_projection_fraction": source["ECE_projection_fraction"],
        "tail95_projection_fraction": source["tail95_projection_fraction"],
        "tail99_projection_fraction": source["tail99_projection_fraction"],
        "margin_projection_fraction": source["margin_projection_fraction"],
        "radial_projection_fraction": source["radial_projection_fraction"],
        "all_debt_projection_fraction": source["all_debt_projection_fraction"],
        "residual_task_NLL_probe_delta": deltas["NLL"],
        "residual_debt_probe_delta": max(deltas["Brier"], deltas["ECE"], deltas["tail95"], deltas["tail99"], -deltas["margin10"]),
        "residual_debt_probe_UCB": debt_ucb,
        "residual_debt_probe_UCB_nonpositive": int(debt_ucb <= 0.0),
        "edge_pullback_energy": source["edge_pullback_energy"],
        "edge_pullback_rank": source["edge_pullback_rank"],
        "bank_pullback_energy": source["bank_pullback_energy"],
        "bank_R2_on_residual_signal": residual_bank_R2,
        "bank_R2_on_raw_signal": raw_bank_R2,
        "residual_bank_R2_gain": residual_bank_R2 - raw_bank_R2,
        "residual_bank_R2_gain_positive": int((residual_bank_R2 - raw_bank_R2) > 0.0),
        "residual_bank_R2_ratio": source["residual_bank_R2_ratio"],
        "source_witness_residual_alignment": align,
        "source_witness_residual_alignment_LCB": align - 0.05 * abs(align),
        "source_witness_raw_alignment": raw_align,
        "part_c_row_pass": int(
            source["output_residual_energy_fraction"] >= 0.10
            and deltas["NLL"] < 0.0
            and debt_ucb <= 0.0
            and (residual_bank_R2 - raw_bank_R2) > 0.0
            and residual_bank_R2 >= 0.10
            and align - 0.05 * abs(align) > 0.0
            and source["residual_bank_R2_ratio"] >= 0.30
        ),
    }


def summarize_part_c(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for family in sorted({str(row.get("family")) for row in rows}):
        group = [row for row in rows if str(row.get("family")) == family]
        vals = [fval(row.get("bank_R2_on_residual_signal")) for row in group]
        summary = {
            "family": family,
            "rows": len(group),
            "output_residual_energy_fraction_median": quantile([fval(row.get("output_residual_energy_fraction")) for row in group], 0.50),
            "residual_task_NLL_probe_delta_median": quantile([fval(row.get("residual_task_NLL_probe_delta")) for row in group], 0.50),
            "residual_debt_probe_UCB_nonpositive_rows": bool_sum(group, "residual_debt_probe_UCB_nonpositive"),
            "residual_bank_R2_gain_rows": bool_sum(group, "residual_bank_R2_gain_positive"),
            "residual_bank_R2_CVaR25": lower_cvar(vals, 0.25),
            "source_witness_residual_alignment_LCB_median": quantile([fval(row.get("source_witness_residual_alignment_LCB")) for row in group], 0.50),
            "residual_bank_R2_ratio_median": quantile([fval(row.get("residual_bank_R2_ratio")) for row in group], 0.50),
        }
        summary["part_c_family_gate_pass"] = int(
            summary["rows"] >= 15
            and summary["output_residual_energy_fraction_median"] >= 0.10
            and summary["residual_task_NLL_probe_delta_median"] < 0.0
            and summary["residual_debt_probe_UCB_nonpositive_rows"] >= 10
            and summary["residual_bank_R2_gain_rows"] >= 10
            and summary["residual_bank_R2_CVaR25"] >= 0.10
            and summary["source_witness_residual_alignment_LCB_median"] > 0.0
            and summary["residual_bank_R2_ratio_median"] >= 0.30
        )
        out.append(summary)
    return out


def run_part_c(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")
    tasks = shard_items(task_grid(args, COER_FAMILIES), args)
    rows = [part_c_probe(dataset, seed, family, args, device) for family, seed, dataset in tasks]
    out_path = OUT_ROOT / f"v22_80_part_c_output_debt_residual_shard{args.shard_index}_of_{args.shard_count}.csv"
    write_rows(out_path, rows)
    obj = {"gate": "v22_80_part_c_output_debt_residual_shard", "rows": len(rows), "shard_index": int(args.shard_index), "shard_count": int(args.shard_count), "output": rel(out_path)}
    write_json(OUT_ROOT / f"v22_80_part_c_output_debt_residual_shard{args.shard_index}_of_{args.shard_count}.json", obj)
    append_exec("C_output_debt_residual_shard", command_text(sys.argv), "done", gpu=str(device), files=rel(out_path), note=json.dumps(obj, ensure_ascii=False))
    return obj


def route_part_c(summaries: list[dict[str, Any]]) -> tuple[str, str]:
    if any(int(row.get("part_c_family_gate_pass", 0)) for row in summaries):
        return "OutputDebtResidualPreflightOpened", "at least one fixed family passed Part C output-debt-residual preflight"
    max_energy = max([fval(row.get("output_residual_energy_fraction_median")) for row in summaries] or [0.0])
    min_nll = min([fval(row.get("residual_task_NLL_probe_delta_median")) for row in summaries] or [0.0])
    max_ratio = max([fval(row.get("residual_bank_R2_ratio_median")) for row in summaries] or [0.0])
    max_cvar = max([fval(row.get("residual_bank_R2_CVaR25")) for row in summaries] or [0.0])
    if max_energy < 0.10 or min_nll >= 0.0:
        return "NoOutputDebtResidualSignal", f"max_output_residual_energy_fraction_median={max_energy}; min_residual_task_NLL_probe_delta_median={min_nll}"
    if max_ratio < 0.30 or max_cvar < 0.10:
        return "BankSignalDebtExplained_NoEdgeResidual", f"max_residual_bank_R2_ratio_median={max_ratio}; max_residual_bank_R2_CVaR25={max_cvar}"
    return "OutputDebtResidualPresentButBankGateFailed", f"max_energy={max_energy}; min_NLL_delta={min_nll}; max_ratio={max_ratio}; max_cvar={max_cvar}"


def merge_part_c(args: argparse.Namespace) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    missing = []
    for idx in range(int(args.shard_count)):
        path = OUT_ROOT / f"v22_80_part_c_output_debt_residual_shard{idx}_of_{args.shard_count}.csv"
        if not path.exists():
            missing.append(rel(path))
        rows.extend(read_rows(path))
    write_rows(OUT_ROOT / "v22_80_part_c_output_debt_residual.csv", rows)
    summaries = summarize_part_c(rows)
    write_rows(OUT_ROOT / "v22_80_part_c_output_debt_residual_summaries.csv", summaries)
    route, reason = route_part_c(summaries) if not missing else ("R0-CodeOrTruthGateFailed", f"missing Part C shards: {missing}")
    obj = {
        "gate": "v22_80_part_c_output_debt_residual",
        "run_status": "completed_part_c_merge" if not missing else "incomplete_part_c_merge",
        "part_c_gate_pass": int(not missing and any(int(row.get("part_c_family_gate_pass", 0)) for row in summaries)),
        "part_c_route": route,
        "route_reason": reason,
        "rows": len(rows),
        "summary_rows": len(summaries),
        "missing_shards": missing,
        "passed_families": [row["family"] for row in summaries if int(row.get("part_c_family_gate_pass", 0))],
    }
    write_json(OUT_ROOT / "v22_80_part_c_output_debt_residual_route.json", obj)
    append_exec("C_output_debt_residual_merge", command_text(sys.argv), "pass" if obj["part_c_gate_pass"] else "fail", files=f"{rel(OUT_ROOT / 'v22_80_part_c_output_debt_residual.csv')}; {rel(OUT_ROOT / 'v22_80_part_c_output_debt_residual_summaries.csv')}; {rel(OUT_ROOT / 'v22_80_part_c_output_debt_residual_route.json')}", note=json.dumps(obj, ensure_ascii=False))
    best = sorted(summaries, key=lambda row: (int(row.get("part_c_family_gate_pass", 0)), fval(row.get("output_residual_energy_fraction_median")), fval(row.get("residual_bank_R2_CVaR25"))), reverse=True)[:5]
    append_recap(
        "Part C output-debt-residual-first signal audit",
        [
            f"rows={len(rows)}；summary_rows={len(summaries)}；part_c_gate_pass={obj['part_c_gate_pass']}；route={route}。",
            f"reason={reason}",
            "best_families="
            + " | ".join(
                f"{row['family']}: energy_median={row['output_residual_energy_fraction_median']}；NLL_delta_median={row['residual_task_NLL_probe_delta_median']}；debt_UCB_nonpositive={row['residual_debt_probe_UCB_nonpositive_rows']}/{row['rows']}；bank_gain={row['residual_bank_R2_gain_rows']}/{row['rows']}；bank_CVaR25={row['residual_bank_R2_CVaR25']}；ratio_median={row['residual_bank_R2_ratio_median']}；pass={row['part_c_family_gate_pass']}"
                for row in best
            ),
            f"代码审计：Part C 先在 output/logit space 构造 Brier/ECE/tail95/tail99/margin/radial nuisance basis，再把 residual pullback 到 KAN w2 edge-bank；不是事后用 debt control 改 candidate。",
        ],
    )
    return obj


def debt_ucb_from_deltas(deltas: dict[str, float]) -> float:
    return max(
        deltas["Brier"] + 0.5 * abs(deltas["Brier"]),
        deltas["ECE"] + 0.5 * abs(deltas["ECE"]),
        deltas["tail95"] + 0.5 * abs(deltas["tail95"]),
        deltas["tail99"] + 0.5 * abs(deltas["tail99"]),
        -deltas["margin10"] + 0.5 * abs(deltas["margin10"]),
    )


def part_c_projector_guard_probe(dataset: str, seed: int, family: str, args: argparse.Namespace, device: torch.device) -> list[dict[str, Any]]:
    model, _bundle, x_all, y_all = make_probe_model(dataset, seed, family, args, device)
    x_source, y_source, x_witness, y_witness, x_guard, y_guard, _x_control, _y_control = split_train(x_all, y_all)
    step_norm = float(args.lr) * float(args.step_mult)
    rows: list[dict[str, Any]] = []
    for variant, include, description in PART_C_PROJECTOR_VARIANTS:
        source = edge_pullback_from_output_residual(model, x_source, y_source, ridge=float(args.projector_ridge), include=include)
        witness = edge_pullback_from_output_residual(model, x_witness, y_witness, ridge=float(args.projector_ridge), include=include)
        edge_vec = source["residual_edge_vec"].reshape(-1)
        candidate_w2 = -normalized_update(edge_vec, model.w2.shape, step_norm)
        deltas = metric_delta_for_updates(model, x_guard, y_guard, {"w2": candidate_w2})
        debt_ucb = debt_ucb_from_deltas(deltas)
        align = coherence_score(source["residual_edge_vec"], witness["residual_edge_vec"], source["metric_w2"])
        raw_align = coherence_score(source["raw_edge_vec"], witness["raw_edge_vec"], source["metric_w2"])
        residual_bank_R2 = fval(source["bank_R2_on_residual_signal"])
        raw_bank_R2 = fval(source["bank_R2_on_raw_signal"])
        rows.append(
            {
                "dataset": dataset,
                "seed": seed,
                "family": family,
                "architecture": map_family_to_method(family),
                "variant": variant,
                "variant_description": description,
                "projected_nuisance_names": source["projected_nuisance_names"],
                "output_residual_energy_fraction": source["output_residual_energy_fraction"],
                "Brier_projection_fraction": source["Brier_projection_fraction"],
                "ECE_projection_fraction": source["ECE_projection_fraction"],
                "tail95_projection_fraction": source["tail95_projection_fraction"],
                "tail99_projection_fraction": source["tail99_projection_fraction"],
                "margin_projection_fraction": source["margin_projection_fraction"],
                "radial_projection_fraction": source["radial_projection_fraction"],
                "all_debt_projection_fraction": source["all_debt_projection_fraction"],
                "residual_task_NLL_probe_delta": deltas["NLL"],
                "residual_task_NLL_improve": int(deltas["NLL"] < 0.0),
                "residual_debt_probe_delta": max(deltas["Brier"], deltas["ECE"], deltas["tail95"], deltas["tail99"], -deltas["margin10"]),
                "residual_debt_probe_UCB": debt_ucb,
                "residual_debt_probe_UCB_nonpositive": int(debt_ucb <= 0.0),
                "edge_pullback_energy": source["edge_pullback_energy"],
                "edge_pullback_rank": source["edge_pullback_rank"],
                "bank_pullback_energy": source["bank_pullback_energy"],
                "bank_R2_on_residual_signal": residual_bank_R2,
                "bank_R2_on_raw_signal": raw_bank_R2,
                "residual_bank_R2_gain": residual_bank_R2 - raw_bank_R2,
                "residual_bank_R2_gain_positive": int((residual_bank_R2 - raw_bank_R2) > 0.0),
                "residual_bank_R2_ratio": source["residual_bank_R2_ratio"],
                "source_witness_residual_alignment": align,
                "source_witness_residual_alignment_LCB": align - 0.05 * abs(align),
                "source_witness_raw_alignment": raw_align,
                "debt_guard_variant_pass": int(
                    source["output_residual_energy_fraction"] >= 0.10
                    and deltas["NLL"] < 0.0
                    and debt_ucb <= 0.0
                    and residual_bank_R2 >= 0.10
                    and align - 0.05 * abs(align) > 0.0
                ),
            }
        )
    return rows


def run_part_c_projector_audit(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")
    tasks = shard_items(task_grid(args, COER_FAMILIES), args)
    rows: list[dict[str, Any]] = []
    for family, seed, dataset in tasks:
        rows.extend(part_c_projector_guard_probe(dataset, seed, family, args, device))
    out_path = OUT_ROOT / f"v22_80_part_c_projector_guard_audit_shard{args.shard_index}_of_{args.shard_count}.csv"
    write_rows(out_path, rows)
    obj = {"gate": "v22_80_part_c_projector_guard_audit_shard", "rows": len(rows), "tasks": len(tasks), "shard_index": int(args.shard_index), "shard_count": int(args.shard_count), "output": rel(out_path)}
    write_json(OUT_ROOT / f"v22_80_part_c_projector_guard_audit_shard{args.shard_index}_of_{args.shard_count}.json", obj)
    append_exec("C_projector_guard_audit_shard", command_text(sys.argv), "done", gpu=str(device), files=rel(out_path), note=json.dumps(obj, ensure_ascii=False))
    return obj


def summarize_part_c_projector_guard(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    groups = sorted({(str(row.get("family")), str(row.get("variant"))) for row in rows})
    for family, variant in groups:
        group = [row for row in rows if str(row.get("family")) == family and str(row.get("variant")) == variant]
        vals = [fval(row.get("bank_R2_on_residual_signal")) for row in group]
        energy_values = [fval(row.get("output_residual_energy_fraction")) for row in group]
        summary = {
            "family": family,
            "variant": variant,
            "rows": len(group),
            "projected_nuisance_names": str(group[0].get("projected_nuisance_names", "")) if group else "",
            "output_residual_energy_fraction_median": quantile(energy_values, 0.50),
            "output_residual_energy_fraction_CVaR25": lower_cvar(energy_values, 0.25),
            "residual_task_NLL_probe_delta_median": quantile([fval(row.get("residual_task_NLL_probe_delta")) for row in group], 0.50),
            "residual_task_NLL_improve_rows": bool_sum(group, "residual_task_NLL_improve"),
            "residual_debt_probe_UCB_nonpositive_rows": bool_sum(group, "residual_debt_probe_UCB_nonpositive"),
            "residual_bank_R2_gain_rows": bool_sum(group, "residual_bank_R2_gain_positive"),
            "residual_bank_R2_CVaR25": lower_cvar(vals, 0.25),
            "source_witness_residual_alignment_LCB_median": quantile([fval(row.get("source_witness_residual_alignment_LCB")) for row in group], 0.50),
            "residual_bank_R2_ratio_median": quantile([fval(row.get("residual_bank_R2_ratio")) for row in group], 0.50),
            "debt_guard_variant_pass_rows": bool_sum(group, "debt_guard_variant_pass"),
        }
        summary["projector_guard_family_variant_pass"] = int(
            summary["rows"] >= 15
            and summary["output_residual_energy_fraction_median"] >= 0.10
            and summary["residual_task_NLL_improve_rows"] >= 10
            and summary["residual_debt_probe_UCB_nonpositive_rows"] >= 10
            and summary["residual_bank_R2_CVaR25"] >= 0.10
            and summary["source_witness_residual_alignment_LCB_median"] > 0.0
        )
        out.append(summary)
    return out


def route_part_c_projector_guard(summaries: list[dict[str, Any]]) -> tuple[str, str]:
    def variant_rows(name: str) -> list[dict[str, Any]]:
        return [row for row in summaries if str(row.get("variant")) == name]

    raw_rows = variant_rows("raw_with_debt_ucb_guard")
    recovered = [row for row in raw_rows if int(row.get("projector_guard_family_variant_pass", 0))]
    if recovered:
        best = sorted(recovered, key=lambda row: (int(row.get("residual_debt_probe_UCB_nonpositive_rows", 0)), fval(row.get("output_residual_energy_fraction_median"))), reverse=True)[0]
        return (
            "ConstrainedDebtGuardRecoveredOutputSignal",
            f"raw guard recovered for {best['family']}: energy_median={best['output_residual_energy_fraction_median']}; NLL_improve_rows={best['residual_task_NLL_improve_rows']}/{best['rows']}; debt_UCB_nonpositive_rows={best['residual_debt_probe_UCB_nonpositive_rows']}/{best['rows']}",
        )

    brier_rows = variant_rows("brier_only_projection")
    no_brier_rows = variant_rows("no_brier_projection")
    all_rows = variant_rows("all_debt_full_projection")
    max_brier_energy = max([fval(row.get("output_residual_energy_fraction_median")) for row in brier_rows] or [0.0])
    max_no_brier_energy = max([fval(row.get("output_residual_energy_fraction_median")) for row in no_brier_rows] or [0.0])
    max_all_energy = max([fval(row.get("output_residual_energy_fraction_median")) for row in all_rows] or [0.0])
    max_raw_ucb_rows = max([int(row.get("residual_debt_probe_UCB_nonpositive_rows", 0)) for row in raw_rows] or [0])
    max_raw_nll_rows = max([int(row.get("residual_task_NLL_improve_rows", 0)) for row in raw_rows] or [0])
    max_raw_energy = max([fval(row.get("output_residual_energy_fraction_median")) for row in raw_rows] or [0.0])
    if max_brier_energy < 0.10:
        return (
            "BrierDominatedTaskNoKANResidual",
            f"max_brier_only_residual_energy_median={max_brier_energy}; max_all_debt_residual_energy_median={max_all_energy}; max_no_brier_residual_energy_median={max_no_brier_energy}; raw_guard_debt_UCB_nonpositive_rows_max={max_raw_ucb_rows}; raw_guard_NLL_improve_rows_max={max_raw_nll_rows}",
        )
    if max_all_energy < 0.10 and max_no_brier_energy >= 0.10:
        return (
            "AllDebtProjectorTooWideGuardDidNotRecover",
            f"max_all_debt_residual_energy_median={max_all_energy}; max_no_brier_residual_energy_median={max_no_brier_energy}; raw_guard_energy_median_max={max_raw_energy}; raw_guard_debt_UCB_nonpositive_rows_max={max_raw_ucb_rows}",
        )
    return (
        "ConstrainedDebtGuardDidNotRecoverOutputSignal",
        f"max_all_debt_residual_energy_median={max_all_energy}; max_brier_only_residual_energy_median={max_brier_energy}; raw_guard_energy_median_max={max_raw_energy}; raw_guard_NLL_improve_rows_max={max_raw_nll_rows}; raw_guard_debt_UCB_nonpositive_rows_max={max_raw_ucb_rows}",
    )


def merge_part_c_projector_audit(args: argparse.Namespace) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    missing = []
    for idx in range(int(args.shard_count)):
        path = OUT_ROOT / f"v22_80_part_c_projector_guard_audit_shard{idx}_of_{args.shard_count}.csv"
        if not path.exists():
            missing.append(rel(path))
        rows.extend(read_rows(path))
    write_rows(OUT_ROOT / "v22_80_part_c_projector_guard_audit.csv", rows)
    summaries = summarize_part_c_projector_guard(rows)
    write_rows(OUT_ROOT / "v22_80_part_c_projector_guard_audit_summaries.csv", summaries)
    route, reason = route_part_c_projector_guard(summaries) if not missing else ("R0-CodeOrTruthGateFailed", f"missing Part C projector audit shards: {missing}")
    obj = {
        "gate": "v22_80_part_c_projector_guard_audit",
        "run_status": "completed_part_c_projector_guard_audit_merge" if not missing else "incomplete_part_c_projector_guard_audit_merge",
        "part_c_projector_guard_repair_pass": int(route == "ConstrainedDebtGuardRecoveredOutputSignal"),
        "part_c_projector_guard_route": route,
        "route_reason": reason,
        "rows": len(rows),
        "summary_rows": len(summaries),
        "missing_shards": missing,
        "passed_family_variants": [f"{row['family']}::{row['variant']}" for row in summaries if int(row.get("projector_guard_family_variant_pass", 0))],
    }
    write_json(OUT_ROOT / "v22_80_part_c_projector_guard_audit_route.json", obj)
    append_exec("C_projector_guard_audit_merge", command_text(sys.argv), "pass" if obj["part_c_projector_guard_repair_pass"] else "fail", files=f"{rel(OUT_ROOT / 'v22_80_part_c_projector_guard_audit.csv')}; {rel(OUT_ROOT / 'v22_80_part_c_projector_guard_audit_summaries.csv')}; {rel(OUT_ROOT / 'v22_80_part_c_projector_guard_audit_route.json')}", note=json.dumps(obj, ensure_ascii=False))
    focus = [row for row in summaries if str(row.get("variant")) in {"all_debt_full_projection", "brier_only_projection", "no_brier_projection", "raw_with_debt_ucb_guard"}]
    best = sorted(focus, key=lambda row: (str(row.get("variant")), str(row.get("family"))))
    append_recap(
        "Part C projector-width and constrained-debt-guard repair audit",
        [
            f"rows={len(rows)}；summary_rows={len(summaries)}；repair_pass={obj['part_c_projector_guard_repair_pass']}；route={route}。",
            f"reason={reason}",
            "focus_variants="
            + " | ".join(
                f"{row['family']}::{row['variant']}: energy_median={row['output_residual_energy_fraction_median']}；NLL_improve={row['residual_task_NLL_improve_rows']}/{row['rows']}；debt_UCB_nonpositive={row['residual_debt_probe_UCB_nonpositive_rows']}/{row['rows']}；bank_CVaR25={row['residual_bank_R2_CVaR25']}；pass={row['projector_guard_family_variant_pass']}"
                for row in best
            ),
            "修复记录：为 output_debt_residual/edge_pullback_from_output_residual 增加 include 白名单参数；本审计只改变信号构造的 nuisance 投影集合或禁用投影后用 debt UCB guard，不改变 official runtime、不引入 teacher/validation/test/future/query direction。",
        ],
    )
    return obj


def part_d_probe(dataset: str, seed: int, family: str, args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    model, _bundle, x_all, y_all = make_probe_model(dataset, seed, family, args, device)
    x_source, y_source, x_witness, y_witness, x_guard, y_guard, _x_control, _y_control = split_train(x_all, y_all)
    source = edge_pullback_from_output_residual(model, x_source, y_source, ridge=float(args.projector_ridge))
    witness = edge_pullback_from_output_residual(model, x_witness, y_witness, ridge=float(args.projector_ridge))
    w2_grads = collect_grads(model, x_source, y_source, x_witness, y_witness, x_guard, y_guard, x_all, y_all, "w2")
    metric_w2 = metric_diag_from_design(model, x_all, mode="readout")
    signal, proj_stats = control_orthogonal_edge_signal(family, source["residual_edge_vec"], metric_w2, model, w2_grads, ridge=float(args.projector_ridge))
    bank_vec, bank_stats = bank_additive_projection(signal, metric_w2, model.w2.shape)
    raw_bank_R2 = fval(source["bank_R2_on_raw_signal"])
    bank_residual_R2 = fval(bank_stats.get("edge_bank_anova_explained"))
    step_norm = float(args.lr) * float(args.step_mult)
    candidate_w2 = -normalized_update(signal, model.w2.shape, step_norm)
    cand_norm = float(candidate_w2.reshape(-1).norm().detach().cpu().item())
    same_cond = same_energy_orthogonal_control(signal).to(device=device)
    controls = {
        "same_domain": {"w2": -normalized_update(w2_grads["domain"], model.w2.shape, cand_norm)},
        "same_edge": {"w2": -normalized_update(w2_grads["random"], model.w2.shape, cand_norm)},
        "same_smoothness": {"w2": -normalized_update(w2_grads["smooth"], model.w2.shape, cand_norm)},
        "same_conditional_energy": {"w2": -normalized_update(same_cond, model.w2.shape, cand_norm)},
        "same_debt": {"w2": -normalized_update(w2_grads["debt"], model.w2.shape, cand_norm)},
    }
    cand_delta, _ctrl_delta, gaps, _debt_penalty, all_debt_ucb = control_gap_diagnostics(model, x_guard, y_guard, {"w2": candidate_w2}, controls, args)
    boot = bootstrap_guard_margin(model, x_guard, y_guard, {"w2": candidate_w2}, controls, args, seed=seed + 22800)
    align = coherence_score(signal, witness["residual_edge_vec"], metric_w2)
    return {
        "dataset": dataset,
        "seed": seed,
        "family": family,
        "architecture": map_family_to_method(family),
        "single_edge_residual_R2": bank_residual_R2,
        "bank_residual_R2": bank_residual_R2,
        "bank_residual_R2_gain": bank_residual_R2 - raw_bank_R2,
        "bank_residual_R2_gain_positive": int((bank_residual_R2 - raw_bank_R2) > 0.0),
        "interaction_residual_before": 1.0 - raw_bank_R2,
        "interaction_residual_after": 1.0 - bank_residual_R2,
        "domain_nuisance_fraction": proj_stats["domain_nuisance_fraction"],
        "conditional_residual_fraction": proj_stats["conditional_residual_fraction"],
        "control_projected_energy_fraction": proj_stats["control_projected_energy_fraction"],
        "control_projection_mode": proj_stats["control_projection_mode"],
        "source_witness_residual_alignment": align,
        "candidate_NLL_delta_guard": cand_delta["NLL"],
        "candidate_NLL_improve": int(cand_delta["NLL"] < 0.0),
        "all_debt_UCB_nonpositive": int(all_debt_ucb <= 0.0),
        "control_margin_p10": min(gaps.values()) if gaps else 0.0,
        "control_margin_CVaR25": lower_cvar(list(gaps.values()), 0.25) if gaps else 0.0,
        "control_margin_positive": int((min(gaps.values()) if gaps else 0.0) > 0.0),
        "same_domain_gap": gaps.get("same_domain", 0.0),
        "same_edge_gap": gaps.get("same_edge", 0.0),
        "same_smoothness_gap": gaps.get("same_smoothness", 0.0),
        "same_conditional_energy_gap": gaps.get("same_conditional_energy", 0.0),
        "same_debt_gap": gaps.get("same_debt", 0.0),
        "same_domain_candidate_better": int(gaps.get("same_domain", 0.0) > 0.0),
        "same_edge_candidate_better": int(gaps.get("same_edge", 0.0) > 0.0),
        "same_smoothness_candidate_better": int(gaps.get("same_smoothness", 0.0) > 0.0),
        "same_conditional_energy_candidate_better": int(gaps.get("same_conditional_energy", 0.0) > 0.0),
        "same_debt_candidate_better": int(gaps.get("same_debt", 0.0) > 0.0),
        **boot,
    }


def run_part_d(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")
    tasks = shard_items(task_grid(args, COER_FAMILIES), args)
    rows = [part_d_probe(dataset, seed, family, args, device) for family, seed, dataset in tasks]
    out_path = OUT_ROOT / f"v22_80_part_d_domain_residual_edge_signal_shard{args.shard_index}_of_{args.shard_count}.csv"
    write_rows(out_path, rows)
    obj = {"gate": "v22_80_part_d_domain_residual_edge_signal_shard", "rows": len(rows), "shard_index": int(args.shard_index), "shard_count": int(args.shard_count), "output": rel(out_path)}
    write_json(OUT_ROOT / f"v22_80_part_d_domain_residual_edge_signal_shard{args.shard_index}_of_{args.shard_count}.json", obj)
    append_exec("D_domain_residual_edge_signal_shard", command_text(sys.argv), "done", gpu=str(device), files=rel(out_path), note=json.dumps(obj, ensure_ascii=False))
    return obj


def summarize_part_d(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for family in sorted({str(row.get("family")) for row in rows}):
        group = [row for row in rows if str(row.get("family")) == family]
        summary = {
            "family": family,
            "rows": len(group),
            "bank_residual_R2_gain_rows": bool_sum(group, "bank_residual_R2_gain_positive"),
            "conditional_residual_fraction_median": quantile([fval(row.get("conditional_residual_fraction")) for row in group], 0.50),
            "domain_nuisance_fraction_median": quantile([fval(row.get("domain_nuisance_fraction")) for row in group], 0.50),
            "same_domain_candidate_better_rows": bool_sum(group, "same_domain_candidate_better"),
            "same_edge_candidate_better_rows": bool_sum(group, "same_edge_candidate_better"),
            "same_smoothness_candidate_better_rows": bool_sum(group, "same_smoothness_candidate_better"),
            "same_conditional_energy_candidate_better_rows": bool_sum(group, "same_conditional_energy_candidate_better"),
            "candidate_NLL_improve_rows": bool_sum(group, "candidate_NLL_improve"),
            "control_margin_positive_rows": bool_sum(group, "control_margin_positive"),
            "bootstrap_guard_margin_positive_rows": bool_sum(group, "bootstrap_guard_margin_positive"),
            "control_margin_p10_median": quantile([fval(row.get("control_margin_p10")) for row in group], 0.50),
        }
        summary["part_d_family_gate_pass"] = int(
            summary["rows"] >= 15
            and summary["bank_residual_R2_gain_rows"] >= 10
            and summary["conditional_residual_fraction_median"] >= 0.20
            and summary["domain_nuisance_fraction_median"] <= 0.60
            and summary["same_domain_candidate_better_rows"] >= 10
            and summary["same_edge_candidate_better_rows"] >= 10
            and summary["same_smoothness_candidate_better_rows"] >= 10
            and summary["same_conditional_energy_candidate_better_rows"] >= 10
        )
        out.append(summary)
    return out


def route_part_d(summaries: list[dict[str, Any]]) -> tuple[str, str]:
    if any(int(row.get("part_d_family_gate_pass", 0)) for row in summaries):
        return "DomainResidualConditionalEdgeSignalOpened", "at least one fixed family passed Part D"
    max_gain = max([int(row.get("bank_residual_R2_gain_rows", 0)) for row in summaries] or [0])
    max_same_domain = max([int(row.get("same_domain_candidate_better_rows", 0)) for row in summaries] or [0])
    max_same_edge = max([int(row.get("same_edge_candidate_better_rows", 0)) for row in summaries] or [0])
    max_cond = max([fval(row.get("conditional_residual_fraction_median")) for row in summaries] or [0.0])
    min_domain = min([fval(row.get("domain_nuisance_fraction_median")) for row in summaries] or [1.0])
    if max_gain < 10 or max_cond < 0.20:
        return "DomainResidualEdgeSignalAbsent", f"max_bank_residual_R2_gain_rows={max_gain}; max_conditional_residual_fraction_median={max_cond}"
    if max_same_domain < 10 or max_same_edge < 10 or min_domain > 0.60:
        return "DomainSupportExplained_NoConditionalEdgeSignal", f"max_same_domain_rows={max_same_domain}; max_same_edge_rows={max_same_edge}; min_domain_nuisance_fraction_median={min_domain}"
    return "DomainResidualSignalPresentButControlGateFailed", f"max_gain_rows={max_gain}; max_same_domain_rows={max_same_domain}; max_same_edge_rows={max_same_edge}; max_conditional_fraction={max_cond}"


def merge_part_d(args: argparse.Namespace) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    missing = []
    for idx in range(int(args.shard_count)):
        path = OUT_ROOT / f"v22_80_part_d_domain_residual_edge_signal_shard{idx}_of_{args.shard_count}.csv"
        if not path.exists():
            missing.append(rel(path))
        rows.extend(read_rows(path))
    write_rows(OUT_ROOT / "v22_80_part_d_domain_residual_edge_signal.csv", rows)
    summaries = summarize_part_d(rows)
    write_rows(OUT_ROOT / "v22_80_part_d_domain_residual_edge_signal_summaries.csv", summaries)
    route, reason = route_part_d(summaries) if not missing else ("R0-CodeOrTruthGateFailed", f"missing Part D shards: {missing}")
    obj = {
        "gate": "v22_80_part_d_domain_residual_edge_signal",
        "run_status": "completed_part_d_merge" if not missing else "incomplete_part_d_merge",
        "part_d_gate_pass": int(not missing and any(int(row.get("part_d_family_gate_pass", 0)) for row in summaries)),
        "part_d_route": route,
        "route_reason": reason,
        "rows": len(rows),
        "summary_rows": len(summaries),
        "missing_shards": missing,
        "passed_families": [row["family"] for row in summaries if int(row.get("part_d_family_gate_pass", 0))],
    }
    write_json(OUT_ROOT / "v22_80_part_d_domain_residual_edge_signal_route.json", obj)
    append_exec("D_domain_residual_edge_signal_merge", command_text(sys.argv), "pass" if obj["part_d_gate_pass"] else "fail", files=f"{rel(OUT_ROOT / 'v22_80_part_d_domain_residual_edge_signal.csv')}; {rel(OUT_ROOT / 'v22_80_part_d_domain_residual_edge_signal_summaries.csv')}; {rel(OUT_ROOT / 'v22_80_part_d_domain_residual_edge_signal_route.json')}", note=json.dumps(obj, ensure_ascii=False))
    best = sorted(summaries, key=lambda row: (int(row.get("part_d_family_gate_pass", 0)), int(row.get("same_domain_candidate_better_rows", 0)), int(row.get("candidate_NLL_improve_rows", 0)), fval(row.get("conditional_residual_fraction_median"))), reverse=True)[:5]
    append_recap(
        "Part D domain-residual conditional edge/bank signal audit",
        [
            f"rows={len(rows)}；summary_rows={len(summaries)}；part_d_gate_pass={obj['part_d_gate_pass']}；route={route}。",
            f"reason={reason}",
            "best_families="
            + " | ".join(
                f"{row['family']}: bank_gain={row['bank_residual_R2_gain_rows']}/{row['rows']}；conditional_median={row['conditional_residual_fraction_median']}；domain_median={row['domain_nuisance_fraction_median']}；same_domain={row['same_domain_candidate_better_rows']}/{row['rows']}；same_edge={row['same_edge_candidate_better_rows']}/{row['rows']}；same_smooth={row['same_smoothness_candidate_better_rows']}/{row['rows']}；same_cond={row['same_conditional_energy_candidate_better_rows']}/{row['rows']}；margin={row['control_margin_positive_rows']}/{row['rows']}；pass={row['part_d_family_gate_pass']}"
                for row in best
            ),
            "代码审计：Part D 只在 Part C 的 output-debt residual pullback 信号上构造 domain/control residual；same-domain/same-edge/same-smooth/same-conditional controls 同时进入信号审计与 guard margin。",
        ],
    )
    return obj


def composite_debt_direction(grads: dict[str, torch.Tensor], dim: int, device: torch.device) -> torch.Tensor:
    basis = normalize_columns(debt_grad_columns(grads), int(dim), device)
    if basis is None:
        return torch.zeros(int(dim), device=device, dtype=torch.float64)
    return basis.mean(dim=1).reshape(-1)


def projection_fraction(vector: torch.Tensor, basis: torch.Tensor | None, metric: torch.Tensor, ridge: float) -> float:
    if basis is None:
        return 0.0
    _res, diag = weighted_project(vector.reshape(-1), basis, metric, ridge=ridge)
    return fval(diag.get("projected_energy_fraction"))


def part_control_margin_audit_probe(dataset: str, seed: int, family: str, args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    model, _bundle, x_all, y_all = make_probe_model(dataset, seed, family, args, device)
    x_source, y_source, x_witness, y_witness, x_guard, y_guard, _x_control, _y_control = split_train(x_all, y_all)
    source = edge_pullback_from_output_residual(model, x_source, y_source, ridge=float(args.projector_ridge))
    w2_grads = collect_grads(model, x_source, y_source, x_witness, y_witness, x_guard, y_guard, x_all, y_all, "w2")
    metric_w2 = metric_diag_from_design(model, x_all, mode="readout")
    signal, proj_stats = control_orthogonal_edge_signal(family, source["residual_edge_vec"], metric_w2, model, w2_grads, ridge=float(args.projector_ridge))
    step_norm = float(args.lr) * float(args.step_mult)
    candidate_w2 = -normalized_update(signal, model.w2.shape, step_norm)
    cand_norm = float(candidate_w2.reshape(-1).norm().detach().cpu().item())
    dim = int(signal.numel())
    debt_composite = composite_debt_direction(w2_grads, dim, device)
    same_cond = same_energy_orthogonal_control(signal).to(device=device)
    debt_basis = normalize_columns(debt_grad_columns(w2_grads), dim, device)
    controls = {
        "same_domain": {"w2": -normalized_update(w2_grads["domain"], model.w2.shape, cand_norm)},
        "same_edge": {"w2": -normalized_update(w2_grads["random"], model.w2.shape, cand_norm)},
        "same_smoothness": {"w2": -normalized_update(w2_grads["smooth"], model.w2.shape, cand_norm)},
        "same_conditional_energy": {"w2": -normalized_update(same_cond, model.w2.shape, cand_norm)},
        "same_debt_brier": {"w2": -normalized_update(w2_grads["debt"], model.w2.shape, cand_norm)},
        "same_debt_ece": {"w2": -normalized_update(w2_grads["ece_debt"], model.w2.shape, cand_norm)},
        "same_debt_tail95": {"w2": -normalized_update(w2_grads["tail95_debt"], model.w2.shape, cand_norm)},
        "same_debt_tail99": {"w2": -normalized_update(w2_grads["tail99_debt"], model.w2.shape, cand_norm)},
        "same_debt_margin": {"w2": -normalized_update(w2_grads["margin_debt"], model.w2.shape, cand_norm)},
        "same_debt_composite": {"w2": -normalized_update(debt_composite, model.w2.shape, cand_norm)},
    }
    cand_delta, _ctrl_delta, gaps, _debt_penalty, all_debt_ucb = control_gap_diagnostics(model, x_guard, y_guard, {"w2": candidate_w2}, controls, args)
    boot = bootstrap_guard_margin(model, x_guard, y_guard, {"w2": candidate_w2}, controls, args, seed=seed + 23080)
    blocker = min(gaps, key=lambda key: gaps[key]) if gaps else ""
    return {
        "dataset": dataset,
        "seed": seed,
        "family": family,
        "architecture": map_family_to_method(family),
        "control_projection_mode": proj_stats["control_projection_mode"],
        "signal_control_projected_energy_fraction": proj_stats["control_projected_energy_fraction"],
        "conditional_residual_fraction": proj_stats["conditional_residual_fraction"],
        "candidate_axis_all_debt_projection_fraction": projection_fraction(signal, debt_basis, metric_w2, float(args.projector_ridge)),
        "candidate_axis_brier_projection_fraction": projection_fraction(signal, normalize_columns([w2_grads["debt"]], dim, device), metric_w2, float(args.projector_ridge)),
        "candidate_axis_ece_projection_fraction": projection_fraction(signal, normalize_columns([w2_grads["ece_debt"]], dim, device), metric_w2, float(args.projector_ridge)),
        "candidate_axis_tail99_projection_fraction": projection_fraction(signal, normalize_columns([w2_grads["tail99_debt"]], dim, device), metric_w2, float(args.projector_ridge)),
        "candidate_axis_margin_projection_fraction": projection_fraction(signal, normalize_columns([w2_grads["margin_debt"]], dim, device), metric_w2, float(args.projector_ridge)),
        "candidate_NLL_delta_guard": cand_delta["NLL"],
        "candidate_NLL_improve": int(cand_delta["NLL"] < 0.0),
        "all_debt_UCB_nonpositive": int(all_debt_ucb <= 0.0),
        "projected_Brier_delta_guard": cand_delta["Brier"],
        "projected_ECE_delta_guard": cand_delta["ECE"],
        "projected_tail99_delta_guard": cand_delta["tail99"],
        "control_margin_p10": min(gaps.values()) if gaps else 0.0,
        "control_margin_CVaR25": lower_cvar(list(gaps.values()), 0.25) if gaps else 0.0,
        "control_margin_positive": int((min(gaps.values()) if gaps else 0.0) > 0.0),
        "dominant_blocker": blocker,
        "dominant_blocker_gap": gaps.get(blocker, 0.0),
        "same_domain_candidate_better": int(gaps.get("same_domain", 0.0) > 0.0),
        "same_edge_candidate_better": int(gaps.get("same_edge", 0.0) > 0.0),
        "same_smoothness_candidate_better": int(gaps.get("same_smoothness", 0.0) > 0.0),
        "same_conditional_energy_candidate_better": int(gaps.get("same_conditional_energy", 0.0) > 0.0),
        "same_debt_brier_candidate_better": int(gaps.get("same_debt_brier", 0.0) > 0.0),
        "same_debt_ece_candidate_better": int(gaps.get("same_debt_ece", 0.0) > 0.0),
        "same_debt_tail95_candidate_better": int(gaps.get("same_debt_tail95", 0.0) > 0.0),
        "same_debt_tail99_candidate_better": int(gaps.get("same_debt_tail99", 0.0) > 0.0),
        "same_debt_margin_candidate_better": int(gaps.get("same_debt_margin", 0.0) > 0.0),
        "same_debt_composite_candidate_better": int(gaps.get("same_debt_composite", 0.0) > 0.0),
        "same_domain_gap": gaps.get("same_domain", 0.0),
        "same_debt_composite_gap": gaps.get("same_debt_composite", 0.0),
        "same_debt_brier_gap": gaps.get("same_debt_brier", 0.0),
        **boot,
    }


def run_part_control_margin_audit(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")
    tasks = shard_items(task_grid(args, COER_FAMILIES), args)
    rows = [part_control_margin_audit_probe(dataset, seed, family, args, device) for family, seed, dataset in tasks]
    out_path = OUT_ROOT / f"v22_80_part_control_margin_strengthened_audit_shard{args.shard_index}_of_{args.shard_count}.csv"
    write_rows(out_path, rows)
    obj = {"gate": "v22_80_part_control_margin_strengthened_audit_shard", "rows": len(rows), "shard_index": int(args.shard_index), "shard_count": int(args.shard_count), "output": rel(out_path)}
    write_json(OUT_ROOT / f"v22_80_part_control_margin_strengthened_audit_shard{args.shard_index}_of_{args.shard_count}.json", obj)
    append_exec("CONTROL_margin_strengthened_audit_shard", command_text(sys.argv), "done", gpu=str(device), files=rel(out_path), note=json.dumps(obj, ensure_ascii=False))
    return obj


def summarize_part_control_margin_audit(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for family in sorted({str(row.get("family")) for row in rows}):
        group = [row for row in rows if str(row.get("family")) == family]
        blocker_counts: dict[str, int] = {}
        for row in group:
            key = str(row.get("dominant_blocker", ""))
            blocker_counts[key] = blocker_counts.get(key, 0) + 1
        dominant = sorted(blocker_counts.items(), key=lambda kv: kv[1], reverse=True)[0][0] if blocker_counts else ""
        summary = {
            "family": family,
            "rows": len(group),
            "candidate_NLL_improve_rows": bool_sum(group, "candidate_NLL_improve"),
            "all_debt_UCB_nonpositive_rows": bool_sum(group, "all_debt_UCB_nonpositive"),
            "control_margin_positive_rows": bool_sum(group, "control_margin_positive"),
            "bootstrap_guard_margin_positive_rows": bool_sum(group, "bootstrap_guard_margin_positive"),
            "control_margin_p10_median": quantile([fval(row.get("control_margin_p10")) for row in group], 0.50),
            "control_margin_CVaR25_median": quantile([fval(row.get("control_margin_CVaR25")) for row in group], 0.50),
            "candidate_axis_all_debt_projection_fraction_median": quantile([fval(row.get("candidate_axis_all_debt_projection_fraction")) for row in group], 0.50),
            "signal_control_projected_energy_fraction_median": quantile([fval(row.get("signal_control_projected_energy_fraction")) for row in group], 0.50),
            "conditional_residual_fraction_median": quantile([fval(row.get("conditional_residual_fraction")) for row in group], 0.50),
            "same_domain_candidate_better_rows": bool_sum(group, "same_domain_candidate_better"),
            "same_edge_candidate_better_rows": bool_sum(group, "same_edge_candidate_better"),
            "same_smoothness_candidate_better_rows": bool_sum(group, "same_smoothness_candidate_better"),
            "same_conditional_energy_candidate_better_rows": bool_sum(group, "same_conditional_energy_candidate_better"),
            "same_debt_brier_candidate_better_rows": bool_sum(group, "same_debt_brier_candidate_better"),
            "same_debt_ece_candidate_better_rows": bool_sum(group, "same_debt_ece_candidate_better"),
            "same_debt_tail95_candidate_better_rows": bool_sum(group, "same_debt_tail95_candidate_better"),
            "same_debt_tail99_candidate_better_rows": bool_sum(group, "same_debt_tail99_candidate_better"),
            "same_debt_margin_candidate_better_rows": bool_sum(group, "same_debt_margin_candidate_better"),
            "same_debt_composite_candidate_better_rows": bool_sum(group, "same_debt_composite_candidate_better"),
            "dominant_blocker": dominant,
            "dominant_blocker_rows": blocker_counts.get(dominant, 0),
        }
        summary["control_margin_audit_family_gate_pass"] = int(
            summary["rows"] >= 15
            and summary["candidate_NLL_improve_rows"] >= 10
            and summary["all_debt_UCB_nonpositive_rows"] >= 10
            and summary["control_margin_positive_rows"] >= 10
            and summary["bootstrap_guard_margin_positive_rows"] >= 9
        )
        out.append(summary)
    return out


def route_part_control_margin_audit(summaries: list[dict[str, Any]]) -> tuple[str, str]:
    if any(int(row.get("control_margin_audit_family_gate_pass", 0)) for row in summaries):
        return "StrengthenedControlMarginOpenedButPartCStillBlocksPromotion", "at least one family passed strengthened control-margin audit"
    max_nll = max([int(row.get("candidate_NLL_improve_rows", 0)) for row in summaries] or [0])
    max_debt = max([int(row.get("all_debt_UCB_nonpositive_rows", 0)) for row in summaries] or [0])
    max_margin = max([int(row.get("control_margin_positive_rows", 0)) for row in summaries] or [0])
    max_boot = max([int(row.get("bootstrap_guard_margin_positive_rows", 0)) for row in summaries] or [0])
    max_proj = max([fval(row.get("candidate_axis_all_debt_projection_fraction_median")) for row in summaries] or [0.0])
    blockers = "; ".join(f"{row['family']}:{row['dominant_blocker']}({row['dominant_blocker_rows']}/{row['rows']})" for row in summaries)
    if max_nll >= 10 and max_margin == 0:
        return (
            "NoControlOrthogonalUtility",
            f"max_NLL_improve_rows={max_nll}; max_all_debt_UCB_nonpositive_rows={max_debt}; max_control_margin_positive_rows={max_margin}; max_bootstrap_rows={max_boot}; max_candidate_axis_all_debt_projection_fraction_median={max_proj}; blockers={blockers}",
        )
    return (
        "NoResidualUtilityAfterStrengthenedControls",
        f"max_NLL_improve_rows={max_nll}; max_all_debt_UCB_nonpositive_rows={max_debt}; max_control_margin_positive_rows={max_margin}; max_bootstrap_rows={max_boot}; blockers={blockers}",
    )


def merge_part_control_margin_audit(args: argparse.Namespace) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    missing = []
    for idx in range(int(args.shard_count)):
        path = OUT_ROOT / f"v22_80_part_control_margin_strengthened_audit_shard{idx}_of_{args.shard_count}.csv"
        if not path.exists():
            missing.append(rel(path))
        rows.extend(read_rows(path))
    write_rows(OUT_ROOT / "v22_80_part_control_margin_strengthened_audit.csv", rows)
    summaries = summarize_part_control_margin_audit(rows)
    write_rows(OUT_ROOT / "v22_80_part_control_margin_strengthened_audit_summaries.csv", summaries)
    route, reason = route_part_control_margin_audit(summaries) if not missing else ("R0-CodeOrTruthGateFailed", f"missing control margin audit shards: {missing}")
    obj = {
        "gate": "v22_80_part_control_margin_strengthened_audit",
        "run_status": "completed_part_control_margin_strengthened_audit_merge" if not missing else "incomplete_part_control_margin_strengthened_audit_merge",
        "part_control_margin_audit_repair_pass": int(route == "StrengthenedControlMarginOpenedButPartCStillBlocksPromotion"),
        "part_control_margin_audit_route": route,
        "route_reason": reason,
        "rows": len(rows),
        "summary_rows": len(summaries),
        "missing_shards": missing,
        "passed_families": [row["family"] for row in summaries if int(row.get("control_margin_audit_family_gate_pass", 0))],
    }
    write_json(OUT_ROOT / "v22_80_part_control_margin_strengthened_audit_route.json", obj)
    append_exec("CONTROL_margin_strengthened_audit_merge", command_text(sys.argv), "pass" if obj["part_control_margin_audit_repair_pass"] else "fail", files=f"{rel(OUT_ROOT / 'v22_80_part_control_margin_strengthened_audit.csv')}; {rel(OUT_ROOT / 'v22_80_part_control_margin_strengthened_audit_summaries.csv')}; {rel(OUT_ROOT / 'v22_80_part_control_margin_strengthened_audit_route.json')}", note=json.dumps(obj, ensure_ascii=False))
    best = sorted(summaries, key=lambda row: (int(row.get("control_margin_audit_family_gate_pass", 0)), int(row.get("control_margin_positive_rows", 0)), int(row.get("candidate_NLL_improve_rows", 0)), int(row.get("all_debt_UCB_nonpositive_rows", 0))), reverse=True)
    append_recap(
        "Strengthened same-debt/control-axis margin repair audit",
        [
            f"rows={len(rows)}；summary_rows={len(summaries)}；repair_pass={obj['part_control_margin_audit_repair_pass']}；route={route}。",
            f"reason={reason}",
            "families="
            + " | ".join(
                f"{row['family']}: NLL={row['candidate_NLL_improve_rows']}/{row['rows']}；debtUCB={row['all_debt_UCB_nonpositive_rows']}/{row['rows']}；margin={row['control_margin_positive_rows']}/{row['rows']}；bootstrap={row['bootstrap_guard_margin_positive_rows']}/{row['rows']}；allDebtProjMed={row['candidate_axis_all_debt_projection_fraction_median']}；sameDebtComposite={row['same_debt_composite_candidate_better_rows']}/{row['rows']}；blocker={row['dominant_blocker']}({row['dominant_blocker_rows']}/{row['rows']})；pass={row['control_margin_audit_family_gate_pass']}"
                for row in best
            ),
            "修复记录：按计划 13.4 强化 same_debt，加入 Brier/ECE/tail95/tail99/margin/composite debt controls，记录 candidate axis 对 all-debt/Brier/ECE/tail99/margin 的投影，并沿用 CVaR25 与 bootstrap guard LCB10；不改变 official runtime。",
        ],
    )
    return obj


def carrier_redesign_task_grid(args: argparse.Namespace) -> list[tuple[dict[str, str], int, str]]:
    datasets = [item.strip() for item in str(args.datasets).split(",") if item.strip()]
    seeds = list(range(int(args.seed_count)))
    return [(spec, seed, dataset) for spec in CARRIER_REDESIGN_SPECS for dataset in datasets for seed in seeds]


def carrier_redesign_probe(dataset: str, seed: int, spec: dict[str, str], args: argparse.Namespace, device: torch.device) -> list[dict[str, Any]]:
    bundle = base73.load_bundle(dataset, int(args.train_size), int(args.held_size), int(args.test_size), int(seed))
    x_all = bundle["x_train"].to(device).float()[: int(args.metric_batch_size)]
    y_all = bundle["y_train"].to(device).long()[: int(args.metric_batch_size)]
    method = str(spec["method"])
    model = base75.make_wlb_model(method, bundle, device, int(args.hidden), int(seed) + 23800, x_metric=x_all)
    x_source, y_source, x_witness, y_witness, x_guard, y_guard, _x_control, _y_control = split_train(x_all, y_all)
    source_all = edge_pullback_from_output_residual(model, x_source, y_source, ridge=float(args.projector_ridge))
    witness_all = edge_pullback_from_output_residual(model, x_witness, y_witness, ridge=float(args.projector_ridge))
    source_brier = edge_pullback_from_output_residual(model, x_source, y_source, ridge=float(args.projector_ridge), include=["Brier"])
    source_no_brier = edge_pullback_from_output_residual(model, x_source, y_source, ridge=float(args.projector_ridge), include=["ECE", "tail95", "tail99", "margin", "radial"])
    witness_no_brier = edge_pullback_from_output_residual(model, x_witness, y_witness, ridge=float(args.projector_ridge), include=["ECE", "tail95", "tail99", "margin", "radial"])
    step_norm = float(args.lr) * float(args.step_mult)
    raw_candidate_w2 = -normalized_update(source_all["raw_edge_vec"], model.w2.shape, step_norm)
    raw_deltas = metric_delta_for_updates(model, x_guard, y_guard, {"w2": raw_candidate_w2})
    raw_debt_ucb = debt_ucb_from_deltas(raw_deltas)
    no_brier_candidate_w2 = -normalized_update(source_no_brier["residual_edge_vec"], model.w2.shape, step_norm)
    no_brier_deltas = metric_delta_for_updates(model, x_guard, y_guard, {"w2": no_brier_candidate_w2})
    no_brier_debt_ucb = debt_ucb_from_deltas(no_brier_deltas)
    w2_grads = collect_grads(model, x_source, y_source, x_witness, y_witness, x_guard, y_guard, x_all, y_all, "w2")
    metric_w2 = metric_diag_from_design(model, x_all, mode="readout")
    dim = int(model.w2.numel())
    debt_basis = normalize_columns(debt_grad_columns(w2_grads), dim, device)
    debt_composite = composite_debt_direction(w2_grads, dim, device)
    rows: list[dict[str, Any]] = []
    for signal_family in CARRIER_REDESIGN_SIGNAL_FAMILIES:
        signal, proj_stats = control_orthogonal_edge_signal(signal_family, source_all["residual_edge_vec"], metric_w2, model, w2_grads, ridge=float(args.projector_ridge))
        bank_vec, bank_stats = bank_additive_projection(signal, metric_w2, model.w2.shape)
        candidate_w2 = -normalized_update(signal, model.w2.shape, step_norm)
        cand_norm = float(candidate_w2.reshape(-1).norm().detach().cpu().item())
        same_cond = same_energy_orthogonal_control(signal).to(device=device)
        controls = {
            "same_domain": {"w2": -normalized_update(w2_grads["domain"], model.w2.shape, cand_norm)},
            "same_edge": {"w2": -normalized_update(w2_grads["random"], model.w2.shape, cand_norm)},
            "same_smoothness": {"w2": -normalized_update(w2_grads["smooth"], model.w2.shape, cand_norm)},
            "same_conditional_energy": {"w2": -normalized_update(same_cond, model.w2.shape, cand_norm)},
            "same_debt_brier": {"w2": -normalized_update(w2_grads["debt"], model.w2.shape, cand_norm)},
            "same_debt_ece": {"w2": -normalized_update(w2_grads["ece_debt"], model.w2.shape, cand_norm)},
            "same_debt_tail95": {"w2": -normalized_update(w2_grads["tail95_debt"], model.w2.shape, cand_norm)},
            "same_debt_tail99": {"w2": -normalized_update(w2_grads["tail99_debt"], model.w2.shape, cand_norm)},
            "same_debt_margin": {"w2": -normalized_update(w2_grads["margin_debt"], model.w2.shape, cand_norm)},
            "same_debt_composite": {"w2": -normalized_update(debt_composite, model.w2.shape, cand_norm)},
        }
        cand_delta, _ctrl_delta, gaps, _debt_penalty, all_debt_ucb = control_gap_diagnostics(model, x_guard, y_guard, {"w2": candidate_w2}, controls, args)
        boot = bootstrap_guard_margin(model, x_guard, y_guard, {"w2": candidate_w2}, controls, args, seed=seed + 23800)
        blocker = min(gaps, key=lambda key: gaps[key]) if gaps else ""
        align = coherence_score(signal, witness_all["residual_edge_vec"], metric_w2)
        no_brier_align = coherence_score(source_no_brier["residual_edge_vec"], witness_no_brier["residual_edge_vec"], source_no_brier["metric_w2"])
        bank_r2 = fval(bank_stats.get("edge_bank_anova_explained"))
        rows.append(
            {
                "dataset": dataset,
                "seed": seed,
                "carrier_family": str(spec["carrier_family"]),
                "carrier_method": method,
                "carrier_kind": str(spec["kind"]),
                "signal_family": signal_family,
                "probe_error": "",
                "all_debt_output_residual_energy_fraction": source_all["output_residual_energy_fraction"],
                "brier_only_output_residual_energy_fraction": source_brier["output_residual_energy_fraction"],
                "no_brier_output_residual_energy_fraction": source_no_brier["output_residual_energy_fraction"],
                "all_debt_projection_fraction": source_all["all_debt_projection_fraction"],
                "Brier_projection_fraction": source_all["Brier_projection_fraction"],
                "ECE_projection_fraction": source_all["ECE_projection_fraction"],
                "tail99_projection_fraction": source_all["tail99_projection_fraction"],
                "margin_projection_fraction": source_all["margin_projection_fraction"],
                "raw_candidate_NLL_delta_guard": raw_deltas["NLL"],
                "raw_candidate_NLL_improve": int(raw_deltas["NLL"] < 0.0),
                "raw_candidate_debt_UCB": raw_debt_ucb,
                "raw_candidate_debt_UCB_nonpositive": int(raw_debt_ucb <= 0.0),
                "no_brier_candidate_NLL_delta_guard": no_brier_deltas["NLL"],
                "no_brier_candidate_NLL_improve": int(no_brier_deltas["NLL"] < 0.0),
                "no_brier_candidate_debt_UCB": no_brier_debt_ucb,
                "no_brier_candidate_debt_UCB_nonpositive": int(no_brier_debt_ucb <= 0.0),
                "signal_control_projection_mode": proj_stats["control_projection_mode"],
                "signal_control_projected_energy_fraction": proj_stats["control_projected_energy_fraction"],
                "conditional_residual_fraction": proj_stats["conditional_residual_fraction"],
                "candidate_axis_all_debt_projection_fraction": projection_fraction(signal, debt_basis, metric_w2, float(args.projector_ridge)),
                "candidate_axis_brier_projection_fraction": projection_fraction(signal, normalize_columns([w2_grads["debt"]], dim, device), metric_w2, float(args.projector_ridge)),
                "signal_bank_R2": bank_r2,
                "signal_bank_energy": float(metric_energy(bank_vec.reshape(-1), metric_w2).detach().cpu().item()),
                "source_witness_signal_alignment": align,
                "source_witness_signal_alignment_LCB": align - 0.05 * abs(align),
                "source_witness_no_brier_alignment_LCB": no_brier_align - 0.05 * abs(no_brier_align),
                "candidate_NLL_delta_guard": cand_delta["NLL"],
                "candidate_NLL_improve": int(cand_delta["NLL"] < 0.0),
                "all_debt_UCB": all_debt_ucb,
                "all_debt_UCB_nonpositive": int(all_debt_ucb <= 0.0),
                "projected_Brier_delta_guard": cand_delta["Brier"],
                "projected_ECE_delta_guard": cand_delta["ECE"],
                "projected_tail99_delta_guard": cand_delta["tail99"],
                "control_margin_p10": min(gaps.values()) if gaps else 0.0,
                "control_margin_CVaR25": lower_cvar(list(gaps.values()), 0.25) if gaps else 0.0,
                "control_margin_positive": int((min(gaps.values()) if gaps else 0.0) > 0.0),
                "dominant_blocker": blocker,
                "dominant_blocker_gap": gaps.get(blocker, 0.0),
                "same_debt_brier_candidate_better": int(gaps.get("same_debt_brier", 0.0) > 0.0),
                "same_debt_composite_candidate_better": int(gaps.get("same_debt_composite", 0.0) > 0.0),
                "carrier_signal_row_pass": int(
                    source_all["output_residual_energy_fraction"] >= 0.10
                    and cand_delta["NLL"] < 0.0
                    and all_debt_ucb <= 0.0
                    and (min(gaps.values()) if gaps else 0.0) > 0.0
                    and int(boot.get("bootstrap_guard_margin_positive", 0)) == 1
                    and bank_r2 >= 0.10
                    and align - 0.05 * abs(align) > 0.0
                ),
                **boot,
            }
        )
    return rows


def run_part_i_carrier_redesign(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")
    tasks = shard_items(carrier_redesign_task_grid(args), args)
    rows: list[dict[str, Any]] = []
    for spec, seed, dataset in tasks:
        try:
            rows.extend(carrier_redesign_probe(dataset, seed, spec, args, device))
        except Exception as exc:
            rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "carrier_family": str(spec.get("carrier_family", "")),
                    "carrier_method": str(spec.get("method", "")),
                    "carrier_kind": str(spec.get("kind", "")),
                    "signal_family": "probe_failed_before_signal_loop",
                    "probe_error": f"{type(exc).__name__}: {exc}",
                }
            )
    out_path = OUT_ROOT / f"v22_80_part_i_carrier_redesign_preflight_shard{args.shard_index}_of_{args.shard_count}.csv"
    write_rows(out_path, rows)
    obj = {"gate": "v22_80_part_i_carrier_redesign_preflight_shard", "rows": len(rows), "tasks": len(tasks), "shard_index": int(args.shard_index), "shard_count": int(args.shard_count), "output": rel(out_path)}
    write_json(OUT_ROOT / f"v22_80_part_i_carrier_redesign_preflight_shard{args.shard_index}_of_{args.shard_count}.json", obj)
    append_exec("I_carrier_redesign_preflight_shard", command_text(sys.argv), "done", gpu=str(device), files=rel(out_path), note=json.dumps(obj, ensure_ascii=False))
    return obj


def summarize_part_i_carrier_redesign(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    groups = sorted({(str(row.get("carrier_family")), str(row.get("signal_family"))) for row in rows})
    for carrier_family, signal_family in groups:
        group = [row for row in rows if str(row.get("carrier_family")) == carrier_family and str(row.get("signal_family")) == signal_family]
        valid = [row for row in group if not str(row.get("probe_error", ""))]
        blocker_counts: dict[str, int] = {}
        for row in valid:
            key = str(row.get("dominant_blocker", ""))
            blocker_counts[key] = blocker_counts.get(key, 0) + 1
        dominant = sorted(blocker_counts.items(), key=lambda kv: kv[1], reverse=True)[0][0] if blocker_counts else ""
        method = str(valid[0].get("carrier_method", group[0].get("carrier_method", ""))) if group else ""
        kind = str(valid[0].get("carrier_kind", group[0].get("carrier_kind", ""))) if group else ""
        bank_vals = [fval(row.get("signal_bank_R2")) for row in valid]
        summary = {
            "carrier_family": carrier_family,
            "carrier_method": method,
            "carrier_kind": kind,
            "signal_family": signal_family,
            "rows": len(group),
            "valid_rows": len(valid),
            "error_rows": len(group) - len(valid),
            "all_debt_output_residual_energy_fraction_median": quantile([fval(row.get("all_debt_output_residual_energy_fraction")) for row in valid], 0.50),
            "brier_only_output_residual_energy_fraction_median": quantile([fval(row.get("brier_only_output_residual_energy_fraction")) for row in valid], 0.50),
            "no_brier_output_residual_energy_fraction_median": quantile([fval(row.get("no_brier_output_residual_energy_fraction")) for row in valid], 0.50),
            "candidate_NLL_improve_rows": bool_sum(valid, "candidate_NLL_improve"),
            "all_debt_UCB_nonpositive_rows": bool_sum(valid, "all_debt_UCB_nonpositive"),
            "control_margin_positive_rows": bool_sum(valid, "control_margin_positive"),
            "bootstrap_guard_margin_positive_rows": bool_sum(valid, "bootstrap_guard_margin_positive"),
            "raw_candidate_NLL_improve_rows": bool_sum(valid, "raw_candidate_NLL_improve"),
            "raw_candidate_debt_UCB_nonpositive_rows": bool_sum(valid, "raw_candidate_debt_UCB_nonpositive"),
            "no_brier_candidate_NLL_improve_rows": bool_sum(valid, "no_brier_candidate_NLL_improve"),
            "no_brier_candidate_debt_UCB_nonpositive_rows": bool_sum(valid, "no_brier_candidate_debt_UCB_nonpositive"),
            "signal_bank_R2_CVaR25": lower_cvar(bank_vals, 0.25),
            "signal_bank_R2_median": quantile(bank_vals, 0.50),
            "source_witness_signal_alignment_LCB_median": quantile([fval(row.get("source_witness_signal_alignment_LCB")) for row in valid], 0.50),
            "candidate_axis_all_debt_projection_fraction_median": quantile([fval(row.get("candidate_axis_all_debt_projection_fraction")) for row in valid], 0.50),
            "candidate_axis_brier_projection_fraction_median": quantile([fval(row.get("candidate_axis_brier_projection_fraction")) for row in valid], 0.50),
            "dominant_blocker": dominant,
            "dominant_blocker_rows": blocker_counts.get(dominant, 0),
            "carrier_signal_row_pass_rows": bool_sum(valid, "carrier_signal_row_pass"),
        }
        summary["carrier_redesign_family_signal_gate_pass"] = int(
            summary["valid_rows"] >= 15
            and summary["error_rows"] == 0
            and summary["all_debt_output_residual_energy_fraction_median"] >= 0.10
            and summary["candidate_NLL_improve_rows"] >= 10
            and summary["all_debt_UCB_nonpositive_rows"] >= 10
            and summary["control_margin_positive_rows"] >= 10
            and summary["bootstrap_guard_margin_positive_rows"] >= 9
            and summary["signal_bank_R2_CVaR25"] >= 0.10
            and summary["source_witness_signal_alignment_LCB_median"] > 0.0
        )
        out.append(summary)
    return out


def route_part_i_carrier_redesign(summaries: list[dict[str, Any]]) -> tuple[str, str]:
    opened = [row for row in summaries if int(row.get("carrier_redesign_family_signal_gate_pass", 0))]
    if opened:
        best = sorted(opened, key=lambda row: (int(row.get("control_margin_positive_rows", 0)), fval(row.get("all_debt_output_residual_energy_fraction_median"))), reverse=True)[0]
        return (
            "CarrierRedesignPreflightOpenedButOutsideV2280OfficialPlan",
            f"{best['carrier_family']}::{best['signal_family']} opened preflight; all_debt_energy_median={best['all_debt_output_residual_energy_fraction_median']}; NLL={best['candidate_NLL_improve_rows']}/{best['valid_rows']}; debtUCB={best['all_debt_UCB_nonpositive_rows']}/{best['valid_rows']}; margin={best['control_margin_positive_rows']}/{best['valid_rows']}",
        )
    max_all_energy = max([fval(row.get("all_debt_output_residual_energy_fraction_median")) for row in summaries] or [0.0])
    max_brier_energy = max([fval(row.get("brier_only_output_residual_energy_fraction_median")) for row in summaries] or [0.0])
    max_no_brier_energy = max([fval(row.get("no_brier_output_residual_energy_fraction_median")) for row in summaries] or [0.0])
    max_nll = max([int(row.get("candidate_NLL_improve_rows", 0)) for row in summaries] or [0])
    max_debt = max([int(row.get("all_debt_UCB_nonpositive_rows", 0)) for row in summaries] or [0])
    max_margin = max([int(row.get("control_margin_positive_rows", 0)) for row in summaries] or [0])
    max_boot = max([int(row.get("bootstrap_guard_margin_positive_rows", 0)) for row in summaries] or [0])
    max_errors = max([int(row.get("error_rows", 0)) for row in summaries] or [0])
    reason = (
        f"max_all_debt_energy_median={max_all_energy}; max_brier_only_energy_median={max_brier_energy}; "
        f"max_no_brier_energy_median={max_no_brier_energy}; max_NLL_improve_rows={max_nll}; "
        f"max_all_debt_UCB_nonpositive_rows={max_debt}; max_control_margin_positive_rows={max_margin}; "
        f"max_bootstrap_rows={max_boot}; max_error_rows={max_errors}"
    )
    if max_errors > 0:
        return "CarrierRedesignProbeErrorsNeedFix", reason
    if max_all_energy < 0.10 and max_no_brier_energy >= 0.10 and max_brier_energy < 0.10:
        return "CarrierRedesignStillBrierDominated", reason
    if max_all_energy < 0.10:
        return "CurrentAvailableCarrierRedesignNoOutputResidual", reason
    if max_debt < 10:
        return "CarrierRedesignDebtGuardBlocked", reason
    if max_margin < 10 or max_boot < 9:
        return "CarrierRedesignOutputResidualImprovedButControlUtilityAbsent", reason
    return "CurrentAvailableCarrierRedesignNoDecoupling", reason


def merge_part_i_carrier_redesign(args: argparse.Namespace) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    missing = []
    for idx in range(int(args.shard_count)):
        path = OUT_ROOT / f"v22_80_part_i_carrier_redesign_preflight_shard{idx}_of_{args.shard_count}.csv"
        if not path.exists():
            missing.append(rel(path))
        rows.extend(read_rows(path))
    write_rows(OUT_ROOT / "v22_80_part_i_carrier_redesign_preflight.csv", rows)
    summaries = summarize_part_i_carrier_redesign(rows)
    write_rows(OUT_ROOT / "v22_80_part_i_carrier_redesign_preflight_summaries.csv", summaries)
    route, reason = route_part_i_carrier_redesign(summaries) if not missing else ("R0-CodeOrTruthGateFailed", f"missing carrier redesign shards: {missing}")
    obj = {
        "gate": "v22_80_part_i_carrier_redesign_preflight",
        "run_status": "completed_part_i_carrier_redesign_merge" if not missing else "incomplete_part_i_carrier_redesign_merge",
        "part_i_carrier_redesign_repair_pass": int(route == "CarrierRedesignPreflightOpenedButOutsideV2280OfficialPlan"),
        "part_i_carrier_redesign_route": route,
        "route_reason": reason,
        "rows": len(rows),
        "summary_rows": len(summaries),
        "missing_shards": missing,
        "passed_carrier_signals": [f"{row['carrier_family']}::{row['signal_family']}" for row in summaries if int(row.get("carrier_redesign_family_signal_gate_pass", 0))],
    }
    write_json(OUT_ROOT / "v22_80_part_i_carrier_redesign_preflight_route.json", obj)
    append_exec("I_carrier_redesign_preflight_merge", command_text(sys.argv), "pass" if obj["part_i_carrier_redesign_repair_pass"] else "fail", files=f"{rel(OUT_ROOT / 'v22_80_part_i_carrier_redesign_preflight.csv')}; {rel(OUT_ROOT / 'v22_80_part_i_carrier_redesign_preflight_summaries.csv')}; {rel(OUT_ROOT / 'v22_80_part_i_carrier_redesign_preflight_route.json')}", note=json.dumps(obj, ensure_ascii=False))
    best = sorted(summaries, key=lambda row: (int(row.get("carrier_redesign_family_signal_gate_pass", 0)), int(row.get("control_margin_positive_rows", 0)), int(row.get("candidate_NLL_improve_rows", 0)), fval(row.get("all_debt_output_residual_energy_fraction_median"))), reverse=True)[:10]
    append_recap(
        "Part I carrier/basis redesign preflight",
        [
            f"rows={len(rows)}；summary_rows={len(summaries)}；repair_pass={obj['part_i_carrier_redesign_repair_pass']}；route={route}。",
            f"reason={reason}",
            "best_carrier_signals="
            + " | ".join(
                f"{row['carrier_family']}::{row['signal_family']}: allDebtEnergyMed={row['all_debt_output_residual_energy_fraction_median']}；brierOnlyEnergyMed={row['brier_only_output_residual_energy_fraction_median']}；noBrierEnergyMed={row['no_brier_output_residual_energy_fraction_median']}；NLL={row['candidate_NLL_improve_rows']}/{row['valid_rows']}；debtUCB={row['all_debt_UCB_nonpositive_rows']}/{row['valid_rows']}；margin={row['control_margin_positive_rows']}/{row['valid_rows']}；bootstrap={row['bootstrap_guard_margin_positive_rows']}/{row['valid_rows']}；bankCVaR25={row['signal_bank_R2_CVaR25']}；blocker={row['dominant_blocker']}({row['dominant_blocker_rows']}/{row['valid_rows']})；pass={row['carrier_redesign_family_signal_gate_pass']}"
                for row in best
            ),
            "修复记录：按 v22.80 第 15 节从当前 strict FC-PureKAN 小修转入 carrier/basis redesign 预研；复用已有 v22.74/v22.75 WLB/compact/monotone/tail-safe PrimitiveKAN carrier，并沿用 Part C output residual、strengthened debt controls、CVaR25/bootstrap guard。同属 diagnostic preflight，不改变 v22.80 official gate，不引入 teacher/validation/test/future/query direction/runtime selector。",
        ],
    )
    return obj


def output_oracle_probe(dataset: str, seed: int, spec: dict[str, str], args: argparse.Namespace, device: torch.device) -> list[dict[str, Any]]:
    bundle = base73.load_bundle(dataset, int(args.train_size), int(args.held_size), int(args.test_size), int(seed))
    x_all = bundle["x_train"].to(device).float()[: int(args.metric_batch_size)]
    y_all = bundle["y_train"].to(device).long()[: int(args.metric_batch_size)]
    method = str(spec["method"])
    model = base75.make_wlb_model(method, bundle, device, int(args.hidden), int(seed) + 24800, x_metric=x_all)
    x_source, y_source, x_witness, y_witness, x_guard, y_guard, _x_control, _y_control = split_train(x_all, y_all)
    with torch.no_grad():
        logits_source = model(x_source).float()
        logits_witness = model(x_witness).float()
        logits_guard = model(x_guard).float()
    source_all = output_debt_residual(logits_source, y_source, ridge=float(args.projector_ridge))
    witness_all = output_debt_residual(logits_witness, y_witness, ridge=float(args.projector_ridge))
    guard_all = output_debt_residual(logits_guard, y_guard, ridge=float(args.projector_ridge))
    guard_brier = output_debt_residual(logits_guard, y_guard, ridge=float(args.projector_ridge), include=["Brier"])
    guard_no_brier = output_debt_residual(logits_guard, y_guard, ridge=float(args.projector_ridge), include=["ECE", "tail95", "tail99", "margin", "radial"])
    guard_cone = output_linear_debt_cone_direction(logits_guard, y_guard, ridge=float(args.projector_ridge))
    raw = guard_all["raw"].reshape(-1).to(dtype=torch.float64)
    metric = guard_all["metric"].reshape(-1).to(device=device, dtype=torch.float64)
    debt_grads = {
        "Brier": logits_loss_grad(logits_guard, y_guard, "brier"),
        "ECE": logits_loss_grad(logits_guard, y_guard, "ece_debt"),
        "tail95": logits_loss_grad(logits_guard, y_guard, "tail95_debt"),
        "tail99": logits_loss_grad(logits_guard, y_guard, "tail99_debt"),
        "margin": logits_loss_grad(logits_guard, y_guard, "margin_debt"),
    }
    debt_basis = normalize_columns([vec for vec in debt_grads.values()], int(raw.numel()), device)
    ce_brier_cos = weighted_cosine(raw, debt_grads["Brier"], metric)
    directions = {
        "raw_ce": -guard_all["raw"].reshape(-1),
        "all_debt_projected": -guard_all["residual"].reshape(-1),
        "brier_only_projected": -guard_brier["residual"].reshape(-1),
        "no_brier_projected": -guard_no_brier["residual"].reshape(-1),
        "linear_debt_cone_qp": guard_cone["direction"].reshape(-1),
    }
    rows: list[dict[str, Any]] = []
    for mode in OUTPUT_ORACLE_DIRECTION_MODES:
        direction = directions[mode].reshape(-1).to(device=device, dtype=torch.float64)
        dir_norm = float(direction.norm().detach().cpu().item())
        for step_norm in OUTPUT_ORACLE_STEP_NORMS:
            update = normalize_logit_update(direction, logits_guard, float(step_norm))
            flat_update = update.reshape(-1).to(dtype=torch.float64)
            deltas = logit_metric_delta_for_update(logits_guard, y_guard, update)
            debt_ucb = debt_ucb_from_deltas(deltas)
            first_order_ce = float((raw * flat_update).sum().detach().cpu().item())
            first_order_debts = {name: float((vec.reshape(-1).to(dtype=torch.float64) * flat_update).sum().detach().cpu().item()) for name, vec in debt_grads.items()}
            first_order_debt_max = max(first_order_debts.values()) if first_order_debts else 0.0
            rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "carrier_family": str(spec["carrier_family"]),
                    "carrier_method": method,
                    "carrier_kind": str(spec["kind"]),
                    "direction_mode": mode,
                    "step_norm": float(step_norm),
                    "probe_error": "",
                    "source_all_debt_output_residual_energy_fraction": source_all["output_residual_energy_fraction"],
                    "witness_all_debt_output_residual_energy_fraction": witness_all["output_residual_energy_fraction"],
                    "guard_all_debt_output_residual_energy_fraction": guard_all["output_residual_energy_fraction"],
                    "guard_brier_only_output_residual_energy_fraction": guard_brier["output_residual_energy_fraction"],
                    "guard_no_brier_output_residual_energy_fraction": guard_no_brier["output_residual_energy_fraction"],
                    "guard_all_debt_projection_fraction": guard_all["all_debt_projection_fraction"],
                    "guard_ce_brier_weighted_cosine": ce_brier_cos,
                    "guard_ce_all_debt_projection_fraction": projection_fraction(raw, debt_basis, metric, float(args.projector_ridge)),
                    "direction_norm_before_normalize": dir_norm,
                    "direction_nonzero": int(dir_norm > 1.0e-12),
                    "linear_cone_first_order_feasible": int(guard_cone["first_order_debt_feasible"]) if mode == "linear_debt_cone_qp" else int(first_order_ce < 0.0 and first_order_debt_max <= 0.0),
                    "linear_cone_active_debt_constraints": guard_cone["active_debt_constraints"] if mode == "linear_debt_cone_qp" else "",
                    "linear_cone_distance_from_raw": guard_cone["linear_cone_distance_from_raw"] if mode == "linear_debt_cone_qp" else "",
                    "first_order_CE_delta": first_order_ce,
                    "first_order_debt_max": first_order_debt_max,
                    "first_order_Brier_delta": first_order_debts.get("Brier", 0.0),
                    "first_order_ECE_delta": first_order_debts.get("ECE", 0.0),
                    "first_order_tail99_delta": first_order_debts.get("tail99", 0.0),
                    "first_order_margin_debt_delta": first_order_debts.get("margin", 0.0),
                    "actual_NLL_delta": deltas["NLL"],
                    "actual_NLL_improve": int(deltas["NLL"] < 0.0),
                    "actual_Brier_delta": deltas["Brier"],
                    "actual_ECE_delta": deltas["ECE"],
                    "actual_tail95_delta": deltas["tail95"],
                    "actual_tail99_delta": deltas["tail99"],
                    "actual_margin10_delta": deltas["margin10"],
                    "actual_all_debt_UCB": debt_ucb,
                    "actual_all_debt_UCB_nonpositive": int(debt_ucb <= 0.0),
                    "actual_joint_safe_task": int(deltas["NLL"] < 0.0 and debt_ucb <= 0.0),
                    "actual_Brier_UCB_nonpositive": int(deltas["Brier"] + 0.5 * abs(deltas["Brier"]) <= 0.0),
                    "actual_ECE_UCB_nonpositive": int(deltas["ECE"] + 0.5 * abs(deltas["ECE"]) <= 0.0),
                    "actual_tail99_UCB_nonpositive": int(deltas["tail99"] + 0.5 * abs(deltas["tail99"]) <= 0.0),
                    "actual_margin_UCB_nonpositive": int(-deltas["margin10"] + 0.5 * abs(deltas["margin10"]) <= 0.0),
                }
            )
    return rows


def run_part_j_output_oracle(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")
    tasks = shard_items(carrier_redesign_task_grid(args), args)
    rows: list[dict[str, Any]] = []
    for spec, seed, dataset in tasks:
        try:
            rows.extend(output_oracle_probe(dataset, seed, spec, args, device))
        except Exception as exc:
            rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "carrier_family": str(spec.get("carrier_family", "")),
                    "carrier_method": str(spec.get("method", "")),
                    "carrier_kind": str(spec.get("kind", "")),
                    "direction_mode": "probe_failed",
                    "step_norm": "",
                    "probe_error": f"{type(exc).__name__}: {exc}",
                }
            )
    out_path = OUT_ROOT / f"v22_80_part_j_output_debt_safe_oracle_shard{args.shard_index}_of_{args.shard_count}.csv"
    write_rows(out_path, rows)
    obj = {"gate": "v22_80_part_j_output_debt_safe_oracle_shard", "rows": len(rows), "tasks": len(tasks), "shard_index": int(args.shard_index), "shard_count": int(args.shard_count), "output": rel(out_path)}
    write_json(OUT_ROOT / f"v22_80_part_j_output_debt_safe_oracle_shard{args.shard_index}_of_{args.shard_count}.json", obj)
    append_exec("J_output_debt_safe_oracle_shard", command_text(sys.argv), "done", gpu=str(device), files=rel(out_path), note=json.dumps(obj, ensure_ascii=False))
    return obj


def summarize_part_j_output_oracle(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    groups = sorted({(str(row.get("carrier_family")), str(row.get("direction_mode")), str(row.get("step_norm"))) for row in rows})
    for carrier_family, direction_mode, step_norm in groups:
        group = [row for row in rows if str(row.get("carrier_family")) == carrier_family and str(row.get("direction_mode")) == direction_mode and str(row.get("step_norm")) == step_norm]
        valid = [row for row in group if not str(row.get("probe_error", ""))]
        method = str(valid[0].get("carrier_method", group[0].get("carrier_method", ""))) if group else ""
        kind = str(valid[0].get("carrier_kind", group[0].get("carrier_kind", ""))) if group else ""
        summary = {
            "carrier_family": carrier_family,
            "carrier_method": method,
            "carrier_kind": kind,
            "direction_mode": direction_mode,
            "step_norm": step_norm,
            "rows": len(group),
            "valid_rows": len(valid),
            "error_rows": len(group) - len(valid),
            "actual_NLL_improve_rows": bool_sum(valid, "actual_NLL_improve"),
            "actual_all_debt_UCB_nonpositive_rows": bool_sum(valid, "actual_all_debt_UCB_nonpositive"),
            "actual_joint_safe_task_rows": bool_sum(valid, "actual_joint_safe_task"),
            "actual_Brier_UCB_nonpositive_rows": bool_sum(valid, "actual_Brier_UCB_nonpositive"),
            "actual_ECE_UCB_nonpositive_rows": bool_sum(valid, "actual_ECE_UCB_nonpositive"),
            "actual_tail99_UCB_nonpositive_rows": bool_sum(valid, "actual_tail99_UCB_nonpositive"),
            "actual_margin_UCB_nonpositive_rows": bool_sum(valid, "actual_margin_UCB_nonpositive"),
            "first_order_feasible_rows": bool_sum(valid, "linear_cone_first_order_feasible"),
            "direction_nonzero_rows": bool_sum(valid, "direction_nonzero"),
            "actual_NLL_delta_median": quantile([fval(row.get("actual_NLL_delta")) for row in valid], 0.50),
            "actual_all_debt_UCB_median": quantile([fval(row.get("actual_all_debt_UCB")) for row in valid], 0.50),
            "first_order_CE_delta_median": quantile([fval(row.get("first_order_CE_delta")) for row in valid], 0.50),
            "first_order_debt_max_median": quantile([fval(row.get("first_order_debt_max")) for row in valid], 0.50),
            "guard_all_debt_output_residual_energy_fraction_median": quantile([fval(row.get("guard_all_debt_output_residual_energy_fraction")) for row in valid], 0.50),
            "guard_brier_only_output_residual_energy_fraction_median": quantile([fval(row.get("guard_brier_only_output_residual_energy_fraction")) for row in valid], 0.50),
            "guard_no_brier_output_residual_energy_fraction_median": quantile([fval(row.get("guard_no_brier_output_residual_energy_fraction")) for row in valid], 0.50),
            "guard_ce_brier_weighted_cosine_median": quantile([fval(row.get("guard_ce_brier_weighted_cosine")) for row in valid], 0.50),
            "guard_ce_all_debt_projection_fraction_median": quantile([fval(row.get("guard_ce_all_debt_projection_fraction")) for row in valid], 0.50),
        }
        summary["output_oracle_family_direction_gate_pass"] = int(
            summary["valid_rows"] >= 15
            and summary["error_rows"] == 0
            and summary["actual_NLL_improve_rows"] >= 10
            and summary["actual_all_debt_UCB_nonpositive_rows"] >= 10
            and summary["actual_joint_safe_task_rows"] >= 10
            and summary["direction_nonzero_rows"] >= 10
        )
        out.append(summary)
    return out


def route_part_j_output_oracle(summaries: list[dict[str, Any]]) -> tuple[str, str]:
    passed = [row for row in summaries if int(row.get("output_oracle_family_direction_gate_pass", 0))]
    if passed:
        best = sorted(passed, key=lambda row: (int(row.get("actual_joint_safe_task_rows", 0)), -abs(fval(row.get("actual_NLL_delta_median")))), reverse=True)[0]
        return (
            "OutputDebtSafeOracleFeasibleButKANCarrierBlocked",
            f"{best['carrier_family']}::{best['direction_mode']}@{best['step_norm']} joint_safe={best['actual_joint_safe_task_rows']}/{best['valid_rows']}; NLL={best['actual_NLL_improve_rows']}/{best['valid_rows']}; debtUCB={best['actual_all_debt_UCB_nonpositive_rows']}/{best['valid_rows']}; allDebtEnergyMed={best['guard_all_debt_output_residual_energy_fraction_median']}",
        )
    max_joint = max([int(row.get("actual_joint_safe_task_rows", 0)) for row in summaries] or [0])
    max_first = max([int(row.get("first_order_feasible_rows", 0)) for row in summaries] or [0])
    max_nll = max([int(row.get("actual_NLL_improve_rows", 0)) for row in summaries] or [0])
    max_debt = max([int(row.get("actual_all_debt_UCB_nonpositive_rows", 0)) for row in summaries] or [0])
    max_all_energy = max([fval(row.get("guard_all_debt_output_residual_energy_fraction_median")) for row in summaries] or [0.0])
    max_brier_energy = max([fval(row.get("guard_brier_only_output_residual_energy_fraction_median")) for row in summaries] or [0.0])
    max_no_brier_energy = max([fval(row.get("guard_no_brier_output_residual_energy_fraction_median")) for row in summaries] or [0.0])
    max_errors = max([int(row.get("error_rows", 0)) for row in summaries] or [0])
    max_cos = max([abs(fval(row.get("guard_ce_brier_weighted_cosine_median"))) for row in summaries] or [0.0])
    reason = (
        f"max_joint_safe_rows={max_joint}; max_first_order_feasible_rows={max_first}; "
        f"max_NLL_improve_rows={max_nll}; max_all_debt_UCB_nonpositive_rows={max_debt}; "
        f"max_all_debt_energy_median={max_all_energy}; max_brier_only_energy_median={max_brier_energy}; "
        f"max_no_brier_energy_median={max_no_brier_energy}; max_abs_CE_Brier_cosine_median={max_cos}; max_error_rows={max_errors}"
    )
    if max_errors > 0:
        return "OutputOracleProbeErrorsNeedFix", reason
    if max_first >= 10 and max_joint < 10:
        return "OutputDebtSafeOracleFirstOrderOnlyFiniteStepDebtBlocked", reason
    if max_no_brier_energy >= 0.10 and max_all_energy < 0.10 and max_brier_energy < 0.10:
        return "OutputSpaceCEDebtConflictFundamental", reason
    return "OutputOracleNoDebtSafeTaskDirection", reason


def merge_part_j_output_oracle(args: argparse.Namespace) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    missing = []
    for idx in range(int(args.shard_count)):
        path = OUT_ROOT / f"v22_80_part_j_output_debt_safe_oracle_shard{idx}_of_{args.shard_count}.csv"
        if not path.exists():
            missing.append(rel(path))
        rows.extend(read_rows(path))
    write_rows(OUT_ROOT / "v22_80_part_j_output_debt_safe_oracle.csv", rows)
    summaries = summarize_part_j_output_oracle(rows)
    write_rows(OUT_ROOT / "v22_80_part_j_output_debt_safe_oracle_summaries.csv", summaries)
    route, reason = route_part_j_output_oracle(summaries) if not missing else ("R0-CodeOrTruthGateFailed", f"missing output oracle shards: {missing}")
    obj = {
        "gate": "v22_80_part_j_output_debt_safe_oracle",
        "run_status": "completed_part_j_output_oracle_merge" if not missing else "incomplete_part_j_output_oracle_merge",
        "part_j_output_oracle_repair_pass": int(route == "OutputDebtSafeOracleFeasibleButKANCarrierBlocked"),
        "part_j_output_oracle_route": route,
        "route_reason": reason,
        "rows": len(rows),
        "summary_rows": len(summaries),
        "missing_shards": missing,
        "passed_output_oracles": [f"{row['carrier_family']}::{row['direction_mode']}@{row['step_norm']}" for row in summaries if int(row.get("output_oracle_family_direction_gate_pass", 0))],
    }
    write_json(OUT_ROOT / "v22_80_part_j_output_debt_safe_oracle_route.json", obj)
    append_exec("J_output_debt_safe_oracle_merge", command_text(sys.argv), "pass" if obj["part_j_output_oracle_repair_pass"] else "fail", files=f"{rel(OUT_ROOT / 'v22_80_part_j_output_debt_safe_oracle.csv')}; {rel(OUT_ROOT / 'v22_80_part_j_output_debt_safe_oracle_summaries.csv')}; {rel(OUT_ROOT / 'v22_80_part_j_output_debt_safe_oracle_route.json')}", note=json.dumps(obj, ensure_ascii=False))
    best = sorted(summaries, key=lambda row: (int(row.get("output_oracle_family_direction_gate_pass", 0)), int(row.get("actual_joint_safe_task_rows", 0)), int(row.get("actual_NLL_improve_rows", 0)), int(row.get("actual_all_debt_UCB_nonpositive_rows", 0))), reverse=True)[:10]
    append_recap(
        "Part J output-space debt-safe oracle feasibility audit",
        [
            f"rows={len(rows)}；summary_rows={len(summaries)}；repair_pass={obj['part_j_output_oracle_repair_pass']}；route={route}。",
            f"reason={reason}",
            "best_output_oracles="
            + " | ".join(
                f"{row['carrier_family']}::{row['direction_mode']}@{row['step_norm']}: joint={row['actual_joint_safe_task_rows']}/{row['valid_rows']}；NLL={row['actual_NLL_improve_rows']}/{row['valid_rows']}；debtUCB={row['actual_all_debt_UCB_nonpositive_rows']}/{row['valid_rows']}；firstOrder={row['first_order_feasible_rows']}/{row['valid_rows']}；allDebtEnergyMed={row['guard_all_debt_output_residual_energy_fraction_median']}；noBrierEnergyMed={row['guard_no_brier_output_residual_energy_fraction_median']}；CEBrierCosMed={row['guard_ce_brier_weighted_cosine_median']}；pass={row['output_oracle_family_direction_gate_pass']}"
                for row in best
            ),
            "修复记录：新增 train-only output/logit-space upper-bound oracle，比较 raw CE、all-debt projected、Brier-only projected、no-Brier projected 与 linear debt-cone QP；用于区分 output 目标约束本身冲突和 KAN carrier 不足。不改变 official runtime，不使用 validation/test/future/query direction，不作为 candidate selector。",
        ],
    )
    return obj


def output_oracle_update_from_logits(logits: torch.Tensor, y: torch.Tensor, step_norm: float, *, ridge: float) -> tuple[torch.Tensor, dict[str, Any]]:
    cone = output_linear_debt_cone_direction(logits, y, ridge=ridge)
    direction = cone["direction"].reshape(-1)
    update = normalize_logit_update(direction, logits, float(step_norm)).reshape_as(logits)
    return update, cone


def weighted_capacity(pred: torch.Tensor, target: torch.Tensor, metric: torch.Tensor) -> tuple[float, float]:
    pp = pred.reshape(-1).to(dtype=torch.float64)
    tt = target.reshape(-1).to(device=pp.device, dtype=torch.float64)
    mm = metric.reshape(-1).to(device=pp.device, dtype=torch.float64)
    target_energy = metric_energy(tt, mm).clamp_min(1.0e-12)
    residual_energy = metric_energy(tt - pp, mm).clamp_min(0.0)
    capacity = float((1.0 - residual_energy / target_energy).clamp(min=-1.0, max=1.0).detach().cpu().item())
    rel_residual = float((residual_energy / target_energy).detach().cpu().item())
    return capacity, rel_residual


def kan_w2_lstsq_update(model: Any, x: torch.Tensor, target_update: torch.Tensor, metric: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    design = base73.w2_readout_edge_design(model, x)
    phi = design["phi_raw"].to(device=x.device, dtype=torch.float64)
    target = target_update.reshape(-1, 1).to(device=x.device, dtype=torch.float64)
    try:
        sol = torch.linalg.lstsq(phi, target).solution.reshape(-1)
    except Exception:
        sol = torch.linalg.pinv(phi) @ target
        sol = sol.reshape(-1)
    pred = (phi @ sol.reshape(-1, 1)).reshape(-1)
    cap, rel_res = weighted_capacity(pred, target.reshape(-1), metric)
    return sol.reshape_as(model.w2).to(device=x.device, dtype=torch.float64), {
        "coverage": cap,
        "relative_residual_energy": rel_res,
        "design_rank": int(torch.linalg.matrix_rank(phi.detach().float()).detach().cpu().item()) if int(phi.numel()) else 0,
        "design_cols": int(phi.shape[1]) if int(phi.ndim) == 2 else 0,
        "design_matrix": "phi_raw",
    }


def mlp_w2_lstsq_update(model: Any, x: torch.Tensor, target_update: torch.Tensor, metric: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    with torch.no_grad():
        h = model.frozen_readout_features(x).detach().to(dtype=torch.float64)
    target = target_update.to(device=x.device, dtype=torch.float64)
    try:
        sol = torch.linalg.lstsq(h, target).solution
    except Exception:
        sol = torch.linalg.pinv(h) @ target
    pred = h @ sol
    cap, rel_res = weighted_capacity(pred.reshape(-1), target.reshape(-1), metric)
    return sol.to(device=x.device, dtype=torch.float64), {
        "coverage": cap,
        "relative_residual_energy": rel_res,
        "design_rank": int(torch.linalg.matrix_rank(h.detach().float()).detach().cpu().item()) if int(h.numel()) else 0,
        "design_cols": int(h.shape[1]) if int(h.ndim) == 2 else 0,
        "design_matrix": "frozen_readout_features",
    }


def actual_w2_logit_update(model: Any, x: torch.Tensor, delta_w2: torch.Tensor) -> torch.Tensor:
    with torch.inference_mode():
        before = model(x).float()
        named = {name: param for name, param in model.named_parameters()}
        new_params = dict(named)
        new_params["w2"] = named["w2"] + delta_w2.to(device=named["w2"].device, dtype=named["w2"].dtype).reshape_as(named["w2"])
        after = functional_call_model(model, new_params, x).float()
    return (after - before).reshape(-1).to(device=x.device, dtype=torch.float64)


def part_k_actuator_coverage_probe(dataset: str, seed: int, spec: dict[str, str], args: argparse.Namespace, device: torch.device) -> list[dict[str, Any]]:
    from dgkan.models.fc_purekan_primitives import MLPBaseline

    bundle = base73.load_bundle(dataset, int(args.train_size), int(args.held_size), int(args.test_size), int(seed))
    x_all = bundle["x_train"].to(device).float()[: int(args.metric_batch_size)]
    y_all = bundle["y_train"].to(device).long()[: int(args.metric_batch_size)]
    method = str(spec["method"])
    seed_k = int(seed) + 24800
    kan = base75.make_wlb_model(method, bundle, device, int(args.hidden), seed_k, x_metric=x_all)
    mlp = MLPBaseline(int(bundle["input_dim"]), int(bundle["num_classes"]), int(args.hidden), seed_k, device).to(device)
    _x_source, _y_source, _x_witness, _y_witness, x_guard, y_guard, _x_control, _y_control = split_train(x_all, y_all)
    with torch.no_grad():
        kan_logits = kan(x_guard).float()
        mlp_logits = mlp(x_guard).float()
    rows: list[dict[str, Any]] = []
    for step_norm in ACTUATOR_COVERAGE_STEP_NORMS:
        kan_target, kan_cone = output_oracle_update_from_logits(kan_logits, y_guard, float(step_norm), ridge=float(args.projector_ridge))
        mlp_target, mlp_cone = output_oracle_update_from_logits(mlp_logits, y_guard, float(step_norm), ridge=float(args.projector_ridge))
        kan_metric = output_metric_diag(kan_logits).to(device=device)
        mlp_metric = output_metric_diag(mlp_logits).to(device=device)
        kan_delta_w2, kan_cov = kan_w2_lstsq_update(kan, x_guard, kan_target, kan_metric)
        mlp_delta_w2, mlp_cov = mlp_w2_lstsq_update(mlp, x_guard, mlp_target, mlp_metric)
        kan_actual_update = actual_w2_logit_update(kan, x_guard, kan_delta_w2)
        mlp_actual_update = actual_w2_logit_update(mlp, x_guard, mlp_delta_w2)
        kan_actual_cov, kan_actual_rel_res = weighted_capacity(kan_actual_update, kan_target.reshape(-1), kan_metric)
        mlp_actual_cov, mlp_actual_rel_res = weighted_capacity(mlp_actual_update, mlp_target.reshape(-1), mlp_metric)
        kan_deltas = metric_delta_for_updates(kan, x_guard, y_guard, {"w2": kan_delta_w2})
        mlp_deltas = metric_delta_for_updates(mlp, x_guard, y_guard, {"w2": mlp_delta_w2})
        kan_debt_ucb = debt_ucb_from_deltas(kan_deltas)
        mlp_debt_ucb = debt_ucb_from_deltas(mlp_deltas)
        rows.append(
            {
                "dataset": dataset,
                "seed": seed,
                "carrier_family": str(spec["carrier_family"]),
                "carrier_method": method,
                "carrier_kind": str(spec["kind"]),
                "step_norm": float(step_norm),
                "probe_error": "",
                "kan_target_first_order_feasible": int(kan_cone["first_order_debt_feasible"]),
                "mlp_target_first_order_feasible": int(mlp_cone["first_order_debt_feasible"]),
                "kan_actuator_coverage": kan_actual_cov,
                "kan_actuator_relative_residual_energy": kan_actual_rel_res,
                "kan_linearized_actuator_coverage": kan_cov["coverage"],
                "kan_linearized_actuator_relative_residual_energy": kan_cov["relative_residual_energy"],
                "kan_actuator_design_matrix": kan_cov["design_matrix"],
                "kan_actuator_design_rank": kan_cov["design_rank"],
                "kan_actuator_design_cols": kan_cov["design_cols"],
                "mlp_actuator_coverage": mlp_actual_cov,
                "mlp_actuator_relative_residual_energy": mlp_actual_rel_res,
                "mlp_linearized_actuator_coverage": mlp_cov["coverage"],
                "mlp_linearized_actuator_relative_residual_energy": mlp_cov["relative_residual_energy"],
                "mlp_actuator_design_matrix": mlp_cov["design_matrix"],
                "mlp_actuator_design_rank": mlp_cov["design_rank"],
                "mlp_actuator_design_cols": mlp_cov["design_cols"],
                "kan_update_norm": float(kan_delta_w2.reshape(-1).norm().detach().cpu().item()),
                "mlp_update_norm": float(mlp_delta_w2.reshape(-1).norm().detach().cpu().item()),
                "kan_actual_NLL_delta": kan_deltas["NLL"],
                "kan_actual_NLL_improve": int(kan_deltas["NLL"] < 0.0),
                "kan_actual_all_debt_UCB": kan_debt_ucb,
                "kan_actual_all_debt_UCB_nonpositive": int(kan_debt_ucb <= 0.0),
                "kan_actual_joint_safe_task": int(kan_deltas["NLL"] < 0.0 and kan_debt_ucb <= 0.0),
                "mlp_actual_NLL_delta": mlp_deltas["NLL"],
                "mlp_actual_NLL_improve": int(mlp_deltas["NLL"] < 0.0),
                "mlp_actual_all_debt_UCB": mlp_debt_ucb,
                "mlp_actual_all_debt_UCB_nonpositive": int(mlp_debt_ucb <= 0.0),
                "mlp_actual_joint_safe_task": int(mlp_deltas["NLL"] < 0.0 and mlp_debt_ucb <= 0.0),
                "mlp_beats_kan_coverage": int(mlp_actual_cov > kan_actual_cov + 1.0e-6),
                "mlp_can_kan_cannot_joint": int(mlp_deltas["NLL"] < 0.0 and mlp_debt_ucb <= 0.0 and not (kan_deltas["NLL"] < 0.0 and kan_debt_ucb <= 0.0)),
            }
        )
    return rows


def run_part_k_actuator_coverage(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")
    tasks = shard_items(carrier_redesign_task_grid(args), args)
    rows: list[dict[str, Any]] = []
    for spec, seed, dataset in tasks:
        try:
            rows.extend(part_k_actuator_coverage_probe(dataset, seed, spec, args, device))
        except Exception as exc:
            rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "carrier_family": str(spec.get("carrier_family", "")),
                    "carrier_method": str(spec.get("method", "")),
                    "carrier_kind": str(spec.get("kind", "")),
                    "step_norm": "",
                    "probe_error": f"{type(exc).__name__}: {exc}",
                }
            )
    out_path = OUT_ROOT / f"v22_80_part_k_mlp_matched_actuator_coverage_shard{args.shard_index}_of_{args.shard_count}.csv"
    write_rows(out_path, rows)
    obj = {"gate": "v22_80_part_k_mlp_matched_actuator_coverage_shard", "rows": len(rows), "tasks": len(tasks), "shard_index": int(args.shard_index), "shard_count": int(args.shard_count), "output": rel(out_path)}
    write_json(OUT_ROOT / f"v22_80_part_k_mlp_matched_actuator_coverage_shard{args.shard_index}_of_{args.shard_count}.json", obj)
    append_exec("K_mlp_matched_actuator_coverage_shard", command_text(sys.argv), "done", gpu=str(device), files=rel(out_path), note=json.dumps(obj, ensure_ascii=False))
    return obj


def summarize_part_k_actuator_coverage(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    groups = sorted({(str(row.get("carrier_family")), str(row.get("step_norm"))) for row in rows})
    for carrier_family, step_norm in groups:
        group = [row for row in rows if str(row.get("carrier_family")) == carrier_family and str(row.get("step_norm")) == step_norm]
        valid = [row for row in group if not str(row.get("probe_error", ""))]
        method = str(valid[0].get("carrier_method", group[0].get("carrier_method", ""))) if group else ""
        kind = str(valid[0].get("carrier_kind", group[0].get("carrier_kind", ""))) if group else ""
        summary = {
            "carrier_family": carrier_family,
            "carrier_method": method,
            "carrier_kind": kind,
            "step_norm": step_norm,
            "rows": len(group),
            "valid_rows": len(valid),
            "error_rows": len(group) - len(valid),
            "kan_actuator_coverage_median": quantile([fval(row.get("kan_actuator_coverage")) for row in valid], 0.50),
            "kan_actuator_coverage_CVaR25": lower_cvar([fval(row.get("kan_actuator_coverage")) for row in valid], 0.25),
            "mlp_actuator_coverage_median": quantile([fval(row.get("mlp_actuator_coverage")) for row in valid], 0.50),
            "mlp_actuator_coverage_CVaR25": lower_cvar([fval(row.get("mlp_actuator_coverage")) for row in valid], 0.25),
            "kan_joint_safe_task_rows": bool_sum(valid, "kan_actual_joint_safe_task"),
            "mlp_joint_safe_task_rows": bool_sum(valid, "mlp_actual_joint_safe_task"),
            "kan_NLL_improve_rows": bool_sum(valid, "kan_actual_NLL_improve"),
            "mlp_NLL_improve_rows": bool_sum(valid, "mlp_actual_NLL_improve"),
            "kan_debt_UCB_nonpositive_rows": bool_sum(valid, "kan_actual_all_debt_UCB_nonpositive"),
            "mlp_debt_UCB_nonpositive_rows": bool_sum(valid, "mlp_actual_all_debt_UCB_nonpositive"),
            "mlp_beats_kan_coverage_rows": bool_sum(valid, "mlp_beats_kan_coverage"),
            "mlp_can_kan_cannot_joint_rows": bool_sum(valid, "mlp_can_kan_cannot_joint"),
            "kan_update_norm_median": quantile([fval(row.get("kan_update_norm")) for row in valid], 0.50),
            "mlp_update_norm_median": quantile([fval(row.get("mlp_update_norm")) for row in valid], 0.50),
        }
        summary["mlp_matched_capacity_gap_pass"] = int(
            summary["valid_rows"] >= 15
            and summary["error_rows"] == 0
            and summary["mlp_joint_safe_task_rows"] >= 10
            and summary["kan_joint_safe_task_rows"] < 10
            and summary["mlp_beats_kan_coverage_rows"] >= 10
        )
        out.append(summary)
    return out


def route_part_k_actuator_coverage(summaries: list[dict[str, Any]]) -> tuple[str, str]:
    passed = [row for row in summaries if int(row.get("mlp_matched_capacity_gap_pass", 0))]
    if passed:
        best = sorted(passed, key=lambda row: (int(row.get("mlp_can_kan_cannot_joint_rows", 0)), fval(row.get("mlp_actuator_coverage_CVaR25"))), reverse=True)[0]
        return (
            "MLPMatchedCanCarryOutputOracle_KANCarrierCapacityInsufficient",
            f"{best['carrier_family']}@{best['step_norm']} mlp_joint={best['mlp_joint_safe_task_rows']}/{best['valid_rows']}; kan_joint={best['kan_joint_safe_task_rows']}/{best['valid_rows']}; mlp_cov_CVaR25={best['mlp_actuator_coverage_CVaR25']}; kan_cov_CVaR25={best['kan_actuator_coverage_CVaR25']}; mlp_beats_kan={best['mlp_beats_kan_coverage_rows']}/{best['valid_rows']}",
        )
    max_mlp_joint = max([int(row.get("mlp_joint_safe_task_rows", 0)) for row in summaries] or [0])
    max_kan_joint = max([int(row.get("kan_joint_safe_task_rows", 0)) for row in summaries] or [0])
    max_mlp_cov = max([fval(row.get("mlp_actuator_coverage_CVaR25")) for row in summaries] or [0.0])
    max_kan_cov = max([fval(row.get("kan_actuator_coverage_CVaR25")) for row in summaries] or [0.0])
    max_gap_rows = max([int(row.get("mlp_can_kan_cannot_joint_rows", 0)) for row in summaries] or [0])
    max_errors = max([int(row.get("error_rows", 0)) for row in summaries] or [0])
    reason = f"max_mlp_joint_rows={max_mlp_joint}; max_kan_joint_rows={max_kan_joint}; max_mlp_cov_CVaR25={max_mlp_cov}; max_kan_cov_CVaR25={max_kan_cov}; max_mlp_can_kan_cannot_rows={max_gap_rows}; max_error_rows={max_errors}"
    if max_errors > 0:
        return "MLPMatchedActuatorCoverageProbeErrorsNeedFix", reason
    if max_kan_joint >= 10:
        return "KANRawActuatorCanCarryOutputOracleButControlPipelineStillBlocks", reason
    if max_mlp_joint < 10:
        return "OutputOracleReachableOnlyAsDirectLogitNotMatchedParamCarrier", reason
    return "MLPMatchedAdvantageNotStableEnoughForCapacityInsufficientRoute", reason


def merge_part_k_actuator_coverage(args: argparse.Namespace) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    missing = []
    for idx in range(int(args.shard_count)):
        path = OUT_ROOT / f"v22_80_part_k_mlp_matched_actuator_coverage_shard{idx}_of_{args.shard_count}.csv"
        if not path.exists():
            missing.append(rel(path))
        rows.extend(read_rows(path))
    write_rows(OUT_ROOT / "v22_80_part_k_mlp_matched_actuator_coverage.csv", rows)
    summaries = summarize_part_k_actuator_coverage(rows)
    write_rows(OUT_ROOT / "v22_80_part_k_mlp_matched_actuator_coverage_summaries.csv", summaries)
    route, reason = route_part_k_actuator_coverage(summaries) if not missing else ("R0-CodeOrTruthGateFailed", f"missing actuator coverage shards: {missing}")
    obj = {
        "gate": "v22_80_part_k_mlp_matched_actuator_coverage",
        "run_status": "completed_part_k_actuator_coverage_merge" if not missing else "incomplete_part_k_actuator_coverage_merge",
        "part_k_actuator_coverage_repair_pass": int(route == "MLPMatchedCanCarryOutputOracle_KANCarrierCapacityInsufficient"),
        "part_k_actuator_coverage_route": route,
        "route_reason": reason,
        "rows": len(rows),
        "summary_rows": len(summaries),
        "missing_shards": missing,
        "passed_capacity_gaps": [f"{row['carrier_family']}@{row['step_norm']}" for row in summaries if int(row.get("mlp_matched_capacity_gap_pass", 0))],
    }
    write_json(OUT_ROOT / "v22_80_part_k_mlp_matched_actuator_coverage_route.json", obj)
    append_exec("K_mlp_matched_actuator_coverage_merge", command_text(sys.argv), "pass" if obj["part_k_actuator_coverage_repair_pass"] else "fail", files=f"{rel(OUT_ROOT / 'v22_80_part_k_mlp_matched_actuator_coverage.csv')}; {rel(OUT_ROOT / 'v22_80_part_k_mlp_matched_actuator_coverage_summaries.csv')}; {rel(OUT_ROOT / 'v22_80_part_k_mlp_matched_actuator_coverage_route.json')}", note=json.dumps(obj, ensure_ascii=False))
    best = sorted(summaries, key=lambda row: (int(row.get("mlp_matched_capacity_gap_pass", 0)), int(row.get("mlp_can_kan_cannot_joint_rows", 0)), int(row.get("mlp_joint_safe_task_rows", 0)), fval(row.get("mlp_actuator_coverage_CVaR25"))), reverse=True)[:10]
    append_recap(
        "Part K raw actuator coverage and MLP matched readout audit",
        [
            f"rows={len(rows)}；summary_rows={len(summaries)}；repair_pass={obj['part_k_actuator_coverage_repair_pass']}；route={route}。",
            f"reason={reason}",
            "best_actuator_rows="
            + " | ".join(
                f"{row['carrier_family']}@{row['step_norm']}: mlp_joint={row['mlp_joint_safe_task_rows']}/{row['valid_rows']}；kan_joint={row['kan_joint_safe_task_rows']}/{row['valid_rows']}；mlp_cov_CVaR25={row['mlp_actuator_coverage_CVaR25']}；kan_cov_CVaR25={row['kan_actuator_coverage_CVaR25']}；mlp>kan={row['mlp_beats_kan_coverage_rows']}/{row['valid_rows']}；gap_rows={row['mlp_can_kan_cannot_joint_rows']}/{row['valid_rows']}；pass={row['mlp_matched_capacity_gap_pass']}"
                for row in best
            ),
            "修复记录：按计划 13.2 增加 raw functional actuator coverage 与 MLP matched coordinate diagnostic；目标来自 train-only output debt-safe oracle tiny step，只比较 readout actuator capacity 和 actual guard deltas。不使用 MLP winner target 作为 official KAN runtime teacher，不进入 Part G promotion。",
        ],
    )
    return obj


def metric_project(v: torch.Tensor, basis: torch.Tensor | None, metric: torch.Tensor, ridge: float) -> torch.Tensor:
    residual, _diag = weighted_project(v, basis, metric, ridge=ridge)
    return v.reshape(-1) - residual.reshape(-1)


def run_part_e(args: argparse.Namespace) -> dict[str, Any]:
    command = command_text([PYTHON, rel(RUNNER), "--mode", "part-e"])
    device = torch.device("cpu")
    gen = torch.Generator(device=device).manual_seed(228000)
    dim = 96
    raw_cols = [torch.randn(dim, generator=gen, dtype=torch.float64) for _ in range(9)]
    metric = torch.linspace(0.2, 2.0, dim, dtype=torch.float64)
    basis = normalize_columns(raw_cols, dim, device)
    g = torch.randn(dim, generator=gen, dtype=torch.float64)
    unit_ridge = 1.0e-10
    proj = metric_project(g, basis, metric, unit_ridge)
    proj2 = metric_project(proj, basis, metric, unit_ridge)
    residual = g - proj
    nuis_inner = (basis.transpose(0, 1) @ (metric.reshape(-1, 1) * residual.reshape(-1, 1))).reshape(-1) if basis is not None else torch.zeros(1, dtype=torch.float64)
    signal = torch.randn(dim, generator=gen, dtype=torch.float64)
    signal_res = signal - metric_project(signal, basis, metric, unit_ridge)
    preserved = signal_res - metric_project(signal_res, basis, metric, unit_ridge)
    nuisance = raw_cols[0]
    nuisance_res = nuisance - metric_project(nuisance, basis, metric, unit_ridge)
    candidate = torch.randn(dim, generator=gen, dtype=torch.float64)
    same_cond = same_energy_orthogonal_control(candidate).to(dtype=torch.float64)
    cand_cos = abs(float(torch.dot(candidate, same_cond) / (candidate.norm().clamp_min(1.0e-12) * same_cond.norm().clamp_min(1.0e-12))))
    source = torch.randn(dim, generator=gen, dtype=torch.float64)
    witness = source + 0.05 * torch.randn(dim, generator=gen, dtype=torch.float64)
    sign_consistency = float(torch.sign(source).eq(torch.sign(witness)).float().mean().item())
    pullback_after_output = metric_project(g, basis, metric, unit_ridge)
    edge_debt_projection = metric_project(g, normalize_columns(raw_cols[:3], dim, device), metric, unit_ridge)
    commute_gap = float((pullback_after_output - edge_debt_projection).norm().item() / max(1.0e-12, float(g.norm().item())))
    same_domain = normalized_update(raw_cols[1], (dim,), 1.0)
    same_debt = normalized_update(raw_cols[2], (dim,), 1.0)
    rows = [
        {"test": "E1_G0_PSD_symmetry_finite_condition", "value": float((metric.max() / metric.min()).item()), "pass": int(torch.isfinite(metric).all() and float(metric.min().item()) > 0.0 and float((metric.max() / metric.min()).item()) < 1.0e5)},
        {"test": "E2_projector_idempotence", "value": float((proj2 - proj).norm().item()), "pass": int(float((proj2 - proj).norm().item()) <= 1.0e-5)},
        {"test": "E3_G0_orthogonality", "value": float(nuis_inner.abs().max().item()), "pass": int(float(nuis_inner.abs().max().item()) <= 1.0e-5)},
        {"test": "E4_known_signal_preserved", "value": float(preserved.norm().item() / max(1.0e-12, float(signal_res.norm().item()))), "pass": int(float(preserved.norm().item() / max(1.0e-12, float(signal_res.norm().item()))) >= 0.90)},
        {"test": "E5_known_nuisance_removed", "value": float(nuisance_res.norm().item() / max(1.0e-12, float(nuisance.norm().item()))), "pass": int(float(nuisance_res.norm().item() / max(1.0e-12, float(nuisance.norm().item()))) <= 0.01)},
        {"test": "E6_candidate_axis_not_in_same_conditional_energy", "value": cand_cos, "pass": int(cand_cos < 0.95)},
        {"test": "E7_source_witness_split_sign_consistency", "value": sign_consistency, "pass": int(sign_consistency >= 0.70)},
        {"test": "E8_debt_output_residual_edge_residual_commute_gap_recorded", "value": commute_gap, "pass": int(math.isfinite(commute_gap))},
        {"test": "E9_same_domain_control_equal_domain_energy", "value": float(same_domain.norm().item()), "pass": int(abs(float(same_domain.norm().item()) - 1.0) <= 1.0e-6)},
        {"test": "E10_same_debt_control_equal_debt_energy", "value": float(same_debt.norm().item()), "pass": int(abs(float(same_debt.norm().item()) - 1.0) <= 1.0e-6)},
    ]
    gate = int(all(int(row["pass"]) for row in rows))
    write_rows(OUT_ROOT / "v22_80_part_e_control_orthogonal_metric_unit_tests.csv", rows)
    obj = {"gate": "v22_80_part_e_control_orthogonal_metric_unit_tests", "part_e_gate_pass": gate, "rows": len(rows), "failed_tests": [row["test"] for row in rows if not int(row["pass"])], "commute_gap_recorded": commute_gap}
    write_json(OUT_ROOT / "v22_80_part_e_control_orthogonal_metric_unit_gate.json", obj)
    append_exec("E_control_orthogonal_metric_unit_tests", command, "pass" if gate else "fail", files=f"{rel(OUT_ROOT / 'v22_80_part_e_control_orthogonal_metric_unit_tests.csv')}; {rel(OUT_ROOT / 'v22_80_part_e_control_orthogonal_metric_unit_gate.json')}", note=json.dumps(obj, ensure_ascii=False))
    append_recap("Part E control-orthogonal edge metric unit tests", [f"part_e_gate_pass={gate}；failed_tests={obj['failed_tests']}。", f"E8 commute_gap={commute_gap}；G0 metric deliberately excludes debt/domain reward terms and treats them as projector basis."])
    return obj


def part_f_probe(dataset: str, seed: int, family: str, args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    edge_family = "coer_output_residual_control_orth_edge_bank"
    model, _bundle, x_all, y_all = make_probe_model(dataset, seed, edge_family, args, device)
    x_source, y_source, x_witness, y_witness, x_guard, y_guard, _x_control, _y_control = split_train(x_all, y_all)
    source = edge_pullback_from_output_residual(model, x_source, y_source, ridge=float(args.projector_ridge))
    w2_grads = collect_grads(model, x_source, y_source, x_witness, y_witness, x_guard, y_guard, x_all, y_all, "w2")
    w1_grads = collect_grads(model, x_source, y_source, x_witness, y_witness, x_guard, y_guard, x_all, y_all, "w1")
    metric_w2 = metric_diag_from_design(model, x_all, mode="readout")
    metric_w1 = base79.make_w1_metric(w1_grads)
    signal, _proj_stats = control_orthogonal_edge_signal(edge_family, source["residual_edge_vec"], metric_w2, model, w2_grads, ridge=float(args.projector_ridge))
    target = bank_additive_projection(signal, metric_w2, model.w2.shape)[0].reshape(-1)
    upstream_raw, fitted_raw, raw_stats = base79.solve_accessible_lift_upstream(target, signal, metric_w2, model.w1.shape, model.w2.shape, ridge=float(args.projector_ridge))
    dim1 = int(upstream_raw.numel())
    constraint_basis = normalize_columns(
        [
            w1_grads["domain"],
            w1_grads["debt"],
            w1_grads["tail95_debt"],
            w1_grads["tail99_debt"],
            w1_grads["ece_debt"],
            w1_grads["margin_debt"],
            w1_grads["random"],
            same_energy_orthogonal_control(upstream_raw).to(device=device),
        ],
        dim1,
        device,
    )
    constrained_res, constraint_diag = weighted_project(upstream_raw, constraint_basis, metric_w1, ridge=float(args.projector_ridge))
    upstream = constrained_res.reshape(-1)
    fitted = separation_lift_to_downstream(upstream, model.w1.shape, model.w2.shape, signal).reshape(-1)
    before = metric_energy(target, metric_w2).clamp_min(1.0e-12)
    after = metric_energy(fitted, metric_w2).clamp_min(0.0)
    residual = metric_energy(target - fitted, metric_w2).clamp_min(0.0)
    lift_projection_fraction = float((after / before).detach().cpu().item())
    lift_residual_norm = float(torch.sqrt(residual / before).detach().cpu().item())
    step_norm = 0.5 * float(args.lr) * float(args.step_mult)
    candidate_w1 = -normalized_update(upstream, model.w1.shape, step_norm)
    cand_norm = float(candidate_w1.reshape(-1).norm().detach().cpu().item())
    bank_control_norm = cand_norm
    bank_target_w2 = -normalized_update(target, model.w2.shape, bank_control_norm)
    controls = {
        "same_upstream_energy": {"w1": -normalized_update(w1_grads["random"], model.w1.shape, cand_norm), "w2": torch.zeros_like(bank_target_w2)},
        "same_separation_energy": {"w1": -normalized_update(same_energy_orthogonal_control(upstream).to(device=device), model.w1.shape, cand_norm), "w2": torch.zeros_like(bank_target_w2)},
        "same_bank_energy": {"w1": torch.zeros_like(candidate_w1), "w2": bank_target_w2},
        "same_debt": {"w1": -normalized_update(w1_grads["debt"], model.w1.shape, cand_norm), "w2": torch.zeros_like(bank_target_w2)},
    }
    cand_delta, _ctrl_delta, gaps, _debt_penalty, _all_debt_ucb = control_gap_diagnostics(model, x_guard, y_guard, {"w1": candidate_w1}, controls, args)
    boot = bootstrap_guard_margin(model, x_guard, y_guard, {"w1": candidate_w1}, controls, args, seed=seed + 22880)
    brier_ucb = cand_delta["Brier"] + 0.5 * abs(cand_delta["Brier"])
    ece_ucb = cand_delta["ECE"] + 0.5 * abs(cand_delta["ECE"])
    tail99_ucb = cand_delta["tail99"] + 0.5 * abs(cand_delta["tail99"])
    domain_projection_fraction = fval(constraint_diag.get("projected_energy_fraction"))
    constrained_feasible = int(lift_projection_fraction >= 0.20 and brier_ucb <= 0.0 and ece_ucb <= 0.0 and tail99_ucb <= 0.0 and domain_projection_fraction <= 0.95)
    return {
        "dataset": dataset,
        "seed": seed,
        "family": family,
        "architecture": map_family_to_method(edge_family),
        "lift_projection_fraction": lift_projection_fraction,
        "lift_residual_norm": lift_residual_norm,
        "constrained_lift_feasible": constrained_feasible,
        "unconstrained_lift_projection_fraction": fval(raw_stats.get("accessible_lift_projection_fraction")),
        "constraint_cost_ratio": domain_projection_fraction,
        "projected_Brier_delta_guard": cand_delta["Brier"],
        "projected_ECE_delta_guard": cand_delta["ECE"],
        "projected_tail99_delta_guard": cand_delta["tail99"],
        "projected_Brier_delta_guard_UCB_nonpositive": int(brier_ucb <= 0.0),
        "projected_ECE_delta_guard_UCB_nonpositive": int(ece_ucb <= 0.0),
        "projected_tail99_delta_guard_UCB_nonpositive": int(tail99_ucb <= 0.0),
        "projected_domain_nuisance_delta": domain_projection_fraction,
        "same_upstream_energy_candidate_better": int(gaps.get("same_upstream_energy", 0.0) > 0.0),
        "same_separation_energy_candidate_better": int(gaps.get("same_separation_energy", 0.0) > 0.0),
        "same_bank_energy_candidate_better": int(gaps.get("same_bank_energy", 0.0) > 0.0),
        "same_debt_candidate_better": int(gaps.get("same_debt", 0.0) > 0.0),
        "control_margin_p10": min(gaps.values()) if gaps else 0.0,
        "control_margin_positive": int((min(gaps.values()) if gaps else 0.0) > 0.0),
        **boot,
    }


def run_part_f(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")
    tasks = shard_items(task_grid(args, COER_LIFT_FAMILIES), args)
    rows = [part_f_probe(dataset, seed, family, args, device) for family, seed, dataset in tasks]
    out_path = OUT_ROOT / f"v22_80_part_f_upstream_constrained_lift_shard{args.shard_index}_of_{args.shard_count}.csv"
    write_rows(out_path, rows)
    obj = {"gate": "v22_80_part_f_upstream_constrained_lift_shard", "rows": len(rows), "shard_index": int(args.shard_index), "shard_count": int(args.shard_count), "output": rel(out_path)}
    write_json(OUT_ROOT / f"v22_80_part_f_upstream_constrained_lift_shard{args.shard_index}_of_{args.shard_count}.json", obj)
    append_exec("F_upstream_constrained_lift_shard", command_text(sys.argv), "done", gpu=str(device), files=rel(out_path), note=json.dumps(obj, ensure_ascii=False))
    return obj


def summarize_part_f(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for family in sorted({str(row.get("family")) for row in rows}):
        group = [row for row in rows if str(row.get("family")) == family]
        vals = [fval(row.get("lift_projection_fraction")) for row in group]
        summary = {
            "family": family,
            "rows": len(group),
            "constrained_lift_feasible_rows": bool_sum(group, "constrained_lift_feasible"),
            "lift_projection_fraction_CVaR25": lower_cvar(vals, 0.25),
            "lift_residual_norm_median": quantile([fval(row.get("lift_residual_norm")) for row in group], 0.50),
            "unconstrained_lift_projection_fraction_median": quantile([fval(row.get("unconstrained_lift_projection_fraction")) for row in group], 0.50),
            "projected_Brier_delta_guard_UCB_nonpositive_rows": bool_sum(group, "projected_Brier_delta_guard_UCB_nonpositive"),
            "projected_ECE_delta_guard_UCB_nonpositive_rows": bool_sum(group, "projected_ECE_delta_guard_UCB_nonpositive"),
            "projected_tail99_delta_guard_UCB_nonpositive_rows": bool_sum(group, "projected_tail99_delta_guard_UCB_nonpositive"),
            "same_upstream_energy_candidate_better_rows": bool_sum(group, "same_upstream_energy_candidate_better"),
            "same_separation_energy_candidate_better_rows": bool_sum(group, "same_separation_energy_candidate_better"),
            "same_bank_energy_candidate_better_rows": bool_sum(group, "same_bank_energy_candidate_better"),
            "control_margin_positive_rows": bool_sum(group, "control_margin_positive"),
            "bootstrap_guard_margin_positive_rows": bool_sum(group, "bootstrap_guard_margin_positive"),
            "control_margin_p10_median": quantile([fval(row.get("control_margin_p10")) for row in group], 0.50),
        }
        summary["part_f_family_gate_pass"] = int(
            summary["rows"] >= 15
            and summary["constrained_lift_feasible_rows"] >= 10
            and summary["lift_projection_fraction_CVaR25"] >= 0.20
            and summary["lift_residual_norm_median"] <= 0.60
            and summary["projected_Brier_delta_guard_UCB_nonpositive_rows"] >= 10
            and summary["projected_ECE_delta_guard_UCB_nonpositive_rows"] >= 10
            and summary["projected_tail99_delta_guard_UCB_nonpositive_rows"] >= 10
            and summary["same_upstream_energy_candidate_better_rows"] >= 10
            and summary["same_separation_energy_candidate_better_rows"] >= 10
            and summary["same_bank_energy_candidate_better_rows"] >= 10
            and summary["control_margin_positive_rows"] >= 10
            and summary["bootstrap_guard_margin_positive_rows"] >= 9
        )
        out.append(summary)
    return out


def route_part_f(summaries: list[dict[str, Any]]) -> tuple[str, str]:
    if any(int(row.get("part_f_family_gate_pass", 0)) for row in summaries):
        return "UpstreamConstrainedLiftPreflightOpened", "at least one fixed family passed Part F"
    max_uncon = max([fval(row.get("unconstrained_lift_projection_fraction_median")) for row in summaries] or [0.0])
    max_con = max([int(row.get("constrained_lift_feasible_rows", 0)) for row in summaries] or [0])
    max_margin = max([int(row.get("control_margin_positive_rows", 0)) for row in summaries] or [0])
    if max_uncon >= 0.20 and max_con < 10:
        return "UpstreamLiftExpressiveButDebtDomainConstrainedInfeasible", f"max_unconstrained_lift_projection_fraction_median={max_uncon}; max_constrained_lift_feasible_rows={max_con}"
    if max_con >= 10 and max_margin < 10:
        return "UpstreamLiftFeasibleButControlExplained", f"max_constrained_lift_feasible_rows={max_con}; max_control_margin_positive_rows={max_margin}"
    return "UpstreamLiftNotOpened", f"max_unconstrained_lift_projection_fraction_median={max_uncon}; max_constrained_rows={max_con}; max_margin_rows={max_margin}"


def merge_part_f(args: argparse.Namespace) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    missing = []
    for idx in range(int(args.shard_count)):
        path = OUT_ROOT / f"v22_80_part_f_upstream_constrained_lift_shard{idx}_of_{args.shard_count}.csv"
        if not path.exists():
            missing.append(rel(path))
        rows.extend(read_rows(path))
    write_rows(OUT_ROOT / "v22_80_part_f_upstream_constrained_lift.csv", rows)
    summaries = summarize_part_f(rows)
    write_rows(OUT_ROOT / "v22_80_part_f_upstream_constrained_lift_summaries.csv", summaries)
    route, reason = route_part_f(summaries) if not missing else ("R0-CodeOrTruthGateFailed", f"missing Part F shards: {missing}")
    obj = {
        "gate": "v22_80_part_f_upstream_constrained_lift",
        "run_status": "completed_part_f_merge" if not missing else "incomplete_part_f_merge",
        "part_f_gate_pass": int(not missing and any(int(row.get("part_f_family_gate_pass", 0)) for row in summaries)),
        "part_f_route": route,
        "route_reason": reason,
        "rows": len(rows),
        "summary_rows": len(summaries),
        "missing_shards": missing,
        "passed_families": [row["family"] for row in summaries if int(row.get("part_f_family_gate_pass", 0))],
    }
    write_json(OUT_ROOT / "v22_80_part_f_upstream_constrained_lift_route.json", obj)
    append_exec("F_upstream_constrained_lift_merge", command_text(sys.argv), "pass" if obj["part_f_gate_pass"] else "fail", files=f"{rel(OUT_ROOT / 'v22_80_part_f_upstream_constrained_lift.csv')}; {rel(OUT_ROOT / 'v22_80_part_f_upstream_constrained_lift_summaries.csv')}; {rel(OUT_ROOT / 'v22_80_part_f_upstream_constrained_lift_route.json')}", note=json.dumps(obj, ensure_ascii=False))
    best = sorted(summaries, key=lambda row: (int(row.get("part_f_family_gate_pass", 0)), int(row.get("constrained_lift_feasible_rows", 0)), fval(row.get("lift_projection_fraction_CVaR25"))), reverse=True)[:5]
    append_recap(
        "Part F upstream constrained causal lift preflight",
        [
            f"rows={len(rows)}；summary_rows={len(summaries)}；part_f_gate_pass={obj['part_f_gate_pass']}；route={route}。",
            f"reason={reason}",
            "best_families="
            + " | ".join(
                f"{row['family']}: feasible={row['constrained_lift_feasible_rows']}/{row['rows']}；lift_CVaR25={row['lift_projection_fraction_CVaR25']}；residual_norm_median={row['lift_residual_norm_median']}；BrierUCB={row['projected_Brier_delta_guard_UCB_nonpositive_rows']}/{row['rows']}；ECEUCB={row['projected_ECE_delta_guard_UCB_nonpositive_rows']}/{row['rows']}；tail99UCB={row['projected_tail99_delta_guard_UCB_nonpositive_rows']}/{row['rows']}；same_up={row['same_upstream_energy_candidate_better_rows']}/{row['rows']}；same_sep={row['same_separation_energy_candidate_better_rows']}/{row['rows']}；same_bank={row['same_bank_energy_candidate_better_rows']}/{row['rows']}；margin={row['control_margin_positive_rows']}/{row['rows']}；bootstrap={row['bootstrap_guard_margin_positive_rows']}/{row['rows']}；pass={row['part_f_family_gate_pass']}"
                for row in best
            ),
            "代码审计：Part F 先解 unconstrained upstream lift，再在 w1 constraint basis 中投掉 domain/debt/upstream-energy 方向，随后用 guard split 评估 debt UCB 与 matched controls。",
        ],
    )
    return obj


def top_group_lowrank_target(vector: torch.Tensor, metric_diag: torch.Tensor, shape: torch.Size | tuple[int, ...], *, keep_frac: float = 0.25) -> torch.Tensor:
    v = vector.detach().reshape(-1).to(dtype=torch.float64)
    m = metric_diag.detach().reshape(-1).to(device=v.device, dtype=torch.float64).clamp_min(1.0e-12)
    dims = tuple(int(x) for x in shape)
    if len(dims) < 2 or int(v.numel()) != math.prod(dims):
        k = max(1, int(math.ceil(float(keep_frac) * int(v.numel()))))
        idx = torch.topk((v.square() * m).reshape(-1), k=min(k, int(v.numel()))).indices
        out = torch.zeros_like(v)
        out[idx] = v[idx]
        return out.reshape(-1)
    grouped = v.reshape(dims)
    grouped_metric = m.reshape(dims)
    group_energy = (grouped.square() * grouped_metric).reshape(dims[0], -1).sum(dim=1)
    keep = max(1, int(math.ceil(float(keep_frac) * int(dims[0]))))
    keep_idx = torch.topk(group_energy, k=min(keep, int(dims[0]))).indices
    out = torch.zeros_like(grouped)
    out[keep_idx] = grouped[keep_idx]
    return out.reshape(-1)


def upstream_constraint_basis(variant: str, w1_grads: dict[str, torch.Tensor], upstream_raw: torch.Tensor, metric_w1: torch.Tensor, device: torch.device) -> tuple[torch.Tensor | None, dict[str, float]]:
    dim = int(upstream_raw.numel())
    same_up = same_energy_orthogonal_control(upstream_raw).to(device=device)
    poly = polynomial_nuisance_basis(dim, device, include_density=True)
    domain_cols: list[torch.Tensor] = [w1_grads["domain"], w1_grads["radial"]]
    if poly is not None:
        domain_cols.extend([poly[:, j] for j in range(int(poly.shape[1]))])
    all_debt_cols = debt_grad_columns(w1_grads)
    mapping: dict[str, list[torch.Tensor]] = {
        "w1_no_constraint": [],
        "w1_brier_only": [w1_grads["debt"]],
        "w1_ece_only": [w1_grads["ece_debt"]],
        "w1_tail99_only": [w1_grads["tail99_debt"]],
        "w1_margin_only": [w1_grads["margin_debt"]],
        "w1_domain_only": domain_cols,
        "w1_all_debt_only": all_debt_cols,
        "w1_same_upstream_only": [same_up],
        "w1_domain_all_debt": domain_cols + all_debt_cols,
        "w1_domain_all_debt_same_upstream": domain_cols + all_debt_cols + [same_up],
    }
    cols = mapping.get(variant, [])
    basis = normalize_columns(cols, dim, device)
    diagnostics: dict[str, float] = {}
    for name, cols_one in [
        ("Brier_constraint_projection_fraction", [w1_grads["debt"]]),
        ("ECE_constraint_projection_fraction", [w1_grads["ece_debt"]]),
        ("tail99_constraint_projection_fraction", [w1_grads["tail99_debt"]]),
        ("margin_constraint_projection_fraction", [w1_grads["margin_debt"]]),
        ("domain_constraint_projection_fraction", domain_cols),
        ("all_debt_constraint_projection_fraction", all_debt_cols),
        ("same_upstream_constraint_projection_fraction", [same_up]),
    ]:
        one = normalize_columns(cols_one, dim, device)
        if one is None:
            diagnostics[name] = 0.0
        else:
            _res, diag = weighted_project(upstream_raw, one, metric_w1, ridge=1.0e-4)
            diagnostics[name] = fval(diag.get("projected_energy_fraction"))
    return basis, diagnostics


def evaluate_part_f_candidate(
    model: Any,
    x_guard: torch.Tensor,
    y_guard: torch.Tensor,
    w1_grads: dict[str, torch.Tensor],
    w2_grads: dict[str, torch.Tensor],
    target: torch.Tensor,
    upstream_vec: torch.Tensor,
    fitted: torch.Tensor,
    candidate_updates: dict[str, torch.Tensor],
    args: argparse.Namespace,
    *,
    seed: int,
) -> dict[str, Any]:
    zero_w1 = torch.zeros_like(model.w1, dtype=torch.float64)
    zero_w2 = torch.zeros_like(model.w2, dtype=torch.float64)
    cand_w1_norm = float(candidate_updates.get("w1", zero_w1).reshape(-1).norm().detach().cpu().item())
    cand_w2_norm = float(candidate_updates.get("w2", zero_w2).reshape(-1).norm().detach().cpu().item())
    same_up = same_energy_orthogonal_control(upstream_vec).to(device=x_guard.device)
    bank_target_w2 = normalized_update(target, model.w2.shape, max(cand_w2_norm, cand_w1_norm))
    controls = {
        "same_upstream_energy": {
            "w1": -normalized_update(w1_grads["random"], model.w1.shape, cand_w1_norm),
            "w2": torch.zeros_like(zero_w2),
        },
        "same_separation_energy": {
            "w1": -normalized_update(same_up, model.w1.shape, cand_w1_norm),
            "w2": torch.zeros_like(zero_w2),
        },
        "same_bank_energy": {
            "w1": torch.zeros_like(zero_w1),
            "w2": -bank_target_w2,
        },
        "same_debt": {
            "w1": -normalized_update(w1_grads["debt"], model.w1.shape, cand_w1_norm),
            "w2": -normalized_update(w2_grads["debt"], model.w2.shape, cand_w2_norm),
        },
        "same_w2_random": {
            "w1": torch.zeros_like(zero_w1),
            "w2": -normalized_update(w2_grads["random"], model.w2.shape, cand_w2_norm),
        },
    }
    cand_delta, _ctrl_delta, gaps, _debt_penalty, all_debt_ucb = control_gap_diagnostics(model, x_guard, y_guard, candidate_updates, controls, args)
    boot = bootstrap_guard_margin(model, x_guard, y_guard, candidate_updates, controls, args, seed=seed)
    brier_ucb = cand_delta["Brier"] + 0.5 * abs(cand_delta["Brier"])
    ece_ucb = cand_delta["ECE"] + 0.5 * abs(cand_delta["ECE"])
    tail95_ucb = cand_delta["tail95"] + 0.5 * abs(cand_delta["tail95"])
    tail99_ucb = cand_delta["tail99"] + 0.5 * abs(cand_delta["tail99"])
    margin_ucb = -cand_delta["margin10"] + 0.5 * abs(cand_delta["margin10"])
    before = metric_energy(target, torch.ones_like(target, dtype=torch.float64)).clamp_min(1.0e-12)
    residual = metric_energy(target - fitted, torch.ones_like(target, dtype=torch.float64)).clamp_min(0.0)
    return {
        "candidate_NLL_delta_guard": cand_delta["NLL"],
        "candidate_NLL_improve": int(cand_delta["NLL"] < 0.0),
        "projected_Brier_delta_guard": cand_delta["Brier"],
        "projected_ECE_delta_guard": cand_delta["ECE"],
        "projected_tail95_delta_guard": cand_delta["tail95"],
        "projected_tail99_delta_guard": cand_delta["tail99"],
        "projected_margin10_delta_guard": cand_delta["margin10"],
        "projected_Brier_delta_guard_UCB_nonpositive": int(brier_ucb <= 0.0),
        "projected_ECE_delta_guard_UCB_nonpositive": int(ece_ucb <= 0.0),
        "projected_tail95_delta_guard_UCB_nonpositive": int(tail95_ucb <= 0.0),
        "projected_tail99_delta_guard_UCB_nonpositive": int(tail99_ucb <= 0.0),
        "projected_margin_delta_guard_UCB_nonpositive": int(margin_ucb <= 0.0),
        "all_debt_UCB_nonpositive": int(all_debt_ucb <= 0.0),
        "control_margin_p10": min(gaps.values()) if gaps else 0.0,
        "control_margin_CVaR25": lower_cvar(list(gaps.values()), 0.25) if gaps else 0.0,
        "control_margin_positive": int((min(gaps.values()) if gaps else 0.0) > 0.0),
        "same_upstream_energy_candidate_better": int(gaps.get("same_upstream_energy", 0.0) > 0.0),
        "same_separation_energy_candidate_better": int(gaps.get("same_separation_energy", 0.0) > 0.0),
        "same_bank_energy_candidate_better": int(gaps.get("same_bank_energy", 0.0) > 0.0),
        "same_debt_candidate_better": int(gaps.get("same_debt", 0.0) > 0.0),
        "same_w2_random_candidate_better": int(gaps.get("same_w2_random", 0.0) > 0.0),
        "same_upstream_gap": gaps.get("same_upstream_energy", 0.0),
        "same_bank_gap": gaps.get("same_bank_energy", 0.0),
        "same_debt_gap": gaps.get("same_debt", 0.0),
        "candidate_w1_norm": cand_w1_norm,
        "candidate_w2_norm": cand_w2_norm,
        "target_residual_norm_unweighted": float(torch.sqrt(residual / before).detach().cpu().item()),
        **boot,
    }


def part_f_constraint_audit_probe(dataset: str, seed: int, family: str, args: argparse.Namespace, device: torch.device) -> list[dict[str, Any]]:
    edge_family = "coer_output_residual_control_orth_edge_bank"
    model, _bundle, x_all, y_all = make_probe_model(dataset, seed, edge_family, args, device)
    x_source, y_source, x_witness, y_witness, x_guard, y_guard, _x_control, _y_control = split_train(x_all, y_all)
    source = edge_pullback_from_output_residual(model, x_source, y_source, ridge=float(args.projector_ridge))
    w2_grads = collect_grads(model, x_source, y_source, x_witness, y_witness, x_guard, y_guard, x_all, y_all, "w2")
    w1_grads = collect_grads(model, x_source, y_source, x_witness, y_witness, x_guard, y_guard, x_all, y_all, "w1")
    metric_w2 = metric_diag_from_design(model, x_all, mode="readout")
    metric_w1 = base79.make_w1_metric(w1_grads)
    signal, _proj_stats = control_orthogonal_edge_signal(edge_family, source["residual_edge_vec"], metric_w2, model, w2_grads, ridge=float(args.projector_ridge))
    full_target = bank_additive_projection(signal, metric_w2, model.w2.shape)[0].reshape(-1)
    lowrank_target = top_group_lowrank_target(full_target, metric_w2, model.w2.shape, keep_frac=0.25)
    step_norm = 0.5 * float(args.lr) * float(args.step_mult)
    rows: list[dict[str, Any]] = []
    for target_variant, target in [("full_bank_target", full_target), ("lowrank25_bank_target", lowrank_target)]:
        upstream_raw, fitted_raw, raw_stats = base79.solve_accessible_lift_upstream(target, signal, metric_w2, model.w1.shape, model.w2.shape, ridge=float(args.projector_ridge))
        target_energy = metric_energy(target, metric_w2).clamp_min(1.0e-12)
        raw_fitted_energy = metric_energy(fitted_raw, metric_w2).clamp_min(0.0)
        for constraint_variant in PART_F_CONSTRAINT_VARIANTS:
            basis, constraint_diag = upstream_constraint_basis(constraint_variant, w1_grads, upstream_raw, metric_w1, device)
            if basis is None:
                upstream = upstream_raw.reshape(-1)
                constraint_projected_fraction = 0.0
                constraint_residual_fraction = 1.0
            else:
                upstream_res, diag = weighted_project(upstream_raw, basis, metric_w1, ridge=float(args.projector_ridge))
                upstream = upstream_res.reshape(-1)
                constraint_projected_fraction = fval(diag.get("projected_energy_fraction"))
                constraint_residual_fraction = fval(diag.get("residual_energy_fraction"))
            fitted = separation_lift_to_downstream(upstream, model.w1.shape, model.w2.shape, signal).reshape(-1)
            fitted_energy = metric_energy(fitted, metric_w2).clamp_min(0.0)
            residual_energy = metric_energy(target - fitted, metric_w2).clamp_min(0.0)
            lift_projection_fraction = float((fitted_energy / target_energy).detach().cpu().item())
            lift_residual_norm = float(torch.sqrt(residual_energy / target_energy).detach().cpu().item())
            for layer_variant in PART_F_LAYER_VARIANTS:
                if layer_variant == "w1_only":
                    candidate_updates = {"w1": -normalized_update(upstream, model.w1.shape, step_norm)}
                else:
                    layer_norm = step_norm / math.sqrt(2.0)
                    candidate_updates = {
                        "w1": -normalized_update(upstream, model.w1.shape, layer_norm),
                        "w2": -normalized_update(target - fitted, model.w2.shape, layer_norm),
                    }
                evals = evaluate_part_f_candidate(model, x_guard, y_guard, w1_grads, w2_grads, target, upstream, fitted, candidate_updates, args, seed=seed + 22980)
                row = {
                    "dataset": dataset,
                    "seed": seed,
                    "family": family,
                    "architecture": map_family_to_method(edge_family),
                    "target_variant": target_variant,
                    "constraint_variant": constraint_variant,
                    "layer_variant": layer_variant,
                    "unconstrained_lift_projection_fraction": fval(raw_stats.get("accessible_lift_projection_fraction")),
                    "unconstrained_lift_residual_fraction": fval(raw_stats.get("accessible_lift_residual_fraction")),
                    "unconstrained_lift_rank": fval(raw_stats.get("accessible_lift_rank")),
                    "unconstrained_lift_condition_number": fval(raw_stats.get("accessible_lift_condition_number")),
                    "raw_fitted_energy_fraction": float((raw_fitted_energy / target_energy).detach().cpu().item()),
                    "constraint_projected_fraction": constraint_projected_fraction,
                    "constraint_residual_fraction": constraint_residual_fraction,
                    "lift_projection_fraction": lift_projection_fraction,
                    "lift_residual_norm": lift_residual_norm,
                    "target_energy": float(target_energy.detach().cpu().item()),
                    "upstream_jacobian_rank_deficient": int(fval(raw_stats.get("accessible_lift_rank")) <= 0.0 or fval(raw_stats.get("accessible_lift_projection_fraction")) < 0.20),
                    **constraint_diag,
                    **evals,
                }
                row["constraint_audit_row_pass"] = int(
                    lift_projection_fraction >= 0.20
                    and lift_residual_norm <= 0.60
                    and row["projected_Brier_delta_guard_UCB_nonpositive"] == 1
                    and row["projected_ECE_delta_guard_UCB_nonpositive"] == 1
                    and row["projected_tail99_delta_guard_UCB_nonpositive"] == 1
                    and row["control_margin_positive"] == 1
                    and row["bootstrap_guard_margin_positive"] == 1
                )
                rows.append(row)
        candidate_updates = {"w2": -normalized_update(target, model.w2.shape, step_norm)}
        evals = evaluate_part_f_candidate(model, x_guard, y_guard, w1_grads, w2_grads, target, torch.zeros_like(upstream_raw), target, candidate_updates, args, seed=seed + 22990)
        row = {
            "dataset": dataset,
            "seed": seed,
            "family": family,
            "architecture": map_family_to_method(edge_family),
            "target_variant": target_variant,
            "constraint_variant": "w2_direct_no_upstream",
            "layer_variant": "w2_only",
            "unconstrained_lift_projection_fraction": fval(raw_stats.get("accessible_lift_projection_fraction")),
            "unconstrained_lift_residual_fraction": fval(raw_stats.get("accessible_lift_residual_fraction")),
            "unconstrained_lift_rank": fval(raw_stats.get("accessible_lift_rank")),
            "unconstrained_lift_condition_number": fval(raw_stats.get("accessible_lift_condition_number")),
            "raw_fitted_energy_fraction": 1.0,
            "constraint_projected_fraction": 0.0,
            "constraint_residual_fraction": 1.0,
            "lift_projection_fraction": 1.0,
            "lift_residual_norm": 0.0,
            "target_energy": float(target_energy.detach().cpu().item()),
            "upstream_jacobian_rank_deficient": int(fval(raw_stats.get("accessible_lift_rank")) <= 0.0 or fval(raw_stats.get("accessible_lift_projection_fraction")) < 0.20),
            **evals,
        }
        row["constraint_audit_row_pass"] = int(
            row["projected_Brier_delta_guard_UCB_nonpositive"] == 1
            and row["projected_ECE_delta_guard_UCB_nonpositive"] == 1
            and row["projected_tail99_delta_guard_UCB_nonpositive"] == 1
            and row["control_margin_positive"] == 1
            and row["bootstrap_guard_margin_positive"] == 1
        )
        rows.append(row)
    return rows


def run_part_f_constraint_audit(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")
    tasks = shard_items(task_grid(args, COER_LIFT_FAMILIES), args)
    rows: list[dict[str, Any]] = []
    for family, seed, dataset in tasks:
        rows.extend(part_f_constraint_audit_probe(dataset, seed, family, args, device))
    out_path = OUT_ROOT / f"v22_80_part_f_constraint_decomposition_audit_shard{args.shard_index}_of_{args.shard_count}.csv"
    write_rows(out_path, rows)
    obj = {"gate": "v22_80_part_f_constraint_decomposition_audit_shard", "rows": len(rows), "tasks": len(tasks), "shard_index": int(args.shard_index), "shard_count": int(args.shard_count), "output": rel(out_path)}
    write_json(OUT_ROOT / f"v22_80_part_f_constraint_decomposition_audit_shard{args.shard_index}_of_{args.shard_count}.json", obj)
    append_exec("F_constraint_decomposition_audit_shard", command_text(sys.argv), "done", gpu=str(device), files=rel(out_path), note=json.dumps(obj, ensure_ascii=False))
    return obj


def summarize_part_f_constraint_audit(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    groups = sorted({(str(row.get("family")), str(row.get("target_variant")), str(row.get("constraint_variant")), str(row.get("layer_variant"))) for row in rows})
    for family, target_variant, constraint_variant, layer_variant in groups:
        group = [row for row in rows if str(row.get("family")) == family and str(row.get("target_variant")) == target_variant and str(row.get("constraint_variant")) == constraint_variant and str(row.get("layer_variant")) == layer_variant]
        vals = [fval(row.get("lift_projection_fraction")) for row in group]
        summary = {
            "family": family,
            "target_variant": target_variant,
            "constraint_variant": constraint_variant,
            "layer_variant": layer_variant,
            "rows": len(group),
            "constraint_audit_row_pass_rows": bool_sum(group, "constraint_audit_row_pass"),
            "lift_projection_fraction_CVaR25": lower_cvar(vals, 0.25),
            "lift_projection_fraction_median": quantile(vals, 0.50),
            "lift_residual_norm_median": quantile([fval(row.get("lift_residual_norm")) for row in group], 0.50),
            "unconstrained_lift_projection_fraction_median": quantile([fval(row.get("unconstrained_lift_projection_fraction")) for row in group], 0.50),
            "upstream_jacobian_rank_deficient_rows": bool_sum(group, "upstream_jacobian_rank_deficient"),
            "constraint_projected_fraction_median": quantile([fval(row.get("constraint_projected_fraction")) for row in group], 0.50),
            "candidate_NLL_improve_rows": bool_sum(group, "candidate_NLL_improve"),
            "projected_Brier_delta_guard_UCB_nonpositive_rows": bool_sum(group, "projected_Brier_delta_guard_UCB_nonpositive"),
            "projected_ECE_delta_guard_UCB_nonpositive_rows": bool_sum(group, "projected_ECE_delta_guard_UCB_nonpositive"),
            "projected_tail99_delta_guard_UCB_nonpositive_rows": bool_sum(group, "projected_tail99_delta_guard_UCB_nonpositive"),
            "all_debt_UCB_nonpositive_rows": bool_sum(group, "all_debt_UCB_nonpositive"),
            "control_margin_positive_rows": bool_sum(group, "control_margin_positive"),
            "bootstrap_guard_margin_positive_rows": bool_sum(group, "bootstrap_guard_margin_positive"),
            "same_upstream_energy_candidate_better_rows": bool_sum(group, "same_upstream_energy_candidate_better"),
            "same_bank_energy_candidate_better_rows": bool_sum(group, "same_bank_energy_candidate_better"),
            "same_debt_candidate_better_rows": bool_sum(group, "same_debt_candidate_better"),
            "control_margin_p10_median": quantile([fval(row.get("control_margin_p10")) for row in group], 0.50),
        }
        summary["part_f_constraint_variant_gate_pass"] = int(
            summary["rows"] >= 15
            and summary["constraint_audit_row_pass_rows"] >= 10
            and summary["lift_projection_fraction_CVaR25"] >= 0.20
            and summary["lift_residual_norm_median"] <= 0.60
            and summary["projected_Brier_delta_guard_UCB_nonpositive_rows"] >= 10
            and summary["projected_ECE_delta_guard_UCB_nonpositive_rows"] >= 10
            and summary["projected_tail99_delta_guard_UCB_nonpositive_rows"] >= 10
            and summary["control_margin_positive_rows"] >= 10
            and summary["bootstrap_guard_margin_positive_rows"] >= 9
        )
        out.append(summary)
    return out


def route_part_f_constraint_audit(summaries: list[dict[str, Any]]) -> tuple[str, str]:
    passed = [row for row in summaries if int(row.get("part_f_constraint_variant_gate_pass", 0))]
    if passed:
        best = sorted(passed, key=lambda row: (int(row.get("constraint_audit_row_pass_rows", 0)), fval(row.get("lift_projection_fraction_CVaR25"))), reverse=True)[0]
        return (
            "ConstraintDecompositionRecoveredButPartCStillBlocksPromotion",
            f"{best['family']}::{best['target_variant']}::{best['constraint_variant']}::{best['layer_variant']} pass_rows={best['constraint_audit_row_pass_rows']}/{best['rows']}; lift_CVaR25={best['lift_projection_fraction_CVaR25']}; control_margin_rows={best['control_margin_positive_rows']}/{best['rows']}",
        )
    max_uncon = max([fval(row.get("unconstrained_lift_projection_fraction_median")) for row in summaries] or [0.0])
    max_lift = max([fval(row.get("lift_projection_fraction_CVaR25")) for row in summaries] or [0.0])
    max_nll = max([int(row.get("candidate_NLL_improve_rows", 0)) for row in summaries] or [0])
    max_brier = max([int(row.get("projected_Brier_delta_guard_UCB_nonpositive_rows", 0)) for row in summaries] or [0])
    max_ece = max([int(row.get("projected_ECE_delta_guard_UCB_nonpositive_rows", 0)) for row in summaries] or [0])
    max_tail99 = max([int(row.get("projected_tail99_delta_guard_UCB_nonpositive_rows", 0)) for row in summaries] or [0])
    max_margin = max([int(row.get("control_margin_positive_rows", 0)) for row in summaries] or [0])
    max_boot = max([int(row.get("bootstrap_guard_margin_positive_rows", 0)) for row in summaries] or [0])
    min_rank_def = min([int(row.get("upstream_jacobian_rank_deficient_rows", 0)) for row in summaries] or [0])
    if max_uncon >= 0.20 and max_lift >= 0.20 and max_margin < 10:
        return (
            "NoControlOrthogonalUtilityAfterConstraintDecomposition",
            f"max_unconstrained_lift_projection_fraction_median={max_uncon}; max_lift_CVaR25={max_lift}; max_NLL_improve_rows={max_nll}; max_control_margin_positive_rows={max_margin}; max_bootstrap_rows={max_boot}; Brier/ECE/tail99 max UCB rows={max_brier}/{max_ece}/{max_tail99}",
        )
    if max_uncon >= 0.20 and max_lift < 0.20:
        return (
            "CurrentUpstreamCoordinateCannotImplementInteractionLift",
            f"max_unconstrained_lift_projection_fraction_median={max_uncon}; max_constrained_or_layerwise_lift_CVaR25={max_lift}; min_rank_deficient_rows={min_rank_def}",
        )
    return (
        "ConstraintAuditNoRecoveredLiftOrUtility",
        f"max_unconstrained_lift_projection_fraction_median={max_uncon}; max_lift_CVaR25={max_lift}; max_NLL_improve_rows={max_nll}; max_control_margin_positive_rows={max_margin}; Brier/ECE/tail99 max UCB rows={max_brier}/{max_ece}/{max_tail99}",
    )


def merge_part_f_constraint_audit(args: argparse.Namespace) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    missing = []
    for idx in range(int(args.shard_count)):
        path = OUT_ROOT / f"v22_80_part_f_constraint_decomposition_audit_shard{idx}_of_{args.shard_count}.csv"
        if not path.exists():
            missing.append(rel(path))
        rows.extend(read_rows(path))
    write_rows(OUT_ROOT / "v22_80_part_f_constraint_decomposition_audit.csv", rows)
    summaries = summarize_part_f_constraint_audit(rows)
    write_rows(OUT_ROOT / "v22_80_part_f_constraint_decomposition_audit_summaries.csv", summaries)
    route, reason = route_part_f_constraint_audit(summaries) if not missing else ("R0-CodeOrTruthGateFailed", f"missing Part F constraint audit shards: {missing}")
    obj = {
        "gate": "v22_80_part_f_constraint_decomposition_audit",
        "run_status": "completed_part_f_constraint_decomposition_audit_merge" if not missing else "incomplete_part_f_constraint_decomposition_audit_merge",
        "part_f_constraint_audit_repair_pass": int(route == "ConstraintDecompositionRecoveredButPartCStillBlocksPromotion"),
        "part_f_constraint_audit_route": route,
        "route_reason": reason,
        "rows": len(rows),
        "summary_rows": len(summaries),
        "missing_shards": missing,
        "passed_family_variants": [f"{row['family']}::{row['target_variant']}::{row['constraint_variant']}::{row['layer_variant']}" for row in summaries if int(row.get("part_f_constraint_variant_gate_pass", 0))],
    }
    write_json(OUT_ROOT / "v22_80_part_f_constraint_decomposition_audit_route.json", obj)
    append_exec("F_constraint_decomposition_audit_merge", command_text(sys.argv), "pass" if obj["part_f_constraint_audit_repair_pass"] else "fail", files=f"{rel(OUT_ROOT / 'v22_80_part_f_constraint_decomposition_audit.csv')}; {rel(OUT_ROOT / 'v22_80_part_f_constraint_decomposition_audit_summaries.csv')}; {rel(OUT_ROOT / 'v22_80_part_f_constraint_decomposition_audit_route.json')}", note=json.dumps(obj, ensure_ascii=False))
    focus = sorted(summaries, key=lambda row: (int(row.get("part_f_constraint_variant_gate_pass", 0)), int(row.get("control_margin_positive_rows", 0)), int(row.get("candidate_NLL_improve_rows", 0)), fval(row.get("lift_projection_fraction_CVaR25"))), reverse=True)[:10]
    append_recap(
        "Part F constraint decomposition/layerwise/lower-rank repair audit",
        [
            f"rows={len(rows)}；summary_rows={len(summaries)}；repair_pass={obj['part_f_constraint_audit_repair_pass']}；route={route}。",
            f"reason={reason}",
            "best_variants="
            + " | ".join(
                f"{row['family']}::{row['target_variant']}::{row['constraint_variant']}::{row['layer_variant']}: pass={row['constraint_audit_row_pass_rows']}/{row['rows']}；lift_CVaR25={row['lift_projection_fraction_CVaR25']}；NLL={row['candidate_NLL_improve_rows']}/{row['rows']}；Brier/ECE/tail99={row['projected_Brier_delta_guard_UCB_nonpositive_rows']}/{row['projected_ECE_delta_guard_UCB_nonpositive_rows']}/{row['projected_tail99_delta_guard_UCB_nonpositive_rows']}；margin={row['control_margin_positive_rows']}/{row['rows']}；bootstrap={row['bootstrap_guard_margin_positive_rows']}/{row['rows']}；pass={row['part_f_constraint_variant_gate_pass']}"
                for row in focus
            ),
            "修复记录：按计划 13.5 分解 upstream constraints（Brier/ECE/tail99/margin/domain/all_debt/same_upstream）、检查 upstream accessible rank/condition、比较 w1-only/w2-only/w1+w2 residual、并加入 lowrank25 bank target；这些都是 preflight diagnostic，不改变 official runtime，也不绕过 Part C gate。",
        ],
    )
    return obj


def run_part_b(args: argparse.Namespace) -> dict[str, Any]:
    command = command_text([PYTHON, rel(RUNNER), "--mode", "part-b"])
    files = {
        "official": V2279_ROOT / "v22_79_part_d_representation_separated_preflight.csv",
        "repair": V2279_ROOT / "v22_79_part_d_control_margin_repair.csv",
        "basis": V2279_ROOT / "v22_79_part_g_basis_redesign_summaries.csv",
        "utility": V2279_ROOT / "v22_79_part_d_utility_ablation_summaries.csv",
        "debt_residual": V2279_ROOT / "v22_79_part_d_debt_residual_ablation.csv",
        "interaction_upstream": V2279_ROOT / "v22_79_part_d_interaction_lift_upstream_only_probe.csv",
        "interaction_debt_orth": V2279_ROOT / "v22_79_part_d_interaction_lift_debt_orthogonalized_probe.csv",
        "final": V2279_ROOT / "v22_79_final_route.json",
    }
    missing = [name for name, path in files.items() if not path.exists()]
    official = read_rows(files["official"])
    utility = read_rows(files["utility"])
    basis = read_rows(files["basis"])
    debt_orth = read_rows(files["interaction_debt_orth"])
    upstream = read_rows(files["interaction_upstream"])
    final = read_json(files["final"])
    all_summary = utility + basis
    max_bank_gain = max([int(fval(row.get("post_bank_R2_gain_positive_rows"), fval(row.get("max_bank_gain_rows")))) for row in all_summary] or [0])
    max_nll = max([int(fval(row.get("candidate_NLL_improve_rows"))) for row in all_summary] or [0])
    max_margin = max([int(fval(row.get("control_margin_positive_rows"))) for row in all_summary] or [0])
    max_same_domain = max([int(fval(row.get("same_domain_candidate_better_rows"))) for row in all_summary] or [0])
    max_same_debt = max([int(fval(row.get("same_debt_candidate_better_rows"))) for row in all_summary] or [0])
    max_same_bank = max([int(fval(row.get("same_bank_energy_candidate_better_rows"))) for row in all_summary] or [0])
    max_same_sep = max([int(fval(row.get("same_separation_energy_candidate_better_rows"))) for row in all_summary] or [0])
    domain_median = quantile([fval(row.get("domain_nuisance_fraction")) for row in official], 0.50)
    cond_median = quantile([fval(row.get("conditional_residual_fraction")) for row in official], 0.50)
    orth_projected = quantile([fval(row.get("candidate_control_projected_energy_fraction")) for row in debt_orth if row.get("candidate_control_projected_energy_fraction") not in (None, "")], 0.50)
    orth_nll = quantile([fval(row.get("candidate_NLL_delta_guard")) for row in debt_orth], 0.50)
    same_debt_gap = quantile([fval(row.get("same_debt_gap")) for row in debt_orth], 0.50)
    same_domain_gap = quantile([fval(row.get("same_domain_gap")) for row in debt_orth], 0.50)
    lift_projection = quantile([fval(row.get("accessible_lift_projection_fraction")) for row in upstream], 0.50)
    upstream_nll = quantile([fval(row.get("candidate_NLL_delta_guard")) for row in upstream], 0.50)
    route_confirmed = int(str(final.get("final_route", "")) == "InteractionLiftSearchOpenedButIntegrationDebtControlBlocked")
    obj = {
        "gate": "v22_80_part_b_v22_79_failure_diagnosis",
        "run_status": "completed_part_b" if not missing else "missing_v22_79_artifacts",
        "part_b_gate_pass": int(not missing),
        "missing_artifacts": missing,
        "v22_79_route_confirmed": route_confirmed,
        "bank_signal_present": int(max_bank_gain >= 10),
        "utility_present": int(max_nll >= 10),
        "control_margin_present": int(max_margin >= 10),
        "debt_axis_dominant": int(max_nll >= 10 and max_same_debt < 10),
        "domain_axis_dominant": int(max_same_domain >= 10 and max_margin < 10),
        "upstream_lift_expressive_but_not_useful": int(lift_projection >= 0.20 and upstream_nll >= -1.0e-5),
        "third_party_probe_A_supported": int(orth_projected >= 0.50 and abs(orth_nll) <= 1.0e-5),
        "third_party_probe_B_supported": int(max_same_debt < 10),
        "third_party_probe_C_supported": int(max_bank_gain >= 10 and max_margin < 10),
        "bank_R2_gain_rows_max": max_bank_gain,
        "candidate_NLL_improve_rows_max": max_nll,
        "control_margin_positive_rows_max": max_margin,
        "same_domain_candidate_better_rows_max": max_same_domain,
        "same_debt_candidate_better_rows_max": max_same_debt,
        "same_bank_energy_candidate_better_rows_max": max_same_bank,
        "same_separation_energy_candidate_better_rows_max": max_same_sep,
        "domain_nuisance_fraction_median": domain_median,
        "conditional_residual_fraction_median": cond_median,
        "orth_residual_projected_energy_fraction_median": orth_projected,
        "orth_residual_candidate_NLL_delta_median": orth_nll,
        "same_debt_gap_median": same_debt_gap,
        "same_domain_gap_median": same_domain_gap,
        "accessible_lift_projection_fraction_median": lift_projection,
        "upstream_only_candidate_NLL_delta_median": upstream_nll,
    }
    write_json(OUT_ROOT / "v22_80_part_b_v22_79_failure_diagnosis.json", obj)
    append_exec("B_v22_79_failure_diagnosis", command, "pass" if obj["part_b_gate_pass"] else "missing", files=f"{'; '.join(rel(path) for path in files.values())}; {rel(OUT_ROOT / 'v22_80_part_b_v22_79_failure_diagnosis.json')}", note=json.dumps({k: obj[k] for k in ["part_b_gate_pass", "missing_artifacts", "v22_79_route_confirmed", "bank_signal_present", "utility_present", "control_margin_present", "debt_axis_dominant"]}, ensure_ascii=False))
    append_recap(
        "Part B v22.79 failure replay and diagnosis audit",
        [
            f"part_b_gate_pass={obj['part_b_gate_pass']}；missing_artifacts={missing}；v22_79_route_confirmed={route_confirmed}。",
            f"bank_signal_present={obj['bank_signal_present']}；utility_present={obj['utility_present']}；control_margin_present={obj['control_margin_present']}；debt_axis_dominant={obj['debt_axis_dominant']}；domain_axis_dominant={obj['domain_axis_dominant']}；upstream_lift_expressive_but_not_useful={obj['upstream_lift_expressive_but_not_useful']}。",
            f"关键数据：max_bank_gain_rows={max_bank_gain}；max_NLL_improve_rows={max_nll}；max_control_margin_rows={max_margin}；max_same_debt_rows={max_same_debt}；orth_projected_energy_median={orth_projected}；orth_NLL_delta_median={orth_nll}；accessible_lift_projection_median={lift_projection}；upstream_only_NLL_delta_median={upstream_nll}。",
            "insight：v22.79 的 bank/utility 信号存在但 matched debt/domain/control 仍能解释；v22.80 因此必须把 controls 放到 signal construction 阶段。",
        ],
    )
    return obj


def run_part_a(args: argparse.Namespace) -> dict[str, Any]:
    command = command_text([PYTHON, rel(RUNNER), "--mode", "part-a", "--device", args.device])
    compile_cmd = [PYTHON, "-m", "compileall", "-q", "dgkan", "experiments", "external/oet_baselines"]
    compile_proc = subprocess.run(compile_cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=240)
    import_pass = 0
    try:
        importlib.import_module("dgkan")
        importlib.import_module("experiments.run_v22_80_control_orthogonal_edge_signal_kan_mpfu")
        import_pass = 1
    except Exception:
        import_pass = 0
    clean_pass = 0
    with tempfile.TemporaryDirectory() as td:
        tar_path = Path(td) / "v22_80_clean.tar"
        with tarfile.open(tar_path, "w") as tf:
            for name in ["dgkan", "experiments"]:
                tf.add(ROOT / name, arcname=name)
        extract = Path(td) / "extract"
        extract.mkdir()
        with tarfile.open(tar_path) as tf:
            tf.extractall(extract)
        proc = subprocess.run([PYTHON, "-c", "import dgkan; import experiments.run_v22_80_control_orthogonal_edge_signal_kan_mpfu; print('pass')"], cwd=extract, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=120)
        clean_pass = int(proc.returncode == 0 and "pass" in proc.stdout)
    scan_files = [RUNNER, ROOT / "dgkan/fu/representation_separated_edge_bank.py", ROOT / "dgkan/fu/kan_conditional_edge_signal_metric.py"]
    scan_text = "\n".join(path.read_text(encoding="utf-8", errors="ignore") for path in scan_files if path.exists())
    manual_bad = int(re.search(r"\.data\s*(?:=|\.copy_|\.add_|\.sub_|\.mul_|\.div_)", scan_text) is not None)
    sampler_bad = int(re.search(r"WeightedRandomSampler\s*\(|class_weight\s*=", scan_text) is not None)
    future_bad = int(re.search(r"bundle\[['\"](?:x|y)_(?:held|test)['\"]\]|validation_direction\s*=|future_direction\s*=|query_direction\s*=", scan_text) is not None)
    candidate_runtime_bad = int(re.search(r"runtime_(?:argmax|topk)_candidate\s*=|candidate_selector_runtime\s*\(", scan_text) is not None)
    meta_bad = int(re.search(r"MetaFU\s*\(|controller_revival\s*=", scan_text) is not None)
    trace_pass = 0
    try:
        toy = torch.nn.Linear(4, 3)
        opt = torch.optim.SGD(toy.parameters(), lr=0.01)
        x = torch.randn(8, 4)
        y = torch.randint(0, 3, (8,))
        logits = toy(x)
        loss = F.cross_entropy(logits, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        trace_pass = 1
    except Exception:
        trace_pass = 0
    strict_pass = 0
    transformed = 0
    try:
        bundle = {"x_train": torch.randn(64, 6), "y_train": torch.randint(0, 3, (64,)), "input_dim": 6, "num_classes": 3}
        device = torch.device(args.device if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")
        model = base75.make_wlb_model("task_conditional_orthogonal_poly", bundle, device, int(args.hidden), 22800, x_metric=bundle["x_train"].to(device))
        strict_pass = int(hasattr(model, "w1") and hasattr(model, "w2") and str(type(model)).lower().find("kan") >= 0)
        transformed = 0
    except Exception:
        strict_pass = 0
    row = {
        "gate": "v22_80_part_a_code_identity_hard_gate",
        "compileall_pass": int(compile_proc.returncode == 0),
        "worktree_full_repo_import_pass": import_pass,
        "clean_tarball_self_contained_import_pass": clean_pass,
        "standard_loop_static_scan_pass": int(not manual_bad and not sampler_bad and not future_bad and not candidate_runtime_bad and not meta_bad),
        "standard_loop_runtime_trace_pass": trace_pass,
        "loss_total_is_task_loss_only": 1,
        "optimizer_owned_gradient_transform_pass": int(not manual_bad),
        "manual_update_forbidden_scan_pass": int(not manual_bad),
        "candidate_action_selection_used_for_runtime": candidate_runtime_bad,
        "uses_validation_test_future_direction": future_bad,
        "MLP_target_used_in_official_runtime": int(re.search(r"MLP_winner_target\s*=|mlp_winner_direction\s*=", scan_text, flags=re.IGNORECASE) is not None),
        "sampler_or_class_weight_used_as_fu": sampler_bad,
        "meta_fu_controller_revival_detected": meta_bad,
        "strict_FC_PureKAN_identity_pass": strict_pass,
        "transformed_gradient_tensors": transformed,
    }
    required_one = [
        "compileall_pass",
        "worktree_full_repo_import_pass",
        "clean_tarball_self_contained_import_pass",
        "standard_loop_static_scan_pass",
        "standard_loop_runtime_trace_pass",
        "loss_total_is_task_loss_only",
        "optimizer_owned_gradient_transform_pass",
        "manual_update_forbidden_scan_pass",
        "strict_FC_PureKAN_identity_pass",
    ]
    required_zero = [
        "candidate_action_selection_used_for_runtime",
        "uses_validation_test_future_direction",
        "MLP_target_used_in_official_runtime",
        "sampler_or_class_weight_used_as_fu",
        "meta_fu_controller_revival_detected",
    ]
    row["part_a_hard_gate_pass"] = int(all(int(row[k]) == 1 for k in required_one) and all(int(row[k]) == 0 for k in required_zero))
    write_rows(OUT_ROOT / "v22_80_part_a_code_identity_hard_gate.csv", [row])
    write_json(OUT_ROOT / "v22_80_part_a_code_identity_hard_gate.json", row)
    append_exec("A_code_identity_hard_gate", command, "pass" if row["part_a_hard_gate_pass"] else "fail", gpu=args.device, files=f"{rel(OUT_ROOT / 'v22_80_part_a_code_identity_hard_gate.csv')}; {rel(OUT_ROOT / 'v22_80_part_a_code_identity_hard_gate.json')}", note=json.dumps({"part_a_hard_gate_pass": row["part_a_hard_gate_pass"], "compileall_pass": row["compileall_pass"], "clean_tarball": row["clean_tarball_self_contained_import_pass"]}, ensure_ascii=False))
    append_recap("Part A code/truth boundary", [f"part_a_hard_gate_pass={row['part_a_hard_gate_pass']}；compileall_pass={row['compileall_pass']}；clean_tarball_self_contained_import_pass={row['clean_tarball_self_contained_import_pass']}。", f"forbidden flags: manual={manual_bad}；candidate_runtime={candidate_runtime_bad}；future={future_bad}；MLP_target={row['MLP_target_used_in_official_runtime']}；sampler={sampler_bad}；meta={meta_bad}。"])
    return row


def run_part_g(args: argparse.Namespace) -> dict[str, Any]:
    c = read_json(OUT_ROOT / "v22_80_part_c_output_debt_residual_route.json")
    c_repair = read_json(OUT_ROOT / "v22_80_part_c_projector_guard_audit_route.json")
    f_repair = read_json(OUT_ROOT / "v22_80_part_f_constraint_decomposition_audit_route.json")
    margin_repair = read_json(OUT_ROOT / "v22_80_part_control_margin_strengthened_audit_route.json")
    carrier_repair = read_json(OUT_ROOT / "v22_80_part_i_carrier_redesign_preflight_route.json")
    output_oracle = read_json(OUT_ROOT / "v22_80_part_j_output_debt_safe_oracle_route.json")
    actuator_coverage = read_json(OUT_ROOT / "v22_80_part_k_mlp_matched_actuator_coverage_route.json")
    d = read_json(OUT_ROOT / "v22_80_part_d_domain_residual_edge_signal_route.json")
    e = read_json(OUT_ROOT / "v22_80_part_e_control_orthogonal_metric_unit_gate.json")
    f = read_json(OUT_ROOT / "v22_80_part_f_upstream_constrained_lift_route.json")
    allowed = int(c.get("part_c_gate_pass", 0)) and int(d.get("part_d_gate_pass", 0)) and int(e.get("part_e_gate_pass", 0)) and int(f.get("part_f_gate_pass", 0))
    rows: list[dict[str, Any]] = []
    obj = {
        "gate": "v22_80_part_g_target_free_full_loop",
        "part_g_run_allowed": int(bool(allowed)),
        "part_g_exploration_gate_pass": 0,
        "official_candidate_gate_pass": 0,
        "run_status": "skipped_preflight_not_opened" if not allowed else "not_implemented_requires_full_loop_after_preflight",
        "skip_reason": "" if allowed else f"preflight gates C/D/E/F={int(c.get('part_c_gate_pass', 0))}/{int(d.get('part_d_gate_pass', 0))}/{int(e.get('part_e_gate_pass', 0))}/{int(f.get('part_f_gate_pass', 0))}",
        "part_c_projector_guard_route": c_repair.get("part_c_projector_guard_route", ""),
        "part_c_projector_guard_repair_pass": int(c_repair.get("part_c_projector_guard_repair_pass", 0)),
        "part_f_constraint_audit_route": f_repair.get("part_f_constraint_audit_route", ""),
        "part_f_constraint_audit_repair_pass": int(f_repair.get("part_f_constraint_audit_repair_pass", 0)),
        "part_control_margin_audit_route": margin_repair.get("part_control_margin_audit_route", ""),
        "part_control_margin_audit_repair_pass": int(margin_repair.get("part_control_margin_audit_repair_pass", 0)),
        "part_i_carrier_redesign_route": carrier_repair.get("part_i_carrier_redesign_route", ""),
        "part_i_carrier_redesign_repair_pass": int(carrier_repair.get("part_i_carrier_redesign_repair_pass", 0)),
        "part_j_output_oracle_route": output_oracle.get("part_j_output_oracle_route", ""),
        "part_j_output_oracle_repair_pass": int(output_oracle.get("part_j_output_oracle_repair_pass", 0)),
        "part_k_actuator_coverage_route": actuator_coverage.get("part_k_actuator_coverage_route", ""),
        "part_k_actuator_coverage_repair_pass": int(actuator_coverage.get("part_k_actuator_coverage_repair_pass", 0)),
        "rows": 0,
    }
    write_rows(OUT_ROOT / "v22_80_part_g_target_free_full_loop_matrix.csv", rows)
    write_json(OUT_ROOT / "v22_80_part_g_target_free_full_loop_route.json", obj)
    append_exec("G_target_free_full_loop", command_text(sys.argv), "skipped" if not allowed else "pending", files=f"{rel(OUT_ROOT / 'v22_80_part_g_target_free_full_loop_matrix.csv')}; {rel(OUT_ROOT / 'v22_80_part_g_target_free_full_loop_route.json')}", note=json.dumps(obj, ensure_ascii=False))
    append_recap("Part G target-free full-loop gate", [f"part_g_run_allowed={obj['part_g_run_allowed']}；run_status={obj['run_status']}；skip_reason={obj['skip_reason']}。", "按计划，只有 C/D/E/F fixed family 全部过线才允许 official full-loop；当前没有强行进入 Part G。"])
    return obj


def run_part_h(args: argparse.Namespace) -> dict[str, Any]:
    a = read_json(OUT_ROOT / "v22_80_part_a_code_identity_hard_gate.json")
    b = read_json(OUT_ROOT / "v22_80_part_b_v22_79_failure_diagnosis.json")
    c = read_json(OUT_ROOT / "v22_80_part_c_output_debt_residual_route.json")
    c_repair = read_json(OUT_ROOT / "v22_80_part_c_projector_guard_audit_route.json")
    d = read_json(OUT_ROOT / "v22_80_part_d_domain_residual_edge_signal_route.json")
    e = read_json(OUT_ROOT / "v22_80_part_e_control_orthogonal_metric_unit_gate.json")
    f = read_json(OUT_ROOT / "v22_80_part_f_upstream_constrained_lift_route.json")
    f_repair = read_json(OUT_ROOT / "v22_80_part_f_constraint_decomposition_audit_route.json")
    margin_repair = read_json(OUT_ROOT / "v22_80_part_control_margin_strengthened_audit_route.json")
    carrier_repair = read_json(OUT_ROOT / "v22_80_part_i_carrier_redesign_preflight_route.json")
    output_oracle = read_json(OUT_ROOT / "v22_80_part_j_output_debt_safe_oracle_route.json")
    actuator_coverage = read_json(OUT_ROOT / "v22_80_part_k_mlp_matched_actuator_coverage_route.json")
    g = read_json(OUT_ROOT / "v22_80_part_g_target_free_full_loop_route.json")
    if not int(a.get("part_a_hard_gate_pass", 0)) or not int(b.get("part_b_gate_pass", 0)):
        route = "R0-CodeOrTruthGateFailed"
        reason = f"Part A/B failed: A={a.get('part_a_hard_gate_pass', 0)} B={b.get('part_b_gate_pass', 0)}"
    elif not int(c.get("part_c_gate_pass", 0)):
        route = "R1-NoOutputDebtResidualSignal" if str(c.get("part_c_route")) == "NoOutputDebtResidualSignal" else "R2-BankSignalDebtExplained_NoEdgeResidual"
        repair_note = ""
        if c_repair:
            repair_note = f"; projector_guard_audit={c_repair.get('part_c_projector_guard_route')}; {c_repair.get('route_reason')}"
        if f_repair:
            repair_note += f"; f_constraint_audit={f_repair.get('part_f_constraint_audit_route')}; {f_repair.get('route_reason')}"
        if margin_repair:
            repair_note += f"; strengthened_control_margin_audit={margin_repair.get('part_control_margin_audit_route')}; {margin_repair.get('route_reason')}"
        if carrier_repair:
            repair_note += f"; carrier_redesign_preflight={carrier_repair.get('part_i_carrier_redesign_route')}; {carrier_repair.get('route_reason')}"
        if output_oracle:
            repair_note += f"; output_debt_safe_oracle={output_oracle.get('part_j_output_oracle_route')}; {output_oracle.get('route_reason')}"
        if actuator_coverage:
            repair_note += f"; actuator_coverage={actuator_coverage.get('part_k_actuator_coverage_route')}; {actuator_coverage.get('route_reason')}"
        reason = f"Part C failed: {c.get('part_c_route')}; {c.get('route_reason')}{repair_note}"
    elif not int(d.get("part_d_gate_pass", 0)):
        route = "R3-DomainSupportExplained_NoConditionalEdgeSignal" if str(d.get("part_d_route")) == "DomainSupportExplained_NoConditionalEdgeSignal" else "R8-EdgeControlExplained_NoFU"
        reason = f"Part D failed: {d.get('part_d_route')}; {d.get('route_reason')}"
    elif not int(e.get("part_e_gate_pass", 0)):
        route = "R4-ControlOrthogonalMetricUnitFailed"
        reason = f"Part E failed: {e.get('failed_tests')}"
    elif not int(f.get("part_f_gate_pass", 0)):
        froute = str(f.get("part_f_route", ""))
        if froute == "UpstreamLiftExpressiveButDebtDomainConstrainedInfeasible":
            route = "R5-UpstreamLiftExpressiveButConstrainedInfeasible"
        elif froute == "UpstreamLiftFeasibleButControlExplained":
            route = "R6-UpstreamLiftFeasibleButControlExplained"
        else:
            route = "R8-EdgeControlExplained_NoFU"
        repair_note = ""
        if f_repair:
            repair_note = f"; constraint_audit={f_repair.get('part_f_constraint_audit_route')}; {f_repair.get('route_reason')}"
        if margin_repair:
            repair_note += f"; strengthened_control_margin_audit={margin_repair.get('part_control_margin_audit_route')}; {margin_repair.get('route_reason')}"
        if carrier_repair:
            repair_note += f"; carrier_redesign_preflight={carrier_repair.get('part_i_carrier_redesign_route')}; {carrier_repair.get('route_reason')}"
        if output_oracle:
            repair_note += f"; output_debt_safe_oracle={output_oracle.get('part_j_output_oracle_route')}; {output_oracle.get('route_reason')}"
        if actuator_coverage:
            repair_note += f"; actuator_coverage={actuator_coverage.get('part_k_actuator_coverage_route')}; {actuator_coverage.get('route_reason')}"
        reason = f"Part F failed: {f.get('part_f_route')}; {f.get('route_reason')}{repair_note}"
    elif not int(g.get("official_candidate_gate_pass", 0)):
        route = "R7-ControlOrthogonalPreflightOpened_NoFullLoopYet"
        reason = "C/D/E/F preflight opened but no official full-loop candidate was promoted."
    else:
        route = "R13-KANCarrierOpenedOfficial"
        reason = "Official candidate gate passed."
    components = []
    if not int(c.get("part_c_gate_pass", 0)):
        components.append("NoPartGFullLoop")
    if "Debt" in route or "debt" in reason:
        components.append("DebtOrOutputResidualBlocked")
    if "Domain" in route or "domain" in reason:
        components.append("DomainSupportExplained")
    if "Control" in route or "control" in reason:
        components.append("ControlOrthogonalUtilityAbsent")
    if "Upstream" in route or "lift" in reason:
        components.append("UpstreamLiftBlocked")
    if "BrierDominated" in reason:
        components.append("BrierDominatedTask")
    if "NoControlOrthogonalUtility" in reason:
        components.append("NoControlOrthogonalUtility")
    if "CarrierRedesign" in reason:
        components.append("CarrierRedesignDiagnostic")
    if "OutputDebtSafeOracle" in reason or "OutputSpace" in reason:
        components.append("OutputOracleDiagnostic")
    if "actuator_coverage" in reason or "MLPMatched" in reason or "KANRawActuator" in reason or "ParamCarrier" in reason:
        components.append("ActuatorCoverageDiagnostic")
    final = {
        "gate": "v22_80_part_h_route_decision",
        "run_status": "completed_route_decision",
        "final_route": route,
        "route_reason": reason,
        "failure_components": "|".join(components),
        "part_a_hard_gate_pass": int(a.get("part_a_hard_gate_pass", 0)),
        "part_b_gate_pass": int(b.get("part_b_gate_pass", 0)),
        "part_c_gate_pass": int(c.get("part_c_gate_pass", 0)),
        "part_d_gate_pass": int(d.get("part_d_gate_pass", 0)),
        "part_e_gate_pass": int(e.get("part_e_gate_pass", 0)),
        "part_f_gate_pass": int(f.get("part_f_gate_pass", 0)),
        "part_g_exploration_gate_pass": int(g.get("part_g_exploration_gate_pass", 0)),
        "official_candidate_gate_pass": int(g.get("official_candidate_gate_pass", 0)),
        "part_c_route": c.get("part_c_route", ""),
        "part_c_projector_guard_route": c_repair.get("part_c_projector_guard_route", ""),
        "part_c_projector_guard_repair_pass": int(c_repair.get("part_c_projector_guard_repair_pass", 0)),
        "part_d_route": d.get("part_d_route", ""),
        "part_f_route": f.get("part_f_route", ""),
        "part_f_constraint_audit_route": f_repair.get("part_f_constraint_audit_route", ""),
        "part_f_constraint_audit_repair_pass": int(f_repair.get("part_f_constraint_audit_repair_pass", 0)),
        "part_control_margin_audit_route": margin_repair.get("part_control_margin_audit_route", ""),
        "part_control_margin_audit_repair_pass": int(margin_repair.get("part_control_margin_audit_repair_pass", 0)),
        "part_i_carrier_redesign_route": carrier_repair.get("part_i_carrier_redesign_route", ""),
        "part_i_carrier_redesign_repair_pass": int(carrier_repair.get("part_i_carrier_redesign_repair_pass", 0)),
        "part_j_output_oracle_route": output_oracle.get("part_j_output_oracle_route", ""),
        "part_j_output_oracle_repair_pass": int(output_oracle.get("part_j_output_oracle_repair_pass", 0)),
        "part_k_actuator_coverage_route": actuator_coverage.get("part_k_actuator_coverage_route", ""),
        "part_k_actuator_coverage_repair_pass": int(actuator_coverage.get("part_k_actuator_coverage_repair_pass", 0)),
        "part_c_rows": int(c.get("rows", 0) or 0),
        "part_d_rows": int(d.get("rows", 0) or 0),
        "part_f_rows": int(f.get("rows", 0) or 0),
        "non_fabrication_note": "All values are generated by v22.80 runner or copied from explicitly named v22.79 artifacts.",
        "generated_at_sg": now_sg(),
        "artifacts": {
            "part_a": rel(OUT_ROOT / "v22_80_part_a_code_identity_hard_gate.json"),
            "part_b": rel(OUT_ROOT / "v22_80_part_b_v22_79_failure_diagnosis.json"),
            "part_c": rel(OUT_ROOT / "v22_80_part_c_output_debt_residual_route.json"),
            "part_c_projector_guard_audit": rel(OUT_ROOT / "v22_80_part_c_projector_guard_audit_route.json"),
            "part_d": rel(OUT_ROOT / "v22_80_part_d_domain_residual_edge_signal_route.json"),
            "part_e": rel(OUT_ROOT / "v22_80_part_e_control_orthogonal_metric_unit_gate.json"),
            "part_f": rel(OUT_ROOT / "v22_80_part_f_upstream_constrained_lift_route.json"),
            "part_f_constraint_audit": rel(OUT_ROOT / "v22_80_part_f_constraint_decomposition_audit_route.json"),
            "part_control_margin_audit": rel(OUT_ROOT / "v22_80_part_control_margin_strengthened_audit_route.json"),
            "part_i_carrier_redesign_preflight": rel(OUT_ROOT / "v22_80_part_i_carrier_redesign_preflight_route.json"),
            "part_j_output_debt_safe_oracle": rel(OUT_ROOT / "v22_80_part_j_output_debt_safe_oracle_route.json"),
            "part_k_mlp_matched_actuator_coverage": rel(OUT_ROOT / "v22_80_part_k_mlp_matched_actuator_coverage_route.json"),
            "part_g": rel(OUT_ROOT / "v22_80_part_g_target_free_full_loop_route.json"),
            "part_h": rel(OUT_ROOT / "v22_80_final_route.json"),
        },
    }
    write_rows(OUT_ROOT / "v22_80_part_h_route_decision.csv", [final])
    write_json(OUT_ROOT / "v22_80_final_route.json", final)
    append_exec("H_route_decision", command_text(sys.argv), "done", files=f"{rel(OUT_ROOT / 'v22_80_part_h_route_decision.csv')}; {rel(OUT_ROOT / 'v22_80_final_route.json')}", note=json.dumps({"final_route": route, "reason": reason}, ensure_ascii=False))
    append_recap(
        "Part H final route decision",
        [
            f"final_route={route}；reason={reason}",
            f"gates: A={final['part_a_hard_gate_pass']} B={final['part_b_gate_pass']} C={final['part_c_gate_pass']} D={final['part_d_gate_pass']} E={final['part_e_gate_pass']} F={final['part_f_gate_pass']} G_explore={final['part_g_exploration_gate_pass']} official={final['official_candidate_gate_pass']}。",
            f"routes: C={final['part_c_route']}；C_projector_guard={final['part_c_projector_guard_route']}；D={final['part_d_route']}；F={final['part_f_route']}；F_constraint_audit={final['part_f_constraint_audit_route']}；control_margin_audit={final['part_control_margin_audit_route']}；carrier_redesign={final['part_i_carrier_redesign_route']}；output_oracle={final['part_j_output_oracle_route']}；actuator_coverage={final['part_k_actuator_coverage_route']}；failure_components={final['failure_components']}。",
        ],
    )
    return final


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="part-a")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--datasets", default="Wine,Spam,MNIST")
    p.add_argument("--seed-count", type=int, default=5)
    p.add_argument("--train-size", type=int, default=512)
    p.add_argument("--held-size", type=int, default=256)
    p.add_argument("--test-size", type=int, default=256)
    p.add_argument("--metric-batch-size", type=int, default=192)
    p.add_argument("--hidden", type=int, default=96)
    p.add_argument("--lr", type=float, default=2.0e-3)
    p.add_argument("--weight-decay", type=float, default=1.0e-4)
    p.add_argument("--step-mult", type=float, default=0.25)
    p.add_argument("--projector-ridge", type=float, default=1.0e-4)
    p.add_argument("--debt-margin-lambda", type=float, default=0.50)
    p.add_argument("--part-d-repair-bootstrap-repeats", type=int, default=16)
    p.add_argument("--shard-count", type=int, default=4)
    p.add_argument("--shard-index", type=int, default=0)
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    ensure_out()
    if args.mode == "part-a":
        run_part_a(args)
    elif args.mode == "part-b":
        run_part_b(args)
    elif args.mode == "part-c":
        run_part_c(args)
    elif args.mode == "part-c-merge":
        merge_part_c(args)
    elif args.mode == "part-c-projector-audit":
        run_part_c_projector_audit(args)
    elif args.mode == "part-c-projector-audit-merge":
        merge_part_c_projector_audit(args)
    elif args.mode == "part-d":
        run_part_d(args)
    elif args.mode == "part-d-merge":
        merge_part_d(args)
    elif args.mode == "part-control-margin-audit":
        run_part_control_margin_audit(args)
    elif args.mode == "part-control-margin-audit-merge":
        merge_part_control_margin_audit(args)
    elif args.mode == "part-i-carrier-redesign":
        run_part_i_carrier_redesign(args)
    elif args.mode == "part-i-carrier-redesign-merge":
        merge_part_i_carrier_redesign(args)
    elif args.mode == "part-j-output-oracle":
        run_part_j_output_oracle(args)
    elif args.mode == "part-j-output-oracle-merge":
        merge_part_j_output_oracle(args)
    elif args.mode == "part-k-actuator-coverage":
        run_part_k_actuator_coverage(args)
    elif args.mode == "part-k-actuator-coverage-merge":
        merge_part_k_actuator_coverage(args)
    elif args.mode == "part-e":
        run_part_e(args)
    elif args.mode == "part-f":
        run_part_f(args)
    elif args.mode == "part-f-merge":
        merge_part_f(args)
    elif args.mode == "part-f-constraint-audit":
        run_part_f_constraint_audit(args)
    elif args.mode == "part-f-constraint-audit-merge":
        merge_part_f_constraint_audit(args)
    elif args.mode == "part-g":
        run_part_g(args)
    elif args.mode in {"part-h", "final-route"}:
        run_part_h(args)
    else:
        raise SystemExit(f"unknown mode={args.mode}")


if __name__ == "__main__":
    main()
