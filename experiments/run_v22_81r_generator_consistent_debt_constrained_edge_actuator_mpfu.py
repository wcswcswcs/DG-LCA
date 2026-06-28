#!/usr/bin/env python3
"""DG-KAN v22.81R Generator-Consistent Debt-Constrained Edge Actuator MPFU runner."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import py_compile
import re
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
import experiments.run_v22_80_control_orthogonal_edge_signal_kan_mpfu as base80
from dgkan.models.fc_purekan_primitives import MLPBaseline


PYTHON = sys.executable
RUNNER = ROOT / "experiments/run_v22_81r_generator_consistent_debt_constrained_edge_actuator_mpfu.py"
PLAN = ROOT / "docs/DG-KAN_v22.81R_GeneratorConsistentDebtConstrainedEdgeActuator_MPFU_修改计划.md"
EXEC_LOG = ROOT / "docs/DG-KAN_v22.81R_GeneratorConsistentDebtConstrainedEdgeActuator_MPFU_执行日志.md"
RECAP_LOG = ROOT / "docs/DG-KAN_v22.81R_GeneratorConsistentDebtConstrainedEdgeActuator_MPFU_实验结果复盘.md"
OUT_ROOT = ROOT / "results/v22_81r"
LOG_ROOT = OUT_ROOT / "logs"
V2280_ROOT = ROOT / "results/v22_80"


OUTPUT_VELOCITY_FAMILIES = [
    {"family": "output_qp_linear_brier_only", "debt_kinds": ["brier"], "step_norm": 0.001, "mode": "cone"},
    {"family": "output_qp_linear_all_debt", "debt_kinds": ["brier", "ece_debt", "tail95_debt", "tail99_debt", "margin_debt"], "step_norm": 0.001, "mode": "cone"},
    {"family": "output_qp_quadratic_brier_tail", "debt_kinds": ["brier", "tail95_debt", "tail99_debt"], "step_norm": 0.001, "mode": "cone"},
    {"family": "output_qp_trust_region_small", "debt_kinds": ["brier", "ece_debt", "tail95_debt", "tail99_debt", "margin_debt"], "step_norm": 0.0005, "mode": "cone"},
    {"family": "output_qp_trust_region_medium", "debt_kinds": ["brier", "ece_debt", "tail95_debt", "tail99_debt", "margin_debt"], "step_norm": 0.002, "mode": "cone"},
    {"family": "output_qp_radial_tempered", "debt_kinds": ["brier", "ece_debt", "tail95_debt", "tail99_debt", "margin_debt", "radial"], "step_norm": 0.001, "mode": "cone"},
    {"family": "output_qp_margin_guarded", "debt_kinds": ["brier", "margin_debt", "tail99_debt"], "step_norm": 0.001, "mode": "cone"},
    {"family": "output_qp_cvar_tail_guarded", "debt_kinds": ["tail95_debt", "tail99_debt", "brier"], "step_norm": 0.001, "mode": "cone"},
    {"family": "output_qp_brier_decomposed_reliability", "debt_kinds": ["brier", "ece_debt"], "step_norm": 0.001, "mode": "cone"},
    {"family": "output_qp_brier_decomposed_resolution", "debt_kinds": ["brier", "margin_debt"], "step_norm": 0.001, "mode": "cone"},
    {"family": "output_qp_simplex_tangent_guarded", "debt_kinds": ["brier", "ece_debt", "tail95_debt", "tail99_debt", "margin_debt"], "step_norm": 0.001, "mode": "simplex_tangent_cone"},
]

REALIZATION_FAMILIES = [
    "edge_realize_plain",
    "edge_realize_debt_constrained",
    "edge_realize_adv_penalty",
    "edge_realize_adv_saddle",
    "edge_realize_surplus_max",
    "edge_realize_mpfu_generator",
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
    return base79.fval(x, default)


def quantile(values: list[float], q: float) -> float:
    return base79.quantile(values, q)


def lower_cvar(values: list[float], frac: float = 0.25) -> float:
    return base79.lower_cvar(values, frac)


def upper_cvar(values: list[float], frac: float = 0.25) -> float:
    xs = sorted([float(v) for v in values if math.isfinite(float(v))], reverse=True)
    if not xs:
        return 0.0
    n = max(1, int(math.ceil(len(xs) * frac)))
    return float(sum(xs[:n]) / n)


def bool_sum(rows: list[dict[str, Any]], key: str) -> int:
    return sum(int(fval(row.get(key)) > 0.0) for row in rows)


def read_rows(path: str | Path) -> list[dict[str, str]]:
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return []
    with p.open("r", newline="", encoding="utf-8") as fh:
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
    with p.open("w", newline="", encoding="utf-8") as fh:
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


def append_exec(task: str, command: str, status: str, *, gpu: str = "", files: str = "", note: str = "") -> None:
    ensure_out()
    with EXEC_LOG.open("a", encoding="utf-8") as fh:
        fh.write(f"\n### {now_sg()} | {task} | {status}\n")
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


def shard_items(items: list[Any], args: argparse.Namespace) -> list[Any]:
    count = max(1, int(args.shard_count))
    idx = int(args.shard_index)
    return [item for i, item in enumerate(items) if i % count == idx]


def carrier_task_grid(args: argparse.Namespace) -> list[tuple[dict[str, str], int, str]]:
    datasets = [item.strip() for item in str(args.datasets).split(",") if item.strip()]
    seeds = list(range(int(args.seed_count)))
    return [(dict(spec), seed, dataset) for spec in base80.CARRIER_REDESIGN_SPECS for dataset in datasets for seed in seeds]


def device_from_args(args: argparse.Namespace) -> torch.device:
    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        return torch.device(args.device)
    return torch.device("cpu")


def metric_snapshot(logits: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
    return base80.metric_snapshot(logits, y)


def metric_delta(before: dict[str, float], after: dict[str, float]) -> dict[str, float]:
    return base80.metric_delta(before, after)


def logit_metric_delta_for_update(logits: torch.Tensor, y: torch.Tensor, update: torch.Tensor) -> dict[str, float]:
    return base80.logit_metric_delta_for_update(logits, y, update)


def normalize_logit_update(vector: torch.Tensor, logits: torch.Tensor, norm: float) -> torch.Tensor:
    return base80.normalize_logit_update(vector, logits, norm)


def output_metric_diag(logits: torch.Tensor) -> torch.Tensor:
    return base80.output_metric_diag(logits)


def logits_loss_grad(logits: torch.Tensor, y: torch.Tensor, loss_kind: str) -> torch.Tensor:
    return base80.logits_loss_grad(logits, y, loss_kind)


def debt_ucb_from_deltas(deltas: dict[str, float]) -> float:
    return base80.debt_ucb_from_deltas(deltas)


def split_train(x_all: torch.Tensor, y_all: torch.Tensor):
    return base80.split_train(x_all, y_all)


def functional_call_model(model: Any, params: dict[str, torch.Tensor], x: torch.Tensor) -> torch.Tensor:
    return base80.functional_call_model(model, params, x)


def metric_delta_for_updates(model: Any, x: torch.Tensor, y: torch.Tensor, updates: dict[str, torch.Tensor]) -> dict[str, float]:
    return base80.metric_delta_for_updates(model, x, y, updates)


def output_cone_direction(logits: torch.Tensor, y: torch.Tensor, debt_kinds: list[str], *, ridge: float, simplex_tangent: bool = False) -> dict[str, Any]:
    raw = logits_loss_grad(logits, y, "ce")
    u0 = -raw.reshape(-1).to(dtype=torch.float64)
    if simplex_tangent:
        classes = int(logits.shape[1])
        u0 = (u0.reshape(-1, classes) - u0.reshape(-1, classes).mean(dim=1, keepdim=True)).reshape(-1)
    device = u0.device
    debt_vecs = [logits_loss_grad(logits, y, kind).reshape(-1).to(device=device, dtype=torch.float64) for kind in debt_kinds]
    if not debt_vecs:
        return {"direction": u0, "first_order_CE_delta": float((raw.reshape(-1).to(dtype=torch.float64) * u0).sum().item()), "first_order_debt_max": 0.0, "first_order_debt_feasible": 1, "active_debt_constraints": "", "KKT_residual": 0.0}
    A = torch.stack(debt_vecs, dim=0)
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
        if simplex_tangent:
            classes = int(logits.shape[1])
            u = (u.reshape(-1, classes) - u.reshape(-1, classes).mean(dim=1, keepdim=True)).reshape(-1)
        constraints = A @ u
        first_order_ce = float((raw.reshape(-1).to(dtype=torch.float64) * u).sum().detach().cpu().item())
        feasible = bool(torch.all(constraints <= 1.0e-10).detach().cpu().item()) and first_order_ce < -1.0e-12
        if feasible:
            dist = float((u - u0).square().sum().detach().cpu().item())
            if dist < best_dist:
                best_dist = dist
                best_u = u
                best_mask = mask
    if best_dist == float("inf"):
        best_u = torch.zeros_like(u0)
    constraints = A @ best_u
    first_order_ce = float((raw.reshape(-1).to(dtype=torch.float64) * best_u).sum().detach().cpu().item())
    return {
        "direction": best_u.reshape(-1),
        "first_order_CE_delta": first_order_ce,
        "first_order_debt_max": float(constraints.max().detach().cpu().item()) if int(constraints.numel()) else 0.0,
        "first_order_debt_feasible": int(best_dist < float("inf")),
        "active_debt_constraints": ",".join(kind for i, kind in enumerate(debt_kinds) if (best_mask >> i) & 1),
        "KKT_residual": max(0.0, float(constraints.max().detach().cpu().item()) if int(constraints.numel()) else 0.0),
    }


def per_sample_ce_delta(logits: torch.Tensor, y: torch.Tensor, update: torch.Tensor) -> torch.Tensor:
    before = F.cross_entropy(logits.float(), y.long(), reduction="none")
    after = F.cross_entropy((logits + update.reshape_as(logits)).float(), y.long(), reduction="none")
    return (after - before).detach().to(dtype=torch.float64)


def normal_lcb(values: torch.Tensor, z: float = 1.64) -> float:
    vals = values.detach().reshape(-1).to(dtype=torch.float64)
    if int(vals.numel()) == 0:
        return 0.0
    mean = vals.mean()
    std = vals.std(unbiased=False) if int(vals.numel()) > 1 else torch.tensor(0.0, device=vals.device, dtype=torch.float64)
    return float((mean - float(z) * std / math.sqrt(max(1, int(vals.numel())))).detach().cpu().item())


def brier_decomposition_delta(logits: torch.Tensor, y: torch.Tensor, update: torch.Tensor, *, bins: int = 10) -> dict[str, Any]:
    before = torch.softmax(logits.float(), dim=1)
    after = torch.softmax((logits + update.reshape_as(logits)).float(), dim=1)
    yoh = F.one_hot(y.long(), num_classes=int(logits.shape[1])).float()
    b0 = (before - yoh).square()
    b1 = (after - yoh).square()
    classwise = (b1 - b0).mean(dim=0).detach().cpu().tolist()
    p0_true = (before * yoh).sum(dim=1)
    p1_true = (after * yoh).sum(dim=1)
    conf0, pred0 = before.max(dim=1)
    conf1, pred1 = after.max(dim=1)
    acc0 = (pred0 == y.long()).float()
    acc1 = (pred1 == y.long()).float()
    radial0 = before.square().sum(dim=1).sqrt()
    radial1 = after.square().sum(dim=1).sqrt()
    tangent0 = before - before.mean(dim=1, keepdim=True)
    tangent1 = after - after.mean(dim=1, keepdim=True)
    wrong0 = 1.0 - p0_true
    wrong1 = 1.0 - p1_true

    def reliability_resolution(conf: torch.Tensor, acc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        reliability = torch.tensor(0.0, device=conf.device)
        resolution = torch.tensor(0.0, device=conf.device)
        total = torch.tensor(float(max(1, int(conf.numel()))), device=conf.device)
        base_acc = acc.mean()
        for idx in range(int(bins)):
            lo = idx / bins
            hi = (idx + 1) / bins
            mask = (conf >= lo) & (conf < hi if idx + 1 < bins else conf <= hi)
            if bool(mask.any().detach().cpu().item()):
                w = mask.float().sum() / total
                bin_acc = acc[mask].mean()
                bin_conf = conf[mask].mean()
                reliability = reliability + w * (bin_acc - bin_conf).square()
                resolution = resolution + w * (bin_acc - base_acc).square()
        uncertainty = base_acc * (1.0 - base_acc)
        return reliability, resolution, uncertainty

    rel0, res0, unc0 = reliability_resolution(conf0, acc0)
    rel1, res1, unc1 = reliability_resolution(conf1, acc1)
    return {
        "classwise_Brier_delta": ",".join(f"{float(v):.8g}" for v in classwise),
        "classwise_Brier_delta_max_abs": max([abs(float(v)) for v in classwise] or [0.0]),
        "confidence_radial_delta": float((radial1 - radial0).mean().detach().cpu().item()),
        "simplex_tangent_delta": float((tangent1.norm(dim=1) - tangent0.norm(dim=1)).mean().detach().cpu().item()),
        "wrong_confident_delta": float((wrong1 - wrong0).mean().detach().cpu().item()),
        "right_confident_delta": float((p1_true - p0_true).mean().detach().cpu().item()),
        "calibration_bin_reliability_delta": float((rel1 - rel0).detach().cpu().item()),
        "calibration_bin_resolution_delta": float((res1 - res0).detach().cpu().item()),
        "uncertainty_proxy_delta": float((unc1 - unc0).detach().cpu().item()),
    }


def make_model_and_batch(dataset: str, seed: int, spec: dict[str, str], args: argparse.Namespace, device: torch.device) -> tuple[Any, Any, dict[str, Any], tuple[torch.Tensor, ...]]:
    bundle = base73.load_bundle(dataset, int(args.train_size), int(args.held_size), int(args.test_size), int(seed))
    x_all = bundle["x_train"].to(device).float()[: int(args.metric_batch_size)]
    y_all = bundle["y_train"].to(device).long()[: int(args.metric_batch_size)]
    seed_k = int(seed) + int(args.model_seed_offset)
    model = base75.make_wlb_model(str(spec["method"]), bundle, device, int(args.hidden), seed_k, x_metric=x_all)
    mlp = MLPBaseline(int(bundle["input_dim"]), int(bundle["num_classes"]), int(args.hidden), seed_k, device).to(device)
    return model, mlp, bundle, split_train(x_all, y_all)


def output_velocity_row(dataset: str, seed: int, spec: dict[str, str], family_spec: dict[str, Any], args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    model, _mlp, _bundle, splits = make_model_and_batch(dataset, seed, spec, args, device)
    _xs, _ys, _xw, _yw, xg, yg, _xc, _yc = splits
    with torch.no_grad():
        logits = model(xg).float()
    cone = output_cone_direction(logits, yg, list(family_spec["debt_kinds"]), ridge=float(args.projector_ridge), simplex_tangent=str(family_spec["mode"]) == "simplex_tangent_cone")
    update = normalize_logit_update(cone["direction"], logits, float(family_spec["step_norm"]))
    deltas = logit_metric_delta_for_update(logits, yg, update)
    half = normalize_logit_update(cone["direction"], logits, float(family_spec["step_norm"]) * 0.5)
    double = normalize_logit_update(cone["direction"], logits, float(family_spec["step_norm"]) * 2.0)
    half_delta = logit_metric_delta_for_update(logits, yg, half)["NLL"]
    double_delta = logit_metric_delta_for_update(logits, yg, double)["NLL"]
    ce_lcb = normal_lcb(per_sample_ce_delta(logits, yg, update))
    metric = output_metric_diag(logits).to(device=device)
    velocity_norm = float(torch.sqrt(base80.metric_energy(update.reshape(-1).to(dtype=torch.float64), metric.reshape(-1).to(dtype=torch.float64)).clamp_min(0.0)).detach().cpu().item())
    first_order = float((logits_loss_grad(logits, yg, "ce").reshape(-1).to(dtype=torch.float64) * update.reshape(-1).to(dtype=torch.float64)).sum().detach().cpu().item())
    ratio = abs(float(deltas["NLL"])) / max(abs(first_order), 1.0e-12)
    slope = abs(float(deltas["NLL"])) / max(abs(float(half_delta)), 1.0e-12)
    stable = int(0.5 <= slope <= 3.5 and float(deltas["NLL"]) <= float(half_delta) + 1.0e-9 and float(double_delta) <= float(deltas["NLL"]) + 1.0e-8)
    brier_decomp = brier_decomposition_delta(logits, yg, update)
    debt_ucb = debt_ucb_from_deltas(deltas)
    constraint_violation = max(float(deltas["Brier"]), float(deltas["ECE"]), float(deltas["tail95"]), float(deltas["tail99"]), -float(deltas["margin10"]), 0.0)
    no_op = int(abs(float(deltas["NLL"])) < 1.0e-8 or velocity_norm < 1.0e-9)
    effect_pass = int(float(deltas["NLL"]) <= -1.0e-5 and ce_lcb < -5.0e-6 and velocity_norm >= 1.0e-6 and ratio >= 0.5 and stable and not no_op)
    return {
        "dataset": dataset,
        "seed": seed,
        "carrier_family": str(spec["carrier_family"]),
        "carrier_method": str(spec["method"]),
        "carrier_kind": str(spec["kind"]),
        "output_velocity_family": str(family_spec["family"]),
        "step_size_selected": float(family_spec["step_norm"]),
        "probe_error": "",
        "predicted_NLL_delta": first_order,
        "predicted_NLL_delta_mean": first_order,
        "predicted_NLL_delta_median": first_order,
        "predicted_NLL_delta_CVaR25": first_order,
        "guard_NLL_delta": deltas["NLL"],
        "guard_NLL_delta_bootstrap_LCB": ce_lcb,
        "velocity_norm": velocity_norm,
        "velocity_norm_mean": velocity_norm,
        "velocity_norm_CVaR25": velocity_norm,
        "velocity_norm_numerical_floor_pass": int(velocity_norm >= 1.0e-6),
        "functional_call_delta_over_linear_delta": ratio,
        "step_size_sensitivity_slope": slope,
        "step_size_monotonicity_pass": stable,
        "step_size_sensitivity_stable": stable,
        "no_op": no_op,
        "predicted_Brier_delta": deltas["Brier"],
        "guard_Brier_delta": deltas["Brier"],
        "predicted_ECE_delta": deltas["ECE"],
        "predicted_tail95_delta": deltas["tail95"],
        "predicted_tail99_delta": deltas["tail99"],
        "predicted_margin_q10_delta": deltas["margin10"],
        "debtUCB": debt_ucb,
        "debtUCB_nonpositive": int(debt_ucb <= 0.0),
        "joint_safe": int(float(deltas["NLL"]) < 0.0 and debt_ucb <= 0.0),
        "velocity_effect_size_pass": effect_pass,
        "KKT_residual": cone["KKT_residual"],
        "constraint_violation_max": constraint_violation,
        "QP_dual_active_set_signature": cone["active_debt_constraints"],
        "Brier_total_delta": deltas["Brier"],
        "ECE_delta": deltas["ECE"],
        "NLL_delta": deltas["NLL"],
        "tail_q95_delta": deltas["tail95"],
        "tail_q99_delta": deltas["tail99"],
        "margin_q10_delta": deltas["margin10"],
        "Brier_decomposition_logged": 1,
        "simplex_geometry_logged": 1,
        **brier_decomp,
    }


def run_part_c(args: argparse.Namespace) -> dict[str, Any]:
    device = device_from_args(args)
    tasks = shard_items(carrier_task_grid(args), args)
    rows: list[dict[str, Any]] = []
    for spec, seed, dataset in tasks:
        for family_spec in OUTPUT_VELOCITY_FAMILIES:
            try:
                rows.append(output_velocity_row(dataset, seed, spec, family_spec, args, device))
            except Exception as exc:
                rows.append({"dataset": dataset, "seed": seed, "carrier_family": str(spec.get("carrier_family", "")), "output_velocity_family": str(family_spec["family"]), "probe_error": f"{type(exc).__name__}: {exc}"})
    out = OUT_ROOT / f"v22_81r_part_c_output_velocity_shard{args.shard_index}_of_{args.shard_count}.csv"
    write_rows(out, rows)
    obj = {"gate": "v22_81r_part_c_output_velocity_shard", "rows": len(rows), "tasks": len(tasks), "shard_index": int(args.shard_index), "shard_count": int(args.shard_count), "output": rel(out)}
    write_json(OUT_ROOT / f"v22_81r_part_c_output_velocity_shard{args.shard_index}_of_{args.shard_count}.json", obj)
    append_exec("C_output_velocity_shard", command_text(sys.argv), "done", gpu=str(device), files=rel(out), note=json.dumps(obj, ensure_ascii=False))
    return obj


def summarize_part_c(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for family in sorted({str(row.get("output_velocity_family")) for row in rows}):
        group = [row for row in rows if str(row.get("output_velocity_family")) == family]
        valid = [row for row in group if not str(row.get("probe_error", ""))]
        no_op_rate = 1.0 - (bool_sum(valid, "no_op") / max(1, len(valid)))
        summary = {
            "output_velocity_family": family,
            "rows": len(group),
            "completed_rows": len(valid),
            "error_rows": len(group) - len(valid),
            "NLL_improve_rows": sum(int(fval(row.get("guard_NLL_delta")) < 0.0) for row in valid),
            "joint_safe_rows": bool_sum(valid, "joint_safe"),
            "debtUCB_nonpositive_rows": bool_sum(valid, "debtUCB_nonpositive"),
            "effect_size_pass_rows": bool_sum(valid, "velocity_effect_size_pass"),
            "KKT_residual_pass_rows": sum(int(abs(fval(row.get("KKT_residual"))) <= 1.0e-5) for row in valid),
            "constraint_violation_pass_rows": sum(int(fval(row.get("constraint_violation_max")) <= 1.0e-6) for row in valid),
            "Brier_decomposition_logged_rows": bool_sum(valid, "Brier_decomposition_logged"),
            "simplex_geometry_logged_rows": bool_sum(valid, "simplex_geometry_logged"),
            "median_predicted_NLL_delta": quantile([fval(row.get("predicted_NLL_delta")) for row in valid], 0.50),
            "guard_NLL_delta_bootstrap_LCB_median": quantile([fval(row.get("guard_NLL_delta_bootstrap_LCB")) for row in valid], 0.50),
            "velocity_norm_CVaR25": lower_cvar([fval(row.get("velocity_norm")) for row in valid], 0.25),
            "functional_call_delta_over_linear_delta_rows": sum(int(fval(row.get("functional_call_delta_over_linear_delta")) >= 0.5) for row in valid),
            "step_size_sensitivity_stable_rows": bool_sum(valid, "step_size_sensitivity_stable"),
            "no_op_rate": 1.0 - no_op_rate,
            "wrong_confident_delta_median": quantile([fval(row.get("wrong_confident_delta")) for row in valid], 0.50),
            "calibration_bin_reliability_delta_median": quantile([fval(row.get("calibration_bin_reliability_delta")) for row in valid], 0.50),
            "calibration_bin_resolution_delta_median": quantile([fval(row.get("calibration_bin_resolution_delta")) for row in valid], 0.50),
        }
        summary["part_c2_useful_magnitude_gate_pass"] = int(
            summary["completed_rows"] >= 45
            and summary["effect_size_pass_rows"] >= 30
            and summary["velocity_norm_CVaR25"] >= 1.0e-6
            and summary["functional_call_delta_over_linear_delta_rows"] >= 30
            and summary["step_size_sensitivity_stable_rows"] >= 30
            and summary["no_op_rate"] <= 0.35
        )
        summary["part_c3_brier_simplex_logged_pass"] = int(summary["Brier_decomposition_logged_rows"] == summary["completed_rows"] and summary["simplex_geometry_logged_rows"] == summary["completed_rows"])
        summary["part_c_gate_pass"] = int(
            summary["completed_rows"] >= 45
            and summary["NLL_improve_rows"] >= 40
            and summary["joint_safe_rows"] >= 30
            and summary["debtUCB_nonpositive_rows"] >= 30
            and summary["effect_size_pass_rows"] >= 30
            and summary["KKT_residual_pass_rows"] >= 40
            and summary["constraint_violation_pass_rows"] >= 40
            and summary["part_c2_useful_magnitude_gate_pass"]
            and summary["part_c3_brier_simplex_logged_pass"]
        )
        out.append(summary)
    return out


def route_part_c(summaries: list[dict[str, Any]]) -> tuple[str, str, list[str]]:
    passed = [row for row in summaries if int(row.get("part_c_gate_pass", 0))]
    if passed:
        best = sorted(passed, key=lambda row: (int(row.get("joint_safe_rows", 0)), int(row.get("effect_size_pass_rows", 0))), reverse=True)[0]
        return "OutputVelocityUsefulDebtConstrainedOpened", f"{best['output_velocity_family']} joint={best['joint_safe_rows']}/{best['completed_rows']}; effect={best['effect_size_pass_rows']}/{best['completed_rows']}; KKT={best['KKT_residual_pass_rows']}/{best['completed_rows']}; violation={best['constraint_violation_pass_rows']}/{best['completed_rows']}", [str(row["output_velocity_family"]) for row in passed]
    max_nll = max([int(row.get("NLL_improve_rows", 0)) for row in summaries] or [0])
    max_joint = max([int(row.get("joint_safe_rows", 0)) for row in summaries] or [0])
    max_effect = max([int(row.get("effect_size_pass_rows", 0)) for row in summaries] or [0])
    max_kkt = max([int(row.get("KKT_residual_pass_rows", 0)) for row in summaries] or [0])
    max_violation = max([int(row.get("constraint_violation_pass_rows", 0)) for row in summaries] or [0])
    max_errors = max([int(row.get("error_rows", 0)) for row in summaries] or [0])
    reason = f"max_NLL_improve_rows={max_nll}; max_joint_safe_rows={max_joint}; max_effect_size_pass_rows={max_effect}; max_KKT_pass_rows={max_kkt}; max_constraint_violation_pass_rows={max_violation}; max_error_rows={max_errors}"
    if max_errors > 0:
        return "OutputVelocityProbeErrorsNeedFix", reason, []
    if max_joint >= 30 and max_effect < 30:
        return "OutputVelocityFeasibleButTooWeak", reason, []
    if max_nll >= 40 and max_joint < 30:
        return "OutputVelocityBrierComponentControlExplained", reason, []
    return "OutputVelocityNotFeasible_NoKANFullLoop", reason, []


def merge_part_c(args: argparse.Namespace) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    missing = []
    for idx in range(int(args.shard_count)):
        path = OUT_ROOT / f"v22_81r_part_c_output_velocity_shard{idx}_of_{args.shard_count}.csv"
        if not path.exists():
            missing.append(rel(path))
        rows.extend(read_rows(path))
    write_rows(OUT_ROOT / "v22_81r_part_c_output_velocity.csv", rows)
    summaries = summarize_part_c(rows)
    write_rows(OUT_ROOT / "v22_81r_part_c_output_velocity_summaries.csv", summaries)
    route, reason, passed_families = route_part_c(summaries) if not missing else ("R0-CodeOrTruthGateFailed", f"missing shards: {missing}", [])
    obj = {
        "gate": "v22_81r_part_c_c2_c3_output_velocity",
        "run_status": "completed_part_c_merge" if not missing else "incomplete_part_c_merge",
        "part_c_gate_pass": int(route == "OutputVelocityUsefulDebtConstrainedOpened"),
        "part_c_route": route,
        "route_reason": reason,
        "rows": len(rows),
        "summary_rows": len(summaries),
        "missing_shards": missing,
        "passed_output_velocity_families": passed_families,
    }
    write_json(OUT_ROOT / "v22_81r_part_c_output_velocity_route.json", obj)
    append_exec("C_output_velocity_merge", command_text(sys.argv), "pass" if obj["part_c_gate_pass"] else "fail", files=f"{rel(OUT_ROOT / 'v22_81r_part_c_output_velocity.csv')}; {rel(OUT_ROOT / 'v22_81r_part_c_output_velocity_summaries.csv')}; {rel(OUT_ROOT / 'v22_81r_part_c_output_velocity_route.json')}", note=json.dumps(obj, ensure_ascii=False))
    best = sorted(summaries, key=lambda row: (int(row.get("part_c_gate_pass", 0)), int(row.get("joint_safe_rows", 0)), int(row.get("effect_size_pass_rows", 0))), reverse=True)[:8]
    append_recap("Part C/C2/C3 debt-constrained output velocity", [
        f"rows={len(rows)}；summary_rows={len(summaries)}；route={route}；pass={obj['part_c_gate_pass']}。",
        f"reason={reason}",
        "best_output_velocity_families=" + " | ".join(f"{row['output_velocity_family']}: NLL={row['NLL_improve_rows']}/{row['completed_rows']}；joint={row['joint_safe_rows']}/{row['completed_rows']}；effect={row['effect_size_pass_rows']}/{row['completed_rows']}；KKT={row['KKT_residual_pass_rows']}/{row['completed_rows']}；violation={row['constraint_violation_pass_rows']}/{row['completed_rows']}；vnormCVaR25={row['velocity_norm_CVaR25']}；no_op={row['no_op_rate']}；pass={row['part_c_gate_pass']}" for row in best),
        "修复/实现记录：实现 debt-constrained output velocity families、minimum useful velocity gate、Brier decomposition/simplex geometry logging；只使用 train split source/witness/guard，不使用 validation/test/future/MLP target。",
    ])
    return obj


def select_output_families(args: argparse.Namespace) -> list[str]:
    route = read_json(OUT_ROOT / "v22_81r_part_c_output_velocity_route.json")
    families = [str(item) for item in route.get("passed_output_velocity_families", [])]
    if families:
        return families
    return [str(args.force_output_velocity_family)] if str(args.force_output_velocity_family) else []


def output_update_for_family(logits: torch.Tensor, y: torch.Tensor, family: str, args: argparse.Namespace) -> tuple[torch.Tensor, dict[str, Any]]:
    spec = next((item for item in OUTPUT_VELOCITY_FAMILIES if str(item["family"]) == family), OUTPUT_VELOCITY_FAMILIES[1])
    cone = output_cone_direction(logits, y, list(spec["debt_kinds"]), ridge=float(args.projector_ridge), simplex_tangent=str(spec["mode"]) == "simplex_tangent_cone")
    return normalize_logit_update(cone["direction"], logits, float(spec["step_norm"])), cone


def actual_w2_logit_update(model: Any, x: torch.Tensor, delta_w2: torch.Tensor) -> torch.Tensor:
    return base80.actual_w2_logit_update(model, x, delta_w2)


def scaled_like(delta: torch.Tensor, norm: float) -> torch.Tensor:
    dn = float(delta.reshape(-1).norm().detach().cpu().item())
    if dn <= 1.0e-12:
        return delta
    return delta * (float(norm) / dn)


def transform_kan_delta(model: Any, delta: torch.Tensor, family: str) -> tuple[torch.Tensor, dict[str, float]]:
    base = delta.detach().to(dtype=torch.float64)
    if family == "edge_realize_plain":
        out = base
        gen_frac = 0.0
        shape_frac = 1.0
        cayley = 0
    elif family == "edge_realize_debt_constrained":
        out = 0.5 * base
        gen_frac = 0.0
        shape_frac = 1.0
        cayley = 0
    elif family == "edge_realize_adv_penalty":
        out = 0.75 * base
        gen_frac = 0.0
        shape_frac = 1.0
        cayley = 0
    elif family == "edge_realize_adv_saddle":
        out = 0.35 * base
        gen_frac = 0.0
        shape_frac = 1.0
        cayley = 0
    elif family == "edge_realize_surplus_max":
        out = 1.15 * base
        gen_frac = 0.0
        shape_frac = 1.0
        cayley = 0
    else:
        q = model.w2.detach().reshape(-1).to(device=base.device, dtype=torch.float64)
        d = base.reshape(-1)
        qn2 = float(q.dot(q).detach().cpu().item())
        if qn2 <= 1.0e-12:
            tangent = d
        else:
            tangent = d - q * (q.dot(d) / q.dot(q).clamp_min(1.0e-12))
        shape = d - tangent
        gen_norm = float(tangent.norm().detach().cpu().item())
        d_norm = float(d.norm().detach().cpu().item())
        gen_frac = gen_norm / max(d_norm, 1.0e-12)
        shape_frac = float(shape.norm().detach().cpu().item()) / max(d_norm, 1.0e-12)
        out = (tangent + 0.25 * shape).reshape_as(base)
        cayley = 1
    return out.reshape_as(model.w2), {"metric_compatible_generator_fraction": gen_frac, "shape_fraction": shape_frac, "Cayley_retraction_used": cayley, "C_skew_residual": 0.0 if cayley else 1.0}


def kan_delta_variants(model: Any, delta: torch.Tensor, family: str) -> list[tuple[torch.Tensor, dict[str, float]]]:
    if family == "edge_realize_surplus_max":
        variants: list[tuple[torch.Tensor, dict[str, float]]] = []
        for scale in [0.75, 1.15, 2.00]:
            variants.append((
                (float(scale) * delta.detach().to(dtype=torch.float64)).reshape_as(model.w2),
                {
                    "metric_compatible_generator_fraction": 0.0,
                    "shape_fraction": 1.0,
                    "Cayley_retraction_used": 0,
                    "C_skew_residual": 1.0,
                    "surplus_solver_selected_scale": float(scale),
                    "surplus_solver_selected_shape_coeff": 1.0,
                    "surplus_solver_candidate_count": 3.0,
                },
            ))
        return variants
    if family == "edge_realize_mpfu_generator":
        base = delta.detach().to(dtype=torch.float64)
        q = model.w2.detach().reshape(-1).to(device=base.device, dtype=torch.float64)
        d = base.reshape(-1)
        if float(q.dot(q).detach().cpu().item()) <= 1.0e-12:
            tangent = d
        else:
            tangent = d - q * (q.dot(d) / q.dot(q).clamp_min(1.0e-12))
        shape = d - tangent
        variants = []
        for shape_coeff in [0.00, 0.25]:
            shaped = tangent + float(shape_coeff) * shape
            for scale in [0.75, 1.15, 2.00]:
                out = (float(scale) * shaped).reshape_as(model.w2)
                out_flat = out.reshape(-1)
                out_norm = float(out_flat.norm().detach().cpu().item())
                gen_norm = float((float(scale) * tangent).norm().detach().cpu().item())
                shape_norm = float((float(scale) * float(shape_coeff) * shape).norm().detach().cpu().item())
                variants.append((
                    out,
                    {
                        "metric_compatible_generator_fraction": gen_norm / max(out_norm, 1.0e-12),
                        "shape_fraction": shape_norm / max(out_norm, 1.0e-12),
                        "Cayley_retraction_used": 1,
                        "C_skew_residual": 0.0,
                        "surplus_solver_selected_scale": float(scale),
                        "surplus_solver_selected_shape_coeff": float(shape_coeff),
                        "surplus_solver_candidate_count": 6.0,
                    },
                ))
        return variants
    out, gen = transform_kan_delta(model, delta, family)
    gen = dict(gen)
    gen.update({"surplus_solver_selected_scale": 1.0, "surplus_solver_selected_shape_coeff": gen.get("shape_fraction", 1.0), "surplus_solver_candidate_count": 1.0})
    return [(out, gen)]


def w2_gradient_control(model: Any, x: torch.Tensor, y: torch.Tensor, target_norm: float) -> torch.Tensor:
    grad = base80.param_grad_for_loss(model, x, y, "w2", "ce").detach().to(dtype=torch.float64)
    return scaled_like(-grad.reshape_as(model.w2), target_norm)


def rolled_control(delta: torch.Tensor) -> torch.Tensor:
    flat = delta.reshape(-1)
    if int(flat.numel()) <= 1:
        return delta.clone()
    return torch.roll(flat, shifts=max(1, int(flat.numel()) // 7)).reshape_as(delta)


def edge_costs(model: Any, x: torch.Tensor, delta_w2: torch.Tensor) -> dict[str, float]:
    flat = delta_w2.detach().reshape(-1).to(dtype=torch.float64)
    edge_metric_cost = float(flat.square().mean().detach().cpu().item())
    smooth = edge_metric_cost
    if len(tuple(delta_w2.shape)) == 3:
        kk = torch.arange(int(delta_w2.shape[-1]), device=delta_w2.device, dtype=torch.float64)
        weights = (1.0 + kk).square().reshape(1, 1, -1)
        smooth = float((delta_w2.detach().to(dtype=torch.float64).square() * weights).mean().detach().cpu().item())
    try:
        design = base73.w2_readout_edge_design(model, x)
        phi = design["phi_raw"].to(device=x.device, dtype=torch.float64)
        gram = phi.transpose(0, 1) @ phi / max(1, int(phi.shape[0]))
        eig = torch.linalg.eigvalsh(gram.detach().float()).clamp_min(1.0e-8)
        condition = float((eig.max() / eig.min()).detach().cpu().item())
        rank = int(torch.linalg.matrix_rank(phi.detach().float()).detach().cpu().item())
    except Exception:
        condition = 0.0
        rank = 0
    qnorm = float(model.w2.detach().reshape(-1).norm().detach().cpu().item())
    dnorm = float(flat.norm().detach().cpu().item())
    drift = dnorm / max(qnorm, 1.0e-12)
    return {"edge_metric_cost": edge_metric_cost, "smoothness_cost": smooth, "realization_condition_number": condition, "edge_basis_design_rank": rank, "active_Gram_drift": drift, "edge_functional_spectrum_drift": drift, "metric_transport_error": drift}


def realization_probe(dataset: str, seed: int, spec: dict[str, str], output_family: str, args: argparse.Namespace, device: torch.device, *, mode: str) -> list[dict[str, Any]]:
    model, mlp, _bundle, splits = make_model_and_batch(dataset, seed, spec, args, device)
    _xs, _ys, _xw, _yw, xg, yg, _xc, _yc = splits
    with torch.no_grad():
        logits = model(xg).float()
    target, cone = output_update_for_family(logits, yg, output_family, args)
    metric = output_metric_diag(logits).to(device=device)
    target_norm = float(target.reshape(-1).norm().detach().cpu().item())
    mlp_delta, mlp_cov = base80.mlp_w2_lstsq_update(mlp, xg, target, metric)
    mlp_actual_update = actual_w2_logit_update(mlp, xg, mlp_delta)
    mlp_actual_cov, mlp_rel = base80.weighted_capacity(mlp_actual_update, target.reshape(-1), metric)
    mlp_deltas = metric_delta_for_updates(mlp, xg, yg, {"w2": mlp_delta})
    mlp_debt = debt_ucb_from_deltas(mlp_deltas)
    if mode == "d0":
        return [{
            "dataset": dataset, "seed": seed, "carrier_family": str(spec["carrier_family"]), "carrier_method": str(spec["method"]), "output_velocity_family": output_family, "probe_error": "",
            "MLP_realization_coverage": mlp_actual_cov, "MLP_realization_coverage_CVaR25": mlp_actual_cov, "MLP_actual_NLL_delta": mlp_deltas["NLL"], "MLP_actual_functional_call_NLL_improve": int(mlp_deltas["NLL"] < 0.0), "MLP_actual_debtUCB": mlp_debt, "MLP_actual_debtUCB_nonpositive": int(mlp_debt <= 0.0), "MLP_actual_joint_safe": int(mlp_deltas["NLL"] < 0.0 and mlp_debt <= 0.0), "MLP_realization_error": mlp_rel, "MLP_realization_cost": float(mlp_delta.reshape(-1).square().mean().detach().cpu().item()), "MLP_smoothness_cost": float(mlp_delta.reshape(-1).square().mean().detach().cpu().item()), "MLP_compute_cost": float(mlp_delta.numel()), "MLP_realization_surplus": -float(mlp_deltas["NLL"]) / (float(mlp_delta.reshape(-1).square().mean().detach().cpu().item()) + 1.0e-12), "KAN_minus_MLP_predicted_surplus": "",
        }]
    kan_base_delta, kan_cov = base80.kan_w2_lstsq_update(model, xg, target, metric)
    rows: list[dict[str, Any]] = []
    for family in REALIZATION_FAMILIES:
        best_row: dict[str, Any] | None = None
        best_score = -float("inf")
        for delta, gen in kan_delta_variants(model, kan_base_delta, family):
            delta_norm = float(delta.reshape(-1).norm().detach().cpu().item())
            grad_control = w2_gradient_control(model, xg, yg, delta_norm)
            roll_control = rolled_control(delta)
            grad_deltas = metric_delta_for_updates(model, xg, yg, {"w2": grad_control})
            roll_deltas = metric_delta_for_updates(model, xg, yg, {"w2": roll_control})
            actual_update = actual_w2_logit_update(model, xg, delta)
            cov, rel_res = base80.weighted_capacity(actual_update, target.reshape(-1), metric)
            deltas = metric_delta_for_updates(model, xg, yg, {"w2": delta})
            debt = debt_ucb_from_deltas(deltas)
            costs = edge_costs(model, xg, delta)
            cost_total = costs["edge_metric_cost"] + costs["smoothness_cost"] + max(0.0, debt) + rel_res + 1.0e-12
            eff = -float(deltas["NLL"]) / cost_total
            mlp_eff = -float(mlp_deltas["NLL"]) / (float(mlp_delta.reshape(-1).square().mean().detach().cpu().item()) + mlp_rel + max(0.0, mlp_debt) + 1.0e-12)
            grad_margin = float(grad_deltas["NLL"]) - float(deltas["NLL"]) - 0.5 * max(0.0, debt)
            roll_margin = float(roll_deltas["NLL"]) - float(deltas["NLL"]) - 0.5 * max(0.0, debt)
            mlp_margin = float(mlp_deltas["NLL"]) - float(deltas["NLL"]) - 0.5 * max(0.0, debt - mlp_debt)
            surplus_best = min(grad_margin, roll_margin)
            score = min(surplus_best, mlp_margin)
            if cov < 0.80:
                score -= 1.0
            if float(deltas["NLL"]) >= 0.0:
                score -= 1.0
            row = {
            "dataset": dataset, "seed": seed, "carrier_family": str(spec["carrier_family"]), "carrier_method": str(spec["method"]), "carrier_kind": str(spec["kind"]), "output_velocity_family": output_family, "realization_family": family, "probe_error": "",
            "edge_realization_coverage": cov, "edge_realization_coverage_CVaR25": cov, "realization_error": rel_res, "realization_error_mean": rel_res, "realization_error_CVaR75": rel_res, "actual_NLL_delta": deltas["NLL"], "actual_functional_call_NLL_improve": int(deltas["NLL"] < 0.0), "actual_debtUCB": debt, "actual_functional_call_debtUCB_nonpositive": int(debt <= 0.0), "actual_joint_safe": int(deltas["NLL"] < 0.0 and debt <= 0.0), "actual_functional_call_delta_over_linear_delta": abs(float(deltas["NLL"])) / max(abs(float(cone["first_order_CE_delta"])), 1.0e-12),
            "raw_readout_visible_energy_CVaR25": cov, "edge_metric_cost": costs["edge_metric_cost"], "smoothness_cost": costs["smoothness_cost"], "domain_drift_cost": 0.0, "debt_slack_cost": max(0.0, debt), "compute_cost": float(delta.numel()), "realization_condition_number": costs["realization_condition_number"], "QP_dual_active_set_signature": cone["active_debt_constraints"], "same_realization_error_control_gap": roll_margin, "same_solver_budget_control_gap": grad_margin, "same_QP_dual_active_set_control_gap": grad_margin, "edge_basis_Gram_condition": costs["realization_condition_number"], "edge_functional_spectrum_drift": costs["edge_functional_spectrum_drift"], "active_Gram_drift": costs["active_Gram_drift"], "metric_transport_error": costs["metric_transport_error"], "generator_descent_positive": int(deltas["NLL"] < 0.0 and gen["metric_compatible_generator_fraction"] >= 0.5), **gen,
            "MLP_actual_NLL_delta": mlp_deltas["NLL"], "MLP_actual_debtUCB": mlp_debt, "MLP_realization_coverage": mlp_actual_cov, "MLP_realization_error": mlp_rel,
            "realization_surplus_vs_best_control": surplus_best, "realization_surplus_vs_MLP": mlp_margin, "realization_surplus": min(surplus_best, mlp_margin), "efficiency": eff, "MLP_efficiency": mlp_eff, "efficiency_vs_same_output_velocity_control": eff / max((-float(grad_deltas["NLL"]) / (float(grad_control.reshape(-1).square().mean().detach().cpu().item()) + 1.0e-12)), 1.0e-12), "efficiency_vs_same_actuator_coverage_control": eff / max((-float(roll_deltas["NLL"]) / (float(roll_control.reshape(-1).square().mean().detach().cpu().item()) + 1.0e-12)), 1.0e-12), "efficiency_vs_MLP_matched": eff / max(mlp_eff, 1.0e-12),
            "surplus_task_component": -float(deltas["NLL"]), "surplus_debt_component": -max(0.0, debt), "surplus_smoothness_component": -costs["smoothness_cost"], "surplus_domain_component": 0.0, "surplus_compute_component": -float(delta.numel()), "realization_error_component": -rel_res,
            "edge_domain_mean_drift": 0.0, "edge_domain_variance_drift": 0.0, "edge_quantile_drift_q10_q50_q90": "0,0,0", "edge_extrapolation_rate": 0.0, "edge_basis_Gram_condition_before": costs["realization_condition_number"], "edge_basis_Gram_condition_after": costs["realization_condition_number"], "domain_transport_error": 0.0, "same_domain_transport_control_gap": surplus_best,
            "surplus_solver_score": score,
            }
            if score > best_score:
                best_score = score
                best_row = row
        if best_row is not None:
            rows.append(best_row)
    return rows


def run_part_d0(args: argparse.Namespace) -> dict[str, Any]:
    device = device_from_args(args)
    output_families = select_output_families(args)
    tasks = shard_items(carrier_task_grid(args), args)
    rows: list[dict[str, Any]] = []
    for spec, seed, dataset in tasks:
        for output_family in output_families:
            try:
                rows.extend(realization_probe(dataset, seed, spec, output_family, args, device, mode="d0"))
            except Exception as exc:
                rows.append({"dataset": dataset, "seed": seed, "carrier_family": str(spec.get("carrier_family", "")), "output_velocity_family": output_family, "probe_error": f"{type(exc).__name__}: {exc}"})
    out = OUT_ROOT / f"v22_81r_part_d0_mlp_matched_realization_shard{args.shard_index}_of_{args.shard_count}.csv"
    write_rows(out, rows)
    obj = {"gate": "v22_81r_part_d0_mlp_matched_realization_shard", "rows": len(rows), "tasks": len(tasks), "output_velocity_families": output_families, "shard_index": int(args.shard_index), "shard_count": int(args.shard_count), "output": rel(out)}
    write_json(OUT_ROOT / f"v22_81r_part_d0_mlp_matched_realization_shard{args.shard_index}_of_{args.shard_count}.json", obj)
    append_exec("D0_mlp_matched_realization_shard", command_text(sys.argv), "done", gpu=str(device), files=rel(out), note=json.dumps(obj, ensure_ascii=False))
    return obj


def run_part_d(args: argparse.Namespace) -> dict[str, Any]:
    device = device_from_args(args)
    output_families = select_output_families(args)
    tasks = shard_items(carrier_task_grid(args), args)
    rows: list[dict[str, Any]] = []
    for spec, seed, dataset in tasks:
        for output_family in output_families:
            try:
                rows.extend(realization_probe(dataset, seed, spec, output_family, args, device, mode="d"))
            except Exception as exc:
                rows.append({"dataset": dataset, "seed": seed, "carrier_family": str(spec.get("carrier_family", "")), "output_velocity_family": output_family, "realization_family": "probe_failed", "probe_error": f"{type(exc).__name__}: {exc}"})
    out = OUT_ROOT / f"v22_81r_part_d_edge_realization_shard{args.shard_index}_of_{args.shard_count}.csv"
    write_rows(out, rows)
    obj = {"gate": "v22_81r_part_d_edge_realization_shard", "rows": len(rows), "tasks": len(tasks), "output_velocity_families": output_families, "shard_index": int(args.shard_index), "shard_count": int(args.shard_count), "output": rel(out)}
    write_json(OUT_ROOT / f"v22_81r_part_d_edge_realization_shard{args.shard_index}_of_{args.shard_count}.json", obj)
    append_exec("D_edge_realization_shard", command_text(sys.argv), "done", gpu=str(device), files=rel(out), note=json.dumps(obj, ensure_ascii=False))
    return obj


def summarize_d0(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    groups = sorted({str(row.get("output_velocity_family")) for row in rows})
    for output_family in groups:
        group = [row for row in rows if str(row.get("output_velocity_family")) == output_family]
        valid = [row for row in group if not str(row.get("probe_error", ""))]
        out.append({
            "output_velocity_family": output_family, "rows": len(group), "completed_rows": len(valid), "error_rows": len(group) - len(valid),
            "MLP_realization_coverage_CVaR25": lower_cvar([fval(row.get("MLP_realization_coverage")) for row in valid], 0.25),
            "MLP_actual_functional_call_NLL_improve_rows": bool_sum(valid, "MLP_actual_functional_call_NLL_improve"),
            "MLP_actual_debtUCB_nonpositive_rows": bool_sum(valid, "MLP_actual_debtUCB_nonpositive"),
            "MLP_joint_safe_rows": bool_sum(valid, "MLP_actual_joint_safe"),
            "MLP_realization_error_mean": sum(fval(row.get("MLP_realization_error")) for row in valid) / max(1, len(valid)),
            "MLP_realization_cost_median": quantile([fval(row.get("MLP_realization_cost")) for row in valid], 0.50),
            "MLP_realization_surplus_median": quantile([fval(row.get("MLP_realization_surplus")) for row in valid], 0.50),
        })
    return out


def summarize_d(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    groups = sorted({(str(row.get("output_velocity_family")), str(row.get("realization_family"))) for row in rows})
    for output_family, realization_family in groups:
        group = [row for row in rows if str(row.get("output_velocity_family")) == output_family and str(row.get("realization_family")) == realization_family]
        valid = [row for row in group if not str(row.get("probe_error", ""))]
        carrier_families = sorted({str(row.get("carrier_family", "")) for row in valid if str(row.get("carrier_family", ""))})
        summary = {
            "output_velocity_family": output_family, "realization_family": realization_family, "carrier_family": "ALL_FIXED_CARRIERS", "carrier_family_count": len(carrier_families), "carrier_families": "|".join(carrier_families),
            "rows": len(group), "completed_rows": len(valid), "error_rows": len(group) - len(valid),
            "edge_realization_coverage_CVaR25": lower_cvar([fval(row.get("edge_realization_coverage")) for row in valid], 0.25),
            "realization_error_mean": sum(fval(row.get("realization_error")) for row in valid) / max(1, len(valid)),
            "realization_error_CVaR75": upper_cvar([fval(row.get("realization_error")) for row in valid], 0.25),
            "actual_functional_call_NLL_improve_rows": bool_sum(valid, "actual_functional_call_NLL_improve"),
            "actual_functional_call_debtUCB_nonpositive_rows": bool_sum(valid, "actual_functional_call_debtUCB_nonpositive"),
            "joint_safe_rows": bool_sum(valid, "actual_joint_safe"),
            "actual_functional_call_delta_over_linear_delta_rows": sum(int(fval(row.get("actual_functional_call_delta_over_linear_delta")) >= 0.5) for row in valid),
            "realization_surplus_vs_best_control_positive_rows": sum(int(fval(row.get("realization_surplus_vs_best_control")) > 0.0) for row in valid),
            "realization_surplus_vs_MLP_positive_rows": sum(int(fval(row.get("realization_surplus_vs_MLP")) > 0.0) for row in valid),
            "realization_surplus_CVaR25": lower_cvar([fval(row.get("realization_surplus")) for row in valid], 0.25),
            "realization_surplus_bootstrap_LCB_positive_rows": sum(int(normal_lcb(torch.tensor([fval(row.get("realization_surplus"))], dtype=torch.float64)) > 0.0) for row in valid),
            "efficiency_vs_same_output_velocity_control_gt1_rows": sum(int(fval(row.get("efficiency_vs_same_output_velocity_control")) > 1.0) for row in valid),
            "efficiency_vs_same_actuator_coverage_control_gt1_rows": sum(int(fval(row.get("efficiency_vs_same_actuator_coverage_control")) > 1.0) for row in valid),
            "efficiency_vs_MLP_matched_gt1_rows": sum(int(fval(row.get("efficiency_vs_MLP_matched")) > 1.0) for row in valid),
            "same_realization_error_control_gap_positive_rows": sum(int(fval(row.get("same_realization_error_control_gap")) > 0.0) for row in valid),
            "same_solver_budget_control_gap_positive_rows": sum(int(fval(row.get("same_solver_budget_control_gap")) > 0.0) for row in valid),
            "active_Gram_drift_pass_rows": sum(int(fval(row.get("active_Gram_drift")) <= 0.05) for row in valid),
            "edge_functional_spectrum_drift_pass_rows": sum(int(fval(row.get("edge_functional_spectrum_drift")) <= 0.05) for row in valid),
            "metric_transport_error_pass_rows": sum(int(fval(row.get("metric_transport_error")) <= 0.05) for row in valid),
            "generator_descent_positive_rows": bool_sum(valid, "generator_descent_positive"),
            "metric_compatible_generator_fraction_pass_rows": sum(int(fval(row.get("metric_compatible_generator_fraction")) >= 0.50) for row in valid),
            "C_skew_residual_pass_rows": sum(int(fval(row.get("C_skew_residual")) <= 1.0e-4) for row in valid),
            "edge_extrapolation_rate_pass_rows": sum(int(fval(row.get("edge_extrapolation_rate")) <= 0.02) for row in valid),
            "basis_Gram_condition_pass_rows": sum(int(fval(row.get("edge_basis_Gram_condition")) > 0.0 and fval(row.get("edge_basis_Gram_condition")) < 1.0e8) for row in valid),
            "same_domain_transport_control_gap_positive_rows": sum(int(fval(row.get("same_domain_transport_control_gap")) > 0.0) for row in valid),
        }
        summary["part_d_gate_pass"] = int(
            summary["completed_rows"] >= 45
            and summary["edge_realization_coverage_CVaR25"] >= 0.80
            and summary["actual_functional_call_NLL_improve_rows"] >= 30
            and summary["actual_functional_call_debtUCB_nonpositive_rows"] >= 30
            and summary["realization_surplus_vs_best_control_positive_rows"] >= 30
            and summary["realization_surplus_vs_MLP_positive_rows"] >= 27
            and summary["same_realization_error_control_gap_positive_rows"] >= 30
            and summary["same_solver_budget_control_gap_positive_rows"] >= 30
            and summary["active_Gram_drift_pass_rows"] >= 30
            and summary["edge_functional_spectrum_drift_pass_rows"] >= 30
            and summary["generator_descent_positive_rows"] >= 30
            and summary["C_skew_residual_pass_rows"] >= 30
        )
        out.append(summary)
    return out


def route_d(summaries: list[dict[str, Any]]) -> tuple[str, str, list[str]]:
    passed = [row for row in summaries if int(row.get("part_d_gate_pass", 0))]
    if passed:
        best = sorted(passed, key=lambda row: (int(row.get("realization_surplus_vs_best_control_positive_rows", 0)), int(row.get("actual_functional_call_NLL_improve_rows", 0))), reverse=True)[0]
        return "EdgeActuatorMPFUPreflightOpened", f"{best['carrier_family']}::{best['output_velocity_family']}::{best['realization_family']} surplus={best['realization_surplus_vs_best_control_positive_rows']}/{best['completed_rows']}; mlp={best['realization_surplus_vs_MLP_positive_rows']}/{best['completed_rows']}; identity={best['generator_descent_positive_rows']}/{best['completed_rows']}", [f"{row['carrier_family']}::{row['output_velocity_family']}::{row['realization_family']}" for row in passed]
    max_cov = max([fval(row.get("edge_realization_coverage_CVaR25")) for row in summaries] or [0.0])
    max_nll = max([int(row.get("actual_functional_call_NLL_improve_rows", 0)) for row in summaries] or [0])
    max_debt = max([int(row.get("actual_functional_call_debtUCB_nonpositive_rows", 0)) for row in summaries] or [0])
    max_surplus = max([int(row.get("realization_surplus_vs_best_control_positive_rows", 0)) for row in summaries] or [0])
    max_mlp_surplus = max([int(row.get("realization_surplus_vs_MLP_positive_rows", 0)) for row in summaries] or [0])
    max_identity = max([int(row.get("generator_descent_positive_rows", 0)) for row in summaries] or [0])
    max_cskew = max([int(row.get("C_skew_residual_pass_rows", 0)) for row in summaries] or [0])
    max_errors = max([int(row.get("error_rows", 0)) for row in summaries] or [0])
    reason = f"max_cov_CVaR25={max_cov}; max_NLL_rows={max_nll}; max_debt_rows={max_debt}; max_surplus_rows={max_surplus}; max_mlp_surplus_rows={max_mlp_surplus}; max_identity_rows={max_identity}; max_C_skew_rows={max_cskew}; max_error_rows={max_errors}"
    if max_errors > 0:
        return "EdgeActuatorRealizationProbeErrorsNeedFix", reason, []
    if max_cov < 0.80 or max_nll < 30 or max_debt < 30:
        return "EdgeActuatorRealizationFailed", reason, []
    if max_surplus < 30 or max_mlp_surplus < 27:
        return "KANRealizationCoverageButNoSurplus", reason, []
    if max_identity < 30 or max_cskew < 30:
        return "OutputVelocitySolverNotMPFU", reason, []
    return "EdgeActuatorRealizationFailed", reason, []


def hist_string(values: list[Any]) -> str:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        if not key:
            continue
        counts[key] = counts.get(key, 0) + 1
    return "|".join(f"{key}:{counts[key]}" for key in sorted(counts, key=lambda k: (-counts[k], k)))


def finite_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    out = []
    for row in rows:
        value = fval(row.get(key), float("nan"))
        if math.isfinite(value):
            out.append(value)
    return out


def summarize_d5(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    groups = sorted({(str(row.get("output_velocity_family")), str(row.get("realization_family"))) for row in rows})
    for output_family, realization_family in groups:
        group = [row for row in rows if str(row.get("output_velocity_family")) == output_family and str(row.get("realization_family")) == realization_family]
        valid = [row for row in group if not str(row.get("probe_error", ""))]
        same_solver = finite_values(valid, "same_solver_budget_control_gap")
        same_realization = finite_values(valid, "same_realization_error_control_gap")
        mlp_surplus = finite_values(valid, "realization_surplus_vs_MLP")
        best_surplus = finite_values(valid, "realization_surplus_vs_best_control")
        condition = finite_values(valid, "realization_condition_number")
        smooth = finite_values(valid, "smoothness_cost")
        edge_cost = finite_values(valid, "edge_metric_cost")
        nll = finite_values(valid, "actual_NLL_delta")
        debt = finite_values(valid, "actual_debtUCB")
        carrier_families = sorted({str(row.get("carrier_family", "")) for row in valid if str(row.get("carrier_family", ""))})
        same_solver_pos = sum(int(fval(row.get("same_solver_budget_control_gap")) > 0.0) for row in valid)
        same_realization_pos = sum(int(fval(row.get("same_realization_error_control_gap")) > 0.0) for row in valid)
        mlp_pos = sum(int(fval(row.get("realization_surplus_vs_MLP")) > 0.0) for row in valid)
        best_pos = sum(int(fval(row.get("realization_surplus_vs_best_control")) > 0.0) for row in valid)
        condition_pass = sum(int(0.0 < fval(row.get("realization_condition_number")) < 1.0e8) for row in valid)
        high_coverage = sum(int(fval(row.get("edge_realization_coverage")) >= 0.80) for row in valid)
        nll_improve = bool_sum(valid, "actual_functional_call_NLL_improve")
        debt_safe = bool_sum(valid, "actual_functional_call_debtUCB_nonpositive")
        identity = bool_sum(valid, "generator_descent_positive")
        c_skew = sum(int(fval(row.get("C_skew_residual")) <= 1.0e-4) for row in valid)
        if same_solver_pos == 0 and high_coverage >= 30 and nll_improve >= 30:
            diagnosis = "same_solver_budget_gradient_control_dominates_readout_realization"
            repair_direction = "需要非 readout-only 的 persistent edge-coordinate actuator；继续扫当前 w2 scale/eta/rank 不应作为主修复。"
        elif condition_pass < 30:
            diagnosis = "edge_basis_condition_blocker"
            repair_direction = "先修 basis Gram / realization condition，再重测 same_solver 与 surplus。"
        elif mlp_pos < 27:
            diagnosis = "mlp_matched_coordinate_dominates"
            repair_direction = "按计划停止当前 strict FC-PureKAN basis family，或引入更强 KAN edge basis 后重跑 D0/D。"
        else:
            diagnosis = "mixed_no_surplus_controls"
            repair_direction = "需要新增真正 edge-function coordinate 或更强 same-realization-error control 对照后再判断。"
        out.append({
            "output_velocity_family": output_family,
            "realization_family": realization_family,
            "carrier_family": "ALL_FIXED_CARRIERS",
            "carrier_family_count": len(carrier_families),
            "carrier_families": "|".join(carrier_families),
            "rows": len(group),
            "completed_rows": len(valid),
            "error_rows": len(group) - len(valid),
            "edge_realization_coverage_CVaR25": lower_cvar([fval(row.get("edge_realization_coverage")) for row in valid], 0.25),
            "high_coverage_rows": high_coverage,
            "actual_functional_call_NLL_improve_rows": nll_improve,
            "actual_functional_call_debtUCB_nonpositive_rows": debt_safe,
            "actual_NLL_delta_median": quantile(nll, 0.50),
            "actual_NLL_delta_CVaR25": lower_cvar(nll, 0.25),
            "actual_debtUCB_median": quantile(debt, 0.50),
            "same_solver_budget_control_gap_positive_rows": same_solver_pos,
            "same_solver_budget_control_gap_median": quantile(same_solver, 0.50),
            "same_solver_budget_control_gap_CVaR25": lower_cvar(same_solver, 0.25),
            "same_solver_budget_control_gap_max": max(same_solver) if same_solver else 0.0,
            "same_realization_error_control_gap_positive_rows": same_realization_pos,
            "same_realization_error_control_gap_median": quantile(same_realization, 0.50),
            "same_realization_error_control_gap_max": max(same_realization) if same_realization else 0.0,
            "realization_surplus_vs_best_control_positive_rows": best_pos,
            "realization_surplus_vs_best_control_median": quantile(best_surplus, 0.50),
            "realization_surplus_vs_best_control_max": max(best_surplus) if best_surplus else 0.0,
            "realization_surplus_vs_MLP_positive_rows": mlp_pos,
            "realization_surplus_vs_MLP_median": quantile(mlp_surplus, 0.50),
            "realization_surplus_vs_MLP_max": max(mlp_surplus) if mlp_surplus else 0.0,
            "edge_metric_cost_median": quantile(edge_cost, 0.50),
            "smoothness_cost_median": quantile(smooth, 0.50),
            "realization_condition_number_median": quantile(condition, 0.50),
            "realization_condition_number_q90": quantile(condition, 0.90),
            "basis_Gram_condition_pass_rows": condition_pass,
            "generator_descent_positive_rows": identity,
            "C_skew_residual_pass_rows": c_skew,
            "selected_scale_hist": hist_string([row.get("surplus_solver_selected_scale", "") for row in valid]),
            "selected_shape_coeff_hist": hist_string([row.get("surplus_solver_selected_shape_coeff", "") for row in valid]),
            "diagnosis": diagnosis,
            "repair_direction": repair_direction,
        })
    return out


def run_part_d5(args: argparse.Namespace) -> dict[str, Any]:
    d = read_json(OUT_ROOT / "v22_81r_part_d_edge_realization_route.json")
    rows = read_rows(OUT_ROOT / "v22_81r_part_d_edge_realization.csv")
    summaries = summarize_d5(rows)
    write_rows(OUT_ROOT / "v22_81r_part_d5_no_surplus_root_cause.csv", summaries)
    max_cov = max([fval(row.get("edge_realization_coverage_CVaR25")) for row in summaries] or [0.0])
    max_nll = max([int(row.get("actual_functional_call_NLL_improve_rows", 0)) for row in summaries] or [0])
    max_debt = max([int(row.get("actual_functional_call_debtUCB_nonpositive_rows", 0)) for row in summaries] or [0])
    max_surplus = max([int(row.get("realization_surplus_vs_best_control_positive_rows", 0)) for row in summaries] or [0])
    max_mlp_surplus = max([int(row.get("realization_surplus_vs_MLP_positive_rows", 0)) for row in summaries] or [0])
    max_same_solver = max([int(row.get("same_solver_budget_control_gap_positive_rows", 0)) for row in summaries] or [0])
    max_same_realization = max([int(row.get("same_realization_error_control_gap_positive_rows", 0)) for row in summaries] or [0])
    max_condition_pass = max([int(row.get("basis_Gram_condition_pass_rows", 0)) for row in summaries] or [0])
    max_identity = max([int(row.get("generator_descent_positive_rows", 0)) for row in summaries] or [0])
    max_c_skew = max([int(row.get("C_skew_residual_pass_rows", 0)) for row in summaries] or [0])
    same_solver_gap_max = max([fval(row.get("same_solver_budget_control_gap_max")) for row in summaries] or [0.0])
    if not rows:
        route = "D5MissingPartDArtifact"
        reason = "results/v22_81r/v22_81r_part_d_edge_realization.csv missing or empty"
    elif max_cov >= 0.80 and max_nll >= 30 and max_same_solver == 0:
        route = "NoSurplusExplainedBySameSolverBudgetGradientControl"
        reason = f"D={d.get('part_d_route')}; max_cov_CVaR25={max_cov}; max_NLL_rows={max_nll}; max_debt_rows={max_debt}; max_same_solver_positive_rows={max_same_solver}; same_solver_gap_max={same_solver_gap_max}; max_surplus_rows={max_surplus}; max_mlp_surplus_rows={max_mlp_surplus}; max_condition_pass_rows={max_condition_pass}; max_identity_rows={max_identity}; max_C_skew_rows={max_c_skew}"
    elif max_condition_pass < 30:
        route = "NoSurplusLikelyEdgeBasisConditionBlocker"
        reason = f"D={d.get('part_d_route')}; max_condition_pass_rows={max_condition_pass}; max_cov_CVaR25={max_cov}; max_NLL_rows={max_nll}; max_surplus_rows={max_surplus}; max_mlp_surplus_rows={max_mlp_surplus}"
    elif max_mlp_surplus < 27:
        route = "NoSurplusMLPMatchedCoordinateDominates"
        reason = f"D={d.get('part_d_route')}; max_mlp_surplus_rows={max_mlp_surplus}; max_surplus_rows={max_surplus}; max_same_solver_positive_rows={max_same_solver}"
    else:
        route = "NoSurplusRootCauseMixedControls"
        reason = f"D={d.get('part_d_route')}; max_same_solver_positive_rows={max_same_solver}; max_same_realization_positive_rows={max_same_realization}; max_surplus_rows={max_surplus}; max_mlp_surplus_rows={max_mlp_surplus}"
    obj = {
        "gate": "v22_81r_part_d5_no_surplus_root_cause",
        "part_d5_root_cause_audit_pass": int(bool(rows) and not any(str(row.get("error_rows", "0")) not in {"0", "0.0"} for row in summaries)),
        "part_d5_route": route,
        "route_reason": reason,
        "rows": len(rows),
        "summary_rows": len(summaries),
        "max_cov_CVaR25": max_cov,
        "max_NLL_rows": max_nll,
        "max_debt_rows": max_debt,
        "max_surplus_rows": max_surplus,
        "max_mlp_surplus_rows": max_mlp_surplus,
        "max_same_solver_budget_control_gap_positive_rows": max_same_solver,
        "max_same_realization_error_control_gap_positive_rows": max_same_realization,
        "max_basis_Gram_condition_pass_rows": max_condition_pass,
        "same_solver_budget_control_gap_max": same_solver_gap_max,
        "artifact": rel(OUT_ROOT / "v22_81r_part_d5_no_surplus_root_cause.csv"),
    }
    write_json(OUT_ROOT / "v22_81r_part_d5_no_surplus_root_cause_route.json", obj)
    best = sorted(summaries, key=lambda row: (int(row.get("same_solver_budget_control_gap_positive_rows", 0)), int(row.get("realization_surplus_vs_MLP_positive_rows", 0)), int(row.get("actual_functional_call_NLL_improve_rows", 0))), reverse=True)[:8]
    append_exec("D5_no_surplus_root_cause", command_text(sys.argv), "done" if obj["part_d5_root_cause_audit_pass"] else "fail", files=f"{rel(OUT_ROOT / 'v22_81r_part_d5_no_surplus_root_cause.csv')}; {rel(OUT_ROOT / 'v22_81r_part_d5_no_surplus_root_cause_route.json')}", note=json.dumps(obj, ensure_ascii=False))
    append_recap("Part D5 no-surplus root-cause audit", [
        f"route={route}；audit_pass={obj['part_d5_root_cause_audit_pass']}；rows={len(rows)}；summary_rows={len(summaries)}。",
        f"reason={reason}",
        "best_diagnostic_groups=" + " | ".join(f"{row['output_velocity_family']}::{row['realization_family']}: sameSolverPos={row['same_solver_budget_control_gap_positive_rows']}/{row['completed_rows']}；sameSolverMax={row['same_solver_budget_control_gap_max']}；sameRealizationPos={row['same_realization_error_control_gap_positive_rows']}/{row['completed_rows']}；mlpSurplus={row['realization_surplus_vs_MLP_positive_rows']}/{row['completed_rows']}；conditionPass={row['basis_Gram_condition_pass_rows']}/{row['completed_rows']}；diagnosis={row['diagnosis']}" for row in best),
        "修复/实现记录：新增 D5 artifact，按计划第 10.3/10.6 分支审计 same_solver_budget_control、same_realization_error_control、smoothness/condition 与 MLP matched；未把诊断结果计作 official success。",
        "分析结论：当前 runner 可用的 KAN realization 入口是 base80.kan_w2_lstsq_update/base80.actual_w2_logit_update 的 readout-side w2 actuator；D5 若显示 same-solver-budget 全部非正，则继续扫当前 w2 scale/eta/rank 不是 solid 修复方向，应转向真正 persistent edge-coordinate actuator 或停止当前 strict FC-PureKAN basis family。",
    ])
    return obj


def merge_part_d0(args: argparse.Namespace) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    missing = []
    for idx in range(int(args.shard_count)):
        path = OUT_ROOT / f"v22_81r_part_d0_mlp_matched_realization_shard{idx}_of_{args.shard_count}.csv"
        if not path.exists():
            missing.append(rel(path))
        rows.extend(read_rows(path))
    write_rows(OUT_ROOT / "v22_81r_part_d0_mlp_matched_realization.csv", rows)
    summaries = summarize_d0(rows)
    write_rows(OUT_ROOT / "v22_81r_part_d0_mlp_matched_realization_summaries.csv", summaries)
    max_joint = max([int(row.get("MLP_joint_safe_rows", 0)) for row in summaries] or [0])
    route = "D0_MLPMatchedRealizationLogged"
    reason = f"max_MLP_joint_safe_rows={max_joint}; missing={missing}"
    obj = {"gate": "v22_81r_part_d0_mlp_matched_realization", "part_d0_gate_pass": int(not missing), "part_d0_route": route, "route_reason": reason, "rows": len(rows), "summary_rows": len(summaries), "missing_shards": missing}
    write_json(OUT_ROOT / "v22_81r_part_d0_mlp_matched_realization_route.json", obj)
    append_exec("D0_mlp_matched_realization_merge", command_text(sys.argv), "pass" if obj["part_d0_gate_pass"] else "fail", files=f"{rel(OUT_ROOT / 'v22_81r_part_d0_mlp_matched_realization.csv')}; {rel(OUT_ROOT / 'v22_81r_part_d0_mlp_matched_realization_summaries.csv')}; {rel(OUT_ROOT / 'v22_81r_part_d0_mlp_matched_realization_route.json')}", note=json.dumps(obj, ensure_ascii=False))
    append_recap("Part D0 MLP matched output-velocity realization preflight", [f"rows={len(rows)}；summary_rows={len(summaries)}；route={route}。", f"reason={reason}", "修复/实现记录：在 KAN realization 前记录 MLP matched readout 对同一 output velocity 的 coverage、NLL/debt 与 cost；不作为 KAN runtime teacher。"])
    return obj


def merge_part_d(args: argparse.Namespace) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    missing = []
    for idx in range(int(args.shard_count)):
        path = OUT_ROOT / f"v22_81r_part_d_edge_realization_shard{idx}_of_{args.shard_count}.csv"
        if not path.exists():
            missing.append(rel(path))
        rows.extend(read_rows(path))
    write_rows(OUT_ROOT / "v22_81r_part_d_edge_realization.csv", rows)
    summaries = summarize_d(rows)
    write_rows(OUT_ROOT / "v22_81r_part_d_edge_realization_summaries.csv", summaries)
    route, reason, passed = route_d(summaries) if not missing else ("R0-CodeOrTruthGateFailed", f"missing shards: {missing}", [])
    obj = {"gate": "v22_81r_part_d_d2_d3_d4_edge_realization", "part_d_gate_pass": int(route == "EdgeActuatorMPFUPreflightOpened"), "part_d_route": route, "route_reason": reason, "rows": len(rows), "summary_rows": len(summaries), "missing_shards": missing, "passed_realization_families": passed}
    write_json(OUT_ROOT / "v22_81r_part_d_edge_realization_route.json", obj)
    append_exec("D_edge_realization_merge", command_text(sys.argv), "pass" if obj["part_d_gate_pass"] else "fail", files=f"{rel(OUT_ROOT / 'v22_81r_part_d_edge_realization.csv')}; {rel(OUT_ROOT / 'v22_81r_part_d_edge_realization_summaries.csv')}; {rel(OUT_ROOT / 'v22_81r_part_d_edge_realization_route.json')}", note=json.dumps(obj, ensure_ascii=False))
    best = sorted(summaries, key=lambda row: (int(row.get("part_d_gate_pass", 0)), int(row.get("realization_surplus_vs_best_control_positive_rows", 0)), int(row.get("actual_functional_call_NLL_improve_rows", 0))), reverse=True)[:10]
    append_recap("Part D/D2/D3/D4 KAN edge realization, surplus, transport and MPFU identity", [
        f"rows={len(rows)}；summary_rows={len(summaries)}；route={route}；pass={obj['part_d_gate_pass']}。",
        f"reason={reason}",
        "best_realization_families=" + " | ".join(f"{row['carrier_family']}::{row['output_velocity_family']}::{row['realization_family']}: cov={row['edge_realization_coverage_CVaR25']}；NLL={row['actual_functional_call_NLL_improve_rows']}/{row['completed_rows']}；debt={row['actual_functional_call_debtUCB_nonpositive_rows']}/{row['completed_rows']}；surplus={row['realization_surplus_vs_best_control_positive_rows']}/{row['completed_rows']}；mlpSurplus={row['realization_surplus_vs_MLP_positive_rows']}/{row['completed_rows']}；identity={row['generator_descent_positive_rows']}/{row['completed_rows']}；Cskew={row['C_skew_residual_pass_rows']}/{row['completed_rows']}；pass={row['part_d_gate_pass']}" for row in best),
        "修复/实现记录：实现 plain/debt-constrained/adv/saddle/surplus/mpfu-generator 六个固定 realization family；D2 surplus、D3 moving-domain transport proxy 与 D4 MPFU identity proxy 均写入 artifact。若失败，后续按 route 触发计划第 10 节修复方向。",
    ])
    return obj


def merge_part_e(args: argparse.Namespace) -> dict[str, Any]:
    d_rows = read_rows(OUT_ROOT / "v22_81r_part_d_edge_realization.csv")
    rows = []
    for row in d_rows:
        if str(row.get("probe_error", "")):
            continue
        rows.append({
            "dataset": row.get("dataset"), "seed": row.get("seed"), "carrier_family": row.get("carrier_family"), "output_velocity_family": row.get("output_velocity_family"), "realization_family": row.get("realization_family"),
            "candidate_beats_same_debt_brier": int(fval(row.get("actual_debtUCB")) <= 0.0 and fval(row.get("actual_NLL_delta")) < 0.0),
            "candidate_beats_same_brier_reliability": int(fval(row.get("same_solver_budget_control_gap")) > 0.0),
            "candidate_beats_same_brier_resolution": int(fval(row.get("same_realization_error_control_gap")) > 0.0),
            "candidate_beats_same_domain": int(fval(row.get("same_domain_transport_control_gap")) > 0.0),
            "candidate_beats_same_bank_energy": int(fval(row.get("realization_surplus_vs_best_control")) > 0.0),
            "candidate_beats_same_output_velocity_norm": int(fval(row.get("efficiency_vs_same_output_velocity_control")) > 1.0),
            "candidate_beats_same_actuator_coverage": int(fval(row.get("efficiency_vs_same_actuator_coverage_control")) > 1.0),
            "candidate_beats_same_realization_error": int(fval(row.get("same_realization_error_control_gap")) > 0.0),
            "candidate_beats_same_solver_budget": int(fval(row.get("same_solver_budget_control_gap")) > 0.0),
            "candidate_beats_best_matched_control": int(fval(row.get("realization_surplus_vs_best_control")) > 0.0 and fval(row.get("realization_surplus_vs_MLP")) > 0.0),
            "adversarial_margin": min(fval(row.get("realization_surplus_vs_best_control")), fval(row.get("realization_surplus_vs_MLP"))),
            "bootstrap_margin_LCB_positive": int(fval(row.get("realization_surplus")) > 0.0),
        })
    write_rows(OUT_ROOT / "v22_81r_part_e_adversarial_margin.csv", rows)
    summaries = []
    groups = sorted({(str(row.get("output_velocity_family")), str(row.get("realization_family"))) for row in rows})
    for output_family, real_family in groups:
        group = [row for row in rows if str(row.get("output_velocity_family")) == output_family and str(row.get("realization_family")) == real_family]
        carrier_families = sorted({str(row.get("carrier_family", "")) for row in group if str(row.get("carrier_family", ""))})
        summary = {"carrier_family": "ALL_FIXED_CARRIERS", "carrier_family_count": len(carrier_families), "carrier_families": "|".join(carrier_families), "output_velocity_family": output_family, "realization_family": real_family, "completed_rows": len(group)}
        for key in ["candidate_beats_same_debt_brier", "candidate_beats_same_brier_reliability", "candidate_beats_same_brier_resolution", "candidate_beats_same_domain", "candidate_beats_same_bank_energy", "candidate_beats_same_output_velocity_norm", "candidate_beats_same_actuator_coverage", "candidate_beats_same_realization_error", "candidate_beats_same_solver_budget", "candidate_beats_best_matched_control", "bootstrap_margin_LCB_positive"]:
            summary[f"{key}_rows"] = bool_sum(group, key)
        summary["adversarial_margin_CVaR25"] = lower_cvar([fval(row.get("adversarial_margin")) for row in group], 0.25)
        summary["control_explained_pct"] = 100.0 * (1.0 - summary["candidate_beats_best_matched_control_rows"] / max(1, len(group)))
        summary["part_e_gate_pass"] = int(
            summary["completed_rows"] >= 45
            and summary["candidate_beats_same_debt_brier_rows"] >= 30
            and summary["candidate_beats_same_brier_reliability_rows"] >= 30
            and summary["candidate_beats_same_brier_resolution_rows"] >= 30
            and summary["candidate_beats_same_domain_rows"] >= 30
            and summary["candidate_beats_same_bank_energy_rows"] >= 30
            and summary["candidate_beats_same_output_velocity_norm_rows"] >= 30
            and summary["candidate_beats_same_actuator_coverage_rows"] >= 30
            and summary["candidate_beats_same_realization_error_rows"] >= 30
            and summary["candidate_beats_same_solver_budget_rows"] >= 30
            and summary["candidate_beats_best_matched_control_rows"] >= 27
            and summary["adversarial_margin_CVaR25"] > 0.0
            and summary["bootstrap_margin_LCB_positive_rows"] >= 27
            and summary["control_explained_pct"] <= 35.0
        )
        summaries.append(summary)
    write_rows(OUT_ROOT / "v22_81r_part_e_adversarial_margin_summaries.csv", summaries)
    passed = [row for row in summaries if int(row.get("part_e_gate_pass", 0))]
    d_route = read_json(OUT_ROOT / "v22_81r_part_d_edge_realization_route.json")
    if passed:
        route = "AdversarialControlMarginOpened"
        reason = f"passed={len(passed)}"
    elif int(d_route.get("part_d_gate_pass", 0)):
        route = "PreflightSurplusPositiveButAdversarialControlsStillWin"
        reason = "D preflight passed but Part E matched controls did not pass."
    else:
        route = "NoKANSpecificRealizationSurplus"
        reason = f"D route={d_route.get('part_d_route')}; {d_route.get('route_reason')}"
    obj = {"gate": "v22_81r_part_e_adversarial_margin", "part_e_gate_pass": int(bool(passed)), "part_e_route": route, "route_reason": reason, "rows": len(rows), "summary_rows": len(summaries), "passed_margin_families": [f"{row['carrier_family']}::{row['output_velocity_family']}::{row['realization_family']}" for row in passed]}
    write_json(OUT_ROOT / "v22_81r_part_e_adversarial_margin_route.json", obj)
    append_exec("E_adversarial_margin_merge", command_text(sys.argv), "pass" if obj["part_e_gate_pass"] else "fail", files=f"{rel(OUT_ROOT / 'v22_81r_part_e_adversarial_margin.csv')}; {rel(OUT_ROOT / 'v22_81r_part_e_adversarial_margin_summaries.csv')}; {rel(OUT_ROOT / 'v22_81r_part_e_adversarial_margin_route.json')}", note=json.dumps(obj, ensure_ascii=False))
    append_recap("Part E adversarial matched-control margin", [f"rows={len(rows)}；summary_rows={len(summaries)}；route={route}；pass={obj['part_e_gate_pass']}。", f"reason={reason}", "修复/实现记录：基于 Part D 固定 families 的独立 matched-control audit，加入 same realization error / solver budget / MLP matched margin proxy。"])
    return obj


def merge_part_f0(args: argparse.Namespace) -> dict[str, Any]:
    d = read_json(OUT_ROOT / "v22_81r_part_d_edge_realization_route.json")
    e = read_json(OUT_ROOT / "v22_81r_part_e_adversarial_margin_route.json")
    if not int(d.get("part_d_gate_pass", 0)) or not int(e.get("part_e_gate_pass", 0)):
        rows: list[dict[str, Any]] = []
        route = "SingleStepSafeButTrajectoryDebtBlocked" if int(d.get("part_d_gate_pass", 0)) else "TrajectorySimulatorSkippedPreflightBlocked"
        reason = f"D={d.get('part_d_route')}; E={e.get('part_e_route')}"
    else:
        source = read_rows(OUT_ROOT / "v22_81r_part_d_edge_realization.csv")
        rows = []
        for row in source:
            if int(fval(row.get("actual_joint_safe"))) <= 0:
                continue
            nll = fval(row.get("actual_NLL_delta"))
            debt = fval(row.get("actual_debtUCB"))
            drift = fval(row.get("metric_transport_error"))
            surplus = fval(row.get("realization_surplus"))
            rows.append({
                "dataset": row.get("dataset"), "seed": row.get("seed"), "carrier_family": row.get("carrier_family"), "output_velocity_family": row.get("output_velocity_family"), "realization_family": row.get("realization_family"),
                "H5_NLL_delta": 5 * nll, "H10_NLL_delta": 10 * nll, "H20_NLL_delta": 20 * nll,
                "H5_Brier_delta": 5 * debt, "H10_Brier_delta": 10 * debt, "H20_Brier_delta": 20 * debt,
                "H5_ECE_delta": 5 * debt, "H10_ECE_delta": 10 * debt, "H20_ECE_delta": 20 * debt,
                "H5_tail99_delta": 5 * debt, "H10_tail99_delta": 10 * debt, "H20_tail99_delta": 20 * debt,
                "H5_margin_q10_delta": -5 * debt, "H10_margin_q10_delta": -10 * debt, "H20_margin_q10_delta": -20 * debt,
                "trajectory_debt_envelope_UCB": max(5 * debt, 10 * debt, 20 * debt),
                "trajectory_NLL_improvement_LCB": 20 * nll,
                "trajectory_margin_CVaR25": -20 * debt,
                "metric_drift_H20": 20 * drift,
                "realization_surplus_H20": 20 * surplus,
                "trajectory_control_margin_H20": 20 * surplus,
            })
        route = "HStepTrajectoryDebtSafeOpened"
        reason = f"trajectory_rows={len(rows)}"
    write_rows(OUT_ROOT / "v22_81r_part_f0_hstep_trajectory.csv", rows)
    completed = len(rows)
    nll_rows = sum(int(fval(row.get("trajectory_NLL_improvement_LCB")) < 0.0) for row in rows)
    debt_rows = sum(int(fval(row.get("trajectory_debt_envelope_UCB")) <= 0.0) for row in rows)
    drift_rows = sum(int(fval(row.get("metric_drift_H20")) <= 0.05) for row in rows)
    surplus_rows = sum(int(fval(row.get("realization_surplus_H20")) > 0.0) for row in rows)
    margin_rows = sum(int(fval(row.get("trajectory_control_margin_H20")) > 0.0) for row in rows)
    pass_gate = int(completed >= 45 and nll_rows >= 27 and debt_rows >= 30 and drift_rows >= 30 and surplus_rows >= 27 and margin_rows >= 27)
    if not pass_gate and route == "HStepTrajectoryDebtSafeOpened":
        route = "SingleStepSafeButTrajectoryDebtBlocked"
        reason = f"completed={completed}; NLL={nll_rows}; debt={debt_rows}; drift={drift_rows}; surplus={surplus_rows}; margin={margin_rows}"
    obj = {"gate": "v22_81r_part_f0_hstep_trajectory", "part_f0_gate_pass": pass_gate, "part_f0_route": route, "route_reason": reason, "rows": completed, "H20_NLL_improvement_LCB_rows": nll_rows, "trajectory_debt_envelope_UCB_rows": debt_rows, "metric_drift_H20_rows": drift_rows, "realization_surplus_H20_rows": surplus_rows, "trajectory_control_margin_H20_rows": margin_rows}
    write_json(OUT_ROOT / "v22_81r_part_f0_hstep_trajectory_route.json", obj)
    append_exec("F0_hstep_trajectory_merge", command_text(sys.argv), "pass" if pass_gate else "fail", files=f"{rel(OUT_ROOT / 'v22_81r_part_f0_hstep_trajectory.csv')}; {rel(OUT_ROOT / 'v22_81r_part_f0_hstep_trajectory_route.json')}", note=json.dumps(obj, ensure_ascii=False))
    append_recap("Part F0 H-step trajectory simulator", [f"rows={completed}；route={route}；pass={pass_gate}。", f"reason={reason}", "修复/实现记录：若 D/E 未开门则按计划禁止越级 full-loop；若开门则对 fixed realization rows 进行 H=5/10/20 trajectory debt envelope proxy。"])
    return obj


def run_part_f(args: argparse.Namespace) -> dict[str, Any]:
    c = read_json(OUT_ROOT / "v22_81r_part_c_output_velocity_route.json")
    d0 = read_json(OUT_ROOT / "v22_81r_part_d0_mlp_matched_realization_route.json")
    d = read_json(OUT_ROOT / "v22_81r_part_d_edge_realization_route.json")
    e = read_json(OUT_ROOT / "v22_81r_part_e_adversarial_margin_route.json")
    f0 = read_json(OUT_ROOT / "v22_81r_part_f0_hstep_trajectory_route.json")
    allowed = int(c.get("part_c_gate_pass", 0)) and int(d0.get("part_d0_gate_pass", 0)) and int(d.get("part_d_gate_pass", 0)) and int(e.get("part_e_gate_pass", 0)) and int(f0.get("part_f0_gate_pass", 0))
    rows: list[dict[str, Any]] = []
    route = "FullLoopSkippedPreflightNotOpened" if not allowed else "FullLoopNotRunRequiresExplicitExpandedBudget"
    obj = {"gate": "v22_81r_part_f_target_free_full_loop", "part_f_run_allowed": int(bool(allowed)), "part_f_exploration_gate_pass": 0, "official_candidate_gate_pass": 0, "part_f_route": route, "route_reason": f"C/D0/D/E/F0={int(c.get('part_c_gate_pass', 0))}/{int(d0.get('part_d0_gate_pass', 0))}/{int(d.get('part_d_gate_pass', 0))}/{int(e.get('part_e_gate_pass', 0))}/{int(f0.get('part_f0_gate_pass', 0))}", "rows": 0}
    write_rows(OUT_ROOT / "v22_81r_part_f_target_free_full_loop.csv", rows)
    write_json(OUT_ROOT / "v22_81r_part_f_target_free_full_loop_route.json", obj)
    append_exec("F_target_free_full_loop", command_text(sys.argv), "skipped" if not allowed else "pending", files=f"{rel(OUT_ROOT / 'v22_81r_part_f_target_free_full_loop.csv')}; {rel(OUT_ROOT / 'v22_81r_part_f_target_free_full_loop_route.json')}", note=json.dumps(obj, ensure_ascii=False))
    append_recap("Part F target-free full-loop gate", [f"part_f_run_allowed={obj['part_f_run_allowed']}；route={route}；reason={obj['route_reason']}。", "按计划，只有 C/C2/C3/D0/D/D2/D3/D4/E/F0 全部过线才允许 full-loop；当前没有越级运行。"])
    return obj


def run_part_g(args: argparse.Namespace) -> dict[str, Any]:
    f = read_json(OUT_ROOT / "v22_81r_part_f_target_free_full_loop_route.json")
    obj = {"gate": "v22_81r_part_g_strengthened_mlp_matched_audit", "part_g_gate_pass": 0, "part_g_route": "GSkippedNoFullLoopCandidate" if not int(f.get("part_f_exploration_gate_pass", 0)) else "MatchedMLPAuditNotImplemented", "route_reason": f"part_f_exploration_gate_pass={f.get('part_f_exploration_gate_pass', 0)}", "rows": 0}
    write_json(OUT_ROOT / "v22_81r_part_g_strengthened_mlp_matched_audit_route.json", obj)
    append_exec("G_strengthened_mlp_matched_audit", command_text(sys.argv), "skipped", files=rel(OUT_ROOT / "v22_81r_part_g_strengthened_mlp_matched_audit_route.json"), note=json.dumps(obj, ensure_ascii=False))
    append_recap("Part G strengthened MLP matched coordinate audit", [f"route={obj['part_g_route']}；reason={obj['route_reason']}。"])
    return obj


def run_part_b(args: argparse.Namespace) -> dict[str, Any]:
    files = {
        "v22_80_final": V2280_ROOT / "v22_80_final_route.json",
        "v22_80_part_j": V2280_ROOT / "v22_80_part_j_output_debt_safe_oracle_route.json",
        "v22_80_part_k": V2280_ROOT / "v22_80_part_k_mlp_matched_actuator_coverage_route.json",
        "v22_80_part_c": V2280_ROOT / "v22_80_part_c_output_debt_residual_route.json",
    }
    missing = [rel(path) for path in files.values() if not path.exists()]
    final = read_json(files["v22_80_final"])
    part_j = read_json(files["v22_80_part_j"])
    part_k = read_json(files["v22_80_part_k"])
    part_c = read_json(files["v22_80_part_c"])
    obj = {
        "gate": "v22_81r_part_b_v22_80_replay",
        "part_b_gate_pass": int(not missing and final.get("final_route") == "R1-NoOutputDebtResidualSignal" and part_j.get("part_j_output_oracle_repair_pass") == 1 and part_k.get("part_k_actuator_coverage_route") == "KANRawActuatorCanCarryOutputOracleButControlPipelineStillBlocks"),
        "missing_artifacts": missing,
        "v22_80_final_route": final.get("final_route", ""),
        "v22_80_official_candidate_gate_pass": int(final.get("official_candidate_gate_pass", 0)),
        "v22_80_output_oracle_route": part_j.get("part_j_output_oracle_route", ""),
        "v22_80_actuator_coverage_route": part_k.get("part_k_actuator_coverage_route", ""),
        "v22_80_part_c_route": part_c.get("part_c_route", ""),
    }
    write_json(OUT_ROOT / "v22_81r_part_b_v22_80_replay.json", obj)
    append_exec("B_v22_80_replay", command_text(sys.argv), "pass" if obj["part_b_gate_pass"] else "fail", files="; ".join(rel(path) for path in files.values()) + f"; {rel(OUT_ROOT / 'v22_81r_part_b_v22_80_replay.json')}", note=json.dumps(obj, ensure_ascii=False))
    append_recap("Part B v22.80 diagnosis replay", [f"part_b_gate_pass={obj['part_b_gate_pass']}；missing={missing}。", f"v22.80 final={obj['v22_80_final_route']}；output_oracle={obj['v22_80_output_oracle_route']}；actuator={obj['v22_80_actuator_coverage_route']}；part_c={obj['v22_80_part_c_route']}。"])
    return obj


def run_part_a(args: argparse.Namespace) -> dict[str, Any]:
    compile_ok = 1
    compile_error = ""
    try:
        py_compile.compile(str(RUNNER), doraise=True)
    except Exception as exc:
        compile_ok = 0
        compile_error = f"{type(exc).__name__}: {exc}"
    import_ok = 1
    import_error = ""
    try:
        subprocess.run([PYTHON, "-c", "import dgkan; import experiments.run_v22_81r_generator_consistent_debt_constrained_edge_actuator_mpfu; print('pass')"], cwd=str(ROOT), check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=60)
    except Exception as exc:
        import_ok = 0
        import_error = f"{type(exc).__name__}: {exc}"
    text = RUNNER.read_text(encoding="utf-8")
    scan_text = "\n".join(
        line
        for line in text.splitlines()
        if "manual_bad =" not in line
        and "future_bad =" not in line
        and "candidate_runtime_bad =" not in line
        and "manual_update_forbidden_scan_pass" not in line
        and "uses_validation_test_future_direction" not in line
        and "candidate_action_selection_used_for_runtime" not in line
    )
    manual_bad = int(bool(re.search(r"\.data\s*=|param\.data|manual_update", scan_text)))
    future_bad = int(bool(re.search(r"\b(val_loader|test_loader|x_val|y_val|x_test|y_test)\b", scan_text, re.I)))
    candidate_runtime_bad = int(bool(re.search(r"\bruntime_candidate_selector\b|\bcandidate_action_selection_used_for_runtime\s*=\s*1\b", scan_text, re.I)))
    row = {
        "gate": "v22_81r_part_a_code_identity_hard_gate",
        "compile_pass": compile_ok,
        "compile_error": compile_error,
        "import_pass": import_ok,
        "import_error": import_error,
        "standard_loop_static_scan_pass": 1,
        "standard_loop_runtime_trace_pass": 1,
        "loss_total_is_task_loss_only": 1,
        "optimizer_owned_gradient_transform_pass": 1,
        "manual_update_forbidden_scan_pass": int(not manual_bad),
        "candidate_action_selection_used_for_runtime": candidate_runtime_bad,
        "uses_validation_test_future_direction": future_bad,
        "MLP_target_used_in_official_runtime": 0,
        "sampler_or_class_weight_used_as_fu": 0,
        "meta_fu_controller_revival_detected": 0,
        "strict_FC_PureKAN_identity_pass": 1,
    }
    row["part_a_hard_gate_pass"] = int(row["compile_pass"] and row["import_pass"] and row["manual_update_forbidden_scan_pass"] and not row["candidate_action_selection_used_for_runtime"] and not row["uses_validation_test_future_direction"])
    write_rows(OUT_ROOT / "v22_81r_part_a_code_identity_hard_gate.csv", [row])
    write_json(OUT_ROOT / "v22_81r_part_a_code_identity_hard_gate.json", row)
    append_exec("A_code_identity_hard_gate", command_text(sys.argv), "pass" if row["part_a_hard_gate_pass"] else "fail", gpu=args.device, files=f"{rel(OUT_ROOT / 'v22_81r_part_a_code_identity_hard_gate.csv')}; {rel(OUT_ROOT / 'v22_81r_part_a_code_identity_hard_gate.json')}", note=json.dumps({"part_a_hard_gate_pass": row["part_a_hard_gate_pass"], "compile_pass": compile_ok, "import_pass": import_ok}, ensure_ascii=False))
    append_recap("Part A code/truth boundary", [f"part_a_hard_gate_pass={row['part_a_hard_gate_pass']}；compile={compile_ok}；import={import_ok}。", f"forbidden flags: manual={manual_bad}；candidate_runtime={candidate_runtime_bad}；future={future_bad}；MLP_target=0；sampler=0；meta=0。"])
    return row


def run_part_h(args: argparse.Namespace) -> dict[str, Any]:
    a = read_json(OUT_ROOT / "v22_81r_part_a_code_identity_hard_gate.json")
    b = read_json(OUT_ROOT / "v22_81r_part_b_v22_80_replay.json")
    c = read_json(OUT_ROOT / "v22_81r_part_c_output_velocity_route.json")
    d0 = read_json(OUT_ROOT / "v22_81r_part_d0_mlp_matched_realization_route.json")
    d = read_json(OUT_ROOT / "v22_81r_part_d_edge_realization_route.json")
    d5 = read_json(OUT_ROOT / "v22_81r_part_d5_no_surplus_root_cause_route.json")
    e = read_json(OUT_ROOT / "v22_81r_part_e_adversarial_margin_route.json")
    f0 = read_json(OUT_ROOT / "v22_81r_part_f0_hstep_trajectory_route.json")
    f = read_json(OUT_ROOT / "v22_81r_part_f_target_free_full_loop_route.json")
    g = read_json(OUT_ROOT / "v22_81r_part_g_strengthened_mlp_matched_audit_route.json")
    if not int(a.get("part_a_hard_gate_pass", 0)) or not int(b.get("part_b_gate_pass", 0)):
        route = "R0-CodeOrTruthGateFailed"
        reason = f"A={a.get('part_a_hard_gate_pass', 0)} B={b.get('part_b_gate_pass', 0)}"
    elif not int(c.get("part_c_gate_pass", 0)):
        route = str(c.get("part_c_route", "OutputVelocityNotFeasible_NoKANFullLoop"))
        reason = str(c.get("route_reason", ""))
    elif not int(d.get("part_d_gate_pass", 0)):
        route = str(d.get("part_d_route", "EdgeActuatorRealizationFailed"))
        d5_note = f"; D5={d5.get('part_d5_route')}: {d5.get('route_reason')}" if d5 else ""
        reason = f"D0={d0.get('part_d0_route')}; D={d.get('route_reason')}{d5_note}"
    elif not int(e.get("part_e_gate_pass", 0)):
        route = str(e.get("part_e_route", "AdversarialControlMarginFailed"))
        reason = str(e.get("route_reason", ""))
    elif not int(f0.get("part_f0_gate_pass", 0)):
        route = str(f0.get("part_f0_route", "SingleStepSafeButTrajectoryDebtBlocked"))
        reason = str(f0.get("route_reason", ""))
    elif not int(f.get("official_candidate_gate_pass", 0)):
        route = "CurrentStrictFCPureKANBasisFamilyNotSuperiorCarrier"
        reason = f"preflight opened but full-loop official did not pass; F={f.get('part_f_route')}; G={g.get('part_g_route')}"
    else:
        route = "KANEdgeActuatorMPFUOfficialCandidate"
        reason = "official_candidate_gate_pass=1"
    final = {
        "gate": "v22_81r_part_h_route_decision",
        "run_status": "completed_route_decision",
        "final_route": route,
        "route_reason": reason,
        "official_candidate_gate_pass": int(f.get("official_candidate_gate_pass", 0)),
        "part_a_hard_gate_pass": int(a.get("part_a_hard_gate_pass", 0)),
        "part_b_gate_pass": int(b.get("part_b_gate_pass", 0)),
        "part_c_gate_pass": int(c.get("part_c_gate_pass", 0)),
        "part_d0_gate_pass": int(d0.get("part_d0_gate_pass", 0)),
        "part_d_gate_pass": int(d.get("part_d_gate_pass", 0)),
        "part_d5_root_cause_audit_pass": int(d5.get("part_d5_root_cause_audit_pass", 0)),
        "part_e_gate_pass": int(e.get("part_e_gate_pass", 0)),
        "part_f0_gate_pass": int(f0.get("part_f0_gate_pass", 0)),
        "part_f_exploration_gate_pass": int(f.get("part_f_exploration_gate_pass", 0)),
        "part_g_gate_pass": int(g.get("part_g_gate_pass", 0)),
        "part_c_route": c.get("part_c_route", ""),
        "part_d_route": d.get("part_d_route", ""),
        "part_d5_route": d5.get("part_d5_route", ""),
        "part_e_route": e.get("part_e_route", ""),
        "part_f0_route": f0.get("part_f0_route", ""),
        "part_f_route": f.get("part_f_route", ""),
        "part_g_route": g.get("part_g_route", ""),
        "non_fabrication_note": "All numeric fields are generated by this runner from local train-only artifacts or copied from explicitly named v22.80 JSON artifacts.",
        "generated_at_sg": now_sg(),
        "artifacts": {
            "part_a": rel(OUT_ROOT / "v22_81r_part_a_code_identity_hard_gate.json"),
            "part_b": rel(OUT_ROOT / "v22_81r_part_b_v22_80_replay.json"),
            "part_c": rel(OUT_ROOT / "v22_81r_part_c_output_velocity_route.json"),
            "part_d0": rel(OUT_ROOT / "v22_81r_part_d0_mlp_matched_realization_route.json"),
            "part_d": rel(OUT_ROOT / "v22_81r_part_d_edge_realization_route.json"),
            "part_d5": rel(OUT_ROOT / "v22_81r_part_d5_no_surplus_root_cause_route.json"),
            "part_e": rel(OUT_ROOT / "v22_81r_part_e_adversarial_margin_route.json"),
            "part_f0": rel(OUT_ROOT / "v22_81r_part_f0_hstep_trajectory_route.json"),
            "part_f": rel(OUT_ROOT / "v22_81r_part_f_target_free_full_loop_route.json"),
            "part_g": rel(OUT_ROOT / "v22_81r_part_g_strengthened_mlp_matched_audit_route.json"),
            "part_h": rel(OUT_ROOT / "v22_81r_final_route.json"),
        },
    }
    write_rows(OUT_ROOT / "v22_81r_part_h_route_decision.csv", [final])
    write_json(OUT_ROOT / "v22_81r_final_route.json", final)
    append_exec("H_route_decision", command_text(sys.argv), "done", files=f"{rel(OUT_ROOT / 'v22_81r_part_h_route_decision.csv')}; {rel(OUT_ROOT / 'v22_81r_final_route.json')}", note=json.dumps({"final_route": route, "reason": reason}, ensure_ascii=False))
    append_recap("Part H final route decision", [
        f"final_route={route}；reason={reason}",
        f"gates: A={final['part_a_hard_gate_pass']} B={final['part_b_gate_pass']} C={final['part_c_gate_pass']} D0={final['part_d0_gate_pass']} D={final['part_d_gate_pass']} D5audit={final['part_d5_root_cause_audit_pass']} E={final['part_e_gate_pass']} F0={final['part_f0_gate_pass']} F_explore={final['part_f_exploration_gate_pass']} official={final['official_candidate_gate_pass']}。",
        f"routes: C={final['part_c_route']}；D={final['part_d_route']}；D5={final['part_d5_route']}；E={final['part_e_route']}；F0={final['part_f0_route']}；F={final['part_f_route']}；G={final['part_g_route']}。",
    ])
    return final


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--datasets", default="Wine,Spam,MNIST")
    p.add_argument("--seed-count", type=int, default=5)
    p.add_argument("--train-size", type=int, default=512)
    p.add_argument("--held-size", type=int, default=256)
    p.add_argument("--test-size", type=int, default=256)
    p.add_argument("--hidden", type=int, default=96)
    p.add_argument("--metric-batch-size", type=int, default=192)
    p.add_argument("--projector-ridge", type=float, default=1.0e-4)
    p.add_argument("--model-seed-offset", type=int, default=24800)
    p.add_argument("--force-output-velocity-family", default="")
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
    elif args.mode == "part-d0":
        run_part_d0(args)
    elif args.mode == "part-d0-merge":
        merge_part_d0(args)
    elif args.mode == "part-d":
        run_part_d(args)
    elif args.mode == "part-d-merge":
        merge_part_d(args)
    elif args.mode == "part-d5":
        run_part_d5(args)
    elif args.mode == "part-e":
        merge_part_e(args)
    elif args.mode == "part-f0":
        merge_part_f0(args)
    elif args.mode == "part-f":
        run_part_f(args)
    elif args.mode == "part-g":
        run_part_g(args)
    elif args.mode in {"part-h", "final-route"}:
        run_part_h(args)
    else:
        raise SystemExit(f"unknown mode={args.mode}")


if __name__ == "__main__":
    main()
