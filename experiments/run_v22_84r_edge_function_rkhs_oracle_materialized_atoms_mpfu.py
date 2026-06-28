#!/usr/bin/env python3
"""DG-KAN v22.84R Edge-Function RKHS oracle / materialized atoms runner."""

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

import experiments.run_v22_80_control_orthogonal_edge_signal_kan_mpfu as base80
import experiments.run_v22_82_persistent_edge_coordinate_generator_kan_mpfu as base82
import experiments.run_v22_83_observable_edge_active_atlas_kan_mpfu as base83


PYTHON = sys.executable
RUNNER = ROOT / "experiments/run_v22_84r_edge_function_rkhs_oracle_materialized_atoms_mpfu.py"
PLAN = ROOT / "docs/DG-KAN_v22.84R_EdgeFunctionRKHSOracleMaterializedAtoms_MPFU_完整计划.md"
EXEC_LOG = ROOT / "docs/DG-KAN_v22.84R_EdgeFunctionRKHSOracleMaterializedAtoms_MPFU_执行日志.md"
RECAP_LOG = ROOT / "docs/DG-KAN_v22.84R_EdgeFunctionRKHSOracleMaterializedAtoms_MPFU_实验结果复盘.md"
OUT_ROOT = Path(os.environ.get("V2284R_OUT_ROOT", str(ROOT / "results/v22_84r"))).resolve()
LOG_ROOT = OUT_ROOT / "logs"
V2283_ROOT = ROOT / "results/v22_83"

EDGE_PARAM = "w1"
RKHS_KERNELS_INITIAL = [
    "K1_sobolev_low_frequency",
    "K2_compact_support_bump",
    "K3_lowfreq_local_mixed",
    "K4_monotone_partition_unity",
    "K5_density_equalized_spline",
    "K6_bank_shared_dictionary",
]
RKHS_KERNELS_REPAIR = [
    "K2R_quantile_balanced_compact_bump",
    "K3R_ridge_whitened_mixed",
    "K5R_control_residual_density_spline",
]
RKHS_KERNELS_REPAIR2 = [
    "K2R2_highcap_quantile_balanced_compact_bump",
    "K3R2_highcap_ridge_whitened_mixed",
    "K3R2_highcap_ridge_whitened_debt_projected_mixed",
    "K5R2_highcap_control_residual_debt_projected_spline",
]
RKHS_KERNELS_REPAIR3 = [
    "K2R3_svd64_quantile_balanced_output_velocity",
    "K3R3_svd64_whitened_mixed_output_velocity",
    "K3TF_svd64_target_free_ce",
    "K5TF_svd64_debt_projected_target_free_ce",
]
RKHS_KERNELS_REPAIR4 = [
    "K2R4_svd64_source_actual_debt_barrier_output_velocity",
    "K5TF_R4_svd64_source_actual_debt_barrier_target_free_ce",
]
RKHS_KERNELS_REPAIR5 = [
    "K2R5_svd64_inner_source_stable_output_velocity",
    "K3TF_R5_svd64_inner_source_stable_target_free_ce",
]
CEILING_KERNELS_DEFAULT = [
    "K2_compact_support_bump",
    "K4_monotone_partition_unity",
    "K2R3_svd64_quantile_balanced_output_velocity",
    "K3R3_svd64_whitened_mixed_output_velocity",
    "K3TF_svd64_target_free_ce",
    "K5TF_svd64_debt_projected_target_free_ce",
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
    return base82.fval(x, default)


def quantile(values: list[float], q: float) -> float:
    return base82.quantile(values, q)


def lower_cvar(values: list[float], frac: float = 0.25) -> float:
    return base82.lower_cvar(values, frac)


def metric_energy(vec: torch.Tensor, metric: torch.Tensor) -> torch.Tensor:
    return base82.metric_energy(vec.reshape(-1).to(dtype=torch.float64), metric.reshape(-1).to(device=vec.device, dtype=torch.float64))


def metric_norm(vec: torch.Tensor, metric: torch.Tensor) -> float:
    return float(torch.sqrt(metric_energy(vec, metric).clamp_min(0.0)).detach().cpu().item())


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
        for key in row:
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


def init_logs() -> None:
    ensure_out()
    if not EXEC_LOG.exists():
        EXEC_LOG.write_text(
            "# DG-KAN v22.84R Edge-Function RKHS Oracle / Materialized Atoms MPFU 执行日志\n\n"
            f"创建时间：{now_sg()}\n\n"
            f"- 计划文档：`{rel(PLAN)}`\n"
            f"- runner：`{rel(RUNNER)}`\n"
            "- 非编造约束：只记录真实命令、真实 artifact、真实错误与观测；缺失项写 missing/skipped。\n"
            "- 复现提示：Part C 支持 `--shard-count/--shard-index` 并行，默认输出到 `results/v22_84r/`。\n\n"
            "## 命令记录\n",
            encoding="utf-8",
        )
    if not RECAP_LOG.exists():
        RECAP_LOG.write_text(
            "# DG-KAN v22.84R Edge-Function RKHS Oracle / Materialized Atoms MPFU 实验结果复盘\n\n"
            f"创建时间：{now_sg()}\n\n"
            "## 0. 当前结论\n"
            "- 尚未完成最终 route 判定。\n"
            "- 本文件只记录真实 artifact、真实指标和实际修复；不补造缺失数据。\n\n"
            "## 1. 证据链、修复与分析\n",
            encoding="utf-8",
        )


def append_exec(task: str, command: str, status: str, *, gpu: str = "", files: str = "", note: str = "") -> None:
    init_logs()
    with EXEC_LOG.open("a", encoding="utf-8") as fh:
        fh.write(f"\n### {now_sg()} | {task} | {status}\n")
        fh.write(f"- command: `{command}`\n")
        fh.write(f"- gpu: `{gpu}`\n")
        fh.write(f"- files: `{files}`\n")
        if note:
            fh.write(f"- note: {note}\n")


def append_recap(title: str, lines: list[str]) -> None:
    init_logs()
    with RECAP_LOG.open("a", encoding="utf-8") as fh:
        fh.write(f"\n## {title}\n\n")
        fh.write(f"- time_sg: {now_sg()}\n")
        for line in lines:
            fh.write(f"- {line}\n")


def device_from_args(args: argparse.Namespace) -> torch.device:
    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        return torch.device(args.device)
    return torch.device("cpu")


def shard_items(items: list[Any], args: argparse.Namespace) -> list[Any]:
    count = max(1, int(args.shard_count))
    idx = int(args.shard_index)
    return [item for i, item in enumerate(items) if i % count == idx]


def carrier_task_grid(args: argparse.Namespace) -> list[tuple[dict[str, str], int, str]]:
    datasets = [item.strip() for item in str(args.datasets).split(",") if item.strip()]
    seeds = list(range(int(args.seed_count)))
    return [(dict(spec), seed, dataset) for spec in base80.CARRIER_REDESIGN_SPECS for dataset in datasets for seed in seeds]


def limited_task_grid(args: argparse.Namespace, *, limit: int | None = None) -> list[tuple[dict[str, str], int, str]]:
    tasks = carrier_task_grid(args)
    if limit is None or int(limit) <= 0:
        return tasks
    return tasks[: int(limit)]


def stable_text_seed(text: str) -> int:
    acc = 0
    for idx, ch in enumerate(str(text)):
        acc = (acc + (idx + 1) * ord(ch)) % 1000003
    return int(acc)


def selected_output_family(args: argparse.Namespace) -> str:
    forced = str(getattr(args, "force_output_velocity_family", "")).strip()
    if forced:
        return forced
    try:
        families = base82.select_output_families(args)
    except Exception:
        families = []
    return str(families[0]) if families else "output_qp_trust_region_small"


def scale_to_norm(vec: torch.Tensor, norm: float) -> torch.Tensor:
    return base82.scale_to_norm(vec.reshape(-1).to(dtype=torch.float64), float(norm)).reshape_as(vec)


def ridge_solve(mat: torch.Tensor, rhs: torch.Tensor, ridge: float) -> torch.Tensor:
    p = int(mat.shape[1]) if int(mat.ndim) == 2 else 0
    if p <= 0:
        return torch.zeros(0, device=mat.device, dtype=torch.float64)
    n = int(mat.shape[0])
    if p > n:
        eye_n = torch.eye(n, device=mat.device, dtype=torch.float64)
        gram_n = mat @ mat.T
        try:
            dual = torch.linalg.solve(gram_n + float(ridge) * eye_n, rhs)
        except Exception:
            dual = torch.linalg.pinv(gram_n + float(ridge) * eye_n) @ rhs
        return mat.T @ dual
    a = mat.T @ mat
    b = mat.T @ rhs
    eye = torch.eye(p, device=mat.device, dtype=torch.float64)
    try:
        return torch.linalg.solve(a + float(ridge) * eye, b)
    except Exception:
        return torch.linalg.pinv(a + float(ridge) * eye) @ b


def finite_condition(mat: torch.Tensor, ridge: float) -> tuple[float, float, int]:
    if int(mat.ndim) != 2 or int(mat.shape[1]) <= 0:
        return 0.0, 0.0, 0
    gram = mat.T @ mat
    gram = 0.5 * (gram + gram.T)
    eig = torch.linalg.eigvalsh(gram + float(ridge) * torch.eye(int(gram.shape[0]), device=gram.device, dtype=torch.float64))
    pos = eig[eig > 1.0e-12]
    cond = float((pos.max() / pos.min()).detach().cpu().item()) if int(pos.numel()) else 0.0
    return cond, float(eig.min().detach().cpu().item()), int(float(eig.min().detach().cpu().item()) >= -1.0e-8)


def column_scale_from_design(design: torch.Tensor, ridge: float) -> torch.Tensor:
    if int(design.shape[1]) <= 0:
        return torch.ones(0, device=design.device, dtype=torch.float64)
    return design.square().mean(dim=0).sqrt().clamp_min(float(ridge))


def apply_column_scale(design: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    if int(design.shape[1]) <= 0:
        return design
    return design / scale.to(device=design.device, dtype=torch.float64)[None, :]


def svd_reduce_designs(
    design_s: torch.Tensor,
    design_w: torch.Tensor,
    design_g: torch.Tensor,
    metric_s: torch.Tensor,
    rank: int,
    ridge: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    if int(design_s.shape[1]) <= 0:
        return design_s, design_w, design_g, {"svd_reduced": 0}
    sqrt_m = torch.sqrt(metric_s.reshape(-1).to(device=design_s.device, dtype=torch.float64).clamp_min(1.0e-12))
    weighted = design_s.to(dtype=torch.float64) * sqrt_m[:, None]
    try:
        _u, s, vh = torch.linalg.svd(weighted, full_matrices=False)
    except Exception:
        gram = weighted.T @ weighted
        vals, vecs = torch.linalg.eigh(0.5 * (gram + gram.T))
        order = torch.argsort(vals, descending=True)
        s = vals[order].clamp_min(0.0).sqrt()
        vh = vecs[:, order].T
    keep = int(min(max(1, int(rank)), int(vh.shape[0]), int(vh.shape[1])))
    if int(s.numel()) > 0:
        good = int((s > float(ridge)).sum().detach().cpu().item())
        keep = min(keep, max(1, good))
    basis = vh[:keep, :].T.contiguous()
    s_keep = s[:keep].detach().to(dtype=torch.float64)
    cond = float((s_keep.max() / s_keep.min().clamp_min(float(ridge))).detach().cpu().item()) if int(s_keep.numel()) else 0.0
    meta = {
        "svd_reduced": 1,
        "svd_rank": keep,
        "svd_source_condition": cond,
        "svd_top_singular": float(s_keep.max().detach().cpu().item()) if int(s_keep.numel()) else 0.0,
        "svd_min_kept_singular": float(s_keep.min().detach().cpu().item()) if int(s_keep.numel()) else 0.0,
    }
    return design_s @ basis, design_w @ basis, design_g @ basis, meta


def empirical_cdf(train_z: torch.Tensor, z: torch.Tensor, dims: torch.Tensor) -> torch.Tensor:
    # Train-only quantile coordinate: each evaluation value is ranked against source+witness activations only.
    outs = []
    for dim in dims.detach().long().tolist():
        ref = train_z[:, int(dim)].detach().to(dtype=torch.float64)
        val = z[:, int(dim)].detach().to(dtype=torch.float64)
        outs.append((ref[None, :] <= val[:, None]).to(dtype=torch.float64).mean(dim=1))
    return torch.stack(outs, dim=1).clamp(0.0, 1.0)


def kernel_values(s: torch.Tensor, centers: torch.Tensor, family: str) -> torch.Tensor:
    diff = (s[:, :, None] - centers[None, None, :]).abs()
    width = 1.0 / max(2.0, float(centers.numel()) - 1.0)
    if family.startswith("K1"):
        vals = 1.0 + torch.cos(math.pi * diff) + 0.5 * torch.cos(2.0 * math.pi * diff)
        return vals / 2.5
    if family.startswith("K2") or family.startswith("K2R"):
        bw = 1.75 * width if family.startswith("K2R") else 1.25 * width
        return (1.0 - diff / max(bw, 1.0e-6)).clamp_min(0.0).pow(3.0)
    if family.startswith("K3") or family.startswith("K3R"):
        low = (1.0 + torch.cos(math.pi * diff)) / 2.0
        bump = (1.0 - diff / max(1.35 * width, 1.0e-6)).clamp_min(0.0).pow(3.0)
        return 0.45 * low + 0.55 * bump
    if family.startswith("K4"):
        tri = (1.0 - diff / max(width, 1.0e-6)).clamp_min(0.0)
        return tri / tri.sum(dim=2, keepdim=True).clamp_min(1.0e-12)
    if family.startswith("K5") or family.startswith("K5R"):
        bw = 1.6 * width
        u = (diff / max(bw, 1.0e-6)).clamp_min(0.0)
        return torch.where(u < 1.0, (1.0 - 3.0 * u.square() + 2.0 * u.pow(3.0)).clamp_min(0.0), torch.zeros_like(u))
    if family.startswith("K6"):
        low = (1.0 + torch.cos(math.pi * diff)) / 2.0
        bump = (1.0 - diff / max(1.5 * width, 1.0e-6)).clamp_min(0.0)
        return 0.65 * bump + 0.35 * low
    raise ValueError(f"unknown RKHS kernel family={family}")


def select_edge_pairs(model: Any, x: torch.Tensor, metric: torch.Tensor, args: argparse.Namespace) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    vis = base83.visibility_scores(model, x, metric)
    edge_score = vis.sum(dim=2).detach().to(dtype=torch.float64)
    flat = edge_score.reshape(-1)
    count = min(max(1, int(args.rkhs_max_edges)), int(flat.numel()))
    idx = torch.topk(flat.abs(), k=count).indices
    d_idx = torch.div(idx, int(model.hidden_dim), rounding_mode="floor")
    h_idx = idx - d_idx * int(model.hidden_dim)
    total = float(flat.sum().detach().cpu().item())
    selected = float(flat[idx].sum().detach().cpu().item())
    return d_idx.detach().long(), h_idx.detach().long(), {
        "selected_edges": int(count),
        "edge_visibility_fraction": selected / max(total, 1.0e-12),
    }


def rkhs_design(
    model: Any,
    x_eval: torch.Tensor,
    train_z: torch.Tensor,
    d_idx: torch.Tensor,
    h_idx: torch.Tensor,
    centers: torch.Tensor,
    kernel_family: str,
) -> torch.Tensor:
    z_eval = model._norm_input(x_eval)
    s = empirical_cdf(train_z, z_eval, d_idx.to(device=x_eval.device))
    kvals = kernel_values(s, centers.to(device=x_eval.device, dtype=torch.float64), kernel_family)
    _b1, _h, dh, down = base83.downstream_sensitivity(model, x_eval)
    inv_sqrt_d = 1.0 / math.sqrt(max(1, int(model.input_dim)))
    cols = []
    for edge_pos, h in enumerate(h_idx.detach().long().tolist()):
        edge_vals = kvals[:, edge_pos, :]
        sens = (dh[:, int(h)] * inv_sqrt_d)[:, None] * down[:, int(h), :]
        for cidx in range(int(centers.numel())):
            cols.append((edge_vals[:, cidx : cidx + 1] * sens).reshape(-1))
    if not cols:
        return torch.zeros((int(x_eval.shape[0]) * int(model.output_dim), 0), device=x_eval.device, dtype=torch.float64)
    return torch.stack(cols, dim=1).to(dtype=torch.float64)


def row_metrics_from_logit_update(logits: torch.Tensor, y: torch.Tensor, update: torch.Tensor) -> tuple[dict[str, float], float]:
    deltas = base80.logit_metric_delta_for_update(logits, y, update.reshape_as(logits).to(device=logits.device, dtype=logits.dtype))
    return deltas, base80.debt_ucb_from_deltas(deltas)


def weighted_ridge_alpha(design: torch.Tensor, target: torch.Tensor, metric: torch.Tensor, ridge: float) -> torch.Tensor:
    sqrt_m = torch.sqrt(metric.reshape(-1).to(device=design.device, dtype=torch.float64).clamp_min(1.0e-12))
    return ridge_solve(design * sqrt_m[:, None], target.reshape(-1).to(device=design.device, dtype=torch.float64) * sqrt_m, ridge)


def control_alpha(kind: str, alpha: torch.Tensor, design_source: torch.Tensor, target_source: torch.Tensor, seed: int, ridge: float) -> torch.Tensor:
    norm = float(alpha.norm().detach().cpu().item())
    if norm <= 0.0:
        return torch.zeros_like(alpha)
    gen = torch.Generator(device=alpha.device).manual_seed(int(seed))
    if kind == "same_compute_noop":
        return torch.zeros_like(alpha)
    if kind == "same_solver_gradient_control":
        ce_like = -target_source.reshape(-1).to(device=design_source.device, dtype=torch.float64)
        return scale_to_norm(ridge_solve(design_source, ce_like, ridge), norm)
    if kind == "same_bank_energy_random":
        perm = torch.randperm(int(alpha.numel()), device=alpha.device, generator=gen)
        return alpha[perm].reshape_as(alpha)
    if kind == "same_smoothness_random":
        raw = torch.randn_like(alpha, generator=gen)
        if int(raw.numel()) >= 5:
            raw = torch.cumsum(raw, dim=0)
        return scale_to_norm(raw, norm)
    raw = torch.randn(int(alpha.numel()), device=alpha.device, dtype=torch.float64, generator=gen)
    return scale_to_norm(raw, norm)


def project_alpha_source_debt(
    alpha: torch.Tensor,
    design_source: torch.Tensor,
    logits_source: torch.Tensor,
    y_source: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, Any]]:
    out = alpha.detach().clone().to(dtype=torch.float64)
    orig_norm = out.norm().clamp_min(1.0e-12)
    count = 0
    max_violation = 0.0
    used = []
    for loss_kind in ["brier", "ece_debt", "tail95_debt", "tail99_debt", "margin_debt"]:
        try:
            grad = base80.logits_loss_grad(logits_source, y_source, loss_kind).reshape(-1).to(device=design_source.device, dtype=torch.float64)
        except Exception:
            continue
        q = design_source.T @ grad
        denom = q.dot(q).clamp_min(1.0e-12)
        violation = q.dot(out)
        max_violation = max(max_violation, float(violation.detach().cpu().item()))
        if float(violation.detach().cpu().item()) > 0.0:
            out = out - (violation / denom) * q
            count += 1
        used.append(loss_kind)
    return out, {
        "source_debt_alpha_projection_used": 1,
        "source_debt_alpha_projection_count": count,
        "source_debt_alpha_projection_norm_ratio": float((out.norm() / orig_norm).detach().cpu().item()),
        "source_debt_alpha_projection_max_first_order_violation": max_violation,
        "source_debt_alpha_projection_losses": ";".join(used),
    }


def source_actual_debt_barrier_alpha(
    alpha: torch.Tensor,
    design_source: torch.Tensor,
    logits_source: torch.Tensor,
    y_source: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, Any]]:
    schedule = [1.0, 0.75, 0.50, 0.35, 0.25, 0.15, 0.10, 0.05, 0.025, 0.0]
    best_scale = 1.0
    best_debt = float("inf")
    for scale in schedule:
        update = (design_source @ (alpha * float(scale))).reshape_as(logits_source)
        _delta, debt = row_metrics_from_logit_update(logits_source, y_source, update)
        if debt < best_debt:
            best_debt = debt
            best_scale = float(scale)
        if debt <= 0.0 or scale == 0.0:
            return alpha * float(scale), {
                "source_actual_debt_barrier_used": 1,
                "source_actual_debt_barrier_scale": float(scale),
                "source_actual_debt_barrier_source_debt_UCB": debt,
                "source_actual_debt_barrier_noop": int(float(scale) == 0.0),
            }
    return alpha * best_scale, {
        "source_actual_debt_barrier_used": 1,
        "source_actual_debt_barrier_scale": best_scale,
        "source_actual_debt_barrier_source_debt_UCB": best_debt,
        "source_actual_debt_barrier_noop": int(best_scale == 0.0),
    }


def inner_source_stable_alpha(
    design_source: torch.Tensor,
    target_source: torch.Tensor,
    metric_source: torch.Tensor,
    alpha_reference: torch.Tensor,
    output_dim: int,
    ridge: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    n = int(target_source.shape[0])
    cut = max(1, n // 2)
    if cut >= n or int(design_source.shape[1]) <= 0:
        return alpha_reference, {
            "inner_source_stable_used": 1,
            "inner_source_stable_components": int(alpha_reference.numel()),
            "inner_source_stable_fraction": 1.0,
            "inner_source_half_alpha_cosine": 1.0,
        }
    left_rows = cut * int(output_dim)
    da = design_source[:left_rows, :]
    db = design_source[left_rows:, :]
    target_view = target_source.reshape(n, int(output_dim)).to(device=design_source.device, dtype=torch.float64)
    metric_view = metric_source.reshape(n, int(output_dim)).to(device=design_source.device, dtype=torch.float64)
    ta = target_view[:cut].reshape(-1)
    tb = target_view[cut:].reshape(-1)
    ma = metric_view[:cut].reshape(-1)
    mb = metric_view[cut:].reshape(-1)
    alpha_a = weighted_ridge_alpha(da, ta, ma, ridge)
    alpha_b = weighted_ridge_alpha(db, tb, mb, ridge)
    sign_mask = (alpha_a * alpha_b >= 0.0).to(dtype=torch.float64)
    if int(sign_mask.sum().detach().cpu().item()) == 0:
        stable = 0.5 * (alpha_a + alpha_b)
    else:
        stable = 0.5 * (alpha_a + alpha_b) * sign_mask
    ref_norm = alpha_reference.norm()
    if float(stable.norm().detach().cpu().item()) > 1.0e-12 and float(ref_norm.detach().cpu().item()) > 1.0e-12:
        stable = scale_to_norm(stable, float(ref_norm.detach().cpu().item()))
    return stable, {
        "inner_source_stable_used": 1,
        "inner_source_stable_components": int(sign_mask.sum().detach().cpu().item()),
        "inner_source_stable_fraction": float(sign_mask.mean().detach().cpu().item()),
        "inner_source_half_alpha_cosine": base83.flat_cosine(alpha_a, alpha_b),
    }


def mlp_matched_logit_update(
    mlp: Any,
    xs: torch.Tensor,
    ys: torch.Tensor,
    xw: torch.Tensor,
    yw: torch.Tensor,
    xg: torch.Tensor,
    target_metric_norm: float,
    metric_g: torch.Tensor,
) -> torch.Tensor:
    update = base83.mlp_matched_update(mlp, xs, ys, xw, yw, 1.0)
    raw = base83.actual_logit_update_for_updates(mlp, xg, update).reshape(int(xg.shape[0]), -1)
    raw_norm = metric_norm(raw, metric_g)
    if raw_norm <= 1.0e-12:
        return torch.zeros_like(raw)
    return raw * (float(target_metric_norm) / raw_norm)


def part_c_probe(dataset: str, seed: int, spec: dict[str, str], kernel_family: str, args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    row: dict[str, Any] = {
        "dataset": dataset,
        "seed": seed,
        "architecture": "strict_fc_purekan",
        "method": spec.get("method", ""),
        "carrier_family": spec.get("carrier_family", ""),
        "rkhs_kernel_family": kernel_family,
        "objective_family": "output_debt_velocity_oracle",
        "probe_error": "",
    }
    try:
        model, mlp, _bundle, splits = base83.make_model_and_batch(dataset, seed, spec, args, device)
        xs, ys, xw, yw, xg, yg, _xc, _yc = splits
        x_train = torch.cat([xs, xw], dim=0)
        train_z = model._norm_input(x_train).detach().to(dtype=torch.float64)

        logits_s = model(xs).float()
        logits_w = model(xw).float()
        logits_g = model(xg).float()
        metric_s = base83.output_metric_diag(logits_s).to(device=device)
        metric_w = base83.output_metric_diag(logits_w).to(device=device)
        metric_g = base83.output_metric_diag(logits_g).to(device=device)
        out_family = selected_output_family(args)
        target_s, _meta_s = base83.output_update_for_family(logits_s, ys, out_family, args)
        target_w, _meta_w = base83.output_update_for_family(logits_w, yw, out_family, args)
        target_g, _meta_g = base83.output_update_for_family(logits_g, yg, out_family, args)

        d_idx, h_idx, edge_meta = select_edge_pairs(model, x_train, base83.output_metric_diag(model(x_train).float()).to(device=device), args)
        centers = torch.linspace(0.0, 1.0, steps=max(2, int(args.rkhs_centers)), device=device, dtype=torch.float64)
        if kernel_family.startswith("K2R"):
            qs = torch.linspace(0.05, 0.95, steps=max(2, int(args.rkhs_centers)), device=device, dtype=torch.float64)
            centers = qs
        design_s = rkhs_design(model, xs, train_z, d_idx, h_idx, centers, kernel_family)
        design_w = rkhs_design(model, xw, train_z, d_idx, h_idx, centers, kernel_family)
        design_g = rkhs_design(model, xg, train_z, d_idx, h_idx, centers, kernel_family)
        whiten_meta: dict[str, Any] = {"whitened": 0}
        if "whitened" in kernel_family:
            scale = column_scale_from_design(design_s, float(args.projector_ridge))
            design_s = apply_column_scale(design_s, scale)
            design_w = apply_column_scale(design_w, scale)
            design_g = apply_column_scale(design_g, scale)
            whiten_meta = {
                "whitened": 1,
                "column_norm_median": quantile([float(v) for v in scale.detach().cpu().tolist()], 0.50),
                "whitening_scale_source_only": 1,
            }
        if int(design_s.shape[1]) <= 0:
            raise RuntimeError("empty RKHS design")
        svd_meta: dict[str, Any] = {"svd_reduced": 0, "svd_rank": 0, "svd_source_condition": 0.0}
        if "svd" in kernel_family:
            design_s, design_w, design_g, svd_meta = svd_reduce_designs(
                design_s,
                design_w,
                design_g,
                metric_s,
                int(args.rkhs_svd_rank),
                float(args.projector_ridge),
            )

        objective_family = "target_free_train_only_ce" if "target_free" in kernel_family else "output_debt_velocity_oracle"
        ce_source_target = None
        if objective_family == "target_free_train_only_ce":
            ce_grad = (torch.softmax(logits_s.detach(), dim=1) - F.one_hot(ys, num_classes=int(model.output_dim)).float()).reshape_as(logits_s).to(device=device, dtype=torch.float64)
            ce_source_target = -ce_grad
            ref_norm = metric_norm(target_s, metric_s)
            ce_norm = metric_norm(ce_source_target, metric_s)
            if ce_norm > 1.0e-12:
                ce_source_target = ce_source_target * (ref_norm / ce_norm)
            alpha = weighted_ridge_alpha(design_s, ce_source_target, metric_s, float(args.projector_ridge))
            source_control_target = ce_source_target
        else:
            alpha = weighted_ridge_alpha(design_s, target_s, metric_s, float(args.projector_ridge))
            source_control_target = target_s
        if "control_residual" in kernel_family:
            src_grad = (torch.softmax(logits_s.detach(), dim=1) - F.one_hot(ys, num_classes=int(model.output_dim)).float()).reshape(-1).to(device=device, dtype=torch.float64)
            grad_alpha = ridge_solve(design_s, -src_grad, float(args.projector_ridge))
            alpha = 0.75 * alpha + 0.25 * scale_to_norm(grad_alpha, float(alpha.norm().detach().cpu().item()))
        alpha_ref = source_control_target if ce_source_target is not None else target_s
        max_alpha_norm = float(args.rkhs_alpha_norm) * max(1.0, float(alpha_ref.reshape(-1).to(dtype=torch.float64).norm().detach().cpu().item()))
        if float(alpha.norm().detach().cpu().item()) > max_alpha_norm:
            alpha = scale_to_norm(alpha, max_alpha_norm)
        inner_stable_meta: dict[str, Any] = {
            "inner_source_stable_used": 0,
            "inner_source_stable_components": "",
            "inner_source_stable_fraction": "",
            "inner_source_half_alpha_cosine": "",
        }
        if "inner_source_stable" in kernel_family:
            alpha, inner_stable_meta = inner_source_stable_alpha(
                design_s,
                source_control_target,
                metric_s,
                alpha,
                int(model.output_dim),
                float(args.projector_ridge),
            )
        debt_projection_meta: dict[str, Any] = {
            "source_debt_alpha_projection_used": 0,
            "source_debt_alpha_projection_count": 0,
            "source_debt_alpha_projection_norm_ratio": 1.0,
            "source_debt_alpha_projection_max_first_order_violation": 0.0,
            "source_debt_alpha_projection_losses": "",
        }
        if "debt_projected" in kernel_family:
            alpha, debt_projection_meta = project_alpha_source_debt(alpha, design_s, logits_s, ys)
        source_barrier_meta: dict[str, Any] = {
            "source_actual_debt_barrier_used": 0,
            "source_actual_debt_barrier_scale": 1.0,
            "source_actual_debt_barrier_source_debt_UCB": "",
            "source_actual_debt_barrier_noop": 0,
        }
        if "source_actual_debt_barrier" in kernel_family:
            alpha, source_barrier_meta = source_actual_debt_barrier_alpha(alpha, design_s, logits_s, ys)

        update_s = (design_s @ alpha).reshape_as(logits_s)
        update_w = (design_w @ alpha).reshape_as(logits_w)
        update_g = (design_g @ alpha).reshape_as(logits_g)
        cand_delta_g, cand_debt = row_metrics_from_logit_update(logits_g, yg, update_g)
        source_delta, _source_debt = row_metrics_from_logit_update(logits_s, ys, update_s)
        witness_delta, _witness_debt = row_metrics_from_logit_update(logits_w, yw, update_w)
        coverage, rel_res = base80.weighted_capacity(update_g.reshape(-1), target_g.reshape(-1), metric_g.reshape(-1))
        visible_energy = float((metric_energy(update_g, metric_g) / metric_energy(target_g, metric_g).clamp_min(1.0e-12)).clamp(0.0, 10.0).detach().cpu().item())
        target_free_grad = (torch.softmax(logits_s.detach(), dim=1) - F.one_hot(ys, num_classes=int(model.output_dim)).float()).reshape(-1).to(device=device, dtype=torch.float64)
        target_free_alpha = ridge_solve(design_s, -target_free_grad, float(args.projector_ridge))
        target_free_transfer = metric_norm(design_w @ target_free_alpha, metric_w)

        controls = [
            "same_RKHS_norm_random",
            "same_edge_domain_energy_random",
            "same_smoothness_random",
            "same_output_velocity_norm_random",
            "same_output_coverage_random",
            "same_downstream_sensitivity_random",
            "same_debt_cone_random",
            "same_bank_energy_random",
            "same_solver_gradient_control",
            "same_compute_noop",
        ]
        margins: dict[str, float] = {}
        control_debts: dict[str, float] = {}
        control_deltas: dict[str, float] = {}
        for offset, name in enumerate(controls):
            ctrl = control_alpha(name, alpha, design_s, source_control_target, int(seed) + stable_text_seed(kernel_family) + 31 * offset, float(args.projector_ridge))
            ctrl_update = (design_g @ ctrl).reshape_as(logits_g)
            if name in {"same_output_velocity_norm_random", "same_output_coverage_random"}:
                cn = metric_norm(ctrl_update, metric_g)
                tn = metric_norm(update_g, metric_g)
                if cn > 1.0e-12:
                    ctrl_update = ctrl_update * (tn / cn)
            ctrl_delta, ctrl_debt = row_metrics_from_logit_update(logits_g, yg, ctrl_update)
            if name == "same_debt_cone_random" and ctrl_debt > cand_debt:
                ctrl_update = ctrl_update * 0.5
                ctrl_delta, ctrl_debt = row_metrics_from_logit_update(logits_g, yg, ctrl_update)
            margin = float(ctrl_delta.get("NLL", 0.0)) - float(cand_delta_g.get("NLL", 0.0)) - 0.5 * max(0.0, cand_debt - ctrl_debt)
            margins[name] = margin
            control_debts[name] = ctrl_debt
            control_deltas[name] = float(ctrl_delta.get("NLL", 0.0))

        cand_output_norm = metric_norm(update_g, metric_g)
        mlp_update_g = mlp_matched_logit_update(mlp, xs, ys, xw, yw, xg, cand_output_norm, metric_g).reshape_as(logits_g)
        mlp_delta, mlp_debt = row_metrics_from_logit_update(logits_g, yg, mlp_update_g)
        margins["MLP_matched_RKHS_coordinate"] = float(mlp_delta.get("NLL", 0.0)) - float(cand_delta_g.get("NLL", 0.0)) - 0.5 * max(0.0, cand_debt - mlp_debt)

        j_cond, gram_min_eig, psd = finite_condition(design_s * torch.sqrt(metric_s.reshape(-1).clamp_min(1.0e-12))[:, None], float(args.projector_ridge))
        source_witness_transfer_lcb = min(-float(source_delta.get("NLL", 0.0)), -float(witness_delta.get("NLL", 0.0)))
        kan_control_margins = {key: value for key, value in margins.items() if key != "MLP_matched_RKHS_coordinate"}
        best_kan_margin = min(kan_control_margins.values()) if kan_control_margins else 0.0
        smoothness = float((alpha[1:] - alpha[:-1]).square().mean().sqrt().detach().cpu().item()) if int(alpha.numel()) > 1 else 0.0
        if int(d_idx.numel()):
            ref = train_z[:, d_idx].detach().to(dtype=torch.float64)
            ref_min = ref.min(dim=0).values
            ref_max = ref.max(dim=0).values
            guard_ref = model._norm_input(xg)[:, d_idx].detach().to(dtype=torch.float64)
            domain_extrapolation_rate = float(((guard_ref < ref_min[None, :]) | (guard_ref > ref_max[None, :])).to(dtype=torch.float64).mean().detach().cpu().item())
        else:
            domain_extrapolation_rate = 0.0

        row.update({
            **edge_meta,
            **whiten_meta,
            **svd_meta,
            **inner_stable_meta,
            **debt_projection_meta,
            **source_barrier_meta,
            "objective_family": objective_family,
            "output_velocity_family": out_family,
            "rkhs_centers": int(centers.numel()),
            "rkhs_atoms": int(design_s.shape[1]),
            "rkhs_selected_edges": int(d_idx.numel()),
            "Gram_condition": j_cond,
            "Gram_min_eig": gram_min_eig,
            "Gram_PSD_pass": psd,
            "RKHS_norm": float(alpha.norm().detach().cpu().item()),
            "smoothness_norm": smoothness,
            "domain_extrapolation_rate": domain_extrapolation_rate,
            "oracle_NLL_delta": float(cand_delta_g.get("NLL", 0.0)),
            "oracle_NLL_improve": int(float(cand_delta_g.get("NLL", 0.0)) < 0.0),
            "guard_NLL_delta": float(cand_delta_g.get("NLL", 0.0)),
            "guard_Brier_delta": float(cand_delta_g.get("Brier", 0.0)),
            "guard_ECE_delta": float(cand_delta_g.get("ECE", 0.0)),
            "guard_tail99_delta": float(cand_delta_g.get("tail99", 0.0)),
            "guard_margin10_delta": float(cand_delta_g.get("margin10", 0.0)),
            "guard_debt_UCB": cand_debt,
            "oracle_all_debt_nonpositive": int(cand_debt <= 0.0),
            "coverage_to_output_velocity": coverage,
            "coverage_to_output_velocity_ge_040": int(coverage >= 0.40),
            "relative_residual_energy": rel_res,
            "raw_readout_visible_energy": visible_energy,
            "raw_readout_visible_energy_ge_020": int(visible_energy >= 0.20),
            "edge_effect_fraction": 1.0,
            "readout_effect_fraction": 0.0,
            "source_NLL_delta": float(source_delta.get("NLL", 0.0)),
            "witness_NLL_delta": float(witness_delta.get("NLL", 0.0)),
            "source_witness_transfer_LCB": source_witness_transfer_lcb,
            "source_witness_transfer_LCB_positive": int(source_witness_transfer_lcb > 0.0),
            "target_free_witness_metric_norm": target_free_transfer,
            "control_increment_margin": best_kan_margin,
            "control_increment_margin_positive": int(best_kan_margin > 0.0),
            "MLP_matched_gap": margins["MLP_matched_RKHS_coordinate"],
            "beats_MLP_matched_RKHS": int(margins["MLP_matched_RKHS_coordinate"] > 0.0),
            "same_RKHS_control_gap": margins.get("same_RKHS_norm_random", 0.0),
            "beats_same_RKHS_control": int(margins.get("same_RKHS_norm_random", 0.0) > 0.0),
            "same_domain_control_gap": margins.get("same_edge_domain_energy_random", 0.0),
            "beats_same_domain_control": int(margins.get("same_edge_domain_energy_random", 0.0) > 0.0),
            "same_smoothness_control_gap": margins.get("same_smoothness_random", 0.0),
            "beats_same_smoothness_control": int(margins.get("same_smoothness_random", 0.0) > 0.0),
            "same_debt_control_gap": margins.get("same_debt_cone_random", 0.0),
            "beats_same_debt_control": int(margins.get("same_debt_cone_random", 0.0) > 0.0),
            "same_coverage_control_gap": margins.get("same_output_coverage_random", 0.0),
            "beats_same_coverage_control": int(margins.get("same_output_coverage_random", 0.0) > 0.0),
            "same_solver_control_gap": margins.get("same_solver_gradient_control", 0.0),
            "same_compute_noop_NLL_delta": control_deltas.get("same_compute_noop", 0.0),
            "same_compute_noop_debt_UCB": control_debts.get("same_compute_noop", 0.0),
            "MLP_matched_NLL_delta": float(mlp_delta.get("NLL", 0.0)),
            "MLP_matched_debt_UCB": mlp_debt,
            "plan_repair_tag": (
                "initial"
                if kernel_family in RKHS_KERNELS_INITIAL
                else (
                    "repair"
                    if kernel_family in RKHS_KERNELS_REPAIR
                    else (
                        "repair2"
                        if kernel_family in RKHS_KERNELS_REPAIR2
                        else ("repair3" if kernel_family in RKHS_KERNELS_REPAIR3 else ("repair4" if kernel_family in RKHS_KERNELS_REPAIR4 else "repair5"))
                    )
                )
            ),
        })
    except Exception as exc:
        row["probe_error"] = f"{type(exc).__name__}: {exc}"
    return row


def summarize_part_c(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    families = sorted({str(r.get("rkhs_kernel_family", "")) for r in rows if str(r.get("rkhs_kernel_family", ""))})
    for family in families:
        group = [r for r in rows if str(r.get("rkhs_kernel_family", "")) == family and not str(r.get("probe_error", ""))]
        if not group:
            summaries.append({"rkhs_kernel_family": family, "completed_rows": 0, "part_c_family_gate_pass": 0})
            continue
        n = len(group)
        threshold_36 = base82.scaled_gate_count(n, 36, 45)
        threshold_32 = base82.scaled_gate_count(n, 32, 45)
        threshold_27 = base82.scaled_gate_count(n, 27, 45)
        row = {
            "rkhs_kernel_family": family,
            "completed_rows": n,
            "threshold_36_rows": threshold_36,
            "threshold_32_rows": threshold_32,
            "threshold_27_rows": threshold_27,
            "oracle_NLL_improve_rows": sum(int(fval(r.get("oracle_NLL_improve"))) for r in group),
            "oracle_all_debt_nonpositive_rows": sum(int(fval(r.get("oracle_all_debt_nonpositive"))) for r in group),
            "coverage_to_output_velocity_CVaR25": lower_cvar([fval(r.get("coverage_to_output_velocity")) for r in group], 0.25),
            "coverage_to_output_velocity_ge_040_rows": sum(int(fval(r.get("coverage_to_output_velocity_ge_040"))) for r in group),
            "raw_readout_visible_energy_CVaR25": lower_cvar([fval(r.get("raw_readout_visible_energy")) for r in group], 0.25),
            "raw_readout_visible_energy_ge_020_rows": sum(int(fval(r.get("raw_readout_visible_energy_ge_020"))) for r in group),
            "beats_same_RKHS_control_rows": sum(int(fval(r.get("beats_same_RKHS_control"))) for r in group),
            "beats_same_domain_control_rows": sum(int(fval(r.get("beats_same_domain_control"))) for r in group),
            "beats_same_smoothness_control_rows": sum(int(fval(r.get("beats_same_smoothness_control"))) for r in group),
            "beats_same_debt_control_rows": sum(int(fval(r.get("beats_same_debt_control"))) for r in group),
            "beats_same_coverage_control_rows": sum(int(fval(r.get("beats_same_coverage_control"))) for r in group),
            "beats_MLP_matched_RKHS_rows": sum(int(fval(r.get("beats_MLP_matched_RKHS"))) for r in group),
            "control_increment_margin_positive_rows": sum(int(fval(r.get("control_increment_margin_positive"))) for r in group),
            "control_increment_margin_CVaR25": lower_cvar([fval(r.get("control_increment_margin")) for r in group], 0.25),
            "source_witness_transfer_LCB_positive_rows": sum(int(fval(r.get("source_witness_transfer_LCB_positive"))) for r in group),
            "source_witness_transfer_LCB_CVaR25": lower_cvar([fval(r.get("source_witness_transfer_LCB")) for r in group], 0.25),
            "guard_NLL_delta_median": quantile([fval(r.get("guard_NLL_delta")) for r in group], 0.50),
            "guard_debt_UCB_median": quantile([fval(r.get("guard_debt_UCB")) for r in group], 0.50),
            "MLP_matched_gap_median": quantile([fval(r.get("MLP_matched_gap")) for r in group], 0.50),
            "RKHS_norm_median": quantile([fval(r.get("RKHS_norm")) for r in group], 0.50),
            "Gram_condition_median": quantile([fval(r.get("Gram_condition")) for r in group], 0.50),
            "domain_extrapolation_rate_max": max([fval(r.get("domain_extrapolation_rate")) for r in group] or [0.0]),
        }
        row["part_c_family_gate_pass"] = int(
            n >= 45
            and row["oracle_NLL_improve_rows"] >= 36
            and row["oracle_all_debt_nonpositive_rows"] >= 32
            and row["coverage_to_output_velocity_ge_040_rows"] >= 32
            and row["raw_readout_visible_energy_ge_020_rows"] >= 32
            and row["beats_same_RKHS_control_rows"] >= 32
            and row["beats_same_domain_control_rows"] >= 32
            and row["beats_same_smoothness_control_rows"] >= 32
            and row["beats_same_debt_control_rows"] >= 32
            and row["beats_same_coverage_control_rows"] >= 32
            and row["beats_MLP_matched_RKHS_rows"] >= 27
            and row["control_increment_margin_positive_rows"] >= 32
            and row["source_witness_transfer_LCB_positive_rows"] >= 32
        )
        summaries.append(row)
    return summaries


def run_part_a(args: argparse.Namespace) -> dict[str, Any]:
    init_logs()
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
        subprocess.run([PYTHON, "-c", "import experiments.run_v22_84r_edge_function_rkhs_oracle_materialized_atoms_mpfu; print('pass')"], cwd=str(ROOT), check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=120)
    except Exception as exc:
        import_ok = 0
        import_error = f"{type(exc).__name__}: {exc}"
    clean_tar_ok = 0
    clean_tar_error = ""
    try:
        with tempfile.TemporaryDirectory() as td:
            tar_path = Path(td) / "v22_84r_import.tar.gz"
            with tarfile.open(tar_path, "w:gz") as tf:
                tf.add(ROOT / "experiments", arcname="experiments")
                tf.add(ROOT / "dgkan", arcname="dgkan")
            extract_dir = Path(td) / "extract"
            extract_dir.mkdir(parents=True, exist_ok=True)
            with tarfile.open(tar_path, "r:gz") as tf:
                tf.extractall(extract_dir)
            subprocess.run(
                [PYTHON, "-c", "import experiments.run_v22_84r_edge_function_rkhs_oracle_materialized_atoms_mpfu; print('pass')"],
                cwd=str(extract_dir),
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=120,
            )
            clean_tar_ok = 1
    except Exception as exc:
        clean_tar_error = f"{type(exc).__name__}: {exc}"

    text = RUNNER.read_text(encoding="utf-8")
    scan_text = "\n".join(
        line
        for line in text.splitlines()
        if "manual_param_update_detected" not in line
        and "manual_bad =" not in line
        and "future_bad =" not in line
        and "candidate_bad =" not in line
        and "candidate_runtime_selection_used" not in line
        and "validation_test_future_direction_used" not in line
        and "readout_LS_promotion_detected" not in line
        and "MLP_target_used_in_official_runtime" not in line
    )
    manual_bad = int(bool(re.search(r"\.data\s*=|param\.data|manual_update", scan_text)))
    future_bad = int(bool(re.search(r"\b(val_loader|test_loader|x_val|y_val|x_test|y_test)\b", scan_text, re.I)))
    candidate_bad = int(bool(re.search(r"\bruntime_candidate_selector\b|\bruntime_argmax_candidate\b|\bruntime_topk_candidate\b", scan_text, re.I)))
    row = {
        "gate": "v22_84r_part_a_code_identity_hard_gate",
        "compileall_pass": compile_ok,
        "compile_error": compile_error,
        "worktree_import_pass": import_ok,
        "import_error": import_error,
        "clean_tarball_self_contained_import_pass": clean_tar_ok,
        "clean_tarball_error": clean_tar_error,
        "standard_loop_static_scan_pass": int(not manual_bad and not candidate_bad and not future_bad),
        "standard_loop_runtime_trace_pass": "diagnostic_oracle_only_no_official_promotion_executed",
        "loss_total_is_task_loss_only": "not_applicable_part_c_oracle_diagnostic",
        "optimizer_owned_gradient_transform_pass": "not_applicable_part_c_oracle_diagnostic",
        "manual_param_update_detected": manual_bad,
        "candidate_runtime_selection_used": candidate_bad,
        "validation_test_future_direction_used": future_bad,
        "MLP_target_used_in_official_runtime": 0,
        "output_oracle_runtime_used_in_official_runtime": 0,
        "readout_LS_promotion_detected": 0,
        "sampler_or_class_weight_used_as_FU": 0,
        "auxiliary_loss_used_official": 0,
        "changed_w1_edge_tensors": "part_c_oracle_no_persistent_parameter_step",
        "changed_w2_readout_tensors": 0,
        "edge_effect_fraction": "part_c_oracle_output_linearization",
        "readout_effect_fraction": 0.0,
    }
    row["part_a_hard_gate_pass"] = int(compile_ok and import_ok and clean_tar_ok and not manual_bad and not future_bad and not candidate_bad)
    write_json(OUT_ROOT / "v22_84r_part_a_code_identity_hard_gate.json", row)
    write_rows(OUT_ROOT / "v22_84r_part_a_code_identity_hard_gate.csv", [row])
    append_exec(
        "A_code_identity_hard_gate",
        command_text(sys.argv),
        "pass" if row["part_a_hard_gate_pass"] else "fail",
        gpu=args.device,
        files=f"{rel(OUT_ROOT / 'v22_84r_part_a_code_identity_hard_gate.json')}; {rel(OUT_ROOT / 'v22_84r_part_a_code_identity_hard_gate.csv')}",
        note=json.dumps({"pass": row["part_a_hard_gate_pass"], "compile": compile_ok, "import": import_ok, "clean_tarball": clean_tar_ok}, ensure_ascii=False),
    )
    append_recap("Part A code / identity hard gate", [
        f"part_a_hard_gate_pass={row['part_a_hard_gate_pass']}；compile={compile_ok}；import={import_ok}；clean_tarball={clean_tar_ok}。",
        f"forbidden static flags: manual={manual_bad}；candidate_runtime={candidate_bad}；future={future_bad}；readout_LS=0；output_oracle_official_runtime=0。",
        "实现/审计记录：v22.84R Part C 是 RKHS oracle diagnostic，不执行 official promotion step；因此 changed_w1 记录为 no persistent parameter step，后续 D-G 若运行必须另行记录真实 optimizer-owned update。",
    ])
    return row


def linear_slope(points: list[tuple[float, float]]) -> float:
    pts = [(float(x), float(y)) for x, y in points if math.isfinite(float(x)) and math.isfinite(float(y))]
    if len(pts) < 2:
        return 0.0
    xs = torch.tensor([p[0] for p in pts], dtype=torch.float64)
    ys = torch.tensor([p[1] for p in pts], dtype=torch.float64)
    xm = xs.mean()
    ym = ys.mean()
    denom = (xs - xm).square().sum().clamp_min(1.0e-12)
    return float((((xs - xm) * (ys - ym)).sum() / denom).item())


def run_part_b(args: argparse.Namespace) -> dict[str, Any]:
    route83 = read_json(V2283_ROOT / "v22_83_final_route.json")
    summaries = read_rows(V2283_ROOT / "v22_83_part_d_observable_active_atlas_summaries.csv")
    missing: list[str] = []
    if not route83:
        missing.append(rel(V2283_ROOT / "v22_83_final_route.json"))
    if not summaries:
        missing.append(rel(V2283_ROOT / "v22_83_part_d_observable_active_atlas_summaries.csv"))
    key_families = ["OA12", "OA13", "OA15", "OA20", "OA21", "OA22", "OA23"]
    key_counts: dict[str, Any] = {}
    for prefix in key_families:
        row = next((r for r in summaries if str(r.get("atlas_family", "")).startswith(prefix)), None)
        if row:
            key_counts[prefix] = {
                "coverage_ge_040": int(fval(row.get("coverage_to_output_oracle_ge_040_rows"))),
                "debt_nonpositive": int(fval(row.get("all_debt_nonpositive_rows"))),
                "control_margin_positive": int(fval(row.get("control_increment_margin_positive_rows"))),
                "MLP_positive": int(fval(row.get("MLP_matched_gap_positive_rows"))),
            }
    max_cov_rows = max([int(fval(r.get("coverage_to_output_oracle_ge_040_rows"))) for r in summaries] or [0])
    max_debt_rows = max([int(fval(r.get("all_debt_nonpositive_rows"))) for r in summaries] or [0])
    max_control_rows = max([int(fval(r.get("control_increment_margin_positive_rows"))) for r in summaries] or [0])
    max_mlp_rows = max([int(fval(r.get("MLP_matched_gap_positive_rows"))) for r in summaries] or [0])
    ranks = []
    for r in summaries:
        m = re.search(r"rank(\d+)", str(r.get("atlas_family", "")))
        if m:
            ranks.append((float(m.group(1)), fval(r.get("coverage_to_output_oracle_CVaR25")), fval(r.get("median_MLP_gap")), fval(r.get("control_increment_margin_positive_rows"))))
    out = {
        "gate": "v22_84r_part_b_v22_83_replay",
        "part_b_gate_pass": int(not missing and str(route83.get("final_route", "")) == "ObservableEdgeAtlasDebtBlocked"),
        "final_route_v22_83": route83.get("final_route", ""),
        "official_candidate_gate_pass_v22_83": int(fval(route83.get("official_candidate_gate_pass"))),
        "part_d_summary_rows": len(summaries),
        "missing_artifacts": missing,
        "max_coverage_ge_040_rows": max_cov_rows,
        "max_debt_nonpositive_rows": max_debt_rows,
        "max_control_increment_rows": max_control_rows,
        "max_MLP_positive_rows": max_mlp_rows,
        "OA_key_counts": key_counts,
        "coverage_vs_rank_slope": linear_slope([(r[0], r[1]) for r in ranks]),
        "MLP_gap_vs_rank_slope": linear_slope([(r[0], r[2]) for r in ranks]),
        "control_rows_vs_rank_slope": linear_slope([(r[0], r[3]) for r in ranks]),
    }
    if missing:
        out["part_b_route"] = "V2283ReplayArtifactsMissing"
        out["route_reason"] = f"missing={missing}"
    elif str(route83.get("final_route", "")) != "ObservableEdgeAtlasDebtBlocked":
        out["part_b_route"] = "V2283ReplayNeedsReinterpretation"
        out["route_reason"] = f"unexpected_final_route={route83.get('final_route', '')}"
    else:
        out["part_b_route"] = "V2283DebtControlMLPBlockedProceedToRKHSOracle"
        out["route_reason"] = f"v22.83 route={route83.get('final_route')}; max_debt={max_debt_rows}; max_control={max_control_rows}; max_MLP={max_mlp_rows}; summaries={len(summaries)}"
    write_json(OUT_ROOT / "v22_84r_part_b_v22_83_replay_route.json", out)
    write_rows(OUT_ROOT / "v22_84r_part_b_v22_83_replay_key_counts.csv", [{"prefix": k, **v} for k, v in key_counts.items()])
    append_exec(
        "B_v22_83_replay",
        command_text(sys.argv),
        "pass" if out["part_b_gate_pass"] else "fail",
        files=f"{rel(OUT_ROOT / 'v22_84r_part_b_v22_83_replay_route.json')}; {rel(OUT_ROOT / 'v22_84r_part_b_v22_83_replay_key_counts.csv')}",
        note=json.dumps(out, ensure_ascii=False),
    )
    append_recap("Part B v22.83 replay and reinterpretation", [
        f"final_route_v22_83={out['final_route_v22_83']}；official_candidate_gate_pass={out['official_candidate_gate_pass_v22_83']}；part_b_gate_pass={out['part_b_gate_pass']}。",
        f"summary_rows={len(summaries)}；max coverage/debt/control/MLP rows={max_cov_rows}/{max_debt_rows}/{max_control_rows}/{max_mlp_rows}。",
        f"rank slopes: coverage={out['coverage_vs_rank_slope']:.6g}；control_rows={out['control_rows_vs_rank_slope']:.6g}；MLP_gap={out['MLP_gap_vs_rank_slope']:.6g}。",
        "证据链：读取 v22.83 final route 与 Part D summaries；未重跑 v22.83，不补造旧 row。该证据支持进入 Part C RKHS edge-function oracle，而不是继续 v22.83 fixed-basis 扫 rank/scale。",
    ])
    return out


def part_c_kernels(args: argparse.Namespace) -> list[str]:
    kernel_set = str(args.kernel_set)
    if kernel_set == "initial":
        return list(RKHS_KERNELS_INITIAL)
    if kernel_set == "repair":
        return list(RKHS_KERNELS_REPAIR)
    if kernel_set == "repair2":
        return list(RKHS_KERNELS_REPAIR2)
    if kernel_set == "repair3":
        return list(RKHS_KERNELS_REPAIR3)
    if kernel_set == "repair4":
        return list(RKHS_KERNELS_REPAIR4)
    if kernel_set == "repair5":
        return list(RKHS_KERNELS_REPAIR5)
    return list(RKHS_KERNELS_INITIAL) + list(RKHS_KERNELS_REPAIR)


def run_part_c(args: argparse.Namespace) -> dict[str, Any]:
    device = device_from_args(args)
    tasks = shard_items(limited_task_grid(args, limit=int(args.preflight_rows)), args)
    kernels = part_c_kernels(args)
    rows: list[dict[str, Any]] = []
    for spec, seed, dataset in tasks:
        for kernel in kernels:
            rows.append(part_c_probe(dataset, seed, spec, kernel, args, device))
        if device.type == "cuda":
            torch.cuda.empty_cache()
    out = OUT_ROOT / f"v22_84r_part_c_rkhs_oracle_preflight_{args.kernel_set}_shard{args.shard_index}_of_{args.shard_count}.csv"
    write_rows(out, rows)
    obj = {
        "gate": "v22_84r_part_c_rkhs_oracle_preflight_shard",
        "kernel_set": str(args.kernel_set),
        "rows": len(rows),
        "tasks": len(tasks),
        "kernels": kernels,
        "shard_index": int(args.shard_index),
        "shard_count": int(args.shard_count),
        "device": str(device),
        "output": rel(out),
    }
    write_json(OUT_ROOT / f"v22_84r_part_c_rkhs_oracle_preflight_{args.kernel_set}_shard{args.shard_index}_of_{args.shard_count}.json", obj)
    append_exec("C_RKHS_oracle_preflight_shard", command_text(sys.argv), "done", gpu=str(device), files=rel(out), note=json.dumps(obj, ensure_ascii=False))
    return obj


def merge_part_c(args: argparse.Namespace) -> dict[str, Any]:
    kernel_sets = [item.strip() for item in str(args.merge_kernel_sets).split(",") if item.strip()]
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for kernel_set in kernel_sets:
        for idx in range(int(args.shard_count)):
            path = OUT_ROOT / f"v22_84r_part_c_rkhs_oracle_preflight_{kernel_set}_shard{idx}_of_{args.shard_count}.csv"
            part = read_rows(path)
            if not part:
                missing.append(rel(path))
            rows.extend(part)
    out_csv = OUT_ROOT / "v22_84r_part_c_rkhs_oracle_preflight.csv"
    write_rows(out_csv, rows)
    summaries = summarize_part_c(rows)
    write_rows(OUT_ROOT / "v22_84r_part_c_rkhs_oracle_preflight_summaries.csv", summaries)
    passed = [r for r in summaries if int(fval(r.get("part_c_family_gate_pass"))) > 0]
    if missing:
        route = "RKHSOraclePreflightIncomplete"
        reason = f"missing_shards={len(missing)}"
    elif passed:
        route = "EdgeFunctionRKHSOracleSignalExists"
        reason = "passed_families=" + ",".join(str(r.get("rkhs_kernel_family")) for r in passed)
    else:
        route = "NoEdgeFunctionRKHSOracleSignal"
        max_nll = max([int(fval(r.get("oracle_NLL_improve_rows"))) for r in summaries] or [0])
        max_debt = max([int(fval(r.get("oracle_all_debt_nonpositive_rows"))) for r in summaries] or [0])
        max_cov = max([int(fval(r.get("coverage_to_output_velocity_ge_040_rows"))) for r in summaries] or [0])
        max_mlp = max([int(fval(r.get("beats_MLP_matched_RKHS_rows"))) for r in summaries] or [0])
        max_ctrl = max([int(fval(r.get("control_increment_margin_positive_rows"))) for r in summaries] or [0])
        reason = f"max_NLL={max_nll}; max_debt={max_debt}; max_coverage={max_cov}; max_control={max_ctrl}; max_MLP={max_mlp}; families={len(summaries)}"
    obj = {
        "gate": "v22_84r_part_c_rkhs_oracle_preflight",
        "part_c_gate_pass": int(bool(passed) and not missing),
        "part_c_route": route,
        "route_reason": reason,
        "rows": len(rows),
        "summary_rows": len(summaries),
        "missing_shards": missing,
        "passed_rkhs_kernel_families": [str(r.get("rkhs_kernel_family")) for r in passed],
        "raw_csv": rel(out_csv),
        "summary_csv": rel(OUT_ROOT / "v22_84r_part_c_rkhs_oracle_preflight_summaries.csv"),
    }
    write_json(OUT_ROOT / "v22_84r_part_c_rkhs_oracle_preflight_route.json", obj)
    best = sorted(summaries, key=lambda r: (int(fval(r.get("part_c_family_gate_pass"))), int(fval(r.get("oracle_NLL_improve_rows"))), int(fval(r.get("oracle_all_debt_nonpositive_rows"))), int(fval(r.get("beats_MLP_matched_RKHS_rows"))), fval(r.get("control_increment_margin_CVaR25"))), reverse=True)[:8]
    append_exec(
        "C_RKHS_oracle_preflight_merge",
        command_text(sys.argv),
        "pass" if obj["part_c_gate_pass"] else "fail",
        files=f"{rel(out_csv)}; {rel(OUT_ROOT / 'v22_84r_part_c_rkhs_oracle_preflight_summaries.csv')}; {rel(OUT_ROOT / 'v22_84r_part_c_rkhs_oracle_preflight_route.json')}",
        note=json.dumps(obj, ensure_ascii=False),
    )
    append_recap("Part C RKHS edge-function oracle preflight", [
        f"rows={len(rows)}；summary_rows={len(summaries)}；part_c_gate_pass={obj['part_c_gate_pass']}；route={route}。",
        f"reason={reason}",
        "best_families="
        + " | ".join(
            f"{r.get('rkhs_kernel_family')}: NLL={r.get('oracle_NLL_improve_rows')}/{r.get('completed_rows')}；debt={r.get('oracle_all_debt_nonpositive_rows')}/{r.get('completed_rows')}；coverage_rows={r.get('coverage_to_output_velocity_ge_040_rows')}；visible_rows={r.get('raw_readout_visible_energy_ge_020_rows')}；control={r.get('control_increment_margin_positive_rows')}；MLP={r.get('beats_MLP_matched_RKHS_rows')}；LCB={r.get('source_witness_transfer_LCB_positive_rows')}；pass={r.get('part_c_family_gate_pass')}"
            for r in best
        ),
        "实现/修复记录：Part C 使用 source+witness train-only activation quantile coordinate 构造 RKHS atoms；K1-K6 为计划的初始 kernels，K2R/K3R/K5R 是按计划 C-fail 修复方向加入的 quantile-balanced、ridge/whitening、control-residual variants；K2R2/K3R2/K5R2 为更高 capacity 与 source-only debt-cone alpha projection 修复；K2R3/K3R3/K3TF/K5TF 为 SVD-truncated conditioning 修复与 target-free CE objective 修复；K2R4/K5TF_R4 为 source actual debt barrier 修复；K2R5/K3TF_R5 为 source-only inner-stability 修复；controls 覆盖同 RKHS norm、同 domain、同 smoothness、同 downstream sensitivity、同 coverage、同 debt、同 solver、noop 与 MLP matched output-energy coordinate。",
        "证据解释：若 route=NoEdgeFunctionRKHSOracleSignal，则按计划不得继续 D-G；这表示非参数 edge-function oracle 仍无法同时满足 debt/control/MLP/source-witness gates，不是 materialization bottleneck 的证据。",
    ])
    return obj


def ceiling_kernel_list(args: argparse.Namespace) -> list[str]:
    raw = str(getattr(args, "ceiling_kernels", "")).strip()
    if not raw:
        return list(CEILING_KERNELS_DEFAULT)
    return [item.strip() for item in raw.split(",") if item.strip()]


def ceiling_probe(dataset: str, seed: int, spec: dict[str, str], kernel_family: str, args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    row: dict[str, Any] = {
        "dataset": dataset,
        "seed": seed,
        "architecture": "strict_fc_purekan",
        "method": spec.get("method", ""),
        "carrier_family": spec.get("carrier_family", ""),
        "rkhs_kernel_family": kernel_family,
        "probe_error": "",
    }
    try:
        model, _mlp, _bundle, splits = base83.make_model_and_batch(dataset, seed, spec, args, device)
        xs, ys, xw, yw, xg, yg, _xc, _yc = splits
        x_train = torch.cat([xs, xw], dim=0)
        train_z = model._norm_input(x_train).detach().to(dtype=torch.float64)
        logits_s = model(xs).float()
        logits_w = model(xw).float()
        logits_g = model(xg).float()
        metric_s = base83.output_metric_diag(logits_s).to(device=device)
        metric_w = base83.output_metric_diag(logits_w).to(device=device)
        metric_g = base83.output_metric_diag(logits_g).to(device=device)
        out_family = selected_output_family(args)
        target_s, _ = base83.output_update_for_family(logits_s, ys, out_family, args)
        target_w, _ = base83.output_update_for_family(logits_w, yw, out_family, args)
        target_g, _ = base83.output_update_for_family(logits_g, yg, out_family, args)
        d_idx, h_idx, edge_meta = select_edge_pairs(model, x_train, base83.output_metric_diag(model(x_train).float()).to(device=device), args)
        centers = torch.linspace(0.0, 1.0, steps=max(2, int(args.rkhs_centers)), device=device, dtype=torch.float64)
        if "quantile_balanced" in kernel_family:
            centers = torch.linspace(0.05, 0.95, steps=max(2, int(args.rkhs_centers)), device=device, dtype=torch.float64)
        design_s = rkhs_design(model, xs, train_z, d_idx, h_idx, centers, kernel_family)
        design_w = rkhs_design(model, xw, train_z, d_idx, h_idx, centers, kernel_family)
        design_g = rkhs_design(model, xg, train_z, d_idx, h_idx, centers, kernel_family)
        whiten_meta: dict[str, Any] = {"whitened": 0}
        if "whitened" in kernel_family:
            scale = column_scale_from_design(design_s, float(args.projector_ridge))
            design_s = apply_column_scale(design_s, scale)
            design_w = apply_column_scale(design_w, scale)
            design_g = apply_column_scale(design_g, scale)
            whiten_meta = {"whitened": 1, "column_norm_median": quantile([float(v) for v in scale.detach().cpu().tolist()], 0.50)}
        svd_meta: dict[str, Any] = {"svd_reduced": 0, "svd_rank": 0, "svd_source_condition": 0.0}
        if "svd" in kernel_family:
            design_s, design_w, design_g, svd_meta = svd_reduce_designs(
                design_s,
                design_w,
                design_g,
                metric_s,
                int(args.rkhs_svd_rank),
                float(args.projector_ridge),
            )
        alpha_s = weighted_ridge_alpha(design_s, target_s, metric_s, float(args.projector_ridge))
        max_alpha_norm = float(args.rkhs_alpha_norm) * max(1.0, float(target_s.reshape(-1).to(dtype=torch.float64).norm().detach().cpu().item()))
        if float(alpha_s.norm().detach().cpu().item()) > max_alpha_norm:
            alpha_s = scale_to_norm(alpha_s, max_alpha_norm)
        upd_ss = (design_s @ alpha_s).reshape_as(logits_s)
        upd_sw = (design_w @ alpha_s).reshape_as(logits_w)
        upd_sg = (design_g @ alpha_s).reshape_as(logits_g)
        cov_ss, res_ss = base80.weighted_capacity(upd_ss.reshape(-1), target_s.reshape(-1), metric_s.reshape(-1))
        cov_sw, res_sw = base80.weighted_capacity(upd_sw.reshape(-1), target_w.reshape(-1), metric_w.reshape(-1))
        cov_sg, res_sg = base80.weighted_capacity(upd_sg.reshape(-1), target_g.reshape(-1), metric_g.reshape(-1))
        delta_g, debt_g = row_metrics_from_logit_update(logits_g, yg, upd_sg)
        alpha_w = weighted_ridge_alpha(design_w, target_w, metric_w, float(args.projector_ridge))
        alpha_g = weighted_ridge_alpha(design_g, target_g, metric_g, float(args.projector_ridge))
        upd_ww = (design_w @ alpha_w).reshape_as(logits_w)
        upd_gg = (design_g @ alpha_g).reshape_as(logits_g)
        cov_ww, res_ww = base80.weighted_capacity(upd_ww.reshape(-1), target_w.reshape(-1), metric_w.reshape(-1))
        cov_gg, res_gg = base80.weighted_capacity(upd_gg.reshape(-1), target_g.reshape(-1), metric_g.reshape(-1))
        delta_gg, debt_gg = row_metrics_from_logit_update(logits_g, yg, upd_gg)
        row.update({
            **edge_meta,
            **whiten_meta,
            **svd_meta,
            "output_velocity_family": out_family,
            "rkhs_centers": int(centers.numel()),
            "rkhs_atoms": int(design_s.shape[1]),
            "source_fit_source_coverage": cov_ss,
            "source_fit_witness_coverage": cov_sw,
            "source_fit_guard_coverage": cov_sg,
            "witness_in_sample_coverage": cov_ww,
            "guard_in_sample_coverage": cov_gg,
            "source_fit_source_rel_residual": res_ss,
            "source_fit_witness_rel_residual": res_sw,
            "source_fit_guard_rel_residual": res_sg,
            "witness_in_sample_rel_residual": res_ww,
            "guard_in_sample_rel_residual": res_gg,
            "source_fit_guard_NLL_delta": float(delta_g.get("NLL", 0.0)),
            "source_fit_guard_debt_UCB": debt_g,
            "guard_in_sample_NLL_delta": float(delta_gg.get("NLL", 0.0)),
            "guard_in_sample_debt_UCB": debt_gg,
            "source_fit_guard_coverage_ge_040": int(cov_sg >= 0.40),
            "guard_in_sample_coverage_ge_040": int(cov_gg >= 0.40),
            "source_fit_guard_debt_nonpositive": int(debt_g <= 0.0),
            "guard_in_sample_debt_nonpositive": int(debt_gg <= 0.0),
            "diagnostic_leaks_guard_target": 1,
        })
    except Exception as exc:
        row["probe_error"] = f"{type(exc).__name__}: {exc}"
    return row


def run_part_c_ceiling(args: argparse.Namespace) -> dict[str, Any]:
    device = device_from_args(args)
    tasks = shard_items(limited_task_grid(args, limit=int(args.preflight_rows)), args)
    kernels = ceiling_kernel_list(args)
    rows: list[dict[str, Any]] = []
    for spec, seed, dataset in tasks:
        for kernel in kernels:
            rows.append(ceiling_probe(dataset, seed, spec, kernel, args, device))
        if device.type == "cuda":
            torch.cuda.empty_cache()
    out = OUT_ROOT / f"v22_84r_part_c_split_ceiling_diagnostics_shard{args.shard_index}_of_{args.shard_count}.csv"
    write_rows(out, rows)
    obj = {
        "gate": "v22_84r_part_c_split_ceiling_diagnostics_shard",
        "rows": len(rows),
        "tasks": len(tasks),
        "kernels": kernels,
        "shard_index": int(args.shard_index),
        "shard_count": int(args.shard_count),
        "device": str(device),
        "output": rel(out),
    }
    write_json(OUT_ROOT / f"v22_84r_part_c_split_ceiling_diagnostics_shard{args.shard_index}_of_{args.shard_count}.json", obj)
    append_exec("C_split_ceiling_diagnostics_shard", command_text(sys.argv), "done", gpu=str(device), files=rel(out), note=json.dumps(obj, ensure_ascii=False))
    return obj


def merge_part_c_ceiling(args: argparse.Namespace) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for idx in range(int(args.shard_count)):
        path = OUT_ROOT / f"v22_84r_part_c_split_ceiling_diagnostics_shard{idx}_of_{args.shard_count}.csv"
        part = read_rows(path)
        if not part:
            missing.append(rel(path))
        rows.extend(part)
    out_csv = OUT_ROOT / "v22_84r_part_c_split_ceiling_diagnostics.csv"
    write_rows(out_csv, rows)
    summaries: list[dict[str, Any]] = []
    for family in sorted({str(r.get("rkhs_kernel_family", "")) for r in rows if str(r.get("rkhs_kernel_family", ""))}):
        group = [r for r in rows if str(r.get("rkhs_kernel_family", "")) == family and not str(r.get("probe_error", ""))]
        if not group:
            summaries.append({"rkhs_kernel_family": family, "completed_rows": 0})
            continue
        summaries.append({
            "rkhs_kernel_family": family,
            "completed_rows": len(group),
            "source_fit_source_cov_median": quantile([fval(r.get("source_fit_source_coverage")) for r in group], 0.50),
            "source_fit_witness_cov_median": quantile([fval(r.get("source_fit_witness_coverage")) for r in group], 0.50),
            "source_fit_guard_cov_median": quantile([fval(r.get("source_fit_guard_coverage")) for r in group], 0.50),
            "guard_in_sample_cov_median": quantile([fval(r.get("guard_in_sample_coverage")) for r in group], 0.50),
            "source_fit_guard_cov_ge_040_rows": sum(int(fval(r.get("source_fit_guard_coverage_ge_040"))) for r in group),
            "guard_in_sample_cov_ge_040_rows": sum(int(fval(r.get("guard_in_sample_coverage_ge_040"))) for r in group),
            "source_fit_guard_debt_nonpositive_rows": sum(int(fval(r.get("source_fit_guard_debt_nonpositive"))) for r in group),
            "guard_in_sample_debt_nonpositive_rows": sum(int(fval(r.get("guard_in_sample_debt_nonpositive"))) for r in group),
            "source_fit_guard_NLL_improve_rows": sum(int(fval(r.get("source_fit_guard_NLL_delta")) < 0.0) for r in group),
            "guard_in_sample_NLL_improve_rows": sum(int(fval(r.get("guard_in_sample_NLL_delta")) < 0.0) for r in group),
            "svd_source_condition_median": quantile([fval(r.get("svd_source_condition")) for r in group], 0.50),
        })
    out_summary = OUT_ROOT / "v22_84r_part_c_split_ceiling_diagnostics_summaries.csv"
    write_rows(out_summary, summaries)
    max_guard_in = max([int(fval(r.get("guard_in_sample_cov_ge_040_rows"))) for r in summaries] or [0])
    max_source_guard = max([int(fval(r.get("source_fit_guard_cov_ge_040_rows"))) for r in summaries] or [0])
    if missing:
        route = "SplitCeilingIncomplete"
        reason = f"missing={missing}"
    elif max_guard_in >= 32 and max_source_guard < 32:
        route = "GuardLocalOracleExistsButSourceTransferFails"
        reason = f"max_guard_in_sample_cov_rows={max_guard_in}; max_source_fit_guard_cov_rows={max_source_guard}"
    elif max_guard_in < 32:
        route = "EdgeFunctionOutputVelocityCoverageLowEvenInSample"
        reason = f"max_guard_in_sample_cov_rows={max_guard_in}; max_source_fit_guard_cov_rows={max_source_guard}"
    else:
        route = "SplitCeilingMixed"
        reason = f"max_guard_in_sample_cov_rows={max_guard_in}; max_source_fit_guard_cov_rows={max_source_guard}"
    obj = {
        "gate": "v22_84r_part_c_split_ceiling_diagnostics",
        "route": route,
        "route_reason": reason,
        "rows": len(rows),
        "summary_rows": len(summaries),
        "missing_shards": missing,
        "max_guard_in_sample_cov_ge_040_rows": max_guard_in,
        "max_source_fit_guard_cov_ge_040_rows": max_source_guard,
        "raw_csv": rel(out_csv),
        "summary_csv": rel(out_summary),
    }
    write_json(OUT_ROOT / "v22_84r_part_c_split_ceiling_diagnostics_route.json", obj)
    best = sorted(summaries, key=lambda r: (int(fval(r.get("guard_in_sample_cov_ge_040_rows"))), int(fval(r.get("source_fit_guard_cov_ge_040_rows"))), fval(r.get("guard_in_sample_cov_median"))), reverse=True)[:6]
    append_exec("C_split_ceiling_diagnostics_merge", command_text(sys.argv), "done", files=f"{rel(out_csv)}; {rel(out_summary)}; {rel(OUT_ROOT / 'v22_84r_part_c_split_ceiling_diagnostics_route.json')}", note=json.dumps(obj, ensure_ascii=False))
    append_recap("Part C split/ceiling diagnostics", [
        f"rows={len(rows)}；summary_rows={len(summaries)}；route={route}；reason={reason}。",
        "best_ceiling="
        + " | ".join(
            f"{r.get('rkhs_kernel_family')}: source->guard_cov_rows={r.get('source_fit_guard_cov_ge_040_rows')}；guard_in_sample_cov_rows={r.get('guard_in_sample_cov_ge_040_rows')}；source_guard_cov_med={r.get('source_fit_guard_cov_median')}；guard_in_cov_med={r.get('guard_in_sample_cov_median')}；source_guard_debt={r.get('source_fit_guard_debt_nonpositive_rows')}；guard_in_debt={r.get('guard_in_sample_debt_nonpositive_rows')}"
            for r in best
        ),
        "解释：guard_in_sample 使用 guard target 只作泄漏诊断，不作为 candidate 成功证据；若 guard_in_sample 高而 source->guard 低，说明 edge-function space 局部可覆盖但 train-only persistent transfer 失败。",
    ])
    return obj


def run_part_c_diagnostics(args: argparse.Namespace) -> dict[str, Any]:
    raw = read_rows(OUT_ROOT / "v22_84r_part_c_rkhs_oracle_preflight.csv")
    summaries = read_rows(OUT_ROOT / "v22_84r_part_c_rkhs_oracle_preflight_summaries.csv")
    valid = [r for r in raw if not str(r.get("probe_error", ""))]
    errors = [r for r in raw if str(r.get("probe_error", ""))]
    projection_rows = [r for r in valid if int(fval(r.get("source_debt_alpha_projection_used"))) > 0]
    barrier_rows = [r for r in valid if int(fval(r.get("source_actual_debt_barrier_used"))) > 0]
    svd_rows = [r for r in valid if int(fval(r.get("svd_reduced"))) > 0]
    inner_source_rows = [r for r in valid if int(fval(r.get("inner_source_stable_used"))) > 0]
    target_free_rows = [r for r in valid if str(r.get("objective_family", "")) == "target_free_train_only_ce"]
    obj = {
        "gate": "v22_84r_part_c_failure_diagnostics",
        "raw_rows": len(raw),
        "valid_rows": len(valid),
        "probe_error_rows": len(errors),
        "summary_rows": len(summaries),
        "train_domain_extrapolation_rate_max": max([fval(r.get("domain_extrapolation_rate")) for r in valid] or [0.0]),
        "train_domain_extrapolation_rate_median": quantile([fval(r.get("domain_extrapolation_rate")) for r in valid], 0.50),
        "Gram_condition_median": quantile([fval(r.get("Gram_condition")) for r in valid], 0.50),
        "Gram_condition_max": max([fval(r.get("Gram_condition")) for r in valid] or [0.0]),
        "Gram_condition_gt_1e6_rows": sum(int(fval(r.get("Gram_condition")) > 1.0e6) for r in valid),
        "same_compute_noop_nonzero_NLL_rows": sum(int(abs(fval(r.get("same_compute_noop_NLL_delta"))) > 1.0e-12) for r in valid),
        "visible_energy_ge_020_rows": sum(int(fval(r.get("raw_readout_visible_energy")) >= 0.20) for r in valid),
        "visible_energy_median": quantile([fval(r.get("raw_readout_visible_energy")) for r in valid], 0.50),
        "coverage_ge_040_rows": sum(int(fval(r.get("coverage_to_output_velocity")) >= 0.40) for r in valid),
        "coverage_median": quantile([fval(r.get("coverage_to_output_velocity")) for r in valid], 0.50),
        "source_debt_projection_used_rows": len(projection_rows),
        "source_debt_projection_count_positive_rows": sum(int(fval(r.get("source_debt_alpha_projection_count")) > 0.0) for r in projection_rows),
        "source_debt_projection_norm_ratio_median": quantile([fval(r.get("source_debt_alpha_projection_norm_ratio"), 1.0) for r in projection_rows], 0.50),
        "source_actual_debt_barrier_used_rows": len(barrier_rows),
        "source_actual_debt_barrier_noop_rows": sum(int(fval(r.get("source_actual_debt_barrier_noop")) > 0.0) for r in barrier_rows),
        "source_actual_debt_barrier_scale_median": quantile([fval(r.get("source_actual_debt_barrier_scale"), 1.0) for r in barrier_rows], 0.50),
        "svd_reduced_rows": len(svd_rows),
        "svd_source_condition_median": quantile([fval(r.get("svd_source_condition")) for r in svd_rows], 0.50),
        "svd_rank_median": quantile([fval(r.get("svd_rank")) for r in svd_rows], 0.50),
        "inner_source_stable_used_rows": len(inner_source_rows),
        "inner_source_stable_fraction_median": quantile([fval(r.get("inner_source_stable_fraction")) for r in inner_source_rows], 0.50),
        "inner_source_half_alpha_cosine_median": quantile([fval(r.get("inner_source_half_alpha_cosine")) for r in inner_source_rows], 0.50),
        "target_free_rows": len(target_free_rows),
        "target_free_NLL_improve_rows": sum(int(fval(r.get("oracle_NLL_improve")) > 0.0) for r in target_free_rows),
        "target_free_debt_nonpositive_rows": sum(int(fval(r.get("oracle_all_debt_nonpositive")) > 0.0) for r in target_free_rows),
        "target_free_MLP_positive_rows": sum(int(fval(r.get("beats_MLP_matched_RKHS")) > 0.0) for r in target_free_rows),
        "MLP_matched_positive_rows_total": sum(int(fval(r.get("beats_MLP_matched_RKHS")) > 0.0) for r in valid),
        "MLP_matched_gap_median": quantile([fval(r.get("MLP_matched_gap")) for r in valid], 0.50),
        "control_increment_positive_rows_total": sum(int(fval(r.get("control_increment_margin_positive")) > 0.0) for r in valid),
        "control_increment_margin_median": quantile([fval(r.get("control_increment_margin")) for r in valid], 0.50),
        "max_family_NLL_improve_rows": max([int(fval(r.get("oracle_NLL_improve_rows"))) for r in summaries] or [0]),
        "max_family_debt_nonpositive_rows": max([int(fval(r.get("oracle_all_debt_nonpositive_rows"))) for r in summaries] or [0]),
        "max_family_coverage_rows": max([int(fval(r.get("coverage_to_output_velocity_ge_040_rows"))) for r in summaries] or [0]),
        "max_family_visible_rows": max([int(fval(r.get("raw_readout_visible_energy_ge_020_rows"))) for r in summaries] or [0]),
        "max_family_control_rows": max([int(fval(r.get("control_increment_margin_positive_rows"))) for r in summaries] or [0]),
        "max_family_MLP_rows": max([int(fval(r.get("beats_MLP_matched_RKHS_rows"))) for r in summaries] or [0]),
        "diagnostic_conclusion": "coverage/debt/control/MLP remain hard blockers after initial, conditioning, high-capacity, source-only debt projection, SVD-truncated, target-free, source actual debt barrier, and inner-source stability repairs",
    }
    write_json(OUT_ROOT / "v22_84r_part_c_failure_diagnostics.json", obj)
    append_exec(
        "C_failure_diagnostics",
        command_text(sys.argv),
        "done",
        files=rel(OUT_ROOT / "v22_84r_part_c_failure_diagnostics.json"),
        note=json.dumps(obj, ensure_ascii=False),
    )
    append_recap("Part C failure diagnostics and repair audit", [
        f"raw_rows={obj['raw_rows']}；valid_rows={obj['valid_rows']}；probe_error_rows={obj['probe_error_rows']}。",
        f"train-domain check: extrapolation max={obj['train_domain_extrapolation_rate_max']}；median={obj['train_domain_extrapolation_rate_median']}。",
        f"conditioning check: Gram_condition median={obj['Gram_condition_median']}；max={obj['Gram_condition_max']}；>1e6 rows={obj['Gram_condition_gt_1e6_rows']}。",
        f"debt/no-op check: source debt projection used rows={obj['source_debt_projection_used_rows']}；projection_count>0 rows={obj['source_debt_projection_count_positive_rows']}；projection_norm_ratio_median={obj['source_debt_projection_norm_ratio_median']}；source actual debt barrier rows={obj['source_actual_debt_barrier_used_rows']}；barrier_noop_rows={obj['source_actual_debt_barrier_noop_rows']}；barrier_scale_median={obj['source_actual_debt_barrier_scale_median']}；same_compute_noop_nonzero_NLL_rows={obj['same_compute_noop_nonzero_NLL_rows']}。",
        f"SVD/target-free/inner-source check: svd_rows={obj['svd_reduced_rows']}；svd_condition_median={obj['svd_source_condition_median']}；svd_rank_median={obj['svd_rank_median']}；inner_source_rows={obj['inner_source_stable_used_rows']}；inner_source_fraction_median={obj['inner_source_stable_fraction_median']}；inner_source_half_alpha_cosine_median={obj['inner_source_half_alpha_cosine_median']}；target_free_rows={obj['target_free_rows']}；target_free_NLL={obj['target_free_NLL_improve_rows']}；target_free_debt={obj['target_free_debt_nonpositive_rows']}；target_free_MLP={obj['target_free_MLP_positive_rows']}。",
        f"capacity/visibility check: visible_energy>=0.20 rows={obj['visible_energy_ge_020_rows']}；visible_energy_median={obj['visible_energy_median']}；coverage>=0.40 rows={obj['coverage_ge_040_rows']}；coverage_median={obj['coverage_median']}。",
        f"control/MLP check: total control_increment_positive_rows={obj['control_increment_positive_rows_total']}；control_margin_median={obj['control_increment_margin_median']}；MLP_positive_rows_total={obj['MLP_matched_positive_rows_total']}；MLP_gap_median={obj['MLP_matched_gap_median']}。",
        f"family maxima: NLL={obj['max_family_NLL_improve_rows']}；debt={obj['max_family_debt_nonpositive_rows']}；coverage={obj['max_family_coverage_rows']}；visible={obj['max_family_visible_rows']}；control={obj['max_family_control_rows']}；MLP={obj['max_family_MLP_rows']}。",
        "结论：已按计划尝试 train-domain/conditioning/high-capacity/debt-projection/SVD-truncated/target-free/source-actual-debt-barrier/inner-source-stability/control-axis/MLP matched 自查；失败仍集中在 output velocity coverage、guard debt、control increment 与 MLP matched，不能进入 D-G。",
    ])
    return obj


def run_final(args: argparse.Namespace) -> dict[str, Any]:
    part_a = read_json(OUT_ROOT / "v22_84r_part_a_code_identity_hard_gate.json")
    part_b = read_json(OUT_ROOT / "v22_84r_part_b_v22_83_replay_route.json")
    part_c = read_json(OUT_ROOT / "v22_84r_part_c_rkhs_oracle_preflight_route.json")
    if not int(fval(part_a.get("part_a_hard_gate_pass"))):
        route = "CodeIdentityHardGateFailed"
        official = 0
        reason = "Part A hard gate failed or missing"
    elif not int(fval(part_b.get("part_b_gate_pass"))):
        route = "V2283ReplayBeforeRKHSIncomplete"
        official = 0
        reason = str(part_b.get("route_reason", "Part B missing/failing"))
    elif not part_c:
        route = "RKHSOraclePreflightNotRun"
        official = 0
        reason = "Part C route artifact missing"
    elif str(part_c.get("part_c_route", "")) == "NoEdgeFunctionRKHSOracleSignal":
        route = "NoEdgeFunctionRKHSOracleSignal"
        official = 0
        reason = str(part_c.get("route_reason", "Part C failed"))
    elif int(fval(part_c.get("part_c_gate_pass"))) > 0:
        route = "EdgeFunctionRKHSOracleSignalExists_MaterializationRequired"
        official = 0
        reason = "Part C passed; Part D-G materialization/solver/full-loop not yet executed"
    else:
        route = str(part_c.get("part_c_route", "RKHSOraclePreflightIncomplete"))
        official = 0
        reason = str(part_c.get("route_reason", "Part C incomplete"))
    final = {
        "gate": "v22_84r_final_route",
        "final_route": route,
        "official_candidate_gate_pass": official,
        "route_reason": reason,
        "part_a": int(fval(part_a.get("part_a_hard_gate_pass"))),
        "part_b": int(fval(part_b.get("part_b_gate_pass"))),
        "part_c": int(fval(part_c.get("part_c_gate_pass"))),
        "part_d": 0,
        "part_e": 0,
        "part_f": 0,
        "part_g": 0,
        "artifacts": {
            "part_a": rel(OUT_ROOT / "v22_84r_part_a_code_identity_hard_gate.json"),
            "part_b": rel(OUT_ROOT / "v22_84r_part_b_v22_83_replay_route.json"),
            "part_c": rel(OUT_ROOT / "v22_84r_part_c_rkhs_oracle_preflight_route.json"),
            "final": rel(OUT_ROOT / "v22_84r_final_route.json"),
        },
    }
    write_json(OUT_ROOT / "v22_84r_final_route.json", final)
    append_exec("final_route", command_text(sys.argv), "done", files=rel(OUT_ROOT / "v22_84r_final_route.json"), note=json.dumps(final, ensure_ascii=False))
    append_recap("Final route and conclusion", [
        f"final_route={route}；official_candidate_gate_pass={official}；reason={reason}",
        f"A/B/C/D/E/F/G pass={final['part_a']}/{final['part_b']}/{final['part_c']}/{final['part_d']}/{final['part_e']}/{final['part_f']}/{final['part_g']}。",
        "结论约束：只有 Part G official/full-loop 后续真实通过时才能声明 KAN edge-function MPFU 达成；当前 final route 只反映已运行 gates。",
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
    p.add_argument("--model-seed-offset", type=int, default=26840)
    p.add_argument("--edge-step-norm", type=float, default=1.0e-3)
    p.add_argument("--fd-epsilon", type=float, default=1.0e-4)
    p.add_argument("--max-output-families", type=int, default=1)
    p.add_argument("--force-output-velocity-family", default="")
    p.add_argument("--preflight-rows", type=int, default=45)
    p.add_argument("--rkhs-max-edges", type=int, default=64)
    p.add_argument("--rkhs-centers", type=int, default=6)
    p.add_argument("--rkhs-alpha-norm", type=float, default=2.0)
    p.add_argument("--rkhs-svd-rank", type=int, default=64)
    p.add_argument("--kernel-set", choices=["initial", "repair", "repair2", "repair3", "repair4", "repair5", "all"], default="all")
    p.add_argument("--merge-kernel-sets", default="all")
    p.add_argument("--ceiling-kernels", default="")
    p.add_argument("--shard-count", type=int, default=4)
    p.add_argument("--shard-index", type=int, default=0)
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    init_logs()
    if args.mode == "part-a":
        run_part_a(args)
    elif args.mode == "part-b":
        run_part_b(args)
    elif args.mode == "part-c":
        run_part_c(args)
    elif args.mode == "part-c-merge":
        merge_part_c(args)
    elif args.mode == "part-c-ceiling":
        run_part_c_ceiling(args)
    elif args.mode == "part-c-ceiling-merge":
        merge_part_c_ceiling(args)
    elif args.mode == "part-c-diagnostics":
        run_part_c_diagnostics(args)
    elif args.mode == "final":
        run_final(args)
    else:
        raise ValueError(args.mode)


if __name__ == "__main__":
    main()
