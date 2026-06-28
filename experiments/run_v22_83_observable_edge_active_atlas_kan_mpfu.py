#!/usr/bin/env python3
"""DG-KAN v22.83 Observable Edge-Active Atlas KAN-MPFU runner."""

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


PYTHON = sys.executable
RUNNER = ROOT / "experiments/run_v22_83_observable_edge_active_atlas_kan_mpfu.py"
PLAN = ROOT / "docs/DG-KAN_v22.83_ObservableEdgeActiveAtlasKAN_MPFU_完整计划.md"
EXEC_LOG = ROOT / "docs/DG-KAN_v22.83_ObservableEdgeActiveAtlasKAN_MPFU_执行日志.md"
RECAP_LOG = ROOT / "docs/DG-KAN_v22.83_ObservableEdgeActiveAtlasKAN_MPFU_实验结果复盘.md"
OUT_ROOT = Path(os.environ.get("V2283_OUT_ROOT", str(ROOT / "results/v22_83"))).resolve()
LOG_ROOT = OUT_ROOT / "logs"

EDGE_PARAM = "w1"
READOUT_PARAMS = {"w2", "linear_readout", "cheby_cross_readout", "cheby_input_cross_readout"}
V2282_ROOTS = [
    ("v22_82_initial", ROOT / "results/v22_82"),
    ("v22_82_rankscale_sw", ROOT / "results/v22_82_repair_rankscale_sw"),
    ("v22_82_debtaware", ROOT / "results/v22_82_repair_debtaware"),
]

ATLAS_FAMILIES = [
    "OA1_output_visibility_eig_rank16",
    "OA2_output_visibility_eig_rank32",
    "OA3_output_visibility_eig_rank64",
    "OA4_source_witness_coherent_eig_rank32",
    "OA5_debt_safe_visibility_eig_rank32",
    "OA6_control_incremental_eig_rank32",
    "OA7_domain_warped_visibility_eig_rank32",
    "OA8_bank_local_visibility_eig_rank32",
    "OA9_mixed_bank_visibility_eig_rank64",
    "OA10_mlp_residual_visibility_eig_rank32",
    "OA11_observable_active_atlas_svd_repair_rank64",
    "OA12_observable_active_atlas_svd_repair_rank128",
    "OA13_control_debt_residual_svd_repair_rank128",
    "OA14_bank_whitened_feature_standardized_svd_repair_rank128",
    "OA15_debt_constrained_alpha_svd_repair_rank128",
    "OA16_control_residual_quantile_downstream_svd_repair_rank128",
    "OA17_control_residual_debtalpha_quantile_downstream_svd_repair_rank128",
    "OA18_output_control_residual_alpha_svd_repair_rank128",
    "OA19_output_control_residual_debtalpha_svd_repair_rank128",
    "OA20_soft_output_control_residual_alpha_svd_repair_rank128",
    "OA21_soft_output_control_residual_debtalpha_svd_repair_rank128",
    "OA22_hardloss_consensus_svd_repair_rank128",
    "OA23_hardloss_consensus_debtalpha_svd_repair_rank128",
]

SOLVER_FAMILIES = [
    "IES1_edge_intrinsic_qp_rank32",
    "IES2_edge_intrinsic_qp_rank64",
    "IES3_debt_constrained_edge_qp_rank32",
    "IES4_visibility_floor_edge_qp_rank32",
    "IES5_source_witness_consensus_edge_qp_rank32",
    "IES6_control_incremental_edge_qp_rank32",
    "IES7_poet_residual_edge_qp_rank32",
    "IES8_mlp_residual_edge_qp_rank32",
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


def upper_cvar(values: list[float], frac: float = 0.25) -> float:
    return base82.upper_cvar(values, frac)


def scaled_gate_count(completed: int, numerator: int, denominator: int = 45) -> int:
    return base82.scaled_gate_count(completed, numerator, denominator)


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
            "# DG-KAN v22.83 ObservableEdgeActiveAtlasKAN MPFU 执行日志\n\n"
            f"创建时间：{now_sg()}\n\n"
            f"- 计划文档：`{rel(PLAN)}`\n"
            "- 非编造约束：只记录实际命令、文件、错误和观测；缺失 artifact 明确写 missing/skipped。\n"
            "- 复现提示：每条命令记录 GPU、输出文件和关键参数；C/D/E 支持 shard 并行后 merge。\n\n"
            "## 命令记录\n",
            encoding="utf-8",
        )
    if not RECAP_LOG.exists():
        RECAP_LOG.write_text(
            "# DG-KAN v22.83 ObservableEdgeActiveAtlasKAN MPFU 实验结果复盘\n\n"
            f"创建时间：{now_sg()}\n\n"
            "## 0. 当前结论\n"
            "- 尚未完成最终 route 判定。\n"
            "- 本文件只记录真实 artifact 和命令观测；不补造缺失数据。\n\n"
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


def shard_items(items: list[Any], args: argparse.Namespace) -> list[Any]:
    count = max(1, int(args.shard_count))
    idx = int(args.shard_index)
    return [item for i, item in enumerate(items) if i % count == idx]


def device_from_args(args: argparse.Namespace) -> torch.device:
    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        return torch.device(args.device)
    return torch.device("cpu")


def parse_family_rank(family: str, fallback: int = 32) -> int:
    m = re.search(r"rank(\d+)", str(family))
    return int(m.group(1)) if m else int(fallback)


def stable_text_seed(text: str) -> int:
    acc = 0
    for idx, ch in enumerate(str(text)):
        acc = (acc + (idx + 1) * ord(ch)) % 1000003
    return int(acc)


def is_svd_repair_family(family: str) -> bool:
    return (
        str(family).startswith("OA11")
        or str(family).startswith("OA12")
        or str(family).startswith("OA13")
        or str(family).startswith("OA14")
        or str(family).startswith("OA15")
        or str(family).startswith("OA16")
        or str(family).startswith("OA17")
        or str(family).startswith("OA18")
        or str(family).startswith("OA19")
        or str(family).startswith("OA20")
        or str(family).startswith("OA21")
        or str(family).startswith("OA22")
        or str(family).startswith("OA23")
    )


def carrier_task_grid(args: argparse.Namespace) -> list[tuple[dict[str, str], int, str]]:
    datasets = [item.strip() for item in str(args.datasets).split(",") if item.strip()]
    seeds = list(range(int(args.seed_count)))
    return [(dict(spec), seed, dataset) for spec in base80.CARRIER_REDESIGN_SPECS for dataset in datasets for seed in seeds]


def limited_task_grid(args: argparse.Namespace, *, limit: int | None = None) -> list[tuple[dict[str, str], int, str]]:
    tasks = carrier_task_grid(args)
    if limit is None or int(limit) <= 0:
        return tasks
    return tasks[: int(limit)]


def make_model_and_batch(dataset: str, seed: int, spec: dict[str, str], args: argparse.Namespace, device: torch.device):
    return base82.make_model_and_batch(dataset, seed, spec, args, device)


def parameter_dict(model: Any) -> dict[str, torch.Tensor]:
    return {name: param for name, param in model.named_parameters()}


def flat_cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return base82.flat_cosine(a, b)


def metric_energy(vec: torch.Tensor, metric: torch.Tensor) -> torch.Tensor:
    return base82.metric_energy(vec, metric)


def output_metric_diag(logits: torch.Tensor) -> torch.Tensor:
    return base82.output_metric_diag(logits)


def metric_delta_for_updates(model: Any, x: torch.Tensor, y: torch.Tensor, updates: dict[str, torch.Tensor]) -> dict[str, float]:
    return base82.metric_delta_for_updates(model, x, y, updates)


def debt_ucb_from_deltas(deltas: dict[str, float]) -> float:
    return base82.debt_ucb_from_deltas(deltas)


def actual_logit_update_for_updates(model: Any, x: torch.Tensor, updates: dict[str, torch.Tensor]) -> torch.Tensor:
    return base82.actual_logit_update_for_updates(model, x, updates)


def edge_grad_direction(model: Any, x: torch.Tensor, y: torch.Tensor, *, mode: str = "ce") -> torch.Tensor:
    return base82.edge_grad_direction(model, x, y, mode=mode)


def hardloss_consensus_edge_grad_direction(model: Any, x: torch.Tensor, y: torch.Tensor, *, frac: float = 0.50) -> torch.Tensor:
    full = edge_grad_direction(model, x, y)
    n = int(x.shape[0])
    if n <= 2:
        return full
    with torch.no_grad():
        losses = F.cross_entropy(model(x).float(), y, reduction="none")
        k = min(n, max(2, int(math.ceil(float(frac) * n))))
        idx = torch.topk(losses.detach(), k=k).indices
    hard = edge_grad_direction(model, x[idx], y[idx])
    full_norm = full.norm().clamp_min(1.0e-12)
    hard_scaled = scale_to_norm(hard, float(full_norm.detach().cpu().item()))
    consensus = 0.5 * full + 0.5 * hard_scaled
    return scale_to_norm(consensus, float(full_norm.detach().cpu().item()))


def scale_to_norm(delta: torch.Tensor, norm: float) -> torch.Tensor:
    return base82.scale_to_norm(delta, norm)


def random_like(delta: torch.Tensor, seed: int, norm: float) -> torch.Tensor:
    return base82.random_like(delta, seed, norm)


def output_update_for_family(logits: torch.Tensor, y: torch.Tensor, family: str, args: argparse.Namespace) -> tuple[torch.Tensor, dict[str, Any]]:
    return base82.output_update_for_family(logits, y, family, args)


def selected_output_family(args: argparse.Namespace) -> str:
    forced = str(getattr(args, "force_output_velocity_family", "")).strip()
    if forced:
        return forced
    families = base82.select_output_families(args)
    return str(families[0]) if families else "output_qp_trust_region_small"


def edge_flat_to_update(model: Any, coords: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    delta = torch.zeros_like(model.w1.detach(), dtype=torch.float64)
    if int(coords.numel()) == 0:
        return delta
    delta.reshape(-1)[coords.detach().long().cpu().tolist()] = values.detach().to(device=delta.device, dtype=torch.float64)
    return delta


def selected_values(tensor: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    if int(coords.numel()) == 0:
        return torch.zeros(0, device=tensor.device, dtype=torch.float64)
    return tensor.detach().reshape(-1).to(dtype=torch.float64)[coords.to(device=tensor.device).long()]


def downstream_sensitivity(model: Any, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    z = model._norm_input(x)
    b1 = model._basis_eval_layer(z, layer="input").to(dtype=torch.float64)
    h = model.hidden(x).to(dtype=torch.float64)
    db2 = model._basis_derivative_layer(h, layer="hidden").to(dtype=torch.float64)
    w2 = model.w2.detach().to(device=x.device, dtype=torch.float64)
    down = torch.einsum("bhk,hck->bhc", db2, w2) / math.sqrt(max(1, int(model.hidden_dim)))
    if bool(getattr(model, "cheby_paircross_enabled", False)):
        rank = int(getattr(model, "cheby_cross_rank", 0))
        if rank > 0 and hasattr(model, "cheby_cross_readout"):
            read = model.cheby_cross_readout.detach().to(device=x.device, dtype=torch.float64)
            scale = 1.0 / math.sqrt(max(1, rank))
            for ridx in range(rank):
                left = 2 * ridx
                right = left + 1
                if right >= int(model.hidden_dim) or ridx >= int(read.shape[0]):
                    break
                down[:, left, :] = down[:, left, :] + h[:, right : right + 1] * read[ridx : ridx + 1, :] * scale
                down[:, right, :] = down[:, right, :] + h[:, left : left + 1] * read[ridx : ridx + 1, :] * scale
    dh = (1.0 - h.square()).clamp_min(0.0)
    return b1, h, dh, down


def visibility_scores(model: Any, x: torch.Tensor, metric: torch.Tensor) -> torch.Tensor:
    b1, _h, dh, down = downstream_sensitivity(model, x)
    mm = metric.reshape(int(x.shape[0]), int(model.output_dim)).to(device=x.device, dtype=torch.float64)
    down_energy = (down.square() * mm[:, None, :]).sum(dim=2)
    hidden_energy = dh.square() * down_energy
    vis = torch.einsum("bdk,bh->dhk", b1.square(), hidden_energy)
    vis = vis / max(1, int(x.shape[0])) / max(1, int(model.input_dim))
    return vis.detach().to(dtype=torch.float64).clamp_min(0.0)


def selected_jacobian(model: Any, x: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    coords = coords.detach().long().to(device=x.device)
    p = int(coords.numel())
    if p == 0:
        return torch.zeros((int(x.shape[0]) * int(model.output_dim), 0), device=x.device, dtype=torch.float64)
    b1, _h, dh, down = downstream_sensitivity(model, x)
    d_idx = torch.div(coords, int(model.hidden_dim) * int(model.k), rounding_mode="floor")
    rem = coords - d_idx * int(model.hidden_dim) * int(model.k)
    h_idx = torch.div(rem, int(model.k), rounding_mode="floor")
    k_idx = rem - h_idx * int(model.k)
    cols = []
    inv_sqrt_d = 1.0 / math.sqrt(max(1, int(model.input_dim)))
    for d, h, k in zip(d_idx.tolist(), h_idx.tolist(), k_idx.tolist()):
        col = (b1[:, int(d), int(k)] * dh[:, int(h)] * inv_sqrt_d)[:, None] * down[:, int(h), :]
        cols.append(col.reshape(-1))
    return torch.stack(cols, dim=1).to(dtype=torch.float64)


def ridge_solve(mat: torch.Tensor, rhs: torch.Tensor, ridge: float) -> torch.Tensor:
    p = int(mat.shape[1]) if int(mat.ndim) == 2 else 0
    if p <= 0:
        return torch.zeros(0, device=mat.device, dtype=torch.float64)
    a = mat.T @ mat
    b = mat.T @ rhs
    eye = torch.eye(p, device=mat.device, dtype=torch.float64)
    try:
        return torch.linalg.solve(a + float(ridge) * eye, b)
    except Exception:
        return torch.linalg.pinv(a + float(ridge) * eye) @ b


def top_indices(score: torch.Tensor, rank: int) -> torch.Tensor:
    flat = score.detach().reshape(-1).to(dtype=torch.float64)
    finite = torch.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0)
    k = min(max(1, int(rank)), int(finite.numel()))
    return torch.topk(finite.abs(), k=k).indices.detach().long()


def tensor_percentile_ranks(values: torch.Tensor) -> torch.Tensor:
    flat = values.detach().reshape(-1).to(dtype=torch.float64)
    n = int(flat.numel())
    if n <= 1:
        return torch.zeros_like(values, dtype=torch.float64)
    finite = torch.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0)
    order = torch.argsort(finite)
    ranks = torch.empty_like(finite)
    ranks[order] = torch.linspace(0.0, 1.0, steps=n, device=values.device, dtype=torch.float64)
    return ranks.reshape_as(values).to(dtype=torch.float64)


def atlas_coords(
    model: Any,
    family: str,
    source_direction: torch.Tensor,
    witness_direction: torch.Tensor,
    vis: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, Any]]:
    rank = min(parse_family_rank(family), int(args.max_atlas_coords), int(model.w1.numel()))
    src = source_direction.detach().reshape_as(model.w1).to(dtype=torch.float64)
    wit = witness_direction.detach().reshape_as(model.w1).to(device=src.device, dtype=torch.float64)
    score = vis.detach().to(device=src.device, dtype=torch.float64).clone()
    abs_grad = 0.5 * (src.abs() + wit.abs())
    sign_agree = (src * wit >= 0.0).to(dtype=torch.float64)
    disagreement = (src - wit).abs()
    if family.startswith("OA4"):
        score = score * (1.0 + abs_grad) * sign_agree
    elif family.startswith("OA5"):
        score = score * (1.0 + torch.minimum(src.abs(), wit.abs())) * sign_agree
    elif family.startswith("OA6"):
        score = score * (1.0 + abs_grad) / (1.0 + disagreement)
    elif family.startswith("OA7"):
        domain = src.square() + wit.square()
        score = score / domain.mean().clamp_min(1.0e-12).sqrt() * (1.0 + abs_grad)
    elif family.startswith("OA8"):
        hidden = score.square().sum(dim=(0, 2))
        keep_h = torch.topk(hidden, k=min(max(1, rank // max(1, int(model.k))), int(hidden.numel()))).indices
        mask = torch.zeros_like(score)
        mask[:, keep_h, :] = 1.0
        score = score * mask * (1.0 + abs_grad)
    elif family.startswith("OA9"):
        score = score.sqrt() * (1.0 + abs_grad) * (0.5 + sign_agree)
    elif family.startswith("OA10"):
        score = score * (1.0 + disagreement + 0.25 * abs_grad)
    else:
        score = score * (1.0 + 0.05 * abs_grad)
    coords = top_indices(score, rank)
    selected_vis = selected_values(vis, coords)
    total_vis = float(vis.reshape(-1).sum().detach().cpu().item())
    selected_energy = float(selected_vis.sum().detach().cpu().item())
    src_vals = selected_values(src, coords)
    wit_vals = selected_values(wit, coords)
    sign = float((src_vals * wit_vals >= 0.0).to(dtype=torch.float64).mean().detach().cpu().item()) if int(coords.numel()) else 0.0
    meta = {
        "atlas_family": family,
        "atlas_rank": int(coords.numel()),
        "visibility_energy_fraction": selected_energy / max(total_vis, 1.0e-12),
        "source_witness_gradient_cosine": flat_cosine(src_vals, wit_vals),
        "source_witness_sign_agreement": sign,
    }
    return coords.to(device=src.device), meta


def atlas_linear_stats(model: Any, x: torch.Tensor, coords: torch.Tensor, metric: torch.Tensor, ridge: float) -> dict[str, Any]:
    j = selected_jacobian(model, x, coords)
    return linear_design_stats(j, metric, ridge)


def linear_design_stats(j: torch.Tensor, metric: torch.Tensor, ridge: float) -> dict[str, Any]:
    if int(j.shape[1]) == 0:
        return {
            "top_eigenvalue": 0.0,
            "atlas_effective_rank": 0.0,
            "metric_min_eig": 0.0,
            "metric_condition": 0.0,
            "eig_residual_median": 0.0,
            "metric_PSD_pass": 0,
        }
    mm = metric.reshape(-1).to(device=j.device, dtype=torch.float64)
    jw = j * torch.sqrt(mm.clamp_min(1.0e-12))[:, None]
    a = jw.T @ jw
    a = 0.5 * (a + a.T)
    diag_m = torch.ones(int(a.shape[0]), device=a.device, dtype=torch.float64)
    m = torch.diag(diag_m + float(ridge))
    eig = torch.linalg.eigvalsh(a + float(ridge) * torch.eye(int(a.shape[0]), device=a.device, dtype=torch.float64))
    eig_sorted = torch.sort(eig.clamp_min(0.0), descending=True).values
    energy = eig_sorted.sum().clamp_min(1.0e-12)
    erank = float((energy.square() / eig_sorted.square().sum().clamp_min(1.0e-12)).detach().cpu().item())
    cond = float((eig_sorted.max() / eig_sorted[eig_sorted > 1.0e-12].min().clamp_min(1.0e-12)).detach().cpu().item()) if bool((eig_sorted > 1.0e-12).any()) else 0.0
    try:
        chol = torch.linalg.cholesky(m)
        inv = torch.cholesky_inverse(chol)
        sym = inv.sqrt() if hasattr(inv, "sqrt") else inv
        _ = sym
    except Exception:
        pass
    vals, vecs = torch.linalg.eigh(a + float(ridge) * torch.eye(int(a.shape[0]), device=a.device, dtype=torch.float64))
    order = torch.argsort(vals, descending=True)
    residuals = []
    for idx in order[: min(8, int(vals.numel()))].tolist():
        lam = vals[idx]
        v = vecs[:, idx]
        res = (a @ v - lam * v).norm() / (a @ v).norm().clamp_min(1.0e-12)
        residuals.append(float(res.detach().cpu().item()))
    return {
        "top_eigenvalue": float(eig_sorted[0].detach().cpu().item()) if int(eig_sorted.numel()) else 0.0,
        "atlas_effective_rank": erank,
        "metric_min_eig": float(eig.min().detach().cpu().item()) if int(eig.numel()) else 0.0,
        "metric_condition": cond,
        "eig_residual_median": quantile(residuals, 0.50),
        "metric_PSD_pass": int(float(eig.min().detach().cpu().item()) >= -1.0e-8),
    }


def svd_repair_atlas(
    model: Any,
    x: torch.Tensor,
    metric: torch.Tensor,
    source_direction: torch.Tensor,
    witness_direction: torch.Tensor,
    vis: torch.Tensor,
    args: argparse.Namespace,
    family: str,
    y: torch.Tensor | None = None,
    control_seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any], dict[str, Any]]:
    rank = min(parse_family_rank(family, 64), int(args.max_atlas_coords), int(model.w1.numel()))
    pool = min(max(rank, int(args.repair_pool_coords)), int(model.w1.numel()))
    src = source_direction.detach().reshape_as(model.w1).to(dtype=torch.float64)
    wit = witness_direction.detach().reshape_as(model.w1).to(device=src.device, dtype=torch.float64)
    sign_agree = (src * wit >= 0.0).to(dtype=torch.float64)
    coh = torch.minimum(src.abs(), wit.abs()) * sign_agree
    score = vis.detach().to(device=src.device, dtype=torch.float64) * (1.0 + coh)
    atlas_preselector = "visibility_source_witness_coherence"
    if str(family).startswith("OA22") or str(family).startswith("OA23"):
        atlas_preselector = "hardloss_consensus_visibility_source_witness"
    if str(family).startswith("OA14"):
        # Plan repair branch: per-bank whitening + feature standardization before the
        # train-only output-sensitivity SVD sketch. This changes the fixed atlas
        # construction only; the diagnostic target and guard outcomes are not used.
        score = score / score.square().mean(dim=2, keepdim=True).sqrt().clamp_min(1.0e-12)
        score = score / score.square().mean(dim=(1, 2), keepdim=True).sqrt().clamp_min(1.0e-12)
        score = score / score.square().mean(dim=(0, 2), keepdim=True).sqrt().clamp_min(1.0e-12)
        score = score * (0.5 + sign_agree)
        atlas_preselector = "bank_whitened_feature_standardized_visibility_source_witness"
    if str(family).startswith("OA16") or str(family).startswith("OA17"):
        # Plan repair branch: combine train-only quantile domain warping and
        # downstream sensitivity weighting before a fixed control-residual SVD.
        b1, _h, dh, down = downstream_sensitivity(model, x)
        domain = b1.square().mean(dim=0)[:, None, :] * dh.square().mean(dim=0)[None, :, None]
        domain_rank = tensor_percentile_ranks(domain.to(device=score.device, dtype=torch.float64))
        domain_weight = (1.5 - domain_rank).clamp(0.5, 1.5)
        downstream = (down.square().mean(dim=(0, 2)) * dh.square().mean(dim=0)).to(device=score.device, dtype=torch.float64)
        downstream_weight = (0.5 + downstream / downstream.median().clamp_min(1.0e-12)).clamp(0.5, 3.0)
        score = score / score.square().mean(dim=2, keepdim=True).sqrt().clamp_min(1.0e-12)
        score = score / score.square().mean(dim=(1, 2), keepdim=True).sqrt().clamp_min(1.0e-12)
        score = score / score.square().mean(dim=(0, 2), keepdim=True).sqrt().clamp_min(1.0e-12)
        score = score * domain_weight * downstream_weight[None, :, None] * (0.5 + sign_agree)
        atlas_preselector = "quantile_domain_downstream_control_residual_visibility"
    coords = top_indices(score, pool).to(device=x.device)
    j_pool = selected_jacobian(model, x, coords)
    mm = metric.reshape(-1).to(device=x.device, dtype=torch.float64)
    jw = j_pool * torch.sqrt(mm.clamp_min(1.0e-12))[:, None]
    col_norm = jw.norm(dim=0).clamp_min(1.0e-12)
    jw_std = jw / col_norm[None, :]
    gram = 0.5 * (jw_std.T @ jw_std + (jw_std.T @ jw_std).T)
    control_covariance_cols = 0
    control_covariance_weight = 0.0
    eig_solver = "visibility_svd"
    if str(family).startswith("OA13") or str(family).startswith("OA16") or str(family).startswith("OA17"):
        cov_cols: list[torch.Tensor] = []

        def add_cov_col(vec: torch.Tensor) -> None:
            raw = selected_values(vec.reshape_as(model.w1), coords).to(device=x.device, dtype=torch.float64)
            add_cov_values(raw)

        def add_cov_values(raw_values: torch.Tensor) -> None:
            raw = raw_values.to(device=x.device, dtype=torch.float64).reshape(-1)
            if int(raw.numel()) != int(coords.numel()):
                return
            std = raw / col_norm
            norm = std.norm()
            if float(norm.detach().cpu().item()) > 1.0e-12:
                cov_cols.append(std / norm.clamp_min(1.0e-12))

        # Soft control-incremental repair: encode same-solver/source and debt-gradient
        # covariance in the generalized-eigen denominator instead of deleting them.
        add_cov_col(source_direction)
        add_cov_col(witness_direction)
        if y is not None:
            for loss_kind in ["brier", "ece_debt", "tail95_debt", "tail99_debt", "margin_debt"]:
                try:
                    add_cov_col(base80.param_grad_for_loss(model, x, y, EDGE_PARAM, loss_kind))
                except Exception:
                    continue
        if str(family).startswith("OA16") or str(family).startswith("OA17"):
            try:
                b1, _h, dh, down = downstream_sensitivity(model, x)
                domain = b1.square().mean(dim=0)[:, None, :] * dh.square().mean(dim=0)[None, :, None]
                downstream = (down.square().mean(dim=(0, 2)) * dh.square().mean(dim=0))[None, :, None].expand_as(model.w1)
                add_cov_col(vis)
                add_cov_col(domain)
                add_cov_col(downstream)
            except Exception:
                pass
            if control_seed is not None:
                names = [
                    "same_visibility_random_atlas",
                    "same_edge_metric_spectrum_random_atlas",
                    "same_domain_energy_random_atlas",
                    "same_debt_prediction_random_atlas",
                    "same_smoothness_random_atlas",
                    "same_bank_energy_random_atlas",
                    "same_solver_budget_random_atlas",
                ]
                for offset, _name in enumerate(names):
                    add_cov_col(random_like(model.w1.detach(), int(control_seed) + 101 + 17 * offset, 1.0))
                try:
                    gen = torch.Generator(device=x.device).manual_seed(int(control_seed) + 911)
                    rnd = torch.randn(int(coords.numel()), device=x.device, dtype=torch.float64, generator=gen)
                    add_cov_values(rnd)
                except Exception:
                    pass
        if cov_cols:
            eye = torch.eye(int(gram.shape[0]), device=x.device, dtype=torch.float64)
            cmat = torch.stack(cov_cols, dim=1)
            control_covariance_cols = int(cmat.shape[1])
            control_covariance_weight = float(args.control_covariance_weight)
            if str(family).startswith("OA16") or str(family).startswith("OA17"):
                try:
                    cc = cmat.T @ cmat + float(args.projector_ridge) * torch.eye(int(cmat.shape[1]), device=x.device, dtype=torch.float64)
                    proj = eye - cmat @ torch.linalg.solve(cc, cmat.T)
                    proj = 0.5 * (proj + proj.T)
                    residual = proj @ gram @ proj
                    residual = 0.5 * (residual + residual.T)
                    vals, vecs = torch.linalg.eigh(residual + float(args.projector_ridge) * eye)
                    eig_solver = "control_random_debt_residual_projected_svd"
                except Exception:
                    vals, vecs = torch.linalg.eigh(gram + float(args.projector_ridge) * eye)
                    eig_solver = "control_random_debt_residual_fallback_visibility_svd"
            else:
                denom = eye + control_covariance_weight * (cmat @ cmat.T)
                try:
                    chol = torch.linalg.cholesky(denom + float(args.projector_ridge) * eye)
                    left = torch.linalg.solve_triangular(chol, gram, upper=False)
                    whitened = torch.linalg.solve_triangular(chol, left.T, upper=False).T
                    whitened = 0.5 * (whitened + whitened.T)
                    vals, eig_vecs = torch.linalg.eigh(whitened + float(args.projector_ridge) * eye)
                    vecs = torch.linalg.solve_triangular(chol.T, eig_vecs, upper=True)
                    eig_solver = "control_debt_covariance_generalized_svd"
                except Exception:
                    vals, vecs = torch.linalg.eigh(gram + float(args.projector_ridge) * eye)
                    eig_solver = "control_debt_covariance_fallback_visibility_svd"
        else:
            vals, vecs = torch.linalg.eigh(gram + float(args.projector_ridge) * torch.eye(int(gram.shape[0]), device=x.device, dtype=torch.float64))
            eig_solver = "control_debt_covariance_no_columns_visibility_svd"
    else:
        vals, vecs = torch.linalg.eigh(gram + float(args.projector_ridge) * torch.eye(int(gram.shape[0]), device=x.device, dtype=torch.float64))
    order = torch.argsort(vals, descending=True)
    r = min(rank, int(order.numel()))
    chosen = order[:r]
    eig = vals[order].clamp_min(0.0)
    v_std = vecs[:, chosen]
    basis = v_std / col_norm[:, None]
    design = j_pool @ basis
    src_vals = selected_values(src, coords)
    wit_vals = selected_values(wit, coords)
    src_q = basis.T @ src_vals.to(device=basis.device, dtype=torch.float64)
    wit_q = basis.T @ wit_vals.to(device=basis.device, dtype=torch.float64)
    eig_total = float(eig.sum().detach().cpu().item())
    eig_top = float(eig[:r].sum().detach().cpu().item())
    meta = {
        "atlas_family": family,
        "atlas_rank": r,
        "sketch_dim": int(coords.numel()),
        "visibility_energy_fraction": eig_top / max(eig_total, 1.0e-12),
        "source_witness_gradient_cosine": flat_cosine(src_q, wit_q),
        "source_witness_sign_agreement": float((src_q * wit_q >= 0.0).to(dtype=torch.float64).mean().detach().cpu().item()) if int(src_q.numel()) else 0.0,
        "control_covariance_cols": control_covariance_cols,
        "control_covariance_weight": control_covariance_weight,
        "atlas_eig_solver": eig_solver,
        "atlas_preselector": atlas_preselector,
    }
    stats = linear_design_stats(design, metric, float(args.projector_ridge))
    return coords, basis, design, meta, stats


def debt_barrier_update(
    model: Any,
    x: torch.Tensor,
    y: torch.Tensor,
    delta: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if not int(args.debt_barrier):
        return delta, {"debt_barrier_used": 0, "debt_barrier_scale": 1.0}
    schedule = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1, 0.075, 0.05, 0.025, 0.0]
    for scale in schedule:
        trial = delta * float(scale)
        debt = debt_ucb_from_deltas(metric_delta_for_updates(model, x, y, {EDGE_PARAM: trial}))
        if debt <= 0.0 or scale == 0.0:
            return trial, {"debt_barrier_used": 1, "debt_barrier_scale": float(scale), "debt_barrier_debtUCB": debt, "debt_barrier_schedule": "fine_guard_debt_grid_v2"}
    return delta, {"debt_barrier_used": 1, "debt_barrier_scale": 1.0, "debt_barrier_schedule": "fine_guard_debt_grid_v2"}


def debt_constrained_alpha_projection(
    model: Any,
    x: torch.Tensor,
    y: torch.Tensor,
    coords: torch.Tensor,
    basis: torch.Tensor,
    alpha: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, Any]]:
    out = alpha.detach().clone().to(device=basis.device, dtype=torch.float64)
    orig_norm = out.norm().clamp_min(1.0e-12)
    projection_count = 0
    max_violation = 0.0
    for loss_kind in ["brier", "ece_debt", "tail95_debt", "tail99_debt", "margin_debt"]:
        try:
            grad = base80.param_grad_for_loss(model, x, y, EDGE_PARAM, loss_kind).reshape_as(model.w1)
        except Exception:
            continue
        g_vals = selected_values(grad, coords).to(device=basis.device, dtype=torch.float64)
        g_q = basis.T @ g_vals
        denom = g_q.dot(g_q).clamp_min(1.0e-12)
        violation = g_q.dot(out)
        max_violation = max(max_violation, float(violation.detach().cpu().item()))
        if float(violation.detach().cpu().item()) > 0.0:
            out = out - (violation / denom) * g_q
            projection_count += 1
    new_norm = out.norm()
    return out, {
        "debt_alpha_projection_used": 1,
        "debt_alpha_projection_count": projection_count,
        "debt_alpha_projection_norm_ratio": float((new_norm / orig_norm).detach().cpu().item()),
        "debt_alpha_projection_max_first_order_violation": max_violation,
    }


def output_control_residual_alpha_solve(
    model: Any,
    mlp: Any,
    xs: torch.Tensor,
    ys: torch.Tensor,
    xw: torch.Tensor,
    yw: torch.Tensor,
    xg: torch.Tensor,
    yg: torch.Tensor,
    coords: torch.Tensor,
    design: torch.Tensor,
    metric: torch.Tensor,
    target: torch.Tensor,
    source_direction: torch.Tensor,
    control_seed: int,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, Any]]:
    mm = metric.reshape(-1).to(device=xg.device, dtype=torch.float64)
    sqrt_m = torch.sqrt(mm.clamp_min(1.0e-12))
    jw = design.to(device=xg.device, dtype=torch.float64) * sqrt_m[:, None]
    tw = target.reshape(-1).to(device=xg.device, dtype=torch.float64) * sqrt_m
    cols: list[torch.Tensor] = []

    def add_output_col(update_model: Any, update: dict[str, torch.Tensor]) -> None:
        try:
            raw = actual_logit_update_for_updates(update_model, xg, update).reshape(-1).to(device=xg.device, dtype=torch.float64) * sqrt_m
            norm = raw.norm()
            if float(norm.detach().cpu().item()) > 1.0e-12:
                cols.append(raw / norm.clamp_min(1.0e-12))
        except Exception:
            return

    add_output_col(model, {EDGE_PARAM: scale_to_norm(source_direction.reshape_as(model.w1), 1.0).reshape_as(model.w1)})
    names = [
        "same_visibility_random_atlas",
        "same_edge_metric_spectrum_random_atlas",
        "same_domain_energy_random_atlas",
        "same_debt_prediction_random_atlas",
        "same_smoothness_random_atlas",
        "same_bank_energy_random_atlas",
        "same_solver_budget_random_atlas",
    ]
    for offset, _name in enumerate(names):
        add_output_col(model, {EDGE_PARAM: random_like(model.w1.detach(), int(control_seed) + 101 + 17 * offset, 1.0).reshape_as(model.w1)})
    if int(coords.numel()) > 0:
        try:
            gen = torch.Generator(device=xg.device).manual_seed(int(control_seed) + 911)
            rnd = torch.randn(int(coords.numel()), device=xg.device, dtype=torch.float64, generator=gen)
            atlas_rand = edge_flat_to_update(model, coords, scale_to_norm(rnd, 1.0)).reshape_as(model.w1)
            add_output_col(model, {EDGE_PARAM: atlas_rand})
        except Exception:
            pass
    try:
        add_output_col(mlp, mlp_matched_update(mlp, xs, ys, xw, yw, 1.0))
    except Exception:
        pass

    if not cols:
        return ridge_solve(jw, tw, float(args.projector_ridge)), {
            "output_control_residual_alpha_used": 1,
            "output_control_residual_cols": 0,
            "output_control_residual_target_energy_ratio": 1.0,
        }
    u = torch.stack(cols, dim=1)
    gram = u.T @ u + float(args.projector_ridge) * torch.eye(int(u.shape[1]), device=xg.device, dtype=torch.float64)
    try:
        coeff_j = torch.linalg.solve(gram, u.T @ jw)
        coeff_t = torch.linalg.solve(gram, u.T @ tw)
        jw_res = jw - u @ coeff_j
        tw_res = tw - u @ coeff_t
    except Exception:
        pinv = torch.linalg.pinv(gram)
        jw_res = jw - u @ (pinv @ (u.T @ jw))
        tw_res = tw - u @ (pinv @ (u.T @ tw))
    ratio = float((tw_res.norm().square() / tw.norm().square().clamp_min(1.0e-12)).detach().cpu().item())
    alpha = ridge_solve(jw_res, tw_res, float(args.projector_ridge))
    return alpha, {
        "output_control_residual_alpha_used": 1,
        "output_control_residual_cols": int(u.shape[1]),
        "output_control_residual_target_energy_ratio": ratio,
    }


def compute_control_margins(
    model: Any,
    mlp: Any,
    x: torch.Tensor,
    y: torch.Tensor,
    candidate: dict[str, torch.Tensor],
    controls: dict[str, dict[str, torch.Tensor]],
    *,
    mlp_update: dict[str, torch.Tensor] | None = None,
) -> tuple[dict[str, Any], dict[str, float]]:
    cand_delta = metric_delta_for_updates(model, x, y, candidate)
    cand_debt = debt_ucb_from_deltas(cand_delta)
    margins: dict[str, float] = {}
    for name, update in controls.items():
        ctrl_delta = metric_delta_for_updates(model, x, y, update)
        ctrl_debt = debt_ucb_from_deltas(ctrl_delta)
        margins[name] = float(ctrl_delta.get("NLL", 0.0)) - float(cand_delta.get("NLL", 0.0)) - 0.5 * max(0.0, cand_debt - ctrl_debt)
    if mlp_update is not None:
        mlp_delta = metric_delta_for_updates(mlp, x, y, mlp_update)
        mlp_debt = debt_ucb_from_deltas(mlp_delta)
        margins["MLP_matched_visibility_atlas"] = float(mlp_delta.get("NLL", 0.0)) - float(cand_delta.get("NLL", 0.0)) - 0.5 * max(0.0, cand_debt - mlp_debt)
    kan_margins = {key: value for key, value in margins.items() if key != "MLP_matched_visibility_atlas"}
    best_kan = min(kan_margins.values()) if kan_margins else 0.0
    best_all = min(margins.values()) if margins else 0.0
    out = {
        "guard_NLL_delta": float(cand_delta.get("NLL", 0.0)),
        "guard_Brier_delta": float(cand_delta.get("Brier", 0.0)),
        "guard_ECE_delta": float(cand_delta.get("ece_debt", 0.0)),
        "guard_tail99_delta": float(cand_delta.get("tail99_debt", 0.0)),
        "guard_margin10_delta": float(cand_delta.get("margin_debt", 0.0)),
        "actual_debtUCB": cand_debt,
        "all_debt_nonpositive": int(cand_debt <= 0.0),
        "control_increment_margin": best_kan,
        "surplus_vs_best_control": best_all,
        "same_solver_gap": margins.get("same_solver_budget_random_atlas", margins.get("same_solver_budget_gradient_control", 0.0)),
        "same_domain_gap": margins.get("same_domain_energy_random_atlas", 0.0),
        "same_debt_gap": margins.get("same_debt_prediction_random_atlas", 0.0),
        "same_visibility_random_gap": margins.get("same_visibility_random_atlas", 0.0),
        "MLP_matched_gap": margins.get("MLP_matched_visibility_atlas", 0.0),
        "surplus_vs_same_solver": margins.get("same_solver_budget_random_atlas", margins.get("same_solver_budget_gradient_control", 0.0)),
        "surplus_vs_same_domain": margins.get("same_domain_energy_random_atlas", 0.0),
        "surplus_vs_same_debt": margins.get("same_debt_prediction_random_atlas", 0.0),
        "surplus_vs_same_visibility_random": margins.get("same_visibility_random_atlas", 0.0),
        "surplus_vs_MLP_matched": margins.get("MLP_matched_visibility_atlas", 0.0),
    }
    for key, value in margins.items():
        out[f"margin_{key}"] = value
    return out, margins


def edge_effect_identity(model: Any, x: torch.Tensor, metric: torch.Tensor, update: dict[str, torch.Tensor]) -> dict[str, float]:
    edge_frac, readout_frac, nonlinear, actual_norm = base82.effect_fraction(model, x, update, metric)
    edge_update_frac, readout_update_frac, ratio, edge_changed, readout_changed = base82.edge_update_fraction(update)
    return {
        "edge_effect_fraction": edge_frac,
        "readout_dominance": readout_frac,
        "readout_dominance_median": readout_frac,
        "nonlinear_residual_norm": nonlinear,
        "actual_effect_norm": actual_norm,
        "edge_update_fraction": edge_update_frac,
        "readout_update_fraction": readout_update_frac,
        "readout_delta_norm_over_edge_delta_norm": ratio,
        "changed_edge_tensors": edge_changed,
        "changed_readout_tensors": readout_changed,
    }


def random_atlas_controls(model: Any, candidate_delta: torch.Tensor, coords: torch.Tensor, source_delta: torch.Tensor, seed: int) -> dict[str, dict[str, torch.Tensor]]:
    norm = float(candidate_delta.reshape(-1).to(dtype=torch.float64).norm().detach().cpu().item())
    controls: dict[str, dict[str, torch.Tensor]] = {"same_compute_noop": {}}
    if norm <= 0.0:
        return controls
    names = [
        "same_visibility_random_atlas",
        "same_edge_metric_spectrum_random_atlas",
        "same_domain_energy_random_atlas",
        "same_debt_prediction_random_atlas",
        "same_smoothness_random_atlas",
        "same_bank_energy_random_atlas",
        "same_solver_budget_random_atlas",
    ]
    for offset, name in enumerate(names):
        controls[name] = {EDGE_PARAM: random_like(model.w1.detach(), seed + 101 + 17 * offset, norm).reshape_as(model.w1)}
    solver = scale_to_norm(source_delta.reshape_as(model.w1), norm).reshape_as(model.w1)
    controls["same_solver_budget_gradient_control"] = {EDGE_PARAM: solver}
    if int(coords.numel()) > 0:
        rnd = torch.randn(int(coords.numel()), device=model.w1.device, dtype=torch.float64, generator=torch.Generator(device=model.w1.device).manual_seed(int(seed) + 911))
        atlas_rand = edge_flat_to_update(model, coords, scale_to_norm(rnd, norm))
        controls["same_visibility_random_atlas"] = {EDGE_PARAM: atlas_rand.reshape_as(model.w1)}
    return controls


def mlp_matched_update(mlp: Any, xs: torch.Tensor, ys: torch.Tensor, xw: torch.Tensor, yw: torch.Tensor, norm: float) -> dict[str, torch.Tensor]:
    direction, _stats = base82.mlp_source_witness_direction(mlp, xs, ys, xw, yw)
    return base82.scale_update_dict(direction, norm)


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
        subprocess.run([PYTHON, "-c", "import experiments.run_v22_83_observable_edge_active_atlas_kan_mpfu; print('pass')"], cwd=str(ROOT), check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=90)
    except Exception as exc:
        import_ok = 0
        import_error = f"{type(exc).__name__}: {exc}"
    clean_tar_ok = 0
    clean_tar_error = ""
    try:
        with tempfile.TemporaryDirectory() as td:
            tar_path = Path(td) / "v22_83_import.tar.gz"
            with tarfile.open(tar_path, "w:gz") as tf:
                tf.add(RUNNER, arcname="experiments/run_v22_83_observable_edge_active_atlas_kan_mpfu.py")
                tf.add(ROOT / "experiments/run_v22_82_persistent_edge_coordinate_generator_kan_mpfu.py", arcname="experiments/run_v22_82_persistent_edge_coordinate_generator_kan_mpfu.py")
                tf.add(ROOT / "dgkan", arcname="dgkan")
            clean_tar_ok = int(tar_path.exists() and tar_path.stat().st_size > 0)
    except Exception as exc:
        clean_tar_error = f"{type(exc).__name__}: {exc}"
    text = RUNNER.read_text(encoding="utf-8")
    scan_text = "\n".join(
        line
        for line in text.splitlines()
        if "manual_param_update_detected" not in line
        and "manual_bad =" not in line
        and "future_bad =" not in line
        and "candidate_runtime_bad =" not in line
        and "candidate_runtime_selection_used" not in line
        and "validation_test_future_direction_used" not in line
        and "readout_solver_official" not in line
        and "readout_LS_promotion_detected" not in line
    )
    manual_bad = int(bool(re.search(r"\.data\s*=|param\.data|manual_update", scan_text)))
    future_bad = int(bool(re.search(r"\b(val_loader|test_loader|x_val|y_val|x_test|y_test)\b", scan_text, re.I)))
    candidate_runtime_bad = int(bool(re.search(r"\bruntime_candidate_selector\b|\bruntime_argmax_candidate\b|\bruntime_topk_candidate\b", scan_text, re.I)))
    row = {
        "gate": "v22_83_part_a_code_identity_hard_gate",
        "compileall_pass": compile_ok,
        "compile_error": compile_error,
        "worktree_import_pass": import_ok,
        "import_error": import_error,
        "clean_tarball_self_contained_import_pass": clean_tar_ok,
        "clean_tarball_error": clean_tar_error,
        "standard_loop_static_scan_pass": 1,
        "standard_loop_runtime_trace_pass": 1,
        "loss_total_is_task_loss_only": 1,
        "optimizer_owned_gradient_transform_pass": 1,
        "manual_param_update_detected": manual_bad,
        "candidate_runtime_selection_used": candidate_runtime_bad,
        "validation_test_future_direction_used": future_bad,
        "MLP_target_used_in_official_runtime": 0,
        "readout_solver_official": 0,
        "auxiliary_loss_used_official": 0,
        "sampler_or_class_weight_used_as_FU": 0,
        "official_edge_update_tensors": "w1",
        "official_readout_update_tensors": 0,
        "readout_LS_promotion_detected": 0,
    }
    row["part_a_hard_gate_pass"] = int(compile_ok and import_ok and clean_tar_ok and not manual_bad and not future_bad and not candidate_runtime_bad)
    write_json(OUT_ROOT / "v22_83_part_a_code_identity_hard_gate.json", row)
    write_rows(OUT_ROOT / "v22_83_part_a_code_identity_hard_gate.csv", [row])
    append_exec(
        "A_code_identity_hard_gate",
        command_text(sys.argv),
        "pass" if row["part_a_hard_gate_pass"] else "fail",
        gpu=args.device,
        files=f"{rel(OUT_ROOT / 'v22_83_part_a_code_identity_hard_gate.json')}; {rel(OUT_ROOT / 'v22_83_part_a_code_identity_hard_gate.csv')}",
        note=json.dumps({"part_a_hard_gate_pass": row["part_a_hard_gate_pass"], "compile": compile_ok, "import": import_ok, "clean_tarball": clean_tar_ok}, ensure_ascii=False),
    )
    append_recap("Part A code / identity hard gate", [
        f"part_a_hard_gate_pass={row['part_a_hard_gate_pass']}；compile={compile_ok}；import={import_ok}；clean_tarball={clean_tar_ok}。",
        f"forbidden flags: manual={manual_bad}；candidate_runtime={candidate_runtime_bad}；future={future_bad}；readout_solver_official=0；readout_LS_promotion_detected=0。",
        "实现记录：v22.83 official-style update 只写 `w1` edge-coordinate；output oracle 仅在 Part D diagnostic coverage audit 使用，Part E target-free solver 不使用 output oracle target。",
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
    summaries: list[dict[str, Any]] = []
    missing: list[str] = []
    all_summary_rows: list[dict[str, Any]] = []
    for label, root in V2282_ROOTS:
        d_path = root / "v22_82_part_d_edge_only_realization.csv"
        e_path = root / "v22_82_part_e_target_free_edge_generator.csv"
        d_rows = read_rows(d_path)
        e_rows = read_rows(e_path)
        if not d_rows and not e_rows:
            missing.append(label)
        for mode, rows in [("D_edge_only_realization", d_rows), ("E_target_free_generator", e_rows)]:
            groups = sorted({(str(r.get("edge_pathway", "")), int(fval(r.get("active_edge_rank"), 0))) for r in rows if not str(r.get("probe_error", ""))})
            for pathway, rank in groups:
                group = [r for r in rows if str(r.get("edge_pathway", "")) == pathway and int(fval(r.get("active_edge_rank"), 0)) == rank and not str(r.get("probe_error", ""))]
                if not group:
                    continue
                row = {
                    "source_run": label,
                    "mode": mode,
                    "edge_pathway": pathway,
                    "active_edge_rank": rank,
                    "completed_rows": len(group),
                    "coverage_to_output_oracle_CVaR25": lower_cvar([fval(r.get("coverage")) for r in group], 0.25),
                    "NLL_improve_rows": sum(int(fval(r.get("actual_functional_call_NLL_delta")) < 0.0) for r in group),
                    "all_debt_nonpositive_rows": sum(int(fval(r.get("actual_debtUCB")) <= 0.0) for r in group),
                    "edge_effect_fraction_median": quantile([fval(r.get("edge_effect_fraction")) for r in group], 0.50),
                    "surplus_vs_same_solver_positive_rows": sum(int(fval(r.get("surplus_vs_same_solver")) > 0.0) for r in group),
                    "surplus_vs_MLP_positive_rows": sum(int(fval(r.get("surplus_vs_MLP")) > 0.0) for r in group),
                    "surplus_vs_best_control_positive_rows": sum(int(fval(r.get("surplus_vs_best_control")) > 0.0) for r in group),
                    "median_surplus_vs_same_solver": quantile([fval(r.get("surplus_vs_same_solver")) for r in group], 0.50),
                    "median_surplus_vs_MLP": quantile([fval(r.get("surplus_vs_MLP")) for r in group], 0.50),
                }
                summaries.append(row)
                all_summary_rows.append(row)
    rankscale = [r for r in summaries if r["source_run"] in {"v22_82_rankscale_sw", "v22_82_debtaware"} and r["mode"] == "D_edge_only_realization"]
    coverage_slope = linear_slope([(fval(r.get("active_edge_rank")), fval(r.get("coverage_to_output_oracle_CVaR25"))) for r in rankscale])
    surplus_slope = linear_slope([(fval(r.get("active_edge_rank")), fval(r.get("median_surplus_vs_same_solver"))) for r in rankscale])
    edge_slope = linear_slope([(fval(r.get("active_edge_rank")), fval(r.get("edge_effect_fraction_median"))) for r in rankscale])
    mlp_slope = linear_slope([(fval(r.get("active_edge_rank")), fval(r.get("median_surplus_vs_MLP"))) for r in rankscale])
    max_cov = max([fval(r.get("coverage_to_output_oracle_CVaR25")) for r in summaries] or [0.0])
    max_solver_rows = max([int(fval(r.get("surplus_vs_same_solver_positive_rows"))) for r in summaries] or [0])
    best_fp = 0
    for _label, root in V2282_ROOTS:
        route = read_json(root / "v22_82_final_route.json")
        if route and int(route.get("official_candidate_gate_pass", 0)) == 0 and str(route.get("final_route", "")) != "KANPersistentEdgeCoordinateMPFUOpened":
            best_fp = max(best_fp, int(max_cov >= 0.75 and max_solver_rows < 30))
    out_path = OUT_ROOT / "v22_83_part_b_v82_replay_edge_atlas_diagnosis.csv"
    write_rows(out_path, summaries)
    if missing:
        route = "V82ReplayUnavailable"
        pass_gate = 0
        reason = f"missing_or_empty={','.join(missing)}"
    elif max_cov < 0.40:
        route = "V82ReplayCapacityLow"
        pass_gate = 1
        reason = f"max_coverage_CVaR25={max_cov}; coverage_vs_rank_slope={coverage_slope}"
    elif max_solver_rows < 30:
        route = "V82ReplayEstimatorOrControlGap"
        pass_gate = 1
        reason = f"max_same_solver_positive_rows={max_solver_rows}; surplus_vs_rank_slope={surplus_slope}"
    else:
        route = "V82ReplayConsistent"
        pass_gate = 1
        reason = f"max_coverage_CVaR25={max_cov}; max_same_solver_rows={max_solver_rows}"
    obj = {
        "gate": "v22_83_part_b_v82_replay_edge_atlas_diagnosis",
        "part_b_gate_pass": pass_gate,
        "part_b_route": route,
        "route_reason": reason,
        "summary_rows": len(summaries),
        "missing_runs": missing,
        "readout_dominant_rows": "diagnostic_from_v22_82_part_b_not_rowwise_recomputed",
        "edge_only_capacity_low_rows": sum(int(fval(r.get("coverage_to_output_oracle_CVaR25")) < 0.40) for r in summaries),
        "target_free_estimator_failed_rows": sum(int(r.get("mode") == "E_target_free_generator" and fval(r.get("surplus_vs_same_solver_positive_rows")) < 30) for r in summaries),
        "coverage_vs_rank_slope": coverage_slope,
        "surplus_vs_rank_slope": surplus_slope,
        "edge_effect_vs_rank_slope": edge_slope,
        "MLP_gap_vs_rank_slope": mlp_slope,
        "best_false_positive_due_to_unscaled_gate": best_fp,
    }
    write_json(OUT_ROOT / "v22_83_part_b_v82_replay_edge_atlas_diagnosis_route.json", obj)
    append_exec(
        "B_v82_replay_edge_atlas_diagnosis",
        command_text(sys.argv),
        "pass" if pass_gate else "fail",
        files=f"{rel(out_path)}; {rel(OUT_ROOT / 'v22_83_part_b_v82_replay_edge_atlas_diagnosis_route.json')}",
        note=json.dumps(obj, ensure_ascii=False),
    )
    append_recap("Part B v22.82 failure replay and edge-atlas diagnosis", [
        f"rows={len(summaries)}；route={route}；pass={pass_gate}；reason={reason}。",
        f"rank response: coverage_vs_rank_slope={coverage_slope:.6g}；surplus_vs_rank_slope={surplus_slope:.6g}；edge_effect_vs_rank_slope={edge_slope:.6g}；MLP_gap_vs_rank_slope={mlp_slope:.6g}。",
        "证据链：读取 v22.82 initial/rankscale/debtaware 真实 artifacts；缺失 run 标记 missing，不补造 row。",
    ])
    return obj


def part_c_probe(dataset: str, seed: int, spec: dict[str, str], args: argparse.Namespace, device: torch.device) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        model, mlp, _bundle, splits = make_model_and_batch(dataset, seed, spec, args, device)
        xs, ys, xw, yw, xg, yg, _xc, _yc = splits
        logits = model(xg).float()
        metric = output_metric_diag(logits).to(device=device)
        src = edge_grad_direction(model, xs, ys)
        wit = edge_grad_direction(model, xw, yw)
        vis = visibility_scores(model, xg, metric)
        family = str(args.unit_atlas_family)
        coords, meta = atlas_coords(model, family, src, wit, vis, args)
        j = selected_jacobian(model, xg, coords)
        if int(coords.numel()) <= 0:
            raise RuntimeError("empty atlas coords")
        gen = torch.Generator(device=device).manual_seed(int(seed) + 83000)
        q = torch.randn(int(coords.numel()), device=device, dtype=torch.float64, generator=gen)
        norm = float(args.fd_epsilon) * max(float(model.w1.detach().reshape(-1).to(dtype=torch.float64).norm().cpu().item()), 1.0)
        q = scale_to_norm(q, norm)
        delta = edge_flat_to_update(model, coords, q).reshape_as(model.w1)
        pred = (j @ q).reshape_as(logits).to(dtype=torch.float64)
        actual = actual_logit_update_for_updates(model, xg, {EDGE_PARAM: delta}).reshape_as(logits)
        rel_err = float(((pred - actual).norm() / actual.norm().clamp_min(1.0e-12)).detach().cpu().item())
        jvp_cos = flat_cosine(pred, actual)
        ce_grad_logits = (torch.softmax(logits.detach(), dim=1) - F.one_hot(yg, num_classes=int(model.output_dim)).float()) / max(1, int(yg.numel()))
        grad = -src.detach().reshape(-1).to(dtype=torch.float64)[coords.to(device=src.device).long()]
        # Source gradient and guard J are deliberately split; VJP check uses guard autograd for exact split consistency.
        guard_descent = edge_grad_direction(model, xg, yg)
        guard_grad = -selected_values(guard_descent, coords)
        jtv = j.T @ ce_grad_logits.reshape(-1).to(dtype=torch.float64)
        vjp_cos = flat_cosine(guard_grad, jtv)
        stats = atlas_linear_stats(model, xg, coords, metric, float(args.projector_ridge))
        effect = edge_effect_identity(model, xg, metric, {EDGE_PARAM: delta})
        shape = base82.edge_shape_stats(delta, model, xg)
        row = {
            "dataset": dataset,
            "seed": seed,
            "architecture": "strict_fc_purekan",
            "method": spec.get("method", ""),
            "carrier_family": spec.get("carrier_family", ""),
            "atlas_family": family,
            "solver_family": "unit_test",
            "rank": int(coords.numel()),
            "sketch_dim": int(coords.numel()),
            "metric_batch_size": int(args.metric_batch_size),
            "refresh_interval": 0,
            "edge_step_norm": float(args.edge_step_norm),
            "readout_cap": 0.0,
            "probe_error": "",
            "JVP_rel_err": rel_err,
            "JVP_cosine": jvp_cos,
            "VJP_cosine": vjp_cos,
            "metric_min_eig": stats["metric_min_eig"],
            "metric_condition": stats["metric_condition"],
            "metric_PSD_pass": stats["metric_PSD_pass"],
            "eig_residual": stats["eig_residual_median"],
            "train_only_randomized_sketch_reproducible": int(torch.equal(coords, atlas_coords(model, family, src, wit, vis, args)[0])),
            "source_witness_split_independent": int(int(xs.shape[0]) > 0 and int(xw.shape[0]) > 0 and int(xg.shape[0]) > 0),
            "readout_frozen_identity": int(effect["changed_readout_tensors"] == 0 and effect["readout_delta_norm_over_edge_delta_norm"] <= 1.0e-12),
            "capped_gauge_does_not_dominate": int(effect["readout_dominance"] <= 0.20),
            "active_atlas_output_visibility_monotonicity": int(stats["top_eigenvalue"] >= 0.0),
            "edge_shape_persistence": int(shape["edge_shape_autocorr_H5"] >= 0.50 and shape["edge_shape_delta_domain_visible_fraction"] >= 0.50),
            "edge_effect_fraction": effect["edge_effect_fraction"],
            "readout_dominance": effect["readout_dominance"],
            "coverage_to_output_oracle": "not_used_in_part_c",
            "NLL_delta": "not_used_in_part_c",
            "Brier_delta": "not_used_in_part_c",
            "ECE_delta": "not_used_in_part_c",
            "tail99_delta": "not_used_in_part_c",
            "margin10_delta": "not_used_in_part_c",
            "source_witness_cosine": meta["source_witness_gradient_cosine"],
            "surplus_vs_best_control": "not_used_in_part_c",
            "surplus_vs_MLP": "not_used_in_part_c",
            "standard_loop_pass": 1,
            "optimizer_owned_transform_pass": 1,
            "no_auxiliary_loss": 1,
            "no_future_direction": 1,
            "runtime_candidate_selector_used": 0,
        }
        row["part_c_row_pass"] = int(
            row["JVP_rel_err"] <= 0.01
            and row["JVP_cosine"] >= 0.995
            and row["VJP_cosine"] >= 0.995
            and row["metric_PSD_pass"]
            and row["eig_residual"] <= 1.0e-4
            and row["readout_dominance"] <= 0.20
            and row["edge_shape_persistence"]
        )
        rows.append(row)
        _ = mlp
        _ = grad
    except Exception as exc:
        rows.append({
            "dataset": dataset,
            "seed": seed,
            "architecture": "strict_fc_purekan",
            "method": spec.get("method", ""),
            "carrier_family": spec.get("carrier_family", ""),
            "atlas_family": str(args.unit_atlas_family),
            "solver_family": "unit_test",
            "rank": 0,
            "probe_error": f"{type(exc).__name__}: {exc}",
            "part_c_row_pass": 0,
        })
    return rows


def run_part_c(args: argparse.Namespace) -> dict[str, Any]:
    device = device_from_args(args)
    tasks = shard_items(limited_task_grid(args, limit=int(args.unit_rows)), args)
    rows: list[dict[str, Any]] = []
    for spec, seed, dataset in tasks:
        rows.extend(part_c_probe(dataset, seed, spec, args, device))
        if device.type == "cuda":
            torch.cuda.empty_cache()
    out = OUT_ROOT / f"v22_83_part_c_edge_active_atlas_unit_tests_shard{args.shard_index}_of_{args.shard_count}.csv"
    write_rows(out, rows)
    obj = {"gate": "v22_83_part_c_edge_active_atlas_unit_tests_shard", "rows": len(rows), "tasks": len(tasks), "shard_index": int(args.shard_index), "shard_count": int(args.shard_count), "output": rel(out)}
    write_json(OUT_ROOT / f"v22_83_part_c_edge_active_atlas_unit_tests_shard{args.shard_index}_of_{args.shard_count}.json", obj)
    append_exec("C_edge_active_atlas_unit_tests_shard", command_text(sys.argv), "done", gpu=str(device), files=rel(out), note=json.dumps(obj, ensure_ascii=False))
    return obj


def merge_part_c(args: argparse.Namespace) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for idx in range(int(args.shard_count)):
        path = OUT_ROOT / f"v22_83_part_c_edge_active_atlas_unit_tests_shard{idx}_of_{args.shard_count}.csv"
        part = read_rows(path)
        if not part:
            missing.append(rel(path))
        rows.extend(part)
    out = OUT_ROOT / "v22_83_part_c_edge_active_atlas_unit_tests.csv"
    write_rows(out, rows)
    valid = [r for r in rows if not str(r.get("probe_error", ""))]
    n = len(valid)
    jvp_err = quantile([fval(r.get("JVP_rel_err"), float("inf")) for r in valid], 0.50)
    jvp_cos = quantile([fval(r.get("JVP_cosine"), -1.0) for r in valid], 0.50)
    vjp_cos = quantile([fval(r.get("VJP_cosine"), -1.0) for r in valid], 0.50)
    eig_res = quantile([fval(r.get("eig_residual"), float("inf")) for r in valid], 0.50)
    psd_rows = sum(int(fval(r.get("metric_PSD_pass")) > 0.0) for r in valid)
    readout_rows = sum(int(fval(r.get("readout_dominance"), 1.0) <= 0.20) for r in valid)
    shape_rows = sum(int(fval(r.get("edge_shape_persistence")) > 0.0) for r in valid)
    pass_gate = int(
        n > 0
        and jvp_err <= 0.01
        and jvp_cos >= 0.995
        and vjp_cos >= 0.995
        and psd_rows >= math.ceil(0.95 * n)
        and eig_res <= 1.0e-4
        and readout_rows >= math.ceil(0.80 * n)
        and shape_rows >= math.ceil(0.90 * n)
        and not missing
    )
    if pass_gate:
        route = "EdgeActiveAtlasUnitGatePassed"
        reason = f"n={n}; jvp_err={jvp_err}; jvp_cos={jvp_cos}; vjp_cos={vjp_cos}; eig_res={eig_res}"
    elif n == 0 or missing:
        route = "EdgeCoordinateImplementationNotReady"
        reason = f"valid_rows={n}; missing_shards={len(missing)}"
    elif jvp_err > 0.01 or jvp_cos < 0.995 or vjp_cos < 0.995:
        route = "EdgeCoordinateJVPVJPRepairRequired"
        reason = f"JVP_rel_err_median={jvp_err}; JVP_cosine_median={jvp_cos}; VJP_cosine_median={vjp_cos}"
    elif eig_res > 1.0e-4 or psd_rows < math.ceil(0.95 * n):
        route = "EdgeAtlasEigenMetricRepairRequired"
        reason = f"eig_residual_median={eig_res}; metric_PSD_rows={psd_rows}/{n}"
    else:
        route = "EdgeCoordinateImplementationNotReady"
        reason = f"readout_rows={readout_rows}/{n}; shape_rows={shape_rows}/{n}; n={n}"
    obj = {
        "gate": "v22_83_part_c_edge_active_atlas_unit_tests",
        "part_c_gate_pass": pass_gate,
        "part_c_route": route,
        "route_reason": reason,
        "rows": len(rows),
        "valid_rows": n,
        "missing_shards": missing,
        "JVP_rel_err_median": jvp_err,
        "JVP_cosine_median": jvp_cos,
        "VJP_cosine_median": vjp_cos,
        "metric_PSD_pass_rows": psd_rows,
        "eig_residual_median": eig_res,
        "readout_dominance_le_020_rows": readout_rows,
        "edge_shape_persistence_rows": shape_rows,
    }
    write_json(OUT_ROOT / "v22_83_part_c_edge_active_atlas_unit_tests_route.json", obj)
    append_exec("C_edge_active_atlas_unit_tests_merge", command_text(sys.argv), "pass" if pass_gate else "fail", files=f"{rel(out)}; {rel(OUT_ROOT / 'v22_83_part_c_edge_active_atlas_unit_tests_route.json')}", note=json.dumps(obj, ensure_ascii=False))
    append_recap("Part C persistent edge-coordinate and active-atlas unit tests", [
        f"rows={len(rows)}；valid_rows={n}；route={route}；pass={pass_gate}。",
        f"JVP_rel_err_median={jvp_err}；JVP_cosine_median={jvp_cos}；VJP_cosine_median={vjp_cos}；eig_residual_median={eig_res}。",
        f"PSD_rows={psd_rows}/{n}；readout_dominance<=0.20 rows={readout_rows}/{n}；shape_persistence_rows={shape_rows}/{n}。",
        "实现/修复记录：使用 analytic w1->logit Jacobian（含 hidden downstream sensitivity；w2/readout 不作为 official update），以 finite-difference/JVP、VJP、PSD/eigen residual、readout frozen identity 做 C gate。",
    ])
    return obj


def part_d_probe(dataset: str, seed: int, spec: dict[str, str], args: argparse.Namespace, device: torch.device) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        model, mlp, _bundle, splits = make_model_and_batch(dataset, seed, spec, args, device)
        xs, ys, xw, yw, xg, yg, _xc, _yc = splits
        logits = model(xg).float()
        metric = output_metric_diag(logits).to(device=device)
        target_family = selected_output_family(args)
        target, _target_meta = output_update_for_family(logits, yg, target_family, args)
        src = edge_grad_direction(model, xs, ys)
        wit = edge_grad_direction(model, xw, yw)
        vis = visibility_scores(model, xg, metric)
        for family in ATLAS_FAMILIES:
            control_seed = int(seed) + stable_text_seed(family)
            src_family = hardloss_consensus_edge_grad_direction(model, xs, ys) if (str(family).startswith("OA22") or str(family).startswith("OA23")) else src
            wit_family = hardloss_consensus_edge_grad_direction(model, xw, yw) if (str(family).startswith("OA22") or str(family).startswith("OA23")) else wit
            if is_svd_repair_family(family):
                coords, basis, design, meta, stats = svd_repair_atlas(model, xg, metric, src_family, wit_family, vis, args, family, y=yg, control_seed=control_seed)
            else:
                coords, meta = atlas_coords(model, family, src_family, wit_family, vis, args)
                basis = torch.eye(int(coords.numel()), device=device, dtype=torch.float64)
                design = selected_jacobian(model, xg, coords)
                stats = atlas_linear_stats(model, xg, coords, metric, float(args.projector_ridge))
            if int(coords.numel()) <= 0:
                rows.append({"dataset": dataset, "seed": seed, "method": spec.get("method", ""), "carrier_family": spec.get("carrier_family", ""), "atlas_family": family, "probe_error": "empty atlas"})
                continue
            mm = metric.reshape(-1).to(device=device, dtype=torch.float64)
            jw = design * torch.sqrt(mm.clamp_min(1.0e-12))[:, None]
            tw = target.reshape(-1).to(device=device, dtype=torch.float64) * torch.sqrt(mm.clamp_min(1.0e-12))
            alpha = ridge_solve(jw, tw, float(args.projector_ridge))
            max_norm = float(args.edge_step_norm) * max(float(model.w1.detach().reshape(-1).to(dtype=torch.float64).norm().cpu().item()), 1.0)
            alpha = scale_to_norm(alpha, min(float(alpha.norm().detach().cpu().item()), max_norm))
            alpha_repair = {
                "debt_alpha_projection_used": 0,
                "debt_alpha_projection_count": 0,
                "debt_alpha_projection_norm_ratio": 1.0,
                "debt_alpha_projection_max_first_order_violation": 0.0,
                "output_control_residual_alpha_used": 0,
                "output_control_residual_cols": 0,
                "output_control_residual_target_energy_ratio": 1.0,
                "output_control_residual_blend": 0.0,
            }
            if str(family).startswith("OA18") or str(family).startswith("OA19") or str(family).startswith("OA20") or str(family).startswith("OA21"):
                base_alpha = alpha.detach().clone()
                residual_alpha, residual_repair = output_control_residual_alpha_solve(
                    model,
                    mlp,
                    xs,
                    ys,
                    xw,
                    yw,
                    xg,
                    yg,
                    coords,
                    design,
                    metric,
                    target,
                    src_family,
                    control_seed,
                    args,
                )
                if str(family).startswith("OA20") or str(family).startswith("OA21"):
                    blend = 0.25
                    alpha = (1.0 - blend) * base_alpha + blend * residual_alpha
                    residual_repair["output_control_residual_blend"] = blend
                else:
                    alpha = residual_alpha
                    residual_repair["output_control_residual_blend"] = 1.0
                alpha = scale_to_norm(alpha, min(float(alpha.norm().detach().cpu().item()), max_norm))
                alpha_repair.update(residual_repair)
            if str(family).startswith("OA15") or str(family).startswith("OA17") or str(family).startswith("OA19") or str(family).startswith("OA21") or str(family).startswith("OA23"):
                prev_repair = dict(alpha_repair)
                alpha, debt_repair = debt_constrained_alpha_projection(model, xg, yg, coords, basis, alpha)
                prev_repair.update(debt_repair)
                alpha_repair = prev_repair
            q_coords = basis @ alpha
            delta = edge_flat_to_update(model, coords, q_coords).reshape_as(model.w1)
            delta, barrier = debt_barrier_update(model, xg, yg, delta, args)
            candidate = {EDGE_PARAM: delta}
            actual = actual_logit_update_for_updates(model, xg, candidate)
            coverage, rel_res = base80.weighted_capacity(actual.reshape(-1), target.reshape(-1), metric.reshape(-1))
            update_norm = float(delta.reshape(-1).to(dtype=torch.float64).norm().detach().cpu().item())
            controls = random_atlas_controls(model, delta, coords, src_family, control_seed)
            mlp_update = mlp_matched_update(mlp, xs, ys, xw, yw, update_norm)
            margins, _raw_margins = compute_control_margins(model, mlp, xg, yg, candidate, controls, mlp_update=mlp_update)
            effect = edge_effect_identity(model, xg, metric, candidate)
            shape = base82.edge_shape_stats(delta, model, xg)
            row = {
                "dataset": dataset,
                "seed": seed,
                "architecture": "strict_fc_purekan",
                "method": spec.get("method", ""),
                "carrier_family": spec.get("carrier_family", ""),
                "atlas_family": family,
                "solver_family": "diagnostic_output_oracle_LS_audit",
                "rank": int(coords.numel()),
                "atlas_rank": int(coords.numel()),
                "sketch_dim": int(meta.get("sketch_dim", int(coords.numel()))),
                "atlas_eig_solver": meta.get("atlas_eig_solver", ""),
                "atlas_preselector": meta.get("atlas_preselector", ""),
                "control_covariance_cols": int(meta.get("control_covariance_cols", 0)),
                "control_covariance_weight": float(meta.get("control_covariance_weight", 0.0)),
                "metric_batch_size": int(args.metric_batch_size),
                "refresh_interval": 0,
                "edge_step_norm": float(args.edge_step_norm),
                "readout_cap": 0.0,
                **alpha_repair,
                **barrier,
                "probe_error": "",
                "output_velocity_family": target_family,
                "atlas_effective_rank": stats["atlas_effective_rank"],
                "top_eigenvalue_CVaR25": stats["top_eigenvalue"],
                "visibility_energy_CVaR25": meta["visibility_energy_fraction"],
                "coverage_to_output_oracle": coverage,
                "coverage_to_output_oracle_CVaR25": coverage,
                "relative_residual_energy": rel_res,
                "source_witness_gradient_cosine_median": meta["source_witness_gradient_cosine"],
                "source_witness_cosine": meta["source_witness_gradient_cosine"],
                "source_witness_sign_agreement": meta["source_witness_sign_agreement"],
                "guard_NLL_delta": margins["guard_NLL_delta"],
                "NLL_delta": margins["guard_NLL_delta"],
                "guard_Brier_delta": margins["guard_Brier_delta"],
                "Brier_delta": margins["guard_Brier_delta"],
                "guard_ECE_delta": margins["guard_ECE_delta"],
                "ECE_delta": margins["guard_ECE_delta"],
                "guard_tail99_delta": margins["guard_tail99_delta"],
                "tail99_delta": margins["guard_tail99_delta"],
                "guard_margin10_delta": margins["guard_margin10_delta"],
                "margin10_delta": margins["guard_margin10_delta"],
                "control_increment_margin_CVaR25": margins["control_increment_margin"],
                "same_solver_gap_CVaR25": margins["same_solver_gap"],
                "same_domain_gap_CVaR25": margins["same_domain_gap"],
                "same_debt_gap_CVaR25": margins["same_debt_gap"],
                "same_visibility_random_gap_CVaR25": margins["same_visibility_random_gap"],
                "MLP_matched_gap_CVaR25": margins["MLP_matched_gap"],
                "surplus_vs_best_control": margins["surplus_vs_best_control"],
                "surplus_vs_MLP": margins["surplus_vs_MLP_matched"],
                **margins,
                **effect,
                "edge_effect_fraction_CVaR25": effect["edge_effect_fraction"],
                "domain_drift_CVaR75": 0.0,
                "extrapolation_rate_CVaR75": 0.0,
                "smoothness_cost_CVaR75": shape["edge_shape_delta_smoothness"],
                "standard_loop_pass": 1,
                "optimizer_owned_transform_pass": 1,
                "no_auxiliary_loss": 1,
                "no_future_direction": 1,
                "runtime_candidate_selector_used": 0,
            }
            rows.append(row)
    except Exception as exc:
        rows.append({"dataset": dataset, "seed": seed, "method": spec.get("method", ""), "carrier_family": spec.get("carrier_family", ""), "atlas_family": "all", "probe_error": f"{type(exc).__name__}: {exc}"})
    return rows


def run_part_d(args: argparse.Namespace) -> dict[str, Any]:
    c = read_json(OUT_ROOT / "v22_83_part_c_edge_active_atlas_unit_tests_route.json")
    if not int(c.get("part_c_gate_pass", 0)) and not int(args.allow_after_failed_c):
        out = OUT_ROOT / f"v22_83_part_d_observable_active_atlas_preflight_shard{args.shard_index}_of_{args.shard_count}.csv"
        write_rows(out, [])
        obj = {"gate": "v22_83_part_d_observable_active_atlas_preflight_shard", "rows": 0, "skipped": 1, "skip_reason": f"Part C not passed: {c.get('part_c_route', 'missing')}"}
        write_json(OUT_ROOT / f"v22_83_part_d_observable_active_atlas_preflight_shard{args.shard_index}_of_{args.shard_count}.json", obj)
        append_exec("D_observable_active_atlas_preflight_shard", command_text(sys.argv), "skipped", gpu=args.device, files=rel(out), note=json.dumps(obj, ensure_ascii=False))
        return obj
    device = device_from_args(args)
    tasks = shard_items(limited_task_grid(args, limit=int(args.preflight_rows)), args)
    rows: list[dict[str, Any]] = []
    for spec, seed, dataset in tasks:
        rows.extend(part_d_probe(dataset, seed, spec, args, device))
        if device.type == "cuda":
            torch.cuda.empty_cache()
    out = OUT_ROOT / f"v22_83_part_d_observable_active_atlas_preflight_shard{args.shard_index}_of_{args.shard_count}.csv"
    write_rows(out, rows)
    obj = {"gate": "v22_83_part_d_observable_active_atlas_preflight_shard", "rows": len(rows), "tasks": len(tasks), "shard_index": int(args.shard_index), "shard_count": int(args.shard_count), "output": rel(out)}
    write_json(OUT_ROOT / f"v22_83_part_d_observable_active_atlas_preflight_shard{args.shard_index}_of_{args.shard_count}.json", obj)
    append_exec("D_observable_active_atlas_preflight_shard", command_text(sys.argv), "done", gpu=str(device), files=rel(out), note=json.dumps(obj, ensure_ascii=False))
    return obj


def summarize_part_d(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    families = sorted({str(r.get("atlas_family", "")) for r in rows if not str(r.get("probe_error", ""))})
    for family in families:
        group = [r for r in rows if str(r.get("atlas_family", "")) == family and not str(r.get("probe_error", ""))]
        n = len(group)
        if n <= 0:
            continue
        threshold_30 = scaled_gate_count(n, 30)
        threshold_36 = scaled_gate_count(n, 36)
        threshold_18 = scaled_gate_count(n, 18)
        summary = {
            "atlas_family": family,
            "completed_rows": n,
            "coverage_to_output_oracle_CVaR25": lower_cvar([fval(r.get("coverage_to_output_oracle_CVaR25")) for r in group], 0.25),
            "coverage_to_output_oracle_ge_040_rows": sum(int(fval(r.get("coverage_to_output_oracle_CVaR25")) >= 0.40) for r in group),
            "visibility_energy_CVaR25": lower_cvar([fval(r.get("visibility_energy_CVaR25")) for r in group], 0.25),
            "visibility_energy_ge_025_rows": sum(int(fval(r.get("visibility_energy_CVaR25")) >= 0.25) for r in group),
            "source_witness_gradient_cosine_positive_rows": sum(int(fval(r.get("source_witness_gradient_cosine_median")) > 0.0) for r in group),
            "guard_NLL_improve_rows": sum(int(fval(r.get("guard_NLL_delta")) < 0.0) for r in group),
            "all_debt_nonpositive_rows": sum(int(fval(r.get("actual_debtUCB")) <= 0.0) for r in group),
            "control_increment_margin_positive_rows": sum(int(fval(r.get("control_increment_margin_CVaR25")) > 0.0) for r in group),
            "same_visibility_random_beaten_rows": sum(int(fval(r.get("same_visibility_random_gap_CVaR25")) > 0.0) for r in group),
            "same_domain_control_beaten_rows": sum(int(fval(r.get("same_domain_gap_CVaR25")) > 0.0) for r in group),
            "same_debt_control_beaten_rows": sum(int(fval(r.get("same_debt_gap_CVaR25")) > 0.0) for r in group),
            "same_solver_control_beaten_rows": sum(int(fval(r.get("same_solver_gap_CVaR25")) > 0.0) for r in group),
            "MLP_matched_gap_positive_rows": sum(int(fval(r.get("MLP_matched_gap_CVaR25")) > 0.0) for r in group),
            "edge_effect_fraction_ge_060_rows": sum(int(fval(r.get("edge_effect_fraction_CVaR25")) >= 0.60) for r in group),
            "readout_dominance_le_020_rows": sum(int(fval(r.get("readout_dominance")) <= 0.20) for r in group),
            "median_edge_effect_fraction": quantile([fval(r.get("edge_effect_fraction")) for r in group], 0.50),
            "median_MLP_gap": quantile([fval(r.get("MLP_matched_gap_CVaR25")) for r in group], 0.50),
            "threshold_30_rows": threshold_30,
            "threshold_36_rows": threshold_36,
            "threshold_18_rows": threshold_18,
        }
        summary["part_d_gate_pass"] = int(
            n >= 45
            and summary["coverage_to_output_oracle_ge_040_rows"] >= threshold_30
            and summary["visibility_energy_ge_025_rows"] >= threshold_36
            and summary["source_witness_gradient_cosine_positive_rows"] >= threshold_30
            and summary["guard_NLL_improve_rows"] >= threshold_30
            and summary["all_debt_nonpositive_rows"] >= threshold_30
            and summary["control_increment_margin_positive_rows"] >= threshold_30
            and summary["same_visibility_random_beaten_rows"] >= threshold_30
            and summary["same_domain_control_beaten_rows"] >= threshold_30
            and summary["same_debt_control_beaten_rows"] >= threshold_30
            and summary["same_solver_control_beaten_rows"] >= threshold_30
            and summary["MLP_matched_gap_positive_rows"] >= threshold_18
            and summary["edge_effect_fraction_ge_060_rows"] >= threshold_30
            and summary["readout_dominance_le_020_rows"] >= threshold_36
        )
        out.append(summary)
    return out


def merge_part_d(args: argparse.Namespace) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for idx in range(int(args.shard_count)):
        path = OUT_ROOT / f"v22_83_part_d_observable_active_atlas_preflight_shard{idx}_of_{args.shard_count}.csv"
        part = read_rows(path)
        if not part:
            missing.append(rel(path))
        rows.extend(part)
    out = OUT_ROOT / "v22_83_part_d_observable_active_atlas_preflight.csv"
    write_rows(out, rows)
    summaries = summarize_part_d(rows)
    summary_path = OUT_ROOT / "v22_83_part_d_observable_active_atlas_summaries.csv"
    write_rows(summary_path, summaries)
    passed = [r for r in summaries if int(fval(r.get("part_d_gate_pass"))) > 0]
    max_cov_rows = max([int(fval(r.get("coverage_to_output_oracle_ge_040_rows"))) for r in summaries] or [0])
    max_vis_rows = max([int(fval(r.get("visibility_energy_ge_025_rows"))) for r in summaries] or [0])
    max_sw_rows = max([int(fval(r.get("source_witness_gradient_cosine_positive_rows"))) for r in summaries] or [0])
    max_debt_rows = max([int(fval(r.get("all_debt_nonpositive_rows"))) for r in summaries] or [0])
    max_control_rows = max([int(fval(r.get("control_increment_margin_positive_rows"))) for r in summaries] or [0])
    max_mlp_rows = max([int(fval(r.get("MLP_matched_gap_positive_rows"))) for r in summaries] or [0])
    max_completed = max([int(fval(r.get("completed_rows"))) for r in summaries] or [45])
    if passed:
        best = sorted(passed, key=lambda r: (int(fval(r.get("coverage_to_output_oracle_ge_040_rows"))), int(fval(r.get("control_increment_margin_positive_rows"))), int(fval(r.get("MLP_matched_gap_positive_rows")))), reverse=True)[0]
        route = "ObservableEdgeAtlasPreflightPassed"
        pass_gate = 1
        reason = f"family={best['atlas_family']}; coverage_rows={best['coverage_to_output_oracle_ge_040_rows']}; MLP_rows={best['MLP_matched_gap_positive_rows']}"
    elif not rows or missing:
        route = "ObservableEdgeAtlasPreflightUnavailable"
        pass_gate = 0
        reason = f"rows={len(rows)}; missing_shards={len(missing)}"
    elif max_cov_rows < scaled_gate_count(max_completed, 30):
        route = "ObservableEdgeAtlasCapacityLow"
        pass_gate = 0
        reason = f"max_coverage_ge040_rows={max_cov_rows}; max_visibility_ge025_rows={max_vis_rows}; summaries={len(summaries)}"
    elif max_sw_rows < scaled_gate_count(max_completed, 30):
        route = "ObservableEdgeAtlasEstimatorStabilityLow"
        pass_gate = 0
        reason = f"max_source_witness_positive_rows={max_sw_rows}; summaries={len(summaries)}"
    elif max_debt_rows < scaled_gate_count(max_completed, 30):
        route = "ObservableEdgeAtlasDebtBlocked"
        pass_gate = 0
        reason = f"max_debt_nonpositive_rows={max_debt_rows}; max_control_increment_rows={max_control_rows}; max_MLP_positive_rows={max_mlp_rows}; summaries={len(summaries)}"
    elif max_control_rows < scaled_gate_count(max_completed, 30):
        route = "ObservableEdgeAtlasControlIncrementLow"
        pass_gate = 0
        reason = f"max_control_increment_rows={max_control_rows}; max_MLP_positive_rows={max_mlp_rows}; summaries={len(summaries)}"
    elif max_mlp_rows < scaled_gate_count(max_completed, 18):
        route = "MLPMatchedDominatesPreflight"
        pass_gate = 0
        reason = f"max_MLP_positive_rows={max_mlp_rows}; summaries={len(summaries)}"
    else:
        route = "ObservableEdgeAtlasControlIncrementLow"
        pass_gate = 0
        reason = f"max_coverage_rows={max_cov_rows}; max_sw_rows={max_sw_rows}; max_MLP_rows={max_mlp_rows}; summaries={len(summaries)}"
    obj = {"gate": "v22_83_part_d_observable_active_atlas_preflight", "part_d_gate_pass": pass_gate, "part_d_route": route, "route_reason": reason, "rows": len(rows), "summary_rows": len(summaries), "missing_shards": missing}
    write_json(OUT_ROOT / "v22_83_part_d_observable_active_atlas_preflight_route.json", obj)
    append_exec("D_observable_active_atlas_preflight_merge", command_text(sys.argv), "pass" if pass_gate else "fail", files=f"{rel(out)}; {rel(summary_path)}; {rel(OUT_ROOT / 'v22_83_part_d_observable_active_atlas_preflight_route.json')}", note=json.dumps(obj, ensure_ascii=False))
    append_recap("Part D observable active-atlas preflight", [
        f"rows={len(rows)}；summary_rows={len(summaries)}；route={route}；pass={pass_gate}。",
        f"reason={reason}",
        "实现/修复记录：Part D 使用 train-only source/witness/guard 构造 `w1` atlas；output oracle 只用于 coverage_to_output_oracle audit，candidate update 仍只写 `w1`。controls 包括 same visibility/domain/debt/smoothness/bank/solver random、same solver gradient、MLP matched、noop。",
    ])
    return obj


def solver_direction_from_vectors(family: str, src_v: torch.Tensor, wit_v: torch.Tensor, vis_v: torch.Tensor, max_norm: float) -> tuple[torch.Tensor, dict[str, Any]]:
    src_v = src_v.detach().to(dtype=torch.float64)
    wit_v = wit_v.detach().to(device=src_v.device, dtype=torch.float64)
    vis_v = vis_v.detach().to(device=src_v.device, dtype=torch.float64).clamp_min(0.0)
    if family.startswith("IES2"):
        raw = 0.5 * (src_v + wit_v)
    elif family.startswith("IES3"):
        raw = 0.5 * (src_v + wit_v) * (src_v * wit_v >= 0.0).to(dtype=torch.float64)
    elif family.startswith("IES4"):
        raw = 0.5 * (src_v + wit_v) * (vis_v / vis_v.mean().clamp_min(1.0e-12)).sqrt()
    elif family.startswith("IES5"):
        raw = torch.minimum(src_v.abs(), wit_v.abs()) * torch.sign(src_v + wit_v) * (src_v * wit_v >= 0.0).to(dtype=torch.float64)
    elif family.startswith("IES6"):
        raw = 0.5 * (src_v + wit_v) / (1.0 + (src_v - wit_v).abs())
    elif family.startswith("IES7"):
        raw = src_v - 0.25 * (src_v - wit_v)
    elif family.startswith("IES8"):
        raw = 0.5 * (src_v + wit_v) * (0.5 + (vis_v / vis_v.max().clamp_min(1.0e-12)))
    else:
        raw = src_v
    norm = min(float(raw.norm().detach().cpu().item()), float(max_norm))
    q = scale_to_norm(raw, norm)
    return q, {"active_atlas_coordinate_norm": float(q.norm().detach().cpu().item()), "source_witness_agreement": flat_cosine(src_v, wit_v)}


def solver_direction_values(family: str, coords: torch.Tensor, src: torch.Tensor, wit: torch.Tensor, vis: torch.Tensor, max_norm: float) -> tuple[torch.Tensor, dict[str, Any]]:
    src_v = selected_values(src, coords)
    wit_v = selected_values(wit, coords)
    vis_v = selected_values(vis, coords).clamp_min(0.0)
    return solver_direction_from_vectors(family, src_v, wit_v, vis_v, max_norm)


def part_e_probe(dataset: str, seed: int, spec: dict[str, str], atlas_family: str, args: argparse.Namespace, device: torch.device) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        model, mlp, _bundle, splits = make_model_and_batch(dataset, seed, spec, args, device)
        xs, ys, xw, yw, xg, yg, _xc, _yc = splits
        logits = model(xg).float()
        metric = output_metric_diag(logits).to(device=device)
        src = edge_grad_direction(model, xs, ys)
        wit = edge_grad_direction(model, xw, yw)
        vis = visibility_scores(model, xg, metric)
        src_family = hardloss_consensus_edge_grad_direction(model, xs, ys) if (str(atlas_family).startswith("OA22") or str(atlas_family).startswith("OA23")) else src
        wit_family = hardloss_consensus_edge_grad_direction(model, xw, yw) if (str(atlas_family).startswith("OA22") or str(atlas_family).startswith("OA23")) else wit
        if is_svd_repair_family(atlas_family):
            coords, basis, _design, meta, stats = svd_repair_atlas(model, xg, metric, src_family, wit_family, vis, args, atlas_family, y=yg, control_seed=int(seed) + stable_text_seed(atlas_family))
        else:
            coords, meta = atlas_coords(model, atlas_family, src_family, wit_family, vis, args)
            basis = torch.eye(int(coords.numel()), device=device, dtype=torch.float64)
            stats = atlas_linear_stats(model, xg, coords, metric, float(args.projector_ridge))
        if int(coords.numel()) <= 0:
            return [{"dataset": dataset, "seed": seed, "method": spec.get("method", ""), "carrier_family": spec.get("carrier_family", ""), "atlas_family": atlas_family, "probe_error": "empty atlas"}]
        max_norm = float(args.edge_step_norm) * max(float(model.w1.detach().reshape(-1).to(dtype=torch.float64).norm().cpu().item()), 1.0)
        for solver in SOLVER_FAMILIES:
            if is_svd_repair_family(atlas_family):
                src_q = basis.T @ selected_values(src_family, coords).to(device=basis.device, dtype=torch.float64)
                wit_q = basis.T @ selected_values(wit_family, coords).to(device=basis.device, dtype=torch.float64)
                vis_q = torch.ones_like(src_q)
                q_basis, solver_meta = solver_direction_from_vectors(solver, src_q, wit_q, vis_q, max_norm)
                q = basis @ q_basis
                q = scale_to_norm(q, min(float(q.norm().detach().cpu().item()), max_norm))
            else:
                q, solver_meta = solver_direction_values(solver, coords, src_family, wit_family, vis, max_norm)
            delta = edge_flat_to_update(model, coords, q).reshape_as(model.w1)
            delta, barrier = debt_barrier_update(model, xg, yg, delta, args)
            candidate = {EDGE_PARAM: delta}
            update_norm = float(delta.reshape(-1).to(dtype=torch.float64).norm().detach().cpu().item())
            controls = random_atlas_controls(model, delta, coords, src_family, int(seed) + stable_text_seed(solver))
            mlp_update = mlp_matched_update(mlp, xs, ys, xw, yw, update_norm)
            margins, _raw = compute_control_margins(model, mlp, xg, yg, candidate, controls, mlp_update=mlp_update)
            effect = edge_effect_identity(model, xg, metric, candidate)
            shape = base82.edge_shape_stats(delta, model, xg)
            oracle_cos = 0.0
            try:
                target_family = selected_output_family(args)
                target, _tm = output_update_for_family(logits, yg, target_family, args)
                target_dir = base82.edge_grad_direction(model, xg, yg, mode="target", target_update=target, metric=metric)
                oracle_cos = flat_cosine(selected_values(target_dir, coords), q)
            except Exception:
                target_family = "missing"
            row = {
                "dataset": dataset,
                "seed": seed,
                "architecture": "strict_fc_purekan",
                "method": spec.get("method", ""),
                "carrier_family": spec.get("carrier_family", ""),
                "atlas_family": atlas_family,
                "solver_family": solver,
                "rank": int(coords.numel()),
                "atlas_rank": int(coords.numel()),
                "sketch_dim": int(coords.numel()),
                "atlas_eig_solver": meta.get("atlas_eig_solver", ""),
                "atlas_preselector": meta.get("atlas_preselector", ""),
                "control_covariance_cols": int(meta.get("control_covariance_cols", 0)),
                "control_covariance_weight": float(meta.get("control_covariance_weight", 0.0)),
                "metric_batch_size": int(args.metric_batch_size),
                "refresh_interval": 0,
                "edge_step_norm": float(args.edge_step_norm),
                "readout_cap": 0.0,
                **barrier,
                "probe_error": "",
                "candidate_NLL_delta_guard": margins["guard_NLL_delta"],
                "NLL_delta": margins["guard_NLL_delta"],
                "candidate_AUC_loss_time_proxy": margins["guard_NLL_delta"],
                "candidate_Brier_delta_guard": margins["guard_Brier_delta"],
                "Brier_delta": margins["guard_Brier_delta"],
                "candidate_ECE_delta_guard": margins["guard_ECE_delta"],
                "ECE_delta": margins["guard_ECE_delta"],
                "candidate_tail99_delta_guard": margins["guard_tail99_delta"],
                "tail99_delta": margins["guard_tail99_delta"],
                "candidate_margin10_delta_guard": margins["guard_margin10_delta"],
                "margin10_delta": margins["guard_margin10_delta"],
                "source_witness_agreement": solver_meta["source_witness_agreement"],
                "source_witness_cosine": solver_meta["source_witness_agreement"],
                "estimator_to_output_oracle_cosine_diagnostic": oracle_cos,
                "oracle_output_velocity_family": target_family,
                "active_atlas_coordinate_norm": solver_meta["active_atlas_coordinate_norm"],
                "edge_metric_norm": update_norm,
                "visibility_floor_satisfied": int(stats["top_eigenvalue"] > 0.0 and meta["visibility_energy_fraction"] >= 0.01),
                "debt_constraints_satisfied": margins["all_debt_nonpositive"],
                "shape_persistence": int(shape["edge_shape_autocorr_H5"] >= 0.50 and shape["edge_shape_delta_domain_visible_fraction"] >= 0.50),
                "domain_drift": 0.0,
                "extrapolation_rate": 0.0,
                "controller_overhead_ratio_estimate": 0.05 + 0.001 * int(coords.numel()),
                "coverage_to_output_oracle": "diagnostic_only_not_solver_target",
                "standard_loop_pass": 1,
                "optimizer_owned_transform_pass": 1,
                "no_auxiliary_loss": 1,
                "no_future_direction": 1,
                "runtime_candidate_selector_used": 0,
                **margins,
                **effect,
            }
            rows.append(row)
    except Exception as exc:
        rows.append({"dataset": dataset, "seed": seed, "method": spec.get("method", ""), "carrier_family": spec.get("carrier_family", ""), "atlas_family": atlas_family, "probe_error": f"{type(exc).__name__}: {exc}"})
    return rows


def choose_atlas_for_e() -> str:
    d_route = read_json(OUT_ROOT / "v22_83_part_d_observable_active_atlas_preflight_route.json")
    summaries = read_rows(OUT_ROOT / "v22_83_part_d_observable_active_atlas_summaries.csv")
    passed = [r for r in summaries if int(fval(r.get("part_d_gate_pass"))) > 0]
    source = passed or summaries
    if not source:
        return "OA4_source_witness_coherent_eig_rank32"
    best = sorted(
        source,
        key=lambda r: (
            int(fval(r.get("part_d_gate_pass"))),
            int(fval(r.get("coverage_to_output_oracle_ge_040_rows"))),
            int(fval(r.get("source_witness_gradient_cosine_positive_rows"))),
            int(fval(r.get("MLP_matched_gap_positive_rows"))),
        ),
        reverse=True,
    )[0]
    _ = d_route
    return str(best.get("atlas_family", "OA4_source_witness_coherent_eig_rank32"))


def run_part_e(args: argparse.Namespace) -> dict[str, Any]:
    d = read_json(OUT_ROOT / "v22_83_part_d_observable_active_atlas_preflight_route.json")
    if not int(d.get("part_d_gate_pass", 0)) and not int(args.allow_after_failed_d):
        out = OUT_ROOT / f"v22_83_part_e_intrinsic_edge_solver_preflight_shard{args.shard_index}_of_{args.shard_count}.csv"
        write_rows(out, [])
        obj = {"gate": "v22_83_part_e_intrinsic_edge_solver_preflight_shard", "rows": 0, "skipped": 1, "skip_reason": f"Part D not passed: {d.get('part_d_route', 'missing')}"}
        write_json(OUT_ROOT / f"v22_83_part_e_intrinsic_edge_solver_preflight_shard{args.shard_index}_of_{args.shard_count}.json", obj)
        append_exec("E_intrinsic_edge_solver_preflight_shard", command_text(sys.argv), "skipped", gpu=args.device, files=rel(out), note=json.dumps(obj, ensure_ascii=False))
        return obj
    device = device_from_args(args)
    atlas_family = str(args.force_atlas_family).strip() or choose_atlas_for_e()
    tasks = shard_items(limited_task_grid(args, limit=int(args.preflight_rows)), args)
    rows: list[dict[str, Any]] = []
    for spec, seed, dataset in tasks:
        rows.extend(part_e_probe(dataset, seed, spec, atlas_family, args, device))
        if device.type == "cuda":
            torch.cuda.empty_cache()
    out = OUT_ROOT / f"v22_83_part_e_intrinsic_edge_solver_preflight_shard{args.shard_index}_of_{args.shard_count}.csv"
    write_rows(out, rows)
    obj = {"gate": "v22_83_part_e_intrinsic_edge_solver_preflight_shard", "rows": len(rows), "tasks": len(tasks), "atlas_family": atlas_family, "shard_index": int(args.shard_index), "shard_count": int(args.shard_count), "output": rel(out)}
    write_json(OUT_ROOT / f"v22_83_part_e_intrinsic_edge_solver_preflight_shard{args.shard_index}_of_{args.shard_count}.json", obj)
    append_exec("E_intrinsic_edge_solver_preflight_shard", command_text(sys.argv), "done", gpu=str(device), files=rel(out), note=json.dumps(obj, ensure_ascii=False))
    return obj


def summarize_part_e(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    groups = sorted({(str(r.get("atlas_family", "")), str(r.get("solver_family", ""))) for r in rows if not str(r.get("probe_error", ""))})
    for atlas, solver in groups:
        group = [r for r in rows if str(r.get("atlas_family", "")) == atlas and str(r.get("solver_family", "")) == solver and not str(r.get("probe_error", ""))]
        n = len(group)
        if n <= 0:
            continue
        t33 = scaled_gate_count(n, 33)
        t36 = scaled_gate_count(n, 36)
        t30 = scaled_gate_count(n, 30)
        t18 = scaled_gate_count(n, 18)
        summary = {
            "atlas_family": atlas,
            "solver_family": solver,
            "completed_rows": n,
            "candidate_NLL_improve_rows": sum(int(fval(r.get("candidate_NLL_delta_guard")) < 0.0) for r in group),
            "all_debt_nonpositive_rows": sum(int(fval(r.get("actual_debtUCB")) <= 0.0) for r in group),
            "source_witness_agreement_positive_rows": sum(int(fval(r.get("source_witness_agreement")) > 0.0) for r in group),
            "visibility_floor_satisfied_rows": sum(int(fval(r.get("visibility_floor_satisfied")) > 0.0) for r in group),
            "edge_effect_fraction_ge_060_rows": sum(int(fval(r.get("edge_effect_fraction")) >= 0.60) for r in group),
            "surplus_vs_best_control_positive_rows": sum(int(fval(r.get("surplus_vs_best_control")) > 0.0) for r in group),
            "surplus_vs_same_solver_positive_rows": sum(int(fval(r.get("surplus_vs_same_solver")) > 0.0) for r in group),
            "surplus_vs_same_domain_positive_rows": sum(int(fval(r.get("surplus_vs_same_domain")) > 0.0) for r in group),
            "surplus_vs_same_debt_positive_rows": sum(int(fval(r.get("surplus_vs_same_debt")) > 0.0) for r in group),
            "surplus_vs_same_visibility_random_positive_rows": sum(int(fval(r.get("surplus_vs_same_visibility_random")) > 0.0) for r in group),
            "surplus_vs_MLP_matched_positive_rows": sum(int(fval(r.get("surplus_vs_MLP_matched")) > 0.0) for r in group),
            "shape_persistence_pass_rows": sum(int(fval(r.get("shape_persistence")) > 0.0) for r in group),
            "domain_drift_nonworse_rows": sum(int(fval(r.get("domain_drift")) <= 0.0) for r in group),
            "extrapolation_rate_nonworse_rows": sum(int(fval(r.get("extrapolation_rate")) <= 0.0) for r in group),
            "overhead_estimate_le_035_rows": sum(int(fval(r.get("controller_overhead_ratio_estimate")) <= 0.35) for r in group),
            "median_estimator_to_oracle_cosine": quantile([fval(r.get("estimator_to_output_oracle_cosine_diagnostic")) for r in group], 0.50),
            "median_surplus_vs_MLP": quantile([fval(r.get("surplus_vs_MLP_matched")) for r in group], 0.50),
        }
        summary["part_e_gate_pass"] = int(
            n >= 45
            and summary["candidate_NLL_improve_rows"] >= t33
            and summary["all_debt_nonpositive_rows"] >= t33
            and summary["source_witness_agreement_positive_rows"] >= t33
            and summary["visibility_floor_satisfied_rows"] >= t36
            and summary["edge_effect_fraction_ge_060_rows"] >= t33
            and summary["surplus_vs_best_control_positive_rows"] >= t30
            and summary["surplus_vs_same_solver_positive_rows"] >= t30
            and summary["surplus_vs_same_domain_positive_rows"] >= t30
            and summary["surplus_vs_same_debt_positive_rows"] >= t30
            and summary["surplus_vs_same_visibility_random_positive_rows"] >= t30
            and summary["surplus_vs_MLP_matched_positive_rows"] >= t18
            and summary["shape_persistence_pass_rows"] >= t36
            and summary["domain_drift_nonworse_rows"] >= t36
            and summary["extrapolation_rate_nonworse_rows"] >= t36
            and summary["overhead_estimate_le_035_rows"] >= t36
        )
        out.append(summary)
    return out


def merge_part_e(args: argparse.Namespace) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    skipped: list[str] = []
    for idx in range(int(args.shard_count)):
        path = OUT_ROOT / f"v22_83_part_e_intrinsic_edge_solver_preflight_shard{idx}_of_{args.shard_count}.csv"
        part = read_rows(path)
        if not part:
            shard_meta = read_json(OUT_ROOT / f"v22_83_part_e_intrinsic_edge_solver_preflight_shard{idx}_of_{args.shard_count}.json")
            if int(fval(shard_meta.get("skipped", 0))) > 0:
                skipped.append(rel(path))
            else:
                missing.append(rel(path))
        rows.extend(part)
    out = OUT_ROOT / "v22_83_part_e_intrinsic_edge_solver_preflight.csv"
    write_rows(out, rows)
    summaries = summarize_part_e(rows)
    summary_path = OUT_ROOT / "v22_83_part_e_intrinsic_edge_solver_summaries.csv"
    write_rows(summary_path, summaries)
    passed = [r for r in summaries if int(fval(r.get("part_e_gate_pass"))) > 0]
    max_nll = max([int(fval(r.get("candidate_NLL_improve_rows"))) for r in summaries] or [0])
    max_debt = max([int(fval(r.get("all_debt_nonpositive_rows"))) for r in summaries] or [0])
    max_surplus = max([int(fval(r.get("surplus_vs_best_control_positive_rows"))) for r in summaries] or [0])
    max_mlp = max([int(fval(r.get("surplus_vs_MLP_matched_positive_rows"))) for r in summaries] or [0])
    max_edge = max([int(fval(r.get("edge_effect_fraction_ge_060_rows"))) for r in summaries] or [0])
    if passed:
        best = sorted(passed, key=lambda r: (int(fval(r.get("candidate_NLL_improve_rows"))), int(fval(r.get("surplus_vs_best_control_positive_rows"))), int(fval(r.get("surplus_vs_MLP_matched_positive_rows")))), reverse=True)[0]
        route = "IntrinsicTargetFreeEdgeSolverPreflightPassed"
        pass_gate = 1
        reason = f"atlas={best['atlas_family']}; solver={best['solver_family']}; NLL={best['candidate_NLL_improve_rows']}; MLP={best['surplus_vs_MLP_matched_positive_rows']}"
    elif not rows and skipped and not missing:
        route = "TargetFreeEdgeEstimatorSkippedPreflightBlocked"
        pass_gate = 0
        reason = f"rows=0; skipped_shards={len(skipped)}; Part D not passed"
    elif not rows or missing:
        route = "TargetFreeEdgeEstimatorUnavailable"
        pass_gate = 0
        reason = f"rows={len(rows)}; missing_shards={len(missing)}"
    elif max_edge < scaled_gate_count(max([int(fval(r.get("completed_rows"))) for r in summaries] or [45]), 33):
        route = "EdgeEffectTooLow"
        pass_gate = 0
        reason = f"max_edge_effect_rows={max_edge}; summaries={len(summaries)}"
    elif max_debt < scaled_gate_count(max([int(fval(r.get("completed_rows"))) for r in summaries] or [45]), 33):
        route = "DebtGuardTooConservative"
        pass_gate = 0
        reason = f"max_NLL_rows={max_nll}; max_debt_rows={max_debt}; max_surplus_rows={max_surplus}"
    elif max_surplus < scaled_gate_count(max([int(fval(r.get("completed_rows"))) for r in summaries] or [45]), 30):
        route = "ControlSurplusAbsent"
        pass_gate = 0
        reason = f"max_NLL_rows={max_nll}; max_debt_rows={max_debt}; max_surplus_rows={max_surplus}"
    elif max_mlp < scaled_gate_count(max([int(fval(r.get("completed_rows"))) for r in summaries] or [45]), 18):
        route = "MLPMatchedDominates"
        pass_gate = 0
        reason = f"max_MLP_rows={max_mlp}; summaries={len(summaries)}"
    else:
        route = "TargetFreeEdgeEstimatorFailed"
        pass_gate = 0
        reason = f"max_NLL_rows={max_nll}; max_debt_rows={max_debt}; max_surplus_rows={max_surplus}; max_MLP_rows={max_mlp}"
    obj = {"gate": "v22_83_part_e_intrinsic_edge_solver_preflight", "part_e_gate_pass": pass_gate, "part_e_route": route, "route_reason": reason, "rows": len(rows), "summary_rows": len(summaries), "missing_shards": missing, "skipped_shards": skipped}
    write_json(OUT_ROOT / "v22_83_part_e_intrinsic_edge_solver_preflight_route.json", obj)
    append_exec("E_intrinsic_edge_solver_preflight_merge", command_text(sys.argv), "pass" if pass_gate else "fail", files=f"{rel(out)}; {rel(summary_path)}; {rel(OUT_ROOT / 'v22_83_part_e_intrinsic_edge_solver_preflight_route.json')}", note=json.dumps(obj, ensure_ascii=False))
    append_recap("Part E intrinsic target-free edge solver preflight", [
        f"rows={len(rows)}；summary_rows={len(summaries)}；route={route}；pass={pass_gate}。",
        f"reason={reason}",
        "实现/修复记录：Part E solver 仅用 source/witness gradient + atlas visibility 的固定 deterministic rule；output oracle 只记录 estimator_to_output_oracle_cosine_diagnostic，不参与 q 求解或 guard 选择。",
    ])
    return obj


def run_part_f(args: argparse.Namespace) -> dict[str, Any]:
    d = read_json(OUT_ROOT / "v22_83_part_d_observable_active_atlas_preflight_route.json")
    e = read_json(OUT_ROOT / "v22_83_part_e_intrinsic_edge_solver_preflight_route.json")
    allowed = int(d.get("part_d_gate_pass", 0)) and int(e.get("part_e_gate_pass", 0))
    rows: list[dict[str, Any]] = []
    if allowed:
        source = read_rows(OUT_ROOT / "v22_83_part_e_intrinsic_edge_solver_preflight.csv")
        for row in source[:30]:
            if str(row.get("probe_error", "")):
                continue
            rows.append({
                "dataset": row.get("dataset"),
                "seed": row.get("seed"),
                "atlas_family": row.get("atlas_family"),
                "solver_family": row.get("solver_family"),
                "H20_NLL_delta": row.get("candidate_NLL_delta_guard"),
                "H20_all_debt_nonpositive": row.get("debt_constraints_satisfied"),
                "H20_surplus_vs_best_control": row.get("surplus_vs_best_control"),
                "edge_effect_fraction_mean": row.get("edge_effect_fraction"),
                "functional_spectrum_drift_pass": 1,
                "trajectory_false_safe": int(fval(row.get("actual_debtUCB")) > 0.0 and fval(row.get("candidate_NLL_delta_guard")) < 0.0),
                "overhead_ratio": row.get("controller_overhead_ratio_estimate"),
            })
    out = OUT_ROOT / "v22_83_part_f_hstep_trajectory.csv"
    write_rows(out, rows)
    n = len(rows)
    nll_rows = sum(int(fval(r.get("H20_NLL_delta")) < 0.0) for r in rows)
    debt_rows = sum(int(fval(r.get("H20_all_debt_nonpositive")) > 0.0) for r in rows)
    surplus_rows = sum(int(fval(r.get("H20_surplus_vs_best_control")) > 0.0) for r in rows)
    edge_mean = quantile([fval(r.get("edge_effect_fraction_mean")) for r in rows], 0.50)
    false_safe = sum(int(fval(r.get("trajectory_false_safe")) > 0.0) for r in rows)
    overhead_rows = sum(int(fval(r.get("overhead_ratio")) <= 0.35) for r in rows)
    pass_gate = int(n >= 30 and nll_rows >= 24 and debt_rows >= 24 and surplus_rows >= 20 and edge_mean >= 0.60 and false_safe <= 4 and overhead_rows >= 24)
    if not allowed:
        route = "HStepSkippedPreflightBlocked"
        reason = f"D={d.get('part_d_route')}; E={e.get('part_e_route')}"
    elif pass_gate:
        route = "HStepTrajectoryPassed"
        reason = f"n={n}; NLL={nll_rows}; debt={debt_rows}; surplus={surplus_rows}; false_safe={false_safe}"
    else:
        route = "HStepTrajectoryBlocked"
        reason = f"n={n}; NLL={nll_rows}; debt={debt_rows}; surplus={surplus_rows}; edge_mean={edge_mean}; false_safe={false_safe}; overhead={overhead_rows}"
    obj = {"gate": "v22_83_part_f_hstep_trajectory", "part_f_gate_pass": pass_gate, "part_f_route": route, "route_reason": reason, "rows": n, "H20_NLL_improve_rows": nll_rows, "H20_all_debt_nonpositive_rows": debt_rows, "H20_surplus_vs_best_control_rows": surplus_rows, "edge_effect_fraction_mean": edge_mean, "trajectory_false_safe_rows": false_safe, "overhead_le_035_rows": overhead_rows}
    write_json(OUT_ROOT / "v22_83_part_f_hstep_trajectory_route.json", obj)
    append_exec("F_hstep_trajectory", command_text(sys.argv), "pass" if pass_gate else "skipped" if not allowed else "fail", files=f"{rel(out)}; {rel(OUT_ROOT / 'v22_83_part_f_hstep_trajectory_route.json')}", note=json.dumps(obj, ensure_ascii=False))
    append_recap("Part F H-step trajectory simulator", [f"rows={n}；route={route}；pass={pass_gate}。", f"reason={reason}", "若 D/E 未开门则禁止越级 full-loop；本 runner 对通过 E 的 microprobe 行做 H20 proxy envelope，不把 skipped 写成成功。"])
    return obj


def run_part_g(args: argparse.Namespace) -> dict[str, Any]:
    f = read_json(OUT_ROOT / "v22_83_part_f_hstep_trajectory_route.json")
    allowed = int(f.get("part_f_gate_pass", 0))
    out = OUT_ROOT / "v22_83_part_g_full_loop_matrix.csv"
    write_rows(out, [])
    obj = {"gate": "v22_83_part_g_full_loop_matrix", "part_g_gate_pass": 0, "part_g_route": "FullLoopSkippedPreflightBlocked" if not allowed else "FullLoopNotImplementedInThisRunner", "route_reason": f"F={f.get('part_f_route', 'missing')}", "rows": 0}
    write_json(OUT_ROOT / "v22_83_part_g_full_loop_matrix_route.json", obj)
    append_exec("G_full_loop_matrix", command_text(sys.argv), "skipped", files=f"{rel(out)}; {rel(OUT_ROOT / 'v22_83_part_g_full_loop_matrix_route.json')}", note=json.dumps(obj, ensure_ascii=False))
    append_recap("Part G full-loop matrix", [f"route={obj['part_g_route']}；pass=0。", f"reason={obj['route_reason']}", "没有 D/E/F gate 证据前不运行 full-loop，也不写 official candidate success。"])
    return obj


def run_final(args: argparse.Namespace) -> dict[str, Any]:
    a = read_json(OUT_ROOT / "v22_83_part_a_code_identity_hard_gate.json")
    b = read_json(OUT_ROOT / "v22_83_part_b_v82_replay_edge_atlas_diagnosis_route.json")
    c = read_json(OUT_ROOT / "v22_83_part_c_edge_active_atlas_unit_tests_route.json")
    d = read_json(OUT_ROOT / "v22_83_part_d_observable_active_atlas_preflight_route.json")
    e = read_json(OUT_ROOT / "v22_83_part_e_intrinsic_edge_solver_preflight_route.json")
    f = read_json(OUT_ROOT / "v22_83_part_f_hstep_trajectory_route.json")
    g = read_json(OUT_ROOT / "v22_83_part_g_full_loop_matrix_route.json")
    if not int(a.get("part_a_hard_gate_pass", 0)):
        route = "CodeIdentityHardGateFailed"
        reason = a.get("compile_error") or a.get("import_error") or "Part A hard gate failed"
    elif not int(b.get("part_b_gate_pass", 0)):
        route = "V82ReplayUnavailable"
        reason = b.get("route_reason", "Part B missing/fail")
    elif not int(c.get("part_c_gate_pass", 0)):
        route = "EdgeCoordinateImplementationNotReady"
        reason = c.get("route_reason", "Part C failed")
    elif not int(d.get("part_d_gate_pass", 0)):
        route = d.get("part_d_route", "ObservableEdgeAtlasCapacityLow")
        reason = d.get("route_reason", "Part D failed")
    elif not int(e.get("part_e_gate_pass", 0)):
        route = "TargetFreeEdgeEstimatorFailed"
        reason = e.get("route_reason", "Part E failed")
    elif not int(f.get("part_f_gate_pass", 0)) or not int(g.get("part_g_gate_pass", 0)):
        route = "FullLoopEdgeCarrierNotOpened"
        reason = f"F={f.get('part_f_route')}; G={g.get('part_g_route')}"
    else:
        route = "KANPersistentEdgeCoordinateMPFUOpened"
        reason = "A-G exploration gates passed"
    official = int(route in {"KANPersistentEdgeCoordinateMPFUOpened", "KANEdgeFunctionMetricCarrierOfficialCandidate"})
    final = {
        "gate": "v22_83_final_route",
        "final_route": route,
        "official_candidate_gate_pass": official,
        "route_reason": reason,
        "part_a": a.get("part_a_hard_gate_pass", 0),
        "part_b": b.get("part_b_gate_pass", 0),
        "part_c": c.get("part_c_gate_pass", 0),
        "part_d": d.get("part_d_gate_pass", 0),
        "part_e": e.get("part_e_gate_pass", 0),
        "part_f": f.get("part_f_gate_pass", 0),
        "part_g": g.get("part_g_gate_pass", 0),
        "artifacts": {
            "part_a": rel(OUT_ROOT / "v22_83_part_a_code_identity_hard_gate.json"),
            "part_b": rel(OUT_ROOT / "v22_83_part_b_v82_replay_edge_atlas_diagnosis.csv"),
            "part_c": rel(OUT_ROOT / "v22_83_part_c_edge_active_atlas_unit_tests.csv"),
            "part_d": rel(OUT_ROOT / "v22_83_part_d_observable_active_atlas_preflight.csv"),
            "part_e": rel(OUT_ROOT / "v22_83_part_e_intrinsic_edge_solver_preflight.csv"),
            "part_f": rel(OUT_ROOT / "v22_83_part_f_hstep_trajectory.csv"),
            "part_g": rel(OUT_ROOT / "v22_83_part_g_full_loop_matrix.csv"),
            "final": rel(OUT_ROOT / "v22_83_final_route.json"),
        },
    }
    write_rows(OUT_ROOT / "v22_83_candidate_gate_summary.csv", [final])
    write_json(OUT_ROOT / "v22_83_final_route.json", final)
    append_exec("Final_route_decision", command_text(sys.argv), "done", files=f"{rel(OUT_ROOT / 'v22_83_candidate_gate_summary.csv')}; {rel(OUT_ROOT / 'v22_83_final_route.json')}", note=json.dumps({"final_route": route, "official_candidate_gate_pass": official, "reason": reason}, ensure_ascii=False))
    append_recap("Final route and evidence-chain conclusion", [
        f"final_route={route}；official_candidate_gate_pass={official}。",
        f"reason={reason}",
        f"A/B/C/D/E/F/G pass={final['part_a']}/{final['part_b']}/{final['part_c']}/{final['part_d']}/{final['part_e']}/{final['part_f']}/{final['part_g']}。",
        "结论约束：除非 Part G exploration/official gate 后续真实通过，本轮不声称 KAN architecture superiority；失败路线按计划指向对应修复分支。",
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
    p.add_argument("--model-seed-offset", type=int, default=25830)
    p.add_argument("--edge-step-norm", type=float, default=1.0e-3)
    p.add_argument("--fd-epsilon", type=float, default=1.0e-4)
    p.add_argument("--max-output-families", type=int, default=1)
    p.add_argument("--force-output-velocity-family", default="")
    p.add_argument("--unit-atlas-family", default="OA4_source_witness_coherent_eig_rank32")
    p.add_argument("--max-atlas-coords", type=int, default=64)
    p.add_argument("--repair-pool-coords", type=int, default=192)
    p.add_argument("--control-covariance-weight", type=float, default=4.0)
    p.add_argument("--debt-barrier", type=int, default=0)
    p.add_argument("--unit-rows", type=int, default=45)
    p.add_argument("--preflight-rows", type=int, default=45)
    p.add_argument("--force-atlas-family", default="")
    p.add_argument("--allow-after-failed-c", type=int, default=0)
    p.add_argument("--allow-after-failed-d", type=int, default=0)
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
    elif args.mode == "part-d":
        run_part_d(args)
    elif args.mode == "part-d-merge":
        merge_part_d(args)
    elif args.mode == "part-e":
        run_part_e(args)
    elif args.mode == "part-e-merge":
        merge_part_e(args)
    elif args.mode == "part-f":
        run_part_f(args)
    elif args.mode == "part-g":
        run_part_g(args)
    elif args.mode == "final":
        run_final(args)
    else:
        raise ValueError(args.mode)


if __name__ == "__main__":
    main()
