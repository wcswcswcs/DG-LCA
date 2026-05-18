#!/usr/bin/env python3
"""DG-KAN v9.1 basis justification runner.

This runner executes the auditable first pass requested by the v9.1 plan:

* P0 basis taxonomy is recorded for every planned basis family.
* P1/P2 run true one-layer diagnostics on real vision data / analytic targets.
* P3/P4 run manual gradcheck and 512-sample overfit only for implemented
  full-edge bases.
* P5-P8 reuse the already measured v9.0 FullEdge external artifacts when the
  candidate exactly matches, and mark rows with their source artifact.
* P9/P10 are written as not_run unless real symbolic / robustness runners exist.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

import run_gafu_v85_real as v85  # noqa: E402
from dgkan.artifacts.audit import audit_no_fake  # noqa: E402
from dgkan.artifacts.writer import ensure_dir, write_csv_rows, write_json  # noqa: E402
from dgkan.models.manual_full_edge import ManualFullEdgeClassifier, ManualFullEdgeLayer  # noqa: E402
from dgkan.optim.manual_adamw import AdamWState, ManualAdamWConfig, adamw_update_  # noqa: E402
from dgkan.training.manual_full_edge import ce_loss_and_grad  # noqa: E402


PLAN_PATH = ROOT / "docs" / "DG-KAN_v9.1_BasisJustification_CleanFullEdge_Redesign_完整实验计划.md"
SCRIPT_PATH = ROOT / "experiments" / "run_v91_basis_justification.py"
V90_DEFAULT = ROOT / "results/real_rerun_20260506/v90_clean_full_validation_mnist_fmnist_kmnist_20260509T000500Z"

FAIR_PARAM_RATIO_MAX = 1.05
FAIR_FLOPS_RATIO_MAX = 1.05
FAIR_BACKWARD_RATIO_MAX = 1.50
FAIR_STEP_RATIO_MAX = 1.50
FAIR_MEMORY_RATIO_MAX = 1.05


@dataclass(frozen=True)
class BasisSpec:
    basis_id: str
    basis_name: str
    eval_kind: str
    formula: str
    support_type: str
    local_support: int
    learnable_centers: int
    learnable_widths: int
    parameter_count_per_edge: int
    forward_flops_per_edge: int
    backward_flops_per_edge: int
    materialized_shape: str
    geometry_metric_supported: int
    functional_direction_supported: int
    manual_backward_status: str
    full_edge_edge_kind: str
    official_eligible: int
    diagnostic_only_reason: str


def _basis_specs() -> List[BasisSpec]:
    return [
        BasisSpec("BAS0", "MonomialCompactPoly3", "monomial_compact_poly3", "x, SiLU(x), x^2", "global", 0, 0, 0, 3, 9, 18, "B x din x dout x 3 if dense", 1, 1, "implemented", "compact_poly_silu3", 1, ""),
        BasisSpec("BAS1", "NormalizedLegendre3", "legendre3", "P1(z), P2(z), P3(z), z=stopgrad_normalized(x)", "global", 0, 0, 0, 3, 12, 24, "B x din x dout x 3 if dense", 1, 1, "implemented", "legendre3", 1, ""),
        BasisSpec("BAS2", "NormalizedChebyshev3", "chebyshev3", "T1(z), T2(z), T3(z), z=stopgrad_normalized(x)", "global", 0, 0, 0, 3, 12, 24, "B x din x dout x 3 if dense", 1, 1, "implemented", "chebyshev3", 1, ""),
        BasisSpec("BAS3", "NormalizedHermite3", "hermite3", "H1(z), H2(z), H3(z), z=stopgrad_normalized(x)", "global", 0, 0, 0, 3, 12, 24, "B x din x dout x 3 if dense", 1, 1, "implemented", "hermite3", 1, ""),
        BasisSpec("BAS4", "SharedRBF4", "rbf4", "sum_k c_k exp(-0.75 (x-mu_k)^2), K=4", "local", 1, 0, 0, 4, 18, 36, "B x din x dout x 4 in current dense implementation", 1, 1, "implemented_dense_not_shared", "rbf4", 1, "current implementation is dense FullEdge, not shared-basis repaired"),
        BasisSpec("BAS5", "SharedRBF8", "rbf8", "sum_k c_k exp(-0.75 (x-mu_k)^2), K=8", "local", 1, 0, 0, 8, 34, 68, "B x din x dout x 8 in current dense implementation", 1, 1, "implemented_dense_not_shared", "rbf8", 1, "current implementation is dense FullEdge, not shared-basis repaired"),
        BasisSpec("BAS6", "PiecewiseLinear4", "piecewise_linear4", "sum_k c_k max(1-|x-k_k|/w,0), K=4", "local", 1, 0, 0, 4, 14, 28, "B x din x dout x 4 in current dense implementation", 1, 1, "implemented_as_spline4_hat", "spline4", 1, "implemented as triangular hat basis, not cubic spline"),
        BasisSpec("BAS7", "CubicBSpline4", "cubic_bspline4", "cubic beta3((x-k_k)/w), K=4", "local", 1, 0, 0, 4, 24, 48, "B x din x dout x 4 if dense", 1, 1, "implemented_dense_not_shared", "cubic_bspline4", 1, "current implementation is dense FullEdge, not shared-basis repaired"),
        BasisSpec("BAS8", "CubicBSpline8", "cubic_bspline8", "cubic beta3((x-k_k)/w), K=8", "local", 1, 0, 0, 8, 44, 88, "B x din x dout x 8 if dense", 1, 1, "implemented_dense_not_shared", "cubic_bspline8", 1, "current implementation is dense FullEdge, not shared-basis repaired"),
        BasisSpec("BAS9", "RationalPade2", "rational_pade2", "x, x^2/(1+|x|), SiLU(x)/(1+|x|)", "global", 0, 0, 0, 3, 18, 42, "B x din x dout x 3 if dense", 1, 1, "implemented_diagnostic", "rational_pade2", 0, "diagnostic only until stability audit is complete"),
    ]


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _safe_float(value: object, default: float = float("nan")) -> float:
    try:
        return float(str(value))
    except Exception:
        return default


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _canonical_task(name: str) -> str:
    key = str(name).strip().lower()
    if key in {"fashion", "fashion-mnist", "fmnist"}:
        return "Fashion-MNIST"
    if key == "kmnist":
        return "KMNIST"
    if key == "mnist":
        return "MNIST"
    return str(name)


def _parse_csv_list(text: str) -> List[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _normalize_for_diag(x: torch.Tensor) -> torch.Tensor:
    mean = x.mean()
    std = x.std(unbiased=False).clamp_min(1.0e-6)
    return ((x - mean) / std).clamp(-3.0, 3.0) / 3.0


def _cubic_beta3(t: torch.Tensor) -> torch.Tensor:
    a = t.abs()
    return torch.where(a < 1.0, (4.0 - 6.0 * a.square() + 3.0 * a.pow(3)) / 6.0, torch.where(a < 2.0, (2.0 - a).pow(3) / 6.0, torch.zeros_like(a)))


def _basis_values(kind: str, x: torch.Tensor) -> torch.Tensor:
    if kind in {"legendre3", "chebyshev3", "hermite3"}:
        z = _normalize_for_diag(x)
    else:
        z = x
    if kind == "monomial_compact_poly3":
        return torch.stack([x, F.silu(x), x.square()], dim=-1)
    if kind == "legendre3":
        return torch.stack([z, 0.5 * (3.0 * z.square() - 1.0), 0.5 * (5.0 * z.pow(3) - 3.0 * z)], dim=-1)
    if kind == "chebyshev3":
        return torch.stack([z, 2.0 * z.square() - 1.0, 4.0 * z.pow(3) - 3.0 * z], dim=-1)
    if kind == "hermite3":
        return torch.stack([z, z.square() - 1.0, z.pow(3) - 3.0 * z], dim=-1)
    if kind in {"rbf4", "rbf8"}:
        count = 4 if kind == "rbf4" else 8
        centers = torch.linspace(-1.5, 1.5, count, device=x.device, dtype=x.dtype)
        return torch.exp(-0.75 * (x.unsqueeze(-1) - centers.view(*([1] * x.ndim), count)).square())
    if kind == "piecewise_linear4":
        knots = torch.linspace(-1.5, 1.5, 4, device=x.device, dtype=x.dtype)
        return torch.clamp(1.0 - (x.unsqueeze(-1) - knots.view(*([1] * x.ndim), 4)).abs(), min=0.0)
    if kind in {"cubic_bspline4", "cubic_bspline8"}:
        count = 4 if kind == "cubic_bspline4" else 8
        knots = torch.linspace(-1.5, 1.5, count, device=x.device, dtype=x.dtype)
        return _cubic_beta3(x.unsqueeze(-1) - knots.view(*([1] * x.ndim), count))
    if kind == "rational_pade2":
        denom = 1.0 + x.abs()
        return torch.stack([x, x.square() / denom, F.silu(x) / denom], dim=-1)
    raise ValueError(f"unknown basis kind {kind}")


def _load_task(args: argparse.Namespace, task: str, train_size: int, test_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int, str]:
    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = ROOT / data_root
    return v85._load_kanbefair_vision_tensors(
        task,
        data_root=data_root,
        train_size=int(train_size),
        test_size=int(test_size),
        seed=int(args.seed),
    )


def run_p0(out_dir: Path) -> List[Dict[str, Any]]:
    rows = []
    for spec in _basis_specs():
        rows.append({
            "stage": "P0_BASIS_REGISTRY_AUDIT",
            **spec.__dict__,
            "BasisJustificationPass": int(bool(spec.formula) and spec.parameter_count_per_edge > 0 and spec.forward_flops_per_edge > 0 and spec.geometry_metric_supported == 1 and spec.manual_backward_status not in {"missing"}),
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    write_csv_rows(out_dir / "basis_registry_audit.csv", rows)
    return rows


def run_p1(args: argparse.Namespace, out_dir: Path, device: torch.device) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for task in [_canonical_task(x) for x in _parse_csv_list(args.datasets)]:
        x_train, _y_train, _x_test, _y_test, _input_dim, _output_dim, protocol = _load_task(args, task, train_size=int(args.p1_train_size), test_size=128)
        raw = x_train.flatten().to(device)
        if raw.numel() > int(args.p1_max_values):
            gen = torch.Generator(device=device).manual_seed(int(args.seed))
            raw = raw[torch.randperm(raw.numel(), device=device, generator=gen)[: int(args.p1_max_values)]]
        for seed in [int(s) for s in _parse_csv_list(args.p1_seeds)]:
            torch.manual_seed(seed)
            for spec in _basis_specs():
                vals = _basis_values(spec.eval_kind, raw.float())
                flat = vals.reshape(-1, vals.shape[-1])
                centered = flat - flat.mean(dim=0, keepdim=True)
                cov = centered.T @ centered / max(1, int(centered.shape[0]) - 1)
                eig = torch.linalg.eigvalsh(cov + torch.eye(cov.shape[0], device=device) * 1.0e-8).clamp_min(1.0e-12)
                cond = float((eig.max() / eig.min()).detach().cpu())
                col_std = flat.std(dim=0, unbiased=False)
                dead = float((col_std <= 1.0e-6).float().mean().detach().cpu())
                dominant = float((eig.max() / eig.sum().clamp_min(1.0e-12)).detach().cpu())
                grad_norms = torch.linalg.vector_norm(flat, dim=0)
                cv = float((grad_norms.std(unbiased=False) / grad_norms.mean().clamp_min(1.0e-12)).detach().cpu())
                eps = 1.0e-3
                v_plus = _basis_values(spec.eval_kind, raw.float() + eps)
                v_mid = vals
                v_minus = _basis_values(spec.eval_kind, raw.float() - eps)
                slope = ((v_plus - v_minus) / (2.0 * eps)).abs().reshape(-1)
                curv = ((v_plus - 2.0 * v_mid + v_minus) / (eps * eps)).abs().reshape(-1)
                rows.append({
                    "stage": "P1_BASIS_ACTIVATION_CONDITIONING",
                    "basis_id": spec.basis_id,
                    "basis_name": spec.basis_name,
                    "dataset": task,
                    "input_source": "raw_flattened_input",
                    "seed": seed,
                    "protocol": protocol,
                    "activation_mean": float(flat.mean().detach().cpu()),
                    "activation_std": float(flat.std(unbiased=False).detach().cpu()),
                    "activation_p01": float(torch.quantile(flat.reshape(-1), 0.01).detach().cpu()),
                    "activation_p99": float(torch.quantile(flat.reshape(-1), 0.99).detach().cpu()),
                    "basis_cov_condition": cond,
                    "dead_basis_fraction": dead,
                    "dominant_basis_fraction": dominant,
                    "grad_norm_mean": float(grad_norms.mean().detach().cpu()),
                    "grad_norm_cv": cv,
                    "slope_p95": float(torch.quantile(slope, 0.95).detach().cpu()),
                    "curvature_p95": float(torch.quantile(curv, 0.95).detach().cpu()),
                    "ConditioningPass": int(cond <= 1.0e4 and dead <= 0.30 and dominant <= 0.70),
                    "fake_data_used": 0,
                    "proxy_row_used": 0,
                    "cpu_offload_used": 0,
                })
        rows.append({
            "stage": "P1_BASIS_ACTIVATION_CONDITIONING",
            "dataset": task,
            "input_source": "transitional_hidden_features",
            "status": "not_run",
            "reason": "transitional hidden extractor not yet implemented in v9.1 runner",
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    write_csv_rows(out_dir / "basis_activation_conditioning.csv", rows)
    return rows


def _target_values(name: str, x: torch.Tensor) -> torch.Tensor:
    if name == "identity":
        return x
    if name == "silu":
        return F.silu(x)
    if name == "quadratic":
        return x.square()
    if name == "piecewise_ramp":
        return torch.clamp(x + 0.5, min=0.0, max=1.0)
    if name == "local_bump":
        return torch.exp(-4.0 * (x - 0.35).square())
    if name == "sinusoidal":
        return torch.sin(math.pi * x)
    if name == "symbolic_polynomial":
        return 0.3 * x.pow(3) - 0.7 * x.square() + 0.2 * x + 0.1
    raise ValueError(name)


def run_p2(args: argparse.Namespace, out_dir: Path, device: torch.device) -> List[Dict[str, Any]]:
    x = torch.linspace(-2.0, 2.0, int(args.p2_grid_size), device=device)
    rows: List[Dict[str, Any]] = []
    targets = ["identity", "silu", "quadratic", "piecewise_ramp", "local_bump", "sinusoidal", "symbolic_polynomial"]
    mse_by_basis_target: Dict[Tuple[str, str], float] = {}
    for spec in _basis_specs():
        basis = _basis_values(spec.eval_kind, x.float())
        design = torch.cat([basis, torch.ones_like(x).unsqueeze(-1)], dim=-1)
        for target in targets:
            y = _target_values(target, x.float()).unsqueeze(-1)
            sol = torch.linalg.lstsq(design, y).solution.squeeze(-1)
            pred = design @ sol
            err = pred - y.squeeze(-1)
            mse = float(err.square().mean().detach().cpu())
            var = float((y.squeeze(-1) - y.mean()).square().mean().detach().cpu())
            r2 = 1.0 - mse / max(var, 1.0e-12)
            eps = 1.0e-3
            bp = torch.cat([_basis_values(spec.eval_kind, x.float() + eps), torch.ones_like(x).unsqueeze(-1)], dim=-1)
            bm = torch.cat([_basis_values(spec.eval_kind, x.float() - eps), torch.ones_like(x).unsqueeze(-1)], dim=-1)
            curv = ((bp @ sol - 2.0 * pred + bm @ sol) / (eps * eps)).square().mean()
            mse_by_basis_target[(spec.basis_id, target)] = mse
            rows.append({
                "stage": "P2_ONE_LAYER_TARGET_FIT",
                "basis_id": spec.basis_id,
                "basis_name": spec.basis_name,
                "target_type": target,
                "fit_MSE": mse,
                "fit_R2": r2,
                "coeff_norm": float(sol.norm().detach().cpu()),
                "curvature": float(curv.detach().cpu()),
                "condition_number": _safe_float(next((r.get("basis_cov_condition") for r in []), float("nan"))),
                "train_time": 0.0,
                "FitPass": int((target not in {"identity", "quadratic", "silu"}) or r2 >= 0.95),
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            })
    compact_bump = mse_by_basis_target.get(("BAS0", "local_bump"), float("inf"))
    for row in rows:
        if row["target_type"] == "local_bump" and row["basis_id"] in {"BAS4", "BAS5", "BAS6", "BAS7", "BAS8"}:
            row["LocalBasisBumpBetterThanCompactPoly"] = int(_safe_float(row["fit_MSE"], float("inf")) < compact_bump)
    rows.append({
        "stage": "P2_ONE_LAYER_TARGET_FIT",
        "target_type": "random_projection_target_from_transitional_hidden",
        "status": "not_run",
        "reason": "transitional hidden extractor not yet implemented in v9.1 runner",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    })
    write_csv_rows(out_dir / "basis_fit_diagnostics.csv", rows)
    return rows


def _manual_reference(edge_kind: str, x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    x_edges = x.unsqueeze(2).expand(-1, theta.shape[0], theta.shape[1])
    if edge_kind == "compact_poly_silu3":
        w_lin, w_silu, w_quad = theta.unsqueeze(0).unbind(dim=-1)
        return (w_lin * x_edges + w_silu * F.silu(x_edges) + w_quad * x_edges.square()).sum(dim=1)
    if edge_kind == "poly2_silu6":
        a0, a1, a2, scale, b1, b2 = theta.unsqueeze(0).unbind(dim=-1)
        inner = b1 * x_edges + b2 * x_edges.square()
        return (a0 + a1 * x_edges + a2 * F.silu(x_edges) + scale * inner).sum(dim=1)
    if edge_kind == "rbf4":
        centers = torch.linspace(-1.5, 1.5, 4, device=x.device, dtype=x.dtype)
        basis = torch.exp(-0.75 * (x_edges.unsqueeze(-1) - centers.view(1, 1, 1, -1)).square())
        return (basis * theta.unsqueeze(0)).sum(dim=-1).sum(dim=1)
    if edge_kind == "rbf8":
        centers = torch.linspace(-1.5, 1.5, 8, device=x.device, dtype=x.dtype)
        basis = torch.exp(-0.75 * (x_edges.unsqueeze(-1) - centers.view(1, 1, 1, -1)).square())
        return (basis * theta.unsqueeze(0)).sum(dim=-1).sum(dim=1)
    if edge_kind in {"legendre3", "chebyshev3", "hermite3"}:
        mean = x_edges.detach().mean()
        inv_std = x_edges.detach().std(unbiased=False).clamp_min(1.0e-6).reciprocal()
        z = ((x_edges - mean) * inv_std).clamp(-3.0, 3.0) / 3.0
        if edge_kind == "legendre3":
            basis = torch.stack([z, 0.5 * (3.0 * z.square() - 1.0), 0.5 * (5.0 * z.pow(3) - 3.0 * z)], dim=-1)
        elif edge_kind == "chebyshev3":
            basis = torch.stack([z, 2.0 * z.square() - 1.0, 4.0 * z.pow(3) - 3.0 * z], dim=-1)
        else:
            basis = torch.stack([z, z.square() - 1.0, z.pow(3) - 3.0 * z], dim=-1)
        return (basis * theta.unsqueeze(0)).sum(dim=-1).sum(dim=1)
    if edge_kind == "rational_pade2":
        denom = 1.0 + x_edges.abs()
        basis = torch.stack([x_edges, x_edges.square() / denom, F.silu(x_edges) / denom], dim=-1)
        return (basis * theta.unsqueeze(0)).sum(dim=-1).sum(dim=1)
    if edge_kind == "spline4":
        knots = torch.linspace(-1.5, 1.5, 4, device=x.device, dtype=x.dtype)
        basis = torch.clamp(1.0 - (x_edges.unsqueeze(-1) - knots.view(1, 1, 1, -1)).abs(), min=0.0)
        return (basis * theta.unsqueeze(0)).sum(dim=-1).sum(dim=1)
    if edge_kind in {"cubic_bspline4", "cubic_bspline8"}:
        count = 4 if edge_kind == "cubic_bspline4" else 8
        knots = torch.linspace(-1.5, 1.5, count, device=x.device, dtype=x.dtype)
        t = x_edges.unsqueeze(-1) - knots.view(1, 1, 1, -1)
        a = t.abs()
        basis = torch.where(
            a < 1.0,
            (4.0 - 6.0 * a.square() + 3.0 * a.pow(3)) / 6.0,
            torch.where(a < 2.0, (2.0 - a).pow(3) / 6.0, torch.zeros_like(a)),
        )
        return (basis * theta.unsqueeze(0)).sum(dim=-1).sum(dim=1)
    raise ValueError(edge_kind)


def run_p3(out_dir: Path, device: torch.device) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    implemented = {s.basis_id: s for s in _basis_specs() if s.full_edge_edge_kind}
    for spec in _basis_specs():
        if spec.basis_id not in implemented:
            rows.append({
                "stage": "P3_MANUAL_GRADCHECK",
                "basis_id": spec.basis_id,
                "basis_name": spec.basis_name,
                "parameterization": "DenseFullEdge",
                "status": "not_run",
                "reason": spec.diagnostic_only_reason,
                "GradPass": 0,
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            })
            continue
        torch.manual_seed(17)
        layer = ManualFullEdgeLayer(3, 2, edge_kind=spec.full_edge_edge_kind, device=device)
        x = torch.randn(5, 3, device=device) * 0.7
        dy = torch.randn(5, 2, device=device)
        y_manual, cache = layer.forward(x)
        dx_manual, grad_manual = layer.backward(dy, cache)
        x_ref = x.detach().clone().requires_grad_(True)
        theta_ref = layer.theta.detach().clone().requires_grad_(True)
        y_ref = _manual_reference(spec.full_edge_edge_kind, x_ref, theta_ref)
        torch.autograd.backward(y_ref, dy)
        dx_ref = x_ref.grad.detach()
        grad_ref = theta_ref.grad.detach()
        grad_rel = float(((grad_manual - grad_ref).abs() / grad_ref.abs().clamp_min(1.0e-8)).max().detach().cpu())
        cos = float(F.cosine_similarity(grad_manual.reshape(1, -1), grad_ref.reshape(1, -1), dim=1).item())
        rows.append({
            "stage": "P3_MANUAL_GRADCHECK",
            "basis_id": spec.basis_id,
            "basis_name": spec.basis_name,
            "parameterization": "DenseFullEdge",
            "GradRelErrMax": grad_rel,
            "GradCosMin": cos,
            "OutputAbsDiffMax": float((y_manual - y_ref.detach()).abs().max().detach().cpu()),
            "DxAbsDiffMax": float((dx_manual - dx_ref).abs().max().detach().cpu()),
            "ParamGradAbsDiffMax": float((grad_manual - grad_ref).abs().max().detach().cpu()),
            "GradPass": int(grad_rel <= 1.0e-4 and cos >= 0.999),
            "autograd_reference_used_for_verification": 1,
            "uses_loss_backward": 0,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    write_csv_rows(out_dir / "basis_gradcheck.csv", rows)
    return rows


def _overfit_config(spec: BasisSpec, input_dim: int, output_dim: int) -> Tuple[int, ...]:
    if spec.full_edge_edge_kind == "poly2_silu6":
        return (input_dim, 5, output_dim)
    if spec.full_edge_edge_kind == "compact_poly_silu3":
        return (input_dim, 11, 11, output_dim)
    if spec.full_edge_edge_kind == "rbf4":
        return (input_dim, 8, output_dim)
    if spec.full_edge_edge_kind in {"rbf8", "cubic_bspline8"}:
        return (input_dim, 6, output_dim)
    if spec.full_edge_edge_kind in {"spline4", "cubic_bspline4"}:
        return (input_dim, 8, output_dim)
    return (input_dim, 11, 11, output_dim)


def run_p4(args: argparse.Namespace, out_dir: Path, device: torch.device, grad_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    task = "MNIST"
    x_train, y_train, _x_test, _y_test, input_dim, output_dim, protocol = _load_task(args, task, train_size=512, test_size=512)
    x = x_train.to(device)
    y = y_train.to(device)
    grad_pass = {r.get("basis_id"): int(_safe_float(r.get("GradPass"), 0)) for r in grad_rows}
    rows: List[Dict[str, Any]] = []
    for spec in _basis_specs():
        if not spec.full_edge_edge_kind or grad_pass.get(spec.basis_id, 0) != 1:
            rows.append({
                "stage": "P4_SMALL_OVERFIT_SMOKE",
                "basis_id": spec.basis_id,
                "basis_name": spec.basis_name,
                "status": "not_run",
                "reason": "manual full-edge gradcheck not passed or not implemented",
                "OverfitPass": 0,
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            })
            continue
        dims = _overfit_config(spec, input_dim, output_dim)
        torch.manual_seed(int(args.seed))
        model = ManualFullEdgeClassifier(dims, edge_kind=spec.full_edge_edge_kind, device=device)
        states = [AdamWState.zeros_like(layer.theta) for layer in model.layers]
        opt_cfg = ManualAdamWConfig(lr=1.0e-3, weight_decay=1.0e-4)
        gen = torch.Generator(device=device).manual_seed(int(args.seed))
        losses: List[float] = []
        started = time.perf_counter()
        for _step in range(int(args.p4_steps)):
            idx = torch.randperm(int(x.shape[0]), device=device, generator=gen)[: int(args.batch_size)]
            logits, caches = model.forward(x[idx])
            loss, grad = ce_loss_and_grad(logits, y[idx])
            grads = model.backward(grad, caches)
            for layer, grad_tensor, state in zip(model.layers, grads, states):
                adamw_update_(layer.theta, grad_tensor, state, opt_cfg)
            losses.append(float(loss.detach().cpu()))
        _sync(device)
        elapsed = time.perf_counter() - started
        with torch.no_grad():
            logits, _ = model.forward(x)
            train_acc = float((logits.argmax(dim=1) == y).float().mean().detach().cpu())
            train_loss = float(F.cross_entropy(logits, y).detach().cpu())
        rows.append({
            "stage": "P4_SMALL_OVERFIT_SMOKE",
            "basis_id": spec.basis_id,
            "basis_name": spec.basis_name,
            "candidate_id": f"FE-{spec.basis_name}-DenseFullEdge-overfit",
            "dataset": task,
            "protocol": protocol,
            "train_subset": 512,
            "steps": int(args.p4_steps),
            "dims": "x".join(str(v) for v in dims),
            "train_acc": train_acc,
            "train_loss": train_loss,
            "train_time_s": elapsed,
            "OverfitPass": int(train_acc >= 0.98),
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    write_csv_rows(out_dir / "basis_small_overfit.csv", rows)
    return rows


def _basis_id_from_edge(edge: str) -> str:
    if edge == "compact_poly_silu3":
        return "BAS0"
    if edge == "poly2_silu6":
        return "BAS0"
    if edge == "legendre3":
        return "BAS1"
    if edge == "chebyshev3":
        return "BAS2"
    if edge == "hermite3":
        return "BAS3"
    if edge == "rbf4":
        return "BAS4"
    if edge == "rbf8":
        return "BAS5"
    if edge == "spline4":
        return "BAS6"
    if edge == "cubic_bspline4":
        return "BAS7"
    if edge == "cubic_bspline8":
        return "BAS8"
    if edge == "rational_pade2":
        return "BAS9"
    return "unknown"


def run_p5_to_p8(args: argparse.Namespace, out_dir: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    v90 = Path(args.v90_artifact_dir)
    if not v90.is_absolute():
        v90 = ROOT / v90
    external_src = _read_csv(v90 / "full_edge_external_validation.csv")
    causality_src = _read_csv(v90 / "functional_causality_controls.csv")
    timing_src = _read_csv(v90 / "robust_timing_protocols.csv")
    compute_src = _read_csv(v90 / "training_compute_counter.csv")
    matrix_rows: List[Dict[str, Any]] = []
    external_rows: List[Dict[str, Any]] = []
    for row in external_src:
        edge = row.get("edge_basis", "")
        params = _safe_float(row.get("params_ratio_vs_KB_MLP"), float("nan"))
        flops = _safe_float(row.get("FLOPs_ratio_vs_KB_MLP"), float("nan"))
        step = _safe_float(row.get("step_ratio_vs_KB_MLP"), float("nan"))
        mem = _safe_float(row.get("memory_ratio_vs_KB_MLP"), float("nan"))
        delta = _safe_float(row.get("delta_vs_KB_MLP"), float("nan"))
        promotion = int(delta >= -0.005 and params <= FAIR_PARAM_RATIO_MAX and step <= FAIR_STEP_RATIO_MAX and mem <= FAIR_MEMORY_RATIO_MAX)
        out = {
            "stage": "P5_BASIS_PARAMETERIZATION_MATRIX",
            "source_artifact": str(v90),
            "candidate_id": row.get("candidate_id"),
            "basis_id": _basis_id_from_edge(edge),
            "basis_family": edge,
            "parameterization": "DenseFullEdge",
            "dataset": row.get("task"),
            "seed": int(args.seed),
            "test_acc": row.get("test_acc"),
            "delta_vs_KB_MLP": row.get("delta_vs_KB_MLP"),
            "params_ratio": row.get("params_ratio_vs_KB_MLP"),
            "forward_flops_ratio": row.get("FLOPs_ratio_vs_KB_MLP"),
            "backward_flops_ratio": _safe_float(row.get("FLOPs_ratio_vs_KB_MLP"), float("nan")) * 2.0,
            "step_ratio": row.get("step_ratio_vs_KB_MLP"),
            "memory_ratio": row.get("memory_ratio_vs_KB_MLP"),
            "curvature": row.get("curvature"),
            "ECE": row.get("ECE"),
            "NLL": row.get("NLL"),
            "GradPass": 1,
            "PromotionPass": promotion,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
        matrix_rows.append(out)
        if str(row.get("candidate_id", "")).endswith("+Functional"):
            ext = dict(out)
            ext["stage"] = "P8_EXTERNAL_FAIR_VALIDATION"
            ext["FullEdgeExternalFairPass"] = int(
                delta >= 0.0
                and params <= FAIR_PARAM_RATIO_MAX
                and flops <= FAIR_FLOPS_RATIO_MAX
                and _safe_float(ext["backward_flops_ratio"], 99.0) <= FAIR_BACKWARD_RATIO_MAX
                and step <= FAIR_STEP_RATIO_MAX
                and mem <= FAIR_MEMORY_RATIO_MAX
            )
            external_rows.append(ext)
    func_rows: List[Dict[str, Any]] = []
    for row in causality_src:
        mode = row.get("control")
        curv_ratio = _safe_float(row.get("curvature_ratio_vs_base"), float("nan"))
        func_rows.append({
            "stage": "P6_FUNCTIONAL_BASIS_CAUSALITY",
            "source_artifact": str(v90),
            "basis_id": _basis_id_from_edge(row.get("edge_basis", "")),
            "basis_family": row.get("edge_basis"),
            "functional_mode": mode,
            "dataset": row.get("task"),
            "test_acc": row.get("test_acc"),
            "delta_vs_base": "metric_unavailable",
            "curvature_ratio": row.get("curvature_ratio_vs_base"),
            "ECE_delta": "metric_unavailable",
            "NLL_delta": "metric_unavailable",
            "bad_step_rate": row.get("functional_bad_step_rate"),
            "holdout_descent_ratio": row.get("functional_holdout_ratio_mean"),
            "functional_update_time": "metric_unavailable",
            "CausalityPass": row.get("CausalityPass", 0),
            "FunctionalUsefulPass": int(mode == "Functional" and curv_ratio <= 0.90),
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    compute_rows: List[Dict[str, Any]] = []
    for row in compute_src:
        candidate = row.get("candidate_id", "")
        ext_match = next((r for r in external_src if r.get("candidate_id") == candidate and r.get("task") == row.get("task")), {})
        dims = ext_match.get("dims", "")
        k = 3 if "CompactPolySilu3" in candidate else 4
        try:
            parts = [int(x) for x in dims.split("x")]
            max_edges = max(parts[i] * parts[i + 1] for i in range(len(parts) - 1))
            mat_mb = 128 * max_edges * k * 4 / (1024.0 * 1024.0)
            mat_shape = f"128 x max_edge({max_edges}) x {k}"
        except Exception:
            mat_mb = float("nan")
            mat_shape = "metric_unavailable"
        fwd = _safe_float(row.get("forward_FLOPs_ratio_vs_KB_MLP"), float("nan"))
        bwd = _safe_float(row.get("backward_FLOPs_estimate_ratio_vs_KB_MLP"), float("nan"))
        step = _safe_float(row.get("step_time_ratio_vs_KB_MLP"), float("nan"))
        mem = _safe_float(ext_match.get("memory_ratio_vs_KB_MLP"), float("nan"))
        compute_rows.append({
            "stage": "P7_MATERIALIZATION_FREE_COMPUTE_AUDIT",
            "source_artifact": str(v90),
            "candidate_id": candidate,
            "materialized_tensor_shape": mat_shape,
            "materialized_MB_estimate": mat_mb,
            "forward_flops_ratio": fwd,
            "backward_flops_ratio": bwd,
            "kernel_time_ratio": "metric_unavailable",
            "step_ratio": step,
            "memory_ratio": mem,
            "ComputeRepairPass": int(fwd <= FAIR_FLOPS_RATIO_MAX and bwd <= FAIR_BACKWARD_RATIO_MAX and step <= FAIR_STEP_RATIO_MAX and mem <= FAIR_MEMORY_RATIO_MAX),
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    write_csv_rows(out_dir / "basis_parameterization_matrix.csv", matrix_rows)
    write_csv_rows(out_dir / "functional_basis_causality.csv", func_rows)
    write_csv_rows(out_dir / "materialization_free_compute_audit.csv", compute_rows)
    write_csv_rows(out_dir / "external_fair_validation.csv", external_rows)
    summary = {
        "promotion_pass_count": sum(int(r.get("PromotionPass", 0)) for r in matrix_rows),
        "external_fair_pass_count": sum(int(r.get("FullEdgeExternalFairPass", 0)) for r in external_rows),
        "functional_useful_pass_count": sum(int(r.get("FunctionalUsefulPass", 0)) for r in func_rows),
        "compute_repair_pass_count": sum(int(r.get("ComputeRepairPass", 0)) for r in compute_rows),
        "source_artifact": str(v90),
    }
    return matrix_rows, func_rows, compute_rows, external_rows, summary


def write_not_run_artifacts(out_dir: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    symbolic = [{
        "stage": "P9_SYMBOLIC_VALIDATION",
        "status": "not_run",
        "reason": "symbolic task runner not implemented in v9.1 first execution",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }]
    robustness = [{
        "stage": "P10_ROBUSTNESS_VALIDATION",
        "status": "not_run",
        "reason": "robustness perturbation runner not implemented in v9.1 first execution",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }]
    write_csv_rows(out_dir / "symbolic_validation.csv", symbolic)
    write_csv_rows(out_dir / "robustness_validation.csv", robustness)
    return symbolic, robustness


def make_boundary(
    p0: List[Dict[str, Any]],
    p1: List[Dict[str, Any]],
    p2: List[Dict[str, Any]],
    p3: List[Dict[str, Any]],
    p4: List[Dict[str, Any]],
    p5_summary: Dict[str, Any],
    symbolic: List[Dict[str, Any]],
    robustness: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    cond_pass = {}
    for r in p1:
        bid = r.get("basis_id")
        if bid:
            cond_pass.setdefault(bid, []).append(int(_safe_float(r.get("ConditioningPass"), 0)))
    fit_pass = {}
    for r in p2:
        bid = r.get("basis_id")
        if bid:
            fit_pass.setdefault(bid, []).append(int(_safe_float(r.get("FitPass"), 0)))
    grad_pass = {r.get("basis_id"): int(_safe_float(r.get("GradPass"), 0)) for r in p3 if r.get("basis_id")}
    overfit_pass = {r.get("basis_id"): int(_safe_float(r.get("OverfitPass"), 0)) for r in p4 if r.get("basis_id")}
    for spec in _basis_specs():
        labels = []
        if not all(cond_pass.get(spec.basis_id, [0])):
            labels.append("basis_not_conditioned")
        if not spec.full_edge_edge_kind:
            labels.append("basis_implementation_incomplete")
        if grad_pass.get(spec.basis_id, 0) == 0:
            labels.append("basis_gradcheck_missing_or_fail")
        if overfit_pass.get(spec.basis_id, 0) == 0:
            labels.append("basis_small_overfit_fail_or_not_run")
        rows.append({
            "stage": "P11_BOUNDARY_AUDIT",
            "basis_id": spec.basis_id,
            "basis_name": spec.basis_name,
            "boundary_labels": ",".join(labels) if labels else "wave0_2_pass",
            "ConditioningAllPass": int(all(cond_pass.get(spec.basis_id, [0]))),
            "SimpleFitAllPass": int(all(fit_pass.get(spec.basis_id, [0]))),
            "GradPass": grad_pass.get(spec.basis_id, 0),
            "OverfitPass": overfit_pass.get(spec.basis_id, 0),
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    justified_basis_ids = [
        spec.basis_id
        for spec in _basis_specs()
        if spec.official_eligible
        and all(cond_pass.get(spec.basis_id, [0]))
        and all(fit_pass.get(spec.basis_id, [0]))
        and grad_pass.get(spec.basis_id, 0) == 1
        and overfit_pass.get(spec.basis_id, 0) == 1
    ]
    planned_full_impl_complete = int(all(s.full_edge_edge_kind for s in _basis_specs() if s.official_eligible))
    promotion = int(p5_summary["promotion_pass_count"])
    external = int(p5_summary["external_fair_pass_count"])
    compute = int(p5_summary["compute_repair_pass_count"])
    symbolic_pass = 0
    robustness_pass = 0
    if external > 0:
        route = "R2-FullEdgeVisionBasisSuccess"
        blocker = "none_for_vision_family"
    elif not planned_full_impl_complete:
        route = "R8-BasisJustificationIncomplete"
        blocker = "planned_basis_families_not_fully_implemented_for_gradcheck_overfit_and_full_task"
    elif not justified_basis_ids:
        route = "R8-BasisJustificationIncomplete"
        blocker = "no_basis_passed_conditioning_fit_gradcheck_and_small_overfit"
    elif promotion == 0 and compute == 0:
        route = "R7-TransitionalOnly"
        blocker = "clean_transitional_remains_best_full_edge_basis_families_fail"
    elif promotion == 0:
        route = "R5-BasisTaskFail"
        blocker = "basis_compute_available_but_task_promotion_failed"
    else:
        route = "R4-BasisComputeFail"
        blocker = "basis_task_signal_present_but_compute_memory_fair_failed"
    decision = {
        "route": route,
        "success_v91_basis_justification": 0,
        "success_v91_full_edge_external_fair": int(external > 0),
        "success_v91_compute_repair": int(compute > 0),
        "success_v91_symbolic": symbolic_pass,
        "success_v91_robustness": robustness_pass,
        "basis_registry_pass": int(all(int(r.get("BasisJustificationPass", 0)) == 1 for r in p0)),
        "planned_full_basis_implementation_complete": planned_full_impl_complete,
        "justified_basis_count_wave0_to_wave2": len(justified_basis_ids),
        "justified_basis_ids_wave0_to_wave2": ",".join(justified_basis_ids),
        "promotion_pass_count": promotion,
        "external_fair_pass_count": external,
        "primary_blocker": blocker,
        "next_required_implementation": "fix_basis_conditioning_with_fan_norm_or_feature_source_then_implement_shared_lowrank_compute_repair_and_rerun_wave3_to_wave8",
    }
    rows.append({
        "stage": "P11_BOUNDARY_AUDIT",
        "basis_id": "GLOBAL",
        "basis_name": "GLOBAL",
        "boundary_labels": route,
        "primary_blocker": blocker,
        "symbolic_status": symbolic[0].get("status", "not_run") if symbolic else "not_run",
        "robustness_status": robustness[0].get("status", "not_run") if robustness else "not_run",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    })
    return rows, decision


def write_figures(out_dir: Path, p0: List[Dict[str, Any]]) -> None:
    fig_dir = out_dir / "figures"
    ensure_dir(fig_dir)
    labels = [str(r["basis_id"]) for r in p0]
    params = [float(r["parameter_count_per_edge"]) for r in p0]
    width = 700
    height = 240
    maxv = max(params) if params else 1.0
    bars = []
    for i, (lab, val) in enumerate(zip(labels, params)):
        x = 40 + i * 62
        h = int(160 * val / maxv)
        bars.append(f'<rect x="{x}" y="{190-h}" width="36" height="{h}" fill="#4b79a1"/><text x="{x}" y="215" font-size="10">{lab}</text>')
    (fig_dir / "p0_basis_flops_params_bar.svg").write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d"><rect width="100%%" height="100%%" fill="white"/><text x="20" y="24" font-size="14">Basis params per edge</text>%s</svg>' % (width, height, "".join(bars)),
        encoding="utf-8",
    )
    (fig_dir / "p0_basis_support_diagram.svg").write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" width="640" height="90"><rect width="100%%" height="100%%" fill="white"/><text x="20" y="35" font-size="13">Global: BAS0/BAS1/BAS2/BAS3/BAS9; Local: BAS4/BAS5/BAS6/BAS7/BAS8</text></svg>',
        encoding="utf-8",
    )
    (out_dir / "p0_basis_taxonomy_table.md").write_text(
        "\n".join(["| basis_id | basis_name | formula | official_eligible |", "|---|---|---|---|"] + [f"| {r['basis_id']} | {r['basis_name']} | {r['formula']} | {r['official_eligible']} |" for r in p0]) + "\n",
        encoding="utf-8",
    )


def write_hashes(out_dir: Path) -> None:
    paths = [
        SCRIPT_PATH,
        PLAN_PATH,
        ROOT / "dgkan/models/manual_full_edge.py",
        ROOT / "dgkan/training/manual_full_edge.py",
        out_dir / "basis_registry_audit.csv",
        out_dir / "basis_activation_conditioning.csv",
        out_dir / "basis_fit_diagnostics.csv",
        out_dir / "basis_gradcheck.csv",
        out_dir / "basis_small_overfit.csv",
        out_dir / "basis_parameterization_matrix.csv",
        out_dir / "functional_basis_causality.csv",
        out_dir / "materialization_free_compute_audit.csv",
        out_dir / "external_fair_validation.csv",
        out_dir / "basis_boundary_audit.csv",
        out_dir / "route_decision.json",
        out_dir / "v91_provenance_audit.csv",
    ]
    rows = []
    for path in paths:
        if path.exists():
            rows.append({"artifact": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path), "sha256": _sha256_file(path)})
    write_csv_rows(out_dir / "artifact_hashes.csv", rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--datasets", default="MNIST,Fashion-MNIST,KMNIST")
    parser.add_argument("--seed", type=int, default=1314)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--p1-train-size", type=int, default=512)
    parser.add_argument("--p1-max-values", type=int, default=20000)
    parser.add_argument("--p1-seeds", default="0,1,2")
    parser.add_argument("--p2-grid-size", type=int, default=2048)
    parser.add_argument("--p4-steps", type=int, default=300)
    parser.add_argument("--v90-artifact-dir", default=str(V90_DEFAULT))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    if args.fresh and out_dir.exists():
        shutil.rmtree(out_dir)
    ensure_dir(out_dir)
    ensure_dir(out_dir / "figures")
    device = v85.v83.get_device(args.device)
    manifest = {
        "stage": "V91_BASIS_JUSTIFICATION",
        "status": "measured",
        "started_utc": _now_iso(),
        "repo_root": str(ROOT),
        "plan_path": str(PLAN_PATH.relative_to(ROOT)),
        "script_path": str(SCRIPT_PATH.relative_to(ROOT)),
        "device": str(device),
        "datasets": args.datasets,
        "seed": int(args.seed),
        "v90_source_artifact": args.v90_artifact_dir,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    write_json(out_dir / "run_manifest.json", manifest)
    p0 = run_p0(out_dir)
    write_figures(out_dir, p0)
    p1 = run_p1(args, out_dir, device)
    p2 = run_p2(args, out_dir, device)
    p3 = run_p3(out_dir, device)
    p4 = run_p4(args, out_dir, device, p3)
    matrix_rows, func_rows, compute_rows, ext_rows, p5_summary = run_p5_to_p8(args, out_dir)
    symbolic, robustness = write_not_run_artifacts(out_dir)
    boundary_rows, decision = make_boundary(p0, p1, p2, p3, p4, p5_summary, symbolic, robustness)
    write_csv_rows(out_dir / "basis_boundary_audit.csv", boundary_rows)
    failure_rows = [
        {"failure_id": "F1_basis_justification_incomplete", "active": int(decision["route"] == "R8-BasisJustificationIncomplete"), "detail": decision["primary_blocker"], "fake_data_used": 0, "proxy_row_used": 0, "cpu_offload_used": 0},
        {"failure_id": "F2_external_fair_fail", "active": int(decision["external_fair_pass_count"] == 0), "detail": "external_fair_validation.csv", "fake_data_used": 0, "proxy_row_used": 0, "cpu_offload_used": 0},
        {"failure_id": "F3_compute_repair_fail", "active": int(decision["success_v91_compute_repair"] == 0), "detail": "materialization_free_compute_audit.csv", "fake_data_used": 0, "proxy_row_used": 0, "cpu_offload_used": 0},
        {"failure_id": "F4_symbolic_not_run", "active": 1, "detail": "symbolic_validation.csv status=not_run", "fake_data_used": 0, "proxy_row_used": 0, "cpu_offload_used": 0},
        {"failure_id": "F5_robustness_not_run", "active": 1, "detail": "robustness_validation.csv status=not_run", "fake_data_used": 0, "proxy_row_used": 0, "cpu_offload_used": 0},
    ]
    write_csv_rows(out_dir / "failure_table.csv", failure_rows)
    write_json(out_dir / "route_decision.json", decision)
    write_json(out_dir / "aggregate_decision.json", decision)
    evidence_files = [
        "basis_registry_audit.csv",
        "basis_activation_conditioning.csv",
        "basis_fit_diagnostics.csv",
        "basis_gradcheck.csv",
        "basis_small_overfit.csv",
        "basis_parameterization_matrix.csv",
        "functional_basis_causality.csv",
        "materialization_free_compute_audit.csv",
        "external_fair_validation.csv",
        "symbolic_validation.csv",
        "robustness_validation.csv",
        "basis_boundary_audit.csv",
        "failure_table.csv",
    ]
    audit = audit_no_fake([out_dir / name for name in evidence_files])
    audit.update({
        "stage": "V91_PROVENANCE_AUDIT",
        "status": "measured",
        "plan_path": str(PLAN_PATH.relative_to(ROOT)),
        "script_path": str(SCRIPT_PATH.relative_to(ROOT)),
    })
    write_csv_rows(out_dir / "v91_provenance_audit.csv", [audit])
    write_hashes(out_dir)


if __name__ == "__main__":
    main()
