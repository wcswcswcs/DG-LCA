#!/usr/bin/env python3
"""DG-KAN v9.2 basis-source and kernel-native gate runner.

This runner implements the first executable v9.2 pass from
``DG-KAN_总计划与v9.2下一步完整实验计划_补充kernel_native_gate.md``.

It deliberately does not claim FullEdge external success unless every upstream
gate opens.  The official first-wave scope is:

* P0 candidate registry / contract audit.
* P1 source x basis conditioning on real KANbeFair vision tensors.
* P2 one-layer basis fit diagnostics.
* P3 materialization-free hybrid residual gradcheck and 512-sample overfit for
  P1/P2 survivors only.
* P4 kernel-native feasibility against a same-parameter manual MLP-match.

P5-P7 artifacts are created as ``not_run`` unless P4 survivors exist.  The KAN
path uses explicit manual forward/backward/update and never calls
``loss.backward``.
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
from dgkan.artifacts.writer import artifact_hash_rows, ensure_dir, write_csv_rows, write_json  # noqa: E402
from dgkan.optim.manual_adamw import AdamWState, ManualAdamWConfig, adamw_update_  # noqa: E402
from dgkan.training.manual_full_edge import ce_loss_and_grad  # noqa: E402


PLAN_PATH = ROOT / "docs" / "DG-KAN_总计划与v9.2下一步完整实验计划_补充kernel_native_gate.md"
SCRIPT_PATH = ROOT / "experiments" / "run_v92_basis_source_kernel_gate.py"

COND_MAX = 1.0e4
DEAD_MAX = 0.30
DOM_MAX = 0.70
GRAD_REL_MAX = 1.0e-4
GRAD_COS_MIN = 0.999
OVERFIT_ACC_MIN = 0.98
P5_MIN_TRAINABILITY_TOL = 0.01


@dataclass(frozen=True)
class SourceSpec:
    source_id: str
    source_name: str
    description: str
    official_priority: int


@dataclass(frozen=True)
class BasisSpec:
    basis_id: str
    basis_name: str
    kind: str
    basis_dim: int
    local_support: int
    official_priority: int
    stability_gate_required: int


@dataclass(frozen=True)
class ParamSpec:
    parameterization_id: str
    name: str
    materializes_dense_edge_tensor: int
    official_eligible: int
    implementation_status: str


@dataclass
class SourceTransform:
    source_id: str
    mu: torch.Tensor
    std: torch.Tensor
    clip: float
    out_div: float
    patch_pool: int = 0

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        z = x
        if self.patch_pool:
            b = z.shape[0]
            z = F.avg_pool2d(z.view(b, 1, 28, 28), kernel_size=2, stride=2).flatten(1)
        if self.source_id == "S0":
            return z
        raw = (z - self.mu.to(z.device, z.dtype)) / self.std.to(z.device, z.dtype).clamp_min(1.0e-6)
        return raw.clamp(-self.clip, self.clip) / self.out_div

    def derivative_scale(self, x: torch.Tensor) -> torch.Tensor:
        if self.patch_pool:
            # The hybrid residual official candidate is only opened for non-patch
            # survivors in this first runner. PatchPool rows remain P1/P2 only.
            raise RuntimeError("P4 hybrid residual derivative for PatchPool is not implemented in this runner")
        if self.source_id == "S0":
            return torch.ones_like(x)
        std = self.std.to(x.device, x.dtype).clamp_min(1.0e-6)
        raw = (x - self.mu.to(x.device, x.dtype)) / std
        active = ((raw >= -self.clip) & (raw <= self.clip)).to(x.dtype)
        return active / (std * self.out_div)


def _sources() -> List[SourceSpec]:
    return [
        SourceSpec("S0", "Raw", "raw flattened input; v9.1 negative control", 0),
        SourceSpec("S1", "StdClip", "per-feature mean/std normalization with clipping", 1),
        SourceSpec("S2", "FanInNorm", "per-feature std normalization plus fan-in scaling", 1),
        SourceSpec("S3", "EdgeAffineNorm", "edge-owned affine-style stopgrad source normalization", 1),
        SourceSpec("S4", "PatchPool", "fixed 2x2 patch average source, then normalized", 1),
    ]


def _bases() -> List[BasisSpec]:
    return [
        BasisSpec("B0", "IdentityOnly", "identity", 1, 0, 1, 0),
        BasisSpec("B1", "Legendre3Residual", "legendre3", 3, 0, 1, 0),
        BasisSpec("B2", "Chebyshev3Residual", "chebyshev3", 3, 0, 1, 0),
        BasisSpec("B3", "PiecewiseLinearResidual", "piecewise_linear4", 4, 1, 1, 0),
        BasisSpec("B4", "SharedRBF4Residual", "rbf4", 4, 1, 1, 0),
        BasisSpec("B5", "BoundedRational2Residual", "bounded_rational2", 3, 0, 1, 1),
    ]


def _param_specs() -> List[ParamSpec]:
    return [
        ParamSpec("P0", "DenseFullEdge negative control", 1, 0, "negative_control_only"),
        ParamSpec("P1", "IdentityPlusLowRankResidual", 0, 1, "implemented_materialization_free"),
        ParamSpec("P2", "IdentityPlusGroupedResidual", 0, 0, "not_implemented_first_wave"),
        ParamSpec("P3", "SharedBasisLowRank", 0, 1, "implemented_materialization_free"),
        ParamSpec("P4", "ActiveKSharedResidual", 0, 1, "implemented_materialization_free_active_k1"),
    ]


def _parse_list(text: str) -> List[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def _parse_source_basis_pairs(text: str) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for item in _parse_list(text):
        if ":" in item:
            source, basis = item.split(":", 1)
        elif "-" in item:
            source, basis = item.split("-", 1)
        else:
            raise ValueError(f"source-basis pair must use SOURCE:BASIS syntax, got {item!r}")
        out.append((source.strip(), basis.strip()))
    return out


def _canonical_task(name: str) -> str:
    key = str(name).strip().lower()
    if key in {"fashion", "fashion-mnist", "fmnist"}:
        return "Fashion-MNIST"
    if key == "kmnist":
        return "KMNIST"
    if key == "mnist":
        return "MNIST"
    return str(name)


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


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


def _sample_values(x: torch.Tensor, max_values: int, seed: int) -> torch.Tensor:
    flat = x.flatten()
    if flat.numel() <= max_values:
        return flat
    gen = torch.Generator(device=flat.device).manual_seed(int(seed))
    idx = torch.randperm(flat.numel(), device=flat.device, generator=gen)[: int(max_values)]
    return flat[idx]


def _fit_source_transform(source_id: str, x_train: torch.Tensor) -> SourceTransform:
    if source_id == "S0":
        return SourceTransform(source_id=source_id, mu=torch.zeros(x_train.shape[1], device=x_train.device), std=torch.ones(x_train.shape[1], device=x_train.device), clip=1.0e9, out_div=1.0)
    if source_id == "S4":
        pooled = F.avg_pool2d(x_train.view(x_train.shape[0], 1, 28, 28), kernel_size=2, stride=2).flatten(1)
        return SourceTransform(
            source_id=source_id,
            mu=pooled.mean(dim=0),
            std=pooled.std(dim=0, unbiased=False).clamp_min(1.0e-6),
            clip=3.0,
            out_div=3.0,
            patch_pool=1,
        )
    clip = 3.0
    out_div = 3.0
    if source_id == "S3":
        clip = 2.0
        out_div = 2.0
    return SourceTransform(
        source_id=source_id,
        mu=x_train.mean(dim=0),
        std=x_train.std(dim=0, unbiased=False).clamp_min(1.0e-6),
        clip=clip,
        out_div=out_div,
        patch_pool=0,
    )


def _basis_values(kind: str, z: torch.Tensor) -> torch.Tensor:
    if kind == "identity":
        return z.unsqueeze(-1)
    if kind == "legendre3":
        return torch.stack([z, 0.5 * (3.0 * z.square() - 1.0), 0.5 * (5.0 * z.pow(3) - 3.0 * z)], dim=-1)
    if kind == "chebyshev3":
        return torch.stack([z, 2.0 * z.square() - 1.0, 4.0 * z.pow(3) - 3.0 * z], dim=-1)
    if kind == "piecewise_linear4":
        knots = torch.linspace(-1.0, 1.0, 4, device=z.device, dtype=z.dtype)
        width = torch.tensor(0.75, device=z.device, dtype=z.dtype)
        return torch.clamp(1.0 - (z.unsqueeze(-1) - knots.view(*([1] * z.ndim), 4)).abs() / width, min=0.0)
    if kind == "rbf4":
        centers = torch.linspace(-1.0, 1.0, 4, device=z.device, dtype=z.dtype)
        return torch.exp(-1.5 * (z.unsqueeze(-1) - centers.view(*([1] * z.ndim), 4)).square())
    if kind == "bounded_rational2":
        denom = 1.0 + F.softplus(0.5 * z.abs())
        return torch.stack([z, z.square() / denom, F.silu(z) / denom], dim=-1)
    raise ValueError(f"unknown basis kind {kind}")


def _basis_derivatives(kind: str, z: torch.Tensor) -> torch.Tensor:
    if kind == "identity":
        return torch.ones_like(z).unsqueeze(-1)
    if kind == "legendre3":
        return torch.stack([torch.ones_like(z), 3.0 * z, 0.5 * (15.0 * z.square() - 3.0)], dim=-1)
    if kind == "chebyshev3":
        return torch.stack([torch.ones_like(z), 4.0 * z, 12.0 * z.square() - 3.0], dim=-1)
    if kind == "piecewise_linear4":
        knots = torch.linspace(-1.0, 1.0, 4, device=z.device, dtype=z.dtype)
        width = torch.tensor(0.75, device=z.device, dtype=z.dtype)
        dist = z.unsqueeze(-1) - knots.view(*([1] * z.ndim), 4)
        active = (dist.abs() < width).to(z.dtype)
        return torch.where(dist >= 0.0, -active / width, active / width)
    if kind == "rbf4":
        centers = torch.linspace(-1.0, 1.0, 4, device=z.device, dtype=z.dtype)
        diff = z.unsqueeze(-1) - centers.view(*([1] * z.ndim), 4)
        basis = torch.exp(-1.5 * diff.square())
        return -3.0 * diff * basis
    if kind == "bounded_rational2":
        abs_z = z.abs()
        soft = F.softplus(0.5 * abs_z)
        denom = 1.0 + soft
        ddenom = 0.5 * torch.sigmoid(0.5 * abs_z) * torch.sign(z)
        sig = torch.sigmoid(z)
        silu = F.silu(z)
        silu_prime = sig * (1.0 + z * (1.0 - sig))
        d2 = (2.0 * z * denom - z.square() * ddenom) / denom.square()
        d3 = (silu_prime * denom - silu * ddenom) / denom.square()
        return torch.stack([torch.ones_like(z), d2, d3], dim=-1)
    raise ValueError(f"unknown basis kind {kind}")


def _corr_mean(flat: torch.Tensor) -> float:
    if flat.shape[1] <= 1:
        return 0.0
    centered = flat - flat.mean(dim=0, keepdim=True)
    std = centered.std(dim=0, unbiased=False).clamp_min(1.0e-12)
    corr = (centered / std).T @ (centered / std) / max(1, flat.shape[0])
    mask = ~torch.eye(corr.shape[0], dtype=torch.bool, device=corr.device)
    return float(corr[mask].abs().mean().detach().cpu())


def run_p0(out_dir: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    registry: List[Dict[str, Any]] = []
    contract_rows: List[Dict[str, Any]] = []
    for source in _sources():
        for basis in _bases():
            for param in _param_specs():
                functional_id = "F0-NoFunctional"
                official = int(param.official_eligible and not param.materializes_dense_edge_tensor and basis.official_priority and source.official_priority)
                candidate_id = f"{source.source_id}-{basis.basis_id}-{param.parameterization_id}-{functional_id}"
                row = {
                    "stage": "P0_CANDIDATE_REGISTRY",
                    "candidate_id": candidate_id,
                    "source_id": source.source_id,
                    "source_name": source.source_name,
                    "basis_id": basis.basis_id,
                    "basis_name": basis.basis_name,
                    "parameterization_id": param.parameterization_id,
                    "parameterization_name": param.name,
                    "functional_id": functional_id,
                    "model_level": "clean_full_edge_hybrid_residual",
                    "loss_type": "CE",
                    "label_smoothing": 0,
                    "external_teacher_used": 0,
                    "self_teacher_used": 0,
                    "geometry_loss_used": 0,
                    "sampler_changed": 0,
                    "class_weight_used": 0,
                    "cpu_offload_used": 0,
                    "uses_loss_backward": 0,
                    "materializes_dense_edge_tensor": param.materializes_dense_edge_tensor,
                    "official_eligible": official,
                    "implementation_status": param.implementation_status,
                    "fake_data_used": 0,
                    "proxy_row_used": 0,
                }
                registry.append(row)
                contract_pass = int(
                    row["loss_type"] == "CE"
                    and row["label_smoothing"] == 0
                    and row["external_teacher_used"] == 0
                    and row["self_teacher_used"] == 0
                    and row["geometry_loss_used"] == 0
                    and row["sampler_changed"] == 0
                    and row["class_weight_used"] == 0
                    and row["cpu_offload_used"] == 0
                    and row["uses_loss_backward"] == 0
                    and (not official or row["materializes_dense_edge_tensor"] == 0)
                )
                contract_rows.append({
                    "stage": "P0_CONTRACT_AUDIT",
                    "candidate_id": candidate_id,
                    "contract_pass": contract_pass,
                    "contract_violation": "" if contract_pass else "contract_or_materialization_violation",
                    "fake_data_used": 0,
                    "proxy_row_used": 0,
                    "cpu_offload_used": 0,
                })
    write_csv_rows(out_dir / "candidate_registry_v92.csv", registry)
    write_csv_rows(out_dir / "contract_audit_v92.csv", contract_rows)
    return registry, contract_rows


def run_p1(args: argparse.Namespace, out_dir: Path, device: torch.device) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for task in [_canonical_task(x) for x in _parse_list(args.datasets)]:
        x_train, _y_train, _x_test, _y_test, _input_dim, _output_dim, protocol = _load_task(args, task, train_size=int(args.p1_train_size), test_size=128)
        x_train = x_train.to(device=device, dtype=torch.float32)
        for source in _sources():
            transform = _fit_source_transform(source.source_id, x_train)
            z_all = transform.apply(x_train)
            values = _sample_values(z_all, int(args.p1_max_values), int(args.seed))
            for seed in [int(s) for s in _parse_list(args.p1_seeds)]:
                for basis in _bases():
                    b = _basis_values(basis.kind, values)
                    flat = b.reshape(-1, b.shape[-1])
                    centered = flat - flat.mean(dim=0, keepdim=True)
                    cov = centered.T @ centered / max(1, centered.shape[0] - 1)
                    eig = torch.linalg.eigvalsh(cov + torch.eye(cov.shape[0], device=device) * 1.0e-8).clamp_min(1.0e-12)
                    cond = float((eig.max() / eig.min()).detach().cpu())
                    dominant = float((eig.max() / eig.sum().clamp_min(1.0e-12)).detach().cpu())
                    col_std = flat.std(dim=0, unbiased=False)
                    dead = float((col_std <= 1.0e-6).float().mean().detach().cpu())
                    grad_norms = torch.linalg.vector_norm(flat, dim=0)
                    eps = 1.0e-3
                    bp = _basis_values(basis.kind, values + eps)
                    bm = _basis_values(basis.kind, values - eps)
                    slope = ((bp - bm) / (2.0 * eps)).abs().reshape(-1)
                    curv = ((bp - 2.0 * b + bm) / (eps * eps)).abs().reshape(-1)
                    pass_flag = int(cond <= COND_MAX and dead <= DEAD_MAX and dominant <= DOM_MAX)
                    rows.append({
                        "stage": "P1_BASIS_SOURCE_CONDITIONING",
                        "source_id": source.source_id,
                        "basis_id": basis.basis_id,
                        "dataset": task,
                        "seed": seed,
                        "input_dim_after_source": int(z_all.shape[1]) if z_all.ndim == 2 else 1,
                        "protocol": protocol,
                        "activation_mean": float(flat.mean().detach().cpu()),
                        "activation_std": float(flat.std(unbiased=False).detach().cpu()),
                        "activation_p01": float(torch.quantile(flat.reshape(-1), 0.01).detach().cpu()),
                        "activation_p99": float(torch.quantile(flat.reshape(-1), 0.99).detach().cpu()),
                        "basis_cov_condition": cond,
                        "dead_basis_fraction": dead,
                        "dominant_basis_fraction": dominant,
                        "basis_channel_corr_mean": _corr_mean(flat),
                        "grad_norm_mean": float(grad_norms.mean().detach().cpu()),
                        "grad_norm_cv": float((grad_norms.std(unbiased=False) / grad_norms.mean().clamp_min(1.0e-12)).detach().cpu()),
                        "slope_p95": float(torch.quantile(slope, 0.95).detach().cpu()),
                        "curvature_p95": float(torch.quantile(curv, 0.95).detach().cpu()),
                        "conditioning_pass": pass_flag,
                        "fake_data_used": 0,
                        "proxy_row_used": 0,
                        "cpu_offload_used": 0,
                    })
    write_csv_rows(out_dir / "basis_source_conditioning.csv", rows)
    return rows


def _passing_source_basis(p1_rows: Sequence[Dict[str, Any]]) -> List[Tuple[str, str]]:
    all_pairs = sorted({(str(r["source_id"]), str(r["basis_id"])) for r in p1_rows})
    out: List[Tuple[str, str]] = []
    for pair in all_pairs:
        pair_rows = [r for r in p1_rows if (str(r["source_id"]), str(r["basis_id"])) == pair]
        if pair_rows and all(int(r["conditioning_pass"]) == 1 for r in pair_rows):
            out.append(pair)
    return out


def _fit_targets(x: torch.Tensor) -> Dict[str, torch.Tensor]:
    return {
        "identity": x,
        "silu": F.silu(x),
        "quadratic": x.square(),
        "piecewise_ramp": torch.clamp(x + 0.25, min=0.0),
        "local_bump": torch.exp(-8.0 * (x - 0.35).square()),
        "low_frequency_sine": torch.sin(math.pi * x),
        "symbolic_polynomial": 0.5 * x.pow(3) - 0.2 * x.square() + x,
    }


def run_p2(args: argparse.Namespace, out_dir: Path, p1_pass_pairs: Sequence[Tuple[str, str]], device: torch.device) -> List[Dict[str, Any]]:
    basis_by_id = {b.basis_id: b for b in _bases()}
    rows: List[Dict[str, Any]] = []
    grid = torch.linspace(-1.0, 1.0, int(args.p2_grid_size), device=device)
    targets = _fit_targets(grid)
    pairs = list(p1_pass_pairs)
    if not pairs:
        for basis in _bases():
            rows.append({
                "stage": "P2_BASIS_ONE_LAYER_FIT",
                "source_id": "not_run",
                "basis_id": basis.basis_id,
                "target_type": "not_run",
                "status": "not_run",
                "reason": "no_source_basis_conditioning_survivor",
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            })
        write_csv_rows(out_dir / "basis_fit_diagnostics.csv", rows)
        return rows
    for source_id, basis_id in pairs:
        basis = basis_by_id[basis_id]
        x = grid
        phi = _basis_values(basis.kind, x)
        ones = torch.ones(phi.shape[0], 1, device=device)
        design = torch.cat([phi.reshape(phi.shape[0], -1), ones], dim=1)
        for target_name, y in targets.items():
            t0 = time.perf_counter()
            sol = torch.linalg.lstsq(design, y.unsqueeze(1)).solution.squeeze(1)
            _sync(device)
            fit_time = time.perf_counter() - t0
            pred = design @ sol
            mse = float((pred - y).square().mean().detach().cpu())
            sst = (y - y.mean()).square().sum().clamp_min(1.0e-12)
            r2 = float((1.0 - (pred - y).square().sum() / sst).detach().cpu())
            cond = float(torch.linalg.cond(design.T @ design + torch.eye(design.shape[1], device=device) * 1.0e-8).detach().cpu())
            coeff_norm = float(sol[:-1].norm().detach().cpu())
            curvature = float((_basis_derivatives(basis.kind, x)[1:] - _basis_derivatives(basis.kind, x)[:-1]).square().mean().detach().cpu())
            pass_flag = int((target_name in {"identity", "silu", "quadratic"} and r2 >= 0.95) or target_name not in {"identity", "silu", "quadratic"})
            rows.append({
                "stage": "P2_BASIS_ONE_LAYER_FIT",
                "source_id": source_id,
                "basis_id": basis_id,
                "target_type": target_name,
                "fit_MSE": mse,
                "fit_R2": r2,
                "coeff_norm": coeff_norm,
                "curvature": curvature,
                "condition_number": cond,
                "train_time": fit_time,
                "fit_pass": pass_flag,
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            })
        rows.append({
            "stage": "P2_BASIS_ONE_LAYER_FIT",
            "source_id": source_id,
            "basis_id": basis_id,
            "target_type": "transitional_hidden_projection",
            "status": "not_run",
            "reason": "transitional_hidden_extractor_not_implemented_in_v92_first_wave",
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    write_csv_rows(out_dir / "basis_fit_diagnostics.csv", rows)
    return rows


def _p2_survivors(p2_rows: Sequence[Dict[str, Any]]) -> List[Tuple[str, str]]:
    pairs = sorted({(str(r.get("source_id", "")), str(r.get("basis_id", ""))) for r in p2_rows if r.get("status", "") != "not_run"})
    out: List[Tuple[str, str]] = []
    for pair in pairs:
        required = [
            r for r in p2_rows
            if (str(r.get("source_id", "")), str(r.get("basis_id", ""))) == pair
            and str(r.get("target_type", "")) in {"identity", "silu", "quadratic"}
        ]
        if len(required) == 3 and all(int(r.get("fit_pass", 0)) == 1 for r in required):
            out.append(pair)
    return out


class HybridResidualLayer:
    def __init__(self, in_features: int, out_features: int, rank: int, basis: BasisSpec, source: SourceTransform, device: torch.device) -> None:
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.rank = int(rank)
        self.basis = basis
        self.source = source
        scale = 1.0 / math.sqrt(max(1, in_features))
        self.W0 = torch.randn(in_features, out_features, device=device) * scale
        self.U = torch.randn(in_features, rank, device=device) * scale
        self.V = torch.randn(out_features, rank, device=device) * scale
        self.A = torch.randn(rank, basis.basis_dim, device=device) * 0.02
        for p in self.parameters():
            p.requires_grad_(False)

    def parameters(self) -> List[torch.Tensor]:
        return [self.W0, self.U, self.V, self.A]

    def parameter_count(self) -> int:
        return sum(int(p.numel()) for p in self.parameters())

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        z = self.source.apply(x)
        dzdx = self.source.derivative_scale(x)
        basis = _basis_values(self.basis.kind, z)
        basis_flat = basis.reshape(basis.shape[0], self.in_features * self.basis.basis_dim)
        residual_matrix = (self.U.unsqueeze(1) * self.A.T.unsqueeze(0)).reshape(self.in_features * self.basis.basis_dim, self.rank)
        r = basis_flat @ residual_matrix
        y = x @ self.W0 + r @ self.V.T
        return y, {
            "x": x,
            "z": z,
            "dzdx": dzdx,
            "basis_flat": basis_flat,
            "residual_matrix": residual_matrix,
            "r": r,
        }

    def backward(self, dy: torch.Tensor, cache: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        x = cache["x"]
        z = cache["z"]
        dzdx = cache["dzdx"]
        basis_flat = cache["basis_flat"]
        residual_matrix = cache["residual_matrix"]
        r = cache["r"]
        dW0 = x.T @ dy
        dx = dy @ self.W0.T
        dV = dy.T @ r
        dr = dy @ self.V
        dM = basis_flat.T @ dr
        dM_blocks = dM.reshape(self.in_features, self.basis.basis_dim, self.rank)
        dU = (dM_blocks * self.A.T.unsqueeze(0)).sum(dim=1)
        dA = torch.einsum("ikr,ir->rk", dM_blocks, self.U)
        dbasis = (dr @ residual_matrix.T).reshape(x.shape[0], self.in_features, self.basis.basis_dim)
        dbasis_dz = _basis_derivatives(self.basis.kind, z)
        dz = (dbasis * dbasis_dz).sum(dim=-1)
        dx = dx + dz * dzdx
        return dx, [dW0, dU, dV, dA]

    def forward_flops(self) -> int:
        k = self.basis.basis_dim
        basis_ops = {
            "identity": 0,
            "legendre3": 6,
            "chebyshev3": 5,
            "piecewise_linear4": 8,
            "rbf4": 16,
            "bounded_rational2": 14,
        }.get(self.basis.kind, 8)
        return int(
            2 * self.in_features * self.out_features
            + basis_ops * self.in_features
            + 2 * self.in_features * k * self.rank
            + 2 * self.rank * self.out_features
        )

    def backward_flops(self) -> int:
        k = self.basis.basis_dim
        deriv_ops = {
            "identity": 0,
            "legendre3": 6,
            "chebyshev3": 5,
            "piecewise_linear4": 8,
            "rbf4": 18,
            "bounded_rational2": 18,
        }.get(self.basis.kind, 8)
        return int(
            4 * self.in_features * self.out_features
            + deriv_ops * self.in_features
            + 6 * self.in_features * k * self.rank
            + 4 * self.rank * self.out_features
        )


class HybridResidualClassifier:
    def __init__(self, in_features: int, hidden: int, out_features: int, rank: int, basis: BasisSpec, source1: SourceTransform, source2: SourceTransform, device: torch.device) -> None:
        self.layer1 = HybridResidualLayer(in_features, hidden, rank, basis, source1, device)
        self.layer2 = HybridResidualLayer(hidden, out_features, rank, basis, source2, device)

    def parameters(self) -> List[torch.Tensor]:
        return self.layer1.parameters() + self.layer2.parameters()

    def parameter_count(self) -> int:
        return sum(int(p.numel()) for p in self.parameters())

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        h, c1 = self.layer1.forward(x)
        y, c2 = self.layer2.forward(h)
        return y, {"c1": c1, "c2": c2}

    def backward(self, dy: torch.Tensor, cache: Dict[str, Any]) -> List[torch.Tensor]:
        dh, g2 = self.layer2.backward(dy, cache["c2"])
        _dx, g1 = self.layer1.backward(dh, cache["c1"])
        return g1 + g2

    def forward_flops(self) -> int:
        return self.layer1.forward_flops() + self.layer2.forward_flops()

    def backward_flops(self) -> int:
        return self.layer1.backward_flops() + self.layer2.backward_flops()


class SharedBasisResidualLayer:
    def __init__(self, in_features: int, out_features: int, rank: int, basis: BasisSpec, source: SourceTransform, device: torch.device, active_basis_indices: Sequence[int] | None = None) -> None:
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.rank = int(rank)
        self.basis = basis
        self.source = source
        self.active_basis_indices = tuple(int(i) for i in active_basis_indices) if active_basis_indices is not None else None
        self.effective_basis_dim = len(self.active_basis_indices) if self.active_basis_indices is not None else basis.basis_dim
        scale = 1.0 / math.sqrt(max(1, in_features))
        self.W0 = torch.randn(in_features, out_features, device=device) * scale
        self.M = torch.randn(in_features * self.effective_basis_dim, rank, device=device) * (0.02 / math.sqrt(max(1, self.effective_basis_dim)))
        self.V = torch.randn(out_features, rank, device=device) * scale
        for p in self.parameters():
            p.requires_grad_(False)

    def parameters(self) -> List[torch.Tensor]:
        return [self.W0, self.M, self.V]

    def parameter_count(self) -> int:
        return sum(int(p.numel()) for p in self.parameters())

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        z = self.source.apply(x)
        dzdx = self.source.derivative_scale(x)
        basis = _layer_basis_values(self, z)
        basis_flat = basis.reshape(basis.shape[0], self.in_features * self.effective_basis_dim)
        r = basis_flat @ self.M
        y = x @ self.W0 + r @ self.V.T
        return y, {
            "x": x,
            "z": z,
            "dzdx": dzdx,
            "basis_flat": basis_flat,
            "r": r,
        }

    def backward(self, dy: torch.Tensor, cache: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        x = cache["x"]
        z = cache["z"]
        dzdx = cache["dzdx"]
        basis_flat = cache["basis_flat"]
        r = cache["r"]
        dW0 = x.T @ dy
        dx = dy @ self.W0.T
        dV = dy.T @ r
        dr = dy @ self.V
        dM = basis_flat.T @ dr
        dbasis = (dr @ self.M.T).reshape(x.shape[0], self.in_features, self.effective_basis_dim)
        dbasis_dz = _layer_basis_derivatives(self, z)
        dz = (dbasis * dbasis_dz).sum(dim=-1)
        dx = dx + dz * dzdx
        return dx, [dW0, dM, dV]

    def forward_flops(self) -> int:
        k = self.effective_basis_dim
        basis_ops = {
            "identity": 0,
            "legendre3": 6,
            "chebyshev3": 4 if self.active_basis_indices == (2,) else 5,
            "piecewise_linear4": 8,
            "rbf4": 16,
            "bounded_rational2": 14,
        }.get(self.basis.kind, 8)
        return int(
            2 * self.in_features * self.out_features
            + basis_ops * self.in_features
            + 2 * self.in_features * k * self.rank
            + 2 * self.rank * self.out_features
        )

    def backward_flops(self) -> int:
        k = self.effective_basis_dim
        deriv_ops = {
            "identity": 0,
            "legendre3": 6,
            "chebyshev3": 4 if self.active_basis_indices == (2,) else 5,
            "piecewise_linear4": 8,
            "rbf4": 18,
            "bounded_rational2": 18,
        }.get(self.basis.kind, 8)
        return int(
            4 * self.in_features * self.out_features
            + deriv_ops * self.in_features
            + 4 * self.in_features * k * self.rank
            + 4 * self.rank * self.out_features
        )


class SharedBasisResidualClassifier:
    def __init__(self, in_features: int, hidden: int, out_features: int, rank: int, basis: BasisSpec, source1: SourceTransform, source2: SourceTransform, device: torch.device, active_basis_indices: Sequence[int] | None = None) -> None:
        self.layer1 = SharedBasisResidualLayer(in_features, hidden, rank, basis, source1, device, active_basis_indices=active_basis_indices)
        self.layer2 = SharedBasisResidualLayer(hidden, out_features, rank, basis, source2, device, active_basis_indices=active_basis_indices)

    def parameters(self) -> List[torch.Tensor]:
        return self.layer1.parameters() + self.layer2.parameters()

    def parameter_count(self) -> int:
        return sum(int(p.numel()) for p in self.parameters())

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        h, c1 = self.layer1.forward(x)
        y, c2 = self.layer2.forward(h)
        return y, {"c1": c1, "c2": c2}

    def backward(self, dy: torch.Tensor, cache: Dict[str, Any]) -> List[torch.Tensor]:
        dh, g2 = self.layer2.backward(dy, cache["c2"])
        _dx, g1 = self.layer1.backward(dh, cache["c1"])
        return g1 + g2

    def forward_flops(self) -> int:
        return self.layer1.forward_flops() + self.layer2.forward_flops()

    def backward_flops(self) -> int:
        return self.layer1.backward_flops() + self.layer2.backward_flops()


ModelT = HybridResidualClassifier | SharedBasisResidualClassifier
LayerT = HybridResidualLayer | SharedBasisResidualLayer


def _layer_basis_values(layer: LayerT, z: torch.Tensor) -> torch.Tensor:
    basis = _basis_values(layer.basis.kind, z)
    active = getattr(layer, "active_basis_indices", None)
    if active is not None:
        idx = torch.tensor(list(active), device=z.device, dtype=torch.long)
        basis = basis.index_select(-1, idx)
    return basis


def _layer_basis_derivatives(layer: LayerT, z: torch.Tensor) -> torch.Tensor:
    deriv = _basis_derivatives(layer.basis.kind, z)
    active = getattr(layer, "active_basis_indices", None)
    if active is not None:
        idx = torch.tensor(list(active), device=z.device, dtype=torch.long)
        deriv = deriv.index_select(-1, idx)
    return deriv


def _make_second_source(model_source1: SourceTransform, layer1: LayerT, x: torch.Tensor, hidden: int) -> SourceTransform:
    with torch.no_grad():
        h, _ = layer1.forward(x)
    return SourceTransform(
        source_id="S5",
        mu=h.mean(dim=0),
        std=h.std(dim=0, unbiased=False).clamp_min(1.0e-6),
        clip=model_source1.clip,
        out_div=model_source1.out_div,
        patch_pool=0,
    )


def _autograd_hybrid_forward(layer: LayerT, x: torch.Tensor) -> torch.Tensor:
    z = layer.source.apply(x)
    basis = _layer_basis_values(layer, z)
    basis_flat = basis.reshape(basis.shape[0], layer.in_features * basis.shape[-1])
    if hasattr(layer, "M"):
        residual_matrix = layer.M
    else:
        residual_matrix = (layer.U.unsqueeze(1) * layer.A.T.unsqueeze(0)).reshape(layer.in_features * layer.basis.basis_dim, layer.rank)
    r = basis_flat @ residual_matrix
    return x @ layer.W0 + r @ layer.V.T


def _layer_set_parameters(layer: LayerT, params: Sequence[torch.Tensor]) -> Tuple[torch.Tensor, ...]:
    if hasattr(layer, "M"):
        old = (layer.W0, layer.M, layer.V)
        layer.W0, layer.M, layer.V = params  # type: ignore[assignment]
        return old
    old = (layer.W0, layer.U, layer.V, layer.A)
    layer.W0, layer.U, layer.V, layer.A = params  # type: ignore[assignment]
    return old


def _gradcheck_layer(layer: LayerT, device: torch.device) -> Dict[str, float]:
    torch.manual_seed(17)
    x = torch.randn(7, layer.in_features, device=device) * 0.25
    dy = torch.randn(7, layer.out_features, device=device)
    y, cache = layer.forward(x)
    dx, grads = layer.backward(dy, cache)

    x_ref = x.detach().clone().requires_grad_(True)
    params_ref = [p.detach().clone().requires_grad_(True) for p in layer.parameters()]
    old = _layer_set_parameters(layer, params_ref)
    y_ref = _autograd_hybrid_forward(layer, x_ref)
    scalar = (y_ref * dy).sum()
    ref = torch.autograd.grad(scalar, [x_ref] + params_ref)
    _layer_set_parameters(layer, old)
    all_manual = [dx] + grads
    abs_diffs = [float((a - b).abs().max().detach().cpu()) for a, b in zip(all_manual, ref)]
    flat_m = torch.cat([g.reshape(-1) for g in all_manual])
    flat_r = torch.cat([g.reshape(-1) for g in ref])
    rel = float(((flat_m - flat_r).norm() / flat_r.norm().clamp_min(1.0e-12)).detach().cpu())
    cos = float(F.cosine_similarity(flat_m, flat_r, dim=0).detach().cpu())
    out_diff = float((y - y_ref.detach()).abs().max().detach().cpu())
    return {
        "GradRelErrMax": rel,
        "GradCosMin": cos,
        "OutputAbsDiffMax": out_diff,
        "DxAbsDiffMax": abs_diffs[0],
        "ParamGradAbsDiffMax": max(abs_diffs[1:]),
    }


def _train_overfit(model: ModelT, x: torch.Tensor, y: torch.Tensor, steps: int, batch_size: int, lr: float) -> Tuple[float, float, List[Dict[str, Any]]]:
    cfg = ManualAdamWConfig(lr=lr, weight_decay=0.0)
    states = [AdamWState.zeros_like(p) for p in model.parameters()]
    trace: List[Dict[str, Any]] = []
    n = int(x.shape[0])
    gen = torch.Generator(device=x.device).manual_seed(23)
    for step in range(1, int(steps) + 1):
        idx = torch.randint(0, n, (min(batch_size, n),), device=x.device, generator=gen)
        xb = x[idx]
        yb = y[idx]
        logits, cache = model.forward(xb)
        loss, grad_logits = ce_loss_and_grad(logits, yb)
        grads = model.backward(grad_logits, cache)
        for p, g, st in zip(model.parameters(), grads, states):
            adamw_update_(p, g, st, cfg)
        if step == 1 or step % max(1, steps // 10) == 0 or step == steps:
            with torch.no_grad():
                full_logits, _ = model.forward(x)
                full_loss, _ = ce_loss_and_grad(full_logits, y)
                acc = float((full_logits.argmax(dim=1) == y).float().mean().detach().cpu())
            trace.append({"step": step, "train_loss": float(full_loss.detach().cpu()), "train_acc": acc})
    with torch.no_grad():
        logits, _ = model.forward(x)
        loss, _ = ce_loss_and_grad(logits, y)
        acc = float((logits.argmax(dim=1) == y).float().mean().detach().cpu())
    return acc, float(loss.detach().cpu()), trace


def run_p3(args: argparse.Namespace, out_dir: Path, p2_survivors: Sequence[Tuple[str, str]], device: torch.device) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, ModelT]]:
    basis_by_id = {b.basis_id: b for b in _bases()}
    grad_rows: List[Dict[str, Any]] = []
    overfit_rows: List[Dict[str, Any]] = []
    models: Dict[str, ModelT] = {}
    if not p2_survivors:
        row = {
            "stage": "P3_HYBRID_RESIDUAL",
            "status": "not_run",
            "reason": "no_P1_P2_survivors",
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
        write_csv_rows(out_dir / "hybrid_residual_gradcheck.csv", [row])
        write_csv_rows(out_dir / "hybrid_residual_overfit.csv", [row])
        return [row], [row], models
    x_train, y_train, _x_test, _y_test, input_dim, output_dim, protocol = _load_task(args, "MNIST", train_size=int(args.p3_train_size), test_size=256)
    x_train = x_train.to(device=device, dtype=torch.float32)
    y_train = y_train.to(device=device)
    x_small = x_train[: int(args.p3_overfit_size)]
    y_small = y_train[: int(args.p3_overfit_size)]
    param_ids = _parse_list(getattr(args, "p3_parameterizations", "P1,P3"))
    for source_id, basis_id in p2_survivors:
        basis = basis_by_id[basis_id]
        if source_id == "S4":
            grad_rows.append({
                "stage": "P3_HYBRID_RESIDUAL_GRADCHECK",
                "candidate_id": f"{source_id}-{basis_id}-P1-F0",
                "source_id": source_id,
                "basis_id": basis_id,
                "status": "not_run",
                "reason": "PatchPool hybrid derivative not implemented in first-wave P3",
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            })
            continue
        source1 = _fit_source_transform(source_id, x_small)
        for param_id in param_ids:
            if param_id == "P1":
                dummy: LayerT = HybridResidualLayer(input_dim, int(args.p3_hidden_dim), int(args.p3_rank), basis, source1, device)
                source2 = _make_second_source(source1, dummy, x_small, int(args.p3_hidden_dim))
                model: ModelT = HybridResidualClassifier(input_dim, int(args.p3_hidden_dim), output_dim, int(args.p3_rank), basis, source1, source2, device)
            elif param_id == "P3":
                dummy = SharedBasisResidualLayer(input_dim, int(args.p3_hidden_dim), int(args.p3_rank), basis, source1, device)
                source2 = _make_second_source(source1, dummy, x_small, int(args.p3_hidden_dim))
                model = SharedBasisResidualClassifier(input_dim, int(args.p3_hidden_dim), output_dim, int(args.p3_rank), basis, source1, source2, device)
            elif param_id == "P4":
                active = (int(args.p4_active_cheb_index),) if basis.kind == "chebyshev3" else (0,)
                dummy = SharedBasisResidualLayer(input_dim, int(args.p3_hidden_dim), int(args.p3_rank), basis, source1, device, active_basis_indices=active)
                source2 = _make_second_source(source1, dummy, x_small, int(args.p3_hidden_dim))
                model = SharedBasisResidualClassifier(input_dim, int(args.p3_hidden_dim), output_dim, int(args.p3_rank), basis, source1, source2, device, active_basis_indices=active)
            else:
                continue
            gc = _gradcheck_layer(model.layer1, device)
            grad_pass = int(gc["GradRelErrMax"] <= GRAD_REL_MAX and gc["GradCosMin"] >= GRAD_COS_MIN)
            candidate_id = f"{source_id}-{basis_id}-{param_id}-F0"
            grad_rows.append({
                "stage": "P3_HYBRID_RESIDUAL_GRADCHECK",
                "candidate_id": candidate_id,
                "source_id": source_id,
                "basis_id": basis_id,
                "parameterization_id": param_id,
                **gc,
                "GradPass": grad_pass,
                "uses_loss_backward": 0,
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            })
            if not grad_pass:
                overfit_rows.append({
                    "stage": "P3_HYBRID_RESIDUAL_OVERFIT",
                    "candidate_id": candidate_id,
                    "source_id": source_id,
                    "basis_id": basis_id,
                    "parameterization_id": param_id,
                    "status": "not_run",
                    "reason": "gradcheck_failed",
                    "fake_data_used": 0,
                    "proxy_row_used": 0,
                    "cpu_offload_used": 0,
                })
                continue
            acc, loss, trace = _train_overfit(model, x_small, y_small, int(args.p3_steps), int(args.batch_size), float(args.p3_lr))
            overfit_pass = int(acc >= OVERFIT_ACC_MIN)
            overfit_rows.append({
                "stage": "P3_HYBRID_RESIDUAL_OVERFIT",
                "candidate_id": candidate_id,
                "source_id": source_id,
                "basis_id": basis_id,
                "parameterization_id": param_id,
                "protocol": protocol,
                "TrainAcc512": acc,
                "TrainLossFinal": loss,
                "OverfitPass": overfit_pass,
                "train_trace_json": json.dumps(trace, sort_keys=True),
                "params": model.parameter_count(),
                "hidden_dim": int(args.p3_hidden_dim),
                "rank": int(args.p3_rank),
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            })
            if overfit_pass:
                models[candidate_id] = model
    write_csv_rows(out_dir / "hybrid_residual_gradcheck.csv", grad_rows)
    write_csv_rows(out_dir / "hybrid_residual_overfit.csv", overfit_rows)
    return grad_rows, overfit_rows, models


def _manual_mlp_step(params: List[torch.Tensor], x: torch.Tensor, y: torch.Tensor, states: List[AdamWState], cfg: ManualAdamWConfig) -> Tuple[float, Dict[str, float]]:
    W1, W2 = params
    t0 = time.perf_counter()
    h_pre = x @ W1
    h = F.silu(h_pre)
    logits = h @ W2
    t1 = time.perf_counter()
    loss, dy = ce_loss_and_grad(logits, y)
    dW2 = h.T @ dy
    dh = dy @ W2.T
    sig = torch.sigmoid(h_pre)
    dpre = dh * sig * (1.0 + h_pre * (1.0 - sig))
    dW1 = x.T @ dpre
    t2 = time.perf_counter()
    for p, g, st in zip(params, [dW1, dW2], states):
        adamw_update_(p, g, st, cfg)
    t3 = time.perf_counter()
    return float(loss.detach().cpu()), {"forward": t1 - t0, "backward": t2 - t1, "update": t3 - t2, "step": t3 - t0}


def _ece_from_logits(logits: torch.Tensor, y: torch.Tensor, bins: int = 10) -> float:
    probs = logits.softmax(dim=1)
    conf, pred = probs.max(dim=1)
    correct = (pred == y).float()
    ece = torch.zeros((), device=logits.device)
    for idx in range(int(bins)):
        lo = idx / bins
        hi = (idx + 1) / bins
        mask = (conf > lo) & (conf <= hi)
        if bool(mask.any()):
            ece = ece + mask.float().mean() * torch.abs(conf[mask].mean() - correct[mask].mean())
    return float(ece.detach().cpu())


def _classification_metrics_from_logits(logits: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
    loss = F.cross_entropy(logits, y)
    log_probs = logits.log_softmax(dim=1)
    per_example_ce = -log_probs[torch.arange(y.numel(), device=y.device), y]
    probs = logits.softmax(dim=1)
    conf, pred = probs.max(dim=1)
    acc = (pred == y).float()
    top2 = logits.topk(k=min(2, logits.shape[1]), dim=1).values
    margin = top2[:, 0] - top2[:, 1] if top2.shape[1] > 1 else top2[:, 0]
    hard = margin <= torch.quantile(margin, 0.10)
    true_logits = logits[torch.arange(y.numel(), device=y.device), y]
    masked_logits = logits.clone()
    masked_logits[torch.arange(y.numel(), device=y.device), y] = -torch.inf
    top_wrong = masked_logits.max(dim=1).values
    wrong_conf = conf[pred != y]
    correct_conf = conf[pred == y]
    ce_p50 = torch.quantile(per_example_ce, 0.50)
    ce_p90 = torch.quantile(per_example_ce, 0.90)
    ce_p99 = torch.quantile(per_example_ce, 0.99)
    return {
        "acc": float(acc.mean().detach().cpu()),
        "loss": float(loss.detach().cpu()),
        "NLL": float(loss.detach().cpu()),
        "ECE": _ece_from_logits(logits, y),
        "CE_p50": float(ce_p50.detach().cpu()),
        "CE_p90": float(ce_p90.detach().cpu()),
        "CE_p99": float(ce_p99.detach().cpu()),
        "CE_p99_over_p50": float((ce_p99 / ce_p50.clamp_min(1.0e-8)).detach().cpu()),
        "logit_norm_mean": float(logits.norm(dim=1).mean().detach().cpu()),
        "logit_norm_p95": float(torch.quantile(logits.norm(dim=1), 0.95).detach().cpu()),
        "correct_logit_mean": float(true_logits.mean().detach().cpu()),
        "top_wrong_logit_mean": float(top_wrong.mean().detach().cpu()),
        "correct_margin_mean": float((true_logits - top_wrong).mean().detach().cpu()),
        "correct_margin_p10": float(torch.quantile(true_logits - top_wrong, 0.10).detach().cpu()),
        "wrong_confidence_p95": float(torch.quantile(wrong_conf, 0.95).detach().cpu()) if bool(wrong_conf.numel()) else 0.0,
        "correct_confidence_mean": float(correct_conf.mean().detach().cpu()) if bool(correct_conf.numel()) else 0.0,
        "margin_p10": float(torch.quantile(margin, 0.10).detach().cpu()),
        "hard_sample_acc": float(acc[hard].mean().detach().cpu()) if bool(hard.any()) else float(acc.mean().detach().cpu()),
    }


def _eval_hybrid_model(model: ModelT, x: torch.Tensor, y: torch.Tensor, batch_size: int) -> Dict[str, float]:
    parts: List[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, int(x.shape[0]), int(batch_size)):
            logits, _ = model.forward(x[start : start + int(batch_size)])
            parts.append(logits)
    return _classification_metrics_from_logits(torch.cat(parts, dim=0), y)


def _eval_mlp_params(params: Sequence[torch.Tensor], x: torch.Tensor, y: torch.Tensor, batch_size: int) -> Dict[str, float]:
    W1, W2 = params
    parts: List[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, int(x.shape[0]), int(batch_size)):
            xb = x[start : start + int(batch_size)]
            parts.append(F.silu(xb @ W1) @ W2)
    return _classification_metrics_from_logits(torch.cat(parts, dim=0), y)


def _adamw_update_foreach_(params: Sequence[torch.Tensor], grads: Sequence[torch.Tensor], states: Sequence[AdamWState], cfg: ManualAdamWConfig) -> None:
    if not params:
        return
    if not hasattr(torch, "_foreach_mul_"):
        for p, g, st in zip(params, grads, states):
            adamw_update_(p, g, st, cfg)
        return
    for st in states:
        st.step += 1
    step = states[0].step
    m_list = [st.m for st in states]
    v_list = [st.v for st in states]
    grad_list = list(grads)
    torch._foreach_mul_(m_list, cfg.beta1)
    torch._foreach_add_(m_list, grad_list, alpha=1.0 - cfg.beta1)
    torch._foreach_mul_(v_list, cfg.beta2)
    torch._foreach_addcmul_(v_list, grad_list, grad_list, value=1.0 - cfg.beta2)
    bias1 = 1.0 - cfg.beta1**step
    bias2 = 1.0 - cfg.beta2**step
    updates = torch._foreach_div(m_list, bias1)
    denoms = torch._foreach_sqrt(v_list)
    torch._foreach_div_(denoms, bias2**0.5)
    torch._foreach_add_(denoms, cfg.eps)
    torch._foreach_div_(updates, denoms)
    if cfg.weight_decay:
        wd_terms = torch._foreach_mul(list(params), cfg.weight_decay)
        torch._foreach_add_(updates, wd_terms)
    torch._foreach_add_(list(params), updates, alpha=-cfg.lr)


def _compiled_ce_grad(logits: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    log_probs = F.log_softmax(logits, dim=1)
    loss = -log_probs[torch.arange(labels.numel(), device=labels.device), labels].mean()
    probs = log_probs.exp()
    probs[torch.arange(labels.numel(), device=labels.device), labels] -= 1.0
    probs = probs / labels.numel()
    return loss, probs


def _cheb_layer_forward_packed(
    x: torch.Tensor,
    W0: torch.Tensor,
    U: torch.Tensor,
    V: torch.Tensor,
    A: torch.Tensor,
    mu: torch.Tensor,
    std: torch.Tensor,
    clip: float,
    out_div: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    raw = (x - mu) / std.clamp_min(1.0e-6)
    z = raw.clamp(-clip, clip) / out_div
    dzdx = (((raw >= -clip) & (raw <= clip)).to(x.dtype)) / (std.clamp_min(1.0e-6) * out_div)
    g0 = z
    g1 = 2.0 * z.square() - 1.0
    g2 = 4.0 * z.pow(3) - 3.0 * z
    G = torch.stack([g0, g1, g2], dim=2).reshape(x.shape[0], x.shape[1] * 3)
    M = (U.unsqueeze(1) * A.T.unsqueeze(0)).reshape(U.shape[0] * 3, U.shape[1])
    r = G @ M
    y = x @ W0 + r @ V.T
    return y, z, dzdx, G, r


def _cheb_layer_backward_packed(
    dy: torch.Tensor,
    x: torch.Tensor,
    z: torch.Tensor,
    dzdx: torch.Tensor,
    G: torch.Tensor,
    r: torch.Tensor,
    W0: torch.Tensor,
    U: torch.Tensor,
    V: torch.Tensor,
    A: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    dW0 = x.T @ dy
    dx = dy @ W0.T
    dV = dy.T @ r
    dr = dy @ V
    M = (U.unsqueeze(1) * A.T.unsqueeze(0)).reshape(U.shape[0] * 3, U.shape[1])
    dM = G.T @ dr
    dM_blocks = dM.reshape(U.shape[0], 3, U.shape[1])
    dU = (dM_blocks * A.T.unsqueeze(0)).sum(dim=1)
    dA = torch.einsum("ikr,ir->rk", dM_blocks, U)
    dG = (dr @ M.T).reshape(x.shape[0], U.shape[0], 3)
    d0 = torch.ones_like(z)
    d1 = 4.0 * z
    d2 = 12.0 * z.square() - 3.0
    dz = dG[..., 0] * d0 + dG[..., 1] * d1 + dG[..., 2] * d2
    dx = dx + dz * dzdx
    return dx, dW0, dU, dV, dA


def _cheb_layer_forward_direct_packed(
    x: torch.Tensor,
    W0: torch.Tensor,
    M: torch.Tensor,
    V: torch.Tensor,
    mu: torch.Tensor,
    std: torch.Tensor,
    clip: float,
    out_div: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    raw = (x - mu) / std.clamp_min(1.0e-6)
    z = raw.clamp(-clip, clip) / out_div
    dzdx = (((raw >= -clip) & (raw <= clip)).to(x.dtype)) / (std.clamp_min(1.0e-6) * out_div)
    g0 = z
    g1 = 2.0 * z.square() - 1.0
    g2 = 4.0 * z.pow(3) - 3.0 * z
    G = torch.stack([g0, g1, g2], dim=2).reshape(x.shape[0], x.shape[1] * 3)
    r = G @ M
    y = x @ W0 + r @ V.T
    return y, z, dzdx, G, r


def _cheb_layer_backward_direct_packed(
    dy: torch.Tensor,
    x: torch.Tensor,
    z: torch.Tensor,
    dzdx: torch.Tensor,
    G: torch.Tensor,
    r: torch.Tensor,
    W0: torch.Tensor,
    M: torch.Tensor,
    V: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    dW0 = x.T @ dy
    dx = dy @ W0.T
    dV = dy.T @ r
    dr = dy @ V
    dM = G.T @ dr
    dG = (dr @ M.T).reshape(x.shape[0], x.shape[1], 3)
    d0 = torch.ones_like(z)
    d1 = 4.0 * z
    d2 = 12.0 * z.square() - 3.0
    dz = dG[..., 0] * d0 + dG[..., 1] * d1 + dG[..., 2] * d2
    dx = dx + dz * dzdx
    return dx, dW0, dM, dV


def _cheb_model_forward_compiled_core(
    x: torch.Tensor,
    W01: torch.Tensor,
    U1: torch.Tensor,
    V1: torch.Tensor,
    A1: torch.Tensor,
    mu1: torch.Tensor,
    std1: torch.Tensor,
    W02: torch.Tensor,
    U2: torch.Tensor,
    V2: torch.Tensor,
    A2: torch.Tensor,
    mu2: torch.Tensor,
    std2: torch.Tensor,
    clip: float,
    out_div: float,
) -> torch.Tensor:
    h, _z1, _dz1, _G1, _r1 = _cheb_layer_forward_packed(x, W01, U1, V1, A1, mu1, std1, clip, out_div)
    y, _z2, _dz2, _G2, _r2 = _cheb_layer_forward_packed(h, W02, U2, V2, A2, mu2, std2, clip, out_div)
    return y


def _cheb_model_fwd_bwd_compiled_core(
    x: torch.Tensor,
    labels: torch.Tensor,
    W01: torch.Tensor,
    U1: torch.Tensor,
    V1: torch.Tensor,
    A1: torch.Tensor,
    mu1: torch.Tensor,
    std1: torch.Tensor,
    W02: torch.Tensor,
    U2: torch.Tensor,
    V2: torch.Tensor,
    A2: torch.Tensor,
    mu2: torch.Tensor,
    std2: torch.Tensor,
    clip: float,
    out_div: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    h, z1, dz1, G1, r1 = _cheb_layer_forward_packed(x, W01, U1, V1, A1, mu1, std1, clip, out_div)
    logits, z2, dz2, G2, r2 = _cheb_layer_forward_packed(h, W02, U2, V2, A2, mu2, std2, clip, out_div)
    loss, dy = _compiled_ce_grad(logits, labels)
    dh, dW02, dU2, dV2, dA2 = _cheb_layer_backward_packed(dy, h, z2, dz2, G2, r2, W02, U2, V2, A2)
    _dx, dW01, dU1, dV1, dA1 = _cheb_layer_backward_packed(dh, x, z1, dz1, G1, r1, W01, U1, V1, A1)
    return loss, dW01, dU1, dV1, dA1, dW02, dU2, dV2, dA2


def _cheb_model_forward_direct_compiled_core(
    x: torch.Tensor,
    W01: torch.Tensor,
    M1: torch.Tensor,
    V1: torch.Tensor,
    mu1: torch.Tensor,
    std1: torch.Tensor,
    W02: torch.Tensor,
    M2: torch.Tensor,
    V2: torch.Tensor,
    mu2: torch.Tensor,
    std2: torch.Tensor,
    clip: float,
    out_div: float,
) -> torch.Tensor:
    h, _z1, _dz1, _G1, _r1 = _cheb_layer_forward_direct_packed(x, W01, M1, V1, mu1, std1, clip, out_div)
    y, _z2, _dz2, _G2, _r2 = _cheb_layer_forward_direct_packed(h, W02, M2, V2, mu2, std2, clip, out_div)
    return y


def _cheb_model_fwd_bwd_direct_compiled_core(
    x: torch.Tensor,
    labels: torch.Tensor,
    W01: torch.Tensor,
    M1: torch.Tensor,
    V1: torch.Tensor,
    mu1: torch.Tensor,
    std1: torch.Tensor,
    W02: torch.Tensor,
    M2: torch.Tensor,
    V2: torch.Tensor,
    mu2: torch.Tensor,
    std2: torch.Tensor,
    clip: float,
    out_div: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    h, z1, dz1, G1, r1 = _cheb_layer_forward_direct_packed(x, W01, M1, V1, mu1, std1, clip, out_div)
    logits, z2, dz2, G2, r2 = _cheb_layer_forward_direct_packed(h, W02, M2, V2, mu2, std2, clip, out_div)
    loss, dy = _compiled_ce_grad(logits, labels)
    dh, dW02, dM2, dV2 = _cheb_layer_backward_direct_packed(dy, h, z2, dz2, G2, r2, W02, M2, V2)
    _dx, dW01, dM1, dV1 = _cheb_layer_backward_direct_packed(dh, x, z1, dz1, G1, r1, W01, M1, V1)
    return loss, dW01, dM1, dV1, dW02, dM2, dV2


def _cheb_active2_layer_forward_direct_packed(
    x: torch.Tensor,
    W0: torch.Tensor,
    M: torch.Tensor,
    V: torch.Tensor,
    mu: torch.Tensor,
    std: torch.Tensor,
    clip: float,
    out_div: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    raw = (x - mu) / std.clamp_min(1.0e-6)
    z = raw.clamp(-clip, clip) / out_div
    dzdx = (((raw >= -clip) & (raw <= clip)).to(x.dtype)) / (std.clamp_min(1.0e-6) * out_div)
    G = 4.0 * z.pow(3) - 3.0 * z
    r = G @ M
    y = x @ W0 + r @ V.T
    return y, z, dzdx, G, r


def _cheb_active2_layer_backward_direct_packed(
    dy: torch.Tensor,
    x: torch.Tensor,
    z: torch.Tensor,
    dzdx: torch.Tensor,
    G: torch.Tensor,
    r: torch.Tensor,
    W0: torch.Tensor,
    M: torch.Tensor,
    V: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    dW0 = x.T @ dy
    dx = dy @ W0.T
    dV = dy.T @ r
    dr = dy @ V
    dM = G.T @ dr
    dG = dr @ M.T
    dz = dG * (12.0 * z.square() - 3.0)
    dx = dx + dz * dzdx
    return dx, dW0, dM, dV


def _cheb_active2_model_forward_direct_compiled_core(
    x: torch.Tensor,
    W01: torch.Tensor,
    M1: torch.Tensor,
    V1: torch.Tensor,
    mu1: torch.Tensor,
    std1: torch.Tensor,
    W02: torch.Tensor,
    M2: torch.Tensor,
    V2: torch.Tensor,
    mu2: torch.Tensor,
    std2: torch.Tensor,
    clip: float,
    out_div: float,
) -> torch.Tensor:
    h, _z1, _dz1, _G1, _r1 = _cheb_active2_layer_forward_direct_packed(x, W01, M1, V1, mu1, std1, clip, out_div)
    y, _z2, _dz2, _G2, _r2 = _cheb_active2_layer_forward_direct_packed(h, W02, M2, V2, mu2, std2, clip, out_div)
    return y


def _cheb_active2_model_fwd_bwd_direct_compiled_core(
    x: torch.Tensor,
    labels: torch.Tensor,
    W01: torch.Tensor,
    M1: torch.Tensor,
    V1: torch.Tensor,
    mu1: torch.Tensor,
    std1: torch.Tensor,
    W02: torch.Tensor,
    M2: torch.Tensor,
    V2: torch.Tensor,
    mu2: torch.Tensor,
    std2: torch.Tensor,
    clip: float,
    out_div: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    h, z1, dz1, G1, r1 = _cheb_active2_layer_forward_direct_packed(x, W01, M1, V1, mu1, std1, clip, out_div)
    logits, z2, dz2, G2, r2 = _cheb_active2_layer_forward_direct_packed(h, W02, M2, V2, mu2, std2, clip, out_div)
    loss, dy = _compiled_ce_grad(logits, labels)
    dh, dW02, dM2, dV2 = _cheb_active2_layer_backward_direct_packed(dy, h, z2, dz2, G2, r2, W02, M2, V2)
    _dx, dW01, dM1, dV1 = _cheb_active2_layer_backward_direct_packed(dh, x, z1, dz1, G1, r1, W01, M1, V1)
    return loss, dW01, dM1, dV1, dW02, dM2, dV2


def _cheb_active1_layer_forward_direct_packed(
    x: torch.Tensor,
    W0: torch.Tensor,
    M: torch.Tensor,
    V: torch.Tensor,
    mu: torch.Tensor,
    std: torch.Tensor,
    clip: float,
    out_div: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    raw = (x - mu) / std.clamp_min(1.0e-6)
    z = raw.clamp(-clip, clip) / out_div
    dzdx = (((raw >= -clip) & (raw <= clip)).to(x.dtype)) / (std.clamp_min(1.0e-6) * out_div)
    G = 2.0 * z.square() - 1.0
    r = G @ M
    y = x @ W0 + r @ V.T
    return y, z, dzdx, G, r


def _cheb_active1_layer_backward_direct_packed(
    dy: torch.Tensor,
    x: torch.Tensor,
    z: torch.Tensor,
    dzdx: torch.Tensor,
    G: torch.Tensor,
    r: torch.Tensor,
    W0: torch.Tensor,
    M: torch.Tensor,
    V: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    dW0 = x.T @ dy
    dx = dy @ W0.T
    dV = dy.T @ r
    dr = dy @ V
    dM = G.T @ dr
    dG = dr @ M.T
    dz = dG * (4.0 * z)
    dx = dx + dz * dzdx
    return dx, dW0, dM, dV


def _cheb_active1_model_forward_direct_compiled_core(
    x: torch.Tensor,
    W01: torch.Tensor,
    M1: torch.Tensor,
    V1: torch.Tensor,
    mu1: torch.Tensor,
    std1: torch.Tensor,
    W02: torch.Tensor,
    M2: torch.Tensor,
    V2: torch.Tensor,
    mu2: torch.Tensor,
    std2: torch.Tensor,
    clip: float,
    out_div: float,
) -> torch.Tensor:
    h, _z1, _dz1, _G1, _r1 = _cheb_active1_layer_forward_direct_packed(x, W01, M1, V1, mu1, std1, clip, out_div)
    y, _z2, _dz2, _G2, _r2 = _cheb_active1_layer_forward_direct_packed(h, W02, M2, V2, mu2, std2, clip, out_div)
    return y


def _cheb_active1_model_fwd_bwd_direct_compiled_core(
    x: torch.Tensor,
    labels: torch.Tensor,
    W01: torch.Tensor,
    M1: torch.Tensor,
    V1: torch.Tensor,
    mu1: torch.Tensor,
    std1: torch.Tensor,
    W02: torch.Tensor,
    M2: torch.Tensor,
    V2: torch.Tensor,
    mu2: torch.Tensor,
    std2: torch.Tensor,
    clip: float,
    out_div: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    h, z1, dz1, G1, r1 = _cheb_active1_layer_forward_direct_packed(x, W01, M1, V1, mu1, std1, clip, out_div)
    logits, z2, dz2, G2, r2 = _cheb_active1_layer_forward_direct_packed(h, W02, M2, V2, mu2, std2, clip, out_div)
    loss, dy = _compiled_ce_grad(logits, labels)
    dh, dW02, dM2, dV2 = _cheb_active1_layer_backward_direct_packed(dy, h, z2, dz2, G2, r2, W02, M2, V2)
    _dx, dW01, dM1, dV1 = _cheb_active1_layer_backward_direct_packed(dh, x, z1, dz1, G1, r1, W01, M1, V1)
    return loss, dW01, dM1, dV1, dW02, dM2, dV2


def _mlp_forward_compiled_core(x: torch.Tensor, W1: torch.Tensor, W2: torch.Tensor) -> torch.Tensor:
    return F.silu(x @ W1) @ W2


def _mlp_fwd_bwd_compiled_core(x: torch.Tensor, labels: torch.Tensor, W1: torch.Tensor, W2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    h_pre = x @ W1
    h = F.silu(h_pre)
    logits = h @ W2
    loss, dy = _compiled_ce_grad(logits, labels)
    dW2 = h.T @ dy
    dh = dy @ W2.T
    sig = torch.sigmoid(h_pre)
    dpre = dh * sig * (1.0 + h_pre * (1.0 - sig))
    dW1 = x.T @ dpre
    return loss, dW1, dW2


_COMPILED_FUNCS: Dict[str, Any] = {}


def _maybe_compile(name: str, fn: Any) -> Any:
    if name not in _COMPILED_FUNCS:
        _COMPILED_FUNCS[name] = torch.compile(fn, mode="reduce-overhead") if hasattr(torch, "compile") else fn
    return _COMPILED_FUNCS[name]


def _bench_callable_ms(fn: Any, reps: int, device: torch.device) -> float:
    _sync(device)
    if device.type == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(int(reps)):
            fn()
        end.record()
        torch.cuda.synchronize(device)
        return float(start.elapsed_time(end) / max(1, int(reps)))
    t0 = time.perf_counter()
    for _ in range(int(reps)):
        fn()
    return (time.perf_counter() - t0) * 1000.0 / max(1, int(reps))


def _hybrid_step(model: ModelT, x: torch.Tensor, y: torch.Tensor, states: List[AdamWState], cfg: ManualAdamWConfig) -> Tuple[float, Dict[str, float]]:
    t0 = time.perf_counter()
    logits, cache = model.forward(x)
    t1 = time.perf_counter()
    loss, dy = ce_loss_and_grad(logits, y)
    grads = model.backward(dy, cache)
    t2 = time.perf_counter()
    for p, g, st in zip(model.parameters(), grads, states):
        adamw_update_(p, g, st, cfg)
    t3 = time.perf_counter()
    return float(loss.detach().cpu()), {"forward": t1 - t0, "backward": t2 - t1, "update": t3 - t2, "step": t3 - t0}


def _estimate_hybrid_memory_mb(model: ModelT, batch_size: int) -> float:
    bytes_per = 4
    params = model.parameter_count() * bytes_per
    k1 = getattr(model.layer1, "effective_basis_dim", model.layer1.basis.basis_dim)
    k2 = getattr(model.layer2, "effective_basis_dim", model.layer2.basis.basis_dim)
    cache = batch_size * (
        model.layer1.in_features * (1 + k1)
        + model.layer1.rank
        + model.layer2.in_features * (1 + k2)
        + model.layer2.rank
    ) * bytes_per
    opt = model.parameter_count() * bytes_per * 2
    return float((params + cache + opt) / (1024.0 * 1024.0))


def _estimate_mlp_memory_mb(in_dim: int, hidden: int, out_dim: int, batch_size: int) -> float:
    params = in_dim * hidden + hidden * out_dim
    cache = batch_size * (hidden * 2 + out_dim)
    opt = params * 2
    return float((params + cache + opt) * 4 / (1024.0 * 1024.0))


def _build_hybrid_model_from_ids(
    source_id: str,
    basis_id: str,
    param_id: str,
    x_source_fit: torch.Tensor,
    input_dim: int,
    output_dim: int,
    hidden: int,
    rank: int,
    device: torch.device,
    active_cheb_index: int = 2,
) -> ModelT:
    basis = {b.basis_id: b for b in _bases()}[basis_id]
    source1 = _fit_source_transform(source_id, x_source_fit)
    if param_id == "P1":
        dummy: LayerT = HybridResidualLayer(input_dim, hidden, rank, basis, source1, device)
        source2 = _make_second_source(source1, dummy, x_source_fit[: min(512, int(x_source_fit.shape[0]))], hidden)
        return HybridResidualClassifier(input_dim, hidden, output_dim, rank, basis, source1, source2, device)
    if param_id == "P3":
        dummy = SharedBasisResidualLayer(input_dim, hidden, rank, basis, source1, device)
        source2 = _make_second_source(source1, dummy, x_source_fit[: min(512, int(x_source_fit.shape[0]))], hidden)
        return SharedBasisResidualClassifier(input_dim, hidden, output_dim, rank, basis, source1, source2, device)
    if param_id == "P4":
        active = (int(active_cheb_index),) if basis.kind == "chebyshev3" else (0,)
        dummy = SharedBasisResidualLayer(input_dim, hidden, rank, basis, source1, device, active_basis_indices=active)
        source2 = _make_second_source(source1, dummy, x_source_fit[: min(512, int(x_source_fit.shape[0]))], hidden)
        return SharedBasisResidualClassifier(input_dim, hidden, output_dim, rank, basis, source1, source2, device, active_basis_indices=active)
    raise ValueError(f"unsupported P5 parameterization {param_id}")


def run_p4(args: argparse.Namespace, out_dir: Path, models: Dict[str, ModelT], device: torch.device) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    bench_rows: List[Dict[str, Any]] = []
    kernel_rows: List[Dict[str, Any]] = []
    if not models:
        row = {
            "stage": "P4_KERNEL_NATIVE_FEASIBILITY",
            "status": "not_run",
            "reason": "no_P3_overfit_survivor",
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
        write_csv_rows(out_dir / "materialization_free_microbench.csv", [row])
        write_csv_rows(out_dir / "kernel_native_feasibility_vs_mlp.csv", [row])
        write_csv_rows(out_dir / "kernelization_required_candidates.csv", [row])
        return [row], [row]
    x_train, y_train, _x_test, _y_test, input_dim, output_dim, _protocol = _load_task(args, "MNIST", train_size=max(1024, int(args.p4_batch_size) * 4), test_size=128)
    x_train = x_train.to(device=device, dtype=torch.float32)
    y_train = y_train.to(device=device)
    x_batch = x_train[: int(args.p4_batch_size)]
    y_batch = y_train[: int(args.p4_batch_size)]
    cfg = ManualAdamWConfig(lr=float(args.p3_lr), weight_decay=0.0)
    for candidate_id, model in models.items():
        params_kan = model.parameter_count()
        best_h = max(1, round(params_kan / max(1, input_dim + output_dim)))
        candidates = [max(1, best_h + delta) for delta in range(-8, 9)]
        hidden = min(candidates, key=lambda h: abs((input_dim * h + h * output_dim) - params_kan))
        params_mlp = input_dim * hidden + hidden * output_dim
        W1 = torch.randn(input_dim, hidden, device=device) / math.sqrt(input_dim)
        W2 = torch.randn(hidden, output_dim, device=device) / math.sqrt(hidden)
        W1.requires_grad_(False)
        W2.requires_grad_(False)
        mlp_states = [AdamWState.zeros_like(W1), AdamWState.zeros_like(W2)]
        kan_states = [AdamWState.zeros_like(p) for p in model.parameters()]
        source_id, basis_id, param_id, _f = candidate_id.split("-")
        if not getattr(args, "disable_p4_compiled_kernel_native_path", False) and model.layer1.basis.kind == "chebyshev3":
            direct_param = hasattr(model.layer1, "M")
            active2_param = direct_param and getattr(model.layer1, "active_basis_indices", None) == (2,)
            active1_param = direct_param and getattr(model.layer1, "active_basis_indices", None) == (1,)
            if active2_param:
                cfwd = _maybe_compile("v92_cheb_active2_direct_forward", _cheb_active2_model_forward_direct_compiled_core)
                cbwd = _maybe_compile("v92_cheb_active2_direct_fwd_bwd", _cheb_active2_model_fwd_bwd_direct_compiled_core)
            elif active1_param:
                cfwd = _maybe_compile("v92_cheb_active1_direct_forward", _cheb_active1_model_forward_direct_compiled_core)
                cbwd = _maybe_compile("v92_cheb_active1_direct_fwd_bwd", _cheb_active1_model_fwd_bwd_direct_compiled_core)
            else:
                cfwd = _maybe_compile("v92_cheb_direct_forward" if direct_param else "v92_cheb_forward", _cheb_model_forward_direct_compiled_core if direct_param else _cheb_model_forward_compiled_core)
                cbwd = _maybe_compile("v92_cheb_direct_fwd_bwd" if direct_param else "v92_cheb_fwd_bwd", _cheb_model_fwd_bwd_direct_compiled_core if direct_param else _cheb_model_fwd_bwd_compiled_core)
            mfwd = _maybe_compile("v92_mlp_forward", _mlp_forward_compiled_core)
            mbwd = _maybe_compile("v92_mlp_fwd_bwd", _mlp_fwd_bwd_compiled_core)
            if direct_param:
                kan_args = (
                    model.layer1.W0, model.layer1.M, model.layer1.V,
                    model.layer1.source.mu.to(device), model.layer1.source.std.to(device),
                    model.layer2.W0, model.layer2.M, model.layer2.V,
                    model.layer2.source.mu.to(device), model.layer2.source.std.to(device),
                    float(model.layer1.source.clip), float(model.layer1.source.out_div),
                )
            else:
                kan_args = (
                    model.layer1.W0, model.layer1.U, model.layer1.V, model.layer1.A,
                    model.layer1.source.mu.to(device), model.layer1.source.std.to(device),
                    model.layer2.W0, model.layer2.U, model.layer2.V, model.layer2.A,
                    model.layer2.source.mu.to(device), model.layer2.source.std.to(device),
                    float(model.layer1.source.clip), float(model.layer1.source.out_div),
                )
            manual_logits, manual_cache = model.forward(x_batch)
            manual_loss, manual_dy = ce_loss_and_grad(manual_logits, y_batch)
            manual_grads = model.backward(manual_dy, manual_cache)
            compiled_logits = cfwd(x_batch, *kan_args).clone()
            compiled_pack_raw = cbwd(x_batch, y_batch, *kan_args)
            compiled_loss = compiled_pack_raw[0].clone()
            compiled_grads = [g.clone() for g in compiled_pack_raw[1:]]
            _sync(device)
            compiled_output_diff = float((manual_logits - compiled_logits).abs().max().detach().cpu())
            compiled_loss_diff = float((manual_loss - compiled_loss).abs().detach().cpu())
            compiled_grad_diff = max(float((a - b).abs().max().detach().cpu()) for a, b in zip(manual_grads, compiled_grads))
            for _ in range(int(args.p4_warmup)):
                cfwd(x_batch, *kan_args)
                cbwd(x_batch, y_batch, *kan_args)
                mfwd(x_batch, W1, W2)
                mbwd(x_batch, y_batch, W1, W2)
            _sync(device)

            kan_states_compiled = [AdamWState.zeros_like(p) for p in model.parameters()]
            mlp_states_compiled = [AdamWState.zeros_like(W1), AdamWState.zeros_like(W2)]

            def kan_step_compiled() -> None:
                pack = cbwd(x_batch, y_batch, *kan_args)
                _adamw_update_foreach_(model.parameters(), pack[1:], kan_states_compiled, cfg)

            def mlp_step_compiled() -> None:
                pack = mbwd(x_batch, y_batch, W1, W2)
                _adamw_update_foreach_([W1, W2], pack[1:], mlp_states_compiled, cfg)

            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            for _ in range(3):
                kan_step_compiled()
            _sync(device)
            peak_kan = float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)) if device.type == "cuda" else _estimate_hybrid_memory_mb(model, int(args.p4_batch_size))
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            for _ in range(3):
                mlp_step_compiled()
            _sync(device)
            peak_mlp = float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)) if device.type == "cuda" else _estimate_mlp_memory_mb(input_dim, hidden, output_dim, int(args.p4_batch_size))

            f_kan = _bench_callable_ms(lambda: cfwd(x_batch, *kan_args), int(args.p4_reps), device)
            fb_kan = _bench_callable_ms(lambda: cbwd(x_batch, y_batch, *kan_args), int(args.p4_reps), device)
            b_kan = max(0.0, fb_kan - f_kan)
            s_kan = _bench_callable_ms(kan_step_compiled, int(args.p4_reps), device)
            f_mlp = _bench_callable_ms(lambda: mfwd(x_batch, W1, W2), int(args.p4_reps), device)
            fb_mlp = _bench_callable_ms(lambda: mbwd(x_batch, y_batch, W1, W2), int(args.p4_reps), device)
            b_mlp = max(0.0, fb_mlp - f_mlp)
            s_mlp = _bench_callable_ms(mlp_step_compiled, int(args.p4_reps), device)

            mlp_flops = 2 * input_dim * hidden + 2 * hidden * output_dim
            mlp_backward_flops = 4 * input_dim * hidden + 4 * hidden * output_dim
            kan_flops = model.forward_flops()
            kan_backward_flops = model.backward_flops()
            forward_ratio = f_kan / max(f_mlp, 1.0e-12)
            backward_ratio = b_kan / max(b_mlp, 1.0e-12)
            step_ratio = s_kan / max(s_mlp, 1.0e-12)
            memory_ratio = peak_kan / max(peak_mlp, 1.0e-12)
            params_ratio = params_kan / max(params_mlp, 1)
            flops_ratio = kan_flops / max(mlp_flops, 1)
            backward_flops_ratio = kan_backward_flops / max(mlp_backward_flops, 1)
            pass_flag = int(
                abs(params_ratio - 1.0) <= 0.05
                and forward_ratio <= 1.25
                and backward_ratio <= 1.50
                and step_ratio <= 1.50
                and memory_ratio <= 1.05
                and flops_ratio <= 1.05
                and backward_flops_ratio <= 1.50
                and compiled_output_diff <= 1.0e-5
                and compiled_grad_diff <= 1.0e-5
            )
            row = {
                "stage": "P4_KERNEL_NATIVE_FEASIBILITY",
                "candidate_id": candidate_id,
                "source_id": source_id,
                "basis_id": basis_id,
                "parameterization_id": param_id,
                "matched_mlp_id": f"CompiledManualMLP-hidden{hidden}",
                "params_kan": params_kan,
                "params_mlp_match": params_mlp,
                "params_ratio_vs_mlp_match": params_ratio,
                "materializes_dense_edge_tensor": 0,
                "materialized_tensor_shape": "basis_flat_B_x_in_times_K_not_dense_edge",
                "materialized_MB": 0.0,
                "compiled_kernel_native_path": 1,
                "compiled_correctness_output_max_abs_diff": compiled_output_diff,
                "compiled_correctness_loss_abs_diff": compiled_loss_diff,
                "compiled_correctness_grad_max_abs_diff": compiled_grad_diff,
                "forward_time_ms_kan": f_kan,
                "forward_time_ms_mlp_match": f_mlp,
                "backward_time_ms_kan": b_kan,
                "backward_time_ms_mlp_match": b_mlp,
                "step_time_ms_kan": s_kan,
                "step_time_ms_mlp_match": s_mlp,
                "peak_memory_MB_kan": peak_kan,
                "peak_memory_MB_mlp_match": peak_mlp,
                "activation_cache_MB_kan": _estimate_hybrid_memory_mb(model, int(args.p4_batch_size)),
                "activation_cache_MB_mlp_match": _estimate_mlp_memory_mb(input_dim, hidden, output_dim, int(args.p4_batch_size)),
                "optimizer_state_MB_kan": params_kan * 2 * 4 / (1024.0 * 1024.0),
                "optimizer_state_MB_mlp_match": params_mlp * 2 * 4 / (1024.0 * 1024.0),
                "functional_update_time_ms": 0.0,
                "forward_ratio_vs_mlp_match": forward_ratio,
                "backward_ratio_vs_mlp_match": backward_ratio,
                "step_ratio_vs_mlp_match": step_ratio,
                "memory_ratio_vs_mlp_match": memory_ratio,
                "forward_FLOPs_ratio": flops_ratio,
                "backward_FLOPs_ratio": backward_flops_ratio,
                "kernel_count_total": "torch_compile_not_decomposed",
                "small_kernel_count": "torch_compile_not_decomposed",
                "basis_eval_time": "compiled_not_measured_separately",
                "residual_projection_time": "compiled_not_measured_separately",
                "output_projection_time": "compiled_not_measured_separately",
                "kernelization_required": int(pass_flag == 0),
                "kernel_native_pass": pass_flag,
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            }
            bench_rows.append(row)
            if not pass_flag:
                kernel_rows.append({
                    "stage": "P4_KERNELIZATION_REQUIRED",
                    "candidate_id": candidate_id,
                    "source_id": source_id,
                    "basis_id": basis_id,
                    "reason": "P4_compiled_forward_backward_step_memory_or_flops_gate_failed",
                    "forward_ratio_vs_mlp_match": forward_ratio,
                    "backward_ratio_vs_mlp_match": backward_ratio,
                    "step_ratio_vs_mlp_match": step_ratio,
                    "memory_ratio_vs_mlp_match": memory_ratio,
                    "forward_FLOPs_ratio": flops_ratio,
                    "backward_FLOPs_ratio": backward_flops_ratio,
                    "next_required_implementation": "triton_or_cuda_fused_basis_eval_residual_projection_backward",
                    "fake_data_used": 0,
                    "proxy_row_used": 0,
                    "cpu_offload_used": 0,
                })
            continue
        for _ in range(int(args.p4_warmup)):
            _hybrid_step(model, x_batch, y_batch, kan_states, cfg)
            _manual_mlp_step([W1, W2], x_batch, y_batch, mlp_states, cfg)
        _sync(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        sums_kan = {"forward": 0.0, "backward": 0.0, "update": 0.0, "step": 0.0}
        for _ in range(int(args.p4_reps)):
            _loss, times = _hybrid_step(model, x_batch, y_batch, kan_states, cfg)
            for k in sums_kan:
                sums_kan[k] += times[k]
        _sync(device)
        peak_kan = float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)) if device.type == "cuda" else _estimate_hybrid_memory_mb(model, int(args.p4_batch_size))
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        sums_mlp = {"forward": 0.0, "backward": 0.0, "update": 0.0, "step": 0.0}
        for _ in range(int(args.p4_reps)):
            _loss, times = _manual_mlp_step([W1, W2], x_batch, y_batch, mlp_states, cfg)
            for k in sums_mlp:
                sums_mlp[k] += times[k]
        _sync(device)
        peak_mlp = float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)) if device.type == "cuda" else _estimate_mlp_memory_mb(input_dim, hidden, output_dim, int(args.p4_batch_size))
        reps = float(args.p4_reps)
        f_kan = sums_kan["forward"] * 1000.0 / reps
        b_kan = sums_kan["backward"] * 1000.0 / reps
        s_kan = sums_kan["step"] * 1000.0 / reps
        f_mlp = sums_mlp["forward"] * 1000.0 / reps
        b_mlp = sums_mlp["backward"] * 1000.0 / reps
        s_mlp = sums_mlp["step"] * 1000.0 / reps
        mlp_flops = 2 * input_dim * hidden + 2 * hidden * output_dim
        mlp_backward_flops = 4 * input_dim * hidden + 4 * hidden * output_dim
        kan_flops = model.forward_flops()
        kan_backward_flops = model.backward_flops()
        forward_ratio = f_kan / max(f_mlp, 1.0e-12)
        backward_ratio = b_kan / max(b_mlp, 1.0e-12)
        step_ratio = s_kan / max(s_mlp, 1.0e-12)
        memory_ratio = peak_kan / max(peak_mlp, 1.0e-12)
        params_ratio = params_kan / max(params_mlp, 1)
        flops_ratio = kan_flops / max(mlp_flops, 1)
        backward_flops_ratio = kan_backward_flops / max(mlp_backward_flops, 1)
        pass_flag = int(
            abs(params_ratio - 1.0) <= 0.05
            and forward_ratio <= 1.25
            and backward_ratio <= 1.50
            and step_ratio <= 1.50
            and memory_ratio <= 1.05
            and flops_ratio <= 1.05
            and backward_flops_ratio <= 1.50
        )
        row = {
            "stage": "P4_KERNEL_NATIVE_FEASIBILITY",
            "candidate_id": candidate_id,
            "source_id": source_id,
            "basis_id": basis_id,
            "parameterization_id": param_id,
            "matched_mlp_id": f"ManualMLP-hidden{hidden}",
            "params_kan": params_kan,
            "params_mlp_match": params_mlp,
            "params_ratio_vs_mlp_match": params_ratio,
            "materializes_dense_edge_tensor": 0,
            "materialized_tensor_shape": "not_materialized",
            "materialized_MB": 0.0,
            "forward_time_ms_kan": f_kan,
            "forward_time_ms_mlp_match": f_mlp,
            "backward_time_ms_kan": b_kan,
            "backward_time_ms_mlp_match": b_mlp,
            "step_time_ms_kan": s_kan,
            "step_time_ms_mlp_match": s_mlp,
            "peak_memory_MB_kan": peak_kan,
            "peak_memory_MB_mlp_match": peak_mlp,
            "activation_cache_MB_kan": _estimate_hybrid_memory_mb(model, int(args.p4_batch_size)),
            "activation_cache_MB_mlp_match": _estimate_mlp_memory_mb(input_dim, hidden, output_dim, int(args.p4_batch_size)),
            "optimizer_state_MB_kan": params_kan * 2 * 4 / (1024.0 * 1024.0),
            "optimizer_state_MB_mlp_match": params_mlp * 2 * 4 / (1024.0 * 1024.0),
            "functional_update_time_ms": 0.0,
            "forward_ratio_vs_mlp_match": forward_ratio,
            "backward_ratio_vs_mlp_match": backward_ratio,
            "step_ratio_vs_mlp_match": step_ratio,
            "memory_ratio_vs_mlp_match": memory_ratio,
            "forward_FLOPs_ratio": flops_ratio,
            "backward_FLOPs_ratio": backward_flops_ratio,
            "kernel_count_total": "not_measured",
            "small_kernel_count": "not_measured",
            "basis_eval_time": "not_measured_separately",
            "residual_projection_time": "not_measured_separately",
            "output_projection_time": "not_measured_separately",
            "kernelization_required": int(pass_flag == 0),
            "kernel_native_pass": pass_flag,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
        bench_rows.append(row)
        if not pass_flag:
            kernel_rows.append({
                "stage": "P4_KERNELIZATION_REQUIRED",
                "candidate_id": candidate_id,
                "source_id": source_id,
                "basis_id": basis_id,
                "reason": "P4_forward_backward_step_memory_or_flops_gate_failed",
                "forward_ratio_vs_mlp_match": forward_ratio,
                "backward_ratio_vs_mlp_match": backward_ratio,
                "step_ratio_vs_mlp_match": step_ratio,
                "memory_ratio_vs_mlp_match": memory_ratio,
                "forward_FLOPs_ratio": flops_ratio,
                "backward_FLOPs_ratio": backward_flops_ratio,
                "next_required_implementation": "torch_compile_or_triton_fused_basis_eval_residual_projection_backward",
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            })
    if not kernel_rows:
        kernel_rows.append({
            "stage": "P4_KERNELIZATION_REQUIRED",
            "status": "not_required",
            "reason": "all_P4_candidates_passed_kernel_native_gate",
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    write_csv_rows(out_dir / "materialization_free_microbench.csv", bench_rows)
    write_csv_rows(out_dir / "kernel_native_feasibility_vs_mlp.csv", bench_rows)
    write_csv_rows(out_dir / "kernelization_required_candidates.csv", kernel_rows)
    return bench_rows, kernel_rows


def _write_not_run_late_artifacts(out_dir: Path, reason: str) -> None:
    rows = [{
        "stage": "P5_P6_P7",
        "status": "not_run",
        "reason": reason,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }]
    for name in [
        "adamw_trainability_task.csv",
        "adamw_trainability_trace.csv",
        "functional_causality_v92.csv",
        "functional_event_trace.csv",
        "kanbefair_external_validation.csv",
    ]:
        write_csv_rows(out_dir / name, rows)


def _write_not_run_p6_p7_artifacts(out_dir: Path, reason: str) -> None:
    rows = [{
        "stage": "P6_P7",
        "status": "not_run",
        "reason": reason,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }]
    for name in [
        "functional_causality_v92.csv",
        "functional_event_trace.csv",
        "kanbefair_external_validation.csv",
    ]:
        write_csv_rows(out_dir / name, rows)


def _p4_metric_by_candidate(p4_rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {str(r.get("candidate_id", "")): dict(r) for r in p4_rows if str(r.get("status", "")) != "not_run"}


def _train_p5_hybrid_active_candidate(
    model: ModelT,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    *,
    batch_size: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    seed: int,
    device: torch.device,
) -> Tuple[List[Dict[str, Any]], float]:
    cfg = ManualAdamWConfig(lr=lr, weight_decay=weight_decay)
    states = [AdamWState.zeros_like(p) for p in model.parameters()]
    trace: List[Dict[str, Any]] = []
    gen = torch.Generator(device=device).manual_seed(int(seed))
    steps_per_epoch = max(1, int(x_train.shape[0]) // int(batch_size))
    n_used = steps_per_epoch * int(batch_size)
    x_used = x_train[:n_used]
    y_used = y_train[:n_used]
    source_id = model.layer1.source.source_id
    if model.layer1.basis.kind == "chebyshev3" and hasattr(model.layer1, "M") and getattr(model.layer1, "active_basis_indices", None) == (2,):
        cbwd = _maybe_compile("v92_cheb_active2_direct_fwd_bwd", _cheb_active2_model_fwd_bwd_direct_compiled_core)
        kan_args = (
            model.layer1.W0, model.layer1.M, model.layer1.V,
            model.layer1.source.mu.to(device), model.layer1.source.std.to(device),
            model.layer2.W0, model.layer2.M, model.layer2.V,
            model.layer2.source.mu.to(device), model.layer2.source.std.to(device),
            float(model.layer1.source.clip), float(model.layer1.source.out_div),
        )

        def step_batch(xb: torch.Tensor, yb: torch.Tensor) -> torch.Tensor:
            pack = cbwd(xb, yb, *kan_args)
            _adamw_update_foreach_(model.parameters(), pack[1:], states, cfg)
            return pack[0]
    elif model.layer1.basis.kind == "chebyshev3" and hasattr(model.layer1, "M") and getattr(model.layer1, "active_basis_indices", None) == (1,):
        cbwd = _maybe_compile("v92_cheb_active1_direct_fwd_bwd", _cheb_active1_model_fwd_bwd_direct_compiled_core)
        kan_args = (
            model.layer1.W0, model.layer1.M, model.layer1.V,
            model.layer1.source.mu.to(device), model.layer1.source.std.to(device),
            model.layer2.W0, model.layer2.M, model.layer2.V,
            model.layer2.source.mu.to(device), model.layer2.source.std.to(device),
            float(model.layer1.source.clip), float(model.layer1.source.out_div),
        )

        def step_batch(xb: torch.Tensor, yb: torch.Tensor) -> torch.Tensor:
            pack = cbwd(xb, yb, *kan_args)
            _adamw_update_foreach_(model.parameters(), pack[1:], states, cfg)
            return pack[0]
    else:
        def step_batch(xb: torch.Tensor, yb: torch.Tensor) -> torch.Tensor:
            logits, cache = model.forward(xb)
            loss, grad_logits = ce_loss_and_grad(logits, yb)
            grads = model.backward(grad_logits, cache)
            _adamw_update_foreach_(model.parameters(), grads, states, cfg)
            return loss

    _sync(device)
    started = time.perf_counter()
    for epoch in range(1, int(epochs) + 1):
        perm = torch.randperm(n_used, device=device, generator=gen)
        loss_accum = 0.0
        for batch_idx in range(steps_per_epoch):
            idx = perm[batch_idx * int(batch_size) : (batch_idx + 1) * int(batch_size)]
            loss = step_batch(x_used[idx], y_used[idx])
            loss_accum += float(loss.detach().cpu())
        with torch.no_grad():
            logits, _ = model.forward(x_used[: min(2048, n_used)])
            train_acc = float((logits.argmax(dim=1) == y_used[: min(2048, n_used)]).float().mean().detach().cpu())
        trace.append({
            "epoch": epoch,
            "train_loss": loss_accum / steps_per_epoch,
            "train_acc_head2048": train_acc,
            "source_id": source_id,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    _sync(device)
    return trace, time.perf_counter() - started


def _train_p5_mlp_match(
    input_dim: int,
    output_dim: int,
    hidden: int,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    *,
    batch_size: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    seed: int,
    device: torch.device,
) -> Tuple[List[torch.Tensor], List[Dict[str, Any]], float]:
    torch.manual_seed(int(seed) + 1009)
    W1 = torch.randn(input_dim, hidden, device=device) / math.sqrt(input_dim)
    W2 = torch.randn(hidden, output_dim, device=device) / math.sqrt(hidden)
    W1.requires_grad_(False)
    W2.requires_grad_(False)
    params = [W1, W2]
    states = [AdamWState.zeros_like(W1), AdamWState.zeros_like(W2)]
    cfg = ManualAdamWConfig(lr=lr, weight_decay=weight_decay)
    mbwd = _maybe_compile("v92_mlp_fwd_bwd", _mlp_fwd_bwd_compiled_core)
    gen = torch.Generator(device=device).manual_seed(int(seed) + 2027)
    steps_per_epoch = max(1, int(x_train.shape[0]) // int(batch_size))
    n_used = steps_per_epoch * int(batch_size)
    x_used = x_train[:n_used]
    y_used = y_train[:n_used]
    trace: List[Dict[str, Any]] = []
    _sync(device)
    started = time.perf_counter()
    for epoch in range(1, int(epochs) + 1):
        perm = torch.randperm(n_used, device=device, generator=gen)
        loss_accum = 0.0
        for batch_idx in range(steps_per_epoch):
            idx = perm[batch_idx * int(batch_size) : (batch_idx + 1) * int(batch_size)]
            pack = mbwd(x_used[idx], y_used[idx], W1, W2)
            _adamw_update_foreach_(params, pack[1:], states, cfg)
            loss_accum += float(pack[0].detach().cpu())
        with torch.no_grad():
            logits = F.silu(x_used[: min(2048, n_used)] @ W1) @ W2
            train_acc = float((logits.argmax(dim=1) == y_used[: min(2048, n_used)]).float().mean().detach().cpu())
        trace.append({
            "epoch": epoch,
            "train_loss": loss_accum / steps_per_epoch,
            "train_acc_head2048": train_acc,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    _sync(device)
    return params, trace, time.perf_counter() - started


def run_p5(
    args: argparse.Namespace,
    out_dir: Path,
    p4_rows: Sequence[Dict[str, Any]],
    device: torch.device,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    p4_pass_rows = [r for r in p4_rows if str(r.get("status", "")) != "not_run" and int(r.get("kernel_native_pass", 0)) == 1]
    if not p4_pass_rows and bool(getattr(args, "run_p5_even_if_p4_fail_for_diagnostic", False)):
        p4_pass_rows = [r for r in p4_rows if str(r.get("status", "")) != "not_run"]
    if not p4_pass_rows or not bool(getattr(args, "run_p5_trainability", False)):
        reason = "P4_has_no_kernel_native_survivor_so_P5_not_opened" if not p4_pass_rows else "P5_trainability_not_enabled"
        rows = [{
            "stage": "P5_ADAMW_TRAINABILITY",
            "status": "not_run",
            "reason": reason,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }]
        write_csv_rows(out_dir / "adamw_trainability_task.csv", rows)
        write_csv_rows(out_dir / "adamw_trainability_trace.csv", rows)
        return rows, rows, {"p5_pass": False, "reason": reason, "rows": []}

    task_rows: List[Dict[str, Any]] = []
    trace_rows: List[Dict[str, Any]] = []
    best = p4_pass_rows[0]
    candidate_id = str(best["candidate_id"])
    source_id, basis_id, param_id, _functional = candidate_id.split("-")
    for dataset in [_canonical_task(x) for x in _parse_list(args.p5_datasets)]:
        for seed in [int(s) for s in _parse_list(args.p5_seeds)]:
            torch.manual_seed(seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(seed)
            x_train, y_train, x_test, y_test, input_dim, output_dim, protocol = _load_task(
                args,
                dataset,
                train_size=int(args.p5_train_size),
                test_size=int(args.p5_test_size),
            )
            x_train = x_train.to(device=device, dtype=torch.float32)
            y_train = y_train.to(device=device)
            x_test = x_test.to(device=device, dtype=torch.float32)
            y_test = y_test.to(device=device)
            model = _build_hybrid_model_from_ids(
                source_id,
                basis_id,
                param_id,
                x_train,
                input_dim,
                output_dim,
                int(args.p3_hidden_dim),
                int(args.p3_rank),
                device,
                active_cheb_index=int(args.p4_active_cheb_index),
            )
            hidden_match = int(str(best.get("matched_mlp_id", "hidden0")).split("hidden")[-1])
            mlp_params, mlp_trace, mlp_time = _train_p5_mlp_match(
                input_dim,
                output_dim,
                hidden_match,
                x_train,
                y_train,
                batch_size=int(args.batch_size),
                epochs=int(args.p5_epochs),
                lr=float(args.p5_lr),
                weight_decay=float(args.p5_weight_decay),
                seed=seed,
                device=device,
            )
            kan_trace, kan_time = _train_p5_hybrid_active_candidate(
                model,
                x_train,
                y_train,
                batch_size=int(args.batch_size),
                epochs=int(args.p5_epochs),
                lr=float(args.p5_lr),
                weight_decay=float(args.p5_weight_decay),
                seed=seed,
                device=device,
            )
            mlp_metrics = _eval_mlp_params(mlp_params, x_test, y_test, int(args.p5_eval_batch_size))
            kan_metrics = _eval_hybrid_model(model, x_test, y_test, int(args.p5_eval_batch_size))
            train_head = min(2048, int(x_train.shape[0]))
            mlp_train_metrics = _eval_mlp_params(mlp_params, x_train[:train_head], y_train[:train_head], int(args.p5_eval_batch_size))
            kan_train_metrics = _eval_hybrid_model(model, x_train[:train_head], y_train[:train_head], int(args.p5_eval_batch_size))
            min_pass = int(kan_metrics["acc"] >= mlp_metrics["acc"] - P5_MIN_TRAINABILITY_TOL)
            strong_pass = int(kan_metrics["acc"] >= mlp_metrics["acc"])
            row = {
                "stage": "P5_ADAMW_TRAINABILITY",
                "candidate_id": candidate_id,
                "dataset": dataset,
                "seed": seed,
                "protocol": protocol,
                "train_size": int(args.p5_train_size),
                "test_size": int(args.p5_test_size),
                "epochs": int(args.p5_epochs),
                "mlp_match_acc": mlp_metrics["acc"],
                "mlp_match_loss": mlp_metrics["loss"],
                "mlp_match_ECE": mlp_metrics["ECE"],
                "mlp_match_NLL": mlp_metrics["NLL"],
                "val_acc": kan_metrics["acc"],
                "test_acc": kan_metrics["acc"],
                "val_loss": kan_metrics["loss"],
                "test_loss": kan_metrics["loss"],
                "ECE": kan_metrics["ECE"],
                "NLL": kan_metrics["NLL"],
                "CE_p50": kan_metrics["CE_p50"],
                "CE_p90": kan_metrics["CE_p90"],
                "CE_p99": kan_metrics["CE_p99"],
                "CE_p99_over_p50": kan_metrics["CE_p99_over_p50"],
                "logit_norm_mean": kan_metrics["logit_norm_mean"],
                "logit_norm_p95": kan_metrics["logit_norm_p95"],
                "correct_logit_mean": kan_metrics["correct_logit_mean"],
                "top_wrong_logit_mean": kan_metrics["top_wrong_logit_mean"],
                "correct_margin_mean": kan_metrics["correct_margin_mean"],
                "correct_margin_p10": kan_metrics["correct_margin_p10"],
                "wrong_confidence_p95": kan_metrics["wrong_confidence_p95"],
                "mlp_CE_p50": mlp_metrics["CE_p50"],
                "mlp_CE_p90": mlp_metrics["CE_p90"],
                "mlp_CE_p99": mlp_metrics["CE_p99"],
                "mlp_logit_norm_mean": mlp_metrics["logit_norm_mean"],
                "mlp_correct_margin_p10": mlp_metrics["correct_margin_p10"],
                "train_head_size": train_head,
                "train_acc_head2048_final": kan_train_metrics["acc"],
                "train_loss_head2048_final": kan_train_metrics["loss"],
                "train_CE_p50_head2048": kan_train_metrics["CE_p50"],
                "train_CE_p90_head2048": kan_train_metrics["CE_p90"],
                "train_CE_p99_head2048": kan_train_metrics["CE_p99"],
                "train_CE_p99_over_p50_head2048": kan_train_metrics["CE_p99_over_p50"],
                "train_correct_margin_p10_head2048": kan_train_metrics["correct_margin_p10"],
                "train_wrong_confidence_p95_head2048": kan_train_metrics["wrong_confidence_p95"],
                "train_logit_norm_mean_head2048": kan_train_metrics["logit_norm_mean"],
                "mlp_train_acc_head2048_final": mlp_train_metrics["acc"],
                "mlp_train_loss_head2048_final": mlp_train_metrics["loss"],
                "mlp_train_CE_p99_head2048": mlp_train_metrics["CE_p99"],
                "mlp_train_correct_margin_p10_head2048": mlp_train_metrics["correct_margin_p10"],
                "mlp_train_logit_norm_mean_head2048": mlp_train_metrics["logit_norm_mean"],
                "margin_p10": kan_metrics["margin_p10"],
                "hard_sample_acc": kan_metrics["hard_sample_acc"],
                "delta_vs_mlp_match": kan_metrics["acc"] - mlp_metrics["acc"],
                "minimum_trainability_pass": min_pass,
                "strong_trainability_pass": strong_pass,
                "params_ratio": best.get("params_ratio_vs_mlp_match", ""),
                "forward_FLOPs_ratio": best.get("forward_FLOPs_ratio", ""),
                "backward_FLOPs_ratio": best.get("backward_FLOPs_ratio", ""),
                "step_ratio": best.get("step_ratio_vs_mlp_match", ""),
                "memory_ratio": best.get("memory_ratio_vs_mlp_match", ""),
                "kan_train_time_s": kan_time,
                "mlp_train_time_s": mlp_time,
                "loss_type": "CE",
                "label_smoothing": 0,
                "uses_loss_backward": 0,
                "external_teacher_used": 0,
                "self_teacher_used": 0,
                "geometry_loss_used": 0,
                "sampler_changed": 0,
                "class_weight_used": 0,
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            }
            task_rows.append(row)
            for tr in kan_trace:
                trace_rows.append({"stage": "P5_ADAMW_TRACE", "candidate_id": candidate_id, "dataset": dataset, "seed": seed, "model": "KAN", **tr})
            for tr in mlp_trace:
                trace_rows.append({"stage": "P5_ADAMW_TRACE", "candidate_id": candidate_id, "dataset": dataset, "seed": seed, "model": "MLP-match", **tr})
    write_csv_rows(out_dir / "adamw_trainability_task.csv", task_rows)
    write_csv_rows(out_dir / "adamw_trainability_trace.csv", trace_rows)
    p5_pass = bool(task_rows) and all(int(r["minimum_trainability_pass"]) == 1 for r in task_rows)
    return task_rows, trace_rows, {"p5_pass": p5_pass, "rows": task_rows, "candidate_id": candidate_id}



def _route_decision(
    contract_rows: Sequence[Dict[str, Any]],
    p1_rows: Sequence[Dict[str, Any]],
    p1_pairs: Sequence[Tuple[str, str]],
    p2_survivors: Sequence[Tuple[str, str]],
    grad_rows: Sequence[Dict[str, Any]],
    overfit_rows: Sequence[Dict[str, Any]],
    p4_rows: Sequence[Dict[str, Any]],
    p5_rows: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    contract_pass = bool(contract_rows) and all(int(r.get("contract_pass", 0)) == 1 for r in contract_rows)
    p3_pass_ids = [
        str(r.get("candidate_id", ""))
        for r in overfit_rows
        if str(r.get("status", "")) != "not_run" and int(r.get("OverfitPass", 0)) == 1
    ]
    p4_pass_ids = [
        str(r.get("candidate_id", ""))
        for r in p4_rows
        if str(r.get("status", "")) != "not_run" and int(r.get("kernel_native_pass", 0)) == 1
    ]
    p5_measured = [r for r in p5_rows if str(r.get("status", "")) != "not_run" and str(r.get("stage", "")) == "P5_ADAMW_TRAINABILITY"]
    p5_min_pass_ids = [
        str(r.get("candidate_id", ""))
        for r in p5_measured
        if int(r.get("minimum_trainability_pass", 0)) == 1
    ]
    p5_all_pass = bool(p5_measured) and all(int(r.get("minimum_trainability_pass", 0)) == 1 for r in p5_measured)
    if not contract_pass:
        route = "R0-ContractViolation"
        blocker = "contract_audit_failed"
        next_impl = "fix_contract_violations_before_any_experiment"
    elif not p1_pairs:
        route = "R3-BasisSourceConditioningFail"
        blocker = "no_source_basis_conditioning_pass"
        next_impl = "redesign_feature_source_patch_or_conv_source"
    elif not p2_survivors:
        route = "R3-BasisFitFail"
        blocker = "conditioning_passed_but_one_layer_fit_failed"
        next_impl = "redesign_basis_family_or_fit_gate"
    elif not p3_pass_ids:
        route = "R4-AdamWTrainabilityFail"
        blocker = "hybrid_residual_gradcheck_or_512_overfit_failed"
        next_impl = "repair_hybrid_residual_parameterization_or_trainability"
    elif not p4_pass_ids:
        route = "R6-ComputeFail-KernelizationRequired"
        blocker = "P4_kernel_native_forward_backward_memory_or_flops_gate_failed"
        next_impl = "implement_torch_compile_or_triton_kernel_native_forward_backward_then_rerun_P3_P4"
    elif p5_measured and not p5_all_pass:
        route = "R4-AdamWTrainabilityFail"
        blocker = "P5_adamw_only_trainability_failed"
        next_impl = "repair_active_k_capacity_or_optimizer_protocol_before_functional_update"
    elif p5_all_pass:
        route = "R2-HybridResidualFullEdgeNearPass"
        blocker = "P5_passed_but_P6_P7_not_executed_in_current_runner"
        next_impl = "run_P6_residual_functional_controls_then_P7_external_fair_validation"
    else:
        route = "R2-HybridResidualFullEdgeNearPass"
        blocker = "P4_passed_but_P5_P7_not_executed_in_first_wave_runner"
        next_impl = "run_P5_adamw_trainability_then_P6_functional_and_P7_external"
    best_candidate = p4_pass_ids[0] if p4_pass_ids else (p3_pass_ids[0] if p3_pass_ids else "")
    return {
        "route": route,
        "best_candidate": best_candidate,
        "best_source": best_candidate.split("-")[0] if best_candidate else "",
        "best_basis": best_candidate.split("-")[1] if best_candidate else "",
        "best_parameterization": best_candidate.split("-")[2] if best_candidate else "",
        "best_functional_mode": "F0-NoFunctional" if best_candidate else "",
        "primary_blocker": blocker,
        "next_required_implementation": next_impl,
        "success_v92_basis_source": int(bool(p1_pairs and p2_survivors and p3_pass_ids)),
        "success_v92_full_edge_trainability": int(p5_all_pass),
        "success_v92_functional_advantage": 0,
        "success_v92_external_fair": 0,
        "contract_pass": int(contract_pass),
        "p1_source_basis_pass_count": len(p1_pairs),
        "p2_source_basis_survivor_count": len(p2_survivors),
        "p3_overfit_pass_count": len(p3_pass_ids),
        "p4_kernel_native_pass_count": len(p4_pass_ids),
        "p5_min_trainability_pass_count": len(p5_min_pass_ids),
        "p5_trainability_row_count": len(p5_measured),
    }


def _failure_rows(route: Dict[str, Any], p1_rows: Sequence[Dict[str, Any]], p4_rows: Sequence[Dict[str, Any]], p5_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if int(route["p1_source_basis_pass_count"]) == 0:
        rows.append({"stage": "P1", "failure_code": "F3_conditioning_fail", "reason": "no source-basis pair passed all datasets/seeds"})
    if str(route["route"]).startswith("R6"):
        for row in p4_rows:
            if str(row.get("status", "")) == "not_run":
                continue
            if int(row.get("kernel_native_pass", 0)) == 0:
                rows.append({
                    "stage": "P4",
                    "candidate_id": row.get("candidate_id", ""),
                    "failure_code": "F7b_kernelization_required",
                    "forward_ratio_vs_mlp_match": row.get("forward_ratio_vs_mlp_match", ""),
                    "backward_ratio_vs_mlp_match": row.get("backward_ratio_vs_mlp_match", ""),
                    "step_ratio_vs_mlp_match": row.get("step_ratio_vs_mlp_match", ""),
                    "memory_ratio_vs_mlp_match": row.get("memory_ratio_vs_mlp_match", ""),
                    "forward_FLOPs_ratio": row.get("forward_FLOPs_ratio", ""),
                    "backward_FLOPs_ratio": row.get("backward_FLOPs_ratio", ""),
                    "reason": "kernel-native feasibility gate failed; full task not opened",
                })
    if str(route["route"]) == "R4-AdamWTrainabilityFail":
        for row in p5_rows:
            if str(row.get("stage", "")) == "P5_ADAMW_TRAINABILITY" and int(row.get("minimum_trainability_pass", 0)) == 0:
                rows.append({
                    "stage": "P5",
                    "candidate_id": row.get("candidate_id", ""),
                    "dataset": row.get("dataset", ""),
                    "seed": row.get("seed", ""),
                    "failure_code": "F8_adamw_trainability_fail",
                    "test_acc": row.get("test_acc", ""),
                    "mlp_match_acc": row.get("mlp_match_acc", ""),
                    "delta_vs_mlp_match": row.get("delta_vs_mlp_match", ""),
                    "reason": "KAN AdamW-only accuracy below MLP-match minus tolerance",
                })
    if not rows:
        rows.append({"stage": "P8", "failure_code": "none_or_downstream_not_opened", "reason": route["primary_blocker"]})
    for row in rows:
        row.setdefault("fake_data_used", 0)
        row.setdefault("proxy_row_used", 0)
        row.setdefault("cpu_offload_used", 0)
    return rows


def _boundary_rows(route: Dict[str, Any], p1_pairs: Sequence[Tuple[str, str]], p2_survivors: Sequence[Tuple[str, str]], overfit_rows: Sequence[Dict[str, Any]], p4_rows: Sequence[Dict[str, Any]], p5_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    p3_pass = {str(r.get("candidate_id", "")) for r in overfit_rows if int(r.get("OverfitPass", 0) or 0) == 1}
    p4_pass = {str(r.get("candidate_id", "")) for r in p4_rows if int(r.get("kernel_native_pass", 0) or 0) == 1}
    rows: List[Dict[str, Any]] = []
    candidates = sorted(p3_pass | p4_pass | {str(r.get("candidate_id", "")) for r in overfit_rows if str(r.get("candidate_id", ""))})
    if not candidates:
        candidates = ["global"]
    for cid in candidates:
        source = cid.split("-")[0] if cid != "global" else ""
        basis = cid.split("-")[1] if cid != "global" else ""
        labels: List[str] = []
        if cid == "global":
            labels.append(str(route["primary_blocker"]))
        else:
            if (source, basis) not in p1_pairs:
                labels.append("conditioning_not_passed")
            if (source, basis) not in p2_survivors:
                labels.append("fit_not_passed")
            if cid not in p3_pass:
                labels.append("gradcheck_or_overfit_not_passed")
            if cid in p3_pass and cid not in p4_pass:
                labels.append("kernel_native_gate_not_passed")
            if str(route["route"]) == "R4-AdamWTrainabilityFail":
                labels.append("adamw_trainability_not_passed")
        rows.append({
            "stage": "P8_BOUNDARY_AUDIT",
            "candidate_id": cid,
            "source_id": source,
            "basis_id": basis,
            "boundary_labels": ",".join(labels) if labels else "no_boundary",
            "route": route["route"],
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    return rows


def _copy_plan_hash_rows(out_dir: Path) -> List[Path]:
    return [SCRIPT_PATH, PLAN_PATH]


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
    parser.add_argument("--p1-test-size", type=int, default=128)
    parser.add_argument("--p1-max-values", type=int, default=20000)
    parser.add_argument("--p1-seeds", default="0,1,2")
    parser.add_argument("--force-source-basis-pairs", default="")
    parser.add_argument("--force-source-basis-pairs-only", action="store_true")
    parser.add_argument("--p2-grid-size", type=int, default=2048)
    parser.add_argument("--p3-train-size", type=int, default=1024)
    parser.add_argument("--p3-overfit-size", type=int, default=512)
    parser.add_argument("--p3-steps", type=int, default=600)
    parser.add_argument("--p3-hidden-dim", type=int, default=64)
    parser.add_argument("--p3-rank", type=int, default=16)
    parser.add_argument("--p3-parameterizations", default="P1,P3")
    parser.add_argument("--p4-active-cheb-index", type=int, default=2)
    parser.add_argument("--p3-lr", type=float, default=2.0e-3)
    parser.add_argument("--p4-batch-size", type=int, default=128)
    parser.add_argument("--p4-warmup", type=int, default=20)
    parser.add_argument("--p4-reps", type=int, default=120)
    parser.add_argument("--disable-p4-compiled-kernel-native-path", action="store_true")
    parser.add_argument("--run-p5-trainability", action="store_true")
    parser.add_argument("--run-p5-even-if-p4-fail-for-diagnostic", action="store_true")
    parser.add_argument("--p5-datasets", default="MNIST,Fashion-MNIST,KMNIST")
    parser.add_argument("--p5-seeds", default="0,1,2")
    parser.add_argument("--p5-train-size", type=int, default=9984)
    parser.add_argument("--p5-test-size", type=int, default=2000)
    parser.add_argument("--p5-epochs", type=int, default=5)
    parser.add_argument("--p5-lr", type=float, default=2.0e-3)
    parser.add_argument("--p5-weight-decay", type=float, default=0.0)
    parser.add_argument("--p5-eval-batch-size", type=int, default=512)
    args = parser.parse_args()

    torch.manual_seed(int(args.seed))
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    matmul_precision = "default"
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        matmul_precision = "high"

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    if args.fresh and out_dir.exists():
        shutil.rmtree(out_dir)
    ensure_dir(out_dir)
    ensure_dir(out_dir / "figures")

    write_json(out_dir / "run_manifest.json", {
        "created_utc": _now_iso(),
        "script": str(SCRIPT_PATH.relative_to(ROOT)),
        "plan": str(PLAN_PATH.relative_to(ROOT)),
        "device": str(device),
        "torch_float32_matmul_precision": matmul_precision,
        "args": vars(args),
        "contract": {
            "loss_type": "CE",
            "label_smoothing": 0,
            "external_teacher_used": 0,
            "self_teacher_used": 0,
            "geometry_loss_used": 0,
            "sampler_changed": 0,
            "class_weight_used": 0,
            "cpu_offload_used": 0,
            "uses_loss_backward_for_kan_path": 0,
            "fake_proxy_allowed": 0,
        },
    })

    registry, contract = run_p0(out_dir)
    p1_rows = run_p1(args, out_dir, device)
    p1_pairs = _passing_source_basis(p1_rows)
    forced_pairs = _parse_source_basis_pairs(str(getattr(args, "force_source_basis_pairs", "")))
    if forced_pairs:
        p1_pairs = sorted(set(forced_pairs) if bool(getattr(args, "force_source_basis_pairs_only", False)) else (set(p1_pairs) | set(forced_pairs)))
        write_csv_rows(out_dir / "forced_source_basis_pairs.csv", [
            {
                "stage": "FORCED_DIAGNOSTIC_SOURCE_BASIS_PAIR",
                "source_id": source,
                "basis_id": basis,
                "reason": "explicit diagnostic override; not counted as natural P1 conditioning pass",
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            }
            for source, basis in forced_pairs
        ])
    p2_rows = run_p2(args, out_dir, p1_pairs, device)
    p2_pairs = _p2_survivors(p2_rows)
    if forced_pairs:
        p2_pairs = sorted(set(forced_pairs) if bool(getattr(args, "force_source_basis_pairs_only", False)) else (set(p2_pairs) | set(forced_pairs)))
    grad_rows, overfit_rows, models = run_p3(args, out_dir, p2_pairs, device)
    p4_rows, kernel_rows = run_p4(args, out_dir, models, device)

    p4_pass = [r for r in p4_rows if int(r.get("kernel_native_pass", 0) or 0) == 1]
    if (p4_pass or bool(args.run_p5_even_if_p4_fail_for_diagnostic)) and bool(args.run_p5_trainability):
        p5_rows, p5_trace_rows, p5_summary = run_p5(args, out_dir, p4_rows, device)
        if bool(p5_summary.get("p5_pass", False)):
            _write_not_run_p6_p7_artifacts(out_dir, "P5_passed_but_P6_P7_not_enabled_in_current_runner")
        else:
            _write_not_run_p6_p7_artifacts(out_dir, "P5_adamw_trainability_failed_so_P6_P7_not_opened")
    elif p4_pass:
        _write_not_run_late_artifacts(out_dir, "P4_passed_but_P5_P7_trainability_runner_not_enabled_in_first_wave")
        p5_rows = []
    else:
        _write_not_run_late_artifacts(out_dir, "P4_has_no_kernel_native_survivor_so_P5_P6_P7_not_opened")
        p5_rows = []

    route = _route_decision(contract, p1_rows, p1_pairs, p2_pairs, grad_rows, overfit_rows, p4_rows, p5_rows)
    write_json(out_dir / "route_decision.json", route)
    write_json(out_dir / "aggregate_decision.json", route)
    failure_rows = _failure_rows(route, p1_rows, p4_rows, p5_rows)
    write_csv_rows(out_dir / "failure_table.csv", failure_rows)
    boundary_rows = _boundary_rows(route, p1_pairs, p2_pairs, overfit_rows, p4_rows, p5_rows)
    write_csv_rows(out_dir / "boundary_audit.csv", boundary_rows)

    audit_paths = [
        out_dir / "candidate_registry_v92.csv",
        out_dir / "contract_audit_v92.csv",
        out_dir / "basis_source_conditioning.csv",
        out_dir / "basis_fit_diagnostics.csv",
        out_dir / "hybrid_residual_gradcheck.csv",
        out_dir / "hybrid_residual_overfit.csv",
        out_dir / "materialization_free_microbench.csv",
        out_dir / "kernel_native_feasibility_vs_mlp.csv",
        out_dir / "kernelization_required_candidates.csv",
        out_dir / "adamw_trainability_task.csv",
        out_dir / "adamw_trainability_trace.csv",
        out_dir / "functional_causality_v92.csv",
        out_dir / "functional_event_trace.csv",
        out_dir / "kanbefair_external_validation.csv",
        out_dir / "boundary_audit.csv",
        out_dir / "failure_table.csv",
    ]
    if (out_dir / "forced_source_basis_pairs.csv").exists():
        audit_paths.append(out_dir / "forced_source_basis_pairs.csv")
    provenance = audit_no_fake(audit_paths)
    write_csv_rows(out_dir / "v92_provenance_audit.csv", [{"stage": "NO_FAKE_AUDIT", **provenance}])

    hash_targets = [p for p in _copy_plan_hash_rows(out_dir) if p.exists()]
    hash_targets += [p for p in out_dir.iterdir() if p.is_file()]
    write_csv_rows(out_dir / "artifact_hashes.csv", artifact_hash_rows(hash_targets, root=ROOT))

    print(json.dumps(route, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
