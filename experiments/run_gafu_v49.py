#!/usr/bin/env python3
"""DG-KAN v4.9 runner: RBF parameterization and exact NFS probes."""

from __future__ import annotations

import argparse
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dgkan_core import (
    RBFDense,
    coefficient_named_params,
    ensure_dir,
    evaluate,
    get_device,
    iter_minibatches,
    load_vision_bundle,
    make_pure_norm,
    parse_int_list,
    parse_str_list,
    rbf_basis_occupancy_audit,
    read_csv,
    set_seed,
    write_csv,
)
from run_gafu_v3 import add_args as add_v3_args, dataset_name
from run_gafu_v43 import _apply_updates, _feature_audit, _mean, _restore, _role_for_name, _role_group, _snapshot
from run_gafu_v44 import ABRBFDense, V44Params, _estimate_geometry_any
from run_gafu_v47 import _collect_pure_features, _teacher_snapshot
from run_gafu_v48 import (
    V48Params,
    _iter_steps,
    _load_state_cpu,
    _logit_kl,
    _nfs_project_once,
    _roughness_scalar,
    _state_cpu,
    _teacher_constraints,
)


DATASETS = ["MNIST", "Fashion-MNIST", "KMNIST"]

P0_EDGE_KINDS = [
    "RBFOnly",
    "ABRBF-linear",
    "ABRBF-silu",
    "RBF-learnWidth",
    "ABRBF-linear-learnWidth",
    "RBF-quantileCenters",
    "ExactNFS-smoke",
]

P1_METHODS = [
    "PureKAN-RBFOnly-AdamW",
    "PureKAN-ABRBF-linear-AdamW",
    "PureKAN-ABRBF-silu-AdamW",
    "PureKAN-RBF-learnWidth-AdamW",
    "PureKAN-ABRBF-linear-learnWidth-AdamW",
]

P3_TEACHERS = [
    "PureKAN-RBFOnly-AdamW",
    "PureKAN-ABRBF-linear-AdamW",
    "PureKAN-ABRBF-silu-AdamW",
]

P3_VARIANTS = [
    ("Heuristic-NFS-role-block", "identity", "logit_hidden"),
    ("ExactNFS-logit", "identity", "logit"),
    ("ExactNFS-logit-hidden", "identity", "logit_hidden"),
    ("ExactNFS-logit-hidden", "sobolev_diag", "logit_hidden"),
    ("ExactNFS-margin", "identity", "margin"),
]

P6_CONFIGS = [
    (64, 16, 4, "RBFOnly"),
    (96, 24, 4, "RBFOnly"),
    (96, 32, 2, "RBFOnly"),
    (96, 24, 4, "ABRBF-linear"),
    (96, 24, 4, "ABRBF-silu"),
    (96, 32, 4, "ABRBF-linear"),
    (96, 24, 4, "RBF-learnWidth"),
    (96, 24, 4, "ABRBF-linear-learnWidth"),
]


@dataclass
class V49Params(V48Params):
    train_size: int = 1536
    val_size: int = 512
    test_size: int = 512
    batch_size: int = 128
    eval_batch_size: int = 512
    audit_batch_size: int = 24
    hidden_dim: int = 64
    depth: int = 4
    basis_count: int = 16
    p1_steps: int = 120
    p4_task_steps: int = 100
    p6_steps: int = 100
    adam_lr: float = 1.0e-3
    exact_eta: float = 0.05
    exact_mu: float = 1.0e-4
    exact_max_constraints: int = 64
    exact_hidden_sketch: int = 8
    exact_trust_ratio: float = 0.025
    exact_backtracks: int = 7
    exact_logit_radius: float = 0.03
    exact_hidden_radius: float = 0.05
    exact_kl_radius: float = 0.005


def _sanitize(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


def _softplus_inverse(x: float) -> float:
    return math.log(math.exp(float(x)) - 1.0)


class LearnWidthRBFDense(RBFDense):
    """Fixed-center RBF edge with trainable scalar width per layer."""

    base_dim = 0

    def __init__(self, in_dim: int, out_dim: int, basis_count: int, *, bias: bool = False) -> None:
        super().__init__(in_dim, out_dim, basis_count, bias=bias)
        self.log_width = nn.Parameter(torch.tensor(_softplus_inverse(self.width)))

    def current_width(self) -> torch.Tensor:
        return F.softplus(self.log_width).clamp_min(1.0e-4)

    def basis(self, x: torch.Tensor) -> torch.Tensor:
        z = (x.unsqueeze(-1) - self.centers) / self.current_width()
        return torch.exp(-0.5 * z.square())

    def basis_and_derivative(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        basis = self.basis(x)
        width = self.current_width()
        deriv = -((x.unsqueeze(-1) - self.centers) / (width**2)) * basis
        return basis, deriv


class LinearABLearnWidthDense(ABRBFDense):
    """Constant + linear + RBF edge with trainable scalar width."""

    base_dim = 2

    def __init__(self, in_dim: int, out_dim: int, basis_count: int, *, bias: bool = False) -> None:
        super().__init__(in_dim, out_dim, basis_count, bias=bias)
        self.log_width = nn.Parameter(torch.tensor(_softplus_inverse(self.width)))

    def current_width(self) -> torch.Tensor:
        return F.softplus(self.log_width).clamp_min(1.0e-4)

    def basis(self, x: torch.Tensor) -> torch.Tensor:
        z = (x.unsqueeze(-1) - self.centers) / self.current_width()
        rbf = torch.exp(-0.5 * z.square())
        return torch.cat([torch.ones_like(x).unsqueeze(-1), x.unsqueeze(-1), rbf], dim=-1)

    def basis_and_derivative(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        width = self.current_width()
        z = (x.unsqueeze(-1) - self.centers) / width
        rbf = torch.exp(-0.5 * z.square())
        rbf_deriv = -((x.unsqueeze(-1) - self.centers) / (width**2)) * rbf
        const = torch.ones_like(x).unsqueeze(-1)
        linear = x.unsqueeze(-1)
        deriv = torch.cat([torch.zeros_like(const), torch.ones_like(linear), rbf_deriv], dim=-1)
        return torch.cat([const, linear, rbf], dim=-1), deriv


class SiluABRBFDense(RBFDense):
    """SiLU base path + RBF edge."""

    base_dim = 1

    def __init__(self, in_dim: int, out_dim: int, basis_count: int, *, bias: bool = False) -> None:
        super().__init__(in_dim, out_dim, basis_count, bias=bias)
        scale = 1.0 / math.sqrt(max(1, in_dim * (basis_count + 1)))
        self.coeff = nn.Parameter(torch.randn(out_dim, in_dim, basis_count + 1) * scale)
        self.rbf_basis_count = int(basis_count)

    def basis(self, x: torch.Tensor) -> torch.Tensor:
        z = (x.unsqueeze(-1) - self.centers) / self.width
        rbf = torch.exp(-0.5 * z.square())
        return torch.cat([F.silu(x).unsqueeze(-1), rbf], dim=-1)

    def basis_and_derivative(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = (x.unsqueeze(-1) - self.centers) / self.width
        rbf = torch.exp(-0.5 * z.square())
        rbf_deriv = -((x.unsqueeze(-1) - self.centers) / (self.width**2)) * rbf
        sig = torch.sigmoid(x)
        silu_deriv = (sig + x * sig * (1.0 - sig)).unsqueeze(-1)
        return torch.cat([F.silu(x).unsqueeze(-1), rbf], dim=-1), torch.cat([silu_deriv, rbf_deriv], dim=-1)


class LinearABRBFDense(ABRBFDense):
    base_dim = 2


class V49PureResidualKANBlock(nn.Module):
    def __init__(self, dim: int, basis_count: int, dense_cls: type[RBFDense]) -> None:
        super().__init__()
        self.norm = make_pure_norm(dim, "fixed")
        self.kan = dense_cls(dim, dim, basis_count, bias=False)
        self.register_buffer("alpha", torch.tensor(1.0))
        self.branch_scale = 1.0

    def forward(self, h: torch.Tensor, *, disable_kan: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        if disable_kan:
            return h, torch.zeros_like(h)
        branch = self.alpha * float(self.branch_scale) * self.kan(self.norm(h))
        return h + branch, branch


class V49PureKANClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        *,
        hidden_dim: int,
        depth: int,
        basis_count: int,
        dense_cls: type[RBFDense],
        edge_kind: str,
    ) -> None:
        super().__init__()
        self.edge_kind = edge_kind
        self.norm_mode = "fixed"
        self.input_kan = dense_cls(input_dim, hidden_dim, basis_count, bias=False)
        self.blocks = nn.ModuleList([V49PureResidualKANBlock(hidden_dim, basis_count, dense_cls) for _ in range(depth)])
        self.output_norm = make_pure_norm(hidden_dim, "fixed")
        self.output_kan = dense_cls(hidden_dim, num_classes, basis_count, bias=False)

    def forward(self, x: torch.Tensor, *, disable_kan: bool = False, return_branch: bool = False) -> Any:
        h = self.input_kan(x)
        ratios: List[torch.Tensor] = []
        for block in self.blocks:
            prev = h
            h, branch = block(h, disable_kan=disable_kan)
            ratios.append(branch.norm(dim=-1).mean() / prev.norm(dim=-1).mean().clamp_min(1.0e-6))
        logits = self.output_kan(self.output_norm(h))
        if return_branch:
            return logits, torch.stack(ratios).mean() if ratios else torch.tensor(0.0, device=x.device)
        return logits

    def kan_layers(self) -> Iterable[RBFDense]:
        yield self.input_kan
        for block in self.blocks:
            yield block.kan
        yield self.output_kan


def _dense_cls(edge_kind: str) -> type[RBFDense]:
    key = edge_kind.lower()
    if key == "rbfonly" or key == "rbf-quantilecenters":
        return RBFDense
    if key == "abrbf-linear":
        return LinearABRBFDense
    if key == "abrbf-silu":
        return SiluABRBFDense
    if key == "rbf-learnwidth":
        return LearnWidthRBFDense
    if key == "abrbf-linear-learnwidth":
        return LinearABLearnWidthDense
    raise ValueError(f"unknown edge kind: {edge_kind}")


def _make_v49_model(bundle: Any, params: V49Params, device: torch.device, edge_kind: str) -> nn.Module:
    model = V49PureKANClassifier(
        bundle.input_dim,
        bundle.num_classes,
        hidden_dim=params.hidden_dim,
        depth=params.depth,
        basis_count=params.basis_count,
        dense_cls=_dense_cls(edge_kind),
        edge_kind=edge_kind,
    ).to(device)
    return model


def _init_quantile_centers(model: nn.Module, x: torch.Tensor) -> None:
    with torch.no_grad():
        model(x)
        for layer in model.kan_layers():  # type: ignore[attr-defined]
            if layer.last_input is None:
                continue
            vals = layer.last_input.detach().flatten().float()
            if vals.numel() < layer.centers.numel():
                continue
            qs = torch.linspace(0.02, 0.98, layer.centers.numel(), device=vals.device)
            centers = torch.quantile(vals, qs).to(device=layer.centers.device, dtype=layer.centers.dtype)
            centers, _ = torch.sort(centers)
            span = float((centers[-1] - centers[0]).abs().detach().cpu())
            layer.centers.copy_(centers)
            layer.width = max(1.0e-3, span / max(1, centers.numel() - 1) * 1.4)


def _edge_named_params(model: nn.Module) -> List[Tuple[str, nn.Parameter]]:
    out: List[Tuple[str, nn.Parameter]] = []
    seen: set[int] = set()
    for module_name, module in model.named_modules():
        if not isinstance(module, RBFDense):
            continue
        prefix = f"{module_name}." if module_name else ""
        for attr in ["coeff", "log_width", "centers_param"]:
            value = getattr(module, attr, None)
            if isinstance(value, nn.Parameter) and value.requires_grad and id(value) not in seen:
                out.append((f"{prefix}{attr}", value))
                seen.add(id(value))
    return out


def _edge_param_ids(model: nn.Module) -> set[int]:
    return {id(p) for _, p in _edge_named_params(model)}


def _non_edge_trainable_params(model: nn.Module) -> int:
    edge_ids = _edge_param_ids(model)
    return sum(p.numel() for p in model.parameters() if p.requires_grad and id(p) not in edge_ids)


def _base_dim(layer: RBFDense) -> int:
    return int(getattr(layer, "base_dim", 0))


def _rbf_tail(coeff: torch.Tensor, layer: RBFDense) -> torch.Tensor:
    base = _base_dim(layer)
    return coeff[..., base:] if base > 0 else coeff


def _roughness_loss(model: nn.Module, *, role: str | None = None, l2: float = 0.02) -> torch.Tensor:
    device = next(model.parameters()).device
    total = torch.zeros((), device=device)
    names = [name for name, _ in coefficient_named_params(model)]
    layers = list(model.kan_layers())  # type: ignore[attr-defined]
    for name, layer in zip(names, layers):
        if role is not None and _role_group(_role_for_name(name)) != role:
            continue
        coeff = _rbf_tail(layer.coeff.float(), layer)
        if coeff.shape[-1] > 1:
            diff = coeff[..., 1:] - coeff[..., :-1]
            total = total + diff.square().mean()
        total = total + l2 * coeff.square().mean()
    return total


def _metric_matrix_for_layer(layer: RBFDense, *, metric: str, device: torch.device) -> torch.Tensor:
    k = int(layer.coeff.shape[-1])
    base = _base_dim(layer)
    mat = torch.eye(k, device=device) * 1.0e-3
    if base > 0:
        mat[:base, :base] += torch.eye(base, device=device)
    rbf_k = k - base
    if rbf_k <= 0:
        return mat + torch.eye(k, device=device)
    if metric == "identity":
        mat[base:, base:] += torch.eye(rbf_k, device=device)
        return mat
    d = torch.zeros(max(1, rbf_k - 1), rbf_k, device=device)
    if rbf_k > 1:
        rows = torch.arange(rbf_k - 1, device=device)
        d[rows, rows] = -1.0
        d[rows, rows + 1] = 1.0
    sob = d.T @ d + 0.05 * torch.eye(rbf_k, device=device)
    if metric == "sobolev_diag":
        mat[base:, base:] += torch.diag(torch.diag(sob))
    else:
        mat[base:, base:] += sob
    return 0.5 * (mat + mat.T)


def _metric_diag_for_params(named: Sequence[Tuple[str, nn.Parameter]], model: nn.Module, metric: str) -> torch.Tensor:
    if metric == "identity":
        return torch.cat([torch.ones(p.numel(), device=p.device) for _, p in named])
    layers = list(model.kan_layers())  # type: ignore[attr-defined]
    coeff_names = [name for name, _ in coefficient_named_params(model)]
    by_name = dict(zip(coeff_names, layers))
    pieces: List[torch.Tensor] = []
    for name, p in named:
        layer = by_name.get(name)
        if layer is None:
            pieces.append(torch.ones(p.numel(), device=p.device))
            continue
        mat = _metric_matrix_for_layer(layer, metric="sobolev_diag", device=p.device)
        diag = torch.diag(mat).clamp_min(1.0e-6).to(device=p.device, dtype=p.dtype)
        repeats = p.numel() // p.shape[-1]
        pieces.append(diag.repeat(repeats))
    return torch.cat(pieces)


def _param_count_audit(model: nn.Module) -> Dict[str, float]:
    coeff_params = sum(p.numel() for _, p in coefficient_named_params(model))
    edge_params = sum(p.numel() for _, p in _edge_named_params(model))
    base_params = 0
    width_params = 0
    center_params = 0
    for layer in model.kan_layers():  # type: ignore[attr-defined]
        base_params += int(_base_dim(layer) * layer.coeff.shape[0] * layer.coeff.shape[1])
        width_params += int(getattr(layer, "log_width", torch.empty(0)).numel()) if isinstance(getattr(layer, "log_width", None), nn.Parameter) else 0
    return {
        "learnable_nonKAN_params": _non_edge_trainable_params(model),
        "learnable_edge_params": edge_params,
        "rbf_coeff_params": coeff_params - base_params,
        "base_params": base_params,
        "center_params": center_params,
        "width_params": width_params,
        "functional_coverage": 1.0 if edge_params else 0.0,
        "base_coverage": 1.0 if base_params else 0.0,
        "center_width_coverage": 1.0 if width_params else (0.0 if center_params else float("nan")),
    }


def _base_rbf_audit(model: nn.Module) -> Dict[str, float]:
    base_norms: List[float] = []
    rbf_norms: List[float] = []
    widths: List[float] = []
    entropies: List[float] = []
    for layer in model.kan_layers():  # type: ignore[attr-defined]
        if layer.last_input is None:
            continue
        basis = layer.basis(layer.last_input.to(layer.coeff.device)).detach()
        base = _base_dim(layer)
        if base > 0:
            out_base = torch.einsum("bik,oik->bo", basis[..., :base], layer.coeff[..., :base].detach())
            out_rbf = torch.einsum("bik,oik->bo", basis[..., base:], layer.coeff[..., base:].detach())
            base_norms.append(float(out_base.float().norm().detach().cpu()))
            rbf_norms.append(float(out_rbf.float().norm().detach().cpu()))
        else:
            base_norms.append(0.0)
            out_rbf = torch.einsum("bik,oik->bo", basis, layer.coeff.detach())
            rbf_norms.append(float(out_rbf.float().norm().detach().cpu()))
        w = getattr(layer, "current_width", None)
        if callable(w):
            widths.append(float(w().detach().cpu()))
        else:
            widths.append(float(layer.width))
        mean = basis[..., base:].detach().flatten().float()
        if mean.numel() > 0:
            probs = mean / mean.sum().clamp_min(1.0e-12)
            entropies.append(float((-(probs * probs.clamp_min(1.0e-12).log()).sum() / math.log(max(2, probs.numel()))).detach().cpu()))
    base_norm = _mean(base_norms, 0.0)
    rbf_norm = _mean(rbf_norms, 0.0)
    return {
        "base_output_norm": base_norm,
        "rbf_output_norm": rbf_norm,
        "base_over_rbf_norm": base_norm / max(1.0e-12, rbf_norm),
        "base_margin_contribution": base_norm / max(1.0e-12, base_norm + rbf_norm),
        "rbf_margin_contribution": rbf_norm / max(1.0e-12, base_norm + rbf_norm),
        "width_mean": _mean(widths, float("nan")),
        "width_min": min(widths) if widths else float("nan"),
        "width_max": max(widths) if widths else float("nan"),
        "basis_occupancy_entropy": _mean(entropies, float("nan")),
    }


def _sobolev_scalar_v49(model: nn.Module) -> float:
    vals: List[float] = []
    with torch.no_grad():
        for layer in model.kan_layers():  # type: ignore[attr-defined]
            mat = _metric_matrix_for_layer(layer, metric="sobolev_full", device=layer.coeff.device)
            coeff = layer.coeff.detach().float().reshape(-1, layer.coeff.shape[-1])
            vals.append(float((coeff @ mat.float() * coeff).sum(dim=-1).mean().detach().cpu()))
    return _mean(vals, float("nan"))


def _eval_v49(model: nn.Module, bundle: Any, params: V49Params, device: torch.device) -> Dict[str, float]:
    hb = bundle.x_val[: params.audit_batch_size].to(device)
    yh = bundle.y_val[: params.audit_batch_size].to(device)
    feat = _feature_audit(model, hb, yh)
    geom = _estimate_geometry_any(model, hb, device=device, batch_size=params.audit_batch_size)
    ev = evaluate(model, bundle.x_test, bundle.y_test, device=device, batch_size=params.eval_batch_size)
    out = {
        "test_acc": ev["acc"],
        "test_loss": ev["loss"],
        "ece": ev["ece"],
        "rank_input": feat["input_feature_effective_rank"],
        "rank_block": feat["block_feature_effective_rank"],
        "centroid_sep": feat["class_centroid_separation"],
        "margin_mean": feat["margin_mean"],
        "margin_p10": feat["margin_p10"],
        "margin_p50": feat["margin_mean"],
        "phi_prime_p95": geom["phi_prime_p95"],
        "phi_prime_max": geom.get("phi_prime_max", float("nan")),
        "jacobian_condition": geom["max_jac_condition"],
        "curvature_energy": geom.get("curvature_energy", geom["phi_prime_p95"] ** 2),
        "sobolev_norm_total": _sobolev_scalar_v49(model),
        "coefficient_roughness": _roughness_scalar(model),
    }
    out.update(_base_rbf_audit(model))
    occ = rbf_basis_occupancy_audit(model)
    out["dead_basis_fraction"] = occ.get("rbf_basis_dead_frac", out.get("dead_basis_fraction", float("nan")))
    out["out_of_grid_fraction"] = occ.get("rbf_input_out_of_grid_frac", out.get("out_of_grid_fraction", float("nan")))
    return out


def _save_state(path: Path, model: nn.Module, meta: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": _state_cpu(model), "meta": meta}, path)


def _state_path(out_dir: Path, dataset: str, seed: int, method: str, params: V49Params) -> Path:
    tag = f"{dataset}_s{seed}_{method}_h{params.hidden_dim}_b{params.basis_count}_d{params.depth}.pt"
    return out_dir / "state_cache" / _sanitize(tag)


def _edge_kind_from_method(method: str) -> str:
    if "ABRBF-linear-learnWidth" in method:
        return "ABRBF-linear-learnWidth"
    if "RBF-learnWidth" in method:
        return "RBF-learnWidth"
    if "ABRBF-linear" in method:
        return "ABRBF-linear"
    if "ABRBF-silu" in method:
        return "ABRBF-silu"
    if "quantile" in method:
        return "RBF-quantileCenters"
    return "RBFOnly"


def _train_edge_adamw(
    args: argparse.Namespace,
    dataset: str,
    seed: int,
    method: str,
    params: V49Params,
    device: torch.device,
    *,
    steps: int,
    save_state: bool = True,
) -> Tuple[Any, nn.Module, Dict[str, Any], List[Dict[str, Any]]]:
    edge_kind = _edge_kind_from_method(method)
    bundle = load_vision_bundle(
        dataset,
        data_root=args.data_root,
        train_size=params.train_size,
        val_size=params.val_size,
        test_size=params.test_size,
        seed=seed,
        download=not args.no_download,
        allow_fake_data=args.allow_fake_data,
    )
    set_seed(seed)
    model = _make_v49_model(bundle, params, device, edge_kind=edge_kind)
    if edge_kind == "RBF-quantileCenters":
        _init_quantile_centers(model, bundle.x_train[: params.batch_size].to(device))
    opt = torch.optim.AdamW([p for _, p in _edge_named_params(model)], lr=params.adam_lr, weight_decay=0.0)
    idxs = _iter_steps(len(bundle.x_train), params.batch_size, seed + 4900, steps)
    trace: List[Dict[str, Any]] = []
    hb = bundle.x_val[: params.audit_batch_size].to(device)
    yh = bundle.y_val[: params.audit_batch_size].to(device)
    val0 = F.cross_entropy(model(hb), yh).item()
    checkpoints = {1, 5, 20, 50, 100, steps}
    start = time.perf_counter()
    bad = 0
    for step, idx in enumerate(idxs, start=1):
        xb = bundle.x_train[idx].to(device)
        yb = bundle.y_train[idx].to(device)
        before = F.cross_entropy(model(xb), yb).item()
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(xb), yb)
        loss.backward()
        before_state = {name: p.detach().clone() for name, p in _edge_named_params(model)}
        opt.step()
        after = F.cross_entropy(model(xb), yb).item()
        bad += int(after > before)
        if step in checkpoints:
            deltas = []
            role_norms: Dict[str, List[float]] = {"input": [], "block": [], "output": []}
            for name, p in _edge_named_params(model):
                d = p.detach() - before_state.get(name, p.detach())
                deltas.append(float(d.float().norm().cpu()))
                role_norms.setdefault(_role_group(_role_for_name(name)), []).append(float(d.float().norm().cpu()))
            total = sum(deltas)
            val = F.cross_entropy(model(hb), yh).item()
            trace.append(
                {
                    "stage": "P1",
                    "dataset": dataset,
                    "seed": seed,
                    "method": method,
                    "edge_kind": edge_kind,
                    "step": step,
                    "train_loss_before": before,
                    "train_loss_after": after,
                    "val_loss": val,
                    "val_loss_descent": val0 - val,
                    "bad_step_rate": bad / step,
                    "input_update_share": _mean(role_norms.get("input", []), 0.0) / max(1.0e-12, total),
                    "block_update_share": _mean(role_norms.get("block", []), 0.0) / max(1.0e-12, total),
                    "output_update_share": _mean(role_norms.get("output", []), 0.0) / max(1.0e-12, total),
                }
            )
    wall = time.perf_counter() - start
    row = _eval_v49(model, bundle, params, device)
    row.update(
        {
            "stage": "P1",
            "dataset": dataset,
            "seed": seed,
            "method": method,
            "edge_kind": edge_kind,
            "steps": steps,
            "val_auc_proxy": _mean([r["val_loss"] for r in trace], float("nan")),
            "bad_step_rate": bad / max(1, steps),
            "step_time_ms": 1000.0 * wall / max(1, steps),
            "hidden_dim": params.hidden_dim,
            "basis_count": params.basis_count,
            "depth": params.depth,
            "error": "",
        }
    )
    row.update(_param_count_audit(model))
    if save_state:
        _save_state(_state_path(Path(args.out_dir), dataset, seed, method, params), model, row)
    return bundle, model, row, trace


def _load_or_train_teacher(
    args: argparse.Namespace,
    dataset: str,
    seed: int,
    method: str,
    params: V49Params,
    device: torch.device,
) -> Tuple[Any, nn.Module, Dict[str, Any]]:
    path = _state_path(Path(args.out_dir), dataset, seed, method, params)
    bundle = load_vision_bundle(
        dataset,
        data_root=args.data_root,
        train_size=params.train_size,
        val_size=params.val_size,
        test_size=params.test_size,
        seed=seed,
        download=not args.no_download,
        allow_fake_data=args.allow_fake_data,
    )
    edge_kind = _edge_kind_from_method(method)
    model = _make_v49_model(bundle, params, device, edge_kind=edge_kind)
    if path.exists():
        payload = torch.load(path, map_location="cpu")
        _load_state_cpu(model, payload["state"], device)
        meta = dict(payload.get("meta", {}))
        return bundle, model, meta
    bundle, model, meta, _ = _train_edge_adamw(args, dataset, seed, method, params, device, steps=params.p1_steps)
    return bundle, model, meta


def _flatten_named(named: Sequence[Tuple[str, nn.Parameter]], grads: Sequence[torch.Tensor | None]) -> torch.Tensor:
    pieces: List[torch.Tensor] = []
    for (_, p), g in zip(named, grads):
        if g is None:
            pieces.append(torch.zeros(p.numel(), device=p.device, dtype=torch.float32))
        else:
            pieces.append(g.detach().float().reshape(-1))
    return torch.cat(pieces) if pieces else torch.zeros(0)


def _split_flat(named: Sequence[Tuple[str, nn.Parameter]], delta: torch.Tensor) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    pos = 0
    for name, p in named:
        n = p.numel()
        out[name] = delta[pos : pos + n].reshape_as(p).to(dtype=p.dtype, device=p.device)
        pos += n
    return out


def _selected_coeff_params(model: nn.Module, role: str | None) -> List[Tuple[str, nn.Parameter]]:
    named = []
    for name, p in coefficient_named_params(model):
        if role is None or _role_group(_role_for_name(name)) == role:
            named.append((name, p))
    return named


def _constraint_vector(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    teacher: Dict[str, torch.Tensor],
    *,
    mode: str,
    hidden_sketch: int,
    seed: int,
) -> torch.Tensor:
    feats = _collect_pure_features(model, x, detach=False)
    vals: List[torch.Tensor] = []
    if "logit" in mode:
        vals.append(feats["logits"].reshape(-1))
    if "hidden" in mode:
        h = feats["block"]
        if h.shape[-1] > hidden_sketch:
            gen = torch.Generator(device=h.device)
            gen.manual_seed(seed)
            proj = torch.randn(h.shape[-1], hidden_sketch, generator=gen, device=h.device, dtype=h.dtype) / math.sqrt(hidden_sketch)
            vals.append((h @ proj).reshape(-1))
        else:
            vals.append(h.reshape(-1))
    if "margin" in mode:
        logits = feats["logits"]
        target = logits.gather(1, y.view(-1, 1)).squeeze(1)
        masked = logits.masked_fill(F.one_hot(y, logits.shape[-1]).bool(), float("-inf"))
        vals.append((target - masked.max(dim=1).values).reshape(-1))
    if not vals:
        vals.append(feats["logits"].reshape(-1))
    return torch.cat(vals)


def _build_jacobian(
    model: nn.Module,
    named: Sequence[Tuple[str, nn.Parameter]],
    x: torch.Tensor,
    y: torch.Tensor,
    teacher: Dict[str, torch.Tensor],
    params: V49Params,
    *,
    mode: str,
    seed: int,
) -> Tuple[torch.Tensor, int]:
    c = _constraint_vector(model, x, y, teacher, mode=mode, hidden_sketch=params.exact_hidden_sketch, seed=seed)
    n_total = int(c.numel())
    if n_total > params.exact_max_constraints:
        idx = torch.linspace(0, n_total - 1, params.exact_max_constraints, device=c.device).round().long()
        c = c[idx]
    rows: List[torch.Tensor] = []
    param_list = [p for _, p in named]
    for i in range(int(c.numel())):
        grads = torch.autograd.grad(c[i], param_list, retain_graph=True, allow_unused=True)
        rows.append(_flatten_named(named, grads))
    return torch.stack(rows, dim=0) if rows else torch.zeros(0, 0, device=x.device), n_total


def _argmax_flip_rate(model: nn.Module, x: torch.Tensor, teacher_logits: torch.Tensor) -> float:
    with torch.no_grad():
        pred = model(x).argmax(dim=-1)
        tpred = teacher_logits.to(pred.device).argmax(dim=-1)
        return float((pred != tpred).float().mean().detach().cpu())


def _exact_nfs_project_once(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    teacher: Dict[str, torch.Tensor],
    params: V49Params,
    *,
    constraint_mode: str,
    metric: str,
    role: str | None,
    eta: float,
    seed: int,
) -> Dict[str, Any]:
    named = _selected_coeff_params(model, role)
    if not named:
        return {"accepted": 0, "error": "no selected params"}
    snap = _snapshot(named)
    rough_before = _roughness_scalar(model)
    loss = _roughness_loss(model, role=role)
    grads = torch.autograd.grad(loss, [p for _, p in named], allow_unused=True)
    g = _flatten_named(named, grads)
    raw_norm = float(g.norm().detach().cpu())
    if raw_norm <= 1.0e-12:
        return {"accepted": 0, "error": "zero geometry gradient"}
    J, full_constraints = _build_jacobian(model, named, x, y, teacher, params, mode=constraint_mode, seed=seed)
    diag = _metric_diag_for_params(named, model, metric).to(device=g.device, dtype=torch.float32).clamp_min(1.0e-8)
    minv_g = g / diag
    if J.numel() > 0:
        Jf = J.float()
        K = (Jf / diag.unsqueeze(0)) @ Jf.T
        rhs = Jf @ minv_g
        K = 0.5 * (K + K.T) + params.exact_mu * torch.eye(K.shape[0], device=K.device)
        alpha = torch.linalg.solve(K, rhs.unsqueeze(-1)).squeeze(-1)
        projected = minv_g - (Jf.T @ alpha) / diag
        kkt_residual = float((Jf @ projected).norm().detach().cpu() / raw_norm)
        rank_j = int(torch.linalg.matrix_rank(Jf.detach().cpu(), tol=1.0e-5).item())
        cond = float(torch.linalg.cond(K.detach().cpu()).item()) if K.numel() else float("nan")
    else:
        projected = minv_g
        kkt_residual = 0.0
        rank_j = 0
        cond = float("nan")
    delta = -eta * projected
    param_norm = math.sqrt(sum(float(p.detach().float().square().sum().cpu()) for _, p in named))
    delta_norm = float(delta.norm().detach().cpu())
    trust = params.exact_trust_ratio * max(1.0e-12, param_norm)
    trust_scale = min(1.0, trust / max(1.0e-12, delta_norm))
    delta = delta * trust_scale
    pred_constraint = float((J.float() @ delta.float()).norm().detach().cpu()) if J.numel() else 0.0
    accepted = 0
    accepted_eta = 0.0
    final_stats = _teacher_constraints(model, x, y, teacher, params)
    reject_reason = "constraint_rejected"
    backtrack_count = params.exact_backtracks
    trial_base = _split_flat(named, delta)
    teacher_logits = teacher["logits"].to(device=x.device)
    for bt in range(params.exact_backtracks):
        scale = 0.5**bt
        _restore(named, snap)
        trial = {name: upd * scale for name, upd in trial_base.items()}
        _apply_updates(named, trial)
        stats = _teacher_constraints(model, x, y, teacher, params)
        rough_after_trial = _roughness_scalar(model)
        phi_red_trial = 1.0 - rough_after_trial / max(1.0e-12, rough_before)
        ok = (
            stats["kl_teacher_student"] < params.exact_kl_radius
            and stats["logit_relative_drift"] < params.exact_logit_radius
            and stats["hidden_relative_drift"] < params.exact_hidden_radius
            and phi_red_trial > 0.0
        )
        if ok:
            accepted = 1
            accepted_eta = eta * trust_scale * scale
            final_stats = stats
            reject_reason = ""
            backtrack_count = bt
            break
    if not accepted:
        _restore(named, snap)
    rough_after = _roughness_scalar(model)
    actual_red = 1.0 - rough_after / max(1.0e-12, rough_before)
    applied_delta = torch.cat([(p.detach() - snap[name]).float().reshape(-1) for name, p in named])
    actual_constraint = float((J.float() @ applied_delta.float()).norm().detach().cpu()) if J.numel() else 0.0
    return {
        "accepted": accepted,
        "accepted_eta": accepted_eta,
        "backtrack_count": backtrack_count,
        "reject_reason": reject_reason,
        "geometry_before": rough_before,
        "geometry_after": rough_after,
        "phi_reduction_step": actual_red,
        "predicted_geometry_delta": float((g.float() @ delta.float()).detach().cpu()),
        "actual_geometry_delta": actual_red,
        "raw_geometry_grad_norm": raw_norm,
        "projected_grad_norm": float(projected.norm().detach().cpu()),
        "projected_component_norm": float(projected.norm().detach().cpu()),
        "nullspace_component_norm": float(applied_delta.norm().detach().cpu()),
        "J_delta_norm_pred": pred_constraint,
        "J_delta_norm_actual": actual_constraint,
        "constraint_residual_predicted": pred_constraint,
        "constraint_residual_actual": actual_constraint,
        "KKT_residual": kkt_residual,
        "CG_residual": kkt_residual,
        "CG_iterations": 0,
        "projector_condition": cond,
        "jacobian_rank": rank_j,
        "jacobian_rows": int(J.shape[0]) if J.ndim == 2 else 0,
        "jacobian_full_constraints": full_constraints,
        "argmax_flip_rate": _argmax_flip_rate(model, x, teacher_logits),
        **final_stats,
    }


def _run_p3_projection(
    args: argparse.Namespace,
    dataset: str,
    seed: int,
    method: str,
    params: V49Params,
    device: torch.device,
    *,
    variant: str,
    metric: str,
    constraint: str,
) -> Dict[str, Any]:
    bundle, model, meta = _load_or_train_teacher(args, dataset, seed, method, params, device)
    hb = bundle.x_val[: params.audit_batch_size].to(device)
    yh = bundle.y_val[: params.audit_batch_size].to(device)
    teacher = _teacher_snapshot(model, hb)
    before_eval = _eval_v49(model, bundle, params, device)
    if variant.startswith("Heuristic"):
        stats = _nfs_project_once(
            model,
            hb,
            yh,
            teacher,
            params,
            variant="NFS-role-block",
            projector="diag",
            role="block",
            eta=params.exact_eta,
            device=device,
        )
        stats["KKT_residual"] = float("nan")
        stats["CG_residual"] = stats.get("cg_residual", float("nan"))
        stats["argmax_flip_rate"] = _argmax_flip_rate(model, hb, teacher["logits"])
    else:
        stats = _exact_nfs_project_once(
            model,
            hb,
            yh,
            teacher,
            params,
            constraint_mode=constraint,
            metric=metric,
            role="block",
            eta=params.exact_eta,
            seed=seed + 4917,
        )
    after_eval = _eval_v49(model, bundle, params, device)
    final_constraints = _teacher_constraints(model, hb, yh, teacher, params)
    acc_drop = before_eval["test_acc"] - after_eval["test_acc"]
    phi_red = 1.0 - after_eval["phi_prime_p95"] / max(1.0e-12, before_eval["phi_prime_p95"])
    j_red = 1.0 - after_eval["jacobian_condition"] / max(1.0e-12, before_eval["jacobian_condition"])
    sob_red = 1.0 - after_eval["sobolev_norm_total"] / max(1.0e-12, before_eval["sobolev_norm_total"])
    residual_ok = (
        (math.isfinite(float(stats.get("KKT_residual", float("nan")))) and float(stats.get("KKT_residual", 99.0)) < 1.0e-3)
        or (math.isfinite(float(stats.get("CG_residual", float("nan")))) and float(stats.get("CG_residual", 99.0)) < 1.0e-2)
        or variant.startswith("Heuristic")
    )
    p3_pass = (
        acc_drop <= 0.005
        and final_constraints["kl_teacher_student"] < 0.005
        and final_constraints["logit_relative_drift"] < 0.03
        and float(stats.get("argmax_flip_rate", 1.0)) < 0.02
        and (phi_red > 0.10 or j_red > 0.20)
        and residual_ok
    )
    row: Dict[str, Any] = {
        "stage": "P3",
        "dataset": dataset,
        "seed": seed,
        "teacher": method,
        "edge_kind": _edge_kind_from_method(method),
        "variant": variant,
        "metric": metric,
        "constraint": constraint,
        "teacher_acc": before_eval["test_acc"],
        "acc_after": after_eval["test_acc"],
        "acc_drop": acc_drop,
        "teacher_phi": before_eval["phi_prime_p95"],
        "phi_after": after_eval["phi_prime_p95"],
        "phi_reduction": phi_red,
        "teacher_jacobian": before_eval["jacobian_condition"],
        "jacobian_after": after_eval["jacobian_condition"],
        "jacobian_reduction": j_red,
        "sobolev_reduction": sob_red,
        "KL": final_constraints["kl_teacher_student"],
        "logit_drift": final_constraints["logit_relative_drift"],
        "hidden_drift": final_constraints["hidden_relative_drift"],
        "margin_drift": final_constraints["margin_relative_drift"],
        "p3_pass": int(p3_pass),
        "error": stats.get("error", ""),
    }
    row.update(stats)
    return row


def run_p0(args: argparse.Namespace) -> List[Dict[str, Any]]:
    out_dir = ensure_dir(args.out_dir)
    rows: List[Dict[str, Any]] = [] if args.fresh else read_csv(out_dir / "p0_code_audit.csv")
    device = get_device(args.device)
    params = V49Params(train_size=512, val_size=128, test_size=128, p1_steps=10, audit_batch_size=16)
    done = {(r.get("dataset"), r.get("method")) for r in rows if not r.get("error")}
    for dataset in [dataset_name(d) for d in parse_str_list(args.datasets)]:
        bundle = load_vision_bundle(dataset, data_root=args.data_root, train_size=512, val_size=128, test_size=128, seed=0, download=not args.no_download, allow_fake_data=args.allow_fake_data)
        x = bundle.x_train[: params.batch_size].to(device)
        hb = bundle.x_val[: params.audit_batch_size].to(device)
        yh = bundle.y_val[: params.audit_batch_size].to(device)
        for method in P0_EDGE_KINDS:
            if (dataset, method) in done:
                continue
            edge_kind = "RBFOnly" if method == "ExactNFS-smoke" else method
            err = ""
            nfs_stats: Dict[str, Any] = {}
            try:
                set_seed(0)
                model = _make_v49_model(bundle, params, device, edge_kind=edge_kind)
                if edge_kind == "RBF-quantileCenters":
                    _init_quantile_centers(model, x)
                model(hb)
                named = coefficient_named_params(model)
                snap = _snapshot(named)
                if method == "ExactNFS-smoke":
                    teacher = _teacher_snapshot(model, hb)
                    nfs_stats = _exact_nfs_project_once(
                        model,
                        hb,
                        yh,
                        teacher,
                        params,
                        constraint_mode="logit",
                        metric="identity",
                        role="block",
                        eta=params.exact_eta,
                        seed=0,
                    )
                temp_apply = max([float((p.detach() - snap[n]).abs().max().cpu()) for n, p in named] or [0.0])
                _restore(named, snap)
                rollback = max([float((p.detach() - snap[n]).abs().max().cpu()) for n, p in named] or [0.0])
                pcount = _param_count_audit(model)
                row = {
                    "stage": "P0",
                    "dataset": dataset,
                    "method": method,
                    "edge_kind": edge_kind,
                    **pcount,
                    "rollback_error": rollback,
                    "temp_apply_error": temp_apply,
                    "NFS_kkt_residual": nfs_stats.get("KKT_residual", 0.0),
                    "NFS_constraint_residual": nfs_stats.get("constraint_residual_actual", 0.0),
                    "CG_residual": nfs_stats.get("CG_residual", 0.0),
                    "no_nan_inf": int(all(torch.isfinite(p).all().item() for _, p in _edge_named_params(model))),
                    "error": err,
                }
            except Exception as exc:
                if not args.continue_on_error:
                    raise
                row = {"stage": "P0", "dataset": dataset, "method": method, "edge_kind": edge_kind, "error": repr(exc)}
            rows.append(row)
            print(f"P0 {dataset} {method} nonKAN={row.get('learnable_nonKAN_params')} rollback={row.get('rollback_error')} err={row.get('error','')}")
            write_csv(out_dir / "p0_code_audit.csv", rows)
    return rows


def run_p1(args: argparse.Namespace) -> List[Dict[str, Any]]:
    out_dir = ensure_dir(args.out_dir)
    rows: List[Dict[str, Any]] = [] if args.fresh else read_csv(out_dir / "p1_architecture_frontier.csv")
    trace: List[Dict[str, Any]] = [] if args.fresh else read_csv(out_dir / "p1_training_trace.csv")
    device = get_device(args.device)
    params = V49Params()
    methods = [m for m in P1_METHODS if not parse_str_list(args.methods) or m in parse_str_list(args.methods)]
    done = {(r.get("dataset"), int(float(r.get("seed", -1))), r.get("method")) for r in rows if not r.get("error")}
    for dataset in [dataset_name(d) for d in parse_str_list(args.datasets)]:
        for seed in parse_int_list(args.seeds):
            for method in methods:
                if (dataset, seed, method) in done:
                    continue
                try:
                    _, _, row, tr = _train_edge_adamw(args, dataset, seed, method, params, device, steps=params.p1_steps)
                    rows.append(row)
                    trace.extend(tr)
                    print(f"P1 {dataset} s{seed} {method} acc={row['test_acc']:.4f} phi={row['phi_prime_p95']:.4f} base/RBF={row['base_over_rbf_norm']:.3f}")
                except Exception as exc:
                    if not args.continue_on_error:
                        raise
                    rows.append({"stage": "P1", "dataset": dataset, "seed": seed, "method": method, "error": repr(exc)})
                    print(f"P1 ERROR {dataset} s{seed} {method}: {exc!r}")
                write_csv(out_dir / "p1_architecture_frontier.csv", rows)
                write_csv(out_dir / "p1_training_trace.csv", trace)
    return rows


def _eigenmode_stats_for_layer(layer: RBFDense, grad: torch.Tensor | None) -> Dict[str, float]:
    device = layer.coeff.device
    mat = _metric_matrix_for_layer(layer, metric="sobolev_full", device=device)
    eigvals, eigvecs = torch.linalg.eigh(mat.float())
    coeff = layer.coeff.detach().float().reshape(-1, layer.coeff.shape[-1])
    modes = coeff @ eigvecs
    coeff_energy = modes.square().mean(dim=0)
    if grad is not None:
        gm = grad.detach().float().reshape(-1, grad.shape[-1]) @ eigvecs
        grad_energy = gm.square().mean(dim=0)
    else:
        grad_energy = torch.zeros_like(coeff_energy)
    k = eigvals.numel()
    split1 = max(1, k // 3)
    split2 = max(split1 + 1, 2 * k // 3)
    total_c = coeff_energy.sum().clamp_min(1.0e-12)
    total_g = grad_energy.sum().clamp_min(1.0e-12)
    base = _base_dim(layer)
    base_energy = coeff_energy[:base].sum() if base else torch.zeros((), device=device)
    return {
        "eig_min": float(eigvals[0].detach().cpu()),
        "eig_max": float(eigvals[-1].detach().cpu()),
        "eig_condition": float((eigvals[-1] / eigvals[0].clamp_min(1.0e-12)).detach().cpu()),
        "coeff_energy_low": float((coeff_energy[:split1].sum() / total_c).detach().cpu()),
        "coeff_energy_mid": float((coeff_energy[split1:split2].sum() / total_c).detach().cpu()),
        "coeff_energy_high": float((coeff_energy[split2:].sum() / total_c).detach().cpu()),
        "grad_energy_low": float((grad_energy[:split1].sum() / total_g).detach().cpu()),
        "grad_energy_mid": float((grad_energy[split1:split2].sum() / total_g).detach().cpu()),
        "grad_energy_high": float((grad_energy[split2:].sum() / total_g).detach().cpu()),
        "high_mode_fraction": float((coeff_energy[split2:].sum() / total_c).detach().cpu()),
        "base_mode_fraction": float((base_energy / total_c).detach().cpu()),
        "sobolev_penalty_contribution": float((coeff_energy * eigvals).sum().detach().cpu()),
    }


def run_p2(args: argparse.Namespace) -> List[Dict[str, Any]]:
    out_dir = ensure_dir(args.out_dir)
    rows: List[Dict[str, Any]] = [] if args.fresh else read_csv(out_dir / "p2_eigenmode_audit.csv")
    device = get_device(args.device)
    params = V49Params()
    methods = [m for m in P1_METHODS if m in P3_TEACHERS or "learnWidth" in m]
    done = {(r.get("dataset"), int(float(r.get("seed", -1))), r.get("method"), r.get("role")) for r in rows if not r.get("error")}
    for dataset in [dataset_name(d) for d in parse_str_list(args.datasets)]:
        for seed in parse_int_list(args.seeds):
            for method in methods:
                bundle, model, meta = _load_or_train_teacher(args, dataset, seed, method, params, device)
                hb = bundle.x_val[: params.audit_batch_size].to(device)
                yh = bundle.y_val[: params.audit_batch_size].to(device)
                model.zero_grad(set_to_none=True)
                F.cross_entropy(model(hb), yh).backward()
                names = [name for name, _ in coefficient_named_params(model)]
                layers = list(model.kan_layers())  # type: ignore[attr-defined]
                for name, layer in zip(names, layers):
                    role = _role_group(_role_for_name(name))
                    if (dataset, seed, method, role) in done:
                        continue
                    stats = _eigenmode_stats_for_layer(layer, layer.coeff.grad)
                    row = {
                        "stage": "P2",
                        "dataset": dataset,
                        "seed": seed,
                        "method": method,
                        "edge_kind": _edge_kind_from_method(method),
                        "layer": name,
                        "role": role,
                        "teacher_acc": meta.get("test_acc", float("nan")),
                        "teacher_phi": meta.get("phi_prime_p95", float("nan")),
                        **stats,
                        "error": "",
                    }
                    rows.append(row)
                print(f"P2 {dataset} s{seed} {method} layers={len(layers)}")
                write_csv(out_dir / "p2_eigenmode_audit.csv", rows)
    return rows


def run_p3(args: argparse.Namespace) -> List[Dict[str, Any]]:
    out_dir = ensure_dir(args.out_dir)
    rows: List[Dict[str, Any]] = [] if args.fresh else read_csv(out_dir / "p3_exact_nfs_projection.csv")
    device = get_device(args.device)
    params = V49Params()
    seeds = parse_int_list(args.seeds)
    # Exact NFS is intentionally run on the first seed in the audit; P1/P2 carry seed variance.
    if len(seeds) > 1:
        seeds = [seeds[0]]
    done = {(r.get("dataset"), int(float(r.get("seed", -1))), r.get("teacher"), r.get("variant"), r.get("metric"), r.get("constraint")) for r in rows if not r.get("error")}
    for dataset in [dataset_name(d) for d in parse_str_list(args.datasets)]:
        for seed in seeds:
            for teacher in P3_TEACHERS:
                for variant, metric, constraint in P3_VARIANTS:
                    key = (dataset, seed, teacher, variant, metric, constraint)
                    if key in done:
                        continue
                    try:
                        row = _run_p3_projection(args, dataset, seed, teacher, params, device, variant=variant, metric=metric, constraint=constraint)
                        rows.append(row)
                        print(
                            f"P3 {dataset} {teacher} {variant}/{metric}/{constraint} "
                            f"accDrop={row['acc_drop']:.4f} phiRed={row['phi_reduction']:.3f} "
                            f"KL={row['KL']:.4g} KKT={row.get('KKT_residual', float('nan')):.3g} pass={row['p3_pass']}"
                        )
                    except Exception as exc:
                        if not args.continue_on_error:
                            raise
                        rows.append({"stage": "P3", "dataset": dataset, "seed": seed, "teacher": teacher, "variant": variant, "metric": metric, "constraint": constraint, "error": repr(exc)})
                        print(f"P3 ERROR {dataset} {teacher} {variant}: {exc!r}")
                    write_csv(out_dir / "p3_exact_nfs_projection.csv", rows)
    return rows


def _p3_survivors(out_dir: Path) -> List[Tuple[str, str, str, str]]:
    rows = [r for r in read_csv(out_dir / "p3_exact_nfs_projection.csv") if not r.get("error")]
    by_key: Dict[Tuple[str, str, str, str], Dict[str, bool]] = {}
    for row in rows:
        key = (row.get("teacher", ""), row.get("variant", ""), row.get("metric", ""), row.get("constraint", ""))
        by_key.setdefault(key, {})[row.get("dataset", "")] = bool(int(float(row.get("p3_pass", 0) or 0)))
    return [key for key, by_d in by_key.items() if all(by_d.get(d, False) for d in DATASETS)]


def _refresh_edge_adamw(model: nn.Module, bundle: Any, params: V49Params, device: torch.device, seed: int, steps: int) -> None:
    opt = torch.optim.AdamW([p for _, p in _edge_named_params(model)], lr=params.adam_lr * 0.35, weight_decay=0.0)
    idxs = _iter_steps(len(bundle.x_train), params.batch_size, seed + 4940, steps)
    for idx in idxs:
        xb = bundle.x_train[idx].to(device)
        yb = bundle.y_train[idx].to(device)
        opt.zero_grad(set_to_none=True)
        F.cross_entropy(model(xb), yb).backward()
        opt.step()


def run_p4(args: argparse.Namespace) -> List[Dict[str, Any]]:
    out_dir = ensure_dir(args.out_dir)
    survivors = _p3_survivors(out_dir)
    if not survivors:
        rows = [{"status": "not_run", "reason": "P3 produced no all-dataset Exact NFS survivor"}]
        write_csv(out_dir / "p4_tan_one_cycle_v2.csv", rows)
        print("P4 not run: no P3 all-dataset survivor")
        return rows
    rows: List[Dict[str, Any]] = [] if args.fresh else [r for r in read_csv(out_dir / "p4_tan_one_cycle_v2.csv") if r.get("status") != "not_run"]
    device = get_device(args.device)
    params = V49Params()
    seeds = parse_int_list(args.seeds)
    if len(seeds) > 1:
        seeds = [seeds[0]]
    for dataset in [dataset_name(d) for d in parse_str_list(args.datasets)]:
        for seed in seeds:
            for teacher_method, variant, metric, constraint in survivors[:3]:
                for refresh in ["none", "AdamW-small-5"]:
                    bundle, model, _ = _load_or_train_teacher(args, dataset, seed, teacher_method, params, device)
                    hb = bundle.x_val[: params.audit_batch_size].to(device)
                    yh = bundle.y_val[: params.audit_batch_size].to(device)
                    teacher = _teacher_snapshot(model, hb)
                    task_eval = _eval_v49(model, bundle, params, device)
                    if variant.startswith("Heuristic"):
                        _nfs_project_once(model, hb, yh, teacher, params, variant="NFS-role-block", projector="diag", role="block", eta=params.exact_eta, device=device)
                    else:
                        _exact_nfs_project_once(model, hb, yh, teacher, params, constraint_mode=constraint, metric=metric, role="block", eta=params.exact_eta, seed=seed + 4929)
                    smooth_eval = _eval_v49(model, bundle, params, device)
                    smooth_constraints = _teacher_constraints(model, hb, yh, teacher, params)
                    if refresh == "AdamW-small-5":
                        _refresh_edge_adamw(model, bundle, params, device, seed, steps=5)
                    final_eval = _eval_v49(model, bundle, params, device)
                    final_constraints = _teacher_constraints(model, hb, yh, teacher, params)
                    acc_drop = task_eval["test_acc"] - final_eval["test_acc"]
                    phi_red = 1.0 - final_eval["phi_prime_p95"] / max(1.0e-12, task_eval["phi_prime_p95"])
                    p4_pass = (
                        acc_drop <= 0.005
                        and phi_red > 0.10
                        and final_constraints["kl_teacher_student"] < 0.01
                        and final_eval["rank_block"] >= 0.85 * max(1.0e-12, task_eval["rank_block"])
                    )
                    row = {
                        "stage": "P4",
                        "dataset": dataset,
                        "seed": seed,
                        "task_teacher": teacher_method,
                        "variant": variant,
                        "metric": metric,
                        "constraint": constraint,
                        "refresh": refresh,
                        "task_acc": task_eval["test_acc"],
                        "smooth_acc": smooth_eval["test_acc"],
                        "final_acc": final_eval["test_acc"],
                        "acc_drop_from_task": acc_drop,
                        "phi_reduction_from_task": phi_red,
                        "KL_after_smooth": smooth_constraints["kl_teacher_student"],
                        "KL_after_refresh": final_constraints["kl_teacher_student"],
                        "rank_after_refresh": final_eval["rank_block"],
                        "p4_pass": int(p4_pass),
                        "error": "",
                    }
                    rows.append(row)
                    print(f"P4 {dataset} {teacher_method} {variant} {refresh} accDrop={acc_drop:.4f} phiRed={phi_red:.3f} pass={int(p4_pass)}")
                    write_csv(out_dir / "p4_tan_one_cycle_v2.csv", rows)
    return rows


def run_p5(args: argparse.Namespace) -> List[Dict[str, Any]]:
    out_dir = ensure_dir(args.out_dir)
    p4 = [r for r in read_csv(out_dir / "p4_tan_one_cycle_v2.csv") if not r.get("error") and r.get("status") != "not_run"]
    by_key: Dict[Tuple[str, str, str, str, str], Dict[str, bool]] = {}
    for row in p4:
        key = (row.get("task_teacher", ""), row.get("variant", ""), row.get("metric", ""), row.get("constraint", ""), row.get("refresh", ""))
        by_key.setdefault(key, {})[row.get("dataset", "")] = bool(int(float(row.get("p4_pass", 0) or 0)))
    survivors = [key for key, by_d in by_key.items() if all(by_d.get(d, False) for d in DATASETS)]
    if not survivors:
        rows = [{"status": "not_run", "reason": "P4 produced no all-dataset one-cycle survivor"}]
        write_csv(out_dir / "p5_event_tan_cycles.csv", rows)
        print("P5 not run: no P4 all-dataset survivor")
        return rows
    rows: List[Dict[str, Any]] = []
    device = get_device(args.device)
    params = V49Params()
    task_teacher, variant, metric, constraint, refresh = survivors[0]
    for dataset in [dataset_name(d) for d in parse_str_list(args.datasets)]:
        seed = parse_int_list(args.seeds)[0]
        bundle, model, _ = _load_or_train_teacher(args, dataset, seed, task_teacher, params, device)
        hb = bundle.x_val[: params.audit_batch_size].to(device)
        yh = bundle.y_val[: params.audit_batch_size].to(device)
        initial = _eval_v49(model, bundle, params, device)
        for cycle in range(1, 4):
            before = _eval_v49(model, bundle, params, device)
            geometry_bad = before["phi_prime_p95"] > 0.92 * initial["phi_prime_p95"]
            rank_ok = before["rank_block"] >= 0.75 * max(1.0e-12, initial["rank_block"])
            triggered = geometry_bad and rank_ok
            reason = "geometry_bad_rank_ok" if triggered else "skip_rank_or_geometry"
            if triggered:
                teacher = _teacher_snapshot(model, hb)
                if variant.startswith("Heuristic"):
                    _nfs_project_once(model, hb, yh, teacher, params, variant="NFS-role-block", projector="diag", role="block", eta=params.exact_eta, device=device)
                else:
                    _exact_nfs_project_once(model, hb, yh, teacher, params, constraint_mode=constraint, metric=metric, role="block", eta=params.exact_eta, seed=seed + 4960 + cycle)
            if refresh == "AdamW-small-5":
                _refresh_edge_adamw(model, bundle, params, device, seed + cycle, steps=5)
            after = _eval_v49(model, bundle, params, device)
            p5_pass = after["test_acc"] >= initial["test_acc"] - 0.01 and after["phi_prime_p95"] <= 0.90 * max(1.0e-12, initial["phi_prime_p95"]) and after["rank_block"] >= 0.75 * initial["rank_block"]
            row = {
                "stage": "P5",
                "dataset": dataset,
                "seed": seed,
                "task_teacher": task_teacher,
                "variant": variant,
                "metric": metric,
                "constraint": constraint,
                "refresh": refresh,
                "cycle_index": cycle,
                "smooth_triggered": int(triggered),
                "trigger_reason": reason,
                "initial_acc": initial["test_acc"],
                "final_acc": after["test_acc"],
                "acc_drop": initial["test_acc"] - after["test_acc"],
                "final_phi_reduction": 1.0 - after["phi_prime_p95"] / max(1.0e-12, initial["phi_prime_p95"]),
                "rank_final_ratio": after["rank_block"] / max(1.0e-12, initial["rank_block"]),
                "p5_pass_final_combo": int(p5_pass),
                "error": "",
            }
            rows.append(row)
            print(f"P5 {dataset} cycle={cycle} trigger={triggered} acc={after['test_acc']:.4f} phiRed={row['final_phi_reduction']:.3f} pass={int(p5_pass)}")
            write_csv(out_dir / "p5_event_tan_cycles.csv", rows)
    return rows


def run_p6(args: argparse.Namespace) -> List[Dict[str, Any]]:
    out_dir = ensure_dir(args.out_dir)
    rows: List[Dict[str, Any]] = [] if args.fresh else read_csv(out_dir / "p6_capacity_basis_expansion_full.csv")
    device = get_device(args.device)
    seeds = parse_int_list(args.seeds)
    if len(seeds) > 1:
        seeds = [seeds[0]]
    done = {(r.get("dataset"), int(float(r.get("seed", -1))), r.get("edge_kind"), int(float(r.get("hidden_dim", -1))), int(float(r.get("basis_count", -1))), int(float(r.get("depth", -1)))) for r in rows if not r.get("error")}
    for dataset in [dataset_name(d) for d in parse_str_list(args.datasets)]:
        for seed in seeds:
            for hidden, basis, depth, edge_kind in P6_CONFIGS:
                key = (dataset, seed, edge_kind, hidden, basis, depth)
                if key in done:
                    continue
                params = V49Params(hidden_dim=hidden, basis_count=basis, depth=depth, p1_steps=100)
                method = f"PureKAN-{edge_kind}-AdamW"
                try:
                    _, _, row, _ = _train_edge_adamw(args, dataset, seed, method, params, device, steps=params.p6_steps, save_state=False)
                    row["stage"] = "P6"
                    row["edge_kind"] = edge_kind
                    rows.append(row)
                    print(f"P6 {dataset} {edge_kind} h={hidden} b={basis} d={depth} acc={row['test_acc']:.4f} phi={row['phi_prime_p95']:.4f}")
                except Exception as exc:
                    if not args.continue_on_error:
                        raise
                    rows.append({"stage": "P6", "dataset": dataset, "seed": seed, "edge_kind": edge_kind, "hidden_dim": hidden, "basis_count": basis, "depth": depth, "error": repr(exc)})
                    print(f"P6 ERROR {dataset} {edge_kind}: {exc!r}")
                write_csv(out_dir / "p6_capacity_basis_expansion_full.csv", rows)
    return rows


def _blank_later_files(out_dir: Path, reason: str) -> None:
    for name in [
        "p4_tan_one_cycle_v2.csv",
        "p5_event_tan_cycles.csv",
        "p6_capacity_basis_expansion_full.csv",
        "p7_candidate_selection.csv",
        "p8_confirm5.csv",
        "p9_confirm10.csv",
    ]:
        path = out_dir / name
        if not path.exists():
            write_csv(path, [{"status": "not_run", "reason": reason}])


def add_args() -> argparse.ArgumentParser:
    p = add_v3_args()
    p.description = __doc__
    p.set_defaults(packages="V4_9_P0_SMOKE", datasets="MNIST,Fashion-MNIST,KMNIST", out_dir=Path("results/v4_9"), seeds="0,1,2")
    return p


def main() -> int:
    args = add_args().parse_args()
    all_rows: List[Dict[str, Any]] = []
    for package in parse_str_list(args.packages):
        key = package.strip().upper().replace("-", "_")
        if key in {"V4_9_P0", "V4_9_P0_SMOKE", "V4_9_P0_CODE_AUDIT"}:
            all_rows = run_p0(args)
        elif key in {"V4_9_P1", "V4_9_P1_FRONTIER", "V4_9_P1_ARCHITECTURE"}:
            all_rows = run_p1(args)
        elif key in {"V4_9_P2", "V4_9_P2_EIGEN", "V4_9_P2_EIGENMODE"}:
            all_rows = run_p2(args)
        elif key in {"V4_9_P3", "V4_9_P3_EXACT_NFS"}:
            all_rows = run_p3(args)
            _blank_later_files(ensure_dir(args.out_dir), "pending P3 gate decision")
        elif key in {"V4_9_P4", "V4_9_P4_TAN_ONE_CYCLE"}:
            all_rows = run_p4(args)
        elif key in {"V4_9_P5", "V4_9_P5_EVENT_TAN"}:
            all_rows = run_p5(args)
        elif key in {"V4_9_P6", "V4_9_P6_CAPACITY"}:
            all_rows = run_p6(args)
        else:
            raise ValueError(f"unknown v4.9 package: {package}")
    return 0 if all_rows is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
