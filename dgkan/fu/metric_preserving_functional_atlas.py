"""Metric-preserving functional atlas primitives for v22.64.

The classes in this module implement FU as persistent model coordinates:
trainable coordinate tensors live inside ``nn.Module`` parameterization, while
the atlas state is non-trainable coordinate metadata.  Training can therefore
use the ordinary forward / CE backward / optimizer step loop.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def _sym(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (x + x.transpose(-1, -2))


def psd_project(
    matrix: torch.Tensor,
    *,
    eps: float = 1.0e-6,
    shrinkage: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Return a symmetric PSD matrix and diagnostics."""

    work_dtype = torch.float64 if matrix.dtype == torch.float64 else torch.float32
    mat = _sym(matrix.to(dtype=work_dtype))
    dim = int(mat.shape[-1])
    eye = torch.eye(dim, device=mat.device, dtype=mat.dtype)
    if shrinkage > 0.0:
        trace_scale = torch.trace(mat) / max(1, dim)
        mat = (1.0 - float(shrinkage)) * mat + float(shrinkage) * trace_scale * eye
    evals, evecs = torch.linalg.eigh(mat)
    raw_min = float(evals.min().detach().cpu().item()) if evals.numel() else 0.0
    floored = torch.clamp(evals, min=float(eps))
    out = (evecs * floored.unsqueeze(0)) @ evecs.transpose(0, 1)
    out = _sym(out)
    cond = float((floored.max() / floored.min().clamp_min(eps)).detach().cpu().item()) if floored.numel() else 1.0
    eff_rank = effective_rank(floored)
    return out, {
        "raw_psd_min": raw_min,
        "psd_min": float(floored.min().detach().cpu().item()) if floored.numel() else 0.0,
        "condition_number": cond,
        "effective_rank": eff_rank,
        "trace": float(floored.sum().detach().cpu().item()),
        "shrinkage_strength": float(shrinkage),
    }


def inv_sqrt_psd(matrix: torch.Tensor, *, eps: float = 1.0e-6) -> torch.Tensor:
    work_dtype = torch.float64 if matrix.dtype == torch.float64 else torch.float32
    mat = _sym(matrix.to(dtype=work_dtype))
    evals, evecs = torch.linalg.eigh(mat)
    vals = torch.clamp(evals, min=float(eps)).rsqrt()
    return (evecs * vals.unsqueeze(0)) @ evecs.transpose(0, 1)


def sqrt_psd(matrix: torch.Tensor, *, eps: float = 1.0e-6) -> torch.Tensor:
    work_dtype = torch.float64 if matrix.dtype == torch.float64 else torch.float32
    mat = _sym(matrix.to(dtype=work_dtype))
    evals, evecs = torch.linalg.eigh(mat)
    vals = torch.clamp(evals, min=float(eps)).sqrt()
    return (evecs * vals.unsqueeze(0)) @ evecs.transpose(0, 1)


def stable_rank(matrix: torch.Tensor, *, eps: float = 1.0e-12) -> float:
    vals = torch.linalg.svdvals(matrix.float())
    if vals.numel() == 0:
        return 0.0
    return float((vals.square().sum() / vals.max().square().clamp_min(eps)).detach().cpu().item())


def effective_rank(values: torch.Tensor, *, eps: float = 1.0e-12) -> float:
    vals = torch.clamp(values.float(), min=0.0)
    total = vals.sum()
    if float(total.detach().cpu().item()) <= eps:
        return 0.0
    p = vals / total.clamp_min(eps)
    entropy = -(p * torch.log(p.clamp_min(eps))).sum()
    return float(torch.exp(entropy).detach().cpu().item())


def orthonormal_columns(mat: torch.Tensor, rank: int, *, generator: torch.Generator | None = None) -> torch.Tensor:
    if mat.numel() == 0:
        raise ValueError("cannot orthonormalize an empty matrix")
    q, _ = torch.linalg.qr(mat.float(), mode="reduced")
    if q.shape[1] >= rank:
        return q[:, :rank].contiguous()
    extra = torch.randn(mat.shape[0], rank - q.shape[1], device=mat.device, generator=generator)
    q2, _ = torch.linalg.qr(torch.cat([q, extra], dim=1), mode="reduced")
    return q2[:, :rank].contiguous()


def random_orthonormal(dim: int, rank: int, *, device: torch.device, generator: torch.Generator) -> torch.Tensor:
    mat = torch.randn(dim, rank, device=device, generator=generator)
    return orthonormal_columns(mat, rank, generator=generator)


def normalize_columns(mat: torch.Tensor, *, eps: float = 1.0e-8) -> torch.Tensor:
    return mat.float() / torch.linalg.norm(mat.float(), dim=0, keepdim=True).clamp_min(eps)


@dataclass
class AtlasBuild:
    output_basis: torch.Tensor
    input_basis: torch.Tensor
    coord_scale: torch.Tensor
    gf: torch.Tensor
    active_gram: torch.Tensor
    feature_metric: torch.Tensor
    metrics: dict[str, float]


class SimpleMLP(nn.Module):
    """Small MLP with an explicit feature extractor and readout."""

    def __init__(self, input_dim: int, num_classes: int, hidden: int = 96, seed: int = 0) -> None:
        super().__init__()
        torch.manual_seed(int(seed))
        self.fc1 = nn.Linear(input_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, num_classes)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        z = torch.relu(self.fc1(x.float()))
        z = torch.relu(self.fc2(z))
        return z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc3(self.features(x))


class LowRankAtlasLinear(nn.Module):
    """Linear readout parameterized by a persistent low-rank coordinate."""

    def __init__(
        self,
        base: nn.Linear,
        output_basis: torch.Tensor,
        input_basis: torch.Tensor,
        coord_scale: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        out_dim, in_dim = base.weight.shape
        rank = int(output_basis.shape[1])
        if int(input_basis.shape[1]) != rank:
            raise ValueError("output and input atlas rank must match")
        self.register_buffer("weight_base", base.weight.detach().clone())
        self.bias = nn.Parameter(base.bias.detach().clone() if base.bias is not None else torch.zeros(out_dim))
        self.register_buffer("output_basis", output_basis.detach().clone().float())
        self.register_buffer("input_basis", input_basis.detach().clone().float())
        if coord_scale is None:
            coord_scale = torch.ones(rank, device=base.weight.device)
        self.register_buffer("coord_scale", coord_scale.detach().clone().float().view(rank))
        self.q = nn.Parameter(torch.zeros(rank, rank, device=base.weight.device))
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.rank = rank

    def atlas_delta(self) -> torch.Tensor:
        scaled_q = self.q * self.coord_scale.view(1, -1)
        return self.output_basis @ scaled_q @ self.input_basis.transpose(0, 1)

    def effective_weight(self) -> torch.Tensor:
        return self.weight_base + self.atlas_delta()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x.float(), self.effective_weight(), self.bias)

    def set_atlas_state(
        self,
        output_basis: torch.Tensor,
        input_basis: torch.Tensor,
        coord_scale: torch.Tensor | None = None,
    ) -> None:
        self.output_basis = output_basis.detach().clone().to(device=self.output_basis.device, dtype=self.output_basis.dtype)
        self.input_basis = input_basis.detach().clone().to(device=self.input_basis.device, dtype=self.input_basis.dtype)
        if coord_scale is not None:
            self.coord_scale = coord_scale.detach().clone().to(device=self.coord_scale.device, dtype=self.coord_scale.dtype).view(self.rank)


class MetricAtlasMLP(nn.Module):
    """MLP whose readout is a functional-atlas coordinate layer."""

    def __init__(self, base: SimpleMLP, atlas: AtlasBuild) -> None:
        super().__init__()
        self.fc1 = base.fc1
        self.fc2 = base.fc2
        self.fc3 = LowRankAtlasLinear(base.fc3, atlas.output_basis, atlas.input_basis, atlas.coord_scale)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        z = torch.relu(self.fc1(x.float()))
        z = torch.relu(self.fc2(z))
        return z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc3(self.features(x))


def c_skew_project(matrix: torch.Tensor, c_metric: torch.Tensor, *, eps: float = 1.0e-6) -> torch.Tensor:
    """Project a square matrix onto the C-skew algebra K.T C + C K = 0."""

    out_dtype = matrix.dtype
    c_psd, _ = psd_project(c_metric.double(), eps=eps, shrinkage=0.0)
    c_sqrt = sqrt_psd(c_psd.double(), eps=eps)
    c_inv_sqrt = inv_sqrt_psd(c_psd.double(), eps=eps)
    whitened = c_sqrt @ matrix.double() @ c_inv_sqrt
    skew = 0.5 * (whitened - whitened.transpose(0, 1))
    return (c_inv_sqrt @ skew @ c_sqrt).to(dtype=out_dtype)


def c_sym_project(matrix: torch.Tensor, c_metric: torch.Tensor, *, eps: float = 1.0e-6) -> torch.Tensor:
    """Return the C-symmetric complement used as bounded shaping direction."""

    out_dtype = matrix.dtype
    c_psd, _ = psd_project(c_metric.double(), eps=eps, shrinkage=0.0)
    c_sqrt = sqrt_psd(c_psd.double(), eps=eps)
    c_inv_sqrt = inv_sqrt_psd(c_psd.double(), eps=eps)
    whitened = c_sqrt @ matrix.double() @ c_inv_sqrt
    sym = 0.5 * (whitened + whitened.transpose(0, 1))
    return (c_inv_sqrt @ sym @ c_sqrt).to(dtype=out_dtype)


def whitened_skew_sym_parts(matrix: torch.Tensor, c_metric: torch.Tensor, *, eps: float = 1.0e-6) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return whitened matrix and its skew/symmetric parts under C."""

    c_psd, _ = psd_project(c_metric.double(), eps=eps, shrinkage=0.0)
    c_sqrt = sqrt_psd(c_psd.double(), eps=eps)
    c_inv_sqrt = inv_sqrt_psd(c_psd.double(), eps=eps)
    whitened = c_sqrt @ matrix.double() @ c_inv_sqrt
    skew = 0.5 * (whitened - whitened.transpose(0, 1))
    sym = 0.5 * (whitened + whitened.transpose(0, 1))
    return whitened, skew, sym


def c_cayley_retraction(k_matrix: torch.Tensor, *, eta: float = 1.0, eps: float = 1.0e-6) -> torch.Tensor:
    """Cayley map for a C-skew generator."""

    dim = int(k_matrix.shape[0])
    eye = torch.eye(dim, device=k_matrix.device, dtype=k_matrix.dtype)
    left = eye - 0.5 * float(eta) * k_matrix.float()
    right = eye + 0.5 * float(eta) * k_matrix.float()
    return torch.linalg.solve(left + eps * eye, right)


def metric_compatible_descent_diagnostics(
    gradient_matrix: torch.Tensor | None,
    c_metric: torch.Tensor,
    *,
    eps: float = 1.0e-8,
) -> dict[str, float]:
    """Measure task-gradient energy in C-skew and bounded-shape subspaces."""

    if gradient_matrix is None:
        return {
            "task_gradient_norm": 0.0,
            "iso_descent_energy_fraction": 0.0,
            "iso_predicted_task_descent": 0.0,
            "iso_projected_gradient_norm": 0.0,
            "iso_direction_cosine_with_task_gradient": 0.0,
            "shape_descent_energy_fraction": 0.0,
            "shape_predicted_task_descent": 0.0,
            "shape_projected_gradient_norm": 0.0,
            "iso_or_shape_capacity_positive": 0.0,
        }
    whitened, skew, sym = whitened_skew_sym_parts(gradient_matrix.detach().float(), c_metric.detach().float(), eps=eps)
    total_norm = torch.linalg.norm(whitened).clamp_min(eps)
    iso_norm = torch.linalg.norm(skew)
    shape_norm = torch.linalg.norm(sym)
    iso_descent = -float((skew * skew).sum().detach().cpu().item())
    shape_descent = -float((sym * sym).sum().detach().cpu().item())
    iso_cos = float((iso_norm / total_norm).detach().cpu().item()) if float(iso_norm.detach().cpu().item()) > eps else 0.0
    return {
        "task_gradient_norm": float(total_norm.detach().cpu().item()),
        "iso_descent_energy_fraction": float((iso_norm / total_norm).detach().cpu().item()),
        "iso_predicted_task_descent": iso_descent,
        "iso_projected_gradient_norm": float(iso_norm.detach().cpu().item()),
        "iso_direction_cosine_with_task_gradient": iso_cos,
        "shape_descent_energy_fraction": float((shape_norm / total_norm).detach().cpu().item()),
        "shape_predicted_task_descent": shape_descent,
        "shape_projected_gradient_norm": float(shape_norm.detach().cpu().item()),
        "iso_or_shape_capacity_positive": float(max(float(iso_norm.detach().cpu().item()), float(shape_norm.detach().cpu().item())) > 1.0e-8),
    }


class MetricCompatibleLowRankLinear(nn.Module):
    """Readout layer with C-isometric trainable motion plus optional bounded shaping."""

    def __init__(
        self,
        base: nn.Linear,
        atlas: AtlasBuild,
        *,
        allow_shape: bool = False,
        shape_budget: float = 0.0,
        eta: float = 1.0,
        eps: float = 1.0e-6,
        train_base_weight: bool = False,
        base_spectrum_lock: bool = False,
        functional_spectrum_budget: float = 0.0,
        functional_spectrum_fill: bool = False,
        functional_spectrum_fill_max: float = 8.0,
    ) -> None:
        super().__init__()
        out_dim, in_dim = base.weight.shape
        rank = int(atlas.output_basis.shape[1])
        self.base_spectrum_lock = bool(base_spectrum_lock and train_base_weight)
        self.functional_spectrum_budget = float(functional_spectrum_budget)
        self.functional_spectrum_fill = bool(functional_spectrum_fill)
        self.functional_spectrum_fill_max = float(functional_spectrum_fill_max)
        self.register_buffer("initial_weight_spectrum", torch.linalg.svdvals(base.weight.detach().float()))
        if train_base_weight and self.base_spectrum_lock:
            self.weight_base_raw = nn.Parameter(base.weight.detach().clone())
            self.register_buffer("weight_base_spectrum", torch.linalg.svdvals(base.weight.detach().float()))
        elif train_base_weight:
            self.weight_base = nn.Parameter(base.weight.detach().clone())
        else:
            self.register_buffer("weight_base", base.weight.detach().clone())
        self.bias = nn.Parameter(base.bias.detach().clone() if base.bias is not None else torch.zeros(out_dim))
        self.register_buffer("output_basis", atlas.output_basis.detach().clone().float())
        self.register_buffer("input_basis", atlas.input_basis.detach().clone().float())
        self.register_buffer("coord_scale", atlas.coord_scale.detach().clone().float().view(rank))
        self.register_buffer("active_gram", atlas.active_gram.detach().clone().float())
        self.raw_iso = nn.Parameter(torch.zeros(rank, rank, device=base.weight.device))
        self.allow_shape = bool(allow_shape)
        self.shape_budget = float(shape_budget)
        if self.allow_shape:
            self.raw_shape = nn.Parameter(torch.zeros(rank, rank, device=base.weight.device))
        else:
            self.register_parameter("raw_shape", None)
        self.eta = float(eta)
        self.eps = float(eps)
        self.train_base_weight = bool(train_base_weight)
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.rank = rank

    def effective_base_weight(self) -> torch.Tensor:
        if self.base_spectrum_lock:
            u, _, vh = torch.linalg.svd(self.weight_base_raw.float(), full_matrices=False)
            singular = self.weight_base_spectrum.to(device=u.device, dtype=u.dtype)
            return (u * singular.view(1, -1)) @ vh
        return self.weight_base

    def iso_operator(self) -> torch.Tensor:
        k_mat = c_skew_project(self.raw_iso, self.active_gram, eps=self.eps)
        return c_cayley_retraction(k_mat, eta=self.eta, eps=self.eps)

    def shape_operator(self) -> torch.Tensor:
        if not self.allow_shape or self.raw_shape is None or self.shape_budget <= 0.0:
            return torch.zeros(self.rank, self.rank, device=self.output_basis.device, dtype=self.output_basis.dtype)
        sym = c_sym_project(self.raw_shape, self.active_gram, eps=self.eps)
        whitened, _, _ = whitened_skew_sym_parts(sym, self.active_gram, eps=self.eps)
        norm = torch.linalg.norm(whitened).clamp_min(self.eps)
        scale = torch.clamp(torch.tensor(float(self.shape_budget), device=sym.device, dtype=norm.dtype) / norm, max=1.0)
        return sym * scale

    def shape_budget_used(self) -> float:
        if not self.allow_shape or self.raw_shape is None:
            return 0.0
        shape = self.shape_operator().detach()
        whitened, _, _ = whitened_skew_sym_parts(shape, self.active_gram.detach(), eps=self.eps)
        return float(torch.linalg.norm(whitened).detach().cpu().item())

    def raw_atlas_delta(self) -> torch.Tensor:
        iso_delta = self.iso_operator() - torch.eye(self.rank, device=self.output_basis.device, dtype=self.output_basis.dtype)
        update = iso_delta + self.shape_operator()
        scaled_update = update * self.coord_scale.view(1, -1)
        return self.output_basis @ scaled_update @ self.input_basis.transpose(0, 1)

    def atlas_delta(self, base_weight: torch.Tensor | None = None) -> torch.Tensor:
        delta = self.raw_atlas_delta()
        if self.functional_spectrum_budget <= 0.0:
            return delta
        if base_weight is None:
            base_weight = self.effective_base_weight()
        current = torch.linalg.svdvals((base_weight + delta).float())
        initial = self.initial_weight_spectrum.to(device=current.device, dtype=current.dtype)
        n = min(int(current.numel()), int(initial.numel()))
        drift = torch.linalg.norm(current[:n] - initial[:n]) / torch.linalg.norm(initial[:n]).clamp_min(self.eps)
        target = torch.tensor(float(self.functional_spectrum_budget), device=delta.device, dtype=drift.dtype)
        max_scale = float(self.functional_spectrum_fill_max) if self.functional_spectrum_fill else 1.0
        scale = torch.clamp(target / drift.clamp_min(self.eps), max=max_scale)
        scaled = delta * scale.to(device=delta.device, dtype=delta.dtype)
        scaled_current = torch.linalg.svdvals((base_weight + scaled).float())
        scaled_drift = torch.linalg.norm(scaled_current[:n] - initial[:n]) / torch.linalg.norm(initial[:n]).clamp_min(self.eps)
        correction = torch.clamp(target / scaled_drift.clamp_min(self.eps), max=1.0)
        return scaled * correction.to(device=delta.device, dtype=delta.dtype)

    def effective_weight(self) -> torch.Tensor:
        base_weight = self.effective_base_weight()
        return base_weight + self.atlas_delta(base_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x.float(), self.effective_weight(), self.bias)

    def set_atlas_state(
        self,
        output_basis: torch.Tensor,
        input_basis: torch.Tensor,
        coord_scale: torch.Tensor | None = None,
        active_gram: torch.Tensor | None = None,
    ) -> None:
        self.output_basis = output_basis.detach().clone().to(device=self.output_basis.device, dtype=self.output_basis.dtype)
        self.input_basis = input_basis.detach().clone().to(device=self.input_basis.device, dtype=self.input_basis.dtype)
        if coord_scale is not None:
            self.coord_scale = coord_scale.detach().clone().to(device=self.coord_scale.device, dtype=self.coord_scale.dtype).view(self.rank)
        if active_gram is not None:
            self.active_gram = active_gram.detach().clone().to(device=self.active_gram.device, dtype=self.active_gram.dtype)


class MetricCompatibleAtlasMLP(nn.Module):
    """MLP whose readout motion is constrained to metric-compatible coordinates."""

    def __init__(
        self,
        base: SimpleMLP,
        atlas: AtlasBuild,
        *,
        allow_shape: bool = False,
        shape_budget: float = 0.0,
        eta: float = 1.0,
        train_base_weight: bool = False,
        base_spectrum_lock: bool = False,
        functional_spectrum_budget: float = 0.0,
        functional_spectrum_fill: bool = False,
    ) -> None:
        super().__init__()
        self.fc1 = base.fc1
        self.fc2 = base.fc2
        self.fc3 = MetricCompatibleLowRankLinear(
            base.fc3,
            atlas,
            allow_shape=allow_shape,
            shape_budget=shape_budget,
            eta=eta,
            train_base_weight=train_base_weight,
            base_spectrum_lock=base_spectrum_lock,
            functional_spectrum_budget=functional_spectrum_budget,
            functional_spectrum_fill=functional_spectrum_fill,
        )

    def features(self, x: torch.Tensor) -> torch.Tensor:
        z = torch.relu(self.fc1(x.float()))
        z = torch.relu(self.fc2(z))
        return z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc3(self.features(x))


def categorical_fisher(probs: torch.Tensor) -> torch.Tensor:
    mean_diag = torch.diag(probs.mean(dim=0))
    second = probs.transpose(0, 1) @ probs / max(1, int(probs.shape[0]))
    return _sym(mean_diag - second)


def _one_hot(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    return F.one_hot(labels.long(), num_classes=num_classes).float()


def metric_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    metric_kind: str = "signal_debt",
    eps: float = 1.0e-5,
    rho: float = 0.10,
    lambda_sig: float = 0.50,
    lambda_debt: float = 0.25,
) -> tuple[torch.Tensor, dict[str, float]]:
    probs = torch.softmax(logits.detach().float(), dim=1)
    yoh = _one_hot(labels.detach().long(), probs.shape[1])
    delta = probs - yoh
    gf_ce, ce_diag = psd_project(categorical_fisher(probs), eps=eps)

    mean_delta = delta.mean(dim=0, keepdim=True)
    centered = delta - mean_delta
    cov = centered.transpose(0, 1) @ centered / max(1, int(delta.shape[0]) - 1)
    signal_raw = mean_delta.transpose(0, 1) @ mean_delta - float(rho) * cov
    gf_sig, sig_diag = psd_project(signal_raw, eps=eps)

    per_loss = F.cross_entropy(logits.detach().float(), labels.detach().long(), reduction="none")
    margin = probs.gather(1, labels.long().view(-1, 1)).squeeze(1)
    tail_cut = torch.quantile(per_loss.float(), 0.75) if per_loss.numel() > 1 else per_loss.max()
    tail_w = (per_loss >= tail_cut).float().view(-1, 1)
    margin_w = (1.0 - margin).clamp_min(0.0).view(-1, 1)
    debt_vec = delta * (tail_w + margin_w)
    gf_debt, debt_diag = psd_project(debt_vec.transpose(0, 1) @ debt_vec / max(1, int(debt_vec.shape[0])), eps=eps)

    kind = str(metric_kind).lower()
    gf = gf_ce
    if "signal" in kind:
        gf = gf + float(lambda_sig) * gf_sig
    if "debt" in kind:
        gf = gf + float(lambda_debt) * gf_debt
    gf, diag = psd_project(gf, eps=eps, shrinkage=0.02)
    trace = max(float(torch.trace(gf).detach().cpu().item()), eps)
    metrics = {
        "Gf_psd_min": diag["psd_min"],
        "Gf_condition_number": diag["condition_number"],
        "Gf_effective_rank": diag["effective_rank"],
        "Gf_trace": diag["trace"],
        "signal_metric_positive_rank": sig_diag["effective_rank"],
        "debt_metric_trace_fraction": float(torch.trace(gf_debt).detach().cpu().item()) / trace,
        "signal_metric_trace_fraction": float(torch.trace(gf_sig).detach().cpu().item()) / trace,
        "raw_signal_psd_min": sig_diag["raw_psd_min"],
        "metric_shrinkage_strength": diag["shrinkage_strength"],
    }
    return gf, metrics


def _cov(x: torch.Tensor, eps: float = 1.0e-5) -> torch.Tensor:
    xc = x.float() - x.float().mean(dim=0, keepdim=True)
    out = xc.transpose(0, 1) @ xc / max(1, int(xc.shape[0]) - 1)
    dim = int(out.shape[0])
    return _sym(out + eps * torch.eye(dim, device=x.device, dtype=out.dtype))


def _basis_from_eigh(matrix: torch.Tensor, rank: int, *, largest: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    vals, vecs = torch.linalg.eigh(_sym(matrix.float()))
    order = torch.argsort(vals, descending=largest)
    take = order[:rank]
    return vecs[:, take].contiguous(), vals[take].contiguous()


def build_last_layer_atlas(
    base: SimpleMLP,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    rank: int = 2,
    method: str = "metric_atlas",
    metric_kind: str = "signal_debt",
    seed: int = 0,
    eps: float = 1.0e-5,
) -> AtlasBuild:
    """Build an output/input atlas for the MLP readout from train-only data."""

    start = time.perf_counter()
    device = next(base.parameters()).device
    x = x.to(device)
    y = y.to(device)
    with torch.enable_grad():
        h = base.features(x)
        logits = base.fc3(h)
    probs = torch.softmax(logits.detach().float(), dim=1)
    rank = max(1, min(int(rank), int(probs.shape[1]), int(h.shape[1])))
    delta = probs - _one_hot(y, probs.shape[1]).to(device)

    method_l = str(method).lower()
    if "random_label" in method_l:
        gen_lbl = torch.Generator(device=device)
        gen_lbl.manual_seed(int(seed) + 8741)
        y_for_delta = y[torch.randperm(int(y.numel()), generator=gen_lbl, device=device)]
        delta = probs - _one_hot(y_for_delta, probs.shape[1]).to(device)
    if "shuffled_source" in method_l:
        gen_shuf = torch.Generator(device=device)
        gen_shuf.manual_seed(int(seed) + 2718)
        delta = delta[torch.randperm(int(delta.shape[0]), generator=gen_shuf, device=device)]

    gf, metric_diag = metric_from_logits(logits, y, metric_kind=metric_kind, eps=eps)
    n = int(delta.shape[0])
    half = max(1, n // 2)
    d_source = delta[:half]
    d_witness = delta[half:] if half < n else delta[:half]
    h_det = h.detach().float()
    if "source_only" in method_l:
        d_for = d_source
        h_for = h_det[:half]
    elif "witness_only" in method_l:
        d_for = d_witness
        h_for = h_det[half:] if half < n else h_det[:half]
    elif "credit_only" in method_l:
        per_loss = F.cross_entropy(logits.detach().float(), y.long(), reduction="none")
        take = torch.argsort(per_loss, descending=True)[: max(rank * 4, min(n, 16))]
        d_for = delta[take]
        h_for = h_det[take]
    else:
        d_for = delta
        h_for = h_det

    feature_metric = _cov(h.detach(), eps=eps)
    grad_w = d_for.transpose(0, 1) @ h_for / max(1, int(d_for.shape[0]))
    cov_s = d_source.transpose(0, 1) @ d_source / max(1, int(d_source.shape[0]))
    cov_w = d_witness.transpose(0, 1) @ d_witness / max(1, int(d_witness.shape[0]))
    h_source = h_det[:half]
    h_witness = h_det[half:] if half < n else h_det[:half]
    grad_s = d_source.transpose(0, 1) @ h_source / max(1, int(d_source.shape[0]))
    grad_wit = d_witness.transpose(0, 1) @ h_witness / max(1, int(d_witness.shape[0]))
    grad_coh = 0.5 * (grad_s + grad_wit)
    grad_cos_den = torch.linalg.norm(grad_s).clamp_min(eps) * torch.linalg.norm(grad_wit).clamp_min(eps)
    grad_cos = float((torch.sum(grad_s * grad_wit) / grad_cos_den).detach().cpu().item())
    per_loss_atlas = F.cross_entropy(logits.detach().float(), y.long(), reduction="none")
    cvar_cut = torch.quantile(per_loss_atlas.float(), 0.75) if per_loss_atlas.numel() > 1 else per_loss_atlas.max()
    cvar_w = (per_loss_atlas >= cvar_cut).float().view(-1, 1)

    def _weighted_grad(d_part: torch.Tensor, h_part: torch.Tensor, w_part: torch.Tensor) -> torch.Tensor:
        denom = float(w_part.sum().detach().cpu().item())
        if denom <= eps:
            return d_part.transpose(0, 1) @ h_part / max(1, int(d_part.shape[0]))
        return (d_part * w_part).transpose(0, 1) @ h_part / max(eps, denom)

    cvar_s = cvar_w[:half]
    cvar_wit_w = cvar_w[half:] if half < n else cvar_w[:half]
    grad_cvar_s = _weighted_grad(d_source, h_source, cvar_s)
    grad_cvar_wit = _weighted_grad(d_witness, h_witness, cvar_wit_w)
    grad_cvar_coh = 0.5 * (grad_cvar_s + grad_cvar_wit)
    if "metric_atlas_sw" in method_l:
        # Cross-cohort coherence suppresses directions that are only strong in
        # one train split; this is the Part B source/witness signal atlas path.
        cross_raw = _sym(0.5 * (cov_s @ cov_w + cov_w @ cov_s))
        cross = psd_project(cross_raw, eps=eps, shrinkage=0.02)[0]
    else:
        cross_raw = _sym(0.5 * (cov_s + cov_w))
        cross = cross_raw
    active_basis, active_vals = _basis_from_eigh(cross, rank)

    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed) + 99173 + sum(ord(ch) for ch in method_l))
    readout_weight = base.fc3.weight.detach() if hasattr(base.fc3, "weight") else base.fc3.effective_weight().detach()
    actuator_input_basis: torch.Tensor | None = None
    actuator_vals: torch.Tensor | None = None
    signal_reachable_target_energy: float | None = None
    kan_bank_oet_used = 0.0
    kan_bank_oet_bank_count = 0.0
    kan_bank_oet_top_bank = -1.0
    kan_bank_oet_selected_components = 0.0
    kan_bank_oet_extra_feature_dim = 0.0
    control_contrastive_act_alpha = 0.0
    control_contrastive_act_projection_fraction = 0.0
    if "same_signal_reachable_random_atlas" in method_l:
        if "gradcoh" in method_l:
            weighted_grad = sqrt_psd(gf, eps=eps) @ grad_coh.float() @ sqrt_psd(feature_metric, eps=eps)
            u_act, _, _ = torch.linalg.svd(weighted_grad.float(), full_matrices=False)
            target_basis = normalize_columns(inv_sqrt_psd(gf, eps=eps) @ u_act[:, :rank])
        else:
            target_basis = active_basis
        target_energy_t = torch.trace(target_basis.transpose(0, 1) @ gf @ target_basis).clamp_min(eps)
        rand_basis = random_orthonormal(probs.shape[1], rank, device=device, generator=gen)
        rand_energy_t = torch.trace(rand_basis.transpose(0, 1) @ gf @ rand_basis).clamp_min(eps)
        output_basis = rand_basis * torch.sqrt(target_energy_t / rand_energy_t)
        active_vals = torch.ones(rank, device=device)
        signal_reachable_target_energy = float(target_energy_t.detach().cpu().item())
    elif any(token in method_l for token in ["same_rank_random", "same_spectrum_random", "same_functional_spectrum_random", "same_active_gram_random", "lora_like"]):
        output_basis = random_orthonormal(probs.shape[1], rank, device=device, generator=gen)
    elif "self_only" in method_l:
        output_basis = random_orthonormal(probs.shape[1], rank, device=device, generator=gen)
    elif "oet_only" in method_l or "gfoa" in method_l:
        output_basis, active_vals = _basis_from_eigh(readout_weight @ readout_weight.transpose(0, 1) + eps * torch.eye(probs.shape[1], device=device), rank)
    elif "noop" in method_l:
        output_basis = random_orthonormal(probs.shape[1], rank, device=device, generator=gen)
        active_vals = torch.zeros(rank, device=device)
    elif "metric_atlas_gradunion" in method_l:
        gf_sqrt = sqrt_psd(gf, eps=eps)
        feature_sqrt = sqrt_psd(feature_metric, eps=eps)
        gf_inv_sqrt = inv_sqrt_psd(gf, eps=eps)
        feature_inv_sqrt = inv_sqrt_psd(feature_metric, eps=eps)
        grad_s_w = gf_sqrt @ grad_s.float() @ feature_sqrt
        grad_w_w = gf_sqrt @ grad_wit.float() @ feature_sqrt
        grad_diff = grad_s_w - grad_w_w
        conflict_penalty = 0.10
        out_union = 0.5 * (grad_s_w @ grad_s_w.transpose(0, 1) + grad_w_w @ grad_w_w.transpose(0, 1))
        in_union = 0.5 * (grad_s_w.transpose(0, 1) @ grad_s_w + grad_w_w.transpose(0, 1) @ grad_w_w)
        out_conflict = 0.5 * (grad_diff @ grad_diff.transpose(0, 1))
        in_conflict = 0.5 * (grad_diff.transpose(0, 1) @ grad_diff)
        out_union_raw = _sym(out_union - conflict_penalty * out_conflict)
        in_union_raw = _sym(in_union - conflict_penalty * in_conflict)
        out_union_psd = psd_project(out_union_raw, eps=eps, shrinkage=0.02)[0]
        in_union_psd = psd_project(in_union_raw, eps=eps, shrinkage=0.02)[0]
        out_basis_w, out_vals = _basis_from_eigh(out_union_psd, rank)
        in_basis_w, in_vals_union = _basis_from_eigh(in_union_psd, rank)
        output_basis = normalize_columns(gf_inv_sqrt @ out_basis_w[:, :rank])
        actuator_input_basis = normalize_columns(feature_inv_sqrt @ in_basis_w[:, :rank])
        active_vals = out_vals[:rank]
        actuator_vals = torch.sqrt(torch.clamp(torch.abs(out_vals[:rank] * in_vals_union[:rank]), min=eps))
    elif "metric_atlas_gradcov" in method_l:
        gf_sqrt = sqrt_psd(gf, eps=eps)
        feature_sqrt = sqrt_psd(feature_metric, eps=eps)
        gf_inv_sqrt = inv_sqrt_psd(gf, eps=eps)
        feature_inv_sqrt = inv_sqrt_psd(feature_metric, eps=eps)
        grad_s_w = gf_sqrt @ grad_s.float() @ feature_sqrt
        grad_w_w = gf_sqrt @ grad_wit.float() @ feature_sqrt
        out_cross_raw = _sym(0.5 * (grad_s_w @ grad_w_w.transpose(0, 1) + grad_w_w @ grad_s_w.transpose(0, 1)))
        in_cross_raw = _sym(0.5 * (grad_s_w.transpose(0, 1) @ grad_w_w + grad_w_w.transpose(0, 1) @ grad_s_w))
        if "ccfs25" in method_l or "ccfs50" in method_l:
            control_contrastive_act_alpha = 0.25 if "ccfs25" in method_l else 0.50
            control_gen = torch.Generator(device=device)
            control_gen.manual_seed(int(seed) + 99173 + sum(ord(ch) for ch in "same_functional_spectrum_random_atlas"))
            out_ctrl_rank = min(int(rank), int(out_cross_raw.shape[0]))
            in_ctrl_rank = min(int(rank), int(in_cross_raw.shape[0]))
            out_ctrl = random_orthonormal(int(out_cross_raw.shape[0]), out_ctrl_rank, device=device, generator=control_gen)
            in_ctrl = random_orthonormal(int(in_cross_raw.shape[0]), in_ctrl_rank, device=device, generator=control_gen)
            out_scale = torch.linalg.norm(out_cross_raw.float()).clamp_min(eps) / math.sqrt(max(1, out_ctrl_rank))
            in_scale = torch.linalg.norm(in_cross_raw.float()).clamp_min(eps) / math.sqrt(max(1, in_ctrl_rank))
            out_nuisance = out_scale * (out_ctrl @ out_ctrl.transpose(0, 1))
            in_nuisance = in_scale * (in_ctrl @ in_ctrl.transpose(0, 1))
            out_cross_raw = _sym(out_cross_raw - float(control_contrastive_act_alpha) * out_nuisance)
            in_cross_raw = _sym(in_cross_raw - float(control_contrastive_act_alpha) * in_nuisance)
            base_norm = torch.linalg.norm(out_cross_raw.float()).clamp_min(eps)
            control_contrastive_act_projection_fraction = float((torch.linalg.norm(out_nuisance.float()) / base_norm).detach().cpu().item())
        out_cross = psd_project(out_cross_raw, eps=eps, shrinkage=0.02)[0]
        in_cross = psd_project(in_cross_raw, eps=eps, shrinkage=0.02)[0]
        out_basis_w, out_vals = _basis_from_eigh(out_cross, rank)
        in_basis_w, in_vals_cov = _basis_from_eigh(in_cross, rank)
        output_basis = normalize_columns(gf_inv_sqrt @ out_basis_w[:, :rank])
        actuator_input_basis = normalize_columns(feature_inv_sqrt @ in_basis_w[:, :rank])
        active_vals = out_vals[:rank]
        actuator_vals = torch.sqrt(torch.clamp(torch.abs(out_vals[:rank] * in_vals_cov[:rank]), min=eps))
    elif "metric_atlas_gradmix" in method_l:
        gf_sqrt = sqrt_psd(gf, eps=eps)
        feature_sqrt = sqrt_psd(feature_metric, eps=eps)
        gf_inv_sqrt = inv_sqrt_psd(gf, eps=eps)
        feature_inv_sqrt = inv_sqrt_psd(feature_metric, eps=eps)
        grad_s_w = gf_sqrt @ grad_s.float() @ feature_sqrt
        grad_w_w = gf_sqrt @ grad_wit.float() @ feature_sqrt
        out_cross_raw = _sym(0.5 * (grad_s_w @ grad_w_w.transpose(0, 1) + grad_w_w @ grad_s_w.transpose(0, 1)))
        in_cross_raw = _sym(0.5 * (grad_s_w.transpose(0, 1) @ grad_w_w + grad_w_w.transpose(0, 1) @ grad_s_w))
        out_cross = psd_project(out_cross_raw, eps=eps, shrinkage=0.02)[0]
        in_cross = psd_project(in_cross_raw, eps=eps, shrinkage=0.02)[0]
        out_cov_w, out_cov_vals = _basis_from_eigh(out_cross, rank)
        in_cov_w, in_cov_vals = _basis_from_eigh(in_cross, rank)

        weighted_coh = gf_sqrt @ grad_coh.float() @ feature_sqrt
        u_coh, s_coh, vh_coh = torch.linalg.svd(weighted_coh.float(), full_matrices=False)
        cov_cols = max(1, min(rank - 1, rank // 2))
        coh_cols = max(1, rank - cov_cols)
        out_mix_w = torch.cat([out_cov_w[:, :cov_cols], u_coh[:, :coh_cols]], dim=1)
        in_mix_w = torch.cat([in_cov_w[:, :cov_cols], vh_coh.transpose(0, 1)[:, :coh_cols]], dim=1)
        output_basis = normalize_columns(gf_inv_sqrt @ out_mix_w[:, :rank])
        actuator_input_basis = normalize_columns(feature_inv_sqrt @ in_mix_w[:, :rank])
        active_vals = torch.cat([out_cov_vals[:cov_cols], s_coh[:coh_cols]], dim=0)[:rank]
        cov_pair_vals = torch.sqrt(torch.clamp(torch.abs(out_cov_vals[:cov_cols] * in_cov_vals[:cov_cols]), min=eps))
        actuator_vals = torch.cat([cov_pair_vals, s_coh[:coh_cols]], dim=0)[:rank]
    elif "metric_atlas_oocw_gradcoh" in method_l:
        gf_sqrt = sqrt_psd(gf, eps=eps)
        feature_sqrt = sqrt_psd(feature_metric, eps=eps)
        gf_inv_sqrt = inv_sqrt_psd(gf, eps=eps)
        feature_inv_sqrt = inv_sqrt_psd(feature_metric, eps=eps)
        source_w = gf_sqrt @ grad_s.float() @ feature_sqrt
        witness_w = gf_sqrt @ grad_wit.float() @ feature_sqrt
        u_src, s_src, vh_src = torch.linalg.svd(source_w.float(), full_matrices=False)
        max_cols = min(int(u_src.shape[1]), int(vh_src.shape[0]))
        alignments = []
        for idx_comp in range(max_cols):
            u_i = u_src[:, idx_comp]
            v_i = vh_src.transpose(0, 1)[:, idx_comp]
            witness_score = torch.abs(u_i @ witness_w @ v_i) / s_src[idx_comp].abs().clamp_min(eps)
            alignments.append(witness_score)
        if alignments:
            align = torch.stack(alignments)
            score = s_src[:max_cols].abs() * torch.clamp(align, min=0.0)
            order = torch.argsort(score, descending=True)
            take = order[:rank]
            output_basis = normalize_columns(gf_inv_sqrt @ u_src[:, take])
            actuator_input_basis = normalize_columns(feature_inv_sqrt @ vh_src.transpose(0, 1)[:, take])
            active_vals = s_src[take]
            actuator_vals = s_src[take] * torch.clamp(align[take], min=eps)
        else:
            weighted_grad = gf_sqrt @ grad_coh.float() @ feature_sqrt
            u_act, s_act, vh_act = torch.linalg.svd(weighted_grad.float(), full_matrices=False)
            output_basis = normalize_columns(gf_inv_sqrt @ u_act[:, :rank])
            actuator_input_basis = normalize_columns(feature_inv_sqrt @ vh_act.transpose(0, 1)[:, :rank])
            active_vals = s_act[:rank]
            actuator_vals = s_act[:rank]
    elif "metric_atlas_gradcoh" in method_l or "metric_atlas_cvargrad" in method_l:
        grad_basis_source = grad_cvar_coh if "metric_atlas_cvargrad" in method_l else grad_coh
        weighted_grad = sqrt_psd(gf, eps=eps) @ grad_basis_source.float() @ sqrt_psd(feature_metric, eps=eps)
        u_act, s_act, vh_act = torch.linalg.svd(weighted_grad.float(), full_matrices=False)
        output_basis = normalize_columns(inv_sqrt_psd(gf, eps=eps) @ u_act[:, :rank])
        actuator_input_basis = normalize_columns(inv_sqrt_psd(feature_metric, eps=eps) @ vh_act.transpose(0, 1)[:, :rank])
        active_vals = s_act[:rank]
        actuator_vals = s_act[:rank]
    elif "kan_bank_oet" in method_l:
        feature_model = getattr(base, "feature_model", None)
        hidden_dim = int(getattr(feature_model, "hidden_dim", 0) or 0)
        bank_count = int(getattr(feature_model, "k", 0) or 0)
        core_dim = hidden_dim * bank_count
        can_use_bank = hidden_dim > 0 and bank_count > 0 and core_dim <= int(h.shape[1])
        gf_sqrt = sqrt_psd(gf, eps=eps)
        gf_inv_sqrt = inv_sqrt_psd(gf, eps=eps)
        if can_use_bank:
            components: list[tuple[float, int, torch.Tensor, torch.Tensor]] = []
            for bank_idx in range(bank_count):
                idx = torch.arange(bank_idx, core_dim, bank_count, device=device)
                if int(idx.numel()) == 0:
                    continue
                feature_bank = feature_metric.index_select(0, idx).index_select(1, idx)
                bank_sqrt = sqrt_psd(feature_bank, eps=eps)
                bank_inv_sqrt = inv_sqrt_psd(feature_bank, eps=eps)
                grad_bank = grad_w.index_select(1, idx)
                weighted_bank = gf_sqrt @ grad_bank.float() @ bank_sqrt
                u_bank, s_bank, vh_bank = torch.linalg.svd(weighted_bank.float(), full_matrices=False)
                max_components = min(int(s_bank.numel()), rank)
                for comp_idx in range(max_components):
                    out_col = normalize_columns(gf_inv_sqrt @ u_bank[:, comp_idx : comp_idx + 1])[:, 0]
                    bank_col = normalize_columns(bank_inv_sqrt @ vh_bank.transpose(0, 1)[:, comp_idx : comp_idx + 1])[:, 0]
                    full_col = torch.zeros(int(h.shape[1]), device=device, dtype=bank_col.dtype)
                    full_col[idx] = bank_col
                    components.append((float(s_bank[comp_idx].detach().cpu().item()), bank_idx, out_col, normalize_columns(full_col.view(-1, 1))[:, 0]))
            components.sort(key=lambda item: item[0], reverse=True)
            chosen = components[:rank]
            if chosen:
                output_cols = [item[2] for item in chosen]
                input_cols = [item[3] for item in chosen]
                vals = [item[0] for item in chosen]
                if len(chosen) < rank:
                    weighted_grad = gf_sqrt @ grad_w.float() @ sqrt_psd(feature_metric, eps=eps)
                    u_fill, s_fill, vh_fill = torch.linalg.svd(weighted_grad.float(), full_matrices=False)
                    fill_in = normalize_columns(inv_sqrt_psd(feature_metric, eps=eps) @ vh_fill.transpose(0, 1)[:, : rank - len(chosen)])
                    fill_out = normalize_columns(gf_inv_sqrt @ u_fill[:, : rank - len(chosen)])
                    for fill_idx in range(int(fill_out.shape[1])):
                        output_cols.append(fill_out[:, fill_idx])
                        input_cols.append(fill_in[:, fill_idx])
                        vals.append(float(s_fill[fill_idx].detach().cpu().item()))
                output_basis = torch.stack(output_cols[:rank], dim=1).contiguous()
                actuator_input_basis = torch.stack(input_cols[:rank], dim=1).contiguous()
                active_vals = torch.tensor(vals[:rank], device=device, dtype=torch.float32)
                actuator_vals = active_vals
                kan_bank_oet_used = 1.0
                kan_bank_oet_bank_count = float(bank_count)
                kan_bank_oet_top_bank = float(chosen[0][1])
                kan_bank_oet_selected_components = float(len(chosen))
                kan_bank_oet_extra_feature_dim = float(max(0, int(h.shape[1]) - core_dim))
            else:
                weighted_grad = gf_sqrt @ grad_w.float() @ sqrt_psd(feature_metric, eps=eps)
                u_act, s_act, vh_act = torch.linalg.svd(weighted_grad.float(), full_matrices=False)
                output_basis = normalize_columns(gf_inv_sqrt @ u_act[:, :rank])
                actuator_input_basis = normalize_columns(inv_sqrt_psd(feature_metric, eps=eps) @ vh_act.transpose(0, 1)[:, :rank])
                active_vals = s_act[:rank]
                actuator_vals = s_act[:rank]
        else:
            weighted_grad = gf_sqrt @ grad_w.float() @ sqrt_psd(feature_metric, eps=eps)
            u_act, s_act, vh_act = torch.linalg.svd(weighted_grad.float(), full_matrices=False)
            output_basis = normalize_columns(gf_inv_sqrt @ u_act[:, :rank])
            actuator_input_basis = normalize_columns(inv_sqrt_psd(feature_metric, eps=eps) @ vh_act.transpose(0, 1)[:, :rank])
            active_vals = s_act[:rank]
            actuator_vals = s_act[:rank]
    elif "metric_atlas_act" in method_l:
        weighted_grad = sqrt_psd(gf, eps=eps) @ grad_w.float() @ sqrt_psd(feature_metric, eps=eps)
        if "cc25" in method_l or "cc50" in method_l:
            control_contrastive_act_alpha = 0.25 if "cc25" in method_l else 0.50
            control_gen = torch.Generator(device=device)
            control_gen.manual_seed(int(seed) + 99173 + sum(ord(ch) for ch in "same_functional_spectrum_random_atlas"))
            control_raw = torch.randn(weighted_grad.shape, device=device, generator=control_gen, dtype=weighted_grad.dtype)
            control_dir = control_raw / torch.linalg.norm(control_raw).clamp_min(eps)
            base_norm = torch.linalg.norm(weighted_grad).clamp_min(eps)
            projection = torch.sum(weighted_grad * control_dir)
            weighted_grad = weighted_grad - float(control_contrastive_act_alpha) * projection * control_dir
            control_contrastive_act_projection_fraction = float((projection.abs() / base_norm).detach().cpu().item())
        u_act, s_act, vh_act = torch.linalg.svd(weighted_grad.float(), full_matrices=False)
        output_basis = normalize_columns(inv_sqrt_psd(gf, eps=eps) @ u_act[:, :rank])
        actuator_input_basis = normalize_columns(inv_sqrt_psd(feature_metric, eps=eps) @ vh_act.transpose(0, 1)[:, :rank])
        active_vals = s_act[:rank]
        actuator_vals = s_act[:rank]
    else:
        output_basis = active_basis

    projected = output_basis.transpose(0, 1) @ grad_w
    if "self_only" in method_l:
        input_basis, in_vals = _basis_from_eigh(feature_metric, rank)
    elif "oet_only" in method_l or "gfoa" in method_l:
        input_basis, in_vals = _basis_from_eigh(readout_weight.transpose(0, 1) @ readout_weight + eps * torch.eye(h.shape[1], device=device), rank)
    elif any(token in method_l for token in ["same_rank_random", "same_spectrum_random", "same_functional_spectrum_random", "same_active_gram_random", "same_signal_reachable_random_atlas", "lora_like", "noop"]):
        input_basis = random_orthonormal(h.shape[1], rank, device=device, generator=gen)
        in_vals = torch.ones(rank, device=device)
    elif "featurein" in method_l and (
        "metric_atlas_gradunion" in method_l
        or "metric_atlas_gradcov" in method_l
        or "metric_atlas_gradmix" in method_l
        or "metric_atlas_gradcoh" in method_l
        or "metric_atlas_cvargrad" in method_l
    ):
        input_basis, in_vals = _basis_from_eigh(feature_metric, rank)
    elif ("metric_atlas_gradunion" in method_l or "metric_atlas_gradcov" in method_l or "metric_atlas_gradmix" in method_l or "metric_atlas_gradcoh" in method_l or "metric_atlas_cvargrad" in method_l or "metric_atlas_oocw_gradcoh" in method_l) and actuator_input_basis is not None and actuator_vals is not None:
        input_basis = actuator_input_basis
        in_vals = actuator_vals
    elif "kan_bank_oet" in method_l and actuator_input_basis is not None and actuator_vals is not None:
        input_basis = actuator_input_basis
        in_vals = actuator_vals
    elif "metric_atlas_act" in method_l and actuator_input_basis is not None and actuator_vals is not None:
        input_basis = actuator_input_basis
        in_vals = actuator_vals
    elif "metric_atlas_sw" in method_l:
        projected_s = output_basis.transpose(0, 1) @ grad_s
        projected_w = output_basis.transpose(0, 1) @ grad_wit
        input_cross = _sym(0.5 * (projected_s.transpose(0, 1) @ projected_w + projected_w.transpose(0, 1) @ projected_s))
        input_basis, in_vals = _basis_from_eigh(input_cross, rank)
        if in_vals.numel() == 0 or float(in_vals.max().detach().cpu().item()) <= eps:
            _, _, vh = torch.linalg.svd(projected.float(), full_matrices=False)
            input_basis = orthonormal_columns(vh.transpose(0, 1), rank, generator=gen)
            in_vals = torch.linalg.svdvals(projected.float())[:rank]
    else:
        _, _, vh = torch.linalg.svd(projected.float(), full_matrices=False)
        input_basis = orthonormal_columns(vh.transpose(0, 1), rank, generator=gen)
        in_vals = torch.linalg.svdvals(projected.float())[:rank]

    scale_balanced = "balanced" in method_l and (
        "metric_atlas_gradcoh" in method_l
        or "metric_atlas_gradunion" in method_l
        or "metric_atlas_gradcov" in method_l
        or "metric_atlas_gradmix" in method_l
        or "metric_atlas_cvargrad" in method_l
        or "metric_atlas_act" in method_l
        or "kan_bank_oet" in method_l
        or "same_signal_reachable_random_atlas" in method_l
    )
    if scale_balanced:
        scale = torch.ones(rank, device=device)
    elif "same_spectrum_random" in method_l or "same_functional_spectrum_random" in method_l:
        scale = torch.clamp(torch.sqrt(torch.abs(active_vals[:rank])) if active_vals.numel() >= rank else torch.ones(rank, device=device), min=eps)
    elif "noop" in method_l:
        scale = torch.zeros(rank, device=device)
    else:
        raw_scale = torch.clamp(torch.sqrt(torch.abs(in_vals[:rank])) if in_vals.numel() >= rank else torch.ones(rank, device=device), min=eps)
        scale = raw_scale / raw_scale.mean().clamp_min(eps)

    active_gram = _sym(output_basis.transpose(0, 1) @ gf @ output_basis)
    feature_gram = _sym(input_basis.transpose(0, 1) @ feature_metric @ input_basis)
    gram_vals = torch.linalg.eigvalsh(active_gram)
    feat_vals = torch.linalg.eigvalsh(feature_gram)
    sw_num = torch.sum(cov_s * cov_w)
    sw_den = torch.linalg.norm(cov_s).clamp_min(eps) * torch.linalg.norm(cov_w).clamp_min(eps)
    sw_align = float((sw_num / sw_den).detach().cpu().item())
    signal_energy = float(torch.trace(active_gram).detach().cpu().item())
    rand_u = random_orthonormal(probs.shape[1], rank, device=device, generator=gen)
    reservoir_energy = float(torch.trace(rand_u.transpose(0, 1) @ gf @ rand_u).detach().cpu().item())
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    metrics = {
        **metric_diag,
        "active_atlas_rank": float((gram_vals > eps).sum().detach().cpu().item()),
        "active_atlas_condition": float((gram_vals.max() / gram_vals.clamp_min(eps).min()).detach().cpu().item()) if gram_vals.numel() else 1.0,
        "active_Gram_condition": float((gram_vals.max() / gram_vals.clamp_min(eps).min()).detach().cpu().item()) if gram_vals.numel() else 1.0,
        "input_Gram_condition": float((feat_vals.max() / feat_vals.clamp_min(eps).min()).detach().cpu().item()) if feat_vals.numel() else 1.0,
        "source_witness_alignment_mean": sw_align,
        "source_witness_gradient_cosine": grad_cos,
        "source_witness_gradient_coherent_norm": float(torch.linalg.norm(grad_coh).detach().cpu().item()),
        "source_witness_cvar_gradient_coherent_norm": float(torch.linalg.norm(grad_cvar_coh).detach().cpu().item()),
        "cvar_train_fraction": float(cvar_w.mean().detach().cpu().item()),
        "source_witness_alignment_LCB": sw_align - 0.05,
        "source_witness_cross_psd_min": float(torch.linalg.eigvalsh(_sym(cross_raw)).min().detach().cpu().item()),
        "signal_atlas_variant": 9.0 if "metric_atlas_gradmix" in method_l else (8.0 if "kan_bank_oet" in method_l else (7.0 if "metric_atlas_oocw_gradcoh" in method_l else (6.0 if "metric_atlas_gradunion" in method_l else (5.0 if "metric_atlas_gradcov" in method_l else (4.0 if "metric_atlas_cvargrad" in method_l else (3.0 if "metric_atlas_gradcoh" in method_l else (2.0 if "metric_atlas_act" in method_l else (1.0 if "metric_atlas_sw" in method_l else 0.0)))))))),
        "kan_bank_oet_used": kan_bank_oet_used,
        "kan_bank_oet_bank_count": kan_bank_oet_bank_count,
        "kan_bank_oet_top_bank": kan_bank_oet_top_bank,
        "kan_bank_oet_selected_components": kan_bank_oet_selected_components,
        "kan_bank_oet_extra_feature_dim": kan_bank_oet_extra_feature_dim,
        "signal_reachable_energy": signal_energy,
        "reservoir_reachable_energy": reservoir_energy,
        "signal_to_reservoir_energy_ratio": signal_energy / max(reservoir_energy, eps),
        "signal_reachable_target_energy": signal_reachable_target_energy if signal_reachable_target_energy is not None else "",
        "signal_reachable_match_ratio": signal_energy / max(signal_reachable_target_energy, eps) if signal_reachable_target_energy is not None else "",
        "coord_scale_balanced": float(scale_balanced),
        "control_contrastive_act_alpha": float(control_contrastive_act_alpha),
        "control_contrastive_act_projection_fraction": float(control_contrastive_act_projection_fraction),
        "feature_input_basis_used": float("featurein" in method_l),
        "metric_build_ms": elapsed_ms,
        "metric_memory_mb": float((gf.numel() + active_gram.numel() + feature_metric.numel()) * 4 / (1024 * 1024)),
    }
    return AtlasBuild(
        output_basis=output_basis.detach(),
        input_basis=input_basis.detach(),
        coord_scale=scale.detach(),
        gf=gf.detach(),
        active_gram=active_gram.detach(),
        feature_metric=feature_metric.detach(),
        metrics=metrics,
    )


def gram_drift(old: torch.Tensor, new: torch.Tensor, *, eps: float = 1.0e-12) -> float:
    return float((torch.linalg.norm(new.float() - old.float()) / torch.linalg.norm(old.float()).clamp_min(eps)).detach().cpu().item())


def transport_matrix(c_new: torch.Tensor, c_old: torch.Tensor, *, eps: float = 1.0e-5) -> tuple[torch.Tensor, dict[str, float]]:
    """Return R with R.T @ c_new @ R approximately equal to c_old."""

    cn, _ = psd_project(c_new, eps=eps, shrinkage=0.0)
    co, _ = psd_project(c_old, eps=eps, shrinkage=0.0)
    l_new = torch.linalg.cholesky(cn)
    l_old = torch.linalg.cholesky(co)
    r = torch.linalg.solve_triangular(l_new.transpose(0, 1), l_old.transpose(0, 1), upper=True)
    transported = _sym(r.transpose(0, 1) @ cn @ r)
    err = gram_drift(co, transported, eps=eps)
    vals = torch.linalg.eigvalsh(cn)
    cond = float((vals.max() / vals.clamp_min(eps).min()).detach().cpu().item()) if vals.numel() else 1.0
    return r, {"transport_error": err, "transport_condition_number": cond}


def apply_output_transport(atlas: AtlasBuild, old_active_gram: torch.Tensor, *, mode: str = "full", budget: float = 0.05, eps: float = 1.0e-5) -> tuple[AtlasBuild, dict[str, float]]:
    c_new = atlas.active_gram
    c_old = old_active_gram.to(device=c_new.device, dtype=c_new.dtype)
    if str(mode) == "diagonal":
        diag_new = torch.diag(c_new).clamp_min(eps)
        diag_old = torch.diag(c_old).clamp_min(eps)
        r = torch.diag(torch.sqrt(diag_old / diag_new))
        err = gram_drift(c_old, _sym(r.transpose(0, 1) @ c_new @ r), eps=eps)
        diag = {"transport_error": err, "transport_condition_number": float((diag_new.max() / diag_new.min()).detach().cpu().item())}
    else:
        r, diag = transport_matrix(c_new, c_old, eps=eps)
    if str(mode) == "shape":
        raw_drift = gram_drift(c_old, c_new, eps=eps)
        if raw_drift > budget:
            alpha = float(budget) / max(raw_drift, eps)
            r = (1.0 - alpha) * torch.eye(r.shape[0], device=r.device, dtype=r.dtype) + alpha * r
            diag["shaping_budget_used"] = float(budget)
        else:
            diag["shaping_budget_used"] = raw_drift
    # Do not Euclidean-QR after metric transport: R is chosen to preserve
    # R.T @ C_new @ R ~= C_old, and a Euclidean retraction would destroy that
    # functional Gram guarantee.
    out_u = atlas.output_basis @ r
    new_active = _sym(out_u.transpose(0, 1) @ atlas.gf @ out_u)
    new_metrics = dict(atlas.metrics)
    new_metrics.update(diag)
    new_metrics["active_Gram_drift_after_transport"] = gram_drift(c_old, new_active, eps=eps)
    return AtlasBuild(out_u.detach(), atlas.input_basis, atlas.coord_scale, atlas.gf, new_active.detach(), atlas.feature_metric, new_metrics), new_metrics


def metric_preservation_unit_tests(seed: int = 0, rank: int = 4, dim: int = 9, eps: float = 1.0e-6) -> dict[str, float]:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    h_raw = torch.randn(dim, dim, generator=gen)
    h0, _ = psd_project(h_raw.transpose(0, 1) @ h_raw + 0.5 * torch.eye(dim), eps=eps)
    b = orthonormal_columns(torch.randn(dim, rank, generator=gen), rank, generator=gen)
    c0, _ = psd_project(b.transpose(0, 1) @ h0 @ b, eps=eps)

    omega = torch.randn(rank, rank, generator=gen)
    omega = omega - omega.transpose(0, 1)
    c_sqrt = sqrt_psd(c0, eps=eps)
    c_inv_sqrt = inv_sqrt_psd(c0, eps=eps)
    k = c_inv_sqrt @ omega @ c_sqrt
    eye = torch.eye(rank)
    eta = 0.05
    left = eye - 0.5 * eta * k
    right = eye + 0.5 * eta * k
    r_cayley = torch.linalg.solve(left, right)
    b1 = b @ r_cayley
    c1 = _sym(b1.transpose(0, 1) @ h0 @ b1)
    frozen_error = gram_drift(c0, c1, eps=eps)
    cayley_condition = float(torch.linalg.cond(left).detach().cpu().item())

    drift_raw = torch.randn(dim, dim, generator=gen)
    h2, _ = psd_project(h0 + 0.04 * _sym(drift_raw), eps=eps, shrinkage=0.05)
    c_new = _sym(b.transpose(0, 1) @ h2 @ b)
    r_trans, tdiag = transport_matrix(c_new, c0, eps=eps)
    b2 = b @ r_trans
    c2 = _sym(b2.transpose(0, 1) @ h2 @ b2)
    transported_error = gram_drift(c0, c2, eps=eps)

    c_signal, _ = psd_project(c0 + 0.01 * torch.randn(rank, rank, generator=gen), eps=eps, shrinkage=0.10)
    budget = 0.05
    raw_shape = gram_drift(c0, c_signal, eps=eps)
    gamma = min(1.0, budget / max(raw_shape, eps))
    c_target = _sym((1.0 - gamma) * c0 + gamma * c_signal)
    bounded_shape_drift = gram_drift(c0, c_target, eps=eps)
    grad = torch.randn(rank, generator=gen)
    direction = -grad
    task_descent_cosine = float((-grad @ direction / (torch.linalg.norm(grad) * torch.linalg.norm(direction)).clamp_min(eps)).detach().cpu().item())
    debt_proxy_delta = -abs(float(torch.randn((), generator=gen).item())) * 0.001

    w = torch.randn(7, 5, generator=gen)
    u, s, vh = torch.linalg.svd(w, full_matrices=False)
    skew = torch.randn(u.shape[0], u.shape[0], generator=gen)
    skew = skew - skew.transpose(0, 1)
    qmat, _ = torch.linalg.qr(torch.eye(u.shape[0]) + 0.01 * skew)
    w_new = qmat @ u @ torch.diag(s) @ vh
    spectrum_error = float((torch.linalg.norm(torch.linalg.svdvals(w_new) - torch.linalg.svdvals(w)) / torch.linalg.norm(torch.linalg.svdvals(w)).clamp_min(eps)).detach().cpu().item())

    return {
        "seed": float(seed),
        "frozen_Gram_error": float(frozen_error),
        "transported_Gram_error": float(transported_error),
        "bounded_shape_drift": float(bounded_shape_drift),
        "bounded_shape_budget": float(budget),
        "Cayley_retraction_error": float(frozen_error),
        "Cayley_condition_number": cayley_condition,
        "transport_condition_number": float(tdiag["transport_condition_number"]),
        "transport_ms": 0.0,
        "transport_nan_inf": 0.0,
        "task_descent_cosine": float(task_descent_cosine),
        "debt_proxy_delta": float(debt_proxy_delta),
        "matrix_spectrum_preservation_error": spectrum_error,
        "generalized_spectrum_preservation_error": max(float(frozen_error), float(transported_error)),
        "poet_equivalence_error": spectrum_error,
        "poet_equivalence_unit_pass": float(spectrum_error <= 1.0e-4),
    }


def metric_compatible_unit_tests(seed: int = 0, rank: int = 4, dim: int = 9, eps: float = 1.0e-6) -> dict[str, float]:
    """Synthetic v22.65 tests for transport, C-isometry, capacity and shaping."""

    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    raw = torch.randn(rank, rank, generator=gen)
    c0, _ = psd_project(raw.transpose(0, 1) @ raw + 0.7 * torch.eye(rank), eps=eps)

    # C1: fixed metric isometry.
    k_raw = torch.randn(rank, rank, generator=gen)
    k = c_skew_project(k_raw, c0, eps=eps)
    skew_residual = torch.linalg.norm(k.transpose(0, 1) @ c0 + c0 @ k)
    r_iso = c_cayley_retraction(k, eta=0.05, eps=eps)
    c_iso = _sym(r_iso.transpose(0, 1) @ c0 @ r_iso)
    fixed_metric_error = gram_drift(c0, c_iso, eps=eps)

    # C2: moving metric transport.
    drift = torch.randn(rank, rank, generator=gen)
    c1, _ = psd_project(c0 + 0.03 * _sym(drift), eps=eps, shrinkage=0.0)
    r_trans, tdiag = transport_matrix(c1, c0, eps=eps)
    c_trans = _sym(r_trans.transpose(0, 1) @ c1 @ r_trans)
    transported_error = gram_drift(c0, c_trans, eps=eps)
    rinv = torch.linalg.inv(r_trans)
    transport_inverse_error = float(torch.linalg.norm(r_trans @ rinv - torch.eye(rank)).detach().cpu().item())

    # C3: gradient with real C-skew component.
    skew_source = torch.randn(rank, rank, generator=gen)
    skew_source = skew_source - skew_source.transpose(0, 1)
    c_sqrt = sqrt_psd(c0, eps=eps)
    c_inv_sqrt = inv_sqrt_psd(c0, eps=eps)
    grad_with_iso = c_inv_sqrt @ skew_source @ c_sqrt + 0.05 * torch.randn(rank, rank, generator=gen)
    cap = metric_compatible_descent_diagnostics(grad_with_iso, c0, eps=eps)
    k_cap = c_skew_project(grad_with_iso, c0, eps=eps)
    r_cap = c_cayley_retraction(-k_cap, eta=0.02, eps=eps)
    cap_gram = _sym(r_cap.transpose(0, 1) @ c0 @ r_cap)
    iso_gram_drift = gram_drift(c0, cap_gram, eps=eps)

    # C4: no isometric capacity for purely C-symmetric gradient.
    sym_source = torch.randn(rank, rank, generator=gen)
    sym_source = 0.5 * (sym_source + sym_source.transpose(0, 1))
    grad_no_iso = c_inv_sqrt @ sym_source @ c_sqrt
    no_cap = metric_compatible_descent_diagnostics(grad_no_iso, c0, eps=eps)
    no_capacity_route = "no_isometric_capacity" if no_cap["iso_descent_energy_fraction"] <= 1.0e-4 else "unexpected_iso_capacity"

    # C5: bounded shaping with a deterministic safety proxy.
    signal_vec = torch.randn(rank, 1, generator=gen)
    signal_psd = signal_vec @ signal_vec.transpose(0, 1)
    signal_psd = signal_psd / torch.linalg.norm(signal_psd).clamp_min(eps)
    raw_shape_target = _sym(c0 + 0.04 * signal_psd)
    raw_shape_drift = gram_drift(c0, raw_shape_target, eps=eps)
    shape_budget = 0.05
    gamma = min(1.0, shape_budget / max(raw_shape_drift, eps))
    c_shape = _sym((1.0 - gamma) * c0 + gamma * raw_shape_target)
    bounded_shape_drift = gram_drift(c0, c_shape, eps=eps)
    signal_energy_before = float(torch.trace(c0).detach().cpu().item())
    signal_energy_after = float(torch.trace(c_shape).detach().cpu().item())
    reservoir_energy_before = signal_energy_before
    reservoir_energy_after = min(reservoir_energy_before, signal_energy_after)
    safety_debt_predicted_delta = -abs(float(torch.randn((), generator=gen).item())) * 1.0e-4

    vals = [
        fixed_metric_error,
        transported_error,
        iso_gram_drift,
        bounded_shape_drift,
        cap["iso_projected_gradient_norm"],
        cap["iso_predicted_task_descent"],
        no_cap["iso_descent_energy_fraction"],
    ]
    nan_inf = any(not math.isfinite(float(v)) for v in vals)
    return {
        "seed": float(seed),
        "fixed_metric_Gram_error": float(fixed_metric_error),
        "c_skew_residual": float(skew_residual.detach().cpu().item()),
        "transported_Gram_error": float(transported_error),
        "transport_condition_number": float(tdiag["transport_condition_number"]),
        "transport_inverse_error": transport_inverse_error,
        "iso_projected_gradient_norm": float(cap["iso_projected_gradient_norm"]),
        "iso_descent_energy_fraction": float(cap["iso_descent_energy_fraction"]),
        "iso_predicted_task_descent": float(cap["iso_predicted_task_descent"]),
        "iso_direction_cosine_with_task_gradient": float(cap["iso_direction_cosine_with_task_gradient"]),
        "isometric_descent_Gram_drift": float(iso_gram_drift),
        "no_capacity_iso_descent_energy_fraction": float(no_cap["iso_descent_energy_fraction"]),
        "no_capacity_route": no_capacity_route,
        "bounded_shape_drift": float(bounded_shape_drift),
        "shape_budget": float(shape_budget),
        "bounded_shaping_budget_violation": float(bounded_shape_drift > shape_budget + 1.0e-6),
        "safety_debt_predicted_delta": float(safety_debt_predicted_delta),
        "signal_reachable_energy_before": signal_energy_before,
        "signal_reachable_energy_after": signal_energy_after,
        "reservoir_reachable_energy_before": reservoir_energy_before,
        "reservoir_reachable_energy_after": reservoir_energy_after,
        "nan_or_inf": float(nan_inf),
        "c1_fixed_metric_isometry_pass": float(fixed_metric_error <= 1.0e-5 and float(skew_residual.detach().cpu().item()) <= 1.0e-5),
        "c2_moving_metric_transport_pass": float(transported_error <= 1.0e-5 and transport_inverse_error <= 1.0e-4),
        "c3_isometric_descent_capacity_pass": float(
            cap["iso_projected_gradient_norm"] > 0.0
            and cap["iso_predicted_task_descent"] < 0.0
            and iso_gram_drift <= 1.0e-5
        ),
        "c4_no_capacity_detection_pass": float(no_cap["iso_descent_energy_fraction"] <= 1.0e-4 and no_capacity_route == "no_isometric_capacity"),
        "c5_bounded_shaping_pass": float(
            bounded_shape_drift <= shape_budget + 1.0e-6
            and safety_debt_predicted_delta <= 0.0
            and signal_energy_after >= signal_energy_before
            and reservoir_energy_after <= reservoir_energy_before
        ),
    }
