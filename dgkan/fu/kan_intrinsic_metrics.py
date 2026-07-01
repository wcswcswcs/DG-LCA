"""KAN-intrinsic metric utilities for v22.96 signal-channel audits."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable

import torch

from dgkan.fu.layer_composite_metric import EPS, condition_number, effective_rank, ridge_condition, sym


def _finite_float(value: torch.Tensor | float) -> float:
    out = float(value.detach().cpu().item()) if isinstance(value, torch.Tensor) else float(value)
    return out if math.isfinite(out) else 0.0


def _eig_summary(mat: torch.Tensor) -> dict[str, float]:
    m = sym(mat).to(dtype=torch.float64)
    vals = torch.linalg.eigvalsh(m)
    vals_abs = vals.abs()
    trace = torch.trace(m)
    diag = torch.diag(m)
    return {
        "trace": _finite_float(trace),
        "fro": _finite_float(m.norm()),
        "condition": condition_number(m),
        "effective_rank": effective_rank(m),
        "min_eig": _finite_float(vals.min()) if int(vals.numel()) else 0.0,
        "max_eig": _finite_float(vals.max()) if int(vals.numel()) else 0.0,
        "top1_abs": _finite_float(vals_abs.max()) if int(vals.numel()) else 0.0,
        "diag_min": _finite_float(diag.min()) if int(diag.numel()) else 0.0,
        "diag_max": _finite_float(diag.max()) if int(diag.numel()) else 0.0,
    }


def mode_metric_feature_diag(basis_name: str, input_dim: int, k: int, *, lambda_partial: float = 1.0e-4, lambda_partial2: float = 1.0e-6, lambda_omega: float = 1.0e-4, device: torch.device | None = None) -> torch.Tensor:
    if str(basis_name) == "chebyshev":
        vals = [1.0 + float(lambda_partial) * (m**2) + float(lambda_partial2) * (m**4) for m in range(int(k))]
    else:
        vals = [1.0]
        freq = 1
        while len(vals) < int(k):
            vals.append(1.0 + float(lambda_omega) * (freq**2))
            if len(vals) < int(k):
                vals.append(1.0 + float(lambda_omega) * (freq**2))
            freq += 1
    return torch.tensor(vals[: int(k)], device=device, dtype=torch.float64).repeat(int(input_dim))


def expand_feature_diag_to_param_vector(model: Any, layer_idx: int, feature_diag: torch.Tensor) -> torch.Tensor:
    param = model.coeffs[int(layer_idx)]
    d_in, d_out, k = int(param.shape[0]), int(param.shape[1]), int(param.shape[2])
    fd = feature_diag.to(device=param.device, dtype=torch.float64).reshape(d_in, k)
    out = torch.empty((d_in, d_out, k), device=param.device, dtype=torch.float64)
    for j in range(d_out):
        out[:, j, :] = fd
    return out.reshape(-1)


def flatten_layer_param_masks(model: Any) -> list[tuple[int, int, int]]:
    spans: list[tuple[int, int, int]] = []
    offset = 0
    for layer_idx, param in enumerate(model.coeffs):
        size = int(param.numel())
        spans.append((int(layer_idx), offset, offset + size))
        offset += size
    return spans


def full_mode_metric_vector(model: Any, *, lambda_partial: float = 1.0e-4, lambda_partial2: float = 1.0e-6, lambda_omega: float = 1.0e-4) -> torch.Tensor:
    parts: list[torch.Tensor] = []
    for layer_idx, param in enumerate(model.coeffs):
        feat = mode_metric_feature_diag(
            str(model.basis_name),
            int(param.shape[0]),
            int(param.shape[2]),
            lambda_partial=float(lambda_partial),
            lambda_partial2=float(lambda_partial2),
            lambda_omega=float(lambda_omega),
            device=param.device,
        )
        parts.append(expand_feature_diag_to_param_vector(model, layer_idx, feat))
    vec = torch.cat(parts).to(dtype=torch.float64)
    return vec / vec.mean().clamp_min(EPS)


def gram_matrix(phi: torch.Tensor, *, ridge: float = 0.0, target_condition: float = 1.0e6) -> tuple[torch.Tensor, dict[str, float]]:
    x = phi.detach().to(dtype=torch.float64)
    gram = sym(x.T @ x)
    ridge_info = {"ridge": 0.0, "condition_before_ridge": condition_number(gram + EPS * torch.eye(int(gram.shape[0]), device=gram.device, dtype=gram.dtype))}
    if ridge > 0.0:
        gram = gram + float(ridge) * torch.eye(int(gram.shape[0]), device=gram.device, dtype=gram.dtype)
    gram, info = ridge_condition(gram, target_condition=float(target_condition), max_ridge_rel=1.0e-2)
    ridge_info.update({"ridge": float(info.get("ridge_added", 0.0)) + float(ridge), "condition_after_ridge": float(info.get("C_condition_after_ridge", condition_number(gram)))})
    return sym(gram), ridge_info


@dataclass
class DataCompositeMetricResult:
    layer_idx: int
    phi_rows: int
    phi_cols: int
    metric_type: str
    matrix: torch.Tensor
    diag: torch.Tensor
    param_diag: torch.Tensor
    summary: dict[str, float | int | str]


class DataCompositeMetric:
    """Builds train-only data composite metrics from actual layer features."""

    def __init__(self, target_condition: float = 1.0e6, ridge: float = 0.0, sketch_rank: int = 16, sketch_seed: int = 0) -> None:
        self.target_condition = float(target_condition)
        self.ridge = float(ridge)
        self.sketch_rank = int(sketch_rank)
        self.sketch_seed = int(sketch_seed)

    def layer_phi(self, model: Any, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        with torch.no_grad():
            _logits, acts = model.forward_with_activations(x)
            return model.layer_phi(acts[int(layer_idx)].detach(), int(layer_idx)).to(dtype=torch.float64)

    def fit_layer(self, model: Any, x: torch.Tensor, layer_idx: int, metric_type: str = "block") -> DataCompositeMetricResult:
        phi = self.layer_phi(model, x, int(layer_idx))
        full, ridge_info = gram_matrix(phi, ridge=self.ridge, target_condition=self.target_condition)
        mtype = str(metric_type)
        if mtype == "diagonal":
            diag = torch.diag(full).clamp_min(EPS)
            mat = torch.diag(diag)
            sketch_rank = 0
        elif mtype == "full_sketch":
            rank = max(1, min(int(self.sketch_rank), int(full.shape[0])))
            gen = torch.Generator(device=phi.device).manual_seed(self.sketch_seed + 7919 * int(layer_idx))
            omega = torch.randn(int(phi.shape[1]), rank, generator=gen, device=phi.device, dtype=torch.float64) / math.sqrt(float(rank))
            y = full @ omega
            q, _ = torch.linalg.qr(y, mode="reduced")
            mat = sym(q @ (q.T @ full @ q) @ q.T)
            mat = mat + float(ridge_info.get("ridge", 0.0) + EPS) * torch.eye(int(mat.shape[0]), device=mat.device, dtype=mat.dtype)
            diag = torch.diag(mat).clamp_min(EPS)
            sketch_rank = rank
        else:
            mat = full
            diag = torch.diag(full).clamp_min(EPS)
            sketch_rank = int(full.shape[0])
        param_diag = expand_feature_diag_to_param_vector(model, int(layer_idx), diag)
        summ = _eig_summary(mat)
        summ.update(
            {
                "layer_idx": int(layer_idx),
                "phi_rows": int(phi.shape[0]),
                "phi_cols": int(phi.shape[1]),
                "metric_type": mtype,
                "ridge": float(ridge_info.get("ridge", 0.0)),
                "sketch_rank": int(sketch_rank),
            }
        )
        return DataCompositeMetricResult(int(layer_idx), int(phi.shape[0]), int(phi.shape[1]), mtype, mat, diag, param_diag, summ)

    def param_metric_vector(self, model: Any, x: torch.Tensor, metric_type: str = "diagonal") -> tuple[torch.Tensor, list[DataCompositeMetricResult]]:
        parts: list[torch.Tensor] = []
        results: list[DataCompositeMetricResult] = []
        for layer_idx, _param in enumerate(model.coeffs):
            res = self.fit_layer(model, x, int(layer_idx), metric_type=metric_type)
            results.append(res)
            parts.append(res.param_diag)
        vec = torch.cat(parts).to(dtype=torch.float64)
        vec = vec / vec.mean().clamp_min(EPS)
        return vec, results


def subspace_angle_from_diags(a: torch.Tensor, b: torch.Tensor) -> float:
    aa = a.detach().to(dtype=torch.float64).reshape(-1)
    bb = b.detach().to(device=aa.device, dtype=torch.float64).reshape(-1)
    n = min(int(aa.numel()), int(bb.numel()))
    aa = aa[:n] - aa[:n].mean()
    bb = bb[:n] - bb[:n].mean()
    denom = aa.norm() * bb.norm()
    if float(denom.detach().cpu().item()) <= 0.0:
        return 0.0
    cos = float((aa @ bb / denom).clamp(-1.0, 1.0).detach().cpu().item())
    return float(math.degrees(math.acos(cos)))


def data_metric_smoke_tests(device: torch.device | None = None) -> dict[str, float]:
    dev = device or torch.device("cpu")
    eye = torch.eye(4, device=dev, dtype=torch.float64)
    gram, _ = gram_matrix(eye, target_condition=1.0e6)
    identity_error = _finite_float((gram - eye).norm())
    dup = torch.tensor([[1.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.5], [1.0, 1.0, 0.0]], device=dev, dtype=torch.float64)
    raw = sym(dup.T @ dup / float(int(dup.shape[0])))
    vals = torch.linalg.eigvalsh(raw)
    ridged, info = gram_matrix(dup, target_condition=1.0e6)
    return {
        "toy_identity_error": identity_error,
        "toy_duplicate_min_eig_before_ridge": _finite_float(vals.min()),
        "toy_duplicate_rank_deficiency_detected": float(vals.min().abs().detach().cpu().item() < 1.0e-9),
        "toy_duplicate_condition_after_ridge": condition_number(ridged),
        "toy_duplicate_ridge": float(info.get("ridge", 0.0)),
    }


__all__ = [
    "DataCompositeMetric",
    "DataCompositeMetricResult",
    "data_metric_smoke_tests",
    "expand_feature_diag_to_param_vector",
    "flatten_layer_param_masks",
    "full_mode_metric_vector",
    "gram_matrix",
    "mode_metric_feature_diag",
    "subspace_angle_from_diags",
]
