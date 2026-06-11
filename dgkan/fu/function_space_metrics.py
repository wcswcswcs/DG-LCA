"""Metric-first functional update helpers for v22.05.

These helpers are conservative train-stream approximations of the metrics in
the v22.05 plan.  They operate on the current batch only and expose both
function-space audit energies and parameter-vector projection proxies.  They
do not read validation/test/future data and they do not synthesize experiment
outcomes.
"""

from __future__ import annotations

from math import isfinite
from typing import Any

import torch
import torch.nn.functional as F


EPS = 1.0e-8


METRIC_NAMES = [
    "G0-L2",
    "G1-DiagFisher",
    "G2-PopRiskDiag",
    "G3-SobolevH1-hidden",
    "G4-RKHS-KNN",
    "G5-Fisher-RKHS",
    "G6-LowNDS",
    "G8-MetricEnsemble",
]


def _safe_norm(x: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(x.detach().float()).clamp_min(EPS)


def _unit_like(x: torch.Tensor) -> torch.Tensor:
    return x / _safe_norm(x).to(device=x.device, dtype=x.dtype)


def logits_for(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    return model(x).float()


def fisher_diag_from_logits(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits.float(), dim=-1)
    return (probs * (1.0 - probs)).clamp_min(EPS)


def l2_energy(delta: torch.Tensor) -> torch.Tensor:
    return delta.float().square().mean()


def fisher_energy(delta: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    weights = fisher_diag_from_logits(logits)
    return (weights * delta.float().square()).mean()


def poprisk_weights(values: torch.Tensor, dim: int = 0) -> torch.Tensor:
    values = values.float()
    mean = values.mean(dim=dim, keepdim=True)
    var = values.var(dim=dim, unbiased=False, keepdim=True)
    return (mean.square() / (var + EPS)).clamp_min(0.0)


def poprisk_energy(delta: torch.Tensor) -> torch.Tensor:
    weights = poprisk_weights(delta, dim=0)
    return (weights * delta.float().square()).mean()


def sobolev_fd_energy(values: torch.Tensor) -> torch.Tensor:
    flat = values.float().reshape(values.shape[0], -1) if values.ndim > 1 else values.float().reshape(1, -1)
    base = flat.square().mean()
    if flat.shape[0] < 2:
        return base
    fd = flat[1:] - flat[:-1]
    return base + fd.square().mean()


def rkhs_graph_energy(values: torch.Tensor, k: int = 4) -> torch.Tensor:
    flat = values.float().reshape(values.shape[0], -1) if values.ndim > 1 else values.float().reshape(1, -1)
    if flat.shape[0] < 3:
        return flat.square().mean()
    dist = torch.cdist(F.normalize(flat, dim=-1), F.normalize(flat, dim=-1), p=2)
    kk = min(max(1, int(k)), flat.shape[0] - 1)
    idx = dist.topk(kk + 1, largest=False).indices[:, 1:]
    src = flat.unsqueeze(1).expand(-1, kk, -1)
    nbr = flat[idx]
    return (src - nbr).square().mean()


def output_metric_energies(model: torch.nn.Module, x: torch.Tensor, update_proxy: torch.Tensor | None = None) -> dict[str, Any]:
    """Return audit energies for the current batch.

    ``update_proxy`` is optional because the truth gate and profiler unit tests
    also use this function without constructing a full Jacobian-vector product.
    When absent, centered logits are used as the measured function signal.
    """

    with torch.no_grad():
        logits = logits_for(model, x)
        delta = logits - logits.mean(dim=0, keepdim=True)
        if update_proxy is not None and update_proxy.numel():
            scale = float(_safe_norm(update_proxy).item())
            delta = delta / max(scale, EPS)
        out = {
            "metric_energy_L2": float(l2_energy(delta).item()),
            "metric_energy_Fisher": float(fisher_energy(delta, logits).item()),
            "metric_energy_PopRisk": float(poprisk_energy(delta).item()),
            "metric_energy_Sobolev": float(sobolev_fd_energy(delta).item()),
            "metric_energy_RKHS": float(rkhs_graph_energy(delta).item()),
        }
        vals = [float(v) for v in out.values() if isinstance(v, float) and isfinite(v)]
        out["metric_energy_mean"] = sum(vals) / len(vals) if vals else 0.0
        return out


def smooth_1d(vector: torch.Tensor, passes: int = 1) -> torch.Tensor:
    if vector.numel() < 3:
        return vector
    x = vector.reshape(1, 1, -1).float()
    kernel = torch.tensor([0.25, 0.5, 0.25], device=vector.device, dtype=torch.float32).reshape(1, 1, 3)
    for _ in range(max(1, int(passes))):
        x = F.pad(x, (1, 1), mode="replicate")
        x = F.conv1d(x, kernel)
    return x.reshape(-1).to(device=vector.device, dtype=vector.dtype)


def low_nds_filter(vector: torch.Tensor) -> torch.Tensor:
    if vector.numel() < 3:
        return vector
    smooth = smooth_1d(vector, passes=2)
    curvature = vector - smooth
    return vector - 0.75 * curvature


def metric_project_vector(gradient: torch.Tensor, metric_name: str) -> tuple[torch.Tensor, dict[str, Any]]:
    g = torch.nan_to_num(gradient.detach(), nan=0.0, posinf=0.0, neginf=0.0)
    if g.numel() == 0:
        return g, {"metric_name": metric_name, "metric_projection_norm": 0.0}
    rms = g.float().square().mean().sqrt().clamp_min(EPS)
    abs_floor = g.float().abs().median().clamp_min(EPS)
    if metric_name == "G0-L2":
        projected = g
    elif metric_name == "G1-DiagFisher":
        projected = g / (g.float().abs().to(device=g.device, dtype=g.dtype) + 0.25 * abs_floor.to(device=g.device, dtype=g.dtype))
    elif metric_name == "G2-PopRiskDiag":
        centered = g - g.mean()
        projected = centered * (centered.abs() >= centered.abs().median()).to(dtype=g.dtype)
    elif metric_name == "G3-SobolevH1-hidden":
        projected = smooth_1d(g, passes=2)
    elif metric_name == "G4-RKHS-KNN":
        projected = 0.65 * smooth_1d(g, passes=3) + 0.35 * g
    elif metric_name == "G5-Fisher-RKHS":
        fisher = g / (g.float().abs().to(device=g.device, dtype=g.dtype) + 0.25 * abs_floor.to(device=g.device, dtype=g.dtype))
        rkhs = smooth_1d(g, passes=4)
        high_freq_guard = smooth_1d(g - rkhs, passes=1)
        projected = 0.20 * fisher + 0.70 * rkhs - 0.10 * high_freq_guard
    elif metric_name == "G6-LowNDS":
        projected = low_nds_filter(g)
    elif metric_name == "G8-MetricEnsemble":
        fisher = metric_project_vector(g, "G1-DiagFisher")[0]
        rkhs = metric_project_vector(g, "G4-RKHS-KNN")[0]
        low_nds = metric_project_vector(g, "G6-LowNDS")[0]
        projected = (fisher + rkhs + low_nds) / 3.0
    else:
        raise ValueError(f"unknown metric_name {metric_name}")
    projected = torch.nan_to_num(projected, nan=0.0, posinf=0.0, neginf=0.0)
    scale = torch.linalg.vector_norm(g.float()).clamp_min(EPS) / torch.linalg.vector_norm(projected.float()).clamp_min(EPS)
    projected = projected * scale.to(device=projected.device, dtype=projected.dtype)
    smooth = smooth_1d(g, passes=2)
    curvature = torch.linalg.vector_norm((g - smooth).float()).square() / torch.linalg.vector_norm(g.float()).square().clamp_min(EPS)
    pcurv = torch.linalg.vector_norm((projected - smooth_1d(projected, passes=2)).float()).square() / torch.linalg.vector_norm(projected.float()).square().clamp_min(EPS)
    diagnostics = {
        "metric_name": metric_name,
        "metric_projection_norm": float(torch.linalg.vector_norm(projected.float()).item()),
        "metric_gradient_norm": float(torch.linalg.vector_norm(g.float()).item()),
        "first_order_gain": float(torch.dot(g.float(), projected.float()).item()),
        "second_order_penalty": float(pcurv.item()),
        "NDS": float(pcurv.item()),
        "raw_gradient_NDS": float(curvature.item()),
        "metric_projection_cosine": float(torch.dot(_unit_like(g).float(), _unit_like(projected).float()).clamp(-1, 1).item()),
    }
    return projected.to(device=gradient.device, dtype=gradient.dtype), diagnostics


def function_space_metric_unit_tests() -> list[dict[str, Any]]:
    g = torch.linspace(-1.0, 1.0, steps=17)
    rows: list[dict[str, Any]] = []
    for metric in METRIC_NAMES:
        projected, diag = metric_project_vector(g, metric)
        rows.append(
            {
                "case": metric,
                "projected_finite": int(torch.isfinite(projected).all().item()),
                "norm_positive": int(float(torch.linalg.vector_norm(projected).item()) > 0.0),
                "diagnostics_finite": int(all(isfinite(float(v)) for v in diag.values() if isinstance(v, (float, int)))),
                "pass": int(torch.isfinite(projected).all().item() and float(torch.linalg.vector_norm(projected).item()) > 0.0),
            }
        )
    return rows


__all__ = [
    "METRIC_NAMES",
    "fisher_diag_from_logits",
    "function_space_metric_unit_tests",
    "metric_project_vector",
    "output_metric_energies",
    "poprisk_energy",
    "rkhs_graph_energy",
    "sobolev_fd_energy",
]
