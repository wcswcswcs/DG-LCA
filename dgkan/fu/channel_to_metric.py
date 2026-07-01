"""Reverse audits from output signal channels back to KAN metric coordinates."""

from __future__ import annotations

import math
from typing import Any

import torch

from dgkan.fu.layer_composite_metric import EPS, sym
from dgkan.fu.signal_channel_estimators import full_output_jacobian


def _as_projector(channel: torch.Tensor) -> torch.Tensor:
    c = channel.detach().to(dtype=torch.float64).cpu()
    if c.ndim != 2:
        raise ValueError("channel must be a matrix")
    if int(c.shape[0]) == int(c.shape[1]) and torch.allclose(c, c.T, atol=1.0e-8, rtol=1.0e-6):
        return sym(c)
    return c @ c.T


def _energy_fraction(projector: torch.Tensor, output_mat: torch.Tensor) -> float:
    p = _as_projector(projector)
    y = output_mat.detach().to(dtype=torch.float64).cpu()
    if int(y.numel()) == 0:
        return 0.0
    w = sym(y @ y.T)
    denom = torch.trace(w).abs().clamp_min(EPS)
    val = torch.trace(p[: int(w.shape[0]), : int(w.shape[1])] @ w).real / denom
    out = float(val.clamp(0.0, 1.0).detach().cpu().item())
    return out if math.isfinite(out) else 0.0


def _mask_vector(model: Any, predicate: Any) -> torch.Tensor:
    parts: list[torch.Tensor] = []
    for layer_idx, param in enumerate(model.coeffs):
        d_in, d_out, k = int(param.shape[0]), int(param.shape[1]), int(param.shape[2])
        mask = torch.zeros((d_in, d_out, k), dtype=torch.float64)
        for i in range(d_in):
            for j in range(d_out):
                for m in range(k):
                    if predicate(int(layer_idx), i, j, m, k):
                        mask[i, j, m] = 1.0
        parts.append(mask.reshape(-1))
    return torch.cat(parts).to(dtype=torch.float64) if parts else torch.empty(0, dtype=torch.float64)


def _mode_bands(basis_name: str, k: int) -> dict[str, range]:
    if str(basis_name) == "chebyshev":
        return {
            "degree_0_1": range(0, min(k, 2)),
            "degree_2_3": range(2, min(k, 4)),
            "degree_4_5": range(4, min(k, 6)),
            "degree_6_plus": range(6, k),
        }
    return {
        "constant": range(0, min(k, 1)),
        "low_harmonic": range(1, min(k, 3)),
        "mid_harmonic": range(3, min(k, 5)),
        "high_harmonic": range(5, k),
    }


def project_channel_to_mode_bands(
    p_sig: torch.Tensor,
    model: Any,
    source_batch: tuple[torch.Tensor, torch.Tensor],
    *,
    max_outputs: int = 96,
    output_mode: str = "margin",
) -> list[dict[str, float | int | str]]:
    x, y = source_batch
    j = full_output_jacobian(model, x, y, max_outputs=int(max_outputs), mode=output_mode)
    rows: list[dict[str, float | int | str]] = []
    for layer_idx, param in enumerate(model.coeffs):
        k = int(param.shape[2])
        for band, idxs in _mode_bands(str(model.basis_name), k).items():
            idx_set = set(int(i) for i in idxs)
            mask = _mask_vector(model, lambda li, _i, _j, m, _k, layer_idx=layer_idx, idx_set=idx_set: li == int(layer_idx) and m in idx_set)
            n = min(int(mask.numel()), int(j.shape[1]))
            if n <= 0:
                frac = 0.0
                trace = 0.0
            else:
                ymat = j[:, :n] * mask[:n].reshape(1, -1)
                frac = _energy_fraction(p_sig, ymat)
                trace = float(torch.trace(sym(ymat @ ymat.T)).abs().detach().cpu().item())
            rows.append({"layer_idx": int(layer_idx), "basis_name": str(model.basis_name), "band": band, "channel_energy_fraction": frac, "output_energy_trace": trace})
    return rows


def project_channel_to_layers(
    p_sig: torch.Tensor,
    model: Any,
    source_batch: tuple[torch.Tensor, torch.Tensor],
    *,
    max_outputs: int = 96,
    output_mode: str = "margin",
) -> list[dict[str, float | int | str]]:
    x, y = source_batch
    j = full_output_jacobian(model, x, y, max_outputs=int(max_outputs), mode=output_mode)
    rows: list[dict[str, float | int | str]] = []
    offset = 0
    for layer_idx, param in enumerate(model.coeffs):
        size = int(param.numel())
        n0, n1 = min(offset, int(j.shape[1])), min(offset + size, int(j.shape[1]))
        ymat = j[:, n0:n1]
        rows.append(
            {
                "layer_idx": int(layer_idx),
                "param_start": int(offset),
                "param_end": int(offset + size),
                "channel_energy_fraction": _energy_fraction(p_sig, ymat),
                "output_energy_trace": float(torch.trace(sym(ymat @ ymat.T)).abs().detach().cpu().item()) if int(ymat.numel()) else 0.0,
            }
        )
        offset += size
    return rows


def project_channel_to_normal_tangent(
    p_sig: torch.Tensor,
    adamw_delta: torch.Tensor,
    metric_projection: dict[str, torch.Tensor],
) -> dict[str, float]:
    j = metric_projection.get("J")
    tangent = metric_projection.get("tangent_delta")
    normal = metric_projection.get("normal_delta")
    delta = adamw_delta.detach().to(dtype=torch.float64).cpu().reshape(-1)
    p = _as_projector(p_sig)

    def frac(vec: torch.Tensor | None) -> float:
        if vec is None:
            return 0.0
        vv = vec.detach().to(dtype=torch.float64).cpu().reshape(-1)
        if j is not None:
            jj = j.detach().to(dtype=torch.float64).cpu()
            n = min(int(jj.shape[1]), int(vv.numel()))
            out = jj[:, :n] @ vv[:n]
        else:
            out = vv
        denom = out.square().sum().clamp_min(EPS)
        return float(((out @ p[: int(out.numel()), : int(out.numel())] @ out) / denom).clamp(0.0, 1.0).detach().cpu().item())

    return {
        "adamw_delta_channel_fraction": frac(delta),
        "tangent_channel_fraction": frac(tangent),
        "normal_channel_fraction": frac(normal),
    }


def reverse_audit_smoke_test() -> dict[str, float]:
    p = torch.eye(2, dtype=torch.float64)
    y = torch.eye(2, dtype=torch.float64)
    return {"reverse_audit_identity_energy": _energy_fraction(p, y)}


__all__ = [
    "project_channel_to_layers",
    "project_channel_to_mode_bands",
    "project_channel_to_normal_tangent",
    "reverse_audit_smoke_test",
]
