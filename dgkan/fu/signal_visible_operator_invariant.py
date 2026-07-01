"""Signal-visible operator invariants for quotient/fiber geometry audits."""

from __future__ import annotations

import math
from typing import Any

import torch

from dgkan.fu.layer_composite_metric import EPS, sym


def signal_visible_operator(j: torch.Tensor, p_sig: torch.Tensor, metric_diag: torch.Tensor) -> torch.Tensor:
    jj = j.detach().to(dtype=torch.float64).cpu()
    p = sym(p_sig.detach().to(dtype=torch.float64).cpu())
    g = metric_diag.detach().to(dtype=torch.float64).cpu().reshape(-1).clamp_min(EPS)
    n_out = min(int(jj.shape[0]), int(p.shape[0]))
    n_param = min(int(jj.shape[1]), int(g.numel()))
    if n_out <= 0 or n_param <= 0:
        return torch.zeros((0, 0), dtype=torch.float64)
    vals, vecs = torch.linalg.eigh(p[:n_out, :n_out])
    psqrt = (vecs * vals.clamp_min(0.0).sqrt().reshape(1, -1)) @ vecs.T
    return psqrt @ jj[:n_out, :n_param] / g[:n_param].sqrt().reshape(1, -1)


def signal_visible_spectrum(j: torch.Tensor, p_sig: torch.Tensor, metric_diag: torch.Tensor, top_k: int = 8) -> dict[str, float | int]:
    op = signal_visible_operator(j, p_sig, metric_diag)
    if int(op.numel()) == 0:
        return {"sv_trace": 0.0, "sv_top": 0.0, "sv_top_mass": 0.0, "sv_effective_rank": 0.0, "sv_rank": 0}
    vals = torch.linalg.svdvals(op).square().clamp_min(0.0)
    vals = torch.sort(vals, descending=True).values
    trace = float(vals.sum().item())
    if trace <= EPS:
        return {"sv_trace": 0.0, "sv_top": 0.0, "sv_top_mass": 0.0, "sv_effective_rank": 0.0, "sv_rank": 0}
    probs = vals / trace
    eff = float(torch.exp(-(probs * probs.clamp_min(EPS).log()).sum()).item())
    return {
        "sv_trace": trace,
        "sv_top": float(vals[0].item()),
        "sv_top_mass": float(vals[: int(top_k)].sum().item()) / max(trace, EPS),
        "sv_effective_rank": eff,
        "sv_rank": int((vals > 1.0e-10).sum().item()),
    }


def spectrum_cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    aa = a.detach().to(dtype=torch.float64).cpu().reshape(-1).clamp_min(0.0)
    bb = b.detach().to(dtype=torch.float64).cpu().reshape(-1).clamp_min(0.0)
    n = min(int(aa.numel()), int(bb.numel()))
    if n <= 0:
        return 0.0
    aa = aa[:n]
    bb = bb[:n]
    denom = aa.norm() * bb.norm()
    if float(denom.item()) <= EPS:
        return 0.0
    out = float((aa @ bb / denom).clamp(0.0, 1.0).item())
    return out if math.isfinite(out) else 0.0


def signal_visible_operator_smoke_test() -> dict[str, float | int]:
    j = torch.eye(3, dtype=torch.float64)
    p = torch.eye(3, dtype=torch.float64)
    g = torch.ones(3, dtype=torch.float64)
    spec = signal_visible_spectrum(j, p, g)
    return {
        "signal_visible_identity_trace_error": abs(float(spec["sv_trace"]) - 3.0),
        "signal_visible_identity_rank_error": abs(int(spec["sv_rank"]) - 3),
    }

