"""Train-stream loss geometry contexts for v22.15 operators."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import torch


EPS = 1.0e-12


@dataclass
class GeometryContext:
    kind: str
    incidence: torch.Tensor | None = None
    mask: torch.Tensor | None = None
    weights: torch.Tensor | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_row(self) -> dict[str, Any]:
        return {
            "operator_context_type": self.kind,
            "incidence_rows": 0 if self.incidence is None else int(self.incidence.shape[0]),
            "incidence_cols": 0 if self.incidence is None else int(self.incidence.shape[1]),
            "mask_count": "" if self.mask is None else int(self.mask.detach().float().sum().item()),
            **self.metadata,
        }


def pointwise_context(size: int, mask: torch.Tensor | None = None, weights: torch.Tensor | None = None) -> GeometryContext:
    return GeometryContext("pointwise", None, mask, weights, {"context_train_stream_only": 1})


def pair_incidence_context(size: int, pairs: Iterable[tuple[int, int]], weights: torch.Tensor | None = None) -> GeometryContext:
    rows = []
    for i, j in pairs:
        row = torch.zeros(size, dtype=torch.float32)
        row[int(i)] = 1.0
        row[int(j)] = -1.0
        rows.append(row)
    incidence = torch.stack(rows, dim=0) if rows else torch.zeros(0, size)
    return GeometryContext("pairwise_incidence", incidence, None, weights, {"context_train_stream_only": 1})


def preference_graph_context(size: int, chosen_rejected: Iterable[tuple[int, int]], weights: torch.Tensor | None = None) -> GeometryContext:
    ctx = pair_incidence_context(size, chosen_rejected, weights)
    ctx.kind = "preference_graph"
    return ctx


def apply_context(vector: torch.Tensor, context: GeometryContext) -> torch.Tensor:
    v = vector.detach().float()
    if context.kind == "pointwise" or context.incidence is None:
        if context.mask is not None:
            m = context.mask.detach().float().reshape_as(v)
            v = v * m
        if context.weights is not None:
            w = context.weights.detach().float().reshape_as(v)
            v = v * w
        return v
    flat = v.reshape(-1)
    out = context.incidence.to(device=flat.device, dtype=flat.dtype) @ flat
    if context.weights is not None:
        out = out * context.weights.detach().float().to(out.device).reshape_as(out)
    return out


def pairwise_antisymmetry_error(vector: torch.Tensor, context: GeometryContext) -> float:
    if context.incidence is None or context.incidence.numel() == 0:
        return 0.0
    flat = vector.detach().float().reshape(-1)
    margins = context.incidence.to(flat.device, flat.dtype) @ flat
    reversed_margins = -margins
    denom = torch.linalg.vector_norm(margins).clamp_min(EPS)
    return float(torch.linalg.vector_norm(margins + reversed_margins).div(denom).item())


def row_collapse_score(vector: torch.Tensor) -> float:
    v = vector.detach().float()
    if v.ndim <= 1:
        return 0.0
    row_mean_energy = v.mean(dim=-1).square().mean()
    total = v.square().mean().clamp_min(EPS)
    return float((row_mean_energy / total).item())


def geometry_aware_operator(
    cotangent: torch.Tensor,
    context: GeometryContext,
    *,
    source_state: torch.Tensor | None = None,
    damping: float = 1.0e-2,
    pair_weight: float = 1.0,
    preserve_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, Any]]:
    c = cotangent.detach().float().reshape(-1)
    n = c.numel()
    eye = torch.eye(n, device=c.device, dtype=c.dtype)
    rhs = -c.clone()
    lhs = float(damping) * eye
    if context.kind == "pointwise" or context.incidence is None:
        if context.mask is not None:
            mask = context.mask.detach().float().reshape(-1).to(c.device)
            lhs = lhs + torch.diag(mask.clamp_min(0.0))
        else:
            lhs = lhs + eye
        if source_state is not None and float(preserve_weight) > 0.0:
            z = source_state.detach().float().reshape(-1).to(c.device)
            lhs = lhs + float(preserve_weight) * eye
            rhs = rhs + float(preserve_weight) * z
    else:
        b = context.incidence.detach().float().to(c.device)
        gram = b.T @ b
        lhs = lhs + eye + float(pair_weight) * gram
        if source_state is not None and float(preserve_weight) > 0.0:
            z = source_state.detach().float().reshape(-1).to(c.device)
            lhs = lhs + float(preserve_weight) * gram
            rhs = rhs + float(preserve_weight) * (gram @ z)
    status = "solve"
    try:
        out = torch.linalg.solve(lhs, rhs)
    except RuntimeError:
        out = torch.linalg.lstsq(lhs, rhs.unsqueeze(1)).solution.squeeze(1)
        status = "lstsq"
    shaped = out.reshape_as(cotangent)
    diag = {
        "operator_context_type": context.kind,
        "pairwise_antisymmetry_error": pairwise_antisymmetry_error(shaped, context),
        "row_collapse_score": row_collapse_score(shaped),
        "geometry_operator_status": status,
        "same_delta_different_C_output_norm": float(torch.linalg.vector_norm(shaped).item()),
    }
    return shaped, diag


__all__ = [
    "GeometryContext",
    "apply_context",
    "geometry_aware_operator",
    "pair_incidence_context",
    "pairwise_antisymmetry_error",
    "pointwise_context",
    "preference_graph_context",
    "row_collapse_score",
]
