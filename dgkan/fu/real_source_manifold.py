"""Small real-history source manifold helpers for MLP FU experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class SourceManifold:
    basis: torch.Tensor
    mean: torch.Tensor
    dim: int
    condition_number: float
    positive_source_loss_fraction: float

    def to_row(self) -> dict[str, Any]:
        return {
            "source_manifold_dim": self.dim,
            "source_manifold_condition_number": self.condition_number,
            "history_positive_source_loss_fraction": self.positive_source_loss_fraction,
        }


def build_source_manifold(
    history: list[torch.Tensor],
    source_loss_gains: list[float] | None = None,
    *,
    dim: int = 8,
    positive_threshold: float = -1.0e-6,
    shrinkage: float = 1.0e-4,
) -> SourceManifold | None:
    if not history:
        return None
    picked: list[torch.Tensor] = []
    gains = source_loss_gains or [1.0 for _ in history]
    for source, gain in zip(history, gains):
        if float(gain) >= float(positive_threshold):
            picked.append(source.detach().float().reshape(-1).cpu())
    if len(picked) < 2:
        picked = [h.detach().float().reshape(-1).cpu() for h in history[-max(2, min(len(history), dim + 1)) :]]
    mat = torch.stack(picked, dim=0)
    mean = mat.mean(dim=0)
    centered = mat - mean
    try:
        _u, s, vh = torch.linalg.svd(centered, full_matrices=False)
    except RuntimeError:
        q, _r = torch.linalg.qr(centered.T, mode="reduced")
        basis = q[:, : min(dim, q.shape[1])].T.contiguous()
        return SourceManifold(basis=basis, mean=mean, dim=int(basis.shape[0]), condition_number=float("inf"), positive_source_loss_fraction=float(len(picked) / max(1, len(history))))
    k = max(1, min(int(dim), int(vh.shape[0])))
    basis = vh[:k].contiguous()
    if float(shrinkage) > 0.0:
        basis = basis / (1.0 + float(shrinkage))
    cond = float((s[0] / s[k - 1].clamp_min(1.0e-12)).item()) if s.numel() >= k else float("inf")
    return SourceManifold(basis=basis, mean=mean, dim=k, condition_number=cond, positive_source_loss_fraction=float(len(picked) / max(1, len(history))))


def project_to_manifold(vector: torch.Tensor, manifold: SourceManifold | None) -> tuple[torch.Tensor, dict[str, Any]]:
    v = vector.detach().float().reshape(-1).cpu()
    if manifold is None or manifold.basis.numel() == 0:
        return vector.detach().clone(), {
            "source_manifold_projection_residual_Gf": "",
            "ActuationR2": "",
            "source_manifold_update_norm": float(torch.linalg.vector_norm(vector.detach().float()).item()),
        }
    basis = manifold.basis.to(v.device)
    mean = manifold.mean.to(v.device)
    centered = v - mean
    coeff = basis @ centered
    projected = mean + basis.T @ coeff
    residual = torch.linalg.vector_norm(v - projected) / torch.linalg.vector_norm(v).clamp_min(1.0e-12)
    r2 = 1.0 - float(residual.item()) ** 2
    return projected.reshape_as(vector).to(vector.device, dtype=vector.dtype), {
        "source_manifold_projection_residual_Gf": float(residual.item()),
        "ActuationR2": float(r2),
        "source_manifold_update_norm": float(torch.linalg.vector_norm(projected).item()),
        **manifold.to_row(),
    }


__all__ = ["SourceManifold", "build_source_manifold", "project_to_manifold"]
