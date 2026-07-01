"""Quotient signal-channel constructions for v22.97Q audits.

These utilities operate on small PSD sketches or projector matrices.  They do
not select runtime optimizer actions; they only build pre-registered geometric
quotient objects for experiment analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

import torch

from dgkan.fu.layer_composite_metric import EPS, sym


def _finite(value: torch.Tensor | float) -> float:
    out = float(value.detach().cpu().item()) if isinstance(value, torch.Tensor) else float(value)
    return out if math.isfinite(out) else 0.0


def _as_square(channel: torch.Tensor | None, dim: int | None = None) -> torch.Tensor:
    if channel is None:
        if dim is None:
            return torch.zeros((0, 0), dtype=torch.float64)
        return torch.zeros((int(dim), int(dim)), dtype=torch.float64)
    c = channel.detach().to(dtype=torch.float64).cpu()
    if c.ndim != 2:
        raise ValueError("channel must be a 2D tensor")
    if int(c.shape[0]) == int(c.shape[1]):
        return sym(c)
    return sym(c @ c.T)


def _common_dim(*channels: torch.Tensor | None) -> int:
    dim = 0
    for channel in channels:
        if channel is None:
            continue
        c = channel.detach()
        if c.ndim != 2:
            continue
        dim = max(dim, int(c.shape[0]))
    return dim


def _pad(mat: torch.Tensor, dim: int) -> torch.Tensor:
    m = _as_square(mat, dim)
    if int(m.shape[0]) == int(dim):
        return m
    out = torch.zeros((int(dim), int(dim)), dtype=torch.float64)
    n = min(int(dim), int(m.shape[0]))
    out[:n, :n] = m[:n, :n]
    return out


def _trace_psd(mat: torch.Tensor) -> float:
    vals = torch.linalg.eigvalsh(sym(mat).to(dtype=torch.float64))
    return _finite(vals.clamp_min(0.0).sum())


def _psd_clip(mat: torch.Tensor, eps: float = 0.0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    vals, vecs = torch.linalg.eigh(sym(mat).to(dtype=torch.float64))
    clipped = vals.clamp_min(float(eps))
    out = sym((vecs * clipped.reshape(1, -1)) @ vecs.T)
    order = torch.argsort(clipped, descending=True)
    return out, clipped[order], vecs[:, order]


def _basis_from_psd(mat: torch.Tensor, rank: int, eps: float = 1.0e-10) -> tuple[torch.Tensor, torch.Tensor]:
    _q, vals, vecs = _psd_clip(mat, 0.0)
    keep = vals > float(eps)
    r = min(max(0, int(rank)), int(keep.sum().item()))
    if r <= 0:
        return torch.zeros((int(mat.shape[0]), 0), dtype=torch.float64), vals
    return vecs[:, :r].contiguous(), vals


def _overlap_fraction(a: torch.Tensor, b: torch.Tensor) -> float:
    aa = _as_square(a)
    bb = _as_square(b, int(aa.shape[0]))
    denom = _trace_psd(aa)
    if denom <= EPS:
        return 0.0
    val = torch.trace(sym(aa) @ sym(bb)).abs() / max(denom, EPS)
    return max(0.0, min(1.0, _finite(val)))


def _projector_from_basis(u: torch.Tensor, dim: int) -> torch.Tensor:
    if int(u.numel()) == 0:
        return torch.zeros((int(dim), int(dim)), dtype=torch.float64)
    q, _ = torch.linalg.qr(u.to(dtype=torch.float64), mode="reduced")
    return sym(q @ q.T)


@dataclass
class QuotientResult:
    matrix: torch.Tensor
    basis: torch.Tensor
    eigenvalues: torch.Tensor
    summary: dict[str, Any] = field(default_factory=dict)

    @property
    def projector(self) -> torch.Tensor:
        return _projector_from_basis(self.basis, int(self.matrix.shape[0]))


class PSDClippedDifferenceQuotient:
    """Build PSD_+(A_pos - alpha A_neg - beta A_mlp - gamma A_debt)."""

    def __init__(
        self,
        *,
        rank: int = 4,
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 1.0,
        trace_normalize: bool = True,
        eps: float = 1.0e-10,
    ) -> None:
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.trace_normalize = bool(trace_normalize)
        self.eps = float(eps)

    def _scaled_bad(self, pos: torch.Tensor, bad: torch.Tensor, weight: float) -> tuple[torch.Tensor, float]:
        if int(bad.numel()) == 0:
            return bad, 0.0
        scale = 1.0
        if self.trace_normalize:
            scale = _trace_psd(pos) / max(_trace_psd(bad), self.eps)
        return float(weight) * float(scale) * bad, float(weight) * float(scale)

    def build(
        self,
        pos: torch.Tensor,
        neg: torch.Tensor | None = None,
        mlp: torch.Tensor | None = None,
        debt: torch.Tensor | None = None,
    ) -> QuotientResult:
        dim = _common_dim(pos, neg, mlp, debt)
        a_pos = _pad(pos, dim)
        a_neg = _pad(_as_square(neg, dim), dim)
        a_mlp = _pad(_as_square(mlp, dim), dim)
        a_debt = _pad(_as_square(debt, dim), dim)
        neg_term, alpha_eff = self._scaled_bad(a_pos, a_neg, self.alpha)
        mlp_term, beta_eff = self._scaled_bad(a_pos, a_mlp, self.beta)
        debt_term, gamma_eff = self._scaled_bad(a_pos, a_debt, self.gamma)
        raw = sym(a_pos - neg_term - mlp_term - debt_term)
        q, vals, _vecs = _psd_clip(raw, 0.0)
        basis, eigvals = _basis_from_psd(q, self.rank, self.eps)
        projector = _projector_from_basis(basis, dim)
        summary = {
            "quotient_method": "psd_clipped_difference",
            "quotient_positive_trace": _trace_psd(q),
            "quotient_raw_min_eig": _finite(torch.linalg.eigvalsh(raw).min()) if dim else 0.0,
            "quotient_rank": int(basis.shape[1]),
            "quotient_top_mass": _finite(vals[: max(1, int(basis.shape[1]))].sum() / vals.clamp_min(0.0).sum().clamp_min(EPS)) if int(vals.numel()) else 0.0,
            "quotient_negative_leakage": _overlap_fraction(projector, a_neg),
            "quotient_mlp_leakage": _overlap_fraction(projector, a_mlp),
            "quotient_debt_leakage": _overlap_fraction(projector, a_debt),
            "alpha_effective": alpha_eff,
            "beta_effective": beta_eff,
            "gamma_effective": gamma_eff,
            "trace_normalize": int(self.trace_normalize),
        }
        return QuotientResult(q, basis, eigvals, summary)


class GeneralizedEigenQuotient:
    """Solve a sketched generalized eigen quotient in the provided channel space."""

    def __init__(self, *, rank: int = 4, ridge: float = 1.0e-3, eps: float = 1.0e-10) -> None:
        self.rank = int(rank)
        self.ridge = float(ridge)
        self.eps = float(eps)

    def build(
        self,
        pos: torch.Tensor,
        neg: torch.Tensor | None = None,
        mlp: torch.Tensor | None = None,
        debt: torch.Tensor | None = None,
        shape_metric: torch.Tensor | None = None,
    ) -> QuotientResult:
        dim = _common_dim(pos, neg, mlp, debt, shape_metric)
        a_pos = _pad(pos, dim)
        bad = _pad(_as_square(neg, dim), dim) + _pad(_as_square(mlp, dim), dim) + _pad(_as_square(debt, dim), dim)
        g = _pad(_as_square(shape_metric, dim), dim) if shape_metric is not None else torch.eye(dim, dtype=torch.float64)
        b = sym(bad + float(self.ridge) * g + self.eps * torch.eye(dim, dtype=torch.float64))
        b_vals, b_vecs = torch.linalg.eigh(b)
        inv_sqrt = (b_vecs * b_vals.clamp_min(self.eps).rsqrt().reshape(1, -1)) @ b_vecs.T
        whitened = sym(inv_sqrt @ a_pos @ inv_sqrt)
        vals, vecs = torch.linalg.eigh(whitened)
        order = torch.argsort(vals, descending=True)
        vals = vals[order]
        vecs = vecs[:, order]
        raw_basis = inv_sqrt @ vecs[:, : max(1, min(self.rank, int(vecs.shape[1])))]
        basis, _ = torch.linalg.qr(raw_basis, mode="reduced")
        positive = vals.clamp_min(0.0)
        r = min(self.rank, int((positive > self.eps).sum().item()))
        basis = basis[:, :r].contiguous() if r > 0 else torch.zeros((dim, 0), dtype=torch.float64)
        q = _projector_from_basis(basis, dim)
        gap = _finite(vals[0] / vals[1].abs().clamp_min(EPS)) if int(vals.numel()) >= 2 else 0.0
        summary = {
            "quotient_method": "generalized_eigen",
            "geneig_top_values": [float(v) for v in vals[: min(8, int(vals.numel()))].detach().cpu().tolist()],
            "geneig_gap": gap,
            "geneig_safe_rank": int(r),
            "geneig_negative_overlap": _overlap_fraction(q, _pad(_as_square(neg, dim), dim)),
            "geneig_debt_overlap": _overlap_fraction(q, _pad(_as_square(debt, dim), dim)),
            "geneig_mlp_overlap": _overlap_fraction(q, _pad(_as_square(mlp, dim), dim)),
            "ridge": self.ridge,
        }
        return QuotientResult(q, basis, vals, summary)


class OrthogonalizedChannelQuotient:
    """Project positive channel away from the joint negative/MLP/debt subspace."""

    def __init__(self, *, rank: int = 4, bad_rank: int = 4, eps: float = 1.0e-10) -> None:
        self.rank = int(rank)
        self.bad_rank = int(bad_rank)
        self.eps = float(eps)

    def build(
        self,
        pos: torch.Tensor,
        neg: torch.Tensor | None = None,
        mlp: torch.Tensor | None = None,
        debt: torch.Tensor | None = None,
    ) -> QuotientResult:
        dim = _common_dim(pos, neg, mlp, debt)
        a_pos = _pad(pos, dim)
        a_neg = _pad(_as_square(neg, dim), dim)
        a_mlp = _pad(_as_square(mlp, dim), dim)
        a_debt = _pad(_as_square(debt, dim), dim)
        bad = sym(a_neg + a_mlp + a_debt)
        ubad, bad_vals = _basis_from_psd(bad, self.bad_rank, self.eps)
        p_bad = _projector_from_basis(ubad, dim)
        eye = torch.eye(dim, dtype=torch.float64)
        residual = sym((eye - p_bad) @ a_pos @ (eye - p_bad))
        q, vals, _vecs = _psd_clip(residual, 0.0)
        basis, eigvals = _basis_from_psd(q, self.rank, self.eps)
        projector = _projector_from_basis(basis, dim)
        before = _trace_psd(a_pos)
        after = _trace_psd(q)
        summary = {
            "quotient_method": "orthogonalized_channel",
            "bad_subspace_rank": int(ubad.shape[1]),
            "positive_energy_before_projection": before,
            "positive_energy_after_projection": after,
            "retention_ratio": after / max(before, EPS),
            "negative_leakage_before": _overlap_fraction(a_pos, a_neg),
            "mlp_leakage_before": _overlap_fraction(a_pos, a_mlp),
            "debt_leakage_before": _overlap_fraction(a_pos, a_debt),
            "negative_leakage_after_projection": _overlap_fraction(projector, a_neg),
            "mlp_leakage_after_projection": _overlap_fraction(projector, a_mlp),
            "debt_leakage_after_projection": _overlap_fraction(projector, a_debt),
            "bad_top_values": [float(v) for v in bad_vals[: min(8, int(bad_vals.numel()))].detach().cpu().tolist()],
        }
        return QuotientResult(q, basis, eigvals, summary)


class NegativeLeakageDecomposition:
    """Compute before/after leakage summaries for named bad-channel components."""

    def decompose(self, positive: torch.Tensor, components: dict[str, torch.Tensor]) -> list[dict[str, Any]]:
        pos = _as_square(positive)
        rows: list[dict[str, Any]] = []
        for name, channel in components.items():
            bad = _pad(_as_square(channel, int(pos.shape[0])), int(pos.shape[0]))
            diff = PSDClippedDifferenceQuotient(rank=max(1, min(4, int(pos.shape[0])))).build(pos, bad)
            rows.append(
                {
                    "component": str(name),
                    "negative_top_mass_before": _overlap_fraction(pos, bad),
                    "negative_top_mass_after": diff.summary.get("quotient_negative_leakage", 0.0),
                    "component_trace": _trace_psd(bad),
                    "quotient_trace_after_component": diff.summary.get("quotient_positive_trace", 0.0),
                }
            )
        return rows


def quotient_toy_smoke_test() -> dict[str, float]:
    pos = torch.diag(torch.tensor([1.0, 0.0], dtype=torch.float64))
    orth_neg = torch.diag(torch.tensor([0.0, 1.0], dtype=torch.float64))
    same_neg = pos.clone()
    builder = PSDClippedDifferenceQuotient(rank=1, trace_normalize=False)
    retained = builder.build(pos, orth_neg).matrix
    suppressed = builder.build(pos, same_neg).matrix
    return {
        "quotient_positive_retention": _trace_psd(retained) / max(_trace_psd(pos), EPS),
        "quotient_identical_channel_suppression": _trace_psd(suppressed) / max(_trace_psd(pos), EPS),
    }


__all__ = [
    "GeneralizedEigenQuotient",
    "NegativeLeakageDecomposition",
    "OrthogonalizedChannelQuotient",
    "PSDClippedDifferenceQuotient",
    "QuotientResult",
    "quotient_toy_smoke_test",
]
