"""Task/debt channel split utilities for v22.97Q.

The classes here build fixed geometric projectors from task and debt channel
sketches.  They are audit-time constructions and do not perform runtime winner
selection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from dgkan.fu.layer_composite_metric import EPS, sym
from dgkan.fu.quotient_signal_channel import GeneralizedEigenQuotient, OrthogonalizedChannelQuotient


def _as_psd(channel: torch.Tensor | None, dim: int | None = None) -> torch.Tensor:
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


def _dim(*channels: torch.Tensor | None) -> int:
    out = 0
    for channel in channels:
        if channel is None:
            continue
        c = channel.detach()
        if c.ndim == 2:
            out = max(out, int(c.shape[0]))
    return out


def _pad(channel: torch.Tensor | None, dim: int) -> torch.Tensor:
    c = _as_psd(channel, dim)
    if int(c.shape[0]) == int(dim):
        return c
    out = torch.zeros((int(dim), int(dim)), dtype=torch.float64)
    n = min(int(dim), int(c.shape[0]))
    out[:n, :n] = c[:n, :n]
    return out


def _trace_psd(channel: torch.Tensor) -> float:
    vals = torch.linalg.eigvalsh(sym(channel).to(dtype=torch.float64))
    return float(vals.clamp_min(0.0).sum().detach().cpu().item())


def _basis(channel: torch.Tensor, rank: int, eps: float = 1.0e-10) -> torch.Tensor:
    vals, vecs = torch.linalg.eigh(sym(channel).to(dtype=torch.float64))
    order = torch.argsort(vals, descending=True)
    vals = vals[order]
    vecs = vecs[:, order]
    r = min(int(rank), int((vals > float(eps)).sum().item()))
    return vecs[:, :r].contiguous() if r > 0 else torch.zeros((int(channel.shape[0]), 0), dtype=torch.float64)


def _projector(u: torch.Tensor, dim: int) -> torch.Tensor:
    if int(u.numel()) == 0:
        return torch.zeros((int(dim), int(dim)), dtype=torch.float64)
    q, _ = torch.linalg.qr(u, mode="reduced")
    return sym(q @ q.T)


def _overlap(a: torch.Tensor, b: torch.Tensor) -> float:
    aa = _as_psd(a)
    bb = _pad(b, int(aa.shape[0]))
    denom = max(_trace_psd(aa), EPS)
    return float((torch.trace(sym(aa) @ sym(bb)).abs() / denom).clamp(0.0, 1.0).detach().cpu().item())


@dataclass
class TaskDebtSplitResult:
    matrix: torch.Tensor
    basis: torch.Tensor
    summary: dict[str, Any] = field(default_factory=dict)

    @property
    def projector(self) -> torch.Tensor:
        return _projector(self.basis, int(self.matrix.shape[0]))


class DebtComponentChannelBuilder:
    """Build named debt channels from vectors, bases, or PSD component sketches."""

    def __init__(self, *, rank: int = 4) -> None:
        self.rank = int(rank)

    def build(self, components: dict[str, torch.Tensor]) -> tuple[torch.Tensor, list[dict[str, Any]]]:
        dim = _dim(*components.values())
        total = torch.zeros((dim, dim), dtype=torch.float64)
        rows: list[dict[str, Any]] = []
        for name, value in components.items():
            mat = _pad(value, dim)
            total = total + mat
            rows.append({"component": str(name), "trace": _trace_psd(mat), "rank": int(_basis(mat, self.rank).shape[1])})
        return sym(total), rows


class TaskDebtGeneralizedEigenSplit:
    """Find task-visible directions regularized by debt visibility."""

    def __init__(self, *, rank: int = 4, ridge: float = 1.0e-3) -> None:
        self.rank = int(rank)
        self.ridge = float(ridge)

    def build(self, task: torch.Tensor, debt: torch.Tensor, shape_metric: torch.Tensor | None = None) -> TaskDebtSplitResult:
        result = GeneralizedEigenQuotient(rank=self.rank, ridge=self.ridge).build(task, neg=debt, shape_metric=shape_metric)
        summary = dict(result.summary)
        summary.update({"split_method": "task_debt_generalized_eigen", "safe_rank": summary.get("geneig_safe_rank", 0)})
        return TaskDebtSplitResult(result.matrix, result.basis, summary)


class DebtOrthogonalizedTaskProjector:
    """Project the task channel away from a debt channel subspace."""

    def __init__(self, *, rank: int = 4, debt_rank: int = 4) -> None:
        self.rank = int(rank)
        self.debt_rank = int(debt_rank)

    def build(self, task: torch.Tensor, debt: torch.Tensor) -> TaskDebtSplitResult:
        result = OrthogonalizedChannelQuotient(rank=self.rank, bad_rank=self.debt_rank).build(task, debt=debt)
        summary = dict(result.summary)
        summary.update({"split_method": "debt_orthogonalized_task_projector", "debt_leakage": summary.get("debt_leakage_after_projection", 0.0)})
        return TaskDebtSplitResult(result.matrix, result.basis, summary)


class ComponentWiseSafeIntersection:
    """Intersect a task channel with orthogonal complements of debt components."""

    def __init__(self, *, rank: int = 4, debt_rank: int = 4) -> None:
        self.rank = int(rank)
        self.debt_rank = int(debt_rank)

    def build(self, task: torch.Tensor, components: dict[str, torch.Tensor]) -> TaskDebtSplitResult:
        dim = _dim(task, *components.values())
        task_m = _pad(task, dim)
        bad = torch.zeros((dim, dim), dtype=torch.float64)
        component_rows: list[dict[str, Any]] = []
        for name, channel in components.items():
            mat = _pad(channel, dim)
            bad = bad + mat
            component_rows.append({"component": str(name), "trace": _trace_psd(mat), "pre_overlap": _overlap(task_m, mat)})
        ubad = _basis(bad, self.debt_rank)
        pbad = _projector(ubad, dim)
        eye = torch.eye(dim, dtype=torch.float64)
        safe = sym((eye - pbad) @ task_m @ (eye - pbad))
        usafe = _basis(safe, self.rank)
        summary = {
            "split_method": "component_wise_safe_intersection",
            "component_count": len(components),
            "bad_subspace_rank": int(ubad.shape[1]),
            "safe_rank": int(usafe.shape[1]),
            "task_trace_before": _trace_psd(task_m),
            "task_trace_after": _trace_psd(safe),
            "retention_ratio": _trace_psd(safe) / max(_trace_psd(task_m), EPS),
            "component_rows": component_rows,
        }
        return TaskDebtSplitResult(safe, usafe, summary)


__all__ = [
    "ComponentWiseSafeIntersection",
    "DebtComponentChannelBuilder",
    "DebtOrthogonalizedTaskProjector",
    "TaskDebtGeneralizedEigenSplit",
    "TaskDebtSplitResult",
]
