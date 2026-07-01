"""Fixed-point diagnostics for signal-induced metric maps."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import torch

from dgkan.fu.layer_composite_metric import EPS


@dataclass
class FixedPointAuditResult:
    converged: int
    iterations: int
    residual_final: float
    residual_max: float
    oscillation_ratio: float
    trace: list[float]


class ChannelFixedPointAudit:
    """Audits whether a deterministic metric map settles under fixed iteration."""

    def __init__(self, max_iter: int = 8, tol: float = 1.0e-4, damping: float = 0.5) -> None:
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.damping = float(damping)

    def run(self, initial: torch.Tensor, update_fn: Callable[[torch.Tensor], torch.Tensor]) -> FixedPointAuditResult:
        x = initial.detach().to(dtype=torch.float64).clone()
        trace: list[float] = []
        prev_res = 0.0
        rises = 0
        for _ in range(max(1, self.max_iter)):
            nxt = update_fn(x).detach().to(device=x.device, dtype=torch.float64)
            if tuple(nxt.shape) != tuple(x.shape):
                raise ValueError("fixed-point update returned incompatible shape")
            mixed = (1.0 - self.damping) * x + self.damping * nxt
            res = float(((mixed - x).norm() / x.norm().clamp_min(EPS)).detach().cpu().item())
            if math.isfinite(prev_res) and trace and res > prev_res * 1.05:
                rises += 1
            trace.append(res if math.isfinite(res) else float("inf"))
            x = mixed
            prev_res = res
            if res <= self.tol:
                break
        residual_final = trace[-1] if trace else 0.0
        residual_max = max(trace or [0.0])
        return FixedPointAuditResult(
            converged=int(residual_final <= self.tol),
            iterations=len(trace),
            residual_final=float(residual_final),
            residual_max=float(residual_max),
            oscillation_ratio=float(rises / max(1, len(trace) - 1)),
            trace=trace,
        )


def fixed_point_smoke_test() -> dict[str, float]:
    audit = ChannelFixedPointAudit(max_iter=12, tol=1.0e-5, damping=0.5)
    initial = torch.ones(4, dtype=torch.float64)

    def update(x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x + 0.5

    result = audit.run(initial, update)
    return {
        "fixed_point_smoke_converged": float(result.converged),
        "fixed_point_smoke_residual_final": result.residual_final,
        "fixed_point_smoke_iterations": float(result.iterations),
    }


__all__ = ["ChannelFixedPointAudit", "FixedPointAuditResult", "fixed_point_smoke_test"]
