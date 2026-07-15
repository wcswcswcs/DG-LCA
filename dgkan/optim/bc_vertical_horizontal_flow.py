"""Thin optimizer-style wrapper for v23.16 BC vertical-horizontal flow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from dgkan.fu.compositional_edge_tangent import build_tangent_cache
from dgkan.fu.compositional_edge_natural_flow import global_pcg_flow


@dataclass
class BCVerticalHorizontalFlowStep:
    accepted: int
    alpha: float
    diagnostics: dict[str, float | str]


class BCVerticalHorizontalFlow:
    """Apply a scalar-trusted BC-VH residual flow step to existing KAN edges."""

    def __init__(
        self,
        model: Any,
        edge_grams: list[torch.Tensor],
        *,
        lam: float = 1.0e-2,
        pcg_iterations: int = 4,
        alphas: tuple[float, ...] = (1.0, 0.5, 0.25, 0.125, 0.0625),
        guard_loss_tolerance: float = 2.0e-3,
    ) -> None:
        self.model = model
        self.edge_grams = edge_grams
        self.lam = float(lam)
        self.pcg_iterations = int(pcg_iterations)
        self.alphas = tuple(float(a) for a in alphas)
        self.guard_loss_tolerance = float(guard_loss_tolerance)

    @staticmethod
    def output_residual(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits.float(), dim=1).to(dtype=logits.dtype)
        target = F.one_hot(y.long(), num_classes=int(logits.shape[1])).to(dtype=logits.dtype)
        return target - probs

    def _snapshot(self) -> list[torch.Tensor]:
        return [p.detach().clone() for p in self.model.coeffs]

    def _restore(self, state: list[torch.Tensor]) -> None:
        with torch.no_grad():
            for param, value in zip(self.model.coeffs, state):
                param.copy_(value)

    def _apply_delta(self, delta: list[torch.Tensor], alpha: float) -> None:
        with torch.no_grad():
            for param, d in zip(self.model.coeffs, delta):
                param.add_(float(alpha) * d.to(device=param.device, dtype=param.dtype))

    def step(self, x_source: torch.Tensor, y_source: torch.Tensor, x_guard: torch.Tensor, y_guard: torch.Tensor) -> BCVerticalHorizontalFlowStep:
        cache = build_tangent_cache(self.model, x_source)
        residual = self.output_residual(cache.logits, y_source)
        flow = global_pcg_flow(
            self.model,
            cache,
            residual,
            self.edge_grams,
            lam=self.lam,
            iterations=self.pcg_iterations,
            preconditioner="vertical_block",
        )
        state = self._snapshot()
        with torch.no_grad():
            before_guard = float(F.cross_entropy(self.model(x_guard).float(), y_guard.long()).detach().cpu().item())
        for alpha in self.alphas:
            self._restore(state)
            self._apply_delta(flow.delta, float(alpha))
            with torch.no_grad():
                after_guard = float(F.cross_entropy(self.model(x_guard).float(), y_guard.long()).detach().cpu().item())
            if after_guard <= before_guard + self.guard_loss_tolerance:
                flow.diagnostics.update({"guard_loss_before": before_guard, "guard_loss_after": after_guard})
                return BCVerticalHorizontalFlowStep(1, float(alpha), flow.diagnostics)
        self._restore(state)
        flow.diagnostics.update({"guard_loss_before": before_guard, "guard_loss_after": before_guard})
        return BCVerticalHorizontalFlowStep(0, 0.0, flow.diagnostics)
