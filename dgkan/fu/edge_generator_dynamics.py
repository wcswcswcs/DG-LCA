"""Optimizer-step edge generator gradient dynamics.

The operator keeps a train-only edge coordinate and applies a bounded gradient
transform to owned edge parameters. It does not update parameters directly;
the optimizer wrapper owns the call boundary and the wrapped optimizer applies
the actual parameter update.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import time
from typing import Iterable

import torch


@dataclass
class EdgeGeneratorDynamicsConfig:
    """Configuration for a bounded edge-gradient transform."""

    eta: float = 1.0e-3
    blend: float = 0.25
    max_norm_ratio: float = 1.25
    eps: float = 1.0e-12


@dataclass
class EdgeGeneratorDynamicsTrace:
    observe_calls: int = 0
    transform_calls: int = 0
    gradients_transformed: int = 0
    optimizer_owned_gradient_transform_pass: int = 0
    direct_parameter_update_attempted: int = 0
    inside_optimizer_step_violations: int = 0
    observe_time_ms: float = 0.0
    transform_time_ms: float = 0.0
    norm_ratio_values: list[float] = field(default_factory=list)
    cosine_values: list[float] = field(default_factory=list)

    def as_dict(self) -> dict[str, float | int]:
        return {
            "egd_observe_calls": self.observe_calls,
            "egd_transform_calls": self.transform_calls,
            "egd_gradients_transformed": self.gradients_transformed,
            "optimizer_owned_gradient_transform_pass": self.optimizer_owned_gradient_transform_pass,
            "direct_parameter_update_attempted": self.direct_parameter_update_attempted,
            "inside_optimizer_step_violations": self.inside_optimizer_step_violations,
            "egd_observe_time_ms": self.observe_time_ms,
            "egd_transform_time_ms": self.transform_time_ms,
            "norm_Pg_over_norm_g": _mean(self.norm_ratio_values, 1.0),
            "cos_g_Pg": _mean(self.cosine_values, 1.0),
        }


def _mean(values: Iterable[float], default: float = 0.0) -> float:
    clean = [float(v) for v in values if math.isfinite(float(v))]
    return float(sum(clean) / len(clean)) if clean else float(default)


def _safe_cos(a: torch.Tensor, b: torch.Tensor) -> float:
    aa = a.detach().reshape(-1).float()
    bb = b.detach().reshape(-1).float()
    denom = aa.norm().clamp_min(1.0e-12) * bb.norm().clamp_min(1.0e-12)
    out = float((aa.dot(bb) / denom).item())
    return out if math.isfinite(out) else 0.0


class EdgeGeneratorDynamics:
    """Bounded train-only edge gradient transform."""

    def __init__(
        self,
        edge_params: Iterable[torch.nn.Parameter],
        *,
        coordinate: torch.Tensor | None = None,
        config: EdgeGeneratorDynamicsConfig | None = None,
    ) -> None:
        self.edge_params = [p for p in edge_params if getattr(p, "requires_grad", False)]
        self.config = config or EdgeGeneratorDynamicsConfig()
        self.trace = EdgeGeneratorDynamicsTrace()
        self._inside_optimizer_step = False
        self._coordinate_by_param: dict[int, torch.Tensor] = {}
        if coordinate is not None:
            self.observe_coordinate(coordinate)

    def owns_param(self, param: torch.nn.Parameter) -> bool:
        return any(param is p for p in self.edge_params)

    def set_inside_optimizer_step(self, value: bool) -> None:
        self._inside_optimizer_step = bool(value)

    def observe_coordinate(self, coordinate: torch.Tensor | None = None) -> dict[str, float | int]:
        start = time.perf_counter()
        self.trace.observe_calls += 1
        for param in self.edge_params:
            if coordinate is None:
                coord = torch.zeros_like(param.detach())
            else:
                coord = coordinate.detach().to(device=param.device, dtype=param.dtype)
                if coord.numel() != param.numel():
                    flat = torch.zeros(param.numel(), device=param.device, dtype=param.dtype)
                    src = coord.reshape(-1)
                    flat[: min(flat.numel(), src.numel())] = src[: min(flat.numel(), src.numel())]
                    coord = flat.reshape_as(param)
                else:
                    coord = coord.reshape_as(param)
            self._coordinate_by_param[id(param)] = coord.clone()
        self.trace.observe_time_ms += (time.perf_counter() - start) * 1000.0
        return self.diagnostics()

    def transform(self, grad: torch.Tensor, param: torch.nn.Parameter, _group: dict | None = None) -> torch.Tensor:
        start = time.perf_counter()
        self.trace.transform_calls += 1
        if not self._inside_optimizer_step:
            self.trace.inside_optimizer_step_violations += 1
            return grad
        coord = self._coordinate_by_param.get(id(param))
        if coord is None:
            coord = torch.zeros_like(grad)
        coord = coord.to(device=grad.device, dtype=grad.dtype).reshape_as(grad)
        new_grad = grad + float(self.config.blend) * coord
        g_norm = grad.detach().norm().clamp_min(float(self.config.eps))
        ng_norm = new_grad.detach().norm().clamp_min(float(self.config.eps))
        max_norm = float(self.config.max_norm_ratio) * g_norm
        if float(ng_norm.item()) > float(max_norm.item()):
            new_grad = new_grad * (max_norm / ng_norm)
            ng_norm = new_grad.detach().norm().clamp_min(float(self.config.eps))
        self.trace.gradients_transformed += 1
        self.trace.optimizer_owned_gradient_transform_pass = 1
        self.trace.norm_ratio_values.append(float((ng_norm / g_norm).item()))
        self.trace.cosine_values.append(_safe_cos(grad, new_grad))
        self.trace.transform_time_ms += (time.perf_counter() - start) * 1000.0
        return new_grad

    def diagnostics(self) -> dict[str, float | int]:
        return self.trace.as_dict()


__all__ = [
    "EdgeGeneratorDynamics",
    "EdgeGeneratorDynamicsConfig",
    "EdgeGeneratorDynamicsTrace",
]
