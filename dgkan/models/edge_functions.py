"""Edge functions used by clean full-edge candidates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch


@dataclass
class EdgeFunctionCache:
    x: torch.Tensor
    silu_x: torch.Tensor
    sig_x: torch.Tensor
    inner: torch.Tensor


class Poly2SiluEdgeFunction:
    """Poly2-SiLU edge basis.

    phi(x) = a0 + a1*x + a2*SiLU(x) + s*(b1*x + b2*x^2).
    Parameters are stored per edge as the final dimension of ``theta``:
    [a0, a1, a2, s, b1, b2].
    """

    parameter_count_per_edge = 6

    def forward(self, x: torch.Tensor, theta: torch.Tensor) -> Tuple[torch.Tensor, EdgeFunctionCache]:
        a0, a1, a2, scale, b1, b2 = theta.unbind(dim=-1)
        sig = torch.sigmoid(x)
        silu_x = x * sig
        inner = b1 * x + b2 * x.square()
        y = a0 + a1 * x + a2 * silu_x + scale * inner
        return y, EdgeFunctionCache(x=x, silu_x=silu_x, sig_x=sig, inner=inner)

    def backward(
        self, dy: torch.Tensor, theta: torch.Tensor, cache: EdgeFunctionCache
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        _a0, a1, a2, scale, b1, b2 = theta.unbind(dim=-1)
        x = cache.x
        silu_prime = cache.sig_x * (1.0 + x * (1.0 - cache.sig_x))
        dx = dy * (a1 + a2 * silu_prime + scale * (b1 + 2.0 * b2 * x))
        grad = torch.stack(
            [
                dy,
                dy * x,
                dy * cache.silu_x,
                dy * cache.inner,
                dy * scale * x,
                dy * scale * x.square(),
            ],
            dim=-1,
        )
        return dx, grad

    def geometry_metrics(self, theta: torch.Tensor) -> Dict[str, float]:
        b2 = theta[..., 5]
        scale = theta[..., 3]
        curvature = (2.0 * scale * b2).square().mean()
        return {"edge_poly2_curvature_mean": float(curvature.detach().cpu())}

    def functional_direction(self, theta: torch.Tensor) -> torch.Tensor:
        direction = torch.zeros_like(theta)
        direction[..., 5] = theta[..., 5]
        return direction

    def parameter_count(self, in_features: int, out_features: int) -> int:
        return in_features * out_features * self.parameter_count_per_edge

    def flops_count(self, in_features: int, out_features: int) -> int:
        return in_features * out_features * 12
