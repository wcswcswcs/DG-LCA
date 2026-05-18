"""Manual full-edge PureKAN layers used by v9 validation.

These classes intentionally avoid ``torch.nn.Module`` and autograd for the
KAN path.  Parameters are plain tensors, and backward returns explicit
parameter gradients for the runner to feed into manual AdamW.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import torch


@dataclass
class ManualEdgeCache:
    x_edges: torch.Tensor
    values: torch.Tensor
    aux: Tuple[torch.Tensor, ...]


class ManualFullEdgeLayer:
    """Layer with ``y_j = sum_i phi_ij(x_i)`` for several edge bases."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        edge_kind: str = "compact_poly_silu3",
        device: torch.device | None = None,
    ) -> None:
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.edge_kind = str(edge_kind)
        if self.edge_kind == "poly2_silu6":
            param_count = 6
        elif self.edge_kind == "compact_poly_silu3":
            param_count = 3
        elif self.edge_kind in {"legendre3", "chebyshev3", "hermite3", "rational_pade2"}:
            param_count = 3
        elif self.edge_kind in {"rbf4", "spline4", "cubic_bspline4"}:
            param_count = 4
        elif self.edge_kind in {"rbf8", "cubic_bspline8"}:
            param_count = 8
        else:
            raise ValueError(f"unknown edge_kind={edge_kind!r}")
        theta = torch.zeros(self.in_features, self.out_features, param_count, device=device)
        self._init_theta(theta)
        theta.requires_grad_(False)
        self.theta = theta

    def _init_theta(self, theta: torch.Tensor) -> None:
        if self.edge_kind == "poly2_silu6":
            theta[..., 1] = torch.randn_like(theta[..., 1]) * 0.02
            theta[..., 2] = torch.randn_like(theta[..., 2]) * 0.02
            theta[..., 3] = 0.01
            theta[..., 4] = torch.randn_like(theta[..., 4]) * 0.002
            theta[..., 5] = torch.randn_like(theta[..., 5]) * 0.002
        elif self.edge_kind == "compact_poly_silu3":
            theta[..., 0] = torch.randn_like(theta[..., 0]) * 0.02
            theta[..., 1] = torch.randn_like(theta[..., 1]) * 0.02
            theta[..., 2] = torch.randn_like(theta[..., 2]) * 0.002
        elif self.edge_kind in {"legendre3", "chebyshev3", "hermite3", "rational_pade2"}:
            theta.normal_(0.0, 0.02)
        elif self.edge_kind in {"rbf4", "rbf8"}:
            theta.normal_(0.0, 0.02)
        elif self.edge_kind in {"spline4", "cubic_bspline4", "cubic_bspline8"}:
            theta.normal_(0.0, 0.02)

    @property
    def parameter_count_per_edge(self) -> int:
        return int(self.theta.shape[-1])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ManualEdgeCache]:
        x_edges = x.unsqueeze(2).expand(-1, self.in_features, self.out_features)
        if self.edge_kind == "poly2_silu6":
            a0, a1, a2, scale, b1, b2 = self.theta.unsqueeze(0).unbind(dim=-1)
            sig = torch.sigmoid(x_edges)
            silu = x_edges * sig
            inner = b1 * x_edges + b2 * x_edges.square()
            values = a0 + a1 * x_edges + a2 * silu + scale * inner
            return values.sum(dim=1), ManualEdgeCache(x_edges=x_edges, values=values, aux=(sig, silu, inner))
        if self.edge_kind == "compact_poly_silu3":
            w_lin, w_silu, w_quad = self.theta.unsqueeze(0).unbind(dim=-1)
            sig = torch.sigmoid(x_edges)
            silu = x_edges * sig
            x2 = x_edges.square()
            values = w_lin * x_edges + w_silu * silu + w_quad * x2
            return values.sum(dim=1), ManualEdgeCache(x_edges=x_edges, values=values, aux=(sig, silu, x2))
        if self.edge_kind in {"legendre3", "chebyshev3", "hermite3"}:
            mean = x_edges.detach().mean()
            inv_std = x_edges.detach().std(unbiased=False).clamp_min(1.0e-6).reciprocal()
            z = ((x_edges - mean) * inv_std).clamp(-3.0, 3.0) / 3.0
            active = (((x_edges - mean) * inv_std) >= -3.0) & (((x_edges - mean) * inv_std) <= 3.0)
            if self.edge_kind == "legendre3":
                basis = torch.stack([z, 0.5 * (3.0 * z.square() - 1.0), 0.5 * (5.0 * z.pow(3) - 3.0 * z)], dim=-1)
            elif self.edge_kind == "chebyshev3":
                basis = torch.stack([z, 2.0 * z.square() - 1.0, 4.0 * z.pow(3) - 3.0 * z], dim=-1)
            else:
                basis = torch.stack([z, z.square() - 1.0, z.pow(3) - 3.0 * z], dim=-1)
            values = (basis * self.theta.unsqueeze(0)).sum(dim=-1)
            return values.sum(dim=1), ManualEdgeCache(x_edges=x_edges, values=values, aux=(basis, z, inv_std / 3.0, active.to(x.dtype)))
        if self.edge_kind == "rational_pade2":
            denom = 1.0 + x_edges.abs()
            sig = torch.sigmoid(x_edges)
            silu = x_edges * sig
            basis = torch.stack([x_edges, x_edges.square() / denom, silu / denom], dim=-1)
            values = (basis * self.theta.unsqueeze(0)).sum(dim=-1)
            return values.sum(dim=1), ManualEdgeCache(x_edges=x_edges, values=values, aux=(basis, denom, sig, silu))
        if self.edge_kind in {"rbf4", "rbf8"}:
            count = 4 if self.edge_kind == "rbf4" else 8
            centers = torch.linspace(-1.5, 1.5, count, device=x.device, dtype=x.dtype)
            gamma = torch.tensor(0.75, device=x.device, dtype=x.dtype)
            basis = torch.exp(-gamma * (x_edges.unsqueeze(-1) - centers.view(1, 1, 1, -1)).square())
            values = (basis * self.theta.unsqueeze(0)).sum(dim=-1)
            return values.sum(dim=1), ManualEdgeCache(x_edges=x_edges, values=values, aux=(basis, centers, gamma))
        if self.edge_kind == "spline4":
            knots = torch.linspace(-1.5, 1.5, 4, device=x.device, dtype=x.dtype)
            width = torch.tensor(1.0, device=x.device, dtype=x.dtype)
            distance = x_edges.unsqueeze(-1) - knots.view(1, 1, 1, -1)
            basis = torch.clamp(1.0 - distance.abs() / width, min=0.0)
            values = (basis * self.theta.unsqueeze(0)).sum(dim=-1)
            return values.sum(dim=1), ManualEdgeCache(x_edges=x_edges, values=values, aux=(basis, distance, width))
        count = 4 if self.edge_kind == "cubic_bspline4" else 8
        knots = torch.linspace(-1.5, 1.5, count, device=x.device, dtype=x.dtype)
        t = x_edges.unsqueeze(-1) - knots.view(1, 1, 1, -1)
        a = t.abs()
        basis = torch.where(
            a < 1.0,
            (4.0 - 6.0 * a.square() + 3.0 * a.pow(3)) / 6.0,
            torch.where(a < 2.0, (2.0 - a).pow(3) / 6.0, torch.zeros_like(a)),
        )
        values = (basis * self.theta.unsqueeze(0)).sum(dim=-1)
        return values.sum(dim=1), ManualEdgeCache(x_edges=x_edges, values=values, aux=(basis, t))

    def backward(self, dy: torch.Tensor, cache: ManualEdgeCache) -> Tuple[torch.Tensor, torch.Tensor]:
        dy_edges = dy.unsqueeze(1).expand(-1, self.in_features, self.out_features)
        x_edges = cache.x_edges
        if self.edge_kind == "poly2_silu6":
            sig, silu, inner = cache.aux
            _a0, a1, a2, scale, b1, b2 = self.theta.unsqueeze(0).unbind(dim=-1)
            silu_prime = sig * (1.0 + x_edges * (1.0 - sig))
            dx_edges = dy_edges * (a1 + a2 * silu_prime + scale * (b1 + 2.0 * b2 * x_edges))
            grad = torch.stack(
                [
                    dy_edges,
                    dy_edges * x_edges,
                    dy_edges * silu,
                    dy_edges * inner,
                    dy_edges * scale * x_edges,
                    dy_edges * scale * x_edges.square(),
                ],
                dim=-1,
            ).sum(dim=0)
            return dx_edges.sum(dim=2), grad
        if self.edge_kind == "compact_poly_silu3":
            sig, silu, x2 = cache.aux
            w_lin, w_silu, w_quad = self.theta.unsqueeze(0).unbind(dim=-1)
            silu_prime = sig * (1.0 + x_edges * (1.0 - sig))
            dx_edges = dy_edges * (w_lin + w_silu * silu_prime + 2.0 * w_quad * x_edges)
            grad = torch.stack([dy_edges * x_edges, dy_edges * silu, dy_edges * x2], dim=-1).sum(dim=0)
            return dx_edges.sum(dim=2), grad
        if self.edge_kind in {"legendre3", "chebyshev3", "hermite3"}:
            basis, z, dzdx, active = cache.aux
            if self.edge_kind == "legendre3":
                dbasis_dz = torch.stack([torch.ones_like(z), 3.0 * z, 0.5 * (15.0 * z.square() - 3.0)], dim=-1)
            elif self.edge_kind == "chebyshev3":
                dbasis_dz = torch.stack([torch.ones_like(z), 4.0 * z, 12.0 * z.square() - 3.0], dim=-1)
            else:
                dbasis_dz = torch.stack([torch.ones_like(z), 2.0 * z, 3.0 * z.square() - 3.0], dim=-1)
            dbasis_dx = dbasis_dz * dzdx * active.unsqueeze(-1)
            theta = self.theta.unsqueeze(0)
            dx_edges = dy_edges * (dbasis_dx * theta).sum(dim=-1)
            grad = (dy_edges.unsqueeze(-1) * basis).sum(dim=0)
            return dx_edges.sum(dim=2), grad
        if self.edge_kind == "rational_pade2":
            basis, denom, sig, silu = cache.aux
            sign = torch.sign(x_edges)
            silu_prime = sig * (1.0 + x_edges * (1.0 - sig))
            d_second = (2.0 * x_edges * denom - x_edges.square() * sign) / denom.square()
            d_third = (silu_prime * denom - silu * sign) / denom.square()
            dbasis_dx = torch.stack([torch.ones_like(x_edges), d_second, d_third], dim=-1)
            dx_edges = dy_edges * (dbasis_dx * self.theta.unsqueeze(0)).sum(dim=-1)
            grad = (dy_edges.unsqueeze(-1) * basis).sum(dim=0)
            return dx_edges.sum(dim=2), grad
        if self.edge_kind in {"rbf4", "rbf8"}:
            basis, centers, gamma = cache.aux
            theta = self.theta.unsqueeze(0)
            dbasis_dx = basis * (-2.0 * gamma * (x_edges.unsqueeze(-1) - centers.view(1, 1, 1, -1)))
            dx_edges = dy_edges * (dbasis_dx * theta).sum(dim=-1)
            grad = (dy_edges.unsqueeze(-1) * basis).sum(dim=0)
            return dx_edges.sum(dim=2), grad
        if self.edge_kind in {"cubic_bspline4", "cubic_bspline8"}:
            basis, t = cache.aux
            a = t.abs()
            sign = torch.sign(t)
            dbasis_dx = torch.where(
                a < 1.0,
                (-12.0 * a + 9.0 * a.square()) * sign / 6.0,
                torch.where(a < 2.0, -0.5 * (2.0 - a).square() * sign, torch.zeros_like(a)),
            )
            dx_edges = dy_edges * (dbasis_dx * self.theta.unsqueeze(0)).sum(dim=-1)
            grad = (dy_edges.unsqueeze(-1) * basis).sum(dim=0)
            return dx_edges.sum(dim=2), grad
        basis, distance, width = cache.aux
        active = (distance.abs() < width).to(x_edges.dtype)
        dbasis_dx = torch.where(distance >= 0.0, -active / width, active / width)
        dx_edges = dy_edges * (dbasis_dx * self.theta.unsqueeze(0)).sum(dim=-1)
        grad = (dy_edges.unsqueeze(-1) * basis).sum(dim=0)
        return dx_edges.sum(dim=2), grad

    def parameter_count(self) -> int:
        return int(self.theta.numel())

    def flops_count(self) -> int:
        per_edge = {
            "compact_poly_silu3": 9,
            "poly2_silu6": 13,
            "legendre3": 12,
            "chebyshev3": 12,
            "hermite3": 12,
            "rbf4": 18,
            "rbf8": 34,
            "spline4": 14,
            "cubic_bspline4": 24,
            "cubic_bspline8": 44,
            "rational_pade2": 18,
        }[self.edge_kind]
        return int(self.in_features * self.out_features * per_edge)

    def curvature(self) -> torch.Tensor:
        if self.edge_kind == "poly2_silu6":
            return (2.0 * self.theta[..., 3] * self.theta[..., 5]).square().mean()
        if self.edge_kind == "compact_poly_silu3":
            return (2.0 * self.theta[..., 2]).square().mean()
        coeff = self.theta
        if coeff.shape[-1] < 3:
            return torch.zeros((), device=coeff.device, dtype=coeff.dtype)
        diff2 = coeff[..., 2:] - 2.0 * coeff[..., 1:-1] + coeff[..., :-2]
        return diff2.square().mean()

    def apply_functional_smoothing_(self, strength: float) -> None:
        alpha = float(strength)
        if alpha <= 0.0:
            return
        if self.edge_kind == "poly2_silu6":
            self.theta[..., 5].mul_(max(0.0, 1.0 - alpha))
            return
        if self.edge_kind == "compact_poly_silu3":
            self.theta[..., 2].mul_(max(0.0, 1.0 - alpha))
            return
        if self.edge_kind == "rational_pade2":
            self.theta[..., 1:].mul_(max(0.0, 1.0 - alpha))
            return
        coeff = self.theta
        if coeff.shape[-1] >= 3:
            old = coeff.clone()
            coeff[..., 1:-1].add_(old[..., :-2] - 2.0 * old[..., 1:-1] + old[..., 2:], alpha=alpha)


class ManualFullEdgeClassifier:
    def __init__(
        self,
        dims: Sequence[int],
        *,
        edge_kind: str = "compact_poly_silu3",
        device: torch.device | None = None,
    ) -> None:
        if len(dims) < 2:
            raise ValueError("dims must include input and output dimensions")
        self.dims = [int(x) for x in dims]
        self.edge_kind = str(edge_kind)
        self.layers = [
            ManualFullEdgeLayer(self.dims[i], self.dims[i + 1], edge_kind=edge_kind, device=device)
            for i in range(len(self.dims) - 1)
        ]

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[ManualEdgeCache]]:
        caches: List[ManualEdgeCache] = []
        h = x
        for layer in self.layers:
            h, cache = layer.forward(h)
            caches.append(cache)
        return h, caches

    def backward(self, dy: torch.Tensor, caches: Sequence[ManualEdgeCache]) -> List[torch.Tensor]:
        grads: List[torch.Tensor] = []
        dh = dy
        for layer, cache in zip(reversed(self.layers), reversed(caches)):
            dh, grad = layer.backward(dh, cache)
            grads.append(grad)
        grads.reverse()
        return grads

    def parameter_count(self) -> int:
        return sum(layer.parameter_count() for layer in self.layers)

    def flops_count(self) -> int:
        return sum(layer.flops_count() for layer in self.layers)

    def curvature(self) -> torch.Tensor:
        vals = [layer.curvature() for layer in self.layers]
        return torch.stack(vals).mean() if vals else torch.zeros(())

    def clone_parameters(self) -> List[torch.Tensor]:
        return [layer.theta.clone() for layer in self.layers]

    def restore_parameters_(self, params: Sequence[torch.Tensor]) -> None:
        for layer, value in zip(self.layers, params):
            layer.theta.copy_(value)

    def apply_functional_smoothing_(self, strength: float) -> None:
        for layer in self.layers:
            layer.apply_functional_smoothing_(strength)
