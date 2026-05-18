"""Full-edge KAN layers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch

from .edge_functions import EdgeFunctionCache, Poly2SiluEdgeFunction


@dataclass
class EdgeLayerCache:
    edge_cache: EdgeFunctionCache
    edge_values: torch.Tensor


class EdgeFunctionLayer:
    """Layer with y_j = sum_i phi_ij(x_i)."""

    def __init__(self, in_features: int, out_features: int, *, device: torch.device | None = None) -> None:
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.edge = Poly2SiluEdgeFunction()
        theta = torch.zeros(self.in_features, self.out_features, self.edge.parameter_count_per_edge, device=device)
        theta[..., 1] = torch.randn(self.in_features, self.out_features, device=device) * 0.02
        theta[..., 2] = torch.randn(self.in_features, self.out_features, device=device) * 0.02
        theta[..., 3] = 0.01
        theta.requires_grad_(False)
        self.theta = theta

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, EdgeLayerCache]:
        edge_input = x.unsqueeze(2).expand(-1, self.in_features, self.out_features)
        values, cache = self.edge.forward(edge_input, self.theta.unsqueeze(0))
        y = values.sum(dim=1)
        return y, EdgeLayerCache(edge_cache=cache, edge_values=values)

    def backward(self, dy: torch.Tensor, cache: EdgeLayerCache) -> Tuple[torch.Tensor, torch.Tensor]:
        dy_edges = dy.unsqueeze(1).expand(-1, self.in_features, self.out_features)
        dx_edges, grad_edges = self.edge.backward(dy_edges, self.theta.unsqueeze(0), cache.edge_cache)
        dx = dx_edges.sum(dim=2)
        grad_theta = grad_edges.sum(dim=0)
        return dx, grad_theta

    def parameter_count(self) -> int:
        return int(self.theta.numel())

    def flops_count(self) -> int:
        return self.edge.flops_count(self.in_features, self.out_features)

    def geometry_metrics(self) -> Dict[str, float]:
        return self.edge.geometry_metrics(self.theta)
