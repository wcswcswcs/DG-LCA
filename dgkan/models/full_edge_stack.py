"""Minimal full-edge stack container for smoke tests."""

from __future__ import annotations

from typing import List, Tuple

import torch

from .edge_layers import EdgeFunctionLayer, EdgeLayerCache


class FullEdgeStack:
    def __init__(self, dims: list[int], *, device: torch.device | None = None) -> None:
        if len(dims) < 2:
            raise ValueError("dims must include input and output")
        self.layers = [EdgeFunctionLayer(dims[i], dims[i + 1], device=device) for i in range(len(dims) - 1)]

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[EdgeLayerCache]]:
        caches: List[EdgeLayerCache] = []
        h = x
        for layer in self.layers:
            h, cache = layer.forward(h)
            caches.append(cache)
        return h, caches

    def parameter_count(self) -> int:
        return sum(layer.parameter_count() for layer in self.layers)
