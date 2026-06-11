"""Control-span projection helpers for v22.13 role-blind operators."""

from __future__ import annotations

from typing import Iterable

import torch


EPS = 1.0e-8


def stable_control_like(x: torch.Tensor) -> torch.Tensor:
    if x.ndim > 1:
        pattern = torch.sin(torch.arange(x.shape[-1], device=x.device, dtype=x.dtype)).reshape(1, -1)
        base = pattern.expand_as(x)
    else:
        base = torch.sin(torch.arange(x.numel(), device=x.device, dtype=x.dtype)).reshape_as(x)
    return match_norm(base, x)


def random_control_like(x: torch.Tensor, seed: int = 2213) -> torch.Tensor:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    if x.ndim > 1:
        raw_cpu = torch.randn((1, int(x.shape[-1])), generator=gen, dtype=x.detach().cpu().dtype)
        raw = raw_cpu.to(device=x.device, dtype=x.dtype).expand_as(x)
    else:
        raw = torch.randn(x.shape, generator=gen, dtype=x.detach().cpu().dtype).to(device=x.device, dtype=x.dtype)
    return match_norm(raw, x)


def mean_axis_control(x: torch.Tensor) -> torch.Tensor:
    return match_norm(x.mean(dim=0, keepdim=True).expand_as(x), x)


def row_axis_control(x: torch.Tensor) -> torch.Tensor:
    return match_norm(x.mean(dim=-1, keepdim=True).expand_as(x), x)


def match_norm(src: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    a = src.detach().float()
    b = ref.detach().float()
    return a * (torch.linalg.vector_norm(b).clamp_min(EPS) / torch.linalg.vector_norm(a).clamp_min(EPS))


def default_control_span(x: torch.Tensor, seed: int = 2213) -> list[torch.Tensor]:
    # Row-axis cotangents can be genuine pairwise/preference signals: removing
    # them as "control" erases ranking adapters.  Keep row_axis_control
    # available for diagnostics, but do not include it in the default nullspace.
    return [mean_axis_control(x), random_control_like(x, seed), stable_control_like(x)]


def projection_fraction(x: torch.Tensor, controls: Iterable[torch.Tensor]) -> tuple[float, float, torch.Tensor]:
    vec = x.detach().float().reshape(-1)
    basis = []
    for ctrl in controls:
        c = ctrl.detach().float().reshape(-1)
        norm = torch.linalg.vector_norm(c)
        if float(norm.item()) > EPS:
            basis.append(c / norm)
    if not basis:
        return 0.0, float(torch.linalg.vector_norm(vec).item()), torch.zeros_like(x.detach().float())
    q = []
    for b in basis:
        v = b.clone()
        for prev in q:
            v = v - (v @ prev) * prev
        n = torch.linalg.vector_norm(v)
        if float(n.item()) > EPS:
            q.append(v / n)
    if not q:
        return 0.0, float(torch.linalg.vector_norm(vec).item()), torch.zeros_like(x.detach().float())
    mat = torch.stack(q, dim=1)
    proj = mat @ (mat.T @ vec)
    denom = torch.linalg.vector_norm(vec).clamp_min(EPS)
    frac = float((torch.linalg.vector_norm(proj) / denom).item())
    residual = float((torch.linalg.vector_norm(vec - proj) / denom).item())
    return frac, residual, proj.reshape_as(x.detach().float())


def remove_control_span(x: torch.Tensor, controls: Iterable[torch.Tensor]) -> torch.Tensor:
    _, _, proj = projection_fraction(x, controls)
    return x.detach().float() - proj


__all__ = [
    "default_control_span",
    "match_norm",
    "projection_fraction",
    "random_control_like",
    "remove_control_span",
    "stable_control_like",
]
