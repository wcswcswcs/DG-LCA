"""D-CHE arbitrary-cotangent native path probe for v22.13."""

from __future__ import annotations

from typing import Any

import torch

from dgkan.kernels import cheby_fused
from dgkan.kernels import fused_chebyshev_k3


def kernel_status() -> dict[str, Any]:
    status = cheby_fused.kernel_status()
    arbitrary_available = int(hasattr(fused_chebyshev_k3, "backward_from_grad_logits"))
    return {
        "carrier": "D-CHE",
        "official_fused_kernel_available": int(status.get("official_fused_kernel_available", 0)),
        "arbitrary_cotangent_backward_available": arbitrary_available,
        "reason": "" if arbitrary_available else "available fused backward accepts supervised labels, not a generic output cotangent contract",
    }


def _flat_grads(model: torch.nn.Module, device: torch.device) -> torch.Tensor:
    chunks = []
    for p in model.parameters():
        if p.requires_grad:
            chunks.append(torch.zeros_like(p).reshape(-1) if p.grad is None else p.grad.detach().reshape(-1))
    return torch.cat(chunks) if chunks else torch.zeros(0, device=device)


def native_fused_vjp(model: torch.nn.Module, x: torch.Tensor, cotangent: torch.Tensor) -> torch.Tensor:
    model.zero_grad(set_to_none=True)
    logits, cache = model.manual_ce_forward_cache(x)
    if not cache or not isinstance(cache[0], str) or cache[0] not in {"cheby_k3_triton_l3_matmul", "cheby_k3_triton_l3_gradbuf"}:
        raise RuntimeError(f"D-CHE native arbitrary-cotangent fused VJP requires Chebyshev K3 Triton cache, got {cache[0] if cache else 'empty'}")
    _variant, cached_x, h = cache
    grad_logits = cotangent.to(device=logits.device, dtype=logits.dtype)
    if grad_logits.shape != logits.shape:
        raise RuntimeError(f"cotangent shape {tuple(grad_logits.shape)} does not match logits {tuple(logits.shape)}")
    fused_chebyshev_k3.backward_from_grad_logits(model, cached_x, grad_logits, logits, h, use_grad_buffers=str(cache[0]).endswith("gradbuf"))
    return _flat_grads(model, x.device)


def native_autograd_vjp(model: torch.nn.Module, x: torch.Tensor, cotangent: torch.Tensor) -> torch.Tensor:
    model.zero_grad(set_to_none=True)
    logits = model(x).float()
    (logits * cotangent.to(device=logits.device, dtype=logits.dtype)).sum().backward()
    chunks = []
    for p in model.parameters():
        if p.requires_grad:
            chunks.append(torch.zeros_like(p).reshape(-1) if p.grad is None else p.grad.detach().reshape(-1))
    return torch.cat(chunks) if chunks else torch.zeros(0, device=x.device)


__all__ = ["kernel_status", "native_autograd_vjp", "native_fused_vjp"]
