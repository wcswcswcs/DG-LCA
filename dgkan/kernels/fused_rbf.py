"""Triton RBF/FastKAN two-layer PrimitiveKAN kernels.

This is a narrow v12.10 family-specific L3 path for fixed-center Gaussian RBF
PrimitiveKAN variants with K=2 or K=4.  It recomputes the fixed basis in
backward instead of materializing dense [B, D, K] or [B, H, K] tensors.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn.functional as F

try:  # pragma: no cover - Triton availability is environment dependent.
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except Exception:  # pragma: no cover
    triton = None
    tl = None
    TRITON_AVAILABLE = False


if TRITON_AVAILABLE:

    @triton.jit
    def _tanh_tl(x):
        return 2.0 / (1.0 + tl.exp(-2.0 * x)) - 1.0


    @triton.jit
    def _rbf_forward_hidden_kernel(
        x_ptr,
        mu_ptr,
        std_ptr,
        centers_ptr,
        scales_ptr,
        w1_ptr,
        h_ptr,
        B: tl.constexpr,
        D: tl.constexpr,
        H: tl.constexpr,
        K: tl.constexpr,
        INV_SQRT_D: tl.constexpr,
        BLOCK_B: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        b_block = tl.program_id(0)
        h_block = tl.program_id(1)
        offs_b = b_block * BLOCK_B + tl.arange(0, BLOCK_B)
        offs_h = h_block * BLOCK_H + tl.arange(0, BLOCK_H)
        offs_d = tl.arange(0, BLOCK_D)
        b_mask = offs_b < B
        h_mask = offs_h < H
        width = tl.maximum(tl.load(scales_ptr), 1.0e-3)
        acc = tl.zeros((BLOCK_B, BLOCK_H), tl.float32)
        for start in range(0, D, BLOCK_D):
            d = start + offs_d
            d_mask = d < D
            xv = tl.load(x_ptr + offs_b[:, None] * D + d[None, :], mask=b_mask[:, None] & d_mask[None, :], other=0.0)
            mu = tl.load(mu_ptr + d, mask=d_mask, other=0.0)
            std = tl.maximum(tl.load(std_ptr + d, mask=d_mask, other=1.0), 1.0e-3)
            z = _tanh_tl((xv - mu[None, :]) / std[None, :])
            for kk in range(0, K):
                center = tl.load(centers_ptr + kk)
                r = (z - center) / width
                phi = tl.exp(-0.5 * r * r)
                w = tl.load(w1_ptr + ((d[:, None] * H + offs_h[None, :]) * K + kk), mask=d_mask[:, None] & h_mask[None, :], other=0.0)
                acc += tl.dot(phi, w, input_precision="tf32x3")
        h_val = _tanh_tl(acc * INV_SQRT_D)
        tl.store(h_ptr + offs_b[:, None] * H + offs_h[None, :], h_val, mask=b_mask[:, None] & h_mask[None, :])


    @triton.jit
    def _rbf_forward_logits_kernel(
        h_ptr,
        centers_ptr,
        scales_ptr,
        w2_ptr,
        logits_ptr,
        B: tl.constexpr,
        H: tl.constexpr,
        C: tl.constexpr,
        K: tl.constexpr,
        INV_SQRT_H: tl.constexpr,
        BLOCK_B: tl.constexpr,
        BLOCK_H: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        b_block = tl.program_id(0)
        c_block = tl.program_id(1)
        offs_b = b_block * BLOCK_B + tl.arange(0, BLOCK_B)
        offs_c = c_block * BLOCK_C + tl.arange(0, BLOCK_C)
        offs_h = tl.arange(0, BLOCK_H)
        b_mask = offs_b < B
        c_mask = offs_c < C
        width = tl.maximum(tl.load(scales_ptr), 1.0e-3)
        acc = tl.zeros((BLOCK_B, BLOCK_C), tl.float32)
        for start in range(0, H, BLOCK_H):
            hh = start + offs_h
            h_mask = hh < H
            h_val = tl.load(h_ptr + offs_b[:, None] * H + hh[None, :], mask=b_mask[:, None] & h_mask[None, :], other=0.0)
            for kk in range(0, K):
                center = tl.load(centers_ptr + kk)
                r = (h_val - center) / width
                phi = tl.exp(-0.5 * r * r)
                w = tl.load(w2_ptr + ((hh[:, None] * C + offs_c[None, :]) * K + kk), mask=h_mask[:, None] & c_mask[None, :], other=0.0)
                acc += tl.dot(phi, w, input_precision="tf32x3")
        tl.store(logits_ptr + offs_b[:, None] * C + offs_c[None, :], acc * INV_SQRT_H, mask=b_mask[:, None] & c_mask[None, :])


    @triton.jit
    def _rbf_w2_grad_kernel(
        grad_logits_ptr,
        h_ptr,
        centers_ptr,
        scales_ptr,
        gw2_ptr,
        B: tl.constexpr,
        H: tl.constexpr,
        C: tl.constexpr,
        K: tl.constexpr,
        INV_SQRT_H: tl.constexpr,
        BLOCK_B: tl.constexpr,
    ):
        hh = tl.program_id(0)
        cc = tl.program_id(1)
        offs = tl.arange(0, BLOCK_B)
        mask = offs < B
        width = tl.maximum(tl.load(scales_ptr), 1.0e-3)
        g = tl.load(grad_logits_ptr + offs * C + cc, mask=mask, other=0.0)
        h_val = tl.load(h_ptr + offs * H + hh, mask=mask, other=0.0)
        for kk in range(0, K):
            center = tl.load(centers_ptr + kk)
            r = (h_val - center) / width
            phi = tl.exp(-0.5 * r * r)
            grad = tl.sum(phi * g, axis=0) * INV_SQRT_H
            tl.store(gw2_ptr + ((hh * C + cc) * K + kk), grad)


    @triton.jit
    def _rbf_w1_grad_blockd_kernel(
        x_ptr,
        mu_ptr,
        std_ptr,
        centers_ptr,
        scales_ptr,
        w2_ptr,
        grad_logits_ptr,
        h_ptr,
        gw1_ptr,
        B: tl.constexpr,
        D: tl.constexpr,
        H: tl.constexpr,
        C: tl.constexpr,
        K: tl.constexpr,
        INV_SQRT_D: tl.constexpr,
        INV_SQRT_H: tl.constexpr,
        BLOCK_B: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        d_block = tl.program_id(0)
        hh = tl.program_id(1)
        offs_b = tl.arange(0, BLOCK_B)
        offs_d = d_block * BLOCK_D + tl.arange(0, BLOCK_D)
        mask_b = offs_b < B
        mask_d = offs_d < D
        width = tl.maximum(tl.load(scales_ptr), 1.0e-3)

        h_val = tl.load(h_ptr + offs_b * H + hh, mask=mask_b, other=0.0)
        grad_h = tl.full((BLOCK_B,), 0.0, tl.float32)
        for cc in range(0, C):
            g = tl.load(grad_logits_ptr + offs_b * C + cc, mask=mask_b, other=0.0)
            local = tl.full((BLOCK_B,), 0.0, tl.float32)
            for kk in range(0, K):
                center = tl.load(centers_ptr + kk)
                r_h = (h_val - center) / width
                phi_h = tl.exp(-0.5 * r_h * r_h)
                dphi_h = -(r_h / width) * phi_h
                w = tl.load(w2_ptr + ((hh * C + cc) * K + kk))
                local += w * dphi_h
            grad_h += g * local * INV_SQRT_H
        grad_pre = grad_h * (1.0 - h_val * h_val)

        xv = tl.load(x_ptr + offs_b[:, None] * D + offs_d[None, :], mask=mask_b[:, None] & mask_d[None, :], other=0.0)
        mu = tl.load(mu_ptr + offs_d, mask=mask_d, other=0.0)
        std = tl.maximum(tl.load(std_ptr + offs_d, mask=mask_d, other=1.0), 1.0e-3)
        z = _tanh_tl((xv - mu[None, :]) / std[None, :])
        gp = grad_pre[:, None]
        for kk in range(0, K):
            center = tl.load(centers_ptr + kk)
            r_z = (z - center) / width
            phi_z = tl.exp(-0.5 * r_z * r_z)
            grad = tl.sum(phi_z * gp, axis=0) * INV_SQRT_D
            tl.store(gw1_ptr + ((offs_d * H + hh) * K + kk), grad, mask=mask_d)


def _check_model(model, k: int | None = None) -> None:
    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is required for fused_rbf")
    if getattr(model.spec, "basis_name", "") not in {"fastkan_rbf", "compact_rbf"}:
        raise ValueError("fused_rbf requires fastkan_rbf or compact_rbf PrimitiveKAN")
    if k is not None and int(model.k) != int(k):
        raise ValueError(f"fused_rbf expected K={k}")
    if int(model.k) not in {2, 4}:
        raise ValueError("fused_rbf currently supports K=2 or K=4")


def forward_matmul(model, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    _check_model(model)
    if x.dtype != torch.float32 or x.device.type != "cuda":
        raise ValueError("fused_rbf forward_matmul requires fp32 CUDA input")
    x = x.contiguous()
    batch = int(x.shape[0])
    h_dim = int(model.hidden_dim)
    c_dim = int(model.output_dim)
    k = int(model.k)
    logits = torch.empty((batch, c_dim), device=x.device, dtype=torch.float32)
    h = torch.empty((batch, h_dim), device=x.device, dtype=torch.float32)
    block_b = 16
    block_d = 64
    block_h = 32
    block_c = max(16, triton.next_power_of_2(c_dim))
    _rbf_forward_hidden_kernel[(triton.cdiv(batch, block_b), triton.cdiv(h_dim, block_h))](
        x,
        model.mu,
        model.std,
        model.centers,
        model.scales,
        model.w1,
        h,
        batch,
        int(model.input_dim),
        h_dim,
        k,
        1.0 / math.sqrt(max(1, int(model.input_dim))),
        BLOCK_B=block_b,
        BLOCK_D=block_d,
        BLOCK_H=block_h,
    )
    _rbf_forward_logits_kernel[(triton.cdiv(batch, block_b), triton.cdiv(c_dim, block_c))](
        h,
        model.centers,
        model.scales,
        model.w2,
        logits,
        batch,
        h_dim,
        c_dim,
        k,
        1.0 / math.sqrt(max(1, h_dim)),
        BLOCK_B=block_b,
        BLOCK_H=block_h,
        BLOCK_C=block_c,
    )
    return logits, h


def backward(model, x: torch.Tensor, y: torch.Tensor, logits: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    _check_model(model)
    if x.dtype != torch.float32 or x.device.type != "cuda":
        raise ValueError("fused_rbf backward requires fp32 CUDA input")
    x = x.contiguous()
    with torch.no_grad():
        loss = F.cross_entropy(logits, y)
        grad_logits = torch.softmax(logits, dim=1)
        grad_logits[torch.arange(int(y.numel()), device=y.device), y] -= 1.0
        grad_logits = (grad_logits / float(max(1, int(y.numel())))).contiguous()
        gw1 = torch.empty_like(model.w1)
        gw2 = torch.empty_like(model.w2)
        batch = int(x.shape[0])
        k = int(model.k)
        block_b = triton.next_power_of_2(batch)
        _rbf_w2_grad_kernel[(int(model.hidden_dim), int(model.output_dim))](
            grad_logits,
            h,
            model.centers,
            model.scales,
            gw2,
            batch,
            int(model.hidden_dim),
            int(model.output_dim),
            k,
            1.0 / math.sqrt(max(1, int(model.hidden_dim))),
            BLOCK_B=block_b,
        )
        block_d = 32
        _rbf_w1_grad_blockd_kernel[(triton.cdiv(int(model.input_dim), block_d), int(model.hidden_dim))](
            x,
            model.mu,
            model.std,
            model.centers,
            model.scales,
            model.w2,
            grad_logits,
            h,
            gw1,
            batch,
            int(model.input_dim),
            int(model.hidden_dim),
            int(model.output_dim),
            k,
            1.0 / math.sqrt(max(1, int(model.input_dim))),
            1.0 / math.sqrt(max(1, int(model.hidden_dim))),
            BLOCK_B=block_b,
            BLOCK_D=block_d,
        )
        model.w1.grad = gw1
        model.w2.grad = gw2
        return loss.detach()
