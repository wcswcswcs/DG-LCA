"""Triton pure-trigonometric Fourier K4/K5 PrimitiveKAN kernels.

These kernels are the D-FOU-Trig counterpart to the low-frequency Fourier
kernel: they recompute sin/cos channels from inputs and hidden activations
inside Triton kernels instead of materializing B x D x K basis tensors.
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
    def _trig_k4_forward_hidden_kernel(
        x_ptr,
        mu_ptr,
        std_ptr,
        w1_ptr,
        h_ptr,
        B: tl.constexpr,
        D: tl.constexpr,
        H: tl.constexpr,
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
        acc = tl.zeros((BLOCK_B, BLOCK_H), tl.float32)
        for start in range(0, D, BLOCK_D):
            d = start + offs_d
            d_mask = d < D
            xv = tl.load(x_ptr + offs_b[:, None] * D + d[None, :], mask=b_mask[:, None] & d_mask[None, :], other=0.0)
            mu = tl.load(mu_ptr + d, mask=d_mask, other=0.0)
            std = tl.maximum(tl.load(std_ptr + d, mask=d_mask, other=1.0), 1.0e-3)
            z = _tanh_tl((xv - mu[None, :]) / std[None, :])
            sin1 = tl.sin(3.141592653589793 * z)
            cos1 = tl.cos(3.141592653589793 * z)
            sin2 = tl.sin(6.283185307179586 * z)
            cos2 = tl.cos(6.283185307179586 * z)
            mask = d_mask[:, None] & h_mask[None, :]
            w10 = tl.load(w1_ptr + ((d[:, None] * H + offs_h[None, :]) * 4), mask=mask, other=0.0)
            w11 = tl.load(w1_ptr + ((d[:, None] * H + offs_h[None, :]) * 4 + 1), mask=mask, other=0.0)
            w12 = tl.load(w1_ptr + ((d[:, None] * H + offs_h[None, :]) * 4 + 2), mask=mask, other=0.0)
            w13 = tl.load(w1_ptr + ((d[:, None] * H + offs_h[None, :]) * 4 + 3), mask=mask, other=0.0)
            acc += (
                tl.dot(sin1, w10, input_precision="tf32x3")
                + tl.dot(cos1, w11, input_precision="tf32x3")
                + tl.dot(sin2, w12, input_precision="tf32x3")
                + tl.dot(cos2, w13, input_precision="tf32x3")
            )
        h_val = _tanh_tl(acc * 0.5 * INV_SQRT_D)
        tl.store(h_ptr + offs_b[:, None] * H + offs_h[None, :], h_val, mask=b_mask[:, None] & h_mask[None, :])


    @triton.jit
    def _trig_k4_forward_logits_kernel(
        h_ptr,
        w2_ptr,
        logits_ptr,
        B: tl.constexpr,
        H: tl.constexpr,
        C: tl.constexpr,
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
        acc = tl.zeros((BLOCK_B, BLOCK_C), tl.float32)
        for start in range(0, H, BLOCK_H):
            hh = start + offs_h
            h_mask = hh < H
            h_val = tl.load(h_ptr + offs_b[:, None] * H + hh[None, :], mask=b_mask[:, None] & h_mask[None, :], other=0.0)
            sin1 = tl.sin(3.141592653589793 * h_val)
            cos1 = tl.cos(3.141592653589793 * h_val)
            sin2 = tl.sin(6.283185307179586 * h_val)
            cos2 = tl.cos(6.283185307179586 * h_val)
            mask = h_mask[:, None] & c_mask[None, :]
            w20 = tl.load(w2_ptr + ((hh[:, None] * C + offs_c[None, :]) * 4), mask=mask, other=0.0)
            w21 = tl.load(w2_ptr + ((hh[:, None] * C + offs_c[None, :]) * 4 + 1), mask=mask, other=0.0)
            w22 = tl.load(w2_ptr + ((hh[:, None] * C + offs_c[None, :]) * 4 + 2), mask=mask, other=0.0)
            w23 = tl.load(w2_ptr + ((hh[:, None] * C + offs_c[None, :]) * 4 + 3), mask=mask, other=0.0)
            acc += (
                tl.dot(sin1, w20, input_precision="tf32x3")
                + tl.dot(cos1, w21, input_precision="tf32x3")
                + tl.dot(sin2, w22, input_precision="tf32x3")
                + tl.dot(cos2, w23, input_precision="tf32x3")
            )
        tl.store(logits_ptr + offs_b[:, None] * C + offs_c[None, :], acc * 0.5 * INV_SQRT_H, mask=b_mask[:, None] & c_mask[None, :])


    @triton.jit
    def _trig_k5_forward_hidden_kernel(
        x_ptr,
        mu_ptr,
        std_ptr,
        w1_ptr,
        h_ptr,
        B: tl.constexpr,
        D: tl.constexpr,
        H: tl.constexpr,
        INV_SQRT_D: tl.constexpr,
        INV_SQRT_5: tl.constexpr,
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
        acc = tl.zeros((BLOCK_B, BLOCK_H), tl.float32)
        for start in range(0, D, BLOCK_D):
            d = start + offs_d
            d_mask = d < D
            xv = tl.load(x_ptr + offs_b[:, None] * D + d[None, :], mask=b_mask[:, None] & d_mask[None, :], other=0.0)
            mu = tl.load(mu_ptr + d, mask=d_mask, other=0.0)
            std = tl.maximum(tl.load(std_ptr + d, mask=d_mask, other=1.0), 1.0e-3)
            z = _tanh_tl((xv - mu[None, :]) / std[None, :])
            sin1 = tl.sin(3.141592653589793 * z)
            cos1 = tl.cos(3.141592653589793 * z)
            sin2 = tl.sin(6.283185307179586 * z)
            cos2 = tl.cos(6.283185307179586 * z)
            mask = d_mask[:, None] & h_mask[None, :]
            w10 = tl.load(w1_ptr + ((d[:, None] * H + offs_h[None, :]) * 5), mask=mask, other=0.0)
            w11 = tl.load(w1_ptr + ((d[:, None] * H + offs_h[None, :]) * 5 + 1), mask=mask, other=0.0)
            w12 = tl.load(w1_ptr + ((d[:, None] * H + offs_h[None, :]) * 5 + 2), mask=mask, other=0.0)
            w13 = tl.load(w1_ptr + ((d[:, None] * H + offs_h[None, :]) * 5 + 3), mask=mask, other=0.0)
            w14 = tl.load(w1_ptr + ((d[:, None] * H + offs_h[None, :]) * 5 + 4), mask=mask, other=0.0)
            acc += (
                tl.sum(w10, axis=0)[None, :]
                + tl.dot(sin1, w11, input_precision="tf32x3")
                + tl.dot(cos1, w12, input_precision="tf32x3")
                + tl.dot(sin2, w13, input_precision="tf32x3")
                + tl.dot(cos2, w14, input_precision="tf32x3")
            )
        h_val = _tanh_tl(acc * INV_SQRT_5 * INV_SQRT_D)
        tl.store(h_ptr + offs_b[:, None] * H + offs_h[None, :], h_val, mask=b_mask[:, None] & h_mask[None, :])


    @triton.jit
    def _trig_k5_forward_logits_kernel(
        h_ptr,
        w2_ptr,
        logits_ptr,
        B: tl.constexpr,
        H: tl.constexpr,
        C: tl.constexpr,
        INV_SQRT_H: tl.constexpr,
        INV_SQRT_5: tl.constexpr,
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
        acc = tl.zeros((BLOCK_B, BLOCK_C), tl.float32)
        for start in range(0, H, BLOCK_H):
            hh = start + offs_h
            h_mask = hh < H
            h_val = tl.load(h_ptr + offs_b[:, None] * H + hh[None, :], mask=b_mask[:, None] & h_mask[None, :], other=0.0)
            sin1 = tl.sin(3.141592653589793 * h_val)
            cos1 = tl.cos(3.141592653589793 * h_val)
            sin2 = tl.sin(6.283185307179586 * h_val)
            cos2 = tl.cos(6.283185307179586 * h_val)
            mask = h_mask[:, None] & c_mask[None, :]
            w20 = tl.load(w2_ptr + ((hh[:, None] * C + offs_c[None, :]) * 5), mask=mask, other=0.0)
            w21 = tl.load(w2_ptr + ((hh[:, None] * C + offs_c[None, :]) * 5 + 1), mask=mask, other=0.0)
            w22 = tl.load(w2_ptr + ((hh[:, None] * C + offs_c[None, :]) * 5 + 2), mask=mask, other=0.0)
            w23 = tl.load(w2_ptr + ((hh[:, None] * C + offs_c[None, :]) * 5 + 3), mask=mask, other=0.0)
            w24 = tl.load(w2_ptr + ((hh[:, None] * C + offs_c[None, :]) * 5 + 4), mask=mask, other=0.0)
            acc += (
                tl.sum(w20, axis=0)[None, :]
                + tl.dot(sin1, w21, input_precision="tf32x3")
                + tl.dot(cos1, w22, input_precision="tf32x3")
                + tl.dot(sin2, w23, input_precision="tf32x3")
                + tl.dot(cos2, w24, input_precision="tf32x3")
            )
        tl.store(logits_ptr + offs_b[:, None] * C + offs_c[None, :], acc * INV_SQRT_5 * INV_SQRT_H, mask=b_mask[:, None] & c_mask[None, :])


    @triton.jit
    def _trig_k4_w2_grad_kernel(
        grad_logits_ptr,
        h_ptr,
        gw2_ptr,
        B: tl.constexpr,
        H: tl.constexpr,
        C: tl.constexpr,
        INV_SQRT_H: tl.constexpr,
        BLOCK_B: tl.constexpr,
    ):
        hh = tl.program_id(0)
        cc = tl.program_id(1)
        offs = tl.arange(0, BLOCK_B)
        mask = offs < B
        g = tl.load(grad_logits_ptr + offs * C + cc, mask=mask, other=0.0)
        h_val = tl.load(h_ptr + offs * H + hh, mask=mask, other=0.0)
        base = (hh * C + cc) * 4
        tl.store(gw2_ptr + base, tl.sum(tl.sin(3.141592653589793 * h_val) * g, axis=0) * 0.5 * INV_SQRT_H)
        tl.store(gw2_ptr + base + 1, tl.sum(tl.cos(3.141592653589793 * h_val) * g, axis=0) * 0.5 * INV_SQRT_H)
        tl.store(gw2_ptr + base + 2, tl.sum(tl.sin(6.283185307179586 * h_val) * g, axis=0) * 0.5 * INV_SQRT_H)
        tl.store(gw2_ptr + base + 3, tl.sum(tl.cos(6.283185307179586 * h_val) * g, axis=0) * 0.5 * INV_SQRT_H)


    @triton.jit
    def _trig_k5_w2_grad_kernel(
        grad_logits_ptr,
        h_ptr,
        gw2_ptr,
        B: tl.constexpr,
        H: tl.constexpr,
        C: tl.constexpr,
        INV_SQRT_H: tl.constexpr,
        INV_SQRT_5: tl.constexpr,
        BLOCK_B: tl.constexpr,
    ):
        hh = tl.program_id(0)
        cc = tl.program_id(1)
        offs = tl.arange(0, BLOCK_B)
        mask = offs < B
        g = tl.load(grad_logits_ptr + offs * C + cc, mask=mask, other=0.0)
        h_val = tl.load(h_ptr + offs * H + hh, mask=mask, other=0.0)
        base = (hh * C + cc) * 5
        scale = INV_SQRT_5 * INV_SQRT_H
        tl.store(gw2_ptr + base, tl.sum(g, axis=0) * scale)
        tl.store(gw2_ptr + base + 1, tl.sum(tl.sin(3.141592653589793 * h_val) * g, axis=0) * scale)
        tl.store(gw2_ptr + base + 2, tl.sum(tl.cos(3.141592653589793 * h_val) * g, axis=0) * scale)
        tl.store(gw2_ptr + base + 3, tl.sum(tl.sin(6.283185307179586 * h_val) * g, axis=0) * scale)
        tl.store(gw2_ptr + base + 4, tl.sum(tl.cos(6.283185307179586 * h_val) * g, axis=0) * scale)


    @triton.jit
    def _trig_k4_w1_grad_kernel(
        x_ptr,
        mu_ptr,
        std_ptr,
        w2_ptr,
        grad_logits_ptr,
        h_ptr,
        gw1_ptr,
        B: tl.constexpr,
        D: tl.constexpr,
        H: tl.constexpr,
        C: tl.constexpr,
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

        h_val = tl.load(h_ptr + offs_b * H + hh, mask=mask_b, other=0.0)
        sin1h = tl.sin(3.141592653589793 * h_val)
        cos1h = tl.cos(3.141592653589793 * h_val)
        sin2h = tl.sin(6.283185307179586 * h_val)
        cos2h = tl.cos(6.283185307179586 * h_val)
        grad_h = tl.full((BLOCK_B,), 0.0, tl.float32)
        for cc in range(0, C):
            g = tl.load(grad_logits_ptr + offs_b * C + cc, mask=mask_b, other=0.0)
            base = (hh * C + cc) * 4
            w20 = tl.load(w2_ptr + base)
            w21 = tl.load(w2_ptr + base + 1)
            w22 = tl.load(w2_ptr + base + 2)
            w23 = tl.load(w2_ptr + base + 3)
            grad_h += g * (
                w20 * 3.141592653589793 * cos1h
                - w21 * 3.141592653589793 * sin1h
                + w22 * 6.283185307179586 * cos2h
                - w23 * 6.283185307179586 * sin2h
            ) * 0.5 * INV_SQRT_H
        grad_pre = grad_h * (1.0 - h_val * h_val)

        xv = tl.load(x_ptr + offs_b[:, None] * D + offs_d[None, :], mask=mask_b[:, None] & mask_d[None, :], other=0.0)
        mu = tl.load(mu_ptr + offs_d, mask=mask_d, other=0.0)
        std = tl.maximum(tl.load(std_ptr + offs_d, mask=mask_d, other=1.0), 1.0e-3)
        z = _tanh_tl((xv - mu[None, :]) / std[None, :])
        gp = grad_pre[:, None]
        base1 = (offs_d * H + hh) * 4
        tl.store(gw1_ptr + base1, tl.sum(tl.sin(3.141592653589793 * z) * gp, axis=0) * 0.5 * INV_SQRT_D, mask=mask_d)
        tl.store(gw1_ptr + base1 + 1, tl.sum(tl.cos(3.141592653589793 * z) * gp, axis=0) * 0.5 * INV_SQRT_D, mask=mask_d)
        tl.store(gw1_ptr + base1 + 2, tl.sum(tl.sin(6.283185307179586 * z) * gp, axis=0) * 0.5 * INV_SQRT_D, mask=mask_d)
        tl.store(gw1_ptr + base1 + 3, tl.sum(tl.cos(6.283185307179586 * z) * gp, axis=0) * 0.5 * INV_SQRT_D, mask=mask_d)


    @triton.jit
    def _trig_k5_w1_grad_kernel(
        x_ptr,
        mu_ptr,
        std_ptr,
        w2_ptr,
        grad_logits_ptr,
        h_ptr,
        gw1_ptr,
        B: tl.constexpr,
        D: tl.constexpr,
        H: tl.constexpr,
        C: tl.constexpr,
        INV_SQRT_D: tl.constexpr,
        INV_SQRT_H: tl.constexpr,
        INV_SQRT_5: tl.constexpr,
        BLOCK_B: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        d_block = tl.program_id(0)
        hh = tl.program_id(1)
        offs_b = tl.arange(0, BLOCK_B)
        offs_d = d_block * BLOCK_D + tl.arange(0, BLOCK_D)
        mask_b = offs_b < B
        mask_d = offs_d < D

        h_val = tl.load(h_ptr + offs_b * H + hh, mask=mask_b, other=0.0)
        sin1h = tl.sin(3.141592653589793 * h_val)
        cos1h = tl.cos(3.141592653589793 * h_val)
        sin2h = tl.sin(6.283185307179586 * h_val)
        cos2h = tl.cos(6.283185307179586 * h_val)
        grad_h = tl.full((BLOCK_B,), 0.0, tl.float32)
        for cc in range(0, C):
            g = tl.load(grad_logits_ptr + offs_b * C + cc, mask=mask_b, other=0.0)
            base = (hh * C + cc) * 5
            w21 = tl.load(w2_ptr + base + 1)
            w22 = tl.load(w2_ptr + base + 2)
            w23 = tl.load(w2_ptr + base + 3)
            w24 = tl.load(w2_ptr + base + 4)
            grad_h += g * (
                w21 * 3.141592653589793 * cos1h
                - w22 * 3.141592653589793 * sin1h
                + w23 * 6.283185307179586 * cos2h
                - w24 * 6.283185307179586 * sin2h
            ) * INV_SQRT_5 * INV_SQRT_H
        grad_pre = grad_h * (1.0 - h_val * h_val)

        xv = tl.load(x_ptr + offs_b[:, None] * D + offs_d[None, :], mask=mask_b[:, None] & mask_d[None, :], other=0.0)
        mu = tl.load(mu_ptr + offs_d, mask=mask_d, other=0.0)
        std = tl.maximum(tl.load(std_ptr + offs_d, mask=mask_d, other=1.0), 1.0e-3)
        z = _tanh_tl((xv - mu[None, :]) / std[None, :])
        gp = grad_pre[:, None]
        base1 = (offs_d * H + hh) * 5
        scale = INV_SQRT_5 * INV_SQRT_D
        tl.store(gw1_ptr + base1, tl.sum(gp, axis=0) * scale, mask=mask_d)
        tl.store(gw1_ptr + base1 + 1, tl.sum(tl.sin(3.141592653589793 * z) * gp, axis=0) * scale, mask=mask_d)
        tl.store(gw1_ptr + base1 + 2, tl.sum(tl.cos(3.141592653589793 * z) * gp, axis=0) * scale, mask=mask_d)
        tl.store(gw1_ptr + base1 + 3, tl.sum(tl.sin(6.283185307179586 * z) * gp, axis=0) * scale, mask=mask_d)
        tl.store(gw1_ptr + base1 + 4, tl.sum(tl.cos(6.283185307179586 * z) * gp, axis=0) * scale, mask=mask_d)


def _check_model(model, basis_name: str, k: int) -> None:
    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available")
    if str(getattr(model.spec, "basis_name", "")) != basis_name or int(model.k) != int(k):
        raise ValueError(f"fused_fourier_trig requires PrimitiveKAN {basis_name} K{k}")
    if model.w1.dtype != torch.float32 or model.w2.dtype != torch.float32:
        raise ValueError("fused_fourier_trig currently supports fp32 weights only")


def forward_matmul_k4(model, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    _check_model(model, "fourier_trig", 4)
    if x.dtype != torch.float32 or x.device.type != "cuda":
        raise ValueError("fused_fourier_trig forward_matmul_k4 requires fp32 CUDA input")
    x = x.contiguous()
    batch = int(x.shape[0])
    h_dim = int(model.hidden_dim)
    c_dim = int(model.output_dim)
    logits = torch.empty((batch, c_dim), device=x.device, dtype=torch.float32)
    h = torch.empty((batch, h_dim), device=x.device, dtype=torch.float32)
    block_b = 16
    block_d = 64
    block_h = 32
    block_c = max(16, triton.next_power_of_2(c_dim))
    _trig_k4_forward_hidden_kernel[(triton.cdiv(batch, block_b), triton.cdiv(h_dim, block_h))](
        x,
        model.mu,
        model.std,
        model.w1,
        h,
        batch,
        int(model.input_dim),
        h_dim,
        1.0 / math.sqrt(max(1, int(model.input_dim))),
        BLOCK_B=block_b,
        BLOCK_D=block_d,
        BLOCK_H=block_h,
    )
    _trig_k4_forward_logits_kernel[(triton.cdiv(batch, block_b), triton.cdiv(c_dim, block_c))](
        h,
        model.w2,
        logits,
        batch,
        h_dim,
        c_dim,
        1.0 / math.sqrt(max(1, h_dim)),
        BLOCK_B=block_b,
        BLOCK_H=block_h,
        BLOCK_C=block_c,
    )
    return logits, h


def forward_matmul_k5(model, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    _check_model(model, "fourier_trig_dc", 5)
    if x.dtype != torch.float32 or x.device.type != "cuda":
        raise ValueError("fused_fourier_trig forward_matmul_k5 requires fp32 CUDA input")
    x = x.contiguous()
    batch = int(x.shape[0])
    h_dim = int(model.hidden_dim)
    c_dim = int(model.output_dim)
    logits = torch.empty((batch, c_dim), device=x.device, dtype=torch.float32)
    h = torch.empty((batch, h_dim), device=x.device, dtype=torch.float32)
    block_b = 16
    block_d = 64
    block_h = 32
    block_c = max(16, triton.next_power_of_2(c_dim))
    inv_sqrt_5 = 1.0 / math.sqrt(5.0)
    _trig_k5_forward_hidden_kernel[(triton.cdiv(batch, block_b), triton.cdiv(h_dim, block_h))](
        x,
        model.mu,
        model.std,
        model.w1,
        h,
        batch,
        int(model.input_dim),
        h_dim,
        1.0 / math.sqrt(max(1, int(model.input_dim))),
        inv_sqrt_5,
        BLOCK_B=block_b,
        BLOCK_D=block_d,
        BLOCK_H=block_h,
    )
    _trig_k5_forward_logits_kernel[(triton.cdiv(batch, block_b), triton.cdiv(c_dim, block_c))](
        h,
        model.w2,
        logits,
        batch,
        h_dim,
        c_dim,
        1.0 / math.sqrt(max(1, h_dim)),
        inv_sqrt_5,
        BLOCK_B=block_b,
        BLOCK_H=block_h,
        BLOCK_C=block_c,
    )
    return logits, h


def backward_from_grad_logits_k4(model, x: torch.Tensor, grad_logits: torch.Tensor, logits: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    _check_model(model, "fourier_trig", 4)
    if x.dtype != torch.float32 or x.device.type != "cuda":
        raise ValueError("fused_fourier_trig backward_from_grad_logits_k4 requires fp32 CUDA input")
    x = x.contiguous()
    with torch.no_grad():
        grad_logits = grad_logits.to(device=logits.device, dtype=logits.dtype).contiguous()
        gw1 = torch.empty_like(model.w1)
        gw2 = torch.empty_like(model.w2)
        batch = int(x.shape[0])
        block_b = triton.next_power_of_2(batch)
        _trig_k4_w2_grad_kernel[(int(model.hidden_dim), int(model.output_dim))](
            grad_logits,
            h,
            gw2,
            batch,
            int(model.hidden_dim),
            int(model.output_dim),
            1.0 / math.sqrt(max(1, int(model.hidden_dim))),
            BLOCK_B=block_b,
        )
        block_d = 32
        _trig_k4_w1_grad_kernel[(triton.cdiv(int(model.input_dim), block_d), int(model.hidden_dim))](
            x,
            model.mu,
            model.std,
            model.w2,
            grad_logits,
            h,
            gw1,
            batch,
            int(model.input_dim),
            int(model.hidden_dim),
            int(model.output_dim),
            1.0 / math.sqrt(max(1, int(model.input_dim))),
            1.0 / math.sqrt(max(1, int(model.hidden_dim))),
            BLOCK_B=block_b,
            BLOCK_D=block_d,
        )
        model.w1.grad = gw1
        model.w2.grad = gw2
        return (logits.detach().float() * grad_logits.detach().float()).sum().detach()


def backward_from_grad_logits_k5(model, x: torch.Tensor, grad_logits: torch.Tensor, logits: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    _check_model(model, "fourier_trig_dc", 5)
    if x.dtype != torch.float32 or x.device.type != "cuda":
        raise ValueError("fused_fourier_trig backward_from_grad_logits_k5 requires fp32 CUDA input")
    x = x.contiguous()
    with torch.no_grad():
        grad_logits = grad_logits.to(device=logits.device, dtype=logits.dtype).contiguous()
        gw1 = torch.empty_like(model.w1)
        gw2 = torch.empty_like(model.w2)
        batch = int(x.shape[0])
        block_b = triton.next_power_of_2(batch)
        inv_sqrt_5 = 1.0 / math.sqrt(5.0)
        _trig_k5_w2_grad_kernel[(int(model.hidden_dim), int(model.output_dim))](
            grad_logits,
            h,
            gw2,
            batch,
            int(model.hidden_dim),
            int(model.output_dim),
            1.0 / math.sqrt(max(1, int(model.hidden_dim))),
            inv_sqrt_5,
            BLOCK_B=block_b,
        )
        block_d = 32
        _trig_k5_w1_grad_kernel[(triton.cdiv(int(model.input_dim), block_d), int(model.hidden_dim))](
            x,
            model.mu,
            model.std,
            model.w2,
            grad_logits,
            h,
            gw1,
            batch,
            int(model.input_dim),
            int(model.hidden_dim),
            int(model.output_dim),
            1.0 / math.sqrt(max(1, int(model.input_dim))),
            1.0 / math.sqrt(max(1, int(model.hidden_dim))),
            inv_sqrt_5,
            BLOCK_B=block_b,
            BLOCK_D=block_d,
        )
        model.w1.grad = gw1
        model.w2.grad = gw2
        return (logits.detach().float() * grad_logits.detach().float()).sum().detach()


def backward_k4(model, x: torch.Tensor, y: torch.Tensor, logits: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        loss = F.cross_entropy(logits, y)
        grad_logits = torch.softmax(logits, dim=1)
        grad_logits[torch.arange(int(y.numel()), device=y.device), y] -= 1.0
        grad_logits = (grad_logits / float(max(1, int(y.numel())))).contiguous()
    backward_from_grad_logits_k4(model, x, grad_logits, logits, h)
    return loss.detach()


def backward_k5(model, x: torch.Tensor, y: torch.Tensor, logits: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        loss = F.cross_entropy(logits, y)
        grad_logits = torch.softmax(logits, dim=1)
        grad_logits[torch.arange(int(y.numel()), device=y.device), y] -= 1.0
        grad_logits = (grad_logits / float(max(1, int(y.numel())))).contiguous()
    backward_from_grad_logits_k5(model, x, grad_logits, logits, h)
    return loss.detach()


__all__ = [
    "TRITON_AVAILABLE",
    "forward_matmul_k4",
    "forward_matmul_k5",
    "backward_from_grad_logits_k4",
    "backward_from_grad_logits_k5",
    "backward_k4",
    "backward_k5",
]
