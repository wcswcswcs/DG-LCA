"""Triton Chebyshev K3 two-layer PrimitiveKAN kernels.

This narrow v12.8.3 repair path is valid for PrimitiveKAN with
basis_name=chebyshev and k=3 or k=4.  It recomputes the fixed Chebyshev
basis directly in the kernels:

    T0(z) = 1, T1(z) = z, T2(z) = 2 z^2 - 1.
    T3(z) = 4 z^3 - 3 z.

The goal is to turn the existing generic analytic manual path into a real
family-specific fused L3 path, so A4 expression can be opened without
pretending a torch-reduction fallback is an official fused kernel.
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
    def _cheby_k3_forward_hidden_kernel(
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
            t2 = 2.0 * z * z - 1.0
            w10 = tl.load(w1_ptr + ((d[:, None] * H + offs_h[None, :]) * 3), mask=d_mask[:, None] & h_mask[None, :], other=0.0)
            w11 = tl.load(w1_ptr + ((d[:, None] * H + offs_h[None, :]) * 3 + 1), mask=d_mask[:, None] & h_mask[None, :], other=0.0)
            w12 = tl.load(w1_ptr + ((d[:, None] * H + offs_h[None, :]) * 3 + 2), mask=d_mask[:, None] & h_mask[None, :], other=0.0)
            const_term = tl.sum(w10, axis=0)
            acc += const_term[None, :] + tl.dot(z, w11, input_precision="tf32x3") + tl.dot(t2, w12, input_precision="tf32x3")
        h_val = _tanh_tl(acc * INV_SQRT_D)
        tl.store(h_ptr + offs_b[:, None] * H + offs_h[None, :], h_val, mask=b_mask[:, None] & h_mask[None, :])


    @triton.jit
    def _cheby_k3_forward_logits_kernel(
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
            t2 = 2.0 * h_val * h_val - 1.0
            w20 = tl.load(w2_ptr + ((hh[:, None] * C + offs_c[None, :]) * 3), mask=h_mask[:, None] & c_mask[None, :], other=0.0)
            w21 = tl.load(w2_ptr + ((hh[:, None] * C + offs_c[None, :]) * 3 + 1), mask=h_mask[:, None] & c_mask[None, :], other=0.0)
            w22 = tl.load(w2_ptr + ((hh[:, None] * C + offs_c[None, :]) * 3 + 2), mask=h_mask[:, None] & c_mask[None, :], other=0.0)
            const_term = tl.sum(w20, axis=0)
            acc += const_term[None, :] + tl.dot(h_val, w21, input_precision="tf32x3") + tl.dot(t2, w22, input_precision="tf32x3")
        tl.store(logits_ptr + offs_b[:, None] * C + offs_c[None, :], acc * INV_SQRT_H, mask=b_mask[:, None] & c_mask[None, :])


    @triton.jit
    def _cheby_k3_w2_grad_kernel(
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
        t2 = 2.0 * h_val * h_val - 1.0
        g0 = tl.sum(g, axis=0) * INV_SQRT_H
        g1 = tl.sum(h_val * g, axis=0) * INV_SQRT_H
        g2 = tl.sum(t2 * g, axis=0) * INV_SQRT_H
        tl.store(gw2_ptr + ((hh * C + cc) * 3), g0)
        tl.store(gw2_ptr + ((hh * C + cc) * 3 + 1), g1)
        tl.store(gw2_ptr + ((hh * C + cc) * 3 + 2), g2)


    @triton.jit
    def _cheby_k3_w1_grad_blockd_kernel(
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
        grad_h = tl.full((BLOCK_B,), 0.0, tl.float32)
        for cc in range(0, C):
            g = tl.load(grad_logits_ptr + offs_b * C + cc, mask=mask_b, other=0.0)
            w21 = tl.load(w2_ptr + ((hh * C + cc) * 3 + 1))
            w22 = tl.load(w2_ptr + ((hh * C + cc) * 3 + 2))
            grad_h += g * (w21 + w22 * 4.0 * h_val) * INV_SQRT_H
        grad_pre = grad_h * (1.0 - h_val * h_val)

        xv = tl.load(
            x_ptr + offs_b[:, None] * D + offs_d[None, :],
            mask=mask_b[:, None] & mask_d[None, :],
            other=0.0,
        )
        mu = tl.load(mu_ptr + offs_d, mask=mask_d, other=0.0)
        std = tl.maximum(tl.load(std_ptr + offs_d, mask=mask_d, other=1.0), 1.0e-3)
        z = _tanh_tl((xv - mu[None, :]) / std[None, :])
        t2 = 2.0 * z * z - 1.0
        gp = grad_pre[:, None]
        g0 = tl.sum(gp, axis=0) * INV_SQRT_D
        g1 = tl.sum(z * gp, axis=0) * INV_SQRT_D
        g2 = tl.sum(t2 * gp, axis=0) * INV_SQRT_D
        tl.store(gw1_ptr + ((offs_d * H + hh) * 3), g0, mask=mask_d)
        tl.store(gw1_ptr + ((offs_d * H + hh) * 3 + 1), g1, mask=mask_d)
        tl.store(gw1_ptr + ((offs_d * H + hh) * 3 + 2), g2, mask=mask_d)


    @triton.jit
    def _cheby_k3_w1_grad_blockd_extra_kernel(
        x_ptr,
        mu_ptr,
        std_ptr,
        w2_ptr,
        grad_logits_ptr,
        extra_grad_h_ptr,
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
        grad_h = tl.load(extra_grad_h_ptr + offs_b * H + hh, mask=mask_b, other=0.0)
        for cc in range(0, C):
            g = tl.load(grad_logits_ptr + offs_b * C + cc, mask=mask_b, other=0.0)
            w21 = tl.load(w2_ptr + ((hh * C + cc) * 3 + 1))
            w22 = tl.load(w2_ptr + ((hh * C + cc) * 3 + 2))
            grad_h += g * (w21 + w22 * 4.0 * h_val) * INV_SQRT_H
        grad_pre = grad_h * (1.0 - h_val * h_val)

        xv = tl.load(
            x_ptr + offs_b[:, None] * D + offs_d[None, :],
            mask=mask_b[:, None] & mask_d[None, :],
            other=0.0,
        )
        mu = tl.load(mu_ptr + offs_d, mask=mask_d, other=0.0)
        std = tl.maximum(tl.load(std_ptr + offs_d, mask=mask_d, other=1.0), 1.0e-3)
        z = _tanh_tl((xv - mu[None, :]) / std[None, :])
        t2 = 2.0 * z * z - 1.0
        gp = grad_pre[:, None]
        g0 = tl.sum(gp, axis=0) * INV_SQRT_D
        g1 = tl.sum(z * gp, axis=0) * INV_SQRT_D
        g2 = tl.sum(t2 * gp, axis=0) * INV_SQRT_D
        tl.store(gw1_ptr + ((offs_d * H + hh) * 3), g0, mask=mask_d)
        tl.store(gw1_ptr + ((offs_d * H + hh) * 3 + 1), g1, mask=mask_d)
        tl.store(gw1_ptr + ((offs_d * H + hh) * 3 + 2), g2, mask=mask_d)


    @triton.jit
    def _cheby_pair_cross_logits_kernel(
        h_ptr,
        cross_w_ptr,
        logits_ptr,
        B: tl.constexpr,
        H: tl.constexpr,
        C: tl.constexpr,
        R: tl.constexpr,
        SCALE: tl.constexpr,
        BLOCK_B: tl.constexpr,
        BLOCK_R: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        b_block = tl.program_id(0)
        c_block = tl.program_id(1)
        offs_b = b_block * BLOCK_B + tl.arange(0, BLOCK_B)
        offs_c = c_block * BLOCK_C + tl.arange(0, BLOCK_C)
        offs_r = tl.arange(0, BLOCK_R)
        b_mask = offs_b < B
        c_mask = offs_c < C
        acc = tl.zeros((BLOCK_B, BLOCK_C), tl.float32)
        for start in range(0, R, BLOCK_R):
            rr = start + offs_r
            r_mask = rr < R
            h0_idx = rr * 2
            h1_idx = h0_idx + 1
            h0 = tl.load(h_ptr + offs_b[:, None] * H + h0_idx[None, :], mask=b_mask[:, None] & r_mask[None, :] & (h1_idx[None, :] < H), other=0.0)
            h1 = tl.load(h_ptr + offs_b[:, None] * H + h1_idx[None, :], mask=b_mask[:, None] & r_mask[None, :] & (h1_idx[None, :] < H), other=0.0)
            prod = h0 * h1
            cw = tl.load(cross_w_ptr + rr[:, None] * C + offs_c[None, :], mask=r_mask[:, None] & c_mask[None, :], other=0.0)
            acc += tl.dot(prod, cw, input_precision="tf32x3")
        old = tl.load(logits_ptr + offs_b[:, None] * C + offs_c[None, :], mask=b_mask[:, None] & c_mask[None, :], other=0.0)
        tl.store(logits_ptr + offs_b[:, None] * C + offs_c[None, :], old + acc * SCALE, mask=b_mask[:, None] & c_mask[None, :])


    @triton.jit
    def _cheby_pair_cross_w_grad_kernel(
        grad_logits_ptr,
        h_ptr,
        gw_cross_ptr,
        B: tl.constexpr,
        H: tl.constexpr,
        C: tl.constexpr,
        R: tl.constexpr,
        SCALE: tl.constexpr,
        BLOCK_B: tl.constexpr,
    ):
        rr = tl.program_id(0)
        cc = tl.program_id(1)
        offs = tl.arange(0, BLOCK_B)
        mask = offs < B
        h0_idx = rr * 2
        h1_idx = h0_idx + 1
        h0 = tl.load(h_ptr + offs * H + h0_idx, mask=mask & (h1_idx < H), other=0.0)
        h1 = tl.load(h_ptr + offs * H + h1_idx, mask=mask & (h1_idx < H), other=0.0)
        g = tl.load(grad_logits_ptr + offs * C + cc, mask=mask, other=0.0)
        val = tl.sum(h0 * h1 * g, axis=0) * SCALE
        tl.store(gw_cross_ptr + rr * C + cc, val, mask=rr < R)


    @triton.jit
    def _cheby_pair_cross_h_grad_kernel(
        grad_logits_ptr,
        h_ptr,
        cross_w_ptr,
        extra_grad_h_ptr,
        B: tl.constexpr,
        H: tl.constexpr,
        C: tl.constexpr,
        R: tl.constexpr,
        SCALE: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        b = tl.program_id(0)
        rr = tl.program_id(1)
        offs_c = tl.arange(0, BLOCK_C)
        c_mask = offs_c < C
        h0_idx = rr * 2
        h1_idx = h0_idx + 1
        g = tl.load(grad_logits_ptr + b * C + offs_c, mask=(b < B) & c_mask, other=0.0)
        cw = tl.load(cross_w_ptr + rr * C + offs_c, mask=(rr < R) & c_mask, other=0.0)
        coeff = tl.sum(g * cw, axis=0) * SCALE
        h0 = tl.load(h_ptr + b * H + h0_idx, mask=(b < B) & (h1_idx < H), other=0.0)
        h1 = tl.load(h_ptr + b * H + h1_idx, mask=(b < B) & (h1_idx < H), other=0.0)
        tl.store(extra_grad_h_ptr + b * H + h0_idx, coeff * h1, mask=(b < B) & (rr < R) & (h1_idx < H))
        tl.store(extra_grad_h_ptr + b * H + h1_idx, coeff * h0, mask=(b < B) & (rr < R) & (h1_idx < H))


    @triton.jit
    def _cheby_k4_forward_hidden_kernel(
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
            z2 = z * z
            t2 = 2.0 * z2 - 1.0
            t3 = 4.0 * z2 * z - 3.0 * z
            w10 = tl.load(w1_ptr + ((d[:, None] * H + offs_h[None, :]) * 4), mask=d_mask[:, None] & h_mask[None, :], other=0.0)
            w11 = tl.load(w1_ptr + ((d[:, None] * H + offs_h[None, :]) * 4 + 1), mask=d_mask[:, None] & h_mask[None, :], other=0.0)
            w12 = tl.load(w1_ptr + ((d[:, None] * H + offs_h[None, :]) * 4 + 2), mask=d_mask[:, None] & h_mask[None, :], other=0.0)
            w13 = tl.load(w1_ptr + ((d[:, None] * H + offs_h[None, :]) * 4 + 3), mask=d_mask[:, None] & h_mask[None, :], other=0.0)
            acc += tl.sum(w10, axis=0)[None, :] + tl.dot(z, w11, input_precision="tf32x3") + tl.dot(t2, w12, input_precision="tf32x3") + tl.dot(t3, w13, input_precision="tf32x3")
        h_val = _tanh_tl(acc * INV_SQRT_D)
        tl.store(h_ptr + offs_b[:, None] * H + offs_h[None, :], h_val, mask=b_mask[:, None] & h_mask[None, :])


    @triton.jit
    def _cheby_k4_forward_logits_kernel(
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
            h2 = h_val * h_val
            t2 = 2.0 * h2 - 1.0
            t3 = 4.0 * h2 * h_val - 3.0 * h_val
            w20 = tl.load(w2_ptr + ((hh[:, None] * C + offs_c[None, :]) * 4), mask=h_mask[:, None] & c_mask[None, :], other=0.0)
            w21 = tl.load(w2_ptr + ((hh[:, None] * C + offs_c[None, :]) * 4 + 1), mask=h_mask[:, None] & c_mask[None, :], other=0.0)
            w22 = tl.load(w2_ptr + ((hh[:, None] * C + offs_c[None, :]) * 4 + 2), mask=h_mask[:, None] & c_mask[None, :], other=0.0)
            w23 = tl.load(w2_ptr + ((hh[:, None] * C + offs_c[None, :]) * 4 + 3), mask=h_mask[:, None] & c_mask[None, :], other=0.0)
            acc += tl.sum(w20, axis=0)[None, :] + tl.dot(h_val, w21, input_precision="tf32x3") + tl.dot(t2, w22, input_precision="tf32x3") + tl.dot(t3, w23, input_precision="tf32x3")
        tl.store(logits_ptr + offs_b[:, None] * C + offs_c[None, :], acc * INV_SQRT_H, mask=b_mask[:, None] & c_mask[None, :])


    @triton.jit
    def _cheby_k4_w2_grad_kernel(
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
        h2 = h_val * h_val
        t2 = 2.0 * h2 - 1.0
        t3 = 4.0 * h2 * h_val - 3.0 * h_val
        base = (hh * C + cc) * 4
        tl.store(gw2_ptr + base, tl.sum(g, axis=0) * INV_SQRT_H)
        tl.store(gw2_ptr + base + 1, tl.sum(h_val * g, axis=0) * INV_SQRT_H)
        tl.store(gw2_ptr + base + 2, tl.sum(t2 * g, axis=0) * INV_SQRT_H)
        tl.store(gw2_ptr + base + 3, tl.sum(t3 * g, axis=0) * INV_SQRT_H)


    @triton.jit
    def _cheby_k4_w1_grad_blockd_kernel(
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
        h2 = h_val * h_val
        grad_h = tl.full((BLOCK_B,), 0.0, tl.float32)
        for cc in range(0, C):
            g = tl.load(grad_logits_ptr + offs_b * C + cc, mask=mask_b, other=0.0)
            base = (hh * C + cc) * 4
            w21 = tl.load(w2_ptr + base + 1)
            w22 = tl.load(w2_ptr + base + 2)
            w23 = tl.load(w2_ptr + base + 3)
            grad_h += g * (w21 + 4.0 * h_val * w22 + (12.0 * h2 - 3.0) * w23) * INV_SQRT_H
        grad_pre = grad_h * (1.0 - h_val * h_val)
        xv = tl.load(
            x_ptr + offs_b[:, None] * D + offs_d[None, :],
            mask=mask_b[:, None] & mask_d[None, :],
            other=0.0,
        )
        mu = tl.load(mu_ptr + offs_d, mask=mask_d, other=0.0)
        std = tl.maximum(tl.load(std_ptr + offs_d, mask=mask_d, other=1.0), 1.0e-3)
        z = _tanh_tl((xv - mu[None, :]) / std[None, :])
        z2 = z * z
        t2 = 2.0 * z2 - 1.0
        t3 = 4.0 * z2 * z - 3.0 * z
        gp = grad_pre[:, None]
        base = (offs_d * H + hh) * 4
        tl.store(gw1_ptr + base, tl.sum(gp, axis=0) * INV_SQRT_D, mask=mask_d)
        tl.store(gw1_ptr + base + 1, tl.sum(z * gp, axis=0) * INV_SQRT_D, mask=mask_d)
        tl.store(gw1_ptr + base + 2, tl.sum(t2 * gp, axis=0) * INV_SQRT_D, mask=mask_d)
        tl.store(gw1_ptr + base + 3, tl.sum(t3 * gp, axis=0) * INV_SQRT_D, mask=mask_d)


def _check_model(model, expected_k: int = 3) -> None:
    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available")
    if getattr(model.spec, "basis_name", "") != "chebyshev" or int(model.k) != int(expected_k):
        raise ValueError(f"fused_chebyshev_k3 requires PrimitiveKAN chebyshev k={expected_k}")
    if model.w1.dtype != torch.float32 or model.w2.dtype != torch.float32:
        raise ValueError("fused_chebyshev_k3 requires fp32 weights")
    if model.w1.device.type != "cuda" or model.w2.device.type != "cuda":
        raise ValueError("fused_chebyshev_k3 requires CUDA weights")


def _persistent_grad_buffer(model, attr_name: str, ref: torch.Tensor) -> torch.Tensor:
    buf = getattr(model, attr_name, None)
    if not isinstance(buf, torch.Tensor) or buf.shape != ref.shape or buf.device != ref.device or buf.dtype != ref.dtype:
        buf = torch.empty_like(ref)
        setattr(model, attr_name, buf)
    return buf


def _cross_rank(model) -> int:
    return int(max(0, min(int(getattr(model, "cheby_cross_rank", 0)), int(model.hidden_dim) // 2)))


def _cross_scale(rank: int) -> float:
    return 1.0 / math.sqrt(max(1, int(rank)))


def forward_matmul(model, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    _check_model(model, 3)
    if x.dtype != torch.float32 or x.device.type != "cuda":
        raise ValueError("fused_chebyshev_k3 forward_matmul requires fp32 CUDA input")
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
    _cheby_k3_forward_hidden_kernel[(triton.cdiv(batch, block_b), triton.cdiv(h_dim, block_h))](
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
    _cheby_k3_forward_logits_kernel[(triton.cdiv(batch, block_b), triton.cdiv(c_dim, block_c))](
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


def _backward_k3_impl(model, x: torch.Tensor, y: torch.Tensor, logits: torch.Tensor, h: torch.Tensor, *, use_grad_buffers: bool) -> torch.Tensor:
    _check_model(model, 3)
    if x.dtype != torch.float32 or x.device.type != "cuda":
        raise ValueError("fused_chebyshev_k3 backward requires fp32 CUDA input")
    x = x.contiguous()
    with torch.no_grad():
        loss = F.cross_entropy(logits, y)
        grad_logits = torch.softmax(logits, dim=1)
        grad_logits[torch.arange(int(y.numel()), device=y.device), y] -= 1.0
        grad_logits = (grad_logits / float(max(1, int(y.numel())))).contiguous()
        if use_grad_buffers:
            gw1 = _persistent_grad_buffer(model, "_cheby_k3_gw1_buffer", model.w1)
            gw2 = _persistent_grad_buffer(model, "_cheby_k3_gw2_buffer", model.w2)
        else:
            gw1 = torch.empty_like(model.w1)
            gw2 = torch.empty_like(model.w2)
        batch = int(x.shape[0])
        block_b = triton.next_power_of_2(batch)
        _cheby_k3_w2_grad_kernel[(int(model.hidden_dim), int(model.output_dim))](
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
        _cheby_k3_w1_grad_blockd_kernel[(triton.cdiv(int(model.input_dim), block_d), int(model.hidden_dim))](
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
        return loss.detach()


def backward(model, x: torch.Tensor, y: torch.Tensor, logits: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    return _backward_k3_impl(model, x, y, logits, h, use_grad_buffers=False)


def backward_gradbuf(model, x: torch.Tensor, y: torch.Tensor, logits: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    return _backward_k3_impl(model, x, y, logits, h, use_grad_buffers=True)


def forward_matmul_paircross(model, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    logits, h = forward_matmul(model, x)
    rank = _cross_rank(model)
    if rank <= 0:
        return logits, h
    block_b = 16
    block_r = max(16, triton.next_power_of_2(rank))
    block_c = max(16, triton.next_power_of_2(int(model.output_dim)))
    _cheby_pair_cross_logits_kernel[(triton.cdiv(int(x.shape[0]), block_b), triton.cdiv(int(model.output_dim), block_c))](
        h,
        model.cheby_cross_readout,
        logits,
        int(x.shape[0]),
        int(model.hidden_dim),
        int(model.output_dim),
        rank,
        _cross_scale(rank),
        BLOCK_B=block_b,
        BLOCK_R=block_r,
        BLOCK_C=block_c,
    )
    return logits, h


def backward_paircross_gradbuf(model, x: torch.Tensor, y: torch.Tensor, logits: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    _check_model(model, 3)
    if x.dtype != torch.float32 or x.device.type != "cuda":
        raise ValueError("fused_chebyshev_k3 backward_paircross_gradbuf requires fp32 CUDA input")
    x = x.contiguous()
    rank = _cross_rank(model)
    if rank <= 0:
        return backward_gradbuf(model, x, y, logits, h)
    with torch.no_grad():
        loss = F.cross_entropy(logits, y)
        grad_logits = torch.softmax(logits, dim=1)
        grad_logits[torch.arange(int(y.numel()), device=y.device), y] -= 1.0
        grad_logits = (grad_logits / float(max(1, int(y.numel())))).contiguous()
        gw1 = _persistent_grad_buffer(model, "_cheby_k3_gw1_buffer", model.w1)
        gw2 = _persistent_grad_buffer(model, "_cheby_k3_gw2_buffer", model.w2)
        gw_cross = _persistent_grad_buffer(model, "_cheby_pair_cross_gw_buffer", model.cheby_cross_readout)
        extra_grad_h = _persistent_grad_buffer(model, "_cheby_pair_cross_extra_h_buffer", h)
        extra_grad_h.zero_()
        batch = int(x.shape[0])
        block_b = triton.next_power_of_2(batch)
        _cheby_k3_w2_grad_kernel[(int(model.hidden_dim), int(model.output_dim))](
            grad_logits,
            h,
            gw2,
            batch,
            int(model.hidden_dim),
            int(model.output_dim),
            1.0 / math.sqrt(max(1, int(model.hidden_dim))),
            BLOCK_B=block_b,
        )
        _cheby_pair_cross_w_grad_kernel[(rank, int(model.output_dim))](
            grad_logits,
            h,
            gw_cross,
            batch,
            int(model.hidden_dim),
            int(model.output_dim),
            rank,
            _cross_scale(rank),
            BLOCK_B=block_b,
        )
        block_c = max(16, triton.next_power_of_2(int(model.output_dim)))
        _cheby_pair_cross_h_grad_kernel[(batch, rank)](
            grad_logits,
            h,
            model.cheby_cross_readout,
            extra_grad_h,
            batch,
            int(model.hidden_dim),
            int(model.output_dim),
            rank,
            _cross_scale(rank),
            BLOCK_C=block_c,
        )
        block_d = 32
        _cheby_k3_w1_grad_blockd_extra_kernel[(triton.cdiv(int(model.input_dim), block_d), int(model.hidden_dim))](
            x,
            model.mu,
            model.std,
            model.w2,
            grad_logits,
            extra_grad_h,
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
        model.cheby_cross_readout.grad = gw_cross
        return loss.detach()


def forward_matmul_k4(model, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    _check_model(model, 4)
    if x.dtype != torch.float32 or x.device.type != "cuda":
        raise ValueError("fused_chebyshev_k3 forward_matmul_k4 requires fp32 CUDA input")
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
    _cheby_k4_forward_hidden_kernel[(triton.cdiv(batch, block_b), triton.cdiv(h_dim, block_h))](
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
    _cheby_k4_forward_logits_kernel[(triton.cdiv(batch, block_b), triton.cdiv(c_dim, block_c))](
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


def backward_k4(model, x: torch.Tensor, y: torch.Tensor, logits: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    _check_model(model, 4)
    if x.dtype != torch.float32 or x.device.type != "cuda":
        raise ValueError("fused_chebyshev_k3 backward_k4 requires fp32 CUDA input")
    x = x.contiguous()
    with torch.no_grad():
        loss = F.cross_entropy(logits, y)
        grad_logits = torch.softmax(logits, dim=1)
        grad_logits[torch.arange(int(y.numel()), device=y.device), y] -= 1.0
        grad_logits = (grad_logits / float(max(1, int(y.numel())))).contiguous()
        gw1 = torch.empty_like(model.w1)
        gw2 = torch.empty_like(model.w2)
        batch = int(x.shape[0])
        block_b = triton.next_power_of_2(batch)
        _cheby_k4_w2_grad_kernel[(int(model.hidden_dim), int(model.output_dim))](
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
        _cheby_k4_w1_grad_blockd_kernel[(triton.cdiv(int(model.input_dim), block_d), int(model.hidden_dim))](
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
        return loss.detach()
