"""Triton Fourier K2/K3 two-layer PrimitiveKAN kernels.

This is a narrow v12.8.3 repair attempt for the classic-family branch.  It is
only valid for PrimitiveKAN with basis_name=fourier_lowfreq, fp32 CUDA inputs,
and fixed centers/scales.  The kernels intentionally recompute the Fourier
basis from x instead of materializing B x D x K workspaces.
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
    def _fourier_k2_forward_kernel(
        x_ptr,
        mu_ptr,
        std_ptr,
        w1_ptr,
        w2_ptr,
        logits_ptr,
        h_ptr,
        D: tl.constexpr,
        H: tl.constexpr,
        C: tl.constexpr,
        INV_SQRT_D: tl.constexpr,
        INV_SQRT_H: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        b = tl.program_id(0)
        offs = tl.arange(0, BLOCK_D)
        for hh in range(0, H):
            acc = tl.full((), 0.0, tl.float32)
            for start in range(0, D, BLOCK_D):
                d = start + offs
                mask = d < D
                xv = tl.load(x_ptr + b * D + d, mask=mask, other=0.0)
                mu = tl.load(mu_ptr + d, mask=mask, other=0.0)
                std = tl.maximum(tl.load(std_ptr + d, mask=mask, other=1.0), 1.0e-3)
                z = _tanh_tl((xv - mu) / std)
                sin_z = tl.sin(3.141592653589793 * z)
                w10 = tl.load(w1_ptr + ((d * H + hh) * 2), mask=mask, other=0.0)
                w11 = tl.load(w1_ptr + ((d * H + hh) * 2 + 1), mask=mask, other=0.0)
                acc += tl.sum(z * w10 + sin_z * w11, axis=0)
            h_val = _tanh_tl(acc * 0.7071067811865476 * INV_SQRT_D)
            tl.store(h_ptr + b * H + hh, h_val)
        for cc in range(0, C):
            acc2 = tl.full((), 0.0, tl.float32)
            for hh in range(0, H):
                h_val = tl.load(h_ptr + b * H + hh)
                sin_h = tl.sin(3.141592653589793 * h_val)
                w20 = tl.load(w2_ptr + ((hh * C + cc) * 2))
                w21 = tl.load(w2_ptr + ((hh * C + cc) * 2 + 1))
                acc2 += h_val * w20 + sin_h * w21
            tl.store(logits_ptr + b * C + cc, acc2 * 0.7071067811865476 * INV_SQRT_H)


    @triton.jit
    def _fourier_k2_forward_blockh_hidden_kernel(
        x_ptr,
        mu_ptr,
        std_ptr,
        w1_ptr,
        h_ptr,
        D: tl.constexpr,
        H: tl.constexpr,
        INV_SQRT_D: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        b = tl.program_id(0)
        h_block = tl.program_id(1)
        offs_d = tl.arange(0, BLOCK_D)
        offs_h = h_block * BLOCK_H + tl.arange(0, BLOCK_H)
        h_mask = offs_h < H
        acc = tl.zeros((BLOCK_H,), tl.float32)
        for start in range(0, D, BLOCK_D):
            d = start + offs_d
            d_mask = d < D
            xv = tl.load(x_ptr + b * D + d, mask=d_mask, other=0.0)
            mu = tl.load(mu_ptr + d, mask=d_mask, other=0.0)
            std = tl.maximum(tl.load(std_ptr + d, mask=d_mask, other=1.0), 1.0e-3)
            z = _tanh_tl((xv - mu) / std)
            sin_z = tl.sin(3.141592653589793 * z)
            mask = d_mask[:, None] & h_mask[None, :]
            w10 = tl.load(w1_ptr + ((d[:, None] * H + offs_h[None, :]) * 2), mask=mask, other=0.0)
            w11 = tl.load(w1_ptr + ((d[:, None] * H + offs_h[None, :]) * 2 + 1), mask=mask, other=0.0)
            acc += tl.sum(z[:, None] * w10 + sin_z[:, None] * w11, axis=0)
        h_val = _tanh_tl(acc * 0.7071067811865476 * INV_SQRT_D)
        tl.store(h_ptr + b * H + offs_h, h_val, mask=h_mask)


    @triton.jit
    def _fourier_k2_forward_blockh_logits_kernel(
        h_ptr,
        w2_ptr,
        logits_ptr,
        H: tl.constexpr,
        C: tl.constexpr,
        INV_SQRT_H: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        b = tl.program_id(0)
        cc = tl.program_id(1)
        offs_h = tl.arange(0, BLOCK_H)
        acc = tl.full((), 0.0, tl.float32)
        for start in range(0, H, BLOCK_H):
            hh = start + offs_h
            mask = hh < H
            h_val = tl.load(h_ptr + b * H + hh, mask=mask, other=0.0)
            sin_h = tl.sin(3.141592653589793 * h_val)
            w20 = tl.load(w2_ptr + ((hh * C + cc) * 2), mask=mask, other=0.0)
            w21 = tl.load(w2_ptr + ((hh * C + cc) * 2 + 1), mask=mask, other=0.0)
            acc += tl.sum(h_val * w20 + sin_h * w21, axis=0)
        tl.store(logits_ptr + b * C + cc, acc * 0.7071067811865476 * INV_SQRT_H)


    @triton.jit
    def _fourier_k2_forward_matmul_hidden_kernel(
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
            sin_z = tl.sin(3.141592653589793 * z)
            w10 = tl.load(w1_ptr + ((d[:, None] * H + offs_h[None, :]) * 2), mask=d_mask[:, None] & h_mask[None, :], other=0.0)
            w11 = tl.load(w1_ptr + ((d[:, None] * H + offs_h[None, :]) * 2 + 1), mask=d_mask[:, None] & h_mask[None, :], other=0.0)
            acc += tl.dot(z, w10, input_precision="tf32x3") + tl.dot(sin_z, w11, input_precision="tf32x3")
        h_val = _tanh_tl(acc * 0.7071067811865476 * INV_SQRT_D)
        tl.store(h_ptr + offs_b[:, None] * H + offs_h[None, :], h_val, mask=b_mask[:, None] & h_mask[None, :])


    @triton.jit
    def _fourier_k2_forward_matmul_logits_kernel(
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
            sin_h = tl.sin(3.141592653589793 * h_val)
            w20 = tl.load(w2_ptr + ((hh[:, None] * C + offs_c[None, :]) * 2), mask=h_mask[:, None] & c_mask[None, :], other=0.0)
            w21 = tl.load(w2_ptr + ((hh[:, None] * C + offs_c[None, :]) * 2 + 1), mask=h_mask[:, None] & c_mask[None, :], other=0.0)
            acc += tl.dot(h_val, w20, input_precision="tf32x3") + tl.dot(sin_h, w21, input_precision="tf32x3")
        tl.store(logits_ptr + offs_b[:, None] * C + offs_c[None, :], acc * 0.7071067811865476 * INV_SQRT_H, mask=b_mask[:, None] & c_mask[None, :])


    @triton.jit
    def _fourier_k3_forward_matmul_hidden_kernel(
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
            sin_z = tl.sin(3.141592653589793 * z)
            cos_z = tl.cos(3.141592653589793 * z)
            w10 = tl.load(w1_ptr + ((d[:, None] * H + offs_h[None, :]) * 3), mask=d_mask[:, None] & h_mask[None, :], other=0.0)
            w11 = tl.load(w1_ptr + ((d[:, None] * H + offs_h[None, :]) * 3 + 1), mask=d_mask[:, None] & h_mask[None, :], other=0.0)
            w12 = tl.load(w1_ptr + ((d[:, None] * H + offs_h[None, :]) * 3 + 2), mask=d_mask[:, None] & h_mask[None, :], other=0.0)
            acc += (
                tl.dot(z, w10, input_precision="tf32x3")
                + tl.dot(sin_z, w11, input_precision="tf32x3")
                + tl.dot(cos_z, w12, input_precision="tf32x3")
            )
        h_val = _tanh_tl(acc * 0.5773502691896258 * INV_SQRT_D)
        tl.store(h_ptr + offs_b[:, None] * H + offs_h[None, :], h_val, mask=b_mask[:, None] & h_mask[None, :])


    @triton.jit
    def _fourier_k3_forward_matmul_logits_kernel(
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
            sin_h = tl.sin(3.141592653589793 * h_val)
            cos_h = tl.cos(3.141592653589793 * h_val)
            w20 = tl.load(w2_ptr + ((hh[:, None] * C + offs_c[None, :]) * 3), mask=h_mask[:, None] & c_mask[None, :], other=0.0)
            w21 = tl.load(w2_ptr + ((hh[:, None] * C + offs_c[None, :]) * 3 + 1), mask=h_mask[:, None] & c_mask[None, :], other=0.0)
            w22 = tl.load(w2_ptr + ((hh[:, None] * C + offs_c[None, :]) * 3 + 2), mask=h_mask[:, None] & c_mask[None, :], other=0.0)
            acc += (
                tl.dot(h_val, w20, input_precision="tf32x3")
                + tl.dot(sin_h, w21, input_precision="tf32x3")
                + tl.dot(cos_h, w22, input_precision="tf32x3")
            )
        tl.store(logits_ptr + offs_b[:, None] * C + offs_c[None, :], acc * 0.5773502691896258 * INV_SQRT_H, mask=b_mask[:, None] & c_mask[None, :])


    @triton.jit
    def _fourier_k4_forward_matmul_hidden_kernel(
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
            w10 = tl.load(w1_ptr + ((d[:, None] * H + offs_h[None, :]) * 4), mask=d_mask[:, None] & h_mask[None, :], other=0.0)
            w11 = tl.load(w1_ptr + ((d[:, None] * H + offs_h[None, :]) * 4 + 1), mask=d_mask[:, None] & h_mask[None, :], other=0.0)
            w12 = tl.load(w1_ptr + ((d[:, None] * H + offs_h[None, :]) * 4 + 2), mask=d_mask[:, None] & h_mask[None, :], other=0.0)
            w13 = tl.load(w1_ptr + ((d[:, None] * H + offs_h[None, :]) * 4 + 3), mask=d_mask[:, None] & h_mask[None, :], other=0.0)
            acc += (
                tl.dot(z, w10, input_precision="tf32x3")
                + tl.dot(sin1, w11, input_precision="tf32x3")
                + tl.dot(cos1, w12, input_precision="tf32x3")
                + tl.dot(sin2, w13, input_precision="tf32x3")
            )
        h_val = _tanh_tl(acc * 0.5 * INV_SQRT_D)
        tl.store(h_ptr + offs_b[:, None] * H + offs_h[None, :], h_val, mask=b_mask[:, None] & h_mask[None, :])


    @triton.jit
    def _fourier_k4_forward_matmul_logits_kernel(
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
            w20 = tl.load(w2_ptr + ((hh[:, None] * C + offs_c[None, :]) * 4), mask=h_mask[:, None] & c_mask[None, :], other=0.0)
            w21 = tl.load(w2_ptr + ((hh[:, None] * C + offs_c[None, :]) * 4 + 1), mask=h_mask[:, None] & c_mask[None, :], other=0.0)
            w22 = tl.load(w2_ptr + ((hh[:, None] * C + offs_c[None, :]) * 4 + 2), mask=h_mask[:, None] & c_mask[None, :], other=0.0)
            w23 = tl.load(w2_ptr + ((hh[:, None] * C + offs_c[None, :]) * 4 + 3), mask=h_mask[:, None] & c_mask[None, :], other=0.0)
            acc += (
                tl.dot(h_val, w20, input_precision="tf32x3")
                + tl.dot(sin1, w21, input_precision="tf32x3")
                + tl.dot(cos1, w22, input_precision="tf32x3")
                + tl.dot(sin2, w23, input_precision="tf32x3")
            )
        tl.store(logits_ptr + offs_b[:, None] * C + offs_c[None, :], acc * 0.5 * INV_SQRT_H, mask=b_mask[:, None] & c_mask[None, :])


    @triton.jit
    def _fourier_linear_residual_add_kernel(
        x_ptr,
        mu_ptr,
        std_ptr,
        direct_ptr,
        logits_ptr,
        B: tl.constexpr,
        D: tl.constexpr,
        C: tl.constexpr,
        INV_SQRT_D: tl.constexpr,
        BLOCK_B: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        b_block = tl.program_id(0)
        c_block = tl.program_id(1)
        offs_b = b_block * BLOCK_B + tl.arange(0, BLOCK_B)
        offs_c = c_block * BLOCK_C + tl.arange(0, BLOCK_C)
        offs_d = tl.arange(0, BLOCK_D)
        b_mask = offs_b < B
        c_mask = offs_c < C
        acc = tl.zeros((BLOCK_B, BLOCK_C), tl.float32)
        for start in range(0, D, BLOCK_D):
            d = start + offs_d
            d_mask = d < D
            xv = tl.load(x_ptr + offs_b[:, None] * D + d[None, :], mask=b_mask[:, None] & d_mask[None, :], other=0.0)
            mu = tl.load(mu_ptr + d, mask=d_mask, other=0.0)
            std = tl.maximum(tl.load(std_ptr + d, mask=d_mask, other=1.0), 1.0e-3)
            z = _tanh_tl((xv - mu[None, :]) / std[None, :])
            w = tl.load(direct_ptr + d[:, None] * C + offs_c[None, :], mask=d_mask[:, None] & c_mask[None, :], other=0.0)
            acc += tl.dot(z, w, input_precision="tf32x3")
        old = tl.load(logits_ptr + offs_b[:, None] * C + offs_c[None, :], mask=b_mask[:, None] & c_mask[None, :], other=0.0)
        tl.store(logits_ptr + offs_b[:, None] * C + offs_c[None, :], old + acc * INV_SQRT_D, mask=b_mask[:, None] & c_mask[None, :])


    @triton.jit
    def _fourier_k2_w2_grad_kernel(
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
        g0 = tl.sum(h_val * g, axis=0) * 0.7071067811865476 * INV_SQRT_H
        g1 = tl.sum(tl.sin(3.141592653589793 * h_val) * g, axis=0) * 0.7071067811865476 * INV_SQRT_H
        tl.store(gw2_ptr + ((hh * C + cc) * 2), g0)
        tl.store(gw2_ptr + ((hh * C + cc) * 2 + 1), g1)


    @triton.jit
    def _fourier_k3_w2_grad_kernel(
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
        g0 = tl.sum(h_val * g, axis=0) * 0.5773502691896258 * INV_SQRT_H
        g1 = tl.sum(tl.sin(3.141592653589793 * h_val) * g, axis=0) * 0.5773502691896258 * INV_SQRT_H
        g2 = tl.sum(tl.cos(3.141592653589793 * h_val) * g, axis=0) * 0.5773502691896258 * INV_SQRT_H
        tl.store(gw2_ptr + ((hh * C + cc) * 3), g0)
        tl.store(gw2_ptr + ((hh * C + cc) * 3 + 1), g1)
        tl.store(gw2_ptr + ((hh * C + cc) * 3 + 2), g2)


    @triton.jit
    def _fourier_k4_w2_grad_kernel(
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
        g0 = tl.sum(h_val * g, axis=0) * 0.5 * INV_SQRT_H
        g1 = tl.sum(tl.sin(3.141592653589793 * h_val) * g, axis=0) * 0.5 * INV_SQRT_H
        g2 = tl.sum(tl.cos(3.141592653589793 * h_val) * g, axis=0) * 0.5 * INV_SQRT_H
        g3 = tl.sum(tl.sin(6.283185307179586 * h_val) * g, axis=0) * 0.5 * INV_SQRT_H
        tl.store(gw2_ptr + ((hh * C + cc) * 4), g0)
        tl.store(gw2_ptr + ((hh * C + cc) * 4 + 1), g1)
        tl.store(gw2_ptr + ((hh * C + cc) * 4 + 2), g2)
        tl.store(gw2_ptr + ((hh * C + cc) * 4 + 3), g3)


    @triton.jit
    def _fourier_k2_w1_grad_kernel(
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
    ):
        d = tl.program_id(0)
        hh = tl.program_id(1)
        offs = tl.arange(0, BLOCK_B)
        mask = offs < B
        xv = tl.load(x_ptr + offs * D + d, mask=mask, other=0.0)
        mu = tl.load(mu_ptr + d)
        std = tl.maximum(tl.load(std_ptr + d), 1.0e-3)
        z = _tanh_tl((xv - mu) / std)
        h_val = tl.load(h_ptr + offs * H + hh, mask=mask, other=0.0)
        grad_h = tl.full((BLOCK_B,), 0.0, tl.float32)
        cos_h = tl.cos(3.141592653589793 * h_val)
        for cc in range(0, C):
            g = tl.load(grad_logits_ptr + offs * C + cc, mask=mask, other=0.0)
            w20 = tl.load(w2_ptr + ((hh * C + cc) * 2))
            w21 = tl.load(w2_ptr + ((hh * C + cc) * 2 + 1))
            grad_h += g * (w20 + w21 * 3.141592653589793 * cos_h) * 0.7071067811865476 * INV_SQRT_H
        grad_pre = grad_h * (1.0 - h_val * h_val)
        g0 = tl.sum(z * grad_pre, axis=0) * 0.7071067811865476 * INV_SQRT_D
        g1 = tl.sum(tl.sin(3.141592653589793 * z) * grad_pre, axis=0) * 0.7071067811865476 * INV_SQRT_D
        tl.store(gw1_ptr + ((d * H + hh) * 2), g0)
        tl.store(gw1_ptr + ((d * H + hh) * 2 + 1), g1)


    @triton.jit
    def _fourier_k2_w1_grad_blockd_kernel(
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
        cos_h = tl.cos(3.141592653589793 * h_val)
        for cc in range(0, C):
            g = tl.load(grad_logits_ptr + offs_b * C + cc, mask=mask_b, other=0.0)
            w20 = tl.load(w2_ptr + ((hh * C + cc) * 2))
            w21 = tl.load(w2_ptr + ((hh * C + cc) * 2 + 1))
            grad_h += g * (w20 + w21 * 3.141592653589793 * cos_h) * 0.7071067811865476 * INV_SQRT_H
        grad_pre = grad_h * (1.0 - h_val * h_val)

        xv = tl.load(
            x_ptr + offs_b[:, None] * D + offs_d[None, :],
            mask=mask_b[:, None] & mask_d[None, :],
            other=0.0,
        )
        mu = tl.load(mu_ptr + offs_d, mask=mask_d, other=0.0)
        std = tl.maximum(tl.load(std_ptr + offs_d, mask=mask_d, other=1.0), 1.0e-3)
        z = _tanh_tl((xv - mu[None, :]) / std[None, :])
        grad_pre_2d = grad_pre[:, None]
        g0 = tl.sum(z * grad_pre_2d, axis=0) * 0.7071067811865476 * INV_SQRT_D
        g1 = tl.sum(tl.sin(3.141592653589793 * z) * grad_pre_2d, axis=0) * 0.7071067811865476 * INV_SQRT_D
        tl.store(gw1_ptr + ((offs_d * H + hh) * 2), g0, mask=mask_d)
        tl.store(gw1_ptr + ((offs_d * H + hh) * 2 + 1), g1, mask=mask_d)


    @triton.jit
    def _fourier_k3_w1_grad_blockd_kernel(
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
        sin_h = tl.sin(3.141592653589793 * h_val)
        cos_h = tl.cos(3.141592653589793 * h_val)
        for cc in range(0, C):
            g = tl.load(grad_logits_ptr + offs_b * C + cc, mask=mask_b, other=0.0)
            w20 = tl.load(w2_ptr + ((hh * C + cc) * 3))
            w21 = tl.load(w2_ptr + ((hh * C + cc) * 3 + 1))
            w22 = tl.load(w2_ptr + ((hh * C + cc) * 3 + 2))
            grad_h += g * (w20 + w21 * 3.141592653589793 * cos_h - w22 * 3.141592653589793 * sin_h) * 0.5773502691896258 * INV_SQRT_H
        grad_pre = grad_h * (1.0 - h_val * h_val)

        xv = tl.load(
            x_ptr + offs_b[:, None] * D + offs_d[None, :],
            mask=mask_b[:, None] & mask_d[None, :],
            other=0.0,
        )
        mu = tl.load(mu_ptr + offs_d, mask=mask_d, other=0.0)
        std = tl.maximum(tl.load(std_ptr + offs_d, mask=mask_d, other=1.0), 1.0e-3)
        z = _tanh_tl((xv - mu[None, :]) / std[None, :])
        grad_pre_2d = grad_pre[:, None]
        g0 = tl.sum(z * grad_pre_2d, axis=0) * 0.5773502691896258 * INV_SQRT_D
        g1 = tl.sum(tl.sin(3.141592653589793 * z) * grad_pre_2d, axis=0) * 0.5773502691896258 * INV_SQRT_D
        g2 = tl.sum(tl.cos(3.141592653589793 * z) * grad_pre_2d, axis=0) * 0.5773502691896258 * INV_SQRT_D
        tl.store(gw1_ptr + ((offs_d * H + hh) * 3), g0, mask=mask_d)
        tl.store(gw1_ptr + ((offs_d * H + hh) * 3 + 1), g1, mask=mask_d)
        tl.store(gw1_ptr + ((offs_d * H + hh) * 3 + 2), g2, mask=mask_d)


    @triton.jit
    def _fourier_k4_w1_grad_blockd_kernel(
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
        sin1 = tl.sin(3.141592653589793 * h_val)
        cos1 = tl.cos(3.141592653589793 * h_val)
        cos2 = tl.cos(6.283185307179586 * h_val)
        for cc in range(0, C):
            g = tl.load(grad_logits_ptr + offs_b * C + cc, mask=mask_b, other=0.0)
            w20 = tl.load(w2_ptr + ((hh * C + cc) * 4))
            w21 = tl.load(w2_ptr + ((hh * C + cc) * 4 + 1))
            w22 = tl.load(w2_ptr + ((hh * C + cc) * 4 + 2))
            w23 = tl.load(w2_ptr + ((hh * C + cc) * 4 + 3))
            grad_h += g * (w20 + w21 * 3.141592653589793 * cos1 - w22 * 3.141592653589793 * sin1 + w23 * 6.283185307179586 * cos2) * 0.5 * INV_SQRT_H
        grad_pre = grad_h * (1.0 - h_val * h_val)

        xv = tl.load(
            x_ptr + offs_b[:, None] * D + offs_d[None, :],
            mask=mask_b[:, None] & mask_d[None, :],
            other=0.0,
        )
        mu = tl.load(mu_ptr + offs_d, mask=mask_d, other=0.0)
        std = tl.maximum(tl.load(std_ptr + offs_d, mask=mask_d, other=1.0), 1.0e-3)
        z = _tanh_tl((xv - mu[None, :]) / std[None, :])
        grad_pre_2d = grad_pre[:, None]
        g0 = tl.sum(z * grad_pre_2d, axis=0) * 0.5 * INV_SQRT_D
        g1 = tl.sum(tl.sin(3.141592653589793 * z) * grad_pre_2d, axis=0) * 0.5 * INV_SQRT_D
        g2 = tl.sum(tl.cos(3.141592653589793 * z) * grad_pre_2d, axis=0) * 0.5 * INV_SQRT_D
        g3 = tl.sum(tl.sin(6.283185307179586 * z) * grad_pre_2d, axis=0) * 0.5 * INV_SQRT_D
        tl.store(gw1_ptr + ((offs_d * H + hh) * 4), g0, mask=mask_d)
        tl.store(gw1_ptr + ((offs_d * H + hh) * 4 + 1), g1, mask=mask_d)
        tl.store(gw1_ptr + ((offs_d * H + hh) * 4 + 2), g2, mask=mask_d)
        tl.store(gw1_ptr + ((offs_d * H + hh) * 4 + 3), g3, mask=mask_d)


    @triton.jit
    def _fourier_linear_residual_grad_kernel(
        x_ptr,
        mu_ptr,
        std_ptr,
        grad_logits_ptr,
        gdirect_ptr,
        B: tl.constexpr,
        D: tl.constexpr,
        C: tl.constexpr,
        INV_SQRT_D: tl.constexpr,
        BLOCK_B: tl.constexpr,
    ):
        d = tl.program_id(0)
        cc = tl.program_id(1)
        offs = tl.arange(0, BLOCK_B)
        mask = offs < B
        xv = tl.load(x_ptr + offs * D + d, mask=mask, other=0.0)
        mu = tl.load(mu_ptr + d)
        std = tl.maximum(tl.load(std_ptr + d), 1.0e-3)
        z = _tanh_tl((xv - mu) / std)
        g = tl.load(grad_logits_ptr + offs * C + cc, mask=mask, other=0.0)
        gd = tl.sum(z * g, axis=0) * INV_SQRT_D
        tl.store(gdirect_ptr + d * C + cc, gd)


def _check_model(model) -> None:
    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available")
    if str(getattr(model.spec, "basis_name", "")) != "fourier_lowfreq" or int(model.k) != 2:
        raise ValueError("fused_fourier_k2 only supports PrimitiveKAN Fourier K2")
    if model.w1.dtype != torch.float32 or model.w2.dtype != torch.float32:
        raise ValueError("fused_fourier_k2 currently supports fp32 weights only")


def _check_model_k3(model) -> None:
    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available")
    if str(getattr(model.spec, "basis_name", "")) != "fourier_lowfreq" or int(model.k) != 3:
        raise ValueError("fused_fourier_k2 K3 path only supports PrimitiveKAN Fourier K3")
    if model.w1.dtype != torch.float32 or model.w2.dtype != torch.float32:
        raise ValueError("fused_fourier_k2 K3 path currently supports fp32 weights only")


def _check_model_k4(model) -> None:
    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available")
    if str(getattr(model.spec, "basis_name", "")) != "fourier_lowfreq" or int(model.k) != 4:
        raise ValueError("fused_fourier_k2 K4 path only supports PrimitiveKAN Fourier K4")
    if model.w1.dtype != torch.float32 or model.w2.dtype != torch.float32:
        raise ValueError("fused_fourier_k2 K4 path currently supports fp32 weights only")


def _check_model_k4_linearres(model) -> None:
    _check_model_k4(model)
    if not hasattr(model, "linear_readout"):
        raise ValueError("fused_fourier_k2 K4 linear residual path requires model.linear_readout")
    if model.linear_readout.dtype != torch.float32:
        raise ValueError("fused_fourier_k2 K4 linear residual path currently supports fp32 direct weights only")


def forward(model, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    _check_model(model)
    if x.dtype != torch.float32 or x.device.type != "cuda":
        raise ValueError("fused_fourier_k2 forward requires fp32 CUDA input")
    x = x.contiguous()
    batch = int(x.shape[0])
    logits = torch.empty((batch, int(model.output_dim)), device=x.device, dtype=torch.float32)
    h = torch.empty((batch, int(model.hidden_dim)), device=x.device, dtype=torch.float32)
    block_d = 256
    _fourier_k2_forward_kernel[(batch,)](
        x,
        model.mu,
        model.std,
        model.w1,
        model.w2,
        logits,
        h,
        int(model.input_dim),
        int(model.hidden_dim),
        int(model.output_dim),
        1.0 / math.sqrt(max(1, int(model.input_dim))),
        1.0 / math.sqrt(max(1, int(model.hidden_dim))),
        BLOCK_D=block_d,
    )
    return logits, h


def forward_blockh(model, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    _check_model(model)
    if x.dtype != torch.float32 or x.device.type != "cuda":
        raise ValueError("fused_fourier_k2 forward_blockh requires fp32 CUDA input")
    x = x.contiguous()
    batch = int(x.shape[0])
    h_dim = int(model.hidden_dim)
    logits = torch.empty((batch, int(model.output_dim)), device=x.device, dtype=torch.float32)
    h = torch.empty((batch, h_dim), device=x.device, dtype=torch.float32)
    block_d = 64
    block_h = 16
    _fourier_k2_forward_blockh_hidden_kernel[(batch, triton.cdiv(h_dim, block_h))](
        x,
        model.mu,
        model.std,
        model.w1,
        h,
        int(model.input_dim),
        h_dim,
        1.0 / math.sqrt(max(1, int(model.input_dim))),
        BLOCK_D=block_d,
        BLOCK_H=block_h,
    )
    _fourier_k2_forward_blockh_logits_kernel[(batch, int(model.output_dim))](
        h,
        model.w2,
        logits,
        h_dim,
        int(model.output_dim),
        1.0 / math.sqrt(max(1, h_dim)),
        BLOCK_H=block_h,
    )
    return logits, h


def forward_matmul(model, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    _check_model(model)
    if x.dtype != torch.float32 or x.device.type != "cuda":
        raise ValueError("fused_fourier_k2 forward_matmul requires fp32 CUDA input")
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
    _fourier_k2_forward_matmul_hidden_kernel[(triton.cdiv(batch, block_b), triton.cdiv(h_dim, block_h))](
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
    _fourier_k2_forward_matmul_logits_kernel[(triton.cdiv(batch, block_b), triton.cdiv(c_dim, block_c))](
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


def forward_matmul_k3(model, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    _check_model_k3(model)
    if x.dtype != torch.float32 or x.device.type != "cuda":
        raise ValueError("fused_fourier_k2 forward_matmul_k3 requires fp32 CUDA input")
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
    _fourier_k3_forward_matmul_hidden_kernel[(triton.cdiv(batch, block_b), triton.cdiv(h_dim, block_h))](
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
    _fourier_k3_forward_matmul_logits_kernel[(triton.cdiv(batch, block_b), triton.cdiv(c_dim, block_c))](
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


def forward_matmul_k4(model, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    _check_model_k4(model)
    if x.dtype != torch.float32 or x.device.type != "cuda":
        raise ValueError("fused_fourier_k2 forward_matmul_k4 requires fp32 CUDA input")
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
    _fourier_k4_forward_matmul_hidden_kernel[(triton.cdiv(batch, block_b), triton.cdiv(h_dim, block_h))](
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
    _fourier_k4_forward_matmul_logits_kernel[(triton.cdiv(batch, block_b), triton.cdiv(c_dim, block_c))](
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


def forward_matmul_k4_linearres(model, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    _check_model_k4_linearres(model)
    if x.dtype != torch.float32 or x.device.type != "cuda":
        raise ValueError("fused_fourier_k2 forward_matmul_k4_linearres requires fp32 CUDA input")
    x = x.contiguous()
    logits, h = forward_matmul_k4(model, x)
    batch = int(x.shape[0])
    c_dim = int(model.output_dim)
    block_b = 16
    block_d = 64
    block_c = max(16, triton.next_power_of_2(c_dim))
    _fourier_linear_residual_add_kernel[(triton.cdiv(batch, block_b), triton.cdiv(c_dim, block_c))](
        x,
        model.mu,
        model.std,
        model.linear_readout,
        logits,
        batch,
        int(model.input_dim),
        c_dim,
        1.0 / math.sqrt(max(1, int(model.input_dim))),
        BLOCK_B=block_b,
        BLOCK_D=block_d,
        BLOCK_C=block_c,
    )
    return logits, h


def backward_from_grad_logits(model, x: torch.Tensor, grad_logits: torch.Tensor, logits: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    _check_model(model)
    if x.dtype != torch.float32 or x.device.type != "cuda":
        raise ValueError("fused_fourier_k2 backward_from_grad_logits requires fp32 CUDA input")
    x = x.contiguous()
    with torch.no_grad():
        grad_logits = grad_logits.to(device=logits.device, dtype=logits.dtype).contiguous()
        gw1 = torch.empty_like(model.w1)
        gw2 = torch.empty_like(model.w2)
        batch = int(x.shape[0])
        block_b = triton.next_power_of_2(batch)
        _fourier_k2_w2_grad_kernel[(int(model.hidden_dim), int(model.output_dim))](
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
        _fourier_k2_w1_grad_blockd_kernel[(triton.cdiv(int(model.input_dim), block_d), int(model.hidden_dim))](
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


def backward(model, x: torch.Tensor, y: torch.Tensor, logits: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    _check_model(model)
    if x.dtype != torch.float32 or x.device.type != "cuda":
        raise ValueError("fused_fourier_k2 backward requires fp32 CUDA input")
    with torch.no_grad():
        loss = F.cross_entropy(logits, y)
        grad_logits = torch.softmax(logits, dim=1)
        grad_logits[torch.arange(int(y.numel()), device=y.device), y] -= 1.0
        grad_logits = (grad_logits / float(max(1, int(y.numel())))).contiguous()
    backward_from_grad_logits(model, x, grad_logits, logits, h)
    return loss.detach()


def backward_k3(model, x: torch.Tensor, y: torch.Tensor, logits: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    _check_model_k3(model)
    if x.dtype != torch.float32 or x.device.type != "cuda":
        raise ValueError("fused_fourier_k2 backward_k3 requires fp32 CUDA input")
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
        _fourier_k3_w2_grad_kernel[(int(model.hidden_dim), int(model.output_dim))](
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
        _fourier_k3_w1_grad_blockd_kernel[(triton.cdiv(int(model.input_dim), block_d), int(model.hidden_dim))](
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


def backward_k4(model, x: torch.Tensor, y: torch.Tensor, logits: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    _check_model_k4(model)
    if x.dtype != torch.float32 or x.device.type != "cuda":
        raise ValueError("fused_fourier_k2 backward_k4 requires fp32 CUDA input")
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
        _fourier_k4_w2_grad_kernel[(int(model.hidden_dim), int(model.output_dim))](
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
        _fourier_k4_w1_grad_blockd_kernel[(triton.cdiv(int(model.input_dim), block_d), int(model.hidden_dim))](
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


def backward_k4_linearres(model, x: torch.Tensor, y: torch.Tensor, logits: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    _check_model_k4_linearres(model)
    if x.dtype != torch.float32 or x.device.type != "cuda":
        raise ValueError("fused_fourier_k2 backward_k4_linearres requires fp32 CUDA input")
    x = x.contiguous()
    with torch.no_grad():
        loss = F.cross_entropy(logits, y)
        grad_logits = torch.softmax(logits, dim=1)
        grad_logits[torch.arange(int(y.numel()), device=y.device), y] -= 1.0
        grad_logits = (grad_logits / float(max(1, int(y.numel())))).contiguous()
        gw1 = torch.empty_like(model.w1)
        gw2 = torch.empty_like(model.w2)
        gdirect = torch.empty_like(model.linear_readout)
        batch = int(x.shape[0])
        block_b = triton.next_power_of_2(batch)
        _fourier_k4_w2_grad_kernel[(int(model.hidden_dim), int(model.output_dim))](
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
        _fourier_k4_w1_grad_blockd_kernel[(triton.cdiv(int(model.input_dim), block_d), int(model.hidden_dim))](
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
        _fourier_linear_residual_grad_kernel[(int(model.input_dim), int(model.output_dim))](
            x,
            model.mu,
            model.std,
            grad_logits,
            gdirect,
            batch,
            int(model.input_dim),
            int(model.output_dim),
            1.0 / math.sqrt(max(1, int(model.input_dim))),
            BLOCK_B=block_b,
        )
        model.w1.grad = gw1
        model.w2.grad = gw2
        model.linear_readout.grad = gdirect
        return loss.detach()
