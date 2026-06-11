"""Triton rational-k4 two-layer PrimitiveKAN kernels.

This is a narrow v22.05 D-RAT officialization candidate for the fixed
``rational_kat_lite`` basis used by PrimitiveKAN.  The backward pass
recomputes the fixed basis instead of materializing dense [B, D, K] or
[B, H, K] tensors.
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
    def _abs_tl(x):
        return tl.where(x >= 0.0, x, -x)


    @triton.jit
    def _rat_forward_hidden_kernel(
        x_ptr,
        mu_ptr,
        std_ptr,
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
        FAST_K2: tl.constexpr,
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
            denom_base = 1.0 + 0.5 * _abs_tl(z) + 0.125 * z * z
            if FAST_K2:
                phi0 = z / denom_base
                phi1 = (z * z) / (denom_base + 0.05)
                w0 = tl.load(w1_ptr + ((d[:, None] * H + offs_h[None, :]) * K), mask=d_mask[:, None] & h_mask[None, :], other=0.0)
                w1 = tl.load(w1_ptr + ((d[:, None] * H + offs_h[None, :]) * K + 1), mask=d_mask[:, None] & h_mask[None, :], other=0.0)
                acc += tl.dot(phi0, w0, input_precision="tf32x3")
                acc += tl.dot(phi1, w1, input_precision="tf32x3")
            else:
                power = z
                for kk in range(0, K):
                    if kk != 0:
                        power = power * z
                    denom = denom_base + 0.05 * kk
                    phi = power / denom
                    w = tl.load(w1_ptr + ((d[:, None] * H + offs_h[None, :]) * K + kk), mask=d_mask[:, None] & h_mask[None, :], other=0.0)
                    acc += tl.dot(phi, w, input_precision="tf32x3")
        h_val = _tanh_tl(acc * INV_SQRT_D)
        tl.store(h_ptr + offs_b[:, None] * H + offs_h[None, :], h_val, mask=b_mask[:, None] & h_mask[None, :])


    @triton.jit
    def _rat_forward_logits_kernel(
        h_ptr,
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
        FAST_K2: tl.constexpr,
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
            denom_base = 1.0 + 0.5 * _abs_tl(h_val) + 0.125 * h_val * h_val
            if FAST_K2:
                phi0 = h_val / denom_base
                phi1 = (h_val * h_val) / (denom_base + 0.05)
                w0 = tl.load(w2_ptr + ((hh[:, None] * C + offs_c[None, :]) * K), mask=h_mask[:, None] & c_mask[None, :], other=0.0)
                w1 = tl.load(w2_ptr + ((hh[:, None] * C + offs_c[None, :]) * K + 1), mask=h_mask[:, None] & c_mask[None, :], other=0.0)
                acc += tl.dot(phi0, w0, input_precision="tf32x3")
                acc += tl.dot(phi1, w1, input_precision="tf32x3")
            else:
                power = h_val
                for kk in range(0, K):
                    if kk != 0:
                        power = power * h_val
                    denom = denom_base + 0.05 * kk
                    phi = power / denom
                    w = tl.load(w2_ptr + ((hh[:, None] * C + offs_c[None, :]) * K + kk), mask=h_mask[:, None] & c_mask[None, :], other=0.0)
                    acc += tl.dot(phi, w, input_precision="tf32x3")
        tl.store(logits_ptr + offs_b[:, None] * C + offs_c[None, :], acc * INV_SQRT_H, mask=b_mask[:, None] & c_mask[None, :])


    @triton.jit
    def _rat_forward_singlelaunch_kernel(
        x_ptr,
        mu_ptr,
        std_ptr,
        w1_ptr,
        w2_ptr,
        h_ptr,
        logits_ptr,
        B: tl.constexpr,
        D: tl.constexpr,
        H: tl.constexpr,
        C: tl.constexpr,
        K: tl.constexpr,
        INV_SQRT_D: tl.constexpr,
        INV_SQRT_H: tl.constexpr,
        BLOCK_B: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_H: tl.constexpr,
        BLOCK_C: tl.constexpr,
        FAST_K2: tl.constexpr,
    ):
        b_block = tl.program_id(0)
        c_block = tl.program_id(1)
        offs_b = b_block * BLOCK_B + tl.arange(0, BLOCK_B)
        offs_c = c_block * BLOCK_C + tl.arange(0, BLOCK_C)
        offs_d = tl.arange(0, BLOCK_D)
        offs_h_local = tl.arange(0, BLOCK_H)
        b_mask = offs_b < B
        c_mask = offs_c < C
        logits_acc = tl.zeros((BLOCK_B, BLOCK_C), tl.float32)
        for h_start in range(0, H, BLOCK_H):
            hh = h_start + offs_h_local
            h_mask = hh < H
            hidden_acc = tl.zeros((BLOCK_B, BLOCK_H), tl.float32)
            for d_start in range(0, D, BLOCK_D):
                d = d_start + offs_d
                d_mask = d < D
                xv = tl.load(x_ptr + offs_b[:, None] * D + d[None, :], mask=b_mask[:, None] & d_mask[None, :], other=0.0)
                mu = tl.load(mu_ptr + d, mask=d_mask, other=0.0)
                std = tl.maximum(tl.load(std_ptr + d, mask=d_mask, other=1.0), 1.0e-3)
                z = _tanh_tl((xv - mu[None, :]) / std[None, :])
                denom_base = 1.0 + 0.5 * _abs_tl(z) + 0.125 * z * z
                if FAST_K2:
                    phi0 = z / denom_base
                    phi1 = (z * z) / (denom_base + 0.05)
                    w10 = tl.load(w1_ptr + ((d[:, None] * H + hh[None, :]) * K), mask=d_mask[:, None] & h_mask[None, :], other=0.0)
                    w11 = tl.load(w1_ptr + ((d[:, None] * H + hh[None, :]) * K + 1), mask=d_mask[:, None] & h_mask[None, :], other=0.0)
                    hidden_acc += tl.dot(phi0, w10, input_precision="tf32x3")
                    hidden_acc += tl.dot(phi1, w11, input_precision="tf32x3")
                else:
                    power = z
                    for kk in range(0, K):
                        if kk != 0:
                            power = power * z
                        denom = denom_base + 0.05 * kk
                        phi = power / denom
                        w1v = tl.load(w1_ptr + ((d[:, None] * H + hh[None, :]) * K + kk), mask=d_mask[:, None] & h_mask[None, :], other=0.0)
                        hidden_acc += tl.dot(phi, w1v, input_precision="tf32x3")
            h_val = _tanh_tl(hidden_acc * INV_SQRT_D)
            tl.store(h_ptr + offs_b[:, None] * H + hh[None, :], h_val, mask=b_mask[:, None] & h_mask[None, :])
            denom_base_h = 1.0 + 0.5 * _abs_tl(h_val) + 0.125 * h_val * h_val
            if FAST_K2:
                phi0_h = h_val / denom_base_h
                phi1_h = (h_val * h_val) / (denom_base_h + 0.05)
                w20 = tl.load(w2_ptr + ((hh[:, None] * C + offs_c[None, :]) * K), mask=h_mask[:, None] & c_mask[None, :], other=0.0)
                w21 = tl.load(w2_ptr + ((hh[:, None] * C + offs_c[None, :]) * K + 1), mask=h_mask[:, None] & c_mask[None, :], other=0.0)
                logits_acc += tl.dot(phi0_h, w20, input_precision="tf32x3")
                logits_acc += tl.dot(phi1_h, w21, input_precision="tf32x3")
            else:
                power_h = h_val
                for kk in range(0, K):
                    if kk != 0:
                        power_h = power_h * h_val
                    denom_h = denom_base_h + 0.05 * kk
                    phi_h = power_h / denom_h
                    w2v = tl.load(w2_ptr + ((hh[:, None] * C + offs_c[None, :]) * K + kk), mask=h_mask[:, None] & c_mask[None, :], other=0.0)
                    logits_acc += tl.dot(phi_h, w2v, input_precision="tf32x3")
        tl.store(logits_ptr + offs_b[:, None] * C + offs_c[None, :], logits_acc * INV_SQRT_H, mask=b_mask[:, None] & c_mask[None, :])


    @triton.jit
    def _rat_w2_grad_kernel(
        grad_logits_ptr,
        h_ptr,
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
        g = tl.load(grad_logits_ptr + offs * C + cc, mask=mask, other=0.0)
        h_val = tl.load(h_ptr + offs * H + hh, mask=mask, other=0.0)
        denom_base = 1.0 + 0.5 * _abs_tl(h_val) + 0.125 * h_val * h_val
        power = h_val
        for kk in range(0, K):
            if kk != 0:
                power = power * h_val
            denom = denom_base + 0.05 * kk
            phi = power / denom
            grad = tl.sum(phi * g, axis=0) * INV_SQRT_H
            tl.store(gw2_ptr + ((hh * C + cc) * K + kk), grad)


    @triton.jit
    def _rat_w1_grad_blockd_kernel(
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

        h_val = tl.load(h_ptr + offs_b * H + hh, mask=mask_b, other=0.0)
        denom_base_h = 1.0 + 0.5 * _abs_tl(h_val) + 0.125 * h_val * h_val
        sign_h = tl.where(h_val > 0.0, 1.0, tl.where(h_val < 0.0, -1.0, 0.0))
        denom_grad_h = 0.5 * sign_h + 0.25 * h_val
        grad_h = tl.full((BLOCK_B,), 0.0, tl.float32)
        for cc in range(0, C):
            g = tl.load(grad_logits_ptr + offs_b * C + cc, mask=mask_b, other=0.0)
            local = tl.full((BLOCK_B,), 0.0, tl.float32)
            power = h_val
            prev_power = tl.full((BLOCK_B,), 1.0, tl.float32)
            for kk in range(0, K):
                if kk != 0:
                    prev_power = power
                    power = power * h_val
                denom = denom_base_h + 0.05 * kk
                num_grad = (kk + 1.0) * prev_power
                dphi_h = (num_grad * denom - power * denom_grad_h) / (denom * denom)
                w = tl.load(w2_ptr + ((hh * C + cc) * K + kk))
                local += w * dphi_h
            grad_h += g * local * INV_SQRT_H
        grad_pre = grad_h * (1.0 - h_val * h_val)

        xv = tl.load(x_ptr + offs_b[:, None] * D + offs_d[None, :], mask=mask_b[:, None] & mask_d[None, :], other=0.0)
        mu = tl.load(mu_ptr + offs_d, mask=mask_d, other=0.0)
        std = tl.maximum(tl.load(std_ptr + offs_d, mask=mask_d, other=1.0), 1.0e-3)
        z = _tanh_tl((xv - mu[None, :]) / std[None, :])
        denom_base_z = 1.0 + 0.5 * _abs_tl(z) + 0.125 * z * z
        gp = grad_pre[:, None]
        power_z = z
        for kk in range(0, K):
            if kk != 0:
                power_z = power_z * z
            denom_z = denom_base_z + 0.05 * kk
            phi_z = power_z / denom_z
            grad = tl.sum(phi_z * gp, axis=0) * INV_SQRT_D
            tl.store(gw1_ptr + ((offs_d * H + hh) * K + kk), grad, mask=mask_d)


def _check_model(model) -> None:
    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is required for fused_rational_k4")
    if getattr(model.spec, "basis_name", "") != "rational_kat_lite":
        raise ValueError("fused_rational_k4 requires rational_kat_lite PrimitiveKAN")
    if int(model.k) not in {2, 4}:
        raise ValueError("fused_rational_k4 currently supports K=2 or K=4")


def _forward_block_h(model) -> int:
    variant = str(getattr(getattr(model, "spec", None), "init_variant", "")).lower()
    for value in (16, 32, 64, 128):
        if f"blockh{value}" in variant or f"block_h{value}" in variant or f"block-h{value}" in variant:
            return value
    return 32


def _forward_block_b(model, batch: int) -> int:
    variant = str(getattr(getattr(model, "spec", None), "init_variant", "")).lower()
    for value in (16, 32, 64, 128):
        if f"blockb{value}" in variant or f"block_b{value}" in variant or f"block-b{value}" in variant:
            return value
    return 64 if int(batch) >= 1024 else 32


def _forward_fast_k2(model) -> bool:
    variant = str(getattr(getattr(model, "spec", None), "init_variant", "")).lower()
    return int(model.k) == 2 and "fastk2" in variant


def _forward_singlelaunch(model) -> bool:
    variant = str(getattr(getattr(model, "spec", None), "init_variant", "")).lower()
    return "singlelaunch" in variant or "single_launch" in variant or "single-launch" in variant


def _forward_num_warps(model) -> int | None:
    variant = str(getattr(getattr(model, "spec", None), "init_variant", "")).lower()
    for value in (1, 2, 4, 8):
        if f"warps{value}" in variant or f"numwarps{value}" in variant or f"num_warps{value}" in variant:
            return value
    return None


def forward_matmul(model, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    _check_model(model)
    if x.dtype != torch.float32 or x.device.type != "cuda":
        raise ValueError("fused_rational_k4 forward_matmul requires fp32 CUDA input")
    x = x.contiguous()
    batch = int(x.shape[0])
    h_dim = int(model.hidden_dim)
    c_dim = int(model.output_dim)
    k = int(model.k)
    logits = torch.empty((batch, c_dim), device=x.device, dtype=torch.float32)
    h = torch.empty((batch, h_dim), device=x.device, dtype=torch.float32)
    block_b = _forward_block_b(model, batch)
    if int(model.input_dim) <= 8:
        block_d = 8
    elif int(model.input_dim) <= 16:
        block_d = 16
    elif int(model.input_dim) <= 32:
        block_d = 32
    else:
        block_d = 64
    block_h = _forward_block_h(model)
    fast_k2 = _forward_fast_k2(model)
    block_c = max(16, triton.next_power_of_2(c_dim))
    num_warps = _forward_num_warps(model)
    if _forward_singlelaunch(model):
        meta = {
            "BLOCK_B": block_b,
            "BLOCK_D": block_d,
            "BLOCK_H": block_h,
            "BLOCK_C": block_c,
            "FAST_K2": fast_k2,
        }
        if num_warps is not None:
            meta["num_warps"] = num_warps
        _rat_forward_singlelaunch_kernel[(triton.cdiv(batch, block_b), triton.cdiv(c_dim, block_c))](
            x,
            model.mu,
            model.std,
            model.w1,
            model.w2,
            h,
            logits,
            batch,
            int(model.input_dim),
            h_dim,
            c_dim,
            k,
            1.0 / math.sqrt(max(1, int(model.input_dim))),
            1.0 / math.sqrt(max(1, h_dim)),
            **meta,
        )
        return logits, h
    hidden_meta = {
        "BLOCK_B": block_b,
        "BLOCK_D": block_d,
        "BLOCK_H": block_h,
        "FAST_K2": fast_k2,
    }
    logits_meta = {
        "BLOCK_B": block_b,
        "BLOCK_H": block_h,
        "BLOCK_C": block_c,
        "FAST_K2": fast_k2,
    }
    if num_warps is not None:
        hidden_meta["num_warps"] = num_warps
        logits_meta["num_warps"] = num_warps
    _rat_forward_hidden_kernel[(triton.cdiv(batch, block_b), triton.cdiv(h_dim, block_h))](
        x,
        model.mu,
        model.std,
        model.w1,
        h,
        batch,
        int(model.input_dim),
        h_dim,
        k,
        1.0 / math.sqrt(max(1, int(model.input_dim))),
        **hidden_meta,
    )
    _rat_forward_logits_kernel[(triton.cdiv(batch, block_b), triton.cdiv(c_dim, block_c))](
        h,
        model.w2,
        logits,
        batch,
        h_dim,
        c_dim,
        k,
        1.0 / math.sqrt(max(1, h_dim)),
        **logits_meta,
    )
    return logits, h


def backward(model, x: torch.Tensor, y: torch.Tensor, logits: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    _check_model(model)
    if x.dtype != torch.float32 or x.device.type != "cuda":
        raise ValueError("fused_rational_k4 backward requires fp32 CUDA input")
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
        _rat_w2_grad_kernel[(int(model.hidden_dim), int(model.output_dim))](
            grad_logits,
            h,
            gw2,
            batch,
            int(model.hidden_dim),
            int(model.output_dim),
            k,
            1.0 / math.sqrt(max(1, int(model.hidden_dim))),
            BLOCK_B=block_b,
        )
        block_d = 32
        _rat_w1_grad_blockd_kernel[(triton.cdiv(int(model.input_dim), block_d), int(model.hidden_dim))](
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
            k,
            1.0 / math.sqrt(max(1, int(model.input_dim))),
            1.0 / math.sqrt(max(1, int(model.hidden_dim))),
            BLOCK_B=block_b,
            BLOCK_D=block_d,
        )
        model.w1.grad = gw1
        model.w2.grad = gw2
        return loss.detach()
