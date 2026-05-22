"""Strict FC-PureKAN primitive basis modules for v12.4 screens.

The module intentionally keeps the architecture simple: every learnable tensor
belongs to an edge function.  Fixed normalizers, centers, and scales are stored
as buffers and never branch on dataset names or labels.
"""

from __future__ import annotations

import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch import nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - Triton is optional outside CUDA runs.
    triton = None
    tl = None


EPS = 1.0e-12


@dataclass(frozen=True)
class PrimitiveSpec:
    candidate_id: str
    basis_family: str
    basis_name: str
    k: int
    hidden_dim: int
    source: str
    local_support: int
    global_support: int
    uses_exp: int
    uses_sin_cos: int
    uses_division: int
    uses_gather_scatter: int = 0
    uses_dense_basis_tensor: int = 1
    diagnostic_only: int = 0
    basis_order: int = 1
    init_variant: str = "default"
    model_kind: str = "edge_kan"


def matched_hidden(param_budget: int, input_dim: int, output_dim: int, k: int) -> int:
    denom = max(1, int(k) * (int(input_dim) + int(output_dim)))
    return max(4, int(round(float(param_budget) / float(denom))))


def _basis_eval(z: torch.Tensor, basis_name: str, k: int, centers: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    k = int(k)
    if basis_name in {"relu_hinge", "rswaf_hinge"}:
        vals = [z]
        for idx in range(k - 1):
            vals.append(F.relu(z - centers[idx]) / math.sqrt(max(1, k - 1)))
        return torch.stack(vals, dim=-1)
    if basis_name in {"compact_rbf", "fastkan_rbf"}:
        width = scales[0].clamp_min(1.0e-3)
        r = (z.unsqueeze(-1) - centers[:k]) / width
        return torch.exp(-0.5 * r.square())
    if basis_name == "legendre":
        vals = [torch.ones_like(z)]
        if k > 1:
            vals.append(z)
        for n in range(2, k):
            vals.append(((2 * n - 1) * z * vals[-1] - (n - 1) * vals[-2]) / n)
        return torch.stack(vals[:k], dim=-1)
    if basis_name == "chebyshev":
        vals = [torch.ones_like(z)]
        if k > 1:
            vals.append(z)
        for _ in range(2, k):
            vals.append(2.0 * z * vals[-1] - vals[-2])
        return torch.stack(vals[:k], dim=-1)
    if basis_name == "fourier_lowfreq":
        vals = [z]
        freq = 1
        while len(vals) < k:
            vals.append(torch.sin(math.pi * freq * z))
            if len(vals) < k:
                vals.append(torch.cos(math.pi * freq * z))
            freq += 1
        return torch.stack(vals[:k], dim=-1) / math.sqrt(max(1, k))
    if basis_name == "ricker_wavelet":
        vals = []
        width = scales[0].clamp_min(1.0e-3)
        for idx in range(k):
            r = (z - centers[idx]) / width
            vals.append((1.0 - r.square()) * torch.exp(-0.5 * r.square()))
        return torch.stack(vals, dim=-1)
    if basis_name == "hat_wavelet":
        vals = []
        width = scales[0].clamp_min(1.0e-3)
        for idx in range(k):
            r = (z - centers[idx]).abs() / width
            inner = F.relu(1.0 - r)
            outer = F.relu(1.0 - 0.5 * r)
            vals.append(inner - 0.5 * outer)
        return torch.stack(vals, dim=-1)
    if basis_name == "bspline_order1":
        width = scales[0].clamp_min(1.0e-3)
        r = (z.unsqueeze(-1) - centers[:k]).abs() / width
        return F.relu(1.0 - r)
    if basis_name == "bspline_relu_combo":
        vals = []
        width = scales[0].clamp_min(1.0e-3)
        local_count = max(1, k - 2)
        for idx in range(local_count):
            vals.append(F.relu(1.0 - (z - centers[idx]).abs() / width))
        vals.append(z)
        vals.append(F.relu(z))
        return torch.stack(vals[:k], dim=-1)
    if basis_name == "poly2_relu_combo":
        vals = [z, z.square() - (1.0 / 3.0), F.relu(z)]
        while len(vals) < k:
            vals.append(vals[-1] * z)
        return torch.stack(vals[:k], dim=-1)
    if basis_name == "rational_kat_lite":
        denom = 1.0 + 0.5 * z.abs() + 0.125 * z.square()
        vals = [z / denom]
        cur = z
        for idx in range(1, k):
            cur = cur * z
            vals.append(cur / (denom + 0.05 * idx))
        return torch.stack(vals[:k], dim=-1)
    raise ValueError(f"unknown primitive basis {basis_name}")


def _basis_derivative(z: torch.Tensor, basis_name: str, k: int, centers: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Derivative of fixed basis channels with respect to scalar input z."""
    k = int(k)
    if basis_name in {"relu_hinge", "rswaf_hinge"}:
        vals = [torch.ones_like(z)]
        for idx in range(k - 1):
            vals.append((z > centers[idx]).to(dtype=z.dtype) / math.sqrt(max(1, k - 1)))
        return torch.stack(vals[:k], dim=-1)
    if basis_name in {"compact_rbf", "fastkan_rbf"}:
        width = scales[0].clamp_min(1.0e-3)
        r = (z.unsqueeze(-1) - centers[:k]) / width
        phi = torch.exp(-0.5 * r.square())
        return -(r / width) * phi
    if basis_name == "legendre":
        vals = [torch.zeros_like(z)]
        polys = [torch.ones_like(z)]
        if k > 1:
            vals.append(torch.ones_like(z))
            polys.append(z)
        for n in range(2, k):
            p = ((2 * n - 1) * z * polys[-1] - (n - 1) * polys[-2]) / n
            d = ((2 * n - 1) * (polys[-1] + z * vals[-1]) - (n - 1) * vals[-2]) / n
            polys.append(p)
            vals.append(d)
        return torch.stack(vals[:k], dim=-1)
    if basis_name == "chebyshev":
        vals = [torch.zeros_like(z)]
        polys = [torch.ones_like(z)]
        if k > 1:
            vals.append(torch.ones_like(z))
            polys.append(z)
        for _ in range(2, k):
            p = 2.0 * z * polys[-1] - polys[-2]
            d = 2.0 * polys[-1] + 2.0 * z * vals[-1] - vals[-2]
            polys.append(p)
            vals.append(d)
        return torch.stack(vals[:k], dim=-1)
    if basis_name == "fourier_lowfreq":
        vals = [torch.ones_like(z)]
        freq = 1
        while len(vals) < k:
            vals.append(math.pi * freq * torch.cos(math.pi * freq * z))
            if len(vals) < k:
                vals.append(-math.pi * freq * torch.sin(math.pi * freq * z))
            freq += 1
        return torch.stack(vals[:k], dim=-1) / math.sqrt(max(1, k))
    if basis_name == "ricker_wavelet":
        vals = []
        width = scales[0].clamp_min(1.0e-3)
        for idx in range(k):
            r = (z - centers[idx]) / width
            vals.append((r.pow(3) - 3.0 * r) * torch.exp(-0.5 * r.square()) / width)
        return torch.stack(vals, dim=-1)
    if basis_name == "hat_wavelet":
        vals = []
        width = scales[0].clamp_min(1.0e-3)
        for idx in range(k):
            u = (z - centers[idx]) / width
            abs_u = u.abs()
            sign = torch.sign(u)
            vals.append(-sign * (abs_u < 1.0).to(dtype=z.dtype) / width + 0.25 * sign * (abs_u < 2.0).to(dtype=z.dtype) / width)
        return torch.stack(vals, dim=-1)
    if basis_name == "bspline_order1":
        width = scales[0].clamp_min(1.0e-3)
        u = (z.unsqueeze(-1) - centers[:k]) / width
        return -torch.sign(u) * (u.abs() < 1.0).to(dtype=z.dtype) / width
    if basis_name == "bspline_relu_combo":
        vals = []
        width = scales[0].clamp_min(1.0e-3)
        local_count = max(1, k - 2)
        for idx in range(local_count):
            u = (z - centers[idx]) / width
            vals.append(-torch.sign(u) * (u.abs() < 1.0).to(dtype=z.dtype) / width)
        vals.append(torch.ones_like(z))
        vals.append((z > 0.0).to(dtype=z.dtype))
        return torch.stack(vals[:k], dim=-1)
    if basis_name == "poly2_relu_combo":
        vals = [torch.ones_like(z), 2.0 * z, (z > 0.0).to(dtype=z.dtype)]
        while len(vals) < k:
            power = len(vals) + 1
            vals.append(float(power) * z.pow(power - 1))
        return torch.stack(vals[:k], dim=-1)
    if basis_name == "rational_kat_lite":
        denom = 1.0 + 0.5 * z.abs() + 0.125 * z.square()
        denom_grad = 0.5 * torch.sign(z) + 0.25 * z
        vals = []
        for idx in range(k):
            power = idx + 1
            numerator = z.pow(power)
            numerator_grad = float(power) * z.pow(power - 1)
            den = denom + (0.05 * idx if idx > 0 else 0.0)
            vals.append((numerator_grad * den - numerator * denom_grad) / den.square().clamp_min(EPS))
        return torch.stack(vals[:k], dim=-1)
    return torch.zeros(*z.shape, k, device=z.device, dtype=z.dtype)


def _basis_channel(z: torch.Tensor, basis_name: str, k: int, centers: torch.Tensor, scales: torch.Tensor, idx: int) -> torch.Tensor:
    if basis_name in {"relu_hinge", "rswaf_hinge"}:
        if idx == 0:
            return z
        return F.relu(z - centers[idx - 1]) / math.sqrt(max(1, k - 1))
    if basis_name in {"compact_rbf", "fastkan_rbf"}:
        width = scales[0].clamp_min(1.0e-3)
        return torch.exp(-0.5 * ((z - centers[idx]) / width).square())
    if basis_name == "bspline_order1":
        width = scales[0].clamp_min(1.0e-3)
        return F.relu(1.0 - (z - centers[idx]).abs() / width)
    if basis_name == "hat_wavelet":
        width = scales[0].clamp_min(1.0e-3)
        r = (z - centers[idx]).abs() / width
        return F.relu(1.0 - r) - 0.5 * F.relu(1.0 - 0.5 * r)
    if basis_name == "bspline_relu_combo":
        local_count = max(1, k - 2)
        if idx < local_count:
            width = scales[0].clamp_min(1.0e-3)
            return F.relu(1.0 - (z - centers[idx]).abs() / width)
        if idx == local_count:
            return z
        return F.relu(z)
    if basis_name == "poly2_relu_combo":
        if idx == 0:
            return z
        if idx == 1:
            return z.square() - (1.0 / 3.0)
        if idx == 2:
            return F.relu(z)
        return z.pow(idx + 1)
    return _basis_eval(z, basis_name, k, centers, scales)[..., idx]


def _basis_derivative_channel(z: torch.Tensor, basis_name: str, k: int, centers: torch.Tensor, scales: torch.Tensor, idx: int) -> torch.Tensor:
    if basis_name in {"relu_hinge", "rswaf_hinge"}:
        if idx == 0:
            return torch.ones_like(z)
        return (z > centers[idx - 1]).to(dtype=z.dtype) / math.sqrt(max(1, k - 1))
    if basis_name in {"compact_rbf", "fastkan_rbf"}:
        width = scales[0].clamp_min(1.0e-3)
        r = (z - centers[idx]) / width
        phi = torch.exp(-0.5 * r.square())
        return -(r / width) * phi
    if basis_name == "hat_wavelet":
        width = scales[0].clamp_min(1.0e-3)
        u = (z - centers[idx]) / width
        abs_u = u.abs()
        sign = torch.sign(u)
        return -sign * (abs_u < 1.0).to(dtype=z.dtype) / width + 0.25 * sign * (abs_u < 2.0).to(dtype=z.dtype) / width
    if basis_name == "bspline_order1":
        width = scales[0].clamp_min(1.0e-3)
        u = (z - centers[idx]) / width
        return -torch.sign(u) * (u.abs() < 1.0).to(dtype=z.dtype) / width
    if basis_name == "bspline_relu_combo":
        local_count = max(1, k - 2)
        if idx < local_count:
            width = scales[0].clamp_min(1.0e-3)
            u = (z - centers[idx]) / width
            return -torch.sign(u) * (u.abs() < 1.0).to(dtype=z.dtype) / width
        if idx == local_count:
            return torch.ones_like(z)
        return (z > 0.0).to(dtype=z.dtype)
    if basis_name == "poly2_relu_combo":
        if idx == 0:
            return torch.ones_like(z)
        if idx == 1:
            return 2.0 * z
        if idx == 2:
            return (z > 0.0).to(dtype=z.dtype)
        power = idx + 1
        return float(power) * z.pow(power - 1)
    return _basis_derivative(z, basis_name, k, centers, scales)[..., idx]


def _stream_mix(z: torch.Tensor, weight: torch.Tensor, basis_name: str, k: int, centers: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    out = None
    for idx in range(int(k)):
        part = _basis_channel(z, basis_name, k, centers, scales, idx) @ weight[..., idx]
        out = part if out is None else out + part
    assert out is not None
    return out


_FLASHKAT_RATIONAL_KERNEL_CACHE = None


def _flashkat_rational_kernels():
    global _FLASHKAT_RATIONAL_KERNEL_CACHE
    if _FLASHKAT_RATIONAL_KERNEL_CACHE is not None:
        return _FLASHKAT_RATIONAL_KERNEL_CACHE
    root = Path(__file__).resolve().parents[2] / "third_party" / "FlashKAT"
    root_s = str(root)
    if root_s not in sys.path:
        sys.path.insert(0, root_s)
    from rational_kat.kernels.flash_rational_triton import (  # type: ignore[import-not-found]
        rational_bwd_triton,
        rational_fwd_triton,
        FlashRationalTriton1DGroup,
    )

    _FLASHKAT_RATIONAL_KERNEL_CACHE = rational_fwd_triton, rational_bwd_triton, FlashRationalTriton1DGroup
    return _FLASHKAT_RATIONAL_KERNEL_CACHE


if triton is not None and tl is not None:

    @triton.jit
    def _pairbucket_hidden_delta_kernel(
        z,
        left,
        right,
        bucket,
        weight,
        out,
        D: tl.constexpr,
        R: tl.constexpr,
        H: tl.constexpr,
        SCALE: tl.constexpr,
        BLOCK_R: tl.constexpr,
    ):
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK_R)
        mask = offs < R
        li = tl.load(left + offs, mask=mask, other=0)
        ri = tl.load(right + offs, mask=mask, other=0)
        bi = tl.load(bucket + offs, mask=mask, other=0)
        wi = tl.load(weight + offs, mask=mask, other=0.0)
        z_base = z + row * D
        vals = tl.load(z_base + li, mask=mask, other=0.0) * tl.load(z_base + ri, mask=mask, other=0.0)
        vals = vals * wi * SCALE
        tl.atomic_add(out + row * H + bi, vals, sem="relaxed", mask=mask)

    @triton.jit
    def _paircross_readout_logits_kernel(
        z,
        left,
        right,
        readout,
        out,
        D: tl.constexpr,
        R: tl.constexpr,
        C: tl.constexpr,
        SCALE: tl.constexpr,
        FEATURE_MODE: tl.constexpr,
        BLOCK_R: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        row = tl.program_id(0)
        c0 = tl.program_id(1) * BLOCK_C
        offs_r = tl.arange(0, BLOCK_R)
        offs_c = c0 + tl.arange(0, BLOCK_C)
        mask_r = offs_r < R
        mask_c = offs_c < C
        li = tl.load(left + offs_r, mask=mask_r, other=0)
        ri = tl.load(right + offs_r, mask=mask_r, other=0)
        z_base = z + row * D
        zl = tl.load(z_base + li, mask=mask_r, other=0.0)
        zr = tl.load(z_base + ri, mask=mask_r, other=0.0)
        vals = zl * zr
        if FEATURE_MODE == 1:
            s = zl + zr
            vals = 0.5 * s * s
        elif FEATURE_MODE == 2:
            d = zl - zr
            vals = 0.5 * d * d
        w = tl.load(readout + offs_r[:, None] * C + offs_c[None, :], mask=mask_r[:, None] & mask_c[None, :], other=0.0)
        acc = tl.sum(vals[:, None] * w, axis=0) * SCALE
        tl.store(out + row * C + offs_c, acc, mask=mask_c)

    @triton.jit
    def _paircross_readout_grad_kernel(
        z,
        left,
        right,
        grad_logits,
        grad_readout,
        D: tl.constexpr,
        R: tl.constexpr,
        C: tl.constexpr,
        B: tl.constexpr,
        SCALE: tl.constexpr,
        FEATURE_MODE: tl.constexpr,
        BLOCK_B: tl.constexpr,
    ):
        r = tl.program_id(0)
        c = tl.program_id(1)
        offs_b = tl.arange(0, BLOCK_B)
        mask_b = offs_b < B
        li = tl.load(left + r)
        ri = tl.load(right + r)
        zl = tl.load(z + offs_b * D + li, mask=mask_b, other=0.0)
        zr = tl.load(z + offs_b * D + ri, mask=mask_b, other=0.0)
        vals = zl * zr
        if FEATURE_MODE == 1:
            s = zl + zr
            vals = 0.5 * s * s
        elif FEATURE_MODE == 2:
            d = zl - zr
            vals = 0.5 * d * d
        gl = tl.load(grad_logits + offs_b * C + c, mask=mask_b, other=0.0)
        acc = tl.sum(vals * gl, axis=0) * SCALE
        tl.store(grad_readout + r * C + c, acc)

    @triton.jit
    def _paircross_readout_logits_block_kernel(
        z,
        left,
        right,
        readout,
        feat_mean,
        feat_inv_std,
        bias,
        out,
        D: tl.constexpr,
        R: tl.constexpr,
        C: tl.constexpr,
        B: tl.constexpr,
        SCALE: tl.constexpr,
        HAS_NORM: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        HAS_CAP: tl.constexpr,
        CAP_VALUE: tl.constexpr,
        FEATURE_MODE: tl.constexpr,
        BLOCK_B: tl.constexpr,
        BLOCK_R: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        b0 = tl.program_id(0) * BLOCK_B
        c0 = tl.program_id(1) * BLOCK_C
        offs_b = b0 + tl.arange(0, BLOCK_B)
        offs_r = tl.arange(0, BLOCK_R)
        offs_c = c0 + tl.arange(0, BLOCK_C)
        mask_b = offs_b < B
        mask_r = offs_r < R
        mask_c = offs_c < C
        li = tl.load(left + offs_r, mask=mask_r, other=0)
        ri = tl.load(right + offs_r, mask=mask_r, other=0)
        zl = tl.load(z + offs_b[:, None] * D + li[None, :], mask=mask_b[:, None] & mask_r[None, :], other=0.0)
        zr = tl.load(z + offs_b[:, None] * D + ri[None, :], mask=mask_b[:, None] & mask_r[None, :], other=0.0)
        vals = (zl * zr).to(tl.float32)
        if FEATURE_MODE == 1:
            s = zl + zr
            vals = (0.5 * s * s).to(tl.float32)
        elif FEATURE_MODE == 2:
            d = zl - zr
            vals = (0.5 * d * d).to(tl.float32)
        w = tl.load(readout + offs_r[:, None] * C + offs_c[None, :], mask=mask_r[:, None] & mask_c[None, :], other=0.0).to(tl.float32)
        if HAS_NORM:
            mu = tl.load(feat_mean + offs_r, mask=mask_r, other=0.0).to(tl.float32)
            inv_std = tl.load(feat_inv_std + offs_r, mask=mask_r, other=1.0).to(tl.float32)
            w_eff = w * inv_std[:, None]
            center = tl.sum(mu[:, None] * w_eff, axis=0)
            acc = (tl.dot(vals, w_eff, input_precision="ieee") - center[None, :]) * SCALE
        else:
            acc = tl.dot(vals, w, input_precision="ieee") * SCALE
        if HAS_BIAS:
            b = tl.load(bias + offs_c, mask=mask_c, other=0.0).to(tl.float32)
            acc = acc + b[None, :]
        if HAS_CAP:
            inv_cap = 1.0 / CAP_VALUE
            t = 2.0 / (1.0 + tl.exp(-2.0 * acc * inv_cap)) - 1.0
            acc = CAP_VALUE * t
        tl.store(out + offs_b[:, None] * C + offs_c[None, :], acc, mask=mask_b[:, None] & mask_c[None, :])

    @triton.jit
    def _paircross_readout_grad_block_kernel(
        z,
        left,
        right,
        grad_logits,
        feat_mean,
        feat_inv_std,
        grad_readout,
        D: tl.constexpr,
        R: tl.constexpr,
        C: tl.constexpr,
        B: tl.constexpr,
        SCALE: tl.constexpr,
        HAS_NORM: tl.constexpr,
        FEATURE_MODE: tl.constexpr,
        BLOCK_B: tl.constexpr,
        BLOCK_R: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        r0 = tl.program_id(0) * BLOCK_R
        c0 = tl.program_id(1) * BLOCK_C
        offs_r = r0 + tl.arange(0, BLOCK_R)
        offs_b = tl.arange(0, BLOCK_B)
        offs_c = c0 + tl.arange(0, BLOCK_C)
        mask_r = offs_r < R
        mask_b = offs_b < B
        mask_c = offs_c < C
        li = tl.load(left + offs_r, mask=mask_r, other=0)
        ri = tl.load(right + offs_r, mask=mask_r, other=0)
        zl = tl.load(z + offs_b[None, :] * D + li[:, None], mask=mask_r[:, None] & mask_b[None, :], other=0.0)
        zr = tl.load(z + offs_b[None, :] * D + ri[:, None], mask=mask_r[:, None] & mask_b[None, :], other=0.0)
        vals = (zl * zr).to(tl.float32)
        if FEATURE_MODE == 1:
            s = zl + zr
            vals = (0.5 * s * s).to(tl.float32)
        elif FEATURE_MODE == 2:
            d = zl - zr
            vals = (0.5 * d * d).to(tl.float32)
        gl = tl.load(grad_logits + offs_b[:, None] * C + offs_c[None, :], mask=mask_b[:, None] & mask_c[None, :], other=0.0).to(tl.float32)
        raw_acc = tl.dot(vals, gl, input_precision="ieee")
        if HAS_NORM:
            mu = tl.load(feat_mean + offs_r, mask=mask_r, other=0.0).to(tl.float32)
            inv_std = tl.load(feat_inv_std + offs_r, mask=mask_r, other=1.0).to(tl.float32)
            sum_gl = tl.sum(gl, axis=0)
            acc = (raw_acc - mu[:, None] * sum_gl[None, :]) * inv_std[:, None] * SCALE
        else:
            acc = raw_acc * SCALE
        tl.store(grad_readout + offs_r[:, None] * C + offs_c[None, :], acc, mask=mask_r[:, None] & mask_c[None, :])

    @triton.jit
    def _paircross_readout_grad_block_atomic_kernel(
        z,
        left,
        right,
        grad_logits,
        feat_mean,
        feat_inv_std,
        grad_readout,
        D: tl.constexpr,
        R: tl.constexpr,
        C: tl.constexpr,
        B: tl.constexpr,
        SCALE: tl.constexpr,
        HAS_NORM: tl.constexpr,
        FEATURE_MODE: tl.constexpr,
        BLOCK_B: tl.constexpr,
        BLOCK_R: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        r0 = tl.program_id(0) * BLOCK_R
        c0 = tl.program_id(1) * BLOCK_C
        b0 = tl.program_id(2) * BLOCK_B
        offs_r = r0 + tl.arange(0, BLOCK_R)
        offs_b = b0 + tl.arange(0, BLOCK_B)
        offs_c = c0 + tl.arange(0, BLOCK_C)
        mask_r = offs_r < R
        mask_b = offs_b < B
        mask_c = offs_c < C
        li = tl.load(left + offs_r, mask=mask_r, other=0)
        ri = tl.load(right + offs_r, mask=mask_r, other=0)
        zl = tl.load(z + offs_b[None, :] * D + li[:, None], mask=mask_r[:, None] & mask_b[None, :], other=0.0)
        zr = tl.load(z + offs_b[None, :] * D + ri[:, None], mask=mask_r[:, None] & mask_b[None, :], other=0.0)
        vals = (zl * zr).to(tl.float32)
        if FEATURE_MODE == 1:
            s = zl + zr
            vals = (0.5 * s * s).to(tl.float32)
        elif FEATURE_MODE == 2:
            d = zl - zr
            vals = (0.5 * d * d).to(tl.float32)
        gl = tl.load(grad_logits + offs_b[:, None] * C + offs_c[None, :], mask=mask_b[:, None] & mask_c[None, :], other=0.0).to(tl.float32)
        raw_acc = tl.dot(vals, gl, input_precision="ieee")
        if HAS_NORM:
            mu = tl.load(feat_mean + offs_r, mask=mask_r, other=0.0).to(tl.float32)
            inv_std = tl.load(feat_inv_std + offs_r, mask=mask_r, other=1.0).to(tl.float32)
            sum_gl = tl.sum(gl, axis=0)
            acc = (raw_acc - mu[:, None] * sum_gl[None, :]) * inv_std[:, None] * SCALE
        else:
            acc = raw_acc * SCALE
        tl.atomic_add(grad_readout + offs_r[:, None] * C + offs_c[None, :], acc, sem="relaxed", mask=mask_r[:, None] & mask_c[None, :])

    @triton.jit
    def _paircross_readout_grad_block_cap_kernel(
        z,
        left,
        right,
        readout,
        grad_logits,
        feat_mean,
        feat_inv_std,
        grad_readout,
        D: tl.constexpr,
        R: tl.constexpr,
        C: tl.constexpr,
        B: tl.constexpr,
        SCALE: tl.constexpr,
        CAP_VALUE: tl.constexpr,
        HAS_NORM: tl.constexpr,
        FEATURE_MODE: tl.constexpr,
        BLOCK_B: tl.constexpr,
        BLOCK_R: tl.constexpr,
        BLOCK_RF: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        r0 = tl.program_id(0) * BLOCK_R
        c0 = tl.program_id(1) * BLOCK_C
        offs_r = r0 + tl.arange(0, BLOCK_R)
        offs_rf = tl.arange(0, BLOCK_RF)
        offs_b = tl.arange(0, BLOCK_B)
        offs_c = c0 + tl.arange(0, BLOCK_C)
        mask_r = offs_r < R
        mask_rf = offs_rf < R
        mask_b = offs_b < B
        mask_c = offs_c < C

        li_f = tl.load(left + offs_rf, mask=mask_rf, other=0)
        ri_f = tl.load(right + offs_rf, mask=mask_rf, other=0)
        zl_f = tl.load(z + offs_b[:, None] * D + li_f[None, :], mask=mask_b[:, None] & mask_rf[None, :], other=0.0)
        zr_f = tl.load(z + offs_b[:, None] * D + ri_f[None, :], mask=mask_b[:, None] & mask_rf[None, :], other=0.0)
        vals_f = (zl_f * zr_f).to(tl.float32)
        if FEATURE_MODE == 1:
            s_f = zl_f + zr_f
            vals_f = (0.5 * s_f * s_f).to(tl.float32)
        elif FEATURE_MODE == 2:
            d_f = zl_f - zr_f
            vals_f = (0.5 * d_f * d_f).to(tl.float32)
        w_f = tl.load(readout + offs_rf[:, None] * C + offs_c[None, :], mask=mask_rf[:, None] & mask_c[None, :], other=0.0).to(tl.float32)
        if HAS_NORM:
            mu_f = tl.load(feat_mean + offs_rf, mask=mask_rf, other=0.0).to(tl.float32)
            inv_f = tl.load(feat_inv_std + offs_rf, mask=mask_rf, other=1.0).to(tl.float32)
            w_eff_f = w_f * inv_f[:, None]
            center = tl.sum(mu_f[:, None] * w_eff_f, axis=0)
            raw_logits = (tl.dot(vals_f, w_eff_f, input_precision="ieee") - center[None, :]) * SCALE
        else:
            raw_logits = tl.dot(vals_f, w_f, input_precision="ieee") * SCALE
        inv_cap = 1.0 / CAP_VALUE
        t = 2.0 / (1.0 + tl.exp(-2.0 * raw_logits * inv_cap)) - 1.0
        deriv = 1.0 - t * t
        gl = tl.load(grad_logits + offs_b[:, None] * C + offs_c[None, :], mask=mask_b[:, None] & mask_c[None, :], other=0.0).to(tl.float32)
        gl_eff = gl * deriv

        li = tl.load(left + offs_r, mask=mask_r, other=0)
        ri = tl.load(right + offs_r, mask=mask_r, other=0)
        zl = tl.load(z + offs_b[None, :] * D + li[:, None], mask=mask_r[:, None] & mask_b[None, :], other=0.0)
        zr = tl.load(z + offs_b[None, :] * D + ri[:, None], mask=mask_r[:, None] & mask_b[None, :], other=0.0)
        vals = (zl * zr).to(tl.float32)
        if FEATURE_MODE == 1:
            s = zl + zr
            vals = (0.5 * s * s).to(tl.float32)
        elif FEATURE_MODE == 2:
            d = zl - zr
            vals = (0.5 * d * d).to(tl.float32)
        raw_acc = tl.dot(vals, gl_eff, input_precision="ieee")
        if HAS_NORM:
            mu = tl.load(feat_mean + offs_r, mask=mask_r, other=0.0).to(tl.float32)
            inv_std = tl.load(feat_inv_std + offs_r, mask=mask_r, other=1.0).to(tl.float32)
            sum_gl = tl.sum(gl_eff, axis=0)
            acc = (raw_acc - mu[:, None] * sum_gl[None, :]) * inv_std[:, None] * SCALE
        else:
            acc = raw_acc * SCALE
        tl.store(grad_readout + offs_r[:, None] * C + offs_c[None, :], acc, mask=mask_r[:, None] & mask_c[None, :])

    @triton.jit
    def _paircross_readout_grad_block_cap_atomic_kernel(
        z,
        left,
        right,
        readout,
        grad_logits,
        feat_mean,
        feat_inv_std,
        grad_readout,
        D: tl.constexpr,
        R: tl.constexpr,
        C: tl.constexpr,
        B: tl.constexpr,
        SCALE: tl.constexpr,
        CAP_VALUE: tl.constexpr,
        HAS_NORM: tl.constexpr,
        FEATURE_MODE: tl.constexpr,
        BLOCK_B: tl.constexpr,
        BLOCK_R: tl.constexpr,
        BLOCK_RF: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        r0 = tl.program_id(0) * BLOCK_R
        c0 = tl.program_id(1) * BLOCK_C
        b0 = tl.program_id(2) * BLOCK_B
        offs_r = r0 + tl.arange(0, BLOCK_R)
        offs_rf = tl.arange(0, BLOCK_RF)
        offs_b = b0 + tl.arange(0, BLOCK_B)
        offs_c = c0 + tl.arange(0, BLOCK_C)
        mask_r = offs_r < R
        mask_rf = offs_rf < R
        mask_b = offs_b < B
        mask_c = offs_c < C

        li_f = tl.load(left + offs_rf, mask=mask_rf, other=0)
        ri_f = tl.load(right + offs_rf, mask=mask_rf, other=0)
        zl_f = tl.load(z + offs_b[:, None] * D + li_f[None, :], mask=mask_b[:, None] & mask_rf[None, :], other=0.0)
        zr_f = tl.load(z + offs_b[:, None] * D + ri_f[None, :], mask=mask_b[:, None] & mask_rf[None, :], other=0.0)
        vals_f = (zl_f * zr_f).to(tl.float32)
        if FEATURE_MODE == 1:
            s_f = zl_f + zr_f
            vals_f = (0.5 * s_f * s_f).to(tl.float32)
        elif FEATURE_MODE == 2:
            d_f = zl_f - zr_f
            vals_f = (0.5 * d_f * d_f).to(tl.float32)
        w_f = tl.load(readout + offs_rf[:, None] * C + offs_c[None, :], mask=mask_rf[:, None] & mask_c[None, :], other=0.0).to(tl.float32)
        if HAS_NORM:
            mu_f = tl.load(feat_mean + offs_rf, mask=mask_rf, other=0.0).to(tl.float32)
            inv_f = tl.load(feat_inv_std + offs_rf, mask=mask_rf, other=1.0).to(tl.float32)
            w_eff_f = w_f * inv_f[:, None]
            center = tl.sum(mu_f[:, None] * w_eff_f, axis=0)
            raw_logits = (tl.dot(vals_f, w_eff_f, input_precision="ieee") - center[None, :]) * SCALE
        else:
            raw_logits = tl.dot(vals_f, w_f, input_precision="ieee") * SCALE
        inv_cap = 1.0 / CAP_VALUE
        t = 2.0 / (1.0 + tl.exp(-2.0 * raw_logits * inv_cap)) - 1.0
        deriv = 1.0 - t * t
        gl = tl.load(grad_logits + offs_b[:, None] * C + offs_c[None, :], mask=mask_b[:, None] & mask_c[None, :], other=0.0).to(tl.float32)
        gl_eff = gl * deriv

        li = tl.load(left + offs_r, mask=mask_r, other=0)
        ri = tl.load(right + offs_r, mask=mask_r, other=0)
        zl = tl.load(z + offs_b[None, :] * D + li[:, None], mask=mask_r[:, None] & mask_b[None, :], other=0.0)
        zr = tl.load(z + offs_b[None, :] * D + ri[:, None], mask=mask_r[:, None] & mask_b[None, :], other=0.0)
        vals = (zl * zr).to(tl.float32)
        if FEATURE_MODE == 1:
            s = zl + zr
            vals = (0.5 * s * s).to(tl.float32)
        elif FEATURE_MODE == 2:
            d = zl - zr
            vals = (0.5 * d * d).to(tl.float32)
        raw_acc = tl.dot(vals, gl_eff, input_precision="ieee")
        if HAS_NORM:
            mu = tl.load(feat_mean + offs_r, mask=mask_r, other=0.0).to(tl.float32)
            inv_std = tl.load(feat_inv_std + offs_r, mask=mask_r, other=1.0).to(tl.float32)
            sum_gl = tl.sum(gl_eff, axis=0)
            acc = (raw_acc - mu[:, None] * sum_gl[None, :]) * inv_std[:, None] * SCALE
        else:
            acc = raw_acc * SCALE
        tl.atomic_add(grad_readout + offs_r[:, None] * C + offs_c[None, :], acc, sem="relaxed", mask=mask_r[:, None] & mask_c[None, :])

    @triton.jit
    def _hidden_tail_readout_vjp_kernel(
        h,
        tail_readout,
        grad_logits,
        grad_tail_readout,
        grad_h_tail,
        H: tl.constexpr,
        C: tl.constexpr,
        B: tl.constexpr,
        SCALE: tl.constexpr,
        CENTER_MIX: tl.constexpr,
        MODE: tl.constexpr,
        BLOCK_H: tl.constexpr,
        BLOCK_B: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        h0 = tl.program_id(0) * BLOCK_H
        c0 = tl.program_id(1) * BLOCK_C
        offs_h = h0 + tl.arange(0, BLOCK_H)
        offs_b = tl.arange(0, BLOCK_B)
        offs_c = c0 + tl.arange(0, BLOCK_C)
        mask_h = offs_h < H
        mask_b = offs_b < B
        mask_c = offs_c < C

        hv_hb = tl.load(h + offs_h[:, None] + offs_b[None, :] * H, mask=mask_h[:, None] & mask_b[None, :], other=0.0).to(tl.float32)
        abs_hv_hb = tl.abs(hv_hb)
        feat_hb = abs_hv_hb
        deriv_hb = tl.where(hv_hb > 0.0, 1.0, tl.where(hv_hb < 0.0, -1.0, 0.0))
        if MODE == 2:
            denom = 1.0 + abs_hv_hb
            feat_hb = hv_hb / denom
            deriv_hb = 1.0 / (denom * denom)
        elif MODE == 3 or MODE == 4:
            denom = 1.0 + abs_hv_hb
            feat_hb = hv_hb / denom
            deriv_hb = 1.0 / (denom * denom)
            mean_feat_b = tl.sum(feat_hb, axis=0) / H
            feat_hb = feat_hb - CENTER_MIX * mean_feat_b[None, :]
        elif MODE == 5:
            denom = 1.0 + abs_hv_hb
            feat_hb = hv_hb + CENTER_MIX * (hv_hb / denom)
            deriv_hb = 1.0 + CENTER_MIX / (denom * denom)
        elif MODE == 6:
            tanh_hb = 2.0 / (1.0 + tl.exp(-2.0 * hv_hb)) - 1.0
            feat_hb = hv_hb + CENTER_MIX * tanh_hb
            deriv_hb = 1.0 + CENTER_MIX * (1.0 - tanh_hb * tanh_hb)
        elif MODE == 7:
            tanh_hb = 2.0 / (1.0 + tl.exp(-2.0 * hv_hb)) - 1.0
            mean_tanh_b = tl.sum(tanh_hb, axis=0) / H
            feat_hb = hv_hb + CENTER_MIX * (tanh_hb - mean_tanh_b[None, :])
            deriv_hb = 1.0 - tanh_hb * tanh_hb
        elif MODE == 8:
            hv2_hb = hv_hb * hv_hb
            denom = 1.0 + hv2_hb
            feat_hb = hv_hb + CENTER_MIX * (hv_hb * abs_hv_hb / denom)
            deriv_hb = 1.0 + CENTER_MIX * (2.0 * abs_hv_hb / (denom * denom))

        gl_bc = tl.load(grad_logits + offs_b[:, None] * C + offs_c[None, :], mask=mask_b[:, None] & mask_c[None, :], other=0.0).to(tl.float32)
        grad_readout = tl.dot(feat_hb, gl_bc, input_precision="ieee") * SCALE
        tl.store(
            grad_tail_readout + offs_h[:, None] * C + offs_c[None, :],
            grad_readout,
            mask=mask_h[:, None] & mask_c[None, :],
        )

        w_hc = tl.load(tail_readout + offs_h[:, None] * C + offs_c[None, :], mask=mask_h[:, None] & mask_c[None, :], other=0.0).to(tl.float32)
        grad_hidden_feat = tl.dot(gl_bc, tl.trans(w_hc), input_precision="ieee")
        if MODE == 3:
            mean_grad_feat = tl.sum(grad_hidden_feat, axis=1) / H
            grad_hidden_feat = grad_hidden_feat - CENTER_MIX * mean_grad_feat[:, None]
        if MODE == 7:
            mean_grad_feat = tl.sum(grad_hidden_feat, axis=1) / H
            grad_h_block = (grad_hidden_feat + CENTER_MIX * (grad_hidden_feat - mean_grad_feat[:, None]) * tl.trans(deriv_hb)) * SCALE
        else:
            grad_h_block = grad_hidden_feat * tl.trans(deriv_hb) * SCALE
        tl.store(
            grad_h_tail + offs_b[:, None] * H + offs_h[None, :],
            grad_h_block,
            mask=mask_b[:, None] & mask_h[None, :],
        )

else:
    _pairbucket_hidden_delta_kernel = None
    _paircross_readout_logits_kernel = None
    _paircross_readout_grad_kernel = None
    _paircross_readout_logits_block_kernel = None
    _paircross_readout_grad_block_kernel = None
    _paircross_readout_grad_block_atomic_kernel = None
    _paircross_readout_grad_block_cap_kernel = None
    _paircross_readout_grad_block_cap_atomic_kernel = None
    _hidden_tail_readout_vjp_kernel = None


def _pairbucket_hidden_delta_triton(
    z: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
    bucket: torch.Tensor,
    weight: torch.Tensor,
    hidden_dim: int,
    scale: float,
) -> torch.Tensor | None:
    if _pairbucket_hidden_delta_kernel is None or not z.is_cuda:
        return None
    zc = z.contiguous()
    rank = int(bucket.numel())
    if rank <= 0:
        return zc.new_zeros((int(zc.shape[0]), int(hidden_dim)))
    block_r = 1 << (rank - 1).bit_length()
    out = torch.zeros((int(zc.shape[0]), int(hidden_dim)), device=zc.device, dtype=zc.dtype)
    _pairbucket_hidden_delta_kernel[(int(zc.shape[0]),)](
        zc,
        left.contiguous(),
        right.contiguous(),
        bucket.contiguous(),
        weight.contiguous(),
        out,
        D=int(zc.shape[1]),
        R=rank,
        H=int(hidden_dim),
        SCALE=float(scale),
        BLOCK_R=block_r,
    )
    return out


def _hidden_tail_readout_vjp_triton(
    h: torch.Tensor,
    tail_readout: torch.Tensor,
    grad_logits: torch.Tensor,
    scale: float,
    mode: int,
    center_mix: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if _hidden_tail_readout_vjp_kernel is None or not h.is_cuda:
        return None
    hc = h.contiguous()
    wc = tail_readout.contiguous()
    gc = grad_logits.contiguous()
    batch = int(hc.shape[0])
    hidden = int(hc.shape[1])
    classes = int(gc.shape[1])
    if hidden <= 0 or classes <= 0:
        return hc.new_zeros(wc.shape), hc.new_zeros(hc.shape)
    block_h = 1 << (hidden - 1).bit_length()
    block_b = 1 << (batch - 1).bit_length()
    block_c = 16
    grad_tail_readout = torch.empty_like(wc)
    grad_h_tail = torch.empty_like(hc)
    _hidden_tail_readout_vjp_kernel[(triton.cdiv(hidden, block_h), triton.cdiv(classes, block_c))](
        hc,
        wc,
        gc,
        grad_tail_readout,
        grad_h_tail,
        H=hidden,
        C=classes,
        B=batch,
        SCALE=float(scale) / math.sqrt(max(1, hidden)),
        CENTER_MIX=float(center_mix),
        MODE=int(mode),
        BLOCK_H=block_h,
        BLOCK_B=block_b,
        BLOCK_C=block_c,
    )
    return grad_tail_readout, grad_h_tail


def _paircross_readout_logits_triton(
    z: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
    readout: torch.Tensor,
    scale: float,
    feature_mode: int = 0,
) -> torch.Tensor | None:
    if _paircross_readout_logits_kernel is None or not z.is_cuda:
        return None
    zc = z.contiguous()
    wc = readout.contiguous()
    rank = int(left.numel())
    classes = int(wc.shape[1])
    if rank <= 0:
        return zc.new_zeros((int(zc.shape[0]), classes))
    block_r = 1 << (rank - 1).bit_length()
    block_c = 16
    out = torch.empty((int(zc.shape[0]), classes), device=zc.device, dtype=zc.dtype)
    _paircross_readout_logits_kernel[(int(zc.shape[0]), triton.cdiv(classes, block_c))](
        zc,
        left.contiguous(),
        right.contiguous(),
        wc,
        out,
        D=int(zc.shape[1]),
        R=rank,
        C=classes,
        SCALE=float(scale) / math.sqrt(max(1, rank)),
        FEATURE_MODE=int(feature_mode),
        BLOCK_R=block_r,
        BLOCK_C=block_c,
    )
    return out


def _paircross_readout_grad_triton(
    z: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
    grad_logits: torch.Tensor,
    scale: float,
    feature_mode: int = 0,
) -> torch.Tensor | None:
    if _paircross_readout_grad_kernel is None or not z.is_cuda:
        return None
    zc = z.contiguous()
    gc = grad_logits.contiguous()
    rank = int(left.numel())
    classes = int(gc.shape[1])
    if rank <= 0:
        return zc.new_zeros((rank, classes))
    block_b = 1 << (int(zc.shape[0]) - 1).bit_length()
    out = torch.empty((rank, classes), device=zc.device, dtype=zc.dtype)
    _paircross_readout_grad_kernel[(rank, classes)](
        zc,
        left.contiguous(),
        right.contiguous(),
        gc,
        out,
        D=int(zc.shape[1]),
        R=rank,
        C=classes,
        B=int(zc.shape[0]),
        SCALE=float(scale) / math.sqrt(max(1, rank)),
        FEATURE_MODE=int(feature_mode),
        BLOCK_B=block_b,
    )
    return out


def _paircross_readout_logits_block_triton(
    z: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
    readout: torch.Tensor,
    scale: float,
    bias: torch.Tensor | None = None,
    feat_mean: torch.Tensor | None = None,
    feat_inv_std: torch.Tensor | None = None,
    block_b_override: int = 16,
    cap_value: float | None = None,
    feature_mode: int = 0,
) -> torch.Tensor | None:
    if _paircross_readout_logits_block_kernel is None or not z.is_cuda:
        return None
    zc = z.contiguous()
    wc = readout.contiguous()
    bc = bias.contiguous() if bias is not None else wc
    mc = feat_mean.contiguous() if feat_mean is not None else left
    sc = feat_inv_std.contiguous() if feat_inv_std is not None else left
    rank = int(left.numel())
    classes = int(wc.shape[1])
    batch = int(zc.shape[0])
    if rank <= 0:
        out = zc.new_zeros((batch, classes))
        if bias is not None:
            out = out + bias.view(1, -1)
        return out
    block_b = max(1, int(block_b_override))
    block_r = 1 << (rank - 1).bit_length()
    block_c = 16
    out = torch.empty((batch, classes), device=zc.device, dtype=zc.dtype)
    _paircross_readout_logits_block_kernel[(triton.cdiv(batch, block_b), triton.cdiv(classes, block_c))](
        zc,
        left.contiguous(),
        right.contiguous(),
        wc,
        mc,
        sc,
        bc,
        out,
        D=int(zc.shape[1]),
        R=rank,
        C=classes,
        B=batch,
        SCALE=float(scale) / math.sqrt(max(1, rank)),
        HAS_NORM=feat_mean is not None and feat_inv_std is not None,
        HAS_BIAS=bias is not None,
        HAS_CAP=cap_value is not None,
        CAP_VALUE=float(cap_value) if cap_value is not None else 1.0,
        FEATURE_MODE=int(feature_mode),
        BLOCK_B=block_b,
        BLOCK_R=block_r,
        BLOCK_C=block_c,
    )
    return out


def _paircross_readout_grad_block_cap_triton(
    z: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
    readout: torch.Tensor,
    grad_logits: torch.Tensor,
    scale: float,
    cap_value: float,
    feat_mean: torch.Tensor | None = None,
    feat_inv_std: torch.Tensor | None = None,
    block_r_override: int = 32,
    feature_mode: int = 0,
) -> torch.Tensor | None:
    if _paircross_readout_grad_block_cap_atomic_kernel is None or not z.is_cuda:
        return None
    zc = z.contiguous()
    wc = readout.contiguous()
    gc = grad_logits.contiguous()
    mc = feat_mean.contiguous() if feat_mean is not None else left
    sc = feat_inv_std.contiguous() if feat_inv_std is not None else left
    rank = int(left.numel())
    classes = int(gc.shape[1])
    batch = int(zc.shape[0])
    if rank <= 0:
        return zc.new_zeros((rank, classes))
    block_b = min(64, 1 << (batch - 1).bit_length())
    block_r_requested = max(1, int(block_r_override))
    block_r = 1 << (block_r_requested - 1).bit_length()
    block_rf = 1 << (rank - 1).bit_length()
    block_c = 16
    out = torch.zeros((rank, classes), device=zc.device, dtype=zc.dtype)
    _paircross_readout_grad_block_cap_atomic_kernel[(triton.cdiv(rank, block_r), triton.cdiv(classes, block_c), triton.cdiv(batch, block_b))](
        zc,
        left.contiguous(),
        right.contiguous(),
        wc,
        gc,
        mc,
        sc,
        out,
        D=int(zc.shape[1]),
        R=rank,
        C=classes,
        B=batch,
        SCALE=float(scale) / math.sqrt(max(1, rank)),
        CAP_VALUE=float(cap_value),
        HAS_NORM=feat_mean is not None and feat_inv_std is not None,
        FEATURE_MODE=int(feature_mode),
        BLOCK_B=block_b,
        BLOCK_R=block_r,
        BLOCK_RF=block_rf,
        BLOCK_C=block_c,
    )
    return out


def _paircross_readout_grad_block_triton(
    z: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
    grad_logits: torch.Tensor,
    scale: float,
    feat_mean: torch.Tensor | None = None,
    feat_inv_std: torch.Tensor | None = None,
    block_r_override: int = 32,
    feature_mode: int = 0,
) -> torch.Tensor | None:
    if _paircross_readout_grad_block_kernel is None or not z.is_cuda:
        return None
    zc = z.contiguous()
    gc = grad_logits.contiguous()
    mc = feat_mean.contiguous() if feat_mean is not None else left
    sc = feat_inv_std.contiguous() if feat_inv_std is not None else left
    rank = int(left.numel())
    classes = int(gc.shape[1])
    batch = int(zc.shape[0])
    if rank <= 0:
        return zc.new_zeros((rank, classes))
    block_b = 1 << (batch - 1).bit_length()
    block_r_requested = max(1, int(block_r_override))
    block_r = 1 << (block_r_requested - 1).bit_length()
    block_c = 16
    out = torch.empty((rank, classes), device=zc.device, dtype=zc.dtype)
    _paircross_readout_grad_block_kernel[(triton.cdiv(rank, block_r), triton.cdiv(classes, block_c))](
        zc,
        left.contiguous(),
        right.contiguous(),
        gc,
        mc,
        sc,
        out,
        D=int(zc.shape[1]),
        R=rank,
        C=classes,
        B=batch,
        SCALE=float(scale) / math.sqrt(max(1, rank)),
        HAS_NORM=feat_mean is not None and feat_inv_std is not None,
        FEATURE_MODE=int(feature_mode),
        BLOCK_B=block_b,
        BLOCK_R=block_r,
        BLOCK_C=block_c,
    )
    return out


def _paircross_readout_grad_block_atomic_triton(
    z: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
    grad_logits: torch.Tensor,
    scale: float,
    feat_mean: torch.Tensor | None = None,
    feat_inv_std: torch.Tensor | None = None,
    block_r_override: int = 64,
    block_b_override: int = 32,
    feature_mode: int = 0,
) -> torch.Tensor | None:
    if _paircross_readout_grad_block_atomic_kernel is None or not z.is_cuda:
        return None
    zc = z.contiguous()
    gc = grad_logits.contiguous()
    mc = feat_mean.contiguous() if feat_mean is not None else left
    sc = feat_inv_std.contiguous() if feat_inv_std is not None else left
    rank = int(left.numel())
    classes = int(gc.shape[1])
    batch = int(zc.shape[0])
    if rank <= 0:
        return zc.new_zeros((rank, classes))
    block_b = min(max(1, int(block_b_override)), 1 << (batch - 1).bit_length())
    block_r_requested = max(1, int(block_r_override))
    block_r = 1 << (block_r_requested - 1).bit_length()
    block_c = 16
    out = torch.zeros((rank, classes), device=zc.device, dtype=zc.dtype)
    _paircross_readout_grad_block_atomic_kernel[
        (triton.cdiv(rank, block_r), triton.cdiv(classes, block_c), triton.cdiv(batch, block_b))
    ](
        zc,
        left.contiguous(),
        right.contiguous(),
        gc,
        mc,
        sc,
        out,
        D=int(zc.shape[1]),
        R=rank,
        C=classes,
        B=batch,
        SCALE=float(scale) / math.sqrt(max(1, rank)),
        HAS_NORM=feat_mean is not None and feat_inv_std is not None,
        FEATURE_MODE=int(feature_mode),
        BLOCK_B=block_b,
        BLOCK_R=block_r,
        BLOCK_C=block_c,
    )
    return out


def _grouped_rational_forward_torch(z: torch.Tensor, numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    groups = int(numerator.shape[0])
    bsz, dim = int(z.shape[0]), int(z.shape[1])
    group_size = max(1, dim // groups)
    zg = z.reshape(bsz, groups, group_size)
    a = numerator[:, None, :]
    b = denominator[:, None, :]
    p = a[..., 5]
    p = p * zg + a[..., 4]
    p = p * zg + a[..., 3]
    p = p * zg + a[..., 2]
    p = p * zg + a[..., 1]
    p = p * zg + a[..., 0]
    ax = zg.abs()
    q = b[..., 3].abs()
    q = q * ax + b[..., 2].abs()
    q = q * ax + b[..., 1].abs()
    q = q * ax + b[..., 0].abs()
    q = q * ax + 1.0
    return (p / q.clamp_min(EPS)).reshape(bsz, dim)


def _grouped_rational_backward_torch(
    grad_out: torch.Tensor,
    z: torch.Tensor,
    numerator: torch.Tensor,
    denominator: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    groups = int(numerator.shape[0])
    bsz, dim = int(z.shape[0]), int(z.shape[1])
    group_size = max(1, dim // groups)
    zg = z.reshape(bsz, groups, group_size)
    gg = grad_out.reshape(bsz, groups, group_size)
    a = numerator[:, None, :]
    b = denominator[:, None, :]
    x2 = zg * zg
    x3 = x2 * zg
    x4 = x3 * zg
    x5 = x4 * zg
    ax = zg.abs()
    ax2 = ax * ax
    ax3 = ax2 * ax
    ax4 = ax3 * ax
    p = a[..., 0] + a[..., 1] * zg + a[..., 2] * x2 + a[..., 3] * x3 + a[..., 4] * x4 + a[..., 5] * x5
    q = 1.0 + b[..., 0].abs() * ax + b[..., 1].abs() * ax2 + b[..., 2].abs() * ax3 + b[..., 3].abs() * ax4
    r = a[..., 1] + 2.0 * a[..., 2] * zg + 3.0 * a[..., 3] * x2 + 4.0 * a[..., 4] * x3 + 5.0 * a[..., 5] * x4
    sign_x = torch.where(zg < 0.0, torch.full_like(zg, -1.0), torch.ones_like(zg))
    s = sign_x * (b[..., 0].abs() + 2.0 * b[..., 1].abs() * ax + 3.0 * b[..., 2].abs() * ax2 + 4.0 * b[..., 3].abs() * ax3)
    mpq2 = -p / q.square().clamp_min(EPS)
    dz = ((r / q.clamp_min(EPS)) + s * mpq2) * gg
    inv_q_g = gg / q.clamp_min(EPS)
    dn = torch.stack(
        [
            inv_q_g.sum(dim=(0, 2)),
            (zg * inv_q_g).sum(dim=(0, 2)),
            (x2 * inv_q_g).sum(dim=(0, 2)),
            (x3 * inv_q_g).sum(dim=(0, 2)),
            (x4 * inv_q_g).sum(dim=(0, 2)),
            (x5 * inv_q_g).sum(dim=(0, 2)),
        ],
        dim=1,
    )
    sign_b = torch.where(denominator < 0.0, torch.full_like(denominator, -1.0), torch.ones_like(denominator))
    db = torch.stack(
        [
            (mpq2 * ax * gg).sum(dim=(0, 2)),
            (mpq2 * ax2 * gg).sum(dim=(0, 2)),
            (mpq2 * ax3 * gg).sum(dim=(0, 2)),
            (mpq2 * ax4 * gg).sum(dim=(0, 2)),
        ],
        dim=1,
    ) * sign_b
    return dz.reshape(bsz, dim), dn, db


class PrimitiveKAN(nn.Module):
    """Two-layer strict edge-basis KAN with fixed basis centers/scales."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        spec: PrimitiveSpec,
        x_for_stats: torch.Tensor,
        seed: int,
        device: torch.device,
        param_budget: int,
    ) -> None:
        super().__init__()
        self.spec = spec
        hidden_dim = int(spec.hidden_dim) if int(spec.hidden_dim) > 0 else matched_hidden(param_budget, input_dim, output_dim, spec.k)
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.hidden_dim = int(hidden_dim)
        self.k = int(spec.k)
        xs = x_for_stats[: min(4096, int(x_for_stats.shape[0]))].to(device=device, dtype=torch.float32)
        mu = xs.mean(dim=0)
        std = xs.std(dim=0).clamp_min(1.0e-3)
        self.register_buffer("mu", mu)
        self.register_buffer("std", std)
        if spec.basis_name in {"relu_hinge", "rswaf_hinge"}:
            centers = torch.linspace(-1.0, 1.0, max(1, self.k - 1), device=device)
        else:
            centers = torch.linspace(-1.0, 1.0, self.k, device=device)
        self.register_buffer("centers", centers)
        self.register_buffer("scales", torch.tensor([max(0.2, 2.0 / max(1, self.k - 1))], device=device))
        gen = torch.Generator(device=device).manual_seed(int(seed))
        fan1 = math.sqrt(max(1, self.input_dim * self.k))
        fan2 = math.sqrt(max(1, self.hidden_dim * self.k))
        self.w1 = nn.Parameter(torch.randn(self.input_dim, self.hidden_dim, self.k, device=device, generator=gen) / fan1)
        self.w2 = nn.Parameter(torch.randn(self.hidden_dim, self.output_dim, self.k, device=device, generator=gen) / fan2)
        variant_lower = str(spec.init_variant).lower()
        self.linear_residual_enabled = "linearres" in variant_lower
        self.linear_residual_raw_scale = "linearraw" in variant_lower
        self.cheby_paircross_enabled = (
            spec.basis_name == "chebyshev"
            and int(spec.k) == 3
            and "paircross" in variant_lower
        )
        self.cheby_cross_rank = 0
        if self.cheby_paircross_enabled:
            cross_rank = 16
            for token, value in (("crossr8", 8), ("crossr16", 16), ("crossr24", 24), ("crossr32", 32), ("crossr48", 48)):
                if token in variant_lower:
                    cross_rank = value
            self.cheby_cross_rank = min(int(cross_rank), max(0, self.hidden_dim // 2))
            self.cheby_cross_readout = nn.Parameter(
                torch.randn(self.cheby_cross_rank, self.output_dim, device=device, generator=gen)
                * (0.10 / math.sqrt(max(1, self.cheby_cross_rank)))
            )
        self.cheby_input_cross_enabled = (
            spec.basis_name == "chebyshev"
            and int(spec.k) == 3
            and "inputcross" in variant_lower
        )
        self.cheby_input_localrot_enabled = self.cheby_input_cross_enabled and "localrot" in variant_lower
        self.cheby_input_localrot2_enabled = self.cheby_input_cross_enabled and "localrot2" in variant_lower
        self.cheby_input_pair_rank = 0
        self.cheby_input_proj_rank = 0
        if self.cheby_input_cross_enabled:
            local_rank = 4
            proj_rank = 8
            for token, value in (("localr2", 2), ("localr4", 4), ("localr8", 8), ("localr16", 16)):
                if token in variant_lower:
                    local_rank = value
            for token, value in (
                ("projr4", 4),
                ("projr8", 8),
                ("projr16", 16),
                ("projr24", 24),
                ("projr32", 32),
                ("projr48", 48),
                ("projr64", 64),
                ("projr96", 96),
                ("projr112", 112),
                ("projr128", 128),
            ):
                if token in variant_lower:
                    proj_rank = value
            self.cheby_input_pair_rank = min(int(local_rank), max(0, self.input_dim // 2))
            self.cheby_input_proj_rank = int(proj_rank)
            if self.cheby_input_proj_rank > 0:
                left = torch.randn(self.input_dim, self.cheby_input_proj_rank, device=device, generator=gen)
                left = left / left.norm(dim=0, keepdim=True).clamp_min(1.0e-6)
                if "projsq" in variant_lower:
                    right = left.clone()
                else:
                    right = torch.randn(self.input_dim, self.cheby_input_proj_rank, device=device, generator=gen)
                    right = right / right.norm(dim=0, keepdim=True).clamp_min(1.0e-6)
                self.register_buffer("cheby_input_proj_left", left)
                self.register_buffer("cheby_input_proj_right", right)
            total_input_cross = self.cheby_input_pair_rank + self.cheby_input_proj_rank
            if self.cheby_input_localrot_enabled:
                total_input_cross += self.cheby_input_pair_rank
            if self.cheby_input_localrot2_enabled:
                total_input_cross += self.cheby_input_pair_rank
            self.cheby_input_cross_readout = nn.Parameter(
                torch.randn(total_input_cross, self.output_dim, device=device, generator=gen)
                * (0.05 / math.sqrt(max(1, total_input_cross)))
            )
        if self.linear_residual_enabled:
            linear_scale = 0.10
            if "linearres025" in variant_lower:
                linear_scale = 0.25
            elif "linearres050" in variant_lower:
                linear_scale = 0.50
            self.linear_readout = nn.Parameter(
                torch.randn(self.input_dim, self.output_dim, device=device, generator=gen)
                * (linear_scale / math.sqrt(max(1, self.input_dim)))
            )
        if spec.init_variant == "fan_scale_repair":
            with torch.no_grad():
                self.w1.mul_(math.sqrt(max(1, self.input_dim)))
                self.w2.mul_(math.sqrt(max(1, self.hidden_dim)))
        if spec.init_variant == "identity_residual_scale":
            with torch.no_grad():
                self.w1.mul_(0.05)
                identity_channel = 1 if self.k > 1 else 0
                diag_cols = min(self.input_dim, self.hidden_dim)
                for idx in range(diag_cols):
                    self.w1[idx, idx, identity_channel] = math.sqrt(float(self.input_dim))
                self.w2.mul_(0.25)
        if spec.init_variant in {"signed_pair_linear", "signed_pair_random_linear"}:
            with torch.no_grad():
                self.w1.mul_(0.05)
                col = 0
                pair_limit = self.hidden_dim if spec.init_variant == "signed_pair_linear" else max(2, self.hidden_dim // 2)
                for start in range(0, self.input_dim - 1, 2):
                    if col + 1 >= pair_limit:
                        break
                    self.w1[start, col, 0] = math.sqrt(self.input_dim / 2.0)
                    self.w1[start + 1, col, 0] = math.sqrt(self.input_dim / 2.0)
                    col += 1
                    self.w1[start, col, 0] = math.sqrt(self.input_dim / 2.0)
                    self.w1[start + 1, col, 0] = -math.sqrt(self.input_dim / 2.0)
                    col += 1
                if spec.init_variant == "signed_pair_random_linear" and col < self.hidden_dim:
                    for idx in range(col, self.hidden_dim):
                        vec = torch.randn(self.input_dim, device=device, generator=gen)
                        vec = vec / vec.norm().clamp_min(1.0e-6)
                        self.w1[:, idx, 0] = vec * math.sqrt(float(self.input_dim))
                elif col < self.hidden_dim:
                    eye_cols = min(self.input_dim, self.hidden_dim - col)
                    for idx in range(eye_cols):
                        self.w1[idx, col + idx, 0] = math.sqrt(float(self.input_dim))

    @property
    def edge_param_count(self) -> int:
        extra = int(self.linear_readout.numel()) if self.linear_residual_enabled else 0
        if self.cheby_paircross_enabled:
            extra += int(self.cheby_cross_readout.numel())
        if self.cheby_input_cross_enabled:
            extra += int(self.cheby_input_cross_readout.numel())
        return int(self.w1.numel() + self.w2.numel() + extra)

    def _norm_input(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh((x - self.mu) / self.std)

    def layer1_basis(self, x: torch.Tensor) -> torch.Tensor:
        return _basis_eval(self._norm_input(x), self.spec.basis_name, self.k, self.centers, self.scales)

    def hidden(self, x: torch.Tensor) -> torch.Tensor:
        z = self._norm_input(x)
        if int(self.spec.uses_dense_basis_tensor) == 0:
            h = _stream_mix(z, self.w1, self.spec.basis_name, self.k, self.centers, self.scales) / math.sqrt(max(1, self.input_dim))
        else:
            b1 = _basis_eval(z, self.spec.basis_name, self.k, self.centers, self.scales)
            h = torch.einsum("bdk,dhk->bh", b1, self.w1) / math.sqrt(max(1, self.input_dim))
        return torch.tanh(h)

    def layer2_basis(self, h: torch.Tensor) -> torch.Tensor:
        return _basis_eval(h, self.spec.basis_name, self.k, self.centers, self.scales)

    def _cheby_paircross_features(self, h: torch.Tensor) -> torch.Tensor:
        rank = int(getattr(self, "cheby_cross_rank", 0))
        if rank <= 0:
            return h.new_zeros((int(h.shape[0]), 0))
        left = h[:, : 2 * rank : 2]
        right = h[:, 1 : 2 * rank : 2]
        return (left * right) / math.sqrt(max(1, rank))

    def _cheby_input_cross_features(self, z: torch.Tensor) -> torch.Tensor:
        if not self.cheby_input_cross_enabled:
            return z.new_zeros((int(z.shape[0]), 0))
        feats = []
        local_rank = int(getattr(self, "cheby_input_pair_rank", 0))
        if local_rank > 0:
            left_local = z[:, : 2 * local_rank : 2]
            right_local = z[:, 1 : 2 * local_rank : 2]
            feats.append(left_local * right_local)
            if getattr(self, "cheby_input_localrot_enabled", False):
                feats.append(0.5 * (left_local + right_local).square())
            if getattr(self, "cheby_input_localrot2_enabled", False):
                feats.append(0.5 * (left_local - right_local).square())
        proj_rank = int(getattr(self, "cheby_input_proj_rank", 0))
        if proj_rank > 0:
            left = z @ self.cheby_input_proj_left
            right = z @ self.cheby_input_proj_right
            feats.append(left * right)
        if not feats:
            return z.new_zeros((int(z.shape[0]), 0))
        out = torch.cat(feats, dim=1)
        return out / math.sqrt(max(1, int(out.shape[1])))

    def _linear_residual_denominator(self) -> float:
        return 1.0 if bool(getattr(self, "linear_residual_raw_scale", False)) else math.sqrt(max(1, self.input_dim))

    def frozen_readout_features(self, x: torch.Tensor) -> torch.Tensor:
        z = self._norm_input(x)
        h = self.hidden(x)
        b2 = self.layer2_basis(h)
        feats = b2.reshape(int(x.shape[0]), -1)
        if self.cheby_paircross_enabled:
            feats = torch.cat([feats, self._cheby_paircross_features(h)], dim=1)
        if self.cheby_input_cross_enabled:
            feats = torch.cat([feats, self._cheby_input_cross_features(z)], dim=1)
        if self.linear_residual_enabled:
            feats = torch.cat([feats, z], dim=1)
        return feats

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self._norm_input(x)
        h = self.hidden(x)
        if int(self.spec.uses_dense_basis_tensor) == 0:
            logits = _stream_mix(h, self.w2, self.spec.basis_name, self.k, self.centers, self.scales) / math.sqrt(max(1, self.hidden_dim))
        else:
            b2 = self.layer2_basis(h)
            logits = torch.einsum("bhk,hck->bc", b2, self.w2) / math.sqrt(max(1, self.hidden_dim))
        if self.cheby_paircross_enabled:
            logits = logits + self._cheby_paircross_features(h) @ self.cheby_cross_readout
        if self.cheby_input_cross_enabled:
            logits = logits + self._cheby_input_cross_features(z) @ self.cheby_input_cross_readout
        if self.linear_residual_enabled:
            logits = logits + (z @ self.linear_readout) / self._linear_residual_denominator()
        return logits

    def basis_diagnostics(self, x: torch.Tensor) -> Dict[str, float]:
        with torch.no_grad():
            z = self._norm_input(x)
            feats = self.frozen_readout_features(x).float()
            energy = feats.square().mean(dim=0)
            prob = energy / energy.sum().clamp_min(EPS)
            entropy = -(prob * (prob + EPS).log()).sum() / math.log(max(2, int(prob.numel())))
            dead = (energy < 1.0e-8).float().mean()
            centered = feats - feats.mean(dim=0, keepdim=True)
            try:
                s = torch.linalg.svdvals(centered[: min(256, int(centered.shape[0]))])
                rank = (s.square().sum().square() / s.pow(4).sum().clamp_min(EPS)).item()
                cond = (s.max() / s[s > 1.0e-7].min()).item() if bool((s > 1.0e-7).any()) else float("inf")
            except RuntimeError:
                rank = 0.0
                cond = float("inf")
            return {
                "basis_entropy": float(entropy.item()),
                "dead_basis_fraction": float(dead.item()),
                "basis_effective_rank": float(rank),
                "basis_condition_proxy": float(cond),
                "basis_output_norm_p95": float(torch.quantile(feats.norm(dim=1), 0.95).item()),
            }

    def manual_kernel_available(self) -> bool:
        return True

    def _manual_fourier_k2_memory_planned(self) -> bool:
        return self.spec.basis_name == "fourier_lowfreq" and int(self.k) == 2

    def _manual_fourier_k3_memory_planned(self) -> bool:
        return self.spec.basis_name == "fourier_lowfreq" and int(self.k) == 3

    def _manual_fourier_k4_memory_planned(self) -> bool:
        return self.spec.basis_name == "fourier_lowfreq" and int(self.k) == 4

    def _manual_fourier_k2_flat_gemm(self) -> bool:
        return self._manual_fourier_k2_memory_planned() and self.spec.init_variant == "fourier_k2_flatgemm_cache"

    def _manual_fourier_k2_flat_gemm_recompute(self) -> bool:
        return self._manual_fourier_k2_memory_planned() and self.spec.init_variant == "fourier_k2_flatgemm_recompute"

    def _manual_fourier_k2_triton_l3(self) -> bool:
        return self._manual_fourier_k2_memory_planned() and self.spec.init_variant == "fourier_k2_triton_l3"

    def _manual_fourier_k2_triton_l3_blockh(self) -> bool:
        return self._manual_fourier_k2_memory_planned() and self.spec.init_variant == "fourier_k2_triton_l3_blockh"

    def _manual_fourier_k2_triton_l3_matmul(self) -> bool:
        return self._manual_fourier_k2_memory_planned() and self.spec.init_variant == "fourier_k2_triton_l3_matmul"

    def _manual_fourier_k3_triton_l3_matmul(self) -> bool:
        return self._manual_fourier_k3_memory_planned() and self.spec.init_variant == "fourier_k3_triton_l3_matmul"

    def _manual_fourier_k4_triton_l3_matmul(self) -> bool:
        return self._manual_fourier_k4_memory_planned() and self.spec.init_variant == "fourier_k4_triton_l3_matmul"

    def _manual_fourier_k4_linearres_triton_l3_matmul(self) -> bool:
        return self._manual_fourier_k4_memory_planned() and str(self.spec.init_variant).startswith("fourier_k4_linearres_triton_l3_matmul")

    def _manual_fourier_k4_linearres_gemm_l3_matmul(self) -> bool:
        return self._manual_fourier_k4_memory_planned() and str(self.spec.init_variant).startswith("fourier_k4_linearres_gemm_l3_matmul")

    def _manual_cheby_k3_triton_l3_matmul(self) -> bool:
        return self.spec.basis_name == "chebyshev" and int(self.k) == 3 and self.spec.init_variant == "cheby_k3_triton_l3_matmul"

    def _manual_cheby_k3_triton_l3_gradbuf(self) -> bool:
        return self.spec.basis_name == "chebyshev" and int(self.k) == 3 and self.spec.init_variant == "cheby_k3_triton_l3_gradbuf"

    def _manual_cheby_k3_paircross_l3_gradbuf(self) -> bool:
        return self.spec.basis_name == "chebyshev" and int(self.k) == 3 and str(self.spec.init_variant).startswith("cheby_k3_paircross")

    def _manual_cheby_k3_inputcross_l3_gradbuf(self) -> bool:
        return self.spec.basis_name == "chebyshev" and int(self.k) == 3 and str(self.spec.init_variant).startswith("cheby_k3_inputcross")

    def _manual_cheby_k4_triton_l3_matmul(self) -> bool:
        return self.spec.basis_name == "chebyshev" and int(self.k) == 4 and self.spec.init_variant == "cheby_k4_triton_l3_matmul"

    def _manual_rbf_triton_l3_matmul(self) -> bool:
        return self.spec.basis_name in {"compact_rbf", "fastkan_rbf"} and int(self.k) in {2, 4} and str(self.spec.init_variant).startswith("rbf_k")

    def _manual_hat_wavelet_triton_l3_matmul(self) -> bool:
        return self.spec.basis_name == "hat_wavelet" and int(self.k) == 4 and self.spec.init_variant == "hat_wavelet_k4_triton_l3_matmul"

    def _manual_stream_recompute(self) -> bool:
        return int(self.spec.uses_dense_basis_tensor) == 0 and self.spec.basis_name in {
            "compact_rbf",
            "fastkan_rbf",
            "hat_wavelet",
        }

    def manual_kernel_variant(self) -> str:
        if self._manual_rbf_triton_l3_matmul():
            return f"rbf_k{int(self.k)}_triton_l3_matmul"
        if self._manual_hat_wavelet_triton_l3_matmul():
            return "hat_wavelet_k4_triton_l3_matmul"
        if self._manual_cheby_k4_triton_l3_matmul():
            return "cheby_k4_triton_l3_matmul"
        if self._manual_cheby_k3_paircross_l3_gradbuf():
            if self.cheby_input_cross_enabled:
                if self.linear_residual_enabled:
                    return "cheby_k3_triton_l3_paircross_inputcross_linearres_gemm_gradbuf"
                return "cheby_k3_triton_l3_paircross_inputcross_gemm_gradbuf"
            return "cheby_k3_triton_l3_paircross_gradbuf"
        if self._manual_cheby_k3_inputcross_l3_gradbuf():
            if self.linear_residual_enabled:
                return "cheby_k3_triton_l3_inputcross_linearres_gemm_gradbuf"
            return "cheby_k3_triton_l3_inputcross_gemm_gradbuf"
        if self._manual_cheby_k3_triton_l3_gradbuf():
            return "cheby_k3_triton_l3_gradbuf"
        if self._manual_cheby_k3_triton_l3_matmul():
            return "cheby_k3_triton_l3_matmul"
        if self._manual_fourier_k4_linearres_gemm_l3_matmul():
            return "fourier_k4_linearres_gemm_l3_matmul"
        if self._manual_fourier_k4_linearres_triton_l3_matmul():
            return "fourier_k4_linearres_triton_l3_matmul"
        if self._manual_fourier_k4_triton_l3_matmul():
            return "fourier_k4_triton_l3_matmul"
        if self._manual_fourier_k3_triton_l3_matmul():
            return "fourier_k3_triton_l3_matmul"
        if self._manual_fourier_k2_triton_l3_matmul():
            return "fourier_k2_triton_l3_matmul"
        if self._manual_fourier_k2_triton_l3_blockh():
            return "fourier_k2_triton_l3_blockh"
        if self._manual_fourier_k2_triton_l3():
            return "fourier_k2_triton_l3"
        if self._manual_fourier_k2_flat_gemm_recompute():
            return "fourier_k2_flat_gemm_recompute"
        if self._manual_fourier_k2_flat_gemm():
            return "fourier_k2_flat_gemm_cache"
        if self._manual_fourier_k2_memory_planned():
            return "fourier_k2_memory_planned_recompute"
        if self._manual_stream_recompute():
            return f"{self.spec.basis_name}_stream_recompute"
        return "generic_fixed_basis_dense_cache"

    def manual_ce_forward_cache(self, x: torch.Tensor):
        with torch.no_grad():
            if self._manual_rbf_triton_l3_matmul():
                from dgkan.kernels import fused_rbf

                logits, h = fused_rbf.forward_matmul(self, x)
                return logits, (f"rbf_k{int(self.k)}_triton_l3_matmul", x, h)
            if self._manual_hat_wavelet_triton_l3_matmul():
                from dgkan.kernels import fused_hat_wavelet

                logits, h = fused_hat_wavelet.forward_matmul(self, x)
                return logits, ("hat_wavelet_k4_triton_l3_matmul", x, h)
            if self._manual_fourier_k4_linearres_triton_l3_matmul():
                from dgkan.kernels import fused_fourier_k2

                logits, h = fused_fourier_k2.forward_matmul_k4_linearres(self, x)
                return logits, ("fourier_k4_linearres_triton_l3_matmul", x, h)
            if self._manual_cheby_k3_triton_l3_matmul():
                from dgkan.kernels import fused_chebyshev_k3

                logits, h = fused_chebyshev_k3.forward_matmul(self, x)
                return logits, ("cheby_k3_triton_l3_matmul", x, h)
            if self._manual_cheby_k3_triton_l3_gradbuf():
                from dgkan.kernels import fused_chebyshev_k3

                logits, h = fused_chebyshev_k3.forward_matmul(self, x)
                return logits, ("cheby_k3_triton_l3_gradbuf", x, h)
            if self._manual_cheby_k3_paircross_l3_gradbuf():
                from dgkan.kernels import fused_chebyshev_k3

                logits, h = fused_chebyshev_k3.forward_matmul_paircross(self, x)
                if self.cheby_input_cross_enabled:
                    z = self._norm_input(x)
                    input_feats = self._cheby_input_cross_features(z)
                    logits = logits + input_feats @ self.cheby_input_cross_readout
                    if self.linear_residual_enabled:
                        logits = logits + (z @ self.linear_readout) / self._linear_residual_denominator()
                        return logits, (
                            "cheby_k3_triton_l3_paircross_inputcross_linearres_gemm_gradbuf",
                            x,
                            h,
                            input_feats,
                            z,
                        )
                    return logits, ("cheby_k3_triton_l3_paircross_inputcross_gemm_gradbuf", x, h, input_feats)
                return logits, ("cheby_k3_triton_l3_paircross_gradbuf", x, h)
            if self._manual_cheby_k3_inputcross_l3_gradbuf():
                from dgkan.kernels import fused_chebyshev_k3

                logits, h = fused_chebyshev_k3.forward_matmul(self, x)
                z = self._norm_input(x)
                input_feats = self._cheby_input_cross_features(z)
                logits = logits + input_feats @ self.cheby_input_cross_readout
                if self.linear_residual_enabled:
                    logits = logits + (z @ self.linear_readout) / self._linear_residual_denominator()
                    return logits, ("cheby_k3_triton_l3_inputcross_linearres_gemm_gradbuf", x, h, input_feats, z)
                return logits, ("cheby_k3_triton_l3_inputcross_gemm_gradbuf", x, h, input_feats)
            if self._manual_cheby_k4_triton_l3_matmul():
                from dgkan.kernels import fused_chebyshev_k3

                logits, h = fused_chebyshev_k3.forward_matmul_k4(self, x)
                return logits, ("cheby_k4_triton_l3_matmul", x, h)
            if self._manual_fourier_k4_linearres_gemm_l3_matmul():
                from dgkan.kernels import fused_fourier_k2

                logits, h = fused_fourier_k2.forward_matmul_k4(self, x)
                z = self._norm_input(x)
                logits = logits + (z @ self.linear_readout) / self._linear_residual_denominator()
                return logits, ("fourier_k4_linearres_gemm_l3_matmul", x, h, z)
            if self._manual_fourier_k4_triton_l3_matmul():
                from dgkan.kernels import fused_fourier_k2

                logits, h = fused_fourier_k2.forward_matmul_k4(self, x)
                return logits, ("fourier_k4_triton_l3_matmul", x, h)
            if self._manual_fourier_k3_triton_l3_matmul():
                from dgkan.kernels import fused_fourier_k2

                logits, h = fused_fourier_k2.forward_matmul_k3(self, x)
                return logits, ("fourier_k3_triton_l3_matmul", x, h)
            if self._manual_fourier_k2_triton_l3_matmul():
                from dgkan.kernels import fused_fourier_k2

                logits, h = fused_fourier_k2.forward_matmul(self, x)
                return logits, ("fourier_k2_triton_l3_matmul", x, h)
            if self._manual_fourier_k2_triton_l3_blockh():
                from dgkan.kernels import fused_fourier_k2

                logits, h = fused_fourier_k2.forward_blockh(self, x)
                return logits, ("fourier_k2_triton_l3_blockh", x, h)
            if self._manual_fourier_k2_triton_l3():
                from dgkan.kernels import fused_fourier_k2

                logits, h = fused_fourier_k2.forward(self, x)
                return logits, ("fourier_k2_triton_l3", x, h)
            z = self._norm_input(x)
            if self._manual_fourier_k2_flat_gemm_recompute():
                inv_sqrt2 = 1.0 / math.sqrt(2.0)
                sin_z = torch.sin(math.pi * z)
                b1_flat = torch.cat([z, sin_z], dim=1)
                w1_flat = torch.cat([self.w1[:, :, 0], self.w1[:, :, 1]], dim=0)
                sqrt_d = math.sqrt(max(1, self.input_dim))
                pre_h = (b1_flat @ w1_flat) * inv_sqrt2 / sqrt_d
                del b1_flat, w1_flat, sin_z
                h = torch.tanh(pre_h)
                sin_h = torch.sin(math.pi * h)
                b2_flat = torch.cat([h, sin_h], dim=1)
                w2_flat = torch.cat([self.w2[:, :, 0], self.w2[:, :, 1]], dim=0)
                sqrt_h = math.sqrt(max(1, self.hidden_dim))
                logits = (b2_flat @ w2_flat) * inv_sqrt2 / sqrt_h
                del b2_flat, w2_flat, sin_h, pre_h
                return logits, ("fourier_k2_flat_gemm_recompute", z, h)
            if self._manual_fourier_k2_flat_gemm():
                inv_sqrt2 = 1.0 / math.sqrt(2.0)
                sin_z = torch.sin(math.pi * z)
                b1_flat = torch.cat([z, sin_z], dim=1)
                w1_flat = torch.cat([self.w1[:, :, 0], self.w1[:, :, 1]], dim=0)
                sqrt_d = math.sqrt(max(1, self.input_dim))
                pre_h = (b1_flat @ w1_flat) * inv_sqrt2 / sqrt_d
                h = torch.tanh(pre_h)
                sin_h = torch.sin(math.pi * h)
                b2_flat = torch.cat([h, sin_h], dim=1)
                w2_flat = torch.cat([self.w2[:, :, 0], self.w2[:, :, 1]], dim=0)
                sqrt_h = math.sqrt(max(1, self.hidden_dim))
                logits = (b2_flat @ w2_flat) * inv_sqrt2 / sqrt_h
                return logits, ("fourier_k2_flat_gemm_cache", b1_flat, h)
            if self._manual_fourier_k2_memory_planned():
                inv_sqrt2 = 1.0 / math.sqrt(2.0)
                sin_z = torch.sin(math.pi * z)
                sqrt_d = math.sqrt(max(1, self.input_dim))
                pre_h = ((z @ self.w1[:, :, 0]) + (sin_z @ self.w1[:, :, 1])) * inv_sqrt2 / sqrt_d
                h = torch.tanh(pre_h)
                sin_h = torch.sin(math.pi * h)
                sqrt_h = math.sqrt(max(1, self.hidden_dim))
                logits = ((h @ self.w2[:, :, 0]) + (sin_h @ self.w2[:, :, 1])) * inv_sqrt2 / sqrt_h
                return logits, ("fourier_k2_memory_planned_recompute", z, h)
            if self._manual_stream_recompute():
                sqrt_d = math.sqrt(max(1, self.input_dim))
                pre_h = _stream_mix(z, self.w1, self.spec.basis_name, self.k, self.centers, self.scales) / sqrt_d
                h = torch.tanh(pre_h)
                sqrt_h = math.sqrt(max(1, self.hidden_dim))
                logits = _stream_mix(h, self.w2, self.spec.basis_name, self.k, self.centers, self.scales) / sqrt_h
                return logits, (f"{self.spec.basis_name}_stream_recompute", z, h)
            b1 = _basis_eval(z, self.spec.basis_name, self.k, self.centers, self.scales)
            pre_h = torch.einsum("bdk,dhk->bh", b1, self.w1) / math.sqrt(max(1, self.input_dim))
            h = torch.tanh(pre_h)
            b2 = _basis_eval(h, self.spec.basis_name, self.k, self.centers, self.scales)
            db2 = _basis_derivative(h, self.spec.basis_name, self.k, self.centers, self.scales)
            logits = torch.einsum("bhk,hck->bc", b2, self.w2) / math.sqrt(max(1, self.hidden_dim))
            return logits, (b1, pre_h, h, b2, db2)

    def manual_ce_backward_from_cache(self, logits: torch.Tensor, cache, y: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            probs = torch.softmax(logits, dim=1)
            loss = F.cross_entropy(logits, y)
            grad_logits = probs
            grad_logits[torch.arange(int(y.numel()), device=y.device), y] -= 1.0
            grad_logits = grad_logits / float(max(1, int(y.numel())))
            if cache and isinstance(cache[0], str) and cache[0] == "fourier_k4_linearres_triton_l3_matmul":
                from dgkan.kernels import fused_fourier_k2

                _variant, x, h = cache
                return fused_fourier_k2.backward_k4_linearres(self, x, y, logits, h)
            if cache and isinstance(cache[0], str) and cache[0] == "cheby_k3_triton_l3_matmul":
                from dgkan.kernels import fused_chebyshev_k3

                _variant, x, h = cache
                return fused_chebyshev_k3.backward(self, x, y, logits, h)
            if cache and isinstance(cache[0], str) and cache[0] == "cheby_k3_triton_l3_gradbuf":
                from dgkan.kernels import fused_chebyshev_k3

                _variant, x, h = cache
                return fused_chebyshev_k3.backward_gradbuf(self, x, y, logits, h)
            if cache and isinstance(cache[0], str) and cache[0] == "cheby_k3_triton_l3_paircross_gradbuf":
                from dgkan.kernels import fused_chebyshev_k3

                _variant, x, h = cache
                return fused_chebyshev_k3.backward_paircross_gradbuf(self, x, y, logits, h)
            if cache and isinstance(cache[0], str) and cache[0] == "cheby_k3_triton_l3_paircross_inputcross_gemm_gradbuf":
                from dgkan.kernels import fused_chebyshev_k3

                _variant, x, h, input_feats = cache
                loss = fused_chebyshev_k3.backward_paircross_gradbuf(self, x, y, logits, h)
                self.cheby_input_cross_readout.grad = (input_feats.transpose(0, 1) @ grad_logits)
                return loss
            if cache and isinstance(cache[0], str) and cache[0] == "cheby_k3_triton_l3_paircross_inputcross_linearres_gemm_gradbuf":
                from dgkan.kernels import fused_chebyshev_k3

                _variant, x, h, input_feats, z = cache
                loss = fused_chebyshev_k3.backward_paircross_gradbuf(self, x, y, logits, h)
                self.cheby_input_cross_readout.grad = (input_feats.transpose(0, 1) @ grad_logits)
                self.linear_readout.grad = (z.transpose(0, 1) @ grad_logits) / self._linear_residual_denominator()
                return loss
            if cache and isinstance(cache[0], str) and cache[0] == "cheby_k3_triton_l3_inputcross_gemm_gradbuf":
                from dgkan.kernels import fused_chebyshev_k3

                _variant, x, h, input_feats = cache
                loss = fused_chebyshev_k3.backward_gradbuf(self, x, y, logits, h)
                self.cheby_input_cross_readout.grad = (input_feats.transpose(0, 1) @ grad_logits)
                return loss
            if cache and isinstance(cache[0], str) and cache[0] == "cheby_k3_triton_l3_inputcross_linearres_gemm_gradbuf":
                from dgkan.kernels import fused_chebyshev_k3

                _variant, x, h, input_feats, z = cache
                loss = fused_chebyshev_k3.backward_gradbuf(self, x, y, logits, h)
                self.cheby_input_cross_readout.grad = (input_feats.transpose(0, 1) @ grad_logits)
                self.linear_readout.grad = (z.transpose(0, 1) @ grad_logits) / self._linear_residual_denominator()
                return loss
            if cache and isinstance(cache[0], str) and cache[0] == "cheby_k4_triton_l3_matmul":
                from dgkan.kernels import fused_chebyshev_k3

                _variant, x, h = cache
                return fused_chebyshev_k3.backward_k4(self, x, y, logits, h)
            if cache and isinstance(cache[0], str) and cache[0] == "fourier_k4_linearres_gemm_l3_matmul":
                from dgkan.kernels import fused_fourier_k2

                _variant, x, h, z = cache
                loss = fused_fourier_k2.backward_k4(self, x, y, logits, h)
                self.linear_readout.grad = (z.transpose(0, 1) @ grad_logits) / self._linear_residual_denominator()
                return loss
            if cache and isinstance(cache[0], str) and cache[0] == "fourier_k4_triton_l3_matmul":
                from dgkan.kernels import fused_fourier_k2

                _variant, x, h = cache
                return fused_fourier_k2.backward_k4(self, x, y, logits, h)
            if cache and isinstance(cache[0], str) and cache[0] == "fourier_k3_triton_l3_matmul":
                from dgkan.kernels import fused_fourier_k2

                _variant, x, h = cache
                return fused_fourier_k2.backward_k3(self, x, y, logits, h)
            if cache and isinstance(cache[0], str) and cache[0] in {"fourier_k2_triton_l3", "fourier_k2_triton_l3_blockh", "fourier_k2_triton_l3_matmul"}:
                from dgkan.kernels import fused_fourier_k2

                _variant, x, h = cache
                return fused_fourier_k2.backward(self, x, y, logits, h)
            if cache and isinstance(cache[0], str) and cache[0] == "fourier_k2_flat_gemm_recompute":
                _variant, z, h = cache
                inv_sqrt2 = 1.0 / math.sqrt(2.0)
                sqrt_h = math.sqrt(max(1, self.hidden_dim))
                sqrt_d = math.sqrt(max(1, self.input_dim))
                sin_h = torch.sin(math.pi * h)
                b2_flat = torch.cat([h, sin_h], dim=1)
                w2_flat = torch.cat([self.w2[:, :, 0], self.w2[:, :, 1]], dim=0)
                grad_w2_flat = (b2_flat.transpose(0, 1) @ grad_logits) * inv_sqrt2 / sqrt_h
                grad_b2_flat = (grad_logits @ w2_flat.transpose(0, 1)) * inv_sqrt2 / sqrt_h
                del b2_flat, w2_flat, sin_h
                grad_w2 = torch.empty_like(self.w2)
                grad_w2[:, :, 0] = grad_w2_flat[: self.hidden_dim, :]
                grad_w2[:, :, 1] = grad_w2_flat[self.hidden_dim :, :]
                self.w2.grad = grad_w2
                grad_h = grad_b2_flat[:, : self.hidden_dim] + grad_b2_flat[:, self.hidden_dim :] * (math.pi * torch.cos(math.pi * h))
                del grad_b2_flat, grad_w2_flat
                grad_pre = grad_h * (1.0 - h.square())
                sin_z = torch.sin(math.pi * z)
                b1_flat = torch.cat([z, sin_z], dim=1)
                grad_w1_flat = (b1_flat.transpose(0, 1) @ grad_pre) * inv_sqrt2 / sqrt_d
                del b1_flat, sin_z
                grad_w1 = torch.empty_like(self.w1)
                grad_w1[:, :, 0] = grad_w1_flat[: self.input_dim, :]
                grad_w1[:, :, 1] = grad_w1_flat[self.input_dim :, :]
                self.w1.grad = grad_w1
                return loss
            if cache and isinstance(cache[0], str) and cache[0] == "fourier_k2_flat_gemm_cache":
                _variant, b1_flat, h = cache
                inv_sqrt2 = 1.0 / math.sqrt(2.0)
                sqrt_h = math.sqrt(max(1, self.hidden_dim))
                sqrt_d = math.sqrt(max(1, self.input_dim))
                sin_h = torch.sin(math.pi * h)
                b2_flat = torch.cat([h, sin_h], dim=1)
                w2_flat = torch.cat([self.w2[:, :, 0], self.w2[:, :, 1]], dim=0)
                grad_w2_flat = (b2_flat.transpose(0, 1) @ grad_logits) * inv_sqrt2 / sqrt_h
                grad_w2 = torch.empty_like(self.w2)
                grad_w2[:, :, 0] = grad_w2_flat[: self.hidden_dim, :]
                grad_w2[:, :, 1] = grad_w2_flat[self.hidden_dim :, :]
                self.w2.grad = grad_w2
                grad_b2_flat = (grad_logits @ w2_flat.transpose(0, 1)) * inv_sqrt2 / sqrt_h
                grad_h = grad_b2_flat[:, : self.hidden_dim] + grad_b2_flat[:, self.hidden_dim :] * (math.pi * torch.cos(math.pi * h))
                grad_pre = grad_h * (1.0 - h.square())
                grad_w1_flat = (b1_flat.transpose(0, 1) @ grad_pre) * inv_sqrt2 / sqrt_d
                grad_w1 = torch.empty_like(self.w1)
                grad_w1[:, :, 0] = grad_w1_flat[: self.input_dim, :]
                grad_w1[:, :, 1] = grad_w1_flat[self.input_dim :, :]
                self.w1.grad = grad_w1
                return loss
            if cache and isinstance(cache[0], str) and cache[0] == "fourier_k2_memory_planned_recompute":
                _variant, z, h = cache
                inv_sqrt2 = 1.0 / math.sqrt(2.0)
                sqrt_h = math.sqrt(max(1, self.hidden_dim))
                sqrt_d = math.sqrt(max(1, self.input_dim))
                sin_h = torch.sin(math.pi * h)
                grad_w2 = torch.empty_like(self.w2)
                grad_w2[:, :, 0] = (h.transpose(0, 1) @ grad_logits) * inv_sqrt2 / sqrt_h
                grad_w2[:, :, 1] = (sin_h.transpose(0, 1) @ grad_logits) * inv_sqrt2 / sqrt_h
                self.w2.grad = grad_w2
                grad_h = (grad_logits @ self.w2[:, :, 0].transpose(0, 1)) * inv_sqrt2 / sqrt_h
                grad_h = grad_h + (grad_logits @ self.w2[:, :, 1].transpose(0, 1)) * (math.pi * torch.cos(math.pi * h) * inv_sqrt2) / sqrt_h
                grad_pre = grad_h * (1.0 - h.square())
                sin_z = torch.sin(math.pi * z)
                grad_w1 = torch.empty_like(self.w1)
                grad_w1[:, :, 0] = (z.transpose(0, 1) @ grad_pre) * inv_sqrt2 / sqrt_d
                grad_w1[:, :, 1] = (sin_z.transpose(0, 1) @ grad_pre) * inv_sqrt2 / sqrt_d
                self.w1.grad = grad_w1
                return loss
            if cache and isinstance(cache[0], str) and str(cache[0]).startswith("rbf_k") and str(cache[0]).endswith("_triton_l3_matmul"):
                from dgkan.kernels import fused_rbf

                _variant, x, h = cache
                return fused_rbf.backward(self, x, y, logits, h)
            if cache and isinstance(cache[0], str) and cache[0] == "hat_wavelet_k4_triton_l3_matmul":
                from dgkan.kernels import fused_hat_wavelet

                _variant, x, h = cache
                return fused_hat_wavelet.backward(self, x, y, logits, h)
            if cache and isinstance(cache[0], str) and str(cache[0]).endswith("_stream_recompute"):
                _variant, z, h = cache
                sqrt_h = math.sqrt(max(1, self.hidden_dim))
                sqrt_d = math.sqrt(max(1, self.input_dim))
                grad_w2 = torch.empty_like(self.w2)
                grad_h = torch.zeros_like(h)
                for basis_idx in range(int(self.k)):
                    b2_i = _basis_channel(h, self.spec.basis_name, self.k, self.centers, self.scales, basis_idx)
                    grad_w2[:, :, basis_idx] = (b2_i.transpose(0, 1) @ grad_logits) / sqrt_h
                    db2_i = _basis_derivative_channel(h, self.spec.basis_name, self.k, self.centers, self.scales, basis_idx)
                    grad_h = grad_h + (grad_logits @ self.w2[:, :, basis_idx].transpose(0, 1)) * db2_i / sqrt_h
                self.w2.grad = grad_w2
                grad_pre = grad_h * (1.0 - h.square())
                grad_w1 = torch.empty_like(self.w1)
                for basis_idx in range(int(self.k)):
                    b1_i = _basis_channel(z, self.spec.basis_name, self.k, self.centers, self.scales, basis_idx)
                    grad_w1[:, :, basis_idx] = (b1_i.transpose(0, 1) @ grad_pre) / sqrt_d
                self.w1.grad = grad_w1
                return loss
            b1, _pre_h, h, b2, db2 = cache
            self.w2.grad = torch.einsum("bhk,bc->hck", b2, grad_logits) / math.sqrt(max(1, self.hidden_dim))
            grad_h = torch.einsum("bc,hck,bhk->bh", grad_logits, self.w2, db2) / math.sqrt(max(1, self.hidden_dim))
            grad_pre = grad_h * (1.0 - h.square())
            self.w1.grad = torch.einsum("bdk,bh->dhk", b1, grad_pre) / math.sqrt(max(1, self.input_dim))
            return loss

    def manual_gradient_audit(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
        params = [p for p in self.parameters() if p.requires_grad]
        self.zero_grad(set_to_none=True)
        logits, cache = self.manual_ce_forward_cache(x)
        manual_logits = logits.detach().clone()
        self.manual_ce_backward_from_cache(logits, cache, y)
        manual_grads = [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in params]
        self.zero_grad(set_to_none=True)
        ref_logits = self.forward(x)
        F.cross_entropy(ref_logits, y).backward()
        ref_grads = [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in params]
        self.zero_grad(set_to_none=True)
        relerrs = []
        coses = []
        for gm, gr in zip(manual_grads, ref_grads):
            denom = gr.norm().clamp_min(1.0e-8)
            relerrs.append(float((gm - gr).norm().div(denom).detach().item()))
            coses.append(float(F.cosine_similarity(gm.flatten(), gr.flatten(), dim=0, eps=1.0e-8).detach().item()))
        return {
            "manual_forward_available": 1,
            "manual_backward_available": 1,
            "grad_relerr_max": max(relerrs) if relerrs else float("inf"),
            "grad_cos_min": min(coses) if coses else -1.0,
            "output_max_abs_error": float((manual_logits - ref_logits.detach()).abs().max().item()),
        }


class GroupedRationalKATKAN(nn.Module):
    """FlashKAT-inspired grouped rational activation with a compact GEMM head."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        spec: PrimitiveSpec,
        x_for_stats: torch.Tensor,
        seed: int,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.spec = spec
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.hidden_dim = int(spec.hidden_dim)
        variant = str(spec.init_variant).lower()
        self.readout_only_backbone_frozen_enabled = "freezebackbone" in variant
        self.rational_coefficients_frozen_enabled = "freezerational" in variant
        self.hidden_bias_enabled = "hiddenbias" in variant
        self.hidden_square_readout_enabled = "hiddensqreadout" in variant
        self.hidden_square_readout_zero_init = "hiddensqreadoutzero" in variant
        self.hidden_square_readout_centered = "hiddensqcenter" in variant
        self.hidden_square_readout_signed = "hiddensqsigned" in variant
        self.hidden_square_readout_scale = 1.0
        hidden_sq_scale_match = re.search(r"hiddensqscale(\d{3})", variant)
        if hidden_sq_scale_match:
            self.hidden_square_readout_scale = float(int(hidden_sq_scale_match.group(1))) / 100.0
        self.hidden_abs_readout_enabled = "hiddenabsreadout" in variant
        self.hidden_abs_readout_zero_init = "hiddenabsreadoutzero" in variant
        self.hidden_abs_readout_centered = "hiddenabscenter" in variant
        self.hidden_abs_center_mix = 1.0
        hidden_abs_center_mix_match = re.search(r"hiddenabscentermix(\d{3})", variant)
        if hidden_abs_center_mix_match:
            self.hidden_abs_center_mix = float(int(hidden_abs_center_mix_match.group(1))) / 100.0
        self.hidden_abs_readout_scale = 1.0
        hidden_abs_scale_match = re.search(r"hiddenabsscale(\d{3})", variant)
        if hidden_abs_scale_match:
            self.hidden_abs_readout_scale = float(int(hidden_abs_scale_match.group(1))) / 100.0
        self.hidden_rat_readout_enabled = "hiddenratreadout" in variant
        self.hidden_rat_readout_zero_init = "hiddenratreadoutzero" in variant
        self.hidden_rat_readout_center_stopgrad = "hiddenratcentersg" in variant
        self.hidden_rat_readout_centered = "hiddenratcenter" in variant
        self.hidden_rat_center_mix = 1.0
        hidden_rat_center_mix_match = re.search(r"hiddenratcentermix(\d{3})", variant)
        if hidden_rat_center_mix_match:
            self.hidden_rat_center_mix = float(int(hidden_rat_center_mix_match.group(1))) / 100.0
        self.hidden_rat_readout_scale = 1.0
        hidden_rat_scale_match = re.search(r"hiddenratscale(\d{3})", variant)
        if hidden_rat_scale_match:
            self.hidden_rat_readout_scale = float(int(hidden_rat_scale_match.group(1))) / 100.0
        self.hidden_rat_residual_scale = 0.0
        hidden_rat_residual_match = re.search(r"hiddenratresidual(\d{3})", variant)
        if hidden_rat_residual_match:
            self.hidden_rat_residual_scale = float(int(hidden_rat_residual_match.group(1))) / 100.0
        self.hidden_tanh_residual_scale = 0.0
        hidden_tanh_residual_match = re.search(r"hiddentanhresidual(\d{3})", variant)
        if hidden_tanh_residual_match:
            self.hidden_tanh_residual_scale = float(int(hidden_tanh_residual_match.group(1))) / 100.0
        self.hidden_center_tanh_residual_scale = 0.0
        hidden_center_tanh_residual_match = re.search(r"hiddencentertanhresidual(\d{3})", variant)
        if hidden_center_tanh_residual_match:
            self.hidden_center_tanh_residual_scale = float(int(hidden_center_tanh_residual_match.group(1))) / 100.0
        self.hidden_signsq_residual_scale = 0.0
        hidden_signsq_residual_match = re.search(r"hiddensignsqresidual(\d{3})", variant)
        if hidden_signsq_residual_match:
            self.hidden_signsq_residual_scale = float(int(hidden_signsq_residual_match.group(1))) / 100.0
        self.hidden_tail_readout_vjp_triton_enabled = "hiddentailgradtriton" in variant
        self.hidden_rat_residual_vjp_triton_enabled = "hiddenresvjptriton" in variant
        self.w2_bias_row_enabled = "w2biasrow" in variant
        groups = 16
        for token, value in (("g8", 8), ("g16", 16), ("g32", 32), ("g49", 49), ("g98", 98)):
            if token in variant:
                groups = value
        while self.input_dim % groups != 0 and groups > 1:
            groups //= 2
        self.group_count = max(1, int(groups))
        self.linear_residual_enabled = "linearres" in variant
        self.linear_residual_scale = 1.0
        linear_gain_match = re.search(r"linearresgain(\d{3,5})", variant)
        if linear_gain_match:
            self.linear_residual_scale = float(int(linear_gain_match.group(1))) / 100.0
        self.linear_residual_gate_enabled = "lineargate" in variant
        self.linear_residual_gate_init = 1.0
        linear_gate_match = re.search(r"lineargate(\d{3})", variant)
        if linear_gate_match:
            self.linear_residual_gate_init = float(int(linear_gate_match.group(1))) / 100.0
        self.input_cross_rank = 0
        self.input_cross_mode = "none"
        self.input_cross_feature_mode = "product"
        self.input_cross_hidden_enabled = False
        self.input_cross_hidden_bucket_enabled = False
        self.input_cross_hidden_direct_readout_enabled = False
        self.input_cross_hidden_rank = 0
        self.input_cross_hidden_scale = 0.0
        self.input_cross_hidden_inv_norm = 1.0
        self.input_cross_readout_scale = 1.0
        self.input_cross_readout_block_triton_enabled = "readblocktriton" in variant
        self.input_cross_readout_triton_enabled = "readtriton" in variant
        self.input_cross_readout_bucket_enabled = False
        self.input_cross_readout_bucket_rank = 0
        self.input_cross_readout_bucket_inv_norm = 1.0
        self.input_cross_signal_rank = 0
        self.input_cross_pca_rank = 0
        self.input_cross_pca_whiten_enabled = False
        self.input_projected_square_rank = 0
        self.input_projected_bilinear_rank = 0
        self.input_cross_readout_block_batch = 32 if "readblockb32" in variant else 16
        self.input_cross_readout_grad_block_rank = 32
        self.input_cross_readout_grad_atomic_enabled = False
        for token, value in (("readgradr64", 64), ("readgradr96", 96), ("readgradr128", 128), ("readgradr136", 136)):
            if token in variant:
                self.input_cross_readout_grad_block_rank = int(value)
        grad_atomic_match = re.search(r"readgrad(?:atomic|split)r(\d+)", variant)
        if grad_atomic_match:
            self.input_cross_readout_grad_atomic_enabled = True
            self.input_cross_readout_grad_block_rank = int(grad_atomic_match.group(1))
        self.input_cross_pair_norm_enabled = "pairnorm" in variant or "pairstd" in variant
        self.input_cross_pair_center_enabled = "pairnorm" in variant
        self.logit_bias_enabled = "logitbias" in variant
        self.logit_rms_norm_sg_enabled = "logitrmsnormsg" in variant
        self.logit_batch_rms_norm_sg_enabled = "logitbatchrmsnormsg" in variant
        self.logit_batch_rms_mix_sg = 0.0
        logit_batch_mix_match = re.search(r"logitbatchrmsmixsg(\d{3})", variant)
        if logit_batch_mix_match:
            self.logit_batch_rms_mix_sg = float(int(logit_batch_mix_match.group(1))) / 100.0
        logit_rms_scale = 1.0
        logit_rms_match = re.search(r"logit(?:batch)?rmsnormsg(\d{3})", variant)
        if logit_rms_match:
            logit_rms_scale = float(int(logit_rms_match.group(1))) / 100.0
        self.register_buffer("logit_rms_norm_scale", torch.tensor([logit_rms_scale], device=device))
        self.input_cross_logit_center_cap_enabled = "crosscentercap" in variant
        self.input_cross_logit_batch_sg_cap_enabled = "crossbatchsgcap" in variant
        self.input_cross_logit_batch_cap_enabled = "crossbatchcap" in variant or self.input_cross_logit_batch_sg_cap_enabled
        self.input_cross_logit_cap_enabled = "crosscap" in variant or self.input_cross_logit_center_cap_enabled or self.input_cross_logit_batch_cap_enabled
        self.input_cross_logit_cap_fused_enabled = "crosscapfused" in variant and not (self.input_cross_logit_center_cap_enabled or self.input_cross_logit_batch_cap_enabled)
        cross_cap = 2.0
        cap_match = re.search(r"cross(?:center|batchsg|batch)?cap(?:fused)?(\d{3})", variant)
        if cap_match:
            cross_cap = float(cap_match.group(1)) / 100.0
        self.register_buffer("input_cross_logit_cap_value", torch.tensor([cross_cap], device=device))
        cross_signal_match = re.search(r"crosssignalr(\d+)", variant)
        if cross_signal_match:
            self.input_cross_signal_rank = max(1, int(cross_signal_match.group(1)))
        cross_pca_whiten_match = re.search(r"crosspcawhiter(\d+)", variant)
        cross_pca_match = re.search(r"crosspcar(\d+)", variant)
        if cross_pca_whiten_match:
            self.input_cross_pca_rank = max(1, int(cross_pca_whiten_match.group(1)))
            self.input_cross_pca_whiten_enabled = True
        elif cross_pca_match:
            self.input_cross_pca_rank = max(1, int(cross_pca_match.group(1)))
        for token, scale in (("readscale010", 0.10), ("readscale025", 0.25), ("readscale050", 0.50)):
            if token in variant:
                self.input_cross_readout_scale = float(scale)
        for token, value in (("inputcrossr32", 32), ("inputcrossr48", 48), ("inputcrossr64", 64), ("inputcrossr96", 96), ("inputcrossr112", 112), ("inputcrossr128", 128)):
            if token in variant:
                self.input_cross_rank = int(value)
                self.input_cross_mode = "random_projection"
        for token, value, scale in (("pairhiddenr136s025", 136, 0.25), ("pairhiddenr136s050", 136, 0.50), ("pairhiddenr112s025", 112, 0.25)):
            if token in variant:
                self.input_cross_rank = int(value)
                self.input_cross_mode = "pair_index"
                self.input_cross_hidden_enabled = True
                self.input_cross_hidden_scale = float(scale)
                self.input_cross_hidden_rank = int(value)
        for token, value, scale in (("pairbucketr136s025", 136, 0.25), ("pairbucketr112s025", 112, 0.25), ("pairbucketr88s025", 88, 0.25)):
            if token in variant:
                self.input_cross_rank = int(value)
                self.input_cross_mode = "pair_index"
                self.input_cross_hidden_enabled = True
                self.input_cross_hidden_bucket_enabled = True
                self.input_cross_hidden_scale = float(scale)
                self.input_cross_hidden_rank = int(value)
        for token, value, scale in (("pairbucketdirectr136s025", 136, 0.25), ("pairbucketdirectr112s025", 112, 0.25)):
            if token in variant:
                self.input_cross_rank = int(value)
                self.input_cross_mode = "pair_index"
                self.input_cross_hidden_enabled = True
                self.input_cross_hidden_bucket_enabled = True
                self.input_cross_hidden_direct_readout_enabled = True
                self.input_cross_hidden_scale = float(scale)
                self.input_cross_hidden_rank = int(value)
        for token, direct_rank, hidden_rank, scale in (
            ("paircrossr136bucketr32s025", 136, 32, 0.25),
            ("paircrossr136bucketr64s025", 136, 64, 0.25),
        ):
            if token in variant:
                self.input_cross_rank = int(direct_rank)
                self.input_cross_mode = "pair_index"
                self.input_cross_hidden_enabled = True
                self.input_cross_hidden_bucket_enabled = True
                self.input_cross_hidden_direct_readout_enabled = True
                self.input_cross_hidden_scale = float(scale)
                self.input_cross_hidden_rank = int(hidden_rank)
        for token, direct_rank, bucket_rank in (
            ("pairreadbucketr136g16", 136, 16),
            ("pairreadbucketr136g32", 136, 32),
            ("pairreadbucketr136g64", 136, 64),
            ("pairreadbucketr136g96", 136, 96),
            ("pairreadbucketr136g112", 136, 112),
        ):
            if token in variant:
                self.input_cross_rank = int(direct_rank)
                self.input_cross_mode = "pair_index"
                self.input_cross_readout_bucket_enabled = True
                self.input_cross_readout_bucket_rank = int(bucket_rank)
        for token, value in (("projsqr32", 32), ("projsqr48", 48), ("projsqr64", 64), ("projsqr96", 96)):
            if token in variant:
                self.input_projected_square_rank = int(value)
        for token, value in (("projbilinr32", 32), ("projbilinr48", 48), ("projbilinr64", 64), ("projbilinr96", 96)):
            if token in variant:
                self.input_projected_bilinear_rank = int(value)
        for token, value in (("paircrossr64", 64), ("paircrossr80", 80), ("paircrossr88", 88), ("paircrossr96", 96), ("paircrossr112", 112), ("paircrossr120", 120), ("paircrossr128", 128), ("paircrossr132", 132), ("paircrossr136", 136), ("paircrossr160", 160)):
            if token in variant:
                self.input_cross_rank = int(value)
                self.input_cross_mode = "pair_index"
        for token, value in (("pairsumsqcrossr64", 64), ("pairsumsqcrossr80", 80), ("pairsumsqcrossr88", 88), ("pairsumsqcrossr96", 96), ("pairsumsqcrossr112", 112), ("pairsumsqcrossr120", 120), ("pairsumsqcrossr128", 128), ("pairsumsqcrossr136", 136)):
            if token in variant:
                self.input_cross_rank = int(value)
                self.input_cross_mode = "pair_index"
                self.input_cross_feature_mode = "sum_square"
        for token, value in (("pairdiffsqcrossr64", 64), ("pairdiffsqcrossr80", 80), ("pairdiffsqcrossr88", 88), ("pairdiffsqcrossr96", 96), ("pairdiffsqcrossr112", 112), ("pairdiffsqcrossr120", 120), ("pairdiffsqcrossr128", 128), ("pairdiffsqcrossr136", 136)):
            if token in variant:
                self.input_cross_rank = int(value)
                self.input_cross_mode = "pair_index"
                self.input_cross_feature_mode = "diff_square"
        for token, value in (("gridpaircrossr96", 96), ("gridpaircrossr112", 112), ("gridpaircrossr136", 136)):
            if token in variant:
                self.input_cross_rank = int(value)
                self.input_cross_mode = "grid_pair_index"
        for token, value in (("gridpairsumsqcrossr96", 96), ("gridpairsumsqcrossr112", 112), ("gridpairsumsqcrossr136", 136)):
            if token in variant:
                self.input_cross_rank = int(value)
                self.input_cross_mode = "grid_pair_index"
                self.input_cross_feature_mode = "sum_square"
        for token, value in (("balancedpaircrossr88", 88), ("balancedpaircrossr96", 96), ("balancedpaircrossr112", 112), ("balancedpaircrossr120", 120), ("balancedpaircrossr128", 128)):
            if token in variant:
                self.input_cross_rank = int(value)
                self.input_cross_mode = "balanced_pair_index"
        for token, value in (("balancedpairsumsqcrossr64", 64), ("balancedpairsumsqcrossr88", 88), ("balancedpairsumsqcrossr96", 96), ("balancedpairsumsqcrossr112", 112), ("balancedpairsumsqcrossr136", 136)):
            if token in variant:
                self.input_cross_rank = int(value)
                self.input_cross_mode = "balanced_pair_index"
                self.input_cross_feature_mode = "sum_square"
        for token, value in (("diagpaircrossr64", 64), ("diagpaircrossr80", 80), ("diagpaircrossr96", 96), ("diagpaircrossr112", 112), ("diagpaircrossr120", 120), ("diagpaircrossr128", 128), ("diagpaircrossr136", 136)):
            if token in variant:
                self.input_cross_rank = int(value)
                self.input_cross_mode = "diag_pair_index"
        for token, value in (("hybridpaircrossr96", 96), ("hybridpaircrossr112", 112), ("hybridpaircrossr120", 120), ("hybridpaircrossr128", 128)):
            if token in variant:
                self.input_cross_rank = int(value)
                self.input_cross_mode = "hybrid_pair_index"
        for token, value in (("balblendpaircrossr120", 120), ("balblendpaircrossr128", 128)):
            if token in variant:
                self.input_cross_rank = int(value)
                self.input_cross_mode = "balanced_blend_pair_index"
        for token, value in (("blendpaircrossr80", 80), ("blendpaircrossr88", 88), ("blendpaircrossr96", 96)):
            if token in variant:
                self.input_cross_rank = int(value)
                self.input_cross_mode = "blend_pair_index"
        self.input_cross_transform_dim = 0
        for token, value in (("transformpairr88", 88), ("transformpairr96", 96)):
            if token in variant:
                self.input_cross_rank = int(value)
                self.input_cross_mode = "transform_pair_index"
                self.input_cross_transform_dim = 16
        xs = x_for_stats[: min(4096, int(x_for_stats.shape[0]))].to(device=device, dtype=torch.float32)
        if self.input_cross_hidden_bucket_enabled:
            bucket_rank_for_norm = int(self.input_cross_hidden_rank or self.input_cross_rank)
            self.input_cross_hidden_inv_norm = 1.0 / math.sqrt(max(1.0, float(bucket_rank_for_norm) / float(max(1, self.hidden_dim))))
        if self.input_cross_readout_bucket_enabled:
            bucket_rank_for_norm = int(self.input_cross_readout_bucket_rank or self.input_cross_rank)
            self.input_cross_readout_bucket_inv_norm = 1.0 / math.sqrt(max(1.0, float(self.input_cross_rank) / float(max(1, bucket_rank_for_norm))))
        self.register_buffer("mu", xs.mean(dim=0))
        self.register_buffer("std", xs.std(dim=0).clamp_min(1.0e-3))
        gen = torch.Generator(device=device).manual_seed(int(seed))
        gelu_num = torch.tensor([-0.00153969, 0.51692871, 0.44075827, 0.0972553, -0.00884418, -0.00378675], device=device)
        gelu_den = torch.tensor([-0.14171056, -0.06481231, -0.0444695, 0.01283776], device=device)
        self.numerator = nn.Parameter(gelu_num.repeat(self.group_count, 1).contiguous())
        self.denominator = nn.Parameter(gelu_den.repeat(self.group_count, 1).contiguous())
        self.w1 = nn.Parameter(torch.randn(self.input_dim, self.hidden_dim, device=device, generator=gen) / math.sqrt(max(1, self.input_dim)))
        self.w2 = nn.Parameter(torch.randn(self.hidden_dim, self.output_dim, device=device, generator=gen) / math.sqrt(max(1, self.hidden_dim)))
        if self.hidden_square_readout_enabled:
            hidden_sq_init = torch.randn(self.hidden_dim, self.output_dim, device=device, generator=gen) * (0.05 / math.sqrt(max(1, self.hidden_dim)))
            if self.hidden_square_readout_zero_init:
                hidden_sq_init.zero_()
            self.hidden_square_readout = nn.Parameter(hidden_sq_init)
        if self.hidden_abs_readout_enabled:
            hidden_abs_init = torch.randn(self.hidden_dim, self.output_dim, device=device, generator=gen) * (0.05 / math.sqrt(max(1, self.hidden_dim)))
            if self.hidden_abs_readout_zero_init:
                hidden_abs_init.zero_()
            self.hidden_abs_readout = nn.Parameter(hidden_abs_init)
        if self.hidden_rat_readout_enabled:
            hidden_rat_init = torch.randn(self.hidden_dim, self.output_dim, device=device, generator=gen) * (0.05 / math.sqrt(max(1, self.hidden_dim)))
            if self.hidden_rat_readout_zero_init:
                hidden_rat_init.zero_()
            self.hidden_rat_readout = nn.Parameter(hidden_rat_init)
        if self.hidden_bias_enabled:
            self.hidden_bias = nn.Parameter(torch.zeros(self.hidden_dim, device=device))
        if self.logit_bias_enabled:
            self.bias = nn.Parameter(torch.zeros(self.output_dim, device=device))
        if self.linear_residual_enabled:
            self.linear_readout = nn.Parameter(torch.randn(self.input_dim, self.output_dim, device=device, generator=gen) * (0.05 / math.sqrt(max(1, self.input_dim))))
            if self.linear_residual_gate_enabled:
                self.linear_residual_gate = nn.Parameter(torch.tensor([self.linear_residual_gate_init], device=device, dtype=torch.float32))
        if self.input_projected_square_rank > 0:
            proj = self._make_fixed_transform(self.input_dim, int(self.input_projected_square_rank), device)
            self.register_buffer("input_projected_square_proj", proj)
            with torch.no_grad():
                z_stats = torch.tanh((xs - self.mu) / self.std)
                sq_stats = (z_stats @ proj).square()
                self.register_buffer("input_projected_square_mean", sq_stats.mean(dim=0).to(dtype=torch.float32))
                self.register_buffer("input_projected_square_inv_std", sq_stats.std(dim=0).clamp_min(1.0e-3).reciprocal().to(dtype=torch.float32))
            self.proj_square_readout = nn.Parameter(torch.randn(int(self.input_projected_square_rank), self.output_dim, device=device, generator=gen) * (0.05 / math.sqrt(max(1, int(self.input_projected_square_rank)))))
        if self.input_projected_bilinear_rank > 0:
            proj_left = self._make_fixed_transform(self.input_dim, int(self.input_projected_bilinear_rank), device)
            proj_right = self._make_fixed_transform_offset(self.input_dim, int(self.input_projected_bilinear_rank), device, start_col=int(self.input_projected_bilinear_rank) + 1)
            self.register_buffer("input_projected_bilinear_left", proj_left)
            self.register_buffer("input_projected_bilinear_right", proj_right)
            with torch.no_grad():
                z_stats = torch.tanh((xs - self.mu) / self.std)
                bilin_stats = (z_stats @ proj_left) * (z_stats @ proj_right)
                self.register_buffer("input_projected_bilinear_mean", bilin_stats.mean(dim=0).to(dtype=torch.float32))
                self.register_buffer("input_projected_bilinear_inv_std", bilin_stats.std(dim=0).clamp_min(1.0e-3).reciprocal().to(dtype=torch.float32))
            self.proj_bilinear_readout = nn.Parameter(torch.randn(int(self.input_projected_bilinear_rank), self.output_dim, device=device, generator=gen) * (0.05 / math.sqrt(max(1, int(self.input_projected_bilinear_rank)))))
        if self.input_cross_rank > 0 and self.input_cross_mode == "random_projection":
            left = torch.randn(self.input_dim, self.input_cross_rank, device=device, generator=gen) / math.sqrt(max(1, self.input_dim))
            right = torch.randn(self.input_dim, self.input_cross_rank, device=device, generator=gen) / math.sqrt(max(1, self.input_dim))
            self.register_buffer("input_cross_left", left)
            self.register_buffer("input_cross_right", right)
            if self.input_cross_hidden_bucket_enabled:
                bucket_rank = int(self.input_cross_hidden_rank or self.input_cross_rank)
                bucket = (torch.arange(bucket_rank, device=device, dtype=torch.long) * 37) % max(1, self.hidden_dim)
                signs = torch.where(torch.arange(bucket_rank, device=device) % 2 == 0, 1.0, -1.0).to(dtype=torch.float32)
                self.register_buffer("input_cross_hidden_bucket", bucket)
                self.register_buffer("input_cross_hidden_weight", signs)
                if self.input_cross_hidden_direct_readout_enabled:
                    self.cross_readout = nn.Parameter(torch.randn(self.input_cross_rank, self.output_dim, device=device, generator=gen) * (0.05 / math.sqrt(max(1, self.input_cross_rank))))
            elif self.input_cross_hidden_enabled:
                hidden_mix = torch.randn(self.input_cross_rank, self.hidden_dim, device=device, generator=gen) / math.sqrt(max(1, self.input_cross_rank))
                self.register_buffer("input_cross_hidden", hidden_mix)
            else:
                self.cross_readout = nn.Parameter(torch.randn(self.input_cross_rank, self.output_dim, device=device, generator=gen) * (0.05 / math.sqrt(max(1, self.input_cross_rank))))

        elif self.input_cross_rank > 0 and self.input_cross_mode in {"pair_index", "balanced_pair_index", "grid_pair_index", "hybrid_pair_index", "blend_pair_index", "balanced_blend_pair_index", "diag_pair_index"}:
            if self.input_cross_mode in {"hybrid_pair_index", "blend_pair_index", "balanced_blend_pair_index"}:
                if self.input_cross_mode == "blend_pair_index":
                    primary_rank = max(1, int(round(float(self.input_cross_rank) * 0.75)))
                elif self.input_cross_mode == "balanced_blend_pair_index":
                    primary_rank = max(1, int(round(float(self.input_cross_rank) * 0.25)))
                else:
                    primary_rank = max(1, int(self.input_cross_rank) // 2)
                secondary_rank = max(0, int(self.input_cross_rank) - primary_rank)
                left_primary, right_primary = self._make_pair_cross_indices(
                    self.input_dim,
                    primary_rank,
                    device,
                    balanced=False,
                    grid_balanced=False,
                )
                left_balanced, right_balanced = self._make_pair_cross_indices(
                    self.input_dim,
                    max(int(self.input_cross_rank) * 2, primary_rank + secondary_rank),
                    device,
                    balanced=True,
                    grid_balanced=False,
                )
                primary_key = torch.minimum(left_primary, right_primary) * int(self.input_dim) + torch.maximum(left_primary, right_primary)
                balanced_key = torch.minimum(left_balanced, right_balanced) * int(self.input_dim) + torch.maximum(left_balanced, right_balanced)
                keep_idx = torch.nonzero(~torch.isin(balanced_key, primary_key), as_tuple=False).flatten()[:secondary_rank]
                if int(keep_idx.numel()) < secondary_rank:
                    fallback_idx = torch.arange(0, min(secondary_rank, int(left_balanced.numel())), device=device, dtype=torch.long)
                    keep_idx = torch.cat([keep_idx, fallback_idx], dim=0)[:secondary_rank]
                left_idx = torch.cat([left_primary, left_balanced.index_select(0, keep_idx)], dim=0)[: int(self.input_cross_rank)]
                right_idx = torch.cat([right_primary, right_balanced.index_select(0, keep_idx)], dim=0)[: int(self.input_cross_rank)]
            elif self.input_cross_mode == "diag_pair_index":
                left_idx, right_idx = self._make_diag_rich_pair_cross_indices(
                    self.input_dim,
                    self.input_cross_rank,
                    device,
                )
            else:
                left_idx, right_idx = self._make_pair_cross_indices(
                    self.input_dim,
                    self.input_cross_rank,
                    device,
                    balanced=self.input_cross_mode == "balanced_pair_index",
                    grid_balanced=self.input_cross_mode == "grid_pair_index",
                )
            self.register_buffer("input_cross_pair_left", left_idx)
            self.register_buffer("input_cross_pair_right", right_idx)
            if self.input_cross_readout_bucket_enabled:
                bucket_rank = int(self.input_cross_readout_bucket_rank or self.input_cross_rank)
                bucket = (torch.arange(self.input_cross_rank, device=device, dtype=torch.long) * 37) % max(1, bucket_rank)
                signs = torch.where(torch.arange(self.input_cross_rank, device=device) % 2 == 0, 1.0, -1.0).to(dtype=torch.float32)
                self.register_buffer("input_cross_readout_bucket", bucket)
                self.register_buffer("input_cross_readout_bucket_weight", signs)
                self.cross_readout = nn.Parameter(torch.randn(bucket_rank, self.output_dim, device=device, generator=gen) * (0.05 / math.sqrt(max(1, bucket_rank))))
            elif self.input_cross_hidden_bucket_enabled:
                bucket_rank = int(self.input_cross_hidden_rank or self.input_cross_rank)
                if bucket_rank != int(self.input_cross_rank):
                    hidden_left, hidden_right = self._make_pair_cross_indices(
                        self.input_dim,
                        bucket_rank,
                        device,
                        balanced=False,
                        grid_balanced=False,
                    )
                    self.register_buffer("input_cross_hidden_pair_left", hidden_left)
                    self.register_buffer("input_cross_hidden_pair_right", hidden_right)
                bucket = (torch.arange(bucket_rank, device=device, dtype=torch.long) * 37) % max(1, self.hidden_dim)
                signs = torch.where(torch.arange(bucket_rank, device=device) % 2 == 0, 1.0, -1.0).to(dtype=torch.float32)
                self.register_buffer("input_cross_hidden_bucket", bucket)
                self.register_buffer("input_cross_hidden_weight", signs)
                if self.input_cross_hidden_direct_readout_enabled:
                    self.cross_readout = nn.Parameter(torch.randn(self.input_cross_rank, self.output_dim, device=device, generator=gen) * (0.05 / math.sqrt(max(1, self.input_cross_rank))))
            elif self.input_cross_hidden_enabled:
                hidden_mix = torch.randn(self.input_cross_rank, self.hidden_dim, device=device, generator=gen) / math.sqrt(max(1, self.input_cross_rank))
                self.register_buffer("input_cross_hidden", hidden_mix)
            else:
                self.cross_readout = nn.Parameter(torch.randn(self.input_cross_rank, self.output_dim, device=device, generator=gen) * (0.05 / math.sqrt(max(1, self.input_cross_rank))))
        elif self.input_cross_rank > 0 and self.input_cross_mode == "transform_pair_index":
            transform_dim = max(2, int(self.input_cross_transform_dim))
            transform = self._make_fixed_transform(self.input_dim, transform_dim, device)
            left_idx, right_idx = self._make_pair_cross_indices(transform_dim, self.input_cross_rank, device, balanced=True)
            self.register_buffer("input_cross_transform", transform)
            self.register_buffer("input_cross_pair_left", left_idx)
            self.register_buffer("input_cross_pair_right", right_idx)
            if self.input_cross_hidden_bucket_enabled:
                bucket_rank = int(self.input_cross_hidden_rank or self.input_cross_rank)
                bucket = (torch.arange(bucket_rank, device=device, dtype=torch.long) * 37) % max(1, self.hidden_dim)
                signs = torch.where(torch.arange(bucket_rank, device=device) % 2 == 0, 1.0, -1.0).to(dtype=torch.float32)
                self.register_buffer("input_cross_hidden_bucket", bucket)
                self.register_buffer("input_cross_hidden_weight", signs)
                if self.input_cross_hidden_direct_readout_enabled:
                    self.cross_readout = nn.Parameter(torch.randn(self.input_cross_rank, self.output_dim, device=device, generator=gen) * (0.05 / math.sqrt(max(1, self.input_cross_rank))))
            elif self.input_cross_hidden_enabled:
                hidden_mix = torch.randn(self.input_cross_rank, self.hidden_dim, device=device, generator=gen) / math.sqrt(max(1, self.input_cross_rank))
                self.register_buffer("input_cross_hidden", hidden_mix)
            else:
                self.cross_readout = nn.Parameter(torch.randn(self.input_cross_rank, self.output_dim, device=device, generator=gen) * (0.05 / math.sqrt(max(1, self.input_cross_rank))))

        if "crosszero" in variant and hasattr(self, "cross_readout"):
            with torch.no_grad():
                self.cross_readout.zero_()
        if self.input_cross_signal_rank > 0 and self.input_cross_rank > 0:
            signal_rank = int(self.input_cross_signal_rank)
            self.cross_signal_readout = nn.Parameter(
                torch.randn(self.input_cross_rank, signal_rank, device=device, generator=gen)
                * (0.05 / math.sqrt(max(1, self.input_cross_rank)))
            )
            self.cross_signal_class = nn.Parameter(
                torch.randn(signal_rank, self.output_dim, device=device, generator=gen)
                * (0.05 / math.sqrt(max(1, signal_rank)))
            )
            if hasattr(self, "cross_readout"):
                self.cross_readout.requires_grad_(False)
        if ("sqzero" in variant or "projsqzero" in variant) and hasattr(self, "proj_square_readout"):
            with torch.no_grad():
                self.proj_square_readout.zero_()
        if ("bilinzero" in variant or "projbilinzero" in variant) and hasattr(self, "proj_bilinear_readout"):
            with torch.no_grad():
                self.proj_bilinear_readout.zero_()
        self._register_pair_cross_stats(xs)
        if self.input_cross_pca_rank > 0 and self.input_cross_rank > 0:
            with torch.no_grad():
                z_stats = self._norm_input(xs)
                cross_stats = self._input_cross(z_stats).to(dtype=torch.float32)
                pca_rank = min(int(self.input_cross_pca_rank), int(cross_stats.shape[1]))
                pca_mean = cross_stats.mean(dim=0)
                centered = cross_stats - pca_mean.view(1, -1)
                try:
                    _u, s, vh = torch.linalg.svd(centered, full_matrices=False)
                    pca_rank = min(pca_rank, int(vh.shape[0]), int(s.numel()))
                    components = vh[:pca_rank].transpose(0, 1).contiguous()
                    denom = math.sqrt(max(1, int(centered.shape[0]) - 1))
                    std = (s[:pca_rank] / denom).clamp_min(1.0e-3)
                except RuntimeError:
                    components = torch.eye(int(cross_stats.shape[1]), device=device, dtype=torch.float32)[:, :pca_rank].contiguous()
                    std = torch.ones((pca_rank,), device=device, dtype=torch.float32)
                scale = std.reciprocal() if self.input_cross_pca_whiten_enabled else torch.ones_like(std)
                self.register_buffer("input_cross_pca_mean", pca_mean.to(dtype=torch.float32))
                self.register_buffer("input_cross_pca_components", components.to(dtype=torch.float32))
                self.register_buffer("input_cross_pca_scale", scale.to(dtype=torch.float32))
            self.input_cross_pca_rank = int(pca_rank)
            self.cross_pca_readout = nn.Parameter(
                torch.randn(int(pca_rank), self.output_dim, device=device, generator=gen)
                * (0.05 / math.sqrt(max(1, int(pca_rank))))
            )
            if "crosszero" in variant:
                with torch.no_grad():
                    self.cross_pca_readout.zero_()
            if hasattr(self, "cross_readout"):
                self.cross_readout.requires_grad_(False)
        if self.readout_only_backbone_frozen_enabled or self.rational_coefficients_frozen_enabled:
            self.numerator.requires_grad_(False)
            self.denominator.requires_grad_(False)
        if self.readout_only_backbone_frozen_enabled:
            self.w1.requires_grad_(False)

    @property
    def edge_param_count(self) -> int:
        extra = int(self.linear_readout.numel()) if self.linear_residual_enabled else 0
        extra += int(self.linear_residual_gate.numel()) if hasattr(self, "linear_residual_gate") else 0
        extra += int(self.hidden_bias.numel()) if hasattr(self, "hidden_bias") else 0
        extra += int(self.hidden_square_readout.numel()) if hasattr(self, "hidden_square_readout") else 0
        if hasattr(self, "cross_readout") and self.input_cross_signal_rank <= 0 and self.input_cross_pca_rank <= 0:
            extra += int(self.cross_readout.numel())
        extra += int(self.cross_signal_readout.numel()) if hasattr(self, "cross_signal_readout") else 0
        extra += int(self.cross_signal_class.numel()) if hasattr(self, "cross_signal_class") else 0
        extra += int(self.cross_pca_readout.numel()) if hasattr(self, "cross_pca_readout") else 0
        extra += int(self.proj_square_readout.numel()) if hasattr(self, "proj_square_readout") else 0
        extra += int(self.proj_bilinear_readout.numel()) if hasattr(self, "proj_bilinear_readout") else 0
        extra += int(self.bias.numel()) if hasattr(self, "bias") else 0
        return int(self.numerator.numel() + self.denominator.numel() + self.w1.numel() + self.w2.numel() + extra)

    def _norm_input(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh((x - self.mu) / self.std)

    def _linear_residual_effective_scale(self) -> float | torch.Tensor:
        scale = float(self.linear_residual_scale)
        gate = getattr(self, "linear_residual_gate", None)
        if gate is None:
            return scale
        return gate.view(()) * scale

    def _linear_residual_effective_scale_value(self) -> float:
        gate = getattr(self, "linear_residual_gate", None)
        if gate is None:
            return float(self.linear_residual_scale)
        return float(self.linear_residual_scale) * float(gate.detach().view(()).item())

    def _apply_w2_bias_row(self, h: torch.Tensor) -> torch.Tensor:
        if not self.w2_bias_row_enabled or int(h.shape[1]) <= 0:
            return h
        out = h.clone()
        out[:, -1] = 1.0
        return out

    @staticmethod
    def _make_pair_cross_indices(input_dim: int, rank: int, device: torch.device, *, balanced: bool = False, grid_balanced: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        dim = int(input_dim)
        target = max(0, int(rank))
        pairs: List[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()

        def add_pair(i: int, j: int) -> None:
            if len(pairs) >= target or dim <= 0:
                return
            a = int(i) % dim
            b = int(j) % dim
            key = (a, b) if a <= b else (b, a)
            if key in seen:
                return
            seen.add(key)
            pairs.append((a, b))

        if dim <= 32:
            for i in range(dim):
                add_pair(i, i)
            offdiag = [(i, j) for i in range(dim) for j in range(i + 1, dim)]
            remaining = max(0, target - len(pairs))
            if balanced and remaining > 0 and offdiag:
                for k in range(min(remaining, len(offdiag))):
                    idx = min(len(offdiag) - 1, (k * len(offdiag)) // max(1, remaining))
                    add_pair(*offdiag[idx])
            else:
                for i, j in offdiag:
                    add_pair(i, j)
                    if len(pairs) >= target:
                        break
        else:
            diag_budget = min(dim, max(1, target // 4))
            for k in range(diag_budget):
                add_pair((k * dim) // max(1, diag_budget), (k * dim) // max(1, diag_budget))
            side = math.isqrt(dim)
            if grid_balanced and side * side == dim:
                remaining = max(0, target - len(pairs))
                global_budget = max(1, remaining // 4) if remaining > 8 else 0
                local_budget = max(0, remaining - global_budget)
                local_edges: List[tuple[int, int]] = []
                for row in range(side):
                    for col in range(side - 1):
                        idx = row * side + col
                        local_edges.append((idx, idx + 1))
                for row in range(side - 1):
                    for col in range(side):
                        idx = row * side + col
                        local_edges.append((idx, idx + side))
                for k in range(min(local_budget, len(local_edges))):
                    idx = min(len(local_edges) - 1, (k * len(local_edges)) // max(1, local_budget))
                    add_pair(*local_edges[idx])
                k = 0
                while len(pairs) < target and k < global_budget * 8 + dim:
                    i = (k * 43 + 5) % dim
                    j = (i + side + 1 + ((k * 97 + 23) % max(1, dim - side - 1))) % dim
                    add_pair(i, j)
                    k += 1
            elif side * side == dim:
                for row in range(side):
                    for col in range(side - 1):
                        idx = row * side + col
                        add_pair(idx, idx + 1)
                        if len(pairs) >= target:
                            break
                    if len(pairs) >= target:
                        break
                for row in range(side - 1):
                    for col in range(side):
                        idx = row * side + col
                        add_pair(idx, idx + side)
                        if len(pairs) >= target:
                            break
                    if len(pairs) >= target:
                        break
            k = 0
            while len(pairs) < target and k < target * 20 + dim:
                i = (k * 37 + 17) % dim
                j = (i + 1 + ((k * 53 + 11) % max(1, dim - 1))) % dim
                add_pair(i, j)
                k += 1
            i = 0
            while len(pairs) < target:
                add_pair(i, (i * 41 + 19) % dim)
                i += 1

        left = torch.tensor([p[0] for p in pairs], device=device, dtype=torch.long)
        right = torch.tensor([p[1] for p in pairs], device=device, dtype=torch.long)
        return left, right

    @staticmethod
    def _make_diag_rich_pair_cross_indices(input_dim: int, rank: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        dim = int(input_dim)
        target = max(0, int(rank))
        pairs: List[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()

        def add_pair(i: int, j: int) -> None:
            if len(pairs) >= target or dim <= 0:
                return
            a = int(i) % dim
            b = int(j) % dim
            key = (a, b) if a <= b else (b, a)
            if key in seen:
                return
            seen.add(key)
            pairs.append((a, b))

        diag_budget = min(dim, max(1, (2 * target) // 3))
        for k in range(diag_budget):
            idx = (k * dim) // max(1, diag_budget)
            add_pair(idx, idx)

        side = math.isqrt(dim)
        if side * side == dim:
            for row in range(side):
                for col in range(side - 1):
                    add_pair(row * side + col, row * side + col + 1)
                    if len(pairs) >= target:
                        break
                if len(pairs) >= target:
                    break
            for row in range(side - 1):
                for col in range(side):
                    add_pair(row * side + col, (row + 1) * side + col)
                    if len(pairs) >= target:
                        break
                if len(pairs) >= target:
                    break

        k = 0
        while len(pairs) < target and k < target * 20 + dim:
            i = (k * 41 + 13) % dim
            j = (i + 1 + ((k * 67 + 19) % max(1, dim - 1))) % dim
            add_pair(i, j)
            k += 1

        while len(pairs) < target:
            idx = len(pairs) % max(1, dim)
            pairs.append((idx, idx))
        left = torch.tensor([p[0] for p in pairs], device=device, dtype=torch.long)
        right = torch.tensor([p[1] for p in pairs], device=device, dtype=torch.long)
        return left, right

    @staticmethod
    def _make_fixed_transform(input_dim: int, transform_dim: int, device: torch.device) -> torch.Tensor:
        dim = int(input_dim)
        out_dim = int(transform_dim)
        rows = torch.arange(dim, device=device, dtype=torch.float32).unsqueeze(1)
        cols = torch.arange(1, out_dim + 1, device=device, dtype=torch.float32).unsqueeze(0)
        mat = torch.cos(math.pi * (rows + 0.5) * cols / float(max(1, dim)))
        mat = mat - mat.mean(dim=0, keepdim=True)
        return mat / mat.norm(dim=0, keepdim=True).clamp_min(1.0e-6)

    @staticmethod
    def _make_fixed_transform_offset(input_dim: int, transform_dim: int, device: torch.device, *, start_col: int) -> torch.Tensor:
        dim = int(input_dim)
        out_dim = int(transform_dim)
        rows = torch.arange(dim, device=device, dtype=torch.float32).unsqueeze(1)
        cols = torch.arange(int(start_col), int(start_col) + out_dim, device=device, dtype=torch.float32).unsqueeze(0)
        mat = torch.cos(math.pi * (rows + 0.5) * cols / float(max(1, dim)))
        mat = mat - mat.mean(dim=0, keepdim=True)
        return mat / mat.norm(dim=0, keepdim=True).clamp_min(1.0e-6)

    def _input_projected_square_features(self, z: torch.Tensor) -> torch.Tensor:
        if self.input_projected_square_rank <= 0 or not hasattr(self, "input_projected_square_proj"):
            return z.new_zeros((int(z.shape[0]), 0))
        q = z @ self.input_projected_square_proj.to(device=z.device, dtype=z.dtype)
        feats = q.square()
        if hasattr(self, "input_projected_square_mean") and hasattr(self, "input_projected_square_inv_std"):
            mean = self.input_projected_square_mean.to(device=z.device, dtype=z.dtype)
            inv_std = self.input_projected_square_inv_std.to(device=z.device, dtype=z.dtype)
            feats = (feats - mean.view(1, -1)) * inv_std.view(1, -1)
        return feats

    def _input_projected_square_denominator(self) -> float:
        return math.sqrt(max(1, int(self.input_projected_square_rank)))

    def _input_projected_bilinear_features(self, z: torch.Tensor) -> torch.Tensor:
        if self.input_projected_bilinear_rank <= 0 or not hasattr(self, "input_projected_bilinear_left"):
            return z.new_zeros((int(z.shape[0]), 0))
        left = self.input_projected_bilinear_left.to(device=z.device, dtype=z.dtype)
        right = self.input_projected_bilinear_right.to(device=z.device, dtype=z.dtype)
        feats = (z @ left) * (z @ right)
        if hasattr(self, "input_projected_bilinear_mean") and hasattr(self, "input_projected_bilinear_inv_std"):
            mean = self.input_projected_bilinear_mean.to(device=z.device, dtype=z.dtype)
            inv_std = self.input_projected_bilinear_inv_std.to(device=z.device, dtype=z.dtype)
            feats = (feats - mean.view(1, -1)) * inv_std.view(1, -1)
        return feats

    def _input_projected_bilinear_denominator(self) -> float:
        return math.sqrt(max(1, int(self.input_projected_bilinear_rank)))

    def _input_pair_feature_mode_id(self) -> int:
        if self.input_cross_feature_mode == "sum_square":
            return 1
        if self.input_cross_feature_mode == "diff_square":
            return 2
        return 0

    def _pair_feature_from_indices(self, z: torch.Tensor, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        zl = z.index_select(1, left)
        zr = z.index_select(1, right)
        if self.input_cross_feature_mode == "sum_square":
            return 0.5 * (zl + zr).square()
        if self.input_cross_feature_mode == "diff_square":
            return 0.5 * (zl - zr).square()
        return zl * zr

    def _input_cross(self, z: torch.Tensor) -> torch.Tensor:
        if self.input_cross_rank <= 0:
            return z.new_zeros((int(z.shape[0]), 0))
        if self.input_cross_mode == "transform_pair_index":
            t = z @ self.input_cross_transform
            cross = self._pair_feature_from_indices(t, self.input_cross_pair_left, self.input_cross_pair_right)
            return self._normalize_pair_cross(cross)
        if self.input_cross_mode in {"pair_index", "balanced_pair_index", "grid_pair_index", "hybrid_pair_index", "blend_pair_index", "balanced_blend_pair_index", "diag_pair_index"}:
            cross = self._pair_feature_from_indices(z, self.input_cross_pair_left, self.input_cross_pair_right)
            return self._normalize_pair_cross(cross)
        cross = (z @ self.input_cross_left) * (z @ self.input_cross_right)
        return self._normalize_pair_cross(cross)

    def _input_cross_readout_denominator(self) -> float:
        if self.input_cross_readout_bucket_enabled:
            return math.sqrt(max(1, int(self.input_cross_readout_bucket_rank or self.input_cross_rank)))
        return math.sqrt(max(1, int(self.input_cross_rank)))

    def _input_cross_readout_bucket_features(self, z: torch.Tensor) -> torch.Tensor:
        if not self.input_cross_readout_bucket_enabled:
            return self._input_cross(z)
        fast_delta = _pairbucket_hidden_delta_triton(
            z,
            self.input_cross_pair_left,
            self.input_cross_pair_right,
            self.input_cross_readout_bucket,
            self.input_cross_readout_bucket_weight,
            int(self.input_cross_readout_bucket_rank),
            float(self.input_cross_readout_bucket_inv_norm),
        )
        if fast_delta is not None:
            return fast_delta
        cross = self._pair_feature_from_indices(z, self.input_cross_pair_left, self.input_cross_pair_right)
        out = cross.new_zeros((int(cross.shape[0]), int(self.input_cross_readout_bucket_rank)))
        bucket = self.input_cross_readout_bucket.view(1, -1).expand(int(cross.shape[0]), -1)
        values = cross * self.input_cross_readout_bucket_weight.view(1, -1)
        out.scatter_add_(1, bucket, values)
        return float(self.input_cross_readout_bucket_inv_norm) * out

    def _register_pair_cross_stats(self, xs: torch.Tensor) -> None:
        if not self.input_cross_pair_norm_enabled or self.input_cross_rank <= 0 or not hasattr(self, "cross_readout"):
            return
        with torch.no_grad():
            z = self._norm_input(xs)
            if self.input_cross_mode == "transform_pair_index" and hasattr(self, "input_cross_transform"):
                t = z @ self.input_cross_transform
                cross = self._pair_feature_from_indices(t, self.input_cross_pair_left, self.input_cross_pair_right)
            elif self.input_cross_mode in {"pair_index", "balanced_pair_index", "grid_pair_index", "hybrid_pair_index", "blend_pair_index", "balanced_blend_pair_index", "diag_pair_index"} and hasattr(self, "input_cross_pair_left"):
                cross = self._pair_feature_from_indices(z, self.input_cross_pair_left, self.input_cross_pair_right)
            elif self.input_cross_mode == "random_projection" and hasattr(self, "input_cross_left"):
                cross = (z @ self.input_cross_left) * (z @ self.input_cross_right)
            else:
                return
            if self.input_cross_pair_center_enabled:
                mean = cross.mean(dim=0).to(dtype=torch.float32)
            else:
                mean = torch.zeros((int(cross.shape[1]),), device=cross.device, dtype=torch.float32)
            inv_std = cross.std(dim=0).clamp_min(1.0e-3).reciprocal().to(dtype=torch.float32)
            self.register_buffer("input_cross_pair_mean", mean)
            self.register_buffer("input_cross_pair_inv_std", inv_std)

    def _normalize_pair_cross(self, cross: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "input_cross_pair_mean") and hasattr(self, "input_cross_pair_inv_std"):
            mean = self.input_cross_pair_mean.to(device=cross.device, dtype=cross.dtype)
            inv_std = self.input_cross_pair_inv_std.to(device=cross.device, dtype=cross.dtype)
            return (cross - mean.view(1, -1)) * inv_std.view(1, -1)
        return cross

    def _pair_norm_buffers_for_kernel(self) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if hasattr(self, "input_cross_pair_mean") and hasattr(self, "input_cross_pair_inv_std"):
            return self.input_cross_pair_mean, self.input_cross_pair_inv_std
        return None, None

    def _pair_readout_for_block_kernel(self) -> torch.Tensor:
        if hasattr(self, "input_cross_pair_inv_std"):
            inv = self.input_cross_pair_inv_std.to(device=self.cross_readout.device, dtype=self.cross_readout.dtype)
            return self.cross_readout * inv.view(-1, 1)
        return self.cross_readout

    def _pair_norm_bias_for_block_kernel(self, bias: torch.Tensor | None) -> torch.Tensor | None:
        if not (hasattr(self, "input_cross_pair_mean") and hasattr(self, "input_cross_pair_inv_std")):
            return bias
        mean = self.input_cross_pair_mean.to(device=self.cross_readout.device, dtype=self.cross_readout.dtype)
        inv = self.input_cross_pair_inv_std.to(device=self.cross_readout.device, dtype=self.cross_readout.dtype)
        norm_offset = -float(self.input_cross_readout_scale) * ((mean * inv).view(1, -1) @ self.cross_readout).view(-1) / math.sqrt(max(1, self.input_cross_rank))
        if bias is not None:
            norm_offset = norm_offset + bias
        return norm_offset

    def _pair_norm_readout_grad_from_block_kernel(self, raw_grad: torch.Tensor, grad_logits: torch.Tensor) -> torch.Tensor:
        if not (hasattr(self, "input_cross_pair_mean") and hasattr(self, "input_cross_pair_inv_std")):
            return raw_grad
        mean = self.input_cross_pair_mean.to(device=raw_grad.device, dtype=raw_grad.dtype)
        inv = self.input_cross_pair_inv_std.to(device=raw_grad.device, dtype=raw_grad.dtype)
        correction = float(self.input_cross_readout_scale) * (mean * inv).view(-1, 1) * grad_logits.sum(dim=0).view(1, -1) / math.sqrt(max(1, self.input_cross_rank))
        return inv.view(-1, 1) * raw_grad - correction

    def _bounded_pair_logits(self, pair_logits: torch.Tensor) -> torch.Tensor:
        if not self.input_cross_logit_cap_enabled:
            return pair_logits
        cap = self.input_cross_logit_cap_value.to(device=pair_logits.device, dtype=pair_logits.dtype).clamp_min(1.0e-3)
        if self.input_cross_logit_center_cap_enabled:
            pair_logits = pair_logits - pair_logits.mean(dim=-1, keepdim=True)
        elif self.input_cross_logit_batch_cap_enabled:
            batch_mean = pair_logits.mean(dim=0, keepdim=True)
            if self.input_cross_logit_batch_sg_cap_enabled:
                batch_mean = batch_mean.detach()
            pair_logits = pair_logits - batch_mean
        return cap * torch.tanh(pair_logits / cap)

    def _bounded_pair_grad_logits(self, pair_raw_logits: torch.Tensor | None, grad_logits: torch.Tensor) -> torch.Tensor:
        if not self.input_cross_logit_cap_enabled or pair_raw_logits is None:
            return grad_logits
        cap = self.input_cross_logit_cap_value.to(device=pair_raw_logits.device, dtype=pair_raw_logits.dtype).clamp_min(1.0e-3)
        if self.input_cross_logit_center_cap_enabled:
            centered = pair_raw_logits - pair_raw_logits.mean(dim=-1, keepdim=True)
            t = torch.tanh(centered / cap)
            grad_centered = grad_logits * (1.0 - t.square())
            return grad_centered - grad_centered.mean(dim=-1, keepdim=True)
        if self.input_cross_logit_batch_cap_enabled:
            centered = pair_raw_logits - pair_raw_logits.mean(dim=0, keepdim=True)
            t = torch.tanh(centered / cap)
            grad_centered = grad_logits * (1.0 - t.square())
            if self.input_cross_logit_batch_sg_cap_enabled:
                return grad_centered
            return grad_centered - grad_centered.mean(dim=0, keepdim=True)
        t = torch.tanh(pair_raw_logits / cap)
        return grad_logits * (1.0 - t.square())

    def _cross_signal_features(self, cross: torch.Tensor) -> torch.Tensor:
        return float(self.input_cross_readout_scale) * (cross @ self.cross_signal_readout) / math.sqrt(max(1, int(self.input_cross_rank)))

    def _cross_signal_class_weight(self) -> torch.Tensor:
        class_weight = self.cross_signal_class
        return class_weight - class_weight.mean(dim=1, keepdim=True)

    def _cross_signal_logits(self, cross: torch.Tensor) -> torch.Tensor:
        signal = self._cross_signal_features(cross)
        class_weight = self._cross_signal_class_weight()
        return (signal @ class_weight) / math.sqrt(max(1, int(self.input_cross_signal_rank)))

    def _accumulate_cross_signal_grads(self, cross: torch.Tensor, grad_logits: torch.Tensor) -> None:
        signal_rank = max(1, int(self.input_cross_signal_rank))
        class_weight = self._cross_signal_class_weight()
        signal = self._cross_signal_features(cross)
        grad_signal = (grad_logits @ class_weight.transpose(0, 1)) / math.sqrt(signal_rank)
        self.cross_signal_readout.grad = (
            float(self.input_cross_readout_scale)
            * (cross.transpose(0, 1) @ grad_signal)
            / math.sqrt(max(1, int(self.input_cross_rank)))
        ).to(dtype=self.cross_signal_readout.dtype)
        grad_class = (signal.transpose(0, 1) @ grad_logits) / math.sqrt(signal_rank)
        grad_class = grad_class - grad_class.mean(dim=1, keepdim=True)
        self.cross_signal_class.grad = grad_class.to(dtype=self.cross_signal_class.dtype)

    def _input_cross_pca_features(self, cross: torch.Tensor) -> torch.Tensor:
        mean = self.input_cross_pca_mean.to(device=cross.device, dtype=cross.dtype)
        components = self.input_cross_pca_components.to(device=cross.device, dtype=cross.dtype)
        scale = self.input_cross_pca_scale.to(device=cross.device, dtype=cross.dtype)
        feats = (cross - mean.view(1, -1)) @ components
        return float(self.input_cross_readout_scale) * feats * scale.view(1, -1)

    def _cross_pca_logits(self, cross: torch.Tensor) -> torch.Tensor:
        feats = self._input_cross_pca_features(cross)
        return (feats @ self.cross_pca_readout) / math.sqrt(max(1, int(self.input_cross_pca_rank)))

    def _accumulate_cross_pca_grads(self, cross: torch.Tensor, grad_logits: torch.Tensor) -> None:
        feats = self._input_cross_pca_features(cross)
        self.cross_pca_readout.grad = (
            feats.transpose(0, 1) @ grad_logits / math.sqrt(max(1, int(self.input_cross_pca_rank)))
        ).to(dtype=self.cross_pca_readout.dtype)

    def _input_cross_hidden_delta(self, cross: torch.Tensor | None = None, z: torch.Tensor | None = None) -> torch.Tensor:
        if self.input_cross_hidden_bucket_enabled:
            scale = self.input_cross_hidden_scale * self.input_cross_hidden_inv_norm
            left = getattr(self, "input_cross_hidden_pair_left", getattr(self, "input_cross_pair_left", None))
            right = getattr(self, "input_cross_hidden_pair_right", getattr(self, "input_cross_pair_right", None))
            if z is not None and left is not None and right is not None:
                fast_delta = _pairbucket_hidden_delta_triton(
                    z,
                    left,
                    right,
                    self.input_cross_hidden_bucket,
                    self.input_cross_hidden_weight,
                    self.hidden_dim,
                    scale,
                )
                if fast_delta is not None:
                    return fast_delta
            if cross is None:
                if z is None:
                    raise RuntimeError("bucket hidden cross features require cross or normalized input z")
                if left is not None and right is not None:
                    cross = self._pair_feature_from_indices(z, left, right)
                else:
                    cross = self._input_cross(z)
            elif int(cross.shape[1]) != int(self.input_cross_hidden_bucket.numel()):
                if z is None or left is None or right is None:
                    raise RuntimeError("bucket hidden direct+hidden path requires hidden pair indices when cross rank differs")
                cross = self._pair_feature_from_indices(z, left, right)
            delta = cross.new_zeros((int(cross.shape[0]), self.hidden_dim))
            bucket = self.input_cross_hidden_bucket.view(1, -1).expand(int(cross.shape[0]), -1)
            values = cross * self.input_cross_hidden_weight.view(1, -1)
            delta.scatter_add_(1, bucket, values)
            return scale * delta
        if cross is None:
            if z is None:
                raise RuntimeError("hidden cross projection requires cross or normalized input z")
            cross = self._input_cross(z)
        return self.input_cross_hidden_scale * ((cross @ self.input_cross_hidden) / math.sqrt(max(1, self.input_cross_rank)))

    def _rational_forward(self, z: torch.Tensor, autograd_enabled: bool) -> torch.Tensor:
        if z.is_cuda:
            try:
                rational_fwd_triton, _rational_bwd_triton, rational_autograd = _flashkat_rational_kernels()
                z3 = z.contiguous().unsqueeze(1)
                if autograd_enabled:
                    return rational_autograd.apply(z3, self.numerator, self.denominator).squeeze(1)
                return rational_fwd_triton(z3, self.numerator, self.denominator).squeeze(1)
            except Exception:
                pass
        return _grouped_rational_forward_torch(z, self.numerator, self.denominator)

    def _rational_backward(self, grad_r: torch.Tensor, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if z.is_cuda:
            try:
                _rational_fwd_triton, rational_bwd_triton, _rational_autograd = _flashkat_rational_kernels()
                dz, dn, dd = rational_bwd_triton(grad_r.contiguous().unsqueeze(1), z.contiguous().unsqueeze(1), self.numerator, self.denominator)
                return dz.squeeze(1), dn, dd
            except Exception:
                pass
        return _grouped_rational_backward_torch(grad_r, z, self.numerator, self.denominator)

    def hidden(self, x: torch.Tensor) -> torch.Tensor:
        z = self._norm_input(x)
        r = self._rational_forward(z, autograd_enabled=torch.is_grad_enabled())
        pre = (r @ self.w1) / math.sqrt(max(1, self.input_dim))
        if self.input_cross_hidden_enabled and self.input_cross_rank > 0:
            cross = None if self.input_cross_hidden_bucket_enabled else self._input_cross(z)
            pre = pre + self._input_cross_hidden_delta(cross, z)
        if self.hidden_bias_enabled:
            pre = pre + self.hidden_bias
        return self._apply_w2_bias_row(torch.tanh(pre))

    def _hidden_square_readout_features(self, h: torch.Tensor) -> torch.Tensor:
        if self.hidden_square_readout_signed:
            sq = h * h.abs()
        else:
            sq = h.square()
        if self.hidden_square_readout_centered:
            sq = sq - sq.mean(dim=1, keepdim=True)
        return sq

    def _hidden_abs_readout_features(self, h: torch.Tensor) -> torch.Tensor:
        abs_h = h.abs()
        if self.hidden_abs_readout_centered:
            abs_h = abs_h - float(self.hidden_abs_center_mix) * abs_h.mean(dim=1, keepdim=True)
        return abs_h

    def _hidden_rat_readout_features(self, h: torch.Tensor) -> torch.Tensor:
        rat_h = h / (1.0 + h.abs())
        if self.hidden_rat_readout_centered:
            center = rat_h.mean(dim=1, keepdim=True)
            if self.hidden_rat_readout_center_stopgrad:
                center = center.detach()
            rat_h = rat_h - float(self.hidden_rat_center_mix) * center
        return rat_h

    def _apply_hidden_rat_residual(self, h: torch.Tensor) -> torch.Tensor:
        rat_scale = float(getattr(self, "hidden_rat_residual_scale", 0.0))
        tanh_scale = float(getattr(self, "hidden_tanh_residual_scale", 0.0))
        center_tanh_scale = float(getattr(self, "hidden_center_tanh_residual_scale", 0.0))
        sign_sq_scale = float(getattr(self, "hidden_signsq_residual_scale", 0.0))
        out = h
        if rat_scale != 0.0:
            out = out + rat_scale * (h / (1.0 + h.abs()))
        if tanh_scale != 0.0:
            out = out + tanh_scale * torch.tanh(h)
        if center_tanh_scale != 0.0:
            tanh_h = torch.tanh(h)
            out = out + center_tanh_scale * (tanh_h - tanh_h.mean(dim=1, keepdim=True))
        if sign_sq_scale != 0.0:
            out = out + sign_sq_scale * (h * h.abs() / (1.0 + h.square()))
        return out

    def _hidden_rat_residual_vjp(self, h_base: torch.Tensor, grad_h_eff: torch.Tensor) -> torch.Tensor:
        rat_scale = float(getattr(self, "hidden_rat_residual_scale", 0.0))
        tanh_scale = float(getattr(self, "hidden_tanh_residual_scale", 0.0))
        center_tanh_scale = float(getattr(self, "hidden_center_tanh_residual_scale", 0.0))
        sign_sq_scale = float(getattr(self, "hidden_signsq_residual_scale", 0.0))
        if rat_scale == 0.0 and tanh_scale == 0.0 and center_tanh_scale == 0.0 and sign_sq_scale == 0.0:
            return grad_h_eff
        grad = grad_h_eff
        if rat_scale != 0.0:
            grad = grad + grad_h_eff * rat_scale / (1.0 + h_base.abs()).square()
        tanh_h = None
        if tanh_scale != 0.0:
            tanh_h = torch.tanh(h_base)
            grad = grad + grad_h_eff * tanh_scale * (1.0 - tanh_h.square())
        if center_tanh_scale != 0.0:
            if tanh_h is None:
                tanh_h = torch.tanh(h_base)
            grad_center = grad_h_eff - grad_h_eff.mean(dim=1, keepdim=True)
            grad = grad + center_tanh_scale * grad_center * (1.0 - tanh_h.square())
        if sign_sq_scale != 0.0:
            denom = 1.0 + h_base.square()
            grad = grad + grad_h_eff * sign_sq_scale * (2.0 * h_base.abs() / denom.square())
        return grad

    def frozen_readout_features(self, x: torch.Tensor) -> torch.Tensor:
        z = self._norm_input(x)
        r = self._rational_forward(z, autograd_enabled=False)
        pre = (r @ self.w1) / math.sqrt(max(1, self.input_dim))
        cross = None
        if self.input_cross_hidden_enabled and self.input_cross_rank > 0:
            cross = None if self.input_cross_hidden_bucket_enabled else self._input_cross(z)
            pre = pre + self._input_cross_hidden_delta(cross, z)
        if self.hidden_bias_enabled:
            pre = pre + self.hidden_bias
        h_base = self._apply_w2_bias_row(torch.tanh(pre))
        h = self._apply_hidden_rat_residual(h_base)
        feats = [h]
        if hasattr(self, "hidden_square_readout"):
            feats.append(float(self.hidden_square_readout_scale) * self._hidden_square_readout_features(h))
        if hasattr(self, "hidden_abs_readout"):
            feats.append(float(self.hidden_abs_readout_scale) * self._hidden_abs_readout_features(h))
        if hasattr(self, "hidden_rat_readout"):
            feats.append(float(self.hidden_rat_readout_scale) * self._hidden_rat_readout_features(h))
        if self.linear_residual_enabled:
            feats.append(self._linear_residual_effective_scale() * z)
        if hasattr(self, "proj_square_readout"):
            feats.append(self._input_projected_square_features(z))
        if hasattr(self, "proj_bilinear_readout"):
            feats.append(self._input_projected_bilinear_features(z))
        if self.input_cross_rank > 0 and self.input_cross_pca_rank > 0 and hasattr(self, "cross_pca_readout"):
            cross_feats = cross if cross is not None else self._input_cross(z)
            feats.append(self._input_cross_pca_features(cross_feats))
        elif self.input_cross_rank > 0 and self.input_cross_signal_rank > 0 and hasattr(self, "cross_signal_readout"):
            cross_feats = cross if cross is not None else self._input_cross(z)
            feats.append(self._cross_signal_features(cross_feats))
        elif self.input_cross_rank > 0 and hasattr(self, "cross_readout"):
            if self.input_cross_readout_bucket_enabled:
                cross_feats = self._input_cross_readout_bucket_features(z)
            else:
                cross_feats = cross if cross is not None else self._input_cross(z)
            feats.append(float(self.input_cross_readout_scale) * cross_feats)
        if hasattr(self, "bias"):
            feats.append(torch.ones((int(x.shape[0]), 1), device=x.device, dtype=h.dtype))
        return torch.cat(feats, dim=1) if len(feats) > 1 else h

    def _apply_logit_rms_norm(self, logits: torch.Tensor) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        mix = float(getattr(self, "logit_batch_rms_mix_sg", 0.0))
        if not (self.logit_rms_norm_sg_enabled or self.logit_batch_rms_norm_sg_enabled or mix != 0.0):
            return logits, None
        scale = self.logit_rms_norm_scale.to(device=logits.device, dtype=logits.dtype)
        if self.logit_batch_rms_norm_sg_enabled or mix != 0.0:
            rms = logits.detach().square().mean().view(1, 1).add(1.0e-6).sqrt()
        else:
            rms = logits.detach().square().mean(dim=1, keepdim=True).add(1.0e-6).sqrt()
        inv = scale / rms.clamp_min(1.0e-6)
        if mix != 0.0:
            inv = (1.0 - mix) + mix * inv
        return logits * inv, inv

    def _logit_norm_kernel_suffix(self) -> str:
        if float(getattr(self, "logit_batch_rms_mix_sg", 0.0)) != 0.0:
            return "_logitbatchrmsmixsg"
        if self.logit_batch_rms_norm_sg_enabled:
            return "_logitbatchrmsnormsg"
        return "_logitrmsnormsg" if self.logit_rms_norm_sg_enabled else ""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self._norm_input(x)
        r = self._rational_forward(z, autograd_enabled=torch.is_grad_enabled())
        pre = (r @ self.w1) / math.sqrt(max(1, self.input_dim))
        cross = None
        needs_cross_tensor = (
            (
                hasattr(self, "cross_readout")
                and not self.input_cross_readout_bucket_enabled
                and self.input_cross_signal_rank <= 0
                and self.input_cross_pca_rank <= 0
            )
            or self.input_cross_signal_rank > 0
            or self.input_cross_pca_rank > 0
            or (self.input_cross_hidden_enabled and not self.input_cross_hidden_bucket_enabled)
        )
        if self.input_cross_rank > 0 and needs_cross_tensor:
            cross = self._input_cross(z)
        if self.input_cross_hidden_enabled:
            pre = pre + self._input_cross_hidden_delta(cross, z)
        if self.hidden_bias_enabled:
            pre = pre + self.hidden_bias
        h_base = self._apply_w2_bias_row(torch.tanh(pre))
        h = self._apply_hidden_rat_residual(h_base)
        logits = (h @ self.w2) / math.sqrt(max(1, self.hidden_dim))
        if hasattr(self, "hidden_square_readout"):
            hidden_sq = self._hidden_square_readout_features(h)
            logits = logits + float(self.hidden_square_readout_scale) * (hidden_sq @ self.hidden_square_readout) / math.sqrt(max(1, self.hidden_dim))
        if hasattr(self, "hidden_abs_readout"):
            hidden_abs = self._hidden_abs_readout_features(h)
            logits = logits + float(self.hidden_abs_readout_scale) * (hidden_abs @ self.hidden_abs_readout) / math.sqrt(max(1, self.hidden_dim))
        if hasattr(self, "hidden_rat_readout"):
            hidden_rat = self._hidden_rat_readout_features(h)
            logits = logits + float(self.hidden_rat_readout_scale) * (hidden_rat @ self.hidden_rat_readout) / math.sqrt(max(1, self.hidden_dim))
        if self.linear_residual_enabled:
            logits = logits + self._linear_residual_effective_scale() * (z @ self.linear_readout) / math.sqrt(max(1, self.input_dim))
        if hasattr(self, "proj_square_readout"):
            sq_feats = self._input_projected_square_features(z)
            logits = logits + (sq_feats @ self.proj_square_readout) / self._input_projected_square_denominator()
        if hasattr(self, "proj_bilinear_readout"):
            bilin_feats = self._input_projected_bilinear_features(z)
            logits = logits + (bilin_feats @ self.proj_bilinear_readout) / self._input_projected_bilinear_denominator()
        if self.input_cross_pca_rank > 0 and cross is not None and hasattr(self, "cross_pca_readout"):
            logits = logits + self._cross_pca_logits(cross)
        elif self.input_cross_signal_rank > 0 and cross is not None and hasattr(self, "cross_signal_readout"):
            logits = logits + self._cross_signal_logits(cross)
        elif hasattr(self, "cross_readout") and self.input_cross_readout_bucket_enabled:
            bucket_cross = self._input_cross_readout_bucket_features(z)
            pair_logits = float(self.input_cross_readout_scale) * (bucket_cross @ self.cross_readout) / self._input_cross_readout_denominator()
            logits = logits + self._bounded_pair_logits(pair_logits)
        elif cross is not None and hasattr(self, "cross_readout"):
            pair_logits = float(self.input_cross_readout_scale) * (cross @ self.cross_readout) / self._input_cross_readout_denominator()
            logits = logits + self._bounded_pair_logits(pair_logits)
        if hasattr(self, "bias"):
            logits = logits + self.bias
        logits, _ = self._apply_logit_rms_norm(logits)
        return logits

    def manual_kernel_variant(self) -> str:
        logit_suffix = self._logit_norm_kernel_suffix()
        if self.input_cross_pca_rank > 0 and self.input_cross_rank > 0:
            suffix = ("_hiddenresvjptriton" if self.hidden_rat_residual_vjp_triton_enabled else "") + logit_suffix
            if self.readout_only_backbone_frozen_enabled:
                if self.hidden_bias_enabled:
                    return f"rational_flashkat_grouped_paircross_pca_gemm_frozenbackbone_hiddenbias{suffix}_l3"
                return f"rational_flashkat_grouped_paircross_pca_gemm_frozenbackbone{suffix}_l3"
            if self.rational_coefficients_frozen_enabled:
                return f"rational_flashkat_grouped_paircross_pca_gemm_freezerational{logit_suffix}_l3"
            return f"rational_flashkat_grouped_paircross_pca_gemm{logit_suffix}_l3"
        if self.input_cross_signal_rank > 0 and self.input_cross_rank > 0:
            suffix = ("_hiddenresvjptriton" if self.hidden_rat_residual_vjp_triton_enabled else "") + logit_suffix
            if self.readout_only_backbone_frozen_enabled:
                if self.hidden_bias_enabled:
                    return f"rational_flashkat_grouped_paircross_signal_gemm_frozenbackbone_hiddenbias{suffix}_l3"
                return f"rational_flashkat_grouped_paircross_signal_gemm_frozenbackbone{suffix}_l3"
            if self.rational_coefficients_frozen_enabled:
                return f"rational_flashkat_grouped_paircross_signal_gemm_freezerational{logit_suffix}_l3"
            return f"rational_flashkat_grouped_paircross_signal_gemm{logit_suffix}_l3"
        if (
            self.readout_only_backbone_frozen_enabled
            and self.input_cross_rank > 0
            and self.input_cross_readout_block_triton_enabled
            and self.input_cross_mode in {"pair_index", "balanced_pair_index", "grid_pair_index", "hybrid_pair_index", "blend_pair_index", "balanced_blend_pair_index", "diag_pair_index"}
        ):
            if self.hidden_rat_residual_vjp_triton_enabled:
                suffix = "_hiddenresvjptriton"
            else:
                suffix = "_hiddentailgradtriton" if self.hidden_tail_readout_vjp_triton_enabled else ""
            suffix += logit_suffix
            if self.hidden_bias_enabled:
                return f"rational_flashkat_grouped_paircross_readout_block_triton_frozenbackbone_hiddenbias{suffix}_l3"
            return f"rational_flashkat_grouped_paircross_readout_block_triton_frozenbackbone{suffix}_l3"
        if (
            self.rational_coefficients_frozen_enabled
            and self.input_cross_rank > 0
            and self.input_cross_readout_block_triton_enabled
            and self.input_cross_mode in {"pair_index", "balanced_pair_index", "grid_pair_index", "hybrid_pair_index", "blend_pair_index", "balanced_blend_pair_index", "diag_pair_index"}
        ):
            return f"rational_flashkat_grouped_paircross_readout_block_triton_freezerational{logit_suffix}_l3"
        if self.input_projected_bilinear_rank > 0 and not hasattr(self, "cross_readout"):
            return "rational_flashkat_grouped_projbilin_readout_gemm_l3"
        if self.input_projected_square_rank > 0 and not hasattr(self, "cross_readout"):
            return "rational_flashkat_grouped_projsq_readout_gemm_l3"
        if self.input_cross_rank > 0:
            if self.input_cross_hidden_enabled:
                if self.input_cross_hidden_bucket_enabled:
                    if self.input_cross_hidden_direct_readout_enabled and int(self.input_cross_hidden_rank or self.input_cross_rank) != int(self.input_cross_rank):
                        return "rational_flashkat_grouped_paircross_bucket_triton_l3"
                    if self.input_cross_hidden_direct_readout_enabled:
                        return "rational_flashkat_grouped_pairbucket_direct_triton_l3"
                    if _pairbucket_hidden_delta_kernel is not None:
                        return "rational_flashkat_grouped_pairbucket_triton_l3"
                    return "rational_flashkat_grouped_pairbucket_l3"
                return "rational_flashkat_grouped_pairhidden_gemm_l3"
            if self.input_cross_readout_bucket_enabled:
                return "rational_flashkat_grouped_pairreadbucket_triton_l3"
            if self.input_cross_readout_block_triton_enabled and self.input_cross_mode in {"pair_index", "balanced_pair_index", "grid_pair_index", "hybrid_pair_index", "blend_pair_index", "balanced_blend_pair_index", "diag_pair_index"}:
                return f"rational_flashkat_grouped_paircross_readout_block_triton{logit_suffix}_l3"
            if self.input_cross_readout_triton_enabled and self.input_cross_mode in {"pair_index", "balanced_pair_index", "grid_pair_index", "hybrid_pair_index", "blend_pair_index", "balanced_blend_pair_index", "diag_pair_index"}:
                return f"rational_flashkat_grouped_paircross_readout_triton{logit_suffix}_l3"
            if self.input_cross_mode in {"pair_index", "balanced_pair_index", "grid_pair_index", "hybrid_pair_index", "blend_pair_index", "balanced_blend_pair_index", "diag_pair_index"}:
                return f"rational_flashkat_grouped_paircross_gemm{logit_suffix}_l3"
            if self.input_cross_mode == "transform_pair_index":
                return "rational_flashkat_grouped_transformpair_gemm_l3"
            return "rational_flashkat_grouped_inputcross_gemm_l3"
        return f"rational_flashkat_grouped_triton_l3_gemm{logit_suffix}"

    def manual_ce_forward_cache(self, x: torch.Tensor):
        with torch.no_grad():
            z = self._norm_input(x)
            r = self._rational_forward(z, autograd_enabled=False)
            pre = (r @ self.w1) / math.sqrt(max(1, self.input_dim))
            cross = None
            needs_cross_tensor = (
                (
                    hasattr(self, "cross_readout")
                    and self.input_cross_pca_rank <= 0
                    and self.input_cross_signal_rank <= 0
                    and not (self.input_cross_readout_bucket_enabled or self.input_cross_readout_triton_enabled or self.input_cross_readout_block_triton_enabled)
                )
                or self.input_cross_signal_rank > 0
                or self.input_cross_pca_rank > 0
                or (self.input_cross_hidden_enabled and not self.input_cross_hidden_bucket_enabled)
            )
            if self.input_cross_rank > 0 and needs_cross_tensor:
                cross = self._input_cross(z)
            if self.input_cross_hidden_enabled:
                pre = pre + self._input_cross_hidden_delta(cross, z)
            if self.hidden_bias_enabled:
                pre = pre + self.hidden_bias
            h_base = self._apply_w2_bias_row(torch.tanh(pre))
            h = self._apply_hidden_rat_residual(h_base)
            logits = (h @ self.w2) / math.sqrt(max(1, self.hidden_dim))
            if hasattr(self, "hidden_square_readout"):
                hidden_sq = self._hidden_square_readout_features(h)
                logits = logits + float(self.hidden_square_readout_scale) * (hidden_sq @ self.hidden_square_readout) / math.sqrt(max(1, self.hidden_dim))
            if hasattr(self, "hidden_abs_readout"):
                hidden_abs = self._hidden_abs_readout_features(h)
                logits = logits + float(self.hidden_abs_readout_scale) * (hidden_abs @ self.hidden_abs_readout) / math.sqrt(max(1, self.hidden_dim))
            if hasattr(self, "hidden_rat_readout"):
                hidden_rat = self._hidden_rat_readout_features(h)
                logits = logits + float(self.hidden_rat_readout_scale) * (hidden_rat @ self.hidden_rat_readout) / math.sqrt(max(1, self.hidden_dim))
            if self.linear_residual_enabled:
                logits = logits + self._linear_residual_effective_scale_value() * (z @ self.linear_readout) / math.sqrt(max(1, self.input_dim))
            bias = getattr(self, "bias", None)
            bias_fused = False
            pair_raw_logits = None
            proj_sq = None
            proj_bilin = None
            if hasattr(self, "proj_square_readout"):
                proj_sq = self._input_projected_square_features(z)
                logits = logits + (proj_sq @ self.proj_square_readout) / self._input_projected_square_denominator()
            if hasattr(self, "proj_bilinear_readout"):
                proj_bilin = self._input_projected_bilinear_features(z)
                logits = logits + (proj_bilin @ self.proj_bilinear_readout) / self._input_projected_bilinear_denominator()
            if self.input_cross_pca_rank > 0 and cross is not None and hasattr(self, "cross_pca_readout"):
                logits = logits + self._cross_pca_logits(cross)
            elif self.input_cross_signal_rank > 0 and cross is not None and hasattr(self, "cross_signal_readout"):
                logits = logits + self._cross_signal_logits(cross)
            elif hasattr(self, "cross_readout") and self.input_cross_readout_bucket_enabled:
                cross = self._input_cross_readout_bucket_features(z)
                pair_raw_logits = float(self.input_cross_readout_scale) * (cross @ self.cross_readout) / self._input_cross_readout_denominator()
                logits = logits + self._bounded_pair_logits(pair_raw_logits)
            elif cross is not None and hasattr(self, "cross_readout"):
                pair_raw_logits = float(self.input_cross_readout_scale) * (cross @ self.cross_readout) / self._input_cross_readout_denominator()
                logits = logits + self._bounded_pair_logits(pair_raw_logits)
            elif hasattr(self, "cross_readout") and (self.input_cross_readout_triton_enabled or self.input_cross_readout_block_triton_enabled):
                if self.input_cross_readout_block_triton_enabled:
                    pair_mean, pair_inv_std = self._pair_norm_buffers_for_kernel()
                    fused_cap_value = None
                    if self.input_cross_logit_cap_fused_enabled:
                        fused_cap_value = float(self.input_cross_logit_cap_value.detach().cpu().item())
                    pair_logits = _paircross_readout_logits_block_triton(
                        z,
                        self.input_cross_pair_left,
                        self.input_cross_pair_right,
                        self.cross_readout,
                        float(self.input_cross_readout_scale),
                        bias=None if self.input_cross_logit_cap_enabled else bias,
                        feat_mean=pair_mean,
                        feat_inv_std=pair_inv_std,
                        block_b_override=int(self.input_cross_readout_block_batch),
                        cap_value=fused_cap_value,
                        feature_mode=self._input_pair_feature_mode_id(),
                    )
                    bias_fused = (not self.input_cross_logit_cap_enabled) and bias is not None and pair_logits is not None
                else:
                    pair_logits = _paircross_readout_logits_triton(
                        z,
                        self.input_cross_pair_left,
                        self.input_cross_pair_right,
                        self.cross_readout,
                        float(self.input_cross_readout_scale),
                        feature_mode=self._input_pair_feature_mode_id(),
                    )
                if pair_logits is None and not self.input_cross_pair_norm_enabled:
                    pair_logits = _paircross_readout_logits_triton(
                        z,
                        self.input_cross_pair_left,
                        self.input_cross_pair_right,
                        self.cross_readout,
                        float(self.input_cross_readout_scale),
                        feature_mode=self._input_pair_feature_mode_id(),
                    )
                if pair_logits is None:
                    cross = self._input_cross(z)
                    pair_raw_logits = float(self.input_cross_readout_scale) * (cross @ self.cross_readout) / self._input_cross_readout_denominator()
                    logits = logits + self._bounded_pair_logits(pair_raw_logits)
                    bias_fused = False
                else:
                    if self.input_cross_logit_cap_fused_enabled:
                        pair_raw_logits = None
                        logits = logits + pair_logits
                    else:
                        pair_raw_logits = pair_logits
                        logits = logits + self._bounded_pair_logits(pair_logits)
            if bias is not None and not bias_fused:
                logits = logits + self.bias
            logits, logit_norm_inv = self._apply_logit_rms_norm(logits)
            return logits, (self.manual_kernel_variant(), z, r, h, cross, pair_raw_logits, proj_sq, proj_bilin, h_base, logit_norm_inv)

    def _pair_readout_grad_for_manual(self, z: torch.Tensor, grad_logits: torch.Tensor) -> torch.Tensor:
        if self.input_cross_readout_block_triton_enabled:
            pair_mean, pair_inv_std = self._pair_norm_buffers_for_kernel()
            if self.input_cross_logit_cap_fused_enabled:
                cap_value = float(self.input_cross_logit_cap_value.detach().cpu().item())
                grad_cross = _paircross_readout_grad_block_cap_triton(
                    z,
                    self.input_cross_pair_left,
                    self.input_cross_pair_right,
                    self.cross_readout,
                    grad_logits,
                    float(self.input_cross_readout_scale),
                    cap_value,
                    feat_mean=pair_mean,
                    feat_inv_std=pair_inv_std,
                    block_r_override=int(self.input_cross_readout_grad_block_rank),
                    feature_mode=self._input_pair_feature_mode_id(),
                )
                if grad_cross is not None:
                    return grad_cross
            if self.input_cross_readout_grad_atomic_enabled:
                grad_cross = _paircross_readout_grad_block_atomic_triton(
                    z,
                    self.input_cross_pair_left,
                    self.input_cross_pair_right,
                    grad_logits,
                    float(self.input_cross_readout_scale),
                    feat_mean=pair_mean,
                    feat_inv_std=pair_inv_std,
                    block_r_override=int(self.input_cross_readout_grad_block_rank),
                    block_b_override=int(self.input_cross_readout_block_batch),
                    feature_mode=self._input_pair_feature_mode_id(),
                )
                if grad_cross is not None:
                    return grad_cross
            grad_cross = _paircross_readout_grad_block_triton(
                z,
                self.input_cross_pair_left,
                self.input_cross_pair_right,
                grad_logits,
                float(self.input_cross_readout_scale),
                feat_mean=pair_mean,
                feat_inv_std=pair_inv_std,
                block_r_override=int(self.input_cross_readout_grad_block_rank),
                feature_mode=self._input_pair_feature_mode_id(),
            )
            if grad_cross is not None:
                return grad_cross
        if not self.input_cross_pair_norm_enabled:
            grad_cross = _paircross_readout_grad_triton(
                z,
                self.input_cross_pair_left,
                self.input_cross_pair_right,
                grad_logits,
                float(self.input_cross_readout_scale),
                feature_mode=self._input_pair_feature_mode_id(),
            )
            if grad_cross is not None:
                return grad_cross
        cross_fallback = self._input_cross(z)
        return float(self.input_cross_readout_scale) * (cross_fallback.transpose(0, 1) @ grad_logits) / math.sqrt(max(1, self.input_cross_rank))

    def manual_logits_backward_from_cache(self, logits: torch.Tensor, cache, grad_logits: torch.Tensor) -> None:
        with torch.no_grad():
            h_base = None
            if len(cache) == 6:
                _variant, z, r, h, cross, pair_raw_logits = cache
                proj_sq = None
                proj_bilin = None
                logit_norm_inv = None
            elif len(cache) == 7:
                _variant, z, r, h, cross, pair_raw_logits, proj_sq = cache
                proj_bilin = None
                logit_norm_inv = None
            elif len(cache) == 8:
                _variant, z, r, h, cross, pair_raw_logits, proj_sq, proj_bilin = cache
                logit_norm_inv = None
            elif len(cache) == 9:
                _variant, z, r, h, cross, pair_raw_logits, proj_sq, proj_bilin, h_base = cache
                logit_norm_inv = None
            else:
                _variant, z, r, h, cross, pair_raw_logits, proj_sq, proj_bilin, h_base, logit_norm_inv = cache
            if h_base is None:
                h_base = h
            if logit_norm_inv is not None:
                grad_logits = grad_logits * logit_norm_inv.to(device=grad_logits.device, dtype=grad_logits.dtype)
            grad_pair_logits = self._bounded_pair_grad_logits(pair_raw_logits, grad_logits)
            use_residual_w2_vjp_triton = (
                self.hidden_rat_residual_vjp_triton_enabled
                and (
                    float(getattr(self, "hidden_rat_residual_scale", 0.0)) != 0.0
                    or float(getattr(self, "hidden_tanh_residual_scale", 0.0)) != 0.0
                    or float(getattr(self, "hidden_center_tanh_residual_scale", 0.0)) != 0.0
                    or float(getattr(self, "hidden_signsq_residual_scale", 0.0)) != 0.0
                )
                and not hasattr(self, "hidden_square_readout")
                and not hasattr(self, "hidden_abs_readout")
                and not hasattr(self, "hidden_rat_readout")
            )
            grad_h_is_base = False
            residual_w2_vjp = None
            if use_residual_w2_vjp_triton:
                residual_mode = 5
                residual_scale = float(getattr(self, "hidden_rat_residual_scale", 0.0))
                if float(getattr(self, "hidden_tanh_residual_scale", 0.0)) != 0.0:
                    residual_mode = 6
                    residual_scale = float(getattr(self, "hidden_tanh_residual_scale", 0.0))
                if float(getattr(self, "hidden_center_tanh_residual_scale", 0.0)) != 0.0:
                    residual_mode = 7
                    residual_scale = float(getattr(self, "hidden_center_tanh_residual_scale", 0.0))
                if float(getattr(self, "hidden_signsq_residual_scale", 0.0)) != 0.0:
                    residual_mode = 8
                    residual_scale = float(getattr(self, "hidden_signsq_residual_scale", 0.0))
                residual_w2_vjp = _hidden_tail_readout_vjp_triton(
                    h_base,
                    self.w2,
                    grad_logits,
                    1.0,
                    mode=residual_mode,
                    center_mix=residual_scale,
                )
            if residual_w2_vjp is not None:
                self.w2.grad = residual_w2_vjp[0].to(dtype=self.w2.dtype)
                grad_h = residual_w2_vjp[1].to(dtype=h.dtype)
                grad_h_is_base = True
            else:
                self.w2.grad = (h.transpose(0, 1) @ grad_logits) / math.sqrt(max(1, self.hidden_dim))
                grad_h = (grad_logits @ self.w2.transpose(0, 1)) / math.sqrt(max(1, self.hidden_dim))
            if hasattr(self, "hidden_square_readout"):
                sq_scale = float(self.hidden_square_readout_scale)
                hidden_sq = self._hidden_square_readout_features(h)
                self.hidden_square_readout.grad = sq_scale * (hidden_sq.transpose(0, 1) @ grad_logits) / math.sqrt(max(1, self.hidden_dim))
                grad_hidden_sq = (grad_logits @ self.hidden_square_readout.transpose(0, 1)) / math.sqrt(max(1, self.hidden_dim))
                if self.hidden_square_readout_centered:
                    grad_hidden_sq = grad_hidden_sq - grad_hidden_sq.mean(dim=1, keepdim=True)
                if self.hidden_square_readout_signed:
                    grad_h = grad_h + sq_scale * 2.0 * h.abs() * grad_hidden_sq
                else:
                    grad_h = grad_h + sq_scale * 2.0 * h * grad_hidden_sq
            if hasattr(self, "hidden_abs_readout"):
                abs_scale = float(self.hidden_abs_readout_scale)
                triton_vjp = None
                if self.hidden_tail_readout_vjp_triton_enabled and not self.hidden_abs_readout_centered:
                    triton_vjp = _hidden_tail_readout_vjp_triton(h, self.hidden_abs_readout, grad_logits, abs_scale, mode=1)
                if triton_vjp is not None:
                    self.hidden_abs_readout.grad = triton_vjp[0].to(dtype=self.hidden_abs_readout.dtype)
                    grad_h = grad_h + triton_vjp[1].to(dtype=grad_h.dtype)
                else:
                    hidden_abs = self._hidden_abs_readout_features(h)
                    self.hidden_abs_readout.grad = abs_scale * (hidden_abs.transpose(0, 1) @ grad_logits) / math.sqrt(max(1, self.hidden_dim))
                    grad_hidden_abs = (grad_logits @ self.hidden_abs_readout.transpose(0, 1)) / math.sqrt(max(1, self.hidden_dim))
                    if self.hidden_abs_readout_centered:
                        grad_hidden_abs = grad_hidden_abs - float(self.hidden_abs_center_mix) * grad_hidden_abs.mean(dim=1, keepdim=True)
                    grad_h = grad_h + abs_scale * h.sign() * grad_hidden_abs
            if hasattr(self, "hidden_rat_readout"):
                rat_scale = float(self.hidden_rat_readout_scale)
                triton_vjp = None
                if self.hidden_tail_readout_vjp_triton_enabled:
                    rat_mode = 2
                    if self.hidden_rat_readout_centered:
                        rat_mode = 4 if self.hidden_rat_readout_center_stopgrad else 3
                    triton_vjp = _hidden_tail_readout_vjp_triton(
                        h,
                        self.hidden_rat_readout,
                        grad_logits,
                        rat_scale,
                        mode=rat_mode,
                        center_mix=float(self.hidden_rat_center_mix),
                    )
                if triton_vjp is not None:
                    self.hidden_rat_readout.grad = triton_vjp[0].to(dtype=self.hidden_rat_readout.dtype)
                    grad_h = grad_h + triton_vjp[1].to(dtype=grad_h.dtype)
                else:
                    hidden_rat = self._hidden_rat_readout_features(h)
                    self.hidden_rat_readout.grad = rat_scale * (hidden_rat.transpose(0, 1) @ grad_logits) / math.sqrt(max(1, self.hidden_dim))
                    grad_hidden_rat = (grad_logits @ self.hidden_rat_readout.transpose(0, 1)) / math.sqrt(max(1, self.hidden_dim))
                    if self.hidden_rat_readout_centered and not self.hidden_rat_readout_center_stopgrad:
                        grad_hidden_rat = grad_hidden_rat - float(self.hidden_rat_center_mix) * grad_hidden_rat.mean(dim=1, keepdim=True)
                    rat_deriv = 1.0 / (1.0 + h.abs()).square()
                    grad_h = grad_h + rat_scale * rat_deriv * grad_hidden_rat
            if self.linear_residual_enabled:
                linear_base = (z @ self.linear_readout) / math.sqrt(max(1, self.input_dim))
                self.linear_readout.grad = self._linear_residual_effective_scale_value() * (z.transpose(0, 1) @ grad_logits) / math.sqrt(max(1, self.input_dim))
                if hasattr(self, "linear_residual_gate"):
                    gate_grad = float(self.linear_residual_scale) * (linear_base * grad_logits).sum()
                    self.linear_residual_gate.grad = gate_grad.reshape(1).to(device=self.linear_residual_gate.device, dtype=self.linear_residual_gate.dtype)
            if hasattr(self, "proj_square_readout"):
                if proj_sq is None:
                    proj_sq = self._input_projected_square_features(z)
                self.proj_square_readout.grad = (proj_sq.transpose(0, 1) @ grad_logits) / self._input_projected_square_denominator()
            if hasattr(self, "proj_bilinear_readout"):
                if proj_bilin is None:
                    proj_bilin = self._input_projected_bilinear_features(z)
                self.proj_bilinear_readout.grad = (proj_bilin.transpose(0, 1) @ grad_logits) / self._input_projected_bilinear_denominator()
            if self.input_cross_pca_rank > 0 and cross is not None and hasattr(self, "cross_pca_readout"):
                self._accumulate_cross_pca_grads(cross, grad_logits)
            elif self.input_cross_signal_rank > 0 and cross is not None and hasattr(self, "cross_signal_readout"):
                self._accumulate_cross_signal_grads(cross, grad_logits)
            elif cross is not None and hasattr(self, "cross_readout"):
                self.cross_readout.grad = float(self.input_cross_readout_scale) * (cross.transpose(0, 1) @ grad_pair_logits) / self._input_cross_readout_denominator()
            elif hasattr(self, "cross_readout") and (self.input_cross_readout_triton_enabled or self.input_cross_readout_block_triton_enabled):
                self.cross_readout.grad = self._pair_readout_grad_for_manual(z, grad_pair_logits).to(dtype=self.cross_readout.dtype)
            if hasattr(self, "bias"):
                self.bias.grad = grad_logits.sum(dim=0).to(dtype=self.bias.dtype)
            if self.readout_only_backbone_frozen_enabled:
                if self.hidden_bias_enabled:
                    grad_h_base = grad_h if grad_h_is_base else self._hidden_rat_residual_vjp(h_base, grad_h)
                    grad_pre = grad_h_base * (1.0 - h_base.square())
                    if self.w2_bias_row_enabled and int(grad_pre.shape[1]) > 0:
                        grad_pre[:, -1] = 0.0
                    self.hidden_bias.grad = grad_pre.sum(dim=0).to(dtype=self.hidden_bias.dtype)
                return None
            grad_h_base = grad_h if grad_h_is_base else self._hidden_rat_residual_vjp(h_base, grad_h)
            grad_pre = grad_h_base * (1.0 - h_base.square())
            if self.w2_bias_row_enabled and int(grad_pre.shape[1]) > 0:
                grad_pre[:, -1] = 0.0
            if self.hidden_bias_enabled:
                self.hidden_bias.grad = grad_pre.sum(dim=0).to(dtype=self.hidden_bias.dtype)
            self.w1.grad = (r.transpose(0, 1) @ grad_pre) / math.sqrt(max(1, self.input_dim))
            if self.rational_coefficients_frozen_enabled:
                return None
            grad_r = (grad_pre @ self.w1.transpose(0, 1)) / math.sqrt(max(1, self.input_dim))
            _dz, dn, dd = self._rational_backward(grad_r, z)
            self.numerator.grad = dn.to(dtype=self.numerator.dtype)
            self.denominator.grad = dd.to(dtype=self.denominator.dtype)
            return None

    def manual_ce_backward_from_cache(self, logits: torch.Tensor, cache, y: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            probs = torch.softmax(logits, dim=1)
            loss = F.cross_entropy(logits, y)
            grad_logits = probs
            grad_logits[torch.arange(int(y.numel()), device=y.device), y] -= 1.0
            grad_logits = grad_logits / float(max(1, int(y.numel())))
            self.manual_logits_backward_from_cache(logits, cache, grad_logits)
            return loss

    def manual_gradient_audit(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
        params = [p for p in self.parameters() if p.requires_grad]
        self.zero_grad(set_to_none=True)
        logits, cache = self.manual_ce_forward_cache(x)
        manual_logits = logits.detach().clone()
        self.manual_ce_backward_from_cache(logits, cache, y)
        manual_grads = [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in params]
        self.zero_grad(set_to_none=True)
        ref_logits = self.forward(x)
        F.cross_entropy(ref_logits, y).backward()
        ref_grads = [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in params]
        self.zero_grad(set_to_none=True)
        relerrs = []
        coses = []
        for gm, gr in zip(manual_grads, ref_grads):
            denom = gr.norm().clamp_min(1.0e-8)
            relerrs.append(float((gm - gr).norm().div(denom).detach().item()))
            coses.append(float(F.cosine_similarity(gm.flatten(), gr.flatten(), dim=0, eps=1.0e-8).detach().item()))
        return {
            "manual_forward_available": 1,
            "manual_backward_available": 1,
            "grad_relerr_max": max(relerrs) if relerrs else float("inf"),
            "grad_cos_min": min(coses) if coses else -1.0,
            "output_max_abs_error": float((manual_logits - ref_logits.detach()).abs().max().item()),
        }

    def basis_diagnostics(self, x: torch.Tensor) -> Dict[str, float]:
        with torch.no_grad():
            z = self._norm_input(x)
            feats = self.frozen_readout_features(x).float()
            energy = feats.square().mean(dim=0)
            prob = energy / energy.sum().clamp_min(EPS)
            entropy = -(prob * (prob + EPS).log()).sum() / math.log(max(2, int(prob.numel())))
            dead = (energy < 1.0e-8).float().mean()
            centered = feats - feats.mean(dim=0, keepdim=True)
            try:
                s = torch.linalg.svdvals(centered[: min(256, int(centered.shape[0]))])
                rank = (s.square().sum().square() / s.pow(4).sum().clamp_min(EPS)).item()
                cond = (s.max() / s[s > 1.0e-7].min()).item() if bool((s > 1.0e-7).any()) else float("inf")
            except RuntimeError:
                rank = 0.0
                cond = float("inf")
            diag = {
                "basis_entropy": float(entropy.item()),
                "dead_basis_fraction": float(dead.item()),
                "dead_basis_frac": float(dead.item()),
                "basis_effective_rank": float(rank),
                "basis_condition_proxy": float(cond),
                "feature_effective_rank": float(rank),
                "feature_condition_number": float(cond),
            }
            groups = int(self.numerator.shape[0])
            bsz, dim = int(z.shape[0]), int(z.shape[1])
            group_size = max(1, dim // groups)
            zg = z.reshape(bsz, groups, group_size).float()
            a = self.numerator.detach().float()[:, None, :]
            b_abs = self.denominator.detach().float().abs()[:, None, :]
            x2 = zg * zg
            x3 = x2 * zg
            x4 = x3 * zg
            x5 = x4 * zg
            ax = zg.abs()
            ax2 = ax * ax
            ax3 = ax2 * ax
            ax4 = ax3 * ax
            p = a[..., 0] + a[..., 1] * zg + a[..., 2] * x2 + a[..., 3] * x3 + a[..., 4] * x4 + a[..., 5] * x5
            q = 1.0 + b_abs[..., 0] * ax + b_abs[..., 1] * ax2 + b_abs[..., 2] * ax3 + b_abs[..., 3] * ax4
            p1 = a[..., 1] + 2.0 * a[..., 2] * zg + 3.0 * a[..., 3] * x2 + 4.0 * a[..., 4] * x3 + 5.0 * a[..., 5] * x4
            p2 = 2.0 * a[..., 2] + 6.0 * a[..., 3] * zg + 12.0 * a[..., 4] * x2 + 20.0 * a[..., 5] * x3
            sign_x = torch.where(zg < 0.0, torch.full_like(zg, -1.0), torch.ones_like(zg))
            q1 = sign_x * (b_abs[..., 0] + 2.0 * b_abs[..., 1] * ax + 3.0 * b_abs[..., 2] * ax2 + 4.0 * b_abs[..., 3] * ax3)
            q2 = 2.0 * b_abs[..., 1] + 6.0 * b_abs[..., 2] * ax + 12.0 * b_abs[..., 3] * ax2
            q_safe = q.clamp_min(EPS)
            r_prime = p1 / q_safe - p * q1 / q_safe.square()
            r_double_prime = p2 / q_safe - 2.0 * p1 * q1 / q_safe.square() - p * q2 / q_safe.square() + 2.0 * p * q1.square() / q_safe.pow(3)
            group_energy = (p / q_safe).square().mean(dim=(0, 2))
            diag.update(
                {
                    "den_min": float(q.min().item()),
                    "den_p01": float(torch.quantile(q.flatten(), 0.01).item()),
                    "den_condition": float((q.max() / q.min().clamp_min(EPS)).item()),
                    "r_prime_p95": float(torch.quantile(r_prime.abs().flatten(), 0.95).item()),
                    "r_double_prime_p95": float(torch.quantile(r_double_prime.abs().flatten(), 0.95).item()),
                    "group_function_diversity": float(group_energy.std(unbiased=False).item() / group_energy.mean().clamp_min(EPS).item()),
                    "group_dead_fraction": float((group_energy < 1.0e-8).float().mean().item()),
                }
            )
            return diag


class QuadraticSketchKAN(nn.Module):
    """Quadratic sketch primitive with learnable basis readout.

    Fixed-projection variants are intentionally marked diagnostic-only in the
    v12.4 manifest. Trainable-projection variants are official-capable
    candidates and must pass the same gates as every other base candidate.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        spec: PrimitiveSpec,
        x_for_stats: torch.Tensor,
        seed: int,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.spec = spec
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.hidden_dim = int(spec.hidden_dim)
        self.k = int(spec.k)
        xs = x_for_stats[: min(4096, int(x_for_stats.shape[0]))].to(device=device, dtype=torch.float32)
        self.register_buffer("mu", xs.mean(dim=0))
        self.register_buffer("std", xs.std(dim=0).clamp_min(1.0e-3))
        gen = torch.Generator(device=device).manual_seed(int(seed) + 7919)
        proj = torch.randn(self.input_dim, self.hidden_dim, device=device, generator=gen) / math.sqrt(max(1, self.input_dim))
        if spec.model_kind == "trainable_quadratic_sketch":
            self.proj = nn.Parameter(proj)
        else:
            self.register_buffer("proj", proj)
        self.readout = nn.Parameter(torch.randn(self.hidden_dim, self.k, self.output_dim, device=device, generator=gen) / math.sqrt(max(1, self.hidden_dim * self.k)))
        self.bias = nn.Parameter(torch.zeros(self.output_dim, device=device))

    @property
    def edge_param_count(self) -> int:
        proj_count = int(self.proj.numel()) if isinstance(self.proj, nn.Parameter) else 0
        return int(proj_count + self.readout.numel() + self.bias.numel())

    def _norm_input(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh((x - self.mu) / self.std)

    def _features_3d(self, x: torch.Tensor) -> torch.Tensor:
        z = self._norm_input(x) @ self.proj
        vals = [z]
        if self.k > 1:
            vals.append(z.square() - z.square().mean(dim=0, keepdim=True).detach())
        while len(vals) < self.k:
            vals.append(vals[-1] * z)
        return torch.stack(vals[: self.k], dim=2)

    def frozen_readout_features(self, x: torch.Tensor) -> torch.Tensor:
        feats = self._features_3d(x)
        return feats.reshape(int(x.shape[0]), -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self._features_3d(x)
        return torch.einsum("bhk,hkc->bc", feats, self.readout) + self.bias

    def basis_diagnostics(self, x: torch.Tensor) -> Dict[str, float]:
        with torch.no_grad():
            feats = self.frozen_readout_features(x).float()
            energy = feats.square().mean(dim=0)
            prob = energy / energy.sum().clamp_min(EPS)
            entropy = -(prob * (prob + EPS).log()).sum() / math.log(max(2, int(prob.numel())))
            dead = (energy < 1.0e-8).float().mean()
            centered = feats - feats.mean(dim=0, keepdim=True)
            try:
                s = torch.linalg.svdvals(centered[: min(256, int(centered.shape[0]))])
                rank = (s.square().sum().square() / s.pow(4).sum().clamp_min(EPS)).item()
                cond = (s.max() / s[s > 1.0e-7].min()).item() if bool((s > 1.0e-7).any()) else float("inf")
            except RuntimeError:
                rank = 0.0
                cond = float("inf")
            return {
                "basis_entropy": float(entropy.item()),
                "dead_basis_fraction": float(dead.item()),
                "basis_effective_rank": float(rank),
                "basis_condition_proxy": float(cond),
                "basis_output_norm_p95": float(torch.quantile(feats.norm(dim=1), 0.95).item()),
            }

    def basis_functional_direction(self, mode: str) -> List[torch.Tensor]:
        with torch.no_grad():
            direction = -self.readout.detach().clone()
            if mode in {"basis_aware_snr_projected", "basis_aware_orthogonal"}:
                mask = torch.ones_like(direction)
                mask[:, 0, :] = 0.25
                direction = direction * mask
            if isinstance(self.proj, nn.Parameter):
                return [torch.zeros_like(self.proj), direction, -self.bias.detach().clone()]
            return [direction, -self.bias.detach().clone()]


class LiteGatedLegendreQuadraticKAN(nn.Module):
    """Low-cost Legendre direct readout plus quadratic sketch primitive.

    B30 is a cheaper follow-up to the B21/B23/B27 GatedHybrid family.  It keeps
    the basis-specific input geometry idea, but removes the second Legendre KAN
    layer: the Legendre branch reads out directly from normalized input basis
    features while the quadratic branch keeps a trainable sketch for interaction
    coverage.  Both input normalizers are fixed from unlabeled train-stream
    statistics and never branch on dataset names or labels.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        spec: PrimitiveSpec,
        x_for_stats: torch.Tensor,
        seed: int,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.spec = spec
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.hidden_dim = int(spec.hidden_dim)
        self.k = 4
        xs = x_for_stats[: min(4096, int(x_for_stats.shape[0]))].to(device=device, dtype=torch.float32)
        self.register_buffer("mu", xs.mean(dim=0))
        self.register_buffer("std", xs.std(dim=0).clamp_min(1.0e-3))
        self.register_buffer("input_target_rms", torch.tensor([0.85], device=device))
        self.register_buffer("centers", torch.linspace(-1.0, 1.0, self.k, device=device))
        self.register_buffer("scales", torch.tensor([max(0.2, 2.0 / max(1, self.k - 1))], device=device))
        gen = torch.Generator(device=device).manual_seed(int(seed) + 30477)
        self.leg_readout = nn.Parameter(
            torch.randn(self.input_dim, self.output_dim, self.k, device=device, generator=gen)
            / math.sqrt(max(1, self.input_dim * self.k))
        )
        self.quad_proj = nn.Parameter(
            torch.randn(self.input_dim, self.hidden_dim, device=device, generator=gen)
            / math.sqrt(max(1, self.input_dim))
        )
        self.quad_readout = nn.Parameter(
            torch.randn(self.hidden_dim, 2, self.output_dim, device=device, generator=gen)
            / math.sqrt(max(1, self.hidden_dim * 2))
        )
        branch_init = [1.0, 0.35]
        if "quadboost" in str(spec.init_variant):
            branch_init = [0.75, 0.75]
        if "midboost" in str(spec.init_variant):
            branch_init = [1.0, 0.55]
        self.branch_scale = nn.Parameter(torch.tensor(branch_init, device=device))
        gain_init = 1.0
        if "temp050" in str(spec.init_variant):
            gain_init = 0.50
        elif "temp075" in str(spec.init_variant):
            gain_init = 0.75
        self.logit_gain = nn.Parameter(torch.tensor([gain_init], device=device))
        self.bias = nn.Parameter(torch.zeros(self.output_dim, device=device))
        self.register_buffer("quad_feature_std", torch.ones(self.hidden_dim, device=device))
        self._calibrate_quadratic_feature_norm(xs)

    @property
    def edge_param_count(self) -> int:
        return int(
            self.leg_readout.numel()
            + self.quad_proj.numel()
            + self.quad_readout.numel()
            + self.branch_scale.numel()
            + self.logit_gain.numel()
            + self.bias.numel()
        )

    def _rms_normalized_input(self, x: torch.Tensor) -> torch.Tensor:
        z = (x - self.mu) / self.std
        rms = z.float().square().mean(dim=1, keepdim=True).sqrt().clamp_min(1.0e-3)
        return z / rms * self.input_target_rms

    def _legendre_input(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self._rms_normalized_input(x))

    def _quadratic_input(self, x: torch.Tensor) -> torch.Tensor:
        return self._rms_normalized_input(x).clamp(-3.0, 3.0)

    def _calibrate_quadratic_feature_norm(self, xs: torch.Tensor) -> None:
        with torch.no_grad():
            sample = xs[: min(2048, int(xs.shape[0]))]
            z = self._quadratic_input(sample) @ self.quad_proj
            self.quad_feature_std.copy_(z.float().std(dim=0).clamp_min(1.0e-4))

    def _legendre_logits_and_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b = _basis_eval(self._legendre_input(x), "legendre", self.k, self.centers, self.scales)
        logits = torch.einsum("bdk,dck->bc", b, self.leg_readout) / math.sqrt(max(1, self.input_dim))
        return logits, b.reshape(int(x.shape[0]), -1)

    def _quadratic_logits_and_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = (self._quadratic_input(x) @ self.quad_proj) / self.quad_feature_std.clamp_min(1.0e-4)
        centered_square = z.square() - z.square().mean(dim=0, keepdim=True).detach()
        feats = torch.stack([z, centered_square], dim=2)
        logits = torch.einsum("bhk,hkc->bc", feats, self.quad_readout)
        return logits, feats.reshape(int(x.shape[0]), -1)

    def frozen_readout_features(self, x: torch.Tensor) -> torch.Tensor:
        _, leg_feats = self._legendre_logits_and_features(x)
        _, quad_feats = self._quadratic_logits_and_features(x)
        return torch.cat([leg_feats, quad_feats], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        leg_logits, _ = self._legendre_logits_and_features(x)
        quad_logits, _ = self._quadratic_logits_and_features(x)
        logits = self.branch_scale[0] * leg_logits + self.branch_scale[1] * quad_logits + self.bias
        return self.logit_gain.clamp(0.25, 4.0) * logits

    def basis_diagnostics(self, x: torch.Tensor) -> Dict[str, float]:
        with torch.no_grad():
            feats = self.frozen_readout_features(x).float()
            energy = feats.square().mean(dim=0)
            prob = energy / energy.sum().clamp_min(EPS)
            entropy = -(prob * (prob + EPS).log()).sum() / math.log(max(2, int(prob.numel())))
            dead = (energy < 1.0e-8).float().mean()
            centered = feats - feats.mean(dim=0, keepdim=True)
            try:
                s = torch.linalg.svdvals(centered[: min(256, int(centered.shape[0]))])
                rank = (s.square().sum().square() / s.pow(4).sum().clamp_min(EPS)).item()
                cond = (s.max() / s[s > 1.0e-7].min()).item() if bool((s > 1.0e-7).any()) else float("inf")
            except RuntimeError:
                rank = 0.0
                cond = float("inf")
            return {
                "basis_entropy": float(entropy.item()),
                "dead_basis_fraction": float(dead.item()),
                "basis_effective_rank": float(rank),
                "basis_condition_proxy": float(cond),
                "basis_output_norm_p95": float(torch.quantile(feats.norm(dim=1), 0.95).item()),
            }

    def basis_functional_direction(self, mode: str) -> List[torch.Tensor]:
        with torch.no_grad():
            leg = -self.leg_readout.detach().clone()
            quad_proj = torch.zeros_like(self.quad_proj)
            quad_readout = -self.quad_readout.detach().clone()
            if mode in {"basis_aware_snr_projected", "basis_aware_orthogonal"}:
                leg[..., 0] *= 0.25
                quad_readout[:, 0, :] *= 0.25
            return [
                leg,
                quad_proj,
                quad_readout,
                torch.zeros_like(self.branch_scale),
                torch.zeros_like(self.logit_gain),
                -self.bias.detach().clone(),
            ]


class SimpleFastTaskGeometryKAN(nn.Module):
    """Compact hinge/direct basis plus minimal quadratic interaction.

    This is the v12.5.2 R1/R6 fallback when full GatedLQ fusion remains too
    slow.  It keeps every learnable tensor in an edge-basis or interaction
    readout role, uses only fixed train-stream normalization, and avoids the
    two-layer Legendre path that dominated the B47 forward bottleneck.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        spec: PrimitiveSpec,
        x_for_stats: torch.Tensor,
        seed: int,
        device: torch.device,
        y_for_stats: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.spec = spec
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.hidden_dim = int(spec.hidden_dim)
        xs = x_for_stats[: min(4096, int(x_for_stats.shape[0]))].to(device=device, dtype=torch.float32)
        self.register_buffer("mu", xs.mean(dim=0))
        self.register_buffer("std", xs.std(dim=0).clamp_min(1.0e-3))
        variant = str(spec.init_variant)
        if "twohinge" in variant:
            hinge_centers = [0.25, 0.75]
        elif "hinge050" in variant:
            hinge_centers = [0.50]
        else:
            hinge_centers = [0.25]
        self.hinge_center0 = float(hinge_centers[0])
        self.register_buffer("hinge_centers", torch.tensor(hinge_centers, device=device))
        z_stats = ((xs - self.mu) / self.std).clamp(-3.0, 3.0)
        self.register_buffer("z2_mean", z_stats.square().mean(dim=0))
        self.register_buffer("z3_mean", z_stats.pow(3).mean(dim=0))
        self.register_buffer("zabs_mean", z_stats.abs().mean(dim=0))
        self.register_buffer("zsignedsq_mean", (z_stats * z_stats.abs()).mean(dim=0))
        self.register_buffer("zabs_scalar_mean", z_stats.abs().mean(dim=1, keepdim=True).mean(dim=0))
        self.register_buffer("zmean_scalar_mean", z_stats.mean(dim=1, keepdim=True).mean(dim=0))
        self.register_buffer("quad_feature_std", torch.ones(self.hidden_dim, device=device))
        gen = torch.Generator(device=device).manual_seed(int(seed) + 48127)
        variant_lower = variant.lower()
        variant_tokens = set(variant_lower.split("_"))
        probe_dirs: torch.Tensor | None = None
        if "trainprobe" in variant_lower and y_for_stats is not None:
            with torch.no_grad():
                y_stats = y_for_stats[: int(xs.shape[0])].to(device=device, dtype=torch.long)
                global_center = z_stats.mean(dim=0)
                dirs = []
                for class_idx in range(self.output_dim):
                    mask = y_stats == int(class_idx)
                    if bool(mask.any()):
                        vec = z_stats[mask].mean(dim=0) - global_center
                    else:
                        vec = torch.zeros(self.input_dim, device=device)
                    vec = vec - vec.mean()
                    dirs.append(vec / vec.norm().clamp_min(1.0e-6))
                stacked = torch.stack(dirs, dim=0)
                if bool(torch.isfinite(stacked).all()) and float(stacked.norm().item()) > 0.0:
                    probe_dirs = stacked
        self.register_buffer(
            "trainprobe_signal_init_uses_labels",
            torch.tensor([1 if "trainprobe" in variant_lower else 0], device=device, dtype=torch.int64),
        )
        self.register_buffer(
            "trainprobe_signal_init_applied",
            torch.tensor([1 if probe_dirs is not None else 0], device=device, dtype=torch.int64),
        )
        self.direct_absmix_square_enabled = "absmixsq" in variant
        self.direct_absmix_square_scale = 0.25
        if "absmixsq010" in variant:
            self.direct_absmix_square_scale = 0.10
        if "absmixsq050" in variant:
            self.direct_absmix_square_scale = 0.50
        self.direct_signed_square_enabled = "sgnsqdiag" in variant
        self.direct_abs_square_enabled = ("absquad" in variant) and not (self.direct_absmix_square_enabled or self.direct_signed_square_enabled)
        self.direct_cubic_enabled = ("cubicdiag" in variant) and not (self.direct_abs_square_enabled or self.direct_absmix_square_enabled or self.direct_signed_square_enabled)
        self.direct_square_enabled = ("sqdiag" in variant) and not (self.direct_abs_square_enabled or self.direct_absmix_square_enabled or self.direct_signed_square_enabled or self.direct_cubic_enabled)
        self.direct_abs_enabled = ("absdiag" in variant) and not (self.direct_abs_square_enabled or self.direct_absmix_square_enabled or self.direct_signed_square_enabled or self.direct_cubic_enabled)
        self.direct_normabs_enabled = "normabs" in variant
        self.direct_meanstat_enabled = "meanstat" in variant
        self.direct_groupabs_count = 8 if "groupabs8" in variant else (4 if "groupabs4" in variant else 0)
        if self.direct_groupabs_count:
            group_size = max(1, self.input_dim // self.direct_groupabs_count)
            group_means = []
            for group_idx in range(self.direct_groupabs_count):
                start = group_idx * group_size
                end = self.input_dim if group_idx == self.direct_groupabs_count - 1 else min(self.input_dim, (group_idx + 1) * group_size)
                group_means.append(z_stats[:, start:end].abs().mean(dim=1, keepdim=True).mean(dim=0))
            self.register_buffer("zgroupabs_mean", torch.cat(group_means, dim=0))
        else:
            self.register_buffer("zgroupabs_mean", torch.empty(0, device=device))
        self.quad_batch_rms_enabled = "rmsq" in variant
        self.quad_tanh_enabled = "boundq" in variant
        self.quad_pairscale_enabled = "pairscale" in variant_lower
        self.quad_sparsep_k = 0
        self.quad_sparsep_style = "random"
        for token, value in (("hybridk4p", 4), ("hybridk8p", 8), ("hybridk12p", 12), ("hybridk16p", 16)):
            if token in variant_lower:
                self.quad_sparsep_k = min(self.input_dim, value)
                self.quad_sparsep_style = "hybrid_local_global"
                break
        for token, value in (("blockk4p", 4), ("blockk8p", 8), ("blockk12p", 12), ("blockk16p", 16)):
            if self.quad_sparsep_k == 0 and token in variant_lower:
                self.quad_sparsep_k = min(self.input_dim, value)
                self.quad_sparsep_style = "block_local"
                break
        for token, value in (("sparsek4p", 4), ("sparsek8p", 8), ("sparsek12p", 12), ("sparsek16p", 16)):
            if self.quad_sparsep_k == 0 and token in variant_lower:
                self.quad_sparsep_k = min(self.input_dim, value)
                break
        self.quad_sparsep_enabled = self.quad_sparsep_k > 0
        active_hidden = self.hidden_dim
        for token, value in (("activep32", 32), ("activep48", 48), ("activep64", 64), ("activep80", 80), ("activep96", 96), ("activep128", 128)):
            if token in variant_lower:
                active_hidden = min(self.hidden_dim, value)
                break
        self.quad_proj_active_hidden = int(active_hidden)
        direct_feature_multiplier = 1 + 2 * int(self.hinge_centers.numel())
        if self.direct_abs_square_enabled:
            direct_feature_multiplier += 2
        elif self.direct_square_enabled or self.direct_abs_enabled or self.direct_absmix_square_enabled or self.direct_signed_square_enabled or self.direct_cubic_enabled:
            direct_feature_multiplier += 1
        direct_feature_count = direct_feature_multiplier * self.input_dim
        if self.direct_normabs_enabled:
            direct_feature_count += 1
        if self.direct_meanstat_enabled:
            direct_feature_count += 1
        direct_feature_count += int(self.direct_groupabs_count)
        scale_direct = math.sqrt(max(1, self.input_dim))
        direct_readout = torch.randn(direct_feature_count, self.output_dim, device=device, generator=gen) / scale_direct
        identity_init_scale = 1.0
        if "identityamp150" in variant:
            identity_init_scale = 1.50
        if identity_init_scale != 1.0:
            direct_readout[: self.input_dim, :].mul_(identity_init_scale)
        hinge_init_scale = 1.0
        if "hingeamp075" in variant:
            hinge_init_scale = 0.75
        if "hingeamp050" in variant:
            hinge_init_scale = 0.50
        if "hingeamp025" in variant:
            hinge_init_scale = 0.25
        if hinge_init_scale != 1.0:
            hinge_start = self.input_dim
            hinge_end = self.input_dim + 2 * int(self.hinge_centers.numel()) * self.input_dim
            direct_readout[hinge_start:hinge_end, :].mul_(hinge_init_scale)
        if self.direct_abs_square_enabled:
            absquad_abs_scale = 0.50
            absquad_square_scale = 0.25
            if "absquad050050" in variant:
                absquad_abs_scale = 0.50
                absquad_square_scale = 0.50
            if "absquad075025" in variant:
                absquad_abs_scale = 0.75
                absquad_square_scale = 0.25
            if "absquad050010" in variant:
                absquad_abs_scale = 0.50
                absquad_square_scale = 0.10
            diag_start = (1 + 2 * int(self.hinge_centers.numel())) * self.input_dim
            direct_readout[diag_start : diag_start + self.input_dim, :].mul_(absquad_abs_scale)
            direct_readout[diag_start + self.input_dim : diag_start + 2 * self.input_dim, :].mul_(absquad_square_scale)
        elif self.direct_absmix_square_enabled:
            absmix_init_scale = 0.50
            if "absmixsq025init025" in variant:
                absmix_init_scale = 0.25
            if "absmixsq025init075" in variant:
                absmix_init_scale = 0.75
            diag_start = (1 + 2 * int(self.hinge_centers.numel())) * self.input_dim
            direct_readout[diag_start : diag_start + self.input_dim, :].mul_(absmix_init_scale)
        elif self.direct_square_enabled or self.direct_abs_enabled or self.direct_signed_square_enabled or self.direct_cubic_enabled:
            square_init_scale = 0.25 if self.direct_abs_enabled else 1.0
            if "sqdiag025" in variant:
                square_init_scale = 0.25
            if "sqdiag010" in variant:
                square_init_scale = 0.10
            if "sgnsqdiag025" in variant:
                square_init_scale = 0.25
            if "sgnsqdiag010" in variant:
                square_init_scale = 0.10
            if "sgnsqdiag050" in variant:
                square_init_scale = 0.50
            if "cubicdiag025" in variant:
                square_init_scale = 0.25
            if "cubicdiag010" in variant:
                square_init_scale = 0.10
            if "cubicdiag050" in variant:
                square_init_scale = 0.50
            if "absdiag050" in variant:
                square_init_scale = 0.50
            if "absdiag075" in variant:
                square_init_scale = 0.75
            if "absdiag100" in variant:
                square_init_scale = 1.00
            if "absdiag025" in variant:
                square_init_scale = 0.25
            if "absdiag010" in variant:
                square_init_scale = 0.10
            diag_start = (1 + 2 * int(self.hinge_centers.numel())) * self.input_dim
            direct_readout[diag_start : diag_start + self.input_dim, :].mul_(square_init_scale)
        scalar_start = direct_feature_multiplier * self.input_dim
        scalar_row = scalar_start
        if self.direct_normabs_enabled:
            normabs_init_scale = 0.25
            if "normabs010" in variant:
                normabs_init_scale = 0.10
            if "normabs025" in variant:
                normabs_init_scale = 0.25
            if "normabs050" in variant:
                normabs_init_scale = 0.50
            if "normabs075" in variant:
                normabs_init_scale = 0.75
            if "normabs100" in variant:
                normabs_init_scale = 1.00
            direct_readout[scalar_row : scalar_row + 1, :].mul_(normabs_init_scale)
            scalar_row += 1
        if self.direct_meanstat_enabled:
            meanstat_init_scale = 0.25
            if "meanstat010" in variant:
                meanstat_init_scale = 0.10
            if "meanstat025" in variant:
                meanstat_init_scale = 0.25
            if "meanstat050" in variant:
                meanstat_init_scale = 0.50
            if "meanstat075" in variant:
                meanstat_init_scale = 0.75
            if "meanstat100" in variant:
                meanstat_init_scale = 1.00
            direct_readout[scalar_row : scalar_row + 1, :].mul_(meanstat_init_scale)
            scalar_row += 1
        if self.direct_groupabs_count:
            groupabs_init_scale = 0.25
            if "groupabs010" in variant:
                groupabs_init_scale = 0.10
            if "groupabs025" in variant:
                groupabs_init_scale = 0.25
            if "groupabs050" in variant:
                groupabs_init_scale = 0.50
            if "groupabs075" in variant:
                groupabs_init_scale = 0.75
            if "groupabs100" in variant:
                groupabs_init_scale = 1.00
            direct_readout[scalar_row : scalar_row + int(self.direct_groupabs_count), :].mul_(groupabs_init_scale)
        if probe_dirs is not None and "trainprobedirect" in variant_lower:
            probe_direct_scale = 0.25
            if "trainprobedirect010" in variant_lower:
                probe_direct_scale = 0.10
            if "trainprobedirect025" in variant_lower:
                probe_direct_scale = 0.25
            if "trainprobedirect050" in variant_lower:
                probe_direct_scale = 0.50
            if "trainprobedirect060" in variant_lower:
                probe_direct_scale = 0.60
            if "trainprobedirect065" in variant_lower:
                probe_direct_scale = 0.65
            if "trainprobedirect075" in variant_lower:
                probe_direct_scale = 0.75
            direct_readout[: self.input_dim, :].copy_(
                probe_dirs.transpose(0, 1).contiguous() * math.sqrt(max(1, self.input_dim)) * float(probe_direct_scale)
            )
        direct_readout_init_scale = 1.0
        if "readinit090" in variant_tokens or "directreadinit090" in variant_tokens:
            direct_readout_init_scale = 0.90
        if "readinit075" in variant_tokens or "directreadinit075" in variant_tokens:
            direct_readout_init_scale = 0.75
        if "readinit050" in variant_tokens or "directreadinit050" in variant_tokens:
            direct_readout_init_scale = 0.50
        if "readinit125" in variant_tokens or "directreadinit125" in variant_tokens:
            direct_readout_init_scale = 1.25
        if direct_readout_init_scale != 1.0:
            direct_readout.mul_(float(direct_readout_init_scale))
        self.direct_readout = nn.Parameter(direct_readout)
        quad_proj = torch.randn(self.input_dim, self.hidden_dim, device=device, generator=gen) / math.sqrt(max(1, self.input_dim))
        if probe_dirs is not None and "trainprobep" in variant_lower:
            with torch.no_grad():
                quad_proj.zero_()
                col = 0
                for class_idx in range(self.output_dim):
                    if col >= self.hidden_dim:
                        break
                    quad_proj[:, col] = probe_dirs[class_idx]
                    col += 1
                for class_a in range(self.output_dim):
                    for class_b in range(class_a + 1, self.output_dim):
                        if col >= self.hidden_dim:
                            break
                        vec = probe_dirs[class_a] - probe_dirs[class_b]
                        vec = vec - vec.mean()
                        quad_proj[:, col] = vec / vec.norm().clamp_min(1.0e-6)
                        col += 1
                    if col >= self.hidden_dim:
                        break
                base_cols = col
                trainprobe_broad_mix = 0.0
                if "signalbroad010" in variant_lower:
                    trainprobe_broad_mix = 0.10
                if "signalbroad015" in variant_lower:
                    trainprobe_broad_mix = 0.15
                if "signalbroad025" in variant_lower:
                    trainprobe_broad_mix = 0.25
                if "signalbroad030" in variant_lower:
                    trainprobe_broad_mix = 0.30
                if "signalbroad032" in variant_lower:
                    trainprobe_broad_mix = 0.32
                if "signalbroad035" in variant_lower:
                    trainprobe_broad_mix = 0.35
                if "signalbroad040" in variant_lower:
                    trainprobe_broad_mix = 0.40
                if "signalbroad050" in variant_lower:
                    trainprobe_broad_mix = 0.50
                trainprobe_signal_block_frac = 0.0
                if "signalblock025" in variant_lower:
                    trainprobe_signal_block_frac = 0.25
                if "signalblock010" in variant_lower:
                    trainprobe_signal_block_frac = 0.10
                if "signalblock015" in variant_lower:
                    trainprobe_signal_block_frac = 0.15
                if "signalblock020" in variant_lower:
                    trainprobe_signal_block_frac = 0.20
                if "signalblock050" in variant_lower:
                    trainprobe_signal_block_frac = 0.50
                while col < self.hidden_dim:
                    vec = torch.randn(self.input_dim, device=device, generator=gen)
                    if base_cols > 0:
                        comps = quad_proj[:, :base_cols]
                        residual = vec - comps @ torch.linalg.lstsq(comps.float(), vec.float()).solution.to(device=device, dtype=vec.dtype)
                        if trainprobe_signal_block_frac > 0.0 and self.input_dim > 1:
                            block_width = max(1, int(round(float(self.input_dim) * float(trainprobe_signal_block_frac))))
                            block_width = min(int(self.input_dim), block_width)
                            start = int((col * max(1, block_width // 2) + 17 * col) % int(self.input_dim))
                            idx = (torch.arange(block_width, device=device) + start) % int(self.input_dim)
                            masked = torch.zeros_like(residual)
                            masked.index_copy_(0, idx, residual.index_select(0, idx))
                            if masked.norm() > 1.0e-6:
                                residual = masked
                        if trainprobe_broad_mix > 0.0:
                            signal_vec = quad_proj[:, col % base_cols]
                            vec = residual.mul(1.0 - trainprobe_broad_mix).add(signal_vec, alpha=trainprobe_broad_mix)
                        else:
                            vec = residual
                    vec = vec - vec.mean()
                    quad_proj[:, col] = vec / vec.norm().clamp_min(1.0e-6)
                    col += 1
        if "pcap" in variant or "pcabandp" in variant:
            with torch.no_grad():
                z_centered = z_stats - z_stats.mean(dim=0, keepdim=True)
                try:
                    _u_pca, _s_pca, vh_pca = torch.linalg.svd(z_centered, full_matrices=False)
                    n_comp = min(int(vh_pca.shape[0]), self.hidden_dim)
                    if n_comp > 0:
                        if "pcabandp" in variant:
                            comp_idx = torch.linspace(0, int(vh_pca.shape[0]) - 1, n_comp, device=device).round().to(dtype=torch.long)
                        else:
                            comp_idx = torch.arange(n_comp, device=device, dtype=torch.long)
                        signs = torch.where(
                            torch.rand(n_comp, device=device, generator=gen) > 0.5,
                            torch.ones(n_comp, device=device),
                            -torch.ones(n_comp, device=device),
                        )
                        quad_proj.zero_()
                        quad_proj[:, :n_comp] = vh_pca.index_select(0, comp_idx).transpose(0, 1).contiguous() * signs.view(1, -1)
                    col = n_comp
                    while col < self.hidden_dim:
                        vec = torch.randn(self.input_dim, device=device, generator=gen)
                        if n_comp > 0:
                            comps = quad_proj[:, :n_comp]
                            vec = vec - comps @ (comps.transpose(0, 1) @ vec)
                        vec = vec - vec.mean()
                        quad_proj[:, col] = vec / vec.norm().clamp_min(1.0e-6)
                        col += 1
                except RuntimeError:
                    pass
        if "orthop" in variant:
            if self.input_dim >= self.hidden_dim:
                frame = torch.randn(self.input_dim, self.hidden_dim, device=device, generator=gen)
                q_frame, r_frame = torch.linalg.qr(frame, mode="reduced")
                signs = torch.sign(torch.diag(r_frame))
                signs = torch.where(signs == 0, torch.ones_like(signs), signs)
                quad_proj = (q_frame * signs.view(1, -1)).contiguous()
            else:
                frame = torch.randn(self.hidden_dim, self.input_dim, device=device, generator=gen)
                q_frame, r_frame = torch.linalg.qr(frame, mode="reduced")
                signs = torch.sign(torch.diag(r_frame))
                signs = torch.where(signs == 0, torch.ones_like(signs), signs)
                quad_proj = (q_frame * signs.view(1, -1)).transpose(0, 1).contiguous()
        if "lowfreqp" in variant:
            quad_proj.zero_()
            col = 0
            side = int(round(math.sqrt(float(self.input_dim))))
            if side * side == self.input_dim:
                yy = torch.linspace(0.0, 1.0, side, device=device)
                xx = torch.linspace(0.0, 1.0, side, device=device)
                gy, gx = torch.meshgrid(yy, xx, indexing="ij")
                freq_pairs = [(0, 0)]
                for freq_sum in range(1, 32):
                    for fy in range(freq_sum + 1):
                        fx = freq_sum - fy
                        freq_pairs.append((fy, fx))
                for fy, fx in freq_pairs:
                    if col >= self.hidden_dim:
                        break
                    basis = torch.cos(math.pi * float(fy) * (gy + 0.5 / max(1, side))) * torch.cos(math.pi * float(fx) * (gx + 0.5 / max(1, side)))
                    vec = basis.reshape(-1)
                    vec = vec - vec.mean()
                    quad_proj[:, col] = vec / vec.norm().clamp_min(1.0e-6)
                    col += 1
            grid = torch.linspace(0.0, 1.0, self.input_dim, device=device)
            freq = 1
            while col < self.hidden_dim:
                vec = torch.sin(math.pi * float(freq) * grid) if col % 2 == 0 else torch.cos(math.pi * float(freq) * grid)
                vec = vec - vec.mean()
                quad_proj[:, col] = vec / vec.norm().clamp_min(1.0e-6)
                col += 1
                if col % 2 == 0:
                    freq += 1
        elif "blockfreqp" in variant:
            quad_proj.zero_()
            col = 0
            side = int(round(math.sqrt(float(self.input_dim))))
            if side * side == self.input_dim:
                tile_count = 4
                tile = max(1, side // tile_count)
                local_modes = [(0, 0), (1, 0), (0, 1), (1, 1), (2, 0), (0, 2), (2, 1), (1, 2), (2, 2), (3, 0)]
                for ty in range(tile_count):
                    for tx in range(tile_count):
                        y0 = ty * tile
                        y1 = side if ty == tile_count - 1 else min(side, (ty + 1) * tile)
                        x0 = tx * tile
                        x1 = side if tx == tile_count - 1 else min(side, (tx + 1) * tile)
                        yy = torch.linspace(0.0, 1.0, max(1, y1 - y0), device=device)
                        xx = torch.linspace(0.0, 1.0, max(1, x1 - x0), device=device)
                        gy, gx = torch.meshgrid(yy, xx, indexing="ij")
                        for fy, fx in local_modes:
                            if col >= self.hidden_dim:
                                break
                            patch = torch.cos(math.pi * float(fy) * (gy + 0.5 / max(1, y1 - y0))) * torch.cos(math.pi * float(fx) * (gx + 0.5 / max(1, x1 - x0)))
                            basis = torch.zeros(side, side, device=device)
                            basis[y0:y1, x0:x1] = patch
                            vec = basis.reshape(-1)
                            vec = vec - vec.mean()
                            quad_proj[:, col] = vec / vec.norm().clamp_min(1.0e-6)
                            col += 1
                        if col >= self.hidden_dim:
                            break
                    if col >= self.hidden_dim:
                        break
            block_count = max(8, min(32, int(math.sqrt(float(max(1, self.input_dim))))))
            while col < self.hidden_dim:
                block_idx = col % block_count
                start = (block_idx * self.input_dim) // block_count
                end = ((block_idx + 1) * self.input_dim) // block_count
                width = max(1, end - start)
                local = torch.linspace(0.0, 1.0, width, device=device)
                mode = (col // block_count) % 4
                if mode == 0:
                    patch = torch.ones(width, device=device)
                elif mode == 1:
                    patch = torch.cos(math.pi * (local + 0.5 / width))
                elif mode == 2:
                    patch = torch.sin(math.pi * (local + 0.5 / width))
                else:
                    patch = torch.cos(2.0 * math.pi * (local + 0.5 / width))
                vec = torch.zeros(self.input_dim, device=device)
                vec[start:end] = patch
                vec = vec - vec.mean()
                quad_proj[:, col] = vec / vec.norm().clamp_min(1.0e-6)
                col += 1
        if "localdensepairtraj" in variant:
            quad_proj.zero_()
            norm = 1.0 / math.sqrt(2.0)
            col = 0
            local_pairs = [(0, 1), (2, 3), (0, 2), (1, 3)]
            for a, b in local_pairs:
                if col >= self.hidden_dim or self.input_dim <= max(a, b):
                    break
                quad_proj[a, col] = norm
                quad_proj[b, col] = norm
                col += 1
                if col >= self.hidden_dim:
                    break
                quad_proj[a, col] = norm
                quad_proj[b, col] = -norm
                col += 1
            while col < self.hidden_dim:
                u = torch.randn(self.input_dim, device=device, generator=gen)
                v = torch.randn(self.input_dim, device=device, generator=gen)
                u = u / u.norm().clamp_min(1.0e-6)
                v = v - (v * u).sum() * u
                v = v / v.norm().clamp_min(1.0e-6)
                plus = (u + v) * norm
                minus = (u - v) * norm
                quad_proj[:, col] = plus / plus.norm().clamp_min(1.0e-6)
                col += 1
                if col >= self.hidden_dim:
                    break
                quad_proj[:, col] = minus / minus.norm().clamp_min(1.0e-6)
                col += 1
        elif "pairtraj" in variant:
            quad_proj.zero_()
            norm = 1.0 / math.sqrt(2.0)
            col = 0
            pair_idx = 0
            while col < self.hidden_dim:
                a = (11 + 37 * pair_idx) % self.input_dim
                b = (53 + 97 * pair_idx) % self.input_dim
                if b == a:
                    b = (b + 1) % self.input_dim
                quad_proj[a, col] = norm
                quad_proj[b, col] = norm
                col += 1
                if col >= self.hidden_dim:
                    break
                quad_proj[a, col] = norm
                quad_proj[b, col] = -norm
                col += 1
                pair_idx += 1
        elif "signedpairtail" in variant:
            quad_proj.zero_()
            norm = 1.0 / math.sqrt(2.0)
            pair_cols = max(2, self.hidden_dim // 2)
            col = 0
            for start in range(0, max(1, self.input_dim - 1), 2):
                if col >= min(pair_cols, self.hidden_dim):
                    break
                a = start % self.input_dim
                b = (start + 1) % self.input_dim
                quad_proj[a, col] = norm
                quad_proj[b, col] = norm
                col += 1
                if col >= min(pair_cols, self.hidden_dim):
                    break
                quad_proj[a, col] = norm
                quad_proj[b, col] = -norm
                col += 1
            while col < self.hidden_dim:
                vec = torch.randn(self.input_dim, device=device, generator=gen)
                vec = vec / vec.norm().clamp_min(1.0e-6)
                quad_proj[:, col] = vec
                col += 1
        elif "signedpairlite" in variant:
            quad_proj.zero_()
            norm = 1.0 / math.sqrt(2.0)
            col = 0
            for start in range(0, max(1, self.input_dim - 1), 2):
                if col >= self.hidden_dim:
                    break
                a = start % self.input_dim
                b = (start + 1) % self.input_dim
                quad_proj[a, col] = norm
                quad_proj[b, col] = norm
                col += 1
                if col >= self.hidden_dim:
                    break
                quad_proj[a, col] = norm
                quad_proj[b, col] = -norm
                col += 1
            while col < self.hidden_dim:
                idx = col % self.input_dim
                quad_proj[idx, col] = 1.0
                col += 1
        if self.quad_pairscale_enabled:
            pair_a = torch.empty(self.hidden_dim, device=device, dtype=torch.long)
            pair_b = torch.empty(self.hidden_dim, device=device, dtype=torch.long)
            pair_sign_b = torch.empty(self.hidden_dim, device=device)
            norm = 1.0 / math.sqrt(2.0)
            pair_idx = 0
            quad_proj.zero_()
            for col in range(self.hidden_dim):
                a = (11 + 37 * pair_idx) % self.input_dim
                b = (53 + 97 * pair_idx) % self.input_dim
                if b == a:
                    b = (b + 1) % self.input_dim
                sign = 1.0 if col % 2 == 0 else -1.0
                pair_a[col] = int(a)
                pair_b[col] = int(b)
                pair_sign_b[col] = float(sign)
                quad_proj[a, col] = norm
                quad_proj[b, col] = sign * norm
                if col % 2 == 1:
                    pair_idx += 1
            self.register_buffer("quad_pair_a", pair_a)
            self.register_buffer("quad_pair_b", pair_b)
            self.register_buffer("quad_pair_sign_b", pair_sign_b)
            self.register_buffer("quad_pair_norm", torch.tensor([norm], device=device))
            scale_init = torch.ones(self.hidden_dim, device=device)
            if "pairscale075" in variant.lower():
                scale_init.fill_(0.75)
            if "pairscale050" in variant.lower():
                scale_init.fill_(0.50)
            self.quad_proj_scale = nn.Parameter(scale_init)
            self.register_buffer("quad_proj", quad_proj)
            self.register_buffer("quad_proj_grad_mask", torch.empty(0, device=device))
        elif self.quad_sparsep_enabled:
            sparse_k = int(self.quad_sparsep_k)
            sparse_idx = torch.empty((self.hidden_dim, sparse_k), device=device, dtype=torch.long)
            sparse_weight = torch.empty((self.hidden_dim, sparse_k), device=device)
            quad_proj.zero_()
            side = int(round(math.sqrt(float(self.input_dim))))
            is_square_image = side * side == self.input_dim
            if self.quad_sparsep_style == "hybrid_local_global" and is_square_image:
                local_k = max(1, sparse_k // 2)
                global_k = max(0, sparse_k - local_k)
                if local_k <= 4:
                    patch_h, patch_w = 2, 2
                elif local_k <= 8:
                    patch_h, patch_w = 2, 4
                else:
                    patch_h, patch_w = 3, 4
                tile_rows = max(1, int(math.ceil(float(side) / float(patch_h))))
                tile_cols = max(1, int(math.ceil(float(side) / float(patch_w))))
                total_tiles = max(1, tile_rows * tile_cols)
                for col in range(self.hidden_dim):
                    used = set()
                    coords = []
                    vals = []
                    tile = col % total_tiles
                    mode = (col // total_tiles) % 4
                    ty = tile // tile_cols
                    tx = tile % tile_cols
                    y0 = min(max(0, side - patch_h), ty * patch_h)
                    x0 = min(max(0, side - patch_w), tx * patch_w)
                    for yy in range(patch_h):
                        for xx in range(patch_w):
                            if len(coords) >= local_k:
                                break
                            idx = int((y0 + yy) * side + (x0 + xx))
                            used.add(idx)
                            coords.append(idx)
                            if mode == 0:
                                vals.append(1.0)
                            elif mode == 1:
                                vals.append(1.0 if xx % 2 == 0 else -1.0)
                            elif mode == 2:
                                vals.append(1.0 if yy % 2 == 0 else -1.0)
                            else:
                                vals.append(1.0 if (yy + xx) % 2 == 0 else -1.0)
                        if len(coords) >= local_k:
                            break
                    for kk in range(global_k):
                        idx = (11 + 37 * col + 97 * kk + 13 * kk * kk) % self.input_dim
                        while int(idx) in used:
                            idx = (idx + 17) % self.input_dim
                        used.add(int(idx))
                        coords.append(int(idx))
                        vals.append(1.0 if kk % 2 == 0 else -1.0)
                    while len(coords) < sparse_k:
                        idx = (coords[-1] + 1) % self.input_dim if coords else 0
                        while idx in used:
                            idx = (idx + 1) % self.input_dim
                        used.add(int(idx))
                        coords.append(int(idx))
                        vals.append(1.0 if len(coords) % 2 == 0 else -1.0)
                    vals_t = torch.tensor(vals, device=device, dtype=torch.float32)
                    vals_t = vals_t / vals_t.norm().clamp_min(1.0e-6)
                    for kk, idx in enumerate(coords[:sparse_k]):
                        sparse_idx[col, kk] = int(idx)
                        sparse_weight[col, kk] = vals_t[kk]
                        quad_proj[int(idx), col] = vals_t[kk]
            elif self.quad_sparsep_style == "hybrid_local_global":
                local_k = max(1, sparse_k // 2)
                global_k = max(0, sparse_k - local_k)
                block_count = max(1, self.input_dim // max(1, local_k))
                for col in range(self.hidden_dim):
                    used = set()
                    coords = []
                    vals = []
                    start = ((col % block_count) * local_k) % self.input_dim
                    for kk in range(local_k):
                        idx = (start + kk) % self.input_dim
                        used.add(int(idx))
                        coords.append(int(idx))
                        vals.append(1.0 if kk % 2 == 0 else -1.0)
                    for kk in range(global_k):
                        idx = (11 + 37 * col + 97 * kk + 13 * kk * kk) % self.input_dim
                        while int(idx) in used:
                            idx = (idx + 17) % self.input_dim
                        used.add(int(idx))
                        coords.append(int(idx))
                        vals.append(1.0 if kk % 2 == 0 else -1.0)
                    vals_t = torch.tensor(vals, device=device, dtype=torch.float32)
                    vals_t = vals_t / vals_t.norm().clamp_min(1.0e-6)
                    for kk, idx in enumerate(coords[:sparse_k]):
                        sparse_idx[col, kk] = int(idx)
                        sparse_weight[col, kk] = vals_t[kk]
                        quad_proj[int(idx), col] = vals_t[kk]
            elif self.quad_sparsep_style == "block_local" and is_square_image:
                if sparse_k <= 4:
                    patch_h, patch_w = 2, 2
                elif sparse_k <= 8:
                    patch_h, patch_w = 2, 4
                elif sparse_k <= 12:
                    patch_h, patch_w = 3, 4
                else:
                    patch_h, patch_w = 4, 4
                tile_rows = max(1, int(math.ceil(float(side) / float(patch_h))))
                tile_cols = max(1, int(math.ceil(float(side) / float(patch_w))))
                total_tiles = max(1, tile_rows * tile_cols)
                for col in range(self.hidden_dim):
                    tile = col % total_tiles
                    mode = (col // total_tiles) % 4
                    ty = tile // tile_cols
                    tx = tile % tile_cols
                    y0 = min(max(0, side - patch_h), ty * patch_h)
                    x0 = min(max(0, side - patch_w), tx * patch_w)
                    coords = []
                    vals = []
                    for yy in range(patch_h):
                        for xx in range(patch_w):
                            if len(coords) >= sparse_k:
                                break
                            y = y0 + yy
                            x = x0 + xx
                            coords.append(int(y * side + x))
                            if mode == 0:
                                vals.append(1.0)
                            elif mode == 1:
                                vals.append(1.0 if xx % 2 == 0 else -1.0)
                            elif mode == 2:
                                vals.append(1.0 if yy % 2 == 0 else -1.0)
                            else:
                                vals.append(1.0 if (yy + xx) % 2 == 0 else -1.0)
                        if len(coords) >= sparse_k:
                            break
                    while len(coords) < sparse_k:
                        idx = (coords[-1] + 1) % self.input_dim if coords else 0
                        while idx in coords:
                            idx = (idx + 1) % self.input_dim
                        coords.append(int(idx))
                        vals.append(1.0 if len(coords) % 2 == 0 else -1.0)
                    vals_t = torch.tensor(vals, device=device, dtype=torch.float32)
                    vals_t = vals_t / vals_t.norm().clamp_min(1.0e-6)
                    for kk, idx in enumerate(coords[:sparse_k]):
                        sparse_idx[col, kk] = int(idx)
                        sparse_weight[col, kk] = vals_t[kk]
                        quad_proj[int(idx), col] = vals_t[kk]
            elif self.quad_sparsep_style == "block_local":
                block_count = max(1, self.input_dim // max(1, sparse_k))
                for col in range(self.hidden_dim):
                    start = ((col % block_count) * sparse_k) % self.input_dim
                    vals = torch.where(torch.arange(sparse_k, device=device) % 2 == 0, 1.0, -1.0)
                    if (col // max(1, block_count)) % 2 == 0:
                        vals = torch.ones_like(vals)
                    vals = vals / vals.norm().clamp_min(1.0e-6)
                    for kk in range(sparse_k):
                        idx = (start + kk) % self.input_dim
                        sparse_idx[col, kk] = int(idx)
                        sparse_weight[col, kk] = vals[kk]
                        quad_proj[int(idx), col] = vals[kk]
            else:
                for col in range(self.hidden_dim):
                    used = set()
                    vals = torch.randn(sparse_k, device=device, generator=gen)
                    signs = torch.where(torch.arange(sparse_k, device=device) % 2 == 0, 1.0, -1.0)
                    vals = vals.abs() * signs
                    vals = vals / vals.norm().clamp_min(1.0e-6)
                    for kk in range(sparse_k):
                        idx = (11 + 37 * col + 97 * kk + 13 * kk * kk) % self.input_dim
                        while int(idx) in used:
                            idx = (idx + 1) % self.input_dim
                        used.add(int(idx))
                        sparse_idx[col, kk] = int(idx)
                        sparse_weight[col, kk] = vals[kk]
                        quad_proj[int(idx), col] = vals[kk]
            if "sparsep050" in variant_lower:
                sparse_weight.mul_(0.50)
                quad_proj.mul_(0.50)
            if "sparsep075" in variant_lower:
                sparse_weight.mul_(0.75)
                quad_proj.mul_(0.75)
            if "sparsep125" in variant_lower:
                sparse_weight.mul_(1.25)
                quad_proj.mul_(1.25)
            self.register_buffer("quad_sparse_idx", sparse_idx)
            self.quad_sparse_weight = nn.Parameter(sparse_weight)
            self.register_buffer("quad_proj", quad_proj)
            self.register_buffer("quad_proj_grad_mask", torch.empty(0, device=device))
        elif "fixedp" in variant:
            self.register_buffer("quad_proj", quad_proj)
        else:
            self.quad_proj = nn.Parameter(quad_proj)
            if self.quad_proj_active_hidden < self.hidden_dim:
                grad_mask = torch.zeros_like(quad_proj)
                grad_mask[:, : self.quad_proj_active_hidden] = 1.0
                self.register_buffer("quad_proj_grad_mask", grad_mask)
                self.quad_proj.register_hook(lambda grad: grad * self.quad_proj_grad_mask)
            else:
                self.register_buffer("quad_proj_grad_mask", torch.empty(0, device=device))
        quad_readout = torch.randn(2 * self.hidden_dim, self.output_dim, device=device, generator=gen) / math.sqrt(max(1, self.hidden_dim * 2))
        quad_readout_init_scale = 1.0
        if "readinit090" in variant_tokens or "quadreadinit090" in variant_tokens:
            quad_readout_init_scale = 0.90
        if "readinit075" in variant_tokens or "quadreadinit075" in variant_tokens:
            quad_readout_init_scale = 0.75
        if "readinit050" in variant_tokens or "quadreadinit050" in variant_tokens:
            quad_readout_init_scale = 0.50
        if "readinit125" in variant_tokens or "quadreadinit125" in variant_tokens:
            quad_readout_init_scale = 1.25
        if quad_readout_init_scale != 1.0:
            quad_readout.mul_(float(quad_readout_init_scale))
        self.quad_readout = nn.Parameter(quad_readout)
        branch_init = [1.0, 0.30]
        if "quad050" in variant:
            branch_init = [1.0, 0.50]
        if "quad040" in variant:
            branch_init = [1.0, 0.40]
        if "quad020" in variant:
            branch_init = [1.0, 0.20]
        if "quad010" in variant:
            branch_init = [1.0, 0.10]
        if "identitytailquad060" in variant:
            branch_init = [1.35, 0.60]
        elif "identitytail130quad040" in variant:
            branch_init = [1.30, 0.40]
        elif "identitytail125quad040" in variant:
            branch_init = [1.25, 0.40]
        elif "identitytail145quad040" in variant:
            branch_init = [1.45, 0.40]
        elif "identitytail140quad040" in variant:
            branch_init = [1.40, 0.40]
        elif "identitytail150quad040" in variant:
            branch_init = [1.50, 0.40]
        elif "identitytailquad040" in variant:
            branch_init = [1.35, 0.40]
        elif "identitytailquad030" in variant:
            branch_init = [1.35, 0.30]
        elif "identitytailquad020" in variant:
            branch_init = [1.35, 0.20]
        elif "identitytailquad050" in variant:
            branch_init = [1.35, 0.50]
        elif "identitytail" in variant:
            branch_init = [1.35, 0.10]
        self.class_branch_scale_enabled = "classbranch" in variant
        if self.class_branch_scale_enabled:
            branch_tensor = torch.stack(
                [
                    torch.full((self.output_dim,), float(branch_init[0]), device=device),
                    torch.full((self.output_dim,), float(branch_init[1]), device=device),
                ]
            )
        else:
            branch_tensor = torch.tensor(branch_init, device=device)
        if "fixedbranch" in variant:
            self.register_buffer("branch_scale", branch_tensor)
        else:
            self.branch_scale = nn.Parameter(branch_tensor)
        gain_init = 0.75
        if "temp050" in variant:
            gain_init = 0.50
        if "temp065" in variant:
            gain_init = 0.65
        if "temp080" in variant:
            gain_init = 0.80
        if "temp085" in variant:
            gain_init = 0.85
        if "temp090" in variant:
            gain_init = 0.90
        if "temp092" in variant:
            gain_init = 0.92
        if "temp093" in variant:
            gain_init = 0.93
        if "temp095" in variant:
            gain_init = 0.95
        if "temp100" in variant:
            gain_init = 1.00
        if "temp105" in variant:
            gain_init = 1.05
        if "temp107" in variant:
            gain_init = 1.07
        if "temp110" in variant:
            gain_init = 1.10
        if "temp115" in variant:
            gain_init = 1.15
        if "temp125" in variant:
            gain_init = 1.25
        self.class_logit_gain_enabled = "classgain" in variant
        self.logit_batch_norm_stopgrad_enabled = "logitbnsg" in variant_lower
        self.logit_norm_enabled = ("logitnorm" in variant) or self.logit_batch_norm_stopgrad_enabled
        self.logit_norm_stopgrad_enabled = "logitnormsg" in variant_lower
        norm_target = 1.50
        if "logitnorm100" in variant or "logitnormsg100" in variant_lower or "logitbnsg100" in variant_lower:
            norm_target = 1.00
        if "logitnorm150" in variant or "logitnormsg150" in variant_lower or "logitbnsg150" in variant_lower:
            norm_target = 1.50
        if "logitnorm200" in variant or "logitnormsg200" in variant_lower or "logitbnsg200" in variant_lower:
            norm_target = 2.00
        self.register_buffer("logit_norm_target", torch.tensor([norm_target], device=device))
        self.logit_softcap_enabled = "logitcap" in variant_lower
        softcap_value = 3.00
        if "logitcap200" in variant_lower:
            softcap_value = 2.00
        if "logitcap250" in variant_lower:
            softcap_value = 2.50
        if "logitcap300" in variant_lower:
            softcap_value = 3.00
        if "logitcap350" in variant_lower:
            softcap_value = 3.50
        if "logitcap400" in variant_lower:
            softcap_value = 4.00
        if "logitcap500" in variant_lower:
            softcap_value = 5.00
        self.register_buffer("logit_softcap_value", torch.tensor([softcap_value], device=device))
        if self.class_logit_gain_enabled:
            gain_tensor = torch.full((self.output_dim,), float(gain_init), device=device)
        else:
            gain_tensor = torch.tensor([gain_init], device=device)
        if "fixedgain" in variant:
            self.register_buffer("logit_gain", gain_tensor)
        else:
            self.logit_gain = nn.Parameter(gain_tensor)
        self.bias = nn.Parameter(torch.zeros(self.output_dim, device=device))
        self._calibrate_quadratic_feature_norm(xs)

    @property
    def edge_param_count(self) -> int:
        return int(
            self.direct_readout.numel()
            + (self.quad_proj.numel() if isinstance(self.quad_proj, nn.Parameter) and self.quad_proj.requires_grad else 0)
            + (self.quad_proj_scale.numel() if hasattr(self, "quad_proj_scale") and isinstance(self.quad_proj_scale, nn.Parameter) and self.quad_proj_scale.requires_grad else 0)
            + (self.quad_sparse_weight.numel() if hasattr(self, "quad_sparse_weight") and isinstance(self.quad_sparse_weight, nn.Parameter) and self.quad_sparse_weight.requires_grad else 0)
            + self.quad_readout.numel()
            + (self.branch_scale.numel() if isinstance(self.branch_scale, nn.Parameter) and self.branch_scale.requires_grad else 0)
            + (self.logit_gain.numel() if isinstance(self.logit_gain, nn.Parameter) and self.logit_gain.requires_grad else 0)
            + self.bias.numel()
        )

    def _input(self, x: torch.Tensor) -> torch.Tensor:
        return ((x - self.mu) / self.std).clamp(-3.0, 3.0)

    def _hinge_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self._input(x)
        centers = self.hinge_centers.view(1, 1, -1)
        zu = z.unsqueeze(-1)
        pos = F.relu(zu - centers).flatten(1)
        neg = F.relu(-zu - centers).flatten(1)
        return z, pos, neg

    def _calibrate_quadratic_feature_norm(self, xs: torch.Tensor) -> None:
        with torch.no_grad():
            z = self._quadratic_projection(self._input(xs[: min(2048, int(xs.shape[0]))]))
            self.quad_feature_std.copy_(z.float().std(dim=0).clamp_min(1.0e-4))

    def _quadratic_projection(self, z: torch.Tensor) -> torch.Tensor:
        if bool(getattr(self, "quad_pairscale_enabled", False)):
            qa = z.index_select(1, self.quad_pair_a)
            qb = z.index_select(1, self.quad_pair_b)
            base = (qa + qb * self.quad_pair_sign_b.view(1, -1)) * self.quad_pair_norm.view(1, -1)
            return base * self.quad_proj_scale.view(1, -1)
        if bool(getattr(self, "quad_sparsep_enabled", False)):
            idx = self.quad_sparse_idx.reshape(-1)
            gathered = z.index_select(1, idx).view(int(z.shape[0]), self.hidden_dim, int(self.quad_sparsep_k))
            return (gathered * self.quad_sparse_weight.view(1, self.hidden_dim, int(self.quad_sparsep_k))).sum(dim=2)
        return z @ self.quad_proj

    def _direct_logits_and_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z, pos, neg = self._hinge_features(x)
        parts = [z, pos, neg]
        if self.direct_abs_square_enabled:
            parts.append(z.abs() - self.zabs_mean)
            parts.append(z.square() - self.z2_mean)
        elif self.direct_absmix_square_enabled:
            parts.append((z.abs() - self.zabs_mean) + float(self.direct_absmix_square_scale) * (z.square() - self.z2_mean))
        elif self.direct_square_enabled:
            parts.append(z.square() - self.z2_mean)
        elif self.direct_signed_square_enabled:
            parts.append(z * z.abs() - self.zsignedsq_mean)
        elif self.direct_cubic_enabled:
            parts.append(z.pow(3) - self.z3_mean)
        elif self.direct_abs_enabled:
            parts.append(z.abs() - self.zabs_mean)
        if self.direct_normabs_enabled:
            normabs = (z.abs().mean(dim=1, keepdim=True) - self.zabs_scalar_mean) * math.sqrt(max(1, self.input_dim))
            parts.append(normabs)
        if self.direct_meanstat_enabled:
            meanstat = (z.mean(dim=1, keepdim=True) - self.zmean_scalar_mean) * math.sqrt(max(1, self.input_dim))
            parts.append(meanstat)
        if self.direct_groupabs_count:
            group_size = max(1, self.input_dim // int(self.direct_groupabs_count))
            group_feats = []
            for group_idx in range(int(self.direct_groupabs_count)):
                start = group_idx * group_size
                end = self.input_dim if group_idx == int(self.direct_groupabs_count) - 1 else min(self.input_dim, (group_idx + 1) * group_size)
                group_abs = (z[:, start:end].abs().mean(dim=1, keepdim=True) - self.zgroupabs_mean[group_idx : group_idx + 1]) * math.sqrt(max(1, end - start))
                group_feats.append(group_abs)
            parts.append(torch.cat(group_feats, dim=1))
        feats = torch.cat(parts, dim=1)
        logits = feats @ self.direct_readout / math.sqrt(max(1, self.input_dim))
        return logits, feats

    def _quadratic_logits_and_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        q = self._quadratic_projection(self._input(x)) / self.quad_feature_std.clamp_min(1.0e-4)
        if self.quad_batch_rms_enabled:
            q = q / q.square().mean(dim=0, keepdim=True).sqrt().detach().clamp_min(1.0e-4)
        if self.quad_tanh_enabled:
            q = torch.tanh(q)
        q2 = q.square() - q.square().mean(dim=0, keepdim=True).detach()
        feats = torch.cat([q, q2], dim=1)
        logits = feats @ self.quad_readout
        return logits, feats

    def frozen_readout_features(self, x: torch.Tensor) -> torch.Tensor:
        _, direct = self._direct_logits_and_features(x)
        _, quad = self._quadratic_logits_and_features(x)
        return torch.cat([direct, quad], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        direct_logits, _ = self._direct_logits_and_features(x)
        quad_logits, _ = self._quadratic_logits_and_features(x)
        if self.class_branch_scale_enabled:
            logits = self.branch_scale[0].view(1, -1) * direct_logits + self.branch_scale[1].view(1, -1) * quad_logits + self.bias
        else:
            logits = self.branch_scale[0] * direct_logits + self.branch_scale[1] * quad_logits + self.bias
        if self.logit_norm_enabled:
            if self.logit_batch_norm_stopgrad_enabled:
                rms = logits.square().mean().sqrt().clamp_min(1.0e-4)
            else:
                rms = logits.square().mean(dim=1, keepdim=True).sqrt().clamp_min(1.0e-4)
            if self.logit_norm_stopgrad_enabled or self.logit_batch_norm_stopgrad_enabled:
                rms = rms.detach()
            logits = self.logit_norm_target * logits / rms
        if self.logit_softcap_enabled:
            cap = self.logit_softcap_value.clamp_min(0.25)
            logits = cap * torch.tanh(logits / cap)
        return self.logit_gain.clamp(0.25, 4.0) * logits

    def manual_kernel_available(self) -> bool:
        return True

    def manual_ce_forward_cache(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        with torch.no_grad():
            z = self._input(x)
            centers = self.hinge_centers.view(1, 1, -1)
            zu = z.unsqueeze(-1)
            pos = F.relu(zu - centers).flatten(1)
            neg = F.relu(-zu - centers).flatten(1)
            direct_parts = [z, pos, neg]
            if self.direct_abs_square_enabled:
                direct_parts.append(z.abs() - self.zabs_mean)
                direct_parts.append(z.square() - self.z2_mean)
            elif self.direct_absmix_square_enabled:
                direct_parts.append((z.abs() - self.zabs_mean) + float(self.direct_absmix_square_scale) * (z.square() - self.z2_mean))
            elif self.direct_square_enabled:
                direct_parts.append(z.square() - self.z2_mean)
            elif self.direct_signed_square_enabled:
                direct_parts.append(z * z.abs() - self.zsignedsq_mean)
            elif self.direct_cubic_enabled:
                direct_parts.append(z.pow(3) - self.z3_mean)
            elif self.direct_abs_enabled:
                direct_parts.append(z.abs() - self.zabs_mean)
            if self.direct_normabs_enabled:
                normabs = (z.abs().mean(dim=1, keepdim=True) - self.zabs_scalar_mean) * math.sqrt(max(1, self.input_dim))
                direct_parts.append(normabs)
            if self.direct_meanstat_enabled:
                meanstat = (z.mean(dim=1, keepdim=True) - self.zmean_scalar_mean) * math.sqrt(max(1, self.input_dim))
                direct_parts.append(meanstat)
            if self.direct_groupabs_count:
                group_size = max(1, self.input_dim // int(self.direct_groupabs_count))
                group_feats = []
                for group_idx in range(int(self.direct_groupabs_count)):
                    start = group_idx * group_size
                    end = self.input_dim if group_idx == int(self.direct_groupabs_count) - 1 else min(self.input_dim, (group_idx + 1) * group_size)
                    group_abs = (z[:, start:end].abs().mean(dim=1, keepdim=True) - self.zgroupabs_mean[group_idx : group_idx + 1]) * math.sqrt(max(1, end - start))
                    group_feats.append(group_abs)
                direct_parts.append(torch.cat(group_feats, dim=1))
            direct_feats = torch.cat(direct_parts, dim=1)
            direct_logits = direct_feats @ self.direct_readout / math.sqrt(max(1, self.input_dim))
            q = self._quadratic_projection(z) / self.quad_feature_std.clamp_min(1.0e-4)
            if self.quad_batch_rms_enabled:
                q = q / q.square().mean(dim=0, keepdim=True).sqrt().detach().clamp_min(1.0e-4)
            if self.quad_tanh_enabled:
                q = torch.tanh(q)
            q2 = q.square() - q.square().mean(dim=0, keepdim=True)
            quad_feats = torch.cat([q, q2], dim=1)
            quad_logits = quad_feats @ self.quad_readout
            if self.class_branch_scale_enabled:
                logits_unscaled = self.branch_scale[0].view(1, -1) * direct_logits + self.branch_scale[1].view(1, -1) * quad_logits + self.bias
            else:
                logits_unscaled = self.branch_scale[0] * direct_logits + self.branch_scale[1] * quad_logits + self.bias
            logits_pre_gain = logits_unscaled
            if self.logit_norm_enabled:
                if self.logit_batch_norm_stopgrad_enabled:
                    rms = logits_unscaled.square().mean().sqrt().clamp_min(1.0e-4)
                else:
                    rms = logits_unscaled.square().mean(dim=1, keepdim=True).sqrt().clamp_min(1.0e-4)
                logits_pre_gain = self.logit_norm_target * logits_unscaled / rms
            logits_pre_softcap = logits_pre_gain
            if self.logit_softcap_enabled:
                cap = self.logit_softcap_value.clamp_min(0.25)
                logits_pre_gain = cap * torch.tanh(logits_pre_gain / cap)
            logits = self.logit_gain.clamp(0.25, 4.0) * logits_pre_gain
            return logits, (z, direct_feats, q, quad_feats, direct_logits, quad_logits, logits_unscaled, logits_pre_gain, logits_pre_softcap)

    def manual_logits_backward_from_cache(self, logits: torch.Tensor, cache: tuple[torch.Tensor, ...], grad_logits: torch.Tensor) -> None:
        with torch.no_grad():
            z, direct_feats, q, quad_feats, direct_logits, quad_logits, logits_unscaled, logits_pre_gain, logits_pre_softcap = cache
            grad_output = grad_logits
            gain = self.logit_gain.clamp(0.25, 4.0)
            g_pre_gain = grad_output * gain
            if self.logit_softcap_enabled:
                cap = self.logit_softcap_value.clamp_min(0.25)
                t = torch.tanh(logits_pre_softcap / cap)
                g_pre_gain = g_pre_gain * (1.0 - t.square())
            if self.logit_norm_enabled:
                if self.logit_batch_norm_stopgrad_enabled:
                    rms = logits_unscaled.square().mean().sqrt().clamp_min(1.0e-4)
                else:
                    rms = logits_unscaled.square().mean(dim=1, keepdim=True).sqrt().clamp_min(1.0e-4)
                if self.logit_norm_stopgrad_enabled or self.logit_batch_norm_stopgrad_enabled:
                    g_logits = self.logit_norm_target * g_pre_gain / rms
                else:
                    dot = (g_pre_gain * logits_unscaled).sum(dim=1, keepdim=True)
                    denom = float(max(1, self.output_dim)) * rms.square()
                    g_logits = self.logit_norm_target * (g_pre_gain - logits_unscaled * dot / denom) / rms
            else:
                g_logits = g_pre_gain
            gain_mask = ((self.logit_gain >= 0.25) & (self.logit_gain <= 4.0)).to(dtype=grad_output.dtype)
            if isinstance(self.logit_gain, nn.Parameter) and self.logit_gain.requires_grad:
                gain_grad = grad_output * logits_pre_gain
                if self.logit_gain.numel() == self.output_dim:
                    self.logit_gain.grad = gain_grad.sum(dim=0).view_as(self.logit_gain) * gain_mask
                else:
                    self.logit_gain.grad = gain_grad.sum().view_as(self.logit_gain) * gain_mask
            if isinstance(self.branch_scale, nn.Parameter) and self.branch_scale.requires_grad:
                if self.class_branch_scale_enabled:
                    self.branch_scale.grad = torch.stack(
                        [
                            (g_logits * direct_logits).sum(dim=0),
                            (g_logits * quad_logits).sum(dim=0),
                        ]
                    ).view_as(self.branch_scale)
                else:
                    self.branch_scale.grad = torch.stack(
                        [
                            (g_logits * direct_logits).sum(),
                            (g_logits * quad_logits).sum(),
                        ]
                    ).view_as(self.branch_scale)
            self.bias.grad = g_logits.sum(dim=0)

            if self.class_branch_scale_enabled:
                grad_direct_logits = g_logits * self.branch_scale[0].view(1, -1)
            else:
                grad_direct_logits = g_logits * self.branch_scale[0]
            self.direct_readout.grad = direct_feats.transpose(0, 1) @ grad_direct_logits / math.sqrt(max(1, self.input_dim))

            if self.class_branch_scale_enabled:
                grad_quad_logits = g_logits * self.branch_scale[1].view(1, -1)
            else:
                grad_quad_logits = g_logits * self.branch_scale[1]
            self.quad_readout.grad = quad_feats.transpose(0, 1) @ grad_quad_logits
            grad_quad_feats = grad_quad_logits @ self.quad_readout.transpose(0, 1)
            grad_q = grad_quad_feats[:, : self.hidden_dim] + 2.0 * q * grad_quad_feats[:, self.hidden_dim :]
            if self.quad_tanh_enabled:
                grad_q = grad_q * (1.0 - q.square())
            if self.quad_batch_rms_enabled:
                q_pre = self._quadratic_projection(z) / self.quad_feature_std.clamp_min(1.0e-4)
                q_rms = q_pre.square().mean(dim=0, keepdim=True).sqrt().detach().clamp_min(1.0e-4)
                grad_q = grad_q / q_rms
            grad_q = grad_q / self.quad_feature_std.clamp_min(1.0e-4)
            if bool(getattr(self, "quad_pairscale_enabled", False)):
                qa = z.index_select(1, self.quad_pair_a)
                qb = z.index_select(1, self.quad_pair_b)
                base = (qa + qb * self.quad_pair_sign_b.view(1, -1)) * self.quad_pair_norm.view(1, -1)
                self.quad_proj_scale.grad = (grad_q * base).sum(dim=0)
            elif bool(getattr(self, "quad_sparsep_enabled", False)):
                idx = self.quad_sparse_idx.reshape(-1)
                gathered = z.index_select(1, idx).view(int(z.shape[0]), self.hidden_dim, int(self.quad_sparsep_k))
                self.quad_sparse_weight.grad = (grad_q.unsqueeze(2) * gathered).sum(dim=0)
            elif isinstance(self.quad_proj, nn.Parameter) and self.quad_proj.requires_grad:
                self.quad_proj.grad = z.transpose(0, 1) @ grad_q

    def manual_ce_backward_from_cache(self, logits: torch.Tensor, cache: tuple[torch.Tensor, ...], y: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            probs = torch.softmax(logits, dim=1)
            loss = F.cross_entropy(logits, y)
            grad_output = probs
            grad_output[torch.arange(int(y.numel()), device=y.device), y] -= 1.0
            grad_output = grad_output / float(max(1, int(y.numel())))
            self.manual_logits_backward_from_cache(logits, cache, grad_output)
            return loss.detach()

    def manual_gradient_audit(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
        params = [p for p in self.parameters() if p.requires_grad]
        self.zero_grad(set_to_none=True)
        logits, cache = self.manual_ce_forward_cache(x)
        self.manual_ce_backward_from_cache(logits, cache, y)
        manual_grads = [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in params]
        self.zero_grad(set_to_none=True)
        ref_logits = self(x)
        F.cross_entropy(ref_logits, y).backward()
        ref_grads = [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in params]
        self.zero_grad(set_to_none=True)
        relerrs: List[float] = []
        coses: List[float] = []
        for gm, gr in zip(manual_grads, ref_grads):
            denom = gr.norm().clamp_min(1.0e-8)
            relerrs.append(float((gm - gr).norm().div(denom).detach().item()))
            if gm.norm().item() > 0.0 and gr.norm().item() > 0.0:
                coses.append(float(F.cosine_similarity(gm.flatten(), gr.flatten(), dim=0).detach().item()))
        return {
            "manual_forward_available": 1.0,
            "manual_backward_available": 1.0,
            "grad_relerr_max": max(relerrs) if relerrs else float("inf"),
            "grad_cos_min": min(coses) if coses else -1.0,
            "output_max_abs_error": float((logits - ref_logits).abs().max().detach().item()),
        }

    def basis_diagnostics(self, x: torch.Tensor) -> Dict[str, float]:
        with torch.no_grad():
            feats = self.frozen_readout_features(x).float()
            energy = feats.square().mean(dim=0)
            prob = energy / energy.sum().clamp_min(EPS)
            entropy = -(prob * (prob + EPS).log()).sum() / math.log(max(2, int(prob.numel())))
            dead = (energy < 1.0e-8).float().mean()
            centered = feats - feats.mean(dim=0, keepdim=True)
            try:
                s = torch.linalg.svdvals(centered[: min(256, int(centered.shape[0]))])
                rank = (s.square().sum().square() / s.pow(4).sum().clamp_min(EPS)).item()
                cond = (s.max() / s[s > 1.0e-7].min()).item() if bool((s > 1.0e-7).any()) else float("inf")
            except RuntimeError:
                rank = 0.0
                cond = float("inf")
            return {
                "basis_entropy": float(entropy.item()),
                "dead_basis_fraction": float(dead.item()),
                "basis_effective_rank": float(rank),
                "basis_condition_proxy": float(cond),
                "basis_output_norm_p95": float(torch.quantile(feats.norm(dim=1), 0.95).item()),
            }

    def basis_functional_direction(self, mode: str) -> List[torch.Tensor]:
        with torch.no_grad():
            direct = -self.direct_readout.detach().clone()
            quad_readout = -self.quad_readout.detach().clone()
            if mode in {"basis_aware_snr_projected", "basis_aware_orthogonal"}:
                direct[self.input_dim :, :].mul_(0.5)
                quad_readout[: self.hidden_dim, :].mul_(0.25)
            out = [direct]
            if isinstance(self.quad_proj, nn.Parameter) and self.quad_proj.requires_grad:
                out.append(torch.zeros_like(self.quad_proj))
            out.append(quad_readout)
            if isinstance(self.branch_scale, nn.Parameter) and self.branch_scale.requires_grad:
                out.append(torch.zeros_like(self.branch_scale))
            out.append(-self.bias.detach().clone())
            if isinstance(self.logit_gain, nn.Parameter) and self.logit_gain.requires_grad:
                out.insert(-1, torch.zeros_like(self.logit_gain))
            return out


def _gated_branch_dims(hidden_dim: int) -> tuple[int, int]:
    legendre_hidden = max(8, int(hidden_dim) // 4)
    quadratic_hidden = max(16, int(hidden_dim) // 2)
    return legendre_hidden, quadratic_hidden


def _legendre4(z: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        [
            torch.ones_like(z),
            z,
            0.5 * (3.0 * z.square() - 1.0),
            0.5 * (5.0 * z * z.square() - 3.0 * z),
        ],
        dim=-1,
    )


def _legendre4_derivative(z: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        [
            torch.zeros_like(z),
            torch.ones_like(z),
            3.0 * z,
            0.5 * (15.0 * z.square() - 3.0),
        ],
        dim=-1,
    )


class _GatedLQFastReuseManualFunction(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        x: torch.Tensor,
        mu: torch.Tensor,
        std: torch.Tensor,
        block_mu: torch.Tensor,
        block_whiten: torch.Tensor,
        block_norm_target_rms: torch.Tensor,
        residual_geom_mix: torch.Tensor,
        leg_w1: torch.Tensor,
        leg_w2: torch.Tensor,
        quad_proj: torch.Tensor,
        quad_readout: torch.Tensor,
        branch_scale: torch.Tensor,
        logit_gain: torch.Tensor,
        bias: torch.Tensor,
        direct_readout: torch.Tensor,
        quad_feature_std: torch.Tensor,
        leg_logit_std: torch.Tensor,
        quad_logit_std: torch.Tensor,
        direct_skip_scale: torch.Tensor,
        input_dim: int,
        input_norm_padded_dim: int,
        input_norm_block_size: int,
        legendre_hidden: int,
        quad_feature_norm_enabled: bool,
        branch_output_norm_enabled: bool,
    ) -> torch.Tensor:
        del input_dim, legendre_hidden
        batch = int(x.shape[0])
        base = (x - mu) / std
        pad = int(input_norm_padded_dim) - int(x.shape[1])
        xpad = F.pad(x, (0, pad)) if pad > 0 else x
        blocks = xpad.reshape(batch, -1, int(input_norm_block_size))
        z = torch.einsum("bni,nij->bnj", blocks - block_mu.unsqueeze(0), block_whiten)
        z = z.reshape(batch, int(input_norm_padded_dim))[:, : int(x.shape[1])]
        rms = z.float().square().mean(dim=1, keepdim=True).sqrt().clamp_min(1.0e-3)
        z = z / rms * block_norm_target_rms
        mixed = (1.0 - residual_geom_mix) * base + residual_geom_mix * z
        leg_z = torch.tanh(mixed)
        quad_x = mixed.clamp(-3.0, 3.0)

        sqrt_input = math.sqrt(max(1, int(x.shape[1])))
        sqrt_leg = math.sqrt(max(1, int(leg_w1.shape[1])))
        b1 = _legendre4(leg_z)
        h_pre = torch.einsum("bdk,dhk->bh", b1, leg_w1) / sqrt_input
        h = torch.tanh(h_pre)
        b2 = _legendre4(h)
        leg_logits = torch.einsum("bhk,hck->bc", b2, leg_w2) / sqrt_leg

        q_raw = quad_x @ quad_proj
        q = q_raw / quad_feature_std.clamp_min(1.0e-4) if bool(quad_feature_norm_enabled) else q_raw
        centered_square = q.square() - q.square().mean(dim=0, keepdim=True).detach()
        quad_feats = torch.stack([q, centered_square], dim=2)
        quad_logits = torch.einsum("bhk,hkc->bc", quad_feats, quad_readout)

        if bool(branch_output_norm_enabled):
            leg_logits_norm = leg_logits / leg_logit_std.clamp_min(1.0e-4)
            quad_logits_norm = quad_logits / quad_logit_std.clamp_min(1.0e-4)
        else:
            leg_logits_norm = leg_logits
            quad_logits_norm = quad_logits

        direct_logits = torch.einsum("bdk,dck->bc", b1, direct_readout) / sqrt_input
        logits_unscaled = (
            branch_scale[0] * leg_logits_norm
            + branch_scale[1] * quad_logits_norm
            + bias
            + direct_skip_scale * direct_logits
        )
        gain = logit_gain.clamp(0.25, 4.0)
        out = gain * logits_unscaled
        ctx.save_for_backward(
            b1,
            h,
            b2,
            quad_x,
            q,
            quad_feats,
            leg_logits_norm,
            quad_logits_norm,
            direct_logits,
            logits_unscaled,
            leg_w2,
            quad_proj,
            quad_readout,
            branch_scale,
            logit_gain,
            quad_feature_std,
            leg_logit_std,
            quad_logit_std,
            direct_skip_scale,
        )
        ctx.sqrt_input = sqrt_input
        ctx.sqrt_leg = sqrt_leg
        ctx.quad_feature_norm_enabled = bool(quad_feature_norm_enabled)
        ctx.branch_output_norm_enabled = bool(branch_output_norm_enabled)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        (
            b1,
            h,
            b2,
            quad_x,
            q,
            quad_feats,
            leg_logits_norm,
            quad_logits_norm,
            direct_logits,
            logits_unscaled,
            leg_w2,
            _quad_proj,
            quad_readout,
            branch_scale,
            logit_gain,
            quad_feature_std,
            leg_logit_std,
            quad_logit_std,
            direct_skip_scale,
        ) = ctx.saved_tensors
        del _quad_proj
        gain = logit_gain.clamp(0.25, 4.0)
        g_logits = grad_output * gain
        gain_mask = ((logit_gain >= 0.25) & (logit_gain <= 4.0)).to(dtype=grad_output.dtype)
        grad_logit_gain = (grad_output * logits_unscaled).sum().view_as(logit_gain) * gain_mask
        grad_branch_scale = torch.stack(
            [
                (g_logits * leg_logits_norm).sum(),
                (g_logits * quad_logits_norm).sum(),
            ]
        ).view_as(branch_scale)
        grad_bias = g_logits.sum(dim=0)

        grad_direct_logits = g_logits * direct_skip_scale
        grad_direct_readout = torch.einsum("bdk,bc->dck", b1, grad_direct_logits) / ctx.sqrt_input

        grad_leg_logits = g_logits * branch_scale[0]
        if ctx.branch_output_norm_enabled:
            grad_leg_logits = grad_leg_logits / leg_logit_std.clamp_min(1.0e-4)
        grad_leg_w2 = torch.einsum("bhk,bc->hck", b2, grad_leg_logits) / ctx.sqrt_leg
        grad_b2 = torch.einsum("bc,hck->bhk", grad_leg_logits, leg_w2) / ctx.sqrt_leg
        grad_h = (grad_b2 * _legendre4_derivative(h)).sum(dim=2)
        grad_h_pre = grad_h * (1.0 - h.square())
        grad_leg_w1 = torch.einsum("bdk,bh->dhk", b1, grad_h_pre) / ctx.sqrt_input

        grad_quad_logits = g_logits * branch_scale[1]
        if ctx.branch_output_norm_enabled:
            grad_quad_logits = grad_quad_logits / quad_logit_std.clamp_min(1.0e-4)
        grad_quad_readout = torch.einsum("bhk,bc->hkc", quad_feats, grad_quad_logits)
        grad_quad_feats = torch.einsum("bc,hkc->bhk", grad_quad_logits, quad_readout)
        grad_q = grad_quad_feats[:, :, 0] + 2.0 * q * grad_quad_feats[:, :, 1]
        if ctx.quad_feature_norm_enabled:
            grad_q = grad_q / quad_feature_std.clamp_min(1.0e-4)
        grad_quad_proj = quad_x.transpose(0, 1) @ grad_q

        return (
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            grad_leg_w1,
            grad_leg_w2,
            grad_quad_proj,
            grad_quad_readout,
            grad_branch_scale,
            grad_logit_gain,
            grad_bias,
            grad_direct_readout,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class GatedLegendreQuadraticKAN(nn.Module):
    """Official-capable coupling of Legendre edge basis and quadratic sketch.

    This primitive is the v12.4 follow-up to the B10/B11 and B3b split:
    B10/B11 have global quadratic coverage, while B3b has synthetic
    trainability but task failure.  The branch coupling keeps all learnable
    tensors as basis weights/projections and does not add an MLP shortcut.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        spec: PrimitiveSpec,
        x_for_stats: torch.Tensor,
        seed: int,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.spec = spec
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.hidden_dim = int(spec.hidden_dim)
        self.k = 4
        self.legendre_hidden, self.quadratic_hidden = _gated_branch_dims(self.hidden_dim)
        xs = x_for_stats[: min(4096, int(x_for_stats.shape[0]))].to(device=device, dtype=torch.float32)
        self.register_buffer("mu", xs.mean(dim=0))
        self.register_buffer("std", xs.std(dim=0).clamp_min(1.0e-3))
        geom_norm_variants = {
            "geom_input_norm",
            "quad_boost_geom_input_norm",
            "geom_input_basis_norm",
            "quadboost_geom_input_basis_norm",
        }
        residual_geom_norm_variants = {
            "residual_geom_input_norm",
            "quad_boost_residual_geom_input_norm",
            "residual_geom_input_basis_norm",
            "quadboost_residual_geom_input_basis_norm",
            "residual_geom_mix15",
            "residual_geom_mix25",
            "quad_boost_residual_geom_mix15",
            "quad_boost_residual_geom_mix25",
            "basis_specific_residual_quad_norm",
            "quad_boost_basis_specific_residual_quad_norm",
            "basis_specific_residual_quad_midboost",
            "basis_specific_residual_quad_lowboost",
            "basis_specific_residual_quad_lowboost_temp065",
            "basis_specific_residual_quad_lowboost_temp075",
            "basis_specific_residual_quad_lowboost_temp050",
            "basis_specific_residual_quad_lowboost_temp050_directskip",
            "basis_specific_residual_quad_lowboost_temp065_directskip",
            "basis_specific_residual_quad_lowboost_temp075_directskip",
            "basis_specific_residual_quad_lowboost_temp050_directskip_fastreuse",
            "basis_specific_residual_quad_lowboost_temp065_directskip_fastreuse",
            "basis_specific_residual_quad_lowboost_temp075_directskip_fastreuse",
            "basis_specific_residual_quad_lowboost_temp075_directskip_fastreuse_manualbw",
            "basis_specific_residual_quad_lowboost_temp075_directskip050_fastreuse",
            "basis_specific_residualmix15_quad_lowboost_temp065_directskip_fastreuse",
            "basis_specific_residualmix15_quad_lowboost_temp075_directskip_fastreuse",
        }
        basis_specific_quad_norm_variants = {
            "basis_specific_quad_norm",
            "quad_boost_basis_specific_quad_norm",
            "basis_specific_quad_norm_temp075",
            "basis_specific_quad_norm_temp050",
            "basis_specific_residual_quad_norm",
            "quad_boost_basis_specific_residual_quad_norm",
            "basis_specific_residual_quad_midboost",
            "basis_specific_residual_quad_lowboost",
            "basis_specific_residual_quad_lowboost_temp065",
            "basis_specific_residual_quad_lowboost_temp075",
            "basis_specific_residual_quad_lowboost_temp050",
            "basis_specific_residual_quad_lowboost_temp050_directskip",
            "basis_specific_residual_quad_lowboost_temp065_directskip",
            "basis_specific_residual_quad_lowboost_temp075_directskip",
            "basis_specific_residual_quad_lowboost_temp050_directskip_fastreuse",
            "basis_specific_residual_quad_lowboost_temp065_directskip_fastreuse",
            "basis_specific_residual_quad_lowboost_temp075_directskip_fastreuse",
            "basis_specific_residual_quad_lowboost_temp075_directskip_fastreuse_manualbw",
            "basis_specific_residual_quad_lowboost_temp075_directskip050_fastreuse",
            "basis_specific_residualmix15_quad_lowboost_temp065_directskip_fastreuse",
            "basis_specific_residualmix15_quad_lowboost_temp075_directskip_fastreuse",
        }
        group_rms_norm_variants = {
            "group_rms_input_norm",
            "quad_boost_group_rms_input_norm",
            "group_rms_input_basis_norm",
            "quadboost_group_rms_input_basis_norm",
        }
        self.input_block_norm_enabled = spec.init_variant in geom_norm_variants
        self.input_residual_block_norm_enabled = spec.init_variant in residual_geom_norm_variants
        self.input_group_rms_norm_enabled = spec.init_variant in group_rms_norm_variants
        self.basis_specific_quad_input_norm_enabled = spec.init_variant in basis_specific_quad_norm_variants
        self.input_norm_block_size = 32
        self.input_norm_padded_dim = int(math.ceil(float(self.input_dim) / float(self.input_norm_block_size)) * self.input_norm_block_size)
        self.register_buffer("geom_std", self.std)
        self.register_buffer("group_norm_target_rms", torch.tensor([0.75], device=device))
        residual_mix = 0.35
        if spec.init_variant in {
            "residual_geom_mix15",
            "quad_boost_residual_geom_mix15",
            "basis_specific_residualmix15_quad_lowboost_temp065_directskip_fastreuse",
            "basis_specific_residualmix15_quad_lowboost_temp075_directskip_fastreuse",
        }:
            residual_mix = 0.15
        if spec.init_variant in {"residual_geom_mix25", "quad_boost_residual_geom_mix25"}:
            residual_mix = 0.25
        if spec.init_variant in {
            "basis_specific_residual_quad_norm",
            "quad_boost_basis_specific_residual_quad_norm",
            "basis_specific_residual_quad_midboost",
            "basis_specific_residual_quad_lowboost",
            "basis_specific_residual_quad_lowboost_temp065",
            "basis_specific_residual_quad_lowboost_temp075",
            "basis_specific_residual_quad_lowboost_temp050",
            "basis_specific_residual_quad_lowboost_temp050_directskip",
            "basis_specific_residual_quad_lowboost_temp065_directskip",
            "basis_specific_residual_quad_lowboost_temp075_directskip",
            "basis_specific_residual_quad_lowboost_temp050_directskip_fastreuse",
            "basis_specific_residual_quad_lowboost_temp065_directskip_fastreuse",
            "basis_specific_residual_quad_lowboost_temp075_directskip_fastreuse",
            "basis_specific_residual_quad_lowboost_temp075_directskip_fastreuse_manualbw",
            "basis_specific_residual_quad_lowboost_temp075_directskip050_fastreuse",
        }:
            residual_mix = 0.25
        self.register_buffer("residual_geom_mix", torch.tensor([residual_mix], device=device))
        if self.input_group_rms_norm_enabled:
            var = xs.var(dim=0, unbiased=False)
            global_var = var.mean().clamp_min(1.0e-6)
            shrink_std = (0.50 * var + 0.50 * global_var).sqrt().clamp_min(1.0e-3)
            self.geom_std.copy_(shrink_std)
        if self.input_block_norm_enabled or self.input_residual_block_norm_enabled:
            pad = self.input_norm_padded_dim - self.input_dim
            xpad = F.pad(xs, (0, pad)) if pad > 0 else xs
            blocks = xpad.reshape(int(xpad.shape[0]), -1, self.input_norm_block_size)
            block_mu = blocks.mean(dim=0)
            centered = blocks - block_mu.unsqueeze(0)
            denom = max(1, int(centered.shape[0]) - 1)
            cov = torch.einsum("nbi,nbj->bij", centered, centered) / float(denom)
            eye = torch.eye(self.input_norm_block_size, device=device).unsqueeze(0)
            trace = cov.diagonal(dim1=1, dim2=2).mean(dim=1).clamp_min(1.0e-6)
            cov = 0.90 * cov + 0.10 * trace[:, None, None] * eye
            evals, evecs = torch.linalg.eigh(cov.float())
            inv_sqrt = torch.diag_embed(evals.clamp_min(1.0e-4).rsqrt())
            block_whiten = evecs @ inv_sqrt @ evecs.transpose(-1, -2)
            self.register_buffer("block_mu", block_mu)
            self.register_buffer("block_whiten", block_whiten)
            self.register_buffer("block_norm_target_rms", torch.tensor([0.75], device=device))
        self.register_buffer("centers", torch.linspace(-1.0, 1.0, self.k, device=device))
        self.register_buffer("scales", torch.tensor([max(0.2, 2.0 / max(1, self.k - 1))], device=device))
        gen = torch.Generator(device=device).manual_seed(int(seed) + 12414)
        self.leg_w1 = nn.Parameter(
            torch.randn(self.input_dim, self.legendre_hidden, self.k, device=device, generator=gen)
            / math.sqrt(max(1, self.input_dim * self.k))
        )
        self.leg_w2 = nn.Parameter(
            torch.randn(self.legendre_hidden, self.output_dim, self.k, device=device, generator=gen)
            / math.sqrt(max(1, self.legendre_hidden * self.k))
        )
        self.quad_proj = nn.Parameter(
            torch.randn(self.input_dim, self.quadratic_hidden, device=device, generator=gen)
            / math.sqrt(max(1, self.input_dim))
        )
        self.quad_readout = nn.Parameter(
            torch.randn(self.quadratic_hidden, 2, self.output_dim, device=device, generator=gen)
            / math.sqrt(max(1, self.quadratic_hidden * 2))
        )
        self.quad_feature_norm_enabled = spec.init_variant in basis_specific_quad_norm_variants
        self.register_buffer("quad_feature_std", torch.ones(self.quadratic_hidden, device=device))
        if self.quad_feature_norm_enabled:
            self._calibrate_quadratic_feature_norm(xs)
        branch_init = [1.0, 0.25]
        if spec.init_variant == "quad_boost":
            branch_init = [0.75, 0.75]
        if spec.init_variant == "loss_gain":
            branch_init = [1.0, 0.75]
        if spec.init_variant == "quad_boost_loss_gain":
            branch_init = [0.75, 0.75]
        if spec.init_variant == "geom_input_norm":
            branch_init = [1.0, 0.25]
        if spec.init_variant == "quad_boost_geom_input_norm":
            branch_init = [0.75, 0.75]
        if spec.init_variant == "basis_norm":
            branch_init = [0.20, 0.20]
        if spec.init_variant == "quad_boost_basis_norm":
            branch_init = [0.15, 0.25]
        if spec.init_variant == "geom_input_basis_norm":
            branch_init = [0.20, 0.20]
        if spec.init_variant == "quadboost_geom_input_basis_norm":
            branch_init = [0.15, 0.25]
        if spec.init_variant == "group_rms_input_norm":
            branch_init = [1.0, 0.25]
        if spec.init_variant == "quad_boost_group_rms_input_norm":
            branch_init = [0.75, 0.75]
        if spec.init_variant == "group_rms_input_basis_norm":
            branch_init = [0.20, 0.20]
        if spec.init_variant == "quadboost_group_rms_input_basis_norm":
            branch_init = [0.15, 0.25]
        if spec.init_variant == "residual_geom_input_norm":
            branch_init = [1.0, 0.25]
        if spec.init_variant == "quad_boost_residual_geom_input_norm":
            branch_init = [0.75, 0.75]
        if spec.init_variant == "residual_geom_input_basis_norm":
            branch_init = [0.20, 0.20]
        if spec.init_variant == "quadboost_residual_geom_input_basis_norm":
            branch_init = [0.15, 0.25]
        if spec.init_variant in {"residual_geom_mix15", "residual_geom_mix25"}:
            branch_init = [1.0, 0.25]
        if spec.init_variant in {"quad_boost_residual_geom_mix15", "quad_boost_residual_geom_mix25"}:
            branch_init = [0.75, 0.75]
        if spec.init_variant == "basis_specific_quad_norm":
            branch_init = [1.0, 0.25]
        if spec.init_variant in {"basis_specific_quad_norm_temp075", "basis_specific_quad_norm_temp050"}:
            branch_init = [1.0, 0.25]
        if spec.init_variant == "quad_boost_basis_specific_quad_norm":
            branch_init = [0.75, 0.75]
        if spec.init_variant == "basis_specific_residual_quad_norm":
            branch_init = [1.0, 0.25]
        if spec.init_variant == "quad_boost_basis_specific_residual_quad_norm":
            branch_init = [0.75, 0.75]
        if spec.init_variant == "basis_specific_residual_quad_midboost":
            branch_init = [1.0, 0.50]
        if spec.init_variant == "basis_specific_residual_quad_lowboost":
            branch_init = [1.0, 0.35]
        if spec.init_variant in {
            "basis_specific_residual_quad_lowboost_temp065",
            "basis_specific_residual_quad_lowboost_temp075",
            "basis_specific_residual_quad_lowboost_temp050",
            "basis_specific_residual_quad_lowboost_temp050_directskip",
            "basis_specific_residual_quad_lowboost_temp065_directskip",
            "basis_specific_residual_quad_lowboost_temp075_directskip",
            "basis_specific_residual_quad_lowboost_temp050_directskip_fastreuse",
            "basis_specific_residual_quad_lowboost_temp065_directskip_fastreuse",
            "basis_specific_residual_quad_lowboost_temp075_directskip_fastreuse",
            "basis_specific_residual_quad_lowboost_temp075_directskip_fastreuse_manualbw",
            "basis_specific_residual_quad_lowboost_temp075_directskip050_fastreuse",
            "basis_specific_residualmix15_quad_lowboost_temp065_directskip_fastreuse",
            "basis_specific_residualmix15_quad_lowboost_temp075_directskip_fastreuse",
        }:
            branch_init = [1.0, 0.35]
        self.branch_scale = nn.Parameter(torch.tensor(branch_init, device=device))
        gain_init = 2.0 if spec.init_variant in {"loss_gain", "quad_boost_loss_gain"} else 1.0
        if spec.init_variant in {"basis_specific_residual_quad_lowboost_temp075", "basis_specific_residual_quad_lowboost_temp075_directskip", "basis_specific_residual_quad_lowboost_temp075_directskip_fastreuse", "basis_specific_residual_quad_lowboost_temp075_directskip_fastreuse_manualbw", "basis_specific_residual_quad_lowboost_temp075_directskip050_fastreuse", "basis_specific_residualmix15_quad_lowboost_temp075_directskip_fastreuse"}:
            gain_init = 0.75
        if spec.init_variant in {"basis_specific_residual_quad_lowboost_temp065", "basis_specific_residual_quad_lowboost_temp065_directskip", "basis_specific_residual_quad_lowboost_temp065_directskip_fastreuse", "basis_specific_residualmix15_quad_lowboost_temp065_directskip_fastreuse"}:
            gain_init = 0.65
        if spec.init_variant in {"basis_specific_residual_quad_lowboost_temp050", "basis_specific_residual_quad_lowboost_temp050_directskip", "basis_specific_residual_quad_lowboost_temp050_directskip_fastreuse"}:
            gain_init = 0.50
        if spec.init_variant == "basis_specific_quad_norm_temp075":
            gain_init = 0.75
        if spec.init_variant == "basis_specific_quad_norm_temp050":
            gain_init = 0.50
        self.logit_gain = nn.Parameter(torch.tensor([gain_init], device=device))
        self.bias = nn.Parameter(torch.zeros(self.output_dim, device=device))
        self.direct_skip_enabled = "directskip" in str(spec.init_variant)
        self.fast_reuse_forward_enabled = "fastreuse" in str(spec.init_variant)
        self.manual_backward_enabled = "manualbw" in str(spec.init_variant)
        if self.direct_skip_enabled:
            self.direct_readout = nn.Parameter(
                torch.randn(self.input_dim, self.output_dim, self.k, device=device, generator=gen)
                / math.sqrt(max(1, self.input_dim * self.k))
            )
        direct_skip_scale = 0.50 if "directskip050" in str(spec.init_variant) else 0.25
        self.register_buffer("direct_skip_scale", torch.tensor([direct_skip_scale], device=device))
        self.register_buffer("leg_logit_std", torch.ones(1, device=device))
        self.register_buffer("quad_logit_std", torch.ones(1, device=device))
        self.branch_output_norm_enabled = spec.init_variant in {
            "basis_norm",
            "quad_boost_basis_norm",
            "geom_input_basis_norm",
            "quadboost_geom_input_basis_norm",
            "group_rms_input_basis_norm",
            "quadboost_group_rms_input_basis_norm",
            "residual_geom_input_basis_norm",
            "quadboost_residual_geom_input_basis_norm",
        }
        if self.branch_output_norm_enabled:
            self._calibrate_branch_output_norm(xs)

    def _calibrate_quadratic_feature_norm(self, xs: torch.Tensor) -> None:
        """Calibrate quadratic projection coordinates without labels."""
        with torch.no_grad():
            sample = xs[: min(2048, int(xs.shape[0]))]
            z = self._quadratic_input(sample) @ self.quad_proj
            self.quad_feature_std.copy_(z.float().std(dim=0).clamp_min(1.0e-4))

    def _calibrate_branch_output_norm(self, xs: torch.Tensor) -> None:
        """Calibrate branch output scale from unlabeled train-stream samples."""
        with torch.no_grad():
            sample = xs[: min(1024, int(xs.shape[0]))]
            leg_logits, _ = self._legendre_logits_and_features(sample)
            quad_logits, _ = self._quadratic_logits_and_features(sample)
            self.leg_logit_std.copy_(leg_logits.float().std().clamp_min(1.0e-4).view(1))
            self.quad_logit_std.copy_(quad_logits.float().std().clamp_min(1.0e-4).view(1))

    @property
    def edge_param_count(self) -> int:
        return int(
            self.leg_w1.numel()
            + self.leg_w2.numel()
            + self.quad_proj.numel()
            + self.quad_readout.numel()
            + self.branch_scale.numel()
            + self.logit_gain.numel()
            + self.bias.numel()
            + (self.direct_readout.numel() if self.direct_skip_enabled else 0)
        )

    def _norm_input(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_residual_block_norm_enabled:
            base = (x - self.mu) / self.std
            pad = self.input_norm_padded_dim - self.input_dim
            xpad = F.pad(x, (0, pad)) if pad > 0 else x
            blocks = xpad.reshape(int(x.shape[0]), -1, self.input_norm_block_size)
            z = torch.einsum("bni,nij->bnj", blocks - self.block_mu.unsqueeze(0), self.block_whiten)
            z = z.reshape(int(x.shape[0]), self.input_norm_padded_dim)[:, : self.input_dim]
            rms = z.float().square().mean(dim=1, keepdim=True).sqrt().clamp_min(1.0e-3)
            z = z / rms * self.block_norm_target_rms
            mixed = (1.0 - self.residual_geom_mix) * base + self.residual_geom_mix * z
            return torch.tanh(mixed)
        if self.input_group_rms_norm_enabled:
            z = (x - self.mu) / self.geom_std.clamp_min(1.0e-3)
            pad = self.input_norm_padded_dim - self.input_dim
            zpad = F.pad(z, (0, pad)) if pad > 0 else z
            blocks = zpad.reshape(int(x.shape[0]), -1, self.input_norm_block_size)
            block_rms = blocks.float().square().mean(dim=2, keepdim=True).sqrt().clamp_min(1.0e-3)
            blocks = blocks / block_rms * self.group_norm_target_rms
            z = blocks.reshape(int(x.shape[0]), self.input_norm_padded_dim)[:, : self.input_dim]
            return torch.tanh(z)
        if self.input_block_norm_enabled:
            pad = self.input_norm_padded_dim - self.input_dim
            xpad = F.pad(x, (0, pad)) if pad > 0 else x
            blocks = xpad.reshape(int(x.shape[0]), -1, self.input_norm_block_size)
            z = torch.einsum("bni,nij->bnj", blocks - self.block_mu.unsqueeze(0), self.block_whiten)
            z = z.reshape(int(x.shape[0]), self.input_norm_padded_dim)[:, : self.input_dim]
            rms = z.float().square().mean(dim=1, keepdim=True).sqrt().clamp_min(1.0e-3)
            z = z / rms * self.block_norm_target_rms
            return torch.tanh(z)
        return torch.tanh((x - self.mu) / self.std)

    def _quadratic_input(self, x: torch.Tensor) -> torch.Tensor:
        """Basis-specific input normalization for the quadratic branch.

        Legendre features need bounded coordinates.  Quadratic features need
        the linear coordinate system to survive the input geometry layer, so
        these variants avoid the final tanh while still using fixed,
        unlabeled train-stream normalization and a conservative clamp.
        """
        if not self.basis_specific_quad_input_norm_enabled:
            return self._norm_input(x)
        base = (x - self.mu) / self.std
        if self.input_residual_block_norm_enabled:
            pad = self.input_norm_padded_dim - self.input_dim
            xpad = F.pad(x, (0, pad)) if pad > 0 else x
            blocks = xpad.reshape(int(x.shape[0]), -1, self.input_norm_block_size)
            z = torch.einsum("bni,nij->bnj", blocks - self.block_mu.unsqueeze(0), self.block_whiten)
            z = z.reshape(int(x.shape[0]), self.input_norm_padded_dim)[:, : self.input_dim]
            rms = z.float().square().mean(dim=1, keepdim=True).sqrt().clamp_min(1.0e-3)
            z = z / rms * self.block_norm_target_rms
            base = (1.0 - self.residual_geom_mix) * base + self.residual_geom_mix * z
        return base.clamp(-3.0, 3.0)

    def _paired_legendre_quadratic_inputs(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return equivalent Legendre and quadratic inputs while sharing norm work."""
        if self.basis_specific_quad_input_norm_enabled and self.input_residual_block_norm_enabled:
            base = (x - self.mu) / self.std
            pad = self.input_norm_padded_dim - self.input_dim
            xpad = F.pad(x, (0, pad)) if pad > 0 else x
            blocks = xpad.reshape(int(x.shape[0]), -1, self.input_norm_block_size)
            z = torch.einsum("bni,nij->bnj", blocks - self.block_mu.unsqueeze(0), self.block_whiten)
            z = z.reshape(int(x.shape[0]), self.input_norm_padded_dim)[:, : self.input_dim]
            rms = z.float().square().mean(dim=1, keepdim=True).sqrt().clamp_min(1.0e-3)
            z = z / rms * self.block_norm_target_rms
            mixed = (1.0 - self.residual_geom_mix) * base + self.residual_geom_mix * z
            return torch.tanh(mixed), mixed.clamp(-3.0, 3.0)
        if self.basis_specific_quad_input_norm_enabled:
            base = (x - self.mu) / self.std
            return self._norm_input(x), base.clamp(-3.0, 3.0)
        z = self._norm_input(x)
        return z, z

    def _forward_reuse_inputs(self, x: torch.Tensor) -> torch.Tensor:
        """Fast path for directskip variants; mathematically matches helper calls."""
        leg_z, quad_x = self._paired_legendre_quadratic_inputs(x)
        b1 = _basis_eval(leg_z, "legendre", self.k, self.centers, self.scales)
        h = torch.einsum("bdk,dhk->bh", b1, self.leg_w1) / math.sqrt(max(1, self.input_dim))
        h = torch.tanh(h)
        b2 = _basis_eval(h, "legendre", self.k, self.centers, self.scales)
        leg_logits = torch.einsum("bhk,hck->bc", b2, self.leg_w2) / math.sqrt(max(1, self.legendre_hidden))

        q = quad_x @ self.quad_proj
        if self.quad_feature_norm_enabled:
            q = q / self.quad_feature_std.clamp_min(1.0e-4)
        centered_square = q.square() - q.square().mean(dim=0, keepdim=True).detach()
        quad_logits = torch.einsum("bhk,hkc->bc", torch.stack([q, centered_square], dim=2), self.quad_readout)

        if self.branch_output_norm_enabled:
            leg_logits = leg_logits / self.leg_logit_std.clamp_min(1.0e-4)
            quad_logits = quad_logits / self.quad_logit_std.clamp_min(1.0e-4)
        logits = self.branch_scale[0] * leg_logits + self.branch_scale[1] * quad_logits + self.bias
        direct_logits = torch.einsum("bdk,dck->bc", b1, self.direct_readout) / math.sqrt(max(1, self.input_dim))
        logits = logits + self.direct_skip_scale * direct_logits
        return self.logit_gain.clamp(0.25, 4.0) * logits

    def _manual_kernel_ready(self) -> bool:
        return bool(
            self.manual_backward_enabled
            and self.fast_reuse_forward_enabled
            and self.direct_skip_enabled
            and self.input_residual_block_norm_enabled
            and self.basis_specific_quad_input_norm_enabled
            and self.k == 4
            and hasattr(self, "block_mu")
            and hasattr(self, "block_whiten")
            and hasattr(self, "direct_readout")
        )

    def _forward_reuse_inputs_manual(self, x: torch.Tensor) -> torch.Tensor:
        return _GatedLQFastReuseManualFunction.apply(
            x,
            self.mu,
            self.std,
            self.block_mu,
            self.block_whiten,
            self.block_norm_target_rms,
            self.residual_geom_mix,
            self.leg_w1,
            self.leg_w2,
            self.quad_proj,
            self.quad_readout,
            self.branch_scale,
            self.logit_gain,
            self.bias,
            self.direct_readout,
            self.quad_feature_std,
            self.leg_logit_std,
            self.quad_logit_std,
            self.direct_skip_scale,
            self.input_dim,
            self.input_norm_padded_dim,
            self.input_norm_block_size,
            self.legendre_hidden,
            self.quad_feature_norm_enabled,
            self.branch_output_norm_enabled,
        )

    def manual_kernel_available(self) -> bool:
        return self._manual_kernel_ready()

    def manual_gradient_audit(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
        if not self._manual_kernel_ready():
            return {
                "manual_forward_available": 0.0,
                "manual_backward_available": 0.0,
                "grad_relerr_max": float("inf"),
                "grad_cos_min": -1.0,
                "output_max_abs_error": float("inf"),
            }
        params = [p for p in self.parameters() if p.requires_grad]
        self.zero_grad(set_to_none=True)
        manual_logits = self._forward_reuse_inputs_manual(x)
        F.cross_entropy(manual_logits, y).backward()
        manual_grads = [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in params]
        self.zero_grad(set_to_none=True)
        ref_logits = self._forward_reuse_inputs(x)
        F.cross_entropy(ref_logits, y).backward()
        ref_grads = [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in params]
        self.zero_grad(set_to_none=True)
        relerrs: List[float] = []
        coses: List[float] = []
        for gm, gr in zip(manual_grads, ref_grads):
            denom = gr.norm().clamp_min(1.0e-8)
            relerrs.append(float((gm - gr).norm().div(denom).detach().item()))
            if gm.norm().item() > 0.0 and gr.norm().item() > 0.0:
                coses.append(float(F.cosine_similarity(gm.flatten(), gr.flatten(), dim=0).detach().item()))
        return {
            "manual_forward_available": 1.0,
            "manual_backward_available": 1.0,
            "grad_relerr_max": max(relerrs) if relerrs else float("inf"),
            "grad_cos_min": min(coses) if coses else -1.0,
            "output_max_abs_error": float((manual_logits - ref_logits).abs().max().detach().item()),
        }

    def manual_ce_forward_cache(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        if not self._manual_kernel_ready():
            raise RuntimeError("manual CE path requested for unsupported GatedLegendreQuadraticKAN variant")
        with torch.no_grad():
            batch = int(x.shape[0])
            base = (x - self.mu) / self.std
            pad = self.input_norm_padded_dim - self.input_dim
            xpad = F.pad(x, (0, pad)) if pad > 0 else x
            blocks = xpad.reshape(batch, -1, self.input_norm_block_size)
            z = torch.einsum("bni,nij->bnj", blocks - self.block_mu.unsqueeze(0), self.block_whiten)
            z = z.reshape(batch, self.input_norm_padded_dim)[:, : self.input_dim]
            rms = z.float().square().mean(dim=1, keepdim=True).sqrt().clamp_min(1.0e-3)
            z = z / rms * self.block_norm_target_rms
            mixed = (1.0 - self.residual_geom_mix) * base + self.residual_geom_mix * z
            leg_z = torch.tanh(mixed)
            quad_x = mixed.clamp(-3.0, 3.0)
            b1 = _legendre4(leg_z)
            h_pre = torch.einsum("bdk,dhk->bh", b1, self.leg_w1) / math.sqrt(max(1, self.input_dim))
            h = torch.tanh(h_pre)
            b2 = _legendre4(h)
            leg_logits = torch.einsum("bhk,hck->bc", b2, self.leg_w2) / math.sqrt(max(1, self.legendre_hidden))
            q_raw = quad_x @ self.quad_proj
            q = q_raw / self.quad_feature_std.clamp_min(1.0e-4) if self.quad_feature_norm_enabled else q_raw
            centered_square = q.square() - q.square().mean(dim=0, keepdim=True)
            quad_feats = torch.stack([q, centered_square], dim=2)
            quad_logits = torch.einsum("bhk,hkc->bc", quad_feats, self.quad_readout)
            leg_logits_norm = leg_logits / self.leg_logit_std.clamp_min(1.0e-4) if self.branch_output_norm_enabled else leg_logits
            quad_logits_norm = quad_logits / self.quad_logit_std.clamp_min(1.0e-4) if self.branch_output_norm_enabled else quad_logits
            direct_logits = torch.einsum("bdk,dck->bc", b1, self.direct_readout) / math.sqrt(max(1, self.input_dim))
            logits_unscaled = (
                self.branch_scale[0] * leg_logits_norm
                + self.branch_scale[1] * quad_logits_norm
                + self.bias
                + self.direct_skip_scale * direct_logits
            )
            logits = self.logit_gain.clamp(0.25, 4.0) * logits_unscaled
            cache = (
                b1,
                h,
                b2,
                quad_x,
                q,
                quad_feats,
                leg_logits_norm,
                quad_logits_norm,
                direct_logits,
                logits_unscaled,
            )
            return logits, cache

    def manual_ce_backward_from_cache(self, logits: torch.Tensor, cache: tuple[torch.Tensor, ...], y: torch.Tensor) -> torch.Tensor:
        if not self._manual_kernel_ready():
            raise RuntimeError("manual CE path requested for unsupported GatedLegendreQuadraticKAN variant")
        with torch.no_grad():
            probs = torch.softmax(logits, dim=1)
            loss = F.cross_entropy(logits, y)
            grad_output = probs
            grad_output[torch.arange(int(y.numel()), device=y.device), y] -= 1.0
            grad_output = grad_output / float(max(1, int(y.numel())))
            (
                b1,
                h,
                b2,
                quad_x,
                q,
                quad_feats,
                leg_logits_norm,
                quad_logits_norm,
                _direct_logits,
                logits_unscaled,
            ) = cache
            del _direct_logits
            gain = self.logit_gain.clamp(0.25, 4.0)
            g_logits = grad_output * gain
            gain_mask = ((self.logit_gain >= 0.25) & (self.logit_gain <= 4.0)).to(dtype=grad_output.dtype)
            self.logit_gain.grad = (grad_output * logits_unscaled).sum().view_as(self.logit_gain) * gain_mask
            self.branch_scale.grad = torch.stack(
                [
                    (g_logits * leg_logits_norm).sum(),
                    (g_logits * quad_logits_norm).sum(),
                ]
            ).view_as(self.branch_scale)
            self.bias.grad = g_logits.sum(dim=0)
            grad_direct_logits = g_logits * self.direct_skip_scale
            self.direct_readout.grad = torch.einsum("bdk,bc->dck", b1, grad_direct_logits) / math.sqrt(max(1, self.input_dim))

            grad_leg_logits = g_logits * self.branch_scale[0]
            if self.branch_output_norm_enabled:
                grad_leg_logits = grad_leg_logits / self.leg_logit_std.clamp_min(1.0e-4)
            self.leg_w2.grad = torch.einsum("bhk,bc->hck", b2, grad_leg_logits) / math.sqrt(max(1, self.legendre_hidden))
            grad_b2 = torch.einsum("bc,hck->bhk", grad_leg_logits, self.leg_w2) / math.sqrt(max(1, self.legendre_hidden))
            grad_h = (grad_b2 * _legendre4_derivative(h)).sum(dim=2)
            grad_h_pre = grad_h * (1.0 - h.square())
            self.leg_w1.grad = torch.einsum("bdk,bh->dhk", b1, grad_h_pre) / math.sqrt(max(1, self.input_dim))

            grad_quad_logits = g_logits * self.branch_scale[1]
            if self.branch_output_norm_enabled:
                grad_quad_logits = grad_quad_logits / self.quad_logit_std.clamp_min(1.0e-4)
            self.quad_readout.grad = torch.einsum("bhk,bc->hkc", quad_feats, grad_quad_logits)
            grad_quad_feats = torch.einsum("bc,hkc->bhk", grad_quad_logits, self.quad_readout)
            grad_q = grad_quad_feats[:, :, 0] + 2.0 * q * grad_quad_feats[:, :, 1]
            if self.quad_feature_norm_enabled:
                grad_q = grad_q / self.quad_feature_std.clamp_min(1.0e-4)
            self.quad_proj.grad = quad_x.transpose(0, 1) @ grad_q
            return loss.detach()

    def _legendre_logits_and_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self._norm_input(x)
        b1 = _basis_eval(z, "legendre", self.k, self.centers, self.scales)
        h = torch.einsum("bdk,dhk->bh", b1, self.leg_w1) / math.sqrt(max(1, self.input_dim))
        h = torch.tanh(h)
        b2 = _basis_eval(h, "legendre", self.k, self.centers, self.scales)
        logits = torch.einsum("bhk,hck->bc", b2, self.leg_w2) / math.sqrt(max(1, self.legendre_hidden))
        return logits, b2.reshape(int(x.shape[0]), -1)

    def _quadratic_logits_and_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self._quadratic_input(x) @ self.quad_proj
        if self.quad_feature_norm_enabled:
            z = z / self.quad_feature_std.clamp_min(1.0e-4)
        centered_square = z.square() - z.square().mean(dim=0, keepdim=True).detach()
        feats = torch.stack([z, centered_square], dim=2)
        logits = torch.einsum("bhk,hkc->bc", feats, self.quad_readout)
        return logits, feats.reshape(int(x.shape[0]), -1)

    def _direct_logits_and_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b = _basis_eval(self._norm_input(x), "legendre", self.k, self.centers, self.scales)
        logits = torch.einsum("bdk,dck->bc", b, self.direct_readout) / math.sqrt(max(1, self.input_dim))
        return logits, b.reshape(int(x.shape[0]), -1)

    def frozen_readout_features(self, x: torch.Tensor) -> torch.Tensor:
        _, leg_feats = self._legendre_logits_and_features(x)
        _, quad_feats = self._quadratic_logits_and_features(x)
        if self.direct_skip_enabled:
            _, direct_feats = self._direct_logits_and_features(x)
            return torch.cat([leg_feats, quad_feats, direct_feats], dim=1)
        return torch.cat([leg_feats, quad_feats], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._manual_kernel_ready():
            return self._forward_reuse_inputs_manual(x)
        if self.fast_reuse_forward_enabled and self.direct_skip_enabled:
            return self._forward_reuse_inputs(x)
        leg_logits, _ = self._legendre_logits_and_features(x)
        quad_logits, _ = self._quadratic_logits_and_features(x)
        if self.branch_output_norm_enabled:
            leg_logits = leg_logits / self.leg_logit_std.clamp_min(1.0e-4)
            quad_logits = quad_logits / self.quad_logit_std.clamp_min(1.0e-4)
        logits = self.branch_scale[0] * leg_logits + self.branch_scale[1] * quad_logits + self.bias
        if self.direct_skip_enabled:
            direct_logits, _ = self._direct_logits_and_features(x)
            logits = logits + self.direct_skip_scale * direct_logits
        return self.logit_gain.clamp(0.25, 4.0) * logits

    def basis_diagnostics(self, x: torch.Tensor) -> Dict[str, float]:
        with torch.no_grad():
            feats = self.frozen_readout_features(x).float()
            energy = feats.square().mean(dim=0)
            prob = energy / energy.sum().clamp_min(EPS)
            entropy = -(prob * (prob + EPS).log()).sum() / math.log(max(2, int(prob.numel())))
            dead = (energy < 1.0e-8).float().mean()
            centered = feats - feats.mean(dim=0, keepdim=True)
            try:
                s = torch.linalg.svdvals(centered[: min(256, int(centered.shape[0]))])
                rank = (s.square().sum().square() / s.pow(4).sum().clamp_min(EPS)).item()
                cond = (s.max() / s[s > 1.0e-7].min()).item() if bool((s > 1.0e-7).any()) else float("inf")
            except RuntimeError:
                rank = 0.0
                cond = float("inf")
            return {
                "basis_entropy": float(entropy.item()),
                "dead_basis_fraction": float(dead.item()),
                "basis_effective_rank": float(rank),
                "basis_condition_proxy": float(cond),
                "basis_output_norm_p95": float(torch.quantile(feats.norm(dim=1), 0.95).item()),
            }

    def basis_functional_direction(self, mode: str) -> List[torch.Tensor]:
        with torch.no_grad():
            leg1 = -self.leg_w1.detach().clone()
            leg2 = -self.leg_w2.detach().clone()
            quad_proj = torch.zeros_like(self.quad_proj)
            quad_readout = -self.quad_readout.detach().clone()
            if mode in {"basis_aware_snr_projected", "basis_aware_orthogonal"}:
                leg1[..., 0] *= 0.25
                leg2[..., 0] *= 0.25
                quad_readout[:, 0, :] *= 0.25
            out = [
                leg1,
                leg2,
                quad_proj,
                quad_readout,
                torch.zeros_like(self.branch_scale),
                torch.zeros_like(self.logit_gain),
                -self.bias.detach().clone(),
            ]
            if self.direct_skip_enabled:
                out.append(-self.direct_readout.detach().clone())
            return out


class MLPBaseline(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, seed: int, device: torch.device) -> None:
        super().__init__()
        gen = torch.Generator(device=device).manual_seed(int(seed))
        self.w0 = nn.Parameter(torch.randn(input_dim, hidden_dim, device=device, generator=gen) / math.sqrt(input_dim))
        self.w1 = nn.Parameter(torch.randn(hidden_dim, hidden_dim, device=device, generator=gen) / math.sqrt(hidden_dim))
        self.w2 = nn.Parameter(torch.randn(hidden_dim, output_dim, device=device, generator=gen) / math.sqrt(hidden_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.silu(x @ self.w0)
        h = F.silu(h @ self.w1)
        return h @ self.w2

    @property
    def hidden_dim(self) -> int:
        return int(self.w1.shape[0])


def count_parameters(module: nn.Module) -> int:
    return sum(int(p.numel()) for p in module.parameters())


def primitive_specs(param_budget: int, input_dim: int, output_dim: int) -> List[PrimitiveSpec]:
    def h(k: int) -> int:
        return matched_hidden(param_budget, input_dim, output_dim, k)

    return [
        PrimitiveSpec("B1r-ReLU-KAN-stream-K2-repair", "Activation", "relu_hinge", 2, h(2), "R1_repair_stream_basis_mix", 1, 0, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=1),
        PrimitiveSpec("B1a-ReLU-KAN-local-hinge-K4", "Activation", "relu_hinge", 4, h(4), "native+plan", 1, 0, 0, 0, 0, basis_order=1),
        PrimitiveSpec("B1b-RSWAF-hinge-K8", "Activation", "rswaf_hinge", 8, h(8), "native+plan", 1, 0, 0, 0, 0, basis_order=1),
        PrimitiveSpec("B2r-FastKAN-RBF-stream-K2-repair", "RBF", "fastkan_rbf", 2, h(2), "R1_repair_stream_basis_mix+third_party/MJKAN+v12.10_rbf_triton_l3", 1, 0, 1, 0, 0, uses_dense_basis_tensor=0, basis_order=1, init_variant="rbf_k2_triton_l3_matmul"),
        PrimitiveSpec("B2s-GaussianRBF-stream-K4-recompute", "RBF", "compact_rbf", 4, h(4), "v12.10_rbf_f2_stream_recompute_no_dense_basis+triton_l3", 1, 0, 1, 0, 0, uses_dense_basis_tensor=0, basis_order=1, init_variant="rbf_k4_triton_l3_matmul"),
        PrimitiveSpec("B2a-GaussianRBF-K4-compact", "RBF", "compact_rbf", 4, h(4), "third_party/MJKAN", 1, 0, 1, 0, 0, basis_order=1),
        PrimitiveSpec("B2b-FastKAN-RBF-K8", "RBF", "fastkan_rbf", 8, h(8), "third_party/MJKAN", 1, 0, 1, 0, 0, basis_order=1),
        PrimitiveSpec("B3a-ChebyKAN-K4", "OrthogonalPolynomial", "chebyshev", 4, h(4), "native+awesome-kan-family", 0, 1, 0, 0, 0, basis_order=4),
        PrimitiveSpec("B3c-ChebyKAN-K3-stream", "OrthogonalPolynomial", "chebyshev", 3, h(3), "v12.8.3_cheby_k3_stream_family_microbench", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=3),
        PrimitiveSpec("B3d-ChebyKAN-K6-stream", "OrthogonalPolynomial", "chebyshev", 6, h(6), "v12.8.3_cheby_k6_expression_repair", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=6),
        PrimitiveSpec("B3e-ChebyKAN-K3-tritonL3-matmulTile", "OrthogonalPolynomial", "chebyshev", 3, h(3), "v12.8.3_cheby_k3_family_specific_triton_l3", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=3, init_variant="cheby_k3_triton_l3_matmul"),
        PrimitiveSpec("B3f-ChebyKAN-K4-tritonL3-matmulTile", "OrthogonalPolynomial", "chebyshev", 4, h(4), "v12.8.3_cheby_k4_family_specific_triton_l3_expression_repair", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="cheby_k4_triton_l3_matmul"),
        PrimitiveSpec("B3j-ChebyKAN-K3-h120-tritonL3-matmulTile", "OrthogonalPolynomial", "chebyshev", 3, 120, "v12.8.3_cheby_k3_h120_capacity_bracket", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=3, init_variant="cheby_k3_triton_l3_matmul"),
        PrimitiveSpec("B3i-ChebyKAN-K3-h128-tritonL3-matmulTile", "OrthogonalPolynomial", "chebyshev", 3, 128, "v12.8.3_cheby_k3_h128_capacity_repair", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=3, init_variant="cheby_k3_triton_l3_matmul"),
        PrimitiveSpec("B3g-ChebyKAN-K3-h160-tritonL3-matmulTile", "OrthogonalPolynomial", "chebyshev", 3, 160, "v12.8.3_cheby_k3_h160_capacity_repair", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=3, init_variant="cheby_k3_triton_l3_matmul"),
        PrimitiveSpec("B3h-ChebyKAN-K3-h192-tritonL3-matmulTile", "OrthogonalPolynomial", "chebyshev", 3, 192, "v12.8.3_cheby_k3_h192_capacity_repair", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=3, init_variant="cheby_k3_triton_l3_matmul"),
        PrimitiveSpec("B3k-ChebyKAN-K3-h120-tritonL3-gradbuf", "OrthogonalPolynomial", "chebyshev", 3, 120, "v12.8.3_cheby_k3_h120_memory_planned_gradbuf", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=3, init_variant="cheby_k3_triton_l3_gradbuf"),
        PrimitiveSpec("B3l-ChebyKAN-K3-h128-tritonL3-gradbuf", "OrthogonalPolynomial", "chebyshev", 3, 128, "v12.8.3_cheby_k3_h128_memory_planned_gradbuf", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=3, init_variant="cheby_k3_triton_l3_gradbuf"),
        PrimitiveSpec("B3m-ChebyKAN-K3-h112-tritonL3-gradbuf", "OrthogonalPolynomial", "chebyshev", 3, 112, "v12.8.3_cheby_k3_h112_memory_gate_lower_bracket", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=3, init_variant="cheby_k3_triton_l3_gradbuf"),
        PrimitiveSpec("B3n-ChebyKAN-K3-h112-paircrossR16-tritonL3-gradbuf", "OrthogonalPolynomial", "chebyshev", 3, 112, "v12.8.3_cheby_k3_h112_paircrossR16_lowrank_product_expression_repair", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=3, init_variant="cheby_k3_paircrossR16_triton_l3_gradbuf"),
        PrimitiveSpec("B3o-ChebyKAN-K3-h112-paircrossR32-tritonL3-gradbuf", "OrthogonalPolynomial", "chebyshev", 3, 112, "v12.8.3_cheby_k3_h112_paircrossR32_lowrank_product_expression_repair", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=3, init_variant="cheby_k3_paircrossR32_triton_l3_gradbuf"),
        PrimitiveSpec("B3y-ChebyKAN-K3-h96-paircrossR32-tritonL3-gradbuf", "OrthogonalPolynomial", "chebyshev", 3, 96, "v12.9_cheby_k3_h96_paircrossR32_efficiency_floor", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=3, init_variant="cheby_k3_paircrossR32_triton_l3_gradbuf"),
        PrimitiveSpec("B3z-ChebyKAN-K3-h88-paircrossR32-tritonL3-gradbuf", "OrthogonalPolynomial", "chebyshev", 3, 88, "v12.9_cheby_k3_h88_paircrossR32_efficiency_floor", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=3, init_variant="cheby_k3_paircrossR32_triton_l3_gradbuf"),
        PrimitiveSpec("B3aa-ChebyKAN-K3-h88-paircrossR32-inputcrossL4P8-tritonL3-gradbuf", "OrthogonalPolynomial", "chebyshev", 3, 88, "v12.9_cheby_k3_h88_paircrossR32_inputcrossP8_expression_repair", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=3, init_variant="cheby_k3_paircrossR32_inputcross_localr4_projr8_triton_l3_gradbuf"),
        PrimitiveSpec("B3ab-ChebyKAN-K3-h88-paircrossR32-inputcrossL4P16-tritonL3-gradbuf", "OrthogonalPolynomial", "chebyshev", 3, 88, "v12.9_cheby_k3_h88_paircrossR32_inputcrossP16_expression_repair", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=3, init_variant="cheby_k3_paircrossR32_inputcross_localr4_projr16_triton_l3_gradbuf"),
        PrimitiveSpec("B3ac-ChebyKAN-K3-h80-paircrossR32-inputcrossL4P16-tritonL3-gradbuf", "OrthogonalPolynomial", "chebyshev", 3, 80, "v12.9_cheby_k3_h80_paircrossR32_inputcrossP16_efficiency_floor", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=3, init_variant="cheby_k3_paircrossR32_inputcross_localr4_projr16_triton_l3_gradbuf"),
        PrimitiveSpec("B3ad-ChebyKAN-K3-h72-paircrossR32-inputcrossL4P32-tritonL3-gradbuf", "OrthogonalPolynomial", "chebyshev", 3, 72, "v12.9_cheby_k3_h72_paircrossR32_inputcrossP32_expression_tradeoff", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=3, init_variant="cheby_k3_paircrossR32_inputcross_localr4_projr32_triton_l3_gradbuf"),
        PrimitiveSpec("B3ae-ChebyKAN-K3-h64-paircrossR32-inputcrossL4P48-tritonL3-gradbuf", "OrthogonalPolynomial", "chebyshev", 3, 64, "v12.9_cheby_k3_h64_paircrossR32_inputcrossP48_expression_bracket", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=3, init_variant="cheby_k3_paircrossR32_inputcross_localr4_projr48_triton_l3_gradbuf"),
        PrimitiveSpec("B3af-ChebyKAN-K3-h72-paircrossR32-inputcrossL4P24-tritonL3-gradbuf", "OrthogonalPolynomial", "chebyshev", 3, 72, "v12.9_cheby_k3_h72_paircrossR32_inputcrossP24_gate_edge", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=3, init_variant="cheby_k3_paircrossR32_inputcross_localr4_projr24_triton_l3_gradbuf"),
        PrimitiveSpec("B3ag-ChebyKAN-K3-h68-paircrossR32-inputcrossL4P24-tritonL3-gradbuf", "OrthogonalPolynomial", "chebyshev", 3, 68, "v12.9_cheby_k3_h68_paircrossR32_inputcrossP24_lower_hidden", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=3, init_variant="cheby_k3_paircrossR32_inputcross_localr4_projr24_triton_l3_gradbuf"),
        PrimitiveSpec("B3ah-ChebyKAN-K3-h80-paircrossR32-inputsqL4P16-tritonL3-gradbuf", "OrthogonalPolynomial", "chebyshev", 3, 80, "v12.9_cheby_k3_h80_paircrossR32_inputsqP16_random_quadratic", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=3, init_variant="cheby_k3_paircrossR32_inputcross_localr4_projr16_projsq_triton_l3_gradbuf"),
        PrimitiveSpec("B3ai-ChebyKAN-K3-h72-paircrossR32-inputsqL4P24-tritonL3-gradbuf", "OrthogonalPolynomial", "chebyshev", 3, 72, "v12.9_cheby_k3_h72_paircrossR32_inputsqP24_random_quadratic", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=3, init_variant="cheby_k3_paircrossR32_inputcross_localr4_projr24_projsq_triton_l3_gradbuf"),
        PrimitiveSpec("B3aj-ChebyKAN-K3-h72-paircrossR32-inputsqL4P8-tritonL3-gradbuf", "OrthogonalPolynomial", "chebyshev", 3, 72, "v12.9_cheby_k3_h72_paircrossR32_inputsqP8_random_quadratic_efficiency_floor", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=3, init_variant="cheby_k3_paircrossR32_inputcross_localr4_projr8_projsq_triton_l3_gradbuf"),
        PrimitiveSpec("B3ak-ChebyKAN-K3-h64-paircrossR32-inputsqL4P8-tritonL3-gradbuf", "OrthogonalPolynomial", "chebyshev", 3, 64, "v12.9_cheby_k3_h64_paircrossR32_inputsqP8_random_quadratic_lower_hidden", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=3, init_variant="cheby_k3_paircrossR32_inputcross_localr4_projr8_projsq_triton_l3_gradbuf"),
        PrimitiveSpec("B3al-ChebyKAN-K3-h72-paircrossR32-inputrot2L4P8-tritonL3-gradbuf", "OrthogonalPolynomial", "chebyshev", 3, 72, "v12.9_cheby_k3_h72_paircrossR32_inputrot2P8_rotated_pairwise", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=3, init_variant="cheby_k3_paircrossR32_inputcross_localr4_localrot2_projr8_triton_l3_gradbuf"),
        PrimitiveSpec("B3am-ChebyKAN-K3-h64-paircrossR32-inputrot2L4P8-tritonL3-gradbuf", "OrthogonalPolynomial", "chebyshev", 3, 64, "v12.9_cheby_k3_h64_paircrossR32_inputrot2P8_rotated_pairwise_lower_hidden", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=3, init_variant="cheby_k3_paircrossR32_inputcross_localr4_localrot2_projr8_triton_l3_gradbuf"),
        PrimitiveSpec("B3p-ChebyKAN-K3-h112-paircrossR32-inputcrossL4P16-tritonL3-gradbuf", "OrthogonalPolynomial", "chebyshev", 3, 112, "v12.8.3_cheby_k3_h112_paircrossR32_input_local4_randomproj16_product_expression_repair", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=3, init_variant="cheby_k3_paircrossR32_inputcross_localr4_projr16_triton_l3_gradbuf"),
        PrimitiveSpec("B3q-ChebyKAN-K3-h112-inputcrossL4P32-tritonL3-gradbuf", "OrthogonalPolynomial", "chebyshev", 3, 112, "v12.8.3_cheby_k3_h112_input_local4_randomproj32_product_expression_repair", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=3, init_variant="cheby_k3_inputcross_localr4_projr32_triton_l3_gradbuf"),
        PrimitiveSpec("B3r-ChebyKAN-K3-h112-inputcrossL4P64-tritonL3-gradbuf", "OrthogonalPolynomial", "chebyshev", 3, 112, "v12.8.3_cheby_k3_h112_input_local4_randomproj64_product_expression_repair", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=3, init_variant="cheby_k3_inputcross_localr4_projr64_triton_l3_gradbuf"),
        PrimitiveSpec("B3t-ChebyKAN-K3-h112-inputcrossL4P96-tritonL3-gradbuf", "OrthogonalPolynomial", "chebyshev", 3, 112, "v12.8.3_cheby_k3_h112_input_local4_randomproj96_product_expression_repair", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=3, init_variant="cheby_k3_inputcross_localr4_projr96_triton_l3_gradbuf"),
        PrimitiveSpec("B3u-ChebyKAN-K3-h112-inputcrossL4P112-tritonL3-gradbuf", "OrthogonalPolynomial", "chebyshev", 3, 112, "v12.8.3_cheby_k3_h112_input_local4_randomproj112_product_expression_efficiency_bracket", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=3, init_variant="cheby_k3_inputcross_localr4_projr112_triton_l3_gradbuf"),
        PrimitiveSpec("B3s-ChebyKAN-K3-h112-inputcrossL4P128-tritonL3-gradbuf", "OrthogonalPolynomial", "chebyshev", 3, 112, "v12.8.3_cheby_k3_h112_input_local4_randomproj128_product_expression_repair", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=3, init_variant="cheby_k3_inputcross_localr4_projr128_triton_l3_gradbuf"),
        PrimitiveSpec("B3v-ChebyKAN-K3-h112-inputcrossL4P128-linearres-tritonL3-gradbuf", "OrthogonalPolynomial", "chebyshev", 3, 112, "v12.8.3_cheby_k3_h112_inputcross_rank128_linear_residual_task_repair", 0, 1, 0, 1, 0, uses_dense_basis_tensor=0, basis_order=3, init_variant="cheby_k3_inputcross_localr4_projr128_triton_l3_gradbuf_linearres010"),
        PrimitiveSpec("B3w-ChebyKAN-K3-h112-inputcrossL4P128-linearres050-tritonL3-gradbuf", "OrthogonalPolynomial", "chebyshev", 3, 112, "v12.8.3_cheby_k3_h112_inputcross_rank128_stronger_linear_residual_task_repair", 0, 1, 0, 1, 0, uses_dense_basis_tensor=0, basis_order=3, init_variant="cheby_k3_inputcross_localr4_projr128_triton_l3_gradbuf_linearres050"),
        PrimitiveSpec("B3x-ChebyKAN-K3-h112-inputcrossL4P128-linearraw050-tritonL3-gradbuf", "OrthogonalPolynomial", "chebyshev", 3, 112, "v12.8.3_cheby_k3_h112_inputcross_rank128_raw_linear_residual_task_repair", 0, 1, 0, 1, 0, uses_dense_basis_tensor=0, basis_order=3, init_variant="cheby_k3_inputcross_localr4_projr128_triton_l3_gradbuf_linearres050_linearraw"),
        PrimitiveSpec("B3b-LegendreKAN-K4", "OrthogonalPolynomial", "legendre", 4, h(4), "native+awesome-kan-family", 0, 1, 0, 0, 0, basis_order=4),
        PrimitiveSpec("B12a-LegendreKAN-K4-fan-scale-repair", "OrthogonalPolynomial", "legendre", 4, h(4), "R3_repair_global_fan_scale_init", 0, 1, 0, 0, 0, basis_order=4, init_variant="fan_scale_repair"),
        PrimitiveSpec("B12b-ChebyKAN-K4-fan-scale-repair", "OrthogonalPolynomial", "chebyshev", 4, h(4), "R3_repair_global_fan_scale_init", 0, 1, 0, 0, 0, basis_order=4, init_variant="fan_scale_repair"),
        PrimitiveSpec("B13a-LegendreKAN-K4-identity-residual-repair", "OrthogonalPolynomial", "legendre", 4, h(4), "R3_repair_identity_residual_scale_init", 0, 1, 0, 0, 0, basis_order=4, init_variant="identity_residual_scale"),
        PrimitiveSpec("B13b-ChebyKAN-K4-identity-residual-repair", "OrthogonalPolynomial", "chebyshev", 4, h(4), "R3_repair_identity_residual_scale_init", 0, 1, 0, 0, 0, basis_order=4, init_variant="identity_residual_scale"),
        PrimitiveSpec("B14a-GatedLegendreQuadratic-h160", "GatedHybrid", "legendre_quadratic", 4, 160, "R5_repair_gated_legendre_quadratic_coupling", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B14b-GatedLegendreQuadratic-h224", "GatedHybrid", "legendre_quadratic", 4, 224, "R5_repair_gated_legendre_quadratic_coupling", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B14c-GatedLegendreQuadratic-h224-quad-boost", "GatedHybrid", "legendre_quadratic", 4, 224, "R5_repair_gated_legendre_quadratic_quad_boost", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="quad_boost", model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B14d-GatedLegendreQuadratic-h228-quad-boost", "GatedHybrid", "legendre_quadratic", 4, 228, "R5_repair_gated_legendre_quadratic_quad_boost_near_param", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="quad_boost", model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B15a-GatedLegendreQuadratic-h224-loss-gain", "GatedHybrid", "legendre_quadratic", 4, 224, "R5_repair_gated_legendre_quadratic_loss_gain", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="loss_gain", model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B15b-GatedLegendreQuadratic-h228-quadboost-loss-gain", "GatedHybrid", "legendre_quadratic", 4, 228, "R5_repair_gated_legendre_quadratic_quadboost_loss_gain", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="quad_boost_loss_gain", model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B16a-GatedLegendreQuadratic-h224-basis-norm", "GatedHybrid", "legendre_quadratic", 4, 224, "R5_repair_basis_specific_output_norm_init", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="basis_norm", model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B16b-GatedLegendreQuadratic-h228-quadboost-basis-norm", "GatedHybrid", "legendre_quadratic", 4, 228, "R5_repair_quadboost_basis_specific_output_norm_init", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="quad_boost_basis_norm", model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B17a-GatedLegendreQuadratic-h224-input-geom-norm", "GatedHybrid", "legendre_quadratic", 4, 224, "R5_repair_fixed_block_zca_rms_input_geometry_norm", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="geom_input_norm", model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B17b-GatedLegendreQuadratic-h228-quadboost-input-geom-norm", "GatedHybrid", "legendre_quadratic", 4, 228, "R5_repair_quadboost_fixed_block_zca_rms_input_geometry_norm", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="quad_boost_geom_input_norm", model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B17c-GatedLegendreQuadratic-h224-input-plus-branch-norm", "GatedHybrid", "legendre_quadratic", 4, 224, "R5_repair_fixed_input_geometry_norm_plus_branch_output_norm", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="geom_input_basis_norm", model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B17d-GatedLegendreQuadratic-h228-quadboost-input-plus-branch-norm", "GatedHybrid", "legendre_quadratic", 4, 228, "R5_repair_quadboost_input_geometry_norm_plus_branch_output_norm", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="quadboost_geom_input_basis_norm", model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B18a-GatedLegendreQuadratic-h224-group-rms-norm", "GatedHybrid", "legendre_quadratic", 4, 224, "R5_repair_light_diagonal_shrink_group_rms_input_norm", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="group_rms_input_norm", model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B18b-GatedLegendreQuadratic-h228-quadboost-group-rms-norm", "GatedHybrid", "legendre_quadratic", 4, 228, "R5_repair_quadboost_light_diagonal_shrink_group_rms_input_norm", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="quad_boost_group_rms_input_norm", model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B18c-GatedLegendreQuadratic-h224-group-rms-plus-branch-norm", "GatedHybrid", "legendre_quadratic", 4, 224, "R5_repair_light_group_rms_input_norm_plus_branch_output_norm", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="group_rms_input_basis_norm", model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B18d-GatedLegendreQuadratic-h228-quadboost-group-rms-plus-branch-norm", "GatedHybrid", "legendre_quadratic", 4, 228, "R5_repair_quadboost_group_rms_input_norm_plus_branch_output_norm", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="quadboost_group_rms_input_basis_norm", model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B19a-GatedLegendreQuadratic-h224-residual-geom-norm", "GatedHybrid", "legendre_quadratic", 4, 224, "R5_repair_residual_block_zca_input_geometry_norm", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="residual_geom_input_norm", model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B19b-GatedLegendreQuadratic-h228-quadboost-residual-geom-norm", "GatedHybrid", "legendre_quadratic", 4, 228, "R5_repair_quadboost_residual_block_zca_input_geometry_norm", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="quad_boost_residual_geom_input_norm", model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B19c-GatedLegendreQuadratic-h224-residual-geom-plus-branch-norm", "GatedHybrid", "legendre_quadratic", 4, 224, "R5_repair_residual_input_geometry_norm_plus_branch_output_norm", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="residual_geom_input_basis_norm", model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B19d-GatedLegendreQuadratic-h228-quadboost-residual-geom-plus-branch-norm", "GatedHybrid", "legendre_quadratic", 4, 228, "R5_repair_quadboost_residual_input_geometry_norm_plus_branch_output_norm", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="quadboost_residual_geom_input_basis_norm", model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B20a-GatedLegendreQuadratic-h224-residual-mix15", "GatedHybrid", "legendre_quadratic", 4, 224, "R5_input_norm_objective_residual_mix_grid_015", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="residual_geom_mix15", model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B20b-GatedLegendreQuadratic-h224-residual-mix25", "GatedHybrid", "legendre_quadratic", 4, 224, "R5_input_norm_objective_residual_mix_grid_025", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="residual_geom_mix25", model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B20c-GatedLegendreQuadratic-h228-quadboost-residual-mix15", "GatedHybrid", "legendre_quadratic", 4, 228, "R5_quadboost_input_norm_objective_residual_mix_grid_015", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="quad_boost_residual_geom_mix15", model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B20d-GatedLegendreQuadratic-h228-quadboost-residual-mix25", "GatedHybrid", "legendre_quadratic", 4, 228, "R5_quadboost_input_norm_objective_residual_mix_grid_025", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="quad_boost_residual_geom_mix25", model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B21a-GatedLegendreQuadratic-h224-basis-specific-quad-norm", "GatedHybrid", "legendre_quadratic", 4, 224, "R5_basis_specific_input_norm_legendre_bounded_quadratic_linear", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="basis_specific_quad_norm", model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B21b-GatedLegendreQuadratic-h228-quadboost-basis-specific-quad-norm", "GatedHybrid", "legendre_quadratic", 4, 228, "R5_quadboost_basis_specific_input_norm_legendre_bounded_quadratic_linear", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="quad_boost_basis_specific_quad_norm", model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B21c-GatedLegendreQuadratic-h224-basis-specific-residual-quad-norm", "GatedHybrid", "legendre_quadratic", 4, 224, "R5_basis_specific_input_norm_legendre_residual_quadratic_linear", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="basis_specific_residual_quad_norm", model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B21d-GatedLegendreQuadratic-h228-quadboost-basis-specific-residual-quad-norm", "GatedHybrid", "legendre_quadratic", 4, 228, "R5_quadboost_basis_specific_input_norm_legendre_residual_quadratic_linear", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="quad_boost_basis_specific_residual_quad_norm", model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B22a-GatedLegendreQuadratic-h228-basis-specific-residual-midboost", "GatedHybrid", "legendre_quadratic", 4, 228, "R5_task_repair_basis_specific_norm_mid_quadratic_branch_scale", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="basis_specific_residual_quad_midboost", model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B22b-GatedLegendreQuadratic-h228-basis-specific-residual-lowboost", "GatedHybrid", "legendre_quadratic", 4, 228, "R5_task_repair_basis_specific_norm_low_quadratic_branch_scale", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="basis_specific_residual_quad_lowboost", model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B23a-GatedLegendreQuadratic-h228-basis-specific-lowboost-temp075", "GatedHybrid", "legendre_quadratic", 4, 228, "R5_auc_ece_repair_lowboost_temperature_075", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="basis_specific_residual_quad_lowboost_temp075", model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B23b-GatedLegendreQuadratic-h228-basis-specific-lowboost-temp050", "GatedHybrid", "legendre_quadratic", 4, 228, "R5_auc_ece_repair_lowboost_temperature_050", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="basis_specific_residual_quad_lowboost_temp050", model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B26a-GatedLegendreQuadratic-h224-basis-specific-temp075", "GatedHybrid", "legendre_quadratic", 4, 224, "R3_auc_ece_repair_plain_basis_specific_temperature_075", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="basis_specific_quad_norm_temp075", model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B26b-GatedLegendreQuadratic-h224-basis-specific-temp050", "GatedHybrid", "legendre_quadratic", 4, 224, "R3_auc_ece_repair_plain_basis_specific_temperature_050", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="basis_specific_quad_norm_temp050", model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B27a-GatedLegendreQuadratic-h224-basis-specific-cosine-lr", "GatedHybrid", "legendre_quadratic", 4, 224, "R3_task_repair_basis_specific_quad_norm_warmup_cosine_lr", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="basis_specific_quad_norm", model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B27b-GatedLegendreQuadratic-h228-basis-specific-residual-cosine-lr", "GatedHybrid", "legendre_quadratic", 4, 228, "R3_task_repair_basis_specific_residual_quad_norm_warmup_cosine_lr", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="quad_boost_basis_specific_residual_quad_norm", model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B27c-GatedLegendreQuadratic-h228-lowboost-temp050-cosine-lr", "GatedHybrid", "legendre_quadratic", 4, 228, "R3_task_repair_lowboost_temp050_warmup_cosine_lr", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="basis_specific_residual_quad_lowboost_temp050", model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B28a-GatedLegendreQuadratic-h228-lowboost-temp050-cosine-lr-final050", "GatedHybrid", "legendre_quadratic", 4, 228, "R3_task_repair_lowboost_temp050_warmup10_cosine_final050", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="basis_specific_residual_quad_lowboost_temp050", model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B28b-GatedLegendreQuadratic-h228-lowboost-temp050-cosine-lr-final075", "GatedHybrid", "legendre_quadratic", 4, 228, "R3_task_repair_lowboost_temp050_warmup10_cosine_final075", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="basis_specific_residual_quad_lowboost_temp050", model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B29a-GatedLegendreQuadratic-h224-lowboost-temp050-cosine-lr-final050", "GatedHybrid", "legendre_quadratic", 4, 224, "R1_R3_repair_h224_lowboost_temp050_warmup10_cosine_final050", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="basis_specific_residual_quad_lowboost_temp050", model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B29b-GatedLegendreQuadratic-h224-lowboost-temp050-cosine-lr-final075", "GatedHybrid", "legendre_quadratic", 4, 224, "R1_R3_repair_h224_lowboost_temp050_warmup10_cosine_final075", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="basis_specific_residual_quad_lowboost_temp050", model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B30a-LiteGatedLegendreQuadratic-h192-temp050-cosine-lr-final050", "LiteGatedHybrid", "legendre_direct_quadratic_sketch", 4, 192, "R1_R3_repair_lite_direct_legendre_quadratic_basis_specific_input_norm", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="lite_basis_specific_temp050", model_kind="lite_gated_legendre_quadratic"),
        PrimitiveSpec("B30b-LiteGatedLegendreQuadratic-h256-temp050-cosine-lr-final050", "LiteGatedHybrid", "legendre_direct_quadratic_sketch", 4, 256, "R1_R3_repair_lite_direct_legendre_quadratic_basis_specific_input_norm", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="lite_basis_specific_temp050", model_kind="lite_gated_legendre_quadratic"),
        PrimitiveSpec("B30c-LiteGatedLegendreQuadratic-h296-temp075-cosine-lr-final075", "LiteGatedHybrid", "legendre_direct_quadratic_sketch", 4, 296, "R1_R3_repair_near_param_lite_direct_legendre_quadratic_basis_specific_input_norm", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="lite_basis_specific_temp075", model_kind="lite_gated_legendre_quadratic"),
        PrimitiveSpec("B31a-LiteGatedLegendreQuadratic-h224-midboost-temp100-cosine-lr-final050", "LiteGatedHybrid", "legendre_direct_quadratic_sketch", 4, 224, "R2_repair_lite_quadratic_branch_midboost_basis_specific_init", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="lite_basis_specific_midboost_temp100", model_kind="lite_gated_legendre_quadratic"),
        PrimitiveSpec("B31b-LiteGatedLegendreQuadratic-h256-quadboost-temp075-cosine-lr-final075", "LiteGatedHybrid", "legendre_direct_quadratic_sketch", 4, 256, "R2_repair_lite_quadratic_branch_quadboost_basis_specific_init", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="lite_basis_specific_quadboost_temp075", model_kind="lite_gated_legendre_quadratic"),
        PrimitiveSpec("B31c-LiteGatedLegendreQuadratic-h256-quadboost-temp100-cosine-lr-final050", "LiteGatedHybrid", "legendre_direct_quadratic_sketch", 4, 256, "R2_repair_lite_quadratic_branch_quadboost_basis_specific_init", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="lite_basis_specific_quadboost_temp100", model_kind="lite_gated_legendre_quadratic"),
        PrimitiveSpec("B32a-GatedLegendreQuadratic-h160-lowboost-temp050-cosine-lr-final050", "GatedHybrid", "legendre_quadratic", 4, 160, "R1_R2_R3_repair_smaller_gated_basis_specific_lowboost_temp050", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="basis_specific_residual_quad_lowboost_temp050", model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B32b-GatedLegendreQuadratic-h192-lowboost-temp050-cosine-lr-final050", "GatedHybrid", "legendre_quadratic", 4, 192, "R1_R2_R3_repair_smaller_gated_basis_specific_lowboost_temp050", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="basis_specific_residual_quad_lowboost_temp050", model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B32c-GatedLegendreQuadratic-h192-lowboost-temp075-cosine-lr-final075", "GatedHybrid", "legendre_quadratic", 4, 192, "R1_R2_R3_repair_smaller_gated_basis_specific_lowboost_temp075", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="basis_specific_residual_quad_lowboost_temp075", model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B33a-GatedLegendreQuadratic-h160-lowboost-temp075-cosine-lr-final075", "GatedHybrid", "legendre_quadratic", 4, 160, "R3_repair_smaller_gated_temperature_hidden_tradeoff", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="basis_specific_residual_quad_lowboost_temp075", model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B33b-GatedLegendreQuadratic-h176-lowboost-temp050-cosine-lr-final050", "GatedHybrid", "legendre_quadratic", 4, 176, "R3_repair_smaller_gated_temperature_hidden_tradeoff", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="basis_specific_residual_quad_lowboost_temp050", model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B33c-GatedLegendreQuadratic-h176-lowboost-temp075-cosine-lr-final075", "GatedHybrid", "legendre_quadratic", 4, 176, "R3_repair_smaller_gated_temperature_hidden_tradeoff", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="basis_specific_residual_quad_lowboost_temp075", model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B34a-GatedLegendreQuadratic-h168-lowboost-temp050-cosine-lr-final050", "GatedHybrid", "legendre_quadratic", 4, 168, "R3_repair_auc_task_tradeoff_hidden_168", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="basis_specific_residual_quad_lowboost_temp050", model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B34b-GatedLegendreQuadratic-h168-lowboost-temp075-cosine-lr-final050", "GatedHybrid", "legendre_quadratic", 4, 168, "R3_repair_auc_task_tradeoff_hidden_168_temp075_final050", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="basis_specific_residual_quad_lowboost_temp075", model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B34c-GatedLegendreQuadratic-h160-lowboost-temp075-cosine-lr-final050", "GatedHybrid", "legendre_quadratic", 4, 160, "R3_repair_auc_task_tradeoff_h160_temp075_final050", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="basis_specific_residual_quad_lowboost_temp075", model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B35a-GatedLegendreQuadratic-h176-lowboost-temp065-cosine-lr-final075", "GatedHybrid", "legendre_quadratic", 4, 176, "R3_repair_auc_task_tradeoff_intermediate_temperature_065", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="basis_specific_residual_quad_lowboost_temp065", model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B35b-GatedLegendreQuadratic-h176-lowboost-temp065-cosine-lr-final050", "GatedHybrid", "legendre_quadratic", 4, 176, "R3_repair_auc_task_tradeoff_intermediate_temperature_065_final050", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="basis_specific_residual_quad_lowboost_temp065", model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B35c-GatedLegendreQuadratic-h168-lowboost-temp065-cosine-lr-final075", "GatedHybrid", "legendre_quadratic", 4, 168, "R3_repair_auc_task_tradeoff_h168_intermediate_temperature_065", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="basis_specific_residual_quad_lowboost_temp065", model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B36a-GatedLegendreQuadratic-h160-lowboost-temp050-directskip-cosine-lr-final050", "GatedHybrid", "legendre_quadratic", 4, 160, "R3_repair_direct_legendre_skip_for_task_stability", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="basis_specific_residual_quad_lowboost_temp050_directskip", model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B36b-GatedLegendreQuadratic-h176-lowboost-temp065-directskip-cosine-lr-final050", "GatedHybrid", "legendre_quadratic", 4, 176, "R3_repair_direct_legendre_skip_for_task_stability_temp065", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="basis_specific_residual_quad_lowboost_temp065_directskip", model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B36c-GatedLegendreQuadratic-h176-lowboost-temp075-directskip-cosine-lr-final075", "GatedHybrid", "legendre_quadratic", 4, 176, "R3_repair_direct_legendre_skip_for_task_stability_temp075", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="basis_specific_residual_quad_lowboost_temp075_directskip", model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B37a-GatedLegendreQuadratic-h176-lowboost-temp075-directskip-branchslow-final075", "GatedHybrid", "legendre_quadratic", 4, 176, "R3_repair_branchwise_optimizer_scale_temp075", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="basis_specific_residual_quad_lowboost_temp075_directskip", model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B37b-GatedLegendreQuadratic-h176-lowboost-temp065-directskip-branchslow-final050", "GatedHybrid", "legendre_quadratic", 4, 176, "R3_repair_branchwise_optimizer_scale_temp065", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="basis_specific_residual_quad_lowboost_temp065_directskip", model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B37c-GatedLegendreQuadratic-h160-lowboost-temp050-directskip-branchslow-final050", "GatedHybrid", "legendre_quadratic", 4, 160, "R3_repair_branchwise_optimizer_scale_temp050", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="basis_specific_residual_quad_lowboost_temp050_directskip", model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B38a-GatedLegendreQuadratic-h176-lowboost-temp075-directskip-legfast-final075", "GatedHybrid", "legendre_quadratic", 4, 176, "R3_repair_branchwise_legendre_direct_fast_temp075", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="basis_specific_residual_quad_lowboost_temp075_directskip", model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B38b-GatedLegendreQuadratic-h176-lowboost-temp065-directskip-legfast-final050", "GatedHybrid", "legendre_quadratic", 4, 176, "R3_repair_branchwise_legendre_direct_fast_temp065", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="basis_specific_residual_quad_lowboost_temp065_directskip", model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B38c-GatedLegendreQuadratic-h160-lowboost-temp050-directskip-legfast-final050", "GatedHybrid", "legendre_quadratic", 4, 160, "R3_repair_branchwise_legendre_direct_fast_temp050", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="basis_specific_residual_quad_lowboost_temp050_directskip", model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B39a-GatedLegendreQuadratic-h176-lowboost-temp075-directskip-cosine-lr-final050", "GatedHybrid", "legendre_quadratic", 4, 176, "R3_repair_b36c_temp075_directskip_final050_singlepoint", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="basis_specific_residual_quad_lowboost_temp075_directskip", model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B41a-GatedLegendreQuadratic-h168-lowboost-temp075-directskip-cosine-lr-final075", "GatedHybrid", "legendre_quadratic", 4, 168, "R1_R3_repair_b36c_light_h168_temp075_directskip", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="basis_specific_residual_quad_lowboost_temp075_directskip", model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B41b-GatedLegendreQuadratic-h168-lowboost-temp065-directskip-cosine-lr-final050", "GatedHybrid", "legendre_quadratic", 4, 168, "R1_R3_repair_b36b_light_h168_temp065_directskip", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="basis_specific_residual_quad_lowboost_temp065_directskip", model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B42a-GatedLegendreQuadratic-h176-lowboost-temp075-directskip-fastreuse-final075", "GatedHybrid", "legendre_quadratic", 4, 176, "R1_repair_manual_reuse_input_norm_and_legendre_basis_temp075", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="basis_specific_residual_quad_lowboost_temp075_directskip_fastreuse", model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B42b-GatedLegendreQuadratic-h168-lowboost-temp075-directskip-fastreuse-final075", "GatedHybrid", "legendre_quadratic", 4, 168, "R1_repair_manual_reuse_input_norm_and_legendre_basis_h168_temp075", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="basis_specific_residual_quad_lowboost_temp075_directskip_fastreuse", model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B42c-GatedLegendreQuadratic-h176-lowboost-temp065-directskip-fastreuse-final050", "GatedHybrid", "legendre_quadratic", 4, 176, "R1_R3_repair_manual_reuse_input_norm_and_legendre_basis_temp065", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="basis_specific_residual_quad_lowboost_temp065_directskip_fastreuse", model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B43a-GatedLegendreQuadratic-h172-lowboost-temp075-directskip-fastreuse-final075", "GatedHybrid", "legendre_quadratic", 4, 172, "R1_R3_repair_h172_fastreuse_capacity_speed_tradeoff_temp075", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="basis_specific_residual_quad_lowboost_temp075_directskip_fastreuse", model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B43b-GatedLegendreQuadratic-h172-lowboost-temp065-directskip-fastreuse-final050", "GatedHybrid", "legendre_quadratic", 4, 172, "R1_R3_repair_h172_fastreuse_capacity_speed_tradeoff_temp065", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="basis_specific_residual_quad_lowboost_temp065_directskip_fastreuse", model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B44a-GatedLegendreQuadratic-h168-lowboost-temp075-directskip-fastreuse-mix15-final075", "GatedHybrid", "legendre_quadratic", 4, 168, "R3_repair_input_norm_residual_mix15_temp075_fastreuse", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="basis_specific_residualmix15_quad_lowboost_temp075_directskip_fastreuse", model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B44b-GatedLegendreQuadratic-h168-lowboost-temp065-directskip-fastreuse-mix15-final050", "GatedHybrid", "legendre_quadratic", 4, 168, "R3_repair_input_norm_residual_mix15_temp065_fastreuse", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="basis_specific_residualmix15_quad_lowboost_temp065_directskip_fastreuse", model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B45a-GatedLegendreQuadratic-h168-lowboost-temp075-directskip-fastreuse-final050", "GatedHybrid", "legendre_quadratic", 4, 168, "R3_repair_b42b_temp075_fastreuse_cosine_final050", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="basis_specific_residual_quad_lowboost_temp075_directskip_fastreuse", model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B46a-GatedLegendreQuadratic-h168-lowboost-temp075-directskip050-fastreuse-final075", "GatedHybrid", "legendre_quadratic", 4, 168, "R3_repair_directskip_scale050_task_geometry_temp075_final075", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="basis_specific_residual_quad_lowboost_temp075_directskip050_fastreuse", model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B46b-GatedLegendreQuadratic-h168-lowboost-temp075-directskip050-fastreuse-final050", "GatedHybrid", "legendre_quadratic", 4, 168, "R3_repair_directskip_scale050_task_geometry_temp075_final050", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="basis_specific_residual_quad_lowboost_temp075_directskip050_fastreuse", model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B47a-GatedLegendreQuadratic-h168-lowboost-temp075-directskip-fastreuse-manualbw-final075", "GatedHybrid", "legendre_quadratic", 4, 168, "R1_repair_true_manual_backward_kernel_temp075_final075", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="basis_specific_residual_quad_lowboost_temp075_directskip_fastreuse_manualbw", model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B47b-GatedLegendreQuadratic-h168-lowboost-temp075-directskip-fastreuse-manualbw-final050", "GatedHybrid", "legendre_quadratic", 4, 168, "R1_R3_repair_true_manual_backward_kernel_temp075_final050", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="basis_specific_residual_quad_lowboost_temp075_directskip_fastreuse_manualbw", model_kind="gated_legendre_quadratic"),
        PrimitiveSpec("B48a-SimpleFastTaskGeometry-h128-temp075", "SimpleFastTaskGeometry", "hinge_direct_quadratic", 2, 128, "R6_simple_fast_task_geometry_hinge_minimal_quadratic", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_quad030_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B48b-SimpleFastTaskGeometry-h192-quad050-temp075", "SimpleFastTaskGeometry", "hinge_direct_quadratic", 2, 192, "R6_simple_fast_task_geometry_hinge_mid_quadratic", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_quad050_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B48c-SimpleFastTaskGeometry-h64-quad020-temp075", "SimpleFastTaskGeometry", "hinge_direct_quadratic", 2, 64, "R6_simple_fast_task_geometry_lowrank_minimal_quadratic_efficiency_repair", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_quad020_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B48d-SimpleFastTaskGeometry-h128-temp050", "SimpleFastTaskGeometry", "hinge_direct_quadratic", 2, 128, "R3_simple_fast_task_geometry_low_temperature_task_repair", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_quad030_temp050", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B48e-SimpleFastTaskGeometry-h128-temp065", "SimpleFastTaskGeometry", "hinge_direct_quadratic", 2, 128, "R3_simple_fast_task_geometry_mid_temperature_auc_accuracy_tradeoff", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_quad030_temp065", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B49a-SimpleFastTaskGeometry-h96-sqdiag-temp075", "SimpleFastTaskGeometry", "hinge_direct_diag_quadratic", 2, 96, "R6_simple_fast_task_geometry_diag_quadratic_direct_channel", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_sqdiag_quad030_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B50a-SimpleFastTaskGeometry-h128-temp100", "SimpleFastTaskGeometry", "hinge_direct_quadratic", 2, 128, "R3_auc_nll_repair_high_logit_gain", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_quad030_temp100", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B50b-SimpleFastTaskGeometry-h96-twohinge-temp075", "SimpleFastTaskGeometry", "twohinge_direct_quadratic", 2, 96, "R6_low_cost_twohinge_task_geometry_repair", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="twohinge_quad020_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B50c-SimpleFastTaskGeometry-h128-twohinge-temp075", "SimpleFastTaskGeometry", "twohinge_direct_quadratic", 2, 128, "R2_R6_twohinge_expression_repair_with_efficiency_slack", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="twohinge_quad030_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B50d-SimpleFastTaskGeometry-h112-twohinge-temp075", "SimpleFastTaskGeometry", "twohinge_direct_quadratic", 2, 112, "R1_R2_twohinge_midpoint_expression_efficiency_tradeoff", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="twohinge_quad020_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B51a-SimpleFastTaskGeometry-h128-fixedP-temp075", "SimpleFastTaskGeometry", "hinge_direct_quadratic_fixedp", 2, 128, "FHQ0_fixed_projection_backward_cost_repair", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_quad030_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B52a-SimpleFastTaskGeometry-h128-fixedP-identitytail-temp075", "SimpleFastTaskGeometry", "hinge_direct_identitytail_quadratic_fixedp", 2, 128, "FHQ_identity_tail_direct_path_task_trajectory_repair", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_quad010_identitytail_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B54a-SimpleFastTaskGeometry-h128-learnableP-identitytail-temp075", "SimpleFastTaskGeometry", "hinge_direct_identitytail_quadratic", 2, 128, "LineP_learnable_projection_identitytail_task_trajectory_repair", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_quad010_identitytail_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B55a-SimpleFastTaskGeometry-h128-learnableP-quad050-temp075", "SimpleFastTaskGeometry", "hinge_direct_quadratic", 2, 128, "LineP_learnable_projection_quadboost_task_auc_repair", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_quad050_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B56a-SimpleFastTaskGeometry-h96-learnableP-quad050-temp075", "SimpleFastTaskGeometry", "hinge_direct_quadratic", 2, 96, "LineP_quadboost_h96_timing_stability_repair", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_quad050_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B56b-SimpleFastTaskGeometry-h112-learnableP-quad050-temp075", "SimpleFastTaskGeometry", "hinge_direct_quadratic", 2, 112, "LineP_quadboost_h112_expression_timing_midpoint", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_quad050_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B56c-SimpleFastTaskGeometry-h120-learnableP-quad050-temp075", "SimpleFastTaskGeometry", "hinge_direct_quadratic", 2, 120, "LineP_quadboost_h120_expression_timing_midpoint", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_quad050_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B57a-SimpleFastTaskGeometry-h112-learnableP-identitytailquad050-temp075", "SimpleFastTaskGeometry", "hinge_direct_identitytail_quadratic", 2, 112, "LineP_identitytail_quad050_h112_expression_repair_with_timing_slack", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_identitytailquad050_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B57b-SimpleFastTaskGeometry-h116-learnableP-identitytailquad050-temp075", "SimpleFastTaskGeometry", "hinge_direct_identitytail_quadratic", 2, 116, "LineP_identitytail_quad050_h116_A4_timing_midpoint", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_identitytailquad050_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B57c-SimpleFastTaskGeometry-h118-learnableP-identitytailquad050-temp075", "SimpleFastTaskGeometry", "hinge_direct_identitytail_quadratic", 2, 118, "LineP_identitytail_quad050_h118_A4_timing_midpoint", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_identitytailquad050_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B58a-SimpleFastTaskGeometry-h128-fixedP-identitytailquad050-temp075", "SimpleFastTaskGeometry", "hinge_direct_identitytail_quadratic_fixedp", 2, 128, "LineP_fixedP_identitytail_quad050_task_trajectory_repair", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_identitytailquad050_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B59a-SimpleFastTaskGeometry-h116-learnableP-identitytailquad060-temp075", "SimpleFastTaskGeometry", "hinge_direct_identitytail_quadratic", 2, 116, "LineP_identitytail_quad060_h116_early_interaction_trajectory_repair", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_identitytailquad060_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B59b-SimpleFastTaskGeometry-h116-learnableP-identitytailquad040-temp075", "SimpleFastTaskGeometry", "hinge_direct_identitytail_quadratic", 2, 116, "LineP_identitytail_quad040_h116_early_interaction_trajectory_repair", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_identitytailquad040_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B60a-SimpleFastTaskGeometry-h116-learnableP-hinge050-identitytailquad050-temp075", "SimpleFastTaskGeometry", "hinge050_direct_identitytail_quadratic", 2, 116, "LineP_hinge050_identitytail_quad050_h116_basis_shape_trajectory_repair", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge050_identitytailquad050_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B60b-SimpleFastTaskGeometry-h112-learnableP-hinge050-identitytailquad050-temp075", "SimpleFastTaskGeometry", "hinge050_direct_identitytail_quadratic", 2, 112, "LineP_hinge050_identitytail_quad050_h112_timing_midpoint", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge050_identitytailquad050_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B61a-SimpleFastTaskGeometry-h112-twohinge-identitytailquad050-temp075", "SimpleFastTaskGeometry", "twohinge_direct_identitytail_quadratic", 2, 112, "LineP_twohinge_identitytail_quad050_h112_shape_and_trajectory_repair", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="twohinge_identitytailquad050_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B62a-SimpleFastTaskGeometry-h128-fixedP-signedpairlitequad050-temp075", "SimpleFastTaskGeometry", "hinge_direct_signedpairlite_quadratic_fixedp", 2, 128, "LineP_signedpairlite_fixed_projection_frame_trajectory_repair", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_signedpairlite_quad050_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B62b-SimpleFastTaskGeometry-h128-learnableP-signedpairlitequad050-temp075", "SimpleFastTaskGeometry", "hinge_direct_signedpairlite_quadratic", 2, 128, "LineP_signedpairlite_learnable_projection_init_trajectory_repair", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_signedpairlite_quad050_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B63a-SimpleFastTaskGeometry-h128-fixedP-signedpairtailquad050-temp075", "SimpleFastTaskGeometry", "hinge_direct_signedpairtail_quadratic_fixedp", 2, 128, "LineP_signedpairlite_random_tail_projection_repair_fixedP", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_signedpairtail_quad050_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B63b-SimpleFastTaskGeometry-h128-learnableP-signedpairtailquad050-temp075", "SimpleFastTaskGeometry", "hinge_direct_signedpairtail_quadratic", 2, 128, "LineP_signedpairlite_random_tail_projection_repair_learnableP", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_signedpairtail_quad050_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B64a-SimpleFastTaskGeometry-h128-fixedP-signedpairtail-boundq-temp075", "SimpleFastTaskGeometry", "hinge_direct_signedpairtail_bounded_quadratic_fixedp", 2, 128, "LineP_signedpairtail_bounded_q_explosion_repair_fixedP", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_signedpairtail_quad050_boundq_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B64b-SimpleFastTaskGeometry-h128-learnableP-signedpairtail-boundq-temp075", "SimpleFastTaskGeometry", "hinge_direct_signedpairtail_bounded_quadratic", 2, 128, "LineP_signedpairtail_bounded_q_explosion_repair_learnableP", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_signedpairtail_quad050_boundq_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B65a-SimpleFastTaskGeometry-h128-fixedP-rmsq-quad050-temp075", "SimpleFastTaskGeometry", "hinge_direct_rmsq_quadratic_fixedp", 2, 128, "LineP_stopgrad_batch_rms_quadratic_trajectory_repair_fixedP", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_rmsq_quad050_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B65b-SimpleFastTaskGeometry-h128-learnableP-rmsq-quad050-temp075", "SimpleFastTaskGeometry", "hinge_direct_rmsq_quadratic", 2, 128, "LineP_stopgrad_batch_rms_quadratic_trajectory_repair_learnableP", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_rmsq_quad050_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B66a-SimpleFastTaskGeometry-h128-fixedP-orthop-quad050-temp075", "SimpleFastTaskGeometry", "hinge_direct_orthogonal_projection_quadratic_fixedp", 2, 128, "LineP_low_coherence_orthogonal_projection_frame_fixedP", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_orthop_quad050_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B66b-SimpleFastTaskGeometry-h128-learnableP-orthop-quad050-temp075", "SimpleFastTaskGeometry", "hinge_direct_orthogonal_projection_quadratic", 2, 128, "LineP_low_coherence_orthogonal_projection_frame_learnableP", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_orthop_quad050_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B67a-SimpleFastTaskGeometry-h112-fixedP-orthop-quad050-temp075", "SimpleFastTaskGeometry", "hinge_direct_orthogonal_projection_quadratic_fixedp", 2, 112, "LineP_B66_orthogonal_projection_efficiency_repair_fixedP_h112", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_orthop_quad050_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B67b-SimpleFastTaskGeometry-h112-learnableP-orthop-quad050-temp075", "SimpleFastTaskGeometry", "hinge_direct_orthogonal_projection_quadratic", 2, 112, "LineP_B66_orthogonal_projection_efficiency_repair_learnableP_h112", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_orthop_quad050_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B68a-SimpleFastTaskGeometry-h128-fixedP-classbranch-quad050-temp075", "SimpleFastTaskGeometry", "hinge_direct_classwise_branch_quadratic_fixedp", 2, 128, "LineP_classwise_direct_quad_branch_scale_fixedP_tail_repair", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_classbranch_quad050_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B68b-SimpleFastTaskGeometry-h128-learnableP-classbranch-quad050-temp075", "SimpleFastTaskGeometry", "hinge_direct_classwise_branch_quadratic", 2, 128, "LineP_classwise_direct_quad_branch_scale_learnableP_tail_repair", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_classbranch_quad050_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B69a-SimpleFastTaskGeometry-h120-fixedP-classbranch-quad050-temp075", "SimpleFastTaskGeometry", "hinge_direct_classwise_branch_quadratic_fixedp", 2, 120, "LineP_B68_classbranch_timing_repair_fixedP_h120", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_classbranch_quad050_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B69b-SimpleFastTaskGeometry-h120-learnableP-classbranch-quad050-temp075", "SimpleFastTaskGeometry", "hinge_direct_classwise_branch_quadratic", 2, 120, "LineP_B68_classbranch_timing_repair_learnableP_h120", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_classbranch_quad050_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B69c-SimpleFastTaskGeometry-h112-fixedP-classbranch-quad050-temp075", "SimpleFastTaskGeometry", "hinge_direct_classwise_branch_quadratic_fixedp", 2, 112, "LineP_B68_classbranch_timing_repair_fixedP_h112", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_classbranch_quad050_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B69d-SimpleFastTaskGeometry-h112-learnableP-classbranch-quad050-temp075", "SimpleFastTaskGeometry", "hinge_direct_classwise_branch_quadratic", 2, 112, "LineP_B68_classbranch_timing_repair_learnableP_h112", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_classbranch_quad050_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B70a-SimpleFastTaskGeometry-h124-fixedP-classbranch-quad050-temp075", "SimpleFastTaskGeometry", "hinge_direct_classwise_branch_quadratic_fixedp", 2, 124, "LineP_B69_classbranch_h124_accuracy_timing_midpoint_fixedP", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_classbranch_quad050_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B70b-SimpleFastTaskGeometry-h124-learnableP-classbranch-quad050-temp075", "SimpleFastTaskGeometry", "hinge_direct_classwise_branch_quadratic", 2, 124, "LineP_B69_classbranch_h124_accuracy_timing_midpoint_learnableP", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_classbranch_quad050_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B71a-SimpleFastTaskGeometry-h126-fixedP-classbranch-quad050-temp075", "SimpleFastTaskGeometry", "hinge_direct_classwise_branch_quadratic_fixedp", 2, 126, "LineP_B70_classbranch_h126_accuracy_timing_midpoint_fixedP", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_classbranch_quad050_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B71b-SimpleFastTaskGeometry-h126-learnableP-classbranch-quad050-temp075", "SimpleFastTaskGeometry", "hinge_direct_classwise_branch_quadratic", 2, 126, "LineP_B70_classbranch_h126_accuracy_timing_midpoint_learnableP", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_classbranch_quad050_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B72a-SimpleFastTaskGeometry-h126-fixedP-classbranch-quad040-temp075", "SimpleFastTaskGeometry", "hinge_direct_classwise_branch_quadratic_fixedp", 2, 126, "LineP_B71_classbranch_quad040_auc_tail_repair_fixedP", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_classbranch_quad040_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B72b-SimpleFastTaskGeometry-h126-learnableP-classbranch-quad040-temp075", "SimpleFastTaskGeometry", "hinge_direct_classwise_branch_quadratic", 2, 126, "LineP_B71_classbranch_quad040_auc_tail_repair_learnableP", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_classbranch_quad040_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B73a-SimpleFastTaskGeometry-h112-fixedP-sqdiag-classbranch-quad050-temp075", "SimpleFastTaskGeometry", "hinge_direct_sqdiag_classwise_branch_quadratic_fixedp", 2, 112, "LineP_B71_sqdiag_direct_energy_classbranch_fixedP_h112", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_sqdiag_classbranch_quad050_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B73b-SimpleFastTaskGeometry-h112-learnableP-sqdiag-classbranch-quad050-temp075", "SimpleFastTaskGeometry", "hinge_direct_sqdiag_classwise_branch_quadratic", 2, 112, "LineP_B71_sqdiag_direct_energy_classbranch_learnableP_h112", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_sqdiag_classbranch_quad050_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B73c-SimpleFastTaskGeometry-h96-fixedP-sqdiag-classbranch-quad050-temp075", "SimpleFastTaskGeometry", "hinge_direct_sqdiag_classwise_branch_quadratic_fixedp", 2, 96, "LineP_B73_sqdiag_direct_energy_timing_repair_fixedP_h96", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_sqdiag_classbranch_quad050_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B73d-SimpleFastTaskGeometry-h96-learnableP-sqdiag-classbranch-quad050-temp075", "SimpleFastTaskGeometry", "hinge_direct_sqdiag_classwise_branch_quadratic", 2, 96, "LineP_B73_sqdiag_direct_energy_timing_repair_learnableP_h96", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_sqdiag_classbranch_quad050_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B74a-SimpleFastTaskGeometry-h116-fixedP-sqdiag-classbranch-quad050-temp075", "SimpleFastTaskGeometry", "hinge_direct_sqdiag_classwise_branch_quadratic_fixedp", 2, 116, "LineP_B73_sqdiag_expression_capacity_repair_fixedP_h116", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_sqdiag_classbranch_quad050_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B74b-SimpleFastTaskGeometry-h116-learnableP-sqdiag-classbranch-quad050-temp075", "SimpleFastTaskGeometry", "hinge_direct_sqdiag_classwise_branch_quadratic", 2, 116, "LineP_B73_sqdiag_expression_capacity_repair_learnableP_h116", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_sqdiag_classbranch_quad050_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B74c-SimpleFastTaskGeometry-h120-fixedP-sqdiag-classbranch-quad050-temp075", "SimpleFastTaskGeometry", "hinge_direct_sqdiag_classwise_branch_quadratic_fixedp", 2, 120, "LineP_B74_sqdiag_expression_capacity_repair_fixedP_h120", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_sqdiag_classbranch_quad050_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B74d-SimpleFastTaskGeometry-h120-learnableP-sqdiag-classbranch-quad050-temp075", "SimpleFastTaskGeometry", "hinge_direct_sqdiag_classwise_branch_quadratic", 2, 120, "LineP_B74_sqdiag_expression_capacity_repair_learnableP_h120", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_sqdiag_classbranch_quad050_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B75a-SimpleFastTaskGeometry-h112-fixedP-sqdiag025-classbranch-quad050-temp075", "SimpleFastTaskGeometry", "hinge_direct_sqdiag025_classwise_branch_quadratic_fixedp", 2, 112, "LineP_B73_low_init_sqdiag_energy_fixedP_h112", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_sqdiag025_classbranch_quad050_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B75b-SimpleFastTaskGeometry-h112-learnableP-sqdiag025-classbranch-quad050-temp075", "SimpleFastTaskGeometry", "hinge_direct_sqdiag025_classwise_branch_quadratic", 2, 112, "LineP_B73_low_init_sqdiag_energy_learnableP_h112", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_sqdiag025_classbranch_quad050_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B75c-SimpleFastTaskGeometry-h112-fixedP-sqdiag010-classbranch-quad050-temp075", "SimpleFastTaskGeometry", "hinge_direct_sqdiag010_classwise_branch_quadratic_fixedp", 2, 112, "LineP_B75_lower_init_sqdiag_energy_fixedP_h112", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_sqdiag010_classbranch_quad050_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B75d-SimpleFastTaskGeometry-h112-learnableP-sqdiag010-classbranch-quad050-temp075", "SimpleFastTaskGeometry", "hinge_direct_sqdiag010_classwise_branch_quadratic", 2, 112, "LineP_B75_lower_init_sqdiag_energy_learnableP_h112", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_sqdiag010_classbranch_quad050_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B76a-SimpleFastTaskGeometry-h126-fixedP-classbranch-identitytailquad040-temp075", "SimpleFastTaskGeometry", "hinge_direct_classwise_identitytail_quadratic_fixedp", 2, 126, "LineP_B71_classbranch_identitytail_direct_boost_fixedP_h126", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_classbranch_identitytailquad040_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B76b-SimpleFastTaskGeometry-h126-learnableP-classbranch-identitytailquad040-temp075", "SimpleFastTaskGeometry", "hinge_direct_classwise_identitytail_quadratic", 2, 126, "LineP_B71_classbranch_identitytail_direct_boost_learnableP_h126", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_classbranch_identitytailquad040_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B76c-SimpleFastTaskGeometry-h126-fixedP-classbranch-identitytailquad050-temp075", "SimpleFastTaskGeometry", "hinge_direct_classwise_identitytail_quadratic_fixedp", 2, 126, "LineP_B76_classbranch_identitytail_quad050_fixedP_h126", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_classbranch_identitytailquad050_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B76d-SimpleFastTaskGeometry-h126-learnableP-classbranch-identitytailquad050-temp075", "SimpleFastTaskGeometry", "hinge_direct_classwise_identitytail_quadratic", 2, 126, "LineP_B76_classbranch_identitytail_quad050_learnableP_h126", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_classbranch_identitytailquad050_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B77a-SimpleFastTaskGeometry-h126-fixedP-classbranch-identitytailquad030-temp075", "SimpleFastTaskGeometry", "hinge_direct_classwise_identitytail_quadratic_fixedp", 2, 126, "LineP_B76_classbranch_identitytail_quad030_fixedP_h126", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_classbranch_identitytailquad030_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B77b-SimpleFastTaskGeometry-h126-learnableP-classbranch-identitytailquad030-temp075", "SimpleFastTaskGeometry", "hinge_direct_classwise_identitytail_quadratic", 2, 126, "LineP_B76_classbranch_identitytail_quad030_learnableP_h126", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_classbranch_identitytailquad030_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B77c-SimpleFastTaskGeometry-h126-fixedP-classbranch-identitytailquad020-temp075", "SimpleFastTaskGeometry", "hinge_direct_classwise_identitytail_quadratic_fixedp", 2, 126, "LineP_B77_classbranch_identitytail_quad020_fixedP_h126", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_classbranch_identitytailquad020_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B77d-SimpleFastTaskGeometry-h126-learnableP-classbranch-identitytailquad020-temp075", "SimpleFastTaskGeometry", "hinge_direct_classwise_identitytail_quadratic", 2, 126, "LineP_B77_classbranch_identitytail_quad020_learnableP_h126", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_classbranch_identitytailquad020_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B78a-SimpleFastTaskGeometry-h126-fixedP-classbranch-identitytailquad040-hingeamp075-temp075", "SimpleFastTaskGeometry", "hingeamp075_direct_classwise_identitytail_quadratic_fixedp", 2, 126, "LineP_B76_hinge_tail_amplitude_repair_fixedP_h126", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_classbranch_identitytailquad040_hingeamp075_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B78b-SimpleFastTaskGeometry-h126-learnableP-classbranch-identitytailquad040-hingeamp075-temp075", "SimpleFastTaskGeometry", "hingeamp075_direct_classwise_identitytail_quadratic", 2, 126, "LineP_B76_hinge_tail_amplitude_repair_learnableP_h126", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_classbranch_identitytailquad040_hingeamp075_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B78c-SimpleFastTaskGeometry-h126-fixedP-classbranch-identitytailquad040-hingeamp050-temp075", "SimpleFastTaskGeometry", "hingeamp050_direct_classwise_identitytail_quadratic_fixedp", 2, 126, "LineP_B78_stronger_hinge_tail_amplitude_fixedP_h126", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_classbranch_identitytailquad040_hingeamp050_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B78d-SimpleFastTaskGeometry-h126-learnableP-classbranch-identitytailquad040-hingeamp050-temp075", "SimpleFastTaskGeometry", "hingeamp050_direct_classwise_identitytail_quadratic", 2, 126, "LineP_B78_stronger_hinge_tail_amplitude_learnableP_h126", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_classbranch_identitytailquad040_hingeamp050_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B79a-SimpleFastTaskGeometry-h126-fixedP-classbranch-identitytailquad040-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_classwise_identitytail_quadratic_fixedp", 2, 126, "LineP_B78_aggressive_hinge_tail_amplitude_fixedP_h126", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_classbranch_identitytailquad040_hingeamp025_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B79b-SimpleFastTaskGeometry-h126-learnableP-classbranch-identitytailquad040-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_classwise_identitytail_quadratic", 2, 126, "LineP_B78_aggressive_hinge_tail_amplitude_learnableP_h126", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_classbranch_identitytailquad040_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B80a-SimpleFastTaskGeometry-h126-fixedP-classbranch-identitytailquad040-hingeamp025-boundq-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_classwise_identitytail_bounded_quadratic_fixedp", 2, 126, "LineP_B79_bounded_quadratic_tail_fixedP_h126", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_classbranch_identitytailquad040_hingeamp025_boundq_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B80b-SimpleFastTaskGeometry-h126-learnableP-classbranch-identitytailquad040-hingeamp025-boundq-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_classwise_identitytail_bounded_quadratic", 2, 126, "LineP_B79_bounded_quadratic_tail_learnableP_h126", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_classbranch_identitytailquad040_hingeamp025_boundq_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B81a-SimpleFastTaskGeometry-h126-fixedP-classbranch-identitytail150quad040-hingeamp025-temp075", "SimpleFastTaskGeometry", "direct150_hingeamp025_classwise_identitytail_quadratic_fixedp", 2, 126, "LineP_B79_direct_branch_trajectory_boost_fixedP_h126", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_classbranch_identitytail150quad040_hingeamp025_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B81b-SimpleFastTaskGeometry-h126-learnableP-classbranch-identitytail150quad040-hingeamp025-temp075", "SimpleFastTaskGeometry", "direct150_hingeamp025_classwise_identitytail_quadratic", 2, 126, "LineP_B79_direct_branch_trajectory_boost_learnableP_h126", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_classbranch_identitytail150quad040_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B82a-SimpleFastTaskGeometry-h126-fixedP-classbranch-identityamp150-identitytailquad040-hingeamp025-temp075", "SimpleFastTaskGeometry", "raw_identity150_hingeamp025_classwise_identitytail_quadratic_fixedp", 2, 126, "LineP_B79_raw_identity_only_trajectory_boost_fixedP_h126", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_classbranch_identitytailquad040_identityamp150_hingeamp025_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B82b-SimpleFastTaskGeometry-h126-learnableP-classbranch-identityamp150-identitytailquad040-hingeamp025-temp075", "SimpleFastTaskGeometry", "raw_identity150_hingeamp025_classwise_identitytail_quadratic", 2, 126, "LineP_B79_raw_identity_only_trajectory_boost_learnableP_h126", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_classbranch_identitytailquad040_identityamp150_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B83a-SimpleFastTaskGeometry-h126-fixedP-absdiag025-classbranch-identitytailquad040-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag025_classwise_identitytail_quadratic_fixedp", 2, 126, "LineP_B79_centered_abs_magnitude_tail_fixedP_h126", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag025_classbranch_identitytailquad040_hingeamp025_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B83b-SimpleFastTaskGeometry-h126-learnableP-absdiag025-classbranch-identitytailquad040-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag025_classwise_identitytail_quadratic", 2, 126, "LineP_B79_centered_abs_magnitude_tail_learnableP_h126", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag025_classbranch_identitytailquad040_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B84a-SimpleFastTaskGeometry-h126-fixedP-absdiag010-classbranch-identitytailquad040-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag010_classwise_identitytail_quadratic_fixedp", 2, 126, "LineP_B83_lower_abs_magnitude_tail_fixedP_h126", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag010_classbranch_identitytailquad040_hingeamp025_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B84b-SimpleFastTaskGeometry-h126-learnableP-absdiag010-classbranch-identitytailquad040-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag010_classwise_identitytail_quadratic", 2, 126, "LineP_B83_lower_abs_magnitude_tail_learnableP_h126", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag010_classbranch_identitytailquad040_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B85a-SimpleFastTaskGeometry-h126-fixedP-absdiag050-classbranch-identitytailquad040-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_classwise_identitytail_quadratic_fixedp", 2, 126, "LineP_B83_stronger_abs_magnitude_tail_fixedP_h126", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_classbranch_identitytailquad040_hingeamp025_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B85b-SimpleFastTaskGeometry-h126-learnableP-absdiag050-classbranch-identitytailquad040-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_classwise_identitytail_quadratic", 2, 126, "LineP_B83_stronger_abs_magnitude_tail_learnableP_h126", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_classbranch_identitytailquad040_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B86a-SimpleFastTaskGeometry-h126-fixedP-absdiag100-classbranch-identitytailquad040-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag100_classwise_identitytail_quadratic_fixedp", 2, 126, "LineP_B85_full_abs_magnitude_tail_fixedP_h126", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag100_classbranch_identitytailquad040_hingeamp025_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B86b-SimpleFastTaskGeometry-h126-learnableP-absdiag100-classbranch-identitytailquad040-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag100_classwise_identitytail_quadratic", 2, 126, "LineP_B85_full_abs_magnitude_tail_learnableP_h126", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag100_classbranch_identitytailquad040_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B87a-SimpleFastTaskGeometry-h126-fixedP-absdiag075-classbranch-identitytailquad040-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag075_classwise_identitytail_quadratic_fixedp", 2, 126, "LineP_B85_B86_abs_magnitude_midpoint_fixedP_h126", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag075_classbranch_identitytailquad040_hingeamp025_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B87b-SimpleFastTaskGeometry-h126-learnableP-absdiag075-classbranch-identitytailquad040-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag075_classwise_identitytail_quadratic", 2, 126, "LineP_B85_B86_abs_magnitude_midpoint_learnableP_h126", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag075_classbranch_identitytailquad040_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B88a-SimpleFastTaskGeometry-h126-fixedP-absdiag100-classbranch-identitytailquad050-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag100_classwise_identitytail_quad050_fixedp", 2, 126, "LineP_B86_abs_magnitude_with_stronger_quad_tail_fixedP_h126", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag100_classbranch_identitytailquad050_hingeamp025_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B88b-SimpleFastTaskGeometry-h126-learnableP-absdiag100-classbranch-identitytailquad050-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag100_classwise_identitytail_quad050", 2, 126, "LineP_B86_abs_magnitude_with_stronger_quad_tail_learnableP_h126", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag100_classbranch_identitytailquad050_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B89a-SimpleFastTaskGeometry-h126-fixedP-absdiag050-classbranch-identitytailquad050-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_classwise_identitytail_quad050_fixedp", 2, 126, "LineP_B85_abs_magnitude_quad050_midpoint_fixedP_h126", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_classbranch_identitytailquad050_hingeamp025_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B89b-SimpleFastTaskGeometry-h126-learnableP-absdiag050-classbranch-identitytailquad050-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_classwise_identitytail_quad050", 2, 126, "LineP_B85_abs_magnitude_quad050_midpoint_learnableP_h126", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_classbranch_identitytailquad050_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B90a-SimpleFastTaskGeometry-h126-fixedP-absdiag100-classbranch-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag100_classwise_identitytail_quad030_fixedp", 2, 126, "LineP_B86_full_abs_magnitude_lower_quad_tail_fixedP_h126", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag100_classbranch_identitytailquad030_hingeamp025_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B90b-SimpleFastTaskGeometry-h126-learnableP-absdiag100-classbranch-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag100_classwise_identitytail_quad030", 2, 126, "LineP_B86_full_abs_magnitude_lower_quad_tail_learnableP_h126", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag100_classbranch_identitytailquad030_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B91a-SimpleFastTaskGeometry-h126-fixedP-absdiag050-classbranch-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_classwise_identitytail_quad030_fixedp", 2, 126, "LineP_B85_mid_abs_magnitude_lower_quad_tail_fixedP_h126", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_classbranch_identitytailquad030_hingeamp025_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B91b-SimpleFastTaskGeometry-h126-learnableP-absdiag050-classbranch-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_classwise_identitytail_quad030", 2, 126, "LineP_B85_mid_abs_magnitude_lower_quad_tail_learnableP_h126", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_classbranch_identitytailquad030_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B92a-SimpleFastTaskGeometry-h126-fixedP-normabs050-classbranch-identitytailquad040-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_normabs050_classwise_identitytail_quadratic_fixedp", 2, 126, "LineP_B85_pooled_abs_magnitude_scalar_fixedP_h126", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_normabs050_classbranch_identitytailquad040_hingeamp025_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B92b-SimpleFastTaskGeometry-h126-learnableP-normabs050-classbranch-identitytailquad040-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_normabs050_classwise_identitytail_quadratic", 2, 126, "LineP_B85_pooled_abs_magnitude_scalar_learnableP_h126", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_normabs050_classbranch_identitytailquad040_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B93a-SimpleFastTaskGeometry-h126-fixedP-normabs100-classbranch-identitytailquad040-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_normabs100_classwise_identitytail_quadratic_fixedp", 2, 126, "LineP_B92_stronger_pooled_abs_scalar_fixedP_h126", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_normabs100_classbranch_identitytailquad040_hingeamp025_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B93b-SimpleFastTaskGeometry-h126-learnableP-normabs100-classbranch-identitytailquad040-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_normabs100_classwise_identitytail_quadratic", 2, 126, "LineP_B92_stronger_pooled_abs_scalar_learnableP_h126", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_normabs100_classbranch_identitytailquad040_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B94a-SimpleFastTaskGeometry-h126-fixedP-absdiag050-normabs010-classbranch-identitytailquad040-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_normabs010_classwise_identitytail_quadratic_fixedp", 2, 126, "LineP_B85_weak_pooled_abs_scalar_on_absdiag_fixedP_h126", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_normabs010_classbranch_identitytailquad040_hingeamp025_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B94b-SimpleFastTaskGeometry-h126-learnableP-absdiag050-normabs010-classbranch-identitytailquad040-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_normabs010_classwise_identitytail_quadratic", 2, 126, "LineP_B85_weak_pooled_abs_scalar_on_absdiag_learnableP_h126", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_normabs010_classbranch_identitytailquad040_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B95a-SimpleFastTaskGeometry-h126-fixedP-absdiag050-normabs025-classbranch-identitytailquad040-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_normabs025_classwise_identitytail_quadratic_fixedp", 2, 126, "LineP_B94_mid_pooled_abs_scalar_on_absdiag_fixedP_h126", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_normabs025_classbranch_identitytailquad040_hingeamp025_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B95b-SimpleFastTaskGeometry-h126-learnableP-absdiag050-normabs025-classbranch-identitytailquad040-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_normabs025_classwise_identitytail_quadratic", 2, 126, "LineP_B94_mid_pooled_abs_scalar_on_absdiag_learnableP_h126", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_normabs025_classbranch_identitytailquad040_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B96a-SimpleFastTaskGeometry-h126-fixedP-absdiag050-classbranch-identitytail140quad040-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_classwise_identitytail140_quadratic_fixedp", 2, 126, "LineP_B85_mild_direct_branch_boost_fixedP_h126", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_classbranch_identitytail140quad040_hingeamp025_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B96b-SimpleFastTaskGeometry-h126-learnableP-absdiag050-classbranch-identitytail140quad040-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_classwise_identitytail140_quadratic", 2, 126, "LineP_B85_mild_direct_branch_boost_learnableP_h126", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_classbranch_identitytail140quad040_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B97a-SimpleFastTaskGeometry-h126-fixedP-absdiag050-classbranch-identitytail145quad040-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_classwise_identitytail145_quadratic_fixedp", 2, 126, "LineP_B96_mid_direct_branch_boost_fixedP_h126", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_classbranch_identitytail145quad040_hingeamp025_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B97b-SimpleFastTaskGeometry-h126-learnableP-absdiag050-classbranch-identitytail145quad040-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_classwise_identitytail145_quadratic", 2, 126, "LineP_B96_mid_direct_branch_boost_learnableP_h126", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_classbranch_identitytail145quad040_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B98a-SimpleFastTaskGeometry-h126-fixedP-absdiag050-classbranch-identitytail130quad040-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_classwise_identitytail130_quadratic_fixedp", 2, 126, "LineP_B85_lower_direct_branch_fixedP_h126", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_classbranch_identitytail130quad040_hingeamp025_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B98b-SimpleFastTaskGeometry-h126-learnableP-absdiag050-classbranch-identitytail130quad040-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_classwise_identitytail130_quadratic", 2, 126, "LineP_B85_lower_direct_branch_learnableP_h126", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_classbranch_identitytail130quad040_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B99a-SimpleFastTaskGeometry-h126-fixedP-absdiag050-classbranch-identitytail125quad040-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_classwise_identitytail125_quadratic_fixedp", 2, 126, "LineP_B98_lower_direct_branch_fixedP_h126", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_classbranch_identitytail125quad040_hingeamp025_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B99b-SimpleFastTaskGeometry-h126-learnableP-absdiag050-classbranch-identitytail125quad040-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_classwise_identitytail125_quadratic", 2, 126, "LineP_B98_lower_direct_branch_learnableP_h126", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_classbranch_identitytail125quad040_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B100a-SimpleFastTaskGeometry-h126-fixedP-absdiag050-classbranch-identitytailquad040-hingeamp050-temp075", "SimpleFastTaskGeometry", "hingeamp050_direct_absdiag050_classwise_identitytail_quadratic_fixedp", 2, 126, "LineP_B85_stronger_hinge_tail_fixedP_h126", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_classbranch_identitytailquad040_hingeamp050_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B100b-SimpleFastTaskGeometry-h126-learnableP-absdiag050-classbranch-identitytailquad040-hingeamp050-temp075", "SimpleFastTaskGeometry", "hingeamp050_direct_absdiag050_classwise_identitytail_quadratic", 2, 126, "LineP_B85_stronger_hinge_tail_learnableP_h126", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_classbranch_identitytailquad040_hingeamp050_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B101a-SimpleFastTaskGeometry-h126-fixedP-absdiag050-signedpairlite-classbranch-identitytailquad040-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_signedpairlite_classwise_identitytail_quadratic_fixedp", 2, 126, "LineP_B85_structured_projection_trajectory_repair_fixedP_h126", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_signedpairlite_classbranch_identitytailquad040_hingeamp025_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B101b-SimpleFastTaskGeometry-h126-learnableP-absdiag050-signedpairlite-classbranch-identitytailquad040-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_signedpairlite_classwise_identitytail_quadratic", 2, 126, "LineP_B85_structured_projection_trajectory_repair_learnableP_h126", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_signedpairlite_classbranch_identitytailquad040_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B102a-SimpleFastTaskGeometry-h126-fixedP-absdiag050-signedpairtail-classbranch-identitytailquad040-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_signedpairtail_classwise_identitytail_quadratic_fixedp", 2, 126, "LineP_B101_structured_projection_random_tail_A4_repair_fixedP_h126", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_signedpairtail_classbranch_identitytailquad040_hingeamp025_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B102b-SimpleFastTaskGeometry-h126-learnableP-absdiag050-signedpairtail-classbranch-identitytailquad040-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_signedpairtail_classwise_identitytail_quadratic", 2, 126, "LineP_B101_structured_projection_random_tail_A4_repair_learnableP_h126", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_signedpairtail_classbranch_identitytailquad040_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B103a-SimpleFastTaskGeometry-h126-fixedP-absdiag050-signedpairtail-boundq-classbranch-identitytailquad040-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_signedpairtail_boundq_classwise_identitytail_quadratic_fixedp", 2, 126, "LineP_B102_bounded_projection_trajectory_repair_fixedP_h126", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_signedpairtail_boundq_classbranch_identitytailquad040_hingeamp025_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B103b-SimpleFastTaskGeometry-h126-learnableP-absdiag050-signedpairtail-boundq-classbranch-identitytailquad040-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_signedpairtail_boundq_classwise_identitytail_quadratic", 2, 126, "LineP_B102_bounded_projection_trajectory_repair_learnableP_h126", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_signedpairtail_boundq_classbranch_identitytailquad040_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B104a-SimpleFastTaskGeometry-h126-fixedP-absdiag050-classbranch-identitytailquad040-hingeamp025-fixedgain-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_classwise_identitytail_quadratic_fixedgain_fixedp", 2, 126, "LineP_B85_fixed_global_gain_trajectory_stabilizer_fixedP_h126", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_classbranch_identitytailquad040_hingeamp025_fixedgain_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B104b-SimpleFastTaskGeometry-h126-learnableP-absdiag050-classbranch-identitytailquad040-hingeamp025-fixedgain-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_classwise_identitytail_quadratic_fixedgain", 2, 126, "LineP_B85_fixed_global_gain_trajectory_stabilizer_learnableP_h126", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_classbranch_identitytailquad040_hingeamp025_fixedgain_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B105a-SimpleFastTaskGeometry-h126-fixedP-absdiag050-classbranch-identitytailquad040-hingeamp025-temp100", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_classwise_identitytail_quadratic_temp100_fixedp", 2, 126, "LineP_B85_higher_initial_gain_nll_trajectory_probe_fixedP_h126", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_classbranch_identitytailquad040_hingeamp025_temp100_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B105b-SimpleFastTaskGeometry-h126-learnableP-absdiag050-classbranch-identitytailquad040-hingeamp025-temp100", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_classwise_identitytail_quadratic_temp100", 2, 126, "LineP_B85_higher_initial_gain_nll_trajectory_probe_learnableP_h126", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_classbranch_identitytailquad040_hingeamp025_temp100", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B106a-SimpleFastTaskGeometry-h126-fixedP-absdiag050-classbranch-classgain-identitytailquad040-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_classwise_gain_identitytail_quadratic_fixedp", 2, 126, "LineP_B85_classwise_gain_low_cost_auc_repair_fixedP_h126", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_classbranch_classgain_identitytailquad040_hingeamp025_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B106b-SimpleFastTaskGeometry-h126-learnableP-absdiag050-classbranch-classgain-identitytailquad040-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_classwise_gain_identitytail_quadratic", 2, 126, "LineP_B85_classwise_gain_low_cost_auc_repair_learnableP_h126", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_classbranch_classgain_identitytailquad040_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B107a-SimpleFastTaskGeometry-h160-fixedP-absdiag050-classbranch-classgain-identitytailquad040-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_classwise_gain_identitytail_quadratic_fixedp_h160", 2, 160, "LineP_B106_kernel_headroom_capacity_repair_fixedP_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_classbranch_classgain_identitytailquad040_hingeamp025_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B107b-SimpleFastTaskGeometry-h160-learnableP-absdiag050-classbranch-classgain-identitytailquad040-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_classwise_gain_identitytail_quadratic_h160", 2, 160, "LineP_B106_kernel_headroom_capacity_repair_learnableP_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_classbranch_classgain_identitytailquad040_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B108a-SimpleFastTaskGeometry-h144-fixedP-absdiag050-classbranch-classgain-identitytailquad040-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_classwise_gain_identitytail_quadratic_fixedp_h144", 2, 144, "LineP_B106_B107_capacity_midpoint_fixedP_h144", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_classbranch_classgain_identitytailquad040_hingeamp025_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B108b-SimpleFastTaskGeometry-h144-learnableP-absdiag050-classbranch-classgain-identitytailquad040-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_classwise_gain_identitytail_quadratic_h144", 2, 144, "LineP_B106_B107_capacity_midpoint_learnableP_h144", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_classbranch_classgain_identitytailquad040_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B109a-SimpleFastTaskGeometry-h160-fixedP-absdiag050-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_classwise_gain_identitytail_quad030_fixedp_h160", 2, 160, "LineP_B107_lower_quadratic_branch_auc_repair_fixedP_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_classbranch_classgain_identitytailquad030_hingeamp025_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B109b-SimpleFastTaskGeometry-h160-learnableP-absdiag050-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_classwise_gain_identitytail_quad030_h160", 2, 160, "LineP_B107_lower_quadratic_branch_auc_repair_learnableP_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_classbranch_classgain_identitytailquad030_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B110a-SimpleFastTaskGeometry-h160-fixedP-absdiag050-classbranch-classgain-identitytailquad020-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_classwise_gain_identitytail_quad020_fixedp_h160", 2, 160, "LineP_B109_lower_quadratic_branch_auc_repair_fixedP_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_classbranch_classgain_identitytailquad020_hingeamp025_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B110b-SimpleFastTaskGeometry-h160-learnableP-absdiag050-classbranch-classgain-identitytailquad020-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_classwise_gain_identitytail_quad020_h160", 2, 160, "LineP_B109_lower_quadratic_branch_auc_repair_learnableP_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_classbranch_classgain_identitytailquad020_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B111a-SimpleFastTaskGeometry-h160-fixedP-absdiag050-classbranch-classgain-fixedgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_fixed_classwise_gain_identitytail_quad030_fixedp_h160", 2, 160, "LineP_B109_fixed_classgain_auc_repair_fixedP_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_classbranch_classgain_fixedgain_identitytailquad030_hingeamp025_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B111b-SimpleFastTaskGeometry-h160-learnableP-absdiag050-classbranch-classgain-fixedgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_fixed_classwise_gain_identitytail_quad030_h160", 2, 160, "LineP_B109_fixed_classgain_auc_repair_learnableP_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_classbranch_classgain_fixedgain_identitytailquad030_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B112a-SimpleFastTaskGeometry-h160-fixedP-absdiag050-normabs050-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_normabs050_classwise_gain_identitytail_quad030_fixedp_h160", 2, 160, "LineP_B109_global_contrast_auc_trajectory_repair_fixedP_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_normabs050_classbranch_classgain_identitytailquad030_hingeamp025_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B112b-SimpleFastTaskGeometry-h160-learnableP-absdiag050-normabs050-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_normabs050_classwise_gain_identitytail_quad030_h160", 2, 160, "LineP_B109_global_contrast_auc_trajectory_repair_learnableP_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_normabs050_classbranch_classgain_identitytailquad030_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B113a-SimpleFastTaskGeometry-h160-fixedP-absdiag050-normabs025-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_normabs025_classwise_gain_identitytail_quad030_fixedp_h160", 2, 160, "LineP_B112_lower_global_contrast_auc_repair_fixedP_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_normabs025_classbranch_classgain_identitytailquad030_hingeamp025_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B113b-SimpleFastTaskGeometry-h160-learnableP-absdiag050-normabs025-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_normabs025_classwise_gain_identitytail_quad030_h160", 2, 160, "LineP_B112_lower_global_contrast_auc_repair_learnableP_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_normabs025_classbranch_classgain_identitytailquad030_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B114a-SimpleFastTaskGeometry-h160-fixedP-absdiag050-normabs100-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_normabs100_classwise_gain_identitytail_quad030_fixedp_h160", 2, 160, "LineP_B112_stronger_global_contrast_auc_repair_fixedP_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_normabs100_classbranch_classgain_identitytailquad030_hingeamp025_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B114b-SimpleFastTaskGeometry-h160-learnableP-absdiag050-normabs100-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_normabs100_classwise_gain_identitytail_quad030_h160", 2, 160, "LineP_B112_stronger_global_contrast_auc_repair_learnableP_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_normabs100_classbranch_classgain_identitytailquad030_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B115a-SimpleFastTaskGeometry-h126-fixedP-absdiag050-normabs050-classbranch-classgain-identitytailquad040-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_normabs050_classwise_gain_identitytail_quad040_fixedp_h126", 2, 126, "LineP_B106_global_contrast_auc_repair_fixedP_h126", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_normabs050_classbranch_classgain_identitytailquad040_hingeamp025_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B115b-SimpleFastTaskGeometry-h126-learnableP-absdiag050-normabs050-classbranch-classgain-identitytailquad040-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_normabs050_classwise_gain_identitytail_quad040_h126", 2, 126, "LineP_B106_global_contrast_auc_repair_learnableP_h126", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_normabs050_classbranch_classgain_identitytailquad040_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B116a-SimpleFastTaskGeometry-h126-fixedP-absdiag050-normabs050-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_normabs050_classwise_gain_identitytail_quad030_fixedp_h126", 2, 126, "LineP_B115_lower_quad_global_contrast_auc_repair_fixedP_h126", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_normabs050_classbranch_classgain_identitytailquad030_hingeamp025_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B116b-SimpleFastTaskGeometry-h126-learnableP-absdiag050-normabs050-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_normabs050_classwise_gain_identitytail_quad030_h126", 2, 126, "LineP_B115_lower_quad_global_contrast_auc_repair_learnableP_h126", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_normabs050_classbranch_classgain_identitytailquad030_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B117a-SimpleFastTaskGeometry-h128-fixedP-absdiag050-normabs050-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_normabs050_classwise_gain_identitytail_quad030_fixedp_h128", 2, 128, "LineP_B116_block128_capacity_balance_auc_repair_fixedP_h128", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_normabs050_classbranch_classgain_identitytailquad030_hingeamp025_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B117b-SimpleFastTaskGeometry-h128-learnableP-absdiag050-normabs050-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_normabs050_classwise_gain_identitytail_quad030_h128", 2, 128, "LineP_B116_block128_capacity_balance_auc_repair_learnableP_h128", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_normabs050_classbranch_classgain_identitytailquad030_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B118a-SimpleFastTaskGeometry-h128-fixedP-absdiag050-normabs050-classbranch-classgain-identitytailquad040-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_normabs050_classwise_gain_identitytail_quad040_fixedp_h128", 2, 128, "LineP_B117_quad040_seed2_auc_repair_fixedP_h128", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_normabs050_classbranch_classgain_identitytailquad040_hingeamp025_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B118b-SimpleFastTaskGeometry-h128-learnableP-absdiag050-normabs050-classbranch-classgain-identitytailquad040-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_normabs050_classwise_gain_identitytail_quad040_h128", 2, 128, "LineP_B117_quad040_seed2_auc_repair_learnableP_h128", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_normabs050_classbranch_classgain_identitytailquad040_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B119a-SimpleFastTaskGeometry-h128-fixedP-absdiag050-normabs050-meanstat050-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_normabs050_meanstat050_classwise_gain_identitytail_quad030_fixedp_h128", 2, 128, "LineP_B117_global_contrast_signed_mean_auc_repair_fixedP_h128", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_normabs050_meanstat050_classbranch_classgain_identitytailquad030_hingeamp025_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B119b-SimpleFastTaskGeometry-h128-learnableP-absdiag050-normabs050-meanstat050-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_normabs050_meanstat050_classwise_gain_identitytail_quad030_h128", 2, 128, "LineP_B117_global_contrast_signed_mean_auc_repair_learnableP_h128", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_normabs050_meanstat050_classbranch_classgain_identitytailquad030_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B120a-SimpleFastTaskGeometry-h128-fixedP-absdiag050-meanstat050-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_meanstat050_classwise_gain_identitytail_quad030_fixedp_h128", 2, 128, "LineP_B117_signed_mean_only_auc_repair_fixedP_h128", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_meanstat050_classbranch_classgain_identitytailquad030_hingeamp025_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B120b-SimpleFastTaskGeometry-h128-learnableP-absdiag050-meanstat050-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_meanstat050_classwise_gain_identitytail_quad030_h128", 2, 128, "LineP_B117_signed_mean_only_auc_repair_learnableP_h128", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_meanstat050_classbranch_classgain_identitytailquad030_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B121a-SimpleFastTaskGeometry-h160-fixedP-absdiag050-meanstat050-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_meanstat050_classwise_gain_identitytail_quad030_fixedp_h160", 2, 160, "LineP_B109_signed_mean_bestline_auc_repair_fixedP_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_meanstat050_classbranch_classgain_identitytailquad030_hingeamp025_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B121b-SimpleFastTaskGeometry-h160-learnableP-absdiag050-meanstat050-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_meanstat050_classwise_gain_identitytail_quad030_h160", 2, 160, "LineP_B109_signed_mean_bestline_auc_repair_learnableP_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_meanstat050_classbranch_classgain_identitytailquad030_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B122a-SimpleFastTaskGeometry-h128-fixedP-absdiag050-groupabs4-050-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_groupabs4_050_classwise_gain_identitytail_quad030_fixedp_h128", 2, 128, "LineP_B117_grouped_abs_low_variance_auc_repair_fixedP_h128", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_groupabs4_groupabs050_classbranch_classgain_identitytailquad030_hingeamp025_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B122b-SimpleFastTaskGeometry-h128-learnableP-absdiag050-groupabs4-050-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_groupabs4_050_classwise_gain_identitytail_quad030_h128", 2, 128, "LineP_B117_grouped_abs_low_variance_auc_repair_learnableP_h128", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_groupabs4_groupabs050_classbranch_classgain_identitytailquad030_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B123a-SimpleFastTaskGeometry-h160-fixedP-absdiag050-groupabs4-050-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_groupabs4_050_classwise_gain_identitytail_quad030_fixedp_h160", 2, 160, "LineP_B109_grouped_abs_bestline_auc_repair_fixedP_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_groupabs4_groupabs050_classbranch_classgain_identitytailquad030_hingeamp025_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B123b-SimpleFastTaskGeometry-h160-learnableP-absdiag050-groupabs4-050-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_groupabs4_050_classwise_gain_identitytail_quad030_h160", 2, 160, "LineP_B109_grouped_abs_bestline_auc_repair_learnableP_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_groupabs4_groupabs050_classbranch_classgain_identitytailquad030_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B124a-SimpleFastTaskGeometry-h160-fixedP-absdiag050-classbranch-classgain-fixedbranch-fixedgain-identitytailquad020-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_fixed_branch_gain_identitytail_quad020_fixedp_h160", 2, 160, "LineP_B110_fixed_branch_gain_CEp99_auc_repair_fixedP_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_classbranch_classgain_fixedbranch_fixedgain_identitytailquad020_hingeamp025_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B124b-SimpleFastTaskGeometry-h160-learnableP-absdiag050-classbranch-classgain-fixedbranch-fixedgain-identitytailquad020-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_fixed_branch_gain_identitytail_quad020_h160", 2, 160, "LineP_B110_fixed_branch_gain_CEp99_auc_repair_learnableP_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_classbranch_classgain_fixedbranch_fixedgain_identitytailquad020_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B125a-SimpleFastTaskGeometry-h160-fixedP-absdiag050-classbranch-classgain-fixedbranch-fixedgain-identitytailquad020-hingeamp025-temp050", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_fixed_branch_low_gain_identitytail_quad020_fixedp_h160", 2, 160, "LineP_B124_fixed_low_gain_CEp99_auc_repair_fixedP_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_classbranch_classgain_fixedbranch_fixedgain_identitytailquad020_hingeamp025_temp050_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B125b-SimpleFastTaskGeometry-h160-learnableP-absdiag050-classbranch-classgain-fixedbranch-fixedgain-identitytailquad020-hingeamp025-temp050", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_fixed_branch_low_gain_identitytail_quad020_h160", 2, 160, "LineP_B124_fixed_low_gain_CEp99_auc_repair_learnableP_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_classbranch_classgain_fixedbranch_fixedgain_identitytailquad020_hingeamp025_temp050", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B126a-SimpleFastTaskGeometry-h160-fixedP-absdiag050-logitnorm150-classbranch-classgain-fixedbranch-fixedgain-identitytailquad020-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_logitnorm_fixed_branch_gain_identitytail_quad020_fixedp_h160", 2, 160, "LineP_B125_samplewise_logit_norm_CEp99_auc_repair_fixedP_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_logitnorm150_classbranch_classgain_fixedbranch_fixedgain_identitytailquad020_hingeamp025_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B126b-SimpleFastTaskGeometry-h160-learnableP-absdiag050-logitnorm150-classbranch-classgain-fixedbranch-fixedgain-identitytailquad020-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_logitnorm_fixed_branch_gain_identitytail_quad020_h160", 2, 160, "LineP_B125_samplewise_logit_norm_CEp99_auc_repair_learnableP_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_logitnorm150_classbranch_classgain_fixedbranch_fixedgain_identitytailquad020_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B127a-SimpleFastTaskGeometry-h160-fixedP-absquad050025-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absquad050025_classwise_gain_identitytail_quad030_fixedp_h160", 2, 160, "LineP_B109_abs_plus_square_tail_single_kernel_auc_repair_fixedP_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absquad050025_classbranch_classgain_identitytailquad030_hingeamp025_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B127b-SimpleFastTaskGeometry-h160-learnableP-absquad050025-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absquad050025_classwise_gain_identitytail_quad030_h160", 2, 160, "LineP_B109_abs_plus_square_tail_single_kernel_auc_repair_learnableP_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absquad050025_classbranch_classgain_identitytailquad030_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B128a-SimpleFastTaskGeometry-h160-fixedP-absmixsq025-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absmixsq025_classwise_gain_identitytail_quad030_fixedp_h160", 2, 160, "LineP_B127_single_tail_abs_square_mix_auc_repair_fixedP_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absmixsq025_classbranch_classgain_identitytailquad030_hingeamp025_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B128b-SimpleFastTaskGeometry-h160-learnableP-absmixsq025-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absmixsq025_classwise_gain_identitytail_quad030_h160", 2, 160, "LineP_B127_single_tail_abs_square_mix_auc_repair_learnableP_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absmixsq025_classbranch_classgain_identitytailquad030_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B129a-SimpleFastTaskGeometry-h160-fixedP-sqdiag025-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_sqdiag025_classwise_gain_identitytail_quad030_fixedp_h160", 2, 160, "LineP_B109_centered_energy_tail_auc_repair_fixedP_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_sqdiag025_classbranch_classgain_identitytailquad030_hingeamp025_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B129b-SimpleFastTaskGeometry-h160-learnableP-sqdiag025-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_sqdiag025_classwise_gain_identitytail_quad030_h160", 2, 160, "LineP_B109_centered_energy_tail_auc_repair_learnableP_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_sqdiag025_classbranch_classgain_identitytailquad030_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B130a-SimpleFastTaskGeometry-h160-fixedP-cubicdiag025-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_cubicdiag025_classwise_gain_identitytail_quad030_fixedp_h160", 2, 160, "LineP_B109_signed_cubic_tail_auc_repair_fixedP_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_cubicdiag025_classbranch_classgain_identitytailquad030_hingeamp025_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B130b-SimpleFastTaskGeometry-h160-learnableP-cubicdiag025-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_cubicdiag025_classwise_gain_identitytail_quad030_h160", 2, 160, "LineP_B109_signed_cubic_tail_auc_repair_learnableP_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_cubicdiag025_classbranch_classgain_identitytailquad030_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B131a-SimpleFastTaskGeometry-h160-fixedP-pairtraj-absdiag050-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_pairtraj_classwise_gain_identitytail_quad030_fixedp_h160", 2, 160, "LineP_B130_single_kernel_friendly_fixed_pair_trajectory_no_proj_grad_fixedP_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_pairtraj_classbranch_classgain_identitytailquad030_hingeamp025_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B131b-SimpleFastTaskGeometry-h160-learnableP-pairtraj-absdiag050-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_pairtraj_classwise_gain_identitytail_quad030_learnableP_h160", 2, 160, "LineP_B130_single_kernel_friendly_fixed_pair_trajectory_learnable_anchor_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_pairtraj_classbranch_classgain_identitytailquad030_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B132a-SimpleFastTaskGeometry-h128-fixedP-pairtraj-absdiag050-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_pairtraj_classwise_gain_identitytail_quad030_fixedp_h128", 2, 128, "LineP_B131_lower_update_cost_pair_trajectory_fixedP_h128", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_pairtraj_classbranch_classgain_identitytailquad030_hingeamp025_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B132b-SimpleFastTaskGeometry-h128-learnableP-pairtraj-absdiag050-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_pairtraj_classwise_gain_identitytail_quad030_learnableP_h128", 2, 128, "LineP_B131_lower_update_cost_pair_trajectory_learnable_anchor_h128", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_pairtraj_classbranch_classgain_identitytailquad030_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B133a-SimpleFastTaskGeometry-h160-fixedP-localdensepairtraj-absdiag050-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_local_dense_pairtraj_classwise_gain_identitytail_quad030_fixedp_h160", 2, 160, "LineP_B131_local_dense_pair_trajectory_expression_repair_fixedP_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_localdensepairtraj_classbranch_classgain_identitytailquad030_hingeamp025_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B133b-SimpleFastTaskGeometry-h160-learnableP-localdensepairtraj-absdiag050-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_local_dense_pairtraj_classwise_gain_identitytail_quad030_learnableP_h160", 2, 160, "LineP_B131_local_dense_pair_trajectory_expression_repair_learnable_anchor_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_localdensepairtraj_classbranch_classgain_identitytailquad030_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B134a-SimpleFastTaskGeometry-h192-fixedP-localdensepairtraj-absdiag050-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_local_dense_pairtraj_classwise_gain_identitytail_quad030_fixedp_h192", 2, 192, "LineP_B133_capacity_check_local_dense_pair_trajectory_fixedP_h192", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_localdensepairtraj_classbranch_classgain_identitytailquad030_hingeamp025_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B134b-SimpleFastTaskGeometry-h192-learnableP-localdensepairtraj-absdiag050-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_local_dense_pairtraj_classwise_gain_identitytail_quad030_learnableP_h192", 2, 192, "LineP_B133_capacity_check_local_dense_pair_trajectory_learnable_anchor_h192", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_localdensepairtraj_classbranch_classgain_identitytailquad030_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B135a-SimpleFastTaskGeometry-h160-fixedP-localdensepairtraj-absdiag050-classbranch-classgain-fixedbranch-fixedgain-identitytailquad020-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_local_dense_pairtraj_fixed_branch_gain_identitytail_quad020_fixedp_h160", 2, 160, "LineP_B133_pairtraj_with_B124_stable_branch_gain_fixedP_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_localdensepairtraj_classbranch_classgain_fixedbranch_fixedgain_identitytailquad020_hingeamp025_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B135b-SimpleFastTaskGeometry-h160-learnableP-localdensepairtraj-absdiag050-classbranch-classgain-fixedbranch-fixedgain-identitytailquad020-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_local_dense_pairtraj_fixed_branch_gain_identitytail_quad020_learnableP_h160", 2, 160, "LineP_B133_pairtraj_with_B124_stable_branch_gain_learnable_anchor_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_localdensepairtraj_classbranch_classgain_fixedbranch_fixedgain_identitytailquad020_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B136a-SimpleFastTaskGeometry-h160-fixedP-lowfreqP-absdiag050-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_lowfreqP_classwise_gain_identitytail_quad030_fixedp_h160", 2, 160, "v12.8.3_B109_low_frequency_fixed_projection_trajectory_repair_fixedP_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_lowfreqp_classbranch_classgain_identitytailquad030_hingeamp025_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B137a-SimpleFastTaskGeometry-h160-fixedP-blockfreqP-absdiag050-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_blockfreqP_classwise_gain_identitytail_quad030_fixedp_h160", 2, 160, "v12.8.3_B109_block_local_frequency_fixed_projection_trajectory_repair_fixedP_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_blockfreqp_classbranch_classgain_identitytailquad030_hingeamp025_temp075_fixedp", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B138b-SimpleFastTaskGeometry-h160-semiFixedP-freezeAfter1-absdiag050-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_semifixedP_freeze_after_epoch1_classwise_gain_identitytail_quad030_h160", 2, 160, "v12.8.3_B109_semifixed_projection_freeze_after_epoch1_trajectory_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_semifixedfreeze1_classbranch_classgain_identitytailquad030_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B139b-SimpleFastTaskGeometry-h160-learnableP-pUpdateEvery4-absdiag050-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_pupdate_every4_classwise_gain_identitytail_quad030_h160", 2, 160, "v12.8.3_B109_projection_gradient_every4_steps_trajectory_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_pupdateevery4_classbranch_classgain_identitytailquad030_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B140b-SimpleFastTaskGeometry-h160-learnableP-pUpdateEvery2-absdiag050-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_pupdate_every2_classwise_gain_identitytail_quad030_h160", 2, 160, "v12.8.3_B109_projection_gradient_every2_steps_trajectory_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_pupdateevery2_classbranch_classgain_identitytailquad030_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B141b-SimpleFastTaskGeometry-h160-learnableP-warm1PUpdateEvery4-absdiag050-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_warm1_pupdate_every4_classwise_gain_identitytail_quad030_h160", 2, 160, "v12.8.3_B109_projection_gradient_warm1_every4_steps_trajectory_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_warm1pupdateevery4_classbranch_classgain_identitytailquad030_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B142b-SimpleFastTaskGeometry-h160-learnableP-warm1PUpdateEvery2-absdiag050-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_warm1_pupdate_every2_classwise_gain_identitytail_quad030_h160", 2, 160, "v12.8.3_B109_projection_gradient_warm1_every2_steps_trajectory_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_warm1pupdateevery2_classbranch_classgain_identitytailquad030_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B143b-SimpleFastTaskGeometry-h160-learnableP-warm2PUpdateEvery4-absdiag050-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_warm2_pupdate_every4_classwise_gain_identitytail_quad030_h160", 2, 160, "v12.8.3_B109_projection_gradient_warm2_every4_steps_trajectory_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_warm2pupdateevery4_classbranch_classgain_identitytailquad030_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B144b-SimpleFastTaskGeometry-h160-learnableP-warm2PUpdateEvery2-absdiag050-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_warm2_pupdate_every2_classwise_gain_identitytail_quad030_h160", 2, 160, "v12.8.3_B109_projection_gradient_warm2_every2_steps_trajectory_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_warm2pupdateevery2_classbranch_classgain_identitytailquad030_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B145b-SimpleFastTaskGeometry-h160-learnableP-activeP64-absdiag050-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_activeP64_classwise_gain_identitytail_quad030_h160", 2, 160, "v12.8.3_B109_active64_projection_gradient_single_kernel_trajectory_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_activep64_classbranch_classgain_identitytailquad030_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B146b-SimpleFastTaskGeometry-h160-learnableP-activeP48-absdiag050-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_activeP48_classwise_gain_identitytail_quad030_h160", 2, 160, "v12.8.3_B109_active48_projection_gradient_efficiency_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_activep48_classbranch_classgain_identitytailquad030_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B147b-SimpleFastTaskGeometry-h160-learnableP-activeP96-absdiag050-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_activeP96_classwise_gain_identitytail_quad030_h160", 2, 160, "v12.8.3_B109_active96_projection_gradient_capacity_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_activep96_classbranch_classgain_identitytailquad030_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B148b-SimpleFastTaskGeometry-h160-learnableP-projlr025-absdiag050-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_projection_lr025_classwise_gain_identitytail_quad030_h160", 2, 160, "v12.8.3_B109_projection_lr025_trajectory_smoothing_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_projlr025_classbranch_classgain_identitytailquad030_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B149b-SimpleFastTaskGeometry-h160-learnableP-projlr010-absdiag050-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_projection_lr010_classwise_gain_identitytail_quad030_h160", 2, 160, "v12.8.3_B109_projection_lr010_trajectory_smoothing_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_projlr010_classbranch_classgain_identitytailquad030_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B150b-SimpleFastTaskGeometry-h160-learnableP-projlr050-absdiag050-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_projection_lr050_classwise_gain_identitytail_quad030_h160", 2, 160, "v12.8.3_B109_projection_lr050_midpoint_trajectory_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_projlr050_classbranch_classgain_identitytailquad030_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B151b-SimpleFastTaskGeometry-h160-learnableP-projlr075-absdiag050-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_projection_lr075_classwise_gain_identitytail_quad030_h160", 2, 160, "v12.8.3_B109_projection_lr075_midpoint_trajectory_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_projlr075_classbranch_classgain_identitytailquad030_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B152b-SimpleFastTaskGeometry-h160-learnableP-projlr100to050e2-absdiag050-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_projection_lr100_to_050_after_epoch2_classwise_gain_identitytail_quad030_h160", 2, 160, "v12.8.3_B109_projection_lr_schedule_100_to_050_after_epoch2_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_projlr100to050e2_classbranch_classgain_identitytailquad030_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B153b-SimpleFastTaskGeometry-h160-learnableP-projlr075to050e2-absdiag050-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_projection_lr075_to_050_after_epoch2_classwise_gain_identitytail_quad030_h160", 2, 160, "v12.8.3_B109_projection_lr_schedule_075_to_050_after_epoch2_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_projlr075to050e2_classbranch_classgain_identitytailquad030_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B154b-SimpleFastTaskGeometry-h160-learnableP-projlr100to050e1-absdiag050-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_projection_lr100_to_050_after_epoch1_classwise_gain_identitytail_quad030_h160", 2, 160, "v12.8.3_B109_projection_lr_schedule_100_to_050_after_epoch1_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_projlr100to050e1_classbranch_classgain_identitytailquad030_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B155b-SimpleFastTaskGeometry-h160-learnableP-projlr075to050e1-absdiag050-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_projection_lr075_to_050_after_epoch1_classwise_gain_identitytail_quad030_h160", 2, 160, "v12.8.3_B109_projection_lr_schedule_075_to_050_after_epoch1_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_projlr075to050e1_classbranch_classgain_identitytailquad030_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B156b-SimpleFastTaskGeometry-h160-learnableP-projlr075to055e1-absdiag050-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_projection_lr075_to_055_after_epoch1_classwise_gain_identitytail_quad030_h160", 2, 160, "v12.8.3_B109_projection_lr_schedule_075_to_055_after_epoch1_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_projlr075to055e1_classbranch_classgain_identitytailquad030_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B157b-SimpleFastTaskGeometry-h160-learnableP-projlr075to060e1-absdiag050-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_projection_lr075_to_060_after_epoch1_classwise_gain_identitytail_quad030_h160", 2, 160, "v12.8.3_B109_projection_lr_schedule_075_to_060_after_epoch1_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_projlr075to060e1_classbranch_classgain_identitytailquad030_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B158b-SimpleFastTaskGeometry-h160-learnableP-projlr075to060e1to050e3-absdiag050-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_projection_lr075_to_060_after_epoch1_to_050_after_epoch3_classwise_gain_identitytail_quad030_h160", 2, 160, "v12.8.3_B109_projection_lr_schedule_075_to_060_after_epoch1_to_050_after_epoch3_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_projlr075to060e1to050e3_classbranch_classgain_identitytailquad030_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B159b-SimpleFastTaskGeometry-h160-learnableP-projlr075to060e1to050e4-absdiag050-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_projection_lr075_to_060_after_epoch1_to_050_after_epoch4_classwise_gain_identitytail_quad030_h160", 2, 160, "v12.8.3_B109_projection_lr_schedule_075_to_060_after_epoch1_to_050_after_epoch4_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_projlr075to060e1to050e4_classbranch_classgain_identitytailquad030_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B160b-SimpleFastTaskGeometry-h160-learnableP-lowfreqP-activeP32-absdiag050-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_lowfreqP_activeP32_classwise_gain_identitytail_quad030_h160", 2, 160, "v12.8.3_B109_low_frequency_projection_update_K32_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_lowfreqp_activep32_classbranch_classgain_identitytailquad030_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B161b-SimpleFastTaskGeometry-h160-learnableP-lowfreqP-activeP64-absdiag050-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_lowfreqP_activeP64_classwise_gain_identitytail_quad030_h160", 2, 160, "v12.8.3_B109_low_frequency_projection_update_K64_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_lowfreqp_activep64_classbranch_classgain_identitytailquad030_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B162b-SimpleFastTaskGeometry-h160-learnableP-blockfreqP-activeP64-absdiag050-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_blockfreqP_activeP64_classwise_gain_identitytail_quad030_h160", 2, 160, "v12.8.3_B109_block_frequency_projection_update_K64_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_blockfreqp_activep64_classbranch_classgain_identitytailquad030_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B163b-SimpleFastTaskGeometry-h160-learnableP-blockfreqP-activeP96-absdiag050-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_blockfreqP_activeP96_classwise_gain_identitytail_quad030_h160", 2, 160, "v12.8.3_B109_block_frequency_projection_update_K96_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_blockfreqp_activep96_classbranch_classgain_identitytailquad030_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B164b-SimpleFastTaskGeometry-h160-pairScaleP-absdiag050-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_pairscaleP_classwise_gain_identitytail_quad030_h160", 2, 160, "v12.8.3_B109_pairscale_projection_H_update_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_pairscale_classbranch_classgain_identitytailquad030_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B165b-SimpleFastTaskGeometry-h160-pairScaleP075-absdiag050-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_pairscaleP075_classwise_gain_identitytail_quad030_h160", 2, 160, "v12.8.3_B109_pairscale075_projection_H_update_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_pairscale075_classbranch_classgain_identitytailquad030_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B166b-SimpleFastTaskGeometry-h160-sparseK8P-projlr050-absdiag050-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_sparseK8P_projlr050_classwise_gain_identitytail_quad030_h160", 2, 160, "v12.8.3_B109_sparseK8_projection_lr050_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_sparsek8p_projlr050_classbranch_classgain_identitytailquad030_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B167b-SimpleFastTaskGeometry-h160-sparseK16P-projlr050-absdiag050-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_sparseK16P_projlr050_classwise_gain_identitytail_quad030_h160", 2, 160, "v12.8.3_B109_sparseK16_projection_lr050_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_sparsek16p_projlr050_classbranch_classgain_identitytailquad030_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B168b-SimpleFastTaskGeometry-h160-sparseK16P-projlr025-absdiag050-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_sparseK16P_projlr025_classwise_gain_identitytail_quad030_h160", 2, 160, "v12.8.3_B109_sparseK16_projection_lr025_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_sparsek16p_projlr025_classbranch_classgain_identitytailquad030_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B169b-SimpleFastTaskGeometry-h160-learnableP-groupabs4-absdiag050-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_groupabs4_tail_signal_classwise_gain_identitytail_quad030_h160", 2, 160, "v12.8.3_B109_groupabs4_tail_local_auc_signal_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_groupabs4_groupabs050_classbranch_classgain_identitytailquad030_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B170b-SimpleFastTaskGeometry-h160-learnableP-groupabs8-absdiag050-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_groupabs8_tail_signal_classwise_gain_identitytail_quad030_h160", 2, 160, "v12.8.3_B109_groupabs8_tail_local_auc_signal_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_groupabs8_groupabs050_classbranch_classgain_identitytailquad030_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B171b-SimpleFastTaskGeometry-h160-learnableP-projlr075to050e1-groupabs4-absdiag050-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_groupabs4_projection_lr075_to_050_after_epoch1_classwise_gain_identitytail_quad030_h160", 2, 160, "v12.8.3_B155_groupabs4_tail_signal_projection_schedule_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_projlr075to050e1_groupabs4_groupabs050_classbranch_classgain_identitytailquad030_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B172b-SimpleFastTaskGeometry-h160-learnableP-projlr075to055e1-groupabs4-absdiag050-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_groupabs4_projection_lr075_to_055_after_epoch1_classwise_gain_identitytail_quad030_h160", 2, 160, "v12.8.3_B169_groupabs4_projection_lr055_tail_auc_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_projlr075to055e1_groupabs4_groupabs050_classbranch_classgain_identitytailquad030_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B173b-SimpleFastTaskGeometry-h160-learnableP-projlr075to060e1-groupabs4-absdiag050-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_groupabs4_projection_lr075_to_060_after_epoch1_classwise_gain_identitytail_quad030_h160", 2, 160, "v12.8.3_B169_groupabs4_projection_lr060_tail_auc_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_projlr075to060e1_groupabs4_groupabs050_classbranch_classgain_identitytailquad030_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B174b-SimpleFastTaskGeometry-h160-learnableP-projlr075to055e1-groupabs8-absdiag050-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_groupabs8_projection_lr075_to_055_after_epoch1_classwise_gain_identitytail_quad030_h160", 2, 160, "v12.8.3_B170_groupabs8_projection_lr055_tail_auc_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_projlr075to055e1_groupabs8_groupabs050_classbranch_classgain_identitytailquad030_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B175b-SimpleFastTaskGeometry-h160-learnableP-projlr075to060e1-groupabs8-absdiag050-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_groupabs8_projection_lr075_to_060_after_epoch1_classwise_gain_identitytail_quad030_h160", 2, 160, "v12.8.3_B170_groupabs8_projection_lr060_tail_auc_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_projlr075to060e1_groupabs8_groupabs050_classbranch_classgain_identitytailquad030_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B176b-SimpleFastTaskGeometry-h160-learnableP-groupabs4-075-absdiag050-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_groupabs4_075_tail_signal_classwise_gain_identitytail_quad030_h160", 2, 160, "v12.8.3_B169_groupabs4_stronger_tail_signal_auc_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_groupabs4_groupabs075_classbranch_classgain_identitytailquad030_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B177b-SimpleFastTaskGeometry-h160-learnableP-groupabs4-025-absdiag050-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_groupabs4_025_tail_signal_classwise_gain_identitytail_quad030_h160", 2, 160, "v12.8.3_B169_groupabs4_weaker_tail_signal_auc_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_groupabs4_groupabs025_classbranch_classgain_identitytailquad030_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B178b-SimpleFastTaskGeometry-h160-learnableP-projlr075to050e1-groupabs4-075-absdiag050-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_groupabs4_075_projection_lr075_to_050_after_epoch1_classwise_gain_identitytail_quad030_h160", 2, 160, "v12.8.3_B171_groupabs4_stronger_tail_signal_projection_schedule_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_projlr075to050e1_groupabs4_groupabs075_classbranch_classgain_identitytailquad030_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B179b-SimpleFastTaskGeometry-h160-learnableP-projlr075to050e1-groupabs4-025-absdiag050-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_groupabs4_025_projection_lr075_to_050_after_epoch1_classwise_gain_identitytail_quad030_h160", 2, 160, "v12.8.3_B171_groupabs4_weaker_tail_signal_projection_schedule_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_projlr075to050e1_groupabs4_groupabs025_classbranch_classgain_identitytailquad030_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B180b-SimpleFastTaskGeometry-h160-blockK8P-projlr050-absdiag050-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_blockK8P_projection_lr050_classwise_gain_identitytail_quad030_h160", 2, 160, "v12.8.3_B109_block_local_sparseK8_projection_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_blockk8p_projlr050_classbranch_classgain_identitytailquad030_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B181b-SimpleFastTaskGeometry-h160-blockK8P-projlr050-groupabs4-absdiag050-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_blockK8P_groupabs4_projection_lr050_classwise_gain_identitytail_quad030_h160", 2, 160, "v12.8.3_B109_block_local_sparseK8_groupabs4_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_blockk8p_projlr050_groupabs4_groupabs050_classbranch_classgain_identitytailquad030_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B182b-SimpleFastTaskGeometry-h160-blockK12P-projlr050-groupabs4-absdiag050-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_blockK12P_groupabs4_projection_lr050_classwise_gain_identitytail_quad030_h160", 2, 160, "v12.8.3_B109_block_local_sparseK12_groupabs4_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_blockk12p_projlr050_groupabs4_groupabs050_classbranch_classgain_identitytailquad030_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B183b-SimpleFastTaskGeometry-h160-blockK8P-projlr025-groupabs4-absdiag050-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_blockK8P_groupabs4_projection_lr025_classwise_gain_identitytail_quad030_h160", 2, 160, "v12.8.3_B109_block_local_sparseK8_groupabs4_low_projection_lr_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_blockk8p_projlr025_groupabs4_groupabs050_classbranch_classgain_identitytailquad030_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B184b-SimpleFastTaskGeometry-h160-hybridK8P-projlr050-absdiag050-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_hybridK8P_projection_lr050_classwise_gain_identitytail_quad030_h160", 2, 160, "v12.8.3_B109_hybrid_local_global_sparseK8_projection_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_hybridk8p_projlr050_classbranch_classgain_identitytailquad030_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B185b-SimpleFastTaskGeometry-h160-hybridK8P-projlr050-groupabs4-absdiag050-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_hybridK8P_groupabs4_projection_lr050_classwise_gain_identitytail_quad030_h160", 2, 160, "v12.8.3_B109_hybrid_local_global_sparseK8_groupabs4_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_hybridk8p_projlr050_groupabs4_groupabs050_classbranch_classgain_identitytailquad030_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B186b-SimpleFastTaskGeometry-h160-hybridK8P-projlr025-groupabs4-absdiag050-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_hybridK8P_groupabs4_projection_lr025_classwise_gain_identitytail_quad030_h160", 2, 160, "v12.8.3_B109_hybrid_local_global_sparseK8_groupabs4_low_projection_lr_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_hybridk8p_projlr025_groupabs4_groupabs050_classbranch_classgain_identitytailquad030_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B187b-SimpleFastTaskGeometry-h160-hybridK12P-projlr050-groupabs4-absdiag050-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_hybridK12P_groupabs4_projection_lr050_classwise_gain_identitytail_quad030_h160", 2, 160, "v12.8.3_B109_hybrid_local_global_sparseK12_groupabs4_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_hybridk12p_projlr050_groupabs4_groupabs050_classbranch_classgain_identitytailquad030_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B188b-SimpleFastTaskGeometry-h160-learnableP-projlr075to050e1-manualAdamW-absdiag050-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_denseP_projection_lr075_to_050_after_epoch1_manual_update_classwise_gain_identitytail_quad030_h160", 2, 160, "v12.8.3_B155_denseP_manual_foreach_adamw_update_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_projlr075to050e1_manualadamw_classbranch_classgain_identitytailquad030_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B189b-SimpleFastTaskGeometry-h160-learnableP-manualAdamW-absdiag050-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_denseP_manual_update_classwise_gain_identitytail_quad030_h160", 2, 160, "v12.8.3_B109_denseP_manual_foreach_adamw_update_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_manualadamw_classbranch_classgain_identitytailquad030_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B190b-SimpleFastTaskGeometry-h160-learnableP-projlr075to050e1-groupabs4-manualAdamW-absdiag050-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_groupabs4_denseP_projection_lr075_to_050_after_epoch1_manual_update_classwise_gain_identitytail_quad030_h160", 2, 160, "v12.8.3_B171_denseP_groupabs4_manual_foreach_adamw_update_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_projlr075to050e1_groupabs4_groupabs050_manualadamw_classbranch_classgain_identitytailquad030_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B191b-SimpleFastTaskGeometry-h160-learnableP-meanstat050-manualAdamW-absdiag050-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_meanstat050_denseP_manual_update_classwise_gain_identitytail_quad030_h160", 2, 160, "v12.8.3_B109_meanstat050_lowcost_trajectory_manual_update_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_meanstat050_manualadamw_classbranch_classgain_identitytailquad030_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B192b-SimpleFastTaskGeometry-h160-learnableP-normabs050-manualAdamW-absdiag050-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_normabs050_denseP_manual_update_classwise_gain_identitytail_quad030_h160", 2, 160, "v12.8.3_B109_normabs050_lowcost_trajectory_manual_update_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_normabs050_manualadamw_classbranch_classgain_identitytailquad030_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B193b-SimpleFastTaskGeometry-h160-learnableP-normabs050-meanstat050-manualAdamW-absdiag050-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_normabs050_meanstat050_denseP_manual_update_classwise_gain_identitytail_quad030_h160", 2, 160, "v12.8.3_B109_normabs_meanstat_lowcost_trajectory_manual_update_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_normabs050_meanstat050_manualadamw_classbranch_classgain_identitytailquad030_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B194b-SimpleFastTaskGeometry-h160-learnableP-rmsQ-manualAdamW-absdiag050-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_rmsq_denseP_manual_update_classwise_gain_identitytail_quad030_h160", 2, 160, "v12.8.3_B109_batch_rms_quadratic_trajectory_manual_update_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_rmsq_manualadamw_classbranch_classgain_identitytailquad030_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B195b-SimpleFastTaskGeometry-h160-learnableP-boundQ-manualAdamW-absdiag050-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_boundq_denseP_manual_update_classwise_gain_identitytail_quad030_h160", 2, 160, "v12.8.3_B109_bounded_quadratic_trajectory_manual_update_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_boundq_manualadamw_classbranch_classgain_identitytailquad030_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B196b-SimpleFastTaskGeometry-h160-learnableP-rmsQ-boundQ-manualAdamW-absdiag050-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_rmsq_boundq_denseP_manual_update_classwise_gain_identitytail_quad030_h160", 2, 160, "v12.8.3_B109_batch_rms_bounded_quadratic_trajectory_manual_update_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_rmsq_boundq_manualadamw_classbranch_classgain_identitytailquad030_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B197b-SimpleFastTaskGeometry-h160-learnableP-rmsQ-projlr075to050e1-manualAdamW-absdiag050-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_rmsq_denseP_projection_lr075_to_050_after_epoch1_manual_update_classwise_gain_identitytail_quad030_h160", 2, 160, "v12.8.3_B155_batch_rms_quadratic_schedule_manual_update_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_rmsq_projlr075to050e1_manualadamw_classbranch_classgain_identitytailquad030_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B198b-SimpleFastTaskGeometry-h160-learnableP-pUpdateEvery4-manualAdamW-absdiag050-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_pupdate_every4_manual_update_classwise_gain_identitytail_quad030_h160", 2, 160, "v12.8.3_B139_projection_update_every4_manual_foreach_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_pupdateevery4_manualadamw_classbranch_classgain_identitytailquad030_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B199b-SimpleFastTaskGeometry-h160-learnableP-pUpdateEvery2-manualAdamW-absdiag050-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_pupdate_every2_manual_update_classwise_gain_identitytail_quad030_h160", 2, 160, "v12.8.3_B140_projection_update_every2_manual_foreach_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_pupdateevery2_manualadamw_classbranch_classgain_identitytailquad030_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B200b-SimpleFastTaskGeometry-h160-learnableP-warm1PUpdateEvery4-manualAdamW-absdiag050-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_warm1_pupdate_every4_manual_update_classwise_gain_identitytail_quad030_h160", 2, 160, "v12.8.3_B141_warm1_projection_update_every4_manual_foreach_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_warm1pupdateevery4_manualadamw_classbranch_classgain_identitytailquad030_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B201b-SimpleFastTaskGeometry-h160-learnableP-warm2PUpdateEvery4-manualAdamW-absdiag050-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_warm2_pupdate_every4_manual_update_classwise_gain_identitytail_quad030_h160", 2, 160, "v12.8.3_B143_warm2_projection_update_every4_manual_foreach_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_warm2pupdateevery4_manualadamw_classbranch_classgain_identitytailquad030_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B202b-SimpleFastTaskGeometry-h160-learnableP-warm2PUpdateEvery2-manualAdamW-absdiag050-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_warm2_pupdate_every2_manual_update_classwise_gain_identitytail_quad030_h160", 2, 160, "v12.8.3_B144_warm2_projection_update_every2_manual_foreach_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_warm2pupdateevery2_manualadamw_classbranch_classgain_identitytailquad030_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B203b-SimpleFastTaskGeometry-h160-learnableP-tritonProjAdamW-manualAdamW-absdiag050-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_denseP_triton_quadproj_adamw_manual_update_classwise_gain_identitytail_quad030_h160", 2, 160, "v12.8.3_B109_denseP_triton_quadproj_adamw_update_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_tritonprojadamw_manualadamw_classbranch_classgain_identitytailquad030_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B204b-SimpleFastTaskGeometry-h160-learnableP-fusedProjGradAdamW-manualAdamW-absdiag050-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_denseP_fused_projgrad_adamw_manual_update_classwise_gain_identitytail_quad030_h160", 2, 160, "v12.8.3_B109_denseP_fused_projection_gradient_adamw_update_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_fusedquadprojadamw_manualadamw_classbranch_classgain_identitytailquad030_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B205b-SimpleFastTaskGeometry-h160-learnableP-fusedProjGradAdamW-logitnorm150-fixedbranch-fixedgain-manualAdamW-absdiag050-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_denseP_fused_projgrad_adamw_logitnorm150_fixed_branch_gain_manual_update_classwise_gain_identitytail_quad030_h160", 2, 160, "v12.8.3_B109_fused_update_samplewise_logitnorm_trajectory_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_logitnorm150_fixedbranch_fixedgain_fusedquadprojadamw_manualadamw_classbranch_classgain_identitytailquad030_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B206b-SimpleFastTaskGeometry-h160-learnableP-fusedProjGradAdamW-fixedbranch-fixedgain-manualAdamW-absdiag050-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_denseP_fused_projgrad_adamw_fixed_branch_gain_manual_update_classwise_gain_identitytail_quad030_h160", 2, 160, "v12.8.3_B109_fused_update_fixed_scalar_trajectory_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_fixedbranch_fixedgain_fusedquadprojadamw_manualadamw_classbranch_classgain_identitytailquad030_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B207b-SimpleFastTaskGeometry-h160-learnableP-fusedProjGradAdamW-logitnorm100-fixedbranch-fixedgain-manualAdamW-absdiag050-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_denseP_fused_projgrad_adamw_logitnorm100_fixed_branch_gain_manual_update_classwise_gain_identitytail_quad030_h160", 2, 160, "v12.8.3_B109_fused_update_lower_logitnorm_trajectory_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_logitnorm100_fixedbranch_fixedgain_fusedquadprojadamw_manualadamw_classbranch_classgain_identitytailquad030_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B208b-SimpleFastTaskGeometry-h160-hybridK12P-logitnorm150-fixedbranch-fixedgain-manualAdamW-absdiag050-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_hybridK12P_logitnorm150_fixed_branch_gain_manual_update_classwise_gain_identitytail_quad030_h160", 2, 160, "v12.8.3_B109_hybrid_sparseK12_logitnorm_trajectory_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_hybridk12p_projlr050_logitnorm150_fixedbranch_fixedgain_manualadamw_classbranch_classgain_identitytailquad030_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B209b-SimpleFastTaskGeometry-h160-hybridK16P-logitnorm150-fixedbranch-fixedgain-manualAdamW-absdiag050-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_hybridK16P_logitnorm150_fixed_branch_gain_manual_update_classwise_gain_identitytail_quad030_h160", 2, 160, "v12.8.3_B109_hybrid_sparseK16_logitnorm_trajectory_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_hybridk16p_projlr050_logitnorm150_fixedbranch_fixedgain_manualadamw_classbranch_classgain_identitytailquad030_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B210b-SimpleFastTaskGeometry-h160-learnableP-fusedProjGradAdamW-logitnormSG150-fixedbranch-fixedgain-manualAdamW-absdiag050-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_denseP_fused_projgrad_adamw_stopgrad_logitnorm150_fixed_branch_gain_manual_update_classwise_gain_identitytail_quad030_h160", 2, 160, "v12.8.3_B109_fused_update_stopgrad_samplewise_logitnorm_trajectory_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_logitnormsg150_fixedbranch_fixedgain_fusedquadprojadamw_manualadamw_classbranch_classgain_identitytailquad030_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B211b-SimpleFastTaskGeometry-h160-learnableP-warm2PUpdateEvery2-fusedProjGradAdamW-logitnormSG150-fixedbranch-fixedgain-manualAdamW-absdiag050-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_denseP_warm2_pupdate_every2_fused_projgrad_adamw_stopgrad_logitnorm150_fixed_branch_gain_manual_update_classwise_gain_identitytail_quad030_h160", 2, 160, "v12.8.3_B109_warm_scheduled_fused_update_stopgrad_logitnorm_trajectory_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_warm2pupdateevery2_logitnormsg150_fixedbranch_fixedgain_fusedquadprojadamw_manualadamw_classbranch_classgain_identitytailquad030_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B212b-SimpleFastTaskGeometry-h160-learnableP-fusedProjGradAdamW-logitBNSG150-fixedbranch-fixedgain-manualAdamW-absdiag050-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_denseP_fused_projgrad_adamw_stopgrad_batch_logitnorm150_fixed_branch_gain_manual_update_classwise_gain_identitytail_quad030_h160", 2, 160, "v12.8.3_B109_fused_update_stopgrad_batch_logitnorm_trajectory_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_logitbnsg150_fixedbranch_fixedgain_fusedquadprojadamw_manualadamw_classbranch_classgain_identitytailquad030_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B213b-SimpleFastTaskGeometry-h160-learnableP-warm2PUpdateEvery2-fusedProjGradAdamW-logitBNSG150-fixedbranch-fixedgain-manualAdamW-absdiag050-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_denseP_warm2_pupdate_every2_fused_projgrad_adamw_stopgrad_batch_logitnorm150_fixed_branch_gain_manual_update_classwise_gain_identitytail_quad030_h160", 2, 160, "v12.8.3_B109_warm_scheduled_fused_update_stopgrad_batch_logitnorm_trajectory_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_warm2pupdateevery2_logitbnsg150_fixedbranch_fixedgain_fusedquadprojadamw_manualadamw_classbranch_classgain_identitytailquad030_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B214b-SimpleFastTaskGeometry-h160-learnableP-fusedProjGradAdamW-absmixsq010-manualAdamW-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absmixsq010_denseP_fused_projgrad_adamw_manual_update_classwise_gain_identitytail_quad030_h160", 2, 160, "v12.8.3_B109_nonlogit_absmixsq010_direct_tail_trajectory_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absmixsq010_fusedquadprojadamw_manualadamw_classbranch_classgain_identitytailquad030_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B215b-SimpleFastTaskGeometry-h160-learnableP-fusedProjGradAdamW-absmixsq050-manualAdamW-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absmixsq050_denseP_fused_projgrad_adamw_manual_update_classwise_gain_identitytail_quad030_h160", 2, 160, "v12.8.3_B109_nonlogit_absmixsq050_direct_tail_trajectory_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absmixsq050_fusedquadprojadamw_manualadamw_classbranch_classgain_identitytailquad030_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B216b-SimpleFastTaskGeometry-h160-learnableP-warm2PUpdateEvery2-fusedProjGradAdamW-absmixsq010-manualAdamW-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absmixsq010_denseP_warm2_pupdate_every2_fused_projgrad_adamw_manual_update_classwise_gain_identitytail_quad030_h160", 2, 160, "v12.8.3_B109_nonlogit_absmixsq010_warm_scheduled_direct_tail_trajectory_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absmixsq010_warm2pupdateevery2_fusedquadprojadamw_manualadamw_classbranch_classgain_identitytailquad030_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B217b-SimpleFastTaskGeometry-h160-learnableP-fusedProjGradAdamW-sgnsqdiag025-manualAdamW-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_signed_square025_denseP_fused_projgrad_adamw_manual_update_classwise_gain_identitytail_quad030_h160", 2, 160, "v12.8.3_B109_nonlogit_signed_square_direct_tail_trajectory_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_sgnsqdiag025_fusedquadprojadamw_manualadamw_classbranch_classgain_identitytailquad030_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B218b-SimpleFastTaskGeometry-h160-learnableP-warm2PUpdateEvery2-fusedProjGradAdamW-sgnsqdiag025-manualAdamW-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_signed_square025_denseP_warm2_pupdate_every2_fused_projgrad_adamw_manual_update_classwise_gain_identitytail_quad030_h160", 2, 160, "v12.8.3_B109_nonlogit_signed_square_warm_scheduled_direct_tail_trajectory_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_sgnsqdiag025_warm2pupdateevery2_fusedquadprojadamw_manualadamw_classbranch_classgain_identitytailquad030_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B219b-SimpleFastTaskGeometry-h160-learnableP-fusedProjGradAdamW-absquad050010-manualAdamW-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absquad050010_denseP_fused_projgrad_adamw_manual_update_classwise_gain_identitytail_quad030_h160", 2, 160, "v12.8.3_B109_nonlogit_abs_plus_weak_square_direct_tail_trajectory_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absquad050010_fusedquadprojadamw_manualadamw_classbranch_classgain_identitytailquad030_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B220b-SimpleFastTaskGeometry-h160-learnableP-warm2PUpdateEvery2-fusedProjGradAdamW-absquad050010-manualAdamW-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absquad050010_denseP_warm2_pupdate_every2_fused_projgrad_adamw_manual_update_classwise_gain_identitytail_quad030_h160", 2, 160, "v12.8.3_B109_nonlogit_abs_plus_weak_square_warm_scheduled_direct_tail_trajectory_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absquad050010_warm2pupdateevery2_fusedquadprojadamw_manualadamw_classbranch_classgain_identitytailquad030_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B221b-SimpleFastTaskGeometry-h160-learnableP-fusedProjGradAdamW-sgnsqdiag025-groupabs4-manualAdamW-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_signed_square025_groupabs4_denseP_fused_projgrad_adamw_manual_update_classwise_gain_identitytail_quad030_h160", 2, 160, "v12.8.3_B109_structured_signed_square_groupabs4_trajectory_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_sgnsqdiag025_groupabs4_groupabs050_fusedquadprojadamw_manualadamw_classbranch_classgain_identitytailquad030_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B222b-SimpleFastTaskGeometry-h160-learnableP-fusedProjGradAdamW-sgnsqdiag025-groupabs8-manualAdamW-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_signed_square025_groupabs8_denseP_fused_projgrad_adamw_manual_update_classwise_gain_identitytail_quad030_h160", 2, 160, "v12.8.3_B109_structured_signed_square_groupabs8_trajectory_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_sgnsqdiag025_groupabs8_groupabs050_fusedquadprojadamw_manualadamw_classbranch_classgain_identitytailquad030_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B223b-SimpleFastTaskGeometry-h160-learnableP-warm2PUpdateEvery2-fusedProjGradAdamW-sgnsqdiag025-groupabs4-manualAdamW-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_signed_square025_groupabs4_denseP_warm2_pupdate_every2_fused_projgrad_adamw_manual_update_classwise_gain_identitytail_quad030_h160", 2, 160, "v12.8.3_B109_structured_signed_square_groupabs4_warm_scheduled_trajectory_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_sgnsqdiag025_groupabs4_groupabs050_warm2pupdateevery2_fusedquadprojadamw_manualadamw_classbranch_classgain_identitytailquad030_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B224b-SimpleFastTaskGeometry-h160-learnableP-orthop-fusedProjGradAdamW-absdiag050-manualAdamW-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_orthop_denseP_fused_projgrad_adamw_manual_update_classwise_gain_identitytail_quad030_h160", 2, 160, "v12.8.3_B109_condition_preserving_orthop_denseP_trajectory_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_orthop_fusedquadprojadamw_manualadamw_classbranch_classgain_identitytailquad030_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B225b-SimpleFastTaskGeometry-h160-learnableP-orthop-fusedProjGradAdamW-sgnsqdiag025-manualAdamW-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_signed_square025_orthop_denseP_fused_projgrad_adamw_manual_update_classwise_gain_identitytail_quad030_h160", 2, 160, "v12.8.3_B109_condition_preserving_orthop_signed_square_trajectory_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_sgnsqdiag025_orthop_fusedquadprojadamw_manualadamw_classbranch_classgain_identitytailquad030_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B226b-SimpleFastTaskGeometry-h160-learnableP-pcaP-fusedProjGradAdamW-absdiag050-manualAdamW-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_pcap_denseP_fused_projgrad_adamw_manual_update_classwise_gain_identitytail_quad030_h160", 2, 160, "v12.8.3_B109_label_free_pcaP_signal_projected_trajectory_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_pcap_fusedquadprojadamw_manualadamw_classbranch_classgain_identitytailquad030_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B227b-SimpleFastTaskGeometry-h160-learnableP-pcaP-fusedProjGradAdamW-sgnsqdiag025-manualAdamW-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_signed_square025_pcap_denseP_fused_projgrad_adamw_manual_update_classwise_gain_identitytail_quad030_h160", 2, 160, "v12.8.3_B109_label_free_pcaP_signed_square_signal_projected_trajectory_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_sgnsqdiag025_pcap_fusedquadprojadamw_manualadamw_classbranch_classgain_identitytailquad030_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B228b-SimpleFastTaskGeometry-h160-learnableP-pcaBandP-fusedProjGradAdamW-absdiag050-manualAdamW-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_pca_band_denseP_fused_projgrad_adamw_manual_update_classwise_gain_identitytail_quad030_h160", 2, 160, "v12.8.3_B109_label_free_pca_band_signal_projected_trajectory_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_pcabandp_fusedquadprojadamw_manualadamw_classbranch_classgain_identitytailquad030_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B229b-SimpleFastTaskGeometry-h160-learnableP-pcaBandP-fusedProjGradAdamW-sgnsqdiag025-manualAdamW-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_signed_square025_pca_band_denseP_fused_projgrad_adamw_manual_update_classwise_gain_identitytail_quad030_h160", 2, 160, "v12.8.3_B109_label_free_pca_band_signed_square_signal_projected_trajectory_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_sgnsqdiag025_pcabandp_fusedquadprojadamw_manualadamw_classbranch_classgain_identitytailquad030_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B230b-SimpleFastTaskGeometry-h160-learnableP-trainProbeP-direct025-fusedProjGradAdamW-absdiag050-manualAdamW-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_trainprobeP_trainprobedirect025_fused_projgrad_adamw_manual_update_classwise_gain_identitytail_quad030_h160", 2, 160, "v12.8.3_B109_audited_train_probe_centroid_signal_direct025_trajectory_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_trainprobeP_trainprobedirect025_fusedquadprojadamw_manualadamw_classbranch_classgain_identitytailquad030_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B231b-SimpleFastTaskGeometry-h160-learnableP-trainProbeP-direct050-fusedProjGradAdamW-absdiag050-manualAdamW-classbranch-classgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_trainprobeP_trainprobedirect050_fused_projgrad_adamw_manual_update_classwise_gain_identitytail_quad030_h160", 2, 160, "v12.8.3_B109_audited_train_probe_centroid_signal_direct050_trajectory_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_trainprobeP_trainprobedirect050_fusedquadprojadamw_manualadamw_classbranch_classgain_identitytailquad030_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B232b-SimpleFastTaskGeometry-h160-learnableP-trainProbeP-direct050-fusedProjGradAdamW-absdiag050-manualAdamW-classbranch-classgain-identitytailquad030-hingeamp025-temp065", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_trainprobeP_trainprobedirect050_fused_projgrad_adamw_manual_update_classwise_gain_identitytail_quad030_temp065_h160", 2, 160, "v12.8.3_B109_audited_train_probe_centroid_signal_direct050_temp065_ece_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_trainprobeP_trainprobedirect050_fusedquadprojadamw_manualadamw_classbranch_classgain_identitytailquad030_hingeamp025_temp065", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B233b-SimpleFastTaskGeometry-h160-learnableP-trainProbeP-direct050-fusedProjGradAdamW-absdiag050-manualAdamW-classbranch-classgain-identitytailquad030-hingeamp025-temp050", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_trainprobeP_trainprobedirect050_fused_projgrad_adamw_manual_update_classwise_gain_identitytail_quad030_temp050_h160", 2, 160, "v12.8.3_B109_audited_train_probe_centroid_signal_direct050_temp050_ece_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_trainprobeP_trainprobedirect050_fusedquadprojadamw_manualadamw_classbranch_classgain_identitytailquad030_hingeamp025_temp050", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B234b-SimpleFastTaskGeometry-h160-learnableP-trainProbeP-direct050-fusedProjGradAdamW-absdiag050-manualAdamW-classbranch-classgain-fixedgain-identitytailquad030-hingeamp025-temp065", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_trainprobeP_trainprobedirect050_fused_projgrad_adamw_manual_update_fixedgain_quad030_temp065_h160", 2, 160, "v12.8.3_B109_audited_train_probe_centroid_signal_direct050_fixedgain_temp065_ece_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_trainprobeP_trainprobedirect050_fusedquadprojadamw_manualadamw_classbranch_classgain_fixedgain_identitytailquad030_hingeamp025_temp065", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B235b-SimpleFastTaskGeometry-h160-learnableP-trainProbeP-direct050-fusedProjGradAdamW-absdiag050-manualAdamW-classbranch-classgain-fixedgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_trainprobeP_trainprobedirect050_fused_projgrad_adamw_manual_update_fixedgain_quad030_temp075_h160", 2, 160, "v12.8.3_B109_audited_train_probe_centroid_signal_direct050_fixedgain_temp075_ece_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_trainprobeP_trainprobedirect050_fusedquadprojadamw_manualadamw_classbranch_classgain_fixedgain_identitytailquad030_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B236b-SimpleFastTaskGeometry-h160-learnableP-trainProbeP-direct050-fusedProjGradAdamW-absdiag050-manualAdamW-classbranch-fixedgain-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_trainprobeP_trainprobedirect050_fused_projgrad_adamw_manual_update_global_fixedgain_quad030_temp075_h160", 2, 160, "v12.8.3_B109_audited_train_probe_centroid_signal_global_fixedgain_temp075_ece_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_trainprobeP_trainprobedirect050_fusedquadprojadamw_manualadamw_classbranch_fixedgain_identitytailquad030_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B237b-SimpleFastTaskGeometry-h160-learnableP-trainProbeP-direct050-fusedProjGradAdamW-absdiag050-manualAdamW-classbranch-fixedgain-identitytailquad030-hingeamp025-temp065", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_trainprobeP_trainprobedirect050_fused_projgrad_adamw_manual_update_global_fixedgain_quad030_temp065_h160", 2, 160, "v12.8.3_B109_audited_train_probe_centroid_signal_global_fixedgain_temp065_ece_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_trainprobeP_trainprobedirect050_fusedquadprojadamw_manualadamw_classbranch_fixedgain_identitytailquad030_hingeamp025_temp065", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B238b-SimpleFastTaskGeometry-h160-learnableP-trainProbeP-direct050-fusedProjGradAdamW-absdiag050-manualAdamW-classbranch-fixedgain-logitcap300-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_trainprobeP_trainprobedirect050_fused_projgrad_adamw_manual_update_global_fixedgain_logitcap300_quad030_temp075_h160", 2, 160, "v12.8.3_B109_audited_train_probe_bounded_logitcap300_ece_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_trainprobeP_trainprobedirect050_fusedquadprojadamw_manualadamw_classbranch_fixedgain_logitcap300_identitytailquad030_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B239b-SimpleFastTaskGeometry-h160-learnableP-trainProbeP-direct050-fusedProjGradAdamW-absdiag050-manualAdamW-classbranch-fixedgain-logitcap250-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_trainprobeP_trainprobedirect050_fused_projgrad_adamw_manual_update_global_fixedgain_logitcap250_quad030_temp075_h160", 2, 160, "v12.8.3_B109_audited_train_probe_bounded_logitcap250_ece_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_trainprobeP_trainprobedirect050_fusedquadprojadamw_manualadamw_classbranch_fixedgain_logitcap250_identitytailquad030_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B240b-SimpleFastTaskGeometry-h160-learnableP-trainProbeP-direct050-fusedProjGradAdamW-absdiag050-manualAdamW-classbranch-fixedgain-logitcap400-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_trainprobeP_trainprobedirect050_fused_projgrad_adamw_manual_update_global_fixedgain_logitcap400_quad030_temp075_h160", 2, 160, "v12.8.3_B109_audited_train_probe_bounded_logitcap400_ece_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_trainprobeP_trainprobedirect050_fusedquadprojadamw_manualadamw_classbranch_fixedgain_logitcap400_identitytailquad030_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B241b-SimpleFastTaskGeometry-h160-learnableP-trainProbeP-direct050-fusedProjGradAdamW-absdiag050-manualAdamW-classbranch-fixedgain-logitcap500-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_trainprobeP_trainprobedirect050_fused_projgrad_adamw_manual_update_global_fixedgain_logitcap500_quad030_temp075_h160", 2, 160, "v12.8.3_B109_audited_train_probe_bounded_logitcap500_ece_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_trainprobeP_trainprobedirect050_fusedquadprojadamw_manualadamw_classbranch_fixedgain_logitcap500_identitytailquad030_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B242b-SimpleFastTaskGeometry-h160-learnableP-trainProbeP-direct050-fusedProjGradAdamW-absdiag050-manualAdamW-classbranch-fixedgain-identitytailquad030-hingeamp025-temp100", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_trainprobeP_trainprobedirect050_fused_projgrad_adamw_manual_update_global_fixedgain_quad030_temp100_h160", 2, 160, "v12.8.3_B109_audited_train_probe_global_fixedgain_temp100_ece_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_trainprobeP_trainprobedirect050_fusedquadprojadamw_manualadamw_classbranch_fixedgain_identitytailquad030_hingeamp025_temp100", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B243b-SimpleFastTaskGeometry-h160-learnableP-trainProbeP-direct050-fusedProjGradAdamW-absdiag050-manualAdamW-classbranch-fixedgain-identitytailquad030-hingeamp025-temp125", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_trainprobeP_trainprobedirect050_fused_projgrad_adamw_manual_update_global_fixedgain_quad030_temp125_h160", 2, 160, "v12.8.3_B109_audited_train_probe_global_fixedgain_temp125_ece_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_trainprobeP_trainprobedirect050_fusedquadprojadamw_manualadamw_classbranch_fixedgain_identitytailquad030_hingeamp025_temp125", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B244b-SimpleFastTaskGeometry-h160-learnableP-trainProbeP-direct050-fusedProjGradAdamW-absdiag050-manualAdamW-classbranch-fixedgain-identitytailquad030-hingeamp025-temp110", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_trainprobeP_trainprobedirect050_fused_projgrad_adamw_manual_update_global_fixedgain_quad030_temp110_h160", 2, 160, "v12.8.3_B109_audited_train_probe_global_fixedgain_temp110_auc_ece_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_trainprobeP_trainprobedirect050_fusedquadprojadamw_manualadamw_classbranch_fixedgain_identitytailquad030_hingeamp025_temp110", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B245b-SimpleFastTaskGeometry-h160-learnableP-trainProbeP-direct050-fusedProjGradAdamW-absdiag050-manualAdamW-classbranch-fixedgain-identitytailquad030-hingeamp025-temp115", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_trainprobeP_trainprobedirect050_fused_projgrad_adamw_manual_update_global_fixedgain_quad030_temp115_h160", 2, 160, "v12.8.3_B109_audited_train_probe_global_fixedgain_temp115_auc_ece_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_trainprobeP_trainprobedirect050_fusedquadprojadamw_manualadamw_classbranch_fixedgain_identitytailquad030_hingeamp025_temp115", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B246b-SimpleFastTaskGeometry-h160-learnableP-trainProbeP-direct050-fusedProjGradAdamW-absdiag050-manualAdamW-classbranch-fixedgain-identitytailquad030-hingeamp025-temp105", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_trainprobeP_trainprobedirect050_fused_projgrad_adamw_manual_update_global_fixedgain_quad030_temp105_h160", 2, 160, "v12.8.3_B109_audited_train_probe_global_fixedgain_temp105_auc_ece_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_trainprobeP_trainprobedirect050_fusedquadprojadamw_manualadamw_classbranch_fixedgain_identitytailquad030_hingeamp025_temp105", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B247b-SimpleFastTaskGeometry-h160-learnableP-trainProbeP-direct050-fusedProjGradAdamW-absdiag050-manualAdamW-classbranch-fixedgain-identitytailquad030-hingeamp025-temp107", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_trainprobeP_trainprobedirect050_fused_projgrad_adamw_manual_update_global_fixedgain_quad030_temp107_h160", 2, 160, "v12.8.3_B109_audited_train_probe_global_fixedgain_temp107_auc_ece_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_trainprobeP_trainprobedirect050_fusedquadprojadamw_manualadamw_classbranch_fixedgain_identitytailquad030_hingeamp025_temp107", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B248b-SimpleFastTaskGeometry-h160-learnableP-trainProbeP-direct050-fusedProjGradAdamW-absdiag050-manualAdamW-classbranch-fixedgain-gainramp075-identitytailquad030-hingeamp025-temp107", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_trainprobeP_trainprobedirect050_fused_projgrad_adamw_manual_update_global_fixedgain_gainramp075_quad030_temp107_h160", 2, 160, "v12.8.3_B109_audited_train_probe_global_fixedgain_temp107_gainramp075_auc_ece_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_trainprobeP_trainprobedirect050_fusedquadprojadamw_manualadamw_classbranch_fixedgain_gainramp075_identitytailquad030_hingeamp025_temp107", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B249b-SimpleFastTaskGeometry-h160-learnableP-trainProbeP-direct050-fusedProjGradAdamW-absdiag050-manualAdamW-classbranch-fixedgain-gainramp075-identitytailquad030-hingeamp025-temp110", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_trainprobeP_trainprobedirect050_fused_projgrad_adamw_manual_update_global_fixedgain_gainramp075_quad030_temp110_h160", 2, 160, "v12.8.3_B109_audited_train_probe_global_fixedgain_temp110_gainramp075_auc_ece_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_trainprobeP_trainprobedirect050_fusedquadprojadamw_manualadamw_classbranch_fixedgain_gainramp075_identitytailquad030_hingeamp025_temp110", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B250b-SimpleFastTaskGeometry-h160-learnableP-trainProbeP-direct050-fusedProjGradAdamW-absdiag050-manualAdamW-classbranch-fixedgain-gainramp065-identitytailquad030-hingeamp025-temp107", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_trainprobeP_trainprobedirect050_fused_projgrad_adamw_manual_update_global_fixedgain_gainramp065_quad030_temp107_h160", 2, 160, "v12.8.3_B109_audited_train_probe_global_fixedgain_temp107_gainramp065_auc_ece_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_trainprobeP_trainprobedirect050_fusedquadprojadamw_manualadamw_classbranch_fixedgain_gainramp065_identitytailquad030_hingeamp025_temp107", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B251b-SimpleFastTaskGeometry-h160-learnableP-trainProbeP-direct050-fusedProjGradAdamW-absdiag050-manualAdamW-classbranch-fixedgain-gainramp050-identitytailquad030-hingeamp025-temp107", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_trainprobeP_trainprobedirect050_fused_projgrad_adamw_manual_update_global_fixedgain_gainramp050_quad030_temp107_h160", 2, 160, "v12.8.3_B109_audited_train_probe_global_fixedgain_temp107_gainramp050_auc_ece_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_trainprobeP_trainprobedirect050_fusedquadprojadamw_manualadamw_classbranch_fixedgain_gainramp050_identitytailquad030_hingeamp025_temp107", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B252b-SimpleFastTaskGeometry-h160-learnableP-trainProbeP-direct075-fusedProjGradAdamW-absdiag050-manualAdamW-classbranch-fixedgain-gainramp065-identitytailquad030-hingeamp025-temp107", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_trainprobeP_trainprobedirect075_fused_projgrad_adamw_manual_update_global_fixedgain_gainramp065_quad030_temp107_h160", 2, 160, "v12.8.3_B109_audited_train_probe_direct075_global_fixedgain_temp107_gainramp065_auc_ece_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_trainprobeP_trainprobedirect075_fusedquadprojadamw_manualadamw_classbranch_fixedgain_gainramp065_identitytailquad030_hingeamp025_temp107", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B253b-SimpleFastTaskGeometry-h160-learnableP-trainProbeP-direct075-fusedProjGradAdamW-absdiag050-manualAdamW-classbranch-fixedgain-gainramp050-identitytailquad030-hingeamp025-temp107", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_trainprobeP_trainprobedirect075_fused_projgrad_adamw_manual_update_global_fixedgain_gainramp050_quad030_temp107_h160", 2, 160, "v12.8.3_B109_audited_train_probe_direct075_global_fixedgain_temp107_gainramp050_auc_ece_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_trainprobeP_trainprobedirect075_fusedquadprojadamw_manualadamw_classbranch_fixedgain_gainramp050_identitytailquad030_hingeamp025_temp107", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B254b-SimpleFastTaskGeometry-h160-learnableP-trainProbeP-direct050-fusedProjGradAdamW-absdiag050-manualAdamW-classbranch-fixedgain-gainramp075-identitytailquad030-hingeamp025-temp115", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_trainprobeP_trainprobedirect050_fused_projgrad_adamw_manual_update_global_fixedgain_gainramp075_quad030_temp115_h160", 2, 160, "v12.8.3_B109_audited_train_probe_global_fixedgain_temp115_gainramp075_auc_ece_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_trainprobeP_trainprobedirect050_fusedquadprojadamw_manualadamw_classbranch_fixedgain_gainramp075_identitytailquad030_hingeamp025_temp115", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B255b-SimpleFastTaskGeometry-h160-learnableP-trainProbeP-direct050-fusedProjGradAdamW-absdiag050-manualAdamW-classbranch-fixedgain-gainramp075-identitytailquad030-hingeamp025-temp105", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_trainprobeP_trainprobedirect050_fused_projgrad_adamw_manual_update_global_fixedgain_gainramp075_quad030_temp105_h160", 2, 160, "v12.8.3_B109_audited_train_probe_global_fixedgain_temp105_gainramp075_auc_confirm10_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_trainprobeP_trainprobedirect050_fusedquadprojadamw_manualadamw_classbranch_fixedgain_gainramp075_identitytailquad030_hingeamp025_temp105", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B256b-SimpleFastTaskGeometry-h160-learnableP-trainProbeP-direct050-fusedProjGradAdamW-absdiag050-manualAdamW-classbranch-fixedgain-gainramp075-identitytailquad030-hingeamp025-temp100", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_trainprobeP_trainprobedirect050_fused_projgrad_adamw_manual_update_global_fixedgain_gainramp075_quad030_temp100_h160", 2, 160, "v12.8.3_B109_audited_train_probe_global_fixedgain_temp100_gainramp075_auc_confirm10_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_trainprobeP_trainprobedirect050_fusedquadprojadamw_manualadamw_classbranch_fixedgain_gainramp075_identitytailquad030_hingeamp025_temp100", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B257b-SimpleFastTaskGeometry-h160-learnableP-trainProbeP-direct050-fusedProjGradAdamW-absdiag050-manualAdamW-classbranch-fixedgain-gainramp075-identitytailquad030-hingeamp025-temp095", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_trainprobeP_trainprobedirect050_fused_projgrad_adamw_manual_update_global_fixedgain_gainramp075_quad030_temp095_h160", 2, 160, "v12.8.3_B109_audited_train_probe_global_fixedgain_temp095_gainramp075_auc_confirm10_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_trainprobeP_trainprobedirect050_fusedquadprojadamw_manualadamw_classbranch_fixedgain_gainramp075_identitytailquad030_hingeamp025_temp095", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B258b-SimpleFastTaskGeometry-h160-learnableP-trainProbeP-direct050-fusedProjGradAdamW-absdiag050-manualAdamW-classbranch-fixedgain-gainramp075-identitytailquad030-hingeamp025-temp090", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_trainprobeP_trainprobedirect050_fused_projgrad_adamw_manual_update_global_fixedgain_gainramp075_quad030_temp090_h160", 2, 160, "v12.8.3_B109_audited_train_probe_global_fixedgain_temp090_gainramp075_auc_confirm10_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_trainprobeP_trainprobedirect050_fusedquadprojadamw_manualadamw_classbranch_fixedgain_gainramp075_identitytailquad030_hingeamp025_temp090", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B259b-SimpleFastTaskGeometry-h160-learnableP-trainProbeP-direct050-fusedProjGradAdamW-absdiag050-manualAdamW-classbranch-fixedgain-gainramp075-identitytailquad030-hingeamp025-temp092", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_trainprobeP_trainprobedirect050_fused_projgrad_adamw_manual_update_global_fixedgain_gainramp075_quad030_temp092_h160", 2, 160, "v12.8.3_B109_audited_train_probe_global_fixedgain_temp092_gainramp075_auc_confirm10_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_trainprobeP_trainprobedirect050_fusedquadprojadamw_manualadamw_classbranch_fixedgain_gainramp075_identitytailquad030_hingeamp025_temp092", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B260b-SimpleFastTaskGeometry-h160-learnableP-trainProbeP-direct050-fusedProjGradAdamW-absdiag050-manualAdamW-classbranch-fixedgain-gainramp075-identitytailquad030-hingeamp025-temp093", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_trainprobeP_trainprobedirect050_fused_projgrad_adamw_manual_update_global_fixedgain_gainramp075_quad030_temp093_h160", 2, 160, "v12.8.3_B109_audited_train_probe_global_fixedgain_temp093_gainramp075_auc_confirm10_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_trainprobeP_trainprobedirect050_fusedquadprojadamw_manualadamw_classbranch_fixedgain_gainramp075_identitytailquad030_hingeamp025_temp093", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B261b-SimpleFastTaskGeometry-h160-learnableP-trainProbeP-signalBroad010-direct050-fusedProjGradAdamW-absdiag050-manualAdamW-classbranch-fixedgain-gainramp075-identitytailquad030-hingeamp025-temp090", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_trainprobeP_signalbroad010_trainprobedirect050_fused_projgrad_adamw_manual_update_global_fixedgain_gainramp075_quad030_temp090_h160", 2, 160, "v12.8.3_B109_trainprobe_broad_signal010_lossagnostic_projection_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_trainprobeP_signalbroad010_trainprobedirect050_fusedquadprojadamw_manualadamw_classbranch_fixedgain_gainramp075_identitytailquad030_hingeamp025_temp090", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B262b-SimpleFastTaskGeometry-h160-learnableP-trainProbeP-signalBroad025-direct050-fusedProjGradAdamW-absdiag050-manualAdamW-classbranch-fixedgain-gainramp075-identitytailquad030-hingeamp025-temp090", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_trainprobeP_signalbroad025_trainprobedirect050_fused_projgrad_adamw_manual_update_global_fixedgain_gainramp075_quad030_temp090_h160", 2, 160, "v12.8.3_B109_trainprobe_broad_signal025_lossagnostic_projection_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_trainprobeP_signalbroad025_trainprobedirect050_fusedquadprojadamw_manualadamw_classbranch_fixedgain_gainramp075_identitytailquad030_hingeamp025_temp090", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B263b-SimpleFastTaskGeometry-h160-learnableP-trainProbeP-signalBroad015-direct050-fusedProjGradAdamW-absdiag050-manualAdamW-classbranch-fixedgain-gainramp075-identitytailquad030-hingeamp025-temp090", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_trainprobeP_signalbroad015_trainprobedirect050_fused_projgrad_adamw_manual_update_global_fixedgain_gainramp075_quad030_temp090_h160", 2, 160, "v12.8.3_B109_trainprobe_broad_signal015_lossagnostic_projection_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_trainprobeP_signalbroad015_trainprobedirect050_fusedquadprojadamw_manualadamw_classbranch_fixedgain_gainramp075_identitytailquad030_hingeamp025_temp090", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B264b-SimpleFastTaskGeometry-h160-learnableP-trainProbeP-signalBroad050-direct050-fusedProjGradAdamW-absdiag050-manualAdamW-classbranch-fixedgain-gainramp075-identitytailquad030-hingeamp025-temp090", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_trainprobeP_signalbroad050_trainprobedirect050_fused_projgrad_adamw_manual_update_global_fixedgain_gainramp075_quad030_temp090_h160", 2, 160, "v12.8.3_B109_trainprobe_broad_signal050_lossagnostic_projection_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_trainprobeP_signalbroad050_trainprobedirect050_fusedquadprojadamw_manualadamw_classbranch_fixedgain_gainramp075_identitytailquad030_hingeamp025_temp090", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B265b-SimpleFastTaskGeometry-h160-learnableP-trainProbeP-signalBroad050-signalBlock025-direct050-fusedProjGradAdamW-absdiag050-manualAdamW-classbranch-fixedgain-gainramp075-identitytailquad030-hingeamp025-temp090", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_trainprobeP_signalbroad050_signalblock025_trainprobedirect050_fused_projgrad_adamw_manual_update_global_fixedgain_gainramp075_quad030_temp090_h160", 2, 160, "v12.8.3_B109_trainprobe_broad_signal050_block025_lossagnostic_projection_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_trainprobeP_signalbroad050_signalblock025_trainprobedirect050_fusedquadprojadamw_manualadamw_classbranch_fixedgain_gainramp075_identitytailquad030_hingeamp025_temp090", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B266b-SimpleFastTaskGeometry-h160-learnableP-trainProbeP-signalBroad050-signalBlock050-direct050-fusedProjGradAdamW-absdiag050-manualAdamW-classbranch-fixedgain-gainramp075-identitytailquad030-hingeamp025-temp090", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_trainprobeP_signalbroad050_signalblock050_trainprobedirect050_fused_projgrad_adamw_manual_update_global_fixedgain_gainramp075_quad030_temp090_h160", 2, 160, "v12.8.3_B109_trainprobe_broad_signal050_block050_lossagnostic_projection_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_trainprobeP_signalbroad050_signalblock050_trainprobedirect050_fusedquadprojadamw_manualadamw_classbranch_fixedgain_gainramp075_identitytailquad030_hingeamp025_temp090", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B267b-SimpleFastTaskGeometry-h160-learnableP-trainProbeP-signalBroad050-signalBlock015-direct050-fusedProjGradAdamW-absdiag050-manualAdamW-classbranch-fixedgain-gainramp075-identitytailquad030-hingeamp025-temp090", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_trainprobeP_signalbroad050_signalblock015_trainprobedirect050_fused_projgrad_adamw_manual_update_global_fixedgain_gainramp075_quad030_temp090_h160", 2, 160, "v12.8.3_B109_trainprobe_broad_signal050_block015_lossagnostic_projection_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_trainprobeP_signalbroad050_signalblock015_trainprobedirect050_fusedquadprojadamw_manualadamw_classbranch_fixedgain_gainramp075_identitytailquad030_hingeamp025_temp090", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B268b-SimpleFastTaskGeometry-h160-learnableP-trainProbeP-signalBroad050-signalBlock020-direct050-fusedProjGradAdamW-absdiag050-manualAdamW-classbranch-fixedgain-gainramp075-identitytailquad030-hingeamp025-temp090", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_trainprobeP_signalbroad050_signalblock020_trainprobedirect050_fused_projgrad_adamw_manual_update_global_fixedgain_gainramp075_quad030_temp090_h160", 2, 160, "v12.8.3_B109_trainprobe_broad_signal050_block020_lossagnostic_projection_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_trainprobeP_signalbroad050_signalblock020_trainprobedirect050_fusedquadprojadamw_manualadamw_classbranch_fixedgain_gainramp075_identitytailquad030_hingeamp025_temp090", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B269b-SimpleFastTaskGeometry-h160-learnableP-trainProbeP-signalBroad040-signalBlock015-direct050-fusedProjGradAdamW-absdiag050-manualAdamW-classbranch-fixedgain-gainramp075-identitytailquad030-hingeamp025-temp090", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_trainprobeP_signalbroad040_signalblock015_trainprobedirect050_fused_projgrad_adamw_manual_update_global_fixedgain_gainramp075_quad030_temp090_h160", 2, 160, "v12.8.3_B109_trainprobe_broad_signal040_block015_lossagnostic_projection_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_trainprobeP_signalbroad040_signalblock015_trainprobedirect050_fusedquadprojadamw_manualadamw_classbranch_fixedgain_gainramp075_identitytailquad030_hingeamp025_temp090", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B270b-SimpleFastTaskGeometry-h160-learnableP-trainProbeP-signalBroad035-signalBlock015-direct050-fusedProjGradAdamW-absdiag050-manualAdamW-classbranch-fixedgain-gainramp075-identitytailquad030-hingeamp025-temp090", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_trainprobeP_signalbroad035_signalblock015_trainprobedirect050_fused_projgrad_adamw_manual_update_global_fixedgain_gainramp075_quad030_temp090_h160", 2, 160, "v12.8.3_B109_trainprobe_broad_signal035_block015_lossagnostic_projection_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_trainprobeP_signalbroad035_signalblock015_trainprobedirect050_fusedquadprojadamw_manualadamw_classbranch_fixedgain_gainramp075_identitytailquad030_hingeamp025_temp090", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B271b-SimpleFastTaskGeometry-h160-learnableP-trainProbeP-signalBroad040-signalBlock015-logitBNSG100-fixedbranch-direct050-fusedProjGradAdamW-absdiag050-manualAdamW-classbranch-fixedgain-gainramp075-identitytailquad030-hingeamp025-temp090", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_trainprobeP_signalbroad040_signalblock015_logitbnsg100_fixedbranch_trainprobedirect050_fused_projgrad_adamw_manual_update_global_fixedgain_gainramp075_quad030_temp090_h160", 2, 160, "v12.8.3_B109_signalblock015_batch_stopgrad_logitnorm100_lossagnostic_ece_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_trainprobeP_signalbroad040_signalblock015_logitbnsg100_fixedbranch_trainprobedirect050_fusedquadprojadamw_manualadamw_classbranch_fixedgain_gainramp075_identitytailquad030_hingeamp025_temp090", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B272b-SimpleFastTaskGeometry-h160-learnableP-trainProbeP-signalBroad040-signalBlock015-logitBNSG150-fixedbranch-direct050-fusedProjGradAdamW-absdiag050-manualAdamW-classbranch-fixedgain-gainramp075-identitytailquad030-hingeamp025-temp090", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_trainprobeP_signalbroad040_signalblock015_logitbnsg150_fixedbranch_trainprobedirect050_fused_projgrad_adamw_manual_update_global_fixedgain_gainramp075_quad030_temp090_h160", 2, 160, "v12.8.3_B109_signalblock015_batch_stopgrad_logitnorm150_lossagnostic_ece_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_trainprobeP_signalbroad040_signalblock015_logitbnsg150_fixedbranch_trainprobedirect050_fusedquadprojadamw_manualadamw_classbranch_fixedgain_gainramp075_identitytailquad030_hingeamp025_temp090", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B273b-SimpleFastTaskGeometry-h160-learnableP-trainProbeP-signalBroad040-signalBlock015-logitcap350-direct050-fusedProjGradAdamW-absdiag050-manualAdamW-classbranch-fixedgain-gainramp075-identitytailquad030-hingeamp025-temp090", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_trainprobeP_signalbroad040_signalblock015_logitcap350_trainprobedirect050_fused_projgrad_adamw_manual_update_global_fixedgain_gainramp075_quad030_temp090_h160", 2, 160, "v12.8.3_B109_signalblock015_softcap350_lossagnostic_ece_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_trainprobeP_signalbroad040_signalblock015_logitcap350_trainprobedirect050_fusedquadprojadamw_manualadamw_classbranch_fixedgain_gainramp075_identitytailquad030_hingeamp025_temp090", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B274b-SimpleFastTaskGeometry-h160-learnableP-trainProbeP-signalBroad040-signalBlock015-logitcap300-direct050-fusedProjGradAdamW-absdiag050-manualAdamW-classbranch-fixedgain-gainramp075-identitytailquad030-hingeamp025-temp090", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_trainprobeP_signalbroad040_signalblock015_logitcap300_trainprobedirect050_fused_projgrad_adamw_manual_update_global_fixedgain_gainramp075_quad030_temp090_h160", 2, 160, "v12.8.3_B109_signalblock015_softcap300_lossagnostic_ece_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_trainprobeP_signalbroad040_signalblock015_logitcap300_trainprobedirect050_fusedquadprojadamw_manualadamw_classbranch_fixedgain_gainramp075_identitytailquad030_hingeamp025_temp090", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B275b-SimpleFastTaskGeometry-h160-learnableP-trainProbeP-signalBroad040-signalBlock015-fixedbranch-direct050-fusedProjGradAdamW-absdiag050-manualAdamW-classbranch-fixedgain-gainramp075-identitytailquad030-hingeamp025-temp090", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_trainprobeP_signalbroad040_signalblock015_fixedbranch_trainprobedirect050_fused_projgrad_adamw_manual_update_global_fixedgain_gainramp075_quad030_temp090_h160", 2, 160, "v12.8.3_B109_signalblock015_fixedbranch_lossagnostic_ece_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_trainprobeP_signalbroad040_signalblock015_fixedbranch_trainprobedirect050_fusedquadprojadamw_manualadamw_classbranch_fixedgain_gainramp075_identitytailquad030_hingeamp025_temp090", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B276b-SimpleFastTaskGeometry-h160-learnableP-trainProbeP-signalBroad035-signalBlock015-fixedbranch-direct050-fusedProjGradAdamW-absdiag050-manualAdamW-classbranch-fixedgain-gainramp075-identitytailquad030-hingeamp025-temp090", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_trainprobeP_signalbroad035_signalblock015_fixedbranch_trainprobedirect050_fused_projgrad_adamw_manual_update_global_fixedgain_gainramp075_quad030_temp090_h160", 2, 160, "v12.8.3_B109_signalblock015_broad035_fixedbranch_lossagnostic_ece_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_trainprobeP_signalbroad035_signalblock015_fixedbranch_trainprobedirect050_fusedquadprojadamw_manualadamw_classbranch_fixedgain_gainramp075_identitytailquad030_hingeamp025_temp090", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B277b-SimpleFastTaskGeometry-h160-learnableP-trainProbeP-signalBroad040-signalBlock015-logitcap500-direct050-fusedProjGradAdamW-absdiag050-manualAdamW-classbranch-fixedgain-gainramp075-identitytailquad030-hingeamp025-temp090", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_trainprobeP_signalbroad040_signalblock015_logitcap500_trainprobedirect050_fused_projgrad_adamw_manual_update_global_fixedgain_gainramp075_quad030_temp090_h160", 2, 160, "v12.8.3_B109_signalblock015_softcap500_lossagnostic_ece_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_trainprobeP_signalbroad040_signalblock015_logitcap500_trainprobedirect050_fusedquadprojadamw_manualadamw_classbranch_fixedgain_gainramp075_identitytailquad030_hingeamp025_temp090", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B278b-SimpleFastTaskGeometry-h160-learnableP-trainProbeP-signalBroad040-signalBlock015-logitcap400-direct050-fusedProjGradAdamW-absdiag050-manualAdamW-classbranch-fixedgain-gainramp075-identitytailquad030-hingeamp025-temp090", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_trainprobeP_signalbroad040_signalblock015_logitcap400_trainprobedirect050_fused_projgrad_adamw_manual_update_global_fixedgain_gainramp075_quad030_temp090_h160", 2, 160, "v12.8.3_B109_signalblock015_softcap400_lossagnostic_ece_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_trainprobeP_signalbroad040_signalblock015_logitcap400_trainprobedirect050_fusedquadprojadamw_manualadamw_classbranch_fixedgain_gainramp075_identitytailquad030_hingeamp025_temp090", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B279b-SimpleFastTaskGeometry-h160-learnableP-trainProbeP-signalBroad040-signalBlock015-boundQ-direct050-fusedProjGradAdamW-absdiag050-manualAdamW-classbranch-fixedgain-gainramp075-identitytailquad030-hingeamp025-temp090", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_trainprobeP_signalbroad040_signalblock015_boundq_trainprobedirect050_fused_projgrad_adamw_manual_update_global_fixedgain_gainramp075_quad030_temp090_h160", 2, 160, "v12.8.3_B109_signalblock015_boundq_lossagnostic_ece_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_trainprobeP_signalbroad040_signalblock015_boundq_trainprobedirect050_fusedquadprojadamw_manualadamw_classbranch_fixedgain_gainramp075_identitytailquad030_hingeamp025_temp090", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B280b-SimpleFastTaskGeometry-h160-learnableP-trainProbeP-signalBroad040-signalBlock015-rmsQ-direct050-fusedProjGradAdamW-absdiag050-manualAdamW-classbranch-fixedgain-gainramp075-identitytailquad030-hingeamp025-temp090", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_trainprobeP_signalbroad040_signalblock015_rmsq_trainprobedirect050_fused_projgrad_adamw_manual_update_global_fixedgain_gainramp075_quad030_temp090_h160", 2, 160, "v12.8.3_B109_signalblock015_rmsq_lossagnostic_ece_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_trainprobeP_signalbroad040_signalblock015_rmsq_trainprobedirect050_fusedquadprojadamw_manualadamw_classbranch_fixedgain_gainramp075_identitytailquad030_hingeamp025_temp090", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B281b-SimpleFastTaskGeometry-h160-learnableP-trainProbeP-signalBroad040-signalBlock015-readInit075-direct050-fusedProjGradAdamW-absdiag050-manualAdamW-classbranch-fixedgain-gainramp075-identitytailquad030-hingeamp025-temp090", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_trainprobeP_signalbroad040_signalblock015_readinit075_trainprobedirect050_fused_projgrad_adamw_manual_update_global_fixedgain_gainramp075_quad030_temp090_h160", 2, 160, "v12.8.3_B109_signalblock015_readout_init075_lossagnostic_ece_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_trainprobeP_signalbroad040_signalblock015_readinit075_trainprobedirect050_fusedquadprojadamw_manualadamw_classbranch_fixedgain_gainramp075_identitytailquad030_hingeamp025_temp090", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B282b-SimpleFastTaskGeometry-h160-learnableP-trainProbeP-signalBroad040-signalBlock015-readInit050-direct050-fusedProjGradAdamW-absdiag050-manualAdamW-classbranch-fixedgain-gainramp075-identitytailquad030-hingeamp025-temp090", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_trainprobeP_signalbroad040_signalblock015_readinit050_trainprobedirect050_fused_projgrad_adamw_manual_update_global_fixedgain_gainramp075_quad030_temp090_h160", 2, 160, "v12.8.3_B109_signalblock015_readout_init050_lossagnostic_ece_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_trainprobeP_signalbroad040_signalblock015_readinit050_trainprobedirect050_fusedquadprojadamw_manualadamw_classbranch_fixedgain_gainramp075_identitytailquad030_hingeamp025_temp090", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B283b-SimpleFastTaskGeometry-h160-learnableP-trainProbeP-signalBroad040-signalBlock015-quadReadInit050-direct050-fusedProjGradAdamW-absdiag050-manualAdamW-classbranch-fixedgain-gainramp075-identitytailquad030-hingeamp025-temp090", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_trainprobeP_signalbroad040_signalblock015_quadreadinit050_trainprobedirect050_fused_projgrad_adamw_manual_update_global_fixedgain_gainramp075_quad030_temp090_h160", 2, 160, "v12.8.3_B109_signalblock015_quad_readout_init050_lossagnostic_ece_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_trainprobeP_signalbroad040_signalblock015_quadreadinit050_trainprobedirect050_fusedquadprojadamw_manualadamw_classbranch_fixedgain_gainramp075_identitytailquad030_hingeamp025_temp090", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B284b-SimpleFastTaskGeometry-h160-learnableP-trainProbeP-signalBroad040-signalBlock015-readInit075-normAbs010-direct050-fusedProjGradAdamW-absdiag050-manualAdamW-classbranch-fixedgain-gainramp075-identitytailquad030-hingeamp025-temp090", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_normabs010_trainprobeP_signalbroad040_signalblock015_readinit075_trainprobedirect050_fused_projgrad_adamw_manual_update_global_fixedgain_gainramp075_quad030_temp090_h160", 2, 160, "v12.8.3_B109_B281_plus_weak_normabs_lossagnostic_ece_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_normabs010_trainprobeP_signalbroad040_signalblock015_readinit075_trainprobedirect050_fusedquadprojadamw_manualadamw_classbranch_fixedgain_gainramp075_identitytailquad030_hingeamp025_temp090", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B285b-SimpleFastTaskGeometry-h160-learnableP-trainProbeP-signalBroad040-signalBlock015-readInit075-meanStat010-direct050-fusedProjGradAdamW-absdiag050-manualAdamW-classbranch-fixedgain-gainramp075-identitytailquad030-hingeamp025-temp090", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_meanstat010_trainprobeP_signalbroad040_signalblock015_readinit075_trainprobedirect050_fused_projgrad_adamw_manual_update_global_fixedgain_gainramp075_quad030_temp090_h160", 2, 160, "v12.8.3_B109_B281_plus_weak_meanstat_lossagnostic_ece_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_meanstat010_trainprobeP_signalbroad040_signalblock015_readinit075_trainprobedirect050_fusedquadprojadamw_manualadamw_classbranch_fixedgain_gainramp075_identitytailquad030_hingeamp025_temp090", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B286b-SimpleFastTaskGeometry-h160-learnableP-trainProbeP-signalBroad040-signalBlock015-readInit075-normAbs010-meanStat010-direct050-fusedProjGradAdamW-absdiag050-manualAdamW-classbranch-fixedgain-gainramp075-identitytailquad030-hingeamp025-temp090", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_normabs010_meanstat010_trainprobeP_signalbroad040_signalblock015_readinit075_trainprobedirect050_fused_projgrad_adamw_manual_update_global_fixedgain_gainramp075_quad030_temp090_h160", 2, 160, "v12.8.3_B109_B281_plus_weak_normabs_meanstat_lossagnostic_ece_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_normabs010_meanstat010_trainprobeP_signalbroad040_signalblock015_readinit075_trainprobedirect050_fusedquadprojadamw_manualadamw_classbranch_fixedgain_gainramp075_identitytailquad030_hingeamp025_temp090", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B287b-SimpleFastTaskGeometry-h160-learnableP-trainProbeP-signalBroad040-signalBlock015-readInit125-direct050-fusedProjGradAdamW-absdiag050-manualAdamW-classbranch-fixedgain-gainramp075-identitytailquad030-hingeamp025-temp090", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_trainprobeP_signalbroad040_signalblock015_readinit125_trainprobedirect050_fused_projgrad_adamw_manual_update_global_fixedgain_gainramp075_quad030_temp090_h160", 2, 160, "v12.8.3_B109_signalblock015_readout_init125_lossagnostic_ece_direction_check_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_trainprobeP_signalbroad040_signalblock015_readinit125_trainprobedirect050_fusedquadprojadamw_manualadamw_classbranch_fixedgain_gainramp075_identitytailquad030_hingeamp025_temp090", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B288b-SimpleFastTaskGeometry-h160-learnableP-trainProbeP-signalBroad040-signalBlock015-directReadInit125-direct050-fusedProjGradAdamW-absdiag050-manualAdamW-classbranch-fixedgain-gainramp075-identitytailquad030-hingeamp025-temp090", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_trainprobeP_signalbroad040_signalblock015_directreadinit125_trainprobedirect050_fused_projgrad_adamw_manual_update_global_fixedgain_gainramp075_quad030_temp090_h160", 2, 160, "v12.8.3_B109_signalblock015_direct_readout_init125_lossagnostic_ece_direction_check_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_trainprobeP_signalbroad040_signalblock015_directreadinit125_trainprobedirect050_fusedquadprojadamw_manualadamw_classbranch_fixedgain_gainramp075_identitytailquad030_hingeamp025_temp090", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B289b-SimpleFastTaskGeometry-h160-learnableP-trainProbeP-signalBroad040-signalBlock015-quadReadInit125-direct050-fusedProjGradAdamW-absdiag050-manualAdamW-classbranch-fixedgain-gainramp075-identitytailquad030-hingeamp025-temp090", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_trainprobeP_signalbroad040_signalblock015_quadreadinit125_trainprobedirect050_fused_projgrad_adamw_manual_update_global_fixedgain_gainramp075_quad030_temp090_h160", 2, 160, "v12.8.3_B109_signalblock015_quad_readout_init125_lossagnostic_ece_direction_check_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_trainprobeP_signalbroad040_signalblock015_quadreadinit125_trainprobedirect050_fusedquadprojadamw_manualadamw_classbranch_fixedgain_gainramp075_identitytailquad030_hingeamp025_temp090", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B290b-SimpleFastTaskGeometry-h160-learnableP-trainProbeP-signalBroad035-signalBlock015-readInit125-direct050-fusedProjGradAdamW-absdiag050-manualAdamW-classbranch-fixedgain-gainramp075-identitytailquad030-hingeamp025-temp090", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_trainprobeP_signalbroad035_signalblock015_readinit125_trainprobedirect050_fused_projgrad_adamw_manual_update_global_fixedgain_gainramp075_quad030_temp090_h160", 2, 160, "v12.8.3_B109_readinit125_broad035_block015_lossagnostic_auc_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_trainprobeP_signalbroad035_signalblock015_readinit125_trainprobedirect050_fusedquadprojadamw_manualadamw_classbranch_fixedgain_gainramp075_identitytailquad030_hingeamp025_temp090", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B291b-SimpleFastTaskGeometry-h160-learnableP-trainProbeP-signalBroad050-signalBlock020-readInit125-direct050-fusedProjGradAdamW-absdiag050-manualAdamW-classbranch-fixedgain-gainramp075-identitytailquad030-hingeamp025-temp090", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_trainprobeP_signalbroad050_signalblock020_readinit125_trainprobedirect050_fused_projgrad_adamw_manual_update_global_fixedgain_gainramp075_quad030_temp090_h160", 2, 160, "v12.8.3_B109_readinit125_broad050_block020_lossagnostic_auc_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_trainprobeP_signalbroad050_signalblock020_readinit125_trainprobedirect050_fusedquadprojadamw_manualadamw_classbranch_fixedgain_gainramp075_identitytailquad030_hingeamp025_temp090", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B292b-SimpleFastTaskGeometry-h160-learnableP-trainProbeP-signalBroad040-signalBlock020-readInit125-direct050-fusedProjGradAdamW-absdiag050-manualAdamW-classbranch-fixedgain-gainramp075-identitytailquad030-hingeamp025-temp090", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_trainprobeP_signalbroad040_signalblock020_readinit125_trainprobedirect050_fused_projgrad_adamw_manual_update_global_fixedgain_gainramp075_quad030_temp090_h160", 2, 160, "v12.8.3_B109_readinit125_broad040_block020_lossagnostic_auc_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_trainprobeP_signalbroad040_signalblock020_readinit125_trainprobedirect050_fusedquadprojadamw_manualadamw_classbranch_fixedgain_gainramp075_identitytailquad030_hingeamp025_temp090", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B293b-SimpleFastTaskGeometry-h160-learnableP-trainProbeP-signalBroad035-signalBlock015-quadReadInit125-direct050-fusedProjGradAdamW-absdiag050-manualAdamW-classbranch-fixedgain-gainramp075-identitytailquad030-hingeamp025-temp090", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_trainprobeP_signalbroad035_signalblock015_quadreadinit125_trainprobedirect050_fused_projgrad_adamw_manual_update_global_fixedgain_gainramp075_quad030_temp090_h160", 2, 160, "v12.8.3_B109_quadreadinit125_broad035_block015_lossagnostic_auc_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_trainprobeP_signalbroad035_signalblock015_quadreadinit125_trainprobedirect050_fusedquadprojadamw_manualadamw_classbranch_fixedgain_gainramp075_identitytailquad030_hingeamp025_temp090", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B294b-SimpleFastTaskGeometry-h160-learnableP-trainProbeP-signalBroad050-signalBlock020-quadReadInit125-direct050-fusedProjGradAdamW-absdiag050-manualAdamW-classbranch-fixedgain-gainramp075-identitytailquad030-hingeamp025-temp090", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_trainprobeP_signalbroad050_signalblock020_quadreadinit125_trainprobedirect050_fused_projgrad_adamw_manual_update_global_fixedgain_gainramp075_quad030_temp090_h160", 2, 160, "v12.8.3_B109_quadreadinit125_broad050_block020_lossagnostic_auc_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_trainprobeP_signalbroad050_signalblock020_quadreadinit125_trainprobedirect050_fusedquadprojadamw_manualadamw_classbranch_fixedgain_gainramp075_identitytailquad030_hingeamp025_temp090", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B295b-SimpleFastTaskGeometry-h160-learnableP-trainProbeP-signalBroad040-signalBlock020-quadReadInit125-direct050-fusedProjGradAdamW-absdiag050-manualAdamW-classbranch-fixedgain-gainramp075-identitytailquad030-hingeamp025-temp090", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_trainprobeP_signalbroad040_signalblock020_quadreadinit125_trainprobedirect050_fused_projgrad_adamw_manual_update_global_fixedgain_gainramp075_quad030_temp090_h160", 2, 160, "v12.8.3_B109_quadreadinit125_broad040_block020_lossagnostic_auc_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_trainprobeP_signalbroad040_signalblock020_quadreadinit125_trainprobedirect050_fusedquadprojadamw_manualadamw_classbranch_fixedgain_gainramp075_identitytailquad030_hingeamp025_temp090", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B296b-SimpleFastTaskGeometry-h160-learnableP-trainProbeP-signalBroad025-signalBlock015-quadReadInit125-direct050-fusedProjGradAdamW-absdiag050-manualAdamW-classbranch-fixedgain-gainramp075-identitytailquad030-hingeamp025-temp090", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_trainprobeP_signalbroad025_signalblock015_quadreadinit125_trainprobedirect050_fused_projgrad_adamw_manual_update_global_fixedgain_gainramp075_quad030_temp090_h160", 2, 160, "v12.8.3_B109_quadreadinit125_broad025_block015_lossagnostic_auc_bracket_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_trainprobeP_signalbroad025_signalblock015_quadreadinit125_trainprobedirect050_fusedquadprojadamw_manualadamw_classbranch_fixedgain_gainramp075_identitytailquad030_hingeamp025_temp090", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B297b-SimpleFastTaskGeometry-h160-learnableP-trainProbeP-signalBroad025-signalBlock020-quadReadInit125-direct050-fusedProjGradAdamW-absdiag050-manualAdamW-classbranch-fixedgain-gainramp075-identitytailquad030-hingeamp025-temp090", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_trainprobeP_signalbroad025_signalblock020_quadreadinit125_trainprobedirect050_fused_projgrad_adamw_manual_update_global_fixedgain_gainramp075_quad030_temp090_h160", 2, 160, "v12.8.3_B109_quadreadinit125_broad025_block020_lossagnostic_auc_bracket_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_trainprobeP_signalbroad025_signalblock020_quadreadinit125_trainprobedirect050_fusedquadprojadamw_manualadamw_classbranch_fixedgain_gainramp075_identitytailquad030_hingeamp025_temp090", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B298b-SimpleFastTaskGeometry-h160-learnableP-trainProbeP-signalBroad035-signalBlock025-quadReadInit125-direct050-fusedProjGradAdamW-absdiag050-manualAdamW-classbranch-fixedgain-gainramp075-identitytailquad030-hingeamp025-temp090", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_trainprobeP_signalbroad035_signalblock025_quadreadinit125_trainprobedirect050_fused_projgrad_adamw_manual_update_global_fixedgain_gainramp075_quad030_temp090_h160", 2, 160, "v12.8.3_B109_quadreadinit125_broad035_block025_lossagnostic_auc_bracket_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_trainprobeP_signalbroad035_signalblock025_quadreadinit125_trainprobedirect050_fusedquadprojadamw_manualadamw_classbranch_fixedgain_gainramp075_identitytailquad030_hingeamp025_temp090", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B299b-SimpleFastTaskGeometry-h160-learnableP-trainProbeP-signalBroad030-signalBlock015-quadReadInit125-direct050-fusedProjGradAdamW-absdiag050-manualAdamW-classbranch-fixedgain-gainramp075-identitytailquad030-hingeamp025-temp090", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_trainprobeP_signalbroad030_signalblock015_quadreadinit125_trainprobedirect050_fused_projgrad_adamw_manual_update_global_fixedgain_gainramp075_quad030_temp090_h160", 2, 160, "v12.8.3_B109_quadreadinit125_broad030_block015_lossagnostic_auc_bracket_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_trainprobeP_signalbroad030_signalblock015_quadreadinit125_trainprobedirect050_fusedquadprojadamw_manualadamw_classbranch_fixedgain_gainramp075_identitytailquad030_hingeamp025_temp090", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B300b-SimpleFastTaskGeometry-h160-learnableP-trainProbeP-signalBroad032-signalBlock015-quadReadInit125-direct050-fusedProjGradAdamW-absdiag050-manualAdamW-classbranch-fixedgain-gainramp075-identitytailquad030-hingeamp025-temp090", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_trainprobeP_signalbroad032_signalblock015_quadreadinit125_trainprobedirect050_fused_projgrad_adamw_manual_update_global_fixedgain_gainramp075_quad030_temp090_h160", 2, 160, "v12.8.3_B109_quadreadinit125_broad032_block015_lossagnostic_auc_bracket_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_trainprobeP_signalbroad032_signalblock015_quadreadinit125_trainprobedirect050_fusedquadprojadamw_manualadamw_classbranch_fixedgain_gainramp075_identitytailquad030_hingeamp025_temp090", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B301b-SimpleFastTaskGeometry-h160-learnableP-trainProbeP-signalBroad035-signalBlock010-quadReadInit125-direct050-fusedProjGradAdamW-absdiag050-manualAdamW-classbranch-fixedgain-gainramp075-identitytailquad030-hingeamp025-temp090", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_trainprobeP_signalbroad035_signalblock010_quadreadinit125_trainprobedirect050_fused_projgrad_adamw_manual_update_global_fixedgain_gainramp075_quad030_temp090_h160", 2, 160, "v12.8.3_B109_quadreadinit125_broad035_block010_lossagnostic_auc_bracket_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_trainprobeP_signalbroad035_signalblock010_quadreadinit125_trainprobedirect050_fusedquadprojadamw_manualadamw_classbranch_fixedgain_gainramp075_identitytailquad030_hingeamp025_temp090", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B302b-SimpleFastTaskGeometry-h160-learnableP-trainProbeP-signalBroad035-signalBlock015-quadReadInit125-quadRamp010-direct050-fusedProjGradAdamW-absdiag050-manualAdamW-classbranch-fixedgain-gainramp075-identitytailquad030-hingeamp025-temp090", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_trainprobeP_signalbroad035_signalblock015_quadreadinit125_quadramp010_trainprobedirect050_fused_projgrad_adamw_manual_update_global_fixedgain_gainramp075_quad030_temp090_h160", 2, 160, "v12.8.3_B109_quadreadinit125_quad_branch_ramp010_lossagnostic_auc_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_trainprobeP_signalbroad035_signalblock015_quadreadinit125_quadramp010_trainprobedirect050_fusedquadprojadamw_manualadamw_classbranch_fixedgain_gainramp075_identitytailquad030_hingeamp025_temp090", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B303b-SimpleFastTaskGeometry-h160-learnableP-trainProbeP-signalBroad035-signalBlock015-quadReadInit125-quadRamp015-direct050-fusedProjGradAdamW-absdiag050-manualAdamW-classbranch-fixedgain-gainramp075-identitytailquad030-hingeamp025-temp090", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_trainprobeP_signalbroad035_signalblock015_quadreadinit125_quadramp015_trainprobedirect050_fused_projgrad_adamw_manual_update_global_fixedgain_gainramp075_quad030_temp090_h160", 2, 160, "v12.8.3_B109_quadreadinit125_quad_branch_ramp015_lossagnostic_auc_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_trainprobeP_signalbroad035_signalblock015_quadreadinit125_quadramp015_trainprobedirect050_fusedquadprojadamw_manualadamw_classbranch_fixedgain_gainramp075_identitytailquad030_hingeamp025_temp090", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B304b-SimpleFastTaskGeometry-h160-learnableP-trainProbeP-signalBroad035-signalBlock015-quadReadInit125-quadRamp020-direct050-fusedProjGradAdamW-absdiag050-manualAdamW-classbranch-fixedgain-gainramp075-identitytailquad030-hingeamp025-temp090", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_trainprobeP_signalbroad035_signalblock015_quadreadinit125_quadramp020_trainprobedirect050_fused_projgrad_adamw_manual_update_global_fixedgain_gainramp075_quad030_temp090_h160", 2, 160, "v12.8.3_B109_quadreadinit125_quad_branch_ramp020_lossagnostic_auc_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_trainprobeP_signalbroad035_signalblock015_quadreadinit125_quadramp020_trainprobedirect050_fusedquadprojadamw_manualadamw_classbranch_fixedgain_gainramp075_identitytailquad030_hingeamp025_temp090", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B305b-SimpleFastTaskGeometry-h160-learnableP-trainProbeP-signalBroad035-signalBlock015-quadReadInit125-quad020-direct050-fusedProjGradAdamW-absdiag050-manualAdamW-classbranch-fixedgain-gainramp075-identitytailquad020-hingeamp025-temp090", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_trainprobeP_signalbroad035_signalblock015_quadreadinit125_trainprobedirect050_fused_projgrad_adamw_manual_update_global_fixedgain_gainramp075_quad020_temp090_h160", 2, 160, "v12.8.3_B109_quadreadinit125_lower_quad_branch_lossagnostic_auc_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_trainprobeP_signalbroad035_signalblock015_quadreadinit125_trainprobedirect050_fusedquadprojadamw_manualadamw_classbranch_fixedgain_gainramp075_identitytailquad020_hingeamp025_temp090", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B306b-SimpleFastTaskGeometry-h160-learnableP-trainProbeP-signalBroad035-signalBlock015-quadReadInit125-direct075-fusedProjGradAdamW-absdiag050-manualAdamW-classbranch-fixedgain-gainramp075-identitytailquad030-hingeamp025-temp090", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_trainprobeP_signalbroad035_signalblock015_quadreadinit125_trainprobedirect075_fused_projgrad_adamw_manual_update_global_fixedgain_gainramp075_quad030_temp090_h160", 2, 160, "v12.8.3_B109_quadreadinit125_direct_signal_boost_lossagnostic_auc_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_trainprobeP_signalbroad035_signalblock015_quadreadinit125_trainprobedirect075_fusedquadprojadamw_manualadamw_classbranch_fixedgain_gainramp075_identitytailquad030_hingeamp025_temp090", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B307b-SimpleFastTaskGeometry-h160-learnableP-trainProbeP-signalBroad035-signalBlock015-quadReadInit125-direct060-fusedProjGradAdamW-absdiag050-manualAdamW-classbranch-fixedgain-gainramp075-identitytailquad030-hingeamp025-temp090", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_trainprobeP_signalbroad035_signalblock015_quadreadinit125_trainprobedirect060_fused_projgrad_adamw_manual_update_global_fixedgain_gainramp075_quad030_temp090_h160", 2, 160, "v12.8.3_B109_quadreadinit125_direct060_signal_bracket_lossagnostic_auc_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_trainprobeP_signalbroad035_signalblock015_quadreadinit125_trainprobedirect060_fusedquadprojadamw_manualadamw_classbranch_fixedgain_gainramp075_identitytailquad030_hingeamp025_temp090", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B308b-SimpleFastTaskGeometry-h160-learnableP-trainProbeP-signalBroad035-signalBlock015-quadReadInit125-direct065-fusedProjGradAdamW-absdiag050-manualAdamW-classbranch-fixedgain-gainramp075-identitytailquad030-hingeamp025-temp090", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_trainprobeP_signalbroad035_signalblock015_quadreadinit125_trainprobedirect065_fused_projgrad_adamw_manual_update_global_fixedgain_gainramp075_quad030_temp090_h160", 2, 160, "v12.8.3_B109_quadreadinit125_direct065_signal_bracket_lossagnostic_auc_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_trainprobeP_signalbroad035_signalblock015_quadreadinit125_trainprobedirect065_fusedquadprojadamw_manualadamw_classbranch_fixedgain_gainramp075_identitytailquad030_hingeamp025_temp090", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B309b-SimpleFastTaskGeometry-h160-learnableP-trainProbeP-signalBroad035-signalBlock015-quadReadInit125-directRamp100-direct050-fusedProjGradAdamW-absdiag050-manualAdamW-classbranch-fixedgain-gainramp075-identitytailquad030-hingeamp025-temp090", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_trainprobeP_signalbroad035_signalblock015_quadreadinit125_directramp100_trainprobedirect050_fused_projgrad_adamw_manual_update_global_fixedgain_gainramp075_quad030_temp090_h160", 2, 160, "v12.8.3_B109_quadreadinit125_direct_branch_ramp100_lossagnostic_auc_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_trainprobeP_signalbroad035_signalblock015_quadreadinit125_directramp100_trainprobedirect050_fusedquadprojadamw_manualadamw_classbranch_fixedgain_gainramp075_identitytailquad030_hingeamp025_temp090", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B310b-SimpleFastTaskGeometry-h160-learnableP-trainProbeP-signalBroad035-signalBlock015-quadReadInit125-directRamp115-direct050-fusedProjGradAdamW-absdiag050-manualAdamW-classbranch-fixedgain-gainramp075-identitytailquad030-hingeamp025-temp090", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_trainprobeP_signalbroad035_signalblock015_quadreadinit125_directramp115_trainprobedirect050_fused_projgrad_adamw_manual_update_global_fixedgain_gainramp075_quad030_temp090_h160", 2, 160, "v12.8.3_B109_quadreadinit125_direct_branch_ramp115_lossagnostic_auc_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_trainprobeP_signalbroad035_signalblock015_quadreadinit125_directramp115_trainprobedirect050_fusedquadprojadamw_manualadamw_classbranch_fixedgain_gainramp075_identitytailquad030_hingeamp025_temp090", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B311b-SimpleFastTaskGeometry-h160-learnableP-trainProbeP-signalBroad035-signalBlock015-quadReadInit125-directRamp125-direct050-fusedProjGradAdamW-absdiag050-manualAdamW-classbranch-fixedgain-gainramp075-identitytailquad030-hingeamp025-temp090", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_trainprobeP_signalbroad035_signalblock015_quadreadinit125_directramp125_trainprobedirect050_fused_projgrad_adamw_manual_update_global_fixedgain_gainramp075_quad030_temp090_h160", 2, 160, "v12.8.3_B109_quadreadinit125_direct_branch_ramp125_lossagnostic_auc_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_trainprobeP_signalbroad035_signalblock015_quadreadinit125_directramp125_trainprobedirect050_fusedquadprojadamw_manualadamw_classbranch_fixedgain_gainramp075_identitytailquad030_hingeamp025_temp090", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B312b-SimpleFastTaskGeometry-h160-learnableP-trainProbeP-signalBroad035-signalBlock015-quadReadInit125-directRamp130-direct050-fusedProjGradAdamW-absdiag050-manualAdamW-classbranch-fixedgain-gainramp075-identitytailquad030-hingeamp025-temp090", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_trainprobeP_signalbroad035_signalblock015_quadreadinit125_directramp130_trainprobedirect050_fused_projgrad_adamw_manual_update_global_fixedgain_gainramp075_quad030_temp090_h160", 2, 160, "v12.8.3_B109_quadreadinit125_direct_branch_ramp130_lossagnostic_auc_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_trainprobeP_signalbroad035_signalblock015_quadreadinit125_directramp130_trainprobedirect050_fusedquadprojadamw_manualadamw_classbranch_fixedgain_gainramp075_identitytailquad030_hingeamp025_temp090", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B313b-SimpleFastTaskGeometry-h160-learnableP-trainProbeP-signalBroad035-signalBlock015-quadReadInit125-directRamp135-direct050-fusedProjGradAdamW-absdiag050-manualAdamW-classbranch-fixedgain-gainramp075-identitytailquad030-hingeamp025-temp090", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_trainprobeP_signalbroad035_signalblock015_quadreadinit125_directramp135_trainprobedirect050_fused_projgrad_adamw_manual_update_global_fixedgain_gainramp075_quad030_temp090_h160", 2, 160, "v12.8.3_B109_quadreadinit125_direct_branch_fixed135_lossagnostic_auc_repair_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_trainprobeP_signalbroad035_signalblock015_quadreadinit125_directramp135_trainprobedirect050_fusedquadprojadamw_manualadamw_classbranch_fixedgain_gainramp075_identitytailquad030_hingeamp025_temp090", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B314b-SimpleFastTaskGeometry-h160-learnableP-trainProbeP-signalBroad035-signalBlock015-quadReadInit125-directRamp105-direct050-fusedProjGradAdamW-absdiag050-manualAdamW-classbranch-fixedgain-gainramp075-identitytailquad030-hingeamp025-temp090", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_trainprobeP_signalbroad035_signalblock015_quadreadinit125_directramp105_trainprobedirect050_fused_projgrad_adamw_manual_update_global_fixedgain_gainramp075_quad030_temp090_h160", 2, 160, "v12.8.3_B109_quadreadinit125_direct_branch_ramp105_lossagnostic_auc_confirm_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_trainprobeP_signalbroad035_signalblock015_quadreadinit125_directramp105_trainprobedirect050_fusedquadprojadamw_manualadamw_classbranch_fixedgain_gainramp075_identitytailquad030_hingeamp025_temp090", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B315b-SimpleFastTaskGeometry-h160-learnableP-trainProbeP-signalBroad035-signalBlock015-quadReadInit125-directRamp110-direct050-fusedProjGradAdamW-absdiag050-manualAdamW-classbranch-fixedgain-gainramp075-identitytailquad030-hingeamp025-temp090", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_trainprobeP_signalbroad035_signalblock015_quadreadinit125_directramp110_trainprobedirect050_fused_projgrad_adamw_manual_update_global_fixedgain_gainramp075_quad030_temp090_h160", 2, 160, "v12.8.3_B109_quadreadinit125_direct_branch_ramp110_lossagnostic_auc_confirm_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_trainprobeP_signalbroad035_signalblock015_quadreadinit125_directramp110_trainprobedirect050_fusedquadprojadamw_manualadamw_classbranch_fixedgain_gainramp075_identitytailquad030_hingeamp025_temp090", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B316b-SimpleFastTaskGeometry-h160-learnableP-trainProbeP-signalBroad035-signalBlock015-quadReadInit125-directRamp106-direct050-fusedProjGradAdamW-absdiag050-manualAdamW-classbranch-fixedgain-gainramp075-identitytailquad030-hingeamp025-temp090", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_trainprobeP_signalbroad035_signalblock015_quadreadinit125_directramp106_trainprobedirect050_fused_projgrad_adamw_manual_update_global_fixedgain_gainramp075_quad030_temp090_h160", 2, 160, "v12.9_B314_strict_auc_direct_branch_ramp106_lossagnostic_bracket_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_trainprobeP_signalbroad035_signalblock015_quadreadinit125_directramp106_trainprobedirect050_fusedquadprojadamw_manualadamw_classbranch_fixedgain_gainramp075_identitytailquad030_hingeamp025_temp090", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B317b-SimpleFastTaskGeometry-h160-learnableP-trainProbeP-signalBroad035-signalBlock015-quadReadInit125-directRamp108-direct050-fusedProjGradAdamW-absdiag050-manualAdamW-classbranch-fixedgain-gainramp075-identitytailquad030-hingeamp025-temp090", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_trainprobeP_signalbroad035_signalblock015_quadreadinit125_directramp108_trainprobedirect050_fused_projgrad_adamw_manual_update_global_fixedgain_gainramp075_quad030_temp090_h160", 2, 160, "v12.9_B314_strict_auc_direct_branch_ramp108_lossagnostic_bracket_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_trainprobeP_signalbroad035_signalblock015_quadreadinit125_directramp108_trainprobedirect050_fusedquadprojadamw_manualadamw_classbranch_fixedgain_gainramp075_identitytailquad030_hingeamp025_temp090", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B318b-SimpleFastTaskGeometry-h160-learnableP-trainProbeP-signalBroad035-signalBlock015-quadReadInit125-directRamp105-logitcap500-direct050-fusedProjGradAdamW-absdiag050-manualAdamW-classbranch-fixedgain-gainramp075-identitytailquad030-hingeamp025-temp090", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_trainprobeP_signalbroad035_signalblock015_quadreadinit125_directramp105_logitcap500_trainprobedirect050_fused_projgrad_adamw_manual_update_global_fixedgain_gainramp075_quad030_temp090_h160", 2, 160, "v12.9_B314_strict_auc_tail_softcap500_lossagnostic_bracket_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_trainprobeP_signalbroad035_signalblock015_quadreadinit125_directramp105_logitcap500_trainprobedirect050_fusedquadprojadamw_manualadamw_classbranch_fixedgain_gainramp075_identitytailquad030_hingeamp025_temp090", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B319b-SimpleFastTaskGeometry-h160-learnableP-trainProbeP-signalBroad035-signalBlock015-quadReadInit125-directRamp105-logitcap400-direct050-fusedProjGradAdamW-absdiag050-manualAdamW-classbranch-fixedgain-gainramp075-identitytailquad030-hingeamp025-temp090", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_trainprobeP_signalbroad035_signalblock015_quadreadinit125_directramp105_logitcap400_trainprobedirect050_fused_projgrad_adamw_manual_update_global_fixedgain_gainramp075_quad030_temp090_h160", 2, 160, "v12.9_B314_strict_auc_tail_softcap400_lossagnostic_bracket_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_trainprobeP_signalbroad035_signalblock015_quadreadinit125_directramp105_logitcap400_trainprobedirect050_fusedquadprojadamw_manualadamw_classbranch_fixedgain_gainramp075_identitytailquad030_hingeamp025_temp090", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B320b-SimpleFastTaskGeometry-h160-learnableP-trainProbeP-signalBroad035-signalBlock015-quadReadInit125-directRamp105-direct050-fusedProjGradAdamW-absdiag050-manualAdamW-classbranch-fixedgain-gainramp075-identitytailquad030-hingeamp025-temp075", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_trainprobeP_signalbroad035_signalblock015_quadreadinit125_directramp105_trainprobedirect050_fused_projgrad_adamw_manual_update_global_fixedgain_gainramp075_quad030_temp075_h160", 2, 160, "v12.9_B314_strict_auc_global_gain075_tail_ce_bracket_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_trainprobeP_signalbroad035_signalblock015_quadreadinit125_directramp105_trainprobedirect050_fusedquadprojadamw_manualadamw_classbranch_fixedgain_gainramp075_identitytailquad030_hingeamp025_temp075", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B321b-SimpleFastTaskGeometry-h160-learnableP-trainProbeP-signalBroad035-signalBlock015-quadReadInit125-directRamp105-direct050-fusedProjGradAdamW-absdiag050-manualAdamW-classbranch-fixedgain-gainramp075-identitytailquad030-hingeamp025-temp085", "SimpleFastTaskGeometry", "hingeamp025_direct_absdiag050_trainprobeP_signalbroad035_signalblock015_quadreadinit125_directramp105_trainprobedirect050_fused_projgrad_adamw_manual_update_global_fixedgain_gainramp075_quad030_temp085_h160", 2, 160, "v12.9_B314_strict_auc_global_gain085_tail_ce_bracket_h160", 1, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="hinge_absdiag050_trainprobeP_signalbroad035_signalblock015_quadreadinit125_directramp105_trainprobedirect050_fusedquadprojadamw_manualadamw_classbranch_fixedgain_gainramp075_identitytailquad030_hingeamp025_temp085", model_kind="simple_fast_task_geometry"),
        PrimitiveSpec("B4a-FourierKAN-lowfreq-K4", "Fourier", "fourier_lowfreq", 4, h(4), "native+awesome-kan-family", 0, 1, 0, 1, 0, basis_order=4),
        PrimitiveSpec("B4b-FourierKAN-lowfreq-K2-stream", "Fourier", "fourier_lowfreq", 2, h(2), "v12.8.3_fourier_k2_stream_family_microbench", 0, 1, 0, 1, 0, uses_dense_basis_tensor=0, basis_order=2),
        PrimitiveSpec("B4c-FourierKAN-lowfreq-K2-flatgemm-manual", "Fourier", "fourier_lowfreq", 2, h(2), "v12.8.3_fourier_k2_flat_gemm_manual_l3_repair", 0, 1, 0, 1, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="fourier_k2_flatgemm_cache"),
        PrimitiveSpec("B4d-FourierKAN-lowfreq-K2-flatgemm-recompute", "Fourier", "fourier_lowfreq", 2, h(2), "v12.8.3_fourier_k2_flat_gemm_recompute_l3_workspace_repair", 0, 1, 0, 1, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="fourier_k2_flatgemm_recompute"),
        PrimitiveSpec("B4e-FourierKAN-lowfreq-K2-tritonL3", "Fourier", "fourier_lowfreq", 2, h(2), "v12.8.3_fourier_k2_triton_l3_fused_forward_grad_repair", 0, 1, 0, 1, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="fourier_k2_triton_l3"),
        PrimitiveSpec("B4f-FourierKAN-lowfreq-K2-tritonL3-blockH", "Fourier", "fourier_lowfreq", 2, h(2), "v12.8.3_fourier_k2_triton_l3_blockh_forward_parallel_repair", 0, 1, 0, 1, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="fourier_k2_triton_l3_blockh"),
        PrimitiveSpec("B4g-FourierKAN-lowfreq-K2-tritonL3-matmulTile", "Fourier", "fourier_lowfreq", 2, h(2), "v12.8.3_fourier_k2_triton_l3_batch_hidden_matmul_tile_repair", 0, 1, 0, 1, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="fourier_k2_triton_l3_matmul"),
        PrimitiveSpec("B4h-FourierKAN-lowfreq-K3-tritonL3-matmulTile", "Fourier", "fourier_lowfreq", 3, h(3), "v12.8.3_fourier_k3_triton_l3_cos_expression_repair", 0, 1, 0, 1, 0, uses_dense_basis_tensor=0, basis_order=3, init_variant="fourier_k3_triton_l3_matmul"),
        PrimitiveSpec("B4i-FourierKAN-lowfreq-K3-h96-tritonL3-matmulTile", "Fourier", "fourier_lowfreq", 3, 96, "v12.8.3_fourier_k3_triton_l3_h96_efficiency_repair", 0, 1, 0, 1, 0, uses_dense_basis_tensor=0, basis_order=3, init_variant="fourier_k3_triton_l3_matmul"),
        PrimitiveSpec("B4j-FourierKAN-lowfreq-K3-h80-tritonL3-matmulTile", "Fourier", "fourier_lowfreq", 3, 80, "v12.8.3_fourier_k3_triton_l3_h80_efficiency_repair", 0, 1, 0, 1, 0, uses_dense_basis_tensor=0, basis_order=3, init_variant="fourier_k3_triton_l3_matmul"),
        PrimitiveSpec("B4k-FourierKAN-lowfreq-K3-h64-tritonL3-matmulTile", "Fourier", "fourier_lowfreq", 3, 64, "v12.8.3_fourier_k3_triton_l3_h64_efficiency_floor_repair", 0, 1, 0, 1, 0, uses_dense_basis_tensor=0, basis_order=3, init_variant="fourier_k3_triton_l3_matmul"),
        PrimitiveSpec("B4l-FourierKAN-lowfreq-K4-h64-tritonL3-matmulTile", "Fourier", "fourier_lowfreq", 4, 64, "v12.8.3_fourier_k4_triton_l3_h64_expression_repair", 0, 1, 0, 1, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="fourier_k4_triton_l3_matmul"),
        PrimitiveSpec("B4m-FourierKAN-lowfreq-K4-h48-tritonL3-matmulTile", "Fourier", "fourier_lowfreq", 4, 48, "v12.8.3_fourier_k4_triton_l3_h48_efficiency_repair", 0, 1, 0, 1, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="fourier_k4_triton_l3_matmul"),
        PrimitiveSpec("B4n-FourierKAN-lowfreq-K4-h64-linearres-tritonL3-matmulTile", "Fourier", "fourier_lowfreq", 4, 64, "v12.8.3_fourier_k4_lowfreq_plus_linear_residual_repair", 0, 1, 0, 1, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="fourier_k4_linearres_triton_l3_matmul_linearres010"),
        PrimitiveSpec("B4o-FourierKAN-lowfreq-K4-h48-linearres-tritonL3-matmulTile", "Fourier", "fourier_lowfreq", 4, 48, "v12.8.3_fourier_k4_lowfreq_plus_linear_residual_h48_repair", 0, 1, 0, 1, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="fourier_k4_linearres_triton_l3_matmul_linearres010"),
        PrimitiveSpec("B4p-FourierKAN-lowfreq-K4-h64-linearres-gemmDirectL3", "Fourier", "fourier_lowfreq", 4, 64, "v12.8.3_fourier_k4_linear_residual_gemm_direct_grad_repair", 0, 1, 0, 1, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="fourier_k4_linearres_gemm_l3_matmul_linearres010"),
        PrimitiveSpec("B4q-FourierKAN-lowfreq-K4-h48-linearres-gemmDirectL3", "Fourier", "fourier_lowfreq", 4, 48, "v12.8.3_fourier_k4_linear_residual_gemm_direct_grad_h48_repair", 0, 1, 0, 1, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="fourier_k4_linearres_gemm_l3_matmul_linearres010"),
        PrimitiveSpec("B4r-FourierKAN-lowfreq-K4-h32-linearres-gemmDirectL3", "Fourier", "fourier_lowfreq", 4, 32, "v12.8.3_fourier_k4_linear_residual_gemm_direct_grad_h32_repair", 0, 1, 0, 1, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="fourier_k4_linearres_gemm_l3_matmul_linearres010"),
        PrimitiveSpec("B4s-FourierKAN-lowfreq-K4-h24-linearres-gemmDirectL3", "Fourier", "fourier_lowfreq", 4, 24, "v12.8.3_fourier_k4_linear_residual_gemm_direct_grad_h24_repair", 0, 1, 0, 1, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="fourier_k4_linearres_gemm_l3_matmul_linearres010"),
        PrimitiveSpec("B4t-FourierKAN-lowfreq-K4-h16-linearres-gemmDirectL3", "Fourier", "fourier_lowfreq", 4, 16, "v12.8.3_fourier_k4_linear_residual_gemm_direct_grad_h16_repair", 0, 1, 0, 1, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="fourier_k4_linearres_gemm_l3_matmul_linearres010"),
        PrimitiveSpec("B4u-FourierKAN-lowfreq-K4-h8-linearres-gemmDirectL3", "Fourier", "fourier_lowfreq", 4, 8, "v12.8.3_fourier_k4_linear_residual_gemm_direct_grad_h8_floor_repair", 0, 1, 0, 1, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="fourier_k4_linearres_gemm_l3_matmul_linearres010"),
        PrimitiveSpec("B4v-FourierKAN-lowfreq-K4-h8-linearres050-gemmDirectL3", "Fourier", "fourier_lowfreq", 4, 8, "v12.8.3_fourier_k4_h8_stronger_linear_residual_expression_repair", 0, 1, 0, 1, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="fourier_k4_linearres_gemm_l3_matmul_linearres050"),
        PrimitiveSpec("B4w-FourierKAN-lowfreq-K4-h8-linearres050-tritonL3-matmulTile", "Fourier", "fourier_lowfreq", 4, 8, "v12.8.3_fourier_k4_h8_linearres050_triton_residual_grad_repair", 0, 1, 0, 1, 0, uses_dense_basis_tensor=0, basis_order=4, init_variant="fourier_k4_linearres_triton_l3_matmul_linearres050"),
        PrimitiveSpec("B5c-RickerWaveletKAN-lite-K4", "Wavelet", "ricker_wavelet", 4, h(4), "native+wavelet-family", 1, 0, 1, 0, 0, basis_order=1),
        PrimitiveSpec("B5h-HatWaveletKAN-local-K4", "Wavelet", "hat_wavelet", 4, h(4), "v12.8.3_family_specific_hat_wavelet_local_support+v12.10_hat_wavelet_triton_l3", 1, 0, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=1, init_variant="hat_wavelet_k4_triton_l3_matmul"),
        PrimitiveSpec("B6r-BSpline-order1-stream-K2-repair", "BSpline", "bspline_order1", 2, h(2), "R1_repair_stream_basis_mix+third_party/KANbeFair", 1, 0, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=1),
        PrimitiveSpec("B6a-BSpline-order1-local-K4", "BSpline", "bspline_order1", 4, h(4), "third_party/KANbeFair", 1, 0, 0, 0, 0, basis_order=1),
        PrimitiveSpec("B6b-BSpline-order1-local-K8-expression-repair", "BSpline", "bspline_order1", 8, h(8), "R2_repair_increase_K+third_party/KANbeFair", 1, 0, 0, 0, 0, basis_order=1),
        PrimitiveSpec("B8d-BSpline-ReLU-lite-combo-K4-repair", "Hybrid", "bspline_relu_combo", 4, h(4), "R2_repair_lite_hybrid", 1, 1, 0, 0, 0, basis_order=1),
        PrimitiveSpec("B9a-Poly2SignedPair-stream-K3-repair", "HybridPoly", "poly2_relu_combo", 3, h(3), "R5_base_repair_signed_pair_poly2", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, basis_order=2, init_variant="signed_pair_linear"),
        PrimitiveSpec("B9b-Poly2SignedPair-h64-diagnostic-repair", "HybridPoly", "poly2_relu_combo", 3, 64, "R2_repair_rank_limited_signed_pair_poly2_diagnostic", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, diagnostic_only=1, basis_order=2, init_variant="signed_pair_linear"),
        PrimitiveSpec("B9c-Poly2SignedPair-h64-K2-diagnostic-repair", "HybridPoly", "poly2_relu_combo", 2, 64, "R2_repair_minimal_signed_pair_poly2_diagnostic", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, diagnostic_only=1, basis_order=2, init_variant="signed_pair_linear"),
        PrimitiveSpec("B9d-Poly2PairRandom-h64-K2-diagnostic-repair", "HybridPoly", "poly2_relu_combo", 2, 64, "R2_repair_global_pair_random_poly2_diagnostic", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, diagnostic_only=1, basis_order=2, init_variant="signed_pair_random_linear"),
        PrimitiveSpec("B10a-QuadraticSketch-h64-diagnostic", "QuadraticSketch", "quadratic_sketch", 2, 64, "R2_repair_fused_global_quadratic_sketch_diagnostic", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, diagnostic_only=1, basis_order=2, model_kind="quadratic_sketch"),
        PrimitiveSpec("B10b-QuadraticSketch-h128-diagnostic", "QuadraticSketch", "quadratic_sketch", 2, 128, "R2_repair_fused_global_quadratic_sketch_diagnostic", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, diagnostic_only=1, basis_order=2, model_kind="quadratic_sketch"),
        PrimitiveSpec("B10c-QuadraticSketch-h256-diagnostic", "QuadraticSketch", "quadratic_sketch", 2, 256, "R2_repair_high_rank_global_quadratic_sketch_diagnostic", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, diagnostic_only=1, basis_order=2, model_kind="quadratic_sketch"),
        PrimitiveSpec("B10d-QuadraticSketch-h512-diagnostic", "QuadraticSketch", "quadratic_sketch", 2, 512, "R2_repair_high_rank_global_quadratic_sketch_diagnostic", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, diagnostic_only=1, basis_order=2, model_kind="quadratic_sketch"),
        PrimitiveSpec("B11a-TrainableQuadraticSketch-h128", "QuadraticSketch", "quadratic_sketch", 2, 128, "R2_repair_trainable_global_quadratic_sketch", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, diagnostic_only=0, basis_order=2, model_kind="trainable_quadratic_sketch"),
        PrimitiveSpec("B11b-TrainableQuadraticSketch-h256", "QuadraticSketch", "quadratic_sketch", 2, 256, "R2_repair_trainable_global_quadratic_sketch", 0, 1, 0, 0, 0, uses_dense_basis_tensor=0, diagnostic_only=0, basis_order=2, model_kind="trainable_quadratic_sketch"),
        PrimitiveSpec("B7b-RationalKAT-flashgroup-G16-h112-linearres-tritonL3", "Rational", "rational_kat_flashgroup", 1, 112, "third_party/FlashKAT/rational_kat grouped rational activation repair", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearres", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7c-RationalKAT-flashgroup-G16-h80-linearres-tritonL3", "Rational", "rational_kat_flashgroup", 1, 80, "third_party/FlashKAT/rational_kat grouped rational activation h80 speed repair", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearres", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7d-RationalKAT-flashgroup-G16-h64-linearres-tritonL3", "Rational", "rational_kat_flashgroup", 1, 64, "third_party/FlashKAT/rational_kat grouped rational activation h64 speed repair", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearres", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7e-RationalKAT-flashgroup-G16-h64-linearres-inputcrossR64-tritonL3", "Rational", "rational_kat_flashgroup", 1, 64, "third_party/FlashKAT/rational_kat grouped rational h64 fixed input-cross R64 expression repair", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearres_inputcrossr64", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7f-RationalKAT-flashgroup-G16-h64-linearres-inputcrossR128-tritonL3", "Rational", "rational_kat_flashgroup", 1, 64, "third_party/FlashKAT/rational_kat grouped rational h64 fixed input-cross R128 expression repair", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearres_inputcrossr128", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7g-RationalKAT-flashgroup-G16-h64-linearres-inputcrossR96-tritonL3", "Rational", "rational_kat_flashgroup", 1, 64, "third_party/FlashKAT/rational_kat grouped rational h64 fixed input-cross R96 efficiency/expression bracket", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearres_inputcrossr96", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7h-RationalKAT-flashgroup-G16-h64-linearres-inputcrossR112-tritonL3", "Rational", "rational_kat_flashgroup", 1, 64, "third_party/FlashKAT/rational_kat grouped rational h64 fixed input-cross R112 efficiency/expression bracket", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearres_inputcrossr112", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7i-RationalKAT-flashgroup-G16-h64-linearres-paircrossR136-tritonL3", "Rational", "rational_kat_flashgroup", 1, 64, "third_party/FlashKAT/rational_kat grouped rational h64 structured pair-cross R136 expression repair", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearres_paircrossr136", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7j-RationalKAT-flashgroup-G16-h64-linearres-paircrossR96-tritonL3", "Rational", "rational_kat_flashgroup", 1, 64, "third_party/FlashKAT/rational_kat grouped rational h64 structured pair-cross R96 efficiency bracket", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearres_paircrossr96", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7k-RationalKAT-flashgroup-G16-h64-linearres-paircrossR88-tritonL3", "Rational", "rational_kat_flashgroup", 1, 64, "third_party/FlashKAT/rational_kat grouped rational h64 structured pair-cross R88 efficiency bracket", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearres_paircrossr88", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7l-RationalKAT-flashgroup-G16-h64-linearres-paircrossR80-tritonL3", "Rational", "rational_kat_flashgroup", 1, 64, "third_party/FlashKAT/rational_kat grouped rational h64 structured pair-cross R80 efficiency bracket", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearres_paircrossr80", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7m-RationalKAT-flashgroup-G16-h64-linearres-balancedPairR88-tritonL3", "Rational", "rational_kat_flashgroup", 1, 64, "third_party/FlashKAT/rational_kat grouped rational h64 balanced structured pair-cross R88 expression repair", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearres_balancedpaircrossr88", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7n-RationalKAT-flashgroup-G16-h64-linearres-balancedPairR96-tritonL3", "Rational", "rational_kat_flashgroup", 1, 64, "third_party/FlashKAT/rational_kat grouped rational h64 balanced structured pair-cross R96 expression/efficiency bracket", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearres_balancedpaircrossr96", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7o-RationalKAT-flashgroup-G16-h64-linearres-transformPairR88-tritonL3", "Rational", "rational_kat_flashgroup", 1, 64, "third_party/FlashKAT/rational_kat grouped rational h64 fixed transform pair-cross R88 rotated-expression repair", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearres_transformpairr88", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7p-RationalKAT-flashgroup-G16-h64-linearres-transformPairR96-tritonL3", "Rational", "rational_kat_flashgroup", 1, 64, "third_party/FlashKAT/rational_kat grouped rational h64 fixed transform pair-cross R96 rotated-expression bracket", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearres_transformpairr96", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7q-RationalKAT-flashgroup-G16-h56-linearres-paircrossR136-tritonL3", "Rational", "rational_kat_flashgroup", 1, 56, "third_party/FlashKAT/rational_kat grouped rational h56 structured pair-cross R136 full-coverage efficiency repair", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearres_paircrossr136", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7r-RationalKAT-flashgroup-G16-h48-linearres-paircrossR136-tritonL3", "Rational", "rational_kat_flashgroup", 1, 48, "third_party/FlashKAT/rational_kat grouped rational h48 structured pair-cross R136 full-coverage efficiency repair", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearres_paircrossr136", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7s-RationalKAT-flashgroup-G16-h40-linearres-paircrossR136-tritonL3", "Rational", "rational_kat_flashgroup", 1, 40, "third_party/FlashKAT/rational_kat grouped rational h40 structured pair-cross R136 full-coverage efficiency repair", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearres_paircrossr136", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7t-RationalKAT-flashgroup-G16-h32-linearres-paircrossR136-tritonL3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 structured pair-cross R136 full-coverage efficiency repair", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearres_paircrossr136", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7u-RationalKAT-flashgroup-G16-h40-linearres-gridPairR136-tritonL3", "Rational", "rational_kat_flashgroup", 1, 40, "third_party/FlashKAT/rational_kat grouped rational h40 grid-balanced pair-cross R136 task-trajectory repair", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearres_gridpaircrossr136", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7v-RationalKAT-flashgroup-G16-h32-linearres-gridPairR136-tritonL3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 grid-balanced pair-cross R136 task-trajectory repair", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearres_gridpaircrossr136", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7w-RationalKAT-flashgroup-G16-h40-linearres-paircrossR112-tritonL3", "Rational", "rational_kat_flashgroup", 1, 40, "third_party/FlashKAT/rational_kat grouped rational h40 structured pair-cross R112 task/expression bracket", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearres_paircrossr112", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7x-RationalKAT-flashgroup-G16-h32-linearres-paircrossR112-tritonL3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 structured pair-cross R112 task/expression bracket", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearres_paircrossr112", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7y-RationalKAT-flashgroup-G16-h40-linearres-paircrossR136-readoutLR300-tritonL3", "Rational", "rational_kat_flashgroup", 1, 40, "third_party/FlashKAT/rational_kat grouped rational h40 pair-cross R136 readout-lr task repair", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearres_paircrossr136_readoutlr300", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7z-RationalKAT-flashgroup-G16-h32-linearres-paircrossR136-readoutLR300-tritonL3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 pair-cross R136 readout-lr task repair", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearres_paircrossr136_readoutlr300", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7aa-RationalKAT-flashgroup-G16-h40-linearres-pairHiddenR136S025-tritonL3", "Rational", "rational_kat_flashgroup", 1, 40, "third_party/FlashKAT/rational_kat grouped rational h40 fixed pair-hidden R136 task-dynamics repair", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearres_pairhiddenr136s025", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7ab-RationalKAT-flashgroup-G16-h32-linearres-pairHiddenR136S025-tritonL3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 fixed pair-hidden R136 task-dynamics repair", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearres_pairhiddenr136s025", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7ac-RationalKAT-flashgroup-G16-h40-linearres-pairBucketR136S025-tritonL3", "Rational", "rational_kat_flashgroup", 1, 40, "third_party/FlashKAT/rational_kat grouped rational h40 bucketed pair-hidden R136 low-cost task-dynamics repair", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearres_pairbucketr136s025", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7ad-RationalKAT-flashgroup-G16-h32-linearres-pairBucketR136S025-tritonL3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 bucketed pair-hidden R136 low-cost task-dynamics repair", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearres_pairbucketr136s025", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7ae-RationalKAT-flashgroup-G16-h40-linearres-pairBucketR112S025-tritonL3", "Rational", "rational_kat_flashgroup", 1, 40, "third_party/FlashKAT/rational_kat grouped rational h40 bucketed pair-hidden R112 efficiency repair", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearres_pairbucketr112s025", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7af-RationalKAT-flashgroup-G16-h32-linearres-pairBucketR112S025-tritonL3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 bucketed pair-hidden R112 efficiency repair", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearres_pairbucketr112s025", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7ag-RationalKAT-flashgroup-G16-h40-linearres-pairBucketR88S025-tritonL3", "Rational", "rational_kat_flashgroup", 1, 40, "third_party/FlashKAT/rational_kat grouped rational h40 bucketed pair-hidden R88 efficiency/expression bracket", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearres_pairbucketr88s025", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7ah-RationalKAT-flashgroup-G16-h32-linearres-pairBucketR88S025-tritonL3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 bucketed pair-hidden R88 efficiency/expression bracket", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearres_pairbucketr88s025", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7ai-RationalKAT-flashgroup-G16-h40-linearres-pairBucketR136S025-coalescedTritonL3", "Rational", "rational_kat_flashgroup", 1, 40, "third_party/FlashKAT/rational_kat grouped rational h40 bucketed pair-hidden R136 coalesced Triton hidden-delta repair", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearres_pairbucketr136s025", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7aj-RationalKAT-flashgroup-G16-h32-linearres-pairBucketR136S025-coalescedTritonL3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 bucketed pair-hidden R136 coalesced Triton hidden-delta repair", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearres_pairbucketr136s025", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7ak-RationalKAT-flashgroup-G16-h40-linearres-pairBucketR112S025-coalescedTritonL3", "Rational", "rational_kat_flashgroup", 1, 40, "third_party/FlashKAT/rational_kat grouped rational h40 bucketed pair-hidden R112 coalesced Triton hidden-delta repair", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearres_pairbucketr112s025", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7al-RationalKAT-flashgroup-G16-h32-linearres-pairBucketR112S025-coalescedTritonL3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 bucketed pair-hidden R112 coalesced Triton hidden-delta repair", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearres_pairbucketr112s025", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7am-RationalKAT-flashgroup-G16-h40-linearres-pairBucketDirectR136S025-coalescedTritonL3", "Rational", "rational_kat_flashgroup", 1, 40, "third_party/FlashKAT/rational_kat grouped rational h40 bucketed hidden R136 plus direct pair readout", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearres_pairbucketdirectr136s025", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7an-RationalKAT-flashgroup-G16-h32-linearres-pairBucketDirectR136S025-coalescedTritonL3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 bucketed hidden R136 plus direct pair readout", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearres_pairbucketdirectr136s025", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7ao-RationalKAT-flashgroup-G16-h40-linearres-pairBucketDirectR112S025-coalescedTritonL3", "Rational", "rational_kat_flashgroup", 1, 40, "third_party/FlashKAT/rational_kat grouped rational h40 bucketed hidden R112 plus direct pair readout", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearres_pairbucketdirectr112s025", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7ap-RationalKAT-flashgroup-G16-h32-linearres-pairBucketDirectR112S025-coalescedTritonL3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 bucketed hidden R112 plus direct pair readout", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearres_pairbucketdirectr112s025", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7aq-RationalKAT-flashgroup-G16-h40-linearres-pairCrossR136-bucketHiddenR32S025-tritonL3", "Rational", "rational_kat_flashgroup", 1, 40, "third_party/FlashKAT/rational_kat grouped rational h40 direct pair R136 plus small bucket hidden R32", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearres_paircrossr136bucketr32s025", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7ar-RationalKAT-flashgroup-G16-h32-linearres-pairCrossR136-bucketHiddenR32S025-tritonL3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 direct pair R136 plus small bucket hidden R32", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearres_paircrossr136bucketr32s025", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7as-RationalKAT-flashgroup-G16-h40-linearres-pairCrossR136-bucketHiddenR64S025-tritonL3", "Rational", "rational_kat_flashgroup", 1, 40, "third_party/FlashKAT/rational_kat grouped rational h40 direct pair R136 plus small bucket hidden R64", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearres_paircrossr136bucketr64s025", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7at-RationalKAT-flashgroup-G16-h32-linearres-pairCrossR136-bucketHiddenR64S025-tritonL3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 direct pair R136 plus small bucket hidden R64", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearres_paircrossr136bucketr64s025", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7au-RationalKAT-flashgroup-G16-h40-linearres-paircrossR136-readscale025-tritonL3", "Rational", "rational_kat_flashgroup", 1, 40, "third_party/FlashKAT/rational_kat grouped rational h40 structured pair-cross R136 with fixed readout scale 0.25", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearres_paircrossr136_readscale025", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7av-RationalKAT-flashgroup-G16-h32-linearres-paircrossR136-readscale025-tritonL3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 structured pair-cross R136 with fixed readout scale 0.25", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearres_paircrossr136_readscale025", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7aw-RationalKAT-flashgroup-G16-h40-linearres-paircrossR136-readscale010-tritonL3", "Rational", "rational_kat_flashgroup", 1, 40, "third_party/FlashKAT/rational_kat grouped rational h40 structured pair-cross R136 with fixed readout scale 0.10", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearres_paircrossr136_readscale010", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7ax-RationalKAT-flashgroup-G16-h32-linearres-paircrossR136-readscale010-tritonL3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 structured pair-cross R136 with fixed readout scale 0.10", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearres_paircrossr136_readscale010", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7ay-RationalKAT-flashgroup-G16-h40-linearres-paircrossR136-readtritonL3", "Rational", "rational_kat_flashgroup", 1, 40, "third_party/FlashKAT/rational_kat grouped rational h40 structured pair-cross R136 with fused Triton readout logits/grad", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearres_paircrossr136_readtriton", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7az-RationalKAT-flashgroup-G16-h32-linearres-paircrossR136-readtritonL3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 structured pair-cross R136 with fused Triton readout logits/grad", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearres_paircrossr136_readtriton", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7ba-RationalKAT-flashgroup-G16-h40-linearres-paircrossR136-readtriton-readscale010-L3", "Rational", "rational_kat_flashgroup", 1, 40, "third_party/FlashKAT/rational_kat grouped rational h40 structured pair-cross R136 fused readout with scale 0.10", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearres_paircrossr136_readtriton_readscale010", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7bb-RationalKAT-flashgroup-G16-h32-linearres-paircrossR136-readtriton-readscale010-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 structured pair-cross R136 fused readout with scale 0.10", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearres_paircrossr136_readtriton_readscale010", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7bc-RationalKAT-flashgroup-G16-h40-linearres-paircrossR136-readblocktritonL3", "Rational", "rational_kat_flashgroup", 1, 40, "third_party/FlashKAT/rational_kat grouped rational h40 structured pair-cross R136 block-tiled Triton readout logits/grad", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearres_paircrossr136_readblocktriton", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7bd-RationalKAT-flashgroup-G16-h32-linearres-paircrossR136-readblocktritonL3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 structured pair-cross R136 block-tiled Triton readout logits/grad", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearres_paircrossr136_readblocktriton", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7be-RationalKAT-flashgroup-G16-h40-linearres-paircrossR136-readblocktriton-readscale010-L3", "Rational", "rational_kat_flashgroup", 1, 40, "third_party/FlashKAT/rational_kat grouped rational h40 structured pair-cross R136 block-tiled Triton readout with scale 0.10", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearres_paircrossr136_readblocktriton_readscale010", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7bf-RationalKAT-flashgroup-G16-h32-linearres-paircrossR136-readblocktriton-readscale010-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 structured pair-cross R136 block-tiled Triton readout with scale 0.10", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearres_paircrossr136_readblocktriton_readscale010", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7bg-RationalKAT-flashgroup-G16-h40-linearres-paircrossR136-readblocktriton-logitbiasL3", "Rational", "rational_kat_flashgroup", 1, 40, "third_party/FlashKAT/rational_kat grouped rational h40 block-tiled pair readout plus learnable logit bias", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearres_paircrossr136_readblocktriton_logitbias", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7bh-RationalKAT-flashgroup-G16-h32-linearres-paircrossR136-readblocktriton-logitbiasL3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 block-tiled pair readout plus learnable logit bias", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearres_paircrossr136_readblocktriton_logitbias", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7bi-RationalKAT-flashgroup-G16-h40-linearres-paircrossR136-readblocktriton-readscale010-logitbiasL3", "Rational", "rational_kat_flashgroup", 1, 40, "third_party/FlashKAT/rational_kat grouped rational h40 block-tiled pair readout scale 0.10 plus learnable logit bias", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearres_paircrossr136_readblocktriton_readscale010_logitbias", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7bj-RationalKAT-flashgroup-G16-h32-linearres-paircrossR136-readblocktriton-readscale010-logitbiasL3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 block-tiled pair readout scale 0.10 plus learnable logit bias", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearres_paircrossr136_readblocktriton_readscale010_logitbias", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7bk-RationalKAT-flashgroup-G16-h40-linearres-paircrossR136-readblocktriton-logitbiasfusedL3", "Rational", "rational_kat_flashgroup", 1, 40, "third_party/FlashKAT/rational_kat grouped rational h40 block-tiled pair readout with bias fused into the logits kernel", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearres_paircrossr136_readblocktriton_logitbiasfused", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7bl-RationalKAT-flashgroup-G16-h32-linearres-paircrossR136-readblocktriton-logitbiasfusedL3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 block-tiled pair readout with bias fused into the logits kernel", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearres_paircrossr136_readblocktriton_logitbiasfused", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7bm-RationalKAT-flashgroup-G16-h40-linearres-paircrossR136-readblocktriton-readscale010-logitbiasfusedL3", "Rational", "rational_kat_flashgroup", 1, 40, "third_party/FlashKAT/rational_kat grouped rational h40 block-tiled pair readout scale 0.10 with bias fused into the logits kernel", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearres_paircrossr136_readblocktriton_readscale010_logitbiasfused", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7bn-RationalKAT-flashgroup-G16-h32-linearres-paircrossR136-readblocktriton-readscale010-logitbiasfusedL3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 block-tiled pair readout scale 0.10 with bias fused into the logits kernel", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearres_paircrossr136_readblocktriton_readscale010_logitbiasfused", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7bo-RationalKAT-flashgroup-G16-h40-linearres-paircrossR112-readblocktritonL3", "Rational", "rational_kat_flashgroup", 1, 40, "third_party/FlashKAT/rational_kat grouped rational h40 block-tiled pair readout R112 efficiency repair", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearres_paircrossr112_readblocktriton", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7bp-RationalKAT-flashgroup-G16-h32-linearres-paircrossR112-readblocktritonL3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 block-tiled pair readout R112 efficiency repair", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearres_paircrossr112_readblocktriton", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7bq-RationalKAT-flashgroup-G16-h40-linearres-paircrossR112-readblocktriton-readscale010-L3", "Rational", "rational_kat_flashgroup", 1, 40, "third_party/FlashKAT/rational_kat grouped rational h40 block-tiled pair readout R112 with scale 0.10", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearres_paircrossr112_readblocktriton_readscale010", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7br-RationalKAT-flashgroup-G16-h32-linearres-paircrossR112-readblocktriton-readscale010-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 block-tiled pair readout R112 with scale 0.10", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearres_paircrossr112_readblocktriton_readscale010", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7bs-RationalKAT-flashgroup-G16-h40-linearres-paircrossR128-readblocktritonL3", "Rational", "rational_kat_flashgroup", 1, 40, "third_party/FlashKAT/rational_kat grouped rational h40 block-tiled pair readout R128 expression/efficiency bracket", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearres_paircrossr128_readblocktriton", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7bt-RationalKAT-flashgroup-G16-h32-linearres-paircrossR128-readblocktritonL3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 block-tiled pair readout R128 expression/efficiency bracket", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearres_paircrossr128_readblocktriton", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7bu-RationalKAT-flashgroup-G16-h40-linearres-paircrossR128-readblocktriton-readscale010-L3", "Rational", "rational_kat_flashgroup", 1, 40, "third_party/FlashKAT/rational_kat grouped rational h40 block-tiled pair readout R128 with scale 0.10", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearres_paircrossr128_readblocktriton_readscale010", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7bv-RationalKAT-flashgroup-G16-h32-linearres-paircrossR128-readblocktriton-readscale010-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 block-tiled pair readout R128 with scale 0.10", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearres_paircrossr128_readblocktriton_readscale010", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7bw-RationalKAT-flashgroup-G16-h40-linearres-paircrossR120-readblocktritonL3", "Rational", "rational_kat_flashgroup", 1, 40, "third_party/FlashKAT/rational_kat grouped rational h40 block-tiled pair readout R120 expression/efficiency bracket", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearres_paircrossr120_readblocktriton", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7bx-RationalKAT-flashgroup-G16-h32-linearres-paircrossR120-readblocktritonL3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 block-tiled pair readout R120 expression/efficiency bracket", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearres_paircrossr120_readblocktriton", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7by-RationalKAT-flashgroup-G16-h40-linearres-paircrossR120-readblocktriton-readscale010-L3", "Rational", "rational_kat_flashgroup", 1, 40, "third_party/FlashKAT/rational_kat grouped rational h40 block-tiled pair readout R120 with scale 0.10", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearres_paircrossr120_readblocktriton_readscale010", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7bz-RationalKAT-flashgroup-G16-h32-linearres-paircrossR120-readblocktriton-readscale010-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 block-tiled pair readout R120 with scale 0.10", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearres_paircrossr120_readblocktriton_readscale010", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7ca-RationalKAT-flashgroup-G16-h40-linearres-paircrossR136-readblocktriton-pairnormL3", "Rational", "rational_kat_flashgroup", 1, 40, "third_party/FlashKAT/rational_kat grouped rational h40 block-tiled pair readout R136 with fixed train-stream pair-feature normalization", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearres_paircrossr136_readblocktriton_pairnorm", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7cb-RationalKAT-flashgroup-G16-h32-linearres-paircrossR136-readblocktriton-pairnormL3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 block-tiled pair readout R136 with fixed train-stream pair-feature normalization", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearres_paircrossr136_readblocktriton_pairnorm", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7cc-RationalKAT-flashgroup-G16-h40-linearres-paircrossR120-readblocktriton-pairnormL3", "Rational", "rational_kat_flashgroup", 1, 40, "third_party/FlashKAT/rational_kat grouped rational h40 block-tiled pair readout R120 with fixed train-stream pair-feature normalization", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearres_paircrossr120_readblocktriton_pairnorm", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7cd-RationalKAT-flashgroup-G16-h32-linearres-paircrossR120-readblocktriton-pairnormL3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 block-tiled pair readout R120 with fixed train-stream pair-feature normalization", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearres_paircrossr120_readblocktriton_pairnorm", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7ce-RationalKAT-flashgroup-G16-h40-linearres-paircrossR136-readblocktriton-pairstdL3", "Rational", "rational_kat_flashgroup", 1, 40, "third_party/FlashKAT/rational_kat grouped rational h40 block-tiled pair readout R136 with fixed pair-feature std scaling but no centering", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearres_paircrossr136_readblocktriton_pairstd", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7cf-RationalKAT-flashgroup-G16-h32-linearres-paircrossR136-readblocktriton-pairstdL3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 block-tiled pair readout R136 with fixed pair-feature std scaling but no centering", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearres_paircrossr136_readblocktriton_pairstd", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7cg-RationalKAT-flashgroup-G16-h40-linearres-paircrossR120-readblocktriton-pairstdL3", "Rational", "rational_kat_flashgroup", 1, 40, "third_party/FlashKAT/rational_kat grouped rational h40 block-tiled pair readout R120 with fixed pair-feature std scaling but no centering", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearres_paircrossr120_readblocktriton_pairstd", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7ch-RationalKAT-flashgroup-G16-h32-linearres-paircrossR120-readblocktriton-pairstdL3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 block-tiled pair readout R120 with fixed pair-feature std scaling but no centering", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearres_paircrossr120_readblocktriton_pairstd", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7ci-RationalKAT-flashgroup-G16-h40-linearres-paircrossR136-readblocktriton-crossReadoutLR025-L3", "Rational", "rational_kat_flashgroup", 1, 40, "third_party/FlashKAT/rational_kat grouped rational h40 block-tiled pair readout R136 with cross-readout-only LR 0.25 trajectory repair", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearres_paircrossr136_readblocktriton_crossreadoutlr025", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7cj-RationalKAT-flashgroup-G16-h32-linearres-paircrossR136-readblocktriton-crossReadoutLR025-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 block-tiled pair readout R136 with cross-readout-only LR 0.25 trajectory repair", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearres_paircrossr136_readblocktriton_crossreadoutlr025", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7ck-RationalKAT-flashgroup-G16-h40-linearres-paircrossR136-readblocktriton-crossZero-L3", "Rational", "rational_kat_flashgroup", 1, 40, "third_party/FlashKAT/rational_kat grouped rational h40 block-tiled pair readout R136 with zero-initialized cross readout", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearres_paircrossr136_readblocktriton_crosszero", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7cl-RationalKAT-flashgroup-G16-h32-linearres-paircrossR136-readblocktriton-crossZero-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 block-tiled pair readout R136 with zero-initialized cross readout", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearres_paircrossr136_readblocktriton_crosszero", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7cm-RationalKAT-flashgroup-G16-h40-linearres-paircrossR128-readblocktriton-crossZero-L3", "Rational", "rational_kat_flashgroup", 1, 40, "third_party/FlashKAT/rational_kat grouped rational h40 block-tiled pair readout R128 with zero-initialized cross readout", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearres_paircrossr128_readblocktriton_crosszero", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7cn-RationalKAT-flashgroup-G16-h32-linearres-paircrossR128-readblocktriton-crossZero-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 block-tiled pair readout R128 with zero-initialized cross readout", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearres_paircrossr128_readblocktriton_crosszero", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7co-RationalKAT-flashgroup-G16-h40-linearres-paircrossR136-readblocktriton-crossZero-b32-L3", "Rational", "rational_kat_flashgroup", 1, 40, "third_party/FlashKAT/rational_kat grouped rational h40 R136 crosszero block-readout with B32 logits tiling", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearres_paircrossr136_readblocktriton_crosszero_readblockb32", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7cp-RationalKAT-flashgroup-G16-h32-linearres-paircrossR136-readblocktriton-crossZero-b32-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 R136 crosszero block-readout with B32 logits tiling", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearres_paircrossr136_readblocktriton_crosszero_readblockb32", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7do-RationalKAT-flashgroup-G16-h40-linearres-paircrossR136-readblocktriton-crossZero-b32-readscale025-L3", "Rational", "rational_kat_flashgroup", 1, 40, "third_party/FlashKAT/rational_kat grouped rational h40 R136 crosszero B32 block-readout with fixed pair-readout scale 0.25", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearres_paircrossr136_readblocktriton_crosszero_readblockb32_readscale025", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7dp-RationalKAT-flashgroup-G16-h32-linearres-paircrossR136-readblocktriton-crossZero-b32-readscale025-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 R136 crosszero B32 block-readout with fixed pair-readout scale 0.25", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearres_paircrossr136_readblocktriton_crosszero_readblockb32_readscale025", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7dq-RationalKAT-flashgroup-G16-h40-linearres-paircrossR136-readblocktriton-crossZero-b32-readscale010-L3", "Rational", "rational_kat_flashgroup", 1, 40, "third_party/FlashKAT/rational_kat grouped rational h40 R136 crosszero B32 block-readout with fixed pair-readout scale 0.10", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearres_paircrossr136_readblocktriton_crosszero_readblockb32_readscale010", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7dr-RationalKAT-flashgroup-G16-h32-linearres-paircrossR136-readblocktriton-crossZero-b32-readscale010-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 R136 crosszero B32 block-readout with fixed pair-readout scale 0.10", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearres_paircrossr136_readblocktriton_crosszero_readblockb32_readscale010", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7ds-RationalKAT-flashgroup-G16-h32-linearres-paircrossR136-readblocktriton-crossZero-b32-readscale025-crossWarm-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 R136 crosszero B32 block-readout with fixed pair-readout scale 0.25 and epoch warmup", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearres_paircrossr136_readblocktriton_crosszero_readblockb32_readscale025_crosswarm", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7dt-RationalKAT-flashgroup-G16-h32-linearres-paircrossR136-readblocktriton-crossZero-b32-readscale010-crossWarm-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 R136 crosszero B32 block-readout with fixed pair-readout scale 0.10 and epoch warmup", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearres_paircrossr136_readblocktriton_crosszero_readblockb32_readscale010_crosswarm", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7du-RationalKAT-flashgroup-G16-h40-linearres-paircrossR112-readblocktriton-crossZero-b32-L3", "Rational", "rational_kat_flashgroup", 1, 40, "third_party/FlashKAT/rational_kat grouped rational h40 R112 crosszero B32 block-readout lower-rank bracket", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearres_paircrossr112_readblocktriton_crosszero_readblockb32", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7dv-RationalKAT-flashgroup-G16-h32-linearres-paircrossR112-readblocktriton-crossZero-b32-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 R112 crosszero B32 block-readout lower-rank bracket", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearres_paircrossr112_readblocktriton_crosszero_readblockb32", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7dw-RationalKAT-flashgroup-G16-h40-linearres-paircrossR136-readblocktriton-crossZero-b32-freezeBackbone-L3", "Rational", "rational_kat_flashgroup", 1, 40, "third_party/FlashKAT/rational_kat grouped rational h40 R136 crosszero B32 block-readout with frozen rational/w1 backbone and readout-only VJP", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearres_paircrossr136_readblocktriton_crosszero_readblockb32_freezebackbone", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7dx-RationalKAT-flashgroup-G16-h32-linearres-paircrossR136-readblocktriton-crossZero-b32-freezeBackbone-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 R136 crosszero B32 block-readout with frozen rational/w1 backbone and readout-only VJP", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearres_paircrossr136_readblocktriton_crosszero_readblockb32_freezebackbone", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7dy-RationalKAT-flashgroup-G16-h40-linearresGain800-paircrossR136-readblocktriton-crossZero-b32-freezeBackbone-L3", "Rational", "rational_kat_flashgroup", 1, 40, "third_party/FlashKAT/rational_kat grouped rational h40 R136 crosszero B32 block-readout with frozen backbone and stronger loss-agnostic linear residual readout", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain800_paircrossr136_readblocktriton_crosszero_readblockb32_freezebackbone", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7dz-RationalKAT-flashgroup-G16-h32-linearresGain800-paircrossR136-readblocktriton-crossZero-b32-freezeBackbone-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 R136 crosszero B32 block-readout with frozen backbone and stronger loss-agnostic linear residual readout", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain800_paircrossr136_readblocktriton_crosszero_readblockb32_freezebackbone", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7ea-RationalKAT-flashgroup-G16-h32-linearresGain1600-paircrossR136-readblocktriton-crossZero-b32-freezeBackbone-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 R136 crosszero B32 frozen-backbone block-readout with linear residual gain 16.0", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain1600_paircrossr136_readblocktriton_crosszero_readblockb32_freezebackbone", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7eb-RationalKAT-flashgroup-G16-h32-linearresGain3200-paircrossR136-readblocktriton-crossZero-b32-freezeBackbone-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 R136 crosszero B32 frozen-backbone block-readout with linear residual gain 32.0", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain3200_paircrossr136_readblocktriton_crosszero_readblockb32_freezebackbone", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7ec-RationalKAT-flashgroup-G16-h32-linearresGain1600-paircrossR136-readblocktriton-crossZero-b32-freezeRational-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 R136 crosszero B32 block-readout with frozen rational coefficients, trainable w1, and linear residual gain 16.0", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain1600_paircrossr136_readblocktriton_crosszero_readblockb32_freezerational", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7ed-RationalKAT-flashgroup-G16-h32-linearresGain3200-paircrossR136-readblocktriton-crossZero-b32-freezeRational-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 R136 crosszero B32 block-readout with frozen rational coefficients, trainable w1, and linear residual gain 32.0", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain3200_paircrossr136_readblocktriton_crosszero_readblockb32_freezerational", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7ee-RationalKAT-flashgroup-G16-h32-linearresGain3200-paircrossR136-readblocktriton-crossZero-b32-freezeBackbone-hiddenBias-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 R136 crosszero B32 frozen-backbone block-readout with trainable hidden bias and linear residual gain 32.0", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain3200_paircrossr136_readblocktriton_crosszero_readblockb32_freezebackbone_hiddenbias", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7ef-RationalKAT-flashgroup-G16-h32-linearresGain6400-paircrossR136-readblocktriton-crossZero-b32-freezeBackbone-hiddenBias-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 R136 crosszero B32 frozen-backbone block-readout with trainable hidden bias and linear residual gain 64.0", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain6400_paircrossr136_readblocktriton_crosszero_readblockb32_freezebackbone_hiddenbias", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7eg-RationalKAT-flashgroup-G16-h32-linearresGain4800-paircrossR136-readblocktriton-crossZero-b32-freezeBackbone-hiddenBias-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 R136 crosszero B32 frozen-backbone block-readout with trainable hidden bias and linear residual gain 48.0", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain4800_paircrossr136_readblocktriton_crosszero_readblockb32_freezebackbone_hiddenbias", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7eh-RationalKAT-flashgroup-G16-h32-linearresGain9600-paircrossR136-readblocktriton-crossZero-b32-freezeBackbone-hiddenBias-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 R136 crosszero B32 frozen-backbone block-readout with trainable hidden bias and linear residual gain 96.0", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain9600_paircrossr136_readblocktriton_crosszero_readblockb32_freezebackbone_hiddenbias", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7ei-RationalKAT-flashgroup-G16-h32-linearresGain12800-paircrossR136-readblocktriton-crossZero-b32-freezeBackbone-hiddenBias-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 R136 crosszero B32 frozen-backbone block-readout with trainable hidden bias and linear residual gain 128.0", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain12800_paircrossr136_readblocktriton_crosszero_readblockb32_freezebackbone_hiddenbias", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7ej-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-freezeBackbone-hiddenBias-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 R136 crosszero B32 frozen-backbone block-readout with trainable hidden bias and linear residual gain 192.0", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_paircrossr136_readblocktriton_crosszero_readblockb32_freezebackbone_hiddenbias", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7ek-RationalKAT-flashgroup-G16-h32-linearresGain25600-paircrossR136-readblocktriton-crossZero-b32-freezeBackbone-hiddenBias-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 R136 crosszero B32 frozen-backbone block-readout with trainable hidden bias and linear residual gain 256.0", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain25600_paircrossr136_readblocktriton_crosszero_readblockb32_freezebackbone_hiddenbias", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7el-RationalKAT-flashgroup-G16-h32-linearresGain38400-paircrossR136-readblocktriton-crossZero-b32-freezeBackbone-hiddenBias-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 R136 crosszero B32 frozen-backbone block-readout with trainable hidden bias and linear residual gain 384.0", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain38400_paircrossr136_readblocktriton_crosszero_readblockb32_freezebackbone_hiddenbias", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7em-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-pairStd-crossZero-b32-freezeBackbone-hiddenBias-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 R136 crosszero B32 frozen-backbone hidden-bias path with train-stream pair-feature std scaling and linear residual gain 192.0", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_paircrossr136_readblocktriton_pairstd_crosszero_readblockb32_freezebackbone_hiddenbias", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7en-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-pairNorm-crossZero-b32-freezeBackbone-hiddenBias-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 R136 crosszero B32 frozen-backbone hidden-bias path with train-stream pair-feature centering/std scaling and linear residual gain 192.0", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_paircrossr136_readblocktriton_pairnorm_crosszero_readblockb32_freezebackbone_hiddenbias", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7eo-RationalKAT-flashgroup-G16-h32-linearresGain19200-linearGate100-paircrossR136-readblocktriton-crossZero-b32-freezeBackbone-hiddenBias-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 R136 crosszero B32 frozen-backbone hidden-bias path with trainable linear residual gate initialized at 1.00", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_lineargate100_paircrossr136_readblocktriton_crosszero_readblockb32_freezebackbone_hiddenbias", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7ep-RationalKAT-flashgroup-G16-h32-linearresGain19200-linearGate075-paircrossR136-readblocktriton-crossZero-b32-freezeBackbone-hiddenBias-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 R136 crosszero B32 frozen-backbone hidden-bias path with trainable linear residual gate initialized at 0.75", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_lineargate075_paircrossr136_readblocktriton_crosszero_readblockb32_freezebackbone_hiddenbias", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7eq-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-freezeBackbone-hiddenBias-logitBias-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 B7ej-shaped frozen-backbone hidden-bias path with trainable logit intercept", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_paircrossr136_readblocktriton_crosszero_readblockb32_freezebackbone_hiddenbias_logitbias", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7er-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-readscale050-freezeBackbone-hiddenBias-logitBias-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 B7ej-shaped frozen-backbone hidden-bias path with trainable logit intercept and fixed 0.50 pair-readout scale", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_paircrossr136_readblocktriton_crosszero_readblockb32_readscale050_freezebackbone_hiddenbias_logitbias", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7es-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-freezeBackbone-hiddenBias-w2BiasRow-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 B7ej-shaped path with output intercept fused into the final w2 row", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_paircrossr136_readblocktriton_crosszero_readblockb32_freezebackbone_hiddenbias_w2biasrow", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7et-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR120-readblocktriton-crossZero-b32-freezeBackbone-hiddenBias-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 B7ej-shaped path with cheaper fixed R120 pair-readout primitive", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_paircrossr120_readblocktriton_crosszero_readblockb32_freezebackbone_hiddenbias", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7eu-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR120-readblocktriton-crossZero-b32-freezeBackbone-hiddenBias-w2BiasRow-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 B7ej-shaped cheaper R120 pair primitive with output intercept fused into w2 row", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_paircrossr120_readblocktriton_crosszero_readblockb32_freezebackbone_hiddenbias_w2biasrow", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7ev-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 B7ej-shaped path with manual foreach AdamW update", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_paircrossr136_readblocktriton_crosszero_readblockb32_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7ew-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-freezeBackbone-hiddenBias-logitBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 B7eq-shaped path with logit intercept and manual foreach AdamW update", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_paircrossr136_readblocktriton_crosszero_readblockb32_freezebackbone_hiddenbias_logitbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7ex-RationalKAT-flashgroup-G16-h32-linearresGain19200-freezeBackbone-hiddenBias-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 frozen-backbone hidden-bias path without pair-readout reductions", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_freezebackbone_hiddenbias", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7ey-RationalKAT-flashgroup-G16-h32-linearresGain19200-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 frozen-backbone hidden-bias path without pair-readout reductions and with manual foreach AdamW update", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7ez-RationalKAT-flashgroup-G16-h32-linearresGain19200-pairReadBucketR136G64-crossZero-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 B7ej-shaped frozen-backbone hidden-bias path with R136 pair features bucketed to G64 and manual foreach AdamW update", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_pairreadbucketr136g64_crosszero_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7fa-RationalKAT-flashgroup-G16-h32-linearresGain19200-pairReadBucketR136G32-crossZero-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 B7ej-shaped frozen-backbone hidden-bias path with R136 pair features bucketed to G32 and manual foreach AdamW update", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_pairreadbucketr136g32_crosszero_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7fb-RationalKAT-flashgroup-G16-h32-linearresGain19200-pairReadBucketR136G96-crossZero-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 B7ej-shaped frozen-backbone hidden-bias path with R136 pair features bucketed to G96 and manual foreach AdamW update", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_pairreadbucketr136g96_crosszero_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7fc-RationalKAT-flashgroup-G16-h32-linearresGain19200-pairReadBucketR136G112-crossZero-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 B7ej-shaped frozen-backbone hidden-bias path with R136 pair features bucketed to G112 and manual foreach AdamW update", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_pairreadbucketr136g112_crosszero_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7fd-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR96-readblocktriton-crossZero-b32-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 B7ej-shaped frozen-backbone hidden-bias path with full R96 pair block-readout and manual foreach AdamW update", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_paircrossr96_readblocktriton_crosszero_readblockb32_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7fe-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR80-readblocktriton-crossZero-b32-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 B7ej-shaped frozen-backbone hidden-bias path with full R80 pair block-readout and manual foreach AdamW update", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_paircrossr80_readblocktriton_crosszero_readblockb32_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7ff-RationalKAT-flashgroup-G16-h32-linearresGain19200-balancedPairCrossR96-readblocktriton-crossZero-b32-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 B7ej-shaped frozen-backbone hidden-bias path with balanced R96 pair block-readout and manual foreach AdamW update", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_balancedpaircrossr96_readblocktriton_crosszero_readblockb32_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7fg-RationalKAT-flashgroup-G16-h32-linearresGain19200-balancedPairCrossR88-readblocktriton-crossZero-b32-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 B7ej-shaped frozen-backbone hidden-bias path with balanced R88 pair block-readout and manual foreach AdamW update", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_balancedpaircrossr88_readblocktriton_crosszero_readblockb32_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7fh-RationalKAT-flashgroup-G16-h32-linearresGain19200-projBilinR96-bilinZero-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 B7ej-shaped frozen-backbone hidden-bias path with fixed dual-DCT projection-bilinear R96 GEMM readout and manual foreach AdamW update", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_projbilinr96_bilinzero_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7fi-RationalKAT-flashgroup-G16-h32-linearresGain19200-projSqR96-sqZero-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 B7ej-shaped frozen-backbone hidden-bias path with fixed DCT projection-square R96 GEMM readout and manual foreach AdamW update", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_projsqr96_sqzero_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7fj-RationalKAT-flashgroup-G16-h32-linearresGain19200-projBilinR64-bilinZero-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 B7ej-shaped frozen-backbone hidden-bias path with fixed dual-DCT projection-bilinear R64 GEMM readout and manual foreach AdamW update", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_projbilinr64_bilinzero_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7fk-RationalKAT-flashgroup-G16-h32-linearresGain19200-projSqR64-sqZero-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 B7ej-shaped frozen-backbone hidden-bias path with fixed DCT projection-square R64 GEMM readout and manual foreach AdamW update", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_projsqr64_sqzero_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7fl-RationalKAT-flashgroup-G16-h32-linearresGain19200-projBilinR32-bilinZero-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 B7ej-shaped frozen-backbone hidden-bias path with fixed dual-DCT projection-bilinear R32 GEMM readout and manual foreach AdamW update", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_projbilinr32_bilinzero_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7fm-RationalKAT-flashgroup-G16-h32-linearresGain19200-projSqR32-sqZero-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 B7ej-shaped frozen-backbone hidden-bias path with fixed DCT projection-square R32 GEMM readout and manual foreach AdamW update", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_projsqr32_sqzero_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7fn-RationalKAT-flashgroup-G16-h32-linearresGain19200-hiddenSqReadout-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 B7ey-shaped frozen-backbone hidden-bias path with trainable hidden-square readout and manual foreach AdamW update", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_hiddensqreadout_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7fo-RationalKAT-flashgroup-G16-h32-linearresGain19200-hiddenSqReadoutZero-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 B7ey-shaped frozen-backbone hidden-bias path with zero-init hidden-square readout and manual foreach AdamW update", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_hiddensqreadoutzero_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7fp-RationalKAT-flashgroup-G16-h32-linearresGain19200-balancedPairCrossR96-readblocktriton-crossZero-b32-hiddenSqReadoutZero-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 balanced R96 pair block-readout plus zero-init hidden-square readout with frozen backbone and manual foreach AdamW update", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_balancedpaircrossr96_readblocktriton_crosszero_readblockb32_hiddensqreadoutzero_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7fq-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR96-readblocktriton-crossZero-b32-hiddenSqReadoutZero-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 R96 pair block-readout plus zero-init hidden-square readout with frozen backbone and manual foreach AdamW update", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_paircrossr96_readblocktriton_crosszero_readblockb32_hiddensqreadoutzero_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7fr-RationalKAT-flashgroup-G16-h32-linearresGain19200-hybridPairCrossR96-readblocktriton-crossZero-b32-hiddenSqReadoutZero-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 hybrid R96 pair block-readout plus zero-init hidden-square readout with frozen backbone and manual foreach AdamW update", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_hybridpaircrossr96_readblocktriton_crosszero_readblockb32_hiddensqreadoutzero_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7fs-RationalKAT-flashgroup-G16-h32-linearresGain19200-hybridPairCrossR112-readblocktriton-crossZero-b32-hiddenSqReadoutZero-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 hybrid R112 pair block-readout plus zero-init hidden-square readout with frozen backbone and manual foreach AdamW update", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_hybridpaircrossr112_readblocktriton_crosszero_readblockb32_hiddensqreadoutzero_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7ft-RationalKAT-flashgroup-G16-h32-linearresGain19200-gridPairCrossR96-readblocktriton-crossZero-b32-hiddenSqReadoutZero-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 grid-balanced R96 pair block-readout plus zero-init hidden-square readout with frozen backbone and manual foreach AdamW update", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_gridpaircrossr96_readblocktriton_crosszero_readblockb32_hiddensqreadoutzero_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7fu-RationalKAT-flashgroup-G16-h32-linearresGain19200-gridPairCrossR112-readblocktriton-crossZero-b32-hiddenSqReadoutZero-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 grid-balanced R112 pair block-readout plus zero-init hidden-square readout with frozen backbone and manual foreach AdamW update", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_gridpaircrossr112_readblocktriton_crosszero_readblockb32_hiddensqreadoutzero_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7fv-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR112-readblocktriton-crossZero-b32-hiddenSqReadoutZero-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 R112 pair block-readout plus zero-init hidden-square readout with frozen backbone and manual foreach AdamW update", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_paircrossr112_readblocktriton_crosszero_readblockb32_hiddensqreadoutzero_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7fw-RationalKAT-flashgroup-G16-h32-linearresGain19200-balancedPairCrossR112-readblocktriton-crossZero-b32-hiddenSqReadoutZero-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 balanced R112 pair block-readout plus zero-init hidden-square readout with frozen backbone and manual foreach AdamW update", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_balancedpaircrossr112_readblocktriton_crosszero_readblockb32_hiddensqreadoutzero_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7fx-RationalKAT-flashgroup-G16-h32-linearresGain19200-pairSumSqCrossR96-readblocktriton-crossZero-b32-hiddenSqReadoutZero-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 R96 sparse pair-sum-square quadratic readout plus zero-init hidden-square readout with frozen backbone and generic manual VJP", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_pairsumsqcrossr96_readblocktriton_crosszero_readblockb32_hiddensqreadoutzero_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7fy-RationalKAT-flashgroup-G16-h32-linearresGain19200-balancedPairSumSqCrossR96-readblocktriton-crossZero-b32-hiddenSqReadoutZero-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 balanced R96 sparse pair-sum-square quadratic readout plus zero-init hidden-square readout with frozen backbone and generic manual VJP", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_balancedpairsumsqcrossr96_readblocktriton_crosszero_readblockb32_hiddensqreadoutzero_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7fz-RationalKAT-flashgroup-G16-h32-linearresGain19200-pairSumSqCrossR64-readblocktriton-crossZero-b32-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 R64 sparse pair-sum-square quadratic readout with frozen backbone and generic manual VJP", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_pairsumsqcrossr64_readblocktriton_crosszero_readblockb32_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7ga-RationalKAT-flashgroup-G16-h32-linearresGain19200-balancedPairSumSqCrossR64-readblocktriton-crossZero-b32-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 balanced R64 sparse pair-sum-square quadratic readout with frozen backbone and generic manual VJP", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_balancedpairsumsqcrossr64_readblocktriton_crosszero_readblockb32_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7gb-RationalKAT-flashgroup-G16-h32-linearresGain19200-diagPairCrossR96-readblocktriton-crossZero-b32-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 diag-rich R96 product-pair quadratic readout with frozen backbone and generic manual VJP", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_diagpaircrossr96_readblocktriton_crosszero_readblockb32_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7gc-RationalKAT-flashgroup-G16-h32-linearresGain19200-diagPairCrossR112-readblocktriton-crossZero-b32-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 diag-rich R112 product-pair quadratic readout with frozen backbone and generic manual VJP", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_diagpaircrossr112_readblocktriton_crosszero_readblockb32_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7gd-RationalKAT-flashgroup-G16-h32-linearresGain19200-diagPairCrossR64-readblocktriton-crossZero-b32-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 diag-rich R64 product-pair quadratic readout with frozen backbone and generic manual VJP", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_diagpaircrossr64_readblocktriton_crosszero_readblockb32_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7ge-RationalKAT-flashgroup-G16-h32-linearresGain19200-diagPairCrossR80-readblocktriton-crossZero-b32-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 diag-rich R80 product-pair quadratic readout with frozen backbone and generic manual VJP", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_diagpaircrossr80_readblocktriton_crosszero_readblockb32_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7gf-RationalKAT-flashgroup-G16-h32-linearresGain19200-blendPairCrossR96-readblocktriton-crossZero-b32-hiddenSqReadoutZero-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 75/25 standard-balanced R96 product-pair readout plus zero-init hidden-square readout with frozen backbone and generic manual VJP", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_blendpaircrossr96_readblocktriton_crosszero_readblockb32_hiddensqreadoutzero_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7gg-RationalKAT-flashgroup-G16-h32-linearresGain19200-blendPairCrossR88-readblocktriton-crossZero-b32-hiddenSqReadoutZero-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 75/25 standard-balanced R88 product-pair readout plus zero-init hidden-square readout with frozen backbone and generic manual VJP", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_blendpaircrossr88_readblocktriton_crosszero_readblockb32_hiddensqreadoutzero_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7gh-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-hiddenSqReadoutZero-freezeBackbone-hiddenBias-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 B7ej full R136 pair block-readout plus zero-init hidden-square readout with frozen backbone and generic VJP", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_paircrossr136_readblocktriton_crosszero_readblockb32_hiddensqreadoutzero_freezebackbone_hiddenbias", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7gi-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-hiddenSqReadoutZero-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 B7ej full R136 pair block-readout plus zero-init hidden-square readout with frozen backbone and generic manual VJP/manual foreach AdamW", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_paircrossr136_readblocktriton_crosszero_readblockb32_hiddensqreadoutzero_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7gj-RationalKAT-flashgroup-G16-h32-linearresGain16000-paircrossR136-readblocktriton-crossZero-b32-hiddenSqReadoutZero-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 B7gi full R136 plus hidden-square-zero with lower linear residual gain 160.00 and generic manual VJP/manual foreach AdamW", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain16000_paircrossr136_readblocktriton_crosszero_readblockb32_hiddensqreadoutzero_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7gk-RationalKAT-flashgroup-G16-h32-linearresGain14400-paircrossR136-readblocktriton-crossZero-b32-hiddenSqReadoutZero-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 B7gi full R136 plus hidden-square-zero with lower linear residual gain 144.00 and generic manual VJP/manual foreach AdamW", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain14400_paircrossr136_readblocktriton_crosszero_readblockb32_hiddensqreadoutzero_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7gl-RationalKAT-flashgroup-G16-h32-linearresGain18400-paircrossR136-readblocktriton-crossZero-b32-hiddenSqReadoutZero-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 B7gi full R136 plus hidden-square-zero with interpolated linear residual gain 184.00 and generic manual VJP/manual foreach AdamW", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain18400_paircrossr136_readblocktriton_crosszero_readblockb32_hiddensqreadoutzero_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7gm-RationalKAT-flashgroup-G16-h32-linearresGain17600-paircrossR136-readblocktriton-crossZero-b32-hiddenSqReadoutZero-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 B7gi full R136 plus hidden-square-zero with interpolated linear residual gain 176.00 and generic manual VJP/manual foreach AdamW", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain17600_paircrossr136_readblocktriton_crosszero_readblockb32_hiddensqreadoutzero_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7gn-RationalKAT-flashgroup-G16-h32-linearresGain18800-paircrossR136-readblocktriton-crossZero-b32-hiddenSqReadoutZero-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 B7gi full R136 plus hidden-square-zero with narrow linear residual gain 188.00 and generic manual VJP/manual foreach AdamW", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain18800_paircrossr136_readblocktriton_crosszero_readblockb32_hiddensqreadoutzero_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7go-RationalKAT-flashgroup-G16-h32-linearresGain19000-paircrossR136-readblocktriton-crossZero-b32-hiddenSqReadoutZero-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 B7gi full R136 plus hidden-square-zero with narrow linear residual gain 190.00 and generic manual VJP/manual foreach AdamW", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19000_paircrossr136_readblocktriton_crosszero_readblockb32_hiddensqreadoutzero_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7gp-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-hiddenSqReadoutZero-hiddenSqScale050-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 B7gi full R136 plus half-scale hidden-square-zero trajectory and generic manual VJP/manual foreach AdamW", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_paircrossr136_readblocktriton_crosszero_readblockb32_hiddensqreadoutzero_hiddensqscale050_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7gq-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-hiddenSqReadoutZero-hiddenSqScale025-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 B7gi full R136 plus quarter-scale hidden-square-zero trajectory and generic manual VJP/manual foreach AdamW", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_paircrossr136_readblocktriton_crosszero_readblockb32_hiddensqreadoutzero_hiddensqscale025_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7gr-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-hiddenSqReadoutZero-hiddenSqScale075-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 B7gi full R136 plus 0.75-scale hidden-square-zero trajectory and generic manual VJP/manual foreach AdamW", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_paircrossr136_readblocktriton_crosszero_readblockb32_hiddensqreadoutzero_hiddensqscale075_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7gs-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-hiddenSqReadoutZero-hiddenSqScale062-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 B7gi full R136 plus 0.62-scale hidden-square-zero trajectory and generic manual VJP/manual foreach AdamW", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_paircrossr136_readblocktriton_crosszero_readblockb32_hiddensqreadoutzero_hiddensqscale062_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7gt-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-hiddenSqReadoutZero-hiddenSqCenter-hiddenSqScale050-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 B7gp but with per-sample centered hidden-square trajectory and generic manual VJP/manual foreach AdamW", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_paircrossr136_readblocktriton_crosszero_readblockb32_hiddensqreadoutzero_hiddensqcenter_hiddensqscale050_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7gu-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-hiddenSqReadoutZero-hiddenSqCenter-hiddenSqScale075-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 B7gr but with per-sample centered hidden-square trajectory and generic manual VJP/manual foreach AdamW", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_paircrossr136_readblocktriton_crosszero_readblockb32_hiddensqreadoutzero_hiddensqcenter_hiddensqscale075_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7gv-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-hiddenSqReadoutZero-hiddenSqSigned-hiddenSqScale050-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 B7gp but with signed quadratic h*abs(h) hidden tail and generic manual VJP/manual foreach AdamW", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_paircrossr136_readblocktriton_crosszero_readblockb32_hiddensqreadoutzero_hiddensqsigned_hiddensqscale050_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7gw-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-hiddenSqReadoutZero-hiddenSqSigned-hiddenSqScale075-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 B7gr but with signed quadratic h*abs(h) hidden tail and generic manual VJP/manual foreach AdamW", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_paircrossr136_readblocktriton_crosszero_readblockb32_hiddensqreadoutzero_hiddensqsigned_hiddensqscale075_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7gx-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-hiddenSqReadout-hiddenSqSigned-hiddenSqScale050-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 B7gv but with nonzero initialized signed quadratic hidden tail and generic manual VJP/manual foreach AdamW", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_paircrossr136_readblocktriton_crosszero_readblockb32_hiddensqreadout_hiddensqsigned_hiddensqscale050_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7gy-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-hiddenSqReadout-hiddenSqScale050-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 B7gp but with nonzero initialized h^2 hidden tail and generic manual VJP/manual foreach AdamW", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_paircrossr136_readblocktriton_crosszero_readblockb32_hiddensqreadout_hiddensqscale050_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7gz-RationalKAT-flashgroup-G16-h32-linearresGain19200-pairDiffSqCrossR96-readblocktriton-crossZero-b32-hiddenSqReadoutZero-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 R96 pair-difference-square readout plus zero-init hidden-square readout with frozen backbone and generic manual VJP", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_pairdiffsqcrossr96_readblocktriton_crosszero_readblockb32_hiddensqreadoutzero_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7ha-RationalKAT-flashgroup-G16-h32-linearresGain19200-pairDiffSqCrossR64-readblocktriton-crossZero-b32-hiddenSqReadoutZero-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 R64 pair-difference-square readout plus zero-init hidden-square readout with frozen backbone and generic manual VJP", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_pairdiffsqcrossr64_readblocktriton_crosszero_readblockb32_hiddensqreadoutzero_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7hb-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-readGradR136-crossZero-b32-hiddenSqReadoutZero-hiddenSqSigned-hiddenSqScale050-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 B7gv with full-rank pair-readout grad tile R136 for deeper reduction audit", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_paircrossr136_readblocktriton_readgradr136_crosszero_readblockb32_hiddensqreadoutzero_hiddensqsigned_hiddensqscale050_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7hc-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-readGradR136-crossZero-b32-hiddenSqReadoutZero-hiddenSqScale050-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 B7gp with full-rank pair-readout grad tile R136 for deeper reduction audit", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_paircrossr136_readblocktriton_readgradr136_crosszero_readblockb32_hiddensqreadoutzero_hiddensqscale050_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7hd-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-readGradAtomicR128-crossZero-b32-hiddenSqReadoutZero-hiddenSqSigned-hiddenSqScale050-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 B7gv with batch-split atomic R128 pair-readout grad accumulation", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_paircrossr136_readblocktriton_readgradatomicr128_crosszero_readblockb32_hiddensqreadoutzero_hiddensqsigned_hiddensqscale050_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7he-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-readGradAtomicR128-crossZero-b32-hiddenSqReadoutZero-hiddenSqScale050-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 B7gp with batch-split atomic R128 pair-readout grad accumulation", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_paircrossr136_readblocktriton_readgradatomicr128_crosszero_readblockb32_hiddensqreadoutzero_hiddensqscale050_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7hf-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-hiddenAbsReadoutZero-hiddenAbsScale050-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 B7ej with zero-init hidden absolute-value readout tail and generic manual VJP", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_paircrossr136_readblocktriton_crosszero_readblockb32_hiddenabsreadoutzero_hiddenabsscale050_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7hg-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-hiddenAbsReadout-hiddenAbsScale050-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 B7ej with small nonzero hidden absolute-value readout tail and generic manual VJP", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_paircrossr136_readblocktriton_crosszero_readblockb32_hiddenabsreadout_hiddenabsscale050_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7hh-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-hiddenAbsReadoutZero-hiddenAbsCenter-hiddenAbsScale050-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 B7hf with centered hidden absolute-value readout tail and generic manual VJP", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_paircrossr136_readblocktriton_crosszero_readblockb32_hiddenabsreadoutzero_hiddenabscenter_hiddenabsscale050_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7hi-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-hiddenAbsReadout-hiddenAbsCenter-hiddenAbsScale050-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 B7hg with centered hidden absolute-value readout tail and generic manual VJP", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_paircrossr136_readblocktriton_crosszero_readblockb32_hiddenabsreadout_hiddenabscenter_hiddenabsscale050_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7hj-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-hiddenAbsReadoutZero-hiddenAbsCenterMix025-hiddenAbsScale050-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 B7hf with 0.25 centered hidden absolute-value readout tail and generic manual VJP", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_paircrossr136_readblocktriton_crosszero_readblockb32_hiddenabsreadoutzero_hiddenabscenter_hiddenabscentermix025_hiddenabsscale050_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7hk-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-hiddenAbsReadoutZero-hiddenAbsCenterMix010-hiddenAbsScale050-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 B7hf with 0.10 centered hidden absolute-value readout tail and generic manual VJP", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_paircrossr136_readblocktriton_crosszero_readblockb32_hiddenabsreadoutzero_hiddenabscenter_hiddenabscentermix010_hiddenabsscale050_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7hl-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-hiddenRatReadoutZero-hiddenRatScale050-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 B7ej with zero-init bounded hidden rational h/(1+abs(h)) readout tail and generic manual VJP", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_paircrossr136_readblocktriton_crosszero_readblockb32_hiddenratreadoutzero_hiddenratscale050_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7hm-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-hiddenRatReadout-hiddenRatScale050-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 B7ej with nonzero bounded hidden rational h/(1+abs(h)) readout tail and generic manual VJP", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_paircrossr136_readblocktriton_crosszero_readblockb32_hiddenratreadout_hiddenratscale050_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7hn-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-hiddenRatReadoutZero-hiddenRatScale050-hiddenTailGradTriton-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 B7hl with Triton fused hidden rational readout-gradient and hidden VJP", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_paircrossr136_readblocktriton_crosszero_readblockb32_hiddenratreadoutzero_hiddenratscale050_hiddentailgradtriton_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7ho-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-hiddenAbsReadoutZero-hiddenAbsScale050-hiddenTailGradTriton-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 B7hf with Triton fused hidden absolute-value readout-gradient and hidden VJP", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_paircrossr136_readblocktriton_crosszero_readblockb32_hiddenabsreadoutzero_hiddenabsscale050_hiddentailgradtriton_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7hp-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-hiddenRatReadoutZero-hiddenRatScale025-hiddenTailGradTriton-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 B7hn with lower 0.25 hidden rational tail scale and Triton VJP", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_paircrossr136_readblocktriton_crosszero_readblockb32_hiddenratreadoutzero_hiddenratscale025_hiddentailgradtriton_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7hq-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-hiddenRatReadoutZero-hiddenRatScale075-hiddenTailGradTriton-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 B7hn with higher 0.75 hidden rational tail scale and Triton VJP", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_paircrossr136_readblocktriton_crosszero_readblockb32_hiddenratreadoutzero_hiddenratscale075_hiddentailgradtriton_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7hr-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-hiddenRatReadoutZero-hiddenRatScale050-hiddenTailGradTriton-tritonReadoutAdamW-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 B7hn plus Triton tensor AdamW for readout params", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_paircrossr136_readblocktriton_crosszero_readblockb32_hiddenratreadoutzero_hiddenratscale050_hiddentailgradtriton_tritonreadoutadamw_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7hs-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-hiddenAbsReadoutZero-hiddenAbsScale050-hiddenTailGradTriton-tritonReadoutAdamW-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 B7ho plus Triton tensor AdamW for readout params", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_paircrossr136_readblocktriton_crosszero_readblockb32_hiddenabsreadoutzero_hiddenabsscale050_hiddentailgradtriton_tritonreadoutadamw_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7hx-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-hiddenRatReadoutZero-hiddenRatScale050-hiddenTailGradTriton-tritonHiddenTailAdamW-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 B7hn plus Triton AdamW only for the hidden rational tail readout", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_paircrossr136_readblocktriton_crosszero_readblockb32_hiddenratreadoutzero_hiddenratscale050_hiddentailgradtriton_tritonhiddentailadamw_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7hy-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-hiddenAbsReadoutZero-hiddenAbsScale050-hiddenTailGradTriton-tritonHiddenTailAdamW-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 B7ho plus Triton AdamW only for the hidden absolute-value tail readout", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_paircrossr136_readblocktriton_crosszero_readblockb32_hiddenabsreadoutzero_hiddenabsscale050_hiddentailgradtriton_tritonhiddentailadamw_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7hz-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-hiddenRatReadoutZero-hiddenRatScale025-hiddenTailGradTriton-tritonHiddenTailAdamW-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 B7hx with lower 0.25 hidden rational tail scale and Triton AdamW only for hidden-tail readout", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_paircrossr136_readblocktriton_crosszero_readblockb32_hiddenratreadoutzero_hiddenratscale025_hiddentailgradtriton_tritonhiddentailadamw_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7ia-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-hiddenRatReadoutZero-hiddenRatScale035-hiddenTailGradTriton-tritonHiddenTailAdamW-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 B7hx with intermediate 0.35 hidden rational tail scale and Triton AdamW only for hidden-tail readout", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_paircrossr136_readblocktriton_crosszero_readblockb32_hiddenratreadoutzero_hiddenratscale035_hiddentailgradtriton_tritonhiddentailadamw_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7ib-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-hiddenRatReadoutZero-hiddenRatScale050-hiddenTailReadoutLR050-hiddenTailGradTriton-tritonHiddenTailAdamW-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 B7hx with 0.50 hidden-tail readout LR scale; loss-agnostic task trajectory repair", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_paircrossr136_readblocktriton_crosszero_readblockb32_hiddenratreadoutzero_hiddenratscale050_hiddentailreadoutlr050_hiddentailgradtriton_tritonhiddentailadamw_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7ic-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-hiddenRatReadoutZero-hiddenRatScale050-hiddenTailReadoutLR025-hiddenTailGradTriton-tritonHiddenTailAdamW-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 B7hx with 0.25 hidden-tail readout LR scale; loss-agnostic task trajectory repair", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_paircrossr136_readblocktriton_crosszero_readblockb32_hiddenratreadoutzero_hiddenratscale050_hiddentailreadoutlr025_hiddentailgradtriton_tritonhiddentailadamw_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7id-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-hiddenRatReadoutZero-hiddenRatScale050-hiddenTailReadoutLR150-hiddenTailGradTriton-tritonHiddenTailAdamW-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 B7hx with 1.50 hidden-tail readout LR scale; loss-agnostic task trajectory repair", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_paircrossr136_readblocktriton_crosszero_readblockb32_hiddenratreadoutzero_hiddenratscale050_hiddentailreadoutlr150_hiddentailgradtriton_tritonhiddentailadamw_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7ie-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-hiddenRatReadoutZero-hiddenRatScale050-hiddenTailReadoutLR200-hiddenTailGradTriton-tritonHiddenTailAdamW-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 B7hx with 2.00 hidden-tail readout LR scale; loss-agnostic task trajectory repair", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_paircrossr136_readblocktriton_crosszero_readblockb32_hiddenratreadoutzero_hiddenratscale050_hiddentailreadoutlr200_hiddentailgradtriton_tritonhiddentailadamw_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7if-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-hiddenRatReadoutZero-hiddenRatScale050-hiddenTailReadoutLR125-hiddenTailGradTriton-tritonHiddenTailAdamW-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 B7hx with 1.25 hidden-tail readout LR scale; midpoint between B7hx and B7id", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_paircrossr136_readblocktriton_crosszero_readblockb32_hiddenratreadoutzero_hiddenratscale050_hiddentailreadoutlr125_hiddentailgradtriton_tritonhiddentailadamw_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7ig-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-hiddenRatReadoutZero-hiddenRatScale050-hiddenTailReadoutLR135-hiddenTailGradTriton-tritonHiddenTailAdamW-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 B7hx with 1.35 hidden-tail readout LR scale; midpoint between B7hx and B7id", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_paircrossr136_readblocktriton_crosszero_readblockb32_hiddenratreadoutzero_hiddenratscale050_hiddentailreadoutlr135_hiddentailgradtriton_tritonhiddentailadamw_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7ih-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-hiddenRatReadoutZero-hiddenRatScale050-hiddenTailReadoutLR100to150e1-hiddenTailGradTriton-tritonHiddenTailAdamW-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 B7hx with hidden-tail readout LR schedule 1.00 to 1.50 at epoch 1; loss-agnostic trajectory schedule", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_paircrossr136_readblocktriton_crosszero_readblockb32_hiddenratreadoutzero_hiddenratscale050_hiddentailreadoutlr100to150e1_hiddentailgradtriton_tritonhiddentailadamw_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7ii-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-hiddenRatReadoutZero-hiddenRatScale050-hiddenTailReadoutLR100to150e2-hiddenTailGradTriton-tritonHiddenTailAdamW-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 B7hx with hidden-tail readout LR schedule 1.00 to 1.50 at epoch 2; loss-agnostic trajectory schedule", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_paircrossr136_readblocktriton_crosszero_readblockb32_hiddenratreadoutzero_hiddenratscale050_hiddentailreadoutlr100to150e2_hiddentailgradtriton_tritonhiddentailadamw_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7ij-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-hiddenRatReadoutZero-hiddenRatCenter-hiddenRatScale050-hiddenTailGradTriton-tritonHiddenTailAdamW-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 B7hx with centered hidden rational tail; loss-agnostic feature geometry repair", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_paircrossr136_readblocktriton_crosszero_readblockb32_hiddenratreadoutzero_hiddenratcenter_hiddenratscale050_hiddentailgradtriton_tritonhiddentailadamw_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7ik-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-hiddenRatReadoutZero-hiddenRatCenterMix025-hiddenRatScale050-hiddenTailGradTriton-tritonHiddenTailAdamW-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 B7hx with 0.25-centered hidden rational tail; loss-agnostic feature geometry repair", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_paircrossr136_readblocktriton_crosszero_readblockb32_hiddenratreadoutzero_hiddenratcentermix025_hiddenratscale050_hiddentailgradtriton_tritonhiddentailadamw_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7il-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-hiddenRatReadoutZero-hiddenRatCenterSG-hiddenRatScale050-hiddenTailGradTriton-tritonHiddenTailAdamW-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 B7hx with stop-gradient centered hidden rational tail; loss-agnostic feature geometry repair", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_paircrossr136_readblocktriton_crosszero_readblockb32_hiddenratreadoutzero_hiddenratcentersg_hiddenratscale050_hiddentailgradtriton_tritonhiddentailadamw_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7im-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-hiddenRatReadoutZero-hiddenRatCenterSGMix025-hiddenRatScale050-hiddenTailGradTriton-tritonHiddenTailAdamW-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 B7hx with 0.25 stop-gradient centered hidden rational tail; loss-agnostic feature geometry repair", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_paircrossr136_readblocktriton_crosszero_readblockb32_hiddenratreadoutzero_hiddenratcentersg_hiddenratcentermix025_hiddenratscale050_hiddentailgradtriton_tritonhiddentailadamw_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7in-RationalKAT-flashgroup-G16-h40-linearres-paircrossR136-readblocktriton-crossZero-b32-hiddenRatResidual050-freezeBackbone-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 40, "third_party/FlashKAT/rational_kat grouped rational h40 R136 crosszero B32 block-readout with bounded rational hidden residual folded into existing w2 readout; no hidden-tail readout parameter", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearres_paircrossr136_readblocktriton_crosszero_readblockb32_hiddenratresidual050_freezebackbone_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7io-RationalKAT-flashgroup-G16-h32-linearres-paircrossR136-readblocktriton-crossZero-b32-hiddenRatResidual050-freezeBackbone-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 R136 crosszero B32 block-readout with bounded rational hidden residual folded into existing w2 readout; no hidden-tail readout parameter", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearres_paircrossr136_readblocktriton_crosszero_readblockb32_hiddenratresidual050_freezebackbone_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7ip-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-hiddenRatResidual050-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 B7ej-shaped high linear-residual path with bounded rational hidden residual folded into existing w2 readout; generic hidden-bias VJP", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_paircrossr136_readblocktriton_crosszero_readblockb32_hiddenratresidual050_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7iq-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-hiddenRatResidual025-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 B7ej-shaped high linear-residual path with lower bounded rational hidden residual folded into existing w2 readout; generic hidden-bias VJP", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_paircrossr136_readblocktriton_crosszero_readblockb32_hiddenratresidual025_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7ir-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-hiddenRatResidual015-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 B7ej-shaped high linear-residual path with 0.15 bounded rational hidden residual folded into existing w2 readout; L3 margin repair", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_paircrossr136_readblocktriton_crosszero_readblockb32_hiddenratresidual015_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7is-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-hiddenRatResidual010-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 B7ej-shaped high linear-residual path with 0.10 bounded rational hidden residual folded into existing w2 readout; L3 margin repair", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_paircrossr136_readblocktriton_crosszero_readblockb32_hiddenratresidual010_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7it-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-hiddenRatResidual018-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 B7ir/B7iq midpoint with 0.18 bounded rational hidden residual folded into existing w2 readout", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_paircrossr136_readblocktriton_crosszero_readblockb32_hiddenratresidual018_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7iu-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-hiddenRatResidual020-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 B7ir/B7iq midpoint with 0.20 bounded rational hidden residual folded into existing w2 readout", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_paircrossr136_readblocktriton_crosszero_readblockb32_hiddenratresidual020_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7iv-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-hiddenRatResidual015-hiddenResVJPTriton-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 B7ir math with generic Triton w2 readout-gradient plus hidden-residual VJP fusion", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_paircrossr136_readblocktriton_crosszero_readblockb32_hiddenratresidual015_hiddenresvjptriton_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7iw-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-hiddenRatResidual010-hiddenResVJPTriton-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 B7is math with generic Triton w2 readout-gradient plus hidden-residual VJP fusion", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_paircrossr136_readblocktriton_crosszero_readblockb32_hiddenratresidual010_hiddenresvjptriton_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7kc-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-hiddenRatResidual005-hiddenResVJPTriton-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "B7iw task-trajectory damping bracket: lower bounded rational hidden residual amplitude 0.10 -> 0.05 with the same generic Triton VJP; no CE-specific tuning", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_paircrossr136_readblocktriton_crosszero_readblockb32_hiddenratresidual005_hiddenresvjptriton_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7lp-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-pairNorm-crossZero-b32-hiddenRatResidual005-hiddenResVJPTriton-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "B7en pair-feature normalization plus B7kc bounded rational hidden residual 0.05; structural task-geometry repair without CE-specific tuning", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_paircrossr136_readblocktriton_pairnorm_crosszero_readblockb32_hiddenratresidual005_hiddenresvjptriton_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7lq-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-pairNorm-crossZero-b32-hiddenRatResidual005-hiddenResVJPTriton-crossBatchSGCap125-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "B7lp plus stop-gradient batch-centered pair-logit cap 1.25; Line C coupling-collapse repair without CE-specific tuning", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_paircrossr136_readblocktriton_pairnorm_crosszero_readblockb32_hiddenratresidual005_hiddenresvjptriton_crossbatchsgcap125_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7lr-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-pairNorm-crossZero-b32-crossBatchSGCap125-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "B7en plus stop-gradient batch-centered pair-logit cap 1.25 without hidden residual; isolates pair-signal geometry for Line C coupling-collapse repair", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_paircrossr136_readblocktriton_pairnorm_crosszero_readblockb32_crossbatchsgcap125_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7ls-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-pairNorm-crossZero-b32-crossBatchSGCap125-freezeRational-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "B7lr with trainable w1 backbone and frozen rational coefficients; tests whether Line C coupling collapse is caused by readout-only trajectory", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_paircrossr136_readblocktriton_pairnorm_crosszero_readblockb32_crossbatchsgcap125_freezerational_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7lt-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-pairNorm-crossZero-b32-hiddenRatResidual005-hiddenResVJPTriton-crossBatchSGCap125-freezeRational-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "B7lq with trainable w1 backbone and frozen rational coefficients; tests backbone signal-channel repair plus bounded hidden residual", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_paircrossr136_readblocktriton_pairnorm_crosszero_readblockb32_hiddenratresidual005_hiddenresvjptriton_crossbatchsgcap125_freezerational_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7lu-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-pairNorm-crossZero-b32-crossSignalR4-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "B7en pairNorm but replaces full per-pair class readout with rank-4 shared pair-signal channel; tests Line C reservoir-trapping repair without CE-specific tuning", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_paircrossr136_readblocktriton_pairnorm_crosszero_readblockb32_crosssignalr4_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7lv-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-pairNorm-crossZero-b32-crossSignalR8-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "B7lu higher shared signal rank 8; brackets expression recovery versus reservoir trapping under the same generic VJP", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_paircrossr136_readblocktriton_pairnorm_crosszero_readblockb32_crosssignalr8_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7lw-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-pairNorm-crossPCAWhiteR96-crossZero-b32-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "B7en pairNorm with fixed unlabeled PCA-whitened pair feature channel R96; tests feature-conditioning repair for Line C coupling without CE-specific tuning", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_paircrossr136_readblocktriton_pairnorm_crosspcawhiter96_crosszero_readblockb32_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7lx-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-pairNorm-crossPCAWhiteR128-crossZero-b32-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "B7lw higher PCA-whitened pair rank R128 to preserve A4 expression while testing Line C feature conditioning", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_paircrossr136_readblocktriton_pairnorm_crosspcawhiter128_crosszero_readblockb32_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7ly-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-pairNorm-crossPCAR96-crossZero-b32-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "B7lw without whitening: fixed unlabeled PCA-rotated pair feature channel R96 to avoid low-energy logit amplification while testing feature conditioning", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_paircrossr136_readblocktriton_pairnorm_crosspcar96_crosszero_readblockb32_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7lz-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-pairNorm-crossPCAR128-crossZero-b32-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "B7lx without whitening: fixed unlabeled PCA-rotated pair feature channel R128 to preserve A4 without whitening-induced logit explosion", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_paircrossr136_readblocktriton_pairnorm_crosspcar128_crosszero_readblockb32_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7ma-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-pairNorm-crossPCAR128-crossZero-b32-logitRMSNormSG-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "B7lz plus samplewise stop-gradient logit RMS normalization; output-geometry repair for A5/Line C without CE-specific tuning", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_paircrossr136_readblocktriton_pairnorm_crosspcar128_crosszero_readblockb32_logitrmsnormsg_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7mb-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-pairNorm-crossZero-b32-hiddenRatResidual005-hiddenResVJPTriton-logitRMSNormSG-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "B7lp plus samplewise stop-gradient logit RMS normalization; tests whether output-scale geometry can reduce A5/Line C collapse without changing loss/data/gates", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_paircrossr136_readblocktriton_pairnorm_crosszero_readblockb32_hiddenratresidual005_hiddenresvjptriton_logitrmsnormsg_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7mc-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-pairNorm-crossPCAR128-crossZero-b32-logitBatchRMSNormSG-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "B7ma cheaper batch-level stop-gradient logit RMS normalization; tests whether global output-scale geometry keeps Line C gain while reopening L3", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_paircrossr136_readblocktriton_pairnorm_crosspcar128_crosszero_readblockb32_logitbatchrmsnormsg_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7md-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-pairNorm-crossZero-b32-hiddenRatResidual005-hiddenResVJPTriton-logitBatchRMSNormSG-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "B7mb cheaper batch-level stop-gradient logit RMS normalization on the B7lp anchor; L3 repair bracket for the samplewise output-geometry result", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_paircrossr136_readblocktriton_pairnorm_crosszero_readblockb32_hiddenratresidual005_hiddenresvjptriton_logitbatchrmsnormsg_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7me-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-pairNorm-crossZero-b32-hiddenRatResidual005-hiddenResVJPTriton-logitBatchRMSNormSG150-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "B7md margin-scale bracket: batch-level stop-gradient logit RMS normalization with scale 1.50 to recover A5 task margins", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_paircrossr136_readblocktriton_pairnorm_crosszero_readblockb32_hiddenratresidual005_hiddenresvjptriton_logitbatchrmsnormsg150_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7mf-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-pairNorm-crossZero-b32-hiddenRatResidual005-hiddenResVJPTriton-logitBatchRMSNormSG200-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "B7md margin-scale bracket: batch-level stop-gradient logit RMS normalization with scale 2.00 to test stronger task-margin recovery", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_paircrossr136_readblocktriton_pairnorm_crosszero_readblockb32_hiddenratresidual005_hiddenresvjptriton_logitbatchrmsnormsg200_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7mg-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-pairNorm-crossZero-b32-hiddenRatResidual005-hiddenResVJPTriton-logitBatchRMSMixSG025-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "B7md residual output-geometry repair: blend 25% batch stop-gradient RMS normalization with 75% original logits to preserve A5 task margins", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_paircrossr136_readblocktriton_pairnorm_crosszero_readblockb32_hiddenratresidual005_hiddenresvjptriton_logitbatchrmsmixsg025_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7mh-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-pairNorm-crossZero-b32-hiddenRatResidual005-hiddenResVJPTriton-logitBatchRMSMixSG050-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "B7md residual output-geometry repair: blend 50% batch stop-gradient RMS normalization with original logits to bracket Line C gain versus A5 harm", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_paircrossr136_readblocktriton_pairnorm_crosszero_readblockb32_hiddenratresidual005_hiddenresvjptriton_logitbatchrmsmixsg050_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7kd-RationalKAT-flashgroup-G16-h32-linearresGain14400-paircrossR136-readblocktriton-crossZero-b32-hiddenRatResidual005-hiddenResVJPTriton-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "B7kc signal-channel bracket: lower linear residual gain 192.0 -> 144.0 while keeping residual amplitude 0.05 and generic Triton hidden-residual VJP", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain14400_paircrossr136_readblocktriton_crosszero_readblockb32_hiddenratresidual005_hiddenresvjptriton_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7ke-RationalKAT-flashgroup-G16-h32-linearresGain09600-paircrossR136-readblocktriton-crossZero-b32-hiddenRatResidual005-hiddenResVJPTriton-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "B7kc stronger signal-channel bracket: lower linear residual gain 192.0 -> 96.0 while keeping residual amplitude 0.05 and generic Triton hidden-residual VJP", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain09600_paircrossr136_readblocktriton_crosszero_readblockb32_hiddenratresidual005_hiddenresvjptriton_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7kf-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR132-readblocktriton-crossZero-b32-hiddenTanhResidual005-hiddenResVJPTriton-crossBatchSGCap125-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "B7jt/B7jx bridge: keep the small smooth tanh hidden residual 0.05 and reduce stop-gradient batch cap 1.50 -> 1.25 to test ECE/worst-delta tradeoff without CE tuning", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_paircrossr132_readblocktriton_crosszero_readblockb32_hiddentanhresidual005_hiddenresvjptriton_crossbatchsgcap125_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7kg-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR132-readblocktriton-crossZero-b32-hiddenTanhResidual005-hiddenResVJPTriton-crossBatchSGCap100-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "B7kf lower-cap NLL/AUC repair: keep small tanh residual 0.05 and reduce stop-gradient batch cap 1.25 -> 1.00 without CE tuning", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_paircrossr132_readblocktriton_crosszero_readblockb32_hiddentanhresidual005_hiddenresvjptriton_crossbatchsgcap100_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7kh-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR132-readblocktriton-crossZero-b32-hiddenTanhResidual005-hiddenResVJPTriton-crossBatchSGCap075-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "B7kf aggressive lower-cap NLL/AUC repair: keep small tanh residual 0.05 and reduce stop-gradient batch cap to 0.75 without CE tuning", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_paircrossr132_readblocktriton_crosszero_readblockb32_hiddentanhresidual005_hiddenresvjptriton_crossbatchsgcap075_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7ki-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR132-readblocktriton-crossZero-b32-crossBatchSGCap100-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "B7jx no-hidden lower-cap NLL/ECE repair: keep R132 no-hidden path and reduce stop-gradient batch cap 1.25 -> 1.00 without CE tuning", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_paircrossr132_readblocktriton_crosszero_readblockb32_crossbatchsgcap100_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7kj-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR132-readblocktriton-crossZero-b32-crossBatchSGCap075-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "B7jx aggressive no-hidden lower-cap NLL/ECE repair: keep R132 no-hidden path and reduce stop-gradient batch cap to 0.75 without CE tuning", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_paircrossr132_readblocktriton_crosszero_readblockb32_crossbatchsgcap075_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7kk-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR128-readblocktriton-crossZero-b32-hiddenTanhResidual005-hiddenResVJPTriton-crossBatchSGCap125-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "B7kf middle-rank repair: reduce R132 -> R128 while keeping small tanh residual 0.05 and stop-gradient batch cap 1.25; tests L3/A4 tradeoff without CE tuning", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_paircrossr128_readblocktriton_crosszero_readblockb32_hiddentanhresidual005_hiddenresvjptriton_crossbatchsgcap125_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7kl-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR120-readblocktriton-crossZero-b32-hiddenTanhResidual005-hiddenResVJPTriton-crossBatchSGCap125-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "B7kf lower middle-rank repair: reduce R132 -> R120 while keeping small tanh residual 0.05 and stop-gradient batch cap 1.25; tests L3/A4 tradeoff without CE tuning", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_paircrossr120_readblocktriton_crosszero_readblockb32_hiddentanhresidual005_hiddenresvjptriton_crossbatchsgcap125_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7km-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR112-readblocktriton-crossZero-b32-hiddenTanhResidual005-hiddenResVJPTriton-crossBatchSGCap125-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "B7kk lower-rank follow-up: reduce R128 -> R112 with the same small tanh residual 0.05 and cap 1.25 to test whether L3 opens before A4 expression collapses", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_paircrossr112_readblocktriton_crosszero_readblockb32_hiddentanhresidual005_hiddenresvjptriton_crossbatchsgcap125_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7kn-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR96-readblocktriton-crossZero-b32-hiddenTanhResidual005-hiddenResVJPTriton-crossBatchSGCap125-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "B7kk lower-rank follow-up: reduce to R96 with the same small tanh residual 0.05 and cap 1.25; expected to test L3 opening versus A4 expression loss", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_paircrossr96_readblocktriton_crosszero_readblockb32_hiddentanhresidual005_hiddenresvjptriton_crossbatchsgcap125_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7ko-RationalKAT-flashgroup-G16-h32-linearresGain19200-hybridPairCrossR112-readblocktriton-crossZero-b32-hiddenTanhResidual005-hiddenResVJPTriton-crossBatchSGCap125-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "B7km expression repair: keep R112 cost but use 50/50 standard-balanced pair coverage to recover rotated/random quadratic span without CE tuning", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_hybridpaircrossr112_readblocktriton_crosszero_readblockb32_hiddentanhresidual005_hiddenresvjptriton_crossbatchsgcap125_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7kp-RationalKAT-flashgroup-G16-h32-linearresGain19200-balancedPairCrossR112-readblocktriton-crossZero-b32-hiddenTanhResidual005-hiddenResVJPTriton-crossBatchSGCap125-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "B7km expression repair: keep R112 cost but switch to balanced pair coverage to improve E6/E8 expression without changing loss or labels", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_balancedpaircrossr112_readblocktriton_crosszero_readblockb32_hiddentanhresidual005_hiddenresvjptriton_crossbatchsgcap125_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7kq-RationalKAT-flashgroup-G16-h32-linearresGain19200-diagPairCrossR112-readblocktriton-crossZero-b32-hiddenTanhResidual005-hiddenResVJPTriton-crossBatchSGCap125-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "B7km expression repair: keep R112 cost but use diag-rich pair coverage to test diagonal/rotated quadratic span under the same generic VJP", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_diagpaircrossr112_readblocktriton_crosszero_readblockb32_hiddentanhresidual005_hiddenresvjptriton_crossbatchsgcap125_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7kr-RationalKAT-flashgroup-G16-h32-linearresGain19200-hybridPairCrossR128-readblocktriton-crossZero-b32-hiddenTanhResidual005-hiddenResVJPTriton-crossBatchSGCap125-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "B7ko higher-rank expression repair: hybrid R128 pair coverage with the same tanh residual/cap to test A4 recovery near the L3 boundary", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_hybridpaircrossr128_readblocktriton_crosszero_readblockb32_hiddentanhresidual005_hiddenresvjptriton_crossbatchsgcap125_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7ks-RationalKAT-flashgroup-G16-h32-linearresGain19200-balancedPairCrossR128-readblocktriton-crossZero-b32-hiddenTanhResidual005-hiddenResVJPTriton-crossBatchSGCap125-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "B7kp higher-rank expression repair: balanced R128 pair coverage to test E6 recovery without changing loss/data/gates", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_balancedpaircrossr128_readblocktriton_crosszero_readblockb32_hiddentanhresidual005_hiddenresvjptriton_crossbatchsgcap125_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7kt-RationalKAT-flashgroup-G16-h32-linearresGain19200-diagPairCrossR128-readblocktriton-crossZero-b32-hiddenTanhResidual005-hiddenResVJPTriton-crossBatchSGCap125-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "B7kq higher-rank expression repair: diag-rich R128 pair coverage to test E8 recovery without changing the objective", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_diagpaircrossr128_readblocktriton_crosszero_readblockb32_hiddentanhresidual005_hiddenresvjptriton_crossbatchsgcap125_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7ku-RationalKAT-flashgroup-G16-h32-linearresGain19200-hybridPairCrossR120-readblocktriton-crossZero-b32-hiddenTanhResidual005-hiddenResVJPTriton-crossBatchSGCap125-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "B7ko midpoint expression repair: hybrid R120 pair coverage to bracket R112/R128 under the same generic VJP", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_hybridpaircrossr120_readblocktriton_crosszero_readblockb32_hiddentanhresidual005_hiddenresvjptriton_crossbatchsgcap125_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7kv-RationalKAT-flashgroup-G16-h32-linearresGain19200-diagPairCrossR120-readblocktriton-crossZero-b32-hiddenTanhResidual005-hiddenResVJPTriton-crossBatchSGCap125-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "B7kq midpoint expression repair: diag-rich R120 pair coverage to bracket E8 recovery against L3 cost", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_diagpaircrossr120_readblocktriton_crosszero_readblockb32_hiddentanhresidual005_hiddenresvjptriton_crossbatchsgcap125_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7kw-RationalKAT-flashgroup-G16-h32-linearresGain19200-balBlendPairCrossR128-readblocktriton-crossZero-b32-hiddenTanhResidual005-hiddenResVJPTriton-crossBatchSGCap125-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "B7ks/B7kr mix repair: R128 with 25% standard and 75% balanced pair coverage to keep E1 while lifting E6/E8; generic VJP only", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_balblendpaircrossr128_readblocktriton_crosszero_readblockb32_hiddentanhresidual005_hiddenresvjptriton_crossbatchsgcap125_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7kx-RationalKAT-flashgroup-G16-h32-linearresGain19200-balBlendPairCrossR120-readblocktriton-crossZero-b32-hiddenTanhResidual005-hiddenResVJPTriton-crossBatchSGCap125-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "B7ks/B7ku mix repair: R120 with 25% standard and 75% balanced pair coverage to bracket expression recovery inside L3 cost", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_balblendpaircrossr120_readblocktriton_crosszero_readblockb32_hiddentanhresidual005_hiddenresvjptriton_crossbatchsgcap125_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7ky-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR80-readblocktriton-crossZero-b32-projBilinR32-bilinZero-hiddenTanhResidual005-hiddenResVJPTriton-crossBatchSGCap125-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "B7fe/B7kn repair: keep cheap R80 pair block and add fixed dual-DCT projection-bilinear R32 readout to test low-cost rotated quadratic expression recovery", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_paircrossr80_readblocktriton_crosszero_readblockb32_projbilinr32_bilinzero_hiddentanhresidual005_hiddenresvjptriton_crossbatchsgcap125_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7kz-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR80-readblocktriton-crossZero-b32-projSqR32-sqZero-hiddenTanhResidual005-hiddenResVJPTriton-crossBatchSGCap125-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "B7fe/B7kn repair: keep cheap R80 pair block and add fixed DCT projection-square R32 readout to test low-cost diagonal rotated quadratic recovery", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_paircrossr80_readblocktriton_crosszero_readblockb32_projsqr32_sqzero_hiddentanhresidual005_hiddenresvjptriton_crossbatchsgcap125_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7la-RationalKAT-flashgroup-G16-h32-linearresGain19200-diagPairCrossR80-readblocktriton-crossZero-b32-projBilinR32-bilinZero-hiddenTanhResidual005-hiddenResVJPTriton-crossBatchSGCap125-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "B7ge/B7kq repair: combine cheap diag-rich R80 pair coverage with fixed projection-bilinear R32 readout to test E8 recovery without CE tuning", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_diagpaircrossr80_readblocktriton_crosszero_readblockb32_projbilinr32_bilinzero_hiddentanhresidual005_hiddenresvjptriton_crossbatchsgcap125_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7lb-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR80-readblocktriton-crossZero-b32-projBilinR64-bilinZero-hiddenTanhResidual005-hiddenResVJPTriton-crossBatchSGCap125-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "B7ky rank bracket: R80 pair block plus fixed dual-DCT projection-bilinear R64 readout to spend remaining L3 margin on rotated quadratic expression", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_paircrossr80_readblocktriton_crosszero_readblockb32_projbilinr64_bilinzero_hiddentanhresidual005_hiddenresvjptriton_crossbatchsgcap125_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7lc-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR80-readblocktriton-crossZero-b32-projSqR64-sqZero-hiddenTanhResidual005-hiddenResVJPTriton-crossBatchSGCap125-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "B7kz rank bracket: R80 pair block plus fixed DCT projection-square R64 readout to spend remaining L3 margin on diagonal rotated quadratic expression", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_paircrossr80_readblocktriton_crosszero_readblockb32_projsqr64_sqzero_hiddentanhresidual005_hiddenresvjptriton_crossbatchsgcap125_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7ld-RationalKAT-flashgroup-G16-h32-linearresGain19200-diagPairCrossR80-readblocktriton-crossZero-b32-projBilinR64-bilinZero-hiddenTanhResidual005-hiddenResVJPTriton-crossBatchSGCap125-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "B7la rank bracket: diag-rich R80 pair block plus fixed projection-bilinear R64 readout for E6/E8-oriented expression repair", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_diagpaircrossr80_readblocktriton_crosszero_readblockb32_projbilinr64_bilinzero_hiddentanhresidual005_hiddenresvjptriton_crossbatchsgcap125_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7le-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR80-readblocktriton-crossZero-b32-projBilinR48-bilinZero-hiddenTanhResidual005-hiddenResVJPTriton-crossBatchSGCap125-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "B7ky/B7lb bridge: R80 pair block plus fixed dual-DCT projection-bilinear R48 readout between weak R32 and slow R64", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_paircrossr80_readblocktriton_crosszero_readblockb32_projbilinr48_bilinzero_hiddentanhresidual005_hiddenresvjptriton_crossbatchsgcap125_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7lf-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR80-readblocktriton-crossZero-b32-projSqR48-sqZero-hiddenTanhResidual005-hiddenResVJPTriton-crossBatchSGCap125-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "B7kz/B7lc bridge: R80 pair block plus fixed DCT projection-square R48 readout between weak R32 and slow R64", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_paircrossr80_readblocktriton_crosszero_readblockb32_projsqr48_sqzero_hiddentanhresidual005_hiddenresvjptriton_crossbatchsgcap125_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7lg-RationalKAT-flashgroup-G16-h32-linearresGain19200-diagPairCrossR80-readblocktriton-crossZero-b32-projBilinR48-bilinZero-hiddenTanhResidual005-hiddenResVJPTriton-crossBatchSGCap125-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "B7la/B7ld bridge: diag-rich R80 pair block plus fixed projection-bilinear R48 readout for E6/E8 repair within L3 margin", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_diagpaircrossr80_readblocktriton_crosszero_readblockb32_projbilinr48_bilinzero_hiddentanhresidual005_hiddenresvjptriton_crossbatchsgcap125_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7lh-RationalKAT-flashgroup-G16-h32-linearresGain19200-pairSumSqCrossR80-readblocktriton-crossZero-b32-hiddenTanhResidual005-hiddenResVJPTriton-crossBatchSGCap125-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "projection-route fallback: R80 pair-sum-square features with the B7kf small tanh residual/cap path to test low-cost rotated quadratic coverage", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_pairsumsqcrossr80_readblocktriton_crosszero_readblockb32_hiddentanhresidual005_hiddenresvjptriton_crossbatchsgcap125_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7li-RationalKAT-flashgroup-G16-h32-linearresGain19200-pairDiffSqCrossR80-readblocktriton-crossZero-b32-hiddenTanhResidual005-hiddenResVJPTriton-crossBatchSGCap125-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "projection-route fallback: R80 pair-difference-square features with the B7kf small tanh residual/cap path to test alternate quadratic coverage", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_pairdiffsqcrossr80_readblocktriton_crosszero_readblockb32_hiddentanhresidual005_hiddenresvjptriton_crossbatchsgcap125_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7lj-RationalKAT-flashgroup-G16-h32-linearresGain19200-blendPairCrossR80-readblocktriton-crossZero-b32-hiddenTanhResidual005-hiddenResVJPTriton-crossBatchSGCap125-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "projection-route fallback: R80 75/25 standard-balanced pair coverage with the B7kf small tanh residual/cap path", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_blendpaircrossr80_readblocktriton_crosszero_readblockb32_hiddentanhresidual005_hiddenresvjptriton_crossbatchsgcap125_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7lk-RationalKAT-flashgroup-G16-h32-linearresGain19200-inputcrossR64-crossZero-hiddenTanhResidual005-hiddenResVJPTriton-crossBatchSGCap125-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "new random-projection quadratic coverage: inputcross R64 with B7kf small tanh residual/cap trajectory, generic VJP, no CE-specific tuning", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_inputcrossr64_crosszero_hiddentanhresidual005_hiddenresvjptriton_crossbatchsgcap125_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7ll-RationalKAT-flashgroup-G16-h32-linearresGain19200-inputcrossR96-crossZero-hiddenTanhResidual005-hiddenResVJPTriton-crossBatchSGCap125-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "new random-projection quadratic coverage: inputcross R96 with B7kf small tanh residual/cap trajectory, generic VJP, no CE-specific tuning", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_inputcrossr96_crosszero_hiddentanhresidual005_hiddenresvjptriton_crossbatchsgcap125_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7lm-RationalKAT-flashgroup-G16-h32-linearresGain19200-inputcrossR112-crossZero-hiddenTanhResidual005-hiddenResVJPTriton-crossBatchSGCap125-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "new random-projection quadratic coverage: inputcross R112 with B7kf small tanh residual/cap trajectory, generic VJP, no CE-specific tuning", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_inputcrossr112_crosszero_hiddentanhresidual005_hiddenresvjptriton_crossbatchsgcap125_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7ln-RationalKAT-flashgroup-G16-h32-linearresGain19200-inputcrossR32-crossZero-hiddenTanhResidual005-hiddenResVJPTriton-crossBatchSGCap125-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "low-rank random-projection quadratic bracket: inputcross R32 with current h32/tanh005/cap125 trajectory", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_inputcrossr32_crosszero_hiddentanhresidual005_hiddenresvjptriton_crossbatchsgcap125_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7lo-RationalKAT-flashgroup-G16-h32-linearresGain19200-inputcrossR48-crossZero-hiddenTanhResidual005-hiddenResVJPTriton-crossBatchSGCap125-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "low-rank random-projection quadratic bracket: inputcross R48 with current h32/tanh005/cap125 trajectory", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_inputcrossr48_crosszero_hiddentanhresidual005_hiddenresvjptriton_crossbatchsgcap125_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7ix-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-hiddenTanhResidual015-hiddenResVJPTriton-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 with smooth tanh hidden residual folded into existing w2 readout and generic Triton residual VJP", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_paircrossr136_readblocktriton_crosszero_readblockb32_hiddentanhresidual015_hiddenresvjptriton_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7iy-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-hiddenTanhResidual010-hiddenResVJPTriton-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 with lower smooth tanh hidden residual folded into existing w2 readout and generic Triton residual VJP", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_paircrossr136_readblocktriton_crosszero_readblockb32_hiddentanhresidual010_hiddenresvjptriton_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7iz-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-hiddenCenterTanhResidual015-hiddenResVJPTriton-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 with centered smooth tanh hidden residual folded into existing w2 readout and generic Triton residual VJP", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_paircrossr136_readblocktriton_crosszero_readblockb32_hiddencentertanhresidual015_hiddenresvjptriton_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7ja-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-hiddenCenterTanhResidual010-hiddenResVJPTriton-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 with lower centered smooth tanh hidden residual folded into existing w2 readout and generic Triton residual VJP", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_paircrossr136_readblocktriton_crosszero_readblockb32_hiddencentertanhresidual010_hiddenresvjptriton_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7jb-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-crossCapFused150-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 B7ir anchor with non-hidden-residual fused pair-logit cap 1.50 and generic VJP", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_paircrossr136_readblocktriton_crosszero_readblockb32_crosscapfused150_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7jc-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-crossCapFused200-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 B7ir anchor with non-hidden-residual fused pair-logit cap 2.00 and generic VJP", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_paircrossr136_readblocktriton_crosszero_readblockb32_crosscapfused200_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7jd-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-crossCenterCap150-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 B7ir anchor with centered pair-logit tanh cap 1.50 and generic VJP", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_paircrossr136_readblocktriton_crosszero_readblockb32_crosscentercap150_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7je-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-crossCenterCap200-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 B7ir anchor with centered pair-logit tanh cap 2.00 and generic VJP", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_paircrossr136_readblocktriton_crosszero_readblockb32_crosscentercap200_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7jh-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-crossBatchCap100-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 B7jf lower batch-centered pair-logit tanh cap 1.00 and generic VJP", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_paircrossr136_readblocktriton_crosszero_readblockb32_crossbatchcap100_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7ji-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-crossBatchCap125-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 B7jf midpoint batch-centered pair-logit tanh cap 1.25 and generic VJP", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_paircrossr136_readblocktriton_crosszero_readblockb32_crossbatchcap125_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7jj-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-crossBatchSGCap150-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 B7jf with batch-centered pair-logit tanh cap 1.50 and stop-gradient batch mean; generic dL/dlogits VJP, not CE-specific", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_paircrossr136_readblocktriton_crosszero_readblockb32_crossbatchsgcap150_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7jk-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-crossBatchSGCap200-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 B7jg with batch-centered pair-logit tanh cap 2.00 and stop-gradient batch mean; generic dL/dlogits VJP, not CE-specific", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_paircrossr136_readblocktriton_crosszero_readblockb32_crossbatchsgcap200_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7jl-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-hiddenRatResidual015-hiddenResVJPTriton-crossBatchSGCap150-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "B7jj plus bounded rational hidden residual 0.15 with generic Triton hidden-residual VJP; loss-agnostic task trajectory combination, not CE-specific", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_paircrossr136_readblocktriton_crosszero_readblockb32_hiddenratresidual015_hiddenresvjptriton_crossbatchsgcap150_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7jm-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-hiddenTanhResidual010-hiddenResVJPTriton-crossBatchSGCap150-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "B7jj plus smooth tanh hidden residual 0.10 with generic Triton hidden-residual VJP; loss-agnostic task trajectory combination, not CE-specific", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_paircrossr136_readblocktriton_crosszero_readblockb32_hiddentanhresidual010_hiddenresvjptriton_crossbatchsgcap150_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7jn-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR96-readblocktriton-crossZero-b32-hiddenTanhResidual010-hiddenResVJPTriton-crossBatchSGCap150-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "B7jm lower-rank R96 cost bracket: smooth tanh hidden residual 0.10 plus stop-gradient batch cap; generic VJP, not CE-specific", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_paircrossr96_readblocktriton_crosszero_readblockb32_hiddentanhresidual010_hiddenresvjptriton_crossbatchsgcap150_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7jo-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR80-readblocktriton-crossZero-b32-hiddenTanhResidual010-hiddenResVJPTriton-crossBatchSGCap150-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "B7jm lower-rank R80 cost bracket: smooth tanh hidden residual 0.10 plus stop-gradient batch cap; generic VJP, not CE-specific", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_paircrossr80_readblocktriton_crosszero_readblockb32_hiddentanhresidual010_hiddenresvjptriton_crossbatchsgcap150_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7jp-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR112-readblocktriton-crossZero-b32-hiddenTanhResidual010-hiddenResVJPTriton-crossBatchSGCap150-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "B7jn/B7jm midpoint R112 expression-capacity bracket: smooth tanh hidden residual 0.10 plus stop-gradient batch cap; generic VJP, not CE-specific", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_paircrossr112_readblocktriton_crosszero_readblockb32_hiddentanhresidual010_hiddenresvjptriton_crossbatchsgcap150_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7jq-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR120-readblocktriton-crossZero-b32-hiddenTanhResidual010-hiddenResVJPTriton-crossBatchSGCap150-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "B7jn/B7jm midpoint R120 expression-capacity bracket: smooth tanh hidden residual 0.10 plus stop-gradient batch cap; generic VJP, not CE-specific", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_paircrossr120_readblocktriton_crosszero_readblockb32_hiddentanhresidual010_hiddenresvjptriton_crossbatchsgcap150_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7jr-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR128-readblocktriton-crossZero-b32-hiddenTanhResidual010-hiddenResVJPTriton-crossBatchSGCap150-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "B7jq/B7jm high-rank R128 expression-capacity bracket: smooth tanh hidden residual 0.10 plus stop-gradient batch cap; generic VJP, not CE-specific", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_paircrossr128_readblocktriton_crosszero_readblockb32_hiddentanhresidual010_hiddenresvjptriton_crossbatchsgcap150_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7js-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR132-readblocktriton-crossZero-b32-hiddenTanhResidual010-hiddenResVJPTriton-crossBatchSGCap150-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "B7jr/B7jm narrow R132 expression-capacity bracket: smooth tanh hidden residual 0.10 plus stop-gradient batch cap; generic VJP, not CE-specific", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_paircrossr132_readblocktriton_crosszero_readblockb32_hiddentanhresidual010_hiddenresvjptriton_crossbatchsgcap150_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7jt-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR132-readblocktriton-crossZero-b32-hiddenTanhResidual005-hiddenResVJPTriton-crossBatchSGCap150-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "B7js task-trajectory damping repair: keep R132 and stop-gradient batch cap but reduce smooth tanh hidden residual from 0.10 to 0.05; generic VJP, not CE-specific", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_paircrossr132_readblocktriton_crosszero_readblockb32_hiddentanhresidual005_hiddenresvjptriton_crossbatchsgcap150_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7ju-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR132-readblocktriton-crossZero-b32-hiddenCenterTanhResidual010-hiddenResVJPTriton-crossBatchSGCap150-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "B7js centered hidden-residual repair: keep R132 and residual amplitude 0.10 but subtract per-sample tanh hidden mean before the residual; generic VJP, not CE-specific", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_paircrossr132_readblocktriton_crosszero_readblockb32_hiddencentertanhresidual010_hiddenresvjptriton_crossbatchsgcap150_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7jv-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR132-readblocktriton-crossZero-b32-hiddenTanhResidual010-hiddenResVJPTriton-crossBatchSGCap125-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "B7js lower pair-logit cap repair: keep R132 and hidden tanh residual 0.10 but reduce stop-gradient batch cap from 1.50 to 1.25; generic VJP, not CE-specific", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_paircrossr132_readblocktriton_crosszero_readblockb32_hiddentanhresidual010_hiddenresvjptriton_crossbatchsgcap125_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7jw-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR132-readblocktriton-crossZero-b32-crossBatchSGCap150-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "B7jj R132 no-hidden-residual bracket: keep stop-gradient batch cap 1.50 and remove the hidden residual channel to isolate task harm; generic VJP, not CE-specific", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_paircrossr132_readblocktriton_crosszero_readblockb32_crossbatchsgcap150_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7jx-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR132-readblocktriton-crossZero-b32-crossBatchSGCap125-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "B7jw no-hidden-residual lower-cap bracket: keep R132 and generic stop-gradient batch VJP but reduce pair-logit cap from 1.50 to 1.25; not CE-specific", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_paircrossr132_readblocktriton_crosszero_readblockb32_crossbatchsgcap125_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7jy-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR132-readblocktriton-crossZero-b32-hiddenSignSqResidual010-hiddenResVJPTriton-crossBatchSGCap150-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "B7jw/B7jx follow-up: sign-preserving bounded quadratic hidden residual h*abs(h)/(1+h^2) scale 0.10 with generic Triton VJP; no CE-specific tuning", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_paircrossr132_readblocktriton_crosszero_readblockb32_hiddensignsqresidual010_hiddenresvjptriton_crossbatchsgcap150_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7jz-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR132-readblocktriton-crossZero-b32-hiddenSignSqResidual015-hiddenResVJPTriton-crossBatchSGCap150-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "B7jy amplitude bracket: sign-preserving bounded quadratic hidden residual h*abs(h)/(1+h^2) scale 0.15 with generic Triton VJP; no CE-specific tuning", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_paircrossr132_readblocktriton_crosszero_readblockb32_hiddensignsqresidual015_hiddenresvjptriton_crossbatchsgcap150_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7ka-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR128-readblocktriton-crossZero-b32-hiddenSignSqResidual010-hiddenResVJPTriton-crossBatchSGCap150-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "B7jy cost bracket: reduce paircross rank R132 -> R128 while keeping sign-preserving bounded quadratic hidden residual scale 0.10; generic VJP, not CE-specific", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_paircrossr128_readblocktriton_crosszero_readblockb32_hiddensignsqresidual010_hiddenresvjptriton_crossbatchsgcap150_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7kb-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR120-readblocktriton-crossZero-b32-hiddenSignSqResidual010-hiddenResVJPTriton-crossBatchSGCap150-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "B7jy lower-cost bracket: reduce paircross rank R132 -> R120 while keeping sign-preserving bounded quadratic hidden residual scale 0.10; generic VJP, not CE-specific", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_paircrossr120_readblocktriton_crosszero_readblockb32_hiddensignsqresidual010_hiddenresvjptriton_crossbatchsgcap150_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7jf-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-crossBatchCap150-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 B7ir anchor with batch-centered pair-logit tanh cap 1.50 and generic VJP", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_paircrossr136_readblocktriton_crosszero_readblockb32_crossbatchcap150_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7jg-RationalKAT-flashgroup-G16-h32-linearresGain19200-paircrossR136-readblocktriton-crossZero-b32-crossBatchCap200-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 B7ir anchor with batch-centered pair-logit tanh cap 2.00 and generic VJP", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_paircrossr136_readblocktriton_crosszero_readblockb32_crossbatchcap200_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7ht-RationalKAT-flashgroup-G16-h32-linearresGain19200-linearResRamp09600-paircrossR136-readblocktriton-crossZero-b32-freezeBackbone-hiddenBias-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 B7ej with loss-agnostic linear residual scale ramp 96.0 to 192.0", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_linearresramp09600_paircrossr136_readblocktriton_crosszero_readblockb32_freezebackbone_hiddenbias", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7hu-RationalKAT-flashgroup-G16-h32-linearresGain19200-linearResRamp14400-paircrossR136-readblocktriton-crossZero-b32-freezeBackbone-hiddenBias-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 B7ej with loss-agnostic linear residual scale ramp 144.0 to 192.0", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_linearresramp14400_paircrossr136_readblocktriton_crosszero_readblockb32_freezebackbone_hiddenbias", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7hv-RationalKAT-flashgroup-G16-h32-linearresGain19200-linearResRamp14400-paircrossR136-readblocktriton-crossZero-b32-hiddenRatReadoutZero-hiddenRatScale050-hiddenTailGradTriton-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 B7hn with loss-agnostic linear residual scale ramp 144.0 to 192.0", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_linearresramp14400_paircrossr136_readblocktriton_crosszero_readblockb32_hiddenratreadoutzero_hiddenratscale050_hiddentailgradtriton_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7hw-RationalKAT-flashgroup-G16-h32-linearresGain19200-linearResRamp09600-paircrossR136-readblocktriton-crossZero-b32-hiddenRatReadoutZero-hiddenRatScale050-hiddenTailGradTriton-freezeBackbone-hiddenBias-manualAdamW-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 B7hn with loss-agnostic linear residual scale ramp 96.0 to 192.0", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearresgain19200_linearresramp09600_paircrossr136_readblocktriton_crosszero_readblockb32_hiddenratreadoutzero_hiddenratscale050_hiddentailgradtriton_freezebackbone_hiddenbias_manualadamw", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7cq-RationalKAT-flashgroup-G16-h40-linearres-paircrossR136-readblocktriton-crossZero-b32-crossCap200-L3", "Rational", "rational_kat_flashgroup", 1, 40, "third_party/FlashKAT/rational_kat grouped rational h40 R136 crosszero B32 block-readout with bounded pair logits cap 2.00", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearres_paircrossr136_readblocktriton_crosszero_readblockb32_crosscap200", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7cr-RationalKAT-flashgroup-G16-h32-linearres-paircrossR136-readblocktriton-crossZero-b32-crossCap200-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 R136 crosszero B32 block-readout with bounded pair logits cap 2.00", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearres_paircrossr136_readblocktriton_crosszero_readblockb32_crosscap200", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7cs-RationalKAT-flashgroup-G16-h40-linearres-paircrossR136-readblocktriton-crossZero-b32-crossWarm-L3", "Rational", "rational_kat_flashgroup", 1, 40, "third_party/FlashKAT/rational_kat grouped rational h40 R136 crosszero B32 block-readout with cross-readout scale warmup", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearres_paircrossr136_readblocktriton_crosszero_readblockb32_crosswarm", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7ct-RationalKAT-flashgroup-G16-h32-linearres-paircrossR136-readblocktriton-crossZero-b32-crossWarm-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 R136 crosszero B32 block-readout with cross-readout scale warmup", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearres_paircrossr136_readblocktriton_crosszero_readblockb32_crosswarm", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7cu-RationalKAT-flashgroup-G16-h40-linearres-paircrossR136-readblocktriton-crossZero-b32-readGradR64-crossWarm-L3", "Rational", "rational_kat_flashgroup", 1, 40, "third_party/FlashKAT/rational_kat grouped rational h40 R136 crosszero B32 block-readout with R64 grad tiling and cross-readout scale warmup", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearres_paircrossr136_readblocktriton_crosszero_readblockb32_readgradr64_crosswarm", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7cv-RationalKAT-flashgroup-G16-h32-linearres-paircrossR136-readblocktriton-crossZero-b32-readGradR64-crossWarm-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 R136 crosszero B32 block-readout with R64 grad tiling and cross-readout scale warmup", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearres_paircrossr136_readblocktriton_crosszero_readblockb32_readgradr64_crosswarm", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7cw-RationalKAT-flashgroup-G16-h40-linearres-paircrossR136-readblocktriton-crossZero-b32-readGradR128-crossWarm-L3", "Rational", "rational_kat_flashgroup", 1, 40, "third_party/FlashKAT/rational_kat grouped rational h40 R136 crosszero B32 block-readout with R128 grad tiling and cross-readout scale warmup", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearres_paircrossr136_readblocktriton_crosszero_readblockb32_readgradr128_crosswarm", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7cx-RationalKAT-flashgroup-G16-h32-linearres-paircrossR136-readblocktriton-crossZero-b32-readGradR128-crossWarm-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 R136 crosszero B32 block-readout with R128 grad tiling and cross-readout scale warmup", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearres_paircrossr136_readblocktriton_crosszero_readblockb32_readgradr128_crosswarm", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7cy-RationalKAT-flashgroup-G16-h40-linearres-paircrossR136-readblocktriton-crossZero-b32-crossCapFused200-L3", "Rational", "rational_kat_flashgroup", 1, 40, "third_party/FlashKAT/rational_kat grouped rational h40 R136 crosszero B32 block-readout with fused bounded pair-logit VJP", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearres_paircrossr136_readblocktriton_crosszero_readblockb32_crosscapfused200", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7cz-RationalKAT-flashgroup-G16-h32-linearres-paircrossR136-readblocktriton-crossZero-b32-crossCapFused200-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 R136 crosszero B32 block-readout with fused bounded pair-logit VJP", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearres_paircrossr136_readblocktriton_crosszero_readblockb32_crosscapfused200", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7da-RationalKAT-flashgroup-G16-h40-linearres-paircrossR136-readblocktriton-crossZero-b32-crossCapFused200Atomic-L3", "Rational", "rational_kat_flashgroup", 1, 40, "third_party/FlashKAT/rational_kat grouped rational h40 R136 crosszero B32 block-readout with fused bounded pair-logit atomic VJP", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearres_paircrossr136_readblocktriton_crosszero_readblockb32_crosscapfused200_capatomic", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7db-RationalKAT-flashgroup-G16-h32-linearres-paircrossR136-readblocktriton-crossZero-b32-crossCapFused200Atomic-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 R136 crosszero B32 block-readout with fused bounded pair-logit atomic VJP", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearres_paircrossr136_readblocktriton_crosszero_readblockb32_crosscapfused200_capatomic", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7dc-RationalKAT-flashgroup-G16-h40-linearres-pairReadBucketR136G32-crossZero-L3", "Rational", "rational_kat_flashgroup", 1, 40, "third_party/FlashKAT/rational_kat grouped rational h40 R136 pair features bucketed to G32 trainable readout", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearres_pairreadbucketr136g32_crosszero", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7dd-RationalKAT-flashgroup-G16-h40-linearres-pairReadBucketR136G64-crossZero-L3", "Rational", "rational_kat_flashgroup", 1, 40, "third_party/FlashKAT/rational_kat grouped rational h40 R136 pair features bucketed to G64 trainable readout", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearres_pairreadbucketr136g64_crosszero", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7de-RationalKAT-flashgroup-G16-h40-linearres-pairReadBucketR136G16-crossZero-L3", "Rational", "rational_kat_flashgroup", 1, 40, "third_party/FlashKAT/rational_kat grouped rational h40 R136 pair features bucketed to G16 trainable readout", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearres_pairreadbucketr136g16_crosszero", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7df-RationalKAT-flashgroup-G16-h32-linearres-pairReadBucketR136G32-crossZero-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 R136 pair features bucketed to G32 trainable readout", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearres_pairreadbucketr136g32_crosszero", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7dg-RationalKAT-flashgroup-G16-h40-linearres-projSqR64-sqZero-L3", "Rational", "rational_kat_flashgroup", 1, 40, "third_party/FlashKAT/rational_kat grouped rational h40 fixed DCT projection-square R64 readout", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearres_projsqr64_sqzero", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7dh-RationalKAT-flashgroup-G16-h32-linearres-projSqR64-sqZero-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 fixed DCT projection-square R64 readout", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearres_projsqr64_sqzero", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7di-RationalKAT-flashgroup-G16-h40-linearres-projSqR32-sqZero-L3", "Rational", "rational_kat_flashgroup", 1, 40, "third_party/FlashKAT/rational_kat grouped rational h40 fixed DCT projection-square R32 readout", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearres_projsqr32_sqzero", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7dj-RationalKAT-flashgroup-G16-h40-linearres-projSqR96-sqZero-L3", "Rational", "rational_kat_flashgroup", 1, 40, "third_party/FlashKAT/rational_kat grouped rational h40 fixed DCT projection-square R96 readout", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearres_projsqr96_sqzero", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7dk-RationalKAT-flashgroup-G16-h32-linearres-projSqR96-sqZero-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 fixed DCT projection-square R96 readout", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearres_projsqr96_sqzero", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7dl-RationalKAT-flashgroup-G16-h40-linearres-projBilinR64-bilinZero-L3", "Rational", "rational_kat_flashgroup", 1, 40, "third_party/FlashKAT/rational_kat grouped rational h40 fixed dual-DCT projection-bilinear R64 readout", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearres_projbilinr64_bilinzero", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7dm-RationalKAT-flashgroup-G16-h32-linearres-projBilinR64-bilinZero-L3", "Rational", "rational_kat_flashgroup", 1, 32, "third_party/FlashKAT/rational_kat grouped rational h32 fixed dual-DCT projection-bilinear R64 readout", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearres_projbilinr64_bilinzero", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7dn-RationalKAT-flashgroup-G16-h40-linearres-projBilinR96-bilinZero-L3", "Rational", "rational_kat_flashgroup", 1, 40, "third_party/FlashKAT/rational_kat grouped rational h40 fixed dual-DCT projection-bilinear R96 readout", 0, 1, 0, 0, 1, uses_dense_basis_tensor=0, basis_order=5, init_variant="flashkat_g16_gelu_linearres_projbilinr96_bilinzero", model_kind="grouped_rational_kat"),
        PrimitiveSpec("B7a-RationalKAT-lite-safe-den-K4", "Rational", "rational_kat_lite", 4, h(4), "third_party/rational_kat_cu", 0, 1, 0, 0, 1, basis_order=4),
    ]


def basis_functional_direction(module: nn.Module, mode: str) -> List[torch.Tensor]:
    if hasattr(module, "basis_functional_direction") and not isinstance(module, PrimitiveKAN):
        return module.basis_functional_direction(mode)  # type: ignore[no-any-return]
    grads: List[torch.Tensor] = []
    with torch.no_grad():
        for p in [module.w1, module.w2]:
            direction = -p.detach().clone()
            if mode in {"basis_aware_snr_projected", "basis_aware_orthogonal"}:
                # Damp only the non-linear channels and keep the first channel
                # closer to task descent; this is a basis-level diagnostic, not
                # an official controller.
                mask = torch.ones_like(direction)
                mask[..., 0] = 0.25
                direction = direction * mask
            grads.append(direction)
    return grads
