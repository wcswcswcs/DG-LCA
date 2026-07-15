"""Fused RTGF optimizer step kernels for small-K edge banks.

The v23.26 efficiency path profiles the radial-tangential generator without
trace telemetry.  The unfused PyTorch implementation is semantically clear but
launches many small elementwise/einsum kernels per edge bank.  This module keeps
the same update algebra for fp32 CUDA tensors with K <= 5 and fuses one
parameter tensor update into a single Triton launch.
"""

from __future__ import annotations

import torch

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
    def _rtgf_step_kernel(
        param_ptr,
        grad_ptr,
        state_m_ptr,
        state_v_ptr,
        metric_ptr,
        metric_inv_ptr,
        rows: tl.constexpr,
        K: tl.constexpr,
        LR: tl.constexpr,
        WEIGHT_DECAY: tl.constexpr,
        BETA1: tl.constexpr,
        BETA2: tl.constexpr,
        BC1: tl.constexpr,
        BC2: tl.constexpr,
        EPS: tl.constexpr,
        NORM_FLOOR: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        row_offsets = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        k_offsets = tl.arange(0, BLOCK_K)
        row_mask = row_offsets < rows
        k_mask = k_offsets < K
        elem_offsets = row_offsets[:, None] * K + k_offsets[None, :]
        elem_mask = row_mask[:, None] & k_mask[None, :]

        a = tl.load(param_ptr + elem_offsets, mask=elem_mask, other=0.0)
        grad = tl.load(grad_ptr + elem_offsets, mask=elem_mask, other=0.0)
        old_m = tl.load(state_m_ptr + elem_offsets, mask=elem_mask, other=0.0)

        col = k_offsets[None, :]
        row = k_offsets[:, None]
        matrix_mask = (row < K) & (col < K)
        metric_t = tl.load(metric_ptr + col * K + row, mask=matrix_mask, other=0.0)
        metric_inv_t = tl.load(metric_inv_ptr + col * K + row, mask=matrix_mask, other=0.0)

        new_m = old_m * BETA1 + grad * (1.0 - BETA1)
        nat_grad = tl.dot(grad, metric_inv_t, input_precision="tf32x3")
        energy = tl.sum(grad * nat_grad, axis=1) / K
        energy = tl.maximum(energy, 0.0)
        old_v = tl.load(state_v_ptr + row_offsets, mask=row_mask, other=0.0)
        new_v = old_v * BETA2 + energy * (1.0 - BETA2)

        m_hat = new_m / BC1
        v_hat = new_v / BC2
        nat_m = tl.dot(m_hat, metric_inv_t, input_precision="tf32x3")
        proposal = -LR * nat_m / (tl.sqrt(v_hat)[:, None] + EPS)

        ma = tl.dot(a, metric_t, input_precision="tf32x3")
        r2 = tl.maximum(tl.sum(a * ma, axis=1), 0.0)
        denom_r2 = tl.maximum(r2, NORM_FLOOR * NORM_FLOOR)
        alpha = tl.sum(proposal * ma, axis=1) / denom_r2
        tangent = proposal - alpha[:, None] * a
        mt = tl.dot(tangent, metric_t, input_precision="tf32x3")
        tangent_norm = tl.sqrt(tl.maximum(tl.sum(tangent * mt, axis=1), 0.0))
        r = tl.sqrt(r2)
        q = tangent / tl.maximum(tangent_norm, EPS)[:, None]
        angle = tangent_norm / tl.maximum(r, EPS)
        shape = tl.cos(angle)[:, None] * a + (r * tl.sin(angle))[:, None] * q
        new_value = tl.exp(alpha)[:, None] * shape
        bootstrap_value = a + proposal
        use_bootstrap = r < NORM_FLOOR
        new_value = tl.where(use_bootstrap[:, None], bootstrap_value, new_value)
        if WEIGHT_DECAY != 0.0:
            new_value = new_value * (1.0 - LR * WEIGHT_DECAY)

        tl.store(param_ptr + elem_offsets, new_value, mask=elem_mask)
        tl.store(state_m_ptr + elem_offsets, new_m, mask=elem_mask)
        tl.store(state_v_ptr + row_offsets, new_v, mask=row_mask)


def rtgf_step_(
    param: torch.Tensor,
    grad: torch.Tensor,
    state_m: torch.Tensor,
    state_v: torch.Tensor,
    metric: torch.Tensor,
    metric_inv: torch.Tensor,
    *,
    step: int,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    norm_floor: float,
    weight_decay: float,
) -> bool:
    """Apply one fused in-place RTGF update, returning False if unsupported."""

    if not TRITON_AVAILABLE:
        return False
    if param.device.type != "cuda" or grad.device.type != "cuda":
        return False
    if param.dtype != torch.float32 or grad.dtype != torch.float32:
        return False
    if not (param.is_contiguous() and grad.is_contiguous() and state_m.is_contiguous() and state_v.is_contiguous()):
        return False
    if metric.device != param.device or metric_inv.device != param.device:
        return False
    if metric.dtype != param.dtype or metric_inv.dtype != param.dtype:
        return False
    k = int(param.shape[-1])
    if k < 1 or k > 5:
        return False
    if int(metric.shape[0]) != k or int(metric.shape[1]) != k:
        return False
    rows = int(param.numel() // k)
    if rows <= 0 or int(state_v.numel()) != rows:
        return False
    block_rows = 128
    # tl.dot requires the reduction dimension to be at least 16.  The real
    # edge-basis dimension is still K; channels >= K are masked to zero.
    block_k = 16
    grid = (triton.cdiv(rows, block_rows),)
    _rtgf_step_kernel[grid](
        param,
        grad,
        state_m,
        state_v,
        metric,
        metric_inv,
        rows,
        k,
        float(lr),
        float(weight_decay),
        float(beta1),
        float(beta2),
        float(1.0 - beta1 ** int(step)),
        float(1.0 - beta2 ** int(step)),
        float(eps),
        float(norm_floor),
        BLOCK_ROWS=block_rows,
        BLOCK_K=block_k,
        num_warps=4,
    )
    return True
