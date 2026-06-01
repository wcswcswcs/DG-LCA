"""Lower-level fused hinge/quadratic kernels for SimpleFastTaskGeometryKAN.

This module owns the v12.6 FHQ Triton path.  Experiment runners should call the
public helpers here instead of carrying kernel definitions inline.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except Exception:  # pragma: no cover - environment dependent
    triton = None
    tl = None
    TRITON_AVAILABLE = False


if TRITON_AVAILABLE:

    @triton.jit
    def _fhq_q_kernel(
        x,
        mu,
        std,
        proj,
        q_std,
        q_out,
        q2_sum,
        D: tl.constexpr,
        H: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_H: tl.constexpr,
        Q_TANH: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_h = tl.program_id(1)
        offs_d = tl.arange(0, BLOCK_D)
        offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
        mask_d = offs_d < D
        mask_h = offs_h < H
        xv = tl.load(x + pid_b * D + offs_d, mask=mask_d, other=0.0)
        muv = tl.load(mu + offs_d, mask=mask_d, other=0.0)
        stdv = tl.load(std + offs_d, mask=mask_d, other=1.0)
        z = (xv - muv) / tl.maximum(stdv, 1.0e-4)
        z = tl.minimum(3.0, tl.maximum(-3.0, z))
        p = tl.load(proj + offs_d[:, None] * H + offs_h[None, :], mask=mask_d[:, None] & mask_h[None, :], other=0.0)
        acc = tl.sum(z[:, None] * p, axis=0)
        scale = tl.load(q_std + offs_h, mask=mask_h, other=1.0)
        qv = acc / tl.maximum(scale, 1.0e-4)
        if Q_TANH:
            qv = 2.0 / (1.0 + tl.exp(-2.0 * qv)) - 1.0
        tl.store(q_out + pid_b * H + offs_h, qv, mask=mask_h)
        tl.atomic_add(q2_sum + offs_h, qv * qv, sem="relaxed", mask=mask_h)

    @triton.jit
    def _fhq_single_logits_kernel(
        x,
        mu,
        std,
        norm_mean,
        mean_mean,
        group_abs_mean,
        direct_readout,
        q_out,
        q2_sum,
        quad_readout,
        branch_scale,
        logit_gain,
        bias,
        logits_out,
        direct_logits_out,
        quad_logits_out,
        B: tl.constexpr,
        D: tl.constexpr,
        H: tl.constexpr,
        C: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_H: tl.constexpr,
        C_BLOCK: tl.constexpr,
        CENTER: tl.constexpr,
        SQRT_D: tl.constexpr,
        Q_RMS: tl.constexpr,
        CLASS_BRANCH: tl.constexpr,
        CLASS_GAIN: tl.constexpr,
        NORM_ABS: tl.constexpr,
        MEAN_STAT: tl.constexpr,
        MEAN_OFFSET: tl.constexpr,
        GROUP_COUNT: tl.constexpr,
        GROUP_OFFSET: tl.constexpr,
        GROUP_SIZE: tl.constexpr,
        GROUP_SQRT: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        offs_c = tl.arange(0, C_BLOCK)
        mask_c = offs_c < C
        offs_d = tl.arange(0, BLOCK_D)
        mask_d = offs_d < D
        xv = tl.load(x + pid_b * D + offs_d, mask=mask_d, other=0.0)
        muv = tl.load(mu + offs_d, mask=mask_d, other=0.0)
        stdv = tl.load(std + offs_d, mask=mask_d, other=1.0)
        z = (xv - muv) / tl.maximum(stdv, 1.0e-4)
        z = tl.minimum(3.0, tl.maximum(-3.0, z))
        pos = tl.maximum(z - CENTER, 0.0)
        neg = tl.maximum(-z - CENTER, 0.0)
        w0 = tl.load(direct_readout + offs_d[:, None] * C + offs_c[None, :], mask=mask_d[:, None] & mask_c[None, :], other=0.0)
        wp = tl.load(direct_readout + (D + offs_d[:, None]) * C + offs_c[None, :], mask=mask_d[:, None] & mask_c[None, :], other=0.0)
        wn = tl.load(direct_readout + (2 * D + offs_d[:, None]) * C + offs_c[None, :], mask=mask_d[:, None] & mask_c[None, :], other=0.0)
        direct_acc = tl.sum(z[:, None] * w0 + pos[:, None] * wp + neg[:, None] * wn, axis=0) / SQRT_D
        if NORM_ABS:
            norm_feat = (tl.sum(tl.maximum(z, -z), axis=0) / D - tl.load(norm_mean)) * SQRT_D
            wnorm = tl.load(direct_readout + (3 * D) * C + offs_c, mask=mask_c, other=0.0)
            direct_acc += norm_feat * wnorm / SQRT_D
        if MEAN_STAT:
            mean_feat = (tl.sum(z, axis=0) / D - tl.load(mean_mean)) * SQRT_D
            wmean = tl.load(direct_readout + MEAN_OFFSET * C + offs_c, mask=mask_c, other=0.0)
            direct_acc += mean_feat * wmean / SQRT_D
        if GROUP_COUNT > 0:
            for group_idx in tl.static_range(0, 8):
                if group_idx < GROUP_COUNT:
                    group_mask = mask_d & (offs_d >= group_idx * GROUP_SIZE) & (offs_d < (group_idx + 1) * GROUP_SIZE)
                    group_feat = (tl.sum(tl.where(group_mask, tl.maximum(z, -z), 0.0), axis=0) / GROUP_SIZE - tl.load(group_abs_mean + group_idx)) * GROUP_SQRT
                    wgroup = tl.load(direct_readout + (GROUP_OFFSET + group_idx) * C + offs_c, mask=mask_c, other=0.0)
                    direct_acc += group_feat * wgroup / SQRT_D
        offs_h = tl.arange(0, BLOCK_H)
        mask_h = offs_h < H
        qv = tl.load(q_out + pid_b * H + offs_h, mask=mask_h, other=0.0)
        qmean = tl.load(q2_sum + offs_h, mask=mask_h, other=0.0) / B
        if Q_RMS:
            qv = qv / tl.sqrt(tl.maximum(qmean, 1.0e-8))
            qmean = 1.0
        q2 = qv * qv - qmean
        wq = tl.load(quad_readout + offs_h[:, None] * C + offs_c[None, :], mask=mask_h[:, None] & mask_c[None, :], other=0.0)
        wq2 = tl.load(quad_readout + (H + offs_h[:, None]) * C + offs_c[None, :], mask=mask_h[:, None] & mask_c[None, :], other=0.0)
        quad_acc = tl.sum(qv[:, None] * wq + q2[:, None] * wq2, axis=0)
        if CLASS_GAIN:
            raw_gain = tl.load(logit_gain + offs_c, mask=mask_c, other=1.0)
        else:
            raw_gain = tl.load(logit_gain) + tl.zeros((C_BLOCK,), dtype=tl.float32)
        gain = tl.minimum(4.0, tl.maximum(0.25, raw_gain))
        if CLASS_BRANCH:
            branch0 = tl.load(branch_scale + offs_c, mask=mask_c, other=0.0)
            branch1 = tl.load(branch_scale + C + offs_c, mask=mask_c, other=0.0)
        else:
            branch0 = tl.load(branch_scale + 0)
            branch1 = tl.load(branch_scale + 1)
        logits = gain * (branch0 * direct_acc + branch1 * quad_acc + tl.load(bias + offs_c, mask=mask_c, other=0.0))
        tl.store(logits_out + pid_b * C + offs_c, logits, mask=mask_c)
        tl.store(direct_logits_out + pid_b * C + offs_c, direct_acc, mask=mask_c)
        tl.store(quad_logits_out + pid_b * C + offs_c, quad_acc, mask=mask_c)

    @triton.jit
    def _fhq_single_sqdiag_logits_kernel(
        x,
        mu,
        std,
        z2_mean,
        z2_mean_extra,
        norm_mean,
        mean_mean,
        group_abs_mean,
        direct_readout,
        q_out,
        q2_sum,
        quad_readout,
        branch_scale,
        logit_gain,
        bias,
        logits_out,
        direct_logits_out,
        quad_logits_out,
        B: tl.constexpr,
        D: tl.constexpr,
        H: tl.constexpr,
        C: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_H: tl.constexpr,
        C_BLOCK: tl.constexpr,
        CENTER: tl.constexpr,
        SQRT_D: tl.constexpr,
        Q_RMS: tl.constexpr,
        CLASS_BRANCH: tl.constexpr,
        CLASS_GAIN: tl.constexpr,
        ABS_DIAG: tl.constexpr,
        ABS_QUAD: tl.constexpr,
        ABS_MIX_SQ: tl.constexpr,
        ABS_MIX_SQ_SCALE: tl.constexpr,
        CUBIC_DIAG: tl.constexpr,
        NORM_ABS: tl.constexpr,
        MEAN_STAT: tl.constexpr,
        MEAN_OFFSET: tl.constexpr,
        GROUP_COUNT: tl.constexpr,
        GROUP_OFFSET: tl.constexpr,
        GROUP_SIZE: tl.constexpr,
        GROUP_SQRT: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        offs_c = tl.arange(0, C_BLOCK)
        mask_c = offs_c < C
        offs_d = tl.arange(0, BLOCK_D)
        mask_d = offs_d < D
        xv = tl.load(x + pid_b * D + offs_d, mask=mask_d, other=0.0)
        muv = tl.load(mu + offs_d, mask=mask_d, other=0.0)
        stdv = tl.load(std + offs_d, mask=mask_d, other=1.0)
        z = (xv - muv) / tl.maximum(stdv, 1.0e-4)
        z = tl.minimum(3.0, tl.maximum(-3.0, z))
        pos = tl.maximum(z - CENTER, 0.0)
        neg = tl.maximum(-z - CENTER, 0.0)
        if ABS_QUAD:
            abs_tail = tl.maximum(z, -z) - tl.load(z2_mean + offs_d, mask=mask_d, other=0.0)
            sq = z * z - tl.load(z2_mean_extra + offs_d, mask=mask_d, other=0.0)
        elif ABS_MIX_SQ:
            abs_tail = tl.maximum(z, -z) - tl.load(z2_mean + offs_d, mask=mask_d, other=0.0)
            sq = abs_tail + ABS_MIX_SQ_SCALE * (z * z - tl.load(z2_mean_extra + offs_d, mask=mask_d, other=0.0))
        elif CUBIC_DIAG:
            sq = z * z * z - tl.load(z2_mean + offs_d, mask=mask_d, other=0.0)
        elif ABS_DIAG:
            sq = tl.maximum(z, -z) - tl.load(z2_mean + offs_d, mask=mask_d, other=0.0)
        else:
            sq = z * z - tl.load(z2_mean + offs_d, mask=mask_d, other=0.0)
        w0 = tl.load(direct_readout + offs_d[:, None] * C + offs_c[None, :], mask=mask_d[:, None] & mask_c[None, :], other=0.0)
        wp = tl.load(direct_readout + (D + offs_d[:, None]) * C + offs_c[None, :], mask=mask_d[:, None] & mask_c[None, :], other=0.0)
        wn = tl.load(direct_readout + (2 * D + offs_d[:, None]) * C + offs_c[None, :], mask=mask_d[:, None] & mask_c[None, :], other=0.0)
        ws = tl.load(direct_readout + (3 * D + offs_d[:, None]) * C + offs_c[None, :], mask=mask_d[:, None] & mask_c[None, :], other=0.0)
        direct_term = z[:, None] * w0 + pos[:, None] * wp + neg[:, None] * wn
        if ABS_QUAD:
            wqdiag = tl.load(direct_readout + (4 * D + offs_d[:, None]) * C + offs_c[None, :], mask=mask_d[:, None] & mask_c[None, :], other=0.0)
            direct_term += abs_tail[:, None] * ws + sq[:, None] * wqdiag
        else:
            direct_term += sq[:, None] * ws
        direct_acc = tl.sum(direct_term, axis=0) / SQRT_D
        if NORM_ABS:
            norm_feat = (tl.sum(tl.maximum(z, -z), axis=0) / D - tl.load(norm_mean)) * SQRT_D
            if ABS_QUAD:
                wnorm = tl.load(direct_readout + (5 * D) * C + offs_c, mask=mask_c, other=0.0)
            else:
                wnorm = tl.load(direct_readout + (4 * D) * C + offs_c, mask=mask_c, other=0.0)
            direct_acc += norm_feat * wnorm / SQRT_D
        if MEAN_STAT:
            mean_feat = (tl.sum(z, axis=0) / D - tl.load(mean_mean)) * SQRT_D
            wmean = tl.load(direct_readout + MEAN_OFFSET * C + offs_c, mask=mask_c, other=0.0)
            direct_acc += mean_feat * wmean / SQRT_D
        if GROUP_COUNT > 0:
            for group_idx in tl.static_range(0, 8):
                if group_idx < GROUP_COUNT:
                    group_mask = mask_d & (offs_d >= group_idx * GROUP_SIZE) & (offs_d < (group_idx + 1) * GROUP_SIZE)
                    group_feat = (tl.sum(tl.where(group_mask, tl.maximum(z, -z), 0.0), axis=0) / GROUP_SIZE - tl.load(group_abs_mean + group_idx)) * GROUP_SQRT
                    wgroup = tl.load(direct_readout + (GROUP_OFFSET + group_idx) * C + offs_c, mask=mask_c, other=0.0)
                    direct_acc += group_feat * wgroup / SQRT_D
        offs_h = tl.arange(0, BLOCK_H)
        mask_h = offs_h < H
        qv = tl.load(q_out + pid_b * H + offs_h, mask=mask_h, other=0.0)
        qmean = tl.load(q2_sum + offs_h, mask=mask_h, other=0.0) / B
        if Q_RMS:
            qv = qv / tl.sqrt(tl.maximum(qmean, 1.0e-8))
            qmean = 1.0
        q2 = qv * qv - qmean
        wq = tl.load(quad_readout + offs_h[:, None] * C + offs_c[None, :], mask=mask_h[:, None] & mask_c[None, :], other=0.0)
        wq2 = tl.load(quad_readout + (H + offs_h[:, None]) * C + offs_c[None, :], mask=mask_h[:, None] & mask_c[None, :], other=0.0)
        quad_acc = tl.sum(qv[:, None] * wq + q2[:, None] * wq2, axis=0)
        if CLASS_GAIN:
            raw_gain = tl.load(logit_gain + offs_c, mask=mask_c, other=1.0)
        else:
            raw_gain = tl.load(logit_gain) + tl.zeros((C_BLOCK,), dtype=tl.float32)
        gain = tl.minimum(4.0, tl.maximum(0.25, raw_gain))
        if CLASS_BRANCH:
            branch0 = tl.load(branch_scale + offs_c, mask=mask_c, other=0.0)
            branch1 = tl.load(branch_scale + C + offs_c, mask=mask_c, other=0.0)
        else:
            branch0 = tl.load(branch_scale + 0)
            branch1 = tl.load(branch_scale + 1)
        logits = gain * (branch0 * direct_acc + branch1 * quad_acc + tl.load(bias + offs_c, mask=mask_c, other=0.0))
        tl.store(logits_out + pid_b * C + offs_c, logits, mask=mask_c)
        tl.store(direct_logits_out + pid_b * C + offs_c, direct_acc, mask=mask_c)
        tl.store(quad_logits_out + pid_b * C + offs_c, quad_acc, mask=mask_c)

    @triton.jit
    def _fhq_twohinge_logits_kernel(
        x,
        mu,
        std,
        direct_readout,
        q_out,
        q2_sum,
        quad_readout,
        branch_scale,
        logit_gain,
        bias,
        logits_out,
        direct_logits_out,
        quad_logits_out,
        B: tl.constexpr,
        D: tl.constexpr,
        H: tl.constexpr,
        C: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_H: tl.constexpr,
        C_BLOCK: tl.constexpr,
        SQRT_D: tl.constexpr,
        Q_RMS: tl.constexpr,
        CLASS_BRANCH: tl.constexpr,
        CLASS_GAIN: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        offs_c = tl.arange(0, C_BLOCK)
        mask_c = offs_c < C
        offs_d = tl.arange(0, BLOCK_D)
        mask_d = offs_d < D
        xv = tl.load(x + pid_b * D + offs_d, mask=mask_d, other=0.0)
        muv = tl.load(mu + offs_d, mask=mask_d, other=0.0)
        stdv = tl.load(std + offs_d, mask=mask_d, other=1.0)
        z = (xv - muv) / tl.maximum(stdv, 1.0e-4)
        z = tl.minimum(3.0, tl.maximum(-3.0, z))
        pos0 = tl.maximum(z - 0.25, 0.0)
        pos1 = tl.maximum(z - 0.75, 0.0)
        neg0 = tl.maximum(-z - 0.25, 0.0)
        neg1 = tl.maximum(-z - 0.75, 0.0)
        w0 = tl.load(direct_readout + offs_d[:, None] * C + offs_c[None, :], mask=mask_d[:, None] & mask_c[None, :], other=0.0)
        wp0 = tl.load(direct_readout + (D + 2 * offs_d[:, None]) * C + offs_c[None, :], mask=mask_d[:, None] & mask_c[None, :], other=0.0)
        wp1 = tl.load(direct_readout + (D + 2 * offs_d[:, None] + 1) * C + offs_c[None, :], mask=mask_d[:, None] & mask_c[None, :], other=0.0)
        wn0 = tl.load(direct_readout + (3 * D + 2 * offs_d[:, None]) * C + offs_c[None, :], mask=mask_d[:, None] & mask_c[None, :], other=0.0)
        wn1 = tl.load(direct_readout + (3 * D + 2 * offs_d[:, None] + 1) * C + offs_c[None, :], mask=mask_d[:, None] & mask_c[None, :], other=0.0)
        direct_acc = tl.sum(z[:, None] * w0 + pos0[:, None] * wp0 + pos1[:, None] * wp1 + neg0[:, None] * wn0 + neg1[:, None] * wn1, axis=0) / SQRT_D
        offs_h = tl.arange(0, BLOCK_H)
        mask_h = offs_h < H
        qv = tl.load(q_out + pid_b * H + offs_h, mask=mask_h, other=0.0)
        qmean = tl.load(q2_sum + offs_h, mask=mask_h, other=0.0) / B
        if Q_RMS:
            qv = qv / tl.sqrt(tl.maximum(qmean, 1.0e-8))
            qmean = 1.0
        q2 = qv * qv - qmean
        wq = tl.load(quad_readout + offs_h[:, None] * C + offs_c[None, :], mask=mask_h[:, None] & mask_c[None, :], other=0.0)
        wq2 = tl.load(quad_readout + (H + offs_h[:, None]) * C + offs_c[None, :], mask=mask_h[:, None] & mask_c[None, :], other=0.0)
        quad_acc = tl.sum(qv[:, None] * wq + q2[:, None] * wq2, axis=0)
        if CLASS_GAIN:
            raw_gain = tl.load(logit_gain + offs_c, mask=mask_c, other=1.0)
        else:
            raw_gain = tl.load(logit_gain) + tl.zeros((C_BLOCK,), dtype=tl.float32)
        gain = tl.minimum(4.0, tl.maximum(0.25, raw_gain))
        if CLASS_BRANCH:
            branch0 = tl.load(branch_scale + offs_c, mask=mask_c, other=0.0)
            branch1 = tl.load(branch_scale + C + offs_c, mask=mask_c, other=0.0)
        else:
            branch0 = tl.load(branch_scale + 0)
            branch1 = tl.load(branch_scale + 1)
        logits = gain * (branch0 * direct_acc + branch1 * quad_acc + tl.load(bias + offs_c, mask=mask_c, other=0.0))
        tl.store(logits_out + pid_b * C + offs_c, logits, mask=mask_c)
        tl.store(direct_logits_out + pid_b * C + offs_c, direct_acc, mask=mask_c)
        tl.store(quad_logits_out + pid_b * C + offs_c, quad_acc, mask=mask_c)

    @triton.jit
    def _fhq_delta_kernel(logits, y, delta_out, B: tl.constexpr, C: tl.constexpr, C_BLOCK: tl.constexpr):
        pid_b = tl.program_id(0)
        offs_c = tl.arange(0, C_BLOCK)
        mask_c = offs_c < C
        lv = tl.load(logits + pid_b * C + offs_c, mask=mask_c, other=-float("inf"))
        m = tl.max(lv, axis=0)
        ex = tl.exp(lv - m)
        probs = ex / tl.sum(tl.where(mask_c, ex, 0.0), axis=0)
        label = tl.load(y + pid_b)
        tl.store(delta_out + pid_b * C + offs_c, (probs - (offs_c == label)) / B, mask=mask_c)

    @triton.jit
    def _fhq_single_direct_grad_kernel(
        x,
        mu,
        std,
        norm_mean,
        mean_mean,
        group_abs_mean,
        delta,
        branch_scale,
        logit_gain,
        grad_direct,
        B: tl.constexpr,
        D: tl.constexpr,
        C: tl.constexpr,
        F: tl.constexpr,
        BLOCK_B: tl.constexpr,
        BLOCK_F: tl.constexpr,
        C_BLOCK: tl.constexpr,
        CENTER: tl.constexpr,
        SQRT_D: tl.constexpr,
        CLASS_BRANCH: tl.constexpr,
        CLASS_GAIN: tl.constexpr,
        NORM_ABS: tl.constexpr,
        NORM_OFFSET: tl.constexpr,
        MEAN_STAT: tl.constexpr,
        MEAN_OFFSET: tl.constexpr,
        GROUP_COUNT: tl.constexpr,
        GROUP_OFFSET: tl.constexpr,
        GROUP_SIZE: tl.constexpr,
        GROUP_SQRT: tl.constexpr,
    ):
        pid_f = tl.program_id(0)
        offs_f = pid_f * BLOCK_F + tl.arange(0, BLOCK_F)
        offs_c = tl.arange(0, C_BLOCK)
        mask_f = offs_f < F
        mask_c = offs_c < C
        d = offs_f % D
        kind = offs_f // D
        acc = tl.zeros((BLOCK_F, C_BLOCK), dtype=tl.float32)
        for b0 in range(0, B, BLOCK_B):
            offs_b = b0 + tl.arange(0, BLOCK_B)
            mask_b = offs_b < B
            xv = tl.load(x + offs_b[:, None] * D + d[None, :], mask=mask_b[:, None] & mask_f[None, :], other=0.0)
            muv = tl.load(mu + d, mask=mask_f, other=0.0)
            stdv = tl.load(std + d, mask=mask_f, other=1.0)
            z = (xv - muv[None, :]) / tl.maximum(stdv[None, :], 1.0e-4)
            z = tl.minimum(3.0, tl.maximum(-3.0, z))
            feat = tl.where(kind[None, :] == 0, z, tl.where(kind[None, :] == 1, tl.maximum(z - CENTER, 0.0), tl.maximum(-z - CENTER, 0.0)))
            gd = tl.load(delta + offs_b[:, None] * C + offs_c[None, :], mask=mask_b[:, None] & mask_c[None, :], other=0.0)
            if CLASS_BRANCH:
                branch0 = tl.load(branch_scale + offs_c, mask=mask_c, other=0.0)
            else:
                branch0 = tl.load(branch_scale + 0) + tl.zeros((C_BLOCK,), dtype=tl.float32)
            if CLASS_GAIN:
                raw_gain = tl.load(logit_gain + offs_c, mask=mask_c, other=1.0)
            else:
                raw_gain = tl.load(logit_gain) + tl.zeros((C_BLOCK,), dtype=tl.float32)
            gain = tl.minimum(4.0, tl.maximum(0.25, raw_gain))
            acc += tl.dot(tl.trans(feat), gd * gain[None, :] * branch0[None, :] / SQRT_D)
            if NORM_ABS:
                raw_mask = mask_f & (kind == 0)
                partial_abs = tl.sum(tl.where(raw_mask[None, :], tl.maximum(z, -z), 0.0), axis=1)
                mean_term = tl.where(pid_f == 0, tl.load(norm_mean) * SQRT_D, 0.0)
                norm_part = partial_abs * SQRT_D / D - mean_term
                norm_acc = tl.sum(norm_part[:, None] * gd * gain[None, :] * branch0[None, :] / SQRT_D, axis=0)
                tl.atomic_add(grad_direct + NORM_OFFSET * C + offs_c, norm_acc, sem="relaxed", mask=mask_c)
            if MEAN_STAT:
                raw_mask = mask_f & (kind == 0)
                partial_mean = tl.sum(tl.where(raw_mask[None, :], z, 0.0), axis=1)
                mean_term = tl.where(pid_f == 0, tl.load(mean_mean) * SQRT_D, 0.0)
                mean_part = partial_mean * SQRT_D / D - mean_term
                mean_acc = tl.sum(mean_part[:, None] * gd * gain[None, :] * branch0[None, :] / SQRT_D, axis=0)
                tl.atomic_add(grad_direct + MEAN_OFFSET * C + offs_c, mean_acc, sem="relaxed", mask=mask_c)
            if GROUP_COUNT > 0:
                for group_idx in tl.static_range(0, 8):
                    if group_idx < GROUP_COUNT:
                        raw_mask = mask_f & (kind == 0) & (d >= group_idx * GROUP_SIZE) & (d < (group_idx + 1) * GROUP_SIZE)
                        partial_abs = tl.sum(tl.where(raw_mask[None, :], tl.maximum(z, -z), 0.0), axis=1)
                        mean_term = tl.where(pid_f == 0, tl.load(group_abs_mean + group_idx) * GROUP_SQRT, 0.0)
                        group_part = partial_abs * GROUP_SQRT / GROUP_SIZE - mean_term
                        group_acc = tl.sum(group_part[:, None] * gd * gain[None, :] * branch0[None, :] / SQRT_D, axis=0)
                        tl.atomic_add(grad_direct + (GROUP_OFFSET + group_idx) * C + offs_c, group_acc, sem="relaxed", mask=mask_c)
        tl.store(grad_direct + offs_f[:, None] * C + offs_c[None, :], acc, mask=mask_f[:, None] & mask_c[None, :])

    @triton.jit
    def _fhq_zero_direct_row_kernel(
        grad_direct,
        C: tl.constexpr,
        C_BLOCK: tl.constexpr,
        ROW: tl.constexpr,
    ):
        offs_c = tl.arange(0, C_BLOCK)
        mask_c = offs_c < C
        tl.store(grad_direct + ROW * C + offs_c, tl.zeros((C_BLOCK,), dtype=tl.float32), mask=mask_c)

    @triton.jit
    def _fhq_groupabs_direct_grad_kernel(
        x,
        mu,
        std,
        group_abs_mean,
        delta,
        branch_scale,
        logit_gain,
        grad_direct,
        B: tl.constexpr,
        D: tl.constexpr,
        C: tl.constexpr,
        GROUP_OFFSET: tl.constexpr,
        GROUP_SIZE: tl.constexpr,
        GROUP_SQRT: tl.constexpr,
        BLOCK_B: tl.constexpr,
        BLOCK_D: tl.constexpr,
        C_BLOCK: tl.constexpr,
        SQRT_D: tl.constexpr,
        CLASS_BRANCH: tl.constexpr,
        CLASS_GAIN: tl.constexpr,
    ):
        pid_g = tl.program_id(0)
        offs_d = pid_g * GROUP_SIZE + tl.arange(0, BLOCK_D)
        offs_c = tl.arange(0, C_BLOCK)
        mask_d = (offs_d < D) & (offs_d < ((pid_g + 1) * GROUP_SIZE))
        mask_c = offs_c < C
        acc = tl.zeros((C_BLOCK,), dtype=tl.float32)
        for b0 in range(0, B, BLOCK_B):
            offs_b = b0 + tl.arange(0, BLOCK_B)
            mask_b = offs_b < B
            xv = tl.load(x + offs_b[:, None] * D + offs_d[None, :], mask=mask_b[:, None] & mask_d[None, :], other=0.0)
            muv = tl.load(mu + offs_d, mask=mask_d, other=0.0)
            stdv = tl.load(std + offs_d, mask=mask_d, other=1.0)
            z = (xv - muv[None, :]) / tl.maximum(stdv[None, :], 1.0e-4)
            z = tl.minimum(3.0, tl.maximum(-3.0, z))
            feat = (tl.sum(tl.maximum(z, -z), axis=1) / GROUP_SIZE - tl.load(group_abs_mean + pid_g)) * GROUP_SQRT
            gd = tl.load(delta + offs_b[:, None] * C + offs_c[None, :], mask=mask_b[:, None] & mask_c[None, :], other=0.0)
            if CLASS_BRANCH:
                branch0 = tl.load(branch_scale + offs_c, mask=mask_c, other=0.0)
            else:
                branch0 = tl.load(branch_scale + 0) + tl.zeros((C_BLOCK,), dtype=tl.float32)
            if CLASS_GAIN:
                raw_gain = tl.load(logit_gain + offs_c, mask=mask_c, other=1.0)
            else:
                raw_gain = tl.load(logit_gain) + tl.zeros((C_BLOCK,), dtype=tl.float32)
            gain = tl.minimum(4.0, tl.maximum(0.25, raw_gain))
            acc += tl.sum(feat[:, None] * gd * gain[None, :] * branch0[None, :] / SQRT_D, axis=0)
        tl.store(grad_direct + (GROUP_OFFSET + pid_g) * C + offs_c, acc, mask=mask_c)

    @triton.jit
    def _fhq_single_sqdiag_direct_grad_kernel(
        x,
        mu,
        std,
        z2_mean,
        z2_mean_extra,
        norm_mean,
        mean_mean,
        group_abs_mean,
        delta,
        branch_scale,
        logit_gain,
        grad_direct,
        B: tl.constexpr,
        D: tl.constexpr,
        C: tl.constexpr,
        F: tl.constexpr,
        BLOCK_B: tl.constexpr,
        BLOCK_F: tl.constexpr,
        C_BLOCK: tl.constexpr,
        CENTER: tl.constexpr,
        SQRT_D: tl.constexpr,
        CLASS_BRANCH: tl.constexpr,
        CLASS_GAIN: tl.constexpr,
        ABS_DIAG: tl.constexpr,
        ABS_QUAD: tl.constexpr,
        ABS_MIX_SQ: tl.constexpr,
        ABS_MIX_SQ_SCALE: tl.constexpr,
        CUBIC_DIAG: tl.constexpr,
        NORM_ABS: tl.constexpr,
        NORM_OFFSET: tl.constexpr,
        MEAN_STAT: tl.constexpr,
        MEAN_OFFSET: tl.constexpr,
        GROUP_COUNT: tl.constexpr,
        GROUP_OFFSET: tl.constexpr,
        GROUP_SIZE: tl.constexpr,
        GROUP_SQRT: tl.constexpr,
    ):
        pid_f = tl.program_id(0)
        offs_f = pid_f * BLOCK_F + tl.arange(0, BLOCK_F)
        offs_c = tl.arange(0, C_BLOCK)
        mask_f = offs_f < F
        mask_c = offs_c < C
        d = offs_f % D
        kind = offs_f // D
        acc = tl.zeros((BLOCK_F, C_BLOCK), dtype=tl.float32)
        for b0 in range(0, B, BLOCK_B):
            offs_b = b0 + tl.arange(0, BLOCK_B)
            mask_b = offs_b < B
            xv = tl.load(x + offs_b[:, None] * D + d[None, :], mask=mask_b[:, None] & mask_f[None, :], other=0.0)
            muv = tl.load(mu + d, mask=mask_f, other=0.0)
            stdv = tl.load(std + d, mask=mask_f, other=1.0)
            z = (xv - muv[None, :]) / tl.maximum(stdv[None, :], 1.0e-4)
            z = tl.minimum(3.0, tl.maximum(-3.0, z))
            if ABS_QUAD:
                abs_tail = tl.maximum(z, -z) - tl.load(z2_mean + d, mask=mask_f, other=0.0)[None, :]
                sq = z * z - tl.load(z2_mean_extra + d, mask=mask_f, other=0.0)[None, :]
            elif ABS_MIX_SQ:
                abs_tail = tl.maximum(z, -z) - tl.load(z2_mean + d, mask=mask_f, other=0.0)[None, :]
                sq = abs_tail + ABS_MIX_SQ_SCALE * (z * z - tl.load(z2_mean_extra + d, mask=mask_f, other=0.0)[None, :])
            elif CUBIC_DIAG:
                sq = z * z * z - tl.load(z2_mean + d, mask=mask_f, other=0.0)[None, :]
            elif ABS_DIAG:
                sq = tl.maximum(z, -z) - tl.load(z2_mean + d, mask=mask_f, other=0.0)[None, :]
            else:
                sq = z * z - tl.load(z2_mean + d, mask=mask_f, other=0.0)[None, :]
            if ABS_QUAD:
                feat = tl.where(kind[None, :] == 0, z, tl.where(kind[None, :] == 1, tl.maximum(z - CENTER, 0.0), tl.where(kind[None, :] == 2, tl.maximum(-z - CENTER, 0.0), tl.where(kind[None, :] == 3, abs_tail, sq))))
            else:
                feat = tl.where(kind[None, :] == 0, z, tl.where(kind[None, :] == 1, tl.maximum(z - CENTER, 0.0), tl.where(kind[None, :] == 2, tl.maximum(-z - CENTER, 0.0), sq)))
            gd = tl.load(delta + offs_b[:, None] * C + offs_c[None, :], mask=mask_b[:, None] & mask_c[None, :], other=0.0)
            if CLASS_BRANCH:
                branch0 = tl.load(branch_scale + offs_c, mask=mask_c, other=0.0)
            else:
                branch0 = tl.load(branch_scale + 0) + tl.zeros((C_BLOCK,), dtype=tl.float32)
            if CLASS_GAIN:
                raw_gain = tl.load(logit_gain + offs_c, mask=mask_c, other=1.0)
            else:
                raw_gain = tl.load(logit_gain) + tl.zeros((C_BLOCK,), dtype=tl.float32)
            gain = tl.minimum(4.0, tl.maximum(0.25, raw_gain))
            acc += tl.dot(tl.trans(feat), gd * gain[None, :] * branch0[None, :] / SQRT_D)
            if NORM_ABS:
                raw_mask = mask_f & (kind == 0)
                partial_abs = tl.sum(tl.where(raw_mask[None, :], tl.maximum(z, -z), 0.0), axis=1)
                mean_term = tl.where(pid_f == 0, tl.load(norm_mean) * SQRT_D, 0.0)
                norm_part = partial_abs * SQRT_D / D - mean_term
                norm_acc = tl.sum(norm_part[:, None] * gd * gain[None, :] * branch0[None, :] / SQRT_D, axis=0)
                tl.atomic_add(grad_direct + NORM_OFFSET * C + offs_c, norm_acc, sem="relaxed", mask=mask_c)
            if MEAN_STAT:
                raw_mask = mask_f & (kind == 0)
                partial_mean = tl.sum(tl.where(raw_mask[None, :], z, 0.0), axis=1)
                mean_term = tl.where(pid_f == 0, tl.load(mean_mean) * SQRT_D, 0.0)
                mean_part = partial_mean * SQRT_D / D - mean_term
                mean_acc = tl.sum(mean_part[:, None] * gd * gain[None, :] * branch0[None, :] / SQRT_D, axis=0)
                tl.atomic_add(grad_direct + MEAN_OFFSET * C + offs_c, mean_acc, sem="relaxed", mask=mask_c)
            if GROUP_COUNT > 0:
                for group_idx in tl.static_range(0, 8):
                    if group_idx < GROUP_COUNT:
                        raw_mask = mask_f & (kind == 0) & (d >= group_idx * GROUP_SIZE) & (d < (group_idx + 1) * GROUP_SIZE)
                        partial_abs = tl.sum(tl.where(raw_mask[None, :], tl.maximum(z, -z), 0.0), axis=1)
                        mean_term = tl.where(pid_f == 0, tl.load(group_abs_mean + group_idx) * GROUP_SQRT, 0.0)
                        group_part = partial_abs * GROUP_SQRT / GROUP_SIZE - mean_term
                        group_acc = tl.sum(group_part[:, None] * gd * gain[None, :] * branch0[None, :] / SQRT_D, axis=0)
                        tl.atomic_add(grad_direct + (GROUP_OFFSET + group_idx) * C + offs_c, group_acc, sem="relaxed", mask=mask_c)
        tl.store(grad_direct + offs_f[:, None] * C + offs_c[None, :], acc, mask=mask_f[:, None] & mask_c[None, :])

    @triton.jit
    def _fhq_twohinge_direct_grad_kernel(
        x,
        mu,
        std,
        delta,
        branch_scale,
        logit_gain,
        grad_direct,
        B: tl.constexpr,
        D: tl.constexpr,
        C: tl.constexpr,
        F: tl.constexpr,
        BLOCK_B: tl.constexpr,
        BLOCK_F: tl.constexpr,
        C_BLOCK: tl.constexpr,
        SQRT_D: tl.constexpr,
        CLASS_BRANCH: tl.constexpr,
        CLASS_GAIN: tl.constexpr,
    ):
        pid_f = tl.program_id(0)
        offs_f = pid_f * BLOCK_F + tl.arange(0, BLOCK_F)
        offs_c = tl.arange(0, C_BLOCK)
        mask_f = offs_f < F
        mask_c = offs_c < C
        is_z = offs_f < D
        is_pos = (offs_f >= D) & (offs_f < 3 * D)
        pos_idx = offs_f - D
        neg_idx = offs_f - 3 * D
        d_pos = pos_idx // 2
        d_neg = neg_idx // 2
        c_pos = pos_idx - 2 * d_pos
        c_neg = neg_idx - 2 * d_neg
        d = tl.where(is_z, offs_f, tl.where(is_pos, d_pos, d_neg))
        center = tl.where(is_pos, tl.where(c_pos == 0, 0.25, 0.75), tl.where(c_neg == 0, 0.25, 0.75))
        acc = tl.zeros((BLOCK_F, C_BLOCK), dtype=tl.float32)
        for b0 in range(0, B, BLOCK_B):
            offs_b = b0 + tl.arange(0, BLOCK_B)
            mask_b = offs_b < B
            xv = tl.load(x + offs_b[:, None] * D + d[None, :], mask=mask_b[:, None] & mask_f[None, :], other=0.0)
            muv = tl.load(mu + d, mask=mask_f, other=0.0)
            stdv = tl.load(std + d, mask=mask_f, other=1.0)
            z = (xv - muv[None, :]) / tl.maximum(stdv[None, :], 1.0e-4)
            z = tl.minimum(3.0, tl.maximum(-3.0, z))
            feat = tl.where(is_z[None, :], z, tl.where(is_pos[None, :], tl.maximum(z - center[None, :], 0.0), tl.maximum(-z - center[None, :], 0.0)))
            gd = tl.load(delta + offs_b[:, None] * C + offs_c[None, :], mask=mask_b[:, None] & mask_c[None, :], other=0.0)
            if CLASS_BRANCH:
                branch0 = tl.load(branch_scale + offs_c, mask=mask_c, other=0.0)
            else:
                branch0 = tl.load(branch_scale + 0) + tl.zeros((C_BLOCK,), dtype=tl.float32)
            if CLASS_GAIN:
                raw_gain = tl.load(logit_gain + offs_c, mask=mask_c, other=1.0)
            else:
                raw_gain = tl.load(logit_gain) + tl.zeros((C_BLOCK,), dtype=tl.float32)
            gain = tl.minimum(4.0, tl.maximum(0.25, raw_gain))
            acc += tl.dot(tl.trans(feat), gd * gain[None, :] * branch0[None, :] / SQRT_D)
        tl.store(grad_direct + offs_f[:, None] * C + offs_c[None, :], acc, mask=mask_f[:, None] & mask_c[None, :])

    @triton.jit
    def _fhq_quad_grad_kernel(q_out, q2_sum, delta, branch_scale, logit_gain, grad_quad, B: tl.constexpr, H: tl.constexpr, C: tl.constexpr, FQ: tl.constexpr, BLOCK_B: tl.constexpr, BLOCK_F: tl.constexpr, C_BLOCK: tl.constexpr, Q_RMS: tl.constexpr, CLASS_BRANCH: tl.constexpr, CLASS_GAIN: tl.constexpr):
        pid_f = tl.program_id(0)
        offs_f = pid_f * BLOCK_F + tl.arange(0, BLOCK_F)
        offs_c = tl.arange(0, C_BLOCK)
        mask_f = offs_f < FQ
        mask_c = offs_c < C
        h = offs_f % H
        kind = offs_f // H
        qmean = tl.load(q2_sum + h, mask=mask_f, other=0.0) / B
        qscale = tl.sqrt(tl.maximum(qmean, 1.0e-8))
        if Q_RMS:
            qmean = 1.0
        acc = tl.zeros((BLOCK_F, C_BLOCK), dtype=tl.float32)
        for b0 in range(0, B, BLOCK_B):
            offs_b = b0 + tl.arange(0, BLOCK_B)
            mask_b = offs_b < B
            qv = tl.load(q_out + offs_b[:, None] * H + h[None, :], mask=mask_b[:, None] & mask_f[None, :], other=0.0)
            if Q_RMS:
                qv = qv / qscale[None, :]
            feat = tl.where(kind[None, :] == 0, qv, qv * qv - qmean[None, :])
            gd = tl.load(delta + offs_b[:, None] * C + offs_c[None, :], mask=mask_b[:, None] & mask_c[None, :], other=0.0)
            if CLASS_BRANCH:
                branch1 = tl.load(branch_scale + C + offs_c, mask=mask_c, other=0.0)
            else:
                branch1 = tl.load(branch_scale + 1) + tl.zeros((C_BLOCK,), dtype=tl.float32)
            if CLASS_GAIN:
                raw_gain = tl.load(logit_gain + offs_c, mask=mask_c, other=1.0)
            else:
                raw_gain = tl.load(logit_gain) + tl.zeros((C_BLOCK,), dtype=tl.float32)
            gain = tl.minimum(4.0, tl.maximum(0.25, raw_gain))
            acc += tl.dot(tl.trans(feat), gd * gain[None, :] * branch1[None, :])
        tl.store(grad_quad + offs_f[:, None] * C + offs_c[None, :], acc, mask=mask_f[:, None] & mask_c[None, :])

    @triton.jit
    def _fhq_proj_grad_kernel(x, mu, std, q_out, q2_sum, delta, quad_readout, branch_scale, logit_gain, q_std, grad_proj, B: tl.constexpr, D: tl.constexpr, H: tl.constexpr, C: tl.constexpr, BLOCK_B: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_H: tl.constexpr, C_BLOCK: tl.constexpr, Q_TANH: tl.constexpr, Q_RMS: tl.constexpr, CLASS_BRANCH: tl.constexpr, CLASS_GAIN: tl.constexpr):
        pid_d = tl.program_id(0)
        pid_h = tl.program_id(1)
        offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
        offs_c = tl.arange(0, C_BLOCK)
        mask_d = offs_d < D
        mask_h = offs_h < H
        mask_c = offs_c < C
        wq = tl.load(quad_readout + offs_h[:, None] * C + offs_c[None, :], mask=mask_h[:, None] & mask_c[None, :], other=0.0)
        wq2 = tl.load(quad_readout + (H + offs_h[:, None]) * C + offs_c[None, :], mask=mask_h[:, None] & mask_c[None, :], other=0.0)
        qscale = tl.load(q_std + offs_h, mask=mask_h, other=1.0)
        q_rms = tl.sqrt(tl.maximum(tl.load(q2_sum + offs_h, mask=mask_h, other=1.0) / B, 1.0e-8))
        acc = tl.zeros((BLOCK_D, BLOCK_H), dtype=tl.float32)
        for b0 in range(0, B, BLOCK_B):
            offs_b = b0 + tl.arange(0, BLOCK_B)
            mask_b = offs_b < B
            xv = tl.load(x + offs_b[:, None] * D + offs_d[None, :], mask=mask_b[:, None] & mask_d[None, :], other=0.0)
            z = (xv - tl.load(mu + offs_d, mask=mask_d, other=0.0)[None, :]) / tl.maximum(tl.load(std + offs_d, mask=mask_d, other=1.0)[None, :], 1.0e-4)
            z = tl.minimum(3.0, tl.maximum(-3.0, z))
            qv = tl.load(q_out + offs_b[:, None] * H + offs_h[None, :], mask=mask_b[:, None] & mask_h[None, :], other=0.0)
            if Q_RMS:
                qv = qv / q_rms[None, :]
            if CLASS_BRANCH:
                branch1 = tl.load(branch_scale + C + offs_c, mask=mask_c, other=0.0)
            else:
                branch1 = tl.load(branch_scale + 1) + tl.zeros((C_BLOCK,), dtype=tl.float32)
            if CLASS_GAIN:
                raw_gain = tl.load(logit_gain + offs_c, mask=mask_c, other=1.0)
            else:
                raw_gain = tl.load(logit_gain) + tl.zeros((C_BLOCK,), dtype=tl.float32)
            gain = tl.minimum(4.0, tl.maximum(0.25, raw_gain))
            gd = tl.load(delta + offs_b[:, None] * C + offs_c[None, :], mask=mask_b[:, None] & mask_c[None, :], other=0.0) * gain[None, :] * branch1[None, :]
            grad_q = tl.dot(gd, tl.trans(wq)) + 2.0 * qv * tl.dot(gd, tl.trans(wq2))
            if Q_TANH:
                grad_q = grad_q * (1.0 - qv * qv)
            if Q_RMS:
                grad_q = grad_q / q_rms[None, :]
            grad_q = grad_q / tl.maximum(qscale[None, :], 1.0e-4)
            acc += tl.dot(tl.trans(z), grad_q)
        tl.store(grad_proj + offs_d[:, None] * H + offs_h[None, :], acc, mask=mask_d[:, None] & mask_h[None, :])

    @triton.jit
    def _fhq_proj_grad_adamw_kernel(x, mu, std, q_out, q2_sum, delta, quad_readout, branch_scale, logit_gain, q_std, quad_proj, exp_avg, exp_avg_sq, step_size, B: tl.constexpr, D: tl.constexpr, H: tl.constexpr, C: tl.constexpr, LR: tl.constexpr, WEIGHT_DECAY: tl.constexpr, BETA1: tl.constexpr, BETA2: tl.constexpr, EPS_ADAM: tl.constexpr, BLOCK_B: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_H: tl.constexpr, C_BLOCK: tl.constexpr, Q_TANH: tl.constexpr, Q_RMS: tl.constexpr, CLASS_BRANCH: tl.constexpr, CLASS_GAIN: tl.constexpr):
        pid_d = tl.program_id(0)
        pid_h = tl.program_id(1)
        offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
        offs_c = tl.arange(0, C_BLOCK)
        mask_d = offs_d < D
        mask_h = offs_h < H
        mask_c = offs_c < C
        wq = tl.load(quad_readout + offs_h[:, None] * C + offs_c[None, :], mask=mask_h[:, None] & mask_c[None, :], other=0.0)
        wq2 = tl.load(quad_readout + (H + offs_h[:, None]) * C + offs_c[None, :], mask=mask_h[:, None] & mask_c[None, :], other=0.0)
        qscale = tl.load(q_std + offs_h, mask=mask_h, other=1.0)
        q_rms = tl.sqrt(tl.maximum(tl.load(q2_sum + offs_h, mask=mask_h, other=1.0) / B, 1.0e-8))
        acc = tl.zeros((BLOCK_D, BLOCK_H), dtype=tl.float32)
        for b0 in range(0, B, BLOCK_B):
            offs_b = b0 + tl.arange(0, BLOCK_B)
            mask_b = offs_b < B
            xv = tl.load(x + offs_b[:, None] * D + offs_d[None, :], mask=mask_b[:, None] & mask_d[None, :], other=0.0)
            z = (xv - tl.load(mu + offs_d, mask=mask_d, other=0.0)[None, :]) / tl.maximum(tl.load(std + offs_d, mask=mask_d, other=1.0)[None, :], 1.0e-4)
            z = tl.minimum(3.0, tl.maximum(-3.0, z))
            qv = tl.load(q_out + offs_b[:, None] * H + offs_h[None, :], mask=mask_b[:, None] & mask_h[None, :], other=0.0)
            if Q_RMS:
                qv = qv / q_rms[None, :]
            if CLASS_BRANCH:
                branch1 = tl.load(branch_scale + C + offs_c, mask=mask_c, other=0.0)
            else:
                branch1 = tl.load(branch_scale + 1) + tl.zeros((C_BLOCK,), dtype=tl.float32)
            if CLASS_GAIN:
                raw_gain = tl.load(logit_gain + offs_c, mask=mask_c, other=1.0)
            else:
                raw_gain = tl.load(logit_gain) + tl.zeros((C_BLOCK,), dtype=tl.float32)
            gain = tl.minimum(4.0, tl.maximum(0.25, raw_gain))
            gd = tl.load(delta + offs_b[:, None] * C + offs_c[None, :], mask=mask_b[:, None] & mask_c[None, :], other=0.0) * gain[None, :] * branch1[None, :]
            grad_q = tl.dot(gd, tl.trans(wq)) + 2.0 * qv * tl.dot(gd, tl.trans(wq2))
            if Q_TANH:
                grad_q = grad_q * (1.0 - qv * qv)
            if Q_RMS:
                grad_q = grad_q / q_rms[None, :]
            grad_q = grad_q / tl.maximum(qscale[None, :], 1.0e-4)
            acc += tl.dot(tl.trans(z), grad_q)
        idx = offs_d[:, None] * H + offs_h[None, :]
        mask = mask_d[:, None] & mask_h[None, :]
        p = tl.load(quad_proj + idx, mask=mask, other=0.0)
        m = tl.load(exp_avg + idx, mask=mask, other=0.0)
        v = tl.load(exp_avg_sq + idx, mask=mask, other=0.0)
        if WEIGHT_DECAY != 0.0:
            p = p * (1.0 - LR * WEIGHT_DECAY)
        m = m * BETA1 + acc * (1.0 - BETA1)
        v = v * BETA2 + acc * acc * (1.0 - BETA2)
        p = p - step_size * m / (tl.sqrt(v) + EPS_ADAM)
        tl.store(quad_proj + idx, p, mask=mask)
        tl.store(exp_avg + idx, m, mask=mask)
        tl.store(exp_avg_sq + idx, v, mask=mask)

    @triton.jit
    def _fhq_adamw_update_kernel(param, grad, exp_avg, exp_avg_sq, step_size, N: tl.constexpr, LR: tl.constexpr, WEIGHT_DECAY: tl.constexpr, BETA1: tl.constexpr, BETA2: tl.constexpr, EPS_ADAM: tl.constexpr, BLOCK_N: tl.constexpr):
        offs = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = offs < N
        p = tl.load(param + offs, mask=mask, other=0.0)
        g = tl.load(grad + offs, mask=mask, other=0.0)
        m = tl.load(exp_avg + offs, mask=mask, other=0.0)
        v = tl.load(exp_avg_sq + offs, mask=mask, other=0.0)
        if WEIGHT_DECAY != 0.0:
            p = p * (1.0 - LR * WEIGHT_DECAY)
        m = m * BETA1 + g * (1.0 - BETA1)
        v = v * BETA2 + g * g * (1.0 - BETA2)
        p = p - step_size * m / (tl.sqrt(v) + EPS_ADAM)
        tl.store(param + offs, p, mask=mask)
        tl.store(exp_avg + offs, m, mask=mask)
        tl.store(exp_avg_sq + offs, v, mask=mask)

    @triton.jit
    def _fhq_scalar_grad_kernel(
        logits,
        delta,
        direct_logits,
        quad_logits,
        logit_gain,
        grad_branch,
        grad_gain,
        grad_bias,
        B: tl.constexpr,
        C: tl.constexpr,
        BLOCK_B: tl.constexpr,
        C_BLOCK: tl.constexpr,
        CLASS_BRANCH: tl.constexpr,
        CLASS_GAIN: tl.constexpr,
    ):
        offs_b = tl.arange(0, BLOCK_B)
        offs_c = tl.arange(0, C_BLOCK)
        mask = (offs_b[:, None] < B) & (offs_c[None, :] < C)
        idx = offs_b[:, None] * C + offs_c[None, :]
        d = tl.load(delta + idx, mask=mask, other=0.0)
        dl = tl.load(direct_logits + idx, mask=mask, other=0.0)
        ql = tl.load(quad_logits + idx, mask=mask, other=0.0)
        lo = tl.load(logits + idx, mask=mask, other=0.0)
        if CLASS_GAIN:
            raw_gain = tl.load(logit_gain + offs_c, mask=offs_c < C, other=1.0)
        else:
            raw_gain = tl.load(logit_gain) + tl.zeros((C_BLOCK,), dtype=tl.float32)
        gain = tl.minimum(4.0, tl.maximum(0.25, raw_gain))
        g_logits = d * gain[None, :]
        branch0_by_c = tl.sum(g_logits * dl, axis=0)
        branch1_by_c = tl.sum(g_logits * ql, axis=0)
        gain_by_c = tl.sum(d * (lo / gain[None, :]), axis=0)
        bias_by_c = tl.sum(g_logits, axis=0)
        if CLASS_BRANCH:
            tl.store(grad_branch + offs_c, branch0_by_c, mask=offs_c < C)
            tl.store(grad_branch + C + offs_c, branch1_by_c, mask=offs_c < C)
        else:
            tl.store(grad_branch + 0, tl.sum(branch0_by_c, axis=0))
            tl.store(grad_branch + 1, tl.sum(branch1_by_c, axis=0))
        gain_mask = (raw_gain >= 0.25) & (raw_gain <= 4.0)
        if CLASS_GAIN:
            tl.store(grad_gain + offs_c, gain_by_c * gain_mask, mask=offs_c < C)
        else:
            raw_gain_scalar = tl.load(logit_gain)
            gain_mask_scalar = (raw_gain_scalar >= 0.25) & (raw_gain_scalar <= 4.0)
            tl.store(grad_gain, tl.sum(gain_by_c, axis=0) * gain_mask_scalar)
        tl.store(grad_bias + offs_c, bias_by_c, mask=offs_c < C)


def center_count(model: torch.nn.Module) -> int:
    centers = getattr(model, "hinge_centers", torch.empty(0))
    return int(centers.numel())


def single_center(model: torch.nn.Module) -> float:
    cached = getattr(model, "hinge_center0", None)
    if cached is not None:
        return float(cached)
    centers = getattr(model, "hinge_centers", torch.empty(0))
    if int(centers.numel()) == 0:
        return 0.25
    return float(centers.flatten()[0].detach().cpu().item())


def class_gain_enabled(model: torch.nn.Module) -> bool:
    gain = getattr(model, "logit_gain", None)
    return bool(getattr(model, "class_logit_gain_enabled", False)) or (
        gain is not None and int(gain.numel()) == int(getattr(model, "output_dim", -1))
    )


def _base_logits(model: torch.nn.Module, direct_logits: torch.Tensor, quad_logits: torch.Tensor) -> torch.Tensor:
    if bool(getattr(model, "class_branch_scale_enabled", False)):
        return model.branch_scale[0].view(1, -1) * direct_logits + model.branch_scale[1].view(1, -1) * quad_logits + model.bias  # type: ignore[attr-defined]
    return model.branch_scale[0] * direct_logits + model.branch_scale[1] * quad_logits + model.bias  # type: ignore[attr-defined]


def _gain(model: torch.nn.Module) -> torch.Tensor:
    return model.logit_gain.clamp(0.25, 4.0)  # type: ignore[attr-defined]


def _apply_logit_norm_if_needed(model: torch.nn.Module, logits: torch.Tensor, direct_logits: torch.Tensor, quad_logits: torch.Tensor) -> torch.Tensor:
    if not bool(getattr(model, "logit_norm_enabled", False)):
        return logits
    with torch.no_grad():
        base = _base_logits(model, direct_logits, quad_logits)
        rms = base.square().mean(dim=1, keepdim=True).sqrt().clamp_min(1.0e-4)
        pre_gain = model.logit_norm_target * base / rms  # type: ignore[attr-defined]
        logits.copy_(_gain(model).view(1, -1) * pre_gain)
    return logits


def _effective_delta_for_kernels(model: torch.nn.Module, logits: torch.Tensor, y: torch.Tensor, direct_logits: torch.Tensor, quad_logits: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    delta_final = delta_into(logits, y, out) if out is not None else delta(logits, y)
    if not bool(getattr(model, "logit_norm_enabled", False)):
        return delta_final
    with torch.no_grad():
        base = _base_logits(model, direct_logits, quad_logits)
        rms = base.square().mean(dim=1, keepdim=True).sqrt().clamp_min(1.0e-4)
        gain = _gain(model).view(1, -1)
        g_pre = delta_final * gain
        dot = (g_pre * base).sum(dim=1, keepdim=True)
        denom = float(max(1, int(base.shape[1]))) * rms.square()
        delta_base = model.logit_norm_target * (g_pre - base * dot / denom) / rms  # type: ignore[attr-defined]
        delta_final.copy_(delta_base / gain.clamp_min(1.0e-6))
    return delta_final


def _effective_delta_from_grad_logits_for_kernels(
    model: torch.nn.Module,
    grad_logits: torch.Tensor,
    direct_logits: torch.Tensor,
    quad_logits: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    delta_final = out if out is not None else torch.empty_like(grad_logits)
    delta_final.copy_(grad_logits)
    if not bool(getattr(model, "logit_norm_enabled", False)):
        return delta_final
    with torch.no_grad():
        base = _base_logits(model, direct_logits, quad_logits)
        rms = base.square().mean(dim=1, keepdim=True).sqrt().clamp_min(1.0e-4)
        gain = _gain(model).view(1, -1)
        g_pre = delta_final * gain
        dot = (g_pre * base).sum(dim=1, keepdim=True)
        denom = float(max(1, int(base.shape[1]))) * rms.square()
        delta_base = model.logit_norm_target * (g_pre - base * dot / denom) / rms  # type: ignore[attr-defined]
        delta_final.copy_(delta_base / gain.clamp_min(1.0e-6))
    return delta_final


def supported_simple(model: torch.nn.Module, input_dim: int, output_dim: int) -> Tuple[bool, str]:
    if not TRITON_AVAILABLE:
        return False, "triton_not_available"
    if not hasattr(model, "quad_proj") or not hasattr(model, "direct_readout"):
        return False, "not_simple_fast_task_geometry"
    if int(getattr(model, "input_dim", input_dim)) != int(input_dim) or int(getattr(model, "output_dim", output_dim)) != int(output_dim):
        return False, "shape_mismatch"
    centers = getattr(model, "hinge_centers", torch.empty(0))
    vals = [float(v) for v in centers.flatten().detach().cpu().tolist()]
    single_ok = len(vals) == 1 and (abs(vals[0] - 0.25) < 1.0e-6 or abs(vals[0] - 0.50) < 1.0e-6)
    two_ok = len(vals) == 2 and abs(vals[0] - 0.25) < 1.0e-6 and abs(vals[1] - 0.75) < 1.0e-6
    if single_ok or two_ok:
        pass
    else:
        return False, "only_single_0p25_0p50_or_twohinge_0p25_0p75_supported"
    has_direct_tail = (
        bool(getattr(model, "direct_square_enabled", False))
        or bool(getattr(model, "direct_abs_enabled", False))
        or bool(getattr(model, "direct_abs_square_enabled", False))
        or bool(getattr(model, "direct_absmix_square_enabled", False))
        or bool(getattr(model, "direct_cubic_enabled", False))
    )
    has_normabs = bool(getattr(model, "direct_normabs_enabled", False))
    has_meanstat = bool(getattr(model, "direct_meanstat_enabled", False))
    group_count = int(getattr(model, "direct_groupabs_count", 0))
    if has_direct_tail and not (len(vals) == 1 and abs(vals[0] - 0.25) < 1.0e-6):
        return False, "direct_tail_supported_only_for_single_hinge"
    if has_normabs and not (len(vals) == 1 and abs(vals[0] - 0.25) < 1.0e-6):
        return False, "normabs_supported_only_for_single_hinge"
    if has_meanstat and not (len(vals) == 1 and abs(vals[0] - 0.25) < 1.0e-6):
        return False, "meanstat_supported_only_for_single_hinge"
    if group_count and not (len(vals) == 1 and abs(vals[0] - 0.25) < 1.0e-6):
        return False, "groupabs_supported_only_for_single_hinge"
    if group_count and int(input_dim) % group_count != 0:
        return False, "groupabs_requires_even_input_groups"
    if group_count not in (0, 4, 8):
        return False, "groupabs_count_must_be_4_or_8"
    if bool(getattr(model, "logit_norm_enabled", False)):
        if isinstance(getattr(model, "branch_scale", None), torch.nn.Parameter) and model.branch_scale.requires_grad:  # type: ignore[attr-defined]
            return False, "logitnorm_fhq_requires_fixed_branch_scale"
        if isinstance(getattr(model, "logit_gain", None), torch.nn.Parameter) and model.logit_gain.requires_grad:  # type: ignore[attr-defined]
            return False, "logitnorm_fhq_requires_fixed_logit_gain"
    if int(input_dim) > 2048 or int(getattr(model, "hidden_dim", 0)) > 512 or int(output_dim) > 16:
        return False, "shape_exceeds_smoke_kernel_limit"
    return True, "supported"


class FHQWorkspace:
    """Reusable CUDA buffers for one SimpleFastTaskGeometry FHQ step."""

    def __init__(self, model: torch.nn.Module, batch_size: int, device: torch.device | None = None, include_grad_proj: bool = True) -> None:
        self.batch_size = int(batch_size)
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        dev = device if device is not None else next(model.parameters()).device
        hidden = int(model.hidden_dim)  # type: ignore[attr-defined]
        classes = int(model.output_dim)  # type: ignore[attr-defined]
        self.q_out = torch.empty((self.batch_size, hidden), device=dev, dtype=torch.float32)
        self.q2_sum = torch.empty((hidden,), device=dev, dtype=torch.float32)
        self.logits = torch.empty((self.batch_size, classes), device=dev, dtype=torch.float32)
        self.direct_logits = torch.empty((self.batch_size, classes), device=dev, dtype=torch.float32)
        self.quad_logits = torch.empty((self.batch_size, classes), device=dev, dtype=torch.float32)
        self.delta = torch.empty((self.batch_size, classes), device=dev, dtype=torch.float32)
        self.grad_direct = torch.empty_like(model.direct_readout, device=dev)  # type: ignore[attr-defined]
        self.grad_quad = torch.empty_like(model.quad_readout, device=dev)  # type: ignore[attr-defined]
        if include_grad_proj and isinstance(model.quad_proj, torch.nn.Parameter):  # type: ignore[attr-defined]
            self.grad_proj = torch.empty_like(model.quad_proj, device=dev)  # type: ignore[attr-defined]
        else:
            self.grad_proj = None
        self.grad_branch = torch.empty_like(model.branch_scale, device=dev)  # type: ignore[attr-defined]
        self.grad_gain = torch.empty_like(model.logit_gain, device=dev)  # type: ignore[attr-defined]
        self.grad_bias = torch.empty_like(model.bias, device=dev)  # type: ignore[attr-defined]


def make_workspace(model: torch.nn.Module, batch_size: int, device: torch.device | None = None, include_grad_proj: bool = True) -> FHQWorkspace:
    return FHQWorkspace(model, batch_size, device=device, include_grad_proj=include_grad_proj)


def forward(model: torch.nn.Module, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    assert TRITON_AVAILABLE and triton is not None
    bsz = int(x.shape[0])
    dim = int(model.input_dim)  # type: ignore[attr-defined]
    hidden = int(model.hidden_dim)  # type: ignore[attr-defined]
    classes = int(model.output_dim)  # type: ignore[attr-defined]
    block_d = triton.next_power_of_2(dim)
    block_h = 32 if hidden <= 256 else 16
    c_block = triton.next_power_of_2(classes)
    class_branch = bool(getattr(model, "class_branch_scale_enabled", False))
    class_gain = class_gain_enabled(model)
    norm_abs = bool(getattr(model, "direct_normabs_enabled", False))
    mean_stat = bool(getattr(model, "direct_meanstat_enabled", False))
    group_count = int(getattr(model, "direct_groupabs_count", 0))
    group_size = max(1, dim // max(1, group_count))
    group_sqrt = math.sqrt(float(group_size))
    q_out = torch.empty((bsz, hidden), device=x.device, dtype=torch.float32)
    q2_sum = torch.zeros((hidden,), device=x.device, dtype=torch.float32)
    logits = torch.empty((bsz, classes), device=x.device, dtype=torch.float32)
    direct_logits = torch.empty((bsz, classes), device=x.device, dtype=torch.float32)
    quad_logits = torch.empty((bsz, classes), device=x.device, dtype=torch.float32)
    _fhq_q_kernel[(bsz, triton.cdiv(hidden, block_h))](x, model.mu.contiguous(), model.std.contiguous(), model.quad_proj.contiguous(), model.quad_feature_std.contiguous(), q_out, q2_sum, D=dim, H=hidden, BLOCK_D=block_d, BLOCK_H=block_h, Q_TANH=bool(getattr(model, "quad_tanh_enabled", False)))  # type: ignore[attr-defined]
    center = single_center(model)
    if bool(getattr(model, "direct_square_enabled", False)) or bool(getattr(model, "direct_abs_enabled", False)) or bool(getattr(model, "direct_abs_square_enabled", False)) or bool(getattr(model, "direct_absmix_square_enabled", False)) or bool(getattr(model, "direct_cubic_enabled", False)):
        abs_quad = bool(getattr(model, "direct_abs_square_enabled", False))
        abs_mix_sq = bool(getattr(model, "direct_absmix_square_enabled", False))
        cubic_diag = bool(getattr(model, "direct_cubic_enabled", False))
        direct_tail_mean = model.z3_mean if cubic_diag else (model.zabs_mean if (bool(getattr(model, "direct_abs_enabled", False)) or abs_quad or abs_mix_sq) else model.z2_mean)  # type: ignore[attr-defined]
        tail_feature_count = (5 if abs_quad else 4) * dim
        _fhq_single_sqdiag_logits_kernel[(bsz,)](x, model.mu.contiguous(), model.std.contiguous(), direct_tail_mean.contiguous(), model.z2_mean.contiguous(), model.zabs_scalar_mean.contiguous(), model.zmean_scalar_mean.contiguous(), model.zgroupabs_mean.contiguous(), model.direct_readout.contiguous(), q_out, q2_sum, model.quad_readout.contiguous(), model.branch_scale.contiguous(), model.logit_gain.contiguous(), model.bias.contiguous(), logits, direct_logits, quad_logits, B=bsz, D=dim, H=hidden, C=classes, BLOCK_D=block_d, BLOCK_H=triton.next_power_of_2(hidden), C_BLOCK=c_block, CENTER=center, SQRT_D=math.sqrt(max(1, dim)), Q_RMS=bool(getattr(model, "quad_batch_rms_enabled", False)), CLASS_BRANCH=class_branch, CLASS_GAIN=class_gain, ABS_DIAG=bool(getattr(model, "direct_abs_enabled", False)), ABS_QUAD=abs_quad, ABS_MIX_SQ=abs_mix_sq, ABS_MIX_SQ_SCALE=float(getattr(model, "direct_absmix_square_scale", 0.25)), CUBIC_DIAG=cubic_diag, NORM_ABS=norm_abs, MEAN_STAT=mean_stat, MEAN_OFFSET=tail_feature_count + int(norm_abs), GROUP_COUNT=group_count, GROUP_OFFSET=tail_feature_count + int(norm_abs) + int(mean_stat), GROUP_SIZE=group_size, GROUP_SQRT=group_sqrt)  # type: ignore[attr-defined]
    elif center_count(model) == 2:
        _fhq_twohinge_logits_kernel[(bsz,)](x, model.mu.contiguous(), model.std.contiguous(), model.direct_readout.contiguous(), q_out, q2_sum, model.quad_readout.contiguous(), model.branch_scale.contiguous(), model.logit_gain.contiguous(), model.bias.contiguous(), logits, direct_logits, quad_logits, B=bsz, D=dim, H=hidden, C=classes, BLOCK_D=block_d, BLOCK_H=triton.next_power_of_2(hidden), C_BLOCK=c_block, SQRT_D=math.sqrt(max(1, dim)), Q_RMS=bool(getattr(model, "quad_batch_rms_enabled", False)), CLASS_BRANCH=class_branch, CLASS_GAIN=class_gain)  # type: ignore[attr-defined]
    else:
        _fhq_single_logits_kernel[(bsz,)](x, model.mu.contiguous(), model.std.contiguous(), model.zabs_scalar_mean.contiguous(), model.zmean_scalar_mean.contiguous(), model.zgroupabs_mean.contiguous(), model.direct_readout.contiguous(), q_out, q2_sum, model.quad_readout.contiguous(), model.branch_scale.contiguous(), model.logit_gain.contiguous(), model.bias.contiguous(), logits, direct_logits, quad_logits, B=bsz, D=dim, H=hidden, C=classes, BLOCK_D=block_d, BLOCK_H=triton.next_power_of_2(hidden), C_BLOCK=c_block, CENTER=center, SQRT_D=math.sqrt(max(1, dim)), Q_RMS=bool(getattr(model, "quad_batch_rms_enabled", False)), CLASS_BRANCH=class_branch, CLASS_GAIN=class_gain, NORM_ABS=norm_abs, MEAN_STAT=mean_stat, MEAN_OFFSET=3 * dim + int(norm_abs), GROUP_COUNT=group_count, GROUP_OFFSET=3 * dim + int(norm_abs) + int(mean_stat), GROUP_SIZE=group_size, GROUP_SQRT=group_sqrt)  # type: ignore[attr-defined]
    _apply_logit_norm_if_needed(model, logits, direct_logits, quad_logits)
    return logits, q_out, q2_sum, direct_logits, quad_logits


def forward_workspace(model: torch.nn.Module, x: torch.Tensor, workspace: FHQWorkspace) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    assert TRITON_AVAILABLE and triton is not None
    bsz = int(x.shape[0])
    if bsz > workspace.batch_size:
        raise ValueError(f"workspace batch_size {workspace.batch_size} is smaller than input batch {bsz}")
    dim = int(model.input_dim)  # type: ignore[attr-defined]
    hidden = int(model.hidden_dim)  # type: ignore[attr-defined]
    classes = int(model.output_dim)  # type: ignore[attr-defined]
    block_d = triton.next_power_of_2(dim)
    block_h = 32 if hidden <= 256 else 16
    c_block = triton.next_power_of_2(classes)
    class_branch = bool(getattr(model, "class_branch_scale_enabled", False))
    class_gain = class_gain_enabled(model)
    norm_abs = bool(getattr(model, "direct_normabs_enabled", False))
    mean_stat = bool(getattr(model, "direct_meanstat_enabled", False))
    group_count = int(getattr(model, "direct_groupabs_count", 0))
    group_size = max(1, dim // max(1, group_count))
    group_sqrt = math.sqrt(float(group_size))
    q_out = workspace.q_out[:bsz]
    logits = workspace.logits[:bsz]
    direct_logits = workspace.direct_logits[:bsz]
    quad_logits = workspace.quad_logits[:bsz]
    q2_sum = workspace.q2_sum
    q2_sum.zero_()
    _fhq_q_kernel[(bsz, triton.cdiv(hidden, block_h))](x, model.mu.contiguous(), model.std.contiguous(), model.quad_proj.contiguous(), model.quad_feature_std.contiguous(), q_out, q2_sum, D=dim, H=hidden, BLOCK_D=block_d, BLOCK_H=block_h, Q_TANH=bool(getattr(model, "quad_tanh_enabled", False)))  # type: ignore[attr-defined]
    center = single_center(model)
    if bool(getattr(model, "direct_square_enabled", False)) or bool(getattr(model, "direct_abs_enabled", False)) or bool(getattr(model, "direct_abs_square_enabled", False)) or bool(getattr(model, "direct_absmix_square_enabled", False)) or bool(getattr(model, "direct_cubic_enabled", False)):
        abs_quad = bool(getattr(model, "direct_abs_square_enabled", False))
        abs_mix_sq = bool(getattr(model, "direct_absmix_square_enabled", False))
        cubic_diag = bool(getattr(model, "direct_cubic_enabled", False))
        direct_tail_mean = model.z3_mean if cubic_diag else (model.zabs_mean if (bool(getattr(model, "direct_abs_enabled", False)) or abs_quad or abs_mix_sq) else model.z2_mean)  # type: ignore[attr-defined]
        tail_feature_count = (5 if abs_quad else 4) * dim
        _fhq_single_sqdiag_logits_kernel[(bsz,)](x, model.mu.contiguous(), model.std.contiguous(), direct_tail_mean.contiguous(), model.z2_mean.contiguous(), model.zabs_scalar_mean.contiguous(), model.zmean_scalar_mean.contiguous(), model.zgroupabs_mean.contiguous(), model.direct_readout.contiguous(), q_out, q2_sum, model.quad_readout.contiguous(), model.branch_scale.contiguous(), model.logit_gain.contiguous(), model.bias.contiguous(), logits, direct_logits, quad_logits, B=bsz, D=dim, H=hidden, C=classes, BLOCK_D=block_d, BLOCK_H=triton.next_power_of_2(hidden), C_BLOCK=c_block, CENTER=center, SQRT_D=math.sqrt(max(1, dim)), Q_RMS=bool(getattr(model, "quad_batch_rms_enabled", False)), CLASS_BRANCH=class_branch, CLASS_GAIN=class_gain, ABS_DIAG=bool(getattr(model, "direct_abs_enabled", False)), ABS_QUAD=abs_quad, ABS_MIX_SQ=abs_mix_sq, ABS_MIX_SQ_SCALE=float(getattr(model, "direct_absmix_square_scale", 0.25)), CUBIC_DIAG=cubic_diag, NORM_ABS=norm_abs, MEAN_STAT=mean_stat, MEAN_OFFSET=tail_feature_count + int(norm_abs), GROUP_COUNT=group_count, GROUP_OFFSET=tail_feature_count + int(norm_abs) + int(mean_stat), GROUP_SIZE=group_size, GROUP_SQRT=group_sqrt)  # type: ignore[attr-defined]
    elif center_count(model) == 2:
        _fhq_twohinge_logits_kernel[(bsz,)](x, model.mu.contiguous(), model.std.contiguous(), model.direct_readout.contiguous(), q_out, q2_sum, model.quad_readout.contiguous(), model.branch_scale.contiguous(), model.logit_gain.contiguous(), model.bias.contiguous(), logits, direct_logits, quad_logits, B=bsz, D=dim, H=hidden, C=classes, BLOCK_D=block_d, BLOCK_H=triton.next_power_of_2(hidden), C_BLOCK=c_block, SQRT_D=math.sqrt(max(1, dim)), Q_RMS=bool(getattr(model, "quad_batch_rms_enabled", False)), CLASS_BRANCH=class_branch, CLASS_GAIN=class_gain)  # type: ignore[attr-defined]
    else:
        _fhq_single_logits_kernel[(bsz,)](x, model.mu.contiguous(), model.std.contiguous(), model.zabs_scalar_mean.contiguous(), model.zmean_scalar_mean.contiguous(), model.zgroupabs_mean.contiguous(), model.direct_readout.contiguous(), q_out, q2_sum, model.quad_readout.contiguous(), model.branch_scale.contiguous(), model.logit_gain.contiguous(), model.bias.contiguous(), logits, direct_logits, quad_logits, B=bsz, D=dim, H=hidden, C=classes, BLOCK_D=block_d, BLOCK_H=triton.next_power_of_2(hidden), C_BLOCK=c_block, CENTER=center, SQRT_D=math.sqrt(max(1, dim)), Q_RMS=bool(getattr(model, "quad_batch_rms_enabled", False)), CLASS_BRANCH=class_branch, CLASS_GAIN=class_gain, NORM_ABS=norm_abs, MEAN_STAT=mean_stat, MEAN_OFFSET=3 * dim + int(norm_abs), GROUP_COUNT=group_count, GROUP_OFFSET=3 * dim + int(norm_abs) + int(mean_stat), GROUP_SIZE=group_size, GROUP_SQRT=group_sqrt)  # type: ignore[attr-defined]
    _apply_logit_norm_if_needed(model, logits, direct_logits, quad_logits)
    return logits, q_out, q2_sum, direct_logits, quad_logits


def delta(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert TRITON_AVAILABLE and triton is not None
    bsz = int(logits.shape[0])
    classes = int(logits.shape[1])
    out = torch.empty_like(logits)
    _fhq_delta_kernel[(bsz,)](logits, y, out, B=bsz, C=classes, C_BLOCK=triton.next_power_of_2(classes))
    return out


def delta_into(logits: torch.Tensor, y: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    assert TRITON_AVAILABLE and triton is not None
    bsz = int(logits.shape[0])
    classes = int(logits.shape[1])
    _fhq_delta_kernel[(bsz,)](logits, y, out, B=bsz, C=classes, C_BLOCK=triton.next_power_of_2(classes))
    return out


def _direct_grad(model: torch.nn.Module, x: torch.Tensor, delta_tensor: torch.Tensor, grad_direct: torch.Tensor) -> None:
    assert TRITON_AVAILABLE and triton is not None
    dim = int(model.input_dim)  # type: ignore[attr-defined]
    classes = int(model.output_dim)  # type: ignore[attr-defined]
    bsz = int(x.shape[0])
    block_f = 64
    c_block = triton.next_power_of_2(classes)
    class_branch = bool(getattr(model, "class_branch_scale_enabled", False))
    class_gain = class_gain_enabled(model)
    norm_abs = bool(getattr(model, "direct_normabs_enabled", False))
    mean_stat = bool(getattr(model, "direct_meanstat_enabled", False))
    group_count = int(getattr(model, "direct_groupabs_count", 0))
    group_size = max(1, dim // max(1, group_count))
    group_sqrt = math.sqrt(float(group_size))
    if bool(getattr(model, "direct_square_enabled", False)) or bool(getattr(model, "direct_abs_enabled", False)) or bool(getattr(model, "direct_abs_square_enabled", False)) or bool(getattr(model, "direct_absmix_square_enabled", False)) or bool(getattr(model, "direct_cubic_enabled", False)):
        abs_quad = bool(getattr(model, "direct_abs_square_enabled", False))
        abs_mix_sq = bool(getattr(model, "direct_absmix_square_enabled", False))
        cubic_diag = bool(getattr(model, "direct_cubic_enabled", False))
        feat_count = (5 if abs_quad else 4) * dim
        direct_tail_mean = model.z3_mean if cubic_diag else (model.zabs_mean if (bool(getattr(model, "direct_abs_enabled", False)) or abs_quad or abs_mix_sq) else model.z2_mean)  # type: ignore[attr-defined]
        if norm_abs:
            _fhq_zero_direct_row_kernel[(1,)](grad_direct, C=classes, C_BLOCK=c_block, ROW=feat_count)
        mean_offset = feat_count + int(norm_abs)
        if mean_stat:
            _fhq_zero_direct_row_kernel[(1,)](grad_direct, C=classes, C_BLOCK=c_block, ROW=mean_offset)
        group_offset = feat_count + int(norm_abs) + int(mean_stat)
        _fhq_single_sqdiag_direct_grad_kernel[(triton.cdiv(feat_count, block_f),)](x, model.mu.contiguous(), model.std.contiguous(), direct_tail_mean.contiguous(), model.z2_mean.contiguous(), model.zabs_scalar_mean.contiguous(), model.zmean_scalar_mean.contiguous(), model.zgroupabs_mean.contiguous(), delta_tensor, model.branch_scale.contiguous(), model.logit_gain.contiguous(), grad_direct, B=bsz, D=dim, C=classes, F=feat_count, BLOCK_B=128, BLOCK_F=block_f, C_BLOCK=c_block, CENTER=single_center(model), SQRT_D=math.sqrt(max(1, dim)), CLASS_BRANCH=class_branch, CLASS_GAIN=class_gain, ABS_DIAG=bool(getattr(model, "direct_abs_enabled", False)), ABS_QUAD=abs_quad, ABS_MIX_SQ=abs_mix_sq, ABS_MIX_SQ_SCALE=float(getattr(model, "direct_absmix_square_scale", 0.25)), CUBIC_DIAG=cubic_diag, NORM_ABS=norm_abs, NORM_OFFSET=feat_count, MEAN_STAT=mean_stat, MEAN_OFFSET=mean_offset, GROUP_COUNT=0, GROUP_OFFSET=group_offset, GROUP_SIZE=group_size, GROUP_SQRT=group_sqrt)  # type: ignore[attr-defined]
        if group_count:
            _fhq_groupabs_direct_grad_kernel[(group_count,)](x, model.mu.contiguous(), model.std.contiguous(), model.zgroupabs_mean.contiguous(), delta_tensor, model.branch_scale.contiguous(), model.logit_gain.contiguous(), grad_direct, B=bsz, D=dim, C=classes, GROUP_OFFSET=group_offset, GROUP_SIZE=group_size, GROUP_SQRT=group_sqrt, BLOCK_B=128, BLOCK_D=triton.next_power_of_2(group_size), C_BLOCK=c_block, SQRT_D=math.sqrt(max(1, dim)), CLASS_BRANCH=class_branch, CLASS_GAIN=class_gain)  # type: ignore[attr-defined]
        return
    if center_count(model) == 2:
        feat_count = 5 * dim
        _fhq_twohinge_direct_grad_kernel[(triton.cdiv(feat_count, block_f),)](x, model.mu.contiguous(), model.std.contiguous(), delta_tensor, model.branch_scale.contiguous(), model.logit_gain.contiguous(), grad_direct, B=bsz, D=dim, C=classes, F=feat_count, BLOCK_B=128, BLOCK_F=block_f, C_BLOCK=c_block, SQRT_D=math.sqrt(max(1, dim)), CLASS_BRANCH=class_branch, CLASS_GAIN=class_gain)  # type: ignore[attr-defined]
    else:
        feat_count = 3 * dim
        if norm_abs:
            _fhq_zero_direct_row_kernel[(1,)](grad_direct, C=classes, C_BLOCK=c_block, ROW=feat_count)
        mean_offset = feat_count + int(norm_abs)
        if mean_stat:
            _fhq_zero_direct_row_kernel[(1,)](grad_direct, C=classes, C_BLOCK=c_block, ROW=mean_offset)
        group_offset = feat_count + int(norm_abs) + int(mean_stat)
        _fhq_single_direct_grad_kernel[(triton.cdiv(feat_count, block_f),)](x, model.mu.contiguous(), model.std.contiguous(), model.zabs_scalar_mean.contiguous(), model.zmean_scalar_mean.contiguous(), model.zgroupabs_mean.contiguous(), delta_tensor, model.branch_scale.contiguous(), model.logit_gain.contiguous(), grad_direct, B=bsz, D=dim, C=classes, F=feat_count, BLOCK_B=128, BLOCK_F=block_f, C_BLOCK=c_block, CENTER=single_center(model), SQRT_D=math.sqrt(max(1, dim)), CLASS_BRANCH=class_branch, CLASS_GAIN=class_gain, NORM_ABS=norm_abs, NORM_OFFSET=feat_count, MEAN_STAT=mean_stat, MEAN_OFFSET=mean_offset, GROUP_COUNT=0, GROUP_OFFSET=group_offset, GROUP_SIZE=group_size, GROUP_SQRT=group_sqrt)  # type: ignore[attr-defined]
        if group_count:
            _fhq_groupabs_direct_grad_kernel[(group_count,)](x, model.mu.contiguous(), model.std.contiguous(), model.zgroupabs_mean.contiguous(), delta_tensor, model.branch_scale.contiguous(), model.logit_gain.contiguous(), grad_direct, B=bsz, D=dim, C=classes, GROUP_OFFSET=group_offset, GROUP_SIZE=group_size, GROUP_SQRT=group_sqrt, BLOCK_B=128, BLOCK_D=triton.next_power_of_2(group_size), C_BLOCK=c_block, SQRT_D=math.sqrt(max(1, dim)), CLASS_BRANCH=class_branch, CLASS_GAIN=class_gain)  # type: ignore[attr-defined]


def _quad_grad(model: torch.nn.Module, q_out: torch.Tensor, q2_sum: torch.Tensor, delta_tensor: torch.Tensor, grad_quad: torch.Tensor) -> None:
    assert TRITON_AVAILABLE and triton is not None
    hidden = int(model.hidden_dim)  # type: ignore[attr-defined]
    classes = int(model.output_dim)  # type: ignore[attr-defined]
    block_f = 64
    _fhq_quad_grad_kernel[(triton.cdiv(2 * hidden, block_f),)](q_out, q2_sum, delta_tensor, model.branch_scale.contiguous(), model.logit_gain.contiguous(), grad_quad, B=int(q_out.shape[0]), H=hidden, C=classes, FQ=2 * hidden, BLOCK_B=128, BLOCK_F=block_f, C_BLOCK=triton.next_power_of_2(classes), Q_RMS=bool(getattr(model, "quad_batch_rms_enabled", False)), CLASS_BRANCH=bool(getattr(model, "class_branch_scale_enabled", False)), CLASS_GAIN=class_gain_enabled(model))  # type: ignore[attr-defined]


def _scalar_grads_into(
    model: torch.nn.Module,
    logits: torch.Tensor,
    delta_tensor: torch.Tensor,
    direct_logits: torch.Tensor,
    quad_logits: torch.Tensor,
    grad_branch: torch.Tensor,
    grad_gain: torch.Tensor,
    grad_bias: torch.Tensor,
) -> None:
    assert TRITON_AVAILABLE and triton is not None
    bsz = int(logits.shape[0])
    classes = int(logits.shape[1])
    _fhq_scalar_grad_kernel[(1,)](logits, delta_tensor, direct_logits, quad_logits, model.logit_gain.contiguous(), grad_branch, grad_gain, grad_bias, B=bsz, C=classes, BLOCK_B=triton.next_power_of_2(bsz), C_BLOCK=triton.next_power_of_2(classes), CLASS_BRANCH=bool(getattr(model, "class_branch_scale_enabled", False)), CLASS_GAIN=class_gain_enabled(model))  # type: ignore[attr-defined]
    if isinstance(model.branch_scale, torch.nn.Parameter) and model.branch_scale.requires_grad:  # type: ignore[attr-defined]
        model.branch_scale.grad = grad_branch  # type: ignore[attr-defined]
    if isinstance(model.logit_gain, torch.nn.Parameter) and model.logit_gain.requires_grad:  # type: ignore[attr-defined]
        model.logit_gain.grad = grad_gain  # type: ignore[attr-defined]
    model.bias.grad = grad_bias  # type: ignore[attr-defined]


def _scalar_grads(model: torch.nn.Module, logits: torch.Tensor, delta_tensor: torch.Tensor, direct_logits: torch.Tensor, quad_logits: torch.Tensor) -> None:
    if TRITON_AVAILABLE and triton is not None and logits.is_cuda:
        grad_branch = torch.empty_like(model.branch_scale)  # type: ignore[attr-defined]
        grad_gain = torch.empty_like(model.logit_gain)  # type: ignore[attr-defined]
        grad_bias = torch.empty_like(model.bias)  # type: ignore[attr-defined]
        _scalar_grads_into(model, logits, delta_tensor, direct_logits, quad_logits, grad_branch, grad_gain, grad_bias)
        return
    with torch.no_grad():
        gain = model.logit_gain.clamp(0.25, 4.0)  # type: ignore[attr-defined]
        logits_unscaled = logits / gain
        g_logits = delta_tensor * gain
        if isinstance(model.branch_scale, torch.nn.Parameter) and model.branch_scale.requires_grad:  # type: ignore[attr-defined]
            if bool(getattr(model, "class_branch_scale_enabled", False)):
                model.branch_scale.grad = torch.stack([(g_logits * direct_logits).sum(dim=0), (g_logits * quad_logits).sum(dim=0)]).view_as(model.branch_scale)  # type: ignore[attr-defined]
            else:
                model.branch_scale.grad = torch.stack([(g_logits * direct_logits).sum(), (g_logits * quad_logits).sum()]).view_as(model.branch_scale)  # type: ignore[attr-defined]
        gain_mask = ((model.logit_gain >= 0.25) & (model.logit_gain <= 4.0)).to(dtype=delta_tensor.dtype)  # type: ignore[attr-defined]
        if isinstance(model.logit_gain, torch.nn.Parameter) and model.logit_gain.requires_grad:  # type: ignore[attr-defined]
            gain_grad = delta_tensor * logits_unscaled
            if class_gain_enabled(model):
                model.logit_gain.grad = gain_grad.sum(dim=0).view_as(model.logit_gain) * gain_mask  # type: ignore[attr-defined]
            else:
                model.logit_gain.grad = gain_grad.sum().view_as(model.logit_gain) * gain_mask  # type: ignore[attr-defined]
        model.bias.grad = g_logits.sum(dim=0)  # type: ignore[attr-defined]


def backward_fixedp(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, logits: torch.Tensor, q_out: torch.Tensor, q2_sum: torch.Tensor, direct_logits: torch.Tensor, quad_logits: torch.Tensor) -> torch.Tensor:
    delta_tensor = _effective_delta_for_kernels(model, logits, y, direct_logits, quad_logits)
    grad_direct = torch.empty_like(model.direct_readout)  # type: ignore[attr-defined]
    grad_quad = torch.empty_like(model.quad_readout)  # type: ignore[attr-defined]
    _direct_grad(model, x, delta_tensor, grad_direct)
    _quad_grad(model, q_out, q2_sum, delta_tensor, grad_quad)
    model.direct_readout.grad = grad_direct  # type: ignore[attr-defined]
    model.quad_readout.grad = grad_quad  # type: ignore[attr-defined]
    if isinstance(model.quad_proj, torch.nn.Parameter):  # type: ignore[attr-defined]
        model.quad_proj.grad = None  # type: ignore[attr-defined]
    _scalar_grads(model, logits, delta_tensor, direct_logits, quad_logits)
    return logits.new_tensor(0.0)


def backward_fixedp_workspace(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, logits: torch.Tensor, q_out: torch.Tensor, q2_sum: torch.Tensor, direct_logits: torch.Tensor, quad_logits: torch.Tensor, workspace: FHQWorkspace) -> torch.Tensor:
    delta_tensor = _effective_delta_for_kernels(model, logits, y, direct_logits, quad_logits, workspace.delta[: int(logits.shape[0])])
    _direct_grad(model, x, delta_tensor, workspace.grad_direct)
    _quad_grad(model, q_out, q2_sum, delta_tensor, workspace.grad_quad)
    model.direct_readout.grad = workspace.grad_direct  # type: ignore[attr-defined]
    model.quad_readout.grad = workspace.grad_quad  # type: ignore[attr-defined]
    if isinstance(model.quad_proj, torch.nn.Parameter):  # type: ignore[attr-defined]
        model.quad_proj.grad = None  # type: ignore[attr-defined]
    _scalar_grads_into(model, logits, delta_tensor, direct_logits, quad_logits, workspace.grad_branch, workspace.grad_gain, workspace.grad_bias)
    return logits.new_tensor(0.0)


def backward_learnablep(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, logits: torch.Tensor, q_out: torch.Tensor, q2_sum: torch.Tensor, direct_logits: torch.Tensor, quad_logits: torch.Tensor) -> torch.Tensor:
    assert TRITON_AVAILABLE and triton is not None
    delta_tensor = _effective_delta_for_kernels(model, logits, y, direct_logits, quad_logits)
    grad_direct = torch.empty_like(model.direct_readout)  # type: ignore[attr-defined]
    grad_quad = torch.empty_like(model.quad_readout)  # type: ignore[attr-defined]
    grad_proj = torch.empty_like(model.quad_proj)  # type: ignore[attr-defined]
    _direct_grad(model, x, delta_tensor, grad_direct)
    _quad_grad(model, q_out, q2_sum, delta_tensor, grad_quad)
    dim = int(model.input_dim)  # type: ignore[attr-defined]
    hidden = int(model.hidden_dim)  # type: ignore[attr-defined]
    classes = int(model.output_dim)  # type: ignore[attr-defined]
    _fhq_proj_grad_kernel[(triton.cdiv(dim, 64), triton.cdiv(hidden, 64))](x, model.mu.contiguous(), model.std.contiguous(), q_out, q2_sum, delta_tensor, model.quad_readout.contiguous(), model.branch_scale.contiguous(), model.logit_gain.contiguous(), model.quad_feature_std.contiguous(), grad_proj, B=int(x.shape[0]), D=dim, H=hidden, C=classes, BLOCK_B=128, BLOCK_D=64, BLOCK_H=64, C_BLOCK=triton.next_power_of_2(classes), Q_TANH=bool(getattr(model, "quad_tanh_enabled", False)), Q_RMS=bool(getattr(model, "quad_batch_rms_enabled", False)), CLASS_BRANCH=bool(getattr(model, "class_branch_scale_enabled", False)), CLASS_GAIN=class_gain_enabled(model))  # type: ignore[attr-defined]
    model.direct_readout.grad = grad_direct  # type: ignore[attr-defined]
    model.quad_readout.grad = grad_quad  # type: ignore[attr-defined]
    model.quad_proj.grad = grad_proj  # type: ignore[attr-defined]
    _scalar_grads(model, logits, delta_tensor, direct_logits, quad_logits)
    return logits.new_tensor(0.0)


def backward_learnablep_workspace(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, logits: torch.Tensor, q_out: torch.Tensor, q2_sum: torch.Tensor, direct_logits: torch.Tensor, quad_logits: torch.Tensor, workspace: FHQWorkspace) -> torch.Tensor:
    assert TRITON_AVAILABLE and triton is not None
    if workspace.grad_proj is None:
        raise ValueError("learnable-P workspace requires a learnable quad_proj")
    delta_tensor = _effective_delta_for_kernels(model, logits, y, direct_logits, quad_logits, workspace.delta[: int(logits.shape[0])])
    _direct_grad(model, x, delta_tensor, workspace.grad_direct)
    _quad_grad(model, q_out, q2_sum, delta_tensor, workspace.grad_quad)
    dim = int(model.input_dim)  # type: ignore[attr-defined]
    hidden = int(model.hidden_dim)  # type: ignore[attr-defined]
    classes = int(model.output_dim)  # type: ignore[attr-defined]
    _fhq_proj_grad_kernel[(triton.cdiv(dim, 64), triton.cdiv(hidden, 64))](x, model.mu.contiguous(), model.std.contiguous(), q_out, q2_sum, delta_tensor, model.quad_readout.contiguous(), model.branch_scale.contiguous(), model.logit_gain.contiguous(), model.quad_feature_std.contiguous(), workspace.grad_proj, B=int(x.shape[0]), D=dim, H=hidden, C=classes, BLOCK_B=128, BLOCK_D=64, BLOCK_H=64, C_BLOCK=triton.next_power_of_2(classes), Q_TANH=bool(getattr(model, "quad_tanh_enabled", False)), Q_RMS=bool(getattr(model, "quad_batch_rms_enabled", False)), CLASS_BRANCH=bool(getattr(model, "class_branch_scale_enabled", False)), CLASS_GAIN=class_gain_enabled(model))  # type: ignore[attr-defined]
    model.direct_readout.grad = workspace.grad_direct  # type: ignore[attr-defined]
    model.quad_readout.grad = workspace.grad_quad  # type: ignore[attr-defined]
    model.quad_proj.grad = workspace.grad_proj  # type: ignore[attr-defined]
    _scalar_grads_into(model, logits, delta_tensor, direct_logits, quad_logits, workspace.grad_branch, workspace.grad_gain, workspace.grad_bias)
    return logits.new_tensor(0.0)


def backward_learnablep_workspace_from_grad_logits(model: torch.nn.Module, x: torch.Tensor, grad_logits: torch.Tensor, logits: torch.Tensor, q_out: torch.Tensor, q2_sum: torch.Tensor, direct_logits: torch.Tensor, quad_logits: torch.Tensor, workspace: FHQWorkspace) -> torch.Tensor:
    assert TRITON_AVAILABLE and triton is not None
    if workspace.grad_proj is None:
        raise ValueError("learnable-P workspace requires a learnable quad_proj")
    delta_tensor = _effective_delta_from_grad_logits_for_kernels(model, grad_logits, direct_logits, quad_logits, workspace.delta[: int(logits.shape[0])])
    _direct_grad(model, x, delta_tensor, workspace.grad_direct)
    _quad_grad(model, q_out, q2_sum, delta_tensor, workspace.grad_quad)
    dim = int(model.input_dim)  # type: ignore[attr-defined]
    hidden = int(model.hidden_dim)  # type: ignore[attr-defined]
    classes = int(model.output_dim)  # type: ignore[attr-defined]
    _fhq_proj_grad_kernel[(triton.cdiv(dim, 64), triton.cdiv(hidden, 64))](x, model.mu.contiguous(), model.std.contiguous(), q_out, q2_sum, delta_tensor, model.quad_readout.contiguous(), model.branch_scale.contiguous(), model.logit_gain.contiguous(), model.quad_feature_std.contiguous(), workspace.grad_proj, B=int(x.shape[0]), D=dim, H=hidden, C=classes, BLOCK_B=128, BLOCK_D=64, BLOCK_H=64, C_BLOCK=triton.next_power_of_2(classes), Q_TANH=bool(getattr(model, "quad_tanh_enabled", False)), Q_RMS=bool(getattr(model, "quad_batch_rms_enabled", False)), CLASS_BRANCH=bool(getattr(model, "class_branch_scale_enabled", False)), CLASS_GAIN=class_gain_enabled(model))  # type: ignore[attr-defined]
    model.direct_readout.grad = workspace.grad_direct  # type: ignore[attr-defined]
    model.quad_readout.grad = workspace.grad_quad  # type: ignore[attr-defined]
    model.quad_proj.grad = workspace.grad_proj  # type: ignore[attr-defined]
    _scalar_grads_into(model, logits, delta_tensor, direct_logits, quad_logits, workspace.grad_branch, workspace.grad_gain, workspace.grad_bias)
    return logits.new_tensor(0.0)


def _quad_proj_adamw_state(model: torch.nn.Module, param: torch.nn.Parameter) -> tuple[torch.Tensor, torch.Tensor]:
    exp_avg = getattr(model, "_fhq_quad_proj_exp_avg", None)
    exp_avg_sq = getattr(model, "_fhq_quad_proj_exp_avg_sq", None)
    if exp_avg is None or exp_avg.shape != param.shape or exp_avg.device != param.device:
        exp_avg = torch.zeros_like(param)
        setattr(model, "_fhq_quad_proj_exp_avg", exp_avg)
    if exp_avg_sq is None or exp_avg_sq.shape != param.shape or exp_avg_sq.device != param.device:
        exp_avg_sq = torch.zeros_like(param)
        setattr(model, "_fhq_quad_proj_exp_avg_sq", exp_avg_sq)
    return exp_avg, exp_avg_sq


@torch.no_grad()
def _adamw_update_quad_proj(model: torch.nn.Module, grad_proj: torch.Tensor, *, lr: float, weight_decay: float, step_count: int, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1.0e-8) -> None:
    param = model.quad_proj  # type: ignore[attr-defined]
    if not isinstance(param, torch.nn.Parameter) or not param.requires_grad:
        return
    exp_avg, exp_avg_sq = _quad_proj_adamw_state(model, param)
    if float(weight_decay) != 0.0:
        param.mul_(1.0 - float(lr) * float(weight_decay))
    exp_avg.mul_(float(beta1)).add_(grad_proj, alpha=1.0 - float(beta1))
    exp_avg_sq.mul_(float(beta2)).addcmul_(grad_proj, grad_proj, value=1.0 - float(beta2))
    step = max(1, int(step_count))
    bias_correction1 = 1.0 - float(beta1) ** step
    bias_correction2 = 1.0 - float(beta2) ** step
    step_size = float(lr) * (bias_correction2 ** 0.5) / bias_correction1
    param.addcdiv_(exp_avg, exp_avg_sq.sqrt().add_(float(eps)), value=-step_size)


@torch.no_grad()
def _triton_adamw_update_quad_proj(model: torch.nn.Module, grad_proj: torch.Tensor, *, lr: float, weight_decay: float, step_count: int, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1.0e-8) -> None:
    assert TRITON_AVAILABLE and triton is not None
    param = model.quad_proj  # type: ignore[attr-defined]
    if not isinstance(param, torch.nn.Parameter) or not param.requires_grad:
        return
    exp_avg, exp_avg_sq = _quad_proj_adamw_state(model, param)
    step = max(1, int(step_count))
    step_size = float(lr) * ((1.0 - float(beta2) ** step) ** 0.5) / (1.0 - float(beta1) ** step)
    n = int(param.numel())
    _fhq_adamw_update_kernel[(triton.cdiv(n, 1024),)](
        param,
        grad_proj,
        exp_avg,
        exp_avg_sq,
        step_size,
        N=n,
        LR=float(lr),
        WEIGHT_DECAY=float(weight_decay),
        BETA1=float(beta1),
        BETA2=float(beta2),
        EPS_ADAM=float(eps),
        BLOCK_N=1024,
    )


def backward_learnablep_workspace_fused_quadproj_adamw_from_grad_logits(model: torch.nn.Module, x: torch.Tensor, grad_logits: torch.Tensor, logits: torch.Tensor, q_out: torch.Tensor, q2_sum: torch.Tensor, direct_logits: torch.Tensor, quad_logits: torch.Tensor, workspace: FHQWorkspace, *, lr: float, weight_decay: float, step_count: int) -> torch.Tensor:
    assert TRITON_AVAILABLE and triton is not None
    delta_tensor = _effective_delta_from_grad_logits_for_kernels(model, grad_logits, direct_logits, quad_logits, workspace.delta[: int(logits.shape[0])])
    _direct_grad(model, x, delta_tensor, workspace.grad_direct)
    _quad_grad(model, q_out, q2_sum, delta_tensor, workspace.grad_quad)
    dim = int(model.input_dim)  # type: ignore[attr-defined]
    hidden = int(model.hidden_dim)  # type: ignore[attr-defined]
    classes = int(model.output_dim)  # type: ignore[attr-defined]
    if workspace.grad_proj is None:
        exp_avg, exp_avg_sq = _quad_proj_adamw_state(model, model.quad_proj)  # type: ignore[attr-defined]
        beta1 = 0.9
        beta2 = 0.999
        step = max(1, int(step_count))
        step_size = float(lr) * ((1.0 - beta2 ** step) ** 0.5) / (1.0 - beta1 ** step)
        _fhq_proj_grad_adamw_kernel[(triton.cdiv(dim, 64), triton.cdiv(hidden, 64))](x, model.mu.contiguous(), model.std.contiguous(), q_out, q2_sum, delta_tensor, model.quad_readout.contiguous(), model.branch_scale.contiguous(), model.logit_gain.contiguous(), model.quad_feature_std.contiguous(), model.quad_proj, exp_avg, exp_avg_sq, step_size, B=int(x.shape[0]), D=dim, H=hidden, C=classes, LR=float(lr), WEIGHT_DECAY=float(weight_decay), BETA1=beta1, BETA2=beta2, EPS_ADAM=1.0e-8, BLOCK_B=128, BLOCK_D=64, BLOCK_H=64, C_BLOCK=triton.next_power_of_2(classes), Q_TANH=bool(getattr(model, "quad_tanh_enabled", False)), Q_RMS=bool(getattr(model, "quad_batch_rms_enabled", False)), CLASS_BRANCH=bool(getattr(model, "class_branch_scale_enabled", False)), CLASS_GAIN=class_gain_enabled(model))  # type: ignore[attr-defined]
    else:
        _fhq_proj_grad_kernel[(triton.cdiv(dim, 64), triton.cdiv(hidden, 64))](x, model.mu.contiguous(), model.std.contiguous(), q_out, q2_sum, delta_tensor, model.quad_readout.contiguous(), model.branch_scale.contiguous(), model.logit_gain.contiguous(), model.quad_feature_std.contiguous(), workspace.grad_proj, B=int(x.shape[0]), D=dim, H=hidden, C=classes, BLOCK_B=128, BLOCK_D=64, BLOCK_H=64, C_BLOCK=triton.next_power_of_2(classes), Q_TANH=bool(getattr(model, "quad_tanh_enabled", False)), Q_RMS=bool(getattr(model, "quad_batch_rms_enabled", False)), CLASS_BRANCH=bool(getattr(model, "class_branch_scale_enabled", False)), CLASS_GAIN=class_gain_enabled(model))  # type: ignore[attr-defined]
        _triton_adamw_update_quad_proj(model, workspace.grad_proj, lr=lr, weight_decay=weight_decay, step_count=step_count)
    model.direct_readout.grad = workspace.grad_direct  # type: ignore[attr-defined]
    model.quad_readout.grad = workspace.grad_quad  # type: ignore[attr-defined]
    model.quad_proj.grad = None  # type: ignore[attr-defined]
    _scalar_grads_into(model, logits, delta_tensor, direct_logits, quad_logits, workspace.grad_branch, workspace.grad_gain, workspace.grad_bias)
    return logits.new_tensor(0.0)


def backward_learnablep_workspace_fused_quadproj_adamw(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, logits: torch.Tensor, q_out: torch.Tensor, q2_sum: torch.Tensor, direct_logits: torch.Tensor, quad_logits: torch.Tensor, workspace: FHQWorkspace, *, lr: float, weight_decay: float, step_count: int) -> torch.Tensor:
    grad_logits = delta_into(logits, y, workspace.delta[: int(logits.shape[0])])
    return backward_learnablep_workspace_fused_quadproj_adamw_from_grad_logits(model, x, grad_logits, logits, q_out, q2_sum, direct_logits, quad_logits, workspace, lr=lr, weight_decay=weight_decay, step_count=step_count)
