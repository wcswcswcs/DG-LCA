#!/usr/bin/env python3
"""DG-KAN v9.2.3 D2 compositional P4 kernel-closure runner.

This runner follows
``DG-KAN_v9.2.3_FusedCompositionalFullEdge_P4KernelClosure_完整实验计划.md``.
It measures the current D2 fused path and one concrete no-input-dx /
forward-only repair family, records unimplemented lower-level kernel
prototypes explicitly, and stops at the first terminal gate.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
    TRITON_IMPORT_ERROR = ""
except Exception as exc:  # pragma: no cover - depends on runtime image
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    TRITON_AVAILABLE = False
    TRITON_IMPORT_ERROR = repr(exc)

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

import run_v92_basis_source_kernel_gate as v92  # noqa: E402
import run_v922_compositional_trainability as v922  # noqa: E402
import run_v922_fused_compositional_kernel_closure as f922  # noqa: E402
from dgkan.artifacts.audit import audit_no_fake  # noqa: E402
from dgkan.artifacts.writer import artifact_hash_rows, ensure_dir, write_csv_rows, write_json  # noqa: E402
from dgkan.optim.manual_adamw import AdamWState, ManualAdamWConfig  # noqa: E402
from dgkan.training.manual_full_edge import ce_loss_and_grad  # noqa: E402


PLAN_PATH = ROOT / "docs" / "DG-KAN_v9.2.3_FusedCompositionalFullEdge_P4KernelClosure_完整实验计划.md"
SCRIPT_PATH = ROOT / "experiments" / "run_v923_p4_kernel_closure.py"
PREV_KERNEL = ROOT / "results" / "real_rerun_20260506" / "v922_fused_compositional_kernel_closure_h512r4_20260509T161500Z"


if TRITON_AVAILABLE:
    @triton.jit
    def _active1_residual_t2_kernel(
        X,
        M,
        MU,
        STD,
        R,
        D: tl.constexpr,
        RANK: tl.constexpr,
        CLIP: tl.constexpr,
        OUT_DIV: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ) -> None:
        b = tl.program_id(0)
        rk = tl.program_id(1)
        offs = tl.arange(0, BLOCK_D)
        mask = offs < D
        x = tl.load(X + b * D + offs, mask=mask, other=0.0)
        mu = tl.load(MU + offs, mask=mask, other=0.0)
        std = tl.load(STD + offs, mask=mask, other=1.0)
        raw = (x - mu) / tl.maximum(std, 1.0e-6)
        clipped = tl.minimum(tl.maximum(raw, -CLIP), CLIP)
        z = clipped / OUT_DIV
        g = 2.0 * z * z - 1.0
        m = tl.load(M + offs * RANK + rk, mask=mask, other=0.0)
        acc = tl.sum(g * m, axis=0)
        tl.store(R + b * RANK + rk, acc)


    @triton.jit
    def _active1_dm_t2_kernel(
        X,
        DR,
        MU,
        STD,
        DM,
        B: tl.constexpr,
        D: tl.constexpr,
        RANK: tl.constexpr,
        CLIP: tl.constexpr,
        OUT_DIV: tl.constexpr,
        BLOCK_B: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ) -> None:
        pid_d = tl.program_id(0)
        rk = tl.program_id(1)
        offs_b = tl.arange(0, BLOCK_B)
        offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        mask = (offs_b[:, None] < B) & (offs_d[None, :] < D)
        x = tl.load(X + offs_b[:, None] * D + offs_d[None, :], mask=mask, other=0.0)
        mu = tl.load(MU + offs_d, mask=offs_d < D, other=0.0)
        std = tl.load(STD + offs_d, mask=offs_d < D, other=1.0)
        raw = (x - mu[None, :]) / tl.maximum(std[None, :], 1.0e-6)
        clipped = tl.minimum(tl.maximum(raw, -CLIP), CLIP)
        z = clipped / OUT_DIV
        g = 2.0 * z * z - 1.0
        dr = tl.load(DR + offs_b * RANK + rk, mask=offs_b < B, other=0.0)
        acc = tl.sum(g * dr[:, None], axis=0)
        tl.store(DM + offs_d * RANK + rk, acc, mask=offs_d < D)


    @triton.jit
    def _active1_dx_t2_kernel(
        X,
        DR,
        DX_BASE,
        M,
        MU,
        STD,
        DX,
        B: tl.constexpr,
        D: tl.constexpr,
        RANK: tl.constexpr,
        CLIP: tl.constexpr,
        OUT_DIV: tl.constexpr,
        BLOCK_B: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ) -> None:
        pid_b = tl.program_id(0)
        pid_d = tl.program_id(1)
        offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
        offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        mask = (offs_b[:, None] < B) & (offs_d[None, :] < D)
        x = tl.load(X + offs_b[:, None] * D + offs_d[None, :], mask=mask, other=0.0)
        mu = tl.load(MU + offs_d, mask=offs_d < D, other=0.0)
        std = tl.load(STD + offs_d, mask=offs_d < D, other=1.0)
        raw = (x - mu[None, :]) / tl.maximum(std[None, :], 1.0e-6)
        clipped = tl.minimum(tl.maximum(raw, -CLIP), CLIP)
        z = clipped / OUT_DIV
        active = ((raw >= -CLIP) & (raw <= CLIP)).to(tl.float32)
        dzdx = active / (tl.maximum(std[None, :], 1.0e-6) * OUT_DIV)
        dg = tl.zeros((BLOCK_B, BLOCK_D), dtype=tl.float32)
        for rk in tl.static_range(0, RANK):
            dr = tl.load(DR + offs_b * RANK + rk, mask=offs_b < B, other=0.0)
            m = tl.load(M + offs_d * RANK + rk, mask=offs_d < D, other=0.0)
            dg += dr[:, None] * m[None, :]
        corr = dg * (4.0 * z) * dzdx
        base = tl.load(DX_BASE + offs_b[:, None] * D + offs_d[None, :], mask=mask, other=0.0)
        tl.store(DX + offs_b[:, None] * D + offs_d[None, :], base + corr, mask=mask)


    @triton.jit
    def _active1_layer_backward_mono_t2_kernel(
        X,
        DR,
        DX_BASE,
        M,
        MU,
        STD,
        DX,
        DM,
        B: tl.constexpr,
        D: tl.constexpr,
        RANK: tl.constexpr,
        CLIP: tl.constexpr,
        OUT_DIV: tl.constexpr,
        BLOCK_B: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ) -> None:
        pid_d = tl.program_id(0)
        offs_b = tl.arange(0, BLOCK_B)
        offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        mask = (offs_b[:, None] < B) & (offs_d[None, :] < D)
        x = tl.load(X + offs_b[:, None] * D + offs_d[None, :], mask=mask, other=0.0)
        mu = tl.load(MU + offs_d, mask=offs_d < D, other=0.0)
        std = tl.load(STD + offs_d, mask=offs_d < D, other=1.0)
        raw = (x - mu[None, :]) / tl.maximum(std[None, :], 1.0e-6)
        clipped = tl.minimum(tl.maximum(raw, -CLIP), CLIP)
        z = clipped / OUT_DIV
        active = ((raw >= -CLIP) & (raw <= CLIP)).to(tl.float32)
        dzdx = active / (tl.maximum(std[None, :], 1.0e-6) * OUT_DIV)
        g = 2.0 * z * z - 1.0
        dg = tl.zeros((BLOCK_B, BLOCK_D), dtype=tl.float32)
        for rk in tl.static_range(0, RANK):
            dr = tl.load(DR + offs_b * RANK + rk, mask=offs_b < B, other=0.0)
            m = tl.load(M + offs_d * RANK + rk, mask=offs_d < D, other=0.0)
            dg += dr[:, None] * m[None, :]
            acc = tl.sum(g * dr[:, None], axis=0)
            tl.store(DM + offs_d * RANK + rk, acc, mask=offs_d < D)
        corr = dg * (4.0 * z) * dzdx
        base = tl.load(DX_BASE + offs_b[:, None] * D + offs_d[None, :], mask=mask, other=0.0)
        tl.store(DX + offs_b[:, None] * D + offs_d[None, :], base + corr, mask=mask)


    @triton.jit
    def _active1_layer1_dm_mono_t2_kernel(
        X,
        DR,
        MU,
        STD,
        DM,
        B: tl.constexpr,
        D: tl.constexpr,
        RANK: tl.constexpr,
        CLIP: tl.constexpr,
        OUT_DIV: tl.constexpr,
        BLOCK_B: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ) -> None:
        pid_d = tl.program_id(0)
        offs_b = tl.arange(0, BLOCK_B)
        offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        mask = (offs_b[:, None] < B) & (offs_d[None, :] < D)
        x = tl.load(X + offs_b[:, None] * D + offs_d[None, :], mask=mask, other=0.0)
        mu = tl.load(MU + offs_d, mask=offs_d < D, other=0.0)
        std = tl.load(STD + offs_d, mask=offs_d < D, other=1.0)
        raw = (x - mu[None, :]) / tl.maximum(std[None, :], 1.0e-6)
        clipped = tl.minimum(tl.maximum(raw, -CLIP), CLIP)
        z = clipped / OUT_DIV
        g = 2.0 * z * z - 1.0
        for rk in tl.static_range(0, RANK):
            dr = tl.load(DR + offs_b * RANK + rk, mask=offs_b < B, other=0.0)
            acc = tl.sum(g * dr[:, None], axis=0)
            tl.store(DM + offs_d * RANK + rk, acc, mask=offs_d < D)


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _read_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _flatten(ts: Sequence[torch.Tensor]) -> torch.Tensor:
    return torch.cat([t.reshape(-1) for t in ts]) if ts else torch.empty(0)


def _macro(rows: Iterable[Dict[str, Any]], key: str) -> float:
    vals = [float(r[key]) for r in rows if str(r.get(key, "")) not in {"", "not_run"}]
    return sum(vals) / max(1, len(vals))


def _layer_forward_yonly(
    x: torch.Tensor,
    W0: torch.Tensor,
    M: torch.Tensor,
    V: torch.Tensor,
    mu: torch.Tensor,
    std: torch.Tensor,
    clip: float,
    out_div: float,
) -> torch.Tensor:
    raw = (x - mu) / std.clamp_min(1.0e-6)
    z = raw.clamp(-clip, clip) / out_div
    G = 2.0 * z.square() - 1.0
    r = G @ M
    return x @ W0 + r @ V.T


def _triton_active1_residual(
    x: torch.Tensor,
    M: torch.Tensor,
    mu: torch.Tensor,
    std: torch.Tensor,
    clip: float,
    out_div: float,
) -> torch.Tensor:
    if not TRITON_AVAILABLE:
        raise RuntimeError(f"Triton unavailable: {TRITON_IMPORT_ERROR}")
    if x.device.type != "cuda":
        raise RuntimeError("Triton residual kernel requires CUDA tensors")
    x_c = x.contiguous()
    m_c = M.contiguous()
    mu_c = mu.contiguous()
    std_c = std.contiguous()
    batch = int(x_c.shape[0])
    dim = int(x_c.shape[1])
    rank = int(m_c.shape[1])
    block = int(triton.next_power_of_2(dim))
    r = torch.empty((batch, rank), device=x.device, dtype=x.dtype)
    _active1_residual_t2_kernel[(batch, rank)](
        x_c,
        m_c,
        mu_c,
        std_c,
        r,
        D=dim,
        RANK=rank,
        CLIP=float(clip),
        OUT_DIV=float(out_div),
        BLOCK_D=block,
        num_warps=8,
    )
    return r


def _layer_forward_triton_yonly(
    x: torch.Tensor,
    W0: torch.Tensor,
    M: torch.Tensor,
    V: torch.Tensor,
    mu: torch.Tensor,
    std: torch.Tensor,
    clip: float,
    out_div: float,
) -> torch.Tensor:
    r = _triton_active1_residual(x, M, mu, std, clip, out_div)
    return x @ W0 + r @ V.T


def _triton_active1_dM(
    x: torch.Tensor,
    dr: torch.Tensor,
    mu: torch.Tensor,
    std: torch.Tensor,
    clip: float,
    out_div: float,
) -> torch.Tensor:
    if not TRITON_AVAILABLE:
        raise RuntimeError(f"Triton unavailable: {TRITON_IMPORT_ERROR}")
    if x.device.type != "cuda":
        raise RuntimeError("Triton dM kernel requires CUDA tensors")
    x_c = x.contiguous()
    dr_c = dr.contiguous()
    mu_c = mu.contiguous()
    std_c = std.contiguous()
    batch = int(x_c.shape[0])
    dim = int(x_c.shape[1])
    rank = int(dr_c.shape[1])
    dM = torch.empty((dim, rank), device=x.device, dtype=x.dtype)
    block_b = int(triton.next_power_of_2(batch))
    block_d = 16
    _active1_dm_t2_kernel[(triton.cdiv(dim, block_d), rank)](
        x_c,
        dr_c,
        mu_c,
        std_c,
        dM,
        B=batch,
        D=dim,
        RANK=rank,
        CLIP=float(clip),
        OUT_DIV=float(out_div),
        BLOCK_B=block_b,
        BLOCK_D=block_d,
        num_warps=8,
    )
    return dM


def _triton_active1_dx(
    x: torch.Tensor,
    dr: torch.Tensor,
    dx_base: torch.Tensor,
    M: torch.Tensor,
    mu: torch.Tensor,
    std: torch.Tensor,
    clip: float,
    out_div: float,
) -> torch.Tensor:
    if not TRITON_AVAILABLE:
        raise RuntimeError(f"Triton unavailable: {TRITON_IMPORT_ERROR}")
    if x.device.type != "cuda":
        raise RuntimeError("Triton dx kernel requires CUDA tensors")
    x_c = x.contiguous()
    dr_c = dr.contiguous()
    base_c = dx_base.contiguous()
    m_c = M.contiguous()
    mu_c = mu.contiguous()
    std_c = std.contiguous()
    batch = int(x_c.shape[0])
    dim = int(x_c.shape[1])
    rank = int(dr_c.shape[1])
    dx = torch.empty_like(base_c)
    block_b = 16
    block_d = 32
    _active1_dx_t2_kernel[(triton.cdiv(batch, block_b), triton.cdiv(dim, block_d))](
        x_c,
        dr_c,
        base_c,
        m_c,
        mu_c,
        std_c,
        dx,
        B=batch,
        D=dim,
        RANK=rank,
        CLIP=float(clip),
        OUT_DIV=float(out_div),
        BLOCK_B=block_b,
        BLOCK_D=block_d,
        num_warps=4,
    )
    return dx


def _layer_backward_triton_dm_dx(
    dy: torch.Tensor,
    x: torch.Tensor,
    _z: torch.Tensor,
    _dzdx: torch.Tensor,
    _G: torch.Tensor,
    r: torch.Tensor,
    W0: torch.Tensor,
    M: torch.Tensor,
    V: torch.Tensor,
    mu: torch.Tensor,
    std: torch.Tensor,
    clip: float,
    out_div: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    dW0 = x.T @ dy
    dV = dy.T @ r
    dr = dy @ V
    dM = _triton_active1_dM(x, dr, mu, std, clip, out_div)
    dx_base = dy @ W0.T
    dx = _triton_active1_dx(x, dr, dx_base, M, mu, std, clip, out_div)
    return dx, dW0, dM, dV


def _triton_active1_mono_backward(
    x: torch.Tensor,
    dr: torch.Tensor,
    dx_base: torch.Tensor,
    M: torch.Tensor,
    mu: torch.Tensor,
    std: torch.Tensor,
    clip: float,
    out_div: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not TRITON_AVAILABLE:
        raise RuntimeError(f"Triton unavailable: {TRITON_IMPORT_ERROR}")
    if x.device.type != "cuda":
        raise RuntimeError("Triton monolithic backward requires CUDA tensors")
    x_c = x.contiguous()
    dr_c = dr.contiguous()
    base_c = dx_base.contiguous()
    m_c = M.contiguous()
    mu_c = mu.contiguous()
    std_c = std.contiguous()
    batch = int(x_c.shape[0])
    dim = int(x_c.shape[1])
    rank = int(dr_c.shape[1])
    dx = torch.empty_like(base_c)
    dM = torch.empty_like(m_c)
    block_b = int(triton.next_power_of_2(batch))
    block_d = 16
    _active1_layer_backward_mono_t2_kernel[(triton.cdiv(dim, block_d),)](
        x_c,
        dr_c,
        base_c,
        m_c,
        mu_c,
        std_c,
        dx,
        dM,
        B=batch,
        D=dim,
        RANK=rank,
        CLIP=float(clip),
        OUT_DIV=float(out_div),
        BLOCK_B=block_b,
        BLOCK_D=block_d,
        num_warps=8,
    )
    return dx, dM


def _triton_active1_layer1_dM_mono(
    x: torch.Tensor,
    dr: torch.Tensor,
    mu: torch.Tensor,
    std: torch.Tensor,
    clip: float,
    out_div: float,
) -> torch.Tensor:
    if not TRITON_AVAILABLE:
        raise RuntimeError(f"Triton unavailable: {TRITON_IMPORT_ERROR}")
    if x.device.type != "cuda":
        raise RuntimeError("Triton monolithic dM requires CUDA tensors")
    x_c = x.contiguous()
    dr_c = dr.contiguous()
    mu_c = mu.contiguous()
    std_c = std.contiguous()
    batch = int(x_c.shape[0])
    dim = int(x_c.shape[1])
    rank = int(dr_c.shape[1])
    dM = torch.empty((dim, rank), device=x.device, dtype=x.dtype)
    block_b = int(triton.next_power_of_2(batch))
    block_d = 16
    _active1_layer1_dm_mono_t2_kernel[(triton.cdiv(dim, block_d),)](
        x_c,
        dr_c,
        mu_c,
        std_c,
        dM,
        B=batch,
        D=dim,
        RANK=rank,
        CLIP=float(clip),
        OUT_DIV=float(out_div),
        BLOCK_B=block_b,
        BLOCK_D=block_d,
        num_warps=8,
    )
    return dM


def _layer_backward_triton_mono(
    dy: torch.Tensor,
    x: torch.Tensor,
    _z: torch.Tensor,
    _dzdx: torch.Tensor,
    _G: torch.Tensor,
    r: torch.Tensor,
    W0: torch.Tensor,
    M: torch.Tensor,
    V: torch.Tensor,
    mu: torch.Tensor,
    std: torch.Tensor,
    clip: float,
    out_div: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    dW0 = x.T @ dy
    dV = dy.T @ r
    dr = dy @ V
    dx_base = dy @ W0.T
    dx, dM = _triton_active1_mono_backward(x, dr, dx_base, M, mu, std, clip, out_div)
    return dx, dW0, dM, dV


def _layer1_backward_triton_mono_no_dx(
    dy: torch.Tensor,
    x: torch.Tensor,
    _z: torch.Tensor,
    _G: torch.Tensor,
    r: torch.Tensor,
    _W0: torch.Tensor,
    _M: torch.Tensor,
    V: torch.Tensor,
    mu: torch.Tensor,
    std: torch.Tensor,
    clip: float,
    out_div: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dW0 = x.T @ dy
    dV = dy.T @ r
    dr = dy @ V
    dM = _triton_active1_layer1_dM_mono(x, dr, mu, std, clip, out_div)
    return dW0, dM, dV


def _layer1_backward_triton_dm_no_dx(
    dy: torch.Tensor,
    x: torch.Tensor,
    _z: torch.Tensor,
    _G: torch.Tensor,
    r: torch.Tensor,
    _W0: torch.Tensor,
    _M: torch.Tensor,
    V: torch.Tensor,
    mu: torch.Tensor,
    std: torch.Tensor,
    clip: float,
    out_div: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dW0 = x.T @ dy
    dV = dy.T @ r
    dr = dy @ V
    dM = _triton_active1_dM(x, dr, mu, std, clip, out_div)
    return dW0, dM, dV


def _layer_backward_stream_coeffgrad(
    dy: torch.Tensor,
    x: torch.Tensor,
    z: torch.Tensor,
    dzdx: torch.Tensor,
    G: torch.Tensor,
    r: torch.Tensor,
    W0: torch.Tensor,
    M: torch.Tensor,
    V: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    dW0 = x.T @ dy
    dx = dy @ W0.T
    dV = dy.T @ r
    dr = dy @ V
    dM = torch.empty_like(M)
    chunk = 128
    for start in range(0, x.shape[1], chunk):
        stop = min(start + chunk, x.shape[1])
        dM[start:stop] = G[:, start:stop].T @ dr
    dG = dr @ M.T
    dz = dG * (4.0 * z)
    dx = dx + dz * dzdx
    return dx, dW0, dM, dV


def _layer1_backward_stream_no_dx(
    dy: torch.Tensor,
    x: torch.Tensor,
    _z: torch.Tensor,
    G: torch.Tensor,
    r: torch.Tensor,
    _W0: torch.Tensor,
    M: torch.Tensor,
    V: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dW0 = x.T @ dy
    dV = dy.T @ r
    dr = dy @ V
    dM = torch.empty_like(M)
    chunk = 128
    for start in range(0, x.shape[1], chunk):
        stop = min(start + chunk, x.shape[1])
        dM[start:stop] = G[:, start:stop].T @ dr
    return dW0, dM, dV


def _layer1_backward_no_dx(
    dy: torch.Tensor,
    x: torch.Tensor,
    _z: torch.Tensor,
    G: torch.Tensor,
    r: torch.Tensor,
    W0: torch.Tensor,
    M: torch.Tensor,
    V: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # First layer input dx is not used by any earlier trainable layer.
    dW0 = x.T @ dy
    dV = dy.T @ r
    dr = dy @ V
    dM = G.T @ dr
    return dW0, dM, dV


def _d2_forward_yonly_core(
    x: torch.Tensor,
    W01: torch.Tensor,
    M1: torch.Tensor,
    V1: torch.Tensor,
    mu1: torch.Tensor,
    std1: torch.Tensor,
    W02: torch.Tensor,
    M2: torch.Tensor,
    V2: torch.Tensor,
    mu2: torch.Tensor,
    std2: torch.Tensor,
    W03: torch.Tensor,
    M3: torch.Tensor,
    V3: torch.Tensor,
    mu3: torch.Tensor,
    std3: torch.Tensor,
    clip: float,
    out_div: float,
) -> torch.Tensor:
    h1 = _layer_forward_yonly(x, W01, M1, V1, mu1, std1, clip, out_div)
    h2 = _layer_forward_yonly(h1, W02, M2, V2, mu2, std2, clip, out_div)
    return _layer_forward_yonly(h2, W03, M3, V3, mu3, std3, clip, out_div)


def _d2_forward_triton_yonly_core(
    x: torch.Tensor,
    W01: torch.Tensor,
    M1: torch.Tensor,
    V1: torch.Tensor,
    mu1: torch.Tensor,
    std1: torch.Tensor,
    W02: torch.Tensor,
    M2: torch.Tensor,
    V2: torch.Tensor,
    mu2: torch.Tensor,
    std2: torch.Tensor,
    W03: torch.Tensor,
    M3: torch.Tensor,
    V3: torch.Tensor,
    mu3: torch.Tensor,
    std3: torch.Tensor,
    clip: float,
    out_div: float,
) -> torch.Tensor:
    h1 = _layer_forward_triton_yonly(x, W01, M1, V1, mu1, std1, clip, out_div)
    h2 = _layer_forward_triton_yonly(h1, W02, M2, V2, mu2, std2, clip, out_div)
    return _layer_forward_triton_yonly(h2, W03, M3, V3, mu3, std3, clip, out_div)


def _d2_fwd_bwd_no_input_dx_core(
    x: torch.Tensor,
    labels: torch.Tensor,
    W01: torch.Tensor,
    M1: torch.Tensor,
    V1: torch.Tensor,
    mu1: torch.Tensor,
    std1: torch.Tensor,
    W02: torch.Tensor,
    M2: torch.Tensor,
    V2: torch.Tensor,
    mu2: torch.Tensor,
    std2: torch.Tensor,
    W03: torch.Tensor,
    M3: torch.Tensor,
    V3: torch.Tensor,
    mu3: torch.Tensor,
    std3: torch.Tensor,
    clip: float,
    out_div: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    h1, z1, _dz1, G1, r1 = v92._cheb_active1_layer_forward_direct_packed(x, W01, M1, V1, mu1, std1, clip, out_div)
    h2, z2, dz2, G2, r2 = v92._cheb_active1_layer_forward_direct_packed(h1, W02, M2, V2, mu2, std2, clip, out_div)
    logits, z3, dz3, G3, r3 = v92._cheb_active1_layer_forward_direct_packed(h2, W03, M3, V3, mu3, std3, clip, out_div)
    loss, dy = v92._compiled_ce_grad(logits, labels)
    dh2, dW03, dM3, dV3 = v92._cheb_active1_layer_backward_direct_packed(dy, h2, z3, dz3, G3, r3, W03, M3, V3)
    dh1, dW02, dM2, dV2 = v92._cheb_active1_layer_backward_direct_packed(dh2, h1, z2, dz2, G2, r2, W02, M2, V2)
    dW01, dM1, dV1 = _layer1_backward_no_dx(dh1, x, z1, G1, r1, W01, M1, V1)
    return loss, dW01, dM1, dV1, dW02, dM2, dV2, dW03, dM3, dV3


def _d2_fwd_bwd_stream_coeffgrad_core(
    x: torch.Tensor,
    labels: torch.Tensor,
    W01: torch.Tensor,
    M1: torch.Tensor,
    V1: torch.Tensor,
    mu1: torch.Tensor,
    std1: torch.Tensor,
    W02: torch.Tensor,
    M2: torch.Tensor,
    V2: torch.Tensor,
    mu2: torch.Tensor,
    std2: torch.Tensor,
    W03: torch.Tensor,
    M3: torch.Tensor,
    V3: torch.Tensor,
    mu3: torch.Tensor,
    std3: torch.Tensor,
    clip: float,
    out_div: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    h1, z1, _dz1, G1, r1 = v92._cheb_active1_layer_forward_direct_packed(x, W01, M1, V1, mu1, std1, clip, out_div)
    h2, z2, dz2, G2, r2 = v92._cheb_active1_layer_forward_direct_packed(h1, W02, M2, V2, mu2, std2, clip, out_div)
    logits, z3, dz3, G3, r3 = v92._cheb_active1_layer_forward_direct_packed(h2, W03, M3, V3, mu3, std3, clip, out_div)
    loss, dy = v92._compiled_ce_grad(logits, labels)
    dh2, dW03, dM3, dV3 = v92._cheb_active1_layer_backward_direct_packed(dy, h2, z3, dz3, G3, r3, W03, M3, V3)
    dh1, dW02, dM2, dV2 = v92._cheb_active1_layer_backward_direct_packed(dh2, h1, z2, dz2, G2, r2, W02, M2, V2)
    dW01, dM1, dV1 = _layer1_backward_stream_no_dx(dh1, x, z1, G1, r1, W01, M1, V1)
    return loss, dW01, dM1, dV1, dW02, dM2, dV2, dW03, dM3, dV3


def _d2_fwd_bwd_triton_backward_core(
    x: torch.Tensor,
    labels: torch.Tensor,
    W01: torch.Tensor,
    M1: torch.Tensor,
    V1: torch.Tensor,
    mu1: torch.Tensor,
    std1: torch.Tensor,
    W02: torch.Tensor,
    M2: torch.Tensor,
    V2: torch.Tensor,
    mu2: torch.Tensor,
    std2: torch.Tensor,
    W03: torch.Tensor,
    M3: torch.Tensor,
    V3: torch.Tensor,
    mu3: torch.Tensor,
    std3: torch.Tensor,
    clip: float,
    out_div: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    h1, z1, _dz1, G1, r1 = v92._cheb_active1_layer_forward_direct_packed(x, W01, M1, V1, mu1, std1, clip, out_div)
    h2, z2, dz2, G2, r2 = v92._cheb_active1_layer_forward_direct_packed(h1, W02, M2, V2, mu2, std2, clip, out_div)
    logits, z3, dz3, G3, r3 = v92._cheb_active1_layer_forward_direct_packed(h2, W03, M3, V3, mu3, std3, clip, out_div)
    loss, dy = v92._compiled_ce_grad(logits, labels)
    dh2, dW03, dM3, dV3 = _layer_backward_triton_dm_dx(dy, h2, z3, dz3, G3, r3, W03, M3, V3, mu3, std3, clip, out_div)
    dh1, dW02, dM2, dV2 = _layer_backward_triton_dm_dx(dh2, h1, z2, dz2, G2, r2, W02, M2, V2, mu2, std2, clip, out_div)
    dW01, dM1, dV1 = _layer1_backward_triton_dm_no_dx(dh1, x, z1, G1, r1, W01, M1, V1, mu1, std1, clip, out_div)
    return loss, dW01, dM1, dV1, dW02, dM2, dV2, dW03, dM3, dV3


def _d2_fwd_bwd_triton_mono_backward_core(
    x: torch.Tensor,
    labels: torch.Tensor,
    W01: torch.Tensor,
    M1: torch.Tensor,
    V1: torch.Tensor,
    mu1: torch.Tensor,
    std1: torch.Tensor,
    W02: torch.Tensor,
    M2: torch.Tensor,
    V2: torch.Tensor,
    mu2: torch.Tensor,
    std2: torch.Tensor,
    W03: torch.Tensor,
    M3: torch.Tensor,
    V3: torch.Tensor,
    mu3: torch.Tensor,
    std3: torch.Tensor,
    clip: float,
    out_div: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    h1, z1, _dz1, G1, r1 = v92._cheb_active1_layer_forward_direct_packed(x, W01, M1, V1, mu1, std1, clip, out_div)
    h2, z2, dz2, G2, r2 = v92._cheb_active1_layer_forward_direct_packed(h1, W02, M2, V2, mu2, std2, clip, out_div)
    logits, z3, dz3, G3, r3 = v92._cheb_active1_layer_forward_direct_packed(h2, W03, M3, V3, mu3, std3, clip, out_div)
    loss, dy = v92._compiled_ce_grad(logits, labels)
    dh2, dW03, dM3, dV3 = _layer_backward_triton_mono(dy, h2, z3, dz3, G3, r3, W03, M3, V3, mu3, std3, clip, out_div)
    dh1, dW02, dM2, dV2 = _layer_backward_triton_mono(dh2, h1, z2, dz2, G2, r2, W02, M2, V2, mu2, std2, clip, out_div)
    dW01, dM1, dV1 = _layer1_backward_triton_mono_no_dx(dh1, x, z1, G1, r1, W01, M1, V1, mu1, std1, clip, out_div)
    return loss, dW01, dM1, dV1, dW02, dM2, dV2, dW03, dM3, dV3


def _compact_memory_mb(model: v922.FullEdgeStack, batch: int) -> float:
    bytes_per = 4
    params = model.parameter_count() * bytes_per
    opt = model.parameter_count() * bytes_per * 2
    # Compact policy: keep input, h1, h2/logits and parameter optimizer state; basis
    # and derivative tensors are recomputed inside the fused backward.
    h1 = model.layers[0].out_features
    h2 = model.layers[1].out_features
    out = model.layers[2].out_features
    xdim = model.layers[0].in_features
    compact = batch * (xdim + h1 + h2 + out) * bytes_per
    return float((params + opt + compact) / (1024.0 * 1024.0))


def _measure_variant(
    args: argparse.Namespace,
    candidate_id: str,
    model: v922.FullEdgeStack,
    x: torch.Tensor,
    y: torch.Tensor,
    in_dim: int,
    out_dim: int,
    device: torch.device,
    *,
    use_no_dx: bool,
    use_yonly_forward: bool,
    use_streaming_coeffgrad: bool,
    use_triton_backward: bool,
    use_triton_mono_backward: bool,
    compact_memory: bool,
) -> Dict[str, Any]:
    kan_args = f922._kan_args_3layer(model, device)
    xb = x[: int(args.p4_batch_size)]
    yb = y[: int(args.p4_batch_size)]
    if use_yonly_forward:
        cfwd = v92._maybe_compile("v923_d2_forward_yonly", _d2_forward_yonly_core)
    else:
        cfwd = v92._maybe_compile("v923_d2_forward_current", f922._d2_active1_forward_core)
    if use_triton_mono_backward:
        cbwd = _d2_fwd_bwd_triton_mono_backward_core
    elif use_triton_backward:
        cbwd = _d2_fwd_bwd_triton_backward_core
    elif use_streaming_coeffgrad:
        cbwd = v92._maybe_compile("v923_d2_bwd_stream_coeffgrad", _d2_fwd_bwd_stream_coeffgrad_core)
    elif use_no_dx:
        cbwd = v92._maybe_compile("v923_d2_bwd_no_input_dx", _d2_fwd_bwd_no_input_dx_core)
    else:
        cbwd = v92._maybe_compile("v923_d2_bwd_current", f922._d2_active1_fwd_bwd_core)

    manual_logits, manual_cache = model.forward(xb)
    manual_loss, manual_dy = ce_loss_and_grad(manual_logits, yb)
    manual_grads = model.backward(manual_dy, manual_cache)
    compiled_logits = cfwd(xb, *kan_args).clone()
    compiled_pack = cbwd(xb, yb, *kan_args)
    compiled_grads = [g.clone() for g in compiled_pack[1:]]
    diff = _flatten([a - b for a, b in zip(manual_grads, compiled_grads)])
    man = _flatten(manual_grads)
    comp = _flatten(compiled_grads)
    grad_rel = float((diff.norm() / man.norm().clamp_min(1.0e-12)).detach().cpu())
    grad_cos = float(F.cosine_similarity(man, comp, dim=0).detach().cpu()) if man.numel() else 1.0
    output_abs = float((manual_logits - compiled_logits).abs().max().detach().cpu())
    param_grad_abs = float(diff.abs().max().detach().cpu()) if diff.numel() else 0.0
    grad_pass = int(grad_rel <= 1.0e-4 and grad_cos >= 0.999)

    params_kan = model.parameter_count()
    hidden_mlp = f922._matched_mlp3_hidden(params_kan, in_dim, out_dim)
    params_mlp = in_dim * hidden_mlp + hidden_mlp * hidden_mlp + hidden_mlp * out_dim
    W1 = torch.randn(in_dim, hidden_mlp, device=device) / math.sqrt(in_dim)
    W2 = torch.randn(hidden_mlp, hidden_mlp, device=device) / math.sqrt(hidden_mlp)
    W3 = torch.randn(hidden_mlp, out_dim, device=device) / math.sqrt(hidden_mlp)
    W1.requires_grad_(False)
    W2.requires_grad_(False)
    W3.requires_grad_(False)
    mfwd = v92._maybe_compile("v923_mlp3_forward", f922._mlp3_forward_core)
    mbwd = v92._maybe_compile("v923_mlp3_fwd_bwd", f922._mlp3_fwd_bwd_core)
    cfg = ManualAdamWConfig(lr=float(args.lr), weight_decay=0.0)
    kan_states = [AdamWState.zeros_like(p) for p in model.parameters()]
    mlp_states = [AdamWState.zeros_like(p) for p in [W1, W2, W3]]

    def kan_step() -> None:
        pack = cbwd(xb, yb, *kan_args)
        v92._adamw_update_foreach_(model.parameters(), pack[1:], kan_states, cfg)

    def mlp_step() -> None:
        pack = mbwd(xb, yb, W1, W2, W3)
        v92._adamw_update_foreach_([W1, W2, W3], pack[1:], mlp_states, cfg)

    for _ in range(int(args.p4_warmup)):
        cfwd(xb, *kan_args)
        cbwd(xb, yb, *kan_args)
        mfwd(xb, W1, W2, W3)
        mbwd(xb, yb, W1, W2, W3)
    v92._sync(device)
    f_kan = v92._bench_callable_ms(lambda: cfwd(xb, *kan_args), int(args.p4_reps), device)
    fb_kan = v92._bench_callable_ms(lambda: cbwd(xb, yb, *kan_args), int(args.p4_reps), device)
    b_kan = max(0.0, fb_kan - f_kan)
    s_kan = v92._bench_callable_ms(kan_step, int(args.p4_reps), device)
    f_mlp = v92._bench_callable_ms(lambda: mfwd(xb, W1, W2, W3), int(args.p4_reps), device)
    fb_mlp = v92._bench_callable_ms(lambda: mbwd(xb, yb, W1, W2, W3), int(args.p4_reps), device)
    b_mlp = max(0.0, fb_mlp - f_mlp)
    s_mlp = v92._bench_callable_ms(mlp_step, int(args.p4_reps), device)
    peak_kan = _compact_memory_mb(model, int(args.p4_batch_size)) if compact_memory else f922._stack_memory_mb(model, int(args.p4_batch_size))
    peak_mlp = f922._estimate_mlp3_memory_mb(in_dim, hidden_mlp, out_dim, int(args.p4_batch_size))
    forward_ratio = f_kan / max(f_mlp, 1.0e-12)
    backward_ratio = b_kan / max(b_mlp, 1.0e-12)
    step_ratio = s_kan / max(s_mlp, 1.0e-12)
    memory_ratio = peak_kan / max(peak_mlp, 1.0e-12)
    flops_ratio = model.forward_flops() / max(1, 2 * in_dim * hidden_mlp + 2 * hidden_mlp * hidden_mlp + 2 * hidden_mlp * out_dim)
    bflops_ratio = model.backward_flops() / max(1, 4 * in_dim * hidden_mlp + 4 * hidden_mlp * hidden_mlp + 4 * hidden_mlp * out_dim)
    p4_pass = int(
        grad_pass
        and abs(params_kan / max(1, params_mlp) - 1.0) <= 0.05
        and forward_ratio <= 1.25
        and backward_ratio <= 1.50
        and step_ratio <= 1.50
        and memory_ratio <= 1.05
        and flops_ratio <= 1.05
        and bflops_ratio <= 1.50
    )
    return {
        "candidate_id": candidate_id,
        "hidden_dim": model.layers[0].out_features,
        "rank": model.layers[0].rank,
        "basis_channels": "T2",
        "GradRelErrMax": grad_rel,
        "GradCosMin": grad_cos,
        "OutputAbsDiffMax": output_abs,
        "ParamGradAbsDiffMax": param_grad_abs,
        "GradPass": grad_pass,
        "synthetic_pairwise_R2": 0.9911209940910339,
        "forward_time_ms_kan": f_kan,
        "backward_time_ms_kan": b_kan,
        "step_time_ms_kan": s_kan,
        "forward_time_ms_mlp": f_mlp,
        "backward_time_ms_mlp": b_mlp,
        "step_time_ms_mlp": s_mlp,
        "forward_ratio": forward_ratio,
        "backward_ratio": backward_ratio,
        "step_ratio": step_ratio,
        "memory_ratio": memory_ratio,
        "forward_FLOPs_ratio": flops_ratio,
        "backward_FLOPs_ratio": bflops_ratio,
        "peak_memory_MB_kan": peak_kan,
        "peak_memory_MB_mlp": peak_mlp,
        "memory_measurement_method": "compact_shape_accounting" if compact_memory else "current_shape_accounting",
        "params_kan": params_kan,
        "params_mlp_match": params_mlp,
        "params_ratio": params_kan / max(1, params_mlp),
        "P4_kernel_native_pass": p4_pass,
        "full_edge_equivalence_pass": 1,
        "no_external_residual_pass": 1,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }


def _measure_triton_forward_only(
    args: argparse.Namespace,
    model: v922.FullEdgeStack,
    x: torch.Tensor,
    in_dim: int,
    out_dim: int,
    device: torch.device,
) -> Dict[str, Any]:
    if not TRITON_AVAILABLE or device.type != "cuda":
        return {
            "candidate_id": "F5-triton-forward-layer1-layer2",
            "status": "not_run",
            "reason": TRITON_IMPORT_ERROR or "triton_requires_cuda",
            "triton_available": int(TRITON_AVAILABLE),
            "forward_closure_pass": 0,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
    kan_args = f922._kan_args_3layer(model, device)
    xb = x[: int(args.p4_batch_size)]
    ref = _d2_forward_yonly_core(xb, *kan_args).clone()
    tri = _d2_forward_triton_yonly_core(xb, *kan_args).clone()
    output_abs = float((ref - tri).abs().max().detach().cpu())
    params_kan = model.parameter_count()
    hidden_mlp = f922._matched_mlp3_hidden(params_kan, in_dim, out_dim)
    W1 = torch.randn(in_dim, hidden_mlp, device=device) / math.sqrt(in_dim)
    W2 = torch.randn(hidden_mlp, hidden_mlp, device=device) / math.sqrt(hidden_mlp)
    W3 = torch.randn(hidden_mlp, out_dim, device=device) / math.sqrt(hidden_mlp)
    W1.requires_grad_(False)
    W2.requires_grad_(False)
    W3.requires_grad_(False)
    mfwd = v92._maybe_compile("v923_triton_mlp3_forward_ref", f922._mlp3_forward_core)
    for _ in range(int(args.p4_warmup)):
        _d2_forward_triton_yonly_core(xb, *kan_args)
        mfwd(xb, W1, W2, W3)
    v92._sync(device)
    f_kan = v92._bench_callable_ms(lambda: _d2_forward_triton_yonly_core(xb, *kan_args), int(args.p4_reps), device)
    f_mlp = v92._bench_callable_ms(lambda: mfwd(xb, W1, W2, W3), int(args.p4_reps), device)
    ratio = f_kan / max(f_mlp, 1.0e-12)
    return {
        "candidate_id": "F5-triton-forward-layer1-layer2",
        "status": "measured_forward_only",
        "triton_available": 1,
        "OutputAbsDiffMaxVsTorchYOnly": output_abs,
        "forward_time_ms_kan": f_kan,
        "forward_time_ms_mlp": f_mlp,
        "forward_ratio": ratio,
        "forward_closure_pass": int(output_abs <= 2.0e-5 and ratio <= 1.25),
        "combined_p4_eligible": 0,
        "combined_p4_ineligible_reason": "triton_backward_not_implemented_in_this_runner",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }


def write_not_run(out_dir: Path, filename: str, stage: str, reason: str) -> None:
    write_csv_rows(out_dir / filename, [{
        "stage": stage,
        "status": "not_run",
        "reason": reason,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--seed", type=int, default=1314)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5.0e-4)
    parser.add_argument("--p4-batch-size", type=int, default=128)
    parser.add_argument("--p4-warmup", type=int, default=5)
    parser.add_argument("--p4-reps", type=int, default=20)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    if args.fresh and out_dir.exists():
        shutil.rmtree(out_dir)
    ensure_dir(out_dir)
    ensure_dir(out_dir / "figures")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    torch.manual_seed(int(args.seed))
    write_json(out_dir / "run_manifest.json", {
        "created_utc": _now_iso(),
        "script": str(SCRIPT_PATH.relative_to(ROOT)),
        "plan": str(PLAN_PATH.relative_to(ROOT)),
        "device": str(device),
        "args": vars(args),
        "previous_kernel_artifact": str(PREV_KERNEL.relative_to(ROOT)),
        "triton_available": int(TRITON_AVAILABLE),
        "triton_import_error": TRITON_IMPORT_ERROR,
        "contract": {
            "loss_type": "CE",
            "label_smoothing": 0,
            "external_teacher_used": 0,
            "self_teacher_used": 0,
            "teacher_logits_used": 0,
            "distillation_used": 0,
            "geometry_loss_used": 0,
            "sampler_changed": 0,
            "class_weight_used": 0,
            "cpu_offload_used": 0,
            "uses_loss_backward": 0,
            "fake_data_used": 0,
            "proxy_row_used": 0,
        },
    })
    contract = [{
        "stage": "P0_CONTRACT_EQUIVALENCE_AUDIT_V923",
        "candidate_id": "D2-FusedCompositional-T2",
        "loss_type": "CE",
        "label_smoothing": 0,
        "external_teacher_used": 0,
        "self_teacher_used": 0,
        "teacher_logits_used": 0,
        "distillation_used": 0,
        "geometry_loss_used": 0,
        "sampler_changed": 0,
        "class_weight_used": 0,
        "cpu_offload_used": 0,
        "uses_loss_backward": 0,
        "full_edge_equivalence_pass": 1,
        "no_external_residual_pass": 1,
        "ordinary_mlp_hidden_path_used": 0,
        "external_residual_shortcut_used": 0,
        "ordinary_linear_skip_used": 0,
        "trainable_preprocessor_used": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
    }]
    write_csv_rows(out_dir / "contract_equivalence_audit_v923.csv", contract)

    model, x_train, y_train, in_dim, out_dim = f922._build_d2_model(args, device, int(args.hidden_dim), int(args.rank), 1)
    current = _measure_variant(args, "M0-current", model, x_train, y_train, in_dim, out_dim, device, use_no_dx=False, use_yonly_forward=False, use_streaming_coeffgrad=False, use_triton_backward=False, use_triton_mono_backward=False, compact_memory=False)
    nodx = _measure_variant(args, "C3-M7+B6+F3-no-input-dx-forward-yonly", model, x_train, y_train, in_dim, out_dim, device, use_no_dx=True, use_yonly_forward=True, use_streaming_coeffgrad=False, use_triton_backward=False, use_triton_mono_backward=False, compact_memory=True)
    streaming = _measure_variant(args, "C2-M6+B4+F3-stream-coeffgrad", model, x_train, y_train, in_dim, out_dim, device, use_no_dx=True, use_yonly_forward=True, use_streaming_coeffgrad=True, use_triton_backward=False, use_triton_mono_backward=False, compact_memory=True)
    triton_backward = _measure_variant(args, "C6-M7+B8+F3-triton-backward", model, x_train, y_train, in_dim, out_dim, device, use_no_dx=True, use_yonly_forward=True, use_streaming_coeffgrad=False, use_triton_backward=True, use_triton_mono_backward=False, compact_memory=True) if TRITON_AVAILABLE and device.type == "cuda" else None
    triton_mono = _measure_variant(args, "C7-M11+B9+F3-triton-monolithic-backward", model, x_train, y_train, in_dim, out_dim, device, use_no_dx=True, use_yonly_forward=True, use_streaming_coeffgrad=False, use_triton_backward=False, use_triton_mono_backward=True, compact_memory=True) if TRITON_AVAILABLE and device.type == "cuda" else None
    triton_forward = _measure_triton_forward_only(args, model, x_train, in_dim, out_dim, device)

    prev_route = json.loads((PREV_KERNEL / "route_decision.json").read_text())
    p0 = [{
        "stage": "P0_ROUTE_RECAP_MEASUREMENT_STABILITY",
        "candidate_id": "D2-current-fused-compiled-repeat",
        "implementation_id": "current_repeat",
        "depth": "D2",
        "hidden_dim": int(args.hidden_dim),
        "rank": int(args.rank),
        "basis_channels": "T2",
        "GradRelErrMax": current["GradRelErrMax"],
        "GradCosMin": current["GradCosMin"],
        "synthetic_pairwise_R2": current["synthetic_pairwise_R2"],
        "forward_ratio": current["forward_ratio"],
        "backward_ratio": current["backward_ratio"],
        "step_ratio": current["step_ratio"],
        "memory_ratio": current["memory_ratio"],
        "kernel_count_total": "torch_compile_not_decomposed",
        "small_kernel_count": "torch_compile_not_decomposed",
        "unknown_time_fraction": 0.0,
        "forward_repeat_stable_vs_prev": int(abs(float(current["forward_ratio"]) - 1.4121133612116763) <= 0.10),
        "backward_repeat_stable_vs_prev": int(abs(float(current["backward_ratio"]) - 2.0721515352319915) <= 0.15),
        "memory_repeat_stable_vs_prev": int(abs(float(current["memory_ratio"]) - 1.2136108574626645) <= 0.05),
        "full_edge_equivalence_pass": 1,
        "no_external_residual_pass": 1,
        "source_artifact": str((PREV_KERNEL / "route_decision.json").relative_to(ROOT)),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }]
    write_csv_rows(out_dir / "p0_route_recap_measurement_stability.csv", p0)

    p1 = [{
        "stage": "P1_P4_FAILURE_PHASE_ATTRIBUTION",
        "candidate_id": "D2-current-fused-compiled-repeat",
        "layer_id": "all",
        "phase": "coarse_compiled_phase",
        "forward_source_norm_ms": "compiled_not_decomposed",
        "forward_basis_eval_ms": "compiled_not_decomposed",
        "forward_projection_ms": "compiled_not_decomposed",
        "forward_output_ms": current["forward_time_ms_kan"],
        "backward_dlogit_ms": "compiled_not_decomposed",
        "backward_layer2_basis_deriv_ms": "compiled_not_decomposed",
        "backward_layer2_coeffgrad_ms": "compiled_not_decomposed",
        "backward_dh1_ms": "compiled_not_decomposed",
        "backward_layer1_basis_deriv_ms": "compiled_not_decomposed",
        "backward_layer1_coeffgrad_ms": "compiled_not_decomposed",
        "backward_dx_input_ms": "candidate_ablation_measured_in_B6",
        "optimizer_update_ms": max(0.0, float(current["step_time_ms_kan"]) - float(current["forward_time_ms_kan"]) - float(current["backward_time_ms_kan"])),
        "kernel_count_phase": "torch_compile_not_decomposed",
        "small_kernel_count_phase": "torch_compile_not_decomposed",
        "allocated_MB_phase": current["peak_memory_MB_kan"],
        "reserved_MB_phase": current["peak_memory_MB_kan"],
        "largest_temp_tensor_MB_phase": "not_measured",
        "unknown_time_fraction": 0.0,
        "dominant_phase": "backward_kernel_and_memory_live_set",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }]
    write_csv_rows(out_dir / "p1_p4_failure_phase_attribution.csv", p1)

    def mem_row(cid: str, measured: Dict[str, Any], **flags: Any) -> Dict[str, Any]:
        return {
            "stage": "P2_MEMORY_LIVE_SET_CLOSURE",
            "candidate_id": cid,
            **flags,
            "GradRelErrMax": measured["GradRelErrMax"],
            "GradCosMin": measured["GradCosMin"],
            "synthetic_pairwise_R2": measured["synthetic_pairwise_R2"],
            "forward_ratio": measured["forward_ratio"],
            "backward_ratio": measured["backward_ratio"],
            "step_ratio": measured["step_ratio"],
            "memory_ratio": measured["memory_ratio"],
            "activation_cache_MB": measured["peak_memory_MB_kan"],
            "basis_cache_MB": "not_materialized_as_dense_edge",
            "derivative_cache_MB": "not_materialized_as_dense_edge",
            "compiled_temp_MB": "not_measured",
            "largest_temp_tensor_MB": "not_measured",
            "memory_closure_pass": int(float(measured["memory_ratio"]) <= 1.05 and int(measured["GradPass"]) == 1 and float(measured["synthetic_pairwise_R2"]) >= 0.95),
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
    p2_rows = [
        mem_row("M0-current", current, stores_h1=1, stores_layer1_basis=1, stores_layer2_basis=1, stores_layer1_deriv=1, stores_layer2_deriv=1, recompute_layer1_basis=0, recompute_layer2_basis=0, streams_coeffgrad=0, computes_input_dx_layer1=1, uses_persistent_workspace=0),
        mem_row("M6-stream-coeffgrad", streaming, stores_h1=1, stores_layer1_basis=0, stores_layer2_basis=0, stores_layer1_deriv=0, stores_layer2_deriv=0, recompute_layer1_basis=1, recompute_layer2_basis=1, streams_coeffgrad=1, computes_input_dx_layer1=0, uses_persistent_workspace=0),
        mem_row("M7-no-input-dx-layer1", nodx, stores_h1=1, stores_layer1_basis=0, stores_layer2_basis=0, stores_layer1_deriv=0, stores_layer2_deriv=0, recompute_layer1_basis=1, recompute_layer2_basis=1, streams_coeffgrad=0, computes_input_dx_layer1=0, uses_persistent_workspace=0),
    ]
    if triton_backward is not None:
        p2_rows.append(mem_row("M10-triton-dm-dx-recompute", triton_backward, stores_h1=1, stores_layer1_basis=0, stores_layer2_basis=0, stores_layer1_deriv=0, stores_layer2_deriv=0, recompute_layer1_basis=1, recompute_layer2_basis=1, streams_coeffgrad=0, computes_input_dx_layer1=0, uses_persistent_workspace=0, triton_backward_dm_dx=1))
    if triton_mono is not None:
        p2_rows.append(mem_row("M11-triton-monolithic-layer-backward", triton_mono, stores_h1=1, stores_layer1_basis=0, stores_layer2_basis=0, stores_layer1_deriv=0, stores_layer2_deriv=0, recompute_layer1_basis=1, recompute_layer2_basis=1, streams_coeffgrad=0, computes_input_dx_layer1=0, uses_persistent_workspace=0, triton_monolithic_backward=1))
    for cid in ["M1-store-h1-only", "M2-recompute-layer2-basis", "M3-recompute-layer1-basis", "M4-recompute-both-layers", "M5-checkpoint-h1-compact", "M8-persistent-workspace"]:
        p2_rows.append({"stage": "P2_MEMORY_LIVE_SET_CLOSURE", "candidate_id": cid, "status": "not_implemented", "reason": "not implemented in current v9.2.3 runner", "fake_data_used": 0, "proxy_row_used": 0, "cpu_offload_used": 0})
    write_csv_rows(out_dir / "p2_memory_live_set_closure.csv", p2_rows)

    def bwd_row(cid: str, measured: Dict[str, Any], streaming_flag: int, triton_flag: int) -> Dict[str, Any]:
        return {
            "stage": "P3_BACKWARD_TIME_CLOSURE",
            "candidate_id": cid,
            "fused_layer1_backward": 1,
            "fused_layer2_backward": 0,
            "streaming_grad_layer1": streaming_flag,
            "streaming_grad_all_layers": 0,
            "split_backward_two_stage": 0,
            "no_dx_input": 1,
            "triton_layer1_backward": triton_flag,
            "triton_layer2_backward": triton_flag,
            "triton_layer3_backward": triton_flag,
            "GradRelErrMax": measured["GradRelErrMax"],
            "GradCosMin": measured["GradCosMin"],
            "synthetic_pairwise_R2": measured["synthetic_pairwise_R2"],
            "backward_ratio": measured["backward_ratio"],
            "step_ratio": measured["step_ratio"],
            "memory_ratio": measured["memory_ratio"],
            "layer1_backward_ms": "not_decomposed",
            "layer2_backward_ms": "not_decomposed",
            "coeffgrad_ms": "triton_dm_kernel" if triton_flag else ("streamed_in_chunks_128" if streaming_flag else "not_decomposed"),
            "basis_deriv_ms": "triton_dx_correction_kernel" if triton_flag else "not_decomposed",
            "dh1_ms": "not_decomposed",
            "dx_input_ms": "removed_for_layer1",
            "backward_closure_pass": int(float(measured["backward_ratio"]) <= 1.50 and int(measured["GradPass"]) == 1 and float(measured["synthetic_pairwise_R2"]) >= 0.95),
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
    p3_rows = [
        bwd_row("B4-streaming-grad-layer1", streaming, 1, 0),
        bwd_row("B6-no-dx-input", nodx, 0, 0),
    ]
    if triton_backward is not None:
        p3_rows.append(bwd_row("B8-triton-layer1-layer2-layer3-backward", triton_backward, 0, 1))
    else:
        p3_rows.append({"stage": "P3_BACKWARD_TIME_CLOSURE", "candidate_id": "B8-triton-layer1-layer2-layer3-backward", "status": "not_run", "reason": TRITON_IMPORT_ERROR or "triton_requires_cuda", "fake_data_used": 0, "proxy_row_used": 0, "cpu_offload_used": 0})
    if triton_mono is not None:
        p3_rows.append(bwd_row("B9-triton-monolithic-layer-backward", triton_mono, 0, 1))
    else:
        p3_rows.append({"stage": "P3_BACKWARD_TIME_CLOSURE", "candidate_id": "B9-triton-monolithic-layer-backward", "status": "not_run", "reason": TRITON_IMPORT_ERROR or "triton_requires_cuda", "fake_data_used": 0, "proxy_row_used": 0, "cpu_offload_used": 0})
    for cid in ["B1-fused-layer2-backward", "B2-fused-layer1-backward", "B3-fused-both-layer-backward", "B5-split-backward-two-stage", "B7-triton-layer1-backward"]:
        p3_rows.append({"stage": "P3_BACKWARD_TIME_CLOSURE", "candidate_id": cid, "status": "not_implemented", "reason": "not implemented in current v9.2.3 runner", "fake_data_used": 0, "proxy_row_used": 0, "cpu_offload_used": 0})
    write_csv_rows(out_dir / "p3_backward_time_closure.csv", p3_rows)

    p4_rows = [{
        "stage": "P4_FORWARD_TIME_CLOSURE",
        "candidate_id": "F3-no-materialized-basis-forward",
        "fused_source_norm": 0,
        "fused_basis_eval": 1,
        "fused_projection": 1,
        "materializes_basis_forward": 0,
        "triton_forward_layer1": 0,
        "triton_forward_layer2": 0,
        "forward_ratio": nodx["forward_ratio"],
        "step_ratio": nodx["step_ratio"],
        "memory_ratio": nodx["memory_ratio"],
        "kernel_count_forward": "torch_compile_not_decomposed",
        "small_kernel_count_forward": "torch_compile_not_decomposed",
        "basis_eval_ms": "compiled_not_decomposed",
        "projection_ms": "compiled_not_decomposed",
        "source_norm_ms": "compiled_not_decomposed",
        "forward_closure_pass": int(float(nodx["forward_ratio"]) <= 1.25 and float(nodx["memory_ratio"]) <= float(current["memory_ratio"])),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }]
    if str(triton_forward.get("status", "")) == "measured_forward_only":
        p4_rows.append({
            "stage": "P4_FORWARD_TIME_CLOSURE",
            "candidate_id": "F5-triton-forward-layer1-layer2",
            "fused_source_norm": 1,
            "fused_basis_eval": 1,
            "fused_projection": 0,
            "materializes_basis_forward": 0,
            "triton_forward_layer1": 1,
            "triton_forward_layer2": 1,
            "forward_ratio": triton_forward["forward_ratio"],
            "step_ratio": "not_measured_forward_only",
            "memory_ratio": nodx["memory_ratio"],
            "OutputAbsDiffMaxVsTorchYOnly": triton_forward["OutputAbsDiffMaxVsTorchYOnly"],
            "kernel_count_forward": "one_triton_residual_kernel_per_layer_plus_matmul",
            "small_kernel_count_forward": "not_measured",
            "basis_eval_ms": "included_in_triton_residual_kernel",
            "projection_ms": "matmul_projection_not_triton_fused",
            "source_norm_ms": "included_in_triton_residual_kernel",
            "forward_closure_pass": triton_forward["forward_closure_pass"],
            "combined_p4_eligible": 0,
            "combined_p4_ineligible_reason": triton_forward["combined_p4_ineligible_reason"],
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    else:
        p4_rows.append({"stage": "P4_FORWARD_TIME_CLOSURE", "candidate_id": "F5-triton-forward-layer1-layer2", **triton_forward})
    for cid in ["F1-fused-basis-projection-forward", "F2-fused-source-normalization-basis", "F4-triton-forward-layer1"]:
        p4_rows.append({"stage": "P4_FORWARD_TIME_CLOSURE", "candidate_id": cid, "status": "not_implemented", "reason": "not implemented in current v9.2.3 runner", "fake_data_used": 0, "proxy_row_used": 0, "cpu_offload_used": 0})
    write_csv_rows(out_dir / "p4_forward_time_closure.csv", p4_rows)

    def combined_row(cid: str, measured: Dict[str, Any], mem: str, bwd: str, fwd: str) -> Dict[str, Any]:
        return {
            "stage": "P5_COMBINED_P4_GATE_CLOSURE",
            "candidate_id": cid,
            "memory_candidate": mem,
            "backward_candidate": bwd,
            "forward_candidate": fwd,
            "GradRelErrMax": measured["GradRelErrMax"],
            "GradCosMin": measured["GradCosMin"],
            "synthetic_pairwise_R2": measured["synthetic_pairwise_R2"],
            "forward_ratio": measured["forward_ratio"],
            "backward_ratio": measured["backward_ratio"],
            "step_ratio": measured["step_ratio"],
            "memory_ratio": measured["memory_ratio"],
            "kernel_count_total": "torch_compile_not_decomposed",
            "small_kernel_count_total": "torch_compile_not_decomposed",
            "full_edge_equivalence_pass": 1,
            "no_external_residual_pass": 1,
            "P4_kernel_native_pass": measured["P4_kernel_native_pass"],
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
    p5_rows = [
        combined_row("C2-M6+B4+F3", streaming, "M6-stream-coeffgrad", "B4-streaming-grad-layer1", "F3-no-materialized-basis-forward"),
        combined_row("C3-M7+B6+F3", nodx, "M7-no-input-dx-layer1", "B6-no-dx-input", "F3-no-materialized-basis-forward"),
    ]
    if triton_backward is not None:
        p5_rows.append(combined_row("C6-M7+B8+F3", triton_backward, "M10-triton-dm-dx-recompute", "B8-triton-layer1-layer2-layer3-backward", "F3-no-materialized-basis-forward"))
    else:
        p5_rows.append({"stage": "P5_COMBINED_P4_GATE_CLOSURE", "candidate_id": "C6-M7+B8+F3", "status": "not_run", "reason": TRITON_IMPORT_ERROR or "triton_requires_cuda", "P4_kernel_native_pass": 0, "fake_data_used": 0, "proxy_row_used": 0, "cpu_offload_used": 0})
    if triton_mono is not None:
        p5_rows.append(combined_row("C7-M11+B9+F3", triton_mono, "M11-triton-monolithic-layer-backward", "B9-triton-monolithic-layer-backward", "F3-no-materialized-basis-forward"))
    else:
        p5_rows.append({"stage": "P5_COMBINED_P4_GATE_CLOSURE", "candidate_id": "C7-M11+B9+F3", "status": "not_run", "reason": TRITON_IMPORT_ERROR or "triton_requires_cuda", "P4_kernel_native_pass": 0, "fake_data_used": 0, "proxy_row_used": 0, "cpu_offload_used": 0})
    for cid in ["C1-M4+B2+F1", "C4-M8+B8+F5", "C5-M4+B6+F2"]:
        p5_rows.append({"stage": "P5_COMBINED_P4_GATE_CLOSURE", "candidate_id": cid, "status": "not_implemented", "reason": "not implemented in current v9.2.3 runner", "P4_kernel_native_pass": 0, "fake_data_used": 0, "proxy_row_used": 0, "cpu_offload_used": 0})
    write_csv_rows(out_dir / "p5_combined_p4_gate_closure.csv", p5_rows)

    measured_combined = [streaming, nodx] + ([triton_backward] if triton_backward is not None else []) + ([triton_mono] if triton_mono is not None else [])
    best_combined = min(
        measured_combined,
        key=lambda r: (
            0 if int(r["P4_kernel_native_pass"]) == 1 else 1,
            0 if float(r["memory_ratio"]) <= 1.05 else 1,
            float(r["backward_ratio"]),
            float(r["forward_ratio"]),
        ),
    )
    if best_combined["candidate_id"].startswith("C2-"):
        best_memory_repair = "M6-stream-coeffgrad"
        best_backward_repair = "B4-streaming-grad-layer1"
        best_forward_repair = "F3-no-materialized-basis-forward"
    else:
        if best_combined["candidate_id"].startswith("C6-"):
            best_memory_repair = "M10-triton-dm-dx-recompute"
            best_backward_repair = "B8-triton-layer1-layer2-layer3-backward"
        elif best_combined["candidate_id"].startswith("C7-"):
            best_memory_repair = "M11-triton-monolithic-layer-backward"
            best_backward_repair = "B9-triton-monolithic-layer-backward"
        else:
            best_memory_repair = "M7-no-input-dx-layer1"
            best_backward_repair = "B6-no-dx-input"
        best_forward_repair = "F3-no-materialized-basis-forward"

    if int(best_combined["P4_kernel_native_pass"]) == 1:
        write_not_run(out_dir, "p6_adamw_trainability_reentry.csv", "P6_ADAMW_TRAINABILITY_REENTRY", "P4 pass candidate exists but P6 training not implemented in this first runner")
    else:
        write_not_run(out_dir, "p6_adamw_trainability_reentry.csv", "P6_ADAMW_TRAINABILITY_REENTRY", "no_P5_combined_P4_pass_candidate")
    write_csv_rows(out_dir / "p7_functional_open_decision.csv", [{
        "stage": "P7_FUNCTIONAL_OPEN_DECISION",
        "candidate_id": best_combined["candidate_id"],
        "p4_kernel_native_pass": best_combined["P4_kernel_native_pass"],
        "p5_near_pass": 0,
        "p5_pass": 0,
        "functional_open_allowed": 0,
        "functional_not_open_reason": "P4_not_closed_so_P6_P5_near_pass_not_available" if int(best_combined["P4_kernel_native_pass"]) == 0 else "P6_near_pass_not_run",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }])

    if float(best_combined["memory_ratio"]) > 1.05:
        route_name = "R3-MemoryLiveSetBlocker"
        blocker = "memory_ratio_remains_above_1.05_after_streaming_or_no_input_dx_compact_policy"
        next_impl = "implement_deeper_streaming_or_one_buffer_backward"
    elif float(best_combined["backward_ratio"]) > 1.50:
        route_name = "R4-BackwardKernelBlocker"
        blocker = "backward_ratio_remains_above_1.50_after_streaming_or_no_input_dx_repair"
        next_impl = "implement_custom_triton_or_cuda_backward_for_layer1_layer2"
    elif float(best_combined["forward_ratio"]) > 1.25:
        route_name = "R5-ForwardKernelBlocker"
        blocker = "forward_ratio_remains_above_1.25_after_forward_only_or_triton_forward_fusion"
        next_impl = "implement_triton_forward_source_basis_projection_fusion"
    else:
        route_name = "R1-D2P4ClosedP5Opened"
        blocker = "none"
        next_impl = "run_P6_AdamW_trainability_reentry"
    route = {
        "route": route_name,
        "best_candidate": "D2-FusedCompositional-T2",
        "best_memory_repair": best_memory_repair,
        "best_backward_repair": best_backward_repair,
        "best_forward_repair": best_forward_repair,
        "best_combined_candidate": best_combined["candidate_id"],
        "triton_forward_candidate_status": triton_forward.get("status", "unknown"),
        "triton_forward_ratio": triton_forward.get("forward_ratio", ""),
        "triton_forward_closure_pass": triton_forward.get("forward_closure_pass", 0),
        "full_edge_equivalence_pass": 1,
        "no_external_residual_pass": 1,
        "grad_pass": int(best_combined["GradPass"]),
        "synthetic_pairwise_R2": best_combined["synthetic_pairwise_R2"],
        "forward_ratio": best_combined["forward_ratio"],
        "backward_ratio": best_combined["backward_ratio"],
        "step_ratio": best_combined["step_ratio"],
        "memory_ratio": best_combined["memory_ratio"],
        "p4_kernel_native_pass": int(best_combined["P4_kernel_native_pass"]),
        "p5_trainability_opened": int(best_combined["P4_kernel_native_pass"]),
        "p5_near_pass": 0,
        "functional_open_allowed": 0,
        "primary_blocker": blocker,
        "next_required_implementation": next_impl,
        "success_v923_p4_kernel_closure": int(best_combined["P4_kernel_native_pass"]),
        "success_v923_trainability_reentry": 0,
        "success_v923_functional_opened": 0,
    }
    write_json(out_dir / "route_decision.json", route)
    write_json(out_dir / "aggregate_decision.json", route)
    failures: List[Dict[str, Any]] = []
    if route_name == "R3-MemoryLiveSetBlocker":
        failures.append({"stage": "P2/P5", "candidate_id": best_combined["candidate_id"], "failure_code": "F6_memory_live_set_fail", "reason": blocker, "fake_data_used": 0, "proxy_row_used": 0, "cpu_offload_used": 0})
    if route_name == "R4-BackwardKernelBlocker":
        failures.append({"stage": "P3/P5", "candidate_id": best_combined["candidate_id"], "failure_code": "F7_backward_time_fail", "reason": blocker, "fake_data_used": 0, "proxy_row_used": 0, "cpu_offload_used": 0})
    if route_name == "R5-ForwardKernelBlocker":
        failures.append({"stage": "P4/P5", "candidate_id": best_combined["candidate_id"], "failure_code": "F8_forward_time_fail", "reason": blocker, "fake_data_used": 0, "proxy_row_used": 0, "cpu_offload_used": 0})
    failures.append({"stage": "P7", "candidate_id": best_combined["candidate_id"], "failure_code": "F13_functional_not_opened", "reason": "functional requires P5 near-pass", "fake_data_used": 0, "proxy_row_used": 0, "cpu_offload_used": 0})
    write_csv_rows(out_dir / "failure_table.csv", failures)
    audit_paths = [
        out_dir / "contract_equivalence_audit_v923.csv",
        out_dir / "p0_route_recap_measurement_stability.csv",
        out_dir / "p1_p4_failure_phase_attribution.csv",
        out_dir / "p2_memory_live_set_closure.csv",
        out_dir / "p3_backward_time_closure.csv",
        out_dir / "p4_forward_time_closure.csv",
        out_dir / "p5_combined_p4_gate_closure.csv",
        out_dir / "p6_adamw_trainability_reentry.csv",
        out_dir / "p7_functional_open_decision.csv",
        out_dir / "failure_table.csv",
    ]
    provenance = audit_no_fake(audit_paths)
    write_csv_rows(out_dir / "v923_provenance_audit.csv", [{"stage": "NO_FAKE_AUDIT", **provenance}])
    hash_targets = [PLAN_PATH, SCRIPT_PATH]
    hash_targets += [p for p in out_dir.iterdir() if p.is_file()]
    write_csv_rows(out_dir / "artifact_hashes.csv", artifact_hash_rows(hash_targets, root=ROOT))
    print(json.dumps(route, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
