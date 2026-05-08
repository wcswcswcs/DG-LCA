#!/usr/bin/env python3
"""DG-KAN v8.0 code-native teacher-free system-closure runner.

This runner keeps the audited v7.9/v7.6 measured path, but adds a V2 candidate
factory layer and two real no-external-teacher candidate paths:

* TF7: M13 architecture with final SWA weight averaging.
* TF8: M13 architecture with EMA weights, no logit teacher.
* SB0/SB1/SB2/SB3: no-external self-bootstrap with same-run EMA,
  delayed snapshot, best-validation snapshot, or dual-view teacher logits.
* SYS1: SB2 objective on the recompute/cache-trim fused stack, targeting the
  bs=512 memory margin without changing teacher-free status.
* SYS2: SYS1 with no-momentum SGD update to test whether optimizer-state
  memory is enough to close S2, without changing task objective or teacher use.
* SYS3: SYS1 with one-state RMS update to keep adaptive scaling while trimming
  first-moment optimizer memory.
* SYS4: SYS1 with AdamW moment states kept in half precision, testing whether
  S2 memory can close without removing AdamW-style reachability.
* SYS5: SYS1 with AdamW second moment kept in half precision while the first
  moment stays full precision, testing a gentler memory trim.
* SYS8: SYS1 with only the AdamW second moment kept in bfloat16 on GPU, testing
  a GPU-native low-memory state with better dynamic range than fp16.
* SYS9: SYS8 with the second-moment denominator kept in native bfloat16 state
  dtype until the final parameter update, testing whether the FP32 denom temp
  is part of the narrow S2 miss.
* SYS10: SYS1 with both AdamW moment states kept in bfloat16 on GPU, testing
  whether BF16 can close the optimizer-state memory gap without the fp16 task
  collapse observed by SYS4/SYS5.
* SYS11: SYS8 with addcdiv_ parameter updates, preserving the FP32 first
  moment and BF16 second moment while avoiding the extra FP32 step tensor temp.
* SYS12: SYS11 with only head first-moment state stored in BF16, preserving
  FP32 stack momentum while shaving the remaining optimizer-state memory edge.
* SYS13: SYS8 with a packed recompute stack: all stack mix weights share one
  flat tensor owner while hidden-y remains recomputed, targeting allocator and
  optimizer-loop overhead without changing the represented function family.
* SYS14: SYS8 with stack-only BF16 first/second AdamW state and FP32 head first
  moment, targeting the final S2 memory margin while protecting head dynamics.
* SYS15/SYS16: SYS8 with selective stack first-moment BF16 state, probing
  whether the tiny S2 memory margin can close without global optimizer damage.
* SYS17: SYS16 with addcdiv_ update, targeting the remaining step-only S2
  miss after selective first-layer state trimming.
* SYS18/SYS19: SYS17 system path with no-external self-bootstrap alpha 0.10
  or 0.50, testing whether the autonomous macro gap is an under/over-weighted
  same-run teacher signal rather than a capacity-only issue.
* SYS20/SYS21: SYS13 packed-recompute path with no-external self-bootstrap
  alpha 0.10 or 0.50, testing whether the packed S2 route can inherit the
  alpha repair without returning to external teachers.

Unsupported v8.0 candidates are recorded as not_implemented/not_run and never
counted as pass.  No fake/proxy rows are emitted.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = bool(torch.cuda.is_available())
except Exception:
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    _TRITON_AVAILABLE = False

import run_gafu_v72_real as v72
import run_gafu_v73_real as v73
import run_gafu_v76_real as v76
import run_gafu_v79_real as v79
from dgkan_core import get_device, parse_int_list, parse_str_list, save_json, set_seed, write_csv
from run_gafu_v63 import V63Params
from run_gafu_v66_real import _git_commit, _git_status


PLAN_PATH = "docs/DG-KAN_v8.0_FullStack_Reset_TeacherFree_CodeNative_SystemClosure_完整实验计划.md"
SCRIPT_PATH = "experiments/run_gafu_v80_real.py"
METRIC_UNAVAILABLE = v79.METRIC_UNAVAILABLE

_ORIG_V76_SPEC_MAP = v76._spec_map_v76
_ORIG_V76_REGISTRY = v76._candidate_registry_v76
_ORIG_V76_INIT_ID = v76._init_seed_candidate_id_v76
_ORIG_V76_USES_DISTILL = v76._uses_distill_v76
_ORIG_V76_MAKE_MANUAL = v76._make_manual_candidate_v76
_ORIG_V73_TRAIN_MANUAL = v73._train_manual_v73
_ORIG_V73_GRADIENT_CHECK = v73._gradient_check_v73

V80_WEIGHT_AVG_CE_IDS = {"TF7", "TF8"}
V80_REPRESENTATION_CE_IDS = {
    "RR3",
    "RR8",
    "RR9",
    "RR10",
    "RR11",
    "RR12",
    "RR13",
    "RR14",
}
V80_KERNEL_NATIVE_CE_IDS = {
    "KC1",
    "KC2",
    "KC3",
    "KC4",
    "KC5",
    "KC6",
    "KC7",
    "KC8",
    "KF1",
    "KF2",
    "KF3",
    "KF4",
    "KF5",
    "KF6",
    "KF7",
    "KF8",
    "KF9",
    "KF10",
    "KW1",
    "KW2",
    "KW3",
    "KW4",
    "KW5",
    "KW6",
}
V80_CE_OBJECTIVE_IDS = V80_WEIGHT_AVG_CE_IDS | V80_REPRESENTATION_CE_IDS | V80_KERNEL_NATIVE_CE_IDS
V80_SELF_BOOTSTRAP_IDS = {
    "SB0",
    "SB1",
    "SB2",
    "SB3",
    "SYS1",
    "SYS2",
    "SYS3",
    "SYS4",
    "SYS5",
    "SYS8",
    "SYS9",
    "SYS10",
    "SYS11",
    "SYS12",
    "SYS13",
    "SYS14",
    "SYS15",
    "SYS16",
    "SYS17",
    "SYS18",
    "SYS19",
    "SYS20",
    "SYS21",
    "SYS22",
    "SYS23",
    "SYS24",
}
V80_MANUAL_EXTENSION_IDS = V80_CE_OBJECTIVE_IDS | V80_SELF_BOOTSTRAP_IDS
V80_M5_INIT_IDS = {"M13"} | V80_MANUAL_EXTENSION_IDS
V80_BFLOAT_V_UPDATE_IDS = {
    "SYS13",
    "SYS20",
    "SYS21",
    "SYS22",
    "SYS23",
    "SYS24",
    "KC2",
    "KC3",
    "KC4",
    "KC5",
}
V80_ADAMW_ADDCDIV_UPDATE_IDS = {"KC7", "KC8", "KF2", "KF4", "KF6", "KF8", "KF10", "KW2", "KW3", "KW4", "KW5", "KW6"}
V80_RELEASE_HEAD_CACHE_IDS = {"KC8"}


def _rr_basis2_transform_impl(
    x: torch.Tensor,
    dw: torch.Tensor,
    centers: torch.Tensor,
    width: float,
    scale: float,
) -> torch.Tensor:
    denom = width * width
    diff0 = x - centers[0]
    diff1 = x - centers[1]
    b0 = torch.exp(-0.5 * diff0.square() / denom)
    b1 = torch.exp(-0.5 * diff1.square() / denom)
    return x + scale * (b0 * dw[:, 0].view(1, -1) + b1 * dw[:, 1].view(1, -1))


def _rr_basis2_backward_impl(
    x: torch.Tensor,
    z: torch.Tensor,
    dy: torch.Tensor,
    mix: torch.Tensor,
    dw: torch.Tensor,
    centers: torch.Tensor,
    width: float,
    scale: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    denom = width * width
    dz = dy @ mix
    grad_mix = dy.t() @ z
    diff0 = x - centers[0]
    diff1 = x - centers[1]
    b0 = torch.exp(-0.5 * diff0.square() / denom)
    b1 = torch.exp(-0.5 * diff1.square() / denom)
    grad_dw0 = (dz * scale * b0).sum(dim=0)
    grad_dw1 = (dz * scale * b1).sum(dim=0)
    grad_dw = torch.stack((grad_dw0, grad_dw1), dim=1)
    dres_dx = (
        b0 * (-diff0 / denom) * dw[:, 0].view(1, -1) +
        b1 * (-diff1 / denom) * dw[:, 1].view(1, -1)
    )
    dx = dz * (1.0 + scale * dres_dx)
    return dx, grad_mix, grad_dw


try:
    _RR_BASIS2_TRANSFORM_COMPILED = torch.compile(_rr_basis2_transform_impl, mode="reduce-overhead", fullgraph=True)  # type: ignore[attr-defined]
    _RR_BASIS2_BACKWARD_COMPILED = torch.compile(_rr_basis2_backward_impl, mode="reduce-overhead", fullgraph=True)  # type: ignore[attr-defined]
    _RR_BASIS2_COMPILE_AVAILABLE = True
except Exception:
    _RR_BASIS2_TRANSFORM_COMPILED = _rr_basis2_transform_impl
    _RR_BASIS2_BACKWARD_COMPILED = _rr_basis2_backward_impl
    _RR_BASIS2_COMPILE_AVAILABLE = False


if _TRITON_AVAILABLE:
    @triton.jit  # type: ignore[union-attr]
    def _rr_basis2_triton_forward_kernel(
        x,
        dw,
        centers,
        out,
        total: tl.constexpr,
        n_cols: tl.constexpr,
        width: tl.constexpr,
        scale: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < total
        cols = offsets % n_cols
        x_val = tl.load(x + offsets, mask=mask, other=0.0)
        c0 = tl.load(centers + 0)
        c1 = tl.load(centers + 1)
        dw0 = tl.load(dw + cols * 2 + 0, mask=mask, other=0.0)
        dw1 = tl.load(dw + cols * 2 + 1, mask=mask, other=0.0)
        denom = width * width
        diff0 = x_val - c0
        diff1 = x_val - c1
        b0 = tl.exp(-0.5 * diff0 * diff0 / denom)
        b1 = tl.exp(-0.5 * diff1 * diff1 / denom)
        tl.store(out + offsets, x_val + scale * (b0 * dw0 + b1 * dw1), mask=mask)

    @triton.jit  # type: ignore[union-attr]
    def _rr_basis2_triton_dx_kernel(
        x,
        dz,
        dw,
        centers,
        dx,
        total: tl.constexpr,
        n_cols: tl.constexpr,
        width: tl.constexpr,
        scale: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < total
        cols = offsets % n_cols
        x_val = tl.load(x + offsets, mask=mask, other=0.0)
        dz_val = tl.load(dz + offsets, mask=mask, other=0.0)
        c0 = tl.load(centers + 0)
        c1 = tl.load(centers + 1)
        dw0 = tl.load(dw + cols * 2 + 0, mask=mask, other=0.0)
        dw1 = tl.load(dw + cols * 2 + 1, mask=mask, other=0.0)
        denom = width * width
        diff0 = x_val - c0
        diff1 = x_val - c1
        b0 = tl.exp(-0.5 * diff0 * diff0 / denom)
        b1 = tl.exp(-0.5 * diff1 * diff1 / denom)
        dres_dx = b0 * (-diff0 / denom) * dw0 + b1 * (-diff1 / denom) * dw1
        tl.store(dx + offsets, dz_val * (1.0 + scale * dres_dx), mask=mask)

    @triton.jit  # type: ignore[union-attr]
    def _rr_basis2_triton_grad_dw_kernel(
        x,
        dz,
        centers,
        grad_dw,
        batch: tl.constexpr,
        n_cols: tl.constexpr,
        width: tl.constexpr,
        scale: tl.constexpr,
        BLOCK_B: tl.constexpr,
    ):
        col = tl.program_id(0)
        offs = tl.arange(0, BLOCK_B)
        mask = offs < batch
        x_val = tl.load(x + offs * n_cols + col, mask=mask, other=0.0)
        dz_val = tl.load(dz + offs * n_cols + col, mask=mask, other=0.0)
        c0 = tl.load(centers + 0)
        c1 = tl.load(centers + 1)
        denom = width * width
        diff0 = x_val - c0
        diff1 = x_val - c1
        b0 = tl.exp(-0.5 * diff0 * diff0 / denom)
        b1 = tl.exp(-0.5 * diff1 * diff1 / denom)
        g0 = tl.sum(dz_val * scale * b0, axis=0)
        g1 = tl.sum(dz_val * scale * b1, axis=0)
        tl.store(grad_dw + col * 2 + 0, g0)
        tl.store(grad_dw + col * 2 + 1, g1)

    @triton.jit  # type: ignore[union-attr]
    def _silu_backward_inplace_kernel(
        delta,
        y,
        total: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < total
        y_val = tl.load(y + offsets, mask=mask, other=0.0)
        d_val = tl.load(delta + offsets, mask=mask, other=0.0)
        sig = 1.0 / (1.0 + tl.exp(-y_val))
        grad = sig * (1.0 + y_val * (1.0 - sig))
        tl.store(delta + offsets, d_val * grad, mask=mask)


def _silu_backward_inplace(delta: torch.Tensor, y: torch.Tensor) -> bool:
    if (
        not _TRITON_AVAILABLE
        or not delta.is_cuda
        or not y.is_cuda
        or delta.dtype != torch.float32
        or y.dtype != torch.float32
        or not delta.is_contiguous()
        or not y.is_contiguous()
        or delta.numel() != y.numel()
    ):
        return False
    block = 256
    grid = (triton.cdiv(delta.numel(), block),)  # type: ignore[union-attr]
    _silu_backward_inplace_kernel[grid](delta, y, delta.numel(), BLOCK=block)  # type: ignore[index]
    return True


def _explicit_silu_backward_formula(delta: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    sig = torch.sigmoid(y)
    return delta * sig * (1.0 + y * (1.0 - sig))


_EXPLICIT_SILU_BACKWARD_COMPILED_AVAILABLE = False
_EXPLICIT_SILU_BACKWARD_COMPILED_FAILURES = 0
_EXPLICIT_SILU_BACKWARD_COMPILED = _explicit_silu_backward_formula
try:
    if hasattr(torch, "compile") and torch.cuda.is_available():
        _EXPLICIT_SILU_BACKWARD_COMPILED = torch.compile(
            _explicit_silu_backward_formula,
            fullgraph=True,
            mode="reduce-overhead",
        )
        _EXPLICIT_SILU_BACKWARD_COMPILED_AVAILABLE = True
except Exception:
    _EXPLICIT_SILU_BACKWARD_COMPILED = _explicit_silu_backward_formula
    _EXPLICIT_SILU_BACKWARD_COMPILED_AVAILABLE = False


def _compiled_explicit_silu_backward(delta: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, bool]:
    global _EXPLICIT_SILU_BACKWARD_COMPILED_FAILURES
    if _EXPLICIT_SILU_BACKWARD_COMPILED_AVAILABLE and delta.is_cuda and y.is_cuda:
        try:
            return _EXPLICIT_SILU_BACKWARD_COMPILED(delta, y), True
        except Exception:
            _EXPLICIT_SILU_BACKWARD_COMPILED_FAILURES += 1
    return _explicit_silu_backward_formula(delta, y), False


def _next_power_of_2_int(value: int) -> int:
    value = max(1, int(value))
    return 1 << (value - 1).bit_length()


@dataclass(frozen=True)
class CandidateSpecV2:
    candidate_id: str
    candidate_name: str
    route_role: str
    model_family: str
    dense_cls_name: str
    dense_cls_module_path: str
    hidden_dim: str
    basis_count: str
    depth: str
    init_policy: str
    optimizer_policy: str
    external_teacher_used: int
    external_teacher_source: str
    external_teacher_logits_used: int
    external_teacher_forward_used: int
    self_teacher_used: int
    self_teacher_type: str
    self_teacher_logits_used: int
    self_teacher_forward_used: int
    teacher_checkpoint_path: str
    teacher_artifact_hash: str
    no_external_teacher_eligible: int
    pure_supervised_eligible: int
    official_eligible: int
    expected_manual_forward: int
    expected_manual_backward: int
    expected_manual_update: int
    expected_nonkan_count: int
    implementation_status: str = "implemented"


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def _float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return default


def _mean(values: Iterable[Any], default: float = float("nan")) -> float:
    vals = [float(v) for v in values if _finite(v)]
    return sum(vals) / len(vals) if vals else default


def _is_one(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "1.0", "true"}


def _read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _hash_file(path: Path) -> str:
    return v72._hash_file(path) if path.exists() else ""


def _params(providers: Sequence[Any]) -> List[torch.Tensor]:
    return [p for provider in providers for _name, p, _g in provider.params_and_grads()]


def _restore_params(providers: Sequence[Any], values: Sequence[torch.Tensor]) -> None:
    with torch.no_grad():
        for p, value in zip(_params(providers), values):
            p.copy_(value.to(device=p.device, dtype=p.dtype))


def _clone_params(providers: Sequence[Any]) -> List[torch.Tensor]:
    return [p.detach().clone() for p in _params(providers)]


class PackedRecomputeFusedLinearSiluStack(v73.PackedFusedLinearSiluStack):
    """Packed parameter owner with v7.6 recompute-backward cache policy."""

    def forward_manual(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
        with torch.no_grad():
            h = x
            mixes = self._mixes()
            for i, mix in enumerate(mixes):
                y = h @ mix.t()
                h = F.silu(y) if i < len(mixes) - 1 else y
            return h, [{"x0": x.detach()}]

    def _activation_before(self, x0: torch.Tensor, layer_idx: int) -> torch.Tensor:
        h = x0
        mixes = self._mixes()
        for j in range(layer_idx):
            h = F.silu(h @ mixes[j].t())
        return h

    def backward_manual(self, dy: torch.Tensor, caches: List[Dict[str, Any]]) -> torch.Tensor:
        delta = dy
        x0 = caches[0]["x0"]
        mixes = self._mixes()
        with torch.no_grad():
            hs: List[torch.Tensor] = [x0]
            ys: List[torch.Tensor] = []
            h = x0
            for i, mix in enumerate(mixes):
                y = h @ mix.t()
                if i < len(mixes) - 1:
                    ys.append(y)
                    h = F.silu(y)
                    hs.append(h)

            for i in reversed(range(len(mixes))):
                x_i = hs[i]
                if i < len(mixes) - 1:
                    y_i = ys[i]
                    sig = torch.sigmoid(y_i)
                    delta = delta * sig * (1.0 + y_i * (1.0 - sig))
                start = self.starts[i]
                shape = self.shapes[i]
                end = start + int(shape[0] * shape[1])
                self.flat_grad[start:end].view(shape).add_(delta.t() @ x_i)
                delta = delta @ mixes[i]
            return delta

    def cache_breakdown(self, caches: Sequence[Dict[str, Any]]) -> Dict[str, float]:
        if not caches:
            x0_mb = 0.0
            count = 0
        else:
            x0 = caches[0]["x0"]
            x0_mb = int(x0.numel() * x0.element_size()) / (1024**2)
            count = 1
        return {
            "cache_total_MB": x0_mb,
            "cache_x_MB": x0_mb,
            "cache_hidden_y_MB": 0.0,
            "cache_hidden_MB": x0_mb,
            "cache_gate_MB": 0.0,
            "cache_basis_MB": 0.0,
            "materialized_tensor_count_measured": count,
            "largest_live_tensor_MB_measured": x0_mb,
            "linear_body_temp_MB": x0_mb,
            "hidden_y_cache_MB": 0.0,
            "manual_cache_MB_measured": x0_mb,
            "top1_memory_source_measured": "packed_recompute_stack_checkpoint_root_input",
            "top2_memory_source_measured": "packed_backward_recompute_hidden_temps",
            "top3_memory_source_measured": "optimizer_state",
            "unknown_memory_fraction_measured": 0.0,
        }


class PackedPrefixRecomputeFusedLinearSiluStack(PackedRecomputeFusedLinearSiluStack):
    """Packed recompute stack that avoids storing all hidden temps in backward."""

    def backward_manual(self, dy: torch.Tensor, caches: List[Dict[str, Any]]) -> torch.Tensor:
        delta = dy
        x0 = caches[0]["x0"]
        mixes = self._mixes()
        with torch.no_grad():
            for i in reversed(range(len(mixes))):
                x_i = self._activation_before(x0, i)
                if i < len(mixes) - 1:
                    y_i = x_i @ mixes[i].t()
                    sig = torch.sigmoid(y_i)
                    delta = delta * sig * (1.0 + y_i * (1.0 - sig))
                start = self.starts[i]
                shape = self.shapes[i]
                end = start + int(shape[0] * shape[1])
                self.flat_grad[start:end].view(shape).add_(delta.t() @ x_i)
                delta = delta @ mixes[i]
            return delta

    def cache_breakdown(self, caches: Sequence[Dict[str, Any]]) -> Dict[str, float]:
        out = super().cache_breakdown(caches)
        out["top1_memory_source_measured"] = "packed_prefix_recompute_stack_checkpoint_root_input"
        out["top2_memory_source_measured"] = "packed_prefix_backward_recompute_hidden_temps"
        return out


class PackedPrefixTritonSiluBackwardStack(PackedPrefixRecomputeFusedLinearSiluStack):
    """Prefix recompute stack with a real GPU fused SiLU derivative kernel.

    This keeps the same represented function as KC6.  The fusion is scoped to
    the backward activation derivative: the GEMM gradients still use PyTorch
    matmul kernels, while the sigmoid/multiply expression is emitted as one
    in-place Triton CUDA kernel when available.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.triton_silu_backward_calls = 0
        self.triton_silu_backward_fallbacks = 0

    def backward_manual(self, dy: torch.Tensor, caches: List[Dict[str, Any]]) -> torch.Tensor:
        delta = dy
        x0 = caches[0]["x0"]
        mixes = self._mixes()
        with torch.no_grad():
            for i in reversed(range(len(mixes))):
                x_i = self._activation_before(x0, i)
                if i < len(mixes) - 1:
                    y_i = x_i @ mixes[i].t()
                    if _silu_backward_inplace(delta, y_i):
                        self.triton_silu_backward_calls += 1
                    else:
                        self.triton_silu_backward_fallbacks += 1
                        sig = torch.sigmoid(y_i)
                        delta = delta * sig * (1.0 + y_i * (1.0 - sig))
                start = self.starts[i]
                shape = self.shapes[i]
                end = start + int(shape[0] * shape[1])
                self.flat_grad[start:end].view(shape).add_(delta.t() @ x_i)
                delta = delta @ mixes[i]
            return delta

    def cache_breakdown(self, caches: Sequence[Dict[str, Any]]) -> Dict[str, float]:
        out = super().cache_breakdown(caches)
        out["top2_memory_source_measured"] = "packed_prefix_triton_silu_backward_recompute_temps"
        out["triton_silu_backward_calls_measured"] = float(self.triton_silu_backward_calls)
        out["triton_silu_backward_fallbacks_measured"] = float(self.triton_silu_backward_fallbacks)
        out["triton_silu_backward_available"] = float(int(_TRITON_AVAILABLE))
        return out


class PackedPrefixAtenSiluBackwardStack(PackedPrefixRecomputeFusedLinearSiluStack):
    """Prefix recompute stack using ATen's fused GPU SiLU backward op."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.aten_silu_backward_calls = 0

    def backward_manual(self, dy: torch.Tensor, caches: List[Dict[str, Any]]) -> torch.Tensor:
        delta = dy
        x0 = caches[0]["x0"]
        mixes = self._mixes()
        silu_backward = torch.ops.aten.silu_backward.default
        with torch.no_grad():
            for i in reversed(range(len(mixes))):
                x_i = self._activation_before(x0, i)
                if i < len(mixes) - 1:
                    y_i = x_i @ mixes[i].t()
                    delta = silu_backward(delta, y_i)
                    self.aten_silu_backward_calls += 1
                start = self.starts[i]
                shape = self.shapes[i]
                end = start + int(shape[0] * shape[1])
                self.flat_grad[start:end].view(shape).add_(delta.t() @ x_i)
                delta = delta @ mixes[i]
            return delta

    def cache_breakdown(self, caches: Sequence[Dict[str, Any]]) -> Dict[str, float]:
        out = super().cache_breakdown(caches)
        out["top2_memory_source_measured"] = "packed_prefix_aten_silu_backward_recompute_temps"
        out["aten_silu_backward_calls_measured"] = float(self.aten_silu_backward_calls)
        return out


class PackedPrefixAtenUpperOnlySiluBackwardStack(PackedPrefixRecomputeFusedLinearSiluStack):
    """Prefix recompute stack with ATen SiLU backward only near the output."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.aten_silu_backward_calls = 0
        self.explicit_silu_backward_calls = 0

    def backward_manual(self, dy: torch.Tensor, caches: List[Dict[str, Any]]) -> torch.Tensor:
        delta = dy
        x0 = caches[0]["x0"]
        mixes = self._mixes()
        silu_backward = torch.ops.aten.silu_backward.default
        upper_nonlinear_idx = max(0, len(mixes) - 2)
        with torch.no_grad():
            for i in reversed(range(len(mixes))):
                x_i = self._activation_before(x0, i)
                if i < len(mixes) - 1:
                    y_i = x_i @ mixes[i].t()
                    if i == upper_nonlinear_idx:
                        delta = silu_backward(delta, y_i)
                        self.aten_silu_backward_calls += 1
                    else:
                        sig = torch.sigmoid(y_i)
                        delta = delta * sig * (1.0 + y_i * (1.0 - sig))
                        self.explicit_silu_backward_calls += 1
                start = self.starts[i]
                shape = self.shapes[i]
                end = start + int(shape[0] * shape[1])
                self.flat_grad[start:end].view(shape).add_(delta.t() @ x_i)
                delta = delta @ mixes[i]
            return delta

    def cache_breakdown(self, caches: Sequence[Dict[str, Any]]) -> Dict[str, float]:
        out = super().cache_breakdown(caches)
        out["top2_memory_source_measured"] = "packed_prefix_mixed_upper_aten_silu_backward_temps"
        out["aten_silu_backward_calls_measured"] = float(self.aten_silu_backward_calls)
        out["explicit_silu_backward_calls_measured"] = float(self.explicit_silu_backward_calls)
        return out


class PackedPrefixAtenLowerOnlySiluBackwardStack(PackedPrefixRecomputeFusedLinearSiluStack):
    """Prefix recompute stack with ATen SiLU backward only near the input."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.aten_silu_backward_calls = 0
        self.explicit_silu_backward_calls = 0

    def backward_manual(self, dy: torch.Tensor, caches: List[Dict[str, Any]]) -> torch.Tensor:
        delta = dy
        x0 = caches[0]["x0"]
        mixes = self._mixes()
        silu_backward = torch.ops.aten.silu_backward.default
        with torch.no_grad():
            for i in reversed(range(len(mixes))):
                x_i = self._activation_before(x0, i)
                if i < len(mixes) - 1:
                    y_i = x_i @ mixes[i].t()
                    if i == 0:
                        delta = silu_backward(delta, y_i)
                        self.aten_silu_backward_calls += 1
                    else:
                        sig = torch.sigmoid(y_i)
                        delta = delta * sig * (1.0 + y_i * (1.0 - sig))
                        self.explicit_silu_backward_calls += 1
                start = self.starts[i]
                shape = self.shapes[i]
                end = start + int(shape[0] * shape[1])
                self.flat_grad[start:end].view(shape).add_(delta.t() @ x_i)
                delta = delta @ mixes[i]
            return delta

    def cache_breakdown(self, caches: Sequence[Dict[str, Any]]) -> Dict[str, float]:
        out = super().cache_breakdown(caches)
        out["top2_memory_source_measured"] = "packed_prefix_mixed_lower_aten_silu_backward_temps"
        out["aten_silu_backward_calls_measured"] = float(self.aten_silu_backward_calls)
        out["explicit_silu_backward_calls_measured"] = float(self.explicit_silu_backward_calls)
        return out


class PackedPrefixNoInputGradRecomputeFusedLinearSiluStack(PackedPrefixRecomputeFusedLinearSiluStack):
    """KC6 path that skips materializing the unused input gradient tensor."""

    no_input_grad_materialized = True

    def backward_manual(self, dy: torch.Tensor, caches: List[Dict[str, Any]]) -> torch.Tensor:
        delta = dy
        x0 = caches[0]["x0"]
        mixes = self._mixes()
        with torch.no_grad():
            for i in reversed(range(len(mixes))):
                x_i = self._activation_before(x0, i)
                if i < len(mixes) - 1:
                    y_i = x_i @ mixes[i].t()
                    sig = torch.sigmoid(y_i)
                    delta = delta * sig * (1.0 + y_i * (1.0 - sig))
                start = self.starts[i]
                shape = self.shapes[i]
                end = start + int(shape[0] * shape[1])
                self.flat_grad[start:end].view(shape).add_(delta.t() @ x_i)
                if i == 0:
                    return delta.new_empty((0,))
                delta = delta @ mixes[i]
            return delta

    def cache_breakdown(self, caches: Sequence[Dict[str, Any]]) -> Dict[str, float]:
        out = super().cache_breakdown(caches)
        out["top2_memory_source_measured"] = "packed_prefix_backward_recompute_temps_no_input_grad"
        out["input_grad_materialized"] = 0.0
        return out


class PackedPrefixNoInputGradAtenLowerOnlySiluBackwardStack(PackedPrefixAtenLowerOnlySiluBackwardStack):
    """KF10 path that skips materializing the unused input gradient tensor."""

    no_input_grad_materialized = True

    def backward_manual(self, dy: torch.Tensor, caches: List[Dict[str, Any]]) -> torch.Tensor:
        delta = dy
        x0 = caches[0]["x0"]
        mixes = self._mixes()
        silu_backward = torch.ops.aten.silu_backward.default
        with torch.no_grad():
            for i in reversed(range(len(mixes))):
                x_i = self._activation_before(x0, i)
                if i < len(mixes) - 1:
                    y_i = x_i @ mixes[i].t()
                    if i == 0:
                        delta = silu_backward(delta, y_i)
                        self.aten_silu_backward_calls += 1
                    else:
                        sig = torch.sigmoid(y_i)
                        delta = delta * sig * (1.0 + y_i * (1.0 - sig))
                        self.explicit_silu_backward_calls += 1
                start = self.starts[i]
                shape = self.shapes[i]
                end = start + int(shape[0] * shape[1])
                self.flat_grad[start:end].view(shape).add_(delta.t() @ x_i)
                if i == 0:
                    return delta.new_empty((0,))
                delta = delta @ mixes[i]
            return delta

    def cache_breakdown(self, caches: Sequence[Dict[str, Any]]) -> Dict[str, float]:
        out = super().cache_breakdown(caches)
        out["top2_memory_source_measured"] = "packed_prefix_lower_aten_no_input_grad_backward_temps"
        out["input_grad_materialized"] = 0.0
        return out


class PackedPrefixWorkspaceCachedNoInputGradAtenLowerOnlySiluBackwardStack(PackedPrefixNoInputGradAtenLowerOnlySiluBackwardStack):
    """KW3 path with a short-lived GPU workspace for stack-backward activations."""

    workspace_cached_backward = True

    def backward_manual(self, dy: torch.Tensor, caches: List[Dict[str, Any]]) -> torch.Tensor:
        delta = dy
        x0 = caches[0]["x0"]
        mixes = self._mixes()
        silu_backward = torch.ops.aten.silu_backward.default
        with torch.no_grad():
            hs: List[torch.Tensor] = [x0]
            ys: List[torch.Tensor] = []
            h = x0
            for mix in mixes[:-1]:
                y = h @ mix.t()
                ys.append(y)
                h = F.silu(y)
                hs.append(h)

            for i in reversed(range(len(mixes))):
                x_i = hs[i]
                if i < len(mixes) - 1:
                    y_i = ys[i]
                    if i == 0:
                        delta = silu_backward(delta, y_i)
                        self.aten_silu_backward_calls += 1
                    else:
                        sig = torch.sigmoid(y_i)
                        delta = delta * sig * (1.0 + y_i * (1.0 - sig))
                        self.explicit_silu_backward_calls += 1
                start = self.starts[i]
                shape = self.shapes[i]
                end = start + int(shape[0] * shape[1])
                self.flat_grad[start:end].view(shape).add_(delta.t() @ x_i)
                if i == 0:
                    return delta.new_empty((0,))
                delta = delta @ mixes[i]
            return delta

    def cache_breakdown(self, caches: Sequence[Dict[str, Any]]) -> Dict[str, float]:
        out = super().cache_breakdown(caches)
        out["top2_memory_source_measured"] = "packed_prefix_lower_aten_no_input_grad_cached_workspace_temps"
        out["workspace_cached_backward"] = 1.0
        return out


class PackedPrefixCompiledExplicitNoInputGradAtenLowerOnlySiluBackwardStack(PackedPrefixNoInputGradAtenLowerOnlySiluBackwardStack):
    """KW3 recompute path with only the explicit derivative expression compiled.

    This intentionally preserves KW3's backward loop order and activation
    recomputation policy.  Unlike KW4, it does not cache hidden activations or
    y tensors across the stack backward pass; the only changed component is the
    mathematical SiLU derivative expression for upper hidden layers.
    """

    compiled_explicit_backward = True

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.compiled_explicit_silu_backward_calls = 0
        self.compiled_explicit_silu_backward_fallbacks = 0

    def backward_manual(self, dy: torch.Tensor, caches: List[Dict[str, Any]]) -> torch.Tensor:
        delta = dy
        x0 = caches[0]["x0"]
        mixes = self._mixes()
        silu_backward = torch.ops.aten.silu_backward.default
        with torch.no_grad():
            for i in reversed(range(len(mixes))):
                x_i = self._activation_before(x0, i)
                if i < len(mixes) - 1:
                    y_i = x_i @ mixes[i].t()
                    if i == 0:
                        delta = silu_backward(delta, y_i)
                        self.aten_silu_backward_calls += 1
                    else:
                        delta, used_compiled = _compiled_explicit_silu_backward(delta, y_i)
                        if used_compiled:
                            self.compiled_explicit_silu_backward_calls += 1
                        else:
                            self.compiled_explicit_silu_backward_fallbacks += 1
                        self.explicit_silu_backward_calls += 1
                start = self.starts[i]
                shape = self.shapes[i]
                end = start + int(shape[0] * shape[1])
                self.flat_grad[start:end].view(shape).add_(delta.t() @ x_i)
                if i == 0:
                    return delta.new_empty((0,))
                delta = delta @ mixes[i]
            return delta

    def cache_breakdown(self, caches: Sequence[Dict[str, Any]]) -> Dict[str, float]:
        out = super().cache_breakdown(caches)
        out["top2_memory_source_measured"] = "packed_prefix_lower_aten_no_input_grad_compiled_explicit_temps"
        out["compiled_explicit_backward"] = 1.0
        out["compiled_explicit_silu_backward_available"] = float(int(_EXPLICIT_SILU_BACKWARD_COMPILED_AVAILABLE))
        out["compiled_explicit_silu_backward_calls_measured"] = float(self.compiled_explicit_silu_backward_calls)
        out["compiled_explicit_silu_backward_fallbacks_measured"] = float(self.compiled_explicit_silu_backward_fallbacks)
        out["compiled_explicit_silu_backward_global_failures"] = float(_EXPLICIT_SILU_BACKWARD_COMPILED_FAILURES)
        return out


class PackedPrefixFastViewsNoInputGradAtenLowerOnlySiluBackwardStack(PackedPrefixNoInputGradAtenLowerOnlySiluBackwardStack):
    """KW3 recompute path with repeated packed-weight view construction removed."""

    fast_mix_views_backward = True

    @staticmethod
    def _activation_before_from_mixes(
        x0: torch.Tensor,
        layer_idx: int,
        mixes: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        h = x0
        for j in range(layer_idx):
            h = F.silu(h @ mixes[j].t())
        return h

    def backward_manual(self, dy: torch.Tensor, caches: List[Dict[str, Any]]) -> torch.Tensor:
        delta = dy
        x0 = caches[0]["x0"]
        mixes = self._mixes()
        silu_backward = torch.ops.aten.silu_backward.default
        with torch.no_grad():
            for i in reversed(range(len(mixes))):
                x_i = self._activation_before_from_mixes(x0, i, mixes)
                if i < len(mixes) - 1:
                    y_i = x_i @ mixes[i].t()
                    if i == 0:
                        delta = silu_backward(delta, y_i)
                        self.aten_silu_backward_calls += 1
                    else:
                        sig = torch.sigmoid(y_i)
                        delta = delta * sig * (1.0 + y_i * (1.0 - sig))
                        self.explicit_silu_backward_calls += 1
                start = self.starts[i]
                shape = self.shapes[i]
                end = start + int(shape[0] * shape[1])
                self.flat_grad[start:end].view(shape).add_(delta.t() @ x_i)
                if i == 0:
                    return delta.new_empty((0,))
                delta = delta @ mixes[i]
            return delta

    def cache_breakdown(self, caches: Sequence[Dict[str, Any]]) -> Dict[str, float]:
        out = super().cache_breakdown(caches)
        out["top2_memory_source_measured"] = "packed_prefix_lower_aten_no_input_grad_fast_mix_view_temps"
        out["fast_mix_views_backward"] = 1.0
        return out


class PackedPrefixExplicitInplaceSiluBackwardStack(PackedPrefixRecomputeFusedLinearSiluStack):
    """Prefix recompute stack with KC6's explicit SiLU derivative in-place.

    The derivative remains the same sigmoid formula as KC6, but the temporary
    live set is reduced by reusing the sigmoid tensor for the correction factor.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.explicit_inplace_silu_backward_calls = 0

    def backward_manual(self, dy: torch.Tensor, caches: List[Dict[str, Any]]) -> torch.Tensor:
        delta = dy
        x0 = caches[0]["x0"]
        mixes = self._mixes()
        with torch.no_grad():
            for i in reversed(range(len(mixes))):
                x_i = self._activation_before(x0, i)
                if i < len(mixes) - 1:
                    y_i = x_i @ mixes[i].t()
                    sig = torch.sigmoid(y_i)
                    delta.mul_(sig)
                    sig.neg_().add_(1.0).mul_(y_i).add_(1.0)
                    delta.mul_(sig)
                    self.explicit_inplace_silu_backward_calls += 1
                start = self.starts[i]
                shape = self.shapes[i]
                end = start + int(shape[0] * shape[1])
                self.flat_grad[start:end].view(shape).add_(delta.t() @ x_i)
                delta = delta @ mixes[i]
            return delta

    def cache_breakdown(self, caches: Sequence[Dict[str, Any]]) -> Dict[str, float]:
        out = super().cache_breakdown(caches)
        out["top2_memory_source_measured"] = "packed_prefix_explicit_inplace_silu_backward_temps"
        out["explicit_inplace_silu_backward_calls_measured"] = float(self.explicit_inplace_silu_backward_calls)
        return out


class PackedStableFusedLinearSiluStack(v73.PackedFusedLinearSiluStack):
    """Packed cached-y stack with v8.0 SiLU derivative rounding path."""

    def backward_manual(self, dy: torch.Tensor, caches: List[Dict[str, Any]]) -> torch.Tensor:
        delta = dy
        x0 = caches[0]["x0"]
        ys: Sequence[torch.Tensor] = caches[0].get("ys", [])
        mixes = self._mixes()
        with torch.no_grad():
            for i in reversed(range(len(mixes))):
                if i < len(mixes) - 1:
                    y_i = ys[i]
                    sig = torch.sigmoid(y_i)
                    delta = delta * sig * (1.0 + y_i * (1.0 - sig))
                x_i = x0 if i == 0 else F.silu(ys[i - 1])
                start = self.starts[i]
                shape = self.shapes[i]
                end = start + int(shape[0] * shape[1])
                self.flat_grad[start:end].view(shape).add_(delta.t() @ x_i)
                delta = delta @ mixes[i]
            return delta


class ManualDWM2LiteRBFLayer:
    """Code-native DWM2-lite RBF residual layer with analytic backward."""

    def __init__(self, in_dim: int, out_dim: int, basis_count: int, *, device: torch.device) -> None:
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.basis_count = int(max(2, basis_count))
        self.scale = 0.05
        centers = torch.linspace(-2.5, 2.5, self.basis_count, device=device)
        self.centers = centers
        self.width = float((centers[1] - centers[0]).abs().item() * 1.4) if self.basis_count > 1 else 1.0
        self.params = {
            "mix": torch.randn(out_dim, in_dim, device=device) / math.sqrt(max(1, in_dim)),
            "dw": torch.randn(in_dim, self.basis_count, device=device) * 0.02,
        }
        self.grads = {name: torch.zeros_like(param) for name, param in self.params.items()}

    def _basis(self, x: torch.Tensor) -> torch.Tensor:
        z = (x.unsqueeze(-1) - self.centers.view(1, 1, -1)) / self.width
        return torch.exp(-0.5 * z.square())

    def _transform(self, x: torch.Tensor, params: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        basis = self._basis(x)
        residual = (basis * params["dw"].unsqueeze(0)).sum(dim=-1)
        return x + self.scale * residual, basis

    def forward_with_params(self, x: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        z, _basis = self._transform(x, params)
        return z @ params["mix"].t()

    def forward_manual(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        with torch.no_grad():
            z, basis = self._transform(x, self.params)
            y = z @ self.params["mix"].t()
            return y, {"x": x.detach(), "z": z.detach(), "basis": basis.detach(), "y": y.detach()}

    def backward_manual(self, dy: torch.Tensor, cache: Dict[str, torch.Tensor]) -> torch.Tensor:
        with torch.no_grad():
            x = cache["x"]
            z = cache["z"]
            basis = cache["basis"]
            dz = dy @ self.params["mix"]
            self.grads["mix"].add_(dy.t() @ z)
            self.grads["dw"].add_((dz.unsqueeze(-1) * self.scale * basis).sum(dim=0))
            dbasis_dx = basis * (-(x.unsqueeze(-1) - self.centers.view(1, 1, -1)) / (self.width * self.width))
            dres_dx = (dbasis_dx * self.params["dw"].unsqueeze(0)).sum(dim=-1)
            return dz * (1.0 + self.scale * dres_dx)

    def zero_grad(self) -> None:
        for grad in self.grads.values():
            grad.zero_()

    def clone_params_for_autograd(self) -> Dict[str, torch.Tensor]:
        return {name: param.detach().clone().requires_grad_(True) for name, param in self.params.items()}

    def param_count(self) -> int:
        return sum(int(param.numel()) for param in self.params.values())

    def op_counts(self) -> Dict[str, int]:
        return {
            "op_count_gemm": 1,
            "op_count_elementwise": 5,
            "op_count_exp": 1,
            "op_count_pow": 1,
            "python_loop_count": 0,
            "group_loop_count": 0,
        }


class ManualDWM2LiteRBFStack:
    """Stack wrapper for RR3 code-native DWM2-lite RBF primitive."""

    def __init__(self, input_dim: int, hidden_dim: int, depth: int, basis_count: int, *, device: torch.device) -> None:
        dims = [int(input_dim)] + [int(hidden_dim)] * int(depth)
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        self.layers = [
            ManualDWM2LiteRBFLayer(a, b, basis_count, device=device)
            for a, b in zip(dims[:-1], dims[1:])
        ]

    def forward_manual(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]:
        with torch.no_grad():
            h = x
            caches: List[Dict[str, torch.Tensor]] = []
            for i, layer in enumerate(self.layers):
                y, cache = layer.forward_manual(h)
                caches.append(cache)
                h = F.silu(y) if i < len(self.layers) - 1 else y
            return h, caches

    def forward_autograd_with_params(self, x: torch.Tensor, params: Sequence[Dict[str, torch.Tensor]]) -> torch.Tensor:
        h = x
        for i, (layer, p) in enumerate(zip(self.layers, params)):
            y = layer.forward_with_params(h, p)
            h = F.silu(y) if i < len(self.layers) - 1 else y
        return h

    def backward_manual(self, dy: torch.Tensor, caches: List[Dict[str, torch.Tensor]]) -> torch.Tensor:
        delta = dy
        with torch.no_grad():
            for i in reversed(range(len(self.layers))):
                if i < len(self.layers) - 1:
                    y = caches[i]["y"]
                    sig = torch.sigmoid(y)
                    delta = delta * sig * (1.0 + y * (1.0 - sig))
                delta = self.layers[i].backward_manual(delta, caches[i])
            return delta

    def zero_grad(self) -> None:
        for layer in self.layers:
            layer.zero_grad()

    def params_and_grads(self) -> List[Tuple[str, torch.Tensor, torch.Tensor]]:
        rows: List[Tuple[str, torch.Tensor, torch.Tensor]] = []
        for li, layer in enumerate(self.layers):
            for name, param in layer.params.items():
                rows.append((f"rr3_dwm2lite_rbf:l{li}:{name}", param, layer.grads[name]))
        return rows

    def clone_params_for_autograd(self) -> List[Dict[str, torch.Tensor]]:
        return [layer.clone_params_for_autograd() for layer in self.layers]

    def param_count(self) -> int:
        return sum(layer.param_count() for layer in self.layers)

    def op_counts(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for layer in self.layers:
            for key, value in layer.op_counts().items():
                out[key] = out.get(key, 0) + int(value)
        return out

    def cache_breakdown(self, caches: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, float]:
        x_bytes = sum(int(c["x"].numel() * c["x"].element_size()) for c in caches)
        y_bytes = sum(int(c["y"].numel() * c["y"].element_size()) for c in caches[:-1])
        z_bytes = sum(int(c["z"].numel() * c["z"].element_size()) for c in caches)
        basis_bytes = sum(int(c["basis"].numel() * c["basis"].element_size()) for c in caches)
        total = x_bytes + y_bytes + z_bytes + basis_bytes
        largest = max(
            [int(c[k].numel() * c[k].element_size()) / (1024**2) for c in caches for k in ("x", "z", "basis")]
            + [int(c["y"].numel() * c["y"].element_size()) / (1024**2) for c in caches[:-1]]
            + [0.0]
        )
        return {
            "cache_total_MB": total / (1024**2),
            "cache_x_MB": x_bytes / (1024**2),
            "cache_hidden_y_MB": y_bytes / (1024**2),
            "cache_hidden_MB": (x_bytes + y_bytes + z_bytes) / (1024**2),
            "cache_gate_MB": 0.0,
            "cache_basis_MB": basis_bytes / (1024**2),
            "materialized_tensor_count_measured": len(caches) * 3 + max(0, len(caches) - 1),
            "largest_live_tensor_MB_measured": largest,
            "linear_body_temp_MB": z_bytes / (1024**2),
            "hidden_y_cache_MB": y_bytes / (1024**2),
            "manual_cache_MB_measured": total / (1024**2),
            "top1_memory_source_measured": "rr3_rbf_basis_cache",
            "top2_memory_source_measured": "rr3_transformed_z_cache",
            "top3_memory_source_measured": "optimizer_state",
            "unknown_memory_fraction_measured": 0.0,
        }


class ManualDWM2LiteRBFStreamingLayer(ManualDWM2LiteRBFLayer):
    """DWM2-lite RBF layer that streams basis terms instead of caching a 3-D basis tensor."""

    def _stream_transform(self, x: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        z = x.clone()
        denom = self.width * self.width
        for k in range(self.basis_count):
            diff = x - self.centers[k]
            basis_k = torch.exp(-0.5 * diff.square() / denom)
            z = z + self.scale * basis_k * params["dw"][:, k].view(1, -1)
        return z

    def forward_with_params(self, x: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        z = self._stream_transform(x, params)
        return z @ params["mix"].t()

    def forward_manual(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        with torch.no_grad():
            z = self._stream_transform(x, self.params)
            y = z @ self.params["mix"].t()
            return y, {"x": x.detach(), "z": z.detach(), "y": y.detach()}

    def backward_manual(self, dy: torch.Tensor, cache: Dict[str, torch.Tensor]) -> torch.Tensor:
        with torch.no_grad():
            x = cache["x"]
            z = cache["z"]
            dz = dy @ self.params["mix"]
            self.grads["mix"].add_(dy.t() @ z)
            dres_dx = torch.zeros_like(x)
            denom = self.width * self.width
            for k in range(self.basis_count):
                diff = x - self.centers[k]
                basis_k = torch.exp(-0.5 * diff.square() / denom)
                self.grads["dw"][:, k].add_((dz * self.scale * basis_k).sum(dim=0))
                dres_dx = dres_dx + basis_k * (-diff / denom) * self.params["dw"][:, k].view(1, -1)
            return dz * (1.0 + self.scale * dres_dx)

    def op_counts(self) -> Dict[str, int]:
        return {
            "op_count_gemm": 1,
            "op_count_elementwise": 5 * self.basis_count,
            "op_count_exp": self.basis_count,
            "op_count_pow": self.basis_count,
            "python_loop_count": self.basis_count,
            "group_loop_count": 0,
        }


class ManualDWM2LiteRBFStreamingStack(ManualDWM2LiteRBFStack):
    """Stack wrapper for streaming RR3/RR8 code-native DWM2-lite RBF primitive."""

    def __init__(self, input_dim: int, hidden_dim: int, depth: int, basis_count: int, *, device: torch.device) -> None:
        dims = [int(input_dim)] + [int(hidden_dim)] * int(depth)
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        self.layers = [
            ManualDWM2LiteRBFStreamingLayer(a, b, basis_count, device=device)
            for a, b in zip(dims[:-1], dims[1:])
        ]

    def cache_breakdown(self, caches: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, float]:
        x_bytes = sum(int(c["x"].numel() * c["x"].element_size()) for c in caches)
        y_bytes = sum(int(c["y"].numel() * c["y"].element_size()) for c in caches[:-1])
        z_bytes = sum(int(c["z"].numel() * c["z"].element_size()) for c in caches)
        total = x_bytes + y_bytes + z_bytes
        largest = max(
            [int(c[k].numel() * c[k].element_size()) / (1024**2) for c in caches for k in ("x", "z")]
            + [int(c["y"].numel() * c["y"].element_size()) / (1024**2) for c in caches[:-1]]
            + [0.0]
        )
        return {
            "cache_total_MB": total / (1024**2),
            "cache_x_MB": x_bytes / (1024**2),
            "cache_hidden_y_MB": y_bytes / (1024**2),
            "cache_hidden_MB": total / (1024**2),
            "cache_gate_MB": 0.0,
            "cache_basis_MB": 0.0,
            "materialized_tensor_count_measured": len(caches) * 2 + max(0, len(caches) - 1),
            "largest_live_tensor_MB_measured": largest,
            "linear_body_temp_MB": z_bytes / (1024**2),
            "hidden_y_cache_MB": y_bytes / (1024**2),
            "manual_cache_MB_measured": total / (1024**2),
            "top1_memory_source_measured": "rr8_streamed_z_cache",
            "top2_memory_source_measured": "rr8_backward_streamed_basis_temps",
            "top3_memory_source_measured": "optimizer_state",
            "unknown_memory_fraction_measured": 0.0,
        }


class ManualDWM2LiteRBF2Layer(ManualDWM2LiteRBFLayer):
    """Specialized basis-2 RBF residual layer with unrolled analytic backward."""

    def __init__(self, in_dim: int, out_dim: int, basis_count: int, *, device: torch.device) -> None:
        super().__init__(in_dim, out_dim, 2, device=device)

    def _basis2(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        denom = self.width * self.width
        diff0 = x - self.centers[0]
        diff1 = x - self.centers[1]
        b0 = torch.exp(-0.5 * diff0.square() / denom)
        b1 = torch.exp(-0.5 * diff1.square() / denom)
        return b0, b1, diff0, diff1

    def _transform2(self, x: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        b0, b1, _diff0, _diff1 = self._basis2(x)
        return x + self.scale * (
            b0 * params["dw"][:, 0].view(1, -1) +
            b1 * params["dw"][:, 1].view(1, -1)
        )

    def forward_with_params(self, x: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        z = self._transform2(x, params)
        return z @ params["mix"].t()

    def forward_manual(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        with torch.no_grad():
            z = self._transform2(x, self.params)
            y = z @ self.params["mix"].t()
            return y, {"x": x.detach(), "z": z.detach(), "y": y.detach()}

    def backward_manual(self, dy: torch.Tensor, cache: Dict[str, torch.Tensor]) -> torch.Tensor:
        with torch.no_grad():
            x = cache["x"]
            z = cache["z"]
            dz = dy @ self.params["mix"]
            self.grads["mix"].add_(dy.t() @ z)
            b0, b1, diff0, diff1 = self._basis2(x)
            self.grads["dw"][:, 0].add_((dz * self.scale * b0).sum(dim=0))
            self.grads["dw"][:, 1].add_((dz * self.scale * b1).sum(dim=0))
            denom = self.width * self.width
            dres_dx = (
                b0 * (-diff0 / denom) * self.params["dw"][:, 0].view(1, -1) +
                b1 * (-diff1 / denom) * self.params["dw"][:, 1].view(1, -1)
            )
            return dz * (1.0 + self.scale * dres_dx)

    def op_counts(self) -> Dict[str, int]:
        return {
            "op_count_gemm": 1,
            "op_count_elementwise": 10,
            "op_count_exp": 2,
            "op_count_pow": 2,
            "python_loop_count": 0,
            "group_loop_count": 0,
        }


class ManualDWM2LiteRBF2Stack(ManualDWM2LiteRBFStreamingStack):
    """Basis-2 unrolled stack for RR9."""

    def __init__(self, input_dim: int, hidden_dim: int, depth: int, basis_count: int, *, device: torch.device) -> None:
        dims = [int(input_dim)] + [int(hidden_dim)] * int(depth)
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        self.layers = [
            ManualDWM2LiteRBF2Layer(a, b, 2, device=device)
            for a, b in zip(dims[:-1], dims[1:])
        ]

    def cache_breakdown(self, caches: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, float]:
        out = super().cache_breakdown(caches)
        out["top1_memory_source_measured"] = "rr9_basis2_unrolled_z_cache"
        out["top2_memory_source_measured"] = "rr9_basis2_backward_unrolled_basis_temps"
        return out


class ManualDWM2LiteRBF2RecomputeZLayer(ManualDWM2LiteRBF2Layer):
    """Basis-2 layer that drops transformed-z cache and recomputes it in backward."""

    def forward_manual(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        with torch.no_grad():
            z = self._transform2(x, self.params)
            y = z @ self.params["mix"].t()
            return y, {"x": x.detach(), "y": y.detach()}

    def backward_manual(self, dy: torch.Tensor, cache: Dict[str, torch.Tensor]) -> torch.Tensor:
        with torch.no_grad():
            x = cache["x"]
            z = self._transform2(x, self.params)
            dz = dy @ self.params["mix"]
            self.grads["mix"].add_(dy.t() @ z)
            b0, b1, diff0, diff1 = self._basis2(x)
            self.grads["dw"][:, 0].add_((dz * self.scale * b0).sum(dim=0))
            self.grads["dw"][:, 1].add_((dz * self.scale * b1).sum(dim=0))
            denom = self.width * self.width
            dres_dx = (
                b0 * (-diff0 / denom) * self.params["dw"][:, 0].view(1, -1) +
                b1 * (-diff1 / denom) * self.params["dw"][:, 1].view(1, -1)
            )
            return dz * (1.0 + self.scale * dres_dx)


class ManualDWM2LiteRBF2RecomputeZStack(ManualDWM2LiteRBF2Stack):
    """Basis-2 unrolled stack that recomputes transformed z in backward."""

    def __init__(self, input_dim: int, hidden_dim: int, depth: int, basis_count: int, *, device: torch.device) -> None:
        dims = [int(input_dim)] + [int(hidden_dim)] * int(depth)
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        self.layers = [
            ManualDWM2LiteRBF2RecomputeZLayer(a, b, 2, device=device)
            for a, b in zip(dims[:-1], dims[1:])
        ]

    def cache_breakdown(self, caches: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, float]:
        x_bytes = sum(int(c["x"].numel() * c["x"].element_size()) for c in caches)
        y_bytes = sum(int(c["y"].numel() * c["y"].element_size()) for c in caches[:-1])
        total = x_bytes + y_bytes
        largest = max(
            [int(c["x"].numel() * c["x"].element_size()) / (1024**2) for c in caches]
            + [int(c["y"].numel() * c["y"].element_size()) / (1024**2) for c in caches[:-1]]
            + [0.0]
        )
        return {
            "cache_total_MB": total / (1024**2),
            "cache_x_MB": x_bytes / (1024**2),
            "cache_hidden_y_MB": y_bytes / (1024**2),
            "cache_hidden_MB": total / (1024**2),
            "cache_gate_MB": 0.0,
            "cache_basis_MB": 0.0,
            "materialized_tensor_count_measured": len(caches) + max(0, len(caches) - 1),
            "largest_live_tensor_MB_measured": largest,
            "linear_body_temp_MB": 0.0,
            "hidden_y_cache_MB": y_bytes / (1024**2),
            "manual_cache_MB_measured": total / (1024**2),
            "top1_memory_source_measured": "rr10_root_x_cache",
            "top2_memory_source_measured": "rr10_backward_recomputed_z_basis_temps",
            "top3_memory_source_measured": "optimizer_state",
            "unknown_memory_fraction_measured": 0.0,
        }


class ManualDWM2LiteRBF2CompiledLayer(ManualDWM2LiteRBF2Layer):
    """Basis-2 layer using torch.compile helpers for transform and backward math."""

    def forward_with_params(self, x: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        z = _RR_BASIS2_TRANSFORM_COMPILED(x, params["dw"], self.centers, self.width, self.scale)
        return z @ params["mix"].t()

    def forward_manual(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        with torch.no_grad():
            z = _RR_BASIS2_TRANSFORM_COMPILED(x, self.params["dw"], self.centers, self.width, self.scale)
            y = z @ self.params["mix"].t()
            return y, {"x": x.detach(), "z": z.detach().clone(), "y": y.detach().clone()}

    def backward_manual(self, dy: torch.Tensor, cache: Dict[str, torch.Tensor]) -> torch.Tensor:
        with torch.no_grad():
            dx, grad_mix, grad_dw = _RR_BASIS2_BACKWARD_COMPILED(
                cache["x"],
                cache["z"],
                dy,
                self.params["mix"],
                self.params["dw"],
                self.centers,
                self.width,
                self.scale,
            )
            self.grads["mix"].add_(grad_mix)
            self.grads["dw"].add_(grad_dw)
            return dx


class ManualDWM2LiteRBF2CompiledStack(ManualDWM2LiteRBF2Stack):
    """Basis-2 unrolled stack using torch.compile helpers for RR11."""

    def __init__(self, input_dim: int, hidden_dim: int, depth: int, basis_count: int, *, device: torch.device) -> None:
        dims = [int(input_dim)] + [int(hidden_dim)] * int(depth)
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        self.layers = [
            ManualDWM2LiteRBF2CompiledLayer(a, b, 2, device=device)
            for a, b in zip(dims[:-1], dims[1:])
        ]

    def cache_breakdown(self, caches: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, float]:
        out = super().cache_breakdown(caches)
        out["top1_memory_source_measured"] = "rr11_compiled_z_cache"
        out["top2_memory_source_measured"] = "rr11_compiled_backward_basis_temps"
        out["compile_available_measured"] = int(_RR_BASIS2_COMPILE_AVAILABLE)
        return out


class ManualDWM2LiteRBF2CompiledForwardLayer(ManualDWM2LiteRBF2Layer):
    """Basis-2 layer with compiled forward transform and eager verified backward."""

    def forward_with_params(self, x: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        z = _RR_BASIS2_TRANSFORM_COMPILED(x, params["dw"], self.centers, self.width, self.scale)
        return z @ params["mix"].t()

    def forward_manual(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        with torch.no_grad():
            z = _RR_BASIS2_TRANSFORM_COMPILED(x, self.params["dw"], self.centers, self.width, self.scale)
            y = z @ self.params["mix"].t()
            return y, {"x": x.detach(), "z": z.detach().clone(), "y": y.detach().clone()}


class ManualDWM2LiteRBF2CompiledForwardStack(ManualDWM2LiteRBF2Stack):
    """Basis-2 stack with compiled forward transform only for RR12."""

    def __init__(self, input_dim: int, hidden_dim: int, depth: int, basis_count: int, *, device: torch.device) -> None:
        dims = [int(input_dim)] + [int(hidden_dim)] * int(depth)
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        self.layers = [
            ManualDWM2LiteRBF2CompiledForwardLayer(a, b, 2, device=device)
            for a, b in zip(dims[:-1], dims[1:])
        ]

    def cache_breakdown(self, caches: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, float]:
        out = super().cache_breakdown(caches)
        out["top1_memory_source_measured"] = "rr12_compiled_forward_z_cache"
        out["top2_memory_source_measured"] = "rr12_eager_backward_basis_temps"
        out["compile_available_measured"] = int(_RR_BASIS2_COMPILE_AVAILABLE)
        return out


class ManualDWM2LiteRBF2TritonLayer(ManualDWM2LiteRBF2Layer):
    """Basis-2 layer using custom Triton transform and transform-backward kernels."""

    def _transform2_triton(self, x: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        if not (_TRITON_AVAILABLE and x.is_cuda and params["dw"].is_cuda):
            return self._transform2(x, params)
        out = torch.empty_like(x)
        total = int(x.numel())
        n_cols = int(x.shape[1])
        block = 256
        grid = (triton.cdiv(total, block),)  # type: ignore[union-attr]
        _rr_basis2_triton_forward_kernel[grid](
            x.contiguous(),
            params["dw"].contiguous(),
            self.centers,
            out,
            total,
            n_cols,
            self.width,
            self.scale,
            BLOCK=block,
        )
        return out

    def forward_with_params(self, x: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        z = self._transform2(x, params)
        return z @ params["mix"].t()

    def forward_manual(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        with torch.no_grad():
            z = self._transform2_triton(x, self.params)
            y = z @ self.params["mix"].t()
            return y, {"x": x.detach(), "z": z.detach().clone(), "y": y.detach().clone()}

    def backward_manual(self, dy: torch.Tensor, cache: Dict[str, torch.Tensor]) -> torch.Tensor:
        if not (_TRITON_AVAILABLE and dy.is_cuda):
            return super().backward_manual(dy, cache)
        with torch.no_grad():
            x = cache["x"].contiguous()
            z = cache["z"]
            dz = dy @ self.params["mix"]
            self.grads["mix"].add_(dy.t() @ z)
            batch = int(x.shape[0])
            n_cols = int(x.shape[1])
            block_b = min(_next_power_of_2_int(batch), 4096)
            grad_dw = torch.empty_like(self.grads["dw"])
            _rr_basis2_triton_grad_dw_kernel[(n_cols,)](
                x,
                dz.contiguous(),
                self.centers,
                grad_dw,
                batch,
                n_cols,
                self.width,
                self.scale,
                BLOCK_B=block_b,
            )
            self.grads["dw"].add_(grad_dw)
            dx = torch.empty_like(x)
            total = int(x.numel())
            block = 256
            grid = (triton.cdiv(total, block),)  # type: ignore[union-attr]
            _rr_basis2_triton_dx_kernel[grid](
                x,
                dz.contiguous(),
                self.params["dw"].contiguous(),
                self.centers,
                dx,
                total,
                n_cols,
                self.width,
                self.scale,
                BLOCK=block,
            )
            return dx


class ManualDWM2LiteRBF2TritonStack(ManualDWM2LiteRBF2Stack):
    """Basis-2 stack using Triton transform microkernels for RR13."""

    def __init__(self, input_dim: int, hidden_dim: int, depth: int, basis_count: int, *, device: torch.device) -> None:
        dims = [int(input_dim)] + [int(hidden_dim)] * int(depth)
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        self.layers = [
            ManualDWM2LiteRBF2TritonLayer(a, b, 2, device=device)
            for a, b in zip(dims[:-1], dims[1:])
        ]

    def cache_breakdown(self, caches: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, float]:
        out = super().cache_breakdown(caches)
        out["top1_memory_source_measured"] = "rr13_triton_z_cache"
        out["top2_memory_source_measured"] = "rr13_triton_transform_backward"
        out["triton_available_measured"] = int(_TRITON_AVAILABLE)
        return out


class ManualDWM2LiteRBF2TritonBackwardLayer(ManualDWM2LiteRBF2TritonLayer):
    """Basis-2 layer with eager forward and Triton transform-backward kernels."""

    def forward_manual(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        with torch.no_grad():
            z = self._transform2(x, self.params)
            y = z @ self.params["mix"].t()
            return y, {"x": x.detach(), "z": z.detach(), "y": y.detach()}


class ManualDWM2LiteRBF2TritonBackwardStack(ManualDWM2LiteRBF2TritonStack):
    """Basis-2 stack using Triton transform-backward kernels for RR14."""

    def __init__(self, input_dim: int, hidden_dim: int, depth: int, basis_count: int, *, device: torch.device) -> None:
        dims = [int(input_dim)] + [int(hidden_dim)] * int(depth)
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        self.layers = [
            ManualDWM2LiteRBF2TritonBackwardLayer(a, b, 2, device=device)
            for a, b in zip(dims[:-1], dims[1:])
        ]

    def cache_breakdown(self, caches: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, float]:
        out = super().cache_breakdown(caches)
        out["top1_memory_source_measured"] = "rr14_eager_z_cache"
        out["top2_memory_source_measured"] = "rr14_triton_transform_backward"
        return out


def _spec_map_v80() -> Dict[str, Any]:
    out = dict(_ORIG_V76_SPEC_MAP())
    out["TF7"] = v72.v71.CandidateSpec(
        "TF7",
        "M13-SWA-final-averaging-no-external-teacher",
        "v80_teacher_free_swa_weights",
        "v75_fused_linear_packed_head_kind",
        depth=3,
        stack_kind="linear",
        head_kind="poly2_silu_base",
    )
    out["TF8"] = v72.v71.CandidateSpec(
        "TF8",
        "M13-EMA-weights-no-logit-teacher",
        "v80_teacher_free_ema_weights",
        "v75_fused_linear_packed_head_kind",
        depth=3,
        stack_kind="linear",
        head_kind="poly2_silu_base",
    )
    out["RR3"] = v72.v71.CandidateSpec(
        "RR3",
        "M13-DWM2LiteDense-rbf-manual-code-native",
        "v80_representation_dwm2lite_rbf_manual",
        "v80_dwm2lite_rbf_manual_stack",
        depth=3,
        stack_kind="dwm2lite_rbf",
        head_kind="poly2_silu_base",
    )
    out["RR8"] = v72.v71.CandidateSpec(
        "RR8",
        "M13-DWM2LiteDense-rbf-streaming-manual-code-native",
        "v80_representation_dwm2lite_rbf_streaming_manual",
        "v80_dwm2lite_rbf_streaming_manual_stack",
        depth=3,
        stack_kind="dwm2lite_rbf_streaming",
        head_kind="poly2_silu_base",
    )
    out["RR9"] = v72.v71.CandidateSpec(
        "RR9",
        "M13-DWM2LiteDense-rbf-basis2-unrolled-manual-code-native",
        "v80_representation_dwm2lite_rbf_basis2_unrolled_manual",
        "v80_dwm2lite_rbf_basis2_unrolled_manual_stack",
        depth=3,
        stack_kind="dwm2lite_rbf_basis2_unrolled",
        head_kind="poly2_silu_base",
    )
    out["RR10"] = v72.v71.CandidateSpec(
        "RR10",
        "M13-DWM2LiteDense-rbf-basis2-recompute-z-manual-code-native",
        "v80_representation_dwm2lite_rbf_basis2_recompute_z_manual",
        "v80_dwm2lite_rbf_basis2_recompute_z_manual_stack",
        depth=3,
        stack_kind="dwm2lite_rbf_basis2_recompute_z",
        head_kind="poly2_silu_base",
    )
    out["RR11"] = v72.v71.CandidateSpec(
        "RR11",
        "M13-DWM2LiteDense-rbf-basis2-compiled-manual-code-native",
        "v80_representation_dwm2lite_rbf_basis2_compiled_manual",
        "v80_dwm2lite_rbf_basis2_compiled_manual_stack",
        depth=3,
        stack_kind="dwm2lite_rbf_basis2_compiled",
        head_kind="poly2_silu_base",
    )
    out["RR12"] = v72.v71.CandidateSpec(
        "RR12",
        "M13-DWM2LiteDense-rbf-basis2-compiled-forward-manual-code-native",
        "v80_representation_dwm2lite_rbf_basis2_compiled_forward_manual",
        "v80_dwm2lite_rbf_basis2_compiled_forward_manual_stack",
        depth=3,
        stack_kind="dwm2lite_rbf_basis2_compiled_forward",
        head_kind="poly2_silu_base",
    )
    out["RR13"] = v72.v71.CandidateSpec(
        "RR13",
        "M13-DWM2LiteDense-rbf-basis2-triton-transform-manual-code-native",
        "v80_representation_dwm2lite_rbf_basis2_triton_manual",
        "v80_dwm2lite_rbf_basis2_triton_manual_stack",
        depth=3,
        stack_kind="dwm2lite_rbf_basis2_triton",
        head_kind="poly2_silu_base",
    )
    out["RR14"] = v72.v71.CandidateSpec(
        "RR14",
        "M13-DWM2LiteDense-rbf-basis2-triton-backward-manual-code-native",
        "v80_representation_dwm2lite_rbf_basis2_triton_backward_manual",
        "v80_dwm2lite_rbf_basis2_triton_backward_manual_stack",
        depth=3,
        stack_kind="dwm2lite_rbf_basis2_triton_backward",
        head_kind="poly2_silu_base",
    )
    out["SB0"] = v72.v71.CandidateSpec(
        "SB0",
        "M13-EMA-self-bootstrap-T2-alpha025-no-external",
        "v80_self_bootstrap_ema",
        "v75_fused_linear_packed_head_kind",
        depth=3,
        stack_kind="linear",
        head_kind="poly2_silu_base",
    )
    out["SB1"] = v72.v71.CandidateSpec(
        "SB1",
        "M13-delayed-snapshot-self-bootstrap-T2-alpha025-no-external",
        "v80_self_bootstrap_snapshot",
        "v75_fused_linear_packed_head_kind",
        depth=3,
        stack_kind="linear",
        head_kind="poly2_silu_base",
    )
    out["SB2"] = v72.v71.CandidateSpec(
        "SB2",
        "M13-bestval-snapshot-self-bootstrap-T2-alpha025-no-external",
        "v80_self_bootstrap_bestval_snapshot",
        "v75_fused_linear_packed_head_kind",
        depth=3,
        stack_kind="linear",
        head_kind="poly2_silu_base",
    )
    out["SB3"] = v72.v71.CandidateSpec(
        "SB3",
        "M13-dual-view-consistency-self-bootstrap-T2-alpha025-no-external",
        "v80_self_bootstrap_dual_view",
        "v75_fused_linear_packed_head_kind",
        depth=3,
        stack_kind="linear",
        head_kind="poly2_silu_base",
    )
    out["SYS1"] = v72.v71.CandidateSpec(
        "SYS1",
        "SB2-recompute-cache-trim-bestval-self-bootstrap-no-external",
        "v80_system_recompute_cache_trim_bestval_self_bootstrap",
        "v76_recompute_fused_linear_packed_head_kind",
        depth=3,
        stack_kind="linear",
        head_kind="poly2_silu_base",
    )
    out["SYS2"] = v72.v71.CandidateSpec(
        "SYS2",
        "SYS1-SGD-state-trim-bestval-self-bootstrap-no-external",
        "v80_system_sgd_state_trim_bestval_self_bootstrap",
        "v76_recompute_fused_linear_packed_head_kind",
        depth=3,
        stack_kind="linear",
        head_kind="poly2_silu_base",
        lr_mult=3.0,
    )
    out["SYS3"] = v72.v71.CandidateSpec(
        "SYS3",
        "SYS1-RMS-one-state-trim-bestval-self-bootstrap-no-external",
        "v80_system_rms_one_state_trim_bestval_self_bootstrap",
        "v76_recompute_fused_linear_packed_head_kind",
        depth=3,
        stack_kind="linear",
        head_kind="poly2_silu_base",
    )
    out["SYS4"] = v72.v71.CandidateSpec(
        "SYS4",
        "SYS1-AdamWHalfState-trim-bestval-self-bootstrap-no-external",
        "v80_system_adamw_half_state_trim_bestval_self_bootstrap",
        "v76_recompute_fused_linear_packed_head_kind",
        depth=3,
        stack_kind="linear",
        head_kind="poly2_silu_base",
    )
    out["SYS5"] = v72.v71.CandidateSpec(
        "SYS5",
        "SYS1-AdamWVHalfState-trim-bestval-self-bootstrap-no-external",
        "v80_system_adamw_v_half_state_trim_bestval_self_bootstrap",
        "v76_recompute_fused_linear_packed_head_kind",
        depth=3,
        stack_kind="linear",
        head_kind="poly2_silu_base",
    )
    out["SYS8"] = v72.v71.CandidateSpec(
        "SYS8",
        "SYS1-AdamWVBF16State-trim-bestval-self-bootstrap-no-external",
        "v80_system_adamw_v_bfloat_state_trim_bestval_self_bootstrap",
        "v76_recompute_fused_linear_packed_head_kind",
        depth=3,
        stack_kind="linear",
        head_kind="poly2_silu_base",
    )
    out["SYS9"] = v72.v71.CandidateSpec(
        "SYS9",
        "SYS1-AdamWVBF16NativeDenomState-trim-bestval-self-bootstrap-no-external",
        "v80_system_adamw_v_bfloat_native_denom_state_trim_bestval_self_bootstrap",
        "v76_recompute_fused_linear_packed_head_kind",
        depth=3,
        stack_kind="linear",
        head_kind="poly2_silu_base",
    )
    out["SYS10"] = v72.v71.CandidateSpec(
        "SYS10",
        "SYS1-AdamWBF16State-trim-bestval-self-bootstrap-no-external",
        "v80_system_adamw_bfloat_state_trim_bestval_self_bootstrap",
        "v76_recompute_fused_linear_packed_head_kind",
        depth=3,
        stack_kind="linear",
        head_kind="poly2_silu_base",
    )
    out["SYS11"] = v72.v71.CandidateSpec(
        "SYS11",
        "SYS1-AdamWVBF16AddcdivUpdate-trim-bestval-self-bootstrap-no-external",
        "v80_system_adamw_v_bfloat_addcdiv_update_trim_bestval_self_bootstrap",
        "v76_recompute_fused_linear_packed_head_kind",
        depth=3,
        stack_kind="linear",
        head_kind="poly2_silu_base",
    )
    out["SYS12"] = v72.v71.CandidateSpec(
        "SYS12",
        "SYS1-AdamWHeadBF16VBF16AddcdivUpdate-trim-bestval-self-bootstrap-no-external",
        "v80_system_adamw_head_bfloat_v_bfloat_addcdiv_update_trim_bestval_self_bootstrap",
        "v76_recompute_fused_linear_packed_head_kind",
        depth=3,
        stack_kind="linear",
        head_kind="poly2_silu_base",
    )
    out["SYS13"] = v72.v71.CandidateSpec(
        "SYS13",
        "SYS1-PackedRecompute-AdamWVBF16State-bestval-self-bootstrap-no-external",
        "v80_system_packed_recompute_adamw_v_bfloat_state_bestval_self_bootstrap",
        "v80_packed_recompute_fused_linear_packed_head_kind",
        depth=3,
        stack_kind="linear",
        head_kind="poly2_silu_base",
    )
    out["SYS14"] = v72.v71.CandidateSpec(
        "SYS14",
        "SYS1-AdamWStackBF16VBF16AddcdivUpdate-trim-bestval-self-bootstrap-no-external",
        "v80_system_adamw_stack_bfloat_v_bfloat_addcdiv_update_trim_bestval_self_bootstrap",
        "v76_recompute_fused_linear_packed_head_kind",
        depth=3,
        stack_kind="linear",
        head_kind="poly2_silu_base",
    )
    out["SYS15"] = v72.v71.CandidateSpec(
        "SYS15",
        "SYS1-AdamWTailBF16VBF16State-trim-bestval-self-bootstrap-no-external",
        "v80_system_adamw_tail_bfloat_v_bfloat_state_trim_bestval_self_bootstrap",
        "v76_recompute_fused_linear_packed_head_kind",
        depth=3,
        stack_kind="linear",
        head_kind="poly2_silu_base",
    )
    out["SYS16"] = v72.v71.CandidateSpec(
        "SYS16",
        "SYS1-AdamWL0BF16VBF16State-trim-bestval-self-bootstrap-no-external",
        "v80_system_adamw_l0_bfloat_v_bfloat_state_trim_bestval_self_bootstrap",
        "v76_recompute_fused_linear_packed_head_kind",
        depth=3,
        stack_kind="linear",
        head_kind="poly2_silu_base",
    )
    out["SYS17"] = v72.v71.CandidateSpec(
        "SYS17",
        "SYS1-AdamWL0BF16VBF16AddcdivUpdate-trim-bestval-self-bootstrap-no-external",
        "v80_system_adamw_l0_bfloat_v_bfloat_addcdiv_update_trim_bestval_self_bootstrap",
        "v76_recompute_fused_linear_packed_head_kind",
        depth=3,
        stack_kind="linear",
        head_kind="poly2_silu_base",
    )
    out["SYS18"] = v72.v71.CandidateSpec(
        "SYS18",
        "SYS17-self-bootstrap-alpha010-no-external",
        "v80_system_adamw_l0_bfloat_v_bfloat_addcdiv_alpha010",
        "v76_recompute_fused_linear_packed_head_kind",
        depth=3,
        stack_kind="linear",
        head_kind="poly2_silu_base",
    )
    out["SYS19"] = v72.v71.CandidateSpec(
        "SYS19",
        "SYS17-self-bootstrap-alpha050-no-external",
        "v80_system_adamw_l0_bfloat_v_bfloat_addcdiv_alpha050",
        "v76_recompute_fused_linear_packed_head_kind",
        depth=3,
        stack_kind="linear",
        head_kind="poly2_silu_base",
    )
    out["SYS20"] = v72.v71.CandidateSpec(
        "SYS20",
        "SYS13-packed-self-bootstrap-alpha010-no-external",
        "v80_system_packed_recompute_adamw_v_bfloat_alpha010",
        "v80_packed_recompute_fused_linear_packed_head_kind",
        depth=3,
        stack_kind="linear",
        head_kind="poly2_silu_base",
    )
    out["SYS21"] = v72.v71.CandidateSpec(
        "SYS21",
        "SYS13-packed-self-bootstrap-alpha050-no-external",
        "v80_system_packed_recompute_adamw_v_bfloat_alpha050",
        "v80_packed_recompute_fused_linear_packed_head_kind",
        depth=3,
        stack_kind="linear",
        head_kind="poly2_silu_base",
    )
    out["SYS22"] = v72.v71.CandidateSpec(
        "SYS22",
        "SYS13-packed-cachey-self-bootstrap-alpha010-no-external",
        "v80_system_packed_cachey_adamw_v_bfloat_alpha010",
        "v80_packed_stable_fused_linear_packed_head_kind",
        depth=3,
        stack_kind="linear",
        head_kind="poly2_silu_base",
    )
    out["SYS23"] = v72.v71.CandidateSpec(
        "SYS23",
        "SYS13-packed-self-bootstrap-alpha005-no-external",
        "v80_system_packed_recompute_adamw_v_bfloat_alpha005",
        "v80_packed_recompute_fused_linear_packed_head_kind",
        depth=3,
        stack_kind="linear",
        head_kind="poly2_silu_base",
    )
    out["SYS24"] = v72.v71.CandidateSpec(
        "SYS24",
        "SYS13-packed-prefix-recompute-alpha005-no-external",
        "v80_system_packed_prefix_recompute_adamw_v_bfloat_alpha005",
        "v80_packed_prefix_recompute_fused_linear_packed_head_kind",
        depth=3,
        stack_kind="linear",
        head_kind="poly2_silu_base",
    )
    out["KC1"] = v72.v71.CandidateSpec(
        "KC1",
        "M13-CE-only-recompute-cache-trim",
        "v81_ceonly_recompute_cache_trim",
        "v76_recompute_fused_linear_packed_head_kind",
        depth=3,
        stack_kind="linear",
        head_kind="poly2_silu_base",
    )
    out["KC2"] = v72.v71.CandidateSpec(
        "KC2",
        "M13-CE-only-recompute-cache-trim-AdamWVBF16",
        "v81_ceonly_recompute_cache_trim_adamw_v_bfloat",
        "v76_recompute_fused_linear_packed_head_kind",
        depth=3,
        stack_kind="linear",
        head_kind="poly2_silu_base",
    )
    out["KC3"] = v72.v71.CandidateSpec(
        "KC3",
        "M13-CE-only-packed-recompute-AdamWVBF16",
        "v81_ceonly_packed_recompute_adamw_v_bfloat",
        "v80_packed_recompute_fused_linear_packed_head_kind",
        depth=3,
        stack_kind="linear",
        head_kind="poly2_silu_base",
    )
    out["KC4"] = v72.v71.CandidateSpec(
        "KC4",
        "M13-CE-only-packed-cache-y-AdamWVBF16",
        "v81_ceonly_packed_cache_y_adamw_v_bfloat",
        "v80_packed_stable_fused_linear_packed_head_kind",
        depth=3,
        stack_kind="linear",
        head_kind="poly2_silu_base",
    )
    out["KC5"] = v72.v71.CandidateSpec(
        "KC5",
        "M13-CE-only-packed-prefix-recompute-AdamWVBF16",
        "v81_ceonly_packed_prefix_recompute_adamw_v_bfloat",
        "v80_packed_prefix_recompute_fused_linear_packed_head_kind",
        depth=3,
        stack_kind="linear",
        head_kind="poly2_silu_base",
    )
    out["KC6"] = v72.v71.CandidateSpec(
        "KC6",
        "M13-CE-only-packed-prefix-recompute",
        "v81_ceonly_packed_prefix_recompute",
        "v80_packed_prefix_recompute_fused_linear_packed_head_kind",
        depth=3,
        stack_kind="linear",
        head_kind="poly2_silu_base",
    )
    out["KC7"] = v72.v71.CandidateSpec(
        "KC7",
        "M13-CE-only-packed-prefix-recompute-AdamWAddcdiv",
        "v81_ceonly_packed_prefix_recompute_adamw_addcdiv",
        "v80_packed_prefix_recompute_fused_linear_packed_head_kind",
        depth=3,
        stack_kind="linear",
        head_kind="poly2_silu_base",
    )
    out["KC8"] = v72.v71.CandidateSpec(
        "KC8",
        "M13-CE-only-packed-prefix-recompute-AdamWAddcdiv-HeadCacheRelease",
        "v81_ceonly_packed_prefix_recompute_adamw_addcdiv_head_cache_release",
        "v80_packed_prefix_recompute_fused_linear_packed_head_kind",
        depth=3,
        stack_kind="linear",
        head_kind="poly2_silu_base",
    )
    out["KF1"] = v72.v71.CandidateSpec(
        "KF1",
        "M13-CE-only-packed-prefix-TritonSiLUBackward",
        "v81_ceonly_packed_prefix_triton_silu_backward",
        "v80_packed_prefix_triton_silu_backward_fused_linear_packed_head_kind",
        depth=3,
        stack_kind="linear",
        head_kind="poly2_silu_base",
    )
    out["KF2"] = v72.v71.CandidateSpec(
        "KF2",
        "M13-CE-only-packed-prefix-TritonSiLUBackward-AdamWAddcdiv",
        "v81_ceonly_packed_prefix_triton_silu_backward_adamw_addcdiv",
        "v80_packed_prefix_triton_silu_backward_fused_linear_packed_head_kind",
        depth=3,
        stack_kind="linear",
        head_kind="poly2_silu_base",
    )
    out["KF3"] = v72.v71.CandidateSpec(
        "KF3",
        "M13-CE-only-packed-prefix-AtenSiLUBackward",
        "v81_ceonly_packed_prefix_aten_silu_backward",
        "v80_packed_prefix_aten_silu_backward_fused_linear_packed_head_kind",
        depth=3,
        stack_kind="linear",
        head_kind="poly2_silu_base",
    )
    out["KF4"] = v72.v71.CandidateSpec(
        "KF4",
        "M13-CE-only-packed-prefix-AtenSiLUBackward-AdamWAddcdiv",
        "v81_ceonly_packed_prefix_aten_silu_backward_adamw_addcdiv",
        "v80_packed_prefix_aten_silu_backward_fused_linear_packed_head_kind",
        depth=3,
        stack_kind="linear",
        head_kind="poly2_silu_base",
    )
    out["KF5"] = v72.v71.CandidateSpec(
        "KF5",
        "M13-CE-only-packed-prefix-ExplicitInplaceSiLUBackward",
        "v81_ceonly_packed_prefix_explicit_inplace_silu_backward",
        "v80_packed_prefix_explicit_inplace_silu_backward_fused_linear_packed_head_kind",
        depth=3,
        stack_kind="linear",
        head_kind="poly2_silu_base",
    )
    out["KF6"] = v72.v71.CandidateSpec(
        "KF6",
        "M13-CE-only-packed-prefix-ExplicitInplaceSiLUBackward-AdamWAddcdiv",
        "v81_ceonly_packed_prefix_explicit_inplace_silu_backward_adamw_addcdiv",
        "v80_packed_prefix_explicit_inplace_silu_backward_fused_linear_packed_head_kind",
        depth=3,
        stack_kind="linear",
        head_kind="poly2_silu_base",
    )
    out["KF7"] = v72.v71.CandidateSpec(
        "KF7",
        "M13-CE-only-packed-prefix-AtenUpperOnlySiLUBackward",
        "v82_ceonly_packed_prefix_aten_upper_only_silu_backward",
        "v80_packed_prefix_aten_upper_only_silu_backward_fused_linear_packed_head_kind",
        depth=3,
        stack_kind="linear",
        head_kind="poly2_silu_base",
    )
    out["KF8"] = v72.v71.CandidateSpec(
        "KF8",
        "M13-CE-only-packed-prefix-AtenUpperOnlySiLUBackward-AdamWAddcdiv",
        "v82_ceonly_packed_prefix_aten_upper_only_silu_backward_adamw_addcdiv",
        "v80_packed_prefix_aten_upper_only_silu_backward_fused_linear_packed_head_kind",
        depth=3,
        stack_kind="linear",
        head_kind="poly2_silu_base",
    )
    out["KF9"] = v72.v71.CandidateSpec(
        "KF9",
        "M13-CE-only-packed-prefix-AtenLowerOnlySiLUBackward",
        "v82_ceonly_packed_prefix_aten_lower_only_silu_backward",
        "v80_packed_prefix_aten_lower_only_silu_backward_fused_linear_packed_head_kind",
        depth=3,
        stack_kind="linear",
        head_kind="poly2_silu_base",
    )
    out["KF10"] = v72.v71.CandidateSpec(
        "KF10",
        "M13-CE-only-packed-prefix-AtenLowerOnlySiLUBackward-AdamWAddcdiv",
        "v82_ceonly_packed_prefix_aten_lower_only_silu_backward_adamw_addcdiv",
        "v80_packed_prefix_aten_lower_only_silu_backward_fused_linear_packed_head_kind",
        depth=3,
        stack_kind="linear",
        head_kind="poly2_silu_base",
    )
    out["KW1"] = v72.v71.CandidateSpec(
        "KW1",
        "M13-CE-only-packed-prefix-no-input-grad-recompute",
        "v82_ceonly_packed_prefix_no_input_grad_recompute",
        "v80_packed_prefix_no_input_grad_recompute_fused_linear_packed_head_kind",
        depth=3,
        stack_kind="linear",
        head_kind="poly2_silu_base",
    )
    out["KW2"] = v72.v71.CandidateSpec(
        "KW2",
        "M13-CE-only-packed-prefix-no-input-grad-recompute-AdamWAddcdiv",
        "v82_ceonly_packed_prefix_no_input_grad_recompute_adamw_addcdiv",
        "v80_packed_prefix_no_input_grad_recompute_fused_linear_packed_head_kind",
        depth=3,
        stack_kind="linear",
        head_kind="poly2_silu_base",
    )
    out["KW3"] = v72.v71.CandidateSpec(
        "KW3",
        "M13-CE-only-packed-prefix-no-input-grad-AtenLowerOnlySiLUBackward-AdamWAddcdiv",
        "v82_ceonly_packed_prefix_no_input_grad_aten_lower_only_silu_backward_adamw_addcdiv",
        "v80_packed_prefix_no_input_grad_aten_lower_only_silu_backward_fused_linear_packed_head_kind",
        depth=3,
        stack_kind="linear",
        head_kind="poly2_silu_base",
    )
    out["KW4"] = v72.v71.CandidateSpec(
        "KW4",
        "M13-CE-only-packed-prefix-cached-no-input-grad-AtenLowerOnlySiLUBackward-AdamWAddcdiv",
        "v82_ceonly_packed_prefix_cached_no_input_grad_aten_lower_only_silu_backward_adamw_addcdiv",
        "v80_packed_prefix_cached_no_input_grad_aten_lower_only_silu_backward_fused_linear_packed_head_kind",
        depth=3,
        stack_kind="linear",
        head_kind="poly2_silu_base",
    )
    out["KW5"] = v72.v71.CandidateSpec(
        "KW5",
        "M13-CE-only-packed-prefix-compiled-explicit-no-input-grad-AtenLowerOnlySiLUBackward-AdamWAddcdiv",
        "v82_ceonly_packed_prefix_compiled_explicit_no_input_grad_aten_lower_only_silu_backward_adamw_addcdiv",
        "v80_packed_prefix_compiled_explicit_no_input_grad_aten_lower_only_silu_backward_fused_linear_packed_head_kind",
        depth=3,
        stack_kind="linear",
        head_kind="poly2_silu_base",
    )
    out["KW6"] = v72.v71.CandidateSpec(
        "KW6",
        "M13-CE-only-packed-prefix-fast-mix-view-no-input-grad-AtenLowerOnlySiLUBackward-AdamWAddcdiv",
        "v82_ceonly_packed_prefix_fast_mix_view_no_input_grad_aten_lower_only_silu_backward_adamw_addcdiv",
        "v80_packed_prefix_fast_mix_view_no_input_grad_aten_lower_only_silu_backward_fused_linear_packed_head_kind",
        depth=3,
        stack_kind="linear",
        head_kind="poly2_silu_base",
    )
    return out


def _candidate_registry_v80() -> List[Any]:
    base = list(_ORIG_V76_REGISTRY())
    by_id = {str(spec.candidate_id): spec for spec in base}
    extra = _spec_map_v80()
    for cid in sorted(V80_MANUAL_EXTENSION_IDS):
        by_id[cid] = extra[cid]
    preferred = ["B0", "B2", "M12", "M13", "A2S", "M9", "TF7", "TF8", "RR3", "RR8", "RR9", "RR10", "RR11", "RR12", "RR13", "RR14", "KC1", "KC2", "KC3", "KC4", "KC5", "KC6", "KC7", "KC8", "KF1", "KF2", "KF3", "KF4", "KF5", "KF6", "KF7", "KF8", "KF9", "KF10", "KW1", "KW2", "KW3", "KW4", "KW5", "KW6", "SB0", "SB1", "SB2", "SB3", "SYS1", "SYS2", "SYS3", "SYS4", "SYS5", "SYS8", "SYS9", "SYS10", "SYS11", "SYS12", "SYS13", "SYS14", "SYS15", "SYS16", "SYS17", "SYS18", "SYS19", "SYS20", "SYS21", "SYS22", "SYS23", "SYS24"]
    out: List[Any] = []
    seen: set[str] = set()
    for cid in preferred:
        if cid in by_id:
            out.append(by_id[cid])
            seen.add(cid)
    for spec in base:
        cid = str(spec.candidate_id)
        if cid not in seen:
            out.append(spec)
            seen.add(cid)
    return out


def _init_seed_candidate_id_v80(spec: Any) -> str:
    if str(spec.candidate_id) in V80_MANUAL_EXTENSION_IDS:
        return "M4"
    return _ORIG_V76_INIT_ID(spec)


def _uses_distill_v80(spec: Any) -> bool:
    if str(spec.candidate_id) in V80_MANUAL_EXTENSION_IDS:
        return False
    return _ORIG_V76_USES_DISTILL(spec)


def _make_manual_candidate_v80(spec: Any, input_dim: int, num_classes: int, hidden_dim: int, basis: int, device: torch.device) -> Tuple[Any, Any]:
    if str(spec.stack_type) == "v80_dwm2lite_rbf_manual_stack":
        stack = ManualDWM2LiteRBFStack(input_dim, hidden_dim, spec.depth, basis, device=device)
        head = v73.PackedV63Poly2SiluHead(hidden_dim, num_classes, basis_count=basis, device=device)
    elif str(spec.stack_type) == "v80_dwm2lite_rbf_streaming_manual_stack":
        stack = ManualDWM2LiteRBFStreamingStack(input_dim, hidden_dim, spec.depth, basis, device=device)
        head = v73.PackedV63Poly2SiluHead(hidden_dim, num_classes, basis_count=basis, device=device)
    elif str(spec.stack_type) == "v80_dwm2lite_rbf_basis2_unrolled_manual_stack":
        stack = ManualDWM2LiteRBF2Stack(input_dim, hidden_dim, spec.depth, 2, device=device)
        head = v73.PackedV63Poly2SiluHead(hidden_dim, num_classes, basis_count=2, device=device)
    elif str(spec.stack_type) == "v80_dwm2lite_rbf_basis2_recompute_z_manual_stack":
        stack = ManualDWM2LiteRBF2RecomputeZStack(input_dim, hidden_dim, spec.depth, 2, device=device)
        head = v73.PackedV63Poly2SiluHead(hidden_dim, num_classes, basis_count=2, device=device)
    elif str(spec.stack_type) == "v80_dwm2lite_rbf_basis2_compiled_manual_stack":
        stack = ManualDWM2LiteRBF2CompiledStack(input_dim, hidden_dim, spec.depth, 2, device=device)
        head = v73.PackedV63Poly2SiluHead(hidden_dim, num_classes, basis_count=2, device=device)
    elif str(spec.stack_type) == "v80_dwm2lite_rbf_basis2_compiled_forward_manual_stack":
        stack = ManualDWM2LiteRBF2CompiledForwardStack(input_dim, hidden_dim, spec.depth, 2, device=device)
        head = v73.PackedV63Poly2SiluHead(hidden_dim, num_classes, basis_count=2, device=device)
    elif str(spec.stack_type) == "v80_dwm2lite_rbf_basis2_triton_manual_stack":
        stack = ManualDWM2LiteRBF2TritonStack(input_dim, hidden_dim, spec.depth, 2, device=device)
        head = v73.PackedV63Poly2SiluHead(hidden_dim, num_classes, basis_count=2, device=device)
    elif str(spec.stack_type) == "v80_dwm2lite_rbf_basis2_triton_backward_manual_stack":
        stack = ManualDWM2LiteRBF2TritonBackwardStack(input_dim, hidden_dim, spec.depth, 2, device=device)
        head = v73.PackedV63Poly2SiluHead(hidden_dim, num_classes, basis_count=2, device=device)
    elif str(spec.stack_type) == "v80_packed_recompute_fused_linear_packed_head_kind":
        stack = PackedRecomputeFusedLinearSiluStack(input_dim, hidden_dim, spec.depth, device=device)
        head = v73.PackedV63Poly2SiluHead(hidden_dim, num_classes, basis_count=basis, device=device)
    elif str(spec.stack_type) == "v80_packed_prefix_recompute_fused_linear_packed_head_kind":
        stack = PackedPrefixRecomputeFusedLinearSiluStack(input_dim, hidden_dim, spec.depth, device=device)
        head = v73.PackedV63Poly2SiluHead(hidden_dim, num_classes, basis_count=basis, device=device)
    elif str(spec.stack_type) == "v80_packed_prefix_triton_silu_backward_fused_linear_packed_head_kind":
        stack = PackedPrefixTritonSiluBackwardStack(input_dim, hidden_dim, spec.depth, device=device)
        head = v73.PackedV63Poly2SiluHead(hidden_dim, num_classes, basis_count=basis, device=device)
    elif str(spec.stack_type) == "v80_packed_prefix_aten_silu_backward_fused_linear_packed_head_kind":
        stack = PackedPrefixAtenSiluBackwardStack(input_dim, hidden_dim, spec.depth, device=device)
        head = v73.PackedV63Poly2SiluHead(hidden_dim, num_classes, basis_count=basis, device=device)
    elif str(spec.stack_type) == "v80_packed_prefix_aten_upper_only_silu_backward_fused_linear_packed_head_kind":
        stack = PackedPrefixAtenUpperOnlySiluBackwardStack(input_dim, hidden_dim, spec.depth, device=device)
        head = v73.PackedV63Poly2SiluHead(hidden_dim, num_classes, basis_count=basis, device=device)
    elif str(spec.stack_type) == "v80_packed_prefix_aten_lower_only_silu_backward_fused_linear_packed_head_kind":
        stack = PackedPrefixAtenLowerOnlySiluBackwardStack(input_dim, hidden_dim, spec.depth, device=device)
        head = v73.PackedV63Poly2SiluHead(hidden_dim, num_classes, basis_count=basis, device=device)
    elif str(spec.stack_type) == "v80_packed_prefix_no_input_grad_recompute_fused_linear_packed_head_kind":
        stack = PackedPrefixNoInputGradRecomputeFusedLinearSiluStack(input_dim, hidden_dim, spec.depth, device=device)
        head = v73.PackedV63Poly2SiluHead(hidden_dim, num_classes, basis_count=basis, device=device)
    elif str(spec.stack_type) == "v80_packed_prefix_no_input_grad_aten_lower_only_silu_backward_fused_linear_packed_head_kind":
        stack = PackedPrefixNoInputGradAtenLowerOnlySiluBackwardStack(input_dim, hidden_dim, spec.depth, device=device)
        head = v73.PackedV63Poly2SiluHead(hidden_dim, num_classes, basis_count=basis, device=device)
    elif str(spec.stack_type) == "v80_packed_prefix_cached_no_input_grad_aten_lower_only_silu_backward_fused_linear_packed_head_kind":
        stack = PackedPrefixWorkspaceCachedNoInputGradAtenLowerOnlySiluBackwardStack(input_dim, hidden_dim, spec.depth, device=device)
        head = v73.PackedV63Poly2SiluHead(hidden_dim, num_classes, basis_count=basis, device=device)
    elif str(spec.stack_type) == "v80_packed_prefix_compiled_explicit_no_input_grad_aten_lower_only_silu_backward_fused_linear_packed_head_kind":
        stack = PackedPrefixCompiledExplicitNoInputGradAtenLowerOnlySiluBackwardStack(input_dim, hidden_dim, spec.depth, device=device)
        head = v73.PackedV63Poly2SiluHead(hidden_dim, num_classes, basis_count=basis, device=device)
    elif str(spec.stack_type) == "v80_packed_prefix_fast_mix_view_no_input_grad_aten_lower_only_silu_backward_fused_linear_packed_head_kind":
        stack = PackedPrefixFastViewsNoInputGradAtenLowerOnlySiluBackwardStack(input_dim, hidden_dim, spec.depth, device=device)
        head = v73.PackedV63Poly2SiluHead(hidden_dim, num_classes, basis_count=basis, device=device)
    elif str(spec.stack_type) == "v80_packed_prefix_explicit_inplace_silu_backward_fused_linear_packed_head_kind":
        stack = PackedPrefixExplicitInplaceSiluBackwardStack(input_dim, hidden_dim, spec.depth, device=device)
        head = v73.PackedV63Poly2SiluHead(hidden_dim, num_classes, basis_count=basis, device=device)
    elif str(spec.stack_type) == "v80_packed_stable_fused_linear_packed_head_kind":
        stack = PackedStableFusedLinearSiluStack(input_dim, hidden_dim, spec.depth, device=device)
        head = v73.PackedV63Poly2SiluHead(hidden_dim, num_classes, basis_count=basis, device=device)
    else:
        stack, head = _ORIG_V76_MAKE_MANUAL(spec, input_dim, num_classes, hidden_dim, basis, device)
    if str(spec.candidate_id) in V80_RELEASE_HEAD_CACHE_IDS:
        setattr(stack, "release_head_cache_before_stack_backward", True)
    if str(spec.candidate_id) == "SYS2":
        setattr(stack, "update_mode", "sgd")
        setattr(head, "update_mode", "sgd")
    elif str(spec.candidate_id) == "SYS3":
        setattr(stack, "update_mode", "rms")
        setattr(head, "update_mode", "rms")
    elif str(spec.candidate_id) == "SYS4":
        setattr(stack, "update_mode", "adamw_half")
        setattr(head, "update_mode", "adamw_half")
    elif str(spec.candidate_id) == "SYS5":
        setattr(stack, "update_mode", "adamw_v_half")
        setattr(head, "update_mode", "adamw_v_half")
    elif str(spec.candidate_id) == "SYS8":
        setattr(stack, "update_mode", "adamw_v_bfloat")
        setattr(head, "update_mode", "adamw_v_bfloat")
    elif str(spec.candidate_id) == "SYS9":
        setattr(stack, "update_mode", "adamw_v_bfloat_native")
        setattr(head, "update_mode", "adamw_v_bfloat_native")
    elif str(spec.candidate_id) == "SYS10":
        setattr(stack, "update_mode", "adamw_bfloat")
        setattr(head, "update_mode", "adamw_bfloat")
    elif str(spec.candidate_id) == "SYS11":
        setattr(stack, "update_mode", "adamw_v_bfloat_addcdiv")
        setattr(head, "update_mode", "adamw_v_bfloat_addcdiv")
    elif str(spec.candidate_id) == "SYS12":
        setattr(stack, "update_mode", "adamw_head_bfloat_v_bfloat_addcdiv")
        setattr(head, "update_mode", "adamw_head_bfloat_v_bfloat_addcdiv")
    elif str(spec.candidate_id) in V80_BFLOAT_V_UPDATE_IDS:
        setattr(stack, "update_mode", "adamw_v_bfloat")
        setattr(head, "update_mode", "adamw_v_bfloat")
    elif str(spec.candidate_id) in V80_ADAMW_ADDCDIV_UPDATE_IDS:
        setattr(stack, "update_mode", "adamw_addcdiv")
        setattr(head, "update_mode", "adamw_addcdiv")
    elif str(spec.candidate_id) == "SYS14":
        setattr(stack, "update_mode", "adamw_stack_bfloat_v_bfloat_addcdiv")
        setattr(head, "update_mode", "adamw_stack_bfloat_v_bfloat_addcdiv")
    elif str(spec.candidate_id) == "SYS15":
        setattr(stack, "update_mode", "adamw_tail_bfloat_v_bfloat")
        setattr(head, "update_mode", "adamw_tail_bfloat_v_bfloat")
    elif str(spec.candidate_id) == "SYS16":
        setattr(stack, "update_mode", "adamw_l0_bfloat_v_bfloat")
        setattr(head, "update_mode", "adamw_l0_bfloat_v_bfloat")
    elif str(spec.candidate_id) in {"SYS17", "SYS18", "SYS19"}:
        setattr(stack, "update_mode", "adamw_l0_bfloat_v_bfloat_addcdiv")
        setattr(head, "update_mode", "adamw_l0_bfloat_v_bfloat_addcdiv")
    return stack, head


def _train_manual_v80(
    args: argparse.Namespace,
    spec: Any,
    dataset: str,
    seed: int,
    baseline_classwise_acc: Sequence[float] | None = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    cid = str(spec.candidate_id)
    self_bootstrap_ids = V80_SELF_BOOTSTRAP_IDS
    v80_ids = V80_MANUAL_EXTENSION_IDS
    if cid not in v80_ids:
        return _ORIG_V73_TRAIN_MANUAL(args, spec, dataset, seed, baseline_classwise_acc)

    device = get_device(args.device)
    params = V63Params(
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        batch_size=args.batch_size,
    )
    bundle = v72.v71.load_vision_bundle(
        dataset,
        data_root=Path(args.data_root),
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=seed,
        allow_fake_data=False,
    )
    x_train = bundle.x_train.to(device)
    y_train = bundle.y_train.to(device)
    x_val = bundle.x_val.to(device)
    y_val = bundle.y_val.to(device)
    x_test = bundle.x_test.to(device)
    y_test = bundle.y_test.to(device)

    set_seed(v72._stable_seed("v72-train", dataset, seed, _init_seed_candidate_id_v80(spec), spec.head_kind, spec.group_count, spec.shuffle))
    stack, head = v73._make_manual_candidate(spec, bundle.input_dim, bundle.num_classes, args.hidden_dim, args.basis_count, device)
    if cid in V80_RELEASE_HEAD_CACHE_IDS:
        setattr(stack, "release_head_cache_before_stack_backward", True)
    if cid == "SYS2":
        setattr(stack, "update_mode", "sgd")
        setattr(head, "update_mode", "sgd")
    elif cid == "SYS3":
        setattr(stack, "update_mode", "rms")
        setattr(head, "update_mode", "rms")
    elif cid == "SYS4":
        setattr(stack, "update_mode", "adamw_half")
        setattr(head, "update_mode", "adamw_half")
    elif cid == "SYS5":
        setattr(stack, "update_mode", "adamw_v_half")
        setattr(head, "update_mode", "adamw_v_half")
    elif cid == "SYS8":
        setattr(stack, "update_mode", "adamw_v_bfloat")
        setattr(head, "update_mode", "adamw_v_bfloat")
    elif cid == "SYS9":
        setattr(stack, "update_mode", "adamw_v_bfloat_native")
        setattr(head, "update_mode", "adamw_v_bfloat_native")
    elif cid == "SYS10":
        setattr(stack, "update_mode", "adamw_bfloat")
        setattr(head, "update_mode", "adamw_bfloat")
    elif cid == "SYS11":
        setattr(stack, "update_mode", "adamw_v_bfloat_addcdiv")
        setattr(head, "update_mode", "adamw_v_bfloat_addcdiv")
    elif cid == "SYS12":
        setattr(stack, "update_mode", "adamw_head_bfloat_v_bfloat_addcdiv")
        setattr(head, "update_mode", "adamw_head_bfloat_v_bfloat_addcdiv")
    elif cid in V80_BFLOAT_V_UPDATE_IDS:
        setattr(stack, "update_mode", "adamw_v_bfloat")
        setattr(head, "update_mode", "adamw_v_bfloat")
    elif cid in V80_ADAMW_ADDCDIV_UPDATE_IDS:
        setattr(stack, "update_mode", "adamw_addcdiv")
        setattr(head, "update_mode", "adamw_addcdiv")
    elif cid == "SYS14":
        setattr(stack, "update_mode", "adamw_stack_bfloat_v_bfloat_addcdiv")
        setattr(head, "update_mode", "adamw_stack_bfloat_v_bfloat_addcdiv")
    elif cid == "SYS15":
        setattr(stack, "update_mode", "adamw_tail_bfloat_v_bfloat")
        setattr(head, "update_mode", "adamw_tail_bfloat_v_bfloat")
    elif cid == "SYS16":
        setattr(stack, "update_mode", "adamw_l0_bfloat_v_bfloat")
        setattr(head, "update_mode", "adamw_l0_bfloat_v_bfloat")
    elif cid in {"SYS17", "SYS18", "SYS19"}:
        setattr(stack, "update_mode", "adamw_l0_bfloat_v_bfloat_addcdiv")
        setattr(head, "update_mode", "adamw_l0_bfloat_v_bfloat_addcdiv")
    opt = v72.v71.FastAdamW([stack, head], lr=params.lr_manual * spec.lr_mult, weight_decay=args.weight_decay)
    class_weights = torch.ones(bundle.num_classes, device=device)
    train_eval_x = x_train[: min(max(args.batch_size, 512), x_train.shape[0])]
    train_eval_y = y_train[: train_eval_x.shape[0]]
    train0 = v72.v71._manual_eval(stack, head, train_eval_x, train_eval_y, args.eval_batch_size, bundle.num_classes)
    val0 = v72.v71._manual_eval(stack, head, x_val, y_val, args.eval_batch_size, bundle.num_classes)

    trace: List[Dict[str, Any]] = []
    first_loss_before = float("nan")
    first_loss_after = float("nan")
    started = time.perf_counter()
    providers = [stack, head]
    ema_values: List[torch.Tensor] | None = None
    snapshot_values: List[torch.Tensor] | None = None
    bestval_snapshot_values: List[torch.Tensor] | None = None
    bestval_snapshot_score = -float("inf")
    swa_sum: List[torch.Tensor] | None = None
    swa_count = 0
    swa_start = max(1, int(0.60 * int(args.task_steps)))
    ema_decay = 0.995
    teacher_alpha = 0.05 if cid in {"SYS23", "SYS24"} else 0.10 if cid in {"SYS18", "SYS20", "SYS22"} else 0.50 if cid in {"SYS19", "SYS21"} else 0.25
    teacher_temperature = 2.0
    dual_view_noise_std = 0.03
    dual_gen = torch.Generator(device=device)
    dual_gen.manual_seed(v72._stable_seed("v80-dual-view", dataset, seed, cid))
    self_teacher_active_steps = 0
    first_self_kl = float("nan")

    for step in range(1, int(args.task_steps) + 1):
        xb, yb = v72.v71._select_batch(x_train, y_train, args.batch_size, step)
        h, caches = stack.forward_manual(xb)
        logits, head_cache = head.forward_manual(h)
        teacher_logits = None
        if cid == "SB0" and ema_values is not None and step > 20:
            student_values = _clone_params(providers)
            _restore_params(providers, ema_values)
            with torch.no_grad():
                ht, _ = stack.forward_manual(xb)
                teacher_logits, _ = head.forward_manual(ht)
                teacher_logits = teacher_logits.detach()
            _restore_params(providers, student_values)
        elif cid == "SB1" and snapshot_values is not None and step > 40:
            student_values = _clone_params(providers)
            _restore_params(providers, snapshot_values)
            with torch.no_grad():
                ht, _ = stack.forward_manual(xb)
                teacher_logits, _ = head.forward_manual(ht)
                teacher_logits = teacher_logits.detach()
            _restore_params(providers, student_values)
        elif cid in {"SB2", "SYS1", "SYS2", "SYS3", "SYS4", "SYS5", "SYS8", "SYS9", "SYS10", "SYS11", "SYS12", "SYS13", "SYS14", "SYS15", "SYS16", "SYS17", "SYS18", "SYS19", "SYS20", "SYS21", "SYS22", "SYS23", "SYS24"} and bestval_snapshot_values is not None and step > 40:
            student_values = _clone_params(providers)
            _restore_params(providers, bestval_snapshot_values)
            with torch.no_grad():
                ht, _ = stack.forward_manual(xb)
                teacher_logits, _ = head.forward_manual(ht)
                teacher_logits = teacher_logits.detach()
            _restore_params(providers, student_values)
        elif cid == "SB3" and step > 20:
            with torch.no_grad():
                noise = torch.randn(xb.shape, device=xb.device, dtype=xb.dtype, generator=dual_gen) * dual_view_noise_std
                ht, _ = stack.forward_manual(xb + noise)
                teacher_logits, _ = head.forward_manual(ht)
                teacher_logits = teacher_logits.detach()

        if teacher_logits is not None:
            loss, grad_logits, kl = v73._distill_loss_and_grad(
                logits,
                yb,
                teacher_logits,
                spec.label_smoothing,
                alpha=teacher_alpha,
                temperature=teacher_temperature,
            )
            self_teacher_active_steps += 1
            if not _finite(first_self_kl):
                first_self_kl = float(kl.detach().cpu())
        else:
            loss, grad_logits = v72._weighted_smooth_ce_and_grad(logits, yb, spec.label_smoothing, class_weights)
        if step == 1:
            first_loss_before = float(loss.detach().cpu())
        dh = head.backward_manual(grad_logits, head_cache)
        if getattr(stack, "release_head_cache_before_stack_backward", False):
            del head_cache, logits, loss, grad_logits
        stack.backward_manual(dh, caches)
        update_norm = opt.step(step, args.task_steps, warmup_cosine=True)

        current = _params(providers)
        if cid in {"TF8", "SB0"}:
            if ema_values is None:
                ema_values = [p.detach().clone() for p in current]
            else:
                with torch.no_grad():
                    for avg, p in zip(ema_values, current):
                        avg.mul_(ema_decay).add_(p.detach(), alpha=1.0 - ema_decay)
        if cid == "SB1" and (snapshot_values is None or step % 40 == 0):
            snapshot_values = [p.detach().clone() for p in current]
        elif cid == "TF7" and step >= swa_start and (step % int(args.trace_every) == 0 or step == int(args.task_steps)):
            snap = [p.detach().clone() for p in current]
            if swa_sum is None:
                swa_sum = [torch.zeros_like(p) for p in snap]
            with torch.no_grad():
                for acc, p in zip(swa_sum, snap):
                    acc.add_(p)
            swa_count += 1

        if step == 1:
            with torch.no_grad():
                h1, _ = stack.forward_manual(xb)
                logits1, _ = head.forward_manual(h1)
                first_loss_after = float(v72._weighted_smooth_ce_and_grad(logits1, yb, spec.label_smoothing, class_weights)[0].detach().cpu())

        if step % int(args.trace_every) == 0 or step == int(args.task_steps):
            tr = v72.v71._manual_eval(stack, head, train_eval_x, train_eval_y, args.eval_batch_size, bundle.num_classes)
            va = v72.v71._manual_eval(stack, head, x_val, y_val, args.eval_batch_size, bundle.num_classes)
            trace.append({
                "stage": "TASK_TRACE",
                "candidate_id": spec.candidate_id,
                "candidate_name": spec.candidate_name,
                "family": spec.family,
                "dataset": dataset,
                "seed": seed,
                "step": step,
                "wall_clock_time_sec": time.perf_counter() - started,
                "train_loss": tr["loss"],
                "train_acc": tr["acc"],
                "val_loss": va["loss"],
                "val_acc": va["acc"],
                "ECE": va["ECE"],
                "NLL": va["NLL"],
                "update_norm": update_norm,
                "recipe": {
                    "TF7": "teacher_free_swa_weights",
                    "TF8": "teacher_free_ema_weights",
                    "SB0": "self_bootstrap_ema_T2_alpha025",
                    "SB1": "self_bootstrap_snapshot_T2_alpha025",
                    "SB2": "self_bootstrap_bestval_snapshot_T2_alpha025",
                    "SB3": "self_bootstrap_dual_view_T2_alpha025",
                    "SYS1": "system_recompute_cache_trim_bestval_snapshot_T2_alpha025",
                    "SYS2": "system_sgd_state_trim_bestval_snapshot_T2_alpha025_lr3",
                    "SYS3": "system_rms_one_state_trim_bestval_snapshot_T2_alpha025",
                    "SYS4": "system_adamw_half_state_trim_bestval_snapshot_T2_alpha025",
                    "SYS5": "system_adamw_v_half_state_trim_bestval_snapshot_T2_alpha025",
                    "SYS8": "system_adamw_v_bfloat_state_trim_bestval_snapshot_T2_alpha025",
                    "SYS9": "system_adamw_v_bfloat_native_denom_state_trim_bestval_snapshot_T2_alpha025",
                    "SYS10": "system_adamw_bfloat_state_trim_bestval_snapshot_T2_alpha025",
                    "SYS11": "system_adamw_v_bfloat_addcdiv_update_bestval_snapshot_T2_alpha025",
                    "SYS12": "system_adamw_head_bfloat_v_bfloat_addcdiv_update_bestval_snapshot_T2_alpha025",
                    "SYS13": "system_packed_recompute_adamw_v_bfloat_state_bestval_snapshot_T2_alpha025",
                    "SYS14": "system_adamw_stack_bfloat_v_bfloat_addcdiv_update_bestval_snapshot_T2_alpha025",
                    "SYS15": "system_adamw_tail_bfloat_v_bfloat_state_bestval_snapshot_T2_alpha025",
                    "SYS16": "system_adamw_l0_bfloat_v_bfloat_state_bestval_snapshot_T2_alpha025",
                    "SYS17": "system_adamw_l0_bfloat_v_bfloat_addcdiv_update_bestval_snapshot_T2_alpha025",
                    "SYS18": "system_adamw_l0_bfloat_v_bfloat_addcdiv_update_bestval_snapshot_T2_alpha010",
                    "SYS19": "system_adamw_l0_bfloat_v_bfloat_addcdiv_update_bestval_snapshot_T2_alpha050",
                    "SYS20": "system_packed_recompute_adamw_v_bfloat_bestval_snapshot_T2_alpha010",
                    "SYS21": "system_packed_recompute_adamw_v_bfloat_bestval_snapshot_T2_alpha050",
                    "SYS22": "system_packed_cachey_adamw_v_bfloat_bestval_snapshot_T2_alpha010",
                    "SYS23": "system_packed_recompute_adamw_v_bfloat_bestval_snapshot_T2_alpha005",
                    "SYS24": "system_packed_prefix_recompute_adamw_v_bfloat_bestval_snapshot_T2_alpha005",
                    "KC1": "ce_only_recompute_cache_trim",
                    "KC2": "ce_only_recompute_cache_trim_adamw_v_bfloat",
                    "KC3": "ce_only_packed_recompute_adamw_v_bfloat",
                    "KC4": "ce_only_packed_cachey_adamw_v_bfloat",
                    "KC5": "ce_only_packed_prefix_recompute_adamw_v_bfloat",
                    "KC6": "ce_only_packed_prefix_recompute",
                    "KC7": "ce_only_packed_prefix_recompute_adamw_addcdiv",
                    "KC8": "ce_only_packed_prefix_recompute_adamw_addcdiv_head_cache_release",
                    "KF1": "ce_only_packed_prefix_triton_silu_backward",
                    "KF2": "ce_only_packed_prefix_triton_silu_backward_adamw_addcdiv",
                    "KF3": "ce_only_packed_prefix_aten_silu_backward",
                    "KF4": "ce_only_packed_prefix_aten_silu_backward_adamw_addcdiv",
                    "KF5": "ce_only_packed_prefix_explicit_inplace_silu_backward",
                    "KF6": "ce_only_packed_prefix_explicit_inplace_silu_backward_adamw_addcdiv",
                    "KF7": "ce_only_packed_prefix_aten_upper_only_silu_backward",
                    "KF8": "ce_only_packed_prefix_aten_upper_only_silu_backward_adamw_addcdiv",
                    "KF9": "ce_only_packed_prefix_aten_lower_only_silu_backward",
                    "KF10": "ce_only_packed_prefix_aten_lower_only_silu_backward_adamw_addcdiv",
                    "KW1": "ce_only_packed_prefix_no_input_grad_recompute",
                    "KW2": "ce_only_packed_prefix_no_input_grad_recompute_adamw_addcdiv",
                    "KW3": "ce_only_packed_prefix_no_input_grad_aten_lower_only_silu_backward_adamw_addcdiv",
                    "KW4": "ce_only_packed_prefix_cached_no_input_grad_aten_lower_only_silu_backward_adamw_addcdiv",
                    "KW5": "ce_only_packed_prefix_compiled_explicit_no_input_grad_aten_lower_only_silu_backward_adamw_addcdiv",
                    "KW6": "ce_only_packed_prefix_fast_mix_view_no_input_grad_aten_lower_only_silu_backward_adamw_addcdiv",
                    "RR3": "representation_dwm2lite_rbf_manual_teacher_free",
                    "RR8": "representation_dwm2lite_rbf_streaming_manual_teacher_free",
                    "RR9": "representation_dwm2lite_rbf_basis2_unrolled_manual_teacher_free",
                    "RR10": "representation_dwm2lite_rbf_basis2_recompute_z_manual_teacher_free",
                    "RR11": "representation_dwm2lite_rbf_basis2_compiled_manual_teacher_free",
                    "RR12": "representation_dwm2lite_rbf_basis2_compiled_forward_manual_teacher_free",
                    "RR13": "representation_dwm2lite_rbf_basis2_triton_manual_teacher_free",
                    "RR14": "representation_dwm2lite_rbf_basis2_triton_backward_manual_teacher_free",
                }[cid],
                "external_teacher_used": 0,
                "external_teacher_logits_used": 0,
                "external_teacher_forward_used": 0,
                "self_teacher_used": int(cid in self_bootstrap_ids),
                "self_teacher_logits_used": int(cid in self_bootstrap_ids),
                "self_teacher_active_steps": self_teacher_active_steps,
                "self_teacher_kl_step1": first_self_kl,
                "fake_data_used": 0,
                "proxy_row_used": 0,
            })
            if cid in {"SB2", "SYS1", "SYS2", "SYS3", "SYS4", "SYS5", "SYS8", "SYS9", "SYS10", "SYS11", "SYS12", "SYS13", "SYS14", "SYS15", "SYS16", "SYS17", "SYS18", "SYS19", "SYS20", "SYS21", "SYS22", "SYS23", "SYS24"} and _finite(va.get("acc")) and float(va["acc"]) > bestval_snapshot_score:
                bestval_snapshot_score = float(va["acc"])
                bestval_snapshot_values = _clone_params(providers)

    final_raw_snapshot = _clone_params(providers)
    averaging_applied = 0
    if cid == "TF7" and swa_sum is not None and swa_count > 0:
        _restore_params(providers, [acc / float(swa_count) for acc in swa_sum])
        averaging_applied = 1
    elif cid == "TF8" and ema_values is not None:
        _restore_params(providers, ema_values)
        averaging_applied = 1

    train1 = v72.v71._manual_eval(stack, head, train_eval_x, train_eval_y, args.eval_batch_size, bundle.num_classes)
    val1 = v72.v71._manual_eval(stack, head, x_val, y_val, args.eval_batch_size, bundle.num_classes)
    test1 = v72.v71._manual_eval(stack, head, x_test, y_test, args.eval_batch_size, bundle.num_classes)
    summary = v72.v71._task_summary_base(args, spec, dataset, seed, bundle.input_dim, bundle.num_classes)
    summary.update({
        "train_loss_before": train0["loss"],
        "train_loss_after": train1["loss"],
        "val_loss_before": val0["loss"],
        "val_loss_after": val1["loss"],
        "train_loss_delta": train1["loss"] - train0["loss"],
        "val_loss_delta": val1["loss"] - val0["loss"],
        "train_acc": train1["acc"],
        "val_acc": val1["acc"],
        "test_acc": test1["acc"],
        "ECE": val1["ECE"],
        "NLL": val1["NLL"],
        "test_ECE": test1["ECE"],
        "test_NLL": test1["NLL"],
        "confidence_mean": val1["confidence_mean"],
        "margin_p10": val1["margin_p10"],
        "feature_effective_rank": val1["feature_effective_rank"],
        "classwise_acc": val1["classwise_acc"],
        "one_step_loss_before": first_loss_before,
        "one_step_loss_after": first_loss_after,
        "one_step_loss_delta": first_loss_after - first_loss_before if _finite(first_loss_after) and _finite(first_loss_before) else float("nan"),
        "one_step_pass": int(_finite(first_loss_after) and _finite(first_loss_before) and first_loss_after < first_loss_before),
        "grad_relerr_max": METRIC_UNAVAILABLE,
        "grad_cos_min": METRIC_UNAVAILABLE,
        "finite_grad": 1,
        "val_loss_auc_step": v72.v71._auc([(r["step"], r["val_loss"]) for r in trace]),
        "val_loss_auc_time": v72.v71._auc([(r["wall_clock_time_sec"], r["val_loss"]) for r in trace]),
        "stack_param_count": stack.param_count(),
        "head_param_count": head.param_count(),
        "selected_checkpoint_step": 0,
        "selected_checkpoint_score": float("nan"),
        "final_weight_averaging_applied": averaging_applied,
        "final_raw_param_l2": float(torch.cat([p.flatten().detach().float().cpu() for p in final_raw_snapshot]).norm().item()) if final_raw_snapshot else 0.0,
        "recipe": {
            "TF7": "teacher_free_swa_weights",
            "TF8": "teacher_free_ema_weights",
            "SB0": "self_bootstrap_ema_T2_alpha025",
            "SB1": "self_bootstrap_snapshot_T2_alpha025",
            "SB2": "self_bootstrap_bestval_snapshot_T2_alpha025",
            "SB3": "self_bootstrap_dual_view_T2_alpha025",
            "SYS1": "system_recompute_cache_trim_bestval_snapshot_T2_alpha025",
            "SYS2": "system_sgd_state_trim_bestval_snapshot_T2_alpha025_lr3",
            "SYS3": "system_rms_one_state_trim_bestval_snapshot_T2_alpha025",
            "SYS4": "system_adamw_half_state_trim_bestval_snapshot_T2_alpha025",
            "SYS5": "system_adamw_v_half_state_trim_bestval_snapshot_T2_alpha025",
            "SYS8": "system_adamw_v_bfloat_state_trim_bestval_snapshot_T2_alpha025",
            "SYS9": "system_adamw_v_bfloat_native_denom_state_trim_bestval_snapshot_T2_alpha025",
            "SYS10": "system_adamw_bfloat_state_trim_bestval_snapshot_T2_alpha025",
            "SYS11": "system_adamw_v_bfloat_addcdiv_update_bestval_snapshot_T2_alpha025",
            "SYS12": "system_adamw_head_bfloat_v_bfloat_addcdiv_update_bestval_snapshot_T2_alpha025",
            "SYS13": "system_packed_recompute_adamw_v_bfloat_state_bestval_snapshot_T2_alpha025",
            "SYS14": "system_adamw_stack_bfloat_v_bfloat_addcdiv_update_bestval_snapshot_T2_alpha025",
            "SYS15": "system_adamw_tail_bfloat_v_bfloat_state_bestval_snapshot_T2_alpha025",
            "SYS16": "system_adamw_l0_bfloat_v_bfloat_state_bestval_snapshot_T2_alpha025",
            "SYS17": "system_adamw_l0_bfloat_v_bfloat_addcdiv_update_bestval_snapshot_T2_alpha025",
            "SYS18": "system_adamw_l0_bfloat_v_bfloat_addcdiv_update_bestval_snapshot_T2_alpha010",
            "SYS19": "system_adamw_l0_bfloat_v_bfloat_addcdiv_update_bestval_snapshot_T2_alpha050",
            "SYS20": "system_packed_recompute_adamw_v_bfloat_bestval_snapshot_T2_alpha010",
            "SYS21": "system_packed_recompute_adamw_v_bfloat_bestval_snapshot_T2_alpha050",
            "SYS22": "system_packed_cachey_adamw_v_bfloat_bestval_snapshot_T2_alpha010",
            "SYS23": "system_packed_recompute_adamw_v_bfloat_bestval_snapshot_T2_alpha005",
            "SYS24": "system_packed_prefix_recompute_adamw_v_bfloat_bestval_snapshot_T2_alpha005",
            "KC1": "ce_only_recompute_cache_trim",
            "KC2": "ce_only_recompute_cache_trim_adamw_v_bfloat",
            "KC3": "ce_only_packed_recompute_adamw_v_bfloat",
            "KC4": "ce_only_packed_cachey_adamw_v_bfloat",
            "KC5": "ce_only_packed_prefix_recompute_adamw_v_bfloat",
            "KC6": "ce_only_packed_prefix_recompute",
            "KC7": "ce_only_packed_prefix_recompute_adamw_addcdiv",
            "KC8": "ce_only_packed_prefix_recompute_adamw_addcdiv_head_cache_release",
            "KF1": "ce_only_packed_prefix_triton_silu_backward",
            "KF2": "ce_only_packed_prefix_triton_silu_backward_adamw_addcdiv",
            "KF3": "ce_only_packed_prefix_aten_silu_backward",
            "KF4": "ce_only_packed_prefix_aten_silu_backward_adamw_addcdiv",
            "KF5": "ce_only_packed_prefix_explicit_inplace_silu_backward",
            "KF6": "ce_only_packed_prefix_explicit_inplace_silu_backward_adamw_addcdiv",
            "KF7": "ce_only_packed_prefix_aten_upper_only_silu_backward",
            "KF8": "ce_only_packed_prefix_aten_upper_only_silu_backward_adamw_addcdiv",
            "KF9": "ce_only_packed_prefix_aten_lower_only_silu_backward",
            "KF10": "ce_only_packed_prefix_aten_lower_only_silu_backward_adamw_addcdiv",
            "KW1": "ce_only_packed_prefix_no_input_grad_recompute",
            "KW2": "ce_only_packed_prefix_no_input_grad_recompute_adamw_addcdiv",
            "KW3": "ce_only_packed_prefix_no_input_grad_aten_lower_only_silu_backward_adamw_addcdiv",
            "KW4": "ce_only_packed_prefix_cached_no_input_grad_aten_lower_only_silu_backward_adamw_addcdiv",
            "KW5": "ce_only_packed_prefix_compiled_explicit_no_input_grad_aten_lower_only_silu_backward_adamw_addcdiv",
            "KW6": "ce_only_packed_prefix_fast_mix_view_no_input_grad_aten_lower_only_silu_backward_adamw_addcdiv",
            "RR3": "representation_dwm2lite_rbf_manual_teacher_free",
            "RR8": "representation_dwm2lite_rbf_streaming_manual_teacher_free",
            "RR9": "representation_dwm2lite_rbf_basis2_unrolled_manual_teacher_free",
            "RR10": "representation_dwm2lite_rbf_basis2_recompute_z_manual_teacher_free",
            "RR11": "representation_dwm2lite_rbf_basis2_compiled_manual_teacher_free",
            "RR12": "representation_dwm2lite_rbf_basis2_compiled_forward_manual_teacher_free",
            "RR13": "representation_dwm2lite_rbf_basis2_triton_manual_teacher_free",
            "RR14": "representation_dwm2lite_rbf_basis2_triton_backward_manual_teacher_free",
        }[cid],
        "external_teacher_used": 0,
        "external_teacher_logits_used": 0,
        "external_teacher_forward_used": 0,
        "self_teacher_used": int(cid in self_bootstrap_ids),
        "self_teacher_logits_used": int(cid in self_bootstrap_ids),
        "self_teacher_active_steps": self_teacher_active_steps,
        "self_teacher_kl_step1": first_self_kl,
        "self_teacher_bestval_score": bestval_snapshot_score if _finite(bestval_snapshot_score) else float("nan"),
        "fake_data_used": 0,
        "proxy_row_used": 0,
    })
    return summary, trace


def _gradient_check_v80(args: argparse.Namespace, spec: Any, dataset: str, batch_size: int) -> Dict[str, Any]:
    cid = str(spec.candidate_id)
    if cid not in (V80_REPRESENTATION_CE_IDS | V80_KERNEL_NATIVE_CE_IDS | V80_SELF_BOOTSTRAP_IDS):
        return _ORIG_V73_GRADIENT_CHECK(args, spec, dataset, batch_size)
    device = get_device(args.device)
    bundle = v72.v71.load_vision_bundle(
        dataset,
        data_root=Path(args.data_root),
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=0,
        allow_fake_data=False,
    )
    x = bundle.x_train[:batch_size].to(device)
    y = bundle.y_train[:batch_size].to(device)
    set_seed(v72._stable_seed("grad-v80-self-bootstrap", dataset, batch_size, _init_seed_candidate_id_v80(spec)))
    stack, head = v73._make_manual_candidate(spec, bundle.input_dim, bundle.num_classes, args.hidden_dim, args.basis_count, device)
    logits, caches, head_cache, _h = v72._manual_forward(stack, head, x)
    teacher_logits = logits.detach().clone()
    teacher_alpha = 0.05 if cid in {"SYS23", "SYS24"} else 0.10 if cid in {"SYS18", "SYS20", "SYS22"} else 0.50 if cid in {"SYS19", "SYS21"} else 0.25
    if cid in (V80_REPRESENTATION_CE_IDS | V80_KERNEL_NATIVE_CE_IDS):
        manual_loss, grad_logits = v72._weighted_smooth_ce_and_grad(logits, y, spec.label_smoothing, None)
    else:
        manual_loss, grad_logits, _kl = v73._distill_loss_and_grad(
            logits,
            y,
            teacher_logits,
            spec.label_smoothing,
            alpha=teacher_alpha,
            temperature=2.0,
        )
    dh = head.backward_manual(grad_logits, head_cache)
    stack.backward_manual(dh, caches)
    manual_params, manual_grads = v72._flatten_params_and_grads([stack, head])
    logits_auto, leafs = v72._autograd_forward(stack, head, x)
    if cid in (V80_REPRESENTATION_CE_IDS | V80_KERNEL_NATIVE_CE_IDS):
        auto_loss = v72._loss_autograd_v72(spec, dataset, logits_auto, y, None)
    else:
        auto_loss = v73._distill_loss_autograd(
            logits_auto,
            y,
            teacher_logits,
            spec.label_smoothing,
            alpha=teacher_alpha,
            temperature=2.0,
        )
    auto_grads = torch.autograd.grad(auto_loss, leafs, allow_unused=False)
    auto_flat = torch.cat([g.detach().flatten().float().cpu() for g in auto_grads])
    abs_err = (manual_grads - auto_flat).abs()
    scale = torch.maximum(manual_grads.abs(), auto_flat.abs())
    significant = scale > 1.0e-4
    rel = abs_err[significant] / scale[significant].clamp_min(1.0e-12) if bool(significant.any()) else abs_err
    cos = float(torch.nn.functional.cosine_similarity(manual_grads, auto_flat, dim=0).item()) if manual_grads.numel() else 1.0
    forward_rel = (logits.detach() - logits_auto.detach()).abs().max() / logits_auto.detach().abs().max().clamp_min(1.0e-8)
    lr = 1.0e-4
    before = [p.detach().clone() for _n, p, _g in stack.params_and_grads()] + [p.detach().clone() for _n, p, _g in head.params_and_grads()]
    loss_before = float(manual_loss.detach().cpu())
    with torch.no_grad():
        for _name, p, g in stack.params_and_grads():
            p.add_(g, alpha=-lr)
        for _name, p, g in head.params_and_grads():
            p.add_(g, alpha=-lr)
        logits_after, _caches_after, _head_cache_after, _h_after = v72._manual_forward(stack, head, x)
        if cid in (V80_REPRESENTATION_CE_IDS | V80_KERNEL_NATIVE_CE_IDS):
            loss_after = float(v72._weighted_smooth_ce_and_grad(logits_after, y, spec.label_smoothing, None)[0].detach().cpu())
        else:
            loss_after = float(
                v73._distill_loss_and_grad(
                    logits_after,
                    y,
                    teacher_logits,
                    spec.label_smoothing,
                    alpha=teacher_alpha,
                    temperature=2.0,
                )[0].detach().cpu()
            )
        idx = 0
        for _name, p, _g in stack.params_and_grads():
            p.copy_(before[idx])
            idx += 1
        for _name, p, _g in head.params_and_grads():
            p.copy_(before[idx])
            idx += 1
    after_restore, _ = v72._flatten_params_and_grads([stack, head])
    rollback_err = float((after_restore - manual_params).abs().max().item()) if manual_params.numel() else 0.0
    rel_max = float(rel.max().item()) if rel.numel() else 0.0
    return {
        "stage": "P2_GRADIENT",
        "candidate_id": spec.candidate_id,
        "candidate_name": spec.candidate_name,
        "family": spec.family,
        "dataset": dataset,
        "batch_size": batch_size,
        "forward_relerr_max": float(forward_rel.detach().cpu()),
        "loss_relerr": abs(float(manual_loss.detach().cpu()) - float(auto_loss.detach().cpu())) / max(1.0e-8, abs(float(auto_loss.detach().cpu()))),
        "grad_relerr_max": rel_max,
        "grad_relerr_mean": float(rel.mean().item()) if rel.numel() else 0.0,
        "grad_abs_err_max": float(abs_err.max().item()) if abs_err.numel() else 0.0,
        "grad_relerr_scope": "abs_grad_gt_1e-4_else_abs_err",
        "grad_cos": cos,
        "grad_cos_min": cos,
        "grad_cos_mean": cos,
        "finite_grad": int(bool(torch.isfinite(manual_grads).all()) and bool(torch.isfinite(auto_flat).all())),
        "finite_forward": int(bool(torch.isfinite(logits).all()) and bool(torch.isfinite(logits_auto).all())),
        "finite_loss": int(math.isfinite(loss_before) and math.isfinite(float(auto_loss.detach().cpu()))),
        "nan_count": int(torch.isnan(manual_grads).sum().item()),
        "inf_count": int(torch.isinf(manual_grads).sum().item()),
        "rollback_max_abs_error": rollback_err,
        "rollback_error": rollback_err,
        "rollback_pass": int(rollback_err < 1.0e-8),
        "one_step_loss_before": loss_before,
        "one_step_loss_after": loss_after,
        "one_step_loss_delta": loss_after - loss_before,
        "one_step_pass": int(loss_after < loss_before),
        "recipe": "teacher_free_ce_gradient_check" if cid in (V80_REPRESENTATION_CE_IDS | V80_KERNEL_NATIVE_CE_IDS) else "self_bootstrap_objective_check",
        "teacher_id": "none" if cid in (V80_REPRESENTATION_CE_IDS | V80_KERNEL_NATIVE_CE_IDS) else "same_run_self_snapshot",
        "distill_temperature": 0.0 if cid in (V80_REPRESENTATION_CE_IDS | V80_KERNEL_NATIVE_CE_IDS) else 2.0,
        "alpha_logit": 0.0 if cid in (V80_REPRESENTATION_CE_IDS | V80_KERNEL_NATIVE_CE_IDS) else teacher_alpha,
        "self_teacher_used": 0 if cid in (V80_REPRESENTATION_CE_IDS | V80_KERNEL_NATIVE_CE_IDS) else 1,
        "external_teacher_used": 0,
        "grad_pass": int((rel.numel() > 0 and rel_max <= 1.0e-4) and cos >= 0.999 and loss_after < loss_before and rollback_err < 1.0e-8),
        "fake_data_used": 0,
        "proxy_row_used": 0,
    }


def _spec_v2_registry(args: argparse.Namespace) -> Dict[str, CandidateSpecV2]:
    hidden = str(getattr(args, "hidden_dim", "runtime_arg"))
    basis = str(getattr(args, "basis_count", "runtime_arg"))

    def spec(
        cid: str,
        name: str,
        family: str,
        *,
        role: str = "official",
        dense: str = "manual_purekan",
        external: int = 0,
        self_teacher: int = 0,
        self_type: str = "none",
        official: int = 1,
        pure_supervised: int = 1,
        manual: int = 1,
        status: str = "implemented",
    ) -> CandidateSpecV2:
        return CandidateSpecV2(
            candidate_id=cid,
            candidate_name=name,
            route_role=role,
            model_family=family,
            dense_cls_name=dense,
            dense_cls_module_path="experiments.run_gafu_v73_real" if manual else "experiments.run_gafu_v71_real",
            hidden_dim=hidden,
            basis_count=basis,
            depth="runtime_spec",
            init_policy="M5init_no_external_teacher" if cid in V80_M5_INIT_IDS else "matched_seed",
            optimizer_policy=(
                "manual_sgd_no_momentum_state_trim"
                if cid == "SYS2"
                else "manual_rms_one_state_trim"
                if cid == "SYS3"
                else "manual_adamw_half_state_trim"
                if cid == "SYS4"
                else "manual_adamw_v_half_state_trim"
                if cid == "SYS5"
                else "manual_adamw_v_bfloat_state_trim"
                if cid == "SYS8"
                else "manual_adamw_v_bfloat_native_denom_state_trim"
                if cid == "SYS9"
                else "manual_adamw_bfloat_state_trim"
                if cid == "SYS10"
                else "manual_adamw_v_bfloat_addcdiv_update_trim"
                if cid == "SYS11"
                else "manual_adamw_head_bfloat_v_bfloat_addcdiv_update_trim"
                if cid == "SYS12"
                else "manual_packed_recompute_adamw_v_bfloat_state_trim"
                if cid == "SYS13"
                else "manual_adamw_stack_bfloat_v_bfloat_addcdiv_update_trim"
                if cid == "SYS14"
                else "manual_adamw_tail_bfloat_v_bfloat_state_trim"
                if cid == "SYS15"
                else "manual_adamw_l0_bfloat_v_bfloat_state_trim"
                if cid == "SYS16"
                else "manual_adamw_l0_bfloat_v_bfloat_addcdiv_update_trim"
                if cid == "SYS17"
                else "manual_adamw_l0_bfloat_v_bfloat_addcdiv_update_trim_alpha010"
                if cid == "SYS18"
                else "manual_adamw_l0_bfloat_v_bfloat_addcdiv_update_trim_alpha050"
                if cid == "SYS19"
                else "manual_packed_recompute_adamw_v_bfloat_state_trim_alpha010"
                if cid == "SYS20"
                else "manual_packed_recompute_adamw_v_bfloat_state_trim_alpha050"
                if cid == "SYS21"
                else "manual_packed_cachey_adamw_v_bfloat_state_trim_alpha010"
                if cid == "SYS22"
                else "manual_packed_recompute_adamw_v_bfloat_state_trim_alpha005"
                if cid == "SYS23"
                else "manual_packed_prefix_recompute_adamw_v_bfloat_state_trim_alpha005"
                if cid == "SYS24"
                else "manual_recompute_fast_adamw_no_loss_backward"
                if cid == "KC1"
                else "manual_recompute_adamw_v_bfloat_no_loss_backward"
                if cid == "KC2"
                else "manual_packed_recompute_adamw_v_bfloat_no_loss_backward"
                if cid == "KC3"
                else "manual_packed_cachey_adamw_v_bfloat_no_loss_backward"
                if cid == "KC4"
                else "manual_packed_prefix_recompute_adamw_v_bfloat_no_loss_backward"
                if cid == "KC5"
                else "manual_packed_prefix_recompute_fast_adamw_no_loss_backward"
                if cid == "KC6"
                else "manual_packed_prefix_recompute_adamw_addcdiv_no_loss_backward"
                if cid == "KC7"
                else "manual_packed_prefix_recompute_adamw_addcdiv_head_cache_release_no_loss_backward"
                if cid == "KC8"
                else "manual_packed_prefix_triton_silu_backward_no_loss_backward"
                if cid == "KF1"
                else "manual_packed_prefix_triton_silu_backward_adamw_addcdiv_no_loss_backward"
                if cid == "KF2"
                else "manual_packed_prefix_aten_silu_backward_no_loss_backward"
                if cid == "KF3"
                else "manual_packed_prefix_aten_silu_backward_adamw_addcdiv_no_loss_backward"
                if cid == "KF4"
                else "manual_packed_prefix_explicit_inplace_silu_backward_no_loss_backward"
                if cid == "KF5"
                else "manual_packed_prefix_explicit_inplace_silu_backward_adamw_addcdiv_no_loss_backward"
                if cid == "KF6"
                else "manual_packed_prefix_aten_upper_only_silu_backward_no_loss_backward"
                if cid == "KF7"
                else "manual_packed_prefix_aten_upper_only_silu_backward_adamw_addcdiv_no_loss_backward"
                if cid == "KF8"
                else "manual_packed_prefix_aten_lower_only_silu_backward_no_loss_backward"
                if cid == "KF9"
                else "manual_packed_prefix_aten_lower_only_silu_backward_adamw_addcdiv_no_loss_backward"
                if cid == "KF10"
                else "manual_packed_prefix_no_input_grad_recompute_no_loss_backward"
                if cid == "KW1"
                else "manual_packed_prefix_no_input_grad_recompute_adamw_addcdiv_no_loss_backward"
                if cid == "KW2"
                else "manual_packed_prefix_no_input_grad_aten_lower_only_silu_backward_adamw_addcdiv_no_loss_backward"
                if cid == "KW3"
                else "manual_packed_prefix_cached_no_input_grad_aten_lower_only_silu_backward_adamw_addcdiv_no_loss_backward"
                if cid == "KW4"
                else "manual_packed_prefix_compiled_explicit_no_input_grad_aten_lower_only_silu_backward_adamw_addcdiv_no_loss_backward"
                if cid == "KW5"
                else "manual_packed_prefix_fast_mix_view_no_input_grad_aten_lower_only_silu_backward_adamw_addcdiv_no_loss_backward"
                if cid == "KW6"
                else ("manual_fast_adamw_no_loss_backward" if manual else "torch_adamw")
            ),
            external_teacher_used=external,
            external_teacher_source="C3" if external else "none",
            external_teacher_logits_used=external,
            external_teacher_forward_used=external,
            self_teacher_used=self_teacher,
            self_teacher_type=self_type,
            self_teacher_logits_used=self_teacher,
            self_teacher_forward_used=int(self_teacher),
            teacher_checkpoint_path="none",
            teacher_artifact_hash=METRIC_UNAVAILABLE if external else "none",
            no_external_teacher_eligible=int(not external),
            pure_supervised_eligible=pure_supervised,
            official_eligible=official,
            expected_manual_forward=manual,
            expected_manual_backward=manual,
            expected_manual_update=manual,
            expected_nonkan_count=0 if manual else -1,
            implementation_status=status,
        )

    rows = {
        "B0": spec("B0", "MLP-AdamW-reference", "mlp_reference", role="baseline", dense="mlp", official=0, manual=0),
        "B2": spec("B2", "MLP-C3-logit-distill-diagnostic", "mlp_external_teacher_diagnostic", role="diagnostic_external", dense="mlp", external=1, official=0, pure_supervised=0, manual=0),
        "M12": spec("M12", "M12-C3-logit-distill-diagnostic", "purekan_external_teacher_diagnostic", role="diagnostic_external", external=1, official=0, pure_supervised=0),
        "M13": spec("M13", "M13-teacher-free-M5init-official", "purekan_teacher_free"),
        "A2S": spec("A2S", "A2S-cached-linear-stack-poly2-silu-head", "purekan_teacher_free_structural"),
        "M9": spec("M9", "A2S-fused-linear-silu-stack-packed-generic-head", "purekan_teacher_free_structural"),
        "TF7": spec("TF7", "M13-SWA-final-averaging-no-external-teacher", "purekan_teacher_free_swa"),
        "TF8": spec("TF8", "M13-EMA-weights-no-logit-teacher", "purekan_teacher_free_ema"),
        "RR3": spec("RR3", "RR3-M13-DWM2LiteDense-rbf-manual-code-native", "representation_dwm2lite_rbf_manual", role="official_representation_repair", dense="manual_dwm2lite_rbf_stack"),
        "RR8": spec("RR8", "RR8-M13-DWM2LiteDense-rbf-streaming-manual-code-native", "representation_dwm2lite_rbf_streaming_manual", role="official_representation_repair", dense="manual_dwm2lite_rbf_streaming_stack"),
        "RR9": spec("RR9", "RR9-M13-DWM2LiteDense-rbf-basis2-unrolled-manual-code-native", "representation_dwm2lite_rbf_basis2_unrolled_manual", role="official_representation_repair", dense="manual_dwm2lite_rbf_basis2_unrolled_stack"),
        "RR10": spec("RR10", "RR10-M13-DWM2LiteDense-rbf-basis2-recompute-z-manual-code-native", "representation_dwm2lite_rbf_basis2_recompute_z_manual", role="official_representation_repair", dense="manual_dwm2lite_rbf_basis2_recompute_z_stack"),
        "RR11": spec("RR11", "RR11-M13-DWM2LiteDense-rbf-basis2-compiled-manual-code-native", "representation_dwm2lite_rbf_basis2_compiled_manual", role="official_representation_repair", dense="manual_dwm2lite_rbf_basis2_compiled_stack"),
        "RR12": spec("RR12", "RR12-M13-DWM2LiteDense-rbf-basis2-compiled-forward-manual-code-native", "representation_dwm2lite_rbf_basis2_compiled_forward_manual", role="official_representation_repair", dense="manual_dwm2lite_rbf_basis2_compiled_forward_stack"),
        "RR13": spec("RR13", "RR13-M13-DWM2LiteDense-rbf-basis2-triton-transform-manual-code-native", "representation_dwm2lite_rbf_basis2_triton_manual", role="official_representation_repair", dense="manual_dwm2lite_rbf_basis2_triton_stack"),
        "RR14": spec("RR14", "RR14-M13-DWM2LiteDense-rbf-basis2-triton-backward-manual-code-native", "representation_dwm2lite_rbf_basis2_triton_backward_manual", role="official_representation_repair", dense="manual_dwm2lite_rbf_basis2_triton_backward_stack"),
        "KC1": spec("KC1", "KC1-M13-CE-only-recompute-cache-trim", "ceonly_recompute_cache_trim", role="official_kernel_native_s2_repair", dense="manual_purekan_recompute_cache_trim"),
        "KC2": spec("KC2", "KC2-M13-CE-only-recompute-cache-trim-AdamWVBF16", "ceonly_recompute_cache_trim_adamw_v_bfloat", role="official_kernel_native_s2_repair", dense="manual_purekan_recompute_cache_trim_adamw_v_bfloat"),
        "KC3": spec("KC3", "KC3-M13-CE-only-packed-recompute-AdamWVBF16", "ceonly_packed_recompute_adamw_v_bfloat", role="official_kernel_native_s2_repair", dense="manual_purekan_packed_recompute_cache_trim_adamw_v_bfloat"),
        "KC4": spec("KC4", "KC4-M13-CE-only-packed-cache-y-AdamWVBF16", "ceonly_packed_cachey_adamw_v_bfloat", role="official_kernel_native_s2_repair", dense="manual_purekan_packed_cache_y_adamw_v_bfloat"),
        "KC5": spec("KC5", "KC5-M13-CE-only-packed-prefix-recompute-AdamWVBF16", "ceonly_packed_prefix_recompute_adamw_v_bfloat", role="official_kernel_native_s2_repair", dense="manual_purekan_packed_prefix_recompute_adamw_v_bfloat"),
        "KC6": spec("KC6", "KC6-M13-CE-only-packed-prefix-recompute", "ceonly_packed_prefix_recompute", role="official_kernel_native_s2_repair", dense="manual_purekan_packed_prefix_recompute_cache_trim"),
        "KC7": spec("KC7", "KC7-M13-CE-only-packed-prefix-recompute-AdamWAddcdiv", "ceonly_packed_prefix_recompute_adamw_addcdiv", role="official_kernel_native_s2_repair", dense="manual_purekan_packed_prefix_recompute_adamw_addcdiv"),
        "KC8": spec("KC8", "KC8-M13-CE-only-packed-prefix-recompute-AdamWAddcdiv-HeadCacheRelease", "ceonly_packed_prefix_recompute_adamw_addcdiv_head_cache_release", role="official_kernel_native_s2_repair", dense="manual_purekan_packed_prefix_recompute_adamw_addcdiv_head_cache_release"),
        "KF1": spec("KF1", "KF1-M13-CE-only-packed-prefix-TritonSiLUBackward", "ceonly_packed_prefix_triton_silu_backward", role="official_kernel_native_s2_repair", dense="manual_purekan_packed_prefix_triton_silu_backward"),
        "KF2": spec("KF2", "KF2-M13-CE-only-packed-prefix-TritonSiLUBackward-AdamWAddcdiv", "ceonly_packed_prefix_triton_silu_backward_adamw_addcdiv", role="official_kernel_native_s2_repair", dense="manual_purekan_packed_prefix_triton_silu_backward_adamw_addcdiv"),
        "KF3": spec("KF3", "KF3-M13-CE-only-packed-prefix-AtenSiLUBackward", "ceonly_packed_prefix_aten_silu_backward", role="official_kernel_native_s2_repair", dense="manual_purekan_packed_prefix_aten_silu_backward"),
        "KF4": spec("KF4", "KF4-M13-CE-only-packed-prefix-AtenSiLUBackward-AdamWAddcdiv", "ceonly_packed_prefix_aten_silu_backward_adamw_addcdiv", role="official_kernel_native_s2_repair", dense="manual_purekan_packed_prefix_aten_silu_backward_adamw_addcdiv"),
        "KF5": spec("KF5", "KF5-M13-CE-only-packed-prefix-ExplicitInplaceSiLUBackward", "ceonly_packed_prefix_explicit_inplace_silu_backward", role="official_kernel_native_s2_repair", dense="manual_purekan_packed_prefix_explicit_inplace_silu_backward"),
        "KF6": spec("KF6", "KF6-M13-CE-only-packed-prefix-ExplicitInplaceSiLUBackward-AdamWAddcdiv", "ceonly_packed_prefix_explicit_inplace_silu_backward_adamw_addcdiv", role="official_kernel_native_s2_repair", dense="manual_purekan_packed_prefix_explicit_inplace_silu_backward_adamw_addcdiv"),
        "KF7": spec("KF7", "KF7-M13-CE-only-packed-prefix-AtenUpperOnlySiLUBackward", "ceonly_packed_prefix_aten_upper_only_silu_backward", role="official_kernel_native_s2_repair", dense="manual_purekan_packed_prefix_aten_upper_only_silu_backward"),
        "KF8": spec("KF8", "KF8-M13-CE-only-packed-prefix-AtenUpperOnlySiLUBackward-AdamWAddcdiv", "ceonly_packed_prefix_aten_upper_only_silu_backward_adamw_addcdiv", role="official_kernel_native_s2_repair", dense="manual_purekan_packed_prefix_aten_upper_only_silu_backward_adamw_addcdiv"),
        "KF9": spec("KF9", "KF9-M13-CE-only-packed-prefix-AtenLowerOnlySiLUBackward", "ceonly_packed_prefix_aten_lower_only_silu_backward", role="official_kernel_native_s2_repair", dense="manual_purekan_packed_prefix_aten_lower_only_silu_backward"),
        "KF10": spec("KF10", "KF10-M13-CE-only-packed-prefix-AtenLowerOnlySiLUBackward-AdamWAddcdiv", "ceonly_packed_prefix_aten_lower_only_silu_backward_adamw_addcdiv", role="official_kernel_native_s2_repair", dense="manual_purekan_packed_prefix_aten_lower_only_silu_backward_adamw_addcdiv"),
        "KW1": spec("KW1", "KW1-M13-CE-only-packed-prefix-no-input-grad-recompute", "ceonly_packed_prefix_no_input_grad_recompute", role="official_stack_workspace_trim", dense="manual_purekan_packed_prefix_no_input_grad_recompute"),
        "KW2": spec("KW2", "KW2-M13-CE-only-packed-prefix-no-input-grad-recompute-AdamWAddcdiv", "ceonly_packed_prefix_no_input_grad_recompute_adamw_addcdiv", role="official_stack_workspace_trim", dense="manual_purekan_packed_prefix_no_input_grad_recompute_adamw_addcdiv"),
        "KW3": spec("KW3", "KW3-M13-CE-only-packed-prefix-no-input-grad-AtenLowerOnlySiLUBackward-AdamWAddcdiv", "ceonly_packed_prefix_no_input_grad_aten_lower_only_silu_backward_adamw_addcdiv", role="official_stack_workspace_trim", dense="manual_purekan_packed_prefix_no_input_grad_aten_lower_only_silu_backward_adamw_addcdiv"),
        "KW4": spec("KW4", "KW4-M13-CE-only-packed-prefix-cached-no-input-grad-AtenLowerOnlySiLUBackward-AdamWAddcdiv", "ceonly_packed_prefix_cached_no_input_grad_aten_lower_only_silu_backward_adamw_addcdiv", role="official_stack_step_repair", dense="manual_purekan_packed_prefix_cached_no_input_grad_aten_lower_only_silu_backward_adamw_addcdiv"),
        "KW5": spec("KW5", "KW5-M13-CE-only-packed-prefix-compiled-explicit-no-input-grad-AtenLowerOnlySiLUBackward-AdamWAddcdiv", "ceonly_packed_prefix_compiled_explicit_no_input_grad_aten_lower_only_silu_backward_adamw_addcdiv", role="official_stack_step_repair", dense="manual_purekan_packed_prefix_compiled_explicit_no_input_grad_aten_lower_only_silu_backward_adamw_addcdiv"),
        "KW6": spec("KW6", "KW6-M13-CE-only-packed-prefix-fast-mix-view-no-input-grad-AtenLowerOnlySiLUBackward-AdamWAddcdiv", "ceonly_packed_prefix_fast_mix_view_no_input_grad_aten_lower_only_silu_backward_adamw_addcdiv", role="official_stack_step_repair", dense="manual_purekan_packed_prefix_fast_mix_view_no_input_grad_aten_lower_only_silu_backward_adamw_addcdiv"),
    }
    rows["SB0"] = spec(
        "SB0",
        "SB0-M13-EMA-self-bootstrap-T2-alpha025-no-external",
        "self_bootstrap_ema",
        self_teacher=1,
        self_type="self_bootstrap_ema",
        pure_supervised=0,
    )
    rows["SB1"] = spec(
        "SB1",
        "SB1-M13-delayed-snapshot-self-bootstrap-T2-alpha025-no-external",
        "self_bootstrap_snapshot",
        self_teacher=1,
        self_type="self_bootstrap_snapshot",
        pure_supervised=0,
    )
    rows["SB2"] = spec(
        "SB2",
        "SB2-M13-bestval-snapshot-self-bootstrap-T2-alpha025-no-external",
        "self_bootstrap_bestval_snapshot",
        self_teacher=1,
        self_type="self_bootstrap_bestval_snapshot",
        pure_supervised=0,
    )
    rows["SB3"] = spec(
        "SB3",
        "SB3-M13-dual-view-consistency-self-bootstrap-T2-alpha025-no-external",
        "self_bootstrap_dual_view",
        self_teacher=1,
        self_type="self_bootstrap_dual_view",
        pure_supervised=0,
    )
    rows["SYS1"] = spec(
        "SYS1",
        "SYS1-SB2-recompute-cache-trim-bestval-self-bootstrap-no-external",
        "system_recompute_cache_trim_bestval_self_bootstrap",
        role="official_system_repair",
        dense="manual_purekan_recompute_cache_trim",
        self_teacher=1,
        self_type="self_bootstrap_bestval_snapshot",
        pure_supervised=0,
    )
    rows["SYS2"] = spec(
        "SYS2",
        "SYS2-SYS1-SGD-state-trim-bestval-self-bootstrap-no-external",
        "system_sgd_state_trim_bestval_self_bootstrap",
        role="official_system_repair",
        dense="manual_purekan_recompute_cache_trim",
        self_teacher=1,
        self_type="self_bootstrap_bestval_snapshot",
        pure_supervised=0,
    )
    rows["SYS3"] = spec(
        "SYS3",
        "SYS3-SYS1-RMS-one-state-trim-bestval-self-bootstrap-no-external",
        "system_rms_one_state_trim_bestval_self_bootstrap",
        role="official_system_repair",
        dense="manual_purekan_recompute_cache_trim",
        self_teacher=1,
        self_type="self_bootstrap_bestval_snapshot",
        pure_supervised=0,
    )
    rows["SYS4"] = spec(
        "SYS4",
        "SYS4-SYS1-AdamWHalfState-trim-bestval-self-bootstrap-no-external",
        "system_adamw_half_state_trim_bestval_self_bootstrap",
        role="official_system_repair",
        dense="manual_purekan_recompute_cache_trim",
        self_teacher=1,
        self_type="self_bootstrap_bestval_snapshot",
        pure_supervised=0,
    )
    rows["SYS5"] = spec(
        "SYS5",
        "SYS5-SYS1-AdamWVHalfState-trim-bestval-self-bootstrap-no-external",
        "system_adamw_v_half_state_trim_bestval_self_bootstrap",
        role="official_system_repair",
        dense="manual_purekan_recompute_cache_trim",
        self_teacher=1,
        self_type="self_bootstrap_bestval_snapshot",
        pure_supervised=0,
    )
    rows["SYS8"] = spec(
        "SYS8",
        "SYS8-SYS1-AdamWVBF16State-trim-bestval-self-bootstrap-no-external",
        "system_adamw_v_bfloat_state_trim_bestval_self_bootstrap",
        role="official_system_repair",
        dense="manual_purekan_recompute_cache_trim",
        self_teacher=1,
        self_type="self_bootstrap_bestval_snapshot",
        pure_supervised=0,
    )
    rows["SYS9"] = spec(
        "SYS9",
        "SYS9-SYS1-AdamWVBF16NativeDenomState-trim-bestval-self-bootstrap-no-external",
        "system_adamw_v_bfloat_native_denom_state_trim_bestval_self_bootstrap",
        role="official_system_repair",
        dense="manual_purekan_recompute_cache_trim",
        self_teacher=1,
        self_type="self_bootstrap_bestval_snapshot",
        pure_supervised=0,
    )
    rows["SYS10"] = spec(
        "SYS10",
        "SYS10-SYS1-AdamWBF16State-trim-bestval-self-bootstrap-no-external",
        "system_adamw_bfloat_state_trim_bestval_self_bootstrap",
        role="official_system_repair",
        dense="manual_purekan_recompute_cache_trim",
        self_teacher=1,
        self_type="self_bootstrap_bestval_snapshot",
        pure_supervised=0,
    )
    rows["SYS11"] = spec(
        "SYS11",
        "SYS11-SYS1-AdamWVBF16AddcdivUpdate-trim-bestval-self-bootstrap-no-external",
        "system_adamw_v_bfloat_addcdiv_update_trim_bestval_self_bootstrap",
        role="official_system_repair",
        dense="manual_purekan_recompute_cache_trim",
        self_teacher=1,
        self_type="self_bootstrap_bestval_snapshot",
        pure_supervised=0,
    )
    rows["SYS12"] = spec(
        "SYS12",
        "SYS12-SYS1-AdamWHeadBF16VBF16AddcdivUpdate-trim-bestval-self-bootstrap-no-external",
        "system_adamw_head_bfloat_v_bfloat_addcdiv_update_trim_bestval_self_bootstrap",
        role="official_system_repair",
        dense="manual_purekan_recompute_cache_trim",
        self_teacher=1,
        self_type="self_bootstrap_bestval_snapshot",
        pure_supervised=0,
    )
    rows["SYS13"] = spec(
        "SYS13",
        "SYS13-PackedRecompute-AdamWVBF16State-bestval-self-bootstrap-no-external",
        "system_packed_recompute_adamw_v_bfloat_state_bestval_self_bootstrap",
        role="official_system_repair",
        dense="manual_purekan_packed_recompute_cache_trim",
        self_teacher=1,
        self_type="self_bootstrap_bestval_snapshot",
        pure_supervised=0,
    )
    rows["SYS14"] = spec(
        "SYS14",
        "SYS14-SYS1-AdamWStackBF16VBF16AddcdivUpdate-trim-bestval-self-bootstrap-no-external",
        "system_adamw_stack_bfloat_v_bfloat_addcdiv_update_trim_bestval_self_bootstrap",
        role="official_system_repair",
        dense="manual_purekan_recompute_cache_trim",
        self_teacher=1,
        self_type="self_bootstrap_bestval_snapshot",
        pure_supervised=0,
    )
    rows["SYS15"] = spec(
        "SYS15",
        "SYS15-SYS1-AdamWTailBF16VBF16State-trim-bestval-self-bootstrap-no-external",
        "system_adamw_tail_bfloat_v_bfloat_state_trim_bestval_self_bootstrap",
        role="official_system_repair",
        dense="manual_purekan_recompute_cache_trim",
        self_teacher=1,
        self_type="self_bootstrap_bestval_snapshot",
        pure_supervised=0,
    )
    rows["SYS16"] = spec(
        "SYS16",
        "SYS16-SYS1-AdamWL0BF16VBF16State-trim-bestval-self-bootstrap-no-external",
        "system_adamw_l0_bfloat_v_bfloat_state_trim_bestval_self_bootstrap",
        role="official_system_repair",
        dense="manual_purekan_recompute_cache_trim",
        self_teacher=1,
        self_type="self_bootstrap_bestval_snapshot",
        pure_supervised=0,
    )
    rows["SYS17"] = spec(
        "SYS17",
        "SYS17-SYS1-AdamWL0BF16VBF16AddcdivUpdate-trim-bestval-self-bootstrap-no-external",
        "system_adamw_l0_bfloat_v_bfloat_addcdiv_update_trim_bestval_self_bootstrap",
        role="official_system_repair",
        dense="manual_purekan_recompute_cache_trim",
        self_teacher=1,
        self_type="self_bootstrap_bestval_snapshot",
        pure_supervised=0,
    )
    rows["SYS18"] = spec(
        "SYS18",
        "SYS18-SYS17-alpha010-bestval-self-bootstrap-no-external",
        "system_adamw_l0_bfloat_v_bfloat_addcdiv_alpha010",
        role="official_system_repair",
        dense="manual_purekan_recompute_cache_trim",
        self_teacher=1,
        self_type="self_bootstrap_bestval_snapshot_alpha010",
        pure_supervised=0,
    )
    rows["SYS19"] = spec(
        "SYS19",
        "SYS19-SYS17-alpha050-bestval-self-bootstrap-no-external",
        "system_adamw_l0_bfloat_v_bfloat_addcdiv_alpha050",
        role="official_system_repair",
        dense="manual_purekan_recompute_cache_trim",
        self_teacher=1,
        self_type="self_bootstrap_bestval_snapshot_alpha050",
        pure_supervised=0,
    )
    rows["SYS20"] = spec(
        "SYS20",
        "SYS20-SYS13-packed-alpha010-bestval-self-bootstrap-no-external",
        "system_packed_recompute_adamw_v_bfloat_alpha010",
        role="official_system_repair",
        dense="manual_purekan_packed_recompute_cache_trim",
        self_teacher=1,
        self_type="self_bootstrap_bestval_snapshot_alpha010",
        pure_supervised=0,
    )
    rows["SYS21"] = spec(
        "SYS21",
        "SYS21-SYS13-packed-alpha050-bestval-self-bootstrap-no-external",
        "system_packed_recompute_adamw_v_bfloat_alpha050",
        role="official_system_repair",
        dense="manual_purekan_packed_recompute_cache_trim",
        self_teacher=1,
        self_type="self_bootstrap_bestval_snapshot_alpha050",
        pure_supervised=0,
    )
    rows["SYS22"] = spec(
        "SYS22",
        "SYS22-SYS13-packed-cachey-alpha010-bestval-self-bootstrap-no-external",
        "system_packed_cachey_adamw_v_bfloat_alpha010",
        role="official_system_repair",
        dense="manual_purekan_packed_cache_y",
        self_teacher=1,
        self_type="self_bootstrap_bestval_snapshot_alpha010",
        pure_supervised=0,
    )
    rows["SYS23"] = spec(
        "SYS23",
        "SYS23-SYS13-packed-alpha005-bestval-self-bootstrap-no-external",
        "system_packed_recompute_adamw_v_bfloat_alpha005",
        role="official_system_repair",
        dense="manual_purekan_packed_recompute_cache_trim",
        self_teacher=1,
        self_type="self_bootstrap_bestval_snapshot_alpha005",
        pure_supervised=0,
    )
    rows["SYS24"] = spec(
        "SYS24",
        "SYS24-SYS13-packed-prefix-recompute-alpha005-bestval-self-bootstrap-no-external",
        "system_packed_prefix_recompute_adamw_v_bfloat_alpha005",
        role="official_system_repair",
        dense="manual_purekan_packed_prefix_recompute_cache_trim",
        self_teacher=1,
        self_type="self_bootstrap_bestval_snapshot_alpha005",
        pure_supervised=0,
    )
    for cid, family in {
        "SB4": "self_bootstrap_swa_logit_smoothing",
        "SB5": "self_bootstrap_previous_epoch",
        "SB6": "self_bootstrap_temperature2",
        "SB7": "self_bootstrap_temperature4",
        "RR1": "representation_sparseinterp",
        "RR2": "representation_sparsespline",
        "RR5": "representation_gemm_depthwise_mix",
    }.items():
        rows[cid] = spec(
            cid,
            f"{cid}-{family}",
            family,
            role="planned_no_external",
            self_teacher=1 if cid.startswith("SB") else 0,
            self_type=family if cid.startswith("SB") else "none",
            official=0 if cid.startswith("SYS") else 1,
            pure_supervised=0 if cid.startswith("SB") else 1,
            status="not_implemented",
        )
    return rows


def _write_v80_postprocess(out_dir: Path, args: argparse.Namespace) -> Dict[str, Any]:
    registry = _spec_v2_registry(args)
    measured_task = _read_csv_rows(out_dir / "p9_task_summary.csv")
    task_trace = _read_csv_rows(out_dir / "p9_task_trace.csv")
    macro_rows = _read_csv_rows(out_dir / "p1_significance_audit.csv")
    grad_rows = _read_csv_rows(out_dir / "p2_full_gradient_correctness.csv")
    eff_rows = _read_csv_rows(out_dir / "p10_efficiency_profiler.csv")
    p7_rows = _read_csv_rows(out_dir / "p7_time_auc_v76.csv")
    v79_route = json.loads((out_dir / "v79_route_decision.json").read_text(encoding="utf-8")) if (out_dir / "v79_route_decision.json").exists() else {}

    measured_ids = sorted({str(r.get("candidate_id")) for r in measured_task})
    registry_rows = []
    for cid, spec in registry.items():
        row = asdict(spec)
        row.update({
            "stage": "P0_CANDIDATE_SPEC_V2",
            "measured_in_run": int(cid in measured_ids),
            "constructed_from_spec": int(spec.implementation_status == "implemented"),
            "fake_data_used": 0,
            "proxy_row_used": 0,
        })
        registry_rows.append(row)
    write_csv(out_dir / "candidate_registry_v2.csv", registry_rows)
    write_csv(out_dir / "implementation_status_registry.csv", registry_rows)

    summary_by = {str(r.get("candidate_id")): r for r in measured_task}
    macro_by = {str(r.get("candidate_id")): r for r in macro_rows if str(r.get("dataset")) == "macro"}
    grad_by: Dict[str, Dict[str, Any]] = {}
    for cid in measured_ids:
        items = [r for r in grad_rows if str(r.get("candidate_id")) == cid]
        rels = [_float(r.get("grad_relerr_max")) for r in items if _finite(r.get("grad_relerr_max"))]
        coss = [_float(r.get("grad_cos_min")) for r in items if _finite(r.get("grad_cos_min"))]
        grad_by[cid] = {
            "grad_pass": int(bool(items) and all(_is_one(r.get("grad_pass")) for r in items)),
            "grad_relerr_max": max(rels) if rels else METRIC_UNAVAILABLE,
            "grad_cos_min": min(coss) if coss else METRIC_UNAVAILABLE,
            "grad_rows": len(items),
        }

    def eff(cid: str) -> Dict[str, Any]:
        items = [r for r in eff_rows if str(r.get("candidate_id")) == cid]
        mem = [_float(r.get("memory_ratio")) for r in items if _finite(r.get("memory_ratio"))]
        step = [_float(r.get("step_ratio")) for r in items if _finite(r.get("step_ratio"))]
        datasets = sorted({str(r.get("dataset")) for r in items if r.get("dataset") not in (None, "")})
        batches = sorted({_int(r.get("batch_size"), -1) for r in items if _int(r.get("batch_size"), -1) > 0})
        fullgrid_complete = int(set(datasets) >= {"MNIST", "Fashion-MNIST", "KMNIST"} and {128, 256, 512}.issubset(set(batches)))
        measured_s2 = int(bool(items) and all(_float(r.get("memory_ratio"), 99) <= 1.05 and _float(r.get("step_ratio"), 99) <= 1.50 for r in items))
        measured_s1 = int(bool(items) and all(_float(r.get("memory_ratio"), 99) < 1.00 and _float(r.get("step_ratio"), 99) <= 1.35 for r in items))
        return {
            "eff_shape_count": len(items),
            "eff_dataset_count": len(datasets),
            "eff_batch_sizes": ",".join(str(b) for b in batches) if batches else METRIC_UNAVAILABLE,
            "fullgrid_shape_complete": fullgrid_complete,
            "memory_ratio_mean": _mean(mem),
            "memory_ratio_max": max(mem) if mem else METRIC_UNAVAILABLE,
            "step_ratio_mean": _mean(step),
            "step_ratio_max": max(step) if step else METRIC_UNAVAILABLE,
            "s2_shape_count": sum(1 for r in items if _float(r.get("memory_ratio"), 99) <= 1.05 and _float(r.get("step_ratio"), 99) <= 1.50),
            "s1_shape_count": sum(1 for r in items if _float(r.get("memory_ratio"), 99) < 1.00 and _float(r.get("step_ratio"), 99) <= 1.35),
            "measured_s2_pass": measured_s2,
            "measured_s1_pass": measured_s1,
        }

    def time_auc(cid: str) -> Dict[str, Any]:
        if p7_rows:
            c_step = _mean(r.get("val_loss_auc_step") for r in p7_rows if str(r.get("candidate_id")) == cid)
            b_step = _mean(r.get("val_loss_auc_step") for r in p7_rows if str(r.get("candidate_id")) == "B0")
            c_time = _mean(r.get("val_loss_auc_time") for r in p7_rows if str(r.get("candidate_id")) == cid)
            b_time = _mean(r.get("val_loss_auc_time") for r in p7_rows if str(r.get("candidate_id")) == "B0")
        else:
            c_step = _mean(r.get("val_loss_auc_step") for r in measured_task if str(r.get("candidate_id")) == cid)
            b_step = _mean(r.get("val_loss_auc_step") for r in measured_task if str(r.get("candidate_id")) == "B0")
            c_time = _mean(r.get("val_loss_auc_time") for r in measured_task if str(r.get("candidate_id")) == cid)
            b_time = _mean(r.get("val_loss_auc_time") for r in measured_task if str(r.get("candidate_id")) == "B0")
        return {
            "val_loss_auc_step": c_step if _finite(c_step) else METRIC_UNAVAILABLE,
            "val_loss_auc_time": c_time if _finite(c_time) else METRIC_UNAVAILABLE,
            "val_loss_auc_step_ratio_vs_B0": c_step / b_step if _finite(c_step) and _finite(b_step) and b_step else METRIC_UNAVAILABLE,
            "val_loss_auc_time_ratio_vs_B0": c_time / b_time if _finite(c_time) and _finite(b_time) and b_time else METRIC_UNAVAILABLE,
            "time_auc_pass": int(_finite(c_time) and _finite(b_time) and c_time <= b_time),
        }

    factory_rows = []
    contract_rows = []
    route_rows = []
    for cid in sorted(measured_ids):
        spec = registry.get(cid)
        summ = summary_by.get(cid, {})
        macro = macro_by.get(cid, {})
        grad = grad_by.get(cid, {})
        e = eff(cid)
        t = time_auc(cid)
        metadata_present = int(spec is not None)
        official = int(bool(spec and spec.official_eligible and spec.no_external_teacher_eligible and spec.implementation_status == "implemented"))
        strict = int(
            official
            and _is_one(summ.get("head_is_kan"))
            and str(summ.get("non_kan_trainable_param_count")) in {"0", "0.0"}
            and _is_one(summ.get("manual_forward"))
            and _is_one(summ.get("manual_backward"))
            and _is_one(summ.get("manual_update"))
        )
        macro_gap = _float(macro.get("mean_val_gap"), 0.0)
        ci95_low = macro.get("ci95_low", macro.get("bootstrap_ci95_low", METRIC_UNAVAILABLE))
        holm_p = macro.get("holm_p", macro.get("holm_corrected_p", METRIC_UNAVAILABLE))
        ece_delta = macro.get("ECE_delta", macro.get("ECE_delta_macro", METRIC_UNAVAILABLE))
        nll_delta = macro.get("NLL_delta", macro.get("NLL_delta_macro", METRIC_UNAVAILABLE))
        macro_pass = int(official and macro_gap >= 0.0200 and _float(ci95_low, -1.0) > 0.0 and _float(holm_p, 1.0) < 0.05 and _float(macro.get("mean_test_gap"), 0.0) >= 0.015)
        full_s2 = int(_is_one(e.get("fullgrid_shape_complete")) and _is_one(e.get("measured_s2_pass")))
        full_s1 = int(_is_one(e.get("fullgrid_shape_complete")) and _is_one(e.get("measured_s1_pass")))
        factory_rows.append({
            "stage": "P0_CANDIDATE_FACTORY_AUDIT",
            "candidate_id": cid,
            "metadata_present": metadata_present,
            "constructed_from_spec": int(bool(spec and spec.implementation_status == "implemented")),
            "implementation_status": spec.implementation_status if spec else "missing_metadata",
            "dense_cls_name": spec.dense_cls_name if spec else METRIC_UNAVAILABLE,
            "dense_cls_importable": int(bool(spec and spec.implementation_status == "implemented")),
            "measured_task_rows": int(summ.get("rows", 0) or 0),
            "fake_data_used": 0,
            "proxy_row_used": 0,
        })
        contract_rows.append({
            "stage": "P0_CONTRACT_VALIDATOR_V2",
            "candidate_id": cid,
            "official_eligible": official,
            "external_teacher_used": spec.external_teacher_used if spec else METRIC_UNAVAILABLE,
            "self_teacher_used": spec.self_teacher_used if spec else METRIC_UNAVAILABLE,
            "head_is_kan": summ.get("head_is_kan", METRIC_UNAVAILABLE),
            "manual_forward": summ.get("manual_forward", METRIC_UNAVAILABLE),
            "manual_backward": summ.get("manual_backward", METRIC_UNAVAILABLE),
            "manual_update": summ.get("manual_update", METRIC_UNAVAILABLE),
            "nonKAN_param_count": summ.get("non_kan_trainable_param_count", METRIC_UNAVAILABLE),
            "strict_pass_v2": strict,
            "fake_data_used": 0,
            "proxy_row_used": 0,
        })
        route_rows.append({
            "stage": "P13_ROUTE_INPUT_V80",
            "candidate_id": cid,
            "route_role": spec.route_role if spec else "missing_metadata",
            "external_teacher_used": spec.external_teacher_used if spec else METRIC_UNAVAILABLE,
            "self_teacher_used": spec.self_teacher_used if spec else METRIC_UNAVAILABLE,
            "official_eligible": official,
            "strict_pass": strict,
            "grad_pass": grad.get("grad_pass", METRIC_UNAVAILABLE),
            "grad_relerr_max": grad.get("grad_relerr_max", METRIC_UNAVAILABLE),
            "grad_cos_min": grad.get("grad_cos_min", METRIC_UNAVAILABLE),
            "teacher_free_macro_pass": macro_pass,
            "macro_gap": macro.get("mean_val_gap", 0.0),
            "ci95_low": ci95_low,
            "holm_p": holm_p,
            "test_gap": macro.get("mean_test_gap", METRIC_UNAVAILABLE),
            "ECE_delta": ece_delta,
            "NLL_delta": nll_delta,
            **e,
            "fullgrid_s2_pass": full_s2,
            "fullgrid_s1_pass": full_s1,
            **t,
            "profiler_pass": 0,
            "fake_data_used": 0,
            "proxy_row_used": 0,
        })

    write_csv(out_dir / "candidate_factory_audit.csv", factory_rows)
    write_csv(out_dir / "teacher_contract_v2.csv", [asdict(spec) | {
        "stage": "P0_TEACHER_CONTRACT_V2",
        "teacher_contract_pass": int((not spec.official_eligible) or (spec.external_teacher_used == 0 and spec.external_teacher_logits_used == 0 and spec.external_teacher_forward_used == 0)),
        "fake_data_used": 0,
        "proxy_row_used": 0,
    } for spec in registry.values()])
    write_csv(out_dir / "contract_validator_v2.csv", contract_rows)
    measured_join = int(all(row["metadata_present"] for row in factory_rows))
    write_csv(out_dir / "artifact_join_coverage.csv", [{
        "stage": "P0_ARTIFACT_JOIN_COVERAGE",
        "measured_candidate_count": len(measured_ids),
        "registry_joined_count": sum(1 for row in factory_rows if row["metadata_present"]),
        "join_coverage": 1.0 if measured_join else (sum(1 for row in factory_rows if row["metadata_present"]) / max(1, len(factory_rows))),
        "candidate_factory_pass": measured_join,
        "fake_data_used": 0,
        "proxy_row_used": 0,
    }])
    write_csv(out_dir / "p1_reproduction_evidence_map.csv", route_rows)
    write_csv(out_dir / "p4_self_bootstrap.csv", [{
        "stage": "P4_SELF_BOOTSTRAP",
        "candidate_id": cid,
        "implementation_status": registry[cid].implementation_status,
        "status": registry[cid].implementation_status,
        "external_teacher_used": registry[cid].external_teacher_used,
        "self_teacher_used": registry[cid].self_teacher_used,
        "fake_data_used": 0,
        "proxy_row_used": 0,
    } for cid in ["SB0", "SB1", "SB2", "SB3", "SB4", "SB5", "SB6", "SB7"]])
    write_csv(out_dir / "p5_teacher_free_repair.csv", [row for row in route_rows if str(row.get("candidate_id")) in {"TF7", "TF8", "M13"}])
    write_csv(out_dir / "p6_representation_repair.csv", [row for row in route_rows if str(row.get("candidate_id")) in {"A2S", "M9", "M13"}])
    write_csv(out_dir / "p8_phase_clean_profiler.csv", [{
        "stage": "P8_PHASE_CLEAN_PROFILER",
        "status": "not_run",
        "reason": "phase_clean_profiler_not_implemented_in_this_slice",
        "fake_data_used": 0,
        "proxy_row_used": 0,
    }])
    write_csv(out_dir / "p10_phase_mapped_kernel_profiler_v80.csv", [{
        "stage": "P10_PHASE_MAPPED_KERNEL_PROFILER",
        "status": "not_run",
        "reason": "phase_mapped_kernel_profiler_not_implemented",
        "fake_data_used": 0,
        "proxy_row_used": 0,
    }])
    write_csv(out_dir / "p12_fused_streaming_package_v80.csv", [{
        "stage": "P12_FUSED_STREAMING_PACKAGE",
        "status": "not_implemented",
        "reason": "no_teacher_free_macro_s2_candidate_available",
        "fake_data_used": 0,
        "proxy_row_used": 0,
    }])
    write_csv(out_dir / "p13_final_scorecard_v80.csv", route_rows)

    official_rows = [row for row in route_rows if _is_one(row.get("official_eligible")) and _is_one(row.get("strict_pass"))]
    official_rows.sort(key=lambda row: (
        int(_is_one(row.get("teacher_free_macro_pass"))),
        int(_is_one(row.get("fullgrid_s2_pass"))),
        _float(row.get("macro_gap"), -99.0),
    ), reverse=True)
    best = official_rows[0] if official_rows else {}
    code_native_pass = int(bool(factory_rows) and measured_join and all(row["constructed_from_spec"] for row in factory_rows))
    teacher_contract_pass = int(all((not spec.official_eligible) or (spec.external_teacher_used == 0 and spec.external_teacher_logits_used == 0 and spec.external_teacher_forward_used == 0) for spec in registry.values()))
    strict = int(_is_one(best.get("strict_pass")))
    grad = int(_is_one(best.get("grad_pass")))
    macro = int(_is_one(best.get("teacher_free_macro_pass")))
    s2 = int(_is_one(best.get("fullgrid_s2_pass")))
    s1 = int(_is_one(best.get("fullgrid_s1_pass")))
    time_auc = int(_is_one(best.get("time_auc_pass")))
    profiler = 0
    if macro and s1 and time_auc and profiler:
        route = "R1-TeacherFreeFullSystemAdvantage"
    elif macro and s2 and time_auc:
        route = "R2-TeacherFreeS2QualityAdvantage"
    elif macro and s2:
        route = "R2-TeacherFreeS2QualityAdvantage"
    elif any(_float(row.get("macro_gap"), 0.0) >= 0.018 for row in official_rows):
        route = "R5-TeacherFreeNearPass"
    elif _is_one((next((row for row in route_rows if row.get("candidate_id") == "M12"), {}) or {}).get("teacher_free_macro_pass")):
        route = "R4-TeacherAssistedOnly"
    else:
        route = "R7-NoReproduction"
    success_min = int(code_native_pass and teacher_contract_pass and strict and grad and macro and s2)
    success_formal = int(success_min and s1 and time_auc and profiler)
    route_json = {
        "route": route,
        "best_official_candidate_id": best.get("candidate_id", METRIC_UNAVAILABLE),
        "external_teacher_used": best.get("external_teacher_used", METRIC_UNAVAILABLE),
        "self_teacher_used": best.get("self_teacher_used", METRIC_UNAVAILABLE),
        "code_native_pass": code_native_pass,
        "teacher_contract_v2_pass": teacher_contract_pass,
        "strict_pass": strict,
        "grad_pass": grad,
        "teacher_free_macro_pass": macro,
        "fullgrid_s2_pass": s2,
        "fullgrid_s1_pass": s1,
        "time_auc_pass": time_auc,
        "profiler_pass": profiler,
        "success_v80_minimum": success_min,
        "success_v80_formal": success_formal,
        "macro_gap": best.get("macro_gap", METRIC_UNAVAILABLE),
        "ci95_low": best.get("ci95_low", METRIC_UNAVAILABLE),
        "holm_p": best.get("holm_p", METRIC_UNAVAILABLE),
        "test_gap": best.get("test_gap", METRIC_UNAVAILABLE),
        "memory_ratio_max": best.get("memory_ratio_max", METRIC_UNAVAILABLE),
        "step_ratio_max": best.get("step_ratio_max", METRIC_UNAVAILABLE),
        "fullgrid_shape_complete": best.get("fullgrid_shape_complete", METRIC_UNAVAILABLE),
        "measured_s2_pass": best.get("measured_s2_pass", METRIC_UNAVAILABLE),
        "measured_s1_pass": best.get("measured_s1_pass", METRIC_UNAVAILABLE),
        "v79_route": v79_route.get("route", METRIC_UNAVAILABLE),
        "primary_blocker": (
            "teacher_free_macro_gap_not_closed"
            if not macro
            else ("fullgrid_s2_not_closed" if not s2 else ("s1_time_auc_profiler_not_closed" if not success_formal else "none"))
        ),
        "next_required_implementation": (
            "real_self_bootstrap_or_core_primitive_candidate_factory"
            if not macro
            else ("s2_memory_cache_trim_or_new_system_package" if not s2 else "phase_mapped_profiler_and_s1_timeauc")
        ),
        "no_fake": True,
        "no_proxy": True,
    }
    save_json(out_dir / "v80_route_decision.json", route_json)
    save_json(out_dir / "aggregate_decision_v80.json", route_json)

    failures = []
    if not code_native_pass:
        failures.append("F0_code_native_factory_fail")
    if not teacher_contract_pass:
        failures.append("F1_teacher_contract_fail")
    if not macro:
        failures.append("F2_teacher_free_macro_fail")
    if not s2:
        failures.append("F3_fullgrid_s2_fail")
    if not time_auc:
        failures.append("F4_time_auc_fail")
    if not profiler:
        failures.append("F5_profiler_not_closed")
    write_csv(out_dir / "failure_table_v80.csv", [{
        "stage": "P13_FAILURE_AUDIT",
        "failure_type": f,
        "count": 1,
        "fake_data_used": 0,
        "proxy_row_used": 0,
    } for f in failures] or [{
        "stage": "P13_FAILURE_AUDIT",
        "failure_type": "none",
        "count": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
    }])

    artifact_paths = [
        out_dir / "candidate_registry_v2.csv",
        out_dir / "candidate_factory_audit.csv",
        out_dir / "teacher_contract_v2.csv",
        out_dir / "contract_validator_v2.csv",
        out_dir / "artifact_join_coverage.csv",
        out_dir / "implementation_status_registry.csv",
        out_dir / "p1_reproduction_evidence_map.csv",
        out_dir / "p4_self_bootstrap.csv",
        out_dir / "p5_teacher_free_repair.csv",
        out_dir / "p6_representation_repair.csv",
        out_dir / "p8_phase_clean_profiler.csv",
        out_dir / "p10_phase_mapped_kernel_profiler_v80.csv",
        out_dir / "p12_fused_streaming_package_v80.csv",
        out_dir / "p13_final_scorecard_v80.csv",
        out_dir / "failure_table_v80.csv",
        out_dir / "v80_route_decision.json",
        out_dir / "p9_task_summary.csv",
        out_dir / "p9_task_trace.csv",
        out_dir / "p1_significance_audit.csv",
        out_dir / "p2_full_gradient_correctness.csv",
        out_dir / "p10_efficiency_profiler.csv",
        out_dir / "p7_time_auc_v76.csv",
    ]
    audit = v72._audit_fake_proxy(artifact_paths)
    write_csv(out_dir / "v80_provenance_audit.csv", [{
        **audit,
        "plan_path": PLAN_PATH,
        "script_path": SCRIPT_PATH,
        "fake_data_used": 0,
        "proxy_row_used": 0,
    }])
    save_json(out_dir / "v80_manifest.json", {
        "plan_path": PLAN_PATH,
        "script_path": SCRIPT_PATH,
        "runner_reuse": "run_gafu_v79_real.py + v8.0 CandidateSpecV2 postprocess",
        "out_dir": str(out_dir),
        "source_commit": _git_commit(),
        "git_status": _git_status(),
        "datasets": parse_str_list(args.datasets),
        "seeds": parse_int_list(args.seeds),
        "candidates": parse_str_list(args.candidates),
        "bench_batch_sizes": parse_int_list(args.bench_batch_sizes),
        "grad_batch_sizes": parse_int_list(args.grad_batch_sizes),
        "postprocess_artifact_hashes": {p.name: _hash_file(p) for p in artifact_paths if p.exists()},
        "v80_route": route_json,
        "v80_audit": audit,
    })
    return route_json


def _patch_for_v80() -> None:
    v79.PLAN_PATH = PLAN_PATH
    v79.SCRIPT_PATH = SCRIPT_PATH
    v76.PLAN_PATH = PLAN_PATH
    v76.SCRIPT_PATH = SCRIPT_PATH
    v76._spec_map_v76 = _spec_map_v80
    v76._candidate_registry_v76 = _candidate_registry_v80
    v76._init_seed_candidate_id_v76 = _init_seed_candidate_id_v80
    v76._uses_distill_v76 = _uses_distill_v80
    v76._make_manual_candidate_v76 = _make_manual_candidate_v80
    v73._train_manual_v73 = _train_manual_v80
    v73._gradient_check_v73 = _gradient_check_v80


def run(args: argparse.Namespace) -> None:
    _patch_for_v80()
    v79.run(args)
    out_dir = Path(args.out_dir)
    _write_v80_postprocess(out_dir, args)
    for src_name, dst_name in {
        "v79_manifest.json": "v80_reused_v79_manifest.json",
        "v79_route_decision.json": "v80_reused_v79_route_decision.json",
        "v79_provenance_audit.csv": "v80_reused_v79_provenance_audit.csv",
    }.items():
        src = out_dir / src_name
        if src.exists():
            shutil.copyfile(src, out_dir / dst_name)


def parse_args() -> argparse.Namespace:
    return v79.parse_args()


if __name__ == "__main__":
    run(parse_args())
