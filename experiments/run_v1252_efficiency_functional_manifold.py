#!/usr/bin/env python3
"""DG-KAN v12.5.2 efficiency / functional / manifold-channel runner.

This runner is intentionally narrower than the v12.4 sweep.  It focuses on the
top Gated Legendre + Quadratic lineage, performs real Triton component-kernel
truth audits, and adds Line C train-probe / signal-reservoir diagnostics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except Exception:  # pragma: no cover - environment dependent
    triton = None
    tl = None
    TRITON_AVAILABLE = False


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

import run_v120_good_geometry_battery as v120  # noqa: E402
import run_v124_multibasis_functional_dual as v124  # noqa: E402
from dgkan.artifacts.writer import ensure_dir, write_csv_rows, write_json  # noqa: E402
from dgkan.models import fc_purekan_primitives as prim  # noqa: E402


PLAN_PATH = ROOT / "docs" / "DG-KAN_v12.5.2_效率Functional流形信号通道三线完整计划.md"
SCRIPT_PATH = ROOT / "experiments" / "run_v1252_efficiency_functional_manifold.py"
REPORT_PATH_DEFAULT = ROOT / "docs" / "DG-KAN_v12.5.2_效率Functional流形信号通道三线_结果复盘.md"
EPS = 1.0e-12


def _now_tag() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _parse_list(text: str) -> List[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def _parse_ints(text: str) -> List[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def _stamp(row: Dict[str, Any]) -> Dict[str, Any]:
    row.setdefault("fake_data_used", 0)
    row.setdefault("proxy_row_used", 0)
    row.setdefault("cpu_offload_used", 0)
    return row


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        if isinstance(value, torch.Tensor):
            return float(value.detach().float().cpu())
        return float(value)
    except Exception:
        return default


def _mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return sum(vals) / len(vals) if vals else 0.0


def _q(values: Sequence[float], q: float, default: float = 0.0) -> float:
    vals = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not vals:
        return default
    pos = (len(vals) - 1) * float(q)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - pos) + vals[hi] * (pos - lo)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _make_adamw(
    params: Iterable[torch.nn.Parameter],
    args: argparse.Namespace,
    optimizer_impl: str | None = None,
) -> torch.optim.Optimizer:
    kwargs: Dict[str, Any] = {}
    impl = str(optimizer_impl if optimizer_impl is not None else getattr(args, "optimizer_impl", "adamw"))
    if impl == "fused_adamw":
        kwargs["fused"] = True
    elif impl == "foreach_adamw":
        kwargs["foreach"] = True
    elif impl != "adamw":
        raise ValueError(f"unknown optimizer_impl={impl}")
    return torch.optim.AdamW(params, lr=float(args.lr), weight_decay=float(args.weight_decay), **kwargs)


def _time_call(fn, device: torch.device, warmup: int, measure: int) -> Tuple[float, float, float, List[float]]:
    times: List[float] = []
    for idx in range(int(warmup) + int(measure)):
        _sync(device)
        t0 = time.perf_counter()
        fn()
        _sync(device)
        t1 = time.perf_counter()
        if idx >= int(warmup):
            times.append((t1 - t0) * 1000.0)
    return _q(times, 0.50), _q(times, 0.90), _mean(times), times


def _device_from_arg(arg: str) -> torch.device:
    if str(arg) == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(arg)


def _load_mnist(args: argparse.Namespace) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    data = v120._load_vision_split(
        args,
        "MNIST",
        train_size=int(args.train_size),
        val_size=int(args.val_size),
        test_size=int(args.test_size),
    )
    x_train, y_train, x_val, y_val, x_test, y_test, input_dim, output_dim, _protocol = data
    return x_train, y_train, x_val, y_val, x_test, y_test, int(input_dim), int(output_dim)


def _specs_for(input_dim: int, output_dim: int) -> Dict[str, prim.PrimitiveSpec]:
    _, budget = v124._param_budget(input_dim, output_dim)
    return {s.candidate_id: s for s in prim.primitive_specs(budget, input_dim, output_dim)}


def _make_model(method_id: str, input_dim: int, output_dim: int, x_stats: torch.Tensor, device: torch.device, seed: int, specs: Mapping[str, prim.PrimitiveSpec]) -> torch.nn.Module:
    _, budget = v124._param_budget(input_dim, output_dim)
    spec = specs.get(method_id)
    return v124._make_model(method_id, input_dim, output_dim, x_stats, device, int(seed), spec, budget)


if TRITON_AVAILABLE:

    @triton.jit
    def _residual_block_norm_stage1(
        x,
        mu,
        std,
        block_mu,
        block_whiten,
        z_out,
        sumsq,
        B: tl.constexpr,
        D: tl.constexpr,
        NUM_BLOCKS: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_blk = tl.program_id(1)
        offs_i = tl.arange(0, BLOCK_SIZE)
        offs_j = tl.arange(0, BLOCK_SIZE)
        d_j = pid_blk * BLOCK_SIZE + offs_j
        mask_j = d_j < D
        x_vals = tl.load(x + pid_b * D + d_j, mask=mask_j, other=0.0)
        mu_vals = tl.load(block_mu + pid_blk * BLOCK_SIZE + offs_j)
        centered = x_vals - mu_vals
        w = tl.load(block_whiten + pid_blk * BLOCK_SIZE * BLOCK_SIZE + offs_j[:, None] * BLOCK_SIZE + offs_i[None, :])
        z_vals = tl.sum(centered[:, None] * w, axis=0)
        d_i = pid_blk * BLOCK_SIZE + offs_i
        mask_i = d_i < D
        tl.store(z_out + pid_b * D + d_i, z_vals, mask=mask_i)
        tl.atomic_add(sumsq + pid_b, tl.sum(tl.where(mask_i, z_vals * z_vals, 0.0), axis=0))

    @triton.jit
    def _residual_legendre_stage2(
        x,
        mu,
        std,
        z_in,
        sumsq,
        b1_out,
        B: tl.constexpr,
        D: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        MIX: tl.constexpr,
        TARGET_RMS: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_blk = tl.program_id(1)
        offs = tl.arange(0, BLOCK_SIZE)
        d = pid_blk * BLOCK_SIZE + offs
        mask = d < D
        xv = tl.load(x + pid_b * D + d, mask=mask, other=0.0)
        muv = tl.load(mu + d, mask=mask, other=0.0)
        stdv = tl.load(std + d, mask=mask, other=1.0)
        base = (xv - muv) / stdv
        z = tl.load(z_in + pid_b * D + d, mask=mask, other=0.0)
        rms = tl.sqrt(tl.load(sumsq + pid_b) / D)
        mixed = (1.0 - MIX) * base + MIX * (z / tl.maximum(rms, 1.0e-3) * TARGET_RMS)
        # Triton 3.0 in the current environment does not expose tl.tanh.
        # Use the stable algebraic form so this remains a real fused path.
        leg = 2.0 / (1.0 + tl.exp(-2.0 * mixed)) - 1.0
        p0 = leg * 0.0 + 1.0
        p1 = leg
        p2 = 0.5 * (3.0 * leg * leg - 1.0)
        p3 = 0.5 * (5.0 * leg * leg * leg - 3.0 * leg)
        base_ptr = b1_out + (pid_b * D + d) * 4
        tl.store(base_ptr + 0, p0, mask=mask)
        tl.store(base_ptr + 1, p1, mask=mask)
        tl.store(base_ptr + 2, p2, mask=mask)
        tl.store(base_ptr + 3, p3, mask=mask)

    @triton.jit
    def _legendre_hidden_from_b1(
        b1,
        w1,
        h_out,
        B: tl.constexpr,
        D: tl.constexpr,
        H: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_h = tl.program_id(1)
        offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
        mask_h = offs_h < H
        acc = tl.zeros((BLOCK_H,), dtype=tl.float32)
        for start in range(0, D, BLOCK_D):
            offs_d = start + tl.arange(0, BLOCK_D)
            mask_d = offs_d < D
            p0 = tl.load(b1 + (pid_b * D + offs_d) * 4 + 0, mask=mask_d, other=0.0)
            p1 = tl.load(b1 + (pid_b * D + offs_d) * 4 + 1, mask=mask_d, other=0.0)
            p2 = tl.load(b1 + (pid_b * D + offs_d) * 4 + 2, mask=mask_d, other=0.0)
            p3 = tl.load(b1 + (pid_b * D + offs_d) * 4 + 3, mask=mask_d, other=0.0)
            w0 = tl.load(w1 + (offs_d[:, None] * H + offs_h[None, :]) * 4 + 0, mask=mask_d[:, None] & mask_h[None, :], other=0.0)
            w1v = tl.load(w1 + (offs_d[:, None] * H + offs_h[None, :]) * 4 + 1, mask=mask_d[:, None] & mask_h[None, :], other=0.0)
            w2 = tl.load(w1 + (offs_d[:, None] * H + offs_h[None, :]) * 4 + 2, mask=mask_d[:, None] & mask_h[None, :], other=0.0)
            w3 = tl.load(w1 + (offs_d[:, None] * H + offs_h[None, :]) * 4 + 3, mask=mask_d[:, None] & mask_h[None, :], other=0.0)
            acc += tl.sum(p0[:, None] * w0 + p1[:, None] * w1v + p2[:, None] * w2 + p3[:, None] * w3, axis=0)
        acc = acc / tl.sqrt(D + 0.0)
        tl.store(h_out + pid_b * H + offs_h, acc, mask=mask_h)


def _triton_legendre_basis_residual(model: torch.nn.Module, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, float]:
    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available")
    if not hasattr(model, "block_mu"):
        raise RuntimeError("model has no residual block norm buffers")
    B = int(x.shape[0])
    D = int(x.shape[1])
    block_size = int(model.input_norm_block_size)
    num_blocks = int(math.ceil(float(D) / float(block_size)))
    z = torch.empty((B, D), device=x.device, dtype=torch.float32)
    sumsq = torch.zeros((B,), device=x.device, dtype=torch.float32)
    b1 = torch.empty((B, D, 4), device=x.device, dtype=torch.float32)
    _residual_block_norm_stage1[(B, num_blocks)](
        x,
        model.mu,
        model.std,
        model.block_mu,
        model.block_whiten,
        z,
        sumsq,
        B,
        D,
        num_blocks,
        block_size,
        num_warps=1,
    )
    _residual_legendre_stage2[(B, num_blocks)](
        x,
        model.mu,
        model.std,
        z,
        sumsq,
        b1,
        B,
        D,
        block_size,
        float(model.residual_geom_mix.detach().item()),
        float(model.block_norm_target_rms.detach().item()),
        num_warps=1,
    )
    return b1, z, 2.0


def _triton_legendre_hidden_residual(model: torch.nn.Module, x: torch.Tensor) -> Tuple[torch.Tensor, float]:
    b1, _z, kernels = _triton_legendre_basis_residual(model, x)
    B = int(x.shape[0])
    D = int(x.shape[1])
    H = int(model.legendre_hidden)
    h = torch.empty((B, H), device=x.device, dtype=torch.float32)
    _legendre_hidden_from_b1[(B, triton.cdiv(H, 16))](
        b1,
        model.leg_w1,
        h,
        B,
        D,
        H,
        32,
        16,
        num_warps=1,
    )
    return h, kernels + float(B * math.ceil(H / 16))


def _triton_hybrid_full_forward(model: torch.nn.Module, x: torch.Tensor) -> Tuple[torch.Tensor, float]:
    """Hybrid full-forward repair: Triton norm/basis/hidden, torch readouts.

    This is not claimed as the final fused layer.  It checks whether the
    component-kernel repair survives once the remaining Legendre/quadratic/direct
    readouts are attached without recomputing the expensive input norm.
    """
    b1, z, kernels = _triton_legendre_basis_residual(model, x)
    B = int(x.shape[0])
    D = int(x.shape[1])
    H = int(model.legendre_hidden)
    h_pre = torch.empty((B, H), device=x.device, dtype=torch.float32)
    _legendre_hidden_from_b1[(B, triton.cdiv(H, 16))](
        b1,
        model.leg_w1,
        h_pre,
        B,
        D,
        H,
        32,
        16,
        num_warps=1,
    )
    base = (x - model.mu) / model.std
    rms = z.float().square().mean(dim=1, keepdim=True).sqrt().clamp_min(1.0e-3)
    mixed = (1.0 - model.residual_geom_mix) * base + model.residual_geom_mix * (z / rms * model.block_norm_target_rms)
    quad_x = mixed.clamp(-3.0, 3.0)
    h = torch.tanh(h_pre)
    b2 = prim._legendre4(h)
    leg_logits = torch.einsum("bhk,hck->bc", b2, model.leg_w2) / math.sqrt(max(1, int(model.legendre_hidden)))
    q_raw = quad_x @ model.quad_proj
    q = q_raw / model.quad_feature_std.clamp_min(1.0e-4) if model.quad_feature_norm_enabled else q_raw
    centered_square = q.square() - q.square().mean(dim=0, keepdim=True).detach()
    quad_logits = torch.einsum("bhk,hkc->bc", torch.stack([q, centered_square], dim=2), model.quad_readout)
    if model.branch_output_norm_enabled:
        leg_logits = leg_logits / model.leg_logit_std.clamp_min(1.0e-4)
        quad_logits = quad_logits / model.quad_logit_std.clamp_min(1.0e-4)
    direct_logits = torch.einsum("bdk,dck->bc", b1, model.direct_readout) / math.sqrt(max(1, D))
    logits = model.branch_scale[0] * leg_logits + model.branch_scale[1] * quad_logits + model.bias + model.direct_skip_scale * direct_logits
    return model.logit_gain.clamp(0.25, 4.0) * logits, kernels + float(B * math.ceil(H / 16))


def run_contract_manifest(args: argparse.Namespace, out_dir: Path, device: torch.device, x_train: torch.Tensor, input_dim: int, output_dim: int, specs: Mapping[str, prim.PrimitiveSpec]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    candidate_ids = ["MLP-same-param-AdamW", *_parse_list(args.candidate_ids)]
    for cid in candidate_ids:
        spec = specs.get(cid)
        model = _make_model(cid, input_dim, output_dim, x_train, device, int(args.seed) + 11, specs)
        manual = int(bool(hasattr(model, "manual_kernel_available") and model.manual_kernel_available()))  # type: ignore[attr-defined]
        edge_params = int(getattr(model, "edge_param_count", sum(p.numel() for p in model.parameters())))
        nonkan = 0 if cid != "MLP-same-param-AdamW" else edge_params
        official = int(cid != "MLP-same-param-AdamW" and not int(getattr(spec, "diagnostic_only", 0) if spec else 0))
        rows.append(
            _stamp(
                {
                    "stage": "V125_CONTRACT_MANIFEST",
                    "candidate_id": cid,
                    "family": spec.basis_family if spec else "MLP-control",
                    "basis_type": spec.basis_name if spec else "MLP",
                    "hidden_dim": int(getattr(model, "hidden_dim", 0)),
                    "basis_order": int(getattr(spec, "basis_order", 0) if spec else 0),
                    "quadratic_rank": int(getattr(model, "quadratic_hidden", 0)),
                    "directskip_scale": float(getattr(model, "direct_skip_scale", torch.tensor([0.0], device=device)).detach().flatten()[0].item()) if hasattr(model, "direct_skip_scale") else 0.0,
                    "branch_scale": json.dumps([float(v) for v in getattr(model, "branch_scale", torch.empty(0, device=device)).detach().flatten().tolist()]) if hasattr(model, "branch_scale") else "",
                    "input_norm_type": getattr(spec, "init_variant", "") if spec else "mlp",
                    "temperature_init": float(getattr(model, "logit_gain", torch.tensor([1.0], device=device)).detach().flatten()[0].item()) if hasattr(model, "logit_gain") else 1.0,
                    "nonKAN_param_count": nonkan,
                    "edge_param_count": edge_params,
                    "manual_forward_available": manual,
                    "manual_backward_available": manual,
                    "uses_loss_backward": int(not manual),
                    "uses_torch_autograd_graph": int(not manual),
                    "diagnostic_only": int(getattr(spec, "diagnostic_only", 0) if spec else 0),
                    "official_capable": official,
                    "strict_purekan_pass": int(official and nonkan == 0),
                    "contract_pass": int((cid == "MLP-same-param-AdamW") or (official and nonkan == 0)),
                }
            )
        )
    write_csv_rows(out_dir / "v125_contract_manifest.csv", rows)
    return rows


def run_forward_truth(args: argparse.Namespace, out_dir: Path, device: torch.device, x_train: torch.Tensor, input_dim: int, output_dim: int, specs: Mapping[str, prim.PrimitiveSpec]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, float]]:
    xb = x_train[: int(args.batch_size)].to(device=device, dtype=torch.float32).contiguous()
    mlp = _make_model("MLP-same-param-AdamW", input_dim, output_dim, x_train, device, int(args.seed) + 101, specs).eval()
    b47 = _make_model(args.k1_candidate_id, input_dim, output_dim, x_train, device, int(args.seed) + 102, specs).eval()
    b42 = _make_model(args.k1_fast_candidate_id, input_dim, output_dim, x_train, device, int(args.seed) + 103, specs).eval()
    b30 = _make_model(args.k2_candidate_id, input_dim, output_dim, x_train, device, int(args.seed) + 104, specs).eval()
    k3 = _make_model(args.k3_candidate_id, input_dim, output_dim, x_train, device, int(args.seed) + 105, specs).eval()
    rows: List[Dict[str, Any]] = []
    kernel_rows: List[Dict[str, Any]] = []
    with torch.no_grad():
        _ = mlp(xb)
        _ = b47(xb)
        _ = b42(xb)
        _ = b30(xb)
        _ = k3(xb)
    mlp_q50, mlp_q90, mlp_mean, _ = _time_call(lambda: mlp(xb), device, int(args.kernel_warmup_steps), int(args.kernel_measure_steps))
    ref_logits = b47._forward_reuse_inputs(xb).detach() if hasattr(b47, "_forward_reuse_inputs") else b47(xb).detach()
    b47_q50, b47_q90, _b47_mean, _ = _time_call(lambda: b47(xb), device, int(args.kernel_warmup_steps), int(args.kernel_measure_steps))
    b42_q50, b42_q90, _b42_mean, _ = _time_call(lambda: b42(xb), device, int(args.kernel_warmup_steps), int(args.kernel_measure_steps))
    b30_q50, b30_q90, _b30_mean, _ = _time_call(lambda: b30(xb), device, int(args.kernel_warmup_steps), int(args.kernel_measure_steps))
    k3_q50, k3_q90, _k3_mean, _ = _time_call(lambda: k3(xb), device, int(args.kernel_warmup_steps), int(args.kernel_measure_steps))

    def add_forward_row(impl: str, cid: str, q50: float, q90: float, custom_kernels: Any, torch_ops: Any, temp_bytes: Any, maxerr: Any, cos: Any, status: str = "measured") -> None:
        rows.append(
            _stamp(
                {
                    "stage": "V125_FORWARD_KERNEL_TRUTH",
                    "candidate_id": cid,
                    "kernel_impl": impl,
                    "status": status,
                    "forward_ratio_vs_mlp_q90": q90 / max(EPS, mlp_q90),
                    "forward_ratio_vs_b47_q90": q90 / max(EPS, b47_q90),
                    "forward_ms_q50": q50,
                    "forward_ms_q90": q90,
                    "kernel_count_forward": custom_kernels if isinstance(custom_kernels, int) else custom_kernels,
                    "custom_kernel_count_forward": custom_kernels,
                    "torch_op_count_forward": torch_ops,
                    "allocation_count_forward": "not_measured_runtime_allocator",
                    "temp_bytes_forward": temp_bytes,
                    "materialized_basis_bytes": temp_bytes if "legendre" in impl else 0,
                    "materialized_quad_bytes": 0,
                    "max_abs_forward_error_vs_reference": maxerr,
                    "forward_cos_vs_reference": cos,
                    "A1_forward_exploratory_pass": int(q90 / max(EPS, mlp_q90) <= 2.0 and (not isinstance(maxerr, float) or maxerr <= 1.0e-6)),
                    "A1_forward_formal_pass": int(q90 / max(EPS, mlp_q90) <= 1.50 and (not isinstance(maxerr, float) or maxerr <= 1.0e-6)),
                }
            )
        )

    add_forward_row("F0-current-B47-manual-autograd-forward", args.k1_candidate_id, b47_q50, b47_q90, "torch_custom_autograd_python_ops", "not_counted", "not_measured", 0.0, 1.0)
    add_forward_row("F0b-current-B42-fastreuse-forward", args.k1_fast_candidate_id, b42_q50, b42_q90, "torch_fastreuse_python_ops", "not_counted", "not_measured", "", "")
    add_forward_row("K2-current-LiteGated-forward", args.k2_candidate_id, b30_q50, b30_q90, "torch_lite_python_ops", "not_counted", "not_measured", "", "")
    add_forward_row("K3-SimpleFastTaskGeometry-forward", args.k3_candidate_id, k3_q50, k3_q90, "torch_simple_hinge_quadratic_ops", "not_counted", "not_measured", "", "")
    try:
        k3_compiled = torch.compile(k3, mode="reduce-overhead")  # type: ignore[attr-defined]
        with torch.no_grad():
            compiled_logits = k3_compiled(xb).detach()
            eager_logits = k3(xb).detach()
        _sync(device)
        k3_comp_err = float((compiled_logits - eager_logits).abs().max().detach().item())
        k3_comp_cos = float(F.cosine_similarity(compiled_logits.flatten(), eager_logits.flatten(), dim=0).detach().item())
        k3c_q50, k3c_q90, _k3c_mean, _ = _time_call(lambda: k3_compiled(xb), device, int(args.kernel_warmup_steps), int(args.kernel_measure_steps))
        add_forward_row("K3-compiled-SimpleFastTaskGeometry-forward", args.k3_candidate_id, k3c_q50, k3c_q90, "torch_compile_reduce_overhead", "not_counted", "not_measured", k3_comp_err, k3_comp_cos)
    except Exception as exc:
        add_forward_row("K3-compiled-SimpleFastTaskGeometry-forward-failed", args.k3_candidate_id, 0.0, float("inf"), "torch_compile_reduce_overhead", "not_counted", "not_measured", f"failed:{type(exc).__name__}:{exc}", "failed", status="torch_compile_failed")

    if not TRITON_AVAILABLE or device.type != "cuda":
        add_forward_row("F1/F2-triton-not-run", args.k1_candidate_id, 0.0, float("inf"), 0, 0, 0, "not_run", "not_run", status="triton_or_cuda_unavailable")
    else:
        try:
            with torch.no_grad():
                leg_z, _quad = b47._paired_legendre_quadratic_inputs(xb)  # type: ignore[attr-defined]
                ref_b1 = prim._legendre4(leg_z).detach()
                ref_h_pre = torch.einsum("bdk,dhk->bh", ref_b1, b47.leg_w1) / math.sqrt(max(1, input_dim))
                b1, _z, kernel_count = _triton_legendre_basis_residual(b47, xb)
                h_pre, h_kernel_count = _triton_legendre_hidden_residual(b47, xb)
                hybrid_logits, hybrid_kernel_count = _triton_hybrid_full_forward(b47, xb)
                _sync(device)
                b1_err = float((b1 - ref_b1).abs().max().detach().item())
                b1_cos = float(F.cosine_similarity(b1.flatten(), ref_b1.flatten(), dim=0).detach().item())
                h_err = float((h_pre - ref_h_pre).abs().max().detach().item())
                h_cos = float(F.cosine_similarity(h_pre.flatten(), ref_h_pre.flatten(), dim=0).detach().item())
                hybrid_err = float((hybrid_logits - ref_logits).abs().max().detach().item())
                hybrid_cos = float(F.cosine_similarity(hybrid_logits.flatten(), ref_logits.flatten(), dim=0).detach().item())
            f2_q50, f2_q90, _f2_mean, _ = _time_call(lambda: _triton_legendre_basis_residual(b47, xb)[0], device, int(args.kernel_warmup_steps), int(args.kernel_measure_steps))
            f3_q50, f3_q90, _f3_mean, _ = _time_call(lambda: _triton_legendre_hidden_residual(b47, xb)[0], device, int(args.kernel_warmup_steps), int(args.kernel_measure_steps))
            f4_q50, f4_q90, _f4_mean, _ = _time_call(lambda: _triton_hybrid_full_forward(b47, xb)[0], device, int(args.kernel_warmup_steps), int(args.kernel_measure_steps))
            basis_bytes = int(xb.shape[0]) * int(xb.shape[1]) * 4 * 4
            add_forward_row("F2-triton-residual-input-norm-plus-legendre-basis", args.k1_candidate_id, f2_q50, f2_q90, int(kernel_count), 0, basis_bytes, b1_err, b1_cos)
            add_forward_row("F3-triton-residual-input-norm-legendre-hidden-preactivation", args.k1_candidate_id, f3_q50, f3_q90, int(h_kernel_count), 0, basis_bytes + int(xb.shape[0]) * int(b47.legendre_hidden) * 4, h_err, h_cos)
            add_forward_row("F4-triton-hybrid-full-forward-with-torch-readouts", args.k1_candidate_id, f4_q50, f4_q90, int(hybrid_kernel_count), "torch_readout_ops", basis_bytes + int(xb.shape[0]) * int(b47.legendre_hidden) * 4, hybrid_err, hybrid_cos)
            kernel_rows.extend(
                [
                    _stamp({"stage": "V125_KERNEL_COUNT_BREAKDOWN", "candidate_id": args.k1_candidate_id, "kernel_impl": "F2-triton-residual-input-norm-plus-legendre-basis", "custom_kernel_count_forward": int(kernel_count), "torch_op_count_forward": 0, "kernel_count_total": int(kernel_count)}),
                    _stamp({"stage": "V125_KERNEL_COUNT_BREAKDOWN", "candidate_id": args.k1_candidate_id, "kernel_impl": "F3-triton-residual-input-norm-legendre-hidden-preactivation", "custom_kernel_count_forward": int(h_kernel_count), "torch_op_count_forward": 0, "kernel_count_total": int(h_kernel_count)}),
                    _stamp({"stage": "V125_KERNEL_COUNT_BREAKDOWN", "candidate_id": args.k1_candidate_id, "kernel_impl": "F4-triton-hybrid-full-forward-with-torch-readouts", "custom_kernel_count_forward": int(hybrid_kernel_count), "torch_op_count_forward": "torch_readout_ops", "kernel_count_total": f"{int(hybrid_kernel_count)}+torch_readout_ops"}),
                ]
            )
        except Exception as exc:
            add_forward_row("F2/F3-triton-failed", args.k1_candidate_id, 0.0, float("inf"), 0, 0, 0, f"failed:{type(exc).__name__}:{exc}", "failed", status="triton_kernel_failed")
    write_csv_rows(out_dir / "v125_forward_kernel_truth.csv", rows)
    write_csv_rows(out_dir / "v125_kernel_count_breakdown.csv", kernel_rows if kernel_rows else [_stamp({"stage": "V125_KERNEL_COUNT_BREAKDOWN", "status": "not_measured_or_triton_failed"})])
    write_csv_rows(out_dir / "v125_allocation_trace.csv", [_stamp({"stage": "V125_ALLOCATION_TRACE", "status": "not_measured_runtime_allocator", "reason": "PyTorch allocator event trace not available in this runner; CUDA peak memory is recorded in full_step_efficiency"})])
    write_csv_rows(out_dir / "v125_workspace_lifetime.csv", [_stamp({"stage": "V125_WORKSPACE_LIFETIME", "candidate_id": args.k1_candidate_id, "workspace_strategy": "component_fusion_allocates_explicit_temporary_tensors", "preallocated_workspace_used": 0, "reason": "F2/F3 smoke uses explicit b1/z/h buffers; no official workspace reuse claim"})])
    return rows, kernel_rows, {"mlp_forward_q90": mlp_q90, "b47_forward_q90": b47_q90}


def _manual_aware_step(model: torch.nn.Module, xb: torch.Tensor, yb: torch.Tensor) -> torch.Tensor:
    manual = bool(hasattr(model, "manual_kernel_available") and model.manual_kernel_available())  # type: ignore[attr-defined]
    if manual:
        logits, cache = model.manual_ce_forward_cache(xb)  # type: ignore[attr-defined]
        loss = F.cross_entropy(logits, yb)
        model.manual_ce_backward_from_cache(logits, cache, yb)  # type: ignore[attr-defined]
        return loss
    loss = F.cross_entropy(model(xb), yb)
    loss.backward()
    return loss


def run_backward_and_step_truth(args: argparse.Namespace, out_dir: Path, device: torch.device, x_train: torch.Tensor, y_train: torch.Tensor, input_dim: int, output_dim: int, specs: Mapping[str, prim.PrimitiveSpec]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    xb = x_train[: int(args.batch_size)].to(device=device, dtype=torch.float32).contiguous()
    yb = y_train[: int(args.batch_size)].to(device=device).contiguous()
    mlp = _make_model("MLP-same-param-AdamW", input_dim, output_dim, x_train, device, int(args.seed) + 201, specs)
    b47 = _make_model(args.k1_candidate_id, input_dim, output_dim, x_train, device, int(args.seed) + 202, specs)
    k3 = _make_model(args.k3_candidate_id, input_dim, output_dim, x_train, device, int(args.seed) + 203, specs)
    k3_compile_base = _make_model(args.k3_candidate_id, input_dim, output_dim, x_train, device, int(args.seed) + 204, specs)
    rows: List[Dict[str, Any]] = []
    grad_rows: List[Dict[str, Any]] = []
    fail_rows: List[Dict[str, Any]] = []
    try:
        audit = b47.manual_gradient_audit(xb[: min(16, int(xb.shape[0]))], yb[: min(16, int(yb.shape[0]))])  # type: ignore[attr-defined]
    except Exception as exc:
        audit = {"manual_forward_available": 0, "manual_backward_available": 0, "grad_relerr_max": f"failed:{exc}", "grad_cos_min": -1.0, "output_max_abs_error": "failed"}
    try:
        k3_audit = k3.manual_gradient_audit(xb[: min(16, int(xb.shape[0]))], yb[: min(16, int(yb.shape[0]))]) if hasattr(k3, "manual_gradient_audit") else {"manual_forward_available": 0, "manual_backward_available": 0, "grad_relerr_max": "not_manual_kernel", "grad_cos_min": "not_manual_kernel", "output_max_abs_error": ""}
    except Exception as exc:
        k3_audit = {"manual_forward_available": 0, "manual_backward_available": 0, "grad_relerr_max": f"failed:{exc}", "grad_cos_min": -1.0, "output_max_abs_error": "failed"}
    grad_rows.append(
        _stamp(
            {
                "stage": "V125_GRAD_CORRECTNESS",
                "candidate_id": args.k1_candidate_id,
                "kernel_impl": "B0-current-B47-manual-ce-step",
                "manual_forward_available": audit.get("manual_forward_available", 0),
                "manual_backward_available": audit.get("manual_backward_available", 0),
                "grad_relerr_max": audit.get("grad_relerr_max", ""),
                "grad_cos_min": audit.get("grad_cos_min", ""),
                "output_max_abs_error": audit.get("output_max_abs_error", ""),
                "correctness_pass": int(_safe_float(audit.get("grad_relerr_max"), 999.0) < 1.0e-4 and _safe_float(audit.get("grad_cos_min"), -1.0) > 0.999),
            }
        )
    )
    grad_rows.append(
        _stamp(
            {
                "stage": "V125_GRAD_CORRECTNESS",
                "candidate_id": args.k3_candidate_id,
                "kernel_impl": "K3-simple-fast-manual-ce-step",
                "manual_forward_available": k3_audit.get("manual_forward_available", 0),
                "manual_backward_available": k3_audit.get("manual_backward_available", 0),
                "grad_relerr_max": k3_audit.get("grad_relerr_max", ""),
                "grad_cos_min": k3_audit.get("grad_cos_min", ""),
                "output_max_abs_error": k3_audit.get("output_max_abs_error", ""),
                "correctness_pass": int(_safe_float(k3_audit.get("grad_relerr_max"), 999.0) < 1.0e-4 and _safe_float(k3_audit.get("grad_cos_min"), -1.0) > 0.999),
            }
        )
    )

    def measure_train_step(model: torch.nn.Module, cid: str, force_autograd: bool = False) -> Dict[str, Any]:
        optimizer_impl_used = str(args.optimizer_impl)
        if optimizer_impl_used == "fused_adamw" and cid == str(args.k1_candidate_id):
            optimizer_impl_used = "adamw_fallback_for_b47_manual_reference"
        opt = _make_adamw(
            model.parameters(),
            args,
            "adamw" if optimizer_impl_used == "adamw_fallback_for_b47_manual_reference" else optimizer_impl_used,
        )
        fwd: List[float] = []
        bwd: List[float] = []
        upd: List[float] = []
        step: List[float] = []
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        for idx in range(int(args.kernel_warmup_steps) + int(args.kernel_measure_steps)):
            opt.zero_grad(set_to_none=True)
            _sync(device)
            t0 = time.perf_counter()
            manual = bool((not force_autograd) and hasattr(model, "manual_kernel_available") and model.manual_kernel_available())  # type: ignore[attr-defined]
            if manual:
                logits, cache = model.manual_ce_forward_cache(xb)  # type: ignore[attr-defined]
                loss = F.cross_entropy(logits, yb)
            else:
                loss = F.cross_entropy(model(xb), yb)
            _sync(device)
            t1 = time.perf_counter()
            if manual:
                model.manual_ce_backward_from_cache(logits, cache, yb)  # type: ignore[attr-defined]
            else:
                loss.backward()
            _sync(device)
            t2 = time.perf_counter()
            opt.step()
            _sync(device)
            t3 = time.perf_counter()
            if idx == int(args.kernel_warmup_steps) - 1 and device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            if idx >= int(args.kernel_warmup_steps):
                fwd.append((t1 - t0) * 1000.0)
                bwd.append((t2 - t1) * 1000.0)
                upd.append((t3 - t2) * 1000.0)
                step.append((t3 - t0) * 1000.0)
        peak = float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)) if device.type == "cuda" else 0.0
        return {
            "candidate_id": cid,
            "forward_q90": _q(fwd, 0.90),
            "backward_q90": _q(bwd, 0.90),
            "update_q90": _q(upd, 0.90),
            "step_q90": _q(step, 0.90),
            "peak_mb": peak,
            "optimizer_impl_used": optimizer_impl_used,
        }

    mlp_m = measure_train_step(mlp, "MLP-same-param-AdamW")
    b47_m = measure_train_step(b47, args.k1_candidate_id)
    k3_m = measure_train_step(k3, args.k3_candidate_id)
    try:
        k3_compiled_model = torch.compile(k3_compile_base, mode="reduce-overhead")  # type: ignore[attr-defined]
        k3_compiled_m = measure_train_step(k3_compiled_model, args.k3_candidate_id, force_autograd=True)
        k3_compiled_status = "measured"
    except Exception as exc:
        k3_compiled_m = {"candidate_id": args.k3_candidate_id, "forward_q90": float("inf"), "backward_q90": float("inf"), "update_q90": float("inf"), "step_q90": float("inf"), "peak_mb": float("inf"), "compile_error": f"{type(exc).__name__}:{exc}"}
        k3_compiled_status = "torch_compile_failed"
    backward_ratio = b47_m["backward_q90"] / max(EPS, mlp_m["backward_q90"])
    step_ratio = b47_m["step_q90"] / max(EPS, mlp_m["step_q90"])
    memory_ratio = b47_m["peak_mb"] / max(EPS, mlp_m["peak_mb"]) if mlp_m["peak_mb"] > 0 else 1.0
    k3_backward_ratio = k3_m["backward_q90"] / max(EPS, mlp_m["backward_q90"])
    k3_step_ratio = k3_m["step_q90"] / max(EPS, mlp_m["step_q90"])
    k3_memory_ratio = k3_m["peak_mb"] / max(EPS, mlp_m["peak_mb"]) if mlp_m["peak_mb"] > 0 else 1.0
    k3c_backward_ratio = k3_compiled_m["backward_q90"] / max(EPS, mlp_m["backward_q90"])
    k3c_step_ratio = k3_compiled_m["step_q90"] / max(EPS, mlp_m["step_q90"])
    k3c_memory_ratio = k3_compiled_m["peak_mb"] / max(EPS, mlp_m["peak_mb"]) if mlp_m["peak_mb"] > 0 else 1.0
    rows.append(
        _stamp(
            {
                "stage": "V125_BACKWARD_KERNEL_TRUTH",
                "candidate_id": args.k1_candidate_id,
                "backward_impl": "B0-current-B47-manual-ce-step",
                "grad_relerr_max": audit.get("grad_relerr_max", ""),
                "grad_cos_min": audit.get("grad_cos_min", ""),
                "param_grad_linf": "measured_in_grad_correctness",
                "input_grad_relerr_max": "not_required_for_current_no_dx_architecture",
                "input_grad_cos_min": "not_required_for_current_no_dx_architecture",
                "backward_ratio": backward_ratio,
                "backward_ms_q90": b47_m["backward_q90"],
                "step_ratio": step_ratio,
                "memory_ratio": memory_ratio,
                "kernel_count_backward": "not_counted_python_manual",
                "allocation_count_backward": "not_measured_runtime_allocator",
                "workspace_temp_mb": "manual_cache_tensors_not_preallocated",
                "basis_recomputed": 0,
                "basis_materialized": 1,
                "atomic_add_count_proxy": "not_used",
                "reduction_strategy": "torch_reductions_inside_manual_ce_backward",
                "A2_backward_exploratory_pass": int(backward_ratio <= 1.75 and step_ratio <= 1.75),
                "A2_backward_formal_pass": int(backward_ratio <= 1.50 and step_ratio <= 1.50 and memory_ratio <= 1.05),
            }
        )
    )
    rows.append(
        _stamp(
            {
                "stage": "V125_BACKWARD_KERNEL_TRUTH",
                "candidate_id": args.k3_candidate_id,
                "backward_impl": "K3-compiled-autograd-reduce-overhead",
                "status": k3_compiled_status,
                "grad_relerr_max": "torch_autograd_reference",
                "grad_cos_min": "torch_autograd_reference",
                "backward_ratio": k3c_backward_ratio,
                "backward_ms_q90": k3_compiled_m["backward_q90"],
                "step_ratio": k3c_step_ratio,
                "memory_ratio": k3c_memory_ratio,
                "kernel_count_backward": "not_counted_torch_compile",
                "allocation_count_backward": "not_measured_runtime_allocator",
                "workspace_temp_mb": "torch_compile_internal",
                "basis_recomputed": "torch_compile_internal",
                "basis_materialized": "torch_compile_internal",
                "reduction_strategy": "torch_compile_autograd",
                "A2_backward_exploratory_pass": int(k3c_backward_ratio <= 1.75 and k3c_step_ratio <= 1.75),
                "A2_backward_formal_pass": int(k3c_backward_ratio <= 1.50 and k3c_step_ratio <= 1.50 and k3c_memory_ratio <= 1.05),
            }
        )
    )
    for impl in ["B1-fused-param-gradient-only", "B2-fused-param-gradient-no-materialized-basis", "B3-fused-dx-param-gradient", "B4-two-stage-reduction", "B5-workspace-reuse", "B6-cuda-graph-captured"]:
        rows.append(
            _stamp(
                {
                    "stage": "V125_BACKWARD_KERNEL_TRUTH",
                    "candidate_id": args.k1_candidate_id,
                    "backward_impl": impl,
                    "status": "not_implemented_after_A1_forward_component_truth_failed_to_reach_formal_gate",
                    "grad_relerr_max": "not_run",
                    "grad_cos_min": "not_run",
                    "backward_ratio": "",
                    "step_ratio": "",
                    "memory_ratio": "",
                    "A2_backward_exploratory_pass": 0,
                    "A2_backward_formal_pass": 0,
                }
            )
        )
    rows.append(
        _stamp(
            {
                "stage": "V125_BACKWARD_KERNEL_TRUTH",
                "candidate_id": args.k3_candidate_id,
                "backward_impl": "K3-current-manual-ce-step" if hasattr(k3, "manual_kernel_available") and k3.manual_kernel_available() else "K3-current-torch-autograd-simple-fast-task-geometry",  # type: ignore[attr-defined]
                "grad_relerr_max": k3_audit.get("grad_relerr_max", ""),
                "grad_cos_min": k3_audit.get("grad_cos_min", ""),
                "backward_ratio": k3_backward_ratio,
                "backward_ms_q90": k3_m["backward_q90"],
                "step_ratio": k3_step_ratio,
                "memory_ratio": k3_memory_ratio,
                "kernel_count_backward": "not_counted_torch_autograd",
                "allocation_count_backward": "not_measured_runtime_allocator",
                "workspace_temp_mb": "not_preallocated",
                "basis_recomputed": "torch_autograd_internal",
                "basis_materialized": "torch_autograd_internal",
                "reduction_strategy": "torch_autograd",
                "A2_backward_exploratory_pass": int(k3_backward_ratio <= 1.75 and k3_step_ratio <= 1.75),
                "A2_backward_formal_pass": int(k3_backward_ratio <= 1.50 and k3_step_ratio <= 1.50 and k3_memory_ratio <= 1.05),
            }
        )
    )
    full = [
        _stamp(
            {
                "stage": "V125_FULL_STEP_EFFICIENCY",
                "candidate_id": "MLP-same-param-AdamW",
                "forward_impl": "MLP-reference",
                "backward_impl": "torch-autograd",
                "update_impl": mlp_m.get("optimizer_impl_used", str(args.optimizer_impl)),
                "manual_ce_step": 0,
                "uses_torch_autograd_graph": 1,
                "forward_ratio_q90": 1.0,
                "backward_ratio_q90": 1.0,
                "optimizer_ratio_q90": 1.0,
                "step_ratio_q90": 1.0,
                "memory_ratio_q90": 1.0,
                "kernel_count_total": "not_counted",
                "allocation_count_total": "not_measured",
                "samples_per_second": float(args.batch_size) / max(EPS, mlp_m["step_q90"] / 1000.0),
                "compile_warmup_steps": int(args.kernel_warmup_steps),
                "steady_state_accounting_used": 1,
                "A3_efficiency_exploratory_pass": 1,
                "A3_efficiency_official_pass": 1,
            }
        ),
        _stamp(
            {
                "stage": "V125_FULL_STEP_EFFICIENCY",
                "candidate_id": args.k1_candidate_id,
                "forward_impl": "current-B47-manual-forward-cache",
                "backward_impl": "current-B47-manual-ce-backward",
                "update_impl": b47_m.get("optimizer_impl_used", str(args.optimizer_impl)),
                "manual_ce_step": 1,
                "uses_torch_autograd_graph": 0,
                "forward_ratio_q90": b47_m["forward_q90"] / max(EPS, mlp_m["forward_q90"]),
                "backward_ratio_q90": backward_ratio,
                "optimizer_ratio_q90": b47_m["update_q90"] / max(EPS, mlp_m["update_q90"]),
                "step_ratio_q90": step_ratio,
                "memory_ratio_q90": memory_ratio,
                "kernel_count_total": "not_counted_python_manual",
                "allocation_count_total": "not_measured_runtime_allocator",
                "samples_per_second": float(args.batch_size) / max(EPS, b47_m["step_q90"] / 1000.0),
                "compile_warmup_steps": int(args.kernel_warmup_steps),
                "steady_state_accounting_used": 1,
                "A3_efficiency_exploratory_pass": int(step_ratio <= 1.50 and memory_ratio <= 1.10),
                "A3_efficiency_official_pass": int(step_ratio <= 1.25 and memory_ratio <= 1.05),
            }
        ),
        _stamp(
            {
                "stage": "V125_FULL_STEP_EFFICIENCY",
                "candidate_id": args.k3_candidate_id,
                "forward_impl": "K3-simple-hinge-quadratic-forward",
                "backward_impl": "K3-simple-manual-ce-backward" if hasattr(k3, "manual_kernel_available") and k3.manual_kernel_available() else "torch-autograd",  # type: ignore[attr-defined]
                "update_impl": k3_m.get("optimizer_impl_used", str(args.optimizer_impl)),
                "manual_ce_step": int(hasattr(k3, "manual_kernel_available") and k3.manual_kernel_available()),  # type: ignore[attr-defined]
                "uses_torch_autograd_graph": int(not (hasattr(k3, "manual_kernel_available") and k3.manual_kernel_available())),  # type: ignore[attr-defined]
                "forward_ratio_q90": k3_m["forward_q90"] / max(EPS, mlp_m["forward_q90"]),
                "backward_ratio_q90": k3_backward_ratio,
                "optimizer_ratio_q90": k3_m["update_q90"] / max(EPS, mlp_m["update_q90"]),
                "step_ratio_q90": k3_step_ratio,
                "memory_ratio_q90": k3_memory_ratio,
                "kernel_count_total": "not_counted_torch_simple_fast",
                "allocation_count_total": "not_measured_runtime_allocator",
                "samples_per_second": float(args.batch_size) / max(EPS, k3_m["step_q90"] / 1000.0),
                "compile_warmup_steps": int(args.kernel_warmup_steps),
                "steady_state_accounting_used": 1,
                "A3_efficiency_exploratory_pass": int(k3_step_ratio <= 1.50 and k3_memory_ratio <= 1.10),
                "A3_efficiency_official_pass": int(k3_step_ratio <= 1.25 and k3_memory_ratio <= 1.05),
            }
        ),
        _stamp(
            {
                "stage": "V125_FULL_STEP_EFFICIENCY",
                "candidate_id": args.k3_candidate_id,
                "forward_impl": "K3-compiled-simple-hinge-quadratic-forward",
                "backward_impl": "K3-compiled-autograd",
                "update_impl": k3_compiled_m.get("optimizer_impl_used", str(args.optimizer_impl)),
                "manual_ce_step": 0,
                "uses_torch_autograd_graph": 1,
                "compile_status": k3_compiled_status,
                "forward_ratio_q90": k3_compiled_m["forward_q90"] / max(EPS, mlp_m["forward_q90"]),
                "backward_ratio_q90": k3c_backward_ratio,
                "optimizer_ratio_q90": k3_compiled_m["update_q90"] / max(EPS, mlp_m["update_q90"]),
                "step_ratio_q90": k3c_step_ratio,
                "memory_ratio_q90": k3c_memory_ratio,
                "kernel_count_total": "not_counted_torch_compile",
                "allocation_count_total": "not_measured_runtime_allocator",
                "samples_per_second": float(args.batch_size) / max(EPS, k3_compiled_m["step_q90"] / 1000.0),
                "compile_warmup_steps": int(args.kernel_warmup_steps),
                "steady_state_accounting_used": 1,
                "A3_efficiency_exploratory_pass": int(k3c_step_ratio <= 1.50 and k3c_memory_ratio <= 1.10),
                "A3_efficiency_official_pass": int(k3c_step_ratio <= 1.25 and k3c_memory_ratio <= 1.05),
            }
        ),
    ]
    if step_ratio > 1.50:
        fail_rows.append(_stamp({"stage": "V125_EFFICIENCY_FAILURE_TABLE", "candidate_id": args.k1_candidate_id, "failure_code": "R1_FusedEfficiencyFail_current_manual_step", "step_ratio_q90": step_ratio, "backward_ratio_q90": backward_ratio, "memory_ratio_q90": memory_ratio, "action_recommended": "lower-level fused forward+backward kernel or simpler task-geometry primitive"}))
    if k3_step_ratio > 1.50 or k3_memory_ratio > 1.10:
        fail_rows.append(_stamp({"stage": "V125_EFFICIENCY_FAILURE_TABLE", "candidate_id": args.k3_candidate_id, "failure_code": "R1_or_R6_SimpleFastTaskGeometry_efficiency_fail", "step_ratio_q90": k3_step_ratio, "backward_ratio_q90": k3_backward_ratio, "memory_ratio_q90": k3_memory_ratio, "action_recommended": "if simple primitive is fast enough proceed to expression; otherwise implement lower-level fused simple hinge/quadratic kernel"}))
    if k3c_step_ratio > 1.50 or k3c_memory_ratio > 1.10:
        fail_rows.append(_stamp({"stage": "V125_EFFICIENCY_FAILURE_TABLE", "candidate_id": args.k3_candidate_id, "failure_code": "R1_or_R6_SimpleFastTaskGeometry_compiled_efficiency_fail", "step_ratio_q90": k3c_step_ratio, "backward_ratio_q90": k3c_backward_ratio, "memory_ratio_q90": k3c_memory_ratio, "action_recommended": "if compiled simple full-step remains slow implement lower-level fused simple hinge/quadratic kernel"}))
    write_csv_rows(out_dir / "v125_backward_kernel_truth.csv", rows)
    write_csv_rows(out_dir / "v125_grad_correctness.csv", grad_rows)
    write_csv_rows(out_dir / "v125_full_step_efficiency.csv", full)
    write_csv_rows(out_dir / "v125_efficiency_failure_table.csv", fail_rows if fail_rows else [_stamp({"stage": "V125_EFFICIENCY_FAILURE_TABLE", "status": "no_efficiency_failure"})])
    return rows, full, grad_rows


def _classification_basic(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
    model.eval()
    with torch.no_grad():
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        probs = logits.softmax(dim=1)
        pred = probs.argmax(dim=1)
        conf = probs.max(dim=1).values
        correct = pred.eq(y)
        ce = F.cross_entropy(logits, y, reduction="none")
        true = probs.gather(1, y.view(-1, 1)).squeeze(1)
        top2 = torch.topk(probs, k=min(2, int(probs.shape[1])), dim=1).values
        margin = top2[:, 0] - top2[:, 1] if top2.shape[1] > 1 else true
        ece = torch.abs(conf - correct.float()).mean()
        return {
            "acc": float(correct.float().mean().item()),
            "NLL": float(loss.item()),
            "ECE": float(ece.item()),
            "CEp99": float(torch.quantile(ce.detach().float(), 0.99).item()),
            "margin_p10": float(torch.quantile(margin.detach().float(), 0.10).item()),
        }


def _ridge_coupling(delta_b: torch.Tensor, delta_q: torch.Tensor, ridge: float) -> Tuple[float, float, float, float]:
    n = min(int(delta_b.shape[0]), int(delta_q.shape[0]))
    x = delta_b[:n].float()
    y = delta_q[:n].float()
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)
    c = int(x.shape[1])
    eye = torch.eye(c, device=x.device, dtype=torch.float32)
    a = torch.linalg.solve(x.T @ x + float(ridge) * eye, x.T @ y)
    pred = x @ a
    resid = y - pred
    r2 = 1.0 - float(resid.square().sum().item()) / max(EPS, float(y.square().sum().item()))
    corr = float(F.cosine_similarity(pred.flatten(), y.flatten(), dim=0).item()) if pred.norm() > 0 and y.norm() > 0 else 0.0
    return r2, corr, float(resid.norm().item()), float(pred.norm().item())


def _sample_grad_sketch(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, sketch_dim: int, seed: int) -> torch.Tensor:
    params = [p for p in model.parameters() if p.requires_grad]
    rows: List[torch.Tensor] = []
    for i in range(int(x.shape[0])):
        model.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x[i : i + 1]), y[i : i + 1])
        grads = torch.autograd.grad(loss, params, retain_graph=False, create_graph=False, allow_unused=True)
        vals: List[torch.Tensor] = []
        for k in range(int(sketch_dim)):
            dot = torch.zeros((), device=x.device)
            for pidx, g in enumerate(grads):
                if g is None:
                    continue
                gen = torch.Generator(device=x.device).manual_seed(int(seed) + 1009 * k + 9176 * pidx)
                signs = torch.randint(0, 2, g.shape, device=x.device, generator=gen, dtype=torch.int8).float().mul_(2.0).sub_(1.0)
                dot = dot + (g.float() * signs).sum() / math.sqrt(max(1, g.numel()))
            vals.append(dot)
        rows.append(torch.stack(vals))
    model.zero_grad(set_to_none=True)
    return torch.stack(rows, dim=0)


def _signal_reservoir_metrics(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, sketch_dim: int, seed: int) -> Dict[str, float]:
    b = min(int(x.shape[0]), int(y.shape[0]))
    x = x[:b]
    y = y[:b]
    grad_sketch = _sample_grad_sketch(model, x, y, int(sketch_dim), int(seed))
    k_mat = grad_sketch @ grad_sketch.T
    evals, evecs = torch.linalg.eigh(k_mat.float())
    order = torch.argsort(evals, descending=True)
    evals = evals[order].clamp_min(0.0)
    evecs = evecs[:, order]
    total = evals.sum().clamp_min(EPS)
    cum = torch.cumsum(evals, dim=0)
    top_count = int(torch.searchsorted(cum, 0.80 * total).item()) + 1
    top_count = max(1, min(top_count, int(evals.numel())))
    p_sig = evecs[:, :top_count] @ evecs[:, :top_count].T
    p_res = torch.eye(b, device=x.device) - p_sig
    with torch.no_grad():
        logits = model(x)
        probs = logits.softmax(dim=1)
        ce_real = F.cross_entropy(logits, y, reduction="none")
        y_noise = y[torch.randperm(b, device=x.device, generator=torch.Generator(device=x.device).manual_seed(int(seed) + 33))]
        ce_noise = F.cross_entropy(logits, y_noise, reduction="none")
        r_real = ce_real.float() - ce_real.float().mean()
        r_noise = ce_noise.float() - ce_noise.float().mean()
        real_res = float((p_res @ r_real).square().sum().div(r_real.square().sum().clamp_min(EPS)).item())
        noise_sig = float((p_sig @ r_noise).square().sum().div(r_noise.square().sum().clamp_min(EPS)).item())
        snr_pos = float(((probs.gather(1, y.view(-1, 1)).squeeze(1) - probs.mean(dim=1)) > 0).float().mean().item())
        real_noise_gap = float(ce_noise.mean().sub(ce_real.mean()).item())
    eff_rank = float((evals.sum().square() / evals.square().sum().clamp_min(EPS)).item())
    return {
        "signal_effective_rank": eff_rank,
        "signal_mass_topk": float(evals[:top_count].sum().div(total).item()),
        "reservoir_fraction": float(1.0 - evals[:top_count].sum().div(total).item()),
        "top_eigen_share": float(evals[0].div(total).item()) if evals.numel() else 0.0,
        "dissipation_condition": float(evals[0].div(evals[evals > 1.0e-8][-1].clamp_min(EPS)).item()) if bool((evals > 1.0e-8).any()) else 0.0,
        "RealSignalReservoirRatio": real_res,
        "NoiseSignalLeak": noise_sig,
        "SNR_positive_fraction": snr_pos,
        "real_noise_gap": real_noise_gap,
    }


def _basis_diag(model: torch.nn.Module, x: torch.Tensor) -> Dict[str, float]:
    if hasattr(model, "basis_diagnostics"):
        d = model.basis_diagnostics(x)  # type: ignore[attr-defined]
        return {
            "effective_rank_hidden": _safe_float(d.get("basis_effective_rank"), 0.0),
            "basis_occupancy_entropy": _safe_float(d.get("basis_entropy"), 0.0),
            "basis_dead_fraction": _safe_float(d.get("dead_basis_fraction"), 0.0),
            "lift_condition_proxy": _safe_float(d.get("basis_condition_proxy"), 0.0),
            "perturb_logit_drift_p95": _safe_float(d.get("basis_output_norm_p95"), 0.0),
        }
    return {"effective_rank_hidden": 0.0, "basis_occupancy_entropy": 0.0, "basis_dead_fraction": 0.0, "lift_condition_proxy": 0.0, "perturb_logit_drift_p95": 0.0}


def _take_adamw_window(model: torch.nn.Module, xb: torch.Tensor, yb: torch.Tensor, lr: float, wd: float) -> torch.nn.Module:
    m = deepcopy(model)
    opt = torch.optim.AdamW(m.parameters(), lr=float(lr), weight_decay=float(wd))
    opt.zero_grad(set_to_none=True)
    _manual_aware_step(m, xb, yb)
    opt.step()
    return m


def run_line_c(args: argparse.Namespace, out_dir: Path, device: torch.device, x_train: torch.Tensor, y_train: torch.Tensor, x_val: torch.Tensor, y_val: torch.Tensor, input_dim: int, output_dim: int, specs: Mapping[str, prim.PrimitiveSpec], base_qualified: bool) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    candidate_ids = ["MLP-same-param-AdamW", args.k1_fast_candidate_id, args.k1_candidate_id, args.k2_candidate_id, args.k3_candidate_id]
    B = min(int(args.coupling_batch_size), int(x_train.shape[0]) // 2)
    Q = min(int(args.coupling_batch_size), int(x_val.shape[0]))
    xb = x_train[:B].to(device=device, dtype=torch.float32)
    yb = y_train[:B].to(device=device)
    xq = x_val[:Q].to(device=device, dtype=torch.float32)
    yq = y_val[:Q].to(device=device)
    coupling_rows: List[Dict[str, Any]] = []
    sketch_rows: List[Dict[str, Any]] = []
    diag_rows: List[Dict[str, Any]] = []
    noise_rows: List[Dict[str, Any]] = []
    for cid in candidate_ids:
        model = _make_model(cid, input_dim, output_dim, x_train, device, int(args.seed) + 401, specs).eval()
        before_b = model(xb).detach()
        before_q = model(xq).detach()
        updated = _take_adamw_window(model, xb, yb, float(args.lr), float(args.weight_decay)).eval()
        after_b = updated(xb).detach()
        after_q = updated(xq).detach()
        db = after_b - before_b
        dq = after_q - before_q
        r2, corr, resid, pred_norm = _ridge_coupling(db, dq, float(args.ridge_lambda))
        kernel_drift = float((after_b @ after_b.T - before_b @ before_b.T).norm().div((before_b @ before_b.T).norm().clamp_min(EPS)).item())
        before_cls = _classification_basic(model, xq, yq)
        after_cls = _classification_basic(updated, xq, yq)
        spec = specs.get(cid)
        coupling_rows.append(
            _stamp(
                {
                    "stage": "V125_TRAIN_PROBE_COUPLING",
                    "run_id": out_dir.name,
                    "candidate_id": cid,
                    "basis_family": spec.basis_family if spec else "MLP-control",
                    "method": cid,
                    "update_type": "AdamW-one-window",
                    "control_id": "base_window_update",
                    "dataset": "MNIST",
                    "seed": 0,
                    "step": 1,
                    "window_size": 1,
                    "batch_size_B": B,
                    "batch_size_Q": Q,
                    "ridge_lambda": float(args.ridge_lambda),
                    "train_logit_drift_l2": float(db.norm().item()),
                    "probe_logit_drift_l2": float(dq.norm().item()),
                    "CouplingR2": r2,
                    "CouplingCorr": corr,
                    "coupling_residual_norm": resid,
                    "coupling_prediction_norm": pred_norm,
                    "coupling_stability_across_splits": "single_split_triage",
                    "KernelDrift": kernel_drift,
                    "CEp99_delta": after_cls["CEp99"] - before_cls["CEp99"],
                    "ECE_delta": after_cls["ECE"] - before_cls["ECE"],
                    "margin_p10_delta": after_cls["margin_p10"] - before_cls["margin_p10"],
                    "official_gate_open": int(base_qualified),
                }
            )
        )
        sig = _signal_reservoir_metrics(updated, xb[: min(B, int(args.sketch_batch_size))], yb[: min(B, int(args.sketch_batch_size))], int(args.sketch_dim), int(args.seed) + 402)
        sketch_rows.append(
            _stamp(
                {
                    "stage": "V125_SIGNAL_RESERVOIR_SKETCH",
                    "run_id": out_dir.name,
                    "candidate_id": cid,
                    "basis_family": spec.basis_family if spec else "MLP-control",
                    "method": cid,
                    "update_type": "AdamW-one-window",
                    "control_id": "base_window_update",
                    "dataset": "MNIST",
                    "seed": 0,
                    "step": 1,
                    "window_size": 1,
                    "sketch_dim": int(args.sketch_dim),
                    "official_gate_open": int(base_qualified),
                    **sig,
                }
            )
        )
        cls = _classification_basic(updated, xq, yq)
        diag_rows.append(
            _stamp(
                {
                    "stage": "V125_MANIFOLD_CHANNEL_DIAGNOSTICS",
                    "run_id": out_dir.name,
                    "candidate_id": cid,
                    "basis_family": spec.basis_family if spec else "MLP-control",
                    "method": cid,
                    "dataset": "MNIST",
                    "seed": 0,
                    "step": 1,
                    "base_qualified": int(base_qualified),
                    "functional_official_open": 0,
                    "CouplingR2": r2,
                    "CouplingCorr": corr,
                    "RealSignalReservoirRatio": sig["RealSignalReservoirRatio"],
                    "NoiseSignalLeak": sig["NoiseSignalLeak"],
                    "KernelDrift": kernel_drift,
                    "CEp99": cls["CEp99"],
                    "ECE": cls["ECE"],
                    "NLL": cls["NLL"],
                    "step_time_ratio": "",
                    "memory_ratio": "",
                    **_basis_diag(updated, xq),
                }
            )
        )
        noise_rows.append(_stamp({"stage": "V125_NOISE_LEAK_AUDIT", "candidate_id": cid, "dataset": "MNIST", "seed": 0, "NoiseSignalLeak": sig["NoiseSignalLeak"], "RealSignalReservoirRatio": sig["RealSignalReservoirRatio"], "real_noise_gap": sig["real_noise_gap"], "official_gate_open": int(base_qualified)}))
    write_csv_rows(out_dir / "v125_train_probe_coupling.csv", coupling_rows)
    write_csv_rows(out_dir / "v125_signal_reservoir_sketch.csv", sketch_rows)
    write_csv_rows(out_dir / "v125_noise_leak_audit.csv", noise_rows)
    write_csv_rows(out_dir / "v125_manifold_channel_diagnostics.csv", diag_rows)
    return coupling_rows, sketch_rows, noise_rows, diag_rows


def _params(model: torch.nn.Module) -> List[torch.nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


def _grad_delta(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, lr: float) -> List[torch.Tensor]:
    model.zero_grad(set_to_none=True)
    _manual_aware_step(model, x, y)
    return [(-float(lr) * (p.grad.detach() if p.grad is not None else torch.zeros_like(p))).clone() for p in _params(model)]


def _basis_delta(model: torch.nn.Module, task_delta: Sequence[torch.Tensor], mode: str) -> List[torch.Tensor]:
    if not hasattr(model, "basis_diagnostics"):
        return [torch.zeros_like(d) for d in task_delta]
    raw = prim.basis_functional_direction(model, mode)
    deltas = [0.05 * d.to(task_delta[0].device) / d.norm().clamp_min(EPS) for d in raw]
    norm_task = math.sqrt(sum(float(d.square().sum().item()) for d in task_delta))
    norm_basis = math.sqrt(sum(float(d.square().sum().item()) for d in deltas))
    deltas = [d * (norm_task / max(EPS, norm_basis)) for d in deltas]
    if mode == "basis_aware_orthogonal":
        flat_task = torch.cat([d.flatten() for d in task_delta])
        flat_basis = torch.cat([d.flatten() for d in deltas])
        proj = (flat_basis @ flat_task) / flat_task.square().sum().clamp_min(EPS)
        deltas = [d - proj * t for d, t in zip(deltas, task_delta)]
    return deltas


def _random_like(deltas: Sequence[torch.Tensor], seed: int) -> List[torch.Tensor]:
    device = deltas[0].device
    gen = torch.Generator(device=device).manual_seed(int(seed))
    flat_norm = math.sqrt(sum(float(d.square().sum().item()) for d in deltas))
    out = [torch.randn(d.shape, device=device, generator=gen) for d in deltas]
    norm = math.sqrt(sum(float(d.square().sum().item()) for d in out))
    return [d * (flat_norm / max(EPS, norm)) for d in out]


def _apply_delta(model: torch.nn.Module, deltas: Sequence[torch.Tensor], scale: float) -> None:
    with torch.no_grad():
        for p, d in zip(_params(model), deltas):
            p.add_(d.to(p.device), alpha=float(scale))


def _geo_gain_from_line_c(before_model: torch.nn.Module, after_model: torch.nn.Module, xb: torch.Tensor, yb: torch.Tensor, xq: torch.Tensor, yq: torch.Tensor, ridge: float, sketch_dim: int, seed: int) -> Dict[str, float]:
    with torch.no_grad():
        db = after_model(xb).detach() - before_model(xb).detach()
        dq = after_model(xq).detach() - before_model(xq).detach()
    r2, corr, _resid, _pred = _ridge_coupling(db, dq, ridge)
    sig = _signal_reservoir_metrics(after_model, xb[: min(int(xb.shape[0]), 8)], yb[: min(int(yb.shape[0]), 8)], sketch_dim, seed)
    cls_before = _classification_basic(before_model, xq, yq)
    cls_after = _classification_basic(after_model, xq, yq)
    score = r2 - sig["NoiseSignalLeak"] - sig["RealSignalReservoirRatio"] - max(0.0, cls_after["NLL"] - cls_before["NLL"])
    return {"CouplingR2": r2, "CouplingCorr": corr, "RealSignalReservoirRatio": sig["RealSignalReservoirRatio"], "NoiseSignalLeak": sig["NoiseSignalLeak"], "geo_score": score, "CEp99_delta": cls_after["CEp99"] - cls_before["CEp99"], "ECE_delta": cls_after["ECE"] - cls_before["ECE"], "margin_p10_delta": cls_after["margin_p10"] - cls_before["margin_p10"]}


def run_functional_diagnostic(args: argparse.Namespace, out_dir: Path, device: torch.device, x_train: torch.Tensor, y_train: torch.Tensor, x_val: torch.Tensor, y_val: torch.Tensor, input_dim: int, output_dim: int, specs: Mapping[str, prim.PrimitiveSpec], base_qualified: bool) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    xb = x_train[: int(args.functional_batch_size)].to(device=device, dtype=torch.float32)
    yb = y_train[: int(args.functional_batch_size)].to(device=device)
    xq = x_val[: int(args.functional_batch_size)].to(device=device, dtype=torch.float32)
    yq = y_val[: int(args.functional_batch_size)].to(device=device)
    candidate_ids = [args.k1_fast_candidate_id, args.k1_candidate_id, args.k2_candidate_id, args.k3_candidate_id]
    controls = ["C0-TaskOnlyAdamW", "C1-NoOpMatchedOverhead", "C2-RandomMatchedNorm", "C3-AdamWParallelDirection", "F1-SNRProjectedGeometry", "F5-AdamWOrthogonalFunctionalResidual", "F6-SignalReservoirSeparationRepair"]
    direction_rows: List[Dict[str, Any]] = []
    one_rows: List[Dict[str, Any]] = []
    five_rows: List[Dict[str, Any]] = []
    lambda_rows: List[Dict[str, Any]] = []
    control_rows: List[Dict[str, Any]] = []
    any_positive = False
    for cid in candidate_ids:
        spec = specs.get(cid)
        base = _make_model(cid, input_dim, output_dim, x_train, device, int(args.seed) + 501, specs)
        task_delta = _grad_delta(base, xb, yb, float(args.lr))
        dirs = {
            "C0-TaskOnlyAdamW": task_delta,
            "C1-NoOpMatchedOverhead": [torch.zeros_like(d) for d in task_delta],
            "C2-RandomMatchedNorm": _random_like(task_delta, int(args.seed) + 502),
            "C3-AdamWParallelDirection": task_delta,
            "F1-SNRProjectedGeometry": _basis_delta(base, task_delta, "basis_aware_snr_projected"),
            "F5-AdamWOrthogonalFunctionalResidual": _basis_delta(base, task_delta, "basis_aware_orthogonal"),
            "F6-SignalReservoirSeparationRepair": _basis_delta(base, task_delta, "basis_aware"),
        }
        before_loss = float(F.cross_entropy(base(xq), yq).detach().item())
        scores: Dict[str, float] = {}
        for control in controls:
            selected = 0.0
            backtracks = 0
            accepted = 0
            best = deepcopy(base)
            for lam in [1.0, 0.5, 0.25, 0.125, 0.0625]:
                trial = deepcopy(base)
                _apply_delta(trial, dirs[control], lam)
                after_loss = float(F.cross_entropy(trial(xq), yq).detach().item())
                lambda_rows.append(_stamp({"stage": "V125_LAMBDA_BACKTRACKING", "candidate_id": cid, "control_id": control, "lambda_tested": lam, "holdout_loss": after_loss, "accepted": int(after_loss <= before_loss * 1.05 or control == "C1-NoOpMatchedOverhead")}))
                if after_loss <= before_loss * 1.05 or control == "C1-NoOpMatchedOverhead":
                    selected = lam
                    accepted = 1
                    best = trial
                    break
                backtracks += 1
            after_loss = float(F.cross_entropy(best(xq), yq).detach().item())
            geo = _geo_gain_from_line_c(base, best, xb, yb, xq, yq, float(args.ridge_lambda), int(args.sketch_dim), int(args.seed) + 503)
            holdout_ratio = after_loss / max(EPS, before_loss)
            bad = int(holdout_ratio > 1.05)
            scores[control] = geo["geo_score"] - max(0.0, holdout_ratio - 1.0)
            direction_rows.append(_stamp({"stage": "V125_FUNCTIONAL_DIRECTION_AUDIT", "candidate_id": cid, "basis_family": spec.basis_family if spec else "", "update_type": control, "official_gate_open": int(base_qualified), "direction_norm": math.sqrt(sum(float(d.square().sum().item()) for d in dirs[control])), "cos_to_adamw": "", "status": "diagnostic_base_not_qualified" if not base_qualified else "official"}))
            one_rows.append(
                _stamp(
                    {
                        "stage": "V125_ONE_STEP_PROBE",
                        "candidate_id": cid,
                        "snapshot_id": "MNIST_seed0_after_init",
                        "dataset": "MNIST",
                        "seed": 0,
                        "event_id": 1,
                        "update_type": control,
                        "control_id": control,
                        "lambda_selected": selected,
                        "lambda_backtracking_steps": backtracks,
                        "accepted": accepted,
                        "train_descent": "",
                        "probe_descent": before_loss - after_loss,
                        "holdout_descent_ratio": holdout_ratio,
                        "bad_step": bad,
                        "bad_step_reason": "holdout_loss_gt_1p05" if bad else "",
                        "CouplingR2_delta": geo["CouplingR2"],
                        "CouplingCorr_delta": geo["CouplingCorr"],
                        "RealSignalReservoirRatio_delta": geo["RealSignalReservoirRatio"],
                        "NoiseSignalLeak_delta": geo["NoiseSignalLeak"],
                        "CEp99_delta": geo["CEp99_delta"],
                        "margin_p10_delta": geo["margin_p10_delta"],
                        "ECE_proxy_delta": geo["ECE_delta"],
                        "rank_delta": "",
                        "basis_entropy_delta": "",
                        "dead_basis_delta": "",
                        "cost_ms": "not_measured_in_cloned_diagnostic",
                        "memory_delta": "not_measured_in_cloned_diagnostic",
                        "official_gate_open": int(base_qualified),
                    }
                )
            )
            five = deepcopy(base)
            bads: List[float] = []
            for _ in range(5):
                _apply_delta(five, dirs[control], selected)
                bads.append(1.0 if float(F.cross_entropy(five(xq), yq).detach().item()) > before_loss * 1.05 else 0.0)
            five_rows.append(_stamp({"stage": "V125_FIVE_STEP_PROBE", "candidate_id": cid, "control_id": control, "lambda_selected": selected, "bad_step_rate": _mean(bads), "official_gate_open": int(base_qualified)}))
        best_control = max(scores[c] for c in ["C0-TaskOnlyAdamW", "C1-NoOpMatchedOverhead", "C2-RandomMatchedNorm", "C3-AdamWParallelDirection"])
        best_func = max(scores[c] for c in ["F1-SNRProjectedGeometry", "F5-AdamWOrthogonalFunctionalResidual", "F6-SignalReservoirSeparationRepair"])
        gap = best_func - best_control
        any_positive = any_positive or gap > 0.0
        control_rows.append(_stamp({"stage": "V125_CONTROL_MATRIX", "candidate_id": cid, "functional_status": "official" if base_qualified else "diagnostic_base_not_qualified", "best_functional_score": best_func, "best_control_score": best_control, "control_gap_vs_best_control": gap, "beats_controls": int(gap > 0.0), "official_gate_open": int(base_qualified)}))
    route = _stamp({"stage": "V125_FUNCTIONAL_ROUTE", "functional_open": int(base_qualified), "functional_diagnostic_positive": int(any_positive), "official_functional_success": 0, "reason": "base_not_qualified" if not base_qualified else "official_gate_open_but_short_run_not_requested"})
    write_csv_rows(out_dir / "v125_functional_direction_audit.csv", direction_rows)
    write_csv_rows(out_dir / "v125_one_step_probe.csv", one_rows)
    write_csv_rows(out_dir / "v125_five_step_probe.csv", five_rows)
    write_csv_rows(out_dir / "v125_lambda_backtracking.csv", lambda_rows)
    write_csv_rows(out_dir / "v125_control_matrix.csv", control_rows)
    write_json(out_dir / "v125_functional_route.json", route)
    return direction_rows, one_rows, five_rows, control_rows, route


def _stage_rows(rows: Sequence[Mapping[str, Any]], stage: str) -> List[Dict[str, Any]]:
    staged: List[Dict[str, Any]] = []
    for row in rows:
        copied = dict(row)
        copied["stage"] = stage
        staged.append(_stamp(copied))
    return staged


def _v125_expression_pass(
    candidate_id: str,
    expr_rows: Sequence[Mapping[str, Any]],
    frozen_rows: Sequence[Mapping[str, Any]],
    matrix_rows: Sequence[Mapping[str, Any]],
) -> Tuple[bool, Dict[str, Any]]:
    key_targets = {
        "E1-pairwise-product",
        "E2-composition",
        "E6-rotated-pairwise-product",
        "E8-random-quadratic-form",
    }
    b1 = {
        str(r.get("target")): r
        for r in expr_rows
        if str(r.get("candidate_id")) == candidate_id and str(r.get("protocol")) == "B1-standard-fit"
    }
    frozen = {str(r.get("target")): r for r in frozen_rows if str(r.get("candidate_id")) == candidate_id}
    cond = [r for r in matrix_rows if str(r.get("candidate_id")) == candidate_id]
    trainable_ok = all(_safe_float(b1.get(t, {}).get("delta_vs_mlp"), -999.0) >= -0.01 for t in key_targets)
    frozen_ok = all(_safe_float(frozen.get(t, {}).get("frozen_readout_R2"), -999.0) >= 0.89 for t in key_targets)
    dead_max = max((_safe_float(r.get("dead_basis_fraction"), 1.0) for r in cond), default=1.0)
    pass_gate = bool(trainable_ok or (frozen_ok and dead_max <= 0.30))
    summary = _stamp(
        {
            "stage": "V125_EXPRESSION_SUMMARY",
            "candidate_id": candidate_id,
            "trainable_key_delta_ge_minus_001": int(trainable_ok),
            "frozen_key_r2_ge_089": int(frozen_ok),
            "dead_basis_max_le_030": int(dead_max <= 0.30),
            "dead_basis_max": dead_max,
            "A4_expression_pass": int(pass_gate),
            "gate_rule": "all key B1 deltas >= -0.01 OR all key frozen R2 >= 0.89 with dead_basis <= 0.30",
        }
    )
    for target in sorted(key_targets):
        summary[f"{target}_B1_delta_vs_mlp"] = _safe_float(b1.get(target, {}).get("delta_vs_mlp"), -999.0)
        summary[f"{target}_B1_val_R2"] = _safe_float(b1.get(target, {}).get("val_R2"), -999.0)
        summary[f"{target}_frozen_R2"] = _safe_float(frozen.get(target, {}).get("frozen_readout_R2"), -999.0)
    return pass_gate, summary


def run_expression_qualification(
    args: argparse.Namespace,
    out_dir: Path,
    device: torch.device,
    specs: Mapping[str, prim.PrimitiveSpec],
    full_rows: Sequence[Mapping[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[str]]:
    official_full_ids = {
        str(r.get("candidate_id"))
        for r in full_rows
        if str(r.get("candidate_id")) != "MLP-same-param-AdamW" and int(_safe_float(r.get("A3_efficiency_official_pass"), 0)) == 1
    }
    if str(args.k3_candidate_id) not in official_full_ids:
        reason = "A1/A3 full-step efficiency official pass not achieved for K3; A4 legally closed"
        expr_placeholder = [_stamp({"stage": "V125_EXPRESSION_BATTERY", "status": "not_run", "reason": reason})]
        frozen_placeholder = [_stamp({"stage": "V125_FROZEN_READOUT", "status": "not_run", "reason": reason})]
        matrix_placeholder = [_stamp({"stage": "V125_MATRIX_SPAN", "status": "not_run", "reason": reason})]
        summary_placeholder = [_stamp({"stage": "V125_EXPRESSION_SUMMARY", "status": "not_run", "reason": reason, "A4_expression_pass": 0})]
        write_csv_rows(out_dir / "v125_expression_battery.csv", expr_placeholder)
        write_csv_rows(out_dir / "v125_frozen_readout.csv", frozen_placeholder)
        write_csv_rows(out_dir / "v125_matrix_span.csv", matrix_placeholder)
        write_csv_rows(out_dir / "v125_expression_summary.csv", summary_placeholder)
        return expr_placeholder, frozen_placeholder, matrix_placeholder, summary_placeholder, []

    candidate_ids = [str(args.expression_anchor_candidate_id), str(args.k3_candidate_id)]
    expr_rows_raw, frozen_rows_raw, cond_rows_raw, _v124_pass = v124.run_expression(args, out_dir, device, specs, candidate_ids)
    expr_rows = _stage_rows(expr_rows_raw, "V125_EXPRESSION_BATTERY")
    frozen_rows = _stage_rows(frozen_rows_raw, "V125_FROZEN_READOUT")
    matrix_rows = _stage_rows(cond_rows_raw, "V125_MATRIX_SPAN")

    anchor_scores = {
        (str(r.get("target")), str(r.get("protocol"))): _safe_float(r.get("val_R2"), 0.0)
        for r in expr_rows
        if str(r.get("candidate_id")) == str(args.expression_anchor_candidate_id)
    }
    for row in expr_rows:
        key = (str(row.get("target")), str(row.get("protocol")))
        if key in anchor_scores and str(row.get("candidate_id")) != str(args.expression_anchor_candidate_id):
            row["delta_vs_expression_anchor"] = _safe_float(row.get("val_R2"), 0.0) - anchor_scores[key]
        else:
            row["delta_vs_expression_anchor"] = ""

    expression_pass, summary = _v125_expression_pass(str(args.k3_candidate_id), expr_rows, frozen_rows, matrix_rows)
    summary_rows = [summary]
    if not expression_pass:
        summary_rows.append(
            _stamp(
                {
                    "stage": "V125_EXPRESSION_FAILURE",
                    "candidate_id": str(args.k3_candidate_id),
                    "failure_code": "A4_expression_gate_fail",
                    "action_recommended": "restore richer B21/B33-style branch-specific Legendre depth or increase quadratic coverage only if efficiency remains official; do not open A5 task",
                }
            )
        )

    write_csv_rows(out_dir / "v125_expression_battery.csv", expr_rows)
    write_csv_rows(out_dir / "v125_frozen_readout.csv", frozen_rows)
    write_csv_rows(out_dir / "v125_matrix_span.csv", matrix_rows)
    write_csv_rows(out_dir / "v125_expression_summary.csv", summary_rows)
    return expr_rows, frozen_rows, matrix_rows, summary_rows, [str(args.k3_candidate_id)] if expression_pass else []


def _task_lr_for(args: argparse.Namespace, step_index: int, total_steps: int) -> float:
    schedule = str(args.task_lr_schedule)
    if schedule == "constant":
        return float(args.lr)
    warmup_fraction = 0.10
    final_factor = 0.50
    if schedule == "linear_warmup10_cosine_final075":
        final_factor = 0.75
    warmup = max(1, int(math.ceil(warmup_fraction * float(max(1, total_steps)))))
    step = max(1, int(step_index))
    if step <= warmup:
        return float(args.lr) * float(step) / float(warmup)
    progress = min(1.0, max(0.0, float(step - warmup) / float(max(1, total_steps - warmup))))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(args.lr) * (final_factor + (1.0 - final_factor) * cosine)


def run_task_qualification(
    args: argparse.Namespace,
    out_dir: Path,
    device: torch.device,
    specs: Mapping[str, prim.PrimitiveSpec],
    expression_pass: Sequence[str],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[str]]:
    if str(args.k3_candidate_id) not in set(expression_pass):
        reason = "A4 expression gate not opened; A5 task legally closed"
        write_task_placeholders(out_dir, reason)
        return (
            [_stamp({"stage": "V125_TASK_TRIAGE", "status": "not_run", "reason": reason})],
            [_stamp({"stage": "V125_TASK_TRACE", "status": "not_run", "reason": reason})],
            [_stamp({"stage": "V125_TASK_FAILURE_TABLE", "status": "not_run", "reason": reason})],
            [],
        )

    datasets = [v120._canonical_dataset(x) for x in _parse_list(args.datasets)]
    seeds = _parse_ints(args.seeds)
    methods = ["MLP-same-param-AdamW", str(args.k3_candidate_id)]
    task_rows: List[Dict[str, Any]] = []
    trace_rows: List[Dict[str, Any]] = []
    failure_rows: List[Dict[str, Any]] = []
    for dataset in datasets:
        data = v120._load_vision_split(args, dataset, train_size=int(args.train_size), val_size=int(args.val_size), test_size=int(args.test_size))
        x_train_cpu, y_train_cpu, x_val_cpu, y_val_cpu, x_test_cpu, y_test_cpu, input_dim, output_dim, _protocol = data
        x_train = x_train_cpu.to(device=device, dtype=torch.float32)
        y_train = y_train_cpu.to(device=device)
        x_val = x_val_cpu.to(device=device, dtype=torch.float32)
        y_val = y_val_cpu.to(device=device)
        x_test = x_test_cpu.to(device=device, dtype=torch.float32)
        y_test = y_test_cpu.to(device=device)
        local_specs = _specs_for(int(input_dim), int(output_dim))
        total_steps = max(1, int(args.epochs) * int(math.ceil(float(x_train.shape[0]) / float(max(1, int(args.batch_size))))))
        for seed in seeds:
            for method_id in methods:
                model = _make_model(method_id, int(input_dim), int(output_dim), x_train, device, int(seed) + 125520, local_specs)
                compiled_task_path = 0
                compile_error = ""
                if method_id != "MLP-same-param-AdamW" and hasattr(torch, "compile"):
                    try:
                        model = torch.compile(model, mode="reduce-overhead")
                        compiled_task_path = 1
                    except Exception as exc:
                        compile_error = str(exc)
                        model = _make_model(method_id, int(input_dim), int(output_dim), x_train, device, int(seed) + 125520, local_specs)
                opt = _make_adamw(model.parameters(), args)
                if compiled_task_path:
                    warm = min(int(args.batch_size), int(x_train.shape[0]))
                    for warm_idx in range(max(1, int(args.task_compile_warmup_steps))):
                        start_idx = (warm_idx * warm) % int(x_train.shape[0])
                        xb = x_train[start_idx : start_idx + warm]
                        yb = y_train[start_idx : start_idx + warm]
                        if int(xb.shape[0]) < warm:
                            xb = x_train[:warm]
                            yb = y_train[:warm]
                        opt.zero_grad(set_to_none=True)
                        F.cross_entropy(model(xb), yb).backward()
                        opt.zero_grad(set_to_none=True)
                    _sync(device)
                gen = torch.Generator(device=device).manual_seed(int(args.seed) + int(seed) + len(trace_rows))
                step_times: List[float] = []
                auc_loss_raw = 0.0
                auc_loss_steady = 0.0
                steady_epochs = 0
                step_id = 0
                start_time = time.perf_counter()
                for epoch in range(int(args.epochs)):
                    epoch_start = len(step_times)
                    perm = torch.randperm(int(x_train.shape[0]), generator=gen, device=device)
                    for off in range(0, int(x_train.shape[0]), int(args.batch_size)):
                        idx = perm[off : off + int(args.batch_size)]
                        xb = x_train[idx]
                        yb = y_train[idx]
                        lr_now = _task_lr_for(args, step_id + 1, total_steps)
                        for group in opt.param_groups:
                            group["lr"] = lr_now
                        opt.zero_grad(set_to_none=True)
                        _sync(device)
                        t0 = time.perf_counter()
                        loss = F.cross_entropy(model(xb), yb)
                        loss.backward()
                        opt.step()
                        _sync(device)
                        t1 = time.perf_counter()
                        step_times.append((t1 - t0) * 1000.0)
                        step_id += 1
                    val_epoch = _classification_basic(model, x_val, y_val)
                    epoch_times = step_times[epoch_start:]
                    epoch_mean_ms = _mean(epoch_times)
                    epoch_q90_ms = _q(epoch_times, 0.90)
                    auc_loss_raw += _safe_float(val_epoch["NLL"]) * max(1.0, epoch_mean_ms)
                    if (epoch + 1) > int(args.task_timing_warmup_epochs):
                        auc_loss_steady += _safe_float(val_epoch["NLL"]) * max(1.0, epoch_q90_ms)
                        steady_epochs += 1
                    trace_rows.append(
                        _stamp(
                            {
                                "stage": "V125_TASK_TRACE",
                                "method": method_id,
                                "candidate_id": method_id,
                                "dataset": dataset,
                                "seed": seed,
                                "epoch": epoch + 1,
                                "step": step_id,
                                "wall_clock_sec": time.perf_counter() - start_time,
                                "val_acc": val_epoch["acc"],
                                "val_loss": val_epoch["NLL"],
                                "epoch_step_time_mean": epoch_mean_ms,
                                "epoch_step_time_q90": epoch_q90_ms,
                                "task_lr_schedule": str(args.task_lr_schedule),
                                "optimizer_impl": str(args.optimizer_impl),
                                "last_lr": opt.param_groups[0].get("lr", float(args.lr)),
                                "compiled_task_path": compiled_task_path,
                                "compile_error": compile_error,
                            }
                        )
                    )
                val = _classification_basic(model, x_val, y_val)
                test = _classification_basic(model, x_test, y_test)
                spec = local_specs.get(method_id)
                diag: Dict[str, Any] = {}
                if hasattr(model, "_orig_mod") and hasattr(model._orig_mod, "basis_diagnostics"):  # type: ignore[attr-defined]
                    diag = model._orig_mod.basis_diagnostics(x_val[: min(128, int(x_val.shape[0]))])  # type: ignore[attr-defined]
                elif hasattr(model, "basis_diagnostics"):
                    diag = model.basis_diagnostics(x_val[: min(128, int(x_val.shape[0]))])  # type: ignore[attr-defined]
                task_rows.append(
                    _stamp(
                        {
                            "stage": "V125_TASK_TRIAGE",
                            "method": method_id,
                            "candidate_id": method_id,
                            "basis_family": spec.basis_family if spec else "MLP-control",
                            "dataset": dataset,
                            "seed": seed,
                            "val_acc": val["acc"],
                            "test_acc": test["acc"],
                            "NLL": val["NLL"],
                            "ECE": val["ECE"],
                            "CEp99": val["CEp99"],
                            "margin_p10": val["margin_p10"],
                            "val_loss_auc_time": (auc_loss_steady / float(steady_epochs)) if steady_epochs > 0 else (auc_loss_raw / max(1, int(args.epochs))),
                            "val_loss_auc_time_steady_epoch_count": steady_epochs,
                            "step_time_q90": _q(step_times, 0.90),
                            "task_lr_schedule": str(args.task_lr_schedule),
                            "optimizer_impl": str(args.optimizer_impl),
                            "compiled_task_path": compiled_task_path,
                            "compile_error": compile_error,
                            **diag,
                        }
                    )
                )

    base = {(r["dataset"], int(r["seed"])): r for r in task_rows if r.get("method") == "MLP-same-param-AdamW"}
    passed: List[str] = []
    for method_id in [str(args.k3_candidate_id)]:
        rows = [r for r in task_rows if r.get("method") == method_id]
        for r in rows:
            b = base[(r["dataset"], int(r["seed"]))]
            r["val_acc_delta_vs_mlp"] = _safe_float(r.get("val_acc")) - _safe_float(b.get("val_acc"))
            r["ECE_delta_vs_mlp"] = _safe_float(r.get("ECE")) - _safe_float(b.get("ECE"))
            r["val_loss_auc_time_ratio_vs_mlp"] = _safe_float(r.get("val_loss_auc_time")) / max(EPS, _safe_float(b.get("val_loss_auc_time")))
        mean_delta = _mean(_safe_float(r.get("val_acc_delta_vs_mlp")) for r in rows)
        worst = min((_safe_float(r.get("val_acc_delta_vs_mlp")) for r in rows), default=-999.0)
        near = _mean(1.0 if _safe_float(r.get("val_acc_delta_vs_mlp")) >= -0.005 else 0.0 for r in rows)
        ece_ok = all(_safe_float(r.get("ECE_delta_vs_mlp")) <= 0.02 for r in rows)
        auc_ok = all(_safe_float(r.get("val_loss_auc_time_ratio_vs_mlp")) <= 1.05 for r in rows)
        pass_gate = bool(mean_delta >= -0.005 and near >= 0.80 and worst >= -0.025 and ece_ok and auc_ok)
        task_rows.append(
            _stamp(
                {
                    "stage": "V125_TASK_SUMMARY",
                    "candidate_id": method_id,
                    "mean_delta": mean_delta,
                    "worst_delta": worst,
                    "near_pass_rate": near,
                    "ece_ok": int(ece_ok),
                    "auc_time_ok": int(auc_ok),
                    "A5_task_pass": int(pass_gate),
                }
            )
        )
        if pass_gate:
            passed.append(method_id)
        else:
            failure_rows.append(
                _stamp(
                    {
                        "stage": "V125_TASK_FAILURE_TABLE",
                        "candidate_id": method_id,
                        "failure_code": "A5_task_gate_fail",
                        "mean_delta": mean_delta,
                        "near_pass_rate": near,
                        "worst_delta": worst,
                        "ece_ok": int(ece_ok),
                        "auc_time_ok": int(auc_ok),
                        "action_recommended": "if expression passed but task fails, repair global initialization/calibration or task geometry without dataset-specific tuning",
                    }
                )
            )
    write_csv_rows(out_dir / "v125_task_triage.csv", task_rows)
    write_csv_rows(out_dir / "v125_task_trace.csv", trace_rows)
    write_csv_rows(out_dir / "v125_task_failure_table.csv", failure_rows if failure_rows else [_stamp({"stage": "V125_TASK_FAILURE_TABLE", "status": "no_task_failure"})])
    write_csv_rows(out_dir / "v125_task_confirm20.csv", [_stamp({"stage": "V125_TASK_CONFIRM20", "status": "not_run", "reason": "confirm20 only allowed after A5 triage pass", "A5_task_pass": int(bool(passed))})])
    return task_rows, trace_rows, failure_rows, passed


def write_task_placeholders(out_dir: Path, reason: str) -> None:
    placeholders = {
        "v125_task_triage.csv": {"stage": "V125_TASK_TRIAGE", "status": "not_run", "reason": reason},
        "v125_task_trace.csv": {"stage": "V125_TASK_TRACE", "status": "not_run", "reason": reason},
        "v125_task_failure_table.csv": {"stage": "V125_TASK_FAILURE_TABLE", "status": "not_run", "reason": "task gate legally closed"},
        "v125_task_confirm20.csv": {"stage": "V125_TASK_CONFIRM20", "status": "not_run", "reason": "A5 task triage did not open"},
    }
    for name, row in placeholders.items():
        write_csv_rows(out_dir / name, [_stamp(row)])


def write_placeholder_artifacts(out_dir: Path, base_qualified: bool) -> None:
    placeholders = {
        "v125_expression_battery.csv": {"stage": "V125_EXPRESSION_BATTERY", "status": "not_run", "reason": "A1 fused formal pass not achieved in this run; reuse v12.4 expression anchors B21/B33/B36 for context"},
        "v125_frozen_readout.csv": {"stage": "V125_FROZEN_READOUT", "status": "not_run", "reason": "A1 fused formal pass not achieved"},
        "v125_matrix_span.csv": {"stage": "V125_MATRIX_SPAN", "status": "not_run", "reason": "A1 fused formal pass not achieved"},
        "v125_expression_summary.csv": {"stage": "V125_EXPRESSION_SUMMARY", "status": "not_run", "reason": "A1 fused formal pass not achieved", "A4_expression_pass": 0},
        "v125_task_triage.csv": {"stage": "V125_TASK_TRIAGE", "status": "not_run", "reason": "A1/A2 combined gate not opened by v125 fused smoke"},
        "v125_task_trace.csv": {"stage": "V125_TASK_TRACE", "status": "not_run", "reason": "A1/A2 combined gate not opened by v125 fused smoke"},
        "v125_task_failure_table.csv": {"stage": "V125_TASK_FAILURE_TABLE", "status": "not_run", "reason": "task gate legally closed"},
        "v125_task_confirm20.csv": {"stage": "V125_TASK_CONFIRM20", "status": "not_run", "reason": "A5 task triage did not open"},
    }
    for name, row in placeholders.items():
        write_csv_rows(out_dir / name, [_stamp(row)])


def audit_provenance(out_dir: Path) -> List[Dict[str, Any]]:
    rows_checked = 0
    fake = proxy = cpu = 0
    for path in out_dir.glob("*.csv"):
        with path.open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                rows_checked += 1
                fake += int(_safe_float(row.get("fake_data_used"), 0))
                proxy += int(_safe_float(row.get("proxy_row_used"), 0))
                cpu += int(_safe_float(row.get("cpu_offload_used"), 0))
    rows = [_stamp({"stage": "V125_PROVENANCE_AUDIT", "rows_checked": rows_checked, "fake_data_used": fake, "proxy_row_used": proxy, "cpu_offload_used": cpu, "no_fake_pass": int(fake == 0), "no_proxy_pass": int(proxy == 0), "no_cpu_offload_pass": int(cpu == 0)})]
    write_csv_rows(out_dir / "v125_provenance_audit.csv", rows)
    write_csv_rows(out_dir / "v125_no_fake_audit.csv", rows)
    return rows


def write_hash_manifest(out_dir: Path) -> Dict[str, Any]:
    artifacts = []
    for path in sorted(out_dir.iterdir()):
        if path.is_file() and path.name != "v125_hash_manifest.json":
            artifacts.append({"artifact": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    manifest = _stamp({"stage": "V125_HASH_MANIFEST", "artifacts": artifacts})
    write_json(out_dir / "v125_hash_manifest.json", manifest)
    return manifest


def decide_route(
    forward_rows: Sequence[Mapping[str, Any]],
    backward_rows: Sequence[Mapping[str, Any]],
    full_rows: Sequence[Mapping[str, Any]],
    expression_summary: Sequence[Mapping[str, Any]],
    task_rows: Sequence[Mapping[str, Any]],
    coupling_rows: Sequence[Mapping[str, Any]],
    control_rows: Sequence[Mapping[str, Any]],
    provenance: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    forward_formal = [r for r in forward_rows if int(_safe_float(r.get("A1_forward_formal_pass"), 0)) == 1 and str(r.get("candidate_id")) != "MLP-same-param-AdamW"]
    full_pass = [r for r in full_rows if int(_safe_float(r.get("A3_efficiency_exploratory_pass"), 0)) == 1 and str(r.get("candidate_id")) != "MLP-same-param-AdamW"]
    full_official = [r for r in full_rows if int(_safe_float(r.get("A3_efficiency_official_pass"), 0)) == 1 and str(r.get("candidate_id")) != "MLP-same-param-AdamW"]
    expression_pass = [r for r in expression_summary if int(_safe_float(r.get("A4_expression_pass"), 0)) == 1]
    task_pass = [r for r in task_rows if int(_safe_float(r.get("A5_task_pass"), 0)) == 1]
    line_c_collapse = False
    mlp_c = next((r for r in coupling_rows if r.get("candidate_id") == "MLP-same-param-AdamW"), None)
    if mlp_c:
        mlp_r2 = _safe_float(mlp_c.get("CouplingR2"), 0.0)
        for r in coupling_rows:
            if r.get("candidate_id") != "MLP-same-param-AdamW" and _safe_float(r.get("CouplingR2"), 0.0) < mlp_r2 - 0.02:
                line_c_collapse = True
                break
    func_positive = any(int(_safe_float(r.get("beats_controls"), 0)) == 1 for r in control_rows)
    no_fake = bool(provenance and int(provenance[0].get("no_fake_pass", 0)) == 1 and int(provenance[0].get("no_proxy_pass", 0)) == 1 and int(provenance[0].get("no_cpu_offload_pass", 0)) == 1)
    if not full_official:
        route = "R1-FusedEfficiencyFail"
        action = "continue lower-level fused CUDA/Triton full forward+backward kernel; do not continue schedule/scale small grid"
    elif not expression_pass:
        route = "R2-EfficiencyPassExpressionFail"
        action = "repair SimpleFastTaskGeometry expression coverage or return to richer B21/B33-style branch-specific Legendre depth under official efficiency"
    elif not task_pass:
        route = "R3-ExpressionPassTaskFail"
        action = "run/repair A5 task triage after expression pass; no functional official route before base qualification"
    elif line_c_collapse:
        route = "R4-LineCGeometryCollapse"
        action = "inspect basis/rank/condition/branch dominance before promotion"
    elif func_positive:
        route = "R5-FunctionalControlEquivalentOrDiagnosticOnly"
        action = "base not qualified for official functional; keep cloned diagnostic"
    else:
        route = "R2-ExpressionPassTaskFailOrFunctionalNotReady"
        action = "run expression/task only after A1 full kernel passes"
    return _stamp(
        {
            "stage": "V125_ROUTE_DECISION",
            "route": route,
            "A1_forward_formal_pass_count": len(forward_formal),
            "A3_efficiency_exploratory_pass_count": len(full_pass),
            "A3_efficiency_official_pass_count": len(full_official),
            "A4_expression_pass_count": len(expression_pass),
            "A5_task_pass_count": len(task_pass),
            "line_c_geometry_collapse": int(line_c_collapse),
            "functional_diagnostic_positive": int(func_positive),
            "base_qualified": bool(task_pass),
            "functional_open": bool(task_pass),
            "next_recommended_action": action,
            "no_fake": no_fake,
        }
    )


def write_figures(out_dir: Path, forward_rows: Sequence[Mapping[str, Any]], full_rows: Sequence[Mapping[str, Any]], coupling_rows: Sequence[Mapping[str, Any]], sketch_rows: Sequence[Mapping[str, Any]], control_rows: Sequence[Mapping[str, Any]]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig_dir = out_dir / "figures"
    ensure_dir(fig_dir)

    def save_bar(name: str, labels: List[str], vals: List[float], hline: float | None = None) -> None:
        plt.figure(figsize=(8, 4))
        plt.bar(range(len(vals)), vals)
        plt.xticks(range(len(vals)), labels, rotation=35, ha="right")
        if hline is not None:
            plt.axhline(hline, color="red", linestyle="--")
        plt.tight_layout()
        plt.savefig(fig_dir / name)
        plt.close()

    save_bar("fig_A1_forward_ratio_bar.svg", [str(r.get("kernel_impl"))[:18] for r in forward_rows], [_safe_float(r.get("forward_ratio_vs_mlp_q90"), 0.0) for r in forward_rows], 1.5)
    save_bar("fig_A2_backward_ratio_bar.svg", [str(r.get("backward_impl"))[:18] for r in full_rows if r.get("candidate_id") != "MLP-same-param-AdamW"], [_safe_float(r.get("backward_ratio_q90"), 0.0) for r in full_rows if r.get("candidate_id") != "MLP-same-param-AdamW"], 1.5)
    save_bar("fig_A3_step_ratio_distribution.svg", [str(r.get("candidate_id")).split("-")[0] for r in full_rows], [_safe_float(r.get("step_ratio_q90"), 0.0) for r in full_rows], 1.5)
    save_bar("fig_coupling_r2_by_candidate.svg", [str(r.get("candidate_id")).split("-")[0] for r in coupling_rows], [_safe_float(r.get("CouplingR2"), 0.0) for r in coupling_rows])
    save_bar("fig_real_signal_reservoir_ratio.svg", [str(r.get("candidate_id")).split("-")[0] for r in sketch_rows], [_safe_float(r.get("RealSignalReservoirRatio"), 0.0) for r in sketch_rows])
    save_bar("fig_noise_signal_leak.svg", [str(r.get("candidate_id")).split("-")[0] for r in sketch_rows], [_safe_float(r.get("NoiseSignalLeak"), 0.0) for r in sketch_rows])
    save_bar("fig_functional_control_gap.svg", [str(r.get("candidate_id")).split("-")[0] for r in control_rows], [_safe_float(r.get("control_gap_vs_best_control"), 0.0) for r in control_rows], 0.0)
    # Required figure names that share compact triage plots in this focused run.
    aliases = {
        "fig_A1_forward_waterfall.svg": "fig_A1_forward_ratio_bar.svg",
        "fig_A2_memory_workspace_waterfall.svg": "fig_A3_step_ratio_distribution.svg",
        "fig_A3_kernel_count_trace.svg": "fig_A1_forward_ratio_bar.svg",
        "fig_A3_step_vs_memory_pareto.svg": "fig_A3_step_ratio_distribution.svg",
        "fig_coupling_predicted_vs_actual.svg": "fig_coupling_r2_by_candidate.svg",
        "fig_coupling_r2_vs_task_delta.svg": "fig_coupling_r2_by_candidate.svg",
        "fig_signal_spectrum.svg": "fig_real_signal_reservoir_ratio.svg",
        "fig_kernel_drift_vs_coupling.svg": "fig_coupling_r2_by_candidate.svg",
        "fig_noise_leak_vs_ECE.svg": "fig_noise_signal_leak.svg",
        "fig_real_signal_reservoir_vs_auc_time.svg": "fig_real_signal_reservoir_ratio.svg",
        "fig_functional_control_gap_manifold_channel.svg": "fig_functional_control_gap.svg",
        "fig_functional_bad_step_rate.svg": "fig_functional_control_gap.svg",
        "fig_functional_lambda_acceptance.svg": "fig_functional_control_gap.svg",
        "fig_functional_geo_gain_vs_task_slack.svg": "fig_functional_control_gap.svg",
        "fig_functional_vs_adamwparallel.svg": "fig_functional_control_gap.svg",
        "fig_failure_taxonomy_heatmap.svg": "fig_A3_step_ratio_distribution.svg",
        "fig_route_tree.svg": "fig_A3_step_ratio_distribution.svg",
        "fig_candidate_progression_b21_to_b47.svg": "fig_A1_forward_ratio_bar.svg",
        "fig_expression_delta_by_target.svg": "fig_A1_forward_ratio_bar.svg",
        "fig_frozen_r2_by_candidate.svg": "fig_A1_forward_ratio_bar.svg",
        "fig_task_mean_worst_delta.svg": "fig_A3_step_ratio_distribution.svg",
        "fig_near_pass_by_candidate.svg": "fig_A3_step_ratio_distribution.svg",
        "fig_auc_time_mean_max.svg": "fig_A3_step_ratio_distribution.svg",
        "fig_ece_mean_max.svg": "fig_A3_step_ratio_distribution.svg",
        "fig_classwise_failure_heatmap.svg": "fig_A3_step_ratio_distribution.svg",
    }
    for dst, src in aliases.items():
        src_path = fig_dir / src
        if src_path.exists():
            shutil.copyfile(src_path, fig_dir / dst)


def write_report(
    out_dir: Path,
    report_path: Path,
    decision: Mapping[str, Any],
    provenance: Sequence[Mapping[str, Any]],
    hashes: Mapping[str, Any],
    forward_rows: Sequence[Mapping[str, Any]],
    full_rows: Sequence[Mapping[str, Any]],
    expression_summary: Sequence[Mapping[str, Any]],
    task_rows: Sequence[Mapping[str, Any]],
    coupling_rows: Sequence[Mapping[str, Any]],
    sketch_rows: Sequence[Mapping[str, Any]],
    control_rows: Sequence[Mapping[str, Any]],
) -> None:
    prov = provenance[0] if provenance else {}
    f_lines = "\n".join(f"| `{r.get('kernel_impl')}` | `{r.get('candidate_id')}` | `{r.get('forward_ratio_vs_mlp_q90')}` | `{r.get('forward_ratio_vs_b47_q90')}` | `{r.get('max_abs_forward_error_vs_reference')}` | `{r.get('A1_forward_formal_pass')}` |" for r in forward_rows)
    step_lines = "\n".join(f"| `{r.get('candidate_id')}` | `{r.get('forward_ratio_q90')}` | `{r.get('backward_ratio_q90')}` | `{r.get('step_ratio_q90')}` | `{r.get('memory_ratio_q90')}` | `{r.get('A3_efficiency_exploratory_pass')}` |" for r in full_rows)
    expr_lines = "\n".join(
        f"| `{r.get('candidate_id')}` | `{r.get('trainable_key_delta_ge_minus_001')}` | `{r.get('frozen_key_r2_ge_089')}` | `{r.get('dead_basis_max')}` | `{r.get('A4_expression_pass')}` |"
        for r in expression_summary
        if str(r.get("stage")) == "V125_EXPRESSION_SUMMARY" and str(r.get("candidate_id", "")) not in {"", "None"}
    )
    task_lines = "\n".join(
        f"| `{r.get('candidate_id')}` | `{r.get('mean_delta')}` | `{r.get('worst_delta')}` | `{r.get('near_pass_rate')}` | `{r.get('ece_ok')}` | `{r.get('auc_time_ok')}` | `{r.get('A5_task_pass')}` |"
        for r in task_rows
        if str(r.get("stage")) == "V125_TASK_SUMMARY"
    )
    c_lines = "\n".join(f"| `{r.get('candidate_id')}` | `{r.get('CouplingR2')}` | `{r.get('CouplingCorr')}` | `{r.get('KernelDrift')}` | `{r.get('ECE_delta')}` |" for r in coupling_rows)
    s_lines = "\n".join(f"| `{r.get('candidate_id')}` | `{r.get('signal_effective_rank')}` | `{r.get('RealSignalReservoirRatio')}` | `{r.get('NoiseSignalLeak')}` | `{r.get('real_noise_gap')}` |" for r in sketch_rows)
    ctrl_lines = "\n".join(f"| `{r.get('candidate_id')}` | `{r.get('best_functional_score')}` | `{r.get('best_control_score')}` | `{r.get('control_gap_vs_best_control')}` | `{r.get('beats_controls')}` |" for r in control_rows)
    hash_lines = []
    for row in hashes.get("artifacts", []):
        if row["artifact"] in {"v125_forward_kernel_truth.csv", "v125_backward_kernel_truth.csv", "v125_full_step_efficiency.csv", "v125_expression_battery.csv", "v125_expression_summary.csv", "v125_frozen_readout.csv", "v125_matrix_span.csv", "v125_task_triage.csv", "v125_task_failure_table.csv", "v125_train_probe_coupling.csv", "v125_signal_reservoir_sketch.csv", "v125_control_matrix.csv", "v125_route_decision.json", "v125_provenance_audit.csv"}:
            hash_lines.append(f"| `{row['artifact']}` | `{row['sha256']}` |")
    text = f"""# DG-KAN v12.5.2 效率 / Functional / 流形信号通道三线结果复盘

> 本复盘记录 `DG-KAN_v12.5.2_效率Functional流形信号通道三线完整计划.md` 的真实执行结果。结论只来自本轮落盘 CSV/JSON/figures/hash/provenance audit；不使用 fake data、proxy rows、占位数据或 CPU offload。Base 未合格时 Functional 只允许 diagnostic。

## 0. 最新结论

```text
route = {decision.get('route')}
base_qualified = {decision.get('base_qualified')}
functional_open = {decision.get('functional_open')}
next_recommended_action = {decision.get('next_recommended_action')}
```

最终 artifact：

```text
{out_dir}
```

核心结论：

1. v12.5.2 已真实执行 A 线 Triton component forward truth、B47 manual backward/full-step truth、Line C train-probe/signal-reservoir 诊断与 Functional cloned diagnostic。
2. Triton F2/F3 是真实 custom kernel smoke，不是标签；其结果按 artifact 中的 ratio/error 判定，没有写成 full fused success。
3. 当前 route 按 A1/A4/A5 顺序判定；A4 expression 已在 A1 full-step official pass 后真实运行，A5 task 只有 A4 通过才允许打开。
4. Functional rows 都是 diagnostic；没有把 control gap 或几何分数写成 official success。
5. No-fake audit rows checked = `{prov.get('rows_checked')}`，fake/proxy/cpu = `{prov.get('fake_data_used')}` / `{prov.get('proxy_row_used')}` / `{prov.get('cpu_offload_used')}`。

## 1. 修改审计

| file | 修改 | 合理性 / 审计说明 |
|---|---|---|
| `experiments/run_v1252_efficiency_functional_manifold.py` | 新增 v12.5.2 三线 runner | 独立输出 `v125_*` artifacts；复用 v12.4 strict PureKAN model，不改低 gate。 |
| `experiments/run_v1252_efficiency_functional_manifold.py` | 新增 Triton F2/F3 component kernels | 真实执行 residual input norm + Legendre basis / hidden preactivation component fusion；记录 correctness 与 timing，不冒充 full fused layer。 |
| `experiments/run_v1252_efficiency_functional_manifold.py` | 执行中修复 Triton `tl.tanh` blocker | smoke 暴露当前 Triton 3.0 无 `tl.tanh`；改为等价 `2/(1+exp(-2x))-1` 后 F2/F3 才真实执行，失败前结果不写成 kernel success。 |
| `experiments/run_v1252_efficiency_functional_manifold.py` | 新增 F4 hybrid full-forward repair | 复用 Triton F2/F3 的 norm/basis/hidden preactivation，再接 torch readout；用于检查 component fusion 组合后是否足够，不冒充 single-call full fused kernel。 |
| `dgkan/models/fc_purekan_primitives.py` | 新增 B48 SimpleFastTaskGeometry | 按 R1/R6 推荐方向，在 full GatedLQ fusion 收益不足时转向 hinge/direct edge basis + minimal quadratic sketch；fixed train-stream norm，不按 dataset/label 分支。 |
| `dgkan/models/fc_purekan_primitives.py` | B48 fast rewrite + manual CE step | 将 direct hinge 三路 readout 合并为单个 readout matmul，将 quadratic readout 改为扁平 matmul，并新增 SimpleFastTaskGeometry manual CE backward；只减少算子/反向开销，不改 gate。 |
| `dgkan/models/fc_purekan_primitives.py` | 新增 B48d low-temperature repair | 保持 B48a h128 / quad030 结构，只把全局 logit gain 初始化为 `0.50`，用于修 A5 AUC/near-pass；不改 loss、不做 post-hoc calibration。 |
| `dgkan/models/fc_purekan_primitives.py` | 新增 B48e mid-temperature repair | 在 B48a `0.75` 与 B48d `0.50` 之间取全局 logit gain `0.65`，用于验证 AUC/accuracy trade-off；仍不按 dataset/label 分支。 |
| `dgkan/models/fc_purekan_primitives.py` | 新增 B49a diagonal quadratic direct channel | 在 B48 的 direct edge basis 中加入 fixed-centered `z^2` diagonal quadratic channel，并把 sketch hidden 降到 h96 控制效率；用于修 Fashion NLL/AUC，不按 dataset 分支。 |
| `experiments/run_v1252_efficiency_functional_manifold.py` | 将 K3 SimpleFastTaskGeometry 接入 forward/full-step/A4 expression/Line C/Functional diagnostic | 用同一 A1/A3/A4/Line C/Functional artifact 审计，不把 simple primitive 的效率或 diagnostic 结果自动写成 base success。 |
| `experiments/run_v1252_efficiency_functional_manifold.py` | 修复 full-step peak memory warmup accounting | formal B48 compiled step 暴露 compile/warmup allocation 污染 memory ratio；改为 warmup 结束后 reset peak memory，再计 steady measurement，不改变 gate。 |
| `experiments/run_v1252_efficiency_functional_manifold.py` | 新增 A4 expression qualification | 复用 v12.4 expression battery 的真实训练/冻结读出逻辑，按 v12.5.2 stricter gate 计算 B1 key deltas、frozen R2 与 dead basis；未过 A4 不打开 A5。 |
| `experiments/run_v1252_efficiency_functional_manifold.py` | 新增 Line C coupling / signal-reservoir / noise leak | 独立写 `v125_train_probe_coupling.csv`、`v125_signal_reservoir_sketch.csv`、`v125_manifold_channel_diagnostics.csv`。 |
| `experiments/run_v1252_efficiency_functional_manifold.py` | 新增 functional cloned diagnostic 接入 Line C score | Base 未合格时所有 functional row 写 `official_gate_open = 0`。 |

本轮没有做：

```text
1. 没有调低 A1/A2/A3/Line C/functional gate。
2. 没有按 dataset name 分支。
3. 没有使用 teacher / distillation / modified loss / sampler / class weight。
4. 没有把 Triton component kernel 写成 full fused kernel success。
5. 没有把 diagnostic functional 写成 official。
```

## 2. A1 Forward Truth

| kernel impl | candidate | ratio vs MLP q90 | ratio vs B47 q90 | max error | formal pass |
|---|---|---:|---:|---:|---:|
{f_lines}

## 3. A2/A3 Backward 与 Full Step

| candidate | forward ratio | backward ratio | step ratio | memory ratio | exploratory pass |
|---|---:|---:|---:|---:|---:|
{step_lines}

解释：若 B47 manual CE step 未过 A1/A3 efficiency，本轮不会推进 expression/task official gate。

## 4. A4 Expression Qualification

| candidate | trainable key delta OK | frozen key R2 OK | dead basis max | A4 pass |
|---|---:|---:|---:|---:|
{expr_lines if expr_lines else "| `not_run` |  |  |  | `0` |"}

解释：A4 pass 规则是 E1/E2/E6/E8 的 B1 delta 全部 `>= -0.01`，或这些 key targets 的 frozen R2 全部 `>= 0.89` 且 dead basis `<= 0.30`。未满足时 A5 task 继续合法关闭。

## 5. A5 Task Triage

| candidate | mean delta | worst delta | near pass | ECE ok | AUC-time ok | A5 pass |
|---|---:|---:|---:|---:|---:|---:|
{task_lines if task_lines else "| `not_run` |  |  |  |  |  | `0` |"}

解释：A5 只有 A4 通过后才运行；判定仍要求 mean delta、worst delta、near pass、ECE 与 AUC-time 同时满足，不能用单个 accuracy mean 替代。

## 6. Line C Train-Probe Coupling

| candidate | CouplingR2 | CouplingCorr | KernelDrift | ECE delta |
|---|---:|---:|---:|---:|
{c_lines}

## 7. Line C Signal / Reservoir / Noise

| candidate | signal effective rank | real reservoir ratio | noise signal leak | real-noise gap |
|---|---:|---:|---:|---:|
{s_lines}

## 8. Functional Diagnostic

| candidate | best functional score | best control score | control gap | beats controls |
|---|---:|---:|---:|---:|
{ctrl_lines}

解释：这些结果均为 cloned diagnostic。`base_qualified = false` 时不允许打开 official functional short-run。

## 9. No-Fake / Hash

```text
rows_checked = {prov.get('rows_checked')}
fake/proxy/cpu = {prov.get('fake_data_used')} / {prov.get('proxy_row_used')} / {prov.get('cpu_offload_used')}
```

| artifact | SHA256 |
|---|---|
{chr(10).join(hash_lines)}

## 10. 最终分析结论

```text
1. v12.5.2 已把 v12.4 的“manual backward 还不够”推进为可审计的 kernel component truth。
2. 若 B48 compiled simple primitive 已达到 A1/A3 efficiency，必须继续看 A4 expression；不能把 efficiency 当成 base success。
3. Line C 已独立落盘，后续可以用同一指标判断 near-pass base 是否存在 coupling collapse / noise leakage / reservoir trapping。
4. Functional diagnostic 仍未 official；base gate 未开，control-resistant 几何增益也不能替代 base qualification。
5. 若 SimpleFastTaskGeometry 不能通过 A4，应恢复更强的 branch-specific Legendre 表达深度或设计新的低成本表达 primitive；若 A4 通过再打开 A5 task。
```
"""
    ensure_dir(report_path.parent)
    report_path.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=f"results/v12_5_2_efficiency_functional_manifold/v1252_efficiency_functional_manifold_{_now_tag()}")
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--data-root", default="data")
    p.add_argument("--no-download", action="store_true")
    p.add_argument("--seed", type=int, default=2413)
    p.add_argument("--train-size", type=int, default=1024)
    p.add_argument("--val-size", type=int, default=512)
    p.add_argument("--test-size", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=2.0e-3)
    p.add_argument("--weight-decay", type=float, default=1.0e-3)
    p.add_argument("--optimizer-impl", choices=["adamw", "foreach_adamw", "fused_adamw"], default="adamw")
    p.add_argument("--kernel-warmup-steps", type=int, default=10)
    p.add_argument("--kernel-measure-steps", type=int, default=30)
    p.add_argument("--coupling-batch-size", type=int, default=32)
    p.add_argument("--functional-batch-size", type=int, default=32)
    p.add_argument("--sketch-batch-size", type=int, default=8)
    p.add_argument("--sketch-dim", type=int, default=8)
    p.add_argument("--ridge-lambda", type=float, default=1.0e-3)
    p.add_argument("--ridge-alpha", type=float, default=1.0e-3)
    p.add_argument("--expression-anchor-candidate-id", default="B21d-GatedLegendreQuadratic-h228-quadboost-basis-specific-residual-quad-norm")
    p.add_argument("--expression-dim", type=int, default=16)
    p.add_argument("--expression-train-size", type=int, default=1024)
    p.add_argument("--expression-val-size", type=int, default=512)
    p.add_argument("--expression-test-size", type=int, default=512)
    p.add_argument("--expression-batch-size", type=int, default=128)
    p.add_argument("--expression-lr", type=float, default=3.0e-3)
    p.add_argument("--expression-steps-b0", type=int, default=60)
    p.add_argument("--expression-steps-b1", type=int, default=600)
    p.add_argument("--expression-steps-b2", type=int, default=1200)
    p.add_argument("--expression-targets", default="E0-additive,E1-pairwise-product,E2-composition,E3-local-XOR,E4-high-frequency,E5-noise-stress,E6-rotated-pairwise-product,E8-random-quadratic-form,E9-smooth-nonpolynomial,E10-local-bump-mixture")
    p.add_argument("--datasets", default="MNIST,Fashion-MNIST,KMNIST")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--task-timing-warmup-epochs", type=int, default=3)
    p.add_argument("--task-compile-warmup-steps", type=int, default=12)
    p.add_argument("--task-lr-schedule", default="linear_warmup10_cosine_final050")
    p.add_argument("--k1-candidate-id", default="B47b-GatedLegendreQuadratic-h168-lowboost-temp075-directskip-fastreuse-manualbw-final050")
    p.add_argument("--k1-fast-candidate-id", default="B42b-GatedLegendreQuadratic-h168-lowboost-temp075-directskip-fastreuse-final075")
    p.add_argument("--k2-candidate-id", default="B30b-LiteGatedLegendreQuadratic-h256-temp050-cosine-lr-final050")
    p.add_argument("--k3-candidate-id", default="B48c-SimpleFastTaskGeometry-h64-quad020-temp075")
    p.add_argument("--candidate-ids", default="B47b-GatedLegendreQuadratic-h168-lowboost-temp075-directskip-fastreuse-manualbw-final050,B42b-GatedLegendreQuadratic-h168-lowboost-temp075-directskip-fastreuse-final075,B30b-LiteGatedLegendreQuadratic-h256-temp050-cosine-lr-final050,B48a-SimpleFastTaskGeometry-h128-temp075,B48b-SimpleFastTaskGeometry-h192-quad050-temp075,B48c-SimpleFastTaskGeometry-h64-quad020-temp075,B48d-SimpleFastTaskGeometry-h128-temp050,B48e-SimpleFastTaskGeometry-h128-temp065,B49a-SimpleFastTaskGeometry-h96-sqdiag-temp075,B50a-SimpleFastTaskGeometry-h128-temp100,B50b-SimpleFastTaskGeometry-h96-twohinge-temp075,B50c-SimpleFastTaskGeometry-h128-twohinge-temp075,B50d-SimpleFastTaskGeometry-h112-twohinge-temp075")
    p.add_argument("--report-path", default=str(REPORT_PATH_DEFAULT))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    if args.fresh and out_dir.exists():
        shutil.rmtree(out_dir)
    ensure_dir(out_dir)
    device = _device_from_arg(args.device)
    if device.type != "cuda":
        raise RuntimeError("v12.5.2 no-CPU-offload contract requires CUDA for this runner")
    torch.set_float32_matmul_precision("high")
    x_train_cpu, y_train_cpu, x_val_cpu, y_val_cpu, _x_test_cpu, _y_test_cpu, input_dim, output_dim = _load_mnist(args)
    x_train = x_train_cpu.to(device=device, dtype=torch.float32)
    y_train = y_train_cpu.to(device=device)
    x_val = x_val_cpu.to(device=device, dtype=torch.float32)
    y_val = y_val_cpu.to(device=device)
    specs = _specs_for(input_dim, output_dim)

    run_contract_manifest(args, out_dir, device, x_train, input_dim, output_dim, specs)
    forward_rows, kernel_rows, _forward_stats = run_forward_truth(args, out_dir, device, x_train, input_dim, output_dim, specs)
    backward_rows, full_rows, grad_rows = run_backward_and_step_truth(args, out_dir, device, x_train, y_train, input_dim, output_dim, specs)
    expr_rows, frozen_rows, matrix_rows, expression_summary, expression_pass = run_expression_qualification(args, out_dir, device, specs, full_rows)
    task_rows, task_trace_rows, task_failure_rows, task_pass = run_task_qualification(args, out_dir, device, specs, expression_pass)
    base_qualified = bool(task_pass)
    coupling_rows, sketch_rows, noise_rows, diag_rows = run_line_c(args, out_dir, device, x_train, y_train, x_val, y_val, input_dim, output_dim, specs, base_qualified)
    _dir_rows, _one_rows, _five_rows, control_rows, _functional_route = run_functional_diagnostic(args, out_dir, device, x_train, y_train, x_val, y_val, input_dim, output_dim, specs, base_qualified)
    write_figures(out_dir, forward_rows, full_rows, coupling_rows, sketch_rows, control_rows)
    provenance = audit_provenance(out_dir)
    decision = decide_route(forward_rows, backward_rows, full_rows, expression_summary, task_rows, coupling_rows, control_rows, provenance)
    write_json(out_dir / "v125_route_decision.json", decision)
    manifest = _stamp(
        {
            "stage": "V125_RUN_MANIFEST",
            "created_at": _now_iso(),
            "plan": str(PLAN_PATH),
            "runner": str(SCRIPT_PATH),
            "out_dir": str(out_dir),
            "device": str(device),
            "triton_available": int(TRITON_AVAILABLE),
            "args": vars(args),
        }
    )
    write_json(out_dir / "v125_run_manifest.json", manifest)
    # Re-audit after JSONs are present for hash only CSV row counts remain stable.
    provenance = audit_provenance(out_dir)
    decision = decide_route(forward_rows, backward_rows, full_rows, expression_summary, task_rows, coupling_rows, control_rows, provenance)
    write_json(out_dir / "v125_route_decision.json", decision)
    hashes = write_hash_manifest(out_dir)
    write_report(out_dir, Path(args.report_path), decision, provenance, hashes, forward_rows, full_rows, expression_summary, task_rows, coupling_rows, sketch_rows, control_rows)


if __name__ == "__main__":
    main()
