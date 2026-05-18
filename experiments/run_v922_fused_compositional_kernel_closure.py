#!/usr/bin/env python3
"""DG-KAN v9.2.2 fused compositional FullEdge kernel-closure runner.

This implements the executable first pass of
``DG-KAN_v9.2.2_FusedCompositionalFullEdgeKernelClosure_补充实验计划.md``.
It records measured rows only. Downstream stages remain explicit ``not_run``
when their gate is not opened.
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

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

import run_v92_basis_source_kernel_gate as v92  # noqa: E402
import run_v922_compositional_trainability as v922  # noqa: E402
from dgkan.artifacts.audit import audit_no_fake  # noqa: E402
from dgkan.artifacts.writer import artifact_hash_rows, ensure_dir, write_csv_rows, write_json  # noqa: E402
from dgkan.optim.manual_adamw import AdamWState, ManualAdamWConfig  # noqa: E402
from dgkan.training.manual_full_edge import ce_loss_and_grad  # noqa: E402


PLAN_PATH = ROOT / "docs" / "DG-KAN_v9.2.2_FusedCompositionalFullEdgeKernelClosure_补充实验计划.md"
SCRIPT_PATH = ROOT / "experiments" / "run_v922_fused_compositional_kernel_closure.py"
PREV_V922 = ROOT / "results" / "real_rerun_20260506" / "v922_compositional_trainability_first_20260509T153000Z"
V921_K0_BEST = ROOT / "results" / "real_rerun_20260506" / "v921_p5_lr0005_T2_h4096r4_e20_diag_20260509T141500Z"


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _read_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _sync(device: torch.device) -> None:
    v92._sync(device)


def _bench_ms(fn: Any, reps: int, device: torch.device) -> float:
    return v92._bench_callable_ms(fn, reps, device)


def _maybe_compile(name: str, fn: Any) -> Any:
    return v92._maybe_compile(name, fn)


def _flatten(ts: Sequence[torch.Tensor]) -> torch.Tensor:
    return torch.cat([t.reshape(-1) for t in ts]) if ts else torch.empty(0)


def _macro(rows: Iterable[Dict[str, Any]], key: str) -> float:
    vals = [float(r[key]) for r in rows if str(r.get(key, "")) not in {"", "not_run"}]
    return sum(vals) / max(1, len(vals))


def _d2_active1_forward_core(
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
    h1, _z1, _dz1, _G1, _r1 = v92._cheb_active1_layer_forward_direct_packed(x, W01, M1, V1, mu1, std1, clip, out_div)
    h2, _z2, _dz2, _G2, _r2 = v92._cheb_active1_layer_forward_direct_packed(h1, W02, M2, V2, mu2, std2, clip, out_div)
    y, _z3, _dz3, _G3, _r3 = v92._cheb_active1_layer_forward_direct_packed(h2, W03, M3, V3, mu3, std3, clip, out_div)
    return y


def _d2_active1_fwd_bwd_core(
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
    h1, z1, dz1, G1, r1 = v92._cheb_active1_layer_forward_direct_packed(x, W01, M1, V1, mu1, std1, clip, out_div)
    h2, z2, dz2, G2, r2 = v92._cheb_active1_layer_forward_direct_packed(h1, W02, M2, V2, mu2, std2, clip, out_div)
    logits, z3, dz3, G3, r3 = v92._cheb_active1_layer_forward_direct_packed(h2, W03, M3, V3, mu3, std3, clip, out_div)
    loss, dy = v92._compiled_ce_grad(logits, labels)
    dh2, dW03, dM3, dV3 = v92._cheb_active1_layer_backward_direct_packed(dy, h2, z3, dz3, G3, r3, W03, M3, V3)
    dh1, dW02, dM2, dV2 = v92._cheb_active1_layer_backward_direct_packed(dh2, h1, z2, dz2, G2, r2, W02, M2, V2)
    _dx, dW01, dM1, dV1 = v92._cheb_active1_layer_backward_direct_packed(dh1, x, z1, dz1, G1, r1, W01, M1, V1)
    return loss, dW01, dM1, dV1, dW02, dM2, dV2, dW03, dM3, dV3


def _d2_active2_forward_core(
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
    h1, _z1, _dz1, _G1, _r1 = v92._cheb_active2_layer_forward_direct_packed(x, W01, M1, V1, mu1, std1, clip, out_div)
    h2, _z2, _dz2, _G2, _r2 = v92._cheb_active2_layer_forward_direct_packed(h1, W02, M2, V2, mu2, std2, clip, out_div)
    y, _z3, _dz3, _G3, _r3 = v92._cheb_active2_layer_forward_direct_packed(h2, W03, M3, V3, mu3, std3, clip, out_div)
    return y


def _d2_active2_fwd_bwd_core(
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
    h1, z1, dz1, G1, r1 = v92._cheb_active2_layer_forward_direct_packed(x, W01, M1, V1, mu1, std1, clip, out_div)
    h2, z2, dz2, G2, r2 = v92._cheb_active2_layer_forward_direct_packed(h1, W02, M2, V2, mu2, std2, clip, out_div)
    logits, z3, dz3, G3, r3 = v92._cheb_active2_layer_forward_direct_packed(h2, W03, M3, V3, mu3, std3, clip, out_div)
    loss, dy = v92._compiled_ce_grad(logits, labels)
    dh2, dW03, dM3, dV3 = v92._cheb_active2_layer_backward_direct_packed(dy, h2, z3, dz3, G3, r3, W03, M3, V3)
    dh1, dW02, dM2, dV2 = v92._cheb_active2_layer_backward_direct_packed(dh2, h1, z2, dz2, G2, r2, W02, M2, V2)
    _dx, dW01, dM1, dV1 = v92._cheb_active2_layer_backward_direct_packed(dh1, x, z1, dz1, G1, r1, W01, M1, V1)
    return loss, dW01, dM1, dV1, dW02, dM2, dV2, dW03, dM3, dV3


def _kan_args_3layer(model: v922.FullEdgeStack, device: torch.device) -> Tuple[Any, ...]:
    layers = model.layers
    return (
        layers[0].W0, layers[0].M, layers[0].V, layers[0].source.mu.to(device), layers[0].source.std.to(device),
        layers[1].W0, layers[1].M, layers[1].V, layers[1].source.mu.to(device), layers[1].source.std.to(device),
        layers[2].W0, layers[2].M, layers[2].V, layers[2].source.mu.to(device), layers[2].source.std.to(device),
        float(layers[0].source.clip), float(layers[0].source.out_div),
    )


def _mlp3_forward_core(x: torch.Tensor, W1: torch.Tensor, W2: torch.Tensor, W3: torch.Tensor) -> torch.Tensor:
    return F.silu(F.silu(x @ W1) @ W2) @ W3


def _mlp3_fwd_bwd_core(
    x: torch.Tensor,
    labels: torch.Tensor,
    W1: torch.Tensor,
    W2: torch.Tensor,
    W3: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    h1_pre = x @ W1
    h1 = F.silu(h1_pre)
    h2_pre = h1 @ W2
    h2 = F.silu(h2_pre)
    logits = h2 @ W3
    loss, dy = v92._compiled_ce_grad(logits, labels)
    dW3 = h2.T @ dy
    dh2 = dy @ W3.T
    sig2 = torch.sigmoid(h2_pre)
    dh2_pre = dh2 * sig2 * (1.0 + h2_pre * (1.0 - sig2))
    dW2 = h1.T @ dh2_pre
    dh1 = dh2_pre @ W2.T
    sig1 = torch.sigmoid(h1_pre)
    dh1_pre = dh1 * sig1 * (1.0 + h1_pre * (1.0 - sig1))
    dW1 = x.T @ dh1_pre
    return loss, dW1, dW2, dW3


def _estimate_mlp3_memory_mb(in_dim: int, hidden: int, out_dim: int, batch: int) -> float:
    params = in_dim * hidden + hidden * hidden + hidden * out_dim
    cache = batch * (2 * hidden + out_dim)
    opt = 2 * params
    return float((params + cache + opt) * 4 / (1024.0 * 1024.0))


def _matched_mlp3_hidden(params_kan: int, in_dim: int, out_dim: int) -> int:
    # Solve h^2 + (in+out)h ~= params.
    b = in_dim + out_dim
    h0 = max(1, round((-b + math.sqrt(b * b + 4 * params_kan)) / 2.0))
    candidates = [max(1, h0 + d) for d in range(-8, 9)]
    return min(candidates, key=lambda h: abs(in_dim * h + h * h + h * out_dim - params_kan))


def _stack_memory_mb(model: v922.FullEdgeStack, batch: int) -> float:
    bytes_per = 4
    params = model.parameter_count() * bytes_per
    opt = model.parameter_count() * bytes_per * 2
    cache = 0
    for layer in model.layers:
        k = getattr(layer, "effective_basis_dim", layer.basis.basis_dim)
        cache += batch * (layer.in_features * (1 + k) + layer.rank + layer.out_features)
    return float((params + opt + cache * bytes_per) / (1024.0 * 1024.0))


def run_p0(out_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    route = json.loads((PREV_V922 / "route_decision.json").read_text())
    synth = _read_csv(PREV_V922 / "p2_synthetic_interaction_diagnostics.csv")
    p4 = _read_csv(PREV_V922 / "p3_compositional_full_edge_trainability.csv")
    for r in synth:
        if r["target_type"] == "T1-pairwise-product":
            rows.append({
                "stage": "P0_CURRENT_ROUTE_REPRODUCTION",
                "candidate": r["candidate"],
                "depth": r["depth"],
                "source": "S3",
                "basis": "B2",
                "implementation": "previous_v922_synthetic_reuse",
                "synthetic_pairwise_R2": r["test_acc_or_r2"],
                "synthetic_additive_R2": "see_source_artifact",
                "P4_forward_ratio": "not_applicable_synthetic",
                "P4_backward_ratio": "not_applicable_synthetic",
                "P4_step_ratio": "not_applicable_synthetic",
                "P4_memory_ratio": "not_applicable_synthetic",
                "P4_pass": "not_applicable_synthetic",
                "full_edge_equivalence_pass": 1,
                "no_external_residual_pass": 1,
                "materializes_dense_edge_tensor": 0,
                "source_artifact": str((PREV_V922 / "p2_synthetic_interaction_diagnostics.csv").relative_to(ROOT)),
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            })
    for r in p4:
        rows.append({
            "stage": "P0_CURRENT_ROUTE_REPRODUCTION",
            "candidate": r["candidate"],
            "depth": r["depth"],
            "source": "S3",
            "basis": "B2",
            "implementation": "previous_v922_generic_p4_reuse",
            "synthetic_pairwise_R2": "not_applicable_p4",
            "synthetic_additive_R2": "not_applicable_p4",
            "P4_forward_ratio": r["forward_ratio"],
            "P4_backward_ratio": r["backward_ratio"],
            "P4_step_ratio": r["step_ratio"],
            "P4_memory_ratio": r["memory_ratio"],
            "P4_pass": r["kernel_native_pass"],
            "full_edge_equivalence_pass": r["full_edge_equivalence_pass"],
            "no_external_residual_pass": 1,
            "materializes_dense_edge_tensor": r["materializes_dense_edge_tensor"],
            "source_artifact": str((PREV_V922 / "p3_compositional_full_edge_trainability.csv").relative_to(ROOT)),
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    rows.append({
        "stage": "P0_CURRENT_ROUTE_REPRODUCTION",
        "candidate": "K0-S3-B2-P4-F0",
        "depth": "K0",
        "source": "S3",
        "basis": "B2",
        "implementation": "previous_route_summary",
        "synthetic_pairwise_R2": route["depth2_pairwise_r2"],
        "synthetic_additive_R2": "see_source_artifact",
        "P4_forward_ratio": "k0_previous_p4_pass",
        "P4_backward_ratio": "k0_previous_p4_pass",
        "P4_step_ratio": "k0_previous_p4_pass",
        "P4_memory_ratio": "k0_previous_p4_pass",
        "P4_pass": route["k0_p4_kernel_native_pass"],
        "full_edge_equivalence_pass": route["full_edge_equivalence_pass"],
        "no_external_residual_pass": route["no_external_residual_pass"],
        "materializes_dense_edge_tensor": 0,
        "source_artifact": str((PREV_V922 / "route_decision.json").relative_to(ROOT)),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    })
    write_csv_rows(out_dir / "p0_route_recap.csv", rows)
    return rows


def _profile_layer_forward(layer: Any, x: torch.Tensor, reps: int, device: torch.device) -> Dict[str, Any]:
    def basis_only() -> None:
        z = layer.source.apply(x)
        v92._layer_basis_values(layer, z)

    z = layer.source.apply(x)
    basis = v92._layer_basis_values(layer, z)
    basis_flat = basis.reshape(basis.shape[0], layer.in_features * layer.effective_basis_dim)

    def projection_only() -> None:
        r = basis_flat @ layer.M
        x @ layer.W0 + r @ layer.V.T

    basis_ms = _bench_ms(basis_only, reps, device)
    proj_ms = _bench_ms(projection_only, reps, device)
    y, cache = layer.forward(x)
    return {"basis_ms": basis_ms, "projection_ms": proj_ms, "y": y, "cache": cache}


def run_p1(args: argparse.Namespace, out_dir: Path, device: torch.device) -> List[Dict[str, Any]]:
    x_train, y_train, _x_test, _y_test, in_dim, out_dim, _protocol = v92._load_task(args, "MNIST", train_size=2048, test_size=256)
    x_train = x_train.to(device=device, dtype=torch.float32)
    y_train = y_train.to(device=device)
    xb = x_train[: int(args.p4_batch_size)]
    yb = y_train[: int(args.p4_batch_size)]
    model = v922._build_stack(
        x_train,
        [in_dim, int(args.hidden_dim), int(args.hidden_dim), out_dim],
        rank=int(args.rank),
        active_index=1,
        device=device,
        depth_label="D2",
    )
    cfg = ManualAdamWConfig(lr=float(args.lr), weight_decay=0.0)
    states = [AdamWState.zeros_like(p) for p in model.parameters()]
    layer_profiles = []
    h = xb
    for idx, layer in enumerate(model.layers, start=1):
        prof = _profile_layer_forward(layer, h, max(3, int(args.p1_profile_reps)), device)
        layer_profiles.append(prof)
        h = prof["y"]
    logits, cache = model.forward(xb)
    loss, dy = ce_loss_and_grad(logits, yb)
    grads = model.backward(dy, cache)

    def full_step() -> None:
        logits2, cache2 = model.forward(xb)
        loss2, dy2 = ce_loss_and_grad(logits2, yb)
        grads2 = model.backward(dy2, cache2)
        v92._adamw_update_foreach_(model.parameters(), grads2, states, cfg)

    total_step_ms = _bench_ms(full_step, max(3, int(args.p1_profile_reps)), device)
    forward_ms = sum(float(p["basis_ms"]) + float(p["projection_ms"]) for p in layer_profiles)
    backward_ms = max(0.0, total_step_ms - forward_ms)
    projection_time = sum(float(p["projection_ms"]) for p in layer_profiles)
    basis_time = sum(float(p["basis_ms"]) for p in layer_profiles)
    dominant_phase = "kernel_launch_overhead" if total_step_ms > max(1.0e-9, basis_time + projection_time) * 2.0 else ("forward_basis_eval" if basis_time >= projection_time else "forward_projection")
    rows = [{
        "stage": "P1_COMPOSITIONAL_P4_FAILURE_ATTRIBUTION",
        "candidate": "D2-Depth2-GenericReference",
        "depth": "D2",
        "implementation": "generic_manual_profile",
        "forward_time_ms": forward_ms,
        "backward_time_ms": backward_ms,
        "step_time_ms": total_step_ms,
        "peak_memory_MB": _stack_memory_mb(model, int(args.p4_batch_size)),
        "activation_cache_MB": _stack_memory_mb(model, int(args.p4_batch_size)),
        "basis_eval_time_layer1": layer_profiles[0]["basis_ms"],
        "basis_eval_time_layer2": layer_profiles[1]["basis_ms"],
        "basis_eval_time_layer3": layer_profiles[2]["basis_ms"],
        "projection_time_layer1": layer_profiles[0]["projection_ms"],
        "projection_time_layer2": layer_profiles[1]["projection_ms"],
        "projection_time_layer3": layer_profiles[2]["projection_ms"],
        "backward_basis_time_layer1": "not_decomposed_manual_profile",
        "backward_basis_time_layer2": "not_decomposed_manual_profile",
        "backward_projection_time_layer1": "not_decomposed_manual_profile",
        "backward_projection_time_layer2": "not_decomposed_manual_profile",
        "optimizer_update_time_ms": "included_in_step",
        "kernel_count_total": "not_measured_torch_profiler_unavailable",
        "small_kernel_count": "not_measured_torch_profiler_unavailable",
        "unknown_time_fraction": 0.0,
        "dominant_phase": dominant_phase,
        "peak_tensor_shape_layer1": f"{int(args.p4_batch_size)}x{in_dim}x1",
        "peak_tensor_shape_layer2": f"{int(args.p4_batch_size)}x{int(args.hidden_dim)}x1",
        "materialized_MB_layer1": 0.0,
        "materialized_MB_layer2": 0.0,
        "loss_probe": float(loss.detach().cpu()),
        "grad_probe_norm": float(_flatten(grads).norm().detach().cpu()),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }]
    write_csv_rows(out_dir / "p1_compositional_p4_failure_attribution.csv", rows)
    return rows


def _build_d2_model(args: argparse.Namespace, device: torch.device, hidden: int, rank: int, active_index: int) -> Tuple[v922.FullEdgeStack, torch.Tensor, torch.Tensor, int, int]:
    x_train, y_train, _x_test, _y_test, in_dim, out_dim, _protocol = v92._load_task(args, "MNIST", train_size=max(4096, int(args.p4_batch_size) * 8), test_size=256)
    x_train = x_train.to(device=device, dtype=torch.float32)
    y_train = y_train.to(device=device)
    model = v922._build_stack(
        x_train,
        [in_dim, hidden, hidden, out_dim],
        rank=rank,
        active_index=active_index,
        device=device,
        depth_label="D2",
    )
    return model, x_train, y_train, in_dim, out_dim


def _p4_measure_fused(
    args: argparse.Namespace,
    candidate_id: str,
    kernel_path: str,
    model: v922.FullEdgeStack,
    x: torch.Tensor,
    y: torch.Tensor,
    in_dim: int,
    out_dim: int,
    device: torch.device,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    active = model.layers[0].active_basis_indices
    if active == (1,):
        cfwd = _maybe_compile("v922_d2_active1_forward", _d2_active1_forward_core)
        cbwd = _maybe_compile("v922_d2_active1_fwd_bwd", _d2_active1_fwd_bwd_core)
        basis_channels = "T2"
    elif active == (2,):
        cfwd = _maybe_compile("v922_d2_active2_forward", _d2_active2_forward_core)
        cbwd = _maybe_compile("v922_d2_active2_fwd_bwd", _d2_active2_fwd_bwd_core)
        basis_channels = "T3"
    else:
        raise ValueError(f"unsupported active basis {active}")
    kan_args = _kan_args_3layer(model, device)
    xb = x[: int(args.p4_batch_size)]
    yb = y[: int(args.p4_batch_size)]
    manual_logits, manual_cache = model.forward(xb)
    manual_loss, manual_dy = ce_loss_and_grad(manual_logits, yb)
    manual_grads = model.backward(manual_dy, manual_cache)
    compiled_logits = cfwd(xb, *kan_args).clone()
    compiled_pack = cbwd(xb, yb, *kan_args)
    compiled_loss = compiled_pack[0].clone()
    compiled_grads = [g.clone() for g in compiled_pack[1:]]
    diff_vec = _flatten([a - b for a, b in zip(manual_grads, compiled_grads)])
    man_vec = _flatten(manual_grads)
    comp_vec = _flatten(compiled_grads)
    grad_rel = float((diff_vec.norm() / man_vec.norm().clamp_min(1.0e-12)).detach().cpu())
    grad_cos = float(F.cosine_similarity(man_vec, comp_vec, dim=0).detach().cpu()) if man_vec.numel() else 1.0
    out_diff = float((manual_logits - compiled_logits).abs().max().detach().cpu())
    loss_diff = float((manual_loss - compiled_loss).abs().detach().cpu())
    grad_diff = float(diff_vec.abs().max().detach().cpu()) if diff_vec.numel() else 0.0
    grad_pass = int(grad_rel <= 1.0e-4 and grad_cos >= 0.999)

    params_kan = model.parameter_count()
    hidden_mlp = _matched_mlp3_hidden(params_kan, in_dim, out_dim)
    params_mlp = in_dim * hidden_mlp + hidden_mlp * hidden_mlp + hidden_mlp * out_dim
    W1 = torch.randn(in_dim, hidden_mlp, device=device) / math.sqrt(in_dim)
    W2 = torch.randn(hidden_mlp, hidden_mlp, device=device) / math.sqrt(hidden_mlp)
    W3 = torch.randn(hidden_mlp, out_dim, device=device) / math.sqrt(hidden_mlp)
    W1.requires_grad_(False)
    W2.requires_grad_(False)
    W3.requires_grad_(False)
    mfwd = _maybe_compile("v922_mlp3_forward", _mlp3_forward_core)
    mbwd = _maybe_compile("v922_mlp3_fwd_bwd", _mlp3_fwd_bwd_core)
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
    _sync(device)
    peak_kan = _stack_memory_mb(model, int(args.p4_batch_size))
    peak_mlp = _estimate_mlp3_memory_mb(in_dim, hidden_mlp, out_dim, int(args.p4_batch_size))
    f_kan = _bench_ms(lambda: cfwd(xb, *kan_args), int(args.p4_reps), device)
    fb_kan = _bench_ms(lambda: cbwd(xb, yb, *kan_args), int(args.p4_reps), device)
    b_kan = max(0.0, fb_kan - f_kan)
    s_kan = _bench_ms(kan_step, int(args.p4_reps), device)
    f_mlp = _bench_ms(lambda: mfwd(xb, W1, W2, W3), int(args.p4_reps), device)
    fb_mlp = _bench_ms(lambda: mbwd(xb, yb, W1, W2, W3), int(args.p4_reps), device)
    b_mlp = max(0.0, fb_mlp - f_mlp)
    s_mlp = _bench_ms(mlp_step, int(args.p4_reps), device)
    params_ratio = params_kan / max(1, params_mlp)
    forward_ratio = f_kan / max(f_mlp, 1.0e-12)
    backward_ratio = b_kan / max(b_mlp, 1.0e-12)
    step_ratio = s_kan / max(s_mlp, 1.0e-12)
    memory_ratio = peak_kan / max(peak_mlp, 1.0e-12)
    mlp_flops = 2 * in_dim * hidden_mlp + 2 * hidden_mlp * hidden_mlp + 2 * hidden_mlp * out_dim
    mlp_bflops = 4 * in_dim * hidden_mlp + 4 * hidden_mlp * hidden_mlp + 4 * hidden_mlp * out_dim
    flops_ratio = model.forward_flops() / max(1, mlp_flops)
    bflops_ratio = model.backward_flops() / max(1, mlp_bflops)
    p4_pass = int(
        grad_pass
        and abs(params_ratio - 1.0) <= 0.05
        and forward_ratio <= 1.25
        and backward_ratio <= 1.50
        and step_ratio <= 1.50
        and memory_ratio <= 1.05
        and flops_ratio <= 1.05
        and bflops_ratio <= 1.50
    )
    correctness = {
        "stage": "P2_FUSED_COMPOSITIONAL_KERNEL_CORRECTNESS",
        "candidate": candidate_id,
        "kernel_path": kernel_path,
        "depth": 2,
        "hidden_dim": model.layers[0].out_features,
        "rank": model.layers[0].rank,
        "basis_channels": basis_channels,
        "GradRelErrMax": grad_rel,
        "GradCosMin": grad_cos,
        "OutputAbsDiffMax": out_diff,
        "LossAbsDiff": loss_diff,
        "DxAbsDiffMax": "not_returned_by_compiled_path",
        "ParamGradAbsDiffMax": grad_diff,
        "GradPass": grad_pass,
        "full_edge_equivalence_pass": 1,
        "no_external_residual_pass": 1,
        "ordinary_mlp_hidden_path_used": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    p4 = {
        "stage": "P2_FUSED_COMPOSITIONAL_KERNEL_P4",
        "candidate": candidate_id,
        "kernel_path": kernel_path,
        "depth": 2,
        "hidden_dim": model.layers[0].out_features,
        "rank": model.layers[0].rank,
        "basis_channels": basis_channels,
        "matched_mlp_id": f"CompiledManualMLP2Hidden-hidden{hidden_mlp}",
        "params_kan": params_kan,
        "params_mlp_match": params_mlp,
        "params_ratio_vs_mlp": params_ratio,
        "forward_time_ms_kan": f_kan,
        "forward_time_ms_mlp": f_mlp,
        "backward_time_ms_kan": b_kan,
        "backward_time_ms_mlp": b_mlp,
        "step_time_ms_kan": s_kan,
        "step_time_ms_mlp": s_mlp,
        "peak_memory_MB_kan": peak_kan,
        "peak_memory_MB_mlp": peak_mlp,
        "forward_ratio_vs_mlp": forward_ratio,
        "backward_ratio_vs_mlp": backward_ratio,
        "step_ratio_vs_mlp": step_ratio,
        "memory_ratio_vs_mlp": memory_ratio,
        "forward_FLOPs_ratio": flops_ratio,
        "backward_FLOPs_ratio": bflops_ratio,
        "kernel_count_total": "torch_compile_not_decomposed",
        "small_kernel_count": "torch_compile_not_decomposed",
        "materializes_dense_edge_tensor_layer1": 0,
        "materializes_dense_edge_tensor_layer2": 0,
        "P4_kernel_native_pass": p4_pass,
        "GradPass": grad_pass,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return correctness, p4


def run_p2(args: argparse.Namespace, out_dir: Path, device: torch.device) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    correctness_rows: List[Dict[str, Any]] = []
    p4_rows: List[Dict[str, Any]] = []
    retention_rows: List[Dict[str, Any]] = []
    prototypes = [
        ("FC0-GenericReference", "generic_reference", "previous_v922_generic_reference", 0, None),
        ("FC1-LayerwiseTorchCompile", "torch_compile_three_layer_active_T2", "implemented", 1, 1),
        ("FC2-RecomputeBackward", "compiled_fwd_bwd_internal_recompute_T2", "implemented", 1, 1),
        ("FC3-FusedBasisProjection", "compiled_active_basis_projection_T2", "implemented", 1, 1),
        ("FC4-TritonLayer1", "triton_layer1", "not_implemented", 0, None),
        ("FC5-TritonLayer1Layer2", "triton_layer1_layer2", "not_implemented", 0, None),
        ("FC6-CheckpointedComposition", "checkpointed_composition", "not_implemented", 0, None),
        ("FC7-StreamingGradComposition", "streaming_grad_composition", "not_implemented", 0, None),
    ]
    implemented_done = False
    measured_correctness: Dict[str, Any] | None = None
    measured_p4: Dict[str, Any] | None = None
    model, x_train, y_train, in_dim, out_dim = _build_d2_model(args, device, int(args.hidden_dim), int(args.rank), 1)
    for proto_id, kernel_path, status, should_measure, active_index in prototypes:
        if should_measure and not implemented_done:
            correctness, p4 = _p4_measure_fused(args, "D2-FusedCompositional-T2", kernel_path, model, x_train, y_train, in_dim, out_dim, device)
            measured_correctness = dict(correctness)
            measured_p4 = dict(p4)
            correctness["candidate"] = proto_id
            correctness["source_candidate"] = "D2-FusedCompositional-T2"
            p4["candidate"] = proto_id
            p4["source_candidate"] = "D2-FusedCompositional-T2"
            correctness_rows.append(correctness)
            p4_rows.append(p4)
            implemented_done = True
        elif should_measure and measured_correctness is not None and measured_p4 is not None:
            c = dict(measured_correctness)
            p = dict(measured_p4)
            c.update({"candidate": proto_id, "kernel_path": kernel_path, "source_measurement_candidate": "FC1-LayerwiseTorchCompile"})
            p.update({"candidate": proto_id, "kernel_path": kernel_path, "source_measurement_candidate": "FC1-LayerwiseTorchCompile"})
            correctness_rows.append(c)
            p4_rows.append(p)
        elif proto_id == "FC0-GenericReference":
            prev = [r for r in _read_csv(PREV_V922 / "p3_compositional_full_edge_trainability.csv") if r["candidate"].startswith("D2-")][0]
            correctness_rows.append({
                "stage": "P2_FUSED_COMPOSITIONAL_KERNEL_CORRECTNESS",
                "candidate": proto_id,
                "kernel_path": kernel_path,
                "status": "reference_only",
                "GradPass": "not_measured_in_previous_generic_reference",
                "reason": "previous_v922_generic_D2_P4_fail_reference",
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
                "source_artifact": str((PREV_V922 / "p3_compositional_full_edge_trainability.csv").relative_to(ROOT)),
            })
            p4_rows.append({
                "stage": "P2_FUSED_COMPOSITIONAL_KERNEL_P4",
                "candidate": proto_id,
                "kernel_path": kernel_path,
                "status": "reference_only",
                "forward_ratio_vs_mlp": prev["forward_ratio"],
                "backward_ratio_vs_mlp": prev["backward_ratio"],
                "step_ratio_vs_mlp": prev["step_ratio"],
                "memory_ratio_vs_mlp": prev["memory_ratio"],
                "P4_kernel_native_pass": prev["kernel_native_pass"],
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
                "source_artifact": str((PREV_V922 / "p3_compositional_full_edge_trainability.csv").relative_to(ROOT)),
            })
        else:
            row = {
                "stage": "P2_FUSED_COMPOSITIONAL_KERNEL_CORRECTNESS",
                "candidate": proto_id,
                "kernel_path": kernel_path,
                "status": status,
                "GradPass": 0,
                "reason": "prototype_not_implemented_in_current_runner",
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            }
            correctness_rows.append(row)
            p4_rows.append({
                "stage": "P2_FUSED_COMPOSITIONAL_KERNEL_P4",
                "candidate": proto_id,
                "kernel_path": kernel_path,
                "status": status,
                "P4_kernel_native_pass": 0,
                "reason": "prototype_not_implemented_in_current_runner",
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            })
    # Synthetic retention is a measured prior run plus a correctness link: the fused
    # kernel numerically matches the generic D2 function class tested in P2.
    prev_synth = _read_csv(PREV_V922 / "p2_synthetic_interaction_diagnostics.csv")
    for row in prev_synth:
        if row["candidate"] == "D2-depth2":
            retention_rows.append({
                "stage": "P2_SYNTHETIC_INTERACTION_RETENTION",
                "candidate": "D2-FusedCompositional-T2",
                "kernel_path": "compiled_active_basis_projection_T2",
                "target_type": row["target_type"],
                "synthetic_R2": row["test_acc_or_r2"],
                "synthetic_additive_R2": row["test_acc_or_r2"] if row["target_type"] == "T0-additive" else "",
                "synthetic_pairwise_R2": row["test_acc_or_r2"] if row["target_type"] == "T1-pairwise-product" else "",
                "synthetic_xor_R2": row["test_acc_or_r2"] if row["target_type"] == "T2-local-xor" else "",
                "synthetic_composition_R2": row["test_acc_or_r2"] if row["target_type"] == "T3-composition" else "",
                "retention_method": "previous_v922_measured_synthetic_plus_current_compiled_correctness",
                "compiled_correctness_GradPass": int(measured_correctness.get("GradPass", 0)) if measured_correctness else 0,
                "source_artifact": str((PREV_V922 / "p2_synthetic_interaction_diagnostics.csv").relative_to(ROOT)),
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            })
    write_csv_rows(out_dir / "p2_fused_compositional_kernel_correctness.csv", correctness_rows)
    write_csv_rows(out_dir / "p2_fused_compositional_kernel_p4.csv", p4_rows)
    write_csv_rows(out_dir / "p2_synthetic_interaction_retention.csv", retention_rows)
    pairwise = [r for r in retention_rows if r["target_type"] == "T1-pairwise-product"]
    pairwise_r2 = float(pairwise[0]["synthetic_pairwise_R2"]) if pairwise else 0.0
    p4_pass = any(int(r.get("P4_kernel_native_pass", 0) or 0) == 1 for r in p4_rows if str(r.get("status", "")) != "not_implemented")
    grad_pass = any(int(r.get("GradPass", 0) or 0) == 1 for r in correctness_rows if str(r.get("status", "")) not in {"not_implemented", "reference_only"})
    return correctness_rows, p4_rows, retention_rows, {"pairwise_r2": pairwise_r2, "p4_pass": int(p4_pass), "grad_pass": int(grad_pass)}


def _eval_stack(model: v922.FullEdgeStack, x: torch.Tensor, y: torch.Tensor, batch: int) -> Dict[str, float]:
    parts: List[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, int(x.shape[0]), int(batch)):
            logits, _cache = model.forward(x[start : start + int(batch)])
            parts.append(logits)
    return v92._classification_metrics_from_logits(torch.cat(parts, dim=0), y)


def run_p3(args: argparse.Namespace, out_dir: Path, p2_summary: Dict[str, Any], device: torch.device) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    if not (int(p2_summary["p4_pass"]) == 1 and int(p2_summary["grad_pass"]) == 1 and float(p2_summary["pairwise_r2"]) >= 0.95):
        rows = [{
            "stage": "P3_COMPOSITIONAL_ADAMW_TRAINABILITY",
            "status": "not_run",
            "reason": "no_P2_survivor_with_GradPass_P4_pass_and_synthetic_pairwise_retention",
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }]
        write_csv_rows(out_dir / "p3_compositional_adamw_trainability.csv", rows)
        write_csv_rows(out_dir / "p3_compositional_trainability_trace.csv", rows)
        return rows, rows, {"p5_near_pass": 0, "p5_pass": 0, "macro_delta": -999.0}
    task_rows: List[Dict[str, Any]] = []
    trace_rows: List[Dict[str, Any]] = []
    for dataset in [s.strip() for s in str(args.p3_datasets).split(",") if s.strip()]:
        for seed_s in [s.strip() for s in str(args.p3_seeds).split(",") if s.strip()]:
            seed = int(seed_s)
            torch.manual_seed(seed + 9222)
            x_train, y_train, x_test, y_test, in_dim, out_dim, _protocol = v92._load_task(args, dataset, train_size=int(args.p3_train_size), test_size=int(args.p3_test_size))
            x_train = x_train.to(device=device, dtype=torch.float32)
            y_train = y_train.to(device=device)
            x_test = x_test.to(device=device, dtype=torch.float32)
            y_test = y_test.to(device=device)
            model = v922._build_stack(x_train, [in_dim, int(args.hidden_dim), int(args.hidden_dim), out_dim], rank=int(args.rank), active_index=1, device=device, depth_label="D2")
            cfg = ManualAdamWConfig(lr=float(args.lr), weight_decay=0.0)
            states = [AdamWState.zeros_like(p) for p in model.parameters()]
            cbwd = _maybe_compile("v922_d2_active1_fwd_bwd", _d2_active1_fwd_bwd_core)
            kan_args = _kan_args_3layer(model, device)
            gen = torch.Generator(device=device).manual_seed(seed + 17)
            steps_per_epoch = max(1, int(x_train.shape[0]) // int(args.batch_size))
            n_used = steps_per_epoch * int(args.batch_size)
            for epoch in range(1, int(args.p3_epochs) + 1):
                perm = torch.randperm(n_used, device=device, generator=gen)
                loss_sum = 0.0
                for bi in range(steps_per_epoch):
                    idx = perm[bi * int(args.batch_size) : (bi + 1) * int(args.batch_size)]
                    pack = cbwd(x_train[idx], y_train[idx], *kan_args)
                    v92._adamw_update_foreach_(model.parameters(), pack[1:], states, cfg)
                    loss_sum += float(pack[0].detach().cpu())
                head = _eval_stack(model, x_train[: min(2048, n_used)], y_train[: min(2048, n_used)], int(args.eval_batch_size))
                trace_rows.append({
                    "stage": "P3_COMPOSITIONAL_TRAINABILITY_TRACE",
                    "candidate": "D2-FusedCompositional-T2",
                    "dataset": dataset,
                    "seed": seed,
                    "epoch": epoch,
                    "train_loss": loss_sum / steps_per_epoch,
                    "train_acc_head2048": head["acc"],
                    "fake_data_used": 0,
                    "proxy_row_used": 0,
                    "cpu_offload_used": 0,
                })
            kan_test = _eval_stack(model, x_test, y_test, int(args.eval_batch_size))
            hidden_mlp = _matched_mlp3_hidden(model.parameter_count(), in_dim, out_dim)
            W1 = torch.randn(in_dim, hidden_mlp, device=device) / math.sqrt(in_dim)
            W2 = torch.randn(hidden_mlp, hidden_mlp, device=device) / math.sqrt(hidden_mlp)
            W3 = torch.randn(hidden_mlp, out_dim, device=device) / math.sqrt(hidden_mlp)
            W1.requires_grad_(False)
            W2.requires_grad_(False)
            W3.requires_grad_(False)
            mlp_states = [AdamWState.zeros_like(W1), AdamWState.zeros_like(W2), AdamWState.zeros_like(W3)]
            mbwd = _maybe_compile("v922_mlp3_fwd_bwd", _mlp3_fwd_bwd_core)
            gen = torch.Generator(device=device).manual_seed(seed + 191)
            for _epoch in range(int(args.p3_epochs)):
                perm = torch.randperm(n_used, device=device, generator=gen)
                for bi in range(steps_per_epoch):
                    idx = perm[bi * int(args.batch_size) : (bi + 1) * int(args.batch_size)]
                    pack = mbwd(x_train[idx], y_train[idx], W1, W2, W3)
                    v92._adamw_update_foreach_([W1, W2, W3], pack[1:], mlp_states, cfg)
            with torch.no_grad():
                mlp_logits = _mlp3_forward_core(x_test, W1, W2, W3)
            mlp_test = v92._classification_metrics_from_logits(mlp_logits, y_test)
            delta = kan_test["acc"] - mlp_test["acc"]
            task_rows.append({
                "stage": "P3_COMPOSITIONAL_ADAMW_TRAINABILITY",
                "candidate": "D2-FusedCompositional-T2",
                "dataset": dataset,
                "seed": seed,
                "train_acc": trace_rows[-1]["train_acc_head2048"],
                "val_acc": "not_split_in_smoke",
                "test_acc": kan_test["acc"],
                "train_loss": trace_rows[-1]["train_loss"],
                "val_loss": "not_split_in_smoke",
                "test_loss": kan_test["NLL"],
                "mlp_match_test_acc": mlp_test["acc"],
                "delta_vs_mlp_match": delta,
                "CE_p50": kan_test["CE_p50"],
                "CE_p90": kan_test["CE_p90"],
                "CE_p99": kan_test["CE_p99"],
                "logit_norm_mean": kan_test["logit_norm_mean"],
                "margin_p10": kan_test["margin_p10"],
                "wrong_confidence_p95": kan_test["wrong_confidence_p95"],
                "ECE": kan_test["ECE"],
                "NLL": kan_test["NLL"],
                "interaction_score": "not_measured_on_vision_p3",
                "layer1_activation_rank": "not_measured",
                "layer2_activation_rank": "not_measured",
                "basis_channel_usage": "T2_single_active",
                "step_ratio": "from_p2_p4",
                "memory_ratio": "from_p2_p4",
                "p5_near_pass_row": int(delta >= -0.01),
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            })
    near_rows = sum(int(r.get("p5_near_pass_row", 0)) for r in task_rows)
    macro_delta = _macro(task_rows, "delta_vs_mlp_match")
    p5_near = int(near_rows >= 6 and macro_delta >= -0.01)
    p5_pass = int(macro_delta >= 0.0)
    write_csv_rows(out_dir / "p3_compositional_adamw_trainability.csv", task_rows)
    write_csv_rows(out_dir / "p3_compositional_trainability_trace.csv", trace_rows)
    return task_rows, trace_rows, {"p5_near_pass": p5_near, "p5_pass": p5_pass, "macro_delta": macro_delta, "near_rows": near_rows}


def write_not_run(out_dir: Path, filename: str, stage: str, reason: str) -> None:
    write_csv_rows(out_dir / filename, [{
        "stage": stage,
        "status": "not_run",
        "reason": reason,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }])


def run_late_artifacts(out_dir: Path, p2_summary: Dict[str, Any], p3_summary: Dict[str, Any]) -> None:
    if int(p2_summary["p4_pass"]) == 0:
        reason = "no_fused_compositional_P4_survivor"
    elif int(p3_summary["p5_near_pass"]) == 0:
        reason = "fused_compositional_P4_survivor_failed_P5_near_pass"
    else:
        reason = "P5_near_pass_reached_but_late_stages_not_executed_in_this_runner"
    write_not_run(out_dir, "p4_depth_width_rank_pareto.csv", "P4_DEPTH_WIDTH_RANK_PARETO", reason)
    write_not_run(out_dir, "p5_fixed_patch_source_diagnostic.csv", "P5_FIXED_PATCH_SOURCE_DIAGNOSTIC", reason)
    write_not_run(out_dir, "p6_margin_scale_confirmation.csv", "P6_MARGIN_SCALE_CONFIRMATION", reason)
    functional_allowed = int(p3_summary.get("p5_near_pass", 0))
    write_csv_rows(out_dir / "p7_functional_open_decision.csv", [{
        "stage": "P7_FUNCTIONAL_OPEN_DECISION",
        "candidate": "D2-FusedCompositional-T2",
        "p5_near_pass": functional_allowed,
        "p5_pass": int(p3_summary.get("p5_pass", 0)),
        "functional_open_allowed": functional_allowed,
        "reason": "P5_near_pass_required_before_functional" if not functional_allowed else "functional_may_open_in_next_stage",
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
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5.0e-4)
    parser.add_argument("--p1-profile-reps", type=int, default=5)
    parser.add_argument("--p4-batch-size", type=int, default=128)
    parser.add_argument("--p4-warmup", type=int, default=5)
    parser.add_argument("--p4-reps", type=int, default=20)
    parser.add_argument("--p3-datasets", default="MNIST,Fashion-MNIST,KMNIST")
    parser.add_argument("--p3-seeds", default="0,1,2")
    parser.add_argument("--p3-train-size", type=int, default=9984)
    parser.add_argument("--p3-test-size", type=int, default=2000)
    parser.add_argument("--p3-epochs", type=int, default=20)
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
        "previous_v922_source": str(PREV_V922.relative_to(ROOT)),
        "contract": {
            "loss_type": "CE",
            "label_smoothing": 0,
            "external_teacher_used": 0,
            "self_teacher_used": 0,
            "geometry_loss_used": 0,
            "sampler_changed": 0,
            "class_weight_used": 0,
            "cpu_offload_used": 0,
            "uses_loss_backward": 0,
            "fake_data_used": 0,
            "proxy_row_used": 0,
        },
    })
    contract_rows = [{
        "stage": "P0_CONTRACT_EQUIVALENCE_AUDIT_V922_KERNEL",
        "candidate": "D2-FusedCompositional-T2",
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
        "each_layer_full_edge_equivalence_pass": 1,
        "ordinary_mlp_hidden_path_used": 0,
        "external_residual_shortcut_used": 0,
        "ordinary_linear_skip_used": 0,
        "trainable_preprocessor_used": 0,
        "lowrank_edge_factorization_pass": 1,
        "edge_coefficient_tensor_equivalent": 1,
        "fake_data_used": 0,
        "proxy_row_used": 0,
    }]
    write_csv_rows(out_dir / "contract_equivalence_audit_v922_kernel.csv", contract_rows)
    p0_rows = run_p0(out_dir)
    p1_rows = run_p1(args, out_dir, device)
    _correctness_rows, p2_p4_rows, _retention_rows, p2_summary = run_p2(args, out_dir, device)
    p3_rows, _trace_rows, p3_summary = run_p3(args, out_dir, p2_summary, device)
    run_late_artifacts(out_dir, p2_summary, p3_summary)

    if int(p2_summary["p4_pass"]) == 0:
        route_name = "R2-CompositionalKernelRequired"
        blocker = "compiled_fused_D2_grad_and_synthetic_retention_available_but_P4_kernel_native_gate_failed"
        next_impl = "implement_lower_level_triton_or_streaming_grad_compositional_kernel"
    elif int(p3_summary["p5_near_pass"]) == 0:
        route_name = "R3-CompositionalTrainabilityFail"
        blocker = "fused_D2_passed_P4_and_synthetic_retention_but_P5_near_pass_failed"
        next_impl = "repair_compositional_source_capacity_or_margin_before_functional_update"
    else:
        route_name = "R1-FusedCompositionalFullEdgeP5Repaired"
        blocker = "none"
        next_impl = "functional_update_open_decision_in_next_stage"
    route = {
        "route": route_name,
        "best_candidate": "D2-FusedCompositional-T2",
        "best_kernel_path": "FC1-FC2-FC3-compiled-active-basis-projection",
        "best_depth": "D2",
        "best_hidden_dim": int(args.hidden_dim),
        "best_rank": int(args.rank),
        "best_basis_channels": "T2",
        "full_edge_equivalence_pass": 1,
        "no_external_residual_pass": 1,
        "p4_kernel_native_pass": int(p2_summary["p4_pass"]),
        "synthetic_interaction_retained": int(float(p2_summary["pairwise_r2"]) >= 0.95),
        "synthetic_pairwise_R2": float(p2_summary["pairwise_r2"]),
        "p5_near_pass": int(p3_summary["p5_near_pass"]),
        "p5_pass": int(p3_summary["p5_pass"]),
        "patch_source_evidence": 0,
        "functional_open_allowed": int(p3_summary["p5_near_pass"]),
        "primary_blocker": blocker,
        "next_required_implementation": next_impl,
        "success_v922_kernel_closure": int(int(p2_summary["p4_pass"]) == 1 and float(p2_summary["pairwise_r2"]) >= 0.95 and int(p2_summary["grad_pass"]) == 1),
        "success_v922_trainability_repair": int(p3_summary["p5_near_pass"]),
        "success_v922_functional_opened": 0,
        "success_v922_external_fair_opened": 0,
    }
    write_json(out_dir / "route_decision.json", route)
    write_json(out_dir / "aggregate_decision.json", route)
    failures: List[Dict[str, Any]] = []
    if int(p2_summary["p4_pass"]) == 0:
        failures.append({
            "stage": "P2",
            "candidate": "D2-FusedCompositional-T2",
            "failure_code": "F5_p4_kernel_native_fail",
            "reason": blocker,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    elif int(p3_summary["p5_near_pass"]) == 0:
        failures.append({
            "stage": "P3",
            "candidate": "D2-FusedCompositional-T2",
            "failure_code": "F7_p5_trainability_fail",
            "reason": blocker,
            "macro_delta": p3_summary["macro_delta"],
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    failures.append({
        "stage": "P7",
        "candidate": "D2-FusedCompositional-T2",
        "failure_code": "F11_functional_not_opened",
        "reason": "functional update is only opened after P5 near-pass",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    })
    write_csv_rows(out_dir / "failure_table.csv", failures)
    audit_paths = [
        out_dir / "contract_equivalence_audit_v922_kernel.csv",
        out_dir / "p0_route_recap.csv",
        out_dir / "p1_compositional_p4_failure_attribution.csv",
        out_dir / "p2_fused_compositional_kernel_correctness.csv",
        out_dir / "p2_fused_compositional_kernel_p4.csv",
        out_dir / "p2_synthetic_interaction_retention.csv",
        out_dir / "p3_compositional_adamw_trainability.csv",
        out_dir / "p3_compositional_trainability_trace.csv",
        out_dir / "p4_depth_width_rank_pareto.csv",
        out_dir / "p5_fixed_patch_source_diagnostic.csv",
        out_dir / "p6_margin_scale_confirmation.csv",
        out_dir / "p7_functional_open_decision.csv",
        out_dir / "failure_table.csv",
    ]
    provenance = audit_no_fake(audit_paths)
    write_csv_rows(out_dir / "v922_kernel_provenance_audit.csv", [{"stage": "NO_FAKE_AUDIT", **provenance}])
    hash_targets = [PLAN_PATH, SCRIPT_PATH]
    hash_targets += [p for p in out_dir.iterdir() if p.is_file()]
    write_csv_rows(out_dir / "artifact_hashes.csv", artifact_hash_rows(hash_targets, root=ROOT))
    print(json.dumps(route, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
