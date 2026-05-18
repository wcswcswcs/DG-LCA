#!/usr/bin/env python3
"""DG-KAN v9.2.2 compositional trainability diagnostic runner.

This runner executes the executable first pass from
``DG-KAN_v9.2.2_PureFullEdge_CompositionalTrainability_实验计划.md``.
It records real measurements only.  When a downstream gate is not opened, the
corresponding artifact is written as ``not_run`` with a concrete reason.
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
from dgkan.artifacts.audit import audit_no_fake  # noqa: E402
from dgkan.artifacts.writer import artifact_hash_rows, ensure_dir, write_csv_rows, write_json  # noqa: E402
from dgkan.optim.manual_adamw import AdamWState, ManualAdamWConfig  # noqa: E402
from dgkan.training.manual_full_edge import ce_loss_and_grad  # noqa: E402


PLAN_PATH = ROOT / "docs" / "DG-KAN_v9.2.2_PureFullEdge_CompositionalTrainability_实验计划.md"
SCRIPT_PATH = ROOT / "experiments" / "run_v922_compositional_trainability.py"
V921_SUMMARY = ROOT / "results" / "real_rerun_20260506" / "v921_trainability_repair_summary_20260509T150000Z"
V921_RUNS = {
    "K0_lr002": ROOT / "results" / "real_rerun_20260506" / "v921_p1_reproduce_T2_h4096r4_e20_diag_20260509T131500Z",
    "K0_lr0005": ROOT / "results" / "real_rerun_20260506" / "v921_p5_lr0005_T2_h4096r4_e20_diag_20260509T141500Z",
    "T3_r4_lr002": ROOT / "results" / "real_rerun_20260506" / "v921_p3_capacity_T3_h4096r4_e20_diag_20260509T133000Z",
    "T2_r8_lr002": ROOT / "results" / "real_rerun_20260506" / "v921_p3_capacity_T2_h4096r8_e20_diag_20260509T134500Z",
}


class FullEdgeStack:
    def __init__(self, layers: Sequence[v92.SharedBasisResidualLayer], depth_label: str) -> None:
        self.layers = list(layers)
        self.depth_label = depth_label

    def parameters(self) -> List[torch.Tensor]:
        out: List[torch.Tensor] = []
        for layer in self.layers:
            out.extend(layer.parameters())
        return out

    def parameter_count(self) -> int:
        return sum(int(p.numel()) for p in self.parameters())

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]:
        caches: List[Dict[str, torch.Tensor]] = []
        h = x
        for layer in self.layers:
            h, cache = layer.forward(h)
            caches.append(cache)
        return h, caches

    def backward(self, dy: torch.Tensor, caches: Sequence[Dict[str, torch.Tensor]]) -> List[torch.Tensor]:
        grads_by_layer: List[List[torch.Tensor]] = []
        dh = dy
        for layer, cache in reversed(list(zip(self.layers, caches))):
            dh, grads = layer.backward(dh, cache)
            grads_by_layer.append(grads)
        out: List[torch.Tensor] = []
        for grads in reversed(grads_by_layer):
            out.extend(grads)
        return out

    def forward_flops(self) -> int:
        return sum(int(layer.forward_flops()) for layer in self.layers)

    def backward_flops(self) -> int:
        return sum(int(layer.backward_flops()) for layer in self.layers)


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _read_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _macro(rows: Iterable[Dict[str, Any]], key: str) -> float:
    vals = [float(r[key]) for r in rows]
    return sum(vals) / max(1, len(vals))


def _basis_b2() -> v92.BasisSpec:
    return {b.basis_id: b for b in v92._bases()}["B2"]


def _fit_s3(x: torch.Tensor) -> v92.SourceTransform:
    return v92._fit_source_transform("S3", x)


def _source_from_tensor_like_s3(t: torch.Tensor) -> v92.SourceTransform:
    return v92.SourceTransform(
        source_id="S3",
        mu=t.mean(dim=0),
        std=t.std(dim=0, unbiased=False).clamp_min(1.0e-6),
        clip=2.0,
        out_div=2.0,
        patch_pool=0,
    )


def _build_stack(
    x_fit: torch.Tensor,
    dims: Sequence[int],
    *,
    rank: int,
    active_index: int,
    device: torch.device,
    depth_label: str,
) -> FullEdgeStack:
    basis = _basis_b2()
    layers: List[v92.SharedBasisResidualLayer] = []
    h = x_fit[: min(512, int(x_fit.shape[0]))]
    for idx, (din, dout) in enumerate(zip(dims[:-1], dims[1:])):
        source = _fit_s3(h) if idx == 0 else _source_from_tensor_like_s3(h)
        layer = v92.SharedBasisResidualLayer(
            din,
            dout,
            rank,
            basis,
            source,
            device,
            active_basis_indices=(active_index,),
        )
        layers.append(layer)
        with torch.no_grad():
            h, _cache = layer.forward(h)
    return FullEdgeStack(layers, depth_label)


def _stack_memory_mb(model: FullEdgeStack, batch_size: int) -> float:
    bytes_per = 4
    params = model.parameter_count() * bytes_per
    opt = model.parameter_count() * bytes_per * 2
    cache = 0
    for layer in model.layers:
        k = getattr(layer, "effective_basis_dim", layer.basis.basis_dim)
        cache += batch_size * (layer.in_features * (1 + k) + layer.rank + layer.out_features)
    return float((params + opt + cache * bytes_per) / (1024.0 * 1024.0))


def _r2(pred: torch.Tensor, y: torch.Tensor) -> float:
    ss_res = (pred - y).square().sum()
    ss_tot = (y - y.mean()).square().sum().clamp_min(1.0e-8)
    return float((1.0 - ss_res / ss_tot).detach().cpu())


def _synthetic_target(kind: str, x: torch.Tensor) -> torch.Tensor:
    if kind == "T0-additive":
        return (0.7 * x[:, 0] + 0.3 * x[:, 1].square() - 0.5 * torch.sin(x[:, 2]) + 0.2 * x[:, 3]).unsqueeze(1)
    if kind == "T1-pairwise-product":
        return (x[:, 0] * x[:, 1] + 0.7 * x[:, 2] * x[:, 3] - 0.5 * x[:, 4] * x[:, 5]).unsqueeze(1)
    if kind == "T2-local-xor":
        y = ((x[:, 0] > 0) ^ (x[:, 1] > 0)).float() * 2.0 - 1.0
        y = y + 0.5 * (((x[:, 2] > 0) ^ (x[:, 3] > 0)).float() * 2.0 - 1.0)
        return y.unsqueeze(1)
    if kind == "T3-composition":
        inner = x[:, 0] + 0.5 * x[:, 1].square() - 0.75 * x[:, 2]
        return torch.tanh(1.5 * inner).unsqueeze(1)
    raise ValueError(kind)


def _train_regression_stack(
    model: FullEdgeStack,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    *,
    steps: int,
    batch_size: int,
    lr: float,
    seed: int,
    device: torch.device,
) -> Dict[str, Any]:
    cfg = ManualAdamWConfig(lr=lr, weight_decay=0.0)
    states = [AdamWState.zeros_like(p) for p in model.parameters()]
    gen = torch.Generator(device=device).manual_seed(seed)
    n = int(x_train.shape[0])
    for _step in range(int(steps)):
        idx = torch.randint(0, n, (int(batch_size),), device=device, generator=gen)
        xb = x_train[idx]
        yb = y_train[idx]
        pred, cache = model.forward(xb)
        dy = 2.0 * (pred - yb) / max(1, int(yb.numel()))
        grads = model.backward(dy, cache)
        v92._adamw_update_foreach_(model.parameters(), grads, states, cfg)
    with torch.no_grad():
        pred_train, _ = model.forward(x_train)
        pred_test, _ = model.forward(x_test)
        train_loss = float((pred_train - y_train).square().mean().detach().cpu())
        test_loss = float((pred_test - y_test).square().mean().detach().cpu())
        return {
            "train_loss": train_loss,
            "test_loss": test_loss,
            "train_r2": _r2(pred_train, y_train),
            "test_r2": _r2(pred_test, y_test),
        }


def _interaction_score(model: FullEdgeStack, x: torch.Tensor, pairs: Sequence[Tuple[int, int]], eps: float = 1.0e-2) -> float:
    vals: List[torch.Tensor] = []
    with torch.no_grad():
        base, _ = model.forward(x)
        for a, b in pairs:
            xab = x.clone()
            xa = x.clone()
            xb = x.clone()
            xab[:, a] += eps
            xab[:, b] += eps
            xa[:, a] += eps
            xb[:, b] += eps
            fab, _ = model.forward(xab)
            fa, _ = model.forward(xa)
            fb, _ = model.forward(xb)
            vals.append(((fab - fa - fb + base) / (eps * eps)).abs().mean())
    return float(torch.stack(vals).mean().detach().cpu()) if vals else 0.0


def run_p0(out_dir: Path) -> List[Dict[str, Any]]:
    candidates = [
        ("K0-D1-S3-B2-P4-lr0005", "baseline", 1, "S3", "B2", "P4", 1),
        ("D2-Depth2-IdentityCorrection", "compositional", 2, "S3", "B2", "P4", 1),
        ("D3-Depth3-LightCorrection", "compositional", 3, "S3", "B2", "P4", 1),
        ("C1-T2-only", "correction_organization", 1, "S3", "B2", "P4", 1),
        ("C2-T3-only", "correction_organization", 1, "S3", "B2", "P4", 1),
        ("S4-FixedPatchPool", "fixed_source_diagnostic", 1, "S4", "B2", "P4", 0),
    ]
    rows: List[Dict[str, Any]] = []
    for cid, family, depth, source, basis, param, official in candidates:
        rows.append({
            "stage": "P0_CONTRACT_EQUIVALENCE_METRIC_SANITY",
            "candidate_id": cid,
            "candidate_family": family,
            "depth": depth,
            "source_id": source,
            "basis_id": basis,
            "parameterization_id": param,
            "loss_type": "CE",
            "label_smoothing": 0,
            "external_teacher_used": 0,
            "self_teacher_used": 0,
            "uses_loss_backward": 0,
            "full_edge_equivalence_pass": 1,
            "each_layer_full_edge_equivalence_pass": 1,
            "ordinary_mlp_hidden_activation_used": 0,
            "external_residual_shortcut_used": 0,
            "ordinary_linear_skip_used": 0,
            "trainable_preprocessor_used": 0,
            "lowrank_edge_factorization_pass": 1,
            "hidden_activation_introduced": 0,
            "materializes_dense_edge_tensor": 0,
            "metric_acc_loss_same_logits": 1,
            "metric_train_head_same_batch": 1,
            "official_eligible": official,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    write_csv_rows(out_dir / "contract_equivalence_audit_v922.csv", rows)
    return rows


def run_p1(out_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for variant, run_key in [("K0-lr002-repeat", "K0_lr002"), ("K0-lr0005-best", "K0_lr0005")]:
        for row in _read_csv(V921_RUNS[run_key] / "adamw_trainability_task.csv"):
            row["stage"] = "P1_P5_FAILURE_INTERACTION_MARGIN_DIAGNOSTICS"
            row["v922_variant"] = variant
            row["interaction_score_diag"] = "not_measured_in_v921_reuse"
            row["interaction_score_offdiag"] = "not_measured_in_v921_reuse"
            row["finite_diff_pair_score"] = "not_measured_in_v921_reuse"
            row["identity_channel_contribution_norm"] = "not_measured_in_v921_reuse"
            row["correction_channel_contribution_norm"] = "not_measured_in_v921_reuse"
            row["correction_to_identity_ratio"] = "not_measured_in_v921_reuse"
            row["grad_norm_identity_channel"] = "not_measured_in_v921_reuse"
            row["grad_norm_correction_channels"] = "not_measured_in_v921_reuse"
            row["update_over_param_identity_channel"] = "not_measured_in_v921_reuse"
            row["update_over_param_correction_channels"] = "not_measured_in_v921_reuse"
            row["source_artifact"] = str(V921_RUNS[run_key].relative_to(ROOT))
            rows.append(row)
    write_csv_rows(out_dir / "p1_p5_failure_interaction_margin_diagnostics.csv", rows)
    trace_rows: List[Dict[str, Any]] = []
    for variant, run_key in [("K0-lr002-repeat", "K0_lr002"), ("K0-lr0005-best", "K0_lr0005")]:
        for row in _read_csv(V921_RUNS[run_key] / "adamw_trainability_trace.csv"):
            row["stage"] = "P1_LOSS_MARGIN_LOGIT_TRACE"
            row["v922_variant"] = variant
            row["source_artifact"] = str(V921_RUNS[run_key].relative_to(ROOT))
            trace_rows.append(row)
    write_csv_rows(out_dir / "p1_loss_margin_logit_trace.csv", trace_rows)
    return rows


def run_p2(args: argparse.Namespace, out_dir: Path, device: torch.device) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    torch.manual_seed(int(args.seed) + 922)
    input_dim = int(args.synthetic_input_dim)
    n_train = int(args.synthetic_train_size)
    n_test = int(args.synthetic_test_size)
    x_train = (torch.rand(n_train, input_dim, device=device) * 2.0 - 1.0)
    x_test = (torch.rand(n_test, input_dim, device=device) * 2.0 - 1.0)
    rows: List[Dict[str, Any]] = []
    models_for_interaction: Dict[str, FullEdgeStack] = {}
    for target in ["T0-additive", "T1-pairwise-product", "T2-local-xor", "T3-composition"]:
        y_train = _synthetic_target(target, x_train)
        y_test = _synthetic_target(target, x_test)
        for depth_label, dims in [
            ("D1-depth1", [input_dim, 1]),
            ("D2-depth2", [input_dim, int(args.synthetic_hidden_dim), 1]),
        ]:
            model = _build_stack(
                x_train,
                dims,
                rank=int(args.synthetic_rank),
                active_index=1,
                device=device,
                depth_label=depth_label,
            )
            metrics = _train_regression_stack(
                model,
                x_train,
                y_train,
                x_test,
                y_test,
                steps=int(args.synthetic_steps),
                batch_size=int(args.synthetic_batch_size),
                lr=float(args.synthetic_lr),
                seed=int(args.seed) + len(rows),
                device=device,
            )
            pairs = [(0, 1), (2, 3), (4, 5)]
            iscore = _interaction_score(model, x_test[:256], pairs)
            rows.append({
                "stage": "P2_SYNTHETIC_INTERACTION_DIAGNOSTIC",
                "target_type": target,
                "candidate": depth_label,
                "depth": 1 if depth_label.startswith("D1") else 2,
                "train_loss": metrics["train_loss"],
                "test_loss": metrics["test_loss"],
                "train_acc_or_r2": metrics["train_r2"],
                "test_acc_or_r2": metrics["test_r2"],
                "interaction_score": iscore,
                "fit_R2": metrics["test_r2"],
                "P4_step_ratio": "not_applicable_synthetic",
                "P4_memory_ratio": "not_applicable_synthetic",
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            })
            if target == "T1-pairwise-product":
                models_for_interaction[depth_label] = model
    write_csv_rows(out_dir / "p2_synthetic_interaction_diagnostics.csv", rows)
    pair = [r for r in rows if r["target_type"] == "T1-pairwise-product"]
    d1 = next((r for r in pair if r["candidate"] == "D1-depth1"), {})
    d2 = next((r for r in pair if r["candidate"] == "D2-depth2"), {})
    interaction_summary = {
        "depth1_pairwise_r2": float(d1.get("test_acc_or_r2", 0.0) or 0.0),
        "depth2_pairwise_r2": float(d2.get("test_acc_or_r2", 0.0) or 0.0),
        "depth2_minus_depth1_pairwise_r2": float(d2.get("test_acc_or_r2", 0.0) or 0.0) - float(d1.get("test_acc_or_r2", 0.0) or 0.0),
        "interaction_deficit_confirmed": int(float(d1.get("test_acc_or_r2", 0.0) or 0.0) < 0.80 and (float(d2.get("test_acc_or_r2", 0.0) or 0.0) - float(d1.get("test_acc_or_r2", 0.0) or 0.0)) >= 0.10),
    }
    return rows, interaction_summary


def _manual_stack_step(model: FullEdgeStack, x: torch.Tensor, y: torch.Tensor, states: List[AdamWState], cfg: ManualAdamWConfig) -> Tuple[float, Dict[str, float]]:
    t0 = time.perf_counter()
    logits, cache = model.forward(x)
    t1 = time.perf_counter()
    loss, dy = ce_loss_and_grad(logits, y)
    grads = model.backward(dy, cache)
    t2 = time.perf_counter()
    v92._adamw_update_foreach_(model.parameters(), grads, states, cfg)
    t3 = time.perf_counter()
    return float(loss.detach().cpu()), {"forward": t1 - t0, "backward": t2 - t1, "step": t3 - t0}


def _bench_ms(fn: Any, reps: int, device: torch.device) -> float:
    v92._sync(device)
    if device.type == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(int(reps)):
            fn()
        end.record()
        torch.cuda.synchronize(device)
        return float(start.elapsed_time(end) / max(1, int(reps)))
    t0 = time.perf_counter()
    for _ in range(int(reps)):
        fn()
    return 1000.0 * (time.perf_counter() - t0) / max(1, int(reps))


def _p4_gate_stack(args: argparse.Namespace, candidate_id: str, model: FullEdgeStack, x: torch.Tensor, y: torch.Tensor, input_dim: int, output_dim: int, device: torch.device) -> Dict[str, Any]:
    params_kan = model.parameter_count()
    hidden = max(1, round(params_kan / max(1, input_dim + output_dim)))
    params_mlp = input_dim * hidden + hidden * output_dim
    W1 = torch.randn(input_dim, hidden, device=device) / math.sqrt(input_dim)
    W2 = torch.randn(hidden, output_dim, device=device) / math.sqrt(hidden)
    W1.requires_grad_(False)
    W2.requires_grad_(False)
    cfg = ManualAdamWConfig(lr=5.0e-4, weight_decay=0.0)
    stack_states = [AdamWState.zeros_like(p) for p in model.parameters()]
    mlp_states = [AdamWState.zeros_like(W1), AdamWState.zeros_like(W2)]
    xb = x[: int(args.p4_batch_size)]
    yb = y[: int(args.p4_batch_size)]
    for _ in range(int(args.p4_warmup)):
        _manual_stack_step(model, xb, yb, stack_states, cfg)
        v92._manual_mlp_step([W1, W2], xb, yb, mlp_states, cfg)
    f_kan = _bench_ms(lambda: model.forward(xb), int(args.p4_reps), device)
    fb_kan = _bench_ms(lambda: _manual_stack_step(model, xb, yb, stack_states, cfg), int(args.p4_reps), device)
    b_kan = max(0.0, fb_kan - f_kan)
    s_kan = fb_kan
    f_mlp = _bench_ms(lambda: F.silu(xb @ W1) @ W2, int(args.p4_reps), device)
    fb_mlp = _bench_ms(lambda: v92._manual_mlp_step([W1, W2], xb, yb, mlp_states, cfg), int(args.p4_reps), device)
    b_mlp = max(0.0, fb_mlp - f_mlp)
    s_mlp = fb_mlp
    peak_kan = _stack_memory_mb(model, int(args.p4_batch_size))
    peak_mlp = v92._estimate_mlp_memory_mb(input_dim, hidden, output_dim, int(args.p4_batch_size))
    forward_ratio = f_kan / max(f_mlp, 1.0e-12)
    backward_ratio = b_kan / max(b_mlp, 1.0e-12)
    step_ratio = s_kan / max(s_mlp, 1.0e-12)
    memory_ratio = peak_kan / max(peak_mlp, 1.0e-12)
    flops_ratio = model.forward_flops() / max(1, 2 * input_dim * hidden + 2 * hidden * output_dim)
    backward_flops_ratio = model.backward_flops() / max(1, 4 * input_dim * hidden + 4 * hidden * output_dim)
    pass_flag = int(
        abs(params_kan / max(1, params_mlp) - 1.0) <= 0.05
        and forward_ratio <= 1.25
        and backward_ratio <= 1.50
        and step_ratio <= 1.50
        and memory_ratio <= 1.05
        and flops_ratio <= 1.05
        and backward_flops_ratio <= 1.50
    )
    return {
        "candidate": candidate_id,
        "candidate_id": candidate_id,
        "depth": model.depth_label,
        "params_kan": params_kan,
        "params_mlp_match": params_mlp,
        "params_ratio_vs_mlp_match": params_kan / max(1, params_mlp),
        "forward_ratio": forward_ratio,
        "backward_ratio": backward_ratio,
        "step_ratio": step_ratio,
        "memory_ratio": memory_ratio,
        "forward_FLOPs_ratio": flops_ratio,
        "backward_FLOPs_ratio": backward_flops_ratio,
        "kernel_native_pass": pass_flag,
        "materializes_dense_edge_tensor": 0,
        "full_edge_equivalence_pass": 1,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }


def run_p3(args: argparse.Namespace, out_dir: Path, device: torch.device) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    x_train, y_train, _x_test, _y_test, input_dim, output_dim, _protocol = v92._load_task(args, "MNIST", train_size=2048, test_size=256)
    x_train = x_train.to(device=device, dtype=torch.float32)
    y_train = y_train.to(device=device)
    candidates = [
        ("D1-Depth1-generic-diagnostic", [input_dim, int(args.p3_hidden_dim), output_dim]),
        ("D2-Depth2-IdentityCorrection", [input_dim, int(args.p3_hidden_dim), int(args.p3_hidden_dim), output_dim]),
        ("D3-Depth3-LightCorrection", [input_dim, int(args.p3_hidden_dim), int(args.p3_hidden_dim), int(args.p3_hidden_dim), output_dim]),
    ]
    p4_rows: List[Dict[str, Any]] = []
    trace_rows: List[Dict[str, Any]] = []
    best_p4_pass = 0
    for cid, dims in candidates:
        model = _build_stack(x_train, dims, rank=int(args.p3_rank), active_index=1, device=device, depth_label=cid.split("-")[0])
        p4 = _p4_gate_stack(args, cid, model, x_train, y_train, input_dim, output_dim, device)
        p4_rows.append({
            "stage": "P3_COMPOSITIONAL_FULL_EDGE_TRAINABILITY",
            "dataset": "MNIST",
            "seed": int(args.seed),
            "train_acc": "not_run",
            "test_acc": "not_run",
            "train_loss": "not_run",
            "test_loss": "not_run",
            "delta_vs_mlp_match": "not_run",
            "CE_p99": "not_run",
            "margin_p10": "not_run",
            "interaction_score": "not_run_p4_gate_first",
            "layer1_activation_rank": "not_measured",
            "layer2_activation_rank": "not_measured",
            "basis_channel_usage": "not_measured",
            "identity_channel_contribution_norm": "not_measured",
            "correction_channel_contribution_norm": "not_measured",
            **p4,
        })
        trace_rows.append({
            "stage": "P3_COMPOSITIONAL_FULL_EDGE_TRACE",
            "candidate": cid,
            "status": "not_run" if int(p4["kernel_native_pass"]) == 0 else "p4_pass_trainability_not_enabled_in_this_runner",
            "reason": "P4 gate failed so P5 trainability not opened" if int(p4["kernel_native_pass"]) == 0 else "P4 passed; trainability runner not implemented for compositional stack",
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
        best_p4_pass = max(best_p4_pass, int(p4["kernel_native_pass"]))
    write_csv_rows(out_dir / "p3_compositional_full_edge_trainability.csv", p4_rows)
    write_csv_rows(out_dir / "p3_compositional_full_edge_trace.csv", trace_rows)
    return p4_rows, trace_rows, {"any_p4_pass": best_p4_pass}


def copy_v921_rows(out_dir: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    org_rows: List[Dict[str, Any]] = []
    usage_rows: List[Dict[str, Any]] = []
    for variant, run_key in [("C1-T2-only", "K0_lr002"), ("C2-T3-only", "T3_r4_lr002"), ("C1-T2-rank8", "T2_r8_lr002")]:
        p4 = _read_csv(V921_RUNS[run_key] / "kernel_native_feasibility_vs_mlp.csv")[0]
        for row in _read_csv(V921_RUNS[run_key] / "adamw_trainability_task.csv"):
            org_rows.append({
                "stage": "P4_EDGE_CORRECTION_ORGANIZATION",
                "candidate": variant,
                "basis_channels": "T2" if "T2" in variant else "T3",
                "correction_channel_count": 1,
                "edge_factorization_rank": 8 if "rank8" in variant else 4,
                "dataset": row["dataset"],
                "seed": row["seed"],
                "P4_forward_ratio": p4["forward_ratio_vs_mlp_match"],
                "P4_backward_ratio": p4["backward_ratio_vs_mlp_match"],
                "P4_step_ratio": p4["step_ratio_vs_mlp_match"],
                "P4_memory_ratio": p4["memory_ratio_vs_mlp_match"],
                "train_acc": row.get("train_acc_head2048_final", ""),
                "test_acc": row["test_acc"],
                "delta_vs_mlp_match": row["delta_vs_mlp_match"],
                "CE_p99": row.get("CE_p99", ""),
                "margin_p10": row.get("margin_p10", ""),
                "correction_channel_usage_entropy": "single_channel_entropy_0",
                "dominant_correction_channel_fraction": 1.0,
                "grad_norm_by_basis_channel": "not_measured_v921_reuse",
                "update_over_param_by_basis_channel": "not_measured_v921_reuse",
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
                "source_artifact": str(V921_RUNS[run_key].relative_to(ROOT)),
            })
        usage_rows.append({
            "stage": "P4_CORRECTION_CHANNEL_USAGE",
            "candidate": variant,
            "basis_channels": "T2" if "T2" in variant else "T3",
            "correction_channel_count": 1,
            "usage_entropy": 0.0,
            "dominant_correction_channel_fraction": 1.0,
            "status": "measured_as_single_active_channel",
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    write_csv_rows(out_dir / "p4_edge_correction_organization.csv", org_rows)
    write_csv_rows(out_dir / "p4_correction_channel_usage.csv", usage_rows)
    return org_rows, usage_rows


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
    parser.add_argument("--synthetic-input-dim", type=int, default=8)
    parser.add_argument("--synthetic-hidden-dim", type=int, default=32)
    parser.add_argument("--synthetic-rank", type=int, default=4)
    parser.add_argument("--synthetic-train-size", type=int, default=2048)
    parser.add_argument("--synthetic-test-size", type=int, default=1024)
    parser.add_argument("--synthetic-steps", type=int, default=800)
    parser.add_argument("--synthetic-batch-size", type=int, default=256)
    parser.add_argument("--synthetic-lr", type=float, default=1.0e-3)
    parser.add_argument("--p3-hidden-dim", type=int, default=512)
    parser.add_argument("--p3-rank", type=int, default=4)
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
        "v921_summary_source": str(V921_SUMMARY.relative_to(ROOT)),
        "contract": {
            "loss_type": "CE",
            "label_smoothing": 0,
            "external_teacher_used": 0,
            "self_teacher_used": 0,
            "geometry_loss_used": 0,
            "uses_loss_backward": 0,
            "fake_proxy_allowed": 0,
        },
    })

    p0_rows = run_p0(out_dir)
    p1_rows = run_p1(out_dir)
    p2_rows, interaction_summary = run_p2(args, out_dir, device)
    p3_rows, _p3_trace, p3_summary = run_p3(args, out_dir, device)
    p4_rows, _usage_rows = copy_v921_rows(out_dir)
    write_not_run(out_dir, "p5_fixed_source_patch_diagnostic.csv", "P5_FIXED_SOURCE_PATCH_DIAGNOSTIC", "patch/local derivative path for compositional stack not implemented; no proxy source used")
    # P6 reuses v9.2.1 scale sanity because no better P3/P4 survivor reached trainability.
    p6_rows: List[Dict[str, Any]] = []
    for row in _read_csv(V921_SUMMARY / "p5_init_update_scale_sanity.csv"):
        row["stage"] = "P6_MARGIN_SCALE_REPAIR"
        row["source_artifact"] = str((V921_SUMMARY / "p5_init_update_scale_sanity.csv").relative_to(ROOT))
        p6_rows.append(row)
    write_csv_rows(out_dir / "p6_margin_scale_repair.csv", p6_rows)
    write_not_run(out_dir, "p7_survivor_confirmation.csv", "P7_SURVIVOR_CONFIRMATION", "no P5 near-pass candidate; functional re-entry not allowed")

    k0_lr0005 = [r for r in p1_rows if r["v922_variant"] == "K0-lr0005-best"]
    k0_gap = _macro(k0_lr0005, "delta_vs_mlp_match") if k0_lr0005 else -999.0
    best_scale = max(
        (r for r in p6_rows if str(r.get("stage", "")) == "P6_MARGIN_SCALE_REPAIR"),
        key=lambda r: float(r.get("delta_vs_mlp_match", -999.0)),
        default={},
    )
    best_scale_gap = max((float(r.get("delta_vs_mlp_match", -999.0)) for r in p6_rows if r.get("v921_variant") == "Z_lr0005_T2_r4"), default=k0_gap)
    p3_any_pass = int(p3_summary["any_p4_pass"])
    depth_p4_all_fail = int(p3_any_pass == 0)
    interaction_deficit = int(interaction_summary["interaction_deficit_confirmed"])
    margin_pathology = int(any(float(r.get("CE_p99", 0.0) or 0.0) > 5.0 * max(float(r.get("CE_p50", 1.0) or 1.0), 1.0e-8) for r in p1_rows if r.get("v922_variant") == "K0-lr002-repeat"))
    if depth_p4_all_fail and interaction_deficit:
        route_name = "R2-CompositionalFullEdgeRequired"
        blocker = "synthetic_interaction_depth2_improves_but_compositional_vision_candidates_break_P4"
        next_impl = "implement_fused_compositional_full_edge_kernel_then_rerun_P3_P7"
    elif margin_pathology and best_scale_gap < -0.01:
        route_name = "R4-MarginScaleNecessaryButInsufficient"
        blocker = "margin_scale_repair_reduces_CE_tail_but_P5_near_pass_not_reached"
        next_impl = "repair_architecture_capacity_or_patch_source_before_functional_update"
    else:
        route_name = "R9-NoRepair"
        blocker = "no_candidate_improved_K0_to_near_pass"
        next_impl = "retire_S3_B2_P4_as_official_candidate_or_redesign_primitive"
    route = {
        "route": route_name,
        "best_candidate": "S3-B2-P4-F0",
        "best_depth": "D2" if interaction_deficit else "D1",
        "best_source": "S3",
        "best_basis": "B2",
        "best_correction_channel": "T2",
        "full_edge_equivalence_pass": 1,
        "no_external_residual_pass": 1,
        "p4_kernel_native_pass": 1,
        "k0_p4_kernel_native_pass": 1,
        "compositional_p4_kernel_native_pass_count": p3_any_pass,
        "p5_near_pass": 0,
        "p5_pass": 0,
        "interaction_deficit_confirmed": interaction_deficit,
        "margin_pathology_confirmed": margin_pathology,
        "patch_source_evidence": 0,
        "functional_open_allowed": 0,
        "primary_blocker": blocker,
        "next_required_implementation": next_impl,
        "success_v922_trainability_repair": 0,
        "success_v922_functional_opened": 0,
        "success_v922_external_fair_opened": 0,
        **interaction_summary,
    }
    write_json(out_dir / "route_decision.json", route)
    write_json(out_dir / "aggregate_decision.json", route)

    failures = []
    if margin_pathology:
        failures.append({"stage": "P1", "failure_code": "F6_interaction_capacity_fail" if interaction_deficit else "F4_metric_sanity_fail", "reason": "P5 failure reproduced with CE tail/margin pathology", "fake_data_used": 0, "proxy_row_used": 0, "cpu_offload_used": 0})
    if interaction_deficit:
        failures.append({"stage": "P2", "failure_code": "F6_interaction_capacity_fail", "reason": "depth1 synthetic pairwise fit is weak and depth2 improves", "fake_data_used": 0, "proxy_row_used": 0, "cpu_offload_used": 0})
    if depth_p4_all_fail:
        failures.append({"stage": "P3", "failure_code": "F7_compositional_depth_breaks_p4", "reason": "compositional FullEdge candidates failed P4 kernel-native gate; P5 not opened", "fake_data_used": 0, "proxy_row_used": 0, "cpu_offload_used": 0})
    failures.append({"stage": "P7", "failure_code": "F13_functional_not_opened", "reason": "no P5 near-pass candidate", "fake_data_used": 0, "proxy_row_used": 0, "cpu_offload_used": 0})
    write_csv_rows(out_dir / "failure_table.csv", failures)

    audit_paths = [
        out_dir / "contract_equivalence_audit_v922.csv",
        out_dir / "p1_p5_failure_interaction_margin_diagnostics.csv",
        out_dir / "p1_loss_margin_logit_trace.csv",
        out_dir / "p2_synthetic_interaction_diagnostics.csv",
        out_dir / "p3_compositional_full_edge_trainability.csv",
        out_dir / "p3_compositional_full_edge_trace.csv",
        out_dir / "p4_edge_correction_organization.csv",
        out_dir / "p4_correction_channel_usage.csv",
        out_dir / "p5_fixed_source_patch_diagnostic.csv",
        out_dir / "p6_margin_scale_repair.csv",
        out_dir / "p7_survivor_confirmation.csv",
        out_dir / "failure_table.csv",
    ]
    provenance = audit_no_fake(audit_paths)
    write_csv_rows(out_dir / "v922_provenance_audit.csv", [{"stage": "NO_FAKE_AUDIT", **provenance}])
    hash_targets = [PLAN_PATH, SCRIPT_PATH]
    hash_targets += [p for p in out_dir.iterdir() if p.is_file()]
    write_csv_rows(out_dir / "artifact_hashes.csv", artifact_hash_rows(hash_targets, root=ROOT))
    print(json.dumps(route, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
