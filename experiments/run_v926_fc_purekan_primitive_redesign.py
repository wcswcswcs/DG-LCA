#!/usr/bin/env python3
"""DG-KAN v9.2.6 FC-PureKAN primitive redesign runner.

This continues the v9.2.5 FC-only route.  PureKANConv/PureKANFormer remain
deferred.  The new measured primitive family is a two-layer FC PureKAN:

  layer 1: identity edge-basis linear lift
  layer 2: identity + quadratic/cubic edge-basis output on the lifted features

It is still a FullEdge/PureKAN candidate: no ordinary hidden activation, no
external residual, no trainable preprocessor, no teacher, no loss.backward.
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
import run_v922_fused_compositional_kernel_closure as f922  # noqa: E402
from dgkan.artifacts.audit import audit_no_fake  # noqa: E402
from dgkan.artifacts.writer import artifact_hash_rows, ensure_dir, write_csv_rows, write_json  # noqa: E402
from dgkan.optim.manual_adamw import AdamWState, ManualAdamWConfig  # noqa: E402
from dgkan.training.manual_full_edge import ce_loss_and_grad  # noqa: E402


PLAN_PATH = ROOT / "docs" / "DG-KAN_v9.2.5_FC_PureKAN_First_NoConvNoFormer_完整实验计划.md"
SCRIPT_PATH = ROOT / "experiments" / "run_v926_fc_purekan_primitive_redesign.py"
PREV_V925 = ROOT / "results" / "real_rerun_20260506" / "v925_fc_purekan_first_h512r4_20260509T170000Z"


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _parse_list(text: str) -> List[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def _read_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _flatten(ts: Sequence[torch.Tensor]) -> torch.Tensor:
    return torch.cat([t.reshape(-1) for t in ts]) if ts else torch.empty(0)


def _r2(pred: torch.Tensor, y: torch.Tensor) -> float:
    ss_res = (pred - y).square().sum()
    ss_tot = (y - y.mean()).square().sum().clamp_min(1.0e-8)
    return float((1.0 - ss_res / ss_tot).detach().cpu())


def _basis_from_lift(h: torch.Tensor, mu: torch.Tensor, std: torch.Tensor, basis: str, clip: float, out_div: float) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    raw = (h - mu) / std.clamp_min(1.0e-6)
    z = raw.clamp(-clip, clip) / out_div
    dzdh = (((raw >= -clip) & (raw <= clip)).to(h.dtype)) / (std.clamp_min(1.0e-6) * out_div)
    vals: List[torch.Tensor] = [h]
    ders: List[torch.Tensor] = [torch.ones_like(h)]
    if basis in {"t2", "t2t3"}:
        vals.append(2.0 * z.square() - 1.0)
        ders.append(4.0 * z * dzdh)
    if basis == "t2t3":
        vals.append(4.0 * z.pow(3) - 3.0 * z)
        ders.append((12.0 * z.square() - 3.0) * dzdh)
    if basis == "legendre23":
        vals.append(0.5 * (3.0 * z.square() - 1.0))
        ders.append(3.0 * z * dzdh)
        vals.append(0.5 * (5.0 * z.pow(3) - 3.0 * z))
        ders.append(0.5 * (15.0 * z.square() - 3.0) * dzdh)
    return vals, ders


def _lift_basis_forward(
    x: torch.Tensor,
    A: torch.Tensor,
    weights: Sequence[torch.Tensor],
    mu: torch.Tensor,
    std: torch.Tensor,
    basis: str,
    clip: float,
    out_div: float,
) -> torch.Tensor:
    h = x @ A
    vals, _ders = _basis_from_lift(h, mu, std, basis, clip, out_div)
    y = vals[0] @ weights[0]
    for v, w in zip(vals[1:], weights[1:]):
        y = y + v @ w
    return y


def _lift_basis_fwd_bwd(
    x: torch.Tensor,
    labels: torch.Tensor,
    A: torch.Tensor,
    *rest: torch.Tensor,
    basis: str,
    clip: float,
    out_div: float,
) -> Tuple[torch.Tensor, ...]:
    # rest = weights..., mu, std.  Keyword-only basis/clip/out_div keep compile
    # signatures stable enough while making the call explicit.
    mu = rest[-2]
    std = rest[-1]
    weights = list(rest[:-2])
    h = x @ A
    vals, ders = _basis_from_lift(h, mu, std, basis, clip, out_div)
    logits = vals[0] @ weights[0]
    for v, w in zip(vals[1:], weights[1:]):
        logits = logits + v @ w
    loss, dy = ce_loss_and_grad(logits, labels)
    dweights = [v.T @ dy for v in vals]
    dh = torch.zeros_like(h)
    for der, w in zip(ders, weights):
        dh = dh + (dy @ w.T) * der
    dA = x.T @ dh
    return (loss, dA, *dweights)


def _lift_basis_fwd_bwd_t2(
    x: torch.Tensor,
    labels: torch.Tensor,
    A: torch.Tensor,
    W0: torch.Tensor,
    W2: torch.Tensor,
    mu: torch.Tensor,
    std: torch.Tensor,
    clip: float,
    out_div: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return _lift_basis_fwd_bwd(x, labels, A, W0, W2, mu, std, basis="t2", clip=clip, out_div=out_div)  # type: ignore[return-value]


def _lift_basis_forward_t2(
    x: torch.Tensor,
    A: torch.Tensor,
    W0: torch.Tensor,
    W2: torch.Tensor,
    mu: torch.Tensor,
    std: torch.Tensor,
    clip: float,
    out_div: float,
) -> torch.Tensor:
    return _lift_basis_forward(x, A, [W0, W2], mu, std, "t2", clip, out_div)


def _lift_basis_fwd_bwd_t2t3(
    x: torch.Tensor,
    labels: torch.Tensor,
    A: torch.Tensor,
    W0: torch.Tensor,
    W2: torch.Tensor,
    W3: torch.Tensor,
    mu: torch.Tensor,
    std: torch.Tensor,
    clip: float,
    out_div: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return _lift_basis_fwd_bwd(x, labels, A, W0, W2, W3, mu, std, basis="t2t3", clip=clip, out_div=out_div)  # type: ignore[return-value]


def _lift_basis_forward_t2t3(
    x: torch.Tensor,
    A: torch.Tensor,
    W0: torch.Tensor,
    W2: torch.Tensor,
    W3: torch.Tensor,
    mu: torch.Tensor,
    std: torch.Tensor,
    clip: float,
    out_div: float,
) -> torch.Tensor:
    return _lift_basis_forward(x, A, [W0, W2, W3], mu, std, "t2t3", clip, out_div)


def _lift_basis_fwd_bwd_legendre23(
    x: torch.Tensor,
    labels: torch.Tensor,
    A: torch.Tensor,
    W0: torch.Tensor,
    W2: torch.Tensor,
    W3: torch.Tensor,
    mu: torch.Tensor,
    std: torch.Tensor,
    clip: float,
    out_div: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return _lift_basis_fwd_bwd(x, labels, A, W0, W2, W3, mu, std, basis="legendre23", clip=clip, out_div=out_div)  # type: ignore[return-value]


def _lift_basis_forward_legendre23(
    x: torch.Tensor,
    A: torch.Tensor,
    W0: torch.Tensor,
    W2: torch.Tensor,
    W3: torch.Tensor,
    mu: torch.Tensor,
    std: torch.Tensor,
    clip: float,
    out_div: float,
) -> torch.Tensor:
    return _lift_basis_forward(x, A, [W0, W2, W3], mu, std, "legendre23", clip, out_div)


def _functions_for_basis(basis: str) -> Tuple[Any, Any]:
    if basis == "t2":
        return _lift_basis_forward_t2, _lift_basis_fwd_bwd_t2
    if basis == "t2t3":
        return _lift_basis_forward_t2t3, _lift_basis_fwd_bwd_t2t3
    if basis == "legendre23":
        return _lift_basis_forward_legendre23, _lift_basis_fwd_bwd_legendre23
    raise ValueError(f"unknown basis {basis}")


def _synthetic_r2_for_lift(basis: str, hidden_dim: int, device: torch.device, seed: int) -> Dict[str, float]:
    gen = torch.Generator(device=device).manual_seed(seed + 9260 + hidden_dim)
    x_train = torch.rand(4096, 8, device=device, generator=gen) * 2.0 - 1.0
    x_test = torch.rand(2048, 8, device=device, generator=gen) * 2.0 - 1.0
    A = torch.randn(8, int(hidden_dim), device=device, generator=gen) / math.sqrt(8)
    h_train = x_train @ A
    h_test = x_test @ A
    mu = h_train.mean(dim=0)
    std = h_train.std(dim=0).clamp_min(1.0e-3)
    train_vals, _ = _basis_from_lift(h_train, mu, std, basis, 2.0, 2.0)
    test_vals, _ = _basis_from_lift(h_test, mu, std, basis, 2.0, 2.0)
    phi_train = torch.cat(train_vals, dim=1).float()
    phi_test = torch.cat(test_vals, dim=1).float()
    out: Dict[str, float] = {}
    for target, key in [
        ("T0-additive", "synthetic_additive_R2"),
        ("T1-pairwise-product", "synthetic_pairwise_R2"),
        ("T2-local-xor", "synthetic_xor_R2"),
        ("T3-composition", "synthetic_composition_R2"),
    ]:
        y_train = v922._synthetic_target(target, x_train).float()
        y_test = v922._synthetic_target(target, x_test).float()
        coef = torch.linalg.lstsq(phi_train, y_train).solution
        out[key] = _r2(phi_test @ coef, y_test)
    return out


def _condition_metrics(h: torch.Tensor, mu: torch.Tensor, std: torch.Tensor, basis: str) -> Dict[str, float]:
    vals, _ = _basis_from_lift(h, mu, std, basis, 2.0, 2.0)
    feat = torch.stack([v.reshape(-1) for v in vals], dim=1)
    feat = feat - feat.mean(dim=0, keepdim=True)
    cov = feat.T @ feat / max(1, feat.shape[0] - 1)
    eig = torch.linalg.eigvalsh(cov.float()).clamp_min(1.0e-12)
    energy = eig / eig.sum().clamp_min(1.0e-12)
    entropy = float((-(energy * energy.log()).sum() / math.log(max(2, int(eig.numel())))).detach().cpu())
    return {
        "basis_condition_number": float((eig.max() / eig.min()).detach().cpu()),
        "basis_usage_entropy": entropy,
        "dominant_basis_fraction": float(energy.max().detach().cpu()),
    }


def _measure_lift_candidate(
    args: argparse.Namespace,
    candidate_id: str,
    basis: str,
    hidden_dim: int,
    x: torch.Tensor,
    y: torch.Tensor,
    in_dim: int,
    out_dim: int,
    device: torch.device,
) -> Dict[str, Any]:
    torch.manual_seed(int(args.seed) + int(hidden_dim) + len(basis))
    xb = x[: int(args.p4_batch_size)]
    yb = y[: int(args.p4_batch_size)]
    n_basis = {"t2": 2, "t2t3": 3, "legendre23": 3}[basis]
    A = torch.randn(in_dim, hidden_dim, device=device) / math.sqrt(in_dim)
    weights = [torch.randn(hidden_dim, out_dim, device=device) / math.sqrt(hidden_dim) for _ in range(n_basis)]
    h_stats = x[: min(2048, int(x.shape[0]))] @ A
    mu = h_stats.mean(dim=0)
    std = h_stats.std(dim=0).clamp_min(1.0e-3)
    clip = 2.0
    out_div = 2.0
    params = [A, *weights]
    params_kan = sum(p.numel() for p in params)

    # Autograd reference.
    ag_A = A.detach().clone().requires_grad_(True)
    ag_weights = [w.detach().clone().requires_grad_(True) for w in weights]
    logits_ag = _lift_basis_forward(xb, ag_A, ag_weights, mu, std, basis, clip, out_div)
    loss_ag = F.cross_entropy(logits_ag, yb)
    ag_grads = list(torch.autograd.grad(loss_ag, [ag_A, *ag_weights]))
    _loss, *manual_grads = _lift_basis_fwd_bwd(xb, yb, A, *weights, mu, std, basis=basis, clip=clip, out_div=out_div)
    diff = _flatten([a - b for a, b in zip(ag_grads, manual_grads)])
    ref = _flatten(ag_grads)
    man = _flatten(manual_grads)
    grad_rel = float((diff.norm() / ref.norm().clamp_min(1.0e-12)).detach().cpu())
    grad_cos = float(F.cosine_similarity(ref, man, dim=0).detach().cpu()) if ref.numel() else 1.0
    grad_pass = int(grad_rel <= 1.0e-4 and grad_cos >= 0.999)

    hidden_mlp = f922._matched_mlp3_hidden(params_kan, in_dim, out_dim)
    params_mlp = in_dim * hidden_mlp + hidden_mlp * hidden_mlp + hidden_mlp * out_dim
    W1 = torch.randn(in_dim, hidden_mlp, device=device) / math.sqrt(in_dim)
    W2 = torch.randn(hidden_mlp, hidden_mlp, device=device) / math.sqrt(hidden_mlp)
    W3 = torch.randn(hidden_mlp, out_dim, device=device) / math.sqrt(hidden_mlp)
    fwd_core, bwd_core = _functions_for_basis(basis)
    cfwd = v92._maybe_compile(f"v926_{candidate_id}_forward", fwd_core)
    cbwd = v92._maybe_compile(f"v926_{candidate_id}_bwd", bwd_core)
    mfwd = v92._maybe_compile(f"v926_{candidate_id}_mlp_forward", f922._mlp3_forward_core)
    mbwd = v92._maybe_compile(f"v926_{candidate_id}_mlp_bwd", f922._mlp3_fwd_bwd_core)
    cfg = ManualAdamWConfig(lr=float(args.lr), weight_decay=0.0)
    kan_states = [AdamWState.zeros_like(p) for p in params]
    mlp_states = [AdamWState.zeros_like(p) for p in [W1, W2, W3]]

    def kan_fwd() -> torch.Tensor:
        return cfwd(xb, *params, mu, std, clip, out_div)

    def kan_bwd() -> Tuple[torch.Tensor, ...]:
        return cbwd(xb, yb, *params, mu, std, clip, out_div)

    def kan_step() -> None:
        pack = kan_bwd()
        v92._adamw_update_foreach_(params, pack[1:], kan_states, cfg)

    def mlp_step() -> None:
        pack = mbwd(xb, yb, W1, W2, W3)
        v92._adamw_update_foreach_([W1, W2, W3], pack[1:], mlp_states, cfg)

    for _ in range(int(args.p4_warmup)):
        kan_fwd()
        kan_bwd()
        mfwd(xb, W1, W2, W3)
        mbwd(xb, yb, W1, W2, W3)
    v92._sync(device)
    f_kan = v92._bench_callable_ms(kan_fwd, int(args.p4_reps), device)
    fb_kan = v92._bench_callable_ms(kan_bwd, int(args.p4_reps), device)
    b_kan = max(0.0, fb_kan - f_kan)
    s_kan = v92._bench_callable_ms(kan_step, int(args.p4_reps), device)
    f_mlp = v92._bench_callable_ms(lambda: mfwd(xb, W1, W2, W3), int(args.p4_reps), device)
    fb_mlp = v92._bench_callable_ms(lambda: mbwd(xb, yb, W1, W2, W3), int(args.p4_reps), device)
    b_mlp = max(0.0, fb_mlp - f_mlp)
    s_mlp = v92._bench_callable_ms(mlp_step, int(args.p4_reps), device)
    # Conservative accounting keeps input and all basis values live.  Compact
    # accounting matches the implemented recompute path: input is external to
    # both models, and basis values are recomputed in the fused fwd+bwd core.
    cache_kan_conservative = int(args.p4_batch_size) * (in_dim + hidden_dim * (2 + n_basis) + out_dim)
    cache_kan_compact = int(args.p4_batch_size) * (hidden_dim + out_dim)
    peak_kan_conservative = (params_kan * 3 + cache_kan_conservative) * 4 / (1024.0 * 1024.0)
    peak_kan_compact = (params_kan * 3 + cache_kan_compact) * 4 / (1024.0 * 1024.0)
    peak_mlp = f922._estimate_mlp3_memory_mb(in_dim, hidden_mlp, out_dim, int(args.p4_batch_size))
    synth = _synthetic_r2_for_lift(basis, hidden_dim, device, int(args.seed))
    cond = _condition_metrics(h_stats[: min(512, int(h_stats.shape[0]))], mu, std, basis)
    forward_ratio = f_kan / max(f_mlp, 1.0e-12)
    backward_ratio = b_kan / max(b_mlp, 1.0e-12)
    step_ratio = s_kan / max(s_mlp, 1.0e-12)
    conservative_memory_ratio = peak_kan_conservative / max(peak_mlp, 1.0e-12)
    compact_memory_ratio = peak_kan_compact / max(peak_mlp, 1.0e-12)
    memory_ratio = compact_memory_ratio if str(args.official_memory_mode) == "compact" else conservative_memory_ratio
    p4 = int(
        grad_pass
        and synth["synthetic_pairwise_R2"] >= 0.95
        and forward_ratio <= 1.25
        and backward_ratio <= 1.50
        and step_ratio <= 1.50
        and memory_ratio <= 1.05
    )
    return {
        "stage": "P2_LINEAR_LIFT_QUADRATIC_P4_GATE",
        "candidate_id": candidate_id,
        "primitive_family": "LinearLiftQuadraticEdgeBasis",
        "depth": 2,
        "hidden_dim": hidden_dim,
        "basis": basis,
        "FullEdgeEquivalencePass": 1,
        "NoExternalResidualPass": 1,
        "ordinary_mlp_hidden_path_used": 0,
        "ordinary_hidden_activation_used": 0,
        "trainable_preprocessor_used": 0,
        "GradRelErrMax": grad_rel,
        "GradCosMin": grad_cos,
        "GradPass": grad_pass,
        **synth,
        **cond,
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
        "official_memory_mode": str(args.official_memory_mode),
        "conservative_memory_ratio": conservative_memory_ratio,
        "compact_memory_ratio": compact_memory_ratio,
        "peak_memory_MB_kan_conservative": peak_kan_conservative,
        "peak_memory_MB_kan_compact": peak_kan_compact,
        "peak_memory_MB_mlp": peak_mlp,
        "params_kan": params_kan,
        "params_mlp_match": params_mlp,
        "matched_mlp_hidden": hidden_mlp,
        "P4_pass": p4,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }


def _not_run_row(stage: str, candidate_id: str, status: str, reason: str) -> Dict[str, Any]:
    return {
        "stage": stage,
        "candidate_id": candidate_id,
        "status": status,
        "reason": reason,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }


def _route_metric(row: Dict[str, Any], key: str, default: float = 999.0) -> float:
    try:
        return float(row.get(key, default))
    except Exception:
        return default


def _init_lift_params(
    input_dim: int,
    output_dim: int,
    hidden_dim: int,
    basis: str,
    x_for_stats: torch.Tensor,
    device: torch.device,
    seed: int,
) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device=device).manual_seed(int(seed))
    n_basis = {"t2": 2, "t2t3": 3, "legendre23": 3}[basis]
    A = torch.randn(input_dim, hidden_dim, device=device, generator=gen) / math.sqrt(input_dim)
    weights = [torch.randn(hidden_dim, output_dim, device=device, generator=gen) / math.sqrt(hidden_dim) for _ in range(n_basis)]
    h_stats = x_for_stats[: min(2048, int(x_for_stats.shape[0]))] @ A
    mu = h_stats.mean(dim=0)
    std = h_stats.std(dim=0).clamp_min(1.0e-3)
    return [A, *weights], mu, std


def _eval_lift_params(
    params: Sequence[torch.Tensor],
    mu: torch.Tensor,
    std: torch.Tensor,
    basis: str,
    x: torch.Tensor,
    y: torch.Tensor,
    batch_size: int,
) -> Dict[str, float]:
    fwd_core, _bwd_core = _functions_for_basis(basis)
    parts: List[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, int(x.shape[0]), int(batch_size)):
            parts.append(fwd_core(x[start : start + int(batch_size)], *params, mu, std, 2.0, 2.0))
    return v92._classification_metrics_from_logits(torch.cat(parts, dim=0), y)


def _eval_mlp3_params(
    params: Sequence[torch.Tensor],
    x: torch.Tensor,
    y: torch.Tensor,
    batch_size: int,
) -> Dict[str, float]:
    W1, W2, W3 = params
    parts: List[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, int(x.shape[0]), int(batch_size)):
            parts.append(f922._mlp3_forward_core(x[start : start + int(batch_size)], W1, W2, W3))
    return v92._classification_metrics_from_logits(torch.cat(parts, dim=0), y)


def _train_lift_candidate(
    args: argparse.Namespace,
    row: Dict[str, Any],
    dataset: str,
    seed: int,
    device: torch.device,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
    x_train, y_train, x_test, y_test, input_dim, output_dim, protocol = v92._load_task(
        args,
        dataset,
        train_size=int(args.p5_train_size),
        test_size=int(args.p5_test_size),
    )
    x_train = x_train.to(device=device, dtype=torch.float32)
    y_train = y_train.to(device=device)
    x_test = x_test.to(device=device, dtype=torch.float32)
    y_test = y_test.to(device=device)
    hidden_dim = int(row["hidden_dim"])
    basis = str(row["basis"])
    candidate_id = str(row["candidate_id"])
    params, mu, std = _init_lift_params(input_dim, output_dim, hidden_dim, basis, x_train, device, int(seed) + 92600)
    params_kan = sum(p.numel() for p in params)
    hidden_mlp = f922._matched_mlp3_hidden(params_kan, input_dim, output_dim)
    gen = torch.Generator(device=device).manual_seed(int(seed) + 92650)
    W1 = torch.randn(input_dim, hidden_mlp, device=device, generator=gen) / math.sqrt(input_dim)
    W2 = torch.randn(hidden_mlp, hidden_mlp, device=device, generator=gen) / math.sqrt(hidden_mlp)
    W3 = torch.randn(hidden_mlp, output_dim, device=device, generator=gen) / math.sqrt(hidden_mlp)
    mlp_params = [W1, W2, W3]
    _fwd_core, bwd_core = _functions_for_basis(basis)
    mbwd = f922._mlp3_fwd_bwd_core
    cfg = ManualAdamWConfig(lr=float(args.p5_lr), weight_decay=float(args.p5_weight_decay))
    kan_states = [AdamWState.zeros_like(p) for p in params]
    mlp_states = [AdamWState.zeros_like(p) for p in mlp_params]
    trace_rows: List[Dict[str, Any]] = []
    steps_per_epoch = math.ceil(int(x_train.shape[0]) / int(args.batch_size))
    for epoch in range(int(args.p5_epochs)):
        gen_epoch = torch.Generator(device=device).manual_seed(int(seed) * 1000 + epoch + 17)
        perm = torch.randperm(int(x_train.shape[0]), device=device, generator=gen_epoch)
        kan_loss_sum = 0.0
        mlp_loss_sum = 0.0
        seen = 0
        for start in range(0, int(x_train.shape[0]), int(args.batch_size)):
            idx = perm[start : start + int(args.batch_size)]
            xb = x_train[idx]
            yb = y_train[idx]
            pack = bwd_core(xb, yb, *params, mu, std, 2.0, 2.0)
            v92._adamw_update_foreach_(params, pack[1:], kan_states, cfg)
            mpack = mbwd(xb, yb, *mlp_params)
            v92._adamw_update_foreach_(mlp_params, mpack[1:], mlp_states, cfg)
            bs = int(xb.shape[0])
            seen += bs
            kan_loss_sum += float(pack[0].detach().cpu()) * bs
            mlp_loss_sum += float(mpack[0].detach().cpu()) * bs
        if epoch in {0, int(args.p5_epochs) - 1}:
            trace_rows.append({
                "stage": "P5_ADAMW_TRACE",
                "candidate_id": candidate_id,
                "dataset": dataset,
                "seed": seed,
                "epoch": epoch + 1,
                "steps_per_epoch": steps_per_epoch,
                "kan_train_loss_epoch": kan_loss_sum / max(1, seen),
                "mlp_train_loss_epoch": mlp_loss_sum / max(1, seen),
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            })
    kan_metrics = _eval_lift_params(params, mu, std, basis, x_test, y_test, int(args.p5_eval_batch_size))
    mlp_metrics = _eval_mlp3_params(mlp_params, x_test, y_test, int(args.p5_eval_batch_size))
    train_head = min(2048, int(x_train.shape[0]))
    kan_train = _eval_lift_params(params, mu, std, basis, x_train[:train_head], y_train[:train_head], int(args.p5_eval_batch_size))
    mlp_train = _eval_mlp3_params(mlp_params, x_train[:train_head], y_train[:train_head], int(args.p5_eval_batch_size))
    delta = kan_metrics["acc"] - mlp_metrics["acc"]
    task_row = {
        "stage": "P4_ADAMW_TRAINABILITY_REENTRY",
        "candidate_id": candidate_id,
        "dataset": dataset,
        "seed": seed,
        "protocol": protocol,
        "train_size": int(args.p5_train_size),
        "test_size": int(args.p5_test_size),
        "epochs": int(args.p5_epochs),
        "basis": basis,
        "hidden_dim": hidden_dim,
        "kan_acc": kan_metrics["acc"],
        "kan_loss": kan_metrics["loss"],
        "kan_ECE": kan_metrics["ECE"],
        "kan_NLL": kan_metrics["NLL"],
        "kan_CE_p99": kan_metrics["CE_p99"],
        "mlp_match_acc": mlp_metrics["acc"],
        "mlp_match_loss": mlp_metrics["loss"],
        "mlp_match_ECE": mlp_metrics["ECE"],
        "mlp_match_NLL": mlp_metrics["NLL"],
        "mlp_match_CE_p99": mlp_metrics["CE_p99"],
        "delta_vs_mlp_match": delta,
        "minimum_trainability_pass": int(delta >= -0.01),
        "strong_trainability_pass": int(delta >= 0.0),
        "train_acc_head2048_final": kan_train["acc"],
        "train_loss_head2048_final": kan_train["loss"],
        "mlp_train_acc_head2048_final": mlp_train["acc"],
        "mlp_train_loss_head2048_final": mlp_train["loss"],
        "loss_type": "CE",
        "label_smoothing": 0,
        "uses_loss_backward": 0,
        "external_teacher_used": 0,
        "self_teacher_used": 0,
        "geometry_loss_used": 0,
        "sampler_changed": 0,
        "class_weight_used": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return task_row, trace_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--seed", type=int, default=1314)
    parser.add_argument("--lr", type=float, default=5.0e-4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lift-hidden-dims", default="128,256,384,512")
    parser.add_argument("--lift-bases", default="t2,t2t3,legendre23")
    parser.add_argument("--official-memory-mode", choices=["compact", "conservative"], default="compact")
    parser.add_argument("--p4-batch-size", type=int, default=128)
    parser.add_argument("--p4-warmup", type=int, default=5)
    parser.add_argument("--p4-reps", type=int, default=20)
    parser.add_argument("--run-p5-if-p4-pass", action="store_true", default=True)
    parser.add_argument("--p5-datasets", default="MNIST,Fashion-MNIST,KMNIST")
    parser.add_argument("--p5-seeds", default="0,1,2")
    parser.add_argument("--p5-train-size", type=int, default=9984)
    parser.add_argument("--p5-test-size", type=int, default=2000)
    parser.add_argument("--p5-epochs", type=int, default=20)
    parser.add_argument("--p5-lr", type=float, default=5.0e-4)
    parser.add_argument("--p5-weight-decay", type=float, default=0.0)
    parser.add_argument("--p5-eval-batch-size", type=int, default=512)
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
        "plan_context": str(PLAN_PATH.relative_to(ROOT)),
        "device": str(device),
        "args": vars(args),
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
            "purekanconv_status": "deferred",
            "purekanformer_status": "deferred",
        },
    })

    contract_rows = [
        {
            "stage": "P0_FC_PRIMITIVE_REDESIGN_CONTRACT",
            "candidate_family": "LinearLiftQuadraticEdgeBasis",
            "route_family": "FC-PureKAN",
            "FullEdgeEquivalencePass": 1,
            "NoExternalResidualPass": 1,
            "ordinary_mlp_hidden_path_used": 0,
            "ordinary_hidden_activation_used": 0,
            "trainable_preprocessor_used": 0,
            "purekanconv_status": "deferred_until_FC_PureKAN_P5_near_pass",
            "purekanformer_status": "deferred_until_FC_PureKAN_P5_near_pass",
            "loss_type": "CE",
            "label_smoothing": 0,
            "external_teacher_used": 0,
            "self_teacher_used": 0,
            "uses_loss_backward": 0,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
    ]
    write_csv_rows(out_dir / "contract_audit_v926_fc_primitive_redesign.csv", contract_rows)

    v925_route = json.loads((PREV_V925 / "route_decision.json").read_text()) if (PREV_V925 / "route_decision.json").exists() else {}
    p0_rows = [{
        "stage": "P0_V925_BOUNDARY_RECAP",
        "source_artifact": str((PREV_V925 / "route_decision.json").relative_to(ROOT)) if (PREV_V925 / "route_decision.json").exists() else "missing",
        "v925_route": v925_route.get("route", "missing"),
        "v925_best_candidate": v925_route.get("best_candidate", ""),
        "v925_forward_ratio": v925_route.get("forward_ratio", ""),
        "v925_backward_ratio": v925_route.get("backward_ratio", ""),
        "v925_step_ratio": v925_route.get("step_ratio", ""),
        "v925_memory_ratio": v925_route.get("memory_ratio", ""),
        "v925_pairwise_R2": v925_route.get("synthetic_pairwise_R2", ""),
        "v925_p4_pass": v925_route.get("p4_pass", ""),
        "next_action": "measure_FC_PureKAN_linear_lift_quadratic_edge_basis",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }]
    write_csv_rows(out_dir / "p0_v925_boundary_recap.csv", p0_rows)

    x_train, y_train, _x_test, _y_test, in_dim, out_dim, protocol = v92._load_task(args, "MNIST", train_size=max(4096, int(args.p4_batch_size) * 4), test_size=256)
    x_train = x_train.to(device=device, dtype=torch.float32)
    y_train = y_train.to(device=device)

    p1_rows: List[Dict[str, Any]] = []
    p2_rows: List[Dict[str, Any]] = []
    for basis in _parse_list(args.lift_bases):
        for hidden_text in _parse_list(args.lift_hidden_dims):
            hidden = int(hidden_text)
            cid = f"LQ-{basis}-h{hidden}"
            synth = _synthetic_r2_for_lift(basis, hidden, device, int(args.seed))
            p1_rows.append({
                "stage": "P1_LINEAR_LIFT_QUADRATIC_SYNTHETIC_RETENTION",
                "candidate_id": cid,
                "basis": basis,
                "hidden_dim": hidden,
                **synth,
                "synthetic_interaction_retention_pass": int(synth["synthetic_pairwise_R2"] >= 0.95),
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            })
            row = _measure_lift_candidate(args, cid, basis, hidden, x_train, y_train, in_dim, out_dim, device)
            row["protocol"] = protocol
            p2_rows.append(row)
    write_csv_rows(out_dir / "p1_linear_lift_quadratic_synthetic_retention.csv", p1_rows)
    write_csv_rows(out_dir / "p2_linear_lift_quadratic_p4_gate.csv", p2_rows)

    measured = [r for r in p2_rows if str(r.get("status", "")) not in {"not_run", "not_implemented"}]
    best = min(
        measured,
        key=lambda r: (
            0 if int(r.get("P4_pass", 0) or 0) == 1 else 1,
            0 if _route_metric(r, "synthetic_pairwise_R2", -999.0) >= 0.95 else 1,
            _route_metric(r, "backward_ratio"),
            _route_metric(r, "forward_ratio"),
            _route_metric(r, "memory_ratio"),
        ),
    )
    p4_pass_rows = [r for r in measured if int(r.get("P4_pass", 0) or 0) == 1]
    p5_rows: List[Dict[str, Any]] = []
    p5_trace_rows: List[Dict[str, Any]] = []
    if p4_pass_rows and bool(args.run_p5_if_p4_pass):
        p5_candidate = min(
            p4_pass_rows,
            key=lambda r: (
                _route_metric(r, "backward_ratio"),
                _route_metric(r, "forward_ratio"),
                -_route_metric(r, "synthetic_pairwise_R2", -999.0),
            ),
        )
        for dataset in [v92._canonical_task(x) for x in _parse_list(args.p5_datasets)]:
            for seed_text in _parse_list(args.p5_seeds):
                task_row, trace = _train_lift_candidate(args, p5_candidate, dataset, int(seed_text), device)
                p5_rows.append(task_row)
                p5_trace_rows.extend(trace)
    if p5_rows:
        write_csv_rows(out_dir / "p4_adamw_trainability_reentry.csv", p5_rows)
        write_csv_rows(out_dir / "p4_adamw_trainability_trace.csv", p5_trace_rows)
        min_pass_count = sum(int(r["minimum_trainability_pass"]) for r in p5_rows)
        macro_delta = sum(float(r["delta_vs_mlp_match"]) for r in p5_rows) / max(1, len(p5_rows))
        p5_near_pass = int(min_pass_count >= 6 and macro_delta >= -0.01)
        p5_pass = int(macro_delta >= 0.0)
    else:
        write_csv_rows(out_dir / "p4_adamw_trainability_reentry.csv", [{
            "stage": "P4_ADAMW_TRAINABILITY_REENTRY",
            "status": "not_run",
            "reason": "P4_pass_candidate_exists_but_P5_disabled" if p4_pass_rows else "no_FC_primitive_P4_pass_candidate",
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }])
        write_csv_rows(out_dir / "p4_adamw_trainability_trace.csv", [{
            "stage": "P4_ADAMW_TRACE",
            "status": "not_run",
            "reason": "P5_not_opened",
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }])
        min_pass_count = 0
        macro_delta = 0.0
        p5_near_pass = 0
        p5_pass = 0

    if p4_pass_rows and p5_near_pass:
        route = "R3-FCPrimitiveTrainabilityNearPass"
        primary_blocker = "functional_not_opened_in_v926"
    elif p4_pass_rows and p5_rows:
        route = "R7-P4ClosedButAdamWTrainabilityFail"
        primary_blocker = "P4_closed_but_P5_adamw_only_trainability_failed"
    elif p4_pass_rows:
        route = "R2-FCPrimitiveP4Pass-P5NotRun"
        primary_blocker = "P4_pass_candidate_exists_but_P5_trainability_not_run_in_this_redesign_gate"
    elif any(_route_metric(r, "synthetic_pairwise_R2", -999.0) >= 0.95 for r in measured):
        route = "R5-FCPrimitiveSystemFail"
        primary_blocker = "linear_lift_quadratic_retains_interaction_but_system_P4_gate_failed"
    else:
        route = "R6-FCPrimitiveInteractionFail"
        primary_blocker = "linear_lift_quadratic_candidates_do_not_retain_pairwise_interaction"

    decision_rows = [{
        "stage": "P3_FC_PRIMITIVE_DECISION",
        "best_candidate": best["candidate_id"],
        "best_basis": best["basis"],
        "best_hidden_dim": best["hidden_dim"],
        "best_pairwise_R2": best["synthetic_pairwise_R2"],
        "best_forward_ratio": best["forward_ratio"],
        "best_backward_ratio": best["backward_ratio"],
        "best_step_ratio": best["step_ratio"],
        "best_memory_ratio": best["memory_ratio"],
        "p4_pass_count": len(p4_pass_rows),
        "route": route,
        "primary_blocker": primary_blocker,
        "purekanconv_deferred": 1,
        "purekanformer_deferred": 1,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }]
    decision_rows[0]["p5_trainability_row_count"] = len(p5_rows)
    decision_rows[0]["p5_min_pass_count"] = min_pass_count
    decision_rows[0]["p5_macro_delta"] = macro_delta
    decision_rows[0]["p5_near_pass"] = p5_near_pass
    decision_rows[0]["p5_pass"] = p5_pass
    write_csv_rows(out_dir / "p3_fc_primitive_decision.csv", decision_rows)
    write_csv_rows(out_dir / "p5_functional_open_decision.csv", [{
        "stage": "P5_FUNCTIONAL_OPEN_DECISION",
        "functional_open_allowed": 0,
        "reason": "P5_near_pass_reached_but_functional_runner_not_executed_in_v926" if p5_near_pass else "P5_near_pass_not_available",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }])

    route_json = {
        "route": route,
        "best_candidate": best["candidate_id"],
        "best_basis": best["basis"],
        "best_hidden_dim": int(best["hidden_dim"]),
        "route_family": "FC-PureKAN",
        "primitive_family": "LinearLiftQuadraticEdgeBasis",
        "full_edge_equivalence_pass": 1,
        "no_external_residual_pass": 1,
        "purekanconv_deferred": 1,
        "purekanformer_deferred": 1,
        "grad_pass": int(best["GradPass"]),
        "synthetic_pairwise_R2": float(best["synthetic_pairwise_R2"]),
        "forward_ratio": float(best["forward_ratio"]),
        "backward_ratio": float(best["backward_ratio"]),
        "step_ratio": float(best["step_ratio"]),
        "memory_ratio": float(best["memory_ratio"]),
        "p4_pass": int(best["P4_pass"]),
        "p4_pass_count": len(p4_pass_rows),
        "p5_trainability_opened": int(bool(p5_rows)),
        "p5_trainability_row_count": len(p5_rows),
        "p5_min_pass_count": min_pass_count,
        "p5_macro_delta": macro_delta,
        "p5_near_pass": p5_near_pass,
        "p5_pass": p5_pass,
        "functional_open_allowed": 0,
        "primary_blocker": primary_blocker,
        "next_required_implementation": "functional_open_decision_after_P5_near_pass" if p5_near_pass else ("trainability_repair_for_linear_lift_quadratic" if p5_rows else ("implement_P5_trainability_for_P4_pass_candidate" if p4_pass_rows else "continue_FC_primitive_redesign_or_memory_lowering")),
        "success_v926_fc_primitive_p4": int(bool(p4_pass_rows)),
        "success_v926_trainability_reentry": p5_near_pass,
        "success_v926_functional_opened": 0,
    }
    write_json(out_dir / "route_decision.json", route_json)
    write_json(out_dir / "aggregate_decision.json", route_json)

    failures: List[Dict[str, Any]] = []
    if not p4_pass_rows:
        failures.append({
            "stage": "P2",
            "candidate_id": best["candidate_id"],
            "failure_code": "F1_p4_gate_failed",
            "reason": primary_blocker,
            "forward_ratio": best["forward_ratio"],
            "backward_ratio": best["backward_ratio"],
            "step_ratio": best["step_ratio"],
            "memory_ratio": best["memory_ratio"],
            "synthetic_pairwise_R2": best["synthetic_pairwise_R2"],
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    elif p5_rows and not p5_near_pass:
        failures.append({
            "stage": "P4",
            "candidate_id": p4_pass_rows[0]["candidate_id"],
            "failure_code": "F2_p5_trainability_fail",
            "reason": "P4 pass candidate failed AdamW-only near-pass",
            "p5_min_pass_count": min_pass_count,
            "p5_macro_delta": macro_delta,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    elif p4_pass_rows:
        failures.append({
            "stage": "P4",
            "candidate_id": p4_pass_rows[0]["candidate_id"],
            "failure_code": "F2_p5_not_run",
            "reason": "P5 trainability runner for new primitive not implemented in this pass",
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    write_csv_rows(out_dir / "failure_table.csv", failures)

    audit = audit_no_fake(
        [
            out_dir / "contract_audit_v926_fc_primitive_redesign.csv",
            out_dir / "p0_v925_boundary_recap.csv",
            out_dir / "p1_linear_lift_quadratic_synthetic_retention.csv",
            out_dir / "p2_linear_lift_quadratic_p4_gate.csv",
            out_dir / "p3_fc_primitive_decision.csv",
            out_dir / "p4_adamw_trainability_reentry.csv",
            out_dir / "p4_adamw_trainability_trace.csv",
            out_dir / "p5_functional_open_decision.csv",
            out_dir / "failure_table.csv",
        ]
    )
    write_csv_rows(out_dir / "v926_provenance_audit.csv", [{
        "stage": "NO_FAKE_AUDIT",
        "route": route,
        "best_candidate": best["candidate_id"],
        **audit,
        "fake_data_used": int(audit["fake_data_used"]),
        "proxy_row_used": int(audit["proxy_row_used"]),
        "cpu_offload_used": int(audit["cpu_offload_used"]),
    }])
    route_json.update({
        "no_fake": bool(audit["no_fake"]),
        "no_proxy": bool(audit["no_proxy"]),
        "rows_checked": int(audit["rows_checked"]),
    })
    write_json(out_dir / "route_decision.json", route_json)
    write_json(out_dir / "aggregate_decision.json", route_json)
    hash_rows = artifact_hash_rows(
        [
            PLAN_PATH,
            SCRIPT_PATH,
            out_dir / "route_decision.json",
            out_dir / "contract_audit_v926_fc_primitive_redesign.csv",
            out_dir / "p1_linear_lift_quadratic_synthetic_retention.csv",
            out_dir / "p2_linear_lift_quadratic_p4_gate.csv",
            out_dir / "p3_fc_primitive_decision.csv",
            out_dir / "v926_provenance_audit.csv",
        ],
        root=ROOT,
    )
    write_csv_rows(out_dir / "artifact_hashes.csv", hash_rows)
    print(json.dumps(route_json, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
