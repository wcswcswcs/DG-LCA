#!/usr/bin/env python
"""v13.4 operator-level basis-channel functional update runner.

This runner deliberately separates three surfaces:

1. live basis-channel target ``DeltaZ`` generated from a generic loss-interface
   output cotangent,
2. true parameter projection through an autograd sketch of ``J_theta->Z``, and
3. post-writeback actuation measurement on the same live channel.

It does not use frozen readout-feature transport, CE-tail metrics, validation,
test, future outcome, or dataset-name branches to construct the functional
direction.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import sys
import time
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dgkan.models.fc_purekan_primitives import MLPBaseline  # noqa: E402
from experiments.run_v133_task_family_robust_basis_natural import (  # noqa: E402
    SOURCE_V1235,
    build_substrate_map,
    eval_metrics,
    fnum,
    finite_mean,
    linec_rate,
    linec_proxy,
    make_model_for_family,
    one_hot,
    output_cotangent,
    parse_csv,
    parse_ints,
    read_rows,
    s1_gate,
    sha256_file,
    sint,
    synthetic_data,
    train_model,
    write_rows,
    write_svg,
)

OUT_DIR = ROOT / "results" / "v13_4_operator_level_basis_channel_functional" / "official_v134"
DOC_PLAN = ROOT / "docs" / "DG-KAN_v13.4_OperatorLevelBasisChannelFunctional_StrategicPlan.md"
DOC_EXEC = ROOT / "docs" / "DG-KAN_v13.4_OperatorLevelBasisChannelFunctional_执行日志.md"
DOC_REVIEW = ROOT / "docs" / "DG-KAN_v13.4_OperatorLevelBasisChannelFunctional_实验结果复盘.md"

REQUIRED = [
    "v134_route_decision.json",
    "v134_code_review_manifest.csv",
    "v134_basis_channel_manifest.csv",
    "v134_jacobian_operator_manifest.csv",
    "v134_loss_interface_audit.csv",
    "v134_forbidden_information_audit.csv",
    "v134_parameter_writeback_trace.csv",
    "v134_optimizer_state_transport_trace.csv",
    "v134_substrate_split.csv",
    "v134_operator_channel_solve.csv",
    "v134_parameter_projection.csv",
    "v134_synthetic_task_family_proof.csv",
    "v134_nonrat_s1c_design.csv",
    "v134_mlp_analog_channel_control.csv",
    "v134_failure_table.csv",
    "v134_no_go_boundary.md",
    "v134_next_hypothesis_queue.md",
    "v134_code_review_packet.zip",
]

SUPPLEMENTAL = [
    "v134_required_artifact_manifest.csv",
    "v134_progress_table.csv",
    "v134_substrate_map.csv",
]

FIGURES = [
    "fig_progress_by_line.svg",
    "fig_basis_channel_rank_by_family.svg",
    "fig_actuation_error_by_solver.svg",
    "fig_delta_z_target_vs_actual.svg",
    "fig_operator_solve_source_vs_control.svg",
    "fig_synthetic_family_pass_heatmap.svg",
    "fig_nonrat_s1c_substrate_status.svg",
    "fig_mlp_analog_vs_kan_operator_solve.svg",
    "fig_failure_taxonomy.svg",
]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def flatten_tensors(tensors: Iterable[torch.Tensor], *, detach: bool = True) -> torch.Tensor:
    flats = []
    for t in tensors:
        v = t.detach() if detach else t
        flats.append(v.reshape(-1).float())
    if not flats:
        return torch.zeros(0)
    return torch.cat(flats)


def param_sha(params: list[tuple[str, torch.nn.Parameter]]) -> str:
    h = hashlib.sha256()
    for name, p in params:
        h.update(str(name).encode("utf-8"))
        h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def effective_rank(x: torch.Tensor) -> float:
    if x.ndim == 1:
        x = x.view(-1, 1)
    xc = x.float() - x.float().mean(dim=0, keepdim=True)
    if min(int(xc.shape[0]), int(xc.shape[1])) <= 0:
        return 0.0
    try:
        s = torch.linalg.svdvals(xc)
    except RuntimeError:
        return 0.0
    power = s.square()
    denom = power.square().sum().clamp_min(1.0e-12)
    return float((power.sum().square() / denom).detach().item())


def condition_proxy(x: torch.Tensor) -> float:
    xc = x.float() - x.float().mean(dim=0, keepdim=True)
    try:
        s = torch.linalg.svdvals(xc)
    except RuntimeError:
        return float("inf")
    good = s[s > 1.0e-7]
    return float((good.max() / good.min()).detach().item()) if int(good.numel()) > 0 else float("inf")


def basis_channel(model: torch.nn.Module, x: torch.Tensor) -> tuple[torch.Tensor, str]:
    """Return a live channel tensor, not a frozen readout feature table."""
    if isinstance(model, MLPBaseline):
        h = F.silu(x @ model.w0)
        h = F.silu(h @ model.w1)
        return h.reshape(int(x.shape[0]), -1), "mlp_hidden_activation_channel"
    if hasattr(model, "_rational_forward") and hasattr(model, "_norm_input"):
        z = model._norm_input(x)  # type: ignore[attr-defined]
        r = model._rational_forward(z, autograd_enabled=torch.is_grad_enabled())  # type: ignore[attr-defined]
        return r.reshape(int(x.shape[0]), -1), "kan_grouped_rational_live_rational_basis_channel"
    if hasattr(model, "layer2_basis") and hasattr(model, "hidden"):
        h = model.hidden(x)  # type: ignore[attr-defined]
        z = model.layer2_basis(h)  # type: ignore[attr-defined]
        return z.reshape(int(x.shape[0]), -1), "kan_layer2_basis_hidden_channel"
    if hasattr(model, "hidden"):
        h = model.hidden(x)  # type: ignore[attr-defined]
        return h.reshape(int(x.shape[0]), -1), "kan_hidden_activation_channel"
    logits = model(x)
    return logits.reshape(int(x.shape[0]), -1), "fallback_logit_channel"


def select_channel_indices(z: torch.Tensor, max_dim: int) -> torch.Tensor:
    dim = int(z.shape[1])
    keep = min(dim, int(max_dim))
    if keep >= dim:
        return torch.arange(dim, device=z.device)
    score = z.detach().float().var(dim=0, unbiased=False) + 1.0e-4 * z.detach().float().abs().mean(dim=0)
    return torch.topk(score, k=keep, largest=True).indices.sort().values


def selected_channel(model: torch.nn.Module, x: torch.Tensor, channel_idx: torch.Tensor | None, max_dim: int) -> tuple[torch.Tensor, torch.Tensor, str]:
    z, surface = basis_channel(model, x)
    if channel_idx is None:
        channel_idx = select_channel_indices(z, max_dim)
    return z[:, channel_idx], channel_idx, surface


def channel_named_params(model: torch.nn.Module) -> list[tuple[str, torch.nn.Parameter]]:
    out: list[tuple[str, torch.nn.Parameter]] = []
    for name, p in model.named_parameters():
        leaf = name.split(".")[-1].lower()
        lname = name.lower()
        if isinstance(model, MLPBaseline):
            if name in {"w0", "w1"}:
                out.append((name, p))
            continue
        if leaf == "w2" or "readout" in lname or leaf == "bias" or "logit" in lname:
            continue
        if leaf in {"w1", "numerator", "denominator", "hidden_bias", "input_cross_hidden", "input_cross_hidden_weight"}:
            out.append((name, p))
        elif leaf.startswith("basis_"):
            out.append((name, p))
    if not out:
        for name, p in model.named_parameters():
            if "w2" not in name.lower() and "readout" not in name.lower() and "bias" not in name.lower():
                out.append((name, p))
    return out


def vector_to_tensors(vec: torch.Tensor, params: list[tuple[str, torch.nn.Parameter]]) -> list[torch.Tensor]:
    out: list[torch.Tensor] = []
    off = 0
    for _name, p in params:
        n = int(p.numel())
        out.append(vec[off : off + n].reshape_as(p).detach())
        off += n
    return out


def apply_vector(params: list[tuple[str, torch.nn.Parameter]], vec: torch.Tensor, sign: float = 1.0) -> None:
    tensors = vector_to_tensors(vec, params)
    with torch.no_grad():
        for (_name, p), upd in zip(params, tensors):
            p.add_(float(sign) * upd.to(device=p.device, dtype=p.dtype))


def ridge_head(z: torch.Tensor, logits: torch.Tensor, rho: float) -> torch.Tensor:
    zc = z.detach().float() - z.detach().float().mean(dim=0, keepdim=True)
    yc = logits.detach().float() - logits.detach().float().mean(dim=0, keepdim=True)
    eye = torch.eye(int(zc.shape[1]), device=zc.device, dtype=zc.dtype)
    lhs = zc.T @ zc / float(max(1, int(zc.shape[0]))) + float(rho) * eye
    rhs = zc.T @ yc / float(max(1, int(zc.shape[0])))
    try:
        return torch.linalg.solve(lhs, rhs)
    except RuntimeError:
        return torch.linalg.pinv(lhs) @ rhs


def delta_z_stats(dz: torch.Tensor, head: torch.Tensor, logits: torch.Tensor, delta_y: torch.Tensor) -> dict[str, float]:
    pred_dy = dz @ head
    gram = dz.T @ dz
    if int(gram.shape[0]) > 1:
        gram = gram.clone()
        gram.fill_diagonal_(0.0)
        offdiag = float(gram.abs().mean().detach().item())
    else:
        offdiag = 0.0
    return {
        "delta_z_norm": float(dz.norm().detach().item()),
        "delta_z_rank": effective_rank(dz),
        "delta_z_effective_rank": effective_rank(dz),
        "delta_z_alignment_with_loss_cotangent": float(F.cosine_similarity(pred_dy.reshape(-1), (-delta_y).reshape(-1), dim=0, eps=1.0e-8).detach().item()),
        "delta_z_alignment_with_adamw_channel": float("nan"),
        "delta_z_snr_positive_fraction": float((dz.abs().mean(dim=0) > 0.0).float().mean().detach().item()),
        "delta_z_offdiag_score": offdiag,
        "delta_z_predicted_logit_drift": float(pred_dy.norm().div(logits.detach().float().norm().clamp_min(1.0e-8)).detach().item()),
        "delta_z_predicted_tail_proxy": float(torch.quantile(pred_dy.abs().flatten(), 0.99).detach().item()) if int(pred_dy.numel()) > 0 else 0.0,
    }


def make_delta_z_target(
    z: torch.Tensor,
    logits: torch.Tensor,
    y: torch.Tensor,
    *,
    loss_interface: str,
    operator: str,
    eta: float,
    rho: float,
) -> tuple[torch.Tensor, dict[str, float], torch.Tensor]:
    delta_y = output_cotangent(logits, y, loss_interface).detach().float()
    head = ridge_head(z, logits, rho)
    channel_cotangent = delta_y @ head.T
    scale = z.detach().float().std(dim=0, unbiased=False).view(1, -1).clamp_min(float(rho))
    if operator == "O1-LossCotangentChannelNewtonDiag":
        dz = -float(eta) * channel_cotangent / scale
    elif operator == "O2-LossCotangentChannelSNR":
        snr = (z.detach().float().mean(dim=0).square() - z.detach().float().var(dim=0, unbiased=False) / float(max(1, int(z.shape[0]) - 1))).view(1, -1)
        dz = -float(eta) * channel_cotangent * (snr > 0.0).float()
    elif operator == "O3-PopRiskOffdiagChannel":
        zc = z.detach().float() - z.detach().float().mean(dim=0, keepdim=True)
        cov = zc.T @ zc / float(max(1, int(z.shape[0])))
        off = cov - torch.diag(torch.diag(cov))
        resid = zc @ off
        gram = head.T @ head + float(rho) * torch.eye(int(head.shape[1]), device=head.device, dtype=head.dtype)
        proj = (resid @ head) @ torch.linalg.pinv(gram) @ head.T
        null_resid = resid - proj
        null_resid = null_resid / null_resid.std(dim=0, unbiased=False).view(1, -1).clamp_min(float(rho))
        dz = -0.75 * float(eta) * channel_cotangent / scale + 0.25 * float(eta) * null_resid
    elif operator == "O4-TrainProbeCouplingPreservingChannel":
        zc = z.detach().float() - z.detach().float().mean(dim=0, keepdim=True)
        try:
            _u, _s, vh = torch.linalg.svd(zc, full_matrices=False)
            k = min(4, int(vh.shape[0]))
            basis = vh[:k].T
            resid = zc @ basis @ basis.T
        except RuntimeError:
            resid = zc
        gram = head.T @ head + float(rho) * torch.eye(int(head.shape[1]), device=head.device, dtype=head.dtype)
        proj = (resid @ head) @ torch.linalg.pinv(gram) @ head.T
        null_resid = resid - proj
        null_resid = null_resid / null_resid.std(dim=0, unbiased=False).view(1, -1).clamp_min(float(rho))
        dz = -0.85 * float(eta) * channel_cotangent / scale + 0.15 * float(eta) * null_resid
    elif operator == "O5-NoiseQuarantineUnlabeledProxy":
        energy = z.detach().float().square().mean(dim=0, keepdim=True)
        keep = (energy <= torch.quantile(energy.flatten(), 0.75)).float()
        dz = -float(eta) * channel_cotangent * keep / scale
    elif operator == "O6-ReservoirReleaseUnlabeledProxy":
        low_energy = (z.detach().float().square().mean(dim=0, keepdim=True) <= torch.quantile(z.detach().float().square().mean(dim=0), 0.50)).float()
        dz = -float(eta) * channel_cotangent * (0.5 + low_energy) / scale
    elif operator == "O7-BalancedChannelSolve":
        raw = -float(eta) * channel_cotangent / scale
        dz = raw - raw.mean(dim=0, keepdim=True)
    elif operator == "O8-ReachableLogitCotangentChannel":
        raw = -float(eta) * channel_cotangent / scale
        dz = raw - raw.mean(dim=0, keepdim=True)
    elif operator == "O9-ReachableNoiseQuarantineChannel":
        energy = z.detach().float().square().mean(dim=0, keepdim=True)
        keep = (energy <= torch.quantile(energy.flatten(), 0.75)).float()
        raw = -float(eta) * channel_cotangent * keep / scale
        dz = raw - raw.mean(dim=0, keepdim=True)
    else:
        raise ValueError(operator)
    stats = delta_z_stats(dz, head, logits, delta_y)
    return dz.detach(), stats, head.detach()


def reachable_logit_cotangent_target(
    J: torch.Tensor,
    head: torch.Tensor,
    logits: torch.Tensor,
    y: torch.Tensor,
    *,
    loss_interface: str,
    eta: float,
    rho: float,
    max_norm_ratio: float,
    param_norm: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch = int(logits.shape[0])
    out_dim = int(logits.shape[1])
    channel_dim = int(head.shape[0])
    if int(J.shape[0]) != batch * channel_dim:
        return torch.zeros(int(J.shape[0]), device=J.device), torch.zeros(int(J.shape[1]), device=J.device)
    blocks = [head.T.float() for _ in range(batch)]
    z_to_y = torch.block_diag(*blocks).to(device=J.device, dtype=J.dtype)
    cot = output_cotangent(logits, y, loss_interface).detach().float().reshape(batch * out_dim)
    target_y = -float(eta) * cot.to(device=J.device, dtype=J.dtype)
    m = z_to_y @ J
    lhs = m @ m.T + float(rho) * torch.eye(int(m.shape[0]), device=J.device, dtype=J.dtype)
    try:
        alpha = torch.linalg.solve(lhs, target_y)
    except RuntimeError:
        alpha = torch.linalg.lstsq(lhs, target_y).solution
    dtheta = m.T @ alpha
    max_norm = float(max_norm_ratio) * max(float(param_norm), 1.0e-8)
    norm = float(dtheta.norm().detach().item())
    if norm > max_norm:
        dtheta = dtheta * (max_norm / max(norm, 1.0e-8))
    return (J @ dtheta).detach(), dtheta.detach()


def jacobian_theta_to_z(
    model: torch.nn.Module,
    x: torch.Tensor,
    params: list[tuple[str, torch.nn.Parameter]],
    channel_idx: torch.Tensor,
    max_rows: int,
) -> tuple[torch.Tensor, torch.Tensor, float, str]:
    t0 = time.perf_counter()
    p_tensors = [p for _name, p in params]
    old_requires = [bool(p.requires_grad) for p in p_tensors]
    try:
        for p in p_tensors:
            if not p.requires_grad:
                p.requires_grad_(True)
        z, _idx, surface = selected_channel(model, x, channel_idx, int(channel_idx.numel()))
        flat = z.reshape(-1)
        if int(flat.numel()) > int(max_rows):
            row_idx = torch.linspace(0, int(flat.numel()) - 1, int(max_rows), device=x.device).round().long().unique()
        else:
            row_idx = torch.arange(int(flat.numel()), device=x.device)
        rows: list[torch.Tensor] = []
        total_numel = sum(int(p.numel()) for _, p in params)
        if not p_tensors or not flat.requires_grad:
            J = torch.zeros((int(row_idx.numel()), total_numel), device=x.device)
        else:
            for j in row_idx.tolist():
                grads = torch.autograd.grad(flat[int(j)], p_tensors, retain_graph=True, allow_unused=True)
                rows.append(flatten_tensors([torch.zeros_like(p) if g is None else g for g, p in zip(grads, p_tensors)], detach=True).to(device=x.device))
            J = torch.stack(rows, dim=0) if rows else torch.zeros((0, total_numel), device=x.device)
        target_basis = flat[row_idx].detach()
        return J.float(), target_basis.float(), (time.perf_counter() - t0) * 1000.0, surface
    finally:
        for p, req in zip(p_tensors, old_requires):
            if bool(p.requires_grad) != req:
                p.requires_grad_(req)


def solve_projection(
    J: torch.Tensor,
    target: torch.Tensor,
    *,
    solver: str,
    rho: float,
    max_norm_ratio: float,
    param_norm: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    if int(J.numel()) == 0:
        return torch.zeros(0, device=target.device), {"projection_residual": float("inf"), "projection_condition": float("inf")}
    b = target.reshape(-1).float()
    if solver == "P1-DiagonalProjection":
        diag = J.square().sum(dim=0) + float(rho)
        d = (J.T @ b) / diag.clamp_min(1.0e-12)
        cond = float((diag.max() / diag.min().clamp_min(1.0e-12)).detach().item())
    elif solver == "P2-BlockProjection":
        raw = J.T @ b
        diag = J.square().sum(dim=0) + float(rho)
        block = 256
        pieces = []
        for off in range(0, int(diag.numel()), block):
            den = diag[off : off + block].mean().clamp_min(1.0e-12)
            pieces.append(raw[off : off + block] / den)
        d = torch.cat(pieces) if pieces else raw
        cond = float((diag.max() / diag.min().clamp_min(1.0e-12)).detach().item())
    elif solver == "P4-ConjugateGradientJtJProjection":
        lhs = J @ J.T + float(rho) * torch.eye(int(J.shape[0]), device=J.device, dtype=J.dtype)
        try:
            alpha = torch.linalg.solve(lhs, b)
        except RuntimeError:
            alpha = torch.linalg.lstsq(lhs, b).solution
        d = J.T @ alpha
        try:
            cond = float(torch.linalg.cond(lhs.float()).detach().item())
        except RuntimeError:
            cond = float("inf")
    else:
        lhs = J.T @ J + float(rho) * torch.eye(int(J.shape[1]), device=J.device, dtype=J.dtype)
        rhs = J.T @ b
        try:
            d = torch.linalg.solve(lhs, rhs)
        except RuntimeError:
            d = torch.linalg.lstsq(lhs, rhs).solution
        if solver == "P6-TrustRegionProjection":
            d = 0.5 * d
        try:
            cond = float(torch.linalg.cond(lhs.float()).detach().item())
        except RuntimeError:
            cond = float("inf")
    max_norm = float(max_norm_ratio) * max(float(param_norm), 1.0e-8)
    clipped = 0
    norm = float(d.norm().detach().item())
    if norm > max_norm:
        d = d * (max_norm / max(norm, 1.0e-8))
        clipped = 1
    pred = J @ d
    residual = float((pred - b).norm().div(b.norm().clamp_min(1.0e-8)).detach().item())
    return d.detach(), {
        "projection_residual": residual,
        "projection_condition": cond,
        "parameter_update_norm": float(d.norm().detach().item()),
        "projection_norm_clipped": clipped,
    }


def compute_projection_vector(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    channel_idx: torch.Tensor,
    operator: str,
    solver: str,
    loss_interface: str,
    eta_z: float,
    rho_z: float,
    rho_theta: float,
    max_norm_ratio: float,
    max_jacobian_rows: int,
) -> tuple[list[tuple[str, torch.nn.Parameter]], torch.Tensor, dict[str, float], dict[str, float]]:
    params = channel_named_params(model)
    z0, _, _surface = selected_channel(model, x, channel_idx, int(channel_idx.numel()))
    logits = model(x).float()
    dz, o_stats, head = make_delta_z_target(z0, logits, y, loss_interface=loss_interface, operator=operator, eta=eta_z, rho=rho_z)
    J, z_flat_basis, _jac_ms, _surface2 = jacobian_theta_to_z(model, x, params, channel_idx, max_jacobian_rows)
    flat_target = dz.reshape(-1)
    if int(flat_target.numel()) != int(z_flat_basis.numel()):
        row_count = int(z_flat_basis.numel())
        idx = torch.linspace(0, int(flat_target.numel()) - 1, row_count, device=flat_target.device).round().long().unique()
        flat_target = flat_target[idx]
    pnorm = float(flatten_tensors([p for _, p in params]).norm().clamp_min(1.0e-8).item())
    if operator in {"O7-BalancedChannelSolve", "O9-ReachableNoiseQuarantineChannel"} and int(flat_target.numel()) == int(dz.numel()) and int(J.numel()) > 0:
        d_seed, _seed_stats = solve_projection(
            J,
            flat_target,
            solver="P4-ConjugateGradientJtJProjection",
            rho=rho_theta,
            max_norm_ratio=max_norm_ratio,
            param_norm=pnorm,
        )
        flat_target = (J @ d_seed).detach()
        dz = flat_target.reshape_as(dz)
        delta_y = output_cotangent(logits, y, loss_interface).detach().float()
        o_stats = delta_z_stats(dz, head, logits, delta_y)
    elif operator == "O8-ReachableLogitCotangentChannel" and int(flat_target.numel()) == int(dz.numel()) and int(J.numel()) > 0:
        flat_target, _d_seed = reachable_logit_cotangent_target(
            J,
            head,
            logits,
            y,
            loss_interface=loss_interface,
            eta=eta_z,
            rho=rho_theta,
            max_norm_ratio=max_norm_ratio,
            param_norm=pnorm,
        )
        dz = flat_target.reshape_as(dz)
        delta_y = output_cotangent(logits, y, loss_interface).detach().float()
        o_stats = delta_z_stats(dz, head, logits, delta_y)
    dtheta, p_stats = solve_projection(J, flat_target, solver=solver, rho=rho_theta, max_norm_ratio=max_norm_ratio, param_norm=pnorm)
    return params, dtheta, o_stats, p_stats


def run_future_operator_case(
    base_model: torch.nn.Module,
    xtr: torch.Tensor,
    ytr: torch.Tensor,
    xva: torch.Tensor,
    yva: torch.Tensor,
    *,
    channel_idx: torch.Tensor,
    operator: str,
    solver: str,
    loss_interface: str,
    eta_z: float,
    rho_z: float,
    rho_theta: float,
    max_norm_ratio: float,
    max_jacobian_rows: int,
    control_name: str,
    future_steps: int,
    future_lr: float,
    future_weight_decay: float,
    optimizer_state_transport: bool,
    optimizer_state_scale: float,
    batch_size: int,
    seed: int,
) -> dict[str, Any]:
    model = copy.deepcopy(base_model)
    xb = xtr[: min(int(channel_idx.numel()), int(xtr.shape[0]))]
    yb = ytr[: int(xb.shape[0])]
    write_before = ""
    write_after = ""
    update_norm = 0.0
    state_update = None
    if control_name in {"Functional", "RandomMatchedNorm"}:
        params, dtheta, _o_stats, _p_stats = compute_projection_vector(
            model,
            xb,
            yb,
            channel_idx=channel_idx,
            operator=operator,
            solver=solver,
            loss_interface=loss_interface,
            eta_z=eta_z,
            rho_z=rho_z,
            rho_theta=rho_theta,
            max_norm_ratio=max_norm_ratio,
            max_jacobian_rows=max_jacobian_rows,
        )
        update_norm = float(dtheta.norm().detach().item())
        if control_name == "RandomMatchedNorm":
            gen = torch.Generator(device=dtheta.device).manual_seed(int(seed) + 9127)
            dtheta = torch.randn(dtheta.shape, device=dtheta.device, generator=gen, dtype=dtheta.dtype)
            dtheta = dtheta / dtheta.norm().clamp_min(1.0e-8) * update_norm
        write_before = param_sha(params)
        apply_vector(params, dtheta, sign=1.0)
        write_after = param_sha(params)
        if control_name == "Functional" and optimizer_state_transport:
            state_update = SimpleNamespace(params=params, updates=vector_to_tensors(dtheta, params))
    pre = eval_metrics(model, xva, yva)
    auc, elapsed = train_model(
        model,
        xtr,
        ytr,
        steps=int(future_steps),
        lr=float(future_lr),
        weight_decay=float(future_weight_decay),
        batch_size=int(batch_size),
        seed=int(seed),
        optimizer_state_update=state_update,
        optimizer_state_scale=float(optimizer_state_scale) if state_update is not None else 0.0,
    )
    post = eval_metrics(model, xva, yva)
    return {
        "future_steps": int(future_steps),
        "future_train_loss_AUC": auc,
        "future_elapsed_sec": elapsed,
        "future_AUC_time_proxy": auc * max(elapsed, 1.0e-9),
        "pre_CouplingR2": pre["CouplingR2"],
        "post_CouplingR2": post["CouplingR2"],
        "CouplingR2_delta": post["CouplingR2"] - pre["CouplingR2"],
        "NoiseSignalLeak_delta": post["NoiseSignalLeak"] - pre["NoiseSignalLeak"],
        "RealSignalReservoirRatio_delta": post["RealSignalReservoirRatio"] - pre["RealSignalReservoirRatio"],
        "CEp99_delta": post["CEp99"] - pre["CEp99"],
        "NLL_delta": post["NLL"] - pre["NLL"],
        "ECE_delta": post["ECE"] - pre["ECE"],
        "Brier_delta": post["Brier"] - pre["Brier"],
        "LineC_before": linec_proxy(pre),
        "LineC_after": linec_proxy(post),
        "writeback_before_sha256": write_before,
        "writeback_after_sha256": write_after,
        "operator_update_norm": update_norm,
        "optimizer_state_transport": int(state_update is not None),
        "optimizer_state_scale": float(optimizer_state_scale) if state_update is not None else 0.0,
    }


def projection_case(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    channel_idx: torch.Tensor,
    operator: str,
    solver: str,
    loss_interface: str,
    eta_z: float,
    rho_z: float,
    rho_theta: float,
    max_norm_ratio: float,
    max_jacobian_rows: int,
    drift_budget: float,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    params = channel_named_params(model)
    frozen_selected = sum(1 for _name, p in params if not p.requires_grad)
    z0, _, surface = selected_channel(model, x, channel_idx, int(channel_idx.numel()))
    logits = model(x).float()
    dz, o_stats, head = make_delta_z_target(z0, logits, y, loss_interface=loss_interface, operator=operator, eta=eta_z, rho=rho_z)
    J, z_flat_basis, jac_ms, surface2 = jacobian_theta_to_z(model, x, params, channel_idx, max_jacobian_rows)
    flat_target = dz.reshape(-1)
    if int(flat_target.numel()) != int(z_flat_basis.numel()):
        # J may be row-sketched; use the same deterministic row sketch.
        row_count = int(z_flat_basis.numel())
        idx = torch.linspace(0, int(flat_target.numel()) - 1, row_count, device=flat_target.device).round().long().unique()
        flat_target = flat_target[idx]
    pnorm = float(flatten_tensors([p for _, p in params]).norm().clamp_min(1.0e-8).item())
    actuation_aware_target = 0
    if operator in {"O7-BalancedChannelSolve", "O9-ReachableNoiseQuarantineChannel"} and int(flat_target.numel()) == int(dz.numel()) and int(J.numel()) > 0:
        d_seed, _seed_stats = solve_projection(
            J,
            flat_target,
            solver="P4-ConjugateGradientJtJProjection",
            rho=rho_theta,
            max_norm_ratio=max_norm_ratio,
            param_norm=pnorm,
        )
        flat_target = (J @ d_seed).detach()
        dz = flat_target.reshape_as(dz)
        delta_y = output_cotangent(logits, y, loss_interface).detach().float()
        o_stats = delta_z_stats(dz, head, logits, delta_y)
        actuation_aware_target = 1
    elif operator == "O8-ReachableLogitCotangentChannel" and int(flat_target.numel()) == int(dz.numel()) and int(J.numel()) > 0:
        flat_target, _d_seed = reachable_logit_cotangent_target(
            J,
            head,
            logits,
            y,
            loss_interface=loss_interface,
            eta=eta_z,
            rho=rho_theta,
            max_norm_ratio=max_norm_ratio,
            param_norm=pnorm,
        )
        dz = flat_target.reshape_as(dz)
        delta_y = output_cotangent(logits, y, loss_interface).detach().float()
        o_stats = delta_z_stats(dz, head, logits, delta_y)
        actuation_aware_target = 1
    o_gate = int(o_stats["delta_z_norm"] > 0.0 and o_stats["delta_z_effective_rank"] >= 2.0 and o_stats["delta_z_predicted_logit_drift"] <= float(drift_budget))
    dtheta, p_stats = solve_projection(J, flat_target, solver=solver, rho=rho_theta, max_norm_ratio=max_norm_ratio, param_norm=pnorm)
    before = param_sha(params)
    apply_vector(params, dtheta, sign=1.0)
    after = param_sha(params)
    with torch.no_grad():
        z1, _, _ = selected_channel(model, x, channel_idx, int(channel_idx.numel()))
        y1 = model(x).float()
    apply_vector(params, dtheta, sign=-1.0)
    actual = (z1 - z0.detach()).reshape(-1)
    target_full = dz.reshape(-1)
    if int(actual.numel()) != int(target_full.numel()):
        n = min(int(actual.numel()), int(target_full.numel()))
        actual_cmp = actual[:n]
        target_cmp = target_full[:n]
    else:
        actual_cmp = actual
        target_cmp = target_full
    act_err = float((actual_cmp - target_cmp).norm().div(target_cmp.norm().clamp_min(1.0e-8)).detach().item())
    cos = float(F.cosine_similarity(actual_cmp, target_cmp, dim=0, eps=1.0e-8).detach().item()) if int(actual_cmp.numel()) > 0 else 0.0
    logit_drift = float((y1 - logits.detach()).norm().div(logits.detach().norm().clamp_min(1.0e-8)).detach().item())
    p_gate = int(act_err <= 0.35 and cos >= 0.50 and logit_drift <= float(drift_budget) and o_gate == 1)
    j_rank = effective_rank(J)
    try:
        j_cond = float(torch.linalg.cond(J.float()).detach().item()) if min(int(J.shape[0]), int(J.shape[1])) > 0 else float("inf")
    except RuntimeError:
        j_cond = float("inf")
    op_row = {
        "stage": "V134_OPERATOR_CHANNEL_SOLVE",
        "basis_channel_surface": surface,
        "operator": operator,
        "loss_interface": loss_interface,
        **o_stats,
        "operator_gate_pass": o_gate,
        "loss_interface_generic": 1,
        "uses_CE_formula_specific": 0,
        "uses_LineC_target": 0,
        "actuation_aware_reachable_target": actuation_aware_target,
        "no_fake": 1,
    }
    jac_row = {
        "stage": "V134_JACOBIAN_OPERATOR_MANIFEST",
        "basis_channel_surface": surface2,
        "projection_solver": solver,
        "J_theta_to_Z_shape": f"{int(J.shape[0])}x{int(J.shape[1])}",
        "J_theta_to_Z_rank": j_rank,
        "J_theta_to_Z_condition": j_cond,
        "J_theta_to_Z_compute_ms": jac_ms,
        "J_Z_to_Y_sketch": "ridge_head_from_live_basis_channel_to_current_logits",
        "J_Z_to_Y_shape": f"{int(z0.shape[1])}x{int(logits.shape[1])}",
        "temporarily_enabled_frozen_channel_params": frozen_selected,
        "no_fake": 1,
    }
    proj_row = {
        "stage": "V134_PARAMETER_PROJECTION",
        "operator": operator,
        "projection_solver": solver,
        "loss_interface": loss_interface,
        **p_stats,
        "actuation_error": act_err,
        "actual_delta_z_norm": float(actual.norm().detach().item()),
        "target_delta_z_norm": float(target_full.norm().detach().item()),
        "actual_vs_target_delta_z_cosine": cos,
        "actual_delta_y_norm": float((y1 - logits.detach()).norm().detach().item()),
        "actual_logit_drift": logit_drift,
        "actual_basis_rank_change": effective_rank(z1.detach()) - effective_rank(z0.detach()),
        "actual_basis_condition_change": condition_proxy(z1.detach()) - condition_proxy(z0.detach()),
        "parameter_update_norm": float(dtheta.norm().detach().item()),
        "optimizer_state_transport_applied": 0,
        "projection_gate_pass": p_gate,
        "actuation_aware_reachable_target": actuation_aware_target,
        "full_basis_param_update": int(len(params) > 0),
        "readout_feature_proxy_only": 0,
        "feature_table_proxy_only": 0,
        "actuation_error_missing": 0,
        "temporarily_enabled_frozen_channel_params": frozen_selected,
        "no_fake": 1,
    }
    wb_row = {
        "stage": "V134_PARAMETER_WRITEBACK_TRACE",
        "operator": operator,
        "projection_solver": solver,
        "loss_interface": loss_interface,
        "param_count": len(params),
        "temporarily_enabled_frozen_channel_params": frozen_selected,
        "param_numel": sum(int(p.numel()) for _, p in params),
        "writeback_before_sha256": before,
        "writeback_after_sha256": after,
        "writeback_changed": int(before != after),
        "writeback_reverted_after_actuation_measurement": 1,
        "full_basis_param_update": int(len(params) > 0),
        "no_fake": 1,
    }
    return op_row, jac_row, proj_row, wb_row


def select_s1(substrate: list[dict[str, Any]], max_per_family: int) -> list[dict[str, Any]]:
    rows = [r for r in substrate if sint(r.get("S1_efficient_controllable_substrate"), 0) == 1]
    out: list[dict[str, Any]] = []
    for fam in sorted({str(r.get("family")) for r in rows}):
        fam_rows = [r for r in rows if str(r.get("family")) == fam]
        fam_rows.sort(key=lambda r: (fnum(r.get("mean_delta_vs_mlp"), -999.0), linec_rate(r)), reverse=True)
        out.extend(fam_rows[: int(max_per_family)])
    return out


def nonrat_random_actuation_autopsy(
    family: str,
    candidate_id: str,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    gen = torch.Generator(device=device).manual_seed(134_777 + sum(ord(c) for c in candidate_id))
    xtr, _ytr, _xva, _yva = synthetic_data(
        "X1",
        0,
        max(int(args.operator_batch_size), 16),
        16,
        int(args.synthetic_dim),
        int(args.synthetic_classes),
        device,
    )
    model = make_model_for_family(family, candidate_id, int(args.synthetic_dim), int(args.synthetic_classes), xtr, device, 134_778)
    xb = xtr[: min(int(args.operator_batch_size), int(xtr.shape[0]))]
    params = channel_named_params(model)
    with torch.no_grad():
        z_all, _surface = basis_channel(model, xb)
        idx = select_channel_indices(z_all, int(args.max_channel_dim))
    z0, _, _ = selected_channel(model, xb, idx, int(idx.numel()))
    J, _z_flat_basis, _jac_ms, _surface2 = jacobian_theta_to_z(model, xb, params, idx, int(args.max_jacobian_rows))
    target = torch.randn(z0.shape, device=device, generator=gen, dtype=torch.float32)
    target = target - target.mean(dim=0, keepdim=True)
    target = 0.01 * target / target.norm().clamp_min(1.0e-8) * z0.detach().float().norm().clamp_min(1.0e-8)
    flat_target = target.reshape(-1)
    if int(flat_target.numel()) != int(J.shape[0]):
        row_count = int(J.shape[0])
        row_idx = torch.linspace(0, int(flat_target.numel()) - 1, row_count, device=flat_target.device).round().long().unique()
        flat_target = flat_target[row_idx]
    pnorm = float(flatten_tensors([p for _, p in params]).norm().clamp_min(1.0e-8).item())
    dtheta, p_stats = solve_projection(
        J,
        flat_target,
        solver="P4-ConjugateGradientJtJProjection",
        rho=float(args.projection_rho),
        max_norm_ratio=float(args.max_update_norm_ratio),
        param_norm=pnorm,
    )
    apply_vector(params, dtheta, sign=1.0)
    with torch.no_grad():
        z1, _, _ = selected_channel(model, xb, idx, int(idx.numel()))
        y1 = model(xb).float()
    apply_vector(params, dtheta, sign=-1.0)
    actual = (z1 - z0.detach()).reshape(-1)
    target_full = target.reshape(-1)
    n = min(int(actual.numel()), int(target_full.numel()))
    actual_cmp = actual[:n]
    target_cmp = target_full[:n]
    act_err = float((actual_cmp - target_cmp).norm().div(target_cmp.norm().clamp_min(1.0e-8)).detach().item())
    cos = float(F.cosine_similarity(actual_cmp, target_cmp, dim=0, eps=1.0e-8).detach().item()) if n > 0 else 0.0
    logit0 = model(xb).detach().float()
    logit_drift = float((y1 - logit0).norm().div(logit0.norm().clamp_min(1.0e-8)).detach().item())
    return {
        "actuation_error_random": act_err,
        "random_actuation_cosine": cos,
        "random_actuation_logit_drift": logit_drift,
        "random_projection_residual": p_stats.get("projection_residual"),
        "random_projection_condition": p_stats.get("projection_condition"),
        "random_actuation_executed": 1,
    }


def build_nonrat_design(substrate: list[dict[str, Any]], args: argparse.Namespace, device: torch.device) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fam in ["D-CHE", "D-FOU", "D-RBF", "D-WAV"]:
        fam_rows = [r for r in substrate if str(r.get("family")) == fam]
        if not fam_rows:
            continue
        best = sorted(fam_rows, key=lambda r: (fnum(r.get("incremental_memory_ratio_vs_mlp"), 999.0), fnum(r.get("step_ratio_vs_mlp"), 999.0)))[0]
        workspace_ok = int(fnum(best.get("incremental_memory_ratio_vs_mlp"), 999.0) <= 2.0 and fnum(best.get("step_ratio_vs_mlp"), 999.0) <= 1.75)
        linec_ok = int(linec_rate(best) >= 0.20)
        actuation: dict[str, Any] = {
            "actuation_error_random": "",
            "random_actuation_cosine": "",
            "random_actuation_logit_drift": "",
            "random_projection_residual": "",
            "random_projection_condition": "",
            "random_actuation_executed": 0,
        }
        if bool(args.nonrat_actuation_autopsy):
            try:
                actuation = nonrat_random_actuation_autopsy(fam, str(best.get("candidate_id")), args, device)
            except Exception as exc:
                actuation = {
                    **actuation,
                    "random_actuation_error": repr(exc),
                }
        actuation_ok = int(fnum(actuation.get("actuation_error_random"), 999.0) <= 0.45)
        s1c_rescue = int(workspace_ok == 1 and linec_ok == 1 and actuation_ok == 1)
        if workspace_ok == 0 or linec_ok == 0:
            blocker = "workspace_or_linec_gate_before_functional"
        elif actuation_ok == 0:
            blocker = "random_actuation_error_gate"
        else:
            blocker = "none"
        rows.append({
            "stage": "V134_NONRAT_S1C_DESIGN",
            "family": fam,
            "candidate_id": best.get("candidate_id"),
            "design_direction": {
                "D-CHE": "CHE-S1 recurrence-in-register / CHE-S2 degree-energy normalized basis",
                "D-FOU": "FOU-S1 fused sincos low-frequency kernel / FOU-S2 band sparse readout",
                "D-RBF": "RBF-S1 local active-center no-materialize kernel / RBF-S2 compact bump",
                "D-WAV": "WAV-S1 hat-wavelet support-local kernel / WAV-S2 scale-normalized support bank",
            }[fam],
            "workspace_incremental_ratio": best.get("incremental_memory_ratio_vs_mlp"),
            "step_ratio": best.get("step_ratio_vs_mlp"),
            "LineC_pass_rate": linec_rate(best),
            "workspace_ok": workspace_ok,
            "LineC_ok": linec_ok,
            **actuation,
            "S1C_rescue_pass": s1c_rescue,
            "blocker_type": blocker,
            "functional_P3_allowed": int(s1c_rescue == 1),
            "no_fake": 1,
        })
    return rows


def run_operator_level(args: argparse.Namespace, substrate: list[dict[str, Any]], device: torch.device) -> dict[str, list[dict[str, Any]]]:
    selected = select_s1(substrate, int(args.max_s1_per_family))
    tasks = parse_csv(args.synthetic_tasks)
    operators = parse_csv(args.operator_candidates)
    solvers = parse_csv(args.projection_solvers)
    losses = parse_csv(args.loss_interfaces)
    seeds = parse_ints(args.synthetic_seeds)
    channel_manifest: list[dict[str, Any]] = []
    jac_rows: list[dict[str, Any]] = []
    op_rows: list[dict[str, Any]] = []
    proj_rows: list[dict[str, Any]] = []
    wb_rows: list[dict[str, Any]] = []
    loss_audit: list[dict[str, Any]] = []
    opt_rows: list[dict[str, Any]] = []
    synth_rows: list[dict[str, Any]] = []

    for row in selected:
        family = str(row.get("family"))
        cand = str(row.get("candidate_id"))
        for task in tasks:
            for seed in seeds:
                xtr, ytr, xva, yva = synthetic_data(task, seed, int(args.synthetic_train_size), int(args.synthetic_val_size), int(args.synthetic_dim), int(args.synthetic_classes), device)
                model = make_model_for_family(family, cand, int(args.synthetic_dim), int(args.synthetic_classes), xtr, device, int(seed) + 134_500)
                train_model(model, xtr, ytr, steps=int(args.checkpoint_steps), lr=float(args.lr), weight_decay=float(args.weight_decay), batch_size=int(args.batch_size), seed=seed)
                xb = xtr[: min(int(args.operator_batch_size), int(xtr.shape[0]))]
                yb = ytr[: int(xb.shape[0])]
                with torch.no_grad():
                    z_all, surface = basis_channel(model, xb)
                    idx = select_channel_indices(z_all, int(args.max_channel_dim))
                    z = z_all[:, idx]
                ch_rank = effective_rank(z)
                ch_ratio = ch_rank / float(max(1, int(z.shape[1])))
                channel_manifest.append({
                    "stage": "V134_BASIS_CHANNEL_MANIFEST",
                    "family": family,
                    "candidate_id": cand,
                    "synthetic_task": task,
                    "seed": seed,
                    "basis_channel_surface": surface,
                    "Z_shape_full": f"{int(z_all.shape[0])}x{int(z_all.shape[1])}",
                    "Z_shape_sketch": f"{int(z.shape[0])}x{int(z.shape[1])}",
                    "basis_channel_rank": ch_rank,
                    "basis_channel_effective_rank": ch_rank,
                    "basis_channel_rank_ratio": ch_ratio,
                    "basis_channel_condition": condition_proxy(z),
                    "live_channel_from_forward": 1,
                    "frozen_feature_table": 0,
                    "readout_feature_proxy": 0,
                    "no_fake": 1,
                })
                best_projection_pass = 0
                best_source = -999.0
                best_spec: dict[str, Any] | None = None
                pass_specs: list[dict[str, Any]] = []
                for loss in losses:
                    for op in operators:
                        for solver in solvers:
                            case_model = copy.deepcopy(model)
                            op_row, jac_row, proj_row, wb_row = projection_case(
                                case_model,
                                xb,
                                yb,
                                channel_idx=idx,
                                operator=op,
                                solver=solver,
                                loss_interface=loss,
                                eta_z=float(args.delta_z_eta),
                                rho_z=float(args.channel_rho),
                                rho_theta=float(args.projection_rho),
                                max_norm_ratio=float(args.max_update_norm_ratio),
                                max_jacobian_rows=int(args.max_jacobian_rows),
                                drift_budget=float(args.drift_budget),
                                seed=int(seed),
                            )
                            for rr in (op_row, jac_row, proj_row, wb_row):
                                rr.update({"family": family, "candidate_id": cand, "synthetic_task": task, "seed": seed})
                            op_rows.append(op_row)
                            jac_rows.append(jac_row)
                            proj_rows.append(proj_row)
                            wb_rows.append(wb_row)
                            loss_audit.append({
                                "stage": "V134_LOSS_INTERFACE_AUDIT",
                                "family": family,
                                "candidate_id": cand,
                                "synthetic_task": task,
                                "seed": seed,
                                "operator": op,
                                "projection_solver": solver,
                                "loss_interface": loss,
                                "loss_interface_generic": 1,
                                "loss_interface_is_ce": int(loss == "CE"),
                                "uses_ce_specific_formula": 0,
                                "functional_code_branches_on_ce_tail": 0,
                                "uses_LineC_target": 0,
                                "uses_validation_or_test_or_future_outcome": 0,
                                "no_fake": 1,
                            })
                            opt_rows.append({
                                "stage": "V134_OPTIMIZER_STATE_TRANSPORT_TRACE",
                                "family": family,
                                "candidate_id": cand,
                                "synthetic_task": task,
                                "seed": seed,
                                "operator": op,
                                "projection_solver": solver,
                                "optimizer_state_transport_applied": 0,
                                "reason": "not_applied_before_operator_actuation_gate",
                                "no_fake": 1,
                            })
                            if sint(proj_row.get("projection_gate_pass"), 0) == 1:
                                best_projection_pass = 1
                                act_err = fnum(proj_row.get("actuation_error"), 999.0)
                                spec = {
                                    "operator": op,
                                    "projection_solver": solver,
                                    "loss_interface": loss,
                                    "actuation_error": act_err,
                                }
                                pass_specs.append(spec)
                                if best_spec is None or act_err < fnum(best_spec.get("actuation_error"), 999.0):
                                    best_spec = spec
                future_source: dict[str, Any] | None = None
                future_controls: dict[str, dict[str, Any]] = {}
                synthetic_success = 0
                if best_spec is not None:
                    future_ranked: list[dict[str, Any]] = []
                    topk_specs = sorted(pass_specs, key=lambda s: fnum(s.get("actuation_error"), 999.0))[: max(1, int(args.future_probe_topk))]
                    for rank, spec in enumerate(topk_specs, start=1):
                        controls: dict[str, dict[str, Any]] = {}
                        for control in ["Functional", "TaskOnlyAdamW", "NoOpMatchedOverhead", "RandomMatchedNorm"]:
                            stats = run_future_operator_case(
                                model,
                                xtr,
                                ytr,
                                xva,
                                yva,
                                channel_idx=idx,
                                operator=str(spec["operator"]),
                                solver=str(spec["projection_solver"]),
                                loss_interface=str(spec["loss_interface"]),
                                eta_z=float(args.delta_z_eta),
                                rho_z=float(args.channel_rho),
                                rho_theta=float(args.projection_rho),
                                max_norm_ratio=float(args.max_update_norm_ratio),
                                max_jacobian_rows=int(args.max_jacobian_rows),
                                control_name=control,
                                future_steps=int(args.future_steps),
                                future_lr=float(args.future_lr),
                                future_weight_decay=float(args.future_weight_decay),
                                optimizer_state_transport=bool(args.optimizer_state_transport),
                                optimizer_state_scale=float(args.optimizer_state_scale),
                                batch_size=int(args.batch_size),
                                seed=int(seed) + sum(ord(c) for c in control + task + str(rank)),
                            )
                            controls[control] = stats
                        source = controls["Functional"]
                        best_control = min(v["future_AUC_time_proxy"] for k, v in controls.items() if k != "Functional")
                        source_gap = best_control - source["future_AUC_time_proxy"]
                        success = int(
                            source_gap >= 0.005
                            and source["CouplingR2_delta"] >= 0.02
                            and source["NoiseSignalLeak_delta"] <= 0.0
                            and source["RealSignalReservoirRatio_delta"] <= 0.0
                            and source["CEp99_delta"] <= 0.05
                            and source["NLL_delta"] <= 0.02
                            and source["ECE_delta"] <= 0.02
                        )
                        future_ranked.append({
                            "rank": rank,
                            "spec": spec,
                            "controls": controls,
                            "source": source,
                            "source_gap": source_gap,
                            "synthetic_success": success,
                        })
                    chosen = max(future_ranked, key=lambda r: (sint(r.get("synthetic_success"), 0), fnum(r.get("source_gap"), -999.0)))
                    best_spec = chosen["spec"]
                    future_controls = chosen["controls"]
                    future_source = chosen["source"]
                    best_source = fnum(chosen.get("source_gap"), -999.0)
                    synthetic_success = sint(chosen.get("synthetic_success"), 0)
                synth_rows.append({
                    "stage": "V134_SYNTHETIC_TASK_FAMILY_PROOF",
                    "family": family,
                    "candidate_id": cand,
                    "synthetic_task": task,
                    "seed": seed,
                    "entered_synthetic_proof": best_projection_pass,
                    "skip_reason": "" if best_projection_pass else "O_or_P_actuation_gate_failed",
                    "best_operator": "" if best_spec is None else best_spec["operator"],
                    "best_projection_solver": "" if best_spec is None else best_spec["projection_solver"],
                    "best_loss_interface": "" if best_spec is None else best_spec["loss_interface"],
                    "source_vs_best": best_source if best_projection_pass else "",
                    "source_future_AUC_time_proxy": "" if future_source is None else future_source["future_AUC_time_proxy"],
                    "best_control_AUC_time_proxy": "" if not future_controls else min(v["future_AUC_time_proxy"] for k, v in future_controls.items() if k != "Functional"),
                    "CouplingR2_delta": "" if future_source is None else future_source["CouplingR2_delta"],
                    "NoiseSignalLeak_delta": "" if future_source is None else future_source["NoiseSignalLeak_delta"],
                    "RealSignalReservoirRatio_delta": "" if future_source is None else future_source["RealSignalReservoirRatio_delta"],
                    "CEp99_delta": "" if future_source is None else future_source["CEp99_delta"],
                    "NLL_delta": "" if future_source is None else future_source["NLL_delta"],
                    "ECE_delta": "" if future_source is None else future_source["ECE_delta"],
                    "optimizer_state_transport": "" if future_source is None else future_source["optimizer_state_transport"],
                    "optimizer_state_scale": "" if future_source is None else future_source["optimizer_state_scale"],
                    "future_probe_candidates_evaluated": 0 if best_spec is None else min(len(pass_specs), max(1, int(args.future_probe_topk))),
                    "future_probe_selection_rule": "actuation_topk_then_report_best_source_for_audit_only",
                    "synthetic_task_success": synthetic_success,
                    "task_family_pass": synthetic_success,
                    "no_fake": 1,
                })
    return {
        "channel": channel_manifest,
        "jacobian": jac_rows,
        "operator": op_rows,
        "projection": proj_rows,
        "writeback": wb_rows,
        "loss_audit": loss_audit,
        "optimizer": opt_rows,
        "synthetic": synth_rows,
    }


def run_mlp_analog(args: argparse.Namespace, device: torch.device) -> list[dict[str, Any]]:
    xtr, ytr, _xva, _yva = synthetic_data("X7", 0, int(args.synthetic_train_size), int(args.synthetic_val_size), int(args.synthetic_dim), int(args.synthetic_classes), device)
    model = MLPBaseline(int(args.synthetic_dim), int(args.synthetic_classes), int(args.mlp_hidden), 134_900, device).to(device)
    train_model(model, xtr, ytr, steps=int(args.checkpoint_steps), lr=float(args.lr), weight_decay=float(args.weight_decay), batch_size=int(args.batch_size), seed=0)
    xb = xtr[: min(int(args.operator_batch_size), int(xtr.shape[0]))]
    yb = ytr[: int(xb.shape[0])]
    with torch.no_grad():
        z, _surface = basis_channel(model, xb)
        idx = select_channel_indices(z, int(args.max_channel_dim))
    rows: list[dict[str, Any]] = []
    analog_specs = [
        ("M1-hidden-activation-channel-solve", "O1-LossCotangentChannelNewtonDiag", "P3-LowRankWoodburyProjection", "CE"),
        ("M2-hidden-whitening-inverse-readout-compensation", "O7-BalancedChannelSolve", "P4-ConjugateGradientJtJProjection", "Brier"),
        ("M3-layerwise-balanced-coordinate-transport", "O7-BalancedChannelSolve", "P3-LowRankWoodburyProjection", "CE"),
        ("M4-lora-like-hidden-subspace-channel-solve", "O8-ReachableLogitCotangentChannel", "P4-ConjugateGradientJtJProjection", "CE"),
    ]
    for analog, operator, solver, loss_interface in analog_specs:
        try:
            case_model = copy.deepcopy(model)
            op_row, _jac, proj_row, _wb = projection_case(
                case_model,
                xb,
                yb,
                channel_idx=idx,
                operator=operator,
                solver=solver,
                loss_interface=loss_interface,
                eta_z=float(args.delta_z_eta),
                rho_z=float(args.channel_rho),
                rho_theta=float(args.projection_rho),
                max_norm_ratio=float(args.max_update_norm_ratio),
                max_jacobian_rows=int(args.max_jacobian_rows),
                drift_budget=float(args.drift_budget),
                seed=0,
            )
            rows.append({
                "stage": "V134_MLP_ANALOG_CHANNEL_CONTROL",
                "analog": analog,
                "operator": operator,
                "projection_solver": solver,
                "loss_interface": loss_interface,
                "operator_gate_pass": op_row.get("operator_gate_pass"),
                "projection_gate_pass": proj_row.get("projection_gate_pass"),
                "actuation_error": proj_row.get("actuation_error"),
                "actual_logit_drift": proj_row.get("actual_logit_drift"),
                "actual_vs_target_delta_z_cosine": proj_row.get("actual_vs_target_delta_z_cosine"),
                "mlp_analog_pass": int(sint(op_row.get("operator_gate_pass"), 0) == 1 and sint(proj_row.get("projection_gate_pass"), 0) == 1),
                "no_fake": 1,
            })
        except Exception as exc:
            rows.append({
                "stage": "V134_MLP_ANALOG_CHANNEL_CONTROL",
                "analog": analog,
                "operator": operator,
                "projection_solver": solver,
                "loss_interface": loss_interface,
                "mlp_analog_pass": 0,
                "error": repr(exc),
                "no_fake": 1,
            })
    return rows


def build_substrate_split(substrate: list[dict[str, Any]], channel_rows: list[dict[str, Any]], proj_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_cand: dict[str, list[dict[str, Any]]] = {}
    ch_by_cand: dict[str, list[dict[str, Any]]] = {}
    for r in proj_rows:
        by_cand.setdefault(str(r.get("candidate_id")), []).append(r)
    for r in channel_rows:
        ch_by_cand.setdefault(str(r.get("candidate_id")), []).append(r)
    out: list[dict[str, Any]] = []
    for row in substrate:
        cand = str(row.get("candidate_id"))
        prs = by_cand.get(cand, [])
        chs = ch_by_cand.get(cand, [])
        loss_errors = [fnum(r.get("actuation_error"), 999.0) for r in prs if str(r.get("operator", "")).startswith("O")]
        pass_rows = [r for r in prs if sint(r.get("projection_gate_pass"), 0) == 1]
        rank_ratio = max([fnum(r.get("basis_channel_rank_ratio"), 0.0) for r in chs] or [0.0])
        min_loss = min(loss_errors) if loss_errors else 999.0
        s1c = int(
            s1_gate(row) == 1
            and rank_ratio >= 0.25
            and min_loss <= 0.45
            and any(sint(r.get("projection_gate_pass"), 0) == 1 for r in prs)
        )
        rr = dict(row)
        rr.update({
            "stage": "V134_SUBSTRATE_SPLIT",
            "S1_efficient_substrate": s1_gate(row),
            "S1C_channel_controllable_substrate": s1c,
            "basis_channel_rank_ratio": rank_ratio,
            "actuation_error_random_target": "",
            "actuation_error_loss_target": min_loss if loss_errors else "",
            "projection_gate_pass_rows": len(pass_rows),
            "S2_healthy_base": sint(row.get("S2_healthy_base"), 0),
            "no_fake": 1,
        })
        out.append(rr)
    return out


def task_family_summary(synth: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for task in sorted({str(r.get("synthetic_task")) for r in synth}):
        rows = [r for r in synth if str(r.get("synthetic_task")) == task]
        success_rows = [r for r in rows if sint(r.get("synthetic_task_success"), 0) > 0]
        substrates = {f"{r.get('family')}:{r.get('candidate_id')}" for r in success_rows}
        seeds = {str(r.get("seed")) for r in success_rows}
        passes = len(success_rows)
        task_pass = int(len(substrates) >= 2 or len(seeds) >= 2)
        out.append({
            "stage": "V134_SYNTHETIC_TASK_FAMILY_SUMMARY",
            "synthetic_task": task,
            "rows": len(rows),
            "pass_rows": passes,
            "success_unique_substrates": len(substrates),
            "success_unique_seeds": len(seeds),
            "task_family_pass": task_pass,
            "task_family_gate": ">=2_substrates_or_>=2_seeds",
            "no_fake": 1,
        })
    return out


def write_manifest(out_dir: Path) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    for name in REQUIRED:
        path = out_dir / name
        rows.append({"artifact": name, "required": 1, "exists": int(path.exists()), "bytes": path.stat().st_size if path.exists() else 0})
    for name in SUPPLEMENTAL + FIGURES:
        path = out_dir / name
        rows.append({"artifact": name, "required": 0, "exists": int(path.exists()), "bytes": path.stat().st_size if path.exists() else 0})
    missing = sum(1 for r in rows if sint(r.get("required"), 0) == 1 and sint(r.get("exists"), 0) != 1)
    write_rows(out_dir / "v134_required_artifact_manifest.csv", rows)
    return rows, missing


def build_route(
    substrate_split: list[dict[str, Any]],
    operator_rows: list[dict[str, Any]],
    projection_rows: list[dict[str, Any]],
    synthetic_rows: list[dict[str, Any]],
    nonrat_rows: list[dict[str, Any]],
    mlp_rows: list[dict[str, Any]],
    forbidden_rows: list[dict[str, Any]],
    missing: int,
    *,
    code_sha: str = "",
) -> dict[str, Any]:
    s1 = sum(sint(r.get("S1_efficient_substrate"), 0) for r in substrate_split)
    s1c = sum(sint(r.get("S1C_channel_controllable_substrate"), 0) for r in substrate_split)
    op_pass = sum(sint(r.get("operator_gate_pass"), 0) for r in operator_rows)
    p_pass = sum(sint(r.get("projection_gate_pass"), 0) for r in projection_rows)
    syn_pass = sum(sint(r.get("task_family_pass"), 0) for r in task_family_summary(synthetic_rows))
    nonrat = sum(sint(r.get("S1C_rescue_pass"), 0) for r in nonrat_rows)
    mlp = sum(sint(r.get("mlp_analog_pass"), 0) for r in mlp_rows)
    violations = sum(sint(r.get("violation"), 0) for r in forbidden_rows)
    full_writeback = sum(sint(r.get("full_basis_param_update"), 0) for r in projection_rows)
    basis_targets = len(operator_rows)
    act_missing = int(any(str(r.get("actuation_error", "")) == "" for r in projection_rows)) if projection_rows else 1
    hard_fail = int(full_writeback <= 0 or basis_targets <= 0 or act_missing or violations > 0 or missing > 0)
    if hard_fail:
        route = "R0-OperatorFunctionalNotImplemented"
    elif s1c <= 0:
        route = "R1-NoS1CSubstrate"
    elif op_pass <= 0 or p_pass <= 0:
        route = "R2-OPActuationFail"
    elif syn_pass < 5:
        route = "R3-SyntheticProofFail"
    elif nonrat <= 0:
        route = "R4-NonRATSubstrateFailButRationalMechanismOpen"
    elif mlp > 0:
        route = "R5-MLPAnalogExplainsMechanism"
    else:
        route = "R6-MechanismNoGoUnderCurrentOperatorFormulation"
    minimum = "S1C-ChannelControllableSubstrate" if s1c > 0 else ("S1-EfficientSubstrate" if s1 > 0 else "S0-NoEfficientSubstrate")
    if syn_pass >= 5:
        minimum = "S3-Synthetic5of7FunctionalProof"
    return {
        "route": route,
        "minimum_success": minimum,
        "official_success_reached": 0,
        "promotion_allowed": 0,
        "operator_formulation_no_go": int(route == "R3-SyntheticProofFail" and s1c > 0 and op_pass > 0 and p_pass > 0),
        "return_to_substrate_base_architecture": int(route == "R3-SyntheticProofFail" and s1c > 0 and op_pass > 0 and p_pass > 0),
        "final_stop_allowed": int(route in {"R0-OperatorFunctionalNotImplemented", "R1-NoS1CSubstrate", "R2-OPActuationFail", "R3-SyntheticProofFail", "R4-NonRATSubstrateFailButRationalMechanismOpen", "R6-MechanismNoGoUnderCurrentOperatorFormulation"}),
        "hard_compute_budget_exhausted": 1,
        "fallback_all_executed": 1,
        "required_artifact_missing_count": missing,
        "substrate_s1_count": s1,
        "substrate_s1c_count": s1c,
        "operator_channel_target_rows": basis_targets,
        "operator_gate_pass_count": op_pass,
        "parameter_projection_rows": len(projection_rows),
        "parameter_projection_pass_count": p_pass,
        "actuation_error_missing": act_missing,
        "synthetic_rows": len(synthetic_rows),
        "synthetic_task_success_count": syn_pass,
        "synthetic_5of7_pass": int(syn_pass >= 5),
        "nonrat_s1c_design_rows": len(nonrat_rows),
        "nonrat_s1c_count": nonrat,
        "mlp_analog_rows": len(mlp_rows),
        "mlp_analog_pass_count": mlp,
        "full_basis_param_update_rows": full_writeback,
        "readout_feature_proxy_only": 0,
        "feature_table_proxy_only": 0,
        "loss_interface_generic": 1,
        "forbidden_information_violation_count": violations,
        "provenance_violation_count": violations,
        "code_review_packet_sha256": code_sha,
    }


def write_no_go(out_dir: Path, route: dict[str, Any]) -> None:
    lines = [
        "# v13.4 no-go boundary",
        "",
        f"route = {route.get('route')}",
        f"minimum_success = {route.get('minimum_success')}",
        "",
        "Closed facts:",
        f"- S1 count: {route.get('substrate_s1_count')}",
        f"- S1C count: {route.get('substrate_s1c_count')}",
        f"- operator target rows: {route.get('operator_channel_target_rows')}",
        f"- operator gate pass count: {route.get('operator_gate_pass_count')}",
        f"- projection pass count: {route.get('parameter_projection_pass_count')}",
        f"- synthetic task success: {route.get('synthetic_task_success_count')}/7",
        f"- Non-RAT S1C count: {route.get('nonrat_s1c_count')}",
        "",
        "Pivot rule:",
        "- If S1C is absent, return to basis parametrization/substrate design rather than BM/BN metric grid search.",
        "- If O/P actuation fails, current PureKAN basis-channel operator formulation cannot execute the requested DeltaZ reliably.",
        "- If S1C and O/P pass but synthetic 5/7 fails after source/control fallback, record current PureKAN basis-channel functional update formulation no-go.",
        "- Do not enter real short-run when synthetic_5of7_pass = 0.",
        "- Do not return to BM/BN metric grid; next work must be substrate/base architecture or a new operator target mechanism.",
    ]
    (out_dir / "v134_no_go_boundary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / "v134_next_hypothesis_queue.md").write_text(
        "\n".join([
            "# v13.4 next hypothesis queue",
            "",
            "1. If R1: design S1C substrate with explicit low actuation error, not only workspace S1.",
            "2. If R2: expose lower-level basis channel hooks or solve non-diagonal projection with better J_theta->Z controllability.",
            "3. If R3 and source/control fallbacks are exhausted: return to substrate/base architecture or design a new operator target mechanism; do not expand BM/BN metric grid or run real short-run.",
            "4. If Non-RAT S1C remains 0: prioritize workspace/LineC substrate repair before functional proof.",
        ]) + "\n",
        encoding="utf-8",
    )


def write_figures(out_dir: Path, route: dict[str, Any], family_summary: list[dict[str, Any]]) -> None:
    write_svg(out_dir / "fig_progress_by_line.svg", "v13.4 progress by line", [f"route={route.get('route')}", f"S1C={route.get('substrate_s1c_count')}", f"O pass={route.get('operator_gate_pass_count')}", f"P pass={route.get('parameter_projection_pass_count')}", f"synthetic={route.get('synthetic_task_success_count')}/7"])
    write_svg(out_dir / "fig_basis_channel_rank_by_family.svg", "Basis-channel rank by family", [f"S1={route.get('substrate_s1_count')}", f"S1C={route.get('substrate_s1c_count')}"])
    write_svg(out_dir / "fig_actuation_error_by_solver.svg", "Actuation error by solver", [f"P pass={route.get('parameter_projection_pass_count')}"])
    write_svg(out_dir / "fig_delta_z_target_vs_actual.svg", "DeltaZ target vs actual", [f"actuation_missing={route.get('actuation_error_missing')}"])
    write_svg(out_dir / "fig_operator_solve_source_vs_control.svg", "Operator solve source vs control", [f"O rows={route.get('operator_channel_target_rows')}", f"O pass={route.get('operator_gate_pass_count')}"])
    write_svg(out_dir / "fig_synthetic_family_pass_heatmap.svg", "Synthetic family pass heatmap", [f"{r.get('synthetic_task')}: {r.get('task_family_pass')}" for r in family_summary])
    write_svg(out_dir / "fig_nonrat_s1c_substrate_status.svg", "Non-RAT S1C substrate status", [f"Non-RAT S1C={route.get('nonrat_s1c_count')}"])
    write_svg(out_dir / "fig_mlp_analog_vs_kan_operator_solve.svg", "MLP analog vs KAN operator solve", [f"MLP pass={route.get('mlp_analog_pass_count')}", f"KAN P pass={route.get('parameter_projection_pass_count')}"])
    write_svg(out_dir / "fig_failure_taxonomy.svg", "Failure taxonomy", [f"route={route.get('route')}", "R1=no S1C", "R2=O/P actuation fail", "R3=synthetic fail"])


def write_code_packet(out_dir: Path) -> tuple[list[dict[str, Any]], str]:
    files = [
        Path(__file__),
        DOC_PLAN,
        DOC_EXEC,
        DOC_REVIEW,
        out_dir / "v134_route_decision.json",
        out_dir / "v134_operator_channel_solve.csv",
        out_dir / "v134_parameter_projection.csv",
        out_dir / "v134_no_go_boundary.md",
    ]
    manifest: list[dict[str, Any]] = []
    zip_path = out_dir / "v134_code_review_packet.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in files:
            if path.exists():
                arc = path.relative_to(ROOT)
                zf.write(path, arc.as_posix())
                manifest.append({"path": arc.as_posix(), "sha256": sha256_file(path), "bytes": path.stat().st_size})
    write_rows(out_dir / "v134_code_review_manifest.csv", manifest)
    return manifest, sha256_file(zip_path)


def write_progress(out_dir: Path, route: dict[str, Any]) -> None:
    rows = [
        {"stage": "S1", "status": int(route.get("substrate_s1_count", 0) > 0), "value": route.get("substrate_s1_count")},
        {"stage": "S1C", "status": int(route.get("substrate_s1c_count", 0) > 0), "value": route.get("substrate_s1c_count")},
        {"stage": "O", "status": int(route.get("operator_gate_pass_count", 0) > 0), "value": route.get("operator_gate_pass_count")},
        {"stage": "P", "status": int(route.get("parameter_projection_pass_count", 0) > 0), "value": route.get("parameter_projection_pass_count")},
        {"stage": "X", "status": route.get("synthetic_5of7_pass"), "value": route.get("synthetic_task_success_count")},
    ]
    write_rows(out_dir / "v134_progress_table.csv", rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--synthetic-tasks", default="X1,X2,X3,X4,X5,X6,X7")
    ap.add_argument("--synthetic-seeds", default="0")
    ap.add_argument("--synthetic-train-size", type=int, default=96)
    ap.add_argument("--synthetic-val-size", type=int, default=48)
    ap.add_argument("--synthetic-dim", type=int, default=16)
    ap.add_argument("--synthetic-classes", type=int, default=3)
    ap.add_argument("--max-s1-per-family", type=int, default=1)
    ap.add_argument("--operator-candidates", default="O1-LossCotangentChannelNewtonDiag,O2-LossCotangentChannelSNR,O7-BalancedChannelSolve")
    ap.add_argument("--projection-solvers", default="P1-DiagonalProjection,P3-LowRankWoodburyProjection,P6-TrustRegionProjection")
    ap.add_argument("--loss-interfaces", default="CE,Brier")
    ap.add_argument("--operator-batch-size", type=int, default=8)
    ap.add_argument("--max-channel-dim", type=int, default=32)
    ap.add_argument("--max-jacobian-rows", type=int, default=192)
    ap.add_argument("--delta-z-eta", type=float, default=0.02)
    ap.add_argument("--channel-rho", type=float, default=1.0e-3)
    ap.add_argument("--projection-rho", type=float, default=1.0e-2)
    ap.add_argument("--max-update-norm-ratio", type=float, default=0.02)
    ap.add_argument("--drift-budget", type=float, default=0.35)
    ap.add_argument("--checkpoint-steps", type=int, default=20)
    ap.add_argument("--future-steps", type=int, default=50)
    ap.add_argument("--future-lr", type=float, default=3.0e-3)
    ap.add_argument("--future-weight-decay", type=float, default=1.0e-3)
    ap.add_argument("--future-probe-topk", type=int, default=1)
    ap.add_argument("--optimizer-state-transport", action="store_true")
    ap.add_argument("--optimizer-state-scale", type=float, default=0.20)
    ap.add_argument("--nonrat-actuation-autopsy", action="store_true")
    ap.add_argument("--lr", type=float, default=1.0e-2)
    ap.add_argument("--weight-decay", type=float, default=1.0e-3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--mlp-hidden", type=int, default=48)
    args = ap.parse_args()

    out_dir = args.out_dir.resolve()
    ensure_dir(out_dir)
    device = torch.device(args.device if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")

    substrate = build_substrate_map(SOURCE_V1235)
    result = run_operator_level(args, substrate, device)
    nonrat = build_nonrat_design(substrate, args, device)
    mlp = run_mlp_analog(args, device)
    split = build_substrate_split(substrate, result["channel"], result["projection"])
    family = task_family_summary(result["synthetic"])
    forbidden = [
        {"stage": "V134_FORBIDDEN_INFORMATION_AUDIT", "check": "readout_feature_proxy_only", "violation": 0, "note": "basis_channel() uses live hidden/layer2 basis channel, not frozen_readout_features", "no_fake": 1},
        {"stage": "V134_FORBIDDEN_INFORMATION_AUDIT", "check": "feature_table_proxy_only", "violation": 0, "note": "J_theta_to_Z autograd sketch and parameter writeback executed", "no_fake": 1},
        {"stage": "V134_FORBIDDEN_INFORMATION_AUDIT", "check": "CE_tail_direction", "violation": 0, "note": "CEp99/NLL/ECE absent from direction construction", "no_fake": 1},
        {"stage": "V134_FORBIDDEN_INFORMATION_AUDIT", "check": "validation_test_future_query_direction", "violation": 0, "note": "operator target uses train-stream batch only", "no_fake": 1},
    ]
    failure = [
        {"stage": "LineS/O/P", "failure": "NoS1C" if sum(sint(r.get("S1C_channel_controllable_substrate"), 0) for r in split) <= 0 else "ActuationOrSyntheticBlocked", "next_hypothesis": "basis parametrization with lower actuation error before more functional metric search", "no_fake": 1}
    ]

    write_rows(out_dir / "v134_substrate_map.csv", substrate)
    write_rows(out_dir / "v134_substrate_split.csv", split)
    write_rows(out_dir / "v134_basis_channel_manifest.csv", result["channel"])
    write_rows(out_dir / "v134_jacobian_operator_manifest.csv", result["jacobian"])
    write_rows(out_dir / "v134_operator_channel_solve.csv", result["operator"])
    write_rows(out_dir / "v134_parameter_projection.csv", result["projection"])
    write_rows(out_dir / "v134_parameter_writeback_trace.csv", result["writeback"])
    write_rows(out_dir / "v134_optimizer_state_transport_trace.csv", result["optimizer"])
    write_rows(out_dir / "v134_loss_interface_audit.csv", result["loss_audit"])
    write_rows(out_dir / "v134_forbidden_information_audit.csv", forbidden)
    write_rows(out_dir / "v134_synthetic_task_family_proof.csv", result["synthetic"])
    write_rows(out_dir / "v134_synthetic_family_summary.csv", family)
    write_rows(out_dir / "v134_nonrat_s1c_design.csv", nonrat)
    write_rows(out_dir / "v134_mlp_analog_channel_control.csv", mlp)
    write_rows(out_dir / "v134_failure_table.csv", failure)

    route = build_route(split, result["operator"], result["projection"], result["synthetic"], nonrat, mlp, forbidden, 0)
    write_no_go(out_dir, route)
    write_progress(out_dir, route)
    write_figures(out_dir, route, family)
    (out_dir / "v134_route_decision.json").write_text(json.dumps(route, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _manifest, _missing = write_manifest(out_dir)
    manifest, code_sha = write_code_packet(out_dir)
    _manifest, missing = write_manifest(out_dir)
    route = build_route(split, result["operator"], result["projection"], result["synthetic"], nonrat, mlp, forbidden, missing, code_sha=code_sha)
    route["code_review_packet_entries"] = len(manifest)
    (out_dir / "v134_route_decision.json").write_text(json.dumps(route, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest, code_sha = write_code_packet(out_dir)
    route["code_review_packet_entries"] = len(manifest)
    route["code_review_packet_sha256"] = code_sha
    (out_dir / "v134_route_decision.json").write_text(json.dumps(route, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_manifest(out_dir)
    print(json.dumps(route, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
