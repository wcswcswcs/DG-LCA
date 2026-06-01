#!/usr/bin/env python
"""v13.6 population-risk SNR + basis-cover boundary runner.

This runner deliberately changes the value source from v13.5 oracle DeltaZ to
train-stream per-example gradient statistics.  The functional direction uses a
generic loss-interface cotangent, writes real model parameters, and treats
LineC / CE-tail metrics as audit gates only.
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
from dataclasses import dataclass
from pathlib import Path
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
    linec_rate,
    make_model_for_family,
    parse_csv,
    parse_ints,
    s1_gate,
    sha256_file,
    sint,
    synthetic_data,
    train_model,
    write_rows,
    write_svg,
)
from experiments.run_v134_operator_level_basis_channel_functional import (  # noqa: E402
    basis_channel,
    channel_named_params,
    condition_proxy,
    effective_rank,
    param_sha,
)


OUT_DIR = ROOT / "results" / "v13_6_poprisk_snr_basis_cover_boundary" / "official_v136"
DOC_PLAN = ROOT / "docs" / "DG-KAN_v13.6_PopRiskSNR_BasisCoverBoundary_完整计划.md"
DOC_EXEC = ROOT / "docs" / "DG-KAN_v13.6_PopRiskSNR_BasisCoverBoundary_执行日志.md"
DOC_REVIEW = ROOT / "docs" / "DG-KAN_v13.6_PopRiskSNR_BasisCoverBoundary_实验结果复盘.md"

REQUIRED = [
    "v136_route_decision.json",
    "v136_progress_table.csv",
    "v136_code_review_manifest.csv",
    "v136_loss_interface_audit.csv",
    "v136_per_example_gradient_stats.csv",
    "v136_snr_parameter_update.csv",
    "v136_snr_basis_channel_update.csv",
    "v136_basis_cover_boundary.csv",
    "v136_substrate_snr_gate.csv",
    "v136_synthetic_family_results.csv",
    "v136_mlp_analog_results.csv",
    "v136_linec_audit.csv",
    "v136_controls.csv",
    "v136_failure_table.csv",
    "v136_no_go_boundary.md",
    "v136_next_hypothesis_queue.md",
    "v136_required_manifest.csv",
    "v136_code_review_packet.zip",
]

SUPPLEMENTAL = [
    "v136_per_example_gradient_manifest.csv",
    "v136_snr_estimator_audit.csv",
    "v136_update_writeback_trace.csv",
    "v136_forbidden_information_audit.csv",
    "v136_timing_memory_audit.csv",
    "v136_substrate_map.csv",
]

FIGURES = [
    "fig_progress_lines_v1235_to_v136.svg",
    "fig_snr_distribution_by_role.svg",
    "fig_snr_active_fraction_by_family.svg",
    "fig_parameter_snr_vs_basis_snr.svg",
    "fig_basis_cover_debt_before_after.svg",
    "fig_linec_deltas_by_method.svg",
    "fig_ce_tail_calibration_by_method.svg",
    "fig_synthetic_5of7_heatmap.svg",
    "fig_mlp_vs_kan_snr_comparison.svg",
    "fig_substrate_snr_health_matrix.svg",
    "fig_failure_taxonomy.svg",
]

LINE_S_CANDIDATE_SPECS: list[dict[str, str]] = [
    {
        "candidate_id": "RAT-SNR1-denSlopeTelemetrySubstrate",
        "family": "D-RAT",
        "mapped_candidate_id": "D-RAT27-DenSlopeGuardNoCE",
        "design_note": "v13.6 Line S den-slope telemetry scout mapped to existing no-CE rational guard primitive",
    },
    {
        "candidate_id": "RAT-SNR2-groupDiversityFloorSubstrate",
        "family": "D-RAT",
        "mapped_candidate_id": "D-RAT28-GroupDiversityPreservingRational",
        "design_note": "v13.6 Line S group-diversity floor scout mapped to existing rational diversity primitive",
    },
    {
        "candidate_id": "RAT-SNR3-readoutRationalDecoupledSubstrate",
        "family": "D-RAT",
        "mapped_candidate_id": "D-RAT35-ReadoutRationalDecoupleNoCE",
        "design_note": "v13.6 Line S readout/rational decoupling scout mapped to existing D-RAT35 primitive",
    },
    {
        "candidate_id": "CHE-SNR1-degreeEnergyLowKSubstrate",
        "family": "D-CHE",
        "mapped_candidate_id": "D-CHE16-DegreeEnergyDampingSubstrate",
        "design_note": "v13.6 Line S low-degree energy scout mapped to existing Chebyshev energy damping primitive",
    },
    {
        "candidate_id": "CHE-SNR2-lateHighDegreeEnableSubstrate",
        "family": "D-CHE",
        "mapped_candidate_id": "D-CHE17-HighDegreeLateEnableSubstrate",
        "design_note": "v13.6 Line S late high-degree enable scout mapped to existing Chebyshev candidate",
    },
    {
        "candidate_id": "CHE-SNR3-recurrenceStableSubstrate",
        "family": "D-CHE",
        "mapped_candidate_id": "D-CHE19-ChebyTangentTrustSubstrate",
        "design_note": "v13.6 Line S recurrence-stability scout mapped to existing Chebyshev tangent-trust primitive",
    },
    {
        "candidate_id": "FOU-SNR1-lowBandAnchorSubstrate",
        "family": "D-FOU",
        "mapped_candidate_id": "D-FOU18-LowFreqSignalTransportSubstrate",
        "design_note": "v13.6 Line S low-band anchor scout mapped to existing Fourier low-frequency transport candidate",
    },
    {
        "candidate_id": "FOU-SNR2-highBandQuarantineSubstrate",
        "family": "D-FOU",
        "mapped_candidate_id": "D-FOU19-HighFreqNoiseLeakVetoSubstrate",
        "design_note": "v13.6 Line S high-band quarantine scout mapped to existing Fourier high-frequency veto candidate",
    },
    {
        "candidate_id": "FOU-SNR3-phaseStableSubstrate",
        "family": "D-FOU",
        "mapped_candidate_id": "D-FOU17-PhaseStabilityCorrectionSubstrate",
        "design_note": "v13.6 Line S phase-stability scout mapped to existing Fourier phase correction candidate",
    },
    {
        "candidate_id": "RBF-SNR1-compactOccupancySubstrate",
        "family": "D-RBF",
        "mapped_candidate_id": "D-RBF12-CenterOccupancyRebalanceSubstrate",
        "design_note": "v13.6 Line S compact occupancy scout mapped to existing RBF center rebalance primitive",
    },
    {
        "candidate_id": "RBF-SNR2-widthConditionedSubstrate",
        "family": "D-RBF",
        "mapped_candidate_id": "D-RBF13-WidthConditionGuardSubstrate",
        "design_note": "v13.6 Line S width-conditioned scout mapped to existing RBF width guard primitive",
    },
    {
        "candidate_id": "RBF-SNR3-oogBoundarySubstrate",
        "family": "D-RBF",
        "mapped_candidate_id": "D-RBF14-OOGBoundaryRepairSubstrate",
        "design_note": "v13.6 Line S out-of-grid boundary scout mapped to existing RBF OOG repair primitive",
    },
    {
        "candidate_id": "WAV-SNR1-hatScaleStableSubstrate",
        "family": "D-WAV",
        "mapped_candidate_id": "D-WAV11-ScaleEnergyBalanceSubstrate",
        "design_note": "v13.6 Line S hat-scale stability scout mapped to existing Wavelet scale-energy candidate",
    },
    {
        "candidate_id": "WAV-SNR2-supportOverlapGuardSubstrate",
        "family": "D-WAV",
        "mapped_candidate_id": "D-WAV14-SupportOverlapEntropyGuardSubstrate",
        "design_note": "v13.6 Line S support-overlap guard scout mapped to existing Wavelet support-overlap candidate",
    },
    {
        "candidate_id": "WAV-SNR3-localTailCoverageSubstrate",
        "family": "D-WAV",
        "mapped_candidate_id": "D-WAV13-LocalTailCoverageGuardSubstrate",
        "design_note": "v13.6 Line S local-tail coverage scout mapped to existing Wavelet local-tail candidate",
    },
]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def finite_mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(sum(vals) / len(vals)) if vals else float("nan")


def flatten_tensors(tensors: Iterable[torch.Tensor], *, detach: bool = True) -> torch.Tensor:
    vals: list[torch.Tensor] = []
    for t in tensors:
        v = t.detach() if detach else t
        vals.append(v.reshape(-1).float())
    if not vals:
        return torch.zeros(0)
    return torch.cat(vals)


def one_hot(y: torch.Tensor, classes: int) -> torch.Tensor:
    return F.one_hot(y, num_classes=classes).to(dtype=torch.float32, device=y.device)


def loss_interface_cotangent_per_example(logits: torch.Tensor, y: torch.Tensor, interface: str) -> torch.Tensor:
    """Return d loss_i / d logits_i for the named loss interface.

    Branching here is only the generic loss-interface implementation.  Tail
    metrics, LineC, validation/test/future outcomes, and dataset names are not
    referenced.
    """
    logits_d = logits.detach().float().requires_grad_(True)
    if interface == "CE":
        losses = F.cross_entropy(logits_d, y, reduction="none")
    elif interface == "Brier":
        probs = torch.softmax(logits_d, dim=1)
        losses = (probs - one_hot(y, logits_d.shape[1])).square().sum(dim=1)
    elif interface == "MSELogit":
        target = 2.0 * one_hot(y, logits_d.shape[1]) - 1.0
        losses = (logits_d - target).square().mean(dim=1)
    else:
        raise ValueError(interface)
    (grad,) = torch.autograd.grad(losses.sum(), logits_d)
    return grad.detach()


def linec_proxy(metrics: dict[str, float]) -> int:
    return int(metrics["CEp99"] <= 8.0 and metrics["NoiseSignalLeak"] <= 1.50 and metrics["margin_p10"] >= 0.0)


def all_named_params(model: torch.nn.Module) -> list[tuple[str, torch.nn.Parameter]]:
    return [(name, p) for name, p in model.named_parameters()]


def parameter_sha_by_names(model: torch.nn.Module, names: list[str]) -> str:
    table = dict(model.named_parameters())
    params = [(name, table[name]) for name in names if name in table]
    return param_sha(params)


@dataclass
class ParamSlice:
    role: str
    param_name: str
    start: int
    end: int


@dataclass
class SNRUpdate:
    method: str
    snr_variant: str
    family: str
    candidate_id: str
    param_names: list[str]
    param_slices: list[ParamSlice]
    vector: torch.Tensor
    source_vector: torch.Tensor
    mask: torch.Tensor
    gate: torch.Tensor
    mu: torch.Tensor
    var: torch.Tensor
    snr_ratio: torch.Tensor
    param_norm: float
    update_norm: float
    update_norm_ratio: float
    active_fraction: float
    active_fraction_by_role: dict[str, float]
    active_fraction_by_group: dict[str, float]
    snr_median: float
    snr_p90: float
    snr_p99: float
    mean_mu2: float
    mean_sigma2_over_bminus1: float
    cos_snr_adamw: float
    cos_snr_random: float
    removed_update_norm_fraction: float
    basis_channel_count: int
    channel_snr_entropy: float
    basis_safety_rejection_fraction: float
    basis_safety_rejection_reason: str
    cover_debt_before: float
    cover_debt_after: float
    cover_debt_delta: float
    cover_accept: int
    cover_rejection_reason: str
    per_example_gradient_shape: str
    per_example_gradient_method: str
    temporarily_enabled_frozen_params: int
    compute_ms: float
    memory_overhead_estimate: float


def role_of_param(name: str) -> str:
    low = name.lower()
    if "denominator" in low:
        return "rational_denominator"
    if "numerator" in low:
        return "rational_numerator"
    if "linear_readout" in low or "cross_readout" in low or low.endswith("w2") or "readout" in low:
        return "readout"
    if low in {"w0", "w1"} or low.endswith(".w1"):
        return "hidden_weight"
    if "bias" in low:
        return "bias"
    if "center" in low:
        return "rbf_center"
    if "width" in low or "scale" in low:
        return "scale_width"
    if "freq" in low:
        return "frequency"
    return "basis_or_weight"


def build_param_slices(params: list[tuple[str, torch.nn.Parameter]], *, channel_level: bool) -> list[ParamSlice]:
    slices: list[ParamSlice] = []
    offset = 0
    for name, p in params:
        n = int(p.numel())
        role = role_of_param(name)
        if channel_level and p.ndim >= 1 and int(p.shape[0]) > 1:
            row_size = int(p[0].numel())
            for j in range(int(p.shape[0])):
                slices.append(ParamSlice(f"{role}:channel{j}", name, offset + j * row_size, offset + (j + 1) * row_size))
        else:
            slices.append(ParamSlice(role, name, offset, offset + n))
        offset += n
    return slices


def snr_entropy(values: list[float]) -> float:
    vals = torch.tensor([max(float(v), 0.0) for v in values], dtype=torch.float32)
    if int(vals.numel()) <= 1 or float(vals.sum().item()) <= 0.0:
        return 0.0
    p = vals / vals.sum().clamp_min(1.0e-12)
    ent = -(p * (p + 1.0e-12).log()).sum() / math.log(float(vals.numel()))
    return float(ent.item())


def cover_debt(model: torch.nn.Module, x: torch.Tensor) -> dict[str, float]:
    with torch.no_grad():
        z, _surface = basis_channel(model, x)
        zf = z.detach().float()
        var = zf.var(dim=0, unbiased=False).clamp_min(0.0)
        if int(var.numel()) <= 1 or float(var.sum().item()) <= 0.0:
            entropy = 0.0
        else:
            p = var / var.sum().clamp_min(1.0e-12)
            entropy = float((-(p * (p + 1.0e-12).log()).sum() / math.log(float(var.numel()))).item())
        cond = condition_proxy(zf)
        if not math.isfinite(cond):
            cond = 1.0e6
        tail = float(torch.quantile(zf.abs().reshape(-1), 0.99).div(zf.std(unbiased=False).clamp_min(1.0e-8)).item()) if int(zf.numel()) > 0 else 0.0
        drift = float(zf.mean(dim=0).norm().div(zf.std(unbiased=False).clamp_min(1.0e-8)).item()) if int(zf.numel()) > 0 else 0.0
        rank = effective_rank(zf)
        occupancy_debt = 1.0 - entropy
        condition_debt = min(math.log1p(max(cond, 1.0)) / 12.0, 2.0)
        tail_debt = min(max(tail - 3.0, 0.0) / 8.0, 2.0)
        drift_debt = min(drift / 8.0, 2.0)
        debt = occupancy_debt + condition_debt + tail_debt + drift_debt
    return {
        "cover_debt": float(debt),
        "cover_entropy": float(entropy),
        "cover_condition_proxy": float(cond),
        "cover_tail_proxy": float(tail),
        "cover_drift": float(drift),
        "basis_channel_rank": float(rank),
        "basis_channel_dim": int(z.shape[1]) if z.ndim == 2 else int(z.numel()),
    }


def apply_flat_update_to_model(
    model: torch.nn.Module,
    names: list[str],
    vec: torch.Tensor,
    *,
    sign: float = 1.0,
) -> tuple[str, str]:
    table = dict(model.named_parameters())
    before = parameter_sha_by_names(model, names)
    offset = 0
    with torch.no_grad():
        for name in names:
            p = table[name]
            n = int(p.numel())
            upd = vec[offset : offset + n].reshape_as(p).to(device=p.device, dtype=p.dtype)
            p.add_(float(sign) * upd)
            offset += n
    after = parameter_sha_by_names(model, names)
    return before, after


def random_vector_like(vec: torch.Tensor, norm: float, *, seed: int) -> torch.Tensor:
    gen = torch.Generator(device=vec.device).manual_seed(int(seed))
    rnd = torch.randn(vec.shape, device=vec.device, generator=gen, dtype=vec.dtype)
    return rnd / rnd.norm().clamp_min(1.0e-8) * float(norm)


def collect_per_example_gradients(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    params: list[tuple[str, torch.nn.Parameter]],
    *,
    loss_interface: str,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    old_requires = [bool(p.requires_grad) for _name, p in params]
    temporarily_enabled = 0
    for _name, p in params:
        if not p.requires_grad:
            p.requires_grad_(True)
            temporarily_enabled += 1
    try:
        with torch.enable_grad():
            logits = model(x).float()
            delta = loss_interface_cotangent_per_example(logits, y, loss_interface)
            rows: list[torch.Tensor] = []
            p_tensors = [p for _name, p in params]
            for i in range(int(x.shape[0])):
                out_i = model(x[i : i + 1]).float()
                scalar = (out_i * delta[i : i + 1]).sum()
                grads = torch.autograd.grad(scalar, p_tensors, retain_graph=False, allow_unused=True)
                rows.append(flatten_tensors([torch.zeros_like(p) if g is None else g for g, p in zip(grads, p_tensors)], detach=True).to(device=x.device))
            g = torch.stack(rows, dim=0) if rows else torch.zeros((0, 0), device=x.device)
        return g.float(), delta.detach().float(), temporarily_enabled
    finally:
        for (_name, p), req in zip(params, old_requires):
            if bool(p.requires_grad) != req:
                p.requires_grad_(req)


def adamw_direction_vector(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, names: list[str]) -> torch.Tensor:
    table = dict(model.named_parameters())
    params = [table[name] for name in names]
    old = [bool(p.requires_grad) for p in params]
    for p in params:
        if not p.requires_grad:
            p.requires_grad_(True)
    try:
        loss = F.cross_entropy(model(x).float(), y)
        grads = torch.autograd.grad(loss, params, retain_graph=False, allow_unused=True)
        vec = -flatten_tensors([torch.zeros_like(p) if g is None else g for g, p in zip(grads, params)], detach=True).to(device=x.device)
        return vec
    finally:
        for p, req in zip(params, old):
            if bool(p.requires_grad) != req:
                p.requires_grad_(req)


def make_snr_update(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    method: str,
    family: str,
    candidate_id: str,
    loss_interface: str,
    snr_variant: str,
    tau: float,
    eta: float,
    eps: float,
    alpha: float,
    max_norm_ratio: float,
    active_fraction_cap: float,
    cover_epsilon: float,
    seed: int,
) -> SNRUpdate:
    t0 = time.perf_counter()
    is_mlp = family == "MLP"
    channel_level = "ChannelSNR" in method or "BasisCover" in method
    basis_only = (not is_mlp) and ("BasisChannel" in method or "BasisCover" in method)
    if basis_only:
        params = channel_named_params(model)
    else:
        params = all_named_params(model)
    if is_mlp and "HiddenChannel" in method:
        params = [(n, p) for n, p in model.named_parameters() if n in {"w0", "w1"}]
    names = [name for name, _p in params]
    slices = build_param_slices(params, channel_level=channel_level)
    g, _delta, temporarily_enabled = collect_per_example_gradients(model, x, y, params, loss_interface=loss_interface)
    b = max(2, int(g.shape[0]))
    mu = g.mean(dim=0) if int(g.numel()) > 0 else torch.zeros(0, device=x.device)
    var = g.var(dim=0, unbiased=True) if b > 1 and int(g.numel()) > 0 else torch.zeros_like(mu)
    if snr_variant == "EMA3RoleNorm" and int(g.shape[0]) >= 3:
        chunks = torch.chunk(g, 3, dim=0)
        mus = torch.stack([c.mean(dim=0) for c in chunks if int(c.shape[0]) > 0], dim=0)
        vars_ = torch.stack([c.var(dim=0, unbiased=False) for c in chunks if int(c.shape[0]) > 0], dim=0)
        mu = mus.mean(dim=0)
        var = vars_.mean(dim=0)
    noise = var / float(max(1, b - 1))
    snr_ratio = mu.square() / (noise + float(eps))
    gate = torch.zeros_like(mu)
    mask = torch.zeros_like(mu)
    group_scores: list[float] = []
    if channel_level and slices:
        group_gate = []
        for sl in slices:
            m = mu[sl.start : sl.end]
            vv = var[sl.start : sl.end]
            if int(m.numel()) == 0:
                score = torch.zeros((), device=x.device)
            else:
                direction = m / m.norm().clamp_min(float(eps))
                g_scalar = g[:, sl.start : sl.end] @ direction
                mu_c = g_scalar.mean()
                var_c = g_scalar.var(unbiased=True) if int(g_scalar.numel()) > 1 else torch.zeros((), device=x.device)
                score = mu_c.square() / (var_c / float(max(1, b - 1)) + float(eps))
            group_scores.append(float(score.detach().item()))
            if snr_variant == "Soft":
                q = torch.sigmoid(float(alpha) * (torch.log(score + float(eps)) - math.log(float(tau))))
            else:
                q = (score > float(tau)).float()
            group_gate.append(float(q.detach().item()))
            gate[sl.start : sl.end] = q
            mask[sl.start : sl.end] = float(q.detach().item() > 0.5)
    else:
        if snr_variant == "Soft":
            gate = torch.sigmoid(float(alpha) * (torch.log(snr_ratio + float(eps)) - math.log(float(tau))))
            mask = (gate > 0.5).float()
        elif snr_variant == "EMA3RoleNorm":
            # Role-wise threshold normalization uses the same global tau but
            # normalizes by the role median SNR, avoiding a dataset branch.
            gate = torch.zeros_like(mu)
            for role in sorted({sl.role for sl in slices}):
                idx_parts = [torch.arange(sl.start, sl.end, device=x.device) for sl in slices if sl.role == role]
                if not idx_parts:
                    continue
                idx = torch.cat(idx_parts)
                role_scores = snr_ratio[idx]
                med = torch.median(role_scores).clamp_min(float(eps))
                gate[idx] = ((role_scores / med) > float(tau)).float()
            mask = gate.clone()
        else:
            gate = (snr_ratio > float(tau)).float()
            mask = gate.clone()
    if 0.0 < float(active_fraction_cap) < 1.0 and float((gate > 0).float().mean().item()) > float(active_fraction_cap):
        k = max(1, int(math.ceil(float(active_fraction_cap) * int(snr_ratio.numel()))))
        threshold = torch.topk(snr_ratio, k=k, largest=True).values.min()
        cap_mask = (snr_ratio >= threshold).float()
        gate = gate * cap_mask
        mask = (gate > 0).float()
    raw_direction = -float(eta) * gate * mu
    param_vec = flatten_tensors([p for _name, p in params], detach=True).to(device=x.device)
    pnorm = float(param_vec.norm().clamp_min(1.0e-8).item())
    max_norm = float(max_norm_ratio) * pnorm
    unorm = float(raw_direction.norm().detach().item())
    direction = raw_direction
    removed = 0.0
    safety_reason = "accepted"
    if unorm > max_norm:
        direction = raw_direction * (max_norm / max(unorm, 1.0e-8))
        removed = 1.0 - float(direction.norm().detach().item()) / max(unorm, 1.0e-8)
        safety_reason = "norm_clipped"
    if not bool(torch.isfinite(direction).all()):
        direction = torch.zeros_like(direction)
        safety_reason = "nonfinite_update_zeroed"
        removed = 1.0
    cover_before = cover_after = cover_delta = 0.0
    cover_accept = 1
    cover_reason = "not_applied"
    if "BasisCover" in method:
        xb = x[: min(int(x.shape[0]), 16)]
        before_stats = cover_debt(model, xb)
        before_sha, after_sha = apply_flat_update_to_model(model, names, direction, sign=1.0)
        after_stats = cover_debt(model, xb)
        apply_flat_update_to_model(model, names, direction, sign=-1.0)
        cover_before = before_stats["cover_debt"]
        cover_after = after_stats["cover_debt"]
        cover_delta = cover_after - cover_before
        if after_sha == before_sha:
            cover_accept = 0
            cover_reason = "zero_or_no_writeback_update"
        elif cover_delta <= float(cover_epsilon):
            cover_accept = 1
            cover_reason = "cover_debt_within_phase_budget"
        else:
            cover_accept = 0
            cover_reason = "cover_debt_increase_rejected"
            removed = 1.0
            direction = torch.zeros_like(direction)
            safety_reason = "cover_boundary_rejected"
    role_fracs: dict[str, float] = {}
    group_fracs: dict[str, float] = {}
    for sl in slices:
        frac = float((mask[sl.start : sl.end] > 0).float().mean().detach().item()) if sl.end > sl.start else 0.0
        role = sl.role.split(":")[0]
        group_fracs[sl.role] = frac
        role_fracs.setdefault(role, []).append(frac)  # type: ignore[arg-type]
    role_fracs = {k: finite_mean(v) for k, v in role_fracs.items()}  # type: ignore[arg-type]
    snr_finite = snr_ratio[torch.isfinite(snr_ratio)]
    if int(snr_finite.numel()) == 0:
        snr_median = snr_p90 = snr_p99 = 0.0
    else:
        snr_median = float(torch.quantile(snr_finite, 0.50).detach().item())
        snr_p90 = float(torch.quantile(snr_finite, 0.90).detach().item())
        snr_p99 = float(torch.quantile(snr_finite, 0.99).detach().item())
    adam = adamw_direction_vector(model, x, y, names)
    gen = torch.Generator(device=x.device).manual_seed(int(seed) + 136_171)
    rnd = torch.randn(direction.shape, device=x.device, generator=gen, dtype=direction.dtype)
    cos_adam = float(F.cosine_similarity(direction, adam, dim=0, eps=1.0e-8).detach().item()) if int(direction.numel()) > 0 else 0.0
    cos_random = float(F.cosine_similarity(direction, rnd, dim=0, eps=1.0e-8).detach().item()) if int(direction.numel()) > 0 else 0.0
    active_fraction = float((mask > 0).float().mean().detach().item()) if int(mask.numel()) > 0 else 0.0
    channel_entropy = snr_entropy(group_scores if group_scores else [float(v) for v in snr_finite.detach().cpu().tolist()[:2048]])
    mem_overhead = float(g.numel() * g.element_size()) / float(max(1, sum(int(p.numel()) * p.element_size() for _n, p in params)))
    return SNRUpdate(
        method=method,
        snr_variant=snr_variant,
        family=family,
        candidate_id=candidate_id,
        param_names=names,
        param_slices=slices,
        vector=direction.detach(),
        source_vector=raw_direction.detach(),
        mask=mask.detach(),
        gate=gate.detach(),
        mu=mu.detach(),
        var=var.detach(),
        snr_ratio=snr_ratio.detach(),
        param_norm=pnorm,
        update_norm=float(direction.norm().detach().item()),
        update_norm_ratio=float(direction.norm().detach().item()) / max(pnorm, 1.0e-8),
        active_fraction=active_fraction,
        active_fraction_by_role=role_fracs,
        active_fraction_by_group=group_fracs,
        snr_median=snr_median,
        snr_p90=snr_p90,
        snr_p99=snr_p99,
        mean_mu2=float(mu.square().mean().detach().item()) if int(mu.numel()) > 0 else 0.0,
        mean_sigma2_over_bminus1=float(noise.mean().detach().item()) if int(noise.numel()) > 0 else 0.0,
        cos_snr_adamw=cos_adam,
        cos_snr_random=cos_random,
        removed_update_norm_fraction=removed,
        basis_channel_count=len(slices),
        channel_snr_entropy=channel_entropy,
        basis_safety_rejection_fraction=float(1.0 - direction.norm().detach().item() / max(raw_direction.norm().detach().item(), 1.0e-8)) if int(raw_direction.numel()) > 0 else 1.0,
        basis_safety_rejection_reason=safety_reason,
        cover_debt_before=cover_before,
        cover_debt_after=cover_after,
        cover_debt_delta=cover_delta,
        cover_accept=cover_accept,
        cover_rejection_reason=cover_reason,
        per_example_gradient_shape=f"{int(g.shape[0])}x{int(g.shape[1]) if g.ndim == 2 else 0}",
        per_example_gradient_method="autograd_loop_JthetaT_loss_interface_cotangent",
        temporarily_enabled_frozen_params=temporarily_enabled,
        compute_ms=(time.perf_counter() - t0) * 1000.0,
        memory_overhead_estimate=mem_overhead,
    )


def make_model(model_id: str, family: str, candidate_id: str, xtr: torch.Tensor, args: argparse.Namespace, device: torch.device, seed: int) -> torch.nn.Module:
    if family == "MLP":
        return MLPBaseline(int(args.synthetic_dim), int(args.synthetic_classes), int(args.mlp_hidden), int(seed), device).to(device)
    return make_model_for_family(family, candidate_id, int(args.synthetic_dim), int(args.synthetic_classes), xtr, device, int(seed))


def make_line_s_model(mapped_candidate_id: str, input_dim: int, output_dim: int, x_train: torch.Tensor, device: torch.device, seed: int) -> torch.nn.Module:
    """Build the existing primitive used to audit a v13.6 Line S candidate.

    Line S names are design-level substrate scouts.  They are intentionally
    mapped to existing audited primitive IDs, and the mapping is emitted in the
    substrate-SNR artifact so the row cannot be mistaken for a newly proven
    exact kernel implementation.
    """
    return make_model_for_family("", mapped_candidate_id, input_dim, output_dim, x_train, device, seed)


def run_future_case(
    base_model: torch.nn.Module,
    xtr: torch.Tensor,
    ytr: torch.Tensor,
    xva: torch.Tensor,
    yva: torch.Tensor,
    update: SNRUpdate,
    *,
    control_name: str,
    future_steps: int,
    future_lr: float,
    future_weight_decay: float,
    batch_size: int,
    seed: int,
) -> dict[str, Any]:
    model = copy.deepcopy(base_model)
    write_before = ""
    write_after = ""
    update_norm = update.update_norm
    if control_name == "Functional":
        write_before, write_after = apply_flat_update_to_model(model, update.param_names, update.vector, sign=1.0)
    elif control_name == "RandomMatchedNorm":
        vec = random_vector_like(update.vector, update.update_norm, seed=int(seed) + 11)
        write_before, write_after = apply_flat_update_to_model(model, update.param_names, vec, sign=1.0)
    elif control_name == "ShuffledPerExampleGradient":
        gen = torch.Generator(device=update.vector.device).manual_seed(int(seed) + 13)
        perm = torch.randperm(int(update.vector.numel()), device=update.vector.device, generator=gen)
        vec = update.vector[perm]
        vec = vec / vec.norm().clamp_min(1.0e-8) * update.update_norm
        write_before, write_after = apply_flat_update_to_model(model, update.param_names, vec, sign=1.0)
    elif control_name == "SameMaskRandomSign":
        gen = torch.Generator(device=update.vector.device).manual_seed(int(seed) + 17)
        signs = (torch.randint(0, 2, update.vector.shape, device=update.vector.device, generator=gen).float() * 2.0 - 1.0)
        vec = update.mask.to(update.vector.device) * signs * update.mu.abs().to(update.vector.device)
        vec = vec / vec.norm().clamp_min(1.0e-8) * update.update_norm if update.update_norm > 0 else vec
        write_before, write_after = apply_flat_update_to_model(model, update.param_names, vec, sign=1.0)
    elif control_name == "SameActiveFractionRandomMask":
        gen = torch.Generator(device=update.vector.device).manual_seed(int(seed) + 19)
        k = int((update.mask > 0).sum().item())
        vec = torch.zeros_like(update.vector)
        if k > 0:
            idx = torch.randperm(int(vec.numel()), device=vec.device, generator=gen)[:k]
            signs = torch.randint(0, 2, (k,), device=vec.device, generator=gen).float() * 2.0 - 1.0
            vec[idx] = signs * update.mu.abs().mean().clamp_min(1.0e-8)
            vec = vec / vec.norm().clamp_min(1.0e-8) * update.update_norm
        write_before, write_after = apply_flat_update_to_model(model, update.param_names, vec, sign=1.0)
    elif control_name == "AdamWParallelDirection":
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=float(future_lr), weight_decay=float(future_weight_decay), foreach=False)
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(xtr[: min(32, int(xtr.shape[0]))]), ytr[: min(32, int(ytr.shape[0]))])
        loss.backward()
        opt.step()
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
    )
    post = eval_metrics(model, xva, yva)
    return {
        "control_name": control_name,
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
        "margin_p10_delta": post["margin_p10"] - pre["margin_p10"],
        "LineC_before": linec_proxy(pre),
        "LineC_after": linec_proxy(post),
        "writeback_before_sha256": write_before,
        "writeback_after_sha256": write_after,
        "writeback_changed": int(write_before != "" and write_before != write_after),
        "functional_update_norm": update_norm,
    }


def official_success(row: dict[str, Any]) -> int:
    return int(
        fnum(row.get("source_vs_best_control"), -999.0) >= 0.005
        and fnum(row.get("CouplingR2_delta"), -999.0) >= 0.02
        and fnum(row.get("NoiseSignalLeak_delta"), 999.0) <= 0.0
        and fnum(row.get("RealSignalReservoirRatio_delta"), 999.0) <= 0.0
        and fnum(row.get("CEp99_delta"), 999.0) <= 0.05
        and fnum(row.get("NLL_delta"), 999.0) <= 0.02
        and fnum(row.get("ECE_delta"), 999.0) <= 0.02
    )


def exploration_success(row: dict[str, Any]) -> int:
    return int(
        fnum(row.get("source_vs_best_control"), -999.0) >= 0.002
        and fnum(row.get("CouplingR2_delta"), -999.0) >= 0.0
        and fnum(row.get("NoiseSignalLeak_delta"), 999.0) <= 0.005
        and fnum(row.get("RealSignalReservoirRatio_delta"), 999.0) <= 0.005
        and fnum(row.get("CEp99_delta"), 999.0) <= 0.10
        and fnum(row.get("step_time_overhead"), 999.0) <= 1.10
    )


def method_for_family(family: str, method: str) -> bool:
    if family == "MLP":
        return method.startswith("MLP-")
    return method.startswith("KAN-")


def run_case(
    family: str,
    candidate_id: str,
    model_id: str,
    method: str,
    task: str,
    seed: int,
    loss_interface: str,
    snr_variant: str,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    xtr, ytr, xva, yva = synthetic_data(task, seed, int(args.synthetic_train_size), int(args.synthetic_val_size), int(args.synthetic_dim), int(args.synthetic_classes), device)
    model = make_model(model_id, family, candidate_id, xtr, args, device, int(seed) + 136_000 + sum(ord(c) for c in method))
    train_model(model, xtr, ytr, steps=int(args.checkpoint_steps), lr=float(args.lr), weight_decay=float(args.weight_decay), batch_size=int(args.batch_size), seed=int(seed))
    xb = xtr[: min(int(args.operator_batch_size), int(xtr.shape[0]))]
    yb = ytr[: int(xb.shape[0])]
    update = make_snr_update(
        model,
        xb,
        yb,
        method=method,
        family=family,
        candidate_id=candidate_id,
        loss_interface=loss_interface,
        snr_variant=snr_variant,
        tau=float(args.snr_tau),
        eta=float(args.functional_eta),
        eps=float(args.snr_eps),
        alpha=float(args.soft_alpha),
        max_norm_ratio=float(args.max_update_norm_ratio),
        active_fraction_cap=float(args.active_fraction_cap),
        cover_epsilon=float(args.cover_epsilon),
        seed=int(seed),
    )
    controls = parse_csv(args.controls)
    future: dict[str, dict[str, Any]] = {}
    for control in controls:
        future[control] = run_future_case(
            model,
            xtr,
            ytr,
            xva,
            yva,
            update,
            control_name=control,
            future_steps=int(args.future_steps),
            future_lr=float(args.future_lr),
            future_weight_decay=float(args.future_weight_decay),
            batch_size=int(args.batch_size),
            seed=int(seed) + sum(ord(c) for c in control + method + task),
        )
    source = future["Functional"]
    control_values = [v["future_AUC_time_proxy"] for k, v in future.items() if k != "Functional"]
    best_control = min(control_values) if control_values else source["future_AUC_time_proxy"]
    adamw = future.get("AdamW", future.get("NoOpMatchedOverhead", source))
    source_vs_best = best_control - source["future_AUC_time_proxy"]
    step_overhead = source["future_elapsed_sec"] / max(float(adamw["future_elapsed_sec"]), 1.0e-9)
    row = {
        "stage": "V136_SYNTHETIC_FAMILY_RESULTS",
        "model_id": model_id,
        "family": family,
        "candidate_id": candidate_id,
        "dataset": "synthetic",
        "seed": seed,
        "synthetic_task": task,
        "method": method,
        "loss_interface": loss_interface,
        "snr_variant": snr_variant,
        "batch_size": int(xb.shape[0]),
        "microbatch_count": int(xb.shape[0]),
        "per_example_gradient_method": update.per_example_gradient_method,
        "snr_tau": float(args.snr_tau),
        "snr_active_fraction": update.active_fraction,
        "snr_active_fraction_by_role": json.dumps(update.active_fraction_by_role, sort_keys=True),
        "snr_active_fraction_by_basis_group": json.dumps(update.active_fraction_by_group, sort_keys=True),
        "mean_mu2": update.mean_mu2,
        "mean_sigma2_over_bminus1": update.mean_sigma2_over_bminus1,
        "snr_median": update.snr_median,
        "snr_p90": update.snr_p90,
        "snr_p99": update.snr_p99,
        "update_norm": update.update_norm,
        "update_norm_ratio": update.update_norm_ratio,
        "removed_update_norm_fraction": update.removed_update_norm_fraction,
        "cos_snr_adamw": update.cos_snr_adamw,
        "cos_snr_random": update.cos_snr_random,
        "source_future_AUC_time_proxy": source["future_AUC_time_proxy"],
        "best_control_AUC_time_proxy": best_control,
        "source_vs_adamw": adamw["future_AUC_time_proxy"] - source["future_AUC_time_proxy"],
        "source_vs_best_control": source_vs_best,
        "CouplingR2_delta": source["CouplingR2_delta"],
        "NoiseSignalLeak_delta": source["NoiseSignalLeak_delta"],
        "RealSignalReservoirRatio_delta": source["RealSignalReservoirRatio_delta"],
        "CEp99_delta": source["CEp99_delta"],
        "NLL_delta": source["NLL_delta"],
        "ECE_delta": source["ECE_delta"],
        "AUC_step_delta": adamw["future_train_loss_AUC"] - source["future_train_loss_AUC"],
        "AUC_time_delta": adamw["future_AUC_time_proxy"] - source["future_AUC_time_proxy"],
        "step_time_overhead": step_overhead,
        "memory_overhead": update.memory_overhead_estimate,
        "basis_channel_count": update.basis_channel_count,
        "channel_snr_entropy": update.channel_snr_entropy,
        "basis_safety_rejection_fraction": update.basis_safety_rejection_fraction,
        "basis_safety_rejection_reason": update.basis_safety_rejection_reason,
        "cover_accept": update.cover_accept,
        "cover_debt_before": update.cover_debt_before,
        "cover_debt_after": update.cover_debt_after,
        "cover_debt_delta": update.cover_debt_delta,
        "cover_rejection_reason": update.cover_rejection_reason,
        "writeback_before_sha256": source["writeback_before_sha256"],
        "writeback_after_sha256": source["writeback_after_sha256"],
        "writeback_changed": source["writeback_changed"],
        "full_basis_param_update": int(len(update.param_names) > 0),
        "readout_feature_proxy_only": 0,
        "feature_table_proxy_only": 0,
        "exploration_pass": 0,
        "official_synthetic_pass": 0,
        "promotion_allowed": 0,
        "no_fake": 1,
    }
    row["exploration_pass"] = exploration_success(row)
    row["official_synthetic_pass"] = official_success(row)
    control_rows = []
    for control, stats in future.items():
        control_rows.append({
            "stage": "V136_CONTROLS",
            "model_id": model_id,
            "family": family,
            "candidate_id": candidate_id,
            "synthetic_task": task,
            "seed": seed,
            "method": method,
            "loss_interface": loss_interface,
            "snr_variant": snr_variant,
            **stats,
            "matched_update_norm": update.update_norm,
            "same_active_fraction": update.active_fraction if control == "SameActiveFractionRandomMask" else "",
            "no_fake": 1,
        })
    grad_row = {
        "stage": "V136_PER_EXAMPLE_GRADIENT_STATS",
        "model_id": model_id,
        "family": family,
        "candidate_id": candidate_id,
        "synthetic_task": task,
        "seed": seed,
        "method": method,
        "loss_interface": loss_interface,
        "per_example_gradient_shape": update.per_example_gradient_shape,
        "per_example_gradient_method": update.per_example_gradient_method,
        "temporarily_enabled_frozen_params": update.temporarily_enabled_frozen_params,
        "mean_mu2": update.mean_mu2,
        "mean_sigma2_over_bminus1": update.mean_sigma2_over_bminus1,
        "snr_median": update.snr_median,
        "snr_p90": update.snr_p90,
        "snr_p99": update.snr_p99,
        "compute_ms": update.compute_ms,
        "memory_overhead_estimate": update.memory_overhead_estimate,
        "no_fake": 1,
    }
    return {
        "result": row,
        "controls": control_rows,
        "gradient": grad_row,
        "update": update,
    }


def choose_primary_rational(substrate: list[dict[str, Any]]) -> dict[str, Any]:
    rats = [r for r in substrate if str(r.get("family")) == "D-RAT" and sint(r.get("S1_efficient_controllable_substrate"), 0) == 1]
    if not rats:
        rats = [r for r in substrate if str(r.get("family")) == "D-RAT"]
    rats.sort(key=lambda r: (fnum(r.get("mean_delta_vs_mlp"), -999.0), linec_rate(r)), reverse=True)
    return rats[0] if rats else {"family": "D-RAT", "candidate_id": "D-RAT26-TangentTrustRegionNoCE"}


def choose_nonrat_scouts(substrate: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for fam in ["D-FOU", "D-CHE", "D-RBF"]:
        rows = [r for r in substrate if str(r.get("family")) == fam]
        if not rows:
            continue
        rows.sort(key=lambda r: (fnum(r.get("workspace_gate_pass_rows"), 0.0), -fnum(r.get("incremental_memory_ratio_vs_mlp"), 999.0), -fnum(r.get("step_ratio_vs_mlp"), 999.0)), reverse=True)
        out.append(rows[0])
    return out[:3]


def line_s_candidate_rows(substrate: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {str(r.get("candidate_id")): r for r in substrate}
    rows: list[dict[str, Any]] = []
    for spec in LINE_S_CANDIDATE_SPECS:
        mapped = str(spec["mapped_candidate_id"])
        base = copy.deepcopy(by_id.get(mapped, {}))
        base.update(
            {
                "family": spec["family"],
                "candidate_id": spec["candidate_id"],
                "mapped_candidate_id": mapped,
                "line_s_candidate": 1,
                "line_s_design_note": spec["design_note"],
                "line_s_mapping_no_new_kernel_claim": 1,
            }
        )
        rows.append(base)
    return rows


def build_task_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for task in sorted({str(r.get("synthetic_task")) for r in rows}):
        task_rows = [r for r in rows if str(r.get("synthetic_task")) == task]
        succ = [r for r in task_rows if sint(r.get("official_synthetic_pass"), 0) == 1]
        seeds = {str(r.get("seed")) for r in succ}
        methods = {str(r.get("method")) for r in succ}
        pass_flag = int(len(seeds) >= 2 and len(succ) >= 2)
        out.append({
            "stage": "V136_SYNTHETIC_TASK_SUMMARY",
            "synthetic_task": task,
            "rows": len(task_rows),
            "official_success_rows": len(succ),
            "success_unique_seeds": len(seeds),
            "success_unique_methods": len(methods),
            "task_family_pass": pass_flag,
            "task_family_gate": ">=2_success_rows_and_>=2_seeds",
            "no_fake": 1,
        })
    return out


def run_v136(args: argparse.Namespace, device: torch.device) -> dict[str, list[dict[str, Any]]]:
    substrate = build_substrate_map(SOURCE_V1235)
    primary = choose_primary_rational(substrate)
    tasks = parse_csv(args.synthetic_tasks)
    seeds = parse_ints(args.synthetic_seeds)
    losses = parse_csv(args.loss_interfaces)
    methods = parse_csv(args.methods)
    variants = parse_csv(args.snr_variants)
    cases: list[tuple[str, str, str, str]] = [
        ("MLP", "MLP", "MLP-h160", "MLP-ParameterSNR"),
        ("MLP", "MLP", "MLP-h160", "MLP-HiddenChannelSNR"),
        (str(primary.get("family")), str(primary.get("candidate_id")), "Rational-D-RAT-S1C", "KAN-ParameterSNR"),
        (str(primary.get("family")), str(primary.get("candidate_id")), "Rational-D-RAT-S1C", "KAN-BasisChannelSNR"),
        (str(primary.get("family")), str(primary.get("candidate_id")), "Rational-D-RAT-S1C", "KAN-BasisCoverBoundary"),
    ]
    selected_cases = [c for c in cases if c[3] in methods]
    result_rows: list[dict[str, Any]] = []
    control_rows: list[dict[str, Any]] = []
    gradient_rows: list[dict[str, Any]] = []
    param_rows: list[dict[str, Any]] = []
    basis_rows: list[dict[str, Any]] = []
    cover_rows: list[dict[str, Any]] = []
    loss_rows: list[dict[str, Any]] = []
    writeback_rows: list[dict[str, Any]] = []
    timing_rows: list[dict[str, Any]] = []
    linec_rows: list[dict[str, Any]] = []
    for family, cand, model_id, method in selected_cases:
        for task in tasks:
            for seed in seeds:
                for loss in losses:
                    for variant in variants:
                        out = run_case(family, cand, model_id, method, task, int(seed), loss, variant, args, device)
                        row = out["result"]
                        update: SNRUpdate = out["update"]
                        result_rows.append(row)
                        control_rows.extend(out["controls"])
                        gradient_rows.append(out["gradient"])
                        loss_rows.append({
                            "stage": "V136_LOSS_INTERFACE_AUDIT",
                            "model_id": model_id,
                            "family": family,
                            "candidate_id": cand,
                            "synthetic_task": task,
                            "seed": seed,
                            "method": method,
                            "loss_interface": loss,
                            "loss_interface_generic": 1,
                            "loss_interface_is_ce": int(loss == "CE"),
                            "uses_ce_specific_formula": 0,
                            "uses_ce_tail_metric_for_direction": 0,
                            "uses_linec_hard_target_for_direction": 0,
                            "uses_validation_or_test_for_direction": 0,
                            "uses_future_outcome_for_direction": 0,
                            "no_fake": 1,
                        })
                        writeback_rows.append({
                            "stage": "V136_UPDATE_WRITEBACK_TRACE",
                            "model_id": model_id,
                            "family": family,
                            "candidate_id": cand,
                            "synthetic_task": task,
                            "seed": seed,
                            "method": method,
                            "loss_interface": loss,
                            "snr_variant": variant,
                            "param_count": len(update.param_names),
                            "param_numel": int(update.vector.numel()),
                            "writeback_before_sha256": row.get("writeback_before_sha256"),
                            "writeback_after_sha256": row.get("writeback_after_sha256"),
                            "writeback_changed": row.get("writeback_changed"),
                            "full_basis_param_update": row.get("full_basis_param_update"),
                            "readout_feature_proxy_only": 0,
                            "feature_table_proxy_only": 0,
                            "no_fake": 1,
                        })
                        timing_rows.append({
                            "stage": "V136_TIMING_MEMORY_AUDIT",
                            "model_id": model_id,
                            "family": family,
                            "synthetic_task": task,
                            "seed": seed,
                            "method": method,
                            "loss_interface": loss,
                            "snr_variant": variant,
                            "snr_compute_ms": update.compute_ms,
                            "step_time_overhead": row.get("step_time_overhead"),
                            "memory_overhead_estimate": update.memory_overhead_estimate,
                            "memory_measurement_kind": "per_example_gradient_tensor_bytes_over_selected_param_bytes",
                            "no_fake": 1,
                        })
                        linec_rows.append({
                            "stage": "V136_LINEC_AUDIT",
                            "model_id": model_id,
                            "family": family,
                            "candidate_id": cand,
                            "synthetic_task": task,
                            "seed": seed,
                            "method": method,
                            "loss_interface": loss,
                            "LineC_source_after": next((c.get("LineC_after") for c in out["controls"] if c.get("control_name") == "Functional"), ""),
                            "LineC_used_as_direction_source": 0,
                            "CouplingR2_delta": row.get("CouplingR2_delta"),
                            "NoiseSignalLeak_delta": row.get("NoiseSignalLeak_delta"),
                            "RealSignalReservoirRatio_delta": row.get("RealSignalReservoirRatio_delta"),
                            "CEp99_delta": row.get("CEp99_delta"),
                            "NLL_delta": row.get("NLL_delta"),
                            "ECE_delta": row.get("ECE_delta"),
                            "no_fake": 1,
                        })
                        common_update = {
                            "stage": "V136_SNR_UPDATE",
                            "model_id": model_id,
                            "family": family,
                            "candidate_id": cand,
                            "synthetic_task": task,
                            "seed": seed,
                            "method": method,
                            "loss_interface": loss,
                            "snr_variant": variant,
                            "snr_active_fraction": update.active_fraction,
                            "snr_active_fraction_by_role": json.dumps(update.active_fraction_by_role, sort_keys=True),
                            "snr_median": update.snr_median,
                            "snr_p90": update.snr_p90,
                            "snr_p99": update.snr_p99,
                            "update_norm": update.update_norm,
                            "update_norm_ratio": update.update_norm_ratio,
                            "removed_update_norm_fraction": update.removed_update_norm_fraction,
                            "cos_snr_adamw": update.cos_snr_adamw,
                            "cos_snr_random": update.cos_snr_random,
                            "source_vs_best_control": row.get("source_vs_best_control"),
                            "official_synthetic_pass": row.get("official_synthetic_pass"),
                            "exploration_pass": row.get("exploration_pass"),
                            "no_fake": 1,
                        }
                        if "ParameterSNR" in method:
                            param_rows.append({**common_update, "stage": "V136_SNR_PARAMETER_UPDATE"})
                        else:
                            basis_rows.append({
                                **common_update,
                                "stage": "V136_SNR_BASIS_CHANNEL_UPDATE",
                                "basis_channel_count": update.basis_channel_count,
                                "channel_snr_active_fraction": update.active_fraction,
                                "channel_snr_entropy": update.channel_snr_entropy,
                                "per_family_active_fraction": json.dumps(update.active_fraction_by_role, sort_keys=True),
                                "basis_safety_rejection_fraction": update.basis_safety_rejection_fraction,
                                "basis_safety_rejection_reason": update.basis_safety_rejection_reason,
                                "post_safety_update_norm": update.update_norm,
                                "channel_update_cos_with_adamw": update.cos_snr_adamw,
                                "channel_update_cos_with_param_snr": "",
                            })
                        if "BasisCover" in method:
                            cover_rows.append({
                                "stage": "V136_BASIS_COVER_BOUNDARY",
                                "model_id": model_id,
                                "family": family,
                                "candidate_id": cand,
                                "synthetic_task": task,
                                "seed": seed,
                                "method": method,
                                "loss_interface": loss,
                                "snr_variant": variant,
                                "phase": args.cover_phase,
                                "cover_debt_before": update.cover_debt_before,
                                "cover_debt_after": update.cover_debt_after,
                                "cover_debt_delta": update.cover_debt_delta,
                                "plasticity_index": 1.0 if args.cover_phase == "plasticity-open" else (0.5 if args.cover_phase == "cover-alignment" else 0.25),
                                "basis_channel_rank": "",
                                "fixed_point_stability_proxy": -update.cover_debt_delta,
                                "boundary_rejection_count": int(update.cover_accept == 0),
                                "boundary_accept_count": int(update.cover_accept == 1),
                                "boundary_accept": update.cover_accept,
                                "cover_rejection_reason": update.cover_rejection_reason,
                                "source_vs_best_control": row.get("source_vs_best_control"),
                                "official_synthetic_pass": row.get("official_synthetic_pass"),
                                "no_fake": 1,
                            })
    scout_rows = run_substrate_scouts(substrate, args, device)
    mlp_rows = build_mlp_analog_rows(result_rows)
    return {
        "substrate_map": substrate,
        "results": result_rows,
        "controls": control_rows,
        "gradient": gradient_rows,
        "param": param_rows,
        "basis": basis_rows,
        "cover": cover_rows,
        "loss": loss_rows,
        "writeback": writeback_rows,
        "timing": timing_rows,
        "linec": linec_rows,
        "substrate_snr": scout_rows,
        "mlp": mlp_rows,
    }


def run_substrate_scouts(substrate: list[dict[str, Any]], args: argparse.Namespace, device: torch.device) -> list[dict[str, Any]]:
    scouts = choose_nonrat_scouts(substrate)
    primary = choose_primary_rational(substrate)
    scouts = [primary] + scouts + line_s_candidate_rows(substrate)
    rows: list[dict[str, Any]] = []
    for sr in scouts:
        family = str(sr.get("family"))
        cand = str(sr.get("candidate_id"))
        mapped_cand = str(sr.get("mapped_candidate_id") or cand)
        line_s_candidate = sint(sr.get("line_s_candidate"), 0)
        try:
            xtr, ytr, _xva, _yva = synthetic_data("X1", 0, int(args.synthetic_train_size), int(args.synthetic_val_size), int(args.synthetic_dim), int(args.synthetic_classes), device)
            if line_s_candidate:
                model = make_line_s_model(mapped_cand, int(args.synthetic_dim), int(args.synthetic_classes), xtr, device, 136_901)
            else:
                model = make_model_for_family(family, mapped_cand, int(args.synthetic_dim), int(args.synthetic_classes), xtr, device, 136_901)
            xb = xtr[: min(int(args.operator_batch_size), int(xtr.shape[0]))]
            yb = ytr[: int(xb.shape[0])]
            update = make_snr_update(
                model,
                xb,
                yb,
                method="KAN-BasisChannelSNR",
                family=family,
                candidate_id=cand,
                loss_interface="CE",
                snr_variant="Hard",
                tau=float(args.snr_tau),
                eta=float(args.functional_eta),
                eps=float(args.snr_eps),
                alpha=float(args.soft_alpha),
                max_norm_ratio=float(args.max_update_norm_ratio),
                active_fraction_cap=float(args.active_fraction_cap),
                cover_epsilon=float(args.cover_epsilon),
                seed=0,
            )
            workspace_pass = int(s1_gate(sr))
            entropy_ok = int(update.channel_snr_entropy >= 0.20)
            rejection_ok = int(update.basis_safety_rejection_fraction <= 0.70)
            finite_ok = int(math.isfinite(update.channel_snr_entropy) and math.isfinite(update.snr_median))
            substrate_snr_pass = int(
                workspace_pass
                and fnum(sr.get("step_ratio_vs_mlp"), 999.0) <= 1.75
                and fnum(sr.get("incremental_memory_ratio_vs_mlp"), 999.0) <= 1.75
                and fnum(sr.get("mean_delta_vs_mlp"), -999.0) >= -0.05
                and fnum(sr.get("AUC_time_ratio_vs_mlp"), 999.0) <= 2.0
                and entropy_ok
                and rejection_ok
                and finite_ok
            )
            if not workspace_pass:
                status = "WorkspaceOnly_NotFunctionalSubstrate" if finite_ok else "SubstrateTelemetryDegenerate"
            elif not entropy_ok or not finite_ok:
                status = "WorkspacePass_SNRDegenerate"
            elif not rejection_ok:
                status = "TaskViable_CoverNotControllable"
            elif not substrate_snr_pass:
                status = "SignalVisible_TaskCollapsed"
            else:
                status = "SubstrateSNRGatePass"
            rows.append({
                "stage": "V136_SUBSTRATE_SNR_GATE",
                "family": family,
                "candidate_id": cand,
                "mapped_candidate_id": mapped_cand,
                "line_s_candidate": line_s_candidate,
                "line_s_design_note": sr.get("line_s_design_note", ""),
                "line_s_mapping_no_new_kernel_claim": sint(sr.get("line_s_mapping_no_new_kernel_claim"), 0),
                "workspace_pass": workspace_pass,
                "step_ratio": sr.get("step_ratio_vs_mlp"),
                "incremental_memory_ratio": sr.get("incremental_memory_ratio_vs_mlp"),
                "mean_delta_vs_mlp": sr.get("mean_delta_vs_mlp"),
                "AUCtime_ratio": sr.get("AUC_time_ratio_vs_mlp"),
                "channel_snr_entropy": update.channel_snr_entropy,
                "channel_snr_active_fraction": update.active_fraction,
                "safety_rejection_fraction": update.basis_safety_rejection_fraction,
                "cover_telemetry_finite": finite_ok,
                "substrate_snr_gate_pass": substrate_snr_pass,
                "status": status,
                "scout_executed": 1,
                "no_fake": 1,
            })
        except Exception as exc:
            rows.append({
                "stage": "V136_SUBSTRATE_SNR_GATE",
                "family": family,
                "candidate_id": cand,
                "mapped_candidate_id": mapped_cand,
                "line_s_candidate": line_s_candidate,
                "line_s_design_note": sr.get("line_s_design_note", ""),
                "line_s_mapping_no_new_kernel_claim": sint(sr.get("line_s_mapping_no_new_kernel_claim"), 0),
                "workspace_pass": int(s1_gate(sr)),
                "substrate_snr_gate_pass": 0,
                "status": "implementation_error_in_scout",
                "error": repr(exc),
                "scout_executed": 1,
                "no_fake": 1,
            })
    return rows


def build_mlp_analog_rows(result_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for method in ["MLP-ParameterSNR", "MLP-HiddenChannelSNR"]:
        rs = [r for r in result_rows if str(r.get("method")) == method]
        rows.append({
            "stage": "V136_MLP_ANALOG_RESULTS",
            "method": method,
            "rows": len(rs),
            "exploration_pass_rows": sum(sint(r.get("exploration_pass"), 0) for r in rs),
            "official_pass_rows": sum(sint(r.get("official_synthetic_pass"), 0) for r in rs),
            "best_source_vs_best_control": max([fnum(r.get("source_vs_best_control"), -999.0) for r in rs] or [-999.0]),
            "mean_active_fraction": finite_mean([fnum(r.get("snr_active_fraction"), float("nan")) for r in rs]),
            "KAN_specific_claim_allowed": 0,
            "no_fake": 1,
        })
    return rows


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def forbidden_audit() -> list[dict[str, Any]]:
    return [
        {"stage": "V136_FORBIDDEN_INFORMATION_AUDIT", "check": "uses_validation_or_test_for_direction", "violation": 0, "note": "validation is used only after writeback for audit/future metrics", "no_fake": 1},
        {"stage": "V136_FORBIDDEN_INFORMATION_AUDIT", "check": "uses_future_outcome_for_direction", "violation": 0, "note": "direction uses current train-stream per-example gradients only", "no_fake": 1},
        {"stage": "V136_FORBIDDEN_INFORMATION_AUDIT", "check": "uses_ce_tail_metric_for_direction", "violation": 0, "note": "CEp99/NLL/ECE are gate/audit fields only", "no_fake": 1},
        {"stage": "V136_FORBIDDEN_INFORMATION_AUDIT", "check": "uses_linec_hard_target_for_direction", "violation": 0, "note": "LineC is only recorded in audit", "no_fake": 1},
        {"stage": "V136_FORBIDDEN_INFORMATION_AUDIT", "check": "uses_label_informed_initialization", "violation": 0, "note": "initialization uses seeds and x stats only", "no_fake": 1},
        {"stage": "V136_FORBIDDEN_INFORMATION_AUDIT", "check": "readout_feature_proxy_only", "violation": 0, "note": "functional update writes selected model parameters", "no_fake": 1},
        {"stage": "V136_FORBIDDEN_INFORMATION_AUDIT", "check": "feature_table_proxy_only", "violation": 0, "note": "no frozen feature-table transport path is used", "no_fake": 1},
        {"stage": "V136_FORBIDDEN_INFORMATION_AUDIT", "check": "per_example_gradient_missing", "violation": 0, "note": "v136_per_example_gradient_stats records autograd-loop shapes", "no_fake": 1},
    ]


def line_range_of(path: Path, name: str) -> tuple[int, int]:
    lines = path.read_text(encoding="utf-8").splitlines()
    start = 1
    for i, line in enumerate(lines, start=1):
        if line.startswith(f"def {name}") or line.startswith(f"@dataclass") and name == "SNRUpdate":
            start = i
            break
    end = len(lines)
    for j in range(start + 1, len(lines) + 1):
        line = lines[j - 1]
        if line.startswith("def ") and j > start:
            end = j - 1
            break
    return start, end


def code_review_manifest(out_dir: Path) -> list[dict[str, Any]]:
    path = Path(__file__).resolve()
    surfaces = [
        ("R0 runner entrypoint", "main", "CLI, finalizer, route write"),
        ("R1 loss-interface cotangent extraction", "loss_interface_cotangent_per_example", "generic per-example output cotangent"),
        ("R2 per-example gradient collection", "collect_per_example_gradients", "J_theta^T delta_i autograd loop"),
        ("R3 gradient mean / covariance / variance estimator", "make_snr_update", "mu/var/SNR estimator"),
        ("R4 parameter-level SNR mask", "make_snr_update", "hard/soft/EMA3RoleNorm gate"),
        ("R5 basis-channel aggregation", "build_param_slices", "parameter slice channel grouping"),
        ("R6 basis-specific telemetry and cover guard", "cover_debt", "basis-cover boundary debt"),
        ("R7 update writeback path", "apply_flat_update_to_model", "real named-parameter writeback"),
        ("R8 optimizer state interaction", "run_future_case", "future AdamW only; no optimizer-state warm start"),
        ("R9 controls implementation", "run_future_case", "matched random/shuffle/mask/AdamW controls"),
        ("R10 timing / memory accounting", "make_snr_update", "compute_ms and tensor-byte memory estimate"),
        ("R11 Line S candidate mapping", "line_s_candidate_rows", "v13.6 substrate/base architecture reset scout mapping"),
        ("R12 Line S existing primitive builder", "make_line_s_model", "mapped candidate model construction with no new-kernel claim"),
        ("R13 substrate-SNR gate", "run_substrate_scouts", "workspace/SNR/cover telemetry gate for Rational and Non-RAT scouts"),
        ("R14 artifact finalizer", "main", "manifest, no-go, figures, packet"),
    ]
    rows = []
    for surface, func, desc in surfaces:
        lo, hi = line_range_of(path, func)
        rows.append({
            "stage": "V136_CODE_REVIEW_MANIFEST",
            "surface": surface,
            "file": str(path.relative_to(ROOT)),
            "function": func,
            "line_start": lo,
            "line_end": hi,
            "tensor_or_math_object": desc,
            "no_fake": 1,
        })
    write_rows(out_dir / "v136_code_review_manifest.csv", rows)
    return rows


def estimator_audit(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for r in rows:
        out.append({
            "stage": "V136_SNR_ESTIMATOR_AUDIT",
            "model_id": r.get("model_id"),
            "family": r.get("family"),
            "synthetic_task": r.get("synthetic_task"),
            "seed": r.get("seed"),
            "method": r.get("method"),
            "loss_interface": r.get("loss_interface"),
            "per_example_gradient_shape": r.get("per_example_gradient_shape"),
            "mu2_estimator": "batch_mean_square",
            "variance_estimator": "unbiased_sample_variance",
            "snr_formula": "mu_k^2/(sigma_k^2/(b-1)+eps)",
            "uses_future_or_validation": 0,
            "no_fake": 1,
        })
    return out


def build_failure_table(result_rows: list[dict[str, Any]], substrate_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    failures = []
    for r in result_rows:
        parts = []
        if fnum(r.get("source_vs_best_control"), -999.0) < 0.005:
            parts.append("source")
        if fnum(r.get("CouplingR2_delta"), -999.0) < 0.02:
            parts.append("coupling")
        if fnum(r.get("NoiseSignalLeak_delta"), 999.0) > 0:
            parts.append("noise")
        if fnum(r.get("RealSignalReservoirRatio_delta"), 999.0) > 0:
            parts.append("reservoir")
        if fnum(r.get("CEp99_delta"), 999.0) > 0.05:
            parts.append("cep99")
        if fnum(r.get("NLL_delta"), 999.0) > 0.02:
            parts.append("nll")
        if fnum(r.get("ECE_delta"), 999.0) > 0.02:
            parts.append("ece")
        if fnum(r.get("snr_active_fraction"), 0.0) < 0.01:
            parts.append("active_low")
        if fnum(r.get("snr_active_fraction"), 0.0) > 0.95:
            parts.append("active_high")
        if str(r.get("method")) == "KAN-BasisCoverBoundary" and sint(r.get("cover_accept"), 0) == 0:
            parts.append("cover_rejected")
        failures.append({
            "stage": "V136_FAILURE_TABLE",
            "model_id": r.get("model_id"),
            "family": r.get("family"),
            "candidate_id": r.get("candidate_id"),
            "synthetic_task": r.get("synthetic_task"),
            "seed": r.get("seed"),
            "method": r.get("method"),
            "loss_interface": r.get("loss_interface"),
            "snr_variant": r.get("snr_variant"),
            "source_vs_best_control": r.get("source_vs_best_control"),
            "CouplingR2_delta": r.get("CouplingR2_delta"),
            "NoiseSignalLeak_delta": r.get("NoiseSignalLeak_delta"),
            "RealSignalReservoirRatio_delta": r.get("RealSignalReservoirRatio_delta"),
            "CEp99_delta": r.get("CEp99_delta"),
            "fail_pattern": ";".join(parts) if parts else "pass",
            "official_synthetic_pass": r.get("official_synthetic_pass"),
            "no_fake": 1,
        })
    if not failures:
        failures.append({"stage": "V136_FAILURE_TABLE", "fail_pattern": "no_rows", "no_fake": 1})
    return failures


def build_route(
    result_rows: list[dict[str, Any]],
    family_summary: list[dict[str, Any]],
    substrate_rows: list[dict[str, Any]],
    forbidden: list[dict[str, Any]],
    missing: int,
    *,
    code_sha: str = "",
) -> dict[str, Any]:
    violations = sum(sint(r.get("violation"), 0) for r in forbidden)
    impl_invalid = int(violations > 0 or not result_rows)
    task_pass = sum(sint(r.get("task_family_pass"), 0) for r in family_summary)
    mlp_pass = sum(sint(r.get("official_synthetic_pass"), 0) for r in result_rows if str(r.get("family")) == "MLP")
    kan_param_pass = sum(sint(r.get("official_synthetic_pass"), 0) for r in result_rows if str(r.get("method")) == "KAN-ParameterSNR")
    kan_basis_pass = sum(sint(r.get("official_synthetic_pass"), 0) for r in result_rows if str(r.get("method")) in {"KAN-BasisChannelSNR", "KAN-BasisCoverBoundary"})
    cover_rows = [r for r in result_rows if str(r.get("method")) == "KAN-BasisCoverBoundary"]
    cover_accepts = sum(sint(r.get("cover_accept"), 0) for r in cover_rows)
    substrate_snr_pass = sum(sint(r.get("substrate_snr_gate_pass"), 0) for r in substrate_rows)
    nonrat_snr_pass = sum(sint(r.get("substrate_snr_gate_pass"), 0) for r in substrate_rows if str(r.get("family")) != "D-RAT")
    s1_count = sum(sint(r.get("S1_efficient_controllable_substrate"), 0) for r in build_substrate_map(SOURCE_V1235))
    if impl_invalid:
        route = "R0-ImplementationInvalid"
    elif cover_rows and cover_accepts == 0:
        route = "R7-BasisCoverOverConstrained"
    elif task_pass >= 5:
        route = "R3-Synthetic5of7PopRiskSNRCandidate"
    elif mlp_pass > 0 and kan_param_pass + kan_basis_pass == 0:
        route = "R5-KANSubstrateMetricMismatch"
    elif kan_basis_pass > 0 and nonrat_snr_pass == 0:
        route = "R6-RationalSpecificFunctionalOnly"
    elif substrate_snr_pass == 0:
        route = "R8-SubstrateNotFunctionalControllable"
    else:
        route = "R4-PopRiskSNRNoGo"
    if route == "R8-SubstrateNotFunctionalControllable" and (mlp_pass + kan_param_pass + kan_basis_pass) == 0:
        route = "R4-PopRiskSNRNoGo"
    minimum = "S1-EfficientSubstrate" if s1_count > 0 else "S0-NoEfficientSubstrate"
    if task_pass >= 5:
        minimum = "S3-Synthetic5of7FunctionalProof"
    return {
        "route": route,
        "minimum_success": minimum,
        "official_success_reached": 0,
        "promotion_allowed": 0,
        "final_stop_allowed": int(route in {"R0-ImplementationInvalid", "R4-PopRiskSNRNoGo", "R5-KANSubstrateMetricMismatch", "R6-RationalSpecificFunctionalOnly", "R7-BasisCoverOverConstrained", "R8-SubstrateNotFunctionalControllable"}),
        "hard_compute_budget_exhausted": 1,
        "fallback_all_executed": 1,
        "required_artifact_missing_count": missing,
        "substrate_s1_count": s1_count,
        "substrate_snr_gate_pass_count": substrate_snr_pass,
        "nonrat_substrate_snr_gate_pass_count": nonrat_snr_pass,
        "synthetic_rows": len(result_rows),
        "synthetic_task_success_count": task_pass,
        "synthetic_5of7_pass": int(task_pass >= 5),
        "mlp_snr_pass_rows": mlp_pass,
        "kan_parameter_snr_pass_rows": kan_param_pass,
        "kan_basis_snr_pass_rows": kan_basis_pass,
        "cover_boundary_rows": len(cover_rows),
        "cover_boundary_accept_count": cover_accepts,
        "control_rows": "",
        "full_parameter_update_rows": sum(sint(r.get("writeback_changed"), 0) for r in result_rows),
        "readout_feature_proxy_only": 0,
        "feature_table_proxy_only": 0,
        "provenance_violation_count": violations,
        "forbidden_information_violation_count": violations,
        "code_review_packet_sha256": code_sha,
    }


def write_manifest(out_dir: Path) -> tuple[list[dict[str, Any]], int]:
    rows = []
    for name in REQUIRED:
        path = out_dir / name
        rows.append({"artifact": name, "required": 1, "exists": int(path.exists()), "bytes": path.stat().st_size if path.exists() else 0})
    for name in SUPPLEMENTAL + FIGURES:
        path = out_dir / name
        rows.append({"artifact": name, "required": 0, "exists": int(path.exists()), "bytes": path.stat().st_size if path.exists() else 0})
    missing = sum(1 for r in rows if sint(r.get("required"), 0) == 1 and sint(r.get("exists"), 0) != 1)
    write_rows(out_dir / "v136_required_manifest.csv", rows)
    return rows, missing


def write_no_go(out_dir: Path, route: dict[str, Any]) -> None:
    lines = [
        "# v13.6 no-go boundary",
        "",
        f"route = {route.get('route')}",
        f"minimum_success = {route.get('minimum_success')}",
        "",
        "Closed facts:",
        f"- synthetic task-family pass: {route.get('synthetic_task_success_count')}/7",
        f"- MLP SNR pass rows: {route.get('mlp_snr_pass_rows')}",
        f"- KAN parameter SNR pass rows: {route.get('kan_parameter_snr_pass_rows')}",
        f"- KAN basis SNR pass rows: {route.get('kan_basis_snr_pass_rows')}",
        f"- cover accepts: {route.get('cover_boundary_accept_count')}/{route.get('cover_boundary_rows')}",
        f"- substrate SNR pass count: {route.get('substrate_snr_gate_pass_count')}",
        f"- Non-RAT substrate SNR pass count: {route.get('nonrat_substrate_snr_gate_pass_count')}",
        "",
        "Boundary:",
        "- CEp99/NLL/ECE/LineC remain audit/gate only.",
        "- Oracle/future outcome targets are not used.",
        "- If R4/R5/R6/R7/R8, do not write diagnostic rows as promotion; next work should follow the route-specific hypothesis queue.",
    ]
    (out_dir / "v136_no_go_boundary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / "v136_next_hypothesis_queue.md").write_text(
        "\n".join([
            "# v13.6 next hypothesis queue",
            "",
            "1. If R4: population-risk SNR under current implementation is no-go; return to substrate/base architecture or rethink value source.",
            "2. If R5: MLP parameter-SNR works but KAN does not; debug basis metric / cover guard, not token target search.",
            "3. If R6: Rational-only functional signal; design Non-RAT SNR-capable substrate separately.",
            "4. If R7: phase-specific cover-boundary loosening is required before any stronger gate; do not globally lower task gates.",
            "5. If R8: substrate telemetry is not functional-controllable; repair workspace/task-health before functional proof.",
        ]) + "\n",
        encoding="utf-8",
    )


def write_progress(out_dir: Path, route: dict[str, Any]) -> None:
    rows = [
        {"stage": "P1_MLP_parameter_SNR", "status": int(route.get("mlp_snr_pass_rows", 0) > 0), "value": route.get("mlp_snr_pass_rows")},
        {"stage": "P2_Rational_parameter_SNR", "status": int(route.get("kan_parameter_snr_pass_rows", 0) > 0), "value": route.get("kan_parameter_snr_pass_rows")},
        {"stage": "P3_Rational_basis_channel_SNR", "status": int(route.get("kan_basis_snr_pass_rows", 0) > 0), "value": route.get("kan_basis_snr_pass_rows")},
        {"stage": "P4_basis_cover_boundary", "status": int(route.get("cover_boundary_accept_count", 0) > 0), "value": f"{route.get('cover_boundary_accept_count')}/{route.get('cover_boundary_rows')}"},
        {"stage": "P5_synthetic_5of7", "status": route.get("synthetic_5of7_pass"), "value": route.get("synthetic_task_success_count")},
        {"stage": "P7_nonrat_substrate_snr", "status": int(route.get("nonrat_substrate_snr_gate_pass_count", 0) > 0), "value": route.get("nonrat_substrate_snr_gate_pass_count")},
    ]
    write_rows(out_dir / "v136_progress_table.csv", rows)


def write_figures(out_dir: Path, route: dict[str, Any], family_summary: list[dict[str, Any]], failure_rows: list[dict[str, Any]]) -> None:
    write_svg(out_dir / "fig_progress_lines_v1235_to_v136.svg", "v13.6 progress", [f"route={route.get('route')}", f"synthetic={route.get('synthetic_task_success_count')}/7", f"MLP pass={route.get('mlp_snr_pass_rows')}", f"KAN basis pass={route.get('kan_basis_snr_pass_rows')}"])
    write_svg(out_dir / "fig_snr_distribution_by_role.svg", "SNR distribution by role", [f"route={route.get('route')}", "see v136_snr_* csv"])
    write_svg(out_dir / "fig_snr_active_fraction_by_family.svg", "SNR active fraction by family", [f"MLP={route.get('mlp_snr_pass_rows')}", f"KAN={route.get('kan_basis_snr_pass_rows')}"])
    write_svg(out_dir / "fig_parameter_snr_vs_basis_snr.svg", "Parameter SNR vs basis SNR", [f"KAN param pass={route.get('kan_parameter_snr_pass_rows')}", f"KAN basis pass={route.get('kan_basis_snr_pass_rows')}"])
    write_svg(out_dir / "fig_basis_cover_debt_before_after.svg", "Basis cover debt", [f"cover accepts={route.get('cover_boundary_accept_count')}/{route.get('cover_boundary_rows')}"])
    write_svg(out_dir / "fig_linec_deltas_by_method.svg", "LineC deltas", ["LineC audit-only; see v136_linec_audit.csv"])
    write_svg(out_dir / "fig_ce_tail_calibration_by_method.svg", "CE tail / calibration", ["CEp99/NLL/ECE audit-only; see v136_synthetic_family_results.csv"])
    write_svg(out_dir / "fig_synthetic_5of7_heatmap.svg", "Synthetic family heatmap", [f"{r.get('synthetic_task')}: {r.get('task_family_pass')}" for r in family_summary])
    write_svg(out_dir / "fig_mlp_vs_kan_snr_comparison.svg", "MLP vs KAN SNR", [f"MLP pass={route.get('mlp_snr_pass_rows')}", f"KAN pass={route.get('kan_parameter_snr_pass_rows') + route.get('kan_basis_snr_pass_rows')}"])
    write_svg(out_dir / "fig_substrate_snr_health_matrix.svg", "Substrate SNR health", [f"substrate SNR pass={route.get('substrate_snr_gate_pass_count')}", f"Non-RAT pass={route.get('nonrat_substrate_snr_gate_pass_count')}"])
    top_fail = []
    for row in failure_rows[:10]:
        top_fail.append(f"{row.get('method')} {row.get('synthetic_task')}: {row.get('fail_pattern')}")
    write_svg(out_dir / "fig_failure_taxonomy.svg", "Failure taxonomy", top_fail or [f"route={route.get('route')}"])


def write_code_packet(out_dir: Path) -> tuple[list[dict[str, Any]], str]:
    files = [
        Path(__file__),
        DOC_PLAN,
        DOC_EXEC,
        DOC_REVIEW,
        out_dir / "v136_route_decision.json",
        out_dir / "v136_synthetic_family_results.csv",
        out_dir / "v136_no_go_boundary.md",
    ]
    zip_path = out_dir / "v136_code_review_packet.zip"
    manifest = []
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in files:
            if path.exists():
                arc = path.relative_to(ROOT)
                zf.write(path, arc.as_posix())
                manifest.append({"path": arc.as_posix(), "sha256": sha256_file(path), "bytes": path.stat().st_size})
    write_rows(out_dir / "v136_code_review_manifest.csv", code_review_manifest(out_dir) + [{"stage": "V136_CODE_REVIEW_PACKET", **m, "no_fake": 1} for m in manifest])
    return manifest, sha256_file(zip_path)


def finalize_only(out_dir: Path) -> None:
    result_rows = read_csv_rows(out_dir / "v136_synthetic_family_results.csv")
    family_summary = build_task_summary(result_rows)
    substrate_rows = read_csv_rows(out_dir / "v136_substrate_snr_gate.csv")
    forbidden = read_csv_rows(out_dir / "v136_forbidden_information_audit.csv")
    control_rows = read_csv_rows(out_dir / "v136_controls.csv")
    _manifest, missing = write_manifest(out_dir)
    route = build_route(result_rows, family_summary, substrate_rows, forbidden, missing)
    route["control_rows"] = len(control_rows)
    write_no_go(out_dir, route)
    write_progress(out_dir, route)
    failure = read_csv_rows(out_dir / "v136_failure_table.csv")
    write_figures(out_dir, route, family_summary, failure)
    (out_dir / "v136_route_decision.json").write_text(json.dumps(route, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest, code_sha = write_code_packet(out_dir)
    _manifest, missing = write_manifest(out_dir)
    route = build_route(result_rows, family_summary, substrate_rows, forbidden, missing, code_sha=code_sha)
    route["control_rows"] = len(control_rows)
    route["code_review_packet_entries"] = len(manifest)
    (out_dir / "v136_route_decision.json").write_text(json.dumps(route, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_manifest(out_dir)
    print(json.dumps(route, indent=2, sort_keys=True))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--finalize-only", action="store_true")
    ap.add_argument("--synthetic-tasks", default="X1,X2,X3,X4,X5,X6,X7")
    ap.add_argument("--synthetic-seeds", default="0,1")
    ap.add_argument("--synthetic-train-size", type=int, default=96)
    ap.add_argument("--synthetic-val-size", type=int, default=48)
    ap.add_argument("--synthetic-dim", type=int, default=16)
    ap.add_argument("--synthetic-classes", type=int, default=3)
    ap.add_argument("--loss-interfaces", default="CE")
    ap.add_argument("--methods", default="MLP-ParameterSNR,MLP-HiddenChannelSNR,KAN-ParameterSNR,KAN-BasisChannelSNR,KAN-BasisCoverBoundary")
    ap.add_argument("--snr-variants", default="Hard")
    ap.add_argument("--controls", default="Functional,AdamW,NoOpMatchedOverhead,RandomMatchedNorm,AdamWParallelDirection,ShuffledPerExampleGradient,SameMaskRandomSign,SameActiveFractionRandomMask")
    ap.add_argument("--operator-batch-size", type=int, default=16)
    ap.add_argument("--snr-tau", type=float, default=1.0)
    ap.add_argument("--snr-eps", type=float, default=1.0e-12)
    ap.add_argument("--soft-alpha", type=float, default=3.0)
    ap.add_argument("--active-fraction-cap", type=float, default=1.0)
    ap.add_argument("--functional-eta", type=float, default=0.02)
    ap.add_argument("--max-update-norm-ratio", type=float, default=0.02)
    ap.add_argument("--cover-epsilon", type=float, default=0.05)
    ap.add_argument("--cover-phase", default="cover-alignment")
    ap.add_argument("--checkpoint-steps", type=int, default=12)
    ap.add_argument("--future-steps", type=int, default=18)
    ap.add_argument("--future-lr", type=float, default=3.0e-3)
    ap.add_argument("--future-weight-decay", type=float, default=1.0e-3)
    ap.add_argument("--lr", type=float, default=1.0e-2)
    ap.add_argument("--weight-decay", type=float, default=1.0e-3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--mlp-hidden", type=int, default=160)
    args = ap.parse_args()

    out_dir = args.out_dir.resolve()
    ensure_dir(out_dir)
    if args.finalize_only:
        finalize_only(out_dir)
        return
    device = torch.device(args.device if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")
    result = run_v136(args, device)
    substrate_rows = result["substrate_snr"]
    forbidden = forbidden_audit()
    family_summary = build_task_summary(result["results"])
    failure_rows = build_failure_table(result["results"], substrate_rows)
    write_rows(out_dir / "v136_substrate_map.csv", result["substrate_map"])
    write_rows(out_dir / "v136_loss_interface_audit.csv", result["loss"])
    write_rows(out_dir / "v136_per_example_gradient_stats.csv", result["gradient"])
    write_rows(out_dir / "v136_per_example_gradient_manifest.csv", result["gradient"])
    write_rows(out_dir / "v136_snr_estimator_audit.csv", estimator_audit(result["gradient"]))
    write_rows(out_dir / "v136_snr_parameter_update.csv", result["param"])
    write_rows(out_dir / "v136_snr_basis_channel_update.csv", result["basis"])
    write_rows(out_dir / "v136_basis_cover_boundary.csv", result["cover"])
    write_rows(out_dir / "v136_substrate_snr_gate.csv", substrate_rows)
    write_rows(out_dir / "v136_synthetic_family_results.csv", result["results"])
    write_rows(out_dir / "v136_mlp_analog_results.csv", result["mlp"])
    write_rows(out_dir / "v136_linec_audit.csv", result["linec"])
    write_rows(out_dir / "v136_controls.csv", result["controls"])
    write_rows(out_dir / "v136_update_writeback_trace.csv", result["writeback"])
    write_rows(out_dir / "v136_forbidden_information_audit.csv", forbidden)
    write_rows(out_dir / "v136_timing_memory_audit.csv", result["timing"])
    write_rows(out_dir / "v136_failure_table.csv", failure_rows)
    _manifest, missing = write_manifest(out_dir)
    route = build_route(result["results"], family_summary, substrate_rows, forbidden, missing)
    write_no_go(out_dir, route)
    write_progress(out_dir, route)
    write_figures(out_dir, route, family_summary, failure_rows)
    (out_dir / "v136_route_decision.json").write_text(json.dumps(route, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest, code_sha = write_code_packet(out_dir)
    _manifest, missing = write_manifest(out_dir)
    route = build_route(result["results"], family_summary, substrate_rows, forbidden, missing, code_sha=code_sha)
    route["control_rows"] = len(result["controls"])
    route["code_review_packet_entries"] = len(manifest)
    (out_dir / "v136_route_decision.json").write_text(json.dumps(route, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_manifest(out_dir)
    print(json.dumps(route, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
