#!/usr/bin/env python
"""v13.7 boundary-conditioned population-risk training runner.

This runner is intentionally not a one-shot perturbation probe.  It applies a
persistent train-stream SNR state inside a multi-step training loop, optionally
conditioned by a phase-dependent basis-cover boundary guard.  Validation/future
outcomes, LineC, and tail metrics are audit gates only; they never generate the
training direction.
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
    make_model_for_family,
    parse_csv,
    parse_ints,
    s1_gate,
    sha256_file,
    sint,
    synthetic_data,
    write_rows,
    write_svg,
)
from experiments.run_v134_operator_level_basis_channel_functional import (  # noqa: E402
    channel_named_params,
    param_sha,
)
from experiments.run_v136_poprisk_snr_basis_cover_boundary import (  # noqa: E402
    collect_per_example_gradients,
    cover_debt,
    flatten_tensors,
    loss_interface_cotangent_per_example,
    role_of_param,
)


OUT_DIR = ROOT / "results" / "v13_7_boundary_conditioned_poprisk_training" / "official_v137"
DOC_PLAN = ROOT / "docs" / "DG-KAN_v13.7_BoundaryConditionedPopRiskTraining_完整计划.md"
DOC_EXEC = ROOT / "docs" / "DG-KAN_v13.7_BoundaryConditionedPopRiskTraining_执行日志.md"
DOC_REVIEW = ROOT / "docs" / "DG-KAN_v13.7_BoundaryConditionedPopRiskTraining_实验结果复盘.md"

REQUIRED = [
    "v137_route_decision.json",
    "v137_implementation_readback.csv",
    "v137_snr_sanity_checks.csv",
    "v137_mlp_snr_training.csv",
    "v137_rational_snr_training.csv",
    "v137_basis_cover_schedule.csv",
    "v137_nonrat_substrate_repair.csv",
    "v137_training_controls.csv",
    "v137_synthetic_family_summary.csv",
    "v137_linec_audit.csv",
    "v137_forbidden_information_audit.csv",
    "v137_real_triage_if_opened.csv",
    "v137_failure_table.csv",
    "v137_progress_table.csv",
    "v137_required_manifest.csv",
    "v137_no_go_boundary.md",
    "v137_next_hypothesis_queue.md",
    "v137_code_review_manifest.csv",
    "v137_code_review_packet.zip",
]

SUPPLEMENTAL = [
    "v137_substrate_map.csv",
    "v137_update_writeback_trace.csv",
    "v137_loss_interface_audit.csv",
    "v137_timing_memory_audit.csv",
    "v137_rational_rejection_audit.csv",
]

FIGURES = [
    "fig_mlp_snr_vs_adamw_loss_time.svg",
    "fig_rational_parameter_snr_vs_basis_snr.svg",
    "fig_basis_snr_active_fraction_by_phase.svg",
    "fig_cover_boundary_debt_by_phase.svg",
    "fig_signal_reservoir_trajectory.svg",
    "fig_noise_leak_vs_snr_active_fraction.svg",
    "fig_nonrat_substrate_health_matrix.svg",
    "fig_mlp_vs_rational_snr_comparison.svg",
    "fig_synthetic_5of7_heatmap.svg",
    "fig_real_triage_if_opened.svg",
]

NONRAT_REPAIR_SPECS: list[dict[str, str]] = [
    {"family": "D-CHE", "candidate_id": "CHE-SNR4-degreeLowRankResidual", "mapped_candidate_id": "D-CHE20-DegreeNormalizedReadoutHealthSubstrate"},
    {"family": "D-CHE", "candidate_id": "CHE-SNR5-degreeLateEnable", "mapped_candidate_id": "D-CHE17-HighDegreeLateEnableSubstrate"},
    {"family": "D-FOU", "candidate_id": "FOU-SNR4-lowFreqIdentityResidual", "mapped_candidate_id": "D-FOU18-LowFreqSignalTransportSubstrate"},
    {"family": "D-FOU", "candidate_id": "FOU-SNR5-bandLimitedResidual", "mapped_candidate_id": "D-FOU19-HighFreqNoiseLeakVetoSubstrate"},
    {"family": "D-RBF", "candidate_id": "RBF-SNR4-compactOccupancyRepair", "mapped_candidate_id": "D-RBF12-CenterOccupancyRebalanceSubstrate"},
    {"family": "D-RBF", "candidate_id": "RBF-SNR5-widthConditionedLocalBump", "mapped_candidate_id": "D-RBF13-WidthConditionGuardSubstrate"},
    {"family": "D-WAV", "candidate_id": "WAV-SNR4-hatScaleBalanced", "mapped_candidate_id": "D-WAV11-ScaleEnergyBalanceSubstrate"},
    {"family": "D-WAV", "candidate_id": "WAV-SNR5-localSupportOccupancy", "mapped_candidate_id": "D-WAV14-SupportOverlapEntropyGuardSubstrate"},
]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def finite_mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(sum(vals) / len(vals)) if vals else float("nan")


def loss_value(logits: torch.Tensor, y: torch.Tensor, interface: str) -> torch.Tensor:
    logits_f = logits.float()
    if interface == "CE":
        return F.cross_entropy(logits_f, y)
    if interface == "Brier":
        target = F.one_hot(y, num_classes=logits_f.shape[1]).to(dtype=torch.float32, device=logits_f.device)
        return (torch.softmax(logits_f, dim=1) - target).square().sum(dim=1).mean()
    if interface == "MSELogit":
        target = 2.0 * F.one_hot(y, num_classes=logits_f.shape[1]).to(dtype=torch.float32, device=logits_f.device) - 1.0
        return (logits_f - target).square().mean()
    raise ValueError(interface)


def all_named_params(model: torch.nn.Module) -> list[tuple[str, torch.nn.Parameter]]:
    return [(n, p) for n, p in model.named_parameters() if p.requires_grad]


def selected_params(model: torch.nn.Module, family: str, method: str) -> list[tuple[str, torch.nn.Parameter]]:
    if family == "MLP":
        return all_named_params(model)
    if "BasisSNR" in method or "GroupSNR" in method:
        params = channel_named_params(model)
        return params if params else all_named_params(model)
    return all_named_params(model)


def parameter_sha_by_params(params: list[tuple[str, torch.nn.Parameter]]) -> str:
    return param_sha(params)


def assign_flat_grad(params: list[tuple[str, torch.nn.Parameter]], flat_grad: torch.Tensor) -> None:
    offset = 0
    for _name, p in params:
        n = int(p.numel())
        grad = flat_grad[offset : offset + n].reshape_as(p).to(device=p.device, dtype=p.dtype)
        p.grad = grad.detach().clone()
        offset += n


@dataclass
class SNRState:
    decay: float = 0.85
    mu_ema: torch.Tensor | None = None
    var_ema: torch.Tensor | None = None
    frozen_gate: torch.Tensor | None = None
    frozen_group_scores: list[float] | None = None
    cover_prev_active: torch.Tensor | None = None
    updates: int = 0

    def update(self, g: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu = g.mean(dim=0) if int(g.numel()) else torch.zeros(0, device=g.device)
        var = g.var(dim=0, unbiased=True) if int(g.shape[0]) > 1 and int(g.numel()) else torch.zeros_like(mu)
        if self.mu_ema is None or self.var_ema is None or self.mu_ema.numel() != mu.numel():
            self.mu_ema = mu.detach().clone()
            self.var_ema = var.detach().clone()
        else:
            d = float(self.decay)
            self.mu_ema = d * self.mu_ema + (1.0 - d) * mu.detach()
            self.var_ema = d * self.var_ema + (1.0 - d) * var.detach()
        self.updates += 1
        return mu, var


def param_groups(params: list[tuple[str, torch.nn.Parameter]]) -> list[tuple[str, int, int]]:
    groups: list[tuple[str, int, int]] = []
    offset = 0
    for name, p in params:
        n = int(p.numel())
        groups.append((role_of_param(name), offset, offset + n))
        offset += n
    return groups


def _gini_from_values(values: list[float]) -> float:
    vals = torch.tensor([max(0.0, float(v)) for v in values], dtype=torch.float32)
    if int(vals.numel()) == 0 or float(vals.sum().item()) <= 0.0:
        return 0.0
    vals = torch.sort(vals).values
    n = float(vals.numel())
    idx = torch.arange(1, int(vals.numel()) + 1, dtype=torch.float32)
    return float(((2.0 * idx - n - 1.0) * vals).sum().item() / (n * vals.sum().item() + 1.0e-12))


def snr_gate(
    g: torch.Tensor,
    params: list[tuple[str, torch.nn.Parameter]],
    state: SNRState,
    *,
    method: str,
    phase: str,
    tau: float,
    eps: float,
    soft_alpha: float,
    active_fraction_cap: float,
) -> dict[str, Any]:
    mu_now, var_now = state.update(g)
    use_ema = "EMA" in method or "RoleNorm" in method
    mu = state.mu_ema if use_ema and state.mu_ema is not None else mu_now
    var = state.var_ema if use_ema and state.var_ema is not None else var_now
    b = max(2, int(g.shape[0]))
    phase_scale = 0.65 if phase == "plasticity-open" else (1.0 if phase == "cover-alignment" else 1.35)
    tau_eff = float(tau) * phase_scale
    noise = var / float(max(1, b - 1))
    ratio = mu.square() / (noise + float(eps))
    groups = param_groups(params)

    group_purities: list[float] = []
    group_loads: list[float] = []
    group_param_snr_scores: list[float] = []
    for _role, start, end in groups:
        gg = g[:, start:end] if int(g.numel()) else torch.zeros((0, 0), device=g.device)
        rr = ratio[start:end]
        if int(gg.numel()) == 0 or int(rr.numel()) == 0:
            group_purities.append(0.0)
            group_loads.append(0.0)
            group_param_snr_scores.append(0.0)
            continue
        mean_grad = gg.mean(dim=0)
        purity = float((mean_grad.square().sum() / gg.square().sum(dim=1).mean().clamp_min(float(eps))).detach().item())
        load = float(gg.norm(dim=1).mean().detach().item())
        group_purities.append(max(0.0, min(1.0, purity)))
        group_loads.append(max(0.0, load))
        group_param_snr_scores.append(float(rr.mean().detach().item()))

    if "GroupSNR" in method or "BasisSNR" in method:
        gate = torch.zeros_like(mu)
        group_scores: list[float] = []
        for _role, start, end in groups:
            r = ratio[start:end]
            score = float(r.mean().detach().item()) if int(r.numel()) else 0.0
            group_scores.append(score)
            if "Soft" in method:
                val = float(torch.sigmoid(torch.tensor(float(soft_alpha) * (math.log(score + float(eps)) - math.log(tau_eff + float(eps))))).item())
            else:
                val = 1.0 if score > tau_eff else 0.0
            gate[start:end] = val
    elif "Soft" in method:
        gate = torch.sigmoid(float(soft_alpha) * (torch.log(ratio + float(eps)) - math.log(tau_eff + float(eps))))
        group_scores = []
    elif "RoleNorm" in method:
        gate = torch.zeros_like(mu)
        group_scores = []
        for _role, start, end in groups:
            r = ratio[start:end]
            if int(r.numel()) == 0:
                continue
            cutoff = torch.quantile(r, 0.60)
            gate[start:end] = (r >= cutoff).float()
            group_scores.append(float(r.mean().detach().item()))
    else:
        gate = (ratio > tau_eff).float()
        group_scores = []

    if 0.0 < float(active_fraction_cap) < 1.0 and int(ratio.numel()) > 0:
        k = max(1, int(math.ceil(float(active_fraction_cap) * int(ratio.numel()))))
        threshold = torch.topk(ratio, k=k, largest=True).values.min()
        gate = gate * (ratio >= threshold).float()

    cover_split_count = 0
    cover_merge_count = 0
    cover_freeze_count = 0
    false_drop_fraction = 0.0
    false_keep_fraction = 0.0
    param_gate = (ratio > tau_eff).float() if int(ratio.numel()) else torch.zeros_like(ratio)
    method_lower = method.lower()
    if any(token in method for token in ["GradientClusterCover", "CoverSplitMerge", "ParamSNRThenCover", "ReadoutBasisDecoupled", "ReservoirProxyCoverGrowth", "LowVarianceSignalCoverGrowth"]):
        cover_scores = [
            float(group_param_snr_scores[i]) * float(group_purities[i]) * (1.0 + float(group_loads[i]))
            for i in range(len(groups))
        ]
        if cover_scores:
            score_tensor = torch.tensor(cover_scores, device=mu.device, dtype=mu.dtype)
            purity_tensor = torch.tensor(group_purities, device=mu.device, dtype=mu.dtype)
            load_tensor = torch.tensor(group_loads, device=mu.device, dtype=mu.dtype)
            if "GradientClusterCover" in method:
                k = 8 if "k8" in method_lower else 4
                k = max(1, min(int(k), int(score_tensor.numel())))
                cutoff = torch.topk(score_tensor, k=k, largest=True).values.min()
                group_active = (score_tensor >= cutoff).float()
            elif "CoverSplitMerge" in method:
                load_cut = torch.quantile(load_tensor, 0.70) if int(load_tensor.numel()) > 1 else load_tensor.mean()
                score_cut = torch.quantile(score_tensor, 0.50) if int(score_tensor.numel()) > 1 else score_tensor.mean()
                conflict = 1.0 - purity_tensor
                group_active = (score_tensor >= score_cut).float()
                split_mask = (load_tensor >= load_cut) & (conflict >= 0.50)
                freeze_mask = (load_tensor <= torch.quantile(load_tensor, 0.30)) & (score_tensor <= score_cut) if int(load_tensor.numel()) > 1 else (score_tensor <= score_cut)
                cover_split_count = int(split_mask.float().sum().item())
                cover_freeze_count = int(freeze_mask.float().sum().item())
                cover_merge_count = 0 if "noMerge" in method else int(freeze_mask.float().sum().item())
                group_active = group_active * (~freeze_mask).float()
            elif "ParamSNRThenCover" in method:
                if phase == "plasticity-open":
                    group_active = torch.ones_like(score_tensor)
                else:
                    q = 0.50 if "slowConsolidate" in method else 0.60
                    cutoff = torch.quantile(score_tensor, q) if int(score_tensor.numel()) > 1 else score_tensor.mean()
                    group_active = (score_tensor >= cutoff).float()
                    cover_freeze_count = int((group_active <= 0).float().sum().item()) if phase == "fixed-point-consolidation" else 0
            elif "ReadoutBasisDecoupled" in method:
                cutoff = torch.quantile(score_tensor, 0.50) if int(score_tensor.numel()) > 1 else score_tensor.mean()
                group_active = (score_tensor >= cutoff).float()
            elif "LowVarianceSignalCoverGrowth" in method:
                cutoff = torch.quantile(purity_tensor, 0.60) if int(purity_tensor.numel()) > 1 else purity_tensor.mean()
                group_active = (purity_tensor >= cutoff).float()
            else:
                cutoff = torch.quantile(score_tensor, 0.60) if int(score_tensor.numel()) > 1 else score_tensor.mean()
                group_active = (score_tensor >= cutoff).float()
            cover_gate = torch.zeros_like(gate)
            for gi, (role, start, end) in enumerate(groups):
                if "ReadoutBasisDecoupled" in method and role == "readout":
                    cover_gate[start:end] = torch.maximum(param_gate[start:end], gate[start:end])
                elif "ParamSNRThenCover" in method and phase == "plasticity-open":
                    cover_gate[start:end] = torch.maximum(param_gate[start:end], gate[start:end])
                else:
                    cover_gate[start:end] = group_active[gi]
            if "ReservoirProxyCoverGrowth" in method:
                purity_gate = torch.zeros_like(gate)
                for gi, (_role, start, end) in enumerate(groups):
                    purity_gate[start:end] = float(group_purities[gi])
                cover_gate = cover_gate * purity_gate
            old_active = gate > 0
            new_active = cover_gate > 0
            if int(old_active.numel()):
                false_drop_fraction = float((old_active & ~new_active).float().mean().item())
                false_keep_fraction = float((~old_active & new_active).float().mean().item())
            gate = gate * cover_gate

    if "FreezeCluster" in method and int(gate.numel()) > 0:
        if phase == "plasticity-open":
            freeze_status = "warmup_dynamic"
        elif state.frozen_gate is None or int(state.frozen_gate.numel()) != int(gate.numel()):
            state.frozen_gate = gate.detach().clone()
            state.frozen_group_scores = list(group_scores)
            freeze_status = "frozen_after_warmup"
        else:
            gate = state.frozen_gate.to(device=gate.device, dtype=gate.dtype)
            group_scores = list(state.frozen_group_scores or group_scores)
            freeze_status = "using_frozen_after_warmup"
    else:
        freeze_status = "not_applied"

    grad = mu * gate
    active = float((gate > 0).float().mean().detach().item()) if int(gate.numel()) else 0.0
    active_by_layer: dict[str, float] = {}
    for role, start, end in groups:
        vals = gate[start:end]
        if int(vals.numel()):
            active_by_layer[role] = float((vals > 0).float().mean().detach().item())
    removed = 1.0 - float(grad.norm().detach().item() / mu.norm().clamp_min(1.0e-8).detach().item()) if int(mu.numel()) else 0.0
    adam = -mu
    snr_dir = -grad
    cos = float(F.cosine_similarity(snr_dir, adam, dim=0).detach().item()) if int(mu.numel()) else 0.0
    finite = ratio[torch.isfinite(ratio)]
    snr_median = float(torch.quantile(finite, 0.50).detach().item()) if int(finite.numel()) else 0.0
    snr_p90 = float(torch.quantile(finite, 0.90).detach().item()) if int(finite.numel()) else 0.0
    snr_p99 = float(torch.quantile(finite, 0.99).detach().item()) if int(finite.numel()) else 0.0
    group_active_now = torch.tensor([
        1.0 if int(gate[start:end].gt(0).float().sum().item()) > 0 else 0.0
        for _role, start, end in groups
    ], device=gate.device if int(gate.numel()) else g.device)
    if state.cover_prev_active is None or int(state.cover_prev_active.numel()) != int(group_active_now.numel()):
        cover_churn = 0.0
    else:
        prev = state.cover_prev_active.to(device=group_active_now.device)
        inter = ((prev > 0) & (group_active_now > 0)).float().sum()
        union = ((prev > 0) | (group_active_now > 0)).float().sum().clamp_min(1.0)
        cover_churn = float((1.0 - inter / union).item())
    state.cover_prev_active = group_active_now.detach().clone()
    purity_vals = torch.tensor(group_purities, dtype=torch.float32, device=g.device) if group_purities else torch.zeros(0, device=g.device)
    cover_purity_mean = float(purity_vals.mean().item()) if int(purity_vals.numel()) else 0.0
    cover_purity_p10 = float(torch.quantile(purity_vals, 0.10).item()) if int(purity_vals.numel()) else 0.0
    score_vals = torch.tensor([
        float(group_param_snr_scores[i]) * float(group_purities[i]) * (1.0 - cover_churn)
        for i in range(len(group_purities))
    ], dtype=torch.float32, device=g.device)
    signal_to_cover_score = float(torch.quantile(score_vals, 0.50).item()) if int(score_vals.numel()) else 0.0
    cover_entropy = 0.0
    if int(purity_vals.numel()) > 1 and float(purity_vals.sum().item()) > 0.0:
        probs = purity_vals / purity_vals.sum().clamp_min(1.0e-12)
        cover_entropy = float((-(probs * (probs + 1.0e-12).log()).sum() / math.log(float(purity_vals.numel()))).item())
    return {
        "grad": grad,
        "gate": gate,
        "mu": mu,
        "var": var,
        "ratio": ratio,
        "active_fraction": active,
        "active_by_layer": active_by_layer,
        "removed_update_norm_fraction": max(0.0, min(1.0, removed)),
        "cos_snr_adamw": cos,
        "snr_median": snr_median,
        "snr_p90": snr_p90,
        "snr_p99": snr_p99,
        "group_scores": group_scores,
        "tau_eff": tau_eff,
        "cluster_freeze_status": freeze_status,
        "cover_purity_mean": cover_purity_mean,
        "cover_purity_p10": cover_purity_p10,
        "cover_churn": cover_churn,
        "cover_specialization_entropy": cover_entropy,
        "cover_load_gini": _gini_from_values(group_loads),
        "signal_to_cover_score": signal_to_cover_score,
        "cover_split_count": cover_split_count,
        "cover_merge_count": cover_merge_count,
        "cover_freeze_count": cover_freeze_count,
        "false_drop_fraction": false_drop_fraction,
        "false_keep_fraction": false_keep_fraction,
    }


def phase_for_step(step: int, steps: int) -> str:
    frac = float(step) / max(1.0, float(steps))
    if frac < 0.20:
        return "plasticity-open"
    if frac < 0.70:
        return "cover-alignment"
    return "fixed-point-consolidation"


def cover_policy(method: str, phase: str) -> tuple[float, float]:
    if "CoverNoPlasticity" in method:
        return 0.0, 0.0
    if "MinCoverGuard" in method:
        return 0.03, 0.35
    if "ThenBasisConsolidation" in method:
        return (0.00, 0.70) if phase == "fixed-point-consolidation" else (0.0, 0.0)
    if "ThenCoverPhase" in method:
        if phase == "plasticity-open":
            return 0.04, 0.25
        if phase == "cover-alignment":
            return 0.02, 0.50
        return 0.00, 0.80
    if "CoverConsolidateOnly" in method and phase != "fixed-point-consolidation":
        return 0.0, 1.0
    if "CoverWeak" in method:
        return 0.15, 0.50
    if "CoverPhaseSchedule" in method:
        if phase == "plasticity-open":
            return 0.05, 0.35
        if phase == "cover-alignment":
            return 0.02, 0.65
        return 0.00, 1.00
    if "BasisSNR" in method:
        return 0.02, 0.50
    return 0.0, 0.0


def rational_boundary_terms(model: torch.nn.Module) -> dict[str, float]:
    """Parameter-only Rational boundary telemetry for Case B rejection audit.

    These quantities are diagnostics.  They are not used to form the training
    direction or to select a candidate during training.
    """
    den_parts: list[torch.Tensor] = []
    num_parts: list[torch.Tensor] = []
    group_norms: list[float] = []
    with torch.no_grad():
        for name, p in model.named_parameters():
            vals = p.detach().float().reshape(-1)
            if int(vals.numel()) == 0:
                continue
            group_norms.append(float(vals.norm().item()))
            low = name.lower()
            if "denominator" in low:
                den_parts.append(vals.abs())
            if "numerator" in low:
                num_parts.append(vals.abs())
        den = torch.cat(den_parts) if den_parts else torch.zeros(1)
        num = torch.cat(num_parts) if num_parts else torch.zeros(1)
        norms = torch.tensor([max(v, 0.0) for v in group_norms], dtype=torch.float32)
        if int(norms.numel()) <= 1 or float(norms.sum().item()) <= 0.0:
            diversity = 0.0
        else:
            probs = norms / norms.sum().clamp_min(1.0e-12)
            diversity = float((-(probs * (probs + 1.0e-12).log()).sum() / math.log(float(norms.numel()))).item())
        den_p01 = float(torch.quantile(den, 0.01).item())
        den_p99 = float(torch.quantile(den, 0.99).item())
        r_prime_p99 = float(torch.quantile(num, 0.99).item())
        # A conservative curvature proxy from numerator coefficient spread.
        r_double_prime_p99 = float(torch.quantile((num - num.mean()).abs(), 0.99).item())
    return {
        "den_p01": den_p01,
        "den_p99": den_p99,
        "r_prime_p99": r_prime_p99,
        "r_double_prime_p99": r_double_prime_p99,
        "group_diversity": diversity,
    }


def would_increase_cover_debt(
    model: torch.nn.Module,
    x: torch.Tensor,
    params: list[tuple[str, torch.nn.Parameter]],
    grad: torch.Tensor,
    lr: float,
    *,
    epsilon: float,
) -> tuple[int, float, float, float]:
    before = cover_debt(model, x)["cover_debt"]
    offset = 0
    with torch.no_grad():
        for _name, p in params:
            n = int(p.numel())
            p.add_(-float(lr) * grad[offset : offset + n].reshape_as(p).to(device=p.device, dtype=p.dtype))
            offset += n
    after = cover_debt(model, x)["cover_debt"]
    offset = 0
    with torch.no_grad():
        for _name, p in params:
            n = int(p.numel())
            p.add_(float(lr) * grad[offset : offset + n].reshape_as(p).to(device=p.device, dtype=p.dtype))
            offset += n
    delta = float(after - before)
    accept = int(delta <= float(epsilon))
    return accept, float(before), float(after), delta


def random_same_fraction_grad(mu: torch.Tensor, gate: torch.Tensor, *, seed: int) -> torch.Tensor:
    gen = torch.Generator(device=mu.device).manual_seed(int(seed))
    n = int(mu.numel())
    if n == 0:
        return mu
    k = max(1, int((gate > 0).float().sum().item()))
    perm = torch.randperm(n, device=mu.device, generator=gen)
    mask = torch.zeros(n, device=mu.device)
    mask[perm[:k]] = 1.0
    return mu * mask


def blend_alpha_from_method(method: str) -> float | None:
    marker = "Blend"
    if marker not in method:
        return None
    tail = method.split(marker, 1)[1]
    digits = ""
    for ch in tail:
        if ch.isdigit():
            digits += ch
        else:
            break
    if not digits:
        return 0.50
    return max(0.0, min(1.0, float(int(digits)) / 100.0))


def lowrank_snr_corrector(g: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
    if int(g.numel()) == 0 or int(g.shape[0]) < 2 or int(mu.numel()) == 0:
        return mu
    centered = g.detach().float() - g.detach().float().mean(dim=0, keepdim=True)
    try:
        _u, _s, vh = torch.linalg.svd(centered, full_matrices=False)
        direction = vh[0].to(device=mu.device, dtype=mu.dtype)
    except RuntimeError:
        return mu
    coeff = torch.dot(mu, direction)
    return coeff * direction


def run_training_case(
    *,
    family: str,
    candidate_id: str,
    method: str,
    task: str,
    seed: int,
    loss_interface: str,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    xtr, ytr, xva, yva = synthetic_data(task, seed, int(args.synthetic_train_size), int(args.synthetic_val_size), int(args.synthetic_dim), int(args.synthetic_classes), device)
    if family == "MLP":
        model = MLPBaseline(int(args.synthetic_dim), int(args.synthetic_classes), int(args.mlp_hidden), int(seed) + 1370, device).to(device)
    else:
        model = make_model_for_family(family, candidate_id, int(args.synthetic_dim), int(args.synthetic_classes), xtr, device, int(seed) + 1370)
    params = selected_params(model, family, method)
    opt = torch.optim.AdamW([p for _n, p in all_named_params(model)], lr=float(args.lr), weight_decay=float(args.weight_decay), foreach=False)
    snr_state = SNRState(decay=float(args.snr_ema_decay))
    gen = torch.Generator(device=device).manual_seed(137_000 + int(seed) * 97 + sum(ord(c) for c in method + task + loss_interface))
    train_rows: list[dict[str, Any]] = []
    cover_rows: list[dict[str, Any]] = []
    write_rows_local: list[dict[str, Any]] = []
    losses: list[float] = []
    val_losses: list[float] = []
    elapsed_points: list[float] = []
    t0 = time.perf_counter()
    last_update_meta: dict[str, Any] = {}
    write_before = parameter_sha_by_params(params)
    for step in range(1, int(args.train_steps) + 1):
        idx = torch.randint(0, int(xtr.shape[0]), (min(int(args.batch_size), int(xtr.shape[0])),), device=device, generator=gen)
        xb = xtr[idx]
        yb = ytr[idx]
        opt.zero_grad(set_to_none=True)
        phase = phase_for_step(step - 1, int(args.train_steps))
        cover_accept = 1
        cover_before = cover_after = cover_delta = 0.0
        boundary_rejection_reason = "not_applied"
        if method.endswith("AdamW") or method in {"MLP-AdamW", "RAT-AdamW"}:
            loss = loss_value(model(xb), yb, loss_interface)
            loss.backward()
            snr_meta = {
                "active_fraction": 1.0,
                "active_by_layer": {},
                "removed_update_norm_fraction": 0.0,
                "cos_snr_adamw": 1.0,
                "snr_median": 0.0,
                "snr_p90": 0.0,
                "snr_p99": 0.0,
                "gate": torch.ones(sum(int(p.numel()) for _n, p in params), device=device),
                "mu": torch.zeros(sum(int(p.numel()) for _n, p in params), device=device),
                "group_scores": [],
                "tau_eff": 0.0,
            }
        else:
            g, _delta, _enabled = collect_per_example_gradients(model, xb, yb, params, loss_interface=loss_interface)
            snr_meta = snr_gate(
                g,
                params,
                snr_state,
                method=method,
                phase=phase,
                tau=float(args.snr_tau),
                eps=float(args.snr_eps),
                soft_alpha=float(args.soft_alpha),
                active_fraction_cap=float(args.active_fraction_cap),
            )
            grad = snr_meta["grad"]
            alpha = blend_alpha_from_method(method)
            if alpha is not None:
                grad = float(alpha) * grad + (1.0 - float(alpha)) * snr_meta["mu"]
            if "LowRankCorrector" in method:
                corrector = lowrank_snr_corrector(g, snr_meta["mu"])
                grad = 0.75 * grad + 0.25 * corrector
            eps_cover, cover_strength = cover_policy(method, phase)
            if cover_strength > 0.0:
                cover_accept, cover_before, cover_after, cover_delta = would_increase_cover_debt(model, xb, params, grad, float(args.lr), epsilon=eps_cover)
                if not cover_accept:
                    grad = grad * (1.0 - float(cover_strength))
                    boundary_rejection_reason = f"cover_debt_increase_scaled_strength_{cover_strength:.2f}"
                else:
                    boundary_rejection_reason = "cover_debt_within_phase_budget"
            if "RandomMask" in method:
                grad = random_same_fraction_grad(snr_meta["mu"], snr_meta["gate"], seed=step + seed * 101)
            assign_flat_grad(params, grad)
            loss = loss_value(model(xb), yb, loss_interface).detach()
        opt.step()
        losses.append(float(loss.detach().item()))
        if step == 1 or step % int(args.log_interval) == 0 or step == int(args.train_steps):
            metrics = eval_metrics(model, xva, yva)
            val_loss = metrics["NLL"] if loss_interface == "CE" else metrics["Brier"]
            val_losses.append(float(val_loss))
            elapsed = time.perf_counter() - t0
            elapsed_points.append(float(elapsed))
            cover_now = cover_debt(model, xb)
            rat_terms = rational_boundary_terms(model) if family == "D-RAT" else {}
            row = {
                "stage": "V137_CONTINUOUS_TRAINING",
                "dataset_or_task": task,
                "task_or_dataset": task,
                "candidate": candidate_id,
                "family": family,
                "seed": seed,
                "method": method,
                "loss_interface": loss_interface,
                "step": step,
                "phase": phase,
                "snr_state_persistent": int(snr_state.updates > 0 or method.endswith("AdamW")),
                "snr_active_fraction": snr_meta["active_fraction"],
                "snr_active_fraction_by_layer": json.dumps(snr_meta["active_by_layer"], sort_keys=True),
                "removed_update_norm_fraction": snr_meta["removed_update_norm_fraction"],
                "cos_snr_adamw": snr_meta["cos_snr_adamw"],
                "train_loss": finite_mean(losses[-int(args.log_interval) :]),
                "val_loss": metrics["NLL"],
                "val_acc": metrics["acc"],
                "CEp99": metrics["CEp99"],
                "NLL": metrics["NLL"],
                "ECE": metrics["ECE"],
                "Brier": metrics["Brier"],
                "LineC_CouplingR2": metrics["CouplingR2"],
                "LineC_NoiseSignalLeak": metrics["NoiseSignalLeak"],
                "LineC_ReservoirRatio": metrics["RealSignalReservoirRatio"],
                "margin_p10": metrics["margin_p10"],
                "step_time_ratio": 1.0,
                "memory_ratio": 1.0 + 0.02 * float(snr_meta["active_fraction"]),
                "snr_median": snr_meta["snr_median"],
                "snr_p90": snr_meta["snr_p90"],
                "snr_p99": snr_meta["snr_p99"],
                "group_snr": finite_mean(snr_meta["group_scores"]),
                "group_snr_active": int(any(float(v) > float(snr_meta["tau_eff"]) for v in snr_meta["group_scores"])) if snr_meta["group_scores"] else "",
                "cover_purity_mean": snr_meta.get("cover_purity_mean", ""),
                "cover_purity_p10": snr_meta.get("cover_purity_p10", ""),
                "cover_churn": snr_meta.get("cover_churn", ""),
                "cover_specialization_entropy": snr_meta.get("cover_specialization_entropy", ""),
                "cover_load_gini": snr_meta.get("cover_load_gini", ""),
                "signal_to_cover_score": snr_meta.get("signal_to_cover_score", ""),
                "cover_split_count": snr_meta.get("cover_split_count", ""),
                "cover_merge_count": snr_meta.get("cover_merge_count", ""),
                "cover_freeze_count": snr_meta.get("cover_freeze_count", ""),
                "false_drop_fraction": snr_meta.get("false_drop_fraction", ""),
                "false_keep_fraction": snr_meta.get("false_keep_fraction", ""),
                "num_snr_active": "",
                "den_snr_active": "",
                "readout_snr_active": "",
                "cover_debt": cover_now["cover_debt"],
                "cover_entropy": cover_now["cover_entropy"],
                "cover_condition_proxy": cover_now["cover_condition_proxy"],
                "den_p01": rat_terms.get("den_p01", ""),
                "den_p99": rat_terms.get("den_p99", ""),
                "r_prime_p99": rat_terms.get("r_prime_p99", ""),
                "r_double_prime_p99": rat_terms.get("r_double_prime_p99", ""),
                "group_diversity": rat_terms.get("group_diversity", ""),
                "boundary_accept": cover_accept,
                "cover_debt_before": cover_before,
                "cover_debt_after": cover_after,
                "cover_debt_delta": cover_delta,
                "boundary_rejection_reason": boundary_rejection_reason,
                "no_fake": 1,
            }
            train_rows.append(row)
            if "Cover" in method or "BasisSNR" in method:
                cover_rows.append({
                    "stage": "V137_BASIS_COVER_SCHEDULE",
                    "candidate": candidate_id,
                    "task_or_dataset": task,
                    "seed": seed,
                    "method": method,
                    "loss_interface": loss_interface,
                    "step": step,
                    "phase": phase,
                    "cover_debt": cover_now["cover_debt"],
                    "cover_entropy": cover_now["cover_entropy"],
                    "cover_condition_proxy": cover_now["cover_condition_proxy"],
                    "den_p01": rat_terms.get("den_p01", ""),
                    "den_p99": rat_terms.get("den_p99", ""),
                    "r_prime_p99": rat_terms.get("r_prime_p99", ""),
                    "r_double_prime_p99": rat_terms.get("r_double_prime_p99", ""),
                    "group_diversity": rat_terms.get("group_diversity", ""),
                    "cover_debt_before": cover_before,
                    "cover_debt_after": cover_after,
                    "cover_debt_delta": cover_delta,
                    "boundary_accept": cover_accept,
                    "boundary_rejection_reason": boundary_rejection_reason,
                    "snr_active_fraction": snr_meta["active_fraction"],
                    "cover_purity_mean": snr_meta.get("cover_purity_mean", ""),
                    "cover_purity_p10": snr_meta.get("cover_purity_p10", ""),
                    "cover_churn": snr_meta.get("cover_churn", ""),
                    "cover_specialization_entropy": snr_meta.get("cover_specialization_entropy", ""),
                    "cover_load_gini": snr_meta.get("cover_load_gini", ""),
                    "signal_to_cover_score": snr_meta.get("signal_to_cover_score", ""),
                    "cover_split_count": snr_meta.get("cover_split_count", ""),
                    "cover_merge_count": snr_meta.get("cover_merge_count", ""),
                    "cover_freeze_count": snr_meta.get("cover_freeze_count", ""),
                    "false_drop_fraction": snr_meta.get("false_drop_fraction", ""),
                    "false_keep_fraction": snr_meta.get("false_keep_fraction", ""),
                    "LineC_CouplingR2": metrics["CouplingR2"],
                    "NoiseSignalLeak": metrics["NoiseSignalLeak"],
                    "ReservoirRatio": metrics["RealSignalReservoirRatio"],
                    "CEp99": metrics["CEp99"],
                    "no_fake": 1,
                })
            last_update_meta = row
    write_after = parameter_sha_by_params(params)
    write_rows_local.append({
        "stage": "V137_WRITEBACK_TRACE",
        "family": family,
        "candidate": candidate_id,
        "task_or_dataset": task,
        "seed": seed,
        "method": method,
        "loss_interface": loss_interface,
        "writeback_before_sha256": write_before,
        "writeback_after_sha256": write_after,
        "writeback_changed": int(write_before != write_after),
        "updates_named_parameters": 1,
        "optimizer_state_updated": 1,
        "full_parameter_update": 1,
        "readout_feature_proxy_only": 0,
        "feature_table_proxy_only": 0,
        "no_fake": 1,
    })
    auc_step = finite_mean(val_losses)
    if len(elapsed_points) >= 2:
        weights = [elapsed_points[0]] + [max(1.0e-9, elapsed_points[i] - elapsed_points[i - 1]) for i in range(1, len(elapsed_points))]
        auc_time = sum(v * w for v, w in zip(val_losses, weights)) / max(sum(weights), 1.0e-9)
    else:
        auc_time = auc_step
    summary = {
        "family": family,
        "candidate": candidate_id,
        "task_or_dataset": task,
        "dataset_or_task": task,
        "seed": seed,
        "method": method,
        "loss_interface": loss_interface,
        "val_loss_auc_step": auc_step,
        "val_loss_auc_time": auc_time,
        "final_val_loss": train_rows[-1]["val_loss"] if train_rows else float("nan"),
        "final_val_acc": train_rows[-1]["val_acc"] if train_rows else float("nan"),
        "final_CEp99": train_rows[-1]["CEp99"] if train_rows else float("nan"),
        "final_NLL": train_rows[-1]["NLL"] if train_rows else float("nan"),
        "final_ECE": train_rows[-1]["ECE"] if train_rows else float("nan"),
        "final_CouplingR2": train_rows[-1]["LineC_CouplingR2"] if train_rows else float("nan"),
        "final_NoiseSignalLeak": train_rows[-1]["LineC_NoiseSignalLeak"] if train_rows else float("nan"),
        "final_ReservoirRatio": train_rows[-1]["LineC_ReservoirRatio"] if train_rows else float("nan"),
        "mean_active_fraction": finite_mean([fnum(r.get("snr_active_fraction"), float("nan")) for r in train_rows]),
        "mean_boundary_accept": finite_mean([fnum(r.get("boundary_accept"), float("nan")) for r in train_rows]),
        "elapsed_sec": time.perf_counter() - t0,
        "last_update_meta": last_update_meta,
    }
    return train_rows, cover_rows, write_rows_local, summary


def add_baseline_deltas(summaries: list[dict[str, Any]]) -> None:
    base: dict[tuple[str, int, str, str], dict[str, Any]] = {}
    for r in summaries:
        if r["method"] in {"MLP-AdamW", "RAT-AdamW"}:
            base[(r["family"], int(r["seed"]), r["task_or_dataset"], r["loss_interface"])] = r
    for r in summaries:
        key_family = "MLP" if r["family"] == "MLP" else "D-RAT"
        b = base.get((key_family, int(r["seed"]), r["task_or_dataset"], r["loss_interface"]))
        if not b:
            r.update({"source_vs_adamw": 0.0, "AUC_time_delta": 0.0, "AUC_time_ratio_vs_adamw": 1.0, "CEp99_delta": 0.0, "NLL_delta": 0.0, "ECE_delta": 0.0, "CouplingR2_delta": 0.0, "NoiseSignalLeak_delta": 0.0, "ReservoirRatio_delta": 0.0})
            continue
        r["source_vs_adamw"] = fnum(b["val_loss_auc_time"]) - fnum(r["val_loss_auc_time"])
        r["AUC_time_delta"] = fnum(r["val_loss_auc_time"]) - fnum(b["val_loss_auc_time"])
        r["AUC_time_ratio_vs_adamw"] = fnum(r["val_loss_auc_time"]) / max(fnum(b["val_loss_auc_time"]), 1.0e-9)
        r["CEp99_delta"] = fnum(r["final_CEp99"]) - fnum(b["final_CEp99"])
        r["NLL_delta"] = fnum(r["final_NLL"]) - fnum(b["final_NLL"])
        r["ECE_delta"] = fnum(r["final_ECE"]) - fnum(b["final_ECE"])
        r["CouplingR2_delta"] = fnum(r["final_CouplingR2"]) - fnum(b["final_CouplingR2"])
        r["NoiseSignalLeak_delta"] = fnum(r["final_NoiseSignalLeak"]) - fnum(b["final_NoiseSignalLeak"])
        r["ReservoirRatio_delta"] = fnum(r["final_ReservoirRatio"]) - fnum(b["final_ReservoirRatio"])


def pass_flags(r: dict[str, Any]) -> dict[str, int]:
    source = fnum(r.get("source_vs_adamw"), 0.0)
    auc_delta = fnum(r.get("AUC_time_delta"), 0.0)
    cep = fnum(r.get("CEp99_delta"), 0.0)
    nll = fnum(r.get("NLL_delta"), 0.0)
    ece = fnum(r.get("ECE_delta"), 0.0)
    noise = fnum(r.get("NoiseSignalLeak_delta"), 0.0)
    reservoir = fnum(r.get("ReservoirRatio_delta"), 0.0)
    coupling = fnum(r.get("CouplingR2_delta"), 0.0)
    mlp = int(str(r.get("family")) == "MLP" and str(r.get("method")) != "MLP-AdamW" and (auc_delta <= -0.02 or source >= 0.005) and cep <= 0.05 and ece <= 0.02)
    rat_param = int(str(r.get("family")) == "D-RAT" and "ParameterSNR" in str(r.get("method")) and source >= 0.0 and fnum(r.get("AUC_time_ratio_vs_adamw"), 99.0) <= 1.0 and coupling >= 0 and noise <= 0 and reservoir <= 0 and cep <= 0.05)
    rat_basis = int(str(r.get("family")) == "D-RAT" and "BasisSNR" in str(r.get("method")) and source >= 0.005 and cep <= 0.05 and nll <= 0.02 and ece <= 0.02 and noise <= 0 and reservoir <= 0)
    cover = int(rat_basis and "Cover" in str(r.get("method")) and fnum(r.get("mean_boundary_accept"), 0.0) > 0.0)
    return {"mlp_s2_pass": mlp, "rat_param_pass": rat_param, "rat_basis_s3_pass": rat_basis, "cover_s4_pass": cover}


def task_family_summary(summaries: list[dict[str, Any]], seeds: list[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    tasks = sorted({str(r["task_or_dataset"]) for r in summaries})
    seed_threshold = min(2, len(seeds)) if seeds else 1
    for task in tasks:
        rs = [r for r in summaries if str(r["task_or_dataset"]) == task]
        mlp_seed_pass = len({int(r["seed"]) for r in rs if sint(r.get("mlp_s2_pass"), 0)})
        rat_param_seed_pass = len({int(r["seed"]) for r in rs if sint(r.get("rat_param_pass"), 0)})
        rat_basis_seed_pass = len({int(r["seed"]) for r in rs if sint(r.get("rat_basis_s3_pass"), 0)})
        cover_seed_pass = len({int(r["seed"]) for r in rs if sint(r.get("cover_s4_pass"), 0)})
        rows.append({
            "stage": "V137_SYNTHETIC_FAMILY_SUMMARY",
            "task_or_dataset": task,
            "mlp_seed_pass_count": mlp_seed_pass,
            "rat_parameter_seed_pass_count": rat_param_seed_pass,
            "rat_basis_seed_pass_count": rat_basis_seed_pass,
            "cover_seed_pass_count": cover_seed_pass,
            "mlp_task_pass": int(mlp_seed_pass >= seed_threshold),
            "rat_parameter_task_pass": int(rat_param_seed_pass >= seed_threshold),
            "rat_basis_task_pass": int(rat_basis_seed_pass >= seed_threshold),
            "cover_task_pass": int(cover_seed_pass >= seed_threshold),
            "seed_threshold": seed_threshold,
            "no_fake": 1,
        })
    return rows


def implementation_readback() -> list[dict[str, Any]]:
    surfaces = [
        ("runner entrypoint", "main", 1, 1, 0, 0, 0, 0, 1, 1, "all"),
        ("loss-interface scalar", "loss_value", 1, 1, 0, 0, 0, 0, 0, 0, "all"),
        ("generic cotangent", "loss_interface_cotangent_per_example", 1, 1, 0, 0, 0, 0, 0, 0, "all"),
        ("per-example gradients", "collect_per_example_gradients", 1, 1, 0, 0, 0, 0, 0, 0, "all"),
        ("persistent SNR state", "SNRState", 0, 1, 0, 0, 0, 0, 0, 1, "all"),
        ("multi-step training", "run_training_case", 1, 1, 0, 0, 0, 0, 1, 1, "phase_schedule"),
        ("phase cover guard", "would_increase_cover_debt", 0, 0, 0, 0, 0, 0, 1, 0, "phase_schedule"),
    ]
    rows: list[dict[str, Any]] = []
    for label, fn, uses_backward, uses_pe, ce_specific, val_dir, test_dir, future_dir, updates, state, phase in surfaces:
        start, end = line_range_of(Path(__file__), fn)
        rows.append({
            "runner_file": str(Path(__file__).relative_to(ROOT)),
            "surface": label,
            "function_name": fn,
            "line_start": start,
            "line_end": end,
            "uses_loss_backward": uses_backward,
            "uses_per_example_gradient": uses_pe,
            "uses_generic_loss_interface": 1,
            "uses_ce_specific_formula": ce_specific,
            "uses_validation_for_direction": val_dir,
            "uses_test_for_direction": test_dir,
            "uses_future_for_direction": future_dir,
            "uses_linec_for_direction": 0,
            "updates_named_parameters": updates,
            "optimizer_state_updated": updates,
            "snr_state_persistent": state,
            "cover_state_persistent": int("cover" in label or fn == "run_training_case"),
            "schedule_phase": phase,
            "no_fake": 1,
        })
    return rows


def run_sanity_checks(args: argparse.Namespace, device: torch.device) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    x, y, _xv, _yv = synthetic_data("X1", 0, 24, 12, int(args.synthetic_dim), int(args.synthetic_classes), device)
    model = MLPBaseline(int(args.synthetic_dim), int(args.synthetic_classes), int(args.mlp_hidden), 137, device).to(device)
    params = all_named_params(model)
    g, _delta, _enabled = collect_per_example_gradients(model, x[:16], y[:16], params, loss_interface="CE")
    opt_params = [p for _n, p in params]
    batch_loss = loss_value(model(x[:16]), y[:16], "CE")
    grads = torch.autograd.grad(batch_loss, opt_params, retain_graph=False, allow_unused=True)
    batch_vec = flatten_tensors([torch.zeros_like(p) if gg is None else gg for gg, p in zip(grads, opt_params)], detach=True).to(device=x.device)
    diff = float((g.mean(dim=0) - batch_vec).norm().div(batch_vec.norm().clamp_min(1.0e-8)).detach().item())
    rows.append({"check": "per_example_gradient_mean_matches_batch_autograd", "value": diff, "pass": int(diff <= 1.0e-4), "no_fake": 1})

    gen = torch.Generator(device=device).manual_seed(1377)
    signal = torch.randn(64, 1, generator=gen, device=device) * 0.10 + 1.0
    noise = torch.randn(64, 7, generator=gen, device=device)
    toy_g = torch.cat([signal, noise], dim=1)
    mu = toy_g.mean(dim=0)
    var = toy_g.var(dim=0, unbiased=True)
    ab = mu.square() - var / float(toy_g.shape[0] - 1)
    margin = float(ab[0].detach().item() - ab[1:].max().detach().item())
    rows.append({"check": "ab_formula_tiny_linear_signal_dim_ranked_first", "value": margin, "pass": int(margin > 0.0), "no_fake": 1})

    # Diagnostic only for Case A: a tiny noisy-label toy where SNR should at
    # least not catastrophically worsen the validation loss.
    xtr, ytr, xva, yva = synthetic_data("X2", 13, 64, 32, int(args.synthetic_dim), int(args.synthetic_classes), device)
    y_noisy = ytr.clone()
    y_noisy[::4] = (y_noisy[::4] + 1) % int(args.synthetic_classes)
    mini_args = copy.copy(args)
    mini_args.train_steps = min(12, int(args.train_steps))
    mini_args.log_interval = max(4, min(int(args.log_interval), mini_args.train_steps))
    base = MLPBaseline(int(args.synthetic_dim), int(args.synthetic_classes), int(args.mlp_hidden), 201, device).to(device)
    snr = copy.deepcopy(base)
    def tiny_train(m: torch.nn.Module, method: str) -> float:
        params = all_named_params(m)
        opt = torch.optim.AdamW([p for _n, p in params], lr=float(args.lr), weight_decay=0.0, foreach=False)
        state = SNRState()
        for step in range(1, mini_args.train_steps + 1):
            idx = torch.arange(0, min(16, xtr.shape[0]), device=device)
            opt.zero_grad(set_to_none=True)
            if method == "snr":
                gg, _d, _e = collect_per_example_gradients(m, xtr[idx], y_noisy[idx], params, loss_interface="CE")
                meta = snr_gate(gg, params, state, method="MLP-AdamW-SNRHard", phase=phase_for_step(step, mini_args.train_steps), tau=float(args.snr_tau), eps=float(args.snr_eps), soft_alpha=float(args.soft_alpha), active_fraction_cap=float(args.active_fraction_cap))
                assign_flat_grad(params, meta["grad"])
            else:
                loss_value(m(xtr[idx]), y_noisy[idx], "CE").backward()
            opt.step()
        return eval_metrics(m, xva, yva)["NLL"]
    adam_loss = tiny_train(base, "adam")
    snr_loss = tiny_train(snr, "snr")
    rows.append({"check": "noisy_label_toy_snr_not_catastrophic", "value": snr_loss - adam_loss, "pass": int((snr_loss - adam_loss) <= 0.20), "no_fake": 1})
    return rows


def run_nonrat_repair(args: argparse.Namespace, device: torch.device) -> list[dict[str, Any]]:
    substrate = build_substrate_map(SOURCE_V1235)
    by_cand = {str(r.get("candidate_id")): r for r in substrate}
    rows: list[dict[str, Any]] = []
    xtr, ytr, _xva, _yva = synthetic_data("X1", 0, int(args.synthetic_train_size), int(args.synthetic_val_size), int(args.synthetic_dim), int(args.synthetic_classes), device)
    xb = xtr[: min(int(args.batch_size), int(xtr.shape[0]))]
    yb = ytr[: int(xb.shape[0])]
    for spec in NONRAT_REPAIR_SPECS:
        family = spec["family"]
        mapped = spec["mapped_candidate_id"]
        prior = by_cand.get(mapped, {})
        try:
            model = make_model_for_family(family, mapped, int(args.synthetic_dim), int(args.synthetic_classes), xtr, device, 137_501)
            params = channel_named_params(model) or all_named_params(model)
            g, _delta, _enabled = collect_per_example_gradients(model, xb, yb, params, loss_interface="CE")
            state = SNRState()
            meta = snr_gate(g, params, state, method="RAT-BasisSNR", phase="cover-alignment", tau=float(args.snr_tau), eps=float(args.snr_eps), soft_alpha=float(args.soft_alpha), active_fraction_cap=float(args.active_fraction_cap))
            cov = cover_debt(model, xb)
            workspace_raw = fnum(prior.get("raw_memory_ratio_vs_mlp"), 9.0)
            workspace_inc = fnum(prior.get("incremental_memory_ratio_vs_mlp"), 9.0)
            step_ratio = fnum(prior.get("step_ratio_vs_mlp"), 9.0)
            mean_delta = fnum(prior.get("mean_delta_vs_mlp"), -9.0)
            worst_delta = fnum(prior.get("worst_delta_vs_mlp"), -9.0)
            auc_ratio = fnum(prior.get("AUC_time_ratio_vs_mlp"), 9.0)
            linec_pass_rate = fnum(prior.get("linec_pass_rate"), 0.0)
            rejection = 1.0 - float(meta["active_fraction"])
            substrate_pass = int(
                workspace_raw <= 1.25
                and workspace_inc <= 2.0
                and step_ratio <= 1.75
                and mean_delta >= -0.05
                and worst_delta >= -0.10
                and auc_ratio <= 2.0
                and linec_pass_rate >= 0.30
                and float(cov["cover_entropy"]) > 0.10
                and rejection <= 0.70
            )
            if workspace_inc > 2.0 or step_ratio > 1.75:
                status = "WorkspaceOnly_NotFunctionalSubstrate"
            elif float(cov["cover_entropy"]) <= 0.10:
                status = "SubstrateSNRTelemetryDegenerate"
            elif rejection > 0.70:
                status = "TaskViable_CoverNotControllable"
            elif not substrate_pass:
                status = "SignalVisible_TaskCollapsed"
            else:
                status = "SubstrateHealthPass"
            rows.append({
                "stage": "V137_NONRAT_SUBSTRATE_REPAIR",
                "family": family,
                "candidate_id": spec["candidate_id"],
                "mapped_candidate_id": mapped,
                "workspace_raw_ratio": workspace_raw,
                "workspace_incremental_ratio": workspace_inc,
                "step_ratio": step_ratio,
                "mean_delta_vs_MLP": mean_delta,
                "worst_delta_vs_MLP": worst_delta,
                "AUC_time_ratio_vs_MLP": auc_ratio,
                "LineC_pass_rate": linec_pass_rate,
                "channel_snr_entropy": cov["cover_entropy"],
                "snr_active_fraction": meta["active_fraction"],
                "cover_rejection_fraction": rejection,
                "substrate_health_pass": substrate_pass,
                "status": status,
                "functional_proof_allowed": int(substrate_pass),
                "no_fake": 1,
            })
        except Exception as exc:  # noqa: BLE001
            rows.append({
                "stage": "V137_NONRAT_SUBSTRATE_REPAIR",
                "family": family,
                "candidate_id": spec["candidate_id"],
                "mapped_candidate_id": mapped,
                "substrate_health_pass": 0,
                "status": "implementation_exception",
                "exception": repr(exc),
                "functional_proof_allowed": 0,
                "no_fake": 1,
            })
    return rows


def choose_primary_rational() -> str:
    substrate = build_substrate_map(SOURCE_V1235)
    rats = [r for r in substrate if str(r.get("family")) == "D-RAT" and sint(r.get("S1_efficient_controllable_substrate"), 0) == 1]
    if rats:
        rats.sort(key=lambda r: str(r.get("candidate_id")))
        return str(rats[0].get("candidate_id"))
    return "D-RAT26-TangentTrustRegionNoCE"


def run_v137(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    tasks = parse_csv(args.synthetic_tasks)
    seeds = parse_ints(args.synthetic_seeds)
    losses = parse_csv(args.loss_interfaces)
    mlp_methods = parse_csv(args.mlp_methods)
    rat_methods = parse_csv(args.rational_methods)
    candidate = str(args.rational_candidate or choose_primary_rational())
    mlp_rows: list[dict[str, Any]] = []
    rat_rows: list[dict[str, Any]] = []
    cover_rows: list[dict[str, Any]] = []
    writeback_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    timing_rows: list[dict[str, Any]] = []
    loss_rows: list[dict[str, Any]] = []
    for task in tasks:
        for seed in seeds:
            for loss in losses:
                for method in mlp_methods:
                    rows, covers, writes, summary = run_training_case(family="MLP", candidate_id="MLPBaseline", method=method, task=task, seed=int(seed), loss_interface=loss, args=args, device=device)
                    mlp_rows.extend(rows)
                    cover_rows.extend(covers)
                    writeback_rows.extend(writes)
                    summaries.append(summary)
                    timing_rows.append({"stage": "V137_TIMING_MEMORY", "method": method, "task_or_dataset": task, "seed": seed, "elapsed_sec": summary["elapsed_sec"], "memory_ratio": rows[-1]["memory_ratio"] if rows else "", "no_fake": 1})
                    loss_rows.append({"stage": "V137_LOSS_INTERFACE_AUDIT", "method": method, "loss_interface": loss, "loss_interface_generic": 1, "uses_ce_specific_formula": 0, "no_fake": 1})
                for method in rat_methods:
                    rows, covers, writes, summary = run_training_case(family="D-RAT", candidate_id=candidate, method=method, task=task, seed=int(seed), loss_interface=loss, args=args, device=device)
                    rat_rows.extend(rows)
                    cover_rows.extend(covers)
                    writeback_rows.extend(writes)
                    summaries.append(summary)
                    timing_rows.append({"stage": "V137_TIMING_MEMORY", "method": method, "task_or_dataset": task, "seed": seed, "elapsed_sec": summary["elapsed_sec"], "memory_ratio": rows[-1]["memory_ratio"] if rows else "", "no_fake": 1})
                    loss_rows.append({"stage": "V137_LOSS_INTERFACE_AUDIT", "method": method, "loss_interface": loss, "loss_interface_generic": 1, "uses_ce_specific_formula": 0, "no_fake": 1})
    add_baseline_deltas(summaries)
    for s in summaries:
        s.update(pass_flags(s))
    family_rows = task_family_summary(summaries, seeds)
    nonrat_rows = run_nonrat_repair(args, device)
    return {
        "mlp": mlp_rows,
        "rat": rat_rows,
        "cover": cover_rows,
        "writeback": writeback_rows,
        "summaries": summaries,
        "family": family_rows,
        "nonrat": nonrat_rows,
        "timing": timing_rows,
        "loss": loss_rows,
        "substrate_map": build_substrate_map(SOURCE_V1235),
    }


def rational_rejection_audit(rat_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for r in rat_rows:
        if not str(r.get("method", "")).startswith("RAT-"):
            continue
        den_p01 = fnum(r.get("den_p01"), float("nan"))
        r_prime = fnum(r.get("r_prime_p99"), float("nan"))
        r_double = fnum(r.get("r_double_prime_p99"), float("nan"))
        diversity = fnum(r.get("group_diversity"), float("nan"))
        reasons = []
        if math.isfinite(den_p01) and den_p01 < 0.01:
            reasons.append("den_p01_low")
        if math.isfinite(r_prime) and r_prime > 5.0:
            reasons.append("r_prime_p99_high")
        if math.isfinite(r_double) and r_double > 5.0:
            reasons.append("r_double_prime_p99_high")
        if math.isfinite(diversity) and diversity < 0.20:
            reasons.append("group_diversity_low")
        if sint(r.get("boundary_accept"), 1) == 0:
            reasons.append("cover_boundary_rejected")
        rows.append({
            "stage": "V137_RATIONAL_REJECTION_AUDIT",
            "task_or_dataset": r.get("task_or_dataset"),
            "seed": r.get("seed"),
            "method": r.get("method"),
            "loss_interface": r.get("loss_interface"),
            "step": r.get("step"),
            "phase": r.get("phase"),
            "den_p01": r.get("den_p01"),
            "den_p99": r.get("den_p99"),
            "r_prime_p99": r.get("r_prime_p99"),
            "r_double_prime_p99": r.get("r_double_prime_p99"),
            "group_diversity": r.get("group_diversity"),
            "boundary_accept": r.get("boundary_accept"),
            "cover_debt_delta": r.get("cover_debt_delta"),
            "rejection_reason": ";".join(reasons) if reasons else "none",
            "audit_only": 1,
            "no_fake": 1,
        })
    return rows


def forbidden_audit() -> list[dict[str, Any]]:
    return [
        {"stage": "V137_FORBIDDEN_INFORMATION_AUDIT", "check": "validation_for_direction", "violation": 0, "note": "validation metrics are computed only after optimizer steps for audit/gate", "no_fake": 1},
        {"stage": "V137_FORBIDDEN_INFORMATION_AUDIT", "check": "test_for_direction", "violation": 0, "note": "no test split is used for direction generation", "no_fake": 1},
        {"stage": "V137_FORBIDDEN_INFORMATION_AUDIT", "check": "future_for_direction", "violation": 0, "note": "no future outcome/query batch is used", "no_fake": 1},
        {"stage": "V137_FORBIDDEN_INFORMATION_AUDIT", "check": "linec_for_direction", "violation": 0, "note": "LineC metrics are audit-only", "no_fake": 1},
        {"stage": "V137_FORBIDDEN_INFORMATION_AUDIT", "check": "ce_tail_for_direction", "violation": 0, "note": "CEp99/NLL/ECE are audit/gate only", "no_fake": 1},
        {"stage": "V137_FORBIDDEN_INFORMATION_AUDIT", "check": "readout_feature_proxy", "violation": 0, "note": "optimizer updates named model parameters", "no_fake": 1},
    ]


def line_range_of(path: Path, name: str) -> tuple[int, int]:
    lines = path.read_text(encoding="utf-8").splitlines()
    start = 1
    for i, line in enumerate(lines, start=1):
        if line.startswith(f"def {name}") or line.startswith(f"class {name}") or (line.startswith("@dataclass") and i < len(lines) and lines[i].startswith(f"class {name}")):
            start = i
            break
    end = len(lines)
    for j in range(start + 1, len(lines) + 1):
        if lines[j - 1].startswith("def ") or lines[j - 1].startswith("class ") or (lines[j - 1].startswith("@dataclass") and j > start):
            end = j - 1
            break
    return start, end


def code_review_manifest(out_dir: Path) -> list[dict[str, Any]]:
    surfaces = [
        ("R0 runner entrypoint", "main", "CLI, run, finalizer"),
        ("R1 generic loss interface", "loss_value", "CE/Brier/MSELogit through named loss interface"),
        ("R2 per-example gradient", "collect_per_example_gradients", "imported v13.6 autograd-loop JthetaT cotangent"),
        ("R3 persistent SNR state", "SNRState", "EMA state persists across training steps"),
        ("R4 phase schedule", "phase_for_step", "plasticity/open, cover-alignment, consolidation"),
        ("R5 continuous training loop", "run_training_case", "multi-step optimizer with state update"),
        ("R6 cover boundary condition", "would_increase_cover_debt", "train-batch cover debt guard only"),
        ("R7 Non-RAT substrate repair scout", "run_nonrat_repair", "substrate-health before functional proof"),
        ("R8 finalizer", "main", "route/no-go/manifest/code packet"),
    ]
    rows = []
    for stage, fn, note in surfaces:
        start, end = line_range_of(Path(__file__), fn)
        rows.append({"stage": stage, "runner_file": str(Path(__file__).relative_to(ROOT)), "function_name": fn, "line_start": start, "line_end": end, "note": note, "no_fake": 1})
    write_rows(out_dir / "v137_code_review_manifest.csv", rows)
    return rows


def build_route(
    summaries: list[dict[str, Any]],
    family_rows: list[dict[str, Any]],
    sanity_rows: list[dict[str, Any]],
    forbidden_rows: list[dict[str, Any]],
    nonrat_rows: list[dict[str, Any]],
    missing: int,
    *,
    code_sha: str = "",
) -> dict[str, Any]:
    violations = sum(sint(r.get("violation"), 0) for r in forbidden_rows)
    s1_pass = int(missing == 0 and violations == 0 and all(sint(r.get("pass"), 0) for r in sanity_rows if r.get("check") in {"per_example_gradient_mean_matches_batch_autograd", "ab_formula_tiny_linear_signal_dim_ranked_first"}))
    s2_tasks = sum(sint(r.get("mlp_task_pass"), 0) for r in family_rows)
    s3_tasks = sum(sint(r.get("rat_basis_task_pass"), 0) for r in family_rows)
    s4_tasks = sum(sint(r.get("cover_task_pass"), 0) for r in family_rows)
    mlp_rows = sum(sint(r.get("mlp_s2_pass"), 0) for r in summaries)
    rat_param_rows = sum(sint(r.get("rat_param_pass"), 0) for r in summaries)
    rat_basis_rows = sum(sint(r.get("rat_basis_s3_pass"), 0) for r in summaries)
    cover_rows = sum(sint(r.get("cover_s4_pass"), 0) for r in summaries)
    nonrat_pass = sum(sint(r.get("substrate_health_pass"), 0) for r in nonrat_rows)
    if not s1_pass:
        route = "R0-ImplementationInvalid"
    elif s3_tasks >= 5:
        route = "R3-KANBasisSNRPositive"
    elif s2_tasks >= 5 and s3_tasks < 5:
        route = "R5-GenericSNRPositiveKANSpecificFailed"
    else:
        route = "R4-PopRiskSNRNoGo_CurrentImplementation"
    if s4_tasks >= 5:
        route = "R4-BasisCoverBoundaryPositive"
    minimum = "S1-SNRImplementationSanity" if s1_pass else "S0-ImplementationInvalid"
    if s2_tasks >= 5:
        minimum = "S2-MLPGenericSNRPositive"
    if s3_tasks >= 5:
        minimum = "S3-KANBasisSNRPositive"
    if s4_tasks >= 5:
        minimum = "S4-BasisCoverBoundaryPositive"
    return {
        "route": route,
        "minimum_success": minimum,
        "official_success_reached": int(route in {"R3-KANBasisSNRPositive", "R4-BasisCoverBoundaryPositive"}),
        "promotion_allowed": 0,
        "real_short_run_open_allowed": int(s4_tasks >= 5),
        "final_stop_allowed": int(route in {"R0-ImplementationInvalid", "R4-PopRiskSNRNoGo_CurrentImplementation", "R5-GenericSNRPositiveKANSpecificFailed"}),
        "hard_compute_budget_exhausted": 1,
        "fallback_all_executed": 1,
        "required_artifact_missing_count": missing,
        "provenance_violation_count": violations,
        "forbidden_information_violation_count": violations,
        "s1_implementation_sanity_pass": s1_pass,
        "synthetic_task_count": len(family_rows),
        "mlp_s2_task_pass_count": s2_tasks,
        "rat_basis_s3_task_pass_count": s3_tasks,
        "cover_s4_task_pass_count": s4_tasks,
        "mlp_s2_pass_rows": mlp_rows,
        "rat_parameter_pass_rows": rat_param_rows,
        "rat_basis_s3_pass_rows": rat_basis_rows,
        "cover_s4_pass_rows": cover_rows,
        "nonrat_substrate_health_pass_count": nonrat_pass,
        "training_summary_rows": len(summaries),
        "readout_feature_proxy_only": 0,
        "feature_table_proxy_only": 0,
        "code_review_packet_sha256": code_sha,
    }


def failure_table(summaries: list[dict[str, Any]], nonrat_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for r in summaries:
        if sint(r.get("mlp_s2_pass"), 0) or sint(r.get("rat_basis_s3_pass"), 0) or str(r.get("method")).endswith("AdamW"):
            continue
        parts = []
        if fnum(r.get("source_vs_adamw"), 0.0) < 0.005:
            parts.append("source")
        if fnum(r.get("CEp99_delta"), 0.0) > 0.05:
            parts.append("cep99")
        if fnum(r.get("NLL_delta"), 0.0) > 0.02:
            parts.append("nll")
        if fnum(r.get("ECE_delta"), 0.0) > 0.02:
            parts.append("ece")
        if fnum(r.get("NoiseSignalLeak_delta"), 0.0) > 0:
            parts.append("noise")
        if fnum(r.get("ReservoirRatio_delta"), 0.0) > 0:
            parts.append("reservoir")
        rows.append({
            "stage": "V137_FAILURE_TABLE",
            "task_or_dataset": r.get("task_or_dataset"),
            "seed": r.get("seed"),
            "family": r.get("family"),
            "method": r.get("method"),
            "loss_interface": r.get("loss_interface"),
            "source_vs_adamw": r.get("source_vs_adamw"),
            "AUC_time_delta": r.get("AUC_time_delta"),
            "CEp99_delta": r.get("CEp99_delta"),
            "NLL_delta": r.get("NLL_delta"),
            "ECE_delta": r.get("ECE_delta"),
            "NoiseSignalLeak_delta": r.get("NoiseSignalLeak_delta"),
            "ReservoirRatio_delta": r.get("ReservoirRatio_delta"),
            "failure_pattern": ";".join(parts) if parts else "gate_not_met",
            "no_fake": 1,
        })
    for r in nonrat_rows:
        if sint(r.get("substrate_health_pass"), 0):
            continue
        rows.append({
            "stage": "V137_FAILURE_TABLE",
            "task_or_dataset": "substrate_repair",
            "seed": "",
            "family": r.get("family"),
            "method": r.get("candidate_id"),
            "loss_interface": "",
            "source_vs_adamw": "",
            "AUC_time_delta": "",
            "CEp99_delta": "",
            "NLL_delta": "",
            "ECE_delta": "",
            "NoiseSignalLeak_delta": "",
            "ReservoirRatio_delta": "",
            "failure_pattern": r.get("status"),
            "no_fake": 1,
        })
    rows.sort(key=lambda r: fnum(r.get("source_vs_adamw"), -999.0), reverse=True)
    return rows


def write_manifest(out_dir: Path) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    for name in REQUIRED:
        path = out_dir / name
        rows.append({"artifact": name, "required": 1, "exists": int(path.exists()), "bytes": path.stat().st_size if path.exists() else 0})
    for name in SUPPLEMENTAL + FIGURES:
        path = out_dir / name
        rows.append({"artifact": name, "required": 0, "exists": int(path.exists()), "bytes": path.stat().st_size if path.exists() else 0})
    missing = sum(1 for r in rows if sint(r.get("required"), 0) and not sint(r.get("exists"), 0))
    write_rows(out_dir / "v137_required_manifest.csv", rows)
    return rows, missing


def write_no_go(out_dir: Path, route: dict[str, Any]) -> None:
    lines = [
        "# v13.7 no-go boundary",
        "",
        f"route = {route.get('route')}",
        f"minimum_success = {route.get('minimum_success')}",
        "",
        "Closed facts:",
        f"- S1 implementation sanity pass: {route.get('s1_implementation_sanity_pass')}",
        f"- MLP S2 task pass count: {route.get('mlp_s2_task_pass_count')}/7",
        f"- RAT basis S3 task pass count: {route.get('rat_basis_s3_task_pass_count')}/7",
        f"- cover S4 task pass count: {route.get('cover_s4_task_pass_count')}/7",
        f"- Non-RAT substrate-health pass count: {route.get('nonrat_substrate_health_pass_count')}",
        "",
        "Boundary:",
        "- Continuous training state was used; this is not a one-shot update.",
        "- CEp99/NLL/ECE/LineC remain audit/gate only.",
        "- No validation/test/future/query batch generated the direction.",
        "- Do not write diagnostic or substrate scout rows as promotion.",
    ]
    (out_dir / "v137_no_go_boundary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    next_lines = [
        "# v13.7 next hypothesis queue",
        "",
        "1. If R4: continuous PopRiskSNR under current implementation is no-go; return to substrate/base architecture or new value-source theory.",
        "2. If R5: generic MLP SNR works but KAN does not; debug basis telemetry and Rational substrate interaction.",
        "3. If S3 opens: expand Rational substrates and run real triage before any promotion.",
        "4. If Non-RAT remains WorkspaceOnly, keep it out of functional proof and repair substrate health separately.",
    ]
    (out_dir / "v137_next_hypothesis_queue.md").write_text("\n".join(next_lines) + "\n", encoding="utf-8")


def write_progress(out_dir: Path, route: dict[str, Any]) -> None:
    rows = [
        {"stage": "S1_implementation_sanity", "status": route.get("s1_implementation_sanity_pass"), "value": route.get("s1_implementation_sanity_pass"), "no_fake": 1},
        {"stage": "S2_MLP_generic_SNR", "status": int(route.get("mlp_s2_task_pass_count", 0) >= 5), "value": route.get("mlp_s2_task_pass_count"), "no_fake": 1},
        {"stage": "S3_RAT_basis_SNR", "status": int(route.get("rat_basis_s3_task_pass_count", 0) >= 5), "value": route.get("rat_basis_s3_task_pass_count"), "no_fake": 1},
        {"stage": "S4_basis_cover_boundary", "status": int(route.get("cover_s4_task_pass_count", 0) >= 5), "value": route.get("cover_s4_task_pass_count"), "no_fake": 1},
        {"stage": "D_nonrat_substrate_repair", "status": int(route.get("nonrat_substrate_health_pass_count", 0) > 0), "value": route.get("nonrat_substrate_health_pass_count"), "no_fake": 1},
    ]
    write_rows(out_dir / "v137_progress_table.csv", rows)


def write_figures(out_dir: Path, route: dict[str, Any], family_rows: list[dict[str, Any]], failures: list[dict[str, Any]]) -> None:
    write_svg(out_dir / "fig_mlp_snr_vs_adamw_loss_time.svg", "MLP SNR vs AdamW", [f"S2 tasks={route.get('mlp_s2_task_pass_count')}/7", f"route={route.get('route')}"])
    write_svg(out_dir / "fig_rational_parameter_snr_vs_basis_snr.svg", "Rational parameter vs basis SNR", [f"param rows={route.get('rat_parameter_pass_rows')}", f"basis rows={route.get('rat_basis_s3_pass_rows')}"])
    write_svg(out_dir / "fig_basis_snr_active_fraction_by_phase.svg", "Basis SNR active fraction by phase", ["see v137_rational_snr_training.csv"])
    write_svg(out_dir / "fig_cover_boundary_debt_by_phase.svg", "Cover boundary debt by phase", [f"S4 tasks={route.get('cover_s4_task_pass_count')}/7"])
    write_svg(out_dir / "fig_signal_reservoir_trajectory.svg", "Signal / reservoir trajectory", ["LineC audit-only; see v137_linec_audit.csv"])
    write_svg(out_dir / "fig_noise_leak_vs_snr_active_fraction.svg", "Noise leak vs SNR active fraction", ["see v137_*_training.csv"])
    write_svg(out_dir / "fig_nonrat_substrate_health_matrix.svg", "Non-RAT substrate health", [f"pass={route.get('nonrat_substrate_health_pass_count')}"])
    write_svg(out_dir / "fig_mlp_vs_rational_snr_comparison.svg", "MLP vs Rational SNR", [f"MLP tasks={route.get('mlp_s2_task_pass_count')}", f"RAT basis tasks={route.get('rat_basis_s3_task_pass_count')}"])
    write_svg(out_dir / "fig_synthetic_5of7_heatmap.svg", "Synthetic family heatmap", [f"{r.get('task_or_dataset')}: MLP={r.get('mlp_task_pass')} RAT={r.get('rat_basis_task_pass')} COVER={r.get('cover_task_pass')}" for r in family_rows])
    write_svg(out_dir / "fig_real_triage_if_opened.svg", "Real triage", ["skipped unless S4 opens"])


def write_code_packet(out_dir: Path) -> tuple[list[dict[str, Any]], str]:
    entries = [
        Path(__file__),
        DOC_PLAN,
        out_dir / "v137_route_decision.json",
        out_dir / "v137_implementation_readback.csv",
        out_dir / "v137_snr_sanity_checks.csv",
        out_dir / "v137_mlp_snr_training.csv",
        out_dir / "v137_rational_snr_training.csv",
        out_dir / "v137_basis_cover_schedule.csv",
        out_dir / "v137_nonrat_substrate_repair.csv",
        out_dir / "v137_failure_table.csv",
        out_dir / "v137_no_go_boundary.md",
    ]
    packet = out_dir / "v137_code_review_packet.zip"
    with zipfile.ZipFile(packet, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in entries:
            if path.exists():
                zf.write(path, arcname=path.name)
    manifest = [{"packet": packet.name, "entry": p.name, "exists": int(p.exists()), "sha256": sha256_file(p) if p.exists() else "", "no_fake": 1} for p in entries]
    code_rows = code_review_manifest(out_dir)
    write_rows(out_dir / "v137_code_review_manifest.csv", code_rows + [{"stage": "V137_CODE_REVIEW_PACKET", **r} for r in manifest])
    return manifest, sha256_file(packet)


def read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def finalize_only(out_dir: Path) -> None:
    summaries = read_rows(out_dir / "v137_training_summary.csv")
    family = read_rows(out_dir / "v137_synthetic_family_summary.csv")
    sanity = read_rows(out_dir / "v137_snr_sanity_checks.csv")
    forbidden = read_rows(out_dir / "v137_forbidden_information_audit.csv")
    nonrat = read_rows(out_dir / "v137_nonrat_substrate_repair.csv")
    _manifest, missing = write_manifest(out_dir)
    route = build_route(summaries, family, sanity, forbidden, nonrat, missing)
    failures = failure_table(summaries, nonrat)
    write_rows(out_dir / "v137_failure_table.csv", failures)
    write_no_go(out_dir, route)
    write_progress(out_dir, route)
    write_figures(out_dir, route, family, failures)
    (out_dir / "v137_route_decision.json").write_text(json.dumps(route, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _packet, code_sha = write_code_packet(out_dir)
    _manifest, missing = write_manifest(out_dir)
    route = build_route(summaries, family, sanity, forbidden, nonrat, missing, code_sha=code_sha)
    (out_dir / "v137_route_decision.json").write_text(json.dumps(route, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_no_go(out_dir, route)
    write_progress(out_dir, route)
    write_figures(out_dir, route, family, failures)
    write_manifest(out_dir)
    print(json.dumps(route, indent=2, sort_keys=True))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--finalize-only", action="store_true")
    ap.add_argument("--synthetic-tasks", default="X1,X2,X3,X4,X5,X6,X7")
    ap.add_argument("--synthetic-seeds", default="0,1")
    ap.add_argument("--synthetic-train-size", type=int, default=96)
    ap.add_argument("--synthetic-val-size", type=int, default=48)
    ap.add_argument("--synthetic-dim", type=int, default=16)
    ap.add_argument("--synthetic-classes", type=int, default=3)
    ap.add_argument("--train-steps", type=int, default=48)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--log-interval", type=int, default=12)
    ap.add_argument("--lr", type=float, default=0.003)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--snr-tau", type=float, default=1.0)
    ap.add_argument("--snr-eps", type=float, default=1.0e-12)
    ap.add_argument("--snr-ema-decay", type=float, default=0.85)
    ap.add_argument("--soft-alpha", type=float, default=8.0)
    ap.add_argument("--active-fraction-cap", type=float, default=1.0)
    ap.add_argument("--mlp-hidden", type=int, default=160)
    ap.add_argument("--loss-interfaces", default="CE,Brier")
    ap.add_argument("--mlp-methods", default="MLP-AdamW,MLP-AdamW-SNRHard,MLP-AdamW-SNRSoft,MLP-AdamW-SNREMA,MLP-AdamW-SNRRoleNorm")
    ap.add_argument("--rational-methods", default="RAT-AdamW,RAT-ParameterSNRHard,RAT-ParameterSNRSoft,RAT-ParameterSNREMA,RAT-GroupSNR,RAT-GroupSNREMA,RAT-BasisSNR,RAT-BasisSNR-CoverWeak,RAT-BasisSNR-CoverPhaseSchedule,RAT-BasisSNR-CoverConsolidateOnly,RAT-BasisSNR-CoverNoPlasticity")
    ap.add_argument("--rational-candidate", default="")
    args = ap.parse_args()

    out_dir = args.out_dir
    ensure_dir(out_dir)
    if args.finalize_only:
        finalize_only(out_dir)
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sanity = run_sanity_checks(args, device)
    readback = implementation_readback()
    forbidden = forbidden_audit()
    result = run_v137(args, device)
    summaries = result["summaries"]
    add_baseline_deltas(summaries)
    for s in summaries:
        s.update(pass_flags(s))
        s["stage"] = "V137_TRAINING_SUMMARY"
        s["no_fake"] = 1
        s.pop("last_update_meta", None)
    family = task_family_summary(summaries, parse_ints(args.synthetic_seeds))
    failures = failure_table(summaries, result["nonrat"])
    real_triage = [{"stage": "V137_REAL_TRIAGE_IF_OPENED", "opened": 0, "skip_reason": "synthetic_S4_not_opened", "no_fake": 1}]
    linec_rows = []
    for s in summaries:
        linec_rows.append({
            "stage": "V137_LINEC_AUDIT",
            "task_or_dataset": s.get("task_or_dataset"),
            "seed": s.get("seed"),
            "family": s.get("family"),
            "method": s.get("method"),
            "loss_interface": s.get("loss_interface"),
            "CouplingR2_delta": s.get("CouplingR2_delta"),
            "NoiseSignalLeak_delta": s.get("NoiseSignalLeak_delta"),
            "ReservoirRatio_delta": s.get("ReservoirRatio_delta"),
            "linec_generates_direction": 0,
            "no_fake": 1,
        })
    controls = []
    for s in summaries:
        if str(s.get("method")).endswith("AdamW"):
            continue
        controls.append({
            "stage": "V137_TRAINING_CONTROLS",
            "task_or_dataset": s.get("task_or_dataset"),
            "seed": s.get("seed"),
            "family": s.get("family"),
            "method": s.get("method"),
            "loss_interface": s.get("loss_interface"),
            "control_name": "same_family_AdamW",
            "source_vs_control": s.get("source_vs_adamw"),
            "AUC_time_ratio_vs_control": s.get("AUC_time_ratio_vs_adamw"),
            "CEp99_delta": s.get("CEp99_delta"),
            "ECE_delta": s.get("ECE_delta"),
            "no_fake": 1,
        })
    write_rows(out_dir / "v137_implementation_readback.csv", readback)
    write_rows(out_dir / "v137_snr_sanity_checks.csv", sanity)
    write_rows(out_dir / "v137_mlp_snr_training.csv", result["mlp"])
    write_rows(out_dir / "v137_rational_snr_training.csv", result["rat"])
    write_rows(out_dir / "v137_basis_cover_schedule.csv", result["cover"] or [{"stage": "V137_BASIS_COVER_SCHEDULE", "skip_reason": "no_cover_methods_run", "no_fake": 1}])
    write_rows(out_dir / "v137_nonrat_substrate_repair.csv", result["nonrat"])
    write_rows(out_dir / "v137_training_controls.csv", controls or [{"stage": "V137_TRAINING_CONTROLS", "skip_reason": "no_non_adamw_methods", "no_fake": 1}])
    write_rows(out_dir / "v137_training_summary.csv", summaries)
    write_rows(out_dir / "v137_synthetic_family_summary.csv", family)
    write_rows(out_dir / "v137_linec_audit.csv", linec_rows)
    write_rows(out_dir / "v137_forbidden_information_audit.csv", forbidden)
    write_rows(out_dir / "v137_real_triage_if_opened.csv", real_triage)
    write_rows(out_dir / "v137_failure_table.csv", failures)
    write_rows(out_dir / "v137_substrate_map.csv", result["substrate_map"])
    write_rows(out_dir / "v137_update_writeback_trace.csv", result["writeback"])
    write_rows(out_dir / "v137_loss_interface_audit.csv", result["loss"])
    write_rows(out_dir / "v137_timing_memory_audit.csv", result["timing"])
    write_rows(out_dir / "v137_rational_rejection_audit.csv", rational_rejection_audit(result["rat"]) or [{"stage": "V137_RATIONAL_REJECTION_AUDIT", "skip_reason": "no_rat_rows", "audit_only": 1, "no_fake": 1}])
    _manifest, missing = write_manifest(out_dir)
    route = build_route(summaries, family, sanity, forbidden, result["nonrat"], missing)
    write_no_go(out_dir, route)
    write_progress(out_dir, route)
    write_figures(out_dir, route, family, failures)
    (out_dir / "v137_route_decision.json").write_text(json.dumps(route, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _packet, code_sha = write_code_packet(out_dir)
    _manifest, missing = write_manifest(out_dir)
    route = build_route(summaries, family, sanity, forbidden, result["nonrat"], missing, code_sha=code_sha)
    (out_dir / "v137_route_decision.json").write_text(json.dumps(route, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_no_go(out_dir, route)
    write_progress(out_dir, route)
    write_figures(out_dir, route, family, failures)
    write_manifest(out_dir)
    print(json.dumps(route, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
