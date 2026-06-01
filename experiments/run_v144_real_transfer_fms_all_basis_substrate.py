#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
import time
from copy import copy
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.run_v1231_basis_kernel_workspace import linec_metrics
from experiments.run_v133_task_family_robust_basis_natural import eval_metrics
from experiments.run_v136_poprisk_snr_basis_cover_boundary import collect_per_example_gradients
from experiments.run_v142_functional_first_all_basis_parallel import (
    FMSState,
    fnum,
    group_utilities,
    loss_value,
    make_case_model,
    named_param_specs,
    sint,
)
from experiments.run_v143_functional_value_constraint_all_basis_substrate import (
    assign_flat_grad,
    base_projection_scales,
    clamp_rational_params,
    flat_existing_grad,
    parse_csv,
    parse_ints,
    projection_stats,
    scale_tensor_by_role,
    value_preserving_scales,
    write_json,
    write_rows,
    write_svg,
)
from experiments.run_v143_real_short_run_gate import load_real_split, output_geometry_model


ROOT = Path("results/v14_4_real_transfer_fms_all_basis_substrate")
PLAN_PATH = Path("docs/DG-KAN_v14.4_RealTransferFMS_AllBasisSubstrate_完整计划.md")
V143_ROOT = Path("results/v14_3_functional_value_constraint_all_basis_substrate")

REQUIRED = [
    "v144_route_decision.json",
    "v144_required_manifest.csv",
    "v144_forbidden_information_audit.csv",
    "v144_code_review_manifest.csv",
    "v144_real_transfer_fms_results.csv",
    "v144_real_transfer_fms_summary.csv",
    "v144_train_stream_proxy.csv",
    "v144_projection_value_retention.csv",
    "v144_source_tail_costate.csv",
    "v144_split_agreement.csv",
    "v144_real_3x3_failure_table.csv",
    "v144_wavelet_substrate_hardening.csv",
    "v144_rbf_substrate_repair.csv",
    "v144_chebyshev_lifetime_repair.csv",
    "v144_fourier_lifetime_repair.csv",
    "v144_all_basis_substrate_status.csv",
    "v144_mlp_generic_fms_control.csv",
    "v144_kan_specific_advantage.csv",
    "v144_linec_audit.csv",
    "v144_tail_calibration_audit.csv",
    "v144_optimizer_state_transport_probe.csv",
    "fig_real_3x3_pass_matrix.svg",
    "fig_source_vs_tail_scatter.svg",
    "fig_projection_retention_vs_source.svg",
    "fig_train_stream_proxy_vs_real_audit.svg",
    "fig_linec_failure_by_dataset_seed.svg",
    "fig_basis_substrate_matrix.svg",
    "fig_kan_specific_advantage.svg",
]

K_CONTROL_METHODS = {"K0-RAT-AdamW", "KCTRL-RandomMatchedProjection"}
MLP_CONTROL_METHODS = {"MLP-AdamW"}


def median(values: list[float]) -> float:
    vals = sorted(v for v in values if math.isfinite(float(v)))
    if not vals:
        return 0.0
    mid = len(vals) // 2
    if len(vals) % 2:
        return float(vals[mid])
    return float(0.5 * (vals[mid - 1] + vals[mid]))


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def sha256_file(path: Path) -> str:
    if not path.exists():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def per_example_loss(logits: torch.Tensor, y: torch.Tensor, loss_interface: str) -> torch.Tensor:
    if loss_interface == "CE":
        return F.cross_entropy(logits.float(), y, reduction="none")
    if loss_interface == "Brier":
        probs = torch.softmax(logits.float(), dim=1)
        target = F.one_hot(y, num_classes=logits.shape[1]).float()
        return (probs - target).square().sum(dim=1)
    raise ValueError(loss_interface)


def train_stream_proxy(model: torch.nn.Module, xb: torch.Tensor, yb: torch.Tensor, loss_interface: str) -> dict[str, float]:
    with torch.no_grad():
        logits = model(xb).detach().float()
        losses = per_example_loss(logits, yb, loss_interface)
        probs = torch.softmax(logits, dim=1)
        top2 = probs.topk(k=min(2, probs.shape[1]), dim=1).values
        margin = top2[:, 0] - (top2[:, 1] if top2.shape[1] > 1 else torch.zeros_like(top2[:, 0]))
        entropy = (-(probs * torch.log(probs.clamp_min(1.0e-8))).sum(dim=1) / math.log(max(2, probs.shape[1]))).mean()
        return {
            "train_loss_mean": float(losses.mean().item()),
            "train_loss_q95": float(torch.quantile(losses.float(), 0.95).item()),
            "train_margin_p10": float(torch.quantile(margin.float(), 0.10).item()),
            "train_logit_rms": float(logits.square().mean().sqrt().item()),
            "train_entropy_norm": float(entropy.item()),
        }


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = float(a.norm().item() * b.norm().item())
    if denom <= 1.0e-8:
        return 0.0
    return float(torch.dot(a, b).item() / denom)


def inverse_logistic(arg: float) -> float:
    stable_arg = max(-60.0, min(60.0, float(arg)))
    return 1.0 / (1.0 + math.exp(stable_arg))


def optimizer_state_flat(specs: list[Any], opt: torch.optim.Optimizer, key: str) -> tuple[torch.Tensor, bool]:
    chunks: list[torch.Tensor] = []
    any_state = False
    for spec in specs:
        state = opt.state.get(spec.param, {})
        tensor = state.get(key)
        if tensor is None:
            chunks.append(torch.zeros_like(spec.param.detach()).flatten())
            continue
        any_state = True
        chunks.append(tensor.detach().flatten().to(device=spec.param.device, dtype=spec.param.dtype))
    return torch.cat(chunks) if chunks else torch.tensor([]), any_state


def set_optimizer_state_from_flat(specs: list[Any], opt: torch.optim.Optimizer, key: str, flat: torch.Tensor) -> None:
    for spec in specs:
        state = opt.state.get(spec.param, {})
        if key not in state:
            continue
        state[key].copy_(flat[spec.start : spec.end].view_as(spec.param).to(dtype=state[key].dtype))


def optimizer_state_mismatch_metrics(
    specs: list[Any],
    opt: torch.optim.Optimizer,
    generic_flat: torch.Tensor,
    projected_flat: torch.Tensor,
) -> dict[str, float]:
    exp_avg, has_moment = optimizer_state_flat(specs, opt, "exp_avg")
    exp_avg_sq, has_rms = optimizer_state_flat(specs, opt, "exp_avg_sq")
    proj_norm = float(projected_flat.norm().item())
    generic_norm = float(generic_flat.norm().item())
    moment_norm = float(exp_avg.norm().item()) if has_moment else 0.0
    rms_norm = float(exp_avg_sq.clamp_min(0.0).sqrt().norm().item()) if has_rms else 0.0
    return {
        "optimizer_state_available": int(has_moment or has_rms),
        "affected_param_fraction": float((projected_flat - generic_flat).abs().gt(1.0e-12).float().mean().item()) if projected_flat.numel() else 0.0,
        "moment_staleness_score": 1.0 - cosine(exp_avg, projected_flat) if has_moment and moment_norm > 1.0e-8 and proj_norm > 1.0e-8 else 0.0,
        "rms_staleness_score": abs(math.log(rms_norm / max(1.0e-8, proj_norm))) if has_rms and rms_norm > 1.0e-8 and proj_norm > 1.0e-8 else 0.0,
        "post_event_update_cos_grad": cosine(projected_flat, generic_flat) if proj_norm > 1.0e-8 and generic_norm > 1.0e-8 else 0.0,
        "post_event_update_cos_adamw": cosine(projected_flat, exp_avg) if has_moment and moment_norm > 1.0e-8 and proj_norm > 1.0e-8 else 0.0,
        "adam_moment_norm": moment_norm,
        "adam_rms_norm": rms_norm,
        "projected_grad_norm": proj_norm,
        "generic_grad_norm": generic_norm,
    }


def apply_optimizer_state_transport(
    specs: list[Any],
    opt: torch.optim.Optimizer,
    generic_flat: torch.Tensor,
    projected_flat: torch.Tensor,
    mode: str,
    transport_mask: torch.Tensor | None = None,
) -> None:
    if mode == "none":
        return
    affected = transport_mask if transport_mask is not None else (projected_flat - generic_flat).abs().gt(1.0e-12)
    exp_avg, has_moment = optimizer_state_flat(specs, opt, "exp_avg")
    exp_avg_sq, has_rms = optimizer_state_flat(specs, opt, "exp_avg_sq")
    if mode == "zero_moment_reset" and has_moment:
        updated = torch.where(affected, torch.zeros_like(exp_avg), exp_avg)
        set_optimizer_state_from_flat(specs, opt, "exp_avg", updated)
    elif mode == "partial_moment_interpolation" and has_moment:
        updated = torch.where(affected, 0.5 * exp_avg + 0.5 * projected_flat, exp_avg)
        set_optimizer_state_from_flat(specs, opt, "exp_avg", updated)
    elif mode == "rms_recompute_microbatch" and has_rms:
        updated_sq = torch.where(affected, projected_flat.square(), exp_avg_sq)
        set_optimizer_state_from_flat(specs, opt, "exp_avg_sq", updated_sq)
    elif mode == "moment_transport_projected_grad" and has_moment:
        updated = torch.where(affected, projected_flat, exp_avg)
        set_optimizer_state_from_flat(specs, opt, "exp_avg", updated)


def apply_generic_moment_reset(
    specs: list[Any],
    opt: torch.optim.Optimizer,
    mask: torch.Tensor,
) -> int:
    exp_avg, has_moment = optimizer_state_flat(specs, opt, "exp_avg")
    if not has_moment or exp_avg.numel() == 0:
        return 0
    if mask.numel() != exp_avg.numel():
        return 0
    updated = torch.where(mask.bool(), torch.zeros_like(exp_avg), exp_avg)
    set_optimizer_state_from_flat(specs, opt, "exp_avg", updated)
    return 1


def random_flat_mask_like(flat: torch.Tensor, fraction: float, gen: torch.Generator) -> torch.Tensor:
    if flat.numel() == 0:
        return torch.zeros_like(flat, dtype=torch.bool)
    frac = max(0.0, min(1.0, float(fraction)))
    count = int(math.ceil(frac * float(flat.numel())))
    count = max(0, min(count, int(flat.numel())))
    mask = torch.zeros_like(flat, dtype=torch.bool)
    if count <= 0:
        return mask
    scores = torch.rand(flat.numel(), generator=gen, device=flat.device)
    idx = torch.topk(scores, k=count).indices
    mask[idx] = True
    return mask


def optimizer_state_transport_mask(
    generic_flat: torch.Tensor,
    projected_flat: torch.Tensor,
    scope: str,
    gen: torch.Generator,
) -> torch.Tensor:
    affected = (projected_flat - generic_flat).abs().gt(1.0e-12)
    delta = (projected_flat - generic_flat).abs()
    if scope == "affected":
        return affected
    if scope == "delta_top25":
        count = int(math.ceil(0.25 * float(affected.sum().item())))
        if count <= 0:
            return affected
        count = min(count, int(affected.numel()))
        idx = torch.topk(delta.reshape(-1), k=count).indices
        sparse_mask = torch.zeros_like(affected, dtype=torch.bool)
        sparse_mask.reshape(-1)[idx] = True
        return sparse_mask
    if scope in {"all_fms_roles", "full_adamw"}:
        return torch.ones_like(affected, dtype=torch.bool)
    if scope == "random_delta_top25":
        count = int(math.ceil(0.25 * float(affected.sum().item())))
        if count <= 0:
            return affected
        count = min(count, int(affected.numel()))
        scores = torch.rand(affected.numel(), generator=gen, device=affected.device)
        idx = torch.topk(scores, k=count).indices
        random_mask = torch.zeros_like(affected, dtype=torch.bool)
        random_mask[idx] = True
        return random_mask
    if scope in {"random_matched", "random_affected_fraction"}:
        count = int(affected.sum().item())
        if count <= 0:
            return affected
        count = min(count, int(affected.numel()))
        scores = torch.rand(affected.numel(), generator=gen, device=affected.device)
        idx = torch.topk(scores, k=count).indices
        random_mask = torch.zeros_like(affected, dtype=torch.bool)
        random_mask[idx] = True
        return random_mask
    return affected


def rt_method_base(method: str) -> str:
    if method == "K8-RAT-GenericFMS-ValuePreservingLineCProxyFreeConstraint":
        return method
    if method == "KCTRL-RandomMatchedProjection":
        return method
    if method == "K-RT5-DelayedBasisConstraint":
        return "K6-RAT-GenericFMS-DelayedBasisConstraint"
    if method in {
        "K-RT1-TrainSplitAgreement",
        "K-RT2-TrainStreamTailTrust",
        "K-RT3-ProjectionValueRetention",
        "K-RT4-SourceTailCoState",
        "K-RT6-ProjectionTailTrust",
        "K-RT7-LateProjectionTailTrust",
        "K-AUC1-SourceTailAUCColocation",
        "K-AUC2-UltraLateAUCGuard",
        "K-AUC3-FinalPulseTailTrust",
        "K-AUC4-FinalPulseIdentityProjection",
        "K-BF1-BasisFreeDelayedProjection",
        "K-CURV1-CurvatureSafeBasisFreeFMS",
        "K-2P1-TwoPhaseSourceTailSeparation",
        "K-2P2-TwoPhaseDelayedSourceTail",
        "K-TR1-GradientAlignedTrustRegion",
        "K-TR2-LateGradientAlignedTrustRegion",
        "K-CF1-TrainBatchDescentFilter",
        "K-CF2-LateTrainBatchDescentFilter",
        "K-FL1-FrontLoadedTrajectoryAssist",
        "K-FL2-FrontLoadedGradientAlignedAssist",
        "K-FMSDIR0-DirectionRemovedStateOnly",
        "K-FMSDIR1-RandomDirectionMatchedState",
        "K-FMSDIR2-AdamWParallelDirectionControl",
    }:
        return "K8-RAT-GenericFMS-ValuePreservingLineCProxyFreeConstraint"
    return method


def adaptive_projection_scales(
    method: str,
    generic_flat: torch.Tensor,
    specs: list[Any],
    base_scales: dict[str, float],
    step: int,
    total_steps: int,
    risk_state: float,
) -> dict[str, float]:
    if method in {
        "K-RT3-ProjectionValueRetention",
        "K-RT6-ProjectionTailTrust",
        "K-RT7-LateProjectionTailTrust",
        "K-AUC1-SourceTailAUCColocation",
        "K-AUC2-UltraLateAUCGuard",
        "K-AUC3-FinalPulseTailTrust",
        "K-BF1-BasisFreeDelayedProjection",
        "K-2P1-TwoPhaseSourceTailSeparation",
        "K-2P2-TwoPhaseDelayedSourceTail",
        "K-CF1-TrainBatchDescentFilter",
        "K-CF2-LateTrainBatchDescentFilter",
        "K-FL1-FrontLoadedTrajectoryAssist",
        "K-FL2-FrontLoadedGradientAlignedAssist",
    }:
        if method == "K-BF1-BasisFreeDelayedProjection":
            phase = float(step + 1) / max(1.0, float(total_steps))
            if phase < 0.80:
                return {k: 1.0 for k in base_scales}
        if method == "K-2P1-TwoPhaseSourceTailSeparation":
            phase = float(step + 1) / max(1.0, float(total_steps))
            if phase < 0.55:
                return {k: 1.0 for k in base_scales}
        if method == "K-2P2-TwoPhaseDelayedSourceTail":
            phase = float(step + 1) / max(1.0, float(total_steps))
            if phase < 0.75:
                return {k: 1.0 for k in base_scales}
        for alpha in [0.25, 0.50, 0.75, 1.0]:
            candidate = {k: 1.0 + alpha * (float(v) - 1.0) for k, v in base_scales.items()}
            projected = scale_tensor_by_role(generic_flat, specs, candidate)
            stats = projection_stats(generic_flat, projected, specs, candidate)
            if stats["value_retention"] >= 0.80 and stats["cos_projected_vs_generic"] >= 0.75:
                return candidate
        return {k: 1.0 for k in base_scales}
    if method in {
        "K-AUC4-FinalPulseIdentityProjection",
        "K-CURV1-CurvatureSafeBasisFreeFMS",
        "K-TR1-GradientAlignedTrustRegion",
        "K-TR2-LateGradientAlignedTrustRegion",
        "K-FL2-FrontLoadedGradientAlignedAssist",
        "K-FMSDIR0-DirectionRemovedStateOnly",
        "K-FMSDIR1-RandomDirectionMatchedState",
        "K-FMSDIR2-AdamWParallelDirectionControl",
    }:
        return {k: 1.0 for k in base_scales}
    if method == "K-RT5-DelayedBasisConstraint":
        phase = float(step + 1) / max(1.0, float(total_steps))
        if phase < 0.33:
            return {k: 1.0 + 0.15 * (float(v) - 1.0) for k, v in base_scales.items()}
        if phase < 0.66 or risk_state > 0.0:
            return {k: 1.0 + 0.50 * (float(v) - 1.0) for k, v in base_scales.items()}
    return base_scales


def lambda_from_method(
    method: str,
    state: dict[str, float],
    proxy: dict[str, float],
    value_signal: float,
    split_agreement: float,
    args: argparse.Namespace,
) -> float:
    lam_max = float(args.rt_lambda_max)
    if method in {"K0-RAT-AdamW", "MLP-AdamW"}:
        return 0.0
    if method in {"K8-RAT-GenericFMS-ValuePreservingLineCProxyFreeConstraint", "KCTRL-RandomMatchedProjection", "MLP-FMS-Amortized"}:
        return lam_max
    if method in {"K-RT1-TrainSplitAgreement", "MLP-FMS-SplitAgreement"}:
        return max(0.0, min(lam_max, lam_max * (split_agreement - float(args.rt_agreement_a0)) / max(1.0e-8, float(args.rt_agreement_a1) - float(args.rt_agreement_a0))))
    prev_q = state.get("prev_loss_q95")
    prev_m = state.get("prev_margin_p10")
    prev_r = state.get("prev_logit_rms")
    risk_delta = 0.0
    if prev_q is not None and prev_m is not None and prev_r is not None:
        risk_delta = (proxy["train_loss_q95"] - prev_q) - (proxy["train_margin_p10"] - prev_m) + 0.25 * (proxy["train_logit_rms"] - prev_r)
    state["risk_state"] = float(args.rt_state_beta) * state.get("risk_state", 0.0) + (1.0 - float(args.rt_state_beta)) * risk_delta
    state["value_state"] = float(args.rt_state_beta) * state.get("value_state", 0.0) + (1.0 - float(args.rt_state_beta)) * math.tanh(max(0.0, value_signal) / 10.0)
    prev_mean = state.get("prev_loss_mean")
    rel_progress = 0.0
    if prev_mean is not None:
        rel_progress = max(0.0, (float(prev_mean) - proxy["train_loss_mean"]) / max(1.0e-6, abs(float(prev_mean))))
    state["loss_rel_progress_ema"] = float(args.rt_state_beta) * state.get("loss_rel_progress_ema", 0.0) + (1.0 - float(args.rt_state_beta)) * rel_progress
    state["prev_loss_mean"] = proxy["train_loss_mean"]
    state["prev_loss_q95"] = proxy["train_loss_q95"]
    state["prev_margin_p10"] = proxy["train_margin_p10"]
    state["prev_logit_rms"] = proxy["train_logit_rms"]
    risk = state.get("risk_state", 0.0)
    value = state.get("value_state", 0.0)
    if method in {"K-RT2-TrainStreamTailTrust", "K-RT6-ProjectionTailTrust"}:
        return max(0.0, min(lam_max, lam_max * inverse_logistic(float(args.rt_risk_scale) * risk)))
    if method == "K-RT7-LateProjectionTailTrust":
        phase = max(0.0, min(1.0, float(proxy.get("train_step_phase", 1.0))))
        ramp = max(0.0, min(1.0, (phase - 0.50) / 0.35))
        trust = inverse_logistic(float(args.rt_risk_scale) * risk)
        return max(0.0, min(lam_max, lam_max * ramp * trust))
    if method == "K-AUC1-SourceTailAUCColocation":
        phase = max(0.0, min(1.0, float(proxy.get("train_step_phase", 1.0))))
        progress = max(0.0, state.get("loss_rel_progress_ema", 0.0))
        trajectory_guard = inverse_logistic(80.0 * (progress - 0.015))
        tail_trust = inverse_logistic(float(args.rt_risk_scale) * risk)
        source_gate = inverse_logistic(-18.0 * (value - 0.015))
        phase_gate = 0.25 + 0.75 * phase
        return max(0.0, min(lam_max, lam_max * trajectory_guard * tail_trust * source_gate * phase_gate))
    if method == "K-AUC2-UltraLateAUCGuard":
        phase = max(0.0, min(1.0, float(proxy.get("train_step_phase", 1.0))))
        ramp = max(0.0, min(1.0, (phase - 0.78) / 0.17))
        tail_trust = inverse_logistic(float(args.rt_risk_scale) * risk)
        source_gate = inverse_logistic(-18.0 * (value - 0.015))
        return max(0.0, min(lam_max, lam_max * 0.75 * ramp * tail_trust * source_gate))
    if method == "K-AUC3-FinalPulseTailTrust":
        phase = max(0.0, min(1.0, float(proxy.get("train_step_phase", 1.0))))
        ramp = max(0.0, min(1.0, (phase - 0.93) / 0.05))
        tail_trust = inverse_logistic(float(args.rt_risk_scale) * risk)
        source_gate = inverse_logistic(-18.0 * (value - 0.010))
        return max(0.0, min(lam_max, lam_max * ramp * tail_trust * source_gate))
    if method == "K-AUC4-FinalPulseIdentityProjection":
        phase = max(0.0, min(1.0, float(proxy.get("train_step_phase", 1.0))))
        ramp = max(0.0, min(1.0, (phase - 0.95) / 0.04))
        source_gate = inverse_logistic(-18.0 * (value - 0.010))
        return max(0.0, min(lam_max, lam_max * 0.80 * ramp * source_gate))
    if method == "K-BF1-BasisFreeDelayedProjection":
        phase = max(0.0, min(1.0, float(proxy.get("train_step_phase", 1.0))))
        progress = max(0.0, state.get("loss_rel_progress_ema", 0.0))
        slow_progress_gate = inverse_logistic(90.0 * (progress - 0.010))
        tail_trust = inverse_logistic(float(args.rt_risk_scale) * risk)
        source_gate = inverse_logistic(-18.0 * (value - 0.012))
        phase_gate = max(0.15, min(1.0, 0.35 + 0.65 * phase))
        return max(0.0, min(lam_max, lam_max * phase_gate * slow_progress_gate * tail_trust * source_gate))
    if method == "K-CURV1-CurvatureSafeBasisFreeFMS":
        phase = max(0.0, min(1.0, float(proxy.get("train_step_phase", 1.0))))
        grad_norm = max(0.0, float(proxy.get("train_grad_norm", 0.0)))
        prev_grad = state.get("prev_train_grad_norm")
        grad_growth = 0.0
        if prev_grad is not None:
            grad_growth = (grad_norm - float(prev_grad)) / max(1.0e-6, abs(float(prev_grad)))
        state["prev_train_grad_norm"] = grad_norm
        grad_growth = max(-10.0, min(10.0, grad_growth))
        curvature_gate = inverse_logistic(8.0 * (grad_growth - 0.15))
        progress = max(0.0, state.get("loss_rel_progress_ema", 0.0))
        progress_gate = inverse_logistic(90.0 * (progress - 0.008))
        source_gate = inverse_logistic(-18.0 * (value - 0.010))
        tail_trust = inverse_logistic(float(args.rt_risk_scale) * risk)
        phase_gate = max(0.20, min(1.0, 0.50 + 0.50 * phase))
        return max(0.0, min(lam_max, lam_max * phase_gate * curvature_gate * progress_gate * source_gate * tail_trust))
    if method == "K-2P1-TwoPhaseSourceTailSeparation":
        phase = max(0.0, min(1.0, float(proxy.get("train_step_phase", 1.0))))
        progress = max(0.0, state.get("loss_rel_progress_ema", 0.0))
        source_gate = inverse_logistic(-18.0 * (value - 0.010))
        progress_guard = inverse_logistic(80.0 * (progress - 0.012))
        tail_trust = inverse_logistic(float(args.rt_risk_scale) * risk)
        if phase < 0.55:
            return max(0.0, min(lam_max, lam_max * 0.55 * source_gate * progress_guard))
        return max(0.0, min(lam_max, lam_max * 0.40 * source_gate * tail_trust))
    if method == "K-2P2-TwoPhaseDelayedSourceTail":
        phase = max(0.0, min(1.0, float(proxy.get("train_step_phase", 1.0))))
        progress = max(0.0, state.get("loss_rel_progress_ema", 0.0))
        source_gate = inverse_logistic(-18.0 * (value - 0.010))
        tail_trust = inverse_logistic(float(args.rt_risk_scale) * risk)
        if phase < 0.35:
            return 0.0
        if phase < 0.75:
            progress_guard = inverse_logistic(80.0 * (progress - 0.010))
            return max(0.0, min(lam_max, lam_max * 0.45 * source_gate * progress_guard))
        late_ramp = max(0.0, min(1.0, (phase - 0.75) / 0.20))
        return max(0.0, min(lam_max, lam_max * 0.35 * late_ramp * source_gate * tail_trust))
    if method == "K-TR1-GradientAlignedTrustRegion":
        phase = max(0.0, min(1.0, float(proxy.get("train_step_phase", 1.0))))
        progress = max(0.0, state.get("loss_rel_progress_ema", 0.0))
        source_gate = inverse_logistic(-18.0 * (value - 0.010))
        tail_trust = inverse_logistic(float(args.rt_risk_scale) * risk)
        progress_guard = inverse_logistic(90.0 * (progress - 0.010))
        phase_gate = 0.20 + 0.80 * phase
        return max(0.0, min(lam_max, lam_max * phase_gate * source_gate * tail_trust * progress_guard))
    if method == "K-TR2-LateGradientAlignedTrustRegion":
        phase = max(0.0, min(1.0, float(proxy.get("train_step_phase", 1.0))))
        source_gate = inverse_logistic(-18.0 * (value - 0.010))
        tail_trust = inverse_logistic(float(args.rt_risk_scale) * risk)
        late_ramp = max(0.0, min(1.0, (phase - 0.82) / 0.15))
        return max(0.0, min(lam_max, lam_max * 0.70 * late_ramp * source_gate * tail_trust))
    if method == "K-CF1-TrainBatchDescentFilter":
        phase = max(0.0, min(1.0, float(proxy.get("train_step_phase", 1.0))))
        progress = max(0.0, state.get("loss_rel_progress_ema", 0.0))
        source_gate = inverse_logistic(-18.0 * (value - 0.010))
        tail_trust = inverse_logistic(float(args.rt_risk_scale) * risk)
        progress_guard = inverse_logistic(90.0 * (progress - 0.012))
        phase_gate = 0.25 + 0.75 * phase
        return max(0.0, min(lam_max, lam_max * phase_gate * source_gate * tail_trust * progress_guard))
    if method == "K-CF2-LateTrainBatchDescentFilter":
        phase = max(0.0, min(1.0, float(proxy.get("train_step_phase", 1.0))))
        source_gate = inverse_logistic(-18.0 * (value - 0.010))
        tail_trust = inverse_logistic(float(args.rt_risk_scale) * risk)
        late_ramp = max(0.0, min(1.0, (phase - 0.78) / 0.18))
        return max(0.0, min(lam_max, lam_max * 0.65 * late_ramp * source_gate * tail_trust))
    if method == "K-FL1-FrontLoadedTrajectoryAssist":
        phase = max(0.0, min(1.0, float(proxy.get("train_step_phase", 1.0))))
        warmup = max(0.0, min(1.0, phase / 0.08))
        cooldown = max(0.0, min(1.0, (0.62 - phase) / 0.42))
        early_gate = warmup * cooldown
        source_gate = inverse_logistic(-18.0 * (value - 0.008))
        tail_trust = inverse_logistic(float(args.rt_risk_scale) * risk)
        return max(0.0, min(lam_max, lam_max * early_gate * (0.35 + 0.65 * source_gate) * tail_trust))
    if method == "K-FL2-FrontLoadedGradientAlignedAssist":
        phase = max(0.0, min(1.0, float(proxy.get("train_step_phase", 1.0))))
        warmup = max(0.0, min(1.0, phase / 0.08))
        cooldown = max(0.0, min(1.0, (0.55 - phase) / 0.35))
        early_gate = warmup * cooldown
        source_gate = inverse_logistic(-18.0 * (value - 0.008))
        tail_trust = inverse_logistic(float(args.rt_risk_scale) * risk)
        return max(0.0, min(lam_max, lam_max * 0.85 * early_gate * (0.35 + 0.65 * source_gate) * tail_trust))
    if method in {"K-RT4-SourceTailCoState", "MLP-FMS-SourceTailCoState"}:
        return max(0.0, min(lam_max, lam_max * value / (abs(risk) + 0.20)))
    if method == "K-RT5-DelayedBasisConstraint":
        return max(0.0, min(lam_max, lam_max * inverse_logistic(2.0 * risk)))
    if method == "K-RT3-ProjectionValueRetention":
        return lam_max
    return lam_max


def gradient_aligned_value_flat(method: str, generic_flat: torch.Tensor, fms_flat: torch.Tensor, lam: float) -> torch.Tensor:
    if method not in {
        "K-TR1-GradientAlignedTrustRegion",
        "K-TR2-LateGradientAlignedTrustRegion",
        "K-FL2-FrontLoadedGradientAlignedAssist",
    }:
        return lam * fms_flat + (1.0 - lam) * generic_flat
    delta = fms_flat - generic_flat
    generic_norm = float(generic_flat.norm().item())
    delta_norm = float(delta.norm().item())
    if lam <= 0.0 or generic_norm <= 1.0e-8 or delta_norm <= 1.0e-8:
        return generic_flat
    aligned_cos = cosine(generic_flat, fms_flat)
    align_gate = max(0.0, min(1.0, (aligned_cos - 0.45) / 0.50))
    if method == "K-TR1-GradientAlignedTrustRegion":
        radius = 0.25
    elif method == "K-FL2-FrontLoadedGradientAlignedAssist":
        radius = 0.20
    else:
        radius = 0.14
    clip = min(1.0, radius * generic_norm / max(1.0e-8, delta_norm))
    return generic_flat + float(lam) * align_gate * clip * delta


def train_batch_descent_filtered_grad(method: str, generic_flat: torch.Tensor, candidate_flat: torch.Tensor) -> torch.Tensor:
    if method not in {"K-CF1-TrainBatchDescentFilter", "K-CF2-LateTrainBatchDescentFilter"}:
        return candidate_flat
    generic_norm = float(generic_flat.norm().item())
    candidate_norm = float(candidate_flat.norm().item())
    generic_sq = float(torch.dot(generic_flat, generic_flat).item())
    if generic_norm <= 1.0e-8 or candidate_norm <= 1.0e-8 or generic_sq <= 1.0e-12:
        return generic_flat
    descent_ratio = float(torch.dot(generic_flat, candidate_flat).item()) / generic_sq
    if descent_ratio <= 0.0:
        return generic_flat
    blend = max(0.0, min(1.0, (descent_ratio - 0.55) / 0.40))
    filtered = blend * candidate_flat + (1.0 - blend) * generic_flat
    norm_cap = 1.15 * generic_norm
    filtered_norm = float(filtered.norm().item())
    if filtered_norm > norm_cap:
        filtered = filtered * (norm_cap / max(1.0e-8, filtered_norm))
    return filtered


def train_case(
    *,
    family: str,
    dataset: str,
    seed: int,
    method: str,
    candidate_id: str,
    loss_interface: str,
    xtr: torch.Tensor,
    ytr: torch.Tensor,
    xva: torch.Tensor,
    yva: torch.Tensor,
    xte: torch.Tensor,
    yte: torch.Tensor,
    input_dim: int,
    output_dim: int,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    case_args = copy(args)
    case_args.synthetic_dim = int(input_dim)
    case_args.synthetic_classes = int(output_dim)
    model_family = "MLP" if family == "MLP" else "D-RAT"
    model = make_case_model(model_family, candidate_id, xtr, seed, case_args, device)
    specs = named_param_specs(model)
    params = [(spec.name, spec.param) for spec in specs]
    generic_keys = [spec.layer for spec in specs]
    adam_beta1 = float(getattr(args, "adam_beta1", 0.9))
    adam_beta2 = float(getattr(args, "adam_beta2", 0.999))
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        betas=(adam_beta1, adam_beta2),
    )
    gen = torch.Generator(device=device).manual_seed(int(seed) + 144_000 + sum(ord(c) for c in method + dataset + family))
    rng = random.Random(int(seed) + 144_900 + len(method))
    state = FMSState(float(args.fms_beta), float(args.fms_strength), int(seed) + 144)
    rt_state: dict[str, float] = {}
    last_key_scales = {key: 1.0 for key in set(generic_keys)}
    last_lambda = 1.0
    last_split_agreement = 1.0
    fms_refresh_count = 0
    state_transport_mode = str(getattr(args, "optimizer_state_transport_mode", "none"))
    state_transport_scope = str(getattr(args, "optimizer_state_transport_scope", "affected"))
    state_probe_enabled = int(getattr(args, "optimizer_state_transport_probe", 0)) == 1
    state_transport_recovery_window = int(getattr(args, "optimizer_state_transport_recovery_window", 0))
    generic_optimizer_control_mode = str(getattr(args, "generic_optimizer_control_mode", "none"))
    generic_optimizer_reset_fraction = float(getattr(args, "generic_optimizer_reset_fraction", 1.0))
    generic_optimizer_warmup_steps = int(getattr(args, "generic_optimizer_warmup_steps", max(1, int(args.fms_update_interval))))
    optimizer_transport_event_count = 0
    generic_optimizer_reset_event_count = 0
    generic_optimizer_reset_fractions: list[float] = []
    last_fms_event_step = -1_000_000_000
    projection_rows: list[dict[str, Any]] = []
    proxy_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    costate_rows: list[dict[str, Any]] = []
    grad_rows: list[dict[str, Any]] = []
    optimizer_probe_rows: list[dict[str, Any]] = []
    trajectory: list[dict[str, float]] = []
    overhead_acc = {
        "per_example_gradient_time_sec": 0.0,
        "fms_state_update_time_sec": 0.0,
        "projection_time_sec": 0.0,
        "state_transport_time_sec": 0.0,
        "optimizer_update_time_sec": 0.0,
        "sync_time_sec": 0.0,
        "artifact_logging_time_sec": 0.0,
        "linec_audit_time_sec": 0.0,
    }
    projection_stats_acc: dict[str, list[float]] = {
        "generic_value_norm": [],
        "projected_value_norm": [],
        "cos_projected_vs_generic": [],
        "value_retention": [],
        "projection_rejection_fraction": [],
    }
    optimizer_stats_acc: dict[str, list[float]] = {
        "moment_staleness_before": [],
        "moment_staleness_after": [],
        "rms_mismatch_before": [],
        "rms_mismatch_after": [],
        "post_event_update_cos_grad_before": [],
        "post_event_update_cos_grad_after": [],
        "post_event_update_cos_adamw_before": [],
        "post_event_update_cos_adamw_after": [],
        "affected_param_fraction": [],
    }
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    for step in range(int(args.train_steps)):
        optimizer_transport_event_this_step = False
        idx = torch.randint(0, xtr.shape[0], (int(args.batch_size),), generator=gen, device=device)
        xb, yb = xtr[idx], ytr[idx]
        proxy = train_stream_proxy(model, xb, yb, loss_interface)
        proxy["train_step_phase"] = float(step + 1) / max(1.0, float(args.train_steps))
        opt.zero_grad(set_to_none=True)
        if method in {"K0-RAT-AdamW", "MLP-AdamW"}:
            loss = loss_value(model(xb), yb, loss_interface)
            loss.backward()
            generic_flat = flat_existing_grad(specs)
            fms_flat = generic_flat
            projected_flat = generic_flat
            role_scales = {"default": 1.0}
            split_agreement = 1.0
            value_signal = 0.0
            lam = 0.0
            stats = projection_stats(generic_flat, projected_flat, specs, role_scales)
        else:
            refresh = step % max(1, int(args.fms_update_interval)) == 0
            optimizer_transport_event_this_step = bool(refresh)
            if refresh:
                last_fms_event_step = step
                fms_refresh_count += 1
                t_grad = time.perf_counter()
                g, _delta, _enabled = collect_per_example_gradients(model, xb, yb, params, loss_interface=loss_interface)
                overhead_acc["per_example_gradient_time_sec"] += time.perf_counter() - t_grad
                t_fms = time.perf_counter()
                utilities, grad_stats = group_utilities(g, specs, generic_keys, "F3-LayerFMS")
                key_scales = state.update(generic_keys, utilities, "F3-LayerFMS", step, int(args.train_steps))
                overhead_acc["fms_state_update_time_sec"] += time.perf_counter() - t_fms
                last_key_scales = dict(key_scales)
                generic_flat = g.mean(dim=0).detach().clone()
                value_signal = float(grad_stats.get("mean_group_snr", 0.0))
                if g.shape[0] >= 2:
                    half = max(1, g.shape[0] // 2)
                    split_agreement = cosine(g[:half].mean(dim=0), g[half:].mean(dim=0))
                else:
                    split_agreement = 0.0
                last_split_agreement = split_agreement
                if step == 0 or step == int(args.train_steps) - 1:
                    grad_rows.append(
                        {
                            "stage": "V144_PER_EXAMPLE_GRADIENT_STATS",
                            "family": family,
                            "dataset": dataset,
                            "seed": seed,
                            "method": method,
                            "step": step + 1,
                            **grad_stats,
                            "direction_uses_validation_test_future_query": 0,
                            "direction_uses_linec_cep99_nll_ece": 0,
                            "promotion_allowed": 0,
                        }
                    )
            else:
                loss = loss_value(model(xb), yb, loss_interface)
                loss.backward()
                generic_flat = flat_existing_grad(specs)
                key_scales = dict(last_key_scales)
                value_signal = rt_state.get("value_state", 0.0) * 10.0
                split_agreement = last_split_agreement
            t_projection = time.perf_counter()
            fms_flat = generic_flat.clone()
            for spec, key in zip(specs, generic_keys):
                fms_flat[spec.start : spec.end].mul_(float(key_scales.get(key, 1.0)))
            proxy["train_grad_norm"] = float(generic_flat.norm().item())
            lam = lambda_from_method(method, rt_state, proxy, value_signal, split_agreement, args)
            last_lambda = lam
            if method in {"K-FMSDIR0-DirectionRemovedStateOnly", "K-FMSDIR2-AdamWParallelDirectionControl"}:
                value_flat = generic_flat.clone()
            elif method == "K-FMSDIR1-RandomDirectionMatchedState":
                delta = fms_flat - generic_flat
                delta_norm = float(delta.norm().item())
                random_dir = torch.randn(generic_flat.shape, generator=gen, device=generic_flat.device, dtype=generic_flat.dtype)
                random_norm = float(random_dir.norm().item())
                if delta_norm > 1.0e-8 and random_norm > 1.0e-8:
                    random_dir = random_dir * (delta_norm / random_norm)
                    value_flat = generic_flat + float(lam) * random_dir
                else:
                    value_flat = generic_flat.clone()
            else:
                value_flat = gradient_aligned_value_flat(method, generic_flat, fms_flat, lam)
            if family == "MLP":
                role_scales = {"default": 1.0}
                projected_flat = value_flat
            else:
                base_method = rt_method_base(method)
                role_scales = base_projection_scales(base_method, step, int(args.train_steps), rng)
                role_scales = value_preserving_scales(base_method, value_flat, specs, role_scales)
                role_scales = adaptive_projection_scales(method, value_flat, specs, role_scales, step, int(args.train_steps), rt_state.get("risk_state", 0.0))
                projected_flat = scale_tensor_by_role(value_flat, specs, role_scales)
                projected_flat = train_batch_descent_filtered_grad(method, generic_flat, projected_flat)
            stats = projection_stats(value_flat, projected_flat, specs, role_scales)
            assign_flat_grad(specs, projected_flat)
            overhead_acc["projection_time_sec"] += time.perf_counter() - t_projection
        optimizer_transport_recovery_this_step = (
            state_transport_recovery_window > 0
            and last_fms_event_step >= 0
            and step > last_fms_event_step
            and (step - last_fms_event_step) <= state_transport_recovery_window
        )
        if (
            state_probe_enabled
            and (optimizer_transport_event_this_step or optimizer_transport_recovery_this_step)
            and method not in K_CONTROL_METHODS
            and method not in MLP_CONTROL_METHODS
        ):
            optimizer_transport_event_count += 1
            t_transport = time.perf_counter()
            transport_mask = optimizer_state_transport_mask(generic_flat, projected_flat, state_transport_scope, gen)
            transport_before = optimizer_state_mismatch_metrics(specs, opt, generic_flat, projected_flat)
            apply_optimizer_state_transport(specs, opt, generic_flat, projected_flat, state_transport_mode, transport_mask)
            transport_after = optimizer_state_mismatch_metrics(specs, opt, generic_flat, projected_flat)
            overhead_acc["state_transport_time_sec"] += time.perf_counter() - t_transport
            optimizer_stats_acc["moment_staleness_before"].append(float(transport_before.get("moment_staleness_score", 0.0)))
            optimizer_stats_acc["moment_staleness_after"].append(float(transport_after.get("moment_staleness_score", 0.0)))
            optimizer_stats_acc["rms_mismatch_before"].append(float(transport_before.get("rms_staleness_score", 0.0)))
            optimizer_stats_acc["rms_mismatch_after"].append(float(transport_after.get("rms_staleness_score", 0.0)))
            optimizer_stats_acc["post_event_update_cos_grad_before"].append(float(transport_before.get("post_event_update_cos_grad", 0.0)))
            optimizer_stats_acc["post_event_update_cos_grad_after"].append(float(transport_after.get("post_event_update_cos_grad", 0.0)))
            optimizer_stats_acc["post_event_update_cos_adamw_before"].append(float(transport_before.get("post_event_update_cos_adamw", 0.0)))
            optimizer_stats_acc["post_event_update_cos_adamw_after"].append(float(transport_after.get("post_event_update_cos_adamw", 0.0)))
            optimizer_stats_acc["affected_param_fraction"].append(float(transport_before.get("affected_param_fraction", 0.0)))
            if step == 0 or step == int(args.train_steps) - 1 or ((step + 1) % max(1, int(args.trace_interval)) == 0) or step % max(1, int(args.fms_update_interval)) == 0:
                optimizer_probe_rows.append(
                    {
                        "stage": "V144_OPTIMIZER_STATE_TRANSPORT_PROBE",
                        "family": family,
                        "dataset": dataset,
                        "seed": seed,
                        "method": method,
                        "step": step + 1,
                        "optimizer_state_transport_probe": 1,
                        "optimizer_state_transport_mode": state_transport_mode,
                        "optimizer_state_transport_scope": state_transport_scope,
                        "optimizer_state_transport_recovery_window": int(state_transport_recovery_window),
                        "optimizer_state_transport_recovery_step": int(optimizer_transport_recovery_this_step),
                        "lambda_fms": float(last_lambda),
                        "affected_param_fraction": transport_before.get("affected_param_fraction", 0.0),
                        "transport_reset_param_fraction": float(transport_mask.float().mean().item()) if transport_mask.numel() else 0.0,
                        "optimizer_state_available_before": transport_before.get("optimizer_state_available", 0),
                        "optimizer_state_available_after": transport_after.get("optimizer_state_available", 0),
                        "moment_staleness_before": transport_before.get("moment_staleness_score", 0.0),
                        "moment_staleness_after": transport_after.get("moment_staleness_score", 0.0),
                        "rms_mismatch_before": transport_before.get("rms_staleness_score", 0.0),
                        "rms_mismatch_after": transport_after.get("rms_staleness_score", 0.0),
                        "post_event_update_cos_grad_before": transport_before.get("post_event_update_cos_grad", 0.0),
                        "post_event_update_cos_grad_after": transport_after.get("post_event_update_cos_grad", 0.0),
                        "post_event_update_cos_adamw_before": transport_before.get("post_event_update_cos_adamw", 0.0),
                        "post_event_update_cos_adamw_after": transport_after.get("post_event_update_cos_adamw", 0.0),
                        "adam_moment_norm_before": transport_before.get("adam_moment_norm", 0.0),
                        "adam_moment_norm_after": transport_after.get("adam_moment_norm", 0.0),
                        "adam_rms_norm_before": transport_before.get("adam_rms_norm", 0.0),
                        "adam_rms_norm_after": transport_after.get("adam_rms_norm", 0.0),
                        "projected_grad_norm": transport_before.get("projected_grad_norm", 0.0),
                        "generic_grad_norm": transport_before.get("generic_grad_norm", 0.0),
                        "direction_uses_validation_test_future_query": 0,
                        "direction_uses_linec_cep99_nll_ece_auc_brier": 0,
                        "promotion_allowed": 0,
                    }
                )
        generic_reset_mask = None
        if generic_optimizer_control_mode in {"periodic_moment_reset", "full_moment_reset_at_fms_intervals"}:
            if step % max(1, int(args.fms_update_interval)) == 0:
                generic_reset_mask = torch.ones_like(generic_flat, dtype=torch.bool)
        elif generic_optimizer_control_mode == "event_matched_random_reset":
            if step % max(1, int(args.fms_update_interval)) == 0:
                generic_reset_mask = random_flat_mask_like(generic_flat, generic_optimizer_reset_fraction, gen)
        elif generic_optimizer_control_mode == "rmsprop_like_no_momentum":
            generic_reset_mask = torch.ones_like(generic_flat, dtype=torch.bool)
        elif generic_optimizer_control_mode == "no_momentum_warmup_then_adamw":
            if step < max(0, int(generic_optimizer_warmup_steps)):
                generic_reset_mask = torch.ones_like(generic_flat, dtype=torch.bool)
        elif generic_optimizer_control_mode == "matched_overhead_no_state_change":
            if step % max(1, int(args.fms_update_interval)) == 0:
                generic_reset_mask = random_flat_mask_like(generic_flat, generic_optimizer_reset_fraction, gen)
        if generic_reset_mask is not None:
            t_transport = time.perf_counter()
            if generic_optimizer_control_mode == "matched_overhead_no_state_change":
                exp_avg, has_moment = optimizer_state_flat(specs, opt, "exp_avg")
                if has_moment and exp_avg.numel() == generic_reset_mask.numel():
                    _ = torch.where(generic_reset_mask.bool(), exp_avg, exp_avg).sum().item()
                    generic_reset_applied = 1
                else:
                    generic_reset_applied = 0
            else:
                generic_reset_applied = apply_generic_moment_reset(specs, opt, generic_reset_mask)
            overhead_acc["state_transport_time_sec"] += time.perf_counter() - t_transport
        else:
            generic_reset_applied = 0
        if generic_reset_mask is not None and generic_reset_applied:
            generic_optimizer_reset_event_count += 1
            generic_optimizer_reset_fractions.append(float(generic_reset_mask.float().mean().item()) if generic_reset_mask.numel() else 0.0)
        t_opt = time.perf_counter()
        opt.step()
        overhead_acc["optimizer_update_time_sec"] += time.perf_counter() - t_opt
        if family != "MLP":
            clamp_rational_params(specs, rt_method_base(method))
        for key in projection_stats_acc:
            projection_stats_acc[key].append(float(stats.get(key, 0.0)))
        if step == 0 or step == int(args.train_steps) - 1 or ((step + 1) % max(1, int(args.trace_interval)) == 0):
            t_artifact = time.perf_counter()
            val_metrics = eval_metrics(model, xva, yva)
            trajectory.append(
                {
                    "step": float(step + 1),
                    "NLL": val_metrics["NLL"],
                    "CEp99": val_metrics["CEp99"],
                    "ECE": val_metrics["ECE"],
                    "acc": val_metrics["acc"],
                }
            )
            base_row = {
                "family": family,
                "dataset": dataset,
                "seed": seed,
                "method": method,
                "loss_interface": loss_interface,
                "step": step + 1,
                "lambda_fms": float(last_lambda),
                "train_split_agreement": float(split_agreement),
                **proxy,
                "value_state": float(rt_state.get("value_state", 0.0)),
                "risk_state": float(rt_state.get("risk_state", 0.0)),
                "train_loss_rel_progress_ema": float(rt_state.get("loss_rel_progress_ema", 0.0)),
                "direction_uses_validation_test_future_query": 0,
                "direction_uses_linec_cep99_nll_ece": 0,
                "promotion_allowed": 0,
            }
            proxy_rows.append({"stage": "V144_TRAIN_STREAM_PROXY", **base_row})
            split_rows.append({"stage": "V144_SPLIT_AGREEMENT", **base_row})
            costate_rows.append({"stage": "V144_SOURCE_TAIL_COSTATE", **base_row})
            projection_rows.append(
                {
                    "stage": "V144_PROJECTION_VALUE_RETENTION",
                    **base_row,
                    **stats,
                    "projection_applied": int(family != "MLP" and method not in {"K0-RAT-AdamW", "K1-RAT-GenericFMS-NoProjection"}),
                    "projection_uses_audit_metric": 0,
                }
            )
            overhead_acc["artifact_logging_time_sec"] += time.perf_counter() - t_artifact
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_memory_bytes = int(torch.cuda.max_memory_allocated(device))
    else:
        peak_memory_bytes = 0
    elapsed = time.perf_counter() - start
    eval_model, output_info = output_geometry_model(model, xtr, args)
    val_final = eval_metrics(eval_model, xva, yva)
    test_final = eval_metrics(eval_model, xte, yte)
    auc_nll = sum(point["NLL"] for point in trajectory) / max(1, len(trajectory))
    auc_cep99 = sum(point["CEp99"] for point in trajectory) / max(1, len(trajectory))
    linec_rows: list[dict[str, Any]] = []
    if family == "MLP" or str(args.linec_mode) == "none":
        linec_votes = [1]
        linec_coupling_r2 = val_final["CouplingR2"]
        linec_noise = val_final["NoiseSignalLeak"]
        linec_reservoir = val_final["RealSignalReservoirRatio"]
    else:
        t_linec = time.perf_counter()
        b = min(int(args.linec_batch_size), int(xtr.shape[0]), int(xva.shape[0]))
        linec_votes = []
        linec_metric_rows: list[dict[str, float]] = []
        for linec_seed in parse_ints(args.linec_seeds):
            try:
                lm = linec_metrics(eval_model, xtr[:b], ytr[:b], xva[:b], yva[:b], int(linec_seed), int(args.linec_sketch_dim), float(args.lr), float(args.weight_decay))
                status = "executed"
                error = ""
            except Exception as exc:  # noqa: BLE001
                lm = {"CouplingR2": float("nan"), "NoiseSignalLeak": float("nan"), "RealSignalReservoirRatio": float("nan")}
                status = "blocked"
                error = f"{type(exc).__name__}: {exc}"
            passed = int(
                fnum(lm.get("CouplingR2"), -999.0) >= 0.15
                and fnum(lm.get("NoiseSignalLeak"), 999.0) <= 0.20
                and fnum(lm.get("RealSignalReservoirRatio"), 999.0) <= 0.70
            )
            linec_votes.append(passed)
            linec_metric_rows.append(lm)
            linec_rows.append(
                {
                    "stage": "V144_LINEC_AUDIT",
                    "family": family,
                    "dataset": dataset,
                    "seed": seed,
                    "method": method,
                    "linec_seed": int(linec_seed),
                    "linec_status": status,
                    "linec_error": error,
                    **lm,
                    "linec_used_for_direction": 0,
                    "promotion_allowed": 0,
                }
            )
        linec_coupling_r2 = sum(fnum(row.get("CouplingR2"), 0.0) for row in linec_metric_rows) / max(1, len(linec_metric_rows))
        linec_noise = sum(fnum(row.get("NoiseSignalLeak"), 0.0) for row in linec_metric_rows) / max(1, len(linec_metric_rows))
        linec_reservoir = sum(fnum(row.get("RealSignalReservoirRatio"), 0.0) for row in linec_metric_rows) / max(1, len(linec_metric_rows))
        overhead_acc["linec_audit_time_sec"] += time.perf_counter() - t_linec
    row = {
        "stage": "V144_REAL_TRANSFER_FMS_RESULT" if family != "MLP" else "V144_MLP_GENERIC_FMS_CONTROL",
        "family": family,
        "candidate_id": candidate_id,
        "dataset": dataset,
        "seed": seed,
        "loss_interface": loss_interface,
        "method": method,
        "control_method": int(method in K_CONTROL_METHODS or method in MLP_CONTROL_METHODS),
        "train_size": int(args.train_size),
        "val_size": int(args.val_size),
        "test_size": int(args.test_size),
        "train_steps": int(args.train_steps),
        "batch_size": int(args.batch_size),
        "adam_beta1": float(adam_beta1),
        "adam_beta2": float(adam_beta2),
        "generic_optimizer_control_mode": generic_optimizer_control_mode,
        "generic_optimizer_reset_fraction": float(generic_optimizer_reset_fraction),
        "generic_optimizer_reset_event_count": int(generic_optimizer_reset_event_count),
        "generic_optimizer_reset_fraction_median": median(generic_optimizer_reset_fractions),
        "elapsed_sec": elapsed,
        "step_time_sec": elapsed / max(1, int(args.train_steps)),
        **overhead_acc,
        "peak_memory_bytes": peak_memory_bytes,
        "NLL": val_final["NLL"],
        "CEp99": val_final["CEp99"],
        "ECE": val_final["ECE"],
        "Brier": val_final["Brier"],
        "acc": val_final["acc"],
        "test_NLL": test_final["NLL"],
        "test_CEp99": test_final["CEp99"],
        "test_ECE": test_final["ECE"],
        "test_Brier": test_final["Brier"],
        "test_acc": test_final["acc"],
        "CouplingR2": linec_coupling_r2,
        "NoiseSignalLeak": linec_noise,
        "RealSignalReservoirRatio": linec_reservoir,
        "margin_p10": val_final["margin_p10"],
        "AUC_NLL": auc_nll,
        "AUC_CEp99": auc_cep99,
        "LineC_pass_rate": sum(linec_votes) / max(1, len(linec_votes)),
        "LineC_majority_pass": int(sum(linec_votes) >= math.ceil(len(linec_votes) / 2)),
        "LineC_pass_count": sum(linec_votes),
        "LineC_seed_count": len(linec_votes),
        "linec_mode": str(args.linec_mode),
        "fms_update_interval": int(args.fms_update_interval),
        "fms_refresh_count": int(fms_refresh_count),
        "optimizer_state_transport_probe": int(state_probe_enabled),
        "optimizer_state_transport_mode": state_transport_mode,
        "optimizer_state_transport_scope": state_transport_scope,
        "optimizer_state_transport_recovery_window": int(state_transport_recovery_window),
        "optimizer_state_transport_event_count": int(optimizer_transport_event_count),
        "direction_uses_validation_test_future_query": 0,
        "direction_uses_linec_cep99_nll_ece": 0,
        **output_info,
        "promotion_allowed": 0,
    }
    for key, values in projection_stats_acc.items():
        row[f"median_{key}"] = median(values)
    for key, values in optimizer_stats_acc.items():
        row[f"median_{key}"] = median(values)
    return {
        "row": row,
        "projection_rows": projection_rows,
        "proxy_rows": proxy_rows,
        "split_rows": split_rows,
        "costate_rows": costate_rows,
        "grad_rows": grad_rows,
        "optimizer_probe_rows": optimizer_probe_rows,
        "linec_rows": linec_rows,
    }


def enrich_rows(rows: list[dict[str, Any]], family_filter: str) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    for row in rows:
        if str(row.get("family")) != family_filter:
            continue
        by_key.setdefault((str(row["dataset"]), int(row["seed"]), str(row["loss_interface"])), []).append(row)
    out: list[dict[str, Any]] = []
    for _key, group in by_key.items():
        controls = [row for row in group if sint(row.get("control_method"), 0) == 1]
        best_auc = min((fnum(row.get("AUC_NLL"), 9.0) for row in controls), default=min(fnum(row.get("AUC_NLL"), 9.0) for row in group))
        best_nll = min((fnum(row.get("NLL"), 9.0) for row in controls), default=min(fnum(row.get("NLL"), 9.0) for row in group))
        adam = next((row for row in group if row.get("method") in {"K0-RAT-AdamW", "MLP-AdamW"}), controls[0] if controls else group[0])
        base_time = max(1.0e-8, fnum(adam.get("step_time_sec"), 1.0))
        base_mem = max(1.0, fnum(adam.get("peak_memory_bytes"), 1.0))
        for row in group:
            item = dict(row)
            item["source_vs_best_control"] = best_nll - fnum(row.get("NLL"), 9.0)
            item["source_vs_adamw"] = fnum(adam.get("NLL"), 9.0) - fnum(row.get("NLL"), 9.0)
            item["AUCtime_ratio_vs_best_control"] = fnum(row.get("AUC_NLL"), 9.0) / max(1.0e-8, best_auc)
            item["CEp99_delta_vs_adamw"] = fnum(row.get("CEp99"), 0.0) - fnum(adam.get("CEp99"), 0.0)
            item["NLL_delta_vs_adamw"] = fnum(row.get("NLL"), 0.0) - fnum(adam.get("NLL"), 0.0)
            item["ECE_delta_vs_adamw"] = fnum(row.get("ECE"), 0.0) - fnum(adam.get("ECE"), 0.0)
            item["step_time_ratio_vs_adamw"] = fnum(row.get("step_time_sec"), 0.0) / base_time
            item["peak_memory_ratio_vs_adamw"] = fnum(row.get("peak_memory_bytes"), 0.0) / base_mem
            item["real_transfer_gate_pass"] = int(
                sint(item.get("control_method"), 0) == 0
                and fnum(item["source_vs_best_control"]) >= 0.005
                and fnum(item["AUCtime_ratio_vs_best_control"]) <= 1.0
                and fnum(item["CEp99_delta_vs_adamw"]) <= 0.05
                and fnum(item["NLL_delta_vs_adamw"]) <= 0.02
                and fnum(item["ECE_delta_vs_adamw"]) <= 0.02
                and sint(item.get("LineC_majority_pass"), 0) == 1
                and fnum(item["step_time_ratio_vs_adamw"]) <= 1.75
                and fnum(item["peak_memory_ratio_vs_adamw"]) <= 1.75
            )
            out.append(item)
    return out


def summarize(rows: list[dict[str, Any]], stage: str) -> list[dict[str, Any]]:
    by_method: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_method.setdefault(str(row.get("method")), []).append(row)
    out: list[dict[str, Any]] = []
    for method, group in sorted(by_method.items()):
        target = [row for row in group if sint(row.get("control_method"), 0) == 0] or group
        out.append(
            {
                "stage": stage,
                "method": method,
                "rows": len(group),
                "dataset_seed_pass_count": len({(str(row.get("dataset")), str(row.get("seed"))) for row in group if sint(row.get("real_transfer_gate_pass"), 0) == 1}),
                "pass_rows": sum(sint(row.get("real_transfer_gate_pass"), 0) for row in group),
                "mean_source_vs_best_control": sum(fnum(row.get("source_vs_best_control"), 0.0) for row in target) / max(1, len(target)),
                "median_source_vs_best_control": median([fnum(row.get("source_vs_best_control"), 0.0) for row in target]),
                "median_auc_time": median([fnum(row.get("AUCtime_ratio_vs_best_control"), 9.0) for row in target]),
                "median_step_time": median([fnum(row.get("step_time_ratio_vs_adamw"), 9.0) for row in target]),
                "linec_majority_pass_rows": sum(sint(row.get("LineC_majority_pass"), 0) for row in target),
                "promotion_allowed": 0,
            }
        )
    return out


def failure_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        if sint(row.get("control_method"), 0) == 1 or sint(row.get("real_transfer_gate_pass"), 0) == 1:
            continue
        reasons = []
        if fnum(row.get("source_vs_best_control"), 0.0) < 0.005:
            reasons.append("source")
        if fnum(row.get("AUCtime_ratio_vs_best_control"), 9.0) > 1.0:
            reasons.append("AUCtime")
        if fnum(row.get("CEp99_delta_vs_adamw"), 0.0) > 0.05:
            reasons.append("CEp99_tail")
        if fnum(row.get("NLL_delta_vs_adamw"), 0.0) > 0.02:
            reasons.append("NLL_tail")
        if fnum(row.get("ECE_delta_vs_adamw"), 0.0) > 0.02:
            reasons.append("ECE_tail")
        if sint(row.get("LineC_majority_pass"), 0) != 1:
            reasons.append("LineC")
        if fnum(row.get("step_time_ratio_vs_adamw"), 0.0) > 1.75:
            reasons.append("step_time")
        out.append(
            {
                "stage": "V144_REAL_3X3_FAILURE_TABLE",
                "dataset": row.get("dataset"),
                "seed": row.get("seed"),
                "method": row.get("method"),
                "source_vs_best_control": row.get("source_vs_best_control"),
                "AUCtime_ratio_vs_best_control": row.get("AUCtime_ratio_vs_best_control"),
                "CEp99_delta_vs_adamw": row.get("CEp99_delta_vs_adamw"),
                "NLL_delta_vs_adamw": row.get("NLL_delta_vs_adamw"),
                "ECE_delta_vs_adamw": row.get("ECE_delta_vs_adamw"),
                "LineC_majority_pass": row.get("LineC_majority_pass"),
                "failure_reasons": "|".join(reasons),
                "promotion_allowed": 0,
            }
        )
    return out


def substrate_status_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    src = V143_ROOT / "rational_projection200_allk_seed012_v143" / "v143_nonrat_substrate_repair.csv"
    rows = read_rows(src)
    all_basis: list[dict[str, Any]] = []
    wavelet: list[dict[str, Any]] = []
    rbf: list[dict[str, Any]] = []
    che: list[dict[str, Any]] = []
    fou: list[dict[str, Any]] = []
    for row in rows:
        family = str(row.get("family", ""))
        out = {
            "stage": "V144_ALL_BASIS_SUBSTRATE_STATUS",
            "family": family,
            "source_artifact": str(src),
            "v143_strict_workspace_pass_rows": row.get("v143_strict_workspace_pass_rows", ""),
            "robust_candidate_count_seed012": row.get("robust_candidate_count_seed012", ""),
            "robust_candidate_ids": row.get("robust_candidate_ids", ""),
            "best_raw_memory_ratio_vs_mlp": row.get("best_raw_memory_ratio_vs_mlp", ""),
            "best_incremental_memory_ratio_vs_mlp": row.get("best_incremental_memory_ratio_vs_mlp", ""),
            "best_step_ratio_vs_mlp": row.get("best_step_ratio_vs_mlp", ""),
            "output_geometry_repair": row.get("output_geometry_repair", ""),
            "new_training_executed": 0,
            "linec_tail_used_for_direction": 0,
            "promotion_allowed": 0,
        }
        all_basis.append(out)
        if family == "D-WAV":
            wavelet.append({"stage": "V144_WAVELET_SUBSTRATE_HARDENING", **out})
        elif family == "D-RBF":
            rbf.append({"stage": "V144_RBF_SUBSTRATE_REPAIR", **out})
        elif family == "D-CHE":
            che.append({"stage": "V144_CHEBYSHEV_LIFETIME_REPAIR", **out})
        elif family == "D-FOU":
            fou.append({"stage": "V144_FOURIER_LIFETIME_REPAIR", **out})
    return all_basis, wavelet, rbf, che, fou


def kan_specific_advantage_rows(k_rows: list[dict[str, Any]], mlp_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    mlp_gain_by_key: dict[tuple[str, int], float] = {}
    for row in mlp_rows:
        if sint(row.get("control_method"), 0) == 0:
            key = (str(row.get("dataset")), int(row.get("seed")))
            mlp_gain_by_key[key] = max(mlp_gain_by_key.get(key, -999.0), fnum(row.get("source_vs_adamw"), -999.0))
    out = []
    for row in k_rows:
        if sint(row.get("control_method"), 0) == 1:
            continue
        key = (str(row.get("dataset")), int(row.get("seed")))
        mlp_gain = mlp_gain_by_key.get(key, 0.0)
        out.append(
            {
                "stage": "V144_KAN_SPECIFIC_ADVANTAGE",
                "dataset": row.get("dataset"),
                "seed": row.get("seed"),
                "method": row.get("method"),
                "kan_source_vs_adamw": row.get("source_vs_adamw"),
                "best_mlp_source_vs_adamw": mlp_gain,
                "kan_specific_advantage": fnum(row.get("source_vs_adamw"), 0.0) - mlp_gain,
                "promotion_allowed": 0,
            }
        )
    return out


def build_route(k_rows: list[dict[str, Any]], args: argparse.Namespace, missing: int) -> dict[str, Any]:
    pass_keys = {(str(row.get("dataset")), int(row.get("seed"))) for row in k_rows if sint(row.get("control_method"), 0) == 0 and sint(row.get("real_transfer_gate_pass"), 0) == 1}
    expected = len(parse_csv(args.datasets)) * len(parse_ints(args.seeds))
    pass_count = len(pass_keys)
    forbidden = 0
    if pass_count >= expected and missing == 0:
        route = "S5-OfficialFunctionalSuccess"
        minimum = "S5-OfficialFunctionalSuccess"
    elif pass_count >= 6:
        route = "S4b-RealTransferExplorationPositive"
        minimum = "S4b-RealTransferExplorationPositive"
    elif pass_count > 0:
        route = "R2-RealTransferFail"
        minimum = "S4-RealShortRunOpened"
    else:
        route = "R2-RealTransferFail"
        minimum = "S4-RealShortRunOpened"
    mean_source = sum(fnum(row.get("source_vs_best_control"), 0.0) for row in k_rows if sint(row.get("control_method"), 0) == 0) / max(1, len([row for row in k_rows if sint(row.get("control_method"), 0) == 0]))
    return {
        "stage": "V144_ROUTE_DECISION",
        "route": route,
        "minimum_success": minimum,
        "official_s5_reached": int(route == "S5-OfficialFunctionalSuccess"),
        "promotion_allowed": int(route == "S5-OfficialFunctionalSuccess" and int(args.compute_budgeted_run) == 0),
        "compute_budgeted_run": int(args.compute_budgeted_run),
        "s3_source_artifact": str(V143_ROOT / "rational_projection200_allk_seed012_v143" / "v143_route_decision.json"),
        "real_short_run_opened": 1,
        "expected_dataset_seed_count": expected,
        "real_dataset_seed_pass_count": pass_count,
        "real_short_run_pass_rows": sum(sint(row.get("real_transfer_gate_pass"), 0) for row in k_rows if sint(row.get("control_method"), 0) == 0),
        "mean_source_vs_best_control_noncontrol": mean_source,
        "required_artifact_missing_count": missing,
        "forbidden_information_violation_count": forbidden,
        "direction_uses_validation_test_future_query": 0,
        "direction_uses_linec_cep99_nll_ece": 0,
        "out_dir": str(args.out_dir),
    }


def write_required_manifest(out_dir: Path) -> int:
    rows = []
    missing = 0
    for name in REQUIRED:
        path = out_dir / name
        exists = int(path.exists())
        missing += int(not exists)
        rows.append({"artifact": name, "exists": exists, "bytes": path.stat().st_size if path.exists() else 0})
    write_rows(out_dir / "v144_required_manifest.csv", rows)
    return missing


def write_code_review_manifest(out_dir: Path) -> None:
    files = [
        Path("experiments/run_v144_real_transfer_fms_all_basis_substrate.py"),
        Path("experiments/run_v143_real_short_run_gate.py"),
        Path("experiments/run_v143_functional_value_constraint_all_basis_substrate.py"),
        PLAN_PATH,
    ]
    write_rows(
        out_dir / "v144_code_review_manifest.csv",
        [
            {
                "path": str(path),
                "exists": int(path.exists()),
                "sha256": sha256_file(path),
            }
            for path in files
        ],
    )


def write_figures(out_dir: Path, route: dict[str, Any], summary: list[dict[str, Any]]) -> None:
    lines = [
        f"route={route['route']}",
        f"pass={route['real_dataset_seed_pass_count']}/{route['expected_dataset_seed_count']}",
        f"S5={route['official_s5_reached']}",
        f"promotion={route['promotion_allowed']}",
    ]
    for fig in [
        "fig_real_3x3_pass_matrix.svg",
        "fig_source_vs_tail_scatter.svg",
        "fig_projection_retention_vs_source.svg",
        "fig_train_stream_proxy_vs_real_audit.svg",
        "fig_linec_failure_by_dataset_seed.svg",
        "fig_basis_substrate_matrix.svg",
        "fig_kan_specific_advantage.svg",
    ]:
        write_svg(out_dir / fig, fig, lines + [f"{row['method']} pass={row['dataset_seed_pass_count']} source={row['mean_source_vs_best_control']:.6f}" for row in summary[:8]])


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    methods = parse_csv(args.methods)
    mlp_methods = [] if bool(args.skip_mlp_control) else parse_csv(args.mlp_methods)
    datasets = parse_csv(args.datasets)
    seeds = parse_ints(args.seeds)
    raw_rows: list[dict[str, Any]] = []
    proxy_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    costate_rows: list[dict[str, Any]] = []
    projection_rows: list[dict[str, Any]] = []
    linec_rows: list[dict[str, Any]] = []
    grad_rows: list[dict[str, Any]] = []
    optimizer_probe_rows: list[dict[str, Any]] = []
    for dataset in datasets:
        for seed in seeds:
            xtr, ytr, xva, yva, xte, yte, input_dim_t, output_dim_t = load_real_split(args, dataset, seed, device)
            input_dim, output_dim = int(input_dim_t.item()), int(output_dim_t.item())
            for method in methods:
                result = train_case(
                    family="D-RAT",
                    dataset=dataset,
                    seed=seed,
                    method=method,
                    candidate_id=str(args.rational_candidate),
                    loss_interface=str(args.loss_interface),
                    xtr=xtr,
                    ytr=ytr,
                    xva=xva,
                    yva=yva,
                    xte=xte,
                    yte=yte,
                    input_dim=input_dim,
                    output_dim=output_dim,
                    args=args,
                    device=device,
                )
                raw_rows.append(result["row"])
                proxy_rows.extend(result["proxy_rows"])
                split_rows.extend(result["split_rows"])
                costate_rows.extend(result["costate_rows"])
                projection_rows.extend(result["projection_rows"])
                linec_rows.extend(result["linec_rows"])
                grad_rows.extend(result["grad_rows"])
                optimizer_probe_rows.extend(result["optimizer_probe_rows"])
            for method in mlp_methods:
                result = train_case(
                    family="MLP",
                    dataset=dataset,
                    seed=seed,
                    method=method,
                    candidate_id="MLPBaseline",
                    loss_interface=str(args.loss_interface),
                    xtr=xtr,
                    ytr=ytr,
                    xva=xva,
                    yva=yva,
                    xte=xte,
                    yte=yte,
                    input_dim=input_dim,
                    output_dim=output_dim,
                    args=args,
                    device=device,
                )
                raw_rows.append(result["row"])
                proxy_rows.extend(result["proxy_rows"])
                split_rows.extend(result["split_rows"])
                costate_rows.extend(result["costate_rows"])
                projection_rows.extend(result["projection_rows"])
                grad_rows.extend(result["grad_rows"])
                optimizer_probe_rows.extend(result["optimizer_probe_rows"])
    k_rows = enrich_rows(raw_rows, "D-RAT")
    mlp_rows = enrich_rows(raw_rows, "MLP")
    k_summary = summarize(k_rows, "V144_REAL_TRANSFER_FMS_SUMMARY")
    mlp_summary = summarize(mlp_rows, "V144_MLP_GENERIC_FMS_SUMMARY")
    failures = failure_table(k_rows)
    all_basis, wavelet, rbf, che, fou = substrate_status_rows()
    advantage = kan_specific_advantage_rows(k_rows, mlp_rows)
    tail_rows = [
        {
            "stage": "V144_TAIL_CALIBRATION_AUDIT",
            "dataset": row.get("dataset"),
            "seed": row.get("seed"),
            "method": row.get("method"),
            "CEp99_delta_vs_adamw": row.get("CEp99_delta_vs_adamw"),
            "NLL_delta_vs_adamw": row.get("NLL_delta_vs_adamw"),
            "ECE_delta_vs_adamw": row.get("ECE_delta_vs_adamw"),
            "tail_metrics_used_for_direction": 0,
            "promotion_allowed": 0,
        }
        for row in k_rows
    ]
    write_rows(out_dir / "v144_real_transfer_fms_results.csv", k_rows)
    write_rows(out_dir / "v144_real_transfer_fms_summary.csv", k_summary)
    write_rows(out_dir / "v144_mlp_generic_fms_control.csv", mlp_rows + mlp_summary)
    write_rows(out_dir / "v144_train_stream_proxy.csv", proxy_rows)
    write_rows(out_dir / "v144_projection_value_retention.csv", projection_rows)
    write_rows(out_dir / "v144_source_tail_costate.csv", costate_rows)
    write_rows(out_dir / "v144_split_agreement.csv", split_rows)
    write_rows(out_dir / "v144_real_3x3_failure_table.csv", failures)
    write_rows(out_dir / "v144_linec_audit.csv", linec_rows)
    write_rows(out_dir / "v144_tail_calibration_audit.csv", tail_rows)
    write_rows(out_dir / "v144_optimizer_state_transport_probe.csv", optimizer_probe_rows)
    write_rows(out_dir / "v144_wavelet_substrate_hardening.csv", wavelet)
    write_rows(out_dir / "v144_rbf_substrate_repair.csv", rbf)
    write_rows(out_dir / "v144_chebyshev_lifetime_repair.csv", che)
    write_rows(out_dir / "v144_fourier_lifetime_repair.csv", fou)
    write_rows(out_dir / "v144_all_basis_substrate_status.csv", all_basis)
    write_rows(out_dir / "v144_kan_specific_advantage.csv", advantage)
    write_rows(
        out_dir / "v144_forbidden_information_audit.csv",
        [
            {
                "direction_uses_validation_test_future_query": 0,
                "direction_uses_linec_cep99_nll_ece": 0,
                "uses_dataset_name_branch": 0,
                "uses_teacher_distillation_sampler_class_weight": 0,
                "uses_label_informed_initialization": 0,
                "promotion_allowed": 0,
            }
        ],
    )
    write_code_review_manifest(out_dir)
    missing = write_required_manifest(out_dir)
    route = build_route(k_rows, args, missing)
    write_json(out_dir / "v144_route_decision.json", route)
    write_figures(out_dir, route, k_summary)
    missing = write_required_manifest(out_dir)
    route = build_route(k_rows, args, missing)
    write_json(out_dir / "v144_route_decision.json", route)
    return route


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DG-KAN v14.4 RealTransferFMS AllBasisSubstrate")
    parser.add_argument("--out-dir", default=str(ROOT / "official_v144"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--datasets", default="MNIST,Fashion-MNIST,KMNIST")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--methods", default="K0-RAT-AdamW,K8-RAT-GenericFMS-ValuePreservingLineCProxyFreeConstraint,K-RT1-TrainSplitAgreement,K-RT2-TrainStreamTailTrust,K-RT3-ProjectionValueRetention,K-RT4-SourceTailCoState,K-RT5-DelayedBasisConstraint,KCTRL-RandomMatchedProjection")
    parser.add_argument("--mlp-methods", default="MLP-AdamW,MLP-FMS-Amortized,MLP-FMS-SplitAgreement,MLP-FMS-SourceTailCoState")
    parser.add_argument("--skip-mlp-control", action="store_true")
    parser.add_argument("--rational-candidate", default="D-RAT28-GroupDiversityPreservingRational")
    parser.add_argument("--loss-interface", default="CE")
    parser.add_argument("--train-size", type=int, default=1024)
    parser.add_argument("--val-size", type=int, default=512)
    parser.add_argument("--test-size", type=int, default=512)
    parser.add_argument("--mlp-hidden", type=int, default=32)
    parser.add_argument("--train-steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--fms-beta", type=float, default=0.99)
    parser.add_argument("--fms-strength", type=float, default=0.10)
    parser.add_argument("--fms-update-interval", type=int, default=80)
    parser.add_argument("--trace-interval", type=int, default=100)
    parser.add_argument("--rt-lambda-max", type=float, default=1.0)
    parser.add_argument("--rt-agreement-a0", type=float, default=0.0)
    parser.add_argument("--rt-agreement-a1", type=float, default=0.5)
    parser.add_argument("--rt-state-beta", type=float, default=0.90)
    parser.add_argument("--rt-risk-scale", type=float, default=4.0)
    parser.add_argument(
        "--output-geometry-repair",
        choices=["none", "fixed050", "train_rms_target050", "train_entropy_t080_100_else050"],
        default="none",
    )
    parser.add_argument(
        "--optimizer-state-transport-mode",
        choices=["none", "zero_moment_reset", "partial_moment_interpolation", "rms_recompute_microbatch", "moment_transport_projected_grad"],
        default="none",
    )
    parser.add_argument(
        "--optimizer-state-transport-scope",
        choices=["affected", "delta_top25", "all_fms_roles", "random_matched", "random_affected_fraction", "random_delta_top25", "full_adamw"],
        default="affected",
    )
    parser.add_argument("--optimizer-state-transport-probe", type=int, default=0)
    parser.add_argument("--optimizer-state-transport-recovery-window", type=int, default=0)
    parser.add_argument(
        "--generic-optimizer-control-mode",
        choices=[
            "none",
            "periodic_moment_reset",
            "event_matched_random_reset",
            "full_moment_reset_at_fms_intervals",
            "rmsprop_like_no_momentum",
            "no_momentum_warmup_then_adamw",
        ],
        default="none",
    )
    parser.add_argument("--generic-optimizer-reset-fraction", type=float, default=1.0)
    parser.add_argument("--generic-optimizer-warmup-steps", type=int, default=80)
    parser.add_argument("--linec-mode", choices=["none", "exact"], default="exact")
    parser.add_argument("--linec-seeds", default="12319500,12319501,12319502")
    parser.add_argument("--linec-batch-size", type=int, default=24)
    parser.add_argument("--linec-sketch-dim", type=int, default=8)
    parser.add_argument("--compute-budgeted-run", type=int, default=0)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    route = run(args)
    print(json.dumps(route, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
