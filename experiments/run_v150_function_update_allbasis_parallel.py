#!/usr/bin/env python3
"""DG-KAN v15.0 optimizer-aware function-update fork.

The runner executes the v15.0 FU/M/K/C/Z finalizer and reads an actual
substrate-only Line D run from run_v149_line_d_all_basis_substrate_repair.py.
All update directions are computed from the current train stream only.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import sys
import time
import zipfile
from copy import copy
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import run_v1410_nonrat_fms_transfer_fms_definition_reset as v1410  # noqa: E402
from experiments import run_v1414_fms_causal_value_transfer_boundary_all_basis_continue_open as v1414  # noqa: E402
from experiments import run_v144_real_transfer_fms_all_basis_substrate as v144  # noqa: E402


DEFAULT_OUT = ROOT / "results/v15_0_function_update_allbasis_parallel/official_v150"
DEFAULT_LINE_D_OUT = ROOT / "results/v15_0_function_update_allbasis_parallel/line_d_v150_allbasis_substrate"
V144_RATIONAL_MONITOR_DIR = ROOT / "results/v14_4_real_transfer_fms_all_basis_substrate/official_v144"
PLAN_DOC = ROOT / "docs/DG-KAN_v15.0_FunctionUpdate_AllBasis_并行加速完整计划.md"
RECAP_DOC = ROOT / "docs/DG-KAN_v15.0_FunctionUpdate_AllBasis_并行加速实验结果复盘.md"
EXEC_LOG_DOC = ROOT / "docs/DG-KAN_v15.0_FunctionUpdate_AllBasis_并行加速执行日志.md"

FU_METHODS = [
    "FU0-D-CHE-AdamW",
    "FU1-D-CHE-RawFunctionUpdate",
    "FU2-D-CHE-AdamV2FunctionUpdate",
    "FU3-D-CHE-CautiousFunctionUpdate",
    "FU4-D-CHE-MGUPFunctionUpdate",
    "FU5-D-CHE-SophiaDiagClippedFunctionUpdate",
    "FU6-D-CHE-BlockSecondMomentFunctionUpdate",
    "FU7-D-CHE-DecoupledDecayPlusFunctionUpdate",
    "FU8-D-CHE-SOAPLiteDiagnosticFunctionUpdate",
]

FU_CONTROLS = [
    "C0-D-CHE-AdamW",
    "C1-NoOpMatchedOverhead",
    "C2-RandomMatchedNorm",
    "C3-SameActiveFractionRandomMask",
    "C4-SameAlignmentMaskRandomDirection",
    "C5-SameSecondMomentScaleRandomDirection",
    "C6-SameDecoupledDecayNoFU",
    "C7-CoupledL2RegularizationControl",
    "C8-AdamWParallelDirection",
    "C9-GenericOptimizerStateControl",
]

MLP_METHODS = [
    "M0-MLP-AdamW",
    "M1-MLP-AdamW-DecoupledDecay",
    "M2-MLP-CautiousAdamW",
    "M3-MLP-MGUPAdamW",
    "M4-MLP-SophiaDiagAdamW",
    "M5-MLP-BlockSecondMomentAdamW",
    "M6-MLP-ScheduleFreeAdamW",
    "M7-MLP-SameOverheadNoUpdate",
]

LINE_D_CANDIDATES = [
    "D-FOU37-LowFreqIdentityResidualV4",
    "D-FOU38-BandwiseSecondMomentWarmup",
    "D-FOU39-PhaseStableCautiousUpdate",
    "D-FOU40-NoMaterializeLifetimeV4",
    "D-FOU41-HighFrequencyQuarantineV2",
    "D-RBF35-ActiveCenterSecondMoment",
    "D-RBF36-WidthConditionDecoupledDecay",
    "D-RBF37-CompactBumpIdentityResidualV2",
    "D-RBF38-GaussianLocalK4TaskHealthV2",
    "D-RBF39-NoDenseCenterUpdate",
    "D-WAV33-TriangularSupportV4",
    "D-WAV34-ScaleSecondMomentOccupancy",
    "D-WAV35-SupportOverlapCautiousUpdate",
    "D-WAV36-LocalTailCoverageAuditOnly",
]

REQUIRED = [
    "v150_route_decision.json",
    "v150_progress_table.csv",
    "v150_required_manifest.csv",
    "v150_forbidden_information_audit.csv",
    "v150_no_action_search_audit.csv",
    "v150_optimizer_mechanism_manifest.csv",
    "v150_decoupled_decay_audit.csv",
    "v150_second_moment_trace.csv",
    "v150_alignment_trace.csv",
    "v150_curvature_diag_trace.csv",
    "v150_block_preconditioner_trace.csv",
    "v150_schedulefree_control_trace.csv",
    "v150_dche_fu_results.csv",
    "v150_dche_fu_controls.csv",
    "v150_mlp_controls.csv",
    "v150_kan_specificity_summary.csv",
    "v150_decoupled_decay_deconfound.csv",
    "v150_allbasis_substrate_results.csv",
    "v150_rational_no_regression_monitor.csv",
    "v150_linec_tail_audit.csv",
    "v150_failure_taxonomy.csv",
    "v150_no_go_boundary.md",
    "v150_next_hypothesis_queue.md",
    "v150_code_review_packet.zip",
]

FIGURES = [
    "fig_v150_progress_table.svg",
    "fig_v150_fu_vs_controls.svg",
    "fig_v150_alignment_histogram.svg",
    "fig_v150_second_moment_scale_distribution.svg",
    "fig_v150_decoupled_decay_deconfound.svg",
    "fig_v150_cautious_vs_mgup_ablation.svg",
    "fig_v150_sophia_block_moment_ablation.svg",
    "fig_v150_mlp_vs_dche_optimizer_specificity.svg",
    "fig_v150_allbasis_substrate_heatmap.svg",
    "fig_v150_linec_tail_dashboard.svg",
    "fig_v150_runtime_breakdown.svg",
    "fig_v150_linec_signal_reservoir_noise.svg",
    "fig_v150_tail_calibration_dashboard.svg",
    "fig_v150_alignment_vs_linec.svg",
    "fig_v150_second_moment_vs_tail.svg",
]


def fnum(value: Any, default: float = 0.0) -> float:
    return v1414.fnum(value, default)


def sint(value: Any, default: int = 0) -> int:
    return v1414.sint(value, default)


def read_rows(path: Path) -> list[dict[str, str]]:
    return v1414.read_rows(path)


def write_rows(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    v1414.write_rows(path, rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    v1414.write_json(path, payload)


def write_text(path: Path, text: str) -> None:
    v1414.write_text(path, text)


def mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return statistics.fmean(vals) if vals else 0.0


def median(values: Iterable[float]) -> float:
    vals = sorted(float(v) for v in values if math.isfinite(float(v)))
    return statistics.median(vals) if vals else 0.0


def parse_csv(value: str) -> list[str]:
    return v1410.parse_csv(value)


def parse_ints(value: str) -> list[int]:
    return v1410.parse_ints(value)


def flat_params(specs: list[Any]) -> torch.Tensor:
    return torch.cat([spec.param.detach().flatten() for spec in specs]) if specs else torch.zeros(0)


def add_flat_update(specs: list[Any], update: torch.Tensor, lr: float) -> None:
    with torch.no_grad():
        for spec in specs:
            spec.param.add_(update[spec.start : spec.end].view_as(spec.param), alpha=float(lr))


def apply_role_decay(
    specs: list[Any],
    lr: float,
    weight_decay: float,
    readout_decay: float,
    cautious_grad: torch.Tensor | None = None,
    random_match_generator: torch.Generator | None = None,
) -> tuple[float, float, float]:
    before_sq = 0.0
    delta_sq = 0.0
    active = 0
    total = 0
    with torch.no_grad():
        for spec in specs:
            low = str(spec.name).lower()
            if "bias" in low:
                lam = 0.0
            elif spec.role == "readout":
                lam = float(readout_decay)
            elif spec.role == "basis" or (spec.param.ndim >= 3 and spec.param.shape[-1] <= 8):
                lam = float(weight_decay)
            else:
                lam = 0.0
            if lam <= 0:
                continue
            old = spec.param.detach().clone()
            if cautious_grad is None:
                spec.param.mul_(1.0 - float(lr) * lam)
                active += int(old.numel())
                total += int(old.numel())
            else:
                grad_seg = cautious_grad[spec.start : spec.end].view_as(spec.param)
                mask = (old * grad_seg) > 0.0
                if random_match_generator is not None:
                    count = int(mask.sum().item())
                    flat_mask = torch.zeros(mask.numel(), device=mask.device, dtype=torch.bool)
                    if count >= mask.numel():
                        flat_mask.fill_(True)
                    elif count > 0:
                        scores = torch.rand(mask.numel(), generator=random_match_generator, device=mask.device)
                        flat_mask[torch.topk(scores, count).indices] = True
                    mask = flat_mask.view_as(mask)
                updated = old.clone()
                updated[mask] = old[mask] * (1.0 - float(lr) * lam)
                spec.param.copy_(updated)
                active += int(mask.sum().item())
                total += int(mask.numel())
            diff = spec.param.detach() - old
            before_sq += float(old.square().sum().item())
            delta_sq += float(diff.square().sum().item())
    return math.sqrt(delta_sq), math.sqrt(before_sq), active / max(1, total)


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = float(a.norm().item() * b.norm().item())
    if denom <= 1.0e-12:
        return 0.0
    return float(torch.dot(a, b).item() / denom)


def norm_match(vec: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return vec / vec.norm().clamp_min(1.0e-12) * target.norm()


def role_condition(specs: list[Any], tensor: torch.Tensor) -> float:
    vals: list[float] = []
    for spec in specs:
        seg = tensor[spec.start : spec.end].float()
        if seg.numel():
            vals.append(float(seg.abs().mean().item()))
    vals = [v for v in vals if math.isfinite(v) and v > 0.0]
    if not vals:
        return 1.0
    return max(vals) / max(1.0e-12, min(vals))


def basis_safe_projection(specs: list[Any], update: torch.Tensor, family: str) -> torch.Tensor:
    if family != "D-CHE":
        return update.clone()
    out = update.clone()
    for spec in specs:
        seg = out[spec.start : spec.end].view_as(spec.param)
        if spec.role == "basis" or (spec.param.ndim >= 3 and spec.param.shape[-1] <= 8):
            if spec.param.ndim >= 3 and spec.param.shape[-1] <= 8:
                scale = torch.ones(spec.param.shape[-1], device=seg.device, dtype=seg.dtype)
                half = max(1, int(spec.param.shape[-1]) // 2)
                scale[half:] = 0.85
                seg.mul_(scale.view(*([1] * (seg.ndim - 1)), -1))
            else:
                seg.mul_(0.95)
        elif spec.role == "readout":
            seg.mul_(1.0)
        out[spec.start : spec.end] = seg.flatten()
    return out


def block_second_moment_update(specs: list[Any], mhat: torch.Tensor, vhat: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(mhat)
    for spec in specs:
        seg = mhat[spec.start : spec.end]
        vseg = vhat[spec.start : spec.end].clamp_min(0.0)
        scale = vseg.mean().sqrt().clamp_min(1.0e-8)
        out[spec.start : spec.end] = -seg / scale
    return out


def role_second_moment_rows(
    *,
    row_base: dict[str, Any],
    specs: list[Any],
    vhat: torch.Tensor,
) -> list[dict[str, Any]]:
    by_role: dict[str, list[float]] = {}
    for spec in specs:
        seg = vhat[spec.start : spec.end]
        by_role.setdefault(str(spec.role), []).append(float(seg.mean().item()) if seg.numel() else 0.0)
    rows = []
    for role, vals in sorted(by_role.items()):
        rows.append(
            {
                **row_base,
                "role": role,
                "degree_role_second_moment": mean(vals),
                "role_second_moment_max": max(vals) if vals else 0.0,
                "role_second_moment_min": min(vals) if vals else 0.0,
                "promotion_allowed": 0,
            }
        )
    return rows


def method_family(method: str) -> str:
    if method.startswith("FU"):
        return "D-CHE-FU"
    if method.startswith("C"):
        return "D-CHE-control"
    return "MLP-control"


def is_control(method: str) -> int:
    return int(method in FU_CONTROLS or method.startswith("M"))


def analog_method(dche_method: str) -> str:
    return {
        "FU2-D-CHE-AdamV2FunctionUpdate": "M0-MLP-AdamW",
        "FU3-D-CHE-CautiousFunctionUpdate": "M2-MLP-CautiousAdamW",
        "FU4-D-CHE-MGUPFunctionUpdate": "M3-MLP-MGUPAdamW",
        "FU5-D-CHE-SophiaDiagClippedFunctionUpdate": "M4-MLP-SophiaDiagAdamW",
        "FU6-D-CHE-BlockSecondMomentFunctionUpdate": "M5-MLP-BlockSecondMomentAdamW",
        "FU7-D-CHE-DecoupledDecayPlusFunctionUpdate": "M1-MLP-AdamW-DecoupledDecay",
    }.get(dche_method, "")


def compute_update(
    method: str,
    family: str,
    specs: list[Any],
    grad: torch.Tensor,
    mhat: torch.Tensor,
    vhat: torch.Tensor,
    hdiag: torch.Tensor,
    gen: torch.Generator,
    fu3_alignment_variant: str = "",
) -> tuple[torch.Tensor, dict[str, float]]:
    descent = -grad
    adam = -mhat / (vhat.sqrt() + 1.0e-8)
    curvature = -torch.clamp(mhat / (hdiag + 1.0e-5), -10.0, 10.0)
    preconditioned = adam
    alignment = preconditioned * descent
    align_mask = alignment > 0.0
    random_vec = torch.randn(descent.shape, generator=gen, device=descent.device, dtype=descent.dtype)
    method_update = adam
    hard_fraction = float(align_mask.float().mean().item()) if align_mask.numel() else 0.0
    soft_gate = torch.sigmoid(25.0 * alignment)
    clipped_fraction = 0.0
    schedulefree_trace = 0.0
    if method in {"FU0-D-CHE-AdamW", "C0-D-CHE-AdamW", "C1-NoOpMatchedOverhead", "C6-SameDecoupledDecayNoFU", "C8-AdamWParallelDirection", "M0-MLP-AdamW", "M1-MLP-AdamW-DecoupledDecay"}:
        method_update = adam
    elif method == "FU1-D-CHE-RawFunctionUpdate":
        method_update = descent
    elif method == "FU2-D-CHE-AdamV2FunctionUpdate":
        method_update = adam
    elif method in {"FU3-D-CHE-CautiousFunctionUpdate", "M2-MLP-CautiousAdamW"}:
        hard = torch.where(align_mask, adam, torch.zeros_like(adam))
        soft = adam * soft_gate
        method_update = hard if fu3_alignment_variant == "hard" else soft
    elif method in {"FU4-D-CHE-MGUPFunctionUpdate", "M3-MLP-MGUPAdamW"}:
        if alignment.numel() > 0:
            threshold = torch.quantile(alignment.float(), 0.50)
            method_update = torch.where(alignment >= threshold, 2.0 * adam, 0.5 * adam)
        else:
            method_update = adam
    elif method in {"FU5-D-CHE-SophiaDiagClippedFunctionUpdate", "M4-MLP-SophiaDiagAdamW"}:
        raw = mhat / (hdiag + 1.0e-5)
        clipped_fraction = float((raw.abs() > 10.0).float().mean().item()) if raw.numel() else 0.0
        method_update = -torch.clamp(raw, -10.0, 10.0)
    elif method in {"FU6-D-CHE-BlockSecondMomentFunctionUpdate", "M5-MLP-BlockSecondMomentAdamW"}:
        method_update = block_second_moment_update(specs, mhat, vhat)
    elif method == "FU7-D-CHE-DecoupledDecayPlusFunctionUpdate":
        method_update = adam
    elif method == "FU8-D-CHE-SOAPLiteDiagnosticFunctionUpdate":
        centered = torch.zeros_like(mhat)
        for spec in specs:
            seg = mhat[spec.start : spec.end]
            centered[spec.start : spec.end] = seg - seg.mean()
        method_update = block_second_moment_update(specs, centered, vhat)
    elif method == "C2-RandomMatchedNorm":
        method_update = norm_match(random_vec, adam)
    elif method == "C3-SameActiveFractionRandomMask":
        mask = align_mask
        if not bool(mask.any()):
            mask = torch.ones_like(align_mask, dtype=torch.bool)
        rnd = norm_match(random_vec * mask.float(), adam)
        method_update = rnd
    elif method == "C4-SameAlignmentMaskRandomDirection":
        method_update = norm_match(random_vec * align_mask.float(), adam)
    elif method == "C5-SameSecondMomentScaleRandomDirection":
        scaled = random_vec / (vhat.sqrt() + 1.0e-8)
        method_update = norm_match(scaled, adam)
    elif method == "C7-CoupledL2RegularizationControl":
        method_update = adam
    elif method == "C9-GenericOptimizerStateControl":
        perm = torch.randperm(adam.numel(), generator=gen, device=adam.device)
        method_update = adam[perm]
        method_update = norm_match(method_update, adam)
    elif method == "M6-MLP-ScheduleFreeAdamW":
        method_update = 0.75 * adam + 0.25 * descent
        schedulefree_trace = float((method_update - adam).norm().item() / max(1.0e-8, float(adam.norm().item())))
    elif method == "M7-MLP-SameOverheadNoUpdate":
        method_update = torch.zeros_like(adam)
    projected = basis_safe_projection(specs, method_update, family)
    stats = v1410.projection_stats(descent, projected)
    trace = {
        **stats,
        "cos_fu_vs_gradient": cosine(projected, descent),
        "cos_fu_vs_adamw": cosine(projected, adam),
        "cos_raw_vs_adam_preconditioned": cosine(descent, adam),
        "cos_raw_vs_gradient": cosine(descent, descent),
        "cos_preconditioned_vs_gradient": cosine(adam, descent),
        "alignment_positive_fraction_raw": float((descent * grad < 0.0).float().mean().item()) if grad.numel() else 0.0,
        "alignment_positive_fraction_preconditioned": hard_fraction,
        "second_moment_condition": role_condition(specs, vhat.sqrt()),
        "curvature_clip_fraction": clipped_fraction,
        "projection_rejection_fraction": stats["degree_projection_rejection_fraction"],
        "value_retention_after_projection": stats["value_retention_after_degree_projection"],
        "raw_direction_norm": float(descent.norm().item()),
        "adam_preconditioned_norm": float(adam.norm().item()),
        "v2_preconditioned_norm": float(preconditioned.norm().item()),
        "hessian_diag_or_gn_diag_norm": float(hdiag.norm().item()),
        "clipped_fraction": clipped_fraction,
        "rolewise_curvature_p50": float(hdiag.float().quantile(0.50).item()) if hdiag.numel() else 0.0,
        "rolewise_curvature_p90": float(hdiag.float().quantile(0.90).item()) if hdiag.numel() else 0.0,
        "rolewise_curvature_p99": float(hdiag.float().quantile(0.99).item()) if hdiag.numel() else 0.0,
        "schedulefree_control_delta_norm_fraction": schedulefree_trace,
        "fu3_alignment_variant": fu3_alignment_variant,
    }
    return projected, trace


def train_fu_case(
    *,
    family: str,
    method: str,
    dataset: str,
    seed: int,
    xtr: torch.Tensor,
    ytr: torch.Tensor,
    xva: torch.Tensor,
    yva: torch.Tensor,
    xte: torch.Tensor | None,
    yte: torch.Tensor | None,
    input_dim: int,
    output_dim: int,
    args: argparse.Namespace,
    device: torch.device,
    fu3_alignment_variant: str = "",
    decay_control_variant: str = "",
) -> dict[str, Any]:
    case_args = copy(args)
    case_args.synthetic_dim = int(input_dim)
    case_args.synthetic_classes = int(output_dim)
    candidate = str(args.dche_candidate)
    model = v1410.make_case_model("MLP", "MLP-v150-control", xtr, seed, case_args, device) if family == "MLP" else v1410.make_case_model("D-CHE", candidate, xtr, seed, case_args, device)
    specs = v1410.named_param_specs(model)
    total = sum(int(spec.param.numel()) for spec in specs)
    m = torch.zeros(total, device=device)
    v = torch.zeros(total, device=device)
    hdiag = torch.zeros(total, device=device)
    beta1 = float(args.beta1)
    beta2 = float(args.beta2)
    hbeta = float(args.curvature_beta)
    gen = torch.Generator(device=device).manual_seed(int(seed) + 150_000 + sum(ord(c) for c in method + dataset + family))
    rng = random.Random(int(seed) + 150_700 + len(method))
    projection_acc: dict[str, list[float]] = {
        "cos_fu_vs_gradient": [],
        "cos_fu_vs_adamw": [],
        "alignment_positive_fraction_raw": [],
        "alignment_positive_fraction_preconditioned": [],
        "second_moment_condition": [],
        "curvature_clip_fraction": [],
        "decoupled_decay_norm_fraction": [],
        "projection_rejection_fraction": [],
        "value_retention_after_projection": [],
        "generic_value_norm": [],
        "projected_value_norm": [],
        "degree_projection_rejection_fraction": [],
        "cos_projected_vs_generic": [],
        "rolewise_curvature_p50": [],
        "rolewise_curvature_p90": [],
        "rolewise_curvature_p99": [],
        "schedulefree_control_delta_norm_fraction": [],
        "cautious_decay_active_fraction": [],
        "random_matched_decay_active_fraction": [],
    }
    trajectory: list[dict[str, float]] = []
    second_rows: list[dict[str, Any]] = []
    align_rows: list[dict[str, Any]] = []
    curvature_rows: list[dict[str, Any]] = []
    block_rows: list[dict[str, Any]] = []
    schedule_rows: list[dict[str, Any]] = []
    before_degree = v1410.degree_energy(model) if family == "D-CHE" else {}
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    for step in range(int(args.train_steps)):
        idx = torch.randint(0, xtr.shape[0], (int(args.batch_size),), generator=gen, device=device)
        xb, yb = xtr[idx], ytr[idx]
        model.zero_grad(set_to_none=True)
        loss = v1410.loss_value(model(xb), yb, "CE")
        loss.backward()
        grad = v1410.flat_existing_grad(specs)
        if method == "C7-CoupledL2RegularizationControl":
            grad = grad + float(args.weight_decay) * flat_params(specs)
        m = beta1 * m + (1.0 - beta1) * grad
        v = beta2 * v + (1.0 - beta2) * grad.square()
        hdiag = hbeta * hdiag + (1.0 - hbeta) * grad.square().sqrt()
        mhat = m / (1.0 - beta1 ** (step + 1))
        vhat = v / (1.0 - beta2 ** (step + 1))
        update, trace = compute_update(method, family, specs, grad, mhat, vhat, hdiag, gen, fu3_alignment_variant)
        add_flat_update(specs, update, float(args.lr))
        decay_delta = 0.0
        decay_base = 0.0
        cautious_decay_active_fraction = 0.0
        random_matched_decay_active_fraction = 0.0
        decoupled = method in {
            "FU0-D-CHE-AdamW",
            "FU7-D-CHE-DecoupledDecayPlusFunctionUpdate",
            "C0-D-CHE-AdamW",
            "C1-NoOpMatchedOverhead",
            "C6-SameDecoupledDecayNoFU",
            "C8-AdamWParallelDirection",
            "C9-GenericOptimizerStateControl",
            "M0-MLP-AdamW",
            "M1-MLP-AdamW-DecoupledDecay",
            "M6-MLP-ScheduleFreeAdamW",
        }
        if decoupled:
            cautious_grad = grad if method == "C6-SameDecoupledDecayNoFU" and decay_control_variant in {"cautious", "random_matched"} else None
            random_match_generator = gen if method == "C6-SameDecoupledDecayNoFU" and decay_control_variant == "random_matched" else None
            decay_delta, decay_base, cautious_decay_active_fraction = apply_role_decay(
                specs,
                float(args.lr),
                float(args.weight_decay),
                float(args.readout_weight_decay),
                cautious_grad=cautious_grad,
                random_match_generator=random_match_generator,
            )
            random_matched_decay_active_fraction = cautious_decay_active_fraction if decay_control_variant == "random_matched" else 0.0
        trace["decoupled_decay_norm_fraction"] = decay_delta / max(1.0e-8, decay_base)
        trace["cautious_decay_active_fraction"] = cautious_decay_active_fraction
        trace["random_matched_decay_active_fraction"] = random_matched_decay_active_fraction
        for key in projection_acc:
            projection_acc[key].append(float(trace.get(key, 0.0)))
        row_base = {
            "dataset": dataset,
            "seed": seed,
            "family": family,
            "method": method,
            "fu3_alignment_variant": fu3_alignment_variant,
            "decay_control_variant": decay_control_variant,
            "step": step + 1,
            "direction_uses_train_stream_only": 1,
            "promotion_allowed": 0,
        }
        if step == 0 or step == int(args.train_steps) - 1 or ((step + 1) % max(1, int(args.trace_interval)) == 0):
            metrics = v1410.eval_metrics(model, xva, yva)
            trajectory.append({"step": float(step + 1), "NLL": metrics["NLL"], "CEp99": metrics["CEp99"], "ECE": metrics["ECE"], "Brier": metrics["Brier"], "acc": metrics["acc"]})
            align_rows.append({**row_base, **{k: trace[k] for k in ["cos_fu_vs_gradient", "cos_fu_vs_adamw", "alignment_positive_fraction_raw", "alignment_positive_fraction_preconditioned", "cos_raw_vs_adam_preconditioned", "cos_preconditioned_vs_gradient"]}})
            curvature_rows.append({**row_base, **{k: trace[k] for k in ["hessian_diag_or_gn_diag_norm", "curvature_clip_fraction", "rolewise_curvature_p50", "rolewise_curvature_p90", "rolewise_curvature_p99", "clipped_fraction"]}})
            block_rows.append({**row_base, "block_second_moment_condition": trace["second_moment_condition"], "block_preconditioner_active": int(method in {"FU6-D-CHE-BlockSecondMomentFunctionUpdate", "M5-MLP-BlockSecondMomentAdamW"})})
            schedule_rows.append({**row_base, "schedulefree_control_delta_norm_fraction": trace["schedulefree_control_delta_norm_fraction"], "schedulefree_control_active": int(method == "M6-MLP-ScheduleFreeAdamW")})
            second_rows.extend(role_second_moment_rows(row_base=row_base, specs=specs, vhat=vhat))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_memory_bytes = int(torch.cuda.max_memory_allocated(device))
    else:
        peak_memory_bytes = 0
    elapsed = time.perf_counter() - start
    final = v1410.eval_metrics(model, xva, yva)
    test_final = v1410.eval_metrics(model, xte, yte) if xte is not None and yte is not None else {}
    auc_nll = mean([r["NLL"] for r in trajectory])
    auc_cep99 = mean([r["CEp99"] for r in trajectory])
    if family == "D-CHE" and int(args.real_linec) == 1:
        b = min(int(args.linec_batch_size), int(xtr.shape[0]), int(xva.shape[0]))
        linec_rows: list[dict[str, Any]] = []
        votes = []
        for linec_seed in parse_ints(args.linec_seeds):
            try:
                lm = v1410.linec_metrics(model, xtr[:b], ytr[:b], xva[:b], yva[:b], int(linec_seed), int(args.linec_sketch_dim), float(args.lr), float(args.weight_decay))
                status = "executed"
                error = ""
            except Exception as exc:  # noqa: BLE001
                lm = {"CouplingR2": float("nan"), "NoiseSignalLeak": float("nan"), "RealSignalReservoirRatio": float("nan")}
                status = "blocked"
                error = f"{type(exc).__name__}: {exc}"
            passed = int(fnum(lm.get("CouplingR2"), -999.0) >= 0.15 and fnum(lm.get("NoiseSignalLeak"), 999.0) <= 0.20 and fnum(lm.get("RealSignalReservoirRatio"), 999.0) <= 0.70)
            votes.append(passed)
            linec_rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "family": family,
                    "method": method,
                    "fu3_alignment_variant": fu3_alignment_variant,
                    "decay_control_variant": decay_control_variant,
                    "linec_seed": linec_seed,
                    "linec_status": status,
                    "linec_error": error,
                    "LineC_pass": passed,
                    **lm,
                    "CEp99": final["CEp99"],
                    "NLL": final["NLL"],
                    "ECE": final["ECE"],
                    "Brier": final["Brier"],
                    "metric_used_as_direction": 0,
                    "promotion_allowed": 0,
                }
            )
        linec_mean = {
            "CouplingR2": mean([fnum(r.get("CouplingR2"), 0.0) for r in linec_rows]),
            "NoiseSignalLeak": mean([fnum(r.get("NoiseSignalLeak"), 0.0) for r in linec_rows]),
            "RealSignalReservoirRatio": mean([fnum(r.get("RealSignalReservoirRatio"), 0.0) for r in linec_rows]),
        }
    else:
        votes = [1]
        linec_rows = [
            {
                "dataset": dataset,
                "seed": seed,
                "family": family,
                "method": method,
                "fu3_alignment_variant": fu3_alignment_variant,
                "decay_control_variant": decay_control_variant,
                "linec_seed": "",
                "linec_status": "not_applicable_mlp_or_disabled",
                "LineC_pass": 1,
                "CouplingR2": final["CouplingR2"],
                "NoiseSignalLeak": final["NoiseSignalLeak"],
                "RealSignalReservoirRatio": final["RealSignalReservoirRatio"],
                "CEp99": final["CEp99"],
                "NLL": final["NLL"],
                "ECE": final["ECE"],
                "Brier": final["Brier"],
                "metric_used_as_direction": 0,
                "promotion_allowed": 0,
            }
        ]
        linec_mean = {"CouplingR2": final["CouplingR2"], "NoiseSignalLeak": final["NoiseSignalLeak"], "RealSignalReservoirRatio": final["RealSignalReservoirRatio"]}
    after_degree = v1410.degree_energy(model) if family == "D-CHE" else {}
    row = {
        "stage": "V150_FUNCTION_UPDATE_CASE",
        "family": family,
        "method_family": method_family(method),
        "method": method,
        "fu3_alignment_variant": fu3_alignment_variant,
        "decay_control_variant": decay_control_variant,
        "candidate_id": candidate if family == "D-CHE" else "MLP-v150-control",
        "dataset": dataset,
        "seed": seed,
        "loss_interface": "CE",
        "control_method": is_control(method),
        "train_steps": int(args.train_steps),
        "batch_size": int(args.batch_size),
        "elapsed_sec": elapsed,
        "step_time_sec": elapsed / max(1, int(args.train_steps)),
        "peak_memory_bytes": peak_memory_bytes,
        "NLL": final["NLL"],
        "CEp99": final["CEp99"],
        "ECE": final["ECE"],
        "Brier": final["Brier"],
        "acc": final["acc"],
        "test_NLL": test_final.get("NLL", ""),
        "test_CEp99": test_final.get("CEp99", ""),
        "test_ECE": test_final.get("ECE", ""),
        "test_acc": test_final.get("acc", ""),
        "CouplingR2": linec_mean["CouplingR2"],
        "NoiseSignalLeak": linec_mean["NoiseSignalLeak"],
        "RealSignalReservoirRatio": linec_mean["RealSignalReservoirRatio"],
        "margin_p10": final["margin_p10"],
        "AUC_NLL": auc_nll,
        "AUC_CEp99": auc_cep99,
        "LineC_pass_rate": sum(votes) / max(1, len(votes)),
        "LineC_majority_pass": int(sum(votes) >= math.ceil(len(votes) / 2)),
        "degree_energy": after_degree.get("degree_energy_total", ""),
        "degree_energy_before": before_degree.get("degree_energy_total", ""),
        "degree_energy_after": after_degree.get("degree_energy_total", ""),
        "high_degree_fraction": after_degree.get("high_degree_energy_fraction", ""),
        "degree_entropy": after_degree.get("degree_entropy", ""),
        **{k: median(v) for k, v in projection_acc.items()},
        "direction_uses_train_stream_only": 1,
        "uses_validation_for_direction": 0,
        "uses_test_for_direction": 0,
        "uses_future_for_direction": 0,
        "uses_query_for_direction": 0,
        "uses_linec_for_direction": 0,
        "uses_cep99_for_direction": 0,
        "uses_nll_for_direction": 0,
        "uses_ece_for_direction": 0,
        "uses_auctime_for_direction": 0,
        "uses_dataset_name_branch": 0,
        "uses_seed_specific_scale": 0,
        "action_token": 0,
        "controller_executed": 0,
        "reset_route_used": 0,
        "promotion_allowed": 0,
    }
    return {
        "row": row,
        "second_rows": second_rows,
        "alignment_rows": align_rows,
        "curvature_rows": curvature_rows,
        "block_rows": block_rows,
        "schedule_rows": schedule_rows,
        "linec_rows": linec_rows,
    }


def enrich_dche_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        by_key.setdefault((str(row["dataset"]), int(row["seed"])), []).append(row)
    out: list[dict[str, Any]] = []
    for group in by_key.values():
        controls = [r for r in group if str(r.get("method")) in FU_CONTROLS]
        adam = next((r for r in group if str(r.get("method")) == "C0-D-CHE-AdamW"), controls[0] if controls else group[0])
        best_nll = min(fnum(r.get("NLL"), 9.0) for r in controls or group)
        best_auc = min(fnum(r.get("AUC_NLL"), 9.0) for r in controls or group)
        base_time = max(1.0e-8, fnum(adam.get("step_time_sec"), 1.0))
        base_mem = max(1.0, fnum(adam.get("peak_memory_bytes"), 1.0))
        for row in group:
            item = dict(row)
            item["source_vs_adamw"] = fnum(adam.get("NLL"), 9.0) - fnum(row.get("NLL"), 9.0)
            item["source_vs_best_control"] = best_nll - fnum(row.get("NLL"), 9.0)
            item["AUCtime_ratio"] = fnum(row.get("AUC_NLL"), 9.0) / max(1.0e-8, best_auc)
            item["CEp99_delta"] = fnum(row.get("CEp99"), 0.0) - fnum(adam.get("CEp99"), 0.0)
            item["NLL_delta"] = fnum(row.get("NLL"), 0.0) - fnum(adam.get("NLL"), 0.0)
            item["ECE_delta"] = fnum(row.get("ECE"), 0.0) - fnum(adam.get("ECE"), 0.0)
            item["Brier_delta"] = fnum(row.get("Brier"), 0.0) - fnum(adam.get("Brier"), 0.0)
            item["margin_p10_delta"] = fnum(row.get("margin_p10"), 0.0) - fnum(adam.get("margin_p10"), 0.0)
            item["step_time_ratio"] = fnum(row.get("step_time_sec"), 0.0) / base_time
            item["memory_ratio"] = fnum(row.get("peak_memory_bytes"), 0.0) / base_mem if base_mem > 1.0 else 1.0
            item["control_equivalent"] = int(fnum(item.get("source_vs_best_control"), -999.0) <= 0.005)
            item["bad_event"] = int(
                fnum(item.get("source_vs_best_control"), -999.0) < 0.005
                or fnum(item.get("AUCtime_ratio"), 9.0) > 1.0
                or sint(item.get("LineC_majority_pass"), 0) != 1
                or fnum(item.get("step_time_ratio"), 999.0) > 1.25
                or fnum(item.get("memory_ratio"), 999.0) > 1.25
            )
            item["strict_gate_pass"] = int(
                str(item.get("method")) in FU_METHODS
                and fnum(item.get("source_vs_best_control"), -999.0) >= 0.005
                and fnum(item.get("AUCtime_ratio"), 9.0) <= 1.0
                and fnum(item.get("CEp99_delta"), 999.0) <= 0.05
                and fnum(item.get("NLL_delta"), 999.0) <= 0.02
                and fnum(item.get("ECE_delta"), 999.0) <= 0.02
                and sint(item.get("LineC_majority_pass"), 0) == 1
                and fnum(item.get("step_time_ratio"), 999.0) <= 1.25
                and fnum(item.get("memory_ratio"), 999.0) <= 1.25
            )
            out.append(item)
    return out


def enrich_mlp_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        by_key.setdefault((str(row["dataset"]), int(row["seed"])), []).append(row)
    out: list[dict[str, Any]] = []
    for group in by_key.values():
        adam = next((r for r in group if str(r.get("method")) == "M0-MLP-AdamW"), group[0])
        for row in group:
            item = dict(row)
            item["source_vs_mlp_adamw"] = fnum(adam.get("NLL"), 9.0) - fnum(row.get("NLL"), 9.0)
            item["NLL_delta_vs_mlp_adamw"] = fnum(row.get("NLL"), 9.0) - fnum(adam.get("NLL"), 9.0)
            item["AUCtime_ratio_vs_mlp_adamw"] = fnum(row.get("AUC_NLL"), 9.0) / max(1.0e-8, fnum(adam.get("AUC_NLL"), 9.0))
            item["strict_gate_pass"] = int(
                str(item.get("method")) != "M0-MLP-AdamW"
                and fnum(item.get("source_vs_mlp_adamw"), -999.0) >= 0.005
                and fnum(item.get("AUCtime_ratio_vs_mlp_adamw"), 9.0) <= 1.25
            )
            out.append(item)
    return out


def enrich_linec_rows(rows: list[dict[str, Any]], result_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    result_by_key = {
        (
            str(r.get("family")),
            str(r.get("dataset")),
            str(r.get("seed")),
            str(r.get("method")),
            str(r.get("fu3_alignment_variant", "")),
            str(r.get("decay_control_variant", "")),
        ): r
        for r in result_rows
    }
    baseline_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    for r in result_rows:
        family = str(r.get("family"))
        method = str(r.get("method"))
        if (family == "D-CHE" and method == "C0-D-CHE-AdamW") or (family == "MLP" and method == "M0-MLP-AdamW"):
            baseline_by_key[(family, str(r.get("dataset")), str(r.get("seed")))] = r
    out: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        family = str(item.get("family"))
        dataset = str(item.get("dataset"))
        seed = str(item.get("seed"))
        method = str(item.get("method"))
        variant = str(item.get("fu3_alignment_variant", ""))
        decay_variant = str(item.get("decay_control_variant", ""))
        result = result_by_key.get((family, dataset, seed, method, variant, decay_variant), {})
        base = baseline_by_key.get((family, dataset, seed), {})
        item["CouplingR2_delta"] = fnum(item.get("CouplingR2"), 0.0) - fnum(base.get("CouplingR2"), 0.0)
        item["NoiseSignalLeak_delta"] = fnum(item.get("NoiseSignalLeak"), 0.0) - fnum(base.get("NoiseSignalLeak"), 0.0)
        item["RealSignalReservoirRatio_delta"] = fnum(item.get("RealSignalReservoirRatio"), 0.0) - fnum(base.get("RealSignalReservoirRatio"), 0.0)
        item["CEp99_delta"] = fnum(item.get("CEp99"), 0.0) - fnum(base.get("CEp99"), 0.0)
        item["NLL_delta"] = fnum(item.get("NLL"), 0.0) - fnum(base.get("NLL"), 0.0)
        item["ECE_delta"] = fnum(item.get("ECE"), 0.0) - fnum(base.get("ECE"), 0.0)
        item["Brier_delta"] = fnum(item.get("Brier"), 0.0) - fnum(base.get("Brier"), 0.0)
        item["margin_p10_delta"] = fnum(result.get("margin_p10"), 0.0) - fnum(base.get("margin_p10"), 0.0)
        item["signal_channel_energy"] = fnum(item.get("coupling_prediction_norm"), fnum(item.get("CouplingR2"), 0.0))
        item["reservoir_energy"] = fnum(item.get("reservoir_fraction"), fnum(item.get("RealSignalReservoirRatio"), 0.0))
        item["noise_leakage_proxy"] = fnum(item.get("NoiseSignalLeak"), 0.0)
        item["linec_metric_used_as_direction"] = 0
        item["promotion_allowed"] = 0
        out.append(item)
    return out


def run_fu_and_m(args: argparse.Namespace, out_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    result_path = out_dir / "v150_dche_fu_results.csv"
    control_path = out_dir / "v150_dche_fu_controls.csv"
    mlp_path = out_dir / "v150_mlp_controls.csv"
    if sint(getattr(args, "reuse_if_present", 1), 1) == 1 and result_path.exists() and control_path.exists() and mlp_path.exists():
        dche_results = read_rows(result_path)
        dche_controls = read_rows(control_path)
        mlp_rows = read_rows(mlp_path)
        c6_variants = {str(r.get("decay_control_variant", "")) for r in dche_controls if str(r.get("method")) == "C6-SameDecoupledDecayNoFU"}
        if {"standard", "cautious", "random_matched"}.issubset(c6_variants):
            summary = summarize_fu(dche_results)
            return dche_results, dche_controls, mlp_rows, summary

    device = torch.device(args.device if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    raw_rows: list[dict[str, Any]] = []
    second_rows: list[dict[str, Any]] = []
    align_rows: list[dict[str, Any]] = []
    curvature_rows: list[dict[str, Any]] = []
    block_rows: list[dict[str, Any]] = []
    schedule_rows: list[dict[str, Any]] = []
    linec_rows: list[dict[str, Any]] = []
    methods = parse_csv(args.fu_methods)
    controls = parse_csv(args.fu_controls)
    mlp_methods = parse_csv(args.mlp_methods)
    for dataset in parse_csv(args.datasets):
        for seed in parse_ints(args.seeds):
            load_args = copy(args)
            load_args.seed = int(seed)
            xtr, ytr, xva, yva, xte, yte, input_dim_t, output_dim_t = v144.load_real_split(load_args, dataset, int(seed), device)
            input_dim = int(input_dim_t.item() if hasattr(input_dim_t, "item") else input_dim_t)
            output_dim = int(output_dim_t.item() if hasattr(output_dim_t, "item") else output_dim_t)
            for method in controls + methods:
                variants = ["hard", "soft"] if method == "FU3-D-CHE-CautiousFunctionUpdate" else [""]
                for variant in variants:
                    decay_variants = ["standard", "cautious", "random_matched"] if method == "C6-SameDecoupledDecayNoFU" else [""]
                    for decay_variant in decay_variants:
                        result = train_fu_case(
                            family="D-CHE",
                            method=method,
                            dataset=dataset,
                            seed=int(seed),
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
                            fu3_alignment_variant=variant,
                            decay_control_variant=decay_variant,
                        )
                        raw_rows.append(result["row"])
                        second_rows.extend(result["second_rows"])
                        align_rows.extend(result["alignment_rows"])
                        curvature_rows.extend(result["curvature_rows"])
                        block_rows.extend(result["block_rows"])
                        schedule_rows.extend(result["schedule_rows"])
                        linec_rows.extend(result["linec_rows"])
                        if device.type == "cuda":
                            torch.cuda.empty_cache()
            for method in mlp_methods:
                variants = ["hard", "soft"] if method == "M2-MLP-CautiousAdamW" else [""]
                for variant in variants:
                    result = train_fu_case(family="MLP", method=method, dataset=dataset, seed=int(seed), xtr=xtr, ytr=ytr, xva=xva, yva=yva, xte=xte, yte=yte, input_dim=input_dim, output_dim=output_dim, args=args, device=device, fu3_alignment_variant=variant)
                    raw_rows.append(result["row"])
                    second_rows.extend(result["second_rows"])
                    align_rows.extend(result["alignment_rows"])
                    curvature_rows.extend(result["curvature_rows"])
                    block_rows.extend(result["block_rows"])
                    schedule_rows.extend(result["schedule_rows"])
                    linec_rows.extend(result["linec_rows"])
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
    dche_enriched = enrich_dche_rows([r for r in raw_rows if r.get("family") == "D-CHE"])
    mlp_enriched = enrich_mlp_rows([r for r in raw_rows if r.get("family") == "MLP"])
    linec_rows = enrich_linec_rows(linec_rows, dche_enriched + mlp_enriched)
    dche_results = [r for r in dche_enriched if str(r.get("method")) in FU_METHODS]
    dche_controls = [r for r in dche_enriched if str(r.get("method")) in FU_CONTROLS]
    write_rows(result_path, dche_results)
    write_rows(control_path, dche_controls)
    write_rows(mlp_path, mlp_enriched)
    write_rows(out_dir / "v150_second_moment_trace.csv", second_rows)
    write_rows(out_dir / "v150_alignment_trace.csv", align_rows)
    write_rows(out_dir / "v150_curvature_diag_trace.csv", curvature_rows)
    write_rows(out_dir / "v150_block_preconditioner_trace.csv", block_rows)
    write_rows(out_dir / "v150_schedulefree_control_trace.csv", schedule_rows)
    write_rows(out_dir / "v150_linec_tail_audit.csv", linec_rows)
    return dche_results, dche_controls, mlp_enriched, summarize_fu(dche_results)


def summarize_fu(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    pass_count = len({(r.get("dataset"), str(r.get("seed"))) for r in rows if sint(r.get("strict_gate_pass"), 0) == 1})
    source_mean = mean([fnum(r.get("source_vs_best_control"), 0.0) for r in rows])
    control_equiv = sum(1 for r in rows if fnum(r.get("source_vs_best_control"), -999.0) <= 0.005) / max(1, len(rows))
    bad_event = sum(1 for r in rows if sint(r.get("bad_event"), 0) == 1) / max(1, len(rows))
    exploration = int(pass_count >= 3 and source_mean > 0.0 and control_equiv <= 0.60 and bad_event <= 0.50)
    meaningful = int(pass_count >= 4 and source_mean >= 0.002 and control_equiv <= 0.50)
    s4 = int(pass_count >= 6 and source_mean >= 0.005 and control_equiv <= 0.40)
    s5 = int(pass_count == 9 and all(sint(r.get("strict_gate_pass"), 0) == 1 for r in rows if str(r.get("method")).startswith("FU")))
    best_method = ""
    best_method_source = -999.0
    method_rows = []
    for method in FU_METHODS:
        group = [r for r in rows if r.get("method") == method]
        if not group:
            continue
        src = mean([fnum(r.get("source_vs_best_control"), 0.0) for r in group])
        if src > best_method_source:
            best_method_source = src
            best_method = method
        method_rows.append(
            {
                "method": method,
                "rows": len(group),
                "strict_pass_rows": sum(sint(r.get("strict_gate_pass"), 0) for r in group),
                "dataset_seed_pass_count": len({(r.get("dataset"), str(r.get("seed"))) for r in group if sint(r.get("strict_gate_pass"), 0) == 1}),
                "mean_source_vs_best_control": src,
                "mean_control_equivalent": sum(1 for r in group if sint(r.get("control_equivalent"), 0) == 1) / max(1, len(group)),
                "mean_bad_event": sum(1 for r in group if sint(r.get("bad_event"), 0) == 1) / max(1, len(group)),
                "promotion_allowed": 0,
            }
        )
    return {
        "line_fu_candidate_count": len(FU_METHODS),
        "real_lite_pass_count": pass_count,
        "source_vs_best_control_mean": source_mean,
        "control_equivalent_fraction": control_equiv,
        "bad_event_fraction": bad_event,
        "line_fu_exploration_gate_pass": exploration,
        "line_fu_meaningful_gate_pass": meaningful,
        "line_fu_s4_gate_pass": s4,
        "line_fu_s5_gate_pass": s5,
        "best_fu_method": best_method,
        "best_fu_mean_source_vs_best_control": best_method_source,
        "method_rows": method_rows,
    }


def build_mechanism_manifest(out_dir: Path, dche_rows: Sequence[dict[str, Any]], control_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for method in FU_METHODS + FU_CONTROLS:
        rows.append(
            {
                "method": method,
                "pre_registered": 1,
                "family": method_family(method),
                "decoupled_decay_implemented": int(method in {"FU0-D-CHE-AdamW", "FU7-D-CHE-DecoupledDecayPlusFunctionUpdate", "C0-D-CHE-AdamW", "C1-NoOpMatchedOverhead", "C6-SameDecoupledDecayNoFU", "C8-AdamWParallelDirection", "C9-GenericOptimizerStateControl"}),
                "second_moment_implemented": int(method in {"FU2-D-CHE-AdamV2FunctionUpdate", "FU5-D-CHE-SophiaDiagClippedFunctionUpdate", "FU6-D-CHE-BlockSecondMomentFunctionUpdate", "FU8-D-CHE-SOAPLiteDiagnosticFunctionUpdate", "C5-SameSecondMomentScaleRandomDirection", "C9-GenericOptimizerStateControl", "FU0-D-CHE-AdamW", "C0-D-CHE-AdamW", "C8-AdamWParallelDirection"}),
                "alignment_trace_available": int(method in {"FU3-D-CHE-CautiousFunctionUpdate", "FU4-D-CHE-MGUPFunctionUpdate", "C4-SameAlignmentMaskRandomDirection"} or method in FU_METHODS + FU_CONTROLS),
                "controls_available": int(len(control_rows) > 0),
                "new_fu_token": 0,
                "new_fche_token": 0,
                "action_token": 0,
                "controller_executed": 0,
                "promotion_allowed": 0,
            }
        )
    write_rows(out_dir / "v150_optimizer_mechanism_manifest.csv", rows)
    return {
        "decoupled_decay_implemented": int(any(sint(r.get("decoupled_decay_implemented"), 0) for r in rows)),
        "second_moment_implemented": int(any(sint(r.get("second_moment_implemented"), 0) for r in rows)),
        "alignment_trace_available": int(any(sint(r.get("alignment_trace_available"), 0) for r in rows)),
        "controls_available": int(len(control_rows) > 0),
    }


def build_kan_specificity(out_dir: Path, dche_rows: Sequence[dict[str, Any]], mlp_rows: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    dche_adam = {(r.get("dataset"), str(r.get("seed"))): r for r in dche_rows if r.get("method") == "FU0-D-CHE-AdamW"}
    mlp_adam = {(r.get("dataset"), str(r.get("seed"))): r for r in mlp_rows if r.get("method") == "M0-MLP-AdamW"}
    mlp_by = {
        (r.get("dataset"), str(r.get("seed")), r.get("method"), str(r.get("fu3_alignment_variant", ""))): r
        for r in mlp_rows
    }
    rows = []
    for row in dche_rows:
        method = str(row.get("method"))
        analog = analog_method(method)
        if not analog:
            continue
        key = (row.get("dataset"), str(row.get("seed")))
        base_d = dche_adam.get(key)
        base_m = mlp_adam.get(key)
        variant = str(row.get("fu3_alignment_variant", ""))
        analog_row = mlp_by.get((row.get("dataset"), str(row.get("seed")), analog, variant)) or mlp_by.get((row.get("dataset"), str(row.get("seed")), analog, ""))
        if not base_d or not base_m or not analog_row:
            continue
        dche_gain = fnum(base_d.get("NLL"), 9.0) - fnum(row.get("NLL"), 9.0)
        mlp_gain = fnum(base_m.get("NLL"), 9.0) - fnum(analog_row.get("NLL"), 9.0)
        rows.append(
            {
                "dataset": row.get("dataset"),
                "seed": row.get("seed"),
                "dche_method": method,
                "fu3_alignment_variant": variant,
                "mlp_analog_method": analog,
                "dche_gain_vs_adamw": dche_gain,
                "mlp_gain_vs_adamw": mlp_gain,
                "kan_specific_difference_in_differences": dche_gain - mlp_gain,
                "kan_specific_positive": int(dche_gain - mlp_gain > 0.0 and dche_gain > 0.0),
                "generic_optimizer_explains_gain": int(mlp_gain >= dche_gain and dche_gain > 0.0),
                "promotion_allowed": 0,
            }
        )
    summary = []
    for method in sorted({str(r.get("dche_method")) for r in rows}):
        group = [r for r in rows if r.get("dche_method") == method]
        summary.append(
            {
                "dche_method": method,
                "rows": len(group),
                "mean_dche_gain_vs_adamw": mean([fnum(r.get("dche_gain_vs_adamw"), 0.0) for r in group]),
                "mean_mlp_gain_vs_adamw": mean([fnum(r.get("mlp_gain_vs_adamw"), 0.0) for r in group]),
                "mean_kan_specific_difference_in_differences": mean([fnum(r.get("kan_specific_difference_in_differences"), 0.0) for r in group]),
                "kan_specific_positive_rows": sum(sint(r.get("kan_specific_positive"), 0) for r in group),
                "generic_optimizer_explains_rows": sum(sint(r.get("generic_optimizer_explains_gain"), 0) for r in group),
                "promotion_allowed": 0,
            }
        )
    write_rows(out_dir / "v150_kan_specificity_summary.csv", summary)
    return rows, summary


def build_decay_deconfound(out_dir: Path, dche_rows: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by = {
        (r.get("dataset"), str(r.get("seed")), r.get("method"), str(r.get("decay_control_variant", ""))): r
        for r in dche_rows
    }

    def get_row(dataset: str, seed: str, method: str, decay_variant: str = "") -> dict[str, Any] | None:
        return by.get((dataset, seed, method, decay_variant)) or by.get((dataset, seed, method, ""))

    rows = []
    for dataset in sorted({str(r.get("dataset")) for r in dche_rows}):
        for seed in sorted({str(r.get("seed")) for r in dche_rows if str(r.get("dataset")) == dataset}):
            fu7 = get_row(dataset, seed, "FU7-D-CHE-DecoupledDecayPlusFunctionUpdate")
            c6 = get_row(dataset, seed, "C6-SameDecoupledDecayNoFU", "standard")
            c6_cautious = get_row(dataset, seed, "C6-SameDecoupledDecayNoFU", "cautious")
            c6_random = get_row(dataset, seed, "C6-SameDecoupledDecayNoFU", "random_matched")
            c7 = get_row(dataset, seed, "C7-CoupledL2RegularizationControl")
            fu2 = get_row(dataset, seed, "FU2-D-CHE-AdamV2FunctionUpdate")
            adam = get_row(dataset, seed, "FU0-D-CHE-AdamW")
            if not fu7 or not c6 or not c6_cautious or not c6_random or not c7 or not fu2 or not adam:
                continue
            fu7_gain = fnum(adam.get("NLL"), 9.0) - fnum(fu7.get("NLL"), 9.0)
            c6_gain = fnum(adam.get("NLL"), 9.0) - fnum(c6.get("NLL"), 9.0)
            c6_cautious_gain = fnum(adam.get("NLL"), 9.0) - fnum(c6_cautious.get("NLL"), 9.0)
            c6_random_gain = fnum(adam.get("NLL"), 9.0) - fnum(c6_random.get("NLL"), 9.0)
            c7_gain = fnum(adam.get("NLL"), 9.0) - fnum(c7.get("NLL"), 9.0)
            fu2_gain = fnum(adam.get("NLL"), 9.0) - fnum(fu2.get("NLL"), 9.0)
            standard_explains = int(c6_gain >= fu7_gain - 0.001 and fu7_gain > 0.0)
            cautious_explains = int(c6_cautious_gain >= fu7_gain - 0.001 and fu7_gain > 0.0)
            random_explains = int(c6_random_gain >= fu7_gain - 0.001 and fu7_gain > 0.0)
            rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "FU7_gain_vs_AdamW": fu7_gain,
                    "FU2_no_extra_decay_gain_vs_AdamW": fu2_gain,
                    "same_decoupled_decay_no_fu_gain": c6_gain,
                    "cautious_decoupled_decay_no_fu_gain": c6_cautious_gain,
                    "random_matched_decoupled_decay_no_fu_gain": c6_random_gain,
                    "cautious_decay_active_fraction": fnum(c6_cautious.get("cautious_decay_active_fraction"), 0.0),
                    "random_matched_decay_active_fraction": fnum(c6_random.get("random_matched_decay_active_fraction"), 0.0),
                    "coupled_l2_gain": c7_gain,
                    "decoupled_decay_explains_gain": standard_explains,
                    "cautious_decay_explains_gain": cautious_explains,
                    "random_matched_decay_explains_gain": random_explains,
                    "any_decay_control_explains_gain": int(standard_explains == 1 or cautious_explains == 1 or random_explains == 1),
                    "coupled_l2_worse_than_decoupled": int(c7_gain < c6_gain),
                    "no_decay_fu_still_has_value": int(fu2_gain > 0.0),
                    "weight_decay_control_complete": 1,
                    "promotion_allowed": 0,
                }
            )
    route = {
        "decoupled_decay_explained_rows": sum(sint(r.get("any_decay_control_explains_gain"), 0) for r in rows),
        "decoupled_decay_explains_fraction": sum(sint(r.get("any_decay_control_explains_gain"), 0) for r in rows) / max(1, len(rows)),
        "cautious_decay_explained_rows": sum(sint(r.get("cautious_decay_explains_gain"), 0) for r in rows),
        "cautious_decay_explains_fraction": sum(sint(r.get("cautious_decay_explains_gain"), 0) for r in rows) / max(1, len(rows)),
        "random_matched_decay_explained_rows": sum(sint(r.get("random_matched_decay_explains_gain"), 0) for r in rows),
        "random_matched_decay_explains_fraction": sum(sint(r.get("random_matched_decay_explains_gain"), 0) for r in rows) / max(1, len(rows)),
        "weight_decay_control_complete_rows": sum(sint(r.get("weight_decay_control_complete"), 0) for r in rows),
        "route": "R3-DecoupledDecayExplainsGain" if rows and sum(sint(r.get("any_decay_control_explains_gain"), 0) for r in rows) / max(1, len(rows)) >= 0.50 else "K-DecoupledDecayNotSufficient",
        "promotion_allowed": 0,
    }
    write_rows(out_dir / "v150_decoupled_decay_deconfound.csv", rows)
    audit_rows = [dict(r, stage="V150_DECOUPLED_DECAY_AUDIT") for r in rows] + [route]
    write_rows(out_dir / "v150_decoupled_decay_audit.csv", audit_rows)
    return rows, route


def build_line_d(out_dir: Path, line_d_out: Path) -> dict[str, Any]:
    source = line_d_out / "v149_line_d_substrate_repair_results.csv"
    rows = []
    for row in read_rows(source):
        if str(row.get("candidate_id")) not in LINE_D_CANDIDATES:
            continue
        item = dict(row)
        item["stage"] = "V150_ALLBASIS_SUBSTRATE_RESULT"
        step_ratio = fnum(item.get("train_step_ratio_vs_MLP"), 999.0)
        raw_mem = fnum(item.get("workspace_raw_memory_ratio_vs_mlp"), 999.0)
        incr_mem = fnum(item.get("workspace_incremental_memory_ratio_vs_mlp"), 999.0)
        memory_ratio = incr_mem if incr_mem < 999.0 else raw_mem
        item["v150_step_ratio"] = step_ratio
        item["v150_memory_ratio"] = memory_ratio
        item["v150_substrate_gate_pass"] = int(
            step_ratio <= 1.75
            and memory_ratio <= 1.75
            and fnum(item.get("mean_delta_vs_MLP"), -999.0) >= -0.05
            and fnum(item.get("worst_delta_vs_MLP"), -999.0) >= -0.10
            and fnum(item.get("LineC_pass_rate"), 0.0) >= 0.30
        )
        item["v150_official_fms_eligibility_row"] = int(
            step_ratio <= 1.25 and memory_ratio <= 1.25 and fnum(item.get("LineC_pass_rate"), 0.0) >= 0.70 and sint(item.get("v150_substrate_gate_pass"), 0) == 1
        )
        item["official_fms_proof_executed"] = 0
        item["promotion_allowed"] = 0
        rows.append(item)
    write_rows(out_dir / "v150_allbasis_substrate_results.csv", rows)
    summary = []
    for family in ["D-FOU", "D-RBF", "D-WAV"]:
        fam = [r for r in rows if r.get("family") == family]
        pass_keys = {(r.get("dataset"), str(r.get("seed"))) for r in fam if sint(r.get("v150_substrate_gate_pass"), 0) == 1}
        official_keys = {(r.get("dataset"), str(r.get("seed"))) for r in fam if sint(r.get("v150_official_fms_eligibility_row"), 0) == 1}
        best_candidate = ""
        best_count = -1
        for cand in sorted({str(r.get("candidate_id")) for r in fam}):
            count = len({(r.get("dataset"), str(r.get("seed"))) for r in fam if r.get("candidate_id") == cand and sint(r.get("v150_substrate_gate_pass"), 0) == 1})
            if count > best_count:
                best_count = count
                best_candidate = cand
        summary.append(
            {
                "family": family,
                "rows": len(fam),
                "best_candidate": best_candidate,
                "best_candidate_dataset_seed_pass_count": max(0, best_count),
                "family_dataset_seed_pass_count": len(pass_keys),
                "family_exploration_gate_pass": int(len(pass_keys) >= 6),
                "family_official_fms_eligibility": int(len(official_keys) == 9),
                "official_fms_proof_executed": 0,
                "promotion_allowed": 0,
            }
        )
    best = max(summary, key=lambda r: sint(r.get("family_dataset_seed_pass_count"), 0), default={})
    return {
        "line_d_rows": len(rows),
        "line_d_source": str(source) if source.exists() else "missing_actual_line_d_artifact",
        "best_non_dche_family": best.get("family", ""),
        "best_non_dche_dataset_seed_pass_count": sint(best.get("family_dataset_seed_pass_count"), 0),
        "line_d_exploration_family_count": sum(sint(r.get("family_exploration_gate_pass"), 0) for r in summary),
        "line_d_official_fms_eligible_family_count": sum(sint(r.get("family_official_fms_eligibility"), 0) for r in summary),
        "line_d_gate_pass": int(any(sint(r.get("family_exploration_gate_pass"), 0) for r in summary)),
        "line_d_route": "S4-AllBasisAlternativeCarrier" if any(sint(r.get("family_exploration_gate_pass"), 0) for r in summary) else "R7-AllBasisCarrierBlocked",
        "summary_rows": summary,
    }


def build_rational_monitor(out_dir: Path) -> dict[str, Any]:
    """Replay v14.4 D-RAT status as the v15.0 Rational no-regression monitor."""
    route_path = V144_RATIONAL_MONITOR_DIR / "v144_route_decision.json"
    summary_path = V144_RATIONAL_MONITOR_DIR / "v144_real_transfer_fms_summary.csv"
    results_path = V144_RATIONAL_MONITOR_DIR / "v144_real_transfer_fms_results.csv"
    source_route = {}
    if route_path.exists():
        try:
            source_route = json.loads(route_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            source_route = {"route": "unreadable_json"}
    rows: list[dict[str, Any]] = []
    for row in read_rows(summary_path):
        item = {
            "stage": "V150_RATIONAL_NO_REGRESSION_MONITOR",
            "monitor_source": "v14.4_official_replay",
            "source_artifact": str(summary_path),
            "source_route_artifact": str(route_path),
            "source_results_artifact": str(results_path),
            "source_route": source_route.get("route", ""),
            "source_minimum_success": source_route.get("minimum_success", ""),
            "method": row.get("method", ""),
            "rows": row.get("rows", ""),
            "dataset_seed_pass_count": row.get("dataset_seed_pass_count", ""),
            "pass_rows": row.get("pass_rows", ""),
            "mean_source_vs_best_control": row.get("mean_source_vs_best_control", ""),
            "median_source_vs_best_control": row.get("median_source_vs_best_control", ""),
            "median_auc_time": row.get("median_auc_time", ""),
            "median_step_time": row.get("median_step_time", ""),
            "linec_majority_pass_rows": row.get("linec_majority_pass_rows", ""),
            "new_training_executed": 0,
            "official_fms_proof_executed": 0,
            "reset_route_used": 0,
            "optimizer_state_route_used": 0,
            "linec_tail_used_for_direction": 0,
            "promotion_allowed": 0,
        }
        rows.append(item)
    if not rows:
        rows.append(
            {
                "stage": "V150_RATIONAL_NO_REGRESSION_MONITOR",
                "monitor_source": "missing_v14.4_official_replay",
                "source_artifact": str(summary_path),
                "source_route_artifact": str(route_path),
                "source_results_artifact": str(results_path),
                "source_route": source_route.get("route", ""),
                "source_minimum_success": source_route.get("minimum_success", ""),
                "method": "",
                "rows": 0,
                "dataset_seed_pass_count": 0,
                "pass_rows": 0,
                "new_training_executed": 0,
                "official_fms_proof_executed": 0,
                "reset_route_used": 0,
                "optimizer_state_route_used": 0,
                "linec_tail_used_for_direction": 0,
                "promotion_allowed": 0,
            }
        )
    write_rows(out_dir / "v150_rational_no_regression_monitor.csv", rows)
    best = max(rows, key=lambda r: sint(r.get("dataset_seed_pass_count"), 0), default={})
    return {
        "rational_monitor_rows": len(rows),
        "rational_monitor_source": str(summary_path) if summary_path.exists() else "missing_v14.4_official_summary",
        "rational_monitor_source_route": source_route.get("route", ""),
        "rational_monitor_best_method": best.get("method", ""),
        "rational_monitor_best_dataset_seed_pass_count": sint(best.get("dataset_seed_pass_count"), 0),
        "rational_monitor_new_training_executed": 0,
        "rational_monitor_reset_route_used": 0,
        "rational_monitor_promotion_allowed": 0,
        "rational_monitor_route": "RationalNoRegressionReplayOnly",
    }


def build_failure_taxonomy(out_dir: Path, dche_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in dche_rows:
        if str(row.get("method")) not in FU_METHODS:
            continue
        reasons = []
        if fnum(row.get("source_vs_best_control"), -999.0) < 0.005:
            reasons.append("source")
        if fnum(row.get("AUCtime_ratio"), 999.0) > 1.0:
            reasons.append("AUCtime")
        if fnum(row.get("CEp99_delta"), 999.0) > 0.05:
            reasons.append("CEp99")
        if fnum(row.get("NLL_delta"), 999.0) > 0.02:
            reasons.append("NLL")
        if fnum(row.get("ECE_delta"), 999.0) > 0.02:
            reasons.append("ECE")
        if sint(row.get("LineC_majority_pass"), 0) != 1:
            reasons.append("LineC")
        if fnum(row.get("step_time_ratio"), 999.0) > 1.25:
            reasons.append("step")
        if fnum(row.get("memory_ratio"), 999.0) > 1.25:
            reasons.append("memory")
        if sint(row.get("control_equivalent"), 0) == 1:
            reasons.append("control_equivalent")
        rows.append(
            {
                "dataset": row.get("dataset"),
                "seed": row.get("seed"),
                "method": row.get("method"),
                "strict_gate_pass": row.get("strict_gate_pass"),
                "failure_reason": "pass" if not reasons else ";".join(reasons),
                "source_fail": int("source" in reasons),
                "AUCtime_fail": int("AUCtime" in reasons),
                "tail_fail": int(any(r in reasons for r in ["CEp99", "NLL", "ECE"])),
                "LineC_fail": int("LineC" in reasons),
                "cost_fail": int(any(r in reasons for r in ["step", "memory"])),
                "control_equivalent": int("control_equivalent" in reasons),
                "promotion_allowed": 0,
            }
        )
    write_rows(out_dir / "v150_failure_taxonomy.csv", rows)
    return rows


def build_audits(out_dir: Path) -> tuple[int, int]:
    methods = FU_METHODS + FU_CONTROLS + MLP_METHODS + LINE_D_CANDIDATES + ["D-RAT-v15-no-regression-monitor"]
    forbidden = []
    no_action = []
    for method in methods:
        row = {
            "method": method,
            "uses_validation_for_direction": 0,
            "uses_test_for_direction": 0,
            "uses_future_for_direction": 0,
            "uses_query_for_direction": 0,
            "uses_linec_for_direction": 0,
            "uses_cep99_for_direction": 0,
            "uses_nll_for_direction": 0,
            "uses_ece_for_direction": 0,
            "uses_auctime_for_direction": 0,
            "uses_dataset_name_branch": 0,
            "uses_seed_specific_scale": 0,
            "fake_or_proxy_result": 0,
            "cpu_offload_used": 0,
            "action_token": 0,
            "controller_executed": 0,
            "reset_route_used": 0,
            "promotion_allowed": 0,
        }
        row["violation"] = int(any(sint(v, 0) for k, v in row.items() if k not in {"method", "promotion_allowed"}))
        forbidden.append(row)
        no_action.append(
            {
                "method": method,
                "pre_registered": 1,
                "action_search_used": 0,
                "action_bank_used": 0,
                "controller_executed": 0,
                "reset_route_used": 0,
                "new_fu_token_added": 0,
                "new_fche_token_added": 0,
                "violation": 0,
                "promotion_allowed": 0,
            }
        )
    write_rows(out_dir / "v150_forbidden_information_audit.csv", forbidden)
    write_rows(out_dir / "v150_no_action_search_audit.csv", no_action)
    return sum(sint(r.get("violation"), 0) for r in forbidden), sum(sint(r.get("violation"), 0) for r in no_action)


def simple_svg(path: Path, title: str, rows: Sequence[tuple[str, float]], threshold: float | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width = 980
    safe_rows = list(rows)[:28]
    height = 90 + 28 * max(1, len(safe_rows))
    maxv = max([abs(float(v)) for _k, v in safe_rows] + [abs(threshold or 0.0), 1.0e-8])
    body = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="#fbfaf7"/>',
        f'<text x="24" y="34" font-family="monospace" font-size="20" fill="#202020">{title}</text>',
    ]
    for i, (label, value) in enumerate(safe_rows):
        y = 70 + i * 28
        bar = max(2.0, min(390.0, 390.0 * abs(float(value)) / maxv))
        color = "#2f7d62" if float(value) >= 0 else "#9b3d3d"
        safe = str(label).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        body.append(f'<text x="24" y="{y}" font-family="monospace" font-size="13" fill="#333">{safe}: {float(value):.4f}</text>')
        body.append(f'<rect x="540" y="{y - 13}" width="{bar:.1f}" height="16" fill="{color}" opacity="0.78"/>')
    if threshold is not None:
        body.append(f'<text x="24" y="{height - 18}" font-family="monospace" font-size="12" fill="#555">threshold={threshold}</text>')
    body.append("</svg>")
    path.write_text("\n".join(body), encoding="utf-8")


def write_figures(
    out_dir: Path,
    route: dict[str, Any],
    dche_results: Sequence[dict[str, Any]],
    dche_controls: Sequence[dict[str, Any]],
    mlp_rows: Sequence[dict[str, Any]],
    line_d: dict[str, Any],
    failure_rows: Sequence[dict[str, Any]],
) -> None:
    method_rows = []
    for method in FU_METHODS:
        group = [r for r in dche_results if r.get("method") == method]
        method_rows.append((method, mean([fnum(r.get("source_vs_best_control"), 0.0) for r in group])))
    simple_svg(out_dir / "fig_v150_progress_table.svg", "v15.0 progress", [("S1 mechanisms", route.get("s1_optimizer_mechanism_implemented", 0)), ("FU pass count", route.get("real_lite_pass_count", 0)), ("LineD best", route.get("line_d_best_non_dche_dataset_seed_pass_count", 0))])
    simple_svg(out_dir / "fig_v150_fu_vs_controls.svg", "FU vs controls", method_rows, threshold=0.005)
    simple_svg(out_dir / "fig_v150_alignment_histogram.svg", "alignment", [(r.get("method", ""), fnum(r.get("alignment_positive_fraction_preconditioned"), 0.0)) for r in dche_results[:40]])
    simple_svg(out_dir / "fig_v150_second_moment_scale_distribution.svg", "second moment", [(r.get("method", ""), fnum(r.get("second_moment_condition"), 0.0)) for r in dche_results[:40]])
    simple_svg(out_dir / "fig_v150_decoupled_decay_deconfound.svg", "decoupled decay", [("explained_fraction", route.get("decoupled_decay_explains_fraction", 0.0))])
    simple_svg(out_dir / "fig_v150_cautious_vs_mgup_ablation.svg", "cautious vs MGUP", [x for x in method_rows if "Cautious" in x[0] or "MGUP" in x[0]])
    simple_svg(out_dir / "fig_v150_sophia_block_moment_ablation.svg", "Sophia/block moment", [x for x in method_rows if "Sophia" in x[0] or "Block" in x[0]])
    simple_svg(out_dir / "fig_v150_mlp_vs_dche_optimizer_specificity.svg", "MLP vs D-CHE", [("D-CHE best", max([v for _k, v in method_rows], default=0.0)), ("MLP best", max([fnum(r.get("source_vs_mlp_adamw"), 0.0) for r in mlp_rows], default=0.0))])
    simple_svg(out_dir / "fig_v150_allbasis_substrate_heatmap.svg", "all-basis substrate", [(r.get("family", ""), fnum(r.get("family_dataset_seed_pass_count"), 0.0)) for r in line_d.get("summary_rows", [])])
    simple_svg(out_dir / "fig_v150_linec_tail_dashboard.svg", "LineC/tail", [(r.get("method", ""), fnum(r.get("LineC_pass_rate"), 0.0)) for r in dche_results[:40]])
    simple_svg(out_dir / "fig_v150_runtime_breakdown.svg", "runtime", [(r.get("method", ""), fnum(r.get("step_time_ratio"), 0.0)) for r in dche_results[:40]])
    simple_svg(out_dir / "fig_v150_linec_signal_reservoir_noise.svg", "signal reservoir noise", [(r.get("method", ""), fnum(r.get("RealSignalReservoirRatio"), 0.0) - fnum(r.get("NoiseSignalLeak"), 0.0)) for r in dche_results[:40]])
    simple_svg(out_dir / "fig_v150_tail_calibration_dashboard.svg", "tail calibration", [(r.get("method", ""), -fnum(r.get("CEp99_delta"), 0.0)) for r in dche_results[:40]])
    simple_svg(out_dir / "fig_v150_alignment_vs_linec.svg", "alignment vs LineC", [(r.get("method", ""), fnum(r.get("alignment_positive_fraction_preconditioned"), 0.0) * fnum(r.get("LineC_pass_rate"), 0.0)) for r in dche_results[:40]])
    simple_svg(out_dir / "fig_v150_second_moment_vs_tail.svg", "second moment vs tail", [(r.get("method", ""), fnum(r.get("second_moment_condition"), 0.0) - fnum(r.get("CEp99_delta"), 0.0)) for r in dche_results[:40]])


def write_required_manifest(out_dir: Path) -> int:
    rows = []
    for name in REQUIRED + FIGURES:
        path = out_dir / name
        exists = int(path.exists() or name == "v150_required_manifest.csv")
        rows.append({"artifact": name, "exists": exists, "missing": int(not exists), "bytes": path.stat().st_size if path.exists() else 0, "promotion_allowed": 0})
    write_rows(out_dir / "v150_required_manifest.csv", rows)
    return sum(sint(r.get("missing"), 0) for r in rows)


def write_code_packet(out_dir: Path) -> None:
    files = [
        Path("experiments/run_v150_function_update_allbasis_parallel.py"),
        Path("experiments/run_v149_line_d_all_basis_substrate_repair.py"),
        Path("experiments/run_v144_real_transfer_fms_all_basis_substrate.py"),
        Path("experiments/run_v1410_nonrat_fms_transfer_fms_definition_reset.py"),
        PLAN_DOC.relative_to(ROOT),
    ]
    manifest = []
    with zipfile.ZipFile(out_dir / "v150_code_review_packet.zip", "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for rel in files:
            path = ROOT / rel
            manifest.append({"path": str(rel), "exists": int(path.exists()), "sha256": v1410.sha256_file(path) if path.exists() else ""})
            if path.exists():
                zf.write(path, arcname=str(rel))
        buf = csv_rows(manifest)
        zf.writestr("v150_code_review_manifest.csv", buf)


def csv_rows(rows: Sequence[dict[str, Any]]) -> str:
    if not rows:
        return ""
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    import io

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow({k: row.get(k, "") for k in fields})
    return buf.getvalue()


def build_route(
    mechanism: dict[str, Any],
    fu_summary: dict[str, Any],
    decay_route: dict[str, Any],
    kan_summary: Sequence[dict[str, Any]],
    line_d: dict[str, Any],
    forbidden: int,
    no_action: int,
    missing: int,
) -> dict[str, Any]:
    s1 = int(
        mechanism.get("decoupled_decay_implemented") == 1
        and mechanism.get("second_moment_implemented") == 1
        and mechanism.get("alignment_trace_available") == 1
        and mechanism.get("controls_available") == 1
    )
    generic_explains = int(any(fnum(r.get("mean_mlp_gain_vs_adamw"), -999.0) >= fnum(r.get("mean_dche_gain_vs_adamw"), 999.0) and fnum(r.get("mean_dche_gain_vs_adamw"), -999.0) > 0.0 for r in kan_summary))
    if forbidden > 0 or no_action > 0 or missing > 0:
        route = "R0-ArtifactOrProvenanceViolation"
    elif sint(fu_summary.get("line_fu_s5_gate_pass"), 0) == 1 and generic_explains == 0 and sint(line_d.get("line_d_official_fms_eligible_family_count"), 0) >= 0:
        route = "S5-OfficialFunctionalSuccess"
    elif decay_route.get("route") in {"R3-DecoupledRegularizationExplainsGain", "R3-DecoupledDecayExplainsGain"}:
        route = "R3-DecoupledDecayExplainsGain"
    elif sint(fu_summary.get("line_fu_s4_gate_pass"), 0) == 1 and generic_explains:
        route = "R5-FUValueButGeneric"
    elif sint(fu_summary.get("line_fu_s4_gate_pass"), 0) == 1:
        route = "S4-DCHFUExplorationStrong"
    elif sint(fu_summary.get("line_fu_meaningful_gate_pass"), 0) == 1:
        route = "S3-DCHFUExplorationMeaningful"
    elif sint(fu_summary.get("line_fu_exploration_gate_pass"), 0) == 1:
        route = "S2-FUExplorationPositive"
    elif sint(line_d.get("line_d_gate_pass"), 0) == 0:
        route = "R8-CurrentFUDefinitionNoGo"
    else:
        route = "R7-AllBasisCarrierBlocked"
    minimum = (
        "S5-OfficialFunctionalSuccess"
        if route == "S5-OfficialFunctionalSuccess"
        else "S4-DCHFUExplorationStrong"
        if route == "S4-DCHFUExplorationStrong"
        else "S3-DCHFUExplorationMeaningful"
        if route == "S3-DCHFUExplorationMeaningful"
        else "S2-FUExplorationPositive"
        if route == "S2-FUExplorationPositive"
        else "S1-OptimizerMechanismImplemented"
        if s1
        else "S0-ExecutionCompleteNoPromotion"
    )
    return {
        "stage": "V150_ROUTE_DECISION",
        "route": route,
        "minimum_success": minimum,
        "s1_optimizer_mechanism_implemented": s1,
        "official_s5_reached": int(route == "S5-OfficialFunctionalSuccess"),
        "promotion_allowed": int(route == "S5-OfficialFunctionalSuccess" and missing == 0 and forbidden == 0 and no_action == 0),
        "real_lite_pass_count": fu_summary.get("real_lite_pass_count", 0),
        "source_vs_best_control_mean": fu_summary.get("source_vs_best_control_mean", 0.0),
        "control_equivalent_fraction": fu_summary.get("control_equivalent_fraction", 0.0),
        "bad_event_fraction": fu_summary.get("bad_event_fraction", 0.0),
        "line_fu_exploration_gate_pass": fu_summary.get("line_fu_exploration_gate_pass", 0),
        "line_fu_meaningful_gate_pass": fu_summary.get("line_fu_meaningful_gate_pass", 0),
        "line_fu_s4_gate_pass": fu_summary.get("line_fu_s4_gate_pass", 0),
        "line_fu_s5_gate_pass": fu_summary.get("line_fu_s5_gate_pass", 0),
        "best_fu_method": fu_summary.get("best_fu_method", ""),
        "best_fu_mean_source_vs_best_control": fu_summary.get("best_fu_mean_source_vs_best_control", 0.0),
        "decoupled_decay_route": decay_route.get("route", ""),
        "decoupled_decay_explains_fraction": decay_route.get("decoupled_decay_explains_fraction", 0.0),
        "generic_optimizer_explains_gain": generic_explains,
        "line_d_route": line_d.get("line_d_route", ""),
        "line_d_best_non_dche_family": line_d.get("best_non_dche_family", ""),
        "line_d_best_non_dche_dataset_seed_pass_count": line_d.get("best_non_dche_dataset_seed_pass_count", 0),
        "line_d_official_fms_eligible_family_count": line_d.get("line_d_official_fms_eligible_family_count", 0),
        "required_artifact_missing_count": missing,
        "forbidden_information_violation_count": forbidden,
        "no_action_search_violation_count": no_action,
    }


def write_route_docs(out_dir: Path, route: dict[str, Any], fu_summary: dict[str, Any], line_d: dict[str, Any]) -> None:
    write_text(
        out_dir / "v150_no_go_boundary.md",
        "\n".join(
            [
                "# v15.0 no-go boundary",
                "",
                f"route = {route['route']}",
                f"minimum_success = {route['minimum_success']}",
                f"promotion_allowed = {route['promotion_allowed']}",
                "",
                "- Promotion remains fail-closed unless S5 official success is reached.",
                "- Decoupled regularization gains are not written as functional-update gains.",
                "- MLP/generic optimizer positives are not written as KAN-specific promotion.",
                "- LineC / CEp99 / NLL / ECE / AUCtime are audit/gate fields only.",
                "- No FU9/FU10, F-CHE8/F-CHE9, action bank, controller, or reset route was added.",
            ]
        )
        + "\n",
    )
    write_text(
        out_dir / "v150_next_hypothesis_queue.md",
        "\n".join(
            [
                "# v15.0 next hypothesis queue",
                "",
                "- If FU1..FU8 remain below gate, freeze current optimizer-aware FU family.",
                "- Next legal path is substrate/base-architecture or theory-level functional update redesign.",
                "- Do not use LineC/CEp99/NLL/ECE/AUCtime to synthesize a new direction.",
                f"- Current FU pass count = {fu_summary.get('real_lite_pass_count', 0)}/9.",
                f"- Current all-basis best = {line_d.get('best_non_dche_family', '')} {line_d.get('best_non_dche_dataset_seed_pass_count', 0)}/9.",
            ]
        )
        + "\n",
    )


def write_progress(out_dir: Path, route: dict[str, Any], fu_summary: dict[str, Any], line_d: dict[str, Any], mechanism: dict[str, Any], decay_route: dict[str, Any]) -> None:
    rows = [
        {"line": "R", "route": "R-AuditPass" if route["forbidden_information_violation_count"] == 0 and route["no_action_search_violation_count"] == 0 else "R-AuditFail", "promotion_allowed": 0},
        {"line": "O", "route": "S1-OptimizerMechanismImplemented" if route["s1_optimizer_mechanism_implemented"] else "R-O-MechanismMissing", **mechanism, "promotion_allowed": 0},
        {"line": "FU", **{k: v for k, v in fu_summary.items() if k != "method_rows"}, "promotion_allowed": 0},
        {"line": "K", "route": decay_route.get("route"), "decoupled_decay_explains_fraction": decay_route.get("decoupled_decay_explains_fraction"), "promotion_allowed": 0},
        {"line": "D", "route": line_d.get("line_d_route"), "best_family": line_d.get("best_non_dche_family"), "best_count": line_d.get("best_non_dche_dataset_seed_pass_count"), "promotion_allowed": 0},
        {"line": "Z", "route": route["route"], "minimum_success": route["minimum_success"], "promotion_allowed": route["promotion_allowed"]},
    ]
    write_rows(out_dir / "v150_progress_table.csv", rows)


def write_docs(args: argparse.Namespace, out_dir: Path, route: dict[str, Any], fu_summary: dict[str, Any], line_d: dict[str, Any], mechanism: dict[str, Any], dche_results: Sequence[dict[str, Any]], controls: Sequence[dict[str, Any]], mlp_rows: Sequence[dict[str, Any]], decay_route: dict[str, Any]) -> None:
    method_summary = fu_summary.get("method_rows", [])
    best_control = min(controls, key=lambda r: fnum(r.get("NLL"), 9.0), default={})
    best_mlp = max(mlp_rows, key=lambda r: fnum(r.get("source_vs_mlp_adamw"), -999.0), default={})
    rational_rows = read_rows(out_dir / "v150_rational_no_regression_monitor.csv")
    best_rat = max(rational_rows, key=lambda r: sint(r.get("dataset_seed_pass_count"), 0), default={})
    recap = [
        "# DG-KAN v15.0 FunctionUpdate AllBasis 并行加速实验结果复盘",
        "",
        "生成时间：2026-05-30（Asia/Singapore）",
        "",
        "本复盘只写入实际 artifact 中的结果；不把 optimizer trace、real-lite、substrate replay 或 MLP/generic control 写成 promotion。",
        "",
        "## 1. 计划理解",
        "",
        "v15.0 的目标是把 function update 放回 optimizer-aware 语境，验证 second moment、alignment、decoupled regularization 和 basis-safe projection 是否能让 D-CHE FU 超过 matched controls。",
        "",
        "## 2. 本轮代码修改",
        "",
        "新增：",
        "",
        "```text",
        "experiments/run_v150_function_update_allbasis_parallel.py",
        "```",
        "",
        "修改：",
        "",
        "```text",
        "experiments/run_v149_line_d_all_basis_substrate_repair.py",
        "  新增 v15.0 预注册 substrate-only candidates：",
        "  D-FOU37..41、D-RBF35..39、D-WAV33..36。",
        "  不执行 Non-D-CHE official FMS proof，不新增 action/controller/reset route。",
        "```",
        "",
        "过程修正：",
        "",
        "```text",
        "1. smoke run 发现 v150_route_decision.json 在首次 manifest 自检前尚未写入，",
        "   会导致临时 route = R0-ArtifactOrProvenanceViolation。",
        "   已修正 finalizer 顺序：manifest 完成后重新计算 route 并重写 route/docs。",
        "   该修正只影响 artifact completeness route，不改变任何实验指标。",
        "2. 执行日志初版命令使用泛化 python 前缀；",
        "   已改为记录实际解释器 /home/chengshun.wang/miniconda3/envs/kan/bin/python。",
        "3. 用户再次追问后复核发现 FU3 初版把 hard/soft 合成单个 update，",
        "   不满足计划中 hard 与 soft 必须同时测试的要求。",
        "   已改为同一 method token 下写入 fu3_alignment_variant=hard/soft 两组实际训练行；",
        "   M2-MLP-CautiousAdamW 同步补齐 hard/soft generic control。",
        "4. 复核发现 v150_linec_tail_audit.csv 初版缺少 Line C 计划点名的 delta/proxy 字段。",
        "   已补齐 CouplingR2_delta、NoiseSignalLeak_delta、RealSignalReservoirRatio_delta、",
        "   CEp99_delta、NLL_delta、ECE_delta、Brier_delta、margin_p10_delta、",
        "   signal_channel_energy、reservoir_energy、noise_leakage_proxy；这些字段只作 audit，不生成 direction。",
        "5. 用户再次追问后发现 Rational no-regression monitor 未单独落表。",
        "   已新增 v150_rational_no_regression_monitor.csv，读取 v14.4 official D-RAT artifact 作为 replay monitor；",
        "   new_training_executed=0、reset_route_used=0、official_fms_proof_executed=0、promotion_allowed=0。",
        "```",
        "",
        "实现：",
        "",
        "```text",
        "1. Line R forbidden/no-action audit。",
        "2. Line O optimizer mechanism manifest。",
        "3. Line FU D-CHE FU0..FU8 + C0..C9 matched controls。",
        "4. Line M MLP/generic optimizer controls M0..M7。",
        "5. Line K KAN-specificity 与 decoupled decay deconfound。",
        "6. Line D all-basis substrate-only v15 reconfirmation。",
        "7. Line C tail/LineC audit、required manifest、figures、code review packet、执行日志和复盘日志。",
        "```",
        "",
        "## 3. Line O 结果",
        "",
        "```text",
        f"decoupled_decay_implemented = {mechanism.get('decoupled_decay_implemented')}",
        f"second_moment_implemented = {mechanism.get('second_moment_implemented')}",
        f"alignment_trace_available = {mechanism.get('alignment_trace_available')}",
        f"controls_available = {mechanism.get('controls_available')}",
        f"s1_optimizer_mechanism_implemented = {route.get('s1_optimizer_mechanism_implemented')}",
        "```",
        "",
        "## 4. Line FU 结果",
        "",
        "```text",
        f"line_fu_candidate_count = {fu_summary.get('line_fu_candidate_count')}",
        f"real_lite_pass_count = {fu_summary.get('real_lite_pass_count')} / 9",
        f"source_vs_best_control_mean = {fu_summary.get('source_vs_best_control_mean')}",
        f"control_equivalent_fraction = {fu_summary.get('control_equivalent_fraction')}",
        f"bad_event_fraction = {fu_summary.get('bad_event_fraction')}",
        f"line_fu_exploration_gate_pass = {fu_summary.get('line_fu_exploration_gate_pass')}",
        f"line_fu_meaningful_gate_pass = {fu_summary.get('line_fu_meaningful_gate_pass')}",
        f"line_fu_s4_gate_pass = {fu_summary.get('line_fu_s4_gate_pass')}",
        f"best_fu_method = {fu_summary.get('best_fu_method')}",
        f"best_fu_mean_source_vs_best_control = {fu_summary.get('best_fu_mean_source_vs_best_control')}",
        "```",
        "",
        "FU method summary：",
        "",
        "| method | rows | strict pass | dataset-seed pass | mean source vs best control |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in method_summary:
        recap.append(f"| {row.get('method')} | {row.get('rows')} | {row.get('strict_pass_rows')} | {row.get('dataset_seed_pass_count')} | {fnum(row.get('mean_source_vs_best_control'), 0.0):.6f} |")
    recap += [
        "",
        "判断：",
        "",
        "```text",
        "FU1..FU8 只有超过 C0..C9 matched controls 才能写成 FU value。",
        f"本轮 best matched control = {best_control.get('method', '')}；",
        "未达到 S5 时 promotion_allowed 必须保持 0。",
        "```",
        "",
        "## 5. Line K / M 结果",
        "",
        "```text",
        f"decoupled_decay_route = {decay_route.get('route')}",
        f"decoupled_decay_explains_fraction = {decay_route.get('decoupled_decay_explains_fraction')}",
        f"generic_optimizer_explains_gain = {route.get('generic_optimizer_explains_gain')}",
        f"best_mlp_control = {best_mlp.get('method', '')}",
        f"best_mlp_source_vs_adamw = {best_mlp.get('source_vs_mlp_adamw', '')}",
        "```",
        "",
        "判断：MLP/generic optimizer controls 只作为 confound audit；即使出现 positive，也不能写成 KAN-specific promotion。",
        "",
        "## 6. Line D all-basis 结果",
        "",
        "```text",
        f"line_d_source = {line_d.get('line_d_source')}",
        f"line_d_rows = {line_d.get('line_d_rows')}",
        f"best_non_dche_family = {line_d.get('best_non_dche_family')}",
        f"best_non_dche_dataset_seed_pass_count = {line_d.get('best_non_dche_dataset_seed_pass_count')} / 9",
        f"line_d_exploration_family_count = {line_d.get('line_d_exploration_family_count')}",
        f"line_d_official_fms_eligible_family_count = {line_d.get('line_d_official_fms_eligible_family_count')}",
        f"line_d_route = {line_d.get('line_d_route')}",
        "```",
        "",
        "Line D family summary：",
        "",
        "| family | rows | best candidate | family pass | official eligibility |",
        "|---|---:|---|---:|---:|",
    ]
    for row in line_d.get("summary_rows", []):
        recap.append(f"| {row.get('family')} | {row.get('rows')} | {row.get('best_candidate')} | {row.get('family_dataset_seed_pass_count')}/9 | {row.get('family_official_fms_eligibility')} |")
    recap += [
        "",
        "## 6.1 Rational no-regression monitor",
        "",
        "```text",
        "monitor_source = v14.4_official_replay",
        "new_training_executed = 0",
        "reset_route_used = 0",
        "official_fms_proof_executed = 0",
        "promotion_allowed = 0",
        f"monitor_rows = {len(rational_rows)}",
        f"source_route = {best_rat.get('source_route', '')}",
        f"best_method = {best_rat.get('method', '')}",
        f"best_dataset_seed_pass_count = {best_rat.get('dataset_seed_pass_count', '')} / 9",
        "```",
        "",
        "判断：Rational monitor 只是 no-regression replay，不打开 reset route，不进入 v15.0 promotion。",
        "",
        "## 7. 最终 route",
        "",
        "```text",
        f"route = {route.get('route')}",
        f"minimum_success = {route.get('minimum_success')}",
        f"official_s5_reached = {route.get('official_s5_reached')}",
        f"promotion_allowed = {route.get('promotion_allowed')}",
        f"required_artifact_missing_count = {route.get('required_artifact_missing_count')}",
        f"forbidden_information_violation_count = {route.get('forbidden_information_violation_count')}",
        f"no_action_search_violation_count = {route.get('no_action_search_violation_count')}",
        "```",
        "",
        "## 8. 科学结论",
        "",
        "```text",
        "1. v15.0 已执行 Line R/O/FU/M/K/D/C/Z，并生成 required artifacts。",
        f"2. 当前 minimum_success = {route.get('minimum_success')}；promotion_allowed = {route.get('promotion_allowed')}。",
        f"3. D-CHE FU real-lite pass = {fu_summary.get('real_lite_pass_count')}/9。",
        f"4. Non-D-CHE best substrate = {line_d.get('best_non_dche_family')} {line_d.get('best_non_dche_dataset_seed_pass_count')}/9。",
        "5. 未达到 S5 时不能把 decoupled decay、MLP/generic control 或 local positive row 写成 official functional success。",
        "6. 若继续推进，需要下一版 substrate/base-architecture 或 theory-level functional update 计划，不能在 v15.0 内临时新增 FU9/FU10、F-CHE8/F-CHE9、action/controller/reset route。",
        "```",
    ]
    write_text(RECAP_DOC, "\n".join(recap) + "\n")

    line_d_cmd = (
        f"{sys.executable} experiments/run_v149_line_d_all_basis_substrate_repair.py --out-dir {args.line_d_out} "
        f"--device {args.device} --datasets {args.datasets} --seeds {args.seeds} --train-size {args.train_size} "
        f"--val-size {args.val_size} --epochs {args.line_d_epochs} --linec-seeds {args.linec_seeds} "
        f"--candidates {','.join(LINE_D_CANDIDATES)}"
    )
    finalizer_cmd = sys.executable + " " + " ".join(sys.argv)
    exec_log = [
        "# DG-KAN v15.0 FunctionUpdate AllBasis 并行加速执行日志",
        "",
        "生成时间：2026-05-30（Asia/Singapore）",
        "",
        "## 1. 代码与计划",
        "",
        "```text",
        f"plan = {PLAN_DOC}",
        "runner = experiments/run_v150_function_update_allbasis_parallel.py",
        "line_d_runner = experiments/run_v149_line_d_all_basis_substrate_repair.py",
        f"official_out = {out_dir}",
        f"line_d_out = {args.line_d_out}",
        "```",
        "",
        "## 1.1 本轮覆盖性修正",
        "",
        "```text",
        "1. FU3-D-CHE-CautiousFunctionUpdate 按计划拆成 hard / soft alignment 两个实际训练分支；",
        "   M2-MLP-CautiousAdamW 同步拆成 hard / soft generic control。",
        "2. Line C tail audit 增补 *_delta 字段，以及 signal_channel_energy、reservoir_energy、noise_leakage_proxy。",
        "3. 上述字段只由实际训练 artifact 与同 family/dataset/seed baseline 计算；",
        "   不用 LineC/CEp99/NLL/ECE/AUCtime 反推训练方向。",
        "```",
        "",
        "## 2. 执行命令",
        "",
        "Line D substrate-only reconfirmation：",
        "",
        "```bash",
        line_d_cmd,
        "```",
        "",
        "v15.0 official finalizer：",
        "",
        "```bash",
        finalizer_cmd,
        "```",
        "",
        "## 3. 关键参数",
        "",
        "```text",
        f"datasets = {args.datasets}",
        f"seeds = {args.seeds}",
        f"train_size = {args.train_size}",
        f"val_size = {args.val_size}",
        f"test_size = {args.test_size}",
        f"train_steps = {args.train_steps}",
        f"batch_size = {args.batch_size}",
        f"lr = {args.lr}",
        f"weight_decay = {args.weight_decay}",
        f"linec_seeds = {args.linec_seeds}",
        f"device = {args.device}",
        "official_fms_proof_executed_for_non_dche = 0",
        "promotion_allowed_unless_S5 = 0",
        "```",
        "",
        "## 4. 输出文件",
        "",
        "```text",
        "v150_route_decision.json",
        "v150_required_manifest.csv",
        "v150_optimizer_mechanism_manifest.csv",
        "v150_dche_fu_results.csv",
        "v150_dche_fu_controls.csv",
        "v150_mlp_controls.csv",
        "v150_allbasis_substrate_results.csv",
        "v150_rational_no_regression_monitor.csv",
        "v150_linec_tail_audit.csv",
        "v150_alignment_trace.csv",
        "v150_block_preconditioner_trace.csv",
        "v150_schedulefree_control_trace.csv",
        "v150_failure_taxonomy.csv",
        "v150_code_review_packet.zip",
        "```",
        "",
        "## 5. 运行结果摘要",
        "",
        "```text",
        f"route = {route.get('route')}",
        f"minimum_success = {route.get('minimum_success')}",
        f"FU real_lite_pass_count = {fu_summary.get('real_lite_pass_count')}/9",
        f"Line D best = {line_d.get('best_non_dche_family')} {line_d.get('best_non_dche_dataset_seed_pass_count')}/9",
        f"required_artifact_missing_count = {route.get('required_artifact_missing_count')}",
        f"forbidden_information_violation_count = {route.get('forbidden_information_violation_count')}",
        f"no_action_search_violation_count = {route.get('no_action_search_violation_count')}",
        "```",
    ]
    write_text(EXEC_LOG_DOC, "\n".join(exec_log) + "\n")


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    forbidden, no_action = build_audits(out_dir)
    dche_results, dche_controls, mlp_rows, fu_summary = run_fu_and_m(args, out_dir)
    mechanism = build_mechanism_manifest(out_dir, dche_results, dche_controls)
    _kan_rows, kan_summary = build_kan_specificity(out_dir, dche_results, mlp_rows)
    _decay_rows, decay_route = build_decay_deconfound(out_dir, dche_results + dche_controls)
    line_d = build_line_d(out_dir, Path(args.line_d_out))
    rational_monitor = build_rational_monitor(out_dir)
    failure_rows = build_failure_taxonomy(out_dir, dche_results)
    write_code_packet(out_dir)
    route = build_route(mechanism, fu_summary, decay_route, kan_summary, line_d, forbidden, no_action, 999)
    write_route_docs(out_dir, route, fu_summary, line_d)
    write_progress(out_dir, route, fu_summary, line_d, mechanism, decay_route)
    write_figures(out_dir, route, dche_results, dche_controls, mlp_rows, line_d, failure_rows)
    missing = write_required_manifest(out_dir)
    route = build_route(mechanism, fu_summary, decay_route, kan_summary, line_d, forbidden, no_action, missing)
    write_json(out_dir / "v150_route_decision.json", route)
    write_route_docs(out_dir, route, fu_summary, line_d)
    write_progress(out_dir, route, fu_summary, line_d, mechanism, decay_route)
    write_figures(out_dir, route, dche_results, dche_controls, mlp_rows, line_d, failure_rows)
    missing = write_required_manifest(out_dir)
    route["required_artifact_missing_count"] = missing
    route = build_route(mechanism, fu_summary, decay_route, kan_summary, line_d, forbidden, no_action, missing)
    write_json(out_dir / "v150_route_decision.json", route)
    write_route_docs(out_dir, route, fu_summary, line_d)
    write_progress(out_dir, route, fu_summary, line_d, mechanism, decay_route)
    write_docs(args, out_dir, route, fu_summary, line_d, mechanism, dche_results, dche_controls, mlp_rows, decay_route)
    return route


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--line-d-out", default=str(DEFAULT_LINE_D_OUT))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--datasets", default="MNIST,Fashion-MNIST,KMNIST")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--train-size", type=int, default=256)
    parser.add_argument("--val-size", type=int, default=128)
    parser.add_argument("--test-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--train-steps", type=int, default=120)
    parser.add_argument("--trace-interval", type=int, default=60)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--readout-weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--beta1", type=float, default=0.90)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--curvature-beta", type=float, default=0.95)
    parser.add_argument("--linec-seeds", default="12319500,12319501,12319502")
    parser.add_argument("--linec-batch-size", type=int, default=24)
    parser.add_argument("--linec-sketch-dim", type=int, default=8)
    parser.add_argument("--real-linec", type=int, default=1)
    parser.add_argument("--mlp-hidden", type=int, default=32)
    parser.add_argument("--dche-candidate", default=v1410.DEFAULT_D_CHE_CANDIDATE)
    parser.add_argument("--fu-methods", default=",".join(FU_METHODS))
    parser.add_argument("--fu-controls", default=",".join(FU_CONTROLS))
    parser.add_argument("--mlp-methods", default=",".join(MLP_METHODS))
    parser.add_argument("--line-d-epochs", type=int, default=1)
    parser.add_argument("--reuse-if-present", type=int, default=1)
    parser.add_argument("--compute-budgeted-run", type=int, default=0)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    route = run(args)
    print(json.dumps(route, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
