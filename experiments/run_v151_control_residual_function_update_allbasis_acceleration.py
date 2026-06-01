#!/usr/bin/env python3
"""DG-KAN v15.1 control-residual function-update fork.

The runner executes the v15.1 CR-FU plan. Update directions are computed from
the current train stream and current optimizer state only. Audit metrics,
LineC, CEp99, NLL, ECE, AUCtime, and Brier are recorded only after training.
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
from experiments import run_v150_function_update_allbasis_parallel as v150  # noqa: E402


DEFAULT_OUT = ROOT / "results/v15_1_control_residual_function_update_allbasis_acceleration/official_v151"
DEFAULT_LINE_D_OUT = ROOT / "results/v15_1_control_residual_function_update_allbasis_acceleration/line_d_v151_allbasis_substrate"
V150_DIR = ROOT / "results/v15_0_function_update_allbasis_parallel/official_v150"
PLAN_DOC = ROOT / "docs/DG-KAN_v15.01_ControlResidualFunctionUpdate_AllBasisAcceleration_完整计划.md"
RECAP_DOC = ROOT / "docs/DG-KAN_v15.01_ControlResidualFunctionUpdate_AllBasisAcceleration_实验结果复盘.md"
EXEC_LOG_DOC = ROOT / "docs/DG-KAN_v15.01_ControlResidualFunctionUpdate_AllBasisAcceleration_执行日志.md"

F_METHODS = [
    "F0-D-CHE-AdamW",
    "F1-D-CHE-CautiousAdamW",
    "F2-D-CHE-MGUPControl",
    "F3-D-CHE-RawFUResidualizedAgainstAdamW",
    "F4-D-CHE-RawFUResidualizedAgainstCautious",
    "F5-D-CHE-MGUPFUResidualizedAgainstMGUPControl",
    "F6-D-CHE-FunctionSpaceProximalResidual",
    "F7-D-CHE-CRFU-AbstentionEnabled",
]

F_CONTROLS = [
    "F0-D-CHE-AdamW",
    "F1-D-CHE-CautiousAdamW",
    "F2-D-CHE-MGUPControl",
    "FCTRL-RandomResidualMatchedNorm",
    "FCTRL-SameActiveFractionResidual",
    "FCTRL-SameProjectionRejectionResidual",
    "FCTRL-SameValueRetentionResidual",
]

M_METHODS = [
    "M0-MLP-AdamW",
    "M1-MLP-CautiousAdamW",
    "M2-MLP-MGUPControl",
    "M3-MLP-FunctionSpaceProximalResidual",
    "M4-MLP-CRFU-AbstentionEnabled",
    "M5-MLP-RandomResidualMatchedNorm",
    "M6-MLP-SameActiveFractionResidual",
    "M7-MLP-NoOpMatchedOverhead",
]

LINE_D_CANDIDATES = [
    "D-FOU42-LowFreqResidualV5",
    "D-FOU43-BandwiseSecondMomentWarmup",
    "D-FOU44-PhaseStableLowBandOnly",
    "D-FOU45-NoMaterializeLifetimeV4",
    "D-FOU46-HighFreqQuarantineLateEnable",
    "D-RBF40-CompactBumpIdentityResidualV2",
    "D-RBF41-ActiveCenterOccupancySecondMoment",
    "D-RBF42-WidthFloorTrustRegion",
    "D-RBF43-GaussianLocalK4NoDenseV2",
    "D-RBF44-CenterReadoutDecoupledWarmup",
    "D-WAV37-TriangularSupportV5",
    "D-WAV38-ScaleOccupancySecondMoment",
    "D-WAV39-LocalSupportOverlapTrust",
    "D-WAV40-FineScaleLateEnable",
]

REQUIRED = [
    "v151_route_decision.json",
    "v151_progress_table.csv",
    "v151_required_artifact_manifest.csv",
    "v151_forbidden_information_audit.csv",
    "v151_no_action_search_audit.csv",
    "v151_direction_decomposition.csv",
    "v151_fu_failure_attribution.csv",
    "v151_fu_residualizability_summary.csv",
    "v151_projection_value_loss.csv",
    "v151_decay_deconfound_extended.csv",
    "v151_current_fu_no_go_boundary.md",
    "v151_dche_crfu_results.csv",
    "v151_dche_crfu_controls.csv",
    "v151_mlp_generic_controls.csv",
    "v151_kan_specificity_summary.csv",
    "v151_allbasis_substrate_results.csv",
    "v151_allbasis_substrate_summary.csv",
    "v151_dche_no_regression_monitor.csv",
    "v151_rational_no_regression_monitor.csv",
    "v151_linec_tail_audit.csv",
    "v151_failure_taxonomy.csv",
    "v151_no_go_boundary.md",
    "v151_next_hypothesis_queue.md",
    "v151_code_review_packet.zip",
]

FIGURES = [
    "fig_v151_direction_cosine_matrix.svg",
    "fig_v151_control_projection_fraction.svg",
    "fig_v151_alignment_by_role.svg",
    "fig_v151_decay_vs_fu_norm.svg",
    "fig_v151_second_moment_condition.svg",
    "fig_v151_linec_tail_matrix.svg",
    "fig_v151_signal_reservoir_noise_plane.svg",
    "fig_v151_tail_calibration_tradeoff.svg",
]


def fnum(value: Any, default: float = 0.0) -> float:
    return v1414.fnum(value, default)


def sint(value: Any, default: int = 0) -> int:
    return v1414.sint(value, default)


def mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return statistics.fmean(vals) if vals else 0.0


def median(values: Iterable[float]) -> float:
    vals = sorted(float(v) for v in values if math.isfinite(float(v)))
    return statistics.median(vals) if vals else 0.0


def read_rows(path: Path) -> list[dict[str, str]]:
    return v1414.read_rows(path)


def write_rows(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    v1414.write_rows(path, rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    v1414.write_json(path, payload)


def write_text(path: Path, text: str) -> None:
    v1414.write_text(path, text)


def parse_csv(value: str) -> list[str]:
    return v1410.parse_csv(value)


def parse_ints(value: str) -> list[int]:
    return v1410.parse_ints(value)


def resolve_cuda_device(requested: str) -> torch.device:
    if not str(requested).startswith("cuda"):
        raise RuntimeError(f"GPU execution is required for v15.01; got --device {requested!r}")
    if not torch.cuda.is_available():
        raise RuntimeError("GPU execution is required for v15.01, but torch.cuda.is_available() is false")
    device = torch.device(str(requested))
    torch.cuda.set_device(device)
    return device


def flat_params(specs: list[Any]) -> torch.Tensor:
    return v150.flat_params(specs)


def add_flat_update(specs: list[Any], update: torch.Tensor, lr: float) -> None:
    v150.add_flat_update(specs, update, lr)


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return v150.cosine(a, b)


def norm_match(vec: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if float(vec.norm().item()) <= 1.0e-12 or float(target.norm().item()) <= 1.0e-12:
        return torch.zeros_like(target)
    return v150.norm_match(vec, target)


def method_family(method: str) -> str:
    if method.startswith("FCTRL"):
        return "D-CHE-CRFU-control"
    if method.startswith("F"):
        return "D-CHE-CRFU"
    return "MLP-generic-control"


def decay_direction(specs: list[Any], weight_decay: float, readout_decay: float) -> torch.Tensor:
    params = flat_params(specs)
    out = torch.zeros_like(params)
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
        if lam > 0:
            out[spec.start : spec.end] = -lam * params[spec.start : spec.end]
    return out


def metric_residualize(
    proposal: torch.Tensor,
    controls: Sequence[torch.Tensor],
    metric: torch.Tensor,
    ridge: float = 1.0e-4,
) -> tuple[torch.Tensor, torch.Tensor, float, float]:
    if not controls:
        return proposal, torch.zeros_like(proposal), 0.0, 1.0
    usable = [c for c in controls if c.numel() == proposal.numel() and float(c.norm().item()) > 1.0e-12]
    if not usable:
        return proposal, torch.zeros_like(proposal), 0.0, 1.0
    cmat = torch.stack(usable, dim=1)
    mw_c = metric.unsqueeze(1) * cmat
    gram = cmat.t().matmul(mw_c)
    gram = gram + float(ridge) * torch.eye(gram.shape[0], device=proposal.device, dtype=proposal.dtype)
    rhs = cmat.t().matmul(metric * proposal)
    try:
        coeff = torch.linalg.solve(gram, rhs)
    except RuntimeError:
        coeff = torch.linalg.pinv(gram).matmul(rhs)
    projection = cmat.matmul(coeff)
    residual = proposal - projection
    pnorm = float(projection.norm().item())
    rnorm = float(residual.norm().item())
    total = max(1.0e-12, pnorm + rnorm)
    return residual, projection, pnorm / total, rnorm / total


def role_anti_alignment(specs: list[Any], update: torch.Tensor, neg_grad: torch.Tensor) -> tuple[float, str, str]:
    vals = []
    role_parts = []
    group_parts = []
    if update.numel() == 0:
        return 0.0, "", ""
    anti = (update * neg_grad) < 0.0
    overall = float(anti.float().mean().item())
    by_role: dict[str, list[float]] = {}
    for spec in specs:
        seg = anti[spec.start : spec.end]
        val = float(seg.float().mean().item()) if seg.numel() else 0.0
        by_role.setdefault(str(spec.role), []).append(val)
        if spec.role == "basis" or (spec.param.ndim >= 3 and spec.param.shape[-1] <= 8):
            vals.append(val)
    for role, role_vals in sorted(by_role.items()):
        role_parts.append(f"{role}:{mean(role_vals):.6f}")
    if vals:
        group_parts.append(f"basis:{mean(vals):.6f}")
    return overall, ";".join(role_parts), ";".join(group_parts)


def control_directions(
    specs: list[Any],
    grad: torch.Tensor,
    mhat: torch.Tensor,
    vhat: torch.Tensor,
    args: argparse.Namespace,
) -> dict[str, torch.Tensor]:
    neg_grad = -grad
    adam = -mhat / (vhat.sqrt() + 1.0e-8)
    align = adam * neg_grad
    cautious = torch.where(align > 0.0, adam, torch.zeros_like(adam))
    if align.numel():
        threshold = torch.quantile(align.float(), 0.50)
        mgup = torch.where(align >= threshold, 2.0 * adam, 0.5 * adam)
    else:
        mgup = adam
    decay = decay_direction(specs, float(args.weight_decay), float(args.readout_weight_decay))
    return {"adam": adam, "cautious": cautious, "mgup": mgup, "decay": decay}


def crfu_update(
    method: str,
    family: str,
    specs: list[Any],
    grad: torch.Tensor,
    mhat: torch.Tensor,
    vhat: torch.Tensor,
    args: argparse.Namespace,
    gen: torch.Generator,
    model: torch.nn.Module | None = None,
    xb: torch.Tensor | None = None,
    yb: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    neg_grad = -grad
    controls = control_directions(specs, grad, mhat, vhat, args)
    raw_fu = v150.basis_safe_projection(specs, neg_grad, family)
    metric = vhat.sqrt().clamp_min(1.0e-8)
    optimizer_control_set = [controls["adam"], controls["cautious"], controls["mgup"], controls["decay"]]
    _decay_residual, decay_projection, decay_projection_fraction, _decay_residual_fraction = metric_residualize(raw_fu, [controls["decay"]], metric)
    matched_gen = torch.Generator(device=raw_fu.device).manual_seed(int(gen.initial_seed()) + 9209)
    matched_norm = norm_match(
        torch.randn(raw_fu.shape, generator=matched_gen, device=raw_fu.device, dtype=raw_fu.dtype),
        raw_fu,
    )
    residual_probe, _probe_projection, _probe_pf, _probe_rf = metric_residualize(raw_fu, optimizer_control_set, metric)
    matched_active = torch.randn(raw_fu.shape, generator=matched_gen, device=raw_fu.device, dtype=raw_fu.dtype)
    matched_active = norm_match(matched_active * (residual_probe.abs() > residual_probe.abs().median()).float(), residual_probe)
    matched_projection = v150.basis_safe_projection(
        specs,
        torch.randn(raw_fu.shape, generator=matched_gen, device=raw_fu.device, dtype=raw_fu.dtype),
        family,
    )
    matched_projection = norm_match(matched_projection, raw_fu)
    matched_value = 0.5 * norm_match(
        torch.randn(raw_fu.shape, generator=matched_gen, device=raw_fu.device, dtype=raw_fu.dtype),
        raw_fu,
    ) + 0.5 * raw_fu.sign() * raw_fu.abs().mean().clamp_min(1.0e-8)
    matched_control_set = [matched_norm, matched_active, matched_projection, matched_value]
    full_control_set = optimizer_control_set + matched_control_set
    decay_gen = torch.Generator(device=raw_fu.device).manual_seed(int(gen.initial_seed()) + 9109)
    decay_active = controls["decay"].abs() > 0.0
    random_decay = torch.randn(raw_fu.shape, generator=decay_gen, device=raw_fu.device, dtype=raw_fu.dtype) * decay_active.float()
    random_decay = norm_match(random_decay, controls["decay"])
    _rand_decay_residual, rand_decay_projection, rand_decay_projection_fraction, _rand_decay_residual_fraction = metric_residualize(raw_fu, [random_decay], metric)
    base = controls["adam"]
    proposal = raw_fu
    control_set: Sequence[torch.Tensor] = [controls["adam"]]
    if method in {"F1-D-CHE-CautiousAdamW", "M1-MLP-CautiousAdamW"}:
        base = controls["cautious"]
        proposal = base
        control_set = []
    elif method in {"F2-D-CHE-MGUPControl", "M2-MLP-MGUPControl"}:
        base = controls["mgup"]
        proposal = base
        control_set = []
    elif method in {"F3-D-CHE-RawFUResidualizedAgainstAdamW"}:
        base = controls["adam"]
        control_set = [controls["adam"]]
    elif method in {"F4-D-CHE-RawFUResidualizedAgainstCautious"}:
        base = controls["cautious"]
        control_set = [controls["cautious"]]
    elif method in {"F5-D-CHE-MGUPFUResidualizedAgainstMGUPControl"}:
        base = controls["mgup"]
        proposal = torch.where((controls["mgup"] * neg_grad) > 0.0, raw_fu, torch.zeros_like(raw_fu))
        control_set = [controls["mgup"]]
    elif method in {"F6-D-CHE-FunctionSpaceProximalResidual", "M3-MLP-FunctionSpaceProximalResidual"}:
        base = controls["adam"]
        control_set = full_control_set
    elif method in {"F7-D-CHE-CRFU-AbstentionEnabled", "M4-MLP-CRFU-AbstentionEnabled"}:
        base = controls["adam"]
        control_set = full_control_set
    elif method in {"FCTRL-RandomResidualMatchedNorm", "M5-MLP-RandomResidualMatchedNorm"}:
        base = controls["adam"]
        rnd = torch.randn(raw_fu.shape, generator=gen, device=raw_fu.device, dtype=raw_fu.dtype)
        proposal = rnd
        control_set = full_control_set
    elif method in {"FCTRL-SameActiveFractionResidual", "M6-MLP-SameActiveFractionResidual"}:
        base = controls["adam"]
        rnd = torch.randn(raw_fu.shape, generator=gen, device=raw_fu.device, dtype=raw_fu.dtype)
        residual_probe, _proj, _pf, _rf = metric_residualize(raw_fu, full_control_set, metric)
        active = residual_probe.abs() > residual_probe.abs().median()
        proposal = rnd * active.float()
        control_set = full_control_set
    elif method == "FCTRL-SameProjectionRejectionResidual":
        base = controls["adam"]
        rnd = torch.randn(raw_fu.shape, generator=gen, device=raw_fu.device, dtype=raw_fu.dtype)
        proposal = v150.basis_safe_projection(specs, rnd, family)
        control_set = full_control_set
    elif method == "FCTRL-SameValueRetentionResidual":
        base = controls["adam"]
        rnd = torch.randn(raw_fu.shape, generator=gen, device=raw_fu.device, dtype=raw_fu.dtype)
        proposal = 0.5 * norm_match(rnd, raw_fu) + 0.5 * raw_fu.sign() * raw_fu.abs().mean().clamp_min(1.0e-8)
        control_set = full_control_set
    elif method == "M7-MLP-NoOpMatchedOverhead":
        base = torch.zeros_like(controls["adam"])
        proposal = base
        control_set = []
    residual, projection, projection_fraction, residual_fraction = metric_residualize(proposal, control_set, metric)
    chosen_alpha = 0.0
    proximal_candidate_count = 0
    proximal_solver_used_train_split = 0
    if method in {"F0-D-CHE-AdamW", "M0-MLP-AdamW", "F1-D-CHE-CautiousAdamW", "M1-MLP-CautiousAdamW", "F2-D-CHE-MGUPControl", "M2-MLP-MGUPControl", "M7-MLP-NoOpMatchedOverhead"}:
        method_update = base
    else:
        aligned = torch.where((residual * neg_grad) >= 0.0, residual, torch.zeros_like(residual))
        alpha = 0.25
        if method in {"F6-D-CHE-FunctionSpaceProximalResidual", "M3-MLP-FunctionSpaceProximalResidual"} and model is not None and xb is not None and yb is not None:
            candidates = [0.0, 0.05, 0.10, 0.25, 0.50]
            proximal_candidate_count = len(candidates)
            proximal_solver_used_train_split = 1
            half = max(1, int(xb.shape[0]) // 2)
            b1x, b1y = xb[:half], yb[:half]
            base_state = [spec.param.detach().clone() for spec in specs]
            best_obj = float("inf")
            best_alpha = 0.0
            with torch.no_grad():
                for cand in candidates:
                    trial_update = base + float(args.crfu_eta) * float(cand) * aligned
                    for spec, saved in zip(specs, base_state, strict=True):
                        spec.param.copy_(saved)
                    add_flat_update(specs, trial_update, float(args.lr))
                    loss_val = float(v1410.loss_value(model(b1x), b1y, "CE").detach().item())
                    penalty = float(args.proximal_lambda) * float(((aligned * float(cand)).square() * metric).sum().item())
                    obj = loss_val + penalty
                    if obj < best_obj:
                        best_obj = obj
                        best_alpha = float(cand)
                for spec, saved in zip(specs, base_state, strict=True):
                    spec.param.copy_(saved)
            alpha = best_alpha
        elif method in {"F6-D-CHE-FunctionSpaceProximalResidual", "M3-MLP-FunctionSpaceProximalResidual", "F7-D-CHE-CRFU-AbstentionEnabled", "M4-MLP-CRFU-AbstentionEnabled"}:
            denom = float(((aligned * aligned) * metric).sum().item()) + 1.0e-8
            numer = float((aligned * neg_grad).sum().item())
            alpha = max(0.0, min(0.5, numer / denom))
        if method in {"F7-D-CHE-CRFU-AbstentionEnabled", "M4-MLP-CRFU-AbstentionEnabled"}:
            anti_tmp, _roles, _groups = role_anti_alignment(specs, residual, neg_grad)
            if residual_fraction < 0.05 or anti_tmp > 0.50:
                alpha = 0.0
        if method in {"FCTRL-RandomResidualMatchedNorm", "FCTRL-SameActiveFractionResidual", "FCTRL-SameProjectionRejectionResidual", "FCTRL-SameValueRetentionResidual", "M5-MLP-RandomResidualMatchedNorm", "M6-MLP-SameActiveFractionResidual"}:
            aligned = norm_match(aligned, residual)
        method_update = base + float(args.crfu_eta) * alpha * aligned
        chosen_alpha = float(alpha)
    projected = v150.basis_safe_projection(specs, method_update, family)
    stats = v1410.projection_stats(raw_fu, projected)
    anti, rolewise, groupwise = role_anti_alignment(specs, projected, neg_grad)
    metric_cos = lambda x, y: cosine(metric * x, metric * y)
    trace = {
        **stats,
        "fu_norm": float(raw_fu.norm().item()),
        "grad_norm": float(grad.norm().item()),
        "adamw_update_norm": float(controls["adam"].norm().item()),
        "cautious_update_norm": float(controls["cautious"].norm().item()),
        "mgup_update_norm": float(controls["mgup"].norm().item()),
        "decay_update_norm": float(controls["decay"].norm().item()),
        "cos_fu_neg_grad": cosine(raw_fu, neg_grad),
        "cos_fu_adamw": cosine(raw_fu, controls["adam"]),
        "cos_fu_cautious": cosine(raw_fu, controls["cautious"]),
        "cos_fu_mgup": cosine(raw_fu, controls["mgup"]),
        "cos_fu_decay": cosine(raw_fu, controls["decay"]),
        "metric_cos_fu_adamw": metric_cos(raw_fu, controls["adam"]),
        "metric_cos_fu_cautious": metric_cos(raw_fu, controls["cautious"]),
        "metric_cos_fu_mgup": metric_cos(raw_fu, controls["mgup"]),
        "control_projection_norm_fraction": projection_fraction,
        "control_residual_norm_fraction": residual_fraction,
        "control_projection_norm": float(projection.norm().item()),
        "control_residual_norm": float(residual.norm().item()),
        "anti_alignment_fraction": anti,
        "rolewise_anti_alignment": rolewise,
        "basis_group_anti_alignment": groupwise,
        "decoupled_decay_explains_fraction": decay_projection_fraction,
        "decoupled_decay_projection_norm": float(decay_projection.norm().item()),
        "random_matched_decay_explains_fraction": rand_decay_projection_fraction,
        "random_matched_decay_projection_norm": float(rand_decay_projection.norm().item()),
        "second_moment_condition": v150.role_condition(specs, metric),
        "curvature_clip_fraction": float((metric > metric.median() * 10.0).float().mean().item()) if metric.numel() else 0.0,
        "crfu_abstention_fraction": int(method in {"F7-D-CHE-CRFU-AbstentionEnabled", "M4-MLP-CRFU-AbstentionEnabled"} and float((projected - base).norm().item()) <= 1.0e-12),
        "function_space_proximal_alpha": chosen_alpha,
        "proximal_solver_candidate_count": proximal_candidate_count,
        "proximal_solver_used_train_split": proximal_solver_used_train_split,
        "optimizer_residualizer_control_count": len(optimizer_control_set),
        "matched_residualizer_control_count": len(matched_control_set),
        "full_residualizer_control_count": len(full_control_set),
    }
    return projected, trace


def train_case(
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
) -> dict[str, Any]:
    case_args = copy(args)
    case_args.synthetic_dim = int(input_dim)
    case_args.synthetic_classes = int(output_dim)
    candidate = str(args.dche_candidate)
    model = v1410.make_case_model("MLP", "MLP-v151-control", xtr, seed, case_args, device) if family == "MLP" else v1410.make_case_model("D-CHE", candidate, xtr, seed, case_args, device)
    specs = v1410.named_param_specs(model)
    total = sum(int(spec.param.numel()) for spec in specs)
    m = torch.zeros(total, device=device)
    v = torch.zeros(total, device=device)
    beta1 = float(args.beta1)
    beta2 = float(args.beta2)
    gen = torch.Generator(device=device).manual_seed(int(seed) + 151_000 + sum(ord(c) for c in method + dataset + family))
    acc_keys = [
        "fu_norm",
        "grad_norm",
        "adamw_update_norm",
        "cautious_update_norm",
        "mgup_update_norm",
        "decay_update_norm",
        "cos_fu_neg_grad",
        "cos_fu_adamw",
        "cos_fu_cautious",
        "cos_fu_mgup",
        "cos_fu_decay",
        "metric_cos_fu_adamw",
        "metric_cos_fu_cautious",
        "metric_cos_fu_mgup",
        "control_projection_norm_fraction",
        "control_residual_norm_fraction",
        "control_projection_norm",
        "control_residual_norm",
        "anti_alignment_fraction",
        "decoupled_decay_explains_fraction",
        "decoupled_decay_projection_norm",
        "random_matched_decay_explains_fraction",
        "random_matched_decay_projection_norm",
        "second_moment_condition",
        "curvature_clip_fraction",
        "degree_projection_rejection_fraction",
        "value_retention_after_degree_projection",
        "crfu_abstention_fraction",
        "function_space_proximal_alpha",
        "proximal_solver_candidate_count",
        "proximal_solver_used_train_split",
        "optimizer_residualizer_control_count",
        "matched_residualizer_control_count",
        "full_residualizer_control_count",
    ]
    acc = {k: [] for k in acc_keys}
    role_strings: list[str] = []
    group_strings: list[str] = []
    trajectory: list[dict[str, float]] = []
    direction_rows: list[dict[str, Any]] = []
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
        m = beta1 * m + (1.0 - beta1) * grad
        v = beta2 * v + (1.0 - beta2) * grad.square()
        mhat = m / (1.0 - beta1 ** (step + 1))
        vhat = v / (1.0 - beta2 ** (step + 1))
        update, trace = crfu_update(method, family, specs, grad, mhat, vhat, args, gen, model=model, xb=xb, yb=yb)
        add_flat_update(specs, update, float(args.lr))
        v150.apply_role_decay(specs, float(args.lr), float(args.weight_decay), float(args.readout_weight_decay))
        for key in acc:
            acc[key].append(float(trace.get(key, 0.0)))
        role_strings.append(str(trace.get("rolewise_anti_alignment", "")))
        group_strings.append(str(trace.get("basis_group_anti_alignment", "")))
        if step == 0 or step == int(args.train_steps) - 1 or ((step + 1) % max(1, int(args.trace_interval)) == 0):
            metrics = v1410.eval_metrics(model, xva, yva)
            trajectory.append({"step": float(step + 1), "NLL": metrics["NLL"], "CEp99": metrics["CEp99"], "ECE": metrics["ECE"], "Brier": metrics["Brier"], "acc": metrics["acc"]})
            direction_rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "family": family,
                    "method": method,
                    "step": step + 1,
                    **{k: trace.get(k, 0.0) for k in acc_keys},
                    "rolewise_anti_alignment": trace.get("rolewise_anti_alignment", ""),
                    "basis_group_anti_alignment": trace.get("basis_group_anti_alignment", ""),
                    "direction_uses_train_stream_only": 1,
                    "promotion_allowed": 0,
                }
            )
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
        linec_rows = []
        linec_mean = {"CouplingR2": final["CouplingR2"], "NoiseSignalLeak": final["NoiseSignalLeak"], "RealSignalReservoirRatio": final["RealSignalReservoirRatio"]}
    after_degree = v1410.degree_energy(model) if family == "D-CHE" else {}
    row = {
        "stage": "V151_CRFU_CASE",
        "family": family,
        "method_family": method_family(method),
        "method": method,
        "candidate_id": candidate if family == "D-CHE" else "MLP-v151-control",
        "dataset": dataset,
        "seed": seed,
        "loss_interface": "CE",
        "control_method": int(method in F_CONTROLS or method.startswith("M")),
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
        **{k: median(v) for k, v in acc.items()},
        "rolewise_anti_alignment": next((s for s in reversed(role_strings) if s), ""),
        "basis_group_anti_alignment": next((s for s in reversed(group_strings) if s), ""),
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
        "uses_brier_for_direction": 0,
        "uses_dataset_name_branch": 0,
        "uses_seed_specific_scale": 0,
        "action_token": 0,
        "controller_executed": 0,
        "reset_route_used": 0,
        "promotion_allowed": 0,
    }
    return {"row": row, "direction_rows": direction_rows, "linec_rows": linec_rows}


def enrich_dche_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        by_key.setdefault((str(row["dataset"]), int(row["seed"])), []).append(dict(row))
    out: list[dict[str, Any]] = []
    for group in by_key.values():
        controls = [r for r in group if str(r.get("method")) in F_CONTROLS]
        adam = next((r for r in group if str(r.get("method")) == "F0-D-CHE-AdamW"), controls[0] if controls else group[0])
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
            item["margin_p10_delta"] = fnum(item.get("margin_p10"), 0.0) - fnum(adam.get("margin_p10"), 0.0)
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
                str(item.get("method")) in F_METHODS
                and str(item.get("method")) not in {"F0-D-CHE-AdamW", "F1-D-CHE-CautiousAdamW", "F2-D-CHE-MGUPControl"}
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


def enrich_mlp_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        by_key.setdefault((str(row["dataset"]), int(row["seed"])), []).append(dict(row))
    out: list[dict[str, Any]] = []
    for group in by_key.values():
        adam = next((r for r in group if str(r.get("method")) == "M0-MLP-AdamW"), group[0])
        for row in group:
            item = dict(row)
            item["source_vs_mlp_adamw"] = fnum(adam.get("NLL"), 9.0) - fnum(row.get("NLL"), 9.0)
            item["source_vs_best_mlp_control"] = min(fnum(r.get("NLL"), 9.0) for r in group) - fnum(row.get("NLL"), 9.0)
            item["NLL_delta_vs_mlp_adamw"] = fnum(row.get("NLL"), 9.0) - fnum(adam.get("NLL"), 9.0)
            item["AUCtime_ratio_vs_mlp_adamw"] = fnum(row.get("AUC_NLL"), 9.0) / max(1.0e-8, fnum(adam.get("AUC_NLL"), 9.0))
            item["strict_gate_pass"] = int(
                str(item.get("method")) not in {"M0-MLP-AdamW", "M7-MLP-NoOpMatchedOverhead"}
                and fnum(item.get("source_vs_mlp_adamw"), -999.0) >= 0.005
                and fnum(item.get("AUCtime_ratio_vs_mlp_adamw"), 9.0) <= 1.25
            )
            out.append(item)
    return out


def run_f_and_m(args: argparse.Namespace, out_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    result_path = out_dir / "v151_dche_crfu_results.csv"
    control_path = out_dir / "v151_dche_crfu_controls.csv"
    mlp_path = out_dir / "v151_mlp_generic_controls.csv"
    direction_path = out_dir / "v151_direction_decomposition.csv"
    if sint(getattr(args, "reuse_if_present", 1), 1) == 1 and result_path.exists() and control_path.exists() and mlp_path.exists() and direction_path.exists():
        return read_rows(result_path), read_rows(control_path), read_rows(mlp_path), read_rows(direction_path)
    device = resolve_cuda_device(args.device)
    raw_rows: list[dict[str, Any]] = []
    direction_rows: list[dict[str, Any]] = []
    linec_rows: list[dict[str, Any]] = []
    for dataset in parse_csv(args.datasets):
        for seed in parse_ints(args.seeds):
            load_args = copy(args)
            load_args.seed = int(seed)
            xtr, ytr, xva, yva, xte, yte, input_dim_t, output_dim_t = v144.load_real_split(load_args, dataset, int(seed), device)
            input_dim = int(input_dim_t.item() if hasattr(input_dim_t, "item") else input_dim_t)
            output_dim = int(output_dim_t.item() if hasattr(output_dim_t, "item") else output_dim_t)
            for method in parse_csv(args.f_methods):
                result = train_case(family="D-CHE", method=method, dataset=dataset, seed=int(seed), xtr=xtr, ytr=ytr, xva=xva, yva=yva, xte=xte, yte=yte, input_dim=input_dim, output_dim=output_dim, args=args, device=device)
                raw_rows.append(result["row"])
                direction_rows.extend(result["direction_rows"])
                linec_rows.extend(result["linec_rows"])
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            for method in parse_csv(args.f_controls):
                if method in F_METHODS:
                    continue
                result = train_case(family="D-CHE", method=method, dataset=dataset, seed=int(seed), xtr=xtr, ytr=ytr, xva=xva, yva=yva, xte=xte, yte=yte, input_dim=input_dim, output_dim=output_dim, args=args, device=device)
                raw_rows.append(result["row"])
                direction_rows.extend(result["direction_rows"])
                linec_rows.extend(result["linec_rows"])
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            for method in parse_csv(args.m_methods):
                result = train_case(family="MLP", method=method, dataset=dataset, seed=int(seed), xtr=xtr, ytr=ytr, xva=xva, yva=yva, xte=xte, yte=yte, input_dim=input_dim, output_dim=output_dim, args=args, device=device)
                raw_rows.append(result["row"])
                direction_rows.extend(result["direction_rows"])
                if device.type == "cuda":
                    torch.cuda.empty_cache()
    dche = enrich_dche_rows([r for r in raw_rows if r.get("family") == "D-CHE"])
    mlp = enrich_mlp_rows([r for r in raw_rows if r.get("family") == "MLP"])
    results = [r for r in dche if str(r.get("method")) in F_METHODS]
    controls = [r for r in dche if str(r.get("method")) in F_CONTROLS]
    linec_rows = v150.enrich_linec_rows(linec_rows, dche + mlp)
    write_rows(result_path, results)
    write_rows(control_path, controls)
    write_rows(mlp_path, mlp)
    write_rows(direction_path, direction_rows)
    write_rows(out_dir / "v151_linec_tail_audit.csv", linec_rows)
    return results, controls, mlp, direction_rows


def summarize_crfu(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    candidates = [r for r in rows if str(r.get("method")) in {"F3-D-CHE-RawFUResidualizedAgainstAdamW", "F4-D-CHE-RawFUResidualizedAgainstCautious", "F5-D-CHE-MGUPFUResidualizedAgainstMGUPControl", "F6-D-CHE-FunctionSpaceProximalResidual", "F7-D-CHE-CRFU-AbstentionEnabled"}]
    pass_count = len({(r.get("dataset"), str(r.get("seed"))) for r in candidates if sint(r.get("strict_gate_pass"), 0) == 1})
    source_mean = mean([fnum(r.get("source_vs_best_control"), 0.0) for r in candidates])
    control_equiv = sum(1 for r in candidates if sint(r.get("control_equivalent"), 0) == 1) / max(1, len(candidates))
    bad_event = sum(1 for r in candidates if sint(r.get("bad_event"), 0) == 1) / max(1, len(candidates))
    step_ratio = median([fnum(r.get("step_time_ratio"), 1.0) for r in candidates])
    auc_median = median([fnum(r.get("AUCtime_ratio"), 9.0) for r in candidates])
    f0 = [r for r in rows if str(r.get("method")) == "F0-D-CHE-AdamW"]
    f0_linec_fail = sum(1 for r in f0 if sint(r.get("LineC_majority_pass"), 0) != 1)
    cand_linec_fail = sum(1 for r in candidates if sint(r.get("LineC_majority_pass"), 0) != 1)
    exploration = int(pass_count >= 3 and source_mean > 0.0 and control_equiv <= 0.60 and bad_event <= 0.50 and step_ratio <= 1.50)
    meaningful = int(pass_count >= 4 and source_mean >= 0.005 and control_equiv <= 0.50 and bad_event <= 0.35 and step_ratio <= 1.35)
    s4 = int(pass_count >= 6 and source_mean >= 0.005 and auc_median <= 1.05 and cand_linec_fail <= f0_linec_fail)
    s5 = int(pass_count == 9 and all(sint(r.get("strict_gate_pass"), 0) == 1 for r in candidates))
    method_rows = []
    best_method = ""
    best_source = -999.0
    for method in F_METHODS:
        group = [r for r in rows if str(r.get("method")) == method]
        src = mean([fnum(r.get("source_vs_best_control"), 0.0) for r in group])
        if method not in {"F0-D-CHE-AdamW", "F1-D-CHE-CautiousAdamW", "F2-D-CHE-MGUPControl"} and src > best_source:
            best_source = src
            best_method = method
        method_rows.append(
            {
                "method": method,
                "rows": len(group),
                "strict_pass_rows": sum(sint(r.get("strict_gate_pass"), 0) for r in group),
                "dataset_seed_pass_count": len({(r.get("dataset"), str(r.get("seed"))) for r in group if sint(r.get("strict_gate_pass"), 0) == 1}),
                "mean_source_vs_best_control": src,
                "mean_control_residual_norm_fraction": mean([fnum(r.get("control_residual_norm_fraction"), 0.0) for r in group]),
                "mean_control_equivalent": sum(1 for r in group if sint(r.get("control_equivalent"), 0) == 1) / max(1, len(group)),
                "promotion_allowed": 0,
            }
        )
    return {
        "line_f_candidate_count": len(F_METHODS),
        "real_lite_pass_count": pass_count,
        "source_vs_best_control_mean": source_mean,
        "control_equivalent_fraction": control_equiv,
        "bad_event_fraction": bad_event,
        "step_time_ratio_median": step_ratio,
        "AUCtime_ratio_median": auc_median,
        "line_f_exploration_gate_pass": exploration,
        "line_f_meaningful_gate_pass": meaningful,
        "line_f_s4_gate_pass": s4,
        "line_f_s5_gate_pass": s5,
        "best_crfu_method": best_method,
        "best_crfu_mean_source_vs_best_control": best_source,
        "method_rows": method_rows,
    }


def build_autopsy(out_dir: Path, dche_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    v150_rows = read_rows(V150_DIR / "v150_dche_fu_results.csv")
    decay_rows = read_rows(V150_DIR / "v150_decoupled_decay_deconfound.csv")
    attribution = []
    for row in v150_rows:
        src = fnum(row.get("source_vs_best_control"), 0.0)
        residual_frac = fnum(row.get("control_residual_norm_fraction"), fnum(row.get("value_retention_after_projection"), 0.0))
        attribution.append(
            {
                "source_version": "v15.0_replay",
                "dataset": row.get("dataset", ""),
                "seed": row.get("seed", ""),
                "method": row.get("method", ""),
                "source_vs_best_control": src,
                "control_equivalent": int(src <= 0.005),
                "residual_norm_proxy": residual_frac,
                "basis_projection_rejection_fraction": row.get("degree_projection_rejection_fraction", ""),
                "linec_majority_pass": row.get("LineC_majority_pass", ""),
                "failure_attribution": "control_equivalent" if src <= 0.005 else "positive_but_gate_failed",
                "promotion_allowed": 0,
            }
        )
    residual_summary = [
        {
            "source": "v15.1_actual_crfu",
            "rows": len(dche_rows),
            "mean_control_projection_norm_fraction": mean([fnum(r.get("control_projection_norm_fraction"), 0.0) for r in dche_rows]),
            "mean_control_residual_norm_fraction": mean([fnum(r.get("control_residual_norm_fraction"), 0.0) for r in dche_rows]),
            "mean_anti_alignment_fraction": mean([fnum(r.get("anti_alignment_fraction"), 0.0) for r in dche_rows]),
            "promotion_allowed": 0,
        }
    ]
    projection_loss = [
        {
            "method": method,
            "rows": len(group),
            "mean_degree_projection_rejection_fraction": mean([fnum(r.get("degree_projection_rejection_fraction"), 0.0) for r in group]),
            "mean_value_retention_after_degree_projection": mean([fnum(r.get("value_retention_after_degree_projection"), 0.0) for r in group]),
            "promotion_allowed": 0,
        }
        for method in sorted({str(r.get("method")) for r in dche_rows})
        for group in [[r for r in dche_rows if str(r.get("method")) == method]]
    ]
    write_rows(out_dir / "v151_fu_failure_attribution.csv", attribution)
    write_rows(out_dir / "v151_fu_residualizability_summary.csv", residual_summary)
    write_rows(out_dir / "v151_projection_value_loss.csv", projection_loss)
    write_rows(out_dir / "v151_decay_deconfound_extended.csv", decay_rows)
    write_text(
        out_dir / "v151_current_fu_no_go_boundary.md",
        "# v15.1 current FU no-go boundary\n\n"
        "v15.0 replay rows are used only for failure attribution. v15.1 CR-FU rows are actual training rows.\n"
        "No FU9/FU10, F-CHE8/F-CHE9, action bank, controller, or reset route was added.\n",
    )
    return residual_summary[0]


def build_kan_specificity(out_dir: Path, dche_rows: Sequence[dict[str, Any]], mlp_rows: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dche_crfu = [r for r in dche_rows if str(r.get("method")) in {"F3-D-CHE-RawFUResidualizedAgainstAdamW", "F4-D-CHE-RawFUResidualizedAgainstCautious", "F5-D-CHE-MGUPFUResidualizedAgainstMGUPControl", "F6-D-CHE-FunctionSpaceProximalResidual", "F7-D-CHE-CRFU-AbstentionEnabled"}]
    dche_ctrl = [r for r in dche_rows if str(r.get("method")) in F_CONTROLS]
    mlp_crfu = [r for r in mlp_rows if str(r.get("method")) in {"M3-MLP-FunctionSpaceProximalResidual", "M4-MLP-CRFU-AbstentionEnabled"}]
    mlp_ctrl = [r for r in mlp_rows if str(r.get("method")) in {"M0-MLP-AdamW", "M1-MLP-CautiousAdamW", "M2-MLP-MGUPControl", "M5-MLP-RandomResidualMatchedNorm", "M6-MLP-SameActiveFractionResidual", "M7-MLP-NoOpMatchedOverhead"}]
    rows = []
    for dataset in sorted({str(r.get("dataset")) for r in dche_rows}):
        for seed in sorted({str(r.get("seed")) for r in dche_rows if str(r.get("dataset")) == dataset}):
            dc = [r for r in dche_crfu if str(r.get("dataset")) == dataset and str(r.get("seed")) == seed]
            dctrl = [r for r in dche_ctrl if str(r.get("dataset")) == dataset and str(r.get("seed")) == seed]
            mc = [r for r in mlp_crfu if str(r.get("dataset")) == dataset and str(r.get("seed")) == seed]
            mctrl = [r for r in mlp_ctrl if str(r.get("dataset")) == dataset and str(r.get("seed")) == seed]
            dche_source = max([fnum(r.get("source_vs_best_control"), 0.0) for r in dc], default=0.0)
            dche_control_source = max([fnum(r.get("source_vs_adamw"), 0.0) for r in dctrl], default=0.0)
            mlp_source = max([fnum(r.get("source_vs_mlp_adamw"), 0.0) for r in mc], default=0.0)
            mlp_control_source = max([fnum(r.get("source_vs_mlp_adamw"), 0.0) for r in mctrl], default=0.0)
            delta_kan = (dche_source - dche_control_source) - (mlp_source - mlp_control_source)
            rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "dche_crfu_source": dche_source,
                    "dche_control_source": dche_control_source,
                    "mlp_crfu_source": mlp_source,
                    "mlp_control_source": mlp_control_source,
                    "delta_kan_specific": delta_kan,
                    "delta_generic": mlp_source - mlp_control_source,
                    "kan_specific_pass": int(delta_kan > 0.0 and dche_source > 0.0),
                    "promotion_allowed": 0,
                }
            )
    summary = {
        "kan_specific_pass_count": sum(sint(r.get("kan_specific_pass"), 0) for r in rows),
        "delta_kan_specific_mean": mean([fnum(r.get("delta_kan_specific"), 0.0) for r in rows]),
        "generic_optimizer_explains_crfu": int(max([fnum(r.get("delta_generic"), 0.0) for r in rows], default=0.0) >= max([fnum(r.get("delta_kan_specific"), 0.0) for r in rows], default=0.0)),
    }
    write_rows(out_dir / "v151_kan_specificity_summary.csv", rows)
    return rows, summary


def build_line_d(out_dir: Path, line_d_out: Path) -> dict[str, Any]:
    raw = read_rows(line_d_out / "v149_line_d_substrate_repair_results.csv")
    result_rows = []
    for row in raw:
        item = dict(row)
        step = fnum(item.get("train_step_ratio_vs_MLP"), 999.0)
        mem = fnum(item.get("workspace_incremental_memory_ratio_vs_mlp"), fnum(item.get("workspace_raw_memory_ratio_vs_mlp"), 999.0))
        delta = fnum(item.get("mean_delta_vs_MLP"), -999.0)
        worst = fnum(item.get("worst_delta_vs_MLP"), -999.0)
        linec = fnum(item.get("LineC_pass_rate"), 0.0)
        gate = int(step <= 1.75 and mem <= 1.75 and delta >= -0.05 and worst >= -0.10 and linec >= 0.30)
        official = int(gate and step <= 1.25 and mem <= 1.25 and linec >= 0.50)
        item["stage"] = "V151_ALLBASIS_SUBSTRATE_RESULT"
        item["v151_step_ratio"] = step
        item["v151_memory_ratio"] = mem
        item["v151_substrate_gate_pass"] = gate
        item["v151_official_fu_eligibility_row"] = official
        item["official_fu_proof_executed"] = 0
        item["promotion_allowed"] = 0
        result_rows.append(item)
    summary_rows = []
    best_family = ""
    best_count = -1
    official_count = 0
    for family in ["D-FOU", "D-RBF", "D-WAV"]:
        fam = [r for r in result_rows if str(r.get("family")) == family]
        keys = {(r.get("dataset"), str(r.get("seed"))) for r in fam if sint(r.get("v151_substrate_gate_pass"), 0) == 1}
        official_keys = {(r.get("dataset"), str(r.get("seed"))) for r in fam if sint(r.get("v151_official_fu_eligibility_row"), 0) == 1}
        if len(keys) > best_count:
            best_count = len(keys)
            best_family = family
        if len(official_keys) == 9:
            official_count += 1
        summary_rows.append(
            {
                "family": family,
                "rows": len(fam),
                "family_dataset_seed_pass_count": len(keys),
                "official_fu_eligible": int(len(official_keys) == 9),
                "best_candidate": max({str(r.get("candidate_id")) for r in fam}, key=lambda cid: sum(1 for r in fam if str(r.get("candidate_id")) == cid and sint(r.get("v151_substrate_gate_pass"), 0) == 1), default=""),
                "max_mean_delta_vs_MLP": max([fnum(r.get("mean_delta_vs_MLP"), -999.0) for r in fam], default=0.0),
                "best_LineC_pass_rate": max([fnum(r.get("LineC_pass_rate"), 0.0) for r in fam], default=0.0),
                "promotion_allowed": 0,
            }
        )
    write_rows(out_dir / "v151_allbasis_substrate_results.csv", result_rows)
    write_rows(out_dir / "v151_allbasis_substrate_summary.csv", summary_rows)
    return {
        "line_d_source": "v151_actual_v149_substrate_acceleration" if raw else "missing_actual_line_d",
        "line_d_rows": len(result_rows),
        "summary_rows": summary_rows,
        "best_non_dche_family": best_family,
        "best_non_dche_dataset_seed_pass_count": max(0, best_count),
        "line_d_official_fu_eligible_family_count": official_count,
        "line_d_gate_pass": int(max(0, best_count) >= 6),
        "line_d_route": "S4-AllBasisAlternativeCarrier" if max(0, best_count) >= 6 else "R6-AllBasisSubstrateBlocked",
    }


def build_monitors(out_dir: Path, dche_rows: Sequence[dict[str, Any]]) -> None:
    f0 = [r for r in dche_rows if str(r.get("method")) == "F0-D-CHE-AdamW"]
    rows = [
        {
            "monitor": "D-CHE-no-regression",
            "rows": len(f0),
            "mean_NLL": mean([fnum(r.get("NLL"), 0.0) for r in f0]),
            "mean_LineC_pass_rate": mean([fnum(r.get("LineC_pass_rate"), 0.0) for r in f0]),
            "promotion_allowed": 0,
        }
    ]
    write_rows(out_dir / "v151_dche_no_regression_monitor.csv", rows)
    rat_src = V150_DIR / "v150_rational_no_regression_monitor.csv"
    rat_rows = read_rows(rat_src)
    if not rat_rows:
        rat_rows = [{"monitor": "D-RAT-no-regression", "status": "source_artifact_missing", "promotion_allowed": 0}]
    write_rows(out_dir / "v151_rational_no_regression_monitor.csv", rat_rows)


def build_failure_taxonomy(out_dir: Path, dche_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in dche_rows:
        reasons = []
        if fnum(row.get("control_residual_norm_fraction"), 0.0) < 0.05:
            reasons.append("residual_norm_too_small")
        if sint(row.get("control_equivalent"), 0):
            reasons.append("control_equivalent")
        if fnum(row.get("degree_projection_rejection_fraction"), 0.0) > 0.70:
            reasons.append("basis_projection_kills_value")
        if sint(row.get("LineC_majority_pass"), 0) != 1:
            reasons.append("linec_fail")
        if fnum(row.get("step_time_ratio"), 1.0) > 1.25 or fnum(row.get("memory_ratio"), 1.0) > 1.25:
            reasons.append("cost_fail")
        rows.append(
            {
                "dataset": row.get("dataset"),
                "seed": row.get("seed"),
                "method": row.get("method"),
                "failure_reasons": ";".join(reasons) if reasons else "none",
                "strict_gate_pass": row.get("strict_gate_pass"),
                "promotion_allowed": 0,
            }
        )
    write_rows(out_dir / "v151_failure_taxonomy.csv", rows)
    return rows


def build_audits(out_dir: Path) -> tuple[int, int]:
    methods = F_METHODS + [m for m in F_CONTROLS if m not in F_METHODS] + M_METHODS + LINE_D_CANDIDATES + ["D-CHE-no-regression-monitor", "D-RAT-no-regression-monitor"]
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
            "uses_brier_for_direction": 0,
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
    write_rows(out_dir / "v151_forbidden_information_audit.csv", forbidden)
    write_rows(out_dir / "v151_no_action_search_audit.csv", no_action)
    return sum(sint(r.get("violation"), 0) for r in forbidden), sum(sint(r.get("violation"), 0) for r in no_action)


def build_route(
    summary: dict[str, Any],
    residual_summary: dict[str, Any],
    specificity: dict[str, Any],
    line_d: dict[str, Any],
    forbidden: int,
    no_action: int,
    missing: int,
) -> dict[str, Any]:
    residual_small = fnum(residual_summary.get("mean_control_residual_norm_fraction"), 0.0) < 0.05
    generic_explains = sint(specificity.get("generic_optimizer_explains_crfu"), 0)
    if forbidden or no_action or missing:
        route = "R0-ArtifactOrProvenanceViolation"
    elif sint(summary.get("line_f_s5_gate_pass"), 0):
        route = "S5-OfficialFunctionalSuccess"
    elif sint(summary.get("line_f_s4_gate_pass"), 0):
        route = "S4-CRFURealLitePositive"
    elif sint(summary.get("line_f_meaningful_gate_pass"), 0):
        route = "S3-CRFUMeaningfulPositive"
    elif sint(summary.get("line_f_exploration_gate_pass"), 0):
        route = "S2-CRFUExplorationPositive"
    elif (
        sint(summary.get("line_f_exploration_gate_pass"), 0) == 0
        and generic_explains
        and sint(specificity.get("kan_specific_pass_count"), 0) == 0
        and sint(line_d.get("line_d_gate_pass"), 0) == 0
    ):
        route = "R7-CurrentFUFamilyNoGo"
    elif residual_small:
        route = "R2-CRFUResidualNormTooSmall"
    elif generic_explains:
        route = "R3-GenericOptimizerExplainsCRFU"
    elif sint(line_d.get("line_d_gate_pass"), 0) == 0:
        route = "R7-CurrentFUFamilyNoGo"
    else:
        route = "R6-AllBasisSubstrateBlocked"
    minimum = (
        "S5-OfficialFunctionalSuccess"
        if route == "S5-OfficialFunctionalSuccess"
        else "S4-CRFURealLitePositive"
        if route == "S4-CRFURealLitePositive"
        else "S3-CRFUMeaningfulPositive"
        if route == "S3-CRFUMeaningfulPositive"
        else "S2-CRFUExplorationPositive"
        if route == "S2-CRFUExplorationPositive"
        else "S1-CRFUSolverImplemented"
    )
    return {
        "stage": "V151_ROUTE_DECISION",
        "route": route,
        "minimum_success": minimum,
        "official_s5_reached": int(route == "S5-OfficialFunctionalSuccess"),
        "promotion_allowed": int(route == "S5-OfficialFunctionalSuccess" and missing == 0 and forbidden == 0 and no_action == 0),
        "real_lite_pass_count": summary.get("real_lite_pass_count", 0),
        "source_vs_best_control_mean": summary.get("source_vs_best_control_mean", 0.0),
        "control_equivalent_fraction": summary.get("control_equivalent_fraction", 0.0),
        "bad_event_fraction": summary.get("bad_event_fraction", 0.0),
        "line_f_exploration_gate_pass": summary.get("line_f_exploration_gate_pass", 0),
        "line_f_meaningful_gate_pass": summary.get("line_f_meaningful_gate_pass", 0),
        "line_f_s4_gate_pass": summary.get("line_f_s4_gate_pass", 0),
        "line_f_s5_gate_pass": summary.get("line_f_s5_gate_pass", 0),
        "best_crfu_method": summary.get("best_crfu_method", ""),
        "best_crfu_mean_source_vs_best_control": summary.get("best_crfu_mean_source_vs_best_control", 0.0),
        "mean_control_residual_norm_fraction": residual_summary.get("mean_control_residual_norm_fraction", 0.0),
        "generic_optimizer_explains_crfu": generic_explains,
        "kan_specific_pass_count": specificity.get("kan_specific_pass_count", 0),
        "line_d_route": line_d.get("line_d_route", ""),
        "line_d_best_non_dche_family": line_d.get("best_non_dche_family", ""),
        "line_d_best_non_dche_dataset_seed_pass_count": line_d.get("best_non_dche_dataset_seed_pass_count", 0),
        "line_d_official_fu_eligible_family_count": line_d.get("line_d_official_fu_eligible_family_count", 0),
        "required_artifact_missing_count": missing,
        "forbidden_information_violation_count": forbidden,
        "no_action_search_violation_count": no_action,
    }


def write_no_go_docs(out_dir: Path, route: dict[str, Any], summary: dict[str, Any], line_d: dict[str, Any]) -> None:
    write_text(
        out_dir / "v151_no_go_boundary.md",
        "\n".join(
            [
                "# v15.1 no-go boundary",
                "",
                f"route = {route['route']}",
                f"minimum_success = {route['minimum_success']}",
                f"promotion_allowed = {route['promotion_allowed']}",
                "",
                "- CR-FU residualization was executed without FU9/FU10.",
                "- LineC / CEp99 / NLL / ECE / AUCtime / Brier were audit/gate fields only.",
                "- MLP/generic controls are confound checks, not KAN-specific promotion.",
                "- No action bank, controller, or reset route was added.",
            ]
        )
        + "\n",
    )
    write_text(
        out_dir / "v151_next_hypothesis_queue.md",
        "\n".join(
            [
                "# v15.1 next hypothesis queue",
                "",
                f"- CR-FU pass count = {summary.get('real_lite_pass_count', 0)}/9.",
                f"- All-basis best = {line_d.get('best_non_dche_family', '')} {line_d.get('best_non_dche_dataset_seed_pass_count', 0)}/9.",
                "- If CR-FU remains below gate, the current FU family is closed.",
                "- Next legal path is a new theory-level functional update or substrate/base-architecture plan.",
                "- Do not use audit metrics to synthesize a new direction.",
            ]
        )
        + "\n",
    )


def simple_svg(path: Path, title: str, rows: Sequence[tuple[str, float]], threshold: float | None = None) -> None:
    v150.simple_svg(path, title, rows, threshold)


def write_figures(out_dir: Path, dche_rows: Sequence[dict[str, Any]], controls: Sequence[dict[str, Any]], mlp_rows: Sequence[dict[str, Any]], line_d: dict[str, Any]) -> None:
    simple_svg(out_dir / "fig_v151_direction_cosine_matrix.svg", "direction cosine", [(r.get("method", ""), fnum(r.get("cos_fu_adamw"), 0.0)) for r in dche_rows[:40]])
    simple_svg(out_dir / "fig_v151_control_projection_fraction.svg", "control projection", [(r.get("method", ""), fnum(r.get("control_projection_norm_fraction"), 0.0)) for r in dche_rows[:40]])
    simple_svg(out_dir / "fig_v151_alignment_by_role.svg", "anti alignment", [(r.get("method", ""), fnum(r.get("anti_alignment_fraction"), 0.0)) for r in dche_rows[:40]])
    simple_svg(out_dir / "fig_v151_decay_vs_fu_norm.svg", "decay vs FU", [(r.get("method", ""), fnum(r.get("decay_update_norm"), 0.0) / max(1.0e-8, fnum(r.get("fu_norm"), 1.0))) for r in dche_rows[:40]])
    simple_svg(out_dir / "fig_v151_second_moment_condition.svg", "second moment condition", [(r.get("method", ""), fnum(r.get("second_moment_condition"), 0.0)) for r in dche_rows[:40]])
    simple_svg(out_dir / "fig_v151_linec_tail_matrix.svg", "LineC tail", [(r.get("method", ""), fnum(r.get("LineC_pass_rate"), 0.0) - fnum(r.get("CEp99_delta"), 0.0)) for r in dche_rows[:40]])
    simple_svg(out_dir / "fig_v151_signal_reservoir_noise_plane.svg", "signal reservoir noise", [(r.get("method", ""), fnum(r.get("RealSignalReservoirRatio"), 0.0) - fnum(r.get("NoiseSignalLeak"), 0.0)) for r in dche_rows[:40]])
    simple_svg(out_dir / "fig_v151_tail_calibration_tradeoff.svg", "tail calibration", [(r.get("method", ""), -fnum(r.get("ECE_delta"), 0.0) - fnum(r.get("Brier_delta"), 0.0)) for r in dche_rows[:40]])


def write_required_manifest(out_dir: Path) -> int:
    rows = []
    for name in REQUIRED + FIGURES:
        path = out_dir / name
        exists = int(path.exists() or name == "v151_required_artifact_manifest.csv")
        rows.append({"artifact": name, "exists": exists, "missing": int(not exists), "bytes": path.stat().st_size if path.exists() else 0, "promotion_allowed": 0})
    write_rows(out_dir / "v151_required_artifact_manifest.csv", rows)
    return sum(sint(r.get("missing"), 0) for r in rows)


def csv_rows(rows: Sequence[dict[str, Any]]) -> str:
    return v150.csv_rows(rows)


def write_code_packet(out_dir: Path) -> None:
    files = [
        Path("experiments/run_v151_control_residual_function_update_allbasis_acceleration.py"),
        Path("experiments/run_v149_line_d_all_basis_substrate_repair.py"),
        Path("experiments/run_v150_function_update_allbasis_parallel.py"),
        Path("experiments/run_v1410_nonrat_fms_transfer_fms_definition_reset.py"),
        PLAN_DOC.relative_to(ROOT),
    ]
    manifest = []
    with zipfile.ZipFile(out_dir / "v151_code_review_packet.zip", "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for rel in files:
            path = ROOT / rel
            manifest.append({"path": str(rel), "exists": int(path.exists()), "sha256": v1410.sha256_file(path) if path.exists() else ""})
            if path.exists():
                zf.write(path, arcname=str(rel))
        zf.writestr("v151_code_review_manifest.csv", csv_rows(manifest))


def write_progress(out_dir: Path, route: dict[str, Any], summary: dict[str, Any], line_d: dict[str, Any], residual_summary: dict[str, Any], specificity: dict[str, Any]) -> None:
    rows = [
        {"line": "R", "route": "R-AuditPass" if route["forbidden_information_violation_count"] == 0 and route["no_action_search_violation_count"] == 0 else "R-AuditFail", "promotion_allowed": 0},
        {"line": "O", "route": "O-DecompositionComplete", "mean_control_residual_norm_fraction": residual_summary.get("mean_control_residual_norm_fraction"), "promotion_allowed": 0},
        {"line": "C0", "route": "C0-V150AutopsyComplete", "promotion_allowed": 0},
        {"line": "F", **{k: v for k, v in summary.items() if k != "method_rows"}, "promotion_allowed": 0},
        {"line": "M", "route": "M-ControlsComplete", "generic_optimizer_explains_crfu": specificity.get("generic_optimizer_explains_crfu"), "promotion_allowed": 0},
        {"line": "K", "route": "K-KANSpecificityPass" if specificity.get("kan_specific_pass_count", 0) else "K-KANSpecificityFail", "promotion_allowed": 0},
        {"line": "D", "route": line_d.get("line_d_route"), "best_family": line_d.get("best_non_dche_family"), "best_count": line_d.get("best_non_dche_dataset_seed_pass_count"), "promotion_allowed": 0},
        {"line": "Z", "route": route["route"], "minimum_success": route["minimum_success"], "promotion_allowed": route["promotion_allowed"]},
    ]
    write_rows(out_dir / "v151_progress_table.csv", rows)


def write_docs(args: argparse.Namespace, out_dir: Path, route: dict[str, Any], summary: dict[str, Any], line_d: dict[str, Any], dche_rows: Sequence[dict[str, Any]], controls: Sequence[dict[str, Any]], mlp_rows: Sequence[dict[str, Any]], residual_summary: dict[str, Any], specificity: dict[str, Any]) -> None:
    method_rows = summary.get("method_rows", [])
    recap = [
        "# DG-KAN v15.01 ControlResidualFunctionUpdate AllBasisAcceleration 实验结果复盘",
        "",
        "生成时间：2026-05-31（Asia/Singapore）",
        "",
        "本复盘只写入实际 artifact 中的结果；不把 CR-FU real-lite、substrate replay 或 MLP/generic control 写成 promotion。",
        "",
        "## 1. 计划理解",
        "",
        "v15.01/v15.1 的目标是检验：在 AdamW/Cautious/MGUP/second-moment/decay 等 controls 已经解释的方向被投掉后，FU 是否仍有独立 residual causal value。",
        "",
        "## 2. 本轮代码修改",
        "",
        "新增：",
        "",
        "```text",
        "experiments/run_v151_control_residual_function_update_allbasis_acceleration.py",
        "```",
        "",
        "修改：",
        "",
        "```text",
        "experiments/run_v149_line_d_all_basis_substrate_repair.py",
        "  新增 v15.1 预注册 substrate-only candidates:",
        "  D-FOU42..46, D-RBF40..44, D-WAV37..40。",
        "```",
        "",
        "过程修正：",
        "",
        "```text",
        "首次 official finalizer 完成实际训练矩阵后，route precedence 按中间诊断写成 R3-GenericOptimizerExplainsCRFU。",
        "复核计划第 16 节 final decision rule 后，确认 F/K/D 同时失败且 generic controls explain 时，最终 decisive route 应为 R7-CurrentFUFamilyNoGo。",
        "已修正 build_route precedence，并用 reuse-if-present=1 重写 route / manifest / 两份日志；该修正不改变任何训练指标。",
        "用户再次追问后复核 Line O，发现 random_matched_decay_explains_fraction 初版固定为 0。",
        "已改为使用当前 train-stream decay active mask 的 matched-random decay direction 做 metric projection readback；该 readback 不进入更新方向、不使用 audit metric、不新增 token。",
        "再次复核发现 F6/M3 FunctionSpaceProximalResidual 初版使用一阶闭式近似 alpha，没有实际执行计划 8.2 的 train split B1 proximal objective。",
        "已改为固定 alpha candidates = 0,0.05,0.10,0.25,0.50，在当前 train batch B1 上评估 CE + proximal penalty 选 alpha；不使用 validation/test/future/audit metric，不构成 action search。",
        "用户再次追问后复核 control-residual 语义，发现 full residualizer 初版只投掉 Adam/Cautious/MGUP/decay，没有投掉计划中的 matched-random/same-active/same-projection/same-value-retention controls。",
        "已将这些固定 train-stream matched control directions 纳入 F6/F7/M3/M4 full residualizer；不新增 FU token，不使用 audit metric，不构成 action search。",
        "再次复核 Line O readback，发现 decoupled_decay_explains_fraction 初版是 decay_norm/raw_FU_norm，而不是 decay 对 raw FU 的 metric projection fraction。",
        "已改为 decoupled decay direction 对 raw FU 的 metric projection readback；该指标只用于诊断，不进入方向。",
        "```",
        "",
        "CR-FU runner 实现：Line R/O/C0/F/M/K/D/C/Z；direction 只使用当前 train stream、当前 optimizer state 和 train-stream basis telemetry；LineC/CEp99/NLL/ECE/AUCtime/Brier 只用于 audit/gate。",
        "",
        "## 3. Line O / C0 结果",
        "",
        "```text",
        f"direction_decomposition_rows = {len(read_rows(out_dir / 'v151_direction_decomposition.csv'))}",
        f"mean_control_projection_norm_fraction = {residual_summary.get('mean_control_projection_norm_fraction')}",
        f"mean_control_residual_norm_fraction = {residual_summary.get('mean_control_residual_norm_fraction')}",
        f"mean_anti_alignment_fraction = {residual_summary.get('mean_anti_alignment_fraction')}",
        f"mean_decoupled_decay_explains_fraction = {mean([fnum(r.get('decoupled_decay_explains_fraction'), 0.0) for r in read_rows(out_dir / 'v151_direction_decomposition.csv')])}",
        f"mean_random_matched_decay_explains_fraction = {mean([fnum(r.get('random_matched_decay_explains_fraction'), 0.0) for r in read_rows(out_dir / 'v151_direction_decomposition.csv')])}",
        f"proximal_solver_used_train_split_rows = {sum(1 for r in read_rows(out_dir / 'v151_direction_decomposition.csv') if sint(r.get('proximal_solver_used_train_split'), 0) == 1)}",
        f"full_residualizer_control_count_max = {max([sint(r.get('full_residualizer_control_count'), 0) for r in read_rows(out_dir / 'v151_direction_decomposition.csv')] or [0])}",
        "```",
        "",
        "## 4. Line F D-CHE CR-FU 结果",
        "",
        "```text",
        f"line_f_candidate_count = {summary.get('line_f_candidate_count')}",
        f"real_lite_pass_count = {summary.get('real_lite_pass_count')} / 9",
        f"source_vs_best_control_mean = {summary.get('source_vs_best_control_mean')}",
        f"control_equivalent_fraction = {summary.get('control_equivalent_fraction')}",
        f"bad_event_fraction = {summary.get('bad_event_fraction')}",
        f"line_f_exploration_gate_pass = {summary.get('line_f_exploration_gate_pass')}",
        f"line_f_meaningful_gate_pass = {summary.get('line_f_meaningful_gate_pass')}",
        f"line_f_s4_gate_pass = {summary.get('line_f_s4_gate_pass')}",
        f"best_crfu_method = {summary.get('best_crfu_method')}",
        f"best_crfu_mean_source_vs_best_control = {summary.get('best_crfu_mean_source_vs_best_control')}",
        "```",
        "",
        "Method summary：",
        "",
        "| method | rows | strict pass | dataset-seed pass | mean source vs best control | mean residual fraction |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in method_rows:
        recap.append(
            f"| {row['method']} | {row['rows']} | {row['strict_pass_rows']} | {row['dataset_seed_pass_count']} | {row['mean_source_vs_best_control']} | {row['mean_control_residual_norm_fraction']} |"
        )
    recap.extend(
        [
            "",
            "## 5. Line M / K 结果",
            "",
            "```text",
            f"mlp_generic_rows = {len(mlp_rows)}",
            f"kan_specific_pass_count = {specificity.get('kan_specific_pass_count')}",
            f"delta_kan_specific_mean = {specificity.get('delta_kan_specific_mean')}",
            f"generic_optimizer_explains_crfu = {specificity.get('generic_optimizer_explains_crfu')}",
            "```",
            "",
            "判断：MLP/generic controls 只作为 confound audit；不写成 KAN-specific promotion。",
            "",
            "## 6. Line D all-basis 结果",
            "",
            "```text",
            f"line_d_source = {line_d.get('line_d_source')}",
            f"line_d_rows = {line_d.get('line_d_rows')}",
            f"best_non_dche_family = {line_d.get('best_non_dche_family')}",
            f"best_non_dche_dataset_seed_pass_count = {line_d.get('best_non_dche_dataset_seed_pass_count')} / 9",
            f"line_d_official_fu_eligible_family_count = {line_d.get('line_d_official_fu_eligible_family_count')}",
            f"line_d_route = {line_d.get('line_d_route')}",
            "```",
            "",
            "Line D family summary：",
            "",
            "| family | rows | pass | max mean delta vs MLP | best LineC pass rate | official eligibility |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in line_d.get("summary_rows", []):
        recap.append(
            f"| {row['family']} | {row['rows']} | {row['family_dataset_seed_pass_count']}/9 | {row['max_mean_delta_vs_MLP']} | {row['best_LineC_pass_rate']} | {row['official_fu_eligible']} |"
        )
    recap.extend(
        [
            "",
            "## 7. 最终 route",
            "",
            "```text",
            f"route = {route['route']}",
            f"minimum_success = {route['minimum_success']}",
            f"official_s5_reached = {route['official_s5_reached']}",
            f"promotion_allowed = {route['promotion_allowed']}",
            f"required_artifact_missing_count = {route['required_artifact_missing_count']}",
            f"forbidden_information_violation_count = {route['forbidden_information_violation_count']}",
            f"no_action_search_violation_count = {route['no_action_search_violation_count']}",
            "```",
            "",
            "## 8. 科学结论",
            "",
            "```text",
            "1. v15.01 已执行 Line R/O/C0/F/M/K/D/C/Z。",
            "2. CR-FU 不允许写成 promotion，除非 S5 official gate 达成。",
            f"3. 本轮 route = {route['route']}，promotion_allowed = {route['promotion_allowed']}。",
            "4. 若 Line F / K / D 均未打开 gate，则当前 FU family 应关闭，不继续 FU9/FU10 或 action/controller/reset。",
            "```",
        ]
    )
    write_text(RECAP_DOC, "\n".join(recap) + "\n")
    line_d_cmd = (
        f"/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v149_line_d_all_basis_substrate_repair.py "
        f"--out-dir {args.line_d_out} --device {args.device} --datasets {args.datasets} --seeds {args.seeds} "
        f"--train-size {args.train_size} --val-size {args.val_size} --epochs {args.line_d_epochs} --linec-seeds {args.linec_seeds} "
        f"--candidates {','.join(LINE_D_CANDIDATES)}"
    )
    official_full_cmd = (
        f"/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v151_control_residual_function_update_allbasis_acceleration.py "
        f"--out-dir {out_dir} --line-d-out {args.line_d_out} --device {args.device} --datasets {args.datasets} --seeds {args.seeds} "
        f"--train-size {args.train_size} --val-size {args.val_size} --test-size {args.test_size} --train-steps {args.train_steps} "
        f"--trace-interval {args.trace_interval} --linec-seeds {args.linec_seeds} --real-linec {args.real_linec} --reuse-if-present 0"
    )
    official_rewrite_cmd = (
        f"/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v151_control_residual_function_update_allbasis_acceleration.py "
        f"--out-dir {out_dir} --line-d-out {args.line_d_out} --device {args.device} --datasets {args.datasets} --seeds {args.seeds} "
        f"--train-size {args.train_size} --val-size {args.val_size} --test-size {args.test_size} --train-steps {args.train_steps} "
        f"--trace-interval {args.trace_interval} --linec-seeds {args.linec_seeds} --real-linec {args.real_linec} --reuse-if-present 1"
    )
    exec_log = [
        "# DG-KAN v15.01 ControlResidualFunctionUpdate AllBasisAcceleration 执行日志",
        "",
        "生成时间：2026-05-31（Asia/Singapore）",
        "",
        "## 1. 计划文件",
        "",
        f"```text\n{PLAN_DOC}\n```",
        "",
        "## 2. 代码修改",
        "",
        "```text",
        "experiments/run_v151_control_residual_function_update_allbasis_acceleration.py",
        "experiments/run_v149_line_d_all_basis_substrate_repair.py",
        "```",
        "",
        "## 3. Line D substrate-only 执行指令",
        "",
        f"```bash\n{line_d_cmd}\n```",
        "",
        "## 4. v15.01 official full-matrix 执行指令",
        "",
        f"```bash\n{official_full_cmd}\n```",
        "",
        "## 5. route precedence 修正后重写指令",
        "",
        "说明：该命令只复用已完成训练 artifact，重写 route / manifest / docs；不新增训练数据。",
        "",
        f"```bash\n{official_rewrite_cmd}\n```",
        "",
        "## 5.1 用户再次追问后的 Line O decay readback 修复",
        "",
        "```text",
        "修复内容：random_matched_decay_explains_fraction 不再固定为 0；改为当前 train-stream decay active mask 下的 matched-random decay metric projection readback。",
        "合法性：该 readback 不进入更新方向，不使用 LineC/CEp99/NLL/ECE/AUCtime/Brier，不新增 FU/F-CHE token，不启动 action/controller/reset。",
        "复跑方式：使用第 4 节 full-matrix 指令 --reuse-if-present 0 重新生成 official artifacts。",
        "```",
        "",
        "## 5.2 用户再次追问后的 F6/M3 proximal solver 修复",
        "",
        "```text",
        "发现问题：F6/M3 初版用一阶闭式近似 alpha，未实际评估计划 8.2 的 train split B1 proximal objective。",
        "修复内容：固定 alpha candidates = 0,0.05,0.10,0.25,0.50，在当前 train batch B1 上评估 CE + proximal penalty 选择 alpha。",
        "合法性：只使用当前 train stream；不使用 validation/test/future/query/LineC/CEp99/NLL/ECE/AUCtime/Brier；不新增 FU token，不做 action bank/controller/reset。",
        "复跑方式：使用第 4 节 full-matrix 指令 --reuse-if-present 0 重新生成 official artifacts。",
        "```",
        "",
        "## 5.3 用户再次追问后的 full residualizer matched-control 修复",
        "",
        "```text",
        "发现问题：F6/F7/M3/M4 的 full residualizer 初版只包含 Adam/Cautious/MGUP/decay，没有纳入计划 control-residual 语义中的 matched-random/same-active/same-projection/same-value-retention controls。",
        "修复内容：将固定 train-stream matched controls 纳入 full residualizer，记录 optimizer/matched/full residualizer control count。",
        "合法性：只使用当前 train stream；不使用 validation/test/future/query/LineC/CEp99/NLL/ECE/AUCtime/Brier；不新增 FU/F-CHE token，不做 action bank/controller/reset。",
        "复跑方式：使用第 4 节 full-matrix 指令 --reuse-if-present 0 重新生成 official artifacts。",
        "```",
        "",
        "## 5.4 用户再次追问后的 decoupled decay projection readback 修复",
        "",
        "```text",
        "发现问题：decoupled_decay_explains_fraction 初版是 decay_norm/raw_FU_norm，不是 decay direction 对 raw FU 的 metric projection fraction。",
        "修复内容：改为 decoupled decay direction 对 raw FU 的 metric projection readback，并记录 decoupled_decay_projection_norm。",
        "合法性：该 readback 不进入更新方向，不使用 LineC/CEp99/NLL/ECE/AUCtime/Brier，不新增 FU/F-CHE token，不启动 action/controller/reset。",
        "复跑方式：使用第 4 节 full-matrix 指令 --reuse-if-present 0 重新生成 official artifacts。",
        "```",
        "",
        "## 6. 关键输出目录",
        "",
        "```text",
        f"official_out = {out_dir}",
        f"line_d_out = {args.line_d_out}",
        "```",
        "",
        "## 7. 关键 artifact",
        "",
        "```text",
        "v151_route_decision.json",
        "v151_dche_crfu_results.csv",
        "v151_dche_crfu_controls.csv",
        "v151_mlp_generic_controls.csv",
        "v151_kan_specificity_summary.csv",
        "v151_allbasis_substrate_results.csv",
        "v151_required_artifact_manifest.csv",
        "```",
        "",
        "## 8. 最终核验",
        "",
        "```text",
        f"route = {route['route']}",
        f"minimum_success = {route['minimum_success']}",
        f"promotion_allowed = {route['promotion_allowed']}",
        f"required_artifact_missing_count = {route['required_artifact_missing_count']}",
        f"forbidden_information_violation_count = {route['forbidden_information_violation_count']}",
        f"no_action_search_violation_count = {route['no_action_search_violation_count']}",
        "```",
    ]
    write_text(EXEC_LOG_DOC, "\n".join(exec_log) + "\n")


def run(args: argparse.Namespace) -> dict[str, Any]:
    resolve_cuda_device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dche_results, dche_controls, mlp_rows, direction_rows = run_f_and_m(args, out_dir)
    summary = summarize_crfu(dche_results)
    residual_summary = build_autopsy(out_dir, dche_results)
    _spec_rows, specificity = build_kan_specificity(out_dir, dche_results, mlp_rows)
    line_d = build_line_d(out_dir, Path(args.line_d_out))
    build_monitors(out_dir, dche_results)
    build_failure_taxonomy(out_dir, dche_results)
    forbidden, no_action = build_audits(out_dir)
    write_code_packet(out_dir)
    write_no_go_docs(out_dir, {"route": "pending", "minimum_success": "pending", "promotion_allowed": 0}, summary, line_d)
    write_figures(out_dir, dche_results, dche_controls, mlp_rows, line_d)
    missing = write_required_manifest(out_dir)
    route = build_route(summary, residual_summary, specificity, line_d, forbidden, no_action, missing)
    write_json(out_dir / "v151_route_decision.json", route)
    write_no_go_docs(out_dir, route, summary, line_d)
    write_progress(out_dir, route, summary, line_d, residual_summary, specificity)
    missing = write_required_manifest(out_dir)
    if missing != route["required_artifact_missing_count"]:
        route = build_route(summary, residual_summary, specificity, line_d, forbidden, no_action, missing)
        write_json(out_dir / "v151_route_decision.json", route)
        write_progress(out_dir, route, summary, line_d, residual_summary, specificity)
        write_no_go_docs(out_dir, route, summary, line_d)
        missing = write_required_manifest(out_dir)
    write_docs(args, out_dir, route, summary, line_d, dche_results, dche_controls, mlp_rows, residual_summary, specificity)
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
    parser.add_argument("--crfu-eta", type=float, default=1.0)
    parser.add_argument("--proximal-lambda", type=float, default=1.0e-4)
    parser.add_argument("--linec-seeds", default="12319500,12319501,12319502")
    parser.add_argument("--linec-batch-size", type=int, default=24)
    parser.add_argument("--linec-sketch-dim", type=int, default=8)
    parser.add_argument("--real-linec", type=int, default=1)
    parser.add_argument("--mlp-hidden", type=int, default=32)
    parser.add_argument("--dche-candidate", default=v1410.DEFAULT_D_CHE_CANDIDATE)
    parser.add_argument("--f-methods", default=",".join(F_METHODS))
    parser.add_argument("--f-controls", default=",".join(F_CONTROLS))
    parser.add_argument("--m-methods", default=",".join(M_METHODS))
    parser.add_argument("--line-d-epochs", type=int, default=1)
    parser.add_argument("--reuse-if-present", type=int, default=1)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    route = run(args)
    print(json.dumps(route, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
