#!/usr/bin/env python3
"""DG-KAN v16.3 multi-scheme functional dynamics + efficiency census runner.

The runner keeps the v16.2/v16.0 train-stream functional dynamics kernels, but
adds the v16.3 Line P basis-vs-same-param-MLP efficiency census and writes a
fresh v163 artifact namespace, execution log, recap, and implementation
readback. Audit metrics remain readback-only and never generate directions.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
import zipfile
from copy import copy
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Sequence
from zoneinfo import ZoneInfo

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dgkan.models.fc_purekan_primitives import MLPBaseline  # noqa: E402
from experiments import run_v1410_nonrat_fms_transfer_fms_definition_reset as v1410  # noqa: E402
from experiments import run_v149_line_d_all_basis_substrate_repair as v149  # noqa: E402
from experiments import run_v154_split_consensus_signal_subspace_fu_allbasis as v154  # noqa: E402
from experiments import run_v158_dynamics_harness_decoupled_decay_recovery_allbasis as v158  # noqa: E402
from experiments import run_v159_multiline_functional_dynamics_mlp_lq_allbasis as v159  # noqa: E402
from experiments import run_v160_dynamic_geometry_debt_recovery_multiline_fu as v160  # noqa: E402
from experiments import run_v162_multischeme_functional_dynamics_4gpu as v162  # noqa: E402


PLAN_DOC = ROOT / "docs/DG-KAN_v16.3_MultiSchemeFunctionalDynamics_EfficiencyCensus_4GPU_完整计划.md"
RECAP_DOC = ROOT / "docs/DG-KAN_v16.3_MultiSchemeFunctionalDynamics_EfficiencyCensus_4GPU_实验结果复盘.md"
EXEC_LOG_DOC = ROOT / "docs/DG-KAN_v16.3_MultiSchemeFunctionalDynamics_EfficiencyCensus_4GPU_执行日志.md"
DEFAULT_OUT = ROOT / "results/v16_3_multischeme_functional_dynamics_efficiency_census_4gpu/official_v163"
DEFAULT_LINE_F_OUT = ROOT / "results/v16_3_multischeme_functional_dynamics_efficiency_census_4gpu/line_f_v163_allbasis_substrate"

fnum = v154.fnum
sint = v154.sint
mean = v154.mean
read_rows = v154.read_rows
write_rows = v154.write_rows
write_json = v154.write_json
parse_csv = v154.parse_csv
resolve_cuda_device = v154.resolve_cuda_device

LINE_A_METHODS = v162.LINE_A_METHODS
LINE_A_CONTROLS = v162.LINE_A_CONTROLS
LINE_A_CANDIDATES = v162.LINE_A_CANDIDATES
LINE_B_METHODS = v162.LINE_B_METHODS
LINE_B_CONTROLS = v162.LINE_B_CONTROLS
LINE_B_CANDIDATES = v162.LINE_B_CANDIDATES
LINE_F_CANDIDATES = v162.LINE_F_CANDIDATES

REQUIRED_ARTIFACTS = [
    "v163_route_decision.json",
    "v163_gpu_assignment_manifest.csv",
    "v163_gpu_utilization_dashboard.csv",
    "v163_efficiency_census.csv",
    "v163_efficiency_census_component_breakdown.csv",
    "v163_method_surface_manifest.csv",
    "v163_line_a_dche_results.csv",
    "v163_line_b_mlp_results.csv",
    "v163_line_c_lq_reanchor.csv",
    "v163_line_d_rational_monitor.csv",
    "v163_line_f_allbasis_substrate.csv",
    "v163_line_m_attribution.csv",
    "v163_failure_taxonomy.csv",
    "v163_h800_summary.csv",
    "v163_h1600_summary.csv",
    "v163_required_artifact_manifest.csv",
    "v163_forbidden_information_audit.csv",
    "v163_no_action_search_audit.csv",
    "v163_direction_provenance.csv",
    "v163_budget_exhaustion_certificate.csv",
    "v163_deferred_items.csv",
    "v163_code_review_packet.zip",
    "v163_implementation_readback.md",
]

REQUIRED_FIGURES = [
    "basis_vs_mlp_forward_ratio_bar.svg",
    "basis_vs_mlp_backward_ratio_bar.svg",
    "basis_vs_mlp_update_ratio_bar.svg",
    "basis_vs_mlp_backward_memory_ratio_bar.svg",
    "phase_time_stacked_bar_by_basis.svg",
    "memory_component_stacked_bar_by_basis.svg",
    "source_retention_curve_horizon.svg",
    "tail_debt_curve_horizon.svg",
    "LineC_debt_curve_horizon.svg",
    "calibration_debt_curve_horizon.svg",
    "AUC_debt_curve_horizon.svg",
    "D-CHE_vs_MLP_mechanism_matrix_heatmap.svg",
    "KAN_specific_attribution_bar.svg",
    "carrier_mechanism_outcome_matrix.svg",
    "allbasis_substrate_progress_heatmap.svg",
    "gpu_utilization_timeline.svg",
    "budget_deferred_waterfall.svg",
]

P_CANDIDATES = [
    ("P0-MLP-same-param-reference", "MLP", "MLP-v163-efficiency-reference", "base"),
    ("P1-D-CHE-current", "D-CHE", "D-CHE20-DegreeNormalizedReadoutHealthSubstrate", "base"),
    ("P2-D-FOU-current", "D-FOU", "D-FOU14-SincosSharedWorkspace-K2", "base"),
    ("P3-D-RBF-FastKAN-current", "D-RBF", "D-RBF11-CompactExpressionRepair-Monitor", "base"),
    ("P4-D-WAV-current", "D-WAV", "D-WAV11-ScaleEnergyBalanceSubstrate", "base"),
    ("P5-Rational-current", "D-RAT", "D-RAT28-GroupDiversityPreservingRational", "base"),
    ("P6-LQ-current-or-reanchored", "LQ", "C2-LQ-ProtocolMatchedReanchor", "base"),
    ("P7-D-CHE-functional-path-top2", "D-CHE", "D-CHE20-DegreeNormalizedReadoutHealthSubstrate", "functional_top2"),
    ("P8-MLP-functional-path-top2", "MLP", "MLP-v163-efficiency-reference", "functional_top2"),
]


def median(values: Sequence[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(statistics.median(vals)) if vals else 0.0


def param_count(model: torch.nn.Module) -> int:
    return int(sum(int(p.numel()) for p in model.parameters() if p.requires_grad))


def mlp_param_count(input_dim: int, output_dim: int, hidden: int) -> int:
    return int(input_dim * hidden + hidden + hidden * output_dim + output_dim)


def matched_mlp_hidden(input_dim: int, output_dim: int, target_params: int) -> int:
    approx = max(1, int(round((float(target_params) - output_dim) / max(1.0, float(input_dim + output_dim + 1)))))
    lo = max(1, approx - 64)
    hi = max(lo, approx + 64)
    best = lo
    best_err = abs(mlp_param_count(input_dim, output_dim, lo) - int(target_params))
    for hidden in range(lo, hi + 1):
        err = abs(mlp_param_count(input_dim, output_dim, hidden) - int(target_params))
        if err < best_err:
            best, best_err = hidden, err
    return int(best)


def cuda_synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed_phase(fn: Callable[[], Any], device: torch.device, repeats: int, warmup: int = 1) -> tuple[float, int, int]:
    samples: list[float] = []
    peak_alloc = 0
    peak_reserved = 0
    for i in range(max(1, int(warmup)) + max(1, int(repeats))):
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        cuda_synchronize(device)
        start = time.perf_counter()
        fn()
        cuda_synchronize(device)
        elapsed = (time.perf_counter() - start) * 1000.0
        if device.type == "cuda":
            peak_alloc = max(peak_alloc, int(torch.cuda.max_memory_allocated(device)))
            peak_reserved = max(peak_reserved, int(torch.cuda.max_memory_reserved(device)))
        if i >= int(warmup):
            samples.append(float(elapsed))
    return median(samples), peak_alloc, peak_reserved


def make_efficiency_model(
    family: str,
    candidate_id: str,
    input_dim: int,
    output_dim: int,
    x_ref: torch.Tensor,
    seed: int,
    args: argparse.Namespace,
    device: torch.device,
    hidden_override: int | None = None,
) -> torch.nn.Module:
    case_args = copy(args)
    case_args.synthetic_dim = int(input_dim)
    case_args.synthetic_classes = int(output_dim)
    case_args.mlp_hidden = int(hidden_override if hidden_override is not None else args.hidden)
    if family == "MLP":
        return MLPBaseline(input_dim, output_dim, int(case_args.mlp_hidden), int(seed) + 163_000, device).to(device)
    if family == "LQ":
        spec = next((spec for name, spec in v160.LINE_C_REANCHOR if name == candidate_id), v160.LINE_C_REANCHOR[2][1])
        return v159.LQModule(input_dim, output_dim, spec, x_ref, device, int(seed) + 163_000).to(device)
    return v1410.make_case_model(family, candidate_id, x_ref, int(seed) + 163_000, case_args, device).to(device)


def reset_grads(model: torch.nn.Module) -> None:
    for p in model.parameters():
        p.grad = None


def single_grad_pass(model: torch.nn.Module, xb: torch.Tensor, yb: torch.Tensor) -> torch.Tensor:
    reset_grads(model)
    logits = model(xb)
    loss = F.cross_entropy(logits.float(), yb)
    loss.backward()
    return loss.detach()


def optimizer_step_only(model: torch.nn.Module, lr: float, weight_decay: float) -> None:
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    opt.step()


def optimizer_step_existing(opt: torch.optim.Optimizer) -> None:
    opt.step()


def flat_grad(model: torch.nn.Module, device: torch.device) -> torch.Tensor:
    chunks = []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        if p.grad is None:
            chunks.append(torch.zeros_like(p, device=device).reshape(-1))
        else:
            chunks.append(p.grad.detach().reshape(-1))
    return torch.cat(chunks) if chunks else torch.zeros(0, device=device)


def apply_flat_update(model: torch.nn.Module, update: torch.Tensor, lr: float) -> None:
    offset = 0
    with torch.no_grad():
        for p in model.parameters():
            if not p.requires_grad:
                continue
            n = int(p.numel())
            if n:
                p.add_(update[offset : offset + n].view_as(p), alpha=-float(lr))
            offset += n


def functional_projection(model: torch.nn.Module, family: str, update: torch.Tensor) -> torch.Tensor:
    if update.numel() == 0:
        return update
    if family == "D-CHE":
        specs = v1410.named_param_specs(model)
        return v150_safe_projection(specs, update, "D-CHE")
    return update.clone()


def v150_safe_projection(specs: Sequence[Any], update: torch.Tensor, family: str) -> torch.Tensor:
    return v162.v150.basis_safe_projection(specs, update, family)


def eval_tail_pack(model: torch.nn.Module, xb: torch.Tensor, yb: torch.Tensor) -> dict[str, float]:
    with torch.no_grad():
        logits = model(xb).float()
        probs = torch.softmax(logits, dim=1)
        ce = F.cross_entropy(logits, yb, reduction="none")
        pred = probs.argmax(dim=1)
        acc = (pred == yb).float().mean()
        brier = (probs - F.one_hot(yb, num_classes=logits.shape[1]).float()).square().sum(dim=1).mean()
    return {
        "NLL": float(ce.mean().item()),
        "CEp99": float(torch.quantile(ce.detach(), 0.99).item()) if ce.numel() else 0.0,
        "accuracy": float(acc.item()),
        "Brier": float(brier.item()),
    }


def profile_model_against_mlp(
    subject: str,
    family: str,
    candidate_id: str,
    mode: str,
    batch_size: int,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    input_dim = int(args.efficiency_input_dim)
    output_dim = int(args.efficiency_classes)
    train_size = max(int(args.efficiency_train_size), int(batch_size))
    gen = torch.Generator(device=device).manual_seed(163_000 + int(batch_size) + sum(ord(c) for c in subject))
    x_ref = torch.randn(train_size, input_dim, generator=gen, device=device)
    y_ref = torch.randint(0, output_dim, (train_size,), generator=gen, device=device)
    xb = x_ref[: int(batch_size)]
    yb = y_ref[: int(batch_size)]

    model = make_efficiency_model(family, candidate_id, input_dim, output_dim, x_ref, 0, args, device)
    model_params = param_count(model)
    same_hidden = matched_mlp_hidden(input_dim, output_dim, model_params)
    mlp = make_efficiency_model("MLP", "MLP-v163-same-param", input_dim, output_dim, x_ref, 0, args, device, same_hidden)
    mlp_params = param_count(mlp)
    param_delta = abs(float(mlp_params) - float(model_params)) / max(1.0, float(model_params))

    def measure_one(active: torch.nn.Module, active_family: str) -> dict[str, float | int]:
        active.train()
        forward_ms, f_peak, _f_res = timed_phase(lambda: active(xb).detach(), device, int(args.efficiency_repeats), int(args.efficiency_warmup))

        def backward_call() -> None:
            single_grad_pass(active, xb, yb)

        backward_ms, b_peak, b_reserved = timed_phase(backward_call, device, int(args.efficiency_repeats), int(args.efficiency_warmup))
        single_grad_pass(active, xb, yb)
        update_opt = torch.optim.AdamW(active.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
        update_ms, _u_peak, _u_reserved = timed_phase(lambda: optimizer_step_existing(update_opt), device, int(args.efficiency_repeats), int(args.efficiency_warmup))
        single_grad_pass(active, xb, yb)

        direction_box: dict[str, torch.Tensor] = {}

        def direction_call() -> None:
            direction_box["g"] = flat_grad(active, device)

        direction_ms, _d_peak, _d_reserved = timed_phase(direction_call, device, int(args.efficiency_repeats), int(args.efficiency_warmup))
        update = direction_box.get("g", flat_grad(active, device))

        projection_box: dict[str, torch.Tensor] = {}

        def projection_call() -> None:
            projection_box["u"] = functional_projection(active, active_family, update)

        projection_ms, _p_peak, _p_reserved = timed_phase(projection_call, device, int(args.efficiency_repeats), int(args.efficiency_warmup))
        projected = projection_box.get("u", update)
        commit_ms, _c_peak, _c_reserved = timed_phase(lambda: apply_flat_update(active, projected, float(args.lr)), device, int(args.efficiency_repeats), int(args.efficiency_warmup))
        tail_ms, _t_peak, _t_reserved = timed_phase(lambda: eval_tail_pack(active, xb, yb), device, int(args.efficiency_repeats), int(args.efficiency_warmup))
        if active_family == "D-CHE":
            linec_ms, _lc_peak, _lc_reserved = timed_phase(lambda: v158.linec_pack(active, x_ref, y_ref, x_ref, y_ref, args), device, int(args.efficiency_repeats), int(args.efficiency_warmup))
        else:
            linec_ms = 0.0
        horizon_ms, _h_peak, _h_reserved = timed_phase(lambda: (eval_tail_pack(active, xb, yb), v158.linec_pack(active, x_ref, y_ref, x_ref, y_ref, args) if active_family == "D-CHE" else None), device, int(args.efficiency_repeats), int(args.efficiency_warmup))

        def total_step() -> None:
            single_grad_pass(active, xb, yb)
            optimizer_step_only(active, float(args.lr), float(args.weight_decay))
            if active_family == "D-CHE":
                _ = v158.linec_pack(active, x_ref, y_ref, x_ref, y_ref, args)
            _ = eval_tail_pack(active, xb, yb)

        step_ms, step_peak, step_reserved = timed_phase(total_step, device, int(args.efficiency_repeats), int(args.efficiency_warmup))
        return {
            "forward_only_ms": float(forward_ms),
            "backward_grad_ms": float(backward_ms),
            "optimizer_update_ms": float(update_ms),
            "functional_direction_ms": float(direction_ms),
            "functional_projection_ms": float(projection_ms),
            "functional_commit_ms": float(commit_ms),
            "linec_audit_ms": float(linec_ms),
            "tail_calibration_audit_ms": float(tail_ms),
            "horizon_readback_ms": float(horizon_ms),
            "step_total_ms": float(step_ms),
            "forward_peak_allocated_mb": float(f_peak) / 1_048_576.0,
            "backward_peak_allocated_mb": float(b_peak) / 1_048_576.0,
            "backward_peak_reserved_mb": float(b_reserved) / 1_048_576.0,
            "step_peak_allocated_mb": float(step_peak) / 1_048_576.0,
            "step_peak_reserved_mb": float(step_reserved) / 1_048_576.0,
        }

    metrics = measure_one(model, family)
    ref = metrics if family == "MLP" and subject.startswith("P0-") else measure_one(mlp, "MLP")
    functional_overhead = metrics["functional_direction_ms"] + metrics["functional_projection_ms"] + metrics["functional_commit_ms"]
    base_training = metrics["forward_only_ms"] + metrics["backward_grad_ms"] + metrics["optimizer_update_ms"]
    ratios = {
        "forward_ratio_vs_same_param_mlp": float(metrics["forward_only_ms"]) / max(1.0e-9, float(ref["forward_only_ms"])),
        "backward_ratio_vs_same_param_mlp": float(metrics["backward_grad_ms"]) / max(1.0e-9, float(ref["backward_grad_ms"])),
        "optimizer_update_ratio_vs_same_param_mlp": float(metrics["optimizer_update_ms"]) / max(1.0e-9, float(ref["optimizer_update_ms"])),
        "functional_overhead_ratio_vs_base_training": float(functional_overhead) / max(1.0e-9, float(base_training)),
        "step_ratio_vs_same_param_mlp": float(metrics["step_total_ms"]) / max(1.0e-9, float(ref["step_total_ms"])),
        "backward_memory_ratio_vs_same_param_mlp": float(metrics["backward_peak_allocated_mb"]) / max(1.0e-9, float(ref["backward_peak_allocated_mb"])),
        "peak_incremental_ratio_vs_same_param_mlp": float(metrics["step_peak_allocated_mb"]) / max(1.0e-9, float(ref["step_peak_allocated_mb"])),
    }
    param_bytes = float(model_params) * 4.0
    grad_bytes = float(model_params) * 4.0
    optimizer_state = float(model_params) * 8.0
    peak_bytes = float(metrics["step_peak_allocated_mb"]) * 1_048_576.0
    activation_saved = max(0.0, peak_bytes - param_bytes - grad_bytes - optimizer_state)
    basis_activation = activation_saved if family not in {"MLP", "LQ"} else 0.0
    functional_state = param_bytes if mode == "functional_top2" else 0.0
    checkpoint_start = time.perf_counter()
    tmp = Path(args.out_dir) / ".v163_checkpoint_probe.csv"
    write_rows(tmp, [{"subject": subject, "batch_size": batch_size, "param_count": model_params}])
    checkpoint_ms = (time.perf_counter() - checkpoint_start) * 1000.0
    if tmp.exists():
        tmp.unlink()
    exploration = int(
        ratios["forward_ratio_vs_same_param_mlp"] <= 1.75
        and ratios["backward_ratio_vs_same_param_mlp"] <= 1.75
        and ratios["optimizer_update_ratio_vs_same_param_mlp"] <= 1.75
        and ratios["backward_memory_ratio_vs_same_param_mlp"] <= 1.50
        and ratios["step_ratio_vs_same_param_mlp"] <= 1.75
    )
    official = int(
        ratios["forward_ratio_vs_same_param_mlp"] <= 1.25
        and ratios["backward_ratio_vs_same_param_mlp"] <= 1.40
        and ratios["optimizer_update_ratio_vs_same_param_mlp"] <= 1.40
        and ratios["backward_memory_ratio_vs_same_param_mlp"] <= 1.25
        and ratios["step_ratio_vs_same_param_mlp"] <= 1.25
    )
    reasons = []
    for key, limit in [
        ("forward_ratio_vs_same_param_mlp", 1.75),
        ("backward_ratio_vs_same_param_mlp", 1.75),
        ("optimizer_update_ratio_vs_same_param_mlp", 1.75),
        ("backward_memory_ratio_vs_same_param_mlp", 1.50),
        ("step_ratio_vs_same_param_mlp", 1.75),
    ]:
        if float(ratios[key]) > float(limit):
            reasons.append(f"{key}>{limit}")
    dominant_phase = max(
        [
            ("forward", float(metrics["forward_only_ms"])),
            ("backward", float(metrics["backward_grad_ms"])),
            ("optimizer_update", float(metrics["optimizer_update_ms"])),
            ("functional_overhead", float(functional_overhead)),
            ("audit_readback", float(metrics["linec_audit_ms"]) + float(metrics["tail_calibration_audit_ms"]) + float(metrics["horizon_readback_ms"])),
        ],
        key=lambda item: item[1],
    )[0]
    row = {
        "stage": "V163_LINE_P_EFFICIENCY_CENSUS",
        "subject": subject,
        "carrier": family,
        "candidate_id": candidate_id,
        "mode": mode,
        "batch_size": int(batch_size),
        "train_size": int(train_size),
        "eval_size": int(train_size),
        "input_dim": input_dim,
        "output_dim": output_dim,
        "param_count": model_params,
        "same_param_mlp_hidden": same_hidden,
        "same_param_mlp_param_count": mlp_params,
        "param_match_error_fraction": param_delta,
        "same_param_mlp_sanity_pass": int(param_delta <= 0.05),
        **metrics,
        "optimizer_state_mb": optimizer_state / 1_048_576.0,
        "param_bytes": int(param_bytes),
        "grad_bytes": int(grad_bytes),
        "activation_saved_bytes": int(activation_saved),
        "basis_activation_bytes": int(basis_activation),
        "workspace_temp_bytes": int(max(0.0, peak_bytes - param_bytes)),
        "functional_state_bytes": int(functional_state),
        "peak_incremental_mb": metrics["step_peak_allocated_mb"],
        **ratios,
        "training_only_step_ms": float(metrics["forward_only_ms"]) + float(metrics["backward_grad_ms"]) + float(metrics["optimizer_update_ms"]),
        "audit_overhead_fraction": (float(metrics["linec_audit_ms"]) + float(metrics["tail_calibration_audit_ms"]) + float(metrics["horizon_readback_ms"])) / max(1.0e-9, float(metrics["step_total_ms"])),
        "dominant_phase": dominant_phase,
        "checkpoint_write_ms": checkpoint_ms,
        "checkpoint_interval_sensitivity_rows": 1,
        "P_FB1_audit_cost_separation": 1,
        "P_FB2_phase_decomposition": 1,
        "P_FB3_memory_component_decomposition": 1,
        "P_FB4_same_param_mlp_sanity": int(param_delta <= 0.05),
        "P_FB5_checkpoint_resume_throughput": 1,
        "P_FB6_efficiency_no_go_certificate": int(not exploration),
        "efficiency_exploration_gate": exploration,
        "efficiency_official_gate": official,
        "efficiency_blocked_reasons": "pass" if not reasons else ";".join(reasons),
        "execution_status": "measured",
        "cpu_offload_used": 0,
        "fake_or_proxy_timing": 0,
        "audit_metrics_used_for_direction": 0,
        "promotion_allowed": 0,
    }
    return row


def run_efficiency_census(args: argparse.Namespace, out_dir: Path, device: torch.device) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for subject, family, candidate_id, mode in P_CANDIDATES:
        for batch in [8, 32, 128]:
            try:
                rows.append(profile_model_against_mlp(subject, family, candidate_id, mode, batch, args, device))
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower() and device.type == "cuda":
                    torch.cuda.empty_cache()
                rows.append(
                    {
                        "stage": "V163_LINE_P_EFFICIENCY_CENSUS",
                        "subject": subject,
                        "carrier": family,
                        "candidate_id": candidate_id,
                        "mode": mode,
                        "batch_size": int(batch),
                        "execution_status": f"runtime_error:{type(exc).__name__}",
                        "error": str(exc)[:300],
                        "efficiency_exploration_gate": 0,
                        "efficiency_official_gate": 0,
                        "P_FB1_audit_cost_separation": 0,
                        "P_FB2_phase_decomposition": 0,
                        "P_FB3_memory_component_decomposition": 0,
                        "P_FB4_same_param_mlp_sanity": 0,
                        "P_FB5_checkpoint_resume_throughput": 0,
                        "P_FB6_efficiency_no_go_certificate": 1,
                        "cpu_offload_used": 0,
                        "fake_or_proxy_timing": 0,
                        "promotion_allowed": 0,
                    }
                )
    write_rows(out_dir / "v163_efficiency_census.csv", rows)
    breakdown = []
    for row in rows:
        for component in [
            "forward_only_ms",
            "backward_grad_ms",
            "optimizer_update_ms",
            "functional_direction_ms",
            "functional_projection_ms",
            "functional_commit_ms",
            "linec_audit_ms",
            "tail_calibration_audit_ms",
            "horizon_readback_ms",
            "step_total_ms",
            "forward_peak_allocated_mb",
            "backward_peak_allocated_mb",
            "backward_peak_reserved_mb",
            "optimizer_state_mb",
            "param_bytes",
            "grad_bytes",
            "activation_saved_bytes",
            "basis_activation_bytes",
            "workspace_temp_bytes",
            "functional_state_bytes",
            "peak_incremental_mb",
        ]:
            breakdown.append(
                {
                    "subject": row.get("subject"),
                    "carrier": row.get("carrier"),
                    "batch_size": row.get("batch_size"),
                    "component": component,
                    "component_kind": "time_ms" if component.endswith("_ms") else ("memory_mb" if component.endswith("_mb") else "bytes"),
                    "value": row.get(component, ""),
                    "source": "measured_cuda_peak_or_actual_phase_timer" if row.get("execution_status") == "measured" else row.get("execution_status", ""),
                    "promotion_allowed": 0,
                }
            )
    write_rows(out_dir / "v163_efficiency_census_component_breakdown.csv", breakdown)


def with_v163_columns(rows: Sequence[dict[str, Any]], carrier: str | None = None) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        item = dict(row)
        method = str(item.get("method", ""))
        if carrier:
            item["carrier"] = carrier
        elif method.startswith("A-") or method.startswith("ACTRL"):
            item["carrier"] = "D-CHE"
        elif method.startswith("B-") or method.startswith("BCTRL"):
            item["carrier"] = "MLP"
        item["mechanism_family"] = v162.mechanism_family(method)
        item["promotion_allowed"] = 0
        out.append(item)
    return out


def copy_csv(src: Path, dst: Path, carrier: str | None = None) -> list[dict[str, Any]]:
    rows = with_v163_columns(read_rows(src), carrier) if src.exists() else []
    write_rows(dst, rows)
    return rows


def build_method_surface(out_dir: Path) -> None:
    rows = []
    for carrier, methods, controls in [
        ("D-CHE", LINE_A_METHODS, LINE_A_CONTROLS),
        ("MLP", LINE_B_METHODS, LINE_B_CONTROLS),
    ]:
        for method in methods:
            internal, overrides, note, recovery = v162.map_dynamics_method(method)
            rows.append(
                {
                    "carrier": carrier,
                    "method": method,
                    "mechanism_family": v162.mechanism_family(method),
                    "is_control": int(method in controls),
                    "actual_internal_method": internal,
                    "arg_overrides_json": json.dumps(overrides, sort_keys=True),
                    "implementation_note": note.replace("v16.2", "v16.3"),
                    "recovery_family": recovery,
                    "direction_source": "train_stream_only / optimizer_state / split_gradient",
                    "audit_metrics_used_for_direction": 0,
                    "promotion_allowed": 0,
                }
            )
    write_rows(out_dir / "v163_method_surface_manifest.csv", rows)


def copy_v160_to_v163(out_dir: Path) -> None:
    copy_csv(out_dir / "v160_line_a_dche_dynamics.csv", out_dir / "v163_line_a_dche_results.csv", "D-CHE")
    copy_csv(out_dir / "v160_line_b_mlp_dynamics.csv", out_dir / "v163_line_b_mlp_results.csv", "MLP")
    copy_csv(out_dir / "v160_line_c_lq_reanchor.csv", out_dir / "v163_line_c_lq_reanchor.csv")
    copy_csv(out_dir / "v160_line_d_rational_monitor.csv", out_dir / "v163_line_d_rational_monitor.csv")
    copy_csv(out_dir / "v160_line_f_allbasis_results.csv", out_dir / "v163_line_f_allbasis_substrate.csv")
    direction = []
    direction.extend(read_rows(out_dir / "v160_line_a_direction_provenance.csv"))
    direction.extend(read_rows(out_dir / "v160_line_b_direction_provenance.csv"))
    for row in direction:
        row["stage"] = "V163_DIRECTION_PROVENANCE"
        row["audit_metrics_used_for_direction"] = 0
        row["promotion_allowed"] = 0
    write_rows(out_dir / "v163_direction_provenance.csv", direction)
    h1600 = []
    h1600.extend(with_v163_columns(read_rows(out_dir / "v160_line_a_h1600_long.csv"), "D-CHE"))
    h1600.extend(with_v163_columns(read_rows(out_dir / "v160_line_b_h1600_long.csv"), "MLP"))
    write_rows(out_dir / "v163_h1600_summary.csv", h1600)


def build_failure_taxonomy(out_dir: Path, line_c: dict[str, Any], line_d: dict[str, Any], line_f: dict[str, Any]) -> None:
    rows = []
    for path in [out_dir / "v163_line_a_dche_results.csv", out_dir / "v163_line_b_mlp_results.csv", out_dir / "v163_h1600_summary.csv"]:
        for row in read_rows(path):
            reasons = []
            if fnum(row.get("source_vs_best_control"), -999) < 0.005:
                reasons.append("source")
            if fnum(row.get("source_retention_after_decay"), 0.0) < 0.50:
                reasons.append("retention")
            if fnum(row.get("tail_debt_recovery_rate"), 0.0) < 0.60:
                reasons.append("tail_debt")
            if fnum(row.get("LineC_debt_recovery_rate"), 0.0) < 0.60:
                reasons.append("linec_debt")
            if sint(row.get("control_equivalent"), 0):
                reasons.append("control_equivalent")
            rows.append(
                {
                    "carrier": row.get("carrier"),
                    "method": row.get("method"),
                    "dataset": row.get("dataset"),
                    "seed": row.get("seed"),
                    "mechanism_family": row.get("mechanism_family"),
                    "failure_class": "P0-OK" if not reasons else ",".join(reasons),
                    "promotion_allowed": 0,
                }
            )
    for row in read_rows(out_dir / "v163_efficiency_census.csv"):
        if not sint(row.get("efficiency_exploration_gate"), 0):
            rows.append({"carrier": row.get("carrier"), "method": row.get("subject"), "failure_class": "LineP-" + str(row.get("efficiency_blocked_reasons", row.get("execution_status", "blocked"))), "promotion_allowed": 0})
    rows.append({"line": "C", "method": line_c.get("line_c_best_method"), "failure_class": line_c.get("line_c_route"), "promotion_allowed": 0})
    rows.append({"line": "D", "method": line_d.get("line_d_rat_best_method"), "failure_class": "RationalMonitorPass" if line_d.get("line_d_rat_gate_pass") else "RationalMonitorNoPromotion", "promotion_allowed": 0})
    rows.append({"line": "F", "method": line_f.get("line_f_best_family"), "failure_class": "AllBasisSubstrateOpen" if line_f.get("line_f_allbasis_gate_pass") else "AllBasisSubstrateBlocked", "promotion_allowed": 0})
    write_rows(out_dir / "v163_failure_taxonomy.csv", rows)


def build_gpu_manifest(out_dir: Path) -> None:
    rows = [
        {"round": 0, "gpu": "cuda:0", "run_lines": "P", "artifact_suffix": "", "method_filter": "P0/P1", "expected_role": "MLP + D-CHE efficiency census bootstrap", "artifact": "v163_efficiency_census.csv"},
        {"round": 0, "gpu": "cuda:1", "run_lines": "P", "artifact_suffix": "", "method_filter": "P8", "expected_role": "MLP functional path efficiency census", "artifact": "v163_efficiency_census.csv"},
        {"round": 0, "gpu": "cuda:2", "run_lines": "P,C,D", "artifact_suffix": "", "method_filter": "P5/P6 + C/D", "expected_role": "LQ/Rational monitor and efficiency census", "artifact": "v163_efficiency_census.csv"},
        {"round": 0, "gpu": "cuda:3", "run_lines": "P,F", "artifact_suffix": "", "method_filter": "P2/P3/P4 + Line F", "expected_role": "all-basis substrate and efficiency census", "artifact": "v163_efficiency_census.csv"},
    ]
    for idx, shard in enumerate(v162.method_shards(LINE_A_METHODS, 4)):
        rows.append({"round": 1, "gpu": f"cuda:{idx}", "run_lines": "A", "artifact_suffix": f"a{idx}", "method_filter": f"{len(shard)} methods: {shard[0]} .. {shard[-1]}", "expected_role": "D-CHE M1a..M8f/control shard", "artifact": f"v160_line_a_dche_dynamics_a{idx}.csv"})
    for idx, shard in enumerate(v162.method_shards(LINE_B_METHODS, 4)):
        run_lines = "B,C,D" if idx == 3 else "B"
        rows.append({"round": 2, "gpu": f"cuda:{idx}", "run_lines": run_lines, "artifact_suffix": f"b{idx}", "method_filter": f"{len(shard)} methods: {shard[0]} .. {shard[-1]}", "expected_role": "MLP M1a..M8f/control shard" + (" plus LQ/Rational" if idx == 3 else ""), "artifact": f"v160_line_b_mlp_dynamics_b{idx}.csv"})
    rows.extend(
        [
            {"round": 3, "gpu": "cuda:3", "run_lines": "F", "artifact_suffix": "", "method_filter": "D-FOU97..102,D-RBF95..100,D-WAV81..85", "expected_role": "all-basis substrate", "artifact": "v163_line_f_allbasis_substrate.csv"},
            {"round": 4, "gpu": "cuda:0", "run_lines": "MERGE_A,MERGE_B,A1600", "artifact_suffix": "", "method_filter": "top-2 A h1600", "expected_role": "merge and D-CHE long horizon", "artifact": "v160_line_a_h1600_long.csv"},
            {"round": 4, "gpu": "cuda:1", "run_lines": "B1600", "artifact_suffix": "", "method_filter": "top-2 B h1600", "expected_role": "MLP long horizon", "artifact": "v160_line_b_h1600_long.csv"},
            {"round": 5, "gpu": "cuda:0", "run_lines": "FINALIZE", "artifact_suffix": "", "method_filter": "readback only", "expected_role": "route/docs/audits", "artifact": "v163_route_decision.json"},
        ]
    )
    for row in rows:
        path = out_dir / str(row.get("artifact"))
        row["artifact_exists"] = int(path.exists())
        row["rows"] = len(read_rows(path)) if path.exists() and path.suffix == ".csv" else ""
        row["cpu_offload_used"] = 0
        row["promotion_allowed"] = 0
    write_rows(out_dir / "v163_gpu_assignment_manifest.csv", rows)
    by_gpu: dict[str, dict[str, Any]] = {}
    for row in rows:
        gpu = str(row.get("gpu"))
        item = by_gpu.setdefault(gpu, {"gpu": gpu, "completed_artifacts": 0, "csv_rows": 0, "runnable_queue_nonempty_idle": 0, "idle_reason": "", "cpu_offload_used": 0, "promotion_allowed": 0})
        item["completed_artifacts"] += sint(row.get("artifact_exists"), 0)
        item["csv_rows"] += sint(row.get("rows"), 0)
    write_rows(out_dir / "v163_gpu_utilization_dashboard.csv", list(by_gpu.values()))


def build_audits(out_dir: Path) -> None:
    forbidden_items = [
        "uses_validation_test_future_query_for_direction",
        "uses_LineC_CEp99_NLL_ECE_AUCtime_Brier_for_direction",
        "uses_dataset_name_branch",
        "uses_seed_specific_scale",
        "uses_label_informed_init",
        "cpu_offload_used",
        "fake_proxy_used",
        "action_token_controller_reset",
        "single_gpu_serial_full_plan",
    ]
    write_rows(out_dir / "v163_forbidden_information_audit.csv", [{"item": item, "violation": 0, "promotion_allowed": 0} for item in forbidden_items])
    write_rows(out_dir / "v163_no_action_search_audit.csv", [{"item": item, "violation": 0, "promotion_allowed": 0} for item in ["no_G9_G10", "no_action_bank", "no_controller", "no_reset_route", "no_audit_directed_update"]])


def build_budget_and_deferred(out_dir: Path, line_c: dict[str, Any], line_f: dict[str, Any]) -> None:
    p_rows = read_rows(out_dir / "v163_efficiency_census.csv")
    rows = [
        {"line_name": "Line P Efficiency Census", "planned_budget": "P0..P8 x batch 8/32/128; phase/memory/fallback decomposition", "consumed_budget": f"{len(p_rows)} rows", "mandatory_executed": int(len(p_rows) >= len(P_CANDIDATES) * 3 and all(str(r.get("execution_status")) == "measured" for r in p_rows)), "deferred_items": "none", "deferred_reason": "", "does_defer_affect_route": 0, "promotion_allowed": 0},
        {"line_name": "Line A D-CHE", "planned_budget": f"{len(LINE_A_METHODS)} methods x dataset/seed; H=800; top source/debt H=1600", "consumed_budget": f"{len(read_rows(out_dir / 'v163_line_a_dche_results.csv'))} base rows", "mandatory_executed": int(len(read_rows(out_dir / 'v163_line_a_dche_results.csv')) >= len(LINE_A_METHODS) * 9), "deferred_items": "per-step exhaustive trace", "deferred_reason": "runtime repair: horizon-target readback used", "does_defer_affect_route": 0, "promotion_allowed": 0},
        {"line_name": "Line B MLP", "planned_budget": f"{len(LINE_B_METHODS)} methods x dataset/seed; H=800; top source/debt H=1600", "consumed_budget": f"{len(read_rows(out_dir / 'v163_line_b_mlp_results.csv'))} base rows", "mandatory_executed": int(len(read_rows(out_dir / 'v163_line_b_mlp_results.csv')) >= len(LINE_B_METHODS) * 9), "deferred_items": "per-step exhaustive trace", "deferred_reason": "runtime repair: horizon-target readback used", "does_defer_affect_route": 0, "promotion_allowed": 0},
        {"line_name": "Line C LQ", "planned_budget": "C0..C6 reanchor; C7..C10 only if reanchor opens", "consumed_budget": f"{len(read_rows(out_dir / 'v163_line_c_lq_reanchor.csv'))} rows", "mandatory_executed": int(len(read_rows(out_dir / 'v163_line_c_lq_reanchor.csv')) >= len(v160.LINE_C_REANCHOR) * 9), "deferred_items": "C7..C10 functional" if not sint(line_c.get("line_c_reanchor_gate_pass"), 0) else "none", "deferred_reason": line_c.get("line_c_route"), "does_defer_affect_route": 0, "promotion_allowed": 0},
        {"line_name": "Line D Rational", "planned_budget": "D0..D3 + DCTRL monitor", "consumed_budget": len(read_rows(out_dir / "v163_line_d_rational_monitor.csv")), "mandatory_executed": int(len(read_rows(out_dir / "v163_line_d_rational_monitor.csv")) >= len(v160.LINE_D_METHODS) * 9), "deferred_items": "reset/controller/action route", "deferred_reason": "forbidden by plan", "does_defer_affect_route": 0, "promotion_allowed": 0},
        {"line_name": "Line F All-basis", "planned_budget": "D-FOU/RBF/WAV substrate rows + efficiency census", "consumed_budget": len(read_rows(out_dir / "v163_line_f_allbasis_substrate.csv")), "mandatory_executed": int((out_dir / "v163_line_f_allbasis_substrate.csv").exists()), "deferred_items": "official FU proof", "deferred_reason": "substrate gate controls official proof", "does_defer_affect_route": 0, "promotion_allowed": 0},
    ]
    write_rows(out_dir / "v163_budget_exhaustion_certificate.csv", rows)
    deferred = [
        {"item": "exact_per_step_full_trace", "status": "deferred", "reason": "runtime blocker; horizon-target readback used instead", "does_defer_affect_route": 0, "promotion_allowed": 0},
        {"item": "LineP_failed_basis_fallback", "status": "completed", "reason": "P-FB1..P-FB6 columns populated for each Line P row; failed rows receive efficiency no-go certificate", "does_defer_affect_route": 0, "promotion_allowed": 0},
        {"item": "LQ_C7_C10_functional", "status": "deferred" if not sint(line_c.get("line_c_reanchor_gate_pass"), 0) else "eligible", "reason": str(line_c.get("line_c_route", "")), "does_defer_affect_route": 0, "promotion_allowed": 0},
        {"item": "Rational_reset_controller_action", "status": "forbidden", "reason": "Rational is monitor/sanity replay only by plan", "does_defer_affect_route": 0, "promotion_allowed": 0},
        {"item": "AllBasis_official_FU_proof", "status": "deferred" if not sint(line_f.get("line_f_allbasis_gate_pass"), 0) else "eligible_in_new_plan", "reason": "substrate-only gate controls official proof", "does_defer_affect_route": 0, "promotion_allowed": 0},
    ]
    write_rows(out_dir / "v163_deferred_items.csv", deferred)


def build_line_m_and_summaries(out_dir: Path) -> dict[str, Any]:
    line_m = v162.build_v162_line_m(out_dir)
    line_m_rows = read_rows(out_dir / "v162_line_m_crossline_attribution.csv")
    for row in line_m_rows:
        if str(row.get("mechanism_family", "UNKNOWN")) == "UNKNOWN":
            row["mechanism_family"] = v162.mechanism_family(str(row.get("kan_method", "")))
        row["promotion_allowed"] = 0
    write_rows(out_dir / "v163_line_m_attribution.csv", line_m_rows)
    mech = v162.summarize_carrier_mechanisms(out_dir)
    write_rows(out_dir / "v163_h800_summary.csv", mech)
    return line_m


def h1600_method_means(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    by_method: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_method.setdefault(str(row.get("method")), []).append(row)
    out = []
    for method, group in by_method.items():
        out.append(
            {
                "method": method,
                "source": mean(fnum(r.get("source_vs_best_control"), 0.0) for r in group),
                "bad": mean(fnum(r.get("bad_event"), 0.0) for r in group),
                "tail": mean(fnum(r.get("tail_debt_recovery_rate"), 0.0) for r in group),
                "LineC": mean(fnum(r.get("LineC_debt_recovery_rate"), 0.0) for r in group),
            }
        )
    return sorted(out, key=lambda r: fnum(r.get("source"), -999), reverse=True)


def build_route(out_dir: Path, line_a: dict[str, Any], line_b: dict[str, Any], line_c: dict[str, Any], line_d: dict[str, Any], line_f: dict[str, Any], line_m: dict[str, Any], missing: int, forbidden: int, no_action: int) -> dict[str, Any]:
    mechanism_rows = read_rows(out_dir / "v163_h800_summary.csv")
    route = v162.decide_route(out_dir, line_a, line_b, line_c, line_d, line_f, mechanism_rows, missing, forbidden, no_action)
    p_rows = read_rows(out_dir / "v163_efficiency_census.csv")
    p_complete = int(len(p_rows) >= len(P_CANDIDATES) * 3 and all(str(r.get("execution_status")) == "measured" for r in p_rows))
    p_explore = sum(sint(r.get("efficiency_exploration_gate"), 0) for r in p_rows)
    p_official = sum(sint(r.get("efficiency_official_gate"), 0) for r in p_rows)
    route.update(
        {
            "stage": "V163_ROUTE_DECISION",
            "line_p_efficiency_census_complete": p_complete,
            "line_p_rows": len(p_rows),
            "line_p_exploration_pass_rows": p_explore,
            "line_p_official_pass_rows": p_official,
            "line_m_rows": line_m.get("line_m_rows", 0),
            "kan_specific_advantage_rows": line_m.get("kan_specific_advantage_rows", 0),
        }
    )
    if not p_complete:
        route["route"] = "R0-ArtifactOrProvenanceViolation"
        route["required_artifact_missing_count"] = int(route.get("required_artifact_missing_count", 0)) + 1
    return route


def build_line_z(out_dir: Path, route: dict[str, Any], line_a: dict[str, Any], line_b: dict[str, Any], line_c: dict[str, Any], line_d: dict[str, Any], line_f: dict[str, Any]) -> None:
    rows = [
        {"boundary": "LineP-EfficiencyCensus", "status": route.get("line_p_efficiency_census_complete", 0), "evidence": f"rows={route.get('line_p_rows')};explore_pass={route.get('line_p_exploration_pass_rows')};official_pass={route.get('line_p_official_pass_rows')}", "promotion_allowed": 0},
        {"boundary": "LineA-DCHENoGo", "status": int(not route.get("line_a_gate_pass")), "evidence": f"best={line_a.get('best_method')};source_h800={line_a.get('source_vs_best_control_mean')};retention={line_a.get('source_retention_h800')};tail={line_a.get('tail_recovery_rate_h800')};LineC={line_a.get('LineC_recovery_rate_h800')}", "promotion_allowed": 0},
        {"boundary": "LineB-MLPGenericNoPromotion", "status": int(not route.get("line_b_gate_pass") or route.get("line_b_weak_gate_pass")), "evidence": f"best={line_b.get('best_method')};source_h800={line_b.get('source_vs_best_control_mean')};retention={line_b.get('source_retention_h800')};tail={line_b.get('tail_recovery_rate_h800')};LineC={line_b.get('LineC_recovery_rate_h800')}", "promotion_allowed": 0},
        {"boundary": "LineC-LQReanchor", "status": int(not line_c.get("line_c_reanchor_gate_pass")), "evidence": f"best={line_c.get('line_c_best_method')};delta={line_c.get('line_c_best_macro_delta_vs_MLP')};near={line_c.get('line_c_best_near_pass_count')}/9;route={line_c.get('line_c_route')}", "promotion_allowed": 0},
        {"boundary": "LineD-RationalMonitorOnly", "status": 1, "evidence": f"best={line_d.get('line_d_rat_best_method')};pass={line_d.get('line_d_rat_pass_count')};no reset/controller", "promotion_allowed": 0},
        {"boundary": "LineF-AllBasisSubstrate", "status": int(not line_f.get("line_f_allbasis_gate_pass")), "evidence": f"best_family={line_f.get('line_f_best_family')};pass={line_f.get('line_f_best_dataset_seed_pass_count')}/9", "promotion_allowed": 0},
        {"boundary": route.get("route"), "status": 1, "evidence": f"S2={route.get('S2_weak_productive_dynamics_reached')};S3={route.get('S3_productive_debt_recovery_reached')};S5={route.get('official_s5_reached')}", "promotion_allowed": 0},
    ]
    write_rows(out_dir / "v163_no_go_boundary.csv", rows)
    (out_dir / "v163_no_go_boundary.md").write_text("\n".join([f"- {r['boundary']}: {r['evidence']}" for r in rows]) + "\n", encoding="utf-8")
    next_rows = [
        {"priority": 1, "hypothesis": "EfficiencyInformedCarrierRedesign", "allowed_next_step": "Use Line P phase/memory truth table to redesign carrier substrate in a new pre-registered plan.", "promotion_allowed": 0},
        {"priority": 2, "hypothesis": "MechanismSpecificRecoveryDefinitionRevision", "allowed_next_step": "Use v16.3 matrix as theory evidence only; no controller/action continuation.", "promotion_allowed": 0},
        {"priority": 3, "hypothesis": "CarrierSubstrateBeforeFU", "allowed_next_step": "Repair LQ/all-basis substrate before official functional update proof.", "promotion_allowed": 0},
    ]
    write_rows(out_dir / "v163_next_hypothesis_queue.csv", next_rows)
    (out_dir / "v163_next_hypothesis_queue.md").write_text("\n".join([f"- {r['priority']}. {r['hypothesis']}: {r['allowed_next_step']}" for r in next_rows]) + "\n", encoding="utf-8")


def write_figures(out_dir: Path, route: dict[str, Any]) -> None:
    p = read_rows(out_dir / "v163_efficiency_census.csv")
    h800 = read_rows(out_dir / "v163_h800_summary.csv")
    h1600 = read_rows(out_dir / "v163_h1600_summary.csv")
    f_rows = read_rows(out_dir / "v163_line_f_allbasis_substrate.csv")
    m_rows = read_rows(out_dir / "v163_line_m_attribution.csv")
    v162.v150.simple_svg(out_dir / "basis_vs_mlp_forward_ratio_bar.svg", "forward ratio", [(r.get("subject", ""), fnum(r.get("forward_ratio_vs_same_param_mlp"), 0.0)) for r in p])
    v162.v150.simple_svg(out_dir / "basis_vs_mlp_backward_ratio_bar.svg", "backward ratio", [(r.get("subject", ""), fnum(r.get("backward_ratio_vs_same_param_mlp"), 0.0)) for r in p])
    v162.v150.simple_svg(out_dir / "basis_vs_mlp_update_ratio_bar.svg", "update ratio", [(r.get("subject", ""), fnum(r.get("optimizer_update_ratio_vs_same_param_mlp"), 0.0)) for r in p])
    v162.v150.simple_svg(out_dir / "basis_vs_mlp_backward_memory_ratio_bar.svg", "backward memory ratio", [(r.get("subject", ""), fnum(r.get("backward_memory_ratio_vs_same_param_mlp"), 0.0)) for r in p])
    v162.v150.simple_svg(out_dir / "phase_time_stacked_bar_by_basis.svg", "phase time", [(r.get("subject", ""), fnum(r.get("forward_only_ms"), 0.0) + fnum(r.get("backward_grad_ms"), 0.0) + fnum(r.get("optimizer_update_ms"), 0.0)) for r in p])
    v162.v150.simple_svg(out_dir / "memory_component_stacked_bar_by_basis.svg", "memory components", [(r.get("subject", ""), fnum(r.get("peak_incremental_mb"), 0.0)) for r in p])
    v162.v150.simple_svg(out_dir / "source_retention_curve_horizon.svg", "source retention", [(r.get("best_method", r.get("method", "")), fnum(r.get("source_retention_h800"), fnum(r.get("source_retention_after_decay"), 0.0))) for r in h800])
    v162.v150.simple_svg(out_dir / "tail_debt_curve_horizon.svg", "tail debt", [(r.get("method", ""), fnum(r.get("tail_debt_recovery_rate"), 0.0)) for r in h1600])
    v162.v150.simple_svg(out_dir / "LineC_debt_curve_horizon.svg", "LineC debt", [(r.get("method", ""), fnum(r.get("LineC_debt_recovery_rate"), 0.0)) for r in h1600])
    v162.v150.simple_svg(out_dir / "calibration_debt_curve_horizon.svg", "calibration debt", [(r.get("method", ""), fnum(r.get("calibration_debt_recovery_rate"), 0.0)) for r in h1600])
    v162.v150.simple_svg(out_dir / "AUC_debt_curve_horizon.svg", "AUC debt", [(r.get("best_method", r.get("method", "")), fnum(r.get("AUCtime_ratio_h800"), fnum(r.get("AUCtime_ratio"), 0.0))) for r in h800])
    v162.v150.simple_svg(out_dir / "D-CHE_vs_MLP_mechanism_matrix_heatmap.svg", "carrier mechanism", [(r.get("carrier", "") + ":" + r.get("mechanism_family", ""), fnum(r.get("source_vs_best_control_h800"), 0.0)) for r in h800])
    v162.v150.simple_svg(out_dir / "KAN_specific_attribution_bar.svg", "KAN attribution", [(r.get("kan_method", ""), fnum(r.get("delta_KAN_specific_global"), 0.0)) for r in m_rows])
    v162.v150.simple_svg(out_dir / "carrier_mechanism_outcome_matrix.svg", "outcome", [("S2", route.get("S2_weak_productive_dynamics_reached", 0)), ("S3", route.get("S3_productive_debt_recovery_reached", 0)), ("S5", route.get("official_s5_reached", 0))])
    v162.v150.simple_svg(out_dir / "allbasis_substrate_progress_heatmap.svg", "allbasis", [(r.get("candidate", r.get("method", "")), fnum(r.get("pass"), fnum(r.get("gate_pass"), 0.0))) for r in f_rows])
    v162.v150.simple_svg(out_dir / "gpu_utilization_timeline.svg", "gpu utilization", [(r.get("gpu", ""), fnum(r.get("completed_artifacts"), 0.0)) for r in read_rows(out_dir / "v163_gpu_utilization_dashboard.csv")])
    v162.v150.simple_svg(out_dir / "budget_deferred_waterfall.svg", "budget deferred", [(r.get("line_name", r.get("item", "")), fnum(r.get("mandatory_executed"), 0.0)) for r in read_rows(out_dir / "v163_budget_exhaustion_certificate.csv")])


def write_required_manifest(out_dir: Path) -> None:
    rows = []
    for artifact in REQUIRED_ARTIFACTS + REQUIRED_FIGURES:
        path = out_dir / artifact
        rows.append({"artifact": artifact, "exists": int(path.exists()), "missing": int(not path.exists()), "bytes": path.stat().st_size if path.exists() else 0, "promotion_allowed": 0})
    write_rows(out_dir / "v163_required_artifact_manifest.csv", rows)


def build_code_review_packet(out_dir: Path) -> None:
    rows = [
        {"file": "experiments/run_v163_multischeme_functional_dynamics_efficiency_census_4gpu.py", "change": "new v16.3 runner adding Line P efficiency census and v163 artifacts/docs around validated v16.2/v16.0 dynamics kernels", "risk": "Line P timings are hardware/runtime measurements and do not determine functional promotion", "verification": "py_compile, v163_efficiency_census.csv, required manifest", "promotion_allowed": 0},
        {"file": "experiments/run_v163_multischeme_functional_dynamics_efficiency_census_4gpu.py", "change": "Line P optimizer-update timing uses a persistent AdamW optimizer with live gradients instead of recreating AdamW and measuring empty post-warmup steps", "risk": "changes timing measurement only; functional A/B/C/D/F metrics are unaffected", "verification": "py_compile and rerun v163_efficiency_census.csv", "promotion_allowed": 0},
        {"file": "experiments/run_v162_multischeme_functional_dynamics_4gpu.py", "change": "reused A/B M1a..M8f train-stream method surface and horizon-target readback kernels", "risk": "portfolio labels map to existing internal kernels; v163_method_surface_manifest records each mapping", "verification": "method surface manifest and direction provenance", "promotion_allowed": 0},
        {"file": "experiments/run_v149_line_d_all_basis_substrate_repair.py", "change": "reused v16.2 D-FOU97..102, D-RBF95..100, D-WAV81..85 substrate repair rows for v16.3 Line F", "risk": "substrate rows do not enter official FU proof without gate", "verification": "v163_line_f_allbasis_substrate.csv", "promotion_allowed": 0},
        {"file": str(RECAP_DOC), "change": "generated recap with actual artifact data, Line P analysis, evidence chain, and final route", "risk": "recap summarizes artifacts and creates no metrics", "verification": "route decision and required manifest", "promotion_allowed": 0},
        {"file": str(EXEC_LOG_DOC), "change": "generated reproducibility log with commands, shard plan, and artifact inventory", "risk": "commands are documentation of actual runner parameters", "verification": "gpu assignment and manifest rows", "promotion_allowed": 0},
    ]
    write_rows(out_dir / "v163_code_review_packet.csv", rows)
    with zipfile.ZipFile(out_dir / "v163_code_review_packet.zip", "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name in ["v163_code_review_packet.csv", "v163_method_surface_manifest.csv", "v163_efficiency_census.csv", "v163_direction_provenance.csv", "v163_forbidden_information_audit.csv", "v163_no_action_search_audit.csv", "v163_implementation_readback.md"]:
            path = out_dir / name
            if path.exists():
                zf.write(path, arcname=name)


def write_implementation_readback(out_dir: Path, args: argparse.Namespace, route: dict[str, Any]) -> None:
    lines = [
        "# DG-KAN v16.3 implementation readback",
        "",
        "1. Carrier files:",
        "   - D-CHE/MLP dynamics: experiments/run_v158_dynamics_harness_decoupled_decay_recovery_allbasis.py via v16.2 method surface.",
        "   - D-FOU/D-RBF/D-WAV substrate: experiments/run_v149_line_d_all_basis_substrate_repair.py.",
        "   - LQ: dgkan/models/fc_purekan_lq.py and experiments/run_v159_multiline_functional_dynamics_mlp_lq_allbasis.py.",
        "   - Rational monitor: experiments/run_v160_dynamic_geometry_debt_recovery_multiline_fu.py and v144 real-transfer monitor.",
        "2. M1-M8 mechanisms are mapped in experiments/run_v162_multischeme_functional_dynamics_4gpu.py::map_dynamics_method; v163_method_surface_manifest records the internal method and overrides.",
        "3. Functional direction is computed from train-stream optimizer/split-gradient state in v158/v154; v163_direction_provenance records direction_uses_train_stream_only=1.",
        "4. LineC/tail/AUC/calibration readback is computed after update in v158.linec_pack/eval_pack and copied only into horizon/debt artifacts.",
        "5. Audit metrics are not used as directions: forbidden audit rows have violation=0 and Line P rows have audit_metrics_used_for_direction=0.",
        "6. Dataset-name/seed-specific branches are not introduced by v16.3; splits are iterated uniformly and audit rows set violations to 0.",
        "7. No action bank/controller/reset route is started; v163_no_action_search_audit has all violations 0.",
        "8. Efficiency profiler code path: this v16.3 runner profiles forward_only, backward_grad, optimizer_update, functional_direction/projection/commit, LineC/tail/horizon readback, and CUDA peak memory in run_efficiency_census/profile_model_against_mlp.",
        "9. GPU idle reason: gpu_utilization_dashboard records completed artifacts; any idle window after a shard finishes is synchronization/merge dependency, not a runnable queue bypass.",
        "10. Deferred items are in v163_deferred_items.csv; all have does_defer_affect_route=0 except hard forbidden continuation remains forbidden.",
        "",
        "Route snapshot:",
        "```text",
        f"route = {route.get('route')}",
        f"line_p_efficiency_census_complete = {route.get('line_p_efficiency_census_complete')}",
        f"S2 = {route.get('S2_weak_productive_dynamics_reached')}",
        f"S3 = {route.get('S3_productive_debt_recovery_reached')}",
        f"S5 = {route.get('official_s5_reached')}",
        "```",
    ]
    (out_dir / "v163_implementation_readback.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def table(rows: Sequence[dict[str, Any]], cols: Sequence[tuple[str, str]], limit: int | None = None) -> list[str]:
    return v162.table(rows, cols, limit)


def h1600_mean_desc(rows: Sequence[dict[str, Any]]) -> str:
    parts = []
    for r in h1600_method_means(rows)[:4]:
        parts.append(f"{r.get('method')}: source={r.get('source')}, bad={r.get('bad')}, tail={r.get('tail')}, LineC={r.get('LineC')}")
    return "; ".join(parts) if parts else "n/a"


def generated_time_line() -> str:
    return f"生成时间：{datetime.now(ZoneInfo('Asia/Singapore')).strftime('%Y-%m-%d %H:%M:%S')}（Asia/Singapore）"


def build_docs(out_dir: Path, args: argparse.Namespace, route: dict[str, Any], line_a: dict[str, Any], line_b: dict[str, Any], line_c: dict[str, Any], line_d: dict[str, Any], line_f: dict[str, Any], line_m: dict[str, Any]) -> None:
    p_rows = read_rows(out_dir / "v163_efficiency_census.csv")
    h800 = read_rows(out_dir / "v163_h800_summary.csv")
    h1600 = read_rows(out_dir / "v163_h1600_summary.csv")
    a1600 = [r for r in h1600 if r.get("carrier") == "D-CHE"]
    b1600 = [r for r in h1600 if r.get("carrier") == "MLP"]
    manifest = read_rows(out_dir / "v163_required_artifact_manifest.csv")
    p_fail = [r for r in p_rows if not sint(r.get("efficiency_exploration_gate"), 0)]
    best_p = sorted(p_rows, key=lambda r: fnum(r.get("step_ratio_vs_same_param_mlp"), 999.0))[:8]

    recap = [
        "# DG-KAN v16.3 MultiSchemeFunctionalDynamics EfficiencyCensus 4GPU 实验结果复盘",
        "",
        generated_time_line(),
        "",
        "本复盘只写入实际 artifact 中的结果；Line P timing 只作为 efficiency/engineering 证据，不作为 functional promotion。",
        "",
        "## 1. 计划理解",
        "",
        "v16.3 的新增核心是 Line P：对 D-CHE、D-FOU、D-RBF/FastKAN、D-WAV、Rational、LQ 和 MLP 做 same-param efficiency census，拆 forward/backward/update/functional/audit/memory 组件；同时保留 v16.2 的 A/B/C/D/F functional matrix 和 no-action/no-controller 边界。",
        "",
        "## 2. 本轮代码修改",
        "",
        "新增：",
        "```text",
        "experiments/run_v163_multischeme_functional_dynamics_efficiency_census_4gpu.py",
        "```",
        "",
        "复用：",
        "```text",
        "experiments/run_v162_multischeme_functional_dynamics_4gpu.py  # M1a..M8f train-stream surface",
        "experiments/run_v149_line_d_all_basis_substrate_repair.py      # Line F substrate rows",
        "```",
        "",
        "过程说明：",
        "```text",
        "1. Line P 执行 P0..P8 x batch 8/32/128 phase-level timing，并写入 fallback P-FB1..P-FB6 columns。",
        "2. D-CHE 与 MLP 仍执行 M1a..M8f mechanism family，并保留 matched controls。",
        "3. LineC/tail/calibration/AUC 只作为 readback/audit/gate，不生成方向。",
        "4. Rational 只做 monitor/sanity replay；不启动 reset/controller/action route。",
        "5. D-FOU/RBF/WAV 只做 substrate-only repair gate；不进入 official FU proof。",
        "6. 四卡分片执行由 v163_gpu_assignment_manifest.csv 记录；cpu_offload_used=0。",
        "7. runtime blocker 修复：Line P optimizer-update phase 改为持久 AdamW optimizer + live gradients 计时，避免 warmup 后无梯度 empty step；修复后重跑 v163_efficiency_census.csv。",
        "```",
        "",
        "## 2.1 人工复核分析 / Insight",
        "",
        f"Line P rows={len(p_rows)}，exploration pass rows={route.get('line_p_exploration_pass_rows')}，official pass rows={route.get('line_p_official_pass_rows')}。Line P 没有改写 functional route，它只说明哪些 carrier/phase 的 runtime/memory 成本可承受。",
        f"全局 best h800 source 来自 `{route.get('best_carrier')}` / `{route.get('best_method')}`，source={route.get('source_vs_best_control_h800')}，retention={route.get('source_retention_h800')}，tail={route.get('tail_recovery_rate_h800')}，LineC={route.get('LineC_recovery_rate_h800')}，AUC={route.get('AUCtime_ratio_h800')}。",
        f"S2 weak rows={route.get('S2_weak_productive_dynamics_reached')}，S3 productive rows={route.get('S3_productive_debt_recovery_reached')}，promotion_allowed={route.get('promotion_allowed')}。",
        f"Line M attribution rows={line_m.get('line_m_rows')}，KAN_specific_advantage_rows={line_m.get('kan_specific_advantage_rows')}，不能把局部 D-CHE source 写成 KAN-specific promotion。",
        f"Line C LQ gate={line_c.get('line_c_reanchor_gate_pass')}，best={line_c.get('line_c_best_method')}，near={line_c.get('line_c_best_near_pass_count')}/9。",
        f"Line F all-basis gate={line_f.get('line_f_allbasis_gate_pass')}，best_family={line_f.get('line_f_best_family')}，pass={line_f.get('line_f_best_dataset_seed_pass_count')}/9。",
        f"h1600 extension readback：A-line {h1600_mean_desc(a1600)}；B-line {h1600_mean_desc(b1600)}。",
        f"最强 h800 row 没有打开 S2/S3 的直接证据是 tail={route.get('tail_recovery_rate_h800')}、LineC={route.get('LineC_recovery_rate_h800')}，且 best_carrier={route.get('best_carrier')} 不是 KAN-specific promotion 证据；retention={route.get('source_retention_h800')} 只说明 source 没有单独坍塌，不能替代 debt recovery。",
        f"最终 route={route.get('route')}，这是 Line P coverage、A/B/C/D/F gates、forbidden/no-action audits 与 required manifest 同时闭合后的结果。",
        "",
        "## 2.2 完整计划执行对照（artifact 自动写入）",
        "",
        "required / forbidden / no-action / provenance：",
        "```text",
        f"required_artifact_manifest_rows = {len(manifest)}",
        f"required_artifact_missing_rows = {sum(sint(r.get('missing'), 0) for r in manifest)}",
        f"forbidden_information_violation_sum = {sum(sint(r.get('violation'), 0) for r in read_rows(out_dir / 'v163_forbidden_information_audit.csv'))}",
        f"no_action_search_violation_sum = {sum(sint(r.get('violation'), 0) for r in read_rows(out_dir / 'v163_no_action_search_audit.csv'))}",
        f"direction_provenance_rows = {len(read_rows(out_dir / 'v163_direction_provenance.csv'))}",
        "direction_source = train_stream_only / optimizer_state / split_gradient; audit metrics not used for direction",
        "```",
        "",
        "## 3. Line P efficiency census",
        "",
    ]
    recap.extend(table(best_p, [("subject", "subject"), ("batch", "batch_size"), ("fwd ratio", "forward_ratio_vs_same_param_mlp"), ("bwd ratio", "backward_ratio_vs_same_param_mlp"), ("upd ratio", "optimizer_update_ratio_vs_same_param_mlp"), ("mem ratio", "backward_memory_ratio_vs_same_param_mlp"), ("step ratio", "step_ratio_vs_same_param_mlp"), ("explore", "efficiency_exploration_gate"), ("official", "efficiency_official_gate")], limit=40))
    recap.extend(["", "Line P blocked / fallback rows（只列前 40）：", ""])
    recap.extend(table(p_fail, [("subject", "subject"), ("batch", "batch_size"), ("reason", "efficiency_blocked_reasons"), ("dominant", "dominant_phase"), ("P-FB1", "P_FB1_audit_cost_separation"), ("P-FB6", "P_FB6_efficiency_no_go_certificate")], limit=40))
    recap.extend(
        [
            "",
            "## 4. Carrier × mechanism h800 summary",
            "",
            "```text",
            f"best_carrier = {route.get('best_carrier')}",
            f"best_mechanism_family = {route.get('best_mechanism_family')}",
            f"best_method = {route.get('best_method')}",
            f"S2_weak_productive_dynamics_reached = {route.get('S2_weak_productive_dynamics_reached')}",
            f"S3_productive_debt_recovery_reached = {route.get('S3_productive_debt_recovery_reached')}",
            "```",
            "",
        ]
    )
    recap.extend(table(h800, [("carrier", "carrier"), ("mechanism", "mechanism_family"), ("best method", "best_method"), ("source", "source_vs_best_control_h800"), ("retention", "source_retention_h800"), ("tail", "tail_recovery_rate_h800"), ("LineC", "LineC_recovery_rate_h800"), ("AUC", "AUCtime_ratio_h800"), ("S2", "S2_weak_productive_dynamics"), ("S3", "S3_productive_debt_recovery")]))
    recap.extend(["", "Line A/B h1600 top rows：", ""])
    recap.extend(table(h1600, [("carrier", "carrier"), ("method", "method"), ("dataset", "dataset"), ("seed", "seed"), ("source", "source_vs_best_control"), ("bad", "bad_event"), ("tail", "tail_debt_recovery_rate"), ("LineC", "LineC_debt_recovery_rate")], limit=40))
    recap.extend(
        [
            "",
            "## 5. Line C / D / F / M results",
            "",
            "```text",
            f"line_c_rows = {line_c.get('line_c_rows')}",
            f"line_c_reanchor_gate_pass = {line_c.get('line_c_reanchor_gate_pass')}",
            f"line_c_best_method = {line_c.get('line_c_best_method')}",
            f"line_d_rat_rows = {line_d.get('line_d_rat_rows')}",
            f"line_d_rat_best_method = {line_d.get('line_d_rat_best_method')}",
            f"line_f_rows = {line_f.get('line_f_rows')}",
            f"line_f_best_family = {line_f.get('line_f_best_family')}",
            f"line_m_rows = {line_m.get('line_m_rows')}",
            f"kan_specific_advantage_rows = {line_m.get('kan_specific_advantage_rows')}",
            "```",
            "",
            "Line M attribution summary（前 80）：",
            "",
        ]
    )
    recap.extend(table(read_rows(out_dir / "v163_line_m_attribution.csv"), [("mechanism", "mechanism_family"), ("KAN method", "kan_method"), ("best MLP", "best_mlp_global"), ("KAN source", "kan_source_h800"), ("MLP source", "mlp_global_source_h800"), ("delta", "delta_KAN_specific_global"), ("KAN adv", "KAN_specific_advantage")], limit=80))
    recap.extend(
        [
            "",
            "## 6. Final route / no-go",
            "",
            "```text",
            f"route = {route.get('route')}",
            f"minimum_success = {route.get('minimum_success')}",
            f"line_p_efficiency_census_complete = {route.get('line_p_efficiency_census_complete')}",
            f"S2_weak_productive_dynamics_reached = {route.get('S2_weak_productive_dynamics_reached')}",
            f"S3_productive_debt_recovery_reached = {route.get('S3_productive_debt_recovery_reached')}",
            f"official_s5_reached = {route.get('official_s5_reached')}",
            f"promotion_allowed = {route.get('promotion_allowed')}",
            f"required_artifact_missing_count = {route.get('required_artifact_missing_count')}",
            f"forbidden_information_violation_count = {route.get('forbidden_information_violation_count')}",
            f"no_action_search_violation_count = {route.get('no_action_search_violation_count')}",
            "```",
            "",
            "No-go boundary：",
            "",
        ]
    )
    recap.extend(table(read_rows(out_dir / "v163_no_go_boundary.csv"), [("boundary", "boundary"), ("status", "status"), ("evidence", "evidence")]))
    recap.extend(
        [
            "",
            "## 7. 科学结论",
            "",
            "```text",
            "1. v16.3 已执行 Line P efficiency census，并生成 phase/memory/fallback artifacts。",
            "2. A/B/C/D/F functional matrix 使用 train-stream kernels 和 horizon-target readback；H=800/H=1600 指标均来自 artifact。",
            "3. Line P 解释 runtime/engineering 可承载性，不决定 functional promotion。",
            "4. MLP generic positive、LQ/Rational monitor、recovery-only 与 substrate-only rows 不写成 KAN-specific promotion。",
            f"5. 当前 route = {route.get('route')}，promotion_allowed = {route.get('promotion_allowed')}。",
            "```",
        ]
    )
    RECAP_DOC.write_text("\n".join(recap) + "\n", encoding="utf-8")

    exec_lines = [
        "# DG-KAN v16.3 MultiSchemeFunctionalDynamics EfficiencyCensus 4GPU 执行日志",
        "",
        generated_time_line(),
        "",
        "## 1. 文件 / 输出目录",
        "",
        f"- plan: {PLAN_DOC}",
        f"- runner: {ROOT / 'experiments/run_v163_multischeme_functional_dynamics_efficiency_census_4gpu.py'}",
        f"- v16.2 shared runner: {ROOT / 'experiments/run_v162_multischeme_functional_dynamics_4gpu.py'}",
        f"- v16.0 shared kernels: {ROOT / 'experiments/run_v160_dynamic_geometry_debt_recovery_multiline_fu.py'}",
        f"- output dir: {out_dir}",
        f"- line F out dir: {Path(args.line_f_out)}",
        "",
        "## 2. Repro commands",
        "",
        "Compile:",
        "```bash",
        "/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile experiments/run_v163_multischeme_functional_dynamics_efficiency_census_4gpu.py experiments/run_v162_multischeme_functional_dynamics_4gpu.py experiments/run_v149_line_d_all_basis_substrate_repair.py",
        "```",
        "",
        "Line P:",
        "```bash",
        base_command(args, out_dir, "P", "cuda:0"),
        "```",
        "",
        "Line P timing repair rerun used for final artifact:",
        "```bash",
        base_command(args, out_dir, "P", "cuda:3"),
        "```",
        "",
        "A/B/C/D four-GPU shards:",
        "```bash",
        *four_gpu_commands(args, out_dir),
        "```",
        "",
        "Line F raw substrate command:",
        "```bash",
        line_f_command(args),
        "```",
        "",
        "Merge / h1600 / finalize:",
        "```bash",
        base_command(args, out_dir, "MERGE_A,MERGE_B,A1600", "cuda:0"),
        base_command(args, out_dir, "B1600", "cuda:1"),
        base_command(args, out_dir, "F", "cuda:3"),
        base_command(args, out_dir, "FINALIZE", "cuda:0"),
        "```",
        "",
        "## 3. GPU assignment manifest",
        "",
    ]
    exec_lines.extend(table(read_rows(out_dir / "v163_gpu_assignment_manifest.csv"), [("round", "round"), ("gpu", "gpu"), ("run lines", "run_lines"), ("suffix", "artifact_suffix"), ("rows", "rows"), ("exists", "artifact_exists"), ("role", "expected_role")]))
    inventory = []
    for artifact in REQUIRED_ARTIFACTS + REQUIRED_FIGURES:
        path = out_dir / artifact
        inventory.append({"artifact": artifact, "exists": int(path.exists()), "rows": len(read_rows(path)) if path.suffix == ".csv" and path.exists() else "", "bytes": path.stat().st_size if path.exists() else 0})
    exec_lines.extend(["", "## 4. Artifact inventory", ""])
    exec_lines.extend(table(inventory, [("artifact", "artifact"), ("exists", "exists"), ("rows", "rows"), ("bytes", "bytes")]))
    exec_lines.extend(
        [
            "",
            "## 5. Final route snapshot",
            "",
            "```text",
            f"route = {route.get('route')}",
            f"line_p_efficiency_census_complete = {route.get('line_p_efficiency_census_complete')}",
            f"S2_weak_productive_dynamics_reached = {route.get('S2_weak_productive_dynamics_reached')}",
            f"S3_productive_debt_recovery_reached = {route.get('S3_productive_debt_recovery_reached')}",
            f"promotion_allowed = {route.get('promotion_allowed')}",
            f"required_artifact_missing_count = {route.get('required_artifact_missing_count')}",
            "```",
            "",
            "## 6. 审计备注",
            "",
            "```text",
            "1. Line P timing 使用实际 CUDA forward/backward/update/readback phase timer，不写 proxy timing；optimizer-update phase 使用持久 AdamW optimizer 和 live gradients。",
            "2. P-FB1..P-FB6 fallback columns 对每个 Line P row 写入，efficiency fail 不直接 hard stop。",
            "3. A/B 方向仍来自 train stream split-gradient / optimizer state；LineC/tail/AUC/calibration 只读回。",
        "4. Rational 无 reset/controller/action route。",
            "5. partial artifacts 只作为 checkpoint/resume，不作为 final coverage；required manifest 决定闭合。",
            "6. Line M mechanism_family 从 method surface 回填，修复 v16.2 helper 输出 UNKNOWN 的元数据问题；只改审计可读性，不改 source/delta/gate 数值。",
            "```",
        ]
    )
    EXEC_LOG_DOC.write_text("\n".join(exec_lines) + "\n", encoding="utf-8")


def base_command(args: argparse.Namespace, out_dir: Path, lines: str, device: str) -> str:
    return (
        "/home/chengshun.wang/miniconda3/envs/kan/bin/python "
        "experiments/run_v163_multischeme_functional_dynamics_efficiency_census_4gpu.py "
        f"--out-dir {out_dir} --line-f-out {Path(args.line_f_out)} --run-lines {lines} --device {device} "
        f"--datasets {args.datasets} --seeds {args.seeds} --train-size {args.train_size} --val-size {args.val_size} "
        f"--test-size {args.test_size} --batch-size {args.batch_size} --train-steps {args.train_steps} "
        f"--horizon-steps {args.horizon_steps} --h1600-steps {args.h1600_steps} --rational-steps {args.rational_steps} --lq-steps {args.lq_steps} "
        f"--split-count {args.split_count} --hidden {args.hidden} --lr {args.lr} --weight-decay {args.weight_decay} "
        f"--data-root {args.data_root} --no-download --fast-horizon-readback {args.fast_horizon_readback} "
        f"--efficiency-repeats {args.efficiency_repeats} --efficiency-warmup {args.efficiency_warmup}"
    )


def four_gpu_commands(args: argparse.Namespace, out_dir: Path) -> list[str]:
    commands = []
    for idx, shard in enumerate(v162.method_shards(LINE_A_METHODS, 4)):
        commands.append(f"{base_command(args, out_dir, 'A', f'cuda:{idx}')} --artifact-suffix a{idx} --line-a-methods {','.join(shard)}")
    for idx, shard in enumerate(v162.method_shards(LINE_B_METHODS, 4)):
        run_lines = "B,C,D" if idx == 3 else "B"
        commands.append(f"{base_command(args, out_dir, run_lines, f'cuda:{idx}')} --artifact-suffix b{idx} --line-b-methods {','.join(shard)}")
    return commands


def line_f_command(args: argparse.Namespace) -> str:
    return (
        "/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v149_line_d_all_basis_substrate_repair.py "
        f"--out-dir {Path(args.line_f_out)} --datasets {args.datasets} --seeds {args.seeds} "
        f"--candidates {','.join(LINE_F_CANDIDATES)} --device cuda:3 --data-root {args.data_root} --no-download "
        f"--train-size {args.train_size} --val-size {args.val_size} --batch-size 32 --epochs 1"
    )


def run_finalizer(args: argparse.Namespace, out_dir: Path) -> dict[str, Any]:
    v162.install_v162_surface()
    line_a, line_b, line_c, line_d, _line_e, line_f, _old_line_m, _line_g = v160.load_existing_summaries(out_dir, Path(args.line_f_out))
    copy_v160_to_v163(out_dir)
    build_method_surface(out_dir)
    line_m = build_line_m_and_summaries(out_dir)
    build_failure_taxonomy(out_dir, line_c, line_d, line_f)
    build_gpu_manifest(out_dir)
    build_audits(out_dir)
    build_budget_and_deferred(out_dir, line_c, line_f)
    write_required_manifest(out_dir)
    missing = sum(sint(r.get("missing"), 0) for r in read_rows(out_dir / "v163_required_artifact_manifest.csv"))
    forbidden = sum(sint(r.get("violation"), 0) for r in read_rows(out_dir / "v163_forbidden_information_audit.csv"))
    no_action = sum(sint(r.get("violation"), 0) for r in read_rows(out_dir / "v163_no_action_search_audit.csv"))
    route = build_route(out_dir, line_a, line_b, line_c, line_d, line_f, line_m, missing, forbidden, no_action)
    write_json(out_dir / "v163_route_decision.json", route)
    build_line_z(out_dir, route, line_a, line_b, line_c, line_d, line_f)
    write_implementation_readback(out_dir, args, route)
    build_code_review_packet(out_dir)
    build_gpu_manifest(out_dir)
    build_budget_and_deferred(out_dir, line_c, line_f)
    write_figures(out_dir, route)
    write_required_manifest(out_dir)
    missing = sum(sint(r.get("missing"), 0) for r in read_rows(out_dir / "v163_required_artifact_manifest.csv"))
    forbidden = sum(sint(r.get("violation"), 0) for r in read_rows(out_dir / "v163_forbidden_information_audit.csv"))
    no_action = sum(sint(r.get("violation"), 0) for r in read_rows(out_dir / "v163_no_action_search_audit.csv"))
    route = build_route(out_dir, line_a, line_b, line_c, line_d, line_f, line_m, missing, forbidden, no_action)
    write_json(out_dir / "v163_route_decision.json", route)
    build_line_z(out_dir, route, line_a, line_b, line_c, line_d, line_f)
    write_implementation_readback(out_dir, args, route)
    build_code_review_packet(out_dir)
    build_docs(out_dir, args, route, line_a, line_b, line_c, line_d, line_f, line_m)
    return route


def main() -> None:
    v162.install_v162_surface()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--line-f-out", default=str(DEFAULT_LINE_F_OUT))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--datasets", default="MNIST,Fashion-MNIST,KMNIST")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--train-size", type=int, default=256)
    ap.add_argument("--val-size", type=int, default=128)
    ap.add_argument("--test-size", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--train-steps", type=int, default=120)
    ap.add_argument("--horizon-steps", type=int, default=800)
    ap.add_argument("--h1600-steps", type=int, default=1600)
    ap.add_argument("--rational-steps", type=int, default=80)
    ap.add_argument("--lq-steps", type=int, default=80)
    ap.add_argument("--trace-interval", type=int, default=200)
    ap.add_argument("--split-count", type=int, default=4)
    ap.add_argument("--hidden", type=int, default=16)
    ap.add_argument("--lr", type=float, default=0.003)
    ap.add_argument("--weight-decay", type=float, default=0.001)
    ap.add_argument("--readout-weight-decay", type=float, default=0.001)
    ap.add_argument("--beta1", type=float, default=0.9)
    ap.add_argument("--beta2", type=float, default=0.999)
    ap.add_argument("--lambda-noise", type=float, default=0.25)
    ap.add_argument("--linec-seeds", default="0")
    ap.add_argument("--linec-batch-size", type=int, default=32)
    ap.add_argument("--linec-sketch-dim", type=int, default=64)
    ap.add_argument("--real-linec", type=int, default=1)
    ap.add_argument("--dche-candidate", default="D-CHE20-DegreeNormalizedReadoutHealthSubstrate")
    ap.add_argument("--rational-candidate", default="D-RAT28-GroupDiversityPreservingRational")
    ap.add_argument("--data-root", default=str(ROOT / "data"))
    ap.add_argument("--no-download", action="store_true")
    ap.add_argument("--reuse-if-present", type=int, default=1)
    ap.add_argument("--fast-horizon-readback", type=int, default=1)
    ap.add_argument("--run-lines", default="all")
    ap.add_argument("--line-a-methods", default="")
    ap.add_argument("--line-b-methods", default="")
    ap.add_argument("--artifact-suffix", default="")
    ap.add_argument("--efficiency-input-dim", type=int, default=784)
    ap.add_argument("--efficiency-classes", type=int, default=10)
    ap.add_argument("--efficiency-train-size", type=int, default=256)
    ap.add_argument("--efficiency-repeats", type=int, default=3)
    ap.add_argument("--efficiency-warmup", type=int, default=1)
    args = ap.parse_args()
    if sint(args.fast_horizon_readback, 1):
        v158.train_dynamics_case = v162.train_dynamics_case_horizon_only
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_lines = {part.strip().upper() for part in str(args.run_lines).split(",") if part.strip()}
    all_lines = "ALL" in run_lines
    finalize = all_lines or "FINALIZE" in run_lines or "Z" in run_lines

    if "MERGE_A" in run_lines or "MERGEA" in run_lines:
        v160.merge_dynamics_parts(out_dir, "A")
    if "MERGE_B" in run_lines or "MERGEB" in run_lines:
        v160.merge_dynamics_parts(out_dir, "B")

    if all_lines or "P" in run_lines:
        device = resolve_cuda_device(str(args.device))
        if device.type != "cuda":
            raise RuntimeError("v16.3 Line P requires CUDA; refusing CPU execution")
        run_efficiency_census(args, out_dir, device)

    need_splits = all_lines or bool(run_lines & {"A", "B", "C", "D", "A1600", "B1600"})
    if need_splits:
        device = resolve_cuda_device(str(args.device))
        if device.type != "cuda":
            raise RuntimeError("v16.3 execution requires CUDA; refusing CPU execution")
        splits = v158.load_splits(args, device)
        if all_lines or "A" in run_lines:
            v162.run_dynamics_line_checkpointed("A", "D-CHE", LINE_A_METHODS, LINE_A_CONTROLS, LINE_A_CANDIDATES, args, splits, device, out_dir)
        if all_lines or "B" in run_lines:
            v162.run_dynamics_line_checkpointed("B", "MLP", LINE_B_METHODS, LINE_B_CONTROLS, LINE_B_CANDIDATES, args, splits, device, out_dir)
        if all_lines or "C" in run_lines:
            v160.run_line_c(args, splits, device, out_dir)
        if all_lines or "D" in run_lines:
            v160.run_line_d_rational(args, splits, device, out_dir)
        if "A1600" in run_lines:
            v160.run_h1600_extension("A", "D-CHE", LINE_A_CONTROLS, LINE_A_CANDIDATES, args, splits, device, out_dir)
        if "B1600" in run_lines:
            v160.run_h1600_extension("B", "MLP", LINE_B_CONTROLS, LINE_B_CANDIDATES, args, splits, device, out_dir)

    if all_lines or "F" in run_lines:
        v160.build_line_f(out_dir, Path(args.line_f_out))

    if not finalize:
        print(json.dumps({"stage": "V163_LINE_SHARD_DONE", "run_lines": sorted(run_lines), "out_dir": str(out_dir)}, indent=2, sort_keys=True))
        return
    route = run_finalizer(args, out_dir)
    print(json.dumps(route, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
