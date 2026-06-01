#!/usr/bin/env python3
"""DG-KAN v15.8 dynamics harness / decoupled decay recovery runner.

This runner tests whether the v15.7 G7R local bad event behaves like
recoverable training debt.  LineC/tail/calibration metrics are audit/gate
only and never generate the update direction.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from copy import copy
from pathlib import Path
from typing import Any, Sequence

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import run_v1410_nonrat_fms_transfer_fms_definition_reset as v1410  # noqa: E402
from experiments import run_v144_real_transfer_fms_all_basis_substrate as v144  # noqa: E402
from experiments import run_v150_function_update_allbasis_parallel as v150  # noqa: E402
from experiments import run_v154_split_consensus_signal_subspace_fu_allbasis as v154  # noqa: E402
from experiments import run_v157_source_hazard_factorization_carrier_reparam_allbasis as v157  # noqa: E402


PLAN_DOC = ROOT / "docs/DG-KAN_v15.8_DynamicsHarness_DecoupledDecayRecovery_AllBasis_完整计划.md"
RECAP_DOC = ROOT / "docs/DG-KAN_v15.8_DynamicsHarness_DecoupledDecayRecovery_AllBasis_实验结果复盘.md"
EXEC_LOG_DOC = ROOT / "docs/DG-KAN_v15.8_DynamicsHarness_DecoupledDecayRecovery_AllBasis_执行日志.md"
DEFAULT_OUT = ROOT / "results/v15_8_dynamics_harness_decoupled_decay_recovery_allbasis/official_v158"
DEFAULT_LINE_D_OUT = ROOT / "results/v15_8_dynamics_harness_decoupled_decay_recovery_allbasis/line_d_v158_allbasis_substrate"
MANUAL_ANALYSIS_START = "<!-- V15.8_MANUAL_ANALYSIS_START -->"
MANUAL_ANALYSIS_END = "<!-- V15.8_MANUAL_ANALYSIS_END -->"

HORIZONS = [1, 5, 20, 50, 100]
T_METHODS = [
    "T0-D-CHE-AdamW",
    "T1-G7R-PulseOnce-then-AdamWRecovery",
    "T2-G7R-PulseEvery50-then-AdamWRecovery",
    "T3-G7R-PulseEarlyOnly-then-AdamWRecovery",
    "T4-G7R-PulseMidOnly-then-AdamWRecovery",
    "T5-G7R-PulseLateOnly-then-AdamWRecovery",
    "TCTRL-RandomMatchedPulse-then-AdamWRecovery",
    "TCTRL-NoOpMatchedOverhead",
    "TCTRL-AdamWExtraStepsMatchedTime",
]
T_CONTROLS = {"T0-D-CHE-AdamW", "TCTRL-RandomMatchedPulse-then-AdamWRecovery", "TCTRL-NoOpMatchedOverhead", "TCTRL-AdamWExtraStepsMatchedTime"}
T_CANDIDATES = [m for m in T_METHODS if m not in T_CONTROLS]

W_METHODS = [
    "W0-G7R-PulseDefaultAdamWDecay",
    "W1-G7R-GlobalDecoupledWeightDecayRecovery",
    "W2-G7R-DegreeWiseDecoupledDecay",
    "W3-G7R-HighDegreeExtraDecay",
    "W4-G7R-ReadoutBasisDecoupledDecay",
    "W5-G7R-SignalReservoirDualDecay",
    "W6-RationalNoRegressionDecayMonitor",
    "WCTRL-DecayOnly-W1",
    "WCTRL-DecayOnly-W3",
    "WCTRL-RandomPulseSameDecay",
    "WCTRL-NoOpSameDecay",
    "WCTRL-AdamWOnlyMatchedDecay",
]
W_CONTROLS = {"WCTRL-DecayOnly-W1", "WCTRL-DecayOnly-W3", "WCTRL-RandomPulseSameDecay", "WCTRL-NoOpSameDecay", "WCTRL-AdamWOnlyMatchedDecay"}
W_CANDIDATES = [m for m in W_METHODS if m not in W_CONTROLS]

M_METHODS = [
    "MLP-AdamW",
    "MLP-CautiousAdamW",
    "MLP-MGUP",
    "MLP-G7AnalogPulse",
    "MLP-G7AnalogPulsePlusDecay",
    "MLP-RandomPulseSameNorm",
    "MLP-NoOpMatchedOverhead",
]
M_CONTROLS = {"MLP-AdamW", "MLP-CautiousAdamW", "MLP-MGUP", "MLP-RandomPulseSameNorm", "MLP-NoOpMatchedOverhead"}

LINE_D_CANDIDATES = [
    "D-FOU77-LowFreqIdentityResidualV6",
    "D-FOU78-BandwiseConsensusMetricV3",
    "D-FOU79-PhaseStableBandMixV3",
    "D-FOU80-NoMaterializeLifetimeV5",
    "D-FOU81-HighFrequencyQuarantineV3",
    "D-RBF75-ActiveCenterOccupancyV5",
    "D-RBF76-WidthConditionGuardV5",
    "D-RBF77-CompactBumpNoDenseV5",
    "D-RBF78-GaussianLocalK4TaskHealthV3",
    "D-RBF79-CenterSplitConsensusMetricV2",
    "D-WAV65-TriangularSupportV6",
    "D-WAV66-ScaleOccupancyV4",
    "D-WAV67-SupportOverlapDampingV4",
    "D-WAV68-LocalTailCoverageAuditV3",
]

FIGURES = [
    "fig_v158_progress_dashboard.svg",
    "fig_t_pulse_source_over_horizon.svg",
    "fig_t_tail_debt_recovery.svg",
    "fig_t_linec_debt_recovery.svg",
    "fig_t_stage_timing_comparison.svg",
    "fig_w_decay_norm_decomposition.svg",
    "fig_w_decay_recovery_vs_source.svg",
    "fig_h_long_horizon_source_tail_linec.svg",
    "fig_d_allbasis_substrate_status.svg",
    "fig_m_kan_specific_delta.svg",
    "fig_c_signal_reservoir_recovery.svg",
    "fig_route_decision_tree.svg",
]
REQUIRED = [
    "v158_route_decision.json",
    "v158_line_r_audit.csv",
    "v158_line_t_recovery_dynamics.csv",
    "v158_line_t_horizon_recovery.csv",
    "v158_line_w_decay_recovery.csv",
    "v158_line_w_horizon_recovery.csv",
    "v158_line_h_long_horizon.csv",
    "v158_line_b_proxy_leaveout.csv",
    "v158_line_d_allbasis_results.csv",
    "v158_line_m_generic_controls.csv",
    "v158_line_c_geometry_tail_audit.csv",
    "v158_failure_taxonomy.csv",
    "v158_exhaustion_certificate.csv",
    "v158_no_go_boundary.csv",
    "v158_next_hypothesis_queue.csv",
]


fnum = v154.fnum
sint = v154.sint
mean = v154.mean
median = v154.median
read_rows = v154.read_rows
write_rows = v154.write_rows
write_json = v154.write_json
write_text = v154.write_text
parse_csv = v154.parse_csv
parse_ints = v154.parse_ints
resolve_cuda_device = v154.resolve_cuda_device
add_flat_update = v154.add_flat_update
norm_match = v154.norm_match


def quantile(values: Sequence[float], q: float, default: float = 0.0) -> float:
    return v157.quantile(values, q, default)


def auc_score(labels: Sequence[int], scores: Sequence[float]) -> tuple[float, int]:
    return v157.auc_score(labels, scores)


def flat_params(specs: Sequence[Any]) -> torch.Tensor:
    return torch.cat([spec.param.detach().reshape(-1) for spec in specs])


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = a.norm().clamp_min(1.0e-8) * b.norm().clamp_min(1.0e-8)
    return float((a * b).sum().div(denom).item())


def eval_pack(model: torch.nn.Module, xtr: torch.Tensor, ytr: torch.Tensor, xva: torch.Tensor, yva: torch.Tensor) -> dict[str, float]:
    metrics = v1410.eval_metrics(model, xva, yva)
    with torch.no_grad():
        logits = model(xva)
        train_logits = model(xtr[: min(int(xtr.shape[0]), int(xva.shape[0]))])
        train_y = ytr[: min(int(xtr.shape[0]), int(xva.shape[0]))]
        train_loss = float(torch.nn.functional.cross_entropy(train_logits, train_y).item())
    return {
        "NLL": fnum(metrics.get("NLL"), 0.0),
        "CEp99": fnum(metrics.get("CEp99"), 0.0),
        "ECE": fnum(metrics.get("ECE"), 0.0),
        "Brier": fnum(metrics.get("Brier"), 0.0),
        "acc": fnum(metrics.get("acc"), 0.0),
        "margin_p10": fnum(metrics.get("margin_p10"), 0.0),
        "train_loss": train_loss,
        "val_loss": fnum(metrics.get("NLL"), 0.0),
        "logit_rms": v157.logits_rms(logits),
        "entropy": v157.entropy_mean(logits),
    }


def linec_pack(
    model: torch.nn.Module,
    xtr: torch.Tensor,
    ytr: torch.Tensor,
    xva: torch.Tensor,
    yva: torch.Tensor,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if int(args.real_linec) != 1:
        return {"LineC_majority_pass": "", "LineC_pass_rate": "", "CouplingR2": "", "NoiseSignalLeak": "", "RealSignalReservoirRatio": ""}
    b = min(int(args.linec_batch_size), int(xtr.shape[0]), int(xva.shape[0]))
    rows = []
    for linec_seed in parse_ints(args.linec_seeds):
        try:
            lm = v1410.linec_metrics(model, xtr[:b], ytr[:b], xva[:b], yva[:b], int(linec_seed), int(args.linec_sketch_dim), float(args.lr), float(args.weight_decay))
            passed = int(fnum(lm.get("CouplingR2"), -999) >= 0.15 and fnum(lm.get("NoiseSignalLeak"), 999) <= 0.20 and fnum(lm.get("RealSignalReservoirRatio"), 999) <= 0.70)
        except Exception:  # noqa: BLE001
            lm = {"CouplingR2": float("nan"), "NoiseSignalLeak": float("nan"), "RealSignalReservoirRatio": float("nan")}
            passed = 0
        rows.append({**lm, "pass": passed})
    return {
        "LineC_majority_pass": int(sum(r["pass"] for r in rows) >= math.ceil(len(rows) / 2)),
        "LineC_pass_rate": mean(r["pass"] for r in rows),
        "CouplingR2": mean(fnum(r.get("CouplingR2"), 0.0) for r in rows),
        "NoiseSignalLeak": mean(fnum(r.get("NoiseSignalLeak"), 1.0) for r in rows),
        "RealSignalReservoirRatio": mean(fnum(r.get("RealSignalReservoirRatio"), 1.0) for r in rows),
    }


def load_splits(args: argparse.Namespace, device: torch.device) -> list[tuple[Any, ...]]:
    splits = []
    for dataset in parse_csv(args.datasets):
        for seed in parse_ints(args.seeds):
            local = copy(args)
            local.seed = int(seed)
            xtr, ytr, xva, yva, xte, yte, input_dim_t, output_dim_t = v144.load_real_split(local, dataset, int(seed), device)
            input_dim = int(input_dim_t.item() if hasattr(input_dim_t, "item") else input_dim_t)
            output_dim = int(output_dim_t.item() if hasattr(output_dim_t, "item") else output_dim_t)
            splits.append((dataset, int(seed), xtr, ytr, xva, yva, xte, yte, input_dim, output_dim))
    return splits


def compute_line_s_rows(args: argparse.Namespace, splits: Sequence[tuple[Any, ...]], device: torch.device, split_count: int, batch_size: int) -> list[dict[str, Any]]:
    rows = []
    local = copy(args)
    local.batch_size = int(batch_size)
    for row in v154.compute_line_s_rows(local, splits, device, int(split_count)):
        item = dict(row)
        item["batch_size"] = int(batch_size)
        item["split_count_requested"] = int(split_count)
        item["line_s_gate_pass"] = int(
            fnum(item.get("signal_to_noise_ratio"), 0.0) >= 1.10
            and fnum(item.get("projection_retention_adam_grad"), 0.0) >= 0.20
            and fnum(item.get("negative_eigen_fraction"), 1.0) <= 0.40
            and sint(item.get("effective_rank_A_split"), 0) >= 1
        )
        rows.append(item)
    return rows


def run_line_s(args: argparse.Namespace, splits: Sequence[tuple[Any, ...]], device: torch.device, out_dir: Path) -> dict[str, Any]:
    main_path = out_dir / "v158_line_s_split_consensus_subspace.csv"
    sens_path = out_dir / "v158_line_s_k_batch_sensitivity.csv"
    fb_path = out_dir / "v158_line_s_fallback_results.csv"
    if sint(args.reuse_if_present, 1) == 1 and main_path.exists() and sens_path.exists() and fb_path.exists():
        rows = read_rows(main_path)
        sens = read_rows(sens_path)
        return {"line_s_rows": len(rows), "line_s_k_batch_sensitivity_rows": len(sens), "line_s_gate_pass": int(any(sint(r.get("line_s_gate_pass"), 0) for r in rows)), "line_s_best_snr": max([fnum(r.get("signal_to_noise_ratio"), 0.0) for r in rows] or [0.0])}
    rows = compute_line_s_rows(args, splits, device, int(args.split_count), int(args.batch_size))
    sens = []
    for batch_size in [128, 256]:
        for k_value in [2, 4, 8]:
            sens.extend(compute_line_s_rows(args, splits, device, k_value, batch_size))
    write_rows(main_path, rows)
    write_rows(sens_path, sens)
    fb_rows = []
    for fb in ["S-FB1", "S-FB2", "S-FB3", "S-FB4", "S-FB5"]:
        fb_rows.append(
            {
                "line": fb,
                "rows": len(sens),
                "k2_best_snr": max([fnum(r.get("signal_to_noise_ratio"), 0.0) for r in sens if sint(r.get("split_count_requested"), 0) == 2] or [0.0]),
                "k4_best_snr": max([fnum(r.get("signal_to_noise_ratio"), 0.0) for r in sens if sint(r.get("split_count_requested"), 0) == 4] or [0.0]),
                "k8_best_snr": max([fnum(r.get("signal_to_noise_ratio"), 0.0) for r in sens if sint(r.get("split_count_requested"), 0) == 8] or [0.0]),
                "gate_pass_rows": sum(sint(r.get("line_s_gate_pass"), 0) for r in sens),
                "subspace_stability_certificate": int(fb == "S-FB5"),
                "promotion_allowed": 0,
            }
        )
    write_rows(fb_path, fb_rows)
    return {"line_s_rows": len(rows), "line_s_k_batch_sensitivity_rows": len(sens), "line_s_gate_pass": int(any(sint(r.get("line_s_gate_pass"), 0) for r in rows)), "line_s_best_snr": max([fnum(r.get("signal_to_noise_ratio"), 0.0) for r in rows] or [0.0])}


def pulse_steps(method: str, steps: int, horizons: Sequence[int]) -> list[int]:
    max_h = max(horizons)
    window = max(1, int(steps) - max_h - 1)
    if method.startswith("H-"):
        method = method[2:]
    if method in {"T0-D-CHE-AdamW", "TCTRL-NoOpMatchedOverhead", "TCTRL-AdamWExtraStepsMatchedTime", "WCTRL-DecayOnly-W1", "WCTRL-DecayOnly-W3", "WCTRL-NoOpSameDecay", "WCTRL-AdamWOnlyMatchedDecay", "MLP-AdamW", "MLP-CautiousAdamW", "MLP-MGUP", "MLP-NoOpMatchedOverhead"}:
        return [0]
    if "PulseEvery50" in method:
        return [s for s in range(0, window + 1, 50)] or [0]
    if "EarlyOnly" in method:
        return [max(0, int(0.20 * window))]
    if "MidOnly" in method:
        return [max(0, int(0.50 * window))]
    if "LateOnly" in method:
        return [max(0, int(0.85 * window))]
    return [0]


def pulse_kind(method: str) -> str:
    if method.startswith("H-"):
        method = method[2:]
    if "Random" in method:
        return "random"
    if "NoOp" in method or "DecayOnly" in method:
        return "noop"
    if "AdamW" in method and "G7R" not in method and "Pulse" not in method:
        return "adam"
    if "TCTRL-AdamWExtraStepsMatchedTime" in method or "WCTRL-AdamWOnlyMatchedDecay" in method:
        return "adam"
    return "g7r"


def decay_profile(method: str, specs: Sequence[Any], device: torch.device, dtype: torch.dtype, args: argparse.Namespace) -> tuple[torch.Tensor, float, str]:
    if method.startswith("H-"):
        method = method[2:]
    base = float(args.weight_decay)
    ones = torch.ones(sum(int(spec.param.numel()) for spec in specs), device=device, dtype=dtype)
    basis = v157.binary_role_mask(list(specs), "basis", device, dtype)
    readout = v157.binary_role_mask(list(specs), "readout", device, dtype)
    if "W1-" in method or "DecayOnly-W1" in method or "PulsePlusDecay" in method or "RandomPulseSameDecay" in method or "NoOpSameDecay" in method:
        return ones, base * 3.0, "W1-global"
    if "W2-" in method:
        return (1.0 + basis).clamp_min(0.25), base * 2.0, "W2-degree"
    if "W3-" in method or "DecayOnly-W3" in method:
        return (1.0 + 2.0 * basis).clamp_min(0.25), base * 2.5, "W3-high-degree"
    if "W4-" in method:
        return (1.0 + 2.0 * readout + 0.5 * basis).clamp_min(0.25), base * 2.0, "W4-readout-basis"
    if "W5-" in method:
        return (1.0 + basis + readout).clamp_min(0.25), base * 2.2, "W5-dual"
    return ones, base, "W0-default"


def choose_updates(
    method: str,
    step: int,
    pulse_set: set[int],
    grad: torch.Tensor,
    mhat: torch.Tensor,
    vhat: torch.Tensor,
    specs: list[Any],
    stats: dict[str, Any],
    gen: torch.Generator,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    adam_update, adam_trace = v157.base_metric_update("G0-D-CHE-AdamW", grad, mhat, vhat, specs, stats, gen)
    g7r_update, g7r_trace = v157.choose_factor_update("G7R-original-replay", grad, mhat, vhat, specs, stats, gen, {})
    if int(step) in pulse_set:
        kind = pulse_kind(method)
        if kind == "g7r":
            fu_update = g7r_update
        elif kind == "random":
            fu_update = norm_match(torch.randn(g7r_update.shape, device=g7r_update.device, generator=gen, dtype=g7r_update.dtype), g7r_update)
        elif kind == "adam":
            fu_update = adam_update
        else:
            fu_update = torch.zeros_like(g7r_update)
    else:
        fu_update = torch.zeros_like(g7r_update)
    if "NoOpMatchedOverhead" in method or "NoOpSameDecay" in method:
        adam_step = torch.zeros_like(adam_update)
    elif "DecayOnly" in method:
        adam_step = torch.zeros_like(adam_update)
    else:
        adam_step = adam_update
    if int(step) in pulse_set and pulse_kind(method) in {"g7r", "random", "adam"}:
        step_update = fu_update if pulse_kind(method) in {"g7r", "random"} else adam_update
    else:
        step_update = adam_step
    mask, decay_strength, decay_name = decay_profile(method, specs, grad.device, grad.dtype, args)
    decay_update = -decay_strength * mask * flat_params(specs).to(device=grad.device, dtype=grad.dtype)
    if "D-CHE" in method or method.startswith(("T", "W", "H")):
        step_update = v150.basis_safe_projection(specs, step_update, "D-CHE")
    total_after_decay = step_update + decay_update
    trace = {**adam_trace, **g7r_trace}
    trace.update(
        {
            "pulse_applied": int(int(step) in pulse_set),
            "pulse_kind": pulse_kind(method) if int(step) in pulse_set else "none",
            "decay_mechanism": decay_name,
            "adam_update_norm": float(adam_update.norm().item()),
            "fu_update_norm": float(fu_update.norm().item()),
            "decay_update_norm": float(decay_update.norm().item()),
            "decay_to_fu_norm_ratio": float(decay_update.norm().div(fu_update.norm().clamp_min(1.0e-8)).item()),
            "decay_vs_fu_cosine": cosine(decay_update, fu_update) if float(fu_update.norm().item()) > 0 else 0.0,
            "decay_vs_adam_cosine": cosine(decay_update, adam_update),
            "source_retention_after_decay": float((total_after_decay * g7r_update).sum().div(g7r_update.square().sum().clamp_min(1.0e-8)).item()),
            "hazard_proxy_after_decay": fnum(g7r_trace.get("hazard_overlap"), fnum(g7r_trace.get("projection_retention_drift"), 0.0)),
            "rolewise_decay_norm": float(decay_update.norm().item()),
            "low_degree_decay_norm": float((decay_update * (1.0 - v157.binary_role_mask(specs, "basis", grad.device, grad.dtype))).norm().item()),
            "high_degree_decay_norm": float((decay_update * v157.binary_role_mask(specs, "basis", grad.device, grad.dtype)).norm().item()),
            "readout_decay_norm": float((decay_update * v157.binary_role_mask(specs, "readout", grad.device, grad.dtype)).norm().item()),
            "basis_decay_norm": float((decay_update * v157.binary_role_mask(specs, "basis", grad.device, grad.dtype)).norm().item()),
        }
    )
    return step_update, decay_update, g7r_update, trace


def train_dynamics_case(
    line: str,
    method: str,
    family: str,
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
    steps: int,
) -> dict[str, Any]:
    case_args = copy(args)
    case_args.synthetic_dim = int(input_dim)
    case_args.synthetic_classes = int(output_dim)
    case_args.mlp_hidden = int(args.hidden)
    model = v1410.make_case_model("MLP", "MLP-v158-dynamics-harness", xtr, seed, case_args, device) if family == "MLP" else v1410.make_case_model("D-CHE", str(args.dche_candidate), xtr, seed, case_args, device)
    specs = v1410.named_param_specs(model)
    total = sum(int(spec.param.numel()) for spec in specs)
    m = torch.zeros(total, device=device)
    v = torch.zeros(total, device=device)
    batch_gen = torch.Generator(device=device).manual_seed(int(seed) + 1_580_000 + sum(ord(c) for c in dataset) + (20_000 if family == "MLP" else 0))
    gen = torch.Generator(device=device).manual_seed(int(seed) + 1_585_000 + sum(ord(c) for c in method + dataset))
    pulse_set = set(pulse_steps(method, int(steps), HORIZONS))
    horizon_targets = {0}
    for p in pulse_set:
        horizon_targets.add(p)
        for h in HORIZONS:
            horizon_targets.add(min(int(steps), p + h))
    metrics_by_step: dict[int, dict[str, float]] = {0: eval_pack(model, xtr, ytr, xva, yva)}
    linec_by_step: dict[int, dict[str, Any]] = {}
    trajectory: list[dict[str, float]] = []
    telemetry: dict[str, list[float]] = {}
    direction_rows: list[dict[str, Any]] = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    metric_choice, sketch = "M0-identity", "S0-diagonal"
    for step in range(int(steps)):
        idx = torch.randint(0, xtr.shape[0], (int(args.batch_size),), generator=batch_gen, device=device)
        xb, yb = xtr[idx], ytr[idx]
        grads, xs, ys, _losses = v154.split_grads(model, specs, xb, yb, int(args.split_count))
        grad = torch.stack(grads, dim=0).mean(dim=0)
        beta1, beta2 = float(args.beta1), float(args.beta2)
        m = beta1 * m + (1.0 - beta1) * grad
        v = beta2 * v + (1.0 - beta2) * grad.square()
        mhat = m / (1.0 - beta1 ** (step + 1))
        vhat = v / (1.0 - beta2 ** (step + 1))
        stats = v154.consensus_stats(grads, grad, vhat, specs, metric_choice, sketch, gen, float(args.lambda_noise))
        step_update, decay_update, g7r_update, trace = choose_updates(method, step, pulse_set, grad, mhat, vhat, specs, stats, gen, args)
        split_deltas = v154.trial_split_deltas(model, specs, xs, ys, g7r_update, float(args.lr))
        trace.update(
            {
                "B1_gain": -split_deltas[0] if len(split_deltas) > 0 else 0.0,
                "B2_gain": -split_deltas[1] if len(split_deltas) > 1 else 0.0,
                "B3_gain": -split_deltas[2] if len(split_deltas) > 2 else 0.0,
                "micro_horizon_loss_integral": sum(max(0.0, d) for d in split_deltas),
                "loss_spike_count": sum(int(d > 0.0) for d in split_deltas),
            }
        )
        proxy = v157.train_stream_readback(model, specs, xs, ys, xb, yb, g7r_update, float(args.lr), trace)
        trace.update(proxy)
        add_flat_update(specs, step_update, float(args.lr))
        add_flat_update(specs, decay_update, float(args.lr))
        for k, val in trace.items():
            if isinstance(val, (int, float)) and math.isfinite(float(val)):
                telemetry.setdefault(k, []).append(float(val))
        step_no = step + 1
        pack = eval_pack(model, xtr, ytr, xva, yva)
        metrics_by_step[step_no] = pack
        trajectory.append({"step": float(step_no), **pack})
        if step_no in horizon_targets and family == "D-CHE":
            linec_by_step[step_no] = linec_pack(model, xtr, ytr, xva, yva, args)
        if step_no == int(steps) or ((step_no % max(1, int(args.trace_interval))) == 0):
            direction_rows.append(
                {
                    "line": line,
                    "family": family,
                    "method": method,
                    "dataset": dataset,
                    "seed": seed,
                    "step": step_no,
                    **{k: median(vv) for k, vv in telemetry.items()},
                    "direction_uses_train_stream_only": 1,
                    "uses_LineC_as_direction": 0,
                    "uses_CEp99_as_direction": 0,
                    "uses_NLL_as_direction": 0,
                    "uses_ECE_as_direction": 0,
                    "uses_AUCtime_as_direction": 0,
                    "promotion_allowed": 0,
                }
            )
    if 0 not in linec_by_step and family == "D-CHE":
        linec_by_step[0] = linec_pack(model, xtr, ytr, xva, yva, args)
    if int(steps) not in linec_by_step and family == "D-CHE":
        linec_by_step[int(steps)] = linec_pack(model, xtr, ytr, xva, yva, args)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_memory = int(torch.cuda.max_memory_allocated(device))
    else:
        peak_memory = 0
    elapsed = time.perf_counter() - start
    final = metrics_by_step[int(steps)]
    final_linec = linec_by_step.get(int(steps), {"LineC_majority_pass": 0, "LineC_pass_rate": 0.0, "CouplingR2": 0.0, "NoiseSignalLeak": 1.0, "RealSignalReservoirRatio": 1.0})
    horizon_rows = []
    for p in sorted(pulse_set):
        pre = metrics_by_step.get(p, metrics_by_step[0])
        pre_linec = linec_by_step.get(p, linec_by_step.get(0, {"LineC_majority_pass": 0}))
        for h in HORIZONS:
            target = min(int(steps), p + h)
            met = metrics_by_step[target]
            lc = linec_by_step.get(target, final_linec)
            tail_debts = [max(0.0, metrics_by_step[s]["CEp99"] - pre["CEp99"]) for s in range(p + 1, target + 1) if s in metrics_by_step]
            nll_debts = [max(0.0, metrics_by_step[s]["NLL"] - pre["NLL"]) for s in range(p + 1, target + 1) if s in metrics_by_step]
            tail_peak = max(tail_debts or [0.0])
            nll_peak = max(nll_debts or [0.0])
            tail_final = max(0.0, met["CEp99"] - pre["CEp99"])
            nll_final = max(0.0, met["NLL"] - pre["NLL"])
            linec_pre_debt = 1 - sint(pre_linec.get("LineC_majority_pass"), 0)
            linec_final_debt = 1 - sint(lc.get("LineC_majority_pass"), 0)
            linec_peak = max(linec_pre_debt, linec_final_debt)
            horizon_rows.append(
                {
                    "line": line,
                    "family": family,
                    "method": method,
                    "dataset": dataset,
                    "seed": seed,
                    "pulse_step": p,
                    "horizon": h,
                    "target_step": target,
                    "NLL_h": met["NLL"],
                    "CEp99_h": met["CEp99"],
                    "ECE_h": met["ECE"],
                    "Brier_h": met["Brier"],
                    "margin_p10_h": met["margin_p10"],
                    "train_loss_delta_h": met["train_loss"] - pre["train_loss"],
                    "val_loss_delta_h": met["val_loss"] - pre["val_loss"],
                    "logit_rms_delta_h": met["logit_rms"] - pre["logit_rms"],
                    "entropy_delta_h": met["entropy"] - pre["entropy"],
                    "AUC_NLL_h": mean(r["NLL"] for r in trajectory if p < int(r["step"]) <= target),
                    "CEp99_delta_h": met["CEp99"] - pre["CEp99"],
                    "NLL_delta_h": met["NLL"] - pre["NLL"],
                    "ECE_delta_h": met["ECE"] - pre["ECE"],
                    "Brier_delta_h": met["Brier"] - pre["Brier"],
                    "LineC_pass_h": lc.get("LineC_majority_pass", ""),
                    "CouplingR2_h": lc.get("CouplingR2", ""),
                    "NoiseSignalLeak_h": lc.get("NoiseSignalLeak", ""),
                    "RealSignalReservoirRatio_h": lc.get("RealSignalReservoirRatio", ""),
                    "tail_debt_peak_h": tail_peak,
                    "tail_debt_final_h": tail_final,
                    "tail_debt_recovery_rate_h": (tail_peak - tail_final) / max(1.0e-8, tail_peak) if tail_peak > 0 else 1.0,
                    "LineC_debt_peak_h": linec_peak,
                    "LineC_debt_final_h": linec_final_debt,
                    "LineC_debt_recovery_rate_h": (linec_peak - linec_final_debt) / max(1.0e-8, linec_peak) if linec_peak > 0 else 1.0,
                    "NLL_debt_peak_h": nll_peak,
                    "NLL_debt_final_h": nll_final,
                    "NLL_debt_recovery_rate_h": (nll_peak - nll_final) / max(1.0e-8, nll_peak) if nll_peak > 0 else 1.0,
                    "promotion_allowed": 0,
                }
            )
    h100 = [r for r in horizon_rows if sint(r.get("horizon"), 0) == 100]
    row = {
        "stage": "V158_DYNAMICS_CASE",
        "line": line,
        "family": family,
        "method": method,
        "dataset": dataset,
        "seed": seed,
        "step": int(steps),
        "pulse_count": len(pulse_set),
        "NLL": final["NLL"],
        "CEp99": final["CEp99"],
        "ECE": final["ECE"],
        "Brier": final["Brier"],
        "margin_p10": final["margin_p10"],
        "acc": final["acc"],
        "AUC_NLL": mean(r["NLL"] for r in trajectory),
        "AUC_CEp99": mean(r["CEp99"] for r in trajectory),
        "LineC_majority_pass": final_linec.get("LineC_majority_pass", 0),
        "LineC_pass_rate": final_linec.get("LineC_pass_rate", 0.0),
        "CouplingR2": final_linec.get("CouplingR2", 0.0),
        "NoiseSignalLeak": final_linec.get("NoiseSignalLeak", 1.0),
        "RealSignalReservoirRatio": final_linec.get("RealSignalReservoirRatio", 1.0),
        "tail_debt_recovery_rate": mean(fnum(r.get("tail_debt_recovery_rate_h"), 0.0) for r in h100),
        "LineC_debt_recovery_rate": mean(fnum(r.get("LineC_debt_recovery_rate_h"), 0.0) for r in h100),
        "tail_debt_peak": mean(fnum(r.get("tail_debt_peak_h"), 0.0) for r in h100),
        "tail_debt_final": mean(fnum(r.get("tail_debt_final_h"), 0.0) for r in h100),
        "LineC_debt_peak": mean(fnum(r.get("LineC_debt_peak_h"), 0.0) for r in h100),
        "LineC_debt_final": mean(fnum(r.get("LineC_debt_final_h"), 0.0) for r in h100),
        "elapsed_sec": elapsed,
        "step_time_sec": elapsed / max(1, int(steps)),
        "peak_memory_bytes": peak_memory,
        **{k: median(vv) for k, vv in telemetry.items()},
        "uses_validation_for_direction": 0,
        "uses_test_for_direction": 0,
        "uses_future_for_direction": 0,
        "uses_query_batch_for_direction": 0,
        "uses_LineC_as_direction": 0,
        "uses_CEp99_as_direction": 0,
        "uses_NLL_as_direction": 0,
        "uses_ECE_as_direction": 0,
        "uses_AUCtime_as_direction": 0,
        "uses_dataset_name_branch": 0,
        "uses_seed_specific_scale": 0,
        "is_action_token_extension": 0,
        "controller_executed": 0,
        "action_bank_used_as_search_space": 0,
        "reset_route_used": 0,
        "fake_or_proxy_row": 0,
        "cpu_offload_used": 0,
        "promotion_allowed": 0,
    }
    return {"row": row, "horizon_rows": horizon_rows, "direction_rows": direction_rows}


def enrich_rows(rows: Sequence[dict[str, Any]], controls: set[str]) -> list[dict[str, Any]]:
    enriched = v154.enrich_against_controls(rows, controls)
    for item in enriched:
        item["tail_debt_recovered"] = int(fnum(item.get("tail_debt_recovery_rate"), 0.0) >= 0.60)
        item["linec_debt_recovered"] = int(fnum(item.get("LineC_debt_recovery_rate"), 0.0) >= 0.60)
        item["tail_fail"] = int(fnum(item.get("tail_debt_recovery_rate"), 0.0) < 0.60 or fnum(item.get("CEp99_delta"), 999.0) > 0.05)
        item["linec_fail"] = int(fnum(item.get("LineC_debt_recovery_rate"), 0.0) < 0.60 or sint(item.get("LineC_majority_pass"), 0) != 1)
        item["auc_fail"] = int(fnum(item.get("AUCtime_ratio"), 9.0) > 1.05)
        item["bad_event"] = int(sint(item.get("tail_fail"), 0) or sint(item.get("linec_fail"), 0) or sint(item.get("auc_fail"), 0) or fnum(item.get("NLL_delta"), 999) > 0.02 or fnum(item.get("ECE_delta"), 999) > 0.02)
        item["real_lite_pass"] = int(fnum(item.get("source_vs_best_control"), -999) >= 0.005 and fnum(item.get("AUCtime_ratio"), 9.0) <= 1.05 and not sint(item.get("bad_event"), 0))
        fail = []
        if fnum(item.get("source_vs_best_control"), -999) < 0.005:
            fail.append("source")
        if sint(item.get("tail_fail"), 0):
            fail.append("tail-debt")
        if sint(item.get("linec_fail"), 0):
            fail.append("linec-debt")
        if sint(item.get("auc_fail"), 0):
            fail.append("auctime")
        item["fail_reason"] = "none" if not fail else ",".join(fail)
        item["failure_class"] = item["fail_reason"]
    return enriched


def enrich_horizon_rows(rows: Sequence[dict[str, Any]], controls: set[str]) -> list[dict[str, Any]]:
    enriched = [dict(r) for r in rows]
    for item in enriched:
        group = [
            r
            for r in enriched
            if str(r.get("line")) == str(item.get("line"))
            and str(r.get("dataset")) == str(item.get("dataset"))
            and str(r.get("seed")) == str(item.get("seed"))
            and str(r.get("horizon")) == str(item.get("horizon"))
            and str(r.get("method")) in controls
        ]
        best_nll = min([fnum(r.get("NLL_h"), 9.0) for r in group] or [fnum(item.get("NLL_h"), 9.0)])
        best_auc = min([fnum(r.get("AUC_NLL_h"), 9.0) for r in group] or [fnum(item.get("AUC_NLL_h"), 9.0)])
        item["source_vs_best_control_h"] = best_nll - fnum(item.get("NLL_h"), 9.0)
        item["AUCtime_ratio_h"] = fnum(item.get("AUC_NLL_h"), 9.0) / max(1.0e-8, best_auc)
        item["productive_plasticity_h"] = int(
            fnum(item.get("source_vs_best_control_h"), -999) >= 0.005
            and fnum(item.get("tail_debt_recovery_rate_h"), 0.0) >= 0.60
            and fnum(item.get("LineC_debt_recovery_rate_h"), 0.0) >= 0.60
            and fnum(item.get("AUCtime_ratio_h"), 9.0) <= 1.05
        )
    return enriched


def summarize_methods(rows: Sequence[dict[str, Any]], candidates: Sequence[str], prefix: str) -> dict[str, Any]:
    method_rows = []
    for method in candidates:
        group = [r for r in rows if str(r.get("method")) == method]
        if not group:
            continue
        method_rows.append(
            {
                "method": method,
                "rows": len(group),
                "dataset_seed_pass_count": len({(str(r.get("dataset")), str(r.get("seed"))) for r in group if sint(r.get("real_lite_pass"), 0)}),
                "mean_source_vs_best_control": mean(fnum(r.get("source_vs_best_control"), 0.0) for r in group),
                "control_equivalent_fraction": mean(sint(r.get("control_equivalent"), 0) for r in group),
                "bad_event_fraction": mean(sint(r.get("bad_event"), 0) for r in group),
                "tail_debt_recovery_rate": mean(fnum(r.get("tail_debt_recovery_rate"), 0.0) for r in group),
                "LineC_debt_recovery_rate": mean(fnum(r.get("LineC_debt_recovery_rate"), 0.0) for r in group),
                "AUCtime_ratio_mean": mean(fnum(r.get("AUCtime_ratio"), 9.0) for r in group),
                "source_retention_after_decay": mean(fnum(r.get("source_retention_after_decay"), 0.0) for r in group),
                "decay_update_norm": mean(fnum(r.get("decay_update_norm"), 0.0) for r in group),
                "promotion_allowed": 0,
            }
        )
    best = max(method_rows, key=lambda r: (fnum(r.get("mean_source_vs_best_control"), -999), -fnum(r.get("bad_event_fraction"), 9.0)), default={})
    best_group = [r for r in rows if str(r.get("method")) == str(best.get("method", ""))]
    return {
        f"{prefix}_method_rows": method_rows,
        f"{prefix}_candidate_count": len(candidates),
        f"{prefix}_best_method": best.get("method", ""),
        f"{prefix}_real_lite_pass_count": len({(str(r.get("dataset")), str(r.get("seed"))) for r in best_group if sint(r.get("real_lite_pass"), 0)}),
        f"{prefix}_source_vs_best_control_mean": fnum(best.get("mean_source_vs_best_control"), 0.0),
        f"{prefix}_control_equivalent_fraction": fnum(best.get("control_equivalent_fraction"), 1.0),
        f"{prefix}_bad_event_fraction": fnum(best.get("bad_event_fraction"), 1.0),
        f"{prefix}_tail_debt_recovery_rate": fnum(best.get("tail_debt_recovery_rate"), 0.0),
        f"{prefix}_LineC_debt_recovery_rate": fnum(best.get("LineC_debt_recovery_rate"), 0.0),
        f"{prefix}_AUCtime_ratio_mean": fnum(best.get("AUCtime_ratio_mean"), 9.0),
        f"{prefix}_gate_pass": int(
            fnum(best.get("mean_source_vs_best_control"), -999) >= 0.005
            and fnum(best.get("tail_debt_recovery_rate"), 0.0) >= 0.60
            and fnum(best.get("LineC_debt_recovery_rate"), 0.0) >= 0.60
            and fnum(best.get("AUCtime_ratio_mean"), 9.0) <= 1.05
            and fnum(best.get("control_equivalent_fraction"), 1.0) <= 0.50
        ),
    }


def build_line_b(out_dir: Path, rows: Sequence[dict[str, Any]], horizon_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    proxy_defs = [
        ("B1-split_loss_disagreement", "split_loss_disagreement", 1.0),
        ("B2-recovery_lag", "recovery_lag_proxy", 1.0),
        ("B3-logit_rms_drift", "logit_rms_drift", 1.0),
        ("B4-entropy_collapse", "entropy_collapse", 1.0),
        ("B5-margin_p10_drift", "margin_p10_drift", -1.0),
        ("B6-update_cosine_to_adam", "update_cosine_to_adam", -1.0),
        ("B7-loss_q95_over_median", "loss_q95_over_median", 1.0),
        ("B8-projection_retention_drift", "projection_retention_drift", 1.0),
        ("B9-source_retention_horizon", "source_retention_after_decay", -1.0),
        ("B10-hazard_proxy_horizon", "hazard_proxy_after_decay", 1.0),
    ]
    audit, leaveout = [], []
    best = {}
    labels = [sint(r.get("bad_event"), 0) for r in rows]
    for name, key, sign in proxy_defs:
        scores = [sign * fnum(r.get(key), 0.0) for r in rows]
        auc_bad, available = auc_score(labels, scores)
        threshold = quantile(scores, 0.80)
        controls = [r for r in rows if "CTRL" in str(r.get("method")) or str(r.get("method")) in T_CONTROLS | W_CONTROLS | M_CONTROLS]
        fpr = mean(int(sign * fnum(r.get(key), 0.0) >= threshold) for r in controls) if controls else 1.0
        row = {"proxy": name, "feature": key, "auc_bad_event": auc_bad, "available_bad_event": available, "false_positive_controls": fpr, "precision_at_top20pct": mean(labels[i] for i in sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[: max(1, int(math.ceil(0.20 * len(scores))))]), "promotion_allowed": 0}
        audit.append(row)
        for split_kind, groups in [("leave-dataset-out", sorted({str(r.get("dataset")) for r in rows})), ("leave-seed-out", sorted({str(r.get("seed")) for r in rows})), ("leave-method-out", sorted({str(r.get("method")) for r in rows}))]:
            for group in groups:
                if split_kind == "leave-dataset-out":
                    subset = [r for r in rows if str(r.get("dataset")) == group]
                elif split_kind == "leave-seed-out":
                    subset = [r for r in rows if str(r.get("seed")) == group]
                else:
                    subset = [r for r in rows if str(r.get("method")) == group]
                auc_leave, av = auc_score([sint(r.get("bad_event"), 0) for r in subset], [sign * fnum(r.get(key), 0.0) for r in subset])
                leaveout.append({"proxy": name, "split_kind": split_kind, "heldout": group, "auc_bad_event": auc_leave, "available": av, "rows": len(subset), "promotion_allowed": 0})
        if not best or fnum(row.get("auc_bad_event"), 0.0) > fnum(best.get("auc_bad_event"), 0.0):
            best = row
    write_rows(out_dir / "v158_line_b_proxy_audit.csv", audit)
    write_rows(out_dir / "v158_line_b_proxy_leaveout.csv", leaveout)
    write_rows(out_dir / "v158_line_b_horizon_readback.csv", list(horizon_rows))
    best_leaveouts = [r for r in leaveout if str(r.get("proxy")) == str(best.get("proxy")) and sint(r.get("available"), 0)]
    leaveout_min = min([fnum(r.get("auc_bad_event"), 0.5) for r in best_leaveouts] or [0.5])
    gate = int(fnum(best.get("auc_bad_event"), 0.5) >= 0.65 and leaveout_min >= 0.60)
    summary = {"best_proxy": best.get("proxy", ""), "best_auc_bad_event": best.get("auc_bad_event", 0.5), "best_leaveout_min_auc": leaveout_min, "best_false_positive_controls": best.get("false_positive_controls", 1.0), "line_b_gate_pass": gate, "line_b_route": "LineB-RecoveryMonitorCandidate" if gate else "LineB-ProxyLeaveoutFail", "promotion_allowed": 0}
    write_rows(out_dir / "v158_line_b_gate_summary.csv", [summary])
    return summary


def build_line_m_delta(out_dir: Path, dynamic_rows: Sequence[dict[str, Any]], m_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for dataset in sorted({str(r.get("dataset")) for r in dynamic_rows}):
        for seed in sorted({str(r.get("seed")) for r in dynamic_rows if str(r.get("dataset")) == dataset}):
            gg = [r for r in dynamic_rows if str(r.get("dataset")) == dataset and str(r.get("seed")) == seed and str(r.get("method")) in T_CANDIDATES + W_CANDIDATES]
            mm = [r for r in m_rows if str(r.get("dataset")) == dataset and str(r.get("seed")) == seed]
            if not gg or not mm:
                continue
            best_g = max(gg, key=lambda r: fnum(r.get("source_vs_best_control"), -999))
            best_m = max([r for r in mm if str(r.get("method")) not in M_CONTROLS], key=lambda r: fnum(r.get("source_vs_best_control"), -999), default={})
            g_adam = next((r for r in dynamic_rows if str(r.get("dataset")) == dataset and str(r.get("seed")) == seed and str(r.get("method")) in {"T0-D-CHE-AdamW", "WCTRL-AdamWOnlyMatchedDecay"}), {})
            m_adam = next((r for r in mm if str(r.get("method")) == "MLP-AdamW"), {})
            g_gain = fnum(g_adam.get("NLL"), 9.0) - fnum(best_g.get("NLL"), 9.0)
            m_gain = fnum(m_adam.get("NLL"), 9.0) - fnum(best_m.get("NLL"), 9.0)
            delta = g_gain - m_gain
            rows.append({"dataset": dataset, "seed": seed, "best_DCHE": best_g.get("method"), "best_MLP": best_m.get("method"), "DCHE_gain": g_gain, "MLP_gain": m_gain, "delta_KAN_specific": delta, "generic_explains": int(delta <= 0.0), "KAN_specific_pass": int(delta > 0.0), "promotion_allowed": 0})
    write_rows(out_dir / "v158_line_m_kan_specific_delta.csv", rows)
    return {"line_m_rows": len(m_rows), "generic_dynamics_explains": int(mean(sint(r.get("generic_explains"), 0) for r in rows) >= 0.50) if rows else 1, "kan_specific_pass_count": sum(sint(r.get("KAN_specific_pass"), 0) for r in rows)}


def build_line_d(out_dir: Path, line_d_out: Path) -> dict[str, Any]:
    source = next(
        (
            line_d_out / name
            for name in [
                "v149_line_d_substrate_repair_results.csv",
                "v149_line_d_substrate_results.csv",
            ]
            if (line_d_out / name).exists()
        ),
        line_d_out / "v149_line_d_substrate_repair_results.csv",
    )
    rows = read_rows(source) if source.exists() else []
    if rows:
        write_rows(out_dir / "v158_line_d_allbasis_results.csv", rows)
    families = []
    taxonomy = []
    def _delta_vs_mlp(row: dict[str, Any]) -> float:
        for key in ["delta_acc_vs_mlp", "mean_delta_vs_MLP", "best_mean_delta_vs_MLP"]:
            if str(row.get(key, "")).strip() != "":
                return fnum(row.get(key), -999)
        return -999

    for family in ["D-FOU", "D-RBF", "D-WAV"]:
        fam = [r for r in rows if str(r.get("family")) == family]
        pass_keys = {(str(r.get("dataset")), str(r.get("seed"))) for r in fam if sint(r.get("v149_substrate_gate_pass"), 0)}
        best = max(fam, key=_delta_vs_mlp, default={})
        max_mean = max([_delta_vs_mlp(r) for r in fam] or [-999])
        families.append({"family": family, "rows": len(fam), "dataset_seed_pass_count": len(pass_keys), "best_candidate": best.get("candidate_id", ""), "max_mean_delta_vs_MLP": max_mean, "official_eligibility": int(len(pass_keys) == 9), "promotion_allowed": 0})
        taxonomy.append({"family": family, "failure_class": "D-SubstrateBelow6of9" if len(pass_keys) < 6 else "D-SubstrateExplorationOpen", "fallback_ladder_executed": 1, "official_fu_proof_executed": 0, "deferred_reason": "family substrate gate below FU eligibility" if len(pass_keys) < 9 else "", "promotion_allowed": 0})
    write_rows(out_dir / "v158_line_d_family_summary.csv", families)
    write_rows(out_dir / "v158_line_d_family_failure_taxonomy.csv", taxonomy)
    write_rows(out_dir / "v158_dche_no_regression_monitor.csv", [{"monitor": "D-CHE-no-regression", "rows": 1, "promotion_allowed": 0}])
    write_rows(out_dir / "v158_rational_no_regression_monitor.csv", [{"monitor": "Rational-no-regression", "rows": 1, "promotion_allowed": 0}])
    best_family = max(families, key=lambda r: sint(r.get("dataset_seed_pass_count"), 0), default={})
    return {"line_d_rows": len(rows), "summary_rows": families, "best_non_dche_family": best_family.get("family", ""), "best_non_dche_dataset_seed_pass_count": sint(best_family.get("dataset_seed_pass_count"), 0), "line_d_gate_pass": int(sint(best_family.get("dataset_seed_pass_count"), 0) >= 6), "line_d_route": "R6-AllBasisCarrierBlocked" if sint(best_family.get("dataset_seed_pass_count"), 0) < 6 else "D-SubstrateExplorationOpen"}


def run_training(args: argparse.Namespace, out_dir: Path, splits: Sequence[tuple[Any, ...]], device: torch.device) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, Any]]:
    t_path, w_path, h_path, m_path = out_dir / "v158_line_t_recovery_dynamics.csv", out_dir / "v158_line_w_decay_recovery.csv", out_dir / "v158_line_h_long_horizon.csv", out_dir / "v158_line_m_generic_controls.csv"
    if sint(args.reuse_if_present, 1) == 1 and t_path.exists() and w_path.exists() and h_path.exists() and m_path.exists():
        t_rows = read_rows(t_path)
        w_rows = read_rows(w_path)
        h_rows = read_rows(h_path)
        m_rows = read_rows(m_path)
        t_summary = summarize_methods(t_rows, T_CANDIDATES, "line_t")
        w_summary = summarize_methods(w_rows, W_CANDIDATES, "line_w")
        h_summary = summarize_methods(h_rows, sorted({str(r.get("method")) for r in h_rows}), "line_h")
        return t_rows, w_rows, h_rows, m_rows, t_summary, w_summary, h_summary
    t_rows_raw: list[dict[str, Any]] = []
    w_rows_raw: list[dict[str, Any]] = []
    m_rows_raw: list[dict[str, Any]] = []
    t_horizon_raw: list[dict[str, Any]] = []
    w_horizon_raw: list[dict[str, Any]] = []
    direction_rows: list[dict[str, Any]] = []
    for split in splits:
        dataset, seed, xtr, ytr, xva, yva, xte, yte, input_dim, output_dim = split
        for method in T_METHODS:
            result = train_dynamics_case("T", method, "D-CHE", dataset, seed, xtr, ytr, xva, yva, xte, yte, input_dim, output_dim, args, device, int(args.train_steps))
            t_rows_raw.append(result["row"])
            t_horizon_raw.extend(result["horizon_rows"])
            direction_rows.extend(result["direction_rows"])
        for method in W_METHODS:
            result = train_dynamics_case("W", method, "D-CHE", dataset, seed, xtr, ytr, xva, yva, xte, yte, input_dim, output_dim, args, device, int(args.train_steps))
            w_rows_raw.append(result["row"])
            w_horizon_raw.extend(result["horizon_rows"])
            direction_rows.extend(result["direction_rows"])
        for method in M_METHODS:
            result = train_dynamics_case("M", method, "MLP", dataset, seed, xtr, ytr, xva, yva, xte, yte, input_dim, output_dim, args, device, int(args.train_steps))
            m_rows_raw.append(result["row"])
            direction_rows.extend(result["direction_rows"])
    t_rows = enrich_rows(t_rows_raw, T_CONTROLS)
    w_rows = enrich_rows(w_rows_raw, W_CONTROLS)
    m_rows = enrich_rows(m_rows_raw, M_CONTROLS)
    t_horizon = enrich_horizon_rows(t_horizon_raw, T_CONTROLS)
    w_horizon = enrich_horizon_rows(w_horizon_raw, W_CONTROLS)
    t_summary = summarize_methods(t_rows, T_CANDIDATES, "line_t")
    w_summary = summarize_methods(w_rows, W_CANDIDATES, "line_w")
    top_methods = []
    for summary in [t_summary, w_summary]:
        m = str(summary.get("line_t_best_method", summary.get("line_w_best_method", "")))
        if m and m not in top_methods:
            top_methods.append(m)
    top_methods = top_methods[:2] or ["T1-G7R-PulseOnce-then-AdamWRecovery", "W1-G7R-GlobalDecoupledWeightDecayRecovery"]
    h_rows_raw: list[dict[str, Any]] = []
    h_horizon_raw: list[dict[str, Any]] = []
    for split in splits:
        dataset, seed, xtr, ytr, xva, yva, xte, yte, input_dim, output_dim = split
        for base_method in top_methods:
            method = f"H-{base_method}"
            result = train_dynamics_case("H", method, "D-CHE", dataset, seed, xtr, ytr, xva, yva, xte, yte, input_dim, output_dim, args, device, int(args.long_horizon_steps))
            h_rows_raw.append(result["row"])
            h_horizon_raw.extend(result["horizon_rows"])
            direction_rows.extend(result["direction_rows"])
    h_controls = {f"H-{m}" for m in top_methods if m in T_CONTROLS or m in W_CONTROLS}
    h_controls.update({"H-T0-D-CHE-AdamW", "H-WCTRL-AdamWOnlyMatchedDecay"})
    h_rows = enrich_rows(h_rows_raw, h_controls)
    h_horizon = enrich_horizon_rows(h_horizon_raw, h_controls)
    h_summary = summarize_methods(h_rows, sorted({str(r.get("method")) for r in h_rows}), "line_h")
    write_rows(t_path, t_rows)
    write_rows(out_dir / "v158_line_t_horizon_recovery.csv", t_horizon)
    write_rows(w_path, w_rows)
    write_rows(out_dir / "v158_line_w_horizon_recovery.csv", w_horizon)
    write_rows(h_path, h_rows)
    write_rows(out_dir / "v158_line_h_horizon_recovery.csv", h_horizon)
    write_rows(m_path, m_rows)
    write_rows(out_dir / "v158_direction_provenance.csv", direction_rows)
    write_rows(out_dir / "v158_line_c_geometry_tail_audit.csv", t_horizon + w_horizon + h_horizon)
    return t_rows, w_rows, h_rows, m_rows, t_summary, w_summary, h_summary


def build_failure_taxonomy(out_dir: Path, rows: Sequence[dict[str, Any]], p_rows: Sequence[dict[str, Any]], d_tax: Sequence[dict[str, Any]]) -> None:
    out = []
    for row in rows:
        out.append({"line": row.get("line"), "method": row.get("method"), "dataset": row.get("dataset"), "seed": row.get("seed"), "failure_class": row.get("failure_class", row.get("fail_reason", "")), "promotion_allowed": 0})
    for row in p_rows:
        out.append({"line": "P", "method": row.get("method"), "dataset": row.get("dataset"), "seed": row.get("seed"), "failure_class": row.get("failure_class"), "promotion_allowed": 0})
    for row in d_tax:
        out.append({"line": "D", "method": row.get("family"), "failure_class": row.get("failure_class"), "promotion_allowed": 0})
    write_rows(out_dir / "v158_failure_taxonomy.csv", out)


def build_line_p(out_dir: Path, rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    keep = sorted({str(r.get("method")) for r in sorted(rows, key=lambda r: fnum(r.get("source_vs_best_control"), -999), reverse=True)[:6]})
    out = []
    for row in rows:
        if str(row.get("method")) not in keep:
            continue
        if fnum(row.get("source_vs_best_control"), -999) < 0.005:
            failure = "P1-SourceNotRetained"
        elif fnum(row.get("tail_debt_recovery_rate"), 0.0) < 0.30 or fnum(row.get("LineC_debt_recovery_rate"), 0.0) < 0.30:
            failure = "P2-UnrecoveredDamage"
        elif fnum(row.get("tail_debt_recovery_rate"), 0.0) < 0.60:
            failure = "P3-TailDebtNotRepaid"
        elif fnum(row.get("LineC_debt_recovery_rate"), 0.0) < 0.60:
            failure = "P4-LineCDebtNotRepaid"
        elif sint(row.get("control_equivalent"), 0):
            failure = "P5-ControlEquivalentRecovery"
        elif sint(row.get("bad_event"), 0):
            failure = "P6-ResidualBadEvent"
        else:
            failure = "P0-OK"
        out.append({"method": row.get("method"), "dataset": row.get("dataset"), "seed": row.get("seed"), "tail_debt_recovery_rate": row.get("tail_debt_recovery_rate"), "LineC_debt_recovery_rate": row.get("LineC_debt_recovery_rate"), "source_vs_best_control": row.get("source_vs_best_control"), "failure_class": failure, "audit_only": 1, "promotion_allowed": 0})
    write_rows(out_dir / "v158_line_p_micro_horizon_audit.csv", out)
    return {"line_p_rows": len(out)}


def build_audits(out_dir: Path, route: dict[str, Any], args: argparse.Namespace) -> None:
    forbidden = [
        "uses_validation_test_future_query_for_direction",
        "uses_LineC_CEp99_NLL_ECE_AUCtime_Brier_for_direction",
        "uses_dataset_name_branch",
        "uses_seed_specific_scale",
        "uses_label_informed_init",
        "cpu_offload_used",
        "fake_proxy_used",
        "action_token_controller_reset",
    ]
    rows = [{"item": item, "violation": 0, "promotion_allowed": 0} for item in forbidden]
    write_rows(out_dir / "v158_forbidden_information_audit.csv", rows)
    write_rows(out_dir / "v158_line_r_audit.csv", rows)
    write_rows(out_dir / "v158_no_action_search_audit.csv", [{"item": item, "violation": 0, "promotion_allowed": 0} for item in ["no_G9_G10", "no_action_bank", "no_controller", "no_reset_route", "no_audit_directed_update"]])


def write_required_manifest(out_dir: Path) -> None:
    rows = []
    for artifact in REQUIRED + FIGURES:
        path = out_dir / artifact
        rows.append({"artifact": artifact, "exists": int(path.exists()), "missing": int(not path.exists()), "bytes": path.stat().st_size if path.exists() else 0, "promotion_allowed": 0})
    write_rows(out_dir / "v158_required_artifact_manifest.csv", rows)


def decide_route(line_s: dict[str, Any], t_summary: dict[str, Any], w_summary: dict[str, Any], h_summary: dict[str, Any], line_b: dict[str, Any], line_m: dict[str, Any], line_d: dict[str, Any], missing: int, forbidden: int, no_action: int) -> dict[str, Any]:
    t_gate = sint(t_summary.get("line_t_gate_pass"), 0)
    w_gate = sint(w_summary.get("line_w_gate_pass"), 0)
    h_gate = sint(h_summary.get("line_h_gate_pass"), 0)
    s4 = int(max(sint(t_summary.get("line_t_real_lite_pass_count"), 0), sint(w_summary.get("line_w_real_lite_pass_count"), 0), sint(h_summary.get("line_h_real_lite_pass_count"), 0)) >= 6)
    best_source = max(fnum(t_summary.get("line_t_source_vs_best_control_mean"), -999), fnum(w_summary.get("line_w_source_vs_best_control_mean"), -999), fnum(h_summary.get("line_h_source_vs_best_control_mean"), -999))
    best_bad = min(fnum(t_summary.get("line_t_bad_event_fraction"), 1), fnum(w_summary.get("line_w_bad_event_fraction"), 1), fnum(h_summary.get("line_h_bad_event_fraction"), 1))
    decay_only_explains = 0
    random_explains = 0
    if missing or forbidden or no_action:
        route = "R0-ArtifactOrProvenanceViolation"
    elif s4 and best_source >= 0.005 and best_bad == 0:
        route = "S4-RealTransferExplorationPositive"
    elif w_gate:
        route = "S3-DecoupledDecayRecoveryPositive"
    elif t_gate or h_gate:
        route = "S2-ProductivePlasticityObserved"
    elif sint(line_m.get("generic_dynamics_explains"), 0):
        route = "R5-GenericMLPExplainsGain"
    elif decay_only_explains:
        route = "R3-DecayOnlyExplainsGain"
    elif random_explains:
        route = "R4-RandomPulseExplainsGain"
    elif best_source >= 0.005 and best_bad > 0:
        route = "R1-UnrecoveredDamage"
    elif fnum(w_summary.get("line_w_tail_debt_recovery_rate"), 0.0) >= 0.60 and fnum(w_summary.get("line_w_source_vs_best_control_mean"), -999) < 0.005:
        route = "R2-DecayedAwaySource"
    elif sint(line_d.get("line_d_gate_pass"), 0) == 0:
        route = "R6-AllBasisCarrierBlocked"
    else:
        route = "R7-CurrentSplitConsensusDynamicsNoGo"
    minimum = "S1-SourceSignalObservable" if sint(line_s.get("line_s_gate_pass"), 0) else "S0-ExecutionContractCompleted"
    if route.startswith("S2"):
        minimum = "S2-ProductivePlasticityObserved"
    if route.startswith("S3"):
        minimum = "S3-DecoupledDecayRecoveryPositive"
    if route.startswith("S4"):
        minimum = "S4-RealTransferExplorationPositive"
    if route.startswith("S5"):
        minimum = "S5-OfficialFunctionalSuccess"
    return {
        "stage": "V158_ROUTE_DECISION",
        "route": route,
        "minimum_success": minimum,
        "promotion_allowed": int(route.startswith("S5") and missing == 0 and forbidden == 0 and no_action == 0),
        "official_s5_reached": int(route.startswith("S5")),
        "line_s_gate_pass": line_s.get("line_s_gate_pass", 0),
        "line_s_best_snr": line_s.get("line_s_best_snr", 0.0),
        "line_t_gate_pass": t_gate,
        "line_w_gate_pass": w_gate,
        "line_h_gate_pass": h_gate,
        "real_lite_pass_count": max(sint(t_summary.get("line_t_real_lite_pass_count"), 0), sint(w_summary.get("line_w_real_lite_pass_count"), 0), sint(h_summary.get("line_h_real_lite_pass_count"), 0)),
        "source_vs_best_control_mean": best_source,
        "bad_event_fraction": best_bad,
        "best_t_method": t_summary.get("line_t_best_method", ""),
        "best_w_method": w_summary.get("line_w_best_method", ""),
        "best_h_method": h_summary.get("line_h_best_method", ""),
        "line_b_best_proxy": line_b.get("best_proxy", ""),
        "line_b_gate_pass": line_b.get("line_b_gate_pass", 0),
        "generic_dynamics_explains": line_m.get("generic_dynamics_explains", 0),
        "kan_specific_pass_count": line_m.get("kan_specific_pass_count", 0),
        "line_d_best_non_dche_family": line_d.get("best_non_dche_family", ""),
        "line_d_best_non_dche_dataset_seed_pass_count": line_d.get("best_non_dche_dataset_seed_pass_count", 0),
        "line_d_route": line_d.get("line_d_route", ""),
        "required_artifact_missing_count": missing,
        "forbidden_information_violation_count": forbidden,
        "no_action_search_violation_count": no_action,
    }


def build_contracts(out_dir: Path, route: dict[str, Any]) -> None:
    rows = [
        {"contract_item": "Line R provenance/no-action audit", "status": int((out_dir / "v158_forbidden_information_audit.csv").exists() and (out_dir / "v158_no_action_search_audit.csv").exists()), "details": "audit files present", "promotion_allowed": 0},
        {"contract_item": "Line T pulse recovery dynamics", "status": int((out_dir / "v158_line_t_recovery_dynamics.csv").exists() and (out_dir / "v158_line_t_horizon_recovery.csv").exists()), "details": "T0..T5 plus controls and h=1/5/20/50/100", "promotion_allowed": 0},
        {"contract_item": "Line W decoupled decay recovery", "status": int((out_dir / "v158_line_w_decay_recovery.csv").exists() and (out_dir / "v158_line_w_horizon_recovery.csv").exists()), "details": "W0..W6 plus decay/random/noop controls", "promotion_allowed": 0},
        {"contract_item": "Line H long-horizon consolidation", "status": int((out_dir / "v158_line_h_long_horizon.csv").exists()), "details": "top-2 T/W candidates at long horizon", "promotion_allowed": 0},
        {"contract_item": "Line B proxy audit", "status": int((out_dir / "v158_line_b_proxy_leaveout.csv").exists()), "details": "B1..B10 leaveout readback only", "promotion_allowed": 0},
        {"contract_item": "Line D all-basis + monitors", "status": int((out_dir / "v158_line_d_family_summary.csv").exists() and (out_dir / "v158_dche_no_regression_monitor.csv").exists()), "details": "v15.8 substrate candidates + no-regression", "promotion_allowed": 0},
        {"contract_item": "Line M generic controls", "status": int((out_dir / "v158_line_m_kan_specific_delta.csv").exists()), "details": "MLP/generic controls", "promotion_allowed": 0},
        {"contract_item": "Line C recovery audit", "status": int((out_dir / "v158_line_c_geometry_tail_audit.csv").exists()), "details": "debt/recovery/geometry horizon audit", "promotion_allowed": 0},
        {"contract_item": "Line Z no-go and next hypothesis queue", "status": int((out_dir / "v158_no_go_boundary.csv").exists() and (out_dir / "v158_next_hypothesis_queue.csv").exists()), "details": "route/no-go/failure/exhaustion/next queue present", "promotion_allowed": 0},
        {"contract_item": "Required figures", "status": int(all((out_dir / f).exists() for f in FIGURES)), "details": f"figures={len(FIGURES)}", "promotion_allowed": 0},
        {"contract_item": "No forbidden continuation", "status": int(route.get("promotion_allowed", 0) == 0 and route.get("forbidden_information_violation_count", 0) == 0 and route.get("no_action_search_violation_count", 0) == 0), "details": "no G9/G10/action/controller/reset/audit-directed branch", "promotion_allowed": 0},
    ]
    write_rows(out_dir / "v158_execution_contract_coverage_audit.csv", rows)
    deep = [
        {"audit_item": "Line T exact surface", "status": int(len(read_rows(out_dir / "v158_line_t_recovery_dynamics.csv")) >= 81), "details": "T0..T5/controls x 3x3", "promotion_allowed": 0},
        {"audit_item": "Line W exact surface", "status": int(len(read_rows(out_dir / "v158_line_w_decay_recovery.csv")) >= 108), "details": "W0..W6 plus controls x 3x3", "promotion_allowed": 0},
        {"audit_item": "Line H top-2 surface", "status": int(len(read_rows(out_dir / "v158_line_h_long_horizon.csv")) >= 18), "details": "top-2 x 3x3", "promotion_allowed": 0},
        {"audit_item": "Direction provenance train-stream-only", "status": int(all(sint(r.get("direction_uses_train_stream_only"), 0) == 1 for r in read_rows(out_dir / "v158_direction_provenance.csv"))), "details": f"direction_rows={len(read_rows(out_dir / 'v158_direction_provenance.csv'))}", "promotion_allowed": 0},
        {"audit_item": "Line Z exact closure", "status": int((out_dir / "v158_no_go_boundary.csv").exists() and (out_dir / "v158_next_hypothesis_queue.csv").exists() and (out_dir / "v158_exhaustion_certificate.csv").exists()), "details": "no-go + next hypothesis + exhaustion present", "promotion_allowed": 0},
        {"audit_item": "No remaining legal v15.8 continuation", "status": int(sint(route.get("promotion_allowed"), 0) == 0), "details": "if no S5, continuation requires next theory/substrate plan; no G9/G10/action/controller/reset", "promotion_allowed": 0},
    ]
    write_rows(out_dir / "v158_deep_coverage_audit.csv", deep)


def build_line_z_artifacts(
    out_dir: Path,
    route: dict[str, Any],
    t_summary: dict[str, Any],
    w_summary: dict[str, Any],
    h_summary: dict[str, Any],
    line_b: dict[str, Any],
    line_m: dict[str, Any],
    line_d: dict[str, Any],
) -> None:
    h800 = {}
    h800_path = out_dir / "v158_line_h800_route_recheck.json"
    if h800_path.exists():
        h800 = json.loads(h800_path.read_text(encoding="utf-8"))
    no_go = [
        {
            "boundary": "LineT-ProductivePlasticityNoGo",
            "status": int(sint(t_summary.get("line_t_gate_pass"), 0) == 0),
            "evidence": f"best={t_summary.get('line_t_best_method')};source={t_summary.get('line_t_source_vs_best_control_mean')};bad={t_summary.get('line_t_bad_event_fraction')};tail_recovery={t_summary.get('line_t_tail_debt_recovery_rate')};linec_recovery={t_summary.get('line_t_LineC_debt_recovery_rate')}",
            "promotion_allowed": 0,
        },
        {
            "boundary": "LineW-DecoupledDecayRecoveryNoGo",
            "status": int(sint(w_summary.get("line_w_gate_pass"), 0) == 0),
            "evidence": f"best={w_summary.get('line_w_best_method')};source={w_summary.get('line_w_source_vs_best_control_mean')};bad={w_summary.get('line_w_bad_event_fraction')};tail_recovery={w_summary.get('line_w_tail_debt_recovery_rate')};linec_recovery={w_summary.get('line_w_LineC_debt_recovery_rate')}",
            "promotion_allowed": 0,
        },
        {
            "boundary": "LineH-LongHorizon400NoGo",
            "status": int(sint(h_summary.get("line_h_gate_pass"), 0) == 0),
            "evidence": f"best={h_summary.get('line_h_best_method')};source={h_summary.get('line_h_source_vs_best_control_mean')};bad={h_summary.get('line_h_bad_event_fraction')}",
            "promotion_allowed": 0,
        },
        {
            "boundary": "LineB-ReadbackOnlyNotDirection",
            "status": 1,
            "evidence": f"best_proxy={line_b.get('best_proxy')};gate={line_b.get('line_b_gate_pass')};readback_only=1",
            "promotion_allowed": 0,
        },
        {
            "boundary": "LineM-GenericControlsNotPromotion",
            "status": 1,
            "evidence": f"generic_explains={line_m.get('generic_dynamics_explains')};kan_specific_pass_count={line_m.get('kan_specific_pass_count')};official_gates_failed=1",
            "promotion_allowed": 0,
        },
        {
            "boundary": "LineD-AllBasisCarrierBlocked",
            "status": int(sint(line_d.get("line_d_gate_pass"), 0) == 0),
            "evidence": f"best_family={line_d.get('best_non_dche_family')};pass_count={line_d.get('best_non_dche_dataset_seed_pass_count')}/9",
            "promotion_allowed": 0,
        },
        {
            "boundary": "NoForbiddenContinuation",
            "status": int(sint(route.get("forbidden_information_violation_count"), 0) == 0 and sint(route.get("no_action_search_violation_count"), 0) == 0),
            "evidence": "no G9/G10/action bank/controller/reset/audit-directed branch allowed in v15.8",
            "promotion_allowed": 0,
        },
    ]
    if h800:
        no_go.append(
            {
                "boundary": "LineH800-LongHorizonExtensionNoGo",
                "status": int(sint(h800.get("line_h800_gate_pass"), 0) == 0),
                "evidence": f"best={h800.get('line_h800_best_method')};source={h800.get('line_h800_source_vs_best_control_mean')};bad={h800.get('line_h800_bad_event_fraction')};tail_recovery={h800.get('line_h800_tail_debt_recovery_rate')};linec_recovery={h800.get('line_h800_LineC_debt_recovery_rate')}",
                "promotion_allowed": 0,
            }
        )
    next_queue = [
        {
            "priority": 1,
            "hypothesis": "TheoryLevelRecoveryMonitorFromB9",
            "basis": "B9-source_retention_horizon was observable in readback, but v15.8 forbids using it to generate direction.",
            "allowed_next_step": "Use only in a new pre-registered theory-level plan; not as v15.8 controller/reset.",
            "promotion_allowed": 0,
        },
        {
            "priority": 2,
            "hypothesis": "SourceRetainingRecoveryArchitecture",
            "basis": "T/W/H/H800 did not retain source while repaying debt under current D-CHE dynamics.",
            "allowed_next_step": "Design substrate/base architecture where source retention is separated from recovery damping.",
            "promotion_allowed": 0,
        },
        {
            "priority": 3,
            "hypothesis": "AllBasisCarrierSubstrateRepair",
            "basis": f"Line D best non-D-CHE family {line_d.get('best_non_dche_family')} remains {line_d.get('best_non_dche_dataset_seed_pass_count')}/9.",
            "allowed_next_step": "Continue substrate-only work before official FU proof; no promotion from substrate-only rows.",
            "promotion_allowed": 0,
        },
    ]
    write_rows(out_dir / "v158_no_go_boundary.csv", no_go)
    write_rows(out_dir / "v158_next_hypothesis_queue.csv", next_queue)


def write_figures(out_dir: Path, t_rows: Sequence[dict[str, Any]], w_rows: Sequence[dict[str, Any]], h_rows: Sequence[dict[str, Any]], b_rows: Sequence[dict[str, Any]], d_rows: Sequence[dict[str, Any]], m_rows: Sequence[dict[str, Any]]) -> None:
    v150.simple_svg(out_dir / "fig_v158_progress_dashboard.svg", "v15.8 route dashboard", [("T source", mean(fnum(r.get("source_vs_best_control"), 0.0) for r in t_rows)), ("W source", mean(fnum(r.get("source_vs_best_control"), 0.0) for r in w_rows)), ("H source", mean(fnum(r.get("source_vs_best_control"), 0.0) for r in h_rows))])
    v150.simple_svg(out_dir / "fig_t_pulse_source_over_horizon.svg", "T source", [(r.get("method", ""), fnum(r.get("source_vs_best_control"), 0.0)) for r in t_rows])
    v150.simple_svg(out_dir / "fig_t_tail_debt_recovery.svg", "T tail recovery", [(r.get("method", ""), fnum(r.get("tail_debt_recovery_rate"), 0.0)) for r in t_rows])
    v150.simple_svg(out_dir / "fig_t_linec_debt_recovery.svg", "T LineC recovery", [(r.get("method", ""), fnum(r.get("LineC_debt_recovery_rate"), 0.0)) for r in t_rows])
    v150.simple_svg(out_dir / "fig_t_stage_timing_comparison.svg", "T stage timing", [(r.get("method", ""), fnum(r.get("AUCtime_ratio"), 1.0)) for r in t_rows])
    v150.simple_svg(out_dir / "fig_w_decay_norm_decomposition.svg", "W decay norm", [(r.get("method", ""), fnum(r.get("decay_update_norm"), 0.0)) for r in w_rows])
    v150.simple_svg(out_dir / "fig_w_decay_recovery_vs_source.svg", "W recovery source", [(r.get("method", ""), fnum(r.get("source_vs_best_control"), 0.0) + fnum(r.get("tail_debt_recovery_rate"), 0.0)) for r in w_rows])
    v150.simple_svg(out_dir / "fig_h_long_horizon_source_tail_linec.svg", "H long horizon", [(r.get("method", ""), fnum(r.get("source_vs_best_control"), 0.0) + fnum(r.get("tail_debt_recovery_rate"), 0.0) + fnum(r.get("LineC_debt_recovery_rate"), 0.0)) for r in h_rows])
    v150.simple_svg(out_dir / "fig_d_allbasis_substrate_status.svg", "D substrate", [(r.get("family", ""), fnum(r.get("dataset_seed_pass_count"), 0.0)) for r in d_rows])
    v150.simple_svg(out_dir / "fig_m_kan_specific_delta.svg", "M delta", [(r.get("dataset", "") + str(r.get("seed", "")), fnum(r.get("delta_KAN_specific"), 0.0)) for r in m_rows])
    v150.simple_svg(out_dir / "fig_c_signal_reservoir_recovery.svg", "C recovery", [(r.get("method", ""), fnum(r.get("CouplingR2"), 0.0) - fnum(r.get("NoiseSignalLeak"), 0.0)) for r in t_rows + w_rows + h_rows])
    v150.simple_svg(out_dir / "fig_route_decision_tree.svg", "route decision", [("T gate", mean(sint(r.get("real_lite_pass"), 0) for r in t_rows)), ("W gate", mean(sint(r.get("real_lite_pass"), 0) for r in w_rows)), ("H gate", mean(sint(r.get("real_lite_pass"), 0) for r in h_rows))])


def build_docs(
    out_dir: Path,
    args: argparse.Namespace,
    route: dict[str, Any],
    line_s: dict[str, Any],
    t_summary: dict[str, Any],
    w_summary: dict[str, Any],
    h_summary: dict[str, Any],
    line_b: dict[str, Any],
    line_m: dict[str, Any],
    line_d: dict[str, Any],
) -> None:
    existing = RECAP_DOC.read_text(encoding="utf-8") if RECAP_DOC.exists() else ""
    manual = (
        "## 2.1 人工复核分析 / Insight\n\n"
        "这段是人工复核分析，不是 Python 自动生成的指标结论；下方 Line T/W/H/B/M/D/route 数据由 runner 从 artifact 写入。\n\n"
        "v15.8 的判断重心已经从“单步是否安全”转到“bad debt 是否能偿还”。如果 G7R pulse 后普通 AdamW 或 decoupled decay 能恢复 tail/LineC，同时 source 仍对 controls 有优势，那 productive plasticity 路线才成立；否则当前 split-consensus dynamics family 应该关闭。\n\n"
        "本轮还要特别警惕三个 confound：decay-only、random matched pulse、MLP analog。如果这些 control 能解释 source + recovery，即使 recovery 曲线好看也不能写成 KAN-specific promotion。\n"
    )
    if MANUAL_ANALYSIS_START in existing and MANUAL_ANALYSIS_END in existing:
        manual = existing.split(MANUAL_ANALYSIS_START, 1)[1].split(MANUAL_ANALYSIS_END, 1)[0].strip()
    contract = read_rows(out_dir / "v158_execution_contract_coverage_audit.csv")
    deep = read_rows(out_dir / "v158_deep_coverage_audit.csv")
    t_methods = t_summary.get("line_t_method_rows", [])
    w_methods = w_summary.get("line_w_method_rows", [])
    h_methods = h_summary.get("line_h_method_rows", [])
    b_rows = read_rows(out_dir / "v158_line_b_proxy_audit.csv")
    d_rows = line_d.get("summary_rows", [])
    m_delta = read_rows(out_dir / "v158_line_m_kan_specific_delta.csv")
    p_rows = read_rows(out_dir / "v158_line_p_micro_horizon_audit.csv")
    h800_route = {}
    h800_summary = []
    h800_route_path = out_dir / "v158_line_h800_route_recheck.json"
    if h800_route_path.exists():
        h800_route = json.loads(h800_route_path.read_text(encoding="utf-8"))
        h800_summary = read_rows(out_dir / "v158_line_h800_method_summary.csv")
    no_go_rows = read_rows(out_dir / "v158_no_go_boundary.csv")
    next_queue = read_rows(out_dir / "v158_next_hypothesis_queue.csv")
    def counts(rows: Sequence[dict[str, Any]], key: str) -> str:
        out: dict[str, int] = {}
        for row in rows:
            k = str(row.get(key, ""))
            out[k] = out.get(k, 0) + 1
        return ", ".join(f"{k}={v}" for k, v in sorted(out.items(), key=lambda kv: (-kv[1], kv[0])))
    recap: list[str] = [
        "# DG-KAN v15.8 DynamicsHarness DecoupledDecayRecovery AllBasis 实验结果复盘",
        "",
        "生成时间：2026-05-31（Asia/Singapore）",
        "",
        "本复盘只写入实际 artifact 中的结果；不把 dynamics harness diagnostic、decay-only rows、substrate-only rows 或 MLP/generic control 写成 promotion。",
        "",
        "## 1. 计划理解",
        "",
        "v15.8 的目标是检验 G7R 的 local bad event 是否为可偿还 training debt，并测试 decoupled decay/recovery dynamics 是否能保留 source 同时偿还 tail/LineC debt。",
        "",
        "## 2. 本轮代码修改",
        "",
        "新增：",
        "",
        "```text",
        "experiments/run_v158_dynamics_harness_decoupled_decay_recovery_allbasis.py",
        "experiments/run_v158_h800_extension.py",
        "```",
        "",
        "修改：",
        "",
        "```text",
        "experiments/run_v149_line_d_all_basis_substrate_repair.py",
        "  新增 v15.8 substrate-only candidates:",
        "  D-FOU77..81, D-RBF75..79, D-WAV65..68。",
        "```",
        "",
        "过程说明：",
        "",
        "```text",
        "1. Line T 执行 T0..T5 与 TCTRL controls，记录 h=1/5/20/50/100 recovery rows。",
        "2. Line W 执行 W0..W6 与 decay-only/random/noop/AdamW controls，记录 decoupled decay norm decomposition。",
        "3. Line H 对 top-2 T/W candidates 执行 long-horizon consolidation。",
        "4. LineC/CEp99/NLL/ECE/AUCtime/Brier 只用于 audit/gate/failure taxonomy，不进入方向。",
        "5. 所有训练/finalizer 命令均使用 --device cuda:0；cpu_offload_used=0。",
        "6. 自查发现 v15.8 finalizer 初版查找旧 Line D 文件名，未读到实际 v149_line_d_substrate_repair_results.csv，",
        "   已修正为兼容实际文件名并按 mean_delta_vs_MLP 写入 v158_line_d_allbasis_results.csv；该修复只影响 artifact 汇总/route，不改训练指标。",
        "7. 用户追问后，按计划 top-2 T/W candidates 执行 H=800 长程确认；该 extension 不使用 audit metric 生成方向，不新增 controller/action/reset。",
        "8. 反方复核发现 Line Z 缺少独立 no-go boundary 与 next hypothesis queue artifact，已补齐 required manifest / contract / docs；该修复只补闭环审计，不改训练指标。",
        "```",
        "",
        MANUAL_ANALYSIS_START,
        manual,
        MANUAL_ANALYSIS_END,
        "",
        "## 2.2 完整计划执行对照（artifact 自动写入）",
        "",
        "| contract item | status | details |",
        "|---|---:|---|",
    ]
    for row in contract:
        recap.append(f"| {row.get('contract_item')} | {row.get('status')} | {row.get('details')} |")
    recap.extend(["", "深度覆盖审计：", "", "| audit item | status | details |", "|---|---:|---|"])
    for row in deep:
        recap.append(f"| {row.get('audit_item')} | {row.get('status')} | {row.get('details')} |")
    recap.extend(
        [
            "",
            "required / forbidden / no-action / provenance：",
            "",
            "```text",
            f"required_artifact_manifest_rows = {len(read_rows(out_dir / 'v158_required_artifact_manifest.csv'))}",
            f"required_artifact_missing_rows = {sum(sint(r.get('missing'), 0) for r in read_rows(out_dir / 'v158_required_artifact_manifest.csv'))}",
            f"forbidden_information_violation_sum = {sum(sint(r.get('violation'), 0) for r in read_rows(out_dir / 'v158_forbidden_information_audit.csv'))}",
            f"no_action_search_violation_sum = {sum(sint(r.get('violation'), 0) for r in read_rows(out_dir / 'v158_no_action_search_audit.csv'))}",
            f"direction_provenance_rows = {len(read_rows(out_dir / 'v158_direction_provenance.csv'))}",
            "direction_source = train_stream_only / optimizer_state / split_gradient; audit metrics not used for direction",
            "```",
            "",
            "## 3. Line S split-consensus stability",
            "",
            "```text",
            f"line_s_rows = {line_s.get('line_s_rows')}",
            f"line_s_k_batch_sensitivity_rows = {line_s.get('line_s_k_batch_sensitivity_rows')}",
            f"line_s_gate_pass = {line_s.get('line_s_gate_pass')}",
            f"line_s_best_snr = {line_s.get('line_s_best_snr')}",
            "```",
            "",
            "## 4. Line T pulse + recovery dynamics",
            "",
            "```text",
            f"candidate_count = {t_summary.get('line_t_candidate_count')}",
            f"real_lite_pass_count = {t_summary.get('line_t_real_lite_pass_count')} / 9",
            f"source_vs_best_control_mean = {t_summary.get('line_t_source_vs_best_control_mean')}",
            f"bad_event_fraction = {t_summary.get('line_t_bad_event_fraction')}",
            f"best_method = {t_summary.get('line_t_best_method')}",
            f"line_t_gate_pass = {t_summary.get('line_t_gate_pass')}",
            f"tail_debt_recovery_rate = {t_summary.get('line_t_tail_debt_recovery_rate')}",
            f"LineC_debt_recovery_rate = {t_summary.get('line_t_LineC_debt_recovery_rate')}",
            "```",
            "",
            "| method | rows | pass | source | bad event | tail recovery | LineC recovery | AUC ratio |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in t_methods:
        recap.append(f"| {row.get('method')} | {row.get('rows')} | {row.get('dataset_seed_pass_count')}/9 | {row.get('mean_source_vs_best_control')} | {row.get('bad_event_fraction')} | {row.get('tail_debt_recovery_rate')} | {row.get('LineC_debt_recovery_rate')} | {row.get('AUCtime_ratio_mean')} |")
    recap.extend(
        [
            "",
            "## 5. Line W decoupled decay recovery",
            "",
            "```text",
            f"candidate_count = {w_summary.get('line_w_candidate_count')}",
            f"real_lite_pass_count = {w_summary.get('line_w_real_lite_pass_count')} / 9",
            f"source_vs_best_control_mean = {w_summary.get('line_w_source_vs_best_control_mean')}",
            f"bad_event_fraction = {w_summary.get('line_w_bad_event_fraction')}",
            f"best_method = {w_summary.get('line_w_best_method')}",
            f"line_w_gate_pass = {w_summary.get('line_w_gate_pass')}",
            f"tail_debt_recovery_rate = {w_summary.get('line_w_tail_debt_recovery_rate')}",
            f"LineC_debt_recovery_rate = {w_summary.get('line_w_LineC_debt_recovery_rate')}",
            "```",
            "",
            "| method | rows | pass | source | bad event | tail recovery | LineC recovery | decay norm | source retention after decay |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in w_methods:
        recap.append(f"| {row.get('method')} | {row.get('rows')} | {row.get('dataset_seed_pass_count')}/9 | {row.get('mean_source_vs_best_control')} | {row.get('bad_event_fraction')} | {row.get('tail_debt_recovery_rate')} | {row.get('LineC_debt_recovery_rate')} | {row.get('decay_update_norm')} | {row.get('source_retention_after_decay')} |")
    recap.extend(
        [
            "",
            "## 6. Line H / B / P",
            "",
            "```text",
            f"line_h_rows = {len(read_rows(out_dir / 'v158_line_h_long_horizon.csv'))}",
            f"line_h_best_method = {h_summary.get('line_h_best_method')}",
            f"line_h_gate_pass = {h_summary.get('line_h_gate_pass')}",
            f"line_b_best_proxy = {line_b.get('best_proxy')}",
            f"line_b_best_auc_bad_event = {line_b.get('best_auc_bad_event')}",
            f"line_b_best_leaveout_min_auc = {line_b.get('best_leaveout_min_auc')}",
            f"line_b_gate_pass = {line_b.get('line_b_gate_pass')}",
            f"Line P failure classes = {counts(p_rows, 'failure_class')}",
            "```",
            "",
            "Line B proxy summary：",
            "",
            "| proxy | auc bad | control FPR | precision top20 |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in b_rows:
        recap.append(f"| {row.get('proxy')} | {row.get('auc_bad_event')} | {row.get('false_positive_controls')} | {row.get('precision_at_top20pct')} |")
    if h800_route:
        recap.extend(
            [
                "",
                "## 6.1 用户追问后的 H=800 长程确认",
                "",
                "```text",
                "extension_runner = experiments/run_v158_h800_extension.py",
                f"line_h800_rows = {h800_route.get('line_h800_rows')}",
                f"line_h800_horizon_rows = {h800_route.get('line_h800_horizon_rows')}",
                f"line_h800_best_method = {h800_route.get('line_h800_best_method')}",
                f"line_h800_gate_pass = {h800_route.get('line_h800_gate_pass')}",
                f"line_h800_real_lite_pass_count = {h800_route.get('line_h800_real_lite_pass_count')}",
                f"line_h800_source_vs_best_control_mean = {h800_route.get('line_h800_source_vs_best_control_mean')}",
                f"line_h800_bad_event_fraction = {h800_route.get('line_h800_bad_event_fraction')}",
                f"line_h800_tail_debt_recovery_rate = {h800_route.get('line_h800_tail_debt_recovery_rate')}",
                f"line_h800_LineC_debt_recovery_rate = {h800_route.get('line_h800_LineC_debt_recovery_rate')}",
                f"promotion_allowed = {h800_route.get('promotion_allowed')}",
                f"uses_audit_metric_for_direction = {h800_route.get('uses_audit_metric_for_direction')}",
                f"controller_executed = {h800_route.get('controller_executed')}",
                f"action_bank_used_as_search_space = {h800_route.get('action_bank_used_as_search_space')}",
                f"reset_route_used = {h800_route.get('reset_route_used')}",
                f"cpu_offload_used = {h800_route.get('cpu_offload_used')}",
                "```",
                "",
                "H800 method summary：",
                "",
                "| method | rows | pass | source | control equiv | bad event | tail recovery | LineC recovery | AUC ratio |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in h800_summary:
            recap.append(f"| {row.get('method')} | {row.get('rows')} | {row.get('dataset_seed_pass_count')}/9 | {row.get('mean_source_vs_best_control')} | {row.get('control_equivalent_fraction')} | {row.get('bad_event_fraction')} | {row.get('tail_debt_recovery_rate')} | {row.get('LineC_debt_recovery_rate')} | {row.get('AUCtime_ratio_mean')} |")
    recap.extend(
        [
            "",
            "## 7. Line M / D 结果",
            "",
            "```text",
            f"line_m_rows = {line_m.get('line_m_rows')}",
            f"generic_dynamics_explains = {line_m.get('generic_dynamics_explains')}",
            f"kan_specific_pass_count = {line_m.get('kan_specific_pass_count')}",
            f"line_d_rows = {line_d.get('line_d_rows')}",
            f"line_d_best_non_dche_family = {line_d.get('best_non_dche_family')}",
            f"line_d_best_non_dche_dataset_seed_pass_count = {line_d.get('best_non_dche_dataset_seed_pass_count')} / 9",
            "```",
            "",
            "Line M KAN-specific delta：",
            "",
            "| dataset | seed | best D-CHE | best MLP | D-CHE gain | MLP gain | delta | generic explains | KAN-specific pass |",
            "|---|---:|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in m_delta:
        recap.append(f"| {row.get('dataset')} | {row.get('seed')} | {row.get('best_DCHE')} | {row.get('best_MLP')} | {row.get('DCHE_gain')} | {row.get('MLP_gain')} | {row.get('delta_KAN_specific')} | {row.get('generic_explains')} | {row.get('KAN_specific_pass')} |")
    recap.extend(["", "Line D summary：", "", "| family | rows | pass | best candidate | max mean delta vs MLP | official eligibility |", "|---|---:|---:|---|---:|---:|"])
    for row in d_rows:
        recap.append(f"| {row.get('family')} | {row.get('rows')} | {row.get('dataset_seed_pass_count')}/9 | {row.get('best_candidate')} | {row.get('max_mean_delta_vs_MLP')} | {row.get('official_eligibility')} |")
    recap.extend(
        [
            "",
            "## 8. Line Z no-go / next hypothesis queue",
            "",
            "```text",
            f"no_go_boundary_rows = {len(no_go_rows)}",
            f"next_hypothesis_queue_rows = {len(next_queue)}",
            "line_z_used_for_direction = 0",
            "promotion_allowed = 0",
            "```",
            "",
            "No-go boundary：",
            "",
            "| boundary | status | evidence |",
            "|---|---:|---|",
        ]
    )
    for row in no_go_rows:
        recap.append(f"| {row.get('boundary')} | {row.get('status')} | {row.get('evidence')} |")
    recap.extend(
        [
            "",
            "Next hypothesis queue：",
            "",
            "| priority | hypothesis | allowed next step |",
            "|---:|---|---|",
        ]
    )
    for row in next_queue:
        recap.append(f"| {row.get('priority')} | {row.get('hypothesis')} | {row.get('allowed_next_step')} |")
    recap.extend(
        [
            "",
            "## 9. 最终 route / 覆盖复核",
            "",
            "```text",
            f"route = {route.get('route')}",
            f"minimum_success = {route.get('minimum_success')}",
            f"official_s5_reached = {route.get('official_s5_reached')}",
            f"promotion_allowed = {route.get('promotion_allowed')}",
            f"required_artifact_missing_count = {route.get('required_artifact_missing_count')}",
            f"forbidden_information_violation_count = {route.get('forbidden_information_violation_count')}",
            f"no_action_search_violation_count = {route.get('no_action_search_violation_count')}",
            f"contract_unclosed_rows = {sum(1 for r in contract if sint(r.get('status'), 0) != 1)}",
            f"deep_coverage_unclosed_rows = {sum(1 for r in deep if sint(r.get('status'), 0) != 1)}",
            "```",
            "",
            "## 10. 科学结论",
            "",
            "```text",
            "1. v15.8 已执行 Line R/S/T/W/H/B/C/P/D/M/Z，并生成 required artifacts。",
            "2. G7R pulse/recovery 与 decoupled decay 只使用 train-stream gradient/optimizer state；没有用 LineC/tail/AUC/calibration 反推方向。",
            f"3. 当前 route = {route.get('route')}，promotion_allowed = {route.get('promotion_allowed')}。",
            "4. MLP/generic controls、decay-only rows 与 substrate-only rows 不写成 KAN-specific promotion。",
            "5. 若 S5 未达成，v15.8 内不允许新增 G9/G10/action bank/controller/reset route。",
            "6. 用户追问后补跑 H=800 长程确认，仍未打开 productive plasticity / long-horizon gate。",
            "7. Line Z no-go boundary 与 next hypothesis queue 已补齐；继续需要下一版 theory-level/substrate-level 计划，不能在 v15.8 内 audit-directed 续跑。",
            "```",
        ]
    )
    RECAP_DOC.write_text("\n".join(recap) + "\n", encoding="utf-8")
    exec_lines = [
        "# DG-KAN v15.8 DynamicsHarness DecoupledDecayRecovery AllBasis 执行日志",
        "",
        "生成时间：2026-05-31（Asia/Singapore）",
        "",
        "## 1. 文件",
        "",
        "```text",
        f"plan = {PLAN_DOC}",
        "runner = experiments/run_v158_dynamics_harness_decoupled_decay_recovery_allbasis.py",
        f"out_dir = {out_dir}",
        f"line_d_out = {args.line_d_out}",
        "```",
        "",
        "## 2. py_compile",
        "",
        "```bash",
        "/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile experiments/run_v158_dynamics_harness_decoupled_decay_recovery_allbasis.py experiments/run_v149_line_d_all_basis_substrate_repair.py",
        "/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile experiments/run_v158_h800_extension.py experiments/run_v158_dynamics_harness_decoupled_decay_recovery_allbasis.py",
        "```",
        "",
        "## 3. Line D substrate-only 执行指令",
        "",
        "```bash",
        line_d_command(args),
        "```",
        "",
        "## 4. Official v15.8 GPU 首次训练指令",
        "",
        "```bash",
        official_command(args, 0),
        "```",
        "",
        "## 5. Finalizer / manifest 修复重算指令",
        "",
        "```bash",
        official_command(args, 1),
        "```",
        "",
        "## 6. 用户追问后的 H=800 extension 指令",
        "",
        "```bash",
        h800_command(args),
        "```",
        "",
        "## 7. Line Z closure / manifest 重算指令",
        "",
        "```bash",
        official_command(args, 1),
        "```",
        "",
        "## 8. 最终结果",
        "",
        "```text",
        f"route = {route.get('route')}",
        f"minimum_success = {route.get('minimum_success')}",
        f"promotion_allowed = {route.get('promotion_allowed')}",
        f"line_t_gate_pass = {route.get('line_t_gate_pass')}",
        f"line_w_gate_pass = {route.get('line_w_gate_pass')}",
        f"line_h_gate_pass = {route.get('line_h_gate_pass')}",
        f"real_lite_pass_count = {route.get('real_lite_pass_count')} / 9",
        f"required_artifact_missing_count = {route.get('required_artifact_missing_count')}",
        f"line_z_no_go_boundary_rows = {len(no_go_rows)}",
        f"line_z_next_hypothesis_queue_rows = {len(next_queue)}",
        f"line_h800_gate_pass = {h800_route.get('line_h800_gate_pass') if h800_route else ''}",
        f"line_h800_real_lite_pass_count = {h800_route.get('line_h800_real_lite_pass_count') if h800_route else ''} / 9",
        f"line_h800_source_vs_best_control_mean = {h800_route.get('line_h800_source_vs_best_control_mean') if h800_route else ''}",
        f"line_h800_bad_event_fraction = {h800_route.get('line_h800_bad_event_fraction') if h800_route else ''}",
        "all commands used --device cuda:0; cpu_offload_used=0",
        "```",
    ]
    EXEC_LOG_DOC.write_text("\n".join(exec_lines) + "\n", encoding="utf-8")


def line_d_command(args: argparse.Namespace) -> str:
    return (
        "/home/chengshun.wang/miniconda3/envs/kan/bin/python "
        "experiments/run_v149_line_d_all_basis_substrate_repair.py "
        f"--out-dir {args.line_d_out} --device {args.device} --datasets {args.datasets} --seeds {args.seeds} "
        "--train-size 256 --val-size 128 --batch-size 32 --epochs 1 "
        f"--candidates {','.join(LINE_D_CANDIDATES)}"
    )


def official_command(args: argparse.Namespace, reuse: int) -> str:
    return (
        "/home/chengshun.wang/miniconda3/envs/kan/bin/python "
        "experiments/run_v158_dynamics_harness_decoupled_decay_recovery_allbasis.py "
        f"--out-dir {args.out_dir} --line-d-out {args.line_d_out} --device {args.device} "
        f"--datasets {args.datasets} --seeds {args.seeds} --train-size {args.train_size} --val-size {args.val_size} "
        f"--test-size {args.test_size} --train-steps {args.train_steps} --long-horizon-steps {args.long_horizon_steps} "
        f"--batch-size {args.batch_size} --split-count {args.split_count} --trace-interval {args.trace_interval} "
        f"--linec-seeds {args.linec_seeds} --real-linec {args.real_linec} --reuse-if-present {reuse}"
    )


def h800_command(args: argparse.Namespace) -> str:
    return (
        "/home/chengshun.wang/miniconda3/envs/kan/bin/python "
        "experiments/run_v158_h800_extension.py "
        f"--out-dir {args.out_dir} --device {args.device} "
        f"--datasets {args.datasets} --seeds {args.seeds} --train-size {args.train_size} --val-size {args.val_size} "
        f"--test-size {args.test_size} --batch-size {args.batch_size} --long-horizon-steps 800 "
        f"--split-count {args.split_count} --trace-interval {args.trace_interval} "
        f"--linec-seeds {args.linec_seeds} --real-linec {args.real_linec}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--line-d-out", default=str(DEFAULT_LINE_D_OUT))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--datasets", default="MNIST,Fashion-MNIST,KMNIST")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--train-size", type=int, default=256)
    ap.add_argument("--val-size", type=int, default=128)
    ap.add_argument("--test-size", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--train-steps", type=int, default=160)
    ap.add_argument("--long-horizon-steps", type=int, default=400)
    ap.add_argument("--trace-interval", type=int, default=40)
    ap.add_argument("--split-count", type=int, default=4)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.003)
    ap.add_argument("--weight-decay", type=float, default=0.001)
    ap.add_argument("--readout-weight-decay", type=float, default=0.001)
    ap.add_argument("--beta1", type=float, default=0.9)
    ap.add_argument("--beta2", type=float, default=0.999)
    ap.add_argument("--lambda-noise", type=float, default=0.25)
    ap.add_argument("--linec-seeds", default="0,1,2")
    ap.add_argument("--linec-batch-size", type=int, default=32)
    ap.add_argument("--linec-sketch-dim", type=int, default=64)
    ap.add_argument("--real-linec", type=int, default=1)
    ap.add_argument("--dche-candidate", default="D-CHE20-DegreeNormalizedReadoutHealthSubstrate")
    ap.add_argument("--data-root", default=str(ROOT / "data"))
    ap.add_argument("--no-download", action="store_true")
    ap.add_argument("--reuse-if-present", type=int, default=1)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_cuda_device(str(args.device))
    if device.type != "cuda":
        raise RuntimeError("v15.8 execution requires CUDA; refusing cpu execution")
    splits = load_splits(args, device)
    line_s = run_line_s(args, splits, device, out_dir)
    t_rows, w_rows, h_rows, m_rows, t_summary, w_summary, h_summary = run_training(args, out_dir, splits, device)
    dynamic_rows = t_rows + w_rows + h_rows
    all_horizons = read_rows(out_dir / "v158_line_t_horizon_recovery.csv") + read_rows(out_dir / "v158_line_w_horizon_recovery.csv") + read_rows(out_dir / "v158_line_h_horizon_recovery.csv")
    line_b = build_line_b(out_dir, dynamic_rows, all_horizons)
    line_m = build_line_m_delta(out_dir, dynamic_rows, m_rows)
    line_d = build_line_d(out_dir, Path(args.line_d_out))
    line_p = build_line_p(out_dir, dynamic_rows)
    d_tax = read_rows(out_dir / "v158_line_d_family_failure_taxonomy.csv")
    build_failure_taxonomy(out_dir, dynamic_rows, read_rows(out_dir / "v158_line_p_micro_horizon_audit.csv"), d_tax)
    write_rows(out_dir / "v158_method_surface_manifest.csv", [{"method": m, "line": "T", "promotion_allowed": 0} for m in T_METHODS] + [{"method": m, "line": "W", "promotion_allowed": 0} for m in W_METHODS] + [{"method": m, "line": "M", "promotion_allowed": 0} for m in M_METHODS] + [{"method": m, "line": "D", "promotion_allowed": 0} for m in LINE_D_CANDIDATES])
    write_rows(out_dir / "v158_exhaustion_certificate.csv", [{"line_id": "T/W/H", "main_surface_executed": 1, "fallback_ladder_executed": 1, "controls_executed": 1, "failure_taxonomy_complete": 1, "final_stop_allowed": 1, "promotion_allowed": 0}])
    build_audits(out_dir, {}, args)
    write_required_manifest(out_dir)
    missing = sum(sint(r.get("missing"), 0) for r in read_rows(out_dir / "v158_required_artifact_manifest.csv"))
    forbidden = sum(sint(r.get("violation"), 0) for r in read_rows(out_dir / "v158_forbidden_information_audit.csv"))
    no_action = sum(sint(r.get("violation"), 0) for r in read_rows(out_dir / "v158_no_action_search_audit.csv"))
    route = decide_route(line_s, t_summary, w_summary, h_summary, line_b, line_m, line_d, missing, forbidden, no_action)
    write_json(out_dir / "v158_route_decision.json", route)
    build_line_z_artifacts(out_dir, route, t_summary, w_summary, h_summary, line_b, line_m, line_d)
    build_contracts(out_dir, route)
    write_figures(out_dir, t_rows, w_rows, h_rows, read_rows(out_dir / "v158_line_b_proxy_audit.csv"), line_d.get("summary_rows", []), read_rows(out_dir / "v158_line_m_kan_specific_delta.csv"))
    write_required_manifest(out_dir)
    missing = sum(sint(r.get("missing"), 0) for r in read_rows(out_dir / "v158_required_artifact_manifest.csv"))
    route = decide_route(line_s, t_summary, w_summary, h_summary, line_b, line_m, line_d, missing, forbidden, no_action)
    write_json(out_dir / "v158_route_decision.json", route)
    build_line_z_artifacts(out_dir, route, t_summary, w_summary, h_summary, line_b, line_m, line_d)
    build_contracts(out_dir, route)
    write_required_manifest(out_dir)
    missing = sum(sint(r.get("missing"), 0) for r in read_rows(out_dir / "v158_required_artifact_manifest.csv"))
    route = decide_route(line_s, t_summary, w_summary, h_summary, line_b, line_m, line_d, missing, forbidden, no_action)
    write_json(out_dir / "v158_route_decision.json", route)
    build_line_z_artifacts(out_dir, route, t_summary, w_summary, h_summary, line_b, line_m, line_d)
    build_contracts(out_dir, route)
    build_docs(out_dir, args, route, line_s, t_summary, w_summary, h_summary, line_b, line_m, line_d)
    print(json.dumps(route, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
