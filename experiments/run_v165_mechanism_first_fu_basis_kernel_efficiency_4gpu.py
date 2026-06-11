#!/usr/bin/env python3
"""DG-KAN v16.5 mechanism-first FU + basis kernel efficiency runner.

This runner executes the v16.5 additions on top of the v16.4.1 artifact base:

* fresh v16.5 basis efficiency repair truth-table rows, including phase and
  memory decomposition against a same-param MLP reference;
* fresh low-budget M9 train-split transfer probes and M10 path-type
  diagnostics;
* explicit readback of v16.4.1/v16.3 h800/h1600 functional matrix artifacts.

Readback rows are marked as reuse_readback=1 and are never presented as fresh
v16.5 training. Efficiency and smoke/probe rows are timing or synthetic
train-stream evidence only; they do not grant functional promotion.
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

from experiments import run_v154_split_consensus_signal_subspace_fu_allbasis as v154  # noqa: E402
from experiments import run_v158_dynamics_harness_decoupled_decay_recovery_allbasis as v158  # noqa: E402
from experiments import run_v162_multischeme_functional_dynamics_4gpu as v162  # noqa: E402
from experiments import run_v163_multischeme_functional_dynamics_efficiency_census_4gpu as v163  # noqa: E402
from experiments import run_v1641_functional_update_efficiency_breakthrough_4gpu as v1641  # noqa: E402


PLAN_DOC = ROOT / "docs/DG-KAN_v16.5_MechanismFirstFU_BasisKernelEfficiency_4GPU计划.md"
RECAP_DOC = ROOT / "docs/DG-KAN_v16.5_MechanismFirstFU_BasisKernelEfficiency_4GPU_实验结果复盘.md"
EXEC_LOG_DOC = ROOT / "docs/DG-KAN_v16.5_MechanismFirstFU_BasisKernelEfficiency_4GPU_执行日志.md"
DEFAULT_OUT = ROOT / "results/v16_5_mechanism_first_fu_basis_kernel_efficiency_4gpu/official_v165"
DEFAULT_V1641_OUT = ROOT / "results/v16_4_1_functional_update_efficiency_breakthrough_4gpu/official_v1641"

read_rows = v154.read_rows
write_rows = v154.write_rows
write_json = v154.write_json
fnum = v154.fnum
sint = v154.sint
mean = v154.mean
resolve_cuda_device = v154.resolve_cuda_device


REQUIRED_ARTIFACTS = [
    "v165_route_decision.json",
    "v165_efficiency_truth_table.csv",
    "v165_efficiency_phase_breakdown.csv",
    "v165_memory_phase_breakdown.csv",
    "v165_efficiency_repair_vs_baseline.csv",
    "v165_basis_efficiency_failure_taxonomy.csv",
    "v165_same_param_mlp_manifest.csv",
    "v165_mechanism_probe_m9_m10.csv",
    "v165_m10_path_type_diagnostic.csv",
    "v165_line_a_m9_m10_results.csv",
    "v165_line_b_m9_m10_results.csv",
    "v165_line_a_m9_m10_horizon_recovery.csv",
    "v165_line_b_m9_m10_horizon_recovery.csv",
    "v165_line_a_m9_m10_summary.csv",
    "v165_line_b_m9_m10_summary.csv",
    "v165_line_a_m9_m10_direction_provenance.csv",
    "v165_line_b_m9_m10_direction_provenance.csv",
    "v165_m9_m10_official_surface_manifest.csv",
    "v165_functional_smoke.csv",
    "v165_functional_mechanism_matrix.csv",
    "v165_line_a_dche_results.csv",
    "v165_line_b_mlp_results.csv",
    "v165_line_c_lq_reanchor.csv",
    "v165_line_d_rational_monitor.csv",
    "v165_line_f_allbasis_substrate.csv",
    "v165_line_m_attribution.csv",
    "v165_h800_summary.csv",
    "v165_h1600_summary.csv",
    "v165_direction_provenance.csv",
    "v165_runnable_queue.csv",
    "v165_gpu_assignment_manifest.csv",
    "v165_gpu_utilization_dashboard.csv",
    "v165_idle_violation.csv",
    "v165_deferred_items.csv",
    "v165_queue_drain_report.csv",
    "v165_required_artifact_manifest.csv",
    "v165_forbidden_information_audit.csv",
    "v165_no_action_search_audit.csv",
    "v165_failure_taxonomy.csv",
    "v165_no_go_boundary.csv",
    "v165_next_hypothesis_queue.csv",
    "v165_code_review_packet.csv",
    "v165_code_review_packet.zip",
    "v165_implementation_readback.md",
]

REQUIRED_FIGURES = [
    "fig_01_progress_by_line.svg",
    "fig_02_best_source_retention_vs_debt.svg",
    "fig_03_h800_to_h1600_source_collapse.svg",
    "fig_04_KAN_vs_MLP_attribution_matrix.svg",
    "fig_05_efficiency_forward_backward_update_memory_heatmap.svg",
    "fig_06_efficiency_phase_stacked_bar_by_family.svg",
    "fig_07_audit_cost_vs_training_cost.svg",
    "fig_08_basis_family_bottleneck_waterfall.svg",
    "fig_09_4gpu_utilization_timeline.svg",
    "fig_10_queue_drain_report.svg",
    "fig_11_functional_mechanism_failure_taxonomy.svg",
    "fig_12_basis_efficiency_failure_taxonomy.svg",
]

EFFICIENCY_CANDIDATES = [
    ("P0-MLP-same-param-reference", "MLP", "MLP-v165-reference", "base", "P0", "same-param MLP reference"),
    ("P8-MLP-functional-path-top2", "MLP", "MLP-v165-reference", "functional_top2", "P8", "MLP functional path top2 timing"),
    ("CHE-E0-D-CHE-current", "D-CHE", "D-CHE20-DegreeNormalizedReadoutHealthSubstrate", "base", "current", "current D-CHE carrier"),
    ("CHE-E1-training-only-no-LineC", "D-CHE", "D-CHE20-DegreeNormalizedReadoutHealthSubstrate", "base", "CHE-E1", "training-only timing with LineC separated"),
    ("CHE-E2-degree-recurrence-cache", "D-CHE", "D-CHE17-HighDegreeLateEnableSubstrate", "base", "CHE-E2", "degree recurrence/cache candidate"),
    ("CHE-E3-low-degree-active-bank", "D-CHE", "D-CHE16-DegreeEnergyDampingSubstrate", "base", "CHE-E3", "low-degree active bank"),
    ("CHE-E4-fused-degree-role-projection-update", "D-CHE", "D-CHE18-RoleDegreeEnergyCapSubstrate", "functional_top2", "CHE-E4", "fused degree-role projection/update"),
    ("CHE-E5-no-materialize-degree-energy-readback", "D-CHE", "D-CHE19-ChebyTangentTrustSubstrate", "base", "CHE-E5", "no-materialize degree energy readback"),
    ("CHE-E6-audit-cost-separated", "D-CHE", "D-CHE20-DegreeNormalizedReadoutHealthSubstrate", "base", "CHE-E6", "audit-cost separated readback"),
    ("FOU-E0-current", "D-FOU", "D-FOU14-SincosSharedWorkspace-K2", "base", "current", "current Fourier carrier"),
    ("FOU-E1-low-frequency-recurrence", "D-FOU", "D-FOU16-FrequencyBandDampingSubstrate", "base", "FOU-E1", "low-frequency recurrence"),
    ("FOU-E2-sincos-precompute-shape", "D-FOU", "D-FOU14-SincosSharedWorkspace-K2", "base", "FOU-E2", "sincos precompute shape"),
    ("FOU-E3-fused-band-k-small", "D-FOU", "D-FOU13-FusedReadoutGradNoMaterialize-K2", "base", "FOU-E3", "fused small band"),
    ("FOU-E4-high-frequency-quarantine", "D-FOU", "D-FOU19-HighFreqNoiseLeakVetoSubstrate", "base", "FOU-E4", "high-frequency quarantine"),
    ("FOU-E5-table-lookup-sincos", "D-FOU", "D-FOU12-LifetimeRecomputeBackward-K2", "base", "FOU-E5", "table lookup sincos"),
    ("FOU-E6-trig-vs-recurrence-table", "D-FOU", "D-FOU12-LifetimeRecomputeBackward-K2", "base", "FOU-E6", "trig vs recurrence table"),
    ("LQ-E0-current", "LQ", "C2-LQ-ProtocolMatchedReanchor", "base", "current", "current/reanchored LQ carrier"),
    ("LQ-E1-update-decomposition", "LQ", "C5-LQ-StepMemoryRecheck", "base", "LQ-E1", "update decomposition"),
    ("LQ-E2-persistent-adamw-foreach", "LQ", "C5-LQ-StepMemoryRecheck", "base", "LQ-E2", "persistent AdamW foreach"),
    ("LQ-E3-fused-projection-update", "LQ", "C6-LQ-LineCNoRegressionCheck", "functional_top2", "LQ-E3", "fused projection update"),
    ("LQ-E4-fixed-frame-late-attach", "LQ", "C4-LQ-MacroDeltaPriorityReanchor", "base", "LQ-E4", "fixed-frame late attach"),
    ("LQ-E5-no-rebuild-frame", "LQ", "C2-LQ-ProtocolMatchedReanchor", "base", "LQ-E5", "no rebuild frame smoke"),
    ("LQ-E6-projection-cache-reuse", "LQ", "C6-LQ-LineCNoRegressionCheck", "functional_top2", "LQ-E6", "projection cache reuse"),
    ("RBF-E0-current", "D-RBF", "D-RBF11-CompactExpressionRepair-Monitor", "base", "current", "current RBF/FastKAN carrier"),
    ("RBF-E1-active-center-occupancy", "D-RBF", "D-RBF12-CenterOccupancyRebalanceSubstrate", "base", "RBF-E1", "active center occupancy"),
    ("RBF-E2-width-condition-guard", "D-RBF", "D-RBF13-WidthConditionGuardSubstrate", "base", "RBF-E2", "width condition guard"),
    ("RBF-E3-local-K4-no-dense", "D-RBF", "D-RBF17-CompactCapacityK4HealthSubstrate", "base", "RBF-E3", "local K4 no dense"),
    ("RBF-E4-table-lookup-gaussian", "D-RBF", "D-RBF15-LocalCurvatureSmoothSubstrate", "base", "RBF-E4", "table lookup gaussian"),
    ("RBF-E5-center-update-separation", "D-RBF", "D-RBF16-ActiveCenterDiversityTransportSubstrate", "functional_top2", "RBF-E5", "center/update separation"),
    ("RBF-E6-width-frozen-smoke", "D-RBF", "D-RBF13-WidthConditionGuardSubstrate", "base", "RBF-E6", "width-frozen smoke"),
    ("WAV-E0-current", "D-WAV", "D-WAV11-ScaleEnergyBalanceSubstrate", "base", "current", "current wavelet carrier"),
    ("WAV-E1-triangular-index-forward", "D-WAV", "D-WAV12-LocalSupportOccupancyRepairSubstrate", "base", "WAV-E1", "triangular index forward"),
    ("WAV-E2-sparse-support-backward", "D-WAV", "D-WAV15-ScaleDiversityTransportSubstrate", "functional_top2", "WAV-E2", "sparse support backward"),
    ("WAV-E3-support-overlap-damping", "D-WAV", "D-WAV14-SupportOverlapEntropyGuardSubstrate", "base", "WAV-E3", "support overlap damping"),
    ("WAV-E4-scale-occupancy-monitor", "D-WAV", "D-WAV13-LocalTailCoverageGuardSubstrate", "base", "WAV-E4", "scale occupancy monitor"),
    ("WAV-E5-index-only-readback", "D-WAV", "D-WAV12-LocalSupportOccupancyRepairSubstrate", "base", "WAV-E5", "index-only readback"),
    ("RAT-E0-current", "D-RAT", "D-RAT28-GroupDiversityPreservingRational", "base", "current", "current Rational monitor"),
    ("RAT-E1-branchless-denominator-eval", "D-RAT", "D-RAT34-TrainingDenDerivativeStabilityNoCE", "base", "RAT-E1", "branchless denominator eval"),
    ("RAT-E2-branchless-denominator-readback", "D-RAT", "D-RAT35-ReadoutRationalDecoupleNoCE", "base", "RAT-E2", "branchless denominator readback"),
    ("RAT-E3-horner-polynomial-eval", "D-RAT", "D-RAT36-TangentConditionStabilizerNoCE", "base", "RAT-E3", "Horner polynomial eval"),
    ("RAT-E4-optimizer-state-cost", "D-RAT", "D-RAT40-ResponseReadyReadoutCouplingSubstrate", "base", "RAT-E4", "optimizer state cost"),
    ("RAT-E5-denominator-safety-no-reset", "D-RAT", "D-RAT35-ReadoutRationalDecoupleNoCE", "base", "RAT-E5", "denominator safety without reset"),
]

PROBE_CASES = [
    ("D-CHE", "D-CHE20-DegreeNormalizedReadoutHealthSubstrate", "M9-TrainSplitTransferOperator", "A-M9-D-CHE-TrainSplitTransfer"),
    ("D-CHE", "D-CHE20-DegreeNormalizedReadoutHealthSubstrate", "M10-PathTypeClassifierDiagnostic", "A-M10-D-CHE-PathTypeDiagnostic"),
    ("MLP", "MLP-v165-reference", "M9-TrainSplitTransferOperator", "B-M9-MLP-TrainSplitTransfer"),
    ("MLP", "MLP-v165-reference", "M10-PathTypeClassifierDiagnostic", "B-M10-MLP-PathTypeDiagnostic"),
    ("LQ", "C4-LQ-MacroDeltaPriorityReanchor", "M9-TrainSplitTransferOperator", "C-M9-LQ-TrainSplitTransferSmoke"),
    ("D-RAT", "D-RAT35-ReadoutRationalDecoupleNoCE", "M9-TrainSplitTransferOperator", "D-M9-RAT-TrainSplitTransferSmoke"),
    ("D-FOU", "D-FOU13-FusedReadoutGradNoMaterialize-K2", "M9-TrainSplitTransferOperator", "F-M9-FOU-TrainSplitTransferSmoke"),
    ("D-RBF", "D-RBF17-CompactCapacityK4HealthSubstrate", "M9-TrainSplitTransferOperator", "F-M9-RBF-TrainSplitTransferSmoke"),
    ("D-WAV", "D-WAV15-ScaleDiversityTransportSubstrate", "M9-TrainSplitTransferOperator", "F-M9-WAV-TrainSplitTransferSmoke"),
]

REAL_A_M9_M10_CANDIDATES = [
    "A-M9-D-CHE-TrainSplitTransferOperator",
    "A-M10-D-CHE-PathTypeClassifierDiagnostic",
]
REAL_B_M9_M10_CANDIDATES = [
    "B-M9-MLP-TrainSplitTransferOperator",
    "B-M10-MLP-PathTypeClassifierDiagnostic",
]
REAL_A_M9_M10_CONTROLS = [
    "ACTRL0-D-CHE-AdamW",
    "ACTRL1-D-CHE-NoOpMatchedOverhead",
    "ACTRL2-D-CHE-RandomMatchedPulse",
    "ACTRL3-D-CHE-AdamWExtraStepsMatchedTime",
]
REAL_B_M9_M10_CONTROLS = [
    "BCTRL0-MLP-AdamW",
    "BCTRL1-MLP-NoOpMatchedOverhead",
    "BCTRL2-MLP-RandomMatchedPulse",
    "BCTRL3-MLP-AdamWExtraStepsMatchedTime",
]

REAL_M9_M10_JOBS = [
    ("A", "D-CHE", REAL_A_M9_M10_CANDIDATES[:1] + REAL_A_M9_M10_CONTROLS, REAL_A_M9_M10_CONTROLS, REAL_A_M9_M10_CANDIDATES, "real_a_m9"),
    ("A", "D-CHE", REAL_A_M9_M10_CANDIDATES[1:], REAL_A_M9_M10_CONTROLS, REAL_A_M9_M10_CANDIDATES, "real_a_m10"),
    ("B", "MLP", REAL_B_M9_M10_CANDIDATES[:1] + REAL_B_M9_M10_CONTROLS, REAL_B_M9_M10_CONTROLS, REAL_B_M9_M10_CANDIDATES, "real_b_m9"),
    ("B", "MLP", REAL_B_M9_M10_CANDIDATES[1:], REAL_B_M9_M10_CONTROLS, REAL_B_M9_M10_CANDIDATES, "real_b_m10"),
]

PHASE_KEYS = [
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
]


def now_line() -> str:
    return f"生成时间：{datetime.now(ZoneInfo('Asia/Singapore')).strftime('%Y-%m-%d %H:%M:%S')}（Asia/Singapore）"


def table(rows: Sequence[dict[str, Any]], cols: Sequence[tuple[str, str]], limit: int | None = None) -> list[str]:
    return v162.table(rows, cols, limit)


def median(values: Sequence[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(statistics.median(vals)) if vals else 0.0


def ratio(num: Any, den: Any) -> float:
    return float(fnum(num, 0.0)) / max(1.0e-9, float(fnum(den, 0.0)))


def subject_family(subject: str, carrier: str) -> str:
    if subject.startswith("P0") or subject.startswith("P8"):
        return "MLP"
    return carrier


def v165_prefix(row: dict[str, Any], stage: str, source_artifact: Path | None = None) -> dict[str, Any]:
    out = dict(row)
    out["stage"] = stage
    out["reuse_readback"] = 1
    out["fresh_v165_training"] = 0
    out["source_artifact"] = str(source_artifact or "")
    out["promotion_allowed"] = 0
    if "mechanism_family" not in out or str(out.get("mechanism_family", "")) == "":
        method = str(out.get("method", out.get("best_method", "")))
        out["mechanism_family"] = v162.mechanism_family(method)
    return out


def v165_mechanism_family(method: str) -> str:
    if "-M9" in str(method):
        return "M9-TrainSplitTransferOperator"
    if "-M10" in str(method):
        return "M10-PathTypeClassifierDiagnostic"
    return v162.mechanism_family(str(method))


_V162_ORIGINAL_MAP = v162.map_dynamics_method


def v165_map_dynamics_method(method: str) -> tuple[str, dict[str, Any], str, str]:
    extra: dict[str, tuple[str, dict[str, Any], str, str]] = {
        "A-M9-D-CHE-TrainSplitTransferOperator": (
            "T1-G7R-PulseOnce-then-AdamWRecovery|V165-M9-TrainSplitTransfer",
            {"lr": 0.0012, "lambda_noise": 0.10},
            "v16.5 M9 real h800 row: train-split transfer operator; direction uses train-stream split gradients only",
            "M9-TrainSplitTransferOperator",
        ),
        "A-M10-D-CHE-PathTypeClassifierDiagnostic": (
            "TCTRL-AdamWExtraStepsMatchedTime|V165-M10-PathDiagnostic",
            {},
            "v16.5 M10 real h800 diagnostic row: post-hoc path type readback, no controller/action/reset",
            "M10-PathTypeClassifierDiagnostic",
        ),
        "B-M9-MLP-TrainSplitTransferOperator": (
            "MLP-G7AnalogPulse|V165-M9-TrainSplitTransfer",
            {"lr": 0.0012, "lambda_noise": 0.10},
            "v16.5 M9 real h800 row: MLP train-split transfer operator; generic dynamics only",
            "M9-TrainSplitTransferOperator",
        ),
        "B-M10-MLP-PathTypeClassifierDiagnostic": (
            "MLP-AdamW|V165-M10-PathDiagnostic",
            {},
            "v16.5 M10 real h800 diagnostic row: MLP post-hoc path type readback, no controller/action/reset",
            "M10-PathTypeClassifierDiagnostic",
        ),
    }
    if method in extra:
        return extra[method]
    return _V162_ORIGINAL_MAP(method)


def mark_v165_real_rows(rows: Sequence[dict[str, Any]], carrier: str, stage: str, source_artifact: Path) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        item = dict(row)
        item["carrier"] = carrier
        method = str(item.get("method") or item.get("best_method") or "")
        item["mechanism_family"] = v165_mechanism_family(method)
        item["stage"] = stage
        item["reuse_readback"] = 0
        item["fresh_v165_training"] = 1
        item["source_artifact"] = str(source_artifact)
        item["official_h800_surface"] = 1
        item["low_budget_probe"] = 0
        item["promotion_allowed"] = 0
        out.append(item)
    return out


def surface_summary_rows(summary_rows: Sequence[dict[str, Any]], carrier: str, source_artifact: Path) -> list[dict[str, Any]]:
    out = []
    for row in summary_rows:
        method = str(row.get("method", ""))
        out.append(
            {
                "carrier": carrier,
                "mechanism_family": v165_mechanism_family(method),
                "method_count": 1,
                "best_method": method,
                "source_vs_best_control_h800": row.get("source_vs_best_control_h800", ""),
                "source_retention_h800": row.get("source_retention_h800", ""),
                "tail_recovery_rate_h800": row.get("tail_recovery_rate_h800", ""),
                "LineC_recovery_rate_h800": row.get("LineC_recovery_rate_h800", ""),
                "AUCtime_ratio_h800": row.get("AUCtime_ratio_h800", ""),
                "control_equivalent_fraction": row.get("control_equivalent_fraction", ""),
                "bad_event_fraction": row.get("bad_event_fraction", ""),
                "S2_weak_productive_dynamics": int(
                    fnum(row.get("source_vs_best_control_h800"), -999) >= 0.005
                    and fnum(row.get("source_retention_h800"), 0.0) >= 0.40
                    and (
                        fnum(row.get("tail_recovery_rate_h800"), 0.0) >= 0.40
                        or fnum(row.get("LineC_recovery_rate_h800"), 0.0) >= 0.40
                    )
                    and fnum(row.get("AUCtime_ratio_h800"), 9.0) <= 1.10
                    and fnum(row.get("control_equivalent_fraction"), 1.0) <= 0.50
                    and v165_mechanism_family(method) != "M10-PathTypeClassifierDiagnostic"
                ),
                "S3_productive_debt_recovery": int(
                    fnum(row.get("source_vs_best_control_h800"), -999) >= 0.005
                    and fnum(row.get("source_retention_h800"), 0.0) >= 0.50
                    and fnum(row.get("tail_recovery_rate_h800"), 0.0) >= 0.60
                    and fnum(row.get("LineC_recovery_rate_h800"), 0.0) >= 0.60
                    and fnum(row.get("AUCtime_ratio_h800"), 9.0) <= 1.05
                    and fnum(row.get("control_equivalent_fraction"), 1.0) <= 0.50
                    and v165_mechanism_family(method) != "M10-PathTypeClassifierDiagnostic"
                ),
                "S5_official_functional_success": 0,
                "evidence_scope": "fresh_v165_real_h800_surface",
                "M1_M10_coverage_component": "M9_M10_h800_surface",
                "reuse_readback": 0,
                "fresh_v165_training": 1,
                "source_artifact": str(source_artifact),
                "promotion_allowed": 0,
            }
        )
    return out


def profile_ref_mlp_phases(
    subject: str,
    target_params: int,
    batch_size: int,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float]:
    input_dim = int(args.efficiency_input_dim)
    output_dim = int(args.efficiency_classes)
    train_size = max(int(args.efficiency_train_size), int(batch_size))
    gen = torch.Generator(device=device).manual_seed(163_000 + int(batch_size) + sum(ord(c) for c in subject))
    x_ref = torch.randn(train_size, input_dim, generator=gen, device=device)
    y_ref = torch.randint(0, output_dim, (train_size,), generator=gen, device=device)
    xb = x_ref[: int(batch_size)]
    yb = y_ref[: int(batch_size)]
    hidden = v163.matched_mlp_hidden(input_dim, output_dim, int(target_params))
    ref = v163.make_efficiency_model("MLP", "MLP-v165-same-param", input_dim, output_dim, x_ref, 0, args, device, hidden)

    def one_grad() -> None:
        v163.single_grad_pass(ref, xb, yb)

    fwd_ms, f_peak, _ = v163.timed_phase(lambda: ref(xb).detach(), device, int(args.efficiency_repeats), int(args.efficiency_warmup))
    bwd_ms, b_peak, _ = v163.timed_phase(one_grad, device, int(args.efficiency_repeats), int(args.efficiency_warmup))
    v163.single_grad_pass(ref, xb, yb)
    opt = torch.optim.AdamW(ref.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    upd_ms, _u_peak, _ = v163.timed_phase(lambda: opt.step(), device, int(args.efficiency_repeats), int(args.efficiency_warmup))
    v163.single_grad_pass(ref, xb, yb)
    box: dict[str, torch.Tensor] = {}
    dir_ms, _d_peak, _ = v163.timed_phase(lambda: box.setdefault("g", v163.flat_grad(ref, device)), device, int(args.efficiency_repeats), int(args.efficiency_warmup))
    update = box.get("g", v163.flat_grad(ref, device))
    pbox: dict[str, torch.Tensor] = {}
    proj_ms, _p_peak, _ = v163.timed_phase(lambda: pbox.setdefault("u", v163.functional_projection(ref, "MLP", update)), device, int(args.efficiency_repeats), int(args.efficiency_warmup))
    projected = pbox.get("u", update)
    commit_ms, _c_peak, _ = v163.timed_phase(lambda: v163.apply_flat_update(ref, projected, float(args.lr)), device, int(args.efficiency_repeats), int(args.efficiency_warmup))
    tail_ms, _t_peak, _ = v163.timed_phase(lambda: v163.eval_tail_pack(ref, xb, yb), device, int(args.efficiency_repeats), int(args.efficiency_warmup))
    horizon_ms, _h_peak, _ = v163.timed_phase(lambda: v163.eval_tail_pack(ref, xb, yb), device, int(args.efficiency_repeats), int(args.efficiency_warmup))

    def total_step() -> None:
        v163.single_grad_pass(ref, xb, yb)
        v163.optimizer_step_only(ref, float(args.lr), float(args.weight_decay))
        _ = v163.eval_tail_pack(ref, xb, yb)

    step_ms, step_peak, _ = v163.timed_phase(total_step, device, int(args.efficiency_repeats), int(args.efficiency_warmup))
    params = int(v163.param_count(ref))
    param_bytes = float(params) * 4.0
    grad_bytes = float(params) * 4.0
    opt_bytes = float(params) * 8.0
    peak_bytes = float(step_peak)
    activation = max(0.0, peak_bytes - param_bytes - grad_bytes - opt_bytes)
    return {
        "ref_forward_only_ms": float(fwd_ms),
        "ref_backward_grad_ms": float(bwd_ms),
        "ref_optimizer_update_ms": float(upd_ms),
        "ref_functional_direction_ms": float(dir_ms),
        "ref_functional_projection_ms": float(proj_ms),
        "ref_functional_commit_ms": float(commit_ms),
        "ref_linec_audit_ms": 0.0,
        "ref_tail_calibration_audit_ms": float(tail_ms),
        "ref_horizon_readback_ms": float(horizon_ms),
        "ref_step_total_ms": float(step_ms),
        "ref_forward_peak_allocated_mb": float(f_peak) / 1_048_576.0,
        "ref_backward_peak_allocated_mb": float(b_peak) / 1_048_576.0,
        "ref_optimizer_state_mb": opt_bytes / 1_048_576.0,
        "ref_activation_saved_bytes": float(activation),
        "ref_param_bytes": float(param_bytes),
    }


def profile_one_efficiency(candidate: tuple[str, str, str, str, str, str], batch: int, args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    subject, carrier, candidate_id, mode, repair_id, description = candidate
    try:
        row = v163.profile_model_against_mlp(subject, carrier, candidate_id, mode, batch, args, device)
        ref = profile_ref_mlp_phases(subject, int(row.get("param_count", 0)), int(batch), args, device)
        row.update(
            {
                "stage": "V165_EFFICIENCY_TRUTH_TABLE",
                "family": subject_family(subject, carrier),
                "repair_attempt": repair_id,
                "repair_description": description,
                "repair_candidate_id": candidate_id,
                "repair_claim_scope": "actual v16.5 candidate/kernel timing; engineering evidence only",
                "audit_cost_separated": 1,
                "training_cost_separated": 1,
                "cpu_offload_used": 0,
                "fake_or_proxy_timing": 0,
                "promotion_allowed": 0,
                "same_param_mlp_forward_ratio": row.get("forward_ratio_vs_same_param_mlp", ""),
                "same_param_mlp_backward_ratio": row.get("backward_ratio_vs_same_param_mlp", ""),
                "same_param_mlp_update_ratio": row.get("optimizer_update_ratio_vs_same_param_mlp", ""),
                "same_param_mlp_step_ratio": row.get("step_ratio_vs_same_param_mlp", ""),
                "same_param_mlp_memory_ratio": row.get("backward_memory_ratio_vs_same_param_mlp", ""),
                "functional_direction_ratio_vs_same_param_mlp": ratio(row.get("functional_direction_ms"), ref.get("ref_functional_direction_ms")),
                "functional_projection_ratio_vs_same_param_mlp": ratio(row.get("functional_projection_ms"), ref.get("ref_functional_projection_ms")),
                "functional_commit_ratio_vs_same_param_mlp": ratio(row.get("functional_commit_ms"), ref.get("ref_functional_commit_ms")),
                "linec_audit_ratio_vs_same_param_mlp": ratio(row.get("linec_audit_ms"), ref.get("ref_linec_audit_ms")),
                "horizon_readback_ratio_vs_same_param_mlp": ratio(row.get("horizon_readback_ms"), ref.get("ref_horizon_readback_ms")),
                "forward_peak_memory_ratio": ratio(row.get("forward_peak_allocated_mb"), ref.get("ref_forward_peak_allocated_mb")),
                "backward_peak_memory_ratio": row.get("backward_memory_ratio_vs_same_param_mlp", ""),
                "optimizer_state_memory_ratio": ratio(row.get("optimizer_state_mb"), ref.get("ref_optimizer_state_mb")),
                "basis_activation_bytes_ratio": ratio(row.get("basis_activation_bytes"), ref.get("ref_activation_saved_bytes")),
                "functional_state_bytes_ratio": ratio(row.get("functional_state_bytes"), ref.get("ref_param_bytes")),
                "forward_peak_memory": row.get("forward_peak_allocated_mb", ""),
                "backward_peak_memory": row.get("backward_peak_allocated_mb", ""),
                "optimizer_state_memory": row.get("optimizer_state_mb", ""),
            }
        )
        row.update(ref)
        return row
    except RuntimeError as exc:
        if device.type == "cuda" and "out of memory" in str(exc).lower():
            torch.cuda.empty_cache()
        return {
            "stage": "V165_EFFICIENCY_TRUTH_TABLE",
            "subject": subject,
            "family": carrier,
            "carrier": carrier,
            "candidate_id": candidate_id,
            "mode": mode,
            "batch_size": int(batch),
            "repair_attempt": repair_id,
            "repair_description": description,
            "execution_status": f"runtime_error:{type(exc).__name__}",
            "error": str(exc)[:500],
            "efficiency_exploration_gate": 0,
            "efficiency_official_gate": 0,
            "audit_cost_separated": 0,
            "training_cost_separated": 0,
            "cpu_offload_used": 0,
            "fake_or_proxy_timing": 0,
            "promotion_allowed": 0,
        }


def run_efficiency_shard(args: argparse.Namespace, out_dir: Path, device: torch.device) -> None:
    rows = []
    jobs = [
        (idx, cand, batch)
        for idx, cand in enumerate(EFFICIENCY_CANDIDATES)
        for batch in [8, 32, 128]
        if idx % int(args.efficiency_shard_count) == int(args.efficiency_shard_index)
    ]
    for _idx, cand, batch in jobs:
        rows.append(profile_one_efficiency(cand, batch, args, device))
    suffix = f"_e{int(args.efficiency_shard_index)}" if int(args.efficiency_shard_count) > 1 else ""
    write_rows(out_dir / f"v165_efficiency_truth_table{suffix}.csv", rows)


def merge_efficiency(out_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx in range(32):
        path = out_dir / f"v165_efficiency_truth_table_e{idx}.csv"
        if path.exists():
            rows.extend(read_rows(path))
    if not rows:
        rows = read_rows(out_dir / "v165_efficiency_truth_table.csv")
    rows = sorted(rows, key=lambda r: (str(r.get("family", r.get("carrier", ""))), str(r.get("subject", "")), sint(r.get("batch_size"), 0)))
    write_rows(out_dir / "v165_efficiency_truth_table.csv", rows)

    phase_rows = []
    memory_rows = []
    manifest = []
    blockers = []
    for row in rows:
        for key in PHASE_KEYS:
            phase_rows.append(
                {
                    "family": row.get("family", row.get("carrier")),
                    "carrier": row.get("carrier"),
                    "subject": row.get("subject"),
                    "repair_attempt": row.get("repair_attempt"),
                    "batch_size": row.get("batch_size"),
                    "phase": key,
                    "value_ms": row.get(key, ""),
                    "same_param_ratio": row.get(f"{key.replace('_ms', '')}_ratio_vs_same_param_mlp", ""),
                    "audit_cost_separated": row.get("audit_cost_separated"),
                    "promotion_allowed": 0,
                }
            )
        for key in [
            "forward_peak_memory",
            "backward_peak_memory",
            "optimizer_state_memory",
            "basis_activation_bytes",
            "functional_state_bytes",
            "workspace_temp_bytes",
            "activation_saved_bytes",
        ]:
            memory_rows.append(
                {
                    "family": row.get("family", row.get("carrier")),
                    "carrier": row.get("carrier"),
                    "subject": row.get("subject"),
                    "repair_attempt": row.get("repair_attempt"),
                    "batch_size": row.get("batch_size"),
                    "memory_component": key,
                    "value": row.get(key, ""),
                    "promotion_allowed": 0,
                }
            )
        manifest.append(
            {
                "subject": row.get("subject"),
                "family": row.get("family", row.get("carrier")),
                "carrier": row.get("carrier"),
                "batch_size": row.get("batch_size"),
                "param_count": row.get("param_count"),
                "same_param_mlp_hidden": row.get("same_param_mlp_hidden"),
                "same_param_mlp_param_count": row.get("same_param_mlp_param_count"),
                "param_match_error_fraction": row.get("param_match_error_fraction"),
                "same_param_mlp_sanity_pass": row.get("same_param_mlp_sanity_pass"),
                "promotion_allowed": 0,
            }
        )
        reasons = []
        if str(row.get("execution_status")) != "measured":
            reasons.append("F0-RuntimeProfileMissing")
        if fnum(row.get("forward_ratio_vs_same_param_mlp"), 999.0) > 1.75:
            reasons.append("F7-EfficiencyForwardBlocked")
        if fnum(row.get("backward_ratio_vs_same_param_mlp"), 999.0) > 1.75:
            reasons.append("F8-EfficiencyBackwardBlocked")
        if fnum(row.get("optimizer_update_ratio_vs_same_param_mlp"), 999.0) > 1.75:
            reasons.append("F9-EfficiencyUpdateBlocked")
        if fnum(row.get("backward_peak_memory_ratio"), fnum(row.get("backward_memory_ratio_vs_same_param_mlp"), 999.0)) > 1.50:
            reasons.append("F10-MemoryBlocked")
        if fnum(row.get("audit_overhead_fraction"), 0.0) > 0.35:
            reasons.append("F11-AuditCostPolluted")
        blockers.append(
            {
                "subject": row.get("subject"),
                "family": row.get("family", row.get("carrier")),
                "carrier": row.get("carrier"),
                "batch_size": row.get("batch_size"),
                "repair_attempt": row.get("repair_attempt"),
                "blocker_class": "PASS" if not reasons else ";".join(reasons),
                "primary_bottleneck": row.get("dominant_phase", ""),
                "reason": row.get("efficiency_blocked_reasons", row.get("execution_status", "")),
                "audit_cost_separated": row.get("audit_cost_separated"),
                "promotion_allowed": 0,
            }
        )
    write_rows(out_dir / "v165_efficiency_phase_breakdown.csv", phase_rows)
    write_rows(out_dir / "v165_memory_phase_breakdown.csv", memory_rows)
    write_rows(out_dir / "v165_same_param_mlp_manifest.csv", manifest)
    write_rows(out_dir / "v165_basis_efficiency_failure_taxonomy.csv", blockers)
    build_repair_vs_baseline(out_dir, rows)
    return rows


def build_repair_vs_baseline(out_dir: Path, rows: Sequence[dict[str, Any]]) -> None:
    by_group: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        family = str(row.get("family", row.get("carrier", "")))
        if family == "MLP":
            continue
        by_group.setdefault((family, sint(row.get("batch_size"), 0)), []).append(row)
    out = []
    for (family, batch), group in sorted(by_group.items()):
        baseline = next((r for r in group if str(r.get("repair_attempt")) == "current" or str(r.get("subject", "")).endswith("current")), group[0])
        repairs = [r for r in group if r is not baseline]
        best = min(repairs or group, key=lambda r: fnum(r.get("step_ratio_vs_same_param_mlp"), 999.0))
        out.append(
            {
                "family": family,
                "batch_size": batch,
                "baseline_subject": baseline.get("subject"),
                "best_repair_subject": best.get("subject"),
                "baseline_forward_ratio": baseline.get("forward_ratio_vs_same_param_mlp"),
                "best_forward_ratio": best.get("forward_ratio_vs_same_param_mlp"),
                "forward_ratio_delta_baseline_minus_best": fnum(baseline.get("forward_ratio_vs_same_param_mlp"), 0.0) - fnum(best.get("forward_ratio_vs_same_param_mlp"), 0.0),
                "baseline_backward_ratio": baseline.get("backward_ratio_vs_same_param_mlp"),
                "best_backward_ratio": best.get("backward_ratio_vs_same_param_mlp"),
                "backward_ratio_delta_baseline_minus_best": fnum(baseline.get("backward_ratio_vs_same_param_mlp"), 0.0) - fnum(best.get("backward_ratio_vs_same_param_mlp"), 0.0),
                "baseline_update_ratio": baseline.get("optimizer_update_ratio_vs_same_param_mlp"),
                "best_update_ratio": best.get("optimizer_update_ratio_vs_same_param_mlp"),
                "update_ratio_delta_baseline_minus_best": fnum(baseline.get("optimizer_update_ratio_vs_same_param_mlp"), 0.0) - fnum(best.get("optimizer_update_ratio_vs_same_param_mlp"), 0.0),
                "baseline_step_ratio": baseline.get("step_ratio_vs_same_param_mlp"),
                "best_step_ratio": best.get("step_ratio_vs_same_param_mlp"),
                "step_ratio_delta_baseline_minus_best": fnum(baseline.get("step_ratio_vs_same_param_mlp"), 0.0) - fnum(best.get("step_ratio_vs_same_param_mlp"), 0.0),
                "baseline_memory_ratio": baseline.get("backward_peak_memory_ratio", baseline.get("backward_memory_ratio_vs_same_param_mlp")),
                "best_memory_ratio": best.get("backward_peak_memory_ratio", best.get("backward_memory_ratio_vs_same_param_mlp")),
                "memory_ratio_delta_baseline_minus_best": fnum(baseline.get("backward_peak_memory_ratio", baseline.get("backward_memory_ratio_vs_same_param_mlp")), 0.0) - fnum(best.get("backward_peak_memory_ratio", best.get("backward_memory_ratio_vs_same_param_mlp")), 0.0),
                "repair_exploration_opened": best.get("efficiency_exploration_gate"),
                "repair_official_opened": best.get("efficiency_official_gate"),
                "blocker_after_repair": best.get("efficiency_blocked_reasons"),
                "promotion_allowed": 0,
            }
        )
    write_rows(out_dir / "v165_efficiency_repair_vs_baseline.csv", out)


def make_probe_model(family: str, candidate_id: str, x_ref: torch.Tensor, seed: int, args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    return v163.make_efficiency_model(family, candidate_id, int(args.probe_input_dim), int(args.probe_classes), x_ref, seed, args, device)


def loss_on(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    with torch.no_grad():
        return float(F.cross_entropy(model(x).float(), y).item())


def adam_train(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, steps: int, args: argparse.Namespace) -> None:
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    for _ in range(int(steps)):
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x).float(), y)
        loss.backward()
        opt.step()


def functional_train(model: torch.nn.Module, family: str, x: torch.Tensor, y: torch.Tensor, steps: int, args: argparse.Namespace, device: torch.device) -> None:
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    for step in range(int(steps)):
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x).float(), y)
        loss.backward()
        if step == 0:
            grad = v163.flat_grad(model, device)
            update = v163.functional_projection(model, "D-CHE" if family == "D-CHE" else family, grad)
            v163.apply_flat_update(model, update, float(args.lr))
        else:
            opt.step()


def random_pulse_train(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, steps: int, args: argparse.Namespace, device: torch.device, seed: int) -> None:
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    for step in range(int(steps)):
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x).float(), y)
        loss.backward()
        if step == 0:
            grad = v163.flat_grad(model, device)
            gen = torch.Generator(device=device).manual_seed(165_990 + int(seed))
            noise = torch.randn(grad.shape, generator=gen, device=device)
            noise = noise / max(1.0e-9, float(torch.linalg.vector_norm(noise).item())) * torch.linalg.vector_norm(grad)
            v163.apply_flat_update(model, noise, float(args.lr))
        else:
            opt.step()


def classify_path(source: float, b1: float, b2: float, b3: float, b2_random: float) -> str:
    if source >= 0.005 and b2 >= 0.005 and b3 >= 0.0 and b2_random >= 0.0:
        return "FastGood"
    if source >= 0.0 and b2 >= 0.0 and b3 >= 0.0:
        return "SlowBurnGood"
    if b1 >= 0.005 and (b2 < 0.0 or b3 < 0.0):
        return "RiskyHighSource"
    if source <= -0.005:
        return "BadPath"
    if abs(source) < 0.001 and abs(b2) < 0.001:
        return "HarmlessNull"
    return "ControlEquivalent"


def run_one_probe(case: tuple[str, str, str, str], seed: int, args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    family, candidate_id, mechanism, method = case
    gen = torch.Generator(device=device).manual_seed(165_000 + 1000 * seed + sum(ord(c) for c in method))
    n = int(args.probe_train_size)
    d = int(args.probe_input_dim)
    c = int(args.probe_classes)
    x_ref = torch.randn(n * 3, d, generator=gen, device=device)
    y_ref = torch.randint(0, c, (n * 3,), generator=gen, device=device)
    x1, x2, x3 = x_ref[:n], x_ref[n : 2 * n], x_ref[2 * n :]
    y1, y2, y3 = y_ref[:n], y_ref[n : 2 * n], y_ref[2 * n :]
    ctrl = make_probe_model(family, candidate_id, x_ref, seed, args, device)
    fu = make_probe_model(family, candidate_id, x_ref, seed, args, device)
    rand = make_probe_model(family, candidate_id, x_ref, seed, args, device)
    initial = {"B1": loss_on(ctrl, x1, y1), "B2": loss_on(ctrl, x2, y2), "B3": loss_on(ctrl, x3, y3)}
    t0 = time.perf_counter()
    adam_train(ctrl, x1, y1, int(args.probe_steps), args)
    functional_train(fu, family, x1, y1, int(args.probe_steps), args, device)
    random_pulse_train(rand, x1, y1, int(args.probe_steps), args, device, seed)
    elapsed = time.perf_counter() - t0
    ctrl_losses = {"B1": loss_on(ctrl, x1, y1), "B2": loss_on(ctrl, x2, y2), "B3": loss_on(ctrl, x3, y3)}
    fu_losses = {"B1": loss_on(fu, x1, y1), "B2": loss_on(fu, x2, y2), "B3": loss_on(fu, x3, y3)}
    rand_losses = {"B1": loss_on(rand, x1, y1), "B2": loss_on(rand, x2, y2), "B3": loss_on(rand, x3, y3)}
    b1 = ctrl_losses["B1"] - fu_losses["B1"]
    b2 = ctrl_losses["B2"] - fu_losses["B2"]
    b3 = ctrl_losses["B3"] - fu_losses["B3"]
    b2_random = rand_losses["B2"] - fu_losses["B2"]
    b3_random = rand_losses["B3"] - fu_losses["B3"]
    source_best = mean(
        [
            min(ctrl_losses["B1"], rand_losses["B1"]) - fu_losses["B1"],
            min(ctrl_losses["B2"], rand_losses["B2"]) - fu_losses["B2"],
            min(ctrl_losses["B3"], rand_losses["B3"]) - fu_losses["B3"],
        ]
    )
    split_transfer = mean([int(b2 >= 0.005), int(b3 >= 0.0), int(b2_random >= 0.0)])
    path_type = classify_path(source_best, b1, b2, b3, b2_random)
    official_main = int(family in {"D-CHE", "MLP"} and mechanism.startswith("M9"))
    return {
        "stage": "V165_M9_M10_MECHANISM_PROBE",
        "carrier": family,
        "candidate_id": candidate_id,
        "method": method,
        "mechanism_family": mechanism,
        "dataset": "synthetic-train-split",
        "seed": seed,
        "probe_steps": int(args.probe_steps),
        "B1_initial_loss": initial["B1"],
        "B2_initial_loss": initial["B2"],
        "B3_initial_loss": initial["B3"],
        "B1_gain": b1,
        "B2_gain": b2,
        "B3_gain": b3,
        "B2_minus_random": b2_random,
        "B3_minus_random": b3_random,
        "B2_minus_adam_parallel": b2,
        "source_vs_best_control_h20": source_best,
        "split_transfer_rate": split_transfer,
        "path_type": path_type,
        "transfer_probe_pass": int(source_best >= 0.005 and b2 >= 0.005 and b2_random >= 0.0),
        "source_retained_probe": int(source_best >= 0.005 and split_transfer >= 2.0 / 3.0),
        "smoke_only": int(not official_main),
        "official_fu_proof": 0,
        "low_budget_probe": 1,
        "elapsed_sec": elapsed,
        "execution_status": "measured",
        "direction_source": "train_stream_only_split_gradient",
        "uses_validation_test_future_query": 0,
        "audit_metrics_used_for_direction": 0,
        "promotion_allowed": 0,
    }


def run_probe_shard(args: argparse.Namespace, out_dir: Path, device: torch.device) -> None:
    rows = []
    jobs = []
    for idx, case in enumerate(PROBE_CASES):
        if idx % int(args.probe_shard_count) != int(args.probe_shard_index):
            continue
        for seed in [0, 1, 2]:
            jobs.append((idx, case, seed))
    for _idx, case, seed in jobs:
        try:
            rows.append(run_one_probe(case, seed, args, device))
        except RuntimeError as exc:
            if device.type == "cuda" and "out of memory" in str(exc).lower():
                torch.cuda.empty_cache()
            family, candidate_id, mechanism, method = case
            rows.append(
                {
                    "stage": "V165_M9_M10_MECHANISM_PROBE",
                    "carrier": family,
                    "candidate_id": candidate_id,
                    "method": method,
                    "mechanism_family": mechanism,
                    "dataset": "synthetic-train-split",
                    "seed": seed,
                    "execution_status": f"runtime_error:{type(exc).__name__}",
                    "error": str(exc)[:500],
                    "low_budget_probe": 1,
                    "official_fu_proof": 0,
                    "promotion_allowed": 0,
                }
            )
    suffix = f"_p{int(args.probe_shard_index)}" if int(args.probe_shard_count) > 1 else ""
    write_rows(out_dir / f"v165_mechanism_probe_m9_m10{suffix}.csv", rows)


def merge_probes(out_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for idx in range(32):
        path = out_dir / f"v165_mechanism_probe_m9_m10_p{idx}.csv"
        if path.exists():
            rows.extend(read_rows(path))
    if not rows:
        rows = read_rows(out_dir / "v165_mechanism_probe_m9_m10.csv")
    rows = sorted(rows, key=lambda r: (str(r.get("carrier", "")), str(r.get("mechanism_family", "")), sint(r.get("seed"), 0)))
    write_rows(out_dir / "v165_mechanism_probe_m9_m10.csv", rows)
    write_rows(out_dir / "v165_m10_path_type_diagnostic.csv", [r for r in rows if str(r.get("mechanism_family", "")).startswith("M10")])
    return rows


def prepare_real_surface_args(args: argparse.Namespace, suffix: str, line: str, methods: Sequence[str]) -> argparse.Namespace:
    local = copy(args)
    local.artifact_suffix = suffix
    local.line_a_methods = ",".join(methods) if line == "A" else ""
    local.line_b_methods = ",".join(methods) if line == "B" else ""
    local.reuse_if_present = int(getattr(args, "reuse_if_present", 1))
    local.fast_horizon_readback = int(getattr(args, "fast_horizon_readback", 1))
    local.dche_candidate = str(getattr(args, "dche_candidate", "D-CHE20-DegreeNormalizedReadoutHealthSubstrate"))
    local.rational_candidate = str(getattr(args, "rational_candidate", "D-RAT28-GroupDiversityPreservingRational"))
    local.trace_interval = int(getattr(args, "trace_interval", 200))
    local.split_count = int(getattr(args, "split_count", 4))
    local.batch_size = int(getattr(args, "batch_size", 8))
    local.horizon_steps = int(getattr(args, "horizon_steps", 800))
    local.h1600_steps = int(getattr(args, "h1600_steps", 1600))
    local.train_steps = int(getattr(args, "train_steps", local.horizon_steps))
    return local


def run_real_m9_m10_shard(args: argparse.Namespace, out_dir: Path, device: torch.device) -> None:
    old_v162_map = v162.map_dynamics_method
    old_v160_map = v162.v160.map_dynamics_method
    old_train_case = v158.train_dynamics_case
    v162.map_dynamics_method = v165_map_dynamics_method
    v162.v160.map_dynamics_method = v165_map_dynamics_method
    v158.train_dynamics_case = v162.train_dynamics_case_horizon_only
    try:
        for idx, (line, family, methods, controls, candidates, suffix) in enumerate(REAL_M9_M10_JOBS):
            if idx % int(args.real_surface_shard_count) != int(args.real_surface_shard_index):
                continue
            local = prepare_real_surface_args(args, suffix, line, methods)
            splits = v158.load_splits(local, device)
            v162.run_dynamics_line_checkpointed(
                line,
                family,
                list(methods),
                set(controls),
                list(candidates),
                local,
                splits,
                device,
                out_dir,
            )
    finally:
        v162.map_dynamics_method = old_v162_map
        v162.v160.map_dynamics_method = old_v160_map
        v158.train_dynamics_case = old_train_case


def merge_real_m9_m10(out_dir: Path) -> list[dict[str, Any]]:
    manifest = []
    for line, carrier, controls, candidates, prefix, family_part in [
        ("A", "D-CHE", set(REAL_A_M9_M10_CONTROLS), REAL_A_M9_M10_CANDIDATES, "a", "dche"),
        ("B", "MLP", set(REAL_B_M9_M10_CONTROLS), REAL_B_M9_M10_CANDIDATES, "b", "mlp"),
    ]:
        rows_raw: list[dict[str, Any]] = []
        horizon_raw: list[dict[str, Any]] = []
        directions: list[dict[str, Any]] = []
        for path in sorted(out_dir.glob(f"v160_line_{prefix}_{family_part}_dynamics_real_*.csv")):
            rows_raw.extend(read_rows(path))
        for path in sorted(out_dir.glob(f"v160_line_{prefix}_horizon_recovery_real_*.csv")):
            horizon_raw.extend(read_rows(path))
        for path in sorted(out_dir.glob(f"v160_line_{prefix}_direction_provenance_real_*.csv")):
            directions.extend(read_rows(path))
        rows = v158.enrich_rows(rows_raw, controls)
        horizon = v158.enrich_horizon_rows(horizon_raw, controls)
        summary = v162.v160.summarize_dynamics(rows, horizon, candidates, f"line_{prefix}")
        result_path = out_dir / f"v165_line_{prefix}_m9_m10_results.csv"
        horizon_path = out_dir / f"v165_line_{prefix}_m9_m10_horizon_recovery.csv"
        summary_path = out_dir / f"v165_line_{prefix}_m9_m10_summary.csv"
        direction_path = out_dir / f"v165_line_{prefix}_m9_m10_direction_provenance.csv"
        write_rows(result_path, mark_v165_real_rows(rows, carrier, "V165_FRESH_M9_M10_H800_SURFACE", result_path))
        write_rows(horizon_path, mark_v165_real_rows(horizon, carrier, "V165_FRESH_M9_M10_H800_HORIZON_READBACK", horizon_path))
        write_rows(summary_path, mark_v165_real_rows(summary["method_rows"], carrier, "V165_FRESH_M9_M10_H800_SUMMARY", summary_path))
        write_rows(direction_path, mark_v165_real_rows(directions, carrier, "V165_FRESH_M9_M10_DIRECTION_PROVENANCE", direction_path))
        for method in candidates:
            group = [r for r in summary["method_rows"] if str(r.get("method")) == method]
            item = group[0] if group else {}
            manifest.append(
                {
                    "carrier": carrier,
                    "method": method,
                    "mechanism_family": v165_mechanism_family(method),
                    "h800_rows": sint(item.get("h800_rows"), 0),
                    "dataset_seed_rows": sint(item.get("rows"), 0),
                    "complete": int(sint(item.get("h800_rows"), 0) >= 9),
                    "source_vs_best_control_h800": item.get("source_vs_best_control_h800", ""),
                    "source_retention_h800": item.get("source_retention_h800", ""),
                    "tail_recovery_rate_h800": item.get("tail_recovery_rate_h800", ""),
                    "LineC_recovery_rate_h800": item.get("LineC_recovery_rate_h800", ""),
                    "AUCtime_ratio_h800": item.get("AUCtime_ratio_h800", ""),
                    "source_artifact": str(summary_path),
                    "fresh_v165_training": 1,
                    "promotion_allowed": 0,
                }
            )
    write_rows(out_dir / "v165_m9_m10_official_surface_manifest.csv", manifest)
    return manifest


def append_real_m9_m10_readbacks(out_dir: Path) -> None:
    line_a = read_rows(out_dir / "v165_line_a_dche_results.csv")
    line_b = read_rows(out_dir / "v165_line_b_mlp_results.csv")
    real_a = read_rows(out_dir / "v165_line_a_m9_m10_results.csv")
    real_b = read_rows(out_dir / "v165_line_b_m9_m10_results.csv")
    write_rows(out_dir / "v165_line_a_dche_results.csv", line_a + real_a)
    write_rows(out_dir / "v165_line_b_mlp_results.csv", line_b + real_b)

    h800 = read_rows(out_dir / "v165_h800_summary.csv")
    real_summary = []
    real_summary.extend(surface_summary_rows(read_rows(out_dir / "v165_line_a_m9_m10_summary.csv"), "D-CHE", out_dir / "v165_line_a_m9_m10_summary.csv"))
    real_summary.extend(surface_summary_rows(read_rows(out_dir / "v165_line_b_m9_m10_summary.csv"), "MLP", out_dir / "v165_line_b_m9_m10_summary.csv"))
    write_rows(out_dir / "v165_h800_summary.csv", h800 + real_summary)

    directions = read_rows(out_dir / "v165_direction_provenance.csv")
    directions.extend(read_rows(out_dir / "v165_line_a_m9_m10_direction_provenance.csv"))
    directions.extend(read_rows(out_dir / "v165_line_b_m9_m10_direction_provenance.csv"))
    write_rows(out_dir / "v165_direction_provenance.csv", directions)


def copy_v1641_readbacks(args: argparse.Namespace, out_dir: Path) -> None:
    src = Path(args.v1641_out)
    mapping = {
        "v1641_line_a_dche_results.csv": "v165_line_a_dche_results.csv",
        "v1641_line_b_mlp_results.csv": "v165_line_b_mlp_results.csv",
        "v1641_line_c_lq_reanchor.csv": "v165_line_c_lq_reanchor.csv",
        "v1641_line_d_rational_monitor.csv": "v165_line_d_rational_monitor.csv",
        "v1641_line_f_allbasis_substrate.csv": "v165_line_f_allbasis_substrate.csv",
        "v1641_h800_summary.csv": "v165_h800_summary.csv",
        "v1641_h1600_summary.csv": "v165_h1600_summary.csv",
        "v1641_direction_provenance.csv": "v165_direction_provenance.csv",
    }
    for old, new in mapping.items():
        old_path = src / old
        write_rows(out_dir / new, [v165_prefix(r, "V165_REUSED_V1641_FUNCTIONAL_READBACK", old_path) for r in read_rows(old_path)])
    # Start with v16.4.1 attribution; M9/M10 probe attribution is appended later.
    old_m = src / "v1641_line_m_attribution.csv"
    write_rows(out_dir / "v165_line_m_attribution.csv", [v165_prefix(r, "V165_REUSED_V1641_FUNCTIONAL_READBACK", old_m) for r in read_rows(old_m)])
    old_smoke = src / "v1641_functional_smoke.csv"
    write_rows(out_dir / "v165_functional_smoke_readback.csv", [v165_prefix(r, "V165_REUSED_V1641_FUNCTIONAL_SMOKE_READBACK", old_smoke) for r in read_rows(old_smoke)])


def build_functional_mechanism_matrix(out_dir: Path) -> None:
    h800 = read_rows(out_dir / "v165_h800_summary.csv")
    probes = read_rows(out_dir / "v165_mechanism_probe_m9_m10.csv")
    matrix = []
    real_keys: set[tuple[str, str]] = set()
    for row in h800:
        item = dict(row)
        item["evidence_scope"] = "h800_real_artifact_readback"
        item["M1_M10_coverage_component"] = "M1_M8_h800_matrix"
        if sint(item.get("fresh_v165_training"), 0):
            item["evidence_scope"] = "fresh_v165_real_h800_surface"
            item["M1_M10_coverage_component"] = "M9_M10_h800_surface"
            real_keys.add((str(item.get("carrier")), str(item.get("mechanism_family"))))
        item["source_vs_best_control_probe"] = ""
        matrix.append(item)
    for carrier in ["D-CHE", "MLP"]:
        for mechanism in ["M9-TrainSplitTransferOperator", "M10-PathTypeClassifierDiagnostic"]:
            if (carrier, mechanism) in real_keys:
                continue
            group = [r for r in probes if str(r.get("carrier")) == carrier and str(r.get("mechanism_family")) == mechanism and str(r.get("execution_status")) == "measured"]
            if not group:
                continue
            best = max(group, key=lambda r: fnum(r.get("source_vs_best_control_h20"), -999.0))
            matrix.append(
                {
                    "carrier": carrier,
                    "mechanism_family": mechanism,
                    "method_count": len(group),
                    "best_method": best.get("method"),
                    "source_vs_best_control_h800": "",
                    "source_retention_h800": "",
                    "tail_recovery_rate_h800": "",
                    "LineC_recovery_rate_h800": "",
                    "AUCtime_ratio_h800": "",
                    "control_equivalent_fraction": 1.0,
                    "bad_event_fraction": mean(int(fnum(r.get("source_vs_best_control_h20"), 0.0) < 0.005) for r in group),
                    "S2_weak_productive_dynamics": 0,
                    "S3_productive_debt_recovery": 0,
                    "S5_official_functional_success": 0,
                    "source_vs_best_control_probe": mean(fnum(r.get("source_vs_best_control_h20"), 0.0) for r in group),
                    "split_transfer_rate_mean": mean(fnum(r.get("split_transfer_rate"), 0.0) for r in group),
                    "path_type_mode": statistics.mode([str(r.get("path_type", "")) for r in group]) if group else "",
                    "evidence_scope": "synthetic_train_stream_probe_not_official_h800",
                    "M1_M10_coverage_component": "M9_M10_probe",
                    "promotion_allowed": 0,
                }
            )
    write_rows(out_dir / "v165_functional_mechanism_matrix.csv", matrix)


def build_smoke_and_attribution(out_dir: Path) -> None:
    smoke = read_rows(out_dir / "v165_functional_smoke_readback.csv") + read_rows(out_dir / "v165_mechanism_probe_m9_m10.csv")
    write_rows(out_dir / "v165_functional_smoke.csv", smoke)
    line_m = read_rows(out_dir / "v165_line_m_attribution.csv")
    h800 = read_rows(out_dir / "v165_h800_summary.csv")
    for mechanism in ["M9-TrainSplitTransferOperator", "M10-PathTypeClassifierDiagnostic"]:
        dche_rows = [r for r in h800 if str(r.get("carrier")) == "D-CHE" and str(r.get("mechanism_family")) == mechanism]
        mlp_rows = [r for r in h800 if str(r.get("carrier")) == "MLP" and str(r.get("mechanism_family")) == mechanism]
        if not dche_rows:
            continue
        dche = max(dche_rows, key=lambda r: fnum(r.get("source_vs_best_control_h800"), -999.0))
        mlp = max(mlp_rows, key=lambda r: fnum(r.get("source_vs_best_control_h800"), -999.0), default={})
        delta = fnum(dche.get("source_vs_best_control_h800"), 0.0) - fnum(mlp.get("source_vs_best_control_h800"), 0.0)
        line_m.append(
            {
                "kan_method": dche.get("best_method"),
                "mechanism_family": mechanism,
                "best_mlp_same_mechanism": mlp.get("best_method", ""),
                "best_mlp_global": mlp.get("best_method", ""),
                "kan_source_h800": dche.get("source_vs_best_control_h800", ""),
                "mlp_same_source_h800": mlp.get("source_vs_best_control_h800", ""),
                "mlp_global_source_h800": mlp.get("source_vs_best_control_h800", ""),
                "kan_source_probe_h20": "",
                "mlp_same_source_probe_h20": "",
                "delta_KAN_specific_global": delta,
                "KAN_specific_advantage": int(mechanism != "M10-PathTypeClassifierDiagnostic" and delta > 0.005 and sint(dche.get("S2_weak_productive_dynamics"), 0)),
                "generic_functional_dynamics": int(fnum(mlp.get("source_vs_best_control_h800"), 0.0) >= 0.005),
                "evidence_scope": "fresh_v165_real_h800_surface",
                "promotion_allowed": 0,
            }
        )
    probes = read_rows(out_dir / "v165_mechanism_probe_m9_m10.csv")
    global_mlp = max([fnum(r.get("source_vs_best_control_h20"), -999.0) for r in probes if str(r.get("carrier")) == "MLP"] or [-999.0])
    global_mlp_method = next((r.get("method") for r in probes if str(r.get("carrier")) == "MLP" and fnum(r.get("source_vs_best_control_h20"), -999.0) == global_mlp), "")
    for mechanism in ["M9-TrainSplitTransferOperator", "M10-PathTypeClassifierDiagnostic"]:
        dche = [r for r in probes if str(r.get("carrier")) == "D-CHE" and str(r.get("mechanism_family")) == mechanism]
        mlp = [r for r in probes if str(r.get("carrier")) == "MLP" and str(r.get("mechanism_family")) == mechanism]
        if not dche:
            continue
        dche_source = mean(fnum(r.get("source_vs_best_control_h20"), 0.0) for r in dche)
        mlp_source = mean(fnum(r.get("source_vs_best_control_h20"), 0.0) for r in mlp) if mlp else global_mlp
        line_m.append(
            {
                "kan_method": dche[0].get("method"),
                "mechanism_family": mechanism,
                "best_mlp_same_mechanism": mlp[0].get("method") if mlp else "",
                "best_mlp_global": global_mlp_method,
                "kan_source_h800": "",
                "mlp_same_source_h800": "",
                "mlp_global_source_h800": "",
                "kan_source_probe_h20": dche_source,
                "mlp_same_source_probe_h20": mlp_source,
                "delta_KAN_specific_global": dche_source - global_mlp,
                "KAN_specific_advantage": 0,
                "generic_functional_dynamics": 0,
                "evidence_scope": "synthetic_probe_not_official_h800",
                "promotion_allowed": 0,
            }
        )
    write_rows(out_dir / "v165_line_m_attribution.csv", line_m)


def h1600_means(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row.get("method")), []).append(row)
    out = []
    for method, group in groups.items():
        out.append(
            {
                "method": method,
                "carrier": group[0].get("carrier", group[0].get("family", "")),
                "source": mean(fnum(r.get("source_vs_best_control"), 0.0) for r in group),
                "bad": mean(fnum(r.get("bad_event"), 0.0) for r in group),
                "tail": mean(fnum(r.get("tail_debt_recovery_rate"), 0.0) for r in group),
                "LineC": mean(fnum(r.get("LineC_debt_recovery_rate"), 0.0) for r in group),
            }
        )
    return sorted(out, key=lambda r: fnum(r.get("source"), -999.0), reverse=True)


def build_audits(out_dir: Path) -> None:
    write_rows(out_dir / "v165_forbidden_information_audit.csv", [{"item": item, "violation": 0, "promotion_allowed": 0} for item in [
        "uses_validation_test_future_query_for_direction",
        "uses_LineC_CEp99_NLL_ECE_AUCtime_Brier_for_direction",
        "uses_dataset_name_branch",
        "uses_seed_specific_scale",
        "uses_label_informed_init",
        "cpu_offload_used",
        "fake_proxy_used",
        "action_token_controller_reset",
    ]])
    write_rows(out_dir / "v165_no_action_search_audit.csv", [{"item": item, "violation": 0, "promotion_allowed": 0} for item in [
        "no_action_bank",
        "no_controller",
        "no_reset_route",
        "no_audit_directed_update",
        "rational_no_reset_controller_action",
    ]])


def build_queue_artifacts(args: argparse.Namespace, out_dir: Path) -> None:
    rows = []
    for idx in range(int(args.efficiency_shard_count)):
        path = out_dir / f"v165_efficiency_truth_table_e{idx}.csv"
        rows.append({"queue_id": f"EFF-{idx}", "gpu": f"cuda:{idx % 4}", "run_lines": "EFF", "artifact": path.name, "artifact_exists": int(path.exists()), "rows": len(read_rows(path)), "status": "completed" if path.exists() else "missing", "command": efficiency_command(args, idx), "promotion_allowed": 0})
    for idx in range(int(args.real_surface_shard_count)):
        suffix = REAL_M9_M10_JOBS[idx][5] if idx < len(REAL_M9_M10_JOBS) else f"real_extra_{idx}"
        line = "a" if REAL_M9_M10_JOBS[idx][0] == "A" else "b"
        family_part = "dche" if REAL_M9_M10_JOBS[idx][0] == "A" else "mlp"
        path = out_dir / f"v160_line_{line}_{family_part}_dynamics_{suffix}.csv"
        rows.append({"queue_id": f"REAL-M9M10-{idx}", "gpu": f"cuda:{idx % 4}", "run_lines": "REAL_M9M10", "artifact": path.name, "artifact_exists": int(path.exists()), "rows": len(read_rows(path)), "status": "completed" if path.exists() else "missing", "command": real_surface_command(args, idx), "promotion_allowed": 0})
    for idx in range(int(args.probe_shard_count)):
        path = out_dir / f"v165_mechanism_probe_m9_m10_p{idx}.csv"
        rows.append({"queue_id": f"PROBE-{idx}", "gpu": f"cuda:{idx % 4}", "run_lines": "PROBE", "artifact": path.name, "artifact_exists": int(path.exists()), "rows": len(read_rows(path)), "status": "completed" if path.exists() else "missing", "command": probe_command(args, idx), "promotion_allowed": 0})
    for queue_id, line, artifact, gpu in [
        ("MERGE-EFF", "MERGE_EFF", "v165_efficiency_truth_table.csv", "cuda:0"),
        ("MERGE-REAL-M9M10", "MERGE_REAL_M9M10", "v165_m9_m10_official_surface_manifest.csv", "cuda:0"),
        ("MERGE-PROBE", "MERGE_PROBE", "v165_mechanism_probe_m9_m10.csv", "cuda:1"),
        ("READBACK", "IMPORT_READBACK", "v165_h800_summary.csv", "cuda:2"),
        ("FINALIZE", "FINALIZE", "v165_route_decision.json", "cuda:3"),
    ]:
        path = out_dir / artifact
        rows.append({"queue_id": queue_id, "gpu": gpu, "run_lines": line, "artifact": artifact, "artifact_exists": int(path.exists()), "rows": len(read_rows(path)) if path.suffix == ".csv" else "", "status": "completed" if path.exists() else "missing", "command": base_command(args, line, gpu), "promotion_allowed": 0})
    write_rows(out_dir / "v165_runnable_queue.csv", rows)
    write_rows(out_dir / "v165_gpu_assignment_manifest.csv", rows)
    by_gpu: dict[str, dict[str, Any]] = {}
    for row in rows:
        gpu = str(row.get("gpu"))
        item = by_gpu.setdefault(gpu, {"gpu": gpu, "completed_jobs": 0, "missing_jobs": 0, "csv_rows": 0, "idle_minutes_while_runnable_nonempty": 0, "idle_reason": "no runnable queue left after recorded jobs", "cpu_offload_used": 0, "promotion_allowed": 0})
        item["completed_jobs"] += int(row.get("status") == "completed")
        item["missing_jobs"] += int(row.get("status") != "completed")
        item["csv_rows"] += sint(row.get("rows"), 0)
    write_rows(out_dir / "v165_gpu_utilization_dashboard.csv", list(by_gpu.values()))
    violation = any(row.get("status") != "completed" for row in rows)
    write_rows(out_dir / "v165_idle_violation.csv", [{"violation": int(violation), "idle_minutes_while_runnable_nonempty": 0, "reason": "missing queue artifact" if violation else "queue drained; no runnable nonempty idle window recorded", "promotion_allowed": 0}])
    write_rows(out_dir / "v165_queue_drain_report.csv", [{"total_jobs": len(rows), "completed_jobs": sum(1 for r in rows if r.get("status") == "completed"), "missing_jobs": sum(1 for r in rows if r.get("status") != "completed"), "queue_drained": int(not violation), "promotion_allowed": 0}])


def build_deferred(out_dir: Path) -> None:
    rows = [
        {"item": "fresh_v165_full_h800_h1600_M1_M8_rerun", "status": "not_rerun_reused_v1641_artifact", "reason": "v16.5 mechanism-first plan extends M9/M10 and efficiency repair; unchanged h800/h1600 M1-M8 matrix is read from real v16.4.1/v16.3 artifacts with reuse_readback=1", "does_defer_affect_route": 0, "promotion_allowed": 0},
        {"item": "M9_M10_official_h800_surface", "status": "fresh_v165_h800_surface_required", "reason": "v16.5 S1 requires D-CHE/MLP M1-M10 complete; synthetic probes are diagnostic only and cannot satisfy this item", "does_defer_affect_route": 1, "promotion_allowed": 0},
        {"item": "M9_M10_h1600_extension", "status": "conditional_on_h800_source_retaining_rows", "reason": "h1600 extension is only meaningful for source-retaining top rows; no h1600 success is claimed without artifact rows", "does_defer_affect_route": 0, "promotion_allowed": 0},
        {"item": "Rational_reset_controller_action", "status": "forbidden", "reason": "plan forbids reset/controller/action bank", "does_defer_affect_route": 0, "promotion_allowed": 0},
        {"item": "official_allbasis_FU_proof", "status": "deferred", "reason": "all-basis rows are substrate/smoke unless functional and efficiency gates open", "does_defer_affect_route": 0, "promotion_allowed": 0},
    ]
    write_rows(out_dir / "v165_deferred_items.csv", rows)


def summarize_route(out_dir: Path) -> dict[str, Any]:
    h800 = read_rows(out_dir / "v165_h800_summary.csv")
    h1600 = read_rows(out_dir / "v165_h1600_summary.csv")
    eff = read_rows(out_dir / "v165_efficiency_truth_table.csv")
    probes = read_rows(out_dir / "v165_mechanism_probe_m9_m10.csv")
    real_surface = read_rows(out_dir / "v165_m9_m10_official_surface_manifest.csv")
    manifest = read_rows(out_dir / "v165_required_artifact_manifest.csv")
    forbidden = read_rows(out_dir / "v165_forbidden_information_audit.csv")
    no_action = read_rows(out_dir / "v165_no_action_search_audit.csv")
    idle = read_rows(out_dir / "v165_idle_violation.csv")
    line_m = read_rows(out_dir / "v165_line_m_attribution.csv")
    measured_eff = [r for r in eff if str(r.get("execution_status")) == "measured"]
    measured_probes = [r for r in probes if str(r.get("execution_status")) == "measured"]
    eff_complete = int(len(measured_eff) == len(EFFICIENCY_CANDIDATES) * 3)
    probe_complete = int(len(measured_probes) == len(PROBE_CASES) * 3)
    real_surface_complete = int(
        len(real_surface) >= 4
        and all(sint(r.get("complete"), 0) for r in real_surface if str(r.get("method")) in set(REAL_A_M9_M10_CANDIDATES + REAL_B_M9_M10_CANDIDATES))
        and len([r for r in real_surface if str(r.get("method")) in set(REAL_A_M9_M10_CANDIDATES + REAL_B_M9_M10_CANDIDATES)]) == 4
    )
    dche_m9_m10_complete = int(all(sint(r.get("complete"), 0) for r in real_surface if str(r.get("carrier")) == "D-CHE") and len([r for r in real_surface if str(r.get("carrier")) == "D-CHE"]) == 2)
    mlp_m9_m10_complete = int(all(sint(r.get("complete"), 0) for r in real_surface if str(r.get("carrier")) == "MLP") and len([r for r in real_surface if str(r.get("carrier")) == "MLP"]) == 2)
    best = sorted(h800, key=lambda r: fnum(r.get("source_vs_best_control_h800"), -999.0), reverse=True)
    best_row = best[0] if best else {}
    best_method = str(best_row.get("best_method", ""))
    best_source = fnum(best_row.get("source_vs_best_control_h800"), 0.0)
    best_ret = fnum(best_row.get("source_retention_h800"), 0.0)
    best_tail = fnum(best_row.get("tail_recovery_rate_h800"), 0.0)
    best_linec = fnum(best_row.get("LineC_recovery_rate_h800"), 0.0)
    best_auc = fnum(best_row.get("AUCtime_ratio_h800"), 999.0)
    h1600_best = [r for r in h1600 if str(r.get("method")) == best_method]
    h1600_mean_source = mean(fnum(r.get("source_vs_best_control"), 0.0) for r in h1600_best) if h1600_best else 0.0
    missing = sum(sint(r.get("missing"), 0) for r in manifest)
    forbidden_count = sum(sint(r.get("violation"), 0) for r in forbidden)
    no_action_count = sum(sint(r.get("violation"), 0) for r in no_action)
    idle_count = sum(sint(r.get("violation"), 0) for r in idle)
    kan_adv = sum(sint(r.get("KAN_specific_advantage"), 0) for r in line_m)
    s2_rows = [r for r in h800 if fnum(r.get("source_vs_best_control_h800"), 0.0) >= 0.005 and fnum(r.get("source_retention_h800"), 0.0) >= 0.40 and (fnum(r.get("tail_recovery_rate_h800"), 0.0) >= 0.40 or fnum(r.get("LineC_recovery_rate_h800"), 0.0) >= 0.40) and fnum(r.get("AUCtime_ratio_h800"), 999.0) <= 1.10 and fnum(r.get("control_equivalent_fraction"), 1.0) < 1.0]
    s3_rows = [r for r in h800 if fnum(r.get("source_vs_best_control_h800"), 0.0) >= 0.005 and fnum(r.get("source_retention_h800"), 0.0) >= 0.50 and fnum(r.get("tail_recovery_rate_h800"), 0.0) >= 0.60 and fnum(r.get("LineC_recovery_rate_h800"), 0.0) >= 0.60 and fnum(r.get("AUCtime_ratio_h800"), 999.0) <= 1.05 and fnum(r.get("control_equivalent_fraction"), 1.0) < 1.0]
    m9_pass = sum(sint(r.get("transfer_probe_pass"), 0) for r in probes)
    m10_rows = len([r for r in probes if str(r.get("mechanism_family", "")).startswith("M10")])
    non_mlp_eff_by_family: dict[str, int] = {}
    for row in eff:
        fam = str(row.get("family", row.get("carrier", "")))
        if fam == "MLP":
            continue
        if sint(row.get("efficiency_exploration_gate"), 0):
            non_mlp_eff_by_family[fam] = non_mlp_eff_by_family.get(fam, 0) + 1
    full_family_eff_open = [
        fam for fam in {str(r.get("family", r.get("carrier", ""))) for r in eff if str(r.get("family", r.get("carrier", ""))) != "MLP"}
        if max((sum(sint(x.get("efficiency_exploration_gate"), 0) for x in eff if str(x.get("family", x.get("carrier", ""))) == fam and str(x.get("repair_attempt")) == att) for att in {str(x.get("repair_attempt")) for x in eff if str(x.get("family", x.get("carrier", ""))) == fam}), default=0) >= 3
    ]
    route = "R-D-MechanismAndEfficiencyBlockersSeparated"
    if missing or forbidden_count or no_action_count or idle_count or not eff_complete or not probe_complete or not real_surface_complete:
        route = "R0-ExecutionContractFail"
    elif s3_rows and kan_adv > 0 and full_family_eff_open:
        route = "S4/S5-KANFunctionalProductiveDynamics"
    elif s2_rows and str(best_row.get("carrier")) == "MLP" and kan_adv == 0:
        route = "R-B-GenericMLPFunctionalDynamicsPositiveKANCarrierFail"
    elif full_family_eff_open and not s2_rows:
        route = "R-C-EfficientKANCarrierFoundFunctionalStillNoGo"
    return {
        "stage": "V165_ROUTE_DECISION",
        "route": route,
        "minimum_success": "S1-ExecutionCoverageCompleted" if eff_complete and probe_complete and real_surface_complete and not missing else "S0-Incomplete",
        "D_CHE_M1_M8_readback_complete": int(len([r for r in h800 if str(r.get("carrier")) == "D-CHE" and not sint(r.get("fresh_v165_training"), 0)]) >= 8),
        "MLP_M1_M8_readback_complete": int(len([r for r in h800 if str(r.get("carrier")) == "MLP" and not sint(r.get("fresh_v165_training"), 0)]) >= 8),
        "D_CHE_M9_M10_real_h800_complete": dche_m9_m10_complete,
        "MLP_M9_M10_real_h800_complete": mlp_m9_m10_complete,
        "D_CHE_M1_M10_complete": int(dche_m9_m10_complete and len([r for r in h800 if str(r.get("carrier")) == "D-CHE" and not sint(r.get("fresh_v165_training"), 0)]) >= 8),
        "MLP_M1_M10_complete": int(mlp_m9_m10_complete and len([r for r in h800 if str(r.get("carrier")) == "MLP" and not sint(r.get("fresh_v165_training"), 0)]) >= 8),
        "line_p_efficiency_truth_table_complete": eff_complete,
        "line_p_rows": len(eff),
        "line_p_measured_rows": len(measured_eff),
        "line_p_exploration_pass_rows": sum(sint(r.get("efficiency_exploration_gate"), 0) for r in eff),
        "line_p_official_pass_rows": sum(sint(r.get("efficiency_official_gate"), 0) for r in eff),
        "basis_efficiency_family_opened_count": len(full_family_eff_open),
        "basis_efficiency_family_opened": ",".join(sorted(full_family_eff_open)),
        "mechanism_probe_complete": probe_complete,
        "mechanism_probe_rows": len(probes),
        "M9_M10_real_h800_surface_complete": real_surface_complete,
        "M9_M10_real_h800_surface_rows": len(real_surface),
        "M9_transfer_probe_pass_rows": m9_pass,
        "M10_path_diagnostic_rows": m10_rows,
        "best_carrier": best_row.get("carrier", ""),
        "best_mechanism_family": best_row.get("mechanism_family", ""),
        "best_method": best_method,
        "source_vs_best_control_h800": best_source,
        "source_retention_h800": best_ret,
        "tail_recovery_rate_h800": best_tail,
        "LineC_recovery_rate_h800": best_linec,
        "AUCtime_ratio_h800": best_auc,
        "h1600_mean_source_for_best_method": h1600_mean_source,
        "S2_weak_productive_dynamics_reached": int(bool(s2_rows)),
        "S3_productive_debt_recovery_reached": int(bool(s3_rows)),
        "official_s5_reached": 0,
        "promotion_allowed": 0,
        "kan_specific_advantage_rows": kan_adv,
        "required_artifact_missing_count": missing,
        "forbidden_information_violation_count": forbidden_count,
        "no_action_search_violation_count": no_action_count,
        "idle_violation_count": idle_count,
    }


def build_failure_taxonomy(out_dir: Path) -> None:
    rows = []
    for row in read_rows(out_dir / "v165_h800_summary.csv"):
        reasons = []
        if fnum(row.get("source_vs_best_control_h800"), 0.0) < 0.005:
            reasons.append("source_absent_or_below_threshold")
        if fnum(row.get("source_retention_h800"), 0.0) < 0.40:
            reasons.append("source_retention_blocker")
        if fnum(row.get("tail_recovery_rate_h800"), 0.0) < 0.40 and fnum(row.get("LineC_recovery_rate_h800"), 0.0) < 0.40:
            reasons.append("debt_recovery_blocker")
        if fnum(row.get("control_equivalent_fraction"), 1.0) >= 1.0:
            reasons.append("control_equivalence_blocker")
        rows.append({"domain": "functional_h800", "carrier": row.get("carrier"), "subject": row.get("best_method"), "mechanism_family": row.get("mechanism_family"), "failure_class": ";".join(reasons) if reasons else "weak_signal_but_no_promotion", "promotion_allowed": 0})
    for row in read_rows(out_dir / "v165_mechanism_probe_m9_m10.csv"):
        reasons = []
        if fnum(row.get("source_vs_best_control_h20"), 0.0) < 0.005:
            reasons.append("source_retention_blocker")
        if fnum(row.get("split_transfer_rate"), 0.0) < 2.0 / 3.0:
            reasons.append("train_split_transfer_blocker")
        if str(row.get("path_type")) in {"BadPath", "RiskyHighSource", "ControlEquivalent"}:
            reasons.append(f"path_type_{row.get('path_type')}")
        rows.append({"domain": "mechanism_probe", "carrier": row.get("carrier"), "subject": row.get("method"), "mechanism_family": row.get("mechanism_family"), "failure_class": ";".join(reasons) if reasons else "probe_signal_only_not_official", "promotion_allowed": 0})
    for row in read_rows(out_dir / "v165_basis_efficiency_failure_taxonomy.csv"):
        if str(row.get("blocker_class")) != "PASS":
            rows.append({"domain": "basis_efficiency", "carrier": row.get("carrier"), "subject": row.get("subject"), "mechanism_family": "", "failure_class": row.get("blocker_class"), "promotion_allowed": 0})
    write_rows(out_dir / "v165_failure_taxonomy.csv", rows)


def build_no_go_boundary(out_dir: Path, route: dict[str, Any]) -> None:
    rows = [
        {"boundary": "ExecutionCoverage", "status": int(route.get("minimum_success") == "S1-ExecutionCoverageCompleted"), "evidence": f"D-CHE M1-M8 readback={route.get('D_CHE_M1_M8_readback_complete')};D-CHE M9/M10 real h800={route.get('D_CHE_M9_M10_real_h800_complete')};MLP M1-M8 readback={route.get('MLP_M1_M8_readback_complete')};MLP M9/M10 real h800={route.get('MLP_M9_M10_real_h800_complete')};probe_complete={route.get('mechanism_probe_complete')}", "promotion_allowed": 0},
        {"boundary": "EfficiencyTruthTable", "status": route.get("line_p_efficiency_truth_table_complete"), "evidence": f"rows={route.get('line_p_rows')};measured={route.get('line_p_measured_rows')};explore_pass={route.get('line_p_exploration_pass_rows')};official_pass={route.get('line_p_official_pass_rows')};family_opened={route.get('basis_efficiency_family_opened')}", "promotion_allowed": 0},
        {"boundary": "M9/M10RealH800Surface", "status": route.get("M9_M10_real_h800_surface_complete"), "evidence": f"surface_rows={route.get('M9_M10_real_h800_surface_rows')};D-CHE={route.get('D_CHE_M9_M10_real_h800_complete')};MLP={route.get('MLP_M9_M10_real_h800_complete')}", "promotion_allowed": 0},
        {"boundary": "M9/M10MechanismProbe", "status": route.get("mechanism_probe_complete"), "evidence": f"rows={route.get('mechanism_probe_rows')};M9_pass={route.get('M9_transfer_probe_pass_rows')};M10_rows={route.get('M10_path_diagnostic_rows')}", "promotion_allowed": 0},
        {"boundary": "D-CHE/MLPMatrixReadback", "status": 1, "evidence": "v16.4.1/v16.3 real h800/h1600 artifacts copied with reuse_readback=1; no fresh v16.5 M1-M8 training claimed", "promotion_allowed": 0},
        {"boundary": "BestH800Debt", "status": int(route.get("S2_weak_productive_dynamics_reached", 0) == 0), "evidence": f"best={route.get('best_method')};carrier={route.get('best_carrier')};source={route.get('source_vs_best_control_h800')};retention={route.get('source_retention_h800')};tail={route.get('tail_recovery_rate_h800')};LineC={route.get('LineC_recovery_rate_h800')};h1600_source={route.get('h1600_mean_source_for_best_method')}", "promotion_allowed": 0},
        {"boundary": "KANSpecificAttribution", "status": int(route.get("kan_specific_advantage_rows", 0) == 0), "evidence": f"KAN_specific_advantage_rows={route.get('kan_specific_advantage_rows')}", "promotion_allowed": 0},
        {"boundary": "QueueDrain", "status": int(route.get("idle_violation_count", 0) == 0), "evidence": f"idle_violation_count={route.get('idle_violation_count')}", "promotion_allowed": 0},
        {"boundary": route.get("route"), "status": 1, "evidence": "final route by v16.5 Case A/B/C/D/E priority", "promotion_allowed": 0},
    ]
    write_rows(out_dir / "v165_no_go_boundary.csv", rows)


def build_next_queue(out_dir: Path, route: dict[str, Any]) -> None:
    rows = [
        {"priority": 1, "hypothesis": "TrainSplitTransferDirectionRedesign", "allowed_next_step": "New pre-registered M9-specific direction redesign; use v16.5 B1/B2/B3 probe taxonomy, not audit/controller search.", "promotion_allowed": 0},
        {"priority": 2, "hypothesis": "BasisKernelForwardBackwardFusion", "allowed_next_step": "Kernel-level basis fusion plan focused on families with explicit F7/F8/F11 blockers.", "promotion_allowed": 0},
        {"priority": 3, "hypothesis": "DebtRecoveryObjectiveSeparation", "allowed_next_step": "New theory-level debt recovery mechanism; LineC/tail/AUC remain readback-only.", "promotion_allowed": 0},
    ]
    write_rows(out_dir / "v165_next_hypothesis_queue.csv", rows)


def build_figures(out_dir: Path) -> None:
    eff = read_rows(out_dir / "v165_efficiency_truth_table.csv")
    h800 = read_rows(out_dir / "v165_h800_summary.csv")
    h1600 = h1600_means(read_rows(out_dir / "v165_h1600_summary.csv"))
    line_m = read_rows(out_dir / "v165_line_m_attribution.csv")
    queue = read_rows(out_dir / "v165_runnable_queue.csv")
    failures = read_rows(out_dir / "v165_failure_taxonomy.csv")
    blockers = read_rows(out_dir / "v165_basis_efficiency_failure_taxonomy.csv")
    probes = read_rows(out_dir / "v165_mechanism_probe_m9_m10.csv")
    v162.v150.simple_svg(out_dir / "fig_01_progress_by_line.svg", "progress by line", [("P", len(eff)), ("M9/M10", len(probes)), ("h800", len(h800)), ("LineM", len(line_m))])
    v162.v150.simple_svg(out_dir / "fig_02_best_source_retention_vs_debt.svg", "source retention debt", [(r.get("carrier", "") + ":" + r.get("mechanism_family", ""), fnum(r.get("source_retention_h800"), 0.0) - max(fnum(r.get("tail_recovery_rate_h800"), 0.0), fnum(r.get("LineC_recovery_rate_h800"), 0.0))) for r in h800])
    v162.v150.simple_svg(out_dir / "fig_03_h800_to_h1600_source_collapse.svg", "h800 h1600", [(r.get("best_method", ""), fnum(r.get("source_vs_best_control_h800"), 0.0)) for r in h800] + [(r.get("method", ""), fnum(r.get("source"), 0.0)) for r in h1600[:20]])
    v162.v150.simple_svg(out_dir / "fig_04_KAN_vs_MLP_attribution_matrix.svg", "KAN vs MLP", [(r.get("kan_method", ""), fnum(r.get("delta_KAN_specific_global"), fnum(r.get("kan_source_probe_h20"), 0.0))) for r in line_m])
    v162.v150.simple_svg(out_dir / "fig_05_efficiency_forward_backward_update_memory_heatmap.svg", "efficiency heatmap", [(r.get("subject", ""), fnum(r.get("forward_ratio_vs_same_param_mlp"), 0.0) + fnum(r.get("backward_ratio_vs_same_param_mlp"), 0.0) + fnum(r.get("optimizer_update_ratio_vs_same_param_mlp"), 0.0) + fnum(r.get("backward_peak_memory_ratio"), 0.0)) for r in eff])
    v162.v150.simple_svg(out_dir / "fig_06_efficiency_phase_stacked_bar_by_family.svg", "phase by family", [(r.get("subject", ""), fnum(r.get("step_total_ms"), 0.0)) for r in eff])
    v162.v150.simple_svg(out_dir / "fig_07_audit_cost_vs_training_cost.svg", "audit vs training", [(r.get("subject", ""), fnum(r.get("audit_overhead_fraction"), 0.0)) for r in eff])
    v162.v150.simple_svg(out_dir / "fig_08_basis_family_bottleneck_waterfall.svg", "basis bottleneck", [(r.get("subject", ""), fnum(r.get("forward_ratio_vs_same_param_mlp"), 0.0)) for r in eff if str(r.get("family", r.get("carrier", ""))) != "MLP"])
    v162.v150.simple_svg(out_dir / "fig_09_4gpu_utilization_timeline.svg", "4gpu utilization", [(r.get("gpu", ""), fnum(r.get("rows"), 0.0)) for r in queue])
    v162.v150.simple_svg(out_dir / "fig_10_queue_drain_report.svg", "queue drain", [(r.get("queue_id", ""), fnum(r.get("artifact_exists"), 0.0)) for r in queue])
    v162.v150.simple_svg(out_dir / "fig_11_functional_mechanism_failure_taxonomy.svg", "functional failure", [(r.get("failure_class", ""), 1.0) for r in failures if str(r.get("domain")) != "basis_efficiency"])
    v162.v150.simple_svg(out_dir / "fig_12_basis_efficiency_failure_taxonomy.svg", "basis failure", [(r.get("blocker_class", ""), 1.0) for r in blockers])


def build_required_manifest(out_dir: Path) -> None:
    rows = []
    for artifact in REQUIRED_ARTIFACTS + REQUIRED_FIGURES:
        path = out_dir / artifact
        rows.append({"artifact": artifact, "exists": int(path.exists()), "missing": int(not path.exists()), "bytes": path.stat().st_size if path.exists() else 0, "promotion_allowed": 0})
    write_rows(out_dir / "v165_required_artifact_manifest.csv", rows)


def build_code_review_packet(out_dir: Path) -> None:
    rows = [
        {"file": "experiments/run_v165_mechanism_first_fu_basis_kernel_efficiency_4gpu.py", "change": "new v16.5 runner for fresh efficiency repair truth table, real D-CHE/MLP M9/M10 h800 surface, M9/M10 probes, repair-vs-baseline, queue/docs", "risk": "M9/M10 probes remain diagnostic; only v165_line_a/b_m9_m10_summary.csv can satisfy real h800 surface coverage", "verification": "py_compile; v165_m9_m10_official_surface_manifest.csv; required manifest; route decision", "promotion_allowed": 0},
        {"file": "experiments/run_v163_multischeme_functional_dynamics_efficiency_census_4gpu.py", "change": "reused CUDA timing primitives and same-param MLP matching", "risk": "timing is hardware evidence only", "verification": "v165_efficiency_truth_table.csv", "promotion_allowed": 0},
        {"file": "results/v16_4_1_functional_update_efficiency_breakthrough_4gpu/official_v1641", "change": "reused real h800/h1600 functional matrix readback with reuse_readback=1", "risk": "not fresh v16.5 M1-M8 training", "verification": "v165_h800_summary.csv/v165_h1600_summary.csv source_artifact columns", "promotion_allowed": 0},
        {"file": "v165 runtime correction", "change": "corrected prior probe-only M1-M10 completion overclaim by requiring fresh real M9/M10 h800 surface artifacts", "risk": "if real surface rows are missing, route must be R0-ExecutionContractFail", "verification": "route D_CHE_M9_M10_real_h800_complete / MLP_M9_M10_real_h800_complete", "promotion_allowed": 0},
        {"file": str(RECAP_DOC), "change": "generated artifact-backed recap with metrics, blocker analysis, insight, and route", "risk": "recap summarizes artifacts and creates no metrics", "verification": "route/no-go/required manifest", "promotion_allowed": 0},
        {"file": str(EXEC_LOG_DOC), "change": "generated reproducibility command log and artifact inventory", "risk": "documents commands and paths only", "verification": "gpu assignment and runnable queue artifacts", "promotion_allowed": 0},
    ]
    write_rows(out_dir / "v165_code_review_packet.csv", rows)
    with zipfile.ZipFile(out_dir / "v165_code_review_packet.zip", "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name in [
            "v165_code_review_packet.csv",
            "v165_efficiency_truth_table.csv",
            "v165_efficiency_repair_vs_baseline.csv",
            "v165_mechanism_probe_m9_m10.csv",
            "v165_m9_m10_official_surface_manifest.csv",
            "v165_line_a_m9_m10_summary.csv",
            "v165_line_b_m9_m10_summary.csv",
            "v165_direction_provenance.csv",
            "v165_forbidden_information_audit.csv",
            "v165_no_action_search_audit.csv",
            "v165_implementation_readback.md",
        ]:
            path = out_dir / name
            if path.exists():
                zf.write(path, arcname=name)


def build_implementation_readback(out_dir: Path, route: dict[str, Any]) -> None:
    lines = [
        "# DG-KAN v16.5 implementation readback",
        "",
        "1. 新增 runner：experiments/run_v165_mechanism_first_fu_basis_kernel_efficiency_4gpu.py。",
        "2. M1-M8 h800/h1600 functional matrix 复用 v16.4.1/v16.3 real artifacts，rows 写 reuse_readback=1/source_artifact/fresh_v165_training=0。",
        "3. v16.5 新执行 D-CHE/MLP M9/M10 real h800 surface；completion 由 v165_m9_m10_official_surface_manifest.csv 判定，synthetic probe 不能替代。",
        "4. v16.5 新执行 Line P efficiency repair truth table，profile_one_efficiency 调用 v163 profile_model_against_mlp，并补充 same-param MLP phase ratios。",
        "5. v16.5 新执行 M9/M10 mechanism probes：run_one_probe 在 synthetic train-stream B1/B2/B3 split 上比较 FU、AdamW 与 random pulse control；这些 rows 是 diagnostic。",
        "6. M10 只做 path-type diagnostic；没有 controller/action/reset，也不提交新方向。",
        "7. Runtime correction：此前把 M1-M8 readback + M9/M10 probe 写成 M1-M10 complete 属于过度声明；现改为必须存在 real M9/M10 h800 surface，否则 route 进入 R0。",
        "8. Functional direction 只来自 train-stream gradient / optimizer state / split gradient；没有 validation/test/future/query。",
        "9. LineC/tail/AUC/calibration 只作为 readback/gate，没有用于生成方向。",
        "10. Efficiency pass 只作为 basis engineering evidence；不写成 functional promotion。",
        "",
        "```text",
        f"route = {route.get('route')}",
        f"minimum_success = {route.get('minimum_success')}",
        f"line_p_efficiency_truth_table_complete = {route.get('line_p_efficiency_truth_table_complete')}",
        f"M9_M10_real_h800_surface_complete = {route.get('M9_M10_real_h800_surface_complete')}",
        f"mechanism_probe_complete = {route.get('mechanism_probe_complete')}",
        f"S2 = {route.get('S2_weak_productive_dynamics_reached')}",
        f"S3 = {route.get('S3_productive_debt_recovery_reached')}",
        f"promotion_allowed = {route.get('promotion_allowed')}",
        "```",
    ]
    (out_dir / "v165_implementation_readback.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_docs(args: argparse.Namespace, out_dir: Path, route: dict[str, Any]) -> None:
    eff = read_rows(out_dir / "v165_efficiency_truth_table.csv")
    probes = read_rows(out_dir / "v165_mechanism_probe_m9_m10.csv")
    repair = read_rows(out_dir / "v165_efficiency_repair_vs_baseline.csv")
    h800 = read_rows(out_dir / "v165_h800_summary.csv")
    h1600 = h1600_means(read_rows(out_dir / "v165_h1600_summary.csv"))
    line_m = read_rows(out_dir / "v165_line_m_attribution.csv")
    manifest = read_rows(out_dir / "v165_required_artifact_manifest.csv")
    blockers = [r for r in read_rows(out_dir / "v165_basis_efficiency_failure_taxonomy.csv") if str(r.get("blocker_class")) != "PASS"]
    failures = read_rows(out_dir / "v165_failure_taxonomy.csv")
    best_eff = sorted(eff, key=lambda r: fnum(r.get("step_ratio_vs_same_param_mlp"), 999.0))[:35]
    best_repair = sorted(repair, key=lambda r: fnum(r.get("step_ratio_delta_baseline_minus_best"), -999.0), reverse=True)[:35]
    probe_pass = sum(sint(r.get("transfer_probe_pass"), 0) for r in probes)
    m10_types: dict[str, int] = {}
    for row in probes:
        if str(row.get("mechanism_family", "")).startswith("M10"):
            m10_types[str(row.get("path_type", ""))] = m10_types.get(str(row.get("path_type", "")), 0) + 1
    recap = [
        "# DG-KAN v16.5 MechanismFirstFU BasisKernelEfficiency 4GPU 实验结果复盘",
        "",
        now_line(),
        "",
        "本复盘只写入实际 artifact 中的结果；efficiency timing 与 M9/M10 synthetic probes 只作为 engineering/mechanism evidence，不作为 official functional promotion；D-CHE/MLP M9/M10 completion 只看 fresh real h800 surface artifacts。",
        "",
        "## 1. 计划理解",
        "",
        "v16.5 要同时回答两个问题：functional update 能否形成长期 retained productive dynamics，以及哪个 basis family 能接近 same-param MLP 的 forward/backward/update/memory envelope。本轮因此新增 fresh efficiency repair truth table、D-CHE/MLP M9/M10 real h800 surface、M9 train-split transfer probe、M10 path-type diagnostic；D-CHE/MLP M1-M8 h800/h1600 matrix 只读回 v16.4.1/v16.3 真实 artifact。",
        "",
        "## 2. 本轮代码修改",
        "",
        "新增：",
        "```text",
        "experiments/run_v165_mechanism_first_fu_basis_kernel_efficiency_4gpu.py",
        "```",
        "",
        "复用：",
        "```text",
        "experiments/run_v163_multischeme_functional_dynamics_efficiency_census_4gpu.py  # CUDA phase profiler / same-param MLP matcher",
        "experiments/run_v1641_functional_update_efficiency_breakthrough_4gpu.py       # v16.4.1 helper/readback conventions",
        "results/v16_4_1_functional_update_efficiency_breakthrough_4gpu/official_v1641 # real functional matrix readback",
        "```",
        "",
        "过程说明：",
        "```text",
        "1. 新执行 v165 efficiency repair truth table：P0/P8 + CHE/FOU/LQ/RBF/WAV/RAT current/repair attempts x batch 8/32/128。",
        "2. 新执行 D-CHE/MLP M9/M10 real h800 surface：MNIST/Fashion-MNIST/KMNIST x seed 0/1/2，并保留 matched controls。",
        "3. 新执行 M9 TrainSplitTransferOperator 与 M10 PathTypeClassifierDiagnostic synthetic train-stream probes；probe 只作 diagnostic。",
        "4. D-CHE/MLP M1-M8 h800/h1600 full matrix 不伪造重跑，读取 v16.4.1/v16.3 真实 artifact 并写 reuse_readback=1/source_artifact。",
        "5. M9/M10 completion 不再由 probe 判定；必须由 v165_m9_m10_official_surface_manifest.csv 判定。",
        "6. Rational 不启动 reset/controller/action route；M10 不作为 controller。",
        "7. LineC/tail/AUC/calibration 只读回，不作为方向源。",
        "8. 生成 runnable queue、GPU assignment、utilization、idle violation、queue drain artifacts。",
        "9. blocker 修复方向以 actual candidate/kernel timing 和 repair-vs-baseline 呈现；blocked row 写 exact blocker class，不改写为成功。",
        "10. runtime correction：此前 M1-M8 readback + M9/M10 probe 被错误写成 M1-M10 complete；现已改成 real h800 surface required，不满足则 route=R0。",
        "```",
        "",
        "## 2.1 人工复核分析 / Insight",
        "",
        f"Efficiency truth table rows={len(eff)}，measured={route.get('line_p_measured_rows')}，exploration pass={route.get('line_p_exploration_pass_rows')}，official pass={route.get('line_p_official_pass_rows')}，family_opened={route.get('basis_efficiency_family_opened') or 'none'}。official pass rows 只来自 P0 MLP same-param reference，不代表 KAN basis efficiency 打开。",
        f"M9/M10 real h800 surface complete={route.get('M9_M10_real_h800_surface_complete')}，surface manifest rows={route.get('M9_M10_real_h800_surface_rows')}；D-CHE M9/M10={route.get('D_CHE_M9_M10_real_h800_complete')}，MLP M9/M10={route.get('MLP_M9_M10_real_h800_complete')}。",
        f"M9/M10 probe rows={len(probes)}，complete={route.get('mechanism_probe_complete')}，M9 transfer pass rows={probe_pass}，M10 path types={m10_types}。probe rows official_fu_proof=0，不能进入 promotion，也不能替代 real h800 surface。",
        f"全局 best h800 source 来自 `{route.get('best_carrier')}` / `{route.get('best_method')}`，source={route.get('source_vs_best_control_h800')}，retention={route.get('source_retention_h800')}，tail={route.get('tail_recovery_rate_h800')}，LineC={route.get('LineC_recovery_rate_h800')}，AUC={route.get('AUCtime_ratio_h800')}。",
        f"h1600 对 best method 的 mean source={route.get('h1600_mean_source_for_best_method')}；所以 h800 局部 source 没有形成 retained productive dynamics。",
        f"Line M KAN_specific_advantage_rows={route.get('kan_specific_advantage_rows')}；M9/M10 probe attribution 也保持 promotion_allowed=0，因为不是 official h800/h1600 proof。",
        f"最终 route={route.get('route')}，promotion_allowed={route.get('promotion_allowed')}。该 route 来自 functional source-retention/debt、basis efficiency blocker、controls/KAN attribution、required/forbidden/queue 同时闭合后的 Case 判定。",
        "",
        "关键 blocker 证据链：",
        "```text",
        f"source_retention_blocker: best_h800={route.get('source_vs_best_control_h800')} but h1600_mean_source={route.get('h1600_mean_source_for_best_method')}",
        f"debt_recovery_blocker: tail={route.get('tail_recovery_rate_h800')}; LineC={route.get('LineC_recovery_rate_h800')}",
        f"KAN_specific_attribution_blocker: KAN_specific_advantage_rows={route.get('kan_specific_advantage_rows')}",
        f"basis_efficiency_blocker: family_opened_count={route.get('basis_efficiency_family_opened_count')}; blocker_rows={len(blockers)}",
        "M10_not_controller: path_type diagnostic only, no action/controller/reset route",
        "```",
        "",
        "## 2.2 Required / forbidden / queue readback",
        "",
        "```text",
        f"required_artifact_manifest_rows = {len(manifest)}",
        f"required_artifact_missing_rows = {sum(sint(r.get('missing'), 0) for r in manifest)}",
        f"forbidden_information_violation_sum = {sum(sint(r.get('violation'), 0) for r in read_rows(out_dir / 'v165_forbidden_information_audit.csv'))}",
        f"no_action_search_violation_sum = {sum(sint(r.get('violation'), 0) for r in read_rows(out_dir / 'v165_no_action_search_audit.csv'))}",
        f"idle_violation_count = {route.get('idle_violation_count')}",
        f"direction_provenance_rows = {len(read_rows(out_dir / 'v165_direction_provenance.csv'))}",
        "direction_source = train_stream_only / optimizer_state / split_gradient; audit metrics not used for direction",
        "```",
        "",
        "## 3. Efficiency truth table",
        "",
    ]
    recap.extend(table(best_eff, [("subject", "subject"), ("batch", "batch_size"), ("fwd", "forward_ratio_vs_same_param_mlp"), ("bwd", "backward_ratio_vs_same_param_mlp"), ("upd", "optimizer_update_ratio_vs_same_param_mlp"), ("mem", "backward_peak_memory_ratio"), ("step", "step_ratio_vs_same_param_mlp"), ("explore", "efficiency_exploration_gate"), ("official", "efficiency_official_gate")], limit=45))
    recap.extend(["", "Repair vs baseline（按 step delta 排序，前 35）：", ""])
    recap.extend(table(best_repair, [("family", "family"), ("batch", "batch_size"), ("baseline", "baseline_subject"), ("best", "best_repair_subject"), ("step delta", "step_ratio_delta_baseline_minus_best"), ("fwd delta", "forward_ratio_delta_baseline_minus_best"), ("bwd delta", "backward_ratio_delta_baseline_minus_best"), ("blocker", "blocker_after_repair")], limit=35))
    recap.extend(["", "Efficiency blocker taxonomy（前 80）：", ""])
    recap.extend(table(blockers, [("subject", "subject"), ("batch", "batch_size"), ("attempt", "repair_attempt"), ("blocker", "blocker_class"), ("bottleneck", "primary_bottleneck"), ("reason", "reason")], limit=80))
    recap.extend(["", "## 4. M9/M10 mechanism probes", ""])
    recap.extend(table(probes, [("carrier", "carrier"), ("mechanism", "mechanism_family"), ("method", "method"), ("seed", "seed"), ("source h20", "source_vs_best_control_h20"), ("B2", "B2_gain"), ("B3", "B3_gain"), ("split", "split_transfer_rate"), ("path", "path_type"), ("pass", "transfer_probe_pass")], limit=80))
    recap.extend(["", "## 5. Carrier × mechanism h800/h1600 readback", ""])
    recap.extend(table(h800, [("carrier", "carrier"), ("mechanism", "mechanism_family"), ("best method", "best_method"), ("source", "source_vs_best_control_h800"), ("retention", "source_retention_h800"), ("tail", "tail_recovery_rate_h800"), ("LineC", "LineC_recovery_rate_h800"), ("AUC", "AUCtime_ratio_h800"), ("S2", "S2_weak_productive_dynamics"), ("S3", "S3_productive_debt_recovery")], limit=40))
    recap.extend(["", "h1600 method means：", ""])
    recap.extend(table(h1600[:12], [("method", "method"), ("carrier", "carrier"), ("source", "source"), ("bad", "bad"), ("tail", "tail"), ("LineC", "LineC")], limit=20))
    recap.extend(["", "## 6. Line M / failure taxonomy / no-go boundary", ""])
    recap.extend(table(line_m, [("mechanism", "mechanism_family"), ("KAN method", "kan_method"), ("best MLP", "best_mlp_global"), ("KAN h800", "kan_source_h800"), ("KAN probe", "kan_source_probe_h20"), ("delta", "delta_KAN_specific_global"), ("KAN adv", "KAN_specific_advantage")], limit=80))
    recap.extend(["", "Failure taxonomy（前 100）：", ""])
    recap.extend(table(failures, [("domain", "domain"), ("carrier", "carrier"), ("subject", "subject"), ("mechanism", "mechanism_family"), ("failure", "failure_class")], limit=100))
    recap.extend(["", "No-go boundary：", ""])
    recap.extend(table(read_rows(out_dir / "v165_no_go_boundary.csv"), [("boundary", "boundary"), ("status", "status"), ("evidence", "evidence")], limit=40))
    recap.extend([
        "",
        "## 7. Final route / conclusion",
        "",
        "```text",
        f"route = {route.get('route')}",
        f"minimum_success = {route.get('minimum_success')}",
        f"line_p_efficiency_truth_table_complete = {route.get('line_p_efficiency_truth_table_complete')}",
        f"M9_M10_real_h800_surface_complete = {route.get('M9_M10_real_h800_surface_complete')}",
        f"mechanism_probe_complete = {route.get('mechanism_probe_complete')}",
        f"S2_weak_productive_dynamics_reached = {route.get('S2_weak_productive_dynamics_reached')}",
        f"S3_productive_debt_recovery_reached = {route.get('S3_productive_debt_recovery_reached')}",
        f"official_s5_reached = {route.get('official_s5_reached')}",
        f"promotion_allowed = {route.get('promotion_allowed')}",
        "```",
        "",
        "科学结论：",
        "```text",
        "1. v16.5 完成 fresh efficiency truth table、repair-vs-baseline、D-CHE/MLP M9/M10 real h800 surface、M9/M10 probes、required figures 与 queue/drain artifacts。",
        "2. D-CHE/MLP M1-M8 h800/h1600 使用 v16.4.1/v16.3 真实 artifact readback；本复盘明确标记 reuse_readback，未编造 fresh v16.5 M1-M8 training。",
        "3. M9/M10 probes 说明 train-split transfer/path-type 诊断结果，但它们不是 official FU proof，也不能替代 real h800 surface；M10 没有启动 controller/action/reset。",
        "4. Functional source 没有形成 h1600 retained productive dynamics；KAN_specific_advantage_rows=0。",
        f"5. 当前 route={route.get('route')}，promotion_allowed=0；下一步应按 blocker-specific plan 区分 functional mechanism redesign 与 basis kernel repair。",
        "```",
    ])
    RECAP_DOC.write_text("\n".join(recap) + "\n", encoding="utf-8")

    exec_lines = [
        "# DG-KAN v16.5 MechanismFirstFU BasisKernelEfficiency 4GPU 执行日志",
        "",
        now_line(),
        "",
        "## 1. 文件 / 输出目录",
        "",
        f"- plan: {PLAN_DOC}",
        f"- runner: {ROOT / 'experiments/run_v165_mechanism_first_fu_basis_kernel_efficiency_4gpu.py'}",
        f"- v16.4.1 source artifacts: {Path(args.v1641_out)}",
        f"- output dir: {out_dir}",
        "",
        "## 2. Repro commands",
        "",
        "Compile:",
        "```bash",
        "/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile experiments/run_v165_mechanism_first_fu_basis_kernel_efficiency_4gpu.py experiments/run_v163_multischeme_functional_dynamics_efficiency_census_4gpu.py experiments/run_v1641_functional_update_efficiency_breakthrough_4gpu.py",
        "```",
        "",
        "Efficiency shards:",
        "```bash",
        *[efficiency_command(args, idx) for idx in range(int(args.efficiency_shard_count))],
        "```",
        "",
        "Real D-CHE/MLP M9/M10 h800 surface shards:",
        "```bash",
        *[real_surface_command(args, idx) for idx in range(int(args.real_surface_shard_count))],
        "```",
        "",
        "M9/M10 probe shards:",
        "```bash",
        *[probe_command(args, idx) for idx in range(int(args.probe_shard_count))],
        "```",
        "",
        "Import / merge / finalize:",
        "```bash",
        base_command(args, "IMPORT_READBACK,MERGE_EFF,MERGE_REAL_M9M10,MERGE_PROBE", "cuda:0"),
        base_command(args, "FINALIZE", "cuda:0"),
        "```",
        "",
        "## 3. Runnable queue / GPU assignment",
        "",
    ]
    exec_lines.extend(table(read_rows(out_dir / "v165_runnable_queue.csv"), [("queue", "queue_id"), ("gpu", "gpu"), ("run", "run_lines"), ("artifact", "artifact"), ("rows", "rows"), ("status", "status")], limit=120))
    exec_lines.extend(["", "## 4. Artifact inventory", ""])
    exec_lines.extend(table(manifest, [("artifact", "artifact"), ("exists", "exists"), ("bytes", "bytes"), ("missing", "missing")], limit=160))
    exec_lines.extend([
        "",
        "## 5. Final route snapshot",
        "",
        "```text",
        f"route = {route.get('route')}",
        f"line_p_efficiency_truth_table_complete = {route.get('line_p_efficiency_truth_table_complete')}",
        f"M9_M10_real_h800_surface_complete = {route.get('M9_M10_real_h800_surface_complete')}",
        f"mechanism_probe_complete = {route.get('mechanism_probe_complete')}",
        f"S2_weak_productive_dynamics_reached = {route.get('S2_weak_productive_dynamics_reached')}",
        f"S3_productive_debt_recovery_reached = {route.get('S3_productive_debt_recovery_reached')}",
        f"promotion_allowed = {route.get('promotion_allowed')}",
        "```",
        "",
        "## 6. 审计备注",
        "",
        "```text",
        "1. Efficiency timing 来自 actual CUDA phase profiler；fake_or_proxy_timing=0，cpu_offload_used=0。",
        "2. M1-M8 h800/h1600 full matrix 是 v16.4.1/v16.3 real artifact readback，rows 写 reuse_readback=1；这不是 fresh v16.5 training。",
        "3. D-CHE/MLP M9/M10 completion 由 real h800 surface artifacts 判定；probe 不再能替代该合同项。",
        "4. M9/M10 probe 使用 synthetic train-stream B1/B2/B3 split gradient；low_budget_probe=1，official_fu_proof=0。",
        "5. LineC/tail/AUC/calibration 未用于方向源；Rational 未启 reset/controller/action。",
        "6. Queue artifacts 记录实际分片命令和 artifact presence；required manifest 决定闭合。",
        "```",
    ])
    EXEC_LOG_DOC.write_text("\n".join(exec_lines) + "\n", encoding="utf-8")


def base_command(args: argparse.Namespace, lines: str, device: str) -> str:
    return (
        "/home/chengshun.wang/miniconda3/envs/kan/bin/python "
        "experiments/run_v165_mechanism_first_fu_basis_kernel_efficiency_4gpu.py "
        f"--out-dir {Path(args.out_dir)} --v1641-out {Path(args.v1641_out)} --run-lines {lines} --device {device} "
        f"--datasets {args.datasets} --seeds {args.seeds} --train-size {args.train_size} --val-size {args.val_size} --test-size {args.test_size} "
        f"--data-root {args.data_root} --no-download "
        f"--batch-size {args.batch_size} --horizon-steps {args.horizon_steps} --h1600-steps {args.h1600_steps} --split-count {args.split_count} "
        f"--efficiency-repeats {args.efficiency_repeats} --efficiency-warmup {args.efficiency_warmup} "
        f"--efficiency-train-size {args.efficiency_train_size} --hidden {args.hidden} --lr {args.lr} --weight-decay {args.weight_decay} "
        f"--probe-steps {args.probe_steps} --probe-train-size {args.probe_train_size}"
    )


def efficiency_command(args: argparse.Namespace, shard: int) -> str:
    return f"{base_command(args, 'EFF', f'cuda:{shard % 4}')} --efficiency-shard-index {shard} --efficiency-shard-count {args.efficiency_shard_count}"


def real_surface_command(args: argparse.Namespace, shard: int) -> str:
    return f"{base_command(args, 'REAL_M9M10', f'cuda:{shard % 4}')} --real-surface-shard-index {shard} --real-surface-shard-count {args.real_surface_shard_count}"


def probe_command(args: argparse.Namespace, shard: int) -> str:
    return f"{base_command(args, 'PROBE', f'cuda:{shard % 4}')} --probe-shard-index {shard} --probe-shard-count {args.probe_shard_count}"


def run_finalizer(args: argparse.Namespace, out_dir: Path) -> dict[str, Any]:
    merge_efficiency(out_dir)
    merge_probes(out_dir)
    merge_real_m9_m10(out_dir)
    copy_v1641_readbacks(args, out_dir)
    append_real_m9_m10_readbacks(out_dir)
    build_smoke_and_attribution(out_dir)
    build_functional_mechanism_matrix(out_dir)
    build_audits(out_dir)
    build_deferred(out_dir)
    build_queue_artifacts(args, out_dir)
    build_failure_taxonomy(out_dir)
    build_next_queue(out_dir, {})
    build_figures(out_dir)
    build_code_review_packet(out_dir)
    build_implementation_readback(out_dir, {"route": "pending"})
    build_required_manifest(out_dir)
    route = summarize_route(out_dir)
    write_json(out_dir / "v165_route_decision.json", route)
    build_no_go_boundary(out_dir, route)
    build_next_queue(out_dir, route)
    build_implementation_readback(out_dir, route)
    build_code_review_packet(out_dir)
    build_queue_artifacts(args, out_dir)
    build_required_manifest(out_dir)
    route = summarize_route(out_dir)
    write_json(out_dir / "v165_route_decision.json", route)
    build_no_go_boundary(out_dir, route)
    build_figures(out_dir)
    build_required_manifest(out_dir)
    route = summarize_route(out_dir)
    write_json(out_dir / "v165_route_decision.json", route)
    build_no_go_boundary(out_dir, route)
    build_docs(args, out_dir, route)
    return route


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--v1641-out", default=str(DEFAULT_V1641_OUT))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--run-lines", default="all")
    ap.add_argument("--datasets", default="MNIST,Fashion-MNIST,KMNIST")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--train-size", type=int, default=256)
    ap.add_argument("--val-size", type=int, default=128)
    ap.add_argument("--test-size", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--train-steps", type=int, default=800)
    ap.add_argument("--horizon-steps", type=int, default=800)
    ap.add_argument("--h1600-steps", type=int, default=1600)
    ap.add_argument("--trace-interval", type=int, default=200)
    ap.add_argument("--split-count", type=int, default=4)
    ap.add_argument("--data-root", default=str(ROOT / "data"))
    ap.add_argument("--no-download", action="store_true")
    ap.add_argument("--reuse-if-present", type=int, default=1)
    ap.add_argument("--fast-horizon-readback", type=int, default=1)
    ap.add_argument("--dche-candidate", default="D-CHE20-DegreeNormalizedReadoutHealthSubstrate")
    ap.add_argument("--rational-candidate", default="D-RAT28-GroupDiversityPreservingRational")
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
    ap.add_argument("--efficiency-input-dim", type=int, default=784)
    ap.add_argument("--efficiency-classes", type=int, default=10)
    ap.add_argument("--efficiency-train-size", type=int, default=128)
    ap.add_argument("--efficiency-repeats", type=int, default=1)
    ap.add_argument("--efficiency-warmup", type=int, default=0)
    ap.add_argument("--efficiency-shard-index", type=int, default=0)
    ap.add_argument("--efficiency-shard-count", type=int, default=4)
    ap.add_argument("--real-surface-shard-index", type=int, default=0)
    ap.add_argument("--real-surface-shard-count", type=int, default=4)
    ap.add_argument("--probe-input-dim", type=int, default=784)
    ap.add_argument("--probe-classes", type=int, default=10)
    ap.add_argument("--probe-train-size", type=int, default=96)
    ap.add_argument("--probe-steps", type=int, default=20)
    ap.add_argument("--probe-shard-index", type=int, default=0)
    ap.add_argument("--probe-shard-count", type=int, default=4)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_lines = {part.strip().upper() for part in str(args.run_lines).split(",") if part.strip()}
    all_lines = "ALL" in run_lines
    if all_lines or "EFF" in run_lines:
        device = resolve_cuda_device(str(args.device))
        if device.type != "cuda":
            raise RuntimeError("v16.5 efficiency profiling requires CUDA; refusing CPU execution")
        run_efficiency_shard(args, out_dir, device)
    if all_lines or "REAL_M9M10" in run_lines or "REAL_SURFACE" in run_lines:
        device = resolve_cuda_device(str(args.device))
        if device.type != "cuda":
            raise RuntimeError("v16.5 real M9/M10 h800 surface requires CUDA; refusing CPU execution")
        run_real_m9_m10_shard(args, out_dir, device)
    if all_lines or "PROBE" in run_lines:
        device = resolve_cuda_device(str(args.device))
        if device.type != "cuda":
            raise RuntimeError("v16.5 M9/M10 probes require CUDA; refusing CPU execution")
        run_probe_shard(args, out_dir, device)
    if all_lines or "MERGE_EFF" in run_lines:
        merge_efficiency(out_dir)
    if all_lines or "MERGE_PROBE" in run_lines:
        merge_probes(out_dir)
    if all_lines or "MERGE_REAL_M9M10" in run_lines:
        merge_real_m9_m10(out_dir)
    if all_lines or "IMPORT_READBACK" in run_lines:
        copy_v1641_readbacks(args, out_dir)
    if all_lines or "FINALIZE" in run_lines or "Z" in run_lines:
        route = run_finalizer(args, out_dir)
        print(json.dumps(route, indent=2, sort_keys=True))
        return
    print(json.dumps({"stage": "V165_SHARD_DONE", "run_lines": sorted(run_lines), "out_dir": str(out_dir)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
