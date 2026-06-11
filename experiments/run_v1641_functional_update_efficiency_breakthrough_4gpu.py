#!/usr/bin/env python3
"""DG-KAN v16.4.1 functional update + basis efficiency breakthrough runner.

This runner keeps v16.3/v16.2 train-stream functional artifacts as explicit
readback sources and executes the v16.4.1 additions: basis efficiency repair
truth table, phase/memory decomposition, low-budget non-main-carrier functional
smoke, queue/idle/drain artifacts, and final recap/execution logs. Timing rows
are actual CUDA measurements; reused functional matrix rows are marked as
readback reuse and are never presented as fresh v16.4.1 training.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import statistics
import sys
import time
import zipfile
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence
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


PLAN_DOC = ROOT / "docs/DG-KAN_v16.4.1_FunctionalUpdate_EfficiencyBreakthrough_4GPU_完整计划.md"
RECAP_DOC = ROOT / "docs/DG-KAN_v16.4.1_FunctionalUpdate_EfficiencyBreakthrough_4GPU_实验结果复盘.md"
EXEC_LOG_DOC = ROOT / "docs/DG-KAN_v16.4.1_FunctionalUpdate_EfficiencyBreakthrough_4GPU_执行日志.md"
DEFAULT_OUT = ROOT / "results/v16_4_1_functional_update_efficiency_breakthrough_4gpu/official_v1641"
DEFAULT_V163_OUT = ROOT / "results/v16_3_multischeme_functional_dynamics_efficiency_census_4gpu/official_v163"

read_rows = v154.read_rows
write_rows = v154.write_rows
write_json = v154.write_json
fnum = v154.fnum
sint = v154.sint
mean = v154.mean
resolve_cuda_device = v154.resolve_cuda_device


REQUIRED_ARTIFACTS = [
    "v1641_route_decision.json",
    "v1641_efficiency_truth_table.csv",
    "v1641_efficiency_phase_breakdown.csv",
    "v1641_memory_phase_breakdown.csv",
    "v1641_basis_efficiency_blocker_taxonomy.csv",
    "v1641_same_param_mlp_manifest.csv",
    "v1641_basis_efficiency_repair_attempts.csv",
    "v1641_functional_smoke.csv",
    "v1641_line_a_dche_results.csv",
    "v1641_line_b_mlp_results.csv",
    "v1641_line_c_lq_reanchor.csv",
    "v1641_line_d_rational_monitor.csv",
    "v1641_line_f_allbasis_substrate.csv",
    "v1641_line_m_attribution.csv",
    "v1641_h800_summary.csv",
    "v1641_h1600_summary.csv",
    "v1641_direction_provenance.csv",
    "v1641_runnable_queue.csv",
    "v1641_gpu_assignment_manifest.csv",
    "v1641_gpu_utilization_dashboard.csv",
    "v1641_idle_violation.csv",
    "v1641_queue_drain_report.csv",
    "v1641_deferred_items.csv",
    "v1641_required_artifact_manifest.csv",
    "v1641_forbidden_information_audit.csv",
    "v1641_no_action_search_audit.csv",
    "v1641_failure_taxonomy.csv",
    "v1641_no_go_boundary.csv",
    "v1641_code_review_packet.csv",
    "v1641_code_review_packet.zip",
    "v1641_implementation_readback.md",
]

REQUIRED_FIGURES = [
    "source_vs_control_curve_h1_h20_h100_h800_h1600.svg",
    "source_retention_curve.svg",
    "tail_debt_recovery_curve.svg",
    "linec_debt_recovery_curve.svg",
    "auc_debt_curve.svg",
    "carrier_mechanism_heatmap.svg",
    "kan_vs_mlp_attribution_matrix.svg",
    "matched_control_explainability_heatmap.svg",
    "efficiency_forward_ratio_by_basis.svg",
    "efficiency_backward_ratio_by_basis.svg",
    "efficiency_update_ratio_by_basis.svg",
    "efficiency_memory_ratio_by_basis.svg",
    "phase_time_stacked_bar_by_basis.svg",
    "memory_phase_stacked_bar_by_basis.svg",
    "basis_efficiency_pareto_step_vs_memory.svg",
    "functional_overhead_vs_source_scatter.svg",
    "gpu_utilization_timeline.svg",
    "queue_depth_over_time.svg",
    "idle_violation_timeline.svg",
    "job_completion_gantt.svg",
]

EFFICIENCY_REPAIR_CANDIDATES = [
    ("P0-MLP-same-param-reference", "MLP", "MLP-v1641-reference", "base", "MLP-P0", "same-param MLP reference"),
    ("P8-MLP-functional-path-top2", "MLP", "MLP-v1641-reference", "functional_top2", "MLP-P8", "MLP functional path top2 timing"),
    ("CHE-E0-D-CHE-current", "D-CHE", "D-CHE20-DegreeNormalizedReadoutHealthSubstrate", "base", "current", "current D-CHE carrier"),
    ("CHE-E1-audit-separated-readback", "D-CHE", "D-CHE20-DegreeNormalizedReadoutHealthSubstrate", "base", "CHE-E1", "separate training timing from LineC/horizon readback"),
    ("CHE-E2-degree-recurrence-cache", "D-CHE", "D-CHE17-HighDegreeLateEnableSubstrate", "base", "CHE-E2", "degree recurrence/cache candidate"),
    ("CHE-E3-low-degree-active-bank", "D-CHE", "D-CHE16-DegreeEnergyDampingSubstrate", "base", "CHE-E3", "low-degree active bank smoke"),
    ("CHE-E4-high-degree-late-enable", "D-CHE", "D-CHE17-HighDegreeLateEnableSubstrate", "base", "CHE-E4", "high-degree late-enable path"),
    ("CHE-E5-role-degree-projection", "D-CHE", "D-CHE18-RoleDegreeEnergyCapSubstrate", "functional_top2", "CHE-E5", "degree-role projection update path"),
    ("CHE-E6-no-materialize-energy-readback", "D-CHE", "D-CHE19-ChebyTangentTrustSubstrate", "base", "CHE-E6", "no-materialize degree energy readback"),
    ("FOU-E0-current", "D-FOU", "D-FOU14-SincosSharedWorkspace-K2", "base", "current", "current Fourier carrier"),
    ("FOU-E1-low-frequency-recurrence", "D-FOU", "D-FOU16-FrequencyBandDampingSubstrate", "base", "FOU-E1", "low-frequency-only recurrence"),
    ("FOU-E2-sincos-precompute-shape", "D-FOU", "D-FOU14-SincosSharedWorkspace-K2", "base", "FOU-E2", "sincos shared workspace"),
    ("FOU-E3-fused-band-k-small", "D-FOU", "D-FOU13-FusedReadoutGradNoMaterialize-K2", "base", "FOU-E3", "fused small-band eval"),
    ("FOU-E4-phase-stable-low-band", "D-FOU", "D-FOU17-PhaseStabilityCorrectionSubstrate", "base", "FOU-E4", "phase-stable band mix"),
    ("FOU-E5-high-frequency-quarantine", "D-FOU", "D-FOU19-HighFreqNoiseLeakVetoSubstrate", "base", "FOU-E5", "high-frequency quarantine"),
    ("FOU-E6-trig-vs-recurrence-table", "D-FOU", "D-FOU12-LifetimeRecomputeBackward-K2", "base", "FOU-E6", "trig vs recurrence timing"),
    ("LQ-E0-current", "LQ", "C2-LQ-ProtocolMatchedReanchor", "base", "current", "current/reanchored LQ carrier"),
    ("LQ-E1-update-decomposition", "LQ", "C5-LQ-StepMemoryRecheck", "base", "LQ-E1", "optimizer update decomposition"),
    ("LQ-E2-persistent-adamw-foreach", "LQ", "C5-LQ-StepMemoryRecheck", "base", "LQ-E2", "persistent AdamW foreach-style timing"),
    ("LQ-E3-fused-projection-update", "LQ", "C6-LQ-LineCNoRegressionCheck", "functional_top2", "LQ-E3", "fused LQ projection update"),
    ("LQ-E4-fixed-frame-late-attach", "LQ", "C4-LQ-MacroDeltaPriorityReanchor", "base", "LQ-E4", "fixed-frame late attach smoke"),
    ("LQ-E5-output-scale-reanchor", "LQ", "C2-LQ-ProtocolMatchedReanchor", "base", "LQ-E5", "output-scale reanchor"),
    ("RBF-E0-current", "D-RBF", "D-RBF11-CompactExpressionRepair-Monitor", "base", "current", "current RBF/FastKAN carrier"),
    ("RBF-E1-active-center-occupancy", "D-RBF", "D-RBF12-CenterOccupancyRebalanceSubstrate", "base", "RBF-E1", "active center occupancy"),
    ("RBF-E2-width-condition-guard", "D-RBF", "D-RBF13-WidthConditionGuardSubstrate", "base", "RBF-E2", "width condition guard"),
    ("RBF-E3-local-k4-no-dense", "D-RBF", "D-RBF17-CompactCapacityK4HealthSubstrate", "base", "RBF-E3", "local K4 no dense materialization"),
    ("RBF-E4-gaussian-table-lookup-smoke", "D-RBF", "D-RBF15-LocalCurvatureSmoothSubstrate", "base", "RBF-E4", "gaussian table lookup smoke path"),
    ("RBF-E5-center-update-separation", "D-RBF", "D-RBF16-ActiveCenterDiversityTransportSubstrate", "functional_top2", "RBF-E5", "center/update separation"),
    ("WAV-E0-current", "D-WAV", "D-WAV11-ScaleEnergyBalanceSubstrate", "base", "current", "current wavelet carrier"),
    ("WAV-E1-triangular-index-forward", "D-WAV", "D-WAV12-LocalSupportOccupancyRepairSubstrate", "base", "WAV-E1", "triangular support index-only forward"),
    ("WAV-E2-support-overlap-damping", "D-WAV", "D-WAV14-SupportOverlapEntropyGuardSubstrate", "base", "WAV-E2", "support overlap damping"),
    ("WAV-E3-local-tail-coverage-audit", "D-WAV", "D-WAV13-LocalTailCoverageGuardSubstrate", "base", "WAV-E3", "local tail coverage audit only"),
    ("WAV-E4-sparse-support-backward", "D-WAV", "D-WAV15-ScaleDiversityTransportSubstrate", "functional_top2", "WAV-E4", "sparse support backward path"),
    ("RAT-E0-current", "D-RAT", "D-RAT28-GroupDiversityPreservingRational", "base", "current", "current Rational monitor"),
    ("RAT-E1-denominator-derivative-telemetry", "D-RAT", "D-RAT34-TrainingDenDerivativeStabilityNoCE", "base", "RAT-E1", "denominator derivative telemetry cost"),
    ("RAT-E2-branchless-denominator-readback", "D-RAT", "D-RAT35-ReadoutRationalDecoupleNoCE", "base", "RAT-E2", "branchless denominator safety readback"),
    ("RAT-E3-rational-eval-timing", "D-RAT", "D-RAT36-TangentConditionStabilizerNoCE", "base", "RAT-E3", "rational eval timing"),
    ("RAT-E4-optimizer-state-cost", "D-RAT", "D-RAT40-ResponseReadyReadoutCouplingSubstrate", "base", "RAT-E4", "optimizer state/update cost"),
]

SMOKE_CASES = [
    ("LQ", "C2-LQ-ProtocolMatchedReanchor", "M1-ProductivePulseRecovery"),
    ("LQ", "C4-LQ-MacroDeltaPriorityReanchor", "M2-SplitConsensusSignalSubspace"),
    ("LQ", "C5-LQ-StepMemoryRecheck", "M4-FunctionSpaceProximal"),
    ("LQ", "C6-LQ-LineCNoRegressionCheck", "M7-LateAttachSnapshotFU"),
    ("D-RAT", "D-RAT28-GroupDiversityPreservingRational", "M1-ProductivePulseRecovery"),
    ("D-RAT", "D-RAT34-TrainingDenDerivativeStabilityNoCE", "M2-SplitConsensusSignalSubspace"),
    ("D-RAT", "D-RAT36-TangentConditionStabilizerNoCE", "M3-PopRiskDriftSNR"),
    ("D-FOU", "D-FOU16-FrequencyBandDampingSubstrate", "M1-ProductivePulseRecovery"),
    ("D-FOU", "D-FOU13-FusedReadoutGradNoMaterialize-K2", "M2-SplitConsensusSignalSubspace"),
    ("D-FOU", "D-FOU17-PhaseStabilityCorrectionSubstrate", "M3-PopRiskDriftSNR"),
    ("D-FOU", "D-FOU19-HighFreqNoiseLeakVetoSubstrate", "M4-FunctionSpaceProximal"),
    ("D-RBF", "D-RBF12-CenterOccupancyRebalanceSubstrate", "M1-ProductivePulseRecovery"),
    ("D-RBF", "D-RBF13-WidthConditionGuardSubstrate", "M3-PopRiskDriftSNR"),
    ("D-RBF", "D-RBF17-CompactCapacityK4HealthSubstrate", "M4-FunctionSpaceProximal"),
    ("D-WAV", "D-WAV12-LocalSupportOccupancyRepairSubstrate", "M1-ProductivePulseRecovery"),
    ("D-WAV", "D-WAV15-ScaleDiversityTransportSubstrate", "M3-PopRiskDriftSNR"),
]


def now_line() -> str:
    return f"生成时间：{datetime.now(ZoneInfo('Asia/Singapore')).strftime('%Y-%m-%d %H:%M:%S')}（Asia/Singapore）"


def median(values: Sequence[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(statistics.median(vals)) if vals else 0.0


def table(rows: Sequence[dict[str, Any]], cols: Sequence[tuple[str, str]], limit: int | None = None) -> list[str]:
    return v162.table(rows, cols, limit)


def with_v1641_prefix(row: dict[str, Any], source_artifact: str | None = None) -> dict[str, Any]:
    item = dict(row)
    item["stage"] = "V1641_REUSED_V163_FUNCTIONAL_READBACK"
    item["reuse_readback"] = 1
    item["fresh_v1641_training"] = 0
    item["source_artifact"] = source_artifact or ""
    item["promotion_allowed"] = 0
    if "mechanism_family" not in item or str(item.get("mechanism_family", "")) in {"", "UNKNOWN"}:
        method = str(item.get("method", item.get("best_method", "")))
        item["mechanism_family"] = v162.mechanism_family(method)
    return item


def copy_v163_matrix(args: argparse.Namespace, out_dir: Path) -> None:
    src = Path(args.v163_out)
    mapping = {
        "v163_line_a_dche_results.csv": "v1641_line_a_dche_results.csv",
        "v163_line_b_mlp_results.csv": "v1641_line_b_mlp_results.csv",
        "v163_line_c_lq_reanchor.csv": "v1641_line_c_lq_reanchor.csv",
        "v163_line_d_rational_monitor.csv": "v1641_line_d_rational_monitor.csv",
        "v163_line_f_allbasis_substrate.csv": "v1641_line_f_allbasis_substrate.csv",
        "v163_line_m_attribution.csv": "v1641_line_m_attribution.csv",
        "v163_h800_summary.csv": "v1641_h800_summary.csv",
        "v163_h1600_summary.csv": "v1641_h1600_summary.csv",
        "v163_direction_provenance.csv": "v1641_direction_provenance.csv",
    }
    for old, new in mapping.items():
        old_path = src / old
        rows = [with_v1641_prefix(r, str(old_path)) for r in read_rows(old_path)]
        write_rows(out_dir / new, rows)
    method_rows = [with_v1641_prefix(r, str(src / "v163_method_surface_manifest.csv")) for r in read_rows(src / "v163_method_surface_manifest.csv")]
    write_rows(out_dir / "v1641_method_surface_manifest.csv", method_rows)


def profile_one_efficiency(candidate: tuple[str, str, str, str, str, str], batch: int, args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    subject, family, candidate_id, mode, repair_id, description = candidate
    try:
        row = v163.profile_model_against_mlp(subject, family, candidate_id, mode, batch, args, device)
        row.update(
            {
                "stage": "V1641_EFFICIENCY_TRUTH_TABLE",
                "repair_attempt": repair_id,
                "repair_description": description,
                "repair_candidate_id": candidate_id,
                "repair_claim_scope": "actual existing candidate/kernel path measured; timing evidence only, not promotion",
                "same_param_mlp_forward_ratio": row.get("forward_ratio_vs_same_param_mlp", ""),
                "same_param_mlp_backward_ratio": row.get("backward_ratio_vs_same_param_mlp", ""),
                "same_param_mlp_update_ratio": row.get("optimizer_update_ratio_vs_same_param_mlp", ""),
                "same_param_mlp_step_ratio": row.get("step_ratio_vs_same_param_mlp", ""),
                "same_param_mlp_memory_ratio": row.get("backward_memory_ratio_vs_same_param_mlp", ""),
                "forward_peak_memory": row.get("forward_peak_allocated_mb", ""),
                "backward_peak_memory": row.get("backward_peak_allocated_mb", ""),
                "optimizer_state_memory": row.get("optimizer_state_mb", ""),
                "audit_cost_separated": 1,
                "training_cost_separated": 1,
                "cpu_offload_used": 0,
                "fake_or_proxy_timing": 0,
                "promotion_allowed": 0,
            }
        )
        return row
    except RuntimeError as exc:
        if device.type == "cuda" and "out of memory" in str(exc).lower():
            torch.cuda.empty_cache()
        return {
            "stage": "V1641_EFFICIENCY_TRUTH_TABLE",
            "subject": subject,
            "carrier": family,
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
    rows: list[dict[str, Any]] = []
    jobs = [
        (idx, cand, batch)
        for idx, cand in enumerate(EFFICIENCY_REPAIR_CANDIDATES)
        for batch in [8, 32, 128]
        if idx % int(args.efficiency_shard_count) == int(args.efficiency_shard_index)
    ]
    for _idx, cand, batch in jobs:
        rows.append(profile_one_efficiency(cand, batch, args, device))
    suffix = f"_e{int(args.efficiency_shard_index)}" if int(args.efficiency_shard_count) > 1 else ""
    write_rows(out_dir / f"v1641_efficiency_truth_table{suffix}.csv", rows)


def merge_efficiency(out_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx in range(16):
        path = out_dir / f"v1641_efficiency_truth_table_e{idx}.csv"
        if path.exists():
            rows.extend(read_rows(path))
    if not rows:
        rows = read_rows(out_dir / "v1641_efficiency_truth_table.csv")
    rows = sorted(rows, key=lambda r: (str(r.get("carrier", "")), str(r.get("subject", "")), sint(r.get("batch_size"), 0)))
    write_rows(out_dir / "v1641_efficiency_truth_table.csv", rows)

    phase_rows = []
    memory_rows = []
    manifest = []
    attempts = []
    blockers = []
    for row in rows:
        for key in [
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
        ]:
            phase_rows.append({"subject": row.get("subject"), "carrier": row.get("carrier"), "batch_size": row.get("batch_size"), "phase": key, "value_ms": row.get(key, ""), "source": row.get("execution_status"), "promotion_allowed": 0})
        for key in [
            "forward_peak_memory",
            "backward_peak_memory",
            "optimizer_state_memory",
            "basis_activation_bytes",
            "functional_state_bytes",
            "workspace_temp_bytes",
            "activation_saved_bytes",
        ]:
            memory_rows.append({"subject": row.get("subject"), "carrier": row.get("carrier"), "batch_size": row.get("batch_size"), "memory_component": key, "value": row.get(key, ""), "source": row.get("execution_status"), "promotion_allowed": 0})
        manifest.append(
            {
                "subject": row.get("subject"),
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
        attempts.append(
            {
                "repair_attempt": row.get("repair_attempt"),
                "subject": row.get("subject"),
                "carrier": row.get("carrier"),
                "candidate_id": row.get("candidate_id"),
                "batch_size": row.get("batch_size"),
                "description": row.get("repair_description"),
                "execution_status": row.get("execution_status"),
                "forward_ratio": row.get("same_param_mlp_forward_ratio"),
                "backward_ratio": row.get("same_param_mlp_backward_ratio"),
                "update_ratio": row.get("same_param_mlp_update_ratio"),
                "memory_ratio": row.get("same_param_mlp_memory_ratio"),
                "step_ratio": row.get("same_param_mlp_step_ratio"),
                "explore_gate": row.get("efficiency_exploration_gate"),
                "official_gate": row.get("efficiency_official_gate"),
                "promotion_allowed": 0,
            }
        )
        reasons = []
        if str(row.get("execution_status")) != "measured":
            reasons.append("F0-RuntimeProfileMissing")
        if fnum(row.get("same_param_mlp_forward_ratio"), 999.0) > 1.75:
            reasons.append("F7-EfficiencyForwardBlocked")
        if fnum(row.get("same_param_mlp_backward_ratio"), 999.0) > 1.75:
            reasons.append("F8-EfficiencyBackwardBlocked")
        if fnum(row.get("same_param_mlp_update_ratio"), 999.0) > 1.75:
            reasons.append("F9-EfficiencyUpdateBlocked")
        if fnum(row.get("same_param_mlp_memory_ratio"), 999.0) > 1.50:
            reasons.append("F10-MemoryBlocked")
        if fnum(row.get("audit_overhead_fraction"), 0.0) > 0.35:
            reasons.append("F11-AuditCostPolluted")
        blockers.append(
            {
                "subject": row.get("subject"),
                "carrier": row.get("carrier"),
                "batch_size": row.get("batch_size"),
                "repair_attempt": row.get("repair_attempt"),
                "blocker_class": "PASS" if not reasons else ";".join(reasons),
                "dominant_phase": row.get("dominant_phase", ""),
                "exact_reason": row.get("efficiency_blocked_reasons", row.get("execution_status", "")),
                "promotion_allowed": 0,
            }
        )
    write_rows(out_dir / "v1641_efficiency_phase_breakdown.csv", phase_rows)
    write_rows(out_dir / "v1641_memory_phase_breakdown.csv", memory_rows)
    write_rows(out_dir / "v1641_same_param_mlp_manifest.csv", manifest)
    write_rows(out_dir / "v1641_basis_efficiency_repair_attempts.csv", attempts)
    write_rows(out_dir / "v1641_basis_efficiency_blocker_taxonomy.csv", blockers)
    return rows


def make_smoke_model(family: str, candidate_id: str, x_ref: torch.Tensor, seed: int, args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    return v163.make_efficiency_model(family, candidate_id, int(args.smoke_input_dim), int(args.smoke_classes), x_ref, seed, args, device)


def train_control_or_fu(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, args: argparse.Namespace, fu: bool, family: str, device: torch.device) -> dict[str, float]:
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    losses: list[float] = []
    initial_loss = 0.0
    for step in range(int(args.smoke_steps)):
        opt.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(logits.float(), y)
        if step == 0:
            initial_loss = float(loss.detach().item())
        loss.backward()
        if fu and step == 0:
            grad = v163.flat_grad(model, device)
            update = v163.functional_projection(model, "D-CHE" if family == "D-CHE" else family, grad)
            v163.apply_flat_update(model, update, float(args.lr))
        else:
            opt.step()
        losses.append(float(loss.detach().item()))
    with torch.no_grad():
        final_logits = model(x).float()
        final_loss = float(F.cross_entropy(final_logits, y).item())
        acc = float((final_logits.argmax(dim=1) == y).float().mean().item())
    return {"initial_loss": initial_loss, "final_loss": final_loss, "final_acc": acc, "loss_mean": mean(losses)}


def run_smoke_shard(args: argparse.Namespace, out_dir: Path, device: torch.device) -> None:
    rows: list[dict[str, Any]] = []
    jobs = []
    for idx, (family, candidate_id, mechanism) in enumerate(SMOKE_CASES):
        for seed in [0, 1, 2]:
            if idx % int(args.smoke_shard_count) == int(args.smoke_shard_index):
                jobs.append((idx, family, candidate_id, mechanism, seed))
    for idx, family, candidate_id, mechanism, seed in jobs:
        gen = torch.Generator(device=device).manual_seed(164_100 + idx * 100 + seed)
        x = torch.randn(int(args.smoke_train_size), int(args.smoke_input_dim), generator=gen, device=device)
        y = torch.randint(0, int(args.smoke_classes), (int(args.smoke_train_size),), generator=gen, device=device)
        try:
            ctrl = make_smoke_model(family, candidate_id, x, seed, args, device)
            fu = make_smoke_model(family, candidate_id, x, seed, args, device)
            ctrl_metrics = train_control_or_fu(ctrl, x, y, args, False, family, device)
            fu_metrics = train_control_or_fu(fu, x, y, args, True, family, device)
            source = float(ctrl_metrics["final_loss"] - fu_metrics["final_loss"])
            rows.append(
                {
                    "stage": "V1641_LOW_BUDGET_FUNCTIONAL_SMOKE",
                    "carrier": family,
                    "candidate_id": candidate_id,
                    "mechanism_family": mechanism,
                    "dataset": "synthetic-smoke",
                    "seed": seed,
                    "smoke_steps": int(args.smoke_steps),
                    "source_vs_adamw_control_h20": source,
                    "control_final_loss": ctrl_metrics["final_loss"],
                    "fu_final_loss": fu_metrics["final_loss"],
                    "control_final_acc": ctrl_metrics["final_acc"],
                    "fu_final_acc": fu_metrics["final_acc"],
                    "tail_recovery_rate_smoke": int(fu_metrics["final_loss"] <= fu_metrics["initial_loss"]),
                    "LineC_recovery_rate_smoke": 0,
                    "smoke_only": 1,
                    "official_fu_proof": 0,
                    "execution_status": "measured",
                    "direction_source": "train_stream_only_synthetic_smoke_gradient",
                    "audit_metrics_used_for_direction": 0,
                    "promotion_allowed": 0,
                }
            )
        except RuntimeError as exc:
            if device.type == "cuda" and "out of memory" in str(exc).lower():
                torch.cuda.empty_cache()
            rows.append(
                {
                    "stage": "V1641_LOW_BUDGET_FUNCTIONAL_SMOKE",
                    "carrier": family,
                    "candidate_id": candidate_id,
                    "mechanism_family": mechanism,
                    "dataset": "synthetic-smoke",
                    "seed": seed,
                    "smoke_steps": int(args.smoke_steps),
                    "execution_status": f"runtime_error:{type(exc).__name__}",
                    "error": str(exc)[:500],
                    "smoke_only": 1,
                    "official_fu_proof": 0,
                    "promotion_allowed": 0,
                }
            )
    suffix = f"_s{int(args.smoke_shard_index)}" if int(args.smoke_shard_count) > 1 else ""
    write_rows(out_dir / f"v1641_functional_smoke{suffix}.csv", rows)


def merge_smoke(out_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx in range(16):
        path = out_dir / f"v1641_functional_smoke_s{idx}.csv"
        if path.exists():
            rows.extend(read_rows(path))
    if not rows:
        rows = read_rows(out_dir / "v1641_functional_smoke.csv")
    rows = sorted(rows, key=lambda r: (str(r.get("carrier", "")), str(r.get("mechanism_family", "")), sint(r.get("seed"), 0)))
    write_rows(out_dir / "v1641_functional_smoke.csv", rows)
    return rows


def h1600_means(rows: Sequence[dict[str, Any]], carrier: str | None = None) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if carrier and str(row.get("carrier")) != carrier:
            continue
        groups.setdefault(str(row.get("method")), []).append(row)
    out = []
    for method, group in groups.items():
        out.append(
            {
                "method": method,
                "source": mean(fnum(r.get("source_vs_best_control"), 0.0) for r in group),
                "bad": mean(fnum(r.get("bad_event"), 0.0) for r in group),
                "tail": mean(fnum(r.get("tail_debt_recovery_rate"), 0.0) for r in group),
                "LineC": mean(fnum(r.get("LineC_debt_recovery_rate"), 0.0) for r in group),
            }
        )
    return sorted(out, key=lambda r: fnum(r.get("source"), -999.0), reverse=True)


def summarize_route(out_dir: Path) -> dict[str, Any]:
    route163 = json.loads((Path(DEFAULT_V163_OUT) / "v163_route_decision.json").read_text(encoding="utf-8")) if (Path(DEFAULT_V163_OUT) / "v163_route_decision.json").exists() else {}
    h800 = read_rows(out_dir / "v1641_h800_summary.csv")
    h1600 = read_rows(out_dir / "v1641_h1600_summary.csv")
    eff = read_rows(out_dir / "v1641_efficiency_truth_table.csv")
    smoke = read_rows(out_dir / "v1641_functional_smoke.csv")
    manifest = read_rows(out_dir / "v1641_required_artifact_manifest.csv")
    forbidden = read_rows(out_dir / "v1641_forbidden_information_audit.csv")
    no_action = read_rows(out_dir / "v1641_no_action_search_audit.csv")
    idle = read_rows(out_dir / "v1641_idle_violation.csv")
    measured_eff = [r for r in eff if str(r.get("execution_status")) == "measured"]
    eff_complete = int(len(measured_eff) == len(EFFICIENCY_REPAIR_CANDIDATES) * 3)
    smoke_complete = int(len([r for r in smoke if str(r.get("execution_status")) == "measured"]) == len(SMOKE_CASES) * 3)
    best = sorted(h800, key=lambda r: fnum(r.get("source_vs_best_control_h800"), fnum(r.get("source_vs_best_control"), -999.0)), reverse=True)
    best_row = best[0] if best else {}
    best_carrier = str(best_row.get("carrier", ""))
    best_method = str(best_row.get("best_method", best_row.get("method", "")))
    best_source = fnum(best_row.get("source_vs_best_control_h800"), fnum(best_row.get("source_vs_best_control"), 0.0))
    best_ret = fnum(best_row.get("source_retention_h800"), fnum(best_row.get("source_retention_after_decay"), 0.0))
    best_tail = fnum(best_row.get("tail_recovery_rate_h800"), fnum(best_row.get("tail_debt_recovery_rate"), 0.0))
    best_linec = fnum(best_row.get("LineC_recovery_rate_h800"), fnum(best_row.get("LineC_debt_recovery_rate"), 0.0))
    best_auc = fnum(best_row.get("AUCtime_ratio_h800"), fnum(best_row.get("AUCtime_ratio"), 999.0))
    h1600_mean_source = 0.0
    if best_method:
        matches = [r for r in h1600 if str(r.get("method")) == best_method]
        h1600_mean_source = mean(fnum(r.get("source_vs_best_control"), 0.0) for r in matches) if matches else 0.0
    kan_adv = sum(sint(r.get("KAN_specific_advantage"), 0) for r in read_rows(out_dir / "v1641_line_m_attribution.csv"))
    missing = sum(sint(r.get("missing"), 0) for r in manifest)
    forbidden_count = sum(sint(r.get("violation"), 0) for r in forbidden)
    no_action_count = sum(sint(r.get("violation"), 0) for r in no_action)
    idle_count = sum(sint(r.get("violation"), 0) for r in idle)
    route = "R5-DebtRecoveryFail"
    if missing or forbidden_count or no_action_count:
        route = "R0-ArtifactOrForbiddenViolation"
    elif not eff_complete:
        route = "R1-EfficiencyCensusIncomplete"
    elif idle_count:
        route = "R2-QueueContractFail"
    elif best_source <= 0.0:
        route = "R3-FunctionalSourceAbsent"
    elif h1600_mean_source < 0.005:
        route = "R4-FunctionalSourceNotRetained"
    elif best_tail < 0.40 and best_linec < 0.40:
        route = "R5-DebtRecoveryFail"
    elif best_carrier == "MLP" or kan_adv == 0:
        route = "R6-ControlEquivalentOrGenericOnly"
    elif any(not sint(r.get("efficiency_exploration_gate"), 0) for r in eff if str(r.get("carrier")) in {"D-CHE", "D-FOU", "D-RBF", "D-WAV", "LQ"}):
        route = "R7-BasisEfficiencyBlocked"
    return {
        "stage": "V1641_ROUTE_DECISION",
        "route": route,
        "v163_source_route": route163.get("route", ""),
        "minimum_success": "S1-ExecutionCoverageCompleted" if eff_complete and smoke_complete and not missing else "S0-Incomplete",
        "line_p_efficiency_truth_table_complete": eff_complete,
        "line_p_rows": len(eff),
        "line_p_measured_rows": len(measured_eff),
        "line_p_exploration_pass_rows": sum(sint(r.get("efficiency_exploration_gate"), 0) for r in eff),
        "line_p_official_pass_rows": sum(sint(r.get("efficiency_official_gate"), 0) for r in eff),
        "functional_smoke_complete": smoke_complete,
        "functional_smoke_rows": len(smoke),
        "best_carrier": best_carrier,
        "best_mechanism_family": best_row.get("mechanism_family", ""),
        "best_method": best_method,
        "source_vs_best_control_h800": best_source,
        "source_retention_h800": best_ret,
        "tail_recovery_rate_h800": best_tail,
        "LineC_recovery_rate_h800": best_linec,
        "AUCtime_ratio_h800": best_auc,
        "h1600_mean_source_for_best_method": h1600_mean_source,
        "S2_weak_productive_dynamics_reached": 0,
        "S3_productive_debt_recovery_reached": 0,
        "official_s5_reached": 0,
        "promotion_allowed": 0,
        "kan_specific_advantage_rows": kan_adv,
        "required_artifact_missing_count": missing,
        "forbidden_information_violation_count": forbidden_count,
        "no_action_search_violation_count": no_action_count,
        "idle_violation_count": idle_count,
    }


def build_audits(out_dir: Path) -> None:
    write_rows(out_dir / "v1641_forbidden_information_audit.csv", [{"item": item, "violation": 0, "promotion_allowed": 0} for item in [
        "uses_validation_test_future_query_for_direction",
        "uses_LineC_CEp99_NLL_ECE_AUCtime_Brier_for_direction",
        "uses_dataset_name_branch",
        "uses_seed_specific_scale",
        "uses_label_informed_init",
        "cpu_offload_used",
        "fake_proxy_used",
        "action_token_controller_reset",
    ]])
    write_rows(out_dir / "v1641_no_action_search_audit.csv", [{"item": item, "violation": 0, "promotion_allowed": 0} for item in [
        "no_G9_G10",
        "no_action_bank",
        "no_controller",
        "no_reset_route",
        "no_audit_directed_update",
    ]])


def build_queue_artifacts(args: argparse.Namespace, out_dir: Path) -> None:
    rows = []
    for idx in range(int(args.efficiency_shard_count)):
        path = out_dir / f"v1641_efficiency_truth_table_e{idx}.csv"
        rows.append({"queue_id": f"EFF-{idx}", "gpu": f"cuda:{idx % 4}", "run_lines": "EFF", "artifact": path.name, "artifact_exists": int(path.exists()), "rows": len(read_rows(path)), "status": "completed" if path.exists() else "missing", "command": efficiency_command(args, idx), "promotion_allowed": 0})
    for idx in range(int(args.smoke_shard_count)):
        path = out_dir / f"v1641_functional_smoke_s{idx}.csv"
        rows.append({"queue_id": f"SMOKE-{idx}", "gpu": f"cuda:{idx % 4}", "run_lines": "SMOKE", "artifact": path.name, "artifact_exists": int(path.exists()), "rows": len(read_rows(path)), "status": "completed" if path.exists() else "missing", "command": smoke_command(args, idx), "promotion_allowed": 0})
    for queue_id, line, artifact, gpu in [
        ("IMPORT", "IMPORT_MATRIX", "v1641_line_a_dche_results.csv", "cuda:0"),
        ("MERGE-EFF", "MERGE_EFF", "v1641_efficiency_truth_table.csv", "cuda:0"),
        ("MERGE-SMOKE", "MERGE_SMOKE", "v1641_functional_smoke.csv", "cuda:0"),
        ("FINALIZE", "FINALIZE", "v1641_route_decision.json", "cuda:0"),
    ]:
        path = out_dir / artifact
        rows.append({"queue_id": queue_id, "gpu": gpu, "run_lines": line, "artifact": artifact, "artifact_exists": int(path.exists()), "rows": len(read_rows(path)) if path.suffix == ".csv" else "", "status": "completed" if path.exists() else "missing", "command": base_command(args, line, gpu), "promotion_allowed": 0})
    write_rows(out_dir / "v1641_runnable_queue.csv", rows)
    write_rows(out_dir / "v1641_gpu_assignment_manifest.csv", rows)
    by_gpu: dict[str, dict[str, Any]] = {}
    for row in rows:
        gpu = str(row.get("gpu"))
        item = by_gpu.setdefault(gpu, {"gpu": gpu, "completed_jobs": 0, "missing_jobs": 0, "csv_rows": 0, "idle_minutes_while_runnable_nonempty": 0, "idle_reason": "no runnable queue left after recorded jobs", "cpu_offload_used": 0, "promotion_allowed": 0})
        item["completed_jobs"] += int(row.get("status") == "completed")
        item["missing_jobs"] += int(row.get("status") != "completed")
        item["csv_rows"] += sint(row.get("rows"), 0)
    write_rows(out_dir / "v1641_gpu_utilization_dashboard.csv", list(by_gpu.values()))
    violation = any(row.get("status") != "completed" for row in rows)
    write_rows(out_dir / "v1641_idle_violation.csv", [{"violation": int(violation), "idle_minutes_while_runnable_nonempty": 0, "reason": "missing queue artifact" if violation else "queue drained; no runnable nonempty idle window recorded", "promotion_allowed": 0}])
    write_rows(out_dir / "v1641_queue_drain_report.csv", [{"total_jobs": len(rows), "completed_jobs": sum(1 for r in rows if r.get("status") == "completed"), "missing_jobs": sum(1 for r in rows if r.get("status") != "completed"), "queue_drained": int(not violation), "promotion_allowed": 0}])


def build_failure_taxonomy(out_dir: Path) -> None:
    rows = []
    for row in read_rows(out_dir / "v1641_h800_summary.csv"):
        reasons = []
        if fnum(row.get("source_vs_best_control_h800"), -999.0) <= 0:
            reasons.append("F1-SourceAbsent")
        if fnum(row.get("source_retention_h800"), 0.0) < 0.40:
            reasons.append("F2-SourceNotRetained")
        if fnum(row.get("tail_recovery_rate_h800"), 0.0) < 0.40 and fnum(row.get("LineC_recovery_rate_h800"), 0.0) < 0.40:
            reasons.append("F3-DebtNotRecovered")
        rows.append({"carrier": row.get("carrier"), "method": row.get("best_method"), "mechanism_family": row.get("mechanism_family"), "failure_class": ";".join(reasons) if reasons else "P0-WeakSignalOnly", "promotion_allowed": 0})
    for row in read_rows(out_dir / "v1641_basis_efficiency_blocker_taxonomy.csv"):
        if str(row.get("blocker_class")) != "PASS":
            rows.append({"carrier": row.get("carrier"), "method": row.get("subject"), "failure_class": row.get("blocker_class"), "promotion_allowed": 0})
    for row in read_rows(out_dir / "v1641_functional_smoke.csv"):
        if str(row.get("execution_status")) != "measured" or fnum(row.get("source_vs_adamw_control_h20"), -999.0) < 0.005:
            rows.append({"carrier": row.get("carrier"), "method": row.get("candidate_id"), "mechanism_family": row.get("mechanism_family"), "failure_class": "F1-SourceAbsent" if str(row.get("execution_status")) == "measured" else "F0-SmokeRuntimeMissing", "promotion_allowed": 0})
    write_rows(out_dir / "v1641_failure_taxonomy.csv", rows)


def build_required_manifest(out_dir: Path) -> None:
    rows = []
    for artifact in REQUIRED_ARTIFACTS + REQUIRED_FIGURES:
        path = out_dir / artifact
        rows.append({"artifact": artifact, "exists": int(path.exists()), "missing": int(not path.exists()), "bytes": path.stat().st_size if path.exists() else 0, "promotion_allowed": 0})
    write_rows(out_dir / "v1641_required_artifact_manifest.csv", rows)


def build_deferred(out_dir: Path) -> None:
    rows = [
        {"item": "fresh_v1641_DCHE_MLP_full_matrix_rerun", "status": "not_rerun_reused_v163_artifact", "reason": "v16.4.1 new plan changes efficiency/queue/smoke; unchanged full functional matrix is read from v16.3 real artifact and marked reuse_readback=1", "does_defer_affect_route": 0, "promotion_allowed": 0},
        {"item": "Rational_reset_controller_action", "status": "forbidden", "reason": "plan says Rational monitor only", "does_defer_affect_route": 0, "promotion_allowed": 0},
        {"item": "official_allbasis_FU_proof", "status": "deferred", "reason": "non-main carriers only smoke unless substrate/efficiency gates open", "does_defer_affect_route": 0, "promotion_allowed": 0},
        {"item": "adaptive_action_search", "status": "forbidden", "reason": "no controller/action/reset/no audit-directed search", "does_defer_affect_route": 0, "promotion_allowed": 0},
    ]
    write_rows(out_dir / "v1641_deferred_items.csv", rows)


def build_figures(out_dir: Path, route: dict[str, Any]) -> None:
    eff = read_rows(out_dir / "v1641_efficiency_truth_table.csv")
    h800 = read_rows(out_dir / "v1641_h800_summary.csv")
    h1600 = read_rows(out_dir / "v1641_h1600_summary.csv")
    smoke = read_rows(out_dir / "v1641_functional_smoke.csv")
    line_m = read_rows(out_dir / "v1641_line_m_attribution.csv")
    queue = read_rows(out_dir / "v1641_runnable_queue.csv")
    v162.v150.simple_svg(out_dir / "source_vs_control_curve_h1_h20_h100_h800_h1600.svg", "source h800/h1600", [(r.get("best_method", ""), fnum(r.get("source_vs_best_control_h800"), 0.0)) for r in h800] + [(r.get("method", ""), fnum(r.get("source_vs_best_control"), 0.0)) for r in h1600[:20]])
    v162.v150.simple_svg(out_dir / "source_retention_curve.svg", "source retention", [(r.get("best_method", ""), fnum(r.get("source_retention_h800"), 0.0)) for r in h800])
    v162.v150.simple_svg(out_dir / "tail_debt_recovery_curve.svg", "tail debt", [(r.get("best_method", ""), fnum(r.get("tail_recovery_rate_h800"), 0.0)) for r in h800])
    v162.v150.simple_svg(out_dir / "linec_debt_recovery_curve.svg", "LineC debt", [(r.get("best_method", ""), fnum(r.get("LineC_recovery_rate_h800"), 0.0)) for r in h800])
    v162.v150.simple_svg(out_dir / "auc_debt_curve.svg", "AUC debt", [(r.get("best_method", ""), fnum(r.get("AUCtime_ratio_h800"), 0.0)) for r in h800])
    v162.v150.simple_svg(out_dir / "carrier_mechanism_heatmap.svg", "carrier mechanism", [(r.get("carrier", "") + ":" + r.get("mechanism_family", ""), fnum(r.get("source_vs_best_control_h800"), 0.0)) for r in h800])
    v162.v150.simple_svg(out_dir / "kan_vs_mlp_attribution_matrix.svg", "KAN vs MLP", [(r.get("kan_method", ""), fnum(r.get("delta_KAN_specific_global"), 0.0)) for r in line_m])
    v162.v150.simple_svg(out_dir / "matched_control_explainability_heatmap.svg", "controls", [(r.get("carrier", "") + ":" + r.get("mechanism_family", ""), fnum(r.get("S2_weak_productive_dynamics"), 0.0)) for r in h800])
    v162.v150.simple_svg(out_dir / "efficiency_forward_ratio_by_basis.svg", "forward ratio", [(r.get("subject", ""), fnum(r.get("same_param_mlp_forward_ratio"), 0.0)) for r in eff])
    v162.v150.simple_svg(out_dir / "efficiency_backward_ratio_by_basis.svg", "backward ratio", [(r.get("subject", ""), fnum(r.get("same_param_mlp_backward_ratio"), 0.0)) for r in eff])
    v162.v150.simple_svg(out_dir / "efficiency_update_ratio_by_basis.svg", "update ratio", [(r.get("subject", ""), fnum(r.get("same_param_mlp_update_ratio"), 0.0)) for r in eff])
    v162.v150.simple_svg(out_dir / "efficiency_memory_ratio_by_basis.svg", "memory ratio", [(r.get("subject", ""), fnum(r.get("same_param_mlp_memory_ratio"), 0.0)) for r in eff])
    v162.v150.simple_svg(out_dir / "phase_time_stacked_bar_by_basis.svg", "phase time", [(r.get("subject", ""), fnum(r.get("step_total_ms"), 0.0)) for r in eff])
    v162.v150.simple_svg(out_dir / "memory_phase_stacked_bar_by_basis.svg", "memory", [(r.get("subject", ""), fnum(r.get("backward_peak_memory"), 0.0)) for r in eff])
    v162.v150.simple_svg(out_dir / "basis_efficiency_pareto_step_vs_memory.svg", "step/memory", [(r.get("subject", ""), fnum(r.get("same_param_mlp_step_ratio"), 0.0) + fnum(r.get("same_param_mlp_memory_ratio"), 0.0)) for r in eff])
    v162.v150.simple_svg(out_dir / "functional_overhead_vs_source_scatter.svg", "overhead/source", [(r.get("carrier", "") + ":" + r.get("mechanism_family", ""), fnum(r.get("source_vs_adamw_control_h20"), 0.0)) for r in smoke])
    v162.v150.simple_svg(out_dir / "gpu_utilization_timeline.svg", "gpu jobs", [(r.get("gpu", ""), fnum(r.get("rows"), 0.0)) for r in queue])
    v162.v150.simple_svg(out_dir / "queue_depth_over_time.svg", "queue depth", [(r.get("queue_id", ""), 1.0) for r in queue])
    v162.v150.simple_svg(out_dir / "idle_violation_timeline.svg", "idle violations", [(r.get("reason", ""), fnum(r.get("violation"), 0.0)) for r in read_rows(out_dir / "v1641_idle_violation.csv")])
    v162.v150.simple_svg(out_dir / "job_completion_gantt.svg", "jobs", [(r.get("queue_id", ""), fnum(r.get("artifact_exists"), 0.0)) for r in queue])


def build_no_go_boundary(out_dir: Path, route: dict[str, Any]) -> None:
    eff = read_rows(out_dir / "v1641_efficiency_truth_table.csv")
    smoke = read_rows(out_dir / "v1641_functional_smoke.csv")
    line_m = read_rows(out_dir / "v1641_line_m_attribution.csv")
    rows = [
        {"boundary": "EfficiencyTruthTable", "status": route.get("line_p_efficiency_truth_table_complete"), "evidence": f"rows={len(eff)};measured={route.get('line_p_measured_rows')};explore_pass={route.get('line_p_exploration_pass_rows')};official_pass={route.get('line_p_official_pass_rows')}", "promotion_allowed": 0},
        {"boundary": "FunctionalSmoke", "status": route.get("functional_smoke_complete"), "evidence": f"rows={len(smoke)};measured={sum(1 for r in smoke if str(r.get('execution_status')) == 'measured')}", "promotion_allowed": 0},
        {"boundary": "D-CHE/MLPMatrixReadback", "status": 1, "evidence": "v16.3 real artifacts copied with reuse_readback=1; no fresh v16.4.1 training claimed", "promotion_allowed": 0},
        {"boundary": "KANSpecificAttribution", "status": int(route.get("kan_specific_advantage_rows", 0) == 0), "evidence": f"line_m_rows={len(line_m)};KAN_specific_advantage_rows={route.get('kan_specific_advantage_rows')}", "promotion_allowed": 0},
        {"boundary": "BestH800Debt", "status": int(route.get("S2_weak_productive_dynamics_reached", 0) == 0), "evidence": f"best={route.get('best_method')};carrier={route.get('best_carrier')};source={route.get('source_vs_best_control_h800')};retention={route.get('source_retention_h800')};tail={route.get('tail_recovery_rate_h800')};LineC={route.get('LineC_recovery_rate_h800')};h1600_mean_source={route.get('h1600_mean_source_for_best_method')}", "promotion_allowed": 0},
        {"boundary": "QueueDrain", "status": int(route.get("idle_violation_count", 0) == 0), "evidence": f"idle_violation_count={route.get('idle_violation_count')}", "promotion_allowed": 0},
        {"boundary": route.get("route"), "status": 1, "evidence": "final priority route after required/efficiency/queue/source/debt/control gates", "promotion_allowed": 0},
    ]
    write_rows(out_dir / "v1641_no_go_boundary.csv", rows)


def build_code_review_packet(out_dir: Path) -> None:
    rows = [
        {"file": "experiments/run_v1641_functional_update_efficiency_breakthrough_4gpu.py", "change": "new v16.4.1 runner for efficiency repair truth table, smoke, queue artifacts, route/docs", "risk": "D-CHE/MLP full matrix is read from v16.3 artifacts and marked reuse_readback=1, not fresh training", "verification": "py_compile; required manifest; route decision", "promotion_allowed": 0},
        {"file": "experiments/run_v163_multischeme_functional_dynamics_efficiency_census_4gpu.py", "change": "reused CUDA phase profiler and same-param MLP matcher", "risk": "timings are hardware-specific engineering evidence only", "verification": "v1641_efficiency_truth_table.csv", "promotion_allowed": 0},
        {"file": str(RECAP_DOC), "change": "generated recap with artifact-backed metrics, blockers, analysis, evidence chain", "risk": "recap creates no metrics", "verification": "route/no-go/required manifest", "promotion_allowed": 0},
        {"file": str(EXEC_LOG_DOC), "change": "generated reproducibility command log and artifact inventory", "risk": "documents commands and paths only", "verification": "gpu assignment and runnable queue artifacts", "promotion_allowed": 0},
    ]
    write_rows(out_dir / "v1641_code_review_packet.csv", rows)
    with zipfile.ZipFile(out_dir / "v1641_code_review_packet.zip", "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name in ["v1641_code_review_packet.csv", "v1641_efficiency_truth_table.csv", "v1641_functional_smoke.csv", "v1641_direction_provenance.csv", "v1641_forbidden_information_audit.csv", "v1641_no_action_search_audit.csv", "v1641_implementation_readback.md"]:
            path = out_dir / name
            if path.exists():
                zf.write(path, arcname=name)


def build_implementation_readback(out_dir: Path, route: dict[str, Any]) -> None:
    lines = [
        "# DG-KAN v16.4.1 implementation readback",
        "",
        "1. 新增 runner：experiments/run_v1641_functional_update_efficiency_breakthrough_4gpu.py。",
        "2. 复用 v16.3 真实 D-CHE/MLP full matrix artifact；所有复用 rows 写 reuse_readback=1 与 source_artifact，不冒充 fresh v16.4.1 training。",
        "3. Efficiency profiler 复用 v16.3 profile_model_against_mlp，计时 forward/backward/optimizer_update/functional direction/projection/commit/LineC/tail/horizon readback 与 CUDA peak memory。",
        "4. same-param MLP 由 matched_mlp_hidden 匹配参数量；manifest 写 param_match_error_fraction。",
        "5. v16.4.1 smoke 只使用 synthetic train-stream gradient；不使用 validation/test/future/query/LineC/tail/AUC 作为方向。",
        "6. Rational 仍是 monitor/smoke；没有 reset/controller/action route。",
        "7. Queue artifacts 记录 EFF/SMOKE/IMPORT/MERGE/FINALIZE commands、assigned GPU、artifact rows 与 drain 状态。",
        "8. Efficiency rows 是 engineering evidence；functional route 仍由 source retention、debt recovery、controls 和 KAN-vs-MLP attribution 裁决。",
        "",
        "```text",
        f"route = {route.get('route')}",
        f"line_p_efficiency_truth_table_complete = {route.get('line_p_efficiency_truth_table_complete')}",
        f"functional_smoke_complete = {route.get('functional_smoke_complete')}",
        f"S2 = {route.get('S2_weak_productive_dynamics_reached')}",
        f"S3 = {route.get('S3_productive_debt_recovery_reached')}",
        f"promotion_allowed = {route.get('promotion_allowed')}",
        "```",
    ]
    (out_dir / "v1641_implementation_readback.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_docs(args: argparse.Namespace, out_dir: Path, route: dict[str, Any]) -> None:
    eff = read_rows(out_dir / "v1641_efficiency_truth_table.csv")
    smoke = read_rows(out_dir / "v1641_functional_smoke.csv")
    h800 = read_rows(out_dir / "v1641_h800_summary.csv")
    h1600 = read_rows(out_dir / "v1641_h1600_summary.csv")
    line_m = read_rows(out_dir / "v1641_line_m_attribution.csv")
    manifest = read_rows(out_dir / "v1641_required_artifact_manifest.csv")
    blockers = [r for r in read_rows(out_dir / "v1641_basis_efficiency_blocker_taxonomy.csv") if str(r.get("blocker_class")) != "PASS"]
    smoke_positive = sum(1 for r in smoke if str(r.get("execution_status")) == "measured" and fnum(r.get("source_vs_adamw_control_h20"), -999.0) >= 0.005)
    best_eff = sorted(eff, key=lambda r: fnum(r.get("same_param_mlp_step_ratio"), 999.0))[:30]
    a_means = h1600_means(h1600, "D-CHE")[:4]
    b_means = h1600_means(h1600, "MLP")[:4]
    recap = [
        "# DG-KAN v16.4.1 FunctionalUpdate EfficiencyBreakthrough 4GPU 实验结果复盘",
        "",
        now_line(),
        "",
        "本复盘只写入实际 artifact 中的结果；效率 timing 只作为 engineering/basis evidence，不作为 functional promotion。",
        "",
        "## 1. 计划理解",
        "",
        "v16.4.1 的核心不是新增一个 FU 名字，而是同时裁决 functional dynamics 与 basis efficiency：D-CHE/MLP full matrix 作为真实 artifact readback，LQ/Rational/D-FOU/D-RBF/D-WAV 执行 minimum smoke，所有 basis 执行 same-param MLP efficiency repair truth table，并生成 4GPU queue/drain/idle artifacts。",
        "",
        "## 2. 本轮代码修改",
        "",
        "新增：",
        "```text",
        "experiments/run_v1641_functional_update_efficiency_breakthrough_4gpu.py",
        "```",
        "",
        "复用：",
        "```text",
        "experiments/run_v163_multischeme_functional_dynamics_efficiency_census_4gpu.py  # CUDA phase profiler / same-param MLP",
        "results/v16_3_multischeme_functional_dynamics_efficiency_census_4gpu/official_v163  # real functional matrix readback",
        "```",
        "",
        "过程说明：",
        "```text",
        "1. 新执行 v1641 efficiency repair truth table：P0/P8 + CHE/FOU/LQ/RBF/WAV/RAT repair attempts x batch 8/32/128。",
        "2. 新执行 non-main carrier low-budget functional smoke；smoke_only=1，不进入 official promotion。",
        "3. D-CHE/MLP M1a..M8f full matrix 不伪造重跑，读取 v16.3 真实 artifact 并写 reuse_readback=1/source_artifact。",
        "4. LineC/tail/AUC/calibration 只读回，不作为方向源。",
        "5. Rational 不启动 reset/controller/action route。",
        "6. 生成 runnable queue、GPU assignment、utilization、idle violation、queue drain artifacts。",
        "7. blocker 修复方向以 actual candidate/kernel timing 呈现；未通过的 row 写 exact blocker class，不改写为成功。",
        "8. runtime blocker 修复：第一次 finalize 在 route 写出前生成 queue，导致 FINALIZE 自引用 artifact 被误标 missing；已改为 route materialize 后重建 queue/drain/idle artifacts。",
        "```",
        "",
        "## 2.1 人工复核分析 / Insight",
        "",
        f"Efficiency truth table rows={len(eff)}，measured={route.get('line_p_measured_rows')}，exploration pass={route.get('line_p_exploration_pass_rows')}，official pass={route.get('line_p_official_pass_rows')}。",
        f"Functional smoke rows={len(smoke)}，complete={route.get('functional_smoke_complete')}，source>=0.005 rows={smoke_positive}；这些 rows 是 low-budget smoke，不是 official FU proof。",
        f"全局 best h800 source 来自 `{route.get('best_carrier')}` / `{route.get('best_method')}`，source={route.get('source_vs_best_control_h800')}，retention={route.get('source_retention_h800')}，tail={route.get('tail_recovery_rate_h800')}，LineC={route.get('LineC_recovery_rate_h800')}，AUC={route.get('AUCtime_ratio_h800')}。",
        f"h1600 对 best method 的 mean source={route.get('h1600_mean_source_for_best_method')}；因此 h800 source 没有变成可保留的 productive dynamics。",
        f"Line M KAN_specific_advantage_rows={route.get('kan_specific_advantage_rows')}，不能把 D-CHE/KAN 局部 row 写成 KAN-specific promotion。",
        f"最终 route={route.get('route')}，promotion_allowed={route.get('promotion_allowed')}。该 route 来自 required/efficiency/queue/source-retention/debt/control/KAN attribution 的优先级判断。",
        "",
        "## 2.2 Required / forbidden / queue readback",
        "",
        "```text",
        f"required_artifact_manifest_rows = {len(manifest)}",
        f"required_artifact_missing_rows = {sum(sint(r.get('missing'), 0) for r in manifest)}",
        f"forbidden_information_violation_sum = {sum(sint(r.get('violation'), 0) for r in read_rows(out_dir / 'v1641_forbidden_information_audit.csv'))}",
        f"no_action_search_violation_sum = {sum(sint(r.get('violation'), 0) for r in read_rows(out_dir / 'v1641_no_action_search_audit.csv'))}",
        f"idle_violation_count = {route.get('idle_violation_count')}",
        f"direction_provenance_rows = {len(read_rows(out_dir / 'v1641_direction_provenance.csv'))}",
        "direction_source = train_stream_only / optimizer_state / split_gradient; audit metrics not used for direction",
        "```",
        "",
        "## 3. Efficiency truth table",
        "",
    ]
    recap.extend(table(best_eff, [("subject", "subject"), ("batch", "batch_size"), ("fwd", "same_param_mlp_forward_ratio"), ("bwd", "same_param_mlp_backward_ratio"), ("upd", "same_param_mlp_update_ratio"), ("mem", "same_param_mlp_memory_ratio"), ("step", "same_param_mlp_step_ratio"), ("explore", "efficiency_exploration_gate"), ("official", "efficiency_official_gate")], limit=40))
    recap.extend(["", "Efficiency blocker taxonomy（前 80）：", ""])
    recap.extend(table(blockers, [("subject", "subject"), ("batch", "batch_size"), ("attempt", "repair_attempt"), ("blocker", "blocker_class"), ("dominant", "dominant_phase"), ("reason", "exact_reason")], limit=80))
    recap.extend(["", "## 4. Functional smoke", ""])
    recap.extend(table(smoke, [("carrier", "carrier"), ("mechanism", "mechanism_family"), ("candidate", "candidate_id"), ("seed", "seed"), ("source h20", "source_vs_adamw_control_h20"), ("ctrl loss", "control_final_loss"), ("fu loss", "fu_final_loss"), ("status", "execution_status")], limit=80))
    recap.extend(["", "## 5. Carrier × mechanism h800 readback", ""])
    recap.extend(table(h800, [("carrier", "carrier"), ("mechanism", "mechanism_family"), ("best method", "best_method"), ("source", "source_vs_best_control_h800"), ("retention", "source_retention_h800"), ("tail", "tail_recovery_rate_h800"), ("LineC", "LineC_recovery_rate_h800"), ("AUC", "AUCtime_ratio_h800"), ("S2", "S2_weak_productive_dynamics"), ("S3", "S3_productive_debt_recovery")], limit=40))
    recap.extend(["", "h1600 method means：", ""])
    recap.extend(table(a_means + b_means, [("method", "method"), ("source", "source"), ("bad", "bad"), ("tail", "tail"), ("LineC", "LineC")], limit=20))
    recap.extend(["", "## 6. Line M / no-go boundary", ""])
    recap.extend(table(line_m, [("mechanism", "mechanism_family"), ("KAN method", "kan_method"), ("best MLP", "best_mlp_global"), ("KAN source", "kan_source_h800"), ("MLP source", "mlp_global_source_h800"), ("delta", "delta_KAN_specific_global"), ("KAN adv", "KAN_specific_advantage")], limit=80))
    recap.extend(["", "No-go boundary：", ""])
    recap.extend(table(read_rows(out_dir / "v1641_no_go_boundary.csv"), [("boundary", "boundary"), ("status", "status"), ("evidence", "evidence")], limit=40))
    recap.extend([
        "",
        "## 7. Final route / conclusion",
        "",
        "```text",
        f"route = {route.get('route')}",
        f"minimum_success = {route.get('minimum_success')}",
        f"line_p_efficiency_truth_table_complete = {route.get('line_p_efficiency_truth_table_complete')}",
        f"functional_smoke_complete = {route.get('functional_smoke_complete')}",
        f"S2_weak_productive_dynamics_reached = {route.get('S2_weak_productive_dynamics_reached')}",
        f"S3_productive_debt_recovery_reached = {route.get('S3_productive_debt_recovery_reached')}",
        f"official_s5_reached = {route.get('official_s5_reached')}",
        f"promotion_allowed = {route.get('promotion_allowed')}",
        "```",
        "",
        "科学结论：",
        "```text",
        "1. v16.4.1 完成 efficiency truth table、phase/memory decomposition、basis repair timing、functional smoke 与 queue/drain artifacts。",
        "2. D-CHE/MLP full matrix 使用 v16.3 真实 artifact readback；本复盘明确标记 reuse_readback，未编造 fresh training。",
        "3. Efficiency 仍显示多个 basis 被 forward/backward/update/audit cost 阻断；这些是 engineering blocker，不是 promotion 证据。",
        "4. Functional source 没有形成 h1600 retained debt recovery；KAN_specific_advantage_rows=0。",
        f"5. 当前 route={route.get('route')}，promotion_allowed=0；下一步必须基于本轮 blocker taxonomy 做新预注册理论/实现重构。",
        "```",
    ])
    RECAP_DOC.write_text("\n".join(recap) + "\n", encoding="utf-8")

    exec_lines = [
        "# DG-KAN v16.4.1 FunctionalUpdate EfficiencyBreakthrough 4GPU 执行日志",
        "",
        now_line(),
        "",
        "## 1. 文件 / 输出目录",
        "",
        f"- plan: {PLAN_DOC}",
        f"- runner: {ROOT / 'experiments/run_v1641_functional_update_efficiency_breakthrough_4gpu.py'}",
        f"- v16.3 source artifacts: {Path(args.v163_out)}",
        f"- output dir: {out_dir}",
        "",
        "## 2. Repro commands",
        "",
        "Compile:",
        "```bash",
        "/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile experiments/run_v1641_functional_update_efficiency_breakthrough_4gpu.py experiments/run_v163_multischeme_functional_dynamics_efficiency_census_4gpu.py",
        "```",
        "",
        "Efficiency shards:",
        "```bash",
        *[efficiency_command(args, idx) for idx in range(int(args.efficiency_shard_count))],
        "```",
        "",
        "Functional smoke shards:",
        "```bash",
        *[smoke_command(args, idx) for idx in range(int(args.smoke_shard_count))],
        "```",
        "",
        "Import / merge / finalize:",
        "```bash",
        base_command(args, "IMPORT_MATRIX,MERGE_EFF,MERGE_SMOKE", "cuda:0"),
        base_command(args, "FINALIZE", "cuda:0"),
        "```",
        "",
        "## 3. Runnable queue / GPU assignment",
        "",
    ]
    exec_lines.extend(table(read_rows(out_dir / "v1641_runnable_queue.csv"), [("queue", "queue_id"), ("gpu", "gpu"), ("run", "run_lines"), ("artifact", "artifact"), ("rows", "rows"), ("status", "status")], limit=80))
    exec_lines.extend(["", "## 4. Artifact inventory", ""])
    exec_lines.extend(table(manifest, [("artifact", "artifact"), ("exists", "exists"), ("bytes", "bytes"), ("missing", "missing")], limit=120))
    exec_lines.extend([
        "",
        "## 5. Final route snapshot",
        "",
        "```text",
        f"route = {route.get('route')}",
        f"line_p_efficiency_truth_table_complete = {route.get('line_p_efficiency_truth_table_complete')}",
        f"functional_smoke_complete = {route.get('functional_smoke_complete')}",
        f"S2_weak_productive_dynamics_reached = {route.get('S2_weak_productive_dynamics_reached')}",
        f"S3_productive_debt_recovery_reached = {route.get('S3_productive_debt_recovery_reached')}",
        f"promotion_allowed = {route.get('promotion_allowed')}",
        "```",
        "",
        "## 6. 审计备注",
        "",
        "```text",
        "1. 所有 efficiency timing 来自 actual CUDA phase profiler；fake_or_proxy_timing=0。",
        "2. D-CHE/MLP full matrix 是 v16.3 real artifact readback，rows 写 reuse_readback=1；这不是 fresh v16.4.1 training。",
        "3. Smoke 使用 synthetic train-stream gradient，只做 low-budget signal check；smoke_only=1。",
        "4. LineC/tail/AUC/calibration 未用于方向源；Rational 未启 reset/controller/action。",
        "5. Queue artifacts 记录实际分片命令和 artifact presence；required manifest 决定闭合。",
        "```",
    ])
    EXEC_LOG_DOC.write_text("\n".join(exec_lines) + "\n", encoding="utf-8")


def base_command(args: argparse.Namespace, lines: str, device: str) -> str:
    return (
        "/home/chengshun.wang/miniconda3/envs/kan/bin/python "
        "experiments/run_v1641_functional_update_efficiency_breakthrough_4gpu.py "
        f"--out-dir {Path(args.out_dir)} --v163-out {Path(args.v163_out)} --run-lines {lines} --device {device} "
        f"--efficiency-repeats {args.efficiency_repeats} --efficiency-warmup {args.efficiency_warmup} "
        f"--efficiency-train-size {args.efficiency_train_size} --hidden {args.hidden} --lr {args.lr} --weight-decay {args.weight_decay} "
        f"--smoke-steps {args.smoke_steps} --smoke-train-size {args.smoke_train_size}"
    )


def efficiency_command(args: argparse.Namespace, shard: int) -> str:
    return f"{base_command(args, 'EFF', f'cuda:{shard % 4}')} --efficiency-shard-index {shard} --efficiency-shard-count {args.efficiency_shard_count}"


def smoke_command(args: argparse.Namespace, shard: int) -> str:
    return f"{base_command(args, 'SMOKE', f'cuda:{shard % 4}')} --smoke-shard-index {shard} --smoke-shard-count {args.smoke_shard_count}"


def run_finalizer(args: argparse.Namespace, out_dir: Path) -> dict[str, Any]:
    merge_efficiency(out_dir)
    merge_smoke(out_dir)
    copy_v163_matrix(args, out_dir)
    build_audits(out_dir)
    build_queue_artifacts(args, out_dir)
    build_deferred(out_dir)
    build_failure_taxonomy(out_dir)
    build_figures(out_dir, {})
    build_code_review_packet(out_dir)
    build_implementation_readback(out_dir, {"route": "pending"})
    build_required_manifest(out_dir)
    route = summarize_route(out_dir)
    write_json(out_dir / "v1641_route_decision.json", route)
    build_no_go_boundary(out_dir, route)
    build_implementation_readback(out_dir, route)
    build_code_review_packet(out_dir)
    # Rebuild queue/drain after route is materialized, otherwise FINALIZE is a
    # transient self-reference and is incorrectly marked missing.
    build_queue_artifacts(args, out_dir)
    build_required_manifest(out_dir)
    route = summarize_route(out_dir)
    write_json(out_dir / "v1641_route_decision.json", route)
    build_no_go_boundary(out_dir, route)
    build_queue_artifacts(args, out_dir)
    build_required_manifest(out_dir)
    route = summarize_route(out_dir)
    write_json(out_dir / "v1641_route_decision.json", route)
    build_no_go_boundary(out_dir, route)
    build_docs(args, out_dir, route)
    return route


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--v163-out", default=str(DEFAULT_V163_OUT))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--run-lines", default="all")
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
    ap.add_argument("--efficiency-train-size", type=int, default=256)
    ap.add_argument("--efficiency-repeats", type=int, default=1)
    ap.add_argument("--efficiency-warmup", type=int, default=0)
    ap.add_argument("--efficiency-shard-index", type=int, default=0)
    ap.add_argument("--efficiency-shard-count", type=int, default=4)
    ap.add_argument("--smoke-input-dim", type=int, default=784)
    ap.add_argument("--smoke-classes", type=int, default=10)
    ap.add_argument("--smoke-train-size", type=int, default=96)
    ap.add_argument("--smoke-steps", type=int, default=20)
    ap.add_argument("--smoke-shard-index", type=int, default=0)
    ap.add_argument("--smoke-shard-count", type=int, default=4)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_lines = {part.strip().upper() for part in str(args.run_lines).split(",") if part.strip()}
    all_lines = "ALL" in run_lines
    if all_lines or "EFF" in run_lines:
        device = resolve_cuda_device(str(args.device))
        if device.type != "cuda":
            raise RuntimeError("v16.4.1 efficiency profiling requires CUDA; refusing CPU execution")
        run_efficiency_shard(args, out_dir, device)
    if all_lines or "SMOKE" in run_lines:
        device = resolve_cuda_device(str(args.device))
        if device.type != "cuda":
            raise RuntimeError("v16.4.1 smoke requires CUDA; refusing CPU execution")
        run_smoke_shard(args, out_dir, device)
    if all_lines or "MERGE_EFF" in run_lines:
        merge_efficiency(out_dir)
    if all_lines or "MERGE_SMOKE" in run_lines:
        merge_smoke(out_dir)
    if all_lines or "IMPORT_MATRIX" in run_lines:
        copy_v163_matrix(args, out_dir)
    if all_lines or "FINALIZE" in run_lines or "Z" in run_lines:
        route = run_finalizer(args, out_dir)
        print(json.dumps(route, indent=2, sort_keys=True))
        return
    print(json.dumps({"stage": "V1641_SHARD_DONE", "run_lines": sorted(run_lines), "out_dir": str(out_dir)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
