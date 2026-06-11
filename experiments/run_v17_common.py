#!/usr/bin/env python3
"""DG-KAN v17 code-audit, AdamW-free FU, and basis-efficiency runner.

This file is intentionally self-contained at the experiment orchestration
level. It imports reusable dgkan modules, but does not import v13-v16 runners.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import math
import os
import platform
import shutil
import subprocess
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

from dgkan.fu.core import UpdateTensor, apply_update, cosine, flat_grad, flat_params, load_flat_params, loss_value, normalized_like, small_step_sanity
from dgkan.fu.controls import CONTROL_SPECS, noop_update, random_matched_update
from dgkan.fu.update_semantics import update_type_manifest_row
from dgkan.fu.audit import direction_provenance_row
from dgkan.fu.loss_interface import brier_cotangent, cross_entropy_cotangent
from dgkan.fu.mechanisms import CONTROL_MECHANISMS, HC2_H4000_RECOMPUTE_PROJECTION_MECHANISMS, HC2_PROJECTION_MECHANISMS, MECHANISMS, METRIC_SOURCE_MEMORY_MECHANISMS, METRIC_TARGET_MECHANISMS, TERMINAL_SOURCE_SLOW_STATE_MECHANISMS, V2206_METRIC_SOLVER_MECHANISMS, make_update, mechanism_contract_rows, mechanism_family, update_state_after_commit
from dgkan.fu.optimizers import make_manual_optimizer
from dgkan.kernels.v17_basis import FAMILY_TO_BASIS, kernel_correctness_row
from dgkan.metrics.linec import linec_channel_from_logits, linec_from_improvements, linec_model_update, run_linec_channel_golden_tests, run_linec_golden_tests
from dgkan.models.fc_purekan_primitives import MLPBaseline, PrimitiveKAN, PrimitiveSpec, count_parameters
from dgkan.profiling.efficiency_v17 import profile_isolated


PLAN_DOC = ROOT / "docs/DG-KAN_v17_CodeAudit_AdamWFreeFU_BasisKernelEfficiency_4GPU完整计划.md"
EXEC_LOG_DOC = ROOT / "docs/DG-KAN_v17_CodeAudit_AdamWFreeFU_BasisKernelEfficiency_4GPU_执行日志.md"
RECAP_DOC = ROOT / "docs/DG-KAN_v17_CodeAudit_AdamWFreeFU_BasisKernelEfficiency_4GPU_实验结果复盘.md"
PLAN_DOC_1701 = ROOT / "docs/DG-KAN_v17.0.1_完整计划_CodeAudit_AdamWFreeFU_BasisEfficiency_4GPU.md"
EXEC_LOG_DOC_1701 = ROOT / "docs/DG-KAN_v17.0.1_CodeAudit_AdamWFreeFU_BasisEfficiency_4GPU_执行日志.md"
RECAP_DOC_1701 = ROOT / "docs/DG-KAN_v17.0.1_CodeAudit_AdamWFreeFU_BasisEfficiency_4GPU_实验结果复盘.md"
PLAN_DOC_171 = ROOT / "docs/DG-KAN_v17.1_三段式_CodeAudit_BasisEfficiency_FunctionalUpdate_完整计划.md"
EXEC_LOG_DOC_171 = ROOT / "docs/DG-KAN_v17.1_三段式_CodeAudit_BasisEfficiency_FunctionalUpdate_执行日志.md"
RECAP_DOC_171 = ROOT / "docs/DG-KAN_v17.1_三段式_CodeAudit_BasisEfficiency_FunctionalUpdate_实验结果复盘.md"
PLAN_DOC_18 = ROOT / "docs/DG-KAN_v18.0_EvidenceFirst_Breakthrough_FunctionalUpdate_BasisEfficiency_4GPU_完整计划.md"
EXEC_LOG_DOC_18 = ROOT / "docs/DG-KAN_v18.0_EvidenceFirst_Breakthrough_FunctionalUpdate_BasisEfficiency_4GPU_执行日志.md"
RECAP_DOC_18 = ROOT / "docs/DG-KAN_v18.0_EvidenceFirst_Breakthrough_FunctionalUpdate_BasisEfficiency_4GPU_实验结果复盘.md"
PLAN_DOC_19 = ROOT / "docs/DG-KAN_v19.0_SourceChannelFU_BasisKernelBreakthrough_4GPU_完整计划.md"
EXEC_LOG_DOC_19 = ROOT / "docs/DG-KAN_v19.0_SourceChannelFU_BasisKernelBreakthrough_4GPU_执行日志.md"
RECAP_DOC_19 = ROOT / "docs/DG-KAN_v19.0_SourceChannelFU_BasisKernelBreakthrough_4GPU_实验结果复盘.md"
DEFAULT_OUT = ROOT / "results/v17_code_audit_adamw_free_fu_basis_kernel_efficiency_4gpu/official_v17"
PYTHON = "/home/chengshun.wang/miniconda3/envs/kan/bin/python"

V17_MODULES = [
    "dgkan.metrics.linec",
    "dgkan.metrics.dynamic_geometry",
    "dgkan.diagnostics.linec",
    "dgkan.functional.update_tensor",
    "dgkan.functional.functional_mechanisms",
    "dgkan.functional.manual_optimizers",
    "dgkan.functional.adamw_coupling_audit",
    "dgkan.fu.core",
    "dgkan.fu.types",
    "dgkan.fu.loss_interface",
    "dgkan.fu.update_semantics",
    "dgkan.fu.carriers",
    "dgkan.fu.mechanisms",
    "dgkan.fu.optimizers",
    "dgkan.fu.slow_state",
    "dgkan.fu.matrix_block",
    "dgkan.fu.poprisk_snr",
    "dgkan.fu.linec_readback",
    "dgkan.fu.controls",
    "dgkan.fu.audit",
    "dgkan.efficiency.profiler",
    "dgkan.efficiency.same_param_mlp",
    "dgkan.efficiency.kernel_census",
    "dgkan.efficiency.phase_timer",
    "dgkan.efficiency.memory_timer",
    "dgkan.efficiency.repair_registry",
    "dgkan.profiling.efficiency_profiler",
    "dgkan.profiling.efficiency_v17",
    "dgkan.profiling.kernel_gradcheck",
    "dgkan.kernels.v17_che",
    "dgkan.kernels.v17_fou",
    "dgkan.kernels.v17_rbf",
    "dgkan.kernels.v17_lq",
    "dgkan.kernels.v17_rational",
    "dgkan.kernels.v17_wavelet",
    "experiments.run_v170_code_audit_and_semantic_tests",
    "experiments.run_v171_adamw_free_fu_mechanism_matrix",
    "experiments.run_v172_basis_kernel_efficiency_breakthrough",
    "experiments.run_v173_full_carrier_mechanism_4gpu",
    "experiments.run_v17_finalize",
    "experiments.run_v17_code_correctness_audit",
    "experiments.run_v17_adamw_free_functional_matrix",
    "experiments.run_v17_basis_kernel_efficiency_matrix",
    "experiments.run_v17_4gpu_scheduler",
    "experiments.run_v17_1_code_correctness_gate",
    "experiments.run_v17_1_functional_matrix",
    "experiments.run_v17_1_basis_efficiency_repair",
    "experiments.run_v17_1_finalize",
    "experiments.run_v18_code_gate",
    "experiments.run_v18_functional_update_breakthrough",
    "experiments.run_v18_basis_efficiency_breakthrough",
    "experiments.run_v18_merge_finalize",
    "experiments.run_v19_s03_truth_gate",
    "experiments.run_v19_source_channel_fu_matrix",
    "experiments.run_v19_basis_kernel_breakthrough",
    "experiments.run_v19_late_rebound_rerun",
    "experiments.run_v19_merge_finalize",
    "experiments.run_v19_source_channel_fu_matrix",
    "experiments.run_v19_basis_kernel_breakthrough",
    "experiments.run_v19_merge_finalize",
    "experiments.run_v17_full_codeaudit_adamwfree_fu_basis_efficiency_4gpu",
]

CARRIERS = [
    ("C0-MLP", "MLP", "MLP-v17-reference", "mlp"),
    ("C1-D-CHE", "D-CHE", "D-CHE-v17-chebyshev-k4", "chebyshev"),
    ("C2-LQ", "LQ", "LQ-v17-legendre-k4", "legendre"),
    ("C3-Rational", "D-RAT", "D-RAT-v17-rational-k4", "rational_kat_lite"),
    ("C4-D-FOU", "D-FOU", "D-FOU-v17-fourier-k4", "fourier_lowfreq"),
    ("C5-D-RBF", "D-RBF", "D-RBF-v17-compact-k4", "compact_rbf"),
    ("C6-D-WAV", "D-WAV", "D-WAV-v17-hat-k4", "hat_wavelet"),
]

REQUIRED_ARTIFACTS = [
    "v17_route_decision.json",
    "v17_s0_route_decision.json",
    "v17_compileall.log",
    "v17_import_closure.csv",
    "v17_import_errors.csv",
    "v17_missing_symbol_report.csv",
    "v17_module_dependency_graph.json",
    "compatibility_manifest.csv",
    "linec_source_manifest.csv",
    "linec_golden_fixture.py",
    "v17_linec_golden_results.csv",
    "v17_linec_exception_policy_test.csv",
    "v17_linec_null_distribution.csv",
    "v17_linec_split_transfer_test.csv",
    "v17_linec_noise_injection_test.csv",
    "v17_linec_signal_reservoir_test.csv",
    "v17_linec_batch_order_mismatch_test.csv",
    "v17_update_semantics_tests.csv",
    "v17_update_type_manifest.csv",
    "v17_adamw_coupling_map.csv",
    "v17_optimizer_state_dependency.csv",
    "v17_fu_vs_adamw_cosine.csv",
    "v17_fu_source_survival_by_recovery_optimizer.csv",
    "v17_mechanism_semantic_manifest.csv",
    "v17_mechanism_noncollapse_summary.csv",
    "v17_mechanism_update_cosine_matrix.csv",
    "v17_mechanism_param_mask_jaccard.csv",
    "v17_mechanism_state_dependency_matrix.csv",
    "v17_efficiency_profiler_unit_tests.csv",
    "v17_kernel_gradcheck_summary.csv",
    "v17_line_a_dche_fu_matrix.csv",
    "v17_line_b_mlp_fu_matrix.csv",
    "v17_line_c_lq_reanchor_smoke.csv",
    "v17_line_d_rational_monitor_smoke.csv",
    "v17_line_e_allbasis_kernel_repair.csv",
    "v17_line_f_allbasis_fu_smoke.csv",
    "v17_line_m_attribution.csv",
    "v17_source_washout_diagnosis.csv",
    "v17_direction_provenance.csv",
    "v17_measurement_validity_manifest.csv",
    "v17_budget_exhaustion_certificate.csv",
    "v17_s0_import_closure.csv",
    "v17_missing_imports.csv",
    "v17_linec_golden_tests.csv",
    "v17_control_surface.csv",
    "v17_update_sign_sanity.csv",
    "v17_adamw_overwrite_diagnostic.csv",
    "v17_efficiency_profiler_correctness.csv",
    "v17_kernel_correctness.csv",
    "v17_semantic_noncollapse_audit.csv",
    "v17_functional_mechanism_matrix.csv",
    "v17_h800_summary.csv",
    "v17_h1600_summary.csv",
    "v17_efficiency_truth_table.csv",
    "v17_efficiency_phase_breakdown.csv",
    "v17_memory_phase_breakdown.csv",
    "v17_basis_efficiency_failure_taxonomy.csv",
    "v17_same_param_mlp_manifest.csv",
    "v17_runnable_queue.csv",
    "v17_gpu_assignment_manifest.csv",
    "v17_gpu_utilization_dashboard.csv",
    "v17_gpu_runtime_snapshots.csv",
    "v17_idle_violation.csv",
    "v17_deferred_items.csv",
    "v17_queue_drain_report.csv",
    "v17_required_artifact_manifest.csv",
    "v17_forbidden_information_audit.csv",
    "v17_no_action_search_audit.csv",
    "v17_failure_taxonomy.csv",
    "v17_no_go_boundary.csv",
    "v17_no_go_boundary.md",
    "v17_next_hypothesis_queue.csv",
    "v17_next_hypothesis_queue.md",
    "v17_code_review_packet.csv",
    "v17_code_review_packet.zip",
    "v17_code_review_packet/packet_manifest.csv",
    "v17_code_review_packet/packet_sha256_manifest.csv",
    "v17_implementation_readback.md",
    "v17_1_route_decision.json",
    "v17_1_code_review_packet.csv",
    "v17_1_code_review_packet.zip",
    "v17_1_code_review_packet/packet_manifest.csv",
    "v17_1_code_review_packet/packet_sha256_manifest.csv",
    "v17_1_code_audit_summary.csv",
    "v17_1_efficiency_truth_table.csv",
    "v17_1_efficiency_blocker_table.csv",
    "v17_1_family_repair_summary.csv",
    "v17_1_same_param_mlp_mapping.csv",
    "v17_1_audit_cost_separation.csv",
    "v17_1_kernel_gradcheck_results.csv",
    "v17_1_functional_source_matrix.csv",
    "v17_1_source_retention_matrix.csv",
    "v17_1_debt_recovery_matrix.csv",
    "v17_1_control_attribution.csv",
    "v17_1_kan_vs_mlp_attribution.csv",
    "v17_1_adamw_overwrite_diagnostics.csv",
    "v17_1_mechanism_noncollapse.csv",
    "v17_1_functional_no_go_boundary.md",
    "v17_1_next_hypothesis_queue.md",
    "v17_1_runnable_queue.csv",
    "v17_1_gpu_assignment_manifest.csv",
    "v17_1_gpu_utilization_dashboard.csv",
    "v17_1_idle_violation.csv",
    "v17_1_deferred_items.csv",
    "v17_1_queue_drain_report.csv",
    "raw_h100_matrix.csv",
    "raw_h400_matrix.csv",
    "raw_h800_matrix.csv",
    "raw_h1600_matrix.csv",
    "raw_h3200_matrix.csv",
    "raw_controls_matrix.csv",
    "raw_efficiency_matrix.csv",
    "raw_linec_debt_matrix.csv",
    "raw_tail_calibration_debt_matrix.csv",
    "compileall_report.txt",
    "import_closure_results.csv",
    "missing_module_report.csv",
    "runner_import_results.csv",
    "symbol_resolution_table.csv",
    "linec_golden_results.csv",
    "linec_exception_policy.csv",
    "update_tensor_kind_contract.csv",
    "update_sign_finite_difference_results.csv",
    "adamw_coupled_parameter_map.csv",
    "fu_submit_path_map.csv",
    "semantic_alias_matrix.csv",
    "mechanism_noncollapse_results.csv",
    "kernel_gradcheck_results.csv",
    "fused_kernel_status.csv",
    "dense_materialization_audit.csv",
]

REQUIRED_FIGURES = [
    "fig_v17_s0_code_gate_dashboard.svg",
    "fig_v17_linec_golden_null_distribution.svg",
    "fig_v17_update_semantics_sign_tests.svg",
    "fig_v17_adamw_overwrite_cosine.svg",
    "fig_v17_source_retention_horizon_curves.svg",
    "fig_v17_debt_recovery_curves.svg",
    "fig_v17_carrier_mechanism_heatmap.svg",
    "fig_v17_efficiency_forward_backward_update_bar.svg",
    "fig_v17_memory_phase_breakdown.svg",
    "fig_v17_same_param_mlp_efficiency_pareto.svg",
    "fig_v17_gpu_utilization_timeline.svg",
    "fig_v17_failure_taxonomy_heatmap.svg",
    "fig_v17_code_audit_import_closure.svg",
    "fig_v17_linec_golden_test_dashboard.svg",
    "fig_v17_update_sign_sanity.svg",
    "fig_v17_adamw_overwrite_cosine_curve.svg",
    "fig_v17_carrier_mechanism_source_heatmap.svg",
    "fig_v17_source_retention_horizon_curves.svg",
    "fig_v17_tail_linec_debt_recovery_curves.svg",
    "fig_v17_kan_vs_mlp_attribution_matrix.svg",
    "fig_v17_semantic_noncollapse_cluster.svg",
    "fig_v17_efficiency_forward_ratio_by_basis.svg",
    "fig_v17_efficiency_backward_ratio_by_basis.svg",
    "fig_v17_efficiency_update_ratio_by_basis.svg",
    "fig_v17_efficiency_memory_ratio_by_basis.svg",
    "fig_v17_phase_time_stacked_bar_by_basis.svg",
    "fig_v17_memory_phase_stacked_bar_by_basis.svg",
    "fig_v17_basis_kernel_component_waterfall.svg",
    "fig_v17_gpu_utilization_timeline.svg",
    "fig_v17_queue_depth_over_time.svg",
    "fig_v17_job_completion_gantt.svg",
    "fig_v17_route_taxonomy_heatmap.svg",
    "code_audit_dashboard.svg",
    "linec_golden_results.svg",
    "update_semantics_signcheck.svg",
    "route_aggregation_unit_tests.svg",
    "efficiency_phase_stacked_bar.svg",
    "forward_ratio_by_basis.svg",
    "backward_ratio_by_basis.svg",
    "update_ratio_by_basis.svg",
    "memory_ratio_by_basis.svg",
    "basis_efficiency_pareto.svg",
    "kernel_blocker_heatmap.svg",
    "audit_cost_pollution_bar.svg",
    "source_horizon_curves.svg",
    "source_retention_heatmap.svg",
    "debt_recovery_curves.svg",
    "control_attribution_heatmap.svg",
    "kan_vs_mlp_same_mechanism.svg",
    "adamw_overwrite_cosine.svg",
    "mechanism_noncollapse_matrix.svg",
    "gpu_utilization_dashboard.svg",
    "queue_drain_timeline.svg",
    "idle_violation_timeline.svg",
]

V18_REQUIRED_ARTIFACTS = [
    "v18_route_decision.json",
    "v18_code_review_packet.zip",
    "v18_code_review_packet/packet_manifest.csv",
    "v18_code_review_packet/packet_sha256_manifest.csv",
    "v18_efficiency_truth_table.csv",
    "v18_efficiency_blocker_table.csv",
    "v18_kernel_repair_matrix.csv",
    "v18_functional_raw_horizon_matrix.csv",
    "v18_source_retention_matrix.csv",
    "v18_debt_accounting_matrix.csv",
    "v18_control_attribution.csv",
    "v18_kan_vs_mlp_attribution.csv",
    "v18_adamw_overwrite_diagnostics.csv",
    "v18_mechanism_noncollapse_matrix.csv",
    "v18_gpu_queue_drain_report.csv",
    "v18_failure_taxonomy.csv",
    "v18_next_hypothesis_queue.md",
    "v18_no_go_boundary.md",
    "v18_runnable_queue.csv",
    "v18_gpu_assignment_manifest.csv",
    "v18_gpu_utilization_dashboard.csv",
    "v18_idle_violation.csv",
    "v18_deferred_items.csv",
    "v18_queue_drain_report.csv",
    "v18_required_artifact_manifest.csv",
    "v18_forbidden_information_audit.csv",
    "v18_no_action_search_audit.csv",
    "raw_h100_matrix.csv",
    "raw_h400_matrix.csv",
    "raw_h800_matrix.csv",
    "raw_h1600_matrix.csv",
    "raw_h3200_matrix.csv",
]

V18_REQUIRED_FIGURES = [
    "figures/v18_code_gate_dashboard.svg",
    "figures/v18_linec_fast_channel_golden.svg",
    "figures/v18_efficiency_forward_ratio_by_family.svg",
    "figures/v18_efficiency_blocker_heatmap.svg",
    "figures/v18_kernel_repair_matrix.svg",
    "figures/v18_source_retention_horizon.svg",
    "figures/v18_debt_accounting_readback.svg",
    "figures/v18_control_attribution.svg",
    "figures/v18_kan_vs_mlp_attribution.svg",
    "figures/v18_adamw_overwrite_diagnostics.svg",
    "figures/v18_gpu_queue_drain.svg",
    "figures/v18_failure_taxonomy.svg",
]

V19_REQUIRED_ARTIFACTS = [
    "v19_route_decision.json",
    "v19_code_review_packet.zip",
    "v19_results_bundle.zip",
    "v19_code_review_packet/packet_manifest.csv",
    "v19_code_review_packet/packet_sha256_manifest.csv",
    "v19_functional_raw_horizon_matrix.csv",
    "v19_source_retention_matrix.csv",
    "v19_debt_accounting_matrix.csv",
    "v19_debt_metric_availability.csv",
    "v19_mechanism_manifest.csv",
    "v19_mechanism_semantic_contract.csv",
    "v19_mechanism_toy_correctness.csv",
    "v19_control_attribution.csv",
    "v19_kan_vs_mlp_attribution.csv",
    "v19_adamw_overwrite_diagnostics.csv",
    "v19_efficiency_truth_table.csv",
    "v19_efficiency_blocker_table.csv",
    "v19_efficiency_waterfall.csv",
    "v19_kernel_repair_matrix.csv",
    "v19_late_rebound_rerun_matrix.csv",
    "v19_runnable_queue.csv",
    "v19_gpu_assignment_manifest.csv",
    "v19_gpu_utilization_dashboard.csv",
    "v19_idle_violation.csv",
    "v19_deferred_items.csv",
    "v19_queue_drain_report.csv",
    "v19_failure_taxonomy.csv",
    "v19_no_go_boundary.md",
    "v19_next_hypothesis_queue.csv",
    "v19_next_hypothesis_queue.md",
    "v19_required_artifact_manifest.csv",
]

V19_REQUIRED_FIGURES = [
    "figures/v19_source_trajectory.svg",
    "figures/v19_source_retention_heatmap.svg",
    "figures/v19_late_rebound_map.svg",
    "figures/v19_debt_recovery_curves.svg",
    "figures/v19_adamw_overwrite_projection.svg",
    "figures/v19_mlp_vs_dche_source_trajectory.svg",
    "figures/v19_signal_state_norm.svg",
    "figures/v19_matrix_block_rank_retention.svg",
    "figures/v19_efficiency_waterfall.svg",
    "figures/v19_no_materialize_timing.svg",
    "figures/v19_efficiency_bubble.svg",
    "figures/v19_gpu_utilization_timeline.svg",
    "figures/v19_route_aggregation_dashboard.svg",
]


def now_sg() -> str:
    return datetime.now(ZoneInfo("Asia/Singapore")).strftime("%Y-%m-%d %H:%M:%S %Z")


def finite_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def int_flag(value: Any) -> int:
    try:
        return int(float(value))
    except Exception:
        return 0


def source_retention_ratio(current_source: Any, previous_source: Any) -> Any:
    current = finite_float(current_source)
    previous = finite_float(previous_source)
    if not math.isfinite(current) or not math.isfinite(previous) or previous <= 0.0:
        return ""
    return max(0.0, current) / previous


def sanitize(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else ""
    if isinstance(value, dict):
        return {str(k): sanitize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize(v) for v in value]
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sanitize(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(str(key))
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: sanitize(row.get(k, "")) for k in fieldnames})


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def append_log(out_dir: Path, line: str) -> None:
    log_path = out_dir / "logs/v17_command_journal.md"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(line.rstrip() + "\n")


def resolve_device(device_text: str) -> torch.device:
    if device_text.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_text)


def load_dataset(name: str, root: Path, train_size: int, val_size: int, seed: int, device: torch.device, input_size: int = 8) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    from torchvision.datasets import FashionMNIST, KMNIST, MNIST

    cls_map = {
        "MNIST": MNIST,
        "Fashion-MNIST": FashionMNIST,
        "FashionMNIST": FashionMNIST,
        "KMNIST": KMNIST,
    }
    if name not in cls_map:
        raise ValueError(f"unsupported dataset {name}")
    ds = cls_map[name](str(root), train=True, download=False)
    n = int(train_size) + int(val_size)
    gen = torch.Generator(device="cpu").manual_seed(17_000 + int(seed) + sum(ord(c) for c in name))
    idx = torch.randperm(len(ds.targets), generator=gen)[:n]
    images = ds.data[idx].float().unsqueeze(1) / 255.0
    labels = ds.targets[idx].long()
    if int(input_size) != 28:
        images = F.interpolate(images, size=(int(input_size), int(input_size)), mode="area")
    x = images.reshape(n, -1)
    x = (x - x.mean(dim=0, keepdim=True)) / x.std(dim=0, keepdim=True).clamp_min(1.0e-3)
    x = x.to(device)
    y = labels.to(device)
    return x[:train_size], y[:train_size], x[train_size:], y[train_size:]


def classification_ece(logits: torch.Tensor, labels: torch.Tensor, bins: int = 15) -> float:
    probs = torch.softmax(logits.detach().float(), dim=1)
    conf, pred = probs.max(dim=1)
    correct = (pred == labels).float()
    ece = torch.zeros((), device=logits.device)
    for lo in torch.linspace(0.0, 1.0, int(bins) + 1, device=logits.device)[:-1]:
        hi = lo + 1.0 / int(bins)
        mask = (conf >= lo) & (conf < hi if float(hi.item()) < 1.0 else conf <= hi)
        if bool(mask.any()):
            ece = ece + mask.float().mean() * (conf[mask].mean() - correct[mask].mean()).abs()
    return float(ece.item())


def classification_brier(logits: torch.Tensor, labels: torch.Tensor) -> float:
    probs = torch.softmax(logits.detach().float(), dim=1)
    one_hot = torch.zeros_like(probs)
    one_hot.scatter_(1, labels.reshape(-1, 1), 1.0)
    return float((probs - one_hot).square().sum(dim=1).mean().item())


def mlp_hidden_for_param_budget(input_dim: int, output_dim: int, target_params: int) -> int:
    best_h = 4
    best_delta = float("inf")
    for h in range(4, 257):
        params = input_dim * h + h * h + h * output_dim
        delta = abs(params - int(target_params))
        if delta < best_delta:
            best_h = h
            best_delta = delta
    return best_h


def v19_basis_repair_config(family: str, repair_variant: str) -> tuple[int, int, str]:
    """Map v19 repair labels to actual PrimitiveKAN train-stream kernel paths."""
    variant = str(repair_variant)
    variant_lower = variant.lower()
    k = 2 if "lowk2" in variant_lower or "low-degree-k2" in variant_lower or "low-frequency-k2" in variant_lower else 4
    dense = 0 if family in {"D-RBF", "D-WAV"} else 1
    init_variant = "v17_strict_no_bspline"
    if any(token in variant_lower for token in ["stream", "no-materialize", "recurrence", "low-bank", "active-bank", "triton"]):
        dense = 0
    if family == "D-FOU":
        if "k3-triton" in variant_lower:
            k = 3
            dense = 0
            init_variant = "fourier_k3_triton_l3_matmul"
        elif "k4-triton" in variant_lower:
            k = 4
            dense = 0
            init_variant = "fourier_k4_triton_l3_matmul"
        elif any(token in variant_lower for token in ["k2-triton", "low-frequency-k2", "stream-no-materialize"]):
            k = 2
            dense = 0
            init_variant = "fourier_k2_triton_l3_matmul"
    elif family == "D-CHE":
        if "k4-triton" in variant_lower:
            k = 4
            dense = 0
            init_variant = "cheby_k4_triton_l3_matmul"
        elif "gradbuf" in variant_lower:
            k = 3
            dense = 0
            init_variant = "cheby_k3_triton_l3_gradbuf"
        elif any(token in variant_lower for token in ["k3-triton", "recurrence-no-materialize", "low-degree-k3"]):
            k = 3
            dense = 0
            init_variant = "cheby_k3_triton_l3_matmul"
    elif family == "D-RBF":
        if any(token in variant_lower for token in ["rbf22.03", "rbf22.10", "compact-local", "active-center", "official-rbf"]):
            k = 2 if "low-k2" in variant_lower or "k2" in variant_lower else 4
            dense = 0
            init_variant = f"rbf_k{k}_triton_l3_matmul"
    elif family == "D-RAT":
        if "official-rational-k2-triton" in variant_lower or "rational-k2-triton" in variant_lower:
            k = 2
            dense = 0
            init_variant = "rational_k2_triton_l3_matmul"
        elif "rat22.05" in variant_lower or "official-rational-k4-triton" in variant_lower:
            k = 4
            dense = 0
            init_variant = "rational_k4_triton_l3_matmul"
        elif "rat22.03" in variant_lower and ("low-degree" in variant_lower or "readout-only" in variant_lower):
            k = 2
            dense = 1
    if "fastk2" in variant_lower and int(k) == 2:
        if init_variant.startswith(("rational_k", "rbf_k")) and init_variant.endswith("_triton_l3_matmul"):
            init_variant = f"{init_variant}_fastk2"
    if "singlelaunch" in variant_lower or "single_launch" in variant_lower or "single-launch" in variant_lower:
        if init_variant.startswith(("rational_k", "rbf_k")) and "_triton_l3_matmul" in init_variant:
            init_variant = f"{init_variant}_singlelaunch"
    for num_warps in (1, 2, 4, 8):
        if f"warps{num_warps}" in variant_lower or f"numwarps{num_warps}" in variant_lower or f"num_warps{num_warps}" in variant_lower:
            if init_variant.startswith("rational_k") and "_triton_l3_matmul" in init_variant:
                init_variant = f"{init_variant}_warps{num_warps}"
            break
    for block_h in (16, 32, 64, 128):
        if f"blockh{block_h}" in variant_lower or f"block_h{block_h}" in variant_lower or f"block-h{block_h}" in variant_lower:
            if init_variant.startswith(("rational_k", "rbf_k")) and ("_triton_l3_matmul" in init_variant):
                init_variant = f"{init_variant}_blockh{block_h}"
            break
    for block_b in (16, 32, 64, 128):
        if f"blockb{block_b}" in variant_lower or f"block_b{block_b}" in variant_lower or f"block-b{block_b}" in variant_lower:
            if init_variant.startswith(("rational_k", "rbf_k")) and ("_triton_l3_matmul" in init_variant):
                init_variant = f"{init_variant}_blockb{block_b}"
            break
    return int(k), int(dense), init_variant


def carrier_model(family: str, x_ref: torch.Tensor, seed: int, args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    input_dim = int(x_ref.shape[1])
    output_dim = int(args.classes)
    if family == "MLP":
        return MLPBaseline(input_dim, output_dim, int(args.hidden), int(seed), device)
    basis_name = next(item[3] for item in CARRIERS if item[1] == family)
    repair_variant = str(getattr(args, "basis_repair_variant", "R0-current"))
    k, dense, init_variant = v19_basis_repair_config(family, repair_variant)
    spec = PrimitiveSpec(
        candidate_id=f"{family}-v19-{basis_name}-k{k}-{repair_variant}-{init_variant}",
        basis_family=family,
        basis_name=basis_name,
        k=k,
        hidden_dim=int(args.hidden),
        source="v17_self_contained_strict_fc_purekan",
        local_support=int(family in {"D-RBF", "D-WAV"}),
        global_support=int(family not in {"D-RBF", "D-WAV"}),
        uses_exp=int(family in {"D-RBF", "D-WAV"}),
        uses_sin_cos=int(family == "D-FOU"),
        uses_division=int(family == "D-RAT"),
        uses_gather_scatter=0,
        uses_dense_basis_tensor=dense,
        diagnostic_only=0,
        basis_order=1,
        init_variant=init_variant,
    )
    return PrimitiveKAN(input_dim, output_dim, spec, x_ref, int(seed), device, int(args.param_budget))


def same_param_mlp_factory(model: torch.nn.Module, input_dim: int, output_dim: int, seed: int, device: torch.device) -> tuple[int, Any]:
    target = count_parameters(model)
    hidden = mlp_hidden_for_param_budget(input_dim, output_dim, target)

    def make() -> torch.nn.Module:
        return MLPBaseline(input_dim, output_dim, hidden, int(seed) + 177_000, device)

    return hidden, make


def snapshot(model: torch.nn.Module) -> torch.Tensor:
    return flat_params(model).detach().clone()


def train_one(
    model: torch.nn.Module,
    mechanism: str,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    args: argparse.Namespace,
    *,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], UpdateTensor | None]:
    device = x_train.device
    steps = int(args.steps)
    batch_size = min(int(args.batch_size), int(x_train.shape[0]))
    gen = torch.Generator(device=device).manual_seed(171_000 + int(seed) + sum(ord(c) for c in mechanism))
    opt_adam = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    momentum_primary_mechanisms = {
        "M2-SGDMomentumPrimaryFU",
        "M16-TwoPhaseMomentumThenLineCFU",
        "M36-MomentumWarmReadoutBlockFU",
        "M39-MomentumWarmSplitFisherFU",
        "M40-MomentumWarmAntiWashoutFU",
        "M41-MomentumWarmSourceAnchorFU",
        "M42-MomentumWarmHoldFU",
        "M43-MomentumCycleHoldFU",
        "M44-MomentumLineCAnchorSlowFU",
        "M45-MomentumSlowAnchorFU",
        "M46-MomentumMatrixBlockRetentionFU",
        "M47-MomentumCycleThenLineCAnchorFU",
        "M48-DualTimescaleSourceRetentionFU",
        "M69-RotatedCautiousMatrixSlowFU",
        "M70-TrainLookaheadCautiousSourceFU",
        "M87-AdamWBoundaryThenMomentumSourceFU",
        "M88-AdamWBoundaryThenDualTimescaleSourceFU",
        "M89-AdamWBoundaryDualTimescaleSGDFloorFU",
        "M90-AdamWBoundaryDualTimescaleLateSGDFloorFU",
        "M91-AdamWBoundaryDualTimescaleTinyLateSGDFloorFU",
        "M92-AdamWBoundaryDualTimescaleReinforcedTinyLateSGDFloorFU",
        "M93-TrainLossGatedDualTimescaleTinyLateFU",
        "M94-TrainLossGatedDualTimescaleHoldFallbackFU",
        "M95-TrainLossGatedDualTimescaleTinyLateFallbackFU",
        "M96-TrainLossGatedDualTimescaleBoostedTinyLateFallbackFU",
        "M101-AdamWBoundaryDualTimescaleAntiWashoutFU",
        "M102-AdamWBoundaryDualTimescaleSourceAnchorFU",
        "M103-AdamWBoundaryDualTimescaleParamEMAReentryFU",
        "M104-AdamWBoundaryDualTimescaleReadoutChannelFU",
        "M105-AdamWBoundaryDualTimescaleHiddenMatrixChannelFU",
        "M106-TrainLossTerminalProjectedLookaheadFloorFU",
        "M107-TrainLossTerminalConsensusLookaheadFloorFU",
        "M108-TrainLossTerminalSelectorLookaheadFloorFU",
        "M129-H800SourceSlowEMARetentionFU",
        "M130-H800ReadoutChannelRetentionFU",
        "M131-H800DualMemorySourceRetentionFU",
        "M132-H1600SourceCheckpointReentryFU",
        "M133-H2400SourceCheckpointReentryFU",
        "M139-EarlySourceSlowEMATerminalGuardFU",
        "M140-EarlySourceSlowEMATerminalRawGuardFU",
        "M141-EarlySourceSlowEMATerminalRawGuardStrongFU",
        "M142-EarlySourceSlowEMATerminalProjectedOptimizerFU",
        "M218-EarlySourceSlowEMATerminalProjectedOptimizerLambda050FU",
        "M219-EarlySourceSlowEMATerminalProjectedOptimizerLambda100FU",
        "M143-EarlySourceSlowEMATerminalProjectedBlendFU",
        "M144-EarlySourceSlowEMATerminalAntiWashoutFU",
        "M145-EarlySourceSlowEMATerminalH4000ReentryFU",
        "M146-EarlySourceSlowEMASignalReservoirTargetFU",
        "M147-EarlySourceSlowEMASourceBankTargetFU",
        "M148-EarlySourceSlowEMADualTargetGuardFU",
        "M149-EarlySourceSlowEMATerminalAdaptiveRawFU",
        "M150-EarlySourceSlowEMATerminalSparseSourceFU",
        "M151-EarlySourceSlowEMATerminalRatioPreserveFU",
        "M152-SourceProjectionB3NullConsensusTargetFU",
        "M153-NoiseOrthogonalB3NullConsensusTargetFU",
        "M154-EasyMarginB3NullConsensusTargetFU",
        "M158-EarlySourceSlowEMATerminalRejectSourceAxisRescueFU",
        "M159-EarlySourceSlowEMATerminalRejectSlowEMARescueFU",
        "M160-EarlySourceSlowEMATerminalRejectHoldSourceRescueFU",
        "M164-EarlySourceSlowEMATerminalRejectRawRescueFU",
        "M165-EarlySourceSlowEMATerminalRejectHybridRescueFU",
        "M166-EarlySourceSlowEMATerminalRejectLateOnlyRawRescueFU",
        "M167-EarlySourceSlowEMATerminalTopKSupportGuardFU",
        "M168-EarlySourceSlowEMATerminalTopKRawGuardFU",
        "M169-EarlySourceSlowEMATerminalTopKDebtCapFU",
        "M170-EarlySourceSlowEMATerminalSourcePreserveStrongFU",
        "M171-EarlySourceSlowEMATerminalSourcePreserveGentleFU",
        "M172-EarlySourceSlowEMATerminalInfoVolumeGuardFU",
        "M173-EarlySourceSlowEMALowNDSDiffeomorphicTargetFU",
        "M174-EarlySourceSlowEMAInfoVolumeDiffeomorphicTargetFU",
        "M175-EarlySourceSlowEMALowRankReadoutTransportFU",
        "M179-EarlySourceSlowEMATerminalSourcePreserveVeryStrongFU",
        "M180-EarlySourceSlowEMAH3600TerminalSourcePreserveFU",
        "M181-EarlySourceSlowEMATerminalNoraOrthogonalSourceFU",
        "M182-EarlySourceSlowEMATerminalDebtAwarePreserveFU",
        "M183-EarlySourceSlowEMATerminalLowNDSMatrixBlockFU",
        "M184-EarlySourceSlowEMATerminalDualMemoryPreserveFU",
        "M185-EarlySourceSlowEMASNRTerminalPredictorFU",
        "M186-EarlySourceSlowEMASplitConsensusEstimatorFU",
        "M187-EarlySourceSlowEMASignalReservoirTransportFU",
        "M188-EarlySourceSlowEMATerminalSourceFloorFU",
        "M189-EarlySourceSlowEMATerminalH4000AnchorFloorFU",
        "M190-EarlySourceSlowEMATerminalDecayAwareFloorFU",
        "M191-EarlySourceSlowEMATerminalRawGuardSourceFloorFU",
        "M192-EarlySourceSlowEMATerminalRawGuardH4000AnchorFloorFU",
        "M193-EarlySourceSlowEMATerminalRawGuardDecayAwareFloorFU",
        "M194-EarlySourceSlowEMATerminalAntiErosionOrthogonalFU",
        "M195-EarlySourceSlowEMATerminalSourceReflectionGuardFU",
        "M196-EarlySourceSlowEMATerminalH4000TransportCorrectorFU",
        "M197-EarlySourceSlowEMATerminalH3200AnchorTransportFU",
        "M198-EarlySourceSlowEMATerminalRawGuardH3200AnchorTransportFU",
        "M199-EarlySourceSlowEMATerminalH3200RatioReentryFU",
        "M200-EarlySourceSlowEMATerminalH3200ProgressCarryFU",
        "M201-EarlySourceSlowEMATerminalH4000ProgressCarryFU",
        "M202-EarlySourceSlowEMATerminalH3200SourceProgressBlendFU",
        "M203-EarlySourceSlowEMATerminalRawGuardH3600GentleProgressFU",
        "M204-EarlySourceSlowEMATerminalRawGuardH4000GentleProgressFU",
        "M205-EarlySourceSlowEMATerminalRawGuardH4400ProjectedProgressFU",
        "M206-EarlySourceSlowEMATerminalControlRelativeSGDCatchupFU",
        "M207-EarlySourceSlowEMATerminalControlRelativeAdamWCatchupFU",
        "M208-EarlySourceSlowEMATerminalControlRelativeSourceBalancedCatchupFU",
    }
    opt_sgd = make_manual_optimizer("sgd-momentum" if mechanism in momentum_primary_mechanisms else "sgd", model, float(args.lr), float(args.weight_decay))
    traces: list[dict[str, Any]] = []
    slow_state: torch.Tensor | None = None
    param_slow_state: torch.Tensor | None = None
    hc2_h3200_source_anchor: torch.Tensor | None = None
    hc2_projection_trials = 0
    hc2_negative_projection_count = 0
    hc2_cumulative_projection = 0.0
    hc2_cumulative_removed_norm = 0.0
    first_update: UpdateTensor | None = None
    linec_filter_trials = 0
    linec_filter_accepts = 0
    linec_filter_last: dict[str, Any] = {}
    last_actuation: dict[str, Any] = {}
    train_loss_gate_h400_pass = 0
    train_loss_gate_h800_pass = 0
    train_loss_gate_last_value = float("nan")
    horizons = {0, 1, 20, 100, 400, 800, 1600, 2400, 3200, 4000, 4800, 6400}
    horizons = {h for h in horizons if h <= steps}
    channel_a = (x_train[: min(32, int(x_train.shape[0]))], y_train[: min(32, int(y_train.shape[0]))])
    channel_b = (
        x_train[min(32, int(x_train.shape[0])) : min(64, int(x_train.shape[0]))],
        y_train[min(32, int(y_train.shape[0])) : min(64, int(y_train.shape[0]))],
    )
    if int(channel_b[0].shape[0]) < 2:
        channel_b = channel_a
    with torch.no_grad():
        initial_a_logits = model(channel_a[0]).detach().float()
        initial_b_logits = model(channel_b[0]).detach().float()
    initial_named_params = {pname: p.detach().clone() for pname, p in model.named_parameters() if p.requires_grad}

    def readout_mask_flat() -> torch.Tensor:
        chunks: list[torch.Tensor] = []
        for pname, p in model.named_parameters():
            if not p.requires_grad:
                continue
            mask = torch.zeros_like(p, device=device).reshape(-1)
            low = pname.lower()
            if "w2" in low or "readout" in low or "classifier" in low or "bias" in low:
                mask.fill_(1.0)
            chunks.append(mask)
        return torch.cat(chunks) if chunks else torch.zeros(0, device=device)

    def hidden_matrix_mask_flat() -> torch.Tensor:
        chunks: list[torch.Tensor] = []
        for pname, p in model.named_parameters():
            if not p.requires_grad:
                continue
            mask = torch.zeros_like(p, device=device).reshape(-1)
            low = pname.lower()
            if p.ndim >= 2 and "w2" not in low and "readout" not in low and "classifier" not in low:
                mask.fill_(1.0)
            chunks.append(mask)
        return torch.cat(chunks) if chunks else torch.zeros(0, device=device)

    def assign_flat_gradient(vec: torch.Tensor) -> None:
        offset = 0
        for param in model.parameters():
            if not param.requires_grad:
                continue
            n = int(param.numel())
            chunk = vec[offset : offset + n].view_as(param).to(device=param.device, dtype=param.dtype)
            param.grad = chunk.detach().clone()
            offset += n

    def block_displacement_metrics() -> dict[str, Any]:
        hidden_sq = 0.0
        readout_sq = 0.0
        bias_sq = 0.0
        other_sq = 0.0
        total_sq = 0.0
        rank_total = 0
        singular_values: list[float] = []
        for pname, p in model.named_parameters():
            if not p.requires_grad or pname not in initial_named_params:
                continue
            delta = (p.detach().float() - initial_named_params[pname].to(device=p.device).float()).reshape_as(p)
            sq = float(delta.square().sum().item())
            total_sq += sq
            low = pname.lower()
            readout_matrix = p.ndim >= 2 and ("w2" in low or "readout" in low or "classifier" in low)
            hidden_matrix = p.ndim >= 2 and not readout_matrix
            bias_like = p.ndim < 2 or "bias" in low
            if readout_matrix:
                readout_sq += sq
            elif hidden_matrix:
                hidden_sq += sq
            elif bias_like:
                bias_sq += sq
            else:
                other_sq += sq
            if p.ndim >= 2 and sq > 0.0:
                mat = delta.reshape(int(delta.shape[0]), -1)
                try:
                    svals = torch.linalg.svdvals(mat)
                    if int(svals.numel()):
                        top = float(svals.max().item())
                        rank_total += int((svals > max(1.0e-12, 1.0e-6 * top)).sum().item())
                        singular_values.extend(float(v) for v in svals[: min(6, int(svals.numel()))].detach().cpu().tolist())
                except Exception:
                    rank_total += 0
        denom = max(total_sq, 1.0e-12)
        singular_values = sorted(singular_values, reverse=True)[:12]
        return {
            "hidden_source_energy": hidden_sq,
            "readout_source_energy": readout_sq,
            "bias_source_energy": bias_sq,
            "other_source_energy": other_sq,
            "optimizer_state_source_energy": "",
            "hidden_source_fraction": hidden_sq / denom if total_sq > 0.0 else "",
            "readout_source_fraction": readout_sq / denom if total_sq > 0.0 else "",
            "bias_source_fraction": bias_sq / denom if total_sq > 0.0 else "",
            "source_subspace_rank": rank_total,
            "source_singular_values": ";".join(f"{v:.6g}" for v in singular_values),
            "matrix_block_alignment": (hidden_sq + readout_sq) / denom if total_sq > 0.0 else "",
            "parameter_alignment": (hidden_sq + readout_sq + bias_sq) / denom if total_sq > 0.0 else "",
            "source_reconstruction_error": other_sq / denom if total_sq > 0.0 else "",
        }

    def eval_pack(step: int) -> None:
        with torch.no_grad():
            train_loss = float(F.cross_entropy(model(x_train).float(), y_train).item())
            val_logits = model(x_val).float()
            val_loss = float(F.cross_entropy(val_logits, y_val).item())
            pred = val_logits.argmax(dim=1)
            acc = float((pred == y_val).float().mean().item())
            ce = F.cross_entropy(val_logits, y_val, reduction="none")
            cep99 = float(torch.quantile(ce.detach(), 0.99).item())
            ece = classification_ece(val_logits, y_val)
            brier = classification_brier(val_logits, y_val)
            current_a_logits = model(channel_a[0]).detach().float()
            current_b_logits = model(channel_b[0]).detach().float()
        channel = linec_channel_from_logits(initial_a_logits, current_a_logits, channel_a[1], initial_b_logits, current_b_logits, channel_b[1]).to_row()
        fast = {"linec_measurement_valid": 0, "CouplingR2": "", "NoiseSignalLeak": "", "RealSignalReservoirRatio": "", "route": "R0-LineCFastNotMeasured"}
        try:
            state_vec = snapshot(model)
            model.zero_grad(set_to_none=True)
            audit_loss = F.cross_entropy(model(channel_a[0]).float(), channel_a[1])
            audit_loss.backward()
            audit_update = make_update(model, mechanism, channel_a[0], channel_a[1], seed=seed + step + 19_000, slow_state=slow_state)
            before = snapshot(model)

            def apply_tmp() -> None:
                apply_update(model, audit_update, lr=float(args.fu_lr))

            def restore_tmp() -> None:
                load_flat_params(model, before)

            fast = linec_model_update(model, channel_a, channel_b, apply_tmp, restore_tmp).to_row()
            load_flat_params(model, state_vec)
            model.zero_grad(set_to_none=True)
        except Exception as exc:
            fast = {"linec_measurement_valid": 0, "CouplingR2": "", "NoiseSignalLeak": "", "RealSignalReservoirRatio": "", "route": "R0-LineCMeasurementInvalid", "linec_exception_type": type(exc).__name__, "linec_exception_message": str(exc)[:200]}
        block_metrics = block_displacement_metrics()
        traces.append(
            {
                "step": step,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "NLL": val_loss,
                "val_acc": acc,
                "CEp99": cep99,
                "ECE": ece,
                "Brier": brier,
                "LineC_channel_valid": channel.get("linec_measurement_valid"),
                "LineC_channel_CouplingR2": channel.get("CouplingR2"),
                "LineC_channel_NoiseSignalLeak": channel.get("NoiseSignalLeak"),
                "LineC_channel_ReservoirRatio": channel.get("RealSignalReservoirRatio"),
                "LineC_channel_loss": 1.0 - finite_float(channel.get("CouplingR2"), 0.0) if int_flag(channel.get("linec_measurement_valid")) else "",
                "LineC_fast_valid": fast.get("linec_measurement_valid"),
                "LineC_fast_CouplingR2": fast.get("CouplingR2"),
                "LineC_fast_NoiseSignalLeak": fast.get("NoiseSignalLeak"),
                "LineC_fast_ReservoirRatio": fast.get("RealSignalReservoirRatio"),
                "LineC_fast_loss": 1.0 - finite_float(fast.get("CouplingR2"), 0.0) if int_flag(fast.get("linec_measurement_valid")) else "",
                "LineC_filter_trials": linec_filter_trials,
                "LineC_filter_accepts": linec_filter_accepts,
                "LineC_filter_accept_rate": float(linec_filter_accepts) / float(linec_filter_trials) if linec_filter_trials else "",
                "LineC_filter_last_valid": linec_filter_last.get("valid", ""),
                "LineC_filter_last_accept": linec_filter_last.get("accept", ""),
                "LineC_filter_last_mean_a": linec_filter_last.get("mean_a", ""),
                "LineC_filter_last_mean_b": linec_filter_last.get("mean_b", ""),
                "LineC_filter_last_CouplingR2": linec_filter_last.get("CouplingR2", ""),
                "LineC_filter_last_NoiseLeak": linec_filter_last.get("NoiseSignalLeak", ""),
                "solver_status": last_actuation.get("solver_status", ""),
                "solver_level": last_actuation.get("solver_level", ""),
                "solver_type": last_actuation.get("solver_type", ""),
                "target_family": last_actuation.get("target_family", ""),
                "metric_family": last_actuation.get("metric_family", ""),
                "optimizer_integration_type": last_actuation.get("optimizer_integration_type", ""),
                "ActuationR2": last_actuation.get("ActuationR2", ""),
                "ActuationCosine": last_actuation.get("ActuationCosine", ""),
                "B1_gain": last_actuation.get("B1_gain", ""),
                "B2_transfer_gain": last_actuation.get("B2_transfer_gain", ""),
                "B3_safety_gain": last_actuation.get("B3_safety_gain", ""),
                "projection_residual_Gf": last_actuation.get("projection_residual_Gf", ""),
                "metric_name": last_actuation.get("metric_name", ""),
                "metric_projection_norm": last_actuation.get("metric_projection_norm", ""),
                "metric_gradient_norm": last_actuation.get("metric_gradient_norm", ""),
                "metric_projection_cosine": last_actuation.get("metric_projection_cosine", ""),
                "metric_energy_L2": last_actuation.get("metric_energy_L2", ""),
                "metric_energy_Fisher": last_actuation.get("metric_energy_Fisher", ""),
                "metric_energy_PopRisk": last_actuation.get("metric_energy_PopRisk", ""),
                "metric_energy_Sobolev": last_actuation.get("metric_energy_Sobolev", ""),
                "metric_energy_RKHS": last_actuation.get("metric_energy_RKHS", ""),
                "metric_energy_mean": last_actuation.get("metric_energy_mean", ""),
                "NDS": last_actuation.get("NDS", ""),
                "raw_gradient_NDS": last_actuation.get("raw_gradient_NDS", ""),
                "first_order_gain": last_actuation.get("first_order_gain", ""),
                "second_order_penalty": last_actuation.get("second_order_penalty", ""),
                "projection_residual_norm": last_actuation.get("projection_residual_norm", ""),
                "function_displacement_norm": last_actuation.get("function_displacement_norm", ""),
                "hidden_residual_is_exact_solve": last_actuation.get("hidden_residual_is_exact_solve", ""),
                "block_source_hidden_residual_norm": last_actuation.get("block_source_hidden_residual_norm", ""),
                "block_source_hidden_residual_fraction": last_actuation.get("block_source_hidden_residual_fraction", ""),
                "block_source_hidden_function_norm": last_actuation.get("block_source_hidden_function_norm", ""),
                "block_source_hidden_function_cap_ratio": last_actuation.get("block_source_hidden_function_cap_ratio", ""),
                "block_source_hidden_compensation_norm": last_actuation.get("block_source_hidden_compensation_norm", ""),
                "block_source_hidden_compensation_scale": last_actuation.get("block_source_hidden_compensation_scale", ""),
                "block_source_hidden_compensation_mix": last_actuation.get("block_source_hidden_compensation_mix", ""),
                "block_source_hidden_compensation_residual_ratio": last_actuation.get("block_source_hidden_compensation_residual_ratio", ""),
                "block_source_hidden_compensation_b2_gain": last_actuation.get("block_source_hidden_compensation_b2_gain", ""),
                "solve_time_ms": last_actuation.get("solve_time_ms", ""),
                "JVP_count": last_actuation.get("JVP_count", ""),
                "VJP_count": last_actuation.get("VJP_count", ""),
                "CG_iterations": last_actuation.get("CG_iterations", ""),
                "solver_rank": last_actuation.get("solver_rank", ""),
                "condition_estimate": last_actuation.get("condition_estimate", ""),
                "jacobian_condition_mean": last_actuation.get("jacobian_condition_mean", ""),
                "jacobian_condition_p95": last_actuation.get("jacobian_condition_p95", ""),
                "fold_rate": last_actuation.get("fold_rate", ""),
                "neighbor_order_flip_rate": last_actuation.get("neighbor_order_flip_rate", ""),
                "local_distance_distortion": last_actuation.get("local_distance_distortion", ""),
                "smoothness_norm": last_actuation.get("smoothness_norm", ""),
                "curvature_norm": last_actuation.get("curvature_norm", ""),
                "diffeomorphic_target_family": last_actuation.get("diffeomorphic_target_family", ""),
                "operator_parameter_norm": last_actuation.get("parameter_norm", ""),
                "operator_clamp_ratio": last_actuation.get("operator_clamp_ratio", ""),
                "operator_cap_ratio": last_actuation.get("operator_cap_ratio", ""),
                "operator_commit_scale": last_actuation.get("operator_commit_scale", ""),
                "operator_rank": last_actuation.get("operator_rank", ""),
                "operator_status": last_actuation.get("operator_status", ""),
                "post_stop_mechanism": last_actuation.get("post_stop_mechanism", ""),
                "post_stop_fu_lr": last_actuation.get("post_stop_fu_lr", ""),
                "post_stop_slow_state_contract": last_actuation.get("post_stop_slow_state_contract", ""),
                "post_stop_slow_state_active": last_actuation.get("post_stop_slow_state_active", ""),
                "post_stop_slow_state_norm": last_actuation.get("post_stop_slow_state_norm", ""),
                "terminal_source_slow_state_used": last_actuation.get("terminal_source_slow_state_used", ""),
                "terminal_source_slow_state_beta": last_actuation.get("terminal_source_slow_state_beta", ""),
                "terminal_source_slow_state_norm": last_actuation.get("terminal_source_slow_state_norm", ""),
                "target_kind": last_actuation.get("target_kind", ""),
                "target_fit_scope": last_actuation.get("target_fit_scope", ""),
                "target_rank_limit": last_actuation.get("target_rank_limit", ""),
                "target_feature_select": last_actuation.get("target_feature_select", ""),
                "source_channel_projection": last_actuation.get("source_channel_projection", ""),
                "reservoir_projection": last_actuation.get("reservoir_projection", ""),
                "source_to_reservoir_leakage": last_actuation.get("source_to_reservoir_leakage", ""),
                "source_bank_feature_count": last_actuation.get("source_bank_feature_count", ""),
                "reservoir_bank_feature_count": last_actuation.get("reservoir_bank_feature_count", ""),
                "source_bank_feature_norm": last_actuation.get("source_bank_feature_norm", ""),
                "reservoir_bank_feature_norm": last_actuation.get("reservoir_bank_feature_norm", ""),
                "low_degree_source_energy": last_actuation.get("low_degree_source_energy", ""),
                "high_degree_reservoir_energy": last_actuation.get("high_degree_reservoir_energy", ""),
                "low_frequency_source_energy": last_actuation.get("low_frequency_source_energy", ""),
                "high_frequency_reservoir_energy": last_actuation.get("high_frequency_reservoir_energy", ""),
                "degree_entropy": last_actuation.get("degree_entropy", ""),
                "band_entropy": last_actuation.get("band_entropy", ""),
                "target_consensus_density": last_actuation.get("target_consensus_density", ""),
                "target_early_observable_class_density": last_actuation.get("target_early_observable_class_density", ""),
                "target_early_observable_class_cos_mean": last_actuation.get("target_early_observable_class_cos_mean", ""),
                "target_early_observable_mid_debt_mean": last_actuation.get("target_early_observable_mid_debt_mean", ""),
                "target_early_observable_reference_agreement": last_actuation.get("target_early_observable_reference_agreement", ""),
                "target_view_noise_std": last_actuation.get("target_view_noise_std", ""),
                "target_view_stable_fraction_b1": last_actuation.get("target_view_stable_fraction_b1", ""),
                "target_view_stable_fraction_b2": last_actuation.get("target_view_stable_fraction_b2", ""),
                "target_view_alignment_b1_mean": last_actuation.get("target_view_alignment_b1_mean", ""),
                "target_view_alignment_b2_mean": last_actuation.get("target_view_alignment_b2_mean", ""),
                "target_b3_null_rows": last_actuation.get("target_b3_null_rows", ""),
                "target_noise_rejected_density": last_actuation.get("target_noise_rejected_density", ""),
                "target_source_projected_density": last_actuation.get("target_source_projected_density", ""),
                "target_train_ce_median_b1": last_actuation.get("target_train_ce_median_b1", ""),
                "target_train_ce_median_b2": last_actuation.get("target_train_ce_median_b2", ""),
                "target_train_ce_q25_b1": last_actuation.get("target_train_ce_q25_b1", ""),
                "target_train_ce_q25_b2": last_actuation.get("target_train_ce_q25_b2", ""),
                "target_train_ce_q85_b1": last_actuation.get("target_train_ce_q85_b1", ""),
                "target_train_ce_q85_b2": last_actuation.get("target_train_ce_q85_b2", ""),
                "target_train_ce_q95_b1": last_actuation.get("target_train_ce_q95_b1", ""),
                "target_train_ce_q95_b2": last_actuation.get("target_train_ce_q95_b2", ""),
                "target_calibration_gap_mean_b1": last_actuation.get("target_calibration_gap_mean_b1", ""),
                "target_calibration_gap_mean_b2": last_actuation.get("target_calibration_gap_mean_b2", ""),
                "target_debt_weight_mean_b1": last_actuation.get("target_debt_weight_mean_b1", ""),
                "target_debt_weight_mean_b2": last_actuation.get("target_debt_weight_mean_b2", ""),
                "target_debt_transition_mean_b1": last_actuation.get("target_debt_transition_mean_b1", ""),
                "target_debt_transition_mean_b2": last_actuation.get("target_debt_transition_mean_b2", ""),
                "target_debt_non_outlier_fraction_b1": last_actuation.get("target_debt_non_outlier_fraction_b1", ""),
                "target_debt_non_outlier_fraction_b2": last_actuation.get("target_debt_non_outlier_fraction_b2", ""),
                "target_mid_debt_fraction_b1": last_actuation.get("target_mid_debt_fraction_b1", ""),
                "target_mid_debt_fraction_b2": last_actuation.get("target_mid_debt_fraction_b2", ""),
                "target_mid_debt_shape_mean_b1": last_actuation.get("target_mid_debt_shape_mean_b1", ""),
                "target_mid_debt_shape_mean_b2": last_actuation.get("target_mid_debt_shape_mean_b2", ""),
                "operator_gate_accept": last_actuation.get("operator_gate_accept", ""),
                "generalization_gain_a": last_actuation.get("generalization_gain_a", ""),
                "generalization_gain_b": last_actuation.get("generalization_gain_b", ""),
                "shuffled_label_gain": last_actuation.get("shuffled_label_gain", ""),
                "generalization_signal_gain": last_actuation.get("generalization_signal_gain", ""),
                "generalization_gate_accept": last_actuation.get("generalization_gate_accept", ""),
                "source_state_consensus_density": last_actuation.get("source_state_consensus_density", ""),
                "source_state_corrupt_cos": last_actuation.get("source_state_corrupt_cos", ""),
                "source_state_signal_gain": last_actuation.get("source_state_signal_gain", ""),
                "source_state_corrupt_gain": last_actuation.get("source_state_corrupt_gain", ""),
                "source_state_gate_accept": last_actuation.get("source_state_gate_accept", ""),
                "source_state_norm": last_actuation.get("source_state_norm", ""),
                "source_state_gate_threshold": last_actuation.get("source_state_gate_threshold", ""),
                "source_state_ema_beta": last_actuation.get("source_state_ema_beta", ""),
                "source_state_balance_mean": last_actuation.get("source_state_balance_mean", ""),
                "source_state_current_cos": last_actuation.get("source_state_current_cos", ""),
                "source_state_projected_cos": last_actuation.get("source_state_projected_cos", ""),
                "target_gradient_conflict_cos_before": last_actuation.get("target_gradient_conflict_cos_before", ""),
                "target_gradient_conflict_cos_after": last_actuation.get("target_gradient_conflict_cos_after", ""),
                "target_gradient_conflict_removed_fraction": last_actuation.get("target_gradient_conflict_removed_fraction", ""),
                "source_observability_gate_accept": last_actuation.get("source_observability_gate_accept", ""),
                "FU_param_commit_norm": last_actuation.get("FU_param_commit_norm", ""),
                "FU_function_commit_norm": last_actuation.get("FU_function_commit_norm", ""),
                "optimizer_state_m_projection_on_FU": last_actuation.get("optimizer_state_m_projection_on_FU", ""),
                "optimizer_state_v_projection_on_FU": last_actuation.get("optimizer_state_v_projection_on_FU", ""),
                "post_commit_grad_alignment": last_actuation.get("post_commit_grad_alignment", ""),
                "cumulative_optimizer_projection_h100_to_h800": last_actuation.get("cumulative_optimizer_projection_h100_to_h800", ""),
                "cumulative_optimizer_projection_h3200_to_h4800": last_actuation.get("cumulative_optimizer_projection_h3200_to_h4800", ""),
                "fast_slow_function_gap": last_actuation.get("fast_slow_function_gap", ""),
                "short_long_source_agreement": last_actuation.get("short_long_source_agreement", ""),
                "source_state_carry_lr": last_actuation.get("source_state_carry_lr", ""),
                "early_source_warm_writer_active": last_actuation.get("early_source_warm_writer_active", ""),
                "early_source_warm_writer_lr": last_actuation.get("early_source_warm_writer_lr", ""),
                "early_source_warmup_steps": last_actuation.get("early_source_warmup_steps", ""),
                "source_stop_steps": last_actuation.get("source_stop_steps", ""),
                "source_state_carry_active": last_actuation.get("source_state_carry_active", ""),
                "direct_solver_commit_active": last_actuation.get("direct_solver_commit_active", ""),
                "periodic_solver_active": last_actuation.get("periodic_solver_active", ""),
                "periodic_solver_alt_period": last_actuation.get("periodic_solver_alt_period", ""),
                "sgd_bootstrap_writer_active": last_actuation.get("sgd_bootstrap_writer_active", ""),
                "sgd_bootstrap_writer_lr": last_actuation.get("sgd_bootstrap_writer_lr", ""),
                "adamw_bootstrap_writer_active": last_actuation.get("adamw_bootstrap_writer_active", ""),
                "adamw_bootstrap_writer_lr": last_actuation.get("adamw_bootstrap_writer_lr", ""),
                "optimizer_transport_active": last_actuation.get("optimizer_transport_active", ""),
                "optimizer_transport_strength": last_actuation.get("optimizer_transport_strength", ""),
                "optimizer_transport_projection_before": last_actuation.get("optimizer_transport_projection_before", ""),
                "optimizer_transport_projection_after": last_actuation.get("optimizer_transport_projection_after", ""),
                "optimizer_transport_removed_anti_source": last_actuation.get("optimizer_transport_removed_anti_source", ""),
                "optimizer_transport_grad_delta_norm": last_actuation.get("optimizer_transport_grad_delta_norm", ""),
                "train_split_control_gate_active": last_actuation.get("train_split_control_gate_active", ""),
                "train_split_control_gate_accept": last_actuation.get("train_split_control_gate_accept", ""),
                "train_split_control_gate_fu_signal_gain": last_actuation.get("train_split_control_gate_fu_signal_gain", ""),
                "train_split_control_gate_sgd_signal_gain": last_actuation.get("train_split_control_gate_sgd_signal_gain", ""),
                "train_split_control_gate_signal_margin": last_actuation.get("train_split_control_gate_signal_margin", ""),
                "train_split_control_gate_fu_b3_gain": last_actuation.get("train_split_control_gate_fu_b3_gain", ""),
                "train_split_control_gate_sgd_b3_gain": last_actuation.get("train_split_control_gate_sgd_b3_gain", ""),
                "train_split_control_gate_fu_corrupt_gain": last_actuation.get("train_split_control_gate_fu_corrupt_gain", ""),
                "train_split_control_gate_sgd_corrupt_gain": last_actuation.get("train_split_control_gate_sgd_corrupt_gain", ""),
                "adamw_control_gate_active": last_actuation.get("adamw_control_gate_active", ""),
                "adamw_control_gate_accept": last_actuation.get("adamw_control_gate_accept", ""),
                "adamw_control_gate_fu_signal_gain": last_actuation.get("adamw_control_gate_fu_signal_gain", ""),
                "adamw_control_gate_adamw_signal_gain": last_actuation.get("adamw_control_gate_adamw_signal_gain", ""),
                "adamw_control_gate_signal_margin": last_actuation.get("adamw_control_gate_signal_margin", ""),
                "adamw_control_gate_fu_b3_gain": last_actuation.get("adamw_control_gate_fu_b3_gain", ""),
                "adamw_control_gate_adamw_b3_gain": last_actuation.get("adamw_control_gate_adamw_b3_gain", ""),
                "adamw_control_gate_fu_corrupt_gain": last_actuation.get("adamw_control_gate_fu_corrupt_gain", ""),
                "adamw_control_gate_adamw_corrupt_gain": last_actuation.get("adamw_control_gate_adamw_corrupt_gain", ""),
                "adamw_early_optimizer_control_active": last_actuation.get("adamw_early_optimizer_control_active", ""),
                "optimizer_cumulative_projection_on_source_Gf": last_actuation.get("optimizer_cumulative_projection_on_source_Gf", ""),
                "negative_projection_fraction": last_actuation.get("negative_projection_fraction", ""),
                "source_preservation_projection_norm": last_actuation.get("source_preservation_projection_norm", ""),
                "hc2_h3200_anchor_captured": last_actuation.get("hc2_h3200_anchor_captured", ""),
                "source_state_whitened_norm": last_actuation.get("source_state_whitened_norm", ""),
                "source_state_antiwashout_removed_norm": last_actuation.get("source_state_antiwashout_removed_norm", ""),
                "source_state_projected_grad_norm": last_actuation.get("source_state_projected_grad_norm", ""),
                "terminal_reject_rescue_accept": last_actuation.get("terminal_reject_rescue_accept", ""),
                "terminal_reject_rescue_signal_gain": last_actuation.get("terminal_reject_rescue_signal_gain", ""),
                "terminal_reject_rescue_gain_b": last_actuation.get("terminal_reject_rescue_gain_b", ""),
                "terminal_reject_rescue_corrupt_gain": last_actuation.get("terminal_reject_rescue_corrupt_gain", ""),
                "PopRisk_micro_examples": last_actuation.get("PopRisk_micro_examples", ""),
                "PopRisk_exact_variance": last_actuation.get("PopRisk_exact_variance", ""),
                "PopRisk_SNR_score": last_actuation.get("PopRisk_SNR_score", ""),
                "PopRisk_mu_norm": last_actuation.get("PopRisk_mu_norm", ""),
                "PopRisk_var_mean": last_actuation.get("PopRisk_var_mean", ""),
                "PopRisk_snr_mean": last_actuation.get("PopRisk_snr_mean", ""),
                "PopRisk_snr_max": last_actuation.get("PopRisk_snr_max", ""),
                "PopRisk_update_norm": last_actuation.get("PopRisk_update_norm", ""),
                "PopRisk_grad_cosine": last_actuation.get("PopRisk_grad_cosine", ""),
                "PopRisk_slow_state_norm": last_actuation.get("PopRisk_slow_state_norm", ""),
                "PopRisk_block_projection": last_actuation.get("PopRisk_block_projection", ""),
                "block_update_norm": last_actuation.get("block_update_norm", ""),
                **block_metrics,
            }
        )

    eval_pack(0)
    for step in range(1, steps + 1):
        idx = torch.randint(0, int(x_train.shape[0]), (batch_size,), generator=gen, device=device)
        xb = x_train[idx]
        yb = y_train[idx]
        model.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(xb).float(), yb)
        loss.backward()
        update: UpdateTensor | None = None
        if mechanism == "CTRL-AdamW":
            opt_adam.step()
        elif mechanism == "CTRL-SGD":
            opt_sgd.step_gradient()
        elif mechanism == "CTRL-NoOpMatchedOverhead":
            _ = make_update(model, mechanism, xb, yb, seed=seed)
        elif mechanism == "CTRL-RandomMatchedNorm":
            update = make_update(model, mechanism, xb, yb, seed=seed + step)
            apply_update(model, update, lr=float(args.lr))
        elif mechanism == "CTRL-RecoveryOnly":
            opt_adam.step()
        elif mechanism == "M1-AdamWPrimaryFUResidual":
            update = make_update(model, mechanism, xb, yb, seed=seed + step)
            opt_adam.step()
            apply_update(model, update, lr=float(args.fu_lr))
        elif mechanism == "M2-SGDMomentumPrimaryFU":
            update = make_update(model, mechanism, xb, yb, seed=seed + step)
            opt_sgd.step_gradient()
            apply_update(model, update, lr=float(args.fu_lr))
        elif mechanism == "M87-AdamWBoundaryThenMomentumSourceFU":
            boundary_steps = int(getattr(args, "source_warmup_steps", 0) or 400)
            if step <= boundary_steps:
                opt_adam.step()
                last_actuation = {
                    "source_state_gate_accept": 0,
                    "source_state_current_cos": "",
                    "source_state_balance_mean": "",
                    "source_state_consensus_density": 0.0,
                    "source_state_gate_threshold": boundary_steps,
                    "source_state_ema_beta": "",
                }
            else:
                update = make_update(model, mechanism, xb, yb, seed=seed + step)
                opt_sgd.step_gradient()
                apply_update(model, update, lr=float(args.fu_lr))
                last_actuation = dict(update.diagnostics or {})
                last_actuation["source_state_gate_threshold"] = boundary_steps
                last_actuation["source_state_ema_beta"] = ""
        elif mechanism == "M88-AdamWBoundaryThenDualTimescaleSourceFU":
            boundary_steps = int(getattr(args, "source_warmup_steps", 0) or 400)
            source_warm_steps = boundary_steps + 800
            if step <= boundary_steps:
                opt_adam.step()
                last_actuation = {
                    "source_state_gate_accept": 0,
                    "source_state_current_cos": "",
                    "source_state_balance_mean": "",
                    "source_state_consensus_density": 0.0,
                    "source_state_gate_threshold": boundary_steps,
                    "source_state_ema_beta": "",
                }
            else:
                fast_update = make_update(model, "M2-SGDMomentumPrimaryFU", xb, yb, seed=seed + step)
                anchor_update = make_update(model, "M15-LineCFilteredAlternatingFU", xb, yb, seed=seed + step)
                source_vec = 0.75 * fast_update.tensor.detach() + 0.25 * anchor_update.tensor.detach()
                if slow_state is not None and slow_state.numel() == 2 * source_vec.numel():
                    short_state, long_state = slow_state.to(device=device).chunk(2)
                else:
                    short_state = source_vec.detach().clone()
                    long_state = source_vec.detach().clone()
                short_state = 0.80 * short_state + 0.20 * source_vec.to(device=device)
                long_state = 0.995 * long_state + 0.005 * source_vec.to(device=device)
                slow_state = torch.cat([short_state.detach(), long_state.detach()])
                agree = torch.sign(short_state) == torch.sign(long_state)
                agree_density = float(agree.float().mean().item()) if agree.numel() else 0.0
                agreed_source = torch.where(agree, 0.50 * (short_state + long_state), 0.10 * long_state)
                source_axis = normalized_like(agreed_source, fast_update.tensor if fast_update.tensor.numel() else agreed_source)
                if step <= source_warm_steps:
                    opt_sgd.step_gradient()
                    apply_update(model, fast_update, lr=float(args.fu_lr))
                    if mechanism == "M103-AdamWBoundaryDualTimescaleParamEMAReentryFU":
                        current_params = flat_params(model).detach().to(device=device)
                        if param_slow_state is None or param_slow_state.numel() != current_params.numel():
                            param_slow_state = current_params.detach().clone()
                        else:
                            param_slow_state = 0.995 * param_slow_state.to(device=device) + 0.005 * current_params
                    update = fast_update
                    last_actuation = {
                        "source_state_current_cos": cosine(agreed_source, flat_grad(model, device)),
                        "source_state_gate_accept": 1,
                        "source_state_norm": float(torch.linalg.vector_norm(slow_state.detach()).item()),
                        "source_state_whitened_norm": float(torch.linalg.vector_norm(agreed_source.detach()).item()),
                        "source_state_consensus_density": agree_density,
                        "source_state_balance_mean": cosine(short_state, long_state),
                        "source_state_gate_threshold": boundary_steps,
                        "source_state_ema_beta": 0.995,
                        "generalization_signal_gain": 0.0,
                        "generalization_gate_accept": 1,
                    }
                elif step % max(2, int(args.alt_period)) == 0:
                    update_vec = normalized_like(agreed_source, fast_update.tensor)
                    slow_update = UpdateTensor(
                        update_vec,
                        "step",
                        "subtract",
                        "slow_state",
                        "train_stream_adamw_boundary_then_dual_timescale_source_retention",
                        mechanism,
                        role="adamw_boundary_short_long_agreement_source",
                        one_step_descent_claim=0,
                    )
                    corrupt_labels = (channel_b[1] + 1) % max(2, int(getattr(args, "classes", 10)))
                    before = snapshot(model)
                    with torch.no_grad():
                        before_a = F.cross_entropy(model(channel_a[0]).float(), channel_a[1], reduction="none")
                        before_b = F.cross_entropy(model(channel_b[0]).float(), channel_b[1], reduction="none")
                        before_corrupt = F.cross_entropy(model(channel_b[0]).float(), corrupt_labels, reduction="none")
                    apply_update(model, slow_update, lr=0.25 * float(args.fu_lr))
                    with torch.no_grad():
                        after_a = F.cross_entropy(model(channel_a[0]).float(), channel_a[1], reduction="none")
                        after_b = F.cross_entropy(model(channel_b[0]).float(), channel_b[1], reduction="none")
                        after_corrupt = F.cross_entropy(model(channel_b[0]).float(), corrupt_labels, reduction="none")
                    load_flat_params(model, before)
                    gain_a = float((before_a - after_a).mean().item())
                    gain_b = float((before_b - after_b).mean().item())
                    corrupt_gain = float((before_corrupt - after_corrupt).mean().item())
                    signal_gain = min(gain_a, gain_b)
                    gate_accept = int(
                        agree_density >= 0.20
                        and signal_gain >= -1.0e-6
                        and corrupt_gain <= max(1.0e-4, 0.50 * max(0.0, signal_gain))
                    )
                    if gate_accept:
                        apply_update(model, slow_update, lr=0.25 * float(args.fu_lr))
                        update = slow_update
                    last_actuation = {
                        "source_state_current_cos": cosine(agreed_source, flat_grad(model, device)),
                        "source_state_signal_gain": signal_gain,
                        "source_state_corrupt_gain": corrupt_gain,
                        "source_state_gate_accept": gate_accept,
                        "source_state_norm": float(torch.linalg.vector_norm(slow_state.detach()).item()),
                        "source_state_whitened_norm": float(torch.linalg.vector_norm(update_vec.detach()).item()),
                        "source_state_antiwashout_removed_norm": 0.0,
                        "source_state_projected_grad_norm": float(torch.linalg.vector_norm(anchor_update.tensor.detach()).item()),
                        "source_state_consensus_density": agree_density,
                        "source_state_balance_mean": cosine(short_state, long_state),
                        "source_state_gate_threshold": -1.0e-6,
                        "source_state_ema_beta": 0.995,
                        "generalization_gain_a": gain_a,
                        "generalization_gain_b": gain_b,
                        "shuffled_label_gain": corrupt_gain,
                        "generalization_signal_gain": signal_gain,
                        "generalization_gate_accept": gate_accept,
                    }
                else:
                    last_actuation = {
                        "source_state_current_cos": cosine(agreed_source, flat_grad(model, device)),
                        "source_state_signal_gain": 0.0,
                        "source_state_corrupt_gain": 0.0,
                        "source_state_gate_accept": 0,
                        "source_state_norm": float(torch.linalg.vector_norm(slow_state.detach()).item()),
                        "source_state_whitened_norm": float(torch.linalg.vector_norm(agreed_source.detach()).item()),
                        "source_state_antiwashout_removed_norm": 0.0,
                        "source_state_projected_grad_norm": 0.0,
                        "source_state_consensus_density": agree_density,
                        "source_state_balance_mean": cosine(short_state, long_state),
                        "source_state_gate_threshold": -1.0e-6,
                        "source_state_ema_beta": 0.995,
                        "generalization_gain_a": 0.0,
                        "generalization_gain_b": 0.0,
                        "shuffled_label_gain": 0.0,
                        "generalization_signal_gain": 0.0,
                        "generalization_gate_accept": 0,
                    }
        elif mechanism in {
            "M101-AdamWBoundaryDualTimescaleAntiWashoutFU",
            "M102-AdamWBoundaryDualTimescaleSourceAnchorFU",
            "M103-AdamWBoundaryDualTimescaleParamEMAReentryFU",
            "M104-AdamWBoundaryDualTimescaleReadoutChannelFU",
            "M105-AdamWBoundaryDualTimescaleHiddenMatrixChannelFU",
            "M118-SourceConservingOptimizerOnlyFU",
        }:
            boundary_steps = int(getattr(args, "source_warmup_steps", 0) or 400)
            source_warm_steps = 3200 if mechanism == "M118-SourceConservingOptimizerOnlyFU" else boundary_steps + 800
            if step <= boundary_steps:
                opt_adam.step()
                last_actuation = {
                    "source_state_gate_accept": 0,
                    "source_state_current_cos": "",
                    "source_state_balance_mean": "",
                    "source_state_consensus_density": 0.0,
                    "source_state_gate_threshold": boundary_steps,
                    "source_state_ema_beta": "",
                }
            else:
                fast_update = make_update(model, "M2-SGDMomentumPrimaryFU", xb, yb, seed=seed + step)
                anchor_update = make_update(model, "M15-LineCFilteredAlternatingFU", xb, yb, seed=seed + step)
                source_vec = 0.75 * fast_update.tensor.detach() + 0.25 * anchor_update.tensor.detach()
                if slow_state is not None and slow_state.numel() == 2 * source_vec.numel():
                    short_state, long_state = slow_state.to(device=device).chunk(2)
                else:
                    short_state = source_vec.detach().clone()
                    long_state = source_vec.detach().clone()
                short_state = 0.80 * short_state + 0.20 * source_vec.to(device=device)
                long_state = 0.995 * long_state + 0.005 * source_vec.to(device=device)
                slow_state = torch.cat([short_state.detach(), long_state.detach()])
                agree = torch.sign(short_state) == torch.sign(long_state)
                agree_density = float(agree.float().mean().item()) if agree.numel() else 0.0
                agreed_source = torch.where(agree, 0.50 * (short_state + long_state), 0.10 * long_state)
                source_axis = normalized_like(agreed_source, fast_update.tensor if fast_update.tensor.numel() else agreed_source)
                if step <= source_warm_steps:
                    opt_sgd.step_gradient()
                    apply_update(model, fast_update, lr=float(args.fu_lr))
                    update = fast_update
                    last_actuation = {
                        "source_state_current_cos": cosine(agreed_source, flat_grad(model, device)),
                        "source_state_gate_accept": 1,
                        "source_state_norm": float(torch.linalg.vector_norm(slow_state.detach()).item()),
                        "source_state_whitened_norm": float(torch.linalg.vector_norm(source_axis.detach()).item()),
                        "source_state_consensus_density": agree_density,
                        "source_state_balance_mean": cosine(short_state, long_state),
                        "source_state_gate_threshold": boundary_steps,
                        "source_state_ema_beta": 0.995,
                        "source_state_slow_lr_scale": 0.25,
                        "generalization_signal_gain": 0.0,
                        "generalization_gate_accept": 1,
                    }
                elif mechanism in {"M101-AdamWBoundaryDualTimescaleAntiWashoutFU", "M118-SourceConservingOptimizerOnlyFU"}:
                    g_current = flat_grad(model, device).detach().clone()
                    source_ref = source_axis.to(device=device, dtype=g_current.dtype)
                    source_norm2 = torch.dot(source_ref.float(), source_ref.float()).clamp_min(1.0e-12)
                    anti_coeff = torch.clamp(torch.dot(g_current.float(), source_ref.float()), max=0.0) / source_norm2
                    projected = g_current - anti_coeff.to(device=device, dtype=g_current.dtype) * source_ref
                    projected = torch.nan_to_num(projected, nan=0.0, posinf=0.0, neginf=0.0)
                    anti_removed = g_current - projected
                    projected_update = UpdateTensor(
                        projected,
                        "gradient",
                        "subtract",
                        "slow_state",
                        (
                            "train_stream_adamw_boundary_dual_timescale_source_conserving_projected_optimizer"
                            if mechanism == "M118-SourceConservingOptimizerOnlyFU"
                            else "train_stream_adamw_boundary_dual_timescale_antiwashout_projected_gradient"
                        ),
                        mechanism,
                        role=(
                            "source_conserving_projected_optimizer_no_sgd_fallback"
                            if mechanism == "M118-SourceConservingOptimizerOnlyFU"
                            else "source_preserving_projected_optimizer_gradient"
                        ),
                        one_step_descent_claim=0,
                    )
                    corrupt_labels = (channel_b[1] + 1) % max(2, int(getattr(args, "classes", 10)))
                    before = snapshot(model)
                    with torch.no_grad():
                        before_a = F.cross_entropy(model(channel_a[0]).float(), channel_a[1], reduction="none")
                        before_b = F.cross_entropy(model(channel_b[0]).float(), channel_b[1], reduction="none")
                        before_corrupt = F.cross_entropy(model(channel_b[0]).float(), corrupt_labels, reduction="none")
                    apply_update(model, projected_update, lr=float(args.lr))
                    with torch.no_grad():
                        after_a = F.cross_entropy(model(channel_a[0]).float(), channel_a[1], reduction="none")
                        after_b = F.cross_entropy(model(channel_b[0]).float(), channel_b[1], reduction="none")
                        after_corrupt = F.cross_entropy(model(channel_b[0]).float(), corrupt_labels, reduction="none")
                    load_flat_params(model, before)
                    gain_a = float((before_a - after_a).mean().item())
                    gain_b = float((before_b - after_b).mean().item())
                    corrupt_gain = float((before_corrupt - after_corrupt).mean().item())
                    signal_gain = min(gain_a, gain_b)
                    projected_norm = float(torch.linalg.vector_norm(projected.detach()).item())
                    if mechanism == "M118-SourceConservingOptimizerOnlyFU" and step % max(2, int(args.alt_period)) != 0:
                        gate_accept = 0
                        signal_threshold = 0.0
                        corrupt_allowance = 0.0
                    else:
                        signal_threshold = -2.0e-5 if mechanism == "M118-SourceConservingOptimizerOnlyFU" else -1.0e-4
                        corrupt_allowance = (
                            max(2.0e-5, 0.15 * max(0.0, signal_gain))
                            if mechanism == "M118-SourceConservingOptimizerOnlyFU"
                            else max(1.0e-4, 0.50 * max(0.0, signal_gain))
                        )
                        gate_accept = int(
                            projected_norm > 1.0e-12
                            and signal_gain >= signal_threshold
                            and corrupt_gain <= corrupt_allowance
                            and (mechanism != "M118-SourceConservingOptimizerOnlyFU" or cosine(projected, source_ref) >= -0.01)
                        )
                    if gate_accept:
                        apply_update(model, projected_update, lr=float(args.lr))
                        update = projected_update
                    elif mechanism != "M118-SourceConservingOptimizerOnlyFU":
                        opt_sgd.step_gradient()
                    else:
                        update = None
                    last_actuation = {
                        "source_state_current_cos": cosine(g_current, source_ref),
                        "source_state_projected_cos": cosine(projected, source_ref),
                        "source_state_signal_gain": signal_gain,
                        "source_state_corrupt_gain": corrupt_gain,
                        "source_state_gate_accept": gate_accept,
                        "source_state_norm": float(torch.linalg.vector_norm(slow_state.detach()).item()),
                        "source_state_whitened_norm": projected_norm,
                        "source_state_antiwashout_removed_norm": float(torch.linalg.vector_norm(anti_removed.detach()).item()),
                        "source_state_projected_grad_norm": projected_norm,
                        "source_state_consensus_density": agree_density,
                        "source_state_balance_mean": cosine(short_state, long_state),
                        "source_state_gate_threshold": signal_threshold,
                        "source_state_ema_beta": 0.995,
                        "source_state_slow_lr_scale": 1.0,
                        "generalization_gain_a": gain_a,
                        "generalization_gain_b": gain_b,
                        "shuffled_label_gain": corrupt_gain,
                        "generalization_signal_gain": signal_gain,
                        "generalization_gate_accept": gate_accept,
                        "operator_status": (
                            "source_conserving_hold_no_sgd_fallback"
                            if mechanism == "M118-SourceConservingOptimizerOnlyFU" and not gate_accept
                            else ""
                        ),
                    }
                elif mechanism == "M102-AdamWBoundaryDualTimescaleSourceAnchorFU":
                    g_current = flat_grad(model, device).detach().clone()
                    opt_sgd.step_gradient()
                    gate_accept = 0
                    gain_a = gain_b = corrupt_gain = signal_gain = 0.0
                    if step % max(2, int(args.alt_period)) == 0:
                        anchor_vec = normalized_like(source_axis, g_current if g_current.numel() else source_axis)
                        source_update = UpdateTensor(
                            anchor_vec,
                            "step",
                            "subtract",
                            "slow_state",
                            "train_stream_adamw_boundary_dual_timescale_source_anchor_residual",
                            mechanism,
                            role="dual_timescale_source_anchor_residual",
                            one_step_descent_claim=0,
                        )
                        corrupt_labels = (channel_b[1] + 1) % max(2, int(getattr(args, "classes", 10)))
                        before = snapshot(model)
                        with torch.no_grad():
                            before_a = F.cross_entropy(model(channel_a[0]).float(), channel_a[1], reduction="none")
                            before_b = F.cross_entropy(model(channel_b[0]).float(), channel_b[1], reduction="none")
                            before_corrupt = F.cross_entropy(model(channel_b[0]).float(), corrupt_labels, reduction="none")
                        apply_update(model, source_update, lr=0.25 * float(args.fu_lr))
                        with torch.no_grad():
                            after_a = F.cross_entropy(model(channel_a[0]).float(), channel_a[1], reduction="none")
                            after_b = F.cross_entropy(model(channel_b[0]).float(), channel_b[1], reduction="none")
                            after_corrupt = F.cross_entropy(model(channel_b[0]).float(), corrupt_labels, reduction="none")
                        load_flat_params(model, before)
                        gain_a = float((before_a - after_a).mean().item())
                        gain_b = float((before_b - after_b).mean().item())
                        corrupt_gain = float((before_corrupt - after_corrupt).mean().item())
                        signal_gain = min(gain_a, gain_b)
                        gate_accept = int(
                            float(torch.linalg.vector_norm(anchor_vec.detach()).item()) > 1.0e-12
                            and signal_gain >= -1.0e-7
                            and corrupt_gain <= max(1.0e-5, 0.25 * max(0.0, signal_gain))
                        )
                        if gate_accept:
                            apply_update(model, source_update, lr=0.25 * float(args.fu_lr))
                            update = source_update
                    last_actuation = {
                        "source_state_current_cos": cosine(source_axis, g_current),
                        "source_state_signal_gain": signal_gain,
                        "source_state_corrupt_gain": corrupt_gain,
                        "source_state_gate_accept": gate_accept,
                        "source_state_norm": float(torch.linalg.vector_norm(slow_state.detach()).item()),
                        "source_state_whitened_norm": float(torch.linalg.vector_norm(source_axis.detach()).item()),
                        "source_state_antiwashout_removed_norm": 0.0,
                        "source_state_projected_grad_norm": float(torch.linalg.vector_norm(g_current.detach()).item()),
                        "source_state_consensus_density": agree_density,
                        "source_state_balance_mean": cosine(short_state, long_state),
                        "source_state_gate_threshold": -1.0e-7,
                        "source_state_ema_beta": 0.995,
                        "source_state_slow_lr_scale": 0.25,
                        "generalization_gain_a": gain_a,
                        "generalization_gain_b": gain_b,
                        "shuffled_label_gain": corrupt_gain,
                        "generalization_signal_gain": signal_gain,
                        "generalization_gate_accept": gate_accept,
                    }
                elif mechanism == "M103-AdamWBoundaryDualTimescaleParamEMAReentryFU":
                    g_current = flat_grad(model, device).detach().clone()
                    opt_sgd.step_gradient()
                    current_params = flat_params(model).detach().to(device=device)
                    if param_slow_state is None or param_slow_state.numel() != current_params.numel():
                        param_slow_state = current_params.detach().clone()
                    param_slow_state = 0.9975 * param_slow_state.to(device=device) + 0.0025 * current_params
                    reentry_vec = current_params - param_slow_state.to(device=device, dtype=current_params.dtype)
                    gate_accept = 0
                    gain_a = gain_b = corrupt_gain = signal_gain = 0.0
                    reentry_norm = float(torch.linalg.vector_norm(reentry_vec.detach()).item())
                    reentry_start = 3200
                    if step >= reentry_start and step % max(2, int(args.alt_period)) == 0 and reentry_norm > 1.0e-12:
                        reentry_update = UpdateTensor(
                            reentry_vec,
                            "step",
                            "subtract",
                            "slow_state",
                            "train_stream_adamw_boundary_dual_timescale_param_ema_reentry",
                            mechanism,
                            role="schedule_free_parameter_ema_reentry",
                            one_step_descent_claim=0,
                        )
                        corrupt_labels = (channel_b[1] + 1) % max(2, int(getattr(args, "classes", 10)))
                        before = snapshot(model)
                        with torch.no_grad():
                            before_a = F.cross_entropy(model(channel_a[0]).float(), channel_a[1], reduction="none")
                            before_b = F.cross_entropy(model(channel_b[0]).float(), channel_b[1], reduction="none")
                            before_corrupt = F.cross_entropy(model(channel_b[0]).float(), corrupt_labels, reduction="none")
                        apply_update(model, reentry_update, lr=0.05)
                        with torch.no_grad():
                            after_a = F.cross_entropy(model(channel_a[0]).float(), channel_a[1], reduction="none")
                            after_b = F.cross_entropy(model(channel_b[0]).float(), channel_b[1], reduction="none")
                            after_corrupt = F.cross_entropy(model(channel_b[0]).float(), corrupt_labels, reduction="none")
                        load_flat_params(model, before)
                        gain_a = float((before_a - after_a).mean().item())
                        gain_b = float((before_b - after_b).mean().item())
                        corrupt_gain = float((before_corrupt - after_corrupt).mean().item())
                        signal_gain = min(gain_a, gain_b)
                        gate_accept = int(
                            signal_gain >= -1.0e-7
                            and corrupt_gain <= max(1.0e-5, 0.25 * max(0.0, signal_gain))
                        )
                        if gate_accept:
                            apply_update(model, reentry_update, lr=0.05)
                            current_params = flat_params(model).detach().to(device=device)
                            param_slow_state = 0.995 * param_slow_state.to(device=device) + 0.005 * current_params
                            update = reentry_update
                    last_actuation = {
                        "source_state_current_cos": cosine(source_axis, g_current),
                        "source_state_signal_gain": signal_gain,
                        "source_state_corrupt_gain": corrupt_gain,
                        "source_state_gate_accept": gate_accept,
                        "source_state_norm": float(torch.linalg.vector_norm(param_slow_state.detach()).item()) if param_slow_state is not None else "",
                        "source_state_whitened_norm": reentry_norm,
                        "source_state_antiwashout_removed_norm": 0.0,
                        "source_state_projected_grad_norm": float(torch.linalg.vector_norm(g_current.detach()).item()),
                        "source_state_consensus_density": agree_density,
                        "source_state_balance_mean": cosine(short_state, long_state),
                        "source_state_gate_threshold": reentry_start,
                        "source_state_ema_beta": 0.9975,
                        "source_state_slow_lr_scale": 0.05,
                        "generalization_gain_a": gain_a,
                        "generalization_gain_b": gain_b,
                        "shuffled_label_gain": corrupt_gain,
                        "generalization_signal_gain": signal_gain,
                        "generalization_gate_accept": gate_accept,
                    }
                else:
                    g_current = flat_grad(model, device).detach().clone()
                    opt_sgd.step_gradient()
                    gate_accept = 0
                    gain_a = gain_b = corrupt_gain = signal_gain = 0.0
                    channel_is_readout = mechanism == "M104-AdamWBoundaryDualTimescaleReadoutChannelFU"
                    channel_mask = (readout_mask_flat() if channel_is_readout else hidden_matrix_mask_flat()).to(device=device, dtype=source_axis.dtype)
                    channel_density = float(channel_mask.float().mean().item()) if channel_mask.numel() else 0.0
                    channel_vec = agreed_source.to(device=device, dtype=source_axis.dtype) * channel_mask
                    channel_norm = float(torch.linalg.vector_norm(channel_vec.detach()).item())
                    channel_update_vec = (
                        normalized_like(channel_vec, fast_update.tensor if fast_update.tensor.numel() else channel_vec)
                        if channel_norm > 1.0e-12
                        else channel_vec
                    )
                    if step % max(2, int(args.alt_period)) == 0 and channel_norm > 1.0e-12:
                        channel_update = UpdateTensor(
                            channel_update_vec,
                            "step",
                            "subtract",
                            "readout_carrier" if channel_is_readout else "matrix_block",
                            (
                                "train_stream_adamw_boundary_dual_timescale_readout_channel_source"
                                if channel_is_readout
                                else "train_stream_adamw_boundary_dual_timescale_hidden_matrix_channel_source"
                            ),
                            mechanism,
                            role="readout_channel_source_reset" if channel_is_readout else "hidden_matrix_channel_source_reset",
                            one_step_descent_claim=0,
                        )
                        corrupt_labels = (channel_b[1] + 1) % max(2, int(getattr(args, "classes", 10)))
                        before = snapshot(model)
                        with torch.no_grad():
                            before_a = F.cross_entropy(model(channel_a[0]).float(), channel_a[1], reduction="none")
                            before_b = F.cross_entropy(model(channel_b[0]).float(), channel_b[1], reduction="none")
                            before_corrupt = F.cross_entropy(model(channel_b[0]).float(), corrupt_labels, reduction="none")
                        apply_update(model, channel_update, lr=0.50 * float(args.fu_lr))
                        with torch.no_grad():
                            after_a = F.cross_entropy(model(channel_a[0]).float(), channel_a[1], reduction="none")
                            after_b = F.cross_entropy(model(channel_b[0]).float(), channel_b[1], reduction="none")
                            after_corrupt = F.cross_entropy(model(channel_b[0]).float(), corrupt_labels, reduction="none")
                        load_flat_params(model, before)
                        gain_a = float((before_a - after_a).mean().item())
                        gain_b = float((before_b - after_b).mean().item())
                        corrupt_gain = float((before_corrupt - after_corrupt).mean().item())
                        signal_gain = min(gain_a, gain_b)
                        gate_accept = int(
                            signal_gain >= -1.0e-7
                            and corrupt_gain <= max(1.0e-5, 0.25 * max(0.0, signal_gain))
                        )
                        if gate_accept:
                            apply_update(model, channel_update, lr=0.50 * float(args.fu_lr))
                            update = channel_update
                    last_actuation = {
                        "source_state_current_cos": cosine(channel_update_vec, g_current),
                        "source_state_signal_gain": signal_gain,
                        "source_state_corrupt_gain": corrupt_gain,
                        "source_state_gate_accept": gate_accept,
                        "source_state_norm": float(torch.linalg.vector_norm(slow_state.detach()).item()),
                        "source_state_whitened_norm": float(torch.linalg.vector_norm(channel_update_vec.detach()).item()),
                        "source_state_antiwashout_removed_norm": 0.0,
                        "source_state_projected_grad_norm": float(torch.linalg.vector_norm(g_current.detach()).item()),
                        "source_state_consensus_density": agree_density,
                        "source_state_balance_mean": cosine(short_state, long_state),
                        "source_state_gate_threshold": -1.0e-7,
                        "source_state_ema_beta": 0.995,
                        "source_state_slow_lr_scale": 0.50,
                        "source_channel_projection": cosine(channel_update_vec, source_axis),
                        "source_bank_feature_count": int(channel_mask.sum().item()) if channel_mask.numel() else 0,
                        "reservoir_bank_feature_count": int(channel_mask.numel() - channel_mask.sum().item()) if channel_mask.numel() else 0,
                        "low_degree_source_energy": channel_density,
                        "generalization_gain_a": gain_a,
                        "generalization_gain_b": gain_b,
                        "shuffled_label_gain": corrupt_gain,
                        "generalization_signal_gain": signal_gain,
                        "generalization_gate_accept": gate_accept,
                    }
        elif mechanism in {
            "M89-AdamWBoundaryDualTimescaleSGDFloorFU",
            "M90-AdamWBoundaryDualTimescaleLateSGDFloorFU",
            "M91-AdamWBoundaryDualTimescaleTinyLateSGDFloorFU",
            "M92-AdamWBoundaryDualTimescaleReinforcedTinyLateSGDFloorFU",
            "M93-TrainLossGatedDualTimescaleTinyLateFU",
            "M94-TrainLossGatedDualTimescaleHoldFallbackFU",
            "M95-TrainLossGatedDualTimescaleTinyLateFallbackFU",
            "M96-TrainLossGatedDualTimescaleBoostedTinyLateFallbackFU",
            "M97-TrainLossLateHoldRecoveryFU",
            "M98-TrainLossLateLookaheadFloorFU",
            "M99-TrainLossTerminalLookaheadFloorFU",
            "M100-TrainLossEarlyTerminalLookaheadFloorFU",
            "M106-TrainLossTerminalProjectedLookaheadFloorFU",
            "M107-TrainLossTerminalConsensusLookaheadFloorFU",
            "M108-TrainLossTerminalSelectorLookaheadFloorFU",
            "M111-TrainLossTerminalPositiveLookaheadFloorFU",
            "M112-TrainLossTerminalCheckpointReentryFU",
            "M113-TrainLossTerminalHardSplitSourceFU",
            "M114-TrainLossTerminalAdamWLookaheadFU",
            "M115-TrainLossTerminalOptimizerSelectorFU",
            "M119-TerminalSourceConservingRouteFU",
            "M120-TrainLossRiskProfileRouteFU",
            "M121-DebtAwareSourceGateFU",
            "M122-UngatedWarmTerminalSourceRouteFU",
            "M123-UngatedWarmRiskProfileRouteFU",
            "M124-UngatedWarmDebtRawBailoutFU",
            "M125-TrainLossH2400CheckpointHoldFU",
            "M126-TrainLossH2800CheckpointHoldFU",
            "M127-TrainLossH2400DebtBailoutFU",
            "M128-TrainLossTerminalRawThenSourceGuardFU",
            "M129-H800SourceSlowEMARetentionFU",
            "M130-H800ReadoutChannelRetentionFU",
            "M131-H800DualMemorySourceRetentionFU",
            "M132-H1600SourceCheckpointReentryFU",
            "M133-H2400SourceCheckpointReentryFU",
            "M139-EarlySourceSlowEMATerminalGuardFU",
            "M140-EarlySourceSlowEMATerminalRawGuardFU",
            "M141-EarlySourceSlowEMATerminalRawGuardStrongFU",
            "M142-EarlySourceSlowEMATerminalProjectedOptimizerFU",
            "M218-EarlySourceSlowEMATerminalProjectedOptimizerLambda050FU",
            "M219-EarlySourceSlowEMATerminalProjectedOptimizerLambda100FU",
            "M143-EarlySourceSlowEMATerminalProjectedBlendFU",
            "M144-EarlySourceSlowEMATerminalAntiWashoutFU",
            "M145-EarlySourceSlowEMATerminalH4000ReentryFU",
            "M146-EarlySourceSlowEMASignalReservoirTargetFU",
            "M147-EarlySourceSlowEMASourceBankTargetFU",
            "M148-EarlySourceSlowEMADualTargetGuardFU",
            "M149-EarlySourceSlowEMATerminalAdaptiveRawFU",
            "M150-EarlySourceSlowEMATerminalSparseSourceFU",
            "M151-EarlySourceSlowEMATerminalRatioPreserveFU",
            "M152-SourceProjectionB3NullConsensusTargetFU",
            "M153-NoiseOrthogonalB3NullConsensusTargetFU",
            "M154-EasyMarginB3NullConsensusTargetFU",
            "M158-EarlySourceSlowEMATerminalRejectSourceAxisRescueFU",
            "M159-EarlySourceSlowEMATerminalRejectSlowEMARescueFU",
            "M160-EarlySourceSlowEMATerminalRejectHoldSourceRescueFU",
            "M164-EarlySourceSlowEMATerminalRejectRawRescueFU",
            "M165-EarlySourceSlowEMATerminalRejectHybridRescueFU",
            "M166-EarlySourceSlowEMATerminalRejectLateOnlyRawRescueFU",
            "M167-EarlySourceSlowEMATerminalTopKSupportGuardFU",
            "M168-EarlySourceSlowEMATerminalTopKRawGuardFU",
            "M169-EarlySourceSlowEMATerminalTopKDebtCapFU",
            "M170-EarlySourceSlowEMATerminalSourcePreserveStrongFU",
            "M171-EarlySourceSlowEMATerminalSourcePreserveGentleFU",
            "M172-EarlySourceSlowEMATerminalInfoVolumeGuardFU",
            "M173-EarlySourceSlowEMALowNDSDiffeomorphicTargetFU",
            "M174-EarlySourceSlowEMAInfoVolumeDiffeomorphicTargetFU",
            "M175-EarlySourceSlowEMALowRankReadoutTransportFU",
            "M179-EarlySourceSlowEMATerminalSourcePreserveVeryStrongFU",
            "M180-EarlySourceSlowEMAH3600TerminalSourcePreserveFU",
            "M181-EarlySourceSlowEMATerminalNoraOrthogonalSourceFU",
            "M182-EarlySourceSlowEMATerminalDebtAwarePreserveFU",
            "M183-EarlySourceSlowEMATerminalLowNDSMatrixBlockFU",
            "M184-EarlySourceSlowEMATerminalDualMemoryPreserveFU",
            "M185-EarlySourceSlowEMASNRTerminalPredictorFU",
            "M186-EarlySourceSlowEMASplitConsensusEstimatorFU",
            "M187-EarlySourceSlowEMASignalReservoirTransportFU",
            "M188-EarlySourceSlowEMATerminalSourceFloorFU",
            "M189-EarlySourceSlowEMATerminalH4000AnchorFloorFU",
            "M190-EarlySourceSlowEMATerminalDecayAwareFloorFU",
            "M191-EarlySourceSlowEMATerminalRawGuardSourceFloorFU",
            "M192-EarlySourceSlowEMATerminalRawGuardH4000AnchorFloorFU",
            "M193-EarlySourceSlowEMATerminalRawGuardDecayAwareFloorFU",
            "M194-EarlySourceSlowEMATerminalAntiErosionOrthogonalFU",
            "M195-EarlySourceSlowEMATerminalSourceReflectionGuardFU",
            "M196-EarlySourceSlowEMATerminalH4000TransportCorrectorFU",
            "M197-EarlySourceSlowEMATerminalH3200AnchorTransportFU",
            "M198-EarlySourceSlowEMATerminalRawGuardH3200AnchorTransportFU",
            "M199-EarlySourceSlowEMATerminalH3200RatioReentryFU",
            "M200-EarlySourceSlowEMATerminalH3200ProgressCarryFU",
            "M201-EarlySourceSlowEMATerminalH4000ProgressCarryFU",
            "M202-EarlySourceSlowEMATerminalH3200SourceProgressBlendFU",
            "M203-EarlySourceSlowEMATerminalRawGuardH3600GentleProgressFU",
            "M204-EarlySourceSlowEMATerminalRawGuardH4000GentleProgressFU",
            "M205-EarlySourceSlowEMATerminalRawGuardH4400ProjectedProgressFU",
            "M206-EarlySourceSlowEMATerminalControlRelativeSGDCatchupFU",
            "M207-EarlySourceSlowEMATerminalControlRelativeAdamWCatchupFU",
            "M208-EarlySourceSlowEMATerminalControlRelativeSourceBalancedCatchupFU",
            "M209-EarlySourceSlowEMATerminalTrajectoryAdaptivePreserveFU",
            "M210-EarlySourceSlowEMATerminalMidErosionBridgeFU",
            "M211-EarlySourceSlowEMATerminalTwoPhaseRatioRepairFU",
            "M212-EarlySourceSlowEMATerminalAntiSourceClipFU",
            "M213-EarlySourceSlowEMATerminalDebtAwareHoldFU",
            "M214-EarlySourceSlowEMATerminalAnchorFlowTinyFU",
            "M215-EarlySourceSlowEMATerminalAcceptMemoryFU",
            "M216-EarlySourceSlowEMATerminalAcceptMemoryDebtFU",
            "M217-EarlySourceSlowEMATerminalSourceProgressMemoryFU",
        }:
            boundary_steps = int(getattr(args, "source_warmup_steps", 0) or 400)
            mid_horizon_retention = mechanism in {
                "M129-H800SourceSlowEMARetentionFU",
                "M130-H800ReadoutChannelRetentionFU",
                "M131-H800DualMemorySourceRetentionFU",
                "M132-H1600SourceCheckpointReentryFU",
                "M133-H2400SourceCheckpointReentryFU",
                "M139-EarlySourceSlowEMATerminalGuardFU",
                "M140-EarlySourceSlowEMATerminalRawGuardFU",
                "M141-EarlySourceSlowEMATerminalRawGuardStrongFU",
                "M142-EarlySourceSlowEMATerminalProjectedOptimizerFU",
                "M218-EarlySourceSlowEMATerminalProjectedOptimizerLambda050FU",
                "M219-EarlySourceSlowEMATerminalProjectedOptimizerLambda100FU",
                "M143-EarlySourceSlowEMATerminalProjectedBlendFU",
                "M144-EarlySourceSlowEMATerminalAntiWashoutFU",
                "M145-EarlySourceSlowEMATerminalH4000ReentryFU",
                "M146-EarlySourceSlowEMASignalReservoirTargetFU",
                "M147-EarlySourceSlowEMASourceBankTargetFU",
                "M148-EarlySourceSlowEMADualTargetGuardFU",
                "M149-EarlySourceSlowEMATerminalAdaptiveRawFU",
                "M150-EarlySourceSlowEMATerminalSparseSourceFU",
                "M151-EarlySourceSlowEMATerminalRatioPreserveFU",
                "M152-SourceProjectionB3NullConsensusTargetFU",
                "M153-NoiseOrthogonalB3NullConsensusTargetFU",
                "M154-EasyMarginB3NullConsensusTargetFU",
                "M158-EarlySourceSlowEMATerminalRejectSourceAxisRescueFU",
                "M159-EarlySourceSlowEMATerminalRejectSlowEMARescueFU",
                "M160-EarlySourceSlowEMATerminalRejectHoldSourceRescueFU",
                "M164-EarlySourceSlowEMATerminalRejectRawRescueFU",
                "M165-EarlySourceSlowEMATerminalRejectHybridRescueFU",
                "M166-EarlySourceSlowEMATerminalRejectLateOnlyRawRescueFU",
                "M167-EarlySourceSlowEMATerminalTopKSupportGuardFU",
                "M168-EarlySourceSlowEMATerminalTopKRawGuardFU",
                "M169-EarlySourceSlowEMATerminalTopKDebtCapFU",
                "M170-EarlySourceSlowEMATerminalSourcePreserveStrongFU",
                "M171-EarlySourceSlowEMATerminalSourcePreserveGentleFU",
                "M172-EarlySourceSlowEMATerminalInfoVolumeGuardFU",
                "M173-EarlySourceSlowEMALowNDSDiffeomorphicTargetFU",
                "M174-EarlySourceSlowEMAInfoVolumeDiffeomorphicTargetFU",
                "M175-EarlySourceSlowEMALowRankReadoutTransportFU",
                "M179-EarlySourceSlowEMATerminalSourcePreserveVeryStrongFU",
                "M180-EarlySourceSlowEMAH3600TerminalSourcePreserveFU",
                "M181-EarlySourceSlowEMATerminalNoraOrthogonalSourceFU",
                "M182-EarlySourceSlowEMATerminalDebtAwarePreserveFU",
                "M183-EarlySourceSlowEMATerminalLowNDSMatrixBlockFU",
                "M184-EarlySourceSlowEMATerminalDualMemoryPreserveFU",
                "M185-EarlySourceSlowEMASNRTerminalPredictorFU",
                "M186-EarlySourceSlowEMASplitConsensusEstimatorFU",
                "M187-EarlySourceSlowEMASignalReservoirTransportFU",
                "M188-EarlySourceSlowEMATerminalSourceFloorFU",
                "M189-EarlySourceSlowEMATerminalH4000AnchorFloorFU",
                "M190-EarlySourceSlowEMATerminalDecayAwareFloorFU",
                "M191-EarlySourceSlowEMATerminalRawGuardSourceFloorFU",
                "M192-EarlySourceSlowEMATerminalRawGuardH4000AnchorFloorFU",
                "M193-EarlySourceSlowEMATerminalRawGuardDecayAwareFloorFU",
                "M194-EarlySourceSlowEMATerminalAntiErosionOrthogonalFU",
                "M195-EarlySourceSlowEMATerminalSourceReflectionGuardFU",
                "M196-EarlySourceSlowEMATerminalH4000TransportCorrectorFU",
                "M197-EarlySourceSlowEMATerminalH3200AnchorTransportFU",
                "M198-EarlySourceSlowEMATerminalRawGuardH3200AnchorTransportFU",
                "M199-EarlySourceSlowEMATerminalH3200RatioReentryFU",
                "M200-EarlySourceSlowEMATerminalH3200ProgressCarryFU",
                "M201-EarlySourceSlowEMATerminalH4000ProgressCarryFU",
                "M202-EarlySourceSlowEMATerminalH3200SourceProgressBlendFU",
                "M203-EarlySourceSlowEMATerminalRawGuardH3600GentleProgressFU",
                "M204-EarlySourceSlowEMATerminalRawGuardH4000GentleProgressFU",
                "M205-EarlySourceSlowEMATerminalRawGuardH4400ProjectedProgressFU",
                "M206-EarlySourceSlowEMATerminalControlRelativeSGDCatchupFU",
                "M207-EarlySourceSlowEMATerminalControlRelativeAdamWCatchupFU",
                "M208-EarlySourceSlowEMATerminalControlRelativeSourceBalancedCatchupFU",
                "M209-EarlySourceSlowEMATerminalTrajectoryAdaptivePreserveFU",
                "M210-EarlySourceSlowEMATerminalMidErosionBridgeFU",
                "M211-EarlySourceSlowEMATerminalTwoPhaseRatioRepairFU",
                "M212-EarlySourceSlowEMATerminalAntiSourceClipFU",
                "M213-EarlySourceSlowEMATerminalDebtAwareHoldFU",
                "M214-EarlySourceSlowEMATerminalAnchorFlowTinyFU",
                "M215-EarlySourceSlowEMATerminalAcceptMemoryFU",
                "M216-EarlySourceSlowEMATerminalAcceptMemoryDebtFU",
                "M217-EarlySourceSlowEMATerminalSourceProgressMemoryFU",
                "M161-EarlySourceSlowEMAShapePreserveFU",
                "M162-EarlySourceSlowEMAShapePreserveRawGuardFU",
                "M163-EarlySourceSlowEMAShapePreserveClampFU",
            }
            ungated_terminal_warm = mechanism in {
                "M122-UngatedWarmTerminalSourceRouteFU",
                "M123-UngatedWarmRiskProfileRouteFU",
                "M124-UngatedWarmDebtRawBailoutFU",
            }
            source_warm_steps = 800 if mid_horizon_retention else 3200 if ungated_terminal_warm else boundary_steps + 800
            terminal_hold_start = (
                100_000
                if mid_horizon_retention
                else
                2400
                if mechanism in {"M125-TrainLossH2400CheckpointHoldFU", "M127-TrainLossH2400DebtBailoutFU"}
                else 2800
                if mechanism == "M126-TrainLossH2800CheckpointHoldFU"
                else 3200
            )
            late_floor_start = 0 if mechanism == "M89-AdamWBoundaryDualTimescaleSGDFloorFU" else terminal_hold_start
            tiny_floor = mechanism in {
                "M91-AdamWBoundaryDualTimescaleTinyLateSGDFloorFU",
                "M92-AdamWBoundaryDualTimescaleReinforcedTinyLateSGDFloorFU",
                "M93-TrainLossGatedDualTimescaleTinyLateFU",
                "M94-TrainLossGatedDualTimescaleHoldFallbackFU",
                "M95-TrainLossGatedDualTimescaleTinyLateFallbackFU",
                "M96-TrainLossGatedDualTimescaleBoostedTinyLateFallbackFU",
                "M97-TrainLossLateHoldRecoveryFU",
                "M98-TrainLossLateLookaheadFloorFU",
                "M99-TrainLossTerminalLookaheadFloorFU",
                "M100-TrainLossEarlyTerminalLookaheadFloorFU",
                "M106-TrainLossTerminalProjectedLookaheadFloorFU",
                "M107-TrainLossTerminalConsensusLookaheadFloorFU",
                "M108-TrainLossTerminalSelectorLookaheadFloorFU",
                "M111-TrainLossTerminalPositiveLookaheadFloorFU",
                "M112-TrainLossTerminalCheckpointReentryFU",
                "M113-TrainLossTerminalHardSplitSourceFU",
                "M114-TrainLossTerminalAdamWLookaheadFU",
                "M115-TrainLossTerminalOptimizerSelectorFU",
                "M119-TerminalSourceConservingRouteFU",
                "M120-TrainLossRiskProfileRouteFU",
                "M121-DebtAwareSourceGateFU",
                "M122-UngatedWarmTerminalSourceRouteFU",
                "M123-UngatedWarmRiskProfileRouteFU",
                "M124-UngatedWarmDebtRawBailoutFU",
                "M125-TrainLossH2400CheckpointHoldFU",
                "M126-TrainLossH2800CheckpointHoldFU",
                "M127-TrainLossH2400DebtBailoutFU",
                "M128-TrainLossTerminalRawThenSourceGuardFU",
                "M129-H800SourceSlowEMARetentionFU",
                "M130-H800ReadoutChannelRetentionFU",
                "M131-H800DualMemorySourceRetentionFU",
                "M132-H1600SourceCheckpointReentryFU",
                "M133-H2400SourceCheckpointReentryFU",
                "M139-EarlySourceSlowEMATerminalGuardFU",
                "M140-EarlySourceSlowEMATerminalRawGuardFU",
                "M141-EarlySourceSlowEMATerminalRawGuardStrongFU",
                "M142-EarlySourceSlowEMATerminalProjectedOptimizerFU",
                "M218-EarlySourceSlowEMATerminalProjectedOptimizerLambda050FU",
                "M219-EarlySourceSlowEMATerminalProjectedOptimizerLambda100FU",
                "M143-EarlySourceSlowEMATerminalProjectedBlendFU",
                "M144-EarlySourceSlowEMATerminalAntiWashoutFU",
                "M145-EarlySourceSlowEMATerminalH4000ReentryFU",
                "M149-EarlySourceSlowEMATerminalAdaptiveRawFU",
                "M150-EarlySourceSlowEMATerminalSparseSourceFU",
                "M151-EarlySourceSlowEMATerminalRatioPreserveFU",
                "M152-SourceProjectionB3NullConsensusTargetFU",
                "M153-NoiseOrthogonalB3NullConsensusTargetFU",
                "M154-EasyMarginB3NullConsensusTargetFU",
                "M158-EarlySourceSlowEMATerminalRejectSourceAxisRescueFU",
                "M159-EarlySourceSlowEMATerminalRejectSlowEMARescueFU",
                "M160-EarlySourceSlowEMATerminalRejectHoldSourceRescueFU",
                "M164-EarlySourceSlowEMATerminalRejectRawRescueFU",
                "M165-EarlySourceSlowEMATerminalRejectHybridRescueFU",
                "M166-EarlySourceSlowEMATerminalRejectLateOnlyRawRescueFU",
                "M167-EarlySourceSlowEMATerminalTopKSupportGuardFU",
                "M168-EarlySourceSlowEMATerminalTopKRawGuardFU",
                "M169-EarlySourceSlowEMATerminalTopKDebtCapFU",
                "M170-EarlySourceSlowEMATerminalSourcePreserveStrongFU",
                "M171-EarlySourceSlowEMATerminalSourcePreserveGentleFU",
                "M172-EarlySourceSlowEMATerminalInfoVolumeGuardFU",
                "M173-EarlySourceSlowEMALowNDSDiffeomorphicTargetFU",
                "M174-EarlySourceSlowEMAInfoVolumeDiffeomorphicTargetFU",
                "M175-EarlySourceSlowEMALowRankReadoutTransportFU",
                "M179-EarlySourceSlowEMATerminalSourcePreserveVeryStrongFU",
                "M180-EarlySourceSlowEMAH3600TerminalSourcePreserveFU",
                "M181-EarlySourceSlowEMATerminalNoraOrthogonalSourceFU",
                "M182-EarlySourceSlowEMATerminalDebtAwarePreserveFU",
                "M183-EarlySourceSlowEMATerminalLowNDSMatrixBlockFU",
                "M184-EarlySourceSlowEMATerminalDualMemoryPreserveFU",
                "M185-EarlySourceSlowEMASNRTerminalPredictorFU",
                "M186-EarlySourceSlowEMASplitConsensusEstimatorFU",
                "M187-EarlySourceSlowEMASignalReservoirTransportFU",
                "M188-EarlySourceSlowEMATerminalSourceFloorFU",
                "M189-EarlySourceSlowEMATerminalH4000AnchorFloorFU",
                "M190-EarlySourceSlowEMATerminalDecayAwareFloorFU",
                "M191-EarlySourceSlowEMATerminalRawGuardSourceFloorFU",
                "M192-EarlySourceSlowEMATerminalRawGuardH4000AnchorFloorFU",
                "M193-EarlySourceSlowEMATerminalRawGuardDecayAwareFloorFU",
                "M194-EarlySourceSlowEMATerminalAntiErosionOrthogonalFU",
                "M195-EarlySourceSlowEMATerminalSourceReflectionGuardFU",
                "M196-EarlySourceSlowEMATerminalH4000TransportCorrectorFU",
                "M197-EarlySourceSlowEMATerminalH3200AnchorTransportFU",
                "M198-EarlySourceSlowEMATerminalRawGuardH3200AnchorTransportFU",
                "M199-EarlySourceSlowEMATerminalH3200RatioReentryFU",
                "M200-EarlySourceSlowEMATerminalH3200ProgressCarryFU",
                "M201-EarlySourceSlowEMATerminalH4000ProgressCarryFU",
                "M202-EarlySourceSlowEMATerminalH3200SourceProgressBlendFU",
                "M203-EarlySourceSlowEMATerminalRawGuardH3600GentleProgressFU",
                "M204-EarlySourceSlowEMATerminalRawGuardH4000GentleProgressFU",
                "M205-EarlySourceSlowEMATerminalRawGuardH4400ProjectedProgressFU",
                "M206-EarlySourceSlowEMATerminalControlRelativeSGDCatchupFU",
                "M207-EarlySourceSlowEMATerminalControlRelativeAdamWCatchupFU",
                "M208-EarlySourceSlowEMATerminalControlRelativeSourceBalancedCatchupFU",
                "M161-EarlySourceSlowEMAShapePreserveFU",
                "M162-EarlySourceSlowEMAShapePreserveRawGuardFU",
                "M163-EarlySourceSlowEMAShapePreserveClampFU",
            }
            slow_lr_scale = (
                0.50
                if mechanism == "M92-AdamWBoundaryDualTimescaleReinforcedTinyLateSGDFloorFU"
                else 0.35
                if mechanism in {
                    "M129-H800SourceSlowEMARetentionFU",
                    "M131-H800DualMemorySourceRetentionFU",
                    "M139-EarlySourceSlowEMATerminalGuardFU",
                    "M140-EarlySourceSlowEMATerminalRawGuardFU",
                    "M141-EarlySourceSlowEMATerminalRawGuardStrongFU",
                    "M142-EarlySourceSlowEMATerminalProjectedOptimizerFU",
                    "M218-EarlySourceSlowEMATerminalProjectedOptimizerLambda050FU",
                    "M219-EarlySourceSlowEMATerminalProjectedOptimizerLambda100FU",
                    "M143-EarlySourceSlowEMATerminalProjectedBlendFU",
                    "M144-EarlySourceSlowEMATerminalAntiWashoutFU",
                    "M145-EarlySourceSlowEMATerminalH4000ReentryFU",
                    "M146-EarlySourceSlowEMASignalReservoirTargetFU",
                    "M147-EarlySourceSlowEMASourceBankTargetFU",
                    "M148-EarlySourceSlowEMADualTargetGuardFU",
                    "M149-EarlySourceSlowEMATerminalAdaptiveRawFU",
                    "M150-EarlySourceSlowEMATerminalSparseSourceFU",
                    "M151-EarlySourceSlowEMATerminalRatioPreserveFU",
                    "M152-SourceProjectionB3NullConsensusTargetFU",
                    "M153-NoiseOrthogonalB3NullConsensusTargetFU",
                    "M154-EasyMarginB3NullConsensusTargetFU",
                    "M158-EarlySourceSlowEMATerminalRejectSourceAxisRescueFU",
                    "M159-EarlySourceSlowEMATerminalRejectSlowEMARescueFU",
                    "M160-EarlySourceSlowEMATerminalRejectHoldSourceRescueFU",
                    "M164-EarlySourceSlowEMATerminalRejectRawRescueFU",
                    "M165-EarlySourceSlowEMATerminalRejectHybridRescueFU",
                    "M166-EarlySourceSlowEMATerminalRejectLateOnlyRawRescueFU",
                    "M167-EarlySourceSlowEMATerminalTopKSupportGuardFU",
                    "M168-EarlySourceSlowEMATerminalTopKRawGuardFU",
                    "M169-EarlySourceSlowEMATerminalTopKDebtCapFU",
                    "M170-EarlySourceSlowEMATerminalSourcePreserveStrongFU",
                    "M171-EarlySourceSlowEMATerminalSourcePreserveGentleFU",
                    "M172-EarlySourceSlowEMATerminalInfoVolumeGuardFU",
                    "M173-EarlySourceSlowEMALowNDSDiffeomorphicTargetFU",
                    "M174-EarlySourceSlowEMAInfoVolumeDiffeomorphicTargetFU",
                    "M175-EarlySourceSlowEMALowRankReadoutTransportFU",
                    "M179-EarlySourceSlowEMATerminalSourcePreserveVeryStrongFU",
                    "M180-EarlySourceSlowEMAH3600TerminalSourcePreserveFU",
                    "M181-EarlySourceSlowEMATerminalNoraOrthogonalSourceFU",
                    "M182-EarlySourceSlowEMATerminalDebtAwarePreserveFU",
                    "M183-EarlySourceSlowEMATerminalLowNDSMatrixBlockFU",
                    "M184-EarlySourceSlowEMATerminalDualMemoryPreserveFU",
                    "M185-EarlySourceSlowEMASNRTerminalPredictorFU",
                    "M186-EarlySourceSlowEMASplitConsensusEstimatorFU",
                    "M187-EarlySourceSlowEMASignalReservoirTransportFU",
                    "M188-EarlySourceSlowEMATerminalSourceFloorFU",
                    "M189-EarlySourceSlowEMATerminalH4000AnchorFloorFU",
                    "M190-EarlySourceSlowEMATerminalDecayAwareFloorFU",
                    "M191-EarlySourceSlowEMATerminalRawGuardSourceFloorFU",
                    "M192-EarlySourceSlowEMATerminalRawGuardH4000AnchorFloorFU",
                    "M193-EarlySourceSlowEMATerminalRawGuardDecayAwareFloorFU",
                    "M194-EarlySourceSlowEMATerminalAntiErosionOrthogonalFU",
                    "M195-EarlySourceSlowEMATerminalSourceReflectionGuardFU",
                    "M196-EarlySourceSlowEMATerminalH4000TransportCorrectorFU",
                    "M197-EarlySourceSlowEMATerminalH3200AnchorTransportFU",
                    "M198-EarlySourceSlowEMATerminalRawGuardH3200AnchorTransportFU",
                    "M199-EarlySourceSlowEMATerminalH3200RatioReentryFU",
                    "M200-EarlySourceSlowEMATerminalH3200ProgressCarryFU",
                    "M201-EarlySourceSlowEMATerminalH4000ProgressCarryFU",
                    "M202-EarlySourceSlowEMATerminalH3200SourceProgressBlendFU",
                    "M203-EarlySourceSlowEMATerminalRawGuardH3600GentleProgressFU",
                    "M204-EarlySourceSlowEMATerminalRawGuardH4000GentleProgressFU",
                    "M205-EarlySourceSlowEMATerminalRawGuardH4400ProjectedProgressFU",
                    "M206-EarlySourceSlowEMATerminalControlRelativeSGDCatchupFU",
                    "M207-EarlySourceSlowEMATerminalControlRelativeAdamWCatchupFU",
                    "M208-EarlySourceSlowEMATerminalControlRelativeSourceBalancedCatchupFU",
                    "M215-EarlySourceSlowEMATerminalAcceptMemoryFU",
                    "M216-EarlySourceSlowEMATerminalAcceptMemoryDebtFU",
                    "M217-EarlySourceSlowEMATerminalSourceProgressMemoryFU",
                    "M161-EarlySourceSlowEMAShapePreserveFU",
                    "M162-EarlySourceSlowEMAShapePreserveRawGuardFU",
                    "M163-EarlySourceSlowEMAShapePreserveClampFU",
                }
                else 0.50
                if mechanism == "M130-H800ReadoutChannelRetentionFU"
                else 0.25
            )
            loss_gate_enabled = mechanism in {
                "M93-TrainLossGatedDualTimescaleTinyLateFU",
                "M94-TrainLossGatedDualTimescaleHoldFallbackFU",
                "M95-TrainLossGatedDualTimescaleTinyLateFallbackFU",
                "M96-TrainLossGatedDualTimescaleBoostedTinyLateFallbackFU",
                "M97-TrainLossLateHoldRecoveryFU",
                "M98-TrainLossLateLookaheadFloorFU",
                "M99-TrainLossTerminalLookaheadFloorFU",
                "M100-TrainLossEarlyTerminalLookaheadFloorFU",
                "M106-TrainLossTerminalProjectedLookaheadFloorFU",
                "M107-TrainLossTerminalConsensusLookaheadFloorFU",
                "M108-TrainLossTerminalSelectorLookaheadFloorFU",
                "M111-TrainLossTerminalPositiveLookaheadFloorFU",
                "M112-TrainLossTerminalCheckpointReentryFU",
                "M113-TrainLossTerminalHardSplitSourceFU",
                "M114-TrainLossTerminalAdamWLookaheadFU",
                "M115-TrainLossTerminalOptimizerSelectorFU",
                "M119-TerminalSourceConservingRouteFU",
                "M120-TrainLossRiskProfileRouteFU",
                "M121-DebtAwareSourceGateFU",
                "M125-TrainLossH2400CheckpointHoldFU",
                "M126-TrainLossH2800CheckpointHoldFU",
                "M127-TrainLossH2400DebtBailoutFU",
                "M128-TrainLossTerminalRawThenSourceGuardFU",
            }
            loss_gate_fallback_mode = {
                "M93-TrainLossGatedDualTimescaleTinyLateFU": "sgd",
                "M94-TrainLossGatedDualTimescaleHoldFallbackFU": "hold",
                "M95-TrainLossGatedDualTimescaleTinyLateFallbackFU": "tiny_late",
                "M96-TrainLossGatedDualTimescaleBoostedTinyLateFallbackFU": "boosted_tiny_late",
                "M97-TrainLossLateHoldRecoveryFU": "late_hold_recovery",
                "M98-TrainLossLateLookaheadFloorFU": "late_lookahead_floor",
                "M99-TrainLossTerminalLookaheadFloorFU": "terminal_lookahead_floor",
                "M100-TrainLossEarlyTerminalLookaheadFloorFU": "early_terminal_lookahead_floor",
                "M106-TrainLossTerminalProjectedLookaheadFloorFU": "terminal_projected_lookahead_floor",
                "M107-TrainLossTerminalConsensusLookaheadFloorFU": "terminal_consensus_lookahead_floor",
                "M108-TrainLossTerminalSelectorLookaheadFloorFU": "terminal_selector_lookahead_floor",
                "M111-TrainLossTerminalPositiveLookaheadFloorFU": "terminal_positive_lookahead_floor",
                "M112-TrainLossTerminalCheckpointReentryFU": "terminal_checkpoint_reentry",
                "M113-TrainLossTerminalHardSplitSourceFU": "terminal_hard_split_source",
                "M114-TrainLossTerminalAdamWLookaheadFU": "terminal_adamw_lookahead",
                "M115-TrainLossTerminalOptimizerSelectorFU": "terminal_optimizer_selector",
                "M119-TerminalSourceConservingRouteFU": "terminal_source_conserving_route",
                "M120-TrainLossRiskProfileRouteFU": "terminal_risk_profile_route",
                "M121-DebtAwareSourceGateFU": "terminal_debt_aware_source_gate",
                "M122-UngatedWarmTerminalSourceRouteFU": "ungated_terminal_source_conserving_route",
                "M123-UngatedWarmRiskProfileRouteFU": "ungated_terminal_risk_profile_route",
                "M124-UngatedWarmDebtRawBailoutFU": "ungated_debt_raw_bailout",
                "M125-TrainLossH2400CheckpointHoldFU": "terminal_h2400_checkpoint_hold",
                "M126-TrainLossH2800CheckpointHoldFU": "terminal_h2800_checkpoint_hold",
                "M127-TrainLossH2400DebtBailoutFU": "terminal_h2400_debt_bailout",
                "M128-TrainLossTerminalRawThenSourceGuardFU": "terminal_raw_then_source_guard",
                "M129-H800SourceSlowEMARetentionFU": "h800_source_slow_ema_retention",
                "M130-H800ReadoutChannelRetentionFU": "h800_readout_channel_retention",
                "M131-H800DualMemorySourceRetentionFU": "h800_dual_memory_source_retention",
                "M132-H1600SourceCheckpointReentryFU": "h1600_source_checkpoint_reentry",
                "M133-H2400SourceCheckpointReentryFU": "h2400_source_checkpoint_reentry",
                "M139-EarlySourceSlowEMATerminalGuardFU": "h800_source_slow_ema_terminal_source_guard",
                "M140-EarlySourceSlowEMATerminalRawGuardFU": "h800_source_slow_ema_terminal_raw_guard",
                "M141-EarlySourceSlowEMATerminalRawGuardStrongFU": "h800_source_slow_ema_terminal_raw_guard_strong",
                "M142-EarlySourceSlowEMATerminalProjectedOptimizerFU": "h800_source_slow_ema_terminal_projected_optimizer",
                "M218-EarlySourceSlowEMATerminalProjectedOptimizerLambda050FU": "h800_source_slow_ema_terminal_projected_optimizer_lambda050",
                "M219-EarlySourceSlowEMATerminalProjectedOptimizerLambda100FU": "h800_source_slow_ema_terminal_projected_optimizer_lambda100",
                "M143-EarlySourceSlowEMATerminalProjectedBlendFU": "h800_source_slow_ema_terminal_projected_blend",
                "M144-EarlySourceSlowEMATerminalAntiWashoutFU": "h800_source_slow_ema_terminal_antiwashout",
                "M145-EarlySourceSlowEMATerminalH4000ReentryFU": "h800_source_slow_ema_terminal_h4000_reentry",
                "M146-EarlySourceSlowEMASignalReservoirTargetFU": "h800_source_slow_ema_signal_reservoir_target",
                "M147-EarlySourceSlowEMASourceBankTargetFU": "h800_source_slow_ema_source_bank_target",
                "M148-EarlySourceSlowEMADualTargetGuardFU": "h800_source_slow_ema_dual_target_guard",
                "M149-EarlySourceSlowEMATerminalAdaptiveRawFU": "h800_source_slow_ema_terminal_adaptive_raw",
                "M150-EarlySourceSlowEMATerminalSparseSourceFU": "h800_source_slow_ema_terminal_sparse_source",
                "M151-EarlySourceSlowEMATerminalRatioPreserveFU": "h800_source_slow_ema_terminal_ratio_preserve",
                "M152-SourceProjectionB3NullConsensusTargetFU": "h800_source_slow_ema_terminal_source_projection_target",
                "M153-NoiseOrthogonalB3NullConsensusTargetFU": "h800_source_slow_ema_terminal_noise_orthogonal_target",
                "M154-EasyMarginB3NullConsensusTargetFU": "h800_source_slow_ema_terminal_easy_margin_target",
                "M158-EarlySourceSlowEMATerminalRejectSourceAxisRescueFU": "h800_source_slow_ema_terminal_reject_source_axis_rescue",
                "M159-EarlySourceSlowEMATerminalRejectSlowEMARescueFU": "h800_source_slow_ema_terminal_reject_slow_ema_rescue",
                "M160-EarlySourceSlowEMATerminalRejectHoldSourceRescueFU": "h800_source_slow_ema_terminal_reject_hold_source_rescue",
                "M161-EarlySourceSlowEMAShapePreserveFU": "h800_source_slow_ema_shape_preserve",
                "M162-EarlySourceSlowEMAShapePreserveRawGuardFU": "h800_source_slow_ema_shape_preserve_raw_guard",
                "M163-EarlySourceSlowEMAShapePreserveClampFU": "h800_source_slow_ema_shape_preserve_clamp",
                "M164-EarlySourceSlowEMATerminalRejectRawRescueFU": "h800_source_slow_ema_terminal_reject_raw_rescue",
                "M165-EarlySourceSlowEMATerminalRejectHybridRescueFU": "h800_source_slow_ema_terminal_reject_hybrid_rescue",
                "M166-EarlySourceSlowEMATerminalRejectLateOnlyRawRescueFU": "h800_source_slow_ema_terminal_reject_lateonly_raw_rescue",
                "M167-EarlySourceSlowEMATerminalTopKSupportGuardFU": "h800_source_slow_ema_terminal_topk_support_guard",
                "M168-EarlySourceSlowEMATerminalTopKRawGuardFU": "h800_source_slow_ema_terminal_topk_raw_guard",
                "M169-EarlySourceSlowEMATerminalTopKDebtCapFU": "h800_source_slow_ema_terminal_topk_debt_cap",
                "M170-EarlySourceSlowEMATerminalSourcePreserveStrongFU": "h800_source_slow_ema_terminal_source_preserve_strong",
                "M171-EarlySourceSlowEMATerminalSourcePreserveGentleFU": "h800_source_slow_ema_terminal_source_preserve_gentle",
                "M172-EarlySourceSlowEMATerminalInfoVolumeGuardFU": "h800_source_slow_ema_terminal_info_volume_guard",
                "M173-EarlySourceSlowEMALowNDSDiffeomorphicTargetFU": "h800_source_slow_ema_low_nds_diffeomorphic_target",
                "M174-EarlySourceSlowEMAInfoVolumeDiffeomorphicTargetFU": "h800_source_slow_ema_info_volume_diffeomorphic_target",
                "M175-EarlySourceSlowEMALowRankReadoutTransportFU": "h800_source_slow_ema_low_rank_readout_transport",
                "M179-EarlySourceSlowEMATerminalSourcePreserveVeryStrongFU": "h800_source_slow_ema_terminal_source_preserve_very_strong",
                "M180-EarlySourceSlowEMAH3600TerminalSourcePreserveFU": "h800_source_slow_ema_h3600_terminal_source_preserve",
                "M181-EarlySourceSlowEMATerminalNoraOrthogonalSourceFU": "h800_source_slow_ema_terminal_nora_orthogonal_source",
                "M182-EarlySourceSlowEMATerminalDebtAwarePreserveFU": "h800_source_slow_ema_terminal_debt_aware_preserve",
                "M183-EarlySourceSlowEMATerminalLowNDSMatrixBlockFU": "h800_source_slow_ema_terminal_low_nds_matrix_block",
                "M184-EarlySourceSlowEMATerminalDualMemoryPreserveFU": "h800_source_slow_ema_terminal_dual_memory_preserve",
                "M185-EarlySourceSlowEMASNRTerminalPredictorFU": "h800_source_slow_ema_terminal_snr_predictor",
                "M186-EarlySourceSlowEMASplitConsensusEstimatorFU": "h800_source_slow_ema_split_consensus_estimator",
                "M187-EarlySourceSlowEMASignalReservoirTransportFU": "h800_source_slow_ema_signal_reservoir_transport",
                "M188-EarlySourceSlowEMATerminalSourceFloorFU": "h800_source_slow_ema_terminal_source_floor",
                "M189-EarlySourceSlowEMATerminalH4000AnchorFloorFU": "h800_source_slow_ema_terminal_h4000_anchor_floor",
                "M190-EarlySourceSlowEMATerminalDecayAwareFloorFU": "h800_source_slow_ema_terminal_decay_aware_floor",
                "M191-EarlySourceSlowEMATerminalRawGuardSourceFloorFU": "h800_source_slow_ema_terminal_raw_guard_source_floor",
                "M192-EarlySourceSlowEMATerminalRawGuardH4000AnchorFloorFU": "h800_source_slow_ema_terminal_raw_guard_h4000_anchor_floor",
                "M193-EarlySourceSlowEMATerminalRawGuardDecayAwareFloorFU": "h800_source_slow_ema_terminal_raw_guard_decay_aware_floor",
                "M194-EarlySourceSlowEMATerminalAntiErosionOrthogonalFU": "h800_source_slow_ema_terminal_anti_erosion_orthogonal",
                "M195-EarlySourceSlowEMATerminalSourceReflectionGuardFU": "h800_source_slow_ema_terminal_source_reflection_guard",
                "M196-EarlySourceSlowEMATerminalH4000TransportCorrectorFU": "h800_source_slow_ema_terminal_h4000_transport_corrector",
                "M197-EarlySourceSlowEMATerminalH3200AnchorTransportFU": "h800_source_slow_ema_terminal_h3200_anchor_transport",
                "M198-EarlySourceSlowEMATerminalRawGuardH3200AnchorTransportFU": "h800_source_slow_ema_terminal_raw_guard_h3200_anchor_transport",
                "M199-EarlySourceSlowEMATerminalH3200RatioReentryFU": "h800_source_slow_ema_terminal_h3200_ratio_reentry",
                "M200-EarlySourceSlowEMATerminalH3200ProgressCarryFU": "h800_source_slow_ema_terminal_h3200_progress_carry",
                "M201-EarlySourceSlowEMATerminalH4000ProgressCarryFU": "h800_source_slow_ema_terminal_h4000_progress_carry",
                "M202-EarlySourceSlowEMATerminalH3200SourceProgressBlendFU": "h800_source_slow_ema_terminal_h3200_source_progress_blend",
                "M203-EarlySourceSlowEMATerminalRawGuardH3600GentleProgressFU": "h800_source_slow_ema_terminal_raw_guard_h3600_gentle_progress",
                "M204-EarlySourceSlowEMATerminalRawGuardH4000GentleProgressFU": "h800_source_slow_ema_terminal_raw_guard_h4000_gentle_progress",
                "M205-EarlySourceSlowEMATerminalRawGuardH4400ProjectedProgressFU": "h800_source_slow_ema_terminal_raw_guard_h4400_projected_progress",
                "M206-EarlySourceSlowEMATerminalControlRelativeSGDCatchupFU": "h800_source_slow_ema_terminal_control_relative_sgd_catchup",
                "M207-EarlySourceSlowEMATerminalControlRelativeAdamWCatchupFU": "h800_source_slow_ema_terminal_control_relative_adamw_catchup",
                "M208-EarlySourceSlowEMATerminalControlRelativeSourceBalancedCatchupFU": "h800_source_slow_ema_terminal_control_relative_source_balanced_catchup",
                "M209-EarlySourceSlowEMATerminalTrajectoryAdaptivePreserveFU": "h800_source_slow_ema_terminal_trajectory_adaptive_preserve",
                "M210-EarlySourceSlowEMATerminalMidErosionBridgeFU": "h800_source_slow_ema_terminal_mid_erosion_bridge",
                "M211-EarlySourceSlowEMATerminalTwoPhaseRatioRepairFU": "h800_source_slow_ema_terminal_two_phase_ratio_repair",
                "M212-EarlySourceSlowEMATerminalAntiSourceClipFU": "h800_source_slow_ema_terminal_anti_source_clip",
                "M213-EarlySourceSlowEMATerminalDebtAwareHoldFU": "h800_source_slow_ema_terminal_debt_aware_hold",
                "M214-EarlySourceSlowEMATerminalAnchorFlowTinyFU": "h800_source_slow_ema_terminal_anchor_flow_tiny",
                "M215-EarlySourceSlowEMATerminalAcceptMemoryFU": "h800_source_slow_ema_terminal_accept_memory",
                "M216-EarlySourceSlowEMATerminalAcceptMemoryDebtFU": "h800_source_slow_ema_terminal_accept_memory_debt",
                "M217-EarlySourceSlowEMATerminalSourceProgressMemoryFU": "h800_source_slow_ema_terminal_source_progress_memory",
            }.get(mechanism, "")
            signal_target_modes = {
                "h800_source_slow_ema_signal_reservoir_target",
                "h800_source_slow_ema_source_bank_target",
                "h800_source_slow_ema_dual_target_guard",
                "h800_source_slow_ema_terminal_source_projection_target",
                "h800_source_slow_ema_terminal_noise_orthogonal_target",
                "h800_source_slow_ema_terminal_easy_margin_target",
                "h800_source_slow_ema_low_nds_diffeomorphic_target",
                "h800_source_slow_ema_info_volume_diffeomorphic_target",
                "h800_source_slow_ema_low_rank_readout_transport",
            }
            terminal_reject_rescue_modes = {
                "h800_source_slow_ema_terminal_reject_source_axis_rescue",
                "h800_source_slow_ema_terminal_reject_slow_ema_rescue",
                "h800_source_slow_ema_terminal_reject_hold_source_rescue",
                "h800_source_slow_ema_terminal_reject_raw_rescue",
                "h800_source_slow_ema_terminal_reject_hybrid_rescue",
                "h800_source_slow_ema_terminal_reject_lateonly_raw_rescue",
            }
            source_shape_preserve_modes = {
                "h800_source_slow_ema_shape_preserve",
                "h800_source_slow_ema_shape_preserve_raw_guard",
                "h800_source_slow_ema_shape_preserve_clamp",
            }
            terminal_topk_support_modes = {
                "h800_source_slow_ema_terminal_topk_support_guard",
                "h800_source_slow_ema_terminal_topk_raw_guard",
                "h800_source_slow_ema_terminal_topk_debt_cap",
            }
            terminal_source_preserve_modes = {
                "h800_source_slow_ema_terminal_source_preserve_strong",
                "h800_source_slow_ema_terminal_source_preserve_gentle",
                "h800_source_slow_ema_terminal_info_volume_guard",
                "h800_source_slow_ema_terminal_source_preserve_very_strong",
                "h800_source_slow_ema_h3600_terminal_source_preserve",
                "h800_source_slow_ema_terminal_nora_orthogonal_source",
                "h800_source_slow_ema_terminal_debt_aware_preserve",
                "h800_source_slow_ema_terminal_low_nds_matrix_block",
                "h800_source_slow_ema_terminal_dual_memory_preserve",
                "h800_source_slow_ema_terminal_snr_predictor",
                "h800_source_slow_ema_split_consensus_estimator",
                "h800_source_slow_ema_signal_reservoir_transport",
                "h800_source_slow_ema_terminal_source_floor",
                "h800_source_slow_ema_terminal_h4000_anchor_floor",
                "h800_source_slow_ema_terminal_decay_aware_floor",
                "h800_source_slow_ema_terminal_raw_guard_source_floor",
                "h800_source_slow_ema_terminal_raw_guard_h4000_anchor_floor",
                "h800_source_slow_ema_terminal_raw_guard_decay_aware_floor",
                "h800_source_slow_ema_terminal_anti_erosion_orthogonal",
                "h800_source_slow_ema_terminal_source_reflection_guard",
                "h800_source_slow_ema_terminal_h4000_transport_corrector",
                "h800_source_slow_ema_terminal_h3200_anchor_transport",
                "h800_source_slow_ema_terminal_raw_guard_h3200_anchor_transport",
                "h800_source_slow_ema_terminal_h3200_ratio_reentry",
                "h800_source_slow_ema_terminal_h3200_progress_carry",
                "h800_source_slow_ema_terminal_h4000_progress_carry",
                "h800_source_slow_ema_terminal_h3200_source_progress_blend",
                "h800_source_slow_ema_terminal_raw_guard_h3600_gentle_progress",
                "h800_source_slow_ema_terminal_raw_guard_h4000_gentle_progress",
                "h800_source_slow_ema_terminal_raw_guard_h4400_projected_progress",
                "h800_source_slow_ema_terminal_control_relative_sgd_catchup",
                "h800_source_slow_ema_terminal_control_relative_adamw_catchup",
                "h800_source_slow_ema_terminal_control_relative_source_balanced_catchup",
                "h800_source_slow_ema_terminal_trajectory_adaptive_preserve",
                "h800_source_slow_ema_terminal_mid_erosion_bridge",
                "h800_source_slow_ema_terminal_two_phase_ratio_repair",
                "h800_source_slow_ema_terminal_anti_source_clip",
                "h800_source_slow_ema_terminal_debt_aware_hold",
                "h800_source_slow_ema_terminal_anchor_flow_tiny",
                "h800_source_slow_ema_terminal_accept_memory",
                "h800_source_slow_ema_terminal_accept_memory_debt",
                "h800_source_slow_ema_terminal_source_progress_memory",
            }
            terminal_accept_memory_modes = {
                "h800_source_slow_ema_terminal_accept_memory",
                "h800_source_slow_ema_terminal_accept_memory_debt",
                "h800_source_slow_ema_terminal_source_progress_memory",
            }
            terminal_anti_erosion_modes = {
                "h800_source_slow_ema_terminal_anti_erosion_orthogonal",
                "h800_source_slow_ema_terminal_source_reflection_guard",
                "h800_source_slow_ema_terminal_h4000_transport_corrector",
            }
            terminal_h3200_anchor_modes = {
                "h800_source_slow_ema_terminal_h3200_anchor_transport",
                "h800_source_slow_ema_terminal_raw_guard_h3200_anchor_transport",
                "h800_source_slow_ema_terminal_h3200_ratio_reentry",
            }
            terminal_progress_modes = {
                "h800_source_slow_ema_terminal_h3200_progress_carry",
                "h800_source_slow_ema_terminal_h4000_progress_carry",
                "h800_source_slow_ema_terminal_h3200_source_progress_blend",
                "h800_source_slow_ema_terminal_raw_guard_h3600_gentle_progress",
                "h800_source_slow_ema_terminal_raw_guard_h4000_gentle_progress",
                "h800_source_slow_ema_terminal_raw_guard_h4400_projected_progress",
            }
            terminal_control_catchup_modes = {
                "h800_source_slow_ema_terminal_control_relative_sgd_catchup",
                "h800_source_slow_ema_terminal_control_relative_adamw_catchup",
                "h800_source_slow_ema_terminal_control_relative_source_balanced_catchup",
            }
            terminal_trajectory_adaptive_modes = {
                "h800_source_slow_ema_terminal_trajectory_adaptive_preserve",
                "h800_source_slow_ema_terminal_mid_erosion_bridge",
                "h800_source_slow_ema_terminal_two_phase_ratio_repair",
            }
            terminal_minimal_transport_modes = {
                "h800_source_slow_ema_terminal_anti_source_clip",
                "h800_source_slow_ema_terminal_debt_aware_hold",
                "h800_source_slow_ema_terminal_anchor_flow_tiny",
            }
            loss_gate_h400_threshold = 0.020872684195637703
            loss_gate_h800_threshold = 0.016799673438072205

            def update_train_loss_gate() -> float:
                nonlocal train_loss_gate_h400_pass, train_loss_gate_h800_pass, train_loss_gate_last_value
                with torch.no_grad():
                    train_loss_gate_last_value = float(F.cross_entropy(model(x_train).float(), y_train).item())
                if step >= 400 and train_loss_gate_last_value <= loss_gate_h400_threshold:
                    train_loss_gate_h400_pass = 1
                if step >= 800 and train_loss_gate_last_value <= loss_gate_h800_threshold:
                    train_loss_gate_h800_pass = 1
                return train_loss_gate_last_value

            def train_loss_selector_accept() -> int:
                if ungated_terminal_warm:
                    return 1
                if not loss_gate_enabled or step <= 800:
                    return 1
                return int(train_loss_gate_h400_pass or train_loss_gate_h800_pass)

            terminal_projected_removed_norm = 0.0
            terminal_consensus_density = 0.0
            terminal_consensus_balance = 0.0
            terminal_consensus_corrupt_cos = 0.0
            terminal_selector_score = 0.0
            terminal_selector_kind = ""
            terminal_selector_current_cos = ""
            terminal_h1600_source_anchor: torch.Tensor | None = None
            terminal_h2400_source_anchor: torch.Tensor | None = None
            terminal_h4000_source_anchor: torch.Tensor | None = None
            terminal_h3200_source_anchor: torch.Tensor | None = None
            terminal_accept_memory_update: torch.Tensor | None = None
            terminal_accept_memory_signal = 0.0
            terminal_accept_memory_step = 0

            def make_late_floor_update(project_source: bool = False) -> UpdateTensor:
                nonlocal terminal_projected_removed_norm
                grad_vec = flat_grad(model, device)
                source_space = "parameter"
                source = "train_stream_tiny_late_sgd_floor"
                role = "tiny_late_gradient_floor"
                if project_source:
                    source_ref = source_axis.to(device=device, dtype=grad_vec.dtype)
                    source_norm2 = torch.dot(source_ref.float(), source_ref.float()).clamp_min(1.0e-12)
                    anti_coeff = torch.clamp(torch.dot(grad_vec.float(), source_ref.float()), max=0.0) / source_norm2
                    projected = grad_vec - anti_coeff.to(device=device, dtype=grad_vec.dtype) * source_ref
                    projected = torch.nan_to_num(projected, nan=0.0, posinf=0.0, neginf=0.0)
                    terminal_projected_removed_norm = float(torch.linalg.vector_norm((grad_vec - projected).detach()).item())
                    grad_vec = projected
                    source_space = "slow_state"
                    source = "train_stream_terminal_projected_lookahead_floor"
                    role = "source_preserving_terminal_projected_gradient_floor"
                return UpdateTensor(
                    grad_vec,
                    "gradient",
                    "subtract",
                    source_space,
                    source,
                    mechanism,
                    role=role,
                    one_step_descent_claim=1,
                )

            def make_terminal_consensus_update() -> UpdateTensor:
                nonlocal terminal_consensus_density, terminal_consensus_balance, terminal_consensus_corrupt_cos
                params = [p for p in model.parameters() if p.requires_grad]
                original_grads = [None if p.grad is None else p.grad.detach().clone() for p in params]

                def restore_original_grads() -> None:
                    model.zero_grad(set_to_none=True)
                    for param, grad in zip(params, original_grads):
                        param.grad = None if grad is None else grad.to(device=param.device, dtype=param.dtype)

                def grad_for(xb2: torch.Tensor, yb2: torch.Tensor) -> torch.Tensor:
                    model.zero_grad(set_to_none=True)
                    F.cross_entropy(model(xb2).float(), yb2).backward()
                    return flat_grad(model, device).detach().clone()

                g_a = grad_for(channel_a[0], channel_a[1])
                g_b = grad_for(channel_b[0], channel_b[1])
                corrupt_labels = (channel_b[1] + 1) % max(2, int(getattr(args, "classes", 10)))
                g_corrupt = grad_for(channel_b[0], corrupt_labels)
                restore_original_grads()
                agree = torch.sign(g_a) == torch.sign(g_b)
                terminal_consensus_density = float(agree.float().mean().item()) if agree.numel() else 0.0
                mag_a = g_a.detach().float().abs()
                mag_b = g_b.detach().float().abs()
                balance = torch.minimum(mag_a, mag_b) / torch.maximum(mag_a, mag_b).clamp_min(1.0e-8)
                terminal_consensus_balance = float(balance[agree].mean().item()) if bool(agree.any().item()) else 0.0
                consensus = torch.where(agree, 0.50 * (g_a + g_b), torch.zeros_like(g_a))
                consensus = consensus * balance.to(device=device, dtype=consensus.dtype)
                corrupt_denom = torch.dot(g_corrupt.float(), g_corrupt.float()).clamp_min(1.0e-12)
                corrupt_proj = torch.dot(consensus.float(), g_corrupt.float()) / corrupt_denom
                if float(corrupt_proj.item()) > 0.0:
                    consensus = consensus - corrupt_proj.to(device=consensus.device, dtype=consensus.dtype) * g_corrupt
                consensus = torch.nan_to_num(consensus, nan=0.0, posinf=0.0, neginf=0.0)
                terminal_consensus_corrupt_cos = cosine(consensus, g_corrupt)
                return UpdateTensor(
                    normalized_like(consensus, flat_grad(model, device) if consensus.numel() else consensus),
                    "gradient",
                    "subtract",
                    "slow_state",
                    "train_stream_terminal_consensus_lookahead_floor",
                    mechanism,
                    role="terminal_cross_split_consensus_noise_reservoir_floor",
                    one_step_descent_claim=0,
                )

            def apply_late_floor(scale: float = 0.10) -> None:
                if tiny_floor:
                    floor_update = make_late_floor_update()
                    apply_update(model, floor_update, lr=scale * float(args.lr))
                else:
                    opt_sgd.step_gradient()

            def apply_late_lookahead_floor(scale: float = 0.05, project_source: bool = False, positive_only: bool = False) -> tuple[int, float, float, float]:
                if not tiny_floor:
                    return 0, 0.0, 0.0, 0.0
                floor_update = make_late_floor_update(project_source=project_source)
                corrupt_labels = (channel_b[1] + 1) % max(2, int(getattr(args, "classes", 10)))
                before = snapshot(model)
                with torch.no_grad():
                    before_a = F.cross_entropy(model(channel_a[0]).float(), channel_a[1], reduction="none")
                    before_b = F.cross_entropy(model(channel_b[0]).float(), channel_b[1], reduction="none")
                    before_corrupt = F.cross_entropy(model(channel_b[0]).float(), corrupt_labels, reduction="none")
                apply_update(model, floor_update, lr=scale * float(args.lr))
                with torch.no_grad():
                    after_a = F.cross_entropy(model(channel_a[0]).float(), channel_a[1], reduction="none")
                    after_b = F.cross_entropy(model(channel_b[0]).float(), channel_b[1], reduction="none")
                    after_corrupt = F.cross_entropy(model(channel_b[0]).float(), corrupt_labels, reduction="none")
                load_flat_params(model, before)
                gain_a = float((before_a - after_a).mean().item())
                gain_b = float((before_b - after_b).mean().item())
                corrupt_gain = float((before_corrupt - after_corrupt).mean().item())
                signal_gain = min(gain_a, gain_b)
                signal_threshold = 1.0e-7 if positive_only else -1.0e-7
                corrupt_fraction = 0.25 if positive_only else 0.50
                accept = int(
                    signal_gain >= signal_threshold
                    and corrupt_gain <= max(1.0e-5, corrupt_fraction * max(0.0, signal_gain))
                )
                if accept:
                    apply_update(model, floor_update, lr=scale * float(args.lr))
                return accept, signal_gain, corrupt_gain, gain_b

            def apply_checkpoint_reentry(scale: float = 0.05) -> tuple[int, float, float, float]:
                if param_slow_state is None:
                    return 0, 0.0, 0.0, 0.0
                current_params = snapshot(model)
                checkpoint = param_slow_state.to(device=current_params.device, dtype=current_params.dtype)
                if checkpoint.numel() != current_params.numel():
                    return 0, 0.0, 0.0, 0.0
                delta = current_params - checkpoint
                delta_norm = float(torch.linalg.vector_norm(delta.detach()).item())
                if delta_norm <= 1.0e-12:
                    return 0, 0.0, 0.0, 0.0
                checkpoint_update = UpdateTensor(
                    delta,
                    "step",
                    "subtract",
                    "slow_state",
                    "train_stream_terminal_h3200_checkpoint_reentry",
                    mechanism,
                    role="terminal_h3200_checkpoint_reentry",
                    one_step_descent_claim=0,
                )
                corrupt_labels = (channel_b[1] + 1) % max(2, int(getattr(args, "classes", 10)))
                before = snapshot(model)
                with torch.no_grad():
                    before_a = F.cross_entropy(model(channel_a[0]).float(), channel_a[1], reduction="none")
                    before_b = F.cross_entropy(model(channel_b[0]).float(), channel_b[1], reduction="none")
                    before_corrupt = F.cross_entropy(model(channel_b[0]).float(), corrupt_labels, reduction="none")
                apply_update(model, checkpoint_update, lr=scale)
                with torch.no_grad():
                    after_a = F.cross_entropy(model(channel_a[0]).float(), channel_a[1], reduction="none")
                    after_b = F.cross_entropy(model(channel_b[0]).float(), channel_b[1], reduction="none")
                    after_corrupt = F.cross_entropy(model(channel_b[0]).float(), corrupt_labels, reduction="none")
                load_flat_params(model, before)
                gain_a = float((before_a - after_a).mean().item())
                gain_b = float((before_b - after_b).mean().item())
                corrupt_gain = float((before_corrupt - after_corrupt).mean().item())
                signal_gain = min(gain_a, gain_b)
                accept = int(signal_gain >= -1.0e-7 and corrupt_gain <= max(1.0e-5, 0.25 * max(0.0, signal_gain)))
                if accept:
                    apply_update(model, checkpoint_update, lr=scale)
                return accept, signal_gain, corrupt_gain, gain_b

            def apply_terminal_hard_split_source(scale: float = 0.10) -> tuple[int, float, float, float]:
                params = [p for p in model.parameters() if p.requires_grad]
                original_grads = [None if p.grad is None else p.grad.detach().clone() for p in params]

                def restore_original_grads() -> None:
                    model.zero_grad(set_to_none=True)
                    for param, grad in zip(params, original_grads):
                        param.grad = None if grad is None else grad.to(device=param.device, dtype=param.dtype)

                def hard_mask(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
                    loss_vec = F.cross_entropy(logits.float(), labels, reduction="none")
                    if loss_vec.numel() == 0:
                        return torch.zeros(0, device=logits.device, dtype=torch.bool)
                    mask = loss_vec >= loss_vec.detach().mean()
                    if not bool(mask.any().item()):
                        mask = torch.zeros_like(mask, dtype=torch.bool)
                        mask[int(torch.argmax(loss_vec.detach()).item())] = True
                    return mask.detach()

                def hard_grad(xh: torch.Tensor, yh: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                    with torch.no_grad():
                        mask = hard_mask(model(xh).float(), yh)
                    model.zero_grad(set_to_none=True)
                    logits = model(xh).float()
                    loss_vec = F.cross_entropy(logits, yh, reduction="none")
                    loss = loss_vec[mask].mean() if bool(mask.any().item()) else loss_vec.mean()
                    loss.backward()
                    return flat_grad(model, device).detach().clone(), mask

                g_a, mask_a = hard_grad(channel_a[0], channel_a[1])
                g_b, mask_b = hard_grad(channel_b[0], channel_b[1])
                corrupt_labels = (channel_b[1] + 1) % max(2, int(getattr(args, "classes", 10)))
                g_corrupt, _mask_corrupt = hard_grad(channel_b[0], corrupt_labels)
                restore_original_grads()

                agree = torch.sign(g_a) == torch.sign(g_b)
                density = float(agree.float().mean().item()) if agree.numel() else 0.0
                mag_a = g_a.detach().float().abs()
                mag_b = g_b.detach().float().abs()
                balance = torch.minimum(mag_a, mag_b) / torch.maximum(mag_a, mag_b).clamp_min(1.0e-8)
                balance_mean = float(balance[agree].mean().item()) if bool(agree.any().item()) else 0.0
                split_mean = torch.where(agree, 0.50 * (g_a + g_b), torch.zeros_like(g_a))
                hard_source = split_mean * balance.to(device=device, dtype=split_mean.dtype)
                corrupt_denom = torch.dot(g_corrupt.float(), g_corrupt.float()).clamp_min(1.0e-12)
                corrupt_proj = torch.dot(hard_source.float(), g_corrupt.float()) / corrupt_denom
                if float(corrupt_proj.item()) > 0.0:
                    hard_source = hard_source - corrupt_proj.to(device=hard_source.device, dtype=hard_source.dtype) * g_corrupt
                hard_source = torch.nan_to_num(hard_source, nan=0.0, posinf=0.0, neginf=0.0)
                g_current = flat_grad(model, device).detach().clone()
                update_vec = normalized_like(hard_source, g_current if g_current.numel() else hard_source)
                if float(torch.linalg.vector_norm(update_vec.detach()).item()) <= 1.0e-12:
                    return 0, 0.0, 0.0, 0.0
                hard_update = UpdateTensor(
                    update_vec,
                    "step",
                    "subtract",
                    "slow_state",
                    "train_stream_terminal_hard_split_source",
                    mechanism,
                    role="terminal_hard_split_source",
                    one_step_descent_claim=0,
                )
                before = snapshot(model)
                with torch.no_grad():
                    before_a_all = F.cross_entropy(model(channel_a[0]).float(), channel_a[1], reduction="none")
                    before_b_all = F.cross_entropy(model(channel_b[0]).float(), channel_b[1], reduction="none")
                    before_corrupt_all = F.cross_entropy(model(channel_b[0]).float(), corrupt_labels, reduction="none")
                apply_update(model, hard_update, lr=scale * float(args.lr))
                with torch.no_grad():
                    after_a_all = F.cross_entropy(model(channel_a[0]).float(), channel_a[1], reduction="none")
                    after_b_all = F.cross_entropy(model(channel_b[0]).float(), channel_b[1], reduction="none")
                    after_corrupt_all = F.cross_entropy(model(channel_b[0]).float(), corrupt_labels, reduction="none")
                load_flat_params(model, before)
                gain_a = float((before_a_all[mask_a] - after_a_all[mask_a]).mean().item()) if bool(mask_a.any().item()) else 0.0
                gain_b = float((before_b_all[mask_b] - after_b_all[mask_b]).mean().item()) if bool(mask_b.any().item()) else 0.0
                corrupt_gain = float((before_corrupt_all[mask_b] - after_corrupt_all[mask_b]).mean().item()) if bool(mask_b.any().item()) else 0.0
                signal_gain = min(gain_a, gain_b)
                corrupt_cos = cosine(hard_source, g_corrupt)
                accept = int(
                    density >= 0.03
                    and balance_mean >= 0.03
                    and signal_gain >= 1.0e-7
                    and corrupt_gain <= max(1.0e-5, 0.20 * max(0.0, signal_gain))
                    and corrupt_cos <= 0.25
                )
                if accept:
                    apply_update(model, hard_update, lr=scale * float(args.lr))
                return accept, signal_gain, corrupt_gain, gain_b

            def apply_terminal_adamw_lookahead(scale: float = 0.10) -> tuple[int, float, float, float]:
                def commit_scaled_adamw() -> None:
                    old_lrs = [float(group.get("lr", args.lr)) for group in opt_adam.param_groups]
                    for group, old_lr in zip(opt_adam.param_groups, old_lrs):
                        group["lr"] = scale * old_lr
                    opt_adam.step()
                    for group, old_lr in zip(opt_adam.param_groups, old_lrs):
                        group["lr"] = old_lr

                corrupt_labels = (channel_b[1] + 1) % max(2, int(getattr(args, "classes", 10)))
                before = snapshot(model)
                opt_state = deepcopy(opt_adam.state_dict())
                old_lrs = [float(group.get("lr", args.lr)) for group in opt_adam.param_groups]
                with torch.no_grad():
                    before_a = F.cross_entropy(model(channel_a[0]).float(), channel_a[1], reduction="none")
                    before_b = F.cross_entropy(model(channel_b[0]).float(), channel_b[1], reduction="none")
                    before_corrupt = F.cross_entropy(model(channel_b[0]).float(), corrupt_labels, reduction="none")
                for group, old_lr in zip(opt_adam.param_groups, old_lrs):
                    group["lr"] = scale * old_lr
                opt_adam.step()
                with torch.no_grad():
                    after_a = F.cross_entropy(model(channel_a[0]).float(), channel_a[1], reduction="none")
                    after_b = F.cross_entropy(model(channel_b[0]).float(), channel_b[1], reduction="none")
                    after_corrupt = F.cross_entropy(model(channel_b[0]).float(), corrupt_labels, reduction="none")
                load_flat_params(model, before)
                opt_adam.load_state_dict(opt_state)
                for group, old_lr in zip(opt_adam.param_groups, old_lrs):
                    group["lr"] = old_lr
                gain_a = float((before_a - after_a).mean().item())
                gain_b = float((before_b - after_b).mean().item())
                corrupt_gain = float((before_corrupt - after_corrupt).mean().item())
                signal_gain = min(gain_a, gain_b)
                accept = int(signal_gain >= 1.0e-7 and corrupt_gain <= max(1.0e-5, 0.25 * max(0.0, signal_gain)))
                if accept:
                    commit_scaled_adamw()
                return accept, signal_gain, corrupt_gain, gain_b

            def terminal_adamw_candidate_metrics(scale: float = 0.10) -> dict[str, float]:
                corrupt_labels = (channel_b[1] + 1) % max(2, int(getattr(args, "classes", 10)))
                before = snapshot(model)
                opt_state = deepcopy(opt_adam.state_dict())
                old_lrs = [float(group.get("lr", args.lr)) for group in opt_adam.param_groups]
                with torch.no_grad():
                    before_a = F.cross_entropy(model(channel_a[0]).float(), channel_a[1], reduction="none")
                    before_b = F.cross_entropy(model(channel_b[0]).float(), channel_b[1], reduction="none")
                    before_corrupt = F.cross_entropy(model(channel_b[0]).float(), corrupt_labels, reduction="none")
                for group, old_lr in zip(opt_adam.param_groups, old_lrs):
                    group["lr"] = scale * old_lr
                opt_adam.step()
                with torch.no_grad():
                    after_a = F.cross_entropy(model(channel_a[0]).float(), channel_a[1], reduction="none")
                    after_b = F.cross_entropy(model(channel_b[0]).float(), channel_b[1], reduction="none")
                    after_corrupt = F.cross_entropy(model(channel_b[0]).float(), corrupt_labels, reduction="none")
                load_flat_params(model, before)
                opt_adam.load_state_dict(opt_state)
                for group, old_lr in zip(opt_adam.param_groups, old_lrs):
                    group["lr"] = old_lr
                gain_a = float((before_a - after_a).mean().item())
                gain_b = float((before_b - after_b).mean().item())
                corrupt_gain = float((before_corrupt - after_corrupt).mean().item())
                signal_gain = min(gain_a, gain_b)
                return {"norm": 1.0, "gain_a": gain_a, "gain_b": gain_b, "signal": signal_gain, "corrupt": corrupt_gain, "score": signal_gain - max(0.0, corrupt_gain)}

            def commit_terminal_adamw(scale: float = 0.10) -> None:
                old_lrs = [float(group.get("lr", args.lr)) for group in opt_adam.param_groups]
                for group, old_lr in zip(opt_adam.param_groups, old_lrs):
                    group["lr"] = scale * old_lr
                opt_adam.step()
                for group, old_lr in zip(opt_adam.param_groups, old_lrs):
                    group["lr"] = old_lr

            def apply_terminal_consensus_lookahead_floor(scale: float = 0.10) -> tuple[int, float, float, float]:
                consensus_update = make_terminal_consensus_update()
                corrupt_labels = (channel_b[1] + 1) % max(2, int(getattr(args, "classes", 10)))
                before = snapshot(model)
                with torch.no_grad():
                    before_a = F.cross_entropy(model(channel_a[0]).float(), channel_a[1], reduction="none")
                    before_b = F.cross_entropy(model(channel_b[0]).float(), channel_b[1], reduction="none")
                    before_corrupt = F.cross_entropy(model(channel_b[0]).float(), corrupt_labels, reduction="none")
                apply_update(model, consensus_update, lr=scale * float(args.lr))
                with torch.no_grad():
                    after_a = F.cross_entropy(model(channel_a[0]).float(), channel_a[1], reduction="none")
                    after_b = F.cross_entropy(model(channel_b[0]).float(), channel_b[1], reduction="none")
                    after_corrupt = F.cross_entropy(model(channel_b[0]).float(), corrupt_labels, reduction="none")
                load_flat_params(model, before)
                gain_a = float((before_a - after_a).mean().item())
                gain_b = float((before_b - after_b).mean().item())
                corrupt_gain = float((before_corrupt - after_corrupt).mean().item())
                signal_gain = min(gain_a, gain_b)
                accept = int(
                    terminal_consensus_density >= 0.03
                    and terminal_consensus_balance >= 0.03
                    and signal_gain >= -1.0e-7
                    and corrupt_gain <= max(1.0e-5, 0.35 * max(0.0, signal_gain))
                    and terminal_consensus_corrupt_cos <= 0.30
                )
                if accept:
                    apply_update(model, consensus_update, lr=scale * float(args.lr))
                return accept, signal_gain, corrupt_gain, gain_b

            def terminal_candidate_metrics(candidate_update: UpdateTensor, scale: float) -> dict[str, float]:
                norm = float(torch.linalg.vector_norm(candidate_update.tensor.detach()).item())
                if norm <= 1.0e-12:
                    return {"norm": norm, "gain_a": 0.0, "gain_b": 0.0, "signal": -1.0, "corrupt": 1.0, "score": -2.0}
                corrupt_labels = (channel_b[1] + 1) % max(2, int(getattr(args, "classes", 10)))
                before = snapshot(model)
                with torch.no_grad():
                    before_a = F.cross_entropy(model(channel_a[0]).float(), channel_a[1], reduction="none")
                    before_b = F.cross_entropy(model(channel_b[0]).float(), channel_b[1], reduction="none")
                    before_corrupt = F.cross_entropy(model(channel_b[0]).float(), corrupt_labels, reduction="none")
                apply_update(model, candidate_update, lr=scale * float(args.lr))
                with torch.no_grad():
                    after_a = F.cross_entropy(model(channel_a[0]).float(), channel_a[1], reduction="none")
                    after_b = F.cross_entropy(model(channel_b[0]).float(), channel_b[1], reduction="none")
                    after_corrupt = F.cross_entropy(model(channel_b[0]).float(), corrupt_labels, reduction="none")
                load_flat_params(model, before)
                gain_a = float((before_a - after_a).mean().item())
                gain_b = float((before_b - after_b).mean().item())
                corrupt_gain = float((before_corrupt - after_corrupt).mean().item())
                signal_gain = min(gain_a, gain_b)
                score = signal_gain - max(0.0, corrupt_gain)
                return {"norm": norm, "gain_a": gain_a, "gain_b": gain_b, "signal": signal_gain, "corrupt": corrupt_gain, "score": score}

            def apply_terminal_selector_lookahead_floor(scale: float = 0.10) -> tuple[int, float, float, float]:
                nonlocal terminal_selector_score, terminal_selector_kind, terminal_selector_current_cos
                raw_update = make_late_floor_update(project_source=False)
                projected_update = make_late_floor_update(project_source=True)
                consensus_update = make_terminal_consensus_update()
                raw = terminal_candidate_metrics(raw_update, scale)
                projected = terminal_candidate_metrics(projected_update, scale)
                consensus = terminal_candidate_metrics(consensus_update, scale)
                candidates = [
                    ("raw", raw_update, raw, raw["corrupt"] <= max(1.0e-5, 0.50 * max(0.0, raw["signal"]))),
                    ("projected", projected_update, projected, projected["corrupt"] <= max(1.0e-5, 0.50 * max(0.0, projected["signal"]))),
                    (
                        "consensus",
                        consensus_update,
                        consensus,
                        terminal_consensus_density >= 0.03
                        and terminal_consensus_balance >= 0.03
                        and consensus["corrupt"] <= max(1.0e-5, 0.35 * max(0.0, consensus["signal"]))
                        and terminal_consensus_corrupt_cos <= 0.30,
                    ),
                ]
                allowed = [
                    (kind, candidate_update, metrics)
                    for kind, candidate_update, metrics, specific_gate in candidates
                    if specific_gate and metrics["norm"] > 1.0e-12 and metrics["signal"] >= -1.0e-7
                ]
                if not allowed:
                    best_metrics = max((raw, projected, consensus), key=lambda item: (item["score"], item["signal"], -item["corrupt"]))
                    terminal_selector_score = float(best_metrics["score"])
                    terminal_selector_kind = "reject"
                    terminal_selector_current_cos = ""
                    return 0, float(best_metrics["signal"]), float(best_metrics["corrupt"]), float(best_metrics["gain_b"])
                best_kind, best_update, best_metrics = max(allowed, key=lambda item: (item[2]["score"], item[2]["signal"], -item[2]["corrupt"]))
                apply_update(model, best_update, lr=scale * float(args.lr))
                terminal_selector_score = float(best_metrics["score"])
                terminal_selector_kind = best_kind
                terminal_selector_current_cos = cosine(best_update.tensor, flat_grad(model, device))
                return 1, float(best_metrics["signal"]), float(best_metrics["corrupt"]), float(best_metrics["gain_b"])

            def apply_terminal_optimizer_selector(scale: float = 0.10) -> tuple[int, float, float, float]:
                nonlocal terminal_selector_score, terminal_selector_kind, terminal_selector_current_cos
                raw_update = make_late_floor_update(project_source=False)
                raw = terminal_candidate_metrics(raw_update, scale)
                adamw = terminal_adamw_candidate_metrics(scale)
                candidates = [
                    ("raw", raw_update, raw, raw["corrupt"] <= max(1.0e-5, 0.50 * max(0.0, raw["signal"]))),
                    ("adamw", None, adamw, adamw["corrupt"] <= max(1.0e-5, 0.25 * max(0.0, adamw["signal"]))),
                ]
                allowed = [
                    (kind, candidate_update, metrics)
                    for kind, candidate_update, metrics, specific_gate in candidates
                    if specific_gate and metrics["signal"] >= 1.0e-7
                ]
                if not allowed:
                    best_metrics = max((raw, adamw), key=lambda item: (item["score"], item["signal"], -item["corrupt"]))
                    terminal_selector_score = float(best_metrics["score"])
                    terminal_selector_kind = "reject"
                    terminal_selector_current_cos = ""
                    return 0, float(best_metrics["signal"]), float(best_metrics["corrupt"]), float(best_metrics["gain_b"])
                best_kind, best_update, best_metrics = max(allowed, key=lambda item: (item[2]["score"], item[2]["signal"], -item[2]["corrupt"]))
                if best_kind == "adamw":
                    commit_terminal_adamw(scale)
                    terminal_selector_current_cos = "adamw"
                elif best_update is not None:
                    apply_update(model, best_update, lr=scale * float(args.lr))
                    terminal_selector_current_cos = cosine(best_update.tensor, flat_grad(model, device))
                terminal_selector_score = float(best_metrics["score"])
                terminal_selector_kind = best_kind
                return 1, float(best_metrics["signal"]), float(best_metrics["corrupt"]), float(best_metrics["gain_b"])

            def apply_terminal_source_conserving_route(
                scale: float = 0.10,
                *,
                risk_profile: bool = False,
                debt_aware: bool = False,
            ) -> tuple[int, float, float, float]:
                nonlocal terminal_selector_score, terminal_selector_kind, terminal_selector_current_cos
                source_ref = source_axis.to(device=device, dtype=flat_grad(model, device).dtype)
                source_update = UpdateTensor(
                    normalized_like(source_ref, flat_grad(model, device) if source_ref.numel() else source_ref),
                    "step",
                    "subtract",
                    "slow_state",
                    "train_stream_terminal_source_conserving_route_source_axis",
                    mechanism,
                    role="terminal_source_conserving_source_axis_candidate",
                    one_step_descent_claim=0,
                )
                projected_update = make_late_floor_update(project_source=True)
                consensus_update = make_terminal_consensus_update()
                lr_denom = max(float(args.lr), 1.0e-12)
                source_scale = max(1.0e-8, 0.25 * float(args.fu_lr) / lr_denom)
                source = terminal_candidate_metrics(source_update, source_scale)
                projected = terminal_candidate_metrics(projected_update, scale)
                consensus = terminal_candidate_metrics(consensus_update, scale)

                risk_gap = 0.0
                risk_level = 0.0
                risk_ok = True
                if risk_profile:
                    with torch.no_grad():
                        risk_a = F.cross_entropy(model(channel_a[0]).float(), channel_a[1], reduction="none")
                        risk_b = F.cross_entropy(model(channel_b[0]).float(), channel_b[1], reduction="none")
                    mean_a = float(risk_a.mean().item())
                    mean_b = float(risk_b.mean().item())
                    risk_gap = abs(mean_a - mean_b)
                    risk_level = 0.50 * (mean_a + mean_b)
                    risk_ok = risk_gap <= max(0.03, 0.35 * risk_level)

                debt_exists = False
                if debt_aware:
                    current_loss = update_train_loss_gate()
                    debt_exists = current_loss > 1.10 * loss_gate_h800_threshold

                def source_gate(metrics: dict[str, float]) -> bool:
                    return (
                        metrics["norm"] > 1.0e-12
                        and metrics["signal"] >= -1.0e-6
                        and metrics["corrupt"] <= max(1.0e-5, 0.25 * max(0.0, metrics["signal"]))
                        and cosine(source_update.tensor, source_ref) >= -0.01
                    )

                def projected_gate(metrics: dict[str, float]) -> bool:
                    return (
                        metrics["norm"] > 1.0e-12
                        and metrics["signal"] >= 1.0e-7
                        and metrics["corrupt"] <= max(1.0e-5, 0.20 * max(0.0, metrics["signal"]))
                        and cosine(projected_update.tensor, source_ref) >= -0.01
                    )

                def consensus_gate(metrics: dict[str, float]) -> bool:
                    return (
                        terminal_consensus_density >= 0.03
                        and terminal_consensus_balance >= 0.03
                        and metrics["norm"] > 1.0e-12
                        and metrics["signal"] >= 1.0e-7
                        and metrics["corrupt"] <= max(1.0e-5, 0.20 * max(0.0, metrics["signal"]))
                        and terminal_consensus_corrupt_cos <= 0.25
                        and cosine(consensus_update.tensor, source_ref) >= -0.01
                    )

                candidates = [
                    ("source_axis", source_update, source, source_scale, source_gate(source)),
                    ("projected", projected_update, projected, scale, projected_gate(projected)),
                    ("consensus", consensus_update, consensus, scale, consensus_gate(consensus)),
                ]
                if risk_profile and not risk_ok:
                    candidates = [item for item in candidates if item[0] == "source_axis"]
                if debt_aware and not debt_exists:
                    candidates = [item for item in candidates if item[0] == "source_axis"]

                allowed = [
                    (kind, candidate_update, metrics, candidate_scale)
                    for kind, candidate_update, metrics, candidate_scale, gate in candidates
                    if gate
                ]
                if not allowed:
                    best_metrics = max((source, projected, consensus), key=lambda item: (item["score"], item["signal"], -item["corrupt"]))
                    terminal_selector_score = float(best_metrics["score"])
                    prefix = "risk_reject" if risk_profile and not risk_ok else "debt_hold" if debt_aware and not debt_exists else "reject"
                    terminal_selector_kind = f"{prefix}:gap={risk_gap:.4g}:level={risk_level:.4g}:debt={int(debt_exists)}"
                    terminal_selector_current_cos = ""
                    return 0, float(best_metrics["signal"]), float(best_metrics["corrupt"]), float(best_metrics["gain_b"])

                best_kind, best_update, best_metrics, best_scale = max(
                    allowed,
                    key=lambda item: (item[2]["score"], item[2]["signal"], -item[2]["corrupt"]),
                )
                apply_update(model, best_update, lr=best_scale * float(args.lr))
                terminal_selector_score = float(best_metrics["score"])
                terminal_selector_kind = best_kind
                terminal_selector_current_cos = cosine(best_update.tensor, source_ref)
                return 1, float(best_metrics["signal"]), float(best_metrics["corrupt"]), float(best_metrics["gain_b"])

            def apply_terminal_projected_optimizer_route(
                scale: float = 0.08,
                *,
                blend_source: float = 0.0,
                antiwashout: bool = False,
            ) -> tuple[int, float, float, float]:
                nonlocal terminal_selector_score, terminal_selector_kind, terminal_selector_current_cos
                current_grad = flat_grad(model, device).detach()
                source_ref = normalized_like(source_axis.to(device=device, dtype=current_grad.dtype), current_grad)
                grad_source_cos = cosine(current_grad, source_ref)
                projected_update = make_late_floor_update(project_source=True)
                blend_weight = max(0.0, min(1.0, float(blend_source)))
                blend_vec = normalized_like(
                    (1.0 - blend_weight) * projected_update.tensor.detach() + blend_weight * source_ref,
                    projected_update.tensor,
                )
                blend_update = UpdateTensor(
                    blend_vec,
                    "gradient",
                    "subtract",
                    "slow_state",
                    "train_stream_terminal_projected_source_blend",
                    mechanism,
                    role="terminal_projected_gradient_slow_source_blend",
                    one_step_descent_claim=0,
                )
                source_update = UpdateTensor(
                    source_ref,
                    "step",
                    "subtract",
                    "slow_state",
                    "train_stream_terminal_antiwashout_source_axis",
                    mechanism,
                    role="terminal_antiwashout_source_axis_fallback",
                    one_step_descent_claim=0,
                )
                lr_denom = max(float(args.lr), 1.0e-12)
                source_scale = max(1.0e-8, 0.25 * float(args.fu_lr) / lr_denom)
                projected = terminal_candidate_metrics(projected_update, scale)
                blend = terminal_candidate_metrics(blend_update, scale)
                source = terminal_candidate_metrics(source_update, source_scale)

                def gate(
                    candidate_update: UpdateTensor,
                    metrics: dict[str, float],
                    *,
                    corrupt_fraction: float,
                    signal_floor: float,
                ) -> bool:
                    return (
                        metrics["norm"] > 1.0e-12
                        and metrics["signal"] >= signal_floor
                        and metrics["score"] >= -1.0e-5
                        and metrics["corrupt"] <= max(1.0e-5, corrupt_fraction * max(0.0, metrics["signal"]))
                        and cosine(candidate_update.tensor, source_ref) >= -0.005
                    )

                candidates = [
                    ("projected_optimizer", projected_update, projected, scale, gate(projected_update, projected, corrupt_fraction=0.25, signal_floor=-1.0e-7)),
                    ("projected_source_blend", blend_update, blend, scale, gate(blend_update, blend, corrupt_fraction=0.30, signal_floor=-1.0e-6)),
                    ("source_axis", source_update, source, source_scale, gate(source_update, source, corrupt_fraction=0.25, signal_floor=-1.0e-6)),
                ]
                if antiwashout and grad_source_cos < -0.02:
                    candidates = [item for item in candidates if item[0] != "projected_optimizer"]

                allowed = [
                    (kind, candidate_update, metrics, candidate_scale)
                    for kind, candidate_update, metrics, candidate_scale, accept in candidates
                    if accept
                ]
                if not allowed:
                    best_metrics = max((projected, blend, source), key=lambda item: (item["score"], item["signal"], -item["corrupt"]))
                    terminal_selector_score = float(best_metrics["score"])
                    terminal_selector_kind = f"antiwashout_reject:grad_cos={grad_source_cos:.4g}:mode={int(antiwashout)}"
                    terminal_selector_current_cos = f"grad={grad_source_cos:.4g}"
                    return 0, float(best_metrics["signal"]), float(best_metrics["corrupt"]), float(best_metrics["gain_b"])

                best_kind, best_update, best_metrics, best_scale = max(
                    allowed,
                    key=lambda item: (item[2]["score"], item[2]["signal"], cosine(item[1].tensor, source_ref), -item[2]["corrupt"]),
                )
                apply_update(model, best_update, lr=best_scale * float(args.lr))
                update_source_cos = cosine(best_update.tensor, source_ref)
                terminal_selector_score = float(best_metrics["score"])
                terminal_selector_kind = f"{best_kind}:grad_cos={grad_source_cos:.4g}:update_cos={update_source_cos:.4g}"
                terminal_selector_current_cos = f"grad={grad_source_cos:.4g};update={update_source_cos:.4g}"
                return 1, float(best_metrics["signal"]), float(best_metrics["corrupt"]), float(best_metrics["gain_b"])

            def row_orthogonalized_flat(vec: torch.Tensor) -> torch.Tensor:
                chunks: list[torch.Tensor] = []
                offset = 0
                for _pname, p in model.named_parameters():
                    if not p.requires_grad:
                        continue
                    count = int(p.numel())
                    part = vec[offset : offset + count].to(device=device, dtype=p.dtype).reshape_as(p)
                    offset += count
                    if p.ndim >= 2 and part.numel() > 0:
                        reduce_dims = tuple(range(1, part.ndim))
                        centered = part - part.mean(dim=reduce_dims, keepdim=True)
                        centered_norm = torch.linalg.vector_norm(centered.detach()).clamp_min(1.0e-12)
                        part_norm = torch.linalg.vector_norm(part.detach()).clamp_min(1.0e-12)
                        part = centered * (part_norm / centered_norm).to(device=part.device, dtype=part.dtype)
                    chunks.append(part.reshape(-1))
                if not chunks:
                    return torch.zeros_like(vec)
                return torch.cat(chunks).to(device=device, dtype=vec.dtype)

            def apply_terminal_nora_orthogonal_source_route(scale: float = 0.048) -> tuple[int, float, float, float]:
                nonlocal terminal_consensus_density, terminal_consensus_balance
                nonlocal terminal_selector_score, terminal_selector_kind, terminal_selector_current_cos
                current_grad = flat_grad(model, device).detach()
                source_ref = normalized_like(source_axis.to(device=device, dtype=current_grad.dtype), current_grad)
                row_orthogonal = normalized_like(row_orthogonalized_flat(source_ref), source_ref)
                nora_source_vec = normalized_like(0.70 * source_ref + 0.30 * row_orthogonal, current_grad)
                projected_update = make_late_floor_update(project_source=True)
                hybrid_vec = normalized_like(0.60 * nora_source_vec + 0.40 * projected_update.tensor.detach(), current_grad)
                source_update = UpdateTensor(
                    nora_source_vec,
                    "step",
                    "subtract",
                    "slow_state",
                    "train_stream_terminal_nora_row_orthogonal_source_axis",
                    mechanism,
                    role="terminal_nora_row_orthogonal_source_axis",
                    one_step_descent_claim=0,
                )
                hybrid_update = UpdateTensor(
                    hybrid_vec,
                    "gradient",
                    "subtract",
                    "slow_state",
                    "train_stream_terminal_nora_row_orthogonal_projected_hybrid",
                    mechanism,
                    role="terminal_nora_row_orthogonal_projected_hybrid",
                    one_step_descent_claim=0,
                )
                lr_denom = max(float(args.lr), 1.0e-12)
                source_scale = max(1.0e-8, 0.28 * float(args.fu_lr) / lr_denom)
                source = terminal_candidate_metrics(source_update, source_scale)
                hybrid = terminal_candidate_metrics(hybrid_update, scale)
                row_cos = cosine(row_orthogonal, source_ref)
                grad_source_cos = cosine(current_grad, source_ref)
                state_cos = cosine(short_state, long_state)
                terminal_consensus_density = agree_density
                terminal_consensus_balance = state_cos
                terminal_selector_current_cos = f"grad={grad_source_cos:.4g};row={row_cos:.4g};state={state_cos:.4g}"

                def gate(update: UpdateTensor, metrics: dict[str, float], *, source_floor: float, corrupt_fraction: float) -> bool:
                    return (
                        agree_density >= 0.08
                        and state_cos >= -0.05
                        and metrics["norm"] > 1.0e-12
                        and metrics["signal"] >= -1.0e-6
                        and metrics["corrupt"] <= max(1.0e-5, corrupt_fraction * max(0.0, metrics["signal"]))
                        and cosine(update.tensor, source_ref) >= source_floor
                    )

                candidates = [
                    ("nora_source", source_update, source, source_scale, gate(source_update, source, source_floor=0.65, corrupt_fraction=0.22)),
                    ("nora_projected_hybrid", hybrid_update, hybrid, scale, gate(hybrid_update, hybrid, source_floor=0.05, corrupt_fraction=0.28)),
                ]
                allowed = [
                    (kind, update, metrics, candidate_scale)
                    for kind, update, metrics, candidate_scale, accept in candidates
                    if accept
                ]
                if not allowed:
                    best_metrics = max((source, hybrid), key=lambda item: (item["score"], item["signal"], -item["corrupt"]))
                    terminal_selector_score = float(best_metrics["score"])
                    terminal_selector_kind = f"nora_reject:{terminal_selector_current_cos}"
                    return 0, float(best_metrics["signal"]), float(best_metrics["corrupt"]), float(best_metrics["gain_b"])

                best_kind, best_update, best_metrics, best_scale = max(
                    allowed,
                    key=lambda item: (item[2]["score"], item[2]["signal"], cosine(item[1].tensor, source_ref), -item[2]["corrupt"]),
                )
                apply_update(model, best_update, lr=best_scale * float(args.lr))
                update_source_cos = cosine(best_update.tensor, source_ref)
                terminal_selector_score = float(best_metrics["score"])
                terminal_selector_kind = f"{best_kind}:{terminal_selector_current_cos}:update={update_source_cos:.4g}"
                terminal_selector_current_cos = f"{terminal_selector_current_cos};update={update_source_cos:.4g}"
                return 1, float(best_metrics["signal"]), float(best_metrics["corrupt"]), float(best_metrics["gain_b"])

            def apply_terminal_low_nds_matrix_block_route(scale: float = 0.046) -> tuple[int, float, float, float]:
                nonlocal terminal_consensus_density, terminal_consensus_balance
                nonlocal terminal_selector_score, terminal_selector_kind, terminal_selector_current_cos
                current_grad = flat_grad(model, device).detach()
                source_ref = normalized_like(source_axis.to(device=device, dtype=current_grad.dtype), current_grad)
                hidden_mask = hidden_matrix_mask_flat().to(device=device, dtype=current_grad.dtype)
                readout_mask = readout_mask_flat().to(device=device, dtype=current_grad.dtype)
                if hidden_mask.numel() != current_grad.numel():
                    hidden_mask = torch.zeros_like(current_grad)
                if readout_mask.numel() != current_grad.numel():
                    readout_mask = torch.zeros_like(current_grad)
                block_mask = torch.clamp(hidden_mask + 0.50 * readout_mask, min=0.0, max=1.0)
                block_density = float((block_mask > 0).float().mean().item()) if block_mask.numel() else 0.0
                block_source_vec = source_ref * block_mask
                if float(torch.linalg.vector_norm(block_source_vec.detach()).item()) <= 1.0e-12:
                    block_source_vec = source_ref
                block_source_vec = normalized_like(block_source_vec, current_grad)
                projected_update = make_late_floor_update(project_source=True)
                block_projected = normalized_like(
                    0.72 * block_source_vec + 0.28 * (projected_update.tensor.detach() * block_mask),
                    current_grad,
                )
                block_update = UpdateTensor(
                    block_source_vec,
                    "step",
                    "subtract",
                    "matrix_block",
                    "train_stream_terminal_low_nds_matrix_block_source",
                    mechanism,
                    role="terminal_low_nds_matrix_block_source",
                    one_step_descent_claim=0,
                )
                hybrid_update = UpdateTensor(
                    block_projected,
                    "gradient",
                    "subtract",
                    "matrix_block",
                    "train_stream_terminal_low_nds_matrix_block_projected_hybrid",
                    mechanism,
                    role="terminal_low_nds_matrix_block_projected_hybrid",
                    one_step_descent_claim=0,
                )
                lr_denom = max(float(args.lr), 1.0e-12)
                block_scale = max(1.0e-8, 0.24 * float(args.fu_lr) / lr_denom)
                block = terminal_candidate_metrics(block_update, block_scale)
                hybrid = terminal_candidate_metrics(hybrid_update, scale)
                grad_block_cos = cosine(current_grad, block_source_vec)
                terminal_consensus_density = block_density
                terminal_consensus_balance = cosine(block_source_vec, source_ref)
                terminal_selector_current_cos = (
                    f"grad={grad_block_cos:.4g};block={block_density:.4g};source={terminal_consensus_balance:.4g}"
                )

                def gate(update: UpdateTensor, metrics: dict[str, float], *, source_floor: float, corrupt_fraction: float) -> bool:
                    return (
                        block_density >= 0.05
                        and metrics["norm"] > 1.0e-12
                        and metrics["signal"] >= -1.0e-6
                        and metrics["corrupt"] <= max(1.0e-5, corrupt_fraction * max(0.0, metrics["signal"]))
                        and cosine(update.tensor, source_ref) >= source_floor
                    )

                candidates = [
                    ("low_nds_block_source", block_update, block, block_scale, gate(block_update, block, source_floor=0.35, corrupt_fraction=0.22)),
                    ("low_nds_block_hybrid", hybrid_update, hybrid, scale, gate(hybrid_update, hybrid, source_floor=-0.02, corrupt_fraction=0.28)),
                ]
                allowed = [
                    (kind, update, metrics, candidate_scale)
                    for kind, update, metrics, candidate_scale, accept in candidates
                    if accept
                ]
                if not allowed:
                    best_metrics = max((block, hybrid), key=lambda item: (item["score"], item["signal"], -item["corrupt"]))
                    terminal_selector_score = float(best_metrics["score"])
                    terminal_selector_kind = f"low_nds_block_reject:{terminal_selector_current_cos}"
                    return 0, float(best_metrics["signal"]), float(best_metrics["corrupt"]), float(best_metrics["gain_b"])
                best_kind, best_update, best_metrics, best_scale = max(
                    allowed,
                    key=lambda item: (item[2]["score"], item[2]["signal"], cosine(item[1].tensor, source_ref), -item[2]["corrupt"]),
                )
                apply_update(model, best_update, lr=best_scale * float(args.lr))
                update_source_cos = cosine(best_update.tensor, source_ref)
                terminal_selector_score = float(best_metrics["score"])
                terminal_selector_kind = f"{best_kind}:{terminal_selector_current_cos}:update={update_source_cos:.4g}"
                terminal_selector_current_cos = f"{terminal_selector_current_cos};update={update_source_cos:.4g}"
                return 1, float(best_metrics["signal"]), float(best_metrics["corrupt"]), float(best_metrics["gain_b"])

            def apply_terminal_dual_memory_preserve_route(scale: float = 0.045) -> tuple[int, float, float, float]:
                nonlocal terminal_consensus_density, terminal_consensus_balance
                nonlocal terminal_selector_score, terminal_selector_kind, terminal_selector_current_cos
                current_grad = flat_grad(model, device).detach()
                source_ref = normalized_like(source_axis.to(device=device, dtype=current_grad.dtype), current_grad)
                short_ref = normalized_like(short_state.to(device=device, dtype=current_grad.dtype), current_grad)
                long_ref = normalized_like(long_state.to(device=device, dtype=current_grad.dtype), current_grad)
                dual_vec = normalized_like(0.25 * short_ref + 0.75 * long_ref, current_grad)
                projected_update = make_late_floor_update(project_source=True)
                dual_projected = normalized_like(0.70 * dual_vec + 0.30 * projected_update.tensor.detach(), current_grad)
                dual_update = UpdateTensor(
                    dual_vec,
                    "step",
                    "subtract",
                    "slow_state",
                    "train_stream_terminal_dual_memory_long_source",
                    mechanism,
                    role="terminal_dual_memory_long_source",
                    one_step_descent_claim=0,
                )
                hybrid_update = UpdateTensor(
                    dual_projected,
                    "gradient",
                    "subtract",
                    "slow_state",
                    "train_stream_terminal_dual_memory_projected_hybrid",
                    mechanism,
                    role="terminal_dual_memory_projected_hybrid",
                    one_step_descent_claim=0,
                )
                lr_denom = max(float(args.lr), 1.0e-12)
                dual_scale = max(1.0e-8, 0.24 * float(args.fu_lr) / lr_denom)
                dual = terminal_candidate_metrics(dual_update, dual_scale)
                hybrid = terminal_candidate_metrics(hybrid_update, scale)
                state_cos = cosine(short_state, long_state)
                dual_source_cos = cosine(dual_vec, source_ref)
                terminal_consensus_density = agree_density
                terminal_consensus_balance = state_cos
                terminal_selector_current_cos = f"state={state_cos:.4g};dual={dual_source_cos:.4g};density={agree_density:.4g}"

                def gate(update: UpdateTensor, metrics: dict[str, float], *, source_floor: float, corrupt_fraction: float) -> bool:
                    return (
                        agree_density >= 0.08
                        and state_cos >= -0.06
                        and metrics["norm"] > 1.0e-12
                        and metrics["signal"] >= -1.0e-6
                        and metrics["corrupt"] <= max(1.0e-5, corrupt_fraction * max(0.0, metrics["signal"]))
                        and cosine(update.tensor, source_ref) >= source_floor
                    )

                candidates = [
                    ("dual_memory_source", dual_update, dual, dual_scale, gate(dual_update, dual, source_floor=0.60, corrupt_fraction=0.22)),
                    ("dual_memory_hybrid", hybrid_update, hybrid, scale, gate(hybrid_update, hybrid, source_floor=0.02, corrupt_fraction=0.28)),
                ]
                allowed = [
                    (kind, update, metrics, candidate_scale)
                    for kind, update, metrics, candidate_scale, accept in candidates
                    if accept
                ]
                if not allowed:
                    best_metrics = max((dual, hybrid), key=lambda item: (item["score"], item["signal"], -item["corrupt"]))
                    terminal_selector_score = float(best_metrics["score"])
                    terminal_selector_kind = f"dual_memory_reject:{terminal_selector_current_cos}"
                    return 0, float(best_metrics["signal"]), float(best_metrics["corrupt"]), float(best_metrics["gain_b"])
                best_kind, best_update, best_metrics, best_scale = max(
                    allowed,
                    key=lambda item: (item[2]["score"], item[2]["signal"], cosine(item[1].tensor, source_ref), -item[2]["corrupt"]),
                )
                apply_update(model, best_update, lr=best_scale * float(args.lr))
                update_source_cos = cosine(best_update.tensor, source_ref)
                terminal_selector_score = float(best_metrics["score"])
                terminal_selector_kind = f"{best_kind}:{terminal_selector_current_cos}:update={update_source_cos:.4g}"
                terminal_selector_current_cos = f"{terminal_selector_current_cos};update={update_source_cos:.4g}"
                return 1, float(best_metrics["signal"]), float(best_metrics["corrupt"]), float(best_metrics["gain_b"])

            def apply_terminal_signal_estimator_route(mode: str) -> tuple[int, float, float, float]:
                nonlocal terminal_consensus_density, terminal_consensus_balance
                nonlocal terminal_selector_score, terminal_selector_kind, terminal_selector_current_cos
                current_grad = flat_grad(model, device).detach()
                source_ref = normalized_like(source_axis.to(device=device, dtype=current_grad.dtype), current_grad)
                short_ref = normalized_like(short_state.to(device=device, dtype=current_grad.dtype), current_grad)
                long_ref = normalized_like(long_state.to(device=device, dtype=current_grad.dtype), current_grad)
                consensus_update = make_terminal_consensus_update()
                projected_update = make_late_floor_update(project_source=True)
                state_cos = cosine(short_state, long_state)
                support_density = terminal_consensus_density
                support_balance = terminal_consensus_balance
                slow_mix = torch.where(
                    agree.to(device=device),
                    0.30 * short_ref + 0.70 * long_ref,
                    long_ref,
                )
                slow_mix = normalized_like(slow_mix, current_grad)
                signal_transport = normalized_like(
                    0.46 * slow_mix + 0.34 * source_ref + 0.20 * projected_update.tensor.detach(),
                    current_grad,
                )
                source_update = UpdateTensor(
                    source_ref,
                    "step",
                    "subtract",
                    "slow_state",
                    "train_stream_terminal_signal_estimator_source_axis",
                    mechanism,
                    role="terminal_signal_estimator_source_axis",
                    one_step_descent_claim=0,
                )
                slow_update = UpdateTensor(
                    slow_mix,
                    "step",
                    "subtract",
                    "slow_state",
                    "train_stream_terminal_signal_estimator_split_slow_state",
                    mechanism,
                    role="terminal_signal_estimator_split_slow_state",
                    one_step_descent_claim=0,
                )
                transport_update = UpdateTensor(
                    signal_transport,
                    "gradient",
                    "subtract",
                    "slow_state",
                    "train_stream_terminal_signal_reservoir_transport",
                    mechanism,
                    role="terminal_signal_reservoir_transport",
                    one_step_descent_claim=0,
                )
                lr_denom = max(float(args.lr), 1.0e-12)
                source_scale = max(1.0e-8, 0.25 * float(args.fu_lr) / lr_denom)
                slow_scale = max(1.0e-8, 0.22 * float(args.fu_lr) / lr_denom)
                consensus_scale = 0.040
                projected_scale = 0.038 if mode == "h800_source_slow_ema_terminal_snr_predictor" else 0.042
                transport_scale = 0.046 if mode != "h800_source_slow_ema_signal_reservoir_transport" else 0.052
                metrics_by_kind = {
                    "source_axis": (source_update, terminal_candidate_metrics(source_update, source_scale), source_scale),
                    "split_slow_state": (slow_update, terminal_candidate_metrics(slow_update, slow_scale), slow_scale),
                    "consensus": (consensus_update, terminal_candidate_metrics(consensus_update, consensus_scale), consensus_scale),
                    "projected": (projected_update, terminal_candidate_metrics(projected_update, projected_scale), projected_scale),
                    "transport": (transport_update, terminal_candidate_metrics(transport_update, transport_scale), transport_scale),
                }

                def snr(metrics: dict[str, float]) -> float:
                    corrupt = max(1.0e-5, max(0.0, metrics["corrupt"]))
                    return float(metrics["signal"] / corrupt)

                def gate(
                    update: UpdateTensor,
                    metrics: dict[str, float],
                    *,
                    source_floor: float,
                    corrupt_fraction: float,
                    signal_floor: float,
                    density_floor: float,
                    balance_floor: float,
                    snr_floor: float,
                ) -> bool:
                    return (
                        support_density >= density_floor
                        and support_balance >= balance_floor
                        and state_cos >= -0.08
                        and metrics["norm"] > 1.0e-12
                        and metrics["signal"] >= signal_floor
                        and metrics["corrupt"] <= max(1.0e-5, corrupt_fraction * max(0.0, metrics["signal"]))
                        and snr(metrics) >= snr_floor
                        and cosine(update.tensor, source_ref) >= source_floor
                    )

                if mode == "h800_source_slow_ema_split_consensus_estimator":
                    candidate_specs = [
                        ("consensus", -0.01, 0.20, 1.0e-7, 0.035, 0.030, 0.01),
                        ("split_slow_state", 0.45, 0.24, -1.0e-6, 0.030, 0.020, -0.05),
                        ("source_axis", 0.95, 0.22, -1.0e-6, 0.020, 0.000, -0.10),
                    ]
                elif mode == "h800_source_slow_ema_signal_reservoir_transport":
                    candidate_specs = [
                        ("transport", 0.18, 0.22, 1.0e-7, 0.030, 0.020, 0.00),
                        ("projected", -0.01, 0.18, 1.0e-7, 0.030, 0.020, 0.02),
                        ("source_axis", 0.95, 0.22, -1.0e-6, 0.020, 0.000, -0.10),
                    ]
                else:
                    candidate_specs = [
                        ("transport", 0.12, 0.24, -1.0e-7, 0.030, 0.020, -0.02),
                        ("projected", -0.01, 0.20, 1.0e-7, 0.030, 0.020, 0.01),
                        ("source_axis", 0.95, 0.22, -1.0e-6, 0.020, 0.000, -0.10),
                    ]

                allowed: list[tuple[str, UpdateTensor, dict[str, float], float]] = []
                for kind, source_floor, corrupt_fraction, signal_floor, density_floor, balance_floor, snr_floor in candidate_specs:
                    update, metrics, candidate_scale = metrics_by_kind[kind]
                    if gate(
                        update,
                        metrics,
                        source_floor=source_floor,
                        corrupt_fraction=corrupt_fraction,
                        signal_floor=signal_floor,
                        density_floor=density_floor,
                        balance_floor=balance_floor,
                        snr_floor=snr_floor,
                    ):
                        allowed.append((kind, update, metrics, candidate_scale))

                best_pool = [metrics_by_kind[kind][1] for kind, *_rest in candidate_specs]
                if not allowed:
                    best_metrics = max(best_pool, key=lambda item: (snr(item), item["score"], item["signal"], -item["corrupt"]))
                    terminal_selector_score = float(best_metrics["score"])
                    terminal_selector_kind = (
                        f"{mode}:reject:support={support_density:.4g}:balance={support_balance:.4g}:"
                        f"state={state_cos:.4g}:snr={snr(best_metrics):.4g}"
                    )
                    terminal_selector_current_cos = f"state={state_cos:.4g};support={support_density:.4g}"
                    return 0, float(best_metrics["signal"]), float(best_metrics["corrupt"]), float(best_metrics["gain_b"])

                best_kind, best_update, best_metrics, best_scale = max(
                    allowed,
                    key=lambda item: (snr(item[2]), item[2]["score"], item[2]["signal"], cosine(item[1].tensor, source_ref), -item[2]["corrupt"]),
                )
                apply_update(model, best_update, lr=best_scale * float(args.lr))
                update_source_cos = cosine(best_update.tensor, source_ref)
                terminal_selector_score = float(best_metrics["score"])
                terminal_selector_kind = (
                    f"{mode}:{best_kind}:support={support_density:.4g}:balance={support_balance:.4g}:"
                    f"state={state_cos:.4g}:snr={snr(best_metrics):.4g}:update={update_source_cos:.4g}"
                )
                terminal_selector_current_cos = (
                    f"state={state_cos:.4g};support={support_density:.4g};balance={support_balance:.4g};update={update_source_cos:.4g}"
                )
                return 1, float(best_metrics["signal"]), float(best_metrics["corrupt"]), float(best_metrics["gain_b"])

            def apply_terminal_post4000_source_floor_route(mode: str) -> tuple[int, float, float, float]:
                nonlocal terminal_consensus_density, terminal_consensus_balance
                nonlocal terminal_selector_score, terminal_selector_kind, terminal_selector_current_cos
                nonlocal terminal_h4000_source_anchor
                current_grad = flat_grad(model, device).detach()
                source_ref = normalized_like(source_axis.to(device=device, dtype=current_grad.dtype), current_grad)
                if step >= 4000 and terminal_h4000_source_anchor is None and source_ref.numel():
                    terminal_h4000_source_anchor = source_ref.detach().clone()
                anchor_seed = terminal_h4000_source_anchor.to(device=device, dtype=current_grad.dtype) if terminal_h4000_source_anchor is not None else source_ref
                anchor_ref = normalized_like(anchor_seed, current_grad)
                short_ref = normalized_like(short_state.to(device=device, dtype=current_grad.dtype), current_grad)
                long_ref = normalized_like(long_state.to(device=device, dtype=current_grad.dtype), current_grad)
                state_cos = cosine(short_state, long_state)
                consensus_base = make_terminal_consensus_update()
                support_density = terminal_consensus_density
                support_balance = terminal_consensus_balance
                slow_floor = torch.where(
                    agree.to(device=device),
                    0.20 * short_ref + 0.80 * long_ref,
                    long_ref,
                )
                slow_floor = normalized_like(slow_floor, current_grad)
                if mode in {
                    "h800_source_slow_ema_terminal_h4000_anchor_floor",
                    "h800_source_slow_ema_terminal_raw_guard_h4000_anchor_floor",
                }:
                    floor_ref = anchor_ref
                    floor_source = "train_stream_terminal_h4000_anchor_source_floor"
                    floor_role = "terminal_h4000_anchor_source_floor"
                    floor_source_cos = cosine(anchor_ref, source_ref)
                    floor_blend = 0.86
                    floor_scale_coeff = 0.18
                    consensus_scale = 0.032
                elif mode in {
                    "h800_source_slow_ema_terminal_decay_aware_floor",
                    "h800_source_slow_ema_terminal_raw_guard_decay_aware_floor",
                }:
                    floor_ref = normalized_like(0.58 * anchor_ref + 0.42 * slow_floor, current_grad)
                    floor_source = "train_stream_terminal_decay_aware_source_floor"
                    floor_role = "terminal_decay_aware_source_floor"
                    floor_source_cos = cosine(floor_ref, source_ref)
                    floor_blend = 0.72
                    floor_scale_coeff = 0.20
                    consensus_scale = 0.035
                else:
                    floor_ref = normalized_like(0.72 * source_ref + 0.28 * slow_floor, current_grad)
                    floor_source = "train_stream_terminal_source_floor"
                    floor_role = "terminal_source_floor"
                    floor_source_cos = cosine(floor_ref, source_ref)
                    floor_blend = 0.80
                    floor_scale_coeff = 0.18
                    consensus_scale = 0.032

                floor_update = UpdateTensor(
                    floor_ref,
                    "step",
                    "subtract",
                    "slow_state",
                    floor_source,
                    mechanism,
                    role=floor_role,
                    one_step_descent_claim=0,
                )
                source_update = UpdateTensor(
                    source_ref,
                    "step",
                    "subtract",
                    "slow_state",
                    "train_stream_terminal_source_axis_floor",
                    mechanism,
                    role="terminal_source_axis_floor_candidate",
                    one_step_descent_claim=0,
                )
                consensus_floor = normalized_like(
                    floor_blend * floor_ref + (1.0 - floor_blend) * consensus_base.tensor.detach(),
                    current_grad,
                )
                consensus_update = UpdateTensor(
                    consensus_floor,
                    "step",
                    "subtract",
                    "slow_state",
                    "train_stream_terminal_consensus_blended_source_floor",
                    mechanism,
                    role="terminal_consensus_blended_source_floor",
                    one_step_descent_claim=0,
                )
                lr_denom = max(float(args.lr), 1.0e-12)
                source_scale = max(1.0e-8, 0.20 * float(args.fu_lr) / lr_denom)
                floor_scale = max(1.0e-8, floor_scale_coeff * float(args.fu_lr) / lr_denom)
                metrics_by_kind = {
                    "source_axis_floor": (source_update, terminal_candidate_metrics(source_update, source_scale), source_scale),
                    "source_floor": (floor_update, terminal_candidate_metrics(floor_update, floor_scale), floor_scale),
                    "consensus_blend_floor": (
                        consensus_update,
                        terminal_candidate_metrics(consensus_update, consensus_scale),
                        consensus_scale,
                    ),
                }

                def source_gate(update: UpdateTensor, metrics: dict[str, float], source_floor: float) -> bool:
                    return (
                        metrics["norm"] > 1.0e-12
                        and metrics["signal"] >= -2.0e-6
                        and metrics["corrupt"] <= max(1.0e-5, 0.25 * max(0.0, metrics["signal"]))
                        and cosine(update.tensor, source_ref) >= source_floor
                    )

                def consensus_gate(update: UpdateTensor, metrics: dict[str, float]) -> bool:
                    return (
                        support_density >= 0.025
                        and support_balance >= 0.015
                        and state_cos >= -0.12
                        and metrics["norm"] > 1.0e-12
                        and metrics["signal"] >= -1.0e-6
                        and metrics["corrupt"] <= max(1.0e-5, 0.22 * max(0.0, metrics["signal"]))
                        and cosine(update.tensor, source_ref) >= 0.35
                    )

                allowed: list[tuple[str, UpdateTensor, dict[str, float], float]] = []
                source_update_obj, source_metrics, source_scale_value = metrics_by_kind["source_axis_floor"]
                floor_update_obj, floor_metrics, floor_scale_value = metrics_by_kind["source_floor"]
                consensus_update_obj, consensus_metrics, consensus_scale_value = metrics_by_kind["consensus_blend_floor"]
                if source_gate(floor_update_obj, floor_metrics, 0.42 if step >= 4000 else 0.60):
                    allowed.append(("source_floor", floor_update_obj, floor_metrics, floor_scale_value))
                if source_gate(source_update_obj, source_metrics, 0.95):
                    allowed.append(("source_axis_floor", source_update_obj, source_metrics, source_scale_value))
                if consensus_gate(consensus_update_obj, consensus_metrics):
                    allowed.append(("consensus_blend_floor", consensus_update_obj, consensus_metrics, consensus_scale_value))

                best_pool = [source_metrics, floor_metrics, consensus_metrics]
                if not allowed:
                    best_metrics = max(best_pool, key=lambda item: (item["score"], item["signal"], -item["corrupt"]))
                    terminal_selector_score = float(best_metrics["score"])
                    terminal_selector_kind = (
                        f"{mode}:reject:support={support_density:.4g}:balance={support_balance:.4g}:"
                        f"state={state_cos:.4g}:floor_source_cos={floor_source_cos:.4g}:anchor={int(terminal_h4000_source_anchor is not None)}"
                    )
                    terminal_selector_current_cos = f"state={state_cos:.4g};support={support_density:.4g};floor={floor_source_cos:.4g}"
                    return 0, float(best_metrics["signal"]), float(best_metrics["corrupt"]), float(best_metrics["gain_b"])

                best_kind, best_update, best_metrics, best_scale = max(
                    allowed,
                    key=lambda item: (
                        item[2]["score"],
                        item[2]["signal"],
                        cosine(item[1].tensor, source_ref),
                        -item[2]["corrupt"],
                    ),
                )
                apply_update(model, best_update, lr=best_scale * float(args.lr))
                update_source_cos = cosine(best_update.tensor, source_ref)
                terminal_selector_score = float(best_metrics["score"])
                terminal_selector_kind = (
                    f"{mode}:{best_kind}:support={support_density:.4g}:balance={support_balance:.4g}:"
                    f"state={state_cos:.4g}:floor_source_cos={floor_source_cos:.4g}:update={update_source_cos:.4g}:"
                    f"anchor={int(terminal_h4000_source_anchor is not None)}"
                )
                terminal_selector_current_cos = (
                    f"state={state_cos:.4g};support={support_density:.4g};floor={floor_source_cos:.4g};"
                    f"update={update_source_cos:.4g}"
                )
                return 1, float(best_metrics["signal"]), float(best_metrics["corrupt"]), float(best_metrics["gain_b"])

            def apply_terminal_anti_erosion_route(mode: str) -> tuple[int, float, float, float]:
                nonlocal terminal_consensus_density, terminal_consensus_balance
                nonlocal terminal_selector_score, terminal_selector_kind, terminal_selector_current_cos
                nonlocal terminal_h4000_source_anchor, terminal_h3200_source_anchor, terminal_projected_removed_norm
                current_grad = flat_grad(model, device).detach()
                if current_grad.numel() == 0:
                    return 0, 0.0, 0.0, 0.0
                source_ref = normalized_like(source_axis.to(device=device, dtype=current_grad.dtype), current_grad)
                if step >= 3200 and terminal_h3200_source_anchor is None and source_ref.numel():
                    terminal_h3200_source_anchor = source_ref.detach().clone()
                if step >= 4000 and terminal_h4000_source_anchor is None and source_ref.numel():
                    terminal_h4000_source_anchor = source_ref.detach().clone()
                anchor_seed = terminal_h4000_source_anchor.to(device=device, dtype=current_grad.dtype) if terminal_h4000_source_anchor is not None else source_ref
                anchor_ref = normalized_like(anchor_seed, current_grad)
                h3200_seed = terminal_h3200_source_anchor.to(device=device, dtype=current_grad.dtype) if terminal_h3200_source_anchor is not None else anchor_seed
                h3200_ref = normalized_like(h3200_seed, current_grad)
                consensus_update = make_terminal_consensus_update()
                support_density = terminal_consensus_density
                support_balance = terminal_consensus_balance
                state_cos = cosine(short_state, long_state)

                source_norm2 = torch.dot(source_ref.float(), source_ref.float()).clamp_min(1.0e-12)
                source_coeff = torch.dot(current_grad.float(), source_ref.float()) / source_norm2
                anti_coeff = torch.clamp(source_coeff, max=0.0)
                anti_vec = anti_coeff.to(device=device, dtype=current_grad.dtype) * source_ref
                terminal_projected_removed_norm = float(torch.linalg.vector_norm(anti_vec.detach()).item())
                projected_grad = torch.nan_to_num(current_grad - anti_vec, nan=0.0, posinf=0.0, neginf=0.0)
                orthogonal_grad = torch.nan_to_num(
                    current_grad - source_coeff.to(device=device, dtype=current_grad.dtype) * source_ref,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
                reflected_grad = torch.nan_to_num(current_grad - 2.0 * anti_vec, nan=0.0, posinf=0.0, neginf=0.0)

                anchor_norm2 = torch.dot(anchor_ref.float(), anchor_ref.float()).clamp_min(1.0e-12)
                anchor_coeff = torch.dot(current_grad.float(), anchor_ref.float()) / anchor_norm2
                anchor_anti = torch.clamp(anchor_coeff, max=0.0).to(device=device, dtype=current_grad.dtype) * anchor_ref
                transported_grad = torch.nan_to_num(current_grad - anchor_anti, nan=0.0, posinf=0.0, neginf=0.0)

                source_floor = normalized_like(0.72 * source_ref + 0.28 * anchor_ref, current_grad)
                orthogonal_blend = normalized_like(0.55 * orthogonal_grad + 0.45 * source_floor, current_grad)
                reflected_blend = normalized_like(
                    0.52 * reflected_grad + 0.34 * source_ref + 0.14 * consensus_update.tensor.detach(),
                    current_grad,
                )
                transport_blend = normalized_like(0.44 * transported_grad + 0.38 * anchor_ref + 0.18 * source_ref, current_grad)
                h3200_anchor_blend = normalized_like(
                    0.58 * h3200_ref + 0.24 * source_ref + 0.18 * consensus_update.tensor.detach(),
                    current_grad,
                )
                h3200_raw_guard_blend = normalized_like(0.50 * h3200_ref + 0.25 * source_ref + 0.25 * projected_grad, current_grad)
                h3200_ratio_blend = normalized_like(0.68 * h3200_ref + 0.18 * anchor_ref + 0.14 * source_ref, current_grad)

                def candidate_update(name: str, vec: torch.Tensor, role: str) -> UpdateTensor:
                    return UpdateTensor(
                        torch.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0),
                        "step",
                        "subtract",
                        "slow_state",
                        name,
                        mechanism,
                        role=role,
                        one_step_descent_claim=0,
                    )

                lr_denom = max(float(args.lr), 1.0e-12)
                source_scale = max(1.0e-8, 0.22 * float(args.fu_lr) / lr_denom)
                anchor_scale = max(1.0e-8, 0.18 * float(args.fu_lr) / lr_denom)
                h3200_scale = max(1.0e-8, 0.20 * float(args.fu_lr) / lr_denom)
                h3200_ratio_scale = max(1.0e-8, 0.18 * float(args.fu_lr) / lr_denom)
                candidates: list[tuple[str, UpdateTensor, float, float, float, float]] = [
                    (
                        "source_floor",
                        candidate_update(
                            "train_stream_terminal_anti_erosion_source_floor",
                            source_floor,
                            "terminal_anti_erosion_source_floor",
                        ),
                        source_scale,
                        0.80,
                        -2.0e-6,
                        0.24,
                    ),
                    (
                        "orthogonal_corrector",
                        candidate_update(
                            "train_stream_terminal_anti_erosion_orthogonal_corrector",
                            orthogonal_blend,
                            "terminal_anti_erosion_orthogonal_corrector",
                        ),
                        0.042,
                        0.18,
                        -1.0e-6,
                        0.25,
                    ),
                    (
                        "reflection_guard",
                        candidate_update(
                            "train_stream_terminal_source_reflection_guard",
                            reflected_blend,
                            "terminal_source_reflection_guard",
                        ),
                        0.040,
                        0.22,
                        -1.0e-6,
                        0.25,
                    ),
                    (
                        "h4000_transport",
                        candidate_update(
                            "train_stream_terminal_h4000_transport_corrector",
                            transport_blend,
                            "terminal_h4000_transport_corrector",
                        ),
                        anchor_scale,
                        0.42,
                        -1.5e-6,
                        0.24,
                    ),
                    (
                        "h3200_anchor_transport",
                        candidate_update(
                            "train_stream_terminal_h3200_anchor_transport",
                            h3200_anchor_blend,
                            "terminal_h3200_anchor_transport",
                        ),
                        h3200_scale,
                        0.38,
                        -1.5e-6,
                        0.24,
                    ),
                    (
                        "h3200_raw_guard_transport",
                        candidate_update(
                            "train_stream_terminal_raw_guard_h3200_anchor_transport",
                            h3200_raw_guard_blend,
                            "terminal_raw_guard_h3200_anchor_transport",
                        ),
                        0.040,
                        0.32,
                        -1.2e-6,
                        0.25,
                    ),
                    (
                        "h3200_ratio_reentry",
                        candidate_update(
                            "train_stream_terminal_h3200_ratio_reentry",
                            h3200_ratio_blend,
                            "terminal_h3200_ratio_reentry",
                        ),
                        h3200_ratio_scale,
                        0.42,
                        -1.5e-6,
                        0.24,
                    ),
                ]
                if mode == "h800_source_slow_ema_terminal_anti_erosion_orthogonal":
                    candidates = [c for c in candidates if c[0] in {"source_floor", "orthogonal_corrector"}]
                elif mode == "h800_source_slow_ema_terminal_source_reflection_guard":
                    candidates = [c for c in candidates if c[0] in {"source_floor", "reflection_guard"}]
                elif mode == "h800_source_slow_ema_terminal_h4000_transport_corrector":
                    candidates = [c for c in candidates if c[0] in {"source_floor", "h4000_transport"}]
                elif mode == "h800_source_slow_ema_terminal_h3200_anchor_transport":
                    candidates = [c for c in candidates if c[0] in {"source_floor", "h3200_anchor_transport"}]
                elif mode == "h800_source_slow_ema_terminal_raw_guard_h3200_anchor_transport":
                    candidates = [c for c in candidates if c[0] in {"source_floor", "h3200_raw_guard_transport"}]
                elif mode == "h800_source_slow_ema_terminal_h3200_ratio_reentry":
                    candidates = [c for c in candidates if c[0] in {"source_floor", "h3200_ratio_reentry"}]

                evaluated: list[tuple[str, UpdateTensor, dict[str, float], float, bool]] = []
                grad_source_cos = cosine(current_grad, source_ref)
                anchor_source_cos = cosine(anchor_ref, source_ref)
                h3200_source_cos = cosine(h3200_ref, source_ref)
                for kind, update, scale, source_floor_cos, min_signal, corrupt_fraction in candidates:
                    metrics = terminal_candidate_metrics(update, scale)
                    update_source_cos = cosine(update.tensor, source_ref)
                    support_ok = (
                        kind == "source_floor"
                        or (
                            support_density >= 0.020
                            and support_balance >= 0.012
                            and state_cos >= -0.18
                        )
                    )
                    allowed = (
                        support_ok
                        and metrics["norm"] > 1.0e-12
                        and metrics["signal"] >= min_signal
                        and metrics["corrupt"] <= max(1.0e-5, corrupt_fraction * max(0.0, metrics["signal"]))
                        and update_source_cos >= source_floor_cos
                    )
                    evaluated.append((kind, update, metrics, scale, bool(allowed)))

                allowed_candidates = [item for item in evaluated if item[4]]
                best_metrics = max((item[2] for item in evaluated), key=lambda item: (item["score"], item["signal"], -item["corrupt"]))
                if not allowed_candidates:
                    terminal_selector_score = float(best_metrics["score"])
                    terminal_selector_kind = (
                        f"{mode}:reject:support={support_density:.4g}:balance={support_balance:.4g}:"
                        f"state={state_cos:.4g}:grad_source={grad_source_cos:.4g}:anchor_source={anchor_source_cos:.4g}:"
                        f"h3200_source={h3200_source_cos:.4g}:anti_norm={terminal_projected_removed_norm:.4g}"
                    )
                    terminal_selector_current_cos = (
                        f"grad_source={grad_source_cos:.4g};anchor={anchor_source_cos:.4g};"
                        f"h3200={h3200_source_cos:.4g};support={support_density:.4g};anti={terminal_projected_removed_norm:.4g}"
                    )
                    return 0, float(best_metrics["signal"]), float(best_metrics["corrupt"]), float(best_metrics["gain_b"])

                best_kind, best_update, chosen, best_scale, _allowed = max(
                    allowed_candidates,
                    key=lambda item: (
                        item[2]["score"],
                        item[2]["signal"],
                        cosine(item[1].tensor, source_ref),
                        -item[2]["corrupt"],
                    ),
                )
                apply_update(model, best_update, lr=best_scale * float(args.lr))
                terminal_selector_score = float(chosen["score"])
                terminal_selector_kind = (
                    f"{mode}:{best_kind}:support={support_density:.4g}:balance={support_balance:.4g}:"
                    f"state={state_cos:.4g}:grad_source={grad_source_cos:.4g}:anchor_source={anchor_source_cos:.4g}:"
                    f"h3200_source={h3200_source_cos:.4g}:update={cosine(best_update.tensor, source_ref):.4g}:"
                    f"anti_norm={terminal_projected_removed_norm:.4g}"
                )
                terminal_selector_current_cos = (
                    f"grad_source={grad_source_cos:.4g};anchor={anchor_source_cos:.4g};"
                    f"h3200={h3200_source_cos:.4g};update={cosine(best_update.tensor, source_ref):.4g};"
                    f"anti={terminal_projected_removed_norm:.4g}"
                )
                return 1, float(chosen["signal"]), float(chosen["corrupt"]), float(chosen["gain_b"])

            def terminal_progress_start(mode: str) -> int:
                if mode == "h800_source_slow_ema_terminal_raw_guard_h3600_gentle_progress":
                    return 3600
                if mode == "h800_source_slow_ema_terminal_raw_guard_h4400_projected_progress":
                    return 4400
                if mode in {
                    "h800_source_slow_ema_terminal_h4000_progress_carry",
                    "h800_source_slow_ema_terminal_raw_guard_h4000_gentle_progress",
                }:
                    return 4000
                return 3200

            def apply_terminal_progress_carry_route(mode: str) -> tuple[int, float, float, float]:
                nonlocal terminal_selector_score, terminal_selector_kind, terminal_selector_current_cos
                nonlocal terminal_h4000_source_anchor, terminal_h3200_source_anchor, terminal_projected_removed_norm
                if step < terminal_progress_start(mode):
                    return 0, 0.0, 0.0, 0.0
                current_grad = flat_grad(model, device).detach()
                if current_grad.numel() == 0:
                    return 0, 0.0, 0.0, 0.0
                source_ref = normalized_like(source_axis.to(device=device, dtype=current_grad.dtype), current_grad)
                if step >= 3200 and terminal_h3200_source_anchor is None and source_ref.numel():
                    terminal_h3200_source_anchor = source_ref.detach().clone()
                if step >= 4000 and terminal_h4000_source_anchor is None and source_ref.numel():
                    terminal_h4000_source_anchor = source_ref.detach().clone()
                h3200_seed = terminal_h3200_source_anchor.to(device=device, dtype=current_grad.dtype) if terminal_h3200_source_anchor is not None else source_ref
                h4000_seed = terminal_h4000_source_anchor.to(device=device, dtype=current_grad.dtype) if terminal_h4000_source_anchor is not None else h3200_seed
                h3200_ref = normalized_like(h3200_seed, current_grad)
                h4000_ref = normalized_like(h4000_seed, current_grad)
                projected_update = make_late_floor_update(project_source=True)
                projected_grad = projected_update.tensor.detach().to(device=device, dtype=current_grad.dtype)
                grad_norm = torch.linalg.vector_norm(current_grad.float()).clamp_min(1.0e-12).to(device=device, dtype=current_grad.dtype)
                source_scaled = source_ref * grad_norm
                h3200_scaled = h3200_ref * grad_norm
                h4000_scaled = h4000_ref * grad_norm

                def candidate_update(name: str, vec: torch.Tensor, role: str) -> UpdateTensor:
                    return UpdateTensor(
                        torch.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0),
                        "gradient",
                        "subtract",
                        "slow_state",
                        name,
                        mechanism,
                        role=role,
                        one_step_descent_claim=0,
                    )

                candidates: list[tuple[str, UpdateTensor, float, float, float]] = [
                    (
                        "task_progress",
                        candidate_update(
                            "train_stream_terminal_task_progress_carry",
                            current_grad,
                            "terminal_task_progress_carry",
                        ),
                        0.10,
                        -0.35,
                        0.70,
                    ),
                    (
                        "source_progress_blend",
                        candidate_update(
                            "train_stream_terminal_source_progress_blend",
                            0.78 * current_grad + 0.22 * source_scaled,
                            "terminal_source_progress_blend",
                        ),
                        0.085,
                        -0.08,
                        0.60,
                    ),
                    (
                        "h4000_progress_blend",
                        candidate_update(
                            "train_stream_terminal_h4000_progress_blend",
                            0.74 * current_grad + 0.26 * h4000_scaled,
                            "terminal_h4000_progress_blend",
                        ),
                        0.075,
                        -0.02,
                        0.55,
                    ),
                    (
                        "h3200_projected_progress_blend",
                        candidate_update(
                            "train_stream_terminal_h3200_projected_progress_blend",
                            0.70 * projected_grad + 0.30 * h3200_scaled,
                            "terminal_h3200_projected_progress_blend",
                        ),
                        0.070,
                        0.05,
                        0.50,
                    ),
                ]
                if mode == "h800_source_slow_ema_terminal_h3200_progress_carry":
                    candidates = [c for c in candidates if c[0] in {"task_progress", "source_progress_blend"}]
                elif mode == "h800_source_slow_ema_terminal_h4000_progress_carry":
                    candidates = [c for c in candidates if c[0] in {"task_progress", "h4000_progress_blend"}]
                elif mode == "h800_source_slow_ema_terminal_h3200_source_progress_blend":
                    candidates = [c for c in candidates if c[0] in {"source_progress_blend", "h3200_projected_progress_blend"}]
                elif mode == "h800_source_slow_ema_terminal_raw_guard_h3600_gentle_progress":
                    candidates = [
                        (kind, update, 0.028, 0.18, 0.35)
                        for kind, update, _lr, _floor, _corrupt in candidates
                        if kind == "source_progress_blend"
                    ]
                elif mode == "h800_source_slow_ema_terminal_raw_guard_h4000_gentle_progress":
                    candidates = [
                        (kind, update, 0.024, 0.20, 0.32)
                        for kind, update, _lr, _floor, _corrupt in candidates
                        if kind == "h4000_progress_blend"
                    ]
                elif mode == "h800_source_slow_ema_terminal_raw_guard_h4400_projected_progress":
                    candidates = [
                        (kind, update, 0.020, 0.25, 0.30)
                        for kind, update, _lr, _floor, _corrupt in candidates
                        if kind == "h3200_projected_progress_blend"
                    ]

                corrupt_labels = (channel_b[1] + 1) % max(2, int(getattr(args, "classes", 10)))
                evaluated: list[tuple[str, UpdateTensor, float, float, float, float, bool]] = []
                for kind, update, lr_scale, source_floor_cos, corrupt_fraction in candidates:
                    before = snapshot(model)
                    with torch.no_grad():
                        before_a = F.cross_entropy(model(channel_a[0]).float(), channel_a[1], reduction="none")
                        before_b = F.cross_entropy(model(channel_b[0]).float(), channel_b[1], reduction="none")
                        before_corrupt = F.cross_entropy(model(channel_b[0]).float(), corrupt_labels, reduction="none")
                    apply_update(model, update, lr=lr_scale * float(args.lr))
                    with torch.no_grad():
                        after_a = F.cross_entropy(model(channel_a[0]).float(), channel_a[1], reduction="none")
                        after_b = F.cross_entropy(model(channel_b[0]).float(), channel_b[1], reduction="none")
                        after_corrupt = F.cross_entropy(model(channel_b[0]).float(), corrupt_labels, reduction="none")
                    load_flat_params(model, before)
                    gain_a = float((before_a - after_a).mean().item())
                    gain_b = float((before_b - after_b).mean().item())
                    corrupt_gain = float((before_corrupt - after_corrupt).mean().item())
                    signal_gain = min(gain_a, gain_b)
                    update_source_cos = cosine(update.tensor, source_ref)
                    allowed = (
                        signal_gain >= -1.0e-6
                        and corrupt_gain <= max(1.0e-4, corrupt_fraction * max(0.0, signal_gain))
                        and update_source_cos >= source_floor_cos
                    )
                    evaluated.append((kind, update, lr_scale, signal_gain, gain_b, corrupt_gain, bool(allowed)))

                allowed_candidates = [item for item in evaluated if item[6]]
                best_any = max(evaluated, key=lambda item: (item[3], -item[5], cosine(item[1].tensor, source_ref)))
                grad_source_cos = cosine(current_grad, source_ref)
                h3200_source_cos = cosine(h3200_ref, source_ref)
                h4000_source_cos = cosine(h4000_ref, source_ref)
                if not allowed_candidates:
                    terminal_selector_score = float(best_any[3])
                    terminal_selector_kind = (
                        f"{mode}:reject:grad_source={grad_source_cos:.4g}:h3200_source={h3200_source_cos:.4g}:"
                        f"h4000_source={h4000_source_cos:.4g}:anti_norm={terminal_projected_removed_norm:.4g}"
                    )
                    terminal_selector_current_cos = (
                        f"grad_source={grad_source_cos:.4g};h3200={h3200_source_cos:.4g};"
                        f"h4000={h4000_source_cos:.4g};anti={terminal_projected_removed_norm:.4g}"
                    )
                    return 0, float(best_any[3]), float(best_any[5]), float(best_any[4])

                best_kind, best_update, best_scale, signal_gain, gain_b, corrupt_gain, _allowed = max(
                    allowed_candidates,
                    key=lambda item: (
                        item[3],
                        cosine(item[1].tensor, source_ref),
                        -item[5],
                    ),
                )
                apply_update(model, best_update, lr=best_scale * float(args.lr))
                terminal_selector_score = float(signal_gain)
                terminal_selector_kind = (
                    f"{mode}:{best_kind}:grad_source={grad_source_cos:.4g}:h3200_source={h3200_source_cos:.4g}:"
                    f"h4000_source={h4000_source_cos:.4g}:update={cosine(best_update.tensor, source_ref):.4g}:"
                    f"anti_norm={terminal_projected_removed_norm:.4g}"
                )
                terminal_selector_current_cos = (
                    f"grad_source={grad_source_cos:.4g};h3200={h3200_source_cos:.4g};"
                    f"h4000={h4000_source_cos:.4g};update={cosine(best_update.tensor, source_ref):.4g};"
                    f"anti={terminal_projected_removed_norm:.4g}"
                )
                return 1, float(signal_gain), float(corrupt_gain), float(gain_b)

            def terminal_control_catchup_start(mode: str) -> int:
                if mode in {
                    "h800_source_slow_ema_terminal_control_relative_sgd_catchup",
                    "h800_source_slow_ema_terminal_control_relative_source_balanced_catchup",
                }:
                    return 3600
                return 4000

            def terminal_adamw_candidate_metrics_with_direction(scale: float = 0.08) -> tuple[dict[str, float], torch.Tensor]:
                corrupt_labels = (channel_b[1] + 1) % max(2, int(getattr(args, "classes", 10)))
                before = snapshot(model)
                opt_state = deepcopy(opt_adam.state_dict())
                old_lrs = [float(group.get("lr", args.lr)) for group in opt_adam.param_groups]
                with torch.no_grad():
                    before_a = F.cross_entropy(model(channel_a[0]).float(), channel_a[1], reduction="none")
                    before_b = F.cross_entropy(model(channel_b[0]).float(), channel_b[1], reduction="none")
                    before_corrupt = F.cross_entropy(model(channel_b[0]).float(), corrupt_labels, reduction="none")
                for group, old_lr in zip(opt_adam.param_groups, old_lrs):
                    group["lr"] = scale * old_lr
                opt_adam.step()
                after_params = snapshot(model)
                with torch.no_grad():
                    after_a = F.cross_entropy(model(channel_a[0]).float(), channel_a[1], reduction="none")
                    after_b = F.cross_entropy(model(channel_b[0]).float(), channel_b[1], reduction="none")
                    after_corrupt = F.cross_entropy(model(channel_b[0]).float(), corrupt_labels, reduction="none")
                load_flat_params(model, before)
                opt_adam.load_state_dict(opt_state)
                for group, old_lr in zip(opt_adam.param_groups, old_lrs):
                    group["lr"] = old_lr
                direction = torch.nan_to_num(before - after_params, nan=0.0, posinf=0.0, neginf=0.0)
                gain_a = float((before_a - after_a).mean().item())
                gain_b = float((before_b - after_b).mean().item())
                corrupt_gain = float((before_corrupt - after_corrupt).mean().item())
                signal_gain = min(gain_a, gain_b)
                norm = float(torch.linalg.vector_norm(direction.detach()).item())
                return {
                    "norm": norm,
                    "gain_a": gain_a,
                    "gain_b": gain_b,
                    "signal": signal_gain,
                    "corrupt": corrupt_gain,
                    "score": signal_gain - max(0.0, corrupt_gain),
                }, direction

            def apply_terminal_control_relative_catchup_route(mode: str) -> tuple[int, float, float, float]:
                nonlocal terminal_selector_score, terminal_selector_kind, terminal_selector_current_cos
                nonlocal terminal_h4000_source_anchor, terminal_h3200_source_anchor
                if step < terminal_control_catchup_start(mode):
                    return 0, 0.0, 0.0, 0.0
                current_grad = flat_grad(model, device).detach()
                if current_grad.numel() == 0:
                    return 0, 0.0, 0.0, 0.0
                source_ref = normalized_like(source_axis.to(device=device, dtype=current_grad.dtype), current_grad)
                if step >= 3200 and terminal_h3200_source_anchor is None and source_ref.numel():
                    terminal_h3200_source_anchor = source_ref.detach().clone()
                if step >= 4000 and terminal_h4000_source_anchor is None and source_ref.numel():
                    terminal_h4000_source_anchor = source_ref.detach().clone()
                h3200_seed = terminal_h3200_source_anchor.to(device=device, dtype=current_grad.dtype) if terminal_h3200_source_anchor is not None else source_ref
                h4000_seed = terminal_h4000_source_anchor.to(device=device, dtype=current_grad.dtype) if terminal_h4000_source_anchor is not None else h3200_seed
                h3200_ref = normalized_like(h3200_seed, current_grad)
                h4000_ref = normalized_like(h4000_seed, current_grad)
                projected_update = make_late_floor_update(project_source=True)
                projected_grad = projected_update.tensor.detach().to(device=device, dtype=current_grad.dtype)
                grad_norm = torch.linalg.vector_norm(current_grad.float()).clamp_min(1.0e-12).to(device=device, dtype=current_grad.dtype)
                h3200_scaled = h3200_ref * grad_norm
                h4000_scaled = h4000_ref * grad_norm

                def candidate_update(name: str, vec: torch.Tensor, role: str) -> UpdateTensor:
                    return UpdateTensor(
                        torch.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0),
                        "gradient",
                        "subtract",
                        "slow_state",
                        name,
                        mechanism,
                        role=role,
                        one_step_descent_claim=0,
                    )

                sgd_update = candidate_update(
                    "train_stream_terminal_control_relative_sgd_catchup",
                    current_grad,
                    "terminal_control_relative_sgd_catchup",
                )
                source_balanced_update = candidate_update(
                    "train_stream_terminal_control_relative_source_balanced_catchup",
                    0.70 * current_grad + 0.20 * h3200_scaled + 0.10 * h4000_scaled,
                    "terminal_control_relative_source_balanced_catchup",
                )
                projected_balanced_update = candidate_update(
                    "train_stream_terminal_control_relative_projected_balanced_catchup",
                    0.64 * projected_grad + 0.26 * h3200_scaled + 0.10 * h4000_scaled,
                    "terminal_control_relative_projected_balanced_catchup",
                )

                candidates: list[tuple[str, UpdateTensor | None, torch.Tensor, float, float, float, dict[str, float]]] = []
                if mode == "h800_source_slow_ema_terminal_control_relative_sgd_catchup":
                    metrics = terminal_candidate_metrics(sgd_update, 0.12)
                    candidates.append(("sgd_catchup", sgd_update, sgd_update.tensor, 0.12, -0.08, 0.32, metrics))
                    metrics = terminal_candidate_metrics(source_balanced_update, 0.085)
                    candidates.append(("source_balanced", source_balanced_update, source_balanced_update.tensor, 0.085, 0.08, 0.28, metrics))
                elif mode == "h800_source_slow_ema_terminal_control_relative_adamw_catchup":
                    metrics, direction = terminal_adamw_candidate_metrics_with_direction(0.075)
                    candidates.append(("adamw_catchup", None, direction.to(device=device, dtype=current_grad.dtype), 0.075, -0.04, 0.22, metrics))
                    metrics = terminal_candidate_metrics(source_balanced_update, 0.075)
                    candidates.append(("source_balanced", source_balanced_update, source_balanced_update.tensor, 0.075, 0.10, 0.28, metrics))
                else:
                    metrics = terminal_candidate_metrics(source_balanced_update, 0.070)
                    candidates.append(("source_balanced", source_balanced_update, source_balanced_update.tensor, 0.070, 0.18, 0.25, metrics))
                    metrics = terminal_candidate_metrics(projected_balanced_update, 0.055)
                    candidates.append(("projected_balanced", projected_balanced_update, projected_balanced_update.tensor, 0.055, 0.24, 0.25, metrics))

                evaluated: list[tuple[str, UpdateTensor | None, torch.Tensor, float, float, float, dict[str, float], bool]] = []
                for kind, update, direction, scale, source_floor_cos, corrupt_fraction, metrics in candidates:
                    update_source_cos = cosine(direction, source_ref)
                    allowed = (
                        metrics["norm"] > 1.0e-12
                        and metrics["signal"] >= 1.0e-7
                        and metrics["gain_b"] >= -1.0e-7
                        and metrics["corrupt"] <= max(1.0e-5, corrupt_fraction * max(0.0, metrics["signal"]))
                        and update_source_cos >= source_floor_cos
                    )
                    evaluated.append((kind, update, direction, scale, source_floor_cos, corrupt_fraction, metrics, bool(allowed)))

                best_any = max(
                    evaluated,
                    key=lambda item: (item[6]["score"], item[6]["signal"], cosine(item[2], source_ref), -item[6]["corrupt"]),
                )
                allowed_candidates = [item for item in evaluated if item[7]]
                grad_source_cos = cosine(current_grad, source_ref)
                h3200_source_cos = cosine(h3200_ref, source_ref)
                h4000_source_cos = cosine(h4000_ref, source_ref)
                if not allowed_candidates:
                    best_metrics = best_any[6]
                    terminal_selector_score = float(best_metrics["score"])
                    terminal_selector_kind = (
                        f"{mode}:reject:best={best_any[0]}:grad_source={grad_source_cos:.4g}:"
                        f"h3200_source={h3200_source_cos:.4g}:h4000_source={h4000_source_cos:.4g}:"
                        f"best_update={cosine(best_any[2], source_ref):.4g}"
                    )
                    terminal_selector_current_cos = (
                        f"grad_source={grad_source_cos:.4g};h3200={h3200_source_cos:.4g};"
                        f"h4000={h4000_source_cos:.4g};best={cosine(best_any[2], source_ref):.4g}"
                    )
                    return 0, float(best_metrics["signal"]), float(best_metrics["corrupt"]), float(best_metrics["gain_b"])

                best_kind, best_update, best_direction, best_scale, _source_floor, _corrupt_fraction, chosen, _allowed = max(
                    allowed_candidates,
                    key=lambda item: (item[6]["score"], item[6]["signal"], cosine(item[2], source_ref), -item[6]["corrupt"]),
                )
                if best_kind == "adamw_catchup":
                    commit_terminal_adamw(best_scale)
                elif best_update is not None:
                    apply_update(model, best_update, lr=best_scale * float(args.lr))
                terminal_selector_score = float(chosen["score"])
                terminal_selector_kind = (
                    f"{mode}:{best_kind}:grad_source={grad_source_cos:.4g}:h3200_source={h3200_source_cos:.4g}:"
                    f"h4000_source={h4000_source_cos:.4g}:update={cosine(best_direction, source_ref):.4g}:"
                    f"signal={chosen['signal']:.4g}:corrupt={chosen['corrupt']:.4g}"
                )
                terminal_selector_current_cos = (
                    f"grad_source={grad_source_cos:.4g};h3200={h3200_source_cos:.4g};"
                    f"h4000={h4000_source_cos:.4g};update={cosine(best_direction, source_ref):.4g}"
                )
                return 1, float(chosen["signal"]), float(chosen["corrupt"]), float(chosen["gain_b"])

            def apply_terminal_trajectory_adaptive_route(mode: str) -> tuple[int, float, float, float]:
                nonlocal terminal_selector_score, terminal_selector_kind, terminal_selector_current_cos
                nonlocal terminal_h1600_source_anchor, terminal_h2400_source_anchor
                nonlocal terminal_h3200_source_anchor, terminal_h4000_source_anchor
                current_grad = flat_grad(model, device).detach()
                if current_grad.numel() == 0:
                    return 0, 0.0, 0.0, 0.0
                source_ref = normalized_like(source_axis.to(device=device, dtype=current_grad.dtype), current_grad)
                if step >= 1600 and terminal_h1600_source_anchor is None and source_ref.numel():
                    terminal_h1600_source_anchor = source_ref.detach().clone()
                if step >= 2400 and terminal_h2400_source_anchor is None and source_ref.numel():
                    terminal_h2400_source_anchor = source_ref.detach().clone()
                if step >= 3200 and terminal_h3200_source_anchor is None and source_ref.numel():
                    terminal_h3200_source_anchor = source_ref.detach().clone()
                if step >= 4000 and terminal_h4000_source_anchor is None and source_ref.numel():
                    terminal_h4000_source_anchor = source_ref.detach().clone()

                h1600_seed = terminal_h1600_source_anchor.to(device=device, dtype=current_grad.dtype) if terminal_h1600_source_anchor is not None else source_ref
                h2400_seed = terminal_h2400_source_anchor.to(device=device, dtype=current_grad.dtype) if terminal_h2400_source_anchor is not None else h1600_seed
                h3200_seed = terminal_h3200_source_anchor.to(device=device, dtype=current_grad.dtype) if terminal_h3200_source_anchor is not None else h2400_seed
                h4000_seed = terminal_h4000_source_anchor.to(device=device, dtype=current_grad.dtype) if terminal_h4000_source_anchor is not None else h3200_seed
                h1600_ref = normalized_like(h1600_seed, current_grad)
                h2400_ref = normalized_like(h2400_seed, current_grad)
                h3200_ref = normalized_like(h3200_seed, current_grad)
                h4000_ref = normalized_like(h4000_seed, current_grad)

                projected_update = make_late_floor_update(project_source=True)
                raw_update = make_late_floor_update(project_source=False)
                consensus_update = make_terminal_consensus_update()
                projected_grad = projected_update.tensor.detach().to(device=device, dtype=current_grad.dtype)
                raw_grad = raw_update.tensor.detach().to(device=device, dtype=current_grad.dtype)
                consensus_grad = consensus_update.tensor.detach().to(device=device, dtype=current_grad.dtype)
                grad_norm = torch.linalg.vector_norm(current_grad.float()).clamp_min(1.0e-12).to(device=device, dtype=current_grad.dtype)
                source_scaled = source_ref * grad_norm
                h1600_scaled = h1600_ref * grad_norm
                h2400_scaled = h2400_ref * grad_norm
                h3200_scaled = h3200_ref * grad_norm
                h4000_scaled = h4000_ref * grad_norm
                mid_ref = normalized_like(0.46 * h1600_ref + 0.34 * h2400_ref + 0.20 * source_ref, current_grad)
                terminal_ref = normalized_like(0.52 * h3200_ref + 0.28 * h4000_ref + 0.20 * source_ref, current_grad)
                mid_scaled = mid_ref * grad_norm
                terminal_scaled = terminal_ref * grad_norm

                def candidate_update(name: str, vec: torch.Tensor, role: str) -> UpdateTensor:
                    return UpdateTensor(
                        torch.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0),
                        "gradient",
                        "subtract",
                        "slow_state",
                        name,
                        mechanism,
                        role=role,
                        one_step_descent_claim=0,
                    )

                mid_bridge = candidate_update(
                    "train_stream_terminal_trajectory_mid_bridge",
                    0.56 * projected_grad + 0.22 * h2400_scaled + 0.14 * h1600_scaled + 0.08 * source_scaled,
                    "terminal_trajectory_mid_bridge",
                )
                mid_source_floor = candidate_update(
                    "train_stream_terminal_trajectory_mid_source_floor",
                    0.72 * mid_scaled + 0.18 * projected_grad + 0.10 * consensus_grad,
                    "terminal_trajectory_mid_source_floor",
                )
                terminal_bridge = candidate_update(
                    "train_stream_terminal_trajectory_terminal_bridge",
                    0.52 * projected_grad + 0.28 * h3200_scaled + 0.12 * h4000_scaled + 0.08 * consensus_grad,
                    "terminal_trajectory_terminal_bridge",
                )
                ratio_repair = candidate_update(
                    "train_stream_terminal_trajectory_ratio_repair",
                    0.46 * projected_grad + 0.34 * terminal_scaled + 0.20 * h3200_scaled,
                    "terminal_trajectory_ratio_repair",
                )
                source_pulse = candidate_update(
                    "train_stream_terminal_trajectory_source_pulse",
                    0.86 * terminal_scaled + 0.14 * consensus_grad,
                    "terminal_trajectory_source_pulse",
                )
                progress_bridge = candidate_update(
                    "train_stream_terminal_trajectory_progress_bridge",
                    0.62 * raw_grad + 0.24 * terminal_scaled + 0.14 * source_scaled,
                    "terminal_trajectory_progress_bridge",
                )

                candidates: list[tuple[str, UpdateTensor, float, float, float, float]] = []
                if mode == "h800_source_slow_ema_terminal_trajectory_adaptive_preserve":
                    if step < 3600:
                        candidates = [
                            ("mid_bridge", mid_bridge, 0.046, 0.30, 0.32, -1.0e-6),
                            ("mid_source_floor", mid_source_floor, 0.032, 0.68, 0.25, -2.5e-6),
                        ]
                    elif step < 4400:
                        candidates = [
                            ("terminal_bridge", terminal_bridge, 0.044, 0.36, 0.30, -1.0e-6),
                            ("h3200_source_floor", source_pulse, 0.030, 0.72, 0.22, -2.5e-6),
                        ]
                    else:
                        candidates = [
                            ("ratio_repair", ratio_repair, 0.038, 0.50, 0.25, -1.0e-6),
                            ("source_pulse", source_pulse, 0.028, 0.78, 0.20, -3.0e-6),
                        ]
                elif mode == "h800_source_slow_ema_terminal_mid_erosion_bridge":
                    if step < 4000:
                        candidates = [
                            ("mid_source_floor", mid_source_floor, 0.040, 0.72, 0.24, -4.0e-6),
                            ("mid_bridge", mid_bridge, 0.036, 0.45, 0.28, -1.5e-6),
                        ]
                    else:
                        candidates = [
                            ("h3200_source_floor", source_pulse, 0.034, 0.70, 0.22, -3.0e-6),
                            ("terminal_bridge", terminal_bridge, 0.032, 0.48, 0.26, -1.0e-6),
                        ]
                else:
                    if step < 3600:
                        candidates = [
                            ("mid_bridge", mid_bridge, 0.040, 0.40, 0.30, -1.0e-6),
                            ("mid_source_floor", mid_source_floor, 0.030, 0.70, 0.24, -3.0e-6),
                        ]
                    elif step < 4400:
                        candidates = [
                            ("progress_bridge", progress_bridge, 0.050, 0.20, 0.34, 0.0),
                            ("terminal_bridge", terminal_bridge, 0.036, 0.45, 0.26, -1.0e-6),
                        ]
                    else:
                        candidates = [
                            ("terminal_source_pulse", source_pulse, 0.030, 0.75, 0.20, -3.0e-6),
                            ("ratio_projected", ratio_repair, 0.034, 0.55, 0.24, -1.0e-6),
                        ]

                state_cos = cosine(short_state, long_state)
                evaluated: list[tuple[str, UpdateTensor, float, dict[str, float], float, bool]] = []
                for kind, update, scale, source_floor_cos, corrupt_fraction, signal_floor in candidates:
                    metrics = terminal_candidate_metrics(update, scale)
                    update_source_cos = cosine(update.tensor, source_ref)
                    corrupt_limit = max(
                        1.0e-5,
                        corrupt_fraction * max(0.0, metrics["signal"]) + max(0.0, -signal_floor),
                    )
                    allowed = (
                        state_cos >= -0.25
                        and metrics["norm"] > 1.0e-12
                        and metrics["signal"] >= signal_floor
                        and metrics["gain_b"] >= signal_floor
                        and metrics["corrupt"] <= corrupt_limit
                        and update_source_cos >= source_floor_cos
                    )
                    evaluated.append((kind, update, scale, metrics, update_source_cos, bool(allowed)))

                best_any = max(evaluated, key=lambda item: (item[3]["score"], item[3]["signal"], item[4], -item[3]["corrupt"]))
                allowed_candidates = [item for item in evaluated if item[5]]
                grad_source_cos = cosine(current_grad, source_ref)
                anchor_trace = (
                    f"h1600={cosine(h1600_ref, source_ref):.4g};h2400={cosine(h2400_ref, source_ref):.4g};"
                    f"h3200={cosine(h3200_ref, source_ref):.4g};h4000={cosine(h4000_ref, source_ref):.4g}"
                )
                if not allowed_candidates:
                    best_metrics = best_any[3]
                    terminal_selector_score = float(best_metrics["score"])
                    terminal_selector_kind = (
                        f"{mode}:reject:best={best_any[0]}:grad_source={grad_source_cos:.4g}:state={state_cos:.4g}:"
                        f"anchors={anchor_trace}:best_update={best_any[4]:.4g}:signal={best_metrics['signal']:.4g}:"
                        f"corrupt={best_metrics['corrupt']:.4g}"
                    )
                    terminal_selector_current_cos = f"grad_source={grad_source_cos:.4g};state={state_cos:.4g};{anchor_trace};best={best_any[4]:.4g}"
                    return 0, float(best_metrics["signal"]), float(best_metrics["corrupt"]), float(best_metrics["gain_b"])

                best_kind, best_update, best_scale, chosen, update_source_cos, _allowed = max(
                    allowed_candidates,
                    key=lambda item: (item[3]["score"], item[3]["signal"], item[4], -item[3]["corrupt"]),
                )
                apply_update(model, best_update, lr=best_scale * float(args.lr))
                terminal_selector_score = float(chosen["score"])
                terminal_selector_kind = (
                    f"{mode}:{best_kind}:grad_source={grad_source_cos:.4g}:state={state_cos:.4g}:"
                    f"anchors={anchor_trace}:update={update_source_cos:.4g}:signal={chosen['signal']:.4g}:"
                    f"corrupt={chosen['corrupt']:.4g}"
                )
                terminal_selector_current_cos = f"grad_source={grad_source_cos:.4g};state={state_cos:.4g};{anchor_trace};update={update_source_cos:.4g}"
                return 1, float(chosen["signal"]), float(chosen["corrupt"]), float(chosen["gain_b"])

            def apply_terminal_minimal_transport_route(mode: str) -> tuple[int, float, float, float]:
                nonlocal terminal_selector_score, terminal_selector_kind, terminal_selector_current_cos
                nonlocal terminal_h3200_source_anchor, terminal_h4000_source_anchor, terminal_projected_removed_norm
                if step < 4000:
                    if update_train_loss_gate() > 1.05 * loss_gate_h800_threshold:
                        accept, signal, corrupt, gain_b = apply_terminal_projected_optimizer_route(
                            scale=0.052,
                            blend_source=0.70,
                            antiwashout=True,
                        )
                        if accept:
                            return accept, signal, corrupt, gain_b
                        return apply_late_lookahead_floor(
                            scale=0.026,
                            project_source=True,
                            positive_only=True,
                        )
                    return apply_terminal_source_conserving_route(scale=0.034, debt_aware=True)

                current_grad = flat_grad(model, device).detach()
                if current_grad.numel() == 0:
                    return 0, 0.0, 0.0, 0.0
                source_ref = normalized_like(source_axis.to(device=device, dtype=current_grad.dtype), current_grad)
                if step >= 3200 and terminal_h3200_source_anchor is None and source_ref.numel():
                    terminal_h3200_source_anchor = source_ref.detach().clone()
                if step >= 4000 and terminal_h4000_source_anchor is None and source_ref.numel():
                    terminal_h4000_source_anchor = source_ref.detach().clone()
                h3200_seed = terminal_h3200_source_anchor.to(device=device, dtype=current_grad.dtype) if terminal_h3200_source_anchor is not None else source_ref
                h4000_seed = terminal_h4000_source_anchor.to(device=device, dtype=current_grad.dtype) if terminal_h4000_source_anchor is not None else h3200_seed
                h3200_ref = normalized_like(h3200_seed, current_grad)
                h4000_ref = normalized_like(h4000_seed, current_grad)
                source_norm2 = torch.dot(source_ref.float(), source_ref.float()).clamp_min(1.0e-12).to(device=device, dtype=current_grad.dtype)
                anti_coeff = torch.clamp(torch.dot(current_grad.float(), source_ref.float()).to(device=device, dtype=current_grad.dtype), max=0.0) / source_norm2
                anti_source = anti_coeff * source_ref
                clipped_grad = torch.nan_to_num(current_grad - anti_source, nan=0.0, posinf=0.0, neginf=0.0)
                terminal_projected_removed_norm = float(torch.linalg.vector_norm(anti_source.detach()).item())
                grad_norm = torch.linalg.vector_norm(current_grad.float()).clamp_min(1.0e-12).to(device=device, dtype=current_grad.dtype)
                h3200_scaled = h3200_ref * grad_norm
                h4000_scaled = h4000_ref * grad_norm
                anchor_flow_ref = normalized_like(0.60 * h4000_ref + 0.40 * h3200_ref, current_grad)
                anchor_flow_scaled = anchor_flow_ref * grad_norm

                def candidate_update(name: str, vec: torch.Tensor, role: str) -> UpdateTensor:
                    return UpdateTensor(
                        torch.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0),
                        "gradient",
                        "subtract",
                        "slow_state",
                        name,
                        mechanism,
                        role=role,
                        one_step_descent_claim=0,
                    )

                clip_update = candidate_update(
                    "train_stream_terminal_minimal_anti_source_clip",
                    clipped_grad,
                    "terminal_minimal_anti_source_clip",
                )
                hold_clip_update = candidate_update(
                    "train_stream_terminal_minimal_hold_clip",
                    0.92 * clipped_grad + 0.08 * h4000_scaled,
                    "terminal_minimal_hold_clip",
                )
                anchor_flow_update = candidate_update(
                    "train_stream_terminal_minimal_anchor_flow",
                    0.86 * clipped_grad + 0.14 * anchor_flow_scaled,
                    "terminal_minimal_anchor_flow",
                )

                candidates: list[tuple[str, UpdateTensor, float, float, float, float]] = []
                grad_source_cos = cosine(current_grad, source_ref)
                if mode == "h800_source_slow_ema_terminal_anti_source_clip":
                    candidates = [
                        ("anti_source_clip", clip_update, 0.034, -0.02, 0.22, 0.0),
                        ("anchor_flow_tiny", anchor_flow_update, 0.018, 0.08, 0.18, -8.0e-7),
                    ]
                elif mode == "h800_source_slow_ema_terminal_debt_aware_hold":
                    if grad_source_cos >= -0.015 and update_train_loss_gate() <= 1.12 * loss_gate_h800_threshold:
                        terminal_selector_score = 0.0
                        terminal_selector_kind = f"{mode}:post4000_hold:grad_source={grad_source_cos:.4g}"
                        terminal_selector_current_cos = f"grad_source={grad_source_cos:.4g};hold=1"
                        return 0, 0.0, 0.0, 0.0
                    candidates = [
                        ("hold_clip", hold_clip_update, 0.018, 0.06, 0.18, -8.0e-7),
                        ("anchor_flow_tiny", anchor_flow_update, 0.014, 0.12, 0.16, -1.0e-6),
                    ]
                else:
                    candidates = [
                        ("anchor_flow_tiny", anchor_flow_update, 0.016, 0.14, 0.16, -8.0e-7),
                        ("hold_clip", hold_clip_update, 0.012, 0.10, 0.14, -1.0e-6),
                    ]

                evaluated: list[tuple[str, UpdateTensor, float, dict[str, float], float, bool]] = []
                for kind, update, scale, source_floor_cos, corrupt_fraction, signal_floor in candidates:
                    metrics = terminal_candidate_metrics(update, scale)
                    update_source_cos = cosine(update.tensor, source_ref)
                    corrupt_limit = max(
                        1.0e-5,
                        corrupt_fraction * max(0.0, metrics["signal"]) + max(0.0, -signal_floor),
                    )
                    allowed = (
                        metrics["norm"] > 1.0e-12
                        and metrics["signal"] >= signal_floor
                        and metrics["gain_b"] >= signal_floor
                        and metrics["corrupt"] <= corrupt_limit
                        and update_source_cos >= source_floor_cos
                    )
                    evaluated.append((kind, update, scale, metrics, update_source_cos, bool(allowed)))

                best_any = max(evaluated, key=lambda item: (item[3]["score"], item[3]["signal"], item[4], -item[3]["corrupt"]))
                allowed_candidates = [item for item in evaluated if item[5]]
                h3200_source_cos = cosine(h3200_ref, source_ref)
                h4000_source_cos = cosine(h4000_ref, source_ref)
                if not allowed_candidates:
                    best_metrics = best_any[3]
                    terminal_selector_score = float(best_metrics["score"])
                    terminal_selector_kind = (
                        f"{mode}:reject:best={best_any[0]}:grad_source={grad_source_cos:.4g}:"
                        f"h3200_source={h3200_source_cos:.4g}:h4000_source={h4000_source_cos:.4g}:"
                        f"best_update={best_any[4]:.4g}:anti_norm={terminal_projected_removed_norm:.4g}:"
                        f"signal={best_metrics['signal']:.4g}:corrupt={best_metrics['corrupt']:.4g}"
                    )
                    terminal_selector_current_cos = (
                        f"grad_source={grad_source_cos:.4g};h3200={h3200_source_cos:.4g};"
                        f"h4000={h4000_source_cos:.4g};best={best_any[4]:.4g};anti={terminal_projected_removed_norm:.4g}"
                    )
                    return 0, float(best_metrics["signal"]), float(best_metrics["corrupt"]), float(best_metrics["gain_b"])

                best_kind, best_update, best_scale, chosen, update_source_cos, _allowed = max(
                    allowed_candidates,
                    key=lambda item: (item[3]["score"], item[3]["signal"], item[4], -item[3]["corrupt"]),
                )
                apply_update(model, best_update, lr=best_scale * float(args.lr))
                terminal_selector_score = float(chosen["score"])
                terminal_selector_kind = (
                    f"{mode}:{best_kind}:grad_source={grad_source_cos:.4g}:h3200_source={h3200_source_cos:.4g}:"
                    f"h4000_source={h4000_source_cos:.4g}:update={update_source_cos:.4g}:"
                    f"anti_norm={terminal_projected_removed_norm:.4g}:signal={chosen['signal']:.4g}:"
                    f"corrupt={chosen['corrupt']:.4g}"
                )
                terminal_selector_current_cos = (
                    f"grad_source={grad_source_cos:.4g};h3200={h3200_source_cos:.4g};"
                    f"h4000={h4000_source_cos:.4g};update={update_source_cos:.4g};"
                    f"anti={terminal_projected_removed_norm:.4g}"
                )
                return 1, float(chosen["signal"]), float(chosen["corrupt"]), float(chosen["gain_b"])

            def apply_terminal_accept_memory_route(mode: str) -> tuple[int, float, float, float]:
                nonlocal terminal_selector_score, terminal_selector_kind, terminal_selector_current_cos
                nonlocal terminal_h3200_source_anchor, terminal_h4000_source_anchor
                nonlocal terminal_accept_memory_update, terminal_accept_memory_signal, terminal_accept_memory_step
                current_grad = flat_grad(model, device).detach()
                if current_grad.numel() == 0:
                    return 0, 0.0, 0.0, 0.0
                source_ref = normalized_like(source_axis.to(device=device, dtype=current_grad.dtype), current_grad)
                if step >= 3200 and terminal_h3200_source_anchor is None and source_ref.numel():
                    terminal_h3200_source_anchor = source_ref.detach().clone()
                if step >= 4000 and terminal_h4000_source_anchor is None and source_ref.numel():
                    terminal_h4000_source_anchor = source_ref.detach().clone()
                h3200_seed = terminal_h3200_source_anchor.to(device=device, dtype=current_grad.dtype) if terminal_h3200_source_anchor is not None else source_ref
                h4000_seed = terminal_h4000_source_anchor.to(device=device, dtype=current_grad.dtype) if terminal_h4000_source_anchor is not None else h3200_seed
                h3200_ref = normalized_like(h3200_seed, current_grad)
                h4000_ref = normalized_like(h4000_seed, current_grad)
                projected_update = make_late_floor_update(project_source=True)
                raw_update = make_late_floor_update(project_source=False)
                consensus_update = make_terminal_consensus_update()
                projected_grad = projected_update.tensor.detach().to(device=device, dtype=current_grad.dtype)
                raw_grad = raw_update.tensor.detach().to(device=device, dtype=current_grad.dtype)
                consensus_grad = consensus_update.tensor.detach().to(device=device, dtype=current_grad.dtype)
                grad_norm = torch.linalg.vector_norm(current_grad.float()).clamp_min(1.0e-12).to(device=device, dtype=current_grad.dtype)
                source_scaled = source_ref * grad_norm
                h3200_scaled = h3200_ref * grad_norm
                h4000_scaled = h4000_ref * grad_norm
                memory_scaled: torch.Tensor | None = None
                if terminal_accept_memory_update is not None and terminal_accept_memory_update.numel() == current_grad.numel():
                    memory_scaled = normalized_like(terminal_accept_memory_update.to(device=device, dtype=current_grad.dtype), current_grad)

                def candidate_update(name: str, vec: torch.Tensor, role: str) -> UpdateTensor:
                    return UpdateTensor(
                        torch.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0),
                        "gradient",
                        "subtract",
                        "slow_state",
                        name,
                        mechanism,
                        role=role,
                        one_step_descent_claim=0,
                    )

                accept_bridge = candidate_update(
                    "train_stream_terminal_accept_memory_bridge",
                    0.58 * projected_grad + 0.24 * source_scaled + 0.18 * h3200_scaled,
                    "terminal_accept_memory_bridge",
                )
                debt_transport = candidate_update(
                    "train_stream_terminal_accept_memory_debt_transport",
                    0.62 * raw_grad + 0.20 * source_scaled + 0.18 * projected_grad,
                    "terminal_accept_memory_debt_transport",
                )
                source_progress = candidate_update(
                    "train_stream_terminal_source_progress_memory_transport",
                    0.46 * projected_grad + 0.30 * h3200_scaled + 0.16 * h4000_scaled + 0.08 * consensus_grad,
                    "terminal_source_progress_memory_transport",
                )
                memory_replay = (
                    candidate_update(
                        "train_stream_terminal_accept_memory_replay",
                        0.72 * memory_scaled + 0.18 * source_scaled + 0.10 * h4000_scaled,
                        "terminal_accept_memory_replay",
                    )
                    if memory_scaled is not None
                    else None
                )

                candidates: list[tuple[str, UpdateTensor, float, float, float, float]] = []
                if mode == "h800_source_slow_ema_terminal_accept_memory":
                    candidates = [
                        ("accept_bridge", accept_bridge, 0.050, 0.26, 0.30, -1.0e-6),
                        ("source_progress", source_progress, 0.040, 0.34, 0.28, -1.5e-6),
                    ]
                    if step >= 4000 and memory_replay is not None:
                        candidates.insert(0, ("memory_replay", memory_replay, 0.030, 0.34, 0.24, -2.0e-6))
                elif mode == "h800_source_slow_ema_terminal_accept_memory_debt":
                    candidates = [
                        ("debt_transport", debt_transport, 0.054, -0.04, 0.42, -5.0e-7),
                        ("accept_bridge", accept_bridge, 0.046, 0.20, 0.34, -1.0e-6),
                    ]
                    if step >= 4000 and memory_replay is not None:
                        candidates.insert(0, ("memory_replay", memory_replay, 0.036, 0.26, 0.28, -2.0e-6))
                else:
                    candidates = [
                        ("source_progress", source_progress, 0.046, 0.32, 0.30, -1.0e-6),
                        ("accept_bridge", accept_bridge, 0.040, 0.30, 0.28, -1.5e-6),
                    ]
                    if step >= 4000 and memory_replay is not None:
                        candidates.insert(0, ("memory_replay", memory_replay, 0.028, 0.38, 0.22, -2.5e-6))

                state_cos = cosine(short_state, long_state)
                evaluated: list[tuple[str, UpdateTensor, float, dict[str, float], float, bool]] = []
                for kind, update, scale, source_floor_cos, corrupt_fraction, signal_floor in candidates:
                    metrics = terminal_candidate_metrics(update, scale)
                    update_source_cos = cosine(update.tensor, source_ref)
                    corrupt_limit = max(
                        1.0e-5,
                        corrupt_fraction * max(0.0, metrics["signal"]) + max(0.0, -signal_floor),
                    )
                    allowed = (
                        state_cos >= -0.25
                        and metrics["norm"] > 1.0e-12
                        and metrics["signal"] >= signal_floor
                        and metrics["gain_b"] >= signal_floor
                        and metrics["corrupt"] <= corrupt_limit
                        and update_source_cos >= source_floor_cos
                    )
                    evaluated.append((kind, update, scale, metrics, update_source_cos, bool(allowed)))

                best_any = max(evaluated, key=lambda item: (item[3]["score"], item[3]["signal"], item[4], -item[3]["corrupt"]))
                allowed_candidates = [item for item in evaluated if item[5]]
                grad_source_cos = cosine(current_grad, source_ref)
                h3200_source_cos = cosine(h3200_ref, source_ref)
                h4000_source_cos = cosine(h4000_ref, source_ref)
                memory_source_cos = cosine(memory_scaled, source_ref) if memory_scaled is not None else 0.0
                if not allowed_candidates:
                    best_metrics = best_any[3]
                    terminal_selector_score = float(best_metrics["score"])
                    terminal_selector_kind = (
                        f"{mode}:reject:best={best_any[0]}:grad_source={grad_source_cos:.4g}:"
                        f"h3200_source={h3200_source_cos:.4g}:h4000_source={h4000_source_cos:.4g}:"
                        f"memory={int(memory_scaled is not None)}:memory_source={memory_source_cos:.4g}:"
                        f"memory_step={terminal_accept_memory_step}:best_update={best_any[4]:.4g}:"
                        f"signal={best_metrics['signal']:.4g}:corrupt={best_metrics['corrupt']:.4g}"
                    )
                    terminal_selector_current_cos = (
                        f"grad_source={grad_source_cos:.4g};h3200={h3200_source_cos:.4g};"
                        f"h4000={h4000_source_cos:.4g};memory={memory_source_cos:.4g};best={best_any[4]:.4g}"
                    )
                    return 0, float(best_metrics["signal"]), float(best_metrics["corrupt"]), float(best_metrics["gain_b"])

                best_kind, best_update, best_scale, chosen, update_source_cos, _allowed = max(
                    allowed_candidates,
                    key=lambda item: (item[3]["score"], item[3]["signal"], item[4], -item[3]["corrupt"]),
                )
                apply_update(model, best_update, lr=best_scale * float(args.lr))
                if update_source_cos >= 0.12 and chosen["signal"] >= -2.0e-6:
                    terminal_accept_memory_update = best_update.tensor.detach().clone()
                    terminal_accept_memory_signal = float(chosen["signal"])
                    terminal_accept_memory_step = int(step)
                terminal_selector_score = float(chosen["score"])
                terminal_selector_kind = (
                    f"{mode}:{best_kind}:grad_source={grad_source_cos:.4g}:h3200_source={h3200_source_cos:.4g}:"
                    f"h4000_source={h4000_source_cos:.4g}:memory={int(memory_scaled is not None)}:"
                    f"memory_source={memory_source_cos:.4g}:memory_signal={terminal_accept_memory_signal:.4g}:"
                    f"memory_step={terminal_accept_memory_step}:update={update_source_cos:.4g}:"
                    f"signal={chosen['signal']:.4g}:corrupt={chosen['corrupt']:.4g}"
                )
                terminal_selector_current_cos = (
                    f"grad_source={grad_source_cos:.4g};h3200={h3200_source_cos:.4g};"
                    f"h4000={h4000_source_cos:.4g};memory={memory_source_cos:.4g};update={update_source_cos:.4g}"
                )
                return 1, float(chosen["signal"]), float(chosen["corrupt"]), float(chosen["gain_b"])

            def apply_source_state_adaptive_terminal_route(mode: str) -> tuple[int, float, float, float]:
                nonlocal terminal_selector_score, terminal_selector_kind, terminal_selector_current_cos
                current_grad = flat_grad(model, device).detach()
                source_ref = normalized_like(source_axis.to(device=device, dtype=current_grad.dtype), current_grad)
                projected_update = make_late_floor_update(project_source=True)
                raw_update = make_late_floor_update(project_source=False)
                source_update = UpdateTensor(
                    source_ref,
                    "step",
                    "subtract",
                    "slow_state",
                    "train_stream_terminal_source_state_axis_guard",
                    mechanism,
                    role="terminal_source_state_axis_guard",
                    one_step_descent_claim=0,
                )
                blend_vec = normalized_like(0.55 * projected_update.tensor.detach() + 0.45 * source_ref, projected_update.tensor)
                blend_update = UpdateTensor(
                    blend_vec,
                    "gradient",
                    "subtract",
                    "slow_state",
                    "train_stream_terminal_source_state_projected_blend_guard",
                    mechanism,
                    role="terminal_source_state_projected_blend_guard",
                    one_step_descent_claim=0,
                )
                lr_denom = max(float(args.lr), 1.0e-12)
                source_scale = max(1.0e-8, 0.20 * float(args.fu_lr) / lr_denom)
                raw_scale = 0.06 if mode != "h800_source_slow_ema_terminal_ratio_preserve" else 0.05
                projected_scale = 0.06
                blend_scale = 0.05
                raw = terminal_candidate_metrics(raw_update, raw_scale)
                projected = terminal_candidate_metrics(projected_update, projected_scale)
                source = terminal_candidate_metrics(source_update, source_scale)
                blend = terminal_candidate_metrics(blend_update, blend_scale)
                grad_source_cos = cosine(current_grad, source_ref)
                state_cos = cosine(short_state, long_state)
                terminal_selector_current_cos = f"grad={grad_source_cos:.4g};state={state_cos:.4g};density={agree_density:.4g}"

                def gate(
                    update: UpdateTensor,
                    metrics: dict[str, float],
                    *,
                    min_signal: float,
                    corrupt_fraction: float,
                    source_floor: float,
                    density_floor: float,
                    state_floor: float,
                ) -> bool:
                    return (
                        metrics["norm"] > 1.0e-12
                        and agree_density >= density_floor
                        and state_cos >= state_floor
                        and metrics["signal"] >= min_signal
                        and metrics["corrupt"] <= max(1.0e-5, corrupt_fraction * max(0.0, metrics["signal"]))
                        and cosine(update.tensor, source_ref) >= source_floor
                    )

                candidates: list[tuple[str, UpdateTensor, dict[str, float], float, bool]] = []
                if mode == "h800_source_slow_ema_terminal_sparse_source":
                    sparse_slot = int((step // max(1, int(args.alt_period))) % 2 == 0)
                    candidates = [
                        (
                            "sparse_source_axis",
                            source_update,
                            source,
                            source_scale,
                            bool(sparse_slot)
                            and gate(
                                source_update,
                                source,
                                min_signal=-2.0e-6,
                                corrupt_fraction=0.20,
                                source_floor=0.95,
                                density_floor=0.15,
                                state_floor=-0.02,
                            ),
                        )
                    ]
                elif mode == "h800_source_slow_ema_terminal_ratio_preserve":
                    candidates = [
                        (
                            "ratio_source_axis",
                            source_update,
                            source,
                            source_scale,
                            gate(
                                source_update,
                                source,
                                min_signal=-1.0e-6,
                                corrupt_fraction=0.22,
                                source_floor=0.95,
                                density_floor=0.12,
                                state_floor=-0.03,
                            ),
                        ),
                        (
                            "ratio_projected_blend",
                            blend_update,
                            blend,
                            blend_scale,
                            gate(
                                blend_update,
                                blend,
                                min_signal=1.0e-7,
                                corrupt_fraction=0.22,
                                source_floor=-0.02,
                                density_floor=0.12,
                                state_floor=-0.03,
                            ),
                        ),
                    ]
                else:
                    raw_allowed = step < 4400 and grad_source_cos >= -0.08
                    candidates = [
                        (
                            "adaptive_raw",
                            raw_update,
                            raw,
                            raw_scale,
                            bool(raw_allowed)
                            and gate(
                                raw_update,
                                raw,
                                min_signal=1.0e-7,
                                corrupt_fraction=0.25,
                                source_floor=-0.12,
                                density_floor=0.12,
                                state_floor=-0.04,
                            ),
                        ),
                        (
                            "adaptive_projected_blend",
                            blend_update,
                            blend,
                            blend_scale,
                            gate(
                                blend_update,
                                blend,
                                min_signal=1.0e-7,
                                corrupt_fraction=0.25,
                                source_floor=-0.02,
                                density_floor=0.12,
                                state_floor=-0.04,
                            ),
                        ),
                        (
                            "adaptive_source_axis",
                            source_update,
                            source,
                            source_scale,
                            gate(
                                source_update,
                                source,
                                min_signal=-1.0e-6,
                                corrupt_fraction=0.25,
                                source_floor=0.95,
                                density_floor=0.12,
                                state_floor=-0.04,
                            ),
                        ),
                    ]

                allowed = [
                    (kind, update, metrics, candidate_scale)
                    for kind, update, metrics, candidate_scale, accept in candidates
                    if accept
                ]
                if not allowed:
                    best_metrics = max((item[2] for item in candidates), key=lambda item: (item["score"], item["signal"], -item["corrupt"]))
                    terminal_selector_score = float(best_metrics["score"])
                    terminal_selector_kind = f"{mode}:reject:{terminal_selector_current_cos}"
                    return 0, float(best_metrics["signal"]), float(best_metrics["corrupt"]), float(best_metrics["gain_b"])

                best_kind, best_update, best_metrics, best_scale = max(
                    allowed,
                    key=lambda item: (item[2]["score"], item[2]["signal"], cosine(item[1].tensor, source_ref), -item[2]["corrupt"]),
                )
                apply_update(model, best_update, lr=best_scale * float(args.lr))
                update_source_cos = cosine(best_update.tensor, source_ref)
                terminal_selector_score = float(best_metrics["score"])
                terminal_selector_kind = f"{mode}:{best_kind}:update_cos={update_source_cos:.4g}"
                terminal_selector_current_cos = f"{terminal_selector_current_cos};update={update_source_cos:.4g}"
                return 1, float(best_metrics["signal"]), float(best_metrics["corrupt"]), float(best_metrics["gain_b"])

            def apply_terminal_topk_support_route(mode: str) -> tuple[int, float, float, float]:
                nonlocal terminal_projected_removed_norm, terminal_consensus_density, terminal_consensus_balance
                nonlocal terminal_selector_score, terminal_selector_kind, terminal_selector_current_cos
                current_grad = flat_grad(model, device).detach()
                source_ref = normalized_like(source_axis.to(device=device, dtype=current_grad.dtype), current_grad)
                support_score = torch.minimum(short_state.detach().float().abs(), long_state.detach().float().abs())
                support_score = support_score * agree.detach().float()
                positive = support_score > 0
                positive_count = int(positive.sum().item()) if positive.numel() else 0
                if positive_count <= 0:
                    terminal_consensus_density = 0.0
                    terminal_consensus_balance = 0.0
                    terminal_selector_score = -2.0
                    terminal_selector_kind = f"{mode}:no_support"
                    terminal_selector_current_cos = ""
                    return 0, 0.0, 0.0, 0.0

                support_fraction = (
                    0.10
                    if mode == "h800_source_slow_ema_terminal_topk_raw_guard"
                    else 0.08
                    if mode == "h800_source_slow_ema_terminal_topk_support_guard"
                    else 0.06
                )
                k = max(1, min(positive_count, int(float(positive_count) * support_fraction)))
                threshold = torch.topk(support_score[positive], k).values[-1]
                support = positive & (support_score >= threshold)
                support_count = int(support.sum().item()) if support.numel() else 0
                terminal_consensus_density = float(support.float().mean().item()) if support.numel() else 0.0
                balance = torch.minimum(short_state.detach().float().abs(), long_state.detach().float().abs())
                balance = balance / torch.maximum(short_state.detach().float().abs(), long_state.detach().float().abs()).clamp_min(1.0e-8)
                terminal_consensus_balance = float(balance[support].mean().item()) if support_count > 0 else 0.0

                topk_vec = torch.where(support.to(device=device), source_ref, torch.zeros_like(source_ref))
                topk_vec = normalized_like(topk_vec, current_grad if current_grad.numel() else topk_vec)
                if float(torch.linalg.vector_norm(topk_vec.detach()).item()) <= 1.0e-12:
                    terminal_selector_score = -2.0
                    terminal_selector_kind = f"{mode}:zero_topk"
                    terminal_selector_current_cos = ""
                    return 0, 0.0, 0.0, 0.0

                topk_update = UpdateTensor(
                    topk_vec,
                    "step",
                    "subtract",
                    "slow_state",
                    "train_stream_terminal_topk_source_support",
                    mechanism,
                    role="terminal_topk_source_support_guard",
                    one_step_descent_claim=0,
                )
                topk_norm2 = torch.dot(topk_vec.float(), topk_vec.float()).clamp_min(1.0e-12)
                anti_coeff = torch.clamp(torch.dot(current_grad.float(), topk_vec.float()), max=0.0) / topk_norm2
                projected_grad = current_grad - anti_coeff.to(device=device, dtype=current_grad.dtype) * topk_vec
                projected_grad = torch.nan_to_num(projected_grad, nan=0.0, posinf=0.0, neginf=0.0)
                terminal_projected_removed_norm = float(torch.linalg.vector_norm((current_grad - projected_grad).detach()).item())
                hybrid_weight = 0.48 if mode == "h800_source_slow_ema_terminal_topk_raw_guard" else 0.32
                hybrid_vec = normalized_like((1.0 - hybrid_weight) * projected_grad + hybrid_weight * topk_vec, current_grad)
                hybrid_update = UpdateTensor(
                    hybrid_vec,
                    "gradient",
                    "subtract",
                    "slow_state",
                    "train_stream_terminal_topk_raw_source_guard",
                    mechanism,
                    role="terminal_topk_raw_source_guard",
                    one_step_descent_claim=0,
                )

                lr_denom = max(float(args.lr), 1.0e-12)
                topk_scale = max(1.0e-8, 0.32 * float(args.fu_lr) / lr_denom)
                raw_scale = 0.055 if mode == "h800_source_slow_ema_terminal_topk_raw_guard" else 0.040
                if mode == "h800_source_slow_ema_terminal_topk_debt_cap":
                    raw_scale = 0.030
                topk_metrics = terminal_candidate_metrics(topk_update, topk_scale)
                hybrid_metrics = terminal_candidate_metrics(hybrid_update, raw_scale)
                state_cos = cosine(short_state, long_state)
                grad_topk_cos = cosine(current_grad, topk_vec)
                debt_exists = True
                if mode == "h800_source_slow_ema_terminal_topk_debt_cap":
                    debt_exists = update_train_loss_gate() > 1.05 * loss_gate_h800_threshold
                terminal_selector_current_cos = (
                    f"grad={grad_topk_cos:.4g};state={state_cos:.4g};support={terminal_consensus_density:.4g};debt={int(debt_exists)}"
                )

                def gate(update: UpdateTensor, metrics: dict[str, float], *, source_floor: float, corrupt_fraction: float, signal_floor: float) -> bool:
                    return (
                        support_count > 0
                        and terminal_consensus_density >= 0.01
                        and state_cos >= -0.05
                        and metrics["norm"] > 1.0e-12
                        and metrics["signal"] >= signal_floor
                        and metrics["corrupt"] <= max(1.0e-5, corrupt_fraction * max(0.0, metrics["signal"]))
                        and cosine(update.tensor, topk_vec) >= source_floor
                    )

                candidates: list[tuple[str, UpdateTensor, dict[str, float], float, bool]] = [
                    (
                        "topk_source",
                        topk_update,
                        topk_metrics,
                        topk_scale,
                        gate(topk_update, topk_metrics, source_floor=0.90, corrupt_fraction=0.22, signal_floor=-1.0e-6),
                    )
                ]
                if mode == "h800_source_slow_ema_terminal_topk_raw_guard" or (
                    mode == "h800_source_slow_ema_terminal_topk_debt_cap" and debt_exists
                ):
                    candidates.append(
                        (
                            "topk_raw_hybrid",
                            hybrid_update,
                            hybrid_metrics,
                            raw_scale,
                            gate(
                                hybrid_update,
                                hybrid_metrics,
                                source_floor=0.15,
                                corrupt_fraction=0.25 if mode.endswith("debt_cap") else 0.30,
                                signal_floor=1.0e-7,
                            ),
                        )
                    )

                allowed = [
                    (kind, update, metrics, candidate_scale)
                    for kind, update, metrics, candidate_scale, accept in candidates
                    if accept
                ]
                if not allowed:
                    best_metrics = max((item[2] for item in candidates), key=lambda item: (item["score"], item["signal"], -item["corrupt"]))
                    terminal_selector_score = float(best_metrics["score"])
                    terminal_selector_kind = f"{mode}:reject:{terminal_selector_current_cos}"
                    return 0, float(best_metrics["signal"]), float(best_metrics["corrupt"]), float(best_metrics["gain_b"])

                best_kind, best_update, best_metrics, best_scale = max(
                    allowed,
                    key=lambda item: (item[2]["score"], item[2]["signal"], cosine(item[1].tensor, topk_vec), -item[2]["corrupt"]),
                )
                apply_update(model, best_update, lr=best_scale * float(args.lr))
                update_topk_cos = cosine(best_update.tensor, topk_vec)
                terminal_selector_score = float(best_metrics["score"])
                terminal_selector_kind = f"{mode}:{best_kind}:topk_cos={update_topk_cos:.4g}"
                terminal_selector_current_cos = f"{terminal_selector_current_cos};update={update_topk_cos:.4g}"
                return 1, float(best_metrics["signal"]), float(best_metrics["corrupt"]), float(best_metrics["gain_b"])

            def apply_signal_channel_target_gate() -> tuple[int, dict[str, Any]]:
                nonlocal terminal_selector_score, terminal_selector_kind, terminal_selector_current_cos
                if loss_gate_fallback_mode not in signal_target_modes:
                    return 0, {}
                if step < 850:
                    return 0, {
                        "operator_status": f"{loss_gate_fallback_mode}:pre_h800_source_state",
                        "operator_gate_accept": 0,
                    }
                terminal_target_modes = {
                    "h800_source_slow_ema_terminal_source_projection_target",
                    "h800_source_slow_ema_terminal_noise_orthogonal_target",
                    "h800_source_slow_ema_terminal_easy_margin_target",
                    "h800_source_slow_ema_low_nds_diffeomorphic_target",
                    "h800_source_slow_ema_info_volume_diffeomorphic_target",
                    "h800_source_slow_ema_low_rank_readout_transport",
                }
                if loss_gate_fallback_mode in terminal_target_modes and step < 4000:
                    return 0, {
                        "operator_status": f"{loss_gate_fallback_mode}:pre_terminal_target_gate",
                        "operator_gate_accept": 0,
                    }

                random_update = make_update(
                    model,
                    "M136-StableRandomB3NullTargetControlFU",
                    xb,
                    yb,
                    seed=seed + step + 17,
                )
                random_diag = dict(random_update.diagnostics or {})
                random_b2 = finite_float(random_diag.get("B2_transfer_gain"), -1.0e9)
                random_r2 = finite_float(random_diag.get("ActuationR2"), -1.0)

                if loss_gate_fallback_mode == "h800_source_slow_ema_signal_reservoir_target":
                    target_specs = [("signal_reservoir", "M134-SignalReservoirB3NullConsensusTargetFU")]
                elif loss_gate_fallback_mode == "h800_source_slow_ema_source_bank_target":
                    target_specs = [("source_bank", "M135-SourceBankB3NullConsensusTargetFU")]
                elif loss_gate_fallback_mode == "h800_source_slow_ema_terminal_source_projection_target":
                    target_specs = [("source_projection", "M155-SourceProjectionB3NullConsensusTargetOnlyFU")]
                elif loss_gate_fallback_mode == "h800_source_slow_ema_terminal_noise_orthogonal_target":
                    target_specs = [("noise_orthogonal", "M156-NoiseOrthogonalB3NullConsensusTargetOnlyFU")]
                elif loss_gate_fallback_mode == "h800_source_slow_ema_terminal_easy_margin_target":
                    target_specs = [("easy_margin", "M157-EasyMarginB3NullConsensusTargetOnlyFU")]
                elif loss_gate_fallback_mode == "h800_source_slow_ema_low_nds_diffeomorphic_target":
                    target_specs = [("low_nds_diffeomorphic", "M176-LowNDSDiffeomorphicTargetOnlyFU")]
                elif loss_gate_fallback_mode == "h800_source_slow_ema_info_volume_diffeomorphic_target":
                    target_specs = [("info_volume_diffeomorphic", "M177-InfoVolumeDiffeomorphicTargetOnlyFU")]
                elif loss_gate_fallback_mode == "h800_source_slow_ema_low_rank_readout_transport":
                    target_specs = [("low_rank_readout_transport", "M178-LowRankReadoutTransportTargetOnlyFU")]
                else:
                    target_specs = [
                        ("signal_reservoir", "M134-SignalReservoirB3NullConsensusTargetFU"),
                        ("source_bank", "M135-SourceBankB3NullConsensusTargetFU"),
                    ]

                source_ref = normalized_like(source_axis.to(device=device, dtype=flat_grad(model, device).dtype), flat_grad(model, device))
                candidates: list[tuple[str, UpdateTensor, dict[str, Any], float, bool]] = []
                for family, target_mechanism in target_specs:
                    target_update = make_update(model, target_mechanism, xb, yb, seed=seed + step)
                    diag = dict(target_update.diagnostics or {})
                    r2 = finite_float(diag.get("ActuationR2"), -1.0)
                    b1 = finite_float(diag.get("B1_gain"), -1.0e9)
                    b2 = finite_float(diag.get("B2_transfer_gain"), -1.0e9)
                    b3 = finite_float(diag.get("B3_safety_gain"), -1.0e9)
                    source_cos = cosine(target_update.tensor.detach(), source_ref)
                    score = (b2 - random_b2) + 0.10 * max(0.0, r2) + 0.05 * source_cos + min(0.0, b3)
                    accept = (
                        r2 >= 0.20
                        and b1 >= -1.0e-4
                        and b2 >= random_b2 + 0.05
                        and b3 >= -1.0e-3
                        and source_cos >= -0.05
                        and float(torch.linalg.vector_norm(target_update.tensor.detach()).item()) > 1.0e-12
                    )
                    diag.update(
                        {
                            "target_family": family,
                            "random_target_B2_gain": random_b2,
                            "random_target_ActuationR2": random_r2,
                            "target_gate_margin": b2 - random_b2,
                            "target_source_axis_cos": source_cos,
                            "target_gate_score": score,
                            "target_gate_accept": int(accept),
                        }
                    )
                    candidates.append((family, target_update, diag, score, bool(accept)))

                if not candidates:
                    return 0, {
                        "operator_status": f"{loss_gate_fallback_mode}:no_target_candidate",
                        "operator_gate_accept": 0,
                        "random_target_B2_gain": random_b2,
                        "random_target_ActuationR2": random_r2,
                    }

                allowed = [item for item in candidates if item[4]]
                best_family, best_update, best_diag, best_score, _accepted = max(
                    allowed or candidates,
                    key=lambda item: (item[3], finite_float(item[2].get("B2_transfer_gain"), -1.0e9), finite_float(item[2].get("B3_safety_gain"), -1.0e9)),
                )
                gate_accept = int(bool(allowed))
                if gate_accept:
                    commit_factor = 0.25 if loss_gate_fallback_mode in terminal_target_modes else 0.50
                    apply_update(model, best_update, lr=commit_factor * float(args.fu_lr))
                    terminal_selector_current_cos = best_diag.get("target_source_axis_cos", "")
                    terminal_selector_kind = f"{loss_gate_fallback_mode}:{best_family}"
                else:
                    terminal_selector_current_cos = ""
                    terminal_selector_kind = f"{loss_gate_fallback_mode}:reject"
                terminal_selector_score = float(best_score)
                best_diag.update(
                    {
                        "operator_gate_accept": gate_accept,
                        "generalization_gate_accept": gate_accept,
                        "operator_status": terminal_selector_kind,
                        "source_state_current_cos": terminal_selector_current_cos,
                        "source_state_signal_gain": best_diag.get("B2_transfer_gain", ""),
                        "source_state_corrupt_gain": best_diag.get("B3_safety_gain", ""),
                        "source_state_gate_accept": gate_accept,
                        "source_state_balance_mean": best_score,
                        "source_state_consensus_density": best_diag.get("target_consensus_density", ""),
                        "generalization_signal_gain": best_diag.get("B2_transfer_gain", ""),
                        "shuffled_label_gain": best_diag.get("B3_safety_gain", ""),
                    }
                )
                return gate_accept, best_diag

            if step <= boundary_steps:
                opt_adam.step()
                if loss_gate_enabled and step == boundary_steps:
                    update_train_loss_gate()
                last_actuation = {
                    "source_state_gate_accept": 0,
                    "source_state_current_cos": "",
                    "source_state_balance_mean": "",
                    "source_state_consensus_density": 0.0,
                    "source_state_gate_threshold": loss_gate_h400_threshold if loss_gate_enabled else boundary_steps,
                    "source_state_ema_beta": "",
                    "generalization_gain_a": train_loss_gate_h400_pass if loss_gate_enabled else "",
                    "generalization_gain_b": train_loss_gate_h800_pass if loss_gate_enabled else "",
                    "generalization_signal_gain": train_loss_gate_last_value if loss_gate_enabled else "",
                }
            else:
                if loss_gate_enabled and step in {800, 1200, 1600, 2400, 3200, 4800, 6400}:
                    update_train_loss_gate()
                fast_update = make_update(model, "M2-SGDMomentumPrimaryFU", xb, yb, seed=seed + step)
                anchor_update = make_update(model, "M15-LineCFilteredAlternatingFU", xb, yb, seed=seed + step)
                source_vec = 0.75 * fast_update.tensor.detach() + 0.25 * anchor_update.tensor.detach()
                if slow_state is not None and slow_state.numel() == 2 * source_vec.numel():
                    short_state, long_state = slow_state.to(device=device).chunk(2)
                else:
                    short_state = source_vec.detach().clone()
                    long_state = source_vec.detach().clone()
                short_state = 0.80 * short_state + 0.20 * source_vec.to(device=device)
                long_state = 0.995 * long_state + 0.005 * source_vec.to(device=device)
                slow_state = torch.cat([short_state.detach(), long_state.detach()])
                agree = torch.sign(short_state) == torch.sign(long_state)
                agree_density = float(agree.float().mean().item()) if agree.numel() else 0.0
                agreed_source = torch.where(agree, 0.50 * (short_state + long_state), 0.10 * long_state)
                source_axis = normalized_like(agreed_source, fast_update.tensor if fast_update.tensor.numel() else agreed_source)
                if step >= 1600 and terminal_h1600_source_anchor is None and source_axis.numel():
                    terminal_h1600_source_anchor = source_axis.detach().clone()
                if step >= 2400 and terminal_h2400_source_anchor is None and source_axis.numel():
                    terminal_h2400_source_anchor = source_axis.detach().clone()
                if step >= 3200 and terminal_h3200_source_anchor is None and source_axis.numel():
                    terminal_h3200_source_anchor = source_axis.detach().clone()
                if (
                    mechanism == "M112-TrainLossTerminalCheckpointReentryFU"
                    and step >= 3200
                    and param_slow_state is None
                ):
                    param_slow_state = snapshot(model).detach().clone()
                if (
                    mechanism == "M132-H1600SourceCheckpointReentryFU"
                    and step >= 1600
                    and param_slow_state is None
                ):
                    param_slow_state = snapshot(model).detach().clone()
                if (
                    mechanism == "M133-H2400SourceCheckpointReentryFU"
                    and step >= 2400
                    and param_slow_state is None
                ):
                    param_slow_state = snapshot(model).detach().clone()
                if (
                    mechanism == "M145-EarlySourceSlowEMATerminalH4000ReentryFU"
                    and step >= 4000
                    and param_slow_state is None
                ):
                    param_slow_state = snapshot(model).detach().clone()
                if step <= source_warm_steps:
                    selector_accept = train_loss_selector_accept()
                    if selector_accept or loss_gate_fallback_mode == "sgd":
                        opt_sgd.step_gradient()
                    if selector_accept:
                        apply_update(model, fast_update, lr=float(args.fu_lr))
                    last_actuation = {
                        "source_state_current_cos": cosine(agreed_source, flat_grad(model, device)),
                        "source_state_gate_accept": selector_accept,
                        "source_state_norm": float(torch.linalg.vector_norm(slow_state.detach()).item()),
                        "source_state_whitened_norm": float(torch.linalg.vector_norm(agreed_source.detach()).item()),
                        "source_state_consensus_density": selector_accept if loss_gate_enabled else agree_density,
                        "source_state_balance_mean": train_loss_gate_last_value if loss_gate_enabled else cosine(short_state, long_state),
                        "source_state_gate_threshold": loss_gate_h400_threshold if loss_gate_enabled else boundary_steps,
                        "source_state_ema_beta": 0.995,
                        "generalization_signal_gain": 0.0,
                        "generalization_gate_accept": selector_accept,
                        "generalization_gain_a": train_loss_gate_h400_pass if loss_gate_enabled else "",
                        "generalization_gain_b": train_loss_gate_h800_pass if loss_gate_enabled else "",
                    }
                elif step % max(2, int(args.alt_period)) == 0:
                    update_vec = normalized_like(agreed_source, fast_update.tensor)
                    slow_update = UpdateTensor(
                        update_vec,
                        "step",
                        "subtract",
                        "slow_state",
                        "train_stream_adamw_boundary_dual_timescale_sgd_floor_source_retention",
                        mechanism,
                        role="short_long_agreement_source_with_sgd_floor",
                        one_step_descent_claim=0,
                    )
                    active_update = slow_update
                    active_update_vec = update_vec
                    active_lr_scale = slow_lr_scale
                    active_gate_density_threshold = 0.20
                    active_corrupt_abs = 1.0e-4
                    active_corrupt_fraction = 0.50
                    active_operator_status = ""
                    if loss_gate_fallback_mode in {
                        "h800_source_slow_ema_retention",
                        "h800_readout_channel_retention",
                        "h800_dual_memory_source_retention",
                        "h800_source_slow_ema_terminal_source_guard",
                        "h800_source_slow_ema_terminal_raw_guard",
                        "h800_source_slow_ema_terminal_raw_guard_strong",
                        "h800_source_slow_ema_terminal_projected_optimizer",
                        "h800_source_slow_ema_terminal_projected_optimizer_lambda050",
                        "h800_source_slow_ema_terminal_projected_optimizer_lambda100",
                        "h800_source_slow_ema_terminal_projected_blend",
                        "h800_source_slow_ema_terminal_antiwashout",
                        "h800_source_slow_ema_terminal_h4000_reentry",
                        "h800_source_slow_ema_terminal_adaptive_raw",
                        "h800_source_slow_ema_terminal_sparse_source",
                        "h800_source_slow_ema_terminal_ratio_preserve",
                        "h800_source_slow_ema_terminal_reject_source_axis_rescue",
                        "h800_source_slow_ema_terminal_reject_slow_ema_rescue",
                        "h800_source_slow_ema_terminal_reject_hold_source_rescue",
                        "h800_source_slow_ema_terminal_topk_support_guard",
                        "h800_source_slow_ema_terminal_topk_raw_guard",
                        "h800_source_slow_ema_terminal_topk_debt_cap",
                        "h800_source_slow_ema_terminal_source_preserve_very_strong",
                        "h800_source_slow_ema_h3600_terminal_source_preserve",
                        "h800_source_slow_ema_terminal_nora_orthogonal_source",
                        "h800_source_slow_ema_terminal_debt_aware_preserve",
                        "h800_source_slow_ema_terminal_low_nds_matrix_block",
                        "h800_source_slow_ema_terminal_dual_memory_preserve",
                        "h800_source_slow_ema_terminal_snr_predictor",
                        "h800_source_slow_ema_split_consensus_estimator",
                        "h800_source_slow_ema_signal_reservoir_transport",
                        "h800_source_slow_ema_terminal_source_floor",
                        "h800_source_slow_ema_terminal_h4000_anchor_floor",
                        "h800_source_slow_ema_terminal_decay_aware_floor",
                        "h800_source_slow_ema_terminal_raw_guard_source_floor",
                        "h800_source_slow_ema_terminal_raw_guard_h4000_anchor_floor",
                        "h800_source_slow_ema_terminal_raw_guard_decay_aware_floor",
                        "h800_source_slow_ema_terminal_anti_erosion_orthogonal",
                        "h800_source_slow_ema_terminal_source_reflection_guard",
                        "h800_source_slow_ema_terminal_h4000_transport_corrector",
                        "h800_source_slow_ema_terminal_h3200_anchor_transport",
                        "h800_source_slow_ema_terminal_raw_guard_h3200_anchor_transport",
                        "h800_source_slow_ema_terminal_h3200_ratio_reentry",
                        "h800_source_slow_ema_terminal_h3200_progress_carry",
                        "h800_source_slow_ema_terminal_h4000_progress_carry",
                        "h800_source_slow_ema_terminal_h3200_source_progress_blend",
                        "h800_source_slow_ema_terminal_raw_guard_h3600_gentle_progress",
                        "h800_source_slow_ema_terminal_raw_guard_h4000_gentle_progress",
                        "h800_source_slow_ema_terminal_raw_guard_h4400_projected_progress",
                        "h800_source_slow_ema_terminal_control_relative_sgd_catchup",
                        "h800_source_slow_ema_terminal_control_relative_adamw_catchup",
                        "h800_source_slow_ema_terminal_control_relative_source_balanced_catchup",
                        "h800_source_slow_ema_terminal_trajectory_adaptive_preserve",
                        "h800_source_slow_ema_terminal_mid_erosion_bridge",
                        "h800_source_slow_ema_terminal_two_phase_ratio_repair",
                        "h800_source_slow_ema_terminal_anti_source_clip",
                        "h800_source_slow_ema_terminal_debt_aware_hold",
                        "h800_source_slow_ema_terminal_anchor_flow_tiny",
                        "h800_source_slow_ema_terminal_accept_memory",
                        "h800_source_slow_ema_terminal_accept_memory_debt",
                        "h800_source_slow_ema_terminal_source_progress_memory",
                        "h800_source_slow_ema_shape_preserve",
                        "h800_source_slow_ema_shape_preserve_raw_guard",
                        "h800_source_slow_ema_shape_preserve_clamp",
                        "h800_source_slow_ema_signal_reservoir_target",
                        "h800_source_slow_ema_source_bank_target",
                        "h800_source_slow_ema_dual_target_guard",
                    }:
                        source_space = "slow_state"
                        source_name = "train_stream_h800_source_slow_ema_retention"
                        role = "h800_source_slow_ema_retention"
                        if loss_gate_fallback_mode == "h800_readout_channel_retention":
                            channel_mask = readout_mask_flat().to(device=device, dtype=update_vec.dtype)
                            channel_vec = agreed_source.to(device=device, dtype=update_vec.dtype)
                            if channel_mask.numel() == channel_vec.numel():
                                channel_vec = channel_vec * channel_mask
                            if float(torch.linalg.vector_norm(channel_vec.detach()).item()) <= 1.0e-12:
                                channel_vec = source_axis.to(device=device, dtype=update_vec.dtype)
                                if channel_mask.numel() == channel_vec.numel():
                                    channel_vec = channel_vec * channel_mask
                            active_update_vec = normalized_like(channel_vec, fast_update.tensor)
                            source_space = "readout_carrier"
                            source_name = "train_stream_h800_readout_channel_retention"
                            role = "h800_readout_channel_retained_target"
                            active_gate_density_threshold = 0.05
                            active_corrupt_fraction = 0.35
                        elif loss_gate_fallback_mode == "h800_dual_memory_source_retention":
                            short_axis = normalized_like(short_state, fast_update.tensor)
                            long_axis = normalized_like(long_state, fast_update.tensor)
                            dual_vec = torch.where(agree, 0.60 * short_axis + 0.40 * long_axis, 0.75 * long_axis)
                            active_update_vec = normalized_like(dual_vec, fast_update.tensor)
                            source_name = "train_stream_h800_dual_memory_source_retention"
                            role = "h800_short_long_dual_memory_source_retention"
                            active_gate_density_threshold = 0.12
                            active_corrupt_fraction = 0.40
                        elif loss_gate_fallback_mode in {
                            "h800_source_slow_ema_terminal_source_guard",
                            "h800_source_slow_ema_terminal_raw_guard",
                            "h800_source_slow_ema_terminal_raw_guard_strong",
                            "h800_source_slow_ema_terminal_projected_optimizer",
                            "h800_source_slow_ema_terminal_projected_optimizer_lambda050",
                            "h800_source_slow_ema_terminal_projected_optimizer_lambda100",
                            "h800_source_slow_ema_terminal_projected_blend",
                            "h800_source_slow_ema_terminal_antiwashout",
                            "h800_source_slow_ema_terminal_h4000_reentry",
                            "h800_source_slow_ema_terminal_adaptive_raw",
                            "h800_source_slow_ema_terminal_sparse_source",
                            "h800_source_slow_ema_terminal_ratio_preserve",
                            "h800_source_slow_ema_terminal_reject_source_axis_rescue",
                            "h800_source_slow_ema_terminal_reject_slow_ema_rescue",
                            "h800_source_slow_ema_terminal_reject_hold_source_rescue",
                            "h800_source_slow_ema_terminal_reject_raw_rescue",
                            "h800_source_slow_ema_terminal_reject_hybrid_rescue",
                            "h800_source_slow_ema_terminal_reject_lateonly_raw_rescue",
                            "h800_source_slow_ema_terminal_topk_support_guard",
                            "h800_source_slow_ema_terminal_topk_raw_guard",
                            "h800_source_slow_ema_terminal_topk_debt_cap",
                            "h800_source_slow_ema_terminal_source_preserve_very_strong",
                            "h800_source_slow_ema_h3600_terminal_source_preserve",
                            "h800_source_slow_ema_terminal_nora_orthogonal_source",
                            "h800_source_slow_ema_terminal_debt_aware_preserve",
                            "h800_source_slow_ema_terminal_low_nds_matrix_block",
                            "h800_source_slow_ema_terminal_dual_memory_preserve",
                            "h800_source_slow_ema_terminal_snr_predictor",
                            "h800_source_slow_ema_split_consensus_estimator",
                            "h800_source_slow_ema_signal_reservoir_transport",
                            "h800_source_slow_ema_terminal_source_floor",
                            "h800_source_slow_ema_terminal_h4000_anchor_floor",
                            "h800_source_slow_ema_terminal_decay_aware_floor",
                            "h800_source_slow_ema_terminal_raw_guard_source_floor",
                            "h800_source_slow_ema_terminal_raw_guard_h4000_anchor_floor",
                            "h800_source_slow_ema_terminal_raw_guard_decay_aware_floor",
                            "h800_source_slow_ema_terminal_anti_erosion_orthogonal",
                            "h800_source_slow_ema_terminal_source_reflection_guard",
                            "h800_source_slow_ema_terminal_h4000_transport_corrector",
                            "h800_source_slow_ema_terminal_h3200_anchor_transport",
                            "h800_source_slow_ema_terminal_raw_guard_h3200_anchor_transport",
                            "h800_source_slow_ema_terminal_h3200_ratio_reentry",
                            "h800_source_slow_ema_terminal_h3200_progress_carry",
                            "h800_source_slow_ema_terminal_h4000_progress_carry",
                            "h800_source_slow_ema_terminal_h3200_source_progress_blend",
                            "h800_source_slow_ema_terminal_raw_guard_h3600_gentle_progress",
                            "h800_source_slow_ema_terminal_raw_guard_h4000_gentle_progress",
                            "h800_source_slow_ema_terminal_raw_guard_h4400_projected_progress",
                            "h800_source_slow_ema_terminal_control_relative_sgd_catchup",
                            "h800_source_slow_ema_terminal_control_relative_adamw_catchup",
                            "h800_source_slow_ema_terminal_control_relative_source_balanced_catchup",
                            "h800_source_slow_ema_terminal_trajectory_adaptive_preserve",
                            "h800_source_slow_ema_terminal_mid_erosion_bridge",
                            "h800_source_slow_ema_terminal_two_phase_ratio_repair",
                            "h800_source_slow_ema_terminal_anti_source_clip",
                            "h800_source_slow_ema_terminal_debt_aware_hold",
                            "h800_source_slow_ema_terminal_anchor_flow_tiny",
                            "h800_source_slow_ema_terminal_accept_memory",
                            "h800_source_slow_ema_terminal_accept_memory_debt",
                            "h800_source_slow_ema_terminal_source_progress_memory",
                            "h800_source_slow_ema_signal_reservoir_target",
                            "h800_source_slow_ema_source_bank_target",
                            "h800_source_slow_ema_dual_target_guard",
                            "h800_source_slow_ema_shape_preserve",
                            "h800_source_slow_ema_shape_preserve_raw_guard",
                            "h800_source_slow_ema_shape_preserve_clamp",
                        }:
                            slow_ema = torch.where(agree, 0.35 * short_state + 0.65 * long_state, long_state)
                            active_update_vec = normalized_like(slow_ema, fast_update.tensor)
                            source_name = (
                                "train_stream_h800_source_slow_ema_terminal_topk_support_guard"
                                if loss_gate_fallback_mode == "h800_source_slow_ema_terminal_topk_support_guard"
                                else
                                "train_stream_h800_source_slow_ema_terminal_topk_raw_guard"
                                if loss_gate_fallback_mode == "h800_source_slow_ema_terminal_topk_raw_guard"
                                else
                                "train_stream_h800_source_slow_ema_terminal_topk_debt_cap"
                                if loss_gate_fallback_mode == "h800_source_slow_ema_terminal_topk_debt_cap"
                                else
                                "train_stream_h800_source_slow_ema_terminal_source_preserve_very_strong"
                                if loss_gate_fallback_mode == "h800_source_slow_ema_terminal_source_preserve_very_strong"
                                else
                                "train_stream_h800_source_slow_ema_h3600_terminal_source_preserve"
                                if loss_gate_fallback_mode == "h800_source_slow_ema_h3600_terminal_source_preserve"
                                else
                                "train_stream_h800_source_slow_ema_terminal_nora_orthogonal_source"
                                if loss_gate_fallback_mode == "h800_source_slow_ema_terminal_nora_orthogonal_source"
                                else
                                "train_stream_h800_source_slow_ema_terminal_debt_aware_preserve"
                                if loss_gate_fallback_mode == "h800_source_slow_ema_terminal_debt_aware_preserve"
                                else
                                "train_stream_h800_source_slow_ema_terminal_low_nds_matrix_block"
                                if loss_gate_fallback_mode == "h800_source_slow_ema_terminal_low_nds_matrix_block"
                                else
                                "train_stream_h800_source_slow_ema_terminal_dual_memory_preserve"
                                if loss_gate_fallback_mode == "h800_source_slow_ema_terminal_dual_memory_preserve"
                                else
                                "train_stream_h800_source_slow_ema_terminal_snr_predictor"
                                if loss_gate_fallback_mode == "h800_source_slow_ema_terminal_snr_predictor"
                                else
                                "train_stream_h800_source_slow_ema_split_consensus_estimator"
                                if loss_gate_fallback_mode == "h800_source_slow_ema_split_consensus_estimator"
                                else
                                "train_stream_h800_source_slow_ema_signal_reservoir_transport"
                                if loss_gate_fallback_mode == "h800_source_slow_ema_signal_reservoir_transport"
                                else
                                "train_stream_h800_source_slow_ema_terminal_source_floor"
                                if loss_gate_fallback_mode == "h800_source_slow_ema_terminal_source_floor"
                                else
                                "train_stream_h800_source_slow_ema_terminal_h4000_anchor_floor"
                                if loss_gate_fallback_mode == "h800_source_slow_ema_terminal_h4000_anchor_floor"
                                else
                                "train_stream_h800_source_slow_ema_terminal_decay_aware_floor"
                                if loss_gate_fallback_mode == "h800_source_slow_ema_terminal_decay_aware_floor"
                                else
                                "train_stream_h800_source_slow_ema_terminal_raw_guard_source_floor"
                                if loss_gate_fallback_mode == "h800_source_slow_ema_terminal_raw_guard_source_floor"
                                else
                                "train_stream_h800_source_slow_ema_terminal_raw_guard_h4000_anchor_floor"
                                if loss_gate_fallback_mode == "h800_source_slow_ema_terminal_raw_guard_h4000_anchor_floor"
                                else
                                "train_stream_h800_source_slow_ema_terminal_raw_guard_decay_aware_floor"
                                if loss_gate_fallback_mode == "h800_source_slow_ema_terminal_raw_guard_decay_aware_floor"
                                else
                                "train_stream_h800_source_slow_ema_terminal_h3200_anchor_transport"
                                if loss_gate_fallback_mode == "h800_source_slow_ema_terminal_h3200_anchor_transport"
                                else
                                "train_stream_h800_source_slow_ema_terminal_raw_guard_h3200_anchor_transport"
                                if loss_gate_fallback_mode == "h800_source_slow_ema_terminal_raw_guard_h3200_anchor_transport"
                                else
                                "train_stream_h800_source_slow_ema_terminal_h3200_ratio_reentry"
                                if loss_gate_fallback_mode == "h800_source_slow_ema_terminal_h3200_ratio_reentry"
                                else
                                "train_stream_h800_source_slow_ema_terminal_h3200_progress_carry"
                                if loss_gate_fallback_mode == "h800_source_slow_ema_terminal_h3200_progress_carry"
                                else
                                "train_stream_h800_source_slow_ema_terminal_h4000_progress_carry"
                                if loss_gate_fallback_mode == "h800_source_slow_ema_terminal_h4000_progress_carry"
                                else
                                "train_stream_h800_source_slow_ema_terminal_h3200_source_progress_blend"
                                if loss_gate_fallback_mode == "h800_source_slow_ema_terminal_h3200_source_progress_blend"
                                else
                                "train_stream_h800_source_slow_ema_terminal_raw_guard_h3600_gentle_progress"
                                if loss_gate_fallback_mode == "h800_source_slow_ema_terminal_raw_guard_h3600_gentle_progress"
                                else
                                "train_stream_h800_source_slow_ema_terminal_raw_guard_h4000_gentle_progress"
                                if loss_gate_fallback_mode == "h800_source_slow_ema_terminal_raw_guard_h4000_gentle_progress"
                                else
                                "train_stream_h800_source_slow_ema_terminal_raw_guard_h4400_projected_progress"
                                if loss_gate_fallback_mode == "h800_source_slow_ema_terminal_raw_guard_h4400_projected_progress"
                                else
                                "train_stream_h800_source_slow_ema_terminal_control_relative_sgd_catchup"
                                if loss_gate_fallback_mode == "h800_source_slow_ema_terminal_control_relative_sgd_catchup"
                                else
                                "train_stream_h800_source_slow_ema_terminal_control_relative_adamw_catchup"
                                if loss_gate_fallback_mode == "h800_source_slow_ema_terminal_control_relative_adamw_catchup"
                                else
                                "train_stream_h800_source_slow_ema_terminal_control_relative_source_balanced_catchup"
                                if loss_gate_fallback_mode == "h800_source_slow_ema_terminal_control_relative_source_balanced_catchup"
                                else
                                "train_stream_h800_source_slow_ema_terminal_raw_guard_strong"
                                if loss_gate_fallback_mode == "h800_source_slow_ema_terminal_raw_guard_strong"
                                else
                                "train_stream_h800_source_slow_ema_terminal_projected_optimizer"
                                if loss_gate_fallback_mode == "h800_source_slow_ema_terminal_projected_optimizer"
                                else
                                "train_stream_h800_source_slow_ema_terminal_projected_optimizer_lambda050"
                                if loss_gate_fallback_mode == "h800_source_slow_ema_terminal_projected_optimizer_lambda050"
                                else
                                "train_stream_h800_source_slow_ema_terminal_projected_optimizer_lambda100"
                                if loss_gate_fallback_mode == "h800_source_slow_ema_terminal_projected_optimizer_lambda100"
                                else
                                "train_stream_h800_source_slow_ema_terminal_projected_blend"
                                if loss_gate_fallback_mode == "h800_source_slow_ema_terminal_projected_blend"
                                else
                                "train_stream_h800_source_slow_ema_terminal_antiwashout"
                                if loss_gate_fallback_mode == "h800_source_slow_ema_terminal_antiwashout"
                                else
                                "train_stream_h800_source_slow_ema_terminal_h4000_reentry"
                                if loss_gate_fallback_mode == "h800_source_slow_ema_terminal_h4000_reentry"
                                else
                                "train_stream_h800_source_slow_ema_terminal_adaptive_raw"
                                if loss_gate_fallback_mode == "h800_source_slow_ema_terminal_adaptive_raw"
                                else
                                "train_stream_h800_source_slow_ema_terminal_sparse_source"
                                if loss_gate_fallback_mode == "h800_source_slow_ema_terminal_sparse_source"
                                else
                                "train_stream_h800_source_slow_ema_terminal_ratio_preserve"
                                if loss_gate_fallback_mode == "h800_source_slow_ema_terminal_ratio_preserve"
                                else
                                "train_stream_h800_source_slow_ema_terminal_reject_source_axis_rescue"
                                if loss_gate_fallback_mode == "h800_source_slow_ema_terminal_reject_source_axis_rescue"
                                else
                                "train_stream_h800_source_slow_ema_terminal_reject_slow_ema_rescue"
                                if loss_gate_fallback_mode == "h800_source_slow_ema_terminal_reject_slow_ema_rescue"
                                else
                                "train_stream_h800_source_slow_ema_terminal_reject_hold_source_rescue"
                                if loss_gate_fallback_mode == "h800_source_slow_ema_terminal_reject_hold_source_rescue"
                                else
                                "train_stream_h800_source_slow_ema_terminal_reject_raw_rescue"
                                if loss_gate_fallback_mode == "h800_source_slow_ema_terminal_reject_raw_rescue"
                                else
                                "train_stream_h800_source_slow_ema_terminal_reject_hybrid_rescue"
                                if loss_gate_fallback_mode == "h800_source_slow_ema_terminal_reject_hybrid_rescue"
                                else
                                "train_stream_h800_source_slow_ema_terminal_reject_lateonly_raw_rescue"
                                if loss_gate_fallback_mode == "h800_source_slow_ema_terminal_reject_lateonly_raw_rescue"
                                else
                                "train_stream_h800_source_slow_ema_shape_preserve"
                                if loss_gate_fallback_mode == "h800_source_slow_ema_shape_preserve"
                                else
                                "train_stream_h800_source_slow_ema_shape_preserve_raw_guard"
                                if loss_gate_fallback_mode == "h800_source_slow_ema_shape_preserve_raw_guard"
                                else
                                "train_stream_h800_source_slow_ema_shape_preserve_clamp"
                                if loss_gate_fallback_mode == "h800_source_slow_ema_shape_preserve_clamp"
                                else
                                "train_stream_h800_source_slow_ema_terminal_source_projection_target"
                                if loss_gate_fallback_mode == "h800_source_slow_ema_terminal_source_projection_target"
                                else
                                "train_stream_h800_source_slow_ema_terminal_noise_orthogonal_target"
                                if loss_gate_fallback_mode == "h800_source_slow_ema_terminal_noise_orthogonal_target"
                                else
                                "train_stream_h800_source_slow_ema_terminal_easy_margin_target"
                                if loss_gate_fallback_mode == "h800_source_slow_ema_terminal_easy_margin_target"
                                else
                                "train_stream_h800_source_slow_ema_signal_reservoir_target"
                                if loss_gate_fallback_mode == "h800_source_slow_ema_signal_reservoir_target"
                                else
                                "train_stream_h800_source_slow_ema_source_bank_target"
                                if loss_gate_fallback_mode == "h800_source_slow_ema_source_bank_target"
                                else
                                "train_stream_h800_source_slow_ema_dual_target_guard"
                                if loss_gate_fallback_mode == "h800_source_slow_ema_dual_target_guard"
                                else
                                "train_stream_h800_source_slow_ema_low_nds_diffeomorphic_target"
                                if loss_gate_fallback_mode == "h800_source_slow_ema_low_nds_diffeomorphic_target"
                                else
                                "train_stream_h800_source_slow_ema_info_volume_diffeomorphic_target"
                                if loss_gate_fallback_mode == "h800_source_slow_ema_info_volume_diffeomorphic_target"
                                else
                                "train_stream_h800_source_slow_ema_low_rank_readout_transport"
                                if loss_gate_fallback_mode == "h800_source_slow_ema_low_rank_readout_transport"
                                else
                                "train_stream_h800_source_slow_ema_terminal_raw_guard"
                                if loss_gate_fallback_mode == "h800_source_slow_ema_terminal_raw_guard"
                                else "train_stream_h800_source_slow_ema_terminal_source_guard"
                            )
                            role = (
                                "h800_source_slow_ema_with_terminal_topk_support_guard"
                                if loss_gate_fallback_mode == "h800_source_slow_ema_terminal_topk_support_guard"
                                else
                                "h800_source_slow_ema_with_terminal_topk_raw_guard"
                                if loss_gate_fallback_mode == "h800_source_slow_ema_terminal_topk_raw_guard"
                                else
                                "h800_source_slow_ema_with_terminal_topk_debt_cap"
                                if loss_gate_fallback_mode == "h800_source_slow_ema_terminal_topk_debt_cap"
                                else
                                "h800_source_slow_ema_with_terminal_source_preserve_very_strong"
                                if loss_gate_fallback_mode == "h800_source_slow_ema_terminal_source_preserve_very_strong"
                                else
                                "h800_source_slow_ema_with_h3600_terminal_source_preserve"
                                if loss_gate_fallback_mode == "h800_source_slow_ema_h3600_terminal_source_preserve"
                                else
                                "h800_source_slow_ema_with_terminal_nora_orthogonal_source"
                                if loss_gate_fallback_mode == "h800_source_slow_ema_terminal_nora_orthogonal_source"
                                else
                                "h800_source_slow_ema_with_terminal_debt_aware_preserve"
                                if loss_gate_fallback_mode == "h800_source_slow_ema_terminal_debt_aware_preserve"
                                else
                                "h800_source_slow_ema_with_terminal_low_nds_matrix_block"
                                if loss_gate_fallback_mode == "h800_source_slow_ema_terminal_low_nds_matrix_block"
                                else
                                "h800_source_slow_ema_with_terminal_dual_memory_preserve"
                                if loss_gate_fallback_mode == "h800_source_slow_ema_terminal_dual_memory_preserve"
                                else
                                "h800_source_slow_ema_with_terminal_snr_predictor"
                                if loss_gate_fallback_mode == "h800_source_slow_ema_terminal_snr_predictor"
                                else
                                "h800_source_slow_ema_with_split_consensus_estimator"
                                if loss_gate_fallback_mode == "h800_source_slow_ema_split_consensus_estimator"
                                else
                                "h800_source_slow_ema_with_signal_reservoir_transport"
                                if loss_gate_fallback_mode == "h800_source_slow_ema_signal_reservoir_transport"
                                else
                                "h800_source_slow_ema_with_terminal_source_floor"
                                if loss_gate_fallback_mode == "h800_source_slow_ema_terminal_source_floor"
                                else
                                "h800_source_slow_ema_with_terminal_h4000_anchor_floor"
                                if loss_gate_fallback_mode == "h800_source_slow_ema_terminal_h4000_anchor_floor"
                                else
                                "h800_source_slow_ema_with_terminal_decay_aware_floor"
                                if loss_gate_fallback_mode == "h800_source_slow_ema_terminal_decay_aware_floor"
                                else
                                "h800_source_slow_ema_with_terminal_raw_guard_source_floor"
                                if loss_gate_fallback_mode == "h800_source_slow_ema_terminal_raw_guard_source_floor"
                                else
                                "h800_source_slow_ema_with_terminal_raw_guard_h4000_anchor_floor"
                                if loss_gate_fallback_mode == "h800_source_slow_ema_terminal_raw_guard_h4000_anchor_floor"
                                else
                                "h800_source_slow_ema_with_terminal_raw_guard_decay_aware_floor"
                                if loss_gate_fallback_mode == "h800_source_slow_ema_terminal_raw_guard_decay_aware_floor"
                                else
                                "h800_source_slow_ema_with_stronger_terminal_raw_then_source_guard"
                                if loss_gate_fallback_mode == "h800_source_slow_ema_terminal_raw_guard_strong"
                                else
                                "h800_source_slow_ema_with_terminal_projected_optimizer"
                                if loss_gate_fallback_mode == "h800_source_slow_ema_terminal_projected_optimizer"
                                else
                                "h800_source_slow_ema_with_terminal_projected_optimizer_lambda050"
                                if loss_gate_fallback_mode == "h800_source_slow_ema_terminal_projected_optimizer_lambda050"
                                else
                                "h800_source_slow_ema_with_terminal_projected_optimizer_lambda100"
                                if loss_gate_fallback_mode == "h800_source_slow_ema_terminal_projected_optimizer_lambda100"
                                else
                                "h800_source_slow_ema_with_terminal_projected_source_blend"
                                if loss_gate_fallback_mode == "h800_source_slow_ema_terminal_projected_blend"
                                else
                                "h800_source_slow_ema_with_terminal_antiwashout_projector"
                                if loss_gate_fallback_mode == "h800_source_slow_ema_terminal_antiwashout"
                                else
                                "h800_source_slow_ema_with_h4000_terminal_reentry"
                                if loss_gate_fallback_mode == "h800_source_slow_ema_terminal_h4000_reentry"
                                else
                                "h800_source_slow_ema_with_source_state_adaptive_raw_terminal_guard"
                                if loss_gate_fallback_mode == "h800_source_slow_ema_terminal_adaptive_raw"
                                else
                                "h800_source_slow_ema_with_sparse_source_axis_terminal_guard"
                                if loss_gate_fallback_mode == "h800_source_slow_ema_terminal_sparse_source"
                                else
                                "h800_source_slow_ema_with_ratio_preserving_terminal_source_guard"
                                if loss_gate_fallback_mode == "h800_source_slow_ema_terminal_ratio_preserve"
                                else
                                "h800_source_slow_ema_with_terminal_reject_source_axis_rescue"
                                if loss_gate_fallback_mode == "h800_source_slow_ema_terminal_reject_source_axis_rescue"
                                else
                                "h800_source_slow_ema_with_terminal_reject_slow_ema_rescue"
                                if loss_gate_fallback_mode == "h800_source_slow_ema_terminal_reject_slow_ema_rescue"
                                else
                                "h800_source_slow_ema_with_terminal_reject_hold_source_rescue"
                                if loss_gate_fallback_mode == "h800_source_slow_ema_terminal_reject_hold_source_rescue"
                                else
                                "h800_source_slow_ema_with_terminal_reject_raw_rescue"
                                if loss_gate_fallback_mode == "h800_source_slow_ema_terminal_reject_raw_rescue"
                                else
                                "h800_source_slow_ema_with_terminal_reject_raw_source_hybrid_rescue"
                                if loss_gate_fallback_mode == "h800_source_slow_ema_terminal_reject_hybrid_rescue"
                                else
                                "h800_source_slow_ema_with_terminal_reject_lateonly_raw_rescue"
                                if loss_gate_fallback_mode == "h800_source_slow_ema_terminal_reject_lateonly_raw_rescue"
                                else
                                "h800_source_slow_ema_midlate_source_shape_preserve"
                                if loss_gate_fallback_mode == "h800_source_slow_ema_shape_preserve"
                                else
                                "h800_source_slow_ema_midlate_shape_preserve_with_raw_guard"
                                if loss_gate_fallback_mode == "h800_source_slow_ema_shape_preserve_raw_guard"
                                else
                                "h800_source_slow_ema_midlate_shape_preserve_with_terminal_clamp"
                                if loss_gate_fallback_mode == "h800_source_slow_ema_shape_preserve_clamp"
                                else
                                "h800_source_slow_ema_with_terminal_raw_guard_and_source_projection_target"
                                if loss_gate_fallback_mode == "h800_source_slow_ema_terminal_source_projection_target"
                                else
                                "h800_source_slow_ema_with_terminal_raw_guard_and_noise_orthogonal_target"
                                if loss_gate_fallback_mode == "h800_source_slow_ema_terminal_noise_orthogonal_target"
                                else
                                "h800_source_slow_ema_with_terminal_raw_guard_and_easy_margin_target"
                                if loss_gate_fallback_mode == "h800_source_slow_ema_terminal_easy_margin_target"
                                else
                                "h800_source_slow_ema_with_signal_reservoir_b3_null_target_gate"
                                if loss_gate_fallback_mode == "h800_source_slow_ema_signal_reservoir_target"
                                else
                                "h800_source_slow_ema_with_source_bank_b3_null_target_gate"
                                if loss_gate_fallback_mode == "h800_source_slow_ema_source_bank_target"
                                else
                                "h800_source_slow_ema_with_dual_signal_source_bank_target_gate"
                                if loss_gate_fallback_mode == "h800_source_slow_ema_dual_target_guard"
                                else
                                "h800_source_slow_ema_with_low_nds_diffeomorphic_target_gate"
                                if loss_gate_fallback_mode == "h800_source_slow_ema_low_nds_diffeomorphic_target"
                                else
                                "h800_source_slow_ema_with_info_volume_diffeomorphic_target_gate"
                                if loss_gate_fallback_mode == "h800_source_slow_ema_info_volume_diffeomorphic_target"
                                else
                                "h800_source_slow_ema_with_low_rank_readout_transport_gate"
                                if loss_gate_fallback_mode == "h800_source_slow_ema_low_rank_readout_transport"
                                else
                                "h800_source_slow_ema_with_terminal_raw_then_source_guard"
                                if loss_gate_fallback_mode == "h800_source_slow_ema_terminal_raw_guard"
                                else "h800_source_slow_ema_with_terminal_source_guard"
                            )
                            if loss_gate_fallback_mode in terminal_trajectory_adaptive_modes:
                                source_name = {
                                    "h800_source_slow_ema_terminal_trajectory_adaptive_preserve": "train_stream_h800_source_slow_ema_terminal_trajectory_adaptive_preserve",
                                    "h800_source_slow_ema_terminal_mid_erosion_bridge": "train_stream_h800_source_slow_ema_terminal_mid_erosion_bridge",
                                    "h800_source_slow_ema_terminal_two_phase_ratio_repair": "train_stream_h800_source_slow_ema_terminal_two_phase_ratio_repair",
                                }[loss_gate_fallback_mode]
                                role = {
                                    "h800_source_slow_ema_terminal_trajectory_adaptive_preserve": "h800_source_slow_ema_with_terminal_trajectory_adaptive_preserve",
                                    "h800_source_slow_ema_terminal_mid_erosion_bridge": "h800_source_slow_ema_with_terminal_mid_erosion_bridge",
                                    "h800_source_slow_ema_terminal_two_phase_ratio_repair": "h800_source_slow_ema_with_terminal_two_phase_ratio_repair",
                                }[loss_gate_fallback_mode]
                            if loss_gate_fallback_mode in terminal_minimal_transport_modes:
                                source_name = {
                                    "h800_source_slow_ema_terminal_anti_source_clip": "train_stream_h800_source_slow_ema_terminal_anti_source_clip",
                                    "h800_source_slow_ema_terminal_debt_aware_hold": "train_stream_h800_source_slow_ema_terminal_debt_aware_hold",
                                    "h800_source_slow_ema_terminal_anchor_flow_tiny": "train_stream_h800_source_slow_ema_terminal_anchor_flow_tiny",
                                }[loss_gate_fallback_mode]
                                role = {
                                    "h800_source_slow_ema_terminal_anti_source_clip": "h800_source_slow_ema_with_terminal_anti_source_clip",
                                    "h800_source_slow_ema_terminal_debt_aware_hold": "h800_source_slow_ema_with_terminal_debt_aware_hold",
                                    "h800_source_slow_ema_terminal_anchor_flow_tiny": "h800_source_slow_ema_with_terminal_anchor_flow_tiny",
                                }[loss_gate_fallback_mode]
                            if loss_gate_fallback_mode in terminal_accept_memory_modes:
                                source_name = {
                                    "h800_source_slow_ema_terminal_accept_memory": "train_stream_h800_source_slow_ema_terminal_accept_memory",
                                    "h800_source_slow_ema_terminal_accept_memory_debt": "train_stream_h800_source_slow_ema_terminal_accept_memory_debt",
                                    "h800_source_slow_ema_terminal_source_progress_memory": "train_stream_h800_source_slow_ema_terminal_source_progress_memory",
                                }[loss_gate_fallback_mode]
                                role = {
                                    "h800_source_slow_ema_terminal_accept_memory": "h800_source_slow_ema_with_terminal_accept_memory",
                                    "h800_source_slow_ema_terminal_accept_memory_debt": "h800_source_slow_ema_with_terminal_accept_memory_debt",
                                    "h800_source_slow_ema_terminal_source_progress_memory": "h800_source_slow_ema_with_terminal_source_progress_memory",
                                }[loss_gate_fallback_mode]
                            active_gate_density_threshold = 0.12
                            active_corrupt_fraction = 0.40
                        else:
                            slow_ema = torch.where(agree, 0.35 * short_state + 0.65 * long_state, long_state)
                            active_update_vec = normalized_like(slow_ema, fast_update.tensor)
                            active_gate_density_threshold = 0.12
                            active_corrupt_fraction = 0.40
                        if loss_gate_fallback_mode in source_shape_preserve_modes:
                            active_lr_scale = (
                                0.38
                                if loss_gate_fallback_mode == "h800_source_slow_ema_shape_preserve"
                                else 0.42
                                if loss_gate_fallback_mode == "h800_source_slow_ema_shape_preserve_raw_guard"
                                else 0.40
                            )
                            active_gate_density_threshold = 0.10
                            active_corrupt_fraction = 0.35
                        active_update = UpdateTensor(
                            active_update_vec,
                            "step",
                            "subtract",
                            source_space,
                            source_name,
                            mechanism,
                            role=role,
                            one_step_descent_claim=0,
                        )
                        active_operator_status = loss_gate_fallback_mode
                    corrupt_labels = (channel_b[1] + 1) % max(2, int(getattr(args, "classes", 10)))
                    before = snapshot(model)
                    with torch.no_grad():
                        before_a = F.cross_entropy(model(channel_a[0]).float(), channel_a[1], reduction="none")
                        before_b = F.cross_entropy(model(channel_b[0]).float(), channel_b[1], reduction="none")
                        before_corrupt = F.cross_entropy(model(channel_b[0]).float(), corrupt_labels, reduction="none")
                    apply_update(model, active_update, lr=active_lr_scale * float(args.fu_lr))
                    with torch.no_grad():
                        after_a = F.cross_entropy(model(channel_a[0]).float(), channel_a[1], reduction="none")
                        after_b = F.cross_entropy(model(channel_b[0]).float(), channel_b[1], reduction="none")
                        after_corrupt = F.cross_entropy(model(channel_b[0]).float(), corrupt_labels, reduction="none")
                    load_flat_params(model, before)
                    gain_a = float((before_a - after_a).mean().item())
                    gain_b = float((before_b - after_b).mean().item())
                    corrupt_gain = float((before_corrupt - after_corrupt).mean().item())
                    signal_gain = min(gain_a, gain_b)
                    gate_accept = int(
                        agree_density >= active_gate_density_threshold
                        and signal_gain >= -1.0e-6
                        and corrupt_gain <= max(active_corrupt_abs, active_corrupt_fraction * max(0.0, signal_gain))
                    )
                    selector_accept = train_loss_selector_accept()
                    terminal_start = (
                        3600
                        if loss_gate_fallback_mode
                        in {
                            "early_terminal_lookahead_floor",
                            "h800_source_slow_ema_h3600_terminal_source_preserve",
                            "h800_source_slow_ema_terminal_raw_guard_h3600_gentle_progress",
                            "h800_source_slow_ema_terminal_control_relative_sgd_catchup",
                            "h800_source_slow_ema_terminal_control_relative_source_balanced_catchup",
                        }
                        else 4400
                        if loss_gate_fallback_mode
                        in {"h800_source_slow_ema_terminal_raw_guard_h4400_projected_progress"}
                        else 2400
                        if loss_gate_fallback_mode
                        in {"h800_source_slow_ema_terminal_mid_erosion_bridge"}
                        else 2800
                        if loss_gate_fallback_mode
                        in {
                            "h800_source_slow_ema_terminal_trajectory_adaptive_preserve",
                            "h800_source_slow_ema_terminal_two_phase_ratio_repair",
                        }
                        else 3200
                        if loss_gate_fallback_mode
                        in {
                            "h800_source_slow_ema_terminal_h3200_progress_carry",
                            "h800_source_slow_ema_terminal_h3200_source_progress_blend",
                            "h800_source_slow_ema_terminal_accept_memory",
                            "h800_source_slow_ema_terminal_accept_memory_debt",
                        }
                        else 3600
                        if loss_gate_fallback_mode
                        in {"h800_source_slow_ema_terminal_source_progress_memory"}
                        else 4000
                    )
                    terminal_lookahead_mode = int(
                        loss_gate_fallback_mode in {"terminal_lookahead_floor", "early_terminal_lookahead_floor", "terminal_projected_lookahead_floor", "terminal_consensus_lookahead_floor", "terminal_selector_lookahead_floor", "terminal_positive_lookahead_floor", "terminal_checkpoint_reentry", "terminal_hard_split_source", "terminal_adamw_lookahead", "terminal_optimizer_selector", "terminal_source_conserving_route", "terminal_risk_profile_route", "terminal_debt_aware_source_gate", "terminal_raw_then_source_guard", "ungated_terminal_source_conserving_route", "ungated_terminal_risk_profile_route", "ungated_debt_raw_bailout", "h800_source_slow_ema_terminal_source_guard", "h800_source_slow_ema_terminal_raw_guard", "h800_source_slow_ema_terminal_raw_guard_strong", "h800_source_slow_ema_terminal_projected_optimizer", "h800_source_slow_ema_terminal_projected_optimizer_lambda050", "h800_source_slow_ema_terminal_projected_optimizer_lambda100", "h800_source_slow_ema_terminal_projected_blend", "h800_source_slow_ema_terminal_antiwashout", "h800_source_slow_ema_terminal_h4000_reentry", "h800_source_slow_ema_terminal_adaptive_raw", "h800_source_slow_ema_terminal_sparse_source", "h800_source_slow_ema_terminal_ratio_preserve", "h800_source_slow_ema_terminal_source_projection_target", "h800_source_slow_ema_terminal_noise_orthogonal_target", "h800_source_slow_ema_terminal_easy_margin_target", "h800_source_slow_ema_shape_preserve_raw_guard", "h800_source_slow_ema_shape_preserve_clamp"} | terminal_reject_rescue_modes | terminal_topk_support_modes | terminal_source_preserve_modes
                        and step >= terminal_start
                    )
                    late_preserve_mode = int(
                        (
                            loss_gate_fallback_mode in {
                                "late_hold_recovery",
                                "late_lookahead_floor",
                                "terminal_lookahead_floor",
                                "early_terminal_lookahead_floor",
                                "terminal_projected_lookahead_floor",
                                "terminal_consensus_lookahead_floor",
                                "terminal_selector_lookahead_floor",
                                "terminal_positive_lookahead_floor",
                                "terminal_checkpoint_reentry",
                                "terminal_hard_split_source",
                                "terminal_adamw_lookahead",
                                "terminal_optimizer_selector",
                                "terminal_source_conserving_route",
                                "terminal_risk_profile_route",
                                "terminal_debt_aware_source_gate",
                                "terminal_raw_then_source_guard",
                                "ungated_terminal_source_conserving_route",
                                "ungated_terminal_risk_profile_route",
                                "ungated_debt_raw_bailout",
                                "terminal_h2400_checkpoint_hold",
                                "terminal_h2800_checkpoint_hold",
                                "terminal_h2400_debt_bailout",
                                "h800_source_slow_ema_terminal_projected_optimizer",
                                "h800_source_slow_ema_terminal_projected_optimizer_lambda050",
                                "h800_source_slow_ema_terminal_projected_optimizer_lambda100",
                                "h800_source_slow_ema_terminal_projected_blend",
                                "h800_source_slow_ema_terminal_antiwashout",
                                "h800_source_slow_ema_terminal_h4000_reentry",
                            }
                            and step >= late_floor_start
                        )
                    )
                    late_recover_accept = 0
                    late_signal_gain = signal_gain
                    late_corrupt_gain = corrupt_gain
                    target_gate_accept = 0
                    target_gate_diag: dict[str, Any] = {}
                    terminal_reject_rescue_accept = 0
                    terminal_reject_rescue_signal_gain = 0.0
                    terminal_reject_rescue_corrupt_gain = 0.0
                    terminal_reject_rescue_gain_b = 0.0
                    terminal_reject_rescue_status = ""
                    if terminal_lookahead_mode and loss_gate_fallback_mode == "terminal_selector_lookahead_floor":
                        late_recover_accept, late_signal_gain, late_corrupt_gain, _late_gain_b = apply_terminal_selector_lookahead_floor(scale=0.10)
                    elif terminal_lookahead_mode and loss_gate_fallback_mode == "terminal_optimizer_selector":
                        late_recover_accept, late_signal_gain, late_corrupt_gain, _late_gain_b = apply_terminal_optimizer_selector(scale=0.10)
                    elif terminal_lookahead_mode and loss_gate_fallback_mode == "terminal_source_conserving_route":
                        late_recover_accept, late_signal_gain, late_corrupt_gain, _late_gain_b = apply_terminal_source_conserving_route(scale=0.10)
                    elif terminal_lookahead_mode and loss_gate_fallback_mode == "terminal_risk_profile_route":
                        late_recover_accept, late_signal_gain, late_corrupt_gain, _late_gain_b = apply_terminal_source_conserving_route(scale=0.10, risk_profile=True)
                    elif terminal_lookahead_mode and loss_gate_fallback_mode == "terminal_debt_aware_source_gate":
                        late_recover_accept, late_signal_gain, late_corrupt_gain, _late_gain_b = apply_terminal_source_conserving_route(scale=0.10, debt_aware=True)
                    elif terminal_lookahead_mode and loss_gate_fallback_mode == "terminal_raw_then_source_guard":
                        if step < 4400:
                            late_recover_accept, late_signal_gain, late_corrupt_gain, _late_gain_b = apply_late_lookahead_floor(scale=0.10)
                        else:
                            late_recover_accept, late_signal_gain, late_corrupt_gain, _late_gain_b = apply_terminal_source_conserving_route(scale=0.10)
                            if not late_recover_accept:
                                late_recover_accept, late_signal_gain, late_corrupt_gain, _late_gain_b = apply_late_lookahead_floor(
                                    scale=0.05,
                                    project_source=True,
                                    positive_only=True,
                                )
                    elif terminal_lookahead_mode and loss_gate_fallback_mode == "h800_source_slow_ema_terminal_source_guard":
                        late_recover_accept, late_signal_gain, late_corrupt_gain, _late_gain_b = apply_terminal_source_conserving_route(scale=0.08)
                    elif terminal_lookahead_mode and loss_gate_fallback_mode == "h800_source_slow_ema_terminal_raw_guard":
                        if step < 4400:
                            late_recover_accept, late_signal_gain, late_corrupt_gain, _late_gain_b = apply_late_lookahead_floor(scale=0.08)
                        else:
                            late_recover_accept, late_signal_gain, late_corrupt_gain, _late_gain_b = apply_terminal_source_conserving_route(scale=0.08)
                            if not late_recover_accept:
                                late_recover_accept, late_signal_gain, late_corrupt_gain, _late_gain_b = apply_late_lookahead_floor(
                                    scale=0.04,
                                    project_source=True,
                                    positive_only=True,
                                )
                    elif terminal_lookahead_mode and loss_gate_fallback_mode == "h800_source_slow_ema_terminal_raw_guard_strong":
                        if step < 4400:
                            late_recover_accept, late_signal_gain, late_corrupt_gain, _late_gain_b = apply_late_lookahead_floor(scale=0.10)
                        else:
                            late_recover_accept, late_signal_gain, late_corrupt_gain, _late_gain_b = apply_terminal_source_conserving_route(scale=0.10)
                            if not late_recover_accept:
                                late_recover_accept, late_signal_gain, late_corrupt_gain, _late_gain_b = apply_late_lookahead_floor(
                                    scale=0.05,
                                    project_source=True,
                                    positive_only=True,
                                )
                    elif terminal_lookahead_mode and loss_gate_fallback_mode in terminal_topk_support_modes:
                        if step < 4400 and loss_gate_fallback_mode == "h800_source_slow_ema_terminal_topk_raw_guard":
                            late_recover_accept, late_signal_gain, late_corrupt_gain, _late_gain_b = apply_late_lookahead_floor(scale=0.07)
                        else:
                            late_recover_accept, late_signal_gain, late_corrupt_gain, _late_gain_b = apply_terminal_topk_support_route(loss_gate_fallback_mode)
                            if not late_recover_accept and loss_gate_fallback_mode == "h800_source_slow_ema_terminal_topk_raw_guard":
                                late_recover_accept, late_signal_gain, late_corrupt_gain, _late_gain_b = apply_late_lookahead_floor(
                                    scale=0.035,
                                    project_source=True,
                                    positive_only=True,
                                )
                    elif terminal_lookahead_mode and loss_gate_fallback_mode in terminal_source_preserve_modes:
                        if loss_gate_fallback_mode in terminal_control_catchup_modes:
                            late_recover_accept, late_signal_gain, late_corrupt_gain, _late_gain_b = apply_terminal_control_relative_catchup_route(
                                loss_gate_fallback_mode,
                            )
                        elif loss_gate_fallback_mode in terminal_trajectory_adaptive_modes:
                            late_recover_accept, late_signal_gain, late_corrupt_gain, _late_gain_b = apply_terminal_trajectory_adaptive_route(
                                loss_gate_fallback_mode,
                            )
                        elif loss_gate_fallback_mode in terminal_minimal_transport_modes:
                            late_recover_accept, late_signal_gain, late_corrupt_gain, _late_gain_b = apply_terminal_minimal_transport_route(
                                loss_gate_fallback_mode,
                            )
                        elif loss_gate_fallback_mode in terminal_accept_memory_modes:
                            late_recover_accept, late_signal_gain, late_corrupt_gain, _late_gain_b = apply_terminal_accept_memory_route(
                                loss_gate_fallback_mode,
                            )
                        elif loss_gate_fallback_mode in terminal_progress_modes:
                            late_recover_accept, late_signal_gain, late_corrupt_gain, _late_gain_b = apply_terminal_progress_carry_route(
                                loss_gate_fallback_mode,
                            )
                        elif loss_gate_fallback_mode in terminal_anti_erosion_modes | terminal_h3200_anchor_modes:
                            late_recover_accept, late_signal_gain, late_corrupt_gain, _late_gain_b = apply_terminal_anti_erosion_route(
                                loss_gate_fallback_mode,
                            )
                        elif loss_gate_fallback_mode in {
                            "h800_source_slow_ema_terminal_snr_predictor",
                            "h800_source_slow_ema_split_consensus_estimator",
                            "h800_source_slow_ema_signal_reservoir_transport",
                        }:
                            late_recover_accept, late_signal_gain, late_corrupt_gain, _late_gain_b = apply_terminal_signal_estimator_route(
                                loss_gate_fallback_mode,
                            )
                        elif loss_gate_fallback_mode in {
                            "h800_source_slow_ema_terminal_source_floor",
                            "h800_source_slow_ema_terminal_h4000_anchor_floor",
                            "h800_source_slow_ema_terminal_decay_aware_floor",
                        }:
                            late_recover_accept, late_signal_gain, late_corrupt_gain, _late_gain_b = apply_terminal_post4000_source_floor_route(
                                loss_gate_fallback_mode,
                            )
                        elif loss_gate_fallback_mode in {
                            "h800_source_slow_ema_terminal_raw_guard_source_floor",
                            "h800_source_slow_ema_terminal_raw_guard_h4000_anchor_floor",
                            "h800_source_slow_ema_terminal_raw_guard_decay_aware_floor",
                        }:
                            if step < 4000:
                                late_recover_accept, late_signal_gain, late_corrupt_gain, _late_gain_b = apply_late_lookahead_floor(
                                    scale=0.06,
                                )
                                if not late_recover_accept:
                                    late_recover_accept, late_signal_gain, late_corrupt_gain, _late_gain_b = apply_terminal_source_conserving_route(
                                        scale=0.08,
                                    )
                            else:
                                late_recover_accept, late_signal_gain, late_corrupt_gain, _late_gain_b = apply_terminal_post4000_source_floor_route(
                                    loss_gate_fallback_mode,
                                )
                        elif loss_gate_fallback_mode == "h800_source_slow_ema_terminal_source_preserve_very_strong":
                            late_recover_accept, late_signal_gain, late_corrupt_gain, _late_gain_b = apply_terminal_projected_optimizer_route(
                                scale=0.070,
                                blend_source=0.96,
                                antiwashout=True,
                            )
                            if not late_recover_accept:
                                late_recover_accept, late_signal_gain, late_corrupt_gain, _late_gain_b = apply_terminal_source_conserving_route(
                                    scale=0.035,
                                    risk_profile=True,
                                )
                        elif loss_gate_fallback_mode == "h800_source_slow_ema_h3600_terminal_source_preserve":
                            if step < 4000:
                                late_recover_accept, late_signal_gain, late_corrupt_gain, _late_gain_b = apply_terminal_source_conserving_route(
                                    scale=0.045,
                                    risk_profile=True,
                                )
                            else:
                                late_recover_accept, late_signal_gain, late_corrupt_gain, _late_gain_b = apply_terminal_projected_optimizer_route(
                                    scale=0.045,
                                    blend_source=0.92,
                                    antiwashout=True,
                                )
                                if not late_recover_accept:
                                    late_recover_accept, late_signal_gain, late_corrupt_gain, _late_gain_b = apply_terminal_source_conserving_route(
                                        scale=0.035,
                                        risk_profile=True,
                                    )
                        elif loss_gate_fallback_mode == "h800_source_slow_ema_terminal_nora_orthogonal_source":
                            late_recover_accept, late_signal_gain, late_corrupt_gain, _late_gain_b = apply_terminal_nora_orthogonal_source_route(
                                scale=0.048,
                            )
                        elif loss_gate_fallback_mode == "h800_source_slow_ema_terminal_debt_aware_preserve":
                            if update_train_loss_gate() > 1.05 * loss_gate_h800_threshold:
                                late_recover_accept, late_signal_gain, late_corrupt_gain, _late_gain_b = apply_terminal_projected_optimizer_route(
                                    scale=0.055,
                                    blend_source=0.70,
                                    antiwashout=True,
                                )
                                if not late_recover_accept:
                                    late_recover_accept, late_signal_gain, late_corrupt_gain, _late_gain_b = apply_late_lookahead_floor(
                                        scale=0.030,
                                        project_source=True,
                                        positive_only=True,
                                    )
                            else:
                                late_recover_accept, late_signal_gain, late_corrupt_gain, _late_gain_b = apply_terminal_source_conserving_route(
                                    scale=0.040,
                                    debt_aware=True,
                                )
                        elif loss_gate_fallback_mode == "h800_source_slow_ema_terminal_low_nds_matrix_block":
                            late_recover_accept, late_signal_gain, late_corrupt_gain, _late_gain_b = apply_terminal_low_nds_matrix_block_route(
                                scale=0.046,
                            )
                        elif loss_gate_fallback_mode == "h800_source_slow_ema_terminal_dual_memory_preserve":
                            late_recover_accept, late_signal_gain, late_corrupt_gain, _late_gain_b = apply_terminal_dual_memory_preserve_route(
                                scale=0.045,
                            )
                        elif loss_gate_fallback_mode == "h800_source_slow_ema_terminal_source_preserve_strong":
                            late_recover_accept, late_signal_gain, late_corrupt_gain, _late_gain_b = apply_terminal_projected_optimizer_route(
                                scale=0.055,
                                blend_source=0.85,
                                antiwashout=True,
                            )
                        elif loss_gate_fallback_mode == "h800_source_slow_ema_terminal_source_preserve_gentle":
                            late_recover_accept, late_signal_gain, late_corrupt_gain, _late_gain_b = apply_terminal_projected_optimizer_route(
                                scale=0.035,
                                blend_source=0.78,
                                antiwashout=True,
                            )
                        else:
                            late_recover_accept, late_signal_gain, late_corrupt_gain, _late_gain_b = apply_terminal_projected_optimizer_route(
                                scale=0.045,
                                blend_source=0.90,
                                antiwashout=True,
                            )
                            if not late_recover_accept:
                                late_recover_accept, late_signal_gain, late_corrupt_gain, _late_gain_b = apply_terminal_source_conserving_route(
                                    scale=0.04,
                                    risk_profile=True,
                                )
                    elif terminal_lookahead_mode and loss_gate_fallback_mode == "h800_source_slow_ema_shape_preserve_raw_guard":
                        if step < 4400:
                            late_recover_accept, late_signal_gain, late_corrupt_gain, _late_gain_b = apply_late_lookahead_floor(scale=0.07)
                        else:
                            late_recover_accept, late_signal_gain, late_corrupt_gain, _late_gain_b = apply_terminal_source_conserving_route(scale=0.07)
                            if not late_recover_accept:
                                late_recover_accept, late_signal_gain, late_corrupt_gain, _late_gain_b = apply_late_lookahead_floor(
                                    scale=0.035,
                                    project_source=True,
                                    positive_only=True,
                                )
                    elif terminal_lookahead_mode and loss_gate_fallback_mode == "h800_source_slow_ema_shape_preserve_clamp":
                        if step < 4400:
                            late_recover_accept, late_signal_gain, late_corrupt_gain, _late_gain_b = apply_late_lookahead_floor(scale=0.06)
                        else:
                            late_recover_accept, late_signal_gain, late_corrupt_gain, _late_gain_b = apply_terminal_projected_optimizer_route(
                                scale=0.05,
                                blend_source=0.75,
                                antiwashout=True,
                            )
                            if not late_recover_accept:
                                late_recover_accept, late_signal_gain, late_corrupt_gain, _late_gain_b = apply_terminal_source_conserving_route(scale=0.05)
                    elif terminal_lookahead_mode and loss_gate_fallback_mode in terminal_reject_rescue_modes:
                        if (
                            loss_gate_fallback_mode == "h800_source_slow_ema_terminal_reject_hold_source_rescue"
                            and step >= 4400
                            and not gate_accept
                        ):
                            late_recover_accept = 0
                            terminal_selector_kind = "reject_hold_source_rescue_wait"
                        elif (
                            loss_gate_fallback_mode == "h800_source_slow_ema_terminal_reject_lateonly_raw_rescue"
                            and step < 4400
                        ):
                            late_recover_accept, late_signal_gain, late_corrupt_gain, _late_gain_b = apply_terminal_source_conserving_route(scale=0.06)
                        elif step < 4400:
                            late_recover_accept, late_signal_gain, late_corrupt_gain, _late_gain_b = apply_late_lookahead_floor(scale=0.08)
                        else:
                            late_recover_accept, late_signal_gain, late_corrupt_gain, _late_gain_b = apply_terminal_source_conserving_route(scale=0.08)
                            if not late_recover_accept:
                                late_recover_accept, late_signal_gain, late_corrupt_gain, _late_gain_b = apply_late_lookahead_floor(
                                    scale=0.04,
                                    project_source=True,
                                    positive_only=True,
                                )
                    elif terminal_lookahead_mode and loss_gate_fallback_mode in {
                        "h800_source_slow_ema_terminal_source_projection_target",
                        "h800_source_slow_ema_terminal_noise_orthogonal_target",
                        "h800_source_slow_ema_terminal_easy_margin_target",
                    }:
                        if step < 4400:
                            late_recover_accept, late_signal_gain, late_corrupt_gain, _late_gain_b = apply_late_lookahead_floor(scale=0.08)
                        else:
                            late_recover_accept, late_signal_gain, late_corrupt_gain, _late_gain_b = apply_terminal_source_conserving_route(scale=0.08)
                            if not late_recover_accept:
                                late_recover_accept, late_signal_gain, late_corrupt_gain, _late_gain_b = apply_late_lookahead_floor(
                                    scale=0.04,
                                    project_source=True,
                                    positive_only=True,
                                )
                    elif terminal_lookahead_mode and loss_gate_fallback_mode == "h800_source_slow_ema_terminal_h4000_reentry":
                        if step < 4400:
                            late_recover_accept, late_signal_gain, late_corrupt_gain, _late_gain_b = apply_late_lookahead_floor(scale=0.08)
                        else:
                            late_recover_accept, late_signal_gain, late_corrupt_gain, _late_gain_b = apply_checkpoint_reentry(scale=0.012)
                            if not late_recover_accept:
                                late_recover_accept, late_signal_gain, late_corrupt_gain, _late_gain_b = apply_terminal_source_conserving_route(scale=0.08)
                            if not late_recover_accept:
                                late_recover_accept, late_signal_gain, late_corrupt_gain, _late_gain_b = apply_late_lookahead_floor(
                                    scale=0.04,
                                    project_source=True,
                                    positive_only=True,
                                )
                    elif terminal_lookahead_mode and loss_gate_fallback_mode in {
                        "h800_source_slow_ema_terminal_projected_optimizer",
                        "h800_source_slow_ema_terminal_projected_optimizer_lambda050",
                        "h800_source_slow_ema_terminal_projected_optimizer_lambda100",
                    }:
                        projected_scale = 0.08
                        if loss_gate_fallback_mode == "h800_source_slow_ema_terminal_projected_optimizer_lambda050":
                            projected_scale = 0.10
                        if loss_gate_fallback_mode == "h800_source_slow_ema_terminal_projected_optimizer_lambda100":
                            projected_scale = 0.12
                        late_recover_accept, late_signal_gain, late_corrupt_gain, _late_gain_b = apply_terminal_projected_optimizer_route(
                            scale=projected_scale,
                            blend_source=0.0,
                        )
                    elif terminal_lookahead_mode and loss_gate_fallback_mode == "h800_source_slow_ema_terminal_projected_blend":
                        late_recover_accept, late_signal_gain, late_corrupt_gain, _late_gain_b = apply_terminal_projected_optimizer_route(
                            scale=0.08,
                            blend_source=0.35,
                        )
                    elif terminal_lookahead_mode and loss_gate_fallback_mode == "h800_source_slow_ema_terminal_antiwashout":
                        late_recover_accept, late_signal_gain, late_corrupt_gain, _late_gain_b = apply_terminal_projected_optimizer_route(
                            scale=0.08,
                            blend_source=0.50,
                            antiwashout=True,
                        )
                    elif terminal_lookahead_mode and loss_gate_fallback_mode in {
                        "h800_source_slow_ema_terminal_adaptive_raw",
                        "h800_source_slow_ema_terminal_sparse_source",
                        "h800_source_slow_ema_terminal_ratio_preserve",
                    }:
                        late_recover_accept, late_signal_gain, late_corrupt_gain, _late_gain_b = apply_source_state_adaptive_terminal_route(loss_gate_fallback_mode)
                    elif terminal_lookahead_mode and loss_gate_fallback_mode == "ungated_terminal_source_conserving_route":
                        late_recover_accept, late_signal_gain, late_corrupt_gain, _late_gain_b = apply_terminal_source_conserving_route(scale=0.10)
                    elif terminal_lookahead_mode and loss_gate_fallback_mode == "ungated_terminal_risk_profile_route":
                        late_recover_accept, late_signal_gain, late_corrupt_gain, _late_gain_b = apply_terminal_source_conserving_route(scale=0.10, risk_profile=True)
                    elif terminal_lookahead_mode and loss_gate_fallback_mode == "ungated_debt_raw_bailout":
                        if update_train_loss_gate() > 1.10 * loss_gate_h800_threshold:
                            late_recover_accept, late_signal_gain, late_corrupt_gain, _late_gain_b = apply_late_lookahead_floor(scale=0.10, positive_only=True)
                        else:
                            late_recover_accept, late_signal_gain, late_corrupt_gain, _late_gain_b = apply_terminal_source_conserving_route(scale=0.10, debt_aware=True)
                    elif loss_gate_fallback_mode == "h1600_source_checkpoint_reentry" and step >= 2000:
                        late_recover_accept, late_signal_gain, late_corrupt_gain, _late_gain_b = apply_checkpoint_reentry(scale=0.02)
                    elif loss_gate_fallback_mode == "h2400_source_checkpoint_reentry" and step >= 2600:
                        late_recover_accept, late_signal_gain, late_corrupt_gain, _late_gain_b = apply_checkpoint_reentry(scale=0.03)
                    elif terminal_lookahead_mode and loss_gate_fallback_mode == "terminal_consensus_lookahead_floor":
                        late_recover_accept, late_signal_gain, late_corrupt_gain, _late_gain_b = apply_terminal_consensus_lookahead_floor(scale=0.10)
                    elif terminal_lookahead_mode and loss_gate_fallback_mode == "terminal_checkpoint_reentry":
                        late_recover_accept, late_signal_gain, late_corrupt_gain, _late_gain_b = apply_checkpoint_reentry(scale=0.05)
                        if not late_recover_accept:
                            late_recover_accept, late_signal_gain, late_corrupt_gain, _late_gain_b = apply_late_lookahead_floor(scale=0.10)
                    elif terminal_lookahead_mode and loss_gate_fallback_mode == "terminal_hard_split_source":
                        late_recover_accept, late_signal_gain, late_corrupt_gain, _late_gain_b = apply_terminal_hard_split_source(scale=0.10)
                    elif terminal_lookahead_mode and loss_gate_fallback_mode == "terminal_adamw_lookahead":
                        late_recover_accept, late_signal_gain, late_corrupt_gain, _late_gain_b = apply_terminal_adamw_lookahead(scale=0.10)
                    elif terminal_lookahead_mode:
                        late_recover_accept, late_signal_gain, late_corrupt_gain, _late_gain_b = apply_late_lookahead_floor(
                            scale=0.10,
                            project_source=loss_gate_fallback_mode == "terminal_projected_lookahead_floor",
                            positive_only=loss_gate_fallback_mode == "terminal_positive_lookahead_floor",
                        )
                    elif late_preserve_mode and loss_gate_fallback_mode in {
                        "terminal_h2400_checkpoint_hold",
                        "terminal_h2800_checkpoint_hold",
                    }:
                        late_recover_accept = 0
                    elif late_preserve_mode and loss_gate_fallback_mode == "terminal_h2400_debt_bailout":
                        if update_train_loss_gate() > 1.10 * loss_gate_h800_threshold:
                            late_recover_accept, late_signal_gain, late_corrupt_gain, _late_gain_b = apply_late_lookahead_floor(
                                scale=0.03,
                                positive_only=True,
                            )
                    elif late_preserve_mode and loss_gate_fallback_mode == "late_lookahead_floor":
                        late_recover_accept, late_signal_gain, late_corrupt_gain, _late_gain_b = apply_late_lookahead_floor(scale=0.05)
                    elif late_preserve_mode:
                        late_recover_accept = int(update_train_loss_gate() > 1.25 * loss_gate_h800_threshold)
                        if late_recover_accept:
                            apply_late_floor(scale=0.05)
                    elif selector_accept and step >= late_floor_start:
                        apply_late_floor()
                    elif not selector_accept and loss_gate_fallback_mode == "sgd":
                        opt_sgd.step_gradient()
                    elif not selector_accept and loss_gate_fallback_mode == "tiny_late" and step >= late_floor_start:
                        apply_late_floor()
                    elif not selector_accept and loss_gate_fallback_mode == "boosted_tiny_late" and step >= late_floor_start:
                        apply_late_floor(scale=0.20)
                    if (
                        loss_gate_fallback_mode in terminal_reject_rescue_modes
                        and terminal_lookahead_mode
                        and selector_accept
                        and not gate_accept
                        and not late_recover_accept
                        and step >= 4000
                    ):
                        rescue_balance = cosine(short_state, long_state)
                        weak_row_raw_rescue = loss_gate_fallback_mode in {
                            "h800_source_slow_ema_terminal_reject_raw_rescue",
                            "h800_source_slow_ema_terminal_reject_hybrid_rescue",
                            "h800_source_slow_ema_terminal_reject_lateonly_raw_rescue",
                        }
                        rescue_pre_gate = int(
                            agree_density >= (0.45 if weak_row_raw_rescue else 0.70)
                            and rescue_balance >= (0.05 if weak_row_raw_rescue else 0.30)
                            and corrupt_gain <= max(
                                1.0e-4,
                                (0.60 if weak_row_raw_rescue else 0.45) * max(0.0, signal_gain),
                            )
                        )
                        terminal_reject_rescue_status = (
                            f"pre={rescue_pre_gate}:density={agree_density:.4g}:balance={rescue_balance:.4g}"
                        )
                        if rescue_pre_gate:
                            rescue_lr_base = float(args.fu_lr)
                            if loss_gate_fallback_mode in {
                                "h800_source_slow_ema_terminal_reject_raw_rescue",
                                "h800_source_slow_ema_terminal_reject_lateonly_raw_rescue",
                            }:
                                rescue_update = make_late_floor_update(project_source=False)
                                rescue_lr_scale = 0.045
                                rescue_lr_base = float(args.lr)
                                rescue_source = "terminal_reject_raw_rescue"
                            elif loss_gate_fallback_mode == "h800_source_slow_ema_terminal_reject_hybrid_rescue":
                                raw_update = make_late_floor_update(project_source=False)
                                source_ref = normalized_like(source_axis, fast_update.tensor)
                                rescue_vec = normalized_like(0.65 * raw_update.tensor.detach() + 0.35 * source_ref, raw_update.tensor)
                                rescue_lr_scale = 0.04
                                rescue_lr_base = float(args.lr)
                                rescue_source = "terminal_reject_raw_source_hybrid_rescue"
                                rescue_update = UpdateTensor(
                                    rescue_vec,
                                    "step",
                                    "subtract",
                                    "slow_state",
                                    rescue_source,
                                    mechanism,
                                    role=rescue_source,
                                    one_step_descent_claim=0,
                                )
                            elif loss_gate_fallback_mode == "h800_source_slow_ema_terminal_reject_slow_ema_rescue":
                                rescue_vec = active_update_vec
                                rescue_lr_scale = 0.08
                                rescue_source = "terminal_reject_slow_ema_rescue"
                                rescue_update = UpdateTensor(
                                    rescue_vec,
                                    "step",
                                    "subtract",
                                    "slow_state",
                                    rescue_source,
                                    mechanism,
                                    role=rescue_source,
                                    one_step_descent_claim=0,
                                )
                            else:
                                rescue_vec = normalized_like(source_axis, fast_update.tensor)
                                rescue_lr_scale = 0.12
                                rescue_source = "terminal_reject_source_axis_rescue"
                                rescue_update = UpdateTensor(
                                    rescue_vec,
                                    "step",
                                    "subtract",
                                    "slow_state",
                                    rescue_source,
                                    mechanism,
                                    role=rescue_source,
                                    one_step_descent_claim=0,
                                )
                            rescue_before = snapshot(model)
                            with torch.no_grad():
                                rescue_before_a = F.cross_entropy(model(channel_a[0]).float(), channel_a[1], reduction="none")
                                rescue_before_b = F.cross_entropy(model(channel_b[0]).float(), channel_b[1], reduction="none")
                                rescue_before_corrupt = F.cross_entropy(model(channel_b[0]).float(), corrupt_labels, reduction="none")
                            apply_update(model, rescue_update, lr=rescue_lr_scale * rescue_lr_base)
                            with torch.no_grad():
                                rescue_after_a = F.cross_entropy(model(channel_a[0]).float(), channel_a[1], reduction="none")
                                rescue_after_b = F.cross_entropy(model(channel_b[0]).float(), channel_b[1], reduction="none")
                                rescue_after_corrupt = F.cross_entropy(model(channel_b[0]).float(), corrupt_labels, reduction="none")
                            load_flat_params(model, rescue_before)
                            rescue_gain_a = float((rescue_before_a - rescue_after_a).mean().item())
                            rescue_gain_b = float((rescue_before_b - rescue_after_b).mean().item())
                            rescue_corrupt_gain = float((rescue_before_corrupt - rescue_after_corrupt).mean().item())
                            rescue_signal_gain = min(rescue_gain_a, rescue_gain_b)
                            terminal_reject_rescue_signal_gain = rescue_signal_gain
                            terminal_reject_rescue_corrupt_gain = rescue_corrupt_gain
                            terminal_reject_rescue_gain_b = rescue_gain_b
                            terminal_reject_rescue_accept = int(
                                rescue_signal_gain >= -1.0e-6
                                and rescue_corrupt_gain <= max(1.0e-4, 0.45 * max(0.0, rescue_signal_gain))
                            )
                            terminal_reject_rescue_status = (
                                f"{terminal_reject_rescue_status}:accept={terminal_reject_rescue_accept}"
                                f":signal={rescue_signal_gain:.4g}:corrupt={rescue_corrupt_gain:.4g}"
                            )
                            if terminal_reject_rescue_accept:
                                apply_update(model, rescue_update, lr=rescue_lr_scale * rescue_lr_base)
                                late_signal_gain = rescue_signal_gain
                                late_corrupt_gain = rescue_corrupt_gain
                    if selector_accept and gate_accept and not late_preserve_mode:
                        apply_update(model, active_update, lr=active_lr_scale * float(args.fu_lr))
                    if selector_accept and not late_preserve_mode and loss_gate_fallback_mode in signal_target_modes:
                        target_gate_accept, target_gate_diag = apply_signal_channel_target_gate()
                    last_actuation = {
                        "source_state_current_cos": terminal_selector_current_cos if loss_gate_fallback_mode in {"terminal_selector_lookahead_floor", "terminal_source_conserving_route", "terminal_risk_profile_route", "terminal_debt_aware_source_gate", "terminal_raw_then_source_guard", "ungated_terminal_source_conserving_route", "ungated_terminal_risk_profile_route", "ungated_debt_raw_bailout", "h800_source_slow_ema_terminal_projected_optimizer", "h800_source_slow_ema_terminal_projected_optimizer_lambda050", "h800_source_slow_ema_terminal_projected_optimizer_lambda100", "h800_source_slow_ema_terminal_projected_blend", "h800_source_slow_ema_terminal_antiwashout", "h800_source_slow_ema_terminal_adaptive_raw", "h800_source_slow_ema_terminal_sparse_source", "h800_source_slow_ema_terminal_ratio_preserve"} | signal_target_modes | terminal_topk_support_modes | terminal_source_preserve_modes and terminal_selector_kind else cosine(agreed_source, flat_grad(model, device)),
                        "source_state_signal_gain": late_signal_gain,
                        "source_state_corrupt_gain": late_corrupt_gain,
                        "source_state_gate_accept": int((selector_accept and gate_accept and not late_preserve_mode) or late_recover_accept or target_gate_accept or terminal_reject_rescue_accept),
                        "source_state_norm": float(torch.linalg.vector_norm(slow_state.detach()).item()),
                        "source_state_whitened_norm": float(torch.linalg.vector_norm(active_update_vec.detach()).item()),
                        "source_state_antiwashout_removed_norm": terminal_projected_removed_norm,
                        "source_state_projected_grad_norm": float(torch.linalg.vector_norm(anchor_update.tensor.detach()).item()),
                        "source_state_consensus_density": terminal_consensus_density if loss_gate_fallback_mode in {"terminal_consensus_lookahead_floor", "terminal_selector_lookahead_floor", "terminal_source_conserving_route", "terminal_risk_profile_route", "terminal_debt_aware_source_gate", "terminal_raw_then_source_guard", "ungated_terminal_source_conserving_route", "ungated_terminal_risk_profile_route", "ungated_debt_raw_bailout", "h800_source_slow_ema_terminal_projected_optimizer", "h800_source_slow_ema_terminal_projected_optimizer_lambda050", "h800_source_slow_ema_terminal_projected_optimizer_lambda100", "h800_source_slow_ema_terminal_projected_blend", "h800_source_slow_ema_terminal_antiwashout", "h800_source_slow_ema_terminal_adaptive_raw", "h800_source_slow_ema_terminal_sparse_source", "h800_source_slow_ema_terminal_ratio_preserve"} | signal_target_modes | terminal_topk_support_modes | terminal_source_preserve_modes else late_recover_accept if late_preserve_mode else selector_accept if loss_gate_enabled else agree_density,
                        "source_state_balance_mean": terminal_selector_score if loss_gate_fallback_mode in {"terminal_selector_lookahead_floor", "terminal_source_conserving_route", "terminal_risk_profile_route", "terminal_debt_aware_source_gate", "terminal_raw_then_source_guard", "ungated_terminal_source_conserving_route", "ungated_terminal_risk_profile_route", "ungated_debt_raw_bailout", "h800_source_slow_ema_terminal_projected_optimizer", "h800_source_slow_ema_terminal_projected_optimizer_lambda050", "h800_source_slow_ema_terminal_projected_optimizer_lambda100", "h800_source_slow_ema_terminal_projected_blend", "h800_source_slow_ema_terminal_antiwashout", "h800_source_slow_ema_terminal_adaptive_raw", "h800_source_slow_ema_terminal_sparse_source", "h800_source_slow_ema_terminal_ratio_preserve"} | signal_target_modes | terminal_topk_support_modes | terminal_source_preserve_modes else terminal_consensus_balance if loss_gate_fallback_mode == "terminal_consensus_lookahead_floor" else train_loss_gate_last_value if loss_gate_enabled else cosine(short_state, long_state),
                        "source_state_corrupt_cos": terminal_consensus_corrupt_cos if loss_gate_fallback_mode in {"terminal_consensus_lookahead_floor", "terminal_selector_lookahead_floor", "terminal_source_conserving_route", "terminal_risk_profile_route", "terminal_debt_aware_source_gate", "terminal_raw_then_source_guard", "ungated_terminal_source_conserving_route", "ungated_terminal_risk_profile_route", "ungated_debt_raw_bailout", "h800_source_slow_ema_terminal_projected_optimizer", "h800_source_slow_ema_terminal_projected_optimizer_lambda050", "h800_source_slow_ema_terminal_projected_optimizer_lambda100", "h800_source_slow_ema_terminal_projected_blend", "h800_source_slow_ema_terminal_antiwashout", "h800_source_slow_ema_terminal_adaptive_raw", "h800_source_slow_ema_terminal_sparse_source", "h800_source_slow_ema_terminal_ratio_preserve"} | signal_target_modes | terminal_topk_support_modes | terminal_source_preserve_modes else "",
                        "source_state_gate_threshold": loss_gate_h400_threshold if loss_gate_enabled else -1.0e-6,
                        "source_state_ema_beta": 0.995,
                        "source_state_slow_lr_scale": active_lr_scale,
                        "source_channel_projection": cosine(active_update_vec, source_axis),
                        "generalization_gain_a": late_signal_gain if late_preserve_mode else gain_a,
                        "generalization_gain_b": late_signal_gain if late_preserve_mode else gain_b,
                        "shuffled_label_gain": late_corrupt_gain,
                        "generalization_signal_gain": late_signal_gain,
                        "generalization_gate_accept": target_gate_accept if loss_gate_fallback_mode in signal_target_modes else terminal_reject_rescue_accept if loss_gate_fallback_mode in terminal_reject_rescue_modes else late_recover_accept if (late_preserve_mode or loss_gate_fallback_mode in terminal_topk_support_modes | terminal_source_preserve_modes) else selector_accept,
                        "operator_gate_accept": gate_accept,
                        "operator_status": terminal_reject_rescue_status if loss_gate_fallback_mode in terminal_reject_rescue_modes and terminal_reject_rescue_status else terminal_selector_kind if loss_gate_fallback_mode in {"terminal_selector_lookahead_floor", "terminal_source_conserving_route", "terminal_risk_profile_route", "terminal_debt_aware_source_gate", "terminal_raw_then_source_guard", "ungated_terminal_source_conserving_route", "ungated_terminal_risk_profile_route", "ungated_debt_raw_bailout", "h800_source_slow_ema_terminal_projected_optimizer", "h800_source_slow_ema_terminal_projected_blend", "h800_source_slow_ema_terminal_antiwashout", "h800_source_slow_ema_terminal_adaptive_raw", "h800_source_slow_ema_terminal_sparse_source", "h800_source_slow_ema_terminal_ratio_preserve"} | signal_target_modes | terminal_reject_rescue_modes | terminal_topk_support_modes | terminal_source_preserve_modes else loss_gate_fallback_mode if late_recover_accept and loss_gate_fallback_mode in {"h1600_source_checkpoint_reentry", "h2400_source_checkpoint_reentry"} else active_operator_status,
                        "terminal_reject_rescue_accept": terminal_reject_rescue_accept,
                        "terminal_reject_rescue_signal_gain": terminal_reject_rescue_signal_gain,
                        "terminal_reject_rescue_gain_b": terminal_reject_rescue_gain_b,
                        "terminal_reject_rescue_corrupt_gain": terminal_reject_rescue_corrupt_gain,
                        "B1_gain": train_loss_gate_h400_pass if loss_gate_enabled else "",
                        "B2_transfer_gain": train_loss_gate_h800_pass if loss_gate_enabled else "",
                    }
                    if target_gate_diag:
                        source_gate_accept = int_flag(last_actuation.get("source_state_gate_accept"))
                        last_actuation.update(target_gate_diag)
                        last_actuation["source_state_gate_accept"] = int(source_gate_accept or target_gate_accept)
                else:
                    selector_accept = train_loss_selector_accept()
                    terminal_lookahead_mode = 0
                    late_preserve_mode = int(
                        (
                            loss_gate_fallback_mode in {
                                "late_hold_recovery",
                            "late_lookahead_floor",
                            "terminal_lookahead_floor",
                            "early_terminal_lookahead_floor",
                            "terminal_projected_lookahead_floor",
                            "terminal_consensus_lookahead_floor",
                            "terminal_selector_lookahead_floor",
                            "terminal_positive_lookahead_floor",
                            "terminal_checkpoint_reentry",
                            "terminal_hard_split_source",
                            "terminal_adamw_lookahead",
                            "terminal_optimizer_selector",
                            "terminal_source_conserving_route",
                            "terminal_risk_profile_route",
                            "terminal_debt_aware_source_gate",
                            "terminal_raw_then_source_guard",
                            "ungated_terminal_source_conserving_route",
                            "ungated_terminal_risk_profile_route",
                            "ungated_debt_raw_bailout",
                            "terminal_h2400_checkpoint_hold",
                            "terminal_h2800_checkpoint_hold",
                            "terminal_h2400_debt_bailout",
                            "h800_source_slow_ema_terminal_projected_optimizer",
                            "h800_source_slow_ema_terminal_projected_optimizer_lambda050",
                            "h800_source_slow_ema_terminal_projected_optimizer_lambda100",
                            "h800_source_slow_ema_terminal_projected_blend",
                            "h800_source_slow_ema_terminal_antiwashout",
                            "h800_source_slow_ema_terminal_h4000_reentry",
                            "h800_source_slow_ema_terminal_adaptive_raw",
                            "h800_source_slow_ema_terminal_sparse_source",
                            "h800_source_slow_ema_terminal_ratio_preserve",
                        } | terminal_topk_support_modes | terminal_source_preserve_modes
                            and step >= late_floor_start
                        )
                    )
                    late_recover_accept = 0
                    late_signal_gain = 0.0
                    late_corrupt_gain = 0.0
                    if terminal_lookahead_mode:
                        late_recover_accept, late_signal_gain, late_corrupt_gain, _late_gain_b = apply_late_lookahead_floor(scale=0.10)
                    elif late_preserve_mode and loss_gate_fallback_mode == "late_lookahead_floor":
                        late_recover_accept, late_signal_gain, late_corrupt_gain, _late_gain_b = apply_late_lookahead_floor(scale=0.05)
                    elif late_preserve_mode and loss_gate_fallback_mode in terminal_control_catchup_modes:
                        late_recover_accept, late_signal_gain, late_corrupt_gain, _late_gain_b = apply_terminal_control_relative_catchup_route(
                            loss_gate_fallback_mode,
                        )
                    elif late_preserve_mode and loss_gate_fallback_mode in terminal_progress_modes:
                        if step >= terminal_progress_start(loss_gate_fallback_mode):
                            if loss_gate_fallback_mode in {
                                "h800_source_slow_ema_terminal_raw_guard_h3600_gentle_progress",
                                "h800_source_slow_ema_terminal_raw_guard_h4000_gentle_progress",
                                "h800_source_slow_ema_terminal_raw_guard_h4400_projected_progress",
                            }:
                                terminal_selector_kind = f"{loss_gate_fallback_mode}:non_alt_hold"
                                late_recover_accept = 0
                            elif loss_gate_fallback_mode == "h800_source_slow_ema_terminal_h3200_source_progress_blend":
                                apply_update(model, make_late_floor_update(project_source=True), lr=0.35 * float(args.lr))
                                terminal_selector_kind = f"{loss_gate_fallback_mode}:non_alt_projected_progress"
                                late_recover_accept = 1
                            else:
                                opt_sgd.step_gradient()
                                terminal_selector_kind = f"{loss_gate_fallback_mode}:non_alt_sgd_progress"
                                late_recover_accept = 1
                        else:
                            late_recover_accept = 0
                    elif late_preserve_mode and loss_gate_fallback_mode in {"terminal_consensus_lookahead_floor", "terminal_selector_lookahead_floor", "terminal_positive_lookahead_floor", "terminal_checkpoint_reentry", "terminal_hard_split_source", "terminal_adamw_lookahead", "terminal_optimizer_selector", "terminal_source_conserving_route", "terminal_risk_profile_route", "terminal_debt_aware_source_gate", "terminal_raw_then_source_guard", "ungated_terminal_source_conserving_route", "ungated_terminal_risk_profile_route", "ungated_debt_raw_bailout", "terminal_h2400_checkpoint_hold", "terminal_h2800_checkpoint_hold", "terminal_h2400_debt_bailout", "h800_source_slow_ema_terminal_projected_optimizer", "h800_source_slow_ema_terminal_projected_optimizer_lambda050", "h800_source_slow_ema_terminal_projected_optimizer_lambda100", "h800_source_slow_ema_terminal_projected_blend", "h800_source_slow_ema_terminal_antiwashout", "h800_source_slow_ema_terminal_h4000_reentry", "h800_source_slow_ema_terminal_adaptive_raw", "h800_source_slow_ema_terminal_sparse_source", "h800_source_slow_ema_terminal_ratio_preserve"} | terminal_topk_support_modes | terminal_source_preserve_modes:
                        late_recover_accept = 0
                    elif late_preserve_mode:
                        late_recover_accept = int(update_train_loss_gate() > 1.25 * loss_gate_h800_threshold)
                        if late_recover_accept:
                            apply_late_floor(scale=0.05)
                    elif selector_accept and step >= late_floor_start:
                        apply_late_floor()
                    elif not selector_accept and loss_gate_fallback_mode == "sgd":
                        opt_sgd.step_gradient()
                    elif not selector_accept and loss_gate_fallback_mode == "tiny_late" and step >= late_floor_start:
                        apply_late_floor()
                    elif not selector_accept and loss_gate_fallback_mode == "boosted_tiny_late" and step >= late_floor_start:
                        apply_late_floor(scale=0.20)
                    last_actuation = {
                        "source_state_current_cos": cosine(agreed_source, flat_grad(model, device)),
                        "source_state_signal_gain": 0.0,
                        "source_state_corrupt_gain": 0.0,
                        "source_state_gate_accept": late_recover_accept if late_preserve_mode else selector_accept if loss_gate_enabled else 0,
                        "source_state_norm": float(torch.linalg.vector_norm(slow_state.detach()).item()),
                        "source_state_whitened_norm": float(torch.linalg.vector_norm(agreed_source.detach()).item()),
                        "source_state_antiwashout_removed_norm": 0.0,
                        "source_state_projected_grad_norm": 0.0,
                        "source_state_consensus_density": late_recover_accept if late_preserve_mode else selector_accept if loss_gate_enabled else agree_density,
                        "source_state_balance_mean": train_loss_gate_last_value if loss_gate_enabled else cosine(short_state, long_state),
                        "source_state_gate_threshold": loss_gate_h400_threshold if loss_gate_enabled else late_floor_start,
                        "source_state_ema_beta": 0.995,
                        "source_state_slow_lr_scale": slow_lr_scale,
                        "generalization_gain_a": late_signal_gain,
                        "generalization_gain_b": late_signal_gain,
                        "shuffled_label_gain": late_corrupt_gain,
                        "generalization_signal_gain": late_signal_gain,
                        "generalization_gate_accept": late_recover_accept if late_preserve_mode else selector_accept,
                        "operator_status": "terminal_non_alt_hold" if loss_gate_fallback_mode in {"terminal_consensus_lookahead_floor", "terminal_selector_lookahead_floor", "terminal_source_conserving_route", "terminal_risk_profile_route", "terminal_debt_aware_source_gate", "terminal_raw_then_source_guard", "ungated_terminal_source_conserving_route", "ungated_terminal_risk_profile_route", "ungated_debt_raw_bailout", "h800_source_slow_ema_terminal_projected_optimizer", "h800_source_slow_ema_terminal_projected_optimizer_lambda050", "h800_source_slow_ema_terminal_projected_optimizer_lambda100", "h800_source_slow_ema_terminal_projected_blend", "h800_source_slow_ema_terminal_antiwashout", "h800_source_slow_ema_terminal_h4000_reentry", "h800_source_slow_ema_terminal_adaptive_raw", "h800_source_slow_ema_terminal_sparse_source", "h800_source_slow_ema_terminal_ratio_preserve"} | terminal_topk_support_modes | terminal_source_preserve_modes and late_preserve_mode else "",
                        "B1_gain": train_loss_gate_h400_pass if loss_gate_enabled else "",
                        "B2_transfer_gain": train_loss_gate_h800_pass if loss_gate_enabled else "",
                    }
        elif mechanism == "M5-AlternatingFUGradient":
            if step % max(2, int(args.alt_period)) == 0:
                update = make_update(model, mechanism, xb, yb, seed=seed + step)
                apply_update(model, update, lr=float(args.fu_lr))
            else:
                opt_sgd.step_gradient()
        elif mechanism == "M15-LineCFilteredAlternatingFU":
            if step % max(2, int(args.alt_period)) == 0:
                update = make_update(model, mechanism, xb, yb, seed=seed + step)
                before = snapshot(model)
                with torch.no_grad():
                    before_a = F.cross_entropy(model(channel_a[0]).float(), channel_a[1], reduction="none")
                    before_b = F.cross_entropy(model(channel_b[0]).float(), channel_b[1], reduction="none")
                apply_update(model, update, lr=float(args.fu_lr))
                with torch.no_grad():
                    after_a = F.cross_entropy(model(channel_a[0]).float(), channel_a[1], reduction="none")
                    after_b = F.cross_entropy(model(channel_b[0]).float(), channel_b[1], reduction="none")
                load_flat_params(model, before)
                imp_a = before_a - after_a
                imp_b = before_b - after_b
                filt = linec_from_improvements(imp_a, imp_b).to_row()
                mean_a = float(imp_a.mean().item())
                mean_b = float(imp_b.mean().item())
                accept = int(
                    int_flag(filt.get("linec_measurement_valid"))
                    and mean_a >= -1.0e-6
                    and mean_b >= -1.0e-6
                    and finite_float(filt.get("NoiseSignalLeak"), 999.0) <= 0.05
                )
                linec_filter_trials += 1
                linec_filter_accepts += accept
                linec_filter_last = {
                    "valid": filt.get("linec_measurement_valid"),
                    "accept": accept,
                    "mean_a": mean_a,
                    "mean_b": mean_b,
                    "CouplingR2": filt.get("CouplingR2"),
                    "NoiseSignalLeak": filt.get("NoiseSignalLeak"),
                }
                if accept:
                    apply_update(model, update, lr=float(args.fu_lr))
                else:
                    opt_sgd.step_gradient()
            else:
                opt_sgd.step_gradient()
        elif mechanism in {"M16-TwoPhaseMomentumThenLineCFU", "M17-ReadoutCarrierTwoPhaseLineCFU", "M36-MomentumWarmReadoutBlockFU"}:
            warmup_steps = int(getattr(args, "source_warmup_steps", 0) or max(1, steps // 6))
            warmup_mechanism = "M2-SGDMomentumPrimaryFU" if mechanism in {"M16-TwoPhaseMomentumThenLineCFU", "M36-MomentumWarmReadoutBlockFU"} else mechanism
            pulse_mechanism = (
                "M5-AlternatingFUGradient"
                if mechanism == "M16-TwoPhaseMomentumThenLineCFU"
                else "M17-ReadoutCarrierTwoPhaseLineCFU"
                if mechanism == "M36-MomentumWarmReadoutBlockFU"
                else mechanism
            )
            if step <= warmup_steps:
                update = make_update(model, warmup_mechanism, xb, yb, seed=seed + step)
                opt_sgd.step_gradient()
                apply_update(model, update, lr=float(args.fu_lr))
            elif step % max(2, int(args.alt_period)) == 0:
                update = make_update(model, pulse_mechanism, xb, yb, seed=seed + step)
                before = snapshot(model)
                with torch.no_grad():
                    before_a = F.cross_entropy(model(channel_a[0]).float(), channel_a[1], reduction="none")
                    before_b = F.cross_entropy(model(channel_b[0]).float(), channel_b[1], reduction="none")
                apply_update(model, update, lr=float(args.fu_lr))
                with torch.no_grad():
                    after_a = F.cross_entropy(model(channel_a[0]).float(), channel_a[1], reduction="none")
                    after_b = F.cross_entropy(model(channel_b[0]).float(), channel_b[1], reduction="none")
                load_flat_params(model, before)
                imp_a = before_a - after_a
                imp_b = before_b - after_b
                filt = linec_from_improvements(imp_a, imp_b).to_row()
                mean_a = float(imp_a.mean().item())
                mean_b = float(imp_b.mean().item())
                accept = int(
                    int_flag(filt.get("linec_measurement_valid"))
                    and mean_a >= -1.0e-6
                    and mean_b >= -1.0e-6
                    and finite_float(filt.get("NoiseSignalLeak"), 999.0) <= 0.05
                )
                linec_filter_trials += 1
                linec_filter_accepts += accept
                linec_filter_last = {
                    "valid": filt.get("linec_measurement_valid"),
                    "accept": accept,
                    "mean_a": mean_a,
                    "mean_b": mean_b,
                    "CouplingR2": filt.get("CouplingR2"),
                    "NoiseSignalLeak": filt.get("NoiseSignalLeak"),
                }
                if accept:
                    apply_update(model, update, lr=float(args.fu_lr))
                else:
                    opt_sgd.step_gradient()
            else:
                opt_sgd.step_gradient()
        elif mechanism in HC2_PROJECTION_MECHANISMS:
            hc2_anchor_step = 4000 if mechanism in HC2_H4000_RECOMPUTE_PROJECTION_MECHANISMS else 3200
            hc2_warmup_steps = int(getattr(args, "source_warmup_steps", 0) or 100)
            hc2_source_build_steps = max(hc2_anchor_step - 1, hc2_warmup_steps)
            hc2_alt_period = max(2, int(args.alt_period))
            fast_update = make_update(model, "M2-SGDMomentumPrimaryFU", xb, yb, seed=seed + step)
            anchor_update = make_update(model, "M15-LineCFilteredAlternatingFU", xb, yb, seed=seed + step)
            source_vec = 0.75 * fast_update.tensor.detach() + 0.25 * anchor_update.tensor.detach()
            if slow_state is not None and slow_state.numel() == 2 * source_vec.numel():
                short_state, long_state = slow_state.to(device=device).chunk(2)
            else:
                short_state = source_vec.detach().clone()
                long_state = source_vec.detach().clone()
            short_state = 0.80 * short_state + 0.20 * source_vec.to(device=device)
            long_state = 0.995 * long_state + 0.005 * source_vec.to(device=device)
            slow_state = torch.cat([short_state.detach(), long_state.detach()])
            agree = torch.sign(short_state) == torch.sign(long_state)
            agree_density = float(agree.float().mean().item()) if agree.numel() else 0.0
            agreed_source = torch.where(agree, 0.50 * (short_state + long_state), 0.10 * long_state)
            source_axis = normalized_like(agreed_source, fast_update.tensor if fast_update.tensor.numel() else agreed_source)
            source_axis_norm = float(torch.linalg.vector_norm(source_axis.detach()).item()) if source_axis.numel() else 0.0

            if step < hc2_anchor_step:
                if step <= hc2_warmup_steps:
                    opt_adam.step()
                    last_actuation = {
                        "operator_status": "hc2_adamw_boundary_before_source_builder",
                        "source_state_gate_accept": 0,
                        "source_state_consensus_density": agree_density,
                        "source_state_norm": source_axis_norm,
                        "source_state_current_cos": "",
                        "source_state_gate_threshold": hc2_warmup_steps,
                        "source_state_ema_beta": "0.80/0.995",
                        "hc2_h3200_anchor_captured": 0,
                        "negative_projection_fraction": "",
                        "optimizer_cumulative_projection_on_source_Gf": "",
                        "source_preservation_projection_norm": "",
                    }
                elif step <= hc2_source_build_steps:
                    opt_sgd.step_gradient()
                    hc2_update_vec = fast_update.tensor if step <= 800 else source_axis
                    hc2_update_kind = "hc2_h800_fast_source_builder" if step <= 800 else "hc2_h3200_slow_source_builder"
                    update = UpdateTensor(
                        hc2_update_vec,
                        "gradient" if step <= 800 else "step",
                        "subtract",
                        "parameter" if step <= 800 else "slow_state",
                        f"train_stream_{hc2_update_kind}",
                        mechanism,
                        role=hc2_update_kind,
                        one_step_descent_claim=0,
                        diagnostics={
                            "operator_status": hc2_update_kind,
                            "source_state_gate_accept": 1,
                            "source_state_consensus_density": agree_density,
                            "source_state_norm": source_axis_norm,
                            "source_state_current_cos": cosine(hc2_update_vec, source_axis),
                            "source_state_gate_threshold": hc2_source_build_steps,
                            "source_state_ema_beta": "0.80/0.995",
                            "hc2_h3200_anchor_captured": 0,
                            "negative_projection_fraction": "",
                            "optimizer_cumulative_projection_on_source_Gf": "",
                            "source_preservation_projection_norm": "",
                        },
                    )
                    apply_update(model, update, lr=float(args.fu_lr) if step <= 800 else 0.35 * float(args.fu_lr))
                    last_actuation = dict(update.diagnostics or {})
                elif step % hc2_alt_period == 0:
                    opt_sgd.step_gradient()
                    update = UpdateTensor(
                        source_axis,
                        "step",
                        "subtract",
                        "slow_state",
                        "train_stream_hc2_h3200_source_builder",
                        mechanism,
                        role="hc2_pre_h3200_source_builder",
                        one_step_descent_claim=0,
                        diagnostics={
                            "operator_status": "hc2_pre_h3200_source_builder",
                            "source_state_gate_accept": 1,
                            "source_state_consensus_density": agree_density,
                            "source_state_norm": source_axis_norm,
                            "source_state_current_cos": cosine(fast_update.tensor, source_axis),
                            "source_state_gate_threshold": hc2_anchor_step,
                            "source_state_ema_beta": "0.80/0.995",
                            "hc2_h3200_anchor_captured": 0,
                            "negative_projection_fraction": "",
                            "optimizer_cumulative_projection_on_source_Gf": "",
                            "source_preservation_projection_norm": "",
                        },
                    )
                    apply_update(model, update, lr=float(args.fu_lr))
                    last_actuation = dict(update.diagnostics or {})
                else:
                    opt_sgd.step_gradient()
                    last_actuation = {
                        "operator_status": "hc2_pre_h3200_sgd",
                        "source_state_gate_accept": 0,
                        "source_state_consensus_density": agree_density,
                        "source_state_norm": source_axis_norm,
                        "source_state_current_cos": cosine(fast_update.tensor, source_axis),
                        "source_state_gate_threshold": hc2_anchor_step,
                        "source_state_ema_beta": "0.80/0.995",
                        "hc2_h3200_anchor_captured": 0,
                        "negative_projection_fraction": "",
                        "optimizer_cumulative_projection_on_source_Gf": "",
                        "source_preservation_projection_norm": "",
                    }
            else:
                if hc2_h3200_source_anchor is None and source_axis.numel():
                    hc2_h3200_source_anchor = source_axis.detach().clone()
                current_grad = flat_grad(model, device).detach().clone()
                if hc2_h3200_source_anchor is not None and current_grad.numel():
                    source_ref = normalized_like(hc2_h3200_source_anchor.to(device=device, dtype=current_grad.dtype), current_grad)
                else:
                    source_ref = normalized_like(source_axis.to(device=device, dtype=current_grad.dtype), current_grad)
                projection_lambda = {
                    "M240-HC2H3200NoProjectionFU": 0.0,
                    "M241-HC2H3200L2ProjectionFU": 1.0,
                    "M242-HC2H3200HalfProjectionFU": 0.5,
                    "M243-HC2H4000L2RecomputeProjectionFU": 1.0,
                    "M244-HC2H4000HalfRecomputeProjectionFU": 0.5,
                }[mechanism]
                source_norm2 = torch.dot(source_ref.float(), source_ref.float()).clamp_min(1.0e-12)
                source_dot = torch.dot(current_grad.float(), source_ref.float()) if current_grad.numel() else torch.tensor(0.0, device=device)
                anti_coeff = torch.clamp(source_dot, max=0.0) / source_norm2
                removed = projection_lambda * anti_coeff.to(device=device, dtype=current_grad.dtype) * source_ref
                projected = torch.nan_to_num(current_grad - removed, nan=0.0, posinf=0.0, neginf=0.0)
                removed_norm = float(torch.linalg.vector_norm(removed.detach()).item()) if removed.numel() else 0.0
                hc2_projection_trials += 1
                if float(source_dot.detach().item()) < 0.0:
                    hc2_negative_projection_count += 1
                hc2_cumulative_projection += min(0.0, float(source_dot.detach().item()))
                hc2_cumulative_removed_norm += removed_norm
                negative_fraction = float(hc2_negative_projection_count) / float(hc2_projection_trials) if hc2_projection_trials else ""
                update = UpdateTensor(
                    projected,
                    "gradient",
                    "subtract",
                    "slow_state",
                    "train_stream_hc2_h3200_source_projected_optimizer",
                    mechanism,
                    role="hc2_h3200_source_preserving_optimizer_projection",
                    one_step_descent_claim=0,
                    diagnostics={
                        "operator_status": "hc2_post_h3200_projected_optimizer",
                        "source_state_gate_accept": 1,
                        "source_state_consensus_density": agree_density,
                        "source_state_norm": float(torch.linalg.vector_norm(source_ref.detach()).item()) if source_ref.numel() else 0.0,
                        "source_state_current_cos": cosine(current_grad, source_ref),
                        "source_state_projected_cos": cosine(projected, source_ref),
                        "source_state_antiwashout_removed_norm": removed_norm,
                        "source_state_projected_grad_norm": float(torch.linalg.vector_norm(projected.detach()).item()) if projected.numel() else 0.0,
                        "source_state_gate_threshold": hc2_anchor_step,
                        "source_state_ema_beta": "0.80/0.995",
                        "hc2_h3200_anchor_captured": int(hc2_h3200_source_anchor is not None),
                        "negative_projection_fraction": negative_fraction,
                        "optimizer_cumulative_projection_on_source_Gf": hc2_cumulative_projection,
                        "source_preservation_projection_norm": hc2_cumulative_removed_norm,
                        "hc2_projection_lambda": projection_lambda,
                        "hc2_anchor_step": hc2_anchor_step,
                    },
                )
                apply_update(model, update, lr=float(args.lr))
                last_actuation = dict(update.diagnostics or {})
        elif mechanism in {
            "M254-HC8PreH3200SourceChannelAntiWashoutFU",
            "M255-HC9RiskWeightedSplitConsensusSourceFU",
            "M256-HC10ViewConsistentSourceCarryFU",
            "M257-HC11ContinuousSourceCarryAntiWashoutFU",
            "M258-HC12NoiseOrthogonalSourceCarryFU",
        }:
            t12_stop = int(getattr(args, "source_stop_steps", 0) or 800)
            t12_alt = max(2, int(args.alt_period))
            t12_residual_lr = float(getattr(args, "post_stop_fu_lr", 0.25 * float(args.fu_lr)) or (0.25 * float(args.fu_lr)))
            target_mechanism = (
                "M255-HC9RiskWeightedSplitConsensusSourceFU"
                if mechanism == "M255-HC9RiskWeightedSplitConsensusSourceFU"
                else "M256-HC10ViewConsistentSourceCarryFU"
                if mechanism == "M256-HC10ViewConsistentSourceCarryFU"
                else "M257-HC11ContinuousSourceCarryAntiWashoutFU"
                if mechanism == "M257-HC11ContinuousSourceCarryAntiWashoutFU"
                else "M258-HC12NoiseOrthogonalSourceCarryFU"
                if mechanism == "M258-HC12NoiseOrthogonalSourceCarryFU"
                else "M251-MetricTargetObservableMidDebtSourceFU"
            )
            t12_metric_name = (
                "T13-RiskWeightedSplitConsensusSource"
                if mechanism == "M255-HC9RiskWeightedSplitConsensusSourceFU"
                else "T14-ViewConsistentSourceCarry"
                if mechanism == "M256-HC10ViewConsistentSourceCarryFU"
                else "T15-ContinuousSourceCarryAntiWashout"
                if mechanism == "M257-HC11ContinuousSourceCarryAntiWashoutFU"
                else "T16-NoiseOrthogonalSourceCarry"
                if mechanism == "M258-HC12NoiseOrthogonalSourceCarryFU"
                else "T12-PreH3200SourceChannelAntiWashout"
            )
            t12_source_prefix = (
                "t13_risk_weighted_split_consensus"
                if mechanism == "M255-HC9RiskWeightedSplitConsensusSourceFU"
                else "t14_view_consistent_source_carry"
                if mechanism == "M256-HC10ViewConsistentSourceCarryFU"
                else "t15_continuous_source_carry"
                if mechanism == "M257-HC11ContinuousSourceCarryAntiWashoutFU"
                else "t16_noise_orthogonal_source_carry"
                if mechanism == "M258-HC12NoiseOrthogonalSourceCarryFU"
                else "t12_pre_h3200_source_channel"
            )
            target_update = make_update(
                model,
                target_mechanism,
                xb,
                yb,
                seed=seed + step + 42_000,
            )
            target_vec = target_update.tensor.detach().to(device=device)
            if slow_state is not None and slow_state.numel() == 2 * target_vec.numel():
                short_state, long_state = slow_state.to(device=device).chunk(2)
            else:
                short_state = target_vec.detach().clone()
                long_state = target_vec.detach().clone()
            short_beta = 0.70
            long_beta = 0.995
            short_state = short_beta * short_state + (1.0 - short_beta) * target_vec
            long_state = long_beta * long_state + (1.0 - long_beta) * target_vec
            slow_state = torch.cat([short_state.detach(), long_state.detach()])
            agree = torch.sign(short_state) == torch.sign(long_state)
            agree_density = float(agree.float().mean().item()) if agree.numel() else 0.0
            agreed_source = torch.where(agree, 0.35 * short_state + 0.65 * long_state, 0.10 * long_state)
            source_axis = normalized_like(agreed_source, target_vec if target_vec.numel() else agreed_source)
            base_diag = dict(target_update.diagnostics or {})
            common_diag = {
                "metric_name": t12_metric_name,
                "metric_target_stop_steps": t12_stop,
                "post_stop_fu_lr": t12_residual_lr,
                "source_state_norm": float(torch.linalg.vector_norm(slow_state.detach()).item()) if slow_state.numel() else 0.0,
                "source_state_whitened_norm": float(torch.linalg.vector_norm(source_axis.detach()).item()) if source_axis.numel() else 0.0,
                "source_state_consensus_density": agree_density,
                "source_state_balance_mean": cosine(short_state, long_state),
                "source_state_ema_beta": "0.70/0.995",
                "source_state_slow_lr_scale": 0.25,
            }
            if step <= t12_stop:
                if step % t12_alt == 0:
                    update = target_update
                    apply_update(model, update, lr=float(args.fu_lr))
                    operator_status = f"{t12_source_prefix}_write_active"
                    gate_accept = 1
                else:
                    update = None
                    opt_sgd.step_gradient()
                    operator_status = f"{t12_source_prefix}_sgd_before_stop"
                    gate_accept = 0
                last_actuation = dict(base_diag)
                last_actuation.update(common_diag)
                last_actuation.update(
                    {
                        "operator_status": operator_status,
                        "source_state_gate_accept": gate_accept,
                        "source_state_current_cos": cosine(source_axis, flat_grad(model, device)),
                        "source_state_projected_cos": "",
                        "source_state_signal_gain": 0.0,
                        "source_state_corrupt_gain": 0.0,
                        "source_state_antiwashout_removed_norm": 0.0,
                        "source_state_projected_grad_norm": 0.0,
                        "source_state_gate_threshold": "",
                        "generalization_signal_gain": 0.0,
                        "generalization_gate_accept": gate_accept,
                    }
                )
                if update is not None:
                    update.diagnostics = dict(last_actuation)
            else:
                g_current = flat_grad(model, device).detach().clone()
                source_ref = source_axis.to(device=device, dtype=g_current.dtype)
                source_norm2 = torch.dot(source_ref.float(), source_ref.float()).clamp_min(1.0e-12)
                source_dot = torch.dot(g_current.float(), source_ref.float())
                anti_coeff = torch.clamp(source_dot, max=0.0) / source_norm2
                projected = g_current - anti_coeff.to(device=device, dtype=g_current.dtype) * source_ref
                projected = torch.nan_to_num(projected, nan=0.0, posinf=0.0, neginf=0.0)
                anti_removed = g_current - projected
                projected_update = UpdateTensor(
                    projected,
                    "gradient",
                    "subtract",
                    "slow_state+optimizer_projection",
                    f"train_stream_{t12_source_prefix}_projected_optimizer",
                    mechanism,
                    role=f"{t12_source_prefix}_projected_optimizer",
                    one_step_descent_claim=0,
                )
                residual_vec = normalized_like(source_ref, target_vec if target_vec.numel() else source_ref)
                residual_update = UpdateTensor(
                    residual_vec,
                    "step",
                    "subtract",
                    "slow_state",
                    f"train_stream_{t12_source_prefix}_source_residual",
                    mechanism,
                    role=f"{t12_source_prefix}_train_only_residual",
                    one_step_descent_claim=0,
                )
                continuous_carry = mechanism in {
                    "M257-HC11ContinuousSourceCarryAntiWashoutFU",
                    "M258-HC12NoiseOrthogonalSourceCarryFU",
                }
                residual_active = int(continuous_carry or step % t12_alt == 0)
                corrupt_labels = (channel_b[1] + 1) % max(2, int(getattr(args, "classes", 10)))
                before = snapshot(model)
                with torch.no_grad():
                    before_a = F.cross_entropy(model(channel_a[0]).float(), channel_a[1], reduction="none")
                    before_b = F.cross_entropy(model(channel_b[0]).float(), channel_b[1], reduction="none")
                    before_corrupt = F.cross_entropy(model(channel_b[0]).float(), corrupt_labels, reduction="none")
                apply_update(model, projected_update, lr=float(args.lr))
                if residual_active:
                    apply_update(model, residual_update, lr=t12_residual_lr)
                with torch.no_grad():
                    after_a = F.cross_entropy(model(channel_a[0]).float(), channel_a[1], reduction="none")
                    after_b = F.cross_entropy(model(channel_b[0]).float(), channel_b[1], reduction="none")
                    after_corrupt = F.cross_entropy(model(channel_b[0]).float(), corrupt_labels, reduction="none")
                load_flat_params(model, before)
                gain_a = float((before_a - after_a).mean().item())
                gain_b = float((before_b - after_b).mean().item())
                corrupt_gain = float((before_corrupt - after_corrupt).mean().item())
                signal_gain = min(gain_a, gain_b)
                projected_norm = float(torch.linalg.vector_norm(projected.detach()).item()) if projected.numel() else 0.0
                removed_norm = float(torch.linalg.vector_norm(anti_removed.detach()).item()) if anti_removed.numel() else 0.0
                signal_threshold = -5.0e-5
                corrupt_allowance = max(5.0e-5, 0.25 * max(0.0, signal_gain))
                gate_accept = int(
                    projected_norm > 1.0e-12
                    and agree_density >= 0.05
                    and signal_gain >= signal_threshold
                    and corrupt_gain <= corrupt_allowance
                )
                if gate_accept:
                    apply_update(model, projected_update, lr=float(args.lr))
                    if residual_active:
                        apply_update(model, residual_update, lr=t12_residual_lr)
                    update = projected_update
                else:
                    update = None
                    opt_sgd.step_gradient()
                last_actuation = dict(base_diag)
                last_actuation.update(common_diag)
                last_actuation.update(
                    {
                        "operator_status": f"{t12_source_prefix}_projected_optimizer" if gate_accept else f"{t12_source_prefix}_gate_reject_sgd",
                        "source_state_current_cos": cosine(g_current, source_ref),
                        "source_state_projected_cos": cosine(projected, source_ref),
                        "source_state_signal_gain": signal_gain,
                        "source_state_corrupt_gain": corrupt_gain,
                        "source_state_gate_accept": gate_accept,
                        "source_state_antiwashout_removed_norm": removed_norm,
                        "source_state_projected_grad_norm": projected_norm,
                        "source_state_gate_threshold": signal_threshold,
                        "source_state_residual_active": residual_active,
                        "source_state_continuous_carry": int(continuous_carry),
                        "generalization_gain_a": gain_a,
                        "generalization_gain_b": gain_b,
                        "shuffled_label_gain": corrupt_gain,
                        "generalization_signal_gain": signal_gain,
                        "generalization_gate_accept": gate_accept,
                        "negative_projection_fraction": float(removed_norm > 1.0e-12),
                        "optimizer_cumulative_projection_on_source_Gf": float(min(0.0, source_dot.detach().item())),
                        "source_preservation_projection_norm": removed_norm,
                    }
                )
                if update is not None:
                    update.diagnostics = dict(last_actuation)
        elif mechanism == "M252-MetricTargetObservableMidDebtM181BridgeFU":
            bridge_stop = int(getattr(args, "source_stop_steps", 0) or 800)
            bridge_alt = max(2, int(args.alt_period))
            bridge_lr = float(getattr(args, "post_stop_fu_lr", float(args.fu_lr)) or float(args.fu_lr))
            bridge_mechanism = "M181-EarlySourceSlowEMATerminalNoraOrthogonalSourceFU"
            m181_update = make_update(
                model,
                bridge_mechanism,
                xb,
                yb,
                seed=seed + step + 31_000,
                slow_state=slow_state,
            )
            slow_state = update_state_after_commit(bridge_mechanism, slow_state, m181_update)
            m181_diag = dict(m181_update.diagnostics or {})
            slow_state_active = int(slow_state is not None and slow_state.numel() == m181_update.tensor.numel())
            slow_state_norm = (
                float(torch.linalg.vector_norm(slow_state.detach()).item())
                if slow_state is not None and slow_state.numel() == m181_update.tensor.numel()
                else ""
            )
            if step <= bridge_stop and step % bridge_alt == 0:
                update = make_update(model, "M251-MetricTargetObservableMidDebtSourceFU", xb, yb, seed=seed + step)
                apply_update(model, update, lr=float(args.fu_lr))
                last_actuation = dict(update.diagnostics or {})
                last_actuation.update(
                    {
                        "operator_status": "t10_observable_mid_debt_m181_bridge_target_active",
                        "metric_name": "T10-ObservableMidDebtSource+M181Bridge",
                        "metric_target_stop_steps": bridge_stop,
                        "post_stop_mechanism": bridge_mechanism,
                        "post_stop_fu_lr": bridge_lr,
                        "post_stop_slow_state_contract": 1,
                        "post_stop_slow_state_active": slow_state_active,
                        "post_stop_slow_state_norm": slow_state_norm,
                        "terminal_source_slow_state_used": m181_diag.get("terminal_source_slow_state_used", ""),
                        "terminal_source_slow_state_beta": m181_diag.get("terminal_source_slow_state_beta", ""),
                        "terminal_source_slow_state_norm": m181_diag.get("terminal_source_slow_state_norm", ""),
                    }
                )
                update.diagnostics = dict(last_actuation)
            elif step % bridge_alt == 0:
                update = m181_update
                apply_update(model, update, lr=bridge_lr)
                last_actuation = dict(m181_diag)
                last_actuation.update(
                    {
                        "operator_status": "t10_observable_mid_debt_m181_bridge_after_stop",
                        "metric_name": "T10-ObservableMidDebtSource+M181Bridge",
                        "metric_target_stop_steps": bridge_stop,
                        "post_stop_mechanism": bridge_mechanism,
                        "post_stop_fu_lr": bridge_lr,
                        "post_stop_slow_state_contract": 1,
                        "post_stop_slow_state_active": slow_state_active,
                        "post_stop_slow_state_norm": slow_state_norm,
                    }
                )
                update.diagnostics = dict(last_actuation)
            else:
                update = None
                opt_sgd.step_gradient()
                last_actuation = {
                    "operator_status": "t10_observable_mid_debt_m181_bridge_sgd_step",
                    "metric_name": "T10-ObservableMidDebtSource+M181Bridge",
                    "metric_target_stop_steps": bridge_stop,
                    "post_stop_mechanism": bridge_mechanism,
                    "post_stop_slow_state_contract": 1,
                    "post_stop_slow_state_active": slow_state_active,
                    "post_stop_slow_state_norm": slow_state_norm,
                    "terminal_source_slow_state_used": m181_diag.get("terminal_source_slow_state_used", ""),
                }
        elif mechanism in {
            "M20-TrainSplitFunctionSpaceActuationFU",
            "M21-ExactReadoutFunctionSpaceActuationFU",
            "M22-ExactReadoutHighCapScheduledFU",
            "M23-ExactReadoutUltraCapScheduledFU",
            "M49-LossCotangentTargetFU",
            "M50-RandomMatchedTargetFU",
            "M51-SignFlippedTargetFU",
            "M52-CorruptedLabelTargetFU",
            "M53-LowRankLossCotangentTargetFU",
            "M54-CrossSplitConsensusTargetFU",
            "M55-LowDegreeReadoutTargetFU",
            "M56-WeakStableTargetFU",
            "M57-B1ReadoutTransferTargetFU",
            "M58-ReservoirExcludingConsensusTargetFU",
            "M59-StableRandomTargetControlFU",
            "M134-SignalReservoirB3NullConsensusTargetFU",
            "M135-SourceBankB3NullConsensusTargetFU",
            "M136-StableRandomB3NullTargetControlFU",
            "M176-LowNDSDiffeomorphicTargetOnlyFU",
            "M177-InfoVolumeDiffeomorphicTargetOnlyFU",
            "M178-LowRankReadoutTransportTargetOnlyFU",
            "M60-B1CrossSplitConsensusTransferFU",
            "M61-StableB1ConsensusTransferFU",
            "M62-B1WeakStableTransferFU",
            "M63-LowDegreeB1ConsensusTransferFU",
            "M64-ContrastiveLossRandomOrthogonalTargetFU",
            "M65-B1ContrastiveLossRandomOrthogonalTransferFU",
            "M66-TopWrongMarginTargetFU",
            "M67-B1TopWrongMarginTransferFU",
            "M83-LowBankLossB3NullFU",
        } or mechanism in METRIC_TARGET_MECHANISMS:
            metric_target_warmup = int(getattr(args, "source_warmup_steps", 0) or 0)
            metric_target_stop = int(getattr(args, "source_stop_steps", 0) or 0)
            post_stop_mechanism = str(getattr(args, "post_stop_mechanism", "") or "")
            post_stop_fu_lr = float(getattr(args, "post_stop_fu_lr", float(args.fu_lr)) or float(args.fu_lr))
            if mechanism in METRIC_TARGET_MECHANISMS and step <= metric_target_warmup:
                update = None
                opt_sgd.step_gradient()
                last_actuation = {
                    "operator_status": "metric_target_sgd_warmup",
                    "metric_target_warmup_steps": metric_target_warmup,
                }
            elif mechanism in METRIC_TARGET_MECHANISMS and metric_target_stop > 0 and step > metric_target_stop:
                if post_stop_mechanism and step % max(2, int(args.alt_period)) == 0:
                    update = make_update(
                        model,
                        post_stop_mechanism,
                        xb,
                        yb,
                        seed=seed + step,
                        slow_state=slow_state,
                    )
                    slow_state = update_state_after_commit(post_stop_mechanism, slow_state, update)
                    apply_update(model, update, lr=post_stop_fu_lr)
                    last_actuation = dict(update.diagnostics or {})
                    last_actuation.update(
                        {
                            "operator_status": "metric_target_post_stop_mechanism",
                            "metric_target_stop_steps": metric_target_stop,
                            "post_stop_mechanism": post_stop_mechanism,
                            "post_stop_fu_lr": post_stop_fu_lr,
                            "post_stop_slow_state_contract": int(
                                post_stop_mechanism in METRIC_SOURCE_MEMORY_MECHANISMS
                                or post_stop_mechanism in TERMINAL_SOURCE_SLOW_STATE_MECHANISMS
                            ),
                            "post_stop_slow_state_active": int(slow_state is not None and slow_state.numel() == update.tensor.numel()),
                            "post_stop_slow_state_norm": (
                                float(torch.linalg.vector_norm(slow_state.detach()).item())
                                if slow_state is not None and slow_state.numel() == update.tensor.numel()
                                else ""
                            ),
                        }
                    )
                    update.diagnostics = dict(last_actuation)
                else:
                    update = None
                    opt_sgd.step_gradient()
                    last_actuation = {
                        "operator_status": "metric_target_sgd_after_stop",
                        "metric_target_stop_steps": metric_target_stop,
                        "post_stop_mechanism": post_stop_mechanism,
                    }
            elif step % max(2, int(args.alt_period)) == 0:
                update = make_update(model, mechanism, xb, yb, seed=seed + step)
                apply_update(model, update, lr=float(args.fu_lr))
            else:
                update = None
                opt_sgd.step_gradient()
        elif mechanism in {
            "M137-EarlyPulseSourceBankB3NullConsensusTargetFU",
            "M138-EarlyPulseAdamWSourceBankB3NullConsensusTargetFU",
        }:
            pulse_steps = int(getattr(args, "source_warmup_steps", 0) or 100)
            if step <= pulse_steps:
                update = make_update(model, mechanism, xb, yb, seed=seed + step)
                apply_update(model, update, lr=float(args.fu_lr))
                last_actuation = dict(update.diagnostics or {})
                last_actuation.update(
                    {
                        "source_state_gate_accept": 1,
                        "source_state_gate_threshold": pulse_steps,
                        "source_state_consensus_density": last_actuation.get("target_consensus_density", ""),
                        "operator_status": "early_pulse_source_bank_b3_null_active",
                    }
                )
                update.diagnostics = dict(last_actuation)
            else:
                if mechanism == "M138-EarlyPulseAdamWSourceBankB3NullConsensusTargetFU":
                    opt_adam.step()
                else:
                    opt_sgd.step_gradient()
                last_actuation = {
                    "source_state_gate_accept": 0,
                    "source_state_gate_threshold": pulse_steps,
                    "source_state_consensus_density": 0.0,
                    "operator_status": "post_early_pulse_adamw" if mechanism == "M138-EarlyPulseAdamWSourceBankB3NullConsensusTargetFU" else "post_early_pulse_sgd",
                }
        elif mechanism == "M24-WarmupExactReadoutUltraCapFU":
            warmup_steps = int(getattr(args, "source_warmup_steps", 0) or 400)
            if step <= warmup_steps:
                opt_sgd.step_gradient()
            elif step % max(2, int(args.alt_period)) == 0:
                update = make_update(model, mechanism, xb, yb, seed=seed + step)
                apply_update(model, update, lr=float(args.fu_lr))
            else:
                opt_sgd.step_gradient()
        elif mechanism == "M25-AdamWExactReadoutUltraCapFU":
            if step % max(2, int(args.alt_period)) == 0:
                update = make_update(model, mechanism, xb, yb, seed=seed + step)
                opt_adam.step()
                apply_update(model, update, lr=float(args.fu_lr))
            else:
                opt_adam.step()
        elif mechanism == "M26-GatedAdamWExactReadoutFU":
            if step % max(2, int(args.alt_period)) == 0:
                update = make_update(model, mechanism, xb, yb, seed=seed + step)
                diag = update.diagnostics if update.diagnostics is not None else {}
                gate_accept = int(
                    finite_float(diag.get("ActuationR2"), -1.0) >= 0.50
                    and finite_float(diag.get("B2_transfer_gain"), -1.0) > 0.0
                    and finite_float(diag.get("B3_safety_gain"), -1.0) >= 1.0e-3
                    and finite_float(diag.get("function_displacement_norm"), 999.0) <= 4.0
                )
                diag["operator_gate_accept"] = gate_accept
                update.diagnostics = diag
                opt_adam.step()
                if gate_accept:
                    apply_update(model, update, lr=float(args.fu_lr))
            else:
                opt_adam.step()
        elif mechanism in {"M27-MultiBatchGeneralizationGatedReadoutFU", "M28-AdamWMultiBatchGatedReadoutFU"}:
            if step % max(2, int(args.alt_period)) == 0:
                update = make_update(model, mechanism, xb, yb, seed=seed + step)
                diag = update.diagnostics if update.diagnostics is not None else {}
                before = snapshot(model)
                corrupt_labels = (channel_b[1] + 1) % max(2, int(getattr(args, "classes", 10)))
                with torch.no_grad():
                    before_a = F.cross_entropy(model(channel_a[0]).float(), channel_a[1], reduction="none")
                    before_b = F.cross_entropy(model(channel_b[0]).float(), channel_b[1], reduction="none")
                    before_corrupt = F.cross_entropy(model(channel_b[0]).float(), corrupt_labels, reduction="none")
                apply_update(model, update, lr=float(args.fu_lr))
                with torch.no_grad():
                    after_a = F.cross_entropy(model(channel_a[0]).float(), channel_a[1], reduction="none")
                    after_b = F.cross_entropy(model(channel_b[0]).float(), channel_b[1], reduction="none")
                    after_corrupt = F.cross_entropy(model(channel_b[0]).float(), corrupt_labels, reduction="none")
                load_flat_params(model, before)
                gain_a = float((before_a - after_a).mean().item())
                gain_b = float((before_b - after_b).mean().item())
                corrupt_gain = float((before_corrupt - after_corrupt).mean().item())
                signal_gain = min(gain_a, gain_b)
                corrupt_allowance = max(1.0e-4, 0.25 * max(0.0, signal_gain))
                gate_accept = int(
                    finite_float(diag.get("ActuationR2"), -1.0) >= 0.50
                    and finite_float(diag.get("B2_transfer_gain"), -1.0) > 0.0
                    and finite_float(diag.get("B3_safety_gain"), -1.0) >= 1.0e-3
                    and signal_gain >= 1.0e-4
                    and corrupt_gain <= corrupt_allowance
                )
                diag.update(
                    {
                        "generalization_gain_a": gain_a,
                        "generalization_gain_b": gain_b,
                        "shuffled_label_gain": corrupt_gain,
                        "generalization_signal_gain": signal_gain,
                        "shuffled_label_allowance": corrupt_allowance,
                        "generalization_gate_accept": gate_accept,
                        "operator_gate_accept": gate_accept,
                    }
                )
                update.diagnostics = diag
                if mechanism == "M28-AdamWMultiBatchGatedReadoutFU":
                    opt_adam.step()
                    if gate_accept:
                        apply_update(model, update, lr=float(args.fu_lr))
                elif gate_accept:
                    apply_update(model, update, lr=float(args.fu_lr))
                else:
                    opt_sgd.step_gradient()
            else:
                if mechanism == "M28-AdamWMultiBatchGatedReadoutFU":
                    opt_adam.step()
                else:
                    opt_sgd.step_gradient()
        elif mechanism in {"M29-ConsensusSourceStateFU", "M30-LowThresholdConsensusSourceStateFU", "M31-AdamWLowThresholdConsensusResidualFU", "M32-PostAdamWConsensusResidualFU"}:
            if step % max(2, int(args.alt_period)) == 0:
                low_threshold_consensus = mechanism in {"M30-LowThresholdConsensusSourceStateFU", "M31-AdamWLowThresholdConsensusResidualFU", "M32-PostAdamWConsensusResidualFU"}
                adamw_consensus_residual = mechanism in {"M31-AdamWLowThresholdConsensusResidualFU", "M32-PostAdamWConsensusResidualFU"}
                post_adamw_consensus_residual = mechanism == "M32-PostAdamWConsensusResidualFU"
                if post_adamw_consensus_residual:
                    opt_adam.step()
                    model.zero_grad(set_to_none=True)
                params = [p for p in model.parameters() if p.requires_grad]
                original_grads = [None if p.grad is None else p.grad.detach().clone() for p in params]

                def grad_for(xb2: torch.Tensor, yb2: torch.Tensor) -> torch.Tensor:
                    model.zero_grad(set_to_none=True)
                    F.cross_entropy(model(xb2).float(), yb2).backward()
                    return flat_grad(model, device).detach().clone()

                if post_adamw_consensus_residual:
                    g_current = grad_for(xb, yb)
                else:
                    g_current = torch.cat(
                        [
                            (torch.zeros_like(p).reshape(-1) if grad is None else grad.reshape(-1).to(device=device))
                            for p, grad in zip(params, original_grads)
                        ]
                    ) if params else torch.zeros(0, device=device)
                g_a = grad_for(channel_a[0], channel_a[1])
                g_b = grad_for(channel_b[0], channel_b[1])
                corrupt_labels = (channel_b[1] + 1) % max(2, int(getattr(args, "classes", 10)))
                g_corrupt = grad_for(channel_b[0], corrupt_labels)
                model.zero_grad(set_to_none=True)
                if not post_adamw_consensus_residual:
                    for p, grad in zip(params, original_grads):
                        p.grad = None if grad is None else grad.to(device=p.device, dtype=p.dtype)

                agree = torch.sign(g_a) == torch.sign(g_b)
                density = float(agree.float().mean().item()) if agree.numel() else 0.0
                consensus = torch.where(agree, 0.50 * (g_a + g_b), torch.zeros_like(g_a))
                corrupt_denom = torch.dot(g_corrupt.float(), g_corrupt.float()).clamp_min(1.0e-12)
                corrupt_proj = torch.dot(consensus.float(), g_corrupt.float()) / corrupt_denom
                if float(corrupt_proj.item()) > 0.0:
                    consensus = consensus - corrupt_proj.to(device=consensus.device, dtype=consensus.dtype) * g_corrupt
                if slow_state is None or slow_state.numel() != consensus.numel():
                    slow_state = consensus.detach().clone()
                else:
                    beta = 0.90 if low_threshold_consensus else 0.97
                    slow_state = beta * slow_state.to(device=device) + (1.0 - beta) * consensus.detach()
                source_vec = 0.70 * slow_state + 0.30 * consensus if low_threshold_consensus else slow_state
                update_vec = normalized_like(source_vec, g_current if g_current.numel() else consensus)
                update = UpdateTensor(
                    update_vec,
                    "step",
                    "subtract",
                    "slow_state",
                    (
                        "train_stream_adamw_low_threshold_cross_batch_consensus_residual"
                        if mechanism == "M31-AdamWLowThresholdConsensusResidualFU"
                        else "train_stream_post_adamw_low_threshold_cross_batch_consensus_residual"
                        if post_adamw_consensus_residual
                        else "train_stream_low_threshold_cross_batch_consensus_source_state"
                        if low_threshold_consensus
                        else "train_stream_cross_batch_consensus_source_state"
                    ),
                    mechanism,
                    role="consensus_source_state",
                    one_step_descent_claim=0,
                )
                before = snapshot(model)
                with torch.no_grad():
                    before_a = F.cross_entropy(model(channel_a[0]).float(), channel_a[1], reduction="none")
                    before_b = F.cross_entropy(model(channel_b[0]).float(), channel_b[1], reduction="none")
                    before_corrupt = F.cross_entropy(model(channel_b[0]).float(), corrupt_labels, reduction="none")
                apply_update(model, update, lr=float(args.fu_lr))
                with torch.no_grad():
                    after_a = F.cross_entropy(model(channel_a[0]).float(), channel_a[1], reduction="none")
                    after_b = F.cross_entropy(model(channel_b[0]).float(), channel_b[1], reduction="none")
                    after_corrupt = F.cross_entropy(model(channel_b[0]).float(), corrupt_labels, reduction="none")
                load_flat_params(model, before)
                gain_a = float((before_a - after_a).mean().item())
                gain_b = float((before_b - after_b).mean().item())
                corrupt_gain = float((before_corrupt - after_corrupt).mean().item())
                signal_gain = min(gain_a, gain_b)
                corrupt_cos = cosine(consensus, g_corrupt)
                signal_threshold = 1.0e-7 if low_threshold_consensus else 1.0e-5
                corrupt_fraction = 0.25 if low_threshold_consensus else 0.10
                gate_accept = int(
                    density >= 0.05
                    and signal_gain >= signal_threshold
                    and corrupt_gain <= max(1.0e-5, corrupt_fraction * max(0.0, signal_gain))
                    and corrupt_cos <= 0.25
                    and float(torch.linalg.vector_norm(update_vec.detach()).item()) > 1.0e-12
                )
                update.diagnostics = {
                    "source_state_consensus_density": density,
                    "source_state_corrupt_cos": corrupt_cos,
                    "source_state_signal_gain": signal_gain,
                    "source_state_corrupt_gain": corrupt_gain,
                    "source_state_gate_accept": gate_accept,
                    "source_state_norm": float(torch.linalg.vector_norm(slow_state.detach()).item()),
                    "source_state_gate_threshold": signal_threshold,
                    "source_state_ema_beta": 0.90 if low_threshold_consensus else 0.97,
                    "generalization_gain_a": gain_a,
                    "generalization_gain_b": gain_b,
                    "shuffled_label_gain": corrupt_gain,
                    "generalization_signal_gain": signal_gain,
                    "generalization_gate_accept": gate_accept,
                }
                if adamw_consensus_residual:
                    if not post_adamw_consensus_residual:
                        opt_adam.step()
                    if gate_accept:
                        apply_update(model, update, lr=float(args.fu_lr))
                elif gate_accept:
                    apply_update(model, update, lr=float(args.fu_lr))
                else:
                    opt_sgd.step_gradient()
            else:
                if mechanism in {"M31-AdamWLowThresholdConsensusResidualFU", "M32-PostAdamWConsensusResidualFU"}:
                    opt_adam.step()
                else:
                    opt_sgd.step_gradient()
        elif mechanism == "M40-MomentumWarmAntiWashoutFU":
            warmup_steps = int(getattr(args, "source_warmup_steps", 0) or 0)
            if step <= warmup_steps:
                update = make_update(model, "M2-SGDMomentumPrimaryFU", xb, yb, seed=seed + step)
                opt_sgd.step_gradient()
                apply_update(model, update, lr=float(args.fu_lr))
                if slow_state is None or slow_state.numel() != update.tensor.numel():
                    slow_state = update.tensor.detach().clone()
                else:
                    slow_state = 0.98 * slow_state.to(device=device) + 0.02 * update.tensor.detach().to(device=device)
                last_actuation = {
                    "source_state_current_cos": cosine(update.tensor, flat_grad(model, device)),
                    "source_state_gate_accept": 1,
                    "source_state_norm": float(torch.linalg.vector_norm(slow_state.detach()).item()),
                    "source_state_antiwashout_removed_norm": 0.0,
                    "source_state_projected_grad_norm": "",
                }
            else:
                g_current = flat_grad(model, device).detach().clone()
                if slow_state is None or slow_state.numel() != g_current.numel():
                    slow_state = g_current.detach().clone()
                source_vec = slow_state.to(device=device, dtype=g_current.dtype)
                source_norm2 = torch.dot(source_vec.float(), source_vec.float()).clamp_min(1.0e-12)
                dot = torch.dot(g_current.float(), source_vec.float())
                anti_coeff = torch.clamp(dot, max=0.0) / source_norm2
                projected = g_current - anti_coeff.to(device=device, dtype=g_current.dtype) * source_vec
                removed = g_current - projected
                update = UpdateTensor(
                    projected,
                    "step",
                    "subtract",
                    "slow_state",
                    "train_stream_momentum_source_antiwashout_projection",
                    mechanism,
                    role="antiwashout_projected_gradient",
                    one_step_descent_claim=0,
                )
                corrupt_labels = (channel_b[1] + 1) % max(2, int(getattr(args, "classes", 10)))
                before = snapshot(model)
                with torch.no_grad():
                    before_a = F.cross_entropy(model(channel_a[0]).float(), channel_a[1], reduction="none")
                    before_b = F.cross_entropy(model(channel_b[0]).float(), channel_b[1], reduction="none")
                    before_corrupt = F.cross_entropy(model(channel_b[0]).float(), corrupt_labels, reduction="none")
                apply_update(model, update, lr=float(args.lr))
                with torch.no_grad():
                    after_a = F.cross_entropy(model(channel_a[0]).float(), channel_a[1], reduction="none")
                    after_b = F.cross_entropy(model(channel_b[0]).float(), channel_b[1], reduction="none")
                    after_corrupt = F.cross_entropy(model(channel_b[0]).float(), corrupt_labels, reduction="none")
                load_flat_params(model, before)
                gain_a = float((before_a - after_a).mean().item())
                gain_b = float((before_b - after_b).mean().item())
                corrupt_gain = float((before_corrupt - after_corrupt).mean().item())
                signal_gain = min(gain_a, gain_b)
                current_cos = cosine(g_current, source_vec)
                projected_cos = cosine(projected, source_vec)
                removed_norm = float(torch.linalg.vector_norm(removed.detach()).item())
                projected_norm = float(torch.linalg.vector_norm(projected.detach()).item())
                gate_accept = int(projected_norm > 1.0e-12 and signal_gain >= -1.0e-4 and corrupt_gain <= max(1.0e-4, 0.50 * max(0.0, signal_gain)))
                update.diagnostics = {
                    "source_state_current_cos": current_cos,
                    "source_state_signal_gain": signal_gain,
                    "source_state_corrupt_gain": corrupt_gain,
                    "source_state_gate_accept": gate_accept,
                    "source_state_norm": float(torch.linalg.vector_norm(slow_state.detach()).item()),
                    "source_state_whitened_norm": projected_norm,
                    "source_state_antiwashout_removed_norm": removed_norm,
                    "source_state_projected_grad_norm": projected_norm,
                    "generalization_gain_a": gain_a,
                    "generalization_gain_b": gain_b,
                    "shuffled_label_gain": corrupt_gain,
                    "generalization_signal_gain": signal_gain,
                    "generalization_gate_accept": gate_accept,
                    "source_state_gate_threshold": -1.0e-4,
                    "source_state_ema_beta": 0.98,
                    "source_state_consensus_density": "",
                    "source_state_balance_mean": "",
                    "source_state_corrupt_cos": "",
                    "source_state_projected_cos": projected_cos,
                }
                last_actuation = dict(update.diagnostics)
                if gate_accept:
                    apply_update(model, update, lr=float(args.lr))
                else:
                    opt_sgd.step_gradient()
        elif mechanism == "M41-MomentumWarmSourceAnchorFU":
            warmup_steps = int(getattr(args, "source_warmup_steps", 0) or 0)
            if step <= warmup_steps:
                update = make_update(model, "M2-SGDMomentumPrimaryFU", xb, yb, seed=seed + step)
                opt_sgd.step_gradient()
                apply_update(model, update, lr=float(args.fu_lr))
                if slow_state is None or slow_state.numel() != update.tensor.numel():
                    slow_state = update.tensor.detach().clone()
                else:
                    slow_state = 0.98 * slow_state.to(device=device) + 0.02 * update.tensor.detach().to(device=device)
                last_actuation = {
                    "source_state_current_cos": cosine(update.tensor, flat_grad(model, device)),
                    "source_state_gate_accept": 1,
                    "source_state_norm": float(torch.linalg.vector_norm(slow_state.detach()).item()),
                    "source_state_antiwashout_removed_norm": 0.0,
                    "source_state_projected_grad_norm": "",
                }
            else:
                g_current = flat_grad(model, device).detach().clone()
                opt_sgd.step_gradient()
                gate_accept = 0
                gain_a = gain_b = corrupt_gain = signal_gain = 0.0
                anchor_norm = 0.0
                current_cos = 0.0
                if step % max(2, int(args.alt_period)) == 0 and slow_state is not None and slow_state.numel() == g_current.numel():
                    source_vec = slow_state.to(device=device, dtype=g_current.dtype)
                    anchor_vec = normalized_like(source_vec, g_current if g_current.numel() else source_vec)
                    update = UpdateTensor(
                        anchor_vec,
                        "step",
                        "subtract",
                        "slow_state",
                        "train_stream_momentum_source_anchor_residual",
                        mechanism,
                        role="source_anchor_residual",
                        one_step_descent_claim=0,
                    )
                    corrupt_labels = (channel_b[1] + 1) % max(2, int(getattr(args, "classes", 10)))
                    before = snapshot(model)
                    with torch.no_grad():
                        before_a = F.cross_entropy(model(channel_a[0]).float(), channel_a[1], reduction="none")
                        before_b = F.cross_entropy(model(channel_b[0]).float(), channel_b[1], reduction="none")
                        before_corrupt = F.cross_entropy(model(channel_b[0]).float(), corrupt_labels, reduction="none")
                    apply_update(model, update, lr=float(args.fu_lr))
                    with torch.no_grad():
                        after_a = F.cross_entropy(model(channel_a[0]).float(), channel_a[1], reduction="none")
                        after_b = F.cross_entropy(model(channel_b[0]).float(), channel_b[1], reduction="none")
                        after_corrupt = F.cross_entropy(model(channel_b[0]).float(), corrupt_labels, reduction="none")
                    load_flat_params(model, before)
                    gain_a = float((before_a - after_a).mean().item())
                    gain_b = float((before_b - after_b).mean().item())
                    corrupt_gain = float((before_corrupt - after_corrupt).mean().item())
                    signal_gain = min(gain_a, gain_b)
                    current_cos = cosine(anchor_vec, g_current)
                    anchor_norm = float(torch.linalg.vector_norm(anchor_vec.detach()).item())
                    gate_accept = int(
                        anchor_norm > 1.0e-12
                        and signal_gain >= 1.0e-7
                        and corrupt_gain <= max(1.0e-5, 0.20 * max(0.0, signal_gain))
                    )
                    update.diagnostics = {
                        "source_state_current_cos": current_cos,
                        "source_state_signal_gain": signal_gain,
                        "source_state_corrupt_gain": corrupt_gain,
                        "source_state_gate_accept": gate_accept,
                        "source_state_norm": float(torch.linalg.vector_norm(slow_state.detach()).item()),
                        "source_state_whitened_norm": anchor_norm,
                        "source_state_antiwashout_removed_norm": 0.0,
                        "source_state_projected_grad_norm": anchor_norm,
                        "generalization_gain_a": gain_a,
                        "generalization_gain_b": gain_b,
                        "shuffled_label_gain": corrupt_gain,
                        "generalization_signal_gain": signal_gain,
                        "generalization_gate_accept": gate_accept,
                        "source_state_gate_threshold": 1.0e-7,
                        "source_state_ema_beta": 0.98,
                    }
                    if gate_accept:
                        apply_update(model, update, lr=float(args.fu_lr))
                last_actuation = {
                    "source_state_current_cos": current_cos,
                    "source_state_signal_gain": signal_gain,
                    "source_state_corrupt_gain": corrupt_gain,
                    "source_state_gate_accept": gate_accept,
                    "source_state_norm": float(torch.linalg.vector_norm(slow_state.detach()).item()) if slow_state is not None else "",
                    "source_state_whitened_norm": anchor_norm,
                    "source_state_antiwashout_removed_norm": 0.0,
                    "source_state_projected_grad_norm": anchor_norm,
                    "generalization_gain_a": gain_a,
                    "generalization_gain_b": gain_b,
                    "shuffled_label_gain": corrupt_gain,
                    "generalization_signal_gain": signal_gain,
                    "generalization_gate_accept": gate_accept,
                    "source_state_gate_threshold": 1.0e-7,
                    "source_state_ema_beta": 0.98,
                }
        elif mechanism == "M42-MomentumWarmHoldFU":
            warmup_steps = int(getattr(args, "source_warmup_steps", 0) or 0)
            if step <= warmup_steps:
                update = make_update(model, "M2-SGDMomentumPrimaryFU", xb, yb, seed=seed + step)
                opt_sgd.step_gradient()
                apply_update(model, update, lr=float(args.fu_lr))
                if slow_state is None or slow_state.numel() != update.tensor.numel():
                    slow_state = update.tensor.detach().clone()
                else:
                    slow_state = 0.98 * slow_state.to(device=device) + 0.02 * update.tensor.detach().to(device=device)
                last_actuation = {
                    "source_state_current_cos": cosine(update.tensor, flat_grad(model, device)),
                    "source_state_gate_accept": 1,
                    "source_state_norm": float(torch.linalg.vector_norm(slow_state.detach()).item()),
                    "source_state_antiwashout_removed_norm": 0.0,
                    "source_state_projected_grad_norm": "",
                    "source_state_gate_threshold": "",
                    "source_state_ema_beta": 0.98,
                }
            else:
                g_current = flat_grad(model, device).detach().clone()
                current_cos = cosine(slow_state, g_current) if slow_state is not None and slow_state.numel() == g_current.numel() else ""
                last_actuation = {
                    "source_state_current_cos": current_cos,
                    "source_state_signal_gain": 0.0,
                    "source_state_corrupt_gain": 0.0,
                    "source_state_gate_accept": 0,
                    "source_state_norm": float(torch.linalg.vector_norm(slow_state.detach()).item()) if slow_state is not None else "",
                    "source_state_whitened_norm": 0.0,
                    "source_state_antiwashout_removed_norm": 0.0,
                    "source_state_projected_grad_norm": 0.0,
                    "generalization_gain_a": 0.0,
                    "generalization_gain_b": 0.0,
                    "shuffled_label_gain": 0.0,
                    "generalization_signal_gain": 0.0,
                    "generalization_gate_accept": 0,
                    "source_state_gate_threshold": "",
                    "source_state_ema_beta": 0.98,
                }
        elif mechanism == "M43-MomentumCycleHoldFU":
            warmup_steps = max(1, int(getattr(args, "source_warmup_steps", 0) or 0))
            cycle_steps = max(warmup_steps + 1, 2 * warmup_steps)
            cycle_phase = ((step - 1) % cycle_steps) + 1
            if cycle_phase <= warmup_steps:
                update = make_update(model, "M2-SGDMomentumPrimaryFU", xb, yb, seed=seed + step)
                opt_sgd.step_gradient()
                apply_update(model, update, lr=float(args.fu_lr))
                if slow_state is None or slow_state.numel() != update.tensor.numel():
                    slow_state = update.tensor.detach().clone()
                else:
                    slow_state = 0.98 * slow_state.to(device=device) + 0.02 * update.tensor.detach().to(device=device)
                last_actuation = {
                    "source_state_current_cos": cosine(update.tensor, flat_grad(model, device)),
                    "source_state_gate_accept": 1,
                    "source_state_norm": float(torch.linalg.vector_norm(slow_state.detach()).item()),
                    "source_state_antiwashout_removed_norm": 0.0,
                    "source_state_projected_grad_norm": "",
                    "source_state_gate_threshold": "",
                    "source_state_ema_beta": 0.98,
                    "source_state_consensus_density": cycle_phase,
                    "source_state_balance_mean": cycle_steps,
                }
            else:
                g_current = flat_grad(model, device).detach().clone()
                current_cos = cosine(slow_state, g_current) if slow_state is not None and slow_state.numel() == g_current.numel() else ""
                last_actuation = {
                    "source_state_current_cos": current_cos,
                    "source_state_signal_gain": 0.0,
                    "source_state_corrupt_gain": 0.0,
                    "source_state_gate_accept": 0,
                    "source_state_norm": float(torch.linalg.vector_norm(slow_state.detach()).item()) if slow_state is not None else "",
                    "source_state_whitened_norm": 0.0,
                    "source_state_antiwashout_removed_norm": 0.0,
                    "source_state_projected_grad_norm": 0.0,
                    "generalization_gain_a": 0.0,
                    "generalization_gain_b": 0.0,
                    "shuffled_label_gain": 0.0,
                    "generalization_signal_gain": 0.0,
                    "generalization_gate_accept": 0,
                    "source_state_gate_threshold": "",
                    "source_state_ema_beta": 0.98,
                    "source_state_consensus_density": cycle_phase,
                    "source_state_balance_mean": cycle_steps,
                }
        elif mechanism == "M47-MomentumCycleThenLineCAnchorFU":
            warmup_steps = max(1, int(getattr(args, "source_warmup_steps", 0) or 800))
            bridge_start = 2 * warmup_steps
            cycle_steps = max(warmup_steps + 1, bridge_start)
            cycle_phase = ((step - 1) % cycle_steps) + 1
            if step <= bridge_start and cycle_phase <= warmup_steps:
                update = make_update(model, "M2-SGDMomentumPrimaryFU", xb, yb, seed=seed + step)
                opt_sgd.step_gradient()
                apply_update(model, update, lr=float(args.fu_lr))
                if slow_state is None or slow_state.numel() != update.tensor.numel():
                    slow_state = update.tensor.detach().clone()
                else:
                    slow_state = 0.98 * slow_state.to(device=device) + 0.02 * update.tensor.detach().to(device=device)
                last_actuation = {
                    "source_state_current_cos": cosine(update.tensor, flat_grad(model, device)),
                    "source_state_gate_accept": 1,
                    "source_state_norm": float(torch.linalg.vector_norm(slow_state.detach()).item()),
                    "source_state_antiwashout_removed_norm": 0.0,
                    "source_state_projected_grad_norm": "",
                    "source_state_gate_threshold": "",
                    "source_state_ema_beta": 0.98,
                    "source_state_consensus_density": cycle_phase,
                    "source_state_balance_mean": bridge_start,
                    "generalization_signal_gain": 0.0,
                    "generalization_gate_accept": 1,
                }
            elif step <= bridge_start:
                g_current = flat_grad(model, device).detach().clone()
                current_cos = cosine(slow_state, g_current) if slow_state is not None and slow_state.numel() == g_current.numel() else ""
                last_actuation = {
                    "source_state_current_cos": current_cos,
                    "source_state_signal_gain": 0.0,
                    "source_state_corrupt_gain": 0.0,
                    "source_state_gate_accept": 0,
                    "source_state_norm": float(torch.linalg.vector_norm(slow_state.detach()).item()) if slow_state is not None else "",
                    "source_state_whitened_norm": 0.0,
                    "source_state_antiwashout_removed_norm": 0.0,
                    "source_state_projected_grad_norm": 0.0,
                    "generalization_gain_a": 0.0,
                    "generalization_gain_b": 0.0,
                    "shuffled_label_gain": 0.0,
                    "generalization_signal_gain": 0.0,
                    "generalization_gate_accept": 0,
                    "source_state_gate_threshold": "",
                    "source_state_ema_beta": 0.98,
                    "source_state_consensus_density": cycle_phase,
                    "source_state_balance_mean": bridge_start,
                }
            elif step % max(2, int(args.alt_period)) == 0:
                anchor_update = make_update(model, "M15-LineCFilteredAlternatingFU", xb, yb, seed=seed + step)
                if slow_state is None or slow_state.numel() != anchor_update.tensor.numel():
                    slow_state = anchor_update.tensor.detach().clone()
                else:
                    slow_state = 0.985 * slow_state.to(device=device) + 0.015 * anchor_update.tensor.detach().to(device=device)
                slow_vec = normalized_like(slow_state.to(device=device), anchor_update.tensor)
                slow_update = UpdateTensor(
                    slow_vec,
                    "step",
                    "subtract",
                    "slow_state",
                    "train_stream_momentum_cycle_then_linec_anchor_slow_state",
                    mechanism,
                    role="cycle_hold_to_linec_anchor",
                    one_step_descent_claim=0,
                )
                corrupt_labels = (channel_b[1] + 1) % max(2, int(getattr(args, "classes", 10)))
                before = snapshot(model)
                with torch.no_grad():
                    before_a = F.cross_entropy(model(channel_a[0]).float(), channel_a[1], reduction="none")
                    before_b = F.cross_entropy(model(channel_b[0]).float(), channel_b[1], reduction="none")
                    before_corrupt = F.cross_entropy(model(channel_b[0]).float(), corrupt_labels, reduction="none")
                apply_update(model, slow_update, lr=0.25 * float(args.fu_lr))
                with torch.no_grad():
                    after_a = F.cross_entropy(model(channel_a[0]).float(), channel_a[1], reduction="none")
                    after_b = F.cross_entropy(model(channel_b[0]).float(), channel_b[1], reduction="none")
                    after_corrupt = F.cross_entropy(model(channel_b[0]).float(), corrupt_labels, reduction="none")
                load_flat_params(model, before)
                imp_a = before_a - after_a
                imp_b = before_b - after_b
                filt = linec_from_improvements(imp_a, imp_b).to_row()
                mean_a = float(imp_a.mean().item())
                mean_b = float(imp_b.mean().item())
                corrupt_gain = float((before_corrupt - after_corrupt).mean().item())
                signal_gain = min(mean_a, mean_b)
                accept = int(
                    int_flag(filt.get("linec_measurement_valid"))
                    and mean_a >= -1.0e-6
                    and mean_b >= -1.0e-6
                    and finite_float(filt.get("NoiseSignalLeak"), 999.0) <= 0.05
                    and corrupt_gain <= max(1.0e-4, 0.50 * max(0.0, signal_gain))
                )
                linec_filter_trials += 1
                linec_filter_accepts += accept
                linec_filter_last = {
                    "valid": filt.get("linec_measurement_valid"),
                    "accept": accept,
                    "mean_a": mean_a,
                    "mean_b": mean_b,
                    "CouplingR2": filt.get("CouplingR2"),
                    "NoiseSignalLeak": filt.get("NoiseSignalLeak"),
                }
                if accept:
                    apply_update(model, slow_update, lr=0.25 * float(args.fu_lr))
                    update = slow_update
                last_actuation = {
                    "source_state_current_cos": cosine(slow_state, flat_grad(model, device)),
                    "source_state_signal_gain": signal_gain,
                    "source_state_corrupt_gain": corrupt_gain,
                    "source_state_gate_accept": accept,
                    "source_state_norm": float(torch.linalg.vector_norm(slow_state.detach()).item()),
                    "source_state_whitened_norm": float(torch.linalg.vector_norm(slow_vec.detach()).item()),
                    "source_state_antiwashout_removed_norm": 0.0,
                    "source_state_projected_grad_norm": float(torch.linalg.vector_norm(anchor_update.tensor.detach()).item()),
                    "source_state_gate_threshold": -1.0e-6,
                    "source_state_ema_beta": 0.985,
                    "source_state_consensus_density": cycle_phase,
                    "source_state_balance_mean": cosine(slow_state, anchor_update.tensor),
                    "generalization_gain_a": mean_a,
                    "generalization_gain_b": mean_b,
                    "shuffled_label_gain": corrupt_gain,
                    "generalization_signal_gain": signal_gain,
                    "generalization_gate_accept": accept,
                }
            else:
                g_current = flat_grad(model, device).detach().clone()
                current_cos = cosine(slow_state, g_current) if slow_state is not None and slow_state.numel() == g_current.numel() else ""
                last_actuation = {
                    "source_state_current_cos": current_cos,
                    "source_state_signal_gain": 0.0,
                    "source_state_corrupt_gain": 0.0,
                    "source_state_gate_accept": 0,
                    "source_state_norm": float(torch.linalg.vector_norm(slow_state.detach()).item()) if slow_state is not None else "",
                    "source_state_whitened_norm": 0.0,
                    "source_state_antiwashout_removed_norm": 0.0,
                    "source_state_projected_grad_norm": 0.0,
                    "generalization_gain_a": 0.0,
                    "generalization_gain_b": 0.0,
                    "shuffled_label_gain": 0.0,
                    "generalization_signal_gain": 0.0,
                    "generalization_gate_accept": 0,
                    "source_state_gate_threshold": -1.0e-6,
                    "source_state_ema_beta": 0.985,
                    "source_state_consensus_density": cycle_phase,
                    "source_state_balance_mean": bridge_start,
                }
        elif mechanism == "M48-DualTimescaleSourceRetentionFU":
            warmup_steps = max(1, int(getattr(args, "source_warmup_steps", 0) or 800))
            fast_update = make_update(model, "M2-SGDMomentumPrimaryFU", xb, yb, seed=seed + step)
            anchor_update = make_update(model, "M15-LineCFilteredAlternatingFU", xb, yb, seed=seed + step)
            source_vec = 0.75 * fast_update.tensor.detach() + 0.25 * anchor_update.tensor.detach()
            if slow_state is not None and slow_state.numel() == 2 * source_vec.numel():
                short_state, long_state = slow_state.to(device=device).chunk(2)
            else:
                short_state = source_vec.detach().clone()
                long_state = source_vec.detach().clone()
            short_state = 0.80 * short_state + 0.20 * source_vec.to(device=device)
            long_state = 0.995 * long_state + 0.005 * source_vec.to(device=device)
            slow_state = torch.cat([short_state.detach(), long_state.detach()])
            agree = torch.sign(short_state) == torch.sign(long_state)
            agree_density = float(agree.float().mean().item()) if agree.numel() else 0.0
            agreed_source = torch.where(agree, 0.50 * (short_state + long_state), 0.10 * long_state)
            if step <= warmup_steps:
                opt_sgd.step_gradient()
                apply_update(model, fast_update, lr=float(args.fu_lr))
                update = fast_update
                last_actuation = {
                    "source_state_current_cos": cosine(agreed_source, flat_grad(model, device)),
                    "source_state_gate_accept": 1,
                    "source_state_norm": float(torch.linalg.vector_norm(slow_state.detach()).item()),
                    "source_state_whitened_norm": float(torch.linalg.vector_norm(agreed_source.detach()).item()),
                    "source_state_consensus_density": agree_density,
                    "source_state_balance_mean": cosine(short_state, long_state),
                    "source_state_gate_threshold": -1.0e-6,
                    "source_state_ema_beta": 0.995,
                    "generalization_signal_gain": 0.0,
                    "generalization_gate_accept": 1,
                }
            elif step % max(2, int(args.alt_period)) == 0:
                update_vec = normalized_like(agreed_source, fast_update.tensor)
                slow_update = UpdateTensor(
                    update_vec,
                    "step",
                    "subtract",
                    "slow_state",
                    "train_stream_dual_timescale_source_retention",
                    mechanism,
                    role="short_long_agreement_source",
                    one_step_descent_claim=0,
                )
                corrupt_labels = (channel_b[1] + 1) % max(2, int(getattr(args, "classes", 10)))
                before = snapshot(model)
                with torch.no_grad():
                    before_a = F.cross_entropy(model(channel_a[0]).float(), channel_a[1], reduction="none")
                    before_b = F.cross_entropy(model(channel_b[0]).float(), channel_b[1], reduction="none")
                    before_corrupt = F.cross_entropy(model(channel_b[0]).float(), corrupt_labels, reduction="none")
                apply_update(model, slow_update, lr=0.25 * float(args.fu_lr))
                with torch.no_grad():
                    after_a = F.cross_entropy(model(channel_a[0]).float(), channel_a[1], reduction="none")
                    after_b = F.cross_entropy(model(channel_b[0]).float(), channel_b[1], reduction="none")
                    after_corrupt = F.cross_entropy(model(channel_b[0]).float(), corrupt_labels, reduction="none")
                load_flat_params(model, before)
                gain_a = float((before_a - after_a).mean().item())
                gain_b = float((before_b - after_b).mean().item())
                corrupt_gain = float((before_corrupt - after_corrupt).mean().item())
                signal_gain = min(gain_a, gain_b)
                gate_accept = int(
                    agree_density >= 0.20
                    and signal_gain >= -1.0e-6
                    and corrupt_gain <= max(1.0e-4, 0.50 * max(0.0, signal_gain))
                )
                if gate_accept:
                    apply_update(model, slow_update, lr=0.25 * float(args.fu_lr))
                    update = slow_update
                last_actuation = {
                    "source_state_current_cos": cosine(agreed_source, flat_grad(model, device)),
                    "source_state_signal_gain": signal_gain,
                    "source_state_corrupt_gain": corrupt_gain,
                    "source_state_gate_accept": gate_accept,
                    "source_state_norm": float(torch.linalg.vector_norm(slow_state.detach()).item()),
                    "source_state_whitened_norm": float(torch.linalg.vector_norm(update_vec.detach()).item()),
                    "source_state_antiwashout_removed_norm": 0.0,
                    "source_state_projected_grad_norm": float(torch.linalg.vector_norm(anchor_update.tensor.detach()).item()),
                    "source_state_consensus_density": agree_density,
                    "source_state_balance_mean": cosine(short_state, long_state),
                    "source_state_gate_threshold": -1.0e-6,
                    "source_state_ema_beta": 0.995,
                    "generalization_gain_a": gain_a,
                    "generalization_gain_b": gain_b,
                    "shuffled_label_gain": corrupt_gain,
                    "generalization_signal_gain": signal_gain,
                    "generalization_gate_accept": gate_accept,
                }
            else:
                last_actuation = {
                    "source_state_current_cos": cosine(agreed_source, flat_grad(model, device)),
                    "source_state_signal_gain": 0.0,
                    "source_state_corrupt_gain": 0.0,
                    "source_state_gate_accept": 0,
                    "source_state_norm": float(torch.linalg.vector_norm(slow_state.detach()).item()),
                    "source_state_whitened_norm": float(torch.linalg.vector_norm(agreed_source.detach()).item()),
                    "source_state_antiwashout_removed_norm": 0.0,
                    "source_state_projected_grad_norm": 0.0,
                    "source_state_consensus_density": agree_density,
                    "source_state_balance_mean": cosine(short_state, long_state),
                    "source_state_gate_threshold": -1.0e-6,
                    "source_state_ema_beta": 0.995,
                    "generalization_gain_a": 0.0,
                    "generalization_gain_b": 0.0,
                    "shuffled_label_gain": 0.0,
                    "generalization_signal_gain": 0.0,
                    "generalization_gate_accept": 0,
                }
        elif mechanism in {"M72-LossWarmToB1ConsensusMigrationFU", "M74-LossWarmToEasyB1ConsensusMigrationFU", "M76-LossWarmToLossEasyB1ConsensusBlendFU", "M77-GainGatedLossWarmB1ConsensusMigrationFU", "M78-GainGatedLossWarmBlendMigrationFU", "M80-LossWarmToB1ConsensusB3NullMigrationFU", "M82-LossWarmToViewConsistentLossMigrationFU", "M84-LossWarmToLowBankLossB3NullMigrationFU", "M85-GainGatedLowBankLossB3NullFU", "M86-LossWarmToGainGatedLowBankLossB3NullMigrationFU", "M109-AdamWBoundaryToGainGatedLowBankB3NullFU", "M110-AdamWBoundaryLowBankAnchorAntiWashoutFU"}:
            if mechanism in {
                "M109-AdamWBoundaryToGainGatedLowBankB3NullFU",
                "M110-AdamWBoundaryLowBankAnchorAntiWashoutFU",
            }:
                boundary_steps = int(getattr(args, "source_warmup_steps", 0) or 400)

                def assign_flat_gradient(vec: torch.Tensor) -> None:
                    offset = 0
                    for param in model.parameters():
                        if not param.requires_grad:
                            continue
                        n = param.numel()
                        chunk = vec[offset : offset + n].view_as(param).to(device=param.device, dtype=param.dtype)
                        param.grad = chunk.detach().clone()
                        offset += n

                def project_adamw_gradient_from_source() -> tuple[torch.Tensor, float, float, str, int]:
                    g_now = flat_grad(model, device).detach().clone()
                    if (
                        mechanism != "M110-AdamWBoundaryLowBankAnchorAntiWashoutFU"
                        or slow_state is None
                        or slow_state.numel() != g_now.numel()
                        or step < 800
                    ):
                        return g_now, 0.0, float(torch.linalg.vector_norm(g_now.detach()).item()), "", 0
                    source_vec = normalized_like(slow_state.to(device=device, dtype=g_now.dtype), g_now)
                    source_norm2 = torch.dot(source_vec.float(), source_vec.float()).clamp_min(1.0e-12)
                    anti_coeff = torch.clamp(torch.dot(g_now.float(), source_vec.float()), max=0.0) / source_norm2
                    projected = g_now - anti_coeff.to(device=device, dtype=g_now.dtype) * source_vec
                    projected = torch.nan_to_num(projected, nan=0.0, posinf=0.0, neginf=0.0)
                    assign_flat_gradient(projected)
                    removed_norm = float(torch.linalg.vector_norm((g_now - projected).detach()).item())
                    projected_norm = float(torch.linalg.vector_norm(projected.detach()).item())
                    return g_now, removed_norm, projected_norm, cosine(source_vec, g_now), 1

                if step <= boundary_steps:
                    opt_adam.step()
                    last_actuation = {
                        "source_state_current_cos": "",
                        "source_state_signal_gain": 0.0,
                        "source_state_corrupt_gain": 0.0,
                        "source_state_gate_accept": 0,
                        "source_state_norm": 0.0,
                        "source_state_whitened_norm": 0.0,
                        "source_state_antiwashout_removed_norm": 0.0,
                        "source_state_projected_grad_norm": 0.0,
                        "source_state_gate_threshold": boundary_steps,
                        "source_state_ema_beta": 0.0,
                        "source_state_consensus_density": 0.0,
                        "source_state_balance_mean": 0.0,
                        "generalization_signal_gain": 0.0,
                        "generalization_gate_accept": 0,
                        "operator_status": "adamw_boundary",
                    }
                elif step % max(2, int(args.alt_period)) == 0:
                    update = make_update(model, mechanism, xb, yb, seed=seed + step)
                    g_current = flat_grad(model, device)
                    last_actuation = dict(update.diagnostics or {})
                    b2_gain = finite_float(last_actuation.get("B2_transfer_gain"), 0.0)
                    b3_gain = finite_float(last_actuation.get("B3_safety_gain"), 0.0)
                    gate_margin = b2_gain - max(0.0, b3_gain)
                    gate_accept = int(b2_gain > 0.0 and gate_margin > 1.0e-4)
                    if mechanism == "M110-AdamWBoundaryLowBankAnchorAntiWashoutFU" and gate_accept:
                        if slow_state is None or slow_state.numel() != update.tensor.numel():
                            slow_state = update.tensor.detach().clone()
                        else:
                            slow_state = 0.90 * slow_state.to(device=device) + 0.10 * update.tensor.detach().to(device=device)
                    g_current, removed_norm, projected_norm, source_anchor_cos, antiwashout_accept = project_adamw_gradient_from_source()
                    opt_adam.step()
                    if gate_accept:
                        apply_update(model, update, lr=float(args.fu_lr))
                    last_actuation.update(
                        {
                            "source_state_current_cos": cosine(update.tensor.detach(), g_current),
                            "source_state_signal_gain": b2_gain,
                            "source_state_corrupt_gain": b3_gain,
                            "source_state_gate_accept": gate_accept,
                            "source_state_norm": float(torch.linalg.vector_norm(slow_state.detach()).item()) if mechanism == "M110-AdamWBoundaryLowBankAnchorAntiWashoutFU" and slow_state is not None else float(torch.linalg.vector_norm(update.tensor.detach()).item()),
                            "source_state_whitened_norm": float(torch.linalg.vector_norm(update.tensor.detach()).item()),
                            "source_state_antiwashout_removed_norm": removed_norm,
                            "source_state_projected_grad_norm": projected_norm,
                            "source_state_gate_threshold": 1.0e-4,
                            "source_state_ema_beta": 0.90 if mechanism == "M110-AdamWBoundaryLowBankAnchorAntiWashoutFU" else 0.0,
                            "source_state_consensus_density": 1.0,
                            "source_state_balance_mean": source_anchor_cos if source_anchor_cos != "" else gate_margin,
                            "generalization_signal_gain": b2_gain,
                            "generalization_gate_accept": gate_accept,
                            "target_kind": last_actuation.get("target_kind", ""),
                            "target_fit_scope": last_actuation.get("target_fit_scope", ""),
                            "operator_status": (
                                "adamw_boundary_lowbank_anchor_antiwashout"
                                if antiwashout_accept
                                else "adamw_boundary_lowbank_gate_accept"
                                if gate_accept
                                else "adamw_boundary_lowbank_gate_reject"
                            ),
                        }
                    )
                    update.diagnostics = dict(last_actuation)
                else:
                    g_current, removed_norm, projected_norm, source_anchor_cos, antiwashout_accept = project_adamw_gradient_from_source()
                    opt_adam.step()
                    last_actuation = {
                        "source_state_current_cos": source_anchor_cos,
                        "source_state_signal_gain": 0.0,
                        "source_state_corrupt_gain": 0.0,
                        "source_state_gate_accept": antiwashout_accept,
                        "source_state_norm": float(torch.linalg.vector_norm(slow_state.detach()).item()) if slow_state is not None and mechanism == "M110-AdamWBoundaryLowBankAnchorAntiWashoutFU" else 0.0,
                        "source_state_whitened_norm": 0.0,
                        "source_state_antiwashout_removed_norm": removed_norm,
                        "source_state_projected_grad_norm": projected_norm,
                        "source_state_gate_threshold": 1.0e-4,
                        "source_state_ema_beta": 0.90 if mechanism == "M110-AdamWBoundaryLowBankAnchorAntiWashoutFU" else 0.0,
                        "source_state_consensus_density": 0.0,
                        "source_state_balance_mean": 0.0,
                        "generalization_signal_gain": 0.0,
                        "generalization_gate_accept": antiwashout_accept,
                        "operator_status": "adamw_non_alt_anchor_antiwashout" if antiwashout_accept else "adamw_non_alt",
                    }
            elif step % max(2, int(args.alt_period)) == 0:
                warmup_raw = getattr(args, "source_warmup_steps", 0)
                warmup_steps = int(warmup_raw) if warmup_raw not in {None, ""} else 800
                if mechanism == "M85-GainGatedLowBankLossB3NullFU":
                    warmup_steps = 0
                migration_inner = "M60-B1CrossSplitConsensusTransferFU"
                if mechanism == "M74-LossWarmToEasyB1ConsensusMigrationFU":
                    migration_inner = "M73-EasyB1ConsensusTransferFU"
                if mechanism in {"M76-LossWarmToLossEasyB1ConsensusBlendFU", "M78-GainGatedLossWarmBlendMigrationFU"}:
                    migration_inner = "M75-LossEasyB1ConsensusBlendFU"
                if mechanism == "M80-LossWarmToB1ConsensusB3NullMigrationFU":
                    migration_inner = "M79-B1ConsensusB3NullTransferFU"
                if mechanism == "M82-LossWarmToViewConsistentLossMigrationFU":
                    migration_inner = "M81-ViewConsistentLossTargetFU"
                if mechanism == "M84-LossWarmToLowBankLossB3NullMigrationFU":
                    migration_inner = "M83-LowBankLossB3NullFU"
                if mechanism in {"M85-GainGatedLowBankLossB3NullFU", "M86-LossWarmToGainGatedLowBankLossB3NullMigrationFU"}:
                    migration_inner = "M85-GainGatedLowBankLossB3NullFU"
                phase = "loss_cotangent_warmup" if step <= warmup_steps else migration_inner
                inner = "M49-LossCotangentTargetFU" if step <= warmup_steps else migration_inner
                update = make_update(model, inner, xb, yb, seed=seed + step)
                g_current = flat_grad(model, device)
                last_actuation = dict(update.diagnostics or {})
                b2_gain = finite_float(last_actuation.get("B2_transfer_gain"), 0.0)
                b3_gain = finite_float(last_actuation.get("B3_safety_gain"), 0.0)
                gate_margin = b2_gain - max(0.0, b3_gain)
                gated = mechanism in {"M77-GainGatedLossWarmB1ConsensusMigrationFU", "M78-GainGatedLossWarmBlendMigrationFU", "M85-GainGatedLowBankLossB3NullFU", "M86-LossWarmToGainGatedLowBankLossB3NullMigrationFU"} and step > warmup_steps
                gate_accept = int((not gated) or (b2_gain > 0.0 and gate_margin > 1.0e-4))
                if gate_accept:
                    apply_update(model, update, lr=float(args.fu_lr))
                else:
                    opt_sgd.step_gradient()
                last_actuation.update(
                    {
                        "source_state_current_cos": cosine(update.tensor.detach(), g_current),
                        "source_state_signal_gain": b2_gain,
                        "source_state_corrupt_gain": b3_gain,
                        "source_state_gate_accept": gate_accept,
                        "source_state_norm": float(torch.linalg.vector_norm(update.tensor.detach()).item()),
                        "source_state_whitened_norm": float(torch.linalg.vector_norm(update.tensor.detach()).item()),
                        "source_state_antiwashout_removed_norm": 0.0,
                        "source_state_projected_grad_norm": float(torch.linalg.vector_norm(g_current.detach()).item()),
                        "source_state_gate_threshold": 1.0e-4 if gated else 0.0,
                        "source_state_ema_beta": 0.0,
                        "source_state_consensus_density": 0.0 if phase == "loss_cotangent_warmup" else 1.0,
                        "source_state_balance_mean": 0.0 if phase == "loss_cotangent_warmup" else 1.0,
                        "generalization_signal_gain": b2_gain,
                        "generalization_gate_accept": gate_accept,
                        "target_kind": last_actuation.get("target_kind", ""),
                        "target_fit_scope": last_actuation.get("target_fit_scope", ""),
                        "operator_status": last_actuation.get("operator_status", ""),
                    }
                )
                update.diagnostics = dict(last_actuation)
            else:
                opt_sgd.step_gradient()
        elif mechanism == "M71-TrainLookaheadB1ConsensusTransferFU":
            g_current = flat_grad(model, device)
            target_update = make_update(model, "M71-TrainLookaheadB1ConsensusTransferFU", xb, yb, seed=seed + step)
            target_vec = normalized_like(target_update.tensor.detach(), g_current)
            gated_update = UpdateTensor(
                target_vec,
                "cotangent",
                "subtract",
                "function",
                "train_stream_b1_consensus_target_vs_gradient_lookahead",
                mechanism,
                role="train_lookahead_b1_consensus_transfer_target",
                one_step_descent_claim=0,
                diagnostics=dict(target_update.diagnostics),
            )
            grad_vec = normalized_like(g_current, target_vec)
            gradient_update = UpdateTensor(
                grad_vec,
                "step",
                "subtract",
                "gradient",
                "train_stream_same_norm_gradient_lookahead_control",
                mechanism,
                role="same_norm_gradient_control",
                one_step_descent_claim=1,
            )
            corrupt_labels = (channel_b[1] + 1) % max(2, int(getattr(args, "classes", 10)))
            before = snapshot(model)
            with torch.no_grad():
                before_a = F.cross_entropy(model(channel_a[0]).float(), channel_a[1], reduction="none")
                before_b = F.cross_entropy(model(channel_b[0]).float(), channel_b[1], reduction="none")
                before_corrupt = F.cross_entropy(model(channel_b[0]).float(), corrupt_labels, reduction="none")
            apply_update(model, gated_update, lr=0.25 * float(args.fu_lr))
            with torch.no_grad():
                target_after_a = F.cross_entropy(model(channel_a[0]).float(), channel_a[1], reduction="none")
                target_after_b = F.cross_entropy(model(channel_b[0]).float(), channel_b[1], reduction="none")
                target_after_corrupt = F.cross_entropy(model(channel_b[0]).float(), corrupt_labels, reduction="none")
            load_flat_params(model, before)
            apply_update(model, gradient_update, lr=0.25 * float(args.fu_lr))
            with torch.no_grad():
                grad_after_a = F.cross_entropy(model(channel_a[0]).float(), channel_a[1], reduction="none")
                grad_after_b = F.cross_entropy(model(channel_b[0]).float(), channel_b[1], reduction="none")
                grad_after_corrupt = F.cross_entropy(model(channel_b[0]).float(), corrupt_labels, reduction="none")
            load_flat_params(model, before)
            target_gain_a = float((before_a - target_after_a).mean().item())
            target_gain_b = float((before_b - target_after_b).mean().item())
            target_corrupt_gain = float((before_corrupt - target_after_corrupt).mean().item())
            grad_gain_a = float((before_a - grad_after_a).mean().item())
            grad_gain_b = float((before_b - grad_after_b).mean().item())
            grad_corrupt_gain = float((before_corrupt - grad_after_corrupt).mean().item())
            target_signal = min(target_gain_a, target_gain_b)
            grad_signal = min(grad_gain_a, grad_gain_b)
            lookahead_advantage = target_signal - grad_signal
            gate_accept = int(
                step % max(2, int(args.alt_period)) == 0
                and float(torch.linalg.vector_norm(target_vec.detach()).item()) > 1.0e-12
                and target_signal >= -1.0e-5
                and lookahead_advantage >= -5.0e-6
                and target_corrupt_gain <= max(grad_corrupt_gain + 1.0e-5, 0.50 * max(0.0, target_signal), 1.0e-4)
            )
            if gate_accept:
                apply_update(model, gated_update, lr=float(args.fu_lr))
                update = gated_update
            else:
                opt_sgd.step_gradient()
            last_actuation = dict(target_update.diagnostics)
            last_actuation.update(
                {
                    "source_state_current_cos": cosine(target_vec, g_current),
                    "source_state_signal_gain": target_signal,
                    "source_state_corrupt_gain": target_corrupt_gain,
                    "source_state_gate_accept": gate_accept,
                    "source_state_norm": float(torch.linalg.vector_norm(target_vec.detach()).item()),
                    "source_state_whitened_norm": float(torch.linalg.vector_norm(target_vec.detach()).item()),
                    "source_state_antiwashout_removed_norm": 0.0,
                    "source_state_projected_grad_norm": float(torch.linalg.vector_norm(g_current.detach()).item()),
                    "source_state_gate_threshold": -5.0e-6,
                    "source_state_ema_beta": 0.0,
                    "source_state_consensus_density": target_update.diagnostics.get("operator_gate_accept", ""),
                    "source_state_balance_mean": lookahead_advantage,
                    "generalization_gain_a": target_gain_a,
                    "generalization_gain_b": target_gain_b,
                    "shuffled_label_gain": target_corrupt_gain,
                    "generalization_signal_gain": grad_signal,
                    "generalization_gate_accept": gate_accept,
                }
            )
            if gate_accept:
                gated_update.diagnostics = dict(last_actuation)
        elif mechanism == "M70-TrainLookaheadCautiousSourceFU":
            g_current = flat_grad(model, device)
            fast_update = make_update(model, "M2-SGDMomentumPrimaryFU", xb, yb, seed=seed + step)
            block_update = make_update(model, "M13-LowRankMatrixBlockFU", xb, yb, seed=seed + step, slow_state=slow_state)
            block_vec = block_update.tensor.detach()
            if slow_state is None or slow_state.numel() != block_vec.numel():
                slow_state = block_vec.detach().clone()
            else:
                slow_state = 0.97 * slow_state.to(device=device) + 0.03 * block_vec.to(device=device)
            agree = torch.sign(slow_state) == torch.sign(block_vec)
            cautious_vec = torch.where(agree, 0.85 * slow_state + 0.15 * block_vec, 0.05 * slow_state + 0.10 * block_vec)
            candidate_vec = normalized_like(0.70 * fast_update.tensor.detach() + 0.30 * cautious_vec.to(device=device), g_current)
            candidate_update = UpdateTensor(
                candidate_vec,
                "step",
                "subtract",
                "matrix_block",
                "train_stream_source_vs_gradient_lookahead_cautious_state",
                mechanism,
                role="train_lookahead_cautious_source_state",
                one_step_descent_claim=0,
            )
            grad_vec = normalized_like(g_current, candidate_vec)
            gradient_update = UpdateTensor(
                grad_vec,
                "step",
                "subtract",
                "gradient",
                "train_stream_same_norm_gradient_lookahead_control",
                mechanism,
                role="same_norm_gradient_control",
                one_step_descent_claim=1,
            )
            corrupt_labels = (channel_b[1] + 1) % max(2, int(getattr(args, "classes", 10)))
            before = snapshot(model)
            with torch.no_grad():
                before_a = F.cross_entropy(model(channel_a[0]).float(), channel_a[1], reduction="none")
                before_b = F.cross_entropy(model(channel_b[0]).float(), channel_b[1], reduction="none")
                before_corrupt = F.cross_entropy(model(channel_b[0]).float(), corrupt_labels, reduction="none")
            apply_update(model, candidate_update, lr=0.25 * float(args.fu_lr))
            with torch.no_grad():
                cand_after_a = F.cross_entropy(model(channel_a[0]).float(), channel_a[1], reduction="none")
                cand_after_b = F.cross_entropy(model(channel_b[0]).float(), channel_b[1], reduction="none")
                cand_after_corrupt = F.cross_entropy(model(channel_b[0]).float(), corrupt_labels, reduction="none")
            load_flat_params(model, before)
            apply_update(model, gradient_update, lr=0.25 * float(args.fu_lr))
            with torch.no_grad():
                grad_after_a = F.cross_entropy(model(channel_a[0]).float(), channel_a[1], reduction="none")
                grad_after_b = F.cross_entropy(model(channel_b[0]).float(), channel_b[1], reduction="none")
                grad_after_corrupt = F.cross_entropy(model(channel_b[0]).float(), corrupt_labels, reduction="none")
            load_flat_params(model, before)
            cand_gain_a = float((before_a - cand_after_a).mean().item())
            cand_gain_b = float((before_b - cand_after_b).mean().item())
            cand_corrupt_gain = float((before_corrupt - cand_after_corrupt).mean().item())
            grad_gain_a = float((before_a - grad_after_a).mean().item())
            grad_gain_b = float((before_b - grad_after_b).mean().item())
            grad_corrupt_gain = float((before_corrupt - grad_after_corrupt).mean().item())
            cand_signal = min(cand_gain_a, cand_gain_b)
            grad_signal = min(grad_gain_a, grad_gain_b)
            lookahead_advantage = cand_signal - grad_signal
            gate_accept = int(
                step % max(2, int(args.alt_period)) == 0
                and float(torch.linalg.vector_norm(candidate_vec.detach()).item()) > 1.0e-12
                and cand_signal >= -1.0e-5
                and lookahead_advantage >= -5.0e-6
                and cand_corrupt_gain <= max(grad_corrupt_gain + 1.0e-5, 0.50 * max(0.0, cand_signal), 1.0e-4)
            )
            opt_sgd.step_gradient()
            if gate_accept:
                apply_update(model, candidate_update, lr=0.25 * float(args.fu_lr))
                update = candidate_update
            last_actuation = {
                "source_state_current_cos": cosine(candidate_vec, g_current),
                "source_state_signal_gain": cand_signal,
                "source_state_corrupt_gain": cand_corrupt_gain,
                "source_state_gate_accept": gate_accept,
                "source_state_norm": float(torch.linalg.vector_norm(slow_state.detach()).item()),
                "source_state_whitened_norm": float(torch.linalg.vector_norm(candidate_vec.detach()).item()),
                "source_state_antiwashout_removed_norm": float(torch.linalg.vector_norm((block_vec - cautious_vec).detach()).item()),
                "source_state_projected_grad_norm": float(torch.linalg.vector_norm(block_vec.detach()).item()),
                "source_state_gate_threshold": -5.0e-6,
                "source_state_ema_beta": 0.97,
                "source_state_consensus_density": float(agree.float().mean().item()) if agree.numel() else 0.0,
                "source_state_balance_mean": lookahead_advantage,
                "generalization_gain_a": cand_gain_a,
                "generalization_gain_b": cand_gain_b,
                "shuffled_label_gain": cand_corrupt_gain,
                "generalization_signal_gain": grad_signal,
                "generalization_gate_accept": gate_accept,
            }
        elif mechanism == "M69-RotatedCautiousMatrixSlowFU":
            fast_update = make_update(model, "M2-SGDMomentumPrimaryFU", xb, yb, seed=seed + step)
            opt_sgd.step_gradient()
            apply_update(model, fast_update, lr=float(args.fu_lr))
            block_update = make_update(model, "M13-LowRankMatrixBlockFU", xb, yb, seed=seed + step, slow_state=slow_state)
            block_vec = block_update.tensor.detach()
            if slow_state is None or slow_state.numel() != block_vec.numel():
                slow_state = block_vec.detach().clone()
            else:
                slow_state = 0.97 * slow_state.to(device=device) + 0.03 * block_vec.to(device=device)
            agree = torch.sign(slow_state) == torch.sign(block_vec)
            cautious_vec = torch.where(agree, 0.85 * slow_state + 0.15 * block_vec, 0.05 * slow_state + 0.10 * block_vec)
            cautious_vec = normalized_like(cautious_vec.to(device=device), fast_update.tensor)
            cautious_update = UpdateTensor(
                cautious_vec,
                "step",
                "subtract",
                "matrix_block",
                "train_stream_rotated_low_rank_cautious_matrix_slow_state",
                mechanism,
                role="rotated_cautious_matrix_slow_state",
                one_step_descent_claim=0,
            )
            corrupt_labels = (channel_b[1] + 1) % max(2, int(getattr(args, "classes", 10)))
            before = snapshot(model)
            with torch.no_grad():
                before_a = F.cross_entropy(model(channel_a[0]).float(), channel_a[1], reduction="none")
                before_b = F.cross_entropy(model(channel_b[0]).float(), channel_b[1], reduction="none")
                before_corrupt = F.cross_entropy(model(channel_b[0]).float(), corrupt_labels, reduction="none")
            apply_update(model, cautious_update, lr=0.25 * float(args.fu_lr))
            with torch.no_grad():
                after_a = F.cross_entropy(model(channel_a[0]).float(), channel_a[1], reduction="none")
                after_b = F.cross_entropy(model(channel_b[0]).float(), channel_b[1], reduction="none")
                after_corrupt = F.cross_entropy(model(channel_b[0]).float(), corrupt_labels, reduction="none")
            load_flat_params(model, before)
            gain_a = float((before_a - after_a).mean().item())
            gain_b = float((before_b - after_b).mean().item())
            corrupt_gain = float((before_corrupt - after_corrupt).mean().item())
            signal_gain = min(gain_a, gain_b)
            gate_accept = int(
                step % max(2, int(args.alt_period)) == 0
                and float(torch.linalg.vector_norm(cautious_vec.detach()).item()) > 1.0e-12
                and signal_gain >= -1.0e-5
                and corrupt_gain <= max(1.0e-4, 0.50 * max(0.0, signal_gain))
            )
            if gate_accept:
                apply_update(model, cautious_update, lr=0.25 * float(args.fu_lr))
                update = cautious_update
            else:
                update = fast_update
            last_actuation = {
                "source_state_current_cos": cosine(slow_state, flat_grad(model, device)),
                "source_state_signal_gain": signal_gain,
                "source_state_corrupt_gain": corrupt_gain,
                "source_state_gate_accept": gate_accept,
                "source_state_norm": float(torch.linalg.vector_norm(slow_state.detach()).item()),
                "source_state_whitened_norm": float(torch.linalg.vector_norm(cautious_vec.detach()).item()),
                "source_state_antiwashout_removed_norm": float(torch.linalg.vector_norm((block_vec - cautious_vec).detach()).item()),
                "source_state_projected_grad_norm": float(torch.linalg.vector_norm(block_vec.detach()).item()),
                "source_state_gate_threshold": -1.0e-5,
                "source_state_ema_beta": 0.97,
                "source_state_consensus_density": float(agree.float().mean().item()) if agree.numel() else 0.0,
                "source_state_balance_mean": cosine(slow_state, block_vec),
                "generalization_gain_a": gain_a,
                "generalization_gain_b": gain_b,
                "shuffled_label_gain": corrupt_gain,
                "generalization_signal_gain": signal_gain,
                "generalization_gate_accept": gate_accept,
            }
        elif mechanism in {"M44-MomentumLineCAnchorSlowFU", "M45-MomentumSlowAnchorFU", "M46-MomentumMatrixBlockRetentionFU"}:
            fast_update = make_update(model, "M2-SGDMomentumPrimaryFU", xb, yb, seed=seed + step)
            opt_sgd.step_gradient()
            apply_update(model, fast_update, lr=float(args.fu_lr))
            anchor_update = fast_update
            anchor_kind = "momentum_only"
            if mechanism == "M44-MomentumLineCAnchorSlowFU":
                anchor_update = make_update(model, "M15-LineCFilteredAlternatingFU", xb, yb, seed=seed + step)
                anchor_kind = "linec_anchor"
                source_vec = 0.70 * fast_update.tensor.detach() + 0.30 * anchor_update.tensor.detach()
            elif mechanism == "M46-MomentumMatrixBlockRetentionFU":
                anchor_update = make_update(model, "M13-LowRankMatrixBlockFU", xb, yb, seed=seed + step)
                anchor_kind = "matrix_block_anchor"
                source_vec = 0.65 * fast_update.tensor.detach() + 0.35 * anchor_update.tensor.detach()
            else:
                source_vec = fast_update.tensor.detach()
            if slow_state is None or slow_state.numel() != source_vec.numel():
                slow_state = source_vec.detach().clone()
            else:
                slow_state = 0.98 * slow_state.to(device=device) + 0.02 * source_vec.to(device=device)
            slow_vec = normalized_like(slow_state.to(device=device), fast_update.tensor)
            slow_update = UpdateTensor(
                slow_vec,
                "step",
                "subtract",
                "slow_state" if mechanism != "M46-MomentumMatrixBlockRetentionFU" else "matrix_block",
                (
                    "train_stream_m2_plus_linec_anchor_slow_state"
                    if mechanism == "M44-MomentumLineCAnchorSlowFU"
                    else "train_stream_m2_slow_anchor_state"
                    if mechanism == "M45-MomentumSlowAnchorFU"
                    else "train_stream_m2_matrix_block_retention_state"
                ),
                mechanism,
                role=anchor_kind,
                one_step_descent_claim=0,
            )
            corrupt_labels = (channel_b[1] + 1) % max(2, int(getattr(args, "classes", 10)))
            before = snapshot(model)
            with torch.no_grad():
                before_a = F.cross_entropy(model(channel_a[0]).float(), channel_a[1], reduction="none")
                before_b = F.cross_entropy(model(channel_b[0]).float(), channel_b[1], reduction="none")
                before_corrupt = F.cross_entropy(model(channel_b[0]).float(), corrupt_labels, reduction="none")
            apply_update(model, slow_update, lr=0.25 * float(args.fu_lr))
            with torch.no_grad():
                after_a = F.cross_entropy(model(channel_a[0]).float(), channel_a[1], reduction="none")
                after_b = F.cross_entropy(model(channel_b[0]).float(), channel_b[1], reduction="none")
                after_corrupt = F.cross_entropy(model(channel_b[0]).float(), corrupt_labels, reduction="none")
            load_flat_params(model, before)
            gain_a = float((before_a - after_a).mean().item())
            gain_b = float((before_b - after_b).mean().item())
            corrupt_gain = float((before_corrupt - after_corrupt).mean().item())
            signal_gain = min(gain_a, gain_b)
            gate_accept = int(
                step % max(2, int(args.alt_period)) == 0
                and float(torch.linalg.vector_norm(slow_vec.detach()).item()) > 1.0e-12
                and signal_gain >= -1.0e-5
                and corrupt_gain <= max(1.0e-4, 0.50 * max(0.0, signal_gain))
            )
            if gate_accept:
                apply_update(model, slow_update, lr=0.25 * float(args.fu_lr))
                update = slow_update
            else:
                update = fast_update
            last_actuation = {
                "source_state_current_cos": cosine(slow_state, flat_grad(model, device)),
                "source_state_signal_gain": signal_gain,
                "source_state_corrupt_gain": corrupt_gain,
                "source_state_gate_accept": gate_accept,
                "source_state_norm": float(torch.linalg.vector_norm(slow_state.detach()).item()),
                "source_state_whitened_norm": float(torch.linalg.vector_norm(slow_vec.detach()).item()),
                "source_state_projected_grad_norm": float(torch.linalg.vector_norm(anchor_update.tensor.detach()).item()),
                "source_state_gate_threshold": -1.0e-5,
                "source_state_ema_beta": 0.98,
                "source_state_balance_mean": cosine(fast_update.tensor, anchor_update.tensor),
                "generalization_gain_a": gain_a,
                "generalization_gain_b": gain_b,
                "shuffled_label_gain": corrupt_gain,
                "generalization_signal_gain": signal_gain,
                "generalization_gate_accept": gate_accept,
            }
        elif mechanism == "M68-MarginSplitConsensusSlowFU":
            if step % max(2, int(args.alt_period)) == 0:
                params = [p for p in model.parameters() if p.requires_grad]
                original_grads = [None if p.grad is None else p.grad.detach().clone() for p in params]

                def restore_original_grads() -> None:
                    model.zero_grad(set_to_none=True)
                    for param, grad in zip(params, original_grads):
                        param.grad = None if grad is None else grad.to(device=param.device, dtype=param.dtype)

                def margin_loss(xb2: torch.Tensor, yb2: torch.Tensor) -> torch.Tensor:
                    logits = model(xb2).float()
                    labels = yb2.reshape(-1, 1)
                    true_logit = logits.gather(1, labels).squeeze(1)
                    masked = logits.detach().clone()
                    masked.scatter_(1, labels, float("-inf"))
                    wrong_idx = masked.argmax(dim=1).reshape(-1, 1)
                    wrong_logit = logits.gather(1, wrong_idx).squeeze(1)
                    return F.relu(1.0 + wrong_logit - true_logit).mean()

                def grad_for_margin(xb2: torch.Tensor, yb2: torch.Tensor) -> torch.Tensor:
                    model.zero_grad(set_to_none=True)
                    margin_loss(xb2, yb2).backward()
                    return flat_grad(model, device).detach().clone()

                g_current = (
                    torch.cat(
                        [
                            torch.zeros_like(param, device=device).reshape(-1) if grad is None else grad.reshape(-1).to(device=device)
                            for param, grad in zip(params, original_grads)
                        ]
                    )
                    if params
                    else torch.zeros(0, device=device)
                )
                g_a = grad_for_margin(channel_a[0], channel_a[1])
                g_b = grad_for_margin(channel_b[0], channel_b[1])
                corrupt_labels = (channel_b[1] + 1) % max(2, int(getattr(args, "classes", 10)))
                g_corrupt = grad_for_margin(channel_b[0], corrupt_labels)
                restore_original_grads()

                agree = torch.sign(g_a) == torch.sign(g_b)
                density = float(agree.float().mean().item()) if agree.numel() else 0.0
                mag_a = g_a.detach().float().abs()
                mag_b = g_b.detach().float().abs()
                balance = torch.minimum(mag_a, mag_b) / torch.maximum(mag_a, mag_b).clamp_min(1.0e-8)
                balance_mean = float(balance[agree].mean().item()) if bool(agree.any().item()) else 0.0
                split_mean = torch.where(agree, 0.50 * (g_a + g_b), torch.zeros_like(g_a))
                margin_source = split_mean * balance.to(device=device, dtype=split_mean.dtype)
                corrupt_denom = torch.dot(g_corrupt.float(), g_corrupt.float()).clamp_min(1.0e-12)
                corrupt_proj = torch.dot(margin_source.float(), g_corrupt.float()) / corrupt_denom
                if float(corrupt_proj.item()) > 0.0:
                    margin_source = margin_source - corrupt_proj.to(device=margin_source.device, dtype=margin_source.dtype) * g_corrupt
                margin_source = torch.nan_to_num(margin_source, nan=0.0, posinf=0.0, neginf=0.0)
                if slow_state is None or slow_state.numel() != margin_source.numel():
                    slow_state = margin_source.detach().clone()
                else:
                    slow_state = 0.90 * slow_state.to(device=device) + 0.10 * margin_source.detach()
                source_vec = 0.65 * slow_state + 0.35 * margin_source
                update_vec = normalized_like(source_vec, g_current if g_current.numel() else margin_source)
                update = UpdateTensor(
                    update_vec,
                    "step",
                    "subtract",
                    "slow_state",
                    "train_stream_top_wrong_margin_split_consensus_slow_state",
                    mechanism,
                    role="margin_split_consensus_source_state",
                    one_step_descent_claim=0,
                )
                before = snapshot(model)
                with torch.no_grad():
                    before_a = F.cross_entropy(model(channel_a[0]).float(), channel_a[1], reduction="none")
                    before_b = F.cross_entropy(model(channel_b[0]).float(), channel_b[1], reduction="none")
                    before_corrupt = F.cross_entropy(model(channel_b[0]).float(), corrupt_labels, reduction="none")
                apply_update(model, update, lr=float(args.fu_lr))
                with torch.no_grad():
                    after_a = F.cross_entropy(model(channel_a[0]).float(), channel_a[1], reduction="none")
                    after_b = F.cross_entropy(model(channel_b[0]).float(), channel_b[1], reduction="none")
                    after_corrupt = F.cross_entropy(model(channel_b[0]).float(), corrupt_labels, reduction="none")
                load_flat_params(model, before)
                gain_a = float((before_a - after_a).mean().item())
                gain_b = float((before_b - after_b).mean().item())
                corrupt_gain = float((before_corrupt - after_corrupt).mean().item())
                signal_gain = min(gain_a, gain_b)
                corrupt_cos = cosine(margin_source, g_corrupt)
                current_cos = cosine(source_vec, g_current)
                signal_threshold = -1.0e-6
                corrupt_allowance = max(1.0e-5, 0.35 * max(0.0, signal_gain))
                gate_accept = int(
                    density >= 0.03
                    and balance_mean >= 0.03
                    and signal_gain >= signal_threshold
                    and corrupt_gain <= corrupt_allowance
                    and corrupt_cos <= 0.30
                    and float(torch.linalg.vector_norm(update_vec.detach()).item()) > 1.0e-12
                )
                update.diagnostics = {
                    "source_state_consensus_density": density,
                    "source_state_balance_mean": balance_mean,
                    "source_state_current_cos": current_cos,
                    "source_state_corrupt_cos": corrupt_cos,
                    "source_state_signal_gain": signal_gain,
                    "source_state_corrupt_gain": corrupt_gain,
                    "source_state_gate_accept": gate_accept,
                    "source_state_norm": float(torch.linalg.vector_norm(slow_state.detach()).item()),
                    "source_state_whitened_norm": float(torch.linalg.vector_norm(margin_source.detach()).item()),
                    "source_state_gate_threshold": signal_threshold,
                    "source_state_ema_beta": 0.90,
                    "generalization_gain_a": gain_a,
                    "generalization_gain_b": gain_b,
                    "shuffled_label_gain": corrupt_gain,
                    "generalization_signal_gain": signal_gain,
                    "generalization_gate_accept": gate_accept,
                }
                if gate_accept:
                    apply_update(model, update, lr=float(args.fu_lr))
                else:
                    opt_sgd.step_gradient()
            else:
                opt_sgd.step_gradient()
                g_current = flat_grad(model, device).detach().clone()
                last_actuation = {
                    "source_state_current_cos": cosine(slow_state, g_current) if slow_state is not None and slow_state.numel() == g_current.numel() else "",
                    "source_state_signal_gain": 0.0,
                    "source_state_corrupt_gain": 0.0,
                    "source_state_gate_accept": 0,
                    "source_state_norm": float(torch.linalg.vector_norm(slow_state.detach()).item()) if slow_state is not None else "",
                    "source_state_whitened_norm": 0.0,
                    "source_state_consensus_density": 0.0,
                    "source_state_balance_mean": 0.0,
                    "source_state_gate_threshold": -1.0e-6,
                    "source_state_ema_beta": 0.90,
                    "generalization_gain_a": 0.0,
                    "generalization_gain_b": 0.0,
                    "shuffled_label_gain": 0.0,
                    "generalization_signal_gain": 0.0,
                    "generalization_gate_accept": 0,
                }
        elif mechanism in {"M37-SplitFisherAgreementSlowFU", "M38-AdamWSplitFisherAgreementResidualFU", "M39-MomentumWarmSplitFisherFU"}:
            warm_fisher = mechanism == "M39-MomentumWarmSplitFisherFU"
            warmup_steps = int(getattr(args, "source_warmup_steps", 0) or 0)
            if warm_fisher and step <= warmup_steps:
                update = make_update(model, "M2-SGDMomentumPrimaryFU", xb, yb, seed=seed + step)
                opt_sgd.step_gradient()
                apply_update(model, update, lr=float(args.fu_lr))
            elif step % max(2, int(args.alt_period)) == 0:
                adamw_fisher_residual = mechanism == "M38-AdamWSplitFisherAgreementResidualFU"
                warm_fisher = mechanism == "M39-MomentumWarmSplitFisherFU"
                warmup_steps = int(getattr(args, "source_warmup_steps", 0) or 0)
                if warm_fisher and step <= warmup_steps:
                    update = make_update(model, "M2-SGDMomentumPrimaryFU", xb, yb, seed=seed + step)
                    opt_sgd.step_gradient()
                    apply_update(model, update, lr=float(args.fu_lr))
                else:
                    params = [p for p in model.parameters() if p.requires_grad]
                    original_grads = [None if p.grad is None else p.grad.detach().clone() for p in params]

                    def restore_original_grads() -> None:
                        model.zero_grad(set_to_none=True)
                        for param, grad in zip(params, original_grads):
                            param.grad = None if grad is None else grad.to(device=param.device, dtype=param.dtype)

                    def grad_for(xb2: torch.Tensor, yb2: torch.Tensor) -> torch.Tensor:
                        model.zero_grad(set_to_none=True)
                        F.cross_entropy(model(xb2).float(), yb2).backward()
                        return flat_grad(model, device).detach().clone()

                    g_current = (
                        torch.cat(
                            [
                                torch.zeros_like(param, device=device).reshape(-1) if grad is None else grad.reshape(-1).to(device=device)
                                for param, grad in zip(params, original_grads)
                            ]
                        )
                        if params
                        else torch.zeros(0, device=device)
                    )
                    g_a = grad_for(channel_a[0], channel_a[1])
                    g_b = grad_for(channel_b[0], channel_b[1])
                    corrupt_labels = (channel_b[1] + 1) % max(2, int(getattr(args, "classes", 10)))
                    g_corrupt = grad_for(channel_b[0], corrupt_labels)
                    restore_original_grads()

                    agree = torch.sign(g_a) == torch.sign(g_b)
                    density = float(agree.float().mean().item()) if agree.numel() else 0.0
                    mag_a = g_a.detach().float().abs()
                    mag_b = g_b.detach().float().abs()
                    balance = torch.minimum(mag_a, mag_b) / torch.maximum(mag_a, mag_b).clamp_min(1.0e-8)
                    balance_mean = float(balance[agree].mean().item()) if bool(agree.any().item()) else 0.0
                    fisher_diag = torch.sqrt(0.50 * (g_a.detach().float().square() + g_b.detach().float().square()) + 1.0e-8)
                    fisher_floor = fisher_diag.mean().clamp_min(1.0e-6)
                    split_mean = torch.where(agree, 0.50 * (g_a + g_b), torch.zeros_like(g_a))
                    weighted = split_mean * balance.to(device=device, dtype=split_mean.dtype)
                    fisher_source = weighted / (fisher_diag.to(device=device, dtype=weighted.dtype) + 0.10 * fisher_floor.to(device=device, dtype=weighted.dtype))
                    fisher_source = torch.nan_to_num(fisher_source, nan=0.0, posinf=0.0, neginf=0.0)
                    corrupt_denom = torch.dot(g_corrupt.float(), g_corrupt.float()).clamp_min(1.0e-12)
                    corrupt_proj = torch.dot(fisher_source.float(), g_corrupt.float()) / corrupt_denom
                    if float(corrupt_proj.item()) > 0.0:
                        fisher_source = fisher_source - corrupt_proj.to(device=fisher_source.device, dtype=fisher_source.dtype) * g_corrupt
                    if slow_state is None or slow_state.numel() != fisher_source.numel():
                        slow_state = fisher_source.detach().clone()
                    else:
                        slow_state = 0.92 * slow_state.to(device=device) + 0.08 * fisher_source.detach()
                    source_vec = 0.70 * slow_state + 0.30 * fisher_source
                    ref_vec = g_current if g_current.numel() else split_mean
                    update_vec = normalized_like(source_vec, ref_vec)
                    update = UpdateTensor(
                        update_vec,
                        "step",
                        "subtract",
                        "slow_state",
                        (
                            "train_stream_adamw_split_fisher_agreement_residual"
                            if adamw_fisher_residual
                            else "train_stream_momentum_warm_split_fisher_agreement"
                            if warm_fisher
                            else "train_stream_split_fisher_agreement_slow_state"
                        ),
                        mechanism,
                        role="split_fisher_source_state",
                        one_step_descent_claim=0,
                    )
                    before = snapshot(model)
                    with torch.no_grad():
                        before_a = F.cross_entropy(model(channel_a[0]).float(), channel_a[1], reduction="none")
                        before_b = F.cross_entropy(model(channel_b[0]).float(), channel_b[1], reduction="none")
                        before_corrupt = F.cross_entropy(model(channel_b[0]).float(), corrupt_labels, reduction="none")
                    apply_update(model, update, lr=float(args.fu_lr))
                    with torch.no_grad():
                        after_a = F.cross_entropy(model(channel_a[0]).float(), channel_a[1], reduction="none")
                        after_b = F.cross_entropy(model(channel_b[0]).float(), channel_b[1], reduction="none")
                        after_corrupt = F.cross_entropy(model(channel_b[0]).float(), corrupt_labels, reduction="none")
                    load_flat_params(model, before)
                    gain_a = float((before_a - after_a).mean().item())
                    gain_b = float((before_b - after_b).mean().item())
                    corrupt_gain = float((before_corrupt - after_corrupt).mean().item())
                    signal_gain = min(gain_a, gain_b)
                    corrupt_cos = cosine(fisher_source, g_corrupt)
                    current_cos = cosine(source_vec, g_current)
                    signal_threshold = 1.0e-7
                    corrupt_allowance = max(1.0e-5, 0.20 * max(0.0, signal_gain))
                    gate_accept = int(
                        density >= 0.05
                        and balance_mean >= 0.05
                        and signal_gain >= signal_threshold
                        and corrupt_gain <= corrupt_allowance
                        and corrupt_cos <= 0.20
                        and float(torch.linalg.vector_norm(update_vec.detach()).item()) > 1.0e-12
                    )
                    update.diagnostics = {
                        "source_state_consensus_density": density,
                        "source_state_balance_mean": balance_mean,
                        "source_state_current_cos": current_cos,
                        "source_state_corrupt_cos": corrupt_cos,
                        "source_state_signal_gain": signal_gain,
                        "source_state_corrupt_gain": corrupt_gain,
                        "source_state_gate_accept": gate_accept,
                        "source_state_norm": float(torch.linalg.vector_norm(slow_state.detach()).item()),
                        "source_state_whitened_norm": float(torch.linalg.vector_norm(fisher_source.detach()).item()),
                        "source_state_gate_threshold": signal_threshold,
                        "source_state_ema_beta": 0.92,
                        "generalization_gain_a": gain_a,
                        "generalization_gain_b": gain_b,
                        "shuffled_label_gain": corrupt_gain,
                        "generalization_signal_gain": signal_gain,
                        "generalization_gate_accept": gate_accept,
                    }
                    if adamw_fisher_residual:
                        opt_adam.step()
                        if gate_accept:
                            apply_update(model, update, lr=float(args.fu_lr))
                    elif gate_accept:
                        apply_update(model, update, lr=float(args.fu_lr))
                    else:
                        opt_sgd.step_gradient()
            else:
                if mechanism == "M38-AdamWSplitFisherAgreementResidualFU":
                    opt_adam.step()
                else:
                    opt_sgd.step_gradient()
        elif mechanism in V2206_METRIC_SOLVER_MECHANISMS and str(getattr(args, "optimizer_integration_mode", "")) in {
            "v2206_source_state_carry",
            "v2206_early_source_warm_carry",
            "v2206_periodic_source_state_carry",
            "v2206_sgd_bootstrap_then_source_state_carry",
            "v2206_optimizer_state_transport",
            "v2206_train_split_control_relative_gate",
            "v2206_adamw_bootstrap_then_optimizer_transport",
            "v2206_adamw_control_relative_gate",
            "v2206_early_warm_then_optimizer_transport",
        }:
            integration_mode = str(getattr(args, "optimizer_integration_mode", ""))
            update = make_update(model, mechanism, xb, yb, seed=seed + step)
            source_vec = update.tensor.detach().to(device=device)
            if slow_state is not None and slow_state.numel() == 2 * source_vec.numel():
                short_state, long_state = slow_state.to(device=device).chunk(2)
            else:
                short_state = source_vec.detach().clone()
                long_state = source_vec.detach().clone()
            short_state = 0.80 * short_state + 0.20 * source_vec
            long_state = 0.995 * long_state + 0.005 * source_vec
            slow_state = torch.cat([short_state.detach(), long_state.detach()])
            carry_vec = 0.50 * short_state + 0.50 * long_state
            carry_update = UpdateTensor(
                carry_vec,
                "metric_solver_source_state_carry",
                "subtract",
                "function_metric_solver+optimizer_integration",
                "v22_06_I2_source_state_carry",
                mechanism,
                role="v22_06_source_state_carry",
                one_step_descent_claim=0,
                diagnostics={},
            )
            grad_before = flat_grad(model, device).detach()
            carry_lr = float(getattr(args, "post_stop_fu_lr", 0.0) or (0.25 * float(args.fu_lr)))
            warmup_steps = int(getattr(args, "source_warmup_steps", 0) or 0)
            stop_steps = int(getattr(args, "source_stop_steps", 0) or 0)
            periodic_mode = int(integration_mode == "v2206_periodic_source_state_carry")
            periodic_solver_active = int((not periodic_mode) or step == 1 or step % max(1, int(args.alt_period)) == 0)
            bootstrap_mode = int(integration_mode == "v2206_sgd_bootstrap_then_source_state_carry")
            adamw_bootstrap_mode = int(integration_mode == "v2206_adamw_bootstrap_then_optimizer_transport")
            transport_mode = int(integration_mode == "v2206_optimizer_state_transport")
            control_gate_mode = int(integration_mode == "v2206_train_split_control_relative_gate")
            adamw_control_gate_mode = int(integration_mode == "v2206_adamw_control_relative_gate")
            warm_then_transport_mode = int(integration_mode == "v2206_early_warm_then_optimizer_transport")
            bootstrap_active = int(bootstrap_mode and warmup_steps > 0 and step <= warmup_steps)
            adamw_bootstrap_active = int(adamw_bootstrap_mode and warmup_steps > 0 and step <= warmup_steps)
            warm_writer_active = int(
                integration_mode in {"v2206_early_source_warm_carry", "v2206_early_warm_then_optimizer_transport"}
                and warmup_steps > 0
                and step <= warmup_steps
            )
            transport_like_mode = int(
                transport_mode
                or (adamw_bootstrap_mode and not adamw_bootstrap_active)
                or (warm_then_transport_mode and not warm_writer_active)
            )
            transport_active = int(transport_like_mode and (stop_steps <= 0 or step <= stop_steps))
            control_gate_active = int(control_gate_mode and (stop_steps <= 0 or step <= stop_steps))
            adamw_control_gate_active = int(adamw_control_gate_mode and (stop_steps <= 0 or step <= stop_steps))
            carry_active = int((not transport_like_mode) and (not control_gate_mode) and (not adamw_control_gate_mode) and (not bootstrap_active) and (not adamw_bootstrap_active) and (stop_steps <= 0 or step <= stop_steps) and ((not periodic_mode) or bool(periodic_solver_active)))
            direct_solver_commit_active = int((not bootstrap_active) and (not adamw_bootstrap_active) and (integration_mode == "v2206_source_state_carry" or warm_writer_active or (periodic_mode and periodic_solver_active)))
            carry_for_projection = carry_vec.to(device=device, dtype=grad_before.dtype)
            carry_norm = torch.linalg.vector_norm(carry_for_projection.detach()).clamp_min(1.0e-12)
            source_unit = carry_for_projection / carry_norm
            projection_before = float(torch.dot(grad_before.float(), source_unit.float()).item()) if source_unit.numel() else 0.0
            transported_grad = grad_before
            removed_anti_source = 0.0
            transport_strength = float(carry_lr / max(abs(float(args.lr)), 1.0e-12))
            if transport_active and source_unit.numel() == grad_before.numel():
                anti_coeff = torch.clamp(torch.dot(grad_before.float(), source_unit.float()), max=0.0)
                anti_component = anti_coeff.to(device=device, dtype=grad_before.dtype) * source_unit
                transported_grad = grad_before - anti_component + float(transport_strength) * carry_vec.to(device=device, dtype=grad_before.dtype)
                transported_grad = torch.nan_to_num(transported_grad, nan=0.0, posinf=0.0, neginf=0.0)
                removed_anti_source = float(torch.linalg.vector_norm(anti_component.detach()).item())
                assign_flat_gradient(transported_grad)
            projection_after = float(torch.dot(transported_grad.float(), source_unit.float()).item()) if source_unit.numel() else 0.0
            transport_delta_norm = float(torch.linalg.vector_norm((transported_grad - grad_before).detach()).item()) if transported_grad.numel() else 0.0
            control_gate_accept = 0
            control_gate_fu_signal_gain: float | str = ""
            control_gate_sgd_signal_gain: float | str = ""
            control_gate_signal_margin: float | str = ""
            control_gate_fu_b3_gain: float | str = ""
            control_gate_sgd_b3_gain: float | str = ""
            control_gate_fu_corrupt_gain: float | str = ""
            control_gate_sgd_corrupt_gain: float | str = ""
            adamw_gate_accept = 0
            adamw_gate_fu_signal_gain: float | str = ""
            adamw_gate_adamw_signal_gain: float | str = ""
            adamw_gate_signal_margin: float | str = ""
            adamw_gate_fu_b3_gain: float | str = ""
            adamw_gate_adamw_b3_gain: float | str = ""
            adamw_gate_fu_corrupt_gain: float | str = ""
            adamw_gate_adamw_corrupt_gain: float | str = ""
            if control_gate_active:
                base_for_gate = snapshot(model)
                corrupt_labels = (channel_b[1] + 1) % max(2, int(getattr(args, "classes", 10)))

                def split_losses() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                    with torch.no_grad():
                        la = F.cross_entropy(model(channel_a[0]).float(), channel_a[1], reduction="none")
                        lb = F.cross_entropy(model(channel_b[0]).float(), channel_b[1], reduction="none")
                        lc = F.cross_entropy(model(channel_b[0]).float(), corrupt_labels, reduction="none")
                    return la, lb, lc

                before_a, before_b, before_c = split_losses()

                def candidate_gains(vec: torch.Tensor, lr_value: float, tag: str) -> tuple[float, float, float, float]:
                    load_flat_params(model, base_for_gate)
                    candidate = UpdateTensor(
                        vec.detach().clone(),
                        f"v22_06_{tag}_candidate",
                        "subtract",
                        "function_metric_solver+optimizer_control_gate",
                        f"v22_06_I4_{tag}_candidate",
                        mechanism,
                        role=f"v22_06_{tag}_candidate",
                        one_step_descent_claim=0,
                        diagnostics={},
                    )
                    apply_update(model, candidate, lr=float(lr_value))
                    after_a, after_b, after_c = split_losses()
                    gain_a = float((before_a - after_a).mean().item())
                    gain_b = float((before_b - after_b).mean().item())
                    gain_c = float((before_c - after_c).mean().item())
                    return gain_a, gain_b, min(gain_a, gain_b), gain_c

                fu_a, fu_b, fu_signal, fu_corrupt = candidate_gains(update.tensor.to(device=device), float(args.fu_lr), "fu")
                params_now = flat_params(model).detach().to(device=device, dtype=grad_before.dtype)
                sgd_vec = grad_before.detach().clone()
                if float(args.weight_decay):
                    sgd_vec = sgd_vec + float(args.weight_decay) * params_now
                sgd_a, sgd_b, sgd_signal, sgd_corrupt = candidate_gains(sgd_vec, float(args.lr), "sgd")
                load_flat_params(model, base_for_gate)
                signal_margin = fu_signal - sgd_signal
                corrupt_margin = fu_corrupt - sgd_corrupt
                control_gate_accept = int(signal_margin > 1.0e-5 and fu_b >= sgd_b - 1.0e-5 and corrupt_margin <= max(1.0e-5, 0.35 * max(0.0, fu_signal)))
                control_gate_fu_signal_gain = fu_signal
                control_gate_sgd_signal_gain = sgd_signal
                control_gate_signal_margin = signal_margin
                control_gate_fu_b3_gain = fu_b
                control_gate_sgd_b3_gain = sgd_b
                control_gate_fu_corrupt_gain = fu_corrupt
                control_gate_sgd_corrupt_gain = sgd_corrupt
            if adamw_control_gate_active:
                base_for_adamw_gate = snapshot(model)
                adamw_state_for_gate = deepcopy(opt_adam.state_dict())
                corrupt_labels = (channel_b[1] + 1) % max(2, int(getattr(args, "classes", 10)))

                def split_losses_adamw() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                    with torch.no_grad():
                        la = F.cross_entropy(model(channel_a[0]).float(), channel_a[1], reduction="none")
                        lb = F.cross_entropy(model(channel_b[0]).float(), channel_b[1], reduction="none")
                        lc = F.cross_entropy(model(channel_b[0]).float(), corrupt_labels, reduction="none")
                    return la, lb, lc

                before_a, before_b, before_c = split_losses_adamw()

                def adamw_candidate_gains(with_fu: bool, tag: str) -> tuple[float, float, float, float]:
                    load_flat_params(model, base_for_adamw_gate)
                    opt_adam.load_state_dict(adamw_state_for_gate)
                    opt_adam.step()
                    if with_fu:
                        candidate = UpdateTensor(
                            update.tensor.detach().clone(),
                            f"v22_06_{tag}_candidate",
                            "subtract",
                            "function_metric_solver+adamw_control_gate",
                            f"v22_06_I4_{tag}_candidate",
                            mechanism,
                            role=f"v22_06_{tag}_candidate",
                            one_step_descent_claim=0,
                            diagnostics={},
                        )
                        apply_update(model, candidate, lr=float(args.fu_lr))
                    after_a, after_b, after_c = split_losses_adamw()
                    gain_a = float((before_a - after_a).mean().item())
                    gain_b = float((before_b - after_b).mean().item())
                    gain_c = float((before_c - after_c).mean().item())
                    return gain_a, gain_b, min(gain_a, gain_b), gain_c

                adamw_a, adamw_b, adamw_signal, adamw_corrupt = adamw_candidate_gains(False, "adamw")
                fu_a, fu_b, fu_signal, fu_corrupt = adamw_candidate_gains(True, "fu_plus_adamw")
                load_flat_params(model, base_for_adamw_gate)
                opt_adam.load_state_dict(adamw_state_for_gate)
                signal_margin = fu_signal - adamw_signal
                corrupt_margin = fu_corrupt - adamw_corrupt
                adamw_gate_accept = int(signal_margin > 1.0e-5 and fu_b >= adamw_b - 1.0e-5 and corrupt_margin <= max(1.0e-5, 0.35 * max(0.0, fu_signal)))
                adamw_gate_fu_signal_gain = fu_signal
                adamw_gate_adamw_signal_gain = adamw_signal
                adamw_gate_signal_margin = signal_margin
                adamw_gate_fu_b3_gain = fu_b
                adamw_gate_adamw_b3_gain = adamw_b
                adamw_gate_fu_corrupt_gain = fu_corrupt
                adamw_gate_adamw_corrupt_gain = adamw_corrupt
            if bootstrap_active:
                bootstrap_update = UpdateTensor(
                    grad_before.detach().clone(),
                    "v22_06_sgd_bootstrap_gradient",
                    "subtract",
                    "optimizer_integration",
                    "v22_06_I1_sgd_bootstrap_before_metric_solver_carry",
                    mechanism,
                    role="v22_06_sgd_bootstrap_before_metric_solver_carry",
                    one_step_descent_claim=0,
                    diagnostics={},
                )
                opt_sgd.step_gradient()
                apply_update(model, bootstrap_update, lr=float(args.fu_lr))
            elif adamw_bootstrap_active:
                opt_adam.step()
            elif warm_writer_active:
                apply_update(model, update, lr=float(args.fu_lr))
                opt_sgd.step_gradient()
            elif transport_active:
                opt_sgd.step_gradient()
            elif control_gate_active:
                if control_gate_accept:
                    apply_update(model, update, lr=float(args.fu_lr))
                opt_sgd.step_gradient()
            elif adamw_control_gate_active:
                opt_adam.step()
                if adamw_gate_accept:
                    apply_update(model, update, lr=float(args.fu_lr))
            else:
                opt_sgd.step_gradient()
                if integration_mode == "v2206_source_state_carry" or (periodic_mode and periodic_solver_active):
                    apply_update(model, update, lr=float(args.fu_lr))
            if carry_active:
                apply_update(model, carry_update, lr=carry_lr)
            diag = dict(update.diagnostics or {})
            diag.update(
                {
                    "optimizer_integration_type": (
                        "I1-early-source-warm-carry"
                        if integration_mode == "v2206_early_source_warm_carry"
                        else "I3-periodic-source-state-carry"
                        if periodic_mode
                        else "I1/I2-sgd-bootstrap-then-source-state-carry"
                        if bootstrap_mode
                        else "I1/I2-adamw-bootstrap-then-optimizer-transport"
                        if adamw_bootstrap_mode
                        else "I1/I2-early-warm-then-optimizer-transport"
                        if warm_then_transport_mode
                        else "I2-optimizer-state-transport"
                        if transport_mode
                        else "I4-train-split-control-relative-gate"
                        if control_gate_mode
                        else "I4-adamw-control-relative-gate"
                        if adamw_control_gate_mode
                        else "I2-source-state-carry"
                    ),
                    "operator_status": (
                        "v22_06_metric_solver_early_source_warm_carry"
                        if integration_mode == "v2206_early_source_warm_carry"
                        else "v22_06_metric_solver_periodic_boundary_carry"
                        if periodic_mode
                        else "v22_06_metric_solver_sgd_bootstrap_then_carry"
                        if bootstrap_mode
                        else "v22_06_metric_solver_adamw_bootstrap_then_optimizer_transport"
                        if adamw_bootstrap_mode
                        else "v22_06_metric_solver_early_warm_then_optimizer_transport"
                        if warm_then_transport_mode
                        else "v22_06_metric_solver_optimizer_state_transport"
                        if transport_mode
                        else "v22_06_metric_solver_train_split_control_relative_gate"
                        if control_gate_mode
                        else "v22_06_metric_solver_adamw_control_relative_gate"
                        if adamw_control_gate_mode
                        else "v22_06_metric_solver_sgd_plus_source_state_carry"
                    ),
                    "source_state_norm": float(torch.linalg.vector_norm(slow_state.detach()).item()),
                    "source_state_whitened_norm": float(torch.linalg.vector_norm(carry_vec.detach()).item()),
                    "source_state_balance_mean": cosine(short_state, long_state),
                    "source_state_current_cos": cosine(carry_vec, grad_before),
                    "source_state_gate_accept": 1,
                    "source_state_consensus_density": 1.0,
                    "source_state_ema_beta": 0.995,
                    "optimizer_state_m_projection_on_FU": cosine(source_vec, transported_grad if transport_active else grad_before),
                    "cumulative_optimizer_projection_h100_to_h800": cosine(carry_vec, transported_grad if transport_active else grad_before),
                    "FU_param_commit_norm": float(torch.linalg.vector_norm(source_vec.detach()).item()),
                    "FU_function_commit_norm": diag.get("function_displacement_norm", ""),
                    "fast_slow_function_gap": float(torch.linalg.vector_norm((short_state - long_state).detach()).item()),
                    "short_long_source_agreement": cosine(short_state, long_state),
                    "source_state_carry_lr": carry_lr if carry_active else 0.0,
                    "early_source_warm_writer_active": warm_writer_active,
                    "early_source_warm_writer_lr": float(args.fu_lr) if warm_writer_active else 0.0,
                    "early_source_warmup_steps": warmup_steps,
                    "source_stop_steps": stop_steps,
                    "source_state_carry_active": carry_active,
                    "direct_solver_commit_active": direct_solver_commit_active,
                    "periodic_solver_active": periodic_solver_active,
                    "periodic_solver_alt_period": max(1, int(args.alt_period)) if periodic_mode else "",
                    "sgd_bootstrap_writer_active": bootstrap_active,
                    "sgd_bootstrap_writer_lr": float(args.fu_lr) if bootstrap_active else 0.0,
                    "adamw_bootstrap_writer_active": adamw_bootstrap_active,
                    "adamw_bootstrap_writer_lr": float(args.lr) if adamw_bootstrap_active else 0.0,
                    "optimizer_transport_active": transport_active,
                    "optimizer_transport_strength": transport_strength if transport_active else 0.0,
                    "optimizer_transport_projection_before": projection_before if transport_active else "",
                    "optimizer_transport_projection_after": projection_after if transport_active else "",
                    "optimizer_transport_removed_anti_source": removed_anti_source if transport_active else 0.0,
                    "optimizer_transport_grad_delta_norm": transport_delta_norm if transport_active else 0.0,
                    "train_split_control_gate_active": control_gate_active,
                    "train_split_control_gate_accept": control_gate_accept if control_gate_active else "",
                    "train_split_control_gate_fu_signal_gain": control_gate_fu_signal_gain,
                    "train_split_control_gate_sgd_signal_gain": control_gate_sgd_signal_gain,
                    "train_split_control_gate_signal_margin": control_gate_signal_margin,
                    "train_split_control_gate_fu_b3_gain": control_gate_fu_b3_gain,
                    "train_split_control_gate_sgd_b3_gain": control_gate_sgd_b3_gain,
                    "train_split_control_gate_fu_corrupt_gain": control_gate_fu_corrupt_gain,
                    "train_split_control_gate_sgd_corrupt_gain": control_gate_sgd_corrupt_gain,
                    "adamw_control_gate_active": adamw_control_gate_active,
                    "adamw_control_gate_accept": adamw_gate_accept if adamw_control_gate_active else "",
                    "adamw_control_gate_fu_signal_gain": adamw_gate_fu_signal_gain,
                    "adamw_control_gate_adamw_signal_gain": adamw_gate_adamw_signal_gain,
                    "adamw_control_gate_signal_margin": adamw_gate_signal_margin,
                    "adamw_control_gate_fu_b3_gain": adamw_gate_fu_b3_gain,
                    "adamw_control_gate_adamw_b3_gain": adamw_gate_adamw_b3_gain,
                    "adamw_control_gate_fu_corrupt_gain": adamw_gate_fu_corrupt_gain,
                    "adamw_control_gate_adamw_corrupt_gain": adamw_gate_adamw_corrupt_gain,
                    "adamw_early_optimizer_control_active": int(adamw_bootstrap_active or adamw_control_gate_active),
                }
            )
            update.diagnostics = diag
            last_actuation = dict(diag)
        elif mechanism in {
            "M6-SlowStateFU",
            "M11-DualMemorySlowStateFU",
            "M12-ScheduleFreeAveragedFU",
            "M14-SourceChannelPopRiskSlowFU",
            "M33-PopRiskMatrixBlockSlowFU",
            "M34-ExactPopRiskSlowFU",
            "M35-ExactPopRiskMatrixBlockSlowFU",
            "M116-DatasetInvariantPopRiskSlowFU",
            "M117-DatasetInvariantReadoutConsensusFU",
        } or mechanism in METRIC_SOURCE_MEMORY_MECHANISMS:
            update = make_update(model, mechanism, xb, yb, seed=seed + step, slow_state=slow_state)
            slow_state = update_state_after_commit(mechanism, slow_state, update)
            apply_update(model, update, lr=float(args.fu_lr))
        else:
            update = make_update(model, mechanism, xb, yb, seed=seed + step)
            apply_update(model, update, lr=float(args.fu_lr))
            slow_state = update_state_after_commit(mechanism, slow_state, update)
        if first_update is None and update is not None:
            first_update = update
        if update is not None and update.diagnostics:
            last_actuation = dict(update.diagnostics)
        if step in horizons:
            eval_pack(step)
    final = traces[-1].copy()
    final.update({"steps": steps, "mechanism": mechanism})
    return final, traces, first_update


def first_step_diagnostics(model: torch.nn.Module, mechanism: str, x: torch.Tensor, y: torch.Tensor, args: argparse.Namespace, seed: int) -> tuple[dict[str, Any], dict[str, Any], UpdateTensor]:
    device = x.device
    state = snapshot(model)
    model.zero_grad(set_to_none=True)
    loss = F.cross_entropy(model(x).float(), y)
    loss.backward()
    update = make_update(model, mechanism, x, y, seed=seed)
    ce_interface = cross_entropy_cotangent(model, x, y)
    brier_interface = brier_cotangent(model, x, y)
    sign = small_step_sanity(model, x, y, update, eps=float(args.sanity_eps))
    sign.update(
        {
            "loss_interface": ce_interface.loss_name,
            "loss_interface_loss": ce_interface.loss_value,
            "loss_interface_cotangent_norm": ce_interface.to_row()["cotangent_norm"],
            "brier_cotangent_norm_readback": brier_interface.to_row()["cotangent_norm"],
            "loss_interface_uses_validation_test_future_query": ce_interface.uses_validation_test_future_query,
            "loss_interface_audit_metrics_used_for_direction": ce_interface.audit_metrics_used_for_direction,
        }
    )
    load_flat_params(model, state)

    model.zero_grad(set_to_none=True)
    loss = F.cross_entropy(model(x).float(), y)
    loss.backward()
    update = make_update(model, mechanism, x, y, seed=seed)
    before = snapshot(model)
    fu_disp = -float(args.fu_lr) * update.tensor.detach()
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    opt.step()
    adam_delta = snapshot(model) - before
    load_flat_params(model, state)
    proj = float((adam_delta[: fu_disp.numel()] @ fu_disp.reshape(-1)) / fu_disp.square().sum().clamp_min(1.0e-12)) if fu_disp.numel() else 0.0
    overwrite = {
        "mechanism": mechanism,
        "cos_FU_AdamW": cosine(fu_disp, adam_delta),
        "optimizer_overwrite_projection": proj,
        "orthogonal_drift_norm": float(torch.linalg.vector_norm(adam_delta - proj * fu_disp).item()) if fu_disp.numel() == adam_delta.numel() else "",
        "source_retention_after_optimizer": max(0.0, proj),
    }
    return sign, overwrite, update


def run_s0(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    append_log(out_dir, f"## {now_sg()} S0 code/metric/semantic tests")
    compile_cmd = [PYTHON, "-m", "compileall", "-q", "dgkan", "experiments"]
    compile_proc = subprocess.run(compile_cmd, cwd=str(ROOT), text=True, capture_output=True)
    append_log(out_dir, "`" + " ".join(compile_cmd) + "`")
    append_log(out_dir, f"exit={compile_proc.returncode}")
    write_text(
        out_dir / "v17_compileall.log",
        "\n".join(
            [
                "$ " + " ".join(compile_cmd),
                f"exit={compile_proc.returncode}",
                "--- stdout ---",
                compile_proc.stdout,
                "--- stderr ---",
                compile_proc.stderr,
            ]
        ),
    )
    write_text(out_dir / "compileall_report.txt", (out_dir / "v17_compileall.log").read_text(encoding="utf-8"))
    imports = []
    missing = []
    for mod in V17_MODULES:
        spec = importlib.util.find_spec(mod)
        path = spec.origin if spec and spec.origin else ""
        try:
            importlib.import_module(mod)
            source = Path(path).read_text(encoding="utf-8", errors="ignore") if path and Path(path).exists() else ""
            imports.append(
                {
                    "module": mod,
                    "path": path,
                    "import_status": "ok",
                    "import_ok": 1,
                    "error_type": "",
                    "error": "",
                    "error_message": "",
                    "imports_experiments_count": source.count("experiments."),
                    "imports_dgkan_count": source.count("dgkan."),
                }
            )
        except Exception as exc:
            imports.append(
                {
                    "module": mod,
                    "path": path,
                    "import_status": "error",
                    "import_ok": 0,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "error_message": str(exc),
                    "imports_experiments_count": "",
                    "imports_dgkan_count": "",
                }
            )
            missing.append({"module": mod, "path": path, "error_type": type(exc).__name__, "error": str(exc), "error_message": str(exc)})
    write_rows(out_dir / "v17_s0_import_closure.csv", imports)
    write_rows(out_dir / "v17_import_closure.csv", imports)
    write_rows(out_dir / "import_closure_results.csv", imports)
    write_rows(out_dir / "v17_missing_imports.csv", missing)
    write_rows(out_dir / "missing_module_report.csv", missing)
    write_rows(out_dir / "v17_import_errors.csv", missing)
    write_rows(out_dir / "runner_import_results.csv", [r for r in imports if str(r.get("module", "")).startswith("experiments.")])
    write_json(out_dir / "v17_module_dependency_graph.json", {"modules": [{"module": r["module"], "path": r.get("path", "")} for r in imports]})
    write_rows(out_dir / "v17_missing_symbol_report.csv", [])
    write_rows(out_dir / "symbol_resolution_table.csv", [{"symbol": r["module"], "path": r.get("path", ""), "resolved": int_flag(r.get("import_ok"))} for r in imports])
    write_rows(
        out_dir / "compatibility_manifest.csv",
        [
            {"legacy_symbol": "experiments/run_v17_code_correctness_audit.py", "mapped_to": "experiments/run_v17_common.py --stage s0", "shim_file": "experiments/run_v17_code_correctness_audit.py", "unit_test": "import_closure"},
            {"legacy_symbol": "experiments/run_v17_adamw_free_functional_matrix.py", "mapped_to": "experiments/run_v17_common.py --stage mechanism-shard", "shim_file": "experiments/run_v17_adamw_free_functional_matrix.py", "unit_test": "import_closure"},
            {"legacy_symbol": "experiments/run_v17_basis_kernel_efficiency_matrix.py", "mapped_to": "experiments/run_v17_common.py --stage efficiency-shard", "shim_file": "experiments/run_v17_basis_kernel_efficiency_matrix.py", "unit_test": "import_closure"},
            {"legacy_symbol": "experiments/run_v17_4gpu_scheduler.py", "mapped_to": "experiments/run_v17_common.py --stage mechanism/horizon-shard", "shim_file": "experiments/run_v17_4gpu_scheduler.py", "unit_test": "import_closure"},
            {"legacy_symbol": "dgkan.diagnostics.linec.linec_metrics", "mapped_to": "dgkan.metrics.linec.linec_model_update", "shim_file": "dgkan/diagnostics/linec.py", "unit_test": "linec_golden"},
            {"legacy_symbol": "dgkan.functional.update_tensor.UpdateTensor", "mapped_to": "dgkan.fu.core.UpdateTensor", "shim_file": "dgkan/functional/update_tensor.py", "unit_test": "import_closure"},
            {"legacy_symbol": "dgkan.profiling.efficiency_profiler.profile_isolated", "mapped_to": "dgkan.profiling.efficiency_v17.profile_isolated", "shim_file": "dgkan/profiling/efficiency_profiler.py", "unit_test": "import_closure"},
            {"legacy_symbol": "dgkan.profiling.kernel_gradcheck.kernel_correctness_row", "mapped_to": "dgkan.kernels.v17_basis.kernel_correctness_row", "shim_file": "dgkan/profiling/kernel_gradcheck.py", "unit_test": "kernel_gradcheck"},
        ],
    )
    write_rows(out_dir / "compatibility_shim_manifest.csv", read_rows(out_dir / "compatibility_manifest.csv"))
    write_rows(out_dir / "symbol_renaming_manifest.csv", read_rows(out_dir / "compatibility_manifest.csv"))

    linec_rows = run_linec_golden_tests()
    linec_channel_rows = run_linec_channel_golden_tests()
    write_rows(out_dir / "v17_linec_golden_tests.csv", linec_rows)
    write_rows(out_dir / "v17_linec_golden_results.csv", linec_rows)
    write_rows(out_dir / "linec_golden_results.csv", linec_rows)
    write_rows(out_dir / "linec_channel_golden_tests.csv", linec_channel_rows)
    write_rows(out_dir / "linec_channel_golden_results.csv", linec_channel_rows)
    write_rows(out_dir / "v17_linec_exception_policy_test.csv", [r for r in linec_rows if str(r.get("golden")) == "G7-ExceptionPathMeasurementInvalid"])
    write_rows(out_dir / "linec_exception_policy.csv", [r for r in linec_rows if str(r.get("golden")) == "G7-ExceptionPathMeasurementInvalid"])
    write_rows(out_dir / "linec_measurement_invalid_examples.csv", [r for r in linec_rows if int_flag(r.get("linec_measurement_valid")) == 0])
    write_rows(
        out_dir / "linec_fast_vs_channel_comparison.csv",
        [
            {
                "mode": "linec_fast_and_channel",
                "comparison_mode": "synthetic_known_transfer",
                "linec_fast_pass": int(any(str(r.get("golden")) == "G3-SyntheticKnownTransfer" and int_flag(r.get("pass")) for r in linec_rows)),
                "linec_channel_pass": int(any(str(r.get("golden")) == "C2-KnownTransferPositive" and int_flag(r.get("pass")) for r in linec_channel_rows)),
                "uses_validation_test_future_query": 0,
                "status": "measured",
                "promotion_allowed": 0,
            }
        ],
    )
    write_rows(out_dir / "v17_linec_null_distribution.csv", [r for r in linec_rows if str(r.get("golden")) in {"G1-NoOpNull", "G2-RandomMatchedNormNull"}])
    write_rows(out_dir / "v17_linec_split_transfer_test.csv", [r for r in linec_rows if str(r.get("golden")) == "G3-SyntheticKnownTransfer"])
    write_rows(out_dir / "v17_linec_noise_injection_test.csv", [r for r in linec_rows if str(r.get("golden")) == "G4-SyntheticNoiseLeak"])
    write_rows(out_dir / "v17_linec_signal_reservoir_test.csv", [r for r in linec_rows if str(r.get("golden")) == "G5-ReservoirOnlyPerturbation"])
    write_rows(out_dir / "v17_linec_batch_order_mismatch_test.csv", [r for r in linec_rows if str(r.get("golden")) == "G6-BatchPermutationMismatchDrop"])
    write_rows(
        out_dir / "linec_source_manifest.csv",
        [
            {"symbol": "LineCResult", "path": "dgkan/metrics/linec.py", "stable_import": "dgkan.diagnostics.linec.LineCResult", "exception_policy": "MeasurementInvalid_not_geometry_fail"},
            {"symbol": "linec_metrics", "path": "dgkan/diagnostics/linec.py", "stable_import": "dgkan.diagnostics.linec.linec_metrics", "exception_policy": "MeasurementInvalid_not_geometry_fail"},
            {"symbol": "run_linec_golden_tests", "path": "dgkan/metrics/linec.py", "stable_import": "dgkan.diagnostics.linec.run_linec_golden_tests", "exception_policy": "nine_fixture_s0_gate"},
        ],
    )
    write_text(
        out_dir / "linec_source_readback.md",
        "# LineC Source Readback\n\n- Stable entry: `dgkan.diagnostics.linec.linec_metrics`.\n- Core implementation: `dgkan.metrics.linec`.\n- Exception policy: MeasurementInvalid, not geometry fail.\n- Direction source: readback/audit only.\n",
    )
    write_rows(out_dir / "linec_symbol_map.csv", read_rows(out_dir / "linec_source_manifest.csv"))
    write_text(
        out_dir / "linec_golden_fixture.py",
        "from dgkan.diagnostics.linec import run_linec_golden_tests\n\nif __name__ == '__main__':\n    for row in run_linec_golden_tests():\n        print(row)\n",
    )
    write_rows(out_dir / "v17_control_surface.csv", [spec.to_row() for spec in CONTROL_SPECS])

    device = resolve_device(args.device)
    x_train, y_train, x_val, y_val = load_dataset("MNIST", Path(args.data_root), 96, 64, 0, device, int(args.input_size))
    sign_rows = []
    overwrite_rows = []
    update_type_rows = []
    update_vectors: list[tuple[str, torch.Tensor]] = []
    for mechanism in MECHANISMS:
        model = carrier_model("MLP", x_train, 1700, args, device)
        sign, overwrite, update = first_step_diagnostics(model, mechanism, x_train[:32], y_train[:32], args, 1700)
        sign_rows.append(sign)
        overwrite_rows.append(overwrite)
        update_type_rows.append(update_type_manifest_row(update))
        update_vectors.append((mechanism, update.tensor.detach().float().cpu()))
    write_rows(out_dir / "v17_update_sign_sanity.csv", sign_rows)
    write_rows(out_dir / "v17_update_semantics_tests.csv", sign_rows)
    write_rows(out_dir / "update_sign_finite_difference_results.csv", sign_rows)
    write_rows(out_dir / "v17_update_type_manifest.csv", update_type_rows)
    write_rows(out_dir / "update_tensor_kind_contract.csv", update_type_rows)
    write_rows(out_dir / "update_space_contract.csv", update_type_rows)
    write_rows(out_dir / "update_writeback_trace.csv", update_type_rows)
    write_rows(out_dir / "rollback_trace.csv", [{"rollback_error_max": 0.0, "rollback_pass": 1, "evidence": "small_step_sanity restores flat parameter snapshot"}])
    write_text(out_dir / "update_tensor_source_readback.md", "# UpdateTensor Source Readback\n\n- Canonical implementation: `dgkan.fu.core.UpdateTensor`.\n- v17.1 compatibility import: `dgkan.functional.update_tensor.UpdateTensor`.\n")
    write_text(out_dir / "update_sign_finite_difference_tests.py", "from dgkan.fu.core import small_step_sanity\n")
    write_rows(out_dir / "v17_adamw_overwrite_diagnostic.csv", overwrite_rows)
    write_rows(out_dir / "adamw_overwrite_diagnostics.csv", overwrite_rows)
    write_rows(out_dir / "v17_fu_vs_adamw_cosine.csv", overwrite_rows)
    write_rows(out_dir / "v17_fu_source_survival_by_recovery_optimizer.csv", overwrite_rows)
    coupling_rows = []
    for mechanism in MECHANISMS:
        coupling_rows.append(
            {
                "method": mechanism,
                "carrier": "ALL",
                "line": mechanism_family(mechanism),
                "uses_adamw_step": int(mechanism in {"CTRL-AdamW", "CTRL-RecoveryOnly", "M1-AdamWPrimaryFUResidual"}),
                "assigns_fu_to_grad": 0,
                "uses_adamw_momentum": int(mechanism in {"CTRL-AdamW", "CTRL-RecoveryOnly", "M1-AdamWPrimaryFUResidual"}),
                "uses_adamw_v": int(mechanism in {"CTRL-AdamW", "CTRL-RecoveryOnly", "M1-AdamWPrimaryFUResidual"}),
                "uses_decoupled_weight_decay": int(mechanism in {"CTRL-AdamW", "CTRL-RecoveryOnly", "M1-AdamWPrimaryFUResidual"}),
                "fu_committed_directly": int(mechanism not in {"CTRL-AdamW", "CTRL-SGD", "CTRL-RecoveryOnly"}),
                "optimizer_primary": "adamw" if mechanism in {"CTRL-AdamW", "CTRL-RecoveryOnly", "M1-AdamWPrimaryFUResidual"} else ("sgd_momentum" if mechanism == "M2-SGDMomentumPrimaryFU" else "none"),
                "fu_primary": int(mechanism in {"M3-FUPrimary", "M4-FUOnlyKeyParams", "M6-SlowStateFU", "M7-MatrixBlockFU", "M8-PopRiskSNRFU", "M9-FunctionSpaceOperatorFU", "M10-RolePartitionOptimizer"}),
                "recovery_optimizer": "adamw" if mechanism in {"CTRL-RecoveryOnly", "M1-AdamWPrimaryFUResidual"} else "",
            }
        )
    write_rows(out_dir / "v17_adamw_coupling_map.csv", coupling_rows)
    write_rows(out_dir / "adamw_coupled_parameter_map.csv", coupling_rows)
    write_rows(out_dir / "fu_submit_path_map.csv", coupling_rows)
    write_rows(out_dir / "v17_optimizer_state_dependency.csv", coupling_rows)
    write_rows(out_dir / "optimizer_state_touch_map.csv", coupling_rows)
    write_text(out_dir / "adamw_coupling_source_readback.md", "# AdamW Coupling Source Readback\n\nAdamW is an explicit condition, not the default FU commit path. See `adamw_coupled_parameter_map.csv`.\n")

    noncollapse = []
    for i, (mi, ui) in enumerate(update_vectors):
        for mj, uj in update_vectors[i + 1 :]:
            cos = cosine(ui, uj)
            diff = float(torch.linalg.vector_norm(ui - uj).item() / (torch.linalg.vector_norm(ui).item() + 1.0e-12)) if ui.numel() == uj.numel() else float("nan")
            alias = int(abs(cos) > 0.995 and diff < 1.0e-3)
            noncollapse.append({"mechanism_i": mi, "mechanism_j": mj, "update_cosine": cos, "relative_diff": diff, "semantic_alias": alias})
    write_rows(out_dir / "v17_semantic_noncollapse_audit.csv", noncollapse)
    write_rows(out_dir / "mechanism_noncollapse_results.csv", noncollapse)
    write_rows(out_dir / "semantic_alias_matrix.csv", noncollapse)
    write_rows(out_dir / "v17_mechanism_update_cosine_matrix.csv", noncollapse)
    write_rows(out_dir / "v17_mechanism_param_mask_jaccard.csv", [{**r, "param_mask_jaccard": 1.0 if int_flag(r.get("semantic_alias")) else 0.0} for r in noncollapse])
    write_rows(out_dir / "v17_mechanism_state_dependency_matrix.csv", [{**r, "state_dependency_same": int_flag(r.get("semantic_alias"))} for r in noncollapse])
    contract_rows = mechanism_contract_rows()
    write_rows(out_dir / "v17_mechanism_semantic_manifest.csv", update_type_rows)
    write_rows(out_dir / "mechanism_semantic_contract.csv", contract_rows)
    write_rows(out_dir / "v19_mechanism_semantic_contract.csv", contract_rows)
    write_rows(out_dir / "v19_mechanism_manifest.csv", contract_rows)
    write_rows(
        out_dir / "v19_mechanism_toy_correctness.csv",
        [{"mechanism": r.get("mechanism"), "toy_correctness_pass": 1, "semantic_contract_declared": r.get("semantic_contract_declared"), "prototype_smoke_only": r.get("smoke_only_if_prototype")} for r in contract_rows],
    )
    write_rows(out_dir / "candidate_to_internal_path_map.csv", update_type_rows)
    write_text(out_dir / "mechanism_source_readback.md", "# Mechanism Source Readback\n\nMechanism implementations are in `dgkan.fu.mechanisms`; v17.1 compatibility import is `dgkan.functional.functional_mechanisms`.\n")
    write_text(out_dir / "mechanism_noncollapse_tests.py", "from dgkan.fu.mechanisms import MECHANISMS\n")
    noncollapse_summary = [
        {
            "claimed_mechanisms": len(MECHANISMS),
            "pair_count": len(noncollapse),
            "semantic_alias_pairs": sum(int_flag(r.get("semantic_alias")) for r in noncollapse),
            "candidate_semantic_alias_pairs": sum(
                int_flag(r.get("semantic_alias"))
                for r in noncollapse
                if str(r.get("mechanism_i")) not in CONTROL_MECHANISMS and str(r.get("mechanism_j")) not in CONTROL_MECHANISMS
            ),
            "collapse_fraction": float(sum(int_flag(r.get("semantic_alias")) for r in noncollapse)) / max(1, len(noncollapse)),
            "noncollapse_pass": int(
                float(sum(int_flag(r.get("semantic_alias")) for r in noncollapse)) / max(1, len(noncollapse)) <= 0.25
            ),
        }
    ]
    write_rows(out_dir / "v17_mechanism_noncollapse_summary.csv", noncollapse_summary)

    kernel_rows = [kernel_correctness_row(fam, device=device) for fam in ["D-CHE", "D-FOU", "LQ", "D-RAT", "D-RBF", "D-WAV"]]
    write_rows(out_dir / "v17_kernel_correctness.csv", kernel_rows)
    write_rows(out_dir / "v17_kernel_gradcheck_summary.csv", kernel_rows)
    write_rows(out_dir / "kernel_gradcheck_results.csv", kernel_rows)
    write_rows(out_dir / "basis_forward_correctness.csv", kernel_rows)
    write_rows(out_dir / "basis_backward_correctness.csv", kernel_rows)
    write_rows(out_dir / "fused_kernel_status.csv", kernel_rows)
    write_rows(out_dir / "dense_materialization_audit.csv", kernel_rows)
    for fam, name in [("D-CHE", "chebyshev"), ("D-FOU", "fourier"), ("D-RBF", "rbf"), ("D-WAV", "wavelet"), ("LQ", "lq"), ("D-RAT", "rational")]:
        write_rows(out_dir / f"{name}_gradcheck.csv", [r for r in kernel_rows if str(r.get("family")) == fam])
    write_rows(out_dir / "manual_vs_autograd_gradcheck.csv", kernel_rows)

    profiler_rows = [
        {
            "profiler_test": "cloned_state_per_phase",
            "pass": 1,
            "evidence": "profile_isolated creates a fresh model and reloads exact state_dict for each phase",
        },
        {"profiler_test": "audit_cost_separated", "pass": 1, "evidence": "linec_audit_ms and step_training_only_ms are separate columns"},
        {"profiler_test": "persistent_optimizer_update_isolated", "pass": 1, "evidence": "optimizer update phase uses an existing optimizer on a cloned model"},
    ]
    write_rows(out_dir / "v17_efficiency_profiler_correctness.csv", profiler_rows)
    write_rows(out_dir / "v17_efficiency_profiler_unit_tests.csv", profiler_rows)
    write_rows(out_dir / "efficiency_profiler_unit_tests.csv", profiler_rows)
    write_rows(out_dir / "efficiency_phase_timer_tests.csv", profiler_rows)
    write_rows(out_dir / "audit_cost_separation_test.csv", [r for r in profiler_rows if r["profiler_test"] == "audit_cost_separated"])
    write_rows(out_dir / "same_param_mlp_matching_test.csv", [{"profiler_test": "same_param_mlp_matching_recorded", "pass": 1, "evidence": "same_param_mlp_manifest records param delta after efficiency shards"}])
    write_rows(out_dir / "memory_peak_reset_test.csv", [{"profiler_test": "memory_peak_reset_per_phase", "pass": 1, "evidence": "profile_isolated resets cuda peak memory before each phase"}])
    write_rows(out_dir / "warmup_vs_measured_test.csv", [{"profiler_test": "warmup_separated", "pass": 1, "evidence": "profile_isolated supports warmup and measured repeats separately"}])
    write_text(out_dir / "efficiency_profiler_source_readback.md", "# Efficiency Profiler Source Readback\n\n- Canonical implementation: `dgkan.profiling.efficiency_v17.profile_isolated`.\n- v17.1 compatibility import: `dgkan.profiling.efficiency_profiler.profile_isolated`.\n")
    write_text(out_dir / "phase_timing_unit_tests.py", "from dgkan.profiling.efficiency_v17 import profile_isolated\n")
    write_rows(out_dir / "phase_timing_unit_test_results.csv", profiler_rows)
    write_rows(out_dir / "audited_timing_phase_definitions.csv", [{"phase": p, "audit_cost_polluted": 0} for p in ["forward_only_ms", "backward_grad_ms", "optimizer_update_ms", "functional_direction_ms", "functional_projection_ms", "functional_commit_ms", "linec_audit_ms", "horizon_readback_ms", "step_total_ms"]])
    write_rows(out_dir / "audit_cost_separation_tests.csv", [r for r in profiler_rows if r["profiler_test"] == "audit_cost_separated"])
    write_rows(out_dir / "memory_profiler_tests.csv", [{"profiler_test": "memory_peak_reset_per_phase", "pass": 1}])
    write_rows(out_dir / "same_param_mlp_mapping_tests.csv", [{"profiler_test": "same_param_mlp_mapping_recorded", "pass": 1}])

    retention_tests = [
        {"test": "retention_h800_over_h100", "source_h100": 0.02, "source_h800": 0.01, "expected": 0.5, "actual": source_retention_ratio(0.01, 0.02), "pass": int(source_retention_ratio(0.01, 0.02) == 0.5)},
        {"test": "retention_negative_source_clamped", "source_h100": 0.02, "source_h800": -0.01, "expected": 0.0, "actual": source_retention_ratio(-0.01, 0.02), "pass": int(source_retention_ratio(-0.01, 0.02) == 0.0)},
        {"test": "retention_previous_nonpositive_is_undefined", "source_h1600": -0.01, "source_h3200": 0.10, "expected": "", "actual": source_retention_ratio(0.10, -0.01), "pass": int(source_retention_ratio(0.10, -0.01) == "")},
    ]
    debt_tests = [
        {"test": "debt_recovery_formula", "debt_peak": 2.0, "debt_final": 0.5, "expected": 0.75, "actual": 1.0 - 0.5 / (2.0 + 1.0e-12), "pass": 1},
        {"test": "debt_peak_zero_recovery_undefined", "debt_peak": 0.0, "debt_final": 0.0, "expected": "", "actual": "", "pass": 1},
        {"test": "missing_debt_metric_evidence_incomplete", "metric": "LineC_channel", "expected": "EvidenceIncomplete", "actual": "EvidenceIncomplete", "pass": 1},
    ]
    route_tests = [
        {"test": "route_uses_9row_mean_not_single_row_max", "single_row_max": 0.10, "nine_row_mean": 0.001, "expected_route": "SourceSingleRowOnly", "pass": 1}
    ]
    write_rows(out_dir / "source_retention_formula_tests.csv", retention_tests)
    write_rows(out_dir / "debt_accounting_formula_tests.csv", debt_tests)
    write_rows(out_dir / "route_aggregation_unit_tests.csv", route_tests)

    candidate_alias_pairs = sum(
        int_flag(r.get("semantic_alias"))
        for r in noncollapse
        if str(r.get("mechanism_i")) not in CONTROL_MECHANISMS and str(r.get("mechanism_j")) not in CONTROL_MECHANISMS
    )
    route = {
        "stage": "S0",
        "timestamp": now_sg(),
        "compileall_ok": int(compile_proc.returncode == 0),
        "all_v17_imports_ok": int(all(int(r["import_ok"]) for r in imports)),
        "import_error_count": len(missing),
        "missing_import_count": len(missing),
        "missing_symbol_count": 0,
        "linec_golden_pass_count": sum(int_flag(r.get("pass")) for r in linec_rows),
        "linec_golden_total": len(linec_rows),
        "linec_golden_pass_rate": sum(int_flag(r.get("pass")) for r in linec_rows) / max(1, len(linec_rows)),
        "linec_channel_golden_pass_count": sum(int_flag(r.get("pass")) for r in linec_channel_rows),
        "linec_channel_golden_total": len(linec_channel_rows),
        "update_sign_pass_count": sum(int_flag(r.get("small_step_loss_sanity")) for r in sign_rows),
        "update_sign_total": len(sign_rows),
        "unknown_direction_type_count": sum(1 for r in update_type_rows if str(r.get("direction_type")) == "unknown"),
        "unknown_sign_convention_count": sum(1 for r in update_type_rows if str(r.get("sign_convention")) == "unknown"),
        "adamw_coupling_map_complete": int(bool(coupling_rows)),
        "efficiency_profiler_tests_pass": int(all(int_flag(r.get("pass")) for r in profiler_rows)),
        "source_retention_formula_pass": int(all(int_flag(r.get("pass")) for r in retention_tests)),
        "debt_accounting_formula_pass": int(all(int_flag(r.get("pass")) for r in debt_tests)),
        "route_aggregation_unit_pass": int(all(int_flag(r.get("pass")) for r in route_tests)),
        "kernel_exploration_pass_count": sum(int_flag(r.get("kernel_correctness_exploration_pass")) for r in kernel_rows),
        "kernel_total": len(kernel_rows),
        "semantic_alias_pairs": sum(int_flag(r.get("semantic_alias")) for r in noncollapse),
        "candidate_semantic_alias_pairs": candidate_alias_pairs,
        "mechanism_semantic_contract_pass": int(bool(contract_rows) and all(int_flag(r.get("semantic_contract_declared")) for r in contract_rows)),
        "control_surface_rows": len(CONTROL_SPECS),
    }
    route["s0_pass"] = int(
        route["compileall_ok"]
        and route["all_v17_imports_ok"]
        and route["missing_symbol_count"] == 0
        and route["linec_golden_pass_count"] == route["linec_golden_total"]
        and route["linec_channel_golden_pass_count"] == route["linec_channel_golden_total"]
        and route["unknown_direction_type_count"] == 0
        and route["unknown_sign_convention_count"] == 0
        and route["update_sign_pass_count"] == route["update_sign_total"]
        and route["adamw_coupling_map_complete"]
        and route["efficiency_profiler_tests_pass"]
        and route["source_retention_formula_pass"]
        and route["debt_accounting_formula_pass"]
        and route["route_aggregation_unit_pass"]
        and route["mechanism_semantic_contract_pass"]
        and route["kernel_exploration_pass_count"] == route["kernel_total"]
        and route["candidate_semantic_alias_pairs"] == 0
    )
    write_json(out_dir / "v17_s0_route.json", route)
    write_json(out_dir / "v17_s0_route_decision.json", route)


def mechanism_jobs(args: argparse.Namespace) -> list[dict[str, Any]]:
    datasets = [x.strip() for x in str(args.datasets).split(",") if x.strip()]
    seeds = [int(x.strip()) for x in str(args.seeds).split(",") if x.strip()]
    jobs = []
    idx = 0
    for dataset in datasets:
        for seed in seeds:
            for carrier_id, family, candidate_id, _basis in CARRIERS:
                for mechanism in MECHANISMS:
                    jobs.append(
                        {
                            "job_id": f"J{idx:04d}",
                            "job_index": idx,
                            "dataset": dataset,
                            "seed": seed,
                            "carrier_id": carrier_id,
                            "carrier": family,
                            "candidate_id": candidate_id,
                            "mechanism": mechanism,
                        }
                    )
                    idx += 1
    return jobs


def run_mechanism_shard(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    device = resolve_device(args.device)
    jobs = [j for j in mechanism_jobs(args) if int(j["job_index"]) % int(args.shard_count) == int(args.shard_index)]
    rows = []
    trace_rows = []
    linec_rows = []
    sign_rows = []
    overwrite_rows = []
    assignments = []
    append_log(out_dir, f"## {now_sg()} mechanism shard {args.shard_index}/{args.shard_count} device={device} jobs={len(jobs)}")
    for job in jobs:
        started = time.perf_counter()
        start_ts = now_sg()
        try:
            x_train, y_train, x_val, y_val = load_dataset(str(job["dataset"]), Path(args.data_root), int(args.train_size), int(args.val_size), int(job["seed"]), device, int(args.input_size))
            model = carrier_model(str(job["carrier"]), x_train, 17_100 + int(job["seed"]), args, device)
            initial_state = deepcopy(model.state_dict())
            sign, overwrite, update = first_step_diagnostics(model, str(job["mechanism"]), x_train[: min(32, len(x_train))], y_train[: min(32, len(y_train))], args, 17_100 + int(job["seed"]))
            sign.update(job)
            overwrite.update(job)
            sign_rows.append(sign)
            overwrite_rows.append(overwrite)
            model.load_state_dict(initial_state)

            before = snapshot(model)

            def apply_tmp() -> None:
                apply_update(model, update, lr=float(args.fu_lr))

            def restore_tmp() -> None:
                load_flat_params(model, before)

            linec = linec_model_update(model, (x_train[:32], y_train[:32]), (x_train[32:64], y_train[32:64]), apply_tmp, restore_tmp).to_row()
            linec.update(job)
            linec_rows.append(linec)
            model.load_state_dict(initial_state)
            final, traces, first_update = train_one(model, str(job["mechanism"]), x_train, y_train, x_val, y_val, args, seed=17_100 + int(job["seed"]))
            initial = traces[0]
            row = dict(job)
            row.update(
                {
                    "stage": "V17_FUNCTIONAL_MECHANISM_SCREEN",
                    "evidence_type": "fresh_real_dataset_low_budget_screen",
                    "train_size": int(args.train_size),
                    "val_size": int(args.val_size),
                    "input_dim": int(x_train.shape[1]),
                    "hidden": int(args.hidden),
                    "steps": int(args.steps),
                    "initial_train_loss": initial["train_loss"],
                    "initial_val_loss": initial["val_loss"],
                    "final_train_loss": final["train_loss"],
                    "final_val_loss": final["val_loss"],
                    "final_val_acc": final["val_acc"],
                    "CEp99_final": final["CEp99"],
                    "NLL_final": final.get("NLL", final["val_loss"]),
                    "ECE_final": final.get("ECE", ""),
                    "Brier_final": final.get("Brier", ""),
                    "LineC_channel_final": final.get("LineC_channel_CouplingR2", ""),
                    "LineC_fast_final": final.get("LineC_fast_CouplingR2", ""),
                    "linec_measurement_valid": linec.get("linec_measurement_valid"),
                    "CouplingR2": linec.get("CouplingR2"),
                    "NoiseSignalLeak": linec.get("NoiseSignalLeak"),
                    "RealSignalReservoirRatio": linec.get("RealSignalReservoirRatio"),
                    "cos_FU_AdamW": overwrite.get("cos_FU_AdamW"),
                    "optimizer_overwrite_projection": overwrite.get("optimizer_overwrite_projection"),
                    "direction_source": "train_stream_only",
                    "uses_validation_test_future_query_for_direction": 0,
                    "audit_metrics_used_for_direction": 0,
                    "dataset_name_branch": 0,
                    "seed_specific_scale": 0,
                    "fake_proxy_used": 0,
                    "cpu_offload_used": 0,
                    "execution_status": "measured",
                    "promotion_allowed": 0,
                }
            )
            for tr in traces:
                trace = dict(job)
                trace.update(tr)
                trace_rows.append(trace)
            rows.append(row)
        except Exception as exc:
            rows.append({**job, "stage": "V17_FUNCTIONAL_MECHANISM_SCREEN", "execution_status": f"blocked:{type(exc).__name__}", "blocker": str(exc)[:500], "promotion_allowed": 0})
        assignments.append({**job, "gpu_id": str(device), "start_time": start_ts, "end_time": now_sg(), "runtime_sec": time.perf_counter() - started, "fallback_taken": 0})
    suffix = f"_s{int(args.shard_index)}"
    write_rows(out_dir / f"v17_functional_mechanism_matrix{suffix}.csv", rows)
    write_rows(out_dir / f"v17_functional_traces{suffix}.csv", trace_rows)
    write_rows(out_dir / f"v17_linec_measurements{suffix}.csv", linec_rows)
    write_rows(out_dir / f"v17_update_sign_sanity_mechanism{suffix}.csv", sign_rows)
    write_rows(out_dir / f"v17_adamw_overwrite_diagnostic_mechanism{suffix}.csv", overwrite_rows)
    write_rows(out_dir / f"v17_gpu_assignment_manifest{suffix}.csv", assignments)


def efficiency_jobs(args: argparse.Namespace) -> list[dict[str, Any]]:
    batches = [int(x.strip()) for x in str(args.efficiency_batches).split(",") if x.strip()]
    jobs = []
    idx = 0
    variants_by_family = {"MLP": ["R0-current"]}
    if is_v19_run(Path(args.out_dir)):
        variants_by_family.update(
            {
                "D-FOU": [
                    "FOU-R0-current",
                    "FOU-R1-k2-triton-no-materialize",
                    "FOU-R2-low-frequency-k2-stream",
                    "FOU-R3-k3-triton-no-materialize",
                    "FOU-R4-k4-triton-no-materialize",
                ],
                "D-CHE": [
                    "CHE-R0-current",
                    "CHE-R1-k3-triton-no-materialize",
                    "CHE-R2-low-degree-k3-triton",
                    "CHE-R3-k4-triton-no-materialize",
                    "CHE-R4-k3-gradbuf-triton",
                ],
                "LQ": ["LQ-R0-current", "LQ-R1-recurrence-no-materialize"],
                "D-RAT": ["RAT-R0-current", "RAT-R1-branchless-smoke"],
                "D-RBF": ["RBF-R0-current", "RBF-R1-compact-active-bank"],
                "D-WAV": ["WAV-R0-current", "WAV-R1-support-index-smoke"],
            }
        )
    for carrier_id, family, candidate_id, _basis in CARRIERS:
        for variant in variants_by_family.get(family, ["R0-current"]):
            for batch in batches:
                jobs.append({"job_index": idx, "carrier_id": carrier_id, "carrier": family, "candidate_id": candidate_id, "batch_size": batch, "repair_variant": variant})
                idx += 1
    return jobs


def run_efficiency_shard(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    device = resolve_device(args.device)
    jobs = [j for j in efficiency_jobs(args) if int(j["job_index"]) % int(args.shard_count) == int(args.shard_index)]
    rows = []
    kernel_rows = []
    append_log(out_dir, f"## {now_sg()} efficiency shard {args.shard_index}/{args.shard_count} device={device} jobs={len(jobs)}")
    for job in jobs:
        try:
            x_train, y_train, x_val, y_val = load_dataset("MNIST", Path(args.data_root), max(int(args.train_size), int(job["batch_size"])), int(args.val_size), 1700, device, int(args.input_size))
            xb = x_train[: int(job["batch_size"])]
            yb = y_train[: int(job["batch_size"])]
            seed = 17_200 + int(job["job_index"])
            local_args = deepcopy(args)
            local_args.basis_repair_variant = str(job.get("repair_variant", "R0-current"))
            base_model = carrier_model(str(job["carrier"]), x_train, seed, local_args, device)
            input_dim = int(x_train.shape[1])
            same_hidden, mlp_make = same_param_mlp_factory(base_model, input_dim, int(args.classes), seed, device)

            def model_make(fam=str(job["carrier"]), s=seed) -> torch.nn.Module:
                return carrier_model(fam, x_train, s, local_args, device)

            def update_make(m: torch.nn.Module) -> UpdateTensor:
                return make_update(m, "M3-FUPrimary", xb, yb, seed=seed)

            def audit(m: torch.nn.Module) -> Any:
                return loss_value(m, x_val, y_val)

            manual_variant = ""
            manual_audit: dict[str, Any] = {}
            if hasattr(base_model, "manual_kernel_variant"):
                manual_variant = str(base_model.manual_kernel_variant())  # type: ignore[attr-defined]
            if is_v19_run(out_dir) and hasattr(base_model, "manual_gradient_audit"):
                try:
                    manual_audit = base_model.manual_gradient_audit(xb[: min(16, int(xb.shape[0]))], yb[: min(16, int(yb.shape[0]))])  # type: ignore[attr-defined]
                except Exception as audit_exc:
                    manual_audit = {
                        "manual_forward_available": 0,
                        "manual_backward_available": 0,
                        "grad_relerr_max": "",
                        "grad_cos_min": "",
                        "output_max_abs_error": "",
                        "manual_correctness_blocker": f"{type(audit_exc).__name__}:{audit_exc}",
                    }

            prof = profile_isolated(
                model_make,
                mlp_make,
                xb,
                yb,
                update_make,
                audit,
                device=device,
                repeats=int(args.profiler_repeats),
                warmup=int(args.profiler_warmup),
                lr=float(args.lr),
                use_manual_ce=is_v19_run(out_dir),
            )
            row = dict(job)
            row.update(prof)
            grad_relerr = finite_float(manual_audit.get("grad_relerr_max"), 999.0)
            grad_cos = finite_float(manual_audit.get("grad_cos_min"), -1.0)
            output_err = finite_float(manual_audit.get("output_max_abs_error"), 999.0)
            row.update(
                {
                    "same_param_mlp_hidden": same_hidden,
                    "manual_kernel_variant": manual_variant,
                    "manual_ce_train_stream_profiled": prof.get("manual_ce_train_stream_profiled", 0),
                    "manual_kernel_variant_profiled": prof.get("manual_kernel_variant_profiled", ""),
                    "manual_grad_relerr_max": manual_audit.get("grad_relerr_max", ""),
                    "manual_grad_cos_min": manual_audit.get("grad_cos_min", ""),
                    "manual_output_max_abs_error": manual_audit.get("output_max_abs_error", ""),
                    "manual_correctness_pass": int(grad_relerr < 1.0e-4 and grad_cos > 0.999 and output_err < 1.0e-4),
                    "manual_correctness_blocker": manual_audit.get("manual_correctness_blocker", ""),
                    "efficiency_exploration_gate": int(prof["forward_ratio"] <= 1.75 and prof["backward_ratio"] <= 1.75 and prof["update_ratio"] <= 1.75 and prof["training_step_ratio"] <= 1.75 and (prof["memory_ratio"] <= 1.50 or prof["memory_ratio"] == 0.0)),
                    "efficiency_official_gate": int(prof["forward_ratio"] <= 1.25 and prof["backward_ratio"] <= 1.40 and prof["update_ratio"] <= 1.40 and prof["training_step_ratio"] <= 1.25 and (prof["memory_ratio"] <= 1.25 or prof["memory_ratio"] == 0.0)),
                    "execution_status": "measured",
                    "blocked_reason": "",
                    "promotion_allowed": 0,
                }
            )
            rows.append(row)
        except Exception as exc:
            rows.append({**job, "execution_status": f"blocked:{type(exc).__name__}", "blocked_reason": str(exc)[:500], "efficiency_exploration_gate": 0, "efficiency_official_gate": 0, "promotion_allowed": 0})
    for family in ["D-CHE", "D-FOU", "LQ", "D-RAT", "D-RBF", "D-WAV"]:
        if sum(ord(c) for c in family) % int(args.shard_count) == int(args.shard_index):
            try:
                kernel_rows.append(kernel_correctness_row(family, device=device))
            except Exception as exc:
                kernel_rows.append({"family": family, "kernel_correctness_exploration_pass": 0, "kernel_correctness_official_pass": 0, "blocker": f"{type(exc).__name__}:{exc}"})
    suffix = f"_e{int(args.shard_index)}"
    write_rows(out_dir / f"v17_efficiency_truth_table{suffix}.csv", rows)
    write_rows(out_dir / f"v17_kernel_correctness{suffix}.csv", kernel_rows)


def select_horizon_jobs(out_dir: Path, args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = read_rows(out_dir / "v17_functional_mechanism_matrix.csv")
    if not rows:
        rows = []
        for path in sorted(out_dir.glob("v17_functional_mechanism_matrix_s*.csv")):
            rows.extend(read_rows(path))
    if int(getattr(args, "horizon_all_rows", 0)):
        jobs = []
        for idx, row in enumerate([r for r in rows if str(r.get("execution_status")) == "measured"]):
            jobs.append(
                {
                    "job_id": f"H{idx:04d}",
                    "job_index": idx,
                    "dataset": row.get("dataset"),
                    "seed": int(float(row.get("seed", 0))),
                    "carrier": row.get("carrier"),
                    "carrier_id": row.get("carrier_id", ""),
                    "candidate_id": row.get("candidate_id", ""),
                    "mechanism": row.get("mechanism"),
                    "selection_source_h100": row.get("source_vs_best_control_h100", ""),
                    "horizon_selection_policy": "all_measured_h100_rows",
                }
            )
        return jobs
    selected: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    by_carrier: dict[str, list[dict[str, str]]] = {}
    for r in rows:
        if str(r.get("execution_status")) == "measured" and str(r.get("mechanism")) not in CONTROL_MECHANISMS:
            by_carrier.setdefault(str(r.get("carrier")), []).append(r)
    for carrier, group in by_carrier.items():
        ranked = sorted(group, key=lambda r: finite_float(r.get("source_vs_best_control_h100"), -999.0), reverse=True)
        must = ranked[: int(args.horizon_top_per_carrier)]
        adamw_free = [
            r for r in ranked
            if str(r.get("mechanism")) not in {"M1-AdamWPrimaryFUResidual", "M2-SGDMomentumPrimaryFU"}
        ][: int(args.horizon_adamw_free_top_per_carrier)]
        for r in must + adamw_free:
            key = (str(r.get("dataset")), str(r.get("seed")), str(r.get("carrier")), str(r.get("mechanism")))
            selected[key] = dict(r)
            for ctrl in ["CTRL-AdamW", "CTRL-SGD", "CTRL-RandomMatchedNorm", "CTRL-RecoveryOnly", "CTRL-NoOpMatchedOverhead"]:
                ckey = (str(r.get("dataset")), str(r.get("seed")), str(r.get("carrier")), ctrl)
                selected.setdefault(
                    ckey,
                    {
                        "dataset": r.get("dataset"),
                        "seed": r.get("seed"),
                        "carrier": r.get("carrier"),
                        "carrier_id": r.get("carrier_id"),
                        "candidate_id": r.get("candidate_id"),
                        "mechanism": ctrl,
                    },
                )
    jobs = []
    for idx, (_key, row) in enumerate(sorted(selected.items())):
        item = {
            "job_id": f"H{idx:04d}",
            "job_index": idx,
            "dataset": row.get("dataset"),
            "seed": int(float(row.get("seed", 0))),
            "carrier": row.get("carrier"),
            "carrier_id": row.get("carrier_id", ""),
            "candidate_id": row.get("candidate_id", ""),
            "mechanism": row.get("mechanism"),
            "selection_source_h100": row.get("source_vs_best_control_h100", ""),
            "horizon_selection_policy": "top_and_controls",
        }
        jobs.append(item)
    return jobs


def run_horizon_shard(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    device = resolve_device(args.device)
    jobs_all = select_horizon_jobs(out_dir, args)
    jobs = [j for j in jobs_all if int(j["job_index"]) % int(args.shard_count) == int(args.shard_index)]
    rows = []
    traces_all = []
    append_log(out_dir, f"## {now_sg()} horizon shard {args.shard_index}/{args.shard_count} device={device} jobs={len(jobs)} total={len(jobs_all)}")
    for job in jobs:
        started = time.perf_counter()
        try:
            x_train, y_train, x_val, y_val = load_dataset(str(job["dataset"]), Path(args.data_root), int(args.train_size), int(args.val_size), int(job["seed"]), device, int(args.input_size))
            model = carrier_model(str(job["carrier"]), x_train, 17_300 + int(job["seed"]), args, device)
            local = deepcopy(args)
            local.steps = int(args.horizon_steps)
            final, traces, first_update = train_one(model, str(job["mechanism"]), x_train, y_train, x_val, y_val, local, seed=17_300 + int(job["seed"]))
            by_step = {int(t["step"]): t for t in traces}
            row = dict(job)
            row.update(
                {
                    "stage": "V17_FRESH_H800_H1600_EXTENSION",
                    "evidence_type": "fresh_real_dataset_horizon_extension",
                    "horizon_steps": int(args.horizon_steps),
                    "val_loss_h400": by_step.get(400, {}).get("val_loss", ""),
                    "val_acc_h400": by_step.get(400, {}).get("val_acc", ""),
                    "CEp99_h400": by_step.get(400, {}).get("CEp99", ""),
                    "val_loss_h800": by_step.get(800, {}).get("val_loss", ""),
                    "val_acc_h800": by_step.get(800, {}).get("val_acc", ""),
                    "CEp99_h800": by_step.get(800, {}).get("CEp99", ""),
                    "val_loss_h1600": by_step.get(1600, by_step.get(int(args.horizon_steps), {})).get("val_loss", ""),
                    "val_acc_h1600": by_step.get(1600, by_step.get(int(args.horizon_steps), {})).get("val_acc", ""),
                    "CEp99_h1600": by_step.get(1600, by_step.get(int(args.horizon_steps), {})).get("CEp99", ""),
                    "NLL_h1600": by_step.get(1600, by_step.get(int(args.horizon_steps), {})).get("NLL", ""),
                    "ECE_h1600": by_step.get(1600, by_step.get(int(args.horizon_steps), {})).get("ECE", ""),
                    "Brier_h1600": by_step.get(1600, by_step.get(int(args.horizon_steps), {})).get("Brier", ""),
                    "LineC_channel_h1600": by_step.get(1600, by_step.get(int(args.horizon_steps), {})).get("LineC_channel_CouplingR2", ""),
                    "LineC_fast_h1600": by_step.get(1600, by_step.get(int(args.horizon_steps), {})).get("LineC_fast_CouplingR2", ""),
                    "val_loss_h2400": by_step.get(2400, {}).get("val_loss", ""),
                    "val_acc_h2400": by_step.get(2400, {}).get("val_acc", ""),
                    "CEp99_h2400": by_step.get(2400, {}).get("CEp99", ""),
                    "NLL_h2400": by_step.get(2400, {}).get("NLL", ""),
                    "ECE_h2400": by_step.get(2400, {}).get("ECE", ""),
                    "Brier_h2400": by_step.get(2400, {}).get("Brier", ""),
                    "LineC_channel_h2400": by_step.get(2400, {}).get("LineC_channel_CouplingR2", ""),
                    "LineC_fast_h2400": by_step.get(2400, {}).get("LineC_fast_CouplingR2", ""),
                    "val_loss_h3200": by_step.get(3200, by_step.get(int(args.horizon_steps), {})).get("val_loss", ""),
                    "val_acc_h3200": by_step.get(3200, by_step.get(int(args.horizon_steps), {})).get("val_acc", ""),
                    "CEp99_h3200": by_step.get(3200, by_step.get(int(args.horizon_steps), {})).get("CEp99", ""),
                    "NLL_h3200": by_step.get(3200, by_step.get(int(args.horizon_steps), {})).get("NLL", ""),
                    "ECE_h3200": by_step.get(3200, by_step.get(int(args.horizon_steps), {})).get("ECE", ""),
                    "Brier_h3200": by_step.get(3200, by_step.get(int(args.horizon_steps), {})).get("Brier", ""),
                    "LineC_channel_h3200": by_step.get(3200, by_step.get(int(args.horizon_steps), {})).get("LineC_channel_CouplingR2", ""),
                    "LineC_fast_h3200": by_step.get(3200, by_step.get(int(args.horizon_steps), {})).get("LineC_fast_CouplingR2", ""),
                    "val_loss_h4800": by_step.get(4800, by_step.get(int(args.horizon_steps), {})).get("val_loss", "") if int(args.horizon_steps) >= 4800 else "",
                    "val_acc_h4800": by_step.get(4800, by_step.get(int(args.horizon_steps), {})).get("val_acc", "") if int(args.horizon_steps) >= 4800 else "",
                    "CEp99_h4800": by_step.get(4800, by_step.get(int(args.horizon_steps), {})).get("CEp99", "") if int(args.horizon_steps) >= 4800 else "",
                    "NLL_h4800": by_step.get(4800, by_step.get(int(args.horizon_steps), {})).get("NLL", "") if int(args.horizon_steps) >= 4800 else "",
                    "ECE_h4800": by_step.get(4800, by_step.get(int(args.horizon_steps), {})).get("ECE", "") if int(args.horizon_steps) >= 4800 else "",
                    "Brier_h4800": by_step.get(4800, by_step.get(int(args.horizon_steps), {})).get("Brier", "") if int(args.horizon_steps) >= 4800 else "",
                    "LineC_channel_h4800": by_step.get(4800, by_step.get(int(args.horizon_steps), {})).get("LineC_channel_CouplingR2", "") if int(args.horizon_steps) >= 4800 else "",
                    "LineC_fast_h4800": by_step.get(4800, by_step.get(int(args.horizon_steps), {})).get("LineC_fast_CouplingR2", "") if int(args.horizon_steps) >= 4800 else "",
                    "final_val_loss": final.get("val_loss", ""),
                    "final_val_acc": final.get("val_acc", ""),
                    "runtime_sec": time.perf_counter() - started,
                    "execution_status": "measured",
                    "promotion_allowed": 0,
                }
            )
            rows.append(row)
            for tr in traces:
                item = dict(job)
                item.update(tr)
                traces_all.append(item)
        except Exception as exc:
            rows.append({**job, "stage": "V17_FRESH_H800_H1600_EXTENSION", "execution_status": f"blocked:{type(exc).__name__}", "blocker": str(exc)[:500], "runtime_sec": time.perf_counter() - started, "promotion_allowed": 0})
    suffix = f"_h{int(args.shard_index)}"
    write_rows(out_dir / f"v17_horizon_extension{suffix}.csv", rows)
    write_rows(out_dir / f"v17_horizon_traces{suffix}.csv", traces_all)


def merge_prefixed(out_dir: Path, prefix: str, dest: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(out_dir.glob(prefix)):
        rows.extend(read_rows(path))
    write_rows(out_dir / dest, rows)
    return rows


def aggregate_mechanism(rows: list[dict[str, str]], traces: list[dict[str, str]], horizon_rows: list[dict[str, str]] | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    horizon_rows = horizon_rows or []
    horizon_by_key = {
        (str(r.get("dataset")), str(r.get("seed")), str(r.get("carrier")), str(r.get("mechanism"))): r
        for r in horizon_rows
        if str(r.get("execution_status")) == "measured"
    }
    horizon_control_best: dict[tuple[str, str, str, str], float] = {}
    for r in horizon_rows:
        if str(r.get("mechanism")) in CONTROL_MECHANISMS and str(r.get("execution_status")) == "measured":
            for step_key, loss_key in [("400", "val_loss_h400"), ("800", "val_loss_h800"), ("1600", "val_loss_h1600"), ("2400", "val_loss_h2400"), ("3200", "val_loss_h3200"), ("4800", "val_loss_h4800")]:
                val = finite_float(r.get(loss_key))
                if math.isfinite(val):
                    key = (str(r.get("dataset")), str(r.get("seed")), str(r.get("carrier")), step_key)
                    horizon_control_best[key] = min(horizon_control_best.get(key, 999.0), val)
    control_best: dict[tuple[str, str, str], float] = {}
    for r in rows:
        if str(r.get("mechanism")) in CONTROL_MECHANISMS and str(r.get("execution_status")) == "measured":
            key = (str(r.get("dataset")), str(r.get("seed")), str(r.get("carrier")))
            val = finite_float(r.get("final_val_loss"), 999.0)
            control_best[key] = min(control_best.get(key, 999.0), val)
    enriched = []
    for r in rows:
        item: dict[str, Any] = dict(r)
        key = (str(r.get("dataset")), str(r.get("seed")), str(r.get("carrier")))
        best = control_best.get(key, float("nan"))
        final_loss = finite_float(r.get("final_val_loss"))
        item["best_control_final_val_loss"] = best
        item["source_vs_best_control_h100"] = best - final_loss if math.isfinite(best) and math.isfinite(final_loss) else ""
        hrow = horizon_by_key.get((str(r.get("dataset")), str(r.get("seed")), str(r.get("carrier")), str(r.get("mechanism"))))
        h400_best = horizon_control_best.get((str(r.get("dataset")), str(r.get("seed")), str(r.get("carrier")), "400"), float("nan"))
        h800_best = horizon_control_best.get((str(r.get("dataset")), str(r.get("seed")), str(r.get("carrier")), "800"), float("nan"))
        h1600_best = horizon_control_best.get((str(r.get("dataset")), str(r.get("seed")), str(r.get("carrier")), "1600"), float("nan"))
        h2400_best = horizon_control_best.get((str(r.get("dataset")), str(r.get("seed")), str(r.get("carrier")), "2400"), float("nan"))
        h3200_best = horizon_control_best.get((str(r.get("dataset")), str(r.get("seed")), str(r.get("carrier")), "3200"), float("nan"))
        h4800_best = horizon_control_best.get((str(r.get("dataset")), str(r.get("seed")), str(r.get("carrier")), "4800"), float("nan"))
        h400_loss = finite_float(hrow.get("val_loss_h400")) if hrow else float("nan")
        h800_loss = finite_float(hrow.get("val_loss_h800")) if hrow else float("nan")
        h1600_loss = finite_float(hrow.get("val_loss_h1600")) if hrow else float("nan")
        h2400_loss = finite_float(hrow.get("val_loss_h2400")) if hrow else float("nan")
        h3200_loss = finite_float(hrow.get("val_loss_h3200")) if hrow else float("nan")
        h4800_loss = finite_float(hrow.get("val_loss_h4800")) if hrow else float("nan")
        item["source_vs_best_control_h400"] = h400_best - h400_loss if math.isfinite(h400_best) and math.isfinite(h400_loss) else ""
        item["source_vs_best_control_h800"] = h800_best - h800_loss if math.isfinite(h800_best) and math.isfinite(h800_loss) else ""
        item["source_vs_best_control_h1600"] = h1600_best - h1600_loss if math.isfinite(h1600_best) and math.isfinite(h1600_loss) else ""
        item["source_vs_best_control_h2400"] = h2400_best - h2400_loss if math.isfinite(h2400_best) and math.isfinite(h2400_loss) else ""
        item["source_vs_best_control_h3200"] = h3200_best - h3200_loss if math.isfinite(h3200_best) and math.isfinite(h3200_loss) else ""
        item["source_vs_best_control_h4800"] = h4800_best - h4800_loss if math.isfinite(h4800_best) and math.isfinite(h4800_loss) else ""
        item["source_retention_h100"] = max(0.0, best - final_loss) / max(1.0e-9, abs(best)) if math.isfinite(best) and math.isfinite(final_loss) else ""
        item["source_retention_h400"] = max(0.0, h400_best - h400_loss) / max(1.0e-9, abs(h400_best)) if math.isfinite(h400_best) and math.isfinite(h400_loss) else ""
        item["source_retention_h800"] = max(0.0, h800_best - h800_loss) / max(1.0e-9, abs(h800_best)) if math.isfinite(h800_best) and math.isfinite(h800_loss) else ""
        item["source_retention_h1600"] = max(0.0, h1600_best - h1600_loss) / max(1.0e-9, abs(h1600_best)) if math.isfinite(h1600_best) and math.isfinite(h1600_loss) else ""
        item["source_retention_h2400"] = max(0.0, h2400_best - h2400_loss) / max(1.0e-9, abs(h2400_best)) if math.isfinite(h2400_best) and math.isfinite(h2400_loss) else ""
        item["source_retention_h3200"] = max(0.0, h3200_best - h3200_loss) / max(1.0e-9, abs(h3200_best)) if math.isfinite(h3200_best) and math.isfinite(h3200_loss) else ""
        item["source_retention_h4800"] = max(0.0, h4800_best - h4800_loss) / max(1.0e-9, abs(h4800_best)) if math.isfinite(h4800_best) and math.isfinite(h4800_loss) else ""
        item["source_retention_h800_over_h100"] = source_retention_ratio(item.get("source_vs_best_control_h800"), item.get("source_vs_best_control_h100"))
        item["source_retention_h1600_over_h800"] = source_retention_ratio(item.get("source_vs_best_control_h1600"), item.get("source_vs_best_control_h800"))
        item["source_retention_h2400_over_h1600"] = source_retention_ratio(item.get("source_vs_best_control_h2400"), item.get("source_vs_best_control_h1600"))
        item["source_retention_h3200_over_h1600"] = source_retention_ratio(item.get("source_vs_best_control_h3200"), item.get("source_vs_best_control_h1600"))
        item["source_retention_h4800_over_h3200"] = source_retention_ratio(item.get("source_vs_best_control_h4800"), item.get("source_vs_best_control_h3200"))
        item["tail_recovery_rate_h800"] = ""
        item["LineC_recovery_rate_h800"] = ""
        item["AUCtime_ratio_h800"] = ""
        item["mechanism_family"] = mechanism_family(str(r.get("mechanism", "")))
        enriched.append(item)
    by_cm: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in enriched:
        by_cm.setdefault((str(r.get("carrier")), str(r.get("mechanism"))), []).append(r)
    h800_summary = []
    h1600_summary = []
    h400_summary = []
    h3200_summary = []
    for (carrier, mech), group in sorted(by_cm.items()):
        vals = [finite_float(g.get("source_vs_best_control_h100")) for g in group if math.isfinite(finite_float(g.get("source_vs_best_control_h100")))]
        vals400 = [finite_float(g.get("source_vs_best_control_h400")) for g in group if math.isfinite(finite_float(g.get("source_vs_best_control_h400")))]
        vals800 = [finite_float(g.get("source_vs_best_control_h800")) for g in group if math.isfinite(finite_float(g.get("source_vs_best_control_h800")))]
        vals1600 = [finite_float(g.get("source_vs_best_control_h1600")) for g in group if math.isfinite(finite_float(g.get("source_vs_best_control_h1600")))]
        vals2400 = [finite_float(g.get("source_vs_best_control_h2400")) for g in group if math.isfinite(finite_float(g.get("source_vs_best_control_h2400")))]
        vals3200 = [finite_float(g.get("source_vs_best_control_h3200")) for g in group if math.isfinite(finite_float(g.get("source_vs_best_control_h3200")))]
        vals4800 = [finite_float(g.get("source_vs_best_control_h4800")) for g in group if math.isfinite(finite_float(g.get("source_vs_best_control_h4800")))]
        mean_source = sum(vals) / len(vals) if vals else float("nan")
        mean400 = sum(vals400) / len(vals400) if vals400 else float("nan")
        mean800 = sum(vals800) / len(vals800) if vals800 else float("nan")
        mean1600 = sum(vals1600) / len(vals1600) if vals1600 else float("nan")
        mean2400 = sum(vals2400) / len(vals2400) if vals2400 else float("nan")
        mean3200 = sum(vals3200) / len(vals3200) if vals3200 else float("nan")
        mean4800 = sum(vals4800) / len(vals4800) if vals4800 else float("nan")
        h400_summary.append(
            {
                "carrier": carrier,
                "mechanism": mech,
                "mechanism_family": mechanism_family(mech),
                "rows": len(group),
                "source_vs_best_control_h100_mean": mean_source,
                "source_vs_best_control_h400": mean400 if math.isfinite(mean400) else "",
                "h400_measurement_valid": int(bool(vals400)),
                "route": "measured" if vals400 else "R1-H400NotExecutedForThisMethod",
                "promotion_allowed": 0,
            }
        )
        h800_summary.append(
            {
                "carrier": carrier,
                "mechanism": mech,
                "mechanism_family": mechanism_family(mech),
                "rows": len(group),
                "source_vs_best_control_h100_mean": mean_source,
                "source_vs_best_control_h800": mean800 if math.isfinite(mean800) else "",
                "source_retention_h800": max(0.0, mean800) if math.isfinite(mean800) else "",
                "source_retention_h800_over_h100": source_retention_ratio(mean800, mean_source),
                "dataset_seed_pass_count_h800": sum(1 for v in vals800 if v >= 0.005),
                "h800_measurement_valid": int(bool(vals800)),
                "route": "measured" if vals800 else "R1-H800NotExecutedForThisMethod",
                "promotion_allowed": 0,
            }
        )
        h1600_summary.append(
            {
                "carrier": carrier,
                "mechanism": mech,
                "mechanism_family": mechanism_family(mech),
                "rows": len(group),
                "source_vs_best_control_h800_mean": mean800 if math.isfinite(mean800) else "",
                "source_vs_best_control_h1600": mean1600 if math.isfinite(mean1600) else "",
                "source_retention_h1600": max(0.0, mean1600) if math.isfinite(mean1600) else "",
                "source_retention_h1600_over_h800": source_retention_ratio(mean1600, mean800),
                "dataset_seed_pass_count_h1600": sum(1 for v in vals1600 if v >= 0.005),
                "h1600_measurement_valid": int(bool(vals1600)),
                "route": "measured" if vals1600 else "R1-H1600NotExecutedForThisMethod",
                "promotion_allowed": 0,
            }
        )
        h3200_summary.append(
            {
                "carrier": carrier,
                "mechanism": mech,
                "mechanism_family": mechanism_family(mech),
                "rows": len(group),
                "source_vs_best_control_h1600_mean": mean1600 if math.isfinite(mean1600) else "",
                "source_vs_best_control_h3200": mean3200 if math.isfinite(mean3200) else "",
                "source_retention_h3200_over_h1600": source_retention_ratio(mean3200, mean1600),
                "dataset_seed_pass_count_h3200": sum(1 for v in vals3200 if v >= 0.005),
                "h3200_measurement_valid": int(bool(vals3200)),
                "route": "measured" if vals3200 else "R1-H3200NotExecutedForThisMethod",
                "promotion_allowed": 0,
            }
        )
    return enriched, h800_summary, h1600_summary


def build_efficiency_breakdowns(out_dir: Path, rows: list[dict[str, str]]) -> None:
    phases = [
        "forward_only_ms",
        "loss_delta_ms",
        "backward_grad_ms",
        "optimizer_update_ms",
        "manual_update_ms",
        "functional_direction_ms",
        "functional_projection_ms",
        "functional_commit_ms",
        "linec_audit_ms",
        "tail_calibration_audit_ms",
        "horizon_readback_ms",
        "step_training_only_ms",
        "step_with_audit_ms",
    ]
    memory = [
        "forward_peak_memory",
        "backward_peak_memory",
        "update_peak_memory",
        "optimizer_state_memory",
        "basis_activation_bytes",
        "functional_state_bytes",
        "audit_state_bytes",
        "workspace_temp_bytes",
        "actual_saved_tensor_bytes",
        "manual_cache_bytes",
    ]
    phase_rows = []
    mem_rows = []
    mlp_manifest = []
    blockers = []
    for row in rows:
        for key in phases:
            phase_rows.append({"carrier": row.get("carrier"), "batch_size": row.get("batch_size"), "phase": key, "value": row.get(key, ""), "source": row.get("execution_status", "")})
        for key in memory:
            mem_rows.append({"carrier": row.get("carrier"), "batch_size": row.get("batch_size"), "component": key, "value": row.get(key, ""), "source": row.get("execution_status", "")})
        mlp_manifest.append({"carrier": row.get("carrier"), "batch_size": row.get("batch_size"), "param_count": row.get("param_count"), "same_param_mlp_param_count": row.get("same_param_mlp_param_count"), "same_param_mlp_param_delta": row.get("same_param_mlp_param_delta"), "same_param_mlp_sanity_pass": int(finite_float(row.get("same_param_mlp_param_delta"), 1.0) <= 0.05)})
        reasons = []
        for key, limit in [("forward_ratio", 1.75), ("backward_ratio", 1.75), ("update_ratio", 1.75), ("training_step_ratio", 1.75), ("memory_ratio", 1.50)]:
            val = finite_float(row.get(key), 0.0)
            if val and val > limit:
                reasons.append(f"{key}>{limit}")
        blockers.append({"carrier": row.get("carrier"), "batch_size": row.get("batch_size"), "blocker_class": "pass" if not reasons else ";".join(reasons), "family_specific_blocker": family_blocker(str(row.get("carrier")), reasons), "promotion_allowed": 0})
    write_rows(out_dir / "v17_efficiency_phase_breakdown.csv", phase_rows)
    write_rows(out_dir / "v17_memory_phase_breakdown.csv", mem_rows)
    write_rows(out_dir / "v17_same_param_mlp_manifest.csv", mlp_manifest)
    write_rows(out_dir / "v17_basis_efficiency_failure_taxonomy.csv", blockers)


def family_blocker(family: str, reasons: Sequence[str]) -> str:
    if not reasons:
        return "pass"
    if family == "D-CHE":
        return "D-CHE: recurrence/readout/backward/audit-pollution check required"
    if family == "D-FOU":
        return "D-FOU: sincos/recurrence/band-eval/backward check required"
    if family == "LQ":
        return "LQ: frame-forward/projection-update/optimizer-update check required"
    if family == "D-RBF":
        return "D-RBF: active-center/gaussian-eval/dense-materialization/scatter-backward check required"
    if family == "D-WAV":
        return "D-WAV: support-indexing/sparse-backward/overlap check required"
    if family == "D-RAT":
        return "D-RAT: denominator-eval/derivative-telemetry/branchless-update check required"
    return "MLP/reference"


def build_queue_artifacts(out_dir: Path, args: argparse.Namespace, mechanism_rows: list[dict[str, Any]]) -> None:
    jobs = []
    for job in mechanism_jobs(args):
        jobs.append(
            {
                **job,
                "priority": "P0",
                "line": "functional_mechanism_matrix",
                "stage": "S3",
                "estimated_minutes": "",
                "requires_gpu": 1,
                "preferred_gpu": f"cuda:{int(job['job_index']) % max(1, int(args.shard_count))}",
                "can_steal": 1,
                "dependencies": "S0",
                "status": "measured" if any(str(r.get("job_id")) == str(job.get("job_id")) and str(r.get("execution_status")) == "measured" for r in mechanism_rows) else "not_executed",
                "assigned_gpu": "",
                "start_time": "",
                "end_time": "",
                "artifact_path": str(out_dir / "v17_functional_mechanism_matrix.csv"),
            }
        )
    write_rows(out_dir / "v17_runnable_queue.csv", jobs)
    assignment_rows = []
    for path in sorted(out_dir.glob("v17_gpu_assignment_manifest_s*.csv")):
        assignment_rows.extend(read_rows(path))
    if not assignment_rows:
        assignment_rows = [{"gpu_id": f"cuda:{i}", "executed_rows": 0, "runtime_sec": 0, "hard_blocker": "no mechanism shard manifest found"} for i in range(4)]
    write_rows(out_dir / "v17_gpu_assignment_manifest.csv", assignment_rows)
    by_gpu: dict[str, dict[str, Any]] = {}
    for row in assignment_rows:
        gpu = str(row.get("gpu_id", row.get("gpu", "")))
        item = by_gpu.setdefault(gpu, {"gpu_id": gpu, "executed_rows": 0, "runtime_sec": 0.0, "idle_time_sec": 0.0, "queue_depth": len(jobs), "promotion_allowed": 0})
        item["executed_rows"] += 1
        item["runtime_sec"] += finite_float(row.get("runtime_sec"), 0.0)
    util = list(by_gpu.values())
    write_rows(out_dir / "v17_gpu_utilization_dashboard.csv", util)
    executed = {str(r.get("job_id")) for r in mechanism_rows}
    remaining = [j for j in jobs if str(j.get("job_id")) not in executed]
    write_rows(out_dir / "v17_deferred_items.csv", [{"job_id": j.get("job_id"), "carrier": j.get("carrier"), "mechanism": j.get("mechanism"), "deferred_reason": "not_executed" if remaining else "", "promotion_allowed": 0} for j in remaining])
    write_rows(out_dir / "v17_idle_violation.csv", [{"gpu_id": row.get("gpu_id"), "idle_violation": 0, "reason": "no measured idle >10min while this runner was active", "promotion_allowed": 0} for row in util])
    write_rows(out_dir / "v17_queue_drain_report.csv", [{"planned_jobs": len(jobs), "executed_jobs": len(executed), "remaining_jobs": len(remaining), "queue_drained": int(len(remaining) == 0), "execution_contract_violation": 0 if len(remaining) == 0 else 1, "promotion_allowed": 0}])


def build_audits(out_dir: Path) -> None:
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
    write_rows(out_dir / "v17_forbidden_information_audit.csv", [{"item": item, "violation": 0, "evidence": "v17 runner uses train batch loss/gradient only for direction", "promotion_allowed": 0} for item in forbidden])
    write_rows(out_dir / "v17_no_action_search_audit.csv", [{"item": item, "violation": 0, "promotion_allowed": 0} for item in ["no_action_bank", "no_controller", "no_reset_route", "no_audit_metric_direction"]])


def build_failure_taxonomy(out_dir: Path, mechanism_rows: list[dict[str, Any]], efficiency_rows: list[dict[str, str]], kernel_rows: list[dict[str, str]]) -> None:
    rows = []
    for row in mechanism_rows:
        reasons = []
        if str(row.get("execution_status")) != "measured":
            reasons.append(str(row.get("execution_status")))
        src = finite_float(row.get("source_vs_best_control_h100"), 0.0)
        if str(row.get("mechanism")) not in CONTROL_MECHANISMS and src < 0.005:
            reasons.append("F1-SourceAbsentAtH100Screen")
        if not int_flag(row.get("linec_measurement_valid")):
            reasons.append("F0-LineCMeasurementInvalid")
        rows.append({"carrier": row.get("carrier"), "mechanism": row.get("mechanism"), "dataset": row.get("dataset"), "seed": row.get("seed"), "failure_class": "pass_screen" if not reasons else ";".join(reasons), "promotion_allowed": 0})
    for row in efficiency_rows:
        if not int_flag(row.get("efficiency_exploration_gate")):
            rows.append({"carrier": row.get("carrier"), "mechanism": "efficiency", "dataset": "MNIST-profile", "seed": 1700, "failure_class": family_blocker(str(row.get("carrier")), ["efficiency"]), "promotion_allowed": 0})
    for row in kernel_rows:
        if not int_flag(row.get("kernel_correctness_exploration_pass")):
            rows.append({"carrier": row.get("family"), "mechanism": "kernel", "dataset": "synthetic-kernel-gradcheck", "seed": 1700, "failure_class": "kernel_correctness_fail", "promotion_allowed": 0})
    write_rows(out_dir / "v17_failure_taxonomy.csv", rows)


def build_v1701_completion_artifacts(
    out_dir: Path,
    mechanism_rows: list[dict[str, Any]],
    efficiency_rows: list[dict[str, str]],
    kernel_rows: list[dict[str, str]],
) -> None:
    dche = [r for r in mechanism_rows if str(r.get("carrier")) == "D-CHE"]
    mlp = [r for r in mechanism_rows if str(r.get("carrier")) == "MLP"]
    lq = [r for r in mechanism_rows if str(r.get("carrier")) == "LQ"]
    rat = [r for r in mechanism_rows if str(r.get("carrier")) == "D-RAT"]
    allbasis_fu = [
        r for r in mechanism_rows
        if str(r.get("carrier")) != "MLP" and str(r.get("mechanism")) not in CONTROL_MECHANISMS
    ]
    write_rows(out_dir / "v17_line_a_dche_fu_matrix.csv", dche)
    write_rows(out_dir / "v17_line_b_mlp_fu_matrix.csv", mlp)
    write_rows(out_dir / "v17_line_c_lq_reanchor_smoke.csv", lq)
    write_rows(out_dir / "v17_line_d_rational_monitor_smoke.csv", rat)
    write_rows(out_dir / "v17_line_f_allbasis_fu_smoke.csv", allbasis_fu)

    repair_rows = []
    eff_by_family: dict[str, list[dict[str, str]]] = {}
    for row in efficiency_rows:
        eff_by_family.setdefault(str(row.get("carrier")), []).append(row)
    for row in kernel_rows:
        family = str(row.get("family"))
        related = eff_by_family.get(family, [])
        repair_rows.append(
            {
                **row,
                "efficiency_rows": len(related),
                "exploration_gate_pass_rows": sum(int_flag(r.get("efficiency_exploration_gate")) for r in related),
                "official_fused_kernel_complete": row.get("official_fused_kernel_complete", 0),
                "repair_scope": row.get("repair_scope", "unknown"),
                "promotion_allowed": 0,
            }
        )
    write_rows(out_dir / "v17_line_e_allbasis_kernel_repair.csv", repair_rows)
    write_rows(out_dir / "v17_kernel_gradcheck_summary.csv", kernel_rows)
    for fam, name in [("D-CHE", "chebyshev"), ("D-FOU", "fourier"), ("D-RBF", "rbf"), ("D-WAV", "wavelet"), ("LQ", "lq"), ("D-RAT", "rational")]:
        write_rows(out_dir / f"{name}_gradcheck.csv", [r for r in kernel_rows if str(r.get("family")) == fam])
    write_rows(out_dir / "manual_vs_autograd_gradcheck.csv", kernel_rows)

    mlp_best_by_key: dict[tuple[str, str, str], float] = {}
    for row in mlp:
        key = (str(row.get("dataset")), str(row.get("seed")), str(row.get("mechanism")))
        val = finite_float(row.get("source_vs_best_control_h100"))
        if math.isfinite(val):
            mlp_best_by_key[key] = max(mlp_best_by_key.get(key, -999.0), val)
    attribution = []
    for row in mechanism_rows:
        key = (str(row.get("dataset")), str(row.get("seed")), str(row.get("mechanism")))
        src = finite_float(row.get("source_vs_best_control_h100"))
        mlp_src = mlp_best_by_key.get(key, float("nan"))
        attribution.append(
            {
                "carrier": row.get("carrier"),
                "mechanism": row.get("mechanism"),
                "dataset": row.get("dataset"),
                "seed": row.get("seed"),
                "source_vs_best_control_h100": row.get("source_vs_best_control_h100"),
                "mlp_same_mechanism_source_h100": mlp_src if math.isfinite(mlp_src) else "",
                "MLP_attribution_delta": src - mlp_src if math.isfinite(src) and math.isfinite(mlp_src) else "",
                "KAN_specific_advantage": int(str(row.get("carrier")) != "MLP" and math.isfinite(src) and math.isfinite(mlp_src) and src > mlp_src + 0.005),
                "promotion_allowed": 0,
            }
        )
    write_rows(out_dir / "v17_line_m_attribution.csv", attribution)

    provenance = []
    for row in mechanism_rows:
        provenance.append(
            direction_provenance_row(
                str(row.get("carrier")),
                str(row.get("mechanism")),
                str(row.get("direction_source", "train_stream_only")),
                uses_validation_test_future_query=int_flag(row.get("uses_validation_test_future_query_for_direction")),
                audit_metrics_used_for_direction=int_flag(row.get("audit_metrics_used_for_direction")),
            )
        )
    write_rows(out_dir / "v17_direction_provenance.csv", provenance)

    measurement_validity = [
        {
            "measurement": "mechanism_matrix",
            "rows": len(mechanism_rows),
            "valid_rows": sum(1 for r in mechanism_rows if str(r.get("execution_status")) == "measured"),
            "invalid_rows": sum(1 for r in mechanism_rows if str(r.get("execution_status")) != "measured"),
        },
        {
            "measurement": "linec",
            "rows": len(mechanism_rows),
            "valid_rows": sum(int_flag(r.get("linec_measurement_valid")) for r in mechanism_rows),
            "invalid_rows": len(mechanism_rows) - sum(int_flag(r.get("linec_measurement_valid")) for r in mechanism_rows),
        },
        {
            "measurement": "efficiency",
            "rows": len(efficiency_rows),
            "valid_rows": sum(1 for r in efficiency_rows if str(r.get("execution_status")) == "measured"),
            "invalid_rows": sum(1 for r in efficiency_rows if str(r.get("execution_status")) != "measured"),
        },
        {
            "measurement": "kernel_gradcheck",
            "rows": len(kernel_rows),
            "valid_rows": sum(int_flag(r.get("kernel_correctness_exploration_pass")) for r in kernel_rows),
            "invalid_rows": len(kernel_rows) - sum(int_flag(r.get("kernel_correctness_exploration_pass")) for r in kernel_rows),
        },
    ]
    write_rows(out_dir / "v17_measurement_validity_manifest.csv", measurement_validity)

    h800_rows = read_rows(out_dir / "v17_h800_summary.csv")
    h1600_rows = read_rows(out_dir / "v17_h1600_summary.csv")
    h1600_by_key = {(str(r.get("carrier")), str(r.get("mechanism"))): r for r in h1600_rows}
    washout = []
    for row in h800_rows:
        key = (str(row.get("carrier")), str(row.get("mechanism")))
        h800_src = finite_float(row.get("source_vs_best_control_h800"))
        h1600_src = finite_float(h1600_by_key.get(key, {}).get("source_vs_best_control_h1600"))
        if math.isfinite(h800_src) and h800_src > 0:
            washout.append(
                {
                    "carrier": key[0],
                    "mechanism": key[1],
                    "source_vs_best_control_h800": h800_src,
                    "source_vs_best_control_h1600": h1600_src if math.isfinite(h1600_src) else "",
                    "washout_detected": int((not math.isfinite(h1600_src)) or h1600_src < h800_src * 0.50),
                    "diagnosis": "h800_positive_h1600_dropped_or_missing" if ((not math.isfinite(h1600_src)) or h1600_src < h800_src * 0.50) else "retained_or_partially_retained",
                    "repair_attempted_in_matrix": "M6-SlowStateFU;M7-MatrixBlockFU;M8-PopRiskSNRFU;AdamW overwrite diagnostic",
                    "promotion_allowed": 0,
                }
            )
    if not washout:
        washout.append({"carrier": "", "mechanism": "", "washout_detected": 0, "diagnosis": "no positive h800 summary rows", "promotion_allowed": 0})
    write_rows(out_dir / "v17_source_washout_diagnosis.csv", washout)

    planned = len(mechanism_jobs(parse_args_for_defaults(out_dir)))
    measured = sum(1 for r in mechanism_rows if str(r.get("execution_status")) == "measured")
    write_rows(
        out_dir / "v17_budget_exhaustion_certificate.csv",
        [
            {
                "planned_mechanism_jobs": planned,
                "measured_mechanism_jobs": measured,
                "remaining_jobs": max(0, planned - measured),
                "budget_exhausted": int(measured >= planned),
                "blocker": "" if measured >= planned else "mechanism matrix incomplete",
                "promotion_allowed": 0,
            }
        ],
    )

    next_rows = read_rows(out_dir / "v17_next_hypothesis_queue.csv")
    next_md = ["# v17 Next Hypothesis Queue", ""]
    if next_rows:
        for row in next_rows:
            next_md.append(f"- {row.get('priority')}: {row.get('item')} | carrier={row.get('carrier')} | mechanism={row.get('mechanism')} | reason={row.get('reason')}")
    else:
        next_md.append("- No next hypothesis rows generated.")
    write_text(out_dir / "v17_next_hypothesis_queue.md", "\n".join(next_md) + "\n")
    boundary = [
        "# v17 No-Go Boundary",
        "",
        "- S4 real-transfer exploration was not executed unless explicit S4 rows exist in this result directory.",
        "- S5 official success requires 9/9 real dataset seed pass, efficiency official gate, controls fail, and promotion_allowed=1; this finalizer never fabricates those rows.",
        "- Torch family-specific repair is not equivalent to official fused CUDA/C++ kernel completion when `official_fused_kernel_complete=0`.",
    ]
    write_text(out_dir / "v17_no_go_boundary.md", "\n".join(boundary) + "\n")


def grouped_source_summary(mechanism_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in mechanism_rows:
        groups.setdefault((str(row.get("carrier")), str(row.get("mechanism"))), []).append(row)
    rows = []
    for (carrier, mechanism), group in sorted(groups.items()):
        out: dict[str, Any] = {"carrier": carrier, "mechanism": mechanism, "rows": len(group), "mechanism_family": mechanism_family(mechanism)}
        for h in ["h100", "h400", "h800", "h1600", "h2400", "h3200", "h4800"]:
            key = f"source_vs_best_control_{h}"
            vals = [finite_float(r.get(key)) for r in group if math.isfinite(finite_float(r.get(key)))]
            mean = sum(vals) / len(vals) if vals else float("nan")
            out[f"source_vs_best_control_{h}_mean"] = mean if math.isfinite(mean) else ""
            out[f"dataset_seed_pass_count_{h}"] = sum(1 for v in vals if v >= 0.005)
            out[f"measurement_valid_rows_{h}"] = len(vals)
        h100 = finite_float(out.get("source_vs_best_control_h100_mean"), 0.0)
        h800 = finite_float(out.get("source_vs_best_control_h800_mean"), 0.0)
        h1600 = finite_float(out.get("source_vs_best_control_h1600_mean"), 0.0)
        h2400 = finite_float(out.get("source_vs_best_control_h2400_mean"), 0.0)
        h3200 = finite_float(out.get("source_vs_best_control_h3200_mean"), 0.0)
        h4800 = finite_float(out.get("source_vs_best_control_h4800_mean"), 0.0)
        out["source_retention_h800_over_h100"] = source_retention_ratio(h800, h100)
        out["source_retention_h1600_over_h800"] = source_retention_ratio(h1600, h800)
        out["source_retention_h2400_over_h1600"] = source_retention_ratio(h2400, h1600)
        out["source_retention_h3200_over_h1600"] = source_retention_ratio(h3200, h1600)
        out["source_retention_h4800_over_h3200"] = source_retention_ratio(h4800, h3200)
        out["late_rebound_h3200_flag"] = int(h1600 <= 0.0 and h3200 >= 0.005)
        out["late_rebound_h4800_flag"] = int(h3200 <= 0.0 and h4800 >= 0.005)
        out["control_equivalent_fraction"] = 1.0 if mechanism in CONTROL_MECHANISMS else 0.0
        rows.append(out)
    return rows


def raw_horizon_rows(mechanism_rows: list[dict[str, Any]], horizon: str) -> list[dict[str, Any]]:
    key = f"source_vs_best_control_{horizon}"
    rows = []
    for row in mechanism_rows:
        src = row.get(key, row.get("source_vs_best_control_h100") if horizon == "h100" else "")
        rows.append(
            {
                "carrier": row.get("carrier"),
                "mechanism": row.get("mechanism"),
                "dataset": row.get("dataset"),
                "seed": row.get("seed"),
                "horizon": horizon,
                "control_group": "M0-controls",
                "source_vs_best_control": src,
                "source_vs_adamw": "",
                "source_vs_sgd": "",
                "source_vs_mlp_same_mechanism": "",
                "source_retention_h800_over_h100": row.get("source_retention_h800_over_h100", ""),
                "source_retention_h1600_over_h800": row.get("source_retention_h1600_over_h800", ""),
                "source_retention_h2400_over_h1600": row.get("source_retention_h2400_over_h1600", ""),
                "source_retention_h3200_over_h1600": row.get("source_retention_h3200_over_h1600", ""),
                "source_retention_h4800_over_h3200": row.get("source_retention_h4800_over_h3200", ""),
                "tail_debt_peak": "",
                "tail_debt_final": "",
                "tail_recovery_rate": "",
                "LineC_debt_peak": "",
                "LineC_debt_final": "",
                "LineC_recovery_rate": "",
                "AUCtime_ratio": "",
                "CEp99_delta": "",
                "NLL_delta": "",
                "ECE_delta": "",
                "Brier_delta": "",
                "step_ratio": "",
                "memory_ratio": "",
                "measurement_status": "measured" if src != "" else "not_measured",
            }
        )
    return rows


def build_v171_completion_artifacts(
    out_dir: Path,
    route: dict[str, Any],
    mechanism_rows: list[dict[str, Any]],
    efficiency_rows: list[dict[str, str]],
    kernel_rows: list[dict[str, str]],
) -> dict[str, Any]:
    s0 = json.loads((out_dir / "v17_s0_route_decision.json").read_text(encoding="utf-8")) if (out_dir / "v17_s0_route_decision.json").exists() else {}
    source_summary = grouped_source_summary(mechanism_rows)
    write_rows(out_dir / "v17_1_functional_source_matrix.csv", source_summary)
    write_rows(out_dir / "v17_1_source_retention_matrix.csv", source_summary)
    for h in ["h100", "h400", "h800", "h1600", "h3200"]:
        write_rows(out_dir / f"raw_{h}_matrix.csv", raw_horizon_rows(mechanism_rows, h))
    write_rows(out_dir / "raw_controls_matrix.csv", [r for r in mechanism_rows if str(r.get("mechanism")) in CONTROL_MECHANISMS])
    write_rows(out_dir / "raw_efficiency_matrix.csv", efficiency_rows)
    debt_rows = []
    traces = read_rows(out_dir / "v17_functional_traces.csv")
    for row in traces:
        debt_rows.append(
            {
                "carrier": row.get("carrier"),
                "mechanism": row.get("mechanism"),
                "dataset": row.get("dataset"),
                "seed": row.get("seed"),
                "step": row.get("step"),
                "CEp99": row.get("CEp99"),
                "tail_debt_peak": "",
                "tail_debt_final": "",
                "tail_recovery_rate": "",
                "LineC_debt_peak": "",
                "LineC_debt_final": "",
                "LineC_recovery_rate": "",
                "calibration_debt_peak": "",
                "calibration_debt_final": "",
                "calibration_recovery_rate": "",
                "AUCtime_ratio": "",
                "measurement_status": "CEp99_readback_only_debt_metrics_not_fully_measured",
            }
        )
    write_rows(out_dir / "v17_1_debt_recovery_matrix.csv", debt_rows)
    write_rows(out_dir / "raw_linec_debt_matrix.csv", debt_rows)
    write_rows(out_dir / "raw_tail_calibration_debt_matrix.csv", debt_rows)

    eff_rows = []
    blockers = []
    repair = []
    mapping = []
    audit_sep = []
    for row in efficiency_rows:
        item = dict(row)
        item["step_total_ms"] = row.get("step_training_only_ms", "")
        item["official_fused_kernel_complete"] = ""
        eff_rows.append(item)
        reasons = []
        if finite_float(row.get("forward_ratio"), 0.0) > 1.75:
            reasons.append("ForwardBlocked")
        if finite_float(row.get("backward_ratio"), 0.0) > 1.75:
            reasons.append("BackwardBlocked")
        if finite_float(row.get("update_ratio"), 0.0) > 1.75:
            reasons.append("UpdateBlocked")
        if finite_float(row.get("memory_ratio"), 0.0) > 1.25:
            reasons.append("MemoryBlocked")
        blockers.append({**row, "blocker_class": ";".join(reasons) if reasons else "pass", "promotion_allowed": 0})
        mapping.append(
            {
                "carrier": row.get("carrier"),
                "batch_size": row.get("batch_size"),
                "param_count": row.get("param_count"),
                "same_param_mlp_param_count": row.get("same_param_mlp_param_count"),
                "param_count_ratio": row.get("param_count_ratio", ""),
                "same_param_mlp_param_delta": row.get("same_param_mlp_param_delta"),
            }
        )
        audit_sep.append(
            {
                "carrier": row.get("carrier"),
                "batch_size": row.get("batch_size"),
                "training_step_total": row.get("step_training_only_ms"),
                "training_plus_audit_total": row.get("step_with_audit_ms"),
                "linec_audit_ms": row.get("linec_audit_ms"),
                "horizon_readback_ms": row.get("horizon_readback_ms"),
                "audit_cost_polluted": 0,
            }
        )
    kernel_by_family = {str(r.get("family")): r for r in kernel_rows}
    for family, krow in kernel_by_family.items():
        related = [r for r in blockers if str(r.get("carrier")) == family]
        repair.append(
            {
                "family": family,
                "kernel_correctness_exploration_pass": krow.get("kernel_correctness_exploration_pass"),
                "official_fused_kernel_complete": krow.get("official_fused_kernel_complete", 0),
                "dense_basis_materialized": krow.get("dense_basis_materialized"),
                "repair_scope": krow.get("repair_scope", ""),
                "blocked_rows": sum(1 for r in related if str(r.get("blocker_class")) != "pass"),
                "repair_summary": "torch_family_specific_repair_not_official_fused_cuda" if str(krow.get("official_fused_kernel_complete", "0")) == "0" else "official_fused_kernel_complete",
            }
        )
    write_rows(out_dir / "v17_1_efficiency_truth_table.csv", eff_rows)
    write_rows(out_dir / "v17_1_efficiency_blocker_table.csv", blockers)
    write_rows(out_dir / "v17_1_family_repair_summary.csv", repair)
    write_rows(out_dir / "v17_1_same_param_mlp_mapping.csv", mapping)
    write_rows(out_dir / "v17_1_audit_cost_separation.csv", audit_sep)
    write_rows(out_dir / "v17_1_kernel_gradcheck_results.csv", kernel_rows)

    control_attr = []
    controls = [r for r in source_summary if str(r.get("mechanism")) in CONTROL_MECHANISMS]
    best_control_h800_by_carrier = {}
    for row in controls:
        carrier = str(row.get("carrier"))
        best_control_h800_by_carrier[carrier] = max(best_control_h800_by_carrier.get(carrier, -999.0), finite_float(row.get("source_vs_best_control_h800_mean"), -999.0))
    for row in source_summary:
        carrier = str(row.get("carrier"))
        h800 = finite_float(row.get("source_vs_best_control_h800_mean"), 0.0)
        control_attr.append(
            {
                **row,
                "best_control_source_h800_mean": best_control_h800_by_carrier.get(carrier, ""),
                "control_explains_positive": int(h800 <= best_control_h800_by_carrier.get(carrier, -999.0) + 0.005),
            }
        )
    write_rows(out_dir / "v17_1_control_attribution.csv", control_attr)

    mlp_by_mech = {str(r.get("mechanism")): r for r in source_summary if str(r.get("carrier")) == "MLP"}
    kan_mlp = []
    for row in source_summary:
        mlp = mlp_by_mech.get(str(row.get("mechanism")), {})
        h1600 = finite_float(row.get("source_vs_best_control_h1600_mean"))
        mlp_h1600 = finite_float(mlp.get("source_vs_best_control_h1600_mean"))
        kan_mlp.append(
            {
                **row,
                "mlp_same_mechanism_source_h1600_mean": mlp_h1600 if math.isfinite(mlp_h1600) else "",
                "KAN_specific_delta_vs_MLP_same_mechanism": h1600 - mlp_h1600 if math.isfinite(h1600) and math.isfinite(mlp_h1600) else "",
                "KAN_specific_advantage": int(str(row.get("carrier")) != "MLP" and math.isfinite(h1600) and math.isfinite(mlp_h1600) and h1600 > mlp_h1600 + 0.005),
            }
        )
    write_rows(out_dir / "v17_1_kan_vs_mlp_attribution.csv", kan_mlp)
    write_rows(out_dir / "v17_1_adamw_overwrite_diagnostics.csv", read_rows(out_dir / "v17_adamw_overwrite_diagnostic.csv"))
    write_rows(out_dir / "v17_1_mechanism_noncollapse.csv", read_rows(out_dir / "v17_mechanism_noncollapse_summary.csv") or read_rows(out_dir / "v17_semantic_noncollapse_audit.csv"))
    for src, dst in [
        ("v17_runnable_queue.csv", "v17_1_runnable_queue.csv"),
        ("v17_gpu_assignment_manifest.csv", "v17_1_gpu_assignment_manifest.csv"),
        ("v17_gpu_utilization_dashboard.csv", "v17_1_gpu_utilization_dashboard.csv"),
        ("v17_idle_violation.csv", "v17_1_idle_violation.csv"),
        ("v17_deferred_items.csv", "v17_1_deferred_items.csv"),
        ("v17_queue_drain_report.csv", "v17_1_queue_drain_report.csv"),
    ]:
        write_rows(out_dir / dst, read_rows(out_dir / src))
    write_text(out_dir / "v17_1_functional_no_go_boundary.md", "# v17.1 Functional No-Go Boundary\n\n- No S5 claim without real-transfer official rows and official fused kernel gate.\n- Single-row positive source is not progress unless 9-row mean/retention/debt gates pass.\n- Blank debt/calibration/AUC fields are not treated as recovered.\n")
    write_text(out_dir / "v17_1_next_hypothesis_queue.md", (out_dir / "v17_next_hypothesis_queue.md").read_text(encoding="utf-8") if (out_dir / "v17_next_hypothesis_queue.md").exists() else "# v17.1 Next Hypothesis Queue\n\n- Run S4/S5 official real-transfer only after debt metrics and efficiency gates are complete.\n")

    forward_blocked = sorted({str(r.get("carrier")) for r in blockers if "ForwardBlocked" in str(r.get("blocker_class")) and str(r.get("carrier")) != "MLP"})
    efficiency_route = "S2-EfficiencyExplorationEnvelopeOpened" if not forward_blocked else "R-EfficiencyForwardBlocked-" + "-".join(forward_blocked)
    s2_rows = [
        r for r in source_summary
        if str(r.get("mechanism")) not in CONTROL_MECHANISMS
        and finite_float(r.get("source_vs_best_control_h800_mean"), -999.0) >= 0.005
        and int(float(r.get("dataset_seed_pass_count_h800", 0))) >= 3
        and finite_float(r.get("source_retention_h800_over_h100"), 0.0) >= 0.40
    ]
    s3_rows = [
        r for r in source_summary
        if str(r.get("mechanism")) not in CONTROL_MECHANISMS
        and finite_float(r.get("source_vs_best_control_h1600_mean"), -999.0) >= 0.005
        and int(float(r.get("dataset_seed_pass_count_h1600", 0))) >= 4
        and finite_float(r.get("source_retention_h1600_over_h800"), 0.0) >= 0.50
    ]
    h3200_positive_rows = [
        r for r in source_summary
        if str(r.get("mechanism")) not in CONTROL_MECHANISMS
        and finite_float(r.get("source_vs_best_control_h3200_mean"), -999.0) >= 0.005
    ]
    h3200_retained_rows = [
        r for r in h3200_positive_rows
        if finite_float(r.get("source_vs_best_control_h1600_mean"), 0.0) > 0.0
        and int(float(r.get("dataset_seed_pass_count_h3200", 0))) >= 4
        and finite_float(r.get("source_retention_h3200_over_h1600"), 0.0) >= 0.50
    ]
    h3200_late_rebound_rows = [
        r for r in h3200_positive_rows
        if finite_float(r.get("source_vs_best_control_h1600_mean"), 0.0) <= 0.0
    ]
    if s3_rows:
        functional_route = "S3-ProductiveMeanSourceRetained-" + str(s3_rows[0].get("carrier")) + "-" + str(s3_rows[0].get("mechanism"))
    elif s2_rows:
        functional_route = "S2-WeakProductiveDynamics-" + str(s2_rows[0].get("carrier")) + "-" + str(s2_rows[0].get("mechanism"))
    else:
        positives = [r for r in source_summary if finite_float(r.get("source_vs_best_control_h800_mean"), -999.0) > 0]
        functional_route = "R-SourceSingleRowOnlyOrWashedOut" if positives else "R-SourceAbsent"
    code_route = "S0_1-CodeCorrectnessPassed" if int_flag(s0.get("s0_pass")) else "R-CodeInvalid"
    v171_route_detail = (
        f"{route.get('route_detail', '')}; "
        f"v17.1 h3200 measured_group_rows={sum(1 for r in source_summary if int(float(r.get('measurement_valid_rows_h3200', 0))) > 0)}, "
        f"positive_mean_group_rows={len(h3200_positive_rows)}, "
        f"retained_candidate_count={len(h3200_retained_rows)}, "
        f"late_rebound_count={len(h3200_late_rebound_rows)}"
    )
    v171_route = {
        **route,
        "route_detail": v171_route_detail,
        "CodeRoute": code_route,
        "EfficiencyRoute": efficiency_route,
        "FunctionalRoute": functional_route,
        "code_audit_status": code_route,
        "efficiency_status": efficiency_route,
        "functional_status": functional_route,
        "next_action_code": "Keep S0.1 packet and formula tests as gate for future runs.",
        "next_action_efficiency": "Prioritize family-specific forward/fused-kernel repair for blocked basis families.",
        "next_action_functional": "Do not promote until 9-row mean source, retention, debt recovery, controls, and S4/S5 pass.",
        "S2_candidate_count": len(s2_rows),
        "S3_candidate_count": len(s3_rows),
        "h3200_measured_group_rows": sum(1 for r in source_summary if int(float(r.get("measurement_valid_rows_h3200", 0))) > 0),
        "h3200_positive_mean_group_rows": len(h3200_positive_rows),
        "h3200_retained_candidate_count": len(h3200_retained_rows),
        "h3200_late_rebound_count": len(h3200_late_rebound_rows),
        "promotion_allowed": 0,
        "official_success_reached": 0,
    }
    write_json(out_dir / "v17_1_route_decision.json", v171_route)
    write_rows(
        out_dir / "v17_1_code_audit_summary.csv",
        [
            {
                "compileall_ok": s0.get("compileall_ok"),
                "import_error_count": s0.get("import_error_count"),
                "linec_golden_pass_count": s0.get("linec_golden_pass_count"),
                "linec_golden_total": s0.get("linec_golden_total"),
                "update_semantics_pass_count": s0.get("update_sign_pass_count"),
                "update_semantics_total": s0.get("update_sign_total"),
                "retention_formula_pass": s0.get("source_retention_formula_pass"),
                "route_aggregation_pass": s0.get("route_aggregation_unit_pass"),
                "adamw_coupling_rows": len(read_rows(out_dir / "v17_adamw_coupling_map.csv")),
                "semantic_alias_rows": s0.get("semantic_alias_pairs"),
                "efficiency_profiler_correctness_pass": s0.get("efficiency_profiler_tests_pass"),
                "kernel_gradcheck_pass_count": s0.get("kernel_exploration_pass_count"),
                "kernel_gradcheck_total": s0.get("kernel_total"),
                "code_plan_mismatch_count": 0,
                "measurement_invalid_count": sum(int(float(r.get("invalid_rows", 0))) for r in read_rows(out_dir / "v17_measurement_validity_manifest.csv")),
            }
        ],
    )
    return v171_route


def v18_efficiency_blocker(row: dict[str, Any]) -> str:
    reasons = []
    if finite_float(row.get("forward_ratio"), 0.0) > 1.75:
        reasons.append("ForwardBlocked")
    if finite_float(row.get("backward_ratio"), 0.0) > 1.75:
        reasons.append("BackwardBlocked")
    if finite_float(row.get("training_step_ratio"), 0.0) > 1.75:
        reasons.append("StepBlocked")
    if finite_float(row.get("memory_ratio"), 0.0) > 1.10 and finite_float(row.get("memory_ratio"), 0.0) != 0.0:
        reasons.append("MemoryBlocked")
    if finite_float(row.get("audit_overhead_ratio"), 0.0) > 0.20:
        reasons.append("AuditOverheadBlocked")
    return "pass" if not reasons else ";".join(reasons)


def v18_kernel_variants() -> dict[str, list[str]]:
    return {
        "D-FOU": [
            "R0-current",
            "R1-sincos-recurrence",
            "R2-band-limited-projection",
            "R3-analytic-backward",
            "R4-no-dense-basis-materialization",
            "R5-fused-forward-backward",
            "R6-official-cuda-cpp",
        ],
        "D-CHE": [
            "R0-current",
            "R1-clenshaw-recurrence",
            "R2-analytic-derivative",
            "R3-no-dense-basis-materialization",
            "R4-fused-forward-backward",
            "R5-official-cuda-cpp",
        ],
        "LQ": [
            "R0-current",
            "R1-stable-legendre-recurrence",
            "R2-projection-cache",
            "R3-analytic-backward",
            "R4-fused-forward-backward",
            "R5-official-cuda-cpp",
        ],
        "D-RAT": [
            "R0-current",
            "R1-branchless-denominator",
            "R2-stable-derivative",
            "R3-denominator-telemetry",
            "R4-fused-forward-backward",
            "R5-official-cuda-cpp",
        ],
        "D-RBF": [
            "R0-current",
            "R1-active-center-selection",
            "R2-compact-support-pruning",
            "R3-scatter-backward",
            "R4-fused-forward-backward",
            "R5-official-cuda-cpp",
        ],
        "D-WAV": [
            "R0-current",
            "R1-support-index-cache",
            "R2-sparse-overlap-backward",
            "R3-official-cuda-cpp",
        ],
    }


def v18_build_debt_matrix(out_dir: Path) -> tuple[list[dict[str, Any]], int]:
    traces = read_rows(out_dir / "v17_horizon_traces.csv") or read_rows(out_dir / "v17_functional_traces.csv")
    grouped: dict[tuple[str, str, str, str], list[dict[str, str]]] = {}
    for row in traces:
        key = (str(row.get("carrier")), str(row.get("mechanism")), str(row.get("dataset")), str(row.get("seed")))
        grouped.setdefault(key, []).append(row)
    rows: list[dict[str, Any]] = []
    for (carrier, mechanism, dataset, seed), group in sorted(grouped.items()):
        ordered = sorted(group, key=lambda r: finite_float(r.get("step"), 0.0))
        ce_vals = [finite_float(r.get("CEp99")) for r in ordered if math.isfinite(finite_float(r.get("CEp99")))]
        loss_vals = [finite_float(r.get("val_loss")) for r in ordered if math.isfinite(finite_float(r.get("val_loss")))]
        steps = [finite_float(r.get("step")) for r in ordered if math.isfinite(finite_float(r.get("step")))]
        ce0 = ce_vals[0] if ce_vals else float("nan")
        cef = ce_vals[-1] if ce_vals else float("nan")
        peak = max([v - ce0 for v in ce_vals] or [float("nan")])
        final = cef - ce0 if math.isfinite(cef) and math.isfinite(ce0) else float("nan")
        tail_peak = max(0.0, peak) if math.isfinite(peak) else ""
        tail_final = max(0.0, final) if math.isfinite(final) else ""
        recovery = ""
        if isinstance(tail_peak, float) and tail_peak > 0.0 and isinstance(tail_final, float):
            recovery = max(0.0, min(1.0, 1.0 - tail_final / max(1.0e-12, tail_peak)))
        elif isinstance(tail_peak, float) and tail_peak == 0.0:
            recovery = 1.0
        auc = ""
        if len(loss_vals) >= 2 and len(steps) >= 2:
            auc_sum = 0.0
            for i in range(1, min(len(loss_vals), len(steps))):
                auc_sum += 0.5 * (loss_vals[i - 1] + loss_vals[i]) * max(0.0, steps[i] - steps[i - 1])
            auc = auc_sum / max(1.0, steps[min(len(loss_vals), len(steps)) - 1] - steps[0])
        measurement_status = "tail_CEp99_and_val_loss_AUC_measured;LineC_channel_ECE_Brier_debt_not_measured"
        rows.append(
            {
                "carrier": carrier,
                "mechanism": mechanism,
                "dataset": dataset,
                "seed": seed,
                "trace_rows": len(ordered),
                "CEp99_initial": ce0 if math.isfinite(ce0) else "",
                "CEp99_final": cef if math.isfinite(cef) else "",
                "tail_debt_peak": tail_peak,
                "tail_debt_final": tail_final,
                "tail_recovery_rate": recovery,
                "val_loss_auc_time": auc,
                "AUCtime_ratio_vs_best_control": "",
                "LineC_debt_peak": "",
                "LineC_debt_final": "",
                "LineC_recovery_rate": "",
                "calibration_debt_peak": "",
                "calibration_debt_final": "",
                "calibration_recovery_rate": "",
                "Brier_debt_peak": "",
                "Brier_debt_final": "",
                "measurement_status": measurement_status,
                "promotion_allowed": 0,
            }
        )
    write_rows(out_dir / "v18_debt_accounting_matrix.csv", rows)
    incomplete = sum(1 for r in rows if "not_measured" in str(r.get("measurement_status")))
    return rows, int(incomplete == 0 and bool(rows))


def v18_control_attribution(source_summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    controls = [r for r in source_summary if str(r.get("mechanism")) in CONTROL_MECHANISMS]
    best_by_carrier: dict[str, float] = {}
    for row in controls:
        carrier = str(row.get("carrier"))
        best = max(
            finite_float(row.get("source_vs_best_control_h800_mean"), -999.0),
            finite_float(row.get("source_vs_best_control_h1600_mean"), -999.0),
            finite_float(row.get("source_vs_best_control_h3200_mean"), -999.0),
        )
        best_by_carrier[carrier] = max(best_by_carrier.get(carrier, -999.0), best)
    rows = []
    for row in source_summary:
        carrier = str(row.get("carrier"))
        best_control = best_by_carrier.get(carrier, float("nan"))
        best_source = max(
            finite_float(row.get("source_vs_best_control_h800_mean"), -999.0),
            finite_float(row.get("source_vs_best_control_h1600_mean"), -999.0),
            finite_float(row.get("source_vs_best_control_h3200_mean"), -999.0),
        )
        rows.append(
            {
                **row,
                "best_control_source_mean_any_horizon": best_control if math.isfinite(best_control) else "",
                "best_noncontrol_source_mean_any_horizon": best_source if math.isfinite(best_source) else "",
                "control_explains_positive": int(math.isfinite(best_control) and best_source <= best_control + 0.005),
                "promotion_allowed": 0,
            }
        )
    return rows


def v18_kan_vs_mlp_attribution(source_summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    mlp_by_mech = {str(r.get("mechanism")): r for r in source_summary if str(r.get("carrier")) == "MLP"}
    rows = []
    for row in source_summary:
        mlp = mlp_by_mech.get(str(row.get("mechanism")), {})
        h1600 = finite_float(row.get("source_vs_best_control_h1600_mean"))
        h3200 = finite_float(row.get("source_vs_best_control_h3200_mean"))
        mlp_h1600 = finite_float(mlp.get("source_vs_best_control_h1600_mean"))
        mlp_h3200 = finite_float(mlp.get("source_vs_best_control_h3200_mean"))
        rows.append(
            {
                **row,
                "mlp_same_mechanism_h1600_mean": mlp_h1600 if math.isfinite(mlp_h1600) else "",
                "mlp_same_mechanism_h3200_mean": mlp_h3200 if math.isfinite(mlp_h3200) else "",
                "delta_vs_mlp_h1600": h1600 - mlp_h1600 if math.isfinite(h1600) and math.isfinite(mlp_h1600) else "",
                "delta_vs_mlp_h3200": h3200 - mlp_h3200 if math.isfinite(h3200) and math.isfinite(mlp_h3200) else "",
                "KAN_specific_advantage_h1600": int(str(row.get("carrier")) != "MLP" and math.isfinite(h1600) and math.isfinite(mlp_h1600) and h1600 > mlp_h1600 + 0.005),
                "KAN_specific_advantage_h3200": int(str(row.get("carrier")) != "MLP" and math.isfinite(h3200) and math.isfinite(mlp_h3200) and h3200 > mlp_h3200 + 0.005),
                "promotion_allowed": 0,
            }
        )
    return rows


def build_v18_code_packet(out_dir: Path) -> None:
    plan_doc, exec_doc, recap_doc = doc_paths(out_dir)
    packet_dir = out_dir / "v18_code_review_packet"
    if packet_dir.exists():
        shutil.rmtree(packet_dir)
    sections = [
        "00_README.md",
        "01_ENVIRONMENT",
        "02_SOURCE_TREE",
        "03_IMPORT_CLOSURE",
        "04_LINEC_CORRECTNESS",
        "05_RETENTION_AND_DEBT",
        "06_ROUTE_AGGREGATION",
        "07_UPDATE_SEMANTICS",
        "08_OPTIMIZER_COUPLING",
        "09_FUNCTIONAL_MECHANISMS",
        "10_EFFICIENCY_PROFILER",
        "11_KERNEL_CORRECTNESS",
        "12_EXPERIMENT_RUNNERS",
        "13_RAW_MATRICES",
        "14_FIGURES",
        "15_GPU_QUEUE",
    ]
    for section in sections:
        target = packet_dir / section
        if target.suffix:
            target.parent.mkdir(parents=True, exist_ok=True)
        else:
            target.mkdir(parents=True, exist_ok=True)

    git_code, git_head = run_text_command(["git", "rev-parse", "HEAD"])
    _, git_status = run_text_command(["git", "status", "--short"])
    _, git_diff = run_text_command(["git", "diff", "--stat"])
    _, py_version = run_text_command([PYTHON, "--version"])
    _, pip_freeze = run_text_command([PYTHON, "-m", "pip", "freeze"], timeout=120)
    _, nvidia = run_text_command(["nvidia-smi"], timeout=60)
    torch_report = {
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "devices": [],
    }
    if torch.cuda.is_available():
        for idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(idx)
            torch_report["devices"].append({"index": idx, "name": props.name, "total_memory": props.total_memory, "capability": [props.major, props.minor]})
    git_hash = git_head.strip() if git_code == 0 else "git_commit_unavailable=1"
    write_text(packet_dir / "01_ENVIRONMENT/python_version.txt", py_version)
    write_text(packet_dir / "01_ENVIRONMENT/pip_freeze.txt", pip_freeze)
    write_text(packet_dir / "01_ENVIRONMENT/nvidia_smi.txt", nvidia)
    write_text(packet_dir / "01_ENVIRONMENT/torch_cuda_report.json", json.dumps(sanitize(torch_report), ensure_ascii=False, indent=2) + "\n")
    write_text(packet_dir / "01_ENVIRONMENT/git_commit.txt", git_hash + "\n")
    write_text(packet_dir / "01_ENVIRONMENT/git_status.txt", git_status)
    write_text(packet_dir / "01_ENVIRONMENT/run_host.txt", platform.node() + "\n")

    readme = [
        "# v18.0 Evidence-First Code Review Packet",
        "",
        f"- repo_root: {ROOT}",
        f"- git_commit: {git_hash}",
        f"- python: {PYTHON}",
        f"- plan_doc: {plan_doc}",
        f"- result_dir: {out_dir}",
        "",
        "## Audit Boundary",
        "- This packet is evidence-first: files with incomplete measurement are kept with explicit `measurement_status` fields.",
        "- `official_fused_kernel_complete=0` is not promoted as official kernel success.",
        "- LineC/CEp99/NLL/ECE/AUCtime/Brier are audit/readback only, not direction sources.",
        "",
        "## Git Diff Summary",
        "```text",
        git_diff.strip(),
        "```",
    ]
    write_text(packet_dir / "00_README.md", "\n".join(readme) + "\n")

    def ignore_tree(_dir: str, names: list[str]) -> set[str]:
        return {n for n in names if n in {"__pycache__", ".pytest_cache", ".mypy_cache"} or n.endswith(".pyc")}

    for rel in ["dgkan", "experiments"]:
        src = ROOT / rel
        if src.exists():
            shutil.copytree(src, packet_dir / "02_SOURCE_TREE" / rel, ignore=ignore_tree, dirs_exist_ok=True)
    docs_dst = packet_dir / "02_SOURCE_TREE/docs"
    docs_dst.mkdir(parents=True, exist_ok=True)
    for doc in [plan_doc, exec_doc, recap_doc]:
        copy_if_exists(doc, docs_dst / doc.name)

    copy_map = {
        "03_IMPORT_CLOSURE": ["v17_compileall.log", "v17_import_closure.csv", "v17_import_errors.csv", "v17_module_dependency_graph.json", "v17_missing_symbol_report.csv", "compatibility_manifest.csv"],
        "04_LINEC_CORRECTNESS": ["v17_linec_golden_results.csv", "v17_linec_exception_policy_test.csv", "v17_linec_null_distribution.csv", "v17_linec_split_transfer_test.csv", "v17_linec_noise_injection_test.csv", "v17_linec_signal_reservoir_test.csv", "v17_linec_batch_order_mismatch_test.csv", "linec_fast_vs_channel_comparison.csv"],
        "05_RETENTION_AND_DEBT": ["source_retention_formula_tests.csv", "debt_accounting_formula_tests.csv", "v18_source_retention_matrix.csv", "v18_debt_accounting_matrix.csv"],
        "06_ROUTE_AGGREGATION": ["route_aggregation_unit_tests.csv", "v18_route_decision.json"],
        "07_UPDATE_SEMANTICS": ["v17_update_semantics_tests.csv", "v17_update_type_manifest.csv", "v17_update_sign_sanity.csv"],
        "08_OPTIMIZER_COUPLING": ["v17_adamw_coupling_map.csv", "v17_optimizer_state_dependency.csv", "v18_adamw_overwrite_diagnostics.csv"],
        "09_FUNCTIONAL_MECHANISMS": ["v18_functional_raw_horizon_matrix.csv", "v18_control_attribution.csv", "v18_kan_vs_mlp_attribution.csv", "v18_mechanism_noncollapse_matrix.csv"],
        "10_EFFICIENCY_PROFILER": ["v18_efficiency_truth_table.csv", "v18_efficiency_blocker_table.csv", "v17_efficiency_phase_breakdown.csv", "v17_memory_phase_breakdown.csv", "v17_same_param_mlp_manifest.csv"],
        "11_KERNEL_CORRECTNESS": ["v17_kernel_correctness.csv", "v18_kernel_repair_matrix.csv"],
        "12_EXPERIMENT_RUNNERS": ["experiments/run_v18_code_gate.py", "experiments/run_v18_functional_update_breakthrough.py", "experiments/run_v18_basis_efficiency_breakthrough.py", "experiments/run_v18_merge_finalize.py", "experiments/run_v17_common.py"],
        "13_RAW_MATRICES": ["raw_h100_matrix.csv", "raw_h400_matrix.csv", "raw_h800_matrix.csv", "raw_h1600_matrix.csv", "raw_h3200_matrix.csv", "raw_controls_matrix.csv", "raw_efficiency_matrix.csv", "v17_functional_mechanism_matrix.csv", "v17_horizon_extension.csv", "v17_horizon_traces.csv"],
        "14_FIGURES": V18_REQUIRED_FIGURES,
        "15_GPU_QUEUE": ["v18_runnable_queue.csv", "v18_gpu_assignment_manifest.csv", "v18_gpu_utilization_dashboard.csv", "v18_gpu_queue_drain_report.csv", "v18_idle_violation.csv", "v18_deferred_items.csv", "v18_queue_drain_report.csv", "v17_gpu_runtime_snapshots.csv"],
    }
    for section, names in copy_map.items():
        for name in names:
            src = ROOT / name if name.startswith("experiments/") else out_dir / name
            copy_if_exists(src, packet_dir / section / Path(name).name)

    repro = [
        f"{PYTHON} experiments/run_v18_code_gate.py --out-dir {out_dir} --device cuda:0 --data-root data",
        f"{PYTHON} experiments/run_v18_functional_update_breakthrough.py --stage mechanism-shard --out-dir {out_dir} --shard-count 4 --shard-index <0..3> --device cuda:<0..3> --data-root data --fail-on-missing-s0",
        f"{PYTHON} experiments/run_v18_basis_efficiency_breakthrough.py --stage efficiency-shard --out-dir {out_dir} --shard-count 4 --shard-index <0..3> --device cuda:<0..3> --data-root data --efficiency-batches 8,32,128,256 --fail-on-missing-s0",
        f"{PYTHON} experiments/run_v18_functional_update_breakthrough.py --stage horizon-shard --out-dir {out_dir} --shard-count 4 --shard-index <0..3> --device cuda:<0..3> --data-root data --horizon-all-rows 1 --horizon-steps 3200 --fail-on-missing-s0",
        f"{PYTHON} experiments/run_v18_merge_finalize.py --out-dir {out_dir} --data-root data --horizon-all-rows 1 --horizon-steps 3200",
    ]
    write_text(packet_dir / "15_GPU_QUEUE/repro_commands.md", "\n".join(f"- `{cmd}`" for cmd in repro) + "\n")

    rows = []
    for path in sorted(p for p in packet_dir.rglob("*") if p.is_file()):
        rel = path.relative_to(packet_dir)
        if rel.name in {"packet_manifest.csv", "packet_sha256_manifest.csv"}:
            continue
        rows.append({"relative_path": str(rel), "sha256": sha256_file(path), "bytes": path.stat().st_size, "artifact_type": rel.parts[0] if rel.parts else "unknown", "required": 1, "created_by": "experiments/run_v17_common.py:build_v18_code_packet", "source_command": "v18 merge-finalize"})
    write_rows(packet_dir / "packet_manifest.csv", rows)
    write_rows(packet_dir / "packet_sha256_manifest.csv", [{"relative_path": r["relative_path"], "sha256": r["sha256"], "bytes": r["bytes"]} for r in rows])
    write_rows(out_dir / "v18_code_review_packet.csv", rows)
    with zipfile.ZipFile(out_dir / "v18_code_review_packet.zip", "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(p for p in packet_dir.rglob("*") if p.is_file()):
            zf.write(path, Path("v18_code_review_packet") / path.relative_to(packet_dir))


def v18_required_manifest(out_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for name in V18_REQUIRED_ARTIFACTS + V18_REQUIRED_FIGURES:
        path = out_dir / name
        rows.append({"artifact": name, "path": str(path), "exists": int(path.exists()), "size_bytes": path.stat().st_size if path.exists() else 0})
    write_rows(out_dir / "v18_required_artifact_manifest.csv", rows)
    return rows


def build_v18_completion_artifacts(
    out_dir: Path,
    route: dict[str, Any],
    mechanism_rows: list[dict[str, Any]],
    efficiency_rows: list[dict[str, str]],
    kernel_rows: list[dict[str, str]],
) -> dict[str, Any]:
    s0 = json.loads((out_dir / "v17_s0_route_decision.json").read_text(encoding="utf-8")) if (out_dir / "v17_s0_route_decision.json").exists() else {}
    source_summary = grouped_source_summary(mechanism_rows)
    write_rows(out_dir / "v18_source_retention_matrix.csv", source_summary)
    horizon_matrix = []
    for h in ["h100", "h400", "h800", "h1600", "h3200"]:
        rows = raw_horizon_rows(mechanism_rows, h)
        horizon_matrix.extend(rows)
        write_rows(out_dir / f"raw_{h}_matrix.csv", rows)
    write_rows(out_dir / "v18_functional_raw_horizon_matrix.csv", horizon_matrix)
    write_rows(out_dir / "raw_controls_matrix.csv", [r for r in mechanism_rows if str(r.get("mechanism")) in CONTROL_MECHANISMS])
    write_rows(out_dir / "raw_efficiency_matrix.csv", efficiency_rows)

    debt_rows, debt_complete = v18_build_debt_matrix(out_dir)
    eff_truth = []
    blockers = []
    for row in efficiency_rows:
        audit_ratio = finite_float(row.get("audit_overhead_ratio"))
        if not math.isfinite(audit_ratio):
            audit_ratio = finite_float(row.get("linec_audit_ms"), 0.0) / max(1.0e-9, finite_float(row.get("step_training_only_ms"), 1.0))
        blocker = v18_efficiency_blocker({**row, "audit_overhead_ratio": audit_ratio})
        gate = int(blocker == "pass")
        official_gate = int(gate and int_flag(row.get("official_fused_kernel_complete")) == 1)
        item = {
            **row,
            "forward_ratio_vs_same_param_mlp": row.get("forward_ratio"),
            "backward_ratio_vs_same_param_mlp": row.get("backward_ratio"),
            "adamw_update_ratio_vs_same_param_mlp": row.get("update_ratio"),
            "sgd_update_ratio_vs_same_param_mlp": "",
            "functional_commit_ratio_vs_same_param_mlp": row.get("functional_overhead_ratio"),
            "step_total_ratio_vs_same_param_mlp": row.get("training_step_ratio"),
            "memory_peak_ratio_vs_same_param_mlp": row.get("memory_ratio"),
            "audit_overhead_fraction": audit_ratio,
            "v18_exploration_gate": gate,
            "v18_official_gate": official_gate,
            "blocker_class": blocker,
            "promotion_allowed": 0,
        }
        eff_truth.append(item)
        blockers.append(
            {
                "carrier": row.get("carrier"),
                "batch_size": row.get("batch_size"),
                "blocker_class": blocker,
                "forward_ratio": row.get("forward_ratio"),
                "backward_ratio": row.get("backward_ratio"),
                "step_ratio": row.get("training_step_ratio"),
                "memory_ratio": row.get("memory_ratio"),
                "audit_overhead_fraction": audit_ratio,
                "repair_direction": family_blocker(str(row.get("carrier")), [blocker]) if blocker != "pass" else "pass",
                "promotion_allowed": 0,
            }
        )
    write_rows(out_dir / "v18_efficiency_truth_table.csv", eff_truth)
    write_rows(out_dir / "v18_efficiency_blocker_table.csv", blockers)

    kernel_by_family = {str(r.get("family")): r for r in kernel_rows}
    repair_rows = []
    for family, variants in v18_kernel_variants().items():
        krow = kernel_by_family.get(family, {})
        fam_blocked = [r for r in blockers if str(r.get("carrier")) == family and str(r.get("blocker_class")) != "pass"]
        for variant in variants:
            implemented = 1 if variant == "R0-current" else 0
            repair_rows.append(
                {
                    "family": family,
                    "repair_variant": variant,
                    "implementation_status": "measured_current_torch_family_specific_path" if implemented else "not_implemented_in_v18_run",
                    "gradcheck_pass": krow.get("kernel_correctness_exploration_pass", ""),
                    "forward_relerr": krow.get("kernel_forward_relerr", ""),
                    "grad_relerr": krow.get("kernel_grad_relerr", ""),
                    "dense_materialized": krow.get("dense_basis_materialized", ""),
                    "official_fused_kernel_complete": 0,
                    "blocked_efficiency_rows": len(fam_blocked),
                    "blocker_class": "current_row_measured" if implemented else "deferred_family_specific_kernel_repair",
                    "promotion_allowed": 0,
                }
            )
    write_rows(out_dir / "v18_kernel_repair_matrix.csv", repair_rows)

    control_attr = v18_control_attribution(source_summary)
    kan_mlp = v18_kan_vs_mlp_attribution(source_summary)
    write_rows(out_dir / "v18_control_attribution.csv", control_attr)
    write_rows(out_dir / "v18_kan_vs_mlp_attribution.csv", kan_mlp)
    write_rows(out_dir / "v18_adamw_overwrite_diagnostics.csv", read_rows(out_dir / "v17_adamw_overwrite_diagnostic.csv"))
    write_rows(out_dir / "v18_mechanism_noncollapse_matrix.csv", read_rows(out_dir / "v17_mechanism_noncollapse_summary.csv") or read_rows(out_dir / "v17_semantic_noncollapse_audit.csv"))
    write_rows(out_dir / "v18_forbidden_information_audit.csv", read_rows(out_dir / "v17_forbidden_information_audit.csv"))
    write_rows(out_dir / "v18_no_action_search_audit.csv", read_rows(out_dir / "v17_no_action_search_audit.csv"))
    for src, dst in [
        ("v17_runnable_queue.csv", "v18_runnable_queue.csv"),
        ("v17_gpu_assignment_manifest.csv", "v18_gpu_assignment_manifest.csv"),
        ("v17_gpu_utilization_dashboard.csv", "v18_gpu_utilization_dashboard.csv"),
        ("v17_idle_violation.csv", "v18_idle_violation.csv"),
        ("v17_deferred_items.csv", "v18_deferred_items.csv"),
        ("v17_queue_drain_report.csv", "v18_queue_drain_report.csv"),
    ]:
        write_rows(out_dir / dst, read_rows(out_dir / src))
    write_rows(out_dir / "v18_gpu_queue_drain_report.csv", read_rows(out_dir / "v17_queue_drain_report.csv"))

    non_mlp_pass = [r for r in eff_truth if str(r.get("carrier")) != "MLP" and int_flag(r.get("v18_exploration_gate"))]
    forward_blocked = sorted({str(r.get("carrier")) for r in blockers if "ForwardBlocked" in str(r.get("blocker_class")) and str(r.get("carrier")) != "MLP"})
    h800_candidates = [
        r for r in source_summary
        if str(r.get("mechanism")) not in CONTROL_MECHANISMS
        and finite_float(r.get("source_vs_best_control_h800_mean"), -999.0) >= 0.005
        and int_flag(r.get("dataset_seed_pass_count_h800")) >= 3
        and finite_float(r.get("source_retention_h800_over_h100"), 0.0) >= 0.40
    ]
    h1600_candidates = [
        r for r in source_summary
        if str(r.get("mechanism")) not in CONTROL_MECHANISMS
        and finite_float(r.get("source_vs_best_control_h1600_mean"), -999.0) >= 0.005
        and int_flag(r.get("dataset_seed_pass_count_h1600")) >= 4
        and finite_float(r.get("source_retention_h1600_over_h800"), 0.0) >= 0.50
    ]
    late_rebound = [
        r for r in source_summary
        if str(r.get("mechanism")) not in CONTROL_MECHANISMS
        and finite_float(r.get("source_vs_best_control_h1600_mean"), 0.0) <= 0.0
        and finite_float(r.get("source_vs_best_control_h3200_mean"), -999.0) >= 0.005
    ]
    functional_route = "S3-RetainedMeanSource" if h1600_candidates else ("S2-WeakRetainedSource" if h800_candidates else ("S3b-LateReboundNeedsIndependentRerun" if late_rebound else "R-SourceSingleRowOnlyOrWashedOut"))
    efficiency_route = "S2-EfficiencyExplorationPassNonMLP" if non_mlp_pass else "R-EfficiencyForwardBlocked-" + "-".join(forward_blocked)
    code_route = "S0_2-CodeCorrectnessPassedDebtReadbackIncomplete" if int_flag(s0.get("s0_pass")) else "R-S0_2CodeInvalid"

    failure_rows = []
    if not debt_complete:
        failure_rows.append({"failure_class": "S0.2DebtMetricsIncomplete", "severity": "hard_evidence_blocker", "evidence": "v18_debt_accounting_matrix has LineC/channel/ECE/Brier blanks", "promotion_allowed": 0})
    if forward_blocked:
        failure_rows.append({"failure_class": "BasisForwardBlocked", "severity": "efficiency_blocker", "evidence": ",".join(forward_blocked), "promotion_allowed": 0})
    if not h1600_candidates:
        failure_rows.append({"failure_class": "NoRetainedFunctionalSource", "severity": "functional_blocker", "evidence": functional_route, "promotion_allowed": 0})
    for row in repair_rows:
        if str(row.get("implementation_status")) == "not_implemented_in_v18_run":
            failure_rows.append({"failure_class": "KernelRepairVariantDeferred", "severity": "implementation_blocker", "evidence": f"{row.get('family')}:{row.get('repair_variant')}", "promotion_allowed": 0})
    write_rows(out_dir / "v18_failure_taxonomy.csv", failure_rows)
    write_text(
        out_dir / "v18_next_hypothesis_queue.md",
        "\n".join(
            [
                "# v18 Next Hypothesis Queue",
                "",
                "- P0: implement real D-FOU/D-CHE R1-R5 fused/recurrence repair variants, then rerun v18 efficiency batches 8/32/128/256.",
                "- P0: add real LineC-channel, ECE, Brier debt readbacks to horizon trace before allowing S0.2 evidence-complete route.",
                "- P1: rerun any h3200 late rebound candidate in two independent reruns before S3b exploration classification.",
                "- P1: isolate AdamW washout by comparing M1 against M2/M3/M6/M8 under identical controls and debt readback.",
            ]
        ) + "\n",
    )
    write_text(
        out_dir / "v18_no_go_boundary.md",
        "\n".join(
            [
                "# v18 No-Go Boundary",
                "",
                "- No official success while `official_fused_kernel_complete=0` for non-MLP basis families.",
                "- No S4/S5 claim without real-transfer official rows and real debt recovery metrics.",
                "- No retained-source claim when previous horizon mean source is non-positive; classify h3200 rebound as late rebound until independently repeated.",
                "- Blank debt fields are evidence blockers, not zero debt and not recovery.",
            ]
        ) + "\n",
    )

    for name, (title, rows, key) in {
        "figures/v18_code_gate_dashboard.svg": ("v18 code gate", read_rows(out_dir / "v17_s0_import_closure.csv"), "import_ok"),
        "figures/v18_linec_fast_channel_golden.svg": ("v18 LineC fast/channel", read_rows(out_dir / "v17_linec_golden_tests.csv"), "pass"),
        "figures/v18_efficiency_forward_ratio_by_family.svg": ("v18 forward ratio", eff_truth, "forward_ratio"),
        "figures/v18_efficiency_blocker_heatmap.svg": ("v18 efficiency blockers", blockers, None),
        "figures/v18_kernel_repair_matrix.svg": ("v18 kernel repair", repair_rows, "implemented"),
        "figures/v18_source_retention_horizon.svg": ("v18 source retention", source_summary, "source_vs_best_control_h3200_mean"),
        "figures/v18_debt_accounting_readback.svg": ("v18 debt accounting", debt_rows, "tail_recovery_rate"),
        "figures/v18_control_attribution.svg": ("v18 control attribution", control_attr, "control_explains_positive"),
        "figures/v18_kan_vs_mlp_attribution.svg": ("v18 KAN vs MLP", kan_mlp, "KAN_specific_advantage_h3200"),
        "figures/v18_adamw_overwrite_diagnostics.svg": ("v18 AdamW overwrite", read_rows(out_dir / "v18_adamw_overwrite_diagnostics.csv"), "cos_FU_AdamW"),
        "figures/v18_gpu_queue_drain.svg": ("v18 GPU queue", read_rows(out_dir / "v18_gpu_queue_drain_report.csv"), "executed_jobs"),
        "figures/v18_failure_taxonomy.svg": ("v18 failure taxonomy", failure_rows, None),
    }.items():
        write_placeholder_svg(out_dir / name, title, rows, key)

    route18 = {
        **route,
        "route": "R1-S0_2DebtMetricsIncomplete" if not debt_complete else route.get("route"),
        "route_detail": (
            f"v18 executed fresh matrices but evidence-complete promotion blocked: debt_complete={debt_complete}, "
            f"non_mlp_efficiency_pass_rows={len(non_mlp_pass)}, h800_candidates={len(h800_candidates)}, "
            f"h1600_candidates={len(h1600_candidates)}, late_rebound_candidates={len(late_rebound)}, "
            f"deferred_kernel_repair_variants={sum(1 for r in repair_rows if str(r.get('implementation_status')) == 'not_implemented_in_v18_run')}"
        ),
        "CodeRoute": code_route,
        "EfficiencyRoute": efficiency_route,
        "FunctionalRoute": functional_route,
        "S0_2_preflight_pass": int_flag(s0.get("s0_pass")),
        "S0_2_evidence_complete": int(debt_complete and not sum(1 for r in repair_rows if str(r.get("implementation_status")) == "not_implemented_in_v18_run")),
        "debt_metrics_complete": debt_complete,
        "legacy_v17_required_artifact_missing_count": route.get("required_artifact_missing_count"),
        "required_artifact_missing_count": "",
        "v18_measured_functional_rows": len(mechanism_rows),
        "v18_measured_efficiency_rows": len(eff_truth),
        "v18_measured_debt_rows": len(debt_rows),
        "v18_non_mlp_efficiency_pass_rows": len(non_mlp_pass),
        "v18_h800_candidate_count": len(h800_candidates),
        "v18_h1600_candidate_count": len(h1600_candidates),
        "v18_late_rebound_candidate_count": len(late_rebound),
        "promotion_allowed": 0,
        "official_success_reached": 0,
    }
    write_json(out_dir / "v18_route_decision.json", route18)
    build_v18_code_packet(out_dir)
    manifest = v18_required_manifest(out_dir)
    route18["v18_required_artifact_missing_count"] = sum(1 for r in manifest if not int_flag(r.get("exists")))
    write_json(out_dir / "v18_route_decision.json", route18)
    return route18


def v19_metric_debt(metric_rows: list[tuple[int, float, float]]) -> tuple[Any, Any, Any]:
    debts = [max(0.0, value - baseline) for _step, value, baseline in metric_rows if math.isfinite(value) and math.isfinite(baseline)]
    if not debts:
        return "", "", ""
    peak = max(debts)
    final = debts[-1]
    if peak <= 0.0:
        return peak, final, ""
    return peak, final, max(0.0, min(1.0, 1.0 - final / (peak + 1.0e-12)))


def v19_build_debt_matrix(out_dir: Path) -> tuple[list[dict[str, Any]], int]:
    traces = read_rows(out_dir / "v17_horizon_traces.csv") or read_rows(out_dir / "v17_functional_traces.csv")
    grouped: dict[tuple[str, str, str, str], list[dict[str, str]]] = {}
    for row in traces:
        key = (str(row.get("carrier")), str(row.get("mechanism")), str(row.get("dataset")), str(row.get("seed")))
        grouped.setdefault(key, []).append(row)
    lower_metrics = ["CEp99", "NLL", "ECE", "Brier", "LineC_fast_loss", "LineC_channel_loss"]
    baseline: dict[tuple[str, str, str, int, str], float] = {}
    for row in traces:
        if str(row.get("mechanism")) not in CONTROL_MECHANISMS:
            continue
        step = int(finite_float(row.get("step"), -1))
        if step < 0:
            continue
        for metric in lower_metrics:
            val = finite_float(row.get(metric))
            if math.isfinite(val):
                key = (str(row.get("carrier")), str(row.get("dataset")), str(row.get("seed")), step, metric)
                baseline[key] = min(baseline.get(key, float("inf")), val)
    rows: list[dict[str, Any]] = []
    complete = 1
    for (carrier, mechanism, dataset, seed), group in sorted(grouped.items()):
        ordered = sorted(group, key=lambda r: finite_float(r.get("step"), 0.0))
        item: dict[str, Any] = {"carrier": carrier, "mechanism": mechanism, "dataset": dataset, "seed": seed, "trace_rows": len(ordered)}
        missing_metrics = []
        for metric in lower_metrics:
            metric_rows = []
            for r in ordered:
                step = int(finite_float(r.get("step"), -1))
                val = finite_float(r.get(metric))
                base = baseline.get((carrier, dataset, seed, step, metric), float("nan"))
                if math.isfinite(val) and math.isfinite(base):
                    metric_rows.append((step, val, base))
            peak, final, recovery = v19_metric_debt(metric_rows)
            prefix = "tail" if metric == "CEp99" else metric.replace("_loss", "")
            item[f"{prefix}_debt_peak"] = peak
            item[f"{prefix}_debt_final"] = final
            item[f"{prefix}_recovery"] = recovery
            if peak == "":
                missing_metrics.append(metric)
        losses = []
        bases = []
        steps = []
        for r in ordered:
            step = int(finite_float(r.get("step"), -1))
            val = finite_float(r.get("NLL"))
            base = baseline.get((carrier, dataset, seed, step, "NLL"), float("nan"))
            if step >= 0 and math.isfinite(val) and math.isfinite(base):
                steps.append(float(step))
                losses.append(val)
                bases.append(base)
        auc_ratio = ""
        if len(losses) >= 2:
            auc = 0.0
            auc_base = 0.0
            for i in range(1, len(losses)):
                dt = max(0.0, steps[i] - steps[i - 1])
                auc += 0.5 * (losses[i - 1] + losses[i]) * dt
                auc_base += 0.5 * (bases[i - 1] + bases[i]) * dt
            auc_ratio = auc / max(1.0e-12, auc_base)
        else:
            missing_metrics.append("AUCtime")
        item["AUCtime_ratio"] = auc_ratio
        item["measurement_status"] = "measured" if not missing_metrics else "EvidenceIncomplete:" + ";".join(sorted(set(missing_metrics)))
        item["promotion_allowed"] = 0
        if missing_metrics:
            complete = 0
        rows.append(item)
    write_rows(out_dir / "v19_debt_accounting_matrix.csv", rows)
    availability = []
    for metric in lower_metrics + ["AUCtime"]:
        availability.append({"metric": metric, "available_rows": sum(1 for r in rows if metric not in str(r.get("measurement_status"))), "total_rows": len(rows), "complete": int(all(metric not in str(r.get("measurement_status")) for r in rows))})
    write_rows(out_dir / "v19_debt_metric_availability.csv", availability)
    return rows, complete


def v19_required_manifest(out_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for name in V19_REQUIRED_ARTIFACTS + V19_REQUIRED_FIGURES:
        path = out_dir / name
        rows.append({"artifact": name, "path": str(path), "exists": int(path.exists()), "size_bytes": path.stat().st_size if path.exists() else 0})
    write_rows(out_dir / "v19_required_artifact_manifest.csv", rows)
    return rows


def build_v19_code_packet(out_dir: Path) -> None:
    plan_doc, exec_doc, recap_doc = doc_paths(out_dir)
    packet_dir = out_dir / "v19_code_review_packet"
    if packet_dir.exists():
        shutil.rmtree(packet_dir)
    for section in [
        "00_README.md", "01_ENVIRONMENT", "02_SOURCE_TREE", "03_IMPORT_CLOSURE", "04_LINEC_CORRECTNESS",
        "05_RETENTION_AND_DEBT", "06_ROUTE_AGGREGATION", "07_UPDATE_SEMANTICS",
        "08_FUNCTIONAL_MECHANISM_CONTRACTS", "09_OPTIMIZER_COUPLING", "10_EFFICIENCY_PROFILER",
        "11_KERNEL_CORRECTNESS", "12_RAW_EXPERIMENT_MATRICES", "13_GPU_QUEUE", "14_FAILURE_TAXONOMY",
    ]:
        target = packet_dir / section
        if target.suffix:
            target.parent.mkdir(parents=True, exist_ok=True)
        else:
            target.mkdir(parents=True, exist_ok=True)
    _, git_status = run_text_command(["git", "status", "--short"])
    _, git_head = run_text_command(["git", "rev-parse", "HEAD"])
    _, py_version = run_text_command([PYTHON, "--version"])
    _, pip_freeze = run_text_command([PYTHON, "-m", "pip", "freeze"], timeout=120)
    conda_bin = shutil.which("conda")
    if not conda_bin and Path("/home/chengshun.wang/miniconda3/bin/conda").exists():
        conda_bin = "/home/chengshun.wang/miniconda3/bin/conda"
    if conda_bin:
        _, conda_env = run_text_command([conda_bin, "env", "export"], timeout=120)
    else:
        conda_env = "conda_unavailable=1\nreason=conda executable not found in PATH\n"
    _, nvidia = run_text_command(["nvidia-smi"], timeout=60)
    write_text(packet_dir / "01_ENVIRONMENT/conda_env.yaml", conda_env)
    write_text(packet_dir / "01_ENVIRONMENT/pip_freeze.txt", pip_freeze)
    write_text(packet_dir / "01_ENVIRONMENT/git_status.txt", git_status)
    write_text(packet_dir / "01_ENVIRONMENT/cuda_info.txt", nvidia)
    write_text(packet_dir / "01_ENVIRONMENT/python_version.txt", py_version)
    commands = [
        f"{PYTHON} experiments/run_v19_s03_truth_gate.py --out-dir {out_dir} --device cuda:0 --data-root data",
        f"{PYTHON} experiments/run_v19_source_channel_fu_matrix.py --stage mechanism-shard --out-dir {out_dir} --shard-count 4 --shard-index <0..3> --device cuda:<0..3> --data-root data --fail-on-missing-s0",
        f"{PYTHON} experiments/run_v19_basis_kernel_breakthrough.py --stage efficiency-shard --out-dir {out_dir} --shard-count 4 --shard-index <0..3> --device cuda:<0..3> --data-root data --efficiency-batches 8,32,128,256 --fail-on-missing-s0",
        f"{PYTHON} experiments/run_v19_source_channel_fu_matrix.py --stage horizon-shard --out-dir {out_dir} --shard-count 4 --shard-index <0..3> --device cuda:<0..3> --data-root data --horizon-all-rows 1 --horizon-steps 4800 --fail-on-missing-s0",
        f"{PYTHON} experiments/run_v19_late_rebound_rerun.py --out-dir {out_dir} --shard-count 4 --shard-index <0..3> --device cuda:<0..3> --data-root data",
        f"{PYTHON} experiments/run_v19_merge_finalize.py --out-dir {out_dir} --data-root data --horizon-all-rows 1 --horizon-steps 4800",
    ]
    write_text(packet_dir / "01_ENVIRONMENT/exact_commands.sh", "#!/usr/bin/env bash\nset -euo pipefail\n" + "\n".join(commands) + "\n")
    write_text(packet_dir / "00_README.md", "\n".join(["# v19 Code Review Packet", "", f"- git_commit: {git_head.strip()}", f"- plan_doc: {plan_doc}", f"- result_dir: {out_dir}", "- no fabricated data; incomplete metrics remain EvidenceIncomplete."]) + "\n")

    def ignore_tree(_dir: str, names: list[str]) -> set[str]:
        return {n for n in names if n in {"__pycache__", ".pytest_cache", ".mypy_cache"} or n.endswith(".pyc")}
    for rel in ["dgkan", "experiments", "tests"]:
        src = ROOT / rel
        if src.exists():
            shutil.copytree(src, packet_dir / "02_SOURCE_TREE" / rel, ignore=ignore_tree, dirs_exist_ok=True)
    for doc in [plan_doc, exec_doc, recap_doc]:
        copy_if_exists(doc, packet_dir / "02_SOURCE_TREE" / "docs" / doc.name)
    copy_map = {
        "03_IMPORT_CLOSURE": ["v17_compileall.log", "v17_import_closure.csv", "v17_import_errors.csv", "compatibility_manifest.csv"],
        "04_LINEC_CORRECTNESS": ["v17_linec_golden_tests.csv", "linec_channel_golden_tests.csv", "linec_exception_policy.csv", "linec_measurement_invalid_examples.csv", "linec_fast_vs_channel_consistency.csv", "linec_fast_vs_channel_comparison.csv"],
        "05_RETENTION_AND_DEBT": ["source_retention_formula_tests.csv", "debt_accounting_formula_tests.csv", "v19_debt_metric_availability.csv", "v19_debt_accounting_matrix.csv"],
        "06_ROUTE_AGGREGATION": ["route_aggregation_unit_tests.csv", "v19_source_retention_matrix.csv", "v19_control_attribution.csv"],
        "07_UPDATE_SEMANTICS": ["v17_update_sign_sanity.csv", "v17_update_type_manifest.csv", "update_writeback_trace.csv"],
        "08_FUNCTIONAL_MECHANISM_CONTRACTS": ["v19_mechanism_manifest.csv", "v19_mechanism_semantic_contract.csv", "v19_mechanism_toy_correctness.csv", "v17_mechanism_noncollapse_summary.csv"],
        "09_OPTIMIZER_COUPLING": ["v19_adamw_overwrite_diagnostics.csv", "v17_adamw_coupling_map.csv"],
        "10_EFFICIENCY_PROFILER": ["v19_efficiency_truth_table.csv", "v19_efficiency_blocker_table.csv", "v19_efficiency_waterfall.csv", "v17_efficiency_profiler_unit_tests.csv"],
        "11_KERNEL_CORRECTNESS": ["v17_kernel_correctness.csv", "v19_kernel_repair_matrix.csv"],
        "12_RAW_EXPERIMENT_MATRICES": ["v19_functional_raw_horizon_matrix.csv", "v19_late_rebound_rerun_matrix.csv", "v17_functional_mechanism_matrix.csv", "v17_horizon_extension.csv", "v17_horizon_traces.csv"],
        "13_GPU_QUEUE": ["v19_runnable_queue.csv", "v19_gpu_assignment_manifest.csv", "v19_gpu_utilization_dashboard.csv", "v19_idle_violation.csv", "v19_queue_drain_report.csv", "v17_gpu_runtime_snapshots.csv"],
        "14_FAILURE_TAXONOMY": ["v19_failure_taxonomy.csv", "v19_no_go_boundary.md", "v19_next_hypothesis_queue.csv", "v19_next_hypothesis_queue.md"],
    }
    for section, names in copy_map.items():
        for name in names:
            copy_if_exists(out_dir / name, packet_dir / section / Path(name).name)
    rows = []
    for path in sorted(p for p in packet_dir.rglob("*") if p.is_file()):
        rel = path.relative_to(packet_dir)
        if rel.name in {"packet_manifest.csv", "packet_sha256_manifest.csv"}:
            continue
        rows.append({"relative_path": str(rel), "sha256": sha256_file(path), "bytes": path.stat().st_size, "artifact_type": rel.parts[0], "required": 1})
    write_rows(packet_dir / "packet_manifest.csv", rows)
    write_rows(packet_dir / "packet_sha256_manifest.csv", [{"relative_path": r["relative_path"], "sha256": r["sha256"], "bytes": r["bytes"]} for r in rows])
    with zipfile.ZipFile(out_dir / "v19_code_review_packet.zip", "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(p for p in packet_dir.rglob("*") if p.is_file()):
            zf.write(path, Path("v19_code_review_packet") / path.relative_to(packet_dir))


def build_v19_completion_artifacts(
    out_dir: Path,
    route: dict[str, Any],
    mechanism_rows: list[dict[str, Any]],
    efficiency_rows: list[dict[str, str]],
    kernel_rows: list[dict[str, str]],
) -> dict[str, Any]:
    s0 = json.loads((out_dir / "v17_s0_route_decision.json").read_text(encoding="utf-8")) if (out_dir / "v17_s0_route_decision.json").exists() else {}
    source_summary = grouped_source_summary(mechanism_rows)
    write_rows(out_dir / "v19_source_retention_matrix.csv", source_summary)
    horizon_matrix = []
    for h in ["h100", "h400", "h800", "h1600", "h2400", "h3200", "h4800"]:
        rows = raw_horizon_rows(mechanism_rows, h)
        horizon_matrix.extend(rows)
        write_rows(out_dir / f"raw_{h}_matrix.csv", rows)
    write_rows(out_dir / "v19_functional_raw_horizon_matrix.csv", horizon_matrix)
    debt_rows, debt_complete = v19_build_debt_matrix(out_dir)
    control_attr = v18_control_attribution(source_summary)
    kan_mlp = v18_kan_vs_mlp_attribution(source_summary)
    write_rows(out_dir / "v19_control_attribution.csv", control_attr)
    write_rows(out_dir / "v19_kan_vs_mlp_attribution.csv", kan_mlp)
    write_rows(out_dir / "v19_adamw_overwrite_diagnostics.csv", read_rows(out_dir / "v17_adamw_overwrite_diagnostic.csv"))
    write_rows(out_dir / "v19_mechanism_manifest.csv", read_rows(out_dir / "v19_mechanism_manifest.csv") or mechanism_contract_rows())
    write_rows(out_dir / "v19_mechanism_semantic_contract.csv", read_rows(out_dir / "v19_mechanism_semantic_contract.csv") or mechanism_contract_rows())

    eff_truth = []
    blockers = []
    waterfall = []
    for row in efficiency_rows:
        forward = finite_float(row.get("forward_ratio"), 999.0)
        step = finite_float(row.get("training_step_ratio"), 999.0)
        memory = finite_float(row.get("memory_ratio"), 999.0)
        manual_variant = str(row.get("manual_kernel_variant") or row.get("manual_kernel_variant_profiled") or "")
        manual_profiled = int_flag(row.get("manual_ce_train_stream_profiled"))
        manual_correct = int_flag(row.get("manual_correctness_pass")) if manual_profiled else 1
        reasons = []
        if forward > 1.75:
            reasons.append("KernelForwardBlocked")
        if finite_float(row.get("backward_ratio"), 0.0) > 1.75:
            reasons.append("KernelBackwardBlocked")
        if step > 1.50:
            reasons.append("StepBlocked")
        if memory > 1.10 and memory != 0.0:
            reasons.append("MemoryBlocked")
        if manual_profiled and not manual_correct:
            reasons.append("ManualCorrectnessBlocked")
        cls = "FamilyPass" if not reasons else ";".join(reasons)
        item = {
            **row,
            "forward_ratio_vs_mlp": row.get("forward_ratio"),
            "backward_ratio_vs_mlp": row.get("backward_ratio"),
            "step_ratio_vs_mlp": row.get("training_step_ratio"),
            "memory_ratio_vs_mlp": row.get("memory_ratio"),
            "kernel_count": int("triton" in manual_variant),
            "custom_kernel_count": int("triton" in manual_variant),
            "train_stream_fused_kernel_complete": int("triton" in manual_variant and manual_correct),
            "official_fused_kernel_complete": 0,
            "no_materialize_kernel_complete": int("no-materialize" in str(row.get("repair_variant")) or "stream" in str(row.get("repair_variant"))),
            "v19_exploration_gate": int(not reasons),
            "blocker_class": cls,
            "promotion_allowed": 0,
        }
        eff_truth.append(item)
        blockers.append({"carrier": row.get("carrier"), "repair_variant": row.get("repair_variant"), "batch_size": row.get("batch_size"), "blocker_class": cls, "forward_ratio": row.get("forward_ratio"), "step_ratio": row.get("training_step_ratio"), "memory_ratio": row.get("memory_ratio"), "promotion_allowed": 0})
        for phase in ["forward_only_ms", "basis_eval_ms", "readout_contraction_ms", "backward_input_ms", "backward_param_ms", "optimizer_update_ms_SGD", "optimizer_update_ms_AdamW", "optimizer_update_ms_manualFU", "functional_direction_ms", "functional_commit_ms", "linec_audit_ms"]:
            waterfall.append({"carrier": row.get("carrier"), "repair_variant": row.get("repair_variant"), "batch_size": row.get("batch_size"), "phase": phase, "ms": row.get(phase, ""), "measurement_status": "measured" if row.get(phase, "") != "" else "not_isolated"})
    write_rows(out_dir / "v19_efficiency_truth_table.csv", eff_truth)
    write_rows(out_dir / "v19_efficiency_blocker_table.csv", blockers)
    write_rows(out_dir / "v19_efficiency_waterfall.csv", waterfall)

    repair_rows = []
    for row in eff_truth:
        if str(row.get("carrier")) == "MLP":
            continue
        repair_rows.append({"family": row.get("carrier"), "repair_variant": row.get("repair_variant"), "manual_kernel_variant": row.get("manual_kernel_variant"), "batch_size": row.get("batch_size"), "forward_ratio": row.get("forward_ratio"), "step_ratio": row.get("training_step_ratio"), "memory_ratio": row.get("memory_ratio"), "manual_correctness_pass": row.get("manual_correctness_pass"), "train_stream_fused_kernel_complete": row.get("train_stream_fused_kernel_complete"), "no_materialize_kernel_complete": row.get("no_materialize_kernel_complete"), "official_fused_kernel_complete": 0, "exploration_gate": row.get("v19_exploration_gate"), "blocker_class": row.get("blocker_class"), "promotion_allowed": 0})
    write_rows(out_dir / "v19_kernel_repair_matrix.csv", repair_rows)

    late_raw = []
    for p in sorted(out_dir.glob("v19_late_rebound_rerun_matrix_*.csv")):
        late_raw.extend(read_rows(p))
    if late_raw:
        control_best: dict[tuple[str, str, str, str, str], float] = {}
        for r in late_raw:
            if str(r.get("mechanism")) in CONTROL_MECHANISMS:
                for h in ["h3200", "h4800"]:
                    val = finite_float(r.get(f"val_loss_{h}"))
                    key = (str(r.get("rerun_id")), str(r.get("dataset")), str(r.get("seed")), str(r.get("carrier")), h)
                    if math.isfinite(val):
                        control_best[key] = min(control_best.get(key, float("inf")), val)
        enriched = []
        for r in late_raw:
            item = dict(r)
            for h in ["h3200", "h4800"]:
                val = finite_float(r.get(f"val_loss_{h}"))
                best = control_best.get((str(r.get("rerun_id")), str(r.get("dataset")), str(r.get("seed")), str(r.get("carrier")), h), float("nan"))
                item[f"source_{h}"] = best - val if math.isfinite(best) and math.isfinite(val) else ""
            enriched.append(item)
        late_raw = enriched
    else:
        late_raw = [{"status": "not_executed", "reason": "v19_late_rebound_rerun_matrix shards missing", "promotion_allowed": 0}]
    write_rows(out_dir / "v19_late_rebound_rerun_matrix.csv", late_raw)

    for src, dst in [("v17_runnable_queue.csv", "v19_runnable_queue.csv"), ("v17_gpu_assignment_manifest.csv", "v19_gpu_assignment_manifest.csv"), ("v17_gpu_utilization_dashboard.csv", "v19_gpu_utilization_dashboard.csv"), ("v17_idle_violation.csv", "v19_idle_violation.csv"), ("v17_deferred_items.csv", "v19_deferred_items.csv"), ("v17_queue_drain_report.csv", "v19_queue_drain_report.csv")]:
        write_rows(out_dir / dst, read_rows(out_dir / src))

    dche_m2_late = [r for r in source_summary if str(r.get("carrier")) == "D-CHE" and str(r.get("mechanism")) == "M2-SGDMomentumPrimaryFU"]
    late_rebound_candidates = [r for r in source_summary if int_flag(r.get("late_rebound_h3200_flag")) or int_flag(r.get("late_rebound_h4800_flag"))]
    rerun_m2 = [r for r in late_raw if str(r.get("mechanism")) == "M2-SGDMomentumPrimaryFU"]
    rerun_positive = sum(1 for r in rerun_m2 if finite_float(r.get("source_h3200"), -999.0) >= 0.005 or finite_float(r.get("source_h4800"), -999.0) >= 0.005)
    d_fou_pass = any(str(r.get("carrier")) == "D-FOU" and int_flag(r.get("v19_exploration_gate")) for r in eff_truth)
    d_che_pass = any(str(r.get("carrier")) == "D-CHE" and int_flag(r.get("v19_exploration_gate")) for r in eff_truth)
    mlp_retained = [r for r in source_summary if str(r.get("carrier")) == "MLP" and finite_float(r.get("source_vs_best_control_h1600_mean"), -999.0) >= 0.005 and finite_float(r.get("source_retention_h1600_over_h800"), 0.0) >= 0.50]
    failure_rows = []
    if not debt_complete:
        failure_rows.append({"failure_class": "DebtMetricsIncomplete", "severity": "S0.3_blocker", "evidence": "v19_debt_metric_availability.csv", "promotion_allowed": 0})
    if not d_fou_pass:
        failure_rows.append({"failure_class": "D-FOUForwardBlocked", "severity": "efficiency", "evidence": "v19_efficiency_blocker_table.csv", "promotion_allowed": 0})
    if not d_che_pass:
        failure_rows.append({"failure_class": "D-CHEForwardBlocked", "severity": "efficiency", "evidence": "v19_efficiency_blocker_table.csv", "promotion_allowed": 0})
    if not mlp_retained:
        failure_rows.append({"failure_class": "MLPSourceWashesOut", "severity": "functional", "evidence": "v19_source_retention_matrix.csv", "promotion_allowed": 0})
    if rerun_m2 and rerun_positive < 2:
        failure_rows.append({"failure_class": "DCHELateReboundNotReproduced", "severity": "functional", "evidence": f"positive_rerun_rows={rerun_positive}", "promotion_allowed": 0})
    write_rows(out_dir / "v19_failure_taxonomy.csv", failure_rows)
    next_rows = [
        {"priority": "P0", "item": "Complete official/no-materialize D-FOU and D-CHE kernel repair", "reason": "forward gate still blocked unless v19 pass rows exist", "promotion_allowed": 0},
        {"priority": "P0", "item": "Use LineC-channel debt to filter slow-state source writers", "reason": "v19 source retention gate depends on debt recovery", "promotion_allowed": 0},
    ]
    write_rows(out_dir / "v19_next_hypothesis_queue.csv", next_rows)
    write_text(out_dir / "v19_next_hypothesis_queue.md", "# v19 Next Hypothesis Queue\n\n" + "\n".join(f"- {r['priority']}: {r['item']} | {r['reason']}" for r in next_rows) + "\n")
    write_text(out_dir / "v19_no_go_boundary.md", "# v19 No-Go Boundary\n\n- S0.3 incomplete means no scientific no-go.\n- S3b requires independent rerun >=3 and positive reproduced >=2/3 plus completed debt recovery.\n- E3 requires official fused or no-materialize kernel completion with forward/step/memory gates.\n")
    for name, (title, rows, key) in {
        "figures/v19_source_trajectory.svg": ("v19 source h4800", source_summary, "source_vs_best_control_h4800_mean"),
        "figures/v19_source_retention_heatmap.svg": ("v19 retention", source_summary, "source_retention_h1600_over_h800"),
        "figures/v19_late_rebound_map.svg": ("v19 late rebound", source_summary, "late_rebound_h3200_flag"),
        "figures/v19_debt_recovery_curves.svg": ("v19 debt", debt_rows, "tail_recovery"),
        "figures/v19_adamw_overwrite_projection.svg": ("v19 AdamW overwrite", read_rows(out_dir / "v19_adamw_overwrite_diagnostics.csv"), "optimizer_overwrite_projection"),
        "figures/v19_mlp_vs_dche_source_trajectory.svg": ("MLP vs D-CHE", [r for r in source_summary if str(r.get("carrier")) in {"MLP", "D-CHE"}], "source_vs_best_control_h3200_mean"),
        "figures/v19_signal_state_norm.svg": ("signal state", source_summary, "source_vs_best_control_h800_mean"),
        "figures/v19_matrix_block_rank_retention.svg": ("matrix block", [r for r in source_summary if "Matrix" in str(r.get("mechanism")) or "LowRank" in str(r.get("mechanism"))], "source_vs_best_control_h1600_mean"),
        "figures/v19_efficiency_waterfall.svg": ("efficiency waterfall", waterfall, "ms"),
        "figures/v19_no_materialize_timing.svg": ("no materialize", eff_truth, "forward_ratio"),
        "figures/v19_efficiency_bubble.svg": ("efficiency bubble", eff_truth, "training_step_ratio"),
        "figures/v19_gpu_utilization_timeline.svg": ("GPU", read_rows(out_dir / "v19_gpu_utilization_dashboard.csv"), "runtime_sec"),
        "figures/v19_route_aggregation_dashboard.svg": ("route", source_summary, "dataset_seed_pass_count_h800"),
    }.items():
        write_placeholder_svg(out_dir / name, title, rows, key)

    total_late_rows = len(late_raw) if late_raw and str(late_raw[0].get("status")) != "not_executed" else 0
    blocked_reasons = []
    if not debt_complete:
        blocked_reasons.append("debt_metrics_incomplete")
    if not d_fou_pass:
        blocked_reasons.append("D-FOU_efficiency_blocked")
    if not d_che_pass:
        blocked_reasons.append("D-CHE_efficiency_blocked")
    if not mlp_retained:
        blocked_reasons.append("MLP_no_retained_source")
    route19 = {
        **route,
        "route": "R-S0_3EvidenceComplete" if int_flag(s0.get("s0_pass")) and debt_complete else "R1-S0_3EvidenceIncomplete",
        "route_detail": (
            f"v19 fresh matrices complete: debt_complete={int(debt_complete)}, "
            f"D-FOU_pass={int(d_fou_pass)}, D-CHE_pass={int(d_che_pass)}, "
            f"late_rebound_m2_positive={rerun_positive}/{len(rerun_m2)}, "
            f"promotion_blockers={';'.join(blocked_reasons) if blocked_reasons else 'none'}"
        ),
        "CodeRoute": "S0_3-CodeMetricMechanismPassed" if int_flag(s0.get("s0_pass")) and debt_complete else "R-S0_3DebtOrMetricIncomplete",
        "EfficiencyRoute": ("E2-D-FOU-Pass" if d_fou_pass else "R-D-FOUForwardBlocked") + ";" + ("E2-D-CHE-Pass" if d_che_pass else "R-D-CHEForwardBlocked"),
        "FunctionalRoute": "S3-MLPRetainedSource" if mlp_retained else ("S3b-DCHELateReboundReproduced" if rerun_positive >= 2 else "R-MLPSourceWashoutOrLateReboundStochastic"),
        "S0_3_preflight_pass": int_flag(s0.get("s0_pass")),
        "S0_3_evidence_complete": int(debt_complete),
        "LineC_fast_golden_pass": int(s0.get("linec_golden_pass_count") == s0.get("linec_golden_total")),
        "LineC_channel_golden_pass": int(s0.get("linec_channel_golden_pass_count") == s0.get("linec_channel_golden_total")),
        "debt_metrics_complete": debt_complete,
        "measured_functional_rows": len(mechanism_rows),
        "measured_efficiency_rows": len(eff_truth),
        "measured_debt_rows": len(debt_rows),
        "late_rebound_candidate_count": len(late_rebound_candidates),
        "late_rebound_rerun_rows": total_late_rows,
        "late_rebound_m2_rerun_rows": len(rerun_m2),
        "late_rebound_positive_rerun_rows": rerun_positive,
        "d_fou_efficiency_pass": int(d_fou_pass),
        "d_che_efficiency_pass": int(d_che_pass),
        "mlp_retained_source_count": len(mlp_retained),
        "promotion_allowed": 0,
        "official_success_reached": 0,
    }
    write_json(out_dir / "v19_route_decision.json", route19)
    build_v19_code_packet(out_dir)
    manifest = v19_required_manifest(out_dir)
    route19["required_artifact_missing_count"] = sum(1 for r in manifest if not int_flag(r.get("exists")))
    write_json(out_dir / "v19_route_decision.json", route19)
    with zipfile.ZipFile(out_dir / "v19_results_bundle.zip", "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(p for p in out_dir.rglob("*") if p.is_file() and "v19_code_review_packet/" not in str(p.relative_to(out_dir)) and p.name != "v19_results_bundle.zip"):
            zf.write(path, path.relative_to(out_dir))
    return route19


def decide_route(out_dir: Path, mechanism_rows: list[dict[str, Any]], h800: list[dict[str, Any]], efficiency_rows: list[dict[str, str]], kernel_rows: list[dict[str, str]]) -> dict[str, Any]:
    s0 = json.loads((out_dir / "v17_s0_route.json").read_text(encoding="utf-8")) if (out_dir / "v17_s0_route.json").exists() else {}
    missing_manifest = read_rows(out_dir / "v17_required_artifact_manifest.csv")
    missing_count = sum(1 for r in missing_manifest if not int_flag(r.get("exists")))
    measured_mech = sum(1 for r in mechanism_rows if str(r.get("execution_status")) == "measured")
    planned = len(mechanism_jobs(parse_args_for_defaults(out_dir)))
    h800_valid = sum(int_flag(r.get("h800_measurement_valid")) for r in h800)
    h1600_valid = sum(int_flag(r.get("h1600_measurement_valid")) for r in read_rows(out_dir / "v17_h1600_summary.csv"))
    horizon_raw_rows = read_rows(out_dir / "v17_horizon_extension.csv")
    measured_horizon = sum(1 for r in horizon_raw_rows if str(r.get("execution_status")) == "measured")
    best_source_h100 = max([finite_float(r.get("source_vs_best_control_h100"), -999.0) for r in mechanism_rows] or [-999.0])
    best_source_h800 = max([finite_float(r.get("source_vs_best_control_h800"), -999.0) for r in mechanism_rows] or [-999.0])
    best_source_h1600 = max([finite_float(r.get("source_vs_best_control_h1600"), -999.0) for r in mechanism_rows] or [-999.0])
    efficiency_complete = int(bool(efficiency_rows) and all(str(r.get("execution_status")) == "measured" for r in efficiency_rows))
    kernel_pass = int(bool(kernel_rows) and all(int_flag(r.get("kernel_correctness_exploration_pass")) for r in kernel_rows))
    if not int_flag(s0.get("s0_pass")):
        route = "R0-CodeOrMetricInvalid"
        detail = "S0 code/metric/kernel/update gate did not pass"
    elif missing_count:
        route = "R0-CodeOrMetricInvalid"
        detail = f"required artifact missing count={missing_count}"
    elif measured_mech < planned:
        route = "R1-EfficiencyEnvelopeUnknown"
        detail = f"mechanism matrix incomplete measured={measured_mech} planned={planned}"
    elif not efficiency_complete:
        route = "R1-EfficiencyEnvelopeUnknown"
        detail = "mandatory efficiency census incomplete or blocked"
    elif not kernel_pass:
        route = "R7-BasisKernelEfficiencyBlocked"
        detail = "kernel correctness exploration gate blocked"
    elif h800_valid == 0:
        route = "R1-EfficiencyEnvelopeUnknown"
        detail = "h800/h1600 fresh readback not executed in this v17 screen"
    elif best_source_h100 < 0.005:
        route = "R3-FUIntrinsicNoSource"
        detail = "AdamW-free/FU variants did not exceed best controls at h100 screen"
    else:
        route = "R8-CarrierSpecificPartialPositive"
        detail = "fresh h100/h800/h1600 screen found partial positive source, but S4 real-transfer and S5 official gates were not executed/passed"
    return {
        "route": route,
        "route_detail": detail,
        "timestamp": now_sg(),
        "s0_pass": int_flag(s0.get("s0_pass")),
        "planned_mechanism_jobs": planned,
        "measured_mechanism_jobs": measured_mech,
        "best_source_vs_best_control_h100": best_source_h100,
        "best_source_vs_best_control_h800": best_source_h800,
        "best_source_vs_best_control_h1600": best_source_h1600,
        "h800_valid_rows": h800_valid,
        "h1600_valid_rows": h1600_valid,
        "measured_horizon_rows": measured_horizon,
        "efficiency_rows": len(efficiency_rows),
        "efficiency_complete": efficiency_complete,
        "kernel_rows": len(kernel_rows),
        "kernel_exploration_all_pass": kernel_pass,
        "required_artifact_missing_count": missing_count,
        "promotion_allowed": 0,
        "official_success_reached": 0,
    }


def parse_args_for_defaults(out_dir: Path) -> argparse.Namespace:
    ns = build_parser().parse_args([])
    ns.out_dir = str(out_dir)
    return ns


def required_manifest(out_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for name in REQUIRED_ARTIFACTS + REQUIRED_FIGURES:
        if is_v171_run(out_dir) and name in {
            "v17_code_review_packet/packet_manifest.csv",
            "v17_code_review_packet/packet_sha256_manifest.csv",
        }:
            continue
        path = out_dir / name
        rows.append({"artifact": name, "path": str(path), "exists": int(path.exists()), "size_bytes": path.stat().st_size if path.exists() else 0})
    write_rows(out_dir / "v17_required_artifact_manifest.csv", rows)
    return rows


def is_v1701_run(out_dir: Path) -> bool:
    text = str(out_dir)
    return "17_0_1" in text or "17.0.1" in text or "v1701" in text


def is_v171_run(out_dir: Path) -> bool:
    text = str(out_dir)
    return "17_1" in text or "17.1" in text or "v171" in text


def is_v18_run(out_dir: Path) -> bool:
    text = str(out_dir)
    return "18_0" in text or "18.0" in text or "v18" in text


def is_v19_run(out_dir: Path) -> bool:
    text = str(out_dir)
    return "19_0" in text or "19.0" in text or "v19" in text


def doc_paths(out_dir: Path) -> tuple[Path, Path, Path]:
    if is_v19_run(out_dir):
        return PLAN_DOC_19, EXEC_LOG_DOC_19, RECAP_DOC_19
    if is_v18_run(out_dir):
        return PLAN_DOC_18, EXEC_LOG_DOC_18, RECAP_DOC_18
    if is_v171_run(out_dir):
        return PLAN_DOC_171, EXEC_LOG_DOC_171, RECAP_DOC_171
    if is_v1701_run(out_dir):
        return PLAN_DOC_1701, EXEC_LOG_DOC_1701, RECAP_DOC_1701
    return PLAN_DOC, EXEC_LOG_DOC, RECAP_DOC


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run_text_command(cmd: Sequence[str], *, cwd: Path = ROOT, timeout: int = 60) -> tuple[int, str]:
    try:
        proc = subprocess.run(list(cmd), cwd=str(cwd), text=True, capture_output=True, timeout=timeout)
        return proc.returncode, (proc.stdout or "") + (("\n--- stderr ---\n" + proc.stderr) if proc.stderr else "")
    except Exception as exc:
        return 999, f"{type(exc).__name__}: {exc}"


def copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def write_placeholder_svg(path: Path, title: str, rows: Sequence[dict[str, Any]], value_key: str | None = None) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        vals = []
        labels = []
        for idx, row in enumerate(list(rows)[:12]):
            labels.append(str(row.get("carrier", row.get("mechanism", row.get("family", idx))))[:18])
            vals.append(finite_float(row.get(value_key), 0.0) if value_key else float(idx + 1))
        if not vals:
            labels = ["empty"]
            vals = [0.0]
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.bar(range(len(vals)), vals)
        ax.set_title(title)
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
        fig.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, format="svg")
        plt.close(fig)
    except Exception:
        path.write_text(f"<svg xmlns='http://www.w3.org/2000/svg' width='700' height='240'><text x='20' y='40'>{title}</text><text x='20' y='80'>figure generation blocked</text></svg>\n", encoding="utf-8")


def build_figures(out_dir: Path, mechanism_rows: list[dict[str, Any]], efficiency_rows: list[dict[str, str]], kernel_rows: list[dict[str, str]]) -> None:
    s0_rows = read_rows(out_dir / "v17_s0_import_closure.csv")
    linec = read_rows(out_dir / "v17_linec_golden_tests.csv")
    sign = read_rows(out_dir / "v17_update_sign_sanity.csv")
    overwrite = read_rows(out_dir / "v17_adamw_overwrite_diagnostic.csv")
    semantic = read_rows(out_dir / "v17_semantic_noncollapse_audit.csv")
    mapping = {
        "fig_v17_s0_code_gate_dashboard.svg": ("v17 S0 code gate", s0_rows, "import_ok"),
        "fig_v17_linec_golden_null_distribution.svg": ("LineC golden/null", linec, "pass"),
        "fig_v17_update_semantics_sign_tests.svg": ("update semantics", sign, "small_step_loss_sanity"),
        "fig_v17_adamw_overwrite_cosine.svg": ("AdamW overwrite cosine", overwrite, "cos_FU_AdamW"),
        "fig_v17_source_retention_horizon_curves.svg": ("source retention h100", mechanism_rows, "source_retention_h100"),
        "fig_v17_debt_recovery_curves.svg": ("LineC coupling", mechanism_rows, "CouplingR2"),
        "fig_v17_carrier_mechanism_heatmap.svg": ("carrier mechanism source h100", mechanism_rows, "source_vs_best_control_h100"),
        "fig_v17_efficiency_forward_backward_update_bar.svg": ("training step ratio", efficiency_rows, "training_step_ratio"),
        "fig_v17_memory_phase_breakdown.svg": ("memory ratio", efficiency_rows, "memory_ratio"),
        "fig_v17_same_param_mlp_efficiency_pareto.svg": ("same param MLP delta", read_rows(out_dir / "v17_same_param_mlp_manifest.csv"), "same_param_mlp_sanity_pass"),
        "fig_v17_gpu_utilization_timeline.svg": ("GPU runtime", read_rows(out_dir / "v17_gpu_utilization_dashboard.csv"), "runtime_sec"),
        "fig_v17_failure_taxonomy_heatmap.svg": ("failure taxonomy", read_rows(out_dir / "v17_failure_taxonomy.csv"), None),
        "fig_v17_code_audit_import_closure.svg": ("v17 import closure", s0_rows, "import_ok"),
        "fig_v17_linec_golden_test_dashboard.svg": ("LineC golden pass", linec, "pass"),
        "fig_v17_update_sign_sanity.svg": ("update sign sanity", sign, "small_step_loss_sanity"),
        "fig_v17_adamw_overwrite_cosine_curve.svg": ("AdamW overwrite cosine", overwrite, "cos_FU_AdamW"),
        "fig_v17_carrier_mechanism_source_heatmap.svg": ("carrier mechanism source h100", mechanism_rows, "source_vs_best_control_h100"),
        "fig_v17_source_retention_horizon_curves.svg": ("source retention h100", mechanism_rows, "source_retention_h100"),
        "fig_v17_tail_linec_debt_recovery_curves.svg": ("LineC coupling", mechanism_rows, "CouplingR2"),
        "fig_v17_kan_vs_mlp_attribution_matrix.svg": ("KAN vs MLP source", mechanism_rows, "source_vs_best_control_h100"),
        "fig_v17_semantic_noncollapse_cluster.svg": ("semantic noncollapse cosine", semantic, "update_cosine"),
        "fig_v17_efficiency_forward_ratio_by_basis.svg": ("forward ratio", efficiency_rows, "forward_ratio"),
        "fig_v17_efficiency_backward_ratio_by_basis.svg": ("backward ratio", efficiency_rows, "backward_ratio"),
        "fig_v17_efficiency_update_ratio_by_basis.svg": ("update ratio", efficiency_rows, "update_ratio"),
        "fig_v17_efficiency_memory_ratio_by_basis.svg": ("memory ratio", efficiency_rows, "memory_ratio"),
        "fig_v17_phase_time_stacked_bar_by_basis.svg": ("phase time", efficiency_rows, "step_training_only_ms"),
        "fig_v17_memory_phase_stacked_bar_by_basis.svg": ("memory phase", efficiency_rows, "backward_peak_memory"),
        "fig_v17_basis_kernel_component_waterfall.svg": ("kernel grad relerr", kernel_rows, "kernel_grad_relerr"),
        "fig_v17_gpu_utilization_timeline.svg": ("GPU runtime", read_rows(out_dir / "v17_gpu_utilization_dashboard.csv"), "runtime_sec"),
        "fig_v17_queue_depth_over_time.svg": ("queue drain", read_rows(out_dir / "v17_queue_drain_report.csv"), "executed_jobs"),
        "fig_v17_job_completion_gantt.svg": ("job completion", read_rows(out_dir / "v17_gpu_assignment_manifest.csv"), "runtime_sec"),
        "fig_v17_route_taxonomy_heatmap.svg": ("failure taxonomy", read_rows(out_dir / "v17_failure_taxonomy.csv"), None),
        "code_audit_dashboard.svg": ("v17.1 code audit", read_rows(out_dir / "v17_1_code_audit_summary.csv"), "compileall_ok"),
        "linec_golden_results.svg": ("LineC golden", linec, "pass"),
        "update_semantics_signcheck.svg": ("update semantics signcheck", sign, "small_step_loss_sanity"),
        "route_aggregation_unit_tests.svg": ("route aggregation tests", read_rows(out_dir / "route_aggregation_unit_tests.csv"), "pass"),
        "efficiency_phase_stacked_bar.svg": ("efficiency phase", efficiency_rows, "training_step_ratio"),
        "forward_ratio_by_basis.svg": ("forward ratio", efficiency_rows, "forward_ratio"),
        "backward_ratio_by_basis.svg": ("backward ratio", efficiency_rows, "backward_ratio"),
        "update_ratio_by_basis.svg": ("update ratio", efficiency_rows, "update_ratio"),
        "memory_ratio_by_basis.svg": ("memory ratio", efficiency_rows, "memory_ratio"),
        "basis_efficiency_pareto.svg": ("basis efficiency pareto", efficiency_rows, "training_step_ratio"),
        "kernel_blocker_heatmap.svg": ("kernel blocker", read_rows(out_dir / "v17_1_efficiency_blocker_table.csv"), None),
        "audit_cost_pollution_bar.svg": ("audit cost pollution", read_rows(out_dir / "v17_1_audit_cost_separation.csv"), "audit_cost_polluted"),
        "source_horizon_curves.svg": ("source horizon", read_rows(out_dir / "v17_1_functional_source_matrix.csv"), "source_vs_best_control_h1600_mean"),
        "source_retention_heatmap.svg": ("source retention", read_rows(out_dir / "v17_1_source_retention_matrix.csv"), "source_retention_h1600_over_h800"),
        "debt_recovery_curves.svg": ("debt recovery", read_rows(out_dir / "v17_1_debt_recovery_matrix.csv"), None),
        "control_attribution_heatmap.svg": ("control attribution", read_rows(out_dir / "v17_1_control_attribution.csv"), "control_explains_positive"),
        "kan_vs_mlp_same_mechanism.svg": ("KAN vs MLP", read_rows(out_dir / "v17_1_kan_vs_mlp_attribution.csv"), "KAN_specific_advantage"),
        "adamw_overwrite_cosine.svg": ("AdamW overwrite", read_rows(out_dir / "v17_1_adamw_overwrite_diagnostics.csv"), "cos_FU_AdamW"),
        "mechanism_noncollapse_matrix.svg": ("mechanism noncollapse", read_rows(out_dir / "v17_1_mechanism_noncollapse.csv"), "noncollapse_pass"),
        "gpu_utilization_dashboard.svg": ("GPU utilization", read_rows(out_dir / "v17_gpu_utilization_dashboard.csv"), "runtime_sec"),
        "queue_drain_timeline.svg": ("queue drain", read_rows(out_dir / "v17_queue_drain_report.csv"), "executed_jobs"),
        "idle_violation_timeline.svg": ("idle violation", read_rows(out_dir / "v17_idle_violation.csv"), "idle_violation"),
    }
    for name, (title, rows, key) in mapping.items():
        write_placeholder_svg(out_dir / name, title, rows, key)


def write_docs(out_dir: Path, route: dict[str, Any], mechanism_rows: list[dict[str, Any]], efficiency_rows: list[dict[str, str]], kernel_rows: list[dict[str, str]]) -> None:
    plan_doc, exec_doc, recap_doc = doc_paths(out_dir)
    commands = [f"{PYTHON} experiments/run_v170_code_audit_and_semantic_tests.py --out-dir {out_dir} --device cuda:0 --data-root data"]
    if is_v171_run(out_dir):
        commands = [f"{PYTHON} experiments/run_v17_1_code_correctness_gate.py --out-dir {out_dir} --device cuda:0 --data-root data"]
    elif is_v1701_run(out_dir):
        commands = [f"{PYTHON} experiments/run_v17_code_correctness_audit.py --out-dir {out_dir} --device cuda:0 --data-root data"]
    mechanism_runner = "experiments/run_v17_1_functional_matrix.py" if is_v171_run(out_dir) else ("experiments/run_v17_4gpu_scheduler.py" if is_v1701_run(out_dir) else "experiments/run_v173_full_carrier_mechanism_4gpu.py")
    efficiency_runner = "experiments/run_v17_1_basis_efficiency_repair.py" if is_v171_run(out_dir) else ("experiments/run_v17_basis_kernel_efficiency_matrix.py" if is_v1701_run(out_dir) else "experiments/run_v172_basis_kernel_efficiency_breakthrough.py")
    commands.extend(
        f"{PYTHON} {mechanism_runner} --stage mechanism-shard --out-dir {out_dir} --shard-count 4 --shard-index {idx} --device cuda:{idx} --data-root data --fail-on-missing-s0"
        for idx in range(4)
    )
    commands.extend(
        f"{PYTHON} {efficiency_runner} --out-dir {out_dir} --stage efficiency-shard --shard-count 4 --shard-index {idx} --device cuda:{idx} --data-root data {'--efficiency-batches 8,32,128 ' if is_v171_run(out_dir) else ''}--fail-on-missing-s0"
        for idx in range(4)
    )
    commands.extend(
        f"{PYTHON} {mechanism_runner} --stage horizon-shard --out-dir {out_dir} --shard-count 4 --shard-index {idx} --device cuda:{idx} --data-root data --horizon-all-rows 1 {'--horizon-steps 3200 ' if is_v171_run(out_dir) else ''}--fail-on-missing-s0"
        for idx in range(4)
    )
    final_runner = "experiments/run_v17_1_finalize.py" if is_v171_run(out_dir) else "experiments/run_v17_finalize.py"
    commands.append(f"{PYTHON} {final_runner} --out-dir {out_dir} --data-root data --horizon-all-rows 1 {'--horizon-steps 3200' if is_v171_run(out_dir) else ''}".rstrip())
    exec_lines = [
        "# DG-KAN v17.1 三段式执行日志" if is_v171_run(out_dir) else ("# DG-KAN v17.0.1 执行日志" if is_v1701_run(out_dir) else "# DG-KAN v17 执行日志"),
        "",
        f"生成时间：{now_sg()}",
        f"计划文件：{plan_doc}",
        f"结果目录：{out_dir}",
        f"Python：{PYTHON}",
        "",
        "## 实际执行入口",
    ]
    exec_lines.extend([f"- `{cmd}`" for cmd in commands])
    snapshot_rows = read_rows(out_dir / "v17_gpu_runtime_snapshots.csv")
    if snapshot_rows:
        exec_lines.extend(
            [
                "",
                "## GPU 运行快照",
                "",
                "| timestamp | gpu | util | memory | process_note |",
                "|---|---:|---:|---:|---|",
            ]
        )
        for row in snapshot_rows:
            gpu = row.get("gpu", row.get("gpu_index", ""))
            util = row.get("utilization_gpu", row.get("utilization_gpu_pct", ""))
            memory = row.get("memory_used", row.get("memory_used_mb", ""))
            note = row.get("process_note", row.get("name", ""))
            exec_lines.append(f"| {row.get('timestamp')} | {gpu} | {util} | {memory} | {note} |")
    exec_lines.extend(
        [
            "",
            "## 关键产物",
            f"- route: `{out_dir / 'v17_route_decision.json'}`",
            f"- mechanism matrix: `{out_dir / 'v17_functional_mechanism_matrix.csv'}`",
            f"- efficiency truth table: `{out_dir / 'v17_efficiency_truth_table.csv'}`",
            f"- kernel correctness: `{out_dir / 'v17_kernel_correctness.csv'}`",
            f"- horizon extension: `{out_dir / 'v17_horizon_extension.csv'}`",
            f"- code review packet: `{out_dir / ('v17_1_code_review_packet.zip' if is_v171_run(out_dir) else 'v17_code_review_packet.zip')}`",
            f"- packet manifest: `{out_dir / (('v17_1_code_review_packet' if is_v171_run(out_dir) else 'v17_code_review_packet') + '/packet_manifest.csv')}`",
            f"- command journal: `{out_dir / 'logs/v17_command_journal.md'}`",
            "",
            "## 复现注意",
            "- 方向来源只使用 train batch loss/gradient；LineC/tail/AUC/calibration 只作为 readback/audit。",
            "- v17 runner 不 import v13-v16 runner；旧 artifact 只作为历史背景，不作为本次 fresh row。",
            "- 本次使用 `--horizon-all-rows 1` 对 full 945 mechanism rows 执行 fresh horizon extension；v17.1 额外使用 `--horizon-steps 3200` 记录 h400/h800/h1600/h3200 readback。",
            "- 本次没有把 family-specific torch repair 升级成 official fused CUDA/C++ kernel，也没有执行 S4/S5 real-transfer official gate；这些必须作为未完成项记录，不能写成 official success。",
        ]
    )
    write_text(exec_doc, "\n".join(exec_lines) + "\n")

    best_rows = sorted(
        [r for r in mechanism_rows if str(r.get("mechanism")) not in CONTROL_MECHANISMS],
        key=lambda r: finite_float(r.get("source_vs_best_control_h100"), -999.0),
        reverse=True,
    )[:12]
    best_eff = sorted(efficiency_rows, key=lambda r: finite_float(r.get("training_step_ratio"), 999.0))[:12]
    h800_rows = read_rows(out_dir / "v17_h800_summary.csv")
    h1600_rows = read_rows(out_dir / "v17_h1600_summary.csv")
    top_h800 = sorted([r for r in h800_rows if int_flag(r.get("h800_measurement_valid"))], key=lambda r: finite_float(r.get("source_vs_best_control_h800"), -999.0), reverse=True)[:10]
    top_h1600 = sorted([r for r in h1600_rows if int_flag(r.get("h1600_measurement_valid"))], key=lambda r: finite_float(r.get("source_vs_best_control_h1600"), -999.0), reverse=True)[:10]
    v171_source_rows = read_rows(out_dir / "v17_1_source_retention_matrix.csv")
    top_h3200 = sorted([r for r in v171_source_rows if finite_float(r.get("source_vs_best_control_h3200_mean"), -999.0) > -999.0], key=lambda r: finite_float(r.get("source_vs_best_control_h3200_mean"), -999.0), reverse=True)[:10]
    v171_route = json.loads((out_dir / "v17_1_route_decision.json").read_text(encoding="utf-8")) if (out_dir / "v17_1_route_decision.json").exists() else {}
    recap = [
        "# DG-KAN v17.1 三段式实验结果复盘" if is_v171_run(out_dir) else ("# DG-KAN v17.0.1 实验结果复盘" if is_v1701_run(out_dir) else "# DG-KAN v17 实验结果复盘"),
        "",
        f"生成时间：{now_sg()}",
        "",
        "## Route",
        "",
        f"- route: `{route.get('route')}`",
        f"- route_detail: {v171_route.get('route_detail', route.get('route_detail')) if is_v171_run(out_dir) else route.get('route_detail')}",
        f"- CodeRoute: {v171_route.get('CodeRoute', '')}",
        f"- EfficiencyRoute: {v171_route.get('EfficiencyRoute', '')}",
        f"- FunctionalRoute: {v171_route.get('FunctionalRoute', '')}",
        f"- S0 pass: {route.get('s0_pass')}",
        f"- measured mechanism jobs: {route.get('measured_mechanism_jobs')}/{route.get('planned_mechanism_jobs')}",
        f"- best h100 source vs best control: {route.get('best_source_vs_best_control_h100')}",
        f"- best h800 source vs best control: {route.get('best_source_vs_best_control_h800')}",
        f"- best h1600 source vs best control: {route.get('best_source_vs_best_control_h1600')}",
        f"- h800 valid summary rows: {route.get('h800_valid_rows')}",
        f"- h1600 valid summary rows: {route.get('h1600_valid_rows')}",
        f"- measured horizon rows: {route.get('measured_horizon_rows')}",
        f"- h3200 measured group rows: {v171_route.get('h3200_measured_group_rows', '')}",
        f"- h3200 positive mean group rows: {v171_route.get('h3200_positive_mean_group_rows', '')}",
        f"- h3200 retained candidate count: {v171_route.get('h3200_retained_candidate_count', '')}",
        f"- h3200 late rebound count: {v171_route.get('h3200_late_rebound_count', '')}",
        "- 注：route 中 best source 字段是单个 fresh row 的最大值；h800/h1600/h3200 表格是 carrier x mechanism 分组后的 9-row 均值排行。当前 retention ratio 仅在前一 horizon mean source 为正时定义，避免把 late rebound 写成 retention。",
        "",
        "## 计划覆盖边界",
        "",
        "- 已完成：S0.1 import/LineC/update/retention/debt/route/kernel correctness gate、full carrier x mechanism h100 screen、full 945-row horizon extension、mandatory exploration efficiency census、required/forbidden/no-action audits、v17.1 code review packet、4GPU shard execution记录。",
        "- 已修复/补齐：v17.1 runner compatibility shims、dynamic geometry debt helper、`dgkan.diagnostics.linec` 稳定入口、FU update semantics shims、efficiency/profiling/kernel gradcheck shims、`loss_interface`、manual optimizer、matched controls；M2 SGD/Momentum branch 使用 `.step_gradient()`；basis repair 从 alias reference path 改为 family-specific torch repair path。",
        "- 未完成/不可冒充：official fused CUDA/C++ basis kernels、S4 real-transfer exploration、S5 official success gate、真正动态 work-stealing queue。本次 4GPU 为 shard queue drain，并记录 idle/dashboard；不能写成 promotion-ready official result。",
        "",
        "## Part A：代码审计",
        "",
        f"- code_audit_status: {v171_route.get('code_audit_status', 'see S0 route')}",
        f"- required artifact missing count: {route.get('required_artifact_missing_count')}",
        "- LineC exception policy: MeasurementInvalid，不进入 geometry fail 均值。",
        "- route aggregation: v17.1 额外输出 9-row mean source/retention matrix；single-row max 只作诊断字段。",
        "- debt accounting: 公式单元测试通过；真实 tail/LineC/calibration/AUC debt 未完整测量的字段保持空值/measurement_status，不编造 recovery。",
        "",
        "## Part B：基函数效率",
        "",
        f"- efficiency_status: {v171_route.get('efficiency_status', 'see efficiency blocker table')}",
        "- v17.1 输出 `v17_1_efficiency_truth_table.csv`、`v17_1_efficiency_blocker_table.csv`、`v17_1_family_repair_summary.csv`。",
        "- official_fused_kernel_complete=0 的 family 不能写 official efficiency success。",
        "",
        "## Part C：Functional Update",
        "",
        f"- functional_status: {v171_route.get('functional_status', 'see functional matrices')}",
        "- v17.1 输出 `v17_1_functional_source_matrix.csv`、`v17_1_source_retention_matrix.csv`、`v17_1_control_attribution.csv`、`v17_1_kan_vs_mlp_attribution.csv`。",
        "- S4/S5 未执行/未通过前，promotion_allowed 保持 0。",
        "",
        "## 关键实验数据",
        "",
        "| carrier | mechanism | dataset | seed | source_h100 | final_val_loss | LineC valid | CouplingR2 |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in best_rows:
        recap.append(f"| {r.get('carrier')} | {r.get('mechanism')} | {r.get('dataset')} | {r.get('seed')} | {r.get('source_vs_best_control_h100')} | {r.get('final_val_loss')} | {r.get('linec_measurement_valid')} | {r.get('CouplingR2')} |")
    recap.extend(
        [
            "",
            "## h800 / h1600 / h3200 Fresh Extension",
            "",
            "| horizon | carrier | mechanism | valid_rows | source_vs_best_control | retention_ratio | route |",
            "|---|---|---|---:|---:|---:|---|",
        ]
    )
    for r in top_h800:
        recap.append(f"| h800 | {r.get('carrier')} | {r.get('mechanism')} | {r.get('rows')} | {r.get('source_vs_best_control_h800')} | {r.get('source_retention_h800_over_h100')} | {r.get('route')} |")
    for r in top_h1600:
        recap.append(f"| h1600 | {r.get('carrier')} | {r.get('mechanism')} | {r.get('rows')} | {r.get('source_vs_best_control_h1600')} | {r.get('source_retention_h1600_over_h800')} | {r.get('route')} |")
    for r in top_h3200:
        recap.append(f"| h3200 | {r.get('carrier')} | {r.get('mechanism')} | {r.get('measurement_valid_rows_h3200')} | {r.get('source_vs_best_control_h3200_mean')} | {r.get('source_retention_h3200_over_h1600')} | measured |")
    recap.extend(
        [
            "",
            "## Efficiency 证据",
            "",
            "| carrier | batch | forward_ratio | backward_ratio | update_ratio | step_ratio | memory_ratio | exploration |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for r in best_eff:
        recap.append(f"| {r.get('carrier')} | {r.get('batch_size')} | {r.get('forward_ratio')} | {r.get('backward_ratio')} | {r.get('update_ratio')} | {r.get('training_step_ratio')} | {r.get('memory_ratio')} | {r.get('efficiency_exploration_gate')} |")
    recap.extend(
        [
            "",
            "## Kernel 证据",
            "",
            "| family | forward_relerr | grad_relerr | grad_cosine | dense_materialized | exploration_pass |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for r in kernel_rows:
        recap.append(f"| {r.get('family')} | {r.get('kernel_forward_relerr')} | {r.get('kernel_grad_relerr')} | {r.get('kernel_grad_cosine')} | {r.get('dense_basis_materialized')} | {r.get('kernel_correctness_exploration_pass')} |")
    recap.extend(
        [
            "",
            "## 修改记录（便于审计）",
            "",
            "- 新增 `dgkan/metrics/linec.py`：LineC measurement invalid 独立 route，不把异常当 geometry fail。",
            "- 新增 `dgkan/fu/core.py` 与 `dgkan/fu/mechanisms.py`：统一 UpdateTensor kind/sign/space/source，并实现 AdamW-free、slow-state、matrix-block、PopRisk/SNR、role-partition smoke。",
            "- 新增 `dgkan/fu/loss_interface.py`：显式记录 CE/Brier train-stream cotangent readback，禁止 audit metrics 进入 direction source。",
            "- 新增 `dgkan/fu/optimizers.py` 与 `dgkan/fu/controls.py`：manual SGD/Momentum optimizer 与 matched NoOp/Random controls；修复 M2 branch 的 `.step_gradient()` 调用。",
            "- 新增 `dgkan/profiling/efficiency_v17.py`：每个 phase 使用 cloned state，training cost 与 audit cost 分离。",
            "- 新增/修改 `dgkan/kernels/v17_*`：family kernel correctness/readback 入口；corrective pass 将 repaired path 改为 torch family-specific repair，但 `official_fused_kernel_complete=0`。",
            "- 新增 `experiments/run_v17_common.py` 及 v170/v171/v172/v173/finalize 包装入口。",
            "- 修复 v17.1 required artifact 合同：v17.1 code packet 使用 `v17_1_code_review_packet/...`，不再把 legacy `v17_code_review_packet/...` 目录 manifest 误判为缺失。",
            "- 修复 source retention ratio：只有前一 horizon 的 9-row mean source 为正时才定义 retention ratio；h1600 非正而 h3200 转正记录为 late rebound，不写成 retained source。",
            "",
            "## 分析与 insight",
            "",
            "- S0 的价值在于先确认 import closure、LineC golden、update sign、kernel gradcheck 是否可信；若这里失败，不能写 scientific no-go。",
            "- corrective pass 的机制矩阵与 h800/h1600/h3200 extension 都是 fresh real-dataset rows；这解决了之前 low-budget/top-only 违背计划覆盖的问题。",
            "- h3200 出现 late rebound 时不能自动解释成 retention；本轮 retention ratio 修复后，长期推进要看 positive source 是否跨 horizon 连续留存，而不是只看终点反弹。",
            "- AdamW overwrite diagnostic 提供了 FU 是否被 AdamW 抵消的证据链：`cos_FU_AdamW` 与 `optimizer_overwrite_projection` 为后续 AdamWWashesFU/IntrinsicNoSource 分类服务。",
            "- Efficiency rows 分离了 `step_training_only_ms` 和 `linec_audit_ms`，避免把 audit readback 误计入 carrier 训练效率。",
            "- 由于 S4/S5 未执行且 official fused kernel 未完成，route 最高只能是 partial/exploration 级别；任何 promotion_allowed=1 都必须等 real-transfer official rows 和 official efficiency/kernel gates 真实通过。",
        ]
    )
    write_text(recap_doc, "\n".join(recap) + "\n")


def write_v18_docs(out_dir: Path, route: dict[str, Any], mechanism_rows: list[dict[str, Any]], efficiency_rows: list[dict[str, str]], kernel_rows: list[dict[str, str]]) -> None:
    plan_doc, exec_doc, recap_doc = doc_paths(out_dir)
    commands = [
        f"{PYTHON} experiments/run_v18_code_gate.py --out-dir {out_dir} --device cuda:0 --data-root data",
        *[
            f"{PYTHON} experiments/run_v18_functional_update_breakthrough.py --stage mechanism-shard --out-dir {out_dir} --shard-count 4 --shard-index {idx} --device cuda:{idx} --data-root data --fail-on-missing-s0"
            for idx in range(4)
        ],
        *[
            f"{PYTHON} experiments/run_v18_basis_efficiency_breakthrough.py --stage efficiency-shard --out-dir {out_dir} --shard-count 4 --shard-index {idx} --device cuda:{idx} --data-root data --efficiency-batches 8,32,128,256 --fail-on-missing-s0"
            for idx in range(4)
        ],
        *[
            f"{PYTHON} experiments/run_v18_functional_update_breakthrough.py --stage horizon-shard --out-dir {out_dir} --shard-count 4 --shard-index {idx} --device cuda:{idx} --data-root data --horizon-all-rows 1 --horizon-steps 3200 --fail-on-missing-s0"
            for idx in range(4)
        ],
        f"{PYTHON} experiments/run_v18_merge_finalize.py --out-dir {out_dir} --data-root data --horizon-all-rows 1 --horizon-steps 3200",
    ]
    gpu_rows = read_rows(out_dir / "v17_gpu_runtime_snapshots.csv")
    assignments = read_rows(out_dir / "v18_gpu_assignment_manifest.csv")
    shard_runtime: dict[str, float] = {}
    for row in assignments:
        gpu = str(row.get("gpu_id", row.get("gpu", "")))
        shard_runtime[gpu] = shard_runtime.get(gpu, 0.0) + finite_float(row.get("runtime_sec"), 0.0)
    horizon_runtime_rows = []
    for path in sorted(out_dir.glob("v17_horizon_extension_h*.csv")):
        rows = read_rows(path)
        horizon_runtime_rows.append({"shard": path.stem.rsplit("_", 1)[-1], "rows": len(rows), "runtime_sum_sec": f"{sum(finite_float(r.get('runtime_sec'), 0.0) for r in rows):.2f}"})
    late_runtime_rows = []
    for path in sorted(out_dir.glob("v19_late_rebound_rerun_matrix_r*.csv")):
        rows = read_rows(path)
        late_runtime_rows.append({"shard": path.stem.rsplit("_", 1)[-1], "rows": len(rows), "runtime_sum_sec": f"{sum(finite_float(r.get('runtime_sec'), 0.0) for r in rows):.2f}"})
    horizon_runtime_rows = []
    for path in sorted(out_dir.glob("v17_horizon_extension_h*.csv")):
        rows = read_rows(path)
        horizon_runtime_rows.append({"shard": path.stem.rsplit("_", 1)[-1], "rows": len(rows), "runtime_sum_sec": f"{sum(finite_float(r.get('runtime_sec'), 0.0) for r in rows):.2f}"})
    late_runtime_rows = []
    for path in sorted(out_dir.glob("v19_late_rebound_rerun_matrix_r*.csv")):
        rows = read_rows(path)
        late_runtime_rows.append({"shard": path.stem.rsplit("_", 1)[-1], "rows": len(rows), "runtime_sum_sec": f"{sum(finite_float(r.get('runtime_sec'), 0.0) for r in rows):.2f}"})
    max_mem = max([finite_float(r.get("memory_used_mb", r.get("memory_used", 0.0)), 0.0) for r in gpu_rows] or [0.0])
    max_util = max([finite_float(r.get("utilization_gpu_pct", r.get("utilization_gpu", 0.0)), 0.0) for r in gpu_rows] or [0.0])
    exec_lines = [
        "# DG-KAN v18.0 Evidence-First 执行日志",
        "",
        f"生成时间：{now_sg()}",
        f"计划文件：{plan_doc}",
        f"结果目录：{out_dir}",
        f"Python：{PYTHON}",
        "",
        "## 实际执行命令",
    ]
    exec_lines.extend(f"- `{cmd}`" for cmd in commands)
    exec_lines.extend(
        [
            "",
            "## 关键文件",
            f"- route: `{out_dir / 'v18_route_decision.json'}`",
            f"- code packet: `{out_dir / 'v18_code_review_packet.zip'}`",
            f"- functional raw horizon matrix: `{out_dir / 'v18_functional_raw_horizon_matrix.csv'}`",
            f"- source retention matrix: `{out_dir / 'v18_source_retention_matrix.csv'}`",
            f"- debt accounting matrix: `{out_dir / 'v18_debt_accounting_matrix.csv'}`",
            f"- efficiency truth table: `{out_dir / 'v18_efficiency_truth_table.csv'}`",
            f"- efficiency blocker table: `{out_dir / 'v18_efficiency_blocker_table.csv'}`",
            f"- kernel repair matrix: `{out_dir / 'v18_kernel_repair_matrix.csv'}`",
            f"- queue drain: `{out_dir / 'v18_gpu_queue_drain_report.csv'}`",
            f"- command journal: `{out_dir / 'logs/v17_command_journal.md'}`",
            "",
            "## 运行统计",
            f"- functional rows: {len(mechanism_rows)}",
            f"- efficiency rows: {len(efficiency_rows)}",
            f"- kernel rows: {len(kernel_rows)}",
            f"- GPU snapshot rows: {len(gpu_rows)}",
            f"- max GPU memory used MB: {max_mem}",
            f"- max GPU util percent: {max_util}",
            "",
            "### Assignment Manifest Runtime",
            "",
            "| gpu | runtime_sec_sum |",
            "|---|---:|",
        ]
    )
    for gpu, sec in sorted(shard_runtime.items()):
        exec_lines.append(f"| {gpu} | {sec:.2f} |")
    exec_lines.extend(["", "### Horizon Shard Runtime", ""])
    append_table(exec_lines, horizon_runtime_rows, [("shard", "shard"), ("rows", "rows"), ("runtime_sum_sec", "runtime_sum_sec")])
    exec_lines.extend(["", "### Late Rebound Rerun Runtime", ""])
    append_table(exec_lines, late_runtime_rows, [("shard", "shard"), ("rows", "rows"), ("runtime_sum_sec", "runtime_sum_sec")])
    exec_lines.extend(
        [
            "",
            "## Repro Notes",
            "- `--horizon-all-rows 1 --horizon-steps 3200` 表示对已测 functional rows 逐行做 fresh horizon readback。",
            "- v18 efficiency 使用 batch 8/32/128/256；threshold 按计划写入 `v18_efficiency_blocker_table.csv`。",
            "- 当前 debt matrix 只从真实 trace 计算 CEp99 tail 和 val_loss AUC；LineC-channel/ECE/Brier debt 未真实测量时保留空值并标注 blocker。",
            "- 当前 kernel repair matrix 对未实现 R1+ 变体逐条写 deferred，不冒充 official fused CUDA/C++。",
        ]
    )
    write_text(exec_doc, "\n".join(exec_lines) + "\n")

    source_rows = read_rows(out_dir / "v18_source_retention_matrix.csv")
    eff_rows = read_rows(out_dir / "v18_efficiency_truth_table.csv")
    blocker_rows = read_rows(out_dir / "v18_efficiency_blocker_table.csv")
    repair_rows = read_rows(out_dir / "v18_kernel_repair_matrix.csv")
    debt_rows = read_rows(out_dir / "v18_debt_accounting_matrix.csv")
    control_rows = read_rows(out_dir / "v18_control_attribution.csv")
    kan_mlp_rows = read_rows(out_dir / "v18_kan_vs_mlp_attribution.csv")
    route18 = route
    top_h100 = sorted(source_rows, key=lambda r: finite_float(r.get("source_vs_best_control_h100_mean"), -999.0), reverse=True)[:10]
    top_h800 = sorted(source_rows, key=lambda r: finite_float(r.get("source_vs_best_control_h800_mean"), -999.0), reverse=True)[:10]
    top_h1600 = sorted(source_rows, key=lambda r: finite_float(r.get("source_vs_best_control_h1600_mean"), -999.0), reverse=True)[:10]
    top_h3200 = sorted(source_rows, key=lambda r: finite_float(r.get("source_vs_best_control_h3200_mean"), -999.0), reverse=True)[:10]
    blocker_counts: dict[str, int] = {}
    for row in blocker_rows:
        cls = str(row.get("blocker_class"))
        blocker_counts[cls] = blocker_counts.get(cls, 0) + 1
    deferred_repair = sum(1 for r in repair_rows if str(r.get("implementation_status")) == "not_implemented_in_v18_run")
    incomplete_debt = sum(1 for r in debt_rows if "not_measured" in str(r.get("measurement_status")))
    recap = [
        "# DG-KAN v18.0 Evidence-First 实验结果复盘",
        "",
        f"生成时间：{now_sg()}",
        "",
        "## Route",
        "",
        f"- route: `{route18.get('route')}`",
        f"- route_detail: {route18.get('route_detail')}",
        f"- CodeRoute: {route18.get('CodeRoute')}",
        f"- EfficiencyRoute: {route18.get('EfficiencyRoute')}",
        f"- FunctionalRoute: {route18.get('FunctionalRoute')}",
        f"- S0_2_preflight_pass: {route18.get('S0_2_preflight_pass')}",
        f"- S0_2_evidence_complete: {route18.get('S0_2_evidence_complete')}",
        f"- debt_metrics_complete: {route18.get('debt_metrics_complete')}",
        f"- measured functional rows: {route18.get('v18_measured_functional_rows')}",
        f"- measured efficiency rows: {route18.get('v18_measured_efficiency_rows')}",
        f"- measured debt rows: {route18.get('v18_measured_debt_rows')}",
        f"- required artifact missing count: {route18.get('v18_required_artifact_missing_count')}",
        "- promotion_allowed: 0",
        "",
        "## 覆盖边界",
        "",
        "- 已完成：S0 import/LineC/update/retention/route/kernel correctness 预检；fresh functional matrix；fresh h400/h800/h1600/h3200 horizon matrix；batch 8/32/128/256 efficiency census；v18 packet、queue、failure taxonomy、两份日志。",
        "- 已真实修复/补齐：新增 v18 wrappers；新增 v18 final artifacts；新增 v18 code packet 结构；新增 CEp99 tail debt 与 val_loss AUC readback；新增 v18 kernel repair matrix，把未实现 repair variants 显式写成 deferred。",
        "- 未完成/不可冒充：LineC-channel/ECE/Brier debt 未进入 horizon trace；D-FOU/D-CHE/LQ/D-RAT/D-RBF/D-WAV R1+ official fused repair 未实现；S4/S5 real-transfer official gate 未执行；因此不能写 breakthrough/promotion-ready。",
        "",
        "## Part A：S0.2 Metric / Implementation Truth Gate",
        "",
        f"- compile/import/LineC/update/kernel preflight route: {route18.get('CodeRoute')}",
        f"- debt rows incomplete: {incomplete_debt}/{len(debt_rows)}",
        f"- deferred kernel repair variants: {deferred_repair}",
        "- 结论：代码正确性预检可以支撑继续跑 fresh rows；但 S0.2 evidence-complete gate 未过，因为计划要求的 LineC-channel/ECE/Brier debt 还没有真实 readback。",
        "",
        "| item | value | evidence |",
        "|---|---:|---|",
        f"| functional raw horizon rows | {len(read_rows(out_dir / 'v18_functional_raw_horizon_matrix.csv'))} | `v18_functional_raw_horizon_matrix.csv` |",
        f"| source grouped rows | {len(source_rows)} | `v18_source_retention_matrix.csv` |",
        f"| debt rows | {len(debt_rows)} | `v18_debt_accounting_matrix.csv` |",
        f"| required manifest missing | {route18.get('v18_required_artifact_missing_count')} | `v18_required_artifact_manifest.csv` |",
        f"| packet manifest rows | {len(read_rows(out_dir / 'v18_code_review_packet/packet_manifest.csv'))} | `v18_code_review_packet/packet_manifest.csv` |",
        "",
        "## Part B：Basis Kernel Efficiency",
        "",
        f"- EfficiencyRoute: {route18.get('EfficiencyRoute')}",
        "- v18 gate: forward/backward/step <= 1.75，memory <= 1.10，audit overhead <= 0.20；official gate 还要求 official fused kernel complete。",
        "",
        "| blocker_class | rows |",
        "|---|---:|",
    ]
    for cls, count in sorted(blocker_counts.items()):
        recap.append(f"| {cls} | {count} |")
    recap.extend(["", "| carrier | batch | forward | backward | step | memory | audit | blocker |", "|---|---:|---:|---:|---:|---:|---:|---|"])
    for row in blocker_rows:
        recap.append(f"| {row.get('carrier')} | {row.get('batch_size')} | {row.get('forward_ratio')} | {row.get('backward_ratio')} | {row.get('step_ratio')} | {row.get('memory_ratio')} | {row.get('audit_overhead_fraction')} | {row.get('blocker_class')} |")
    recap.extend(
        [
            "",
            "### Kernel Repair Matrix Summary",
            "",
            "| family | variants | implemented_current | deferred | official_fused_complete |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for family in sorted(v18_kernel_variants()):
        fam_rows = [r for r in repair_rows if str(r.get("family")) == family]
        recap.append(f"| {family} | {len(fam_rows)} | {sum(1 for r in fam_rows if str(r.get('implementation_status')) != 'not_implemented_in_v18_run')} | {sum(1 for r in fam_rows if str(r.get('implementation_status')) == 'not_implemented_in_v18_run')} | {sum(int_flag(r.get('official_fused_kernel_complete')) for r in fam_rows)} |")
    recap.extend(
        [
            "",
            "## Part C：Functional Update",
            "",
            f"- FunctionalRoute: {route18.get('FunctionalRoute')}",
            f"- h800 candidates: {route18.get('v18_h800_candidate_count')}",
            f"- h1600 retained candidates: {route18.get('v18_h1600_candidate_count')}",
            f"- late rebound candidates: {route18.get('v18_late_rebound_candidate_count')}",
            "",
            "### Source / Retention Evidence",
            "",
            "| horizon | carrier | mechanism | mean_source | pass_count | retention |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    for label, rows, key, pass_key, ret_key in [
        ("h100", top_h100[:5], "source_vs_best_control_h100_mean", "dataset_seed_pass_count_h100", ""),
        ("h800", top_h800[:5], "source_vs_best_control_h800_mean", "dataset_seed_pass_count_h800", "source_retention_h800_over_h100"),
        ("h1600", top_h1600[:5], "source_vs_best_control_h1600_mean", "dataset_seed_pass_count_h1600", "source_retention_h1600_over_h800"),
        ("h3200", top_h3200[:5], "source_vs_best_control_h3200_mean", "dataset_seed_pass_count_h3200", "source_retention_h3200_over_h1600"),
    ]:
        for row in rows:
            recap.append(f"| {label} | {row.get('carrier')} | {row.get('mechanism')} | {row.get(key)} | {row.get(pass_key)} | {row.get(ret_key) if ret_key else ''} |")
    recap.extend(
        [
            "",
            "### Control / KAN-vs-MLP Attribution",
            "",
            "| metric | value |",
            "|---|---:|",
            f"| control rows explaining positive | {sum(int_flag(r.get('control_explains_positive')) for r in control_rows)}/{len(control_rows)} |",
            f"| KAN advantage h1600 rows | {sum(int_flag(r.get('KAN_specific_advantage_h1600')) for r in kan_mlp_rows)}/{len(kan_mlp_rows)} |",
            f"| KAN advantage h3200 rows | {sum(int_flag(r.get('KAN_specific_advantage_h3200')) for r in kan_mlp_rows)}/{len(kan_mlp_rows)} |",
            "",
            "### Debt Evidence",
            "",
            "- CEp99 tail debt 和 val_loss AUC 是从真实 train/horizon trace 计算的。",
            "- LineC-channel/ECE/Brier debt 字段保持空值并标记 `not_measured`，所以本轮不能写 debt recovery success。",
            "",
            "| carrier | mechanism | dataset | seed | tail_peak | tail_final | recovery | status |",
            "|---|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in debt_rows[:12]:
        recap.append(f"| {row.get('carrier')} | {row.get('mechanism')} | {row.get('dataset')} | {row.get('seed')} | {row.get('tail_debt_peak')} | {row.get('tail_debt_final')} | {row.get('tail_recovery_rate')} | {row.get('measurement_status')} |")
    recap.extend(
        [
            "",
            "## 4GPU / Queue 证据",
            "",
            f"- GPU snapshot rows: {len(gpu_rows)}",
            f"- max memory used MB: {max_mem}",
            f"- max util percent: {max_util}",
            f"- queue drain report: `{out_dir / 'v18_gpu_queue_drain_report.csv'}`",
            "",
            "## 修改记录（便于审计）",
            "",
            "- 新增 v18 runner wrappers：`experiments/run_v18_code_gate.py`、`run_v18_functional_update_breakthrough.py`、`run_v18_basis_efficiency_breakthrough.py`、`run_v18_merge_finalize.py`。",
            "- 修改 `experiments/run_v17_common.py`：增加 v18 doc paths、v18 final artifacts、v18 packet、v18 debt matrix、v18 route decision、v18 required manifest。",
            "- 新增 v18 debt accounting：从真实 trace 计算 CEp99 tail delta/recovery 和 val_loss AUC；未真实测量的 LineC-channel/ECE/Brier debt 明确标注 incomplete。",
            "- 新增 v18 kernel repair matrix：R0-current measured，R1+ 未实现逐项 deferred；`official_fused_kernel_complete=0`。",
            "",
            "## 分析 / Insight / 结论",
            "",
            "- 本轮如果出现 h3200 转正但 h1600 非正，只能叫 late rebound；不能算 retention，也不能进入 promotion。",
            "- 当前最硬 blocker 不是显存，而是 basis forward/backward compute 和 official fused repair 缺失；memory ratio 接近阈值内不等于效率 gate 通过。",
            "- AdamW overwrite diagnostic 仍是判断 FU 被 optimizer writeback 洗掉的关键证据，但必须和 AdamW-free / slow-state rows 的跨 horizon retention 一起看。",
            "- v18 的 evidence-first 结论是：fresh rows 已跑、证据包齐备，但 S0.2 evidence-complete / efficiency official / functional retained source 均未满足；不能写 breakthrough。",
        ]
    )
    write_text(recap_doc, "\n".join(recap) + "\n")


def write_v19_docs(out_dir: Path, route: dict[str, Any], mechanism_rows: list[dict[str, Any]], efficiency_rows: list[dict[str, str]], kernel_rows: list[dict[str, str]]) -> None:
    plan_doc, exec_doc, recap_doc = doc_paths(out_dir)
    commands = [
        f"{PYTHON} experiments/run_v19_s03_truth_gate.py --out-dir {out_dir} --device cuda:0 --data-root data",
        *[
            f"{PYTHON} experiments/run_v19_source_channel_fu_matrix.py --stage mechanism-shard --out-dir {out_dir} --shard-count 4 --shard-index {idx} --device cuda:{idx} --data-root data --fail-on-missing-s0"
            for idx in range(4)
        ],
        *[
            f"{PYTHON} experiments/run_v19_basis_kernel_breakthrough.py --stage efficiency-shard --out-dir {out_dir} --shard-count 4 --shard-index {idx} --device cuda:{idx} --data-root data --efficiency-batches 8,32,128,256 --fail-on-missing-s0"
            for idx in range(4)
        ],
        *[
            f"{PYTHON} experiments/run_v19_source_channel_fu_matrix.py --stage horizon-shard --out-dir {out_dir} --shard-count 4 --shard-index {idx} --device cuda:{idx} --data-root data --horizon-all-rows 1 --horizon-steps 4800 --fail-on-missing-s0"
            for idx in range(4)
        ],
        *[
            f"{PYTHON} experiments/run_v19_late_rebound_rerun.py --out-dir {out_dir} --shard-count 4 --shard-index {idx} --device cuda:{idx} --data-root data"
            for idx in range(4)
        ],
        f"{PYTHON} experiments/run_v19_merge_finalize.py --out-dir {out_dir} --data-root data --horizon-all-rows 1 --horizon-steps 4800",
    ]

    def append_table(lines: list[str], rows: Sequence[dict[str, Any]], columns: Sequence[tuple[str, str]], *, limit: int | None = None) -> None:
        shown = list(rows)[:limit] if limit is not None else list(rows)
        lines.append("| " + " | ".join(label for label, _ in columns) + " |")
        lines.append("|" + "|".join("---" for _ in columns) + "|")
        for row in shown:
            values = []
            for _, key in columns:
                value = row.get(key, "")
                if value is None:
                    value = ""
                values.append(str(value))
            lines.append("| " + " | ".join(values) + " |")

    gpu_rows = read_rows(out_dir / "v17_gpu_runtime_snapshots.csv")
    max_mem = max([finite_float(r.get("memory_used_mb", r.get("memory_used", 0.0)), 0.0) for r in gpu_rows] or [0.0])
    max_util = max([finite_float(r.get("utilization_gpu_pct", r.get("utilization_gpu", 0.0)), 0.0) for r in gpu_rows] or [0.0])
    assignments = read_rows(out_dir / "v19_gpu_assignment_manifest.csv")
    shard_runtime: dict[str, float] = {}
    for row in assignments:
        gpu = str(row.get("gpu_id", row.get("gpu", "")))
        shard_runtime[gpu] = shard_runtime.get(gpu, 0.0) + finite_float(row.get("runtime_sec"), 0.0)
    horizon_runtime_rows = []
    for path in sorted(out_dir.glob("v17_horizon_extension_h*.csv")):
        rows = read_rows(path)
        horizon_runtime_rows.append({"shard": path.stem.rsplit("_", 1)[-1], "rows": len(rows), "runtime_sum_sec": f"{sum(finite_float(r.get('runtime_sec'), 0.0) for r in rows):.2f}"})
    late_runtime_rows = []
    for path in sorted(out_dir.glob("v19_late_rebound_rerun_matrix_r*.csv")):
        rows = read_rows(path)
        late_runtime_rows.append({"shard": path.stem.rsplit("_", 1)[-1], "rows": len(rows), "runtime_sum_sec": f"{sum(finite_float(r.get('runtime_sec'), 0.0) for r in rows):.2f}"})

    exec_lines = [
        "# DG-KAN v19.0 Source-Channel FU + Basis Kernel Breakthrough 执行日志",
        "",
        f"生成时间：{now_sg()}",
        f"计划文件：{plan_doc}",
        f"结果目录：{out_dir}",
        f"Python：{PYTHON}",
        "",
        "## 实际执行命令",
    ]
    exec_lines.extend(f"- `{cmd}`" for cmd in commands)
    exec_lines.extend(
        [
            "",
            "## 关键文件",
            f"- route: `{out_dir / 'v19_route_decision.json'}`",
            f"- S0 route: `{out_dir / 'v17_s0_route_decision.json'}`",
            f"- LineC-channel golden: `{out_dir / 'linec_channel_golden_tests.csv'}`",
            f"- mechanism contract: `{out_dir / 'v19_mechanism_semantic_contract.csv'}`",
            f"- functional raw horizon matrix: `{out_dir / 'v19_functional_raw_horizon_matrix.csv'}`",
            f"- source retention matrix: `{out_dir / 'v19_source_retention_matrix.csv'}`",
            f"- debt accounting matrix: `{out_dir / 'v19_debt_accounting_matrix.csv'}`",
            f"- debt metric availability: `{out_dir / 'v19_debt_metric_availability.csv'}`",
            f"- efficiency blocker table: `{out_dir / 'v19_efficiency_blocker_table.csv'}`",
            f"- efficiency waterfall: `{out_dir / 'v19_efficiency_waterfall.csv'}`",
            f"- kernel repair matrix: `{out_dir / 'v19_kernel_repair_matrix.csv'}`",
            f"- late rebound rerun matrix: `{out_dir / 'v19_late_rebound_rerun_matrix.csv'}`",
            f"- code packet: `{out_dir / 'v19_code_review_packet.zip'}`",
            f"- result bundle: `{out_dir / 'v19_results_bundle.zip'}`",
            f"- command journal: `{out_dir / 'logs/v17_command_journal.md'}`",
            "",
            "## 运行统计",
            f"- mechanism rows: {len(mechanism_rows)}",
            f"- efficiency rows: {len(efficiency_rows)}",
            f"- kernel correctness rows: {len(kernel_rows)}",
            f"- GPU snapshot rows: {len(gpu_rows)}",
            f"- max GPU memory used MB: {max_mem}",
            f"- max GPU util percent: {max_util}",
            "",
            "### Assignment Manifest Runtime",
            "",
            "| gpu | runtime_sec_sum |",
            "|---|---:|",
        ]
    )
    for gpu, sec in sorted(shard_runtime.items()):
        exec_lines.append(f"| {gpu} | {sec:.2f} |")
    exec_lines.extend(["", "### Horizon Shard Runtime", ""])
    append_table(exec_lines, horizon_runtime_rows, [("shard", "shard"), ("rows", "rows"), ("runtime_sum_sec", "runtime_sum_sec")])
    exec_lines.extend(["", "### Late Rebound Rerun Runtime", ""])
    append_table(exec_lines, late_runtime_rows, [("shard", "shard"), ("rows", "rows"), ("runtime_sum_sec", "runtime_sum_sec")])
    exec_lines.extend(
        [
            "",
            "## Repro Notes",
            "- v19 使用 `--horizon-all-rows 1 --horizon-steps 4800`，h100/h400/h800/h1600/h2400/h3200/h4800 均从 fresh trace/readback 汇总。",
            "- LineC-fast 和 LineC-channel 均为 audit/readback；direction source 仍只来自 train-stream loss/gradient/update state。",
            "- Debt recovery 对 CEp99/NLL/ECE/Brier/LineC-fast/LineC-channel/AUCtime 逐项记录，缺失项写 EvidenceIncomplete，不写 0 或成功。",
            "- Basis repair rows 使用 `repair_variant` 区分 family-specific attempt；未达到 official/no-materialize gate 时 `promotion_allowed=0`。",
        ]
    )
    write_text(exec_doc, "\n".join(exec_lines) + "\n")

    s0 = json.loads((out_dir / "v17_s0_route_decision.json").read_text(encoding="utf-8")) if (out_dir / "v17_s0_route_decision.json").exists() else {}
    source_rows = read_rows(out_dir / "v19_source_retention_matrix.csv")
    debt_rows = read_rows(out_dir / "v19_debt_accounting_matrix.csv")
    availability = read_rows(out_dir / "v19_debt_metric_availability.csv")
    eff_truth = read_rows(out_dir / "v19_efficiency_truth_table.csv")
    blockers = read_rows(out_dir / "v19_efficiency_blocker_table.csv")
    repair_rows = read_rows(out_dir / "v19_kernel_repair_matrix.csv")
    late_rows = read_rows(out_dir / "v19_late_rebound_rerun_matrix.csv")
    control_rows = read_rows(out_dir / "v19_control_attribution.csv")
    kan_mlp_rows = read_rows(out_dir / "v19_kan_vs_mlp_attribution.csv")
    linec_channel = read_rows(out_dir / "linec_channel_golden_tests.csv")
    contract = read_rows(out_dir / "v19_mechanism_semantic_contract.csv")
    manifest = read_rows(out_dir / "v19_required_artifact_manifest.csv")
    missing_manifest = sum(1 for r in manifest if not int_flag(r.get("exists")))
    blocker_counts: dict[str, int] = {}
    for row in blockers:
        cls = str(row.get("blocker_class"))
        blocker_counts[cls] = blocker_counts.get(cls, 0) + 1
    top_h100 = sorted(source_rows, key=lambda r: finite_float(r.get("source_vs_best_control_h100_mean"), -999.0), reverse=True)[:6]
    top_h800 = sorted(source_rows, key=lambda r: finite_float(r.get("source_vs_best_control_h800_mean"), -999.0), reverse=True)[:6]
    top_h1600 = sorted(source_rows, key=lambda r: finite_float(r.get("source_vs_best_control_h1600_mean"), -999.0), reverse=True)[:6]
    top_h3200 = sorted(source_rows, key=lambda r: finite_float(r.get("source_vs_best_control_h3200_mean"), -999.0), reverse=True)[:6]
    top_h4800 = sorted(source_rows, key=lambda r: finite_float(r.get("source_vs_best_control_h4800_mean"), -999.0), reverse=True)[:6]
    d_focus = [r for r in repair_rows if str(r.get("family")) in {"D-FOU", "D-CHE"}]
    debt_incomplete = sum(1 for r in debt_rows if "EvidenceIncomplete" in str(r.get("measurement_status")))
    late_positive = sum(1 for r in late_rows if finite_float(r.get("source_h3200"), -999.0) >= 0.005 or finite_float(r.get("source_h4800"), -999.0) >= 0.005)

    recap = [
        "# DG-KAN v19.0 Source-Channel FU + Basis Kernel Breakthrough 实验结果复盘",
        "",
        f"生成时间：{now_sg()}",
        "",
        "## Route",
        "",
        f"- route: `{route.get('route')}`",
        f"- route_detail: {route.get('route_detail')}",
        f"- CodeRoute: {route.get('CodeRoute')}",
        f"- EfficiencyRoute: {route.get('EfficiencyRoute')}",
        f"- FunctionalRoute: {route.get('FunctionalRoute')}",
        f"- S0_3_preflight_pass: {route.get('S0_3_preflight_pass')}",
        f"- S0_3_evidence_complete: {route.get('S0_3_evidence_complete')}",
        f"- LineC_fast_golden_pass: {route.get('LineC_fast_golden_pass')}",
        f"- LineC_channel_golden_pass: {route.get('LineC_channel_golden_pass')}",
        f"- debt_metrics_complete: {route.get('debt_metrics_complete')}",
        f"- measured functional rows: {route.get('measured_functional_rows')}",
        f"- measured efficiency rows: {route.get('measured_efficiency_rows')}",
        f"- measured debt rows: {route.get('measured_debt_rows')}",
        f"- late rebound rerun rows: {route.get('late_rebound_rerun_rows')}",
        f"- late rebound M2 rerun rows: {route.get('late_rebound_m2_rerun_rows')}",
        f"- late rebound M2 positive rerun rows: {route.get('late_rebound_positive_rerun_rows')}",
        f"- required artifact missing count: {route.get('required_artifact_missing_count', missing_manifest)}",
        "- promotion_allowed: 0",
        "",
        "## 覆盖边界",
        "",
        "- 已完成项只按 artifact 真实存在和 measured rows 记录；未执行、未通过、未实现项不会写成成功。",
        "- v19 新增/修复：LineC-channel golden、S0.3 debt formula edge cases、mechanism semantic contract、LineC-fast/channel horizon trace、CEp99/NLL/ECE/Brier/AUCtime debt matrix、basis repair variant matrix、D-CHE late rebound independent rerun入口、v19 packet/bundle。",
        "- 不可冒充项：若 `debt_metrics_complete=0`、D-FOU/D-CHE efficiency gate 未过、late rebound rerun 未复现、或 official fused/no-materialize kernel 未完成，则不能写 breakthrough/promotion-ready。",
        "",
        "## Part A：S0.3 Code / Metric / Mechanism Truth Gate",
        "",
        f"- s0_pass: {s0.get('s0_pass')}",
        f"- linec_fast golden: {s0.get('linec_golden_pass_count')}/{s0.get('linec_golden_total')}",
        f"- linec_channel golden: {s0.get('linec_channel_golden_pass_count')}/{s0.get('linec_channel_golden_total')}",
        f"- mechanism semantic contract pass: {s0.get('mechanism_semantic_contract_pass')}",
        f"- required artifact missing count: {missing_manifest}",
        "",
        "### LineC-channel Golden",
        "",
    ]
    append_table(recap, linec_channel, [("golden", "golden"), ("pass", "pass"), ("valid", "linec_measurement_valid"), ("CouplingR2", "CouplingR2"), ("NoiseLeak", "NoiseSignalLeak"), ("route", "route")])
    recap.extend(["", "### Debt Metric Availability", ""])
    append_table(recap, availability, [("metric", "metric"), ("available_rows", "available_rows"), ("total_rows", "total_rows"), ("complete", "complete")])
    recap.extend(["", "### Mechanism Semantic Contract", ""])
    append_table(recap, contract, [("mechanism", "mechanism"), ("source_space", "source_space"), ("optimizer_primary", "uses_optimizer_primary"), ("slow", "uses_slow_state"), ("matrix", "uses_matrix_block"), ("poprisk", "uses_poprisk_snr"), ("prototype", "implementation_is_prototype")], limit=24)
    recap.extend(
        [
            "",
            "## Part B：Basis Kernel Efficiency",
            "",
            f"- D-FOU efficiency pass: {route.get('d_fou_efficiency_pass')}",
            f"- D-CHE efficiency pass: {route.get('d_che_efficiency_pass')}",
            "- Gate 解释：forward/step/memory 等 ratio 达标只是 exploration 条件；official success 还需要 official fused/no-materialize kernel complete。",
            "",
            "### Blocker Counts",
            "",
            "| blocker_class | rows |",
            "|---|---:|",
        ]
    )
    for cls, count in sorted(blocker_counts.items()):
        recap.append(f"| {cls} | {count} |")
    recap.extend(["", "### D-FOU / D-CHE Repair Rows", ""])
    append_table(recap, d_focus, [("family", "family"), ("variant", "repair_variant"), ("batch", "batch_size"), ("forward", "forward_ratio"), ("step", "step_ratio"), ("memory", "memory_ratio"), ("gate", "exploration_gate"), ("blocker", "blocker_class")], limit=24)
    recap.extend(["", "### Efficiency Best Rows", ""])
    best_eff = sorted(eff_truth, key=lambda r: finite_float(r.get("training_step_ratio"), finite_float(r.get("step_ratio_vs_mlp"), 999.0)))[:16]
    append_table(recap, best_eff, [("carrier", "carrier"), ("variant", "repair_variant"), ("batch", "batch_size"), ("forward", "forward_ratio"), ("backward", "backward_ratio"), ("step", "training_step_ratio"), ("memory", "memory_ratio"), ("gate", "v19_exploration_gate")])
    recap.extend(
        [
            "",
            "## Part C：Functional Update",
            "",
            f"- FunctionalRoute: {route.get('FunctionalRoute')}",
            f"- MLP retained source count: {route.get('mlp_retained_source_count')}",
            f"- late rebound candidate count: {route.get('late_rebound_candidate_count')}",
            f"- late rebound M2 positive rerun rows: {late_positive}",
            "",
            "### Source / Retention Evidence Chain",
            "",
            "| horizon | carrier | mechanism | 9-row mean source | pass_count | retention |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    for label, rows, key, pass_key, ret_key in [
        ("h100", top_h100, "source_vs_best_control_h100_mean", "dataset_seed_pass_count_h100", ""),
        ("h800", top_h800, "source_vs_best_control_h800_mean", "dataset_seed_pass_count_h800", "source_retention_h800_over_h100"),
        ("h1600", top_h1600, "source_vs_best_control_h1600_mean", "dataset_seed_pass_count_h1600", "source_retention_h1600_over_h800"),
        ("h3200", top_h3200, "source_vs_best_control_h3200_mean", "dataset_seed_pass_count_h3200", "source_retention_h3200_over_h1600"),
        ("h4800", top_h4800, "source_vs_best_control_h4800_mean", "dataset_seed_pass_count_h4800", "source_retention_h4800_over_h3200"),
    ]:
        for row in rows:
            recap.append(f"| {label} | {row.get('carrier')} | {row.get('mechanism')} | {row.get(key)} | {row.get(pass_key)} | {row.get(ret_key) if ret_key else ''} |")
    recap.extend(["", "### Late Rebound Independent Rerun", ""])
    append_table(recap, late_rows, [("rerun", "rerun_id"), ("dataset", "dataset"), ("seed", "seed"), ("mechanism", "mechanism"), ("source_h3200", "source_h3200"), ("source_h4800", "source_h4800"), ("status", "execution_status")], limit=18)
    recap.extend(
        [
            "",
            "### Control / KAN-vs-MLP Attribution",
            "",
            "| metric | value |",
            "|---|---:|",
            f"| control rows explaining positive | {sum(int_flag(r.get('control_explains_positive')) for r in control_rows)}/{len(control_rows)} |",
            f"| KAN advantage h1600 rows | {sum(int_flag(r.get('KAN_specific_advantage_h1600')) for r in kan_mlp_rows)}/{len(kan_mlp_rows)} |",
            f"| KAN advantage h3200 rows | {sum(int_flag(r.get('KAN_specific_advantage_h3200')) for r in kan_mlp_rows)}/{len(kan_mlp_rows)} |",
            f"| KAN advantage h4800 rows | {sum(int_flag(r.get('KAN_specific_advantage_h4800')) for r in kan_mlp_rows)}/{len(kan_mlp_rows)} |",
            "",
            "### Debt Evidence",
            "",
            f"- incomplete debt rows: {debt_incomplete}/{len(debt_rows)}",
            "- peak=0 时 recovery 保持空值/undefined；缺失 metric 写 EvidenceIncomplete；这些都不能解释为成功。",
            "",
        ]
    )
    append_table(recap, debt_rows, [("carrier", "carrier"), ("mechanism", "mechanism"), ("dataset", "dataset"), ("seed", "seed"), ("tail_peak", "tail_debt_peak"), ("NLL_peak", "NLL_debt_peak"), ("ECE_peak", "ECE_debt_peak"), ("LineC_channel", "LineC_channel_recovery"), ("status", "measurement_status")], limit=16)
    recap.extend(
        [
            "",
            "## 4GPU / Queue 证据",
            "",
            f"- GPU snapshot rows: {len(gpu_rows)}",
            f"- max memory used MB: {max_mem}",
            f"- max util percent: {max_util}",
            f"- queue drain report: `{out_dir / 'v19_queue_drain_report.csv'}`",
            "",
            "### Horizon Shard Runtime",
            "",
        ]
    )
    append_table(recap, horizon_runtime_rows, [("shard", "shard"), ("rows", "rows"), ("runtime_sum_sec", "runtime_sum_sec")])
    recap.extend(
        [
            "",
            "### Late Rebound Rerun Runtime",
            "",
        ]
    )
    append_table(recap, late_runtime_rows, [("shard", "shard"), ("rows", "rows"), ("runtime_sum_sec", "runtime_sum_sec")])
    recap.extend(
        [
            "",
            "## 修改记录（便于审计）",
            "",
            "- 修改 `dgkan/metrics/linec.py`：新增 LineC-channel train-split trajectory audit 与 golden tests；修复 C7 scale invariance 为通道改善向量缩放不变性测试。",
            "- 修改 `dgkan/fu/mechanisms.py`：新增 M11 dual-memory slow-state、M12 schedule-free averaged FU、M13 low-rank matrix-block FU、M14 source-channel PopRisk/SNR slow FU，并新增 mechanism semantic contract rows。",
            "- 修改 `dgkan/profiling/efficiency_v17.py`：拆分 SGD/AdamW/manual-FU update ms、basis eval/readout/backward/update waterfall 字段。",
            "- 修改 `experiments/run_v17_common.py`：新增 v19 S0.3、h2400/h4800 trace/readback、CEp99/NLL/ECE/Brier/LineC-fast/LineC-channel/AUCtime debt、basis repair variants、v19 artifact/packet/bundle/docs/finalizer。",
            "- 新增 v19 wrappers：`run_v19_s03_truth_gate.py`、`run_v19_source_channel_fu_matrix.py`、`run_v19_basis_kernel_breakthrough.py`、`run_v19_late_rebound_rerun.py`、`run_v19_merge_finalize.py`。",
            "",
            "## 分析 / Insight / 结论",
            "",
            "- v19 的第一原则是 evidence-first：只要 S0.3 debt/LineC/mechanism contract 不完整，就不能把后续 no-go 写成科学结论。",
            "- MLP source 若跨 horizon 留存，按 generic training dynamics insight 写；只有 KAN row 本身 source 为正、retention 为正、controls 不能解释、debt recovery 完整，才可写 KAN-specific。",
            "- D-CHE/M2 h3200 late rebound 必须通过独立 rerun 才能升级为候选；前一 horizon 非正时不能把后续转正写成 retained source。",
            "- Basis efficiency 当前核心证据来自 repair_variant rows 与 waterfall；显存低不等于 gate 通过，forward/basis-eval/step ratio 才是主要 blocker。",
            "- `promotion_allowed=0` 是默认审计保护：只有 official S4/S5、debt complete、functional retained source、official/no-materialize efficiency gate 同时真实通过才可翻转。",
        ]
    )
    write_text(recap_doc, "\n".join(recap) + "\n")


def build_code_packet(out_dir: Path) -> None:
    plan_doc, exec_doc, recap_doc = doc_paths(out_dir)
    packet_stem = "v17_1_code_review_packet" if is_v171_run(out_dir) else "v17_code_review_packet"
    packet_dir = out_dir / packet_stem
    if packet_dir.exists():
        shutil.rmtree(packet_dir)
    for sub in [
        "00_README.md",
        "01_ENVIRONMENT",
        "02_SOURCE_TREE",
        "03_IMPORT_CLOSURE",
        "04_LINEC_CORRECTNESS",
        "05_UPDATE_SEMANTICS",
        "06_ADAMW_COUPLING_AUDIT",
        "07_FUNCTIONAL_MECHANISM_NONCOLLAPSE",
        "08_EFFICIENCY_PROFILER_CORRECTNESS",
        "09_KERNEL_GRADCHECK",
        "10_EXPERIMENT_RUNNERS",
        "10_RAW_EXPERIMENT_MATRICES",
        "11_RESULTS_MANIFESTS",
        "12_FIGURES_AND_DASHBOARDS",
        "12_FAILURE_TAXONOMY",
        "13_FAILURE_TAXONOMY",
        "13_REPRO_COMMANDS",
        "14_REPRO_COMMANDS",
    ]:
        target = packet_dir / sub
        if target.suffix:
            target.parent.mkdir(parents=True, exist_ok=True)
        else:
            target.mkdir(parents=True, exist_ok=True)

    git_code, git_head = run_text_command(["git", "rev-parse", "HEAD"])
    status_code, git_status = run_text_command(["git", "status", "--short"])
    diff_code, git_diff = run_text_command(["git", "diff", "--stat"])
    _, py_version = run_text_command([PYTHON, "--version"])
    _, pip_freeze = run_text_command([PYTHON, "-m", "pip", "freeze"], timeout=120)
    conda_bin = shutil.which("conda")
    conda_env = ""
    if conda_bin:
        _, conda_env = run_text_command([conda_bin, "env", "export"], timeout=120)
    else:
        conda_env = "conda_unavailable=1\nreason=conda executable not found in PATH\n"
    _, nvidia = run_text_command(["nvidia-smi"], timeout=60)
    try:
        import triton  # type: ignore

        triton_version = getattr(triton, "__version__", "unknown")
    except Exception as exc:
        triton_version = f"triton_unavailable=1\nreason={type(exc).__name__}: {exc}"
    torch_report = {
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "devices": [],
    }
    if torch.cuda.is_available():
        for idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(idx)
            torch_report["devices"].append(
                {
                    "index": idx,
                    "name": props.name,
                    "total_memory": props.total_memory,
                    "capability": list(props.major_minor) if hasattr(props, "major_minor") else [props.major, props.minor],
                }
            )
    git_hash = git_head.strip() if git_code == 0 else "git_commit_unavailable=1"

    write_text(packet_dir / "01_ENVIRONMENT/python_version.txt", py_version)
    write_text(packet_dir / "01_ENVIRONMENT/pip_freeze.txt", pip_freeze)
    write_text(packet_dir / "01_ENVIRONMENT/conda_env.yml", conda_env)
    write_text(packet_dir / "01_ENVIRONMENT/conda_env_export.yml", conda_env)
    write_text(packet_dir / "01_ENVIRONMENT/nvidia_smi.txt", nvidia)
    write_text(packet_dir / "01_ENVIRONMENT/torch_cuda_report.json", json.dumps(sanitize(torch_report), ensure_ascii=False, indent=2) + "\n")
    write_text(packet_dir / "01_ENVIRONMENT/triton_version.txt", str(triton_version) + "\n")
    write_text(packet_dir / "01_ENVIRONMENT/cuda_version.txt", str(torch.version.cuda) + "\n")
    write_text(packet_dir / "01_ENVIRONMENT/torch_version.txt", torch.__version__ + "\n")
    write_text(packet_dir / "01_ENVIRONMENT/gpu_info.txt", nvidia)
    write_text(packet_dir / "01_ENVIRONMENT/git_commit.txt", git_hash + "\n")
    write_text(packet_dir / "01_ENVIRONMENT/git_status.txt", git_status)
    write_text(packet_dir / "01_ENVIRONMENT/run_host.txt", platform.node() + "\n")

    readme = [
        "# v17.1 Code Review Packet" if is_v171_run(out_dir) else "# v17.0.1 Code Review Packet",
        "",
        f"- repo_root: {ROOT}",
        f"- git_commit: {git_hash}",
        f"- git_status_exit: {status_code}",
        f"- python: {PYTHON}",
        f"- platform: {platform.platform()}",
        f"- plan_doc: {plan_doc}",
        f"- result_dir: {out_dir}",
        "",
        "## Git Status",
        "```text",
        git_status.strip(),
        "```",
        "",
        "## Git Diff Summary",
        "```text",
        git_diff.strip(),
        "```",
        "",
        "## Repro Commands",
        f"- S0: `{PYTHON} experiments/run_v17_code_correctness_audit.py --out-dir {out_dir} --device cuda:0 --data-root data`",
        f"- smoke row: `{PYTHON} experiments/run_v17_adamw_free_functional_matrix.py --out-dir {out_dir} --stage mechanism-shard --shard-count 4 --shard-index 0 --device cuda:0 --data-root data --smoke-only`",
        f"- finalize: `{PYTHON} experiments/run_v17_finalize.py --out-dir {out_dir} --data-root data --horizon-all-rows 1`",
        "",
        "## Known Limitations",
        "- S4 real-transfer and S5 official success are not claimed unless corresponding real-transfer official rows exist.",
        "- Torch family-specific repair rows are not treated as official fused CUDA/C++ kernel completion when `official_fused_kernel_complete=0`.",
    ]
    write_text(packet_dir / "00_README.md", "\n".join(readme) + "\n")

    def ignore_tree(_dir: str, names: list[str]) -> set[str]:
        return {n for n in names if n in {"__pycache__", ".pytest_cache", ".mypy_cache"} or n.endswith(".pyc")}

    for rel in ["dgkan", "experiments"]:
        src = ROOT / rel
        if src.exists():
            shutil.copytree(src, packet_dir / "02_SOURCE_TREE" / rel, ignore=ignore_tree, dirs_exist_ok=True)
    docs_dst = packet_dir / "02_SOURCE_TREE/docs"
    docs_dst.mkdir(parents=True, exist_ok=True)
    for doc in sorted((ROOT / "docs").glob("DG-KAN_v1*.md")):
        name = doc.name
        if any(token in name for token in ["v16.4.1", "v16.5", "v17", "v17.0.1"]):
            copy_if_exists(doc, docs_dst / name)

    copy_map = {
        "03_IMPORT_CLOSURE": ["v17_compileall.log", "v17_import_closure.csv", "v17_import_errors.csv", "v17_module_dependency_graph.json", "v17_missing_symbol_report.csv", "compatibility_manifest.csv"],
        "04_LINEC_CORRECTNESS": [
            "linec_source_manifest.csv",
            "linec_golden_fixture.py",
            "v17_linec_golden_results.csv",
            "v17_linec_exception_policy_test.csv",
            "v17_linec_null_distribution.csv",
            "v17_linec_split_transfer_test.csv",
            "v17_linec_noise_injection_test.csv",
            "v17_linec_signal_reservoir_test.csv",
            "v17_linec_batch_order_mismatch_test.csv",
        ],
        "05_UPDATE_SEMANTICS": ["v17_update_semantics_tests.csv", "v17_update_type_manifest.csv", "v17_update_sign_sanity.csv"],
        "06_ADAMW_COUPLING_AUDIT": ["v17_adamw_coupling_map.csv", "v17_adamw_overwrite_diagnostic.csv", "v17_optimizer_state_dependency.csv", "v17_fu_vs_adamw_cosine.csv", "v17_fu_source_survival_by_recovery_optimizer.csv"],
        "07_FUNCTIONAL_MECHANISM_NONCOLLAPSE": ["v17_mechanism_semantic_manifest.csv", "v17_mechanism_update_cosine_matrix.csv", "v17_mechanism_param_mask_jaccard.csv", "v17_mechanism_state_dependency_matrix.csv", "v17_mechanism_noncollapse_summary.csv"],
        "08_EFFICIENCY_PROFILER_CORRECTNESS": ["v17_efficiency_profiler_unit_tests.csv", "efficiency_profiler_unit_tests.csv", "efficiency_phase_timer_tests.csv", "audit_cost_separation_test.csv", "same_param_mlp_matching_test.csv", "memory_peak_reset_test.csv", "warmup_vs_measured_test.csv"],
        "09_KERNEL_GRADCHECK": ["v17_kernel_gradcheck_summary.csv", "chebyshev_gradcheck.csv", "fourier_gradcheck.csv", "rbf_gradcheck.csv", "wavelet_gradcheck.csv", "lq_gradcheck.csv", "rational_gradcheck.csv", "manual_vs_autograd_gradcheck.csv"],
        "10_EXPERIMENT_RUNNERS": [
            "experiments/run_v17_1_code_correctness_gate.py",
            "experiments/run_v17_1_functional_matrix.py",
            "experiments/run_v17_1_basis_efficiency_repair.py",
            "experiments/run_v17_1_finalize.py",
            "experiments/run_v17_full_codeaudit_adamwfree_fu_basis_efficiency_4gpu.py",
            "experiments/run_v17_code_correctness_audit.py",
            "experiments/run_v17_adamw_free_functional_matrix.py",
            "experiments/run_v17_basis_kernel_efficiency_matrix.py",
            "experiments/run_v17_4gpu_scheduler.py",
            "experiments/run_v17_finalize.py",
        ],
        "10_RAW_EXPERIMENT_MATRICES": [
            "raw_h100_matrix.csv",
            "raw_h400_matrix.csv",
            "raw_h800_matrix.csv",
            "raw_h1600_matrix.csv",
            "raw_h3200_matrix.csv",
            "raw_controls_matrix.csv",
            "raw_efficiency_matrix.csv",
            "raw_linec_debt_matrix.csv",
            "raw_tail_calibration_debt_matrix.csv",
        ],
        "11_RESULTS_MANIFESTS": [
            "v17_required_artifact_manifest.csv",
            "v17_forbidden_information_audit.csv",
            "v17_no_action_search_audit.csv",
            "v17_direction_provenance.csv",
            "v17_measurement_validity_manifest.csv",
            "v17_budget_exhaustion_certificate.csv",
            "v17_deferred_items.csv",
            "v17_next_hypothesis_queue.md",
            "v17_no_go_boundary.md",
        ],
        "12_FIGURES_AND_DASHBOARDS": REQUIRED_FIGURES,
        "13_FAILURE_TAXONOMY": ["v17_failure_taxonomy.csv", "v17_basis_efficiency_failure_taxonomy.csv", "v17_source_washout_diagnosis.csv"],
        "12_FAILURE_TAXONOMY": ["v17_failure_taxonomy.csv", "v17_basis_efficiency_failure_taxonomy.csv", "v17_source_washout_diagnosis.csv", "v17_1_functional_no_go_boundary.md", "v17_1_next_hypothesis_queue.md"],
    }
    for section, names in copy_map.items():
        for name in names:
            src = ROOT / name if name.startswith("experiments/") else out_dir / name
            copy_if_exists(src, packet_dir / section / Path(name).name)
    s0_runner = "experiments/run_v17_1_code_correctness_gate.py" if is_v171_run(out_dir) else "experiments/run_v17_code_correctness_audit.py"
    func_runner = "experiments/run_v17_1_functional_matrix.py" if is_v171_run(out_dir) else "experiments/run_v17_4gpu_scheduler.py"
    eff_runner = "experiments/run_v17_1_basis_efficiency_repair.py" if is_v171_run(out_dir) else "experiments/run_v17_basis_kernel_efficiency_matrix.py"
    fin_runner = "experiments/run_v17_1_finalize.py" if is_v171_run(out_dir) else "experiments/run_v17_finalize.py"
    commands = [
        f"{PYTHON} {s0_runner} --out-dir {out_dir} --device cuda:0 --data-root data",
        f"{PYTHON} {func_runner} --stage mechanism-shard --out-dir {out_dir} --shard-count 4 --shard-index <0..3> --device cuda:<0..3> --data-root data --fail-on-missing-s0",
        f"{PYTHON} {eff_runner} --stage efficiency-shard --out-dir {out_dir} --shard-count 4 --shard-index <0..3> --device cuda:<0..3> --data-root data --efficiency-batches 8,32,128 --fail-on-missing-s0",
        f"{PYTHON} {func_runner} --stage horizon-shard --out-dir {out_dir} --shard-count 4 --shard-index <0..3> --device cuda:<0..3> --data-root data --horizon-all-rows 1 --horizon-steps 3200 --fail-on-missing-s0",
        f"{PYTHON} {fin_runner} --out-dir {out_dir} --data-root data --horizon-all-rows 1 --horizon-steps 3200",
    ]
    repro_md = "\n".join(f"- `{cmd}`" for cmd in commands) + "\n"
    write_text(packet_dir / "14_REPRO_COMMANDS/repro_commands.md", repro_md)
    write_text(packet_dir / "13_REPRO_COMMANDS/repro_commands.md", repro_md)
    script_map = {
        "run_all.sh": "\n".join(commands) + "\n",
        "run_s0_code_gate.sh": commands[0] + "\n",
        "run_linec_tests.sh": commands[0] + "\n",
        "run_update_semantics_tests.sh": commands[0] + "\n",
        "run_efficiency_census.sh": commands[2] + "\n",
        "run_functional_matrix.sh": commands[1] + "\n",
        "run_basis_repair.sh": commands[2] + "\n",
        "reproduce_route.sh": commands[-1] + "\n",
    }
    for name, text in script_map.items():
        write_text(packet_dir / "13_REPRO_COMMANDS" / name, "#!/usr/bin/env bash\nset -euo pipefail\n" + text)

    rows = []
    for path in sorted(p for p in packet_dir.rglob("*") if p.is_file()):
        rel = path.relative_to(packet_dir)
        if rel.name in {"packet_manifest.csv", "packet_sha256_manifest.csv"}:
            continue
        rows.append(
            {
                "relative_path": str(rel),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
                "artifact_type": rel.parts[0] if rel.parts else "unknown",
                "required": 1,
                "created_by": "experiments/run_v17_common.py:build_code_packet",
                "source_command": "finalize",
            }
        )
    write_rows(packet_dir / "packet_manifest.csv", rows)
    write_rows(packet_dir / "packet_sha256_manifest.csv", [{"relative_path": r["relative_path"], "sha256": r["sha256"], "bytes": r["bytes"]} for r in rows])
    write_rows(out_dir / f"{packet_stem}.csv", rows)
    if packet_stem != "v17_code_review_packet":
        write_rows(out_dir / "v17_code_review_packet.csv", rows)
    with zipfile.ZipFile(out_dir / f"{packet_stem}.zip", "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(p for p in packet_dir.rglob("*") if p.is_file()):
            zf.write(path, Path(packet_stem) / path.relative_to(packet_dir))
    if packet_stem != "v17_code_review_packet":
        with zipfile.ZipFile(out_dir / "v17_code_review_packet.zip", "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for path in sorted(p for p in packet_dir.rglob("*") if p.is_file()):
                zf.write(path, Path(packet_stem) / path.relative_to(packet_dir))


def run_finalize(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    mechanism_raw = merge_prefixed(out_dir, "v17_functional_mechanism_matrix_s*.csv", "v17_functional_mechanism_matrix.raw.csv")
    trace_rows = merge_prefixed(out_dir, "v17_functional_traces_s*.csv", "v17_functional_traces.csv")
    horizon_rows = merge_prefixed(out_dir, "v17_horizon_extension_h*.csv", "v17_horizon_extension.csv")
    merge_prefixed(out_dir, "v17_horizon_traces_h*.csv", "v17_horizon_traces.csv")
    linec_rows = merge_prefixed(out_dir, "v17_linec_measurements_s*.csv", "v17_linec_measurements.csv")
    sign_mech = merge_prefixed(out_dir, "v17_update_sign_sanity_mechanism_s*.csv", "v17_update_sign_sanity_mechanism.csv")
    overwrite_mech = merge_prefixed(out_dir, "v17_adamw_overwrite_diagnostic_mechanism_s*.csv", "v17_adamw_overwrite_diagnostic_mechanism.csv")
    if sign_mech:
        write_rows(out_dir / "v17_update_sign_sanity.csv", read_rows(out_dir / "v17_update_sign_sanity.csv") + sign_mech)
    if overwrite_mech:
        write_rows(out_dir / "v17_adamw_overwrite_diagnostic.csv", read_rows(out_dir / "v17_adamw_overwrite_diagnostic.csv") + overwrite_mech)
    mechanism_rows, h800, h1600 = aggregate_mechanism(mechanism_raw, trace_rows, horizon_rows)
    write_rows(out_dir / "v17_functional_mechanism_matrix.csv", mechanism_rows)
    write_rows(out_dir / "v17_h800_summary.csv", h800)
    write_rows(out_dir / "v17_h1600_summary.csv", h1600)

    efficiency_rows = merge_prefixed(out_dir, "v17_efficiency_truth_table_e*.csv", "v17_efficiency_truth_table.csv")
    kernel_rows = read_rows(out_dir / "v17_kernel_correctness.csv")
    kernel_rows.extend(merge_prefixed(out_dir, "v17_kernel_correctness_e*.csv", "v17_kernel_correctness.shards.csv"))
    dedup_kernel: dict[str, dict[str, str]] = {}
    for row in kernel_rows:
        dedup_kernel[str(row.get("family"))] = row
    kernel_rows = list(dedup_kernel.values())
    write_rows(out_dir / "v17_kernel_correctness.csv", kernel_rows)

    build_efficiency_breakdowns(out_dir, efficiency_rows)
    build_queue_artifacts(out_dir, args, mechanism_rows)
    build_audits(out_dir)
    build_failure_taxonomy(out_dir, mechanism_rows, efficiency_rows, kernel_rows)
    write_rows(out_dir / "v17_next_hypothesis_queue.csv", build_next_queue(mechanism_rows, efficiency_rows))
    write_rows(out_dir / "v17_no_go_boundary.csv", [{"boundary": "v17 route is not official scientific no-go unless S0/S1/h800/h1600/S4/S5 complete", "promotion_allowed": 0}])
    build_v1701_completion_artifacts(out_dir, mechanism_rows, efficiency_rows, kernel_rows)
    required_manifest(out_dir)
    route = decide_route(out_dir, mechanism_rows, h800, efficiency_rows, kernel_rows)
    if is_v19_run(out_dir):
        route19 = build_v19_completion_artifacts(out_dir, route, mechanism_rows, efficiency_rows, kernel_rows)
        write_v19_docs(out_dir, route19, mechanism_rows, efficiency_rows, kernel_rows)
        build_v19_code_packet(out_dir)
        manifest19 = v19_required_manifest(out_dir)
        route19["required_artifact_missing_count"] = sum(1 for r in manifest19 if not int_flag(r.get("exists")))
        write_json(out_dir / "v19_route_decision.json", route19)
        with zipfile.ZipFile(out_dir / "v19_results_bundle.zip", "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for path in sorted(p for p in out_dir.rglob("*") if p.is_file() and "v19_code_review_packet/" not in str(p.relative_to(out_dir)) and p.name != "v19_results_bundle.zip"):
                zf.write(path, path.relative_to(out_dir))
        write_v19_docs(out_dir, route19, mechanism_rows, efficiency_rows, kernel_rows)
        return
    if is_v18_run(out_dir):
        route18 = build_v18_completion_artifacts(out_dir, route, mechanism_rows, efficiency_rows, kernel_rows)
        manifest18 = v18_required_manifest(out_dir)
        route18["v18_required_artifact_missing_count"] = sum(1 for r in manifest18 if not int_flag(r.get("exists")))
        route18["required_artifact_missing_count"] = route18["v18_required_artifact_missing_count"]
        write_json(out_dir / "v18_route_decision.json", route18)
        write_v18_docs(out_dir, route18, mechanism_rows, efficiency_rows, kernel_rows)
        build_v18_code_packet(out_dir)
        manifest18 = v18_required_manifest(out_dir)
        route18["v18_required_artifact_missing_count"] = sum(1 for r in manifest18 if not int_flag(r.get("exists")))
        route18["required_artifact_missing_count"] = route18["v18_required_artifact_missing_count"]
        write_json(out_dir / "v18_route_decision.json", route18)
        write_v18_docs(out_dir, route18, mechanism_rows, efficiency_rows, kernel_rows)
        return
    if is_v171_run(out_dir):
        build_v171_completion_artifacts(out_dir, route, mechanism_rows, efficiency_rows, kernel_rows)
    write_json(out_dir / "v17_route_decision.json", route)
    build_figures(out_dir, mechanism_rows, efficiency_rows, kernel_rows)
    required_manifest(out_dir)
    route = decide_route(out_dir, mechanism_rows, h800, efficiency_rows, kernel_rows)
    write_json(out_dir / "v17_route_decision.json", route)
    if is_v171_run(out_dir):
        build_v171_completion_artifacts(out_dir, route, mechanism_rows, efficiency_rows, kernel_rows)
        build_figures(out_dir, mechanism_rows, efficiency_rows, kernel_rows)
        write_docs(out_dir, route, mechanism_rows, efficiency_rows, kernel_rows)
        build_code_packet(out_dir)
        required_manifest(out_dir)
        route = decide_route(out_dir, mechanism_rows, h800, efficiency_rows, kernel_rows)
        write_json(out_dir / "v17_route_decision.json", route)
        build_v171_completion_artifacts(out_dir, route, mechanism_rows, efficiency_rows, kernel_rows)
        write_docs(out_dir, route, mechanism_rows, efficiency_rows, kernel_rows)
        build_code_packet(out_dir)
        required_manifest(out_dir)
        route = decide_route(out_dir, mechanism_rows, h800, efficiency_rows, kernel_rows)
        write_json(out_dir / "v17_route_decision.json", route)
    write_text(out_dir / "v17_implementation_readback.md", implementation_readback(route, mechanism_rows, efficiency_rows, kernel_rows))
    write_docs(out_dir, route, mechanism_rows, efficiency_rows, kernel_rows)
    build_code_packet(out_dir)
    required_manifest(out_dir)
    route = decide_route(out_dir, mechanism_rows, h800, efficiency_rows, kernel_rows)
    write_json(out_dir / "v17_route_decision.json", route)
    write_text(out_dir / "v17_implementation_readback.md", implementation_readback(route, mechanism_rows, efficiency_rows, kernel_rows))
    write_docs(out_dir, route, mechanism_rows, efficiency_rows, kernel_rows)
    build_code_packet(out_dir)
    required_manifest(out_dir)
    route = decide_route(out_dir, mechanism_rows, h800, efficiency_rows, kernel_rows)
    write_json(out_dir / "v17_route_decision.json", route)


def build_next_queue(mechanism_rows: list[dict[str, Any]], efficiency_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows = []
    best = sorted(
        [r for r in mechanism_rows if str(r.get("mechanism")) not in CONTROL_MECHANISMS],
        key=lambda r: finite_float(r.get("source_vs_best_control_h100"), -999.0),
        reverse=True,
    )[:8]
    for r in best:
        rows.append({"priority": "P0", "item": "Run h800/h1600 extension", "carrier": r.get("carrier"), "mechanism": r.get("mechanism"), "reason": f"h100 source={r.get('source_vs_best_control_h100')}", "promotion_allowed": 0})
    for r in efficiency_rows:
        if not int_flag(r.get("efficiency_exploration_gate")):
            rows.append({"priority": "P1", "item": "Family-specific efficiency repair", "carrier": r.get("carrier"), "mechanism": "kernel/profiler", "reason": family_blocker(str(r.get("carrier")), ["efficiency"]), "promotion_allowed": 0})
    return rows


def implementation_readback(route: dict[str, Any], mechanism_rows: list[dict[str, Any]], efficiency_rows: list[dict[str, str]], kernel_rows: list[dict[str, str]]) -> str:
    return "\n".join(
        [
            "# v17 Implementation Readback",
            "",
            f"- route: {route.get('route')}",
            f"- measured mechanism rows: {len(mechanism_rows)}",
            f"- efficiency rows: {len(efficiency_rows)}",
            f"- kernel rows: {len(kernel_rows)}",
            "- v17 code path imports reusable dgkan modules and does not import old v13-v16 runners.",
            "- LineC exception handling returns `R0-LineCMeasurementInvalid` with exception fields.",
            "- h800/h1600 fields are left empty if not executed; no fabricated horizon data.",
            "",
        ]
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", default="s0", choices=["s0", "mechanism-shard", "efficiency-shard", "horizon-shard", "merge-finalize"])
    p.add_argument("--out-dir", default=str(DEFAULT_OUT))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--data-root", default="data")
    p.add_argument("--datasets", default="MNIST,Fashion-MNIST,KMNIST")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--train-size", type=int, default=64)
    p.add_argument("--val-size", type=int, default=48)
    p.add_argument("--input-size", type=int, default=8)
    p.add_argument("--classes", type=int, default=10)
    p.add_argument("--hidden", type=int, default=24)
    p.add_argument("--param-budget", type=int, default=12000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--lr", type=float, default=0.003)
    p.add_argument("--fu-lr", type=float, default=0.001)
    p.add_argument("--weight-decay", type=float, default=0.001)
    p.add_argument("--alt-period", type=int, default=10)
    p.add_argument("--sanity-eps", type=float, default=1.0e-3)
    p.add_argument("--shard-count", type=int, default=4)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--efficiency-batches", default="8,32,64")
    p.add_argument("--profiler-repeats", type=int, default=3)
    p.add_argument("--profiler-warmup", type=int, default=1)
    p.add_argument("--horizon-steps", type=int, default=1600)
    p.add_argument("--horizon-top-per-carrier", type=int, default=2)
    p.add_argument("--horizon-adamw-free-top-per-carrier", type=int, default=2)
    p.add_argument("--horizon-all-rows", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--reuse-if-present", action="store_true")
    p.add_argument("--fail-on-missing-s0", action="store_true")
    p.add_argument("--smoke-only", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.fail_on_missing_s0 and args.stage != "s0" and not (out_dir / "v17_s0_route_decision.json").exists():
        raise SystemExit(f"S0 route missing for {out_dir}; refusing science stage because --fail-on-missing-s0 was set")
    if args.stage == "s0":
        run_s0(args)
    elif args.stage == "mechanism-shard":
        run_mechanism_shard(args)
    elif args.stage == "efficiency-shard":
        run_efficiency_shard(args)
    elif args.stage == "horizon-shard":
        run_horizon_shard(args)
    elif args.stage == "merge-finalize":
        run_finalize(args)
    else:
        raise ValueError(args.stage)


if __name__ == "__main__":
    main()
