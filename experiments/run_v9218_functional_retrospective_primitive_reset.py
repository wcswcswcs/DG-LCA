#!/usr/bin/env python3
"""DG-KAN v9.2.18 functional retrospective and primitive reset runner.

This runner is intentionally source-artifact driven.  v8.x functional rows are
reanalyzed from the historical CSV/JSON artifacts that were actually produced.
When a v9-style strong control is absent from those source artifacts, the row is
written as not_measured_in_source_artifact instead of being imputed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dgkan.artifacts.audit import audit_no_fake  # noqa: E402
from dgkan.artifacts.writer import (  # noqa: E402
    artifact_hash_rows,
    ensure_dir,
    read_csv_rows,
    sha256_file,
    write_csv_rows,
    write_json,
)


PLAN_PATH = ROOT / "docs" / "DG-KAN_v9.2.18_Functional_Retrospective_Primitive_Reset_完整实验计划.md"
SCRIPT_PATH = ROOT / "experiments" / "run_v9218_functional_retrospective_primitive_reset.py"
RESULT_ROOT = ROOT / "results" / "real_rerun_20260506"

SRC_V87 = RESULT_ROOT / "v87_full_chain_fmnist_kmnist_selected_kw4_formal_repeat_20260508T203000Z"
SRC_V88 = RESULT_ROOT / "v88_wave0_p0_timing_p2_p3_p4_p6_probe3_warm1_20260509T023000Z"
SRC_V926 = RESULT_ROOT / "v926_fc_purekan_primitive_redesign_liftquad_compact_p5_20260509T174500Z"
SRC_V927 = RESULT_ROOT / "v927_fc_purekan_lq_fullpass_functional_gate_20260509T190000Z"
SRC_V9214 = RESULT_ROOT / "v9214_p4qualified_functional_actuator_closure_first_20260510T010000Z"
SRC_V9215 = RESULT_ROOT / "v9215_control_resistant_functional_causality_first_20260510T030000Z"
SRC_V9216 = RESULT_ROOT / "v9216_causal_target_discovery_first_20260510T040000Z"
SRC_V9217 = RESULT_ROOT / "v9217_signal_aligned_functional_or_reset_first_20260510T050000Z"

STRONG_CONTROLS = [
    "AdamWOnly",
    "FT7/AdaptiveFunctional",
    "NoOpMatchedOverhead",
    "RandomMatchedNorm",
    "ShuffledRoleMask",
    "InvertedRoleMask",
    "AdamWParallelSameNorm",
    "AdamWParallelTrustRatio-0.003",
    "AdamWParallelTrustRatio-0.01",
    "AdamWParallelTrustRatio-0.03",
    "LRScale-1.003",
    "LRScale-1.01",
    "LRScale-1.03",
    "LRScale-1.10",
]


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in ("", None, "nan", "NaN", "metric_unavailable"):
            return default
        return float(value)
    except Exception:
        return default


def _to_int(value: Any, default: int = 0) -> int:
    try:
        if value in ("", None, "metric_unavailable"):
            return default
        return int(float(value))
    except Exception:
        return default


def _mean(values: Sequence[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return sum(vals) / max(1, len(vals))


def _max(values: Sequence[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return max(vals) if vals else 0.0


def _count_rows(path: Path) -> int:
    return len(read_csv_rows(path))


def _missing_row(stage: str, artifact: str, reason: str, **extra: Any) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "stage": stage,
        "artifact": artifact,
        "status": "not_run",
        "reason": reason,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    row.update(extra)
    return row


def _write_svg(path: Path, title: str, lines: Sequence[str]) -> None:
    ensure_dir(path.parent)
    text = "\n".join(
        f'<text x="28" y="{90 + i * 26}" font-family="Arial, sans-serif" font-size="15" fill="#374151">{line}</text>'
        for i, line in enumerate(lines)
    )
    path.write_text(
        f"""<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="260" viewBox="0 0 1100 260">
  <rect width="1100" height="260" fill="#f8fafc"/>
  <text x="28" y="52" font-family="Arial, sans-serif" font-size="25" fill="#111827">{title}</text>
  {text}
  <text x="28" y="236" font-family="Arial, sans-serif" font-size="12" fill="#6b7280">Generated only from measured source CSV/JSON fields and explicit not-run rows.</text>
</svg>
""",
        encoding="utf-8",
    )


def _p0_history_audit() -> List[Dict[str, Any]]:
    v87 = _read_json(SRC_V87 / "route_decision.json")
    v88 = _read_json(SRC_V88 / "route_decision.json")
    v927 = _read_json(SRC_V927 / "route_decision.json")
    v9214 = _read_json(SRC_V9214 / "route_decision.json")
    v9215 = _read_json(SRC_V9215 / "route_decision.json")
    v9216 = _read_json(SRC_V9216 / "route_decision.json")
    v9217 = _read_json(SRC_V9217 / "route_decision.json")
    rows = [
        {
            "version": "v8.3",
            "route": "FT7-role-wise-functional-minimum",
            "candidate": "FT7",
            "architecture_family": "transitional packed linear-SiLU stack + KAN-style head",
            "strict_full_edge_purekan": 0,
            "transitional_route": 1,
            "functional_type": "role-wise guarded FT7",
            "controls_used": "NoOp/Random/role controls in historical protocol",
            "adamwparallel_control_used": 0,
            "lr_control_used": 0,
            "p4_pass": "historical_system_gate",
            "p5_nearpass": "not_strict_v9_contract",
            "p5_fullpass": "not_strict_v9_contract",
            "functional_pass": 1,
            "external_fair_pass": 0,
            "broad_strong_claimed": 0,
            "no_fake_proxy": 1,
            "classification": "Transitional functional success",
            "source": "doc_context_no_new_rows",
        },
        {
            "version": "v8.4",
            "route": "high-rep timing minimum",
            "candidate": "FT7",
            "architecture_family": "transitional packed linear-SiLU stack + KAN-style head",
            "strict_full_edge_purekan": 0,
            "transitional_route": 1,
            "functional_type": "formal functional diagnostic",
            "controls_used": "NoOp/Random/role controls",
            "adamwparallel_control_used": 0,
            "lr_control_used": 0,
            "p4_pass": "minimum_high_rep",
            "p5_nearpass": "not_strict_v9_contract",
            "p5_fullpass": "not_strict_v9_contract",
            "functional_pass": 1,
            "external_fair_pass": 0,
            "broad_strong_claimed": 0,
            "no_fake_proxy": 1,
            "classification": "Diagnostic assisted success",
            "source": "DG-KAN_v8.4 recap",
        },
        {
            "version": "v8.7",
            "route": v87.get("route", ""),
            "candidate": f"KW4-hidden{v87.get('p4_selected_hidden_dim', '')}",
            "architecture_family": "transitional packed linear-SiLU stack + poly2_silu KAN head",
            "strict_full_edge_purekan": 0,
            "transitional_route": 1,
            "functional_type": "Adaptive-FT-P / FT7 selected route",
            "controls_used": "NoOp/Random/adaptive controls; no AdamWParallel/LR in source",
            "adamwparallel_control_used": 0,
            "lr_control_used": 0,
            "p4_pass": v87.get("p4_selected_task_geometry_confirmation_pass", 0),
            "p5_nearpass": "not_strict_v9_contract",
            "p5_fullpass": "not_strict_v9_contract",
            "functional_pass": v87.get("success_v87_adaptive", 0),
            "external_fair_pass": v87.get("success_v87_external_fair", 0),
            "broad_strong_claimed": 0,
            "no_fake_proxy": 1,
            "classification": "Transitional functional success",
            "source": str(SRC_V87.relative_to(ROOT)),
        },
        {
            "version": "v8.8",
            "route": v88.get("route", ""),
            "candidate": f"KW4-hidden{v88.get('p4_selected_hidden_dim', '')}",
            "architecture_family": "transitional independent screen",
            "strict_full_edge_purekan": 0,
            "transitional_route": 1,
            "functional_type": "independent broad screen diagnostic",
            "controls_used": "v8 controls only; fresh P3 timing/broad not fully closed",
            "adamwparallel_control_used": 0,
            "lr_control_used": 0,
            "p4_pass": v88.get("p4_selected_task_geometry_confirmation_pass", 0),
            "p5_nearpass": "not_strict_v9_contract",
            "p5_fullpass": "not_strict_v9_contract",
            "functional_pass": v88.get("success_v87_adaptive", 0),
            "external_fair_pass": v88.get("success_v87_external_fair", 0),
            "broad_strong_claimed": 0,
            "no_fake_proxy": 1,
            "classification": "System-only / transitional diagnostic",
            "source": str(SRC_V88.relative_to(ROOT)),
        },
        {
            "version": "v9.2.6-v9.2.7",
            "route": v927.get("route", ""),
            "candidate": "LQ-t2-h256",
            "architecture_family": "strict FC-PureKAN LinearLiftQuadraticEdgeBasis",
            "strict_full_edge_purekan": 1,
            "transitional_route": 0,
            "functional_type": "none opened",
            "controls_used": "P4/P5 near-pass; functional not opened",
            "adamwparallel_control_used": 0,
            "lr_control_used": 0,
            "p4_pass": v927.get("p4_pass", 1),
            "p5_nearpass": v927.get("p5_near_pass", 1),
            "p5_fullpass": v927.get("p5_pass", 0),
            "functional_pass": 0,
            "external_fair_pass": 0,
            "broad_strong_claimed": 0,
            "no_fake_proxy": 1,
            "classification": "Strict FC-PureKAN near-pass base",
            "source": str(SRC_V927.relative_to(ROOT)),
        },
        {
            "version": "v9.2.14",
            "route": v9214.get("route", ""),
            "candidate": v9214.get("best_actuator_candidate", "A7c-BasisEntropy-ValueOnly"),
            "architecture_family": "strict FC-PureKAN LQ + P4 actuator",
            "strict_full_edge_purekan": 1,
            "transitional_route": 0,
            "functional_type": "actuator paired replay",
            "controls_used": "AdamWParallel/Random/Shuffled/NoOp/actuator controls",
            "adamwparallel_control_used": 1,
            "lr_control_used": 0,
            "p4_pass": v9214.get("p4_closure_pass", 1),
            "p5_nearpass": v9214.get("actuator_base_near_pass", 1),
            "p5_fullpass": 0,
            "functional_pass": v9214.get("paired_replay_pass", 0),
            "external_fair_pass": 0,
            "broad_strong_claimed": 0,
            "no_fake_proxy": 1,
            "classification": "No functional advantage",
            "source": str(SRC_V9214.relative_to(ROOT)),
        },
        {
            "version": "v9.2.15-v9.2.16",
            "route": v9216.get("route", v9215.get("route", "")),
            "candidate": "A4/A7c control-resistant matrix",
            "architecture_family": "strict FC-PureKAN actuator target discovery",
            "strict_full_edge_purekan": 1,
            "transitional_route": 0,
            "functional_type": "control-resistant target/solver replay",
            "controls_used": "AdamWParallel/LR-like/Random/Shuffled/NoOp",
            "adamwparallel_control_used": 1,
            "lr_control_used": 0,
            "p4_pass": 1,
            "p5_nearpass": 1,
            "p5_fullpass": 0,
            "functional_pass": 0,
            "external_fair_pass": 0,
            "broad_strong_claimed": 0,
            "no_fake_proxy": 1,
            "classification": "No functional advantage",
            "source": f"{SRC_V9215.name};{SRC_V9216.name}",
        },
        {
            "version": "v9.2.17",
            "route": v9217.get("route", ""),
            "candidate": v9217.get("best_functional_candidate", ""),
            "architecture_family": "strict FC-PureKAN signal-aligned functional audit",
            "strict_full_edge_purekan": 1,
            "transitional_route": 0,
            "functional_type": "signal-aligned metric vs LR controls",
            "controls_used": "AdamWParallel and scalar LR controls",
            "adamwparallel_control_used": 1,
            "lr_control_used": 1,
            "p4_pass": 1,
            "p5_nearpass": 1,
            "p5_fullpass": 0,
            "functional_pass": 0,
            "external_fair_pass": 0,
            "broad_strong_claimed": 0,
            "no_fake_proxy": 1,
            "classification": "No functional advantage / LR-equivalent signal",
            "source": str(SRC_V9217.relative_to(ROOT)),
        },
    ]
    for row in rows:
        row.update({"fake_data_used": 0, "proxy_row_used": 0, "cpu_offload_used": 0})
    return rows


def _p1_v8_strong_control_replay() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    raw_path = SRC_V87 / "adaptive_functional_one_step_raw.csv"
    raw = read_csv_rows(raw_path)
    grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in raw:
        grouped[row.get("controller_id", "")].append(row)
    mapping = {
        "Fixed-FT7-stride128": "FT7/AdaptiveFunctional",
        "Adaptive-FT-P": "FT7/AdaptiveFunctional",
        "NoOp": "NoOpMatchedOverhead",
        "RandomFunc": "RandomMatchedNorm",
    }
    for controller, branch in mapping.items():
        src = grouped.get(controller, [])
        if not src:
            continue
        rows.append(
            {
                "stage": "P1_V8_STRONG_CONTROL_REPLAY_V9218",
                "status": "source_measured_v87_one_step",
                "source_artifact": str(raw_path.relative_to(ROOT)),
                "candidate": "V8-KW3-hidden28-source-one-step",
                "dataset": "MNIST",
                "seed_scope": "1314,1315,1316",
                "horizon": 1,
                "branch": branch,
                "source_controller_id": controller,
                "rows": len(src),
                "test_acc": "not_measured_in_v87_one_step_raw",
                "delta_vs_adamw": "not_measured_in_v87_one_step_raw",
                "CEp99": "not_measured_in_v87_one_step_raw",
                "margin_p10": "not_measured_in_v87_one_step_raw",
                "wrong_confidence_p95": "not_measured_in_v87_one_step_raw",
                "ECE": "not_measured_in_v87_one_step_raw",
                "NLL": "not_measured_in_v87_one_step_raw",
                "curvature": _mean([_to_float(r.get("edge_curvature_norm_after")) for r in src]),
                "local_lipschitz": "not_measured_in_v87_one_step_raw",
                "functional_event_count": sum(_to_int(r.get("event_triggered")) for r in src),
                "event_coverage": _mean([_to_float(r.get("event_triggered")) for r in src]),
                "step_ratio_q90": "not_measured_in_v87_one_step_raw",
                "memory_ratio": "not_measured_in_v87_one_step_raw",
                "control_rank": "legacy_source_control_only",
                "actual_holdout_descent_mean": _mean([_to_float(r.get("actual_holdout_descent")) for r in src]),
                "holdout_descent_ratio_mean": _mean([_to_float(r.get("holdout_descent_ratio")) for r in src]),
                "bad_step_rate": _mean([_to_float(r.get("bad_step_flag")) for r in src]),
                "curvature_reduction_mean": _mean([_to_float(r.get("curvature_reduction")) for r in src]),
                "functional_update_norm_mean": _mean([_to_float(r.get("functional_update_norm")) for r in src]),
                "cos_corrected_with_task_mean": _mean([_to_float(r.get("cos_corrected_with_task")) for r in src]),
                "real_beats_adamwparallel": "not_evaluable_missing_adamwparallel_control",
                "real_beats_best_lr": "not_evaluable_missing_lr_control",
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            }
        )
    mult_path = SRC_V87 / "adaptive_functional_multistep.csv"
    mult = read_csv_rows(mult_path)
    for row in mult:
        rows.append(
            {
                "stage": "P1_V8_STRONG_CONTROL_REPLAY_V9218",
                "status": "source_measured_v87_multistep_legacy_controls",
                "source_artifact": str(mult_path.relative_to(ROOT)),
                "candidate": "V8-KW4-hidden28-selected-multistep",
                "dataset": row.get("dataset", ""),
                "seed_scope": row.get("seed", ""),
                "horizon": row.get("checkpoint_step", ""),
                "branch": row.get("controller_id", ""),
                "source_controller_id": row.get("controller_id", ""),
                "rows": 1,
                "test_acc": row.get("test_acc", ""),
                "delta_vs_adamw": "not_measured_in_v87_multistep_source",
                "CEp99": "not_measured_in_v87_multistep_source",
                "margin_p10": "not_measured_in_v87_multistep_source",
                "wrong_confidence_p95": "not_measured_in_v87_multistep_source",
                "ECE": row.get("ECE", ""),
                "NLL": row.get("NLL", ""),
                "curvature": row.get("curvature", ""),
                "local_lipschitz": "not_measured_in_v87_multistep_source",
                "functional_event_count": row.get("event_count", ""),
                "event_coverage": "not_directly_measured_in_v87_multistep_source",
                "step_ratio_q90": "not_measured_in_v87_multistep_source",
                "memory_ratio": "not_measured_in_v87_multistep_source",
                "control_rank": "legacy_source_control_only",
                "actual_holdout_descent_mean": "not_measured_in_v87_multistep_source",
                "holdout_descent_ratio_mean": row.get("holdout_descent_ratio_mean", ""),
                "bad_step_rate": row.get("bad_step_rate", ""),
                "curvature_reduction_mean": "not_measured_in_v87_multistep_source",
                "functional_update_norm_mean": "not_measured_in_v87_multistep_source",
                "cos_corrected_with_task_mean": "not_measured_in_v87_multistep_source",
                "real_beats_adamwparallel": "not_evaluable_missing_adamwparallel_control",
                "real_beats_best_lr": "not_evaluable_missing_lr_control",
                "fake_data_used": row.get("fake_data_used", 0),
                "proxy_row_used": row.get("proxy_row_used", 0),
                "cpu_offload_used": row.get("cpu_offload_used", 0),
            }
        )
    measured_branches = {str(r.get("branch")) for r in rows}
    for control in STRONG_CONTROLS:
        if control in measured_branches or control == "FT7/AdaptiveFunctional":
            continue
        rows.append(
            {
                "stage": "P1_V8_STRONG_CONTROL_REPLAY_V9218",
                "status": "not_measured_in_source_artifact",
                "source_artifact": str(raw_path.relative_to(ROOT)),
                "candidate": "V8-strong-control-required",
                "dataset": "MNIST,Fashion-MNIST,KMNIST",
                "seed_scope": "0,1,2,3,4",
                "horizon": "1,5,20,80",
                "branch": control,
                "source_controller_id": "not_available",
                "rows": 0,
                "test_acc": "",
                "delta_vs_adamw": "",
                "CEp99": "",
                "margin_p10": "",
                "wrong_confidence_p95": "",
                "ECE": "",
                "NLL": "",
                "curvature": "",
                "local_lipschitz": "",
                "functional_event_count": "",
                "event_coverage": "",
                "step_ratio_q90": "",
                "memory_ratio": "",
                "control_rank": "not_evaluable",
                "actual_holdout_descent_mean": "",
                "holdout_descent_ratio_mean": "",
                "bad_step_rate": "",
                "curvature_reduction_mean": "",
                "functional_update_norm_mean": "",
                "cos_corrected_with_task_mean": "",
                "real_beats_adamwparallel": "not_evaluable_missing_control",
                "real_beats_best_lr": "not_evaluable_missing_control",
                "reason": "v8 source artifacts do not contain this v9-style strong control",
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            }
        )
    return rows


def _p2_v8_mechanism_attribution(p1_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    raw = read_csv_rows(SRC_V87 / "adaptive_functional_one_step_raw.csv")
    groups: Dict[tuple[str, str], List[Dict[str, str]]] = defaultdict(list)
    for row in raw:
        groups[(row.get("controller_id", ""), "stack")].append(row)
        groups[(row.get("controller_id", ""), "head")].append(row)
    rows: List[Dict[str, Any]] = []
    for (controller, role), src in groups.items():
        if controller not in {"Fixed-FT7-stride128", "Adaptive-FT-P", "NoOp", "RandomFunc"}:
            continue
        rows.append(
            {
                "stage": "P2_V8_FUNCTIONAL_MECHANISM_ATTRIBUTION_V9218",
                "status": "source_measured_v87_one_step",
                "source_artifact": str((SRC_V87 / "adaptive_functional_one_step_raw.csv").relative_to(ROOT)),
                "controller_id": controller,
                "role": role,
                "role_update_norm": "not_directly_stored; role_update_share_available",
                "role_update_share_mean": _mean([_to_float(r.get(f"role_update_share", 0.0)) for r in src]),
                "role_snr": "not_measured_in_v87_source",
                "role_curvature": _mean([_to_float(r.get(f"role_curv_norm_{role}", 0.0)) for r in src]),
                "role_event_frequency": _mean([_to_float(r.get("event_triggered", 0.0)) for r in src]),
                "branch_ratio": "not_measured_in_v87_source",
                "effective_derivative_scale": "not_measured_in_v87_source",
                "cos_functional_adamw": _mean([_to_float(r.get("cos_functional_with_task", 0.0)) for r in src]),
                "cos_functional_random": "not_measured_as_pairwise_control",
                "cos_functional_lrcontrol": "not_measured_missing_lr_control",
                "functional_norm_vs_adamw": _mean([_to_float(r.get("functional_update_norm", 0.0)) for r in src]),
                "pre_holdout_loss": "not_stored_in_v87_source",
                "post_holdout_loss": "not_stored_in_v87_source",
                "CEp99_delta": "not_measured_in_v87_source",
                "margin_delta": "not_measured_in_v87_source",
                "curvature_delta": _mean([_to_float(r.get("curvature_reduction", 0.0)) for r in src]),
                "mechanism_classification": "positive_vs_legacy_controls_but_unresolved_vs_adamwparallel_lr",
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            }
        )
    return rows


def _p3_transfer_audit() -> List[Dict[str, Any]]:
    v9217 = _read_json(SRC_V9217 / "route_decision.json")
    v9214 = _read_json(SRC_V9214 / "route_decision.json")
    rows = [
        {
            "candidate": "T0-LQ-AdamWOnly",
            "status": "source_measured_v927_v9217",
            "full_edge_equivalence_pass": 1,
            "external_residual_used": 0,
            "ordinary_mlp_path_used": 0,
            "edge_owned_param_fraction": 1.0,
            "p4_forward_q90": "source_v927_p4_pass",
            "p4_backward_q90": "source_v927_p4_pass",
            "p4_step_q90": "source_v927_p4_pass",
            "p4_memory": "source_v927_compact_memory_pass",
            "p5_nearpass": 1,
            "paired_replay_real_vs_adamwparallel": "no_functional_update",
            "paired_replay_real_vs_lrcontrol": "no_functional_update",
            "branch_ratio": "not_applicable",
            "effective_derivative_scale": "not_applicable",
            "CEp99": "not_source_transfer_metric",
            "margin_p10": "not_source_transfer_metric",
            "ECE": "not_source_transfer_metric",
            "curvature": "not_source_transfer_metric",
            "transfer_pass": 0,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        },
        {
            "candidate": "T4-A7c-RoleSNRMetric",
            "status": "source_measured_v9214_v9217",
            "full_edge_equivalence_pass": 1,
            "external_residual_used": 0,
            "ordinary_mlp_path_used": 0,
            "edge_owned_param_fraction": 1.0,
            "p4_forward_q90": v9214.get("forward_ratio_q90", ""),
            "p4_backward_q90": v9214.get("backward_ratio_q90", ""),
            "p4_step_q90": v9214.get("step_ratio_q90", ""),
            "p4_memory": v9214.get("memory_ratio", ""),
            "p5_nearpass": v9214.get("actuator_base_near_pass", 1),
            "paired_replay_real_vs_adamwparallel": "fail_source_v9214_v9217",
            "paired_replay_real_vs_lrcontrol": "fail_source_v9217_lr_equivalence",
            "branch_ratio": "not_transferred",
            "effective_derivative_scale": "not_transferred",
            "CEp99": "functional_loses_to_controls",
            "margin_p10": "functional_loses_to_controls",
            "ECE": "not_source_transfer_metric",
            "curvature": "not_source_transfer_metric",
            "transfer_pass": 0,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        },
    ]
    for cid in [
        "T1-LQ-RoleSNRMetric",
        "T2-LQ-BranchActivityController",
        "T3-LQ-EffectiveDerivativeController",
        "T5-A4e-RoleSNRMetric",
        "T6-LQ-FT7StyleEventGuard",
    ]:
        rows.append(
            _missing_row(
                "P3_V8_V9_TRANSFER_AUDIT_V9218",
                "p3_v8_v9_transfer_audit.csv",
                "not_implemented_as_new_transfer_candidate_in_this_retrospective_runner",
                candidate=cid,
                transfer_pass=0,
            )
        )
    for row in rows:
        row.setdefault("stage", "P3_V8_V9_TRANSFER_AUDIT_V9218")
    return rows


def _p4_primitive_factory_reset() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    p926 = read_csv_rows(SRC_V926 / "p2_linear_lift_quadratic_p4_gate.csv")
    by_candidate = {r.get("candidate_id", ""): r for r in p926}
    route926 = _read_json(SRC_V926 / "route_decision.json")
    route9214 = _read_json(SRC_V9214 / "route_decision.json")
    for basis_id, src_name in [
        ("B0-LQ-T2-current", "LQ-t2-h256"),
        ("B3-LQ-Legendre2-only", "LQ-legendre23-h128"),
        ("B4-LQ-normalized-Legendre2", "LQ-legendre23-h256"),
    ]:
        src = by_candidate.get(src_name, {})
        rows.append(
            {
                "stage": "P4_PRIMITIVE_BASIS_FACTORY_RESET_V9218",
                "status": "source_measured_v926",
                "basis_family": basis_id,
                "source_candidate": src_name,
                "basis_formula": src.get("basis", ""),
                "derivative_formula": "source_LQ_manual_backward",
                "conditioning_pass": "not_reported_in_v926_p2",
                "basis_condition_number": "not_reported_in_v926_p2",
                "dominant_basis_fraction": "not_reported_in_v926_p2",
                "dead_basis_fraction": "not_reported_in_v926_p2",
                "synthetic_pairwise_R2": src.get("synthetic_pairwise_R2", route926.get("synthetic_pairwise_R2", "")),
                "local_bump_R2": "not_measured",
                "GradRelErrMax": src.get("GradRelErrMax", ""),
                "GradCosMin": src.get("GradCosMin", ""),
                "p4_forward_q90": src.get("forward_ratio", ""),
                "p4_backward_q90": src.get("backward_ratio", ""),
                "p4_step_q90": src.get("step_ratio", ""),
                "p4_memory": src.get("compact_memory_ratio", src.get("memory_ratio", "")),
                "p4_pass": src.get("P4_pass", src.get("p4_pass", "")),
                "p5_nearpass_rate": "8/9" if src_name == "LQ-t2-h256" else "",
                "macro_delta": route926.get("p5_macro_delta", ""),
                "KMNIST_delta": "source_trainability_rows_not_in_this_artifact",
                "functional_actuatability_R2": "not_measured_for_base",
                "non_adamw_output_displacement": "not_measured_for_base",
                "paired_replay_vs_adamwparallel": "not_measured_for_base",
                "paired_replay_vs_lrcontrol": "not_measured_for_base",
                "factory_pass": 0,
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            }
        )
    rows.append(
        {
            "stage": "P4_PRIMITIVE_BASIS_FACTORY_RESET_V9218",
            "status": "source_measured_v9214",
            "basis_family": "B9-LQ-functional-actuator-channel",
            "source_candidate": route9214.get("best_actuator_candidate", "A7c-BasisEntropy-ValueOnly"),
            "basis_formula": "basis_entropy_value_only_actuator",
            "derivative_formula": "value-only stopgrad basis, output-edge coeff trained",
            "conditioning_pass": "source_contract_pass",
            "basis_condition_number": "not_in_route_json",
            "dominant_basis_fraction": "not_in_route_json",
            "dead_basis_fraction": "not_in_route_json",
            "synthetic_pairwise_R2": "base_retained",
            "local_bump_R2": "not_measured",
            "GradRelErrMax": "source_p2_grad_pass",
            "GradCosMin": "source_p2_grad_pass",
            "p4_forward_q90": route9214.get("forward_ratio_q90", ""),
            "p4_backward_q90": route9214.get("backward_ratio_q90", ""),
            "p4_step_q90": route9214.get("step_ratio_q90", ""),
            "p4_memory": route9214.get("memory_ratio", ""),
            "p4_pass": route9214.get("p4_closure_pass", 1),
            "p5_nearpass_rate": route9214.get("actuator_base_near_pass", 1),
            "macro_delta": "source_v9214_macro_delta=-0.0041666627",
            "KMNIST_delta": "KMNIST_seed0_still_miss",
            "functional_actuatability_R2": route9214.get("target_fit_R2", ""),
            "non_adamw_output_displacement": route9214.get("output_displacement_ratio_rz", ""),
            "paired_replay_vs_adamwparallel": "fail_source_v9214_v9215_v9216",
            "paired_replay_vs_lrcontrol": "fail_source_v9217",
            "factory_pass": 0,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
    )
    for cid in [
        "B1-LQ-centered-T2",
        "B2-LQ-normalized-T2",
        "B5-LQ-bounded-rational-base",
        "B6-LQ-piecewise-linear2-base",
        "B7-LQ-shared-RBF4-base",
        "B8-LQ-mixed-T2-rational",
    ]:
        rows.append(
            _missing_row(
                "P4_PRIMITIVE_BASIS_FACTORY_RESET_V9218",
                "p4_primitive_basis_factory_reset.csv",
                "not_implemented_as_base_primitive_in_this_retrospective_runner",
                basis_family=cid,
                factory_pass=0,
            )
        )
    return rows


def _p5_adamw_fullpass_repair() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    route927 = _read_json(SRC_V927 / "route_decision.json")
    route9214 = _read_json(SRC_V9214 / "route_decision.json")
    repairs = read_csv_rows(SRC_V927 / "p5_fullpass_repair_candidates.csv")
    rows.append(
        {
            "stage": "P5_ADAMW_ONLY_FULLPASS_REPAIR_V9218",
            "status": "source_measured_v927_robust_nearpass",
            "candidate": "LQ-t2-h256",
            "dataset": "MNIST,Fashion-MNIST,KMNIST",
            "seed": "0..9",
            "test_acc": "see source p3_robust_nearpass_confirmation.csv",
            "delta_vs_mlp": route927.get("p5_macro_delta", ""),
            "near_pass": route927.get("p5_near_pass", 1),
            "full_pass": route927.get("p5_pass", 0),
            "CEp99": "source_attribution_rows_only",
            "margin_p10": "source_attribution_rows_only",
            "ECE": "not_in_route_json",
            "NLL": "not_in_route_json",
            "basis_entropy": "source_p4_attribution",
            "lift_condition_number": "source_p4_attribution",
            "effective_rank": "not_measured",
            "p4_step_q90": "source_p4_pass",
            "memory_ratio": route927.get("compact_memory_ratio", ""),
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
    )
    rows.append(
        {
            "stage": "P5_ADAMW_ONLY_FULLPASS_REPAIR_V9218",
            "status": "source_measured_v9214_actuator_base",
            "candidate": "A7c-BasisEntropy-ValueOnly",
            "dataset": "MNIST,Fashion-MNIST,KMNIST",
            "seed": "0,1,2",
            "test_acc": "see source p4_p4_qualified_actuator_base_qualification.csv",
            "delta_vs_mlp": "-0.0041666627",
            "near_pass": route9214.get("actuator_base_near_pass", 1),
            "full_pass": 0,
            "CEp99": "not_measured_in_base_qualification",
            "margin_p10": "not_measured_in_base_qualification",
            "ECE": "not_measured_in_base_qualification",
            "NLL": "not_measured_in_base_qualification",
            "basis_entropy": "actuator_basis_entropy",
            "lift_condition_number": "not_measured_in_base_qualification",
            "effective_rank": "not_measured_in_base_qualification",
            "p4_step_q90": route9214.get("step_ratio_q90", ""),
            "memory_ratio": route9214.get("memory_ratio", ""),
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
    )
    for r in repairs:
        rows.append(
            {
                "stage": "P5_ADAMW_ONLY_FULLPASS_REPAIR_V9218",
                "status": "source_measured_v927_repair_smoke",
                "candidate": r.get("candidate_id", r.get("candidate", "")),
                "dataset": "MNIST,Fashion-MNIST,KMNIST",
                "seed": "0,1,2",
                "test_acc": "see source p5_fullpass_repair_candidates.csv",
                "delta_vs_mlp": r.get("macro_delta", ""),
                "near_pass": r.get("near_pass", ""),
                "full_pass": r.get("full_pass", 0),
                "CEp99": "not_measured_in_repair_smoke",
                "margin_p10": "not_measured_in_repair_smoke",
                "ECE": "not_measured_in_repair_smoke",
                "NLL": "not_measured_in_repair_smoke",
                "basis_entropy": "not_measured_in_repair_smoke",
                "lift_condition_number": "not_measured_in_repair_smoke",
                "effective_rank": "not_measured_in_repair_smoke",
                "p4_step_q90": r.get("P4_pass", ""),
                "memory_ratio": "source_p4_pass_flag_only",
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            }
        )
    return rows


def _contract_rows() -> List[Dict[str, Any]]:
    rows = []
    for item in [
        "loss_type_CE",
        "label_smoothing_zero",
        "teacher_off",
        "sampler_class_weight_off",
        "cpu_offload_off",
        "fake_proxy_off",
        "PureKANConv_Former_deferred",
        "source_rows_not_imputed",
    ]:
        rows.append(
            {
                "stage": "CONTRACT_AUDIT_V9218",
                "contract_item": item,
                "pass": 1,
                "loss_type": "CE",
                "label_smoothing": 0,
                "external_teacher_used": 0,
                "self_teacher_used": 0,
                "geometry_loss_used": 0,
                "sampler_changed": 0,
                "class_weight_used": 0,
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
                "uses_loss_backward": 0,
                "reason": "retrospective source-artifact audit; missing strong controls are explicit not_measured rows",
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DG-KAN v9.2.18 functional retrospective primitive reset")
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--seed", type=int, default=1314)
    return parser.parse_args()


def run(args: argparse.Namespace) -> Dict[str, Any]:
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    if args.fresh and out_dir.exists():
        shutil.rmtree(out_dir)
    ensure_dir(out_dir)
    fig_dir = ensure_dir(out_dir / "figures")

    manifest = {
        "stage": "v9218_functional_retrospective_primitive_reset",
        "created_at": _now_iso(),
        "plan_path": str(PLAN_PATH),
        "script_path": str(SCRIPT_PATH),
        "device_arg": args.device,
        "data_root": args.data_root,
        "seed": args.seed,
        "source_artifacts": {
            "v87": str(SRC_V87),
            "v88": str(SRC_V88),
            "v926": str(SRC_V926),
            "v927": str(SRC_V927),
            "v9214": str(SRC_V9214),
            "v9215": str(SRC_V9215),
            "v9216": str(SRC_V9216),
            "v9217": str(SRC_V9217),
        },
    }
    write_json(out_dir / "run_manifest.json", manifest)

    contract_rows = _contract_rows()
    p0_rows = _p0_history_audit()
    p1_rows = _p1_v8_strong_control_replay()
    p2_rows = _p2_v8_mechanism_attribution(p1_rows)
    p3_rows = _p3_transfer_audit()
    p4_rows = _p4_primitive_factory_reset()
    p5_rows = _p5_adamw_fullpass_repair()

    write_csv_rows(out_dir / "contract_audit_v9218.csv", contract_rows)
    write_csv_rows(out_dir / "p0_history_unified_audit.csv", p0_rows)
    write_csv_rows(out_dir / "p1_v8_strong_control_replay.csv", p1_rows)
    write_csv_rows(out_dir / "p2_v8_functional_mechanism_attribution.csv", p2_rows)
    write_csv_rows(out_dir / "p3_v8_v9_transfer_audit.csv", p3_rows)
    write_csv_rows(out_dir / "p4_primitive_basis_factory_reset.csv", p4_rows)
    write_csv_rows(out_dir / "p5_adamw_only_fullpass_repair.csv", p5_rows)

    missing_strong = [
        r for r in p1_rows if r.get("status") == "not_measured_in_source_artifact"
    ]
    v8_strong_evaluable = int(len(missing_strong) == 0)
    v8_survives = 0
    v8_lr_equiv = 0
    v9_transfer_pass = int(any(_to_int(r.get("transfer_pass")) for r in p3_rows))
    primitive_factory_pass = int(any(_to_int(r.get("factory_pass")) for r in p4_rows))
    adamw_fullpass = int(any(_to_int(r.get("full_pass")) for r in p5_rows))
    functional_actuatability_pass = int(
        any(str(r.get("paired_replay_vs_adamwparallel", "")).startswith("pass") for r in p4_rows)
    )
    external_ready = int(v8_survives and v9_transfer_pass and adamw_fullpass)

    if not v8_strong_evaluable:
        route = "R8-FunctionalPaused"
        primary = "v8_success_not_reaudited_with_v9_style_adamwparallel_lr_controls_in_available_source_artifacts"
        next_impl = "implement_true_v8_strong_control_replay_or_continue_primitive_basis_factory_reset"
    elif v8_lr_equiv:
        route = "R2-v8FunctionalIsLREquivalent"
        primary = "v8_functional_explained_by_lr_controls"
        next_impl = "return_to_primitive_basis_factory"
    elif v8_survives and not v9_transfer_pass:
        route = "R3-FunctionalConceptValid_PrimitiveMismatch"
        primary = "v8_survives_but_v9_transfer_fails"
        next_impl = "strict_fc_purekan_actuatability_primitive_redesign"
    elif adamw_fullpass:
        route = "R5-AdamWFullPassNoFunctional"
        primary = "base_fullpass_without_functional"
        next_impl = "external_fair_after_strong_baseline_gate"
    elif primitive_factory_pass:
        route = "R6-ReturnToPrimitiveBasisFactory"
        primary = "primitive_factory_has_candidate_but_functional_not_ready"
        next_impl = "P5_fullpass_and_control_resistant_replay"
    else:
        route = "R6-ReturnToPrimitiveBasisFactory"
        primary = "current_functional_family_exhausted_and_no_adamw_fullpass"
        next_impl = "new_fc_purekan_primitive_basis_factory_with_control_resistant_actuatability"

    route_decision = {
        "route": route,
        "base_candidate": "LQ-t2-h256",
        "v8_strong_control_replay_evaluable": v8_strong_evaluable,
        "v8_missing_strong_control_count": len(missing_strong),
        "v8_source_legacy_measured_rows": len([r for r in p1_rows if str(r.get("status", "")).startswith("source_measured")]),
        "v8_survives_controls": v8_survives,
        "v8_lr_equivalent": v8_lr_equiv,
        "v9_transfer_pass": v9_transfer_pass,
        "primitive_factory_pass": primitive_factory_pass,
        "adamw_fullpass": adamw_fullpass,
        "functional_actuatability_pass": functional_actuatability_pass,
        "external_ready": external_ready,
        "primary_blocker": primary,
        "next_required_implementation": next_impl,
        "success_v9218_functional_retained": int(v8_survives or v9_transfer_pass),
        "success_v9218_primitive_reset": primitive_factory_pass,
        "success_v9218_external_ready": external_ready,
    }
    write_json(out_dir / "route_decision.json", route_decision)
    write_csv_rows(out_dir / "p6_route_decision.csv", [route_decision])

    aggregate = {
        "p0_rows": len(p0_rows),
        "p1_rows": len(p1_rows),
        "p2_rows": len(p2_rows),
        "p3_rows": len(p3_rows),
        "p4_rows": len(p4_rows),
        "p5_rows": len(p5_rows),
        "p1_legacy_source_measured_rows": route_decision["v8_source_legacy_measured_rows"],
        "p1_missing_strong_control_rows": len(missing_strong),
        "route": route,
    }
    write_json(out_dir / "aggregate_decision.json", aggregate)

    failure_rows = [
        {
            "stage": "P1",
            "failure": "missing_v9_style_strong_controls_in_v8_source_artifacts",
            "severity": "terminal_for_v8_survival_claim",
            "count": len(missing_strong),
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        },
        {
            "stage": "P3-P5",
            "failure": "v9_current_family_has_no_control_resistant_functional_advantage_or_fullpass_source",
            "severity": "terminal_for_external_ready",
            "count": 1,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        },
    ]
    write_csv_rows(out_dir / "failure_table.csv", failure_rows)

    _write_svg(
        fig_dir / "p0_version_route_lattice.svg",
        "v9.2.18 Route Lattice",
        [
            "v8.x = transitional functional success under legacy controls.",
            "v9.x LQ/A4/A7c = strict FC-PureKAN near-pass/P4, but no control-resistant functional advantage.",
            "v8 strong-control replay is not evaluable from existing source artifacts.",
        ],
    )
    _write_svg(
        fig_dir / "p1_v8_functional_vs_strong_controls.svg",
        "v8 Functional vs Strong Controls",
        [
            f"Legacy source-measured rows: {route_decision['v8_source_legacy_measured_rows']}.",
            f"Missing v9-style strong control rows: {len(missing_strong)}.",
            "No AdamWParallel/LR survival claim is made.",
        ],
    )
    _write_svg(
        fig_dir / "p4_basis_factory_pareto.svg",
        "Primitive Factory Source Pareto",
        [
            "LQ/A7c source candidates retain P4/P5 near-pass signals.",
            "Existing functional replay remains control-equivalent.",
            "New base primitive candidates are not implemented in this retrospective runner.",
        ],
    )

    audited_paths = [
        out_dir / "contract_audit_v9218.csv",
        out_dir / "p0_history_unified_audit.csv",
        out_dir / "p1_v8_strong_control_replay.csv",
        out_dir / "p2_v8_functional_mechanism_attribution.csv",
        out_dir / "p3_v8_v9_transfer_audit.csv",
        out_dir / "p4_primitive_basis_factory_reset.csv",
        out_dir / "p5_adamw_only_fullpass_repair.csv",
        out_dir / "p6_route_decision.csv",
        out_dir / "failure_table.csv",
    ]
    audit = audit_no_fake(audited_paths)
    write_csv_rows(out_dir / "v9218_provenance_audit.csv", [audit])
    hash_rows = artifact_hash_rows(
        [
            PLAN_PATH,
            SCRIPT_PATH,
            out_dir / "route_decision.json",
            out_dir / "aggregate_decision.json",
            *audited_paths,
            out_dir / "v9218_provenance_audit.csv",
        ],
        root=ROOT,
    )
    write_csv_rows(out_dir / "artifact_hashes_v9218.csv", hash_rows)

    return route_decision


def main() -> None:
    args = parse_args()
    decision = run(args)
    print(json.dumps(decision, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
