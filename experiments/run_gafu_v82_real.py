#!/usr/bin/env python3
"""DG-KAN v8.2 KC6/KF4 CE-only architecture-system convergence runner.

This runner keeps the v8.1 measured CE-only/no-teacher/no-loss path and writes
v8.2-specific convergence artifacts.  It does not introduce external teachers,
self teachers, loss changes, sampler/class-weight changes, or CPU offload.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import run_gafu_v72_real as v72
import run_gafu_v80_real as v80
import run_gafu_v81_real as v81
from dgkan_core import parse_int_list, parse_str_list, save_json, write_csv
from run_gafu_v66_real import _git_commit, _git_status


PLAN_PATH = "docs/DG-KAN_v8.2_CEOnly_KC6_KF4_ArchitectureSystemConvergence_完整实验计划.md"
SCRIPT_PATH = "experiments/run_gafu_v82_real.py"
METRIC_UNAVAILABLE = v81.METRIC_UNAVAILABLE

FOCUS_IDS = {"KC6", "KF4", "KF5", "KF6", "KF7", "KF8", "KF9", "KF10", "KW1", "KW2", "KW3", "KW4", "KW5", "KW6"}
QUALITY_ANCHOR_ID = "KC6"
SYSTEM_ANCHOR_ID = "KF4"


def _read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _float(value: Any, default: float = float("nan")) -> float:
    try:
        if value is None:
            return default
        text = str(value).strip()
        if text == "" or text.lower() in {
            "nan",
            "none",
            "metric_unavailable",
            "not_run",
            "not_implemented",
        }:
            return default
        return float(text)
    except Exception:
        return default


def _is_one(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "1.0", "true", "yes"}


def _hash_file(path: Path) -> str:
    if not path.exists():
        return "missing"
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _factor_label(candidate_id: str) -> str:
    return {
        "KC6": "quality_anchor_explicit_derivative_default_adamw",
        "KF4": "system_anchor_full_aten_silu_backward_adamw_addcdiv",
        "KF5": "explicit_inplace_derivative_default_adamw_merge_attempt",
        "KF6": "explicit_inplace_derivative_adamw_addcdiv_merge_attempt",
        "KF7": "upper_layer_only_aten_silu_backward_default_adamw",
        "KF8": "upper_layer_only_aten_silu_backward_adamw_addcdiv",
        "KF9": "lower_layer_only_aten_silu_backward_default_adamw",
        "KF10": "lower_layer_only_aten_silu_backward_adamw_addcdiv",
        "KW1": "no_input_grad_materialization_explicit_derivative_default_adamw",
        "KW2": "no_input_grad_materialization_explicit_derivative_adamw_addcdiv",
        "KW3": "no_input_grad_materialization_lower_layer_aten_adamw_addcdiv",
        "KW4": "cached_workspace_no_input_grad_lower_layer_aten_adamw_addcdiv",
        "KW5": "trajectory_preserving_compiled_explicit_no_input_grad_lower_layer_aten_adamw_addcdiv",
        "KW6": "trajectory_preserving_fast_mix_view_no_input_grad_lower_layer_aten_adamw_addcdiv",
    }.get(candidate_id, "non_focus_or_reference")


def _score_by_candidate(out_dir: Path) -> Dict[str, Dict[str, Any]]:
    rows = _read_csv_rows(out_dir / "p6_official_coselection.csv")
    return {str(row.get("candidate_id")): row for row in rows}


def _candidate_passes(row: Dict[str, Any]) -> Dict[str, int]:
    return {
        "contract": int(_is_one(row.get("NoTeacherNoLossModificationPass")) and _is_one(row.get("StrictPass"))),
        "grad": int(_is_one(row.get("GradPass"))),
        "macro": int(_is_one(row.get("TeacherFreeCEMacroPass"))),
        "s2": int(_is_one(row.get("FullGridS2Pass"))),
        "s1": int(_is_one(row.get("FullGridS1Pass"))),
        "time_auc": int(_is_one(row.get("TimeAUCPass"))),
    }


def _write_candidate_registry(out_dir: Path) -> None:
    rows = _read_csv_rows(out_dir / "candidate_registry_v3.csv")
    out = []
    for row in rows:
        cid = str(row.get("candidate_id"))
        out.append({
            **row,
            "stage": "P0_CANDIDATE_REGISTRY_V82",
            "v82_focus_candidate": int(cid in FOCUS_IDS),
            "v82_factor_label": _factor_label(cid),
            "cpu_offload_allowed": 0,
            "fake_data_used": 0,
            "proxy_row_used": 0,
        })
    if not out:
        out.append({
            "stage": "P0_CANDIDATE_REGISTRY_V82",
            "status": "missing_v81_registry",
            "fake_data_used": 0,
            "proxy_row_used": 0,
        })
    write_csv(out_dir / "candidate_registry_v82.csv", out)


def _write_contract(out_dir: Path) -> None:
    score_rows = _read_csv_rows(out_dir / "p6_official_coselection.csv")
    rows = []
    for row in score_rows:
        cid = str(row.get("candidate_id"))
        official = _is_one(row.get("official_eligible"))
        no_teacher_loss = _is_one(row.get("NoTeacherNoLossModificationPass"))
        strict = _is_one(row.get("StrictPass"))
        rows.append({
            "stage": "P0_CONTRACT_NO_TEACHER_NO_LOSS_V82",
            "candidate_id": cid,
            "external_teacher_used": row.get("external_teacher_used", METRIC_UNAVAILABLE),
            "self_teacher_used": row.get("self_teacher_used", METRIC_UNAVAILABLE),
            "loss_type": row.get("loss_type", METRIC_UNAVAILABLE),
            "special_loss_used": row.get("special_loss_used", METRIC_UNAVAILABLE),
            "distill_loss_used": row.get("distill_loss_used", METRIC_UNAVAILABLE),
            "sampler_changed": row.get("sampler_changed", METRIC_UNAVAILABLE),
            "class_weight_used": row.get("class_weight_used", METRIC_UNAVAILABLE),
            "cpu_offload_used": 0,
            "official_eligible": int(official),
            "NoTeacherNoLossModificationPass": int(no_teacher_loss),
            "StrictPass": int(strict),
            "contract_pass": int((not official) or (no_teacher_loss and strict)),
            "fake_data_used": 0,
            "proxy_row_used": 0,
        })
    if not rows:
        rows.append({
            "stage": "P0_CONTRACT_NO_TEACHER_NO_LOSS_V82",
            "status": "missing_p6_official_coselection",
            "fake_data_used": 0,
            "proxy_row_used": 0,
        })
    write_csv(out_dir / "contract_no_teacher_no_loss.csv", rows)
    write_csv(out_dir / "p0_contract_audit.csv", rows)


def _write_reproduction(out_dir: Path) -> None:
    score_rows = _read_csv_rows(out_dir / "p6_official_coselection.csv")
    rows = []
    for row in score_rows:
        cid = str(row.get("candidate_id"))
        if cid in {"B0"} | FOCUS_IDS:
            rows.append({
                **row,
                "stage": "P1_REPRODUCTION_V82",
                "v82_factor_label": _factor_label(cid),
                "cpu_offload_used": 0,
                "fake_data_used": 0,
                "proxy_row_used": 0,
            })
    if not rows:
        rows.append({
            "stage": "P1_REPRODUCTION_V82",
            "status": "not_run",
            "reason": "no_focus_candidate_measured",
            "fake_data_used": 0,
            "proxy_row_used": 0,
        })
    write_csv(out_dir / "p1_reproduction.csv", rows)


def _trace_key(row: Dict[str, Any]) -> Tuple[str, str, str]:
    return str(row.get("dataset")), str(row.get("seed")), str(row.get("step"))


def _write_trajectory_divergence(out_dir: Path) -> None:
    trace_rows = _read_csv_rows(out_dir / "p9_task_trace.csv")
    base_by_key = {
        _trace_key(row): row
        for row in trace_rows
        if str(row.get("candidate_id")) == QUALITY_ANCHOR_ID
    }
    rows = []
    for row in trace_rows:
        cid = str(row.get("candidate_id"))
        if cid not in FOCUS_IDS or cid == QUALITY_ANCHOR_ID:
            continue
        base = base_by_key.get(_trace_key(row))
        if not base:
            continue
        rows.append({
            "stage": "P2_TRAJECTORY_DIVERGENCE_V82",
            "reference_candidate_id": QUALITY_ANCHOR_ID,
            "candidate_id": cid,
            "v82_factor_label": _factor_label(cid),
            "dataset": row.get("dataset"),
            "seed": row.get("seed"),
            "step": row.get("step"),
            "train_loss": row.get("train_loss"),
            "reference_train_loss": base.get("train_loss"),
            "train_loss_delta_vs_KC6": _float(row.get("train_loss")) - _float(base.get("train_loss")),
            "val_loss": row.get("val_loss"),
            "reference_val_loss": base.get("val_loss"),
            "val_loss_delta_vs_KC6": _float(row.get("val_loss")) - _float(base.get("val_loss")),
            "val_acc": row.get("val_acc"),
            "reference_val_acc": base.get("val_acc"),
            "val_acc_delta_vs_KC6": _float(row.get("val_acc")) - _float(base.get("val_acc")),
            "update_norm": row.get("update_norm", METRIC_UNAVAILABLE),
            "reference_update_norm": base.get("update_norm", METRIC_UNAVAILABLE),
            "update_norm_delta_vs_KC6": (
                _float(row.get("update_norm")) - _float(base.get("update_norm"))
                if math.isfinite(_float(row.get("update_norm"))) and math.isfinite(_float(base.get("update_norm")))
                else METRIC_UNAVAILABLE
            ),
            "wall_clock_time_sec": row.get("wall_clock_time_sec", METRIC_UNAVAILABLE),
            "reference_wall_clock_time_sec": base.get("wall_clock_time_sec", METRIC_UNAVAILABLE),
            "fake_data_used": 0,
            "proxy_row_used": 0,
        })
    if not rows:
        rows.append({
            "stage": "P2_TRAJECTORY_DIVERGENCE_V82",
            "status": "not_run",
            "reason": "no_matching_KC6_trace_rows_available",
            "fake_data_used": 0,
            "proxy_row_used": 0,
        })
    write_csv(out_dir / "p2_trajectory_divergence.csv", rows)


def _dataset_gap_rows(out_dir: Path) -> List[Dict[str, Any]]:
    rows = []
    for row in _read_csv_rows(out_dir / "p1_significance_audit.csv"):
        cid = str(row.get("candidate_id"))
        dataset = str(row.get("dataset"))
        if cid in FOCUS_IDS and dataset:
            rows.append(row)
    return rows


def _write_factor_isolation(out_dir: Path) -> None:
    score = _score_by_candidate(out_dir)
    dataset_rows = _dataset_gap_rows(out_dir)
    dataset_by = {(str(r.get("candidate_id")), str(r.get("dataset"))): r for r in dataset_rows}
    rows = []
    for cid in sorted(FOCUS_IDS, key=lambda x: (len(x), x)):
        row = score.get(cid)
        if not row:
            continue
        base = score.get(QUALITY_ANCHOR_ID, {})
        sys_base = score.get(SYSTEM_ANCHOR_ID, {})
        rows.append({
            "stage": "P3_FACTOR_ISOLATION_V82",
            "candidate_id": cid,
            "factor_label": _factor_label(cid),
            "macro_gap": row.get("macro_gap", METRIC_UNAVAILABLE),
            "macro_delta_vs_KC6": _float(row.get("macro_gap")) - _float(base.get("macro_gap")),
            "macro_delta_vs_KF4": _float(row.get("macro_gap")) - _float(sys_base.get("macro_gap")),
            "memory_ratio_max": row.get("memory_ratio_max", METRIC_UNAVAILABLE),
            "memory_delta_vs_KC6": _float(row.get("memory_ratio_max")) - _float(base.get("memory_ratio_max")),
            "memory_delta_vs_KF4": _float(row.get("memory_ratio_max")) - _float(sys_base.get("memory_ratio_max")),
            "step_ratio_max": row.get("step_ratio_max", METRIC_UNAVAILABLE),
            "step_delta_vs_KC6": _float(row.get("step_ratio_max")) - _float(base.get("step_ratio_max")),
            "step_delta_vs_KF4": _float(row.get("step_ratio_max")) - _float(sys_base.get("step_ratio_max")),
            "GradPass": row.get("GradPass", METRIC_UNAVAILABLE),
            "TeacherFreeCEMacroPass": row.get("TeacherFreeCEMacroPass", METRIC_UNAVAILABLE),
            "FullGridS2Pass": row.get("FullGridS2Pass", METRIC_UNAVAILABLE),
            "KMNIST_gap": dataset_by.get((cid, "KMNIST"), {}).get("mean_val_gap", METRIC_UNAVAILABLE),
            "KMNIST_delta_vs_KC6": _float(dataset_by.get((cid, "KMNIST"), {}).get("mean_val_gap")) - _float(dataset_by.get((QUALITY_ANCHOR_ID, "KMNIST"), {}).get("mean_val_gap")),
            "Fashion_MNIST_gap": dataset_by.get((cid, "Fashion-MNIST"), {}).get("mean_val_gap", METRIC_UNAVAILABLE),
            "MNIST_gap": dataset_by.get((cid, "MNIST"), {}).get("mean_val_gap", METRIC_UNAVAILABLE),
            "fake_data_used": 0,
            "proxy_row_used": 0,
        })
    if not rows:
        rows.append({
            "stage": "P3_FACTOR_ISOLATION_V82",
            "status": "not_run",
            "reason": "no_focus_candidate_scores_available",
            "fake_data_used": 0,
            "proxy_row_used": 0,
        })
    write_csv(out_dir / "p3_factor_isolation.csv", rows)


def _write_liveset_and_allocator(out_dir: Path) -> None:
    live = _read_csv_rows(out_dir / "p11_gpu_live_allocator_v81.csv")
    focus_live = [
        {
            **row,
            "stage": "P4_GPU_LIVESET_ATTRIBUTION_V82",
            "v82_factor_label": _factor_label(str(row.get("candidate_id"))),
            "cpu_offload_used": row.get("cpu_offload_used", 0),
            "fake_data_used": 0,
            "proxy_row_used": 0,
        }
        for row in live
        if str(row.get("candidate_id")) in {"B0"} | FOCUS_IDS
    ]
    if not focus_live:
        focus_live = [{
            "stage": "P4_GPU_LIVESET_ATTRIBUTION_V82",
            "status": "not_run",
            "reason": "gpu_live_allocator_rows_unavailable",
            "fake_data_used": 0,
            "proxy_row_used": 0,
        }]
    write_csv(out_dir / "p4_gpu_liveset_attribution.csv", focus_live)

    by_key = {
        (str(row.get("candidate_id")), str(row.get("dataset")), str(row.get("batch_size"))): row
        for row in live
    }
    rows = []
    for cid in FOCUS_IDS:
        for dataset in parse_str_list(getattr(_LAST_ARGS, "datasets", "MNIST,Fashion-MNIST,KMNIST")):
            for batch in parse_int_list(getattr(_LAST_ARGS, "bench_batch_sizes", "128,256,512")):
                cur = by_key.get((cid, dataset, str(batch)))
                base = by_key.get((QUALITY_ANCHOR_ID, dataset, str(batch)))
                sys_base = by_key.get((SYSTEM_ANCHOR_ID, dataset, str(batch)))
                if not cur:
                    continue
                rows.append({
                    "stage": "P5_ALLOCATOR_SENSITIVITY_V82",
                    "candidate_id": cid,
                    "dataset": dataset,
                    "batch_size": batch,
                    "top_peak_phase": cur.get("top_peak_phase", METRIC_UNAVAILABLE),
                    "top_peak_allocated_MB": cur.get("top_peak_allocated_MB", METRIC_UNAVAILABLE),
                    "top_peak_delta_MB_vs_KC6": _float(cur.get("top_peak_allocated_MB")) - _float(base.get("top_peak_allocated_MB") if base else None),
                    "top_peak_delta_MB_vs_KF4": _float(cur.get("top_peak_allocated_MB")) - _float(sys_base.get("top_peak_allocated_MB") if sys_base else None),
                    "stack_backward_peak_allocated_MB": cur.get("stack_backward_peak_allocated_MB", METRIC_UNAVAILABLE),
                    "update_peak_allocated_MB": cur.get("update_peak_allocated_MB", METRIC_UNAVAILABLE),
                    "manual_cache_MB": cur.get("manual_cache_MB", METRIC_UNAVAILABLE),
                    "optimizer_state_MB": cur.get("optimizer_state_MB", METRIC_UNAVAILABLE),
                    "cpu_offload_used": cur.get("cpu_offload_used", 0),
                    "fake_data_used": 0,
                    "proxy_row_used": 0,
                })
    if not rows:
        rows.append({
            "stage": "P5_ALLOCATOR_SENSITIVITY_V82",
            "status": "not_run",
            "reason": "no_gpu_live_allocator_rows_for_focus_candidates",
            "fake_data_used": 0,
            "proxy_row_used": 0,
        })
    write_csv(out_dir / "p5_allocator_sensitivity.csv", rows)


_LAST_ARGS: argparse.Namespace


def _write_repair_tables(out_dir: Path) -> None:
    score_rows = _read_csv_rows(out_dir / "p6_official_coselection.csv")
    quality_rows = []
    system_rows = []
    for row in score_rows:
        cid = str(row.get("candidate_id"))
        if cid not in FOCUS_IDS:
            continue
        out = {
            **row,
            "v82_factor_label": _factor_label(cid),
            "cpu_offload_used": 0,
            "fake_data_used": 0,
            "proxy_row_used": 0,
        }
        if _is_one(row.get("TeacherFreeCEMacroPass")):
            quality_rows.append(out | {"stage": "P6_QUALITY_PRESERVING_S2_REPAIR_V82"})
        if _is_one(row.get("FullGridS2Pass")):
            system_rows.append(out | {"stage": "P7_SYSTEM_PRESERVING_QUALITY_RECOVERY_V82"})
    if not quality_rows:
        quality_rows = [{
            "stage": "P6_QUALITY_PRESERVING_S2_REPAIR_V82",
            "status": "no_macro_pass_focus_candidate",
            "fake_data_used": 0,
            "proxy_row_used": 0,
        }]
    if not system_rows:
        system_rows = [{
            "stage": "P7_SYSTEM_PRESERVING_QUALITY_RECOVERY_V82",
            "status": "no_s2_pass_focus_candidate",
            "fake_data_used": 0,
            "proxy_row_used": 0,
        }]
    write_csv(out_dir / "p6_quality_preserving_s2_repair.csv", quality_rows)
    write_csv(out_dir / "p7_system_preserving_quality_recovery.csv", system_rows)
    write_csv(out_dir / "p8_official_coselection.csv", [
        {
            **row,
            "stage": "P8_OFFICIAL_COSELECTION_V82",
            "cpu_offload_used": 0,
            "fake_data_used": 0,
            "proxy_row_used": 0,
        }
        for row in score_rows
    ])


def _write_not_run_system_tables(out_dir: Path, minimum_success: int) -> None:
    reason = (
        "minimum_success_reached_but_formal_system_steps_not_executed_in_this_slice"
        if minimum_success
        else "no_single_CE_only_candidate_satisfies_macro_and_FullGridS2"
    )
    for name, stage in [
        ("p9_time_accounting.csv", "P9_TIME_ACCOUNTING_V82"),
        ("p10_phase_mapped_profiler.csv", "P10_PHASE_MAPPED_PROFILER_V82"),
        ("p11_s1_memory_package.csv", "P11_S1_MEMORY_PACKAGE_V82"),
    ]:
        write_csv(out_dir / name, [{
            "stage": stage,
            "status": "not_run",
            "reason": reason,
            "fake_data_used": 0,
            "proxy_row_used": 0,
        }])


def _route_decision(out_dir: Path) -> Dict[str, Any]:
    score_rows = [
        row for row in _read_csv_rows(out_dir / "p6_official_coselection.csv")
        if str(row.get("candidate_id")) in FOCUS_IDS and _is_one(row.get("official_eligible"))
    ]
    if not score_rows:
        route = {
            "route": "R8-CodeContractOrMeasurementMissing",
            "success_v82_minimum": 0,
            "success_v82_formal": 0,
            "no_fake": True,
            "no_proxy": True,
        }
        save_json(out_dir / "route_decision.json", route)
        save_json(out_dir / "aggregate_decision.json", route)
        return route

    for row in score_rows:
        row["_passes"] = _candidate_passes(row)
    merged = [row for row in score_rows if row["_passes"]["contract"] and row["_passes"]["grad"] and row["_passes"]["macro"] and row["_passes"]["s2"]]
    quality = [row for row in score_rows if row["_passes"]["contract"] and row["_passes"]["grad"] and row["_passes"]["macro"]]
    system = [row for row in score_rows if row["_passes"]["contract"] and row["_passes"]["grad"] and row["_passes"]["s2"]]
    quality_best = max(quality or score_rows, key=lambda r: _float(r.get("macro_gap"), -99.0))
    system_best = max(system or score_rows, key=lambda r: (
        int(r["_passes"]["s2"]),
        -_float(r.get("memory_ratio_max"), 99.0),
        -_float(r.get("step_ratio_max"), 99.0),
        _float(r.get("macro_gap"), -99.0),
    ))
    if merged:
        best = max(merged, key=lambda r: (_float(r.get("macro_gap"), -99.0), -_float(r.get("memory_ratio_max"), 99.0)))
        route_name = "R1-CEOnlyArchitectureSystemConverged"
    elif quality and system:
        best = quality_best
        route_name = "R4-QualitySystemSplitNotMerged"
    elif quality:
        best = quality_best
        route_name = "R2-QualityPassSystemFail"
    elif system:
        best = system_best
        route_name = "R3-SystemPassQualityFail"
    else:
        best = max(score_rows, key=lambda r: _float(r.get("macro_gap"), -99.0))
        route_name = "R5-NoConvergenceNearPassOrFail"
    success_min = int(bool(merged))
    success_formal = 0
    route = {
        "route": route_name,
        "best_official_candidate_id": best.get("candidate_id", METRIC_UNAVAILABLE),
        "quality_best_candidate_id": quality_best.get("candidate_id", METRIC_UNAVAILABLE),
        "system_best_candidate_id": system_best.get("candidate_id", METRIC_UNAVAILABLE),
        "code_native_pass": 1,
        "no_teacher_no_loss_pass": int(best["_passes"]["contract"]),
        "strict_pass": int(_is_one(best.get("StrictPass"))),
        "grad_pass": int(best["_passes"]["grad"]),
        "teacher_free_ce_macro_pass": int(best["_passes"]["macro"]),
        "fullgrid_s2_pass": int(best["_passes"]["s2"]),
        "fullgrid_s1_pass": int(best["_passes"]["s1"]),
        "time_auc_pass": int(best["_passes"]["time_auc"]),
        "profiler_pass": 0,
        "success_v82_minimum": success_min,
        "success_v82_formal": success_formal,
        "best_macro_gap": best.get("macro_gap", METRIC_UNAVAILABLE),
        "best_ci95_low": best.get("ci95_low", METRIC_UNAVAILABLE),
        "best_holm_p": best.get("Holm_p", METRIC_UNAVAILABLE),
        "best_test_gap": best.get("test_gap", METRIC_UNAVAILABLE),
        "best_memory_ratio_max": best.get("memory_ratio_max", METRIC_UNAVAILABLE),
        "best_step_ratio_max": best.get("step_ratio_max", METRIC_UNAVAILABLE),
        "quality_best_macro_gap": quality_best.get("macro_gap", METRIC_UNAVAILABLE),
        "quality_best_memory_ratio_max": quality_best.get("memory_ratio_max", METRIC_UNAVAILABLE),
        "quality_best_step_ratio_max": quality_best.get("step_ratio_max", METRIC_UNAVAILABLE),
        "system_best_macro_gap": system_best.get("macro_gap", METRIC_UNAVAILABLE),
        "system_best_memory_ratio_max": system_best.get("memory_ratio_max", METRIC_UNAVAILABLE),
        "system_best_step_ratio_max": system_best.get("step_ratio_max", METRIC_UNAVAILABLE),
        "primary_blocker": "none" if success_min else ("quality_system_split_not_merged" if quality and system else "quality_or_system_gate_not_closed"),
        "next_required_implementation": "formal_S1_TimeAUC_profiler" if success_min else "KC6_preserving_workspace_or_KF4_preserving_late_trajectory_repair",
        "cpu_offload_used": 0,
        "no_fake": True,
        "no_proxy": True,
    }
    save_json(out_dir / "route_decision.json", route)
    save_json(out_dir / "aggregate_decision.json", route)
    return route


def _write_failure_table(out_dir: Path, route: Dict[str, Any]) -> None:
    failures = []
    if not int(route.get("success_v82_minimum", 0)):
        failures.append(("F4_quality_system_split", int(route.get("route") == "R4-QualitySystemSplitNotMerged")))
        failures.append(("F5_teacher_free_ce_macro_or_s2_fail", 1))
    if int(route.get("profiler_pass", 0)) == 0:
        failures.append(("F13_profiler_incomplete", 1))
    if not failures:
        failures = [("none", 0)]
    write_csv(out_dir / "failure_table.csv", [
        {"failure_type": name, "count": count, "fake_data_used": 0, "proxy_row_used": 0}
        for name, count in failures
    ])


def _write_manifest_and_audit(out_dir: Path, args: argparse.Namespace, route: Dict[str, Any]) -> None:
    artifact_names = [
        "candidate_registry_v82.csv",
        "contract_no_teacher_no_loss.csv",
        "p0_contract_audit.csv",
        "p1_reproduction.csv",
        "p2_trajectory_divergence.csv",
        "p3_factor_isolation.csv",
        "p4_gpu_liveset_attribution.csv",
        "p5_allocator_sensitivity.csv",
        "p6_quality_preserving_s2_repair.csv",
        "p7_system_preserving_quality_recovery.csv",
        "p8_official_coselection.csv",
        "p9_time_accounting.csv",
        "p10_phase_mapped_profiler.csv",
        "p11_s1_memory_package.csv",
        "route_decision.json",
        "aggregate_decision.json",
        "failure_table.csv",
        "p9_task_trace.csv",
        "p1_significance_audit.csv",
        "p2_full_gradient_correctness.csv",
        "p10_efficiency_profiler.csv",
        "p11_gpu_live_allocator_v81.csv",
    ]
    artifact_paths = [out_dir / name for name in artifact_names]
    audit = v72._audit_fake_proxy(artifact_paths)
    write_csv(out_dir / "v82_provenance_audit.csv", [{
        **audit,
        "plan_path": PLAN_PATH,
        "script_path": SCRIPT_PATH,
        "cpu_offload_allowed": 0,
        "cpu_offload_used": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
    }])
    artifact_paths.append(out_dir / "v82_provenance_audit.csv")
    save_json(out_dir / "v82_manifest.json", {
        "plan_path": PLAN_PATH,
        "script_path": SCRIPT_PATH,
        "runner_reuse": "run_gafu_v81_real.py measured path + v8.2 convergence postprocess",
        "out_dir": str(out_dir),
        "source_commit": _git_commit(),
        "git_status": _git_status(),
        "datasets": parse_str_list(args.datasets),
        "seeds": parse_int_list(args.seeds),
        "candidates": parse_str_list(args.candidates),
        "bench_batch_sizes": parse_int_list(args.bench_batch_sizes),
        "grad_batch_sizes": parse_int_list(args.grad_batch_sizes),
        "artifact_hashes": {p.name: _hash_file(p) for p in artifact_paths if p.exists()},
        "route_decision": route,
        "provenance_audit": audit,
        "cpu_offload_allowed": 0,
    })


def _write_v82_postprocess(out_dir: Path, args: argparse.Namespace) -> Dict[str, Any]:
    global _LAST_ARGS
    _LAST_ARGS = args
    _write_candidate_registry(out_dir)
    _write_contract(out_dir)
    _write_reproduction(out_dir)
    _write_trajectory_divergence(out_dir)
    _write_factor_isolation(out_dir)
    _write_liveset_and_allocator(out_dir)
    _write_repair_tables(out_dir)
    route = _route_decision(out_dir)
    _write_not_run_system_tables(out_dir, int(route.get("success_v82_minimum", 0)))
    _write_failure_table(out_dir, route)
    _write_manifest_and_audit(out_dir, args, route)
    return route


def run(args: argparse.Namespace) -> None:
    v80.PLAN_PATH = PLAN_PATH
    v80.SCRIPT_PATH = SCRIPT_PATH
    v81.PLAN_PATH = PLAN_PATH
    v81.SCRIPT_PATH = SCRIPT_PATH
    v81.run(args)
    out_dir = Path(args.out_dir)
    route = _write_v82_postprocess(out_dir, args)
    for src_name, dst_name in {
        "v81_manifest.json": "v82_reused_v81_manifest.json",
        "v81_route_decision.json": "v82_reused_v81_route_decision.json",
        "v81_provenance_audit.csv": "v82_reused_v81_provenance_audit.csv",
    }.items():
        src = out_dir / src_name
        if src.exists():
            shutil.copyfile(src, out_dir / dst_name)
    save_json(out_dir / "v82_run_complete.json", {
        "route": route,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    })


def parse_args() -> argparse.Namespace:
    return v81.parse_args()


if __name__ == "__main__":
    run(parse_args())
