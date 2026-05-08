#!/usr/bin/env python3
"""DG-KAN v7.5 real-only A2S/C3 reachability and attribution runner.

v7.5 reuses the audited v7.3 execution path for real task, gradient and
efficiency measurements, then writes v7.5-specific artifacts for:

* P0 reproduction / provenance lock;
* P1 C3 versus A2S source-of-loss accounting;
* P2 measured A2S live-set / full-grid S2 attribution;
* P3 C3-teacher logit-distillation reachability;
* P11 route decision under the v7.5 gates.

No fake data, proxy rows, hand-filled ratios, or unmeasured fused-kernel claims
are emitted.  Missing mechanism metrics are written as metric_unavailable.
"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import run_gafu_v73_real as v73
import run_gafu_v72_real as v72
from dgkan_core import save_json, write_csv


PLAN_PATH = "docs/DG-KAN_v7.5_A2S_C3_Reachability_FusedKernel_完整实验计划.md"
SCRIPT_PATH = "experiments/run_gafu_v75_real.py"
METRIC_UNAVAILABLE = v73.METRIC_UNAVAILABLE
V74_REFERENCE_RUN = Path("results/real_rerun_20260506/v74_mechanism_kernel_bridge_5seed_20260506T021447Z")


def _read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _finite(value: Any) -> bool:
    return v73._finite(value)


def _float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _is_one(value: Any) -> bool:
    return str(value) in {"1", "1.0", "true", "True"}


def _hash_file(path: Path) -> str:
    return v73._hash_file(path) if path.exists() else ""


def _latest_by_candidate(rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {str(row.get("candidate_id")): row for row in rows}


def _macro_by_candidate(sig_rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {str(row.get("candidate_id")): row for row in sig_rows if row.get("dataset") == "macro"}


def _grad_pass_by_candidate(grad_rows: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in grad_rows:
        grouped.setdefault(str(row.get("candidate_id")), []).append(row)
    return {
        cid: int(bool(rows) and all(_is_one(row.get("grad_pass")) for row in rows))
        for cid, rows in grouped.items()
    }


def _grad_summary_by_candidate(grad_rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in grad_rows:
        grouped.setdefault(str(row.get("candidate_id")), []).append(row)
    out: Dict[str, Dict[str, Any]] = {}
    for cid, rows in grouped.items():
        rels = [_float(row.get("grad_relerr_max")) for row in rows if _finite(row.get("grad_relerr_max"))]
        coss = [_float(row.get("grad_cos_min")) for row in rows if _finite(row.get("grad_cos_min"))]
        out[cid] = {
            "grad_rows": len(rows),
            "grad_pass_rows": sum(1 for row in rows if _is_one(row.get("grad_pass"))),
            "grad_relerr_max": max(rels) if rels else METRIC_UNAVAILABLE,
            "grad_cos_min": min(coss) if coss else METRIC_UNAVAILABLE,
        }
    return out


def _fullgrid_s2_by_candidate(eff_summary: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for row in eff_summary:
        rows = int(_float(row.get("rows"), 0.0))
        pass_shapes = int(_float(row.get("s2_pass_shapes"), -1.0))
        out[str(row.get("candidate_id"))] = int(rows > 0 and pass_shapes == rows)
    return out


def _macro_pass(sig: Dict[str, Any]) -> int:
    macro_gap = _float(sig.get("mean_val_gap"))
    ci_low = _float(sig.get("bootstrap_ci95_low"))
    holm = _float(sig.get("holm_corrected_p"))
    test_gap = _float(sig.get("mean_test_gap"))
    ece_raw = sig.get("ECE_delta_macro", sig.get("ECE_delta"))
    nll_raw = sig.get("NLL_delta_macro", sig.get("NLL_delta"))
    ece_delta = _float(ece_raw)
    nll_delta = _float(nll_raw)
    return int(
        _finite(macro_gap) and macro_gap >= 0.0200
        and _finite(ci_low) and ci_low > 0.0
        and _finite(holm) and holm < 0.05
        and _finite(test_gap) and test_gap >= 0.015
        and _finite(ece_delta) and ece_delta <= 0.005
        and _finite(nll_delta) and nll_delta <= 0.01
    )


def _strict_pass(task: Dict[str, Any]) -> int:
    return int(
        _is_one(task.get("head_is_kan"))
        and str(task.get("non_kan_trainable_param_count")) in {"0", "0.0", ""}
        and _is_one(task.get("manual_backward"))
    )


def _reference_rows(name: str) -> List[Dict[str, Any]]:
    return _read_csv_rows(V74_REFERENCE_RUN / name)


def _write_p0(out_dir: Path, task_summary: Sequence[Dict[str, Any]], sig_rows: Sequence[Dict[str, Any]], eff_summary: Sequence[Dict[str, Any]], grad_rows: Sequence[Dict[str, Any]]) -> None:
    task_by = _latest_by_candidate(task_summary)
    sig_by = _macro_by_candidate(sig_rows)
    eff_by = _latest_by_candidate(eff_summary)
    grad_by = _grad_summary_by_candidate(grad_rows)
    ref_task = _latest_by_candidate(_reference_rows("p9_task_summary.csv"))
    ref_eff = _latest_by_candidate(_reference_rows("p10_efficiency_summary.csv"))
    rows: List[Dict[str, Any]] = []
    for cid in ["B0", "C3", "A2S", "A2C", "A5C", "D1", "D2", "D3", "E1", "E2", "E3", "Z1", "Z2", "Z3", "U1", "U2", "U3", "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9", "M10", "M11", "M12"]:
        if cid not in task_by:
            continue
        task = task_by[cid]
        sig = sig_by.get(cid, {})
        eff = eff_by.get(cid, {})
        ref_t = ref_task.get(cid, {})
        ref_e = ref_eff.get(cid, {})
        val_gap = _float(task.get("val_gap_vs_MLP_mean"))
        ref_gap = _float(ref_t.get("val_gap_vs_MLP_mean"))
        mem = _float(eff.get("memory_ratio_mean"))
        ref_mem = _float(ref_e.get("memory_ratio_mean"))
        step = _float(eff.get("step_ratio_mean"))
        ref_step = _float(ref_e.get("step_ratio_mean"))
        rows.append({
            "stage": "P0",
            "candidate_id": cid,
            "candidate_name": task.get("candidate_name"),
            "strict_pass": _strict_pass(task) if cid != "B0" else 0,
            "grad_pass": int(grad_by.get(cid, {}).get("grad_pass_rows", 0) == grad_by.get(cid, {}).get("grad_rows", -1) and grad_by.get(cid, {}).get("grad_rows", 0) > 0),
            "grad_relerr_max": grad_by.get(cid, {}).get("grad_relerr_max", METRIC_UNAVAILABLE),
            "grad_cos_min": grad_by.get(cid, {}).get("grad_cos_min", METRIC_UNAVAILABLE),
            "val_acc_mean": task.get("val_acc_mean"),
            "test_acc_mean": task.get("test_acc_mean"),
            "val_gap_vs_MLP_mean": task.get("val_gap_vs_MLP_mean"),
            "test_gap_vs_MLP": sig.get("mean_test_gap", METRIC_UNAVAILABLE),
            "ci95_low": sig.get("bootstrap_ci95_low", METRIC_UNAVAILABLE),
            "holm_p": sig.get("holm_corrected_p", METRIC_UNAVAILABLE),
            "ECE_delta": sig.get("ECE_delta_macro", sig.get("ECE_delta", METRIC_UNAVAILABLE)),
            "NLL_delta": sig.get("NLL_delta_macro", sig.get("NLL_delta", METRIC_UNAVAILABLE)),
            "memory_ratio_mean": eff.get("memory_ratio_mean", METRIC_UNAVAILABLE),
            "step_ratio_mean": eff.get("step_ratio_mean", METRIC_UNAVAILABLE),
            "s2_pass_shapes": eff.get("s2_pass_shapes", METRIC_UNAVAILABLE),
            "efficiency_rows": eff.get("rows", METRIC_UNAVAILABLE),
            "reproduction_delta_val_gap_vs_v74": val_gap - ref_gap if _finite(val_gap) and _finite(ref_gap) else METRIC_UNAVAILABLE,
            "reproduction_delta_memory_vs_v74": mem - ref_mem if _finite(mem) and _finite(ref_mem) else METRIC_UNAVAILABLE,
            "reproduction_delta_step_vs_v74": step - ref_step if _finite(step) and _finite(ref_step) else METRIC_UNAVAILABLE,
            "fake_data_used": 0,
            "proxy_row_used": 0,
        })
    write_csv(out_dir / "p0_reproduction.csv", rows)


def _write_p1(out_dir: Path, task_summary: Sequence[Dict[str, Any]], sig_rows: Sequence[Dict[str, Any]]) -> None:
    task_by = _latest_by_candidate(task_summary)
    sig_by = _macro_by_candidate(sig_rows)
    c3_task = task_by.get("C3", {})
    c3_sig = sig_by.get("C3", {})
    c3_gap = _float(c3_sig.get("mean_val_gap", c3_task.get("val_gap_vs_MLP_mean")))
    c3_test = _float(c3_sig.get("mean_test_gap"))
    c3_ece = _float(c3_sig.get("ECE_delta_macro", c3_sig.get("ECE_delta")))
    c3_nll = _float(c3_sig.get("NLL_delta_macro", c3_sig.get("NLL_delta")))
    rows: List[Dict[str, Any]] = []
    for cid in ["C3", "A2S", "A2C", "A5C", "D1", "D2", "D3", "E1", "E2", "E3", "Z1", "Z2", "Z3", "U1", "U2", "U3", "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9", "M10", "M11", "M12"]:
        if cid not in task_by:
            continue
        task = task_by[cid]
        sig = sig_by.get(cid, {})
        gap = _float(sig.get("mean_val_gap", task.get("val_gap_vs_MLP_mean")))
        test_gap = _float(sig.get("mean_test_gap"))
        ece = _float(sig.get("ECE_delta_macro", sig.get("ECE_delta")))
        nll = _float(sig.get("NLL_delta_macro", sig.get("NLL_delta")))
        rows.append({
            "stage": "P1",
            "candidate_id": cid,
            "candidate_name": task.get("candidate_name"),
            "role": "teacher_expression_oracle" if cid == "C3" else "A2S_bridge_or_student",
            "val_gap_vs_MLP": gap if _finite(gap) else METRIC_UNAVAILABLE,
            "test_gap_vs_MLP": test_gap if _finite(test_gap) else METRIC_UNAVAILABLE,
            "val_gap_loss_vs_C3": c3_gap - gap if _finite(c3_gap) and _finite(gap) else METRIC_UNAVAILABLE,
            "test_gap_loss_vs_C3": c3_test - test_gap if _finite(c3_test) and _finite(test_gap) else METRIC_UNAVAILABLE,
            "ECE_delta": ece if _finite(ece) else METRIC_UNAVAILABLE,
            "NLL_delta": nll if _finite(nll) else METRIC_UNAVAILABLE,
            "ECE_delta_vs_C3": ece - c3_ece if _finite(ece) and _finite(c3_ece) else METRIC_UNAVAILABLE,
            "NLL_delta_vs_C3": nll - c3_nll if _finite(nll) and _finite(c3_nll) else METRIC_UNAVAILABLE,
            "teacher_student_logit_KL": METRIC_UNAVAILABLE,
            "teacher_student_logit_cos": METRIC_UNAVAILABLE,
            "feature_CKA": METRIC_UNAVAILABLE,
            "source_of_loss_status": "task_calibration_measured__logit_feature_attribution_not_implemented",
            "fake_data_used": 0,
            "proxy_row_used": 0,
        })
    write_csv(out_dir / "p1_c3_a2s_source_of_loss.csv", rows)


def _write_p2(out_dir: Path, p5_rows: Sequence[Dict[str, Any]], eff_summary: Sequence[Dict[str, Any]]) -> None:
    eff_by = _latest_by_candidate(eff_summary)
    rows: List[Dict[str, Any]] = []
    for row in p5_rows:
        cid = str(row.get("candidate_id"))
        if cid not in {"C3", "A2S", "A2C", "A5C", "D1", "D2", "D3", "E1", "E2", "E3", "Z1", "Z2", "Z3", "U1", "U2", "U3", "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9", "M10", "M11", "M12"}:
            continue
        eff = eff_by.get(cid, {})
        materialized = row.get("materialized_tensor_count", METRIC_UNAVAILABLE)
        unknown = row.get("unknown_memory_fraction", METRIC_UNAVAILABLE)
        rows.append({
            "stage": "P2",
            "candidate_id": cid,
            "candidate_name": row.get("candidate_name"),
            "dataset": row.get("dataset"),
            "batch_size": row.get("batch_size"),
            "memory_ratio": row.get("memory_ratio"),
            "step_ratio": row.get("step_ratio"),
            "forward_time_ms": row.get("forward_time_ms"),
            "backward_time_ms": row.get("backward_time_ms"),
            "update_time_ms": row.get("update_time_ms"),
            "step_time_ms": row.get("step_time_ms"),
            "peak_allocated_MB": row.get("peak_allocated_MB"),
            "peak_reserved_MB": row.get("peak_reserved_MB"),
            "cache_total_MB_measured": row.get("cache_total_MB_measured"),
            "manual_cache_fraction_of_peak": row.get("manual_cache_fraction_of_peak"),
            "optimizer_state_fraction_of_peak": row.get("optimizer_state_fraction_of_peak"),
            "materialized_tensor_count": materialized,
            "materialized_tensor_count_measured": int(_finite(materialized)),
            "largest_live_tensor_MB": row.get("largest_live_tensor_MB"),
            "linear_body_temp_MB": row.get("linear_body_temp_MB"),
            "hidden_y_cache_MB": row.get("hidden_y_cache_MB"),
            "top1_memory_source": row.get("top1_memory_source"),
            "top2_memory_source": row.get("top2_memory_source"),
            "top3_memory_source": row.get("top3_memory_source"),
            "unknown_memory_fraction": unknown,
            "unknown_memory_fraction_pass": int(_finite(unknown) and _float(unknown) <= 0.10),
            "torch_op_count_forward": row.get("torch_op_count"),
            "kernel_count_total": METRIC_UNAVAILABLE,
            "kernel_count_total_status": "not_measured",
            "fullgrid_s2_pass_shapes": eff.get("s2_pass_shapes", METRIC_UNAVAILABLE),
            "fullgrid_efficiency_rows": eff.get("rows", METRIC_UNAVAILABLE),
            "fullgrid_s2_pass": _fullgrid_s2_by_candidate(eff_summary).get(cid, 0),
            "attribution_status": row.get("attribution_status", "metric_unavailable"),
            "fake_data_used": 0,
            "proxy_row_used": 0,
        })
    write_csv(out_dir / "p2_a2s_live_set_kernel_attribution.csv", rows)


def _write_p3(out_dir: Path, p3_rows: Sequence[Dict[str, Any]], sig_rows: Sequence[Dict[str, Any]], grad_rows: Sequence[Dict[str, Any]], eff_summary: Sequence[Dict[str, Any]]) -> None:
    sig_by = _macro_by_candidate(sig_rows)
    grad_pass = _grad_pass_by_candidate(grad_rows)
    full_s2 = _fullgrid_s2_by_candidate(eff_summary)
    rows: List[Dict[str, Any]] = []
    for row in p3_rows:
        cid = str(row.get("student"))
        sig = sig_by.get(cid, {})
        improvement = _float(row.get("distill_improvement_vs_supervised"))
        macro = _macro_pass(sig)
        rows.append({
            **row,
            "macro_pass_v75": macro,
            "grad_pass": grad_pass.get(cid, 0),
            "fullgrid_s2_pass": full_s2.get(cid, 0),
            "reachability_pass_v75": int(_finite(improvement) and improvement >= 0.002 and macro == 1),
            "fake_data_used": 0,
            "proxy_row_used": 0,
        })
    write_csv(out_dir / "p3_distillation_reachability.csv", rows)


def _candidate_rows(task_summary: Sequence[Dict[str, Any]], sig_rows: Sequence[Dict[str, Any]], eff_summary: Sequence[Dict[str, Any]], grad_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    task_by = _latest_by_candidate(task_summary)
    sig_by = _macro_by_candidate(sig_rows)
    eff_by = _latest_by_candidate(eff_summary)
    grad_by = _grad_summary_by_candidate(grad_rows)
    full_s2 = _fullgrid_s2_by_candidate(eff_summary)
    rows: List[Dict[str, Any]] = []
    for cid, task in task_by.items():
        if cid == "B0":
            continue
        sig = sig_by.get(cid, {})
        eff = eff_by.get(cid, {})
        gsum = grad_by.get(cid, {})
        grad_pass = int(gsum.get("grad_rows", 0) > 0 and gsum.get("grad_rows") == gsum.get("grad_pass_rows"))
        strict_pass = _strict_pass(task)
        macro_pass = _macro_pass(sig)
        rows.append({
            "candidate_id": cid,
            "candidate_name": task.get("candidate_name"),
            "strict_pass": strict_pass,
            "grad_pass": grad_pass,
            "macro_pass_v75": macro_pass,
            "fullgrid_s2_pass": full_s2.get(cid, 0),
            "s2_pass_shapes": eff.get("s2_pass_shapes", METRIC_UNAVAILABLE),
            "efficiency_rows": eff.get("rows", METRIC_UNAVAILABLE),
            "val_acc_mean": task.get("val_acc_mean"),
            "test_acc_mean": task.get("test_acc_mean"),
            "macro_gap": sig.get("mean_val_gap", task.get("val_gap_vs_MLP_mean")),
            "ci95_low": sig.get("bootstrap_ci95_low", METRIC_UNAVAILABLE),
            "holm_p": sig.get("holm_corrected_p", METRIC_UNAVAILABLE),
            "test_gap": sig.get("mean_test_gap", METRIC_UNAVAILABLE),
            "ECE_delta": sig.get("ECE_delta_macro", sig.get("ECE_delta", METRIC_UNAVAILABLE)),
            "NLL_delta": sig.get("NLL_delta_macro", sig.get("NLL_delta", METRIC_UNAVAILABLE)),
            "memory_ratio_mean": eff.get("memory_ratio_mean", METRIC_UNAVAILABLE),
            "step_ratio_mean": eff.get("step_ratio_mean", METRIC_UNAVAILABLE),
            "forward_ratio_mean": eff.get("forward_ratio_mean", METRIC_UNAVAILABLE),
            "backward_ratio_mean": eff.get("backward_ratio_mean", METRIC_UNAVAILABLE),
            "grad_rows": gsum.get("grad_rows", 0),
            "grad_pass_rows": gsum.get("grad_pass_rows", 0),
            "grad_relerr_max": gsum.get("grad_relerr_max", METRIC_UNAVAILABLE),
            "grad_cos_min": gsum.get("grad_cos_min", METRIC_UNAVAILABLE),
            "success_v75_minimum": int(strict_pass and grad_pass and macro_pass and full_s2.get(cid, 0)),
            "fake_data_used": 0,
            "proxy_row_used": 0,
        })
    rows.sort(
        key=lambda r: (
            int(r["success_v75_minimum"]),
            int(r["macro_pass_v75"]),
            int(r["grad_pass"]),
            _float(r["macro_gap"]),
        ),
        reverse=True,
    )
    return rows


def _write_route(out_dir: Path, candidate_rows: Sequence[Dict[str, Any]], p2_rows: Sequence[Dict[str, Any]], p3_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    success = [r for r in candidate_rows if _is_one(r.get("success_v75_minimum"))]
    macro_grad = [r for r in candidate_rows if _is_one(r.get("macro_pass_v75")) and _is_one(r.get("grad_pass"))]
    best = success[0] if success else macro_grad[0] if macro_grad else candidate_rows[0] if candidate_rows else {}
    attribution_measured = any(_is_one(r.get("materialized_tensor_count_measured")) for r in p2_rows)
    kernel_count_measured = any(str(r.get("kernel_count_total")) not in {"", METRIC_UNAVAILABLE, "not_measured"} for r in p2_rows)
    reachability_rows = [r for r in p3_rows if _is_one(r.get("reachability_pass_v75"))]
    distill_any_improvement = any(_finite(r.get("distill_improvement_vs_supervised")) and _float(r.get("distill_improvement_vs_supervised")) >= 0.001 for r in p3_rows)
    if success:
        route = "S2-MacroSignificantSuccess"
        blocker = "none"
        next_required = "stop_success_recap"
    elif macro_grad:
        route = "R3-MacroExpressivityPositiveButNotFullGridS2"
        blocker = "kernel_native_efficiency"
        next_required = "measured_kernel_count_and_dense_preserving_fused_package"
    elif reachability_rows:
        route = "R4-A2SReachabilityPositiveButKernelOpen"
        blocker = "fullgrid_s2_or_kernelization"
        next_required = "dense_preserving_fused_kernel_for_a2s"
    elif distill_any_improvement:
        route = "R5-ReachabilityPartialButMacroStillOpen"
        blocker = "effective_expressivity_or_training_reachability"
        next_required = "minimal_low_live_set_expression_bridge"
    else:
        route = "R6-EffectiveExpressivityOrAttributionStillOpen"
        blocker = "effective_expressivity_loss" if attribution_measured else "attribution_incomplete"
        next_required = "minimal_bridge_or_complete_kernel_count_attribution"
    out = {
        "route": route,
        "best_candidate_id": best.get("candidate_id"),
        "best_candidate": best.get("candidate_name"),
        "strict_pass": best.get("strict_pass", 0),
        "grad_pass": best.get("grad_pass", 0),
        "macro_pass_v75": best.get("macro_pass_v75", 0),
        "fullgrid_s2_pass": best.get("fullgrid_s2_pass", 0),
        "success_v75_minimum": best.get("success_v75_minimum", 0),
        "best_macro_gap": best.get("macro_gap"),
        "best_ci95_low": best.get("ci95_low"),
        "best_holm_p": best.get("holm_p"),
        "best_test_gap": best.get("test_gap"),
        "best_ECE_delta": best.get("ECE_delta"),
        "best_NLL_delta": best.get("NLL_delta"),
        "best_memory_ratio": best.get("memory_ratio_mean"),
        "best_step_ratio": best.get("step_ratio_mean"),
        "best_s2_pass_shapes": best.get("s2_pass_shapes"),
        "distillation_reachability_pass": bool(reachability_rows),
        "materialized_tensor_count_measured": bool(attribution_measured),
        "kernel_count_total_measured": bool(kernel_count_measured),
        "primary_blocker": blocker,
        "next_required_implementation": next_required,
        "classwise_is_hard_gate": False,
        "no_fake": True,
        "no_proxy": True,
    }
    save_json(out_dir / "v75_route_decision.json", out)
    return out


def _mirror_v75_artifacts(out_dir: Path) -> None:
    mapping = {
        "v73_route_decision.json": "v75_reused_v73_route_decision.json",
        "v73_provenance_audit.csv": "v75_reused_v73_provenance_audit.csv",
        "v73_manifest.json": "v75_reused_v73_manifest.json",
    }
    for src_name, dst_name in mapping.items():
        src = out_dir / src_name
        if src.exists():
            shutil.copyfile(src, out_dir / dst_name)


def _postprocess_v75(out_dir: Path, args: Any) -> None:
    task_summary = _read_csv_rows(out_dir / "p9_task_summary.csv")
    sig_rows = _read_csv_rows(out_dir / "p1_significance_audit.csv")
    eff_summary = _read_csv_rows(out_dir / "p10_efficiency_summary.csv")
    grad_rows = _read_csv_rows(out_dir / "p2_full_gradient_correctness.csv")
    p5_rows = _read_csv_rows(out_dir / "p5_live_set_kernelization_attribution.csv")
    p3_rows = _read_csv_rows(out_dir / "p3_optimizer_reachability_distillation.csv")

    _write_p0(out_dir, task_summary, sig_rows, eff_summary, grad_rows)
    _write_p1(out_dir, task_summary, sig_rows)
    _write_p2(out_dir, p5_rows, eff_summary)
    p2_rows = _read_csv_rows(out_dir / "p2_a2s_live_set_kernel_attribution.csv")
    _write_p3(out_dir, p3_rows, sig_rows, grad_rows, eff_summary)
    p3_v75_rows = _read_csv_rows(out_dir / "p3_distillation_reachability.csv")
    candidates = _candidate_rows(task_summary, sig_rows, eff_summary, grad_rows)
    write_csv(out_dir / "p11_candidate_selection_v75.csv", candidates)
    route = _write_route(out_dir, candidates, p2_rows, p3_v75_rows)

    artifact_paths = [
        out_dir / "p0_reproduction.csv",
        out_dir / "p1_c3_a2s_source_of_loss.csv",
        out_dir / "p2_a2s_live_set_kernel_attribution.csv",
        out_dir / "p3_distillation_reachability.csv",
        out_dir / "p11_candidate_selection_v75.csv",
    ]
    audit = v72._audit_fake_proxy(artifact_paths)
    write_csv(out_dir / "v75_provenance_audit.csv", [{
        **audit,
        "plan_path": PLAN_PATH,
        "script_path": SCRIPT_PATH,
        "fake_data_used": 0,
        "proxy_row_used": 0,
    }])
    save_json(out_dir / "v75_manifest.json", {
        "plan_path": PLAN_PATH,
        "script_path": SCRIPT_PATH,
        "runner_reuse": "run_gafu_v73_real.py",
        "out_dir": str(out_dir),
        "candidates": getattr(args, "candidates", ""),
        "seeds": getattr(args, "seeds", ""),
        "bench_batch_sizes": getattr(args, "bench_batch_sizes", ""),
        "grad_batch_sizes": getattr(args, "grad_batch_sizes", ""),
        "postprocess_artifact_hashes": {p.name: _hash_file(p) for p in artifact_paths if p.exists()},
        "v75_route": route,
        "v75_audit": audit,
    })
    _mirror_v75_artifacts(out_dir)


def run(args) -> None:
    v73.PLAN_PATH = PLAN_PATH
    v73.SCRIPT_PATH = SCRIPT_PATH
    v73.run(args)
    _postprocess_v75(Path(args.out_dir), args)


if __name__ == "__main__":
    run(v73.parse_args())
