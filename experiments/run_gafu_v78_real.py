#!/usr/bin/env python3
"""DG-KAN v7.8 teacher-free official PureKAN-NG runner.

v7.8 reuses the v7.6 real training/evaluation implementation, but changes the
route decision: C3-teacher-distilled candidates are diagnostic only.  Official
success is evaluated on no-external-teacher PureKAN candidates, currently M13.

All added rows are post-processed from measured artifacts.  Missing or
unimplemented diagnostics are left as metric_unavailable; no fake/proxy rows are
introduced.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import run_gafu_v76_real as v76
import run_gafu_v72_real as v72
from dgkan_core import parse_int_list, parse_str_list, save_json, write_csv
from run_gafu_v66_real import _git_commit, _git_status


PLAN_PATH = "docs/DG-KAN_v7.8_TeacherFree_Official_PureKANNG_完整实验计划.md"
SCRIPT_PATH = "experiments/run_gafu_v78_real.py"
METRIC_UNAVAILABLE = v76.METRIC_UNAVAILABLE


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def _float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _mean(values: Iterable[Any], default: float = float("nan")) -> float:
    vals = [float(v) for v in values if _finite(v)]
    return sum(vals) / len(vals) if vals else default


def _read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _hash_file(path: Path) -> str:
    return v72._hash_file(path) if path.exists() else ""


def _is_one(value: Any) -> bool:
    return str(value) in {"1", "1.0", "true", "True"}


def _summary_by_candidate(rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {str(row.get("candidate_id")): row for row in rows}


def _macro_by_candidate(rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {str(row.get("candidate_id")): row for row in rows if str(row.get("dataset")) == "macro"}


def _grad_summary(rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("candidate_id")), []).append(row)
    out: Dict[str, Dict[str, Any]] = {}
    for cid, items in grouped.items():
        rels = [_float(row.get("grad_relerr_max")) for row in items if _finite(row.get("grad_relerr_max"))]
        coss = [_float(row.get("grad_cos_min")) for row in items if _finite(row.get("grad_cos_min"))]
        out[cid] = {
            "grad_rows": len(items),
            "grad_pass_rows": sum(1 for row in items if _is_one(row.get("grad_pass"))),
            "grad_pass": int(bool(items) and all(_is_one(row.get("grad_pass")) for row in items)),
            "grad_relerr_max": max(rels) if rels else METRIC_UNAVAILABLE,
            "grad_cos_min": min(coss) if coss else METRIC_UNAVAILABLE,
        }
    return out


def _eff_counts(rows: Sequence[Dict[str, Any]], cid: str) -> Dict[str, Any]:
    items = [r for r in rows if str(r.get("candidate_id")) == cid]
    mems = [_float(r.get("memory_ratio")) for r in items if _finite(r.get("memory_ratio"))]
    steps = [_float(r.get("step_ratio")) for r in items if _finite(r.get("step_ratio"))]
    return {
        "eff_rows": len(items),
        "memory_ratio_mean": _mean(mems),
        "memory_ratio_max": max(mems) if mems else METRIC_UNAVAILABLE,
        "step_ratio_mean": _mean(steps),
        "step_ratio_max": max(steps) if steps else METRIC_UNAVAILABLE,
        "s2_shape_count": sum(
            1 for r in items
            if _float(r.get("memory_ratio"), 99.0) <= 1.05 and _float(r.get("step_ratio"), 99.0) <= 1.50
        ),
        "s1_shape_count": sum(
            1 for r in items
            if _float(r.get("memory_ratio"), 99.0) < 1.00 and _float(r.get("step_ratio"), 99.0) <= 1.35
        ),
    }


def _seed_win_rate(task_rows: Sequence[Dict[str, Any]], candidate: str, baseline: str = "B0") -> Tuple[float, int, int]:
    by_seed: Dict[int, Dict[str, List[float]]] = {}
    for row in task_rows:
        cid = str(row.get("candidate_id"))
        if cid not in {candidate, baseline}:
            continue
        try:
            seed = int(row.get("seed"))
        except Exception:
            continue
        by_seed.setdefault(seed, {}).setdefault(cid, []).append(_float(row.get("val_acc")))
    wins = 0
    total = 0
    for data in by_seed.values():
        if candidate in data and baseline in data:
            total += 1
            wins += int(_mean(data[candidate]) > _mean(data[baseline]))
    return (wins / total if total else float("nan"), wins, total)


def _macro_gap_between(task_rows: Sequence[Dict[str, Any]], a: str, b: str) -> float:
    datasets = sorted({str(row.get("dataset")) for row in task_rows})
    gaps: List[float] = []
    for ds in datasets:
        av = _mean(row.get("val_acc") for row in task_rows if str(row.get("candidate_id")) == a and str(row.get("dataset")) == ds)
        bv = _mean(row.get("val_acc") for row in task_rows if str(row.get("candidate_id")) == b and str(row.get("dataset")) == ds)
        if _finite(av) and _finite(bv):
            gaps.append(av - bv)
    return _mean(gaps)


def _nll_between(task_rows: Sequence[Dict[str, Any]], a: str, b: str) -> float:
    av = _mean(row.get("NLL") for row in task_rows if str(row.get("candidate_id")) == a)
    bv = _mean(row.get("NLL") for row in task_rows if str(row.get("candidate_id")) == b)
    return av - bv if _finite(av) and _finite(bv) else float("nan")


def _time_auc_ratio(p7_rows: Sequence[Dict[str, Any]], candidate: str, baseline: str = "B0") -> Dict[str, Any]:
    c_loss_step = _mean(row.get("val_loss_auc_step") for row in p7_rows if str(row.get("candidate_id")) == candidate)
    b_loss_step = _mean(row.get("val_loss_auc_step") for row in p7_rows if str(row.get("candidate_id")) == baseline)
    c_loss_time = _mean(row.get("val_loss_auc_time") for row in p7_rows if str(row.get("candidate_id")) == candidate)
    b_loss_time = _mean(row.get("val_loss_auc_time") for row in p7_rows if str(row.get("candidate_id")) == baseline)
    return {
        "val_loss_auc_step": c_loss_step if _finite(c_loss_step) else METRIC_UNAVAILABLE,
        "val_loss_auc_time": c_loss_time if _finite(c_loss_time) else METRIC_UNAVAILABLE,
        "val_loss_auc_step_ratio_vs_B0": c_loss_step / b_loss_step if _finite(c_loss_step) and _finite(b_loss_step) and b_loss_step != 0 else METRIC_UNAVAILABLE,
        "val_loss_auc_time_ratio_vs_B0": c_loss_time / b_loss_time if _finite(c_loss_time) and _finite(b_loss_time) and b_loss_time != 0 else METRIC_UNAVAILABLE,
        "time_auc_pass": int(_finite(c_loss_time) and _finite(b_loss_time) and c_loss_time <= b_loss_time),
    }


def _external_teacher_used(cid: str) -> int:
    return int(cid in {"B2", "M12"})


def _teacher_role(cid: str) -> str:
    return {
        "B0": "MLP-AdamW-reference",
        "B2": "diagnostic-MLP-C3-distill",
        "M12": "diagnostic-M12-C3-distill",
        "M13": "official-teacher-free-PureKAN",
    }.get(cid, "other")


def _candidate_row(
    cid: str,
    *,
    task_by: Dict[str, Dict[str, Any]],
    macro_by: Dict[str, Dict[str, Any]],
    grad_by: Dict[str, Dict[str, Any]],
    eff_rows: Sequence[Dict[str, Any]],
    p7_rows: Sequence[Dict[str, Any]],
    task_rows: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    task = task_by.get(cid, {})
    macro = macro_by.get(cid, {})
    grad = grad_by.get(cid, {})
    eff = _eff_counts(eff_rows, cid)
    auc = _time_auc_ratio(p7_rows, cid)
    win_rate, wins, total = _seed_win_rate(task_rows, cid, "B0")
    macro_gap = _float(macro.get("mean_val_gap"))
    ci_low = _float(macro.get("bootstrap_ci95_low"))
    holm = _float(macro.get("holm_corrected_p"))
    test_gap = _float(macro.get("mean_test_gap"))
    macro_pass = int(
        _finite(macro_gap) and macro_gap >= 0.0200
        and _finite(ci_low) and ci_low > 0
        and _finite(holm) and holm < 0.05
        and _finite(test_gap) and test_gap >= 0.015
    )
    strict_pass = int(
        str(task.get("head_is_kan")) in {"1", "1.0"}
        and str(task.get("non_kan_trainable_param_count")) in {"0", "0.0", ""}
    ) if cid not in {"B0", "B2"} else 0
    return {
        "candidate_id": cid,
        "candidate_name": task.get("candidate_name", METRIC_UNAVAILABLE),
        "role": _teacher_role(cid),
        "external_teacher_used": _external_teacher_used(cid),
        "self_teacher_used": 0,
        "teacher_candidate": "C3" if _external_teacher_used(cid) else "none",
        "teacher_mode": "online_C3_logits" if _external_teacher_used(cid) else "none",
        "teacher_logits_used": _external_teacher_used(cid),
        "strict_pass": strict_pass,
        "grad_pass": grad.get("grad_pass", 0 if cid not in {"B0", "B2"} else METRIC_UNAVAILABLE),
        "grad_rows": grad.get("grad_rows", 0),
        "grad_relerr_max": grad.get("grad_relerr_max", METRIC_UNAVAILABLE),
        "grad_cos_min": grad.get("grad_cos_min", METRIC_UNAVAILABLE),
        "val_acc_mean": task.get("val_acc_mean", METRIC_UNAVAILABLE),
        "test_acc_mean": task.get("test_acc_mean", METRIC_UNAVAILABLE),
        "macro_val_gap_mean": macro.get("mean_val_gap", task.get("val_gap_vs_MLP_mean", METRIC_UNAVAILABLE)),
        "macro_ci95_low": macro.get("bootstrap_ci95_low", METRIC_UNAVAILABLE),
        "holm_p": macro.get("holm_corrected_p", METRIC_UNAVAILABLE),
        "test_gap": macro.get("mean_test_gap", METRIC_UNAVAILABLE),
        "ECE_delta": macro.get("ECE_delta_macro", macro.get("ECE_delta", METRIC_UNAVAILABLE)),
        "NLL_delta": macro.get("NLL_delta_macro", macro.get("NLL_delta", METRIC_UNAVAILABLE)),
        "seed_win_rate": win_rate if _finite(win_rate) else METRIC_UNAVAILABLE,
        "seed_win_count": wins,
        "seed_count": total,
        "macro_significant_pass": macro_pass,
        "teacher_free_macro_pass": int(not _external_teacher_used(cid) and cid not in {"B0", "B2"} and macro_pass),
        "memory_ratio_mean": eff["memory_ratio_mean"],
        "memory_ratio_max": eff["memory_ratio_max"],
        "step_ratio_mean": eff["step_ratio_mean"],
        "step_ratio_max": eff["step_ratio_max"],
        "s2_shape_count": eff["s2_shape_count"],
        "s1_shape_count": eff["s1_shape_count"],
        "eff_shape_count": eff["eff_rows"],
        "fullgrid_s2_pass": int(eff["eff_rows"] > 0 and eff["s2_shape_count"] == eff["eff_rows"]),
        "fullgrid_s1_pass": int(eff["eff_rows"] > 0 and eff["s1_shape_count"] == eff["eff_rows"]),
        **auc,
        "fake_data_used": 0,
        "proxy_row_used": 0,
    }


def _write_v78_postprocess(out_dir: Path, args: argparse.Namespace) -> Dict[str, Any]:
    task_rows = _read_csv_rows(out_dir / "p9_task_gate.csv")
    task_summary = _read_csv_rows(out_dir / "p9_task_summary.csv")
    sig_rows = _read_csv_rows(out_dir / "p1_significance_audit.csv")
    grad_rows = _read_csv_rows(out_dir / "p2_full_gradient_correctness.csv")
    eff_rows = _read_csv_rows(out_dir / "p10_efficiency_profiler.csv")
    p7_rows = _read_csv_rows(out_dir / "p7_time_auc_v76.csv")
    task_by = _summary_by_candidate(task_summary)
    macro_by = _macro_by_candidate(sig_rows)
    grad_by = _grad_summary(grad_rows)
    preferred = ["B0", "B2", "M12", "M13", "M4", "M9"]
    candidate_ids = [cid for cid in preferred if cid in task_by]
    candidate_ids.extend(sorted(cid for cid in task_by if cid not in set(candidate_ids)))
    rows = [
        _candidate_row(
            cid,
            task_by=task_by,
            macro_by=macro_by,
            grad_by=grad_by,
            eff_rows=eff_rows,
            p7_rows=p7_rows,
            task_rows=task_rows,
        )
        for cid in candidate_ids
    ]
    write_csv(out_dir / "p0_contract_reproduction_v78.csv", rows)
    write_csv(out_dir / "p1_teacher_free_baseline_v78.csv", [r for r in rows if not _external_teacher_used(str(r["candidate_id"]))])
    write_csv(out_dir / "p6_official_teacher_free_10seed_v78.csv", [r for r in rows if not _external_teacher_used(str(r["candidate_id"]))])
    write_csv(out_dir / "p7_teacher_free_time_auc_v78.csv", [r for r in rows if not _external_teacher_used(str(r["candidate_id"]))])
    write_csv(out_dir / "p9_teacher_free_s1_memory_v78.csv", [
        {
            "candidate_id": row.get("candidate_id"),
            "dataset": row.get("dataset"),
            "batch_size": row.get("batch_size"),
            "external_teacher_used": 0 if str(row.get("candidate_id")) == "M13" else _external_teacher_used(str(row.get("candidate_id"))),
            "memory_ratio": row.get("memory_ratio"),
            "step_ratio": row.get("step_ratio"),
            "peak_allocated_MB": row.get("peak_allocated_MB"),
            "root_input_cache_MB": row.get("linear_body_temp_MB", row.get("cache_x_MB", METRIC_UNAVAILABLE)),
            "hidden_y_cache_MB": row.get("hidden_y_cache_MB", METRIC_UNAVAILABLE),
            "manual_cache_MB": row.get("manual_cache_MB_measured", row.get("cache_total_MB_measured", METRIC_UNAVAILABLE)),
            "optimizer_state_MB": row.get("optimizer_state_MB", METRIC_UNAVAILABLE),
            "top1_memory_source": row.get("top1_memory_source", METRIC_UNAVAILABLE),
            "top2_memory_source": row.get("top2_memory_source", METRIC_UNAVAILABLE),
            "top3_memory_source": row.get("top3_memory_source", METRIC_UNAVAILABLE),
            "unknown_memory_fraction": row.get("unknown_memory_fraction", METRIC_UNAVAILABLE),
            "s2_pass": int(_float(row.get("memory_ratio"), 99.0) <= 1.05 and _float(row.get("step_ratio"), 99.0) <= 1.50),
            "s1_pass": int(_float(row.get("memory_ratio"), 99.0) < 1.00 and _float(row.get("step_ratio"), 99.0) <= 1.35),
            "fake_data_used": 0,
            "proxy_row_used": 0,
        }
        for row in eff_rows
        if str(row.get("candidate_id")) in {"M13", "M12"}
    ])
    diagnostic_rows: List[Dict[str, Any]] = []
    if "M12" in task_by and "M13" in task_by:
        diagnostic_rows.append({
            "stage": "P5",
            "diagnostic": "external_teacher_gain",
            "student_teacher_assisted": "M12",
            "teacher_free_control": "M13",
            "external_teacher_used": 1,
            "teacher_candidate": "C3",
            "macro_gap_M12_minus_M13": _macro_gap_between(task_rows, "M12", "M13"),
            "NLL_M12_minus_M13": _nll_between(task_rows, "M12", "M13"),
            "teacher_dependency_mild": int(_finite(_macro_gap_between(task_rows, "M12", "M13")) and _macro_gap_between(task_rows, "M12", "M13") <= 0.003),
            "teacher_dependency_strong": int(_finite(_macro_gap_between(task_rows, "M12", "M13")) and _macro_gap_between(task_rows, "M12", "M13") >= 0.010),
            "fake_data_used": 0,
            "proxy_row_used": 0,
        })
    if "B2" in task_by and "M12" in task_by:
        diagnostic_rows.append({
            "stage": "P5",
            "diagnostic": "mlp_distill_fairness",
            "student_teacher_assisted": "M12",
            "mlp_distill_control": "B2",
            "external_teacher_used": 1,
            "teacher_candidate": "C3",
            "macro_gap_M12_minus_B2": _macro_gap_between(task_rows, "M12", "B2"),
            "NLL_M12_minus_B2": _nll_between(task_rows, "M12", "B2"),
            "MLP_C3_reaches_M12_within_0p005": int(
                _finite(_macro_gap_between(task_rows, "M12", "B2"))
                and _macro_gap_between(task_rows, "M12", "B2") <= 0.005
            ),
            "fake_data_used": 0,
            "proxy_row_used": 0,
        })
    write_csv(out_dir / "p5_external_teacher_diagnostic_v78.csv", diagnostic_rows)

    official_candidates = [
        r for r in rows
        if str(r.get("candidate_id")) not in {"B0", "B2"}
        and not _external_teacher_used(str(r.get("candidate_id")))
    ]
    official_candidates.sort(
        key=lambda r: (
            int(_is_one(r.get("teacher_free_macro_pass"))),
            _float(r.get("macro_val_gap_mean"), -99.0),
            int(_is_one(r.get("fullgrid_s2_pass"))),
        ),
        reverse=True,
    )
    official = official_candidates[0] if official_candidates else {}
    strict = int(_is_one(official.get("strict_pass")))
    grad = int(_is_one(official.get("grad_pass")))
    macro = int(_is_one(official.get("teacher_free_macro_pass")))
    s2 = int(_is_one(official.get("fullgrid_s2_pass")))
    s1 = int(_is_one(official.get("fullgrid_s1_pass")))
    time_auc = int(_is_one(official.get("time_auc_pass")))
    minimum = int(strict and grad and macro and s2)
    formal = int(minimum and s1 and time_auc)
    if formal:
        route = "S1-TeacherFreeFormalSuccess"
        blocker = "none"
    elif minimum:
        route = "S2-TeacherFreeMinimumSuccess"
        blocker = "s1_or_time_auc_not_closed"
    elif not official_candidates:
        route = "R0-TeacherFreeCandidateNotRun"
        blocker = "missing_teacher_free_official_candidate"
    elif not macro:
        assisted = next((r for r in rows if str(r.get("candidate_id")) == "M12"), {})
        if _is_one(assisted.get("macro_significant_pass")):
            route = "R1-TeacherAssistedSuccessOnly"
            blocker = "teacher_free_macro_gap_not_closed"
        else:
            route = "R2-TeacherFreeMacroNotClosed"
            blocker = "teacher_free_macro_gap_not_closed"
    elif not s2:
        route = "R3-TeacherFreeEfficiencyNotClosed"
        blocker = "teacher_free_s2_fail"
    else:
        route = "R4-SystemEvidenceIncomplete"
        blocker = "missing_required_artifacts"
    route_json = {
        "route": route,
        "best_official_candidate_id": official.get("candidate_id", METRIC_UNAVAILABLE),
        "external_teacher_used": 0,
        "self_teacher_used": 0,
        "teacher_candidate": "none",
        "strict_pass": strict,
        "grad_pass": grad,
        "teacher_free_macro_significant_pass": macro,
        "fullgrid_s2_pass": s2,
        "fullgrid_s1_pass": s1,
        "time_auc_pass": time_auc,
        "success_v78_minimum": minimum,
        "success_v78_formal": formal,
        "macro_gap": official.get("macro_val_gap_mean", METRIC_UNAVAILABLE),
        "ci95_low": official.get("macro_ci95_low", METRIC_UNAVAILABLE),
        "holm_p": official.get("holm_p", METRIC_UNAVAILABLE),
        "test_gap": official.get("test_gap", METRIC_UNAVAILABLE),
        "ECE_delta": official.get("ECE_delta", METRIC_UNAVAILABLE),
        "NLL_delta": official.get("NLL_delta", METRIC_UNAVAILABLE),
        "memory_ratio_mean": official.get("memory_ratio_mean", METRIC_UNAVAILABLE),
        "memory_ratio_max": official.get("memory_ratio_max", METRIC_UNAVAILABLE),
        "step_ratio_mean": official.get("step_ratio_mean", METRIC_UNAVAILABLE),
        "step_ratio_max": official.get("step_ratio_max", METRIC_UNAVAILABLE),
        "s2_shape_count": official.get("s2_shape_count", METRIC_UNAVAILABLE),
        "s1_shape_count": official.get("s1_shape_count", METRIC_UNAVAILABLE),
        "val_loss_auc_step_ratio_vs_B0": official.get("val_loss_auc_step_ratio_vs_B0", METRIC_UNAVAILABLE),
        "val_loss_auc_time_ratio_vs_B0": official.get("val_loss_auc_time_ratio_vs_B0", METRIC_UNAVAILABLE),
        "diagnostic_m12_c3_macro_gap": next((r.get("macro_val_gap_mean") for r in rows if str(r.get("candidate_id")) == "M12"), METRIC_UNAVAILABLE),
        "diagnostic_m12_minus_m13_gap": _macro_gap_between(task_rows, "M12", "M13") if "M12" in task_by and "M13" in task_by else METRIC_UNAVAILABLE,
        "primary_blocker": blocker,
        "classwise_is_hard_gate": False,
        "no_fake": True,
        "no_proxy": True,
    }
    write_csv(out_dir / "p14_candidate_selection_v78.csv", [{**route_json, "stage": "P14", "fake_data_used": 0, "proxy_row_used": 0}])
    save_json(out_dir / "v78_route_decision.json", route_json)

    artifact_paths = [
        out_dir / "candidate_registry.csv",
        out_dir / "p9_task_gate.csv",
        out_dir / "p9_task_trace.csv",
        out_dir / "p9_task_summary.csv",
        out_dir / "p1_significance_audit.csv",
        out_dir / "p2_full_gradient_correctness.csv",
        out_dir / "p10_efficiency_profiler.csv",
        out_dir / "p10_efficiency_summary.csv",
        out_dir / "p7_time_auc_v76.csv",
        out_dir / "p0_contract_reproduction_v78.csv",
        out_dir / "p1_teacher_free_baseline_v78.csv",
        out_dir / "p5_external_teacher_diagnostic_v78.csv",
        out_dir / "p6_official_teacher_free_10seed_v78.csv",
        out_dir / "p7_teacher_free_time_auc_v78.csv",
        out_dir / "p9_teacher_free_s1_memory_v78.csv",
        out_dir / "p14_candidate_selection_v78.csv",
    ]
    audit = v72._audit_fake_proxy(artifact_paths)
    write_csv(out_dir / "v78_provenance_audit.csv", [{
        **audit,
        "plan_path": PLAN_PATH,
        "script_path": SCRIPT_PATH,
        "fake_data_used": 0,
        "proxy_row_used": 0,
    }])
    save_json(out_dir / "v78_manifest.json", {
        "plan_path": PLAN_PATH,
        "script_path": SCRIPT_PATH,
        "runner_reuse": "run_gafu_v76_real.py",
        "out_dir": str(out_dir),
        "source_commit": _git_commit(),
        "git_status": _git_status(),
        "datasets": parse_str_list(args.datasets),
        "seeds": parse_int_list(args.seeds),
        "candidates": parse_str_list(args.candidates),
        "bench_batch_sizes": parse_int_list(args.bench_batch_sizes),
        "grad_batch_sizes": parse_int_list(args.grad_batch_sizes),
        "postprocess_artifact_hashes": {p.name: _hash_file(p) for p in artifact_paths if p.exists()},
        "v78_route": route_json,
        "v78_audit": audit,
    })
    return route_json


def run(args: argparse.Namespace) -> None:
    v76.PLAN_PATH = PLAN_PATH
    v76.SCRIPT_PATH = SCRIPT_PATH
    v76.run(args)
    out_dir = Path(args.out_dir)
    _write_v78_postprocess(out_dir, args)
    for src_name, dst_name in {
        "v76_manifest.json": "v78_reused_v76_manifest.json",
        "v76_route_decision.json": "v78_reused_v76_route_decision.json",
        "v76_provenance_audit.csv": "v78_reused_v76_provenance_audit.csv",
    }.items():
        src = out_dir / src_name
        if src.exists():
            shutil.copyfile(src, out_dir / dst_name)


def parse_args() -> argparse.Namespace:
    return v76.parse_args()


if __name__ == "__main__":
    run(parse_args())
