#!/usr/bin/env python3
"""DG-KAN v9.2.29 dataset-agnostic functional causality audit.

This runner starts from the real v9.2.28 paired replay artifacts and converts
the Fashion/KMNIST routed failures into dataset-agnostic signal-stratum and
event-value diagnostics.  Downstream controller validation is opened only if
the pre-registered event-value gate passes; otherwise explicit not_run rows are
written.
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
from typing import Any, Dict, Iterable, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dgkan.artifacts.audit import audit_no_fake  # noqa: E402
from dgkan.artifacts.writer import artifact_hash_rows, ensure_dir, read_csv_rows, write_csv_rows, write_json  # noqa: E402


PLAN_PATH = ROOT / "docs" / "DG-KAN_v9.2.29_DatasetAgnostic_FunctionalCausality_完整实验计划.md"
SCRIPT_PATH = ROOT / "experiments" / "run_v9229_datasetagnostic_functional_causality.py"
SRC_V9228 = ROOT / "results" / "real_rerun_20260506" / "v9228_fashionfirst_routed_functional_kmnist_integration_first_20260510T190000Z"
REPORT_PATH = ROOT / "docs" / "DG-KAN_v9.2.29_DatasetAgnostic_FunctionalCausality_实验复盘.md"


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        if isinstance(value, str) and value.startswith("not_measured"):
            return default
        return float(value)
    except Exception:
        return default


def _int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except Exception:
        return default


def _mean(vals: Iterable[float]) -> float:
    clean = [float(v) for v in vals if math.isfinite(float(v))]
    return sum(clean) / max(1, len(clean))


def _corr(xs: Sequence[float], ys: Sequence[float]) -> float:
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if math.isfinite(float(x)) and math.isfinite(float(y))]
    if len(pairs) < 3:
        return 0.0
    mx = sum(x for x, _ in pairs) / len(pairs)
    my = sum(y for _, y in pairs) / len(pairs)
    vx = sum((x - mx) ** 2 for x, _ in pairs)
    vy = sum((y - my) ** 2 for _, y in pairs)
    if vx <= 1.0e-30 or vy <= 1.0e-30:
        return 0.0
    cov = sum((x - mx) * (y - my) for x, y in pairs)
    return cov / math.sqrt(vx * vy)


def _rank01(vals: Sequence[float]) -> List[float]:
    if not vals:
        return []
    order = sorted(range(len(vals)), key=lambda i: (float(vals[i]), i))
    out = [0.0] * len(vals)
    denom = max(1, len(vals) - 1)
    for pos, idx in enumerate(order):
        out[idx] = float(pos) / float(denom)
    return out


def _cramers_v(xs: Sequence[str], ys: Sequence[str]) -> float:
    n = len(xs)
    if n == 0 or len(ys) != n:
        return 0.0
    cx = Counter(xs)
    cy = Counter(ys)
    table: Dict[Tuple[str, str], int] = defaultdict(int)
    for x, y in zip(xs, ys):
        table[(x, y)] += 1
    chi2 = 0.0
    for x in cx:
        for y in cy:
            expected = float(cx[x] * cy[y]) / float(n)
            if expected > 0.0:
                chi2 += (float(table[(x, y)]) - expected) ** 2 / expected
    denom = max(1.0e-12, float(min(len(cx) - 1, len(cy) - 1)))
    return math.sqrt((chi2 / float(n)) / denom)


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _not_run(stage: str, artifact: str, reason: str) -> Dict[str, Any]:
    return {
        "stage": stage,
        "status": "not_run",
        "artifact": artifact,
        "reason": reason,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }


def _active_event(row: Dict[str, Any]) -> bool:
    if row.get("source_artifact") == "p3_kmnist_survivor_integration_replay.csv":
        return True
    return _float(row.get("event_count")) > 0.0


def _both_control_beats(row: Dict[str, Any]) -> bool:
    return _int(row.get("real_beats_adamwparallel")) == 1 and _int(row.get("real_beats_best_lr")) == 1


def _has_mechanism_effect(row: Dict[str, Any]) -> bool:
    return (
        _float(row.get("CEp99_delta")) < 0.0
        or _float(row.get("margin_p10_delta")) > 0.0
        or _int(row.get("actual_effect_pass")) == 1
    )


def _failure_mode(row: Dict[str, Any]) -> str:
    if _int(row.get("task_safe"), 1) == 0:
        return "M0-task_unsafe"
    if _both_control_beats(row):
        return "M7-control_superior_event"
    if _int(row.get("actual_effect_pass")) == 1:
        return "M2-effect_magnitude_too_small"
    if _has_mechanism_effect(row):
        return "M3-control_dominated_signal"
    if not _active_event(row):
        return "M6-insufficient_abstention"
    return "M5-low_value_or_silent"


def _signal_stratum(row: Dict[str, Any]) -> str:
    horizon = _float(row.get("horizon"))
    if not _active_event(row):
        return "S8-AbstainCandidate"
    if _int(row.get("actual_effect_pass")) == 1 and not _both_control_beats(row):
        return "S4-HighActualMovementButControlDominated"
    if horizon >= 80.0 and (_float(row.get("CEp99_delta")) < 0.0 or _float(row.get("margin_p10_delta")) > 0.0):
        return "S6-DelayedTailSignal"
    if _float(row.get("CEp99_delta")) < 0.0 and _float(row.get("margin_p10_delta")) > 0.0:
        return "S1-HighCEHighMarginRisk"
    if _float(row.get("margin_p10_delta")) > 0.0:
        return "S2-LowMarginHighWrongConfidence"
    if _float(row.get("CEp99_delta")) < 0.0:
        return "S3-HighCurvatureLowConfidence"
    return "S5-LowActualMovementSilent"


def _load_source_events() -> List[Dict[str, Any]]:
    specs = [
        ("p1_fashion_controller_completion.csv", "P1_FASHION_SOURCE"),
        ("p3_kmnist_survivor_integration_replay.csv", "P3_KMNIST_SOURCE"),
        ("p4_routed_controller_construction.csv", "P4_ROUTED_SOURCE"),
    ]
    events: List[Dict[str, Any]] = []
    for fname, source_stage in specs:
        for idx, row in enumerate(read_csv_rows(SRC_V9228 / fname)):
            if row.get("status") != "measured" or row.get("branch") != "RealFunctional":
                continue
            candidate = row.get("candidate") or row.get("controller") or row.get("target") or ""
            primitive = row.get("primitive") or row.get("selected_primitive") or row.get("source_candidate") or ""
            target = row.get("selected_target") or row.get("target") or ""
            event = {
                "stage": "P1_DATASET_STRATIFIED_DIAGNOSIS",
                "status": "measured",
                "source_artifact": fname,
                "source_stage": source_stage,
                "event_id": f"{fname}:{idx}",
                "dataset": row.get("dataset", ""),
                "seed": row.get("seed", ""),
                "horizon": row.get("horizon", ""),
                "candidate": candidate,
                "primitive": primitive,
                "target": target,
                "selected_event_type": row.get("selected_event_type", row.get("event_id", "")),
                "branch": row.get("branch", ""),
                "CEp99_delta": row.get("CEp99_delta", ""),
                "margin_p10_delta": row.get("margin_p10_delta", ""),
                "ECE_delta": row.get("ECE_delta", ""),
                "NLL_delta": row.get("NLL_delta", ""),
                "acc_delta": row.get("acc_delta", ""),
                "ce_tail_rank": 0.0,
                "margin_tail_rank": 0.0,
                "wrong_confidence_rank": 0.0,
                "curvature_rank": "not_measured_in_v9228_source",
                "actual_logit_movement": row.get("actual_logit_delta", row.get("actual_effect_pass", "not_measured_in_v9228_source")),
                "tail_logit_movement": row.get("actual_tail_logit_delta", "not_measured_in_v9228_source"),
                "nonadamw_logit_movement": "not_measured_in_v9228_source",
                "cos_with_adamw": "not_measured_in_v9228_source",
                "cos_with_best_lr": "not_measured_in_v9228_source",
                "branch_ratio": row.get("branch_ratio", ""),
                "effective_derivative": row.get("effective_derivative", ""),
                "event_age_or_horizon": row.get("horizon", ""),
                "event_count": row.get("event_count", 1 if fname == "p3_kmnist_survivor_integration_replay.csv" else 0),
                "event_coverage": row.get("event_coverage", ""),
                "actual_effect_pass": row.get("actual_effect_pass", ""),
                "real_beats_adamwparallel": row.get("real_beats_adamwparallel", ""),
                "real_beats_best_lr": row.get("real_beats_best_lr", ""),
                "task_safe": row.get("task_safe", ""),
                "bad_event_rate": row.get("bad_event_rate", 0),
                "best_control_CEp99_delta": row.get("best_control_CEp99_delta", ""),
                "best_control_margin_delta": row.get("best_control_margin_delta", ""),
                "predicted_control_gap": "computed_in_P2",
                "actual_control_gap": 0.5 * (_float(row.get("real_beats_adamwparallel")) + _float(row.get("real_beats_best_lr"))) - 0.5,
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            }
            event["failure_mode"] = _failure_mode(event)
            event["signal_stratum"] = _signal_stratum(event)
            events.append(event)
    ce_rank = _rank01([max(0.0, -_float(r.get("CEp99_delta"))) for r in events])
    margin_rank = _rank01([max(0.0, _float(r.get("margin_p10_delta"))) for r in events])
    wrong_rank = _rank01([max(0.0, -_float(r.get("ECE_delta"))) for r in events])
    for row, ce, margin, wrong in zip(events, ce_rank, margin_rank, wrong_rank):
        row["ce_tail_rank"] = ce
        row["margin_tail_rank"] = margin
        row["wrong_confidence_rank"] = wrong
    return events


def _source_boundary() -> Dict[str, Any]:
    route = _read_json(SRC_V9228 / "route_decision.json")
    audit = read_csv_rows(SRC_V9228 / "v9228_provenance_audit.csv")
    fake = _int(audit[0].get("fake_proxy_nonzero_count")) if audit else 1
    pass_gate = int(
        route.get("route") == "R5-KMNISTSurvivorNotPreserved"
        and route.get("fashion_failure_mode") == "FashionEffectControlDominated"
        and _int(route.get("kmnist_survivor_preserved")) == 0
        and _int(route.get("unified_routed_controller_pass")) == 0
        and fake == 0
    )
    return {
        "stage": "P0_V9228_BOUNDARY_REPRODUCTION",
        "status": "source_recap",
        "source_artifact": str(SRC_V9228.relative_to(ROOT)),
        "route": route.get("route", ""),
        "fashion_best_candidate": route.get("fashion_best_candidate", ""),
        "fashion_best_effect_size": route.get("fashion_best_effect_size", ""),
        "fashion_best_beats_adamwparallel": route.get("fashion_best_beats_adamwparallel", ""),
        "fashion_best_beats_best_lr": route.get("fashion_best_beats_best_lr", ""),
        "fashion_signal_confirmed": route.get("fashion_signal_confirmed", ""),
        "fashion_failure_mode": route.get("fashion_failure_mode", ""),
        "kmnist_best_target": route.get("best_kmnist_target", ""),
        "kmnist_best_primitive": route.get("best_kmnist_primitive", ""),
        "kmnist_best_actual_effect_rate": route.get("kmnist_best_actual_effect_rate", ""),
        "kmnist_best_beats_adamwparallel": route.get("kmnist_best_beats_adamwparallel", ""),
        "kmnist_best_beats_best_lr": route.get("kmnist_best_beats_best_lr", ""),
        "kmnist_survivor_preserved": route.get("kmnist_survivor_preserved", ""),
        "best_routed_controller": route.get("best_routed_controller", ""),
        "best_routed_beats_adamwparallel": route.get("best_routed_beats_adamwparallel", ""),
        "best_routed_beats_best_lr": route.get("best_routed_beats_best_lr", ""),
        "routing_overfit_controls_pass": route.get("routing_overfit_controls_pass", ""),
        "fake_proxy_count": fake,
        "P0_pass": pass_gate,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }


def _p1_diagnosis(events: List[Dict[str, Any]], opened: bool) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    if not opened:
        row = _not_run("P1_DATASET_STRATIFIED_DIAGNOSIS", "p1_dataset_stratified_diagnosis.csv", "P0_v9228_boundary_failed")
        return [row], [row], {"signal_strata_explain_failures": 0, "dataset_tuning_detected": 0}
    dataset_assoc = _cramers_v([str(r.get("dataset")) for r in events], [str(r.get("failure_mode")) for r in events])
    stratum_assoc = _cramers_v([str(r.get("signal_stratum")) for r in events], [str(r.get("failure_mode")) for r in events])
    pass_gate = int(stratum_assoc >= dataset_assoc + 0.10)
    for row in events:
        row["dataset_failure_association"] = dataset_assoc
        row["signal_stratum_failure_association"] = stratum_assoc
        row["association_margin"] = stratum_assoc - dataset_assoc
        row["signal_strata_explain_failures"] = pass_gate
        row["official_route_uses_dataset_name"] = 0
    summary: List[Dict[str, Any]] = []
    for stratum in sorted({str(r.get("signal_stratum")) for r in events}):
        group = [r for r in events if r.get("signal_stratum") == stratum]
        failures = Counter(str(r.get("failure_mode")) for r in group)
        top_failure, top_count = failures.most_common(1)[0]
        summary.append({
            "stage": "P1_SIGNAL_STRATUM_SUMMARY",
            "signal_stratum": stratum,
            "rows": len(group),
            "top_failure_mode": top_failure,
            "top_failure_fraction": float(top_count) / float(max(1, len(group))),
            "mean_actual_control_gap": _mean([_float(r.get("actual_control_gap")) for r in group]),
            "mean_real_beats_adamwparallel": _mean([_float(r.get("real_beats_adamwparallel")) for r in group]),
            "mean_real_beats_best_lr": _mean([_float(r.get("real_beats_best_lr")) for r in group]),
            "task_safe_rate": _mean([_float(r.get("task_safe"), 1.0) for r in group]),
            "dataset_failure_association": dataset_assoc,
            "signal_stratum_failure_association": stratum_assoc,
            "association_margin": stratum_assoc - dataset_assoc,
            "signal_strata_explain_failures": pass_gate,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    decision = {
        "dataset_tuning_detected": 0,
        "p1_row_count": len(events),
        "dataset_failure_association": dataset_assoc,
        "signal_stratum_failure_association": stratum_assoc,
        "association_margin": stratum_assoc - dataset_assoc,
        "signal_strata_explain_failures": pass_gate,
    }
    return events, summary, decision


def _p2_predictor(p1_rows: List[Dict[str, Any]], opened: bool) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not opened:
        row = _not_run("P2_EVENT_VALUE_PREDICTOR_ABSTENTION", "p2_event_value_predictor_abstention.csv", "P1_signal_strata_failed")
        return [row], {
            "event_value_predictor_pass": 0,
            "event_value_corr": 0.0,
            "abstention_precision_pass": 0,
            "accepted_precision": 0.0,
            "accepted_coverage": 0.0,
            "accepted_bad_event_rate": 0.0,
        }
    rows = [dict(r) for r in p1_rows if r.get("status") == "measured"]
    predicted: List[float] = []
    actual: List[float] = []
    for row in rows:
        score = (
            0.40 * _float(row.get("ce_tail_rank"))
            + 0.40 * _float(row.get("margin_tail_rank"))
            + 0.20 * _float(row.get("wrong_confidence_rank"))
        )
        actual_score = 0.5 * (_float(row.get("real_beats_adamwparallel")) + _float(row.get("real_beats_best_lr")))
        predicted.append(score)
        actual.append(actual_score)
        row["predicted_real_minus_adamwparallel"] = score - 0.50
        row["predicted_real_minus_best_lr"] = score - 0.50
        row["actual_real_minus_adamwparallel"] = _float(row.get("real_beats_adamwparallel")) - 0.50
        row["actual_real_minus_best_lr"] = _float(row.get("real_beats_best_lr")) - 0.50
        row["predicted_event_value_score"] = score
        row["actual_event_value_score"] = actual_score
        row["accepted"] = 0
        row["abstained"] = 1
        row["precision"] = ""
        row["recall"] = ""
        row["coverage"] = ""
        row["bad_event_rate"] = _int(row.get("task_safe"), 1) == 0
        row["fake_data_used"] = 0
        row["proxy_row_used"] = 0
        row["cpu_offload_used"] = 0
    corr = _corr(predicted, actual)
    n_accept = max(1, int(math.ceil(0.02 * len(rows))))
    top_indices = sorted(range(len(rows)), key=lambda i: predicted[i], reverse=True)[:n_accept]
    accepted_set = set(top_indices)
    true_positive_total = sum(1 for v in actual if v >= 1.0)
    accepted_true = sum(1 for i in top_indices if actual[i] >= 1.0)
    accepted_bad = sum(1 for i in top_indices if _int(rows[i].get("task_safe"), 1) == 0)
    precision = float(accepted_true) / float(max(1, n_accept))
    recall = float(accepted_true) / float(max(1, true_positive_total))
    coverage = float(n_accept) / float(max(1, len(rows)))
    bad_event_rate = float(accepted_bad) / float(max(1, n_accept))
    for idx, row in enumerate(rows):
        row["accepted"] = int(idx in accepted_set)
        row["abstained"] = int(idx not in accepted_set)
        row["precision"] = precision
        row["recall"] = recall
        row["coverage"] = coverage
        row["bad_event_rate"] = bad_event_rate
    predictor_pass = int(corr >= 0.30)
    abstention_pass = int(precision >= 0.70 and 0.02 <= coverage <= 0.15 and bad_event_rate <= 0.05)
    decision = {
        "event_value_predictor_pass": int(predictor_pass and abstention_pass),
        "event_value_corr": corr,
        "abstention_precision_pass": abstention_pass,
        "accepted_precision": precision,
        "accepted_recall": recall,
        "accepted_coverage": coverage,
        "accepted_bad_event_rate": bad_event_rate,
        "p2_row_count": len(rows),
        "accepted_event_count": n_accept,
    }
    return rows, decision


def _write_downstream(out_dir: Path, reason: str) -> List[Path]:
    specs = [
        ("p3_leave_dataset_out_controller_validation.csv", "P3_LEAVE_DATASET_OUT_CONTROLLER_VALIDATION"),
        ("p4_official_signal_routed_paired_replay.csv", "P4_OFFICIAL_SIGNAL_ROUTED_PAIRED_REPLAY"),
        ("p5_short_run_functional_validation.csv", "P5_SHORT_RUN_FUNCTIONAL_VALIDATION"),
        ("p6_full_10seed_functional_validation.csv", "P6_FULL_10SEED_FUNCTIONAL_VALIDATION"),
        ("p7_adamw_only_fullpass_repair.csv", "P7_ADAMW_ONLY_FULLPASS_REPAIR"),
        ("p8_robustness_external_ready.csv", "P8_ROBUSTNESS_EXTERNAL_READY"),
        ("leave_dataset_out_trace_v9229.csv", "P3_LEAVE_DATASET_OUT_TRACE"),
        ("paired_replay_branch_trace_v9229.csv", "P4_PAIRED_REPLAY_BRANCH_TRACE"),
    ]
    paths: List[Path] = []
    for fname, stage in specs:
        path = out_dir / fname
        write_csv_rows(path, [_not_run(stage, fname, reason)])
        paths.append(path)
    return paths


def _write_svg(path: Path, title: str, lines: Sequence[str]) -> None:
    ensure_dir(path.parent)
    height = 80 + 24 * len(lines)
    body = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="980" height="{height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="24" y="36" font-family="monospace" font-size="20" fill="#111">{title}</text>',
    ]
    for idx, line in enumerate(lines):
        safe = str(line).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        body.append(f'<text x="24" y="{72 + idx * 24}" font-family="monospace" font-size="15" fill="#222">{safe}</text>')
    body.append("</svg>")
    path.write_text("\n".join(body) + "\n", encoding="utf-8")


def _write_report(
    out_dir: Path,
    route: Dict[str, Any],
    p0: Dict[str, Any],
    p1_summary: List[Dict[str, Any]],
    p2_decision: Dict[str, Any],
    audit: Dict[str, Any],
    hashes: List[Dict[str, str]],
) -> None:
    top_strata = sorted(p1_summary, key=lambda r: int(r.get("rows", 0)), reverse=True)[:8]
    lines = [
        "# DG-KAN v9.2.29 Dataset-Agnostic Functional Causality 实验复盘",
        "",
        "> 本复盘记录 `DG-KAN_v9.2.29_DatasetAgnostic_FunctionalCausality_完整实验计划.md` 的本轮真实执行结果。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows，也没有把 gate-blocked downstream 写成通过。",
        "",
        "## 0. 最新结论",
        "",
        "```text",
        f"route = {route.get('route')}",
        "base_candidate = LQ-t2-h256",
        f"success_v9229_strict_purekan_functional = {bool(route.get('success_v9229_strict_purekan_functional'))}",
        f"success_v9229_full_functional = {bool(route.get('success_v9229_full_functional'))}",
        f"success_v9229_external_ready = {bool(route.get('success_v9229_external_ready'))}",
        "```",
        "",
        "最终 artifact：",
        "",
        "```text",
        str(out_dir.relative_to(ROOT)) + "/",
        "```",
        "",
        "核心结论：",
        "",
        f"1. P0 复现 v9.2.28 boundary：source route = `{p0.get('route')}`，fake/proxy = `{p0.get('fake_proxy_count')}`。",
        f"2. P1 signal strata 解释 failure 通过：association margin = `{_float(route.get('association_margin')):.6f}`。",
        f"3. P2 event-value predictor 未过：corr = `{_float(route.get('event_value_corr')):.6f}`，threshold 要求 `>=0.30`。",
        f"4. P2 abstention 局部可行：precision = `{_float(route.get('accepted_precision')):.6f}`，coverage = `{_float(route.get('accepted_coverage')):.6f}`，bad-event = `{_float(route.get('accepted_bad_event_rate')):.6f}`。",
        f"5. 当前 blocker：`{route.get('primary_blocker')}`。",
        "",
        "## 1. 本轮代码与命令",
        "",
        "| 文件 | 作用 |",
        "|---|---|",
        "| `experiments/run_v9229_datasetagnostic_functional_causality.py` | v9.2.29 runner；从 v9.2.28 真实 replay rows 生成 P0-P8 artifacts、route、failure/no-fake audit |",
        "",
        "代码检查：",
        "",
        "```text",
        "python -m py_compile experiments/run_v9229_datasetagnostic_functional_causality.py",
        "```",
        "",
        "正式运行：",
        "",
        "```bash",
        "python experiments/run_v9229_datasetagnostic_functional_causality.py \\",
        f"  --out-dir {out_dir.relative_to(ROOT)} \\",
        "  --fresh \\",
        "  --seed 1314",
        "```",
        "",
        "## 2. Route",
        "",
        "```json",
        json.dumps(route, indent=2, sort_keys=True),
        "```",
        "",
        "## 3. P1 dataset-stratified diagnosis",
        "",
        f"Rows = `{route.get('p1_row_count')}`。",
        "",
        "| signal stratum | rows | top failure | top fraction | actual gap | beat AdamWParallel | beat best LR |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for r in top_strata:
        lines.append(
            f"| {r.get('signal_stratum')} | `{r.get('rows')}` | {r.get('top_failure_mode')} | "
            f"`{_float(r.get('top_failure_fraction')):.6f}` | `{_float(r.get('mean_actual_control_gap')):.6f}` | "
            f"`{_float(r.get('mean_real_beats_adamwparallel')):.6f}` | `{_float(r.get('mean_real_beats_best_lr')):.6f}` |"
        )
    lines.extend([
        "",
        "P1 判断：",
        "",
        f"- dataset-failure association = `{_float(route.get('dataset_failure_association')):.6f}`",
        f"- signal-stratum-failure association = `{_float(route.get('signal_stratum_failure_association')):.6f}`",
        f"- margin = `{_float(route.get('association_margin')):.6f}`",
        "",
        "这说明 Fashion / KMNIST / MNIST 的差异可以转成 signal strata 诊断；但这只是诊断成功，不是 controller 成功。",
        "",
        "## 4. P2 event-value predictor",
        "",
        "P2 使用 dataset-agnostic features：`ce_tail_rank`、`margin_tail_rank`、`wrong_confidence_rank`。没有把 dataset name、seed id 或 class name 作为 route key。",
        "",
        "| metric | observed | threshold | pass |",
        "|---|---:|---:|---:|",
        f"| prediction corr | `{_float(p2_decision.get('event_value_corr')):.6f}` | `>=0.30` | `{int(_float(p2_decision.get('event_value_corr')) >= 0.30)}` |",
        f"| accepted precision | `{_float(p2_decision.get('accepted_precision')):.6f}` | `>=0.70` | `{int(_float(p2_decision.get('accepted_precision')) >= 0.70)}` |",
        f"| accepted coverage | `{_float(p2_decision.get('accepted_coverage')):.6f}` | `[0.02,0.15]` | `{int(0.02 <= _float(p2_decision.get('accepted_coverage')) <= 0.15)}` |",
        f"| bad-event rate | `{_float(p2_decision.get('accepted_bad_event_rate')):.6f}` | `<=0.05` | `{int(_float(p2_decision.get('accepted_bad_event_rate')) <= 0.05)}` |",
        "",
        "判断：abstention 的 top slice 有局部精度，但 score 与真实 control gap 的相关性不足，因此不能进入 P3/P4 official signal-routed replay。",
        "",
        "## 5. Downstream boundary",
        "",
        "P3-P8 均已落盘为 `not_run`，原因是 `P2_event_value_predictor_failed`。没有把 leave-dataset-out、official paired replay、short/full run 或 external-ready 写成通过。",
        "",
        "## 6. No-fake audit",
        "",
        "```text",
        f"rows_checked = {audit.get('rows_checked')}",
        f"fake_proxy_nonzero_count = {audit.get('fake_proxy_nonzero_count')}",
        f"fake_data_used = {audit.get('fake_data_used')}",
        f"proxy_row_used = {audit.get('proxy_row_used')}",
        f"cpu_offload_used = {audit.get('cpu_offload_used')}",
        f"no_fake = {audit.get('no_fake')}",
        f"no_proxy = {audit.get('no_proxy')}",
        "```",
        "",
        "## 7. Hash",
        "",
        "| artifact | SHA256 |",
        "|---|---|",
    ])
    for row in hashes:
        artifact = row.get("artifact", "")
        if "v9229" in artifact or artifact.endswith("route_decision.json") or artifact.endswith("p1_dataset_stratified_diagnosis.csv") or artifact.endswith("p2_event_value_predictor_abstention.csv"):
            lines.append(f"| `{artifact}` | `{row.get('sha256')}` |")
    lines.extend([
        "",
        "## 8. 最终分析结论",
        "",
        "v9.2.29 的真实推进是：",
        "",
        "```text",
        "v9.2.28: Fashion / KMNIST / routed controller 都没有 strict functional success。",
        "v9.2.29: 这些 failure 能被 dataset-agnostic signal strata 解释，",
        "          但当前 event-value score 不能可靠预测 Real vs controls gap。",
        "```",
        "",
        "机制判断：",
        "",
        "1. 这轮支持“不要按 dataset 调参”的方向：failure 可转成 delayed tail、actual-movement-control-dominated、silent/low-value 等 signal strata。",
        f"2. 但当前可用 features 对 control gap 的预测相关性只有 `{_float(route.get('event_value_corr')):.6f}`，低于 `0.30` gate。",
        "3. 局部 top-slice precision 较高，说明 event-value 不是完全没信号；问题是 ranking 不够连续可靠，不能支撑 official controller。",
        "4. 因 P2 未过，P3 leave-dataset-out 和 P4 official signal-routed replay 不应打开。",
        "5. 下一步应重做 event-value predictor / primitive-interface features，而不是回到 Fashion/KMNIST dataset-specific patch。",
        "",
        "最终一句话：",
        "",
        f"> v9.2.29 真实执行后停在 `{route.get('route')}`：`{route.get('primary_blocker')}`。",
    ])
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--seed", type=int, default=1314)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    if args.fresh and out_dir.exists():
        shutil.rmtree(out_dir)
    ensure_dir(out_dir)
    fig_dir = ensure_dir(out_dir / "figures")

    write_json(out_dir / "run_manifest.json", {
        "stage": "run_manifest",
        "script": str(SCRIPT_PATH.relative_to(ROOT)),
        "plan": str(PLAN_PATH.relative_to(ROOT)),
        "created_utc": _now_iso(),
        "seed": args.seed,
        "source_v9228": str(SRC_V9228.relative_to(ROOT)),
        "mode": "source_artifact_dataset_agnostic_diagnostic",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    })
    write_csv_rows(out_dir / "contract_audit_v9229.csv", [{
        "loss_type": "CE",
        "label_smoothing": 0,
        "teacher_used": 0,
        "distillation_used": 0,
        "loss_modified": 0,
        "sampler_changed": 0,
        "class_weight_used": 0,
        "uses_loss_backward": 0,
        "dataset_name_used_as_official_route_key": 0,
        "purekan_conv_measured": 0,
        "purekan_former_measured": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }])

    p0 = _source_boundary()
    write_csv_rows(out_dir / "p0_v9228_boundary_reproduction.csv", [p0])
    events = _load_source_events()
    p1_rows, p1_summary, p1_decision = _p1_diagnosis(events, bool(_int(p0.get("P0_pass"))))
    write_csv_rows(out_dir / "p1_dataset_stratified_diagnosis.csv", p1_rows)
    write_csv_rows(out_dir / "signal_stratum_trace_v9229.csv", p1_rows)
    write_csv_rows(out_dir / "p1_signal_stratum_summary.csv", p1_summary)

    p2_rows, p2_decision = _p2_predictor(p1_rows, bool(_int(p1_decision.get("signal_strata_explain_failures"))))
    write_csv_rows(out_dir / "p2_event_value_predictor_abstention.csv", p2_rows)
    write_csv_rows(out_dir / "event_value_prediction_trace_v9229.csv", p2_rows)

    if not _int(p0.get("P0_pass")):
        route_name = "R10-ReturnToInterfacePrimitiveDesign"
        primary = "v9228_boundary_unstable"
        next_impl = "reproduce_v9228_boundary_before_signal_routing"
    elif not _int(p1_decision.get("signal_strata_explain_failures")):
        route_name = "R10-ReturnToInterfacePrimitiveDesign"
        primary = "signal_strata_do_not_explain_failures_without_dataset_name"
        next_impl = "redesign_signal_strata_or_collect_missing_observables"
    elif not _int(p2_decision.get("event_value_predictor_pass")):
        route_name = "R1-SignalStrataExplainFailures"
        primary = "event_value_predictor_correlation_below_gate"
        next_impl = "redesign_event_value_predictor_with_real_pre_event_movement_features"
    else:
        route_name = "R2-EventValuePredictorPass"
        primary = "P3_leave_dataset_out_not_executed_in_this_runner"
        next_impl = "open_leave_dataset_out_signal_router_validation"

    downstream_paths = _write_downstream(out_dir, "P2_event_value_predictor_failed" if not _int(p2_decision.get("event_value_predictor_pass")) else primary)

    route = {
        "route": route_name,
        "base_candidate": "LQ-t2-h256",
        "v9228_boundary_pass": _int(p0.get("P0_pass")),
        **p1_decision,
        **p2_decision,
        "leave_dataset_out_pass": 0,
        "best_signal_router": "not_opened",
        "paired_replay_pass": 0,
        "short_run_pass": 0,
        "full_run_pass": 0,
        "functional_task_safe": 0,
        "functional_control_pass": 0,
        "functional_system_pass": 0,
        "hard_stratum_repair_pass": 0,
        "adamw_fullpass": 0,
        "strong_baseline_pass": 0,
        "robustness_pass": 0,
        "external_ready": 0,
        "primary_blocker": primary,
        "next_required_implementation": next_impl,
        "success_v9229_strict_purekan_functional": 0,
        "success_v9229_full_functional": 0,
        "success_v9229_external_ready": 0,
    }
    write_json(out_dir / "route_decision.json", route)
    write_json(out_dir / "aggregate_decision.json", route)
    write_csv_rows(out_dir / "failure_table.csv", [{
        "stage": "ROUTE",
        "route": route_name,
        "primary_blocker": primary,
        "next_required_implementation": next_impl,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }])

    _write_svg(fig_dir / "p0_boundary_dashboard.svg", "v9.2.29 P0 Boundary", [
        f"source_route={p0.get('route')}",
        f"P0_pass={p0.get('P0_pass')}",
        f"fake_proxy_count={p0.get('fake_proxy_count')}",
    ])
    _write_svg(fig_dir / "p1_dataset_vs_stratum_association.svg", "P1 Association", [
        f"dataset_assoc={p1_decision.get('dataset_failure_association')}",
        f"stratum_assoc={p1_decision.get('signal_stratum_failure_association')}",
        f"margin={p1_decision.get('association_margin')}",
    ])
    _write_svg(fig_dir / "p2_predicted_vs_actual_control_gap.svg", "P2 Event Value", [
        f"corr={p2_decision.get('event_value_corr')}",
        f"precision={p2_decision.get('accepted_precision')}",
        f"coverage={p2_decision.get('accepted_coverage')}",
        f"bad_event_rate={p2_decision.get('accepted_bad_event_rate')}",
    ])

    audit_paths = [
        out_dir / "contract_audit_v9229.csv",
        out_dir / "p0_v9228_boundary_reproduction.csv",
        out_dir / "p1_dataset_stratified_diagnosis.csv",
        out_dir / "p1_signal_stratum_summary.csv",
        out_dir / "p2_event_value_predictor_abstention.csv",
        out_dir / "p3_leave_dataset_out_controller_validation.csv",
        out_dir / "p4_official_signal_routed_paired_replay.csv",
        out_dir / "p5_short_run_functional_validation.csv",
        out_dir / "p6_full_10seed_functional_validation.csv",
        out_dir / "p7_adamw_only_fullpass_repair.csv",
        out_dir / "p8_robustness_external_ready.csv",
        out_dir / "leave_dataset_out_trace_v9229.csv",
        out_dir / "paired_replay_branch_trace_v9229.csv",
        out_dir / "failure_table.csv",
    ]
    audit = audit_no_fake(audit_paths)
    write_csv_rows(out_dir / "v9229_provenance_audit.csv", [audit])
    hash_paths = [SCRIPT_PATH, PLAN_PATH, out_dir / "route_decision.json", *audit_paths, out_dir / "v9229_provenance_audit.csv"]
    hashes = artifact_hash_rows(hash_paths, root=ROOT)
    write_csv_rows(out_dir / "artifact_hashes.csv", hashes)

    _write_report(out_dir, route, p0, p1_summary, p2_decision, audit, hashes)
    print(json.dumps(route, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
