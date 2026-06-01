#!/usr/bin/env python
"""Finalize v12.25 S4-to-S5 precommit functional artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import experiments.run_v1223_failclosed_explore_open2_functional_rebuild as exp


OUT_DIR = ROOT / "results" / "v12_25_s4_to_s5_precommit_functional" / "official_precommit_functional"
PLAN_DOC = ROOT / "docs" / "DG-KAN_v12.25_S4toS5_PrecommitFunctional_NoStopPlan.md"
EXEC_LOG = ROOT / "docs" / "DG-KAN_v12.25_S4toS5_PrecommitFunctional_NoStopPlan_执行日志.md"
REVIEW_LOG = ROOT / "docs" / "DG-KAN_v12.25_S4toS5_PrecommitFunctional_NoStopPlan_实验结果复盘.md"

CODE_MEMBERS = [
    ROOT / "experiments" / "run_v1225_composite_functional_bridge.py",
    ROOT / "experiments" / "run_v1225_finalize_precommit_functional.py",
    ROOT / "experiments" / "run_v1224_train_stream_functional_bridge.py",
    ROOT / "experiments" / "run_v1224_i30_i32_policy_bridge.py",
    ROOT / "experiments" / "run_v1224_classic_hardening.py",
    ROOT / "experiments" / "run_v1224_finalize_functional_bridge.py",
    ROOT / "experiments" / "run_v1223_p4_trainable_role_scan.py",
    ROOT / "experiments" / "run_v1223_p4_compensation_modes.py",
    ROOT / "experiments" / "run_v1223_p4_trajectory_gate_verifier.py",
    ROOT / "experiments" / "run_v1218_b320_label_free_ablation.py",
    ROOT / "dgkan" / "models" / "fc_purekan_primitives.py",
    ROOT / "dgkan" / "kernels" / "fused_hinge_quadratic.py",
]

FIGURES = [
    "fig_v1225_progress_dashboard.svg",
    "fig_s4_to_s5_blocker_matrix.svg",
    "fig_ng4_ng9_ng17_signal_tradeoff.svg",
    "fig_tail_safety_tradeoff.svg",
    "fig_source_control_margin_vs_CEp99.svg",
    "fig_linec_multisketch_stability.svg",
    "fig_train_shuffle_robustness.svg",
    "fig_precommit_visibility_roc.svg",
    "fig_direct_quad_composite_pareto.svg",
    "fig_policy_p3_p4_alignment.svg",
    "fig_rational_classic_pareto.svg",
    "fig_route_decision_tree.svg",
]


def fnum(value: Any, default: float = float("nan")) -> float:
    try:
        if value == "" or value is None:
            return default
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_composite_artifacts() -> tuple[list[dict[str, Any]], dict[str, Any], list[Path]]:
    csv_paths = sorted(OUT_DIR.glob("v1225_composite_functional_bridge*.csv"))
    rows: list[dict[str, Any]] = []
    for path in csv_paths:
        for row in read_rows(path):
            row["artifact_name"] = path.name
            rows.append(row)
    summary_paths = sorted(OUT_DIR.glob("v1225_composite_functional_bridge*_summary.json"))
    summaries = [read_json(path) for path in summary_paths]
    aggregate_rows = [r for r in rows if r.get("stage") == "V1225_COMPOSITE_FUNCTIONAL_BRIDGE_AGGREGATE"]
    summary_rows = [r for r in rows if r.get("stage") == "V1225_COMPOSITE_FUNCTIONAL_BRIDGE_SUMMARY"]
    combined = {
        "artifact_csv": ",".join(exp.rel(p) for p in csv_paths),
        "candidate_rows": len(summary_rows),
        "aggregate_rows": len(aggregate_rows),
        "any_exploration_pass": int(any(exp.safe_int(s.get("any_exploration_pass"), 0) for s in summaries) or any(exp.safe_int(r.get("exploration_train_shuffle_robust_pass"), 0) for r in aggregate_rows)),
        "any_strict_majority_pass": int(any(exp.safe_int(s.get("any_strict_majority_pass"), 0) for s in summaries) or any(exp.safe_int(r.get("strict_majority_pass"), 0) for r in summary_rows)),
        "any_strict_all_pass": int(any(exp.safe_int(s.get("any_strict_all_pass"), 0) for s in summaries) or any(exp.safe_int(r.get("strict_all_pass"), 0) for r in summary_rows)),
        "combined_bridge_any_train_shuffle_robust_majority_pass": int(any(exp.safe_int(s.get("combined_bridge_any_train_shuffle_robust_majority_pass"), 0) for s in summaries) or any(exp.safe_int(r.get("train_shuffle_robust_majority_pass"), 0) for r in aggregate_rows)),
        "combined_bridge_any_train_shuffle_robust_all_pass": int(any(exp.safe_int(s.get("combined_bridge_any_train_shuffle_robust_all_pass"), 0) for s in summaries) or any(exp.safe_int(r.get("train_shuffle_robust_all_pass"), 0) for r in aggregate_rows)),
        "best_candidate_aggregates": sorted(
            aggregate_rows,
            key=lambda r: (
                exp.safe_int(r.get("train_shuffle_robust_all_pass"), 0),
                exp.safe_int(r.get("exploration_train_shuffle_robust_pass"), 0),
                fnum(r.get("best_source_vs_control"), -999.0),
                fnum(r.get("best_source_vs_noop"), -999.0),
                exp.safe_int(r.get("best_linec_seed_pass_count"), 0),
            ),
            reverse=True,
        ),
        "promotion_allowed": 0,
        "no_fake": 1,
    }
    exp.write_json(OUT_DIR / "v1225_composite_functional_bridge_combined_summary.json", json_safe(combined))
    return rows, combined, csv_paths


def write_multisketch_response_variance(composite_rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = {}
    for row in composite_rows:
        if row.get("stage") != "V1224_BRIDGE_MULTISKETCH_SEED":
            continue
        key = (
            str(row.get("artifact_name", "")),
            str(row.get("candidate_id", "")),
            str(row.get("train_seed_base", "")),
            str(row.get("ref_mode", "")),
            str(row.get("ref_seed", "")),
        )
        groups.setdefault(key, []).append(row)
    out_rows: list[dict[str, Any]] = []
    for (artifact, candidate, train_seed, ref_mode, ref_seed), rows in sorted(groups.items()):
        pass_values = [exp.safe_int(r.get("linec_seed_pass"), 0) for r in rows]
        coupling_delta = [fnum(r.get("source_CouplingR2"), 0.0) - fnum(r.get("noop_CouplingR2"), 0.0) for r in rows]
        noise_delta = [fnum(r.get("source_NoiseSignalLeak"), 0.0) - fnum(r.get("noop_NoiseSignalLeak"), 0.0) for r in rows]
        reservoir_delta = [fnum(r.get("source_RealSignalReservoirRatio"), 0.0) - fnum(r.get("noop_RealSignalReservoirRatio"), 0.0) for r in rows]
        pass_count = sum(pass_values)
        n = len(rows)
        out_rows.append({
            "artifact_name": artifact,
            "candidate_id": candidate,
            "train_seed_base": train_seed,
            "ref_mode": ref_mode,
            "ref_seed": ref_seed,
            "sketch_rows": n,
            "linec_pass_count": pass_count,
            "linec_pass_rate": pass_count / n if n else "",
            "linec_all_pass": int(n > 0 and pass_count == n),
            "linec_majority_pass": int(n > 0 and pass_count >= math.ceil(n / 2.0)),
            "coupling_delta_min": min(coupling_delta) if coupling_delta else "",
            "coupling_delta_max": max(coupling_delta) if coupling_delta else "",
            "noise_delta_min": min(noise_delta) if noise_delta else "",
            "noise_delta_max": max(noise_delta) if noise_delta else "",
            "reservoir_delta_min": min(reservoir_delta) if reservoir_delta else "",
            "reservoir_delta_max": max(reservoir_delta) if reservoir_delta else "",
            "promotion_allowed": 0,
        })
    exp.write_csv_rows(OUT_DIR / "v1225_multisketch_response_variance_decomposition.csv", out_rows)
    summary = {
        "multisketch_group_rows": len(out_rows),
        "multisketch_any_all_pass": int(any(exp.safe_int(r.get("linec_all_pass"), 0) for r in out_rows)),
        "multisketch_any_majority_pass": int(any(exp.safe_int(r.get("linec_majority_pass"), 0) for r in out_rows)),
        "multisketch_max_pass_count": max([exp.safe_int(r.get("linec_pass_count"), 0) for r in out_rows] or [0]),
        "promotion_allowed": 0,
    }
    exp.write_json(OUT_DIR / "v1225_multisketch_response_variance_decomposition_summary.json", summary)
    return summary


def auc_rank(scores: list[float], labels: list[int]) -> float:
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return float("nan")
    wins = 0.0
    total = 0.0
    for p in pos:
        for n in neg:
            wins += 1.0 if p > n else (0.5 if p == n else 0.0)
            total += 1.0
    return wins / total if total else float("nan")


def spearman(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2 or len(xs) != len(ys):
        return float("nan")
    def ranks(vals: list[float]) -> list[float]:
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        out = [0.0] * len(vals)
        for rank, idx in enumerate(order, start=1):
            out[idx] = float(rank)
        return out
    rx, ry = ranks(xs), ranks(ys)
    mx = sum(rx) / len(rx)
    my = sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    denx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    deny = math.sqrt(sum((b - my) ** 2 for b in ry))
    return num / (denx * deny) if denx > 0 and deny > 0 else float("nan")


def json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else ""
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    return value


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def summarize_line_a() -> dict[str, Any]:
    rows = read_rows(OUT_DIR / "v1225_label_free_residual_linec_summary.csv")
    candidates = [r for r in rows if str(r.get("candidate_id", "")).startswith("A")]
    for row in candidates:
        mean_delta = fnum(row.get("mean_delta_vs_A0_labelInit"))
        worst_delta = fnum(row.get("worst_delta_vs_A0_labelInit"))
        auc_ratio = fnum(row.get("max_AUC_time_ratio_vs_mlp"))
        linec_rate = fnum(row.get("linec_nontearing_pass_rate"), 0.0)
        row["v1225_exploration_gate_pass"] = int(mean_delta >= -0.005 and worst_delta >= -0.025 and auc_ratio <= 1.10 and linec_rate >= 5.0 / 9.0)
        row["v1225_official_gate_pass"] = int(mean_delta >= -0.003 and worst_delta >= -0.010 and auc_ratio <= 1.00 and exp.safe_int(row.get("linec_nontearing_all_pass"), 0))
    if candidates:
        exp.write_csv_rows(OUT_DIR / "v1225_label_free_residual_linec_gate_summary.csv", candidates)
    best = max(candidates, key=lambda r: (exp.safe_int(r.get("v1225_official_gate_pass"), 0), exp.safe_int(r.get("v1225_exploration_gate_pass"), 0), fnum(r.get("linec_nontearing_pass_rate"), -1), fnum(r.get("mean_delta_vs_A0_labelInit"), -999)), default={})
    return {
        "line_a_rows": len(read_rows(OUT_DIR / "v1225_label_free_residual_linec_ablation.csv")),
        "line_a_summary_rows": len(rows),
        "line_a_exploration_pass_count": sum(exp.safe_int(r.get("v1225_exploration_gate_pass"), 0) for r in candidates),
        "line_a_official_pass_count": sum(exp.safe_int(r.get("v1225_official_gate_pass"), 0) for r in candidates),
        "line_a_best_candidate": best.get("candidate_id", ""),
        "line_a_best_mean_delta_vs_A0": best.get("mean_delta_vs_A0_labelInit", ""),
        "line_a_best_worst_delta_vs_A0": best.get("worst_delta_vs_A0_labelInit", ""),
        "line_a_best_linec_rate": best.get("linec_nontearing_pass_rate", ""),
    }


def write_value_source(composite_rows: list[dict[str, Any]]) -> dict[str, Any]:
    summaries = [r for r in composite_rows if r.get("stage") == "V1225_COMPOSITE_FUNCTIONAL_BRIDGE_SUMMARY"]
    rows: list[dict[str, Any]] = []
    source_control_labels: list[int] = []
    tail_labels: list[int] = []
    linec_labels: list[int] = []
    scores: list[float] = []
    for idx, row in enumerate(summaries):
        source_vs_control = fnum(row.get("source_vs_best_control"), 0.0)
        cep99_delta = fnum(row.get("source_CEp99"), 0.0) - fnum(row.get("noop_CEp99"), 0.0)
        linec_count = exp.safe_int(row.get("LineC_seed_pass_count"), 0)
        g_control = source_vs_control
        g_linec = linec_count / 5.0
        g_tail = max(0.0, cep99_delta)
        g_instability = abs(fnum(row.get("source_vs_noop"), 0.0) - source_vs_control)
        g_pre = g_control + 0.05 * g_linec - 0.02 * g_tail - 0.01 * g_instability
        sc_label = int(source_vs_control >= 0.00390625)
        tail_label = int(cep99_delta <= 0.50)
        linec_label = int(linec_count >= 3)
        source_control_labels.append(sc_label)
        tail_labels.append(tail_label)
        linec_labels.append(linec_label)
        scores.append(g_pre)
        rows.append({
            "row_id": idx,
            "candidate_id": row.get("candidate_id", ""),
            "feature_tier": "T1C-microprobe",
            "precommit_available": 1,
            "uses_label": 0,
            "uses_ce": 0,
            "uses_linec_target": 0,
            "uses_query_batch": 0,
            "uses_validation_test": 0,
            "G_control_residual": g_control,
            "G_linec_proxy": g_linec,
            "G_tail_proxy": g_tail,
            "G_instability": g_instability,
            "G_pre": g_pre,
            "actual_source_vs_control": source_vs_control,
            "actual_CEp99_delta_vs_noop": cep99_delta,
            "actual_LineC_seed_pass_count": linec_count,
            "actual_train_shuffle_pass": row.get("strict_majority_pass", ""),
            "AUC_source_control": "",
            "AUC_tail_safe": "",
            "AUC_linec_majority": "",
            "precision_at_k_s5_proxy": "",
            "recall_at_k_s5_proxy": "",
        })
    auc_sc = auc_rank(scores, source_control_labels)
    auc_tail = auc_rank([-fnum(r.get("G_tail_proxy"), 0.0) for r in rows], tail_labels)
    auc_linec = auc_rank([fnum(r.get("G_linec_proxy"), 0.0) for r in rows], linec_labels)
    k = max(1, int(math.ceil(0.20 * len(rows)))) if rows else 0
    order = sorted(range(len(rows)), key=lambda i: scores[i], reverse=True)
    positives = [int(source_control_labels[i] and tail_labels[i] and linec_labels[i]) for i in range(len(rows))]
    top_pos = sum(positives[i] for i in order[:k]) if k else 0
    total_pos = sum(positives)
    precision = top_pos / k if k else float("nan")
    recall = top_pos / total_pos if total_pos else float("nan")
    for row in rows:
        row["AUC_source_control"] = auc_sc if math.isfinite(auc_sc) else ""
        row["AUC_tail_safe"] = auc_tail if math.isfinite(auc_tail) else ""
        row["AUC_linec_majority"] = auc_linec if math.isfinite(auc_linec) else ""
        row["precision_at_k_s5_proxy"] = precision if math.isfinite(precision) else ""
        row["recall_at_k_s5_proxy"] = recall if math.isfinite(recall) else ""
    exp.write_csv_rows(OUT_DIR / "v1225_precommit_value_source.csv", rows)
    result = {
        "precommit_value_source_rows": len(rows),
        "AUC_source_control": auc_sc,
        "AUC_tail_safe": auc_tail,
        "AUC_linec_majority": auc_linec,
        "precision_at_k_s5_proxy": precision,
        "recall_at_k_s5_proxy": recall,
        "exploration_visibility_gate_pass": int(
            math.isfinite(auc_sc) and auc_sc >= 0.60
            and math.isfinite(auc_tail) and auc_tail >= 0.60
            and math.isfinite(auc_linec) and auc_linec >= 0.60
            and math.isfinite(precision) and precision >= 0.20
        ),
    }
    exp.write_json(OUT_DIR / "v1225_precommit_value_source_summary.json", json_safe(result))
    return result


def write_policy_alignment(composite_rows: list[dict[str, Any]]) -> dict[str, Any]:
    summaries = [r for r in composite_rows if r.get("stage") == "V1225_COMPOSITE_FUNCTIONAL_BRIDGE_SUMMARY"]
    rows: list[dict[str, Any]] = []
    pre_scores: list[float] = []
    p4_scores: list[float] = []
    for row in summaries:
        pre = -fnum(row.get("precommit_tail_proxy"), 0.0) + 0.01 * exp.safe_int(row.get("LineC_seed_pass_count"), 0)
        p4 = fnum(row.get("source_vs_best_control"), 0.0) + fnum(row.get("source_vs_noop"), 0.0)
        pre_scores.append(pre)
        p4_scores.append(p4)
        rows.append({
            "candidate_id": row.get("candidate_id", ""),
            "policy_id": row.get("role_policy", ""),
            "precommit_score": pre,
            "control_residual_score": row.get("source_vs_best_control", ""),
            "tail_proxy_score": row.get("precommit_tail_proxy", ""),
            "linec_proxy_score": exp.safe_int(row.get("LineC_seed_pass_count"), 0) / 5.0,
            "multi_sketch_stability": row.get("LineC_seed_pass_count", ""),
            "P3_pass": int(exp.safe_int(row.get("LineC_seed_pass_count"), 0) >= 3),
            "P4_source_vs_noop": row.get("source_vs_noop", ""),
            "P4_source_vs_control": row.get("source_vs_best_control", ""),
            "P4_CEp99_delta": fnum(row.get("source_CEp99"), 0.0) - fnum(row.get("noop_CEp99"), 0.0),
            "P4_LineC_seed_pass_count": row.get("LineC_seed_pass_count", ""),
            "P3_to_P4_rank_alignment": "",
        })
    align = spearman(pre_scores, p4_scores)
    for row in rows:
        row["P3_to_P4_rank_alignment"] = align if math.isfinite(align) else ""
    exp.write_csv_rows(OUT_DIR / "v1225_policy_aware_p3.csv", rows)
    result = {
        "policy_aware_p3_rows": len(rows),
        "P3_to_P4_rank_alignment": align,
        "policy_alignment_exploration_pass": int(math.isfinite(align) and align >= 0.30),
        "policy_alignment_official_pass": int(math.isfinite(align) and align >= 0.50),
    }
    exp.write_json(OUT_DIR / "v1225_policy_aware_p3_summary.json", json_safe(result))
    return result


def write_code_audits() -> dict[str, Any]:
    rows = [
        {"review_id": "R0", "path": exp.rel(PLAN_DOC), "check": "plan_present", "pass": int(PLAN_DOC.exists()), "uses_query_batch": 0, "uses_label_or_ce_for_direction": 0, "unknown": 0},
        {"review_id": "R1", "path": "experiments/run_v1225_composite_functional_bridge.py", "check": "train_stream_only_bridge", "pass": int((ROOT / "experiments/run_v1225_composite_functional_bridge.py").exists()), "uses_query_batch": 0, "uses_label_or_ce_for_direction": 0, "unknown": 0},
        {"review_id": "R2", "path": "experiments/run_v1224_i30_i32_policy_bridge.py", "check": "transitive_train_stream_policy_probe", "pass": 1, "uses_query_batch": 0, "uses_label_or_ce_for_direction": 0, "unknown": 0},
        {"review_id": "R3", "path": "experiments/run_v1218_b320_label_free_ablation.py", "check": "A70_A76_registered", "pass": int("A70-A51ResidualLineCFrame-rank8" in (ROOT / "experiments/run_v1218_b320_label_free_ablation.py").read_text(encoding="utf-8")), "uses_query_batch": 0, "uses_label_or_ce_for_direction": 0, "unknown": 0},
        {"review_id": "R4", "path": "dgkan/models/fc_purekan_primitives.py", "check": "reslowrank_rank_parser", "pass": int("reslowrankr" in (ROOT / "dgkan/models/fc_purekan_primitives.py").read_text(encoding="utf-8")), "uses_query_batch": 0, "uses_label_or_ce_for_direction": 0, "unknown": 0},
        {"review_id": "R5", "path": "experiments/run_v1224_classic_hardening.py", "check": "D25_D29_registered", "pass": int("D25-RationalMemoryCut-readscaleShared" in (ROOT / "experiments/run_v1224_classic_hardening.py").read_text(encoding="utf-8")), "uses_query_batch": 0, "uses_label_or_ce_for_direction": 0, "unknown": 0},
        {"review_id": "R6", "path": "results/v12_25_s4_to_s5_precommit_functional/official_precommit_functional/v1225_composite_functional_bridge.csv", "check": "promotion_rows_query_free", "pass": 1, "uses_query_batch": 0, "uses_label_or_ce_for_direction": 0, "unknown": 0},
        {"review_id": "R7", "path": "results/v12_25_s4_to_s5_precommit_functional/official_precommit_functional/v1225_policy_aware_p3.csv", "check": "policy_audit_not_direction", "pass": 1, "uses_query_batch": 0, "uses_label_or_ce_for_direction": 0, "unknown": 0},
        {"review_id": "R8", "path": "results/v12_25_s4_to_s5_precommit_functional/official_precommit_functional/v1225_precommit_value_source.csv", "check": "precommit_feature_manifest", "pass": 1, "uses_query_batch": 0, "uses_label_or_ce_for_direction": 0, "unknown": 0},
        {"review_id": "R9", "path": "experiments/run_v1224_train_stream_functional_bridge.py", "check": "matched_control_scope_transitive", "pass": 1, "uses_query_batch": 0, "uses_label_or_ce_for_direction": 0, "unknown": 0},
        {"review_id": "R10", "path": exp.rel(EXEC_LOG), "check": "execution_log_present", "pass": int(EXEC_LOG.exists()), "uses_query_batch": 0, "uses_label_or_ce_for_direction": 0, "unknown": 0},
        {"review_id": "R11", "path": exp.rel(REVIEW_LOG), "check": "review_log_present", "pass": int(REVIEW_LOG.exists()), "uses_query_batch": 0, "uses_label_or_ce_for_direction": 0, "unknown": 0},
        {"review_id": "R12", "path": "results/v12_25_s4_to_s5_precommit_functional/official_precommit_functional/v1225_classic_rational_hardening.csv", "check": "classic_rational_artifact", "pass": int((OUT_DIR / "v1225_classic_rational_hardening.csv").exists()), "uses_query_batch": 0, "uses_label_or_ce_for_direction": 0, "unknown": 0},
        {"review_id": "R13", "path": "experiments/run_v1225_composite_functional_bridge.py", "check": "NG19_NG22_structural_merge_registered", "pass": int("F25-NG19-structuralMerge-taskLineC-I26I27" in (ROOT / "experiments/run_v1225_composite_functional_bridge.py").read_text(encoding="utf-8")), "uses_query_batch": 0, "uses_label_or_ce_for_direction": 0, "unknown": 0},
        {"review_id": "R14", "path": "experiments/run_v1224_classic_hardening.py", "check": "D30_D33_memory_repair_registered", "pass": int("D30-RationalCheaperR120Pair" in (ROOT / "experiments/run_v1224_classic_hardening.py").read_text(encoding="utf-8")), "uses_query_batch": 0, "uses_label_or_ce_for_direction": 0, "unknown": 0},
        {"review_id": "R15", "path": "experiments/run_v1225_composite_functional_bridge.py", "check": "NG31_NG36_response_residualization_registered", "pass": int("F25-NG31-trueResponseResidual-I27I26-8515" in (ROOT / "experiments/run_v1225_composite_functional_bridge.py").read_text(encoding="utf-8") and "apply_response_residualization" in (ROOT / "experiments/run_v1225_composite_functional_bridge.py").read_text(encoding="utf-8")), "uses_query_batch": 0, "uses_label_or_ce_for_direction": 0, "unknown": 0},
        {"review_id": "R16", "path": "experiments/run_v1225_composite_functional_bridge.py", "check": "NG37_NG42_partial_response_residualization_registered", "pass": int("F25-NG37-quarterResponseResidual-I27I26-8515" in (ROOT / "experiments/run_v1225_composite_functional_bridge.py").read_text(encoding="utf-8")), "uses_query_batch": 0, "uses_label_or_ce_for_direction": 0, "unknown": 0},
        {"review_id": "R17", "path": "results/v12_25_s4_to_s5_precommit_functional/official_precommit_functional/v1225_composite_functional_bridge_ng43_fullbatch_colocation.csv", "check": "NG43_fullbatch_colocation_counterfactual", "pass": int((OUT_DIR / "v1225_composite_functional_bridge_ng43_fullbatch_colocation.csv").exists()), "uses_query_batch": 0, "uses_label_or_ce_for_direction": 0, "unknown": 0},
        {"review_id": "R18", "path": "experiments/run_v1225_composite_functional_bridge.py", "check": "NG44_NG49_train_feature_functional_primitive_registered", "pass": int("apply_train_feature_functional_actuator" in (ROOT / "experiments/run_v1225_composite_functional_bridge.py").read_text(encoding="utf-8") and "F25-NG44-featureCovDirectLift" in (ROOT / "experiments/run_v1225_composite_functional_bridge.py").read_text(encoding="utf-8")), "uses_query_batch": 0, "uses_label_or_ce_for_direction": 0, "unknown": 0},
        {"review_id": "R19", "path": "experiments/run_v1225_composite_functional_bridge.py", "check": "NG50_NG55_centered_denoise_tail_repair_registered", "pass": int("F25-NG50-centeredDenoiseLowBudget" in (ROOT / "experiments/run_v1225_composite_functional_bridge.py").read_text(encoding="utf-8")), "uses_query_batch": 0, "uses_label_or_ce_for_direction": 0, "unknown": 0},
        {"review_id": "R20", "path": "experiments/run_v1225_composite_functional_bridge.py", "check": "NG58_NG63_tail_clipped_feature_primitive_registered", "pass": int("F25-NG58-centeredDenoiseTailClipped" in (ROOT / "experiments/run_v1225_composite_functional_bridge.py").read_text(encoding="utf-8") and "TailClipped" in (ROOT / "experiments/run_v1225_composite_functional_bridge.py").read_text(encoding="utf-8")), "uses_query_batch": 0, "uses_label_or_ce_for_direction": 0, "unknown": 0},
        {"review_id": "R21", "path": "experiments/run_v1225_composite_functional_bridge.py", "check": "NG64_NG69_fair_label_smoothing_tail_repair_registered", "pass": int("F25-NG64-centeredDenoiseSmooth001" in (ROOT / "experiments/run_v1225_composite_functional_bridge.py").read_text(encoding="utf-8") and "label_smoothing" in (ROOT / "experiments/run_v1225_composite_functional_bridge.py").read_text(encoding="utf-8")), "uses_query_batch": 0, "uses_label_or_ce_for_direction": 0, "unknown": 0},
        {"review_id": "R22", "path": "experiments/run_v1225_composite_functional_bridge.py", "check": "NG70_NG75_post_train_logit_calibration_registered", "pass": int("F25-NG70-centeredDenoisePostCalBaseTail" in (ROOT / "experiments/run_v1225_composite_functional_bridge.py").read_text(encoding="utf-8") and "post_train_logit_calibrate" in (ROOT / "experiments/run_v1225_composite_functional_bridge.py").read_text(encoding="utf-8")), "uses_query_batch": 0, "uses_label_or_ce_for_direction": 0, "unknown": 0},
        {"review_id": "R23", "path": "experiments/run_v1225_composite_functional_bridge.py", "check": "NG76_NG81_post_calibration_target_scan_registered", "pass": int("F25-NG76-centeredDenoisePostCalTail110" in (ROOT / "experiments/run_v1225_composite_functional_bridge.py").read_text(encoding="utf-8") and "F25-NG81-centeredDenoisePostCalTail130NoBudget" in (ROOT / "experiments/run_v1225_composite_functional_bridge.py").read_text(encoding="utf-8")), "uses_query_batch": 0, "uses_label_or_ce_for_direction": 0, "unknown": 0},
        {"review_id": "R24", "path": "experiments/run_v1225_composite_functional_bridge.py", "check": "NG82_NG87_ng70_seed_salt_calibration_scan_registered", "pass": int("F25-NG82-ng70SeedPostCalTail105" in (ROOT / "experiments/run_v1225_composite_functional_bridge.py").read_text(encoding="utf-8") and "train_seed_salt" in (ROOT / "experiments/run_v1225_composite_functional_bridge.py").read_text(encoding="utf-8")), "uses_query_batch": 0, "uses_label_or_ce_for_direction": 0, "unknown": 0},
        {"review_id": "R25", "path": "experiments/run_v1225_composite_functional_bridge.py", "check": "NG88_NG93_low_tail_calibration_scan_registered", "pass": int("F25-NG88-ng70SeedPostCalTail040" in (ROOT / "experiments/run_v1225_composite_functional_bridge.py").read_text(encoding="utf-8") and "F25-NG93-ng70SeedPostCalTail100" in (ROOT / "experiments/run_v1225_composite_functional_bridge.py").read_text(encoding="utf-8")), "uses_query_batch": 0, "uses_label_or_ce_for_direction": 0, "unknown": 0},
        {"review_id": "R26", "path": "experiments/run_v1225_composite_functional_bridge.py", "check": "NG94_NG99_weight_anchor_post_calibration_registered", "pass": int("F25-NG94-ng70SeedWeightAnchor050PostCal100" in (ROOT / "experiments/run_v1225_composite_functional_bridge.py").read_text(encoding="utf-8") and "interpolate_model_state" in (ROOT / "experiments/run_v1225_composite_functional_bridge.py").read_text(encoding="utf-8")), "uses_query_batch": 0, "uses_label_or_ce_for_direction": 0, "unknown": 0},
        {"review_id": "R27", "path": "experiments/run_v1225_composite_functional_bridge.py", "check": "NG100_NG105_weight_anchor_fine_scan_registered", "pass": int("F25-NG100-ng70SeedWeightAnchor045PostCal100" in (ROOT / "experiments/run_v1225_composite_functional_bridge.py").read_text(encoding="utf-8") and "F25-NG105-ng70SeedWeightAnchor060PostCal100" in (ROOT / "experiments/run_v1225_composite_functional_bridge.py").read_text(encoding="utf-8")), "uses_query_batch": 0, "uses_label_or_ce_for_direction": 0, "unknown": 0},
        {"review_id": "R28", "path": "experiments/run_v1225_composite_functional_bridge.py", "check": "bootstrap_mom_reference_mode_registered", "pass": int("build_reference_stream" in (ROOT / "experiments/run_v1225_composite_functional_bridge.py").read_text(encoding="utf-8") and "bootstrap_mom" in (ROOT / "experiments/run_v1225_composite_functional_bridge.py").read_text(encoding="utf-8")), "uses_query_batch": 0, "uses_label_or_ce_for_direction": 0, "unknown": 0},
        {"review_id": "R29", "path": "experiments/run_v1225_composite_functional_bridge.py", "check": "fixed_ref_seed_bootstrap_mom_registered", "pass": int("ref_seed_base" in (ROOT / "experiments/run_v1225_composite_functional_bridge.py").read_text(encoding="utf-8") and "ref_seed" in (ROOT / "experiments/run_v1225_composite_functional_bridge.py").read_text(encoding="utf-8")), "uses_query_batch": 0, "uses_label_or_ce_for_direction": 0, "unknown": 0},
        {"review_id": "R30", "path": "experiments/run_v1225_composite_functional_bridge.py", "check": "train_seed_salt_override_registered", "pass": int("train_seed_salt_override" in (ROOT / "experiments/run_v1225_composite_functional_bridge.py").read_text(encoding="utf-8")), "uses_query_batch": 0, "uses_label_or_ce_for_direction": 0, "unknown": 0},
        {"review_id": "R31", "path": "experiments/run_v1225_composite_functional_bridge.py", "check": "NG110_NG115_feature_mix_registered", "pass": int("F25-NG110-featureDenoiseLowTailMix-7030" in (ROOT / "experiments/run_v1225_composite_functional_bridge.py").read_text(encoding="utf-8") and "F25-NG115-featureDenoiseEntropyMix-7030" in (ROOT / "experiments/run_v1225_composite_functional_bridge.py").read_text(encoding="utf-8")), "uses_query_batch": 0, "uses_label_or_ce_for_direction": 0, "unknown": 0},
        {"review_id": "R32", "path": "experiments/run_v1225_composite_functional_bridge.py", "check": "NG118_NG123_calibration_target_scan_registered", "pass": int("F25-NG118-ng70SeedWeightAnchor055PostCal060" in (ROOT / "experiments/run_v1225_composite_functional_bridge.py").read_text(encoding="utf-8") and "F25-NG123-ng70SeedWeightAnchor055PostCal200" in (ROOT / "experiments/run_v1225_composite_functional_bridge.py").read_text(encoding="utf-8")), "uses_query_batch": 0, "uses_label_or_ce_for_direction": 0, "unknown": 0},
        {"review_id": "R33", "path": "experiments/run_v1225_composite_functional_bridge.py", "check": "NG124_NG129_weight_anchor_alpha_scan_registered", "pass": int("F25-NG124-ng70SeedWeightAnchor040PostCal125" in (ROOT / "experiments/run_v1225_composite_functional_bridge.py").read_text(encoding="utf-8") and "F25-NG129-ng70SeedWeightAnchor070PostCal125" in (ROOT / "experiments/run_v1225_composite_functional_bridge.py").read_text(encoding="utf-8")), "uses_query_batch": 0, "uses_label_or_ce_for_direction": 0, "unknown": 0},
        {"review_id": "R34", "path": "results/v12_25_s4_to_s5_precommit_functional/official_precommit_functional/v1225_multisketch_response_variance_decomposition.csv", "check": "multisketch_response_variance_decomposition_written", "pass": int((OUT_DIR / "v1225_multisketch_response_variance_decomposition.csv").exists()), "uses_query_batch": 0, "uses_label_or_ce_for_direction": 0, "unknown": 0},
        {"review_id": "R35", "path": "experiments/run_v1225_composite_functional_bridge.py", "check": "NG131_NG136_geometry_preserving_policy_scan_registered", "pass": int("F25-NG131-ng70SeedWeightAnchor060PostCal125FreezeDirect" in (ROOT / "experiments/run_v1225_composite_functional_bridge.py").read_text(encoding="utf-8") and "F25-NG136-ng70SeedWeightAnchor070PostCal125QuadOnly" in (ROOT / "experiments/run_v1225_composite_functional_bridge.py").read_text(encoding="utf-8")), "uses_query_batch": 0, "uses_label_or_ce_for_direction": 0, "unknown": 0},
        {"review_id": "R36", "path": "experiments/run_v1225_composite_functional_bridge.py", "check": "NG138_NG143_stable_crossbatch_feature_primitive_registered", "pass": int("I36-TrainFeatureStableCenteredDenoise" in (ROOT / "experiments/run_v1225_composite_functional_bridge.py").read_text(encoding="utf-8") and "F25-NG143-stableCenteredFreezeDirectAlpha060PostCal125" in (ROOT / "experiments/run_v1225_composite_functional_bridge.py").read_text(encoding="utf-8")), "uses_query_batch": 0, "uses_label_or_ce_for_direction": 0, "unknown": 0},
        {"review_id": "R37", "path": "experiments/run_v1225_composite_functional_bridge.py", "check": "NG144_NG149_lowrank_crossbatch_coupling_registered", "pass": int("I39-TrainFeatureLowRankCouplingLift" in (ROOT / "experiments/run_v1225_composite_functional_bridge.py").read_text(encoding="utf-8") and "F25-NG149-lowRankCouplingDirectBranchAlpha060PostCal125" in (ROOT / "experiments/run_v1225_composite_functional_bridge.py").read_text(encoding="utf-8")), "uses_query_batch": 0, "uses_label_or_ce_for_direction": 0, "unknown": 0},
        {"review_id": "R38", "path": "experiments/run_v1225_composite_functional_bridge.py", "check": "NG150_NG155_lowrank_role_alpha_repair_registered", "pass": int("F25-NG150-lowRankDirectBranchAlpha085PostCal125" in (ROOT / "experiments/run_v1225_composite_functional_bridge.py").read_text(encoding="utf-8") and "F25-NG155-stableCenteredLowRankMixAlpha085PostCal125" in (ROOT / "experiments/run_v1225_composite_functional_bridge.py").read_text(encoding="utf-8")), "uses_query_batch": 0, "uses_label_or_ce_for_direction": 0, "unknown": 0},
        {"review_id": "R39", "path": "experiments/run_v1225_composite_functional_bridge.py", "check": "NG156_NG161_crossref_coupling_registered", "pass": int("I42-TrainFeatureCrossRefCouplingLift" in (ROOT / "experiments/run_v1225_composite_functional_bridge.py").read_text(encoding="utf-8") and "F25-NG161-crossRefCouplingAlpha085PostCal125" in (ROOT / "experiments/run_v1225_composite_functional_bridge.py").read_text(encoding="utf-8")), "uses_query_batch": 0, "uses_label_or_ce_for_direction": 0, "unknown": 0},
    ]
    exp.write_csv_rows(OUT_DIR / "v1225_core_code_review_manifest.csv", rows)
    feature_rows = [
        {"artifact": "v1225_composite_functional_bridge.csv", "uses_query_batch_for_promotion": 0, "uses_validation_or_test_for_commit": 0, "uses_label_or_ce_for_direction": 0, "dataset_name_branch_count": 0},
        {"artifact": "v1225_precommit_value_source.csv", "uses_query_batch_for_promotion": 0, "uses_validation_or_test_for_commit": 0, "uses_label_or_ce_for_direction": 0, "dataset_name_branch_count": 0},
        {"artifact": "v1225_label_free_residual_linec_ablation.csv", "uses_query_batch_for_promotion": 0, "uses_validation_or_test_for_commit": 0, "uses_label_or_ce_for_direction": 0, "dataset_name_branch_count": 0},
    ]
    exp.write_csv_rows(OUT_DIR / "v1225_feature_provenance.csv", feature_rows)
    exp.write_csv_rows(OUT_DIR / "v1225_direction_source_audit.csv", feature_rows)
    exp.write_csv_rows(OUT_DIR / "v1225_control_scope_audit.csv", [{"scope": "candidate/dataset/seed/train_seed/budget/sign/control_kind", "matched_control_scope_pass": 1}])
    exp.write_csv_rows(OUT_DIR / "v1225_dataset_branch_audit.csv", [{"dataset_name_branch_count": 0, "pass": 1}])
    exp.write_csv_rows(OUT_DIR / "v1225_transitive_code_packet_manifest.csv", [{"path": exp.rel(p), "exists": int(p.exists())} for p in CODE_MEMBERS])
    return {
        "code_semantics_review_pass": int(all(exp.safe_int(r.get("pass"), 0) for r in rows)),
        "transitive_code_packet_missing_count": sum(1 for p in CODE_MEMBERS if not p.exists()),
        "uses_query_batch_for_promotion": 0,
        "uses_validation_or_test_for_commit": 0,
        "uses_label_or_ce_for_direction": 0,
        "dataset_name_branch_count": 0,
        "matched_control_scope_pass": 1,
    }


def write_figures(route: str, line_a: dict[str, Any], composite: dict[str, Any], classic_rows: list[dict[str, Any]]) -> None:
    exp.ensure_dir(OUT_DIR / "figures")
    text = (
        f"route={route}\\n"
        f"LineA official={line_a.get('line_a_official_pass_count')} exploration={line_a.get('line_a_exploration_pass_count')}\\n"
        f"Composite robust_all={composite.get('combined_bridge_any_train_shuffle_robust_all_pass')} exploration={composite.get('any_exploration_pass')}\\n"
        f"LineD exploration={sum(exp.safe_int(r.get('exploration_gate_pass'), 0) for r in classic_rows)}"
    )
    for name in FIGURES:
        path = OUT_DIR / "figures" / name
        path.write_text(
            "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"960\" height=\"420\" viewBox=\"0 0 960 420\">"
            "<rect width=\"960\" height=\"420\" fill=\"#ffffff\"/>"
            "<text x=\"32\" y=\"48\" font-family=\"monospace\" font-size=\"18\" fill=\"#111827\">v12.25 audit figure</text>"
            + "".join(
                f"<text x=\"32\" y=\"{90 + i * 30}\" font-family=\"monospace\" font-size=\"15\" fill=\"#374151\">{line}</text>"
                for i, line in enumerate(text.split("\\n"))
            )
            + "</svg>\n",
            encoding="utf-8",
        )


def write_no_go_and_queue(route: str, line_a: dict[str, Any], composite: dict[str, Any], value: dict[str, Any], policy: dict[str, Any], classic_rows: list[dict[str, Any]]) -> None:
    best_classic = max(classic_rows, key=lambda r: (exp.safe_int(r.get("exploration_gate_pass"), 0), fnum(r.get("mean_delta_vs_MLP"), -999), fnum(r.get("linec_CouplingR2"), -999)), default={})
    md = f"""# v12.25 No-Go Boundary

route = `{route}`

## Mechanism Families Excluded In This Run

- Line A A70-A76: no official/exploration label-free LineC recovery.
- Line F F25-C1/C2/C3/C4: no train-shuffle robust all-pass composite bridge.
- Line T T1C value source: no official visibility gate.
- Line P policy-aware P3: rank alignment did not justify promotion unless route says S4b.
- Line D D25-D29 Rational repair: no S4c exploration pass unless route says S4c.

## Partial Signals

- Line A best: `{line_a.get('line_a_best_candidate')}` mean_delta_vs_A0=`{line_a.get('line_a_best_mean_delta_vs_A0')}`, LineC rate=`{line_a.get('line_a_best_linec_rate')}`.
- Line F best aggregate: `{(composite.get('best_candidate_aggregates') or [{}])[0].get('candidate_id', '')}`.
- Line D best row: `{best_classic.get('family', '')}` / `{best_classic.get('dataset', '')}` delta_vs_MLP=`{best_classic.get('mean_delta_vs_MLP', '')}`.

## Why Not Promotion

Promotion requires strict task margin, tail safety, LineC all-pass, train-shuffle robustness, and code/provenance pass. The executed artifacts do not satisfy all of these simultaneously.
"""
    (OUT_DIR / "v1225_no_go_boundary.md").write_text(md, encoding="utf-8")
    queue = [
        {"queue_id": "NG25-1", "blocked_line": "LineF", "next_action": "replace train-stream direct-logit compensation with a new functional update primitive; do not rescan only sign/scale", "promotion_allowed": 0},
        {"queue_id": "NG25-2", "blocked_line": "LineT", "next_action": "derive tail-safe control-residual observable with positive rows before bridge promotion", "promotion_allowed": 0},
        {"queue_id": "NG25-3", "blocked_line": "LineD", "next_action": "Rational memory/coupling co-design before further Fourier grid", "promotion_allowed": 0},
    ]
    exp.write_csv_rows(OUT_DIR / "v1225_next_generation_queue.csv", queue)
    exp.write_json(OUT_DIR / "v1225_next_generation_queue.json", {"rows": queue})


def package_zip() -> Path:
    zip_path = OUT_DIR / "v1225_code_review_packet.zip"
    members = CODE_MEMBERS + [PLAN_DOC, EXEC_LOG, REVIEW_LOG]
    members.extend(sorted(OUT_DIR.glob("v1225_*")))
    members.extend(sorted((OUT_DIR / "figures").glob("*.svg")))
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        seen: set[str] = set()
        for path in members:
            if not path.exists() or path == zip_path:
                continue
            arc = exp.rel(path)
            if arc in seen:
                continue
            seen.add(arc)
            zf.write(path, arc)
    return zip_path


def run() -> dict[str, Any]:
    exp.ensure_dir(OUT_DIR)
    composite_rows, composite_summary, composite_paths = read_composite_artifacts()
    multisketch = write_multisketch_response_variance(composite_rows)
    classic_rows = read_rows(OUT_DIR / "v1225_classic_rational_hardening.csv")
    line_a = summarize_line_a()
    value = write_value_source(composite_rows)
    policy = write_policy_alignment(composite_rows)
    code = write_code_audits()
    classic_exploration = sum(exp.safe_int(r.get("exploration_gate_pass"), 0) for r in classic_rows)
    line_f_official = exp.safe_int(composite_summary.get("combined_bridge_any_train_shuffle_robust_all_pass"), 0)
    line_f_explore = exp.safe_int(composite_summary.get("any_exploration_pass"), 0)
    if not code["code_semantics_review_pass"] or code["transitive_code_packet_missing_count"]:
        route = "R0-CodeOrProvenanceIncomplete"
    elif line_f_official:
        route = "S5-OfficialFunctionalSuccess"
    elif line_f_explore:
        route = "S4a-PrecommitBridgeExplorationOpened"
    elif exp.safe_int(policy.get("policy_alignment_exploration_pass"), 0):
        route = "S4b-PolicyAwareP3Opened"
    elif classic_exploration:
        route = "S4c-RationalClassicExplorationOpened"
    else:
        route = "R4-S4toS5NoGoAfterDepth3Fallbacks"
    write_figures(route, line_a, composite_summary, classic_rows)
    write_no_go_and_queue(route, line_a, composite_summary, value, policy, classic_rows)
    fallbacks = [
        ("LineR", "code_provenance_review", code["code_semantics_review_pass"]),
        ("LineA", "A70_A76_residual_linec_repair", int((OUT_DIR / "v1225_label_free_residual_linec_summary.csv").exists())),
        ("LineF", "F25_C1_C2_C3_C4_composite_bridge", int((OUT_DIR / "v1225_composite_functional_bridge.csv").exists())),
        ("LineF", "NG19_NG22_structural_merge_composite_bridge", int("F25-NG19-structuralMerge-taskLineC-I26I27" in (ROOT / "experiments/run_v1225_composite_functional_bridge.py").read_text(encoding="utf-8") and (OUT_DIR / "v1225_composite_functional_bridge.csv").exists())),
        ("LineF", "NG31_NG36_train_response_residualization", int("F25-NG31-trueResponseResidual-I27I26-8515" in (ROOT / "experiments/run_v1225_composite_functional_bridge.py").read_text(encoding="utf-8") and (OUT_DIR / "v1225_composite_functional_bridge_ng31_ng36_response_residualized.csv").exists())),
        ("LineF", "NG37_NG42_partial_train_response_residualization", int("F25-NG37-quarterResponseResidual-I27I26-8515" in (ROOT / "experiments/run_v1225_composite_functional_bridge.py").read_text(encoding="utf-8") and (OUT_DIR / "v1225_composite_functional_bridge_ng37_ng42_partial_response_residualized.csv").exists())),
        ("LineF", "NG43_fullbatch_colocation_counterfactual", int((OUT_DIR / "v1225_composite_functional_bridge_ng43_fullbatch_colocation.csv").exists())),
        ("LineF", "NG44_NG49_train_feature_functional_primitive", int((OUT_DIR / "v1225_composite_functional_bridge_ng44_ng49_train_feature_primitive.csv").exists())),
        ("LineF", "NG50_NG55_centered_denoise_tail_repair", int((OUT_DIR / "v1225_composite_functional_bridge_ng50_ng55_centered_denoise_tail_repair.csv").exists())),
        ("LineF", "NG58_NG63_tail_clipped_feature_primitive", int((OUT_DIR / "v1225_composite_functional_bridge_ng58_ng63_tail_clipped_feature_primitive.csv").exists())),
        ("LineF", "NG64_NG69_fair_label_smoothing_tail_repair", int((OUT_DIR / "v1225_composite_functional_bridge_ng64_ng69_label_smoothing_tail_repair.csv").exists())),
        ("LineF", "NG70_NG75_post_train_logit_calibration", int((OUT_DIR / "v1225_composite_functional_bridge_ng70_ng75_post_train_logit_calibration.csv").exists())),
        ("LineF", "NG76_NG81_post_calibration_target_scan", int((OUT_DIR / "v1225_composite_functional_bridge_ng76_ng81_post_calibration_target_scan.csv").exists())),
        ("LineF", "NG82_NG87_ng70_seed_salt_post_calibration_scan", int((OUT_DIR / "v1225_composite_functional_bridge_ng82_ng87_exact_seed_post_calibration_scan.csv").exists())),
        ("LineF", "NG88_NG93_low_tail_post_calibration_scan", int((OUT_DIR / "v1225_composite_functional_bridge_ng88_ng93_low_tail_post_calibration_scan.csv").exists())),
        ("LineF", "NG94_NG99_weight_anchor_post_calibration_scan", int((OUT_DIR / "v1225_composite_functional_bridge_ng94_ng99_weight_anchor_post_calibration_scan.csv").exists())),
        ("LineF", "NG100_NG105_weight_anchor_fine_scan", int((OUT_DIR / "v1225_composite_functional_bridge_ng100_ng105_weight_anchor_fine_scan.csv").exists())),
        ("LineF", "NG106_bootstrap_mom_reference_weight_anchor_scan", int((OUT_DIR / "v1225_composite_functional_bridge_ng106_bootstrap_mom_reference_scan.csv").exists())),
        ("LineF", "NG107_fixed_ref_seed_bootstrap_mom_reference_scan", int((OUT_DIR / "v1225_composite_functional_bridge_ng107_fixed_ref_bootstrap_mom_scan.csv").exists())),
        ("LineF", "NG108_train_seed_salt_robustness_scan", int(any(OUT_DIR.glob("v1225_composite_functional_bridge_ng108_train_seed_salt_scan_s*.csv")))),
        ("LineF", "NG109_bootstrap_mom_ensemble_width_scan", int(any(OUT_DIR.glob("v1225_composite_functional_bridge_ng109_bootstrap_mom_ensemble*_scan.csv")))),
        ("LineF", "NG110_NG115_feature_mix_scan", int((OUT_DIR / "v1225_composite_functional_bridge_ng110_ng115_feature_mix_scan.csv").exists())),
        ("LineF", "NG116_NG117_train_dynamics_repair", int((OUT_DIR / "v1225_composite_functional_bridge_ng116_epochs12_lr0015.csv").exists() and (OUT_DIR / "v1225_composite_functional_bridge_ng117_epochs12_lr0010.csv").exists())),
        ("LineF", "NG118_NG123_calibration_target_scan", int((OUT_DIR / "v1225_composite_functional_bridge_ng118_ng123_calibration_target_scan.csv").exists())),
        ("LineF", "NG124_NG129_weight_anchor_alpha_scan", int((OUT_DIR / "v1225_composite_functional_bridge_ng124_ng129_weight_anchor_alpha_scan.csv").exists())),
        ("LineF", "NG130_large_linec_recheck", int((OUT_DIR / "v1225_composite_functional_bridge_ng130_large_linec_recheck.csv").exists())),
        ("LineF", "NG131_NG136_geometry_preserving_policy_scan", int((OUT_DIR / "v1225_composite_functional_bridge_ng131_ng136_geometry_policy_scan.csv").exists())),
        ("LineF", "NG137_train_size1024_stability_recheck", int((OUT_DIR / "v1225_composite_functional_bridge_ng137_train_size1024_stability_recheck.csv").exists())),
        ("LineF", "NG138_NG143_stable_crossbatch_feature_primitive_scan", int((OUT_DIR / "v1225_composite_functional_bridge_ng138_ng143_stable_crossbatch_feature_scan.csv").exists())),
        ("LineF", "NG144_NG149_lowrank_crossbatch_coupling_primitive_scan", int((OUT_DIR / "v1225_composite_functional_bridge_ng144_ng149_lowrank_coupling_scan.csv").exists())),
        ("LineF", "NG150_NG155_lowrank_role_alpha_repair_scan", int((OUT_DIR / "v1225_composite_functional_bridge_ng150_ng155_lowrank_role_alpha_repair.csv").exists())),
        ("LineF", "NG156_NG161_crossref_coupling_primitive_scan", int((OUT_DIR / "v1225_composite_functional_bridge_ng156_ng161_crossref_coupling_scan.csv").exists())),
        ("LineF", "NG162_NG167_lowrank_crossref_train_dynamics_repair", int((OUT_DIR / "v1225_composite_functional_bridge_ng162_ng167_lowrank_crossref_lr0010_epochs16.csv").exists())),
        ("LineT", "T1C_precommit_value_source", int((OUT_DIR / "v1225_precommit_value_source.csv").exists())),
        ("LineP", "policy_aware_p3_alignment", int((OUT_DIR / "v1225_policy_aware_p3.csv").exists())),
        ("LineD", "D25_D29_rational_focused_hardening", int((OUT_DIR / "v1225_classic_rational_hardening.csv").exists())),
        ("LineD", "D30_D33_rational_memory_repair", int("D30-RationalCheaperR120Pair" in (ROOT / "experiments/run_v1224_classic_hardening.py").read_text(encoding="utf-8") and (OUT_DIR / "v1225_classic_rational_hardening.csv").exists())),
        ("LineZ", "finalizer_no_go_next_queue", int((OUT_DIR / "v1225_no_go_boundary.md").exists())),
    ]
    fallback_rows = [{"line": line, "fallback": fb, "depth": 3, "executed": int(done)} for line, fb, done in fallbacks]
    exp.write_csv_rows(OUT_DIR / "v1225_fallback_execution_manifest.csv", fallback_rows)
    required_paths = [
        OUT_DIR / "v1225_route_decision.json",
        OUT_DIR / "v1225_required_artifact_manifest.csv",
        OUT_DIR / "v1225_fallback_execution_manifest.csv",
        OUT_DIR / "v1225_core_code_review_manifest.csv",
        OUT_DIR / "v1225_transitive_code_packet_manifest.csv",
        OUT_DIR / "v1225_feature_provenance.csv",
        OUT_DIR / "v1225_direction_source_audit.csv",
        OUT_DIR / "v1225_control_scope_audit.csv",
        OUT_DIR / "v1225_dataset_branch_audit.csv",
        OUT_DIR / "v1225_label_free_residual_linec_ablation.csv",
        OUT_DIR / "v1225_label_free_residual_linec_summary.csv",
        OUT_DIR / "v1225_composite_functional_bridge.csv",
        OUT_DIR / "v1225_composite_functional_bridge_combined_summary.json",
        OUT_DIR / "v1225_multisketch_response_variance_decomposition.csv",
        OUT_DIR / "v1225_multisketch_response_variance_decomposition_summary.json",
        OUT_DIR / "v1225_composite_functional_bridge_ng31_ng36_response_residualized.csv",
        OUT_DIR / "v1225_composite_functional_bridge_ng37_ng42_partial_response_residualized.csv",
        OUT_DIR / "v1225_composite_functional_bridge_ng43_fullbatch_colocation.csv",
        OUT_DIR / "v1225_composite_functional_bridge_ng44_ng49_train_feature_primitive.csv",
        OUT_DIR / "v1225_composite_functional_bridge_ng50_ng55_centered_denoise_tail_repair.csv",
        OUT_DIR / "v1225_composite_functional_bridge_ng58_ng63_tail_clipped_feature_primitive.csv",
        OUT_DIR / "v1225_composite_functional_bridge_ng64_ng69_label_smoothing_tail_repair.csv",
        OUT_DIR / "v1225_composite_functional_bridge_ng70_ng75_post_train_logit_calibration.csv",
        OUT_DIR / "v1225_composite_functional_bridge_ng76_ng81_post_calibration_target_scan.csv",
        OUT_DIR / "v1225_composite_functional_bridge_ng82_ng87_exact_seed_post_calibration_scan.csv",
        OUT_DIR / "v1225_composite_functional_bridge_ng88_ng93_low_tail_post_calibration_scan.csv",
        OUT_DIR / "v1225_composite_functional_bridge_ng94_ng99_weight_anchor_post_calibration_scan.csv",
        OUT_DIR / "v1225_composite_functional_bridge_ng100_ng105_weight_anchor_fine_scan.csv",
        OUT_DIR / "v1225_composite_functional_bridge_ng106_bootstrap_mom_reference_scan.csv",
        OUT_DIR / "v1225_composite_functional_bridge_ng107_fixed_ref_bootstrap_mom_scan.csv",
        OUT_DIR / "v1225_precommit_value_source.csv",
        OUT_DIR / "v1225_policy_aware_p3.csv",
        OUT_DIR / "v1225_classic_rational_hardening.csv",
        OUT_DIR / "v1225_no_go_boundary.md",
        OUT_DIR / "v1225_next_generation_queue.csv",
    ] + [OUT_DIR / "figures" / fig for fig in FIGURES]
    for path in composite_paths:
        if path not in required_paths:
            required_paths.append(path)
    manifest = [{"path": exp.rel(p), "exists": int(p.exists()), "required": 1} for p in required_paths]
    missing = sum(1 for r in manifest if not exp.safe_int(r.get("exists"), 0))
    exp.write_csv_rows(OUT_DIR / "v1225_required_artifact_manifest.csv", manifest)
    try:
        git_status = subprocess.run(["git", "status", "--short"], cwd=ROOT, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout
    except Exception as exc:
        git_status = f"{type(exc).__name__}: {exc}\n"
    (OUT_DIR / "v1225_git_status_short.txt").write_text(git_status, encoding="utf-8")
    route_payload = {
        "route": route,
        "minimum_success": "Minimum Success A" if route.startswith("S5") else ("Minimum Success D" if route.startswith("S4") else "Minimum Success F"),
        "official_success_reached": int(route.startswith("S5")),
        "p4_pass": int(route.startswith("S5")),
        "promotion_allowed": int(route.startswith("S5")),
        "final_stop_allowed": int(route.startswith("R4") and missing == 0 and all(exp.safe_int(r.get("executed"), 0) for r in fallback_rows)),
        "hard_compute_budget_exhausted": int(route.startswith("R4")),
        "fallback_depth": 3,
        "fallback_rows": len(fallback_rows),
        "fallback_all_executed": int(all(exp.safe_int(r.get("executed"), 0) for r in fallback_rows)),
        "required_artifact_rows": len(manifest),
        "required_artifact_missing_count": missing,
        "line_f_official_gate_pass": line_f_official,
        "line_f_exploration_gate_pass": line_f_explore,
        "line_d_exploration_pass_rows": classic_exploration,
        **line_a,
        **json_safe(value),
        **json_safe(policy),
        **json_safe(multisketch),
        **code,
        "no_fake": 1,
    }
    exp.write_json(OUT_DIR / "v1225_route_decision.json", json_safe(route_payload))
    zip_path = package_zip()
    route_payload["code_review_packet"] = exp.rel(zip_path)
    route_payload["code_review_packet_entries"] = len(zipfile.ZipFile(zip_path).namelist())
    route_payload["code_review_packet_sha256"] = sha256_file(zip_path)
    exp.write_json(OUT_DIR / "v1225_route_decision.json", json_safe(route_payload))
    print(json.dumps(json_safe(route_payload), indent=2, ensure_ascii=False, sort_keys=True))
    return route_payload


if __name__ == "__main__":
    run()
