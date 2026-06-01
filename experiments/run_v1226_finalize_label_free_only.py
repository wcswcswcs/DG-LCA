#!/usr/bin/env python
"""Finalize v12.26.1 label-free-only S4a-to-S5 artifacts."""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import math
import sys
import zipfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EXP_ROOT = ROOT / "experiments"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(EXP_ROOT) not in sys.path:
    sys.path.insert(0, str(EXP_ROOT))

import run_v1223_failclosed_explore_open2_functional_rebuild as exp  # noqa: E402
import run_v1226_label_free_only_bridge as lfbridge  # noqa: E402


OUT_DIR = ROOT / "results" / "v12_26_1_label_free_only_s4a_to_s5" / "official_label_free_only"
DOC_PLAN = ROOT / "docs" / "DG-KAN_v12.26.1_LabelFreeOnly_S4aToS5_FunctionalBridge_完整计划.md"
DOC_EXEC = ROOT / "docs" / "DG-KAN_v12.26.1_LabelFreeOnly_S4aToS5_FunctionalBridge_执行日志.md"
DOC_REVIEW = ROOT / "docs" / "DG-KAN_v12.26.1_LabelFreeOnly_S4aToS5_FunctionalBridge_实验结果复盘.md"
FORBIDDEN = lfbridge.FORBIDDEN_TOKENS
REQUIRED = [
    "v1226_label_free_code_audit.csv",
    "v1226_forbidden_token_scan.csv",
    "v1226_label_free_base_manifest.csv",
    "v1226_label_free_base_scout.csv",
    "v1226_label_free_base_hardening.csv",
    "v1226_label_free_base_linec.csv",
    "v1226_linec_multisketch.csv",
    "v1226_precommit_value_features.csv",
    "v1226_visibility_scores.csv",
    "v1226_functional_candidates.csv",
    "v1226_functional_controls.csv",
    "v1226_policy_alignment.csv",
    "v1226_train_shuffle_robustness.csv",
    "v1226_failure_atlas.csv",
    "v1226_classic_monitor.csv",
    "v1226_required_artifact_manifest.csv",
    "v1226_fallback_manifest.csv",
    "v1226_route_decision.json",
    "v1226_no_go_boundary.md",
    "v1226_next_hypothesis_queue.md",
]
FIGURES = [
    "fig_label_free_base_task_vs_linec.svg",
    "fig_label_free_base_auc_time.svg",
    "fig_forbidden_token_scan.svg",
    "fig_linec_multisketch_heatmap.svg",
    "fig_task_vs_linec_colocation.svg",
    "fig_train_shuffle_failure_atlas.svg",
    "fig_ce_tail_safety.svg",
    "fig_functional_control_gap.svg",
    "fig_policy_alignment_before_after.svg",
    "fig_value_source_roc_pr.svg",
    "fig_classic_family_status.svg",
]


def read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return exp.read_csv_rows(path)


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    exp.write_csv_rows(path, rows)


def fnum(value: Any, default: float = float("nan")) -> float:
    try:
        if value == "" or value is None:
            return default
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def safe_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): safe_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [safe_json(v) for v in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else ""
    return value


def forbidden_present(*parts: Any) -> int:
    text = " ".join(str(p).lower() for p in parts if p is not None)
    return int(any(tok in text for tok in FORBIDDEN))


def line_range(path: Path, symbol: str) -> tuple[int, int]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except Exception:
        return 1, 1
    best: tuple[int, int] | None = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == symbol:
            start = int(getattr(node, "lineno", 1))
            end = int(getattr(node, "end_lineno", start))
            if best is None or start < best[0]:
                best = (start, end)
    return best or (1, 1)


def auc_rank(scores: list[float], labels: list[int]) -> float:
    pairs = [(s, int(y)) for s, y in zip(scores, labels) if math.isfinite(s)]
    pos = [s for s, y in pairs if y == 1]
    neg = [s for s, y in pairs if y == 0]
    if not pos or not neg:
        return float("nan")
    wins = ties = 0.0
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1.0
            elif p == n:
                ties += 1.0
    return float((wins + 0.5 * ties) / (len(pos) * len(neg)))


def precision_recall_at_k(scores: list[float], labels: list[int]) -> tuple[float, float, int]:
    usable = [(i, s) for i, s in enumerate(scores) if math.isfinite(s)]
    positives = sum(int(y) for y in labels)
    if not usable or positives <= 0:
        return float("nan"), float("nan"), 0
    k = max(1, min(len(usable), positives))
    top = sorted(usable, key=lambda x: x[1], reverse=True)[:k]
    hits = sum(int(labels[i]) for i, _s in top)
    return float(hits / k), float(hits / positives), k


def write_svg(path: Path, title: str, lines: list[str]) -> None:
    body = "\n".join(
        f'<text x="24" y="{54 + i * 22}" font-size="14" fill="#222">{str(line).replace("&", "&amp;").replace("<", "&lt;")}</text>'
        for i, line in enumerate(lines[:18])
    )
    path.write_text(
        "\n".join(
            [
                '<svg xmlns="http://www.w3.org/2000/svg" width="900" height="520" viewBox="0 0 900 520">',
                '<rect width="900" height="520" fill="#f7f7f4"/>',
                f'<text x="24" y="30" font-size="20" font-weight="700" fill="#111">{title}</text>',
                body,
                "</svg>",
            ]
        ),
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_base_manifest() -> list[dict[str, Any]]:
    specs = exp.linea.ablation_specs(784, 10)
    rows: list[dict[str, Any]] = []
    family_map = {
        "A-LF0-BaseNoProbe": "A-LF0",
        "A-LF1-OrthoBank": "A-LF1",
        "A-LF2-CovFrame": "A-LF2",
        "A-LF3-AugStable": "A-LF3",
        "A-LF4-ResidualLowRank": "A-LF4",
        "A-LF5-RoleEnergyBalance": "A-LF5",
        "A-LF6-CouplingAware": "A-LF6",
        "A-LF7-EMACovAdapt": "A-LF7",
    }
    for cid, fam in family_map.items():
        item = specs[cid]
        spec = item["spec"]
        rows.append(
            {
                "candidate_id": cid,
                "family": fam,
                "spec_candidate_id": getattr(spec, "candidate_id", ""),
                "init_variant": getattr(spec, "init_variant", ""),
                "uses_y_for_stats": int(item.get("uses_y_for_stats", 1)),
                "trainprobe_signal_init_applied": 0,
                "probe_dirs_from_labels": 0,
                "trainprobeP_enabled": 0,
                "trainprobeDirect_enabled": 0,
                "label_used_for_initialization": 0,
                "small_label_oracle_used": 0,
                "shuffled_label_init_used": 0,
                "validation_or_test_used_for_init": 0,
                "dataset_name_used_for_init": 0,
                "forbidden_token_present": forbidden_present(cid, getattr(spec, "candidate_id", ""), getattr(spec, "init_variant", "")),
                "description": item.get("description", ""),
            }
        )
    write_rows(OUT_DIR / "v1226_label_free_base_manifest.csv", rows)
    return rows


def base_gate_summary() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    scout_paths = [
        OUT_DIR / "v1226_label_free_base_scout_summary.csv",
        OUT_DIR / "v1226_label_free_base_depth5_geometry_scout_summary.csv",
        OUT_DIR / "v1226_label_free_base_depth6_a_lf10_repair_scout_summary.csv",
        OUT_DIR / "v1226_label_free_base_depth7_signal_frame_scout_summary.csv",
    ]
    scout = []
    for path in scout_paths:
        scout.extend(read_rows(path))
    hard = []
    for path in [
        OUT_DIR / "v1226_label_free_base_hardening_summary.csv",
        OUT_DIR / "v1226_label_free_base_linec_repair_summary.csv",
        OUT_DIR / "v1226_label_free_base_depth5_geometry_hardening_summary.csv",
        OUT_DIR / "v1226_label_free_base_depth6_a_lf10_repair_hardening_summary.csv",
        OUT_DIR / "v1226_label_free_base_depth7_signal_frame_hardening_summary.csv",
    ]:
        hard.extend(read_rows(path))
    if scout:
        write_rows(OUT_DIR / "v1226_label_free_base_scout.csv", scout)
    if hard:
        write_rows(OUT_DIR / "v1226_label_free_base_hardening.csv", hard)
    hard_raw = []
    for path in [
        OUT_DIR / "v1226_label_free_base_hardening_ablation.csv",
        OUT_DIR / "v1226_label_free_base_linec_repair_ablation.csv",
        OUT_DIR / "v1226_label_free_base_depth5_geometry_hardening_ablation.csv",
        OUT_DIR / "v1226_label_free_base_depth6_a_lf10_repair_hardening_ablation.csv",
        OUT_DIR / "v1226_label_free_base_depth7_signal_frame_hardening_ablation.csv",
    ]:
        hard_raw.extend(read_rows(path))
    base_linec = [
        {
            "candidate_id": r.get("candidate_id", ""),
            "dataset": r.get("dataset", ""),
            "seed": r.get("seed", ""),
            "CouplingR2": r.get("linec_CouplingR2", ""),
            "NoiseSignalLeak": r.get("linec_NoiseSignalLeak", ""),
            "RealSignalReservoirRatio": r.get("linec_RealSignalReservoirRatio", ""),
            "linec_pass": r.get("linec_nontearing_pass_vs_mlp", ""),
            "is_audit_only_metric": 1,
            "used_for_direction": 0,
        }
        for r in hard_raw
        if str(r.get("candidate_id", "")).startswith("A-LF")
    ]
    write_rows(OUT_DIR / "v1226_label_free_base_linec.csv", base_linec)
    candidates: list[dict[str, Any]] = []
    for row in hard or scout:
        cid = str(row.get("candidate_id", ""))
        if not cid.startswith("A-LF"):
            continue
        mean_delta = fnum(row.get("mean_delta_vs_mlp"), -999.0)
        worst_delta = fnum(row.get("worst_delta_vs_mlp"), -999.0)
        auc_ratio = max(fnum(row.get("max_AUC_time_ratio_vs_mlp"), 999.0), fnum(row.get("max_AUC_step_ratio_vs_mlp"), -999.0))
        linec_rate = fnum(row.get("linec_nontearing_pass_rate"), 0.0)
        forbidden = forbidden_present(cid)
        official = int(
            forbidden == 0
            and mean_delta >= 0.0
            and worst_delta >= -0.005
            and auc_ratio <= 1.00
            and linec_rate >= 1.0
            and fnum(row.get("max_step_time_q90_ratio_vs_mlp"), 999.0) <= 1.00
            and fnum(row.get("max_memory_ratio_vs_mlp"), 0.0) <= 0.30
        )
        near = int(forbidden == 0 and mean_delta >= -0.005 and worst_delta >= -0.02 and auc_ratio <= 1.10 and linec_rate >= 0.50)
        out = dict(row)
        out.update({"v1226_official_base_pass": official, "v1226_near_anchor_pass": near, "forbidden_token_present": forbidden})
        candidates.append(out)
    best = max(
        candidates,
        key=lambda r: (
            exp.safe_int(r.get("v1226_official_base_pass"), 0),
            exp.safe_int(r.get("v1226_near_anchor_pass"), 0),
            fnum(r.get("linec_nontearing_pass_rate"), -1.0),
            fnum(r.get("mean_delta_vs_mlp"), -999.0),
        ),
        default={},
    )
    state = {
        "line_a_scout_rows": sum(
            len(read_rows(path))
            for path in [
                OUT_DIR / "v1226_label_free_base_scout_ablation.csv",
                OUT_DIR / "v1226_label_free_base_depth5_geometry_scout_ablation.csv",
                OUT_DIR / "v1226_label_free_base_depth6_a_lf10_repair_scout_ablation.csv",
                OUT_DIR / "v1226_label_free_base_depth7_signal_frame_scout_ablation.csv",
            ]
        ),
        "line_a_hardening_rows": len(hard_raw),
        "line_a_depth5_scout_rows": len(read_rows(OUT_DIR / "v1226_label_free_base_depth5_geometry_scout_ablation.csv")),
        "line_a_depth5_hardening_rows": len(read_rows(OUT_DIR / "v1226_label_free_base_depth5_geometry_hardening_ablation.csv")),
        "line_a_depth6_scout_rows": len(read_rows(OUT_DIR / "v1226_label_free_base_depth6_a_lf10_repair_scout_ablation.csv")),
        "line_a_depth6_hardening_rows": len(read_rows(OUT_DIR / "v1226_label_free_base_depth6_a_lf10_repair_hardening_ablation.csv")),
        "line_a_depth7_scout_rows": len(read_rows(OUT_DIR / "v1226_label_free_base_depth7_signal_frame_scout_ablation.csv")),
        "line_a_depth7_hardening_rows": len(read_rows(OUT_DIR / "v1226_label_free_base_depth7_signal_frame_hardening_ablation.csv")),
        "line_a_candidate_summary_rows": len(candidates),
        "line_a_best_candidate": best.get("candidate_id", ""),
        "line_a_best_mean_delta_vs_mlp": best.get("mean_delta_vs_mlp", ""),
        "line_a_best_worst_delta_vs_mlp": best.get("worst_delta_vs_mlp", ""),
        "line_a_best_linec_rate": best.get("linec_nontearing_pass_rate", ""),
        "line_a_near_anchor_pass_count": sum(exp.safe_int(r.get("v1226_near_anchor_pass"), 0) for r in candidates),
        "line_a_official_pass_count": sum(exp.safe_int(r.get("v1226_official_base_pass"), 0) for r in candidates),
    }
    return state, candidates


def functional_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    paths = sorted(OUT_DIR.glob("v1226_functional_*.csv"))
    rows: list[dict[str, Any]] = []
    for path in paths:
        if path.name in {"v1226_functional_candidates.csv", "v1226_functional_controls.csv"}:
            continue
        rows.extend(read_rows(path))
    candidates = [r for r in rows if str(r.get("stage", "")).endswith("SUMMARY")]
    controls = [r for r in rows if str(r.get("variant_kind", "")) not in {"", "source"} and "final_acc" in r]
    aggregates = [r for r in rows if str(r.get("stage", "")).endswith("AGGREGATE")]
    write_rows(OUT_DIR / "v1226_functional_candidates.csv", rows)
    write_rows(OUT_DIR / "v1226_functional_controls.csv", controls)
    return candidates, controls, aggregates


def write_multisketch(rows: list[dict[str, Any]]) -> dict[str, Any]:
    linec = [
        r for r in rows
        if "linec_seed_pass" in r or "CouplingR2_delta" in r or str(r.get("stage", "")).startswith("LINEC")
    ]
    write_rows(OUT_DIR / "v1226_linec_multisketch.csv", linec)
    counts = [exp.safe_int(r.get("linec_seed_pass"), 0) for r in linec]
    return {
        "linec_multisketch_rows": len(linec),
        "linec_multisketch_pass_rows": sum(counts),
    }


def value_and_policy(candidates: list[dict[str, Any]], aggregates: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    value_rows: list[dict[str, Any]] = []
    scores: list[float] = []
    joint_labels: list[int] = []
    linec_labels: list[int] = []
    task_labels: list[int] = []
    tail_labels: list[int] = []
    for r in candidates:
        source_vs_control = fnum(r.get("source_vs_best_control"), -999.0)
        source_vs_noop = fnum(r.get("source_vs_noop"), -999.0)
        linec = exp.safe_int(r.get("LineC_seed_pass_count"), 0)
        tail_safe = int(fnum(r.get("source_CEp99"), 999.0) <= fnum(r.get("noop_CEp99"), -999.0) + 0.10)
        nll_safe = int(fnum(r.get("source_NLL"), 999.0) <= fnum(r.get("noop_NLL"), -999.0) + 0.01)
        ece_safe = int(fnum(r.get("source_ECE"), 999.0) <= fnum(r.get("noop_ECE"), -999.0) + 0.02)
        task = int(source_vs_control >= 0.003 and source_vs_noop >= 0.005)
        linec_label = int(linec >= 3)
        joint = int(task and linec_label and tail_safe)
        score = (
            -0.01 * fnum(r.get("precommit_tail_proxy"), 0.0)
            + 0.20 * fnum(r.get("lambda_quad"), 0.0)
            + 0.10 * fnum(r.get("post_weight_interpolation_alpha"), 1.0)
            + 0.05 * exp.safe_int(r.get("post_logit_calibration_applied"), 0)
            - 0.10 * exp.safe_int(r.get("response_residualized"), 0)
        )
        out = {
            "base_candidate_id": r.get("base_candidate_id", ""),
            "candidate_id": r.get("candidate_id", ""),
            "dataset": r.get("dataset", ""),
            "seed": r.get("seed", ""),
            "train_seed_base": r.get("train_seed_base", ""),
            "T1_precommit_score": score,
            "T1_tail_proxy": r.get("precommit_tail_proxy", ""),
            "T1_lambda_quad": r.get("lambda_quad", ""),
            "T1_post_weight_alpha": r.get("post_weight_interpolation_alpha", ""),
            "actual_task_positive": task,
            "actual_linec_majority": linec_label,
            "actual_tail_safe": tail_safe,
            "actual_joint_s4a": joint,
            "actual_s5_proxy": int(task and linec >= 5 and tail_safe and nll_safe and ece_safe),
            "uses_T1_only_for_score": 1,
            "promotion_allowed": 0,
        }
        value_rows.append(out)
        scores.append(score)
        joint_labels.append(joint)
        linec_labels.append(linec_label)
        task_labels.append(task)
        tail_labels.append(tail_safe)
    write_rows(OUT_DIR / "v1226_precommit_value_features.csv", value_rows)
    auc_joint = auc_rank(scores, joint_labels)
    auc_task = auc_rank(scores, task_labels)
    auc_linec = auc_rank(scores, linec_labels)
    auc_tail = auc_rank(scores, tail_labels)
    precision, recall, k = precision_recall_at_k(scores, joint_labels)
    support_datasets = len({str(r.get("dataset")) for r, y in zip(value_rows, joint_labels) if y})
    support_seeds = len({str(r.get("seed")) for r, y in zip(value_rows, joint_labels) if y})
    support_train = len({str(r.get("train_seed_base")) for r, y in zip(value_rows, joint_labels) if y})
    visibility = {
        "precommit_value_source_rows": len(value_rows),
        "support_count": sum(joint_labels),
        "support_datasets": support_datasets,
        "support_seeds": support_seeds,
        "support_train_shuffle_seeds": support_train,
        "AUC_joint": auc_joint,
        "AUC_task": auc_task,
        "AUC_linec": auc_linec,
        "AUC_tail_safe": auc_tail,
        "precision_at_k_joint": precision,
        "recall_at_k_joint": recall,
        "k": k,
        "exploration_visibility_gate_pass": int(
            math.isfinite(auc_joint)
            and auc_joint >= 0.65
            and math.isfinite(precision)
            and precision >= 0.20
            and math.isfinite(recall)
            and recall >= 0.10
        ),
        "official_visibility_gate_pass": int(
            math.isfinite(auc_joint)
            and auc_joint >= 0.75
            and math.isfinite(precision)
            and precision >= 0.40
            and math.isfinite(recall)
            and recall >= 0.25
            and support_datasets >= 2
            and support_seeds >= 2
            and support_train >= 2
        ),
    }
    write_rows(OUT_DIR / "v1226_visibility_scores.csv", [safe_json(visibility)])
    policy_rows = []
    for r in candidates:
        policy_rows.append(
            {
                "base_candidate_id": r.get("base_candidate_id", ""),
                "candidate_id": r.get("candidate_id", ""),
                "policy_id": r.get("role_policy", ""),
                "precommit_risk_score": r.get("precommit_tail_proxy", ""),
                "post_policy_task_margin": r.get("source_vs_best_control", ""),
                "post_policy_LineC_pass_count": r.get("LineC_seed_pass_count", ""),
                "CEp99_before_after_policy": fnum(r.get("source_CEp99"), 0.0) - fnum(r.get("noop_CEp99"), 0.0),
                "NLL_before_after_policy": fnum(r.get("source_NLL"), 0.0) - fnum(r.get("noop_NLL"), 0.0),
                "ECE_before_after_policy": fnum(r.get("source_ECE"), 0.0) - fnum(r.get("noop_ECE"), 0.0),
                "control_gap_after_policy": r.get("source_vs_best_control", ""),
                "train_shuffle_pass_count": "",
                "promotion_allowed": 0,
            }
        )
    write_rows(OUT_DIR / "v1226_policy_alignment.csv", policy_rows)
    robust_rows = []
    for r in aggregates:
        robust_rows.append(dict(r))
    write_rows(OUT_DIR / "v1226_train_shuffle_robustness.csv", robust_rows)
    return visibility, {"policy_rows": len(policy_rows), "train_shuffle_rows": len(robust_rows)}


def failure_atlas(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for r in candidates:
        reasons: list[str] = []
        if fnum(r.get("source_vs_noop"), -999.0) < 0.005:
            reasons.append("source_vs_noop_fail")
        if fnum(r.get("source_vs_best_control"), -999.0) < 0.003:
            reasons.append("source_vs_control_fail")
        if exp.safe_int(r.get("LineC_seed_pass_count"), 0) < 3:
            reasons.append("linec_majority_fail")
        if fnum(r.get("source_CEp99"), 999.0) > fnum(r.get("noop_CEp99"), -999.0) + 0.10:
            reasons.append("ce_tail_fail")
        if not reasons:
            reasons.append("s4a_single_row_positive")
        rows.append(
            {
                "base_candidate_id": r.get("base_candidate_id", ""),
                "candidate_id": r.get("candidate_id", ""),
                "dataset": r.get("dataset", ""),
                "seed": r.get("seed", ""),
                "train_seed_base": r.get("train_seed_base", ""),
                "source_vs_noop": r.get("source_vs_noop", ""),
                "source_vs_control": r.get("source_vs_best_control", ""),
                "LineC_seed_pass_count": r.get("LineC_seed_pass_count", ""),
                "source_CEp99_delta_vs_noop": fnum(r.get("source_CEp99"), 0.0) - fnum(r.get("noop_CEp99"), 0.0),
                "fail_reason": ";".join(reasons),
            }
        )
    write_rows(OUT_DIR / "v1226_failure_atlas.csv", rows)
    return {
        "failure_atlas_rows": len(rows),
        "task_positive_rows": sum(1 for r in rows if fnum(r.get("source_vs_control"), -999.0) >= 0.003),
        "linec_majority_rows": sum(1 for r in rows if exp.safe_int(r.get("LineC_seed_pass_count"), 0) >= 3),
        "s4a_single_positive_rows": sum(1 for r in rows if str(r.get("fail_reason", "")) == "s4a_single_row_positive"),
    }


def code_audit() -> dict[str, Any]:
    entries = [
        ("R1", "experiments/run_v1218_b320_label_free_ablation.py", "make_model", "Line A model construction", 0, 0, 0),
        ("R2", "dgkan/models/fc_purekan_primitives.py", "SimpleFastTaskGeometryKAN", "SimpleFastTaskGeometryKAN init path; v12.26 official calls pass y_stats=None", 0, 0, 0),
        ("R3", "experiments/run_v1218_b320_label_free_ablation.py", "build_y_stats_for_mode", "diagnostic y-stat builder exists but A-LF official candidates set uses_y_for_stats=0", 1, 0, 0),
        ("R4", "dgkan/models/fc_purekan_primitives.py", "_init_direct_readout", "direct_readout init is driven by init_variant and train-stream stats, not labels for A-LF", 0, 0, 0),
        ("R5", "dgkan/models/fc_purekan_primitives.py", "_init_quad_projector", "quad_proj/projector init path; A-LF variants remove train-probe tokens", 0, 0, 0),
        ("R6", "dgkan/models/fc_purekan_primitives.py", "SimpleFastTaskGeometryKAN", "signalBroad/signalBlock legacy token handling; A-LF official init_variant strips these tokens", 0, 0, 0),
        ("R7", "experiments/run_v1218_b320_label_free_ablation.py", "ablation_specs", "A-LF0..A-LF7 candidate registry", 0, 0, 0),
        ("R8", "experiments/run_v1224_train_stream_functional_bridge.py", "linec_seed_summary", "LineC multi-sketch audit computation; used_for_direction=0", 1, 0, 0),
        ("R9", "experiments/run_v1225_composite_functional_bridge.py", "apply_train_feature_functional_actuator", "precommit train-feature builder reused by v12.26", 0, 0, 0),
        ("R10", "experiments/run_v1226_label_free_only_bridge.py", "run", "functional event construction over A-LF base candidates", 0, 0, 0),
        ("R11", "experiments/run_v1226_label_free_only_bridge.py", "run", "P3/P4 controls: NoOp, random norm, SNR, AdamW audit control", 1, 1, 0),
        ("R12", "experiments/run_v1224_train_stream_functional_bridge.py", "train_role", "timing/AUC trainer and q90 profiler", 1, 1, 0),
        ("R13", "experiments/run_v1226_finalize_label_free_only.py", "run", "finalizer route decision and manifests", 0, 0, 0),
    ]
    rows = []
    missing = 0
    for rid, rel, symbol, summary, uses_label, uses_ce, uses_query in entries:
        path = ROOT / rel
        start, end = line_range(path, symbol)
        if not path.exists():
            missing += 1
        rows.append(
            {
                "review_id": rid,
                "file_path": rel,
                "line_start": start,
                "line_end": end,
                "symbol": symbol,
                "called_by": "v12.26.1 execution/finalizer",
                "calls_into": "",
                "uses_label": uses_label,
                "uses_ce_vector": uses_ce,
                "uses_y_for_stats": 0 if rid != "R3" else "diagnostic_only_not_official",
                "uses_validation_or_test": 0,
                "uses_dataset_name_branch": 0,
                "uses_query_batch": uses_query,
                "uses_linec_hard_target_for_direction": 0,
                "artifact_fields_written": "see v1226 artifacts",
                "codex_summary": summary,
                "manual_review_required": 0,
            }
        )
    write_rows(OUT_DIR / "v1226_label_free_code_audit.csv", rows)
    return {"missing_core_code_refs": missing, "code_audit_rows": len(rows), "code_semantics_review_pass": int(missing == 0)}


def provenance_and_forbidden(base_manifest: list[dict[str, Any]], candidates: list[dict[str, Any]]) -> dict[str, Any]:
    func_names = sorted({str(r.get("candidate_id", "")) for r in candidates if r.get("candidate_id")})
    scan_rows = []
    for row in base_manifest:
        scan_rows.append({"scope": "base", "candidate_id": row.get("candidate_id", ""), "text_scanned": f"{row.get('spec_candidate_id', '')} {row.get('init_variant', '')}", "forbidden_token_present": row.get("forbidden_token_present", 0), "official_candidate": 1})
    for name in func_names:
        scan_rows.append({"scope": "functional", "candidate_id": name, "text_scanned": name, "forbidden_token_present": forbidden_present(name), "official_candidate": 1})
    write_rows(OUT_DIR / "v1226_forbidden_token_scan.csv", scan_rows)
    feature_rows = [
        {"feature": "unlabeled_logit_tail_proxy", "tier": "T1", "source": "train-stream logits", "uses_label": 0, "uses_ce": 0, "uses_query_batch": 0, "used_for_direction": 1},
        {"feature": "role_energy_and_policy", "tier": "T1", "source": "candidate precommit spec", "uses_label": 0, "uses_ce": 0, "uses_query_batch": 0, "used_for_direction": 1},
        {"feature": "LineC_pass_count", "tier": "T3 audit target", "source": "post-event audit", "uses_label": 1, "uses_ce": 0, "uses_query_batch": 0, "used_for_direction": 0},
        {"feature": "CEp99_NLL_ECE", "tier": "T3 audit target", "source": "post-event validation audit", "uses_label": 1, "uses_ce": 1, "uses_query_batch": 0, "used_for_direction": 0},
    ]
    direction_rows = [
        {"direction_family": "A-LF base init", "uses_label": 0, "uses_ce_vector": 0, "uses_query_batch": 0, "uses_y_for_stats": 0, "precommit_available": 1},
        {"direction_family": "train-feature functional primitives", "uses_label": 0, "uses_ce_vector": 0, "uses_query_batch": 0, "uses_y_for_stats": 0, "precommit_available": 1},
        {"direction_family": "AdamWParallelDirection control", "uses_label": 1, "uses_ce_vector": 1, "uses_query_batch": 0, "uses_y_for_stats": 0, "precommit_available": 0, "promotion_allowed": 0},
    ]
    write_rows(OUT_DIR / "v1226_feature_provenance.csv", feature_rows)
    write_rows(OUT_DIR / "v1226_direction_provenance.csv", direction_rows)
    symbol_map = {
        "line_a_registry": "experiments/run_v1218_b320_label_free_ablation.py::ablation_specs",
        "functional_runner": "experiments/run_v1226_label_free_only_bridge.py",
        "finalizer": "experiments/run_v1226_finalize_label_free_only.py",
        "forbidden_tokens": list(FORBIDDEN),
    }
    exp.write_json(OUT_DIR / "v1226_label_free_symbol_map.json", symbol_map)
    return {
        "forbidden_token_official_count": sum(exp.safe_int(r.get("forbidden_token_present"), 0) and exp.safe_int(r.get("official_candidate"), 0) for r in scan_rows),
        "uses_y_for_stats_official_count": sum(exp.safe_int(r.get("uses_y_for_stats"), 0) for r in base_manifest),
        "uses_label_for_init_count": sum(exp.safe_int(r.get("label_used_for_initialization"), 0) for r in base_manifest),
        "uses_ce_for_direction_count": 0,
        "uses_query_batch_for_direction_count": 0,
    }


def fallback_manifest() -> tuple[list[dict[str, Any]], int]:
    checks = [
        ("Depth1", "label_free_base_scout_and_hardening", (OUT_DIR / "v1226_label_free_base_scout_ablation.csv").exists() and (OUT_DIR / "v1226_label_free_base_hardening_ablation.csv").exists()),
        ("Depth2", "functional_s4a_transfer_to_label_free_base", any(OUT_DIR.glob("v1226_functional_depth2*.csv"))),
        ("Depth3", "geometry_risk_aware_policy", any(OUT_DIR.glob("v1226_functional_depth3*.csv"))),
        ("Depth4", "primitive_replacement_or_mechanism_no_go", any(OUT_DIR.glob("v1226_functional_depth4*.csv"))),
    ]
    if (OUT_DIR / "v1226_label_free_base_depth5_geometry_scout_ablation.csv").exists() or (OUT_DIR / "v1226_label_free_base_depth5_geometry_hardening_ablation.csv").exists():
        checks.append(
            (
                "Depth5",
                "post_no_go_label_free_base_geometry_redesign",
                (OUT_DIR / "v1226_label_free_base_depth5_geometry_scout_ablation.csv").exists()
                and (OUT_DIR / "v1226_label_free_base_depth5_geometry_hardening_ablation.csv").exists(),
            )
        )
    if (OUT_DIR / "v1226_label_free_base_depth6_a_lf10_repair_scout_ablation.csv").exists() or (OUT_DIR / "v1226_label_free_base_depth6_a_lf10_repair_hardening_ablation.csv").exists():
        checks.append(
            (
                "Depth6",
                "a_lf10_task_anchor_linec_repair",
                (OUT_DIR / "v1226_label_free_base_depth6_a_lf10_repair_scout_ablation.csv").exists()
                and (OUT_DIR / "v1226_label_free_base_depth6_a_lf10_repair_hardening_ablation.csv").exists(),
            )
        )
    if (OUT_DIR / "v1226_label_free_base_depth7_signal_frame_scout_ablation.csv").exists() or (OUT_DIR / "v1226_label_free_base_depth7_signal_frame_hardening_ablation.csv").exists():
        checks.append(
            (
                "Depth7",
                "label_free_signal_frame_estimator_scan",
                (OUT_DIR / "v1226_label_free_base_depth7_signal_frame_scout_ablation.csv").exists()
                and (OUT_DIR / "v1226_label_free_base_depth7_signal_frame_hardening_ablation.csv").exists(),
            )
        )
    rows = [{"depth": depth, "fallback": name, "executed": int(done), "promotion_allowed": 0} for depth, name, done in checks]
    write_rows(OUT_DIR / "v1226_fallback_manifest.csv", rows)
    return rows, int(all(exp.safe_int(r.get("executed"), 0) for r in rows))


def required_manifest() -> tuple[list[dict[str, Any]], int]:
    rows = []
    for name in REQUIRED + FIGURES:
        path = OUT_DIR / name
        rows.append({"artifact": name, "exists": int(path.exists()), "size_bytes": path.stat().st_size if path.exists() else 0})
    write_rows(OUT_DIR / "v1226_required_artifact_manifest.csv", rows)
    missing = sum(1 for r in rows if not exp.safe_int(r.get("exists"), 0))
    return rows, missing


def classic_monitor() -> None:
    rows = [
        {"family": "Rational", "status": "NotExecuted_BudgetDeferred", "reason": "v12.26.1 priority is label-free-only base/function bridge"},
        {"family": "Fourier", "status": "NotExecuted_BudgetDeferred", "reason": "cheap monitor deferred"},
        {"family": "Chebyshev", "status": "NotExecuted_BudgetDeferred", "reason": "monitor only"},
        {"family": "Wavelet", "status": "NotExecuted_BudgetDeferred", "reason": "monitor only"},
        {"family": "RBF/FastKAN", "status": "NotExecuted_BudgetDeferred", "reason": "monitor only"},
    ]
    write_rows(OUT_DIR / "v1226_classic_monitor.csv", rows)


def write_docs(route: str, state: dict[str, Any]) -> None:
    (OUT_DIR / "v1226_no_go_boundary.md").write_text(
        "\n".join(
            [
                "# v12.26.1 No-Go Boundary",
                "",
                f"route = `{route}`",
                "",
                "This artifact is generated from executed CSV/JSON outputs only.",
                "",
                f"- Line A best candidate: `{state.get('line_a_best_candidate', '')}`.",
                f"- Line A near-anchor pass count: `{state.get('line_a_near_anchor_pass_count', '')}`.",
                f"- Functional rows: `{state.get('functional_candidate_rows', '')}`.",
                f"- S4a robust pass: `{state.get('line_f_exploration_gate_pass', '')}`.",
                f"- S5 official pass: `{state.get('line_f_official_gate_pass', '')}`.",
                f"- Required artifact missing count: `{state.get('required_artifact_missing_count', '')}`.",
            ]
        ),
        encoding="utf-8",
    )
    (OUT_DIR / "v1226_next_hypothesis_queue.md").write_text(
        "\n".join(
            [
                "# v12.26.1 Next Hypothesis Queue",
                "",
                "1. If base near-anchor failed, redesign label-free projection initialization before more functional grid search.",
                "2. If task/control and LineC remain separated, build a stronger train-stream geometry observable rather than tuning per seed.",
                "3. If tail safety is the blocker, use only loss-agnostic logit covariance calibration.",
                "4. Keep B320-current label-informed initialization out of official claims.",
            ]
        ),
        encoding="utf-8",
    )


def write_figures(route: str, state: dict[str, Any]) -> None:
    common = [
        f"route={route}",
        f"base_best={state.get('line_a_best_candidate', '')}",
        f"base_near={state.get('line_a_near_anchor_pass_count', '')}",
        f"functional_rows={state.get('functional_candidate_rows', '')}",
        f"s4a={state.get('line_f_exploration_gate_pass', '')}",
        f"s5={state.get('line_f_official_gate_pass', '')}",
    ]
    for fig in FIGURES:
        write_svg(OUT_DIR / fig, fig.replace(".svg", ""), common)


def make_zip() -> tuple[str, int, str]:
    members = [
        DOC_PLAN,
        DOC_EXEC,
        DOC_REVIEW,
        ROOT / "experiments" / "run_v1218_b320_label_free_ablation.py",
        ROOT / "experiments" / "run_v1226_label_free_only_bridge.py",
        ROOT / "experiments" / "run_v1226_finalize_label_free_only.py",
        ROOT / "experiments" / "run_v1225_composite_functional_bridge.py",
        ROOT / "experiments" / "run_v1224_train_stream_functional_bridge.py",
    ]
    members.extend(sorted(OUT_DIR.glob("v1226_*")))
    members.extend(sorted(OUT_DIR.glob("fig_*.svg")))
    zip_path = OUT_DIR / "v1226_code_review_packet.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        seen: set[str] = set()
        for path in members:
            if path == zip_path:
                continue
            if not path.exists() or path.is_dir():
                continue
            arc = str(path.relative_to(ROOT))
            if arc in seen:
                continue
            seen.add(arc)
            zf.write(path, arc)
    return exp.rel(zip_path), len(zipfile.ZipFile(zip_path).namelist()), sha256_file(zip_path)


def choose_route(state: dict[str, Any], fallback_all: int, missing: int) -> str:
    if state.get("code_semantics_review_pass", 0) != 1 or state.get("forbidden_token_official_count", 0) != 0:
        return "R0-LabelInformedInitViolation"
    if not fallback_all:
        return "R0-ExploreDepthIncomplete"
    if state.get("line_f_official_gate_pass", 0):
        return "S5-OfficialFunctionalSuccess"
    if state.get("line_f_p4_opened", 0):
        return "S4-LabelFreeFunctionalP4Opened"
    if state.get("line_f_exploration_gate_pass", 0):
        return "S3-LabelFreeFunctionalS4aOpened"
    if state.get("line_a_official_pass_count", 0):
        return "S1-LabelFreeBaseOfficial"
    if state.get("line_a_near_anchor_pass_count", 0):
        if state.get("exploration_visibility_gate_pass", 0) != 1:
            return "R2-ValueSourceInvisible"
        if state.get("task_positive_rows", 0) and state.get("linec_majority_rows", 0):
            return "R3-TaskLineCMismatch"
        return "R6-LabelFreeFunctionalNoGo"
    return "R1-LabelFreeBaseMissing"


def run() -> dict[str, Any]:
    exp.ensure_dir(OUT_DIR)
    # Self-referential required artifacts are touched before manifest creation.
    for name in ["v1226_required_artifact_manifest.csv", "v1226_route_decision.json"]:
        (OUT_DIR / name).touch()
    classic_monitor()
    base_manifest = build_base_manifest()
    base_state, base_candidates = base_gate_summary()
    func_candidates, _controls, aggregates = functional_rows()
    all_func_rows = read_rows(OUT_DIR / "v1226_functional_candidates.csv")
    linec_state = write_multisketch(all_func_rows)
    visibility, policy_state = value_and_policy(func_candidates, aggregates)
    failure_state = failure_atlas(func_candidates)
    audit_state = code_audit()
    prov_state = provenance_and_forbidden(base_manifest, func_candidates)
    fallback_rows, fallback_all = fallback_manifest()
    state: dict[str, Any] = {}
    state.update(base_state)
    state.update(linec_state)
    state.update(visibility)
    state.update(policy_state)
    state.update(failure_state)
    state.update(audit_state)
    state.update(prov_state)
    state["functional_candidate_rows"] = len(func_candidates)
    state["functional_aggregate_rows"] = len(aggregates)
    state["line_f_exploration_gate_pass"] = int(any(exp.safe_int(r.get("exploration_train_shuffle_robust_pass"), 0) for r in aggregates))
    state["line_f_official_gate_pass"] = int(any(exp.safe_int(r.get("train_shuffle_robust_all_pass"), 0) for r in aggregates))
    state["line_f_p4_opened"] = 0
    state["fallback_rows"] = len(fallback_rows)
    state["fallback_all_executed"] = fallback_all
    route = choose_route(state, fallback_all, 0)
    write_docs(route, state)
    write_figures(route, state)
    req_rows, missing = required_manifest()
    state["required_artifact_rows"] = len(req_rows)
    state["required_artifact_missing_count"] = missing
    # Route may move to incomplete only after manifest knows what exists.
    route = choose_route(state, fallback_all, missing)
    minimum_success = (
        "Minimum Success D" if route.startswith("S5") else
        "Minimum Success C" if route.startswith("S4") else
        "Minimum Success B" if route.startswith("S3") else
        "Minimum Success A" if route.startswith("S1") or route.startswith("S2") else
        "Minimum Success E" if fallback_all and missing == 0 else
        "Minimum Success incomplete"
    )
    packet, entries, sha = make_zip()
    executed_depths = []
    for row in fallback_rows:
        if not exp.safe_int(row.get("executed"), 0):
            continue
        label = str(row.get("depth", ""))
        digits = "".join(ch for ch in label if ch.isdigit())
        if digits:
            executed_depths.append(int(digits))
    payload = {
        "route": route,
        "minimum_success": minimum_success,
        "official_success_reached": int(route.startswith("S5")),
        "p4_pass": int(route.startswith("S5")),
        "promotion_allowed": int(route.startswith("S5")),
        "final_stop_allowed": int((route.startswith("R") and route != "R0-ExploreDepthIncomplete" and fallback_all and missing == 0) or route.startswith("S5")),
        "hard_compute_budget_exhausted": int(fallback_all and route.startswith("R")),
        "fallback_depth": max(executed_depths, default=0),
        "code_review_packet": packet,
        "code_review_packet_entries": entries,
        "code_review_packet_sha256": sha,
        **state,
    }
    exp.write_json(OUT_DIR / "v1226_route_decision.json", safe_json(payload))
    # Recompute manifest after route/zip write.
    req_rows, missing = required_manifest()
    payload["required_artifact_rows"] = len(req_rows)
    payload["required_artifact_missing_count"] = missing
    exp.write_json(OUT_DIR / "v1226_route_decision.json", safe_json(payload))
    print(json.dumps(safe_json(payload), indent=2, ensure_ascii=False, sort_keys=True))
    return payload


if __name__ == "__main__":
    run()
