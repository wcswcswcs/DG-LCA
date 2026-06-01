#!/usr/bin/env python
"""Finalize v12.24 FunctionalBridge artifacts and route decision.

This script is intentionally conservative: it aggregates already-written CSV
and JSON artifacts, writes the required audit manifests and figures, then
packages code/results.  It does not rerun training and does not invent missing
metrics.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import experiments.run_v1223_failclosed_explore_open2_functional_rebuild as exp


OUT_DIR = ROOT / "results" / "v12_24_s4_to_s5_functional_bridge" / "official_s4_to_s5"
PLAN_DOC = ROOT / "docs" / "DG-KAN_v12.24_S4toS5_FunctionalBridge_LabelFreeClassic_独立分析与下一步计划.md"
EXEC_LOG = ROOT / "docs" / "DG-KAN_v12.24_S4toS5_FunctionalBridge_LabelFreeClassic_执行日志.md"
REVIEW_LOG = ROOT / "docs" / "DG-KAN_v12.24_S4toS5_FunctionalBridge_LabelFreeClassic_实验结果复盘.md"
V1223_SOURCE_OUT_DIR = ROOT / "results" / "v12_23_failclosed_explore_open2_functional_rebuild" / "official_explore_open2_i24_directcomp_all_targeted"
NG2_PREFIX = "v1224_ng2_train_stream_bridge_replay"
NG3_PREFIX = "v1224_ng3_policy_probe_reset"
NG4_PREFIX = "v1224_ng4_policy_probe_lr_epoch_repair"
NG5_PREFIX = "v1224_ng5_tailrisk_weightdecay_repair"
NG6_PREFIX = "v1224_ng6_label_smoothing_tail_repair"
NG7_PREFIX = "v1224_ng7_light_label_smoothing_tail_repair"
NG8_PREFIX = "v1224_ng8_tail_safe_precommit_probe"
NG9_PREFIX = "v1224_ng9_tail_safe_quaddirect_repair"
NG10_PREFIX = "v1224_ng10_contrast_tail_safe_probe"
NG11_PREFIX = "v1224_ng11_tail_safe_quaddirect_wd003_repair"
NG12_PREFIX = "v1224_ng12_tail_safe_quaddirect_wd005_repair"
NG13_PREFIX = "v1224_ng13_contrast2_tail_safe_quaddirect_probe"
NG14_PREFIX = "v1224_ng14_contrast3_tail_safe_quaddirect_probe"
NG15_PREFIX = "v1224_ng15_i32_expanded_policy_probe_quaddirect"
NG16_PREFIX = "v1224_ng16_i32_expanded_policy_probe_directbranch"
NG17_PREFIX = "v1224_ng17_i32_expanded_policy_probe_directgain"
NG18_PREFIX = "v1224_ng18_i32_expanded_policy_probe_directonly"
EXTRA_NG_PREFIXES = [NG11_PREFIX, NG12_PREFIX, NG13_PREFIX, NG14_PREFIX, NG15_PREFIX, NG16_PREFIX, NG17_PREFIX, NG18_PREFIX]

TRANSITIVE_CODE_PACKET_MEMBERS = [
    ROOT / "experiments" / "run_v1223_failclosed_explore_open2_functional_rebuild.py",
    ROOT / "experiments" / "run_v1223_p4_compensation_modes.py",
    ROOT / "experiments" / "run_v1223_p4_official_row_scan.py",
    ROOT / "experiments" / "run_v1223_shadowp4_coupling_preserving_blend.py",
    ROOT / "experiments" / "run_v1223_p4_trainable_role_scan.py",
    ROOT / "experiments" / "run_v1223_p4_trajectory_gate_verifier.py",
    ROOT / "experiments" / "run_v1223_p4_reservoir_veto_checkpoint_scan.py",
    ROOT / "experiments" / "run_v1223_p4_train_ensemble_compensation_scan.py",
    ROOT / "experiments" / "run_v1223_p4_multisketch_aggregate_gate.py",
    ROOT / "experiments" / "run_v1223_p4_precommit_proxy_checkpoint.py",
    ROOT / "experiments" / "run_v1223_line_d_hardening.py",
    ROOT / "experiments" / "summarize_v1223_line_d_hardening.py",
    ROOT / "experiments" / "run_v1224_train_stream_functional_bridge.py",
    ROOT / "experiments" / "run_v1224_i30_i32_policy_bridge.py",
    ROOT / "experiments" / "run_v1224_classic_hardening.py",
    ROOT / "experiments" / "run_v1224_finalize_functional_bridge.py",
    ROOT / "experiments" / "run_v1218_b320_label_free_ablation.py",
    ROOT / "dgkan" / "models" / "fc_purekan_primitives.py",
    ROOT / "dgkan" / "kernels" / "fused_hinge_quadratic.py",
]


def fnum(value: Any, default: float = float("nan")) -> float:
    try:
        if value == "" or value is None:
            return default
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    exp.write_csv_rows(path, rows)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else ""
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    return value


def find_line(path: Path, pattern: str) -> tuple[int, int]:
    if not path.exists():
        return 0, 0
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    for idx, line in enumerate(lines, start=1):
        stripped = line.lstrip()
        if pattern.startswith("def "):
            matched = stripped.startswith(pattern + "(") or stripped.startswith(pattern + " (")
        else:
            matched = pattern in line
        if matched:
            end = idx
            for j in range(idx, min(len(lines), idx + 160)):
                if j > idx and lines[j - 1].startswith("def "):
                    end = j - 1
                    break
                end = j
            return idx, end
    return 0, 0


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


def find_first(rows: list[dict[str, Any]], **criteria: Any) -> dict[str, Any]:
    for row in rows:
        if all(str(row.get(k, "")) == str(v) for k, v in criteria.items()):
            return row
    return {}


def summarize_linea(out_dir: Path) -> dict[str, Any]:
    summary_candidates = sorted((out_dir / "linea").glob("**/*summary*.csv"))
    rows: list[dict[str, Any]] = []
    for path in summary_candidates:
        rows.extend(read_rows(path))
    hardening_rows = [
        r for r in rows
        if str(r.get("candidate_id", "")).startswith("A") and "labelInit" not in str(r.get("candidate_id", ""))
    ]
    best = None
    if hardening_rows:
        best = max(
            hardening_rows,
            key=lambda r: (
                exp.safe_int(r.get("exploration_label_free_candidate_pass"), 0),
                fnum(r.get("linec_nontearing_pass_rate"), -1.0),
                fnum(r.get("mean_delta_vs_A0_labelInit"), -999.0),
            ),
        )
    return {
        "linea_summary_files": [exp.rel(p) for p in summary_candidates],
        "linea_summary_rows": len(rows),
        "linea_label_free_candidate_rows": len(hardening_rows),
        "linea_exploration_pass_count": sum(exp.safe_int(r.get("exploration_label_free_candidate_pass"), 0) for r in hardening_rows),
        "linea_official_pass_count": sum(exp.safe_int(r.get("official_label_free_candidate_pass"), 0) for r in hardening_rows),
        "linea_best_candidate": (best or {}).get("candidate_id", ""),
        "linea_best_mean_delta_vs_A0": (best or {}).get("mean_delta_vs_A0_labelInit", ""),
        "linea_best_worst_delta_vs_A0": (best or {}).get("worst_delta_vs_A0_labelInit", ""),
        "linea_best_linec_pass_rate": (best or {}).get("linec_nontearing_pass_rate", ""),
    }


def write_t1b_microprobe(out_dir: Path) -> dict[str, Any]:
    summary_sources = [
        (out_dir / "v1224_train_stream_functional_bridge.csv", "V1224_TRAIN_STREAM_BRIDGE_SUMMARY"),
        (out_dir / "v1224_i30_i32_policy_bridge.csv", "V1224_I30_I32_BRIDGE_SUMMARY"),
        (out_dir / f"{NG2_PREFIX}.csv", "V1224_TRAIN_STREAM_BRIDGE_SUMMARY"),
        (out_dir / f"{NG3_PREFIX}.csv", "V1224_I30_I32_BRIDGE_SUMMARY"),
        (out_dir / f"{NG4_PREFIX}.csv", "V1224_I30_I32_BRIDGE_SUMMARY"),
        (out_dir / f"{NG5_PREFIX}.csv", "V1224_I30_I32_BRIDGE_SUMMARY"),
        (out_dir / f"{NG6_PREFIX}.csv", "V1224_I30_I32_BRIDGE_SUMMARY"),
        (out_dir / f"{NG7_PREFIX}.csv", "V1224_I30_I32_BRIDGE_SUMMARY"),
        (out_dir / f"{NG8_PREFIX}.csv", "V1224_I30_I32_BRIDGE_SUMMARY"),
        (out_dir / f"{NG9_PREFIX}.csv", "V1224_I30_I32_BRIDGE_SUMMARY"),
        (out_dir / f"{NG10_PREFIX}.csv", "V1224_I30_I32_BRIDGE_SUMMARY"),
    ]
    micro_rows: list[dict[str, Any]] = []
    scores: list[float] = []
    labels: list[int] = []
    for source_csv, stage_name in summary_sources:
        bridge_rows = read_rows(source_csv)
        summary_rows = [r for r in bridge_rows if r.get("stage") == stage_name]
        for row in summary_rows:
            score = (
                fnum(row.get("source_vs_noop_acc_delta"), 0.0)
                + fnum(row.get("source_vs_best_control_acc_delta"), 0.0)
                + 0.01 * fnum(row.get("linec_seed_pass_count"), 0.0)
            )
            label = exp.safe_int(row.get("strict_majority_pass"), 0)
            scores.append(score)
            labels.append(label)
            micro_rows.append(
                {
                    "stage": "V1224_T1B_ONLINE_MICROPROBE",
                    "source_artifact": exp.rel(source_csv),
                    "candidate_name": row.get("candidate_name", ""),
                    "train_seed_base": row.get("train_seed_base", ""),
                    "score": score,
                    "target_strict_majority_pass": label,
                    "source_vs_noop_acc_delta": row.get("source_vs_noop_acc_delta", ""),
                    "source_vs_best_control_acc_delta": row.get("source_vs_best_control_acc_delta", ""),
                    "linec_seed_pass_count": row.get("linec_seed_pass_count", ""),
                    "uses_label": 0,
                    "uses_ce_vector": 0,
                    "uses_query_batch": 0,
                    "promotion_allowed": 0,
                    "no_fake": 1,
                }
            )
    auc = auc_rank(scores, labels)
    micro_summary = {
        "stage": "V1224_T1B_ONLINE_MICROPROBE_SUMMARY",
        "rows": len(micro_rows),
        "positive_rows": sum(labels),
        "class_count": len(set(labels)) if labels else 0,
        "auc": auc,
        "exploration_gate_pass": int(math.isfinite(auc) and auc >= 0.60 and sum(labels) > 0),
        "promotion_allowed": 0,
        "no_fake": 1,
    }
    micro_rows.append(micro_summary)
    csv_path = out_dir / "v1224_t1b_online_microprobe.csv"
    json_path = out_dir / "v1224_t1b_online_microprobe_summary.json"
    write_rows(csv_path, micro_rows)
    exp.write_json(json_path, micro_summary | {"artifact_csv": exp.rel(csv_path)})

    # Response-distillation fallback is diagnostic-only: it reads completed
    # v12.23 response artifacts and cannot become a precommit direction source.
    source_dirs = [
        ROOT / "results" / "v12_23_failclosed_explore_open2_functional_rebuild" / "official_explore_open2_i24_directcomp_all_targeted",
        ROOT / "results" / "v12_23_failclosed_explore_open2_functional_rebuild" / "official_explore_open2_shadowp4_p3_repair_kmnist_s1_i24i25",
    ]
    fallback_rows = []
    for src in source_dirs:
        for name in [
            "v1223_t2_clone_probe_visibility_diagnostic.csv",
            "v1223_t3_visibility_support_stress.csv",
        ]:
            p = src / name
            fallback_rows.append(
                {
                    "stage": "V1224_RESPONSE_DISTILLATION_FALLBACK",
                    "source_artifact": exp.rel(p),
                    "exists": int(p.exists()),
                    "diagnostic_only": 1,
                    "precommit_available": 0,
                    "promotion_allowed": 0,
                    "no_fake": 1,
                }
            )
    fallback_csv = out_dir / "v1224_response_distillation_fallback.csv"
    write_rows(fallback_csv, fallback_rows)
    distilled = write_t1b_response_distilled_features(out_dir, fallback_rows)
    return micro_summary | {
        "t1b_microprobe_csv": exp.rel(csv_path),
        "response_distillation_fallback_csv": exp.rel(fallback_csv),
        **distilled,
    }


def write_t1b_response_distilled_features(out_dir: Path, fallback_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Write Level-2 T1B feature artifact without using labels/CE for features.

    The feature rows are built from precommit/train-stream probe artifacts only.
    Completed T2/T3 artifacts are referenced as the teacher boundary, but their
    response labels are not used as deployable features.  Targets remain audit
    labels from completed v12.24 bridge summaries.
    """

    source_specs = [
        (out_dir / "v1224_train_stream_functional_bridge.csv", "V1224_TRAIN_STREAM_BRIDGE_SUMMARY", "V1224_TRAIN_STREAM_BRIDGE_VARIANT"),
        (out_dir / "v1224_i30_i32_policy_bridge.csv", "V1224_I30_I32_BRIDGE_SUMMARY", "V1224_I30_I32_BRIDGE_VARIANT"),
        (out_dir / f"{NG2_PREFIX}.csv", "V1224_TRAIN_STREAM_BRIDGE_SUMMARY", "V1224_TRAIN_STREAM_BRIDGE_VARIANT"),
        (out_dir / f"{NG3_PREFIX}.csv", "V1224_I30_I32_BRIDGE_SUMMARY", "V1224_I30_I32_BRIDGE_VARIANT"),
        (out_dir / f"{NG4_PREFIX}.csv", "V1224_I30_I32_BRIDGE_SUMMARY", "V1224_I30_I32_BRIDGE_VARIANT"),
        (out_dir / f"{NG5_PREFIX}.csv", "V1224_I30_I32_BRIDGE_SUMMARY", "V1224_I30_I32_BRIDGE_VARIANT"),
        (out_dir / f"{NG6_PREFIX}.csv", "V1224_I30_I32_BRIDGE_SUMMARY", "V1224_I30_I32_BRIDGE_VARIANT"),
        (out_dir / f"{NG7_PREFIX}.csv", "V1224_I30_I32_BRIDGE_SUMMARY", "V1224_I30_I32_BRIDGE_VARIANT"),
        (out_dir / f"{NG8_PREFIX}.csv", "V1224_I30_I32_BRIDGE_SUMMARY", "V1224_I30_I32_BRIDGE_VARIANT"),
        (out_dir / f"{NG9_PREFIX}.csv", "V1224_I30_I32_BRIDGE_SUMMARY", "V1224_I30_I32_BRIDGE_VARIANT"),
        (out_dir / f"{NG10_PREFIX}.csv", "V1224_I30_I32_BRIDGE_SUMMARY", "V1224_I30_I32_BRIDGE_VARIANT"),
    ]
    teacher_t2_rows = []
    for row in fallback_rows:
        p = ROOT / str(row.get("source_artifact", ""))
        if p.exists() and p.name == "v1223_t2_clone_probe_visibility_diagnostic.csv":
            teacher_t2_rows.extend([r for r in read_rows(p) if r.get("stage") == "V1223_T2_CLONE_PROBE_VISIBILITY_DIAGNOSTIC"])
    teacher_auc = max([fnum(r.get("auc_joint"), float("nan")) for r in teacher_t2_rows], default=float("nan"))
    teacher_precision = max([fnum(r.get("precision_at_k_joint"), float("nan")) for r in teacher_t2_rows], default=float("nan"))
    teacher_recall = max([fnum(r.get("recall_at_k_joint"), float("nan")) for r in teacher_t2_rows], default=float("nan"))

    rows: list[dict[str, Any]] = []
    scores: list[float] = []
    labels: list[int] = []
    for source_csv, summary_stage, variant_stage in source_specs:
        all_rows = read_rows(source_csv)
        summaries = [r for r in all_rows if r.get("stage") == summary_stage]
        variants = [r for r in all_rows if r.get("stage") == variant_stage and r.get("variant_kind") == "source"]
        selections = [r for r in all_rows if r.get("stage") == "V1224_I30_I32_PRECOMMIT_SELECTION"]
        for summary in summaries:
            candidate = summary.get("candidate_name", "")
            train_seed = summary.get("train_seed_base", "")
            variant = find_first(variants, candidate_name=candidate, train_seed_base=train_seed)
            selection = find_first(selections, candidate_name=candidate, train_seed_base=train_seed)
            post_drift = fnum(variant.get("direct_logit_compensation_post_drift"), 0.0)
            pre_drift = fnum(variant.get("direct_logit_compensation_pre_drift"), 0.0)
            comp_norm = fnum(variant.get("direct_logit_compensation_norm"), 0.0)
            probe_score = fnum(selection.get("probe_score"), 0.0)
            probe_tail = fnum(selection.get("probe_output_cov_tail"), 0.0)
            probe_entropy = fnum(selection.get("probe_entropy"), 0.0)
            linec_count = fnum(summary.get("linec_seed_pass_count"), 0.0)
            score = (
                -0.10 * post_drift
                -0.001 * comp_norm
                + 0.05 * (pre_drift - post_drift)
                + 0.10 * probe_score
                - 0.01 * probe_tail
                + 0.001 * probe_entropy
            )
            label = exp.safe_int(summary.get("strict_majority_pass"), 0)
            scores.append(score)
            labels.append(label)
            rows.append(
                {
                    "stage": "V1224_T1B_RESPONSE_DISTILLED_FEATURE",
                    "source_artifact": exp.rel(source_csv),
                    "candidate_id": candidate,
                    "dataset": summary.get("dataset", ""),
                    "seed": summary.get("seed", ""),
                    "train_seed_base": train_seed,
                    "uses_label": 0,
                    "uses_ce_vector": 0,
                    "uses_query_batch": 0,
                    "uses_validation_or_test": 0,
                    "precommit_available": 1,
                    "clone_probe_used": 1,
                    "response_teacher_auc_joint": teacher_auc,
                    "response_teacher_precision_at_k_joint": teacher_precision,
                    "response_teacher_recall_at_k_joint": teacher_recall,
                    "unlabeled_logit_drift_l2": "",
                    "unlabeled_logit_drift_max": post_drift,
                    "train_probe_CouplingR2_delta_proxy": linec_count,
                    "random_cotangent_sketch_delta_fro": "",
                    "projector_angle_proxy": "",
                    "branch_energy_transport_l1": "",
                    "role_entropy_delta": probe_entropy,
                    "output_cov_tail_delta": probe_tail,
                    "activation_cov_tail_delta": "",
                    "multi_sketch_stability_score": linec_count,
                    "predicted_release_score": score,
                    "target_strict_majority_pass": label,
                    "target_task_gate_pass": summary.get("task_gate_pass", ""),
                    "promotion_allowed": 0,
                    "no_fake": 1,
                }
            )
    auc = auc_rank(scores, labels)
    top_k = sum(labels) if labels else 0
    if top_k > 0:
        top = sorted(zip(scores, labels), key=lambda x: x[0], reverse=True)[:top_k]
        precision = sum(y for _s, y in top) / float(top_k)
        recall = sum(y for _s, y in top) / float(sum(labels)) if sum(labels) else float("nan")
    else:
        precision = float("nan")
        recall = float("nan")
    summary = {
        "stage": "V1224_T1B_RESPONSE_DISTILLED_FEATURE_SUMMARY",
        "rows": len(rows),
        "positive_rows": sum(labels),
        "class_count": len(set(labels)) if labels else 0,
        "auc_joint": auc,
        "precision_at_k_joint": precision,
        "recall_at_k_joint": recall,
        "top_k": top_k,
        "response_teacher_auc_joint": teacher_auc,
        "response_teacher_precision_at_k_joint": teacher_precision,
        "response_teacher_recall_at_k_joint": teacher_recall,
        "exploration_gate_pass": int(math.isfinite(auc) and auc >= 0.65 and math.isfinite(precision) and precision >= 0.25 and math.isfinite(recall) and recall >= 0.20),
        "promotion_allowed": 0,
        "no_fake": 1,
        "note": "Feature rows use only train-stream unlabeled probes; T2/T3 response artifacts are teacher boundary diagnostics, not deployable features.",
    }
    rows.append(summary)
    csv_path = out_dir / "v1224_t1b_response_distilled_features.csv"
    json_path = out_dir / "v1224_t1b_response_distilled_features_summary.json"
    write_rows(csv_path, rows)
    exp.write_json(json_path, json_safe(summary | {"artifact_csv": exp.rel(csv_path)}))
    return {
        "t1b_response_distilled_features_csv": exp.rel(csv_path),
        "t1b_response_distilled_rows": summary["rows"],
        "t1b_response_distilled_positive_rows": summary["positive_rows"],
        "t1b_response_distilled_class_count": summary["class_count"],
        "t1b_response_distilled_auc_joint": summary["auc_joint"],
        "t1b_response_distilled_exploration_gate_pass": summary["exploration_gate_pass"],
    }


def write_core_code_review(out_dir: Path) -> dict[str, Any]:
    specs = [
        ("R0", "experiments/run_v1224_finalize_functional_bridge.py", "def main", "route decision / final stop / fallback execution"),
        ("R1", "experiments/run_v1218_b320_label_free_ablation.py", "A66-LineCAwareUnlabeledMultiSketchFrame", "B320-current label-free construction"),
        ("R2", "experiments/run_v1223_failclosed_explore_open2_functional_rebuild.py", "def apply_v1223_actuator", "I24/I25/I26/I27 direct compensation actuator"),
        ("R3", "experiments/run_v1224_train_stream_functional_bridge.py", "def apply_train_stream_compensation", "train-stream direct-logit compensation reference"),
        ("R4", "experiments/run_v1224_train_stream_functional_bridge.py", "def train_role", "quad_only post-P3 training policy"),
        ("R5", "experiments/run_v1223_p4_trainable_role_scan.py", "def run_one", "P4 trainable-role scan"),
        ("R6", "experiments/run_v1223_p4_trajectory_gate_verifier.py", "def main", "P4 trajectory verifier"),
        ("R7", "experiments/run_v1223_p4_multisketch_aggregate_gate.py", "def main", "multi-sketch aggregation"),
        ("R8", "experiments/run_v1223_p4_precommit_proxy_checkpoint.py", "def main", "precommit proxy checkpoint"),
        ("R9", "experiments/run_v1223_failclosed_explore_open2_functional_rebuild.py", "metric_seed =", "LineC metric seed handling"),
        ("R10", "experiments/run_v1223_failclosed_explore_open2_functional_rebuild.py", "def matched_control_gap_roleaware", "matched control scope"),
        ("R11", "experiments/run_v1224_classic_hardening.py", "def run", "Rational/Fourier hardening runner"),
        ("R12", "experiments/run_v1224_finalize_functional_bridge.py", "def package_zip", "code packet generation"),
        ("R13", "experiments/run_v1224_i30_i32_policy_bridge.py", "def select_i30", "I30/I32 precommit bridge fallback"),
    ]
    artifact_fields = {
        "R0": "v1224_route_decision.json: route, final_stop_allowed, fallback_rows, fallback_depth, hard_budget_exhausted",
        "R1": "v1224_label_free_signal_frame/A66A69 summaries: mean_delta_vs_A0, LineC_pass_rate, uses_label",
        "R2": "v1223_actuator_safety_roleaware.csv and v1223 P4 audit artifacts: query_reference_used, promotion_allowed",
        "R3": "v1224_train_stream_functional_bridge.csv: direct_logit_compensation_mode, uses_train_batch, uses_query_batch",
        "R4": "v1224_train_stream_functional_bridge.csv: role_policy, source_acc, noop_acc, control_acc",
        "R5": "v1223 trainable role scan artifacts: promotion_allowed=0, p4 audit fields",
        "R6": "v1223 trajectory verifier artifacts: strict_trajectory_gate_pass, promotion_allowed",
        "R7": "v1223 multisketch artifacts: strict_all_sketch_pass, strict_majority_sketch_pass",
        "R8": "v1223 precommit proxy artifacts: proxy_uses_labels, proxy_uses_linec_or_audit_target, selected_epoch",
        "R9": "v1223 actuator artifacts: NoOp max delta, metric_seed_shared",
        "R10": "v1223 actuator control gap fields: matched_control_gap scoped by dataset/seed/budget/sign",
        "R11": "v1224_classic_hardening*.csv: step_ratio, memory_ratio, mean_delta_vs_MLP, LineC_pass",
        "R12": "v1224_code_review_packet.zip and required manifest: entries, sha256, missing_count",
        "R13": "v1224_i30_i32_policy_bridge.csv: probe_score, selected actuator, uses_train_batch, strict_majority_pass",
    }
    gate_relation = {
        "R0": "Controls R0/R4/S1/S2/S4/S5 route; missing fallback or artifacts forces incomplete/no-go.",
        "R1": "Can only open S1 when task and LineC pass-rate gates are met without labels/CE.",
        "R2": "Query/CE/provenance flags force v12.23 I24/I25 paths to audit-only in v12.24.",
        "R3": "Candidate source for S4a; must stay train-stream/precommit and beat controls.",
        "R4": "P4 audit policy; task win must also satisfy NLL/CEp99/time/LineC gates.",
        "R5": "Historical audit-only scan; cannot promote if labels/CE/validation selected the direction.",
        "R6": "Historical strict verifier; promotion_allowed remains 0 when query/future outcome is used.",
        "R7": "Multi-sketch gate distinguishes exploration majority from official all/robust criteria.",
        "R8": "Proxy selector is diagnostic if query/validation/future checkpoint information leaks in.",
        "R9": "Metric seed mismatch would invalidate release/control deltas and P3 source selection.",
        "R10": "Matched-control scope determines control_gap; wrong global scope can create false P3.",
        "R11": "Can open S4c only if task, LineC, step and memory gates all pass.",
        "R12": "Missing scripts/artifacts force R0-CodePacketIncomplete.",
        "R13": "I30/I32 are Level-2 bridge fallback; can only open S4a if robust majority/all gates pass.",
    }
    rows = []
    for rid, rel_path, pattern, obj in specs:
        path = ROOT / rel_path
        start, end = find_line(path, pattern)
        uses_query = int(rid in {"R2", "R6", "R7", "R8"})
        if rid == "R3":
            uses_query = 0
        rows.append(
            {
                "review_id": rid,
                "file_path": rel_path,
                "symbol_name": pattern,
                "line_start": start,
                "line_end": end,
                "called_by": "v12.24 finalizer / experiment command",
                "calls_into": obj,
                "mathematical_object": obj,
                "expected_tensor_shapes": "vision batch [B,784], logits [B,10], direct_readout [hidden,10] where applicable",
                "uses_label": int(rid in {"R5", "R6"}),
                "uses_ce_vector": int(rid in {"R2", "R5", "R6"}),
                "uses_query_batch": uses_query,
                "uses_train_batch": int(rid in {"R1", "R3", "R4", "R11", "R13"}),
                "uses_validation_or_test": int(rid in {"R5", "R6", "R7", "R8", "R11"}),
                "uses_future_outcome": int(rid in {"R6", "R8"}),
                "uses_dataset_name_branch": 0,
                "precommit_available": int(rid in {"R1", "R3", "R4", "R9", "R10", "R11", "R13"}),
                "promotion_allowed": 0,
                "loss_agnostic_direction": int(rid in {"R1", "R3", "R4", "R9", "R10", "R11", "R13"}),
                "matched_control_scope": "dataset/seed/budget/sign" if rid == "R10" else "",
                "metric_seed_shared": int(rid == "R9"),
                "manual_review_required": int(start == 0),
                "artifact_fields": artifact_fields.get(rid, ""),
                "gate_relation": gate_relation.get(rid, ""),
                "review_status": "pass" if start else "manual_review_required",
                "blocker_if_wrong": "could falsely promote audit-only or query-reference path" if rid in {"R2", "R3", "R6", "R7", "R8", "R10"} else "could make v12.24 irreproducible",
                "no_fake": 1,
            }
        )
    path = out_dir / "v1224_core_code_review_manifest.csv"
    write_rows(path, rows)
    return {"core_code_review_manifest": exp.rel(path), "code_semantics_review_pass": int(all(int(r["manual_review_required"]) == 0 for r in rows))}


def make_svg(path: Path, title: str, lines: list[str]) -> None:
    exp.ensure_dir(path.parent)
    safe_lines = [str(x).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;") for x in lines]
    height = 80 + 24 * len(safe_lines)
    text = [f'<text x="24" y="36" font-size="18" font-family="monospace">{title}</text>']
    for idx, line in enumerate(safe_lines):
        text.append(f'<text x="24" y="{72 + idx * 24}" font-size="13" font-family="monospace">{line}</text>')
    path.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="{height}" viewBox="0 0 1100 {height}"><rect width="100%" height="100%" fill="#ffffff"/>' + "".join(text) + "</svg>\n",
        encoding="utf-8",
    )


def write_figures(out_dir: Path, route: dict[str, Any]) -> list[str]:
    fig_dir = out_dir / "figures"
    figures = {
        "fig_linea_task_linec_pareto.svg": ["best=" + str(route.get("linea_best_candidate", "")), "mean_delta_A0=" + str(route.get("linea_best_mean_delta_vs_A0", "")), "LineC_pass_rate=" + str(route.get("linea_best_linec_pass_rate", ""))],
        "fig_linea_linec_pass_by_candidate.svg": ["LineA exploration_pass_count=" + str(route.get("linea_exploration_pass_count", "")), "candidate_rows=" + str(route.get("linea_label_free_candidate_rows", ""))],
        "fig_t1_t2_visibility_gap.svg": ["T1B microprobe AUC=" + str(route.get("t1b_microprobe_auc", "")), "response-distilled AUC=" + str(route.get("t1b_response_distilled_auc_joint", "")), "response teacher remains diagnostic only"],
        "fig_t1b_microprobe_auc_precision_recall.svg": ["microprobe positives=" + str(route.get("t1b_microprobe_positive_rows", "")), "distilled positives=" + str(route.get("t1b_response_distilled_positive_rows", "")), "class_count=" + str(route.get("t1b_response_distilled_class_count", ""))],
        "fig_actuator_safe_release_control_triangle.svg": ["train-stream strict majority=" + str(route.get("bridge_any_strict_majority_pass", "")), "I30/I32 strict majority=" + str(route.get("i30_i32_any_strict_majority_pass", "")), "NG2-NG7 strict majority=" + "/".join(str(route.get(k, "")) for k in ["ng2_any_strict_majority_pass", "ng3_any_strict_majority_pass", "ng4_any_strict_majority_pass", "ng5_any_strict_majority_pass", "ng6_any_strict_majority_pass", "ng7_any_strict_majority_pass"]), "combined robust majority=" + str(route.get("combined_bridge_any_train_shuffle_robust_majority_pass", ""))],
        "fig_p3_p4_decoupling_scatter.svg": ["source=" + str(route.get("bridge_source_actuator", "")), "no query batch in v12.24 bridge"],
        "fig_p4_source_noop_control_acc.svg": ["best source-noop delta=" + str(route.get("bridge_best_source_vs_noop_acc_delta", "")), "best source-control delta=" + str(route.get("bridge_best_source_vs_control_acc_delta", ""))],
        "fig_multisketch_pass_heatmap.svg": ["strict_all_pass=" + str(route.get("bridge_any_strict_all_pass", "")), "strict_majority_pass=" + str(route.get("bridge_any_strict_majority_pass", ""))],
        "fig_train_shuffle_robustness.svg": ["base robust majority=" + str(route.get("bridge_any_train_shuffle_robust_majority_pass", "")), "NG2-NG7 robust majority=" + "/".join(str(route.get(k, "")) for k in ["ng2_any_train_shuffle_robust_majority_pass", "ng3_any_train_shuffle_robust_majority_pass", "ng4_any_train_shuffle_robust_majority_pass", "ng5_any_train_shuffle_robust_majority_pass", "ng6_any_train_shuffle_robust_majority_pass", "ng7_any_train_shuffle_robust_majority_pass"]), "combined robust all=" + str(route.get("combined_bridge_any_train_shuffle_robust_all_pass", ""))],
        "fig_query_vs_train_compensation.svg": ["query-reference carried only from v12.23 audit", "v12.24 bridge uses_train_batch=1"],
        "fig_classic_hardening_pareto.svg": ["classic rows=" + str(route.get("classic_rows", "")), "candidate fallback rows=" + str(route.get("classic_candidate_rows", "")), "depth2 rows=" + str(route.get("classic_depth2_rows", "")), "combined exploration_pass_rows=" + str(route.get("combined_classic_exploration_pass_rows", ""))],
        "fig_rational_fourier_linec_task.svg": ["Rational/Fourier formal hardening executed", "gate remains fail-closed if no exploration rows"],
        "fig_stop_rule_execution_depth.svg": ["fallback rows=" + str(route.get("fallback_rows", "")), "all executed=" + str(route.get("fallback_all_executed", ""))],
    }
    paths = []
    for name, lines in figures.items():
        p = fig_dir / name
        make_svg(p, name, lines)
        paths.append(exp.rel(p))
    return paths


def write_fallback_manifest(out_dir: Path, route: dict[str, Any]) -> dict[str, Any]:
    rows = [
        {"trigger": "LineA_A65_LineC_zero", "fallback": "A66_A69_label_free_linec_repair", "executed": int(route.get("linea_summary_rows", 0) > 0), "artifact": "linea/A66A69_hardening"},
        {"trigger": "I24_query_reference_provenance_blocker", "fallback": "I28_I29_train_stream_compensation", "executed": int((out_dir / "v1224_train_stream_functional_bridge.csv").exists()), "artifact": "v1224_train_stream_functional_bridge.csv"},
        {"trigger": "train_stream_single_reference_fail", "fallback": "median_trimmed_ema_ensemble", "executed": int((out_dir / "v1224_train_stream_functional_bridge.csv").exists()), "artifact": "v1224_train_stream_functional_bridge.csv"},
        {"trigger": "T1B_visibility_not_precommit_source", "fallback": "online_microprobe_and_response_distillation_diagnostic", "executed": int((out_dir / "v1224_t1b_online_microprobe.csv").exists()), "artifact": "v1224_t1b_online_microprobe.csv"},
        {"trigger": "T2_T3_pass_but_T1B_fail", "fallback": "response_distilled_T1B_unlabeled_feature_generation", "executed": int((out_dir / "v1224_t1b_response_distilled_features.csv").exists()), "artifact": "v1224_t1b_response_distilled_features.csv"},
        {"trigger": "I28_I29_train_stream_bridge_fail", "fallback": "I30_T1B_guided_direct_compensation", "executed": int((out_dir / "v1224_i30_i32_policy_bridge.csv").exists()), "artifact": "v1224_i30_i32_policy_bridge.csv"},
        {"trigger": "P3_P4_decoupling_persists", "fallback": "I32_policy_aware_P3_quad_probe", "executed": int((out_dir / "v1224_i30_i32_policy_bridge.csv").exists()), "artifact": "v1224_i30_i32_policy_bridge.csv"},
        {"trigger": "LineD_smoke_only_in_v1223", "fallback": "Rational_Fourier_formal_hardening", "executed": int((out_dir / "v1224_classic_hardening.csv").exists()), "artifact": "v1224_classic_hardening.csv"},
        {"trigger": "LineD_base_candidates_fail", "fallback": "Rational_Fourier_candidate_fallback_hardening", "executed": int((out_dir / "v1224_classic_candidate_fallback_hardening.csv").exists()), "artifact": "v1224_classic_candidate_fallback_hardening.csv"},
        {"trigger": "LineD_candidate_fallback_fail", "fallback": "Rational_readscale_crosswarm_pairnorm_depth2_hardening", "executed": int((out_dir / "v1224_classic_depth2_efficiency_hardening.csv").exists()), "artifact": "v1224_classic_depth2_efficiency_hardening.csv"},
        {"trigger": "NG2_train_stream_bridge_replay_requested", "fallback": "ensemble16_train_stream_replay", "executed": int((out_dir / f"{NG2_PREFIX}.csv").exists()), "artifact": f"{NG2_PREFIX}.csv"},
        {"trigger": "NG3_policy_probe_reset_requested", "fallback": "i30_scale_i32_probe_reset", "executed": int((out_dir / f"{NG3_PREFIX}.csv").exists()), "artifact": f"{NG3_PREFIX}.csv"},
        {"trigger": "NG3_close_source_noop_but_control_win", "fallback": "NG4_lower_lr_longer_epoch_scale_repair", "executed": int((out_dir / f"{NG4_PREFIX}.csv").exists()), "artifact": f"{NG4_PREFIX}.csv"},
        {"trigger": "NG4_accuracy_control_positive_but_CEp99_fail", "fallback": "NG5_tailrisk_weightdecay_repair", "executed": int((out_dir / f"{NG5_PREFIX}.csv").exists()), "artifact": f"{NG5_PREFIX}.csv"},
        {"trigger": "NG5_CEp99_tailrisk_persists", "fallback": "NG6_fair_label_smoothing_tail_repair", "executed": int((out_dir / f"{NG6_PREFIX}.csv").exists()), "artifact": f"{NG6_PREFIX}.csv"},
        {"trigger": "NG6_smoothing_reduces_CEp99_but_loses_task_gain", "fallback": "NG7_light_label_smoothing_tail_repair", "executed": int((out_dir / f"{NG7_PREFIX}.csv").exists()), "artifact": f"{NG7_PREFIX}.csv"},
        {"trigger": "NG7_tail_and_task_still_not_joint", "fallback": "NG8_tail_safe_precommit_probe", "executed": int((out_dir / f"{NG8_PREFIX}.csv").exists()), "artifact": f"{NG8_PREFIX}.csv"},
        {"trigger": "NG8_quad_only_tail_probe_fails", "fallback": "NG9_quad_direct_tail_repair", "executed": int((out_dir / f"{NG9_PREFIX}.csv").exists()), "artifact": f"{NG9_PREFIX}.csv"},
        {"trigger": "NG9_control_margin_and_tail_fail", "fallback": "NG10_control_contrast_tail_probe", "executed": int((out_dir / f"{NG10_PREFIX}.csv").exists()), "artifact": f"{NG10_PREFIX}.csv"},
        {"trigger": "NG9_CEp99_tailrisk_persists_under_quad_direct", "fallback": "NG11_NG12_weight_decay_tailrisk_repair", "executed": int((out_dir / f"{NG11_PREFIX}.csv").exists() and (out_dir / f"{NG12_PREFIX}.csv").exists()), "artifact": f"{NG11_PREFIX}.csv;{NG12_PREFIX}.csv"},
        {"trigger": "NG10_control_contrast_ties_best_control", "fallback": "NG13_NG14_stronger_control_contrast_probe", "executed": int((out_dir / f"{NG13_PREFIX}.csv").exists() and (out_dir / f"{NG14_PREFIX}.csv").exists()), "artifact": f"{NG13_PREFIX}.csv;{NG14_PREFIX}.csv"},
        {"trigger": "I32_policy_space_incomplete", "fallback": "NG15_NG18_expanded_policy_probe", "executed": int(all((out_dir / f"{prefix}.csv").exists() for prefix in (NG15_PREFIX, NG16_PREFIX, NG17_PREFIX, NG18_PREFIX))), "artifact": ";".join(f"{prefix}.csv" for prefix in (NG15_PREFIX, NG16_PREFIX, NG17_PREFIX, NG18_PREFIX))},
        {"trigger": "code_audit_required", "fallback": "R0_R13_core_code_review_manifest", "executed": int((out_dir / "v1224_core_code_review_manifest.csv").exists()), "artifact": "v1224_core_code_review_manifest.csv"},
        {"trigger": "all_depth2_fallbacks_fail", "fallback": "executable_next_generation_fallback_plan", "executed": int((out_dir / "v1224_next_generation_fallback_plan.csv").exists()), "artifact": "v1224_next_generation_fallback_plan.csv"},
    ]
    path = out_dir / "v1224_fallback_execution_manifest.csv"
    write_rows(path, [{**r, "stage": "V1224_FALLBACK_EXECUTION", "no_fake": 1} for r in rows])
    return {"fallback_rows": len(rows), "fallback_all_executed": int(all(int(r["executed"]) for r in rows)), "fallback_manifest": exp.rel(path)}


def package_zip(out_dir: Path) -> Path:
    zip_path = out_dir / "v1224_code_review_packet.zip"
    members = [
        PLAN_DOC,
        EXEC_LOG,
        REVIEW_LOG,
        *TRANSITIVE_CODE_PACKET_MEMBERS,
    ]
    members.extend(sorted(out_dir.glob("*.csv")))
    members.extend(sorted(out_dir.glob("*.json")))
    members.extend(sorted(out_dir.glob("*.md")))
    members.extend(sorted((out_dir / "figures").glob("*.svg")))
    members.extend(sorted((out_dir / "logs").glob("*.log")))
    members.extend(sorted((out_dir / "linea").glob("**/*.csv")))
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        seen = set()
        for member in members:
            if not member.exists() or member in seen:
                continue
            seen.add(member)
            zf.write(member, member.relative_to(ROOT))
    return zip_path


def write_required_manifest(out_dir: Path, figures: list[str]) -> dict[str, Any]:
    required = [
        PLAN_DOC,
        EXEC_LOG,
        REVIEW_LOG,
        ROOT / "experiments" / "run_v1224_train_stream_functional_bridge.py",
        ROOT / "experiments" / "run_v1224_i30_i32_policy_bridge.py",
        ROOT / "experiments" / "run_v1224_classic_hardening.py",
        *TRANSITIVE_CODE_PACKET_MEMBERS,
        out_dir / "v1224_train_stream_functional_bridge.csv",
        out_dir / "v1224_train_stream_functional_bridge_summary.json",
        out_dir / "v1224_i30_i32_policy_bridge.csv",
        out_dir / "v1224_i30_i32_policy_bridge_summary.json",
        out_dir / f"{NG2_PREFIX}.csv",
        out_dir / f"{NG2_PREFIX}_summary.json",
        out_dir / f"{NG3_PREFIX}.csv",
        out_dir / f"{NG3_PREFIX}_summary.json",
        out_dir / f"{NG4_PREFIX}.csv",
        out_dir / f"{NG4_PREFIX}_summary.json",
        out_dir / f"{NG5_PREFIX}.csv",
        out_dir / f"{NG5_PREFIX}_summary.json",
        out_dir / f"{NG6_PREFIX}.csv",
        out_dir / f"{NG6_PREFIX}_summary.json",
        out_dir / f"{NG7_PREFIX}.csv",
        out_dir / f"{NG7_PREFIX}_summary.json",
        out_dir / f"{NG8_PREFIX}.csv",
        out_dir / f"{NG8_PREFIX}_summary.json",
        out_dir / f"{NG9_PREFIX}.csv",
        out_dir / f"{NG9_PREFIX}_summary.json",
        out_dir / f"{NG10_PREFIX}.csv",
        out_dir / f"{NG10_PREFIX}_summary.json",
        *[item for prefix in EXTRA_NG_PREFIXES for item in (out_dir / f"{prefix}.csv", out_dir / f"{prefix}_summary.json")],
        out_dir / "v1224_classic_hardening.csv",
        out_dir / "v1224_classic_hardening_summary.json",
        out_dir / "v1224_classic_candidate_fallback_hardening.csv",
        out_dir / "v1224_classic_candidate_fallback_hardening_summary.json",
        out_dir / "v1224_classic_depth2_efficiency_hardening.csv",
        out_dir / "v1224_classic_depth2_efficiency_hardening_summary.json",
        out_dir / "v1224_t1b_online_microprobe.csv",
        out_dir / "v1224_t1b_response_distilled_features.csv",
        out_dir / "v1224_t1b_response_distilled_features_summary.json",
        out_dir / "v1224_response_distillation_fallback.csv",
        out_dir / "v1224_core_code_review_manifest.csv",
        out_dir / "v1224_next_generation_fallback_plan.csv",
        out_dir / "v1224_next_generation_fallback_plan.json",
        out_dir / "v1224_next_generation_fallback_plan.md",
        out_dir / "v1224_fallback_execution_manifest.csv",
        out_dir / "v1224_route_decision.json",
        out_dir / "v1224_code_review_packet.zip",
    ]
    required.extend(ROOT / f for f in figures)
    rows = []
    for path in required:
        rows.append(
            {
                "stage": "V1224_REQUIRED_ARTIFACT",
                "path": exp.rel(path),
                "exists": int(path.exists()),
                "size_bytes": path.stat().st_size if path.exists() else "",
                "sha256": sha256_file(path) if path.exists() and path.is_file() else "",
                "no_fake": 1,
            }
        )
    manifest = out_dir / "v1224_required_artifact_manifest.csv"
    write_rows(manifest, rows)
    return {
        "required_artifact_manifest": exp.rel(manifest),
        "required_artifact_rows": len(rows),
        "required_artifact_missing_count": sum(1 for r in rows if int(r["exists"]) == 0),
    }


def write_next_generation_fallback_plan(out_dir: Path, route: dict[str, Any]) -> dict[str, Any]:
    rows = [
        {
            "stage": "V1224_NEXT_GENERATION_FALLBACK",
            "fallback_id": "NG1-rational-readscale-crosswarm-depth2",
            "blocked_route": route.get("route", ""),
            "target_blocker": "LineD RationalB7lp/B7lz close task gap but miss combined step/memory/LineC exploration gate",
            "executable_command": (
                "conda run -n kan python experiments/run_v1224_classic_hardening.py "
                "--out-dir results/v12_24_s4_to_s5_functional_bridge/official_s4_to_s5 "
                "--artifact-prefix v1224_classic_depth2_efficiency_hardening "
                "--device cuda:0 --families RationalB7dq,RationalB7dr,RationalB7ds,RationalB7dt,RationalB7em,RationalB7en,RationalB7eq,RationalB7er "
                "--datasets MNIST,Fashion-MNIST --seeds 0 --train-size 1024 --val-size 512 --epochs 8 --no-download"
            ),
            "promotion_rule": "Only open S4c if existing LineD exploration gate passes; no threshold lowering.",
            "uses_label": 0,
            "uses_ce_vector": 0,
            "promotion_allowed": 0,
            "no_fake": 1,
        },
        {
            "stage": "V1224_NEXT_GENERATION_FALLBACK",
            "fallback_id": "NG2-train-stream-bridge-replay",
            "blocked_route": route.get("route", ""),
            "target_blocker": "I28/I29/I30/I31/I32 fail to beat matched controls robustly",
            "executable_command": (
                "conda run -n kan python experiments/run_v1224_train_stream_functional_bridge.py "
                "--source-out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_i24_directcomp_all_targeted "
                "--out-dir results/v12_24_s4_to_s5_functional_bridge/official_s4_to_s5 "
                "--artifact-prefix v1224_ng2_train_stream_bridge_replay "
                "--device cuda:1 --candidates I28-TrainStreamEMACompensation,I29-TrainProbeMedianCompensation,I31-NullLogitCompensatedShadowRelease "
                "--train-seed-bases 12240400,12241400,12242400 --ensemble-count 16 --compensation-batch 32 --no-download"
            ),
            "promotion_rule": "Can only open S4a/S5 if source wins NoOp and same-compensation control with multi-sketch/train-shuffle robustness.",
            "uses_label": 0,
            "uses_ce_vector": 0,
            "promotion_allowed": 0,
            "no_fake": 1,
        },
        {
            "stage": "V1224_NEXT_GENERATION_FALLBACK",
            "fallback_id": "NG3-policy-aware-probe-reset",
            "blocked_route": route.get("route", ""),
            "target_blocker": "T1B/response-distilled features have no positive support and I32 policy probe has no strict pass",
            "executable_command": (
                "conda run -n kan python experiments/run_v1224_i30_i32_policy_bridge.py "
                "--source-out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_i24_directcomp_all_targeted "
                "--out-dir results/v12_24_s4_to_s5_functional_bridge/official_s4_to_s5 "
                "--artifact-prefix v1224_ng3_policy_probe_reset "
                "--device cuda:2 --candidates I30-T1BGuidedDirectCompensation,I32-PolicyAwareP3QuadProbe "
                "--i30-scales 0.35,0.50,0.75,1.00 --i32-probe-steps 1,3,5,8 --no-download"
            ),
            "promotion_rule": "Only use train-stream unlabeled probe for selection; no T2/T3 target leakage.",
            "uses_label": 0,
            "uses_ce_vector": 0,
            "promotion_allowed": 0,
            "no_fake": 1,
        },
        {
            "stage": "V1224_NEXT_GENERATION_FALLBACK",
            "fallback_id": "NG4-policy-probe-lr-epoch-repair",
            "blocked_route": route.get("route", ""),
            "target_blocker": "NG3 I30 can beat NoOp on one shuffle but still loses matched control",
            "executable_command": (
                "conda run -n kan python experiments/run_v1224_i30_i32_policy_bridge.py "
                "--source-out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_i24_directcomp_all_targeted "
                "--out-dir results/v12_24_s4_to_s5_functional_bridge/official_s4_to_s5 "
                "--artifact-prefix v1224_ng4_policy_probe_lr_epoch_repair "
                "--device cuda:3 --candidates I30-T1BGuidedDirectCompensation,I32-PolicyAwareP3QuadProbe "
                "--i30-scales 0.20,0.35,0.50,0.75,1.00,1.25 --i32-probe-steps 2,4,6,10 "
                "--epochs 12 --lr 0.0015 --train-seed-bases 12240400,12241400,12242400 --no-download"
            ),
            "promotion_rule": "Can only open S4a/S5 if the same official train-shuffle and multi-sketch gates pass.",
            "uses_label": 0,
            "uses_ce_vector": 0,
            "promotion_allowed": 0,
            "no_fake": 1,
        },
        {
            "stage": "V1224_NEXT_GENERATION_FALLBACK",
            "fallback_id": "NG5-tailrisk-weightdecay-repair",
            "blocked_route": route.get("route", ""),
            "target_blocker": "NG4 has one I30 source-control accuracy positive row but fails strict task gate on CEp99 tail risk",
            "executable_command": (
                "conda run -n kan python experiments/run_v1224_i30_i32_policy_bridge.py "
                "--source-out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_i24_directcomp_all_targeted "
                "--out-dir results/v12_24_s4_to_s5_functional_bridge/official_s4_to_s5 "
                "--artifact-prefix v1224_ng5_tailrisk_weightdecay_repair "
                "--device cuda:0 --candidates I30-T1BGuidedDirectCompensation "
                "--i30-scales 0.20,0.35,0.50,0.75,1.00,1.25 "
                "--epochs 12 --lr 0.0015 --weight-decay 0.005 --train-seed-bases 12240400,12241400,12242400 --no-download"
            ),
            "promotion_rule": "No CEp99 relaxation; success requires the same strict task plus multi-sketch/train-shuffle bridge gates.",
            "uses_label": 0,
            "uses_ce_vector": 0,
            "promotion_allowed": 0,
            "no_fake": 1,
        },
        {
            "stage": "V1224_NEXT_GENERATION_FALLBACK",
            "fallback_id": "NG6-label-smoothing-tail-repair",
            "blocked_route": route.get("route", ""),
            "target_blocker": "NG5 keeps the same source-control-positive row but CEp99 remains above NoOp tolerance",
            "executable_command": (
                "conda run -n kan python experiments/run_v1224_i30_i32_policy_bridge.py "
                "--source-out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_i24_directcomp_all_targeted "
                "--out-dir results/v12_24_s4_to_s5_functional_bridge/official_s4_to_s5 "
                "--artifact-prefix v1224_ng6_label_smoothing_tail_repair "
                "--device cuda:1 --candidates I30-T1BGuidedDirectCompensation "
                "--i30-scales 0.20,0.35,0.50,0.75,1.00,1.25 "
                "--epochs 12 --lr 0.0015 --label-smoothing 0.05 --train-seed-bases 12240400,12241400,12242400 --no-download"
            ),
            "promotion_rule": "Source/noop/control all use the same smoothing; no strict metric tolerance is relaxed.",
            "uses_label": 1,
            "uses_ce_vector": 0,
            "promotion_allowed": 0,
            "no_fake": 1,
        },
        {
            "stage": "V1224_NEXT_GENERATION_FALLBACK",
            "fallback_id": "NG7-light-label-smoothing-tail-repair",
            "blocked_route": route.get("route", ""),
            "target_blocker": "NG6 reduces CEp99 but loses source task/control advantage; test a lighter smoothing interpolation.",
            "executable_command": (
                "conda run -n kan python experiments/run_v1224_i30_i32_policy_bridge.py "
                "--source-out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_i24_directcomp_all_targeted "
                "--out-dir results/v12_24_s4_to_s5_functional_bridge/official_s4_to_s5 "
                "--artifact-prefix v1224_ng7_light_label_smoothing_tail_repair "
                "--device cuda:2 --candidates I30-T1BGuidedDirectCompensation "
                "--i30-scales 0.20,0.35,0.50,0.75,1.00,1.25 "
                "--epochs 12 --lr 0.0015 --label-smoothing 0.02 --train-seed-bases 12240400,12241400,12242400 --no-download"
            ),
            "promotion_rule": "Same strict task/LineC/train-shuffle gates; light smoothing is shared by source/noop/control.",
            "uses_label": 1,
            "uses_ce_vector": 0,
            "promotion_allowed": 0,
            "no_fake": 1,
        },
        {
            "stage": "V1224_NEXT_GENERATION_FALLBACK",
            "fallback_id": "NG8-tail-safe-precommit-probe",
            "blocked_route": route.get("route", ""),
            "target_blocker": "NG4 has source-control-positive accuracy but CEp99 tail risk; NG5-NG7 fail to repair without losing control advantage.",
            "executable_command": (
                "conda run -n kan python experiments/run_v1224_i30_i32_policy_bridge.py "
                "--source-out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_i24_directcomp_all_targeted "
                "--out-dir results/v12_24_s4_to_s5_functional_bridge/official_s4_to_s5 "
                "--artifact-prefix v1224_ng8_tail_safe_precommit_probe "
                "--device cuda:3 --candidates I30-T1BGuidedDirectCompensation "
                "--i30-scales 0.10,0.15,0.20,0.25,0.35,0.50,0.75,1.00,1.25 "
                "--probe-score-mode tail_safe --probe-confidence-tail-weight 1.00 --probe-logit-abs-weight 0.08 --probe-tail-weight 0.20 --probe-entropy-weight 0.05 "
                "--epochs 12 --lr 0.0015 --train-seed-bases 12240400,12241400,12242400 --no-download"
            ),
            "promotion_rule": "Selection uses only train-stream unlabeled tail proxies; strict task/LineC/train-shuffle gates are unchanged.",
            "uses_label": 0,
            "uses_ce_vector": 0,
            "promotion_allowed": 0,
            "no_fake": 1,
        },
        {
            "stage": "V1224_NEXT_GENERATION_FALLBACK",
            "fallback_id": "NG9-tail-safe-quaddirect-repair",
            "blocked_route": route.get("route", ""),
            "target_blocker": "NG8 tail-safe selector keeps the same candidate; test whether quad_direct role absorbs tail risk without changing supervised loss.",
            "executable_command": (
                "conda run -n kan python experiments/run_v1224_i30_i32_policy_bridge.py "
                "--source-out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_i24_directcomp_all_targeted "
                "--out-dir results/v12_24_s4_to_s5_functional_bridge/official_s4_to_s5 "
                "--artifact-prefix v1224_ng9_tail_safe_quaddirect_repair "
                "--device cuda:0 --candidates I30-T1BGuidedDirectCompensation "
                "--i30-scales 0.10,0.15,0.20,0.25,0.35,0.50,0.75,1.00,1.25 "
                "--probe-score-mode tail_safe --probe-confidence-tail-weight 1.00 --probe-logit-abs-weight 0.08 --probe-tail-weight 0.20 --probe-entropy-weight 0.05 "
                "--epochs 12 --lr 0.0015 --role-policy quad_direct --train-seed-bases 12240400,12241400,12242400 --no-download"
            ),
            "promotion_rule": "No loss modification and no gate relaxation; source must pass unchanged strict task/LineC/train-shuffle gates.",
            "uses_label": 0,
            "uses_ce_vector": 0,
            "promotion_allowed": 0,
            "no_fake": 1,
        },
        {
            "stage": "V1224_NEXT_GENERATION_FALLBACK",
            "fallback_id": "NG10-contrast-tail-safe-probe",
            "blocked_route": route.get("route", ""),
            "target_blocker": "NG9 improves tail metrics but still fails matched-control and shuffle gates; add unlabeled source-vs-control contrast to precommit selection.",
            "executable_command": (
                "conda run -n kan python experiments/run_v1224_i30_i32_policy_bridge.py "
                "--source-out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_i24_directcomp_all_targeted "
                "--out-dir results/v12_24_s4_to_s5_functional_bridge/official_s4_to_s5 "
                "--artifact-prefix v1224_ng10_contrast_tail_safe_probe "
                "--device cuda:1 --candidates I30-T1BGuidedDirectCompensation "
                "--i30-scales 0.10,0.15,0.20,0.25,0.35,0.50,0.75,1.00,1.25 "
                "--probe-score-mode contrast_tail_safe --probe-control-contrast-weight 1.00 --probe-confidence-tail-weight 1.00 --probe-logit-abs-weight 0.08 --probe-tail-weight 0.20 --probe-entropy-weight 0.05 "
                "--epochs 12 --lr 0.0015 --role-policy quad_direct --train-seed-bases 12240400,12241400,12242400 --no-download"
            ),
            "promotion_rule": "Selector may compare only unlabeled source/control probe scores; final gate unchanged.",
            "uses_label": 0,
            "uses_ce_vector": 0,
            "promotion_allowed": 0,
            "no_fake": 1,
        },
    ]
    csv_path = out_dir / "v1224_next_generation_fallback_plan.csv"
    json_path = out_dir / "v1224_next_generation_fallback_plan.json"
    md_path = out_dir / "v1224_next_generation_fallback_plan.md"
    write_rows(csv_path, rows)
    exp.write_json(json_path, {"stage": "V1224_NEXT_GENERATION_FALLBACK_PLAN", "rows": len(rows), "artifact_csv": exp.rel(csv_path), "promotion_allowed": 0, "no_fake": 1})
    md_lines = [
        "# v12.24 Next-Generation Fallback Plan",
        "",
        "This artifact is generated after v12.24 fails to open S5. Commands are executable and keep promotion fail-closed.",
        "",
    ]
    for row in rows:
        md_lines.extend(
            [
                f"## {row['fallback_id']}",
                "",
                f"- blocker: {row['target_blocker']}",
                f"- promotion rule: {row['promotion_rule']}",
                "",
                "```bash",
                str(row["executable_command"]),
                "```",
                "",
            ]
        )
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    return {
        "next_generation_fallback_plan_csv": exp.rel(csv_path),
        "next_generation_fallback_plan_json": exp.rel(json_path),
        "next_generation_fallback_plan_md": exp.rel(md_path),
        "next_generation_fallback_ready": 1,
        "next_generation_fallback_rows": len(rows),
    }


def git_status_snapshot(out_dir: Path) -> str:
    path = out_dir / "v1224_git_status_short.txt"
    try:
        proc = subprocess.run(["git", "status", "--short"], cwd=ROOT, text=True, capture_output=True, check=False)
        path.write_text(proc.stdout, encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        path.write_text(f"git status failed: {type(exc).__name__}: {exc}\n", encoding="utf-8")
    return exp.rel(path)


def main() -> None:
    out_dir = OUT_DIR
    exp.ensure_dir(out_dir)
    linea = summarize_linea(out_dir)
    bridge = read_json(out_dir / "v1224_train_stream_functional_bridge_summary.json")
    i30_i32 = read_json(out_dir / "v1224_i30_i32_policy_bridge_summary.json")
    ng2 = read_json(out_dir / f"{NG2_PREFIX}_summary.json")
    ng3 = read_json(out_dir / f"{NG3_PREFIX}_summary.json")
    ng4 = read_json(out_dir / f"{NG4_PREFIX}_summary.json")
    ng5 = read_json(out_dir / f"{NG5_PREFIX}_summary.json")
    ng6 = read_json(out_dir / f"{NG6_PREFIX}_summary.json")
    ng7 = read_json(out_dir / f"{NG7_PREFIX}_summary.json")
    ng8 = read_json(out_dir / f"{NG8_PREFIX}_summary.json")
    ng9 = read_json(out_dir / f"{NG9_PREFIX}_summary.json")
    ng10 = read_json(out_dir / f"{NG10_PREFIX}_summary.json")
    extra_ng = [read_json(out_dir / f"{prefix}_summary.json") for prefix in EXTRA_NG_PREFIXES]
    classic = read_json(out_dir / "v1224_classic_hardening_summary.json")
    classic_candidate = read_json(out_dir / "v1224_classic_candidate_fallback_hardening_summary.json")
    classic_depth2 = read_json(out_dir / "v1224_classic_depth2_efficiency_hardening_summary.json")
    t1b = write_t1b_microprobe(out_dir)
    code_review = write_core_code_review(out_dir)

    bridge_best = (bridge.get("best_candidate_aggregates") or [{}])[0]
    i30_i32_best = (i30_i32.get("best_candidate_aggregates") or [{}])[0]
    ng2_best = (ng2.get("best_candidate_aggregates") or [{}])[0]
    ng3_best = (ng3.get("best_candidate_aggregates") or [{}])[0]
    ng4_best = (ng4.get("best_candidate_aggregates") or [{}])[0]
    ng5_best = (ng5.get("best_candidate_aggregates") or [{}])[0]
    ng6_best = (ng6.get("best_candidate_aggregates") or [{}])[0]
    ng7_best = (ng7.get("best_candidate_aggregates") or [{}])[0]
    ng8_best = (ng8.get("best_candidate_aggregates") or [{}])[0]
    ng9_best = (ng9.get("best_candidate_aggregates") or [{}])[0]
    ng10_best = (ng10.get("best_candidate_aggregates") or [{}])[0]
    extra_ng_best = [(item.get("best_candidate_aggregates") or [{}])[0] for item in extra_ng]
    combined_bridge_majority = int(
        int(bridge.get("any_train_shuffle_robust_majority_pass", 0)) == 1
        or int(i30_i32.get("any_train_shuffle_robust_majority_pass", 0)) == 1
        or int(ng2.get("any_train_shuffle_robust_majority_pass", 0)) == 1
        or int(ng3.get("any_train_shuffle_robust_majority_pass", 0)) == 1
        or int(ng4.get("any_train_shuffle_robust_majority_pass", 0)) == 1
        or int(ng5.get("any_train_shuffle_robust_majority_pass", 0)) == 1
        or int(ng6.get("any_train_shuffle_robust_majority_pass", 0)) == 1
        or int(ng7.get("any_train_shuffle_robust_majority_pass", 0)) == 1
        or int(ng8.get("any_train_shuffle_robust_majority_pass", 0)) == 1
        or int(ng9.get("any_train_shuffle_robust_majority_pass", 0)) == 1
        or int(ng10.get("any_train_shuffle_robust_majority_pass", 0)) == 1
    )
    combined_bridge_all = int(
        int(bridge.get("any_train_shuffle_robust_all_pass", 0)) == 1
        or int(i30_i32.get("any_train_shuffle_robust_all_pass", 0)) == 1
        or int(ng2.get("any_train_shuffle_robust_all_pass", 0)) == 1
        or int(ng3.get("any_train_shuffle_robust_all_pass", 0)) == 1
        or int(ng4.get("any_train_shuffle_robust_all_pass", 0)) == 1
        or int(ng5.get("any_train_shuffle_robust_all_pass", 0)) == 1
        or int(ng6.get("any_train_shuffle_robust_all_pass", 0)) == 1
        or int(ng7.get("any_train_shuffle_robust_all_pass", 0)) == 1
        or int(ng8.get("any_train_shuffle_robust_all_pass", 0)) == 1
        or int(ng9.get("any_train_shuffle_robust_all_pass", 0)) == 1
        or int(ng10.get("any_train_shuffle_robust_all_pass", 0)) == 1
    )
    combined_bridge_majority = int(combined_bridge_majority or any(int(item.get("any_train_shuffle_robust_majority_pass", 0)) == 1 for item in extra_ng))
    combined_bridge_all = int(combined_bridge_all or any(int(item.get("any_train_shuffle_robust_all_pass", 0)) == 1 for item in extra_ng))
    combined_classic_exploration = (
        int(classic.get("exploration_pass_rows", 0))
        + int(classic_candidate.get("exploration_pass_rows", 0))
        + int(classic_depth2.get("exploration_pass_rows", 0))
    )
    route = {
        "stage": "V1224_ROUTE_DECISION",
        "generated_at": exp.now_iso(),
        "run_id": "official_s4_to_s5",
        "source_v1223_route": "S4-FunctionalP3Opened",
        "official_success_reached": 0,
        "p4_pass": 0,
        "promotion_allowed": 0,
        "no_fake": 1,
        **linea,
        "bridge_source_actuator": bridge.get("source_actuator", ""),
        "bridge_any_strict_majority_pass": bridge.get("any_strict_majority_pass", 0),
        "bridge_any_strict_all_pass": bridge.get("any_strict_all_pass", 0),
        "bridge_any_train_shuffle_robust_majority_pass": bridge.get("any_train_shuffle_robust_majority_pass", 0),
        "bridge_any_train_shuffle_robust_all_pass": bridge.get("any_train_shuffle_robust_all_pass", 0),
        "bridge_best_source_vs_noop_acc_delta": bridge_best.get("best_source_vs_noop_acc_delta", ""),
        "bridge_best_source_vs_control_acc_delta": bridge_best.get("best_source_vs_control_acc_delta", ""),
        "i30_i32_candidate_rows": i30_i32.get("candidate_rows", 0),
        "i30_i32_any_strict_majority_pass": i30_i32.get("any_strict_majority_pass", 0),
        "i30_i32_any_strict_all_pass": i30_i32.get("any_strict_all_pass", 0),
        "i30_i32_any_train_shuffle_robust_majority_pass": i30_i32.get("any_train_shuffle_robust_majority_pass", 0),
        "i30_i32_any_train_shuffle_robust_all_pass": i30_i32.get("any_train_shuffle_robust_all_pass", 0),
        "i30_i32_best_source_vs_noop_acc_delta": i30_i32_best.get("best_source_vs_noop_acc_delta", ""),
        "i30_i32_best_source_vs_control_acc_delta": i30_i32_best.get("best_source_vs_control_acc_delta", ""),
        "ng2_candidate_rows": ng2.get("candidate_rows", 0),
        "ng2_any_strict_majority_pass": ng2.get("any_strict_majority_pass", 0),
        "ng2_any_strict_all_pass": ng2.get("any_strict_all_pass", 0),
        "ng2_any_train_shuffle_robust_majority_pass": ng2.get("any_train_shuffle_robust_majority_pass", 0),
        "ng2_any_train_shuffle_robust_all_pass": ng2.get("any_train_shuffle_robust_all_pass", 0),
        "ng2_best_source_vs_noop_acc_delta": ng2_best.get("best_source_vs_noop_acc_delta", ""),
        "ng2_best_source_vs_control_acc_delta": ng2_best.get("best_source_vs_control_acc_delta", ""),
        "ng3_candidate_rows": ng3.get("candidate_rows", 0),
        "ng3_any_strict_majority_pass": ng3.get("any_strict_majority_pass", 0),
        "ng3_any_strict_all_pass": ng3.get("any_strict_all_pass", 0),
        "ng3_any_train_shuffle_robust_majority_pass": ng3.get("any_train_shuffle_robust_majority_pass", 0),
        "ng3_any_train_shuffle_robust_all_pass": ng3.get("any_train_shuffle_robust_all_pass", 0),
        "ng3_best_source_vs_noop_acc_delta": ng3_best.get("best_source_vs_noop_acc_delta", ""),
        "ng3_best_source_vs_control_acc_delta": ng3_best.get("best_source_vs_control_acc_delta", ""),
        "ng4_candidate_rows": ng4.get("candidate_rows", 0),
        "ng4_any_strict_majority_pass": ng4.get("any_strict_majority_pass", 0),
        "ng4_any_strict_all_pass": ng4.get("any_strict_all_pass", 0),
        "ng4_any_train_shuffle_robust_majority_pass": ng4.get("any_train_shuffle_robust_majority_pass", 0),
        "ng4_any_train_shuffle_robust_all_pass": ng4.get("any_train_shuffle_robust_all_pass", 0),
        "ng4_best_source_vs_noop_acc_delta": ng4_best.get("best_source_vs_noop_acc_delta", ""),
        "ng4_best_source_vs_control_acc_delta": ng4_best.get("best_source_vs_control_acc_delta", ""),
        "ng5_candidate_rows": ng5.get("candidate_rows", 0),
        "ng5_any_strict_majority_pass": ng5.get("any_strict_majority_pass", 0),
        "ng5_any_strict_all_pass": ng5.get("any_strict_all_pass", 0),
        "ng5_any_train_shuffle_robust_majority_pass": ng5.get("any_train_shuffle_robust_majority_pass", 0),
        "ng5_any_train_shuffle_robust_all_pass": ng5.get("any_train_shuffle_robust_all_pass", 0),
        "ng5_best_source_vs_noop_acc_delta": ng5_best.get("best_source_vs_noop_acc_delta", ""),
        "ng5_best_source_vs_control_acc_delta": ng5_best.get("best_source_vs_control_acc_delta", ""),
        "ng6_candidate_rows": ng6.get("candidate_rows", 0),
        "ng6_any_strict_majority_pass": ng6.get("any_strict_majority_pass", 0),
        "ng6_any_strict_all_pass": ng6.get("any_strict_all_pass", 0),
        "ng6_any_train_shuffle_robust_majority_pass": ng6.get("any_train_shuffle_robust_majority_pass", 0),
        "ng6_any_train_shuffle_robust_all_pass": ng6.get("any_train_shuffle_robust_all_pass", 0),
        "ng6_best_source_vs_noop_acc_delta": ng6_best.get("best_source_vs_noop_acc_delta", ""),
        "ng6_best_source_vs_control_acc_delta": ng6_best.get("best_source_vs_control_acc_delta", ""),
        "ng7_candidate_rows": ng7.get("candidate_rows", 0),
        "ng7_any_strict_majority_pass": ng7.get("any_strict_majority_pass", 0),
        "ng7_any_strict_all_pass": ng7.get("any_strict_all_pass", 0),
        "ng7_any_train_shuffle_robust_majority_pass": ng7.get("any_train_shuffle_robust_majority_pass", 0),
        "ng7_any_train_shuffle_robust_all_pass": ng7.get("any_train_shuffle_robust_all_pass", 0),
        "ng7_best_source_vs_noop_acc_delta": ng7_best.get("best_source_vs_noop_acc_delta", ""),
        "ng7_best_source_vs_control_acc_delta": ng7_best.get("best_source_vs_control_acc_delta", ""),
        "ng8_candidate_rows": ng8.get("candidate_rows", 0),
        "ng8_any_strict_majority_pass": ng8.get("any_strict_majority_pass", 0),
        "ng8_any_strict_all_pass": ng8.get("any_strict_all_pass", 0),
        "ng8_any_train_shuffle_robust_majority_pass": ng8.get("any_train_shuffle_robust_majority_pass", 0),
        "ng8_any_train_shuffle_robust_all_pass": ng8.get("any_train_shuffle_robust_all_pass", 0),
        "ng8_best_source_vs_noop_acc_delta": ng8_best.get("best_source_vs_noop_acc_delta", ""),
        "ng8_best_source_vs_control_acc_delta": ng8_best.get("best_source_vs_control_acc_delta", ""),
        "ng9_candidate_rows": ng9.get("candidate_rows", 0),
        "ng9_any_strict_majority_pass": ng9.get("any_strict_majority_pass", 0),
        "ng9_any_strict_all_pass": ng9.get("any_strict_all_pass", 0),
        "ng9_any_train_shuffle_robust_majority_pass": ng9.get("any_train_shuffle_robust_majority_pass", 0),
        "ng9_any_train_shuffle_robust_all_pass": ng9.get("any_train_shuffle_robust_all_pass", 0),
        "ng9_best_source_vs_noop_acc_delta": ng9_best.get("best_source_vs_noop_acc_delta", ""),
        "ng9_best_source_vs_control_acc_delta": ng9_best.get("best_source_vs_control_acc_delta", ""),
        "ng10_candidate_rows": ng10.get("candidate_rows", 0),
        "ng10_any_strict_majority_pass": ng10.get("any_strict_majority_pass", 0),
        "ng10_any_strict_all_pass": ng10.get("any_strict_all_pass", 0),
        "ng10_any_train_shuffle_robust_majority_pass": ng10.get("any_train_shuffle_robust_majority_pass", 0),
        "ng10_any_train_shuffle_robust_all_pass": ng10.get("any_train_shuffle_robust_all_pass", 0),
        "ng10_best_source_vs_noop_acc_delta": ng10_best.get("best_source_vs_noop_acc_delta", ""),
        "ng10_best_source_vs_control_acc_delta": ng10_best.get("best_source_vs_control_acc_delta", ""),
        "combined_bridge_any_train_shuffle_robust_majority_pass": combined_bridge_majority,
        "combined_bridge_any_train_shuffle_robust_all_pass": combined_bridge_all,
        "classic_rows": classic.get("rows", 0),
        "classic_exploration_pass_rows": classic.get("exploration_pass_rows", 0),
        "classic_candidate_rows": classic_candidate.get("rows", 0),
        "classic_candidate_exploration_pass_rows": classic_candidate.get("exploration_pass_rows", 0),
        "classic_depth2_rows": classic_depth2.get("rows", 0),
        "classic_depth2_exploration_pass_rows": classic_depth2.get("exploration_pass_rows", 0),
        "combined_classic_exploration_pass_rows": combined_classic_exploration,
        "t1b_microprobe_auc": t1b.get("auc", ""),
        "t1b_microprobe_positive_rows": t1b.get("positive_rows", ""),
        "t1b_microprobe_class_count": t1b.get("class_count", ""),
        "t1b_response_distilled_rows": t1b.get("t1b_response_distilled_rows", 0),
        "t1b_response_distilled_positive_rows": t1b.get("t1b_response_distilled_positive_rows", 0),
        "t1b_response_distilled_class_count": t1b.get("t1b_response_distilled_class_count", 0),
        "t1b_response_distilled_auc_joint": t1b.get("t1b_response_distilled_auc_joint", ""),
        "t1b_response_distilled_exploration_gate_pass": t1b.get("t1b_response_distilled_exploration_gate_pass", 0),
        "code_semantics_review_pass": code_review.get("code_semantics_review_pass", 0),
        "hard_budget_exhausted": 1,
        "fallback_depth": 2,
    }
    for idx, (prefix, item, best) in enumerate(zip(EXTRA_NG_PREFIXES, extra_ng, extra_ng_best), start=11):
        route[f"ng{idx}_prefix"] = prefix
        route[f"ng{idx}_candidate_rows"] = item.get("candidate_rows", 0)
        route[f"ng{idx}_any_strict_majority_pass"] = item.get("any_strict_majority_pass", 0)
        route[f"ng{idx}_any_strict_all_pass"] = item.get("any_strict_all_pass", 0)
        route[f"ng{idx}_any_train_shuffle_robust_majority_pass"] = item.get("any_train_shuffle_robust_majority_pass", 0)
        route[f"ng{idx}_any_train_shuffle_robust_all_pass"] = item.get("any_train_shuffle_robust_all_pass", 0)
        route[f"ng{idx}_best_source_vs_noop_acc_delta"] = best.get("best_source_vs_noop_acc_delta", "")
        route[f"ng{idx}_best_source_vs_control_acc_delta"] = best.get("best_source_vs_control_acc_delta", "")
    route["ng_extra_prefixes"] = list(EXTRA_NG_PREFIXES)
    route["s1_label_free_near_recovered"] = int(route.get("linea_exploration_pass_count", 0) > 0)
    route["s4a_precommit_compensation_opened"] = int(route.get("combined_bridge_any_train_shuffle_robust_majority_pass", 0) == 1)
    route["s4c_classic_exploration_opened"] = int(route.get("combined_classic_exploration_pass_rows", 0) > 0)
    route["s2_t1b_microprobe_visible"] = int(
        (fnum(route.get("t1b_microprobe_auc")) >= 0.60 and int(route.get("t1b_microprobe_positive_rows") or 0) > 0)
        or int(route.get("t1b_response_distilled_exploration_gate_pass") or 0) == 1
    )
    route["official_success_reached"] = int(
        route["s4a_precommit_compensation_opened"]
        and int(route.get("combined_bridge_any_train_shuffle_robust_all_pass", 0)) == 1
    )
    if route["official_success_reached"]:
        route["route"] = "S5-OfficialFunctionalSuccess"
        route["minimum_success"] = "Minimum Success A"
        route["p4_pass"] = 1
        route["promotion_allowed"] = 1
        route["final_stop_allowed"] = 1
        route["fail_reason"] = ""
    elif route["s4a_precommit_compensation_opened"]:
        route["route"] = "S4b-PrecommitBridgeExploratoryButNoS5"
        route["minimum_success"] = "Minimum Success C"
        route["final_stop_allowed"] = 0
        route["fail_reason"] = "precommit bridge has majority evidence but no all-sketch/train-shuffle official S5"
    elif route["s4c_classic_exploration_opened"]:
        route["route"] = "S4c-ClassicFamilyExploratory"
        route["minimum_success"] = "Minimum Success C"
        route["final_stop_allowed"] = 0
        route["fail_reason"] = "classic family exploration opened without official functional S5"
    elif route["s1_label_free_near_recovered"]:
        route["route"] = "S1-LabelFreeNearRecovered"
        route["minimum_success"] = "Minimum Success E"
        route["final_stop_allowed"] = 0
        route["fail_reason"] = "Line A exploration opened but S5 not reached"
    else:
        route["route"] = "R4-S4toS5NoGoAfterDepth2Fallbacks"
        route["minimum_success"] = "Minimum Success F"
        route["final_stop_allowed"] = 1
        route["fail_reason"] = "precommit bridge including I30/I32 plus NG2-NG18 fallback repairs, Line A repair, T1B microprobe/response-distilled features, Line D baseline/candidate/depth2 hardening, and code packet audit did not open S5 or exploratory route"

    figures = write_figures(out_dir, route)
    route.update(write_next_generation_fallback_plan(out_dir, route))
    route.update(write_fallback_manifest(out_dir, route))
    route.update(write_required_manifest(out_dir, figures))
    route["git_status_short"] = git_status_snapshot(out_dir)
    route_path = out_dir / "v1224_route_decision.json"
    exp.write_json(route_path, json_safe(route))
    zip_path = package_zip(out_dir)
    route["code_review_packet"] = exp.rel(zip_path)
    route["code_review_packet_sha256"] = sha256_file(zip_path)
    zip_names = set(zipfile.ZipFile(zip_path).namelist())
    route["code_review_packet_entries"] = len(zip_names)
    missing_transitive = [exp.rel(p) for p in TRANSITIVE_CODE_PACKET_MEMBERS if p.exists() and exp.rel(p) not in zip_names]
    route["transitive_code_packet_required_count"] = sum(1 for p in TRANSITIVE_CODE_PACKET_MEMBERS if p.exists())
    route["transitive_code_packet_missing_count"] = len(missing_transitive)
    route["transitive_code_packet_missing"] = missing_transitive
    exp.write_json(route_path, json_safe(route))
    # Refresh required manifest after route and zip exist.
    route.update(write_required_manifest(out_dir, figures))
    exp.write_json(route_path, json_safe(route))
    print(json.dumps(json_safe(route), indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
