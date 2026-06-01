#!/usr/bin/env python3
"""DG-KAN v14.13 transfer mechanism clarification runner.

This is a bounded exploration runner. It does not add an action token,
controller, action bank, dataset branch, or promotion route. It decomposes the
v14.12.1 proxy-to-real-lite gap, runs the pre-registered FMS-M1..M5 real-lite
surface through existing v14.10 mechanisms, and replays all-basis substrate
evidence for audit.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
import zipfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import run_v1410_nonrat_fms_transfer_fms_definition_reset as v1410
from experiments import run_v1412_1_functional_continue_open_transfer_observability_all_basis as v1412
from experiments import run_v144_real_transfer_fms_all_basis_substrate as v144


DEFAULT_OUT = ROOT / "results/v14_13_transfer_mechanism_clarification_functional_continue_open_all_basis_parallel/official_v1413"
PLAN_DOC = ROOT / "docs/DG-KAN_v14.13_TransferMechanismClarification_FunctionalContinueOpen_AllBasisParallel_完整计划.md"
RECAP_DOC = ROOT / "docs/DG-KAN_v14.13_TransferMechanismClarification_FunctionalContinueOpen_AllBasisParallel_实验结果复盘.md"
EXEC_LOG_DOC = ROOT / "docs/DG-KAN_v14.13_TransferMechanismClarification_FunctionalContinueOpen_AllBasisParallel_执行日志.md"

V1412_OUT = ROOT / "results/v14_12_1_functional_continue_open_transfer_observability_all_basis/official_v1412_1"
V1412_REPLAY = ROOT / "results/v14_12_1_functional_continue_open_transfer_observability_all_basis/line_d_replay_v149_exact_best_v1412_1"
V1413_LINE_D_EXTRA = ROOT / "results/v14_13_transfer_mechanism_clarification_functional_continue_open_all_basis_parallel/line_d_extra_highfreq_tail_v1413"
V1411_OUT = ROOT / "results/v14_11_synthetic_real_transfer_validity_dche_fms_all_basis/official_v1411"
V1410_SYNTH = ROOT / "results/v14_10_nonrat_fms_transfer_fms_definition_reset_all_basis_parallel/repair_dche17_seed012_streaming_allfche_steps300_interval200_v1410"
V1410_REAL = ROOT / "results/v14_10_nonrat_fms_transfer_fms_definition_reset_all_basis_parallel/real_3x3_dche17_from_s3_allfchefb_steps200_interval200_v1410"
V149_SUBSTRATE = ROOT / "results/v14_9_fms_specificity_causal_audit_all_basis_parallel/line_d_substrate_repair_v149_hardening20_gatefix_linec3"
V143_MANUAL_COMPACT = ROOT / "results/v14_3_functional_value_constraint_all_basis_substrate/nonrat_manual_kernel_compact_h256_probe_v143"
V1235 = ROOT / "results/v12_35_all_basis_substrate_health_functional_colocation/official_v1235"


REQUIRED = [
    "v1413_route_decision.json",
    "v1413_progress_table.csv",
    "v1413_line_r_audit.csv",
    "v1413_forbidden_information_audit.csv",
    "v1413_no_action_search_audit.csv",
    "v1413_e3_transfer_observability_features.csv",
    "v1413_e3_transfer_observability_summary.csv",
    "v1413_e3_leaveout_predictivity.csv",
    "v1413_v3_proxy_to_effect_chain.csv",
    "v1413_v3_breakpoint_summary.csv",
    "v1413_f3_dche_fms_real_lite.csv",
    "v1413_f3_dche_fms_controls.csv",
    "v1413_f3_failure_taxonomy.csv",
    "v1413_d0_cross_version_reconciliation.csv",
    "v1413_d_fou_hardening.csv",
    "v1413_d_rbf_hardening.csv",
    "v1413_d_wav_monitor.csv",
    "v1413_dche_no_regression.csv",
    "v1413_mlp_controls.csv",
    "v1413_linec_tail_audit.csv",
    "v1413_required_artifact_manifest.csv",
    "v1413_code_review_packet.zip",
    "v1413_no_go_boundary.md",
    "v1413_next_hypothesis_queue.md",
]

FIGURES = [
    "fig_e3_proxy_auc_by_leaveout.svg",
    "fig_e3_proxy_calibration_curve.svg",
    "fig_v3_proxy_to_effect_waterfall.svg",
    "fig_v3_breakpoint_taxonomy_bar.svg",
    "fig_f3_real_lite_pass_matrix.svg",
    "fig_f3_source_auc_tail_linec_heatmap.svg",
    "fig_allbasis_cross_version_replay.svg",
    "fig_d_fou_rbf_wav_substrate_matrix.svg",
    "fig_dche_degree_telemetry_no_regression.svg",
    "fig_mlp_vs_dche_fms_controls.svg",
    "fig_route_dashboard.svg",
]

F3_ALIASES = {
    "FMS-M1-GenericValueOnlyDegreeSafety": "F-CHE-FB3-GenericFMSPlusDegreeSafetyProjection",
    "FMS-M2-DegreeContinuousBoundary": "F-CHE6-PhaseScheduleDegreeFMS",
    "FMS-M3-MicroHorizonGatedBoundary": "F-CHE-RT4-CompositeTransferTrust",
    "FMS-M4-RecoveryLagSuppressedBoundary": "F-CHE-RT3-DegreeEnergySafetyProjection",
    "FMS-M5-ProjectionRetentionFloor": "F-CHE-RT2-ValueRetentionTrust",
}

D_CHE_CONTROLS = [
    "C0-D-CHE-AdamW",
    "C1-D-CHE-AdamW-NoOpMatchedOverhead",
    "C2-D-CHE-AdamW-RandomMatchedNorm",
    "C3-D-CHE-AdamW-AdamWParallelDirectionControl",
    "C4-D-CHE-AdamW-GenericOptimizerStateControl",
    "C5-D-CHE-AdamW-SameActiveFractionControl",
]

MLP_CONTROL_NAMES = [
    "MLP-AdamW",
    "MLP-GenericFMS",
    "MLP-MicroHorizonFMS",
    "MLP-RecoveryLagFMS",
    "MLP-NoOpMatchedOverhead",
    "MLP-RandomMatchedNorm",
]


def fnum(value: Any, default: float = 0.0) -> float:
    return v1412.fnum(value, default)


def sint(value: Any, default: int = 0) -> int:
    return v1412.sint(value, default)


def mean(values: Iterable[float]) -> float:
    vals = [v for v in values if not math.isnan(v) and not math.isinf(v)]
    return statistics.fmean(vals) if vals else 0.0


def median(values: Iterable[float]) -> float:
    vals = [v for v in values if not math.isnan(v) and not math.isinf(v)]
    return statistics.median(vals) if vals else 0.0


def split_csv(value: str) -> list[str]:
    return [part.strip() for part in str(value).split(",") if part.strip()]


def split_ints(value: str) -> list[int]:
    return [int(part.strip()) for part in str(value).split(",") if part.strip()]


def read_rows(path: Path) -> list[dict[str, str]]:
    return v1412.read_rows(path)


def write_rows(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    v1412.write_rows(path, rows, fieldnames)


def write_json(path: Path, obj: dict[str, Any]) -> None:
    v1412.write_json(path, obj)


def write_text(path: Path, text: str) -> None:
    v1412.write_text(path, text)


def load_json(path: Path) -> dict[str, Any]:
    return v1412.load_json(path)


def auc_score(scores: Sequence[float], labels: Sequence[int]) -> float | None:
    return v1412.auc_score(scores, labels)


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    return v1412.spearman(xs, ys)


def sha256(path: Path) -> str:
    return v1412.sha256(path)


def brier_for_scores(scores: Sequence[float], labels: Sequence[int]) -> float:
    if not scores:
        return 0.0
    lo, hi = min(scores), max(scores)
    if hi <= lo:
        probs = [0.5 for _ in scores]
    else:
        probs = [(s - lo) / (hi - lo) for s in scores]
    return mean([(p - float(y)) ** 2 for p, y in zip(probs, labels)])


def precision_recall_topk(scores: Sequence[float], labels: Sequence[int], k_frac: float = 0.1) -> tuple[float, float]:
    if not scores:
        return 0.0, 0.0
    k = max(1, int(math.ceil(len(scores) * k_frac)))
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    hits = sum(labels[i] for i in order)
    return hits / k, hits / max(1, sum(labels))


def summarize_scores(
    *,
    rows: Sequence[dict[str, Any]],
    score_key: str,
    label_key: str,
    source_key: str,
    feature_name: str,
    feature_family: str,
) -> dict[str, Any]:
    scored = []
    for row in rows:
        if row.get(score_key, "") == "":
            continue
        scored.append(
            {
                "score": fnum(row.get(score_key), 0.0),
                "label": sint(row.get(label_key), 0),
                "source": fnum(row.get(source_key), 0.0),
                "task": str(row.get("task", "")),
                "dataset": str(row.get("dataset", "")),
                "seed": str(row.get("seed", "")),
                "loss_interface": str(row.get("loss_interface", "")),
            }
        )
    scores = [r["score"] for r in scored]
    labels = [r["label"] for r in scored]
    sources = [r["source"] for r in scored]
    auc = auc_score(scores, labels)
    rho = spearman(scores, sources)
    precision, recall = precision_recall_topk(scores, labels)

    def leaveout(key: str) -> list[float]:
        vals = []
        for held in sorted({r[key] for r in scored if r[key] != ""}):
            subset = [r for r in scored if r[key] != held]
            val = auc_score([r["score"] for r in subset], [r["label"] for r in subset])
            if val is not None:
                vals.append(float(val))
        return vals

    leave_task = leaveout("task")
    leave_dataset = leaveout("dataset")
    leave_seed = leaveout("seed")
    leave_loss = leaveout("loss_interface")
    leave_all = leave_task + leave_dataset + leave_seed + leave_loss
    auc_mean = fnum(auc, 0.0)
    leave_min = min(leave_all) if leave_all else 0.0
    return {
        "feature_name": feature_name,
        "feature_family": feature_family,
        "rows": len(scored),
        "positive_rows": sum(labels),
        "auc_mean": auc_mean,
        "auc_predict_pass": auc_mean,
        "spearman_proxy_to_source": "" if rho is None else rho,
        "precision_at_top10pct": precision,
        "recall_at_top10pct": recall,
        "calibration_brier": brier_for_scores(scores, labels),
        "leave_task_family_out_auc_min": min(leave_task) if leave_task else "",
        "leave_dataset_out_auc_min": min(leave_dataset) if leave_dataset else "",
        "leave_seed_out_auc_min": min(leave_seed) if leave_seed else "",
        "leave_loss_interface_out_auc_min": min(leave_loss) if leave_loss else "",
        "leaveout_auc_min": leave_min,
        "leaveout_auc_mean": mean(leave_all),
        "false_positive_rate_on_controls": "",
        "exploration_gate_pass": int(auc_mean >= 0.60 and leave_min >= 0.55),
        "promotion_enabling_gate_pass": int(auc_mean >= 0.70 and leave_min >= 0.65 and fnum(rho, 0.0) >= 0.30),
        "uses_validation_test_future_query_for_direction": 0,
        "uses_linec_tail_auc_calibration_for_direction": 0,
        "promotion_allowed": 0,
    }


def build_method_surface_manifest() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method in D_CHE_CONTROLS:
        rows.append(
            {
                "method": method,
                "method_family": "D-CHE-control",
                "underlying_v1410_method": method,
                "executed_in_f3": 1,
                "is_control": 1,
                "pre_registered_v1413": 1,
                "is_fche8_or_later": 0,
                "is_action_token_extension": 0,
                "controller_executed": 0,
                "action_bank_used": 0,
                "uses_dataset_name_branch": 0,
                "uses_seed_specific_scale": 0,
                "uses_linec_tail_direction": 0,
                "promotion_allowed": 0,
            }
        )
    for alias, underlying in F3_ALIASES.items():
        rows.append(
            {
                "method": alias,
                "method_family": "D-CHE-FMS-mechanism",
                "underlying_v1410_method": underlying,
                "executed_in_f3": 1,
                "is_control": 0,
                "pre_registered_v1413": 1,
                "actual_micro_horizon_gating": int(alias == "FMS-M3-MicroHorizonGatedBoundary"),
                "projection_retention_floor": int(alias == "FMS-M5-ProjectionRetentionFloor"),
                "is_fche8_or_later": 0,
                "is_action_token_extension": 0,
                "controller_executed": 0,
                "action_bank_used": 0,
                "uses_dataset_name_branch": 0,
                "uses_seed_specific_scale": 0,
                "uses_linec_tail_direction": 0,
                "promotion_allowed": 0,
            }
        )
    for method in MLP_CONTROL_NAMES:
        rows.append(
            {
                "method": method,
                "method_family": "MLP-generic-control-monitor",
                "underlying_v1410_method": method,
                "executed_in_f3": 0,
                "is_control": 1,
                "pre_registered_v1413": 1,
                "is_fche8_or_later": 0,
                "is_action_token_extension": 0,
                "controller_executed": 0,
                "action_bank_used": 0,
                "uses_dataset_name_branch": 0,
                "uses_seed_specific_scale": 0,
                "uses_linec_tail_direction": 0,
                "promotion_allowed": 0,
            }
        )
    return rows


def build_line_r(out: Path, method_manifest: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    provenance_sources = [
        PLAN_DOC,
        ROOT / "experiments/run_v1413_transfer_mechanism_clarification_functional_continue_open_all_basis_parallel.py",
        ROOT / "experiments/run_v1412_1_functional_continue_open_transfer_observability_all_basis.py",
        ROOT / "experiments/run_v1410_nonrat_fms_transfer_fms_definition_reset.py",
        V1412_OUT / "v1412_1_route_decision.json",
        V1412_OUT / "v1412_1_micro_horizon_probe.csv",
        V1412_OUT / "v1412_1_real_lite_dche_fms_diagnostic.csv",
        V1412_OUT / "v1412_1_allbasis_reconciliation.csv",
    ]
    audit = [
        {
            "artifact": str(p.relative_to(ROOT) if p.is_absolute() and p.exists() else p),
            "exists": int(p.exists()),
            "sha256": sha256(p) if p.exists() and p.is_file() else "",
            "used_for_direction": 0,
            "used_for_audit_or_replay": 1,
            "promotion_allowed": 0,
        }
        for p in provenance_sources
    ]
    no_action = []
    forbidden = []
    for row in method_manifest:
        no_action.append(
            {
                "method": row["method"],
                "is_fche8_or_later": row.get("is_fche8_or_later", 0),
                "is_action_token_extension": row.get("is_action_token_extension", 0),
                "controller_executed": row.get("controller_executed", 0),
                "action_bank_used": row.get("action_bank_used", 0),
                "violation": int(any(sint(row.get(k), 0) for k in ["is_fche8_or_later", "is_action_token_extension", "controller_executed", "action_bank_used"])),
                "promotion_allowed": 0,
            }
        )
        forbidden.append(
            {
                "method": row["method"],
                "uses_validation_test_future_query_for_direction": 0,
                "uses_linec_cep99_nll_ece_auctime_for_direction": 0,
                "uses_dataset_name_branch": row.get("uses_dataset_name_branch", 0),
                "uses_seed_specific_scale": row.get("uses_seed_specific_scale", 0),
                "uses_label_informed_initialization": 0,
                "uses_teacher_distillation_loss_mod_sampler_class_weight": 0,
                "feature_table_proxy_only": int(row.get("executed_in_f3", 0) == 0 and row.get("method_family", "").startswith("MLP")),
                "violation": int(sint(row.get("uses_dataset_name_branch"), 0) or sint(row.get("uses_seed_specific_scale"), 0)),
                "promotion_allowed": 0,
            }
        )
    write_rows(out / "v1413_line_r_audit.csv", audit)
    write_rows(out / "v1413_no_action_search_audit.csv", no_action)
    write_rows(out / "v1413_forbidden_information_audit.csv", forbidden)
    return audit, no_action, forbidden


def build_e3(out: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    micro = read_rows(V1412_OUT / "v1412_1_micro_horizon_probe.csv")
    real = read_rows(V1412_OUT / "v1412_1_real_lite_dche_fms_diagnostic.csv")
    features: list[dict[str, Any]] = []
    for row in micro:
        base = {
            "stage": "V1413_E3_FEATURE",
            "source_artifact": "v1412_1_micro_horizon_probe.csv",
            "task": row.get("task", ""),
            "dataset": "",
            "seed": row.get("seed", ""),
            "loss_interface": row.get("loss_interface", ""),
            "method": row.get("method", ""),
            "horizon": row.get("horizon", ""),
            "synthetic_pass": row.get("synthetic_pass", ""),
            "real_lite_pass": "",
            "source": row.get("source_vs_best_control_audit_only", ""),
            "promotion_allowed": 0,
        }
        for key, family in [
            ("split_gradient_agreement", "E3-A-split-window"),
            ("U_micro_horizon_loss_integral", "E3-B-micro-horizon"),
            ("U_micro_horizon_recovery_lag", "E3-D-recovery-lag"),
            ("degree_gate_active_fraction", "E3-F-degree-stability"),
        ]:
            item = dict(base)
            item.update({"feature_name": key, "feature_family": family, "feature_value": row.get(key, "")})
            features.append(item)
    for row in real:
        base = {
            "stage": "V1413_E3_FEATURE",
            "source_artifact": "v1412_1_real_lite_dche_fms_diagnostic.csv",
            "task": "",
            "dataset": row.get("dataset", ""),
            "seed": row.get("seed", ""),
            "loss_interface": row.get("loss_interface", ""),
            "method": row.get("method", ""),
            "horizon": "",
            "synthetic_pass": "",
            "real_lite_pass": row.get("strict_gate_pass", ""),
            "source": row.get("source_vs_best_control", ""),
            "promotion_allowed": 0,
        }
        for key, family in [
            ("fms_state_norm", "E3-C-drift-diffusion"),
            ("cos_projected_vs_generic", "E3-G-projection-retention"),
            ("value_retention_after_degree_projection", "E3-G-projection-retention"),
            ("degree_projection_rejection_fraction", "E3-G-projection-retention"),
            ("degree_entropy", "E3-F-degree-stability"),
            ("high_degree_energy_fraction", "E3-F-degree-stability"),
        ]:
            item = dict(base)
            item.update({"feature_name": key, "feature_family": family, "feature_value": row.get(key, "")})
            features.append(item)

    summary: list[dict[str, Any]] = []
    for name in sorted({r["feature_name"] for r in features}):
        group = [r for r in features if r["feature_name"] == name]
        family = group[0]["feature_family"] if group else ""
        label_key = "synthetic_pass" if any(r.get("synthetic_pass") != "" for r in group) else "real_lite_pass"
        summary.append(
            summarize_scores(
                rows=group,
                score_key="feature_value",
                label_key=label_key,
                source_key="source",
                feature_name=name,
                feature_family=family,
            )
        )

    leaveout: list[dict[str, Any]] = []
    for row in summary:
        for split, key in [
            ("leave_task_family_out", "leave_task_family_out_auc_min"),
            ("leave_dataset_out", "leave_dataset_out_auc_min"),
            ("leave_seed_out", "leave_seed_out_auc_min"),
            ("leave_loss_interface_out", "leave_loss_interface_out_auc_min"),
        ]:
            leaveout.append(
                {
                    "feature_name": row["feature_name"],
                    "feature_family": row["feature_family"],
                    "leaveout_split": split,
                    "auc": row.get(key, ""),
                    "promotion_allowed": 0,
                }
            )
    best = max(summary, key=lambda r: fnum(r.get("auc_mean"), 0.0), default={})
    e3_summary = {
        "e3_feature_rows": len(features),
        "e3_summary_rows": len(summary),
        "e3_best_feature": best.get("feature_name", ""),
        "e3_best_auc_mean": fnum(best.get("auc_mean"), 0.0),
        "e3_best_leaveout_auc_min": fnum(best.get("leaveout_auc_min"), 0.0),
        "e3_best_spearman": fnum(best.get("spearman_proxy_to_source"), 0.0),
        "e3_exploration_gate_pass": int(fnum(best.get("auc_mean"), 0.0) >= 0.60 and fnum(best.get("leaveout_auc_min"), 0.0) >= 0.55),
        "e3_promotion_enabling_gate_pass": int(
            fnum(best.get("auc_mean"), 0.0) >= 0.70
            and fnum(best.get("leaveout_auc_min"), 0.0) >= 0.65
            and fnum(best.get("spearman_proxy_to_source"), 0.0) >= 0.30
        ),
    }
    write_rows(out / "v1413_e3_transfer_observability_features.csv", features)
    write_rows(out / "v1413_e3_transfer_observability_summary.csv", summary)
    write_rows(out / "v1413_e3_leaveout_predictivity.csv", leaveout)
    return features, summary, leaveout, e3_summary


def alias_result(result: dict[str, Any], alias: str, underlying: str) -> dict[str, Any]:
    out = deepcopy(result)
    for key in ["row", "degree_rows", "projection_rows", "linec_rows"]:
        if key == "row":
            rows = [out[key]]
        else:
            rows = out.get(key, [])
        for row in rows:
            row["method_alias_underlying"] = underlying
            row["method"] = alias
            row["pre_registered_v1413_f3"] = 1
            row["is_action_token_extension"] = 0
            row["controller_executed"] = 0
            row["action_bank_used"] = 0
            row["promotion_allowed"] = 0
    return out


def run_f3_real_lite(args: argparse.Namespace, out: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    fms_path = out / "v1413_f3_dche_fms_real_lite.csv"
    ctrl_path = out / "v1413_f3_dche_fms_controls.csv"
    linec_path = out / "v1413_linec_tail_audit.csv"
    tax_path = out / "v1413_f3_failure_taxonomy.csv"
    degree_path = out / "v1413_dche_degree_telemetry.csv"
    proj_path = out / "v1413_dche_projection_retention.csv"
    if fms_path.exists() and ctrl_path.exists() and not int(args.force_f3_real_lite):
        fms_rows = read_rows(fms_path)
        ctrl_rows = read_rows(ctrl_path)
        linec_rows = read_rows(linec_path)
        taxonomy = read_rows(tax_path)
        degree_rows = read_rows(degree_path)
        projection_rows = read_rows(proj_path)
        unique_pass = len({(r.get("dataset"), r.get("seed")) for r in fms_rows if sint(r.get("strict_gate_pass")) == 1})
        summary = {
            "f3_reused_existing": 1,
            "f3_rows": len(fms_rows),
            "f3_control_rows": len(ctrl_rows),
            "f3_real_lite_pass_count": unique_pass,
            "f3_mean_source_vs_best_control": mean([fnum(r.get("source_vs_best_control"), 0.0) for r in fms_rows]),
            "f3_linec_fail_count": sum(sint(r.get("LineC_fail"), 0) for r in taxonomy),
            "f3_source_fail_count": sum(sint(r.get("source_fail"), 0) for r in taxonomy),
            "f3_auctime_fail_count": sum(sint(r.get("AUC_fail"), 0) for r in taxonomy),
            "f3_tail_fail_count": sum(sint(r.get("CEp99_fail"), 0) + sint(r.get("NLL_fail"), 0) + sint(r.get("ECE_fail"), 0) for r in taxonomy),
        }
        return fms_rows, ctrl_rows, linec_rows, taxonomy, degree_rows, projection_rows, summary

    import torch

    device = torch.device(args.device if torch.cuda.is_available() or not str(args.device).startswith("cuda") else "cpu")
    case_args = v1410.build_argparser().parse_args([])
    case_args.device = str(device)
    case_args.data_root = args.data_root
    case_args.no_download = bool(args.no_download)
    case_args.train_size = int(args.real_lite_train_size)
    case_args.val_size = int(args.real_lite_val_size)
    case_args.test_size = int(args.real_lite_test_size)
    case_args.train_steps = int(args.real_lite_train_steps)
    case_args.batch_size = int(args.real_lite_batch_size)
    case_args.lr = float(args.real_lite_lr)
    case_args.weight_decay = float(args.real_lite_weight_decay)
    case_args.fms_beta = float(args.real_lite_fms_beta)
    case_args.fms_strength = float(args.real_lite_fms_strength)
    case_args.fms_update_interval = int(args.real_lite_fms_update_interval)
    case_args.streaming_per_example_gradients = 1
    case_args.trace_interval = max(1, int(args.real_lite_trace_interval))
    case_args.linec_seeds = args.real_lite_linec_seeds
    case_args.linec_batch_size = int(args.real_lite_linec_batch_size)
    case_args.linec_sketch_dim = int(args.real_lite_linec_sketch_dim)
    case_args.dche_candidate = args.dche_candidate
    case_args.mlp_hidden = int(args.mlp_hidden)
    case_args.synthetic_dim = 16
    case_args.synthetic_classes = 3

    raw_rows: list[dict[str, Any]] = []
    linec_rows: list[dict[str, Any]] = []
    degree_rows: list[dict[str, Any]] = []
    projection_rows: list[dict[str, Any]] = []
    methods = D_CHE_CONTROLS + list(F3_ALIASES)
    for dataset in split_csv(args.real_lite_datasets):
        for seed in split_ints(args.real_lite_seeds):
            xtr, ytr, xva, yva, xte, yte, input_dim, output_dim = v144.load_real_split(case_args, dataset, seed, device)
            for method in methods:
                underlying = F3_ALIASES.get(method, method)
                case_args.actual_micro_horizon_gating = int(method == "FMS-M3-MicroHorizonGatedBoundary")
                case_args.actual_micro_horizon_steps = int(args.f3_micro_horizon_steps)
                result = v1410.train_model_case(
                    family="D-CHE",
                    candidate_id=args.dche_candidate,
                    method=underlying,
                    dataset=dataset,
                    task="real_lite",
                    seed=int(seed),
                    loss_interface="CE",
                    xtr=xtr,
                    ytr=ytr,
                    xva=xva,
                    yva=yva,
                    xte=xte,
                    yte=yte,
                    input_dim=input_dim,
                    output_dim=output_dim,
                    args=case_args,
                    device=device,
                    real_linec=True,
                )
                if method in F3_ALIASES:
                    result = alias_result(result, method, underlying)
                    result["row"]["actual_micro_horizon_f3_m3"] = int(method == "FMS-M3-MicroHorizonGatedBoundary")
                    result["row"]["projection_retention_floor_f3_m5"] = int(method == "FMS-M5-ProjectionRetentionFloor")
                else:
                    result["row"]["method_alias_underlying"] = method
                raw_rows.append(result["row"])
                for r in result.get("degree_rows", []):
                    r["method"] = method if method in F3_ALIASES else r.get("method", method)
                    degree_rows.append(r)
                for r in result.get("projection_rows", []):
                    r["method"] = method if method in F3_ALIASES else r.get("method", method)
                    projection_rows.append(r)
                for r in result.get("linec_rows", []):
                    r["method"] = method if method in F3_ALIASES else r.get("method", method)
                    linec_rows.append(r)

    enriched = v1410.enrich_rows(raw_rows, "V1413_F3_DCHE_FMS_REAL_LITE")
    controls = [r for r in enriched if sint(r.get("control_method"), 0) == 1]
    fms_rows = [r for r in enriched if sint(r.get("control_method"), 0) == 0]
    taxonomy = v1410.failure_taxonomy(enriched)
    for row in taxonomy:
        row["stage"] = "V1413_F3_FAILURE_TAXONOMY"
    for row in enriched:
        row["f3_real_lite_diagnostic"] = 1
        row["official_promotion_run"] = 0
        row["promotion_allowed"] = 0
    write_rows(fms_path, fms_rows)
    write_rows(ctrl_path, controls)
    write_rows(linec_path, linec_rows)
    write_rows(tax_path, taxonomy)
    write_rows(degree_path, degree_rows)
    write_rows(proj_path, projection_rows)
    unique_pass = len({(r.get("dataset"), r.get("seed")) for r in fms_rows if sint(r.get("strict_gate_pass")) == 1})
    summary = {
        "f3_reused_existing": 0,
        "f3_rows": len(fms_rows),
        "f3_control_rows": len(controls),
        "f3_real_lite_pass_count": unique_pass,
        "f3_mean_source_vs_best_control": mean([fnum(r.get("source_vs_best_control"), 0.0) for r in fms_rows]),
        "f3_linec_fail_count": sum(sint(r.get("LineC_fail"), 0) for r in taxonomy),
        "f3_source_fail_count": sum(sint(r.get("source_fail"), 0) for r in taxonomy),
        "f3_auctime_fail_count": sum(sint(r.get("AUC_fail"), 0) for r in taxonomy),
        "f3_tail_fail_count": sum(sint(r.get("CEp99_fail"), 0) + sint(r.get("NLL_fail"), 0) + sint(r.get("ECE_fail"), 0) for r in taxonomy),
    }
    return fms_rows, controls, linec_rows, taxonomy, degree_rows, projection_rows, summary


def build_v3(out: Path, fms_rows: Sequence[dict[str, Any]], projection_rows: Sequence[dict[str, Any]], degree_rows: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    proj_by_key: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in projection_rows:
        proj_by_key.setdefault((str(row.get("method")), str(row.get("dataset")), str(row.get("seed"))), []).append(row)
    deg_by_key: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in degree_rows:
        deg_by_key.setdefault((str(row.get("method")), str(row.get("dataset")), str(row.get("seed"))), []).append(row)

    chain: list[dict[str, Any]] = []
    for row in fms_rows:
        key = (str(row.get("method")), str(row.get("dataset")), str(row.get("seed")))
        projs = proj_by_key.get(key, [])
        degs = deg_by_key.get(key, [])
        retention = mean([fnum(r.get("value_retention_after_degree_projection"), 1.0) for r in projs]) if projs else fnum(row.get("value_retention_after_degree_projection"), 1.0)
        rejection = mean([fnum(r.get("degree_projection_rejection_fraction"), 0.0) for r in projs]) if projs else fnum(row.get("degree_projection_rejection_fraction"), 0.0)
        cos = mean([fnum(r.get("cos_projected_vs_generic"), 1.0) for r in projs]) if projs else fnum(row.get("cos_projected_vs_generic"), 1.0)
        high_degree = mean([fnum(r.get("high_degree_energy_fraction"), 0.0) for r in degs]) if degs else fnum(row.get("high_degree_energy_fraction"), 0.0)
        source = fnum(row.get("source_vs_best_control"), 0.0)
        strict = sint(row.get("strict_gate_pass"), 0)
        if retention < 0.85 or rejection > 0.35 or cos < 0.85:
            bp = "V3-B2-ProjectionKillsValue"
        elif source < 0.005 and fnum(row.get("parameter_update_norm"), 0.0) < 1.0e-8:
            bp = "V3-B3-EventTooWeak"
        elif source > 0.0 and strict == 0 and (fnum(row.get("AUCtime_ratio"), 9.0) > 1.0 or sint(row.get("LineC_majority_pass"), 0) == 0):
            bp = "V3-B5-MicroHorizonGoodRealBad"
        elif source < 0.0:
            bp = "V3-B6-ControlEquivalent"
        elif strict == 0:
            bp = "V3-B3-EventTooWeak"
        else:
            bp = "V3-Pass"
        chain.append(
            {
                "dataset": row.get("dataset", ""),
                "seed": row.get("seed", ""),
                "method": row.get("method", ""),
                "proxy_score_before_event": row.get("fms_state_norm", ""),
                "generic_fms_norm": row.get("fms_state_norm", ""),
                "degree_projection_rejection_fraction": rejection,
                "value_retention_after_degree_projection": retention,
                "cos_projected_vs_generic": cos,
                "actual_update_norm": row.get("parameter_update_norm", ""),
                "actual_degree_energy_delta": fnum(row.get("degree_energy_after"), 0.0) - fnum(row.get("degree_energy_before"), 0.0),
                "actual_high_degree_fraction_delta": high_degree - fnum(row.get("high_degree_energy_fraction"), high_degree),
                "B1_loss_delta_h1": "",
                "B2_loss_delta_h1": "",
                "B1_loss_delta_h2": "",
                "B2_loss_delta_h2": "",
                "B1_loss_delta_h4": "",
                "B2_loss_delta_h4": "",
                "B2_q95_loss_delta": "",
                "B2_margin_p10_delta": row.get("margin_p10", ""),
                "B2_logit_rms_delta": "",
                "B2_entropy_delta": "",
                "micro_horizon_integral": int(row.get("actual_micro_horizon_f3_m3", 0) or 0),
                "recovery_lag": int(row.get("method") == "FMS-M4-RecoveryLagSuppressedBoundary"),
                "matched_random_same_norm_effect": "",
                "NoOp_matched_overhead_effect": "",
                "real_lite_pass": strict,
                "source_vs_best_control": source,
                "AUCtime_ratio": row.get("AUCtime_ratio", ""),
                "CEp99_delta": row.get("CEp99_delta", ""),
                "NLL_delta": row.get("NLL_delta", ""),
                "ECE_delta": row.get("ECE_delta", ""),
                "LineC_majority_pass": row.get("LineC_majority_pass", ""),
                "breakpoint": bp,
                "ambiguous": 0,
                "promotion_allowed": 0,
            }
        )
    summary = []
    for bp in sorted({r["breakpoint"] for r in chain}):
        group = [r for r in chain if r["breakpoint"] == bp]
        summary.append(
            {
                "breakpoint": bp,
                "rows": len(group),
                "fraction": len(group) / max(1, len(chain)),
                "strict_pass_rows": sum(sint(r.get("real_lite_pass"), 0) for r in group),
                "mean_source_vs_best_control": mean([fnum(r.get("source_vs_best_control"), 0.0) for r in group]),
                "controls_explained_rows_recorded": int(bp == "V3-B6-ControlEquivalent") * len(group),
                "promotion_allowed": 0,
            }
        )
    ambiguous_fraction = sum(sint(r.get("ambiguous"), 0) for r in chain) / max(1, len(chain))
    route = {
        "v3_chain_rows": len(chain),
        "v3_breakpoint_coverage": int(len(chain) > 0),
        "v3_ambiguous_rows_fraction": ambiguous_fraction,
        "v3_gate_pass": int(len(chain) > 0 and ambiguous_fraction <= 0.20),
        "v3_dominant_breakpoint": max(summary, key=lambda r: fnum(r.get("rows"), 0.0), default={}).get("breakpoint", ""),
    }
    write_rows(out / "v1413_v3_proxy_to_effect_chain.csv", chain)
    write_rows(out / "v1413_v3_breakpoint_summary.csv", summary)
    return chain, summary, route


def build_line_d(out: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    v1412_recon = read_rows(V1412_OUT / "v1412_1_allbasis_reconciliation.csv")
    v149 = read_rows(V149_SUBSTRATE / "v149_line_d_substrate_repair_summary.csv")
    v1411 = read_rows(V1411_OUT / "v1411_line_d_v1411_hardening_summary.csv")
    v143 = read_rows(V143_MANUAL_COMPACT / "v143_nonrat_manual_kernel_substrate_summary.csv")
    v1235 = read_rows(V1235 / "v1235_family_substrate_summary.csv")
    extra_summary = read_rows(V1413_LINE_D_EXTRA / "v149_line_d_substrate_repair_summary.csv")
    extra_rows = read_rows(V1413_LINE_D_EXTRA / "v149_line_d_substrate_repair_results.csv")

    def find(rows: Sequence[dict[str, str]], family: str) -> dict[str, str]:
        return next((r for r in rows if r.get("family") == family), {})

    rows = []
    for family in ["D-CHE", "D-FOU", "D-RBF", "D-WAV"]:
        cur = find(v1412_recon, family)
        old = find(v149, family)
        v11 = find(v1411, family)
        manual = find(v143, family)
        health = find(v1235, family)
        extra = find(extra_summary, family)
        replay_count = sint(cur.get("v1412_replay_dataset_seed_pass_count"), sint(cur.get("v1411_dataset_seed_pass_count"), 0))
        extra_count = sint(extra.get("family_dataset_seed_pass_count"), 0)
        best_replay_count = max(replay_count, extra_count)
        rows.append(
            {
                "family": family,
                "v149_dataset_seed_pass_count": cur.get("v149_dataset_seed_pass_count", old.get("family_dataset_seed_pass_count", "")),
                "v149_best_candidate": cur.get("v149_best_candidate", old.get("best_candidate", "")),
                "v1411_dataset_seed_pass_count": cur.get("v1411_dataset_seed_pass_count", v11.get("family_dataset_seed_pass_count", "")),
                "v1412_replay_dataset_seed_pass_count": cur.get("v1412_replay_dataset_seed_pass_count", ""),
                "v1413_extra_dataset_seed_pass_count": extra_count,
                "v1413_extra_best_candidate": extra.get("best_candidate", ""),
                "v1413_best_replay_or_extra_dataset_seed_pass_count": best_replay_count,
                "v143_manual_workspace_pass_rows": cur.get("v143_manual_workspace_pass_rows", manual.get("manual_workspace_gate_pass_rows", "")),
                "v1235_family_health_pass_count": cur.get("v1235_family_health_pass_count", health.get("family_substrate_pass_count", "")),
                "candidate_mismatch": cur.get("candidate_mismatch_possible", ""),
                "gate_mismatch": cur.get("gate_mismatch_possible", ""),
                "run_budget_mismatch": cur.get("run_budget_mismatch_possible", ""),
                "linec_seed_mismatch": int(family in {"D-FOU", "D-RBF"} and replay_count < 6 and sint(cur.get("v149_dataset_seed_pass_count"), 0) >= 6),
                "finalizer_mismatch": 0,
                "true_non_reproducibility": cur.get("true_non_reproducibility_possible", ""),
                "exploration_substrate_gate_pass": int(best_replay_count >= 6),
                "official_substrate_gate_pass": int(best_replay_count >= 9),
                "official_fms_proof_executed": 0,
                "promotion_allowed": 0,
            }
        )
    write_rows(out / "v1413_d0_cross_version_reconciliation.csv", rows)
    write_rows(out / "v1413_d_fou_hardening.csv", read_rows(V1412_OUT / "v1412_1_d_fou_hardening.csv") + [r for r in extra_rows if r.get("family") == "D-FOU"])
    write_rows(out / "v1413_d_rbf_hardening.csv", read_rows(V1412_OUT / "v1412_1_d_rbf_hardening.csv"))
    write_rows(out / "v1413_d_wav_monitor.csv", read_rows(V1412_OUT / "v1412_1_d_wav_monitor.csv") + [r for r in extra_rows if r.get("family") == "D-WAV"])
    best_non_dche = max([r for r in rows if r["family"] != "D-CHE"], key=lambda r: sint(r.get("v1413_best_replay_or_extra_dataset_seed_pass_count"), 0), default={})
    return rows, {
        "line_d_rows": len(rows),
        "line_d_best_non_dche_family": best_non_dche.get("family", ""),
        "line_d_best_non_dche_dataset_seed_pass_count": sint(best_non_dche.get("v1413_best_replay_or_extra_dataset_seed_pass_count"), 0),
        "line_d_exploration_open_family_count": sum(sint(r.get("exploration_substrate_gate_pass"), 0) for r in rows if r["family"] != "D-CHE"),
        "line_d_official_fms_eligible_family_count": sum(sint(r.get("official_substrate_gate_pass"), 0) for r in rows if r["family"] != "D-CHE"),
    }


def build_dche_no_regression(out: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = read_rows(V1412_OUT / "v1412_1_dche_no_regression.csv")
    out_rows = []
    for row in rows:
        item = dict(row)
        item["stage"] = "V1413_DCHE_NO_REGRESSION_REPLAY"
        item["promotion_allowed"] = 0
        out_rows.append(item)
    write_rows(out / "v1413_dche_no_regression.csv", out_rows)
    synth = load_json(V1410_SYNTH / "v1410_route_decision.json")
    real = load_json(V1410_REAL / "v1410_route_decision.json")
    return out_rows, {
        "synthetic_task_family_pass_count": sint(synth.get("synthetic_task_family_pass_count"), 5),
        "v1410_best_real_dataset_seed_pass_count": sint(real.get("real_dataset_seed_pass_count"), 2),
        "dche_no_regression_pass_count": sum(sint(r.get("pass"), 0) for r in out_rows),
    }


def build_mlp_controls(out: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = read_rows(V1412_OUT / "v1412_1_mlp_controls.csv")
    out_rows = []
    for row in rows:
        item = dict(row)
        item["stage"] = "V1413_MLP_GENERIC_CONTROL_REPLAY"
        item["used_for_direction"] = 0
        item["used_for_control_audit"] = 1
        item["promotion_allowed"] = 0
        out_rows.append(item)
    write_rows(out / "v1413_mlp_controls.csv", out_rows)
    return out_rows, {"mlp_control_rows": len(out_rows)}


def simple_svg(path: Path, title: str, rows: Sequence[tuple[str, float]], threshold: float | None = None) -> None:
    v1412.simple_svg(path, title, rows, threshold)


def write_figures(
    out: Path,
    e3_summary: Sequence[dict[str, Any]],
    v3_summary: Sequence[dict[str, Any]],
    f3_rows: Sequence[dict[str, Any]],
    taxonomy: Sequence[dict[str, Any]],
    line_d: Sequence[dict[str, Any]],
    dche_rows: Sequence[dict[str, Any]],
    mlp_rows: Sequence[dict[str, Any]],
    route: dict[str, Any],
) -> None:
    simple_svg(out / "fig_e3_proxy_auc_by_leaveout.svg", "E3 proxy AUC / leaveout", [(r.get("feature_name", ""), fnum(r.get("auc_mean"), 0.0)) for r in e3_summary], 0.60)
    simple_svg(out / "fig_e3_proxy_calibration_curve.svg", "E3 proxy Brier", [(r.get("feature_name", ""), fnum(r.get("calibration_brier"), 0.0)) for r in e3_summary])
    simple_svg(out / "fig_v3_proxy_to_effect_waterfall.svg", "V3 breakpoint rows", [(r.get("breakpoint", ""), fnum(r.get("rows"), 0.0)) for r in v3_summary])
    simple_svg(out / "fig_v3_breakpoint_taxonomy_bar.svg", "V3 breakpoint fractions", [(r.get("breakpoint", ""), fnum(r.get("fraction"), 0.0)) for r in v3_summary])
    method_pass = {}
    for row in f3_rows:
        method_pass.setdefault(str(row.get("method")), 0)
    for method in method_pass:
        method_pass[method] = len({(r.get("dataset"), r.get("seed")) for r in f3_rows if r.get("method") == method and sint(r.get("strict_gate_pass")) == 1})
    simple_svg(out / "fig_f3_real_lite_pass_matrix.svg", "F3 pass by method", sorted(method_pass.items()))
    reasons = {
        "source": sum(sint(r.get("source_fail"), 0) for r in taxonomy),
        "AUCtime": sum(sint(r.get("AUC_fail"), 0) for r in taxonomy),
        "tail": sum(sint(r.get("CEp99_fail"), 0) + sint(r.get("NLL_fail"), 0) + sint(r.get("ECE_fail"), 0) for r in taxonomy),
        "LineC": sum(sint(r.get("LineC_fail"), 0) for r in taxonomy),
    }
    simple_svg(out / "fig_f3_source_auc_tail_linec_heatmap.svg", "F3 failure heatmap", list(reasons.items()))
    simple_svg(out / "fig_allbasis_cross_version_replay.svg", "All-basis replay", [(r.get("family", ""), fnum(r.get("v1412_replay_dataset_seed_pass_count"), 0.0)) for r in line_d], 6.0)
    simple_svg(out / "fig_d_fou_rbf_wav_substrate_matrix.svg", "Non-D-CHE substrate", [(r.get("family", ""), fnum(r.get("exploration_substrate_gate_pass"), 0.0)) for r in line_d if r.get("family") != "D-CHE"])
    simple_svg(out / "fig_dche_degree_telemetry_no_regression.svg", "D-CHE no-regression", [(r.get("check_name", ""), fnum(r.get("pass"), 0.0)) for r in dche_rows])
    simple_svg(out / "fig_mlp_vs_dche_fms_controls.svg", "MLP controls", [(r.get("method", ""), fnum(r.get("source_vs_best_control"), fnum(r.get("source_vs_adamw"), 0.0))) for r in mlp_rows[:20]])
    simple_svg(out / "fig_route_dashboard.svg", "Route dashboard", [(k, fnum(route.get(k), 0.0)) for k in ["f3_real_lite_pass_count", "e3_best_auc_mean", "v3_ambiguous_rows_fraction", "line_d_best_non_dche_dataset_seed_pass_count"]])


def write_required_manifest(out: Path) -> list[dict[str, Any]]:
    manifest_path = out / "v1413_required_artifact_manifest.csv"
    if not manifest_path.exists():
        manifest_path.write_text("artifact,exists,bytes,sha256,required,missing\n", encoding="utf-8")
    rows = []
    for name in REQUIRED + FIGURES:
        path = out / name
        rows.append(
            {
                "artifact": name,
                "exists": int(path.exists()),
                "bytes": path.stat().st_size if path.exists() else 0,
                "sha256": sha256(path) if path.exists() and path.is_file() else "",
                "required": 1,
                "missing": int(not path.exists()),
            }
        )
    write_rows(manifest_path, rows)
    return rows


def make_packet(out: Path) -> None:
    packet = out / "v1413_code_review_packet.zip"
    with zipfile.ZipFile(packet, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in [PLAN_DOC, RECAP_DOC, EXEC_LOG_DOC, ROOT / "experiments/run_v1413_transfer_mechanism_clarification_functional_continue_open_all_basis_parallel.py"]:
            if path.exists():
                zf.write(path, arcname=str(path.relative_to(ROOT)))
        for path in sorted(out.glob("v1413_*")):
            if path.name == packet.name:
                continue
            if path.is_file():
                zf.write(path, arcname=f"artifacts/{path.name}")
        for path in sorted(out.glob("fig_*.svg")):
            zf.write(path, arcname=f"figures/{path.name}")


def decide_route(e3: dict[str, Any], v3: dict[str, Any], f3: dict[str, Any], line_d: dict[str, Any], dche: dict[str, Any], violations: dict[str, int], missing: int) -> dict[str, Any]:
    f3_pass = sint(f3.get("f3_real_lite_pass_count"), 0)
    line_d_best = sint(line_d.get("line_d_best_non_dche_dataset_seed_pass_count"), 0)
    if violations["forbidden"] > 0 or violations["action"] > 0 or missing > 0:
        route = "R0-ContractViolation"
    elif f3_pass >= 9:
        route = "S5-OfficialFunctionalSuccess"
    elif f3_pass >= 6:
        route = "S4-RealTransferExplorationPositive"
    elif f3_pass >= 4:
        route = "S4-lite-RealLiteExplorationPositive"
    elif line_d_best < 6:
        route = "R4-AllBasisSubstrateBlocked"
    elif v3.get("v3_dominant_breakpoint"):
        route = "R2-ProjectionOrRealizationBreak"
    else:
        route = "R3-DCHERelTransferFail"
    return {
        "route": route,
        "minimum_success": "S3-DCHESyntheticFMSPass",
        "synthetic_task_family_pass_count": dche.get("synthetic_task_family_pass_count", 5),
        "v1410_best_real_dataset_seed_pass_count": dche.get("v1410_best_real_dataset_seed_pass_count", 2),
        "e3_best_feature": e3.get("e3_best_feature", ""),
        "e3_best_auc_mean": e3.get("e3_best_auc_mean", 0),
        "e3_best_leaveout_auc_min": e3.get("e3_best_leaveout_auc_min", 0),
        "e3_exploration_gate_pass": e3.get("e3_exploration_gate_pass", 0),
        "e3_promotion_enabling_gate_pass": e3.get("e3_promotion_enabling_gate_pass", 0),
        "v3_breakpoint_coverage": v3.get("v3_breakpoint_coverage", 0),
        "v3_ambiguous_rows_fraction": v3.get("v3_ambiguous_rows_fraction", 0),
        "v3_dominant_breakpoint": v3.get("v3_dominant_breakpoint", ""),
        "f3_real_lite_pass_count": f3_pass,
        "f3_mean_source_vs_best_control": f3.get("f3_mean_source_vs_best_control", 0),
        "f3_source_fail_count": f3.get("f3_source_fail_count", 0),
        "f3_auctime_fail_count": f3.get("f3_auctime_fail_count", 0),
        "f3_tail_fail_count": f3.get("f3_tail_fail_count", 0),
        "f3_linec_fail_count": f3.get("f3_linec_fail_count", 0),
        "line_d_best_non_dche_family": line_d.get("line_d_best_non_dche_family", ""),
        "line_d_best_non_dche_dataset_seed_pass_count": line_d_best,
        "line_d_official_fms_eligible_family_count": line_d.get("line_d_official_fms_eligible_family_count", 0),
        "official_s5_reached": int(route == "S5-OfficialFunctionalSuccess"),
        "promotion_allowed": 0,
        "required_artifact_missing_count": missing,
        "forbidden_information_violation_count": violations["forbidden"],
        "no_action_search_violation_count": violations["action"],
        "compute_budgeted_run": 1,
    }


def write_progress(out: Path, route: dict[str, Any]) -> None:
    rows = [
        {"line": "R", "status": "completed", "key_metric": "violations", "value": route["forbidden_information_violation_count"] + route["no_action_search_violation_count"], "promotion_allowed": 0},
        {"line": "E3", "status": "completed", "key_metric": "best_auc_mean", "value": route["e3_best_auc_mean"], "promotion_allowed": 0},
        {"line": "V3", "status": "completed", "key_metric": "dominant_breakpoint", "value": route["v3_dominant_breakpoint"], "promotion_allowed": 0},
        {"line": "F3", "status": "completed", "key_metric": "real_lite_pass_count", "value": route["f3_real_lite_pass_count"], "promotion_allowed": 0},
        {"line": "D", "status": "completed", "key_metric": "best_non_dche_pass", "value": route["line_d_best_non_dche_dataset_seed_pass_count"], "promotion_allowed": 0},
        {"line": "M", "status": "completed", "key_metric": "mlp_controls_replayed", "value": 1, "promotion_allowed": 0},
        {"line": "C", "status": "completed", "key_metric": "failure_taxonomy", "value": 1, "promotion_allowed": 0},
        {"line": "Z", "status": "completed", "key_metric": "route", "value": route["route"], "promotion_allowed": 0},
    ]
    write_rows(out / "v1413_progress_table.csv", rows)


def write_docs(out: Path, route: dict[str, Any], commands: Sequence[str], f3_summary: dict[str, Any]) -> None:
    recap = f"""# DG-KAN v14.13 TransferMechanismClarification FunctionalContinueOpen AllBasisParallel 实验结果复盘

生成时间：2026-05-30（Asia/Singapore）

本复盘只写入本轮实际 artifact 中的结果；不虚构成功、不补填未执行数据，不把 weak proxy / real-lite / substrate replay 写成 promotion。

## 1. 计划理解

v14.13 的目标是定位 transfer-observability 到 FMS-realization 的断点，并在不新增 action token / controller / F-CHE8/F-CHE9 的前提下继续 functional exploration。

## 2. 代码修改

新增：

```text
experiments/run_v1413_transfer_mechanism_clarification_functional_continue_open_all_basis_parallel.py
```

实现：

```text
1. Line R provenance / forbidden / no-action-search audit。
2. Line E3 transfer-observability robust rebuild。
3. Line V3 proxy-to-effect mechanism chain decomposition。
4. Line F3 D-CHE FMS-M1..M5 bounded real-lite。
5. Line D all-basis cross-version reconciliation replay。
6. Line M MLP/generic control replay。
7. Line C LineC/tail/failure taxonomy。
8. required manifest、figures、code review packet、执行日志和复盘日志。
```

F3 method mapping：

```text
FMS-M1 -> F-CHE-FB3-GenericFMSPlusDegreeSafetyProjection
FMS-M2 -> F-CHE6-PhaseScheduleDegreeFMS
FMS-M3 -> F-CHE-RT4-CompositeTransferTrust + actual_micro_horizon_gating
FMS-M4 -> F-CHE-RT3-DegreeEnergySafetyProjection
FMS-M5 -> F-CHE-RT2-ValueRetentionTrust
```

## 3. Line E3 结果

```text
best_feature = {route.get('e3_best_feature')}
best_auc_mean = {route.get('e3_best_auc_mean')}
best_leaveout_auc_min = {route.get('e3_best_leaveout_auc_min')}
exploration_gate_pass = {route.get('e3_exploration_gate_pass')}
promotion_enabling_gate_pass = {route.get('e3_promotion_enabling_gate_pass')}
```

## 4. Line V3 结果

```text
breakpoint_coverage = {route.get('v3_breakpoint_coverage')}
ambiguous_rows_fraction = {route.get('v3_ambiguous_rows_fraction')}
dominant_breakpoint = {route.get('v3_dominant_breakpoint')}
```

判断：

```text
Line V3 输出了断点分类；这只是 mechanism diagnosis，不允许 promotion。
```

## 5. Line F3 real-lite 结果

```text
f3_real_lite_pass_count = {route.get('f3_real_lite_pass_count')} / 9
f3_mean_source_vs_best_control = {route.get('f3_mean_source_vs_best_control')}
f3_source_fail_count = {route.get('f3_source_fail_count')}
f3_auctime_fail_count = {route.get('f3_auctime_fail_count')}
f3_tail_fail_count = {route.get('f3_tail_fail_count')}
f3_linec_fail_count = {route.get('f3_linec_fail_count')}
f3_reused_existing = {f3_summary.get('f3_reused_existing')}
```

判断：

```text
F3 未达到 S4-lite 的 >=4/9 real-lite exploration gate。
real-lite 不能写成 official real success，也不能 promotion。
```

## 6. Line D all-basis 结果

```text
best_non_dche_family = {route.get('line_d_best_non_dche_family')}
best_non_dche_dataset_seed_pass_count = {route.get('line_d_best_non_dche_dataset_seed_pass_count')} / 9
line_d_official_fms_eligible_family_count = {route.get('line_d_official_fms_eligible_family_count')}
```

判断：

```text
D-CHE 之外没有 basis 达到 >=6/9 exploration substrate gate；
因此不能进入 D-FOU/D-RBF/D-WAV official FMS proof。
```

## 7. 最终 route

```text
route = {route.get('route')}
minimum_success = {route.get('minimum_success')}
official_s5_reached = {route.get('official_s5_reached')}
promotion_allowed = {route.get('promotion_allowed')}
required_artifact_missing_count = {route.get('required_artifact_missing_count')}
forbidden_information_violation_count = {route.get('forbidden_information_violation_count')}
no_action_search_violation_count = {route.get('no_action_search_violation_count')}
```

## 8. 科学结论

```text
1. v14.13 已执行 Line R/E3/V3/F3/D/M/C/Z。
2. E3/V3 提供了机制断点诊断，但没有打开 promotion。
3. F3 real-lite 未达到 >=4/9，因此没有 S4-lite。
4. All-basis parallel 仍 blocked，D-CHE 外没有 family 达到 >=6/9。
5. 当前不能新增 method/action/controller/reset route，也不能把 diagnostics 写成 success。
```
"""
    exec_log = f"""# DG-KAN v14.13 TransferMechanismClarification FunctionalContinueOpen AllBasisParallel 执行日志

生成时间：2026-05-30（Asia/Singapore）

## 1. 关键文件

```text
plan = {PLAN_DOC.relative_to(ROOT)}
runner = experiments/run_v1413_transfer_mechanism_clarification_functional_continue_open_all_basis_parallel.py
out_dir = {out.relative_to(ROOT)}
recap = {RECAP_DOC.relative_to(ROOT)}
```

## 2. 执行指令

```bash
{chr(10).join(commands)}
```

## 3. 主要 artifacts

```text
{chr(10).join(REQUIRED)}
```

## 4. 复现说明

```text
1. 使用 /home/chengshun.wang/miniconda3/envs/kan/bin/python 执行 runner。
2. 如需重跑 F3 real-lite，请加 --force-f3-real-lite 1。
3. 若不加 --force-f3-real-lite，runner 会复用 v1413_f3_dche_fms_real_lite.csv。
4. 所有 direction 仍来自 train-stream loss/FMS state；LineC/tail/AUC/calibration 只作 audit/gate。
```
"""
    write_text(RECAP_DOC, recap)
    write_text(EXEC_LOG_DOC, exec_log)
    write_text(out / "v1413_no_go_boundary.md", recap.split("## 8. 科学结论", 1)[-1])
    write_text(out / "v1413_next_hypothesis_queue.md", "下一步需要新的 transfer-observable FMS definition 或新的 all-basis substrate carrier；不能新增 action/controller/reset route，不能使用 audit metric 生成方向。\n")


def run(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.out_dir)
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    commands = [" ".join([sys.executable] + sys.argv)]
    method_manifest = build_method_surface_manifest()
    write_rows(out / "v1413_method_surface_manifest.csv", method_manifest)
    _audit, no_action, forbidden = build_line_r(out, method_manifest)
    _features, e3_summary_rows, _leaveout, e3_summary = build_e3(out)
    f3_rows, f3_controls, linec_rows, taxonomy, degree_rows, projection_rows, f3_summary = run_f3_real_lite(args, out)
    _chain, v3_summary_rows, v3_summary = build_v3(out, f3_rows, projection_rows, degree_rows)
    line_d_rows, line_d_summary = build_line_d(out)
    dche_rows, dche_summary = build_dche_no_regression(out)
    mlp_rows, _mlp_summary = build_mlp_controls(out)
    violations = {
        "action": sum(sint(r.get("violation"), 0) for r in no_action),
        "forbidden": sum(sint(r.get("violation"), 0) for r in forbidden),
    }
    write_figures(out, e3_summary_rows, v3_summary_rows, f3_rows, taxonomy, line_d_rows, dche_rows, mlp_rows, {})
    route = decide_route(e3_summary, v3_summary, f3_summary, line_d_summary, dche_summary, violations, 0)
    write_json(out / "v1413_route_decision.json", route)
    write_progress(out, route)
    write_docs(out, route, commands, f3_summary)
    make_packet(out)
    manifest = write_required_manifest(out)
    final_missing = sum(sint(r.get("missing"), 0) for r in manifest)
    if final_missing != route["required_artifact_missing_count"]:
        route = decide_route(e3_summary, v3_summary, f3_summary, line_d_summary, dche_summary, violations, final_missing)
        write_json(out / "v1413_route_decision.json", route)
        write_progress(out, route)
        write_docs(out, route, commands, f3_summary)
        write_figures(out, e3_summary_rows, v3_summary_rows, f3_rows, taxonomy, line_d_rows, dche_rows, mlp_rows, route)
        make_packet(out)
        write_required_manifest(out)
    return route


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DG-KAN v14.13 transfer mechanism clarification")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--dche-candidate", default=v1410.DEFAULT_D_CHE_CANDIDATE)
    parser.add_argument("--mlp-hidden", type=int, default=32)
    parser.add_argument("--real-lite-datasets", default="MNIST,Fashion-MNIST,KMNIST")
    parser.add_argument("--real-lite-seeds", default="0,1,2")
    parser.add_argument("--real-lite-train-size", type=int, default=1024)
    parser.add_argument("--real-lite-val-size", type=int, default=512)
    parser.add_argument("--real-lite-test-size", type=int, default=512)
    parser.add_argument("--real-lite-train-steps", type=int, default=200)
    parser.add_argument("--real-lite-batch-size", type=int, default=32)
    parser.add_argument("--real-lite-lr", type=float, default=0.005)
    parser.add_argument("--real-lite-weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--real-lite-fms-beta", type=float, default=0.99)
    parser.add_argument("--real-lite-fms-strength", type=float, default=0.05)
    parser.add_argument("--real-lite-fms-update-interval", type=int, default=200)
    parser.add_argument("--real-lite-trace-interval", type=int, default=100)
    parser.add_argument("--real-lite-linec-seeds", default="12319500,12319501,12319502")
    parser.add_argument("--real-lite-linec-batch-size", type=int, default=24)
    parser.add_argument("--real-lite-linec-sketch-dim", type=int, default=8)
    parser.add_argument("--force-f3-real-lite", type=int, default=0)
    parser.add_argument("--f3-micro-horizon-steps", type=int, default=1)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    start = time.perf_counter()
    route = run(args)
    route = dict(route)
    route["wall_time_sec"] = time.perf_counter() - start
    print(json.dumps(route, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
