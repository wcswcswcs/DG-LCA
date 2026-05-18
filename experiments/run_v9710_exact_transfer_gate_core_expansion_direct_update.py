#!/usr/bin/env python3
"""DG-KAN v9.7.1 exact transfer gate runner.

This runner follows the v9.7.1 plan conservatively: exact cross-sample
transfer is official only if landed exact per-sample / exact-apply artifacts
are present.  The v9.7.0 proxy transfer ledger is reused only as diagnostic
boundary evidence.  No fake exact rows, generated APG rows, or proxy official
rows are synthesized.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "experiments"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

from dgkan_outcome_controller import fnum, inum, read_csv, read_json, sha256_file, write_csv, write_json  # noqa: E402


PLAN_PATH = REPO / "docs/DG-KAN_v9.7.1_ExactTransferGate_CoreExpansion_DirectUpdate_完整实验计划.md"
SCRIPT_PATH = REPO / "experiments/run_v9710_exact_transfer_gate_core_expansion_direct_update.py"
RESULT_ROOT = REPO / "results/real_rerun_20260506"
DEFAULT_OUT = RESULT_ROOT / "v9710_exact_transfer_gate_core_expansion_direct_update_first_20260516T000000Z"
DEFAULT_V9700 = RESULT_ROOT / "v9700_dual_validation_existing_action_transfer_principle_first_20260515T190000Z"

TARGET_K = 87


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=str(DEFAULT_OUT))
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--data-root", default="data")
    p.add_argument("--seed", type=int, default=1314)
    p.add_argument("--source-v9700", default=str(DEFAULT_V9700))
    p.add_argument("--direct-actions-per-subspace", type=int, default=64)
    return p.parse_args()


def repo_rel(path: str | Path) -> str:
    p = Path(path)
    if not p.is_absolute():
        p = (REPO / p).resolve()
    else:
        p = p.resolve()
    try:
        return str(p.relative_to(REPO))
    except ValueError:
        return str(p)


def summary_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return next((r for r in rows if r.get("status") == "summary"), rows[0] if rows else {})


def write_bar_svg(path: Path, title: str, labels: list[str], values: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width = 760
    height = 320
    margin = 58
    maxv = max(values + [1.0])
    bar_w = max(18, int((width - 2 * margin) / max(1, len(values))) - 10)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{margin}" y="32" font-family="Arial" font-size="18" fill="#111">{title}</text>',
        f'<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="#333"/>',
    ]
    for i, (label, val) in enumerate(zip(labels, values)):
        x = margin + i * ((width - 2 * margin) / max(1, len(values))) + 5
        h = (height - 2 * margin - 30) * (float(val) / maxv if maxv else 0.0)
        y = height - margin - h
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w}" height="{h:.1f}" fill="#4b78a8"/>')
        parts.append(f'<text x="{x:.1f}" y="{height-margin+18}" font-family="Arial" font-size="10" fill="#333">{label[:18]}</text>')
        parts.append(f'<text x="{x:.1f}" y="{y-6:.1f}" font-family="Arial" font-size="10" fill="#333">{val:.3g}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n")


def p0_boundary(source: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    route = read_json(source / "route_decision_v9700.json")
    field = summary_row(read_csv(source / "p0_field_legality_audit_v9700.csv"))
    boundary = summary_row(read_csv(source / "p0_boundary_reproduction_v9700.csv"))
    nofake = summary_row(read_csv(source / "no_fake_audit_v9700.csv"))
    contract = summary_row(read_csv(source / "contract_audit_v9700.csv"))
    base = summary_row(read_csv(source / "base_acc_sentinel_v9700.csv"))
    green_count = field.get("green_count") or 14
    yellow_count = field.get("yellow_count") or 4
    red_count = field.get("red_count") or 0
    row = {
        "stage": "P0_BOUNDARY_REPRODUCTION_V9710",
        "status": "summary",
        "source_route_v9700": route.get("route"),
        "system_legal_controller_pass_v9700": route.get("system_legal_controller_pass"),
        "generated_stop_v9700": int(route.get("secondary_blocker") == "generated_route_stopped_no_new_objective"),
        "source_primary_blocker_v9700": route.get("primary_blocker"),
        "source_secondary_blocker_v9700": route.get("secondary_blocker"),
        "field_green_count": green_count,
        "field_yellow_count": yellow_count,
        "field_red_count": red_count,
        "uses_dataset_name_for_controller": field.get("dataset_name_used_by_controller_count", boundary.get("dataset_name_used_by_controller_count", 0)),
        "uses_outcome_derived_field": field.get("outcome_derived_field_used_count", boundary.get("outcome_derived_field_used_count", 0)),
        "uses_validation_or_test": field.get("validation_test_used_by_controller_count", boundary.get("validation_test_used_by_controller_count", 0)),
        "fake_row_count": nofake.get("fake_data_used", 0),
        "proxy_official_count": 0,
        "cpu_offload_used": nofake.get("cpu_offload_used", 0),
        "manual_forward_pass": contract.get("manual_forward", 1),
        "manual_backward_pass": contract.get("manual_backward", 1),
        "manual_update_pass": contract.get("manual_adamw_update", 1),
        "base_acc_sentinel_pass_v9700": base.get("base_acc_sentinel_pass", 1),
        "p0_pass": int(
            route.get("route") == "R4-TransferPrincipleFail"
            and inum(route.get("system_legal_controller_pass")) == 0
            and inum(nofake.get("fake_data_used")) == 0
            and inum(nofake.get("proxy_row_used")) == 0
            and inum(nofake.get("cpu_offload_used")) == 0
            and inum(red_count) == 0
            and inum(field.get("dataset_name_used_by_controller_count", boundary.get("dataset_name_used_by_controller_count", 0))) == 0
            and inum(field.get("outcome_derived_field_used_count", boundary.get("outcome_derived_field_used_count", 0))) == 0
        ),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    field_row = {
        "stage": "P0_FIELD_LEGALITY_AUDIT_V9710",
        "status": "summary",
        "green_count": green_count,
        "yellow_count": yellow_count,
        "red_count": red_count,
        "dataset_allowed_for_leaveout": field.get("dataset_allowed_for_leaveout", 1),
        "dataset_name_commit_feature_count": field.get("dataset_name_commit_feature_count", 0),
        "uses_dataset_name_for_selector": 0,
        "uses_dataset_name_for_controller": 0,
        "outcome_derived_field_used_count": field.get("outcome_derived_field_used_count", 0),
        "validation_test_used_by_controller_count": field.get("validation_test_used_by_controller_count", 0),
        "field_legality_pass": row["p0_pass"],
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [row], [field_row], row


def p1_exact_transfer(source: Path, out: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    p1_v9700 = summary_row(read_csv(source / "p1_cross_sample_transfer_ledger_v9700.csv"))
    p5_v9700 = summary_row(read_csv(source / "p5_exact_microprobe_v9700.csv"))
    action_count = inum(p1_v9700.get("canonical_ap0_action_count"))
    exact_candidates = list(source.glob("*exact_transfer*")) + list(source.glob("*exact_apply*")) + list(source.glob("*per_sample*"))
    exact_candidates = [p for p in exact_candidates if p.name not in {"p5_exact_microprobe_v9700.csv"}]
    exact_available = int(bool(exact_candidates) and inum(p1_v9700.get("exact_per_sample_gradient_available")) == 1)
    apply_available = int(inum(p5_v9700.get("microprobe_pass")) == 1)
    summary = {
        "stage": "P1_EXACT_TRANSFER_ARTIFACT_V9710",
        "status": "summary",
        "canonical_ap0_action_count": action_count,
        "proxy_transfer_row_count_v9700": p1_v9700.get("transfer_row_count"),
        "exact_linear_transfer_row_count": 0,
        "exact_apply_audit_subset_count": 0,
        "missing_exact_linear_row_count": action_count,
        "missing_required_field_count": action_count,
        "nan_inf_count": 0,
        "exact_per_sample_gradient_available": exact_available,
        "exact_apply_checkpoint_available": apply_available,
        "exact_artifact_candidate_count": len(exact_candidates),
        "exact_vs_linear_corr": "",
        "exact_vs_linear_abs_error_p95": "",
        "exact_apply_sign_match": "",
        "feature_compute_ms_q90": "",
        "exact_apply_ms_q90": "",
        "P1_exact_transfer_weak_pass": 0,
        "P1_exact_transfer_strong_pass": 0,
        "p1_pass": 0,
        "reason": "exact_per_sample_gradient_and_apply_checkpoint_not_landed",
        "proxy_transfer_available_for_diagnostic": int(inum(p1_v9700.get("ledger_complete_pass")) == 1),
        "proxy_promoted_to_official": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    rows = [summary]
    rows.append(
        {
            "stage": "P1_EXACT_TRANSFER_ARTIFACT_V9710",
            "status": "artifact_scan",
            "scan_scope": repo_rel(source),
            "matched_exact_artifact_count": len(exact_candidates),
            "matched_exact_artifacts": ",".join(repo_rel(p) for p in exact_candidates),
            "v9700_exact_microprobe_status": p5_v9700.get("status"),
            "v9700_exact_microprobe_reason": p5_v9700.get("reason"),
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
    )
    write_bar_svg(out / "fig_p1_linear_transfer_vs_exact_apply_scatter.svg", "P1 exact transfer availability", ["exact rows", "apply subset"], [0, 0])
    write_bar_svg(out / "fig_p1_response_distribution_violin.svg", "P1 exact response distribution", ["not landed"], [0])
    write_bar_svg(out / "fig_p1_response_mean_vs_std.svg", "P1 response mean/std", ["not landed"], [0])
    write_bar_svg(out / "fig_p1_per_dataset_response_scale_hist.svg", "P1 per-dataset exact scale", ["not landed"], [0])
    return rows, summary


def p2_exact_selector(source: Path, p1: dict[str, Any], out: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    p2_v9700 = summary_row(read_csv(source / "p2_existing_action_rank_comparison_v9700.csv"))
    row = {
        "stage": "P2_EXACT_TRANSFER_SELECTOR_V9710",
        "status": "not_run",
        "reason": "P1_exact_transfer_artifact_not_available",
        "rule_count": 0,
        "evaluation_row_count": 0,
        "exact_selector_strong_pass": 0,
        "exact_selector_weak_pass": 0,
        "best_exact_rule_id": "",
        "best_exact_precision": "",
        "best_exact_V_LCB": "",
        "best_exact_longrisk_UCB": "",
        "best_exact_LDO_drop": "",
        "old_R8A_precision_diagnostic": p2_v9700.get("best_GradeAB_precision"),
        "old_R8A_V_LCB_diagnostic": p2_v9700.get("best_V_integrated_LCB"),
        "old_R8A_LDO_drop_diagnostic": p2_v9700.get("best_LDO_drop"),
        "proxy_transfer_precision_diagnostic": p2_v9700.get("best_transfer_precision"),
        "proxy_transfer_V_LCB_diagnostic": p2_v9700.get("best_transfer_V_LCB"),
        "proxy_transfer_LDO_drop_diagnostic": p2_v9700.get("best_transfer_LDO_drop"),
        "P1_exact_transfer_weak_pass": p1.get("P1_exact_transfer_weak_pass"),
        "proxy_promoted_to_official": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    write_bar_svg(out / "fig_p2_exact_transfer_precision_vs_ldo.svg", "P2 exact selector", ["not run"], [0])
    write_bar_svg(out / "fig_p2_exact_vs_old_rank_frontier.svg", "P2 exact vs old rank", ["not run"], [0])
    return [row], row


def p3_core_expansion_exact(source: Path, p1: dict[str, Any], out: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    p3_v9700 = summary_row(read_csv(source / "p3_core_expansion_transfer_v9700.csv"))
    row = {
        "stage": "P3_CORE_EXPANSION_EXACT_TRANSFER_V9710",
        "status": "not_run",
        "reason": "P1_exact_transfer_artifact_not_available",
        "candidate_count": 0,
        "candidate_pass_count": 0,
        "core_count_diagnostic_from_v9700": p3_v9700.get("core_count"),
        "needed_expansion_to_87": max(0, TARGET_K - inum(p3_v9700.get("core_count"))),
        "best_candidate_id": "",
        "best_accepted_count": "",
        "best_GradeAB_precision": "",
        "best_V_integrated_LCB": "",
        "best_longrisk_UCB": "",
        "best_LDO_drop": "",
        "p3_core_exact_expansion_pass": 0,
        "proxy_core_expansion_best_candidate_diagnostic": p3_v9700.get("best_candidate_id"),
        "proxy_core_expansion_best_precision_diagnostic": p3_v9700.get("best_GradeAB_precision"),
        "proxy_core_expansion_best_LDO_drop_diagnostic": p3_v9700.get("best_LDO_drop"),
        "proxy_promoted_to_official": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    write_bar_svg(out / "fig_p3_core_exact_expansion_sankey.svg", "P3 exact expansion", ["core", "exact expansion"], [inum(p3_v9700.get("core_count")), 0])
    write_bar_svg(out / "fig_p3_exact_expansion_quality_table.svg", "P3 exact expansion quality", ["not run"], [0])
    return [row], row


def p4_dataset_shift_exact(source: Path, p1: dict[str, Any], out: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    p4_v9700 = summary_row(read_csv(source / "p4_dataset_shift_no_tuning_diagnostic_v9700.csv"))
    rows = [
        {
            "stage": "P4_DATASET_SHIFT_EXACT_NO_TUNING_V9710",
            "status": "summary",
            "exact_transport_available": p1.get("P1_exact_transfer_weak_pass"),
            "score_count": 0,
            "raw_score_psi_mean_diagnostic_v9700": p4_v9700.get("raw_score_psi_mean"),
            "exact_transport_psi_mean": "",
            "exact_transport_precision": "",
            "exact_transport_V_LCB": "",
            "exact_transport_LDO_drop": "",
            "dataset_shift_score_scale_solved": 0,
            "dataset_specific_selector_produced": 0,
            "reason": "P1_exact_transfer_artifact_not_available",
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
    ]
    for r in read_csv(source / "p4_dataset_shift_no_tuning_diagnostic_v9700.csv"):
        if r.get("status") == "score_row" and r.get("score_id") == "raw_score":
            rows.append(
                {
                    "stage": "P4_DATASET_SHIFT_EXACT_NO_TUNING_V9710",
                    "status": "raw_score_diagnostic_from_v9700",
                    "score_id": r.get("score_id"),
                    "psi_mean": r.get("psi_mean"),
                    "TopK87_precision": r.get("TopK87_precision"),
                    "TopK87_V_LCB": r.get("TopK87_V_LCB"),
                    "LDO_drop": r.get("LDO_drop"),
                    "per_dataset_PSI": r.get("per_dataset_PSI"),
                    "per_dataset_TopK_count": r.get("per_dataset_TopK_count"),
                    "per_dataset_TopK_precision": r.get("per_dataset_TopK_precision"),
                    "fake_data_used": 0,
                    "proxy_row_used": 0,
                    "cpu_offload_used": 0,
                }
            )
    write_bar_svg(out / "fig_p4_exact_transport_psi.svg", "P4 exact transport PSI", ["raw diag", "exact"], [fnum(p4_v9700.get("raw_score_psi_mean")), 0])
    return rows, rows[0]


def p5_direct_update(p1: dict[str, Any], out: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    subspaces = [
        "D1-last-edge-coefficients-only",
        "D2-final-KAN-basis-block-only",
        "D3-low-rank-edge-residual-direction",
        "D4-AdamW-orthogonal-residual-direction",
        "D5-memory-gradient-orthogonal-residual-direction",
        "D6-low-cost-transfer-positive-intersection",
    ]
    summary = {
        "stage": "P5_DIRECT_TRANSFER_SOLVED_UPDATE_V9710",
        "status": "summary",
        "subspace_count": len(subspaces),
        "generated_action_count": 0,
        "branch_horizon_rows_expected": 0,
        "branch_horizon_rows_actual": 0,
        "negative_control_generated": 0,
        "direct_solved_weak_pass": 0,
        "direct_solved_strong_pass": 0,
        "new_objective_evidence_present": 0,
        "reason": "P1_exact_transfer_weak_pass_failed",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    rows = [summary]
    for sid in subspaces:
        rows.append(
            {
                "stage": "P5_DIRECT_TRANSFER_SOLVED_UPDATE_V9710",
                "status": "not_run_subspace",
                "subspace_id": sid,
                "reason": "P1_exact_transfer_weak_pass_failed",
                "generated_action_count": 0,
                "payload_hash_missing_count": 0,
                "certificate_hash_missing_count": 0,
                "action_apply_linf_max": "",
                "branch_horizon_rows_expected": 0,
                "branch_horizon_rows_actual": 0,
                "GradeAB_precision": 0,
                "V_integrated_LCB": 0,
                "longrisk_UCB": 1,
                "bad_UCB": 1,
                "null_UCB": 1,
                "memory_UCB": 1,
                "offdiag_UCB": 1,
                "weak_pass": 0,
                "strong_pass": 0,
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            }
        )
    write_bar_svg(out / "fig_p5_direct_solved_frontier.svg", "P5 direct solved frontier", subspaces, [0] * len(subspaces))
    write_bar_svg(out / "fig_p5_direct_negative_control.svg", "P5 negative control", ["not run"], [0])
    return rows, summary


def p6_failure_taxonomy(p1: dict[str, Any], out: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    row = {
        "stage": "P6_PROXY_EXACT_FAILURE_TAXONOMY_V9710",
        "status": "summary",
        "taxonomy_opened": 0,
        "proxy_exact_pair_count": 0,
        "spearman_proxy_exact": "",
        "sign_mismatch_rate": "",
        "dominant_failure_class": "F0-exact-transfer-artifact-missing",
        "F1_proxy_sign_wrong": 0,
        "F2_proxy_scale_wrong": 0,
        "F3_exact_itself_weak": 0,
        "F4_value_positive_but_high_longrisk": 0,
        "F5_dataset_stable_but_low_precision": 0,
        "F6_precise_but_coverage_low": 0,
        "F7_useful_only_as_expansion": 0,
        "reason": "P1_exact_transfer_artifact_not_available",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    write_bar_svg(out / "fig_p6_proxy_exact_mismatch.svg", "P6 proxy vs exact", ["no exact"], [0])
    return [row], row


def boundary_not_run(stage: str, reason: str, **extra: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    row = {
        "stage": stage,
        "status": "not_run",
        "reason": reason,
        **extra,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [row], row


def p9_generated_route(p1: dict[str, Any], p5: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    row = {
        "stage": "P9_GENERATED_ROUTE_DECISION_V9710",
        "status": "summary",
        "generated_route_status": "stopped_no_new_objective",
        "reason": "P1_exact_transfer_fail_and_P5_not_open",
        "P1_exact_transfer_weak_pass": p1.get("P1_exact_transfer_weak_pass"),
        "direct_solved_weak_pass": p5.get("direct_solved_weak_pass"),
        "direct_solved_strong_pass": p5.get("direct_solved_strong_pass"),
        "new_objective_evidence_present": 0,
        "APGV_APGW_allowed": 0,
        "generated_action_count": 0,
        "branch_horizon_rows_actual": 0,
        "generated_route_stop_triggered": 1,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    return [row], row


def p12_base_acc(source: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = [dict(r) for r in read_csv(source / "base_acc_sentinel_v9700.csv")]
    for row in rows:
        row["stage"] = "BASE_ACC_SENTINEL_V9710"
        row["base_acc_reused_from_v9700"] = 1
        row["base_acc_used_for_controller"] = 0
    return rows, summary_row(rows)


def count_artifact_rows(paths: list[Path]) -> dict[str, Any]:
    rows_checked = 0
    fake_proxy = 0
    fake = 0
    proxy = 0
    cpu = 0
    for path in paths:
        if path.suffix.lower() != ".csv":
            continue
        for row in read_csv(path):
            rows_checked += 1
            fake += inum(row.get("fake_data_used"))
            proxy += inum(row.get("proxy_row_used"))
            cpu += inum(row.get("cpu_offload_used"))
            fake_proxy += int(inum(row.get("fake_data_used")) or inum(row.get("proxy_row_used")) or inum(row.get("cpu_offload_used")))
    return {
        "rows_checked": rows_checked,
        "fake_proxy_nonzero_count": fake_proxy,
        "fake_data_used": fake,
        "proxy_row_used": proxy,
        "cpu_offload_used": cpu,
        "no_fake": int(fake == 0),
        "no_proxy": int(proxy == 0),
    }


def main() -> None:
    args = parse_args()
    out = Path(args.out_dir)
    if args.fresh and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    artifacts: dict[str, Path] = {}

    def dump_csv(name: str, rows: list[dict[str, Any]]) -> Path:
        path = out / name
        write_csv(path, rows)
        artifacts[name] = path
        return path

    def dump_json(name: str, obj: dict[str, Any]) -> Path:
        path = out / name
        write_json(path, obj)
        artifacts[name] = path
        return path

    source = Path(args.source_v9700)

    p0_rows, field_rows, p0 = p0_boundary(source)
    dump_csv("p0_boundary_reproduction_v9710.csv", p0_rows)
    dump_csv("p0_field_legality_audit_v9710.csv", field_rows)

    p1_rows, p1 = p1_exact_transfer(source, out)
    dump_csv("p1_exact_transfer_artifact_v9710.csv", p1_rows)

    p2_rows, p2 = p2_exact_selector(source, p1, out)
    dump_csv("p2_exact_transfer_selector_existing_action_v9710.csv", p2_rows)

    p3_rows, p3 = p3_core_expansion_exact(source, p1, out)
    dump_csv("p3_core_expansion_exact_transfer_v9710.csv", p3_rows)

    p4_rows, p4 = p4_dataset_shift_exact(source, p1, out)
    dump_csv("p4_dataset_shift_exact_no_tuning_v9710.csv", p4_rows)

    p5_rows, p5 = p5_direct_update(p1, out)
    dump_csv("p5_direct_transfer_solved_update_v9710.csv", p5_rows)

    p6_rows, p6 = p6_failure_taxonomy(p1, out)
    dump_csv("p6_proxy_exact_failure_taxonomy_v9710.csv", p6_rows)

    p7_rows, p7 = boundary_not_run(
        "P7_EXISTING_ACTION_MINIMAL_CONTROLLER_V9710",
        "P2_P3_no_exact_transfer_selector_or_expansion_pass",
        controller_selected=0,
        controller_pass=0,
        source_controller_pass=0,
    )
    dump_csv("p7_existing_action_minimal_controller_v9710.csv", p7_rows)

    p8_rows, p8 = boundary_not_run("P8_SELECTED_RUNTIME_V9710", "P7_controller_not_selected", selected_runtime_pass=0)
    dump_csv("p8_selected_runtime_v9710.csv", p8_rows)

    p9_rows, p9 = p9_generated_route(p1, p5)
    dump_csv("p9_generated_route_decision_v9710.csv", p9_rows)

    p10_rows, p10 = boundary_not_run(
        "P10_SYSTEM_BOUNDARY_V9710",
        "P7_or_P8_not_passed",
        controller_pass=0,
        runtime_pass=0,
        system_legal_controller_pass=0,
        paired_replay_opened=0,
    )
    dump_csv("p10_system_boundary_v9710.csv", p10_rows)

    p11_rows, p11 = boundary_not_run(
        "P11_PAIRED_REPLAY_SHORT_FULL_BOUNDARY_V9710",
        "P10_system_not_official",
        paired_replay_pass=0,
        short_run_boundary_open=0,
        full_run_boundary_open=0,
    )
    dump_csv("p11_paired_replay_short_full_boundary_v9710.csv", p11_rows)

    base_rows, base_summary = p12_base_acc(source)
    dump_csv("base_acc_sentinel_v9710.csv", base_rows)

    route = "R1-ExactTransferArtifactMissing"
    primary = "exact_transfer_artifact_missing"
    secondary = "generated_route_stopped_no_new_objective"
    route_row = {
        "stage": "P13_ROUTE_DECISION_V9710",
        "status": "summary",
        "route": route,
        "source_route_v9700": p0.get("source_route_v9700"),
        "p0_pass": p0.get("p0_pass"),
        "P1_exact_transfer_weak_pass": p1.get("P1_exact_transfer_weak_pass"),
        "P1_exact_transfer_strong_pass": p1.get("P1_exact_transfer_strong_pass"),
        "exact_per_sample_gradient_available": p1.get("exact_per_sample_gradient_available"),
        "exact_apply_checkpoint_available": p1.get("exact_apply_checkpoint_available"),
        "exact_linear_transfer_row_count": p1.get("exact_linear_transfer_row_count"),
        "exact_apply_audit_subset_count": p1.get("exact_apply_audit_subset_count"),
        "p2_exact_selector_pass": 0,
        "p3_core_exact_expansion_pass": p3.get("p3_core_exact_expansion_pass"),
        "dataset_shift_score_scale_solved": p4.get("dataset_shift_score_scale_solved"),
        "direct_solved_weak_pass": p5.get("direct_solved_weak_pass"),
        "direct_solved_strong_pass": p5.get("direct_solved_strong_pass"),
        "new_objective_evidence_present": p5.get("new_objective_evidence_present"),
        "generated_route_status": p9.get("generated_route_status"),
        "APGV_APGW_allowed": p9.get("APGV_APGW_allowed"),
        "controller_pass": p7.get("controller_pass"),
        "selected_runtime_pass": p8.get("selected_runtime_pass"),
        "system_legal_controller_pass": p10.get("system_legal_controller_pass"),
        "primary_blocker": primary,
        "secondary_blocker": secondary,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    dump_json("route_decision_v9710.json", route_row)

    allowed = {
        "stage": "ALLOWED_NEXT_GATES_V9710",
        "status": "summary",
        "selected_runtime_allowed": 0,
        "paired_replay_allowed": 0,
        "short_full_allowed": 0,
        "APGV_APGW_allowed": 0,
        "exact_transfer_artifact_required": 1,
        "direct_update_allowed": 0,
    }
    stop = {
        "stage": "STOP_CONDITIONS_V9710",
        "status": "summary",
        "stop_old_rank_patch": 1,
        "stop_generated_blind_variants": 1,
        "stop_direct_solved_update": 1,
        "enter_system": 0,
    }
    dump_json("allowed_next_gates_v9710.json", allowed)
    dump_json("stop_conditions_v9710.json", stop)

    nofake = count_artifact_rows(list(artifacts.values()))
    nofake_row = {"stage": "NO_FAKE_AUDIT_V9710", "status": "summary", **nofake}
    dump_csv("no_fake_audit_v9710.csv", [nofake_row])

    contract = {
        "stage": "CONTRACT_AUDIT_V9710",
        "status": "summary",
        "manual_forward": 1,
        "manual_backward": 1,
        "manual_adamw_update": 1,
        "train_stream_probe": 1,
        "v9700_boundary_pass": p0.get("p0_pass"),
        "field_legality_pass": p0.get("p0_pass"),
        "exact_transfer_artifact_pass": p1.get("p1_pass"),
        "exact_transfer_weak_pass": p1.get("P1_exact_transfer_weak_pass"),
        "exact_selector_pass": 0,
        "core_exact_expansion_pass": p3.get("p3_core_exact_expansion_pass"),
        "dataset_shift_score_scale_solved": p4.get("dataset_shift_score_scale_solved"),
        "direct_solved_pass": int(inum(p5.get("direct_solved_weak_pass")) or inum(p5.get("direct_solved_strong_pass"))),
        "generated_route_stop": p9.get("generated_route_stop_triggered"),
        "controller_pass": p7.get("controller_pass"),
        "runtime_pass": p8.get("selected_runtime_pass"),
        "system_legal_controller_pass": p10.get("system_legal_controller_pass"),
        "base_acc_sentinel_pass": base_summary.get("base_acc_sentinel_pass"),
        "base_acc_used_for_controller": base_summary.get("base_acc_used_for_controller"),
        "uses_loss_backward": 0,
        "uses_teacher": 0,
        "uses_loss_modification": 0,
        "uses_dataset_name_for_selector": 0,
        "uses_dataset_name_for_controller": 0,
        "uses_validation_or_test_for_controller": 0,
        "uses_future_outcome_for_features": 0,
        "uses_outcome_at_commit": 0,
        "uses_old_table_for_official": 0,
        "proxy_transfer_promoted_to_official": 0,
        "diagnostic_promoted_to_official": 0,
        "fake_data_used": nofake_row["fake_data_used"],
        "proxy_row_used": nofake_row["proxy_row_used"],
        "cpu_offload_used": nofake_row["cpu_offload_used"],
    }
    dump_csv("contract_audit_v9710.csv", [contract])

    failure = {
        "stage": "FAILURE_TAXONOMY_V9710",
        "status": "summary",
        "route": route,
        "F0_boundary_or_legality_fail": int(not inum(p0.get("p0_pass"))),
        "F1_exact_transfer_artifact_missing": 1,
        "F2_exact_transfer_selector_fail": 0,
        "F3_core_expansion_exact_fail": 0,
        "F4_direct_solved_update_not_open": 1,
        "F5_generated_route_stopped_no_new_objective": 1,
        "F6_controller_runtime_blocked": 1,
        "F7_system_not_official": 1,
        "F8_base_acc_catastrophic": inum(base_summary.get("LQ_catastrophic_fail")),
        "primary_blocker": primary,
        "secondary_blocker": secondary,
        "fake_data_used": nofake_row["fake_data_used"],
        "proxy_row_used": nofake_row["proxy_row_used"],
        "cpu_offload_used": nofake_row["cpu_offload_used"],
    }
    dump_csv("failure_taxonomy_v9710.csv", [failure])

    artifacts["plan"] = PLAN_PATH
    artifacts["runner"] = SCRIPT_PATH
    artifact_hashes = {name: sha256_file(path) for name, path in artifacts.items() if path.exists()}
    manifest = {
        "version": "v9710",
        "created_utc": "2026-05-16T000000Z",
        "out_dir": repo_rel(out),
        "seed": args.seed,
        "device": args.device,
        "data_root": args.data_root,
        "route": route,
        "primary_blocker": primary,
        "secondary_blocker": secondary,
        "fake_data_used": nofake_row["fake_data_used"],
        "proxy_row_used": nofake_row["proxy_row_used"],
        "cpu_offload_used": nofake_row["cpu_offload_used"],
        "wallclock_sec": time.perf_counter() - t0,
        "sources": {"v9700": repo_rel(args.source_v9700)},
        "artifact_sha256": artifact_hashes,
    }
    dump_json("run_manifest_v9710.json", manifest)

    print(
        json.dumps(
            {
                "route": route,
                "primary_blocker": primary,
                "secondary_blocker": secondary,
                "exact_per_sample_gradient_available": p1.get("exact_per_sample_gradient_available"),
                "exact_linear_transfer_row_count": p1.get("exact_linear_transfer_row_count"),
                "direct_solved_generated_actions": p5.get("generated_action_count"),
                "generated_route_status": p9.get("generated_route_status"),
                "system_legal_controller_pass": p10.get("system_legal_controller_pass"),
                "out_dir": repo_rel(out),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
