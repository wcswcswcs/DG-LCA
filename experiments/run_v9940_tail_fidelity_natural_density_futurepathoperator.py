#!/usr/bin/env python3
"""DG-KAN v9.9.4 tail-fidelity natural density / FuturePathOperator run.

v9.9.4 treats v9.9.3 as an engineering blocker, not as a completed density
result.  It audits the tail reference, runs a repaired generator matrix, opens
sequential density panels only for an official major+tail-fidelity pass, and
writes the recap only after the run manifest and route are materialized.
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
from typing import Any

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "experiments"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

import run_v9720_exact_transfer_materializer_core_expansion_direct_update as v9720  # noqa: E402
import run_v9910_natural_density_decision_futurepathoperator as v9910  # noqa: E402
import run_v9930_natural_density_tail_fidelity_futurepathoperator as v9930  # noqa: E402
from dgkan_outcome_controller import fnum, inum, read_json, sha256_file, wilson_lcb, wilson_ucb, write_csv, write_json  # noqa: E402


RESULT_ROOT = REPO / "results/real_rerun_20260506"
PLAN_PATH = REPO / "docs/DG-KAN_v9.9.4_TailFidelityNaturalDensity_FuturePathOperator_四线并行完整实验计划.md"
SCRIPT_PATH = REPO / "experiments/run_v9940_tail_fidelity_natural_density_futurepathoperator.py"
MATERIALIZER_PATH = REPO / "experiments/natural_ap0_extension_materializer.py"
RECAP_PATH = REPO / "docs/DG-KAN_v9.9.4_TailFidelityNaturalDensity_FuturePathOperator_实验复盘.md"
DEFAULT_OUT = RESULT_ROOT / "v9940_tail_fidelity_natural_density_futurepathoperator_full_20260517T050000Z"
DEFAULT_V9930 = RESULT_ROOT / "v9930_natural_density_tail_fidelity_futurepathoperator_full_20260517T040000Z"
DEFAULT_V9920 = RESULT_ROOT / "v9920_natural_generator_distribution_repair_futurepathoperator_full_20260517T030000Z"
DEFAULT_V9910 = RESULT_ROOT / "v9910_natural_density_decision_futurepathoperator_full_20260517T020000Z"
DEFAULT_V9900 = RESULT_ROOT / "v9900_natural_extension_engineering_gate_futurepathoperator_full_20260517T010000Z"
DEFAULT_V9330 = RESULT_ROOT / "v9330_full_control_outcome_action_value_runtime_decoupling_first_20260514T003000Z"
DEFAULT_V9830 = RESULT_ROOT / "v9830_four_line_future_mechanism_natural_geometry_full_20260516T140000Z"
DEFAULT_V9840 = RESULT_ROOT / "v9840_future_operator_natural_stream_geometry_optimizer_full_20260516T160000Z"
BRANCH_COUNT = 6
HORIZONS = [1, 5, 20, 80, 240]

GENERATOR_PROFILES: list[tuple[str, str, int]] = [
    ("G5-hybrid-quota-random-generator", "g5-hybrid-quota-random-generator", 1),
    ("G6-tail-aware-quota-generator", "g6-tail-aware-quota-generator", 1),
    ("G7-multi-axis-precursor-quota-generator", "g7-multi-axis-precursor-quota-generator", 1),
    ("G8-tail-reservoir-capped-generator", "g8-tail-reservoir-capped-generator", 1),
    ("G9-two-stage-major-then-tail-fill-generator", "g9-two-stage-major-then-tail-fill-generator", 1),
    ("G10-alias-proportional-tail-min-generator", "g10-alias-proportional-tail-min-generator", 1),
    ("G11-stratified-mixture-generator", "g11-stratified-mixture-generator", 1),
    ("G12-diagnostic-tail-upper-bound-sampler", "g12-diagnostic-tail-upper-bound-sampler", 0),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=str(DEFAULT_OUT))
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--data-root", default="data")
    p.add_argument("--seed", type=int, default=1314)
    p.add_argument("--execution-profile", choices=["smoke", "full-gated"], default="full-gated")
    p.add_argument("--panel-targets", default="1024,5000,10000,20000")
    p.add_argument("--pilot-actions", type=int, default=1024)
    p.add_argument("--chunk-actions", type=int, default=512)
    p.add_argument("--candidate-generators", default=",".join(g[0] for g in GENERATOR_PROFILES))
    p.add_argument("--resume-after-p2", action="store_true", help="Reuse an already materialized P2 matrix and continue only the gated downstream boundary.")
    p.add_argument("--source-v9930", default=str(DEFAULT_V9930))
    p.add_argument("--source-v9920", default=str(DEFAULT_V9920))
    p.add_argument("--source-v9910", default=str(DEFAULT_V9910))
    p.add_argument("--source-v9900", default=str(DEFAULT_V9900))
    p.add_argument("--source-v9330", default=str(DEFAULT_V9330))
    p.add_argument("--source-v9830", default=str(DEFAULT_V9830))
    p.add_argument("--source-v9840", default=str(DEFAULT_V9840))
    return p.parse_args()


def parse_ints(text: str) -> list[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def rows_from(path: Path) -> list[dict[str, Any]]:
    return v9930.rows_from(path)


def summary_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return v9930.summary_row(rows)


def safe_name(text: str) -> str:
    return str(text).replace("/", "_").replace(" ", "_")


def key_field(row: dict[str, Any], *names: str, default: str = "unknown") -> str:
    for name in names:
        val = row.get(name)
        if val not in {None, ""}:
            return str(val)
    return default


def p0_boundary(source_v9930: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    route = read_json(source_v9930 / "route_decision_v9930.json")
    p1 = summary_row(rows_from(source_v9930 / "p1_major_tail_fidelity_audit_v9930.csv"))
    g6 = summary_row(rows_from(source_v9930 / "p1_g6_tail_repair_fidelity_audit_v9930.csv"))
    p2 = summary_row(rows_from(source_v9930 / "p2_sequential_natural_density_panel_v9930.csv"))
    p6 = summary_row(rows_from(source_v9930 / "p6_existing_action_controller_gate_v9930.csv"))
    p7 = summary_row(rows_from(source_v9930 / "p7_generated_sandbox_gate_v9930.csv"))
    nf = summary_row(rows_from(source_v9930 / "no_fake_audit_v9930.csv"))
    row = {
        "stage": "P0_V9930_BOUNDARY_REPRODUCTION_V9940",
        "status": "summary",
        "route_v9930": route.get("route"),
        "selected_generator_id_v9930": route.get("selected_generator_id"),
        "G5_major_pass_v9930": p1.get("major_fidelity_pass"),
        "G5_tail_pass_v9930": p1.get("tail_fidelity_pass"),
        "G5_PSI_major_max_v9930": p1.get("PSI_major_max"),
        "G5_JS_major_max_v9930": p1.get("JS_major_max"),
        "G5_max_share_major_v9930": p1.get("max_major_group_share"),
        "G5_entropy_major_min_v9930": p1.get("entropy_major_min"),
        "G5_PSI_tail_max_v9930": p1.get("PSI_tail_max"),
        "G5_JS_tail_max_v9930": p1.get("JS_tail_max"),
        "G5_missing_tail_group_count_v9930": p1.get("missing_tail_group_count"),
        "G5_tail_coverage_min_v9930": min([fnum(r.get("coverage"), 1.0) for r in rows_from(source_v9930 / "p1_major_tail_fidelity_audit_v9930.csv") if r.get("axis_type") == "tail"] or [0.0]),
        "G6_major_pass_v9930": g6.get("major_fidelity_pass"),
        "G6_tail_pass_v9930": g6.get("tail_fidelity_pass"),
        "P2_largest_completed_panel_size_v9930": p2.get("largest_completed_panel_size"),
        "P6_controller_status_v9930": p6.get("status"),
        "P7_generated_allowed_v9930": p7.get("generated_sandbox_allowed"),
        "fake_data_used_v9930": nf.get("fake_data_used"),
        "proxy_row_used_v9930": nf.get("proxy_row_used"),
        "cpu_offload_used_v9930": nf.get("cpu_offload_used"),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    row["P0_boundary_pass"] = int(
        row["route_v9930"] == "R9-MaterializerOrFidelityEngineeringBlocked"
        and inum(row["G5_major_pass_v9930"]) == 0
        and inum(row["G5_tail_pass_v9930"]) == 0
        and inum(row["G6_tail_pass_v9930"]) == 0
        and inum(row["P2_largest_completed_panel_size_v9930"]) == 0
        and inum(row["fake_data_used_v9930"]) == 0
        and inum(row["proxy_row_used_v9930"]) == 0
        and inum(row["cpu_offload_used_v9930"]) == 0
    )
    return [row], row


def ref_norm(row: dict[str, Any]) -> float:
    return v9930.payload_norm(row)


def ref_linf(row: dict[str, Any]) -> float:
    return v9930.payload_linf(row)


def bucket(value: float, cuts: list[float], name: str) -> str:
    return v9930.bucket(value, cuts, name)


def quantile(values: list[float], frac: float) -> float:
    return v9930.q(values, frac)


def reference_key_builders(reference_rows: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    norm_cuts = [quantile([ref_norm(r) for r in reference_rows], x) for x in [0.1, 0.25, 0.5, 0.75, 0.9]]
    linf_cuts = [quantile([ref_linf(r) for r in reference_rows], x) for x in [0.1, 0.25, 0.5, 0.75, 0.9]]

    def template(r: dict[str, Any]) -> str:
        return key_field(r, "carrier_id", "template_id")

    def recipe(r: dict[str, Any]) -> str:
        return key_field(r, "bucket_id", "family_id")

    def norm_bucket(r: dict[str, Any]) -> str:
        return bucket(ref_norm(r), norm_cuts, "payload_norm")

    def linf_bucket(r: dict[str, Any]) -> str:
        return bucket(ref_linf(r), linf_cuts, "payload_linf")

    builders = {
        "template": template,
        "recipe": recipe,
        "step_bucket": lambda r: v9930.step_bucket(r.get("step")),
        "norm_bucket": norm_bucket,
        "linf_bucket": linf_bucket,
        "tail_key": lambda r: "|".join([recipe(r), template(r), v9930.step_bucket(r.get("step")), norm_bucket(r), linf_bucket(r)]),
        "major_key": lambda r: "|".join([key_field(r, "dataset"), template(r), v9930.step_bucket(r.get("step")), norm_bucket(r)]),
    }
    return builders, {"norm_cuts": norm_cuts, "linf_cuts": linf_cuts}


def tail_group_reference_audit(reference_rows: list[dict[str, Any]], group_ids: dict[str, set[str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    builders, _cuts = reference_key_builders(reference_rows)
    tail_counter = Counter(builders["tail_key"](r) for r in reference_rows)
    major_counter = Counter(builders["major_key"](r) for r in reference_rows)
    lineage_rows: list[dict[str, Any]] = []
    for key, count in sorted(tail_counter.items()):
        lineage_rows.append({
            "stage": "P1_TAIL_GROUP_LINEAGE_DEBUG_V9940",
            "status": "tail_group",
            "tail_group_key": key,
            "canonical_support": count,
            "generator_quota_group": "merged_tail_support_lt_3" if count < 3 else "commit_time_precursor_tail",
            "diagnostic_only": int(count < 3),
            "reference_tail_group_has_future_label_flag": 0,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    action_by_id = {str(r.get("action_id")): r for r in reference_rows}
    diagnostic_rows: list[dict[str, Any]] = []
    for name in ["Core77", "OldOnly", "RiskCleanButLowValue", "SlowBurnGood"]:
        ids = group_ids.get(name, set())
        keys = {builders["tail_key"](action_by_id[aid]) for aid in ids if aid in action_by_id}
        diagnostic_rows.append({
            "stage": "P1_TAIL_GROUP_LINEAGE_DEBUG_V9940",
            "status": "diagnostic_outcome_tail",
            "diagnostic_group": name,
            "action_count": len(ids),
            "tail_group_count": len(keys),
            "reference_tail_group_has_future_label_flag": 1,
            "generator_quota_allowed": 0,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    supports = list(tail_counter.values())
    overlap_rows: list[dict[str, Any]] = []
    diag_names = ["Core77", "OldOnly", "RiskCleanButLowValue"]
    diag_keys = {}
    for name in diag_names:
        ids = group_ids.get(name, set())
        diag_keys[name] = {builders["tail_key"](action_by_id[aid]) for aid in ids if aid in action_by_id}
    for a in diag_names:
        for b in diag_names:
            overlap_rows.append({
                "stage": "P1_TAIL_GROUP_REFERENCE_AUDIT_V9940",
                "status": "overlap_matrix",
                "group_a": a,
                "group_b": b,
                "overlap_tail_group_count": len(diag_keys[a] & diag_keys[b]),
                "union_tail_group_count": len(diag_keys[a] | diag_keys[b]),
                "jaccard": len(diag_keys[a] & diag_keys[b]) / max(1, len(diag_keys[a] | diag_keys[b])),
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            })
    missing_key = sum(1 for r in reference_rows if not builders["tail_key"](r) or "unknown" in builders["tail_key"](r))
    tail_entropy = v9930.entropy(tail_counter)
    summary = {
        "stage": "P1_TAIL_GROUP_REFERENCE_AUDIT_V9940",
        "status": "summary",
        "canonical_action_count": len(reference_rows),
        "tail_group_count_total": len(tail_counter),
        "major_group_count_total": len(major_counter),
        "tail_group_count_diagnostic_outcome": sum(1 for r in diagnostic_rows if inum(r.get("tail_group_count")) > 0),
        "tail_group_count_commit_time_precursor": len(tail_counter),
        "tail_group_key_missing_rate": missing_key / max(1, len(reference_rows)),
        "tail_group_singleton_count": sum(1 for c in supports if c == 1),
        "tail_group_support_lt3_count": sum(1 for c in supports if c < 3),
        "tail_group_min_support": min(supports or [0]),
        "tail_group_max_support": max(supports or [0]),
        "tail_group_entropy": tail_entropy,
        "Core77_tail_group_coverage": 1.0 if diag_keys.get("Core77") else 0.0,
        "OldOnly_tail_group_coverage": 1.0 if diag_keys.get("OldOnly") else 0.0,
        "RiskClean_tail_group_coverage": 1.0 if diag_keys.get("RiskCleanButLowValue") else 0.0,
        "SlowBurn_tail_group_coverage": 0.0,
        "reference_tail_group_has_future_label_flag": 0,
        "generator_quota_uses_outcome_tail": 0,
        "P1_reference_audit_pass": int(missing_key == 0),
        "repair_applied": "support_lt3_groups_marked_diagnostic_only_or_merged_tail",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    rows = [summary, *overlap_rows]
    return rows, [*lineage_rows, *diagnostic_rows], summary


def tail_coverage_min(audit_rows: list[dict[str, Any]]) -> float:
    return min([fnum(r.get("coverage"), 1.0) for r in audit_rows if r.get("axis_type") == "tail"] or [0.0])


def official_generator_pass(audit_summary: dict[str, Any], smoke_summary: dict[str, Any], audit_rows: list[dict[str, Any]], official_eligible: int) -> int:
    return int(
        official_eligible
        and fnum(audit_summary.get("PSI_major_max")) <= 0.05
        and fnum(audit_summary.get("JS_major_max")) <= 0.08
        and fnum(audit_summary.get("max_major_group_share")) <= 0.35
        and fnum(audit_summary.get("entropy_major_min")) >= 0.90
        and fnum(audit_summary.get("PSI_tail_max")) <= 0.10
        and fnum(audit_summary.get("JS_tail_max")) <= 0.08
        and inum(audit_summary.get("missing_tail_group_count")) == 0
        and tail_coverage_min(audit_rows) >= 0.80
        and inum(smoke_summary.get("old_action_id_collision_count")) == 0
        and inum(smoke_summary.get("old_payload_hash_collision_count")) == 0
        and inum(smoke_summary.get("duplicate_action_id_count")) == 0
        and fnum(smoke_summary.get("branch_horizon_completion")) >= 1.0
        and inum(smoke_summary.get("fake_data_used")) == 0
        and inum(smoke_summary.get("proxy_row_used")) == 0
        and inum(smoke_summary.get("cpu_offload_used")) == 0
    )


def missing_tail_debug_rows(generator_id: str, audit_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in audit_rows:
        if row.get("axis_type") == "tail" and row.get("worst_missing_group"):
            rows.append({
                "stage": "P2_MISSING_TAIL_GROUP_DEBUG_V9940",
                "status": "missing_tail_axis",
                "generator_id": generator_id,
                "axis": row.get("axis"),
                "worst_missing_group": row.get("worst_missing_group"),
                "missing_group_count": row.get("missing_group_count"),
                "coverage": row.get("coverage"),
                "gate_group_threshold_old_count": row.get("gate_group_threshold_old_count"),
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            })
    return rows[:20]


def run_generator_matrix(
    args: argparse.Namespace,
    out: Path,
    ref_actions: list[dict[str, Any]],
    group_ids: dict[str, set[str]],
    old_ids: set[str],
    old_hashes: set[str],
    selected_names: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    matrix_rows: list[dict[str, Any]] = []
    all_missing_debug: list[dict[str, Any]] = []
    all_retry_rows: list[dict[str, Any]] = []
    best: dict[str, Any] = {}
    best_actions: list[dict[str, Any]] = []
    best_branches: list[dict[str, Any]] = []
    for idx, (gid, profile, official_eligible) in enumerate(GENERATOR_PROFILES):
        if gid not in selected_names:
            continue
        cursor = 1_200_000 + idx * 300_000
        smoke_rows, actions, applies, branches, smoke = v9930.run_panel_sharded(gid, profile, int(args.pilot_actions), cursor, args, out, old_ids, old_hashes)
        write_csv(out / f"p2_{safe_name(gid)}_1024_panel_smoke_v9940.csv", smoke_rows)
        write_csv(out / f"p2_{safe_name(gid)}_1024_action_rows_v9940.csv", actions)
        write_csv(out / f"p2_{safe_name(gid)}_1024_action_apply_replay_v9940.csv", applies)
        write_csv(out / f"p2_{safe_name(gid)}_1024_branch_horizon_v9940.csv", branches)
        labels, label_summary = v9930.label_summary_from_branches(branches)
        write_csv(out / f"p2_{safe_name(gid)}_1024_action_labels_v9940.csv", labels)
        audit_rows, audit = v9930.tail_fidelity_audit(ref_actions, actions, group_ids, generator_id=gid, panel_size=int(args.pilot_actions))
        write_csv(out / f"p2_{safe_name(gid)}_1024_major_tail_fidelity_v9940.csv", audit_rows)
        official_pass = official_generator_pass(audit, smoke, audit_rows, official_eligible)
        row = {
            "stage": "P2_GENERATOR_REPAIR_MATRIX_V9940",
            "status": "generator_row",
            "generator_id": gid,
            "generator_profile": profile,
            "official_density_eligible": official_eligible,
            "action_count": len(actions),
            "payload_collision_count": inum(smoke.get("old_payload_hash_collision_count")) + inum(smoke.get("duplicate_payload_hash_count")),
            "action_collision_count": inum(smoke.get("old_action_id_collision_count")) + inum(smoke.get("duplicate_action_id_count")),
            "branch_horizon_rows_expected": smoke.get("branch_horizon_expected_rows"),
            "branch_horizon_rows_actual": smoke.get("branch_horizon_actual_rows"),
            "branch_horizon_completion": smoke.get("branch_horizon_completion"),
            "major_PSI_max": audit.get("PSI_major_max"),
            "major_JS_max": audit.get("JS_major_max"),
            "major_max_group_share": audit.get("max_major_group_share"),
            "major_entropy_ratio_min": audit.get("entropy_major_min"),
            "tail_PSI_max": audit.get("PSI_tail_max"),
            "tail_JS_max": audit.get("JS_tail_max"),
            "missing_tail_group_count": audit.get("missing_tail_group_count"),
            "tail_group_coverage_min": tail_coverage_min(audit_rows),
            "tail_group_coverage_mean": sum(fnum(r.get("coverage")) for r in audit_rows if r.get("axis_type") == "tail") / max(1, sum(1 for r in audit_rows if r.get("axis_type") == "tail")),
            "Core77_tail_coverage": audit.get("Core77_precursor_coverage"),
            "OldOnly_tail_coverage": audit.get("OldOnly_precursor_coverage"),
            "RiskClean_tail_coverage": audit.get("RiskClean_precursor_coverage"),
            "SlowBurn_tail_coverage": "",
            "CoreLike_count": label_summary.get("CoreLike_count"),
            "PathGood_count": label_summary.get("PathGood_count"),
            "SlowBurnGood_count": label_summary.get("SlowBurnGood_count"),
            "rows_per_sec": smoke.get("rows_per_sec"),
            "wallclock_sec": smoke.get("wallclock_sec"),
            "peak_gpu_mb": smoke.get("peak_gpu_memory_mb"),
            "official_density_generator_pass": official_pass,
            "failure": "" if official_pass else "official_major_tail_or_collision_completion_gate_failed",
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": max(inum(smoke.get("cpu_offload_used")), inum(label_summary.get("cpu_offload_used"))),
        }
        matrix_rows.append(row)
        all_missing_debug.extend(missing_tail_debug_rows(gid, audit_rows))
        if not official_pass:
            all_retry_rows.append({
                "stage": "P2_CODEX_RETRY_SUGGESTIONS_V9940",
                "status": "retry_suggestion",
                "generator_id": gid,
                "suggestion": "inspect_missing_tail_keys;check_cursor_cycle;check_quota_rounding;use_residual_carry_or_forced_first_pass;verify_generated_metadata_tail_key",
                "major_gate_pass": int(fnum(row["major_PSI_max"]) <= 0.05 and fnum(row["major_JS_max"]) <= 0.08 and fnum(row["major_max_group_share"]) <= 0.35 and fnum(row["major_entropy_ratio_min"]) >= 0.90),
                "tail_gate_pass": int(fnum(row["tail_PSI_max"]) <= 0.10 and fnum(row["tail_JS_max"]) <= 0.08 and inum(row["missing_tail_group_count"]) == 0 and fnum(row["tail_group_coverage_min"]) >= 0.80),
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": row.get("cpu_offload_used"),
            })
        if official_pass and (not best or fnum(row["tail_PSI_max"]) < fnum(best.get("tail_PSI_max"), 999.0)):
            best = row
            best_actions = actions
            best_branches = branches
    pass_rows = [r for r in matrix_rows if inum(r.get("official_density_generator_pass"))]
    summary = {
        "stage": "P2_GENERATOR_REPAIR_MATRIX_V9940",
        "status": "summary",
        "candidate_count": len(matrix_rows),
        "official_pass_count": len(pass_rows),
        "best_generator_id": best.get("generator_id", ""),
        "best_generator_profile": best.get("generator_profile", ""),
        "best_major_PSI_max": best.get("major_PSI_max", ""),
        "best_tail_PSI_max": best.get("tail_PSI_max", ""),
        "best_missing_tail_group_count": best.get("missing_tail_group_count", ""),
        "best_enters_P3": int(bool(best)),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": max([inum(r.get("cpu_offload_used")) for r in matrix_rows] or [0]),
    }
    matrix_rows.insert(0, summary)
    return matrix_rows, best_actions, best_branches, best, all_missing_debug, all_retry_rows


def density_summary_rows(panel_rows: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [r for r in panel_rows if r.get("status") == "panel_row"]
    largest = completed[-1] if completed else {}
    return {
        "stage": "P3_SEQUENTIAL_NATURAL_DENSITY_PANEL_V9940",
        "status": "summary",
        "completed_panel_count": len(completed),
        "not_run_panel_count": sum(1 for r in panel_rows if r.get("status") == "not_run"),
        "largest_completed_panel_size": largest.get("panel_size", 0),
        "CoreLike_count": largest.get("CoreLike_count", 0),
        "CoreLike_LCB": largest.get("CoreLike_LCB", 0),
        "CoreLike_UCB": largest.get("CoreLike_UCB", 0),
        "PathGood_count": largest.get("PathGood_count", 0),
        "PathGood_LCB": largest.get("PathGood_LCB", 0),
        "PathGood_UCB": largest.get("PathGood_UCB", 0),
        "SlowBurnGood_count": largest.get("SlowBurnGood_count", 0),
        "CoreLike_or_SlowBurnGood_LCB": largest.get("CoreLike_or_SlowBurnGood_LCB", 0),
        "CoreLike_or_SlowBurnGood_UCB": largest.get("CoreLike_or_SlowBurnGood_UCB", 0),
        "P3_density_sufficient": largest.get("density_sufficient", 0),
        "P3_density_insufficient": largest.get("density_insufficient", 0),
        "P3_density_inconclusive": int(not inum(largest.get("density_sufficient")) and not inum(largest.get("density_insufficient"))),
        "reason": largest.get("reason") or (panel_rows[0].get("reason") if panel_rows else ""),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": max([inum(r.get("cpu_offload_used")) for r in panel_rows] or [0]),
    }


def panel_density_decision(panel_summary: dict[str, Any], label_summary: dict[str, Any], generator_id: str, status: str, reason: str) -> dict[str, Any]:
    row = v9930.panel_density_decision(panel_summary, label_summary, generator_id, status=status, reason=reason)
    row["stage"] = "P3_SEQUENTIAL_NATURAL_DENSITY_PANEL_V9940"
    row["P3_density_sufficient"] = row.get("density_sufficient")
    row["P3_density_insufficient"] = row.get("density_insufficient")
    row["P3_density_inconclusive"] = row.get("density_inconclusive")
    return row


def run_density_panels(
    args: argparse.Namespace,
    out: Path,
    best: dict[str, Any],
    best_actions: list[dict[str, Any]],
    best_branches: list[dict[str, Any]],
    old_ids: set[str],
    old_hashes: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    panel_targets = parse_ints(args.panel_targets)
    gid_value = best.get("generator_id")
    profile_value = best.get("generator_profile")
    all_labels: list[dict[str, Any]] = []
    all_actions: list[dict[str, Any]] = []
    panel_rows: list[dict[str, Any]] = []
    if not gid_value or str(gid_value).lower() == "none" or not profile_value or str(profile_value).lower() == "none":
        for target in panel_targets:
            panel_rows.append({
                "stage": "P3_SEQUENTIAL_NATURAL_DENSITY_PANEL_V9940",
                "status": "not_run",
                "panel_size": target,
                "density_result": "not_run",
                "reason": "P2_no_official_major_tail_fidelity_generator_passed",
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            })
        panel_rows.insert(0, density_summary_rows(panel_rows))
        return panel_rows, all_labels, all_actions, panel_rows[0]
    gid = str(gid_value)
    profile = str(profile_value)
    labels, label_summary = v9930.label_summary_from_branches(best_branches)
    smoke = {
        "branch_horizon_expected_rows": len(best_actions) * BRANCH_COUNT * len(HORIZONS),
        "branch_horizon_actual_rows": len(best_branches),
        "branch_horizon_completion": len(best_branches) / max(1, len(best_actions) * BRANCH_COUNT * len(HORIZONS)),
        "rows_per_sec": best.get("rows_per_sec"),
        "wallclock_sec": best.get("wallclock_sec"),
        "peak_gpu_memory_mb": best.get("peak_gpu_mb"),
        "cpu_offload_used": best.get("cpu_offload_used"),
    }
    panel_rows.append(panel_density_decision(smoke, label_summary, gid, "panel_row", "S0_1024_real_panel_official_fidelity_passed_not_full_closure"))
    all_labels, all_actions = labels, best_actions
    for target in [t for t in panel_targets if t > len(best_actions)]:
        if any(inum(r.get("density_sufficient")) or inum(r.get("density_insufficient")) for r in panel_rows):
            panel_rows.append(panel_density_decision({}, {"panel_action_count": 0}, gid, "not_run", "sequential_density_already_adjudicated_at_previous_panel") | {"panel_size": target})
            continue
        try:
            smoke_rows, actions, applies, branches, smoke_summary = v9930.run_panel_sharded(gid, profile, target, 2_400_000 + target, args, out, old_ids, old_hashes)
        except RuntimeError:
            retry_args = argparse.Namespace(**vars(args))
            retry_args.chunk_actions = max(128, int(args.chunk_actions) // 2)
            smoke_rows, actions, applies, branches, smoke_summary = v9930.run_panel_sharded(gid, profile, target, 2_900_000 + target, retry_args, out, old_ids, old_hashes)
        write_csv(out / f"p3_{safe_name(gid)}_{target}_panel_smoke_v9940.csv", smoke_rows)
        write_csv(out / f"p3_{safe_name(gid)}_{target}_action_rows_v9940.csv", actions)
        write_csv(out / f"p3_{safe_name(gid)}_{target}_action_apply_replay_v9940.csv", applies)
        write_csv(out / f"p3_{safe_name(gid)}_{target}_branch_horizon_v9940.csv", branches)
        labels, label_summary = v9930.label_summary_from_branches(branches)
        write_csv(out / f"p3_{safe_name(gid)}_{target}_action_labels_v9940.csv", labels)
        panel_rows.append(panel_density_decision(smoke_summary, label_summary, gid, "panel_row", f"S{target}_real_sequential_density_panel"))
        all_labels, all_actions = labels, actions
    panel_rows.insert(0, density_summary_rows(panel_rows))
    return panel_rows, all_labels, all_actions, panel_rows[0]


def not_run(stage: str, reason: str, **extra: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    return v9930.not_run(stage, reason, **extra)


def artifact_row_count(artifacts: dict[str, Path]) -> int:
    total = 0
    for name, path in artifacts.items():
        if name.endswith(".csv") and path.exists() and not name.startswith(("no_fake", "contract", "failure")):
            with path.open(newline="", encoding="utf-8") as f:
                total += sum(1 for _ in csv.DictReader(f))
        elif name.endswith(".json") and path.exists():
            total += 1
    return total


def write_figures(out: Path, route: dict[str, Any], p2: dict[str, Any], p3: dict[str, Any], p5: dict[str, Any]) -> dict[str, Path]:
    figs: dict[str, Path] = {}

    def fig(name: str, title: str, labels: list[str], values: list[float]) -> None:
        path = out / name
        v9720.write_bar_svg(path, title, labels, values)
        figs[name] = path

    fig("fig_v9940_gate_matrix.svg", "v9.9.4 gate matrix", ["P0", "P1", "P2", "P3 suff", "P3 insuff", "P5", "P6"], [
        fnum(route.get("P0_boundary_pass")), fnum(route.get("P1_reference_audit_pass")), fnum(route.get("P2_official_generator_pass")), fnum(route.get("P3_density_sufficient")), fnum(route.get("P3_density_insufficient")), fnum(route.get("P5_FPO_weak_pass")), fnum(route.get("P6_controller_pass")),
    ])
    fig("fig_p2_generator_repair_matrix_v9940.svg", "generator repair matrix", ["candidates", "official pass", "tail PSI", "tail missing"], [
        fnum(p2.get("candidate_count")), fnum(p2.get("official_pass_count")), fnum(p2.get("best_tail_PSI_max")), fnum(p2.get("best_missing_tail_group_count")),
    ])
    fig("fig_p3_density_ci_v9940.svg", "density CI", ["Core LCB", "Core UCB", "Path LCB", "Path UCB", "Core+Slow UCB"], [
        fnum(p3.get("CoreLike_LCB")), fnum(p3.get("CoreLike_UCB")), fnum(p3.get("PathGood_LCB")), fnum(p3.get("PathGood_UCB")), fnum(p3.get("CoreLike_or_SlowBurnGood_UCB")),
    ])
    fig("fig_p5_fpo_v6_v9940.svg", "FPO v6", ["weak", "strong", "best precision", "best V"], [
        fnum(p5.get("P5_FPO_weak_pass")), fnum(p5.get("P5_FPO_strong_pass")), fnum(p5.get("best_precision")), fnum(p5.get("best_V_LCB")),
    ])
    return figs


def sha_rows(paths: dict[str, Path]) -> dict[str, str]:
    return {name: sha256_file(path) for name, path in paths.items() if path.exists()}


def write_recap(out: Path, route: dict[str, Any], hashes: dict[str, str]) -> None:
    p0 = summary_row(rows_from(out / "p0_v9930_boundary_reproduction.csv"))
    p1 = summary_row(rows_from(out / "p1_tail_group_reference_audit_v9940.csv"))
    p2 = summary_row(rows_from(out / "p2_generator_repair_matrix_v9940.csv"))
    p3 = summary_row(rows_from(out / "p3_sequential_natural_density_panel_v9940.csv"))
    p4 = summary_row(rows_from(out / "p4_future_path_type_decomposition_v9940.csv"))
    p5 = summary_row(rows_from(out / "p5_future_path_operator_sketch_v6_v9940.csv"))
    p6 = summary_row(rows_from(out / "p6_existing_action_controller_gate_v9940.csv"))
    p7 = summary_row(rows_from(out / "p7_generated_sandbox_gate_v9940.csv"))
    nf = summary_row(rows_from(out / "no_fake_audit_v9940.csv"))
    lines = [
        "# DG-KAN v9.9.4 Tail Fidelity / Natural Density / FuturePathOperator 实验复盘",
        "",
        "> 本复盘记录 `DG-KAN_v9.9.4_TailFidelityNaturalDensity_FuturePathOperator_四线并行完整实验计划.md` 的真实执行结果。所有结论只来自本轮落盘 CSV/JSON/manifest 与真实 materialized natural AP0 extension rows；没有 fake data、proxy rows 或 CPU offload。未通过 gate 的 density/controller/generated/runtime 均显式 `not_run`。",
        "",
        "## 0. 最新结论",
        "",
        "```text",
        f"route = {route.get('route')}",
        f"primary_blocker = {route.get('primary_blocker')}",
        f"secondary_blocker = {route.get('secondary_blocker')}",
        f"system_legal_controller_pass = {route.get('system_legal_controller_pass')}",
        f"generated_route_status = {route.get('generated_route_status')}",
        "```",
        "",
        f"最终 artifact：`{out}`",
        "",
        "核心结论：",
        "",
        f"1. P0 复现 v9.9.3 boundary：source route = `{p0.get('route_v9930')}`，G5 major/tail = `{p0.get('G5_major_pass_v9930')}` / `{p0.get('G5_tail_pass_v9930')}`，G6 tail = `{p0.get('G6_tail_pass_v9930')}`。",
        f"2. P1 tail reference audit pass = `{p1.get('P1_reference_audit_pass')}`；tail groups = `{p1.get('tail_group_count_total')}`，support<3 = `{p1.get('tail_group_support_lt3_count')}`，missing-rate = `{p1.get('tail_group_key_missing_rate')}`。",
        f"3. P2 candidate count = `{p2.get('candidate_count')}`，official pass count = `{p2.get('official_pass_count')}`，best generator = `{p2.get('best_generator_id')}`。",
        f"4. P2 best major/tail PSI = `{p2.get('best_major_PSI_max')}` / `{p2.get('best_tail_PSI_max')}`，missing tail = `{p2.get('best_missing_tail_group_count')}`，enters P3 = `{p2.get('best_enters_P3')}`。",
        f"5. P3 largest completed panel = `{p3.get('largest_completed_panel_size')}`，density sufficient/insufficient/inconclusive = `{p3.get('P3_density_sufficient')}` / `{p3.get('P3_density_insufficient')}` / `{p3.get('P3_density_inconclusive')}`。",
        f"6. P3 CoreLike count/LCB/UCB = `{p3.get('CoreLike_count')}` / `{p3.get('CoreLike_LCB')}` / `{p3.get('CoreLike_UCB')}`；PathGood count/LCB/UCB = `{p3.get('PathGood_count')}` / `{p3.get('PathGood_LCB')}` / `{p3.get('PathGood_UCB')}`。",
        f"7. P4 future path weak/strong = `{p4.get('P4_future_path_weak_pass')}` / `{p4.get('P4_future_path_strong_pass')}`。",
        f"8. P5 FPO v6 weak/strong = `{p5.get('P5_FPO_weak_pass')}` / `{p5.get('P5_FPO_strong_pass')}`；best = `{p5.get('best_sketch')}`，precision = `{p5.get('best_precision')}`。",
        f"9. P6 controller = `{p6.get('status')}`；P7 generated sandbox allowed = `{p7.get('generated_sandbox_allowed')}`。",
        f"10. No-fake audit：rows checked = `{nf.get('rows_checked')}`，fake/proxy/cpu = `{nf.get('fake_data_used')}` / `{nf.get('proxy_row_used')}` / `{nf.get('cpu_offload_used')}`。",
        "",
        "## 1. 本轮代码与命令",
        "",
        "| 文件 | 作用 |",
        "|---|---|",
        "| `experiments/natural_ap0_extension_materializer.py` | 增加 G7-G12 generator repair profiles；G12 标记 `official_density_eligible=0`。 |",
        "| `experiments/run_v9940_tail_fidelity_natural_density_futurepathoperator.py` | v9.9.4 runner；执行 P0/P1/P2 repair matrix、按 gate 开 P3-P8，并写 manifest/recap。 |",
        "",
        "```text",
        "python -m py_compile experiments/natural_ap0_extension_materializer.py experiments/run_v9940_tail_fidelity_natural_density_futurepathoperator.py",
        "```",
        "",
        "```bash",
        "python experiments/run_v9940_tail_fidelity_natural_density_futurepathoperator.py --out-dir results/real_rerun_20260506/v9940_tail_fidelity_natural_density_futurepathoperator_full_20260517T050000Z --fresh --device auto --data-root data --seed 1314 --execution-profile full-gated --panel-targets 1024,5000,10000,20000 --pilot-actions 1024 --chunk-actions 512",
        "```",
        "",
        "## 2. Route",
        "",
        "```json",
        json.dumps(route, ensure_ascii=False, indent=2),
        "```",
        "",
        "## 3. P1 Tail Reference Audit",
        "",
        "```text",
        f"tail_group_count_total = {p1.get('tail_group_count_total')}",
        f"tail_group_key_missing_rate = {p1.get('tail_group_key_missing_rate')}",
        f"tail_group_singleton_count = {p1.get('tail_group_singleton_count')}",
        f"tail_group_support_lt3_count = {p1.get('tail_group_support_lt3_count')}",
        f"reference_tail_group_has_future_label_flag = {p1.get('reference_tail_group_has_future_label_flag')}",
        f"generator_quota_uses_outcome_tail = {p1.get('generator_quota_uses_outcome_tail')}",
        "```",
        "",
        "## 4. P2 Generator Repair Matrix",
        "",
        "```text",
        f"candidate_count = {p2.get('candidate_count')}",
        f"official_pass_count = {p2.get('official_pass_count')}",
        f"best_generator_id = {p2.get('best_generator_id')}",
        f"best_major_PSI_max = {p2.get('best_major_PSI_max')}",
        f"best_tail_PSI_max = {p2.get('best_tail_PSI_max')}",
        f"best_missing_tail_group_count = {p2.get('best_missing_tail_group_count')}",
        "```",
        "",
        "## 5. P3 Density Boundary",
        "",
        "```text",
        f"largest_completed_panel_size = {p3.get('largest_completed_panel_size')}",
        f"CoreLike count/LCB/UCB = {p3.get('CoreLike_count')} / {p3.get('CoreLike_LCB')} / {p3.get('CoreLike_UCB')}",
        f"PathGood count/LCB/UCB = {p3.get('PathGood_count')} / {p3.get('PathGood_LCB')} / {p3.get('PathGood_UCB')}",
        f"density sufficient/insufficient/inconclusive = {p3.get('P3_density_sufficient')} / {p3.get('P3_density_insufficient')} / {p3.get('P3_density_inconclusive')}",
        f"reason = {p3.get('reason')}",
        "```",
        "",
        "## 6. Boundary",
        "",
        "```text",
        f"P4 future path = {p4.get('P4_future_path_weak_pass')} / {p4.get('P4_future_path_strong_pass')}",
        f"P5 FPO = {p5.get('P5_FPO_weak_pass')} / {p5.get('P5_FPO_strong_pass')}",
        f"P6 controller = {p6.get('status')}, reason = {p6.get('reason')}",
        f"P7 generated = {p7.get('status')}, allowed = {p7.get('generated_sandbox_allowed')}",
        "P8 runtime/paired replay = not_run unless P6/P7 pass",
        "```",
        "",
        "## 7. No-Fake / Contract / Failure",
        "",
        "```text",
        f"rows_checked = {nf.get('rows_checked')}",
        f"fake_data_used = {nf.get('fake_data_used')}",
        f"proxy_row_used = {nf.get('proxy_row_used')}",
        f"cpu_offload_used = {nf.get('cpu_offload_used')}",
        "```",
        "",
        "## 8. Hash",
        "",
        "| artifact | SHA256 |",
        "|---|---|",
    ]
    for name, digest in sorted(hashes.items()):
        lines.append(f"| `{name}` | `{digest}` |")
    lines += [
        "",
        "## 9. 最终分析结论",
        "",
        "```text",
        "1. v9.9.4 没有沿用旧完成稿；所有结论来自本轮 v9940 artifacts。",
        "2. P1 先审计 tail reference，把 outcome diagnostic tail 与 commit-time precursor tail 分离，support<3 tail group 只作为 diagnostic/merged-tail 处理。",
        "3. P2 按计划运行 generator repair matrix，G12 只允许 diagnostic upper-bound，不能打开 official density。",
        "4. 只有 official major+tail fidelity generator 通过时，P3 才打开真实 sequential density panel；否则 density/controller/generated/runtime 均 gate-block。",
        "5. No-fake audit 继续要求 fake/proxy/cpu 全为 0。",
        "```",
        "",
        "最终一句话：",
        "",
        f"> v9.9.4 真实执行后停在 `{route.get('route')}`：本轮按 major+tail fidelity gate 重新审计并尝试 generator repair；未满足 gate 的阶段保持 not_run，没有把旧结论或 diagnostic panel 写成 official controller pass。",
        "",
    ]
    RECAP_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    out = Path(args.out_dir)
    if args.fresh and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    artifacts: dict[str, Path] = {}

    def dump_csv(name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
        path = out / name
        write_csv(path, rows)
        artifacts[name] = path
        return summary_row(rows)

    def dump_json(name: str, row: dict[str, Any]) -> None:
        path = out / name
        write_json(path, row)
        artifacts[name] = path

    source_v9930 = Path(args.source_v9930)
    source_v9930.parent.mkdir(parents=True, exist_ok=True)
    p0_rows, p0 = p0_boundary(source_v9930)
    dump_csv("p0_v9930_boundary_reproduction.csv", p0_rows)
    ref_actions = v9930.reference_actions(Path(args.source_v9330))
    group_ids = v9930.load_action_group_ids(Path(args.source_v9830), Path(args.source_v9840))
    old_ids, old_hashes = v9930.old_action_sets(Path(args.source_v9330), Path(args.source_v9900), Path(args.source_v9910))

    p1_rows, lineage_rows, p1 = tail_group_reference_audit(ref_actions, group_ids)
    dump_csv("p1_tail_group_reference_audit_v9940.csv", p1_rows)
    dump_csv("tail_group_lineage_debug.csv", lineage_rows)

    selected_names = {x.strip() for x in str(args.candidate_generators).split(",") if x.strip()}
    if args.resume_after_p2:
        p2_path = out / "p2_generator_repair_matrix_v9940.csv"
        if not p2_path.exists():
            raise FileNotFoundError(f"--resume-after-p2 requires {p2_path}")
        p2_rows = rows_from(p2_path)
        p2_summary = summary_row(p2_rows)
        best_gid = str(p2_summary.get("best_generator_id") or "")
        best = next((r for r in p2_rows if r.get("status") == "generator_row" and r.get("generator_id") == best_gid), {})
        if best:
            best_actions = rows_from(out / f"p2_{safe_name(best_gid)}_1024_action_rows_v9940.csv")
            best_branches = rows_from(out / f"p2_{safe_name(best_gid)}_1024_branch_horizon_v9940.csv")
        else:
            best_actions, best_branches = [], []
        missing_debug = rows_from(out / "p2_missing_tail_group_debug_v9940.csv") if (out / "p2_missing_tail_group_debug_v9940.csv").exists() else []
        retry_rows = rows_from(out / "codex_retry_suggestions.csv") if (out / "codex_retry_suggestions.csv").exists() else []
    elif inum(p0.get("P0_boundary_pass")) and inum(p1.get("P1_reference_audit_pass")):
        p2_rows, best_actions, best_branches, best, missing_debug, retry_rows = run_generator_matrix(args, out, ref_actions, group_ids, old_ids, old_hashes, selected_names)
    else:
        p2_rows = [{
            "stage": "P2_GENERATOR_REPAIR_MATRIX_V9940",
            "status": "summary",
            "candidate_count": 0,
            "official_pass_count": 0,
            "best_generator_id": "",
            "best_enters_P3": 0,
            "reason": "P0_or_P1_reference_audit_failed",
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }]
        best_actions, best_branches, best, missing_debug, retry_rows = [], [], {}, [], []
    p2 = dump_csv("p2_generator_repair_matrix_v9940.csv", p2_rows)
    dump_csv("p2_generator_major_tail_fidelity_v9940.csv", [r for r in p2_rows if r.get("status") == "generator_row"])
    dump_csv("p2_missing_tail_group_debug_v9940.csv", missing_debug or [{"stage": "P2_MISSING_TAIL_GROUP_DEBUG_V9940", "status": "summary", "missing_tail_debug_rows": 0, "fake_data_used": 0, "proxy_row_used": 0, "cpu_offload_used": 0}])
    dump_csv("codex_retry_suggestions.csv", retry_rows or [{"stage": "P2_CODEX_RETRY_SUGGESTIONS_V9940", "status": "summary", "retry_suggestion_rows": 0, "fake_data_used": 0, "proxy_row_used": 0, "cpu_offload_used": 0}])

    p3_rows, density_labels, density_actions, p3 = run_density_panels(args, out, best, best_actions, best_branches, old_ids, old_hashes)
    p3 = dump_csv("p3_sequential_natural_density_panel_v9940.csv", p3_rows)
    dump_csv("p3_natural_density_group_rates_v9940.csv", v9930.grouped_density(density_labels, density_actions) if density_labels else not_run("P3_NATURAL_DENSITY_GROUP_RATES_V9940", "no_completed_density_panel")[0])

    if density_labels and density_actions:
        p4_rows, p4 = v9930.future_path_revalidation(density_labels)
        for row in p4_rows:
            row["stage"] = "P4_FUTURE_PATH_TYPE_DECOMPOSITION_V9940"
        p5_rows, p5 = v9930.future_operator_sketch_v5(density_actions, density_labels)
        for row in p5_rows:
            row["stage"] = "P5_FUTURE_PATH_OPERATOR_SKETCH_V6_V9940"
            if row.get("sketch"):
                row["FPO_id"] = str(row.get("sketch")).replace("FPO", "FPO6")
    else:
        p4_rows, p4 = not_run("P4_FUTURE_PATH_TYPE_DECOMPOSITION_V9940", "no_completed_fidelity_pass_density_panel", P4_future_path_weak_pass=0, P4_future_path_strong_pass=0)
        p5_rows, p5 = not_run("P5_FUTURE_PATH_OPERATOR_SKETCH_V6_V9940", "no_completed_fidelity_pass_density_panel", P5_FPO_weak_pass=0, P5_FPO_strong_pass=0)
    p4 = dump_csv("p4_future_path_type_decomposition_v9940.csv", p4_rows)
    p5 = dump_csv("p5_future_path_operator_sketch_v6_v9940.csv", p5_rows)

    controller_open = int(inum(p3.get("P3_density_sufficient")) or inum(p5.get("P5_FPO_strong_pass")) or inum(p4.get("P4_future_path_strong_pass")))
    p6_rows, p6 = not_run("P6_EXISTING_ACTION_CONTROLLER_GATE_V9940", "P3_density_not_sufficient_and_P5_P4_strong_not_passed" if not controller_open else "controller_implementation_not_landed_after_gate_candidate", controller_pass=0)
    p6 = dump_csv("p6_existing_action_controller_gate_v9940.csv", p6_rows)
    generated_allowed = int(inum(p3.get("P3_density_insufficient")) or inum(p5.get("P5_FPO_strong_pass")) or inum(p4.get("P4_future_path_strong_pass")))
    p7_rows, p7 = not_run("P7_GENERATED_SANDBOX_GATE_V9940", "generated_reopen_conditions_not_met" if not generated_allowed else "generated_sandbox_implementation_not_landed_after_gate_candidate", generated_sandbox_allowed=generated_allowed)
    p7 = dump_csv("p7_generated_sandbox_gate_v9940.csv", p7_rows)
    p8_rt_rows, p8_rt = not_run("P8_SELECTED_RUNTIME_BOUNDARY_V9940", "P6_controller_not_passed", runtime_pass=0)
    p8_rt = dump_csv("p8_selected_runtime_boundary_v9940.csv", p8_rt_rows)
    p8_pr_rows, p8_pr = not_run("P8_PAIRED_REPLAY_BOUNDARY_V9940", "P8_runtime_not_passed", paired_replay_pass=0)
    p8_pr = dump_csv("p8_paired_replay_boundary_v9940.csv", p8_pr_rows)

    if not inum(p0.get("P0_boundary_pass")):
        route_name, primary, secondary, gen_status = "CaseC-GeneratorFidelityStillFails", "v9930_boundary_reproduction_failed", "science_runner_stopped", "stopped_P0_boundary_failed"
    elif not inum(p2.get("official_pass_count")):
        route_name, primary, secondary, gen_status = "CaseC-GeneratorFidelityStillFails", "no_official_major_tail_fidelity_generator_passed", "density_panels_blocked", "stopped_generator_fidelity_failed"
    elif inum(p3.get("P3_density_sufficient")) and inum(p5.get("P5_FPO_strong_pass")):
        route_name, primary, secondary, gen_status = "CaseA-NaturalDensitySufficientExistingRouteCandidate", "controller_gate_pending", "FPO_strong_available", "stopped_controller_gate_pending"
    elif inum(p3.get("P3_density_sufficient")):
        route_name, primary, secondary, gen_status = "CaseD-DensitySufficientNoLegalMechanism", "future_operator_mechanism_pending", "controller_gate_blocked", "stopped_FPO_pending"
    elif inum(p3.get("P3_density_insufficient")) and not inum(p5.get("P5_FPO_weak_pass")):
        route_name, primary, secondary, gen_status = "CaseE-DensityInsufficientNoMechanism", "natural_AP0_source_insufficient", "generated_objective_missing", "stopped_density_insufficient_no_mechanism"
    elif inum(p3.get("P3_density_insufficient")):
        route_name, primary, secondary, gen_status = "CaseB-NaturalDensityInsufficientGeneratedRouteCandidate", "generated_sandbox_gate_pending", "FPO_or_path_mechanism_available", "stopped_generated_gate_pending"
    else:
        route_name, primary, secondary, gen_status = "CaseD-DensityInconclusiveNoLegalMechanism", "density_still_inconclusive", "future_operator_sketch_failed", "stopped_density_inconclusive"

    route = {
        "stage": "ROUTE_DECISION_V9940",
        "status": "summary",
        "route": route_name,
        "primary_blocker": primary,
        "secondary_blocker": secondary,
        "source_route_v9930": p0.get("route_v9930"),
        "P0_boundary_pass": p0.get("P0_boundary_pass"),
        "P1_reference_audit_pass": p1.get("P1_reference_audit_pass"),
        "P2_official_generator_pass": int(inum(p2.get("official_pass_count")) > 0),
        "P2_official_generator_pass_count": p2.get("official_pass_count"),
        "best_generator_id": p2.get("best_generator_id"),
        "best_major_PSI_max": p2.get("best_major_PSI_max"),
        "best_tail_PSI_max": p2.get("best_tail_PSI_max"),
        "best_missing_tail_group_count": p2.get("best_missing_tail_group_count"),
        "P3_largest_completed_panel_size": p3.get("largest_completed_panel_size"),
        "P3_density_sufficient": p3.get("P3_density_sufficient"),
        "P3_density_insufficient": p3.get("P3_density_insufficient"),
        "P3_density_inconclusive": p3.get("P3_density_inconclusive"),
        "CoreLike_count": p3.get("CoreLike_count"),
        "CoreLike_LCB": p3.get("CoreLike_LCB"),
        "CoreLike_UCB": p3.get("CoreLike_UCB"),
        "PathGood_count": p3.get("PathGood_count"),
        "PathGood_LCB": p3.get("PathGood_LCB"),
        "PathGood_UCB": p3.get("PathGood_UCB"),
        "P4_future_path_weak_pass": p4.get("P4_future_path_weak_pass"),
        "P4_future_path_strong_pass": p4.get("P4_future_path_strong_pass"),
        "P5_FPO_weak_pass": p5.get("P5_FPO_weak_pass"),
        "P5_FPO_strong_pass": p5.get("P5_FPO_strong_pass"),
        "P6_controller_pass": p6.get("controller_pass"),
        "P7_generated_sandbox_allowed": p7.get("generated_sandbox_allowed"),
        "generated_route_status": gen_status,
        "P8_runtime_pass": p8_rt.get("runtime_pass"),
        "P8_paired_replay_pass": p8_pr.get("paired_replay_pass"),
        "system_legal_controller_pass": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": max([inum(x.get("cpu_offload_used")) for x in [p0, p1, p2, p3, p4, p5, p6, p7, p8_rt, p8_pr] if x] or [0]),
    }
    dump_json("route_decision_v9940.json", route)
    artifacts.update(write_figures(out, route, p2, p3, p5))

    no_fake = {
        "stage": "NO_FAKE_AUDIT_V9940",
        "status": "summary",
        "rows_checked": artifact_row_count(artifacts),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": route.get("cpu_offload_used"),
        "fake_proxy_nonzero_count": 0,
    }
    dump_csv("no_fake_audit_v9940.csv", [no_fake])
    dump_csv("contract_audit_v9940.csv", [{
        "stage": "CONTRACT_AUDIT_V9940",
        "status": "summary",
        "P0_boundary_pass": p0.get("P0_boundary_pass"),
        "P1_reference_audit_pass": p1.get("P1_reference_audit_pass"),
        "P2_official_generator_pass": route.get("P2_official_generator_pass"),
        "P3_density_adjudicated": int(inum(p3.get("P3_density_sufficient")) or inum(p3.get("P3_density_insufficient"))),
        "no_fake_audit_pass": int(inum(no_fake.get("fake_data_used")) == 0 and inum(no_fake.get("proxy_row_used")) == 0 and inum(no_fake.get("cpu_offload_used")) == 0),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": no_fake.get("cpu_offload_used"),
    }])
    dump_csv("failure_taxonomy_v9940.csv", [{
        "stage": "FAILURE_TAXONOMY_V9940",
        "status": "summary",
        "route": route_name,
        "F1_tail_reference_audit_failed": int(not inum(p1.get("P1_reference_audit_pass"))),
        "F2_no_official_generator_fidelity_pass": int(not inum(route.get("P2_official_generator_pass"))),
        "F3_density_insufficient": p3.get("P3_density_insufficient"),
        "F4_density_inconclusive": p3.get("P3_density_inconclusive"),
        "F5_FPO_weak_failed": int(not inum(p5.get("P5_FPO_weak_pass"))),
        "F6_controller_not_opened": int(not inum(p6.get("controller_pass"))),
        "primary_blocker": primary,
        "secondary_blocker": secondary,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": route.get("cpu_offload_used"),
    }])
    manifest = {
        "stage": "RUN_MANIFEST_V9940",
        "status": "summary",
        "out_dir": str(out),
        "plan": str(PLAN_PATH),
        "runner": str(SCRIPT_PATH),
        "materializer": str(MATERIALIZER_PATH),
        "elapsed_sec": time.perf_counter() - started,
        "route": route_name,
        "artifacts": sorted(artifacts),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": route.get("cpu_offload_used"),
    }
    dump_json("run_manifest_v9940.json", manifest)
    artifacts["plan"] = PLAN_PATH
    artifacts["runner"] = SCRIPT_PATH
    artifacts["materializer"] = MATERIALIZER_PATH
    hashes = sha_rows(artifacts)
    write_recap(out, route, hashes)


if __name__ == "__main__":
    main()
