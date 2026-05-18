#!/usr/bin/env python3
"""DG-KAN v9.8.9 natural stream / FuturePathOperator four-line run.

This runner follows the v9.8.9 fail-fast contract.  If the repo still lacks a
real natural AP0 extension action generator, it stops the science runner and
writes engineering-only blocker artifacts instead of fabricating natural stream
rows or replay labels.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "experiments"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

import run_v9720_exact_transfer_materializer_core_expansion_direct_update as v9720  # noqa: E402
from dgkan_outcome_controller import fnum, inum, read_csv, read_json, sha256_file, write_csv, write_json  # noqa: E402


RESULT_ROOT = REPO / "results/real_rerun_20260506"
PLAN_PATH = REPO / "docs/DG-KAN_v9.8.9_自然扩流_FuturePathOperator_四线并行_完整实验计划.md"
SCRIPT_PATH = REPO / "experiments/run_v9890_natural_stream_futurepathoperator_four_line.py"
RECAP_PATH = REPO / "docs/DG-KAN_v9.8.9_NaturalStream_FuturePathOperator_FourLine_实验复盘.md"
DEFAULT_OUT = RESULT_ROOT / "v9890_natural_stream_futurepathoperator_four_line_full_20260517T000000Z"
DEFAULT_V9880 = RESULT_ROOT / "v9880_four_line_natural_futurepathoperator_lowcost_full_20260516T230000Z"
HORIZONS = [1, 5, 20, 80, 240]
BRANCH_COUNT = 6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--seed", type=int, default=1314)
    parser.add_argument("--execution-profile", choices=["smoke", "full-gated"], default="full-gated")
    parser.add_argument("--panel-targets", default="2876,5000,10000,20000")
    parser.add_argument("--source-v9880", default=str(DEFAULT_V9880))
    return parser.parse_args()


def parse_ints(text: str) -> list[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def summary_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return next((r for r in rows if r.get("status") == "summary"), rows[0] if rows else {})


def rows_from(path: Path) -> list[dict[str, Any]]:
    return read_csv(path) if path.exists() else []


def json_counter(items: list[str]) -> str:
    return json.dumps(dict(Counter(items)), ensure_ascii=False, sort_keys=True)


def p0_boundary(source_v9880: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    route = read_json(source_v9880 / "route_decision_v9880.json")
    p1 = summary_row(rows_from(source_v9880 / "p1_natural_extension_generator_entrypoint_v9880.csv"))
    p3 = summary_row(rows_from(source_v9880 / "p3_natural_density_panel_v9880.csv"))
    p4 = summary_row(rows_from(source_v9880 / "p4_future_path_type_mechanism_v9880.csv"))
    p5 = summary_row(rows_from(source_v9880 / "p5_low_cost_future_operator_proxy_v2_v9880.csv"))
    p7 = summary_row(rows_from(source_v9880 / "p7_generated_reopen_gate_v9880.csv"))
    nf = summary_row(rows_from(source_v9880 / "no_fake_audit_v9880.csv"))
    row = {
        "stage": "P0_BOUNDARY_REPRODUCTION_V9890",
        "status": "summary",
        "source_route_v9880": route.get("route"),
        "source_primary_blocker_v9880": route.get("primary_blocker"),
        "natural_extension_action_generator_found_v9880": p1.get("natural_extension_action_generator_found"),
        "entrypoint_count_v9880": p1.get("entrypoint_count"),
        "P3_density_inconclusive_v9880": p3.get("P3_density_inconclusive"),
        "P4_future_path_weak_pass_v9880": p4.get("P4_weak_pass"),
        "P5_low_cost_proxy_weak_pass_v9880": p5.get("P5_weak_pass"),
        "system_legal_controller_pass_v9880": route.get("system_legal_controller_pass"),
        "generated_sandbox_allowed_v9880": p7.get("generated_sandbox_allowed"),
        "fake_data_used_v9880": nf.get("fake_data_used"),
        "proxy_row_used_v9880": nf.get("proxy_row_used"),
        "cpu_offload_used_v9880": nf.get("cpu_offload_used"),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    row["P0_pass"] = int(
        row["source_route_v9880"] == "R7-EngineeringBlocked"
        and inum(row["natural_extension_action_generator_found_v9880"]) == 0
        and inum(row["P4_future_path_weak_pass_v9880"]) == 1
        and inum(row["P5_low_cost_proxy_weak_pass_v9880"]) == 0
        and inum(row["system_legal_controller_pass_v9880"]) == 0
        and inum(row["generated_sandbox_allowed_v9880"]) == 0
        and inum(row["fake_data_used_v9880"]) == 0
        and inum(row["proxy_row_used_v9880"]) == 0
        and inum(row["cpu_offload_used_v9880"]) == 0
    )
    return [row], row


def p1_entrypoint_audit() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    valid_patterns = [
        "natural_AP0_action_extension_generator",
        "natural_ap0_action_extension_generator",
        "generate_natural_ap0_extension_actions",
        "materialize_natural_ap0_action_stream",
        "natural_stream_action_generator",
    ]
    rows: list[dict[str, Any]] = []
    legacy_hits: list[str] = []
    for path in sorted((REPO / "experiments").glob("run_*.py")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        for pattern in valid_patterns:
            if f"def {pattern}" in text or f"class {pattern}" in text:
                rows.append({
                    "stage": "P1_NATURAL_GENERATOR_ENTRYPOINT_AUDIT_V9890",
                    "status": "entrypoint_candidate",
                    "generator_module": str(path.relative_to(REPO)),
                    "generator_function": pattern,
                    "required_inputs": "",
                    "required_outputs": "",
                    "missing_dependency_list": "",
                    "valid_for_natural_AP0_extension": 1,
                    "fake_data_used": 0,
                    "proxy_row_used": 0,
                    "cpu_offload_used": 0,
                })
        if "source_generator" in text or "generated_action" in text or "APG" in text:
            legacy_hits.append(path.name)

    missing = [
        "candidate generation function",
        "train stream event source",
        "action payload builder",
        "action_id allocator",
        "candidate_id allocator",
        "payload hash writer",
        "action apply replay hook",
        "branch-horizon materializer hook",
    ]
    summary = {
        "stage": "P1_NATURAL_GENERATOR_ENTRYPOINT_AUDIT_V9890",
        "status": "summary",
        "entrypoint_count": len(rows),
        "natural_extension_action_generator_found": int(bool(rows)),
        "generator_module": rows[0].get("generator_module") if rows else "",
        "generator_function": rows[0].get("generator_function") if rows else "",
        "required_inputs": "train stream event source; action payload builder; action_id allocator; candidate_id allocator",
        "required_outputs": "new action rows; payload hashes; action apply replay rows; branch-horizon labels",
        "missing_dependency_list": "; ".join([] if rows else missing),
        "legacy_generated_or_source_generator_file_count": len(set(legacy_hits)),
        "legacy_hits_are_not_natural_AP0_extension": 1,
        "P1a_entrypoint_search_pass": int(bool(rows)),
        "P1_weak_pass": 0,
        "P1_strong_pass": 0,
        "reason": "" if rows else "no_landed_natural_AP0_extension_action_generator_found",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    rows.insert(0, summary)
    if not summary["natural_extension_action_generator_found"]:
        rows.extend([
            {
                "stage": "P1_NATURAL_GENERATOR_ENTRYPOINT_AUDIT_V9890",
                "status": "implementation_contract",
                "generator_module": "experiments/natural_ap0_extension_materializer.py",
                "generator_function": "generate_natural_ap0_extension_actions",
                "required_inputs": "seed; data_root; train_stream_event_cursor; target_action_count; AP0 payload schema",
                "required_outputs": "new_action_id; candidate_id; dataset/seed/step_bucket; payload_hash; payload_json; lifecycle provenance",
                "missing_dependency_list": "module_not_landed",
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            },
            {
                "stage": "P1_NATURAL_GENERATOR_ENTRYPOINT_AUDIT_V9890",
                "status": "single_action_smoke_spec",
                "generator_module": "experiments/natural_ap0_extension_materializer.py",
                "generator_function": "generate_natural_ap0_extension_actions",
                "required_inputs": "target_action_count=1",
                "required_outputs": "unique action_id=1; unique payload_hash=1; action_apply_linf_max<=1e-7; h1/h5/h20/h80/h240 branch-horizon rows if available",
                "missing_dependency_list": "blocked_until_generator_lands",
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            },
        ])
    return rows, summary


def p1_preflight_rows(stage: str, action_count: int, reason: str) -> list[dict[str, Any]]:
    return [{
        "stage": stage,
        "status": "not_run",
        "natural_action_count": 0,
        "new_action_count": 0,
        "reused_action_count": 0,
        "duplicate_action_id_count": "",
        "duplicate_payload_hash_count": "",
        "payload_hash_missing_count": "",
        "action_apply_linf_max": "",
        "action_apply_cosine_min": "",
        "branch_horizon_expected_rows": action_count * BRANCH_COUNT * len(HORIZONS),
        "branch_horizon_actual_rows": 0,
        "branch_horizon_completion": 0,
        "rows_per_sec": 0,
        "wallclock_sec": 0,
        "peak_gpu_memory_mb": 0,
        "unresolved_exception_count": "",
        "label_exclusivity_violation_count": "",
        "reason": reason,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }]


def p2_density_panel(source_v9880: Path, panel_targets: list[int]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prev = rows_from(source_v9880 / "p3_natural_density_panel_v9880.csv")
    old_summary = summary_row(prev)
    existing = next((r for r in prev if r.get("status") == "panel_row"), {})
    diag = next((r for r in prev if r.get("status") == "diagnostic_preflight_row"), {})
    rows: list[dict[str, Any]] = []
    if existing:
        rows.append({
            "stage": "P2_NATURAL_DENSITY_PANEL_V9890",
            "status": "existing_reference_panel",
            "panel_name": "PanelA-existing-2876",
            "action_count": existing.get("action_count") or existing.get("panel_size"),
            "new_action_count": 0,
            "CoreLike_count": existing.get("CoreLike_count"),
            "CoreLike_rate": existing.get("CoreLike_rate"),
            "CoreLike_LCB": existing.get("CoreLike_LCB"),
            "CoreLike_UCB": existing.get("CoreLike_UCB"),
            "PathGood_count": "",
            "PathGood_rate": "",
            "PathGood_LCB": "",
            "PathGood_UCB": "",
            "SlowBurnGood_count": "",
            "SlowBurnGood_rate": "",
            "SlowBurnGood_LCB": "",
            "SlowBurnGood_UCB": "",
            "RiskyHighAUV_count": "",
            "SafeLowValue_count": "",
            "GradeAB_count": "",
            "ValuePositiveNoLongRisk_count": "",
            "density_result": "existing_reference_inconclusive",
            "reason": "reference_only_not_new_extension",
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    if diag:
        rows.append({
            "stage": "P2_NATURAL_DENSITY_PANEL_V9890",
            "status": "diagnostic_preflight_reference",
            "panel_name": "v9870-diagnostic-256-existing-action-preflight",
            "action_count": diag.get("action_count") or diag.get("panel_size"),
            "new_action_count": 0,
            "CoreLike_count": diag.get("CoreLike_count"),
            "CoreLike_rate": diag.get("CoreLike_rate"),
            "CoreLike_LCB": diag.get("CoreLike_LCB"),
            "CoreLike_UCB": diag.get("CoreLike_UCB"),
            "PathGood_count": diag.get("PathGood_count"),
            "PathGood_rate": diag.get("PathGood_rate"),
            "PathGood_LCB": diag.get("PathGood_LCB"),
            "PathGood_UCB": diag.get("PathGood_UCB"),
            "SlowBurnGood_count": diag.get("SlowBurnGood_count"),
            "SlowBurnGood_rate": diag.get("SlowBurnGood_rate"),
            "density_result": "diagnostic_reference_not_official_extension_panel",
            "reason": "reference_only_not_new_extension",
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    for target in [t for t in panel_targets if t > 2876]:
        rows.append({
            "stage": "P2_NATURAL_DENSITY_PANEL_V9890",
            "status": "not_run",
            "panel_name": f"Panel-natural-{target}",
            "action_count": 0,
            "new_action_count": 0,
            "branch_horizon_expected_rows": target * BRANCH_COUNT * len(HORIZONS),
            "branch_horizon_actual_rows": 0,
            "density_result": "not_run_generator_missing",
            "reason": "P1a_natural_extension_generator_missing_fail_fast",
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    summary = {
        "stage": "P2_NATURAL_DENSITY_PANEL_V9890",
        "status": "summary",
        "completed_natural_extension_panel_count": 0,
        "existing_reference_panel_count": sum(1 for r in rows if r.get("status") == "existing_reference_panel"),
        "not_run_panel_count": sum(1 for r in rows if r.get("status") == "not_run"),
        "PanelA_CoreLike_LCB": old_summary.get("PanelA_CoreLike_LCB"),
        "PanelA_CoreLike_UCB": old_summary.get("PanelA_CoreLike_UCB"),
        "diagnostic_PathGood_LCB": old_summary.get("diagnostic_PathGood_LCB"),
        "diagnostic_PathGood_UCB": old_summary.get("diagnostic_PathGood_UCB"),
        "P2_density_sufficient": 0,
        "P2_density_insufficient": 0,
        "P2_density_inconclusive": 1,
        "reason": "natural_extension_generator_missing_no_new_density_claim",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    rows.insert(0, summary)
    return rows, summary


def p2_density_by_dataset_family_template(source_v9880: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prev = rows_from(source_v9880 / "p3_natural_density_panel_v9880.csv")
    existing = next((r for r in prev if r.get("status") == "panel_row"), {})
    rows: list[dict[str, Any]] = []
    for field, entity_type in [("per_dataset_rate", "dataset"), ("per_family_rate", "family"), ("per_template_rate", "template")]:
        raw = existing.get(field) or "{}"
        try:
            rates = json.loads(raw)
        except Exception:
            rates = {}
        for name, rate in rates.items():
            rows.append({
                "stage": "P2_DENSITY_BY_DATASET_FAMILY_TEMPLATE_V9890",
                "status": "existing_reference_rate",
                "entity_type": entity_type,
                "entity_id": name,
                "CoreLike_rate": rate,
                "source_panel": "PanelA-existing-2876",
                "density_claim_scope": "reference_only_not_natural_extension",
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            })
    summary = {
        "stage": "P2_DENSITY_BY_DATASET_FAMILY_TEMPLATE_V9890",
        "status": "summary",
        "reference_rate_rows": len(rows),
        "new_extension_rate_rows": 0,
        "min_dataset_rate": min([fnum(r.get("CoreLike_rate")) for r in rows if r.get("entity_type") == "dataset"], default=0),
        "min_family_rate": min([fnum(r.get("CoreLike_rate")) for r in rows if r.get("entity_type") == "family"], default=0),
        "min_template_rate": min([fnum(r.get("CoreLike_rate")) for r in rows if r.get("entity_type") == "template"], default=0),
        "reason": "reference_only_P1a_generator_missing",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    rows.insert(0, summary)
    return rows, summary


def not_run_artifact(stage: str, reason: str, **extra: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    row = {
        "stage": stage,
        "status": "not_run",
        "reason": reason,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
        **extra,
    }
    return [row], row


def engineering_task_list() -> list[dict[str, Any]]:
    tasks = [
        ("T1", "land_generator_module", "Create experiments/natural_ap0_extension_materializer.py with generate_natural_ap0_extension_actions(seed,data_root,target_action_count,cursor)."),
        ("T2", "bind_action_lifecycle", "Use the existing AP0 action schema and write action_id/candidate_id/payload_hash/provenance without reusing old action rows."),
        ("T3", "bind_action_apply", "Expose action apply replay hook and verify action_apply_linf_max <= 1e-7 on a single new action."),
        ("T4", "bind_branch_horizon", "Materialize RealFunctional/AdamWParallel/bestLR/NoOp/Random/Shuffled branch-horizon rows for h1,h5,h20,h80,h240."),
        ("T5", "single_action_smoke", "Run target_action_count=1 and verify unique action_id, payload_hash, no-transform sanity, no fake/proxy/cpu."),
        ("T6", "sixteen_action_smoke", "Run target_action_count=16 and verify completion=1.0, duplicate counts=0, label exclusivity violations=0."),
        ("T7", "two_fifty_six_smoke", "Run target_action_count=256 and record rows/sec, wallclock, peak GPU memory, NaN/Inf, exception count."),
    ]
    return [{
        "stage": "ENGINEERING_MATERIALIZER_TASK_LIST_V9890",
        "status": "task",
        "task_id": tid,
        "task_name": name,
        "implementation_contract": contract,
        "required_before_science_full_run": 1,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    } for tid, name, contract in tasks]


def field_legality() -> list[dict[str, Any]]:
    rows = [
        ("green", "future-operator sketch fields computed at training time; memory/hard-tail/current-batch response; cost fields", "allowed only after generator/materializer lands"),
        ("yellow", "existing-reference density and v9.8.8 future-path diagnostics", "diagnostic/reference only"),
        ("red", "future outcome labels; CoreLike/PathGood labels; old table labels; dataset routing", "not allowed in official controller"),
    ]
    return [{
        "stage": "FIELD_LEGALITY_LEDGER_V9890",
        "status": "field_legality_row",
        "field_color": color,
        "fields": fields,
        "reason": reason,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    } for color, fields, reason in rows]


def write_figures(out: Path, route: dict[str, Any], p2: dict[str, Any]) -> dict[str, Path]:
    figs: dict[str, Path] = {}
    def fig(name: str, title: str, labels: list[str], values: list[float]) -> None:
        path = out / name
        v9720.write_bar_svg(path, title, labels, values)
        figs[name] = path

    fig("fig_v9890_four_line_gate_matrix.svg", "four-line gate matrix", ["P1", "P2", "P3", "P4", "P5", "P6"], [0, 0, 0, 0, 0, 0])
    fig("fig_p1_generator_lifecycle_flow.svg", "generator lifecycle", ["entry", "single", "16", "256", "5000"], [0, 0, 0, 0, 0])
    fig("fig_p1_branch_horizon_completion.svg", "branch horizon completion", ["single", "16", "256"], [0, 0, 0])
    fig("fig_p2_corelike_density_curve.svg", "CoreLike density LCB", ["2876", "5000", "10000", "20000"], [fnum(p2.get("PanelA_CoreLike_LCB")), 0, 0, 0])
    fig("fig_p2_pathgood_density_curve.svg", "PathGood density LCB", ["diag256", "5000", "10000", "20000"], [fnum(p2.get("diagnostic_PathGood_LCB")), 0, 0, 0])
    fig("fig_p2_density_CI_vs_panel_size.svg", "density CI", ["LCB", "UCB"], [fnum(p2.get("PanelA_CoreLike_LCB")), fnum(p2.get("PanelA_CoreLike_UCB"))])
    fig("fig_p2_dataset_family_density_heatmap.svg", "density reference rows", ["existing", "new"], [1, 0])
    fig("fig_p3_future_path_V_curves.svg", "P3 not run", ["not_run"], [0])
    fig("fig_p3_future_path_risk_curves.svg", "P3 risk not run", ["not_run"], [0])
    fig("fig_p3_delayed_gain_vs_longrisk.svg", "P3 delayed not run", ["not_run"], [0])
    fig("fig_p3_AUV_vs_RiskAdjustedAUV.svg", "P3 AUV not run", ["not_run"], [0])
    fig("fig_p4_proxy_precision_cost_pareto.svg", "P4 proxy not run", ["not_run"], [0])
    fig("fig_p4_proxy_score_vs_RAUV.svg", "P4 score not run", ["not_run"], [0])
    fig("fig_p4_proxy_failure_cases.svg", "P4 failure", ["P1a missing"], [1])
    fig("fig_p5_controller_accepted_region_pareto.svg", "controller not run", ["not_run"], [0])
    fig("fig_p6_generated_sandbox_gate.svg", "generated gate", ["allowed"], [fnum(route.get("P6_generated_sandbox_allowed"))])
    fig("fig_p7_runtime_component_stack.svg", "runtime not run", ["not_run"], [0])
    fig("fig_stop_pivot_matrix.svg", "stop pivot", ["C1 missing", "engineering task"], [1, 1])
    return figs


def artifact_row_count(artifacts: dict[str, Path]) -> int:
    total = 0
    for name, path in artifacts.items():
        if name.endswith(".csv") and path.exists() and not name.startswith(("no_fake", "contract", "failure")):
            with path.open(newline="", encoding="utf-8") as f:
                total += sum(1 for _ in csv.DictReader(f))
        elif name.endswith(".json") and path.exists():
            total += 1
    return total


def sha_rows(paths: dict[str, Path]) -> dict[str, str]:
    return {name: sha256_file(path) for name, path in paths.items() if path.exists()}


def write_recap(out: Path, route: dict[str, Any], hashes: dict[str, str]) -> None:
    p0 = summary_row(rows_from(out / "p0_boundary_reproduction_v9890.csv"))
    p1 = summary_row(rows_from(out / "p1_natural_generator_entrypoint_audit_v9890.csv"))
    p2 = summary_row(rows_from(out / "p2_natural_density_panel_v9890.csv"))
    p3 = summary_row(rows_from(out / "p3_future_path_type_decomposition_v9890.csv"))
    p4 = summary_row(rows_from(out / "p4_future_operator_sketch_v9890.csv"))
    p5 = summary_row(rows_from(out / "p5_controller_gate_v9890.csv"))
    p6 = summary_row(rows_from(out / "p6_generated_sandbox_gate_v9890.csv"))
    nf = summary_row(rows_from(out / "no_fake_audit_v9890.csv"))
    tasks = rows_from(out / "engineering_materializer_task_list_v9890.csv")
    lines = [
        "# DG-KAN v9.8.9 Natural Stream / FuturePathOperator / Four-Line 实验复盘",
        "",
        "> 本复盘记录 `DG-KAN_v9.8.9_自然扩流_FuturePathOperator_四线并行_完整实验计划.md` 的真实执行结果。所有结论只来自落盘 CSV/JSON/manifest 或 v9.8.8 reference artifacts；P1a 找不到真实 natural AP0 extension generator 时，后续 science stages 均显式 `not_run`，没有 fake data、proxy rows 或 CPU offload。",
        "",
        "## 0. 最新结论",
        "",
        "```text",
        f"route = {route.get('route')}",
        f"primary_blocker = {route.get('primary_blocker')}",
        f"secondary_blocker = {route.get('secondary_blocker')}",
        f"route_recommendation = {route.get('route_recommendation')}",
        f"system_legal_controller_pass = {route.get('system_legal_controller_pass')}",
        f"generated_route_status = {route.get('generated_route_status')}",
        "```",
        "",
        f"最终 artifact：`{out}`",
        "",
        "核心结论：",
        "",
        f"1. P0 复现 v9.8.8 boundary：source route = `{p0.get('source_route_v9880')}`，P4 weak = `{p0.get('P4_future_path_weak_pass_v9880')}`，P5 weak = `{p0.get('P5_low_cost_proxy_weak_pass_v9880')}`，system = `{p0.get('system_legal_controller_pass_v9880')}`。",
        f"2. P1a natural extension generator entrypoint 仍未落地：entrypoint_count = `{p1.get('entrypoint_count')}`，found = `{p1.get('natural_extension_action_generator_found')}`。",
        f"3. P1b/P1c/P1d single/16/256 preflight 全部 `not_run`，原因 = `P1a_natural_extension_generator_missing_fail_fast`。",
        f"4. P2 只写 existing reference：PanelA CoreLike LCB/UCB = `{p2.get('PanelA_CoreLike_LCB')}` / `{p2.get('PanelA_CoreLike_UCB')}`；5000/10000/20000 natural panels `not_run`。",
        f"5. P3 future path decomposition = `{p3.get('status')}`；P4 FuturePathOperator Sketch = `{p4.get('status')}`，均因 P1a fail-fast 停止。",
        f"6. P5 controller = `{p5.get('status')}`；P6 generated sandbox allowed = `{p6.get('P6_generated_sandbox_allowed')}`。",
        f"7. Engineering materializer task rows = `{len(tasks)}`。",
        f"8. No-fake audit：rows checked = `{nf.get('rows_checked')}`，fake/proxy/cpu = `{nf.get('fake_data_used')}` / `{nf.get('proxy_row_used')}` / `{nf.get('cpu_offload_used')}`。",
        "",
        "## 1. 本轮代码与命令",
        "",
        "| 文件 | 作用 |",
        "|---|---|",
        "| `experiments/run_v9890_natural_stream_futurepathoperator_four_line.py` | v9.8.9 runner；复现 v9.8.8 boundary，执行 P1a generator audit；若缺 generator，按 fail-fast 写 engineering-only blocker artifacts。 |",
        "",
        "```text",
        "python -m py_compile experiments/run_v9890_natural_stream_futurepathoperator_four_line.py",
        "```",
        "",
        "```bash",
        "python experiments/run_v9890_natural_stream_futurepathoperator_four_line.py --out-dir results/real_rerun_20260506/v9890_natural_stream_futurepathoperator_four_line_full_20260517T000000Z --fresh --device auto --data-root data --seed 1314 --execution-profile full-gated --panel-targets 2876,5000,10000,20000",
        "```",
        "",
        "## 2. Route",
        "",
        "```json",
        json.dumps(route, ensure_ascii=False, indent=2),
        "```",
        "",
        "## 3. P1 Engineering Contract",
        "",
        "| task | contract |",
        "|---|---|",
    ]
    for task in tasks:
        lines.append(f"| `{task.get('task_name')}` | {task.get('implementation_contract')} |")
    lines += [
        "",
        "判断：v9.8.9 没有继续跑完整 science runner。P1a 缺 generator 后，下一步必须是工程化 natural AP0 extension materializer，而不是继续复用旧 AP0 replay 做 density/controller claim。",
        "",
        "## 4. P2 Density Boundary",
        "",
        "```text",
        f"existing PanelA CoreLike LCB/UCB = {p2.get('PanelA_CoreLike_LCB')} / {p2.get('PanelA_CoreLike_UCB')}",
        f"diagnostic PathGood LCB/UCB = {p2.get('diagnostic_PathGood_LCB')} / {p2.get('diagnostic_PathGood_UCB')}",
        "PanelB/C/D natural extension = not_run",
        "density sufficient/insufficient/inconclusive = 0/0/1",
        "```",
        "",
        "判断：这些是 reference rows，不是 v9.8.9 新自然扩流结果；不能声称 density sufficient 或 insufficient。",
        "",
        "## 5. Boundary",
        "",
        "```text",
        "P3 future path type decomposition = not_run",
        "P4 FuturePathOperator Sketch v3 = not_run",
        "P5 controller = not_run",
        "P6 generated sandbox = not_run",
        "P7 runtime = not_run",
        "P8 paired replay = not_run",
        "```",
        "",
        "## 6. No-Fake / Contract / Failure",
        "",
        "```text",
        f"rows_checked = {nf.get('rows_checked')}",
        f"fake_data_used = {nf.get('fake_data_used')}",
        f"proxy_row_used = {nf.get('proxy_row_used')}",
        f"cpu_offload_used = {nf.get('cpu_offload_used')}",
        "```",
        "",
        "## 7. Hash",
        "",
        "| artifact | SHA256 |",
        "|---|---|",
    ]
    for name, digest in sorted(hashes.items()):
        lines.append(f"| `{name}` | `{digest}` |")
    lines += [
        "",
        "## 8. 最终分析结论",
        "",
        "```text",
        "1. v9.8.9 按计划执行了 P0/P1a，并确认 v9.8.8 的工程 blocker 仍存在。",
        "2. P1a 仍找不到真实 natural AP0 extension action generator，因此按 fail-fast 停止 science full run。",
        "3. P1b/P1c/P1d/P2B-D/P3/P4/P5/P6/P7/P8 全部显式 not_run，没有 fake/proxy rows。",
        "4. 本轮有效推进是把下一步收敛为 engineering-only materializer task list 和 single-action smoke spec。",
        "5. strict PureKAN functional 仍未成功，existing-action route 不能继续声称 density 或 controller closure。",
        "```",
        "",
        "最终一句话：",
        "",
        f"> v9.8.9 真实执行后停在 `{route.get('route')}`：自然 AP0 extension generator 仍未落地，因此本轮按计划停止完整 science runner，只留下 materializer 工程合同和 smoke 测试规格；没有进入 controller、generated sandbox、runtime 或 paired replay。",
        "",
    ]
    RECAP_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    out = Path(args.out_dir)
    if args.fresh and out.exists():
        import shutil

        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    panel_targets = parse_ints(args.panel_targets)
    source_v9880 = Path(args.source_v9880)
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

    p0 = dump_csv("p0_boundary_reproduction_v9890.csv", p0_boundary(source_v9880)[0])
    p1_rows, p1 = p1_entrypoint_audit()
    dump_csv("p1_natural_generator_entrypoint_audit_v9890.csv", p1_rows)
    reason = "P1a_natural_extension_generator_missing_fail_fast"
    dump_csv("p1_single_action_preflight_v9890.csv", p1_preflight_rows("P1_SINGLE_ACTION_PREFLIGHT_V9890", 1, reason))
    dump_csv("p1_16_action_preflight_v9890.csv", p1_preflight_rows("P1_16_ACTION_PREFLIGHT_V9890", 16, reason))
    dump_csv("p1_256_action_smoke_v9890.csv", p1_preflight_rows("P1_256_ACTION_SMOKE_V9890", 256, reason))
    p2_rows, p2 = p2_density_panel(source_v9880, panel_targets)
    dump_csv("p2_natural_density_panel_v9890.csv", p2_rows)
    dump_csv("p2_density_by_dataset_family_template_v9890.csv", p2_density_by_dataset_family_template(source_v9880)[0])
    p3 = dump_csv("p3_future_path_type_decomposition_v9890.csv", not_run_artifact("P3_FUTURE_PATH_TYPE_DECOMPOSITION_V9890", reason, P3_weak_pass=0, P3_strong_pass=0)[0])
    p4 = dump_csv("p4_future_operator_sketch_v9890.csv", not_run_artifact("P4_FUTURE_OPERATOR_SKETCH_V9890", reason, FPS1_pass=0, FPS2_pass=0, FPS3_pass=0, FPS4_pass=0, FPS5_pass=0, P4_weak_pass=0, P4_strong_pass=0)[0])
    p5 = dump_csv("p5_controller_gate_v9890.csv", not_run_artifact("P5_CONTROLLER_GATE_V9890", "P1a_generator_missing_no_density_or_B_sketch", controller_pass=0)[0])
    p6 = dump_csv("p6_generated_sandbox_gate_v9890.csv", not_run_artifact("P6_GENERATED_SANDBOX_GATE_V9890", "P1a_generator_missing_no_D1_D2_D3_condition", P6_generated_sandbox_allowed=0, generated_actions=0, generated_sandbox_weak_pass=0)[0])
    p7 = dump_csv("p7_runtime_boundary_v9890.csv", not_run_artifact("P7_RUNTIME_BOUNDARY_V9890", "P5_controller_not_passed", runtime_pass=0)[0])
    p8 = dump_csv("p8_paired_replay_boundary_v9890.csv", not_run_artifact("P8_PAIRED_REPLAY_BOUNDARY_V9890", "P7_runtime_not_passed", paired_replay_pass=0, short_full_pass=0)[0])
    dump_csv("engineering_materializer_task_list_v9890.csv", engineering_task_list())
    dump_csv("field_legality_ledger_v9890.csv", field_legality())

    route = {
        "stage": "ROUTE_DECISION_V9890",
        "status": "summary",
        "route": "R-C1-NaturalGeneratorEntryMissing",
        "primary_blocker": "natural_extension_generator_missing",
        "secondary_blocker": "science_full_run_stopped_by_fail_fast",
        "route_recommendation": "engineering_only_materializer_task",
        "source_route_v9880": p0.get("source_route_v9880"),
        "P0_boundary_pass": p0.get("P0_pass"),
        "P1a_entrypoint_search_pass": p1.get("P1a_entrypoint_search_pass"),
        "P1_weak_pass": 0,
        "P1_strong_pass": 0,
        "P2_density_sufficient": p2.get("P2_density_sufficient"),
        "P2_density_insufficient": p2.get("P2_density_insufficient"),
        "P2_density_inconclusive": p2.get("P2_density_inconclusive"),
        "P3_future_path_weak_pass": p3.get("P3_weak_pass", 0),
        "P3_future_path_strong_pass": p3.get("P3_strong_pass", 0),
        "P4_future_operator_sketch_weak_pass": p4.get("P4_weak_pass", 0),
        "P4_future_operator_sketch_strong_pass": p4.get("P4_strong_pass", 0),
        "P5_controller_pass": p5.get("controller_pass"),
        "P6_generated_sandbox_allowed": p6.get("P6_generated_sandbox_allowed"),
        "generated_route_status": "stopped_P1a_generator_missing",
        "P7_runtime_pass": p7.get("runtime_pass"),
        "P8_paired_replay_pass": p8.get("paired_replay_pass"),
        "system_legal_controller_pass": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    dump_json("route_decision_v9890.json", route)
    artifacts.update(write_figures(out, route, p2))

    no_fake = {
        "stage": "NO_FAKE_AUDIT_V9890",
        "status": "summary",
        "rows_checked": artifact_row_count(artifacts),
        "fake_proxy_nonzero_count": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
        "no_fake": 1,
        "no_proxy": 1,
    }
    dump_csv("no_fake_audit_v9890.csv", [no_fake])
    contract = {
        "stage": "CONTRACT_AUDIT_V9890",
        "status": "summary",
        "v9880_boundary_pass": p0.get("P0_pass"),
        "P1a_generator_found": p1.get("natural_extension_action_generator_found"),
        "P1b_single_action_preflight": 0,
        "P1c_16_action_preflight": 0,
        "P1d_256_action_smoke": 0,
        "P2_density_sufficient/insufficient/inconclusive": f"{p2.get('P2_density_sufficient')}/{p2.get('P2_density_insufficient')}/{p2.get('P2_density_inconclusive')}",
        "controller/generated/runtime/system": "0/0/0/0",
        "diagnostic_promoted_to_official": 0,
        "fake/proxy/cpu_offload": "0/0/0",
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    dump_csv("contract_audit_v9890.csv", [contract])
    failure = {
        "stage": "FAILURE_TAXONOMY_V9890",
        "status": "summary",
        "route": route["route"],
        "F0_boundary_fail": int(not inum(p0.get("P0_pass"))),
        "F1_natural_generator_entry_missing": 1,
        "F2_natural_density_not_adjudicated": 1,
        "F3_future_operator_sketch_not_run": 1,
        "F4_controller_generated_runtime_blocked": 1,
        "primary_blocker": route["primary_blocker"],
        "secondary_blocker": route["secondary_blocker"],
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    dump_csv("failure_taxonomy_v9890.csv", [failure])

    hashes = sha_rows({"plan": PLAN_PATH, "runner": SCRIPT_PATH, **artifacts})
    manifest = {
        "stage": "RUN_MANIFEST_V9890",
        "status": "summary",
        "out_dir": str(out),
        "execution_profile": args.execution_profile,
        "device": args.device,
        "data_root": args.data_root,
        "seed": args.seed,
        "panel_targets": args.panel_targets,
        "source_v9880": args.source_v9880,
        "wallclock_sec": time.perf_counter() - started,
        "route": route["route"],
        "primary_blocker": route["primary_blocker"],
        "secondary_blocker": route["secondary_blocker"],
        "artifact_hashes": hashes,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    dump_json("run_manifest_v9890.json", manifest)
    hashes_for_recap = sha_rows({"plan": PLAN_PATH, "runner": SCRIPT_PATH, **artifacts})
    write_recap(out, route, hashes_for_recap)

    print(json.dumps({
        "out_dir": str(out),
        "route": route["route"],
        "primary_blocker": route["primary_blocker"],
        "secondary_blocker": route["secondary_blocker"],
        "P1a_entrypoint_search_pass": route["P1a_entrypoint_search_pass"],
        "P2_density_inconclusive": route["P2_density_inconclusive"],
        "P6_generated_sandbox_allowed": route["P6_generated_sandbox_allowed"],
        "system_legal_controller_pass": route["system_legal_controller_pass"],
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
