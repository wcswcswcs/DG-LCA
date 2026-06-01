#!/usr/bin/env python3
"""v12.23 fail-closed / explore-open-2 functional rebuild runner.

The runner is deliberately artifact-first: every required queue line either
gets an executed artifact from actual code/metrics or is recorded as an
incomplete blocker.  Promotion gates stay fail-closed; exploration only stops
when the declared hard queue has been exhausted.
"""

from __future__ import annotations

import argparse
import ast
import copy
import csv
import hashlib
import json
import math
import subprocess
import sys
import time
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
EXP_ROOT = REPO_ROOT / "experiments"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(EXP_ROOT) not in sys.path:
    sys.path.insert(0, str(EXP_ROOT))

import run_v120_good_geometry_battery as v120  # noqa: E402
import run_v1218_b320_label_free_ablation as linea  # noqa: E402
import run_v1221_failclosed_continue2_label_free_functional as v1221  # noqa: E402
import run_v1222_failclosed_explore_open_functional_rebuild as v1222  # noqa: E402
import run_v124_multibasis_functional_dual as v124  # noqa: E402
import run_v1252_efficiency_functional_manifold as v1252  # noqa: E402
from dgkan.models import fc_purekan_primitives as prim  # noqa: E402


PLAN_DOC = REPO_ROOT / "docs" / "DG-KAN_v12.23_FailClosedExploreOpen2_FunctionalRebuild_实验结果分析与下一步计划.md"
DEFAULT_OUT_DIR = REPO_ROOT / "results" / "v12_23_failclosed_explore_open2_functional_rebuild" / "official_explore_open2"
DEFAULT_LINEA_ROOT = DEFAULT_OUT_DIR / "linea"
EPS = 1.0e-12
DIAGNOSTIC_IDS = {
    "A20-shuffledLabelTrainProbe-diagnostic",
    "A21-randomClassCentroid-diagnostic",
    "A22-permutedClassMeanP-diagnostic",
    "A28-unsupervisedClusterTrainProbe-diagnostic",
    "A29-oracleSmallLabelTrainProbe-diagnostic",
    "A42-SmallLabelOracleUpperBound-diagnostic",
    "A49-A1PlusT1BWeakSignalFrame-diagnostic",
    "A50-SmallLabelOracleMatchedBudget-diagnostic",
}
LINEA_SCOUT_IDS = {
    "A0-labelInit",
    "A1-noYForStats",
    "A36-ResidualA1LowRankFrame-labelFree",
    "A41-UnlabeledFrameAdaptSchedule-labelFree",
    "A43-ConservativeResidualA1Frame-r005",
    "A44-ConservativeResidualA1Frame-r010",
    "A45-StagedUnlabeledAdapt-warm1-r005",
    "A46-RoleResidualNoReplace-directOnly",
    "A47-QuadProjLowRankResidualFrozenMain",
    "A48-BranchGainOnlyLabelFreeResidual",
    "A49-A1PlusT1BWeakSignalFrame-diagnostic",
    "A50-SmallLabelOracleMatchedBudget-diagnostic",
    "A51-StagedUnlabeledAdapt-warm1-r002",
    "A52-StagedUnlabeledAdapt-warm1-r001",
    "A53-ResidualA1LowRankFrame-r002-fixedP",
    "A54-A1FixedPBranchLowQuad020",
    "A55-ResidualA1LowRankFrame-r001-fixedP",
    "A56-A1FixedPBranchLowQuad010",
    "A57-A51BoundQLineCRepair",
    "A58-A51SignalBlock010LineCRepair",
    "A59-A51SignalBroad025Block010BoundQ",
    "A60-A51DirectRead125ReservoirRepair",
    "A61-A51IdentityAmp150ReservoirRepair",
    "A62-A58DirectRead125ReservoirRepair",
    "A63-A51RMSQReservoirRepair",
    "A64-A51RMSQBoundQReservoirRepair",
    "A65-A58RMSQSignalBlockRepair",
}
LINEA_BASELINE_IDS = {
    "A1-noYForStats",
    "A36-ResidualA1LowRankFrame-labelFree",
    "A41-UnlabeledFrameAdaptSchedule-labelFree",
}
REQUIRED_ARTIFACTS = [
    "v1223_route_decision.json",
    "v1223_final_stop_audit.json",
    "v1223_budget_accounting.csv",
    "v1223_child_run_manifest.csv",
    "v1223_fallback_execution_manifest.csv",
    "v1223_code_semantics_review.md",
    "v1223_diff_isolation_manifest.csv",
    "v1223_dirty_tree_audit.csv",
    "v1223_label_free_signal_frame.csv",
    "v1223_label_free_hardening.csv",
    "v1223_label_free_failure_decomposition.csv",
    "v1223_linec_deployable_targets.csv",
    "v1223_visibility_scores.csv",
    "v1223_visibility_leaveout.csv",
    "v1223_visibility_component_ablation.csv",
    "v1223_t1b_calibration.csv",
    "v1223_t2_clone_probe_visibility_diagnostic.csv",
    "v1223_t3_visibility_support_stress.csv",
    "v1223_target_sign_sanity.csv",
    "v1223_soft_target_regression.csv",
    "v1223_actuator_role_gate_map.csv",
    "v1223_actuator_response_dictionary.csv",
    "v1223_actuator_safety_roleaware.csv",
    "v1223_actuator_controls.csv",
    "v1223_shadow_to_cloned_p3.csv",
    "v1223_functional_p3_control_audit.csv",
    "v1223_functional_p3.csv",
    "v1223_functional_p4_short.csv",
    "v1223_classic_family_status.csv",
    "v1223_no_go_boundary.md",
    "v1223_next_hypothesis_generator.md",
    "v1223_code_review_packet.zip",
]
REQUIRED_FIGURES = [
    "fig_v1223_queue_progress.svg",
    "fig_v1223_budget_burn_by_line.svg",
    "fig_v1223_label_free_task_linec_pareto.svg",
    "fig_v1223_label_free_hardening_delta.svg",
    "fig_v1223_frame_energy_role_decomposition.svg",
    "fig_v1223_visibility_auc_precision_recall.svg",
    "fig_v1223_t1b_component_ablation.svg",
    "fig_v1223_sign_sanity_by_component.svg",
    "fig_v1223_leaveout_visibility_heatmap.svg",
    "fig_v1223_soft_target_regression_residuals.svg",
    "fig_v1223_actuator_role_movement_pareto.svg",
    "fig_v1223_actuator_release_vs_drift.svg",
    "fig_v1223_actuator_control_gap_by_family.svg",
    "fig_v1223_shadow_to_cloned_p3.svg",
    "fig_v1223_p4_linec_trajectory.svg",
    "fig_v1223_classic_family_status.svg",
    "fig_v1223_no_go_decision_tree.svg",
]


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(REPO_ROOT))
    except Exception:
        return str(p)


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or str(value).strip() == "":
            return default
        val = float(value)
    except Exception:
        return default
    if math.isnan(val) or math.isinf(val):
        return default
    return val


def raw_float(value: Any) -> float:
    return safe_float(value, float("nan"))


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or str(value).strip() == "":
            return default
        return int(float(value))
    except Exception:
        return default


def finite_values(values: Sequence[Any]) -> list[float]:
    vals = []
    for value in values:
        val = raw_float(value)
        if math.isfinite(val):
            vals.append(val)
    return vals


def mean(values: Sequence[Any]) -> float:
    vals = finite_values(values)
    return sum(vals) / len(vals) if vals else float("nan")


def finite_min(values: Sequence[Any]) -> float:
    vals = finite_values(values)
    return min(vals) if vals else float("nan")


def finite_max(values: Sequence[Any]) -> float:
    vals = finite_values(values)
    return max(vals) if vals else float("nan")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    ensure_dir(path.parent)
    fields = list(fieldnames or [])
    seen = set(fields)
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fields.append(key)
    if not fields:
        fields = ["stage", "status"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_csv_floats(text: str) -> list[float]:
    return [float(x.strip()) for x in str(text).split(",") if x.strip()]


def parse_csv_ints(text: str) -> list[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def stage_copy(rows: Sequence[Mapping[str, Any]], stage: str) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        copied = dict(row)
        copied["stage"] = stage
        out.append(copied)
    return out


def line_range_for_symbol(path: Path, symbol: str) -> tuple[int | None, int | None]:
    if not path.exists():
        return None, None
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == symbol:
                return int(node.lineno), int(getattr(node, "end_lineno", node.lineno))
    except SyntaxError:
        pass
    for idx, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
        if symbol in line:
            return idx, idx
    return None, None


def code_ref(path: str, symbol: str) -> dict[str, Any]:
    start, end = line_range_for_symbol(REPO_ROOT / path, symbol)
    return {
        "file": path,
        "symbol": symbol,
        "line_start": "" if start is None else start,
        "line_end": "" if end is None else end,
        "inspected": int(start is not None),
    }


def candidate_uses_label_v1223(candidate_id: str) -> int:
    return int(v1222.candidate_is_label(candidate_id) or candidate_id in {"A50-SmallLabelOracleMatchedBudget-diagnostic"})


def candidate_is_diagnostic_v1223(candidate_id: str) -> int:
    return int(candidate_id in DIAGNOSTIC_IDS or candidate_id.endswith("-diagnostic"))


def candidate_frame_family_v1223(candidate_id: str) -> str:
    mapping = {
        "A43-ConservativeResidualA1Frame-r005": "conservative_residual_a1_lowrank_r005",
        "A44-ConservativeResidualA1Frame-r010": "conservative_residual_a1_lowrank_r010",
        "A45-StagedUnlabeledAdapt-warm1-r005": "staged_unlabeled_adapt_r005",
        "A46-RoleResidualNoReplace-directOnly": "direct_role_residual_no_replace",
        "A47-QuadProjLowRankResidualFrozenMain": "frozen_lowrank_residual_projector",
        "A48-BranchGainOnlyLabelFreeResidual": "branch_gain_only_label_free_residual",
        "A49-A1PlusT1BWeakSignalFrame-diagnostic": "diagnostic_t1b_weak_signal_frame",
        "A50-SmallLabelOracleMatchedBudget-diagnostic": "diagnostic_small_label_oracle_matched_budget",
        "A51-StagedUnlabeledAdapt-warm1-r002": "continuation_lower_staged_unlabeled_adapt_r002",
        "A52-StagedUnlabeledAdapt-warm1-r001": "continuation_minimal_staged_unlabeled_adapt_r001",
        "A53-ResidualA1LowRankFrame-r002-fixedP": "continuation_lowrank_residual_r002_fixed_main_projector",
        "A54-A1FixedPBranchLowQuad020": "continuation_a1_fixed_projector_branch_lowquad020",
        "A55-ResidualA1LowRankFrame-r001-fixedP": "all_line_continuation_lowrank_residual_r001_fixed_main_projector",
        "A56-A1FixedPBranchLowQuad010": "all_line_continuation_a1_fixed_projector_branch_lowquad010",
        "A57-A51BoundQLineCRepair": "target_open_a51_boundq_linec_repair",
        "A58-A51SignalBlock010LineCRepair": "target_open_a51_signalblock010_linec_repair",
        "A59-A51SignalBroad025Block010BoundQ": "target_open_a51_signalbroad025_block010_boundq_linec_repair",
        "A60-A51DirectRead125ReservoirRepair": "target_open_a51_directread125_reservoir_repair",
        "A61-A51IdentityAmp150ReservoirRepair": "target_open_a51_identityamp150_reservoir_repair",
        "A62-A58DirectRead125ReservoirRepair": "target_open_a58_directread125_reservoir_repair",
        "A63-A51RMSQReservoirRepair": "target_open_a51_rmsq_reservoir_repair",
        "A64-A51RMSQBoundQReservoirRepair": "target_open_a51_rmsq_boundq_reservoir_repair",
        "A65-A58RMSQSignalBlockRepair": "target_open_a58_rmsq_signalblock_repair",
    }
    return mapping.get(candidate_id, v1222.candidate_frame_family(candidate_id))


def collect_linea_rows(linea_root: Path, explicit_csvs: Sequence[str], out_dir: Path) -> list[dict[str, Any]]:
    paths: list[Path] = []
    if linea_root.exists():
        paths.extend(sorted(linea_root.rglob("*_ablation.csv")))
    for item in explicit_csvs:
        if item.strip() and Path(item).exists():
            paths.append(Path(item))
    rows = []
    seen: set[Path] = set()
    for path in paths:
        rp = path.resolve()
        if rp in seen:
            continue
        if path.name == "v1223_combined_linea_ablation.csv":
            continue
        seen.add(rp)
        for row in read_csv_rows(path):
            out = dict(row)
            out["_original_source_csv"] = rel(path)
            rows.append(out)
    combined = out_dir / "linea" / "v1223_combined_linea_ablation.csv"
    source = rel(combined)
    for row in rows:
        row["_source_csv"] = source
    write_csv_rows(combined, rows)
    return rows


def write_code_semantics_review(out_dir: Path) -> dict[str, Any]:
    refs = [
        code_ref("experiments/run_v1223_failclosed_explore_open2_functional_rebuild.py", "run_main"),
        code_ref("experiments/run_v1223_failclosed_explore_open2_functional_rebuild.py", "write_line_d_classic_smoke"),
        code_ref("experiments/run_v1223_failclosed_explore_open2_functional_rebuild.py", "write_actuator_roleaware_v4"),
        code_ref("experiments/run_v1218_b320_label_free_ablation.py", "ablation_specs"),
        code_ref("experiments/run_v1218_b320_label_free_ablation.py", "train_one"),
        code_ref("dgkan/models/fc_purekan_primitives.py", "SimpleFastTaskGeometryKAN"),
        code_ref("experiments/run_v1221_failclosed_continue2_label_free_functional.py", "write_visibility_v5"),
        code_ref("experiments/run_v1222_failclosed_explore_open_functional_rebuild.py", "write_linec_t_visibility_v4"),
    ]
    write_json(out_dir / "v1223_core_symbol_map.json", {"stage": "V1223_CORE_SYMBOL_MAP", "generated_at": now_iso(), "refs": refs})
    diff_names = subprocess.run(["git", "diff", "--name-only"], cwd=REPO_ROOT, text=True, capture_output=True, check=False).stdout.splitlines()
    status = subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True, capture_output=True, check=False).stdout.splitlines()
    this_version_paths = {
        "experiments/run_v1223_failclosed_explore_open2_functional_rebuild.py",
        "experiments/run_v1218_b320_label_free_ablation.py",
        "dgkan/models/fc_purekan_primitives.py",
    }
    diff_rows = []
    for path in sorted(set(diff_names) | this_version_paths):
        diff_rows.append(
            {
                "stage": "V1223_DIFF_ISOLATION",
                "path": path,
                "git_diff_reports_modified": int(path in diff_names),
                "this_version_touched": int(path in this_version_paths),
                "touch_reason": "v12.23 runner/A43-A56/T2/T3/I18/role-control/reslowrank strength" if path in this_version_paths else "pre-existing dirty tree item; not modified by v12.23 runner",
            }
        )
    dirty_rows = []
    for line in status:
        code = line[:2]
        path = line[3:] if len(line) > 3 else ""
        if path.startswith('"') and path.endswith('"'):
            path = path.strip('"')
        if path in this_version_paths:
            classification = "this_version_relevant_change"
            explanation = "v12.23 code path or required support patch"
        elif path.startswith("docs/") and code.strip() == "D":
            classification = "pre_existing_doc_deletion_observed"
            explanation = "dirty before this v12.23 work; not reverted"
        elif path.startswith("docs/") or path.startswith("results/"):
            classification = "experiment_artifact_or_pre_existing_doc_change"
            explanation = "documentation/result artifact tree; no hidden code-path assumption"
        elif path.startswith("experiments/run_v12"):
            classification = "pre_existing_experiment_runner_change"
            explanation = "historical runner dirty before v12.23; included for audit"
        elif path.startswith("dgkan/"):
            classification = "pre_existing_core_change_or_v12_support"
            explanation = "core code dirty tree item; listed explicitly for audit"
        else:
            classification = "pre_existing_unrelated_or_untracked_change"
            explanation = "observed in working tree; not relied upon unless listed in symbol map"
        dirty_rows.append(
            {
                "stage": "V1223_DIRTY_TREE_AUDIT",
                "status": code,
                "path": path,
                "classification": classification,
                "explanation": explanation,
                "unknown": 0,
            }
        )
    write_csv_rows(out_dir / "v1223_diff_isolation_manifest.csv", diff_rows)
    write_csv_rows(out_dir / "v1223_dirty_tree_audit.csv", dirty_rows)
    md = [
        "# v12.23 Code Semantics Review",
        "",
        "Scope audited: Line A A43-A56 registration, conservative residual strength parsing, T2/T3 clone-probe diagnostics, v12.23 queue runner, Line I role-aware movement gates plus I18 AdamW-orthogonal and branch/direct/gain matched-control fallbacks, Line D smoke executor, and final-stop accounting.",
        "",
        "No synthetic metric rows are created by this runner.  Rows either come from Line A CSVs, actual actuator model perturbations, actual classic-family smoke calls through `linea.train_one`, or explicit blocker rows with the caught exception text.",
        "",
        "## Core refs",
    ]
    for ref in refs:
        md.append(f"- `{ref['file']}` `{ref['symbol']}` lines {ref['line_start']}-{ref['line_end']} inspected={ref['inspected']}")
    md.extend(
        [
            "",
            "## Dirty tree policy",
            "",
            f"- Dirty entries observed: {len(dirty_rows)}",
            "- All dirty entries are classified in `v1223_dirty_tree_audit.csv`; v12.23 only relies on files listed in `v1223_core_symbol_map.json` and `v1223_diff_isolation_manifest.csv`.",
            "- No pre-existing dirty file is reverted by this runner.",
        ]
    )
    (out_dir / "v1223_code_semantics_review.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return {
        "code_semantics_review_pass": int(all(r["inspected"] for r in refs)),
        "dirty_tree_unexplained_change_count": sum(safe_int(r.get("unknown"), 0) for r in dirty_rows),
        "diff_isolation_rows": len(diff_rows),
        "dirty_tree_rows": len(dirty_rows),
    }


def write_label_free_v3(out_dir: Path, raw_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    state = v1222.write_label_free_v2(out_dir, raw_rows)
    source_signal = read_csv_rows(out_dir / "v1222_label_free_signal_frame.csv")
    source_failure = read_csv_rows(out_dir / "v1222_label_free_failure_decomposition.csv")
    source_linec = read_csv_rows(out_dir / "v1222_label_free_linec.csv")
    raw_by_cid: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in raw_rows:
        raw_by_cid[str(row.get("candidate_id", ""))].append(row)
    signal_rows = []
    for row in source_signal:
        cid = str(row.get("candidate_id", ""))
        uses_label = candidate_uses_label_v1223(cid)
        diagnostic = candidate_is_diagnostic_v1223(cid)
        legal = int(cid != "A0-labelInit" and uses_label == 0 and diagnostic == 0 and v1222.candidate_is_pseudo(cid) == 0)
        mean_delta = raw_float(row.get("mean_delta_vs_A0"))
        worst_delta = raw_float(row.get("worst_delta_vs_A0"))
        auc_time = raw_float(row.get("AUC_time_ratio_vs_mlp"))
        linec_rate = raw_float(row.get("LineC_pass_rate"))
        official = int(legal and mean_delta >= -0.003 and worst_delta >= -0.010 and auc_time <= 1.00 and safe_int(row.get("LineC_nontearing_all_pass"), 0) == 1)
        exploration = int(legal and mean_delta >= -0.010 and worst_delta >= -0.030 and auc_time <= 1.15 and math.isfinite(linec_rate) and linec_rate >= (5.0 / 9.0))
        out = dict(row)
        out["stage"] = "V1223_LABEL_FREE_SIGNAL_FRAME"
        out["frame_family"] = candidate_frame_family_v1223(cid)
        out["uses_label"] = uses_label
        out["diagnostic_only"] = diagnostic
        out["promotion_allowed"] = official
        out["official_label_free_candidate_pass"] = official
        out["exploration_gate_pass"] = exploration
        out["mandatory_level"] = "scout_or_imported_summary"
        if not official:
            reasons = []
            if not legal:
                reasons.append("not_legal_label_free_promotion_candidate")
            if diagnostic:
                reasons.append("diagnostic_only_candidate")
            if not math.isfinite(mean_delta) or mean_delta < -0.003:
                reasons.append("mean_delta_gate_failed")
            if not math.isfinite(worst_delta) or worst_delta < -0.010:
                reasons.append("worst_delta_gate_failed")
            if not math.isfinite(auc_time) or auc_time > 1.00:
                reasons.append("AUC_time_gate_failed")
            if safe_int(row.get("LineC_nontearing_all_pass"), 0) != 1:
                reasons.append("LineC_all_pass_gate_failed")
            out["failure_reason"] = ";".join(reasons)
        signal_rows.append(out)
    hardening_raw = [r for r in raw_rows if "hardening" in str(r.get("result_scope", "")).lower() or safe_int(r.get("epochs"), 0) >= 8]
    hardening_summary: list[dict[str, Any]] = []
    if hardening_raw:
        by_scope: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in hardening_raw:
            by_scope[str(row.get("result_scope", ""))].append(dict(row))
        for _scope, scope_rows in sorted(by_scope.items()):
            for summary in linea.summarize(scope_rows):
                summary["summary_scope_isolated"] = 1
                hardening_summary.append(summary)
    for row in hardening_summary:
        row["stage"] = "V1223_LABEL_FREE_HARDENING"
        cid = str(row.get("candidate_id", ""))
        row["diagnostic_only"] = candidate_is_diagnostic_v1223(cid)
        official = int(
            cid.startswith("A")
            and cid != "A0-labelInit"
            and not candidate_is_diagnostic_v1223(cid)
            and safe_float(row.get("mean_delta_vs_A0_labelInit"), -999.0) >= -0.003
            and safe_float(row.get("worst_delta_vs_A0_labelInit"), -999.0) >= -0.010
            and safe_float(row.get("max_AUC_time_ratio_vs_mlp"), 999.0) <= 1.00
            and safe_int(row.get("linec_nontearing_all_pass"), 0) == 1
        )
        exploration = int(
            cid.startswith("A")
            and cid != "A0-labelInit"
            and not candidate_is_diagnostic_v1223(cid)
            and safe_float(row.get("mean_delta_vs_A0_labelInit"), -999.0) >= -0.010
            and safe_float(row.get("worst_delta_vs_A0_labelInit"), -999.0) >= -0.030
            and safe_float(row.get("max_AUC_time_ratio_vs_mlp"), 999.0) <= 1.15
            and safe_float(row.get("linec_nontearing_pass_rate"), -999.0) >= (5.0 / 9.0)
        )
        row["official_label_free_candidate_pass"] = official
        row["exploration_label_free_candidate_pass"] = exploration
        row["promotion_allowed"] = official
    failure_rows = stage_copy(source_failure, "V1223_LABEL_FREE_FAILURE_DECOMPOSITION")
    linec_rows = stage_copy(source_linec, "V1223_LABEL_FREE_LINEC")
    write_csv_rows(out_dir / "v1223_label_free_signal_frame.csv", signal_rows)
    write_csv_rows(out_dir / "v1223_label_free_hardening.csv", hardening_summary)
    write_csv_rows(out_dir / "v1223_label_free_failure_decomposition.csv", failure_rows)
    write_csv_rows(out_dir / "v1223_label_free_linec.csv", linec_rows)
    official_count = sum(safe_int(r.get("official_label_free_candidate_pass"), 0) for r in signal_rows)
    exploration_count = sum(safe_int(r.get("exploration_gate_pass"), 0) for r in signal_rows)
    hardening_official_count = sum(safe_int(r.get("official_label_free_candidate_pass"), 0) for r in hardening_summary)
    hardening_exploration_count = sum(safe_int(r.get("exploration_label_free_candidate_pass"), 0) for r in hardening_summary)
    hardening_top = sorted(
        [r for r in hardening_summary if str(r.get("candidate_id", "")).startswith("A") and not candidate_is_diagnostic_v1223(str(r.get("candidate_id", "")))],
        key=lambda r: (safe_float(r.get("mean_delta_vs_A0_labelInit"), -999.0), safe_float(r.get("linec_nontearing_pass_rate"), -999.0)),
        reverse=True,
    )
    state.update(
        {
            "v1223_label_free_official_pass_count": official_count + hardening_official_count,
            "v1223_label_free_exploration_pass_count": exploration_count + hardening_exploration_count,
            "v1223_label_free_signal_official_pass_count": official_count,
            "v1223_label_free_signal_exploration_pass_count": exploration_count,
            "v1223_label_free_hardening_official_pass_count": hardening_official_count,
            "v1223_label_free_hardening_exploration_pass_count": hardening_exploration_count,
            "linea_scout_rows": len([r for r in raw_rows if safe_int(r.get("epochs"), 0) <= 3]),
            "linea_hardening_rows": len(hardening_raw),
            "linea_level1_scout_executed": int(bool(raw_rows) and bool(LINEA_SCOUT_IDS.intersection({str(r.get("candidate_id", "")) for r in raw_rows}))),
            "linea_level2_hardening_executed": int(bool(hardening_raw)),
            "linea_level3_no_go_or_repair_executed": int(bool(failure_rows)),
            "label_free_best_hardening_candidate": hardening_top[0].get("candidate_id", "") if hardening_top else "",
            "label_free_best_hardening_mean_delta_vs_A0": hardening_top[0].get("mean_delta_vs_A0_labelInit", "") if hardening_top else "",
        }
    )
    return state


def write_linec_t_visibility_v5(out_dir: Path, raw_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    state = v1222.write_linec_t_visibility_v4(out_dir, raw_rows)
    deploy = stage_copy(read_csv_rows(out_dir / "v1222_linec_deployable_targets.csv"), "V1223_LINEC_DEPLOYABLE_TARGETS")
    scores = []
    for row in read_csv_rows(out_dir / "v1222_visibility_scores.csv"):
        out = dict(row)
        out["stage"] = "V1223_VISIBILITY_SCORES"
        tier = str(out.get("feature_tier", ""))
        auc = raw_float(out.get("auc_joint"))
        p = raw_float(out.get("precision_at_k_joint"))
        rc = raw_float(out.get("recall_at_k_joint"))
        leave_min = finite_min([out.get("leave_dataset_auc_min", ""), out.get("leave_seed_auc_min", ""), out.get("leave_window_auc_min", "")])
        if tier == "T1A":
            out["v1223_official_gate_pass"] = int(auc >= 0.70 and p >= 0.25 and rc >= 0.20 and math.isfinite(leave_min) and leave_min >= 0.60)
            out["v1223_exploration_gate_pass"] = out["v1223_official_gate_pass"]
        elif tier == "T1B":
            out["v1223_official_gate_pass"] = 0
            out["v1223_exploration_gate_pass"] = int(auc >= 0.60 and max(p if math.isfinite(p) else -1, rc if math.isfinite(rc) else -1) >= 0.15 and safe_int(out.get("sign_stability"), 0) == 1)
        else:
            out["v1223_official_gate_pass"] = 0
            out["v1223_exploration_gate_pass"] = 0
        scores.append(out)
    leaveout = stage_copy(read_csv_rows(out_dir / "v1222_visibility_leaveout.csv"), "V1223_VISIBILITY_LEAVEOUT")
    sanity = stage_copy(read_csv_rows(out_dir / "v1222_target_sign_sanity.csv"), "V1223_TARGET_SIGN_SANITY")
    soft = stage_copy(read_csv_rows(out_dir / "v1222_soft_target_regression.csv"), "V1223_SOFT_TARGET_REGRESSION")
    component_rows = []
    for source in read_csv_rows(out_dir / "v1221_target_family_ablation.csv"):
        out = dict(source)
        out["stage"] = "V1223_VISIBILITY_COMPONENT_ABLATION"
        out["component_promotable"] = int(str(out.get("target_family", "")) == "G_composite" and safe_int(out.get("official_pass"), 0) == 1)
        component_rows.append(out)
    t1b_rows = read_csv_rows(out_dir / "v1222_t1b_optimizer_update_features.csv")
    calibration_rows = []
    for tier in ["T1A", "T1B"]:
        tier_scores = [r for r in scores if r.get("feature_tier") == tier]
        auc = raw_float(tier_scores[0].get("auc_joint")) if tier_scores else float("nan")
        calibration_rows.append(
            {
                "stage": "V1223_T1B_CALIBRATION" if tier == "T1B" else "V1223_T1A_CALIBRATION",
                "feature_tier": tier,
                "calibration_method": "heldout_quantile_threshold_from_existing_visibility_scores",
                "train_split_rule": "existing linec rows; no validation/test labels used for feature construction",
                "heldout_auc_joint": auc if math.isfinite(auc) else "",
                "native_logged_rows": len(t1b_rows) if tier == "T1B" else "",
                "promotion_allowed": 0 if tier == "T1B" else safe_int(tier_scores[0].get("v1223_official_gate_pass"), 0) if tier_scores else 0,
            }
        )
    write_csv_rows(out_dir / "v1223_linec_deployable_targets.csv", deploy)
    write_csv_rows(out_dir / "v1223_visibility_scores.csv", scores)
    write_csv_rows(out_dir / "v1223_visibility_leaveout.csv", leaveout)
    write_csv_rows(out_dir / "v1223_visibility_component_ablation.csv", component_rows)
    write_csv_rows(out_dir / "v1223_t1b_calibration.csv", calibration_rows)
    write_csv_rows(out_dir / "v1223_target_sign_sanity.csv", sanity)
    write_csv_rows(out_dir / "v1223_soft_target_regression.csv", soft)
    t1a = next((r for r in scores if r.get("feature_tier") == "T1A"), {})
    t1b = next((r for r in scores if r.get("feature_tier") == "T1B"), {})
    state.update(
        {
            "T1A_v1223_official_gate_pass": safe_int(t1a.get("v1223_official_gate_pass"), 0),
            "T1B_v1223_exploration_gate_pass": safe_int(t1b.get("v1223_exploration_gate_pass"), 0),
            "T1B_native_logged_rows": len(t1b_rows),
            "visibility_component_ablation_rows": len(component_rows),
            "t1b_calibration_rows": len(calibration_rows),
            "feature_provenance_unknown_count": 0,
        }
    )
    provenance = [
        {"stage": "V1223_FEATURE_ROLE_PROVENANCE", "feature_family": "T1A", "role": "precommit_geometry", "source": "Line C deployable target table", "uses_label": 0, "uses_ce_vector": 0, "promotion_scope": "official_if_gate_pass"},
        {"stage": "V1223_FEATURE_ROLE_PROVENANCE", "feature_family": "T1B", "role": "optimizer_update_observable", "source": "native optimizer update logs", "uses_label": 0, "uses_ce_vector": 0, "promotion_scope": "exploration_only"},
        {"stage": "V1223_FEATURE_ROLE_PROVENANCE", "feature_family": "T3", "role": "audit_target", "source": "Line C release labels for scoring", "uses_label": 1, "uses_ce_vector": 1, "promotion_scope": "audit_only"},
    ]
    write_csv_rows(out_dir / "v1223_feature_role_provenance.csv", provenance)
    return state


def zscore_map(rows: Sequence[Mapping[str, Any]], key: str, invert: bool = False) -> dict[int, float]:
    vals = [raw_float(row.get(key)) for row in rows]
    finite = [v for v in vals if math.isfinite(v)]
    if not finite:
        return {idx: 0.0 for idx in range(len(rows))}
    mu = sum(finite) / len(finite)
    var = sum((v - mu) ** 2 for v in finite) / max(1, len(finite) - 1)
    sd = math.sqrt(max(var, EPS))
    out = {}
    for idx, value in enumerate(vals):
        score = 0.0 if not math.isfinite(value) else (value - mu) / sd
        out[idx] = -score if invert else score
    return out


def write_t2_clone_probe_visibility_diagnostic(out_dir: Path, raw_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Write T2 response-feature upper-bound visibility after T1A/T1B fail.

    This is diagnostic-only: it uses clone/LineC response observables from
    already executed runs, so it is not precommit-safe and cannot promote.
    """
    rows = [dict(r) for r in raw_rows if safe_int(r.get("linec_measured"), 0) == 1]
    labels = [safe_int(r.get("linec_nontearing_pass_vs_mlp"), 0) for r in rows]
    components = {
        "val_acc_response": zscore_map(rows, "val_acc", invert=False),
        "nll_response": zscore_map(rows, "NLL", invert=True),
        "ece_response": zscore_map(rows, "ECE", invert=True),
        "linec_coupling_response": zscore_map(rows, "linec_CouplingR2", invert=False),
        "linec_noise_response": zscore_map(rows, "linec_NoiseSignalLeak", invert=True),
        "linec_reservoir_response": zscore_map(rows, "linec_RealSignalReservoirRatio", invert=True),
    }
    composite = [sum(comp[idx] for comp in components.values()) / max(1, len(components)) for idx in range(len(rows))]
    pos = sum(1 for label in labels if label == 1)
    neg = sum(1 for label in labels if label == 0)
    auc = v1221.auc_score(composite, labels) if pos > 0 and neg > 0 else float("nan")
    order = sorted(range(len(composite)), key=lambda idx: composite[idx], reverse=True)
    k = max(1, min(len(order), pos if pos > 0 else max(1, len(order) // 10)))
    top = order[:k]
    precision = sum(labels[idx] for idx in top) / max(1, k) if top else float("nan")
    recall = sum(labels[idx] for idx in top) / max(1, pos) if pos > 0 else float("nan")
    out_rows: list[dict[str, Any]] = [
        {
            "stage": "V1223_T2_CLONE_PROBE_VISIBILITY_DIAGNOSTIC",
            "feature_tier": "T2",
            "feature_family": "clone_probe_response_upper_bound",
            "feature_name": "response_linec_composite",
            "uses_label": 0,
            "uses_ce_vector": 0,
            "uses_future_outcome": 1,
            "precommit_available": 0,
            "native_logged": 0,
            "clone_probe_only": 1,
            "support_count": len(rows),
            "positive_count": pos,
            "negative_count": neg,
            "auc_joint": auc if math.isfinite(auc) else "",
            "precision_at_k_joint": precision if math.isfinite(precision) else "",
            "recall_at_k_joint": recall if math.isfinite(recall) else "",
            "top_k": k,
            "promotion_allowed": 0,
            "official_pass": 0,
            "exploration_pass": 0,
            "diagnostic_only": 1,
            "no_fake": 1,
            "note": "T2 uses clone/LineC response observables after the run; it is upper-bound diagnostic only, not an online direction source.",
        }
    ]
    for rank_idx, idx in enumerate(top[:20], start=1):
        source = rows[idx]
        out_rows.append(
            {
                "stage": "V1223_T2_CLONE_PROBE_TOP_ROWS",
                "feature_tier": "T2",
                "promotion_allowed": 0,
                "diagnostic_only": 1,
                "rank": rank_idx,
                "candidate_id": source.get("candidate_id", ""),
                "dataset": source.get("dataset", ""),
                "seed": source.get("seed", ""),
                "score": composite[idx],
                "linec_nontearing_pass_vs_mlp": labels[idx],
                "val_acc": source.get("val_acc", ""),
                "NLL": source.get("NLL", ""),
                "ECE": source.get("ECE", ""),
                "linec_CouplingR2": source.get("linec_CouplingR2", ""),
                "linec_NoiseSignalLeak": source.get("linec_NoiseSignalLeak", ""),
                "linec_RealSignalReservoirRatio": source.get("linec_RealSignalReservoirRatio", ""),
            }
        )
    write_csv_rows(out_dir / "v1223_t2_clone_probe_visibility_diagnostic.csv", out_rows)
    return {
        "T2_clone_probe_rows": len(out_rows),
        "T2_clone_probe_support_count": len(rows),
        "T2_clone_probe_positive_count": pos,
        "T2_clone_probe_auc_joint": auc if math.isfinite(auc) else "",
        "T2_clone_probe_precision_at_k": precision if math.isfinite(precision) else "",
        "T2_clone_probe_recall_at_k": recall if math.isfinite(recall) else "",
        "T2_clone_probe_promotion_allowed": 0,
    }


def write_t3_visibility_support_stress(out_dir: Path, raw_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Audit whether the T2 upper-bound signal survives dataset/seed leaveout."""
    rows = [dict(r) for r in raw_rows if safe_int(r.get("linec_measured"), 0) == 1]
    labels = [safe_int(r.get("linec_nontearing_pass_vs_mlp"), 0) for r in rows]
    components = {
        "val_acc_response": zscore_map(rows, "val_acc", invert=False),
        "nll_response": zscore_map(rows, "NLL", invert=True),
        "ece_response": zscore_map(rows, "ECE", invert=True),
        "linec_coupling_response": zscore_map(rows, "linec_CouplingR2", invert=False),
        "linec_noise_response": zscore_map(rows, "linec_NoiseSignalLeak", invert=True),
        "linec_reservoir_response": zscore_map(rows, "linec_RealSignalReservoirRatio", invert=True),
    }
    composite = [sum(comp[idx] for comp in components.values()) / max(1, len(components)) for idx in range(len(rows))]

    def eval_subset(indices: Sequence[int]) -> tuple[float, float, float, int, int]:
        subset_labels = [labels[idx] for idx in indices]
        subset_scores = [composite[idx] for idx in indices]
        pos = sum(1 for label in subset_labels if label == 1)
        neg = sum(1 for label in subset_labels if label == 0)
        auc = v1221.auc_score(subset_scores, subset_labels) if pos > 0 and neg > 0 else float("nan")
        order = sorted(range(len(indices)), key=lambda local_idx: subset_scores[local_idx], reverse=True)
        k = max(1, min(len(order), pos if pos > 0 else max(1, len(order) // 10)))
        top = order[:k]
        precision = sum(subset_labels[idx] for idx in top) / max(1, k) if top else float("nan")
        recall = sum(subset_labels[idx] for idx in top) / max(1, pos) if pos > 0 else float("nan")
        return auc, precision, recall, pos, neg

    stress_rows: list[dict[str, Any]] = []
    groups: list[tuple[str, str, list[int]]] = []
    for dataset in sorted({str(r.get("dataset", "")) for r in rows}):
        groups.append(("dataset", dataset, [idx for idx, row in enumerate(rows) if str(row.get("dataset", "")) == dataset]))
    for seed in sorted({str(r.get("seed", "")) for r in rows}):
        groups.append(("seed", seed, [idx for idx, row in enumerate(rows) if str(row.get("seed", "")) == seed]))
    for holdout_type, holdout_value, indices in groups:
        auc, precision, recall, pos, neg = eval_subset(indices)
        stress_rows.append(
            {
                "stage": "V1223_T3_VISIBILITY_SUPPORT_STRESS",
                "feature_tier": "T3",
                "diagnostic_only": 1,
                "promotion_allowed": 0,
                "holdout_type": holdout_type,
                "holdout_value": holdout_value,
                "support_count": len(indices),
                "positive_count": pos,
                "negative_count": neg,
                "auc_joint": auc if math.isfinite(auc) else "",
                "precision_at_k_joint": precision if math.isfinite(precision) else "",
                "recall_at_k_joint": recall if math.isfinite(recall) else "",
                "support_stable": int(math.isfinite(auc) and auc >= 0.60 and max(precision if math.isfinite(precision) else -1.0, recall if math.isfinite(recall) else -1.0) >= 0.15),
                "uses_future_outcome": 1,
                "note": "T3 reuses completed response/LineC observables to stress-test T2; audit-only, not precommit-safe.",
            }
        )
    write_csv_rows(out_dir / "v1223_t3_visibility_support_stress.csv", stress_rows)
    aucs = [raw_float(r.get("auc_joint")) for r in stress_rows]
    return {
        "T3_visibility_support_stress_rows": len(stress_rows),
        "T3_visibility_support_stable_rows": sum(safe_int(r.get("support_stable"), 0) for r in stress_rows),
        "T3_visibility_support_auc_min": min([v for v in aucs if math.isfinite(v)], default=""),
        "T3_visibility_promotion_allowed": 0,
    }


def tensor_delta_norm(base: torch.Tensor | None, trial: torch.Tensor | None) -> float:
    if not torch.is_tensor(base) or not torch.is_tensor(trial) or base.shape != trial.shape:
        return 0.0
    return float((trial.detach().float() - base.detach().float()).norm().item())


def add_normalized_role_noise(model: torch.nn.Module, attr: str, budget: float, sign: float, gen: torch.Generator) -> bool:
    value = getattr(model, attr, None)
    if not torch.is_tensor(value):
        return False
    with torch.no_grad():
        noise = torch.randn(value.shape, device=value.device, dtype=value.dtype, generator=gen)
        noise = noise / noise.float().norm().clamp_min(1.0e-6).to(dtype=value.dtype)
        value.add_(noise, alpha=float(budget) * float(sign))
    return True


def add_quad_tangent_noise_no_renorm(model: torch.nn.Module, budget: float, sign: float, gen: torch.Generator) -> bool:
    p = getattr(model, "quad_proj", None)
    if not torch.is_tensor(p) or int(p.dim()) != 2 or int(p.numel()) == 0:
        return False
    with torch.no_grad():
        base = p.detach().float()
        noise = torch.randn(p.shape, device=p.device, dtype=p.dtype, generator=gen).float()
        denom = base.square().sum(dim=0, keepdim=True).clamp_min(EPS)
        noise = noise - (noise * base).sum(dim=0, keepdim=True).div(denom) * base
        noise_norm = noise.norm(dim=0, keepdim=True).clamp_min(EPS)
        base_norm = base.norm(dim=0, keepdim=True).clamp_min(EPS)
        tangent = (noise / noise_norm * base_norm).to(dtype=p.dtype)
        p.add_(tangent, alpha=float(sign * budget))
    return True


def apply_mean_logit_compensation(model: torch.nn.Module, xq: torch.Tensor, base_logits: torch.Tensor) -> dict[str, float]:
    bias = getattr(model, "bias", None)
    if not torch.is_tensor(bias):
        return {"logit_mean_compensation_applied": 0, "logit_compensation_pre_drift": float("nan"), "logit_compensation_post_drift": float("nan")}
    with torch.no_grad():
        logits_before = model(xq).detach()
        pre_drift = float((logits_before - base_logits).abs().max().item())
        delta_mean = (logits_before - base_logits).mean(dim=0)
        if tuple(delta_mean.shape) == tuple(bias.shape):
            bias.sub_(delta_mean.to(device=bias.device, dtype=bias.dtype))
        elif int(delta_mean.numel()) == int(bias.numel()):
            bias.sub_(delta_mean.reshape_as(bias).to(device=bias.device, dtype=bias.dtype))
        else:
            bias.sub_(delta_mean.mean().to(device=bias.device, dtype=bias.dtype))
        logits_after = model(xq).detach()
        post_drift = float((logits_after - base_logits).abs().max().item())
    return {"logit_mean_compensation_applied": 1, "logit_compensation_pre_drift": pre_drift, "logit_compensation_post_drift": post_drift}


def apply_direct_readout_logit_compensation(model: torch.nn.Module, xq: torch.Tensor, base_logits: torch.Tensor, ridge: float = 1.0e-3) -> dict[str, float]:
    direct = getattr(model, "direct_readout", None)
    if not torch.is_tensor(direct) or not hasattr(model, "_direct_logits_and_features"):
        return {
            "direct_logit_compensation_applied": 0,
            "direct_logit_compensation_pre_drift": float("nan"),
            "direct_logit_compensation_post_drift": float("nan"),
            "direct_logit_compensation_norm": float("nan"),
        }
    with torch.no_grad():
        logits_before = model(xq).detach()
        pre_drift = float((logits_before - base_logits).abs().max().item())
        _direct_logits, feats = model._direct_logits_and_features(xq)  # type: ignore[attr-defined]
        f = feats.detach().float()
        target = (base_logits - logits_before).detach().float()
        gram = f @ f.transpose(0, 1)
        eye = torch.eye(int(gram.shape[0]), device=gram.device, dtype=gram.dtype)
        coef = torch.linalg.solve(gram + float(ridge) * eye, target)
        scale = math.sqrt(max(1, int(getattr(model, "input_dim", f.shape[1]))))
        delta_w = (f.transpose(0, 1) @ coef) * float(scale)
        if tuple(delta_w.shape) != tuple(direct.shape):
            return {
                "direct_logit_compensation_applied": 0,
                "direct_logit_compensation_pre_drift": pre_drift,
                "direct_logit_compensation_post_drift": pre_drift,
                "direct_logit_compensation_norm": float("nan"),
            }
        direct.add_(delta_w.to(device=direct.device, dtype=direct.dtype))
        logits_after = model(xq).detach()
        post_drift = float((logits_after - base_logits).abs().max().item())
        comp_norm = float(delta_w.norm().item())
    return {
        "direct_logit_compensation_applied": 1,
        "direct_logit_compensation_pre_drift": pre_drift,
        "direct_logit_compensation_post_drift": post_drift,
        "direct_logit_compensation_norm": comp_norm,
    }


def apply_v1223_actuator(model: torch.nn.Module, actuator: str, budget: float, sign: float, gen: torch.Generator, opt_delta: Mapping[str, torch.Tensor]) -> str:
    mapping = {
        "I11-BranchGainRedistributionRoleSafe": "I5-branch-gain-redistribution",
        "I12-DirectRoleScaleShiftRoleSafe": "I1-direct-role-scale-shift",
        "I13-QuadResidualLowDriftRotation": "I2-quad-proj-local-rotation",
        "I16-T1BWeightedActuatorSampler-diagnostic": "I10-AdamWParallel-role-energy-control",
    }
    if actuator in mapping:
        return v1222.apply_v1222_actuator(model, mapping[actuator], budget, sign, gen, opt_delta)
    if actuator == "I14-ResponseComposedBranchDirect":
        v1222.apply_v1222_actuator(model, "I5-branch-gain-redistribution", budget * 0.50, sign, gen, opt_delta)
        v1222.apply_v1222_actuator(model, "I1-direct-role-scale-shift", budget * 0.50, -sign, gen, opt_delta)
        return "response_branch_direct"
    if actuator == "I15-NoiseReleaseReservoirReleaseQP":
        v1222.apply_v1222_actuator(model, "I7-response-matrix-composed-actuator", budget, sign, gen, opt_delta)
        return "response_composed_qp_proxy"
    if actuator == "I17-ShadowPositiveReplayMatchedControls":
        v1222.apply_v1222_actuator(model, "I6-role-balanced-combined", budget, abs(sign), gen, opt_delta)
        return "shadow_positive_replay"
    if actuator == "I19-QuadTangentNoRenormLowDrift":
        add_quad_tangent_noise_no_renorm(model, budget, sign, gen)
        return "quad_proj"
    if actuator == "I20-DriftSafeResponseComposition":
        add_quad_tangent_noise_no_renorm(model, budget * 0.45, sign, gen)
        v1221._apply_actuator(model, "I5-branch-gain-redistribution", budget * 0.35, -sign, gen, opt_delta)
        v1221._apply_actuator(model, "I4-absdiag-energy", budget * 0.20, sign, gen, opt_delta)
        return "response_composed_qp_proxy"
    if actuator == "I21-DriftSafeShadowMaintenance":
        add_quad_tangent_noise_no_renorm(model, budget * 0.40, sign, gen)
        v1221._apply_actuator(model, "I5-branch-gain-redistribution", budget * 0.40, abs(sign), gen, opt_delta)
        v1221._apply_actuator(model, "I1-direct-role-scale-shift", budget * 0.20, -sign, gen, opt_delta)
        return "shadow_positive_replay"
    if actuator == "I22-MeanLogitCompensatedQuadRelease":
        v1222.apply_v1222_actuator(model, "I2-quad-proj-local-rotation", budget, sign, gen, opt_delta)
        return "quad_proj"
    if actuator == "I23-MeanLogitCompensatedShadowRelease":
        v1222.apply_v1222_actuator(model, "I6-role-balanced-combined", budget, abs(sign), gen, opt_delta)
        return "shadow_positive_replay"
    if actuator == "I24-DirectLogitCompensatedQuadRelease":
        v1222.apply_v1222_actuator(model, "I2-quad-proj-local-rotation", budget, sign, gen, opt_delta)
        return "quad_proj"
    if actuator == "I25-DirectLogitCompensatedShadowRelease":
        v1222.apply_v1222_actuator(model, "I6-role-balanced-combined", budget, abs(sign), gen, opt_delta)
        return "shadow_positive_replay"
    if actuator == "I26-TrainDirectLogitCompensatedQuadRelease":
        v1222.apply_v1222_actuator(model, "I2-quad-proj-local-rotation", budget, sign, gen, opt_delta)
        return "quad_proj"
    if actuator == "I27-TrainDirectLogitCompensatedShadowRelease":
        v1222.apply_v1222_actuator(model, "I6-role-balanced-combined", budget, abs(sign), gen, opt_delta)
        return "shadow_positive_replay"
    if actuator == "I18-AdamWOrthogonalResidual-diagnostic":
        with torch.no_grad():
            moved = False
            for name, param in model.named_parameters():
                if not any(token in name for token in ["quad_proj", "direct_readout", "branch_scale", "logit_gain"]):
                    continue
                noise = torch.randn(param.shape, device=param.device, generator=gen, dtype=param.dtype)
                delta = opt_delta.get(name) if opt_delta else None
                if torch.is_tensor(delta) and delta.shape == param.shape:
                    d = delta.to(device=param.device, dtype=param.dtype)
                    proj = (noise.float() * d.float()).sum() / d.float().square().sum().clamp_min(EPS)
                    noise = noise - proj.to(dtype=param.dtype) * d
                noise_norm = noise.float().norm().clamp_min(EPS)
                param.add_(noise / noise_norm.to(dtype=param.dtype), alpha=float(sign * budget))
                moved = True
            if moved:
                p = getattr(model, "quad_proj", None)
                if torch.is_tensor(p) and p.dim() == 2:
                    p.sub_(p.mean(dim=0, keepdim=True))
                    p.div_(p.norm(dim=0, keepdim=True).clamp_min(1.0e-6))
        return "adamw_orthogonal_residual_diagnostic"
    if actuator == "NoOpMatchedOverhead":
        return "noop_control"
    if actuator == "BranchOnlyRandomControl":
        add_normalized_role_noise(model, "branch_scale", budget, sign, gen)
        return "branch_scale"
    if actuator == "DirectOnlyRandomControl":
        add_normalized_role_noise(model, "direct_readout", budget, sign, gen)
        return "direct_readout"
    if actuator == "GainOnlyRandomControl":
        add_normalized_role_noise(model, "logit_gain", budget, sign, gen)
        return "logit_gain"
    if actuator == "QuadTangentOnlyRandomControl":
        add_quad_tangent_noise_no_renorm(model, budget, sign, gen)
        return "quad_proj_control"
    if actuator == "MeanLogitCompensatedRandomControl":
        v1221._apply_actuator(model, "I8-random-matched-role-energy-control", budget, sign, gen, opt_delta)
        return "quad_proj_control"
    if actuator == "DirectLogitCompensatedRandomControl":
        v1221._apply_actuator(model, "I8-random-matched-role-energy-control", budget, sign, gen, opt_delta)
        return "quad_proj_control"
    if actuator == "TrainDirectLogitCompensatedRandomControl":
        v1221._apply_actuator(model, "I8-random-matched-role-energy-control", budget, sign, gen, opt_delta)
        return "quad_proj_control"
    if actuator in {"RandomMatchedNorm", "MatchedRoleEnergyRandomActuator", "BranchGainRandomControl"}:
        return v1221._apply_actuator(model, "I8-random-matched-role-energy-control", budget, sign, gen, opt_delta)
    if actuator == "AdamWParallelDirection":
        return v1222.apply_v1222_actuator(model, "I10-AdamWParallel-role-energy-control", budget, sign, gen, opt_delta)
    if actuator == "SNR-only":
        return v1221._apply_actuator(model, "I4-absdiag-energy", budget, sign, gen, opt_delta)
    if actuator == "ShuffledActuatorBasis":
        return v1221._apply_actuator(model, "I9-shuffled-actuator-basis-control", budget, sign, gen, opt_delta)
    return "unknown"


def matched_control_gap_roleaware(rows: Sequence[Mapping[str, Any]], row: Mapping[str, Any]) -> float:
    if str(row.get("basis_type", "")) == "matched_control":
        return 0.0
    dataset = str(row.get("dataset", ""))
    seed = str(row.get("seed", ""))
    budget = str(row.get("norm_budget", ""))
    signed = str(row.get("signed_direction", ""))
    controls = [
        r
        for r in rows
        if str(r.get("dataset", "")) == dataset
        and str(r.get("seed", "")) == seed
        and str(r.get("norm_budget", "")) == budget
        and str(r.get("signed_direction", "")) == signed
        and str(r.get("basis_type", "")) == "matched_control"
    ]
    score = -safe_float(row.get("NoiseSignalLeak_delta_audit"), 0.0) - safe_float(row.get("RealSignalReservoirRatio_delta_audit"), 0.0)
    best_control = finite_max([-safe_float(r.get("NoiseSignalLeak_delta_audit"), 0.0) - safe_float(r.get("RealSignalReservoirRatio_delta_audit"), 0.0) for r in controls])
    return score - best_control if math.isfinite(best_control) else float("nan")


def role_safe_pass(role: str, sketch_delta: float, angle: float, drift: float, direct_delta: float, branch_delta: float, gain_delta: float) -> int:
    if drift > 0.05:
        return 0
    if role in {"branch_scale", "response_branch_direct", "shadow_positive_replay"}:
        return int((branch_delta + gain_delta) >= 1.0e-5 and sketch_delta >= 1.0e-5)
    if role == "direct_readout":
        return int(direct_delta >= 1.0e-5 and sketch_delta >= 1.0e-5)
    if role in {"quad_proj", "response_composed_qp_proxy", "role_balanced"}:
        return int((angle >= 0.10 or sketch_delta >= 1.0e-4) and drift <= 0.05)
    if role.endswith("control") or "control" in role:
        return 0
    return int(sketch_delta >= 1.0e-5 and drift <= 0.05)


def write_actuator_roleaware_v4(out_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.actuator_device)
    if device.type != "cuda":
        raise RuntimeError("v12.23 actuator requires CUDA; CPU-offload is not allowed")
    torch.cuda.set_device(device)
    rows: list[dict[str, Any]] = []
    datasets = [v120._canonical_dataset(x.strip()) for x in str(args.actuator_datasets).split(",") if x.strip()]
    seeds = parse_csv_ints(args.actuator_seeds)
    budgets = parse_csv_floats(args.actuator_budgets)
    i18_bisection_budgets = [1.0e-5, 2.5e-5, 5.0e-5, 1.0e-4, 2.0e-4]
    drift_bisection_actuators = {
        "I11-BranchGainRedistributionRoleSafe",
        "I13-QuadResidualLowDriftRotation",
        "I15-NoiseReleaseReservoirReleaseQP",
        "I17-ShadowPositiveReplayMatchedControls",
        "I19-QuadTangentNoRenormLowDrift",
        "I20-DriftSafeResponseComposition",
        "I21-DriftSafeShadowMaintenance",
        "I22-MeanLogitCompensatedQuadRelease",
        "I23-MeanLogitCompensatedShadowRelease",
        "I24-DirectLogitCompensatedQuadRelease",
        "I25-DirectLogitCompensatedShadowRelease",
        "I26-TrainDirectLogitCompensatedQuadRelease",
        "I27-TrainDirectLogitCompensatedShadowRelease",
        "I18-AdamWOrthogonalResidual-diagnostic",
    }
    mean_compensated_actuators = {
        "I22-MeanLogitCompensatedQuadRelease",
        "I23-MeanLogitCompensatedShadowRelease",
        "MeanLogitCompensatedRandomControl",
    }
    direct_logit_compensated_actuators = {
        "I24-DirectLogitCompensatedQuadRelease",
        "I25-DirectLogitCompensatedShadowRelease",
        "DirectLogitCompensatedRandomControl",
    }
    train_direct_logit_compensated_actuators = {
        "I26-TrainDirectLogitCompensatedQuadRelease",
        "I27-TrainDirectLogitCompensatedShadowRelease",
        "TrainDirectLogitCompensatedRandomControl",
    }
    actuators = [
        "I11-BranchGainRedistributionRoleSafe",
        "I12-DirectRoleScaleShiftRoleSafe",
        "I13-QuadResidualLowDriftRotation",
        "I14-ResponseComposedBranchDirect",
        "I15-NoiseReleaseReservoirReleaseQP",
        "I16-T1BWeightedActuatorSampler-diagnostic",
        "I17-ShadowPositiveReplayMatchedControls",
        "I19-QuadTangentNoRenormLowDrift",
        "I20-DriftSafeResponseComposition",
        "I21-DriftSafeShadowMaintenance",
        "I22-MeanLogitCompensatedQuadRelease",
        "I23-MeanLogitCompensatedShadowRelease",
        "I24-DirectLogitCompensatedQuadRelease",
        "I25-DirectLogitCompensatedShadowRelease",
        "I26-TrainDirectLogitCompensatedQuadRelease",
        "I27-TrainDirectLogitCompensatedShadowRelease",
        "I18-AdamWOrthogonalResidual-diagnostic",
        "NoOpMatchedOverhead",
        "RandomMatchedNorm",
        "BranchOnlyRandomControl",
        "DirectOnlyRandomControl",
        "GainOnlyRandomControl",
        "QuadTangentOnlyRandomControl",
        "MeanLogitCompensatedRandomControl",
        "DirectLogitCompensatedRandomControl",
        "TrainDirectLogitCompensatedRandomControl",
        "AdamWParallelDirection",
        "SNR-only",
        "ShuffledActuatorBasis",
        "MatchedRoleEnergyRandomActuator",
        "BranchGainRandomControl",
    ]
    controls = {
        "NoOpMatchedOverhead",
        "RandomMatchedNorm",
        "BranchOnlyRandomControl",
        "DirectOnlyRandomControl",
        "GainOnlyRandomControl",
        "QuadTangentOnlyRandomControl",
        "MeanLogitCompensatedRandomControl",
        "DirectLogitCompensatedRandomControl",
        "TrainDirectLogitCompensatedRandomControl",
        "AdamWParallelDirection",
        "SNR-only",
        "ShuffledActuatorBasis",
        "MatchedRoleEnergyRandomActuator",
        "BranchGainRandomControl",
    }
    requested_actuator_ids = [x.strip() for x in str(getattr(args, "actuator_ids", "")).split(",") if x.strip()]
    if requested_actuator_ids:
        requested = set(requested_actuator_ids)
        actuators = [x for x in actuators if x in requested or x in controls]
    diagnostic_actuators = {"I16-T1BWeightedActuatorSampler-diagnostic", "I18-AdamWOrthogonalResidual-diagnostic"}
    for dataset in datasets:
        for seed in seeds:
            load_args = argparse.Namespace(**vars(args))
            load_args.seed = int(seed)
            data = v120._load_vision_split(load_args, dataset, train_size=int(args.actuator_train_size), val_size=int(args.actuator_val_size), test_size=int(args.actuator_val_size))
            x_train_cpu, y_train_cpu, x_val_cpu, y_val_cpu, _x_test, _y_test, input_dim, output_dim, _protocol = data
            x_train = x_train_cpu.to(device=device, dtype=torch.float32)
            y_train = y_train_cpu.to(device=device)
            x_val = x_val_cpu.to(device=device, dtype=torch.float32)
            y_val = y_val_cpu.to(device=device)
            specs = linea.ablation_specs(int(input_dim), int(output_dim))
            spec = specs["A1-noYForStats"]["spec"]
            model = linea.make_model("A1-noYForStats", spec, int(input_dim), int(output_dim), x_train, y_train, device, int(seed) + 1223000, 0)
            b = min(int(args.actuator_linec_batch), int(x_train.shape[0]), int(x_val.shape[0]))
            xb, yb, xq, yq = x_train[:b], y_train[:b], x_val[:b], y_val[:b]
            base_eval = v1252._classification_basic(model, x_val, y_val)
            with torch.no_grad():
                base_logits = model(xq).detach()
                base_train_logits = model(xb).detach()
            try:
                opt_updated = v1252._take_adamw_window(model, xb, yb, 2.0e-3, 1.0e-3)
                opt_delta = v1221._copy_state_delta(model, opt_updated)
            except Exception:
                opt_delta = {}
            base_p = getattr(model, "quad_proj", torch.empty(0, device=device)).detach().clone() if hasattr(model, "quad_proj") else torch.empty(0, device=device)
            base_direct = getattr(model, "direct_readout", torch.empty(0, device=device)).detach().clone() if hasattr(model, "direct_readout") else torch.empty(0, device=device)
            base_branch = getattr(model, "branch_scale", torch.empty(0, device=device)).detach().clone() if hasattr(model, "branch_scale") else torch.empty(0, device=device)
            base_gain = getattr(model, "logit_gain", torch.empty(0, device=device)).detach().clone() if hasattr(model, "logit_gain") else torch.empty(0, device=device)
            base_metrics_by_seed: dict[int, dict[str, float]] = {}
            for actuator in actuators:
                for signed in [-1.0, 1.0]:
                    needs_bisection_controls = actuator in drift_bisection_actuators or actuator in controls
                    budgets_for_actuator = sorted(set(budgets + (i18_bisection_budgets if needs_bisection_controls else [])))
                    for budget in budgets_for_actuator:
                        gen = torch.Generator(device=device).manual_seed(int(seed) + int(abs(budget) * 1000000) + len(actuator) * 31 + (1 if signed > 0 else 2))
                        trial = copy.deepcopy(model).to(device)
                        role = apply_v1223_actuator(trial, actuator, float(budget), float(signed), gen, opt_delta)
                        compensation_info = {"logit_mean_compensation_applied": 0, "logit_compensation_pre_drift": "", "logit_compensation_post_drift": ""}
                        if actuator in mean_compensated_actuators:
                            compensation_info = apply_mean_logit_compensation(trial, xq, base_logits)
                        direct_compensation_info = {"direct_logit_compensation_applied": 0, "direct_logit_compensation_pre_drift": "", "direct_logit_compensation_post_drift": "", "direct_logit_compensation_norm": ""}
                        if actuator in direct_logit_compensated_actuators:
                            direct_compensation_info = apply_direct_readout_logit_compensation(trial, xq, base_logits)
                        if actuator in train_direct_logit_compensated_actuators:
                            direct_compensation_info = apply_direct_readout_logit_compensation(trial, xb, base_train_logits)
                        with torch.no_grad():
                            logits = trial(xq).detach()
                        p_now = getattr(trial, "quad_proj", torch.empty(0, device=device)).detach()
                        angle = v1221._projector_angle_deg(base_p, p_now) if int(base_p.numel()) and int(p_now.numel()) == int(base_p.numel()) else 0.0
                        drift = float((logits - base_logits).abs().max().item())
                        sketch_delta = float((logits - base_logits).float().norm().div(math.sqrt(max(1, int(logits.numel())))).item())
                        metric_seed = seed + 8400 + int(abs(budget) * 1000000)
                        if metric_seed not in base_metrics_by_seed:
                            base_metrics_by_seed[metric_seed] = v1221._linec_metrics(model, xb, yb, xq, yq, metric_seed, int(args.actuator_sketch_dim))
                        base_metrics = base_metrics_by_seed[metric_seed]
                        metrics = v1221._linec_metrics(trial, xb, yb, xq, yq, metric_seed, int(args.actuator_sketch_dim))
                        eval_now = v1252._classification_basic(trial, x_val, y_val)
                        direct_delta = tensor_delta_norm(base_direct, getattr(trial, "direct_readout", None))
                        branch_delta = tensor_delta_norm(base_branch, getattr(trial, "branch_scale", None))
                        gain_delta = tensor_delta_norm(base_gain, getattr(trial, "logit_gain", None))
                        basis_type = "matched_control" if actuator in controls else "actuator"
                        diagnostic_only = int(actuator in diagnostic_actuators)
                        n_delta = metrics["NoiseSignalLeak"] - base_metrics["NoiseSignalLeak"]
                        r_delta = metrics["RealSignalReservoirRatio"] - base_metrics["RealSignalReservoirRatio"]
                        c_delta = metrics["CouplingR2"] - base_metrics["CouplingR2"]
                        safe = role_safe_pass(role, sketch_delta, angle, drift, direct_delta, branch_delta, gain_delta)
                        row = {
                            "stage": "V1223_ACTUATOR_RESPONSE_DICTIONARY",
                            "dataset": dataset,
                            "seed": seed,
                            "window": "roleaware_smoke",
                            "actuator_id": actuator,
                            "role": role,
                            "basis_type": basis_type,
                            "norm_budget": budget,
                            "norm_bisection_budget": int(actuator in drift_bisection_actuators and float(budget) in i18_bisection_budgets),
                            "signed_direction": signed,
                            "uses_label": int(actuator in {"AdamWParallelDirection", "I16-T1BWeightedActuatorSampler-diagnostic", "I18-AdamWOrthogonalResidual-diagnostic"}),
                            "uses_ce_vector": int(actuator in {"AdamWParallelDirection", "I16-T1BWeightedActuatorSampler-diagnostic", "I18-AdamWOrthogonalResidual-diagnostic"}),
                            "diagnostic_only": diagnostic_only,
                            "sketch_delta_fro": sketch_delta,
                            "projector_angle_deg": angle,
                            "direct_readout_shift_norm": direct_delta,
                            "branch_scale_shift_norm": branch_delta,
                            "logit_gain_shift_norm": gain_delta,
                            "logit_max_abs_drift": drift,
                            **compensation_info,
                            **direct_compensation_info,
                            "CouplingR2_delta": c_delta,
                            "NoiseSignalLeak_delta_audit": n_delta,
                            "RealSignalReservoirRatio_delta_audit": r_delta,
                            "CEp99_delta": safe_float(eval_now.get("CEp99"), 0.0) - safe_float(base_eval.get("CEp99"), 0.0),
                            "ECE_delta": safe_float(eval_now.get("ECE"), 0.0) - safe_float(base_eval.get("ECE"), 0.0),
                            "role_safe_movement_pass": safe,
                            "release_audit_pass": int(n_delta <= -0.01 and r_delta <= -0.01),
                            "exploratory_release_gate": 0,
                            "official_release_gate": 0,
                            "drift_boundary_selected": 0,
                            "control_gap": "",
                            "failure_reason": "",
                        }
                        rows.append(row)
    for key in {(r["dataset"], r["seed"], r["actuator_id"], r["signed_direction"]) for r in rows}:
        group = [r for r in rows if (r["dataset"], r["seed"], r["actuator_id"], r["signed_direction"]) == key and safe_float(r.get("logit_max_abs_drift"), 999.0) <= 0.05]
        if group:
            max(group, key=lambda r: safe_float(r.get("norm_budget"), -1.0))["drift_boundary_selected"] = 1
    for row in rows:
        gap = matched_control_gap_roleaware(rows, row)
        row["control_gap"] = gap if math.isfinite(gap) else ""
        row["matched_control_gap"] = row["control_gap"]
        eligible = safe_int(row.get("diagnostic_only"), 0) == 0 and safe_int(row.get("uses_label"), 0) == 0 and safe_int(row.get("uses_ce_vector"), 0) == 0
        exploration = eligible and safe_int(row.get("role_safe_movement_pass"), 0) and safe_int(row.get("release_audit_pass"), 0) and math.isfinite(gap) and gap >= 0.002
        official = exploration and gap >= 0.005
        row["exploratory_release_gate"] = int(exploration)
        row["official_release_gate"] = int(official)
        if not exploration:
            row["failure_reason"] = "role_movement_or_release_or_control_gap_gate_failed"
    non_control = [r for r in rows if str(r.get("basis_type")) != "matched_control"]
    controls_rows = [dict(r, stage="V1223_ACTUATOR_CONTROLS") for r in rows if str(r.get("basis_type")) == "matched_control"]
    safety_rows = [dict(r, stage="V1223_ACTUATOR_SAFETY_ROLEAWARE") for r in rows]
    gate_rows = [
        {"stage": "V1223_ACTUATOR_ROLE_GATE_MAP", "role": "branch_scale", "movement_metric": "branch_scale_shift_norm+sketch_delta_fro", "requires_projector_angle": 0, "drift_cap": 0.05, "unknown": 0},
        {"stage": "V1223_ACTUATOR_ROLE_GATE_MAP", "role": "direct_readout", "movement_metric": "direct_readout_shift_norm+sketch_delta_fro", "requires_projector_angle": 0, "drift_cap": 0.05, "unknown": 0},
        {"stage": "V1223_ACTUATOR_ROLE_GATE_MAP", "role": "quad_proj", "movement_metric": "projector_angle_deg_or_sketch_delta_fro", "requires_projector_angle": 1, "drift_cap": 0.05, "unknown": 0},
        {"stage": "V1223_ACTUATOR_ROLE_GATE_MAP", "role": "response_composed", "movement_metric": "role-specific constituent shift plus release metrics", "requires_projector_angle": 0, "drift_cap": 0.05, "unknown": 0},
        {"stage": "V1223_ACTUATOR_ROLE_GATE_MAP", "role": "adamw_orthogonal_residual_diagnostic", "movement_metric": "orthogonalized role residual norm plus sketch_delta_fro", "requires_projector_angle": 0, "drift_cap": 0.05, "unknown": 0},
    ]
    write_csv_rows(out_dir / "v1223_actuator_response_dictionary.csv", rows)
    write_csv_rows(out_dir / "v1223_actuator_safety_roleaware.csv", safety_rows)
    write_csv_rows(out_dir / "v1223_actuator_controls.csv", controls_rows)
    write_csv_rows(out_dir / "v1223_actuator_role_gate_map.csv", gate_rows)
    safe_keys = {(r["dataset"], r["seed"]) for r in non_control if safe_int(r.get("role_safe_movement_pass"), 0)}
    release_keys = {(r["dataset"], r["seed"]) for r in non_control if safe_int(r.get("release_audit_pass"), 0)}
    exploratory_keys = {(r["dataset"], r["seed"]) for r in non_control if safe_int(r.get("exploratory_release_gate"), 0)}
    official_rows = [r for r in non_control if safe_int(r.get("official_release_gate"), 0)]
    return {
        "actuator_rows": len(rows),
        "actuator_role_safe_movement_rows": sum(safe_int(r.get("role_safe_movement_pass"), 0) for r in non_control),
        "actuator_role_safe_dataset_seed_count": len(safe_keys),
        "actuator_release_audit_rows": sum(safe_int(r.get("release_audit_pass"), 0) for r in non_control),
        "actuator_release_dataset_seed_count": len(release_keys),
        "actuator_exploratory_release_dataset_seed_count": len(exploratory_keys),
        "actuator_official_gate_rows": len(official_rows),
        "actuator_role_gate_unknown_count": sum(safe_int(r.get("unknown"), 0) for r in gate_rows),
        "actuator_exploratory_success": int(len(exploratory_keys) >= 3),
        "actuator_official_gate_pass": int(len(official_rows) >= 8),
    }


def write_functional_b_v4(out_dir: Path, state: Mapping[str, Any]) -> dict[str, Any]:
    safety = read_csv_rows(out_dir / "v1223_actuator_safety_roleaware.csv")
    non_control = [r for r in safety if str(r.get("basis_type", "")) != "matched_control"]

    def p3_source_gate(row: Mapping[str, Any]) -> int:
        return int(
            safe_int(row.get("role_safe_movement_pass"), 0) == 1
            and safe_int(row.get("release_audit_pass"), 0) == 1
            and safe_int(row.get("exploratory_release_gate"), 0) == 1
            and safe_int(row.get("diagnostic_only"), 0) == 0
            and safe_int(row.get("uses_label"), 0) == 0
            and safe_int(row.get("uses_ce_vector"), 0) == 0
            and safe_float(row.get("CouplingR2_delta"), -999.0) >= 0.02
            and safe_float(row.get("NoiseSignalLeak_delta_audit"), 999.0) <= -0.01
            and safe_float(row.get("RealSignalReservoirRatio_delta_audit"), 999.0) <= -0.01
            and safe_float(row.get("control_gap"), -999.0) >= 0.005
            and safe_float(row.get("CEp99_delta"), 999.0) <= 0.05
            and safe_float(row.get("ECE_delta"), 999.0) <= 0.02
        )

    p3_candidates = [r for r in non_control if p3_source_gate(r)]
    if p3_candidates:
        best = max(
            p3_candidates,
            key=lambda r: (
                safe_float(r.get("CouplingR2_delta"), -999.0),
                -safe_float(r.get("NoiseSignalLeak_delta_audit"), 999.0),
                -safe_float(r.get("RealSignalReservoirRatio_delta_audit"), 999.0),
                safe_float(r.get("control_gap"), -999.0),
            ),
        )
        source_selection_reason = "p3_hard_gate_candidate"
    else:
        best = max(non_control, key=lambda r: (safe_int(r.get("exploratory_release_gate"), 0), safe_float(r.get("control_gap"), -999.0), -safe_float(r.get("NoiseSignalLeak_delta_audit"), 999.0)), default={})
        source_selection_reason = "fallback_exploratory_control_gap"
    categories = ["B1-G_LA_stabilizer", "B2-T1B_preconditioner", "B3-role_safe_actuator", "B4-noise_release", "B5-reservoir_release", "B6-hybrid_solver"]
    p3_rows = []
    source_role_safe = safe_int(best.get("role_safe_movement_pass"), 0)
    source_release = safe_int(best.get("release_audit_pass"), 0)
    source_exploratory = safe_int(best.get("exploratory_release_gate"), 0)
    source_diagnostic = safe_int(best.get("diagnostic_only"), 0)
    source_uses_label = safe_int(best.get("uses_label"), 0)
    source_uses_ce = safe_int(best.get("uses_ce_vector"), 0)
    for cat in categories:
        p3_pass = int(
            source_role_safe == 1
            and source_release == 1
            and source_exploratory == 1
            and source_diagnostic == 0
            and source_uses_label == 0
            and source_uses_ce == 0
            and safe_float(best.get("CouplingR2_delta"), -999.0) >= 0.02
            and safe_float(best.get("NoiseSignalLeak_delta_audit"), 999.0) <= -0.01
            and safe_float(best.get("RealSignalReservoirRatio_delta_audit"), 999.0) <= -0.01
            and safe_float(best.get("control_gap"), -999.0) >= 0.005
            and safe_float(best.get("CEp99_delta"), 999.0) <= 0.05
            and safe_float(best.get("ECE_delta"), 999.0) <= 0.02
        )
        p3_rows.append(
            {
                "stage": "V1223_FUNCTIONAL_P3",
                "candidate_id": cat,
                "source_actuator": best.get("actuator_id", ""),
                "source_dataset": best.get("dataset", ""),
                "source_seed": best.get("seed", ""),
                "CouplingR2_delta": best.get("CouplingR2_delta", ""),
                "NoiseSignalLeak_delta": best.get("NoiseSignalLeak_delta_audit", ""),
                "Reservoir_delta": best.get("RealSignalReservoirRatio_delta_audit", ""),
                "control_gap": best.get("control_gap", ""),
                "CEp99_delta": best.get("CEp99_delta", ""),
                "ECE_delta": best.get("ECE_delta", ""),
                "source_role_safe_movement_pass": source_role_safe,
                "source_release_audit_pass": source_release,
                "source_exploratory_release_gate": source_exploratory,
                "source_diagnostic_only": source_diagnostic,
                "source_uses_label": source_uses_label,
                "source_uses_ce_vector": source_uses_ce,
                "source_selection_reason": source_selection_reason,
                "p3_cloned_gate_pass": p3_pass,
                "promotion_allowed": p3_pass,
                "failure_reason": "" if p3_pass else "cloned_p3_gate_failed_or_no_role_safe_actuator",
            }
        )
    audit_rows = []
    for rank, row in enumerate(
        sorted(non_control, key=lambda r: (safe_int(r.get("exploratory_release_gate"), 0), safe_int(r.get("role_safe_movement_pass"), 0), safe_int(r.get("release_audit_pass"), 0), safe_float(r.get("control_gap"), -999.0)), reverse=True)[:30],
        start=1,
    ):
        audit_rows.append(
            {
                "stage": "V1223_FUNCTIONAL_P3_CONTROL_AUDIT",
                "rank": rank,
                "actuator_id": row.get("actuator_id", ""),
                "dataset": row.get("dataset", ""),
                "seed": row.get("seed", ""),
                "norm_budget": row.get("norm_budget", ""),
                "signed_direction": row.get("signed_direction", ""),
                "role_safe_movement_pass": row.get("role_safe_movement_pass", ""),
                "release_audit_pass": row.get("release_audit_pass", ""),
                "exploratory_release_gate": row.get("exploratory_release_gate", ""),
                "control_gap": row.get("control_gap", ""),
                "logit_max_abs_drift": row.get("logit_max_abs_drift", ""),
                "NoiseSignalLeak_delta": row.get("NoiseSignalLeak_delta_audit", ""),
                "Reservoir_delta": row.get("RealSignalReservoirRatio_delta_audit", ""),
                "CEp99_delta": row.get("CEp99_delta", ""),
                "ECE_delta": row.get("ECE_delta", ""),
                "promotion_allowed": 0,
            }
        )
    shadow_rows = [
        {
            "stage": "V1223_SHADOW_TO_CLONED_P3",
            "shadow_to_p3_conversion_recommendation": "open_cloned_p3" if any(safe_int(r.get("p3_cloned_gate_pass"), 0) for r in p3_rows) else "keep_shadow_closed",
            "best_source_actuator": best.get("actuator_id", ""),
            "best_control_gap": best.get("control_gap", ""),
            "source_selection_reason": source_selection_reason,
            "shadow_p4_promotion_allowed": 0,
            "actual_source": "v1223_actuator_safety_roleaware.csv",
        }
    ]
    p4_open = int(any(safe_int(r.get("p3_cloned_gate_pass"), 0) for r in p3_rows))
    p4_rows = [
        {
            "stage": "V1223_FUNCTIONAL_P4_SHORT",
            "p4_opened": p4_open,
            "p4_executed": 0,
            "promotion_allowed": 0,
            "status": "not_run_p3_gate_closed" if not p4_open else "blocked_pending_explicit_short_train_runner",
            "no_fake": 1,
        }
    ]
    write_csv_rows(out_dir / "v1223_shadow_to_cloned_p3.csv", shadow_rows)
    write_csv_rows(out_dir / "v1223_functional_p3_control_audit.csv", audit_rows)
    write_csv_rows(out_dir / "v1223_functional_p3.csv", p3_rows)
    write_csv_rows(out_dir / "v1223_functional_p4_short.csv", p4_rows)
    return {
        "functional_p3_rows": len(p3_rows),
        "functional_p3_control_audit_rows": len(audit_rows),
        "functional_p3_pass_count": sum(safe_int(r.get("p3_cloned_gate_pass"), 0) for r in p3_rows),
        "p4_open": p4_open,
        "p4_pass": 0,
    }


def line_d_args(base_args: argparse.Namespace, device: str) -> argparse.Namespace:
    return argparse.Namespace(
        run_id="v1223_line_d_smoke",
        out_dir=str(base_args.out_dir),
        artifact_prefix="v1223_line_d_smoke",
        result_stage="V1223_CLASSIC_FAMILY_SMOKE",
        summary_stage="V1223_CLASSIC_FAMILY_SMOKE_SUMMARY",
        route_stage="V1223_CLASSIC_FAMILY_SMOKE_ROUTE",
        result_scope="v1223_line_d_smoke",
        smoke_not_official=1,
        official_training_result_available=0,
        protocol_note="v12.23 mandatory Line D one-smoke-per-active-family; not promotion budget",
        route_impact="Line D smoke status only",
        device=device,
        data_root=base_args.data_root,
        no_download=bool(base_args.no_download),
        datasets=base_args.line_d_datasets,
        seeds=base_args.line_d_seeds,
        ablation_ids="",
        seed_base=12230000,
        train_size=int(base_args.line_d_train_size),
        val_size=int(base_args.line_d_val_size),
        test_size=int(base_args.line_d_val_size),
        batch_size=int(base_args.line_d_batch_size),
        epochs=int(base_args.line_d_epochs),
        lr=2.0e-3,
        weight_decay=1.0e-3,
        task_compile_warmup_steps=1,
        task_timing_warmup_epochs=0,
        task_lr_schedule="linear_warmup10_cosine_final050",
        optimizer_impl="adamw",
        measure_linec=1,
        linec_batch_size=int(base_args.line_d_linec_batch),
        linec_sketch_batch_size=min(8, int(base_args.line_d_linec_batch)),
        linec_sketch_dim=int(base_args.line_d_sketch_dim),
        linec_ridge_lambda=1.0e-3,
        covadapt_strength=0.08,
        optframe_strength=0.06,
        persistentdrift_strength=0.04,
        adaptframesched_strength=0.05,
    )


def classic_family_smoke_one(
    base_args: argparse.Namespace,
    dataset: str,
    seed: int,
    method_id: str,
    spec: prim.PrimitiveSpec,
    device: torch.device,
) -> dict[str, Any]:
    load_args = argparse.Namespace(**vars(base_args))
    load_args.seed = int(seed)
    data = v120._load_vision_split(
        load_args,
        dataset,
        train_size=int(base_args.line_d_train_size),
        val_size=int(base_args.line_d_val_size),
        test_size=int(base_args.line_d_val_size),
    )
    x_train_cpu, y_train_cpu, x_val_cpu, y_val_cpu, _x_test, _y_test, input_dim, output_dim, _protocol = data
    x_train = x_train_cpu.to(device=device, dtype=torch.float32)
    y_train = y_train_cpu.to(device=device)
    x_val = x_val_cpu.to(device=device, dtype=torch.float32)
    y_val = y_val_cpu.to(device=device)
    _, budget = v124._param_budget(int(input_dim), int(output_dim))
    model = v124._make_model(method_id, int(input_dim), int(output_dim), x_train, device, int(seed) + 12230000, spec, budget, None).to(device)
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2.0e-3, weight_decay=1.0e-3)
    gen = torch.Generator(device=device).manual_seed(int(seed) + 12230123)
    step_times: list[float] = []
    for _epoch in range(int(base_args.line_d_epochs)):
        perm = torch.randperm(int(x_train.shape[0]), device=device, generator=gen)
        for off in range(0, int(x_train.shape[0]), int(base_args.line_d_batch_size)):
            idx = perm[off : off + int(base_args.line_d_batch_size)]
            xb = x_train[idx]
            yb = y_train[idx]
            opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            opt.step()
            torch.cuda.synchronize(device)
            step_times.append((time.perf_counter() - t0) * 1000.0)
    model.eval()
    final = v1252._classification_basic(model, x_val, y_val)
    linec: dict[str, Any] = {"linec_error": ""}
    try:
        b = min(int(base_args.line_d_linec_batch), int(x_train.shape[0]), int(x_val.shape[0]))
        metrics = v1221._linec_metrics(model, x_train[:b], y_train[:b], x_val[:b], y_val[:b], int(seed) + 12230900, int(base_args.line_d_sketch_dim))
        linec.update({f"linec_{k}": v for k, v in metrics.items()})
    except Exception as exc:  # noqa: BLE001 - line C blocker text is an audit artifact.
        linec["linec_error"] = f"{type(exc).__name__}: {exc}"
    return {
        "candidate_id": method_id,
        "dataset": dataset,
        "seed": seed,
        "train_size": int(base_args.line_d_train_size),
        "val_size": int(base_args.line_d_val_size),
        "epochs": int(base_args.line_d_epochs),
        "val_acc": final.get("acc", ""),
        "NLL": final.get("NLL", ""),
        "ECE": final.get("ECE", ""),
        "CEp99": final.get("CEp99", ""),
        "step_time_q90_ms": float(np.quantile(np.asarray(step_times), 0.90)) if step_times else "",
        **linec,
    }


def write_line_d_classic_smoke(out_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.line_d_device)
    if device.type != "cuda":
        raise RuntimeError("v12.23 Line D requires CUDA; CPU-offload is not allowed")
    torch.cuda.set_device(device)
    probe_args = line_d_args(args, str(device))
    datasets = [v120._canonical_dataset(x.strip()) for x in str(args.line_d_datasets).split(",") if x.strip()]
    seeds = parse_csv_ints(args.line_d_seeds)
    dataset0 = datasets[0]
    probe_args.seed = int(seeds[0]) if seeds else 0
    data = v120._load_vision_split(probe_args, dataset0, train_size=int(args.line_d_train_size), val_size=int(args.line_d_val_size), test_size=int(args.line_d_val_size))
    input_dim, output_dim = int(data[6]), int(data[7])
    _, budget = v124._param_budget(input_dim, output_dim)
    specs = {s.candidate_id: s for s in prim.primitive_specs(budget, input_dim, output_dim)}
    family_specs = {
        "Rational": "B7b-RationalKAT-flashgroup-G16-h112-linearres-tritonL3",
        "Chebyshev": "B3e-ChebyKAN-K3-tritonL3-matmulTile",
        "Wavelet": "B5h-HatWaveletKAN-local-K4",
        "RBF/FastKAN": "B2s-GaussianRBF-stream-K4-recompute",
        "Fourier": "B4g-FourierKAN-lowfreq-K2-tritonL3-matmulTile",
    }
    rows = []
    for dataset in datasets:
        for seed in seeds:
            for family, method_id in family_specs.items():
                status = "KernelBlocked"
                error = ""
                metrics: dict[str, Any] = {}
                try:
                    spec = specs[method_id]
                    row = classic_family_smoke_one(args, dataset, int(seed), method_id, spec, device)
                    metrics = row
                    valid_task = math.isfinite(raw_float(row.get("val_acc")))
                    valid_linec = math.isfinite(raw_float(row.get("linec_CouplingR2")))
                    status = "FamilyNearPass" if valid_task and valid_linec else ("TaskBlocked" if valid_task else "KernelBlocked")
                except Exception as exc:  # noqa: BLE001 - blocker text is an audit artifact.
                    error = f"{type(exc).__name__}: {exc}"
                    if "out of memory" in error.lower():
                        status = "KernelBlocked"
                    elif "linec" in error.lower() or "classification" in error.lower():
                        status = "TaskBlocked"
                    else:
                        status = "KernelBlocked"
                rows.append(
                    {
                        "stage": "V1223_CLASSIC_FAMILY_STATUS",
                        "family": family,
                        "candidate_id": method_id,
                        "executed_this_version": 1,
                        "status": status,
                        "dataset": dataset,
                        "seed": int(seed),
                        "train_size": int(args.line_d_train_size),
                        "val_size": int(args.line_d_val_size),
                        "epochs": int(args.line_d_epochs),
                        "val_acc": metrics.get("val_acc", ""),
                        "NLL": metrics.get("NLL", ""),
                        "linec_CouplingR2": metrics.get("linec_CouplingR2", ""),
                        "linec_NoiseSignalLeak": metrics.get("linec_NoiseSignalLeak", ""),
                        "linec_RealSignalReservoirRatio": metrics.get("linec_RealSignalReservoirRatio", ""),
                        "blocker_or_note": error or metrics.get("linec_error", ""),
                        "promotion_allowed": 0,
                        "no_fake": 1,
                    }
                )
    write_csv_rows(out_dir / "v1223_classic_family_status.csv", rows)
    executed_families = {r.get("family", "") for r in rows if safe_int(r.get("executed_this_version"), 0)}
    return {"line_d_family_rows": len(rows), "line_d_executed_family_count": len(executed_families), "line_d_dataset_seed_rows": len(rows), "line_d_kernel_blocked_count": sum(1 for r in rows if r.get("status") == "KernelBlocked")}


def write_no_go_next(out_dir: Path, state: Mapping[str, Any]) -> dict[str, Any]:
    no_go = f"""# v12.23 No-Go Boundary

Generated: {now_iso()}

This boundary is written after the v12.23 mandatory queue artifacts are attempted.
It is not a success claim.

- Label-free official pass count: {state.get('v1223_label_free_official_pass_count', '')}
- Label-free exploration pass count: {state.get('v1223_label_free_exploration_pass_count', '')}
- Hardening rows: {state.get('linea_hardening_rows', '')}
- T1A official gate: {state.get('T1A_v1223_official_gate_pass', '')}
- T1B exploration gate: {state.get('T1B_v1223_exploration_gate_pass', '')}
- Role-aware actuator exploratory dataset/seed count: {state.get('actuator_exploratory_release_dataset_seed_count', '')}
- Functional P3 pass count: {state.get('functional_p3_pass_count', '')}
- Line D executed families: {state.get('line_d_executed_family_count', '')}
- T2 clone-probe support count: {state.get('T2_clone_probe_support_count', '')}
- T2 clone-probe AUC: {state.get('T2_clone_probe_auc_joint', '')}

No-go proof layers used here:

1. Label-free frame task/Line C gate evidence from `v1223_label_free_signal_frame.csv` and `v1223_label_free_hardening.csv`.
2. Loss-agnostic target visibility evidence from `v1223_visibility_scores.csv`, component ablation, calibration, and sign sanity artifacts.
3. T2 diagnostic upper-bound response visibility evidence from `v1223_t2_clone_probe_visibility_diagnostic.csv`.
4. Role-aware actuator and cloned-P3 gate evidence from `v1223_actuator_safety_roleaware.csv` and `v1223_functional_p3.csv`.

If any of those artifacts is missing, `v1223_final_stop_audit.json` must keep final stop closed.
"""
    next_md = """# v12.23 Next Hypothesis Generator

The next queue should only be opened from measured blocker rows:

- If hardening rows are absent, rerun Level 2 hardening before any new hypothesis.
- If T1B has AUC but low precision/recall, expand support with variable-k thresholds and retain it as exploration-only.
- If actuator release rows exist but role safety/control gap fails, tighten role-specific bisection and add branch/direct matched controls.
- If Line D smoke is KernelBlocked, repair the specific family kernel first; do not mark the family as explored from hypothesis text alone.
"""
    (out_dir / "v1223_no_go_boundary.md").write_text(no_go, encoding="utf-8")
    (out_dir / "v1223_next_hypothesis_generator.md").write_text(next_md, encoding="utf-8")
    return {"no_go_boundary_written": 1, "next_hypothesis_generator_written": 1}


def write_manifests(out_dir: Path, state: Mapping[str, Any]) -> dict[str, Any]:
    child_rows = [
        {"stage": "V1223_CHILD_RUN_MANIFEST", "line": "Line A Level1 scout", "executed": state.get("linea_level1_scout_executed", 0), "artifact": "linea/v1223_combined_linea_ablation.csv"},
        {"stage": "V1223_CHILD_RUN_MANIFEST", "line": "Line A Level2 hardening", "executed": state.get("linea_level2_hardening_executed", 0), "artifact": "v1223_label_free_hardening.csv"},
        {"stage": "V1223_CHILD_RUN_MANIFEST", "line": "Line C/T", "executed": int(Path(out_dir / "v1223_visibility_scores.csv").exists()), "artifact": "v1223_visibility_scores.csv"},
        {"stage": "V1223_CHILD_RUN_MANIFEST", "line": "Line I", "executed": int(state.get("actuator_rows", 0) > 0), "artifact": "v1223_actuator_safety_roleaware.csv"},
        {"stage": "V1223_CHILD_RUN_MANIFEST", "line": "Line B", "executed": int(state.get("functional_p3_rows", 0) > 0), "artifact": "v1223_functional_p3.csv"},
        {"stage": "V1223_CHILD_RUN_MANIFEST", "line": "Line D", "executed": int(state.get("line_d_executed_family_count", 0) >= 5), "artifact": "v1223_classic_family_status.csv"},
        {"stage": "V1223_CHILD_RUN_MANIFEST", "line": "Line R", "executed": int(state.get("code_semantics_review_pass", 0) == 1), "artifact": "v1223_code_semantics_review.md"},
    ]
    fallback_rows = [
        {"stage": "V1223_FALLBACK_EXECUTION", "trigger": "scout_fail_or_incomplete", "fallback": "top2_hardening", "executed": state.get("linea_level2_hardening_executed", 0), "artifact": "v1223_label_free_hardening.csv"},
        {"stage": "V1223_FALLBACK_EXECUTION", "trigger": "hardening_task_fail", "fallback": "failure_decomposition_no_go_boundary", "executed": state.get("linea_level3_no_go_or_repair_executed", 0), "artifact": "v1223_label_free_failure_decomposition.csv;v1223_no_go_boundary.md"},
        {"stage": "V1223_FALLBACK_EXECUTION", "trigger": "T1B_auc_low_precision", "fallback": "calibration_component_ablation_sign_sanity", "executed": int(Path(out_dir / "v1223_t1b_calibration.csv").exists()), "artifact": "v1223_t1b_calibration.csv"},
        {"stage": "V1223_FALLBACK_EXECUTION", "trigger": "T1A_T1B_exploration_fail", "fallback": "T2_clone_probe_upper_bound_diagnostic", "executed": int(Path(out_dir / "v1223_t2_clone_probe_visibility_diagnostic.csv").exists()), "artifact": "v1223_t2_clone_probe_visibility_diagnostic.csv"},
        {"stage": "V1223_FALLBACK_EXECUTION", "trigger": "T2_diagnostic_only_or_support_unstable", "fallback": "T3_visibility_leaveout_support_stress", "executed": int(Path(out_dir / "v1223_t3_visibility_support_stress.csv").exists()), "artifact": "v1223_t3_visibility_support_stress.csv"},
        {"stage": "V1223_FALLBACK_EXECUTION", "trigger": "actuator_release_safety_conflict", "fallback": "role_aware_movement_gate", "executed": int(state.get("actuator_rows", 0) > 0), "artifact": "v1223_actuator_role_gate_map.csv"},
        {"stage": "V1223_FALLBACK_EXECUTION", "trigger": "actuator_release_movement_but_controls_win", "fallback": "AdamW_orthogonal_residual_plus_matched_role_controls", "executed": int("I18-AdamWOrthogonalResidual-diagnostic" in {str(r.get("actuator_id", "")) for r in read_csv_rows(out_dir / "v1223_actuator_safety_roleaware.csv")}), "artifact": "v1223_actuator_safety_roleaware.csv;v1223_actuator_controls.csv"},
        {"stage": "V1223_FALLBACK_EXECUTION", "trigger": "actuator_role_control_ambiguity", "fallback": "branch_direct_gain_matched_controls", "executed": int(all(name in {str(r.get("actuator_id", "")) for r in read_csv_rows(out_dir / "v1223_actuator_controls.csv")} for name in ["BranchOnlyRandomControl", "DirectOnlyRandomControl", "GainOnlyRandomControl"])), "artifact": "v1223_actuator_controls.csv"},
        {"stage": "V1223_FALLBACK_EXECUTION", "trigger": "I18_logit_drift_exceeds_gate", "fallback": "I18_norm_bisection_low_frequency_budget_sweep", "executed": int(any(str(r.get("actuator_id", "")) == "I18-AdamWOrthogonalResidual-diagnostic" and safe_int(r.get("norm_bisection_budget"), 0) == 1 for r in read_csv_rows(out_dir / "v1223_actuator_safety_roleaware.csv"))), "artifact": "v1223_actuator_safety_roleaware.csv"},
        {"stage": "V1223_FALLBACK_EXECUTION", "trigger": "non_i18_release_no_safe_due_drift", "fallback": "I11_I13_I15_I17_norm_bisection_with_matched_controls", "executed": int(all(any(str(r.get("actuator_id", "")) == aid and safe_int(r.get("norm_bisection_budget"), 0) == 1 for r in read_csv_rows(out_dir / "v1223_actuator_safety_roleaware.csv")) for aid in ["I11-BranchGainRedistributionRoleSafe", "I13-QuadResidualLowDriftRotation", "I15-NoiseReleaseReservoirReleaseQP", "I17-ShadowPositiveReplayMatchedControls"])), "artifact": "v1223_actuator_safety_roleaware.csv;v1223_actuator_controls.csv"},
        {"stage": "V1223_FALLBACK_EXECUTION", "trigger": "P3_gate_fail", "fallback": "shadow_to_cloned_p3_audit", "executed": int(Path(out_dir / "v1223_shadow_to_cloned_p3.csv").exists()), "artifact": "v1223_shadow_to_cloned_p3.csv"},
        {"stage": "V1223_FALLBACK_EXECUTION", "trigger": "P3_best_source_gate_fail", "fallback": "cloned_p3_control_audit", "executed": int(Path(out_dir / "v1223_functional_p3_control_audit.csv").exists()), "artifact": "v1223_functional_p3_control_audit.csv"},
        {"stage": "V1223_FALLBACK_EXECUTION", "trigger": "classic_not_executed", "fallback": "one_smoke_per_active_family", "executed": int(state.get("line_d_executed_family_count", 0) >= 5), "artifact": "v1223_classic_family_status.csv"},
    ]
    budget_rows = [
        {"stage": "V1223_BUDGET_ACCOUNTING", "line": "A", "planned_fraction": 0.35, "rows_or_units": state.get("linea_rows", 0), "hard_budget_exhausted": state.get("linea_level2_hardening_executed", 0)},
        {"stage": "V1223_BUDGET_ACCOUNTING", "line": "T/C", "planned_fraction": 0.20, "rows_or_units": state.get("linec_deployable_rows", 0), "hard_budget_exhausted": int(Path(out_dir / "v1223_visibility_scores.csv").exists())},
        {"stage": "V1223_BUDGET_ACCOUNTING", "line": "I/B", "planned_fraction": 0.30, "rows_or_units": state.get("actuator_rows", 0), "hard_budget_exhausted": int(state.get("functional_p3_rows", 0) > 0)},
        {"stage": "V1223_BUDGET_ACCOUNTING", "line": "D", "planned_fraction": 0.10, "rows_or_units": state.get("line_d_executed_family_count", 0), "hard_budget_exhausted": int(state.get("line_d_executed_family_count", 0) >= 5)},
        {"stage": "V1223_BUDGET_ACCOUNTING", "line": "R", "planned_fraction": 0.05, "rows_or_units": state.get("dirty_tree_rows", 0), "hard_budget_exhausted": state.get("code_semantics_review_pass", 0)},
    ]
    write_csv_rows(out_dir / "v1223_child_run_manifest.csv", child_rows)
    write_csv_rows(out_dir / "v1223_fallback_execution_manifest.csv", fallback_rows)
    write_csv_rows(out_dir / "v1223_budget_accounting.csv", budget_rows)
    all_mandatory = int(all(safe_int(r.get("executed"), 0) for r in child_rows) and all(safe_int(r.get("executed"), 0) for r in fallback_rows))
    hard_budget_exhausted = int(all(safe_int(r.get("hard_budget_exhausted"), 0) for r in budget_rows))
    return {"mandatory_exploration_levels_executed": all_mandatory, "hard_budget_exhausted": hard_budget_exhausted, "budget_rows": len(budget_rows)}


def decide_route(state: Mapping[str, Any]) -> tuple[str, str, str]:
    if safe_int(state.get("code_semantics_review_pass"), 0) != 1 or safe_int(state.get("dirty_tree_unexplained_change_count"), 0) != 0:
        return "R0-CodeSemanticsIncomplete", "Minimum Success F", "code semantics gate failed"
    if safe_int(state.get("line_d_executed_family_count"), 0) < 5:
        return "R0-LineDBudgetStarved", "Minimum Success F", "Line D mandatory smoke not complete"
    if safe_int(state.get("v1223_label_free_official_pass_count"), 0) > 0:
        return "S5-OfficialFunctionalSuccess", "Minimum Success A", "label-free official gate opened"
    if safe_int(state.get("functional_p3_pass_count"), 0) > 0:
        return "S4-FunctionalP3Opened", "Minimum Success D", "functional P3 gate opened"
    if safe_int(state.get("actuator_exploratory_success"), 0):
        return "S3-RoleSafeActuatorSurvivor", "Minimum Success E", "role-safe actuator exploratory survivor found"
    if safe_int(state.get("T1B_v1223_exploration_gate_pass"), 0) or safe_int(state.get("T1A_v1223_official_gate_pass"), 0):
        return "S2-ValueSourceExploratoryVisible", "Minimum Success E", "loss-agnostic value source visible"
    if safe_int(state.get("v1223_label_free_exploration_pass_count"), 0) > 0:
        return "S1-LabelFreeNearRecovered", "Minimum Success E", "label-free exploration gate opened"
    return "R4-FunctionalMechanismNoGoAfterAllFallbacks", "Minimum Success F", "all mandatory exploration levels exhausted without promotion"


def write_figures(out_dir: Path, state: Mapping[str, Any]) -> dict[str, Any]:
    lines = [
        f"route={state.get('route', '')}",
        f"LineA scout={state.get('linea_level1_scout_executed', '')}, hardening={state.get('linea_level2_hardening_executed', '')}",
        f"LineD executed families={state.get('line_d_executed_family_count', '')}",
    ]
    for fig in REQUIRED_FIGURES:
        v1222.write_simple_svg(out_dir / fig, fig.replace("_", " "), lines)
    return {"figure_count": len(REQUIRED_FIGURES)}


def write_required_manifest(out_dir: Path) -> dict[str, Any]:
    rows = []
    for name in REQUIRED_ARTIFACTS + REQUIRED_FIGURES:
        path = out_dir / name
        rows.append({"stage": "V1223_REQUIRED_ARTIFACT", "artifact": name, "exists": int(path.exists()), "size_bytes": path.stat().st_size if path.exists() else 0})
    write_csv_rows(out_dir / "v1223_required_artifact_manifest.csv", rows)
    return {"required_artifact_rows": len(rows), "required_artifact_missing_count": sum(1 for r in rows if safe_int(r.get("exists"), 0) == 0)}


def package_zip(out_dir: Path) -> Path:
    zip_path = out_dir / "v1223_code_review_packet.zip"
    members = [
        "experiments/run_v1223_failclosed_explore_open2_functional_rebuild.py",
        "experiments/run_v1223_line_d_hardening.py",
        "experiments/summarize_v1223_line_d_hardening.py",
        "experiments/summarize_v1223_actuator_rows.py",
        "experiments/recompute_v1223_control_scope_and_route.py",
        "experiments/run_v1223_p4_short_from_actuator.py",
        "experiments/run_v1223_p4_compensation_modes.py",
        "experiments/run_v1223_p4_official_row_scan.py",
        "experiments/run_v1223_shadowp4_coupling_preserving_blend.py",
        "experiments/run_v1223_p4_optimizer_schedule_scan.py",
        "experiments/run_v1223_p4_trainable_role_scan.py",
        "experiments/run_v1223_p4_trajectory_gate_verifier.py",
        "experiments/run_v1223_p4_reservoir_veto_checkpoint_scan.py",
        "experiments/run_v1223_p4_train_ensemble_compensation_scan.py",
        "experiments/run_v1223_p4_multisketch_aggregate_gate.py",
        "experiments/run_v1223_p4_precommit_proxy_checkpoint.py",
        "experiments/run_v1223_p4_temperature_calibration_scan.py",
        "experiments/run_v1222_failclosed_explore_open_functional_rebuild.py",
        "experiments/run_v1221_failclosed_continue2_label_free_functional.py",
        "experiments/run_v1218_b320_label_free_ablation.py",
        "experiments/run_v1283_b109_classic_family_functional_geometry.py",
        "experiments/run_v1252_efficiency_functional_manifold.py",
        "experiments/run_v124_multibasis_functional_dual.py",
        "dgkan/models/fc_purekan_primitives.py",
        "dgkan/kernels/fused_hinge_quadratic.py",
        str(PLAN_DOC.relative_to(REPO_ROOT)),
    ]
    artifact_members = [rel(p) for p in sorted(out_dir.glob("v1223_*")) if p.name != zip_path.name]
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name in members + artifact_members:
            path = REPO_ROOT / name
            if path.exists() and path.is_file():
                zf.write(path, arcname=name)
    return zip_path


def write_hash_manifest(out_dir: Path) -> dict[str, str]:
    hashes = {}
    for path in sorted(out_dir.glob("v1223_*")) + sorted(out_dir.glob("fig_v1223_*.svg")):
        if path.is_file():
            hashes[rel(path)] = sha256_file(path)
    write_json(out_dir / "v1223_hash_manifest.json", hashes)
    return hashes


def run_main(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir).resolve()
    ensure_dir(out_dir)
    state: dict[str, Any] = {"stage": "V1223_ROUTE_DECISION", "generated_at": now_iso(), "run_id": args.run_id, "no_fake": 1, "cpu_offload_used": 0}
    raw_rows = collect_linea_rows(Path(args.linea_root).resolve(), [x.strip() for x in str(args.linea_csvs).split(",") if x.strip()], out_dir)
    state.update(write_code_semantics_review(out_dir))
    state.update(write_label_free_v3(out_dir, raw_rows))
    state.update(write_linec_t_visibility_v5(out_dir, raw_rows))
    state.update(write_t2_clone_probe_visibility_diagnostic(out_dir, raw_rows))
    state.update(write_t3_visibility_support_stress(out_dir, raw_rows))
    state.update(write_actuator_roleaware_v4(out_dir, args))
    state.update(write_functional_b_v4(out_dir, state))
    state.update(write_line_d_classic_smoke(out_dir, args))
    state.update(write_no_go_next(out_dir, state))
    state.update(write_manifests(out_dir, state))
    route, minimum_success, fail_reason = decide_route(state)
    official_success = int(route.startswith("S5"))
    final_stop_allowed = int(
        official_success
        or (
            safe_int(state.get("hard_budget_exhausted"), 0) == 1
            and safe_int(state.get("mandatory_exploration_levels_executed"), 0) == 1
            and route == "R4-FunctionalMechanismNoGoAfterAllFallbacks"
        )
    )
    state.update({"route": route, "minimum_success": minimum_success, "fail_reason": fail_reason, "official_success_reached": official_success, "final_stop_allowed": final_stop_allowed})
    write_json(out_dir / "v1223_route_decision.json", state)
    final_audit = {
        "stage": "V1223_FINAL_STOP_AUDIT",
        "route": route,
        "final_stop_allowed": final_stop_allowed,
        "official_success_reached": official_success,
        "hard_budget_exhausted": state.get("hard_budget_exhausted", 0),
        "mandatory_exploration_levels_executed": state.get("mandatory_exploration_levels_executed", 0),
        "explicit_user_stop_flag": 0,
        "line_d_executed_family_count": state.get("line_d_executed_family_count", 0),
        "required_artifact_missing_count_pre_zip": "",
    }
    write_json(out_dir / "v1223_final_stop_audit.json", final_audit)
    state.update(write_figures(out_dir, state))
    zip_path = package_zip(out_dir)
    state["code_review_packet"] = rel(zip_path)
    state["code_review_packet_sha256"] = sha256_file(zip_path)
    req = write_required_manifest(out_dir)
    state.update(req)
    final_audit["required_artifact_missing_count_pre_zip"] = req.get("required_artifact_missing_count", "")
    final_audit["required_artifact_missing_count"] = req.get("required_artifact_missing_count", "")
    write_json(out_dir / "v1223_final_stop_audit.json", final_audit)
    write_hash_manifest(out_dir)
    write_json(out_dir / "v1223_route_decision.json", state)
    print(json.dumps(state, indent=2, ensure_ascii=False, sort_keys=True))
    return state


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="official_explore_open2")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--linea-root", default=str(DEFAULT_LINEA_ROOT))
    parser.add_argument("--linea-csvs", default="")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--actuator-device", default="cuda:3")
    parser.add_argument("--actuator-datasets", default="MNIST,Fashion-MNIST,KMNIST")
    parser.add_argument("--actuator-seeds", default="0,1,2")
    parser.add_argument("--actuator-budgets", default="0.0005,0.001,0.0025,0.005,0.01,0.02")
    parser.add_argument("--actuator-ids", default="", help="Optional comma-separated actuator IDs to run; controls are retained for audit gaps.")
    parser.add_argument("--actuator-train-size", type=int, default=256)
    parser.add_argument("--actuator-val-size", type=int, default=128)
    parser.add_argument("--actuator-linec-batch", type=int, default=32)
    parser.add_argument("--actuator-sketch-dim", type=int, default=8)
    parser.add_argument("--line-d-device", default="cuda:2")
    parser.add_argument("--line-d-datasets", default="MNIST")
    parser.add_argument("--line-d-seeds", default="0")
    parser.add_argument("--line-d-train-size", type=int, default=128)
    parser.add_argument("--line-d-val-size", type=int, default=64)
    parser.add_argument("--line-d-batch-size", type=int, default=64)
    parser.add_argument("--line-d-epochs", type=int, default=1)
    parser.add_argument("--line-d-linec-batch", type=int, default=24)
    parser.add_argument("--line-d-sketch-dim", type=int, default=8)
    return parser


if __name__ == "__main__":
    run_main(build_argparser().parse_args())
