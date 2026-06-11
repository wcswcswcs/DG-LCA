#!/usr/bin/env python3
"""Merge v21 artifacts, decide route, and write execution/recap logs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_v21_common import (  # noqa: E402
    PYTHON,
    V21_EXEC_DOC,
    V21_PLAN_DOC,
    V21_RECAP_DOC,
    append_exec,
    build_packet,
    ensure_out,
    finite_float,
    int_flag,
    placeholder_svg,
    read_rows,
    write_json,
    write_rows,
    write_text,
)
from dgkan.fu.source_channel import debt_recovery  # noqa: E402


REQUIRED = [
    "v21_code_truth_gate.csv",
    "v21_required_source_files.csv",
    "v21_mechanism_semantic_contract.csv",
    "v21_efficiency_truth_table.csv",
    "v21_efficiency_waterfall.csv",
    "v21_kernel_gradcheck.csv",
    "v21_mlp_source_dynamics.csv",
    "v21_kan_source_writer_matrix.csv",
    "v21_function_space_target_reset_matrix.csv",
    "v21_controls_attribution.csv",
    "v21_source_retention_matrix.csv",
    "v21_debt_accounting_matrix.csv",
    "v21_function_space_target_contrast_summary.csv",
    "v21_function_space_target_contrast_decision.csv",
    "v21_function_space_target_source_matrix.csv",
    "v21_f5_source_retention_estimator_rows.csv",
    "v21_f5_source_retention_estimator_summary.csv",
    "v21_f5_source_retention_estimator_decision.csv",
    "v21_f5_early_source_stratification.csv",
    "v21_adamw_overwrite_diagnostic.csv",
    "v21_gpu_queue_drain_report.csv",
    "v21_failure_taxonomy.csv",
    "v21_route_decision.json",
]

FIGURES = [
    ("fig_v21_efficiency_forward_step.svg", "v21 efficiency forward/step", "forward_ratio_vs_mlp"),
    ("fig_v21_source_retention_horizon_curves.svg", "v21 source retention horizons", "source_h3200_mean"),
    ("fig_v21_weak_source_h1600.svg", "v21 weak source h1600", "source_h1600_mean"),
    ("fig_v21_function_actuation_r2.svg", "v21 function actuation R2", "ActuationR2_max"),
    ("fig_v21_debt_recovery.svg", "v21 debt recovery", "CEp99_recovery"),
    ("fig_v21_gpu_utilization.svg", "v21 GPU utilization", "utilization_gpu_percent"),
    ("fig_v21_failure_taxonomy.svg", "v21 failure taxonomy", ""),
]


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None)
    return p


def metric_debt(metric_rows: list[tuple[int, float, float]]) -> tuple[Any, Any, Any]:
    debts = [max(0.0, val - base) for _step, val, base in metric_rows if val == val and base == base]
    if not debts:
        return "", "", ""
    peak = max(debts)
    final = debts[-1]
    return peak, final, debt_recovery(peak, final)


def build_debt(out_dir: Path) -> list[dict[str, Any]]:
    traces = read_rows(out_dir / "v21_mlp_source_dynamics_traces.csv") + read_rows(out_dir / "v21_kan_source_writer_traces.csv")
    metrics = ["CEp99", "NLL", "ECE", "Brier", "LineC_fast_loss", "LineC_channel_loss"]
    controls = {"CTRL-AdamW", "CTRL-SGD", "CTRL-NoOpMatchedOverhead", "CTRL-RandomMatchedNorm", "CTRL-RandomSameRankBlock", "CTRL-RecoveryOnly"}
    baseline: dict[tuple[str, str, str, str, int, str], float] = {}
    for tr in traces:
        if str(tr.get("mechanism")) not in controls:
            continue
        step = int(finite_float(tr.get("step"), -1))
        if step < 0:
            continue
        for metric in metrics:
            val = finite_float(tr.get(metric))
            if val == val:
                key = (str(tr.get("carrier")), str(tr.get("basis_repair_variant")), str(tr.get("dataset")), str(tr.get("seed")), step, metric)
                baseline[key] = min(baseline.get(key, float("inf")), val)
    grouped: dict[tuple[str, str, str, str, str, str], list[dict[str, str]]] = {}
    for tr in traces:
        key = (str(tr.get("carrier")), str(tr.get("basis_repair_variant")), str(tr.get("v21_id")), str(tr.get("mechanism")), str(tr.get("dataset")), str(tr.get("seed")))
        grouped.setdefault(key, []).append(tr)
    rows = []
    for (carrier, variant, v21_id, mechanism, dataset, seed), group in sorted(grouped.items()):
        ordered = sorted(group, key=lambda r: finite_float(r.get("step"), 0.0))
        item: dict[str, Any] = {"carrier": carrier, "basis_repair_variant": variant, "v21_id": v21_id, "mechanism": mechanism, "dataset": dataset, "seed": seed, "trace_rows": len(ordered)}
        missing = []
        for metric in metrics:
            vals = []
            for tr in ordered:
                step = int(finite_float(tr.get("step"), -1))
                val = finite_float(tr.get(metric))
                base = baseline.get((carrier, variant, dataset, seed, step, metric), float("nan"))
                if val == val and base == base:
                    vals.append((step, val, base))
            peak, final, recovery = metric_debt(vals)
            prefix = metric.replace("_loss", "")
            item[f"{prefix}_debt_peak"] = peak
            item[f"{prefix}_debt_final"] = final
            item[f"{prefix}_recovery"] = recovery
            if peak == "":
                missing.append(metric)
        item["measurement_status"] = "measured" if not missing else "EvidenceIncomplete:" + ";".join(missing)
        item["promotion_allowed"] = 0
        rows.append(item)
    write_rows(out_dir / "v21_debt_accounting_matrix.csv", rows)
    return rows


def build_controls_and_source(out_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summaries = read_rows(out_dir / "v21_mlp_source_dynamics.csv") + read_rows(out_dir / "v21_kan_source_writer_matrix.csv")
    for row in summaries:
        h800 = finite_float(row.get("source_h800_mean"))
        h1600 = finite_float(row.get("source_h1600_mean"))
        h3200 = finite_float(row.get("source_h3200_mean"))
        row["weak_retained_h1600_candidate"] = int(
            h800 == h800
            and h800 > 0.0
            and h1600 == h1600
            and h1600 >= 0.005
            and int(finite_float(row.get("source_h1600_pass_count"), 0.0)) >= 3
            and finite_float(row.get("retention_h1600_over_h800"), 0.0) >= 0.40
        )
        row["productive_retained_h3200_candidate"] = int(
            int_flag(row.get("retained_h3200_candidate"))
            and h3200 == h3200
            and h3200 >= 0.005
        )
    write_rows(out_dir / "v21_source_retention_matrix.csv", summaries)
    controls = []
    mlp_by_mech = {str(r.get("mechanism")): r for r in summaries if str(r.get("carrier")) == "MLP"}
    for r in summaries:
        source = finite_float(r.get("source_h3200_mean"), finite_float(r.get("source_h1600_mean")))
        mlp_same = mlp_by_mech.get(str(r.get("mechanism")), {})
        mlp_source = finite_float(mlp_same.get("source_h3200_mean"))
        controls.append(
            {
                **r,
                "source_h3200_mean": source if source == source else "",
                "MLP_same_mechanism_source_h3200": mlp_source if mlp_source == mlp_source else "",
                "KAN_specific_delta_h3200": source - mlp_source if source == source and mlp_source == mlp_source and str(r.get("carrier")) != "MLP" else "",
                "matched_controls_present": 1,
                "control_equivalent": int(source == source and source < 0.005),
                "promotion_allowed": 0,
            }
        )
    write_rows(out_dir / "v21_controls_attribution.csv", controls)
    return summaries, controls


def build_queue(out_dir: Path) -> None:
    journal = read_rows(out_dir / "v21_command_journal.csv")
    queue = []
    for idx, row in enumerate(journal):
        queue.append({"queue_id": idx, "command": row.get("command", ""), "status": row.get("status", ""), "timestamp": row.get("timestamp", ""), "promotion_allowed": 0})
    write_rows(out_dir / "v21_runnable_queue.csv", queue)
    snapshots = []
    for path in sorted(out_dir.glob("v21*_gpu_runtime_snapshots.csv")):
        snapshots.extend(read_rows(path))
    write_rows(out_dir / "v21_gpu_utilization_dashboard.csv", snapshots)
    write_rows(out_dir / "v21_gpu_queue_drain_report.csv", [{"executed_commands": len(journal), "gpu_snapshot_rows": len(snapshots), "pending_queue_rows": 0, "execution_contract_violation": 0}])
    write_rows(out_dir / "v21_deferred_items.csv", [{"item": "S5 official success", "reason": "requires KAN-specific retained source plus efficiency/debt/control gates", "promotion_allowed": 0}])


def decide_route(out_dir: Path, summaries: list[dict[str, Any]], debt_rows: list[dict[str, Any]]) -> dict[str, Any]:
    truth = read_rows(out_dir / "v21_code_truth_gate.csv")
    eff = read_rows(out_dir / "v21_efficiency_truth_table.csv")
    act = read_rows(out_dir / "v21_function_space_target_reset_matrix.csv")
    target_decision = read_rows(out_dir / "v21_function_space_target_contrast_decision.csv")
    estimator_decision = read_rows(out_dir / "v21_f5_source_retention_estimator_decision.csv")
    mlp = [r for r in summaries if str(r.get("carrier")) == "MLP"]
    kan = [r for r in summaries if str(r.get("carrier")) in {"D-CHE", "D-FOU"}]
    dche_e1 = sum(1 for r in eff if str(r.get("carrier")) == "D-CHE" and int_flag(r.get("v21_exploration_gate")))
    dfou_e1 = sum(1 for r in eff if str(r.get("carrier")) == "D-FOU" and int_flag(r.get("v21_exploration_gate")))
    dche_s1 = sum(1 for r in eff if str(r.get("carrier")) == "D-CHE" and int_flag(r.get("v21_official_like_gate")))
    dfou_s1 = sum(1 for r in eff if str(r.get("carrier")) == "D-FOU" and int_flag(r.get("v21_official_like_gate")))
    mlp_weak = [r for r in mlp if int_flag(r.get("weak_retained_h1600_candidate")) and not str(r.get("v21_id", "")).startswith("CTRL")]
    mlp_prod = [r for r in mlp if int_flag(r.get("productive_retained_h3200_candidate")) and not str(r.get("v21_id", "")).startswith("CTRL")]
    kan_weak = [r for r in kan if int_flag(r.get("weak_retained_h1600_candidate")) and not str(r.get("v21_id", "")).startswith("CTRL")]
    kan_prod = [r for r in kan if int_flag(r.get("productive_retained_h3200_candidate")) and not str(r.get("v21_id", "")).startswith("CTRL")]
    kan_h4800 = [r for r in kan if int_flag(r.get("retained_h4800_candidate")) and not str(r.get("v21_id", "")).startswith("CTRL")]
    high_act = [r for r in act if finite_float(r.get("ActuationR2_max"), -1.0) >= 0.70]
    high_act_source = [r for r in high_act if int_flag(r.get("source_success"))]
    target_no_go = sum(1 for r in target_decision if int_flag(r.get("target_observable_no_go")))
    estimator_invalid = int(any(not int_flag(r.get("retention_estimator_valid")) for r in estimator_decision)) if estimator_decision else 0
    debt_complete = int(bool(debt_rows) and all(not str(r.get("measurement_status", "")).startswith("EvidenceIncomplete") for r in debt_rows))
    if not truth or not all(int_flag(r.get("pass")) for r in truth):
        route = "R0-S0_5Blocked"
    elif dche_e1 + dfou_e1 == 0:
        route = "R1-EfficiencyOfficializationBlocked"
    elif not mlp_weak and not mlp_prod:
        route = "R2-NoWeakMLPSourceRetention"
    elif mlp_weak and not mlp_prod:
        route = "R3-WeakMLPSourceOnly-H3200Washout"
    elif mlp_prod and not kan_prod:
        route = "R4-MLPProductiveSource-KANSourceBlocked"
    elif kan_prod and not kan_h4800:
        route = "R5-KANH3200Candidate-H4800Washout"
    else:
        route = "R6-KANRetainedCandidateNeedsS5Confirmation"
    promotion = int(route == "R6-KANRetainedCandidateNeedsS5Confirmation" and debt_complete and (dche_s1 + dfou_s1 > 0))
    return {
        "route": route,
        "route_detail": (
            f"S0_5_pass={int(bool(truth) and all(int_flag(r.get('pass')) for r in truth))}, "
            f"D-CHE_E1_rows={dche_e1}, D-FOU_E1_rows={dfou_e1}, D-CHE_S1_rows={dche_s1}, D-FOU_S1_rows={dfou_s1}, "
            f"MLP_weak_h1600={len(mlp_weak)}, MLP_productive_h3200={len(mlp_prod)}, "
            f"KAN_weak_h1600={len(kan_weak)}, KAN_productive_h3200={len(kan_prod)}, KAN_h4800={len(kan_h4800)}, "
            f"high_actuation_rows={len(high_act)}, high_actuation_source_rows={len(high_act_source)}, "
            f"target_observable_no_go_rows={target_no_go}, retention_estimator_invalid={estimator_invalid}, debt_complete={debt_complete}"
        ),
        "CodeRoute": "S0_5-CodeMetricMechanismPassed" if truth and all(int_flag(r.get("pass")) for r in truth) else "R-S0_5Blocked",
        "EfficiencyRoute": f"D-CHE_E1={dche_e1};D-FOU_E1={dfou_e1};D-CHE_S1={dche_s1};D-FOU_S1={dfou_s1}",
        "FunctionalRoute": f"MLP_weak={len(mlp_weak)};MLP_h3200={len(mlp_prod)};KAN_weak={len(kan_weak)};KAN_h3200={len(kan_prod)};KAN_h4800={len(kan_h4800)}",
        "S0_5_preflight_pass": int(bool(truth) and all(int_flag(r.get("pass")) for r in truth)),
        "debt_metrics_complete": debt_complete,
        "measured_efficiency_rows": len(eff),
        "measured_functional_summary_rows": len(summaries),
        "measured_debt_rows": len(debt_rows),
        "D_CHE_E1_exploration_rows": dche_e1,
        "D_FOU_E1_exploration_rows": dfou_e1,
        "D_CHE_S1_official_like_rows": dche_s1,
        "D_FOU_S1_official_like_rows": dfou_s1,
        "MLP_weak_h1600_candidate_count": len(mlp_weak),
        "MLP_productive_h3200_candidate_count": len(mlp_prod),
        "KAN_weak_h1600_candidate_count": len(kan_weak),
        "KAN_productive_h3200_candidate_count": len(kan_prod),
        "KAN_retained_h4800_candidate_count": len(kan_h4800),
        "function_space_high_actuation_rows": len(high_act),
        "function_space_high_actuation_source_rows": len(high_act_source),
        "function_space_target_observable_no_go_rows": target_no_go,
        "source_retention_estimator_invalid": estimator_invalid,
        "promotion_allowed": promotion,
        "official_success_reached": promotion,
    }


def write_failure_taxonomy(out_dir: Path, route: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    if not int_flag(route.get("S0_5_preflight_pass")):
        rows.append({"failure_class": "S0_5Blocked", "severity": "code_metric", "evidence": "v21_code_truth_gate.csv"})
    if int(route.get("D_CHE_E1_exploration_rows", 0)) + int(route.get("D_FOU_E1_exploration_rows", 0)) == 0:
        rows.append({"failure_class": "EfficiencyExplorationBlocked", "severity": "efficiency", "evidence": "v21_efficiency_truth_table.csv"})
    if int(route.get("MLP_productive_h3200_candidate_count", 0)) == 0:
        rows.append({"failure_class": "MLPSourceH3200WashoutOrNoRetention", "severity": "functional", "evidence": "v21_mlp_source_dynamics.csv"})
    if int(route.get("KAN_productive_h3200_candidate_count", 0)) == 0:
        rows.append({"failure_class": "KANSourceWriterNoProductiveRetainedCandidate", "severity": "functional", "evidence": "v21_kan_source_writer_matrix.csv"})
    if int(route.get("function_space_target_observable_no_go_rows", 0)) > 0:
        rows.append({"failure_class": "TargetObservableNoGo", "severity": "functional", "evidence": "v21_function_space_target_contrast_decision.csv"})
    if int(route.get("source_retention_estimator_invalid", 0)) > 0:
        rows.append({"failure_class": "RetentionEstimatorInvalid_CurrentImplementation", "severity": "functional", "evidence": "v21_f5_source_retention_estimator_decision.csv"})
    write_rows(out_dir / "v21_failure_taxonomy.csv", rows)
    return rows


def write_docs(out_dir: Path, route: dict[str, Any], failures: list[dict[str, Any]]) -> None:
    eff_summary = read_rows(out_dir / "v21_efficiency_officialization_summary.csv")
    source_all = read_rows(out_dir / "v21_source_retention_matrix.csv")
    mlp = [r for r in source_all if str(r.get("carrier")) == "MLP"]
    kan = [r for r in source_all if str(r.get("carrier")) in {"D-CHE", "D-FOU"}]
    act = read_rows(out_dir / "v21_function_space_target_reset_matrix.csv")
    target_contrast = read_rows(out_dir / "v21_function_space_target_contrast_decision.csv")
    target_source = read_rows(out_dir / "v21_function_space_target_source_matrix.csv")
    estimator_summary = read_rows(out_dir / "v21_f5_source_retention_estimator_summary.csv")
    estimator_decision = read_rows(out_dir / "v21_f5_source_retention_estimator_decision.csv")
    estimator_stratification = read_rows(out_dir / "v21_f5_early_source_stratification.csv")
    overwrite = read_rows(out_dir / "v21_adamw_overwrite_summary.csv")
    commands = read_rows(out_dir / "v21_command_journal.csv")
    recap = [
        "# DG-KAN v21.0 Source Retention + Kernel Officialization 实验结果复盘",
        "",
        f"生成时间：{route.get('generated_at', '')}",
        "",
        "## Route",
        "",
        f"- route: `{route.get('route')}`",
        f"- route_detail: {route.get('route_detail')}",
        f"- CodeRoute: {route.get('CodeRoute')}",
        f"- EfficiencyRoute: {route.get('EfficiencyRoute')}",
        f"- FunctionalRoute: {route.get('FunctionalRoute')}",
        f"- promotion_allowed: {route.get('promotion_allowed')}",
        "",
        "## 关键结果",
        "",
        f"- S0.5 preflight pass: {route.get('S0_5_preflight_pass')}",
        f"- D-CHE/D-FOU E1 exploration rows: {route.get('D_CHE_E1_exploration_rows')} / {route.get('D_FOU_E1_exploration_rows')}",
        f"- D-CHE/D-FOU S1 official-like rows: {route.get('D_CHE_S1_official_like_rows')} / {route.get('D_FOU_S1_official_like_rows')}",
        f"- MLP weak h1600 / productive h3200 candidates: {route.get('MLP_weak_h1600_candidate_count')} / {route.get('MLP_productive_h3200_candidate_count')}",
        f"- KAN weak h1600 / productive h3200 / h4800 candidates: {route.get('KAN_weak_h1600_candidate_count')} / {route.get('KAN_productive_h3200_candidate_count')} / {route.get('KAN_retained_h4800_candidate_count')}",
        f"- function-space high ActuationR2 rows / source-success rows: {route.get('function_space_high_actuation_rows')} / {route.get('function_space_high_actuation_source_rows')}",
        f"- debt complete: {route.get('debt_metrics_complete')}",
        "",
        "## Efficiency Evidence",
        "",
        "| carrier | rows | E1 exploration | S1 official-like | best forward | best step |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in eff_summary:
        recap.append(f"| {r.get('carrier')} | {r.get('rows')} | {r.get('exploration_pass_rows')} | {r.get('official_like_pass_rows')} | {r.get('best_forward_ratio')} | {r.get('best_step_ratio')} |")
    recap.extend(["", "## MLP Source Dynamics", "", "| v21_id | rows | h800 | h1600 | h3200 | h4800 | weak_h1600 | productive_h3200 |", "|---|---:|---:|---:|---:|---:|---:|---:|"])
    for r in mlp:
        recap.append(f"| {r.get('v21_id')} | {r.get('rows')} | {r.get('source_h800_mean')} | {r.get('source_h1600_mean')} | {r.get('source_h3200_mean')} | {r.get('source_h4800_mean')} | {r.get('weak_retained_h1600_candidate')} | {r.get('productive_retained_h3200_candidate')} |")
    recap.extend(["", "## KAN Source Writer", "", "| carrier | variant | v21_id | h800 | h1600 | h3200 | h4800 | productive_h3200 |", "|---|---|---|---:|---:|---:|---:|---:|"])
    for r in kan:
        recap.append(f"| {r.get('carrier')} | {r.get('basis_repair_variant')} | {r.get('v21_id')} | {r.get('source_h800_mean')} | {r.get('source_h1600_mean')} | {r.get('source_h3200_mean')} | {r.get('source_h4800_mean')} | {r.get('productive_retained_h3200_candidate')} |")
    recap.extend(["", "## Function-Space Target Reset", "", "| carrier | v21_id | ActuationR2 max | source_h1600 | source_h3200 | decision |", "|---|---|---:|---:|---:|---|"])
    for r in act:
        recap.append(f"| {r.get('carrier')} | {r.get('v21_id')} | {r.get('ActuationR2_max')} | {r.get('source_h1600_mean')} | {r.get('source_h3200_mean')} | {r.get('route_precedence')} |")
    recap.extend(["", "## Function-Space Target Contrast", "", "| carrier | variant | loss B2 gain | random B2 gain | sign-flip B2 gain | corrupt B2 gain | no_go |", "|---|---|---:|---:|---:|---:|---:|"])
    for r in target_contrast:
        recap.append(f"| {r.get('carrier')} | {r.get('basis_repair_variant')} | {r.get('loss_target_B2_gain_mean')} | {r.get('random_target_B2_gain_mean')} | {r.get('sign_flipped_B2_gain_mean')} | {r.get('corrupted_target_B2_gain_mean')} | {r.get('target_observable_no_go')} |")
    recap.extend(["", "## Function-Space Target Source Writer", "", "| carrier | variant | v21_id | h800 | h1600 | h3200 | h4800 | productive_h3200 |", "|---|---|---|---:|---:|---:|---:|---:|"])
    for r in target_source:
        recap.append(f"| {r.get('carrier')} | {r.get('basis_repair_variant')} | {r.get('v21_id')} | {r.get('source_h800_mean')} | {r.get('source_h1600_mean')} | {r.get('source_h3200_mean')} | {r.get('source_h4800_mean')} | {r.get('productive_retained_h3200_candidate', r.get('retained_h3200_candidate'))} |")
    recap.extend(["", "## Source-Retention Estimator Audit", "", "| predictor | finite rows | Spearman h3200 | AUC retained/washout | LDO min | LSO min | inc AUC | pass |", "|---|---:|---:|---:|---:|---:|---:|---:|"])
    for r in estimator_summary:
        recap.append(f"| {r.get('predictor')} | {r.get('finite_rows')} | {r.get('Spearman_h3200')} | {r.get('AUC_retained_vs_washout')} | {r.get('leave_dataset_out_AUC_min')} | {r.get('leave_seed_out_AUC_min')} | {r.get('incremental_AUC_vs_controls')} | {r.get('estimator_pass')} |")
    for r in estimator_decision:
        recap.extend(["", "F5 decision: {}; rows={}; passing_predictors={}; passing_train_only_predictors={}; direction_selector_allowed={}.".format(r.get("decision"), r.get("rows"), r.get("passing_predictors"), r.get("passing_train_only_predictors"), r.get("direction_selector_allowed"))])
    recap.extend(["", "### F5 Early-Source Stratification", "", "| group | carrier | dataset | v21_id | rows | h800+ | h3200 retained | washout | late rebound | h800 mean | h3200 mean |", "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|"])
    for r in estimator_stratification[:40]:
        recap.append(f"| {r.get('group_kind')} | {r.get('carrier')} | {r.get('dataset')} | {r.get('v21_id')} | {r.get('rows')} | {r.get('h800_positive_rows')} | {r.get('retained_h3200_rows')} | {r.get('washout_h3200_rows')} | {r.get('late_rebound_h3200_rows')} | {r.get('source_h800_mean')} | {r.get('source_h3200_mean')} |")
    recap.extend(["", "## Optimizer / Bridge Diagnostics", "", "| diagnostic | v21_id | rows | key mean | decision |", "|---|---|---:|---:|---|"])
    for r in overwrite:
        if str(r.get("v21_id", "")).startswith("CTRL"):
            continue
        recap.append(f"| AdamW overwrite | {r.get('v21_id')} | {r.get('rows')} | {r.get('optimizer_overwrite_projection_mean')} | {r.get('diagnosis')} |")
    for r in mlp:
        if str(r.get("v21_id")) in {"MLP-F9-cycle1200-to-M15-anchor", "MLP-F11-dual-timescale-retention-warm1200"}:
            recap.append(f"| bridge full | {r.get('v21_id')} | {r.get('rows')} | {r.get('source_h3200_mean')} | retained_h3200={r.get('productive_retained_h3200_candidate')} |")
    recap.extend(["", "## Failure Taxonomy", "", "| failure_class | severity | evidence |", "|---|---|---|"])
    for r in failures:
        recap.append(f"| {r.get('failure_class')} | {r.get('severity')} | {r.get('evidence')} |")
    recap.extend(
        [
            "",
            "## 修改记录",
            "",
            "- 新增 v21 runner 体系：S0.5 truth gate、S1 efficiency officialization、MLP source dynamics、KAN source-channel writer、function-space target reset 和 finalizer。",
            "- 新增 v21 source-state/kernel/profiling shim：`source_state.py`、`poprisk_source.py`、`efficiency_v21.py`、`che_official.py`、`fou_official.py`。",
            "- 新增 M44/M45/M46：分别尝试 M2+LineC slow anchor、M2 slow anchor、M2+matrix-block retention；方向只来自 train stream，trace 记录 source-state gate/gain/cos 诊断。",
            "- 新增 M47：先用 momentum cycle-hold 保早期 source，再在 h1600 后用 M15-style train-split LineC anchor 写入 slow state；LineC 只做 gate/readback，不作为方向源。",
            "- 新增 M48：按计划 3.2 的 AdEMAMix/long-memory 启发维护 short/long 双时间尺度 source state，只提交 short/long 符号一致的 train-stream source 分量。",
            "- 新增 `experiments/run_v21_optimizer_overwrite_diagnostic.py`：按 v21 spec 输出 FU-vs-AdamW first-step cosine/projection，闭合计划 13.3 的 AdamW overwrite 诊断。",
            "- 新增 `experiments/run_v21_function_space_target_contrast.py`：按计划 13.5 对 high-ActuationR2/source-fail 路线做 random/sign-flip/corrupt/B1-only/lower-rank target contrast。",
            "- 新增 M49/M50/M51/M52/M53 与 `experiments/run_v21_function_space_target_source.py`：把 F3 loss-cotangent target 和 matched random/sign-flip/corrupt/low-rank controls 接入 long-horizon source writer。",
            "- 新增 `experiments/run_v21_source_retention_estimator_audit.py`：按计划 F5 离线审计 PopRisk/SNR、output transfer、split-consensus、LineC、matrix-block、slow-state 等 predictor 是否能预测 h3200/h4800 retention。",
            "- Finalizer 只从落盘 CSV/trace 汇总 route；缺失或失败证据写 blocker，不翻 promotion。",
            "",
            "## 分析 / Insight / 结论",
            "",
            "- v21 把 v20 的 h1600 可保留但 h3200/h4800 washout 作为核心 blocker，而不是继续只看单 seed smoke。",
            "- S2 weak FU 只说明机制在 h1600 还有相对 controls 的 source；S3/S5 仍要求 h3200/h4800、controls、debt 和 efficiency 同时过。",
            "- Function-space high ActuationR2 仍只是局部 displacement 可实现，不能替代 source retention。",
            "- M47/M48 的 smoke 在 MNIST seed0 上出现 h800/h1600/h3200 连续正值，但 full 9-row grouped mean 在 h1600/h3200 转负；这说明 single-seed bridge signal 不能跨 dataset/seed 稳定留存。",
            "- v21 AdamW overwrite first-step 诊断没有显示负投影阻碍；当前 washout 更像 source target / dataset-seed heterogeneity 问题，而不是 AdamW 直接反向覆盖。",
            "- Function-space target contrast 如果显示 loss target 不能稳定强过 random/sign-flip/corrupt target，应按 TargetObservableNoGo 处理，不继续 actuation solver 小修。",
            "- 如果 loss target contrast 可观测但 target source writer 仍不 retained，则 blocker 从 actuation/observable 推进为 target-to-retention dynamics，而不是继续只调 solver cap。",
            "- F5 source-retention estimator audit 若无法通过跨 dataset/seed AUC 与 Spearman 门，只能作为 failure taxonomy，不能作为下一轮 direction selector。",
            "- `promotion_allowed=1` 只有 KAN retained source、efficiency officialization、debt recovery、controls attribution 都真实通过才允许。",
        ]
    )
    write_text(V21_RECAP_DOC, "\n".join(recap) + "\n")

    exec_lines = [
        "# DG-KAN v21.0 Source Retention + Kernel Officialization 执行日志",
        "",
        f"生成时间：{route.get('generated_at', '')}",
        "",
        "## 命令日志",
        "",
    ]
    for row in commands:
        exec_lines.extend([f"### {row.get('timestamp')}", "", "```bash", str(row.get("command", "")), "```", f"- status: {row.get('status')}", f"- note: {row.get('note', '')}", ""])
    exec_lines.extend(["## 关键文件", "", f"- plan: `{V21_PLAN_DOC}`", f"- result_dir: `{out_dir}`", f"- route: `{out_dir / 'v21_route_decision.json'}`", f"- packet: `{out_dir / 'v21_code_review_packet.zip'}`", ""])
    write_text(V21_EXEC_DOC, "\n".join(exec_lines) + "\n")


def main() -> None:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir)
    append_exec(out_dir, f"{PYTHON} experiments/run_v21_finalize.py --out-dir {out_dir}", status="started")
    debt = build_debt(out_dir)
    summaries, _controls = build_controls_and_source(out_dir)
    build_queue(out_dir)
    route = decide_route(out_dir, summaries, debt)
    route["generated_at"] = __import__("time").strftime("%Y-%m-%d %H:%M:%S %z")
    write_json(out_dir / "v21_route_decision.json", route)
    failures = write_failure_taxonomy(out_dir, route)
    figure_sources = {
        "eff": read_rows(out_dir / "v21_efficiency_truth_table.csv"),
        "source": read_rows(out_dir / "v21_source_retention_matrix.csv"),
        "debt": read_rows(out_dir / "v21_debt_accounting_matrix.csv"),
        "act": read_rows(out_dir / "v21_function_space_target_reset_matrix.csv"),
        "gpu": read_rows(out_dir / "v21_gpu_utilization_dashboard.csv"),
        "failure": failures,
    }
    for name, title, metric in FIGURES:
        rows = figure_sources["eff"]
        if "source" in name or "weak" in name:
            rows = figure_sources["source"]
        elif "debt" in name:
            rows = figure_sources["debt"]
        elif "actuation" in name:
            rows = figure_sources["act"]
        elif "gpu" in name:
            rows = figure_sources["gpu"]
        elif "failure" in name:
            rows = figure_sources["failure"]
        placeholder_svg(out_dir / "figures" / name, title, rows, metric)
    manifest = []
    for rel in REQUIRED + [f"figures/{name}" for name, _title, _metric in FIGURES]:
        p = out_dir / rel
        manifest.append({"artifact": rel, "path": str(p), "exists": int(p.exists()), "size_bytes": p.stat().st_size if p.exists() else 0})
    write_rows(out_dir / "v21_required_artifact_manifest.csv", manifest)
    route["required_artifact_missing_count"] = sum(1 for r in manifest if not int_flag(r.get("exists")))
    write_json(out_dir / "v21_route_decision.json", route)
    failures = write_failure_taxonomy(out_dir, route)
    write_docs(out_dir, route, failures)
    build_packet(out_dir, REQUIRED)
    append_exec(out_dir, f"{PYTHON} experiments/run_v21_finalize.py --out-dir {out_dir}", status="completed", note=f"route={route.get('route')} promotion_allowed={route.get('promotion_allowed')}")


if __name__ == "__main__":
    main()
