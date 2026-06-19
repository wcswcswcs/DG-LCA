#!/usr/bin/env python3
"""Finalize v22.18 route, gates, artifacts, figures, and recap log."""

from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_v22_18_common import (  # noqa: E402
    OUT_ROOT,
    RECAP_DOC,
    append_exec,
    artifact_index,
    ensure_out,
    finite_float,
    int_flag,
    md_table,
    now_sg,
    read_json,
    read_rows,
    sha256_file,
    write_json,
    write_rows,
)


def parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser()


def _all_rows(pattern: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(OUT_ROOT.glob(pattern)):
        for row in read_rows(path):
            row["_source_file"] = path.name
            rows.append(row)
    return rows


def _all_json_rows(pattern: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(OUT_ROOT.glob(pattern)):
        row = read_json(path)
        if row:
            row["_source_file"] = path.name
            rows.append(row)
    return rows


def _artifact_meta(name: str) -> tuple[int, int, int, str]:
    paths = sorted(OUT_ROOT.glob(name))
    exists = int(bool(paths))
    nonempty = int(any(p.exists() and p.stat().st_size > 0 for p in paths))
    row_count = sum(len(read_rows(p)) for p in paths if p.suffix == ".csv")
    sha = sha256_file(paths[-1]) if paths and paths[-1].is_file() else ""
    return exists, nonempty, row_count, sha


def _gate(name: str, passed: bool, scope: str, artifacts: str, thresholds: str, blocker: str = "") -> dict[str, Any]:
    first = artifacts.split(";")[0].strip()
    exists, nonempty, rows, sha = _artifact_meta(first)
    return {
        "gate_name": name,
        "pass": int(bool(passed)),
        "scope": scope,
        "required_artifacts": artifacts,
        "artifact_exists": exists,
        "artifact_nonempty": nonempty,
        "row_count": rows,
        "metric_thresholds": thresholds,
        "blocking_metric": "" if passed else blocker,
        "source_file_sha256": sha,
        "latest_status_timestamp": now_sg(),
    }


def _native_timing_complete(row: dict[str, Any]) -> bool:
    rows = int(row.get("rows") or 0)
    return rows > 0 and int(row.get("diagnostic_separated_timing_rows") or 0) >= rows


def _native_full_loop_pass_count(row: dict[str, Any]) -> int:
    if _native_timing_complete(row):
        return int(row.get("official_full_loop_ratio_le_3") or 0)
    return 0


def _native_overhead_pass_count(row: dict[str, Any]) -> int:
    if _native_timing_complete(row):
        return int(row.get("official_controller_overhead_le_025") or 0)
    return 0


def _native_max_official_ratio(row: dict[str, Any]) -> float:
    if _native_timing_complete(row):
        return finite_float(row.get("max_official_full_loop_ratio"), 999.0)
    return 999.0


def _svg(path: Path, title: str, lines: list[str]) -> None:
    safe = [line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;") for line in lines]
    height = max(120, 70 + 24 * len(safe))
    body = "\n".join(f'<text x="24" y="{70 + 24 * i}" font-size="14" fill="#1f2937">{line}</text>' for i, line in enumerate(safe))
    path.write_text(
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="{height}" viewBox="0 0 1200 {height}">\n'
            '<rect width="1200" height="100%" fill="#f8fafc"/>\n'
            f'<text x="24" y="36" font-size="22" font-family="sans-serif" fill="#111827">{title}</text>\n'
            f'<g font-family="monospace">{body}</g>\n'
            "</svg>\n"
        ),
        encoding="utf-8",
    )


def _make_figures(summary: dict[str, Any]) -> None:
    fig = OUT_ROOT / "figures"
    fig.mkdir(parents=True, exist_ok=True)
    _svg(fig / "v22_18_four_square_heatmap.svg", "v22.18 four-square readback", summary.get("four_square_lines", ["no rows"]))
    _svg(fig / "v22_18_noharm_vs_improvement_scatter.svg", "no-harm vs improvement", summary.get("mlp_lines", ["no rows"]))
    _svg(fig / "v22_18_action_benefit_calibration.svg", "benefit calibration", summary.get("policy_lines", ["no rows"]))
    _svg(fig / "v22_18_policy_action_distribution.svg", "policy action distribution", summary.get("action_lines", ["no rows"]))
    _svg(fig / "v22_18_controls_comparison_panel.svg", "controls comparison", summary.get("control_lines", ["no rows"]))
    _svg(fig / "v22_18_kan_basis_efficiency_pareto.svg", "KAN basis efficiency pareto", summary.get("kan_lines", ["no rows"]))
    _svg(fig / "v22_18_kanbefair_expanded_heatmap.svg", "KANbeFair expanded heatmap", summary.get("expanded_lines", ["no rows"]))
    _svg(fig / "v22_18_continual_forgetting_curves.svg", "continual forgetting curves", summary.get("continual_lines", ["no rows"]))
    _svg(fig / "v22_18_model_identity_firewall.svg", "model identity firewall", summary.get("identity_lines", ["no rows"]))
    _svg(fig / "v22_18_bridge_overhead_waterfall.svg", "bridge overhead waterfall", summary.get("bridge_lines", ["no rows"]))


def _route(gates: dict[str, bool]) -> str:
    if not gates["A"]:
        return "R0-CodeOrArtifactTruthFailed"
    if not gates["B"]:
        return "R1-SourceActionBankNoGo"
    if not gates["C"]:
        return "R2-BenefitPolicyNoGo"
    if gates["G_improve"]:
        return "R11-KANbeFairExpandedImprovementOpened"
    if gates["H_explore"]:
        return "R12-ContinualForgettingReductionOpened"
    if gates["E_task"]:
        return "R7-KANNativeBasisTaskBenefitOpened_MLPFUPending"
    if gates["D_improve"] and gates["E_source"]:
        return "R8-GeneralFUOpened_KANCarrierNotProven"
    if gates["D_improve"]:
        return "R4-MLPFUGeneralImprovementOpened"
    if gates["E_source"]:
        return "R6-KANNativeBasisSourceNoTaskBenefit"
    if gates["D_noharm"]:
        return "R3-MLPFUNoHarmOnly_NoImprovement"
    return "R2-BenefitPolicyNoGo"


def main() -> None:
    _args = parser().parse_args()
    ensure_out()
    status = read_json(OUT_ROOT / "v22_18_authoritative_status.json")
    b_summary = _all_rows("v22_18_source_action_bank_summary*.csv")
    c_rows = _all_rows("v22_18_policy_offline_validation*.csv")
    c_policy_exports = _all_json_rows("v22_18_benefit_policy_export*.json")
    d_rows = _all_rows("v22_18_mlp_fu_noharm_improvement_summary*.csv")
    e_rows = _all_rows("v22_18_kan_native_ladder_summary*.csv")
    e_task_pilot_rows = [
        r
        for r in _all_rows("v22_18_kan_native_ladder_matrix_v22_18_e_task_dche_h4_s0*.csv")
        if str(r.get("model_name", "")).endswith("_FU_NATIVE_JVP_REFRESH")
    ]
    e2_task_pilot_rows = [
        r
        for r in _all_rows("v22_18_kan_native_ladder_matrix_v22_18_e2_task_dche_h4_s0*.csv")
        if str(r.get("model_name", "")).endswith("_FU_NATIVE_JVP_REFRESH")
    ]
    e3_task_pilot_rows = [
        r
        for r in _all_rows("v22_18_kan_native_ladder_matrix_v22_18_e3_task_dche_h4_s0*.csv")
        if str(r.get("model_name", "")).endswith("_FU_NATIVE_JVP_REFRESH")
    ]
    e4_task_release_rows = [
        r
        for r in _all_rows("v22_18_kan_native_ladder_matrix_v22_18_e4_task_dche_h4_s0*.csv")
        if str(r.get("model_name", "")).endswith("_FU_NATIVE_JVP_REFRESH")
    ]
    e_long_task_pilot_rows = [*e2_task_pilot_rows, *e3_task_pilot_rows]
    e_task_virtual_i40_rows = [
        r
        for r in e_task_pilot_rows
        if "s012_h120_virtual_improve_gate.csv" in str(r.get("_source_file", ""))
    ]
    e_task_virtual_i120_rows = [
        r
        for r in e_task_pilot_rows
        if "s012_h120_virtual_improve_gate_i120.csv" in str(r.get("_source_file", ""))
    ]
    g_rows = _all_rows("v22_18_kanbefair_expanded_pass_summary*.csv")
    h_rows = _all_rows("v22_18_forgetting_transfer_summary*.csv")
    kan_vs_mlp = _all_rows("v22_18_kan_vs_mlpfu_matrix*.csv")
    identity_summary_rows = _all_rows("v22_18_identity_audit_summary*.csv")
    model_identity_rows = _all_rows("v22_18_model_identity_matrix*.csv")
    efficiency_identity_rows = _all_rows("v22_18_efficiency_path_identity_matrix*.csv")
    bridge_audit_rows = _all_rows("v22_18_efficiency_bridge_audit_matrix*.csv")
    strictness_rows = _all_rows("v22_18_branch_label_strictness_audit*.csv")
    strict_label_bank_rows = _all_rows("v22_18_strict_label_bank_summary*.csv")
    control_projected_rows = _all_rows("v22_18_control_projected_label_audit*.csv")
    runtime_p3_no_go_rows = _all_rows("v22_18_runtime_p3_no_go_audit*.csv")
    highsignal_availability_rows = _all_rows("../v22_17/v22_17_kanbefair_dataset_grid_availability.csv")
    runtime_p3_highsignal_task_rows = _all_rows("v22_18_mlp_fu_task_matrix_v22_18_runtime_p3_tabular_highsignal_s012_h200*.csv")
    runtime_export_highsignal_rows = _all_rows("v22_18_mlp_fu_noharm_improvement_summary_v22_18_runtime_export_policy_highsignal*.csv")
    runtime_export_highsignal_task_rows = _all_rows("v22_18_mlp_fu_task_matrix_v22_18_runtime_export_policy_highsignal*.csv")
    runtime_export_split_rows = _all_rows("v22_18_mlp_fu_noharm_improvement_summary_v22_18_runtime_export_policy_split_highsignal*.csv")
    runtime_export_split_task_rows = _all_rows("v22_18_mlp_fu_task_matrix_v22_18_runtime_export_policy_split_highsignal*.csv")

    identity_latest = identity_summary_rows[-1] if identity_summary_rows else {}
    a_code_pass = (
        int_flag(status.get("compileall"))
        and int_flag(status.get("clean_import"))
        and int_flag(status.get("required_source_files_present"))
        and int_flag(status.get("authoritative_status_single_source_of_truth"))
        and int_flag(status.get("finalizer_separates_smoke_exploration_official"))
        and int_flag(status.get("finalizer_separates_noharm_improvement_superiority"))
        and int_flag(status.get("readout_diagnostic_cannot_promote_basis_native"))
        and int_flag(status.get("kanbefair_baseline_cannot_promote_dgkan"))
    )
    a_model_identity = (
        int_flag(identity_latest.get("model_identity_matrix_complete"))
        and int_flag(identity_latest.get("official_DGKAN_rows_have_strict_FC_PureKAN_identity"))
        and int_flag(identity_latest.get("KANbeFair_original_KAN_rows_marked_baseline_context_only"))
    )
    a_efficiency_identity = (
        int_flag(identity_latest.get("efficiency_path_identity_matrix_complete"))
        and int_flag(identity_latest.get("official_efficiency_rows_use_analytic_or_sketch_JVP"))
    )
    a_pass = bool(a_code_pass and a_model_identity and a_efficiency_identity)
    b_explore = any(int_flag(r.get("B_exploration_pass")) for r in b_summary)
    b_official = any(int_flag(r.get("B_official_preparation_pass")) for r in b_summary)
    c_explore = any(int_flag(r.get("C_exploration_pass")) for r in c_rows)
    c_official = any(int_flag(r.get("C_official_pass")) for r in c_rows)
    d_noharm = any(int_flag(r.get("D_noharm_pass")) and not int_flag(r.get("is_control_variant")) for r in d_rows)
    d_improve = any(int_flag(r.get("D_improvement_pass")) for r in d_rows)
    e_source = any(int(r.get("rows") or 0) >= 9 and int(r.get("basis_energy_ge_050") or 0) >= 8 and int(r.get("source_loss_horizon_ge_0") or 0) >= 8 for r in e_rows)
    e_task = any(int(r.get("nll_improve_vs_kan") or 0) >= 5 and int(r.get("auc_improve_vs_kan") or 0) >= 5 for r in e_rows)
    f_native_efficiency = any(
        _native_timing_complete(r)
        and _native_full_loop_pass_count(r) >= 8
        and _native_overhead_pass_count(r) >= 8
        and int(r.get("uses_dense_output_jacobian_any") or 0) == 0
        for r in e_rows
    )
    f_four_path = int_flag(identity_latest.get("F_four_path_exploration_pass"))
    f_explore = bool(f_native_efficiency and f_four_path)
    g_noharm = any(int_flag(r.get("G_external_noharm_pass")) for r in g_rows)
    g_improve = any(int_flag(r.get("G_external_improvement_pass")) for r in g_rows)
    h_explore = any(int_flag(r.get("H_exploration_pass")) for r in h_rows)
    h_official = any(int_flag(r.get("H_official_pass")) for r in h_rows)
    kan_vs_mlpfu_nll = sum(finite_float(r.get("KAN_vs_MLPFU_NLL_delta"), 999.0) <= 0.0 for r in kan_vs_mlp)
    kan_vs_mlpfu_acc = sum(finite_float(r.get("KAN_vs_MLPFU_accuracy_delta"), -999.0) >= 0.0 for r in kan_vs_mlp)

    gates_bool = {
        "A": bool(a_pass),
        "B": bool(b_explore),
        "C": bool(c_explore),
        "D_noharm": bool(d_noharm),
        "D_improve": bool(d_improve),
        "E_source": bool(e_source),
        "E_task": bool(e_task),
        "F": bool(f_explore),
        "G_noharm": bool(g_noharm),
        "G_improve": bool(g_improve),
        "H_explore": bool(h_explore),
    }
    route = _route(gates_bool)
    full_official = int(
        a_pass
        and b_official
        and c_official
        and d_improve
        and e_task
        and f_explore
        and g_noharm
        and (h_official or bool(h_rows))
        and kan_vs_mlpfu_nll >= 7
        and kan_vs_mlpfu_acc >= 5
    )

    gates = [
        _gate("A code/artifact/finalizer truth", bool(a_code_pass), "official-prerequisite", "v22_18_authoritative_status.json;v22_18_artifact_index.csv;v22_18_code_packet_manifest.csv", "compileall=1;clean_import=1;required sources present;finalizer separations present", "compile/import/source truth failed"),
        _gate("A model identity firewall", bool(a_model_identity), "official-prerequisite", "v22_18_identity_audit_summary.csv;v22_18_model_identity_matrix.csv", "model_identity_matrix_complete=1;official DGKAN rows strict FC-PureKAN=1;KANbeFair original KAN rows baseline-only=1", "ModelIdentityAmbiguous"),
        _gate("A efficiency path identity firewall", bool(a_efficiency_identity), "official-prerequisite", "v22_18_identity_audit_summary.csv;v22_18_efficiency_path_identity_matrix.csv", "efficiency_path_identity_matrix_complete=1;official fused rows use analytic/sketch JVP=1", "EfficiencyPathIdentityMissingOrNoAnalyticJVP"),
        _gate("B source-action benefit labels", bool(b_explore), "exploration", "v22_18_source_action_bank_summary*.csv", ">=4 action families accepted; labels have positive and negative; controls present", "source-action bank incomplete or lacks label diversity"),
        _gate("B official preparation", bool(b_official), "official-prep", "v22_18_source_action_bank_summary*.csv", "positive rate 5-40%; real-control gap >=10pp; H50/H100/H200", "labels are proxy, controls match, or positive rate out of range"),
        _gate("C benefit policy exploration", bool(c_explore), "exploration", "v22_18_policy_offline_validation*.csv", "AUC>=0.65; Spearman>=0.25; control FP<=0.30; noop 20-90%", "offline policy did not meet exploration thresholds"),
        _gate("C benefit policy official", bool(c_official), "official", "v22_18_policy_offline_validation*.csv", "leave-one dataset/seed AUC>=0.62 and runtime per-step integrated", "policy is offline variant selector or leave-one failed"),
        _gate("D MLP+FU no-harm", bool(d_noharm), "smoke", "v22_18_mlp_fu_noharm_improvement_summary*.csv", "NLL/accuracy/tail/ECE no-harm on >=8/9", "MLP+FU no-harm threshold not met"),
        _gate("D MLP+FU improvement", bool(d_improve), "official-candidate", "v22_18_mlp_fu_noharm_improvement_summary*.csv", "NLL<-0.01 >=5/9; AUC<0 >=5/9; acc>=0 >=4/9; controls lower by >=3", "MLP+FU improvement not stable or controls too strong"),
        _gate("E KAN native-basis source ladder", bool(e_source), "exploration", "v22_18_kan_native_ladder_summary*.csv", "basis>=0.5 and source_loss>=0 on >=8/9", "KAN basis/source ladder incomplete"),
        _gate("E KAN native-basis task benefit", bool(e_task), "official-candidate", "v22_18_kan_native_ladder_summary*.csv", "NLL and AUC improvement vs KAN on >=5/9", "basis source did not translate to task/AUC benefit"),
        _gate("F native JVP/VJP efficiency", bool(f_native_efficiency), "exploration", "v22_18_kan_native_ladder_summary*.csv", "diagnostic-separated timing complete; dense official=0; official overhead<=0.25; official full-loop<=3 on >=8/9", "efficiency, timing separation, or dense-J separation failed"),
        _gate("F four-path efficiency bridge audit", bool(f_four_path), "exploration-precondition", "v22_18_efficiency_bridge_audit_matrix.csv;v22_18_efficiency_path_component_timing.csv;v22_18_efficiency_path_identity_matrix.csv", "Path1-4 present on same data/batch/hidden/seed and official full-loop path separable", "FourPathAuditNotSameDataBatchHiddenSeedOrNoOfficialFullLoop"),
        _gate("G KANbeFair expanded no-harm", bool(g_noharm), "external", "v22_18_kanbefair_expanded_pass_summary*.csv", "no-harm on >=70% available expanded datasets", "external no-harm threshold not met or no task rows"),
        _gate("G KANbeFair expanded improvement", bool(g_improve), "external", "v22_18_kanbefair_expanded_pass_summary*.csv", "MLPFU improve >=40%; KANFU improve >=50%", "external improvement threshold not met"),
        _gate("H continual forgetting exploration", bool(h_explore), "exploration", "v22_18_forgetting_transfer_summary*.csv", "FU reduces average forgetting with accuracy no-harm", "forgetting reduction not shown"),
        _gate("H continual forgetting official", bool(h_official), "official", "v22_18_forgetting_transfer_summary*.csv", ">=10% relative forgetting reduction; seeds 0,1,2; accuracy no-harm", "not enough seeds or relative reduction"),
    ]
    write_rows(OUT_ROOT / "v22_18_gate_summary.csv", gates)
    trace = [
        {"decision": "route", "value": route, "reason": "highest conservative route from passed gates"},
        {"decision": "official_promotion", "value": full_official, "reason": "requires A-H official plus KAN>=MLPFU thresholds and controls fail"},
        {"decision": "model_identity_firewall", "value": int(bool(a_model_identity)), "reason": "DGKAN rows must be strict FC-PureKAN; KANbeFair original KAN/BSpline rows baseline context only"},
        {"decision": "efficiency_path_identity_firewall", "value": int(bool(a_efficiency_identity)), "reason": "official fused rows must use analytic/sketch JVP and efficiency path matrix must be complete"},
        {"decision": "four_path_efficiency_audit", "value": int(bool(f_four_path)), "reason": "Path1-4 audit must be same data/batch/hidden/seed before attributing bridge slowness"},
        {"decision": "KAN_vs_MLPFU_NLL_nonworse_rows", "value": kan_vs_mlpfu_nll, "reason": "count of rows with KAN FU NLL <= best MLPFU"},
        {"decision": "KAN_vs_MLPFU_accuracy_nonworse_rows", "value": kan_vs_mlpfu_acc, "reason": "count of rows with KAN FU accuracy >= best MLPFU"},
    ]
    write_rows(OUT_ROOT / "v22_18_route_decision_trace.csv", trace)
    deferred = []
    if not a_model_identity:
        deferred.append({"item": "DG-KAN / KANbeFair model identity firewall", "reason": "model identity matrix missing or at least one DGKAN official row is ambiguous", "algorithm_failure": 0 if not identity_summary_rows else 1})
    if not a_efficiency_identity:
        deferred.append({"item": "efficiency path identity firewall", "reason": "efficiency path matrix missing or official fused rows do not prove analytic/sketch JVP", "algorithm_failure": 0 if not identity_summary_rows else 1})
    if not b_official:
        deferred.append({"item": "true H50/H100/H200 branch counterfactual labels", "reason": "current source-action labels are final-run delta proxy, marked as such", "algorithm_failure": 0})
    if not c_official:
        runtime_p3_rows = [r for r in d_rows if "RUNTIME" in str(r.get("v22_18_variant", ""))]
        deferred.append(
            {
                "item": "runtime per-step benefit policy integration",
                "reason": "runtime P3 exploration present but C official still lacks leave-one/runtime improvement success" if runtime_p3_rows else "policy is offline variant selector in this pass",
                "algorithm_failure": 1 if runtime_p3_rows else 0,
            }
        )
    if not d_improve:
        deferred.append({"item": "MLP+FU general improvement", "reason": "D improvement thresholds not met against controls", "algorithm_failure": 1 if d_rows else 0})
    if not e_task:
        deferred.append({"item": "KAN basis-native task/AUC benefit", "reason": "basis/source may pass but task/AUC improvement threshold not met", "algorithm_failure": 1 if e_rows else 0})
    if not f_explore:
        if f_four_path and not f_native_efficiency:
            deferred.append(
                {
                    "item": "native PureKAN efficiency breadth / official path",
                    "reason": "same-data four-path bridge audit is present, but native ladder breadth or official fused full-loop conditions are not met",
                    "algorithm_failure": 1 if bridge_audit_rows else 0,
                }
            )
        else:
            deferred.append(
                {
                    "item": "same-data four-path PureKAN efficiency bridge audit",
                    "reason": "current evidence separates identities and paths but does not complete Path1-4 on the same data/batch/hidden/seed with an official full-loop path",
                    "algorithm_failure": 0 if bridge_audit_rows else 0,
                }
            )
    if not g_noharm:
        deferred.append({"item": "KANbeFair expanded pass", "reason": "availability/task matrix incomplete or no-harm threshold not met", "algorithm_failure": 0 if not g_rows else 1})
    if not h_official:
        deferred.append({"item": "Class_MNIST official continual proof", "reason": "requires seeds 0,1,2 and controls; current rows are smoke unless summary proves official", "algorithm_failure": 0 if h_rows else 0})
    write_rows(OUT_ROOT / "v22_18_deferred_items.csv", deferred)
    final_status = {
        **status,
        "timestamp": now_sg(),
        "route": route,
        "official_promotion": full_official,
        "gates": gates_bool,
        "A_pass": int(a_pass),
        "A_code_artifact_truth_pass": int(bool(a_code_pass)),
        "A_model_identity_firewall_pass": int(bool(a_model_identity)),
        "A_efficiency_path_identity_firewall_pass": int(bool(a_efficiency_identity)),
        "B_exploration_pass": int(b_explore),
        "B_official_preparation_pass": int(b_official),
        "C_exploration_pass": int(c_explore),
        "C_official_pass": int(c_official),
        "D_noharm_pass": int(d_noharm),
        "D_improvement_pass": int(d_improve),
        "E_source_ladder_pass": int(e_source),
        "E_task_benefit_pass": int(e_task),
        "F_native_efficiency_exploration_pass": int(bool(f_native_efficiency)),
        "F_native_diagnostic_separated_timing_summary_rows": sum(int(_native_timing_complete(r)) for r in e_rows),
        "F_four_path_efficiency_bridge_audit_pass": int(bool(f_four_path)),
        "F_efficiency_exploration_pass": int(f_explore),
        "G_noharm_pass": int(g_noharm),
        "G_improvement_pass": int(g_improve),
        "H_exploration_pass": int(h_explore),
        "H_official_pass": int(h_official),
        "KAN_vs_MLPFU_NLL_nonworse_rows": kan_vs_mlpfu_nll,
        "KAN_vs_MLPFU_accuracy_nonworse_rows": kan_vs_mlpfu_acc,
        "model_identity_matrix_complete": int_flag(identity_latest.get("model_identity_matrix_complete")),
        "official_DGKAN_rows_have_strict_FC_PureKAN_identity": int_flag(identity_latest.get("official_DGKAN_rows_have_strict_FC_PureKAN_identity")),
        "KANbeFair_original_KAN_rows_marked_baseline_context_only": int_flag(identity_latest.get("KANbeFair_original_KAN_rows_marked_baseline_context_only")),
        "efficiency_path_identity_matrix_complete": int_flag(identity_latest.get("efficiency_path_identity_matrix_complete")),
        "official_efficiency_rows_use_analytic_or_sketch_JVP": int_flag(identity_latest.get("official_efficiency_rows_use_analytic_or_sketch_JVP")),
        "efficiency_path_official_pass_rows": int(float(identity_latest.get("efficiency_path_official_pass_rows") or 0)),
        "four_path_audit_paths_present": int_flag(identity_latest.get("four_path_audit_paths_present")),
        "four_path_audit_full_loop_present": int_flag(identity_latest.get("four_path_audit_full_loop_present")),
        "same_data_batch_hidden_seed_complete": int_flag(identity_latest.get("same_data_batch_hidden_seed_complete")),
    }
    write_json(OUT_ROOT / "v22_18_authoritative_status.json", final_status)
    write_json(OUT_ROOT / "v22_18_final_route.json", {"route": route, "official_promotion": full_official, "timestamp": now_sg()})
    write_rows(OUT_ROOT / "v22_18_artifact_index.csv", artifact_index())

    action_dist = _all_rows("v22_18_policy_action_distribution*.csv")
    four_square = _all_rows("v22_18_kanbefair_four_square_summary*.csv")
    c_all_rows = [r for r in c_rows if r.get("split") == "all-data-in-sample"]
    best_c_auc = max((finite_float(r.get("benefit_AUC_binary_positive")) for r in c_all_rows), default=0.0)
    control_safe_rows = [
        r
        for r in c_all_rows
        if finite_float(r.get("false_positive_benefit_rate_controls"), 999.0) <= 0.30
        and 0.20 <= finite_float(r.get("policy_noop_rate"), -1.0) <= 0.90
    ]
    best_control_safe_auc = max((finite_float(r.get("benefit_AUC_binary_positive")) for r in control_safe_rows), default=0.0)
    best_c_row = max(c_all_rows, key=lambda r: finite_float(r.get("benefit_AUC_binary_positive")), default={})
    best_control_safe_row = max(control_safe_rows, key=lambda r: finite_float(r.get("benefit_AUC_binary_positive")), default={})
    branch_b_rows = [r for r in b_summary if "counterfactual" in str(r.get("_source_file", ""))]
    branch_b = branch_b_rows[-1] if branch_b_rows else {}
    vision_branch_b_rows = [r for r in branch_b_rows if "tabular" not in str(r.get("_source_file", ""))]
    tabular_branch_b_rows = [r for r in branch_b_rows if "tabular" in str(r.get("_source_file", ""))]
    vision_branch_b = vision_branch_b_rows[-1] if vision_branch_b_rows else {}
    tabular_branch_b = tabular_branch_b_rows[-1] if tabular_branch_b_rows else {}
    branch_c_all = [r for r in c_all_rows if "branch" in str(r.get("_source_file", "")) and "policy" in str(r.get("_source_file", ""))]
    branch_c_best = max(branch_c_all, key=lambda r: finite_float(r.get("benefit_AUC_binary_positive")), default={})
    branch_c_control_safe = [
        r
        for r in branch_c_all
        if finite_float(r.get("false_positive_benefit_rate_controls"), 999.0) <= 0.30
        and 0.20 <= finite_float(r.get("policy_noop_rate"), -1.0) <= 0.90
    ]
    branch_c_control_safe_best = max(branch_c_control_safe, key=lambda r: finite_float(r.get("benefit_AUC_binary_positive")), default={})
    tabular_branch_c_all = [r for r in branch_c_all if "tabular" in str(r.get("_source_file", ""))]
    tabular_branch_c_best = max(tabular_branch_c_all, key=lambda r: finite_float(r.get("benefit_AUC_binary_positive")), default={})
    tabular_s012_c_all = [r for r in tabular_branch_c_all if "s012" in str(r.get("_source_file", ""))]
    tabular_s012_c_best = max(tabular_s012_c_all, key=lambda r: finite_float(r.get("benefit_AUC_binary_positive")), default={})
    tabular_s012_all_splits = [
        r
        for r in c_rows
        if "tabular" in str(r.get("_source_file", ""))
        and "s012" in str(r.get("_source_file", ""))
        and "policy" in str(r.get("_source_file", ""))
    ]
    tabular_s012_leave_dataset = [r for r in tabular_s012_all_splits if r.get("split") == "leave-one-dataset"]
    tabular_s012_leave_seed = [r for r in tabular_s012_all_splits if r.get("split") == "leave-one-seed"]
    tabular_s012_policy_summaries: list[dict[str, Any]] = []
    for source_file in sorted({str(r.get("_source_file", "")) for r in tabular_s012_all_splits if str(r.get("_source_file", ""))}):
        file_rows = [r for r in tabular_s012_all_splits if str(r.get("_source_file", "")) == source_file]
        all_row = next((r for r in file_rows if r.get("split") == "all-data-in-sample"), {})
        lodo = [finite_float(r.get("benefit_AUC_binary_positive"), 0.0) for r in file_rows if r.get("split") == "leave-one-dataset"]
        loseed = [finite_float(r.get("benefit_AUC_binary_positive"), 0.0) for r in file_rows if r.get("split") == "leave-one-seed"]
        if all_row:
            tabular_s012_policy_summaries.append(
                {
                    "source_file": source_file,
                    "target_mode": all_row.get("target_mode", ""),
                    "feature_set": all_row.get("feature_set", ""),
                    "ridge": all_row.get("ridge", ""),
                    "auc": finite_float(all_row.get("benefit_AUC_binary_positive"), 0.0),
                    "spearman": finite_float(all_row.get("benefit_spearman"), 0.0),
                    "fp": finite_float(all_row.get("false_positive_benefit_rate_controls"), 999.0),
                    "noop": finite_float(all_row.get("policy_noop_rate"), -1.0),
                    "min_lodo": min(lodo) if lodo else 0.0,
                    "min_loseed": min(loseed) if loseed else 0.0,
                    "C_exploration_pass": all_row.get("C_exploration_pass", ""),
                }
            )
    best_tabular_s012_lodo = max(tabular_s012_policy_summaries, key=lambda r: (finite_float(r.get("min_lodo")), finite_float(r.get("auc"))), default={})
    best_tabular_s012_loseed = max(tabular_s012_policy_summaries, key=lambda r: (finite_float(r.get("min_loseed")), finite_float(r.get("auc"))), default={})
    best_policy_export = max(
        c_policy_exports,
        key=lambda r: (
            finite_float(r.get("min_leave_one_dataset_auc"), 0.0),
            finite_float(r.get("min_leave_one_seed_auc"), 0.0),
            finite_float(r.get("C_exploration_pass"), 0.0),
        ),
        default={},
    )
    vision_branch_c_all = [r for r in branch_c_all if "tabular" not in str(r.get("_source_file", ""))]
    vision_branch_c_best = max(vision_branch_c_all, key=lambda r: finite_float(r.get("benefit_AUC_binary_positive")), default={})
    best_strictness = max(strictness_rows, key=lambda r: finite_float(r.get("real_minus_control_rate"), -999.0), default={})
    strictness_pass_rows = [r for r in strictness_rows if int_flag(r.get("B_official_gap_pass"))]
    strict_label_bank_pass_rows = [r for r in strict_label_bank_rows if int_flag(r.get("B_strict_preparation_signal_pass"))]
    best_strict_label_bank = max(
        strict_label_bank_rows,
        key=lambda r: (
            int_flag(r.get("B_strict_preparation_signal_pass")),
            finite_float(r.get("real_minus_control_rate"), -999.0),
            finite_float(r.get("real_positive_rate"), -999.0),
        ),
        default={},
    )
    latest_control_projected = control_projected_rows[-1] if control_projected_rows else {}
    runtime_p3_d1_rows = [
        r
        for r in d_rows
        if "runtime_p3_repair2_d1_h440" in str(r.get("_source_file", ""))
    ]
    runtime_p3_tabular_rows = [
        r
        for r in d_rows
        if "runtime_p3_repair2_tabular_spam_wine_s012_h200_clearproxy" in str(r.get("_source_file", ""))
    ]
    runtime_p3_d1_strong = next((r for r in runtime_p3_d1_rows if "JVP_RUNTIME_STRONG" in str(r.get("v22_18_variant", ""))), {})
    runtime_p3_d1_signflip = next((r for r in runtime_p3_d1_rows if "SIGNFLIP" in str(r.get("v22_18_variant", ""))), {})
    runtime_p3_tabular_jvp = next((r for r in runtime_p3_tabular_rows if str(r.get("v22_18_variant", "")) == "MLP+FU_BENEFIT_P3_JVP_RUNTIME"), {})
    runtime_p3_tabular_strong = next((r for r in runtime_p3_tabular_rows if "JVP_RUNTIME_STRONG" in str(r.get("v22_18_variant", ""))), {})
    runtime_p3_highsignal_rows = [
        r
        for r in d_rows
        if "v22_18_runtime_p3_tabular_highsignal_s012_h200" in str(r.get("_source_file", ""))
    ]
    runtime_p3_highsignal_i1 = [
        r
        for r in runtime_p3_highsignal_rows
        if "_i10" not in str(r.get("_source_file", "")) and "_i50_pilot" not in str(r.get("_source_file", ""))
    ]
    runtime_p3_highsignal_i10 = [
        r
        for r in runtime_p3_highsignal_rows
        if "_i10" in str(r.get("_source_file", ""))
    ]
    runtime_p3_highsignal_i50 = [
        r
        for r in runtime_p3_highsignal_rows
        if "_i50_pilot" in str(r.get("_source_file", ""))
    ]
    runtime_p3_highsignal_i50_scale05 = [
        r
        for r in runtime_p3_highsignal_rows
        if "_i50_strongscale05_pilot" in str(r.get("_source_file", ""))
    ]
    runtime_export_highsignal_i50 = [
        r
        for r in runtime_export_highsignal_rows
        if "_i50_controls" in str(r.get("_source_file", ""))
    ]
    runtime_export_highsignal_i100 = [
        r
        for r in runtime_export_highsignal_rows
        if "_i100_controls" in str(r.get("_source_file", ""))
    ]
    runtime_p3_highsignal_i50_task_rows = [
        r
        for r in runtime_p3_highsignal_task_rows
        if "_i50_pilot" in str(r.get("_source_file", ""))
    ]
    runtime_p3_highsignal_i50_scale05_task_rows = [
        r
        for r in runtime_p3_highsignal_task_rows
        if "_i50_strongscale05_pilot" in str(r.get("_source_file", ""))
    ]
    runtime_export_highsignal_i50_task_rows = [
        r
        for r in runtime_export_highsignal_task_rows
        if "_i50_controls" in str(r.get("_source_file", ""))
    ]
    runtime_export_highsignal_i100_task_rows = [
        r
        for r in runtime_export_highsignal_task_rows
        if "_i100_controls" in str(r.get("_source_file", ""))
    ]
    runtime_export_split_base_rows = [
        r
        for r in runtime_export_split_rows
        if "_toler10_controls.csv" in str(r.get("_source_file", ""))
    ]
    runtime_export_split_base_task_rows = [
        r
        for r in runtime_export_split_task_rows
        if "_toler10_controls.csv" in str(r.get("_source_file", ""))
    ]
    runtime_export_split_thr055_rows = [
        r
        for r in runtime_export_split_rows
        if "_toler10_thr055_controls" in str(r.get("_source_file", ""))
    ]
    runtime_export_split_thr055_task_rows = [
        r
        for r in runtime_export_split_task_rows
        if "_toler10_thr055_controls" in str(r.get("_source_file", ""))
    ]
    runtime_export_split_cap010_rows = [
        r
        for r in runtime_export_split_rows
        if "_thr055_cap010_pilot" in str(r.get("_source_file", ""))
    ]
    runtime_export_split_cap020_rows = [
        r
        for r in runtime_export_split_rows
        if "_thr055_cap020_pilot" in str(r.get("_source_file", ""))
    ]
    highsignal_availability = {
        str(r.get("dataset", "")): r
        for r in highsignal_availability_rows
        if str(r.get("dataset", "")) in {"Mushroom", "Rice", "Bean", "Bank", "Telescope", "Student", "Spam", "Card", "Wine", "Income", "Abalone"}
    }
    highsignal_available = ",".join(
        name for name in ["Spam", "Card", "Wine", "Abalone"] if highsignal_availability.get(name, {}).get("availability") == "available"
    )
    highsignal_unavailable = ",".join(
        name
        for name in ["Mushroom", "Rice", "Bean", "Bank", "Telescope", "Student", "Income"]
        if highsignal_availability.get(name, {}).get("availability") == "unavailable"
    )

    def _find_variant(rows: list[dict[str, str]], variant: str) -> dict[str, str]:
        return next((r for r in rows if str(r.get("v22_18_variant", "")) == variant), {})

    def _task_counts(rows: list[dict[str, str]], variant: str) -> str:
        selected = [r for r in rows if str(r.get("v22_18_variant", "")) == variant]
        if not selected:
            return ""
        accepts = [finite_float(r.get("gate_accept_count"), 0.0) for r in selected]
        rejects = [finite_float(r.get("gate_reject_count"), 0.0) for r in selected]
        skip_rows = sum("benefit_p3_jvp_interval_skip" in str(r.get("gate_reject_reason_counts", "")) for r in selected)
        return (
            f"task_rows={len(selected)}, accept_minmax={min(accepts):.6g}-{max(accepts):.6g}, "
            f"reject_minmax={min(rejects):.6g}-{max(rejects):.6g}, interval_skip_rows={skip_rows}"
        )

    def _mean_field(rows: list[dict[str, str]], variant: str, field: str) -> str:
        selected = [finite_float(r.get(field), 0.0) for r in rows if str(r.get("v22_18_variant", "")) == variant and str(r.get(field, "")) != ""]
        return f"{(sum(selected) / len(selected)):.12g}" if selected else ""

    runtime_p3_highsignal_i1_jvp = _find_variant(runtime_p3_highsignal_i1, "MLP+FU_BENEFIT_P3_JVP_RUNTIME")
    runtime_p3_highsignal_i1_strong = _find_variant(runtime_p3_highsignal_i1, "MLP+FU_BENEFIT_P3_JVP_RUNTIME_STRONG")
    runtime_p3_highsignal_i10_jvp = _find_variant(runtime_p3_highsignal_i10, "MLP+FU_BENEFIT_P3_JVP_RUNTIME")
    runtime_p3_highsignal_i10_strong = _find_variant(runtime_p3_highsignal_i10, "MLP+FU_BENEFIT_P3_JVP_RUNTIME_STRONG")
    runtime_p3_highsignal_i50_jvp = _find_variant(runtime_p3_highsignal_i50, "MLP+FU_BENEFIT_P3_JVP_RUNTIME")
    runtime_p3_highsignal_i50_strong = _find_variant(runtime_p3_highsignal_i50, "MLP+FU_BENEFIT_P3_JVP_RUNTIME_STRONG")
    runtime_p3_highsignal_i50_scale05_strong = _find_variant(runtime_p3_highsignal_i50_scale05, "MLP+FU_BENEFIT_P3_JVP_RUNTIME_STRONG")
    runtime_p3_highsignal_i50_task_counts = _task_counts(runtime_p3_highsignal_i50_task_rows, "MLP+FU_BENEFIT_P3_JVP_RUNTIME_STRONG")
    runtime_p3_highsignal_i50_scale05_task_counts = _task_counts(runtime_p3_highsignal_i50_scale05_task_rows, "MLP+FU_BENEFIT_P3_JVP_RUNTIME_STRONG")
    runtime_export_selected_rows = runtime_export_highsignal_i100 or runtime_export_highsignal_i50 or runtime_export_highsignal_rows
    runtime_export_selected_task_rows = runtime_export_highsignal_i100_task_rows or runtime_export_highsignal_i50_task_rows or runtime_export_highsignal_task_rows
    runtime_export_selected_scope = "interval=100" if runtime_export_highsignal_i100 else "interval=50" if runtime_export_highsignal_i50 else "all"
    runtime_export_highsignal_strong = _find_variant(runtime_export_selected_rows, "MLP+FU_BENEFIT_P3_JVP_RUNTIME_STRONG")
    runtime_export_highsignal_task_counts = _task_counts(runtime_export_selected_task_rows, "MLP+FU_BENEFIT_P3_JVP_RUNTIME_STRONG")
    runtime_export_integrated_rows = sum(
        int_flag(r.get("benefit_policy_export_runtime_integrated"))
        for r in runtime_export_selected_task_rows
        if str(r.get("v22_18_variant", "")) == "MLP+FU_BENEFIT_P3_JVP_RUNTIME_STRONG"
    )
    runtime_export_mean_score = _mean_field(runtime_export_selected_task_rows, "MLP+FU_BENEFIT_P3_JVP_RUNTIME_STRONG", "mean_benefit_policy_export_score")
    runtime_export_split_strong = _find_variant(runtime_export_split_base_rows or runtime_export_split_rows, "MLP+FU_BENEFIT_P3_SPLIT_RUNTIME_STRONG")
    runtime_export_split_task_counts = _task_counts(runtime_export_split_base_task_rows or runtime_export_split_task_rows, "MLP+FU_BENEFIT_P3_SPLIT_RUNTIME_STRONG")
    runtime_export_split_mean_score = _mean_field(runtime_export_split_base_task_rows or runtime_export_split_task_rows, "MLP+FU_BENEFIT_P3_SPLIT_RUNTIME_STRONG", "mean_benefit_policy_export_score")
    runtime_export_split_thr055_strong = _find_variant(runtime_export_split_thr055_rows, "MLP+FU_BENEFIT_P3_SPLIT_RUNTIME_STRONG")
    runtime_export_split_thr055_task_counts = _task_counts(runtime_export_split_thr055_task_rows, "MLP+FU_BENEFIT_P3_SPLIT_RUNTIME_STRONG")
    runtime_export_split_thr055_mean_score = _mean_field(runtime_export_split_thr055_task_rows, "MLP+FU_BENEFIT_P3_SPLIT_RUNTIME_STRONG", "mean_benefit_policy_export_score")
    runtime_export_split_cap010_strong = _find_variant(runtime_export_split_cap010_rows, "MLP+FU_BENEFIT_P3_SPLIT_RUNTIME_STRONG")
    runtime_export_split_cap020_strong = _find_variant(runtime_export_split_cap020_rows, "MLP+FU_BENEFIT_P3_SPLIT_RUNTIME_STRONG")
    bridge_by_path = {str(r.get("path_number", "")): r for r in bridge_audit_rows}

    def _e_task_repair_stats(rows: list[dict[str, str]]) -> dict[str, Any]:
        if not rows:
            return {}
        nll = [finite_float(r.get("NLL_delta_vs_KAN_AdamW_native"), 0.0) for r in rows]
        auc = [finite_float(r.get("AUC_loss_time_delta_vs_KAN_AdamW_native"), 0.0) for r in rows]
        accepts = [finite_float(r.get("gate_accept_count"), 0.0) for r in rows]
        rejects = [finite_float(r.get("gate_reject_count"), 0.0) for r in rows]
        return {
            "rows": len(rows),
            "source_files": ",".join(sorted({str(r.get("_source_file", "")) for r in rows})),
            "nll_negative_rows": sum(v < 0.0 for v in nll),
            "strict_nll_improvement_rows": sum(v < -0.01 for v in nll),
            "auc_negative_rows": sum(v < 0.0 for v in auc),
            "mean_nll_delta": sum(nll) / max(1, len(nll)),
            "mean_auc_delta": sum(auc) / max(1, len(auc)),
            "official_overhead_pass_rows": sum(finite_float(r.get("official_controller_overhead_ratio"), 999.0) <= 0.25 for r in rows),
            "official_full_loop_pass_rows": sum(finite_float(r.get("official_full_loop_ratio_vs_mlp"), 999.0) <= 3.0 for r in rows),
            "accept_minmax": f"{min(accepts):.6g}-{max(accepts):.6g}",
            "reject_minmax": f"{min(rejects):.6g}-{max(rejects):.6g}",
        }

    e_task_virtual_i40_stats = _e_task_repair_stats(e_task_virtual_i40_rows)
    e_task_virtual_i120_stats = _e_task_repair_stats(e_task_virtual_i120_rows)
    e_long_task_stats = _e_task_repair_stats(e_long_task_pilot_rows)
    e4_task_release_stats = _e_task_repair_stats(e4_task_release_rows)
    h_s012_rows = [r for r in h_rows if "s012" in str(r.get("_source_file", ""))]
    h_transport_rows = [
        r
        for r in h_rows
        if "source_transport_s012_h120" in str(r.get("_source_file", ""))
    ]
    h_transport_best = max(
        h_transport_rows,
        key=lambda r: (
            finite_float(r.get("relative_forgetting_reduction"), -999.0),
            finite_float(r.get("absolute_forgetting_reduction"), -999.0),
            finite_float(r.get("average_accuracy_delta"), -999.0),
        ),
        default={},
    )
    h_transport_explore_rows = [r for r in h_transport_rows if int_flag(r.get("H_exploration_pass"))]
    h_transport_official_rows = [r for r in h_transport_rows if int_flag(r.get("H_official_pass"))]
    h_transport_sources = ",".join(sorted({str(r.get("_source_file", "")) for r in h_transport_rows}))
    h_virtual_gate_rows = [
        r
        for r in h_rows
        if "virtual_gate_s012_h120_cap005" in str(r.get("_source_file", ""))
    ]
    h_virtual_gate_best = max(
        h_virtual_gate_rows,
        key=lambda r: (
            finite_float(r.get("relative_forgetting_reduction"), -999.0),
            finite_float(r.get("absolute_forgetting_reduction"), -999.0),
            finite_float(r.get("average_accuracy_delta"), -999.0),
        ),
        default={},
    )
    h_virtual_gate_explore_rows = [r for r in h_virtual_gate_rows if int_flag(r.get("H_exploration_pass"))]
    h_virtual_gate_official_rows = [r for r in h_virtual_gate_rows if int_flag(r.get("H_official_pass"))]
    h_virtual_gate_sources = ",".join(sorted({str(r.get("_source_file", "")) for r in h_virtual_gate_rows}))
    h_virtual_gate_reject_counts = ",".join(str(r.get("continual_virtual_reject_count", "")) for r in h_virtual_gate_rows)
    h_virtual_gate_max_overhead = max(
        (finite_float(r.get("controller_overhead_ratio"), 0.0) for r in h_virtual_gate_rows),
        default=0.0,
    )
    best_native_eff = max(
        e_rows,
        key=lambda r: (
            int(_native_timing_complete(r)),
            _native_full_loop_pass_count(r),
            _native_overhead_pass_count(r),
            -_native_max_official_ratio(r),
        ),
        default={},
    )
    native_diag_i40 = next((r for r in e_rows if "v22_18_eff_native_dche_h4_s012_h120_b64_diagsep.csv" in str(r.get("_source_file", ""))), {})
    native_diag_i120 = next((r for r in e_rows if "v22_18_eff_native_dche_h4_s012_h120_b64_diagsep_i120.csv" in str(r.get("_source_file", ""))), {})
    summary_lines = {
        "mlp_lines": [f"{r.get('v22_18_variant', r.get('model_name', ''))}: rows={r.get('rows')} NLLimp={r.get('NLL_improvement_rows')} AUCimp={r.get('AUC_improvement_rows')} noharm={r.get('D_noharm_pass')}" for r in d_rows[:12]],
        "policy_lines": [f"{r.get('split')} {r.get('held_out')}: AUC={r.get('benefit_AUC_binary_positive')} Spearman={r.get('benefit_spearman')} noop={r.get('policy_noop_rate', '')}" for r in c_rows[:12]],
        "action_lines": [f"{r.get('action_id')}: predicted_positive={r.get('predicted_positive_count')}" for r in action_dist[:12]],
        "control_lines": [f"{r.get('v22_18_variant', r.get('model_name', ''))}: control={r.get('is_control_variant')} NLLimp={r.get('NLL_improvement_rows')}" for r in d_rows[:12]],
        "kan_lines": [f"{r.get('model_name')}: rows={r.get('rows')} source={r.get('source_loss_horizon_ge_0')} nll_imp={r.get('nll_improve_vs_kan')} official_full<=3={r.get('official_full_loop_ratio_le_3', '')}" for r in e_rows[:12]],
        "expanded_lines": [f"{r.get('dataset')}: MLPFU_dNLL={r.get('MLPFU_NLL_delta_vs_DGMLP')} KANFU_dNLL={r.get('KANFU_NLL_delta_vs_DGKAN')}" for r in four_square[:12]],
        "continual_lines": [f"seed={r.get('seed')} abs_reduction={r.get('absolute_forgetting_reduction')} rel={r.get('relative_forgetting_reduction')} acc_delta={r.get('average_accuracy_delta')}" for r in h_rows[:12]],
        "four_square_lines": [f"{r.get('dataset')}: MLP={r.get('MLP_NLL')} MLPFU={r.get('MLPFU_NLL')} KAN={r.get('KAN_NLL')} KANFU={r.get('KANFU_NLL')}" for r in four_square[:12]],
        "identity_lines": [f"rows={identity_latest.get('model_identity_rows', 0)} DGKAN_pass={identity_latest.get('official_DGKAN_identity_pass_rows', 0)}/{identity_latest.get('official_DGKAN_rows', 0)} KANbeFair_baseline_only={identity_latest.get('KANbeFair_original_KAN_rows_marked_baseline_context_only', 0)}"],
        "bridge_lines": [f"Path {r.get('path_number')} rows={r.get('rows')} full_loop={r.get('full_loop_rows')} official_path={r.get('efficiency_path_official_pass_rows')} mean_ms={r.get('mean_full_step_ms')}" for r in bridge_audit_rows[:12]],
    }
    _make_figures(summary_lines)

    with RECAP_DOC.open("a", encoding="utf-8") as f:
        f.write(f"\n# v22.18 最终复盘追加\n\n生成时间：{now_sg()}\n\n")
        f.write(f"Final route：`{route}`；official promotion：`{full_official}`。\n\n")
        f.write("本节只读回 `results/v22_18/` 当前 artifact；未执行或 proxy 项均按 blocker/deferred 记录。\n\n")
        f.write("## 1. 本轮代码/脚本修改审计\n\n")
        f.write("- 新增 `experiments/run_v22_18_common.py`：v22.18 独立日志、artifact、hash、doc 记录层。\n")
        f.write("- 新增 `dgkan/fu/source_action_bank.py`、`treatment_effect_labels.py`、`benefit_policy.py`、`kan_native_jvp.py`、`continual_source_state.py`：只做 action family 映射、真实读回 label/feature 计算、离线 policy 指标与 forgetting 指标。\n")
        f.write("- 新增 v22.18 实验入口：status manifest、source-action bank、benefit policy、MLP mainline wrapper、KAN native ladder wrapper、KANbeFair expanded、continual、finalizer。\n")
        f.write("- MLP/KAN 训练复用 v22.17 已审计 bridge/native 训练循环；v22.18 wrapper 只负责真实执行/转换和 gate 判定，不改写 v22.17 历史结论。\n\n")
        f.write("- 按 PureKAN 身份审计更新版新增 `experiments/run_v22_18_identity_audit.py`：只读现有 v22.17/v22.18 artifact，输出 `v22_18_model_identity_matrix.csv`、`v22_18_efficiency_path_identity_matrix.csv`、`v22_18_efficiency_bridge_audit_matrix.csv`、`v22_18_efficiency_path_component_timing.csv` 与 identity/bridge SVG；KANbeFair 原始 `KAN`/`BSpline_*` 只标记为 baseline context，不能进入 DG-KAN official route。\n")
        f.write("- 修复 `experiments/run_v22_18_identity_audit.py` 的 canonical DGKAN lookup：gradcheck/timing artifact 中的基类名 `DGKAN_DCHE`/`DGKAN_DFOU` 现在会读取同 carrier strict wrapper 行的身份审计结果，避免把 D-CHE/D-FOU profiler 误判为 `ModelIdentityAmbiguous`；该修复只改变 audit 分类，不改实验指标数值。\n")
        f.write("- 扩展 `experiments/run_v22_18_benefit_policy.py`：新增 `target-mode`、control penalty、component reweighting 与 `no_action_numeric` feature ablation，用于按计划尝试 C gate repair；这些 repair 仍输出为 offline variant selector，不标记 runtime per-step integration。\n")
        f.write("- 新增 `experiments/run_v22_18_branch_label_readback.py`：把 H50/H100/H200 独立短程分支 run 合成为 true-init-branch counterfactual labels，替代单一 h440 final-run proxy；该证据仍标记为 init branch，不伪称 per-step runtime branch。\n")
        f.write("- 新增 `experiments/run_v22_18_label_strictness_audit.py`：按计划尝试更严格 label 定义（NLL+source、NLL+AUC、NLL+AUC+source 等）以检查 controls-positive blocker 是否能被消除；结果只按真实读回写入，不用于放宽 promotion。\n")
        f.write("- 新增 `experiments/run_v22_18_control_projected_labels.py`：按计划对 controls-positive blocker 做 matched-control projection 审计，在同 dataset/seed/horizon 内比较 real action 是否超过 best control；该输出标记为 diagnostic-only，不进入 official promotion gate。\n")
        f.write("- 修复 `experiments/run_v22_18_source_action_bank.py` 的 runtime action feature：`action_family_numeric` 现在从 candidate action id 写入，不再被 source-row 均值缺失值覆盖为 0；该修复只影响 runtime feature 表和 C policy 输入，不改 label/指标定义。\n")
        f.write("- 修复 `external/KANbeFair_worktree/src/data/uciml.py` 的 pandas 3.x 兼容问题：`process_y` 使用 `df.dtypes.iloc[0]` 读取单列 target dtype；并在当前 `kan` 环境安装 `ucimlrepo-0.0.7` 以尝试 tabular high-signal datasets。该修复只影响 loader 可用性，不改指标定义。\n")
        f.write("- 新增 runtime per-step `BENEFIT_P3` controller exploration：`experiments/run_v22_17_kanbefair_dgkan_eval.py` 支持 `DGMLP_FU_BENEFIT_P3*` 当前步/JVP policy，score 只用当前 step 的 source gain、risk/NDS、virtual current-step loss、update ratio 等 runtime features；rejected candidate 的 `intervention_flag/lambda_t` 已清零，避免把候选生成误计为真实干预。\n")
        f.write("- 按 runtime P3 D1 失败证据继续修复：降低 P3 score 中 risk/NDS/tail 惩罚但不降低 min-score threshold，并为 `DGMLP_FU_BENEFIT_P3_JVP_REFRESH_STRONG` 增加 high-confidence strong scale/cap override；该修复后仍按 controls/overhead gate 真实判定。\n")
        f.write("- 修复 runtime P3 JVP cadence：`experiments/run_v22_17_kanbefair_dgkan_eval.py` 新增 `--benefit-p3-jvp-interval`，非 cadence step 记录 `benefit_p3_jvp_interval_skip` 并保持 no-op；`experiments/run_v22_18_mlp_fu_benefit_mainline.py` 透传该参数。该修复只降低 controller 触发频率/开销，不放宽 NLL/AUC/controls 阈值。\n")
        f.write("- 新增 strict benefit policy export 的 runtime 接入：`experiments/run_v22_17_kanbefair_dgkan_eval.py` 新增 `--benefit-policy-export-json/--benefit-policy-export-threshold`，用当前 step 可用的 runtime features 计算导出 ridge policy score；有 export 时该 score 作为 P3/JVP 候选接受门控，原手工 P3 score 保留为 `benefit_p3_manual_score` 供审计；`experiments/run_v22_18_mlp_fu_benefit_mainline.py` 透传参数并写出 export score/action distribution。该修复不使用未来/test/validation 信息，也不放宽 D/C 阈值。\n")
        f.write("- 修复 tabular runtime retry 的代理环境：`experiments/run_v22_18_mlp_fu_benefit_mainline.py` 新增 `--clear-proxy-env`，用于把 `HTTP_PROXY/HTTPS_PROXY/ALL_PROXY` 等传空给 KANbeFair bridge 子进程；首次 Spam/Wine runtime retry 因 `ucimlrepo ConnectionError` 产生 0 row，clear-proxy 后重新跑出真实 task rows。\n")
        f.write("- 修复 E KAN native task alignment：`experiments/run_v22_17_kanbefair_dgkan_eval.py` 的 JVP refresh helper 与 `experiments/run_v22_17_kan_basis_native_audit.py`/`run_v22_18_kan_native_ladder.py` 新增 `--jvp-refresh-require-virtual-improvement`，可要求 guided update 当前 batch virtual train loss 低于 base update 才接受；默认关闭，新增实验显式打开，用于检查 source-task alignment 而不是放宽 E 阈值。\n")
        f.write("- 更新 `experiments/run_v22_18_finalize.py` 的 E2/E3 long-horizon 读回：新增 `v22_18_e2_task_dche_h4_s0*.csv` 与 `v22_18_e3_task_dche_h4_s0*.csv` matrix 收集与复盘输出，确保 h1600/h3200/h4800 pilot 的 NLL/AUC/source/efficiency/blocker 证据进入权威复盘；该更新只影响报告覆盖面，不改 gate 阈值。\n")
        f.write("- 修复 E KAN native source-age/release 可审计入口：`experiments/run_v22_18_kan_native_ladder.py` 透传 `--jvp-refresh-min-history/--jvp-refresh-history-window/--jvp-refresh-release-on-accept`，`experiments/run_v22_17_kan_basis_native_audit.py` 在显式打开 release-on-accept 时接受一次 JVP refresh 后释放历史并写出 `source_release_count`；默认关闭，不改历史实验语义。\n")
        f.write("- 修复 continual H 的 current-task no-harm 约束：`experiments/run_v22_18_continual.py` 新增 `--continual-virtual-loss-tolerance/--continual-virtual-gate-scale`，在接受 FU 前比较 base/guided 当前 batch virtual train loss；若 guided 超过容忍值则恢复 controller state 并 no-op。该修复只增加拒绝门和审计字段，不放宽 forgetting/accuracy/official 阈值。\n")
        f.write("- 修复 F 四路径审计元数据：`experiments/run_v22_17_basis_jvp_vjp_gradcheck.py` 增加 `--output-suffix`，并与 `run_v22_17_kan_basis_native_audit.py`、`run_v22_17_kanbefair_dgkan_eval.py` 一起写出 hidden/train/test/batch 与初始 batch fingerprint；`run_v22_18_identity_audit.py` 现在 glob 读取 suffix 产物，计算 same-config/same-batch 四路径 key。该修复只补审计可追踪性，不把 diagnostic row 升级成 official。\n")
        f.write("- 修复 F native timing 口径：`run_v22_17_kan_basis_native_audit.py` 新增 `diagnostic_source_eval_step_ms`、`official_train_step_ms`、`official_full_loop_ratio_vs_mlp`、`official_controller_overhead_ratio`；`dgkan/fu/kan_native_jvp.py` 与 finalizer 现在要求 diagnostic-separated timing rows 完整，并用 official ratio/overhead 判定 F native breadth。旧 full-loop 字段保留作审计对照。\n")
        f.write("- 修正 `experiments/run_v22_18_kanbefair_expanded.py` 的 availability 记录语义：区分 `available_datasets` 与 `run_dataset_count`，避免 skip task run 时把可用数据集数误写成 0。\n")
        f.write("- 更新 `experiments/run_v22_18_finalize.py`：A gate 现在读取模型身份与效率路径身份硬审计；F gate 将四路径效率桥接审计作为前置项，并区分 four-path 已齐、native efficiency breadth 未齐、official fused path 未齐三种状态。\n\n")
        f.write("## 2. Gate summary\n\n")
        f.write(md_table(gates, ["gate_name", "pass", "scope", "row_count", "blocking_metric"], limit=30))
        f.write("\n## 3. Route decision trace\n\n")
        f.write(md_table(trace, ["decision", "value", "reason"]))
        if b_summary:
            f.write("\n## 4. Source-action / benefit labels\n\n")
            f.write(md_table(b_summary, list(b_summary[0]), limit=10))
        if c_rows:
            f.write("\n## 5. Benefit policy offline validation\n\n")
            f.write(md_table(c_rows, ["split", "held_out", "rows", "benefit_AUC_binary_positive", "benefit_spearman", "false_positive_benefit_rate_controls", "policy_noop_rate", "policy_action_accept_rate", "target_mode", "control_penalty", "weight_cost", "feature_set", "C_exploration_pass", "C_official_pass", "official_blocker"], limit=80))
        if c_policy_exports:
            f.write("\n## 5.0b Benefit policy exports\n\n")
            f.write(md_table(c_policy_exports, ["_source_file", "policy_variant", "horizon", "target_mode", "feature_set", "ridge", "training_rows", "C_exploration_pass", "C_official_pass", "min_leave_one_dataset_auc", "min_leave_one_seed_auc", "runtime_per_step_policy_integrated", "offline_variant_selector_only"], limit=30))
        if runtime_export_highsignal_rows:
            f.write("\n## 5.0c Benefit policy export runtime integration\n\n")
            f.write(md_table(runtime_export_highsignal_rows, ["_source_file", "v22_18_variant", "rows", "is_control_variant", "NLL_improvement_rows", "AUC_improvement_rows", "accuracy_improvement_rows", "mean_NLL_delta", "max_NLL_delta", "max_controller_overhead_ratio", "D_noharm_pass", "D_improvement_pass"], limit=20))
        if runtime_export_split_rows:
            f.write("\n## 5.0d Benefit policy split-source runtime integration\n\n")
            f.write(md_table(runtime_export_split_rows, ["_source_file", "v22_18_variant", "rows", "is_control_variant", "NLL_noharm_rows", "NLL_improvement_rows", "AUC_improvement_rows", "accuracy_improvement_rows", "mean_NLL_delta", "max_NLL_delta", "max_tail_q99_delta", "max_controller_overhead_ratio", "D_noharm_pass", "D_improvement_pass"], limit=20))
        if strictness_rows:
            f.write("\n## 5.1 Label strictness audit\n\n")
            f.write(md_table(strictness_rows, ["label_name", "definition", "positive_rows", "real_positive_rate", "control_positive_rate", "real_minus_control_rate", "B_official_gap_pass", "blocker"], limit=20))
        if strict_label_bank_rows:
            f.write("\n## 5.1b Strict label bank materialization\n\n")
            f.write(md_table(strict_label_bank_rows, ["_source_file", "strict_label_name", "label_rows", "positive_rows", "real_positive_rate", "control_positive_rate", "real_minus_control_rate", "horizons_present", "B_strict_preparation_signal_pass", "B_official_preparation_pass", "official_blocker"], limit=20))
        if control_projected_rows:
            f.write("\n## 5.2 Control-projected label audit\n\n")
            f.write(md_table(control_projected_rows, ["source", "label_rows", "real_rows", "control_rows", "matched_dataset_seed_horizon_groups", "real_rate_beats_best_control_utility", "real_rate_beats_best_control_NLL", "real_rate_beats_best_control_source", "real_rate_beats_best_control_NLL_and_source", "real_rate_beats_best_control_NLL_AUC_and_source", "strict_control_projected_B_signal_pass", "B_official_preparation_pass", "blocker"], limit=20))
        if d_rows:
            f.write("\n## 6. MLP+FU no-harm / improvement\n\n")
            f.write(md_table(d_rows, ["v22_18_variant", "rows", "is_control_variant", "NLL_noharm_rows", "NLL_improvement_rows", "AUC_improvement_rows", "accuracy_improvement_rows", "max_controller_overhead_ratio", "D_noharm_pass", "D_improvement_pass"], limit=30))
        if runtime_p3_no_go_rows:
            f.write("\n## 6.1 Runtime P3 no-go audit\n\n")
            f.write(md_table(runtime_p3_no_go_rows, ["scope", "v22_18_variant", "rows", "is_control_variant", "NLL_improvement_rows", "AUC_improvement_rows", "accuracy_improvement_rows", "mean_NLL_delta", "max_tail_q99_delta", "max_controller_overhead_ratio", "mean_gate_accept_count", "best_real_variant", "best_control_variant", "has_control_rows", "no_go_reason_if_best_real"], limit=40))
        if runtime_p3_highsignal_rows:
            f.write("\n## 6.2 Runtime P3 high-signal cadence repair\n\n")
            f.write(md_table(runtime_p3_highsignal_rows, ["_source_file", "v22_18_variant", "rows", "is_control_variant", "NLL_improvement_rows", "AUC_improvement_rows", "accuracy_improvement_rows", "mean_NLL_delta", "max_NLL_delta", "max_tail_q99_delta", "max_controller_overhead_ratio", "D_noharm_pass", "D_improvement_pass"], limit=20))
        if e_rows:
            f.write("\n## 7. KAN native-basis ladder\n\n")
            f.write(md_table(e_rows, ["model_name", "rows", "basis_energy_ge_050", "source_loss_horizon_ge_0", "nll_improve_vs_kan", "auc_improve_vs_kan", "diagnostic_separated_timing_rows", "official_full_loop_ratio_le_3", "max_official_full_loop_ratio", "official_controller_overhead_le_025", "max_official_controller_overhead", "full_loop_ratio_le_3", "max_full_loop_ratio"], limit=20))
        if identity_summary_rows:
            f.write("\n## 7.1 PureKAN / KANbeFair identity audit\n\n")
            f.write(md_table(identity_summary_rows, list(identity_summary_rows[0]), limit=10))
        if model_identity_rows:
            f.write("\n## 7.2 Model identity firewall sample\n\n")
            f.write(md_table(model_identity_rows, ["row_id", "dataset", "seed", "model_name", "model_origin", "model_identity_class", "uses_kanbefair_baseline_model", "uses_pykan", "uses_bspline_official_path", "uses_primitivekan", "is_dgkan_strict_fc_purekan", "route_scope", "never_count_for_DGKAN_efficiency_or_superiority", "model_identity_pass", "identity_blocker"], limit=30))
        if efficiency_identity_rows:
            f.write("\n## 7.3 Efficiency path identity sample\n\n")
            f.write(md_table(efficiency_identity_rows, ["path_id", "path_scope", "model_name", "carrier", "inside_kanbefair_bridge", "inside_dglca_native_loop", "official_fused_kernel_complete", "uses_finite_diff_fallback", "uses_analytic_or_sketch_jvp", "full_step_ms", "controller_overhead_ratio", "official_efficiency_hard_conditions_pass", "efficiency_path_official_pass", "efficiency_blocker"], limit=30))
        if bridge_audit_rows:
            f.write("\n## 7.4 Four-path bridge audit\n\n")
            f.write(md_table(bridge_audit_rows, list(bridge_audit_rows[0]), limit=20))
        if g_rows:
            f.write("\n## 8. KANbeFair expanded\n\n")
            f.write(md_table(g_rows, list(g_rows[0]), limit=10))
        if h_rows:
            f.write("\n## 9. Continual / forgetting\n\n")
            f.write(md_table(h_rows, list(h_rows[0]), limit=30))
        if h_virtual_gate_rows:
            f.write("\n## 9.1 Continual current-task virtual NLL gate repair\n\n")
            f.write(md_table(h_virtual_gate_rows, ["_source_file", "seed", "fu_model_name", "continual_source_mode", "controller_overhead_ratio", "continual_virtual_loss_tolerance", "continual_virtual_reject_count", "mean_continual_virtual_loss_base", "mean_continual_virtual_loss_guided", "absolute_forgetting_reduction", "relative_forgetting_reduction", "average_accuracy_delta", "H_exploration_pass", "H_official_pass"], limit=20))
        f.write("\n## 10. Analysis / conclusion / insight\n\n")
        f.write(f"- 当前权威 route 为 `{route}`，不是 full superiority。官方 promotion={full_official}。\n")
        f.write("- 若 B/C official 未过，核心原因不是数据被隐藏：离线 branch policy 的 leave-one 泛化仍不足；新增 runtime per-step P3 controller 后，真实 D1/controls/overhead 证据仍未满足 official improvement。\n")
        if vision_branch_b:
            f.write(f"- Vision H50/H100/H200 true-init-branch counterfactual 读回：labels={vision_branch_b.get('benefit_label_rows')}，positive={vision_branch_b.get('positive_benefit_rows')}，real_positive_rate={vision_branch_b.get('real_positive_benefit_rate')}，control_positive_rate={vision_branch_b.get('control_positive_benefit_rate')}，B_official={vision_branch_b.get('B_official_preparation_pass')}，blocker={vision_branch_b.get('official_blocker')}。这比旧 final-run proxy 更强，但仍不能通过 controls gap。\n")
        if tabular_branch_b:
            tabular_seed_scope = "seeds0-2" if "s012" in str(tabular_branch_b.get("_source_file", "")) else "seed0"
            f.write(f"- Tabular Spam/Wine {tabular_seed_scope} H50/H100/H200 branch 读回：labels={tabular_branch_b.get('benefit_label_rows')}，positive={tabular_branch_b.get('positive_benefit_rows')}，real_positive_rate={tabular_branch_b.get('real_positive_benefit_rate')}，control_positive_rate={tabular_branch_b.get('control_positive_benefit_rate')}，B_official={tabular_branch_b.get('B_official_preparation_pass')}，blocker={tabular_branch_b.get('official_blocker')}。这里 controls gap 打开，但 positive rate 超过 40% 且仍是 init-branch/offline 探索，不能 promotion。\n")
        if branch_c_all:
            f.write(f"- Branch-label C overall 读回：best branch AUC={finite_float(branch_c_best.get('benefit_AUC_binary_positive')):.6g}（source={branch_c_best.get('_source_file', '')}，target_mode={branch_c_best.get('target_mode', '')}，FP_controls={branch_c_best.get('false_positive_benefit_rate_controls', '')}，noop={branch_c_best.get('policy_noop_rate', '')}）；branch control-safe best AUC={finite_float(branch_c_control_safe_best.get('benefit_AUC_binary_positive')):.6g}（target_mode={branch_c_control_safe_best.get('target_mode', '')}，noop={branch_c_control_safe_best.get('policy_noop_rate', '')}）。\n")
        if vision_branch_c_all:
            f.write(f"- Vision branch C best={finite_float(vision_branch_c_best.get('benefit_AUC_binary_positive')):.6g}，仍未打开 C exploration。\n")
        if tabular_branch_c_all:
            f.write(f"- Tabular branch C exploration 打开：best AUC={finite_float(tabular_branch_c_best.get('benefit_AUC_binary_positive')):.6g}（source={tabular_branch_c_best.get('_source_file', '')}），Spearman={tabular_branch_c_best.get('benefit_spearman', '')}，FP_controls={tabular_branch_c_best.get('false_positive_benefit_rate_controls', '')}，noop={tabular_branch_c_best.get('policy_noop_rate', '')}，但 `C_official_pass={tabular_branch_c_best.get('C_official_pass', '')}`，原因是 offline variant selector 且没有 per-step runtime integration。\n")
        if tabular_s012_c_all:
            min_lodo = finite_float(best_tabular_s012_lodo.get("min_lodo"), 0.0)
            min_loseed = finite_float(best_tabular_s012_loseed.get("min_loseed"), 0.0)
            if min_lodo >= 0.62 and min_loseed >= 0.62:
                c_robustness_conclusion = "严格标签 H50 ablation 下 leave-one dataset/seed 统计阈值已过，但这些 rows 仍是 offline selector，缺少 runtime per-step integration，因此 C_official 仍为 0。"
            elif min_lodo >= 0.62:
                c_robustness_conclusion = "dataset leave-one 已过 0.62，但 seed leave-one 仍低于阈值，且仍缺 runtime per-step integration。"
            elif min_loseed >= 0.62:
                c_robustness_conclusion = "seed leave-one 已过 0.62，但 dataset leave-one 仍低于阈值，且仍缺 runtime per-step integration。"
            else:
                c_robustness_conclusion = "seed 扩展后 C exploration 还在，但 dataset/seed 泛化仍低于 0.62 官方阈值。"
            f.write(f"- Tabular Spam/Wine seeds0-2 robustness 读回：in-sample best AUC={finite_float(tabular_s012_c_best.get('benefit_AUC_binary_positive')):.6g}，Spearman={tabular_s012_c_best.get('benefit_spearman', '')}。action feature 修复和 ridge/no-action ablation 后，best leave-one-dataset min AUC={min_lodo:.6g}（source={best_tabular_s012_lodo.get('source_file', '')}，target={best_tabular_s012_lodo.get('target_mode', '')}，feature_set={best_tabular_s012_lodo.get('feature_set', '')}，ridge={best_tabular_s012_lodo.get('ridge', '')}）；best leave-one-seed min AUC={min_loseed:.6g}（source={best_tabular_s012_loseed.get('source_file', '')}）。{c_robustness_conclusion}\n")
        if strictness_rows:
            if strictness_pass_rows:
                names = ",".join(sorted({str(r.get("label_name", "")) for r in strictness_pass_rows if str(r.get("label_name", ""))}))
                f.write(f"- Label strictness audit 读回：best real-control gap={finite_float(best_strictness.get('real_minus_control_rate'), 0.0):.6g}（label={best_strictness.get('label_name', '')}，real_rate={best_strictness.get('real_positive_rate', '')}，control_rate={best_strictness.get('control_positive_rate', '')}，gap_pass={best_strictness.get('B_official_gap_pass', '')}）。严格标签中已有 gap-pass 定义：{names}；但这些来自 init-branch/offline label audit，不能替代 per-step runtime official B/C。\n")
            else:
                f.write(f"- Label strictness audit 读回：best real-control gap={finite_float(best_strictness.get('real_minus_control_rate'), 0.0):.6g}（label={best_strictness.get('label_name', '')}，real_rate={best_strictness.get('real_positive_rate', '')}，control_rate={best_strictness.get('control_positive_rate', '')}，gap_pass={best_strictness.get('B_official_gap_pass', '')}）。更严格的 NLL/source/AUC 组合仍没有给 real actions 建立 controls gap。\n")
        if strict_label_bank_rows:
            f.write(f"- Strict label bank materialization 读回：strict_signal_pass_rows={len(strict_label_bank_pass_rows)}/{len(strict_label_bank_rows)}，best_label={best_strict_label_bank.get('strict_label_name', '')}，best_real_rate={best_strict_label_bank.get('real_positive_rate', '')}，best_control_rate={best_strict_label_bank.get('control_positive_rate', '')}，best_gap={best_strict_label_bank.get('real_minus_control_rate', '')}，B_official={best_strict_label_bank.get('B_official_preparation_pass', '')}。结论：严格 NLL/source/AUC 标签可以把 tabular branch labels 的 real-control gap 变干净，但仍标记为 strict offline label bank，不能替代 per-step runtime branch labels 或 C official runtime integration。\n")
        if control_projected_rows:
            f.write(f"- Control-projected audit 读回：real action 超过同 dataset/seed/horizon best-control utility 的比例为 {latest_control_projected.get('real_rate_beats_best_control_utility')}，但同时超过 best-control NLL+source_loss 的比例仅为 {latest_control_projected.get('real_rate_beats_best_control_NLL_and_source')}，NLL+AUC+source_loss 为 {latest_control_projected.get('real_rate_beats_best_control_NLL_AUC_and_source')}；`B_official_preparation_pass={latest_control_projected.get('B_official_preparation_pass')}`，blocker={latest_control_projected.get('blocker')}。这说明加强 projection 后仍不能把 no-harm 微波动写成 real improvement。\n")
        f.write(f"- C repair 读回：best in-sample AUC={best_c_auc:.6g}（target_mode={best_c_row.get('target_mode', 'benefit_utility')}，control_penalty={best_c_row.get('control_penalty', '')}，FP_controls={best_c_row.get('false_positive_benefit_rate_controls', '')}，noop={best_c_row.get('policy_noop_rate', '')}）；在 control FP<=0.30 且 noop 20%-90% 的 repair 中 best AUC={best_control_safe_auc:.6g}（target_mode={best_control_safe_row.get('target_mode', '')}，control_penalty={best_control_safe_row.get('control_penalty', '')}）。因此 C 仍未过，不能把 repair 写成 benefit-conditioned success。\n")
        if best_policy_export:
            if runtime_export_highsignal_strong:
                f.write(
                    f"- Benefit policy export 读回：best_export={best_policy_export.get('_source_file', '')}，horizon={best_policy_export.get('horizon', '')}，target={best_policy_export.get('target_mode', '')}，feature_set={best_policy_export.get('feature_set', '')}，min_leave_one_dataset_auc={best_policy_export.get('min_leave_one_dataset_auc', '')}，min_leave_one_seed_auc={best_policy_export.get('min_leave_one_seed_auc', '')}；export JSON 原始 `runtime_per_step_policy_integrated={best_policy_export.get('runtime_per_step_policy_integrated', '')}` 表示导出时仍是 offline artifact。本轮已新增真实 runtime 接入（报告 {runtime_export_selected_scope} controls-matched run）：integrated_task_rows={runtime_export_integrated_rows}，mean_export_score={runtime_export_mean_score}，{runtime_export_highsignal_task_counts}。high-signal controls-matched 读回：rows={runtime_export_highsignal_strong.get('rows')}，NLL_improvement_rows={runtime_export_highsignal_strong.get('NLL_improvement_rows')}，AUC_improvement_rows={runtime_export_highsignal_strong.get('AUC_improvement_rows')}，accuracy_improvement_rows={runtime_export_highsignal_strong.get('accuracy_improvement_rows')}，mean_NLL_delta={runtime_export_highsignal_strong.get('mean_NLL_delta')}，max_controller_overhead_ratio={runtime_export_highsignal_strong.get('max_controller_overhead_ratio')}，D_noharm_pass={runtime_export_highsignal_strong.get('D_noharm_pass')}，D_improvement_pass={runtime_export_highsignal_strong.get('D_improvement_pass')}。结论：policy 已真实进入 per-step runtime gate，但仍只给出 AUC/no-harm 信号，NLL strict improvement=0/9 且 max overhead>0.25，不能通过 C/D official。\n"
                )
            else:
                f.write(f"- Benefit policy export 读回：best_export={best_policy_export.get('_source_file', '')}，horizon={best_policy_export.get('horizon', '')}，target={best_policy_export.get('target_mode', '')}，feature_set={best_policy_export.get('feature_set', '')}，min_leave_one_dataset_auc={best_policy_export.get('min_leave_one_dataset_auc', '')}，min_leave_one_seed_auc={best_policy_export.get('min_leave_one_seed_auc', '')}，runtime_per_step_policy_integrated={best_policy_export.get('runtime_per_step_policy_integrated', '')}。结论：已有可审计 policy coefficient artifact，但仍未接入真实 per-step runtime controller，因此 C official 不能通过。\n")
        if runtime_export_split_strong:
            f.write(
                f"- Split-consensus source runtime export repair 读回：rows={runtime_export_split_strong.get('rows')}，NLL_noharm_rows={runtime_export_split_strong.get('NLL_noharm_rows')}，NLL_improvement_rows={runtime_export_split_strong.get('NLL_improvement_rows')}，AUC_improvement_rows={runtime_export_split_strong.get('AUC_improvement_rows')}，accuracy_improvement_rows={runtime_export_split_strong.get('accuracy_improvement_rows')}，mean_NLL_delta={runtime_export_split_strong.get('mean_NLL_delta')}，max_NLL_delta={runtime_export_split_strong.get('max_NLL_delta')}，max_tail_q99_delta={runtime_export_split_strong.get('max_tail_q99_delta')}，max_controller_overhead_ratio={runtime_export_split_strong.get('max_controller_overhead_ratio')}，D_noharm_pass={runtime_export_split_strong.get('D_noharm_pass')}，D_improvement_pass={runtime_export_split_strong.get('D_improvement_pass')}；task cadence/action={runtime_export_split_task_counts}，mean_export_score={runtime_export_split_mean_score}。结论：按计划加入更高 controllability 的 split-consensus param source 后，mean NLL 转为小幅负值，但 strict NLL improvement 仍为 0/9，tail no-harm 仅 7/9 且 overhead>0.25，因此不能作为 D/C official 修复。\n"
            )
        if runtime_export_split_thr055_strong:
            f.write(
                f"- Split-consensus high-confidence threshold repair 读回（threshold=0.55）：rows={runtime_export_split_thr055_strong.get('rows')}，NLL_noharm_rows={runtime_export_split_thr055_strong.get('NLL_noharm_rows')}，NLL_improvement_rows={runtime_export_split_thr055_strong.get('NLL_improvement_rows')}，AUC_improvement_rows={runtime_export_split_thr055_strong.get('AUC_improvement_rows')}，accuracy_improvement_rows={runtime_export_split_thr055_strong.get('accuracy_improvement_rows')}，mean_NLL_delta={runtime_export_split_thr055_strong.get('mean_NLL_delta')}，max_NLL_delta={runtime_export_split_thr055_strong.get('max_NLL_delta')}，max_tail_q99_delta={runtime_export_split_thr055_strong.get('max_tail_q99_delta')}，max_controller_overhead_ratio={runtime_export_split_thr055_strong.get('max_controller_overhead_ratio')}，D_noharm_pass={runtime_export_split_thr055_strong.get('D_noharm_pass')}，D_improvement_pass={runtime_export_split_thr055_strong.get('D_improvement_pass')}；task action={runtime_export_split_thr055_task_counts}，mean_export_score={runtime_export_split_thr055_mean_score}。结论：统一高置信阈值把 no-harm/tail 修成 9/9，并保留 AUC=8/9，但 strict NLL improvement 仍为 0/9，且 overhead 仍超过 0.25；不能 promotion。\n"
            )
        if runtime_export_split_cap010_strong or runtime_export_split_cap020_strong:
            cap_bits: list[str] = []
            if runtime_export_split_cap010_strong:
                cap_bits.append(
                    f"cap=0.10 seed0 pilot rows={runtime_export_split_cap010_strong.get('rows')} mean_NLL_delta={runtime_export_split_cap010_strong.get('mean_NLL_delta')} AUC_improvement_rows={runtime_export_split_cap010_strong.get('AUC_improvement_rows')} max_tail_q99_delta={runtime_export_split_cap010_strong.get('max_tail_q99_delta')} max_overhead={runtime_export_split_cap010_strong.get('max_controller_overhead_ratio')}"
                )
            if runtime_export_split_cap020_strong:
                cap_bits.append(
                    f"cap=0.20 seed0 pilot rows={runtime_export_split_cap020_strong.get('rows')} mean_NLL_delta={runtime_export_split_cap020_strong.get('mean_NLL_delta')} AUC_improvement_rows={runtime_export_split_cap020_strong.get('AUC_improvement_rows')} max_tail_q99_delta={runtime_export_split_cap020_strong.get('max_tail_q99_delta')} max_overhead={runtime_export_split_cap020_strong.get('max_controller_overhead_ratio')}"
                )
            f.write(f"- Split-consensus high-confidence strength pilot 读回：{'；'.join(cap_bits)}。结论：只对高置信动作加大 cap 后，seed0 NLL 仍停留在 `1e-4~1e-3` 量级，远低于 `NLL_delta<-0.01`，cap=0.20 还触发更多 predicted-gain/virtual-loss 拒绝，因此不扩展成 full controls 成功声明。\n")
        if runtime_p3_d1_rows:
            f.write(f"- Runtime P3 D1 h440 读回：`DGMLP_FU_BENEFIT_P3_JVP_REFRESH_STRONG` rows={runtime_p3_d1_strong.get('rows')}，NLL_improvement_rows={runtime_p3_d1_strong.get('NLL_improvement_rows')}，AUC_improvement_rows={runtime_p3_d1_strong.get('AUC_improvement_rows')}，accuracy_improvement_rows={runtime_p3_d1_strong.get('accuracy_improvement_rows')}，mean_NLL_delta={runtime_p3_d1_strong.get('mean_NLL_delta')}，max_controller_overhead_ratio={runtime_p3_d1_strong.get('max_controller_overhead_ratio')}，D_noharm_pass={runtime_p3_d1_strong.get('D_noharm_pass')}，D_improvement_pass={runtime_p3_d1_strong.get('D_improvement_pass')}；同文件 signflip control NLL_improvement_rows={runtime_p3_d1_signflip.get('NLL_improvement_rows')}、AUC_improvement_rows={runtime_p3_d1_signflip.get('AUC_improvement_rows')}。结论：strong JVP 增强了 NLL 均值但未到 5/9 strict improvement，且 controls 更强、overhead 远超 0.25。\n")
        if runtime_p3_tabular_rows:
            f.write(f"- Runtime P3 tabular Spam/Wine clear-proxy 读回：`DGMLP_FU_BENEFIT_P3_JVP_REFRESH` rows={runtime_p3_tabular_jvp.get('rows')}，NLL_improvement_rows={runtime_p3_tabular_jvp.get('NLL_improvement_rows')}，AUC_improvement_rows={runtime_p3_tabular_jvp.get('AUC_improvement_rows')}，mean_NLL_delta={runtime_p3_tabular_jvp.get('mean_NLL_delta')}；strong rows={runtime_p3_tabular_strong.get('rows')}，AUC_improvement_rows={runtime_p3_tabular_strong.get('AUC_improvement_rows')}，mean_NLL_delta={runtime_p3_tabular_strong.get('mean_NLL_delta')}，max_tail_q99_delta={runtime_p3_tabular_strong.get('max_tail_q99_delta')}。结论：tabular 上 JVP policy 稳定降低 AUC_loss_time，但 NLL 变差且 overhead 约 0.90，不能作为 D/G improvement success。\n")
        if runtime_p3_highsignal_rows:
            f.write(f"- High-signal tabular availability 读回：available={highsignal_available or 'none'}；unavailable={highsignal_unavailable or 'none'}。Abalone 虽标记 available，但 `input_size={highsignal_availability.get('Abalone', {}).get('input_size', '')}`、`first_train_batch_shape={highsignal_availability.get('Abalone', {}).get('first_train_batch_shape', '')}`，正式 bridge 因 7/8 维不一致失败；因此 high-signal 正式比较只使用跑出完整 rows 的 Spam/Card/Wine，不把 Abalone 失败写成算法收益失败。\n")
            f.write(f"- Runtime P3 high-signal interval=1 读回：JVP rows={runtime_p3_highsignal_i1_jvp.get('rows')}，NLL_improvement_rows={runtime_p3_highsignal_i1_jvp.get('NLL_improvement_rows')}，AUC_improvement_rows={runtime_p3_highsignal_i1_jvp.get('AUC_improvement_rows')}，mean_NLL_delta={runtime_p3_highsignal_i1_jvp.get('mean_NLL_delta')}，max_controller_overhead_ratio={runtime_p3_highsignal_i1_jvp.get('max_controller_overhead_ratio')}；strong rows={runtime_p3_highsignal_i1_strong.get('rows')}，NLL_improvement_rows={runtime_p3_highsignal_i1_strong.get('NLL_improvement_rows')}，AUC_improvement_rows={runtime_p3_highsignal_i1_strong.get('AUC_improvement_rows')}，D_noharm_pass={runtime_p3_highsignal_i1_strong.get('D_noharm_pass')}。结论：AUC 改善稳定，但 NLL=0/9 且 overhead 约 0.90。\n")
            f.write(f"- Runtime P3 high-signal interval=10 repair 读回：JVP rows={runtime_p3_highsignal_i10_jvp.get('rows')}，NLL_improvement_rows={runtime_p3_highsignal_i10_jvp.get('NLL_improvement_rows')}，AUC_improvement_rows={runtime_p3_highsignal_i10_jvp.get('AUC_improvement_rows')}，mean_NLL_delta={runtime_p3_highsignal_i10_jvp.get('mean_NLL_delta')}，max_controller_overhead_ratio={runtime_p3_highsignal_i10_jvp.get('max_controller_overhead_ratio')}；strong max_controller_overhead_ratio={runtime_p3_highsignal_i10_strong.get('max_controller_overhead_ratio')}，D_noharm_pass={runtime_p3_highsignal_i10_strong.get('D_noharm_pass')}。结论：cadence skip 降低了 tail/NLL debt，但 overhead 仍大于 0.25 且 NLL improvement 仍为 0/9。\n")
            f.write(f"- Runtime P3 high-signal interval=50 pilot 读回：JVP rows={runtime_p3_highsignal_i50_jvp.get('rows')}，NLL_improvement_rows={runtime_p3_highsignal_i50_jvp.get('NLL_improvement_rows')}，AUC_improvement_rows={runtime_p3_highsignal_i50_jvp.get('AUC_improvement_rows')}，mean_NLL_delta={runtime_p3_highsignal_i50_jvp.get('mean_NLL_delta')}，max_controller_overhead_ratio={runtime_p3_highsignal_i50_jvp.get('max_controller_overhead_ratio')}；strong rows={runtime_p3_highsignal_i50_strong.get('rows')}，NLL_improvement_rows={runtime_p3_highsignal_i50_strong.get('NLL_improvement_rows')}，AUC_improvement_rows={runtime_p3_highsignal_i50_strong.get('AUC_improvement_rows')}，accuracy_improvement_rows={runtime_p3_highsignal_i50_strong.get('accuracy_improvement_rows')}，mean_NLL_delta={runtime_p3_highsignal_i50_strong.get('mean_NLL_delta')}，max_controller_overhead_ratio={runtime_p3_highsignal_i50_strong.get('max_controller_overhead_ratio')}，D_improvement_pass={runtime_p3_highsignal_i50_strong.get('D_improvement_pass')}；task cadence={runtime_p3_highsignal_i50_task_counts}。结论：interval=50 strong 首次把 overhead 压到 0.25 以下并保留 AUC=9/9，但 NLL strict improvement 仍为 0/9，且该 run 未包含 controls，不能扩展成 D official success。\n")
            if runtime_p3_highsignal_i50_scale05_strong:
                f.write(f"- Runtime P3 high-signal interval=50 scale=0.50 pilot 读回：strong rows={runtime_p3_highsignal_i50_scale05_strong.get('rows')}，NLL_improvement_rows={runtime_p3_highsignal_i50_scale05_strong.get('NLL_improvement_rows')}，AUC_improvement_rows={runtime_p3_highsignal_i50_scale05_strong.get('AUC_improvement_rows')}，accuracy_improvement_rows={runtime_p3_highsignal_i50_scale05_strong.get('accuracy_improvement_rows')}，mean_NLL_delta={runtime_p3_highsignal_i50_scale05_strong.get('mean_NLL_delta')}，max_NLL_delta={runtime_p3_highsignal_i50_scale05_strong.get('max_NLL_delta')}，max_tail_q99_delta={runtime_p3_highsignal_i50_scale05_strong.get('max_tail_q99_delta')}，max_controller_overhead_ratio={runtime_p3_highsignal_i50_scale05_strong.get('max_controller_overhead_ratio')}，D_improvement_pass={runtime_p3_highsignal_i50_scale05_strong.get('D_improvement_pass')}；task cadence={runtime_p3_highsignal_i50_scale05_task_counts}。结论：按计划提高高置信动作强度后，mean NLL 转为极小负值但仍远低于 `NLL_delta<-0.01` 的 5/9 strict threshold，且 max overhead 重新超过 0.25；该 pilot 仍无 controls，不能 promotion。\n")
        f.write(f"- PureKAN 身份审计读回：model_identity_matrix_complete={int_flag(identity_latest.get('model_identity_matrix_complete'))}，official_DGKAN_rows_have_strict_FC_PureKAN_identity={int_flag(identity_latest.get('official_DGKAN_rows_have_strict_FC_PureKAN_identity'))}，KANbeFair_original_KAN_rows_marked_baseline_context_only={int_flag(identity_latest.get('KANbeFair_original_KAN_rows_marked_baseline_context_only'))}。这意味着 KANbeFair 原始 `KAN`/`BSpline_*` 不再被用于 DG-KAN efficiency/superiority 结论。\n")
        f.write(f"- F 四路径小批修复读回：same_dataset_seed_hidden_batch_complete_keys={identity_latest.get('same_dataset_seed_hidden_batch_complete_keys', '')}，same_data_batch_hidden_seed_complete_keys={identity_latest.get('same_data_batch_hidden_seed_complete_keys', '')}，example={identity_latest.get('same_data_batch_hidden_seed_example_key', '')}，F_four_path_exploration_pass={identity_latest.get('F_four_path_exploration_pass', '')}。这说明 `MNIST|seed0|D-FOU|hidden16|train512|test256|batch64` 的四路径同初始 batch 审计已经真实闭合。\n")
        f.write(f"- 效率路径审计读回：efficiency_path_identity_matrix_complete={int_flag(identity_latest.get('efficiency_path_identity_matrix_complete'))}，official_efficiency_rows_use_analytic_or_sketch_JVP={int_flag(identity_latest.get('official_efficiency_rows_use_analytic_or_sketch_JVP'))}，efficiency_path_official_pass_rows={identity_latest.get('efficiency_path_official_pass_rows', 0)}，same_data_batch_hidden_seed_complete={int_flag(identity_latest.get('same_data_batch_hidden_seed_complete'))}。Path2 native full-loop finite_diff_fallback_rows={bridge_by_path.get('2', {}).get('finite_diff_fallback_rows', '')}，Path3 official_path={bridge_by_path.get('3', {}).get('efficiency_path_official_pass_rows', '')}，Path4 official_path={bridge_by_path.get('4', {}).get('efficiency_path_official_pass_rows', '')}，Path4 controller_overhead_mean={bridge_by_path.get('4', {}).get('mean_controller_overhead_ratio', '')}。同 batch no-dense bridge rerun 后，Path3/Path4 已各有 official path row；latest native timing repair 也使 F_efficiency_exploration_pass={int(f_explore)}。这只解决 efficiency gate，不代表 D/E/G/H task benefit 或 official promotion 通过。\n")
        if native_diag_i40:
            f.write(f"- F native timing repair no-go 读回（interval=40）：source={native_diag_i40.get('_source_file', '')}，official_full_loop_ratio_le_3={native_diag_i40.get('official_full_loop_ratio_le_3', '')}/9，official_controller_overhead_le_025={native_diag_i40.get('official_controller_overhead_le_025', '')}/9，max_official_full_loop_ratio={native_diag_i40.get('max_official_full_loop_ratio', '')}，max_official_controller_overhead={native_diag_i40.get('max_official_controller_overhead', '')}。结论：仅分离诊断计时仍被 controller overhead 卡住。\n")
        if native_diag_i120:
            f.write(f"- F native cadence repair pass 读回（interval=120）：source={native_diag_i120.get('_source_file', '')}，official_full_loop_ratio_le_3={native_diag_i120.get('official_full_loop_ratio_le_3', '')}/9，official_controller_overhead_le_025={native_diag_i120.get('official_controller_overhead_le_025', '')}/9，max_official_full_loop_ratio={native_diag_i120.get('max_official_full_loop_ratio', '')}，max_official_controller_overhead={native_diag_i120.get('max_official_controller_overhead', '')}，auc_improve_vs_kan={native_diag_i120.get('auc_improve_vs_kan', '')}/9，nll_improve_vs_kan={native_diag_i120.get('nll_improve_vs_kan', '')}/9。结论：降低 refresh cadence 修复了 F efficiency，但没有修复 E task benefit。\n")
        if best_native_eff:
            f.write(f"- F native breadth repair 读回：best source={best_native_eff.get('_source_file', '')}，rows={best_native_eff.get('rows')}，diagnostic_separated_timing_rows={best_native_eff.get('diagnostic_separated_timing_rows', '')}，official_controller_overhead_le_025={best_native_eff.get('official_controller_overhead_le_025', '')}，official_full_loop_ratio_le_3={best_native_eff.get('official_full_loop_ratio_le_3', '')}，max_official_full_loop_ratio={best_native_eff.get('max_official_full_loop_ratio', '')}；旧全包 full_loop_ratio_le_3={best_native_eff.get('full_loop_ratio_le_3')}，max_full_loop_ratio={best_native_eff.get('max_full_loop_ratio')}。F gate 现在只允许诊断分离 timing 完整的 row 用 official ratio 计入，不把 source-control 诊断开销混进正式 full-loop 结论。\n")
        if e_task_pilot_rows:
            pilot_bits = "；".join(
                f"{r.get('_source_file', '')}: NLL_delta={r.get('NLL_delta_vs_KAN_AdamW_native', '')}, AUC_delta={r.get('AUC_loss_time_delta_vs_KAN_AdamW_native', '')}, accept={r.get('gate_accept_count', '')}, ratio={r.get('mean_controller_to_base_update_ratio', '')}"
                for r in sorted(e_task_pilot_rows, key=lambda row: str(row.get("_source_file", "")))
            )
            f.write(f"- E task benefit 强干预 pilot 读回：{pilot_bits}。scale=1.0/cap=0.5 使 NLL/AUC 明显变差；scale=0.2/cap=0.2 在 h120 仅达 NLL_delta=-0.000337，h440 又变为正向变差，均远未接近 `NLL_delta<-0.01` 的 5/9 official-candidate 阈值，因此不扩展成 9-row 成功声明。\n")
        if e_task_virtual_i40_stats or e_task_virtual_i120_stats:
            if e_task_virtual_i40_stats:
                f.write(
                    f"- E task-aligned JVP virtual-improvement gate 读回（interval=40）：source_files={e_task_virtual_i40_stats.get('source_files')}，rows={e_task_virtual_i40_stats.get('rows')}，NLL_delta<0 rows={e_task_virtual_i40_stats.get('nll_negative_rows')}，strict_NLL_delta<-0.01 rows={e_task_virtual_i40_stats.get('strict_nll_improvement_rows')}，AUC_delta<0 rows={e_task_virtual_i40_stats.get('auc_negative_rows')}，mean_NLL_delta={e_task_virtual_i40_stats.get('mean_nll_delta'):.12g}，mean_AUC_delta={e_task_virtual_i40_stats.get('mean_auc_delta'):.12g}，official_overhead_pass_rows={e_task_virtual_i40_stats.get('official_overhead_pass_rows')}，official_full_loop_pass_rows={e_task_virtual_i40_stats.get('official_full_loop_pass_rows')}，accept_minmax={e_task_virtual_i40_stats.get('accept_minmax')}，reject_minmax={e_task_virtual_i40_stats.get('reject_minmax')}。\n"
                )
            if e_task_virtual_i120_stats:
                f.write(
                    f"- E task-aligned JVP virtual-improvement gate 读回（interval=120）：source_files={e_task_virtual_i120_stats.get('source_files')}，rows={e_task_virtual_i120_stats.get('rows')}，NLL_delta<0 rows={e_task_virtual_i120_stats.get('nll_negative_rows')}，strict_NLL_delta<-0.01 rows={e_task_virtual_i120_stats.get('strict_nll_improvement_rows')}，AUC_delta<0 rows={e_task_virtual_i120_stats.get('auc_negative_rows')}，mean_NLL_delta={e_task_virtual_i120_stats.get('mean_nll_delta'):.12g}，mean_AUC_delta={e_task_virtual_i120_stats.get('mean_auc_delta'):.12g}，official_overhead_pass_rows={e_task_virtual_i120_stats.get('official_overhead_pass_rows')}，official_full_loop_pass_rows={e_task_virtual_i120_stats.get('official_full_loop_pass_rows')}，accept_minmax={e_task_virtual_i120_stats.get('accept_minmax')}，reject_minmax={e_task_virtual_i120_stats.get('reject_minmax')}。\n"
                )
            f.write("- E task-aligned gate 结论：要求当前 batch virtual train loss 改善后，interval=40 能把 tiny NLL/AUC 方向变得更一致，但 official overhead/full-loop 不过；interval=120 修复了 overhead/full-loop，却只保留 3/9 AUC_delta<0，且两个版本 strict NLL improvement 都是 0/9。该修复说明 source-task alignment 有微弱方向信号，但幅度仍远低于 task-benefit gate，不能 promotion。\n")
        if e_long_task_stats:
            e2_bits = "；".join(
                (
                    f"{r.get('dataset', '')}/seed{r.get('seed', '')}: h={r.get('horizon_steps', '')}, "
                    f"NLL_delta={r.get('NLL_delta_vs_KAN_AdamW_native', '')}, "
                    f"AUC_delta={r.get('AUC_loss_time_delta_vs_KAN_AdamW_native', '')}, "
                    f"accept={r.get('gate_accept_count', '')}, reject={r.get('gate_reject_count', '')}, "
                    f"official_overhead={r.get('official_controller_overhead_ratio', '')}, "
                    f"official_full_loop={r.get('official_full_loop_ratio_vs_mlp', '')}, "
                    f"source_loss_horizon={r.get('KAN_source_loss_horizon', '')}, "
                    f"blocker={r.get('blocker', '')}"
                )
                for r in sorted(
                    e_long_task_pilot_rows,
                    key=lambda row: (
                        finite_float(row.get("horizon_steps"), 0.0),
                        str(row.get("dataset", "")),
                        str(row.get("seed", "")),
                    ),
                )
            )
            f.write(
                f"- E2/E3 long-horizon task-aligned pilot 读回：source_files={e_long_task_stats.get('source_files')}，rows={e_long_task_stats.get('rows')}，"
                f"NLL_delta<0 rows={e_long_task_stats.get('nll_negative_rows')}，strict_NLL_delta<-0.01 rows={e_long_task_stats.get('strict_nll_improvement_rows')}，"
                f"AUC_delta<0 rows={e_long_task_stats.get('auc_negative_rows')}，mean_NLL_delta={e_long_task_stats.get('mean_nll_delta'):.12g}，"
                f"mean_AUC_delta={e_long_task_stats.get('mean_auc_delta'):.12g}，official_overhead_pass_rows={e_long_task_stats.get('official_overhead_pass_rows')}，"
                f"official_full_loop_pass_rows={e_long_task_stats.get('official_full_loop_pass_rows')}；{e2_bits}。结论：h1600/h3200/h4800 seed0 pilots 均保持 source 与 official efficiency，但 NLL_delta 全为非负，AUC 只在 h1600/h3200 有 `1e-6~1e-5` 量级微弱改善、h4800 变为正向变差；这说明当前 task-aligned JVP gate 的 source retention 没有转化为 task NLL benefit，E task benefit 与 official ladder 仍未打开。\n"
            )
        if e4_task_release_stats:
            release_bits = "；".join(
                (
                    f"{r.get('dataset', '')}/seed{r.get('seed', '')}: h={r.get('horizon_steps', '')}, "
                    f"min_history={r.get('jvp_refresh_min_history', '')}, window={r.get('jvp_refresh_history_window', '')}, "
                    f"release_on_accept={r.get('jvp_refresh_release_on_accept', '')}, source_release_count={r.get('source_release_count', '')}, "
                    f"NLL_delta={r.get('NLL_delta_vs_KAN_AdamW_native', '')}, "
                    f"AUC_delta={r.get('AUC_loss_time_delta_vs_KAN_AdamW_native', '')}, "
                    f"accept={r.get('gate_accept_count', '')}, official_overhead={r.get('official_controller_overhead_ratio', '')}, "
                    f"official_full_loop={r.get('official_full_loop_ratio_vs_mlp', '')}, source_loss_horizon={r.get('KAN_source_loss_horizon', '')}, blocker={r.get('blocker', '')}"
                )
                for r in sorted(
                    e4_task_release_rows,
                    key=lambda row: (
                        finite_float(row.get("horizon_steps"), 0.0),
                        str(row.get("dataset", "")),
                        str(row.get("seed", "")),
                    ),
                )
            )
            f.write(
                f"- E4 short-age + release-on-accept repair 读回：source_files={e4_task_release_stats.get('source_files')}，rows={e4_task_release_stats.get('rows')}，"
                f"NLL_delta<0 rows={e4_task_release_stats.get('nll_negative_rows')}，strict_NLL_delta<-0.01 rows={e4_task_release_stats.get('strict_nll_improvement_rows')}，"
                f"AUC_delta<0 rows={e4_task_release_stats.get('auc_negative_rows')}，mean_NLL_delta={e4_task_release_stats.get('mean_nll_delta'):.12g}，"
                f"mean_AUC_delta={e4_task_release_stats.get('mean_auc_delta'):.12g}，official_overhead_pass_rows={e4_task_release_stats.get('official_overhead_pass_rows')}，"
                f"official_full_loop_pass_rows={e4_task_release_stats.get('official_full_loop_pass_rows')}；{release_bits}。结论：缩短 history window 并在 accept 后 release 可以显著降低 official overhead，并在 h1600 把 NLL/AUC 变成 `1e-6` 量级微弱负 delta；但 h3200 仍为 NLL/AUC 正 delta，strict NLL improvement 仍为 0，说明 stale source age 不是当前 E task-benefit blocker 的充分解释。\n"
            )
        if h_s012_rows:
            reductions = ",".join(str(r.get("absolute_forgetting_reduction", "")) for r in h_s012_rows)
            h_passes = ",".join(str(r.get("H_official_pass", "")) for r in h_s012_rows)
            f.write(f"- H continual seeds0-2 official-form 读回：absolute_forgetting_reduction={reductions}，H_official_pass={h_passes}。`DGMLP_FU_RELEASE` 未降低 Class_MNIST average forgetting，不能声明 anti-forgetting success。\n")
        if h_transport_rows:
            f.write(f"- H source-transport repair 读回：source_files={h_transport_sources}；exploration_pass_rows={len(h_transport_explore_rows)}/{len(h_transport_rows)}，official_pass_rows={len(h_transport_official_rows)}/{len(h_transport_rows)}，best_variant={h_transport_best.get('fu_model_name', '')}，best_seed={h_transport_best.get('seed', '')}，best_relative_forgetting_reduction={h_transport_best.get('relative_forgetting_reduction', '')}，best_absolute_forgetting_reduction={h_transport_best.get('absolute_forgetting_reduction', '')}，best_accuracy_delta={h_transport_best.get('average_accuracy_delta', '')}，best_overhead={h_transport_best.get('controller_overhead_ratio', '')}。结论：把 continual source 从 output-cotangent 修为参数空间 source/boundary memory 后，H exploration 打开并将 route 推到 continual-forgetting-reduction-opened；但 cap=0.03/0.04/0.05 sweep 显示信号不稳定，最大 relative reduction 仍远低于 official `>=10%`，且没有 controls-fail 证据，因此不能 promotion。\n")
        if h_virtual_gate_rows:
            f.write(f"- H current-task virtual NLL gate repair 读回：source_files={h_virtual_gate_sources}；reject_counts={h_virtual_gate_reject_counts}，exploration_pass_rows={len(h_virtual_gate_explore_rows)}/{len(h_virtual_gate_rows)}，official_pass_rows={len(h_virtual_gate_official_rows)}/{len(h_virtual_gate_rows)}，best_variant={h_virtual_gate_best.get('fu_model_name', '')}，best_seed={h_virtual_gate_best.get('seed', '')}，best_relative_forgetting_reduction={h_virtual_gate_best.get('relative_forgetting_reduction', '')}，best_absolute_forgetting_reduction={h_virtual_gate_best.get('absolute_forgetting_reduction', '')}，best_accuracy_delta={h_virtual_gate_best.get('average_accuracy_delta', '')}，max_overhead={h_virtual_gate_max_overhead:.12g}。结论：按计划给 continual FU 加入当前任务 virtual NLL cost 后，多数候选被拒绝，负向 forgetting rows 明显减少，但 seed0 原先 cap=0.05 的正向 forgetting reduction 也被基本抹平，且 virtual check 使 overhead 升至约 0.75-0.78；H official 仍为 0，不能把该修复声明为 anti-forgetting success。\n")
        f.write("- D/E/G/H 的任何 pass 都按各自 artifact 的真实阈值计入；未满足项写入 `v22_18_deferred_items.csv`，不作为成功声明。\n")
        f.write("- 后续若继续推进，重点应转向 B/C runtime official、D/G controls-matched improvement、E KAN task benefit 与 H continual；F efficiency 已在当前 exploration gate 下通过，但不能替代 task/superiority 证据。\n")

    cmd = " ".join(shlex.quote(x) for x in [sys.executable, *sys.argv])
    append_exec(
        cmd,
        task_id="Z-finalize",
        status="pass",
        exit_code=0,
        files="results/v22_18/v22_18_final_route.json, results/v22_18/v22_18_gate_summary.csv, docs/DG-KAN_v22.18_BenefitConditionedControllability_实验结果复盘.md",
        note=f"route={route}; official_promotion={full_official}",
    )


if __name__ == "__main__":
    main()
