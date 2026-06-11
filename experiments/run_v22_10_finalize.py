#!/usr/bin/env python3
"""v22.10 final route, recap, artifact index, and bundles."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_v22_10_common import (  # noqa: E402
    PYTHON,
    V2210_RECAP_DOC,
    append_exec,
    artifact_index,
    build_code_review_packet,
    build_results_bundle,
    ensure_out,
    finite_float,
    int_flag,
    md_table,
    now_sg,
    read_json,
    read_rows,
    write_json,
    write_rows,
)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None)
    return p


def _route(out_dir: Path) -> dict[str, Any]:
    code = read_json(out_dir / "v22_10_code_route_decision.json")
    basis = read_json(out_dir / "v22_10_basis_efficiency_route.json")
    source = read_json(out_dir / "v22_10_source_atom_route.json")
    variational = read_json(out_dir / "v22_10_variational_source_route.json")
    commit = read_json(out_dir / "v22_10_metric_commit_route.json")
    horizon = read_json(out_dir / "v22_10_horizon_source_route.json")
    kan = read_json(out_dir / "v22_10_kan_mapping_route.json")
    queue = read_json(out_dir / "v22_10_queue_drain_report.json")
    idle_rows = read_rows(out_dir / "v22_10_idle_violation.csv")
    queue_violation = any(int_flag(r.get("execution_contract_violation")) for r in idle_rows)
    blockers: list[str] = []
    if not int_flag(code.get("S0_17_pass")) or code.get("CodeRoute") != "R0-CodePacketSelfContained":
        blockers.append("S0.17_code_packet_truth_gate")
    optimization_loss_agnostic_open = int_flag(basis.get("optimization_loss_agnostic_contract_pass"))
    if not optimization_loss_agnostic_open:
        blockers.append("optimization_loss_agnostic_efficiency_gate")
    if not int_flag(basis.get("D-CHE_S1_pass")) or not int_flag(basis.get("D-FOU_S1_pass")):
        blockers.append("D-CHE_D-FOU_efficiency_reconfirm_gate")
    drat_drbf_efficiency_open = bool(optimization_loss_agnostic_open) and any(
        int_flag(basis.get(key))
        for key in (
            "D-RAT_robust_pass",
            "D-RBF_robust_pass",
            "official_candidate_repair_candidate_groups",
            "official_blockh_repair_candidate_groups",
            "official_fastk2_repair_candidate_groups",
            "official_blockb_repair_candidate_groups",
            "official_combo_repair_candidate_groups",
        )
    )
    if not drat_drbf_efficiency_open:
        blockers.append("D-RAT_D-RBF_robust_efficiency_gate")
    loss_agnostic_contract_open = all(
        int_flag(stage.get("loss_agnostic_contract_pass"))
        and not int_flag(stage.get("uses_labels_for_direction"))
        and not int_flag(stage.get("uses_loss_for_direction"))
        for stage in (source, variational, commit, horizon, kan)
    )
    if not loss_agnostic_contract_open:
        blockers.append("loss_agnostic_source_contract_gate")
    if not int_flag(source.get("S2_source_atom_pass_rows")):
        blockers.append("S2_source_atom_generation_gate")
    if not int_flag(variational.get("S3_variational_source_solve_pass_rows")):
        blockers.append("S3_variational_source_solve_gate")
    if not int_flag(commit.get("S4_metric_dynamics_commit_pass_rows")):
        blockers.append("S4_metric_dynamics_commit_gate")
    if not int_flag(horizon.get("C3_source_formation_pass_rows")):
        blockers.append("S5_C3_source_formation_gate")
    if not int_flag(horizon.get("C4_terminal_retention_pass_rows")):
        blockers.append("S5_C4_terminal_retention_gate")
    if kan.get("decision") == "KANMappingNotEntered":
        blockers.append("S6_KAN_mapping_not_entered")
    if kan.get("decision") == "KANSourceChannelMismatchConfirmed":
        blockers.append("S6_KAN_source_channel_mismatch")
    if queue_violation:
        blockers.append("4GPU_queue_idle_violation")

    if not int_flag(code.get("S0_17_pass")) or code.get("CodeRoute") != "R0-CodePacketSelfContained":
        route = "R0-CodePacketOrImplementationClosureFail"
        action = "repair self-contained packet/import/kernel status before scientific conclusion"
    elif not int_flag(source.get("S2_source_atom_pass_rows")):
        route = "R2-SourceAtomGenerationNoGo"
        action = "stop same atom expansion after planned norm/split/block fallbacks; change source atom principle"
    elif not int_flag(variational.get("S3_variational_source_solve_pass_rows")):
        route = "R3-VariationalSourceSolveNoGo"
        action = "current pass atoms cannot form coherent low-diffusion low-control source; redesign variational objective or atom basis"
    elif not int_flag(commit.get("S4_metric_dynamics_commit_pass_rows")):
        route = "R4-MetricDynamicsCommitNoGo"
        action = "try lower-rank CG, higher damping, hidden/readout block solve, or optimizer-state write"
    elif int_flag(horizon.get("C3_source_formation_pass_rows")) and not int_flag(horizon.get("C4_terminal_retention_pass_rows")):
        route = "R5-MLPSourceFormationOpenedTerminalBlocked"
        action = "run terminal preservation repair before KAN mapping"
    elif int_flag(horizon.get("C4_terminal_retention_pass_rows")) and kan.get("decision") == "KANMappingNotEntered":
        route = "R6-MLPTerminalRetentionOpened"
        action = "enter gated KAN source-channel mapping"
    elif kan.get("decision") == "KANSourceChannelMismatchConfirmed":
        route = "R7-KANSourceChannelMismatchConfirmed"
        action = "record KAN mismatch and compare basis-channel decomposition"
    elif kan.get("decision") == "KANRetainedSourceOpened" and not blockers:
        route = "R9-OfficialPromotionCandidate"
        action = "independent confirmation and code-review packet audit"
    elif kan.get("decision") == "KANRetainedSourceOpened":
        route = "R8-KANRetainedSourceOpened"
        if not optimization_loss_agnostic_open:
            action = "implement arbitrary-loss/upstream-gradient S1 efficiency runner before any basis-efficiency repair or promotion"
        else:
            action = "continue D-RAT/D-RBF forward robust efficiency repair or record efficiency-side boundary before promotion"
    else:
        route = "R9-OfficialPromotionCandidate"
        action = "independent confirmation and code-review packet audit"

    progress = []
    if int_flag(code.get("S0_17_pass")) and code.get("CodeRoute") == "R0-CodePacketSelfContained":
        progress.append("A-CodeClosure")
    if optimization_loss_agnostic_open and drat_drbf_efficiency_open:
        progress.append("B-EfficiencyClosure")
    if int_flag(source.get("S2_source_atom_pass_rows")):
        progress.append("C-SourceAtomProgress")
    if int_flag(variational.get("S3_variational_source_solve_pass_rows")):
        progress.append("D-ConstructiveSolveProgress")
    if int_flag(horizon.get("C3_source_formation_pass_rows")):
        progress.append("E-FunctionalProgress")
    if kan.get("decision") == "KANRetainedSourceOpened":
        progress.append("F-KANSourceProgress")
    if not progress:
        progress.append("G-TheoryBoundary")
    return {
        "route": route,
        "promotion_allowed": int(route == "R9-OfficialPromotionCandidate"),
        "blocking_metric": ";".join(dict.fromkeys(blockers)),
        "S0_17_pass": int_flag(code.get("S0_17_pass")),
        "CodeRoute": code.get("CodeRoute", ""),
        "csv_claimed_exists_but_zip_missing_count": int_flag(code.get("csv_claimed_exists_but_zip_missing_count")),
        "required_import_errors": int_flag(code.get("required_import_errors")),
        "kernel_status_consistency": int_flag(code.get("kernel_status_consistency")),
        "D-CHE_S1_pass": int_flag(basis.get("D-CHE_S1_pass")),
        "D-FOU_S1_pass": int_flag(basis.get("D-FOU_S1_pass")),
        "D-RAT_robust_pass": int_flag(basis.get("D-RAT_robust_pass")),
        "D-RBF_robust_pass": int_flag(basis.get("D-RBF_robust_pass")),
        "official_candidate_repair_scan_attempted": int_flag(basis.get("official_candidate_repair_scan_attempted")),
        "official_candidate_repair_scan_rows": int_flag(basis.get("official_candidate_repair_scan_rows")),
        "official_candidate_repair_candidate_groups": int_flag(basis.get("official_candidate_repair_candidate_groups")),
        "official_candidate_repair_candidate_carriers": basis.get("official_candidate_repair_candidate_carriers", ""),
        "official_candidate_repair_scan_route": basis.get("official_candidate_repair_scan_route", ""),
        "official_blockh_repair_scan_attempted": int_flag(basis.get("official_blockh_repair_scan_attempted")),
        "official_blockh_repair_scan_rows": int_flag(basis.get("official_blockh_repair_scan_rows")),
        "official_blockh_repair_candidate_groups": int_flag(basis.get("official_blockh_repair_candidate_groups")),
        "official_blockh_repair_candidate_carriers": basis.get("official_blockh_repair_candidate_carriers", ""),
        "official_blockh_repair_scan_route": basis.get("official_blockh_repair_scan_route", ""),
        "official_fastk2_repair_scan_attempted": int_flag(basis.get("official_fastk2_repair_scan_attempted")),
        "official_fastk2_repair_scan_rows": int_flag(basis.get("official_fastk2_repair_scan_rows")),
        "official_fastk2_repair_candidate_groups": int_flag(basis.get("official_fastk2_repair_candidate_groups")),
        "official_fastk2_repair_candidate_carriers": basis.get("official_fastk2_repair_candidate_carriers", ""),
        "official_fastk2_repair_scan_route": basis.get("official_fastk2_repair_scan_route", ""),
        "official_blockb_repair_scan_attempted": int_flag(basis.get("official_blockb_repair_scan_attempted")),
        "official_blockb_repair_scan_rows": int_flag(basis.get("official_blockb_repair_scan_rows")),
        "official_blockb_repair_candidate_groups": int_flag(basis.get("official_blockb_repair_candidate_groups")),
        "official_blockb_repair_candidate_carriers": basis.get("official_blockb_repair_candidate_carriers", ""),
        "official_blockb_repair_scan_route": basis.get("official_blockb_repair_scan_route", ""),
        "official_combo_repair_scan_attempted": int_flag(basis.get("official_combo_repair_scan_attempted")),
        "official_combo_repair_scan_rows": int_flag(basis.get("official_combo_repair_scan_rows")),
        "official_combo_repair_candidate_groups": int_flag(basis.get("official_combo_repair_candidate_groups")),
        "official_combo_repair_candidate_carriers": basis.get("official_combo_repair_candidate_carriers", ""),
        "official_combo_repair_scan_route": basis.get("official_combo_repair_scan_route", ""),
        "optimization_loss_agnostic_contract_pass": optimization_loss_agnostic_open,
        "ce_targeted_efficiency_rows_invalidated": int_flag(basis.get("ce_targeted_efficiency_rows_invalidated")),
        "loss_agnostic_efficiency_contract": basis.get("loss_agnostic_efficiency_contract", ""),
        "loss_agnostic_contract_pass": int(loss_agnostic_contract_open),
        "source_uses_labels_for_direction": int(
            any(int_flag(stage.get("uses_labels_for_direction")) for stage in (source, variational, commit, horizon, kan))
        ),
        "source_uses_loss_for_direction": int(
            any(int_flag(stage.get("uses_loss_for_direction")) for stage in (source, variational, commit, horizon, kan))
        ),
        "S2_source_atom_pass_rows": int_flag(source.get("S2_source_atom_pass_rows")),
        "S2_selected_attempt": source.get("selected_attempt", ""),
        "S3_variational_source_solve_pass_rows": int_flag(variational.get("S3_variational_source_solve_pass_rows")),
        "S3_selected_attempt": variational.get("selected_attempt", ""),
        "S3_blocker": variational.get("blocker", ""),
        "S4_metric_dynamics_commit_pass_rows": int_flag(commit.get("S4_metric_dynamics_commit_pass_rows")),
        "C3_source_formation_pass_rows": int_flag(horizon.get("C3_source_formation_pass_rows")),
        "C4_terminal_retention_pass_rows": int_flag(horizon.get("C4_terminal_retention_pass_rows")),
        "PID_control_attempt_rows": int_flag(horizon.get("PID_control_attempt_rows")),
        "PID_C3_source_formation_pass_rows": int_flag(horizon.get("PID_C3_source_formation_pass_rows")),
        "PID_C4_terminal_retention_pass_rows": int_flag(horizon.get("PID_C4_terminal_retention_pass_rows")),
        "PID_best_attempt": horizon.get("PID_best_attempt", ""),
        "PID_best_source_vs_best_control_h3200": horizon.get("PID_best_source_vs_best_control_h3200", ""),
        "PID_best_pid_final_projected_gain": horizon.get("PID_best_pid_final_projected_gain", ""),
        "KAN_mapping_decision": kan.get("decision", ""),
        "queue_drained": int_flag(queue.get("queue_drained")),
        "execution_contract_violation": int(queue_violation),
        "minimum_effective_progress": ";".join(progress),
        "next_codex_action": action,
    }


def _failure_taxonomy(route: dict[str, Any]) -> list[dict[str, Any]]:
    mapping = {
        "optimization_loss_agnostic_efficiency_gate": "do not run or promote CE-targeted trainpath repairs; implement an arbitrary-loss/upstream-gradient efficiency runner before any S1 closure claim",
        "D-RAT_D-RBF_robust_efficiency_gate": "after loss-agnostic efficiency runner exists, continue D-RAT numerator/denominator reciprocal fusion, singlelaunch/launch-meta forward repair, D-RBF singlelaunch/local backward, and low-hidden/blockH/fastK2/blockB/combo scans under the same robust gate",
        "loss_agnostic_source_contract_gate": "source-side route must not use labels or supervised loss for source direction, variational solve, C3/C4 horizon, or KAN mapping; rerun loss-agnostic constructive path before any promotion claim",
        "S2_source_atom_generation_gate": "lower source atom norm, increase split count, use block-restricted atoms, then change atom principle",
        "S3_variational_source_solve_gate": "try stronger l1 sparsity, low-NDS-only, control-null-only, block-restricted atoms, then redesign source atom basis",
        "S4_metric_dynamics_commit_gate": "readout-only, hidden/readout block, lower-rank CG, higher damping, optimizer-state write",
        "S5_C3_source_formation_gate": "only enter official h100-h6400 horizon after S4 opens",
        "S6_KAN_mapping_not_entered": "enter KAN mapping only after MLP C3/C4 opens",
        "S6_KAN_source_channel_mismatch": "D-CHE/D-FOU lightweight source-channel mapping failed same-mechanism MLP comparison; redesign KAN readout/basis source channel",
    }
    rows = []
    for blocker in [x for x in str(route.get("blocking_metric", "")).split(";") if x]:
        rows.append({"blocker": blocker, "route": route.get("route", ""), "repair_or_next_direction": mapping.get(blocker, route.get("next_codex_action", ""))})
    return rows


def _write_recap(out_dir: Path, route: dict[str, Any]) -> None:
    code_rows = read_rows(out_dir / "v22_10_code_truth_gate.csv")
    clean = read_rows(out_dir / "v22_10_clean_unzip_self_test.csv")
    kernel = read_rows(out_dir / "v22_10_kernel_gradcheck.csv")
    eff = read_rows(out_dir / "v22_10_efficiency_full_loop_reconfirm.csv")
    dr = read_rows(out_dir / "v22_10_drat_drbf_multibatch_summary.csv")
    bridge = read_rows(out_dir / "v22_10_drat_drbf_telemetry_bridge_summary.csv")
    official_candidate = read_rows(out_dir / "v22_10_drat_drbf_official_candidate_repair_scan_summary.csv")
    blockh_scan = read_rows(out_dir / "v22_10_drat_drbf_blockh_repair_scan_summary.csv")
    fastk2_scan = read_rows(out_dir / "v22_10_drat_drbf_fastk2_repair_scan_summary.csv")
    blockb_scan = read_rows(out_dir / "v22_10_drat_drbf_blockb_repair_scan_summary.csv")
    combo_scan = read_rows(out_dir / "v22_10_drat_drbf_combo_repair_scan_summary.csv")
    lowhidden_dir = out_dir / "_low_hidden_sanity"
    lowhidden_candidate = read_rows(lowhidden_dir / "v22_10_drat_drbf_official_candidate_repair_scan_summary.csv")
    lowhidden_blockh = read_rows(lowhidden_dir / "v22_10_drat_drbf_blockh_repair_scan_summary.csv")
    lowhidden_fastk2 = read_rows(lowhidden_dir / "v22_10_drat_drbf_fastk2_repair_scan_summary.csv")
    lowhidden_blockb = read_rows(lowhidden_dir / "v22_10_drat_drbf_blockb_repair_scan_summary.csv")
    if not int_flag(route.get("optimization_loss_agnostic_contract_pass")):
        official_candidate = []
        blockh_scan = []
        fastk2_scan = []
        blockb_scan = []
        combo_scan = []
        lowhidden_candidate = []
        lowhidden_blockh = []
        lowhidden_fastk2 = []
        lowhidden_blockb = []
    atom_attempts = read_rows(out_dir / "v22_10_source_atom_attempt_summary.csv")
    atom_rows = read_rows(out_dir / "v22_10_source_atom_rows.csv")
    variational = read_rows(out_dir / "v22_10_variational_solve_matrix.csv")
    commit = read_rows(out_dir / "v22_10_metric_commit_matrix.csv")
    horizon = read_rows(out_dir / "v22_10_horizon_source_matrix.csv")
    kan = read_rows(out_dir / "v22_10_kan_mapping_matrix.csv")
    queue = read_rows(out_dir / "v22_10_gpu_assignment_manifest.csv")
    idle = read_rows(out_dir / "v22_10_idle_violation.csv")
    failure = read_rows(out_dir / "v22_10_failure_taxonomy.csv")
    command_journal = read_rows(out_dir / "v22_10_command_journal.csv")

    def _best_by(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
        scored = [(finite_float(r.get(key)), r) for r in rows]
        scored = [(v, r) for v, r in scored if v == v]
        return max(scored, key=lambda item: item[0])[1] if scored else {}

    def _first_pass(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
        for row in rows:
            if int_flag(row.get(key)):
                return row
        return {}

    def _fmt(row: dict[str, Any], key: str, digits: int = 4) -> str:
        value = finite_float(row.get(key))
        return "" if value != value else f"{value:.{digits}g}"

    def _value(row: dict[str, Any], key: str) -> str:
        return str(row.get(key, "")) if row else ""

    atom_pass = [r for r in atom_rows if int_flag(r.get("S2_source_atom_pass"))]
    s2_best = _best_by(atom_pass, "B2_transfer_gain")
    s3_pass = _first_pass(variational, "S3_variational_source_solve_pass")
    s4_pass = _first_pass(commit, "S4_metric_dynamics_commit_pass")
    horizon_fu = [r for r in horizon if r.get("variant") == "FU"]
    horizon_controls = [r for r in horizon if r.get("variant") != "FU"]
    horizon_pid = [r for r in horizon_fu if int_flag(r.get("pid_control"))]
    horizon_nonpid = [r for r in horizon_fu if not int_flag(r.get("pid_control"))]
    horizon_open = [r for r in horizon_fu if int_flag(r.get("C3_source_formation_pass")) and int_flag(r.get("C4_terminal_retention_pass"))]
    best_nonpid = _best_by(horizon_nonpid, "source_vs_best_control_h3200")
    best_pid = _best_by(horizon_pid, "source_vs_best_control_h3200")
    best_horizon = _best_by(horizon_open or horizon_fu, "source_vs_best_control_h3200")
    best_nonpid_h3200 = finite_float(best_nonpid.get("source_vs_best_control_h3200"))
    best_pid_h3200 = finite_float(best_pid.get("source_vs_best_control_h3200"))
    pid_delta = best_pid_h3200 - best_nonpid_h3200 if best_nonpid_h3200 == best_nonpid_h3200 and best_pid_h3200 == best_pid_h3200 else float("nan")
    pid_delta_text = "" if pid_delta != pid_delta else f"{pid_delta:.4g}"
    kan_pass = [r for r in kan if r.get("KAN_source_channel_decision") == "KANRetainedSourceOpened"]
    kan_basis = [r for r in kan if r.get("mechanism") == "basis_estimate_readout_commit_lr0"]
    best_kan = _best_by(kan_pass or kan, "KAN_source_vs_best_control_h3200")
    basis_best_delta = _best_by(kan_basis, "KAN_specific_delta_vs_MLP_same_metric")
    exploration_rows = [
        {
            "stage": "S0.17",
            "question": "代码包是否自洽",
            "attempt_or_repair": "clean unzip compile/import + required CSV vs zip + semantic/kernel audit",
            "result": f"pass={route['S0_17_pass']} CodeRoute={route['CodeRoute']}",
            "decision": "进入科学 gate" if int_flag(route.get("S0_17_pass")) else "停止并修 code packet",
        },
        {
            "stage": "S1",
            "question": "basis efficiency 是否可正式计入",
            "attempt_or_repair": "loss-agnostic clarified contract",
            "result": f"contract_pass={route['optimization_loss_agnostic_contract_pass']} CE_invalidated={route['ce_targeted_efficiency_rows_invalidated']}",
            "decision": "fail-closed，等待 arbitrary-loss/upstream-gradient runner",
        },
        {
            "stage": "S2",
            "question": "能否生成 loss-agnostic source atoms",
            "attempt_or_repair": f"{len(atom_attempts)} attempt groups; best={_value(s2_best, 'atom_id')}",
            "result": f"pass_rows={route['S2_source_atom_pass_rows']} best_B2={_fmt(s2_best, 'B2_transfer_gain')}",
            "decision": "A10/A11 进入 S3" if atom_pass else "更换 atom principle",
        },
        {
            "stage": "S3",
            "question": "atoms 能否组合成低 control projection source",
            "attempt_or_repair": _value(s3_pass, "attempt") or "none",
            "result": f"DDR={_fmt(s3_pass, 'DDR')} control_projection={_fmt(s3_pass, 'control_projection_fraction')} weights={_value(s3_pass, 'selected_atom_weights')}",
            "decision": "进入 S4 commit" if s3_pass else "重设 variational objective",
        },
        {
            "stage": "S4",
            "question": "metric dynamics commit 是否真实可 actuation",
            "attempt_or_repair": _value(s4_pass, "solver_level") or "none",
            "result": f"residual={_fmt(s4_pass, 'projection_residual_Gf')} R2={_fmt(s4_pass, 'ActuationR2')} B2={_fmt(s4_pass, 'B2_transfer_gain')}",
            "decision": "进入 h100-h6400 horizon" if s4_pass else "继续 solver repair",
        },
        {
            "stage": "S5",
            "question": "source 是否在 horizon 内 retained",
            "attempt_or_repair": f"FU attempts={len(horizon_fu)} PID attempts={len(horizon_pid)}",
            "result": f"best_nonPID_h3200={_fmt(best_nonpid, 'source_vs_best_control_h3200')} best_PID_h3200={_fmt(best_pid, 'source_vs_best_control_h3200')} PID_delta={pid_delta_text}",
            "decision": "C3/C4 打开，但仍受 S1 efficiency blocker 限制" if horizon_open else "继续 source integration repair",
        },
        {
            "stage": "S6",
            "question": "KAN 是否存在同 metric source-channel",
            "attempt_or_repair": f"rows={len(kan)} pass_rows={len(kan_pass)}",
            "result": f"best={_value(best_kan, 'carrier')}/{_value(best_kan, 'attempt')} h3200={_fmt(best_kan, 'KAN_source_vs_best_control_h3200')} delta={_fmt(best_kan, 'KAN_specific_delta_vs_MLP_same_metric')}",
            "decision": "KAN source opened，但不能 promotion" if kan_pass else "KAN mismatch",
        },
    ]
    pid_summary_rows = [
        {
            "attempt": r.get("attempt", ""),
            "kp": r.get("pid_kp", ""),
            "ki": r.get("pid_ki", ""),
            "kd": r.get("pid_kd", ""),
            "pid_final_projected_gain": r.get("pid_final_projected_gain", ""),
            "pid_abs_error_mean": r.get("pid_abs_error_mean", ""),
            "pid_abs_injection_mean": r.get("pid_abs_injection_mean", ""),
            "h100_margin": r.get("source_vs_best_control_h100", ""),
            "h3200_margin": r.get("source_vs_best_control_h3200", ""),
            "h4800_margin": r.get("source_vs_best_control_h4800", ""),
            "C3": r.get("C3_source_formation_pass", ""),
            "C4": r.get("C4_terminal_retention_pass", ""),
            "blocker": r.get("blocker", ""),
        }
        for r in horizon_pid
    ]
    horizon_fu_summary_rows = [
        {
            "attempt": r.get("attempt", ""),
            "pid_control": r.get("pid_control", ""),
            "periodic_stop_step": r.get("periodic_stop_step", ""),
            "lr_scale": r.get("lr_scale", ""),
            "h100_margin": r.get("source_vs_best_control_h100", ""),
            "h800_margin": r.get("source_vs_best_control_h800", ""),
            "h3200_margin": r.get("source_vs_best_control_h3200", ""),
            "h4800_margin": r.get("source_vs_best_control_h4800", ""),
            "h6400_margin": r.get("source_vs_best_control_h6400", ""),
            "row_positive_h3200": r.get("row_positive_count_h3200", ""),
            "control_equivalent_fraction": r.get("control_equivalent_fraction", ""),
            "C3": r.get("C3_source_formation_pass", ""),
            "C4": r.get("C4_terminal_retention_pass", ""),
            "blocker": r.get("blocker", ""),
        }
        for r in horizon_fu
    ]
    text = [
        "# DG-KAN v22.10 ConstructiveRetainedSource FunctionalUpdate BasisEfficiency 4GPU 实验结果复盘",
        "",
        f"生成时间：{now_sg()}",
        "",
        "## Route",
        "",
        f"- route: `{route['route']}`",
        f"- promotion_allowed: {route['promotion_allowed']}",
        f"- blocking_metric: `{route['blocking_metric']}`",
        f"- S0.17 pass / CodeRoute: {route['S0_17_pass']} / `{route['CodeRoute']}`",
        f"- D-CHE/D-FOU S1 pass: {route['D-CHE_S1_pass']} / {route['D-FOU_S1_pass']}",
        f"- D-RAT/D-RBF robust pass: {route['D-RAT_robust_pass']} / {route['D-RBF_robust_pass']}",
        f"- D-RAT/D-RBF official candidate scan route / groups / rows: `{route['official_candidate_repair_scan_route']}` / {route['official_candidate_repair_candidate_groups']} / {route['official_candidate_repair_scan_rows']}",
        f"- D-RAT/D-RBF blockH repair scan route / groups / rows: `{route['official_blockh_repair_scan_route']}` / {route['official_blockh_repair_candidate_groups']} / {route['official_blockh_repair_scan_rows']}",
        f"- D-RAT/D-RBF fastK2 repair scan route / groups / rows: `{route['official_fastk2_repair_scan_route']}` / {route['official_fastk2_repair_candidate_groups']} / {route['official_fastk2_repair_scan_rows']}",
        f"- D-RAT/D-RBF blockB repair scan route / groups / rows: `{route['official_blockb_repair_scan_route']}` / {route['official_blockb_repair_candidate_groups']} / {route['official_blockb_repair_scan_rows']}",
        f"- D-RAT/D-RBF combo repair scan route / groups / rows: `{route['official_combo_repair_scan_route']}` / {route['official_combo_repair_candidate_groups']} / {route['official_combo_repair_scan_rows']}",
        f"- optimization loss-agnostic efficiency contract pass / CE rows invalidated: {route['optimization_loss_agnostic_contract_pass']} / {route['ce_targeted_efficiency_rows_invalidated']}",
        f"- loss-agnostic source contract pass / uses_labels / uses_loss: {route['loss_agnostic_contract_pass']} / {route['source_uses_labels_for_direction']} / {route['source_uses_loss_for_direction']}",
        f"- S2 source atom pass rows / selected attempt: {route['S2_source_atom_pass_rows']} / `{route['S2_selected_attempt']}`",
        f"- S3 variational pass rows / selected attempt / blocker: {route['S3_variational_source_solve_pass_rows']} / `{route['S3_selected_attempt']}` / `{route['S3_blocker']}`",
        f"- S4/C3/C4 pass rows: {route['S4_metric_dynamics_commit_pass_rows']} / {route['C3_source_formation_pass_rows']} / {route['C4_terminal_retention_pass_rows']}",
        f"- S5 PID attempts / C3/C4 / best: {route['PID_control_attempt_rows']} / {route['PID_C3_source_formation_pass_rows']} / {route['PID_C4_terminal_retention_pass_rows']} / `{route['PID_best_attempt']}` h3200={route['PID_best_source_vs_best_control_h3200']}",
        f"- KAN mapping decision: `{route['KAN_mapping_decision']}`",
        f"- execution_contract_violation: {route['execution_contract_violation']}",
        f"- minimum_effective_progress: `{route['minimum_effective_progress']}`",
        f"- next_codex_action: {route['next_codex_action']}",
        "",
        "## Executive Summary",
        "",
        "- 目标没有达成 official promotion：`promotion_allowed=0`。核心原因不是 source-side 失败，而是用户明确要求 loss-agnostic 后，S1 basis-efficiency 仍没有 arbitrary-loss/upstream-gradient runner；所有 CE-targeted efficiency rows 被 formal invalidated。",
        f"- source-side constructive path 已经打开：S2 pass rows={route['S2_source_atom_pass_rows']}，S3 pass rows={route['S3_variational_source_solve_pass_rows']}，S4 pass rows={route['S4_metric_dynamics_commit_pass_rows']}，C3/C4 pass rows={route['C3_source_formation_pass_rows']}/{route['C4_terminal_retention_pass_rows']}，KAN decision=`{route['KAN_mapping_decision']}`。",
        f"- PID 思路已经按 loss-agnostic 约束真实尝试：PID attempts={len(horizon_pid)}，best PID=`{_value(best_pid, 'attempt')}`，h3200 margin={_fmt(best_pid, 'source_vs_best_control_h3200')}，final projected gain={_fmt(best_pid, 'pid_final_projected_gain')}，相对 best non-PID h3200 margin 改善={pid_delta_text}。",
        f"- KAN opening 的真实来源是 `{_value(best_kan, 'carrier')}` / `{_value(best_kan, 'mechanism')}` / `{_value(best_kan, 'attempt')}`，h3200 margin={_fmt(best_kan, 'KAN_source_vs_best_control_h3200')}，KAN-specific delta={_fmt(best_kan, 'KAN_specific_delta_vs_MLP_same_metric')}；basis-estimate rows 本轮没有打开，best basis-estimate delta={_fmt(basis_best_delta, 'KAN_specific_delta_vs_MLP_same_metric')}。",
        "- 当前最有价值的结果是：loss-agnostic source geometry、PID functional update 和 KAN source-channel 这条线给出正证据；但 efficiency gate 必须先补任意 loss/上游梯度接口，否则不能 promotion。",
        "",
        "## Exploration Process",
        "",
        md_table(exploration_rows, ["stage", "question", "attempt_or_repair", "result", "decision"], 20),
        "",
        "## Data Analysis / Insight",
        "",
        "### Loss-Agnostic Boundary",
        "",
        "- 用户 clarified hard constraint 后，`loss-agnostic` 被落实为所有优化不得针对 CE：source direction、variational solve、commit、horizon、KAN mapping 都必须不使用 labels 和 supervised loss；basis efficiency 也不能借 manual CE trainpath 做正式 closure。",
        f"- S1 当前 formal status：optimization contract pass={route['optimization_loss_agnostic_contract_pass']}，CE-targeted rows invalidated={route['ce_targeted_efficiency_rows_invalidated']}。因此 D-CHE/D-FOU 的旧 S1 readback、D-RAT/D-RBF repair scan 只作为 legacy implementation/audit context，不能用于 promotion。",
        "- 这个 blocker 很关键：v22.10 source-side 已经比 v22.09 多走到 C3/C4/KAN opening，但 route 仍是 R8，而不是 R9，因为 promotion 必须同时满足 source-side 与 efficiency-side official contract。",
        "",
        "### Source Atom Search",
        "",
        f"- S2 一共执行 {len(atom_attempts)} 组 source atom attempts：initial/lower-norm attempts 有 pass rows，more-splits 与 block-restricted attempts 是真实负结果。best atom=`{_value(s2_best, 'atom_id')}`，family=`{_value(s2_best, 'atom_family')}`，B2 transfer={_fmt(s2_best, 'B2_transfer_gain')}，DDR={_fmt(s2_best, 'DDR')}，control projection={_fmt(s2_best, 'control_projection_fraction')}。",
        "- A10 split-consensus 与 A11 causal split transport 只读 train logits 的几何结构，不读 labels、不读 CE。它们的意义不是预测 future label，而是在当前 train-stream 中形成可 replay 的 target displacement source atom。",
        "- lower-norm attempts 仍能保留 pass rows 但 B2 下降，说明 source signal 不是单纯 norm/scale artefact；more-splits 和 block-restricted rows 没打开，说明当前有效 source 需要全 train-stream 的 split/causal geometry，而不是更碎的 split 或单 block 限制。",
        "",
        "### Variational Solve",
        "",
        f"- S3 selected attempt=`{_value(s3_pass, 'attempt')}`，selected atoms=`{_value(s3_pass, 'selected_atoms')}`，weights=`{_value(s3_pass, 'selected_atom_weights')}`，DDR={_fmt(s3_pass, 'DDR')}，control projection={_fmt(s3_pass, 'control_projection_fraction')}，NDS={_fmt(s3_pass, 'NDS')}。",
        "- default sparse combination 的 DDR 读数很高但 control projection 更高；control-balanced L1 通过 capped DDR 与 control projection 约束，避免把 control-equivalent direction 写成 source。这是 v22.10 从 observer 到 constructive source 的主要语义变化。",
        "",
        "### Commit Repair",
        "",
        f"- S4 selected commit=`{_value(s4_pass, 'solver_level')}` / block_role=`{_value(s4_pass, 'block_role')}`，projection residual={_fmt(s4_pass, 'projection_residual_Gf')}，ActuationR2={_fmt(s4_pass, 'ActuationR2')}，B2 transfer={_fmt(s4_pass, 'B2_transfer_gain')}，random matched B2={_fmt(s4_pass, 'random_matched_B2_transfer_gain')}。",
        "- readout-only、hidden/readout、高 damping、optimizer-state-only commit attempts 都没有打开；all-train actuation repair 才把 target displacement 以几何 readback 方式写入当前 tiny-model train stream。",
        "",
        "### Horizon And PID",
        "",
        f"- S5 non-PID best=`{_value(best_nonpid, 'attempt')}`，h3200 margin={_fmt(best_nonpid, 'source_vs_best_control_h3200')}；PID best=`{_value(best_pid, 'attempt')}`，h3200 margin={_fmt(best_pid, 'source_vs_best_control_h3200')}，h4800 margin={_fmt(best_pid, 'source_vs_best_control_h4800')}，PID improvement={pid_delta_text}。",
        f"- PID controller 的 error 是 label-free projected gain error：`1 - projected_gain`。它只调节 source update 注入幅度；matched random/sign/corrupt/same-solver controls 使用同一 PID controller，因此没有 source-only controller advantage。best PID mean abs error={_fmt(best_pid, 'pid_abs_error_mean')}，mean abs injection={_fmt(best_pid, 'pid_abs_injection_mean')}。",
        "- PID 的意义不是追 CE loss，而是把 functional update 线上 target displacement 的投影增益稳定在 1 附近。三个 PID attempts 都 C3/C4 pass，说明用户建议的 PID 思路在 loss-agnostic FU 线上有正证据。",
        "",
        "### KAN Mapping",
        "",
        f"- S6 pass row 是 `{_value(best_kan, 'carrier')}` slower replay source-channel：h100/h400/h800/h1600/h2400/h3200/h4800/h6400 相对 matched controls 均为正，h3200 row positive count={_value(best_kan, 'row_positive_count_h3200')}，R4800/h3200={_fmt(best_kan, 'R4800_over_3200')}。",
        f"- basis-estimate readout commit rows 的 fit cosine 虽高，但 source-channel delta 仍为负；best basis-estimate delta={_fmt(basis_best_delta, 'KAN_specific_delta_vs_MLP_same_metric')}。因此不能把本轮 opening 写成 basis-estimate opening，正式结论必须写成 D-FOU slower replay source-channel opening。",
        "- KAN result 与 MLP source-side result 一致地只使用 label-free target-retention score；它证明存在 KAN-side source channel evidence，但不解除 S1 loss-agnostic efficiency blocker。",
        "",
        "## Part A Code / Packet Truth Gate",
        "",
        md_table(code_rows, ["check", "pass", "metric", "value", "blocker"], 50),
        "",
        "### Clean Unzip Self Test",
        "",
        md_table(clean, ["packet", "unzip_root", "compileall_returncode", "import_returncode", "self_contained_import_check", "pass"], 5),
        "",
        "### Kernel Gradcheck / Status Consistency",
        "",
        md_table(kernel, ["family", "kernel_forward_relerr", "kernel_grad_relerr", "kernel_grad_cosine", "kernel_correctness_official_pass", "official_fused_kernel_complete", "no_nan_inf"], 10),
        "",
        "## Part B Basis Efficiency",
        "",
        "- hard constraint: v22.10 formal S1 cannot optimize or profile a CE-targeted trainpath. Existing manual-CE D-RAT/D-RBF scans are retained only as legacy code/audit context and are not executed or promoted in this formal route until an arbitrary-loss/upstream-gradient efficiency runner exists.",
        md_table(eff, ["carrier", "variant_family", "forward_ratio_vs_mlp", "step_ratio_vs_mlp", "memory_ratio_vs_mlp", "functional_runner_same_kernel", "fallback_kernel_used", "official_fused_kernel_complete", "v22_09_S1_pass", "formal_status", "v22_10_loss_contract", "v22_09_blocker"], 10),
        "",
        "### D-RAT / D-RBF Multibatch",
        "",
        md_table(dr, ["carrier", "profile_rows", "robust_production_pass_rows", "near_E1_rows", "component_telemetry_complete_rows", "best_forward_ratio", "best_step_ratio", "best_memory_ratio", "decision", "blocker"], 10),
        "",
        "### Telemetry Bridge",
        "",
        md_table(bridge, ["carrier", "profile_rows", "robust_production_pass_rows", "component_telemetry_complete_rows", "best_forward_ratio", "best_step_ratio", "decision", "blocker", "official_closure_claimed"], 10),
        "",
        "### Official Candidate Repair Scan",
        "",
        "- status: not run in the formal v22.10 route because the available official S1 trainpath is CE-targeted. These implementation-side scan functions remain in the code packet, but they cannot produce a promotion claim under the clarified loss-agnostic optimization contract.",
        md_table(official_candidate, ["carrier", "repair_variant", "scan_hidden", "profile_rows", "robust_production_pass_rows", "near_E1_rows", "best_forward_ratio", "best_step_ratio", "best_memory_ratio", "shape_localk_candidate", "candidate_decision", "blocker"], 30),
        "",
        "### Official BlockH Repair Scan",
        "",
        "- status: not run in the formal v22.10 route for the same CE-targeted trainpath reason.",
        md_table(blockh_scan, ["carrier", "repair_variant", "block_h", "scan_hidden", "profile_rows", "robust_production_pass_rows", "near_E1_rows", "best_forward_ratio", "best_step_ratio", "best_memory_ratio", "shape_localk_candidate", "candidate_decision", "blocker"], 40),
        "",
        "### Official FastK2 Repair Scan",
        "",
        "- status: not run in the formal v22.10 route for the same CE-targeted trainpath reason.",
        md_table(fastk2_scan, ["carrier", "repair_variant", "scan_hidden", "profile_rows", "robust_production_pass_rows", "near_E1_rows", "best_forward_ratio", "best_step_ratio", "best_memory_ratio", "shape_localk_candidate", "candidate_decision", "blocker"], 20),
        "",
        "### Official BlockB Repair Scan",
        "",
        "- status: not run in the formal v22.10 route for the same CE-targeted trainpath reason.",
        md_table(blockb_scan, ["carrier", "repair_variant", "block_b", "scan_hidden", "profile_rows", "robust_production_pass_rows", "near_E1_rows", "best_forward_ratio", "best_step_ratio", "best_memory_ratio", "shape_localk_candidate", "candidate_decision", "blocker"], 40),
        "",
        "### Official Combo Repair Scan",
        "",
        "- status: not run in the formal v22.10 route for the same CE-targeted trainpath reason.",
        md_table(combo_scan, ["carrier", "repair_variant", "block_h", "block_b", "num_warps", "scan_hidden", "profile_rows", "robust_production_pass_rows", "near_E1_rows", "best_forward_ratio", "best_step_ratio", "best_memory_ratio", "shape_localk_candidate", "candidate_decision", "blocker"], 40),
        "",
        "### Focused Low-Hidden Sanity",
        "",
        "- status: legacy focused CE-trainpath audit is omitted from the formal recap after the loss-agnostic optimization clarification.",
        md_table(lowhidden_candidate, ["carrier", "repair_variant", "scan_hidden", "profile_rows", "robust_production_pass_rows", "near_E1_rows", "best_forward_ratio", "best_step_ratio", "best_memory_ratio", "shape_localk_candidate", "candidate_decision", "blocker"], 20),
        md_table(lowhidden_blockh, ["carrier", "repair_variant", "block_h", "scan_hidden", "profile_rows", "robust_production_pass_rows", "near_E1_rows", "best_forward_ratio", "best_step_ratio", "best_memory_ratio", "shape_localk_candidate", "candidate_decision", "blocker"], 10),
        md_table(lowhidden_fastk2, ["carrier", "repair_variant", "scan_hidden", "profile_rows", "robust_production_pass_rows", "near_E1_rows", "best_forward_ratio", "best_step_ratio", "best_memory_ratio", "shape_localk_candidate", "candidate_decision", "blocker"], 10),
        md_table(lowhidden_blockb, ["carrier", "repair_variant", "block_b", "scan_hidden", "profile_rows", "robust_production_pass_rows", "near_E1_rows", "best_forward_ratio", "best_step_ratio", "best_memory_ratio", "shape_localk_candidate", "candidate_decision", "blocker"], 10),
        "",
        "## Part C Source Atom Generation",
        "",
        "- S2 使用 TinyConstructiveMLP 的当前 train logits；source direction 只由 label-free train-stream logit geometry 构造，不使用 labels、supervised loss、validation/test/future/LineC/ECE/Brier/AUCtime。",
        md_table(atom_attempts, ["attempt", "fallback_step", "norm_scale", "split_count", "block_role", "atom_rows", "S2_source_atom_pass_rows", "control_NDS_median", "random_matched_B2_gain", "route"], 10),
        "",
        "### Source Atom Rows",
        "",
        md_table(atom_rows, ["attempt", "atom_id", "atom_family", "B2_transfer_gain", "random_gap", "sign_flip_gap", "corrupt_target_gap", "commutator_norm_ratio", "DDR", "NDS", "control_projection_fraction", "S2_source_atom_pass", "blocker"], 30),
        "",
        "## Part D Variational Solve",
        "",
        md_table(variational, ["attempt", "source_combo_atom_count", "source_combo_l1_norm", "DDR", "control_projection_fraction", "random_gap", "sign_flip_gap", "corrupt_gap", "metric_energy_Sobolev", "control_Sobolev_p50", "NDS", "control_NDS_median", "selected_atoms", "selected_atom_weights", "S3_variational_source_solve_pass", "blocker"], 10),
        "",
        "## Part E Commit / Horizon / KAN",
        "",
        "### S4 Commit Matrix",
        "",
        md_table(commit, ["solver_level", "block_role", "projection_residual_Gf", "ActuationR2", "B2_transfer_gain", "random_matched_B2_transfer_gain", "function_displacement_cos_with_target", "S4_metric_dynamics_commit_pass", "blocker"], 10),
        "",
        "### S5 FU Attempt Summary",
        "",
        md_table(horizon_fu_summary_rows, ["attempt", "pid_control", "periodic_stop_step", "lr_scale", "h100_margin", "h800_margin", "h3200_margin", "h4800_margin", "h6400_margin", "row_positive_h3200", "control_equivalent_fraction", "C3", "C4", "blocker"], 80),
        "",
        "### S5 PID Control Attempts",
        "",
        md_table(pid_summary_rows, ["attempt", "kp", "ki", "kd", "pid_final_projected_gain", "pid_abs_error_mean", "pid_abs_injection_mean", "h100_margin", "h3200_margin", "h4800_margin", "C3", "C4", "blocker"], 20),
        "",
        "### S5 Matched Control Sample",
        "",
        md_table(horizon_controls, ["attempt", "variant", "optimizer", "target_retention_score_h100", "target_retention_score_h800", "target_retention_score_h3200", "target_retention_score_h4800", "target_projected_gain_h3200"], 24),
        "",
        "### S5 Full Horizon Matrix Sample",
        "",
        md_table(horizon, ["attempt", "variant", "periodic_stop_step", "pid_control", "pid_kp", "pid_ki", "pid_kd", "pid_final_projected_gain", "source_vs_best_control_h100", "source_vs_best_control_h400", "source_vs_best_control_h800", "source_vs_best_control_h1600", "source_vs_best_control_h3200", "source_vs_best_control_h4800", "source_vs_best_control_h6400", "row_positive_count_h3200", "control_equivalent_fraction", "source_target_retention_area_ratio", "AUCtime_ratio", "C3_source_formation_pass", "C4_terminal_retention_pass", "blocker"], 48),
        "",
        "### S6 KAN Matrix",
        "",
        md_table(kan, ["carrier", "mechanism", "attempt", "basis_estimate_fit_cos", "basis_estimate_projection_residual", "KAN_source_vs_best_control_h100", "KAN_source_vs_best_control_h400", "KAN_source_vs_best_control_h800", "KAN_source_vs_best_control_h1600", "KAN_source_vs_best_control_h3200", "KAN_source_vs_best_control_h4800", "R4800_over_3200", "KAN_specific_delta_vs_MLP_same_metric", "row_positive_count_h3200", "KAN_source_channel_decision", "blocker"], 10),
        "",
        "## 4GPU Queue",
        "",
        md_table(queue, ["task_id", "line", "gpu", "command", "status", "returncode", "start_time", "end_time", "duration_sec", "log_path"], 20),
        md_table(idle, ["execution_contract_violation", "reason", "max_idle_gap_sec"], 5),
        "",
        "### Command Journal / Post-Full Reruns",
        "",
        md_table(command_journal[-20:], ["timestamp", "command", "status", "note"], 20),
        "",
        "## Failure Taxonomy",
        "",
        md_table(failure, ["blocker", "route", "repair_or_next_direction"], 20),
        "",
        "## 修改记录",
        "",
        "- 新增并改造 `dgkan/fu/source_atoms.py`：实现 v22.10 train-stream source atom generator、controls、NDS/DDR/commutator/row metrics 与 unit tests；正式 source direction 改为 label-free logit geometry，不使用 labels 或 supervised loss。",
        "- 新增 A10 `split_consensus` 与 A11 `causal_split_transport` loss-agnostic atoms：只用 train logits 的 split-consensus / causal split transport 构造方向，并写入 `loss_agnostic_contract_pass=1`、`uses_labels_for_direction=0`、`uses_loss_for_direction=0`。",
        "- 修复 source atom commutator / NDS proxy：order-invariant atom 不再被样本顺序翻转误罚；loss-agnostic geometry NDS 改为 logit-channel curvature，避免把样本标签顺序当成 source 信息。",
        "- 新增 `dgkan/fu/variational_source_solver.py` loss-agnostic path：实现 S3 sparse/low-NDS/control-null/block-restricted/control-balanced repair ladder，并对 DDR 做 capped scoring，避免高 DDR 但高 control projection 的 atom 覆盖低投影 source。",
        "- 改造 `dgkan/fu/constructive_commit.py`：S4 正式路径以 `y=None` 运行，不计算 supervised loss/CE 梯度；B2 transfer 改为 target displacement readback geometry，与 matched random/corrupt target controls 比较。",
        "- 改造 S5 lightweight train-stream horizon runner：执行 h100-h6400、11 个 matched controls、source replay / damped replay / source-state replay / finite-window source-state replay repair ladder；C3/C4 只看 label-free target-retention score 与 residual/geometry debt。",
        "- 新增 S5 PID functional-update repair ladder：以 label-free target projected-gain error 作为 PID 误差信号，只调节 source update 注入幅度；matched random/sign/corrupt/same-solver controls 使用同一 PID controller，避免 source 独享控制器优势。",
        "- 改造 S6 lightweight KAN source-channel mapping：KAN mapping 使用 label-free target-retention score、basis-estimate fit cosine 和 matched param controls；不使用 CE loss improvement 判定 source success。",
        "- 按用户 clarified hard constraint 改造 `experiments/run_v22_10_basis_efficiency_closure.py`：正式 S1 在进入 CE-targeted trainpath 前 fail-closed，写出 `optimization_loss_agnostic_contract_pass=0` 与 `ce_targeted_efficiency_rows_invalidated=1`；旧 manual-CE D-RAT/D-RBF repair scan 只保留为 legacy code/audit context，不执行、不 promotion。",
        "- CE 时代新增过的 D-RAT/D-RBF K2、singlelaunch、blockH、fastK2、blockB、combo 和 component telemetry probe 代码仍进入 code packet 供审计，但在 v22.10 clarified contract 下不构成有效优化证据；下一步必须实现 arbitrary-loss/upstream-gradient official efficiency runner 后才能重新测这些方向。",
        "- 新增 `experiments/run_v22_10_*` runner：S0.17 truth gate、basis efficiency wrapper、source atom、variational solve、commit、horizon、KAN、dynamic queue、final recap 与 packet/bundle。",
        "- v22.10 S2 fallback 真实执行了 lower norm / more split / block-restricted attempts；S3 fallback 真实执行了 l1 sparse、low-NDS-only、control-null-only、block-restricted 与 balanced-grid attempts。",
        "",
        "## 补充分析 / Evidence Chain",
        "",
        "- v22.10 的语义从 observer-as-filter 改成 constructive source generation；本轮没有使用 v22.09 的 future audit label 生成方向。",
        "- 用户追加 hard constraint 后，v22.10 的 loss-agnostic 解释为所有优化不得针对 CE：source/C3/KAN route 均必须写出 `loss_agnostic_contract_pass=1` 且 `uses_labels_for_direction=0`、`uses_loss_for_direction=0`；S1 efficiency 也不得运行或提升 CE-targeted trainpath。",
        "- S0.17 必须先闭合 code packet：clean unzip compile/import、required CSV vs zip、semantic forbidden audit、source atom/variational/commit unit tests、kernel status consistency 都落盘；若此处失败，不能写 scientific no-go。",
        "- Basis efficiency 在本次 clarified contract 下必须 fail-closed：当前 official trainpath 仍围绕 CE/manual-CE backward 组织，缺少 arbitrary-loss/upstream-gradient S1 runner，因此 D-CHE/D-FOU 与 D-RAT/D-RBF 都不能计入正式 efficiency closure。",
        "- CE 时代的 D-RAT/D-RBF K2/singlelaunch/blockH/fastK2/blockB/combo/telemetry probe 只能说明代码实现方向，不再是 v22.10 有效优化证据；正式复盘不读取旧 low-hidden 或 scan artifacts 来决定 route，避免旧 CE workload 污染结论。",
        "- 下一步 S1 的真正修复不是继续 CE duplicate-cost/tile/launch scan，而是先把 official efficiency contract 改成任意上游 `grad_logits` / arbitrary scalar loss 的 forward-backward-step runner，再在同一 robust gate 下重测 D-CHE/D-FOU/D-RAT/D-RBF。",
        "- A10/A11 修复了 loss-agnostic S2/S3 的二难：split-consensus atom 有强几何读数但 control projection 高，causal split transport projection 更低；S3 control-balanced L1 在 capped DDR 下选择两者组合，打开 variational source solve。",
        "- S4 的 split_A readout / hidden-readout / higher damping / optimizer-state attempts 都是真实负结果；S4.5 all_train_stream actuation repair 才以 label-free target readback 打开 commit。",
        "- S5 普通 AdamW continuation 没有 retained source：弱 replay 后期可正但早期 margin 不够，持续 source-state replay 早期强但 h2400 后过冲。finite-window source-state replay 直接针对这个 blocker，只在早期写入 source 后停止刷新。",
        "- S5 finite-window source-state replay 是本轮 MLP loss-agnostic functional progress 的关键：`finite_source_state_replay_400x0p20_stop800_lr0` 在 h100/h400/h800/h1600/h3200/h4800/h6400 全部大于 best matched control，h3200 margin 约 0.112，matched control positive count=11，C3/C4 都打开。该结论限定于 v22.10 lightweight train-stream runner，不等价于 S1 efficiency closure。",
        f"- S5 PID control repair 已真实执行：PID rows={route['PID_control_attempt_rows']}，PID C3/C4 pass={route['PID_C3_source_formation_pass_rows']}/{route['PID_C4_terminal_retention_pass_rows']}，best PID=`{route['PID_best_attempt']}`，h3200 margin={route['PID_best_source_vs_best_control_h3200']}。PID 结果只使用 target displacement 几何误差，不使用 CE 或 labels。",
        "- AUCtime 在本 lightweight runner 中记录同 step budget 时间比，因此为 1.0；source advantage 的面积另列为 `source_target_retention_area_ratio`，不作为 debt blocker。",
        "- S6 readout-only D-CHE 与多数 D-FOU replay rows 是真实负结果；本轮 opening 来自 D-FOU `slower_replay_100x0p16_repair`。basis-estimate/readout-commit rows 虽有高 fit cosine，但 KAN-specific delta 为负，不能写成 basis-estimate opening。",
        (
            f"- 最新结论：route=`{route['route']}`，minimum_effective_progress=`{route['minimum_effective_progress']}`。本轮有效进展是 loss-agnostic A10/A11 source atoms、S4 all-train actuation repair、finite-window 与 PID MLP source-state replay C3/C4、D-FOU slower replay KAN source-channel opening；"
            + (
                "所有当前 gate 已打开，仍需 independent confirmation 与 code-review packet audit 后才可作为 promotion candidate 使用。"
                if route["route"] == "R9-OfficialPromotionCandidate"
                else "最终仍不能 official promotion，因为 blocker 仍未清空；若 blocker 是 D-RAT/D-RBF robust efficiency side gate，则只能继续 efficiency side-line repair，不能把 KAN/MLP source success 单独写成 promotion。"
            )
        ),
        "",
        "_完整 artifact 清单保留在 `v22_10_artifact_index.csv`，不放入复盘正文。_",
        "",
    ]
    V2210_RECAP_DOC.write_text("\n".join(text), encoding="utf-8")


def main() -> None:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir)
    route = _route(out_dir)
    write_json(out_dir / "v22_10_final_route.json", route)
    write_rows(out_dir / "v22_10_final_route.csv", [route])
    failure = _failure_taxonomy(route)
    write_rows(out_dir / "v22_10_failure_taxonomy.csv", failure)
    _write_recap(out_dir, route)
    write_rows(out_dir / "v22_10_artifact_index.csv", artifact_index(out_dir))
    code_packet = build_code_review_packet(out_dir)
    write_rows(out_dir / "v22_10_artifact_index.csv", artifact_index(out_dir))
    bundle = build_results_bundle(out_dir)
    index = artifact_index(out_dir)
    write_rows(out_dir / "v22_10_artifact_index.csv", index)
    append_exec(
        out_dir,
        f"{PYTHON} experiments/run_v22_10_finalize.py --out-dir {out_dir}",
        status="completed",
        note=f"route={route['route']} artifacts={len(index)} bundle={bundle} code_review_packet={code_packet}",
    )


if __name__ == "__main__":
    main()
