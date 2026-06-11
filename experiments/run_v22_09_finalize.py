#!/usr/bin/env python3
"""v22.09 final route, recap, artifact index, and bundles."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_v22_09_common import (  # noqa: E402
    PYTHON,
    V2209_RECAP_DOC,
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


def _best(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    valid = [r for r in rows if finite_float(r.get(key), float("nan")) == finite_float(r.get(key), float("nan"))]
    return max(valid, key=lambda r: finite_float(r.get(key), -1.0e99)) if valid else {}


def _route(out_dir: Path) -> dict[str, Any]:
    code = read_json(out_dir / "v22_09_code_route_decision.json")
    basis = read_json(out_dir / "v22_09_basis_efficiency_route.json")
    bridge_summary = read_rows(out_dir / "v22_09_drat_drbf_telemetry_bridge_summary.csv")
    shape_scan = read_json(out_dir / "v22_09_drat_drbf_shape_localk_repair_scan_route.json")
    observer = read_json(out_dir / "v22_09_retained_source_observer_route.json")
    bvfr = read_json(out_dir / "v22_09_boundary_flow_replay_route.json")
    osh = read_json(out_dir / "v22_09_optimizer_state_holonomy_route.json")
    rost = read_json(out_dir / "v22_09_row_orthogonal_source_transport_route.json")
    ivsc = read_json(out_dir / "v22_09_info_volume_signal_channel_route.json")
    fsmr = read_json(out_dir / "v22_09_fast_slow_memory_resonance_route.json")
    fsmr_repair = read_json(out_dir / "v22_09_fast_slow_memory_damped_repair_route.json")
    trac = read_json(out_dir / "v22_09_time_reversal_adjoint_route.json")
    trac_repair = read_json(out_dir / "v22_09_time_reversal_adjoint_damped_repair_route.json")
    solver = read_json(out_dir / "v22_09_metric_dynamics_solver_route.json")
    terminal = read_json(out_dir / "v22_09_terminal_preservation_route.json")
    kan = read_json(out_dir / "v22_09_kan_mapping_route.json")
    queue = read_json(out_dir / "v22_09_queue_drain_report.json")
    idle_rows = read_rows(out_dir / "v22_09_idle_violation.csv")
    queue_violation = any(int_flag(r.get("execution_contract_violation")) for r in idle_rows)

    blockers: list[str] = []
    if not int_flag(code.get("S0_16_pass")):
        blockers.append("S0.16_code_packet_truth_gate")
    if str(code.get("CodeRoute", "")) != "R0-CodePacketSelfContained":
        blockers.append("code_packet_not_self_contained")
    if not int_flag(basis.get("D-CHE_S1_pass")) or not int_flag(basis.get("D-FOU_S1_pass")):
        blockers.append("D-CHE_D-FOU_efficiency_reconfirm_gate")
    if not int_flag(basis.get("D-RAT_D-RBF_multibatch_closed")):
        blockers.append("D-RAT_D-RBF_multibatch_robust_or_telemetry")
    if not int_flag(observer.get("C0_retained_source_observer_pass_rows")):
        blockers.append("retained_source_certificate_gate")
    if bvfr and not int_flag(bvfr.get("official_C3_pass_rows")):
        blockers.append("post_boundary_BVFR_gate")
    if osh and not int_flag(osh.get("official_C3_pass_rows")):
        blockers.append("post_state_holonomy_gate")
    if rost and not int_flag(rost.get("official_C3_pass_rows")):
        blockers.append("post_row_orthogonal_ROST_gate")
    if ivsc and not int_flag(ivsc.get("official_C3_pass_rows")):
        blockers.append("post_info_volume_IVSC_gate")
    if fsmr and not int_flag(fsmr.get("official_C3_pass_rows")):
        blockers.append("post_fast_slow_FSMR_gate")
    if fsmr_repair and not int_flag(fsmr_repair.get("official_C3_pass_rows")):
        blockers.append("post_fast_slow_FSMR_damped_repair_gate")
    if trac and not int_flag(trac.get("official_C3_pass_rows")):
        blockers.append("post_time_reversal_TRAC_gate")
    if trac_repair and not int_flag(trac_repair.get("official_C3_pass_rows")):
        blockers.append("post_time_reversal_TRAC_damped_repair_gate")
    if not int_flag(solver.get("C2_true_block_solver_pass_rows")):
        blockers.append("C2_true_block_solver_or_observer_gate")
    if not int_flag(solver.get("C3_source_formation_pass_rows")):
        blockers.append("C3_source_formation_gate")
    if not int_flag(terminal.get("C4_terminal_preservation_pass_rows")):
        blockers.append("terminal_preservation_not_entered_or_failed_C3_gate")
    if str(kan.get("decision", "")) == "KANMappingNotEntered":
        blockers.append("KAN_mapping_not_entered_C3_or_C4_gate")
    if queue_violation:
        blockers.append("4GPU_queue_idle_violation")

    if not int_flag(code.get("S0_16_pass")) or str(code.get("CodeRoute", "")) != "R0-CodePacketSelfContained":
        route = "R0-CodePacketNotSelfContained"
        action = "rebuild packet from final unzip root and rerun clean compile/import before scientific route"
    elif not int_flag(basis.get("D-CHE_S1_pass")) or not int_flag(basis.get("D-FOU_S1_pass")):
        route = "R-BasisEfficiencyRegression_v22.09"
        action = "check fallback kernel path, profiler path, and metric solver overhead for D-CHE/D-FOU"
    elif fsmr_repair and int_flag(fsmr_repair.get("fresh_C3_damped_repair_pass_rows")):
        route = "PostFastSlowMemoryDampedRepairFreshSmokeOfficialBlocked_v22.09"
        action = "FSMR damped repair opened a fresh smoke row, but official observer/C2/C3 gates remain closed; repeat verify before any promotion"
    elif fsmr and int_flag(fsmr.get("fresh_C3_fast_slow_memory_pass_rows")):
        route = "PostFastSlowMemoryFreshSmokeOfficialBlocked_v22.09"
        action = "FSMR opened a fresh smoke row, but official observer/C2/C3 gates remain closed; repeat verify before any promotion"
    elif ivsc and int_flag(ivsc.get("fresh_C3_info_volume_pass_rows")):
        route = "PostInfoVolumeFreshSmokeOfficialBlocked_v22.09"
        action = "IVSC opened a fresh smoke row, but official observer/C2/C3 gates remain closed; repeat verify before any promotion"
    elif rost and int_flag(rost.get("fresh_C3_row_orthogonal_pass_rows")):
        route = "PostRowOrthogonalFreshSmokeOfficialBlocked_v22.09"
        action = "ROST opened a fresh smoke row, but official observer/C2/C3 gates remain closed; repeat verify before any promotion"
    elif osh and int_flag(osh.get("fresh_C3_state_holonomy_pass_rows")):
        route = "PostStateHolonomyFreshSmokeOfficialBlocked_v22.09"
        action = "OSH opened a fresh smoke row, but official observer/C2/C3 gates remain closed; repeat verify before any promotion"
    elif bvfr and int_flag(bvfr.get("fresh_C3_boundary_replay_pass_rows")):
        route = "PostBoundaryBVFRFreshSmokeOfficialBlocked_v22.09"
        action = "BVFR opened a fresh smoke row, but official observer/C2/C3 gates remain closed; repeat verify before any promotion"
    elif trac_repair and int_flag(trac_repair.get("fresh_C3_trac_damped_repair_pass_rows")):
        route = "PostTimeReversalAdjointDampedRepairFreshSmokeOfficialBlocked_v22.09"
        action = "TRAC damped repair opened a fresh smoke row, but official observer/C2/C3 gates remain closed; repeat verify before any promotion"
    elif trac and int_flag(trac.get("fresh_C3_time_reversal_adjoint_pass_rows")):
        route = "PostTimeReversalAdjointFreshSmokeOfficialBlocked_v22.09"
        action = "TRAC opened a fresh smoke row, but official observer/C2/C3 gates remain closed; repeat verify before any promotion"
    elif not int_flag(observer.get("C0_retained_source_observer_pass_rows")):
        route = "RetainedSourceObserverLocalNoGo_v22.09"
        action = "C-O13/C-O14/C-O15, BVFR, OSH, ROST, IVSC, FSMR, FSMR damped repair, TRAC, and TRAC damped repair did not open official source observability; record boundary or change first principle again"
    elif not int_flag(solver.get("C2_true_block_solver_pass_rows")):
        route = "R-SolverProjectionBlocked_v22.09"
        action = "reduce target rank, use damping and block-restricted low-rank CG before C3"
    elif not int_flag(solver.get("C3_source_formation_pass_rows")):
        route = "R-C3SourceFormationBlocked_v22.09"
        action = "compare direct commit, optimizer-state injection, slow source-state, block memory, and row-orthogonal updates"
    elif not int_flag(terminal.get("C4_terminal_preservation_pass_rows")):
        route = "R-TerminalPreservationBlocked_v22.09"
        action = "run terminal erosion autopsy and source-preserving late projection"
    elif str(kan.get("decision", "")) == "KANMappingNotEntered":
        route = "KANMappingNotEntered"
        action = "enter KAN mapping only after MLP C3/C4"
    else:
        route = "S5-KANSourceChannelOpened_v22.09"
        action = "independent confirmation and same-metric MLP/KAN comparison"

    minimum_progress = []
    if int_flag(code.get("S0_16_pass")) and str(code.get("CodeRoute", "")) == "R0-CodePacketSelfContained":
        minimum_progress.append("A-CodeClosure")
    if int_flag(basis.get("D-RAT_robust_pass")) or int_flag(basis.get("D-RBF_robust_pass")):
        minimum_progress.append("B-EfficiencyClosure")
    if shape_scan and int_flag(shape_scan.get("shape_localk_candidate_groups")):
        minimum_progress.append("B-ShapeLocalKRepairCandidate")
    if finite_float(observer.get("best_heldout_precision_at_top20"), -1.0) >= 0.40:
        minimum_progress.append("C-ObserverProgress")
    if bvfr and int_flag(bvfr.get("fresh_C3_boundary_replay_pass_rows")):
        minimum_progress.append("C-BVFRFreshSmoke")
    if osh and int_flag(osh.get("fresh_C3_state_holonomy_pass_rows")):
        minimum_progress.append("C-OSHFreshSmoke")
    if rost and int_flag(rost.get("fresh_C3_row_orthogonal_pass_rows")):
        minimum_progress.append("C-ROSTFreshSmoke")
    if ivsc and int_flag(ivsc.get("fresh_C3_info_volume_pass_rows")):
        minimum_progress.append("C-IVSCFreshSmoke")
    if fsmr and int_flag(fsmr.get("fresh_C3_fast_slow_memory_pass_rows")):
        minimum_progress.append("C-FSMRFreshSmoke")
    if fsmr_repair and int_flag(fsmr_repair.get("fresh_C3_damped_repair_pass_rows")):
        minimum_progress.append("C-FSMRDampedFreshSmoke")
    if trac and int_flag(trac.get("fresh_C3_time_reversal_adjoint_pass_rows")):
        minimum_progress.append("C-TRACFreshSmoke")
    if trac_repair and int_flag(trac_repair.get("fresh_C3_trac_damped_repair_pass_rows")):
        minimum_progress.append("C-TRACDampedFreshSmoke")
    if int_flag(solver.get("C3_source_formation_pass_rows")):
        minimum_progress.append("D-FunctionalProgress")
    if route == "RetainedSourceObserverLocalNoGo_v22.09" and not minimum_progress:
        minimum_progress.append("E-TheoryBoundary")

    dedup: list[str] = []
    for item in blockers:
        for part in str(item).split(";"):
            part = part.strip()
            if part and part not in dedup:
                dedup.append(part)
    return {
        "route": route,
        "promotion_allowed": int(route.startswith("S5-")),
        "blocking_metric": ";".join(dedup),
        "S0_16_pass": int_flag(code.get("S0_16_pass")),
        "CodeRoute": code.get("CodeRoute", ""),
        "csv_claimed_exists_but_zip_missing_count": int_flag(code.get("csv_claimed_exists_but_zip_missing_count")),
        "required_import_errors": int_flag(code.get("required_import_errors")),
        "D-CHE_S1_pass": int_flag(basis.get("D-CHE_S1_pass")),
        "D-FOU_S1_pass": int_flag(basis.get("D-FOU_S1_pass")),
        "D-RAT_robust_pass": int_flag(basis.get("D-RAT_robust_pass")),
        "D-RBF_robust_pass": int_flag(basis.get("D-RBF_robust_pass")),
        "D-RAT_D-RBF_multibatch_closed": int_flag(basis.get("D-RAT_D-RBF_multibatch_closed")),
        "D-RAT_D-RBF_telemetry_bridge_robust_rows": sum(int_flag(r.get("robust_production_pass_rows")) for r in bridge_summary),
        "D-RAT_D-RBF_telemetry_bridge_complete_rows": sum(int_flag(r.get("component_telemetry_complete_rows")) for r in bridge_summary),
        "D-RAT_D-RBF_telemetry_bridge_blocker": ";".join(dict.fromkeys(part for r in bridge_summary for part in str(r.get("blocker", "")).split(";") if part)),
        "D-RAT_D-RBF_shape_localK_route": shape_scan.get("route", "") if shape_scan else "",
        "D-RAT_D-RBF_shape_localK_candidate_groups": int_flag(shape_scan.get("shape_localk_candidate_groups")) if shape_scan else 0,
        "D-RAT_D-RBF_shape_localK_scan_rows": int_flag(shape_scan.get("scan_rows")) if shape_scan else 0,
        "C0_retained_source_observer_pass_rows": int_flag(observer.get("C0_retained_source_observer_pass_rows")),
        "official_early_chain_positive_rows": int_flag(observer.get("official_early_chain_positive_rows")),
        "retained_h800_h3200_positive_rows": int_flag(observer.get("retained_h800_h3200_positive_rows")),
        "best_observer_heldout_precision_at_top20": observer.get("best_heldout_precision_at_top20", ""),
        "post_boundary_BVFR_route": bvfr.get("route", "") if bvfr else "",
        "post_boundary_BVFR_train_only_pass_rows": int_flag(bvfr.get("BVFR_train_only_certificate_pass_rows")) if bvfr else 0,
        "post_boundary_BVFR_fresh_C3_pass_rows": int_flag(bvfr.get("fresh_C3_boundary_replay_pass_rows")) if bvfr else 0,
        "post_boundary_BVFR_official_C3_pass_rows": int_flag(bvfr.get("official_C3_pass_rows")) if bvfr else 0,
        "post_state_holonomy_route": osh.get("route", "") if osh else "",
        "post_state_holonomy_train_only_pass_rows": int_flag(osh.get("OSH_train_only_certificate_pass_rows")) if osh else 0,
        "post_state_holonomy_fresh_C3_pass_rows": int_flag(osh.get("fresh_C3_state_holonomy_pass_rows")) if osh else 0,
        "post_state_holonomy_official_C3_pass_rows": int_flag(osh.get("official_C3_pass_rows")) if osh else 0,
        "post_row_orthogonal_route": rost.get("route", "") if rost else "",
        "post_row_orthogonal_train_only_pass_rows": int_flag(rost.get("ROST_train_only_certificate_pass_rows")) if rost else 0,
        "post_row_orthogonal_fresh_C3_pass_rows": int_flag(rost.get("fresh_C3_row_orthogonal_pass_rows")) if rost else 0,
        "post_row_orthogonal_official_C3_pass_rows": int_flag(rost.get("official_C3_pass_rows")) if rost else 0,
        "post_info_volume_route": ivsc.get("route", "") if ivsc else "",
        "post_info_volume_train_only_pass_rows": int_flag(ivsc.get("IVSC_train_only_certificate_pass_rows")) if ivsc else 0,
        "post_info_volume_fresh_C3_pass_rows": int_flag(ivsc.get("fresh_C3_info_volume_pass_rows")) if ivsc else 0,
        "post_info_volume_official_C3_pass_rows": int_flag(ivsc.get("official_C3_pass_rows")) if ivsc else 0,
        "post_fast_slow_route": fsmr.get("route", "") if fsmr else "",
        "post_fast_slow_train_only_pass_rows": int_flag(fsmr.get("FSMR_train_only_certificate_pass_rows")) if fsmr else 0,
        "post_fast_slow_fresh_C3_pass_rows": int_flag(fsmr.get("fresh_C3_fast_slow_memory_pass_rows")) if fsmr else 0,
        "post_fast_slow_official_C3_pass_rows": int_flag(fsmr.get("official_C3_pass_rows")) if fsmr else 0,
        "post_fast_slow_damped_repair_route": fsmr_repair.get("route", "") if fsmr_repair else "",
        "post_fast_slow_damped_repair_fresh_C3_pass_rows": int_flag(fsmr_repair.get("fresh_C3_damped_repair_pass_rows")) if fsmr_repair else 0,
        "post_fast_slow_damped_repair_official_C3_pass_rows": int_flag(fsmr_repair.get("official_C3_pass_rows")) if fsmr_repair else 0,
        "post_time_reversal_adjoint_route": trac.get("route", "") if trac else "",
        "post_time_reversal_adjoint_train_only_pass_rows": int_flag(trac.get("TRAC_train_only_certificate_pass_rows")) if trac else 0,
        "post_time_reversal_adjoint_fresh_C3_pass_rows": int_flag(trac.get("fresh_C3_time_reversal_adjoint_pass_rows")) if trac else 0,
        "post_time_reversal_adjoint_official_C3_pass_rows": int_flag(trac.get("official_C3_pass_rows")) if trac else 0,
        "post_time_reversal_adjoint_damped_repair_route": trac_repair.get("route", "") if trac_repair else "",
        "post_time_reversal_adjoint_damped_repair_fresh_C3_pass_rows": int_flag(trac_repair.get("fresh_C3_trac_damped_repair_pass_rows")) if trac_repair else 0,
        "post_time_reversal_adjoint_damped_repair_official_C3_pass_rows": int_flag(trac_repair.get("official_C3_pass_rows")) if trac_repair else 0,
        "C2_true_block_solver_pass_rows": int_flag(solver.get("C2_true_block_solver_pass_rows")),
        "C3_source_formation_pass_rows": int_flag(solver.get("C3_source_formation_pass_rows")),
        "C4_terminal_preservation_pass_rows": int_flag(terminal.get("C4_terminal_preservation_pass_rows")),
        "KAN_mapping_decision": kan.get("decision", ""),
        "queue_drained": int_flag(queue.get("queue_drained")),
        "execution_contract_violation": int(queue_violation),
        "minimum_effective_progress": ";".join(minimum_progress),
        "next_codex_action": action,
    }


def _failure_taxonomy(route: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for blocker in [x for x in str(route.get("blocking_metric", "")).split(";") if x]:
        if blocker == "D-RAT_D-RBF_multibatch_robust_or_telemetry":
            repair = "component telemetry completion; remove telemetry from train path; fused rational num/den reciprocal or RBF active local backward"
        elif blocker == "retained_source_certificate_gate":
            repair = "stop C-O13/C-O15 variants; define a new first-principles retained-source certificate"
        elif blocker == "post_boundary_BVFR_gate":
            repair = "BVFR boundary replay failed fresh/official gate; change principle, not thresholds"
        elif blocker == "post_state_holonomy_gate":
            repair = "OSH optimizer-state holonomy failed fresh/official gate; record boundary or change principle again"
        elif blocker == "post_row_orthogonal_ROST_gate":
            repair = "ROST row-orthogonal source transport failed fresh/official gate; record row-stability boundary or change principle"
        elif blocker == "post_info_volume_IVSC_gate":
            repair = "IVSC information-volume signal-channel failed fresh/official gate; record signal-channel boundary or change principle"
        elif blocker == "post_fast_slow_FSMR_gate":
            repair = "FSMR fast-slow source memory failed fresh/official gate; record source-memory boundary or change principle"
        elif blocker == "post_fast_slow_FSMR_damped_repair_gate":
            repair = "FSMR damped integration repair failed fresh/official gate; record integration-stability boundary or change principle"
        elif blocker == "post_time_reversal_TRAC_gate":
            repair = "TRAC time-reversal adjoint consistency failed fresh/official gate; record adjoint-transport boundary or change principle"
        elif blocker == "post_time_reversal_TRAC_damped_repair_gate":
            repair = "TRAC damped integration repair failed fresh/official gate; record integration-stability boundary or change principle"
        elif blocker == "C2_true_block_solver_or_observer_gate":
            repair = "repair only after C0 opens; reduce rank/damping/block CG"
        elif blocker == "C3_source_formation_gate":
            repair = "optimizer-state and slow source-state integration comparison"
        else:
            repair = route.get("next_codex_action", "")
        rows.append({"blocker": blocker, "route": route.get("route", ""), "repair_or_next_direction": repair})
    return rows


def _write_recap(out_dir: Path, route: dict[str, Any]) -> None:
    code_rows = read_rows(out_dir / "v22_09_code_truth_gate.csv")
    clean = read_rows(out_dir / "v22_09_clean_unzip_self_test.csv")
    packet_compare = read_rows(out_dir / "v22_09_packet_required_compare.csv")
    eff = read_rows(out_dir / "v22_09_efficiency_full_loop_reconfirm.csv")
    dr = read_rows(out_dir / "v22_09_drat_drbf_multibatch_summary.csv")
    bridge = read_rows(out_dir / "v22_09_drat_drbf_telemetry_bridge_summary.csv")
    tile64 = read_rows(out_dir / "v22_09_drat_drbf_tile64_repair_summary.csv")
    exp2_full = read_rows(out_dir / "v22_09_drat_drbf_rbf_exp2_fullpath_repair_summary.csv")
    exp2_forward = read_rows(out_dir / "v22_09_drat_drbf_rbf_exp2_forwardonly_repair_summary.csv")
    shape_scan = read_rows(out_dir / "v22_09_drat_drbf_shape_localk_repair_scan_summary.csv")
    repair = read_rows(out_dir / "v22_09_drat_drbf_repair_attempts_summary.csv")
    smoke = read_rows(out_dir / "v22_09_drat_drbf_limited_functional_smoke.csv")
    obs = read_rows(out_dir / "v22_09_retained_source_observer_summary.csv")
    top = read_rows(out_dir / "v22_09_retained_source_observer_top_candidates.csv")
    bvfr_summary = read_rows(out_dir / "v22_09_boundary_flow_replay_summary.csv")
    bvfr_selected = read_rows(out_dir / "v22_09_boundary_flow_replay_selected_candidates.csv")
    bvfr_fresh = read_rows(out_dir / "v22_09_boundary_flow_replay_fresh_c3_summary.csv")
    osh_summary = read_rows(out_dir / "v22_09_optimizer_state_holonomy_summary.csv")
    osh_selected = read_rows(out_dir / "v22_09_optimizer_state_holonomy_selected_candidates.csv")
    osh_fresh = read_rows(out_dir / "v22_09_optimizer_state_holonomy_fresh_c3_summary.csv")
    rost_summary = read_rows(out_dir / "v22_09_row_orthogonal_source_transport_summary.csv")
    rost_selected = read_rows(out_dir / "v22_09_row_orthogonal_source_transport_selected_candidates.csv")
    rost_fresh = read_rows(out_dir / "v22_09_row_orthogonal_source_transport_fresh_c3_summary.csv")
    ivsc_summary = read_rows(out_dir / "v22_09_info_volume_signal_channel_summary.csv")
    ivsc_selected = read_rows(out_dir / "v22_09_info_volume_signal_channel_selected_candidates.csv")
    ivsc_fresh = read_rows(out_dir / "v22_09_info_volume_signal_channel_fresh_c3_summary.csv")
    fsmr_summary = read_rows(out_dir / "v22_09_fast_slow_memory_resonance_summary.csv")
    fsmr_selected = read_rows(out_dir / "v22_09_fast_slow_memory_resonance_selected_candidates.csv")
    fsmr_fresh = read_rows(out_dir / "v22_09_fast_slow_memory_resonance_fresh_c3_summary.csv")
    fsmr_repair_fresh = read_rows(out_dir / "v22_09_fast_slow_memory_damped_repair_fresh_c3_summary.csv")
    trac_summary = read_rows(out_dir / "v22_09_time_reversal_adjoint_summary.csv")
    trac_selected = read_rows(out_dir / "v22_09_time_reversal_adjoint_selected_candidates.csv")
    trac_fresh = read_rows(out_dir / "v22_09_time_reversal_adjoint_fresh_c3_summary.csv")
    trac_repair_fresh = read_rows(out_dir / "v22_09_time_reversal_adjoint_damped_repair_fresh_c3_summary.csv")
    solver_summary = read_rows(out_dir / "v22_09_metric_solver_summary.csv")
    solver_matrix = read_rows(out_dir / "v22_09_metric_solver_matrix.csv")
    source_formation = read_rows(out_dir / "v22_09_source_formation_matrix.csv")
    terminal = read_rows(out_dir / "v22_09_terminal_preservation_matrix.csv")
    kan = read_rows(out_dir / "v22_09_kan_mapping_matrix.csv")
    queue = read_rows(out_dir / "v22_09_gpu_assignment_manifest.csv")
    idle = read_rows(out_dir / "v22_09_idle_violation.csv")
    failure = read_rows(out_dir / "v22_09_failure_taxonomy.csv")
    best_obs = _best(obs, "heldout_precision_at_top20")
    text = [
        "# DG-KAN v22.09 RetainedSourceObservability FunctionalUpdate BasisEfficiency 4GPU 实验结果复盘",
        "",
        f"生成时间：{now_sg()}",
        "",
        "## Route",
        "",
        f"- route: `{route['route']}`",
        f"- promotion_allowed: {route['promotion_allowed']}",
        f"- blocking_metric: `{route['blocking_metric']}`",
        f"- S0.16 pass / CodeRoute: {route['S0_16_pass']} / `{route['CodeRoute']}`",
        f"- D-CHE/D-FOU S1 pass: {route['D-CHE_S1_pass']} / {route['D-FOU_S1_pass']}",
        f"- D-RAT/D-RBF robust pass: {route['D-RAT_robust_pass']} / {route['D-RBF_robust_pass']}",
        f"- D-RAT/D-RBF telemetry bridge robust rows / telemetry rows / blocker: {route['D-RAT_D-RBF_telemetry_bridge_robust_rows']} / {route['D-RAT_D-RBF_telemetry_bridge_complete_rows']} / `{route['D-RAT_D-RBF_telemetry_bridge_blocker']}`",
        f"- D-RAT/D-RBF shape/local-K scan route / candidate groups / rows: `{route['D-RAT_D-RBF_shape_localK_route']}` / {route['D-RAT_D-RBF_shape_localK_candidate_groups']} / {route['D-RAT_D-RBF_shape_localK_scan_rows']}",
        f"- C0 retained-source certificate pass rows: {route['C0_retained_source_observer_pass_rows']}",
        f"- official early-chain positive rows: {route['official_early_chain_positive_rows']}",
        f"- retained h800+h3200 positive rows: {route['retained_h800_h3200_positive_rows']}",
        f"- post-boundary BVFR route: `{route['post_boundary_BVFR_route']}`",
        f"- post-boundary BVFR train-only/fresh/official pass rows: {route['post_boundary_BVFR_train_only_pass_rows']} / {route['post_boundary_BVFR_fresh_C3_pass_rows']} / {route['post_boundary_BVFR_official_C3_pass_rows']}",
        f"- post-state OSH route: `{route['post_state_holonomy_route']}`",
        f"- post-state OSH train-only/fresh/official pass rows: {route['post_state_holonomy_train_only_pass_rows']} / {route['post_state_holonomy_fresh_C3_pass_rows']} / {route['post_state_holonomy_official_C3_pass_rows']}",
        f"- post-row ROST route: `{route['post_row_orthogonal_route']}`",
        f"- post-row ROST train-only/fresh/official pass rows: {route['post_row_orthogonal_train_only_pass_rows']} / {route['post_row_orthogonal_fresh_C3_pass_rows']} / {route['post_row_orthogonal_official_C3_pass_rows']}",
        f"- post-info IVSC route: `{route['post_info_volume_route']}`",
        f"- post-info IVSC train-only/fresh/official pass rows: {route['post_info_volume_train_only_pass_rows']} / {route['post_info_volume_fresh_C3_pass_rows']} / {route['post_info_volume_official_C3_pass_rows']}",
        f"- post-fast-slow FSMR route: `{route['post_fast_slow_route']}`",
        f"- post-fast-slow FSMR train-only/fresh/official pass rows: {route['post_fast_slow_train_only_pass_rows']} / {route['post_fast_slow_fresh_C3_pass_rows']} / {route['post_fast_slow_official_C3_pass_rows']}",
        f"- post-fast-slow FSMR damped repair route: `{route['post_fast_slow_damped_repair_route']}`",
        f"- post-fast-slow FSMR damped repair fresh/official pass rows: {route['post_fast_slow_damped_repair_fresh_C3_pass_rows']} / {route['post_fast_slow_damped_repair_official_C3_pass_rows']}",
        f"- post-time-reversal TRAC route: `{route['post_time_reversal_adjoint_route']}`",
        f"- post-time-reversal TRAC train-only/fresh/official pass rows: {route['post_time_reversal_adjoint_train_only_pass_rows']} / {route['post_time_reversal_adjoint_fresh_C3_pass_rows']} / {route['post_time_reversal_adjoint_official_C3_pass_rows']}",
        f"- post-time-reversal TRAC damped repair route: `{route['post_time_reversal_adjoint_damped_repair_route']}`",
        f"- post-time-reversal TRAC damped repair fresh/official pass rows: {route['post_time_reversal_adjoint_damped_repair_fresh_C3_pass_rows']} / {route['post_time_reversal_adjoint_damped_repair_official_C3_pass_rows']}",
        f"- C2/C3/C4 pass rows: {route['C2_true_block_solver_pass_rows']} / {route['C3_source_formation_pass_rows']} / {route['C4_terminal_preservation_pass_rows']}",
        f"- KAN mapping decision: `{route['KAN_mapping_decision']}`",
        f"- execution_contract_violation: {route['execution_contract_violation']}",
        f"- minimum_effective_progress: `{route['minimum_effective_progress']}`",
        f"- next_codex_action: {route['next_codex_action']}",
        "",
        "## Part A Code / Packet Truth Gate",
        "",
        md_table(code_rows, ["check", "pass", "metric", "value", "blocker"], 40),
        "",
        "### Clean Unzip Self Test",
        "",
        md_table(clean, ["packet", "unzip_root", "compileall_returncode", "import_returncode", "self_contained_import_check", "pass"], 5),
        "",
        "### Required CSV vs Zip",
        "",
        md_table(packet_compare, ["path", "csv_exists", "zip_contains", "csv_claimed_exists_but_zip_missing"], 20),
        "",
        "## Part B Basis Efficiency",
        "",
        md_table(eff, ["carrier", "variant_family", "forward_ratio_vs_mlp", "backward_ratio_vs_mlp", "step_ratio_vs_mlp", "memory_ratio_vs_mlp", "functional_runner_same_kernel", "fallback_kernel_used", "official_fused_kernel_complete", "v22_09_S1_pass", "v22_09_blocker"], 20),
        "",
        "### D-RAT / D-RBF Multibatch",
        "",
        md_table(dr, ["carrier", "profile_rows", "robust_production_pass_rows", "near_E1_rows", "component_telemetry_complete_rows", "best_forward_ratio", "best_step_ratio", "best_memory_ratio", "decision", "blocker", "repair_attempt_rows", "repair_micro_near_E1_rows", "repair_best_variant", "repair_attempt_status", "repair_blocker", "official_closure_claimed"], 20),
        "",
        "### D-RAT / D-RBF Telemetry Bridge",
        "",
        "- bridge purpose: combine official fused runner timing with component-complete repair telemetry by carrier/batch to test whether telemetry incompleteness is still the decisive blocker. This is audit-only and does not claim official closure.",
        md_table(bridge, ["carrier", "profile_rows", "robust_production_pass_rows", "near_E1_rows", "component_telemetry_complete_rows", "best_forward_ratio", "best_backward_ratio", "best_step_ratio", "best_memory_ratio", "decision", "blocker"], 20),
        "",
        "### D-RAT / D-RBF Tile64 Repair Attempt",
        "",
        "- repair result: changing official Triton forward batch tile to 64 for B>=128 was tried, measured, and then reverted because it worsened forward ratio. The failed run is retained as artifact evidence.",
        md_table(tile64, ["carrier", "profile_rows", "robust_production_pass_rows", "near_E1_rows", "component_telemetry_complete_rows", "best_forward_ratio", "best_backward_ratio", "best_step_ratio", "best_memory_ratio", "decision", "blocker"], 20),
        "",
        "### D-RBF Exp2 Repair Attempts",
        "",
        "- repair result: replacing Gaussian `exp(-0.5*r*r)` with an equivalent `exp2` path was tried as both full forward/backward and forward-only variants. Neither opened robust production; the full path improved one forward readout but introduced step/backward blocker, and the forward-only path remained worse than the stable direct-`exp` kernel. Both failed runs are retained as audit artifacts and the final kernel was restored.",
        "#### Full Path Exp2",
        "",
        md_table(exp2_full, ["carrier", "profile_rows", "robust_production_pass_rows", "near_E1_rows", "component_telemetry_complete_rows", "best_forward_ratio", "best_backward_ratio", "best_step_ratio", "best_memory_ratio", "decision", "blocker"], 20),
        "",
        "#### Forward-Only Exp2",
        "",
        md_table(exp2_forward, ["carrier", "profile_rows", "robust_production_pass_rows", "near_E1_rows", "component_telemetry_complete_rows", "best_forward_ratio", "best_backward_ratio", "best_step_ratio", "best_memory_ratio", "decision", "blocker"], 20),
        "",
        "### D-RAT / D-RBF Shape Local-K Repair Scan",
        "",
        "- repair result: h64/h96/h128 and D-RBF K2/K4 were scanned under the same v22.09 ratio gates. These rows are diagnostic only; they do not replace the official h128/K4 main gate.",
        md_table(shape_scan, ["carrier", "repair_variant", "scan_hidden", "profile_rows", "robust_production_pass_rows", "near_E1_rows", "best_forward_ratio", "best_step_ratio", "best_memory_ratio", "shape_localk_candidate", "official_closure_claimed", "blocker"], 30),
        "",
        "### Repair / Limited Smoke",
        "",
        md_table(repair, ["carrier", "repair_attempt_rows", "repair_micro_near_E1_rows", "repair_component_telemetry_complete_rows", "repair_best_forward_ratio", "repair_best_step_ratio", "repair_best_memory_ratio", "repair_best_variant", "repair_attempt_status", "repair_blocker", "official_closure_claimed"], 20),
        md_table(smoke, ["carrier", "scope", "promotion_allowed", "limited_smoke_executed", "status", "reason"], 20),
        "",
        "## Part C Retained-Source Observability",
        "",
        f"- best heldout observer: `{best_obs.get('certificate_family', '')}` heldout_precision={best_obs.get('heldout_precision_at_top20', '')} precision@20={best_obs.get('precision_at_top20', '')}",
        md_table(obs, ["certificate_family", "rows", "positive_official_early_chain_rows", "positive_h3200_rows", "positive_retained_h800_h3200_rows", "AUC_predict_official_early_chain", "AUC_predict_h3200_positive", "AUC_predict_retained_h800_h3200_positive", "precision_at_top20", "heldout_precision_at_top20", "control_equivalent_fraction", "C0_retained_source_certificate_pass", "blocker"], 10),
        "",
        "### Top C-O13 Candidates",
        "",
        md_table(top, ["job_order", "v21_id", "dataset", "seed", "C-O13_score", "commutator_norm_ratio", "flow_order_gap", "source_direction_reproducibility", "future_audit_source_h100", "future_audit_source_h400", "future_audit_source_h800", "future_audit_source_h3200", "official_early_chain_h100_h400_h800_positive"], 12),
        "",
        "### Post-Boundary BVFR Certificate",
        "",
        "- trigger: C-O13/C-O14/C-O15 observer gate stayed closed; this follow-up tests a different first-principles boundary-value flow replay certificate. The source step is treated as a train-only boundary condition, replayed through short local optimizer flow, and compared with random/sign/corrupt boundary controls. Future source columns remain audit labels only.",
        md_table(bvfr_summary, ["certificate_family", "rows", "positive_official_early_chain_rows", "positive_h3200_rows", "positive_retained_h800_h3200_rows", "AUC_predict_official_early_chain", "AUC_predict_h3200_positive", "AUC_predict_retained_h800_h3200_positive", "precision_at_top20", "BVFR_train_only_certificate_pass_rows", "BVFR_official_observer_pass", "blocker"], 5),
        "",
        "#### BVFR Selected Candidates",
        "",
        md_table(bvfr_selected, ["job_order", "v21_id", "mechanism", "dataset", "seed", "BVFR_score", "source_boundary_replay_B_loss_gain", "B_boundary_replay_control_gap", "boundary_replay_norm_ratio", "boundary_replay_source_cosine", "future_audit_source_h800", "future_audit_source_h3200", "BVFR_train_only_certificate_pass"], 5),
        "",
        "#### BVFR Fresh C3",
        "",
        md_table(bvfr_fresh, ["v21_id", "dataset", "seed", "BVFR_score", "source_vs_best_control_h100", "source_vs_best_control_h400", "source_vs_best_control_h800", "source_vs_best_control_h1600", "source_vs_best_control_h2400", "source_vs_best_control_h3200", "boundary_replay_source_cosine_mean", "boundary_replay_norm_ratio_mean", "debt_not_exploded", "BVFR_fresh_C3_boundary_replay_pass", "official_C3_pass", "blocker"], 5),
        "",
        "### Post-State OSH Certificate",
        "",
        "- trigger: BVFR failed fresh/official gate; this follow-up implements the plan's train-flow algebra recommendation with optimizer-state holonomy. A source boundary is embedded into short AdamW state cycles over split orders, and fresh runs inject a live source/momentum mixed direction against AdamW/SGD/NoOp/random matched controls.",
        md_table(osh_summary, ["certificate_family", "rows", "positive_official_early_chain_rows", "positive_h3200_rows", "positive_retained_h800_h3200_rows", "AUC_predict_official_early_chain", "AUC_predict_h3200_positive", "AUC_predict_retained_h800_h3200_positive", "precision_at_top20", "OSH_train_only_certificate_pass_rows", "OSH_official_observer_pass", "blocker"], 5),
        "",
        "#### OSH Selected Candidates",
        "",
        md_table(osh_selected, ["job_order", "v21_id", "mechanism", "dataset", "seed", "OSH_score", "source_stateful_B_loss_gain", "B_stateful_control_gap", "stateful_holonomy_ratio", "stateful_holonomy_control_gap", "source_momentum_cosine", "future_audit_source_h800", "future_audit_source_h3200", "OSH_train_only_certificate_pass"], 5),
        "",
        "#### OSH Fresh C3",
        "",
        md_table(osh_fresh, ["v21_id", "dataset", "seed", "OSH_score", "source_vs_best_control_h100", "source_vs_best_control_h400", "source_vs_best_control_h800", "source_vs_best_control_h1600", "source_vs_best_control_h2400", "source_vs_best_control_h3200", "source_momentum_cosine_mean", "momentum_norm_ratio_mean", "debt_not_exploded", "OSH_fresh_C3_state_holonomy_pass", "official_C3_pass", "blocker"], 5),
        "",
        "### Post-Row ROST Certificate",
        "",
        "- trigger: OSH failed fresh/official gate; this follow-up implements the plan's row-wise angular stability direction. Candidate source updates are projected into each weight row's tangent space to suppress row-norm jitter, then compared with random/sign/corrupt row-orthogonal controls and fresh AdamW/SGD/NoOp/random matched controls.",
        md_table(rost_summary, ["certificate_family", "rows", "positive_official_early_chain_rows", "positive_h3200_rows", "positive_retained_h800_h3200_rows", "AUC_predict_official_early_chain", "AUC_predict_h3200_positive", "AUC_predict_retained_h800_h3200_positive", "precision_at_top20", "ROST_train_only_certificate_pass_rows", "ROST_official_observer_pass", "blocker"], 5),
        "",
        "#### ROST Selected Candidates",
        "",
        md_table(rost_selected, ["job_order", "v21_id", "mechanism", "dataset", "seed", "ROST_score", "source_row_orthogonal_B_loss_gain", "B_row_orthogonal_control_gap", "row_norm_drift", "row_tangential_energy_fraction", "row_tangent_source_cosine", "future_audit_source_h800", "future_audit_source_h3200", "ROST_train_only_certificate_pass"], 5),
        "",
        "#### ROST Fresh C3",
        "",
        md_table(rost_fresh, ["v21_id", "dataset", "seed", "ROST_score", "source_vs_best_control_h100", "source_vs_best_control_h400", "source_vs_best_control_h800", "source_vs_best_control_h1600", "source_vs_best_control_h2400", "source_vs_best_control_h3200", "row_tangential_energy_fraction_mean", "row_tangent_source_cosine_mean", "debt_not_exploded", "ROST_fresh_C3_row_orthogonal_pass", "official_C3_pass", "blocker"], 5),
        "",
        "### Post-Info IVSC Certificate",
        "",
        "- trigger: ROST failed fresh/official gate; this follow-up implements the plan's signal-channel / information-volume direction. It measures per-example train-logit displacement drift, diffusion, effective rank, and information volume, then runs fresh source updates against AdamW/SGD/NoOp/random matched controls.",
        md_table(ivsc_summary, ["certificate_family", "rows", "positive_official_early_chain_rows", "positive_h3200_rows", "positive_retained_h800_h3200_rows", "AUC_predict_official_early_chain", "AUC_predict_h3200_positive", "AUC_predict_retained_h800_h3200_positive", "precision_at_top20", "IVSC_train_only_certificate_pass_rows", "IVSC_official_observer_pass", "blocker"], 5),
        "",
        "#### IVSC Selected Candidates",
        "",
        md_table(ivsc_selected, ["job_order", "v21_id", "mechanism", "dataset", "seed", "IVSC_score", "source_IVSC_B_loss_gain", "B_IVSC_control_gap", "DDR", "DDR_control_gap", "effective_rank", "info_volume", "future_audit_source_h800", "future_audit_source_h3200", "IVSC_train_only_certificate_pass"], 5),
        "",
        "#### IVSC Fresh C3",
        "",
        md_table(ivsc_fresh, ["v21_id", "dataset", "seed", "IVSC_score", "source_vs_best_control_h100", "source_vs_best_control_h400", "source_vs_best_control_h800", "source_vs_best_control_h1600", "source_vs_best_control_h2400", "source_vs_best_control_h3200", "DDR_mean", "signal_energy_fraction_mean", "effective_rank_mean", "debt_not_exploded", "IVSC_fresh_C3_info_volume_pass", "official_C3_pass", "blocker"], 5),
        "",
        "### Post-Fast-Slow FSMR Certificate",
        "",
        "- trigger: IVSC failed fresh/official gate; this follow-up implements the plan's fast/slow source-memory and source-state age direction. It builds short train-only source-memory traces, checks slow-memory alignment/overwrite against random/sign/corrupt controls, then runs fresh AdamW+slow-source-memory updates against AdamW/SGD/NoOp/random matched controls.",
        md_table(fsmr_summary, ["certificate_family", "rows", "positive_official_early_chain_rows", "positive_h3200_rows", "positive_retained_h800_h3200_rows", "AUC_predict_official_early_chain", "AUC_predict_h3200_positive", "AUC_predict_retained_h800_h3200_positive", "precision_at_top20", "FSMR_train_only_certificate_pass_rows", "FSMR_official_observer_pass", "blocker"], 5),
        "",
        "#### FSMR Selected Candidates",
        "",
        md_table(fsmr_selected, ["job_order", "v21_id", "mechanism", "dataset", "seed", "FSMR_score", "source_FSMR_B_loss_gain", "B_FSMR_control_gap", "slow_fast_alignment_mean", "source_state_age_fraction", "source_state_overwrite_fraction", "slow_memory_source_cosine", "future_audit_source_h800", "future_audit_source_h3200", "FSMR_train_only_certificate_pass"], 5),
        "",
        "#### FSMR Fresh C3",
        "",
        md_table(fsmr_fresh, ["v21_id", "dataset", "seed", "FSMR_score", "source_vs_best_control_h100", "source_vs_best_control_h400", "source_vs_best_control_h800", "source_vs_best_control_h1600", "source_vs_best_control_h2400", "source_vs_best_control_h3200", "slow_fast_alignment_mean", "source_state_overwrite_fraction_mean", "memory_loss_gain_mean", "debt_not_exploded", "FSMR_fresh_C3_fast_slow_memory_pass", "official_C3_pass", "blocker"], 5),
        "",
        "#### FSMR Damped Repair Fresh C3",
        "",
        "- repair trigger: FSMR fresh rows showed early local positives followed by h800-h3200 washout and one debt failure; this repair lowers FU injection, raises slow-memory beta, and refreshes less often. It is an integration-stability test only, not a new official observer.",
        md_table(fsmr_repair_fresh, ["v21_id", "dataset", "seed", "FSMR_score", "source_vs_best_control_h100", "source_vs_best_control_h400", "source_vs_best_control_h800", "source_vs_best_control_h1600", "source_vs_best_control_h2400", "source_vs_best_control_h3200", "slow_fast_alignment_mean", "source_state_overwrite_fraction_mean", "memory_loss_gain_mean", "debt_not_exploded", "FSMR_fresh_C3_fast_slow_memory_pass", "official_C3_pass", "blocker"], 5),
        "",
        "### Post-Time-Reversal TRAC Certificate",
        "",
        "- trigger: FSMR damped repair failed fresh/official gate; this follow-up changes principle again. TRAC treats a candidate update as a local train-flow coordinate: after a source step and short AdamW transport, an adjoint inverse update should partially close a train-only round trip and remain separated from random/sign/corrupt controls. Future source columns remain audit labels only.",
        md_table(trac_summary, ["certificate_family", "rows", "positive_official_early_chain_rows", "positive_h3200_rows", "positive_retained_h800_h3200_rows", "AUC_predict_official_early_chain", "AUC_predict_h3200_positive", "AUC_predict_retained_h800_h3200_positive", "precision_at_top20", "TRAC_train_only_certificate_pass_rows", "TRAC_official_observer_pass", "blocker"], 5),
        "",
        "#### TRAC Selected Candidates",
        "",
        md_table(trac_selected, ["job_order", "v21_id", "mechanism", "dataset", "seed", "TRAC_score", "source_TRAC_B_loss_gain", "B_TRAC_control_gap", "source_transport_cosine", "source_adjoint_closure_ratio", "adjoint_closure_control_gap", "source_roundtrip_loss_gap", "future_audit_source_h800", "future_audit_source_h3200", "TRAC_train_only_certificate_pass"], 5),
        "",
        "#### TRAC Fresh C3",
        "",
        md_table(trac_fresh, ["v21_id", "dataset", "seed", "TRAC_score", "source_vs_best_control_h100", "source_vs_best_control_h400", "source_vs_best_control_h800", "source_vs_best_control_h1600", "source_vs_best_control_h2400", "source_vs_best_control_h3200", "adjoint_transport_cosine_mean", "adjoint_transport_residual_ratio_mean", "adjoint_loss_gain_mean", "debt_not_exploded", "TRAC_fresh_C3_time_reversal_adjoint_pass", "official_C3_pass", "blocker"], 5),
        "",
        "#### TRAC Damped Repair Fresh C3",
        "",
        "- repair trigger: TRAC fresh rows lost to controls from h100 and both selected rows failed debt; this repair lowers FU injection, reduces transported-source mixing, and refreshes less often. It is an integration-stability test only, not a new official observer.",
        md_table(trac_repair_fresh, ["v21_id", "dataset", "seed", "TRAC_score", "source_vs_best_control_h100", "source_vs_best_control_h400", "source_vs_best_control_h800", "source_vs_best_control_h1600", "source_vs_best_control_h2400", "source_vs_best_control_h3200", "adjoint_transport_cosine_mean", "adjoint_transport_residual_ratio_mean", "adjoint_loss_gain_mean", "debt_not_exploded", "TRAC_fresh_C3_time_reversal_adjoint_pass", "official_C3_pass", "blocker"], 5),
        "",
        "## Part D Metric Solver / Source Formation / Terminal / KAN",
        "",
        md_table(solver_summary, ["block_role", "rows", "C2_solver_gate_pass_rows", "projection_residual_Gf_min", "ActuationR2_max", "B2_transfer_gain_max", "blocker"], 20),
        md_table(solver_matrix, ["case", "target_family", "block_role", "projection_residual_Gf", "ActuationR2", "B2_transfer_gain", "solve_time_ms", "NDS", "C2_solver_gate_pass", "blocker"], 10),
        md_table(source_formation, ["source_formation_status", "reason", "C3_source_formation_pass", "official_C3_pass"], 10),
        md_table(terminal, ["terminal_preservation_status", "source_h3200", "source_h4800", "R4800_over_3200", "C4_terminal_preservation_pass", "blocker"], 10),
        md_table(kan, ["mapping_status", "blocker", "KAN_source_channel_decision", "KAN_specific_delta_vs_MLP_same_metric"], 10),
        "",
        "## 4GPU Queue",
        "",
        md_table(queue, ["task_id", "line", "gpu", "command", "status", "returncode", "start_time", "end_time", "duration_sec", "log_path"], 20),
        md_table(idle, ["execution_contract_violation", "reason", "max_idle_gap_sec"], 10),
        "",
        "## Failure Taxonomy",
        "",
        md_table(failure, ["blocker", "route", "repair_or_next_direction"], 20),
        "",
        "## 修改记录",
        "",
        "- 新增 `dgkan/fu/retained_source_certificate.py`、`train_flow_commutator.py`、`source_state_dynamics.py`、`optimizer_state_integration.py`：把 v22.09 C-O13/C-O14/C-O15 与 optimizer-state evidence 写成纯函数和 unit tests。",
        "- 新增 `dgkan/profiling/efficiency_v22_09.py`：记录 D-RAT/D-RBF component telemetry keys 与 unit tests。",
        "- 新增 `experiments/run_v22_09_common.py`：独立 v22.09 结果目录、日志、required source list、SHA manifest、code packet、bundle helper。",
        "- 新增 `experiments/run_v22_09_s016_truth_gate.py`：基于 final code packet 解压目录执行 compile/import self-contained gate，并核对 required CSV 与 zip entries。",
        "- 新增并更新 `experiments/run_v22_09_basis_efficiency_closure.py`：D-CHE/D-FOU measured artifact readback，D-RAT/D-RBF multibatch officialization、repair waterfall、limited smoke 与 telemetry bridge；bridge 只作 audit，不声明 official closure。",
        "- 尝试并撤回 D-RAT/D-RBF official Triton forward tile64 repair：把 `dgkan/kernels/fused_rational_k4.py` 与 `dgkan/kernels/fused_rbf.py` 的 B>=128 forward tile 改成 64 后重测，forward ratio 变差，因此恢复原策略；失败结果保留为 `v22_09_drat_drbf_tile64_repair_*` artifact。",
        "- 尝试并撤回 D-RBF `exp2` Gaussian repair：分别测试 full forward/backward 与 forward-only 两种等价指数路径；均未打开 robust production，最终恢复稳定 direct-`exp` kernel，失败结果保留为 `v22_09_drat_drbf_rbf_exp2_*` artifact。",
        "- 新增 `experiments/run_v22_09_drat_drbf_shape_localk_repair_scan.py`：扫描 D-RAT/D-RBF hidden=64/96/128 与 D-RBF K2/K4 local-K 形态，作为 forward blocker 的诊断 evidence；不替代 official h128/K4 closure gate。",
        "- 新增 `experiments/run_v22_09_retained_source_observer.py`：执行 C-O13 train-flow algebra、C-O14 drift-diffusion signal-channel、C-O15 block-coordinate source-state 三类 train-only certificate scoring；future source 只作 audit label。",
        "- 新增 `experiments/run_v22_09_post_boundary_flow_replay_certificate.py`：在 C-O13/C-O14/C-O15 blocked 后尝试 BVFR boundary-value flow replay certificate；用 train-only boundary condition + local optimizer replay residual 构造方向，并跑 fresh h100-h3200 vs AdamW/SGD/NoOp/random matched controls。",
        "- 新增 `experiments/run_v22_09_post_optimizer_state_holonomy_certificate.py`：在 BVFR blocked 后尝试 OSH optimizer-state holonomy certificate；用多 split 短 AdamW state cycle 的 holonomy / momentum alignment 构造 train-only 证据，并跑 source+momentum fresh h100-h3200 对照。",
        "- 新增 `experiments/run_v22_09_post_row_orthogonal_source_transport_certificate.py`：在 OSH blocked 后尝试 ROST row-orthogonal source transport certificate；把 source/control update 投影到 row-tangent 子空间，记录 row_norm_drift / tangential energy / train transfer，并跑 row-orthogonal fresh h100-h3200 对照。",
        "- 新增 `experiments/run_v22_09_post_info_volume_signal_channel_certificate.py`：在 ROST blocked 后尝试 IVSC information-volume signal-channel certificate；用 train-only per-example logit displacement 记录 drift/diffusion/DDR/effective_rank/info_volume，并跑 info-volume fresh h100-h3200 对照。",
        "- 新增 `experiments/run_v22_09_post_fast_slow_memory_resonance_certificate.py`：在 IVSC blocked 后尝试 FSMR fast/slow source-memory resonance certificate；用 train-only 短轨迹 slow source memory 的 alignment/age/overwrite 构造证据，并跑 AdamW+slow-memory fresh h100-h3200 对照。",
        "- 新增 `experiments/run_v22_09_post_fast_slow_memory_damped_repair.py`：针对 FSMR fresh 的 h800-h3200 washout/debt blocker，重跑低 FU 强度、高 beta、低刷新频率的 damped integration repair；仍只作为 fresh repair evidence，不声明 official。",
        "- 新增 `experiments/run_v22_09_post_time_reversal_adjoint_certificate.py`：在 FSMR damped repair blocked 后尝试 TRAC time-reversal adjoint consistency certificate；用 train-only source-step + AdamW transport + adjoint inverse round-trip closure 构造证据，并跑 time-reversal-adjoint fresh h100-h3200 对照。",
        "- 新增 `experiments/run_v22_09_post_time_reversal_adjoint_damped_repair.py`：针对 TRAC fresh 的 h100 起点失败与 debt blocker，重跑低 FU 强度、低 transport mix、低刷新频率的 damped integration repair；仍只作为 fresh repair evidence，不声明 official。",
        "- 修复 BVFR/OSH fresh selection：fresh C3 优先选择 train-only certificate pass rows；只有没有 pass row 时才退回按 score 选取，避免把未过证书硬门的高分候选误作 fresh source candidate。",
        "- 新增 `experiments/run_v22_09_metric_dynamics_solver.py`、`run_v22_09_terminal_preservation.py`、`run_v22_09_kan_mapping.py`：按 C0/C2/C3/C4 gate fail-closed，不强行进入 terminal/KAN。",
        "- 新增 `experiments/run_v22_09_full.py` 与 `run_v22_09_finalize.py`：4GPU queue、final route、复盘、failure taxonomy、artifact index、code review packet 与 results bundle。",
        "",
        "## 分析 / Insight / 证据链",
        "",
        "- v22.09 修复了 v22.08 code packet 的审计漏洞：truth gate 不只检查 repo 文件存在，还构建 `v22_09_code_review_packet.zip`，解压后执行 `compileall` 与 required import closure，并记录 `csv_claimed_exists_but_zip_missing_count`。",
        "- D-CHE/D-FOU 仍按 measured full-loop readback 判定，不补造缺失 timing 字段；它们是否 S1 pass 由 forward/step/memory/same-kernel/fallback/official fused gate 决定。",
        "- D-RAT/D-RBF 使用 v22.09 更严格的 robust gate：>=3/4 batch sizes 需要 forward<=1.25、step<=1.25、memory<=1.05、gradcheck、official fused、same-kernel、component telemetry 同时成立。repair rows 不会被提升成 official closure。",
        "- D-RAT/D-RBF telemetry bridge 将 official fused runner timing 与 component-complete repair telemetry 对齐后，component_telemetry_complete_rows 变为 4/4，但 robust rows 仍为 0，blocker 收敛到 `forward_ratio`。因此不能把 telemetry 缺口修成 efficiency closure。",
        "- Tile64 kernel repair 是真实负结果：B>=128 使用 64 batch tile 后 D-RAT/D-RBF forward ratio 变差，说明当前 forward blocker 不是简单 grid launch/tile overhead；该改动已撤回，失败数据保留供审计。",
        "- D-RBF exp2 repair 是真实负结果：base-2 equivalent exponential 没有稳定改善 official robust gate；full path 引入 step/backward blocker，forward-only path 仍未优于稳定 direct-`exp` 路径。因此不能把 exp approximation 写成 D-RBF closure。",
        "- Shape/local-K scan 只检验当前 blocker 是否来自 hidden/K 形态；即便某个低 hidden 或 K2 组出现 candidate，也只说明下一轮可设计新的 official carrier contract，不能 retroactively 关闭本轮 h128/K4 D-RAT/D-RBF gate。",
        "- C-O13/C-O14/C-O15 均只用 v22.07 train-only C0/C1/C2 artifact 构造证书分数，future horizon 只作为 audit label。若 official early-chain 正例为 0，AUC/precision official gate 必须 fail-closed。",
        "- BVFR 是 C-O13/C-O14/C-O15 之后的 post-boundary 尝试，不是同族 threshold/cap/scale 调参：它把候选 source update 作为边界条件，在短局部 AdamW replay 后取 residual direction，再用 disjoint train folds 和 random/sign/corrupt boundary controls 检查是否仍有可用训练流结构。",
        "- BVFR fresh C3 仍只允许作为 fresh evidence；如果 h100/h400/h800/h1600/h3200/debt gate 或 official observer/C2/C3 未同时打开，final route 必须保持 promotion blocked。",
        "- OSH 进一步把计划中的 optimizer-state/vector-field consistency 显式化：source boundary 必须在 AdamW stateful split cycle 中呈现低 holonomy、正 transfer、momentum alignment，并在 fresh run 中通过 live AdamW moment mixed update 战胜 AdamW/SGD/NoOp/random controls。",
        "- OSH 初次跑出 9 行 train-only pass，但 selected fresh top-2 未过 train-only gate；因此先修正 selection 语义再重跑，不能把选择逻辑瑕疵直接写成科学结论。",
        "- OSH 若只有 train-only rows 打开，仍不等价于 retained-source formation；必须以 fresh horizon/debt 与 official observer/C2/C3 共同开门为准。",
        "- ROST 是按计划中 Nora / row-wise angular stability 推荐新增的第一性尝试：若 retained source 被 row norm jitter 擦除，row-tangent source transport 应降低 row_norm_drift，同时保持 B/C train transfer 并在 fresh horizon 打败 AdamW/SGD/NoOp/random controls。",
        "- ROST 若只有 train-only rows 或局部 h100 正值，仍不等价于 source formation；必须 h100/h400/h800/h1600/h3200/debt 与 official observer/C2/C3 同时打开才可 promotion。",
        "- IVSC 是按计划中 signal-channel / reservoir / information-volume 推荐新增的第一性尝试：source update 的 per-example train-logit 位移应呈现 coherent drift > diffusion、非退化 effective rank，并且这些信号不能被 random/sign/corrupt controls 解释。",
        "- IVSC 若只在 train-only DDR/effective-rank 上有读数，仍不等价于 retained source；fresh h100-h3200/debt 和 official observer/C2/C3 gate 未开时必须保持 blocked。",
        "- FSMR 是按计划中 fast/slow source memory、AdEMAMix 式慢记忆与 source-state age 推荐新增的第一性尝试：若 source 真在训练流中形成可保留慢态，短轨迹 slow memory 应与 fast source update 保持正 alignment、低 overwrite，并相对 random/sign/corrupt memory controls 有可用 train transfer。",
        "- FSMR fresh C3 仍只比较 `FSMR-AdamWPlusFastSlowMemoryFU` 对 AdamW/SGD/NoOp/random matched controls 的真实 h100-h3200 horizon；即使 train-only memory certificate rows 打开，没有 fresh/debt 与 official observer/C2/C3 同开也不能进入 Line D。",
        "- FSMR damped repair 是 blocker-directed integration test：若失败来自注入过强或 slow-memory overwrite 过大，降低 `fu_lr`、提高 `memory_beta`、降低 refresh frequency 应改善 h800-h3200。若仍失败，就支持 source-memory principle 在当前候选/runner 下未形成稳健 retained source。",
        "- TRAC 是 FSMR-damped 后再次更换的第一性尝试，不是 slow-memory 阈值调参：若 source update 真是局部训练流坐标，source step 经短 AdamW transport 后，其 adjoint inverse 应能在 train-only round trip 中降低 closure residual，并相对 random/sign/corrupt controls 保持 transfer 和 transport consistency。",
        "- TRAC fresh C3 仍只比较 `TRAC-AdamWPlusTimeReversalAdjointFU` 对 AdamW/SGD/NoOp/random matched controls 的真实 h100-h3200 horizon；没有 fresh/debt 与 official observer/C2/C3 同开时不能进入 Line D。",
        "- TRAC damped repair 是 blocker-directed integration test：如果 TRAC 失败主要来自注入过强、transport mixing 过大或刷新过频，降低 `fu_lr`、降低 `transport_mix`、提高 refresh interval 应改善 h100-h3200。若仍失败，就支持 adjoint-transport principle 在当前候选/runner 下未形成稳健 retained source。",
        "- retained h800+h3200 positive rows 仍可作为旁证，但不能替代 official early-chain h100+h400+h800 label，也不能打开 Line D。",
        "- C2 solver matrix 是真实 tiny-model recompute evidence；但当 C0 retained-source certificate 未打开时，C2 只说明 solver integrity，不声明 C1 target 或 C3 source formation。",
        "- Terminal preservation 与 KAN mapping 均严格依赖 MLP C3/C4。blocked-before-gate 的 rows 是状态证据，不是实验成功或失败 horizon。",
        f"- 最新结论：route=`{route['route']}`，minimum_effective_progress=`{route['minimum_effective_progress']}`。若 BVFR/OSH/ROST/IVSC/FSMR/FSMR-damped/TRAC/TRAC-damped 仍未打开 official gate，本轮应写成实现/审计进展与 post-boundary negative evidence；不能把 train-only/fresh smoke 包装成 promotion。",
        "",
        "_完整 artifact 清单保留在 `v22_09_artifact_index.csv`，不放入复盘正文。_",
        "",
    ]
    V2209_RECAP_DOC.write_text("\n".join(text), encoding="utf-8")


def main() -> None:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir)
    route = _route(out_dir)
    write_json(out_dir / "v22_09_final_route.json", route)
    write_rows(out_dir / "v22_09_final_route.csv", [route])
    failure = _failure_taxonomy(route)
    write_rows(out_dir / "v22_09_failure_taxonomy.csv", failure)
    _write_recap(out_dir, route)
    index = artifact_index(out_dir)
    write_rows(out_dir / "v22_09_artifact_index.csv", index)
    code_packet = build_code_review_packet(out_dir)
    index = artifact_index(out_dir)
    write_rows(out_dir / "v22_09_artifact_index.csv", index)
    bundle = build_results_bundle(out_dir)
    index = artifact_index(out_dir)
    write_rows(out_dir / "v22_09_artifact_index.csv", index)
    append_exec(
        out_dir,
        f"{PYTHON} experiments/run_v22_09_finalize.py --out-dir {out_dir}",
        status="completed",
        note=f"route={route['route']} artifacts={len(index)} bundle={bundle} code_review_packet={code_packet}",
    )


if __name__ == "__main__":
    main()
