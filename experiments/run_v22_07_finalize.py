#!/usr/bin/env python3
"""v22.07 final route, recap, and packet builder."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys
import zipfile
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_v22_07_common import (  # noqa: E402
    PYTHON,
    V2207_RECAP_DOC,
    append_exec,
    artifact_index,
    build_packet,
    ensure_out,
    finite_float,
    int_flag,
    md_table,
    now_sg,
    read_json,
    read_rows,
    run_cmd,
    simple_svg,
    write_json,
    write_rows,
)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None)
    p.add_argument("--run-clean-self-test", type=int, default=1)
    return p


def _run_clean_self_test(out_dir: Path) -> tuple[int, str]:
    zip_path, _bundle = build_packet(out_dir)
    clean_root = out_dir / "clean_unzip_self_test"
    if clean_root.exists():
        shutil.rmtree(clean_root)
    clean_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(clean_root)
    source = clean_root / "02_SOURCE_TREE"
    command = [
        PYTHON,
        "experiments/run_v22_07_s014_truth_gate.py",
        "--mode",
        "all",
        "--source-root",
        str(source),
        "--self-contained-import-check",
        "1",
        "--out-dir",
        str(out_dir),
    ]
    code, log = run_cmd(command, cwd=source, timeout=1500)
    (out_dir / "v22_07_clean_unzip_self_test.log").write_text(log, encoding="utf-8")
    return code, str(source)


def _route(out_dir: Path) -> dict[str, Any]:
    code_rows = read_rows(out_dir / "v22_07_code_truth_gate.csv")
    code_pass = bool(code_rows) and all(int_flag(r.get("pass")) for r in code_rows)
    eff = read_json(out_dir / "v22_07_efficiency_route.json")
    dr = read_json(out_dir / "v22_07_drat_drbf_multibatch_route.json")
    metric = read_json(out_dir / "v22_07_metric_dynamics_route.json")
    repair = read_json(out_dir / "v22_07_source_observability_repair_route.json")
    c3_repair = read_json(out_dir / "v22_07_c3_repair_smoke_route.json")
    c3_integration = read_json(out_dir / "v22_07_c3_integration_repair_route.json")
    c3_semantic = read_json(out_dir / "v22_07_c3_semantic_target_repair_route.json")
    c3_preservation = read_json(out_dir / "v22_07_c3_preservation_repair_route.json")
    source_theory = read_json(out_dir / "v22_07_source_theory_repair_route.json")
    c3_semantic_summary = read_rows(out_dir / "v22_07_c3_semantic_target_repair_summary.csv")
    c3_preservation_summary = read_rows(out_dir / "v22_07_c3_preservation_repair_summary.csv")
    retained_solver = read_json(out_dir / "v22_07_retained_metric_solver_repair_route.json")
    retained_solver_summary = read_rows(out_dir / "v22_07_retained_metric_solver_repair_summary.csv")
    stagefcp = read_json(out_dir / "v22_07_stagefcp_hybrid_repair_route.json")
    stagefcp_summary = read_rows(out_dir / "v22_07_stagefcp_hybrid_repair_summary.csv")
    observer2 = read_json(out_dir / "v22_07_observer2_source_theory_route.json")
    observer2_summary = read_rows(out_dir / "v22_07_observer2_source_theory_summary.csv")
    observer3 = read_json(out_dir / "v22_07_observer3_flow_theory_route.json")
    observer3_summary = read_rows(out_dir / "v22_07_observer3_flow_theory_summary.csv")
    observer4 = read_json(out_dir / "v22_07_observer4_transfer_meta_route.json")
    observer4_summary = read_rows(out_dir / "v22_07_observer4_transfer_meta_summary.csv")
    observer5 = read_json(out_dir / "v22_07_observer5_activation_manifold_route.json")
    observer5_summary = read_rows(out_dir / "v22_07_observer5_activation_manifold_summary.csv")
    observer6 = read_json(out_dir / "v22_07_observer6_curvature_guard_route.json")
    observer6_summary = read_rows(out_dir / "v22_07_observer6_curvature_guard_summary.csv")
    observer7 = read_json(out_dir / "v22_07_observer7_control_nullspace_route.json")
    observer7_summary = read_rows(out_dir / "v22_07_observer7_control_nullspace_summary.csv")
    source_observer_nogo = read_json(out_dir / "v22_07_source_observer_nogo_route.json")
    terminal = read_json(out_dir / "v22_07_terminal_preservation_route.json")
    kan = read_json(out_dir / "v22_07_kan_source_mapping_route.json")
    queue = read_json(out_dir / "v22_07_queue_drain_report.json")
    queue_violation_rows = read_rows(out_dir / "v22_07_idle_violation.csv")
    queue_violation = any(int_flag(r.get("execution_contract_violation")) for r in queue_violation_rows)
    blockers = []
    if not code_pass:
        blockers.append("S0.14")
    if not int_flag(eff.get("D-CHE_S1_pass")) or not int_flag(eff.get("D-FOU_S1_pass")):
        blockers.append("D-CHE_or_D-FOU_S1")
    if not int_flag(dr.get("multibatch_status_closed")):
        blockers.append("D-RAT_D-RBF_multibatch_robust_or_telemetry")
    repair_c2_opened = int_flag(repair.get("C2_repair_pass_rows"))
    repair_c3_opened = int_flag(c3_repair.get("C3_repair_smoke_pass_rows"))
    integration_early_opened = int_flag(c3_integration.get("C3_integration_early_pass_rows"))
    semantic_early_opened = int_flag(c3_semantic.get("C3_semantic_early_pass_rows"))
    semantic_full_opened = int_flag(c3_semantic.get("C3_semantic_full_pass_rows"))
    semantic_h1600_pos = sum(1 for r in c3_semantic_summary if finite_float(r.get("source_vs_best_control_h1600"), -999.0) >= 0.005)
    semantic_h3200_pos = sum(1 for r in c3_semantic_summary if finite_float(r.get("source_vs_best_control_h3200"), -999.0) >= 0.005)
    preservation_rows = int_flag(c3_preservation.get("C3_preservation_rows"))
    preservation_early_opened = int_flag(c3_preservation.get("C3_preservation_early_pass_rows"))
    preservation_full_opened = int_flag(c3_preservation.get("C3_preservation_full_pass_rows"))
    preservation_h1600_pos = sum(1 for r in c3_preservation_summary if finite_float(r.get("source_vs_best_control_h1600"), -999.0) >= 0.005)
    preservation_h3200_pos = sum(1 for r in c3_preservation_summary if finite_float(r.get("source_vs_best_control_h3200"), -999.0) >= 0.005)
    retained_rows = int_flag(retained_solver.get("C3_retained_solver_rows"))
    retained_early_opened = int_flag(retained_solver.get("C3_retained_solver_early_pass_rows"))
    retained_full_opened = int_flag(retained_solver.get("C3_retained_solver_full_pass_rows"))
    retained_official_c2_rows = int_flag(retained_solver.get("official_C2_solver_claimed_rows"))
    retained_h1600_pos = sum(1 for r in retained_solver_summary if finite_float(r.get("source_vs_best_control_h1600"), -999.0) >= 0.005)
    retained_h3200_pos = sum(1 for r in retained_solver_summary if finite_float(r.get("source_vs_best_control_h3200"), -999.0) >= 0.005)
    stagefcp_rows = int_flag(stagefcp.get("C3_stagefcp_rows"))
    stagefcp_early_opened = int_flag(stagefcp.get("C3_stagefcp_early_pass_rows"))
    stagefcp_full_opened = int_flag(stagefcp.get("C3_stagefcp_full_pass_rows"))
    stagefcp_official_c2_rows = int_flag(stagefcp.get("official_C2_solver_claimed_rows"))
    stagefcp_h1600_pos = sum(1 for r in stagefcp_summary if finite_float(r.get("source_vs_best_control_h1600"), -999.0) >= 0.005)
    stagefcp_h3200_pos = sum(1 for r in stagefcp_summary if finite_float(r.get("source_vs_best_control_h3200"), -999.0) >= 0.005)
    observer2_rows = int_flag(observer2.get("C3_observer2_rows"))
    observer2_early_opened = int_flag(observer2.get("C3_observer2_early_pass_rows"))
    observer2_full_opened = int_flag(observer2.get("C3_observer2_full_pass_rows"))
    observer2_c1_rows = int_flag(observer2.get("C1_observer2_pass_rows"))
    observer2_h1600_pos = sum(1 for r in observer2_summary if finite_float(r.get("source_vs_best_control_h1600"), -999.0) >= 0.005)
    observer2_h3200_pos = sum(1 for r in observer2_summary if finite_float(r.get("source_vs_best_control_h3200"), -999.0) >= 0.005)
    observer3_rows = int_flag(observer3.get("C3_observer3_rows"))
    observer3_early_opened = int_flag(observer3.get("C3_observer3_early_pass_rows"))
    observer3_full_opened = int_flag(observer3.get("C3_observer3_full_pass_rows"))
    observer3_c1_rows = int_flag(observer3.get("C1_observer3_pass_rows"))
    observer3_h1600_pos = sum(1 for r in observer3_summary if finite_float(r.get("source_vs_best_control_h1600"), -999.0) >= 0.005)
    observer3_h3200_pos = sum(1 for r in observer3_summary if finite_float(r.get("source_vs_best_control_h3200"), -999.0) >= 0.005)
    observer4_rows = int_flag(observer4.get("C3_observer4_rows"))
    observer4_early_opened = int_flag(observer4.get("C3_observer4_early_pass_rows"))
    observer4_full_opened = int_flag(observer4.get("C3_observer4_full_pass_rows"))
    observer4_c1_rows = int_flag(observer4.get("C1_observer4_pass_rows"))
    observer4_h1600_pos = sum(1 for r in observer4_summary if finite_float(r.get("source_vs_best_control_h1600"), -999.0) >= 0.005)
    observer4_h3200_pos = sum(1 for r in observer4_summary if finite_float(r.get("source_vs_best_control_h3200"), -999.0) >= 0.005)
    observer5_rows = int_flag(observer5.get("C3_observer5_rows"))
    observer5_early_opened = int_flag(observer5.get("C3_observer5_early_pass_rows"))
    observer5_full_opened = int_flag(observer5.get("C3_observer5_full_pass_rows"))
    observer5_c1_rows = int_flag(observer5.get("C1_observer5_pass_rows"))
    observer5_h1600_pos = sum(1 for r in observer5_summary if finite_float(r.get("source_vs_best_control_h1600"), -999.0) >= 0.005)
    observer5_h3200_pos = sum(1 for r in observer5_summary if finite_float(r.get("source_vs_best_control_h3200"), -999.0) >= 0.005)
    observer6_rows = int_flag(observer6.get("C3_observer6_rows"))
    observer6_early_opened = int_flag(observer6.get("C3_observer6_early_pass_rows"))
    observer6_full_opened = int_flag(observer6.get("C3_observer6_full_pass_rows"))
    observer6_c1_rows = int_flag(observer6.get("C1_observer6_pass_rows"))
    observer6_h1600_pos = sum(1 for r in observer6_summary if finite_float(r.get("source_vs_best_control_h1600"), -999.0) >= 0.005)
    observer6_h3200_pos = sum(1 for r in observer6_summary if finite_float(r.get("source_vs_best_control_h3200"), -999.0) >= 0.005)
    observer7_rows = int_flag(observer7.get("C3_observer7_rows"))
    observer7_early_opened = int_flag(observer7.get("C3_observer7_early_pass_rows"))
    observer7_full_opened = int_flag(observer7.get("C3_observer7_full_pass_rows"))
    observer7_c1_rows = int_flag(observer7.get("C1_observer7_pass_rows"))
    observer7_h1600_pos = sum(1 for r in observer7_summary if finite_float(r.get("source_vs_best_control_h1600"), -999.0) >= 0.005)
    observer7_h3200_pos = sum(1 for r in observer7_summary if finite_float(r.get("source_vs_best_control_h3200"), -999.0) >= 0.005)
    if not int_flag(metric.get("C3_pass_rows")) and not repair_c3_opened:
        if repair_c2_opened:
            blockers.append("C3_source_formation_gate")
            if str(c3_repair.get("route", "")) == "C3-RepairSmokeBlocked":
                blockers.append("C3_repair_smoke_horizon_gate")
            if str(c3_integration.get("route", "")) == "C3-IntegrationRepairBlocked":
                blockers.append("C3_integration_horizon_gate")
            if str(c3_semantic.get("route", "")) == "C3-SemanticTargetRepairBlocked":
                if not int_flag(c3_semantic.get("C1_semantic_pass_rows")):
                    blockers.append("C1_semantic_target_gate")
                blockers.append("C3_semantic_h3200_retention_gate" if semantic_early_opened and not semantic_full_opened else "C3_semantic_target_horizon_gate")
            if str(c3_preservation.get("route", "")):
                if preservation_rows and not preservation_full_opened:
                    blockers.append("C3_preservation_h3200_retention_gate")
                if preservation_full_opened and not int_flag(c3_preservation.get("official_C2_solver_claimed")):
                    blockers.append("C3_preservation_official_C2_solver_gate")
            if str(source_theory.get("route", "")):
                if not int_flag(source_theory.get("source_theory_pass")):
                    blockers.append("source_theory_strict_retained_observer_gate")
            if str(retained_solver.get("route", "")):
                if retained_rows and not retained_full_opened:
                    blockers.append("C3_retained_metric_solver_h3200_gate")
                if retained_rows and not retained_official_c2_rows:
                    blockers.append("retained_metric_solver_official_C2_gate")
            if str(stagefcp.get("route", "")):
                if stagefcp_rows and not stagefcp_full_opened:
                    blockers.append("C3_stagefcp_h3200_retention_gate")
                if stagefcp_rows and not stagefcp_official_c2_rows:
                    blockers.append("stagefcp_official_C2_gate")
            if str(observer2.get("route", "")):
                if observer2_rows and not observer2_c1_rows:
                    blockers.append("observer2_source_theory_C1_gate")
                if observer2_rows and not observer2_full_opened:
                    blockers.append("C3_observer2_source_theory_h3200_gate")
            if str(observer3.get("route", "")):
                if observer3_rows and not observer3_c1_rows:
                    blockers.append("observer3_flow_theory_C1_gate")
                if observer3_rows and not observer3_full_opened:
                    blockers.append("C3_observer3_flow_theory_h3200_gate")
            if str(observer4.get("route", "")):
                if observer4_rows and not observer4_c1_rows:
                    blockers.append("observer4_transfer_meta_C1_gate")
                if observer4_rows and not observer4_full_opened:
                    blockers.append("C3_observer4_transfer_meta_h3200_gate")
            if str(observer5.get("route", "")):
                if observer5_rows and not observer5_c1_rows:
                    blockers.append("observer5_activation_manifold_C1_gate")
                if observer5_rows and not observer5_full_opened:
                    blockers.append("C3_observer5_activation_manifold_h3200_gate")
            if str(observer6.get("route", "")):
                if observer6_rows and not observer6_c1_rows:
                    blockers.append("observer6_curvature_guard_C1_gate")
                if observer6_rows and not observer6_full_opened:
                    blockers.append("C3_observer6_curvature_guard_h3200_gate")
            if str(observer7.get("route", "")):
                if observer7_rows and not observer7_c1_rows:
                    blockers.append("observer7_control_nullspace_C1_gate")
                if observer7_rows and not observer7_full_opened:
                    blockers.append("C3_observer7_control_nullspace_h3200_gate")
        else:
            blockers.append(str(metric.get("blocker", "C3_source_formation")))
    if str(terminal.get("decision", "")).startswith("D0D1Blocked"):
        blockers.append("terminal_preservation_not_entered_C3_source_gate" if repair_c2_opened else str(terminal.get("blocker", "terminal_preservation")))
    if str(kan.get("decision", "")) in {"KANMappingNotEntered", "KANSourceChannelWriterMissing"}:
        blockers.append("KAN_mapping_not_entered_C3_source_gate" if repair_c2_opened else str(kan.get("blocker", "KAN_mapping")))
    if queue_violation:
        blockers.append("4GPU_queue_idle_violation")
    dedup_blockers = []
    for item in blockers:
        for part in str(item).split(";"):
            part = part.strip()
            if part and part not in dedup_blockers:
                dedup_blockers.append(part)
    route_name = str(metric.get("functional_route", "R0-InProgress"))
    if int_flag(source_observer_nogo.get("local_no_go_boundary_claimed")):
        route_name = "C3-SourceObserverLocalNoGoBoundary"
    elif observer7_rows and observer7_early_opened and not observer7_full_opened:
        route_name = "C3-Observer7ControlNullspaceBlocked-H3200Washout"
    elif observer7_rows and str(observer7.get("route", "")) == "C3-Observer7ControlNullspaceBlocked":
        route_name = "C3-Observer7ControlNullspaceBlocked"
    elif observer7_full_opened:
        route_name = "C3-Observer7ControlNullspaceOpened-C2OfficialNotClaimed"
    elif observer6_rows and observer6_early_opened and not observer6_full_opened:
        route_name = "C3-Observer6CurvatureGuardBlocked-H3200Washout"
    elif observer6_rows and str(observer6.get("route", "")) == "C3-Observer6CurvatureGuardBlocked":
        route_name = "C3-Observer6CurvatureGuardBlocked"
    elif observer6_full_opened:
        route_name = "C3-Observer6CurvatureGuardOpened-C2OfficialNotClaimed"
    elif observer5_rows and observer5_early_opened and not observer5_full_opened:
        route_name = "C3-Observer5ActivationManifoldBlocked-H3200Washout"
    elif observer5_rows and str(observer5.get("route", "")) == "C3-Observer5ActivationManifoldBlocked":
        route_name = "C3-Observer5ActivationManifoldBlocked"
    elif observer5_full_opened:
        route_name = "C3-Observer5ActivationManifoldOpened-C2OfficialNotClaimed"
    elif observer4_rows and observer4_early_opened and not observer4_full_opened:
        route_name = "C3-Observer4TransferMetaBlocked-H3200Washout"
    elif observer4_rows and str(observer4.get("route", "")) == "C3-Observer4TransferMetaBlocked":
        route_name = "C3-Observer4TransferMetaBlocked"
    elif observer4_full_opened:
        route_name = "C3-Observer4TransferMetaOpened-C2OfficialNotClaimed"
    elif observer3_rows and observer3_early_opened and not observer3_full_opened:
        route_name = "C3-Observer3FlowTheoryBlocked-H3200Washout"
    elif observer3_rows and str(observer3.get("route", "")) == "C3-Observer3FlowTheoryBlocked":
        route_name = "C3-Observer3FlowTheoryBlocked"
    elif observer3_full_opened:
        route_name = "C3-Observer3FlowTheoryOpened-C2OfficialNotClaimed"
    elif observer2_rows and observer2_early_opened and not observer2_full_opened:
        route_name = "C3-Observer2SourceTheoryBlocked-H3200Washout"
    elif observer2_rows and str(observer2.get("route", "")) == "C3-Observer2SourceTheoryBlocked":
        route_name = "C3-Observer2SourceTheoryBlocked"
    elif observer2_full_opened:
        route_name = "C3-Observer2SourceTheoryOpened-C2OfficialNotClaimed"
    elif retained_rows and retained_early_opened and not retained_full_opened:
        route_name = "C3-RetainedMetricSolverBlocked-H3200Washout"
    elif retained_rows and str(retained_solver.get("route", "")) == "C3-RetainedMetricSolverBlocked":
        route_name = "C3-RetainedMetricSolverBlocked"
    elif retained_full_opened and not retained_official_c2_rows:
        route_name = "C3-RetainedMetricSolverOpened-C2OfficialNotClaimed"
    elif stagefcp_rows and stagefcp_early_opened and not stagefcp_full_opened:
        route_name = "C3-StageFCPHybridBlocked-H3200Washout"
    elif stagefcp_rows and str(stagefcp.get("route", "")) == "C3-StageFCPHybridBlocked":
        route_name = "C3-StageFCPHybridBlocked"
    elif stagefcp_full_opened and not stagefcp_official_c2_rows:
        route_name = "C3-StageFCPHybridOpened-C2OfficialNotClaimed"
    elif preservation_rows and preservation_early_opened and not preservation_full_opened:
        route_name = "C3-PreservationRepairBlocked-H3200Washout"
    elif preservation_rows and str(c3_preservation.get("route", "")) == "C3-PreservationRepairBlocked":
        route_name = "C3-PreservationRepairBlocked"
    elif preservation_full_opened:
        route_name = "C3-PreservationRepairOpened-C2OfficialNotClaimed"
    elif semantic_early_opened and not semantic_full_opened:
        route_name = "C3-SemanticEarlyOpened-H3200Washout"
    elif str(c3_semantic.get("route", "")) == "C3-SemanticTargetRepairBlocked":
        route_name = "C3-SemanticTargetRepairBlocked"
    elif str(c3_semantic.get("route", "")) == "C3-SemanticTargetRepairOpened":
        route_name = "C3-SemanticTargetRepairOpened-C2OfficialNotClaimed"
    elif str(c3_integration.get("route", "")) == "C3-IntegrationRepairBlocked":
        route_name = "C3-IntegrationRepairBlocked"
    elif integration_early_opened:
        route_name = "C3-IntegrationEarlySourceOpened-C3FullNotRun"
    elif str(c3_repair.get("route", "")) == "C3-RepairSmokeBlocked":
        route_name = "C3-RepairSmokeBlocked"
    elif repair_c2_opened:
        route_name = str(repair.get("repair_functional_route", route_name))
    if not code_pass:
        route_name = "R0-CodeMetricInvalid"
    elif queue_violation:
        route_name = f"{route_name}_ExecutionContractViolation"
    if int_flag(source_observer_nogo.get("local_no_go_boundary_claimed")):
        next_action = "local no-go boundary reached for attempted train-only source-observer families; stop same-family observer/solver variants and require a genuinely new source observability principle outside the attempted families, or explicitly record this as a local no-go boundary"
    elif observer7_rows and not observer7_full_opened:
        next_action = "Observer7 control-nullspace target also failed C1/full C3; current gradient, optimizer-flow, transfer-meta, activation-manifold, curvature-guard, and train-control-nullspace families are blocked, so next work needs a genuinely new source observability principle or an explicit no-go boundary"
    elif observer6_rows and not observer6_full_opened:
        next_action = "Observer6 lowered NDS with train-only HVP but still failed C1/full C3; current observer, activation-manifold, and curvature-guard families are blocked, so next work needs a genuinely new source observability principle or an explicit no-go boundary"
    elif observer5_rows and not observer5_full_opened:
        next_action = "Observer5 activation-manifold target failed C1/full C3; current gradient, optimizer-flow, transfer-meta, and representation-manifold observer families are blocked, so next work needs an explicit no-go boundary or a genuinely new principle beyond these families"
    elif observer4_rows and not observer4_full_opened:
        next_action = "Observer4 transfer meta-gradient opened only transient h100/h400/h800 and failed C1/h3200; stop this observer family and require a genuinely new principle or explicit no-go boundary"
    elif observer3_rows and not observer3_full_opened:
        next_action = "stop current train-only optimizer-flow observer family; Observer3 also failed C1/full C3, so next work requires a genuinely new observability principle or an explicit no-go boundary"
    elif observer2_rows and not observer2_full_opened:
        next_action = "stop current source-observability family; Observer2 also failed C1/full C3, so more work needs a genuinely new train-only observability theory rather than another bounded tweak"
    elif retained_rows and not retained_full_opened:
        next_action = "stop source-theory smoke, retained metric-solver, and staged semantic+metric solver variants; current train-only observer family needs new source observability theory before more C3/KAN work"
    elif stagefcp_rows and not stagefcp_full_opened:
        next_action = "stop staged semantic+metric solver hybrids; retained source still needs new train-only source target/solver theory before terminal/KAN"
    elif preservation_rows and not preservation_full_opened:
        next_action = "stop same-family semantic preservation variants; formulate new train-only retained-source theory and official C2 solver before terminal/KAN"
    elif preservation_full_opened:
        next_action = "independently re-run official C2/C3 solver path; do not promote semantic preservation smoke until official solver is claimed"
    elif semantic_early_opened and not semantic_full_opened:
        next_action = "treat semantic source as transient; design optimizer-state preservation/anti-washout with official C2 solver before any terminal/KAN run"
    elif str(c3_semantic.get("route", "")) == "C3-SemanticTargetRepairBlocked":
        next_action = "stop same-family target/control-gap/integration tweaks; formulate new train-only source observability theory before another C3 run"
    elif str(c3_integration.get("route", "")) == "C3-IntegrationRepairBlocked":
        next_action = "redesign target/source semantics beyond target magnitude, control-gap, and optimizer-integration repairs; then rerun C3 horizon"
    elif repair_c2_opened and not repair_c3_opened:
        next_action = "repair target-to-training-dynamics integration; avoid pure target magnitude rescale; then rerun C3 horizon"
    elif not int_flag(metric.get("C3_pass_rows")):
        next_action = "new source observability theory before KAN mapping"
    else:
        next_action = "run D1 preservation and KAN source-channel mapping"
    return {
        "route": route_name,
        "promotion_allowed": 0,
        "blocking_metric": ";".join(dedup_blockers),
        "S0_14_pass": int(code_pass),
        "D-CHE_S1_pass": int_flag(eff.get("D-CHE_S1_pass")),
        "D-FOU_S1_pass": int_flag(eff.get("D-FOU_S1_pass")),
        "DRAT_DRBF_multibatch_closed": int_flag(dr.get("multibatch_status_closed")),
        "C0_pass": int_flag(metric.get("C0_pass")),
        "C1_pass_rows": metric.get("C1_pass_rows", ""),
        "C2_pass_rows": metric.get("C2_pass_rows", ""),
        "C3_pass_rows": metric.get("C3_pass_rows", ""),
        "functional_route": metric.get("functional_route", ""),
        "repair_functional_route": repair.get("repair_functional_route", ""),
        "C0_repair_pass": int_flag(repair.get("C0_repair_pass")),
        "C1_repair_pass_rows": repair.get("C1_repair_pass_rows", ""),
        "C2_repair_pass_rows": repair.get("C2_repair_pass_rows", ""),
        "C3_repair_smoke_pass_rows": c3_repair.get("C3_repair_smoke_pass_rows", ""),
        "C3_repair_smoke_route": c3_repair.get("route", ""),
        "C3_integration_early_pass_rows": c3_integration.get("C3_integration_early_pass_rows", ""),
        "C3_integration_full_pass_rows": c3_integration.get("C3_integration_full_pass_rows", ""),
        "C3_integration_route": c3_integration.get("route", ""),
        "C1_semantic_pass_rows": c3_semantic.get("C1_semantic_pass_rows", ""),
        "C3_semantic_early_pass_rows": c3_semantic.get("C3_semantic_early_pass_rows", ""),
        "C3_semantic_full_pass_rows": c3_semantic.get("C3_semantic_full_pass_rows", ""),
        "C3_semantic_h1600_positive_rows": semantic_h1600_pos,
        "C3_semantic_h3200_positive_rows": semantic_h3200_pos,
        "C3_semantic_route": c3_semantic.get("route", ""),
        "C3_preservation_rows": c3_preservation.get("C3_preservation_rows", ""),
        "C3_preservation_early_pass_rows": c3_preservation.get("C3_preservation_early_pass_rows", ""),
        "C3_preservation_full_pass_rows": c3_preservation.get("C3_preservation_full_pass_rows", ""),
        "C3_preservation_h1600_positive_rows": preservation_h1600_pos,
        "C3_preservation_h3200_positive_rows": preservation_h3200_pos,
        "C3_preservation_route": c3_preservation.get("route", ""),
        "C3_preservation_blocker": c3_preservation.get("blocker", ""),
        "source_theory_route": source_theory.get("route", ""),
        "source_theory_limited_smoke_selection": source_theory.get("limited_smoke_selection", ""),
        "source_theory_retained_positive_rows": source_theory.get("retained_h800_h3200_positive", ""),
        "source_theory_official_early_chain_positive_rows": source_theory.get("official_early_chain_h100_h400_h800_positive", ""),
        "source_theory_smoke_score": source_theory.get("smoke_best_retained_score", ""),
        "source_theory_smoke_AUC": source_theory.get("smoke_best_retained_AUC", ""),
        "source_theory_smoke_precision_at_top20": source_theory.get("smoke_best_retained_precision_at_top20", ""),
        "C3_retained_solver_rows": retained_solver.get("C3_retained_solver_rows", ""),
        "C3_retained_solver_early_pass_rows": retained_solver.get("C3_retained_solver_early_pass_rows", ""),
        "C3_retained_solver_full_pass_rows": retained_solver.get("C3_retained_solver_full_pass_rows", ""),
        "C3_retained_solver_h1600_positive_rows": retained_h1600_pos,
        "C3_retained_solver_h3200_positive_rows": retained_h3200_pos,
        "C3_retained_solver_official_C2_rows": retained_solver.get("official_C2_solver_claimed_rows", ""),
        "C3_retained_solver_route": retained_solver.get("route", ""),
        "C3_retained_solver_blocker": retained_solver.get("blocker", ""),
        "C3_stagefcp_rows": stagefcp.get("C3_stagefcp_rows", ""),
        "C3_stagefcp_early_pass_rows": stagefcp.get("C3_stagefcp_early_pass_rows", ""),
        "C3_stagefcp_full_pass_rows": stagefcp.get("C3_stagefcp_full_pass_rows", ""),
        "C3_stagefcp_h1600_positive_rows": stagefcp_h1600_pos,
        "C3_stagefcp_h3200_positive_rows": stagefcp_h3200_pos,
        "C3_stagefcp_official_C2_rows": stagefcp.get("official_C2_solver_claimed_rows", ""),
        "C3_stagefcp_route": stagefcp.get("route", ""),
        "C3_stagefcp_blocker": stagefcp.get("blocker", ""),
        "C1_observer2_pass_rows": observer2.get("C1_observer2_pass_rows", ""),
        "C3_observer2_rows": observer2.get("C3_observer2_rows", ""),
        "C3_observer2_early_pass_rows": observer2.get("C3_observer2_early_pass_rows", ""),
        "C3_observer2_full_pass_rows": observer2.get("C3_observer2_full_pass_rows", ""),
        "C3_observer2_h1600_positive_rows": observer2_h1600_pos,
        "C3_observer2_h3200_positive_rows": observer2_h3200_pos,
        "C3_observer2_official_C2_rows": observer2.get("official_C2_solver_claimed_rows", ""),
        "C3_observer2_route": observer2.get("route", ""),
        "C3_observer2_blocker": observer2.get("blocker", ""),
        "C1_observer3_pass_rows": observer3.get("C1_observer3_pass_rows", ""),
        "C3_observer3_rows": observer3.get("C3_observer3_rows", ""),
        "C3_observer3_early_pass_rows": observer3.get("C3_observer3_early_pass_rows", ""),
        "C3_observer3_full_pass_rows": observer3.get("C3_observer3_full_pass_rows", ""),
        "C3_observer3_h1600_positive_rows": observer3_h1600_pos,
        "C3_observer3_h3200_positive_rows": observer3_h3200_pos,
        "C3_observer3_official_C2_rows": observer3.get("official_C2_solver_claimed_rows", ""),
        "C3_observer3_route": observer3.get("route", ""),
        "C3_observer3_blocker": observer3.get("blocker", ""),
        "C1_observer4_pass_rows": observer4.get("C1_observer4_pass_rows", ""),
        "C3_observer4_rows": observer4.get("C3_observer4_rows", ""),
        "C3_observer4_early_pass_rows": observer4.get("C3_observer4_early_pass_rows", ""),
        "C3_observer4_full_pass_rows": observer4.get("C3_observer4_full_pass_rows", ""),
        "C3_observer4_h1600_positive_rows": observer4_h1600_pos,
        "C3_observer4_h3200_positive_rows": observer4_h3200_pos,
        "C3_observer4_official_C2_rows": observer4.get("official_C2_solver_claimed_rows", ""),
        "C3_observer4_route": observer4.get("route", ""),
        "C3_observer4_blocker": observer4.get("blocker", ""),
        "C1_observer5_pass_rows": observer5.get("C1_observer5_pass_rows", ""),
        "C3_observer5_rows": observer5.get("C3_observer5_rows", ""),
        "C3_observer5_early_pass_rows": observer5.get("C3_observer5_early_pass_rows", ""),
        "C3_observer5_full_pass_rows": observer5.get("C3_observer5_full_pass_rows", ""),
        "C3_observer5_h1600_positive_rows": observer5_h1600_pos,
        "C3_observer5_h3200_positive_rows": observer5_h3200_pos,
        "C3_observer5_official_C2_rows": observer5.get("official_C2_solver_claimed_rows", ""),
        "C3_observer5_route": observer5.get("route", ""),
        "C3_observer5_blocker": observer5.get("blocker", ""),
        "C1_observer6_pass_rows": observer6.get("C1_observer6_pass_rows", ""),
        "C3_observer6_rows": observer6.get("C3_observer6_rows", ""),
        "C3_observer6_early_pass_rows": observer6.get("C3_observer6_early_pass_rows", ""),
        "C3_observer6_full_pass_rows": observer6.get("C3_observer6_full_pass_rows", ""),
        "C3_observer6_h1600_positive_rows": observer6_h1600_pos,
        "C3_observer6_h3200_positive_rows": observer6_h3200_pos,
        "C3_observer6_official_C2_rows": observer6.get("official_C2_solver_claimed_rows", ""),
        "C3_observer6_route": observer6.get("route", ""),
        "C3_observer6_blocker": observer6.get("blocker", ""),
        "C1_observer7_pass_rows": observer7.get("C1_observer7_pass_rows", ""),
        "C3_observer7_rows": observer7.get("C3_observer7_rows", ""),
        "C3_observer7_early_pass_rows": observer7.get("C3_observer7_early_pass_rows", ""),
        "C3_observer7_full_pass_rows": observer7.get("C3_observer7_full_pass_rows", ""),
        "C3_observer7_h1600_positive_rows": observer7_h1600_pos,
        "C3_observer7_h3200_positive_rows": observer7_h3200_pos,
        "C3_observer7_official_C2_rows": observer7.get("official_C2_solver_claimed_rows", ""),
        "C3_observer7_route": observer7.get("route", ""),
        "C3_observer7_blocker": observer7.get("blocker", ""),
        "source_observer_local_no_go": source_observer_nogo.get("local_no_go_boundary_claimed", ""),
        "source_observer_scope": source_observer_nogo.get("scope", ""),
        "source_observer_universal_no_go_claimed": source_observer_nogo.get("universal_scientific_no_go_claimed", ""),
        "source_observer_family_count": source_observer_nogo.get("observer_family_count", ""),
        "source_observer_rows": source_observer_nogo.get("observer_rows", ""),
        "source_observer_early_pass_rows": source_observer_nogo.get("observer_early_pass_rows", ""),
        "source_observer_full_pass_rows": source_observer_nogo.get("observer_full_pass_rows", ""),
        "source_observer_c1_pass_rows": source_observer_nogo.get("observer_c1_pass_rows", ""),
        "source_observer_h1600_positive_rows": source_observer_nogo.get("observer_h1600_positive_rows", ""),
        "source_observer_h3200_positive_rows": source_observer_nogo.get("observer_h3200_positive_rows", ""),
        "source_observer_best_C1_B2": source_observer_nogo.get("best_observer_C1_B2_transfer_gain", ""),
        "source_observer_best_h3200": source_observer_nogo.get("best_observer_h3200", ""),
        "source_observer_nogo_route": source_observer_nogo.get("route", ""),
        "source_observer_nogo_blocker": source_observer_nogo.get("blocker", ""),
        "terminal_preservation_decision": terminal.get("decision", ""),
        "KAN_mapping_decision": kan.get("decision", ""),
        "queue_completed": queue.get("completed_tasks", ""),
        "queue_violation": int(queue_violation),
        "next_codex_action": next_action,
    }


def _write_required_figures(out_dir: Path) -> None:
    placeholders = {
        "code_truth_dashboard.svg": ("v22.07 code truth dashboard", read_rows(out_dir / "v22_07_code_truth_gate.csv"), "pass"),
        "semantic_alias_heatmap_parameter.svg": ("v22.07 semantic alias parameter", read_rows(out_dir / "v22_07_functional_alias_matrix.csv"), "semantic_alias"),
        "semantic_alias_heatmap_function_displacement.svg": ("v22.07 semantic alias function displacement", read_rows(out_dir / "v22_07_function_displacement_alias_matrix.csv"), "function_displacement_cosine"),
        "gpu_utilization_timeline.svg": ("v22.07 GPU utilization timeline", read_rows(out_dir / "v22_07_gpu_utilization_timeline.csv"), "duration_sec"),
    }
    for name, (title, rows, metric) in placeholders.items():
        path = out_dir / "figures" / name
        if not path.exists():
            simple_svg(path, title, rows, metric)


def _best_source_row(c3_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not c3_rows:
        return {}
    return sorted(
        c3_rows,
        key=lambda r: (
            int_flag(r.get("C3_source_formation_pass")),
            finite_float(r.get("source_vs_best_control_h3200"), -999.0),
            finite_float(r.get("source_vs_best_control_h800"), -999.0),
        ),
        reverse=True,
    )[0]


def _write_compat_bundle(out_dir: Path) -> Path:
    compat = out_dir / "v22_06_results_bundle.zip"
    if compat.exists():
        compat.unlink()
    packet = out_dir / "v22_07_code_review_packet"
    with zipfile.ZipFile(compat, "w", compression=zipfile.ZIP_DEFLATED) as z:
        if packet.exists():
            for path in sorted(packet.rglob("*")):
                if path.is_file():
                    z.write(path, Path("v22_07_code_review_packet") / path.relative_to(packet))
        for path in sorted(out_dir.rglob("*")):
            if (
                not path.is_file()
                or path.suffix.lower() == ".zip"
                or "v22_07_code_review_packet" in str(path)
                or "clean_unzip_self_test" in path.parts
            ):
                continue
            z.write(path, Path("results") / path.relative_to(out_dir))
    return compat


def main() -> None:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir).resolve()
    if int(args.run_clean_self_test):
        code, source = _run_clean_self_test(out_dir)
        append_exec(
            out_dir,
            f"{PYTHON} experiments/run_v22_07_s014_truth_gate.py --mode all --source-root {source} --self-contained-import-check 1 --out-dir {out_dir}",
            status="completed" if code == 0 else "blocked",
            note=f"clean_unzip_returncode={code}",
        )
    _write_required_figures(out_dir)
    route = _route(out_dir)
    write_json(out_dir / "v22_07_final_route.json", route)
    write_rows(out_dir / "v22_07_final_route.csv", [route])
    artifact_rows = artifact_index(out_dir)
    write_rows(out_dir / "v22_07_artifact_index.csv", artifact_rows)

    code_rows = read_rows(out_dir / "v22_07_code_truth_gate.csv")
    semantic_rows = read_rows(out_dir / "v22_07_functional_semantic_contract.csv")
    alias_rows = read_rows(out_dir / "v22_07_semantic_noncollapse_summary.csv")
    eff_rows = read_rows(out_dir / "v22_07_efficiency_full_loop_reconfirm.csv")
    dr_rows = read_rows(out_dir / "v22_07_drat_drbf_multibatch_summary.csv")
    dr_repair_rows = read_rows(out_dir / "v22_07_drat_drbf_repair_attempts_summary.csv")
    c0_summary = read_rows(out_dir / "v22_07_c0_no_commit_source_estimator_summary.csv")
    c1_summary = read_rows(out_dir / "v22_07_c1_target_contrast_summary.csv")
    c2_summary = read_rows(out_dir / "v22_07_c2_metric_solver_summary.csv")
    c3_rows = read_rows(out_dir / "v22_07_c3_source_formation_matrix.csv")
    c0_repair_summary = read_rows(out_dir / "v22_07_c0_estimator_repair_summary.csv")
    c1_repair_summary = read_rows(out_dir / "v22_07_c1_target_repair_summary.csv")
    c2_repair_summary = read_rows(out_dir / "v22_07_c2_solver_repair_summary.csv")
    c3_repair_summary = read_rows(out_dir / "v22_07_c3_repair_smoke_summary.csv")
    source_repair_route = read_json(out_dir / "v22_07_source_observability_repair_route.json")
    c3_repair_route = read_json(out_dir / "v22_07_c3_repair_smoke_route.json")
    c3_integration_summary = read_rows(out_dir / "v22_07_c3_integration_repair_summary.csv")
    c3_integration_route = read_json(out_dir / "v22_07_c3_integration_repair_route.json")
    c3_semantic_c1 = read_rows(out_dir / "v22_07_c3_semantic_target_repair_c1.csv")
    c3_semantic_summary = read_rows(out_dir / "v22_07_c3_semantic_target_repair_summary.csv")
    c3_semantic_route = read_json(out_dir / "v22_07_c3_semantic_target_repair_route.json")
    c3_preservation_summary = read_rows(out_dir / "v22_07_c3_preservation_repair_summary.csv")
    c3_preservation_route = read_json(out_dir / "v22_07_c3_preservation_repair_route.json")
    source_theory_summary = read_rows(out_dir / "v22_07_source_theory_repair_summary.csv")
    source_theory_top = read_rows(out_dir / "v22_07_source_theory_top_candidates.csv")
    source_theory_route = read_json(out_dir / "v22_07_source_theory_repair_route.json")
    retained_solver_summary = read_rows(out_dir / "v22_07_retained_metric_solver_repair_summary.csv")
    retained_solver_route = read_json(out_dir / "v22_07_retained_metric_solver_repair_route.json")
    stagefcp_summary = read_rows(out_dir / "v22_07_stagefcp_hybrid_repair_summary.csv")
    stagefcp_route = read_json(out_dir / "v22_07_stagefcp_hybrid_repair_route.json")
    observer2_c1 = read_rows(out_dir / "v22_07_observer2_source_theory_c1.csv")
    observer2_summary = read_rows(out_dir / "v22_07_observer2_source_theory_summary.csv")
    observer2_route = read_json(out_dir / "v22_07_observer2_source_theory_route.json")
    observer3_c1 = read_rows(out_dir / "v22_07_observer3_flow_theory_c1.csv")
    observer3_summary = read_rows(out_dir / "v22_07_observer3_flow_theory_summary.csv")
    observer3_route = read_json(out_dir / "v22_07_observer3_flow_theory_route.json")
    observer4_c1 = read_rows(out_dir / "v22_07_observer4_transfer_meta_c1.csv")
    observer4_summary = read_rows(out_dir / "v22_07_observer4_transfer_meta_summary.csv")
    observer4_route = read_json(out_dir / "v22_07_observer4_transfer_meta_route.json")
    observer5_c1 = read_rows(out_dir / "v22_07_observer5_activation_manifold_c1.csv")
    observer5_summary = read_rows(out_dir / "v22_07_observer5_activation_manifold_summary.csv")
    observer5_route = read_json(out_dir / "v22_07_observer5_activation_manifold_route.json")
    observer6_c1 = read_rows(out_dir / "v22_07_observer6_curvature_guard_c1.csv")
    observer6_summary = read_rows(out_dir / "v22_07_observer6_curvature_guard_summary.csv")
    observer6_route = read_json(out_dir / "v22_07_observer6_curvature_guard_route.json")
    observer7_c1 = read_rows(out_dir / "v22_07_observer7_control_nullspace_c1.csv")
    observer7_summary = read_rows(out_dir / "v22_07_observer7_control_nullspace_summary.csv")
    observer7_route = read_json(out_dir / "v22_07_observer7_control_nullspace_route.json")
    source_observer_nogo_summary = read_rows(out_dir / "v22_07_source_observer_nogo_family_summary.csv")
    source_observer_nogo_route = read_json(out_dir / "v22_07_source_observer_nogo_route.json")
    terminal_rows = read_rows(out_dir / "v22_07_terminal_preservation_summary.csv")
    kan_rows = read_rows(out_dir / "v22_07_kan_source_mapping_summary.csv")
    queue_rows = read_rows(out_dir / "v22_07_gpu_assignment_manifest.csv")
    idle_rows = read_rows(out_dir / "v22_07_idle_violation.csv")
    best = _best_source_row(c3_rows)
    best_c0 = sorted(c0_summary, key=lambda r: finite_float(r.get("AUC_predict_h800_positive"), -1.0), reverse=True)[0] if c0_summary else {}
    best_pres_h1600 = sorted(c3_preservation_summary, key=lambda r: finite_float(r.get("source_vs_best_control_h1600"), -999.0), reverse=True)[0] if c3_preservation_summary else {}
    best_pres_h3200 = sorted(c3_preservation_summary, key=lambda r: finite_float(r.get("source_vs_best_control_h3200"), -999.0), reverse=True)[0] if c3_preservation_summary else {}
    best_ret_h1600 = sorted(retained_solver_summary, key=lambda r: finite_float(r.get("source_vs_best_control_h1600"), -999.0), reverse=True)[0] if retained_solver_summary else {}
    best_ret_h3200 = sorted(retained_solver_summary, key=lambda r: finite_float(r.get("source_vs_best_control_h3200"), -999.0), reverse=True)[0] if retained_solver_summary else {}
    best_stage_h1600 = sorted(stagefcp_summary, key=lambda r: finite_float(r.get("source_vs_best_control_h1600"), -999.0), reverse=True)[0] if stagefcp_summary else {}
    best_stage_h3200 = sorted(stagefcp_summary, key=lambda r: finite_float(r.get("source_vs_best_control_h3200"), -999.0), reverse=True)[0] if stagefcp_summary else {}
    best_obs_h1600 = sorted(observer2_summary, key=lambda r: finite_float(r.get("source_vs_best_control_h1600"), -999.0), reverse=True)[0] if observer2_summary else {}
    best_obs_h3200 = sorted(observer2_summary, key=lambda r: finite_float(r.get("source_vs_best_control_h3200"), -999.0), reverse=True)[0] if observer2_summary else {}
    best_flow_h1600 = sorted(observer3_summary, key=lambda r: finite_float(r.get("source_vs_best_control_h1600"), -999.0), reverse=True)[0] if observer3_summary else {}
    best_flow_h3200 = sorted(observer3_summary, key=lambda r: finite_float(r.get("source_vs_best_control_h3200"), -999.0), reverse=True)[0] if observer3_summary else {}
    best_meta_h1600 = sorted(observer4_summary, key=lambda r: finite_float(r.get("source_vs_best_control_h1600"), -999.0), reverse=True)[0] if observer4_summary else {}
    best_meta_h3200 = sorted(observer4_summary, key=lambda r: finite_float(r.get("source_vs_best_control_h3200"), -999.0), reverse=True)[0] if observer4_summary else {}
    best_mani_h1600 = sorted(observer5_summary, key=lambda r: finite_float(r.get("source_vs_best_control_h1600"), -999.0), reverse=True)[0] if observer5_summary else {}
    best_mani_h3200 = sorted(observer5_summary, key=lambda r: finite_float(r.get("source_vs_best_control_h3200"), -999.0), reverse=True)[0] if observer5_summary else {}
    best_curv_h1600 = sorted(observer6_summary, key=lambda r: finite_float(r.get("source_vs_best_control_h1600"), -999.0), reverse=True)[0] if observer6_summary else {}
    best_curv_h3200 = sorted(observer6_summary, key=lambda r: finite_float(r.get("source_vs_best_control_h3200"), -999.0), reverse=True)[0] if observer6_summary else {}
    best_null_h1600 = sorted(observer7_summary, key=lambda r: finite_float(r.get("source_vs_best_control_h1600"), -999.0), reverse=True)[0] if observer7_summary else {}
    best_null_h3200 = sorted(observer7_summary, key=lambda r: finite_float(r.get("source_vs_best_control_h3200"), -999.0), reverse=True)[0] if observer7_summary else {}

    recap = [
        "# DG-KAN v22.07 MetricDynamics FunctionalUpdate BasisEfficiency 4GPU 实验结果复盘",
        "",
        f"生成时间：{now_sg()}",
        "",
        "## Route",
        "",
        f"- route: `{route.get('route')}`",
        f"- promotion_allowed: {route.get('promotion_allowed')}",
        f"- blocking_metric: `{route.get('blocking_metric')}`",
        f"- S0.14 pass: {route.get('S0_14_pass')}",
        f"- D-CHE/D-FOU S1 pass: {route.get('D-CHE_S1_pass')} / {route.get('D-FOU_S1_pass')}",
        f"- D-RAT/D-RBF multibatch closed: {route.get('DRAT_DRBF_multibatch_closed')}",
        f"- C0/C1/C2/C3: {route.get('C0_pass')}/{route.get('C1_pass_rows')}/{route.get('C2_pass_rows')}/{route.get('C3_pass_rows')}",
        f"- repair C0/C1/C2/C3-smoke: {route.get('C0_repair_pass')}/{route.get('C1_repair_pass_rows')}/{route.get('C2_repair_pass_rows')}/{route.get('C3_repair_smoke_pass_rows')}",
        f"- C3 integration repair early/full route: {route.get('C3_integration_early_pass_rows')}/{route.get('C3_integration_full_pass_rows')} `{route.get('C3_integration_route')}`",
        f"- source theory repair: `{route.get('source_theory_route')}` score={route.get('source_theory_smoke_score')} AUC={route.get('source_theory_smoke_AUC')} p20={route.get('source_theory_smoke_precision_at_top20')}",
        f"- C1/C3 semantic target repair: {route.get('C1_semantic_pass_rows')}/{route.get('C3_semantic_early_pass_rows')}/{route.get('C3_semantic_full_pass_rows')} h1600_pos={route.get('C3_semantic_h1600_positive_rows')} h3200_pos={route.get('C3_semantic_h3200_positive_rows')} `{route.get('C3_semantic_route')}`",
        f"- C3 preservation repair: rows={route.get('C3_preservation_rows')} early/full={route.get('C3_preservation_early_pass_rows')}/{route.get('C3_preservation_full_pass_rows')} h1600_pos={route.get('C3_preservation_h1600_positive_rows')} h3200_pos={route.get('C3_preservation_h3200_positive_rows')} `{route.get('C3_preservation_route')}` blocker=`{route.get('C3_preservation_blocker')}`",
        f"- C3 retained metric solver repair: rows={route.get('C3_retained_solver_rows')} early/full={route.get('C3_retained_solver_early_pass_rows')}/{route.get('C3_retained_solver_full_pass_rows')} h1600_pos={route.get('C3_retained_solver_h1600_positive_rows')} h3200_pos={route.get('C3_retained_solver_h3200_positive_rows')} official_C2_rows={route.get('C3_retained_solver_official_C2_rows')} `{route.get('C3_retained_solver_route')}` blocker=`{route.get('C3_retained_solver_blocker')}`",
        f"- C3 Stage F/C/P hybrid repair: rows={route.get('C3_stagefcp_rows')} early/full={route.get('C3_stagefcp_early_pass_rows')}/{route.get('C3_stagefcp_full_pass_rows')} h1600_pos={route.get('C3_stagefcp_h1600_positive_rows')} h3200_pos={route.get('C3_stagefcp_h3200_positive_rows')} official_C2_rows={route.get('C3_stagefcp_official_C2_rows')} `{route.get('C3_stagefcp_route')}` blocker=`{route.get('C3_stagefcp_blocker')}`",
        f"- C3 Observer2 source theory: C1_pass={route.get('C1_observer2_pass_rows')} rows={route.get('C3_observer2_rows')} early/full={route.get('C3_observer2_early_pass_rows')}/{route.get('C3_observer2_full_pass_rows')} h1600_pos={route.get('C3_observer2_h1600_positive_rows')} h3200_pos={route.get('C3_observer2_h3200_positive_rows')} `{route.get('C3_observer2_route')}` blocker=`{route.get('C3_observer2_blocker')}`",
        f"- C3 Observer3 optimizer-flow theory: C1_pass={route.get('C1_observer3_pass_rows')} rows={route.get('C3_observer3_rows')} early/full={route.get('C3_observer3_early_pass_rows')}/{route.get('C3_observer3_full_pass_rows')} h1600_pos={route.get('C3_observer3_h1600_positive_rows')} h3200_pos={route.get('C3_observer3_h3200_positive_rows')} `{route.get('C3_observer3_route')}` blocker=`{route.get('C3_observer3_blocker')}`",
        f"- C3 Observer4 transfer meta-gradient: C1_pass={route.get('C1_observer4_pass_rows')} rows={route.get('C3_observer4_rows')} early/full={route.get('C3_observer4_early_pass_rows')}/{route.get('C3_observer4_full_pass_rows')} h1600_pos={route.get('C3_observer4_h1600_positive_rows')} h3200_pos={route.get('C3_observer4_h3200_positive_rows')} `{route.get('C3_observer4_route')}` blocker=`{route.get('C3_observer4_blocker')}`",
        f"- C3 Observer5 activation-manifold theory: C1_pass={route.get('C1_observer5_pass_rows')} rows={route.get('C3_observer5_rows')} early/full={route.get('C3_observer5_early_pass_rows')}/{route.get('C3_observer5_full_pass_rows')} h1600_pos={route.get('C3_observer5_h1600_positive_rows')} h3200_pos={route.get('C3_observer5_h3200_positive_rows')} `{route.get('C3_observer5_route')}` blocker=`{route.get('C3_observer5_blocker')}`",
        f"- C3 Observer6 curvature-guard theory: C1_pass={route.get('C1_observer6_pass_rows')} rows={route.get('C3_observer6_rows')} early/full={route.get('C3_observer6_early_pass_rows')}/{route.get('C3_observer6_full_pass_rows')} h1600_pos={route.get('C3_observer6_h1600_positive_rows')} h3200_pos={route.get('C3_observer6_h3200_positive_rows')} `{route.get('C3_observer6_route')}` blocker=`{route.get('C3_observer6_blocker')}`",
        f"- C3 Observer7 control-nullspace theory: C1_pass={route.get('C1_observer7_pass_rows')} rows={route.get('C3_observer7_rows')} early/full={route.get('C3_observer7_early_pass_rows')}/{route.get('C3_observer7_full_pass_rows')} h1600_pos={route.get('C3_observer7_h1600_positive_rows')} h3200_pos={route.get('C3_observer7_h3200_positive_rows')} `{route.get('C3_observer7_route')}` blocker=`{route.get('C3_observer7_blocker')}`",
        f"- source-observer local no-go boundary: local_no_go={route.get('source_observer_local_no_go')} families={route.get('source_observer_family_count')} rows={route.get('source_observer_rows')} observer_c1/full={route.get('source_observer_c1_pass_rows')}/{route.get('source_observer_full_pass_rows')} h1600/h3200_pos={route.get('source_observer_h1600_positive_rows')}/{route.get('source_observer_h3200_positive_rows')} universal_no_go={route.get('source_observer_universal_no_go_claimed')} `{route.get('source_observer_nogo_route')}`",
        f"- repair_functional_route: `{route.get('repair_functional_route')}`",
        f"- terminal_preservation_decision: `{route.get('terminal_preservation_decision')}`",
        f"- KAN_mapping_decision: `{route.get('KAN_mapping_decision')}`",
        f"- next_codex_action: {route.get('next_codex_action')}",
        "",
        "## Part A S0.14 Code / Semantic / Solver Gate",
        "",
        md_table(code_rows, ["check", "pass", "metric", "value", "blocker"], 40),
        "",
        "### Functional Semantic Contract",
        "",
        md_table(semantic_rows, ["mechanism_id", "target_constructor_type", "metric_operator_type", "solver_type", "uses_future_or_validation", "uses_audit_metric_for_direction", "is_proxy"], 20),
        "",
        "### Semantic Alias Summary",
        "",
        md_table(alias_rows, ["pairs", "semantic_alias_pairs", "undeclared_alias_pairs", "pass", "gradient_norm"], 10),
        "",
        "## Part B Basis Efficiency / Officialization",
        "",
        md_table(eff_rows, ["carrier", "variant_family", "forward_ratio_vs_mlp", "backward_ratio_vs_mlp", "step_ratio_vs_mlp", "memory_ratio_vs_mlp", "same_kernel_functional_runner_proof", "fallback_kernel_used", "v22_07_S1_pass", "v22_07_blocker"], 20),
        "",
        md_table(dr_rows, ["carrier", "profile_rows", "robust_production_pass_rows", "near_E1_rows", "component_telemetry_complete_rows", "best_forward_ratio", "best_step_ratio", "best_memory_ratio", "decision", "blocker", "repair_attempt_rows", "repair_micro_near_E1_rows", "repair_nonreference_micro_near_E1_rows", "repair_best_variant", "repair_attempt_status", "repair_blocker"], 20),
        "",
        "### D-RAT / D-RBF Blocker Repair Attempt",
        "",
        md_table(dr_repair_rows, ["carrier", "repair_attempt_rows", "repair_micro_near_E1_rows", "repair_nonreference_micro_near_E1_rows", "repair_component_telemetry_complete_rows", "repair_best_forward_ratio", "repair_best_step_ratio", "repair_best_memory_ratio", "repair_best_variant", "repair_best_batch_size", "repair_attempt_status", "repair_blocker", "official_closure_claimed"], 20),
        "",
        "## Part C Metric-As-Dynamics Functional Update",
        "",
        f"- best C0 estimator: `{best_c0.get('estimator', '')}` AUC_h800={best_c0.get('AUC_predict_h800_positive', '')} precision_top20={best_c0.get('precision_at_top20', '')}",
        f"- best source row: `{best.get('v21_id', '')}` h100={best.get('source_vs_best_control_h100', '')} h400={best.get('source_vs_best_control_h400', '')} h800={best.get('source_vs_best_control_h800', '')} h1600={best.get('source_vs_best_control_h1600', '')} h3200={best.get('source_vs_best_control_h3200', '')} C3={best.get('C3_source_formation_pass', '')}",
        "",
        "### C0 No-Commit Source Estimator",
        "",
        md_table(c0_summary, ["estimator", "rows", "AUC_predict_h800_positive", "AUC_predict_h3200_positive", "Spearman_score_vs_source_h800", "precision_at_top20", "recall_at_top20", "control_equivalent_fraction", "C0_pass", "blocker"], 20),
        "",
        "### C1 Target Contrast",
        "",
        md_table(c1_summary, ["target_family", "metric_family", "rows", "C1_pass_rows", "B2_transfer_gain_mean", "B3_safety_gain_mean", "random_target_gap_mean", "sign_flip_gap_mean", "corrupt_gap_mean"], 30),
        "",
        "### C2 True Metric Solver",
        "",
        md_table(c2_summary, ["solver_level", "target_family", "metric_family", "rows", "C2_pass_rows", "ActuationR2_mean", "projection_residual_Gf_mean", "B2_transfer_gain_mean", "solve_time_ms_mean", "make_update_wall_ms_mean"], 30),
        "",
        "### C3 Source Formation",
        "",
        md_table(c3_rows, ["v21_id", "rows", "source_vs_best_control_h100", "source_vs_best_control_h400", "source_vs_best_control_h800", "source_vs_best_control_h1600", "source_vs_best_control_h2400", "source_vs_best_control_h3200", "row_positive_count_h3200", "C1_pass_rows", "C2_pass_rows", "entered_after_C0C1C2", "debt_not_exploded", "C3_source_formation_pass", "C3_blocker"], 40),
        "",
        "### C0/C1/C2 Repair Attempts",
        "",
        md_table(c0_repair_summary, ["repair_estimator", "AUC_predict_h800_positive", "precision_at_top20", "control_equivalent_fraction", "C0_repair_pass", "repair_actions", "blocker"], 20),
        "",
        md_table(c1_repair_summary, ["target_repair_variant", "rows", "C1_repair_pass_rows", "B2_transfer_gain_mean", "B3_safety_gain_mean", "random_target_gap_mean", "sign_flip_gap_mean", "corrupt_gap_mean", "best_v21_id", "best_C1_repair_pass"], 20),
        "",
        md_table(c2_repair_summary, ["target_repair_variant", "rows", "C2_repair_pass_rows", "projection_applicable_rows", "projection_residual_Gf_mean", "ActuationR2_mean", "B2_transfer_gain_mean", "B3_safety_gain_mean", "best_v21_id", "best_C2_repair_pass", "blocker"], 20),
        "",
        "### C3 Repair Smoke",
        "",
        f"- C3 repair smoke route: `{c3_repair_route.get('route', '')}`",
        f"- row_positive_count_h100/h400/h800/h3200: {c3_repair_route.get('row_positive_count_h100', '')}/{c3_repair_route.get('row_positive_count_h400', '')}/{c3_repair_route.get('row_positive_count_h800', '')}/{c3_repair_route.get('row_positive_count_h3200', '')}",
        md_table(c3_repair_summary, ["v21_id", "target_repair_variant", "source_vs_best_control_h100", "source_vs_best_control_h400", "source_vs_best_control_h800", "source_vs_best_control_h1600", "source_vs_best_control_h3200", "debt_not_exploded", "C3_repair_smoke_pass", "official_C3_pass", "blocker"], 20),
        "",
        "### C3 Integration Repair",
        "",
        f"- C3 integration repair route: `{c3_integration_route.get('route', '')}`",
        f"- row_positive_count_h100/h400/h800: {c3_integration_route.get('row_positive_count_h100', '')}/{c3_integration_route.get('row_positive_count_h400', '')}/{c3_integration_route.get('row_positive_count_h800', '')}",
        md_table(c3_integration_summary, ["run_kind", "v21_id", "target_repair_variant", "source_vs_best_control_h100", "source_vs_best_control_h400", "source_vs_best_control_h800", "source_vs_best_control_h1600", "source_vs_best_control_h3200", "target_injection_norm_sum", "conflict_projection_count", "C3_integration_early_pass", "official_C3_pass", "blocker"], 20),
        "",
        "### Source Theory Repair",
        "",
        f"- source theory route: `{source_theory_route.get('route', '')}`",
        f"- retained h800+h3200 positives / official h100+h400+h800 positives: {source_theory_route.get('retained_h800_h3200_positive', '')}/{source_theory_route.get('official_early_chain_h100_h400_h800_positive', '')}",
        f"- smoke score: `{source_theory_route.get('smoke_best_retained_score', '')}` AUC={source_theory_route.get('smoke_best_retained_AUC', '')} precision@20={source_theory_route.get('smoke_best_retained_precision_at_top20', '')}",
        md_table(source_theory_summary, ["score", "label", "positive_rows", "AUC", "precision_at_top20", "recall_at_top20", "control_equivalent_fraction", "direction_allowed", "source_theory_pass", "limited_smoke_selection", "blocker"], 36),
        "",
        md_table(source_theory_top, ["rank", "selection_score", "selection_score_value", "job_order", "v21_id", "dataset", "future_audit_source_h100", "future_audit_source_h400", "future_audit_source_h800", "future_audit_source_h3200", "retained_h800_h3200_positive", "official_early_chain_h100_h400_h800_positive"], 16),
        "",
        "### C3 Semantic Target Repair",
        "",
        f"- C3 semantic target repair route: `{c3_semantic_route.get('route', '')}`",
        f"- C1_semantic_pass_rows / C3_semantic_early_pass_rows / h800 positive rows: {c3_semantic_route.get('C1_semantic_pass_rows', '')}/{c3_semantic_route.get('C3_semantic_early_pass_rows', '')}/{c3_semantic_route.get('row_positive_count_h800', '')}",
        md_table(c3_semantic_c1, ["run_kind", "v21_id", "B2_transfer_gain", "B3_safety_gain", "random_target_gap", "sign_flip_gap", "corrupt_gap", "agreement_density", "corrupt_removed_norm", "target_role", "C1_semantic_pass", "official_C2_solver_claimed", "blocker"], 24),
        "",
        md_table(c3_semantic_summary, ["run_kind", "v21_id", "source_vs_best_control_h100", "source_vs_best_control_h400", "source_vs_best_control_h800", "source_vs_best_control_h1600", "source_vs_best_control_h3200", "target_injection_norm_sum", "target_refresh_count", "agreement_density_mean", "corrupt_removed_norm_mean", "C1_semantic_pass", "C3_semantic_early_pass", "C3_semantic_full_pass", "official_C2_solver_claimed", "official_C3_pass", "blocker"], 32),
        "",
        "### C3 Preservation Repair",
        "",
        f"- C3 preservation repair route: `{c3_preservation_route.get('route', '')}`",
        f"- rows / early / full: {c3_preservation_route.get('C3_preservation_rows', '')}/{c3_preservation_route.get('C3_preservation_early_pass_rows', '')}/{c3_preservation_route.get('C3_preservation_full_pass_rows', '')}",
        f"- h100/h400/h800/h1600/h3200 positive rows: {c3_preservation_route.get('row_positive_count_h100', '')}/{c3_preservation_route.get('row_positive_count_h400', '')}/{c3_preservation_route.get('row_positive_count_h800', '')}/{c3_preservation_route.get('row_positive_count_h1600', '')}/{c3_preservation_route.get('row_positive_count_h3200', '')}",
        f"- official_C2_solver_claimed / promotion_allowed: {c3_preservation_route.get('official_C2_solver_claimed', '')}/{c3_preservation_route.get('promotion_allowed', '')}",
        md_table(c3_preservation_summary, ["run_kind", "semantic_kind", "v21_id", "preserve_start", "source_vs_best_control_h100", "source_vs_best_control_h400", "source_vs_best_control_h800", "source_vs_best_control_h1600", "source_vs_best_control_h3200", "target_injection_norm_sum", "antiwashout_removed_norm_sum", "row_orthogonal_norm_sum", "optimizer_transport_norm_sum", "negative_projection_count", "C3_preservation_early_pass", "C3_preservation_full_pass", "official_C2_solver_claimed", "blocker"], 40),
        "",
        "### C3 Retained Metric Solver Repair",
        "",
        f"- C3 retained metric solver route: `{retained_solver_route.get('route', '')}`",
        f"- rows / early / full / official_C2_rows: {retained_solver_route.get('C3_retained_solver_rows', '')}/{retained_solver_route.get('C3_retained_solver_early_pass_rows', '')}/{retained_solver_route.get('C3_retained_solver_full_pass_rows', '')}/{retained_solver_route.get('official_C2_solver_claimed_rows', '')}",
        f"- h100/h400/h800/h1600/h3200 positive rows: {retained_solver_route.get('row_positive_count_h100', '')}/{retained_solver_route.get('row_positive_count_h400', '')}/{retained_solver_route.get('row_positive_count_h800', '')}/{retained_solver_route.get('row_positive_count_h1600', '')}/{retained_solver_route.get('row_positive_count_h3200', '')}",
        md_table(retained_solver_summary, ["run_kind", "solver_name", "commit_variant", "v21_id", "dataset", "source_vs_best_control_h100", "source_vs_best_control_h400", "source_vs_best_control_h800", "source_vs_best_control_h1600", "source_vs_best_control_h3200", "solver_count", "solver_quality_pass_count", "solver_projection_residual_mean", "solver_ActuationR2_mean", "solver_B2_transfer_gain_mean", "solver_B3_safety_gain_min", "C3_retained_solver_early_pass", "C3_retained_solver_full_pass", "official_C2_solver_claimed", "blocker"], 44),
        "",
        "### C3 Stage F/C/P Hybrid Repair",
        "",
        f"- C3 Stage F/C/P hybrid route: `{stagefcp_route.get('route', '')}`",
        f"- rows / early / full / official_C2_rows: {stagefcp_route.get('C3_stagefcp_rows', '')}/{stagefcp_route.get('C3_stagefcp_early_pass_rows', '')}/{stagefcp_route.get('C3_stagefcp_full_pass_rows', '')}/{stagefcp_route.get('official_C2_solver_claimed_rows', '')}",
        f"- h100/h400/h800/h1600/h3200 positive rows: {stagefcp_route.get('row_positive_count_h100', '')}/{stagefcp_route.get('row_positive_count_h400', '')}/{stagefcp_route.get('row_positive_count_h800', '')}/{stagefcp_route.get('row_positive_count_h1600', '')}/{stagefcp_route.get('row_positive_count_h3200', '')}",
        md_table(stagefcp_summary, ["run_kind", "semantic_kind", "solver_name", "switch_step", "v21_id", "source_vs_best_control_h100", "source_vs_best_control_h400", "source_vs_best_control_h800", "source_vs_best_control_h1600", "source_vs_best_control_h3200", "semantic_injection_norm_sum", "solver_update_norm_sum", "solver_count", "solver_quality_pass_count", "solver_projection_residual_mean", "solver_ActuationR2_mean", "solver_B2_transfer_gain_mean", "C3_stagefcp_early_pass", "C3_stagefcp_full_pass", "official_C2_solver_claimed", "blocker"], 36),
        "",
        "### C3 Observer2 Source Theory",
        "",
        f"- C3 Observer2 source theory route: `{observer2_route.get('route', '')}`",
        f"- C1 pass / rows / early / full: {observer2_route.get('C1_observer2_pass_rows', '')}/{observer2_route.get('C3_observer2_rows', '')}/{observer2_route.get('C3_observer2_early_pass_rows', '')}/{observer2_route.get('C3_observer2_full_pass_rows', '')}",
        f"- h100/h400/h800/h1600/h3200 positive rows: {observer2_route.get('row_positive_count_h100', '')}/{observer2_route.get('row_positive_count_h400', '')}/{observer2_route.get('row_positive_count_h800', '')}/{observer2_route.get('row_positive_count_h1600', '')}/{observer2_route.get('row_positive_count_h3200', '')}",
        md_table(observer2_c1, ["run_kind", "v21_id", "dataset", "target_norm", "observer_density", "observer_snr_mean", "margin_band_fraction", "B2_transfer_gain", "B3_safety_gain", "random_target_gap", "sign_flip_gap", "corrupt_gap", "C1_observer2_pass", "official_C2_solver_claimed", "blocker"], 24),
        "",
        md_table(observer2_summary, ["run_kind", "v21_id", "dataset", "source_vs_best_control_h100", "source_vs_best_control_h400", "source_vs_best_control_h800", "source_vs_best_control_h1600", "source_vs_best_control_h3200", "target_norm_mean", "injection_norm_sum", "observer_density_mean", "corrupt_removed_norm_mean", "C1_observer2_pass", "C3_observer2_early_pass", "C3_observer2_full_pass", "official_C2_solver_claimed", "blocker"], 24),
        "",
        "### C3 Observer3 Optimizer-Flow Theory",
        "",
        f"- C3 Observer3 optimizer-flow theory route: `{observer3_route.get('route', '')}`",
        f"- C1 pass / rows / early / full: {observer3_route.get('C1_observer3_pass_rows', '')}/{observer3_route.get('C3_observer3_rows', '')}/{observer3_route.get('C3_observer3_early_pass_rows', '')}/{observer3_route.get('C3_observer3_full_pass_rows', '')}",
        f"- h100/h400/h800/h1600/h3200 positive rows: {observer3_route.get('row_positive_count_h100', '')}/{observer3_route.get('row_positive_count_h400', '')}/{observer3_route.get('row_positive_count_h800', '')}/{observer3_route.get('row_positive_count_h1600', '')}/{observer3_route.get('row_positive_count_h3200', '')}",
        md_table(observer3_c1, ["run_kind", "v21_id", "dataset", "target_norm", "flow_steps", "flow_consensus_density", "B2_transfer_gain", "B3_safety_gain", "random_target_gap", "sign_flip_gap", "corrupt_gap", "C1_observer3_pass", "official_C2_solver_claimed", "blocker"], 24),
        "",
        md_table(observer3_summary, ["run_kind", "v21_id", "dataset", "source_vs_best_control_h100", "source_vs_best_control_h400", "source_vs_best_control_h800", "source_vs_best_control_h1600", "source_vs_best_control_h3200", "target_norm_mean", "injection_norm_sum", "flow_consensus_density_mean", "corrupt_removed_norm_mean", "C1_observer3_pass", "C3_observer3_early_pass", "C3_observer3_full_pass", "official_C2_solver_claimed", "blocker"], 24),
        "",
        "### C3 Observer4 Transfer Meta-Gradient",
        "",
        f"- C3 Observer4 transfer meta-gradient route: `{observer4_route.get('route', '')}`",
        f"- C1 pass / rows / early / full: {observer4_route.get('C1_observer4_pass_rows', '')}/{observer4_route.get('C3_observer4_rows', '')}/{observer4_route.get('C3_observer4_early_pass_rows', '')}/{observer4_route.get('C3_observer4_full_pass_rows', '')}",
        f"- h100/h400/h800/h1600/h3200 positive rows: {observer4_route.get('row_positive_count_h100', '')}/{observer4_route.get('row_positive_count_h400', '')}/{observer4_route.get('row_positive_count_h800', '')}/{observer4_route.get('row_positive_count_h1600', '')}/{observer4_route.get('row_positive_count_h3200', '')}",
        md_table(observer4_c1, ["run_kind", "v21_id", "dataset", "target_norm", "inner_lr", "transfer_agreement_density", "B2_transfer_gain", "B3_safety_gain", "random_target_gap", "sign_flip_gap", "corrupt_gap", "C1_observer4_pass", "official_C2_solver_claimed", "blocker"], 24),
        "",
        md_table(observer4_summary, ["run_kind", "v21_id", "dataset", "source_vs_best_control_h100", "source_vs_best_control_h400", "source_vs_best_control_h800", "source_vs_best_control_h1600", "source_vs_best_control_h3200", "target_norm_mean", "injection_norm_sum", "transfer_agreement_density_mean", "corrupt_removed_norm_mean", "C1_observer4_pass", "C3_observer4_early_pass", "C3_observer4_full_pass", "official_C2_solver_claimed", "blocker"], 24),
        "",
        "### C3 Observer5 Activation-Manifold Theory",
        "",
        f"- C3 Observer5 activation-manifold route: `{observer5_route.get('route', '')}`",
        f"- C1 pass / rows / early / full: {observer5_route.get('C1_observer5_pass_rows', '')}/{observer5_route.get('C3_observer5_rows', '')}/{observer5_route.get('C3_observer5_early_pass_rows', '')}/{observer5_route.get('C3_observer5_full_pass_rows', '')}",
        f"- h100/h400/h800/h1600/h3200 positive rows: {observer5_route.get('row_positive_count_h100', '')}/{observer5_route.get('row_positive_count_h400', '')}/{observer5_route.get('row_positive_count_h800', '')}/{observer5_route.get('row_positive_count_h1600', '')}/{observer5_route.get('row_positive_count_h3200', '')}",
        md_table(observer5_c1, ["run_kind", "v21_id", "dataset", "target_norm", "manifold_consensus_density", "intrinsic_dimension", "info_volume", "condition_number_of_source_subspace", "neighbor_preservation", "signal_channel_projection", "reservoir_projection", "B2_transfer_gain", "B3_safety_gain", "random_target_gap", "sign_flip_gap", "corrupt_gap", "C1_observer5_pass", "official_C2_solver_claimed", "blocker"], 24),
        "",
        md_table(observer5_summary, ["run_kind", "v21_id", "dataset", "source_vs_best_control_h100", "source_vs_best_control_h400", "source_vs_best_control_h800", "source_vs_best_control_h1600", "source_vs_best_control_h3200", "target_norm_mean", "injection_norm_sum", "manifold_consensus_density_mean", "intrinsic_dimension_mean", "info_volume_mean", "neighbor_preservation_mean", "signal_channel_projection_mean", "reservoir_projection_mean", "C1_observer5_pass", "C3_observer5_early_pass", "C3_observer5_full_pass", "official_C2_solver_claimed", "blocker"], 24),
        "",
        "### C3 Observer6 Curvature-Guard Theory",
        "",
        f"- C3 Observer6 curvature-guard route: `{observer6_route.get('route', '')}`",
        f"- C1 pass / rows / early / full: {observer6_route.get('C1_observer6_pass_rows', '')}/{observer6_route.get('C3_observer6_rows', '')}/{observer6_route.get('C3_observer6_early_pass_rows', '')}/{observer6_route.get('C3_observer6_full_pass_rows', '')}",
        f"- h100/h400/h800/h1600/h3200 positive rows: {observer6_route.get('row_positive_count_h100', '')}/{observer6_route.get('row_positive_count_h400', '')}/{observer6_route.get('row_positive_count_h800', '')}/{observer6_route.get('row_positive_count_h1600', '')}/{observer6_route.get('row_positive_count_h3200', '')}",
        md_table(observer6_c1, ["run_kind", "v21_id", "dataset", "target_norm", "raw_NDS", "guarded_NDS", "curvature_removed_norm", "hvp_norm", "B2_transfer_gain", "B3_safety_gain", "random_target_gap", "sign_flip_gap", "corrupt_gap", "C1_observer6_pass", "official_C2_solver_claimed", "blocker"], 24),
        "",
        md_table(observer6_summary, ["run_kind", "v21_id", "dataset", "source_vs_best_control_h100", "source_vs_best_control_h400", "source_vs_best_control_h800", "source_vs_best_control_h1600", "source_vs_best_control_h3200", "target_norm_mean", "injection_norm_sum", "raw_NDS_mean", "guarded_NDS_mean", "curvature_removed_norm_mean", "C1_observer6_pass", "C3_observer6_early_pass", "C3_observer6_full_pass", "official_C2_solver_claimed", "blocker"], 24),
        "",
        "### C3 Observer7 Control-Nullspace Theory",
        "",
        f"- C3 Observer7 control-nullspace route: `{observer7_route.get('route', '')}`",
        f"- C1 pass / rows / early / full: {observer7_route.get('C1_observer7_pass_rows', '')}/{observer7_route.get('C3_observer7_rows', '')}/{observer7_route.get('C3_observer7_early_pass_rows', '')}/{observer7_route.get('C3_observer7_full_pass_rows', '')}",
        f"- h100/h400/h800/h1600/h3200 positive rows: {observer7_route.get('row_positive_count_h100', '')}/{observer7_route.get('row_positive_count_h400', '')}/{observer7_route.get('row_positive_count_h800', '')}/{observer7_route.get('row_positive_count_h1600', '')}/{observer7_route.get('row_positive_count_h3200', '')}",
        md_table(observer7_c1, ["run_kind", "v21_id", "dataset", "target_norm", "control_basis_rank", "control_projection_removed_norm", "control_null_residual_fraction", "max_control_cosine", "ce_cosine", "adam_flow_cosine", "sgd_flow_cosine", "B2_transfer_gain", "B3_safety_gain", "random_target_gap", "sign_flip_gap", "corrupt_gap", "C1_observer7_pass", "official_C2_solver_claimed", "blocker"], 24),
        "",
        md_table(observer7_summary, ["run_kind", "v21_id", "dataset", "source_vs_best_control_h100", "source_vs_best_control_h400", "source_vs_best_control_h800", "source_vs_best_control_h1600", "source_vs_best_control_h3200", "target_norm_mean", "injection_norm_sum", "control_null_residual_fraction_mean", "control_projection_removed_norm_mean", "C1_observer7_pass", "C3_observer7_early_pass", "C3_observer7_full_pass", "official_C2_solver_claimed", "blocker"], 24),
        "",
        "### Source-Observer Local No-Go Boundary",
        "",
        f"- route: `{source_observer_nogo_route.get('route', '')}`",
        f"- scope: `{source_observer_nogo_route.get('scope', '')}`; universal_scientific_no_go_claimed={source_observer_nogo_route.get('universal_scientific_no_go_claimed', '')}",
        f"- observer families / rows / early / full: {source_observer_nogo_route.get('observer_family_count', '')}/{source_observer_nogo_route.get('observer_rows', '')}/{source_observer_nogo_route.get('observer_early_pass_rows', '')}/{source_observer_nogo_route.get('observer_full_pass_rows', '')}",
        f"- observer C1 pass / h1600 positive / h3200 positive: {source_observer_nogo_route.get('observer_c1_pass_rows', '')}/{source_observer_nogo_route.get('observer_h1600_positive_rows', '')}/{source_observer_nogo_route.get('observer_h3200_positive_rows', '')}",
        f"- best observer C1 B2 / best observer h3200: {source_observer_nogo_route.get('best_observer_C1_B2_transfer_gain', '')}/{source_observer_nogo_route.get('best_observer_h3200', '')}",
        md_table(source_observer_nogo_summary, ["family", "route", "rows", "c1_pass_rows", "best_C1_B2_transfer_gain", "early_pass_rows", "full_pass_rows", "h1600_positive_rows", "h3200_positive_rows", "best_h800", "best_h1600", "best_h3200", "family_boundary_failed"], 20),
        "",
        "## Part D Terminal Preservation / Part E KAN Mapping",
        "",
        md_table(terminal_rows, ["v21_id", "source_h3200", "source_h4800", "R4800_over_3200", "C5_productive_terminal_source", "preservation_status", "blocker"], 20),
        "",
        md_table(kan_rows, ["mapping_status", "blocker", "best_metric_v21_id", "KAN_source_channel_decision", "KAN_specific_delta_vs_MLP_same_metric"], 20),
        "",
        "## 4GPU Queue",
        "",
        md_table(queue_rows, ["task_id", "line", "gpu", "command", "status", "start_time", "end_time", "duration_sec"], 20),
        "",
        md_table(idle_rows, ["execution_contract_violation", "reason", "max_idle_gap_sec"], 10),
        "",
        "## 修改记录",
        "",
        "- 新增 `experiments/run_v22_07_common.py`：独立 v22.07 结果目录、执行日志、复盘、packet/bundle 构建与 artifact index。",
        "- 新增 `experiments/run_v22_07_s014_truth_gate.py`：复用底层 truth tests，但输出 v22.07 S0.14 code/semantic/solver gate。",
        "- 新增 `experiments/run_v22_07_efficiency_reconfirm.py`：从 v22.04 full-loop 实测 artifact 读回 D-CHE/D-FOU ratio；缺失的新 v22.07 timing 字段保留为空，不补造。",
        "- 新增 `experiments/run_v22_07_drat_drbf_multibatch.py`：本轮实测 D-RAT/D-RBF batch 128/256/512/1024 official trainpath，并按 3/4 robust gate 汇总。",
        "- 更新 `experiments/run_v22_07_drat_drbf_multibatch.py`：当 multibatch official closure blocked 时，按计划调用已有 active-repair 微基准尝试 numerator/denominator/reciprocal、telemetry-free、local-K、exp approximation 等修复方向；结果只写成 micro-kernel repair evidence，不写成 official fused closure。",
        "- 新增 `experiments/run_v22_07_metric_dynamics_fu.py`：按 C0/C1/C2/C3 生成 no-commit source-estimator、target contrast、metric solver 和 source formation 矩阵；C0/C1 不提交参数，future source 只作 audit label。",
        "- 新增 `experiments/run_v22_07_terminal_preservation.py` 与 `experiments/run_v22_07_kan_source_mapping.py`：按 C3/D1 gate fail-closed，不强行启动 h4800/KAN。",
        "- 新增 `experiments/run_v22_07_finalize.py` 与 `experiments/run_v22_07_full.py`：生成队列记录、复盘、zip 和 clean unzip self-test。",
        "- 新增 `experiments/run_v22_07_source_observability_repair.py`：按计划尝试 class/dataset invariant z-score、control-gap、B3 safety、reservoir removal、hidden/readout separation，并对 top repaired C0 candidates 重建 no-commit target contrast；future source 只作 audit label。",
        "- 新增 `experiments/run_v22_07_c3_repair_smoke.py`：对 C2 repaired pass 的 top candidates 运行受限 horizon smoke；全量 top9 h3200 因每步重算 metric solver 超时被记录为 runtime blocker，随后跑 top2 h800 smoke。",
        "- 新增 `experiments/run_v22_07_c3_integration_repair.py`：按计划的推荐方向继续尝试 target-to-training-dynamics 集成修复；包含 gradient transport cap、short/long EMA、low-NDS orthogonal 和 post-AdamW slow anchor，并与 AdamW/SGD/NoOp/random matched controls 对比。",
        "- 新增并更新 `experiments/run_v22_07_c3_semantic_target_repair.py`：在 integration repair blocked 后尝试 train-only split-consensus、class-balanced、corrupt-reject、readout-channel、contrastive-minus-corrupt 与 direct semantic residual targets；该脚本明确 `official_C2_solver_claimed=0`，不把 semantic smoke 写成 true metric solver pass。",
        "- 新增 `experiments/run_v22_07_source_theory_repair.py`：把 observer 目标从 any h800 positive 改为 retained h800+h3200 positive；新增 `Q5_moderate_solver_feasible_retention` 非单调 observer，并只允许 smoke selection，不写成 strict source-theory pass。",
        "- 更新 `experiments/run_v22_07_c3_semantic_target_repair.py`：支持 `--selection-file`、补全执行日志中的 `--top-k/--steps/--target-scale/--target-refresh`，并用 source-theory top candidates 跑 h800/h3200 semantic smoke。",
        "- 新增并更新 `experiments/run_v22_07_c3_preservation_repair.py`：在 semantic early source 打开但 h1600/h3200 washout 后，按计划尝试 source-preserving projection、reflect opposing gradient、source floor、dual EMA、row-orthogonal source update、source-aligned AdamW state transport，以及 h400/h800 preservation-start grid；该脚本明确 `official_C2_solver_claimed=0`，不把 preservation smoke 写成 official C3 pass。",
        "- 新增 `experiments/run_v22_07_retained_metric_solver_repair.py`：在 source-theory strict gate 未开但允许 limited smoke 后，使用 train-only observer top candidates 运行 audited v22.06 metric readout solver；补跑 top-k=4 覆盖 rank 3 retained candidates，并记录 direct/slowmix、solver quality、official_C2 claim gate。",
        "- 新增 `experiments/run_v22_07_stagefcp_hybrid_repair.py`：按计划的 Stage F/C/P 解释，把 semantic target 用作 formation，随后切到 audited v22.06 metric solver 做 consolidation/preservation；记录 switch_step、semantic injection、solver update 与 official C2/C3 gate。",
        "- 新增 `experiments/run_v22_07_observer2_source_theory.py`：在 retained metric solver 和 Stage F/C/P 均 blocked 后，尝试新的 train-only source observability theory：split-gradient persistence/SNR、corrupt-gradient rejection、train-margin band、row-block balance、long-memory carry。先跑 target_scale=1.0，因 C1 gain 过小再做一次 bounded rescale 到 16.0；两次命令均写入执行日志，最终结果以 scale=16.0 版本为准。",
        "- 新增 `experiments/run_v22_07_observer3_flow_theory.py`：在 Observer2 仍 blocked 后，尝试新的 train-only optimizer-flow forecast theory：clone 当前模型，短程 AdamW train-only unroll，取参数位移作为 gradient-like target，并测试 corrupt-flow rejection、split-flow consensus、row balance。该路径仍不使用 validation/test/future 生成方向，也不声明 official C2。",
        "- 新增 `experiments/run_v22_07_observer4_transfer_meta.py`：在 Observer3 blocked 后，尝试 train-only transfer meta-gradient：B1 virtual step 后读取 B2/B3 gradient，测试 B2/B3 consensus、corrupt rejection、conflict-free 与 row balance。先跑 inner_lr=0.003，因打开 early chain 但 h1600/h3200 washout，再做一次 bounded inner_lr=0.03 target-construction 修复；两次命令均写入执行日志，最终结果以 inner_lr=0.03 版本为准。",
        "- 新增并更新 `experiments/run_v22_07_observer5_activation_manifold.py`：在 Observer4 仍 h3200 washout 后，按计划 2.5/2.6 换成 train-only activation-manifold source observability：centroid margin、information-volume floor、neighbor preservation、consensus 与 row-balanced unfold targets，并记录 intrinsic dimension、info volume、condition number、neighbor preservation、fold/local-stretch、signal/reservoir projection；首轮 C1/h3200 blocked 后，又按 target construction / reservoir removal 方向增加 hidden/readout channel repair variants。该路径仍不使用 validation/test/future 生成方向，也不声明 official C2。",
        "- 新增 `experiments/run_v22_07_observer6_curvature_guard.py`：在 Observer5 仍 h3200 washout 后，按计划 2.1 的 Muon/NDS 假设尝试 train-only HVP curvature guard：对 activation target 估计 Hessian-vector product，移除高曲率分量，并测试 low-NDS、long EMA 与 preserve800 两时间尺度；记录 raw_NDS、guarded_NDS、curvature_removed_norm、HVP norm 与 h100-h3200 source。该路径仍不使用 validation/test/future 生成方向，也不声明 official C2。",
        "- 新增 `experiments/run_v22_07_observer7_control_nullspace.py`：在 Observer6 仍 blocked 后，尝试 train-only control-nullspace source target；将 CE gradient、短程 AdamW flow、短程 SGD flow 与 corrupt-label gradient 作为控制子空间并投影掉，再测试 neighbor/centroid/CE residual target。记录 control_basis_rank、projection_removed_norm、control_null_residual_fraction、control cosines 和 h100-h3200 source；该路径仍不使用 validation/test/future 生成方向，也不声明 official C2。",
        "- 新增 `experiments/run_v22_07_source_observer_nogo.py`：只读汇总 semantic/preservation/retained-solver/stage-FCP 与 Observer2-7 的 C1/C3 结果，按计划 12.2 生成 local source-observer no-go boundary。该脚本声明 scope=attempted_train_only_source_observer_families_only、universal_scientific_no_go_claimed=0，不把局部失败写成普遍科学不可能。",
        "- 本轮没有改动 `dgkan/fu/mechanisms.py` 或核心 FU 算法；v22.07 是 MetricDynamics 审计和门控升级，不把它写成新 mechanism 成功。",
        "",
        "## 分析 / Insight / 证据链",
        "",
        "- S0.14 使用 clean unzip source tree 再运行 truth gate；如果 clean unzip 失败，route 会停在 code/packet blocker。",
        "- C0 的 no-commit rows 由初始 MLP、同一 train batch 和 `make_update` 生成；`future_audit_source_h*` 来自 v22.06 fresh source-retention artifact，只用于 AUC/Spearman/precision audit。",
        f"- 当前 best C0 estimator 是 `{best_c0.get('estimator', '')}`，AUC_h800={best_c0.get('AUC_predict_h800_positive', '')}；若 C0_pass=0，说明这些 train-only estimator 仍不足以可靠预测 early source。",
        "- C1 target contrast 显式比较 random matched target、sign-flip target 和 corrupt-label target；若 `C1_pass_rows` 很少或为 0，不能继续把问题归咎于 solver。",
        "- C2 分开记录 solver quality 与 make_update wall time；即使 ActuationR2/projection residual 好，也不等价于 source formation。",
        f"- 当前 best C3 row `{best.get('v21_id', '')}` 的 h800={best.get('source_vs_best_control_h800', '')}、h3200={best.get('source_vs_best_control_h3200', '')}；C3 gate={best.get('C3_source_formation_pass', '')}，所以 terminal/KAN gate 的进入状态必须按此判定。",
        "- D-CHE/D-FOU 是历史 full-loop measured artifact readback；functional_direction_ms/metric_solver_ms/backward/no_materialize 等 v22.07 新字段若为空，复盘中保留为空并作为证据缺口，而不是估计填充。",
        "- D-RAT/D-RBF multibatch status 不只看 single batch pass；若 component telemetry incomplete 或 robust rows <3/4，会记录为 closure blocker，并列出计划中的 repair direction。",
        "- D-RAT/D-RBF repair attempt 若出现 `MicroNearE1RunnerBlocked`，只能说明微基准方向有性能/数值线索；由于 `official_closure_claimed=0` 且 runner/kernel mismatch 仍在，不能提升为 robust official status。",
        "- 如果本轮仍停在 C1/C2 有进展但 C3 early source fail，结论是当前 train-only source estimator/target family 不足，需要新的 source observability theory，而不是继续堆更多 metric solver 名字。",
        f"- 本轮继续推进后，C0 repair 打开：best estimator `{source_repair_route.get('best_C0_repair_estimator', '')}`，AUC_h800={source_repair_route.get('best_C0_repair_AUC_h800', '')}，precision@20={source_repair_route.get('best_C0_repair_precision_top20', '')}；该方向使用 target-control gap，不使用 future source 生成方向。",
        "- C1/C2 repair 打开主要依赖 `all_x64` target rescale：这说明 no-commit / one-step actuation 可以被 target magnitude 打开，但这不是 source formation 证据。",
        f"- C3 repair smoke 结果为 `{c3_repair_route.get('route', '')}`；top2 到 h800 后 row_positive_count_h800={c3_repair_route.get('row_positive_count_h800', '')}，说明 target rescale 的 actuation 在训练动力系统中被 controls 超过或洗掉。",
        f"- C3 integration repair 结果为 `{c3_integration_route.get('route', '')}`；8 个 source/control 集成变体中 early pass rows={c3_integration_route.get('C3_integration_early_pass_rows', '')}，h800 positive rows={c3_integration_route.get('row_positive_count_h800', '')}。conflict projection、short/long EMA、low-NDS orthogonal、post-AdamW slow anchor 都没有把 h800 source-vs-control 打正，说明 blocker 不只是 target magnitude，而是 target-control-gap 信号没有稳定穿过 AdamW/SGD 训练动力学。",
        f"- Source theory repair 结果为 `{source_theory_route.get('route', '')}`；历史 204 行中 official h100+h400+h800 positive rows={source_theory_route.get('official_early_chain_h100_h400_h800_positive', '')}，retained h800+h3200 positive rows={source_theory_route.get('retained_h800_h3200_positive', '')}。`Q5_moderate_solver_feasible_retention` 对 retained label 的 AUC={source_theory_route.get('smoke_best_retained_AUC', '')}、precision@20={source_theory_route.get('smoke_best_retained_precision_at_top20', '')}，只够 smoke selection，不够 strict source-theory pass。",
        f"- C3 semantic target extended smoke 结果为 `{c3_semantic_route.get('route', '')}`；C3_semantic_early_pass_rows={c3_semantic_route.get('C3_semantic_early_pass_rows', '')}，但 h1600 positive rows={route.get('C3_semantic_h1600_positive_rows', '')}、h3200 positive rows={route.get('C3_semantic_h3200_positive_rows', '')}、C3_semantic_full_pass_rows={c3_semantic_route.get('C3_semantic_full_pass_rows', '')}。best h3200 仍为负，说明这次打开的是 transient early source，不是 retained source。",
        f"- C3 preservation repair 结果为 `{c3_preservation_route.get('route', '')}`；summary rows={c3_preservation_route.get('C3_preservation_rows', '')}，early pass rows={c3_preservation_route.get('C3_preservation_early_pass_rows', '')}，full pass rows={c3_preservation_route.get('C3_preservation_full_pass_rows', '')}，h1600 positive rows={c3_preservation_route.get('row_positive_count_h1600', '')}，h3200 positive rows={c3_preservation_route.get('row_positive_count_h3200', '')}。该轮实际尝试了 h800 preservation start 与 h400 consolidation start；best h1600={best_pres_h1600.get('source_vs_best_control_h1600', '')} (`{best_pres_h1600.get('run_kind', '')}` preserve_start={best_pres_h1600.get('preserve_start', '')})，best h3200={best_pres_h3200.get('source_vs_best_control_h3200', '')} (`{best_pres_h3200.get('run_kind', '')}` preserve_start={best_pres_h3200.get('preserve_start', '')})，仍为负，说明 projection、row-orthogonal、dual-state、source-aligned optimizer transport 这组同族 preservation 修复没有解决 long-horizon retention。",
        f"- Retained metric solver repair 结果为 `{retained_solver_route.get('route', '')}`；top-k=4 后 rows={retained_solver_route.get('C3_retained_solver_rows', '')}，early pass rows={retained_solver_route.get('C3_retained_solver_early_pass_rows', '')}，full pass rows={retained_solver_route.get('C3_retained_solver_full_pass_rows', '')}，official_C2 rows={retained_solver_route.get('official_C2_solver_claimed_rows', '')}，h1600/h3200 positive rows={retained_solver_route.get('row_positive_count_h1600', '')}/{retained_solver_route.get('row_positive_count_h3200', '')}。best h1600={best_ret_h1600.get('source_vs_best_control_h1600', '')} (`{best_ret_h1600.get('run_kind', '')}`)，best h3200={best_ret_h3200.get('source_vs_best_control_h3200', '')} (`{best_ret_h3200.get('run_kind', '')}`)，仍为负；同时 direct solver 也没有满足 repeated official C2 quality，所以不能把 smoke 写成 official solver success。",
        f"- Stage F/C/P hybrid repair 结果为 `{stagefcp_route.get('route', '')}`；rows={stagefcp_route.get('C3_stagefcp_rows', '')}，early pass rows={stagefcp_route.get('C3_stagefcp_early_pass_rows', '')}，full pass rows={stagefcp_route.get('C3_stagefcp_full_pass_rows', '')}，official_C2 rows={stagefcp_route.get('official_C2_solver_claimed_rows', '')}，h1600/h3200 positive rows={stagefcp_route.get('row_positive_count_h1600', '')}/{stagefcp_route.get('row_positive_count_h3200', '')}。best h1600={best_stage_h1600.get('source_vs_best_control_h1600', '')} (`{best_stage_h1600.get('run_kind', '')}`)，best h3200={best_stage_h3200.get('source_vs_best_control_h3200', '')} (`{best_stage_h3200.get('run_kind', '')}`)，说明 formation/consolidation/preservation 分阶段实现仍未阻止 h1600/h3200 washout。",
        f"- Observer2 source theory 结果为 `{observer2_route.get('route', '')}`；C1 pass rows={observer2_route.get('C1_observer2_pass_rows', '')}，summary rows={observer2_route.get('C3_observer2_rows', '')}，early/full={observer2_route.get('C3_observer2_early_pass_rows', '')}/{observer2_route.get('C3_observer2_full_pass_rows', '')}，h1600/h3200 positive rows={observer2_route.get('row_positive_count_h1600', '')}/{observer2_route.get('row_positive_count_h3200', '')}。bounded rescale 到 16.0 后 best C1 B2 仍只有 0.000759 量级，未达到 +0.005 C1 gate；best h1600={best_obs_h1600.get('source_vs_best_control_h1600', '')} (`{best_obs_h1600.get('run_kind', '')}`)，best h3200={best_obs_h3200.get('source_vs_best_control_h3200', '')} (`{best_obs_h3200.get('run_kind', '')}`)，仍为负。",
        f"- Observer3 optimizer-flow theory 结果为 `{observer3_route.get('route', '')}`；C1 pass rows={observer3_route.get('C1_observer3_pass_rows', '')}，summary rows={observer3_route.get('C3_observer3_rows', '')}，early/full={observer3_route.get('C3_observer3_early_pass_rows', '')}/{observer3_route.get('C3_observer3_full_pass_rows', '')}，h1600/h3200 positive rows={observer3_route.get('row_positive_count_h1600', '')}/{observer3_route.get('row_positive_count_h3200', '')}。best C1 B2={max([finite_float(r.get('B2_transfer_gain'), -999.0) for r in observer3_c1], default='')} 仍低于 +0.005 gate；best h1600={best_flow_h1600.get('source_vs_best_control_h1600', '')} (`{best_flow_h1600.get('run_kind', '')}`)，best h3200={best_flow_h3200.get('source_vs_best_control_h3200', '')} (`{best_flow_h3200.get('run_kind', '')}`)，仍为负。split-flow consensus 只给出 h800=0.006283 的局部正值，h100/h400/h1600/h3200 均未成链。",
        f"- Observer4 transfer meta-gradient 结果为 `{observer4_route.get('route', '')}`；C1 pass rows={observer4_route.get('C1_observer4_pass_rows', '')}，summary rows={observer4_route.get('C3_observer4_rows', '')}，early/full={observer4_route.get('C3_observer4_early_pass_rows', '')}/{observer4_route.get('C3_observer4_full_pass_rows', '')}，h1600/h3200 positive rows={observer4_route.get('row_positive_count_h1600', '')}/{observer4_route.get('row_positive_count_h3200', '')}。bounded inner_lr=0.03 后 best C1 B2={max([finite_float(r.get('B2_transfer_gain'), -999.0) for r in observer4_c1], default='')}，仍低于 +0.005 gate；best early row `{best_meta_h3200.get('run_kind', '')}` 打开 h100={best_meta_h3200.get('source_vs_best_control_h100', '')} / h400={best_meta_h3200.get('source_vs_best_control_h400', '')} / h800={best_meta_h3200.get('source_vs_best_control_h800', '')}，但 h1600={best_meta_h1600.get('source_vs_best_control_h1600', '')}、h3200={best_meta_h3200.get('source_vs_best_control_h3200', '')} 仍为负。",
        f"- Observer5 activation-manifold theory 结果为 `{observer5_route.get('route', '')}`；C1 pass rows={observer5_route.get('C1_observer5_pass_rows', '')}，summary rows={observer5_route.get('C3_observer5_rows', '')}，early/full={observer5_route.get('C3_observer5_early_pass_rows', '')}/{observer5_route.get('C3_observer5_full_pass_rows', '')}，h1600/h3200 positive rows={observer5_route.get('row_positive_count_h1600', '')}/{observer5_route.get('row_positive_count_h3200', '')}。best C1 B2={max([finite_float(r.get('B2_transfer_gain'), -999.0) for r in observer5_c1], default='')}；best h1600={best_mani_h1600.get('source_vs_best_control_h1600', '')} (`{best_mani_h1600.get('run_kind', '')}`)，best h3200={best_mani_h3200.get('source_vs_best_control_h3200', '')} (`{best_mani_h3200.get('run_kind', '')}`)。该路径把计划中的 info-volume / neighbor / signal-channel 证据落盘，但仍按 gate 判定，不把 representation smoke 写成 official source。",
        f"- Observer6 curvature-guard theory 结果为 `{observer6_route.get('route', '')}`；C1 pass rows={observer6_route.get('C1_observer6_pass_rows', '')}，summary rows={observer6_route.get('C3_observer6_rows', '')}，early/full={observer6_route.get('C3_observer6_early_pass_rows', '')}/{observer6_route.get('C3_observer6_full_pass_rows', '')}，h1600/h3200 positive rows={observer6_route.get('row_positive_count_h1600', '')}/{observer6_route.get('row_positive_count_h3200', '')}。best C1 B2={max([finite_float(r.get('B2_transfer_gain'), -999.0) for r in observer6_c1], default='')}；best h1600={best_curv_h1600.get('source_vs_best_control_h1600', '')} (`{best_curv_h1600.get('run_kind', '')}`)，best h3200={best_curv_h3200.get('source_vs_best_control_h3200', '')} (`{best_curv_h3200.get('run_kind', '')}`)。best early row raw_NDS_mean={best_curv_h3200.get('raw_NDS_mean', '')}、guarded_NDS_mean={best_curv_h3200.get('guarded_NDS_mean', '')}，说明 curvature guard 确实降低了 NDS，但 h1600/h3200 仍为负。",
        f"- Observer7 control-nullspace theory 结果为 `{observer7_route.get('route', '')}`；C1 pass rows={observer7_route.get('C1_observer7_pass_rows', '')}，summary rows={observer7_route.get('C3_observer7_rows', '')}，early/full={observer7_route.get('C3_observer7_early_pass_rows', '')}/{observer7_route.get('C3_observer7_full_pass_rows', '')}，h1600/h3200 positive rows={observer7_route.get('row_positive_count_h1600', '')}/{observer7_route.get('row_positive_count_h3200', '')}。best C1 B2={max([finite_float(r.get('B2_transfer_gain'), -999.0) for r in observer7_c1], default='')}；best h1600={best_null_h1600.get('source_vs_best_control_h1600', '')} (`{best_null_h1600.get('run_kind', '')}`)，best h3200={best_null_h3200.get('source_vs_best_control_h3200', '')} (`{best_null_h3200.get('run_kind', '')}`)。best h3200 row control_null_residual_fraction_mean={best_null_h3200.get('control_null_residual_fraction_mean', '')}、projection_removed_norm_mean={best_null_h3200.get('control_projection_removed_norm_mean', '')}；即使把普通 CE/AdamW/SGD/corrupt 控制方向投影掉，长期 source-vs-control 仍为负，说明 blocker 不只是源方向落在普通训练控制流子空间里。",
        f"- Source-observer local no-go boundary 结果为 `{source_observer_nogo_route.get('route', '')}`；scope=`{source_observer_nogo_route.get('scope', '')}`，universal_scientific_no_go_claimed={source_observer_nogo_route.get('universal_scientific_no_go_claimed', '')}。Observer2-7 合计 families={source_observer_nogo_route.get('observer_family_count', '')}、rows={source_observer_nogo_route.get('observer_rows', '')}、early pass rows={source_observer_nogo_route.get('observer_early_pass_rows', '')}，但 C1 pass rows={source_observer_nogo_route.get('observer_c1_pass_rows', '')}、full C3 rows={source_observer_nogo_route.get('observer_full_pass_rows', '')}、h1600/h3200 positive rows={source_observer_nogo_route.get('observer_h1600_positive_rows', '')}/{source_observer_nogo_route.get('observer_h3200_positive_rows', '')}。best observer C1 B2={source_observer_nogo_route.get('best_observer_C1_B2_transfer_gain', '')}，best observer h3200={source_observer_nogo_route.get('best_observer_h3200', '')}。",
        "- 因此最新结论从 Observer7 blocked 推进到 attempted train-only source-observer local no-go boundary：按计划尝试 optimizer-state integration、source preservation、retained metric solver、Stage F/C/P hybrid、Observer2、Observer3、Observer4、Observer5、Observer6、Observer7 后，仍没有 retained source，也没有 official C2 solver claim，不能进入 terminal preservation/KAN mapping promotion。",
        "- 按计划第 13 节，这时不能继续写成“再调 metric solver”；更合适的边界是当前已试过的 train-only gradient persistence、optimizer-flow、transfer-meta、activation-manifold、curvature-guard 和 control-nullspace observer families 均不足，需要新的 source observability principle 或明确 no-go。继续 KAN-FU 或 terminal preservation 会浪费 GPU 并造成错误 promotion 风险。",
        "",
        "_完整 artifact 清单保留在 `v22_07_artifact_index.csv`，不放入复盘正文。_",
    ]
    V2207_RECAP_DOC.write_text("\n".join(recap) + "\n", encoding="utf-8")
    append_exec(out_dir, f"{PYTHON} experiments/run_v22_07_finalize.py --out-dir {out_dir}", status="completed", note=f"route={route['route']} promotion={route['promotion_allowed']}")
    build_packet(out_dir)
    compat = _write_compat_bundle(out_dir)
    append_exec(out_dir, f"zip artifact refresh {compat}", status="completed", note=f"compat_bundle={compat.name} includes v22.07 code_review_packet and results")
    write_rows(out_dir / "v22_07_artifact_index.csv", artifact_index(out_dir))
    _write_compat_bundle(out_dir)


if __name__ == "__main__":
    main()
