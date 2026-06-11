#!/usr/bin/env python3
"""v22.08 final route, recap, artifact index, and results bundle."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_v22_08_common import (  # noqa: E402
    PYTHON,
    V2208_RECAP_DOC,
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


def _best(rows: list[dict[str, Any]], key: str, default: str = "") -> dict[str, Any]:
    valid = [r for r in rows if finite_float(r.get(key), float("nan")) == finite_float(r.get(key), float("nan"))]
    if not valid:
        return {}
    return max(valid, key=lambda r: finite_float(r.get(key), -1.0e99))


def _route(out_dir: Path) -> dict[str, Any]:
    code = read_json(out_dir / "v22_08_code_route_decision.json")
    eff = read_json(out_dir / "v22_08_efficiency_route.json")
    dr = read_json(out_dir / "v22_08_drat_drbf_multibatch_route.json")
    obs = read_json(out_dir / "v22_08_retained_source_observer_route.json")
    solver = read_json(out_dir / "v22_08_metric_dynamics_solver_route.json")
    terminal = read_json(out_dir / "v22_08_terminal_preservation_route.json")
    kan = read_json(out_dir / "v22_08_kan_source_mapping_route.json")
    post = read_json(out_dir / "v22_08_post_nogo_source_state_route.json")
    cow = read_json(out_dir / "v22_08_post_nogo_counterfactual_washout_route.json")
    ntk = read_json(out_dir / "v22_08_post_nogo_ntk_channel_route.json")
    aug = read_json(out_dir / "v22_08_post_nogo_aug_tangent_route.json")
    bbc = read_json(out_dir / "v22_08_post_nogo_bootstrap_basin_route.json")
    cfi = read_json(out_dir / "v22_08_post_nogo_crossfit_influence_route.json")
    tfc = read_json(out_dir / "v22_08_post_nogo_train_flow_commutator_route.json")
    tfcv = read_json(out_dir / "v22_08_train_flow_commutator_verify_route.json")
    queue = read_json(out_dir / "v22_08_queue_drain_report.json")
    idle_rows = read_rows(out_dir / "v22_08_idle_violation.csv")
    queue_violation = any(int_flag(r.get("execution_contract_violation")) for r in idle_rows)
    blockers: list[str] = []
    if not int_flag(code.get("S0_15_pass")):
        blockers.append("S0.15_code_metric_solver_gate")
    if not int_flag(eff.get("D-CHE_S1_pass")) or not int_flag(eff.get("D-FOU_S1_pass")):
        blockers.append("D-CHE_D-FOU_efficiency_reconfirm_gate")
    if not int_flag(dr.get("multibatch_status_closed")):
        blockers.append("D-RAT_D-RBF_multibatch_robust_or_telemetry")
    if not int_flag(obs.get("C0_retained_source_observer_pass_rows")):
        blockers.append("retained_source_observer_gate")
    if not int_flag(solver.get("C2_true_block_solver_pass_rows")):
        blockers.append("C2_true_block_solver_or_observer_gate")
    if not int_flag(solver.get("C3_source_formation_pass_rows")):
        blockers.append("C3_source_formation_gate")
    if str(terminal.get("decision", "")).startswith("D0D1Blocked"):
        blockers.append("terminal_preservation_not_entered_C3_source_gate")
    if str(kan.get("decision", "")) == "KANMappingNotEntered":
        blockers.append("KAN_mapping_not_entered_C3_source_gate")
    if post and not int_flag(post.get("official_C3_pass_rows")):
        blockers.append("post_nogo_C-O6_source_state_certificate_gate")
    if cow and not int_flag(cow.get("official_C3_pass_rows")):
        blockers.append("post_nogo_C-O7_counterfactual_washout_gate")
    if ntk and not int_flag(ntk.get("official_C3_pass_rows")):
        blockers.append("post_nogo_C-O8_ntk_jacobian_channel_gate")
    if aug and not int_flag(aug.get("official_C3_pass_rows")):
        blockers.append("post_nogo_C-O9_aug_tangent_gate")
    if bbc and not int_flag(bbc.get("official_C3_pass_rows")):
        blockers.append("post_nogo_C-O10_bootstrap_basin_gate")
    if cfi and not int_flag(cfi.get("official_C3_pass_rows")):
        blockers.append("post_nogo_C-O11_crossfit_influence_gate")
    if tfc and not int_flag(tfc.get("official_C3_pass_rows")):
        blockers.append("post_nogo_C-O12_train_flow_commutator_gate")
    if tfcv and not int_flag(tfcv.get("official_C3_pass_rows")):
        blockers.append("post_nogo_C-O12_repeat_verify_official_gate")
    if queue_violation:
        blockers.append("4GPU_queue_idle_violation")

    if not int_flag(code.get("S0_15_pass")):
        route = "R-CodeMetricMechanismInvalid"
        action = "fix S0.15 code/import/C2 recompute/kernel gate before scientific route"
    elif int_flag(solver.get("C3_source_formation_pass_rows")) and int_flag(terminal.get("C4_terminal_preservation_entered")):
        route = "S4-MLPTerminalRetentionOpened"
        action = "complete KAN mapping under the same source metric"
    elif int_flag(solver.get("C3_source_formation_pass_rows")):
        route = "S3-MLPSourceFormationOpened"
        action = "run terminal h4800 preservation and stable-random controls"
    elif post and int_flag(post.get("fresh_C3_source_state_pass_rows")):
        route = "R-PostNoGoSourceStateFreshSmokeOnly"
        action = "C-O6 opened fresh smoke but not official C3; formalize/verify official C2/C3 terminal gate before promotion"
    elif cow and int_flag(cow.get("fresh_C3_source_washout_pass_rows")):
        route = "R-PostNoGoCounterfactualWashoutFreshSmokeOnly"
        action = "C-O7 opened fresh smoke but not official C3; formalize/verify official C2/C3 terminal gate before promotion"
    elif ntk and int_flag(ntk.get("fresh_C3_eigen_channel_pass_rows")):
        route = "R-PostNoGoNTKChannelFreshSmokeOnly"
        action = "C-O8 opened fresh smoke but not official C3; formalize/verify official C2/C3 terminal gate before promotion"
    elif tfcv and int_flag(tfcv.get("TFC_repeat_robust_fresh_C3_verified_rows")):
        route = "R-PostNoGoTrainFlowCommutatorRobustFreshSmokeOnly"
        action = "C-O12 repeated fresh smoke verified but official observer/C2/C3 gates remain closed; do not promote without formal official C2/C3 construction"
    elif tfcv:
        route = "R-PostNoGoTrainFlowCommutatorRepeatVerifyBlocked"
        action = "C-O12 fresh smoke failed repeat verification; continue only with a genuinely new source-observability principle outside C-O1..C-O12, or record the local boundary"
    elif tfc and int_flag(tfc.get("fresh_C3_train_flow_commutator_pass_rows")):
        route = "R-PostNoGoTrainFlowCommutatorFreshSmokeOnly"
        action = "C-O12 opened fresh smoke but not official C3; formalize/verify official C2/C3 terminal gate before promotion"
    elif cfi and int_flag(cfi.get("fresh_C3_crossfit_influence_pass_rows")):
        route = "R-PostNoGoCrossFitInfluenceFreshSmokeOnly"
        action = "C-O11 opened fresh smoke but not official C3; formalize/verify official C2/C3 terminal gate before promotion"
    elif bbc and int_flag(bbc.get("fresh_C3_bootstrap_basin_pass_rows")):
        route = "R-PostNoGoBootstrapBasinFreshSmokeOnly"
        action = "C-O10 opened fresh smoke but not official C3; formalize/verify official C2/C3 terminal gate before promotion"
    elif aug and int_flag(aug.get("fresh_C3_aug_tangent_pass_rows")):
        route = "R-PostNoGoAugTangentFreshSmokeOnly"
        action = "C-O9 opened fresh smoke but not official C3; formalize/verify official C2/C3 terminal gate before promotion"
    elif not int_flag(obs.get("C0_retained_source_observer_pass_rows")):
        route = "R-ObserverLocalNoGoBoundary"
        if tfc:
            action = "C-O1..C-O12 attempted train-only source-observability local boundary reached; require external theory or explicitly record the boundary"
        elif cfi:
            action = "C-O1..C-O11 attempted train-only source-observability local boundary reached; require external theory or explicitly record the boundary"
        elif bbc:
            action = "C-O1..C-O10 attempted train-only source-observability local boundary reached; require external theory or explicitly record the boundary"
        elif aug:
            action = "C-O1..C-O9 attempted train-only source-observability local boundary reached; require external theory or explicitly record the boundary"
        elif ntk:
            action = "C-O1..C-O8 attempted train-only source-observability local boundary reached; require external theory or explicitly record the boundary"
        else:
            action = "stop same-family observer scale/cap/floor tuning; require a genuinely new train-only retained-source certificate"
    elif not int_flag(solver.get("C2_true_block_solver_pass_rows")):
        route = "R-SolverProjectionBlocked"
        action = "repair true block solver with damping/rank/CG before C3"
    elif not int_flag(dr.get("multibatch_status_closed")):
        route = "R-EfficiencyCarrierBlocked-DRAT-DRBF"
        action = "continue D-RAT/D-RBF component waterfall repair only; no full FU promotion"
    else:
        route = "R-ObserverLocalNoGoBoundary"
        action = "route fallback: no MLP source formation rows opened"
    dedup: list[str] = []
    for item in blockers:
        for part in str(item).split(";"):
            part = part.strip()
            if part and part not in dedup:
                dedup.append(part)
    return {
        "route": route,
        "promotion_allowed": 0,
        "blocking_metric": ";".join(dedup),
        "S0_15_pass": int_flag(code.get("S0_15_pass")),
        "D-CHE_S1_pass": int_flag(eff.get("D-CHE_S1_pass")),
        "D-FOU_S1_pass": int_flag(eff.get("D-FOU_S1_pass")),
        "D-RAT_D-RBF_multibatch_closed": int_flag(dr.get("multibatch_status_closed")),
        "C0_retained_source_observer_pass_rows": int_flag(obs.get("C0_retained_source_observer_pass_rows")),
        "official_early_chain_positive_rows": int_flag(obs.get("official_early_chain_positive_rows")),
        "retained_h800_h3200_positive_rows": int_flag(obs.get("retained_h800_h3200_positive_rows")),
        "C2_true_block_solver_pass_rows": int_flag(solver.get("C2_true_block_solver_pass_rows")),
        "C3_source_formation_pass_rows": int_flag(solver.get("C3_source_formation_pass_rows")),
        "post_nogo_C-O6_route": post.get("route", ""),
        "post_nogo_C-O6_train_only_pass_rows": int_flag(post.get("PSSC_train_only_certificate_pass_rows")),
        "post_nogo_C-O6_fresh_C3_pass_rows": int_flag(post.get("fresh_C3_source_state_pass_rows")),
        "post_nogo_C-O6_official_C3_pass_rows": int_flag(post.get("official_C3_pass_rows")),
        "post_nogo_C-O7_route": cow.get("route", ""),
        "post_nogo_C-O7_train_only_pass_rows": int_flag(cow.get("COW_train_only_certificate_pass_rows")),
        "post_nogo_C-O7_fresh_C3_pass_rows": int_flag(cow.get("fresh_C3_source_washout_pass_rows")),
        "post_nogo_C-O7_official_C3_pass_rows": int_flag(cow.get("official_C3_pass_rows")),
        "post_nogo_C-O8_route": ntk.get("route", ""),
        "post_nogo_C-O8_train_only_pass_rows": int_flag(ntk.get("NTKC_train_only_certificate_pass_rows")),
        "post_nogo_C-O8_fresh_C3_pass_rows": int_flag(ntk.get("fresh_C3_eigen_channel_pass_rows")),
        "post_nogo_C-O8_official_C3_pass_rows": int_flag(ntk.get("official_C3_pass_rows")),
        "post_nogo_C-O9_route": aug.get("route", ""),
        "post_nogo_C-O9_train_only_pass_rows": int_flag(aug.get("ATC_train_only_certificate_pass_rows")),
        "post_nogo_C-O9_fresh_C3_pass_rows": int_flag(aug.get("fresh_C3_aug_tangent_pass_rows")),
        "post_nogo_C-O9_official_C3_pass_rows": int_flag(aug.get("official_C3_pass_rows")),
        "post_nogo_C-O10_route": bbc.get("route", ""),
        "post_nogo_C-O10_train_only_pass_rows": int_flag(bbc.get("BBC_train_only_certificate_pass_rows")),
        "post_nogo_C-O10_fresh_C3_pass_rows": int_flag(bbc.get("fresh_C3_bootstrap_basin_pass_rows")),
        "post_nogo_C-O10_official_C3_pass_rows": int_flag(bbc.get("official_C3_pass_rows")),
        "post_nogo_C-O11_route": cfi.get("route", ""),
        "post_nogo_C-O11_train_only_pass_rows": int_flag(cfi.get("CFI_train_only_certificate_pass_rows")),
        "post_nogo_C-O11_fresh_C3_pass_rows": int_flag(cfi.get("fresh_C3_crossfit_influence_pass_rows")),
        "post_nogo_C-O11_official_C3_pass_rows": int_flag(cfi.get("official_C3_pass_rows")),
        "post_nogo_C-O12_route": tfc.get("route", ""),
        "post_nogo_C-O12_train_only_pass_rows": int_flag(tfc.get("TFC_train_only_certificate_pass_rows")),
        "post_nogo_C-O12_fresh_C3_pass_rows": int_flag(tfc.get("fresh_C3_train_flow_commutator_pass_rows")),
        "post_nogo_C-O12_official_C3_pass_rows": int_flag(tfc.get("official_C3_pass_rows")),
        "post_nogo_C-O12_verify_route": tfcv.get("route", ""),
        "post_nogo_C-O12_verify_robust_fresh_rows": int_flag(tfcv.get("TFC_repeat_robust_fresh_C3_verified_rows")),
        "post_nogo_C-O12_verify_official_C3_pass_rows": int_flag(tfcv.get("official_C3_pass_rows")),
        "terminal_preservation_decision": terminal.get("decision", ""),
        "KAN_mapping_decision": kan.get("decision", ""),
        "queue_drained": int_flag(queue.get("queue_drained")),
        "execution_contract_violation": int(queue_violation),
        "next_codex_action": action,
    }


def _write_recap(out_dir: Path, route: dict[str, Any]) -> None:
    code_rows = read_rows(out_dir / "v22_08_code_truth_gate.csv")
    c2_recompute = read_rows(out_dir / "v22_08_c2_recompute_tests.csv")
    eff_rows = read_rows(out_dir / "v22_08_efficiency_full_loop_reconfirm.csv")
    dr_summary = read_rows(out_dir / "v22_08_drat_drbf_multibatch_summary.csv")
    dr_repair = read_rows(out_dir / "v22_08_drat_drbf_repair_attempts_summary.csv")
    obs_summary = read_rows(out_dir / "v22_08_retained_source_observer_summary.csv")
    obs_top = read_rows(out_dir / "v22_08_retained_source_observer_top_candidates.csv")
    post_route = read_json(out_dir / "v22_08_post_nogo_source_state_route.json")
    post_summary = read_rows(out_dir / "v22_08_post_nogo_source_state_certificate_summary.csv")
    post_selected = read_rows(out_dir / "v22_08_post_nogo_source_state_selected_candidates.csv")
    post_fresh_summary = read_rows(out_dir / "v22_08_post_nogo_source_state_fresh_c3_summary.csv")
    cow_route = read_json(out_dir / "v22_08_post_nogo_counterfactual_washout_route.json")
    cow_summary = read_rows(out_dir / "v22_08_post_nogo_counterfactual_washout_summary.csv")
    cow_selected = read_rows(out_dir / "v22_08_post_nogo_counterfactual_washout_selected_candidates.csv")
    cow_fresh_summary = read_rows(out_dir / "v22_08_post_nogo_counterfactual_washout_fresh_c3_summary.csv")
    ntk_route = read_json(out_dir / "v22_08_post_nogo_ntk_channel_route.json")
    ntk_summary = read_rows(out_dir / "v22_08_post_nogo_ntk_channel_summary.csv")
    ntk_selected = read_rows(out_dir / "v22_08_post_nogo_ntk_channel_selected_candidates.csv")
    ntk_fresh_summary = read_rows(out_dir / "v22_08_post_nogo_ntk_channel_fresh_c3_summary.csv")
    aug_route = read_json(out_dir / "v22_08_post_nogo_aug_tangent_route.json")
    aug_summary = read_rows(out_dir / "v22_08_post_nogo_aug_tangent_summary.csv")
    aug_selected = read_rows(out_dir / "v22_08_post_nogo_aug_tangent_selected_candidates.csv")
    aug_fresh_summary = read_rows(out_dir / "v22_08_post_nogo_aug_tangent_fresh_c3_summary.csv")
    bbc_route = read_json(out_dir / "v22_08_post_nogo_bootstrap_basin_route.json")
    bbc_summary = read_rows(out_dir / "v22_08_post_nogo_bootstrap_basin_summary.csv")
    bbc_selected = read_rows(out_dir / "v22_08_post_nogo_bootstrap_basin_selected_candidates.csv")
    bbc_fresh_summary = read_rows(out_dir / "v22_08_post_nogo_bootstrap_basin_fresh_c3_summary.csv")
    cfi_route = read_json(out_dir / "v22_08_post_nogo_crossfit_influence_route.json")
    cfi_summary = read_rows(out_dir / "v22_08_post_nogo_crossfit_influence_summary.csv")
    cfi_selected = read_rows(out_dir / "v22_08_post_nogo_crossfit_influence_selected_candidates.csv")
    cfi_fresh_summary = read_rows(out_dir / "v22_08_post_nogo_crossfit_influence_fresh_c3_summary.csv")
    tfc_route = read_json(out_dir / "v22_08_post_nogo_train_flow_commutator_route.json")
    tfc_summary = read_rows(out_dir / "v22_08_post_nogo_train_flow_commutator_summary.csv")
    tfc_selected = read_rows(out_dir / "v22_08_post_nogo_train_flow_commutator_selected_candidates.csv")
    tfc_fresh_summary = read_rows(out_dir / "v22_08_post_nogo_train_flow_commutator_fresh_c3_summary.csv")
    tfcv_route = read_json(out_dir / "v22_08_train_flow_commutator_verify_route.json")
    tfcv_summary = read_rows(out_dir / "v22_08_train_flow_commutator_verify_repeat_summary.csv")
    tfcv_fresh_summary = read_rows(out_dir / "v22_08_train_flow_commutator_verify_fresh_c3_summary.csv")
    solver_summary = read_rows(out_dir / "v22_08_c2_true_block_solver_summary.csv")
    terminal = read_rows(out_dir / "v22_08_terminal_preservation_summary.csv")
    kan = read_rows(out_dir / "v22_08_kan_source_mapping_summary.csv")
    queue = read_rows(out_dir / "v22_08_gpu_assignment_manifest.csv")
    idle = read_rows(out_dir / "v22_08_idle_violation.csv")
    best_obs_retained = _best(obs_summary, "AUC_predict_retained_h800_h3200_positive")
    best_obs_early = _best(obs_summary, "AUC_predict_official_early_chain")
    best_top = obs_top[0] if obs_top else {}
    c2_scale_notes = []
    if c2_recompute:
        c2_scale_notes.append(f"rows={len(c2_recompute)}")
        c2_scale_notes.append(f"pass_rows={sum(int_flag(r.get('C2_recompute_test_pass')) for r in c2_recompute)}")
        c2_scale_notes.append(f"projection_ratio_identical_rows={sum(int_flag(r.get('scale_projection_residual_identical')) for r in c2_recompute)}")
    text = [
        "# DG-KAN v22.08 RetainedSourceDynamics FunctionalUpdate BasisEfficiency 4GPU 实验结果复盘",
        "",
        f"生成时间：{now_sg()}",
        "",
        "## Route",
        "",
        f"- route: `{route['route']}`",
        f"- promotion_allowed: {route['promotion_allowed']}",
        f"- blocking_metric: `{route['blocking_metric']}`",
        f"- S0.15 pass: {route['S0_15_pass']}",
        f"- D-CHE/D-FOU S1 pass: {route['D-CHE_S1_pass']} / {route['D-FOU_S1_pass']}",
        f"- D-RAT/D-RBF multibatch closed: {route['D-RAT_D-RBF_multibatch_closed']}",
        f"- C0 retained-source observer pass rows: {route['C0_retained_source_observer_pass_rows']}",
        f"- official early-chain positive rows: {route['official_early_chain_positive_rows']}",
        f"- retained h800+h3200 positive rows: {route['retained_h800_h3200_positive_rows']}",
        f"- C2 true block solver pass rows: {route['C2_true_block_solver_pass_rows']}",
        f"- C3 source formation pass rows: {route['C3_source_formation_pass_rows']}",
        f"- post-no-go C-O6 route: `{route.get('post_nogo_C-O6_route', '')}`",
        f"- post-no-go C-O6 train-only/fresh/official pass rows: {route.get('post_nogo_C-O6_train_only_pass_rows', 0)}/{route.get('post_nogo_C-O6_fresh_C3_pass_rows', 0)}/{route.get('post_nogo_C-O6_official_C3_pass_rows', 0)}",
        f"- post-no-go C-O7 route: `{route.get('post_nogo_C-O7_route', '')}`",
        f"- post-no-go C-O7 train-only/fresh/official pass rows: {route.get('post_nogo_C-O7_train_only_pass_rows', 0)}/{route.get('post_nogo_C-O7_fresh_C3_pass_rows', 0)}/{route.get('post_nogo_C-O7_official_C3_pass_rows', 0)}",
        f"- post-no-go C-O8 route: `{route.get('post_nogo_C-O8_route', '')}`",
        f"- post-no-go C-O8 train-only/fresh/official pass rows: {route.get('post_nogo_C-O8_train_only_pass_rows', 0)}/{route.get('post_nogo_C-O8_fresh_C3_pass_rows', 0)}/{route.get('post_nogo_C-O8_official_C3_pass_rows', 0)}",
        f"- post-no-go C-O9 route: `{route.get('post_nogo_C-O9_route', '')}`",
        f"- post-no-go C-O9 train-only/fresh/official pass rows: {route.get('post_nogo_C-O9_train_only_pass_rows', 0)}/{route.get('post_nogo_C-O9_fresh_C3_pass_rows', 0)}/{route.get('post_nogo_C-O9_official_C3_pass_rows', 0)}",
        f"- post-no-go C-O10 route: `{route.get('post_nogo_C-O10_route', '')}`",
        f"- post-no-go C-O10 train-only/fresh/official pass rows: {route.get('post_nogo_C-O10_train_only_pass_rows', 0)}/{route.get('post_nogo_C-O10_fresh_C3_pass_rows', 0)}/{route.get('post_nogo_C-O10_official_C3_pass_rows', 0)}",
        f"- post-no-go C-O11 route: `{route.get('post_nogo_C-O11_route', '')}`",
        f"- post-no-go C-O11 train-only/fresh/official pass rows: {route.get('post_nogo_C-O11_train_only_pass_rows', 0)}/{route.get('post_nogo_C-O11_fresh_C3_pass_rows', 0)}/{route.get('post_nogo_C-O11_official_C3_pass_rows', 0)}",
        f"- post-no-go C-O12 route: `{route.get('post_nogo_C-O12_route', '')}`",
        f"- post-no-go C-O12 train-only/fresh/official pass rows: {route.get('post_nogo_C-O12_train_only_pass_rows', 0)}/{route.get('post_nogo_C-O12_fresh_C3_pass_rows', 0)}/{route.get('post_nogo_C-O12_official_C3_pass_rows', 0)}",
        f"- post-no-go C-O12 verify route: `{route.get('post_nogo_C-O12_verify_route', '')}`",
        f"- post-no-go C-O12 verify robust-fresh/official pass rows: {route.get('post_nogo_C-O12_verify_robust_fresh_rows', 0)}/{route.get('post_nogo_C-O12_verify_official_C3_pass_rows', 0)}",
        f"- terminal_preservation_decision: `{route['terminal_preservation_decision']}`",
        f"- KAN_mapping_decision: `{route['KAN_mapping_decision']}`",
        f"- execution_contract_violation: {route['execution_contract_violation']}",
        f"- next_codex_action: {route['next_codex_action']}",
        "",
        "## Part A S0.15 Code / Metric / Solver Gate",
        "",
        md_table(code_rows, ["check", "pass", "metric", "value", "blocker"], 40),
        "",
        "### C2 Repair Recompute Correctness",
        "",
        "- audit note: `projection_residual_Gf` is a ratio and can remain scale-invariant for an exact linear readout solve; v22.08 therefore also records target_norm/update_norm scaling and block-restricted masks. This is written as evidence, not as promotion.",
        f"- recompute summary: {'; '.join(c2_scale_notes)}",
        md_table(c2_recompute, ["case", "target_scale", "block_role", "projection_residual_Gf", "ActuationR2", "B2_transfer_gain", "parameter_update_norm", "target_norm_L2", "block_restricted_solver", "C2_recompute_test_pass", "blocker"], 20),
        "",
        "## Part B Basis Efficiency / Officialization",
        "",
        md_table(eff_rows, ["carrier", "variant_family", "forward_ratio_vs_mlp", "backward_ratio_vs_mlp", "step_ratio_vs_mlp", "memory_ratio_vs_mlp", "functional_direction_ms", "metric_solver_ms", "LineC_audit_ms", "same_kernel_functional_runner_proof", "fallback_kernel_used", "official_fused_kernel_complete", "v22_08_S1_pass", "v22_08_blocker"], 20),
        "",
        "### D-RAT / D-RBF Multibatch",
        "",
        md_table(dr_summary, ["carrier", "profile_rows", "robust_production_pass_rows", "near_E1_rows", "component_telemetry_complete_rows", "best_forward_ratio", "best_step_ratio", "best_memory_ratio", "decision", "blocker", "repair_attempt_rows", "repair_micro_near_E1_rows", "repair_best_variant", "repair_attempt_status", "repair_blocker", "official_closure_claimed"], 20),
        "",
        "### D-RAT / D-RBF Repair Attempt",
        "",
        md_table(dr_repair, ["carrier", "repair_attempt_rows", "repair_micro_near_E1_rows", "repair_nonreference_micro_near_E1_rows", "repair_component_telemetry_complete_rows", "repair_best_forward_ratio", "repair_best_step_ratio", "repair_best_memory_ratio", "repair_best_variant", "repair_best_batch_size", "repair_attempt_status", "repair_blocker", "official_closure_claimed"], 20),
        "",
        "## Part C Retained-Source Observer Theory",
        "",
        f"- best retained-label observer: `{best_obs_retained.get('observer_family', '')}` AUC_retained={best_obs_retained.get('AUC_predict_retained_h800_h3200_positive', '')} precision@20={best_obs_retained.get('precision_at_top20', '')}",
        f"- best official-early observer: `{best_obs_early.get('observer_family', '')}` AUC_official_early={best_obs_early.get('AUC_predict_official_early_chain', '')} precision@20={best_obs_early.get('precision_at_top20', '')}",
        f"- best top candidate by retained score: `{best_top.get('v21_id', '')}` dataset={best_top.get('dataset', '')} h100={best_top.get('future_audit_source_h100', '')} h400={best_top.get('future_audit_source_h400', '')} h800={best_top.get('future_audit_source_h800', '')} h3200={best_top.get('future_audit_source_h3200', '')}",
        "",
        md_table(obs_summary, ["observer_family", "rows", "positive_official_early_chain_rows", "positive_h3200_rows", "positive_retained_h800_h3200_rows", "AUC_predict_official_early_chain", "AUC_predict_h3200_positive", "AUC_predict_retained_h800_h3200_positive", "precision_at_top20", "heldout_dataset_seed_precision_at_top20", "control_equivalent_fraction", "C0_retained_source_observer_pass", "blocker"], 20),
        "",
        "### Top Candidates",
        "",
        md_table(obs_top, ["job_order", "v21_id", "dataset", "future_audit_source_h100", "future_audit_source_h400", "future_audit_source_h800", "future_audit_source_h1600", "future_audit_source_h3200", "official_early_chain_h100_h400_h800_positive", "retained_h800_h3200_positive"], 12),
        "",
        "### Post-No-Go C-O6 Source-State Persistence",
        "",
        "- trigger: v22.08 reached `R-ObserverLocalNoGoBoundary`; per plan Case C, this run stops C-O1..C-O5 scale/cap/floor variants and tries a new train-only pathwise source-state persistence certificate.",
        f"- route: `{post_route.get('route', '')}` previous_observer_route=`{post_route.get('previous_observer_route', '')}` blocker=`{post_route.get('blocker', '')}`",
        f"- probe rows / measured rows: {post_route.get('probe_rows', '')}/{post_route.get('probe_measured_rows', '')}",
        f"- train-only pass / fresh C3 pass / official C3 pass: {post_route.get('PSSC_train_only_certificate_pass_rows', '')}/{post_route.get('fresh_C3_source_state_pass_rows', '')}/{post_route.get('official_C3_pass_rows', '')}",
        "",
        md_table(post_summary, ["certificate_family", "rows", "positive_official_early_chain_rows", "positive_h3200_rows", "positive_retained_h800_h3200_rows", "AUC_predict_official_early_chain", "AUC_predict_h3200_positive", "AUC_predict_retained_h800_h3200_positive", "precision_at_top20", "PSSC_train_only_certificate_pass_rows", "PSSC_official_observer_pass", "blocker"], 10),
        "",
        "#### C-O6 Selected Candidates",
        "",
        md_table(post_selected, ["job_order", "v21_id", "mechanism", "dataset", "seed", "PSSC_score", "source_state_coherence", "source_state_pair_cosine_mean", "max_control_cosine", "B2_transfer_gain", "B3_safety_gain", "future_audit_source_h800", "future_audit_source_h3200", "PSSC_train_only_certificate_pass"], 8),
        "",
        "#### C-O6 Fresh Source-State C3",
        "",
        md_table(post_fresh_summary, ["v21_id", "dataset", "seed", "PSSC_score", "source_vs_best_control_h100", "source_vs_best_control_h400", "source_vs_best_control_h800", "source_vs_best_control_h1600", "source_vs_best_control_h2400", "source_vs_best_control_h3200", "source_state_alignment_mean", "debt_not_exploded", "PSSC_fresh_C3_source_state_pass", "official_C3_pass", "blocker"], 12),
        "",
        "### Post-No-Go C-O7 Counterfactual Washout",
        "",
        "- trigger: C-O6 source-state persistence failed fresh C3; this follow-up tests a distinct train-only principle: whether a candidate direction leaves persistent functional displacement after a short AdamW washout compared with sign-flip, corrupt-label and random matched controls.",
        f"- route: `{cow_route.get('route', '')}` previous_post_nogo_route=`{cow_route.get('previous_post_nogo_route', '')}` blocker=`{cow_route.get('blocker', '')}`",
        f"- probe rows / measured rows: {cow_route.get('probe_rows', '')}/{cow_route.get('probe_measured_rows', '')}",
        f"- train-only pass / fresh C3 pass / official C3 pass: {cow_route.get('COW_train_only_certificate_pass_rows', '')}/{cow_route.get('fresh_C3_source_washout_pass_rows', '')}/{cow_route.get('official_C3_pass_rows', '')}",
        "",
        md_table(cow_summary, ["certificate_family", "rows", "positive_official_early_chain_rows", "positive_h3200_rows", "positive_retained_h800_h3200_rows", "AUC_predict_official_early_chain", "AUC_predict_h3200_positive", "AUC_predict_retained_h800_h3200_positive", "precision_at_top20", "COW_train_only_certificate_pass_rows", "COW_official_observer_pass", "blocker"], 10),
        "",
        "#### C-O7 Selected Candidates",
        "",
        md_table(cow_selected, ["job_order", "v21_id", "mechanism", "dataset", "seed", "COW_score", "source_washout_retention_ratio", "source_retention_control_gap", "source_B2_control_gap_after_washout", "source_B3_control_gap_after_washout", "future_audit_source_h800", "future_audit_source_h3200", "COW_train_only_certificate_pass"], 8),
        "",
        "#### C-O7 Fresh Counterfactual-Washout C3",
        "",
        md_table(cow_fresh_summary, ["v21_id", "dataset", "seed", "COW_score", "source_vs_best_control_h100", "source_vs_best_control_h400", "source_vs_best_control_h800", "source_vs_best_control_h1600", "source_vs_best_control_h2400", "source_vs_best_control_h3200", "injection_count", "injection_norm_sum", "debt_not_exploded", "COW_fresh_C3_source_washout_pass", "official_C3_pass", "blocker"], 12),
        "",
        "### Post-No-Go C-O8 NTK / Jacobian Eigen-Channel",
        "",
        "- trigger: C-O7 counterfactual washout failed fresh C3; this run tests a distinct train-only operator principle: candidate directions should live in a controllable train-margin Jacobian mid-spectrum channel rather than a near-null or control-equivalent channel.",
        f"- route: `{ntk_route.get('route', '')}` previous_post_nogo_route=`{ntk_route.get('previous_post_nogo_route', '')}` blocker=`{ntk_route.get('blocker', '')}`",
        f"- probe rows / measured rows: {ntk_route.get('probe_rows', '')}/{ntk_route.get('probe_measured_rows', '')}",
        f"- train-only pass / fresh C3 pass / official C3 pass: {ntk_route.get('NTKC_train_only_certificate_pass_rows', '')}/{ntk_route.get('fresh_C3_eigen_channel_pass_rows', '')}/{ntk_route.get('official_C3_pass_rows', '')}",
        "",
        md_table(ntk_summary, ["certificate_family", "rows", "positive_official_early_chain_rows", "positive_h3200_rows", "positive_retained_h800_h3200_rows", "AUC_predict_official_early_chain", "AUC_predict_h3200_positive", "AUC_predict_retained_h800_h3200_positive", "precision_at_top20", "NTKC_train_only_certificate_pass_rows", "NTKC_official_observer_pass", "blocker"], 10),
        "",
        "#### C-O8 Selected Candidates",
        "",
        md_table(ntk_selected, ["job_order", "v21_id", "mechanism", "dataset", "seed", "NTKC_score", "source_mid_eigen_channel_fraction", "source_jacobian_null_energy_fraction", "source_train_margin_gain_mean", "mid_channel_control_gap", "margin_control_gap", "future_audit_source_h800", "future_audit_source_h3200", "NTKC_train_only_certificate_pass"], 8),
        "",
        "#### C-O8 Fresh Eigen-Channel C3",
        "",
        md_table(ntk_fresh_summary, ["v21_id", "dataset", "seed", "NTKC_score", "source_vs_best_control_h100", "source_vs_best_control_h400", "source_vs_best_control_h800", "source_vs_best_control_h1600", "source_vs_best_control_h2400", "source_vs_best_control_h3200", "projected_mid_channel_fraction_mean", "injection_count", "injection_norm_sum", "debt_not_exploded", "NTKC_fresh_C3_eigen_channel_pass", "official_C3_pass", "blocker"], 12),
        "",
        "### Post-No-Go C-O9 Augmentation Tangent Consistency",
        "",
        "- trigger: C-O8 NTK/Jacobian channel failed fresh C3; this run tests a distinct train-only input-neighborhood principle: candidate directions should improve local augmentation consistency and train margin relative to random/sign-flip/corrupt controls.",
        f"- route: `{aug_route.get('route', '')}` previous_post_nogo_route=`{aug_route.get('previous_post_nogo_route', '')}` blocker=`{aug_route.get('blocker', '')}`",
        f"- probe rows / measured rows: {aug_route.get('probe_rows', '')}/{aug_route.get('probe_measured_rows', '')}",
        f"- train-only pass / fresh C3 pass / official C3 pass: {aug_route.get('ATC_train_only_certificate_pass_rows', '')}/{aug_route.get('fresh_C3_aug_tangent_pass_rows', '')}/{aug_route.get('official_C3_pass_rows', '')}",
        "",
        md_table(aug_summary, ["certificate_family", "rows", "positive_official_early_chain_rows", "positive_h3200_rows", "positive_retained_h800_h3200_rows", "AUC_predict_official_early_chain", "AUC_predict_h3200_positive", "AUC_predict_retained_h800_h3200_positive", "precision_at_top20", "ATC_train_only_certificate_pass_rows", "ATC_official_observer_pass", "blocker"], 10),
        "",
        "#### C-O9 Selected Candidates",
        "",
        md_table(aug_selected, ["job_order", "v21_id", "mechanism", "dataset", "seed", "ATC_score", "source_aug_tangent_consistency_gain", "consistency_control_gap", "source_aug_tangent_margin_gain", "margin_control_gap", "B2_control_gap", "B3_control_gap", "future_audit_source_h800", "future_audit_source_h3200", "ATC_train_only_certificate_pass"], 8),
        "",
        "#### C-O9 Fresh Aug-Tangent C3",
        "",
        md_table(aug_fresh_summary, ["v21_id", "dataset", "seed", "ATC_score", "source_vs_best_control_h100", "source_vs_best_control_h400", "source_vs_best_control_h800", "source_vs_best_control_h1600", "source_vs_best_control_h2400", "source_vs_best_control_h3200", "aug_consistency_gain_mean", "injection_count", "injection_norm_sum", "debt_not_exploded", "ATC_fresh_C3_aug_tangent_pass", "official_C3_pass", "blocker"], 12),
        "",
        "### Post-No-Go C-O10 Bootstrap Basin Consensus",
        "",
        "- trigger: C-O9 augmentation tangent consistency failed fresh C3; this run tests a distinct train-only basin principle: candidate directions should align with short-run bootstrap sub-training consensus directions and remain separated from random/sign-flip/corrupt controls.",
        f"- route: `{bbc_route.get('route', '')}` previous_post_nogo_route=`{bbc_route.get('previous_post_nogo_route', '')}` blocker=`{bbc_route.get('blocker', '')}`",
        f"- probe rows / measured rows: {bbc_route.get('probe_rows', '')}/{bbc_route.get('probe_measured_rows', '')}",
        f"- train-only pass / fresh C3 pass / official C3 pass: {bbc_route.get('BBC_train_only_certificate_pass_rows', '')}/{bbc_route.get('fresh_C3_bootstrap_basin_pass_rows', '')}/{bbc_route.get('official_C3_pass_rows', '')}",
        "",
        md_table(bbc_summary, ["certificate_family", "rows", "positive_official_early_chain_rows", "positive_h3200_rows", "positive_retained_h800_h3200_rows", "AUC_predict_official_early_chain", "AUC_predict_h3200_positive", "AUC_predict_retained_h800_h3200_positive", "precision_at_top20", "BBC_train_only_certificate_pass_rows", "BBC_official_observer_pass", "blocker"], 10),
        "",
        "#### C-O10 Selected Candidates",
        "",
        md_table(bbc_selected, ["job_order", "v21_id", "mechanism", "dataset", "seed", "BBC_score", "source_basin_cosine", "basin_alignment_control_gap", "bootstrap_pair_cosine_mean", "bootstrap_consensus_norm_ratio", "train_loss_control_gap", "B2_control_gap", "B3_control_gap", "future_audit_source_h800", "future_audit_source_h3200", "BBC_train_only_certificate_pass"], 8),
        "",
        "#### C-O10 Fresh Bootstrap-Basin C3",
        "",
        md_table(bbc_fresh_summary, ["v21_id", "dataset", "seed", "BBC_score", "source_vs_best_control_h100", "source_vs_best_control_h400", "source_vs_best_control_h800", "source_vs_best_control_h1600", "source_vs_best_control_h2400", "source_vs_best_control_h3200", "bootstrap_pair_cosine_mean", "bootstrap_consensus_norm_ratio", "raw_basin_cosine_mean", "train_loss_gain_mean", "injection_count", "injection_norm_sum", "debt_not_exploded", "BBC_fresh_C3_bootstrap_basin_pass", "official_C3_pass", "blocker"], 12),
        "",
        "### Post-No-Go C-O11 Cross-Fit Influence Transport",
        "",
        "- trigger: C-O10 bootstrap basin consensus failed fresh C3; this run tests a distinct train-only cross-fit principle: a source direction built on one train fold should transport through an influence direction from a disjoint train fold and remain safe on a third train fold relative to random/sign-flip/corrupt controls.",
        f"- route: `{cfi_route.get('route', '')}` previous_post_nogo_route=`{cfi_route.get('previous_post_nogo_route', '')}` blocker=`{cfi_route.get('blocker', '')}`",
        f"- probe rows / measured rows: {cfi_route.get('probe_rows', '')}/{cfi_route.get('probe_measured_rows', '')}",
        f"- train-only pass / fresh C3 pass / official C3 pass: {cfi_route.get('CFI_train_only_certificate_pass_rows', '')}/{cfi_route.get('fresh_C3_crossfit_influence_pass_rows', '')}/{cfi_route.get('official_C3_pass_rows', '')}",
        "",
        md_table(cfi_summary, ["certificate_family", "rows", "positive_official_early_chain_rows", "positive_h3200_rows", "positive_retained_h800_h3200_rows", "AUC_predict_official_early_chain", "AUC_predict_h3200_positive", "AUC_predict_retained_h800_h3200_positive", "precision_at_top20", "CFI_train_only_certificate_pass_rows", "CFI_official_observer_pass", "blocker"], 10),
        "",
        "#### C-O11 Selected Candidates",
        "",
        md_table(cfi_selected, ["job_order", "v21_id", "mechanism", "dataset", "seed", "CFI_score", "source_crossfit_B_loss_gain", "B_crossfit_control_gap", "source_crossfit_C_safety_loss_gain", "C_safety_control_gap", "source_influence_cosine", "crossfit_alignment_control_gap", "future_audit_source_h800", "future_audit_source_h3200", "CFI_train_only_certificate_pass"], 8),
        "",
        "#### C-O11 Fresh Cross-Fit Influence C3",
        "",
        md_table(cfi_fresh_summary, ["v21_id", "dataset", "seed", "CFI_score", "source_vs_best_control_h100", "source_vs_best_control_h400", "source_vs_best_control_h800", "source_vs_best_control_h1600", "source_vs_best_control_h2400", "source_vs_best_control_h3200", "raw_crossfit_cosine_mean", "crossfit_loss_gain_mean", "injection_count", "injection_norm_sum", "debt_not_exploded", "CFI_fresh_C3_crossfit_influence_pass", "official_C3_pass", "blocker"], 12),
        "",
        "### Post-No-Go C-O12 Train-Flow Commutator",
        "",
        "- trigger: C-O11 cross-fit influence transport failed fresh C3; this run tests a distinct train-only operator principle: a candidate source step should have a useful finite commutator with local train loss flow, distinguishable from random/sign-flip/corrupt-label commutators.",
        f"- route: `{tfc_route.get('route', '')}` previous_post_nogo_route=`{tfc_route.get('previous_post_nogo_route', '')}` blocker=`{tfc_route.get('blocker', '')}`",
        f"- probe rows / measured rows: {tfc_route.get('probe_rows', '')}/{tfc_route.get('probe_measured_rows', '')}",
        f"- train-only pass / fresh C3 pass / official C3 pass: {tfc_route.get('TFC_train_only_certificate_pass_rows', '')}/{tfc_route.get('fresh_C3_train_flow_commutator_pass_rows', '')}/{tfc_route.get('official_C3_pass_rows', '')}",
        "",
        md_table(tfc_summary, ["certificate_family", "rows", "positive_official_early_chain_rows", "positive_h3200_rows", "positive_retained_h800_h3200_rows", "AUC_predict_official_early_chain", "AUC_predict_h3200_positive", "AUC_predict_retained_h800_h3200_positive", "precision_at_top20", "TFC_train_only_certificate_pass_rows", "TFC_official_observer_pass", "blocker"], 10),
        "",
        "#### C-O12 Selected Candidates",
        "",
        md_table(tfc_selected, ["job_order", "v21_id", "mechanism", "dataset", "seed", "TFC_score", "source_commutator_B_loss_gain", "B_commutator_control_gap", "source_commutator_norm_ratio", "source_commutator_cosine", "commutator_alignment_control_gap", "future_audit_source_h800", "future_audit_source_h3200", "TFC_train_only_certificate_pass"], 8),
        "",
        "#### C-O12 Fresh Train-Flow Commutator C3",
        "",
        md_table(tfc_fresh_summary, ["v21_id", "dataset", "seed", "TFC_score", "source_vs_best_control_h100", "source_vs_best_control_h400", "source_vs_best_control_h800", "source_vs_best_control_h1600", "source_vs_best_control_h2400", "source_vs_best_control_h3200", "raw_commutator_cosine_mean", "commutator_norm_ratio_mean", "commutator_loss_gain_mean", "injection_count", "injection_norm_sum", "debt_not_exploded", "TFC_fresh_C3_train_flow_commutator_pass", "official_C3_pass", "blocker"], 12),
        "",
        "### C-O12 Repeat Verification",
        "",
        "- trigger: C-O12 opened fresh smoke; this verification reruns the opened train-flow commutator candidate under repeated train seeds with the same AdamW/SGD/NoOp/random controls. Passing here is still fresh evidence only, not official C2/C3.",
        f"- route: `{tfcv_route.get('route', '')}` blocker=`{tfcv_route.get('blocker', '')}`",
        f"- verify candidates / repeat rows / robust fresh rows / official C3 rows: {tfcv_route.get('verify_candidate_rows', '')}/{tfcv_route.get('repeat_rows', '')}/{tfcv_route.get('TFC_repeat_robust_fresh_C3_verified_rows', '')}/{tfcv_route.get('official_C3_pass_rows', '')}",
        "",
        md_table(tfcv_summary, ["v21_id", "dataset", "seed", "repeat_rows", "required_repeats", "fresh_C3_repeat_pass_rows", "debt_not_exploded_rows", "source_vs_best_control_h100_min", "source_vs_best_control_h400_min", "source_vs_best_control_h800_min", "source_vs_best_control_h1600_min", "source_vs_best_control_h3200_min", "TFC_repeat_robust_fresh_C3_verified", "official_observer_pass", "official_C2_solver_claimed", "official_C3_pass", "blocker"], 12),
        "",
        "#### C-O12 Verify Fresh Rows",
        "",
        md_table(tfcv_fresh_summary, ["v21_id", "dataset", "seed", "TFC_score", "source_vs_best_control_h100", "source_vs_best_control_h400", "source_vs_best_control_h800", "source_vs_best_control_h1600", "source_vs_best_control_h2400", "source_vs_best_control_h3200", "raw_commutator_cosine_mean", "commutator_norm_ratio_mean", "commutator_loss_gain_mean", "debt_not_exploded", "TFC_fresh_C3_train_flow_commutator_pass", "official_C3_pass", "blocker"], 12),
        "",
        "## Part D Metric-As-Dynamics / Terminal / KAN",
        "",
        md_table(solver_summary, ["block_role", "rows", "C2_solver_gate_pass_rows", "projection_residual_Gf_min", "ActuationR2_max", "B2_transfer_gain_max", "blocker"], 20),
        "",
        md_table(terminal, ["source_h3200", "source_h4800", "R4800_over_3200", "C5_productive_terminal_source", "preservation_status", "blocker"], 10),
        "",
        md_table(kan, ["mapping_status", "blocker", "best_metric_v21_id", "KAN_source_channel_decision", "KAN_specific_delta_vs_MLP_same_metric"], 10),
        "",
        "## 4GPU Queue",
        "",
        md_table(queue, ["task_id", "line", "gpu", "command", "status", "returncode", "start_time", "end_time", "duration_sec", "log_path"], 20),
        "",
        md_table(idle, ["execution_contract_violation", "reason", "max_idle_gap_sec"], 10),
        "",
        "## 修改记录",
        "",
        "- 新增 `experiments/run_v22_08_common.py`：独立 v22.08 结果目录、执行日志、复盘、artifact index 与 bundle helper。",
        "- 更新 `dgkan/fu/metric_solver.py`：给 `solve_metric_readout_update` 增加默认不改变旧行为的 `target_scale` / `block_role` 参数，并输出 block-restricted diagnostics；用于 v22.08 C2 recompute hard gate。",
        "- 新增 `experiments/run_v22_08_s015_truth_gate.py`：S0.15 code/import/LineC/source-chain/solver/semantic/kernel/profiler gate，并新增 C2 repair recompute tests。",
        "- 新增 `experiments/run_v22_08_efficiency_reconfirm.py`：按 v22.08 字段读回 D-CHE/D-FOU measured full-loop artifact，不补造缺失 timing 字段。",
        "- 新增 `experiments/run_v22_08_drat_drbf_multibatch.py`：复用底层 profiler/repair 函数重新跑 D-RAT/D-RBF multibatch 与 component repair waterfall，输出 v22.08 artifact。",
        "- 新增 `experiments/run_v22_08_retained_source_observer.py`：把 v22.08 C-O1..C-O5 retained-source observer 作为 no-commit audit 评分；future source 只作为 label，不生成方向。",
        "- 新增 `experiments/run_v22_08_metric_dynamics_solver.py`、`run_v22_08_terminal_preservation.py`、`run_v22_08_kan_source_mapping.py`：严格按 observer/C3 gate fail-closed，不强行进入 h4800/KAN。",
        "- 新增 `experiments/run_v22_08_post_nogo_source_state_certificate.py`：在 local no-go 后尝试 C-O6 pathwise source-state persistence certificate；方向只来自 train-stream micro-batch source-state consistency/control-null evidence，future source 只作 audit label，并另跑 fresh source-state integration vs AdamW/SGD/NoOp/random controls。",
        "- 新增 `experiments/run_v22_08_post_nogo_counterfactual_washout.py`：在 C-O6 blocked 后继续尝试 C-O7 counterfactual washout persistence；短程 AdamW washout 后比较 source/sign-flip/corrupt/random matched 方向的训练流功能残留，并跑 periodic persistent FU fresh h3200 对照。",
        "- 新增 `experiments/run_v22_08_post_nogo_ntk_channel_certificate.py`：在 C-O7 blocked 后继续尝试 C-O8 NTK/Jacobian eigen-channel certificate；只用 train-margin Jacobian 局部谱通道评估方向可控性，并跑 mid-spectrum projected FU fresh h3200 对照。",
        "- 新增 `experiments/run_v22_08_post_nogo_aug_tangent_certificate.py`：在 C-O8 blocked 后继续尝试 C-O9 augmentation tangent consistency certificate；只用 train-only 输入邻域增强一致性/训练 margin/control-gap 构造证据，并跑 augmentation-tangent FU fresh h3200 对照。",
        "- 新增 `experiments/run_v22_08_post_nogo_bootstrap_basin_certificate.py`：在 C-O9 blocked 后继续尝试 C-O10 bootstrap basin consensus certificate；只用 train-only bootstrap 子训练流形成短程盆地方向，并跑 bootstrap-basin FU fresh h3200 对照。",
        "- 新增 `experiments/run_v22_08_post_nogo_crossfit_influence_certificate.py`：在 C-O10 blocked 后继续尝试 C-O11 cross-fit influence transport certificate；用互斥 train folds 的 source/influence/safety 证据检验方向跨训练子集迁移，并跑 cross-fit influence FU fresh h3200 对照。",
        "- 新增 `experiments/run_v22_08_post_nogo_train_flow_commutator_certificate.py`：在 C-O11 blocked 后继续尝试 C-O12 train-flow commutator certificate；用 source step 与局部 train loss flow 的有限非交换子检验方向是否携带可用训练流结构，并跑 train-flow-commutator FU fresh h3200 对照。",
        "- 新增 `experiments/run_v22_08_train_flow_commutator_fresh_verify.py`：当 C-O12 fresh smoke 打开后，按 repeat train seeds 重新验证同一候选对 AdamW/SGD/NoOp/random controls 的 h100-h3200 source-vs-control；通过也只写成 fresh verification，不声明 official C2/C3。",
        "- 新增 `experiments/run_v22_08_full.py` 与 `experiments/run_v22_08_finalize.py`：4GPU queue、日志、复盘、final route 和 bundle。",
        "- 更新 `experiments/run_v22_08_common.py` 与 `experiments/run_v22_08_finalize.py`：新增 `v22_08_code_review_packet.zip` 打包和 C-O6/C-O7/C-O8/C-O9/C-O10/C-O11/C-O12/verify post-no-go 复盘汇总；results bundle 与 code review packet 均重新生成。",
        "",
        "## 分析 / Insight / 证据链",
        "",
        "- S0.15 不是只跑旧 S0.14：本轮新增 C2 recompute rows，分别覆盖 target_scale=1/64 与 all/hidden/readout block role，且记录 JVP/VJP/CG、target_norm、update_norm 与 block-restricted mask。",
        "- C2 recompute 中 `projection_residual_Gf` 是 ratio，因此 scale=1 与 scale=64 可以出现相同比值；这不被改写成不同数值。本轮用 target_norm/update_norm scaling 与 block_role mask 作为是否重算的直接证据。",
        "- D-CHE/D-FOU 仍来自 measured full-loop artifact readback；缺失的 functional_direction_ms/metric_solver_ms/LineC_audit_ms 字段保留为空，没有估计填补。",
        "- D-RAT/D-RBF 若 robust multibatch 未闭合，只允许 component repair / limited smoke；repair rows 即使 micro-near-E1，也因 `official_closure_claimed=0` 不提升为 official carrier。",
        "- Retained-source observer 的 official early-chain label 是 h100+h400+h800 同时为正；本轮读回矩阵中该 label 正例为 0，因此所有 C-O families fail-closed。这个结论只针对 v22.08 已尝试的 C-O 评分定义，不是 universal no-go。",
        "- retained h800+h3200 label 仍可有 AUC/precision 读数，但它不能替代 official early-chain gate；因此不能用 late retained/late rebound 进入 Line D。",
        "- Line D 按计划依赖 C-O observer gate。observer fail 后，C2 recompute 只作为 code/solver integrity evidence，不作为 C3 source-formation promotion。",
        "- Terminal preservation 与 KAN mapping 均因 C3 未打开而未进入；这避免了把 blocked-before-gate 的 h4800/KAN 表写成实验成功。",
        "- C-O6 不是 C-O1..C-O5 的同族 scale/cap/floor 调参：它从同一机制在多个合法 train micro-batch 上的 signed displacement 一致性构造 slow source-state probe，并显式记录 max_control_cosine/radial_fraction，试图区分普通 CE/corrupt 控制流。",
        "- C-O6 的 future source 仍只用于 audit label；fresh C3 部分只比较 `PSSC-AdamWPlusSlowSourceState` 对 AdamW/SGD/NoOp/random matched controls 的 horizon delta。即使 fresh smoke 有局部正值，也不会被写成 official C3，除非 h100/h400/h800/h1600/h3200 与 debt gate 同时打开且后续 official gate 补齐。",
        "- C-O7 进一步检查“被短程 AdamW 洗出后仍保留的功能位移”是否可作为 retained-source certificate。该 probe 仍只用 train stream 构造方向和对照；fresh C3 使用 `COW-AdamWPlusPeriodicPersistentFU` 与 AdamW/SGD/NoOp/random periodic controls 比较 horizon delta。",
        "- C-O8 换成 operator-level train-margin Jacobian eigen-channel 证书：候选方向必须在局部 NTK/Jacobian 中谱可控通道里产生 margin effect，并与 random/sign-flip/corrupt controls 区分。fresh C3 使用 frozen mid-spectrum projection，而不是继续调前面 observer 的 scale/cap/floor。",
        "- C-O9 再换成 train-only 输入邻域/augmentation tangent 证书：候选方向必须在 train augmentation 一致性与 margin 上优于 random/sign-flip/corrupt controls。fresh C3 使用 `ATC-AdamWPlusAugTangentFU` 与 AdamW/SGD/NoOp/random matched controls 比较 horizon delta。",
        "- C-O10 再换成 train-only bootstrap basin-consensus 证书：候选方向必须与多个 bootstrap 子训练流形成的短程 basin consensus direction 对齐，并在 train/B2/B3 上不输给 random/sign-flip/corrupt controls。fresh C3 使用 `BBC-AdamWPlusBootstrapBasinFU` 与 AdamW/SGD/NoOp/random matched bootstrap controls 比较 horizon delta。",
        "- C-O10 的 train-only certificate rows 打开不等价于 source formation：本轮 C-O10 official observer 仍为 0，fresh C3 pass rows 仍为 0，top fresh rows 到 h3200 仍为负。因此不能把 bootstrap basin 共识写成 promotion 或 official C3。",
        "- C-O11 再换成 train-only cross-fit influence transport 证书：候选方向必须在 A 折生成后，经 B 折 influence direction 外推，并在 C 折安全检查上不输给 random/sign-flip/corrupt controls。fresh C3 使用 `CFI-AdamWPlusCrossFitInfluenceFU` 与 AdamW/SGD/NoOp/random matched cross-fit controls 比较 horizon delta。",
        "- C-O11 的 train-only certificate rows 若打开仍不等价于 source formation：只有 official observer 与 fresh h100/h400/h800/h1600/h3200/debt gate 同时打开，才允许后续 official C2/C3/terminal 论证；否则必须保持 blocked。",
        "- C-O12 再换成 train-only train-flow commutator 证书：候选方向必须与局部 train loss flow 产生有限非交换残差，并且该残差在 B/C train folds 上不输给 random/sign-flip/corrupt commutator controls。fresh C3 使用 `TFC-AdamWPlusTrainFlowCommutatorFU` 与 AdamW/SGD/NoOp/random matched train-flow-commutator controls 比较 horizon delta。",
        "- C-O12 的 train-only certificate rows 若打开仍不等价于 source formation：只有 official observer 与 fresh h100/h400/h800/h1600/h3200/debt gate 同时打开，才允许后续 official C2/C3/terminal 论证；否则必须保持 blocked。",
        "- C-O12 repeat verification 若 robust fresh 通过，仍然只是重复 fresh smoke：本轮没有 official observer positive label、没有 official C2 solver claim、没有 official C3 claim，因此不能进入 terminal/KAN promotion。",
        f"- 最新结论：v22.08 把 v22.07 的 C2 repair 语义漏洞修成可审计重算；C-O6/C-O7/C-O8/C-O9/C-O10/C-O11 之后，C-O12 route=`{tfc_route.get('route', '')}`，fresh_C3_pass_rows={tfc_route.get('fresh_C3_train_flow_commutator_pass_rows', '')}，official_C3_pass_rows={tfc_route.get('official_C3_pass_rows', '')}；repeat route=`{tfcv_route.get('route', '')}`，robust_fresh_rows={tfcv_route.get('TFC_repeat_robust_fresh_C3_verified_rows', '')}，official_C3_pass_rows={tfcv_route.get('official_C3_pass_rows', '')}。本轮 C-O12 single-run fresh smoke 打开，但 repeat verification 未稳住，因此 promotion 仍 blocked；继续推进需要 C-O1..C-O12 之外的新 source-observability principle，不能把这条 unstable smoke 写成成功。",
        "",
        "_完整 artifact 清单保留在 `v22_08_artifact_index.csv`，不放入复盘正文。_",
        "",
    ]
    V2208_RECAP_DOC.write_text("\n".join(text), encoding="utf-8")


def main() -> None:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir)
    route = _route(out_dir)
    write_json(out_dir / "v22_08_final_route.json", route)
    write_rows(out_dir / "v22_08_final_route.csv", [route])
    _write_recap(out_dir, route)
    index = artifact_index(out_dir)
    write_rows(out_dir / "v22_08_artifact_index.csv", index)
    code_packet = build_code_review_packet(out_dir)
    index = artifact_index(out_dir)
    write_rows(out_dir / "v22_08_artifact_index.csv", index)
    bundle = build_results_bundle(out_dir)
    index = artifact_index(out_dir)
    write_rows(out_dir / "v22_08_artifact_index.csv", index)
    append_exec(
        out_dir,
        f"{PYTHON} experiments/run_v22_08_finalize.py --out-dir {out_dir}",
        status="completed",
        note=f"route={route['route']} artifacts={len(index)} bundle={bundle} code_review_packet={code_packet}",
    )
    code_packet = build_code_review_packet(out_dir)
    index = artifact_index(out_dir)
    write_rows(out_dir / "v22_08_artifact_index.csv", index)
    bundle = build_results_bundle(out_dir)
    index = artifact_index(out_dir)
    write_rows(out_dir / "v22_08_artifact_index.csv", index)


if __name__ == "__main__":
    main()
