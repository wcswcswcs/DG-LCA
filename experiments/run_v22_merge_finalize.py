#!/usr/bin/env python3
"""v22 final route/manifest/recap builder."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_v22_common import (  # noqa: E402
    PYTHON,
    V22_RECAP_DOC,
    append_exec,
    append_text,
    build_packet,
    ensure_out,
    finite_float,
    init_docs,
    int_flag,
    now_sg,
    placeholder_svg,
    read_json,
    read_rows,
    sha256_file,
    write_json,
    write_rows,
    write_text,
)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None)
    return p


def md_table(rows: list[dict[str, Any]], fields: list[str], max_rows: int = 40) -> str:
    shown = rows[:max_rows]
    if not shown:
        return "\n_无落盘 rows。_\n"
    out = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for row in shown:
        out.append("| " + " | ".join(str(row.get(f, "")) for f in fields) + " |")
    if len(rows) > max_rows:
        out.append(f"\n_仅显示前 {max_rows} / {len(rows)} rows；完整 CSV 见 artifact。_")
    return "\n".join(out) + "\n"


def summarize(out_dir: Path) -> dict[str, Any]:
    truth = read_rows(out_dir / "v22_code_truth_gate.csv")
    eff = read_rows(out_dir / "v22_efficiency_officialization_summary.csv")
    source_summary = read_rows(out_dir / "v22_source_chain_summary.csv")
    source_decision = read_json(out_dir / "v22_source_chain_decision.json")
    observability_decision = read_json(out_dir / "v22_source_observability_decision.json")
    f46_decision = read_json(out_dir / "v22_f46_train_loss_selector_decision.json")
    f48_decision = read_json(out_dir / "v22_f48_trainloss_holdfallback_decision.json")
    f49_decision = read_json(out_dir / "v22_f49_trainloss_tinyfallback_decision.json")
    f50_decision = read_json(out_dir / "v22_f50_trainloss_boosted_tinyfallback_decision.json")
    f51_decision = read_json(out_dir / "v22_f51_late_hold_recovery_decision.json")
    f52_decision = read_json(out_dir / "v22_f52_late_lookahead_floor_decision.json")
    f53_decision = read_json(out_dir / "v22_f53_terminal_lookahead_floor_decision.json")
    f54_decision = read_json(out_dir / "v22_f54_early_terminal_lookahead_floor_decision.json")
    f55_decision = read_json(out_dir / "v22_f55_split_fisher_source_reset_decision.json")
    f56_decision = read_json(out_dir / "v22_f56_momentum_warm_split_fisher_source_decision.json")
    f57_decision = read_json(out_dir / "v22_f57_antiwashout_source_reset_decision.json")
    f58_decision = read_json(out_dir / "v22_f58_source_anchor_reset_decision.json")
    f59_decision = read_json(out_dir / "v22_f59_param_ema_reentry_reset_decision.json")
    f60_decision = read_json(out_dir / "v22_f60_readout_channel_reset_decision.json")
    f61_decision = read_json(out_dir / "v22_f61_hidden_matrix_channel_reset_decision.json")
    f62_decision = read_json(out_dir / "v22_f62_terminal_projected_lookahead_floor_decision.json")
    f63_decision = read_json(out_dir / "v22_f63_terminal_consensus_lookahead_floor_decision.json")
    f64_decision = read_json(out_dir / "v22_f64_terminal_selector_lookahead_floor_decision.json")
    f65_decision = read_json(out_dir / "v22_f65_adamw_split_fisher_residual_decision.json")
    f66_decision = read_json(out_dir / "v22_f66_kan_readout_commit_decision.json")
    f67_decision = read_json(out_dir / "v22_f67_kan_lowbank_b3_null_decision.json")
    f68_decision = read_json(out_dir / "v22_f68_adamw_boundary_lowbank_b3_null_decision.json")
    f69_decision = read_json(out_dir / "v22_f69_adamw_lowbank_anchor_antiwashout_decision.json")
    f70_decision = read_json(out_dir / "v22_f70_terminal_positive_lookahead_floor_decision.json")
    f71_decision = read_json(out_dir / "v22_f71_terminal_checkpoint_reentry_decision.json")
    f72_decision = read_json(out_dir / "v22_f72_terminal_hard_split_source_decision.json")
    f73_decision = read_json(out_dir / "v22_f73_terminal_adamw_lookahead_decision.json")
    f74_decision = read_json(out_dir / "v22_f74_terminal_optimizer_selector_decision.json")
    required = [
        "v22_code_truth_gate.csv",
        "v22_import_closure.csv",
        "v22_source_chain_formula_tests.csv",
        "v22_efficiency_truth_table.csv",
        "v22_efficiency_officialization_summary.csv",
        "v22_source_chain_matrix.csv",
        "v22_source_chain_summary.csv",
        "v22_source_chain_decision.json",
        "v22_source_observability_predictor_scan.csv",
        "v22_source_observability_decision.json",
        "v22_f40_phase_reset_summary.csv",
        "v22_f40_phase_reset_decision.json",
        "v22_f41_phase_reset_summary.csv",
        "v22_f41_phase_reset_decision.json",
        "v22_f42_phase_reset_summary.csv",
        "v22_f42_phase_reset_decision.json",
        "v22_f43_phase_reset_summary.csv",
        "v22_f43_phase_reset_decision.json",
        "v22_f44_phase_reset_summary.csv",
        "v22_f44_phase_reset_decision.json",
        "v22_f45_phase_reset_summary.csv",
        "v22_f45_phase_reset_decision.json",
        "v22_f47_trainloss_gated_summary.csv",
        "v22_f47_trainloss_gated_decision.json",
        "v22_f48_trainloss_holdfallback_summary.csv",
        "v22_f48_trainloss_holdfallback_decision.json",
        "v22_f49_trainloss_tinyfallback_summary.csv",
        "v22_f49_trainloss_tinyfallback_decision.json",
        "v22_f50_trainloss_boosted_tinyfallback_summary.csv",
        "v22_f50_trainloss_boosted_tinyfallback_decision.json",
        "v22_f51_late_hold_recovery_summary.csv",
        "v22_f51_late_hold_recovery_decision.json",
        "v22_f52_late_lookahead_floor_summary.csv",
        "v22_f52_late_lookahead_floor_decision.json",
        "v22_f53_terminal_lookahead_floor_summary.csv",
        "v22_f53_terminal_lookahead_floor_decision.json",
        "v22_f54_early_terminal_lookahead_floor_summary.csv",
        "v22_f54_early_terminal_lookahead_floor_decision.json",
        "v22_f55_split_fisher_source_reset_summary.csv",
        "v22_f55_split_fisher_source_reset_decision.json",
        "v22_f56_momentum_warm_split_fisher_source_summary.csv",
        "v22_f56_momentum_warm_split_fisher_source_decision.json",
        "v22_f57_antiwashout_source_reset_summary.csv",
        "v22_f57_antiwashout_source_reset_decision.json",
        "v22_f58_source_anchor_reset_summary.csv",
        "v22_f58_source_anchor_reset_decision.json",
        "v22_f59_param_ema_reentry_reset_summary.csv",
        "v22_f59_param_ema_reentry_reset_decision.json",
        "v22_f60_readout_channel_reset_summary.csv",
        "v22_f60_readout_channel_reset_decision.json",
        "v22_f61_hidden_matrix_channel_reset_summary.csv",
        "v22_f61_hidden_matrix_channel_reset_decision.json",
        "v22_f62_terminal_projected_lookahead_floor_summary.csv",
        "v22_f62_terminal_projected_lookahead_floor_decision.json",
        "v22_f63_terminal_consensus_lookahead_floor_summary.csv",
        "v22_f63_terminal_consensus_lookahead_floor_decision.json",
        "v22_f64_terminal_selector_lookahead_floor_summary.csv",
        "v22_f64_terminal_selector_lookahead_floor_decision.json",
        "v22_f65_adamw_split_fisher_residual_summary.csv",
        "v22_f65_adamw_split_fisher_residual_decision.json",
        "v22_f66_kan_readout_commit_summary.csv",
        "v22_f66_kan_readout_commit_decision.json",
        "v22_f67_kan_lowbank_b3_null_summary.csv",
        "v22_f67_kan_lowbank_b3_null_decision.json",
        "v22_f68_adamw_boundary_lowbank_b3_null_summary.csv",
        "v22_f68_adamw_boundary_lowbank_b3_null_decision.json",
        "v22_f69_adamw_lowbank_anchor_antiwashout_summary.csv",
        "v22_f69_adamw_lowbank_anchor_antiwashout_decision.json",
        "v22_f70_terminal_positive_lookahead_floor_summary.csv",
        "v22_f70_terminal_positive_lookahead_floor_decision.json",
        "v22_f71_terminal_checkpoint_reentry_summary.csv",
        "v22_f71_terminal_checkpoint_reentry_decision.json",
        "v22_f72_terminal_hard_split_source_summary.csv",
        "v22_f72_terminal_hard_split_source_decision.json",
        "v22_f73_terminal_adamw_lookahead_summary.csv",
        "v22_f73_terminal_adamw_lookahead_decision.json",
        "v22_f74_terminal_optimizer_selector_summary.csv",
        "v22_f74_terminal_optimizer_selector_decision.json",
        "v22_f46_train_loss_selector_threshold.csv",
        "v22_f46_train_loss_selector_holdout.csv",
        "v22_f46_train_loss_selector_selected_groups.csv",
        "v22_f46_train_loss_selector_decision.json",
        "v22_command_journal.csv",
    ]
    manifest = []
    for name in required:
        path = out_dir / name
        manifest.append(
            {
                "artifact": name,
                "required": 1,
                "exists": int(path.exists()),
                "size_bytes": path.stat().st_size if path.exists() else 0,
                "sha256": sha256_file(path) if path.exists() and path.is_file() else "",
            }
        )
    s0_pass = int(bool(truth) and all(int_flag(r.get("pass")) for r in truth))
    dche_s1 = sum(int_flag(r.get("S1_pass_rows")) for r in eff if str(r.get("carrier")) == "D-CHE")
    dfou_s1 = sum(int_flag(r.get("S1_pass_rows")) for r in eff if str(r.get("carrier")) == "D-FOU")
    dche_e1 = sum(int_flag(r.get("E1_pass_rows")) for r in eff if str(r.get("carrier")) == "D-CHE")
    dfou_e1 = sum(int_flag(r.get("E1_pass_rows")) for r in eff if str(r.get("carrier")) == "D-FOU")
    candidate_early = int(source_decision.get("candidate_early_chain_groups", 0) or 0)
    candidate_cont = int(source_decision.get("candidate_continuous_groups", 0) or 0)
    candidate_h3200 = int(source_decision.get("candidate_productive_h3200_groups", 0) or 0)
    candidate_h4800 = int(source_decision.get("candidate_productive_h4800_groups", 0) or 0)
    independent_pass = 0
    missing = sum(1 for r in manifest if not int_flag(r.get("exists")))
    if missing:
        route = "R0-RequiredArtifactMissing"
    elif not s0_pass:
        route = "R0-S07TruthGateFailed"
    elif candidate_early == 0:
        route = "R2-EarlySourceChainNotOpened-SourceObservabilityNoGo" if observability_decision.get("decision") == "SourceObservabilityNoGo" else "R2-EarlySourceChainNotOpened"
    elif candidate_cont == 0:
        if f48_decision.get("decision"):
            route = (
                "R3-EarlyChainNoContinuousRetention-F50BoostedTinyFallbackNoContinuousRetention"
                if f50_decision.get("decision")
                else "R3-EarlyChainNoContinuousRetention-F49TinyFallbackNoContinuousRetention"
                if f49_decision.get("decision")
                else "R3-EarlyChainNoContinuousRetention-F48HoldFallbackNoContinuousRetention"
            )
        elif int_flag(f46_decision.get("next_repair_recommended")):
            route = "R3-EarlyChainNoContinuousRetention-F46TrainLossSelectorRepairHypothesis"
        else:
            route = "R3-EarlyChainNoContinuousRetention"
    elif candidate_h4800 == 0:
        if f74_decision.get("decision"):
            route = "R4-ContinuousH3200NoH4800-F57F58F59F60F61F62F63F64F65F66F67F68F69F70F71F72F73F74SourceTheoryResetFailed"
        elif f73_decision.get("decision"):
            route = "R4-ContinuousH3200NoH4800-F57F58F59F60F61F62F63F64F65F66F67F68F69F70F71F72F73SourceTheoryResetFailed"
        elif f72_decision.get("decision"):
            route = "R4-ContinuousH3200NoH4800-F57F58F59F60F61F62F63F64F65F66F67F68F69F70F71F72SourceTheoryResetFailed"
        elif f71_decision.get("decision"):
            route = "R4-ContinuousH3200NoH4800-F57F58F59F60F61F62F63F64F65F66F67F68F69F70F71SourceTheoryResetFailed"
        elif f70_decision.get("decision"):
            route = "R4-ContinuousH3200NoH4800-F57F58F59F60F61F62F63F64F65F66F67F68F69F70SourceTheoryResetFailed"
        elif f69_decision.get("decision"):
            route = "R4-ContinuousH3200NoH4800-F57F58F59F60F61F62F63F64F65F66F67F68F69SourceTheoryResetFailed"
        elif f68_decision.get("decision"):
            route = "R4-ContinuousH3200NoH4800-F57F58F59F60F61F62F63F64F65F66F67F68SourceTheoryResetFailed"
        elif f67_decision.get("decision"):
            route = "R4-ContinuousH3200NoH4800-F57F58F59F60F61F62F63F64F65F66F67SourceTheoryResetFailed"
        elif f66_decision.get("decision"):
            route = "R4-ContinuousH3200NoH4800-F57F58F59F60F61F62F63F64F65F66SourceTheoryResetFailed"
        elif f65_decision.get("decision"):
            route = "R4-ContinuousH3200NoH4800-F57F58F59F60F61F62F63F64F65SourceTheoryResetFailed"
        elif f64_decision.get("decision"):
            route = "R4-ContinuousH3200NoH4800-F57F58F59F60F61F62F63F64SourceTheoryResetFailed"
        elif f63_decision.get("decision"):
            route = "R4-ContinuousH3200NoH4800-F57F58F59F60F61F62F63SourceTheoryResetFailed"
        elif f62_decision.get("decision"):
            route = "R4-ContinuousH3200NoH4800-F57F58F59F60F61F62SourceTheoryResetFailed"
        elif f61_decision.get("decision"):
            route = "R4-ContinuousH3200NoH4800-F57F58F59F60F61SourceTheoryResetFailed"
        elif f60_decision.get("decision"):
            route = "R4-ContinuousH3200NoH4800-F57F58F59F60SourceTheoryResetFailed"
        elif f59_decision.get("decision"):
            route = "R4-ContinuousH3200NoH4800-F57F58F59SourceTheoryResetFailed"
        elif f58_decision.get("decision"):
            route = "R4-ContinuousH3200NoH4800-F57F58SourceTheoryResetFailed"
        elif f57_decision.get("decision"):
            route = "R4-ContinuousH3200NoH4800-F57AntiWashoutSourceResetFailed"
        elif f56_decision.get("decision"):
            route = "R4-ContinuousH3200NoH4800-F53TerminalNearMiss-F55F56SplitFisherResetFailed"
        elif f55_decision.get("decision"):
            route = "R4-ContinuousH3200NoH4800-F53TerminalNearMiss-F55SplitFisherResetFailed"
        elif f54_decision.get("decision"):
            route = "R4-ContinuousH3200NoH4800-F53TerminalNearMiss-F54EarlyTerminalRegressed"
        elif f53_decision.get("decision"):
            route = "R4-ContinuousH3200NoH4800-F53TerminalLookaheadH4800Failed"
        elif f51_decision.get("decision"):
            route = "R4-ContinuousH3200NoH4800-F51LateHoldRecoveryH4800Failed"
        else:
            route = (
                "R4-ContinuousH3200NoH4800-F50BoostedTinyFallbackH4800Failed"
                if f50_decision.get("decision")
                else "R4-ContinuousH3200NoH4800"
            )
    elif not independent_pass:
        route = "R5-NeedsIndependentConfirmation"
    else:
        route = "S5-PromotionReady"
    promotion_allowed = int(
        route == "S5-PromotionReady"
        and s0_pass
        and dche_s1 > 0
        and dfou_s1 > 0
        and candidate_h4800 > 0
        and independent_pass
    )
    decision = {
        "route": route,
        "route_detail": (
            f"S0_7_pass={s0_pass}, D-CHE_E1_rows={dche_e1}, D-FOU_E1_rows={dfou_e1}, "
            f"D-CHE_S1_rows={dche_s1}, D-FOU_S1_rows={dfou_s1}, "
            f"candidate_early_chain={candidate_early}, candidate_continuous={candidate_cont}, "
            f"candidate_h3200={candidate_h3200}, candidate_h4800={candidate_h4800}, independent={independent_pass}"
        ),
        "CodeRoute": "S0.7-CodeMetricMechanismPassed" if s0_pass else "S0.7-FailedOrMissing",
        "EfficiencyRoute": f"D-CHE_E1={dche_e1};D-FOU_E1={dfou_e1};D-CHE_S1={dche_s1};D-FOU_S1={dfou_s1}",
        "FunctionalRoute": f"early={candidate_early};continuous={candidate_cont};h3200={candidate_h3200};h4800={candidate_h4800};independent={independent_pass}",
        "promotion_allowed": promotion_allowed,
        "required_artifact_missing_count": missing,
        "truth_rows": len(truth),
        "efficiency_rows": sum(int(r.get("rows", 0) or 0) for r in eff),
        "source_chain_rows": int(source_decision.get("source_chain_rows", 0) or 0),
        "source_chain_groups": int(source_decision.get("source_chain_groups", 0) or 0),
        "late_rebound_groups": int(source_decision.get("late_rebound_groups", 0) or 0),
        "source_observability_decision": observability_decision.get("decision", ""),
        "passing_train_only_predictors": int(observability_decision.get("passing_train_only_predictors", 0) or 0),
        "passing_legal_predictors": int(observability_decision.get("passing_legal_predictors", observability_decision.get("passing_train_only_predictors", 0)) or 0),
        "f46_selector_decision": f46_decision.get("decision", ""),
        "f46_next_repair_recommended": int_flag(f46_decision.get("next_repair_recommended")),
        "f48_online_repair_decision": f48_decision.get("decision", ""),
        "f49_online_repair_decision": f49_decision.get("decision", ""),
        "f50_online_repair_decision": f50_decision.get("decision", ""),
        "f51_online_repair_decision": f51_decision.get("decision", ""),
        "f52_online_repair_decision": f52_decision.get("decision", ""),
        "f53_online_repair_decision": f53_decision.get("decision", ""),
        "f54_online_repair_decision": f54_decision.get("decision", ""),
        "f55_source_reset_decision": f55_decision.get("decision", ""),
        "f56_source_reset_decision": f56_decision.get("decision", ""),
        "f57_source_reset_decision": f57_decision.get("decision", ""),
        "f58_source_reset_decision": f58_decision.get("decision", ""),
        "f59_source_reset_decision": f59_decision.get("decision", ""),
        "f60_source_reset_decision": f60_decision.get("decision", ""),
        "f61_source_reset_decision": f61_decision.get("decision", ""),
        "f62_source_reset_decision": f62_decision.get("decision", ""),
        "f63_source_reset_decision": f63_decision.get("decision", ""),
        "f64_source_reset_decision": f64_decision.get("decision", ""),
        "f65_source_reset_decision": f65_decision.get("decision", ""),
        "f66_kan_readout_commit_decision": f66_decision.get("decision", ""),
        "f67_kan_lowbank_b3_null_decision": f67_decision.get("decision", ""),
        "f68_adamw_boundary_lowbank_b3_null_decision": f68_decision.get("decision", ""),
        "f69_adamw_lowbank_anchor_antiwashout_decision": f69_decision.get("decision", ""),
        "f70_terminal_positive_lookahead_floor_decision": f70_decision.get("decision", ""),
        "f71_terminal_checkpoint_reentry_decision": f71_decision.get("decision", ""),
        "f72_terminal_hard_split_source_decision": f72_decision.get("decision", ""),
        "f73_terminal_adamw_lookahead_decision": f73_decision.get("decision", ""),
        "f74_terminal_optimizer_selector_decision": f74_decision.get("decision", ""),
        "f53_candidate_h4800": f53_decision.get("candidate_h4800", ""),
        "f54_candidate_h4800": f54_decision.get("candidate_h4800", ""),
        "f55_candidate_h4800": f55_decision.get("candidate_h4800", ""),
        "f56_candidate_h4800": f56_decision.get("candidate_h4800", ""),
        "f57_candidate_h4800": f57_decision.get("candidate_h4800", ""),
        "f58_candidate_h4800": f58_decision.get("candidate_h4800", ""),
        "f59_candidate_h4800": f59_decision.get("candidate_h4800", ""),
        "f60_candidate_h4800": f60_decision.get("candidate_h4800", ""),
        "f61_candidate_h4800": f61_decision.get("candidate_h4800", ""),
        "f62_candidate_h4800": f62_decision.get("candidate_h4800", ""),
        "f63_candidate_h4800": f63_decision.get("candidate_h4800", ""),
        "f64_candidate_h4800": f64_decision.get("candidate_h4800", ""),
        "f65_candidate_h4800": f65_decision.get("candidate_h4800", ""),
        "f66_candidate_h4800": f66_decision.get("candidate_h4800", ""),
        "f67_candidate_h4800": f67_decision.get("candidate_h4800", ""),
        "f68_candidate_h4800": f68_decision.get("candidate_h4800", ""),
        "f69_candidate_h4800": f69_decision.get("candidate_h4800", ""),
        "f70_candidate_h4800": f70_decision.get("candidate_h4800", ""),
        "f71_candidate_h4800": f71_decision.get("candidate_h4800", ""),
        "f72_candidate_h4800": f72_decision.get("candidate_h4800", ""),
        "f73_candidate_h4800": f73_decision.get("candidate_h4800", ""),
        "f74_candidate_h4800": f74_decision.get("candidate_h4800", ""),
    }
    write_json(out_dir / "v22_route_decision.json", decision)
    write_rows(out_dir / "v22_route_decision.csv", [decision])
    write_rows(out_dir / "v22_required_artifact_manifest.csv", manifest)
    placeholder_svg(out_dir / "figures" / "v22_source_chain_h800.svg", source_summary, "h800")
    placeholder_svg(out_dir / "figures" / "v22_efficiency_step_ratio.svg", eff, "best_step_ratio")
    return decision


def render_recap(out_dir: Path, decision: dict[str, Any]) -> None:
    truth = read_rows(out_dir / "v22_code_truth_gate.csv")
    eff = read_rows(out_dir / "v22_efficiency_officialization_summary.csv")
    source_summary = read_rows(out_dir / "v22_source_chain_summary.csv")
    obs_scan = read_rows(out_dir / "v22_source_observability_predictor_scan.csv")
    obs_decision = read_json(out_dir / "v22_source_observability_decision.json")
    source_matrix = read_rows(out_dir / "v22_source_chain_matrix.csv")
    f40_summary = read_rows(out_dir / "v22_f40_phase_reset_summary.csv")
    f40_decision = read_json(out_dir / "v22_f40_phase_reset_decision.json")
    f41_summary = read_rows(out_dir / "v22_f41_phase_reset_summary.csv")
    f41_decision = read_json(out_dir / "v22_f41_phase_reset_decision.json")
    f42_summary = read_rows(out_dir / "v22_f42_phase_reset_summary.csv")
    f42_decision = read_json(out_dir / "v22_f42_phase_reset_decision.json")
    f43_summary = read_rows(out_dir / "v22_f43_phase_reset_summary.csv")
    f43_decision = read_json(out_dir / "v22_f43_phase_reset_decision.json")
    f44_summary = read_rows(out_dir / "v22_f44_phase_reset_summary.csv")
    f44_decision = read_json(out_dir / "v22_f44_phase_reset_decision.json")
    f45_summary = read_rows(out_dir / "v22_f45_phase_reset_summary.csv")
    f45_decision = read_json(out_dir / "v22_f45_phase_reset_decision.json")
    f47_summary = read_rows(out_dir / "v22_f47_trainloss_gated_summary.csv")
    f47_decision = read_json(out_dir / "v22_f47_trainloss_gated_decision.json")
    f48_summary = read_rows(out_dir / "v22_f48_trainloss_holdfallback_summary.csv")
    f48_decision = read_json(out_dir / "v22_f48_trainloss_holdfallback_decision.json")
    f49_summary = read_rows(out_dir / "v22_f49_trainloss_tinyfallback_summary.csv")
    f49_decision = read_json(out_dir / "v22_f49_trainloss_tinyfallback_decision.json")
    f50_summary = read_rows(out_dir / "v22_f50_trainloss_boosted_tinyfallback_summary.csv")
    f50_decision = read_json(out_dir / "v22_f50_trainloss_boosted_tinyfallback_decision.json")
    f51_summary = read_rows(out_dir / "v22_f51_late_hold_recovery_summary.csv")
    f51_decision = read_json(out_dir / "v22_f51_late_hold_recovery_decision.json")
    f52_summary = read_rows(out_dir / "v22_f52_late_lookahead_floor_summary.csv")
    f52_decision = read_json(out_dir / "v22_f52_late_lookahead_floor_decision.json")
    f53_summary = read_rows(out_dir / "v22_f53_terminal_lookahead_floor_summary.csv")
    f53_decision = read_json(out_dir / "v22_f53_terminal_lookahead_floor_decision.json")
    f54_summary = read_rows(out_dir / "v22_f54_early_terminal_lookahead_floor_summary.csv")
    f54_decision = read_json(out_dir / "v22_f54_early_terminal_lookahead_floor_decision.json")
    f55_summary = read_rows(out_dir / "v22_f55_split_fisher_source_reset_summary.csv")
    f55_decision = read_json(out_dir / "v22_f55_split_fisher_source_reset_decision.json")
    f56_summary = read_rows(out_dir / "v22_f56_momentum_warm_split_fisher_source_summary.csv")
    f56_decision = read_json(out_dir / "v22_f56_momentum_warm_split_fisher_source_decision.json")
    f57_summary = read_rows(out_dir / "v22_f57_antiwashout_source_reset_summary.csv")
    f57_decision = read_json(out_dir / "v22_f57_antiwashout_source_reset_decision.json")
    f58_summary = read_rows(out_dir / "v22_f58_source_anchor_reset_summary.csv")
    f58_decision = read_json(out_dir / "v22_f58_source_anchor_reset_decision.json")
    f59_summary = read_rows(out_dir / "v22_f59_param_ema_reentry_reset_summary.csv")
    f59_decision = read_json(out_dir / "v22_f59_param_ema_reentry_reset_decision.json")
    f60_summary = read_rows(out_dir / "v22_f60_readout_channel_reset_summary.csv")
    f60_decision = read_json(out_dir / "v22_f60_readout_channel_reset_decision.json")
    f61_summary = read_rows(out_dir / "v22_f61_hidden_matrix_channel_reset_summary.csv")
    f61_decision = read_json(out_dir / "v22_f61_hidden_matrix_channel_reset_decision.json")
    f62_summary = read_rows(out_dir / "v22_f62_terminal_projected_lookahead_floor_summary.csv")
    f62_decision = read_json(out_dir / "v22_f62_terminal_projected_lookahead_floor_decision.json")
    f63_summary = read_rows(out_dir / "v22_f63_terminal_consensus_lookahead_floor_summary.csv")
    f63_decision = read_json(out_dir / "v22_f63_terminal_consensus_lookahead_floor_decision.json")
    f64_summary = read_rows(out_dir / "v22_f64_terminal_selector_lookahead_floor_summary.csv")
    f64_decision = read_json(out_dir / "v22_f64_terminal_selector_lookahead_floor_decision.json")
    f65_summary = read_rows(out_dir / "v22_f65_adamw_split_fisher_residual_summary.csv")
    f65_decision = read_json(out_dir / "v22_f65_adamw_split_fisher_residual_decision.json")
    f66_summary = read_rows(out_dir / "v22_f66_kan_readout_commit_summary.csv")
    f66_decision = read_json(out_dir / "v22_f66_kan_readout_commit_decision.json")
    f67_summary = read_rows(out_dir / "v22_f67_kan_lowbank_b3_null_summary.csv")
    f67_decision = read_json(out_dir / "v22_f67_kan_lowbank_b3_null_decision.json")
    f68_summary = read_rows(out_dir / "v22_f68_adamw_boundary_lowbank_b3_null_summary.csv")
    f68_decision = read_json(out_dir / "v22_f68_adamw_boundary_lowbank_b3_null_decision.json")
    f69_summary = read_rows(out_dir / "v22_f69_adamw_lowbank_anchor_antiwashout_summary.csv")
    f69_decision = read_json(out_dir / "v22_f69_adamw_lowbank_anchor_antiwashout_decision.json")
    f70_summary = read_rows(out_dir / "v22_f70_terminal_positive_lookahead_floor_summary.csv")
    f70_decision = read_json(out_dir / "v22_f70_terminal_positive_lookahead_floor_decision.json")
    f71_summary = read_rows(out_dir / "v22_f71_terminal_checkpoint_reentry_summary.csv")
    f71_decision = read_json(out_dir / "v22_f71_terminal_checkpoint_reentry_decision.json")
    f72_summary = read_rows(out_dir / "v22_f72_terminal_hard_split_source_summary.csv")
    f72_decision = read_json(out_dir / "v22_f72_terminal_hard_split_source_decision.json")
    f73_summary = read_rows(out_dir / "v22_f73_terminal_adamw_lookahead_summary.csv")
    f73_decision = read_json(out_dir / "v22_f73_terminal_adamw_lookahead_decision.json")
    f74_summary = read_rows(out_dir / "v22_f74_terminal_optimizer_selector_summary.csv")
    f74_decision = read_json(out_dir / "v22_f74_terminal_optimizer_selector_decision.json")
    f46_threshold = read_rows(out_dir / "v22_f46_train_loss_selector_threshold.csv")
    f46_holdout = read_rows(out_dir / "v22_f46_train_loss_selector_holdout.csv")
    f46_groups = read_rows(out_dir / "v22_f46_train_loss_selector_selected_groups.csv")
    f46_decision = read_json(out_dir / "v22_f46_train_loss_selector_decision.json")
    terminal_compare_rows = [
        r
        for r in source_matrix
        if str(r.get("v22_id", "")).startswith(("MLP-F53", "MLP-F70", "MLP-F71", "MLP-F72", "MLP-F73", "MLP-F74"))
    ]
    terminal_compare_rows = sorted(
        terminal_compare_rows,
        key=lambda r: (
            str(r.get("v22_id", "")),
            str(r.get("dataset", "")),
            int(r.get("seed", 0) or 0),
        ),
    )
    source_sorted = sorted(
        source_summary,
        key=lambda r: (
            -int_flag(r.get("early_source_chain_group")),
            -finite_float(r.get("h800"), -999.0),
            str(r.get("carrier", "")),
            str(r.get("v22_id", "")),
        ),
    )
    text = [
        "# DG-KAN v22.0 EarlySourceRetentionFU DCHE/DFOU KernelOfficialization 4GPU 实验结果复盘",
        "",
        f"生成时间：{now_sg()}",
        "",
        "## Route",
        "",
        f"- route: `{decision.get('route')}`",
        f"- route_detail: {decision.get('route_detail')}",
        f"- CodeRoute: {decision.get('CodeRoute')}",
        f"- EfficiencyRoute: {decision.get('EfficiencyRoute')}",
        f"- FunctionalRoute: {decision.get('FunctionalRoute')}",
        f"- promotion_allowed: {decision.get('promotion_allowed')}",
        "",
        "## 是否达成 v22 目标",
        "",
        "没有达成。" if not int_flag(decision.get("promotion_allowed")) else "达成。",
        "",
        "- S0.7 preflight pass: " + ("1" if str(decision.get("CodeRoute")) == "S0.7-CodeMetricMechanismPassed" else "0"),
        f"- D-CHE / D-FOU E1 rows: {decision.get('EfficiencyRoute')}",
        f"- source-chain rows/groups: {decision.get('source_chain_rows')} / {decision.get('source_chain_groups')}",
        f"- late rebound groups: {decision.get('late_rebound_groups')}",
        f"- source observability decision: {decision.get('source_observability_decision')}",
        f"- passing train-only predictors: {decision.get('passing_train_only_predictors')}",
        f"- passing legal C4 predictors: {decision.get('passing_legal_predictors')}",
        f"- F46 selector decision: {decision.get('f46_selector_decision')}",
        f"- F47 online repair decision: {f47_decision.get('decision', '')}",
        f"- F48 hold-fallback repair decision: {f48_decision.get('decision', '')}",
        f"- F49 tiny-fallback repair decision: {f49_decision.get('decision', '')}",
        f"- F50 boosted tiny-fallback repair decision: {f50_decision.get('decision', '')}",
        f"- F51 late-hold recovery decision: {f51_decision.get('decision', '')}",
        f"- F52 late lookahead-floor decision: {f52_decision.get('decision', '')}",
        f"- F53 terminal lookahead-floor decision: {f53_decision.get('decision', '')}",
        f"- F54 early-terminal lookahead-floor decision: {f54_decision.get('decision', '')}",
        f"- F55 split-Fisher source reset decision: {f55_decision.get('decision', '')}",
        f"- F56 momentum-warm split-Fisher source reset decision: {f56_decision.get('decision', '')}",
        f"- F57 anti-washout source reset decision: {f57_decision.get('decision', '')}",
        f"- F58 source-anchor reset decision: {f58_decision.get('decision', '')}",
        f"- F59 param-EMA reentry reset decision: {f59_decision.get('decision', '')}",
        f"- F60 readout-channel reset decision: {f60_decision.get('decision', '')}",
        f"- F61 hidden-matrix channel reset decision: {f61_decision.get('decision', '')}",
        f"- F62 terminal projected lookahead-floor decision: {f62_decision.get('decision', '')}",
        f"- F63 terminal consensus lookahead-floor decision: {f63_decision.get('decision', '')}",
        f"- F64 terminal selector lookahead-floor decision: {f64_decision.get('decision', '')}",
        f"- F65 AdamW split-Fisher residual decision: {f65_decision.get('decision', '')}",
        f"- F66 KAN readout-only commit decision: {f66_decision.get('decision', '')}",
        f"- F67 KAN low-bank B3-null decision: {f67_decision.get('decision', '')}",
        f"- F68 AdamW-boundary low-bank B3-null decision: {f68_decision.get('decision', '')}",
        f"- F69 AdamW low-bank anchor anti-washout decision: {f69_decision.get('decision', '')}",
        f"- F70 terminal positive lookahead-floor decision: {f70_decision.get('decision', '')}",
        f"- F71 terminal h3200 checkpoint reentry decision: {f71_decision.get('decision', '')}",
        f"- F72 terminal hard-split source decision: {f72_decision.get('decision', '')}",
        f"- F73 terminal AdamW-lookahead decision: {f73_decision.get('decision', '')}",
        f"- F74 terminal optimizer-selector decision: {f74_decision.get('decision', '')}",
        f"- required artifact missing count: {decision.get('required_artifact_missing_count')}",
        "",
        "## S0.7 Truth Gate",
        "",
        md_table(truth, ["check", "pass", "metric", "value", "blocker"]),
        "## Efficiency Evidence",
        "",
        md_table(eff, ["carrier", "rows", "E1_pass_rows", "S1_pass_rows", "E2_pass_rows", "best_forward_ratio", "best_step_ratio"]),
        "## Source-Chain Evidence",
        "",
        md_table(
            source_sorted,
            [
                "carrier",
                "variant",
                "v22_id",
                "rows",
                "h100",
                "h400",
                "h800",
                "h1600",
                "h3200",
                "h4800",
                "early_source_chain_group",
                "continuous_retention_group",
                "productive_h3200_group",
                "productive_h4800_group",
                "late_rebound_group",
                "source_chain_blocker",
            ],
            max_rows=80,
        ),
        "## C4 Source Observability Audit",
        "",
        f"- decision: `{obs_decision.get('decision', '')}`",
        f"- audit rows / candidate rows / candidate early rows: {obs_decision.get('audit_rows', '')} / {obs_decision.get('candidate_rows', '')} / {obs_decision.get('candidate_early_rows', '')}",
        f"- passing train-only predictors: {obs_decision.get('passing_train_only_predictors', '')}",
        "",
        md_table(obs_scan, ["predictor", "kind", "finite_rows", "positive_rows", "continuous_positive_rows", "AUC_early_chain", "AUC_continuous_retention", "dataset_min_AUC", "carrier_min_AUC", "Spearman_h800", "pass"], max_rows=40),
        "## F46 Train-Loss Selector Held-Out Validation",
        "",
        f"- decision: `{f46_decision.get('decision', '')}`",
        f"- fit / holdout rows: {f46_decision.get('fit_rows', '')} / {f46_decision.get('holdout_rows', '')}",
        f"- best predictor: {f46_decision.get('best_predictor', '')}",
        f"- threshold raw / score: {f46_decision.get('threshold_raw', '')} / {f46_decision.get('threshold_score', '')}",
        f"- holdout selected rows / continuous rows: {f46_decision.get('holdout_selected_rows', '')} / {f46_decision.get('holdout_selected_continuous_rows', '')}",
        f"- holdout selected h100/h400/h800/h1600/h3200/h4800: {f46_decision.get('holdout_selected_h100', '')} / {f46_decision.get('holdout_selected_h400', '')} / {f46_decision.get('holdout_selected_h800', '')} / {f46_decision.get('holdout_selected_h1600', '')} / {f46_decision.get('holdout_selected_h3200', '')} / {f46_decision.get('holdout_selected_h4800', '')}",
        f"- holdout group early / continuous / h4800: {f46_decision.get('holdout_group_early_chain', '')} / {f46_decision.get('holdout_group_continuous', '')} / {f46_decision.get('holdout_group_h4800', '')}",
        f"- next repair recommended: {f46_decision.get('next_repair_recommended', '')}",
        "",
        md_table(f46_threshold, ["predictor", "fit_rows", "fit_early_rows", "fit_continuous_rows", "threshold_raw", "tp", "fp", "fn", "tn", "precision", "recall", "f1", "youden"], max_rows=10),
        md_table(f46_holdout, ["split", "predictor", "rows", "continuous_rows_all", "selected_rows", "selected_continuous_rows", "selected_h800", "selected_h1600", "selected_h3200", "selected_h4800", "selected_group_continuous", "selected_group_h4800", "selected_group_blocker"], max_rows=30),
        md_table(f46_groups, ["split", "predictor", "carrier", "v22_id", "rows", "continuous_rows", "h800", "h1600", "h3200", "h4800", "continuous_retention_group", "productive_h4800_group", "source_chain_blocker"], max_rows=30),
        "## F40 AdamW Boundary Phase Reset",
        "",
        f"- decision: `{f40_decision.get('decision', '')}`",
        f"- rows / groups: {f40_decision.get('rows', '')} / {f40_decision.get('groups', '')}",
        f"- candidate h100/h400/h800/h1600: {f40_decision.get('candidate_h100', '')} / {f40_decision.get('candidate_h400', '')} / {f40_decision.get('candidate_h800', '')} / {f40_decision.get('candidate_h1600', '')}",
        f"- candidate early/continuous/h3200/h4800: {f40_decision.get('candidate_early_chain', '')} / {f40_decision.get('candidate_continuous', '')} / {f40_decision.get('candidate_productive_h3200', '')} / {f40_decision.get('candidate_productive_h4800', '')}",
        "",
        md_table(f40_summary, ["carrier", "v22_id", "rows", "blocked_rows", "h100", "h400", "h800", "h1600", "h3200", "h4800", "h1600_retention_ratio", "h3200_retention_ratio", "h4800_retention_ratio", "early_source_chain_group", "continuous_retention_group", "source_chain_blocker"], max_rows=20),
        "## F41 Dual-Timescale Phase Reset",
        "",
        f"- decision: `{f41_decision.get('decision', '')}`",
        f"- rows / groups: {f41_decision.get('rows', '')} / {f41_decision.get('groups', '')}",
        f"- candidate h100/h400/h800/h1600/h3200/h4800: {f41_decision.get('candidate_h100', '')} / {f41_decision.get('candidate_h400', '')} / {f41_decision.get('candidate_h800', '')} / {f41_decision.get('candidate_h1600', '')} / {f41_decision.get('candidate_h3200', '')} / {f41_decision.get('candidate_h4800', '')}",
        f"- retention ratios h1600/h800, h3200/h1600, h4800/h3200: {f41_decision.get('candidate_h1600_retention_ratio', '')} / {f41_decision.get('candidate_h3200_retention_ratio', '')} / {f41_decision.get('candidate_h4800_retention_ratio', '')}",
        f"- candidate early/continuous/h3200/h4800: {f41_decision.get('candidate_early_chain', '')} / {f41_decision.get('candidate_continuous', '')} / {f41_decision.get('candidate_productive_h3200', '')} / {f41_decision.get('candidate_productive_h4800', '')}",
        "",
        md_table(f41_summary, ["carrier", "v22_id", "rows", "blocked_rows", "h100", "h400", "h800", "h1600", "h3200", "h4800", "h1600_retention_ratio", "h3200_retention_ratio", "h4800_retention_ratio", "early_source_chain_group", "continuous_retention_group", "source_chain_blocker"], max_rows=20),
        "## F42 Dual-Timescale With SGD Floor",
        "",
        f"- decision: `{f42_decision.get('decision', '')}`",
        f"- rows / groups: {f42_decision.get('rows', '')} / {f42_decision.get('groups', '')}",
        f"- candidate h100/h400/h800/h1600/h3200/h4800: {f42_decision.get('candidate_h100', '')} / {f42_decision.get('candidate_h400', '')} / {f42_decision.get('candidate_h800', '')} / {f42_decision.get('candidate_h1600', '')} / {f42_decision.get('candidate_h3200', '')} / {f42_decision.get('candidate_h4800', '')}",
        f"- retention ratios h1600/h800, h3200/h1600, h4800/h3200: {f42_decision.get('candidate_h1600_retention_ratio', '')} / {f42_decision.get('candidate_h3200_retention_ratio', '')} / {f42_decision.get('candidate_h4800_retention_ratio', '')}",
        f"- candidate early/continuous/h3200/h4800: {f42_decision.get('candidate_early_chain', '')} / {f42_decision.get('candidate_continuous', '')} / {f42_decision.get('candidate_productive_h3200', '')} / {f42_decision.get('candidate_productive_h4800', '')}",
        "",
        md_table(f42_summary, ["carrier", "v22_id", "rows", "blocked_rows", "h100", "h400", "h800", "h1600", "h3200", "h4800", "h1600_retention_ratio", "h3200_retention_ratio", "h4800_retention_ratio", "early_source_chain_group", "continuous_retention_group", "source_chain_blocker"], max_rows=20),
        "## F43 Late SGD Floor After H3200",
        "",
        f"- decision: `{f43_decision.get('decision', '')}`",
        f"- rows / groups: {f43_decision.get('rows', '')} / {f43_decision.get('groups', '')}",
        f"- candidate h100/h400/h800/h1600/h3200/h4800: {f43_decision.get('candidate_h100', '')} / {f43_decision.get('candidate_h400', '')} / {f43_decision.get('candidate_h800', '')} / {f43_decision.get('candidate_h1600', '')} / {f43_decision.get('candidate_h3200', '')} / {f43_decision.get('candidate_h4800', '')}",
        f"- retention ratios h1600/h800, h3200/h1600, h4800/h3200: {f43_decision.get('candidate_h1600_retention_ratio', '')} / {f43_decision.get('candidate_h3200_retention_ratio', '')} / {f43_decision.get('candidate_h4800_retention_ratio', '')}",
        f"- candidate early/continuous/h3200/h4800: {f43_decision.get('candidate_early_chain', '')} / {f43_decision.get('candidate_continuous', '')} / {f43_decision.get('candidate_productive_h3200', '')} / {f43_decision.get('candidate_productive_h4800', '')}",
        "",
        md_table(f43_summary, ["carrier", "v22_id", "rows", "blocked_rows", "h100", "h400", "h800", "h1600", "h3200", "h4800", "h1600_retention_ratio", "h3200_retention_ratio", "h4800_retention_ratio", "early_source_chain_group", "continuous_retention_group", "source_chain_blocker"], max_rows=20),
        "## F44 Tiny Late SGD Floor",
        "",
        f"- decision: `{f44_decision.get('decision', '')}`",
        f"- rows / groups: {f44_decision.get('rows', '')} / {f44_decision.get('groups', '')}",
        f"- candidate h100/h400/h800/h1600/h3200/h4800: {f44_decision.get('candidate_h100', '')} / {f44_decision.get('candidate_h400', '')} / {f44_decision.get('candidate_h800', '')} / {f44_decision.get('candidate_h1600', '')} / {f44_decision.get('candidate_h3200', '')} / {f44_decision.get('candidate_h4800', '')}",
        f"- retention ratios h1600/h800, h3200/h1600, h4800/h3200: {f44_decision.get('candidate_h1600_retention_ratio', '')} / {f44_decision.get('candidate_h3200_retention_ratio', '')} / {f44_decision.get('candidate_h4800_retention_ratio', '')}",
        f"- candidate early/continuous/h3200/h4800: {f44_decision.get('candidate_early_chain', '')} / {f44_decision.get('candidate_continuous', '')} / {f44_decision.get('candidate_productive_h3200', '')} / {f44_decision.get('candidate_productive_h4800', '')}",
        "",
        md_table(f44_summary, ["carrier", "v22_id", "rows", "blocked_rows", "h100", "h400", "h800", "h1600", "h3200", "h4800", "h1600_retention_ratio", "h3200_retention_ratio", "h4800_retention_ratio", "early_source_chain_group", "continuous_retention_group", "source_chain_blocker"], max_rows=20),
        "## F45 Reinforced Tiny Late Floor",
        "",
        f"- decision: `{f45_decision.get('decision', '')}`",
        f"- rows / groups: {f45_decision.get('rows', '')} / {f45_decision.get('groups', '')}",
        f"- candidate h100/h400/h800/h1600/h3200/h4800: {f45_decision.get('candidate_h100', '')} / {f45_decision.get('candidate_h400', '')} / {f45_decision.get('candidate_h800', '')} / {f45_decision.get('candidate_h1600', '')} / {f45_decision.get('candidate_h3200', '')} / {f45_decision.get('candidate_h4800', '')}",
        f"- retention ratios h1600/h800, h3200/h1600, h4800/h3200: {f45_decision.get('candidate_h1600_retention_ratio', '')} / {f45_decision.get('candidate_h3200_retention_ratio', '')} / {f45_decision.get('candidate_h4800_retention_ratio', '')}",
        f"- candidate early/continuous/h3200/h4800: {f45_decision.get('candidate_early_chain', '')} / {f45_decision.get('candidate_continuous', '')} / {f45_decision.get('candidate_productive_h3200', '')} / {f45_decision.get('candidate_productive_h4800', '')}",
        "",
        md_table(f45_summary, ["carrier", "v22_id", "rows", "blocked_rows", "h100", "h400", "h800", "h1600", "h3200", "h4800", "h1600_retention_ratio", "h3200_retention_ratio", "h4800_retention_ratio", "early_source_chain_group", "continuous_retention_group", "source_chain_blocker"], max_rows=20),
        "## F47 Train-Loss Gated Late Retention Repair",
        "",
        f"- decision: `{f47_decision.get('decision', '')}`",
        f"- rows / groups: {f47_decision.get('rows', '')} / {f47_decision.get('groups', '')}",
        f"- candidate h100/h400/h800/h1600/h3200/h4800: {f47_decision.get('candidate_h100', '')} / {f47_decision.get('candidate_h400', '')} / {f47_decision.get('candidate_h800', '')} / {f47_decision.get('candidate_h1600', '')} / {f47_decision.get('candidate_h3200', '')} / {f47_decision.get('candidate_h4800', '')}",
        f"- retention ratios h1600/h800, h3200/h1600, h4800/h3200: {f47_decision.get('candidate_h1600_retention_ratio', '')} / {f47_decision.get('candidate_h3200_retention_ratio', '')} / {f47_decision.get('candidate_h4800_retention_ratio', '')}",
        f"- candidate early/continuous/h3200/h4800: {f47_decision.get('candidate_early_chain', '')} / {f47_decision.get('candidate_continuous', '')} / {f47_decision.get('candidate_productive_h3200', '')} / {f47_decision.get('candidate_productive_h4800', '')}",
        "",
        md_table(f47_summary, ["carrier", "v22_id", "rows", "blocked_rows", "h100", "h400", "h800", "h1600", "h3200", "h4800", "h1600_retention_ratio", "h3200_retention_ratio", "h4800_retention_ratio", "early_source_chain_group", "continuous_retention_group", "source_chain_blocker"], max_rows=20),
        "## F48 Train-Loss Gate With Hold Fallback",
        "",
        f"- decision: `{f48_decision.get('decision', '')}`",
        f"- rows / groups: {f48_decision.get('rows', '')} / {f48_decision.get('groups', '')}",
        f"- candidate h100/h400/h800/h1600/h3200/h4800: {f48_decision.get('candidate_h100', '')} / {f48_decision.get('candidate_h400', '')} / {f48_decision.get('candidate_h800', '')} / {f48_decision.get('candidate_h1600', '')} / {f48_decision.get('candidate_h3200', '')} / {f48_decision.get('candidate_h4800', '')}",
        f"- retention ratios h1600/h800, h3200/h1600, h4800/h3200: {f48_decision.get('candidate_h1600_retention_ratio', '')} / {f48_decision.get('candidate_h3200_retention_ratio', '')} / {f48_decision.get('candidate_h4800_retention_ratio', '')}",
        f"- candidate early/continuous/h3200/h4800: {f48_decision.get('candidate_early_chain', '')} / {f48_decision.get('candidate_continuous', '')} / {f48_decision.get('candidate_productive_h3200', '')} / {f48_decision.get('candidate_productive_h4800', '')}",
        "",
        md_table(f48_summary, ["carrier", "v22_id", "rows", "blocked_rows", "h100", "h400", "h800", "h1600", "h3200", "h4800", "h1600_retention_ratio", "h3200_retention_ratio", "h4800_retention_ratio", "early_source_chain_group", "continuous_retention_group", "source_chain_blocker"], max_rows=20),
        "## F49 Train-Loss Gate With Tiny Late Fallback",
        "",
        f"- decision: `{f49_decision.get('decision', '')}`",
        f"- rows / groups: {f49_decision.get('rows', '')} / {f49_decision.get('groups', '')}",
        f"- candidate h100/h400/h800/h1600/h3200/h4800: {f49_decision.get('candidate_h100', '')} / {f49_decision.get('candidate_h400', '')} / {f49_decision.get('candidate_h800', '')} / {f49_decision.get('candidate_h1600', '')} / {f49_decision.get('candidate_h3200', '')} / {f49_decision.get('candidate_h4800', '')}",
        f"- retention ratios h1600/h800, h3200/h1600, h4800/h3200: {f49_decision.get('candidate_h1600_retention_ratio', '')} / {f49_decision.get('candidate_h3200_retention_ratio', '')} / {f49_decision.get('candidate_h4800_retention_ratio', '')}",
        f"- candidate early/continuous/h3200/h4800: {f49_decision.get('candidate_early_chain', '')} / {f49_decision.get('candidate_continuous', '')} / {f49_decision.get('candidate_productive_h3200', '')} / {f49_decision.get('candidate_productive_h4800', '')}",
        "",
        md_table(f49_summary, ["carrier", "v22_id", "rows", "blocked_rows", "h100", "h400", "h800", "h1600", "h3200", "h4800", "h1600_retention_ratio", "h3200_retention_ratio", "h4800_retention_ratio", "early_source_chain_group", "continuous_retention_group", "source_chain_blocker"], max_rows=20),
        "## F50 Boosted Tiny Late Fallback",
        "",
        f"- decision: `{f50_decision.get('decision', '')}`",
        f"- rows / groups: {f50_decision.get('rows', '')} / {f50_decision.get('groups', '')}",
        f"- candidate h100/h400/h800/h1600/h3200/h4800: {f50_decision.get('candidate_h100', '')} / {f50_decision.get('candidate_h400', '')} / {f50_decision.get('candidate_h800', '')} / {f50_decision.get('candidate_h1600', '')} / {f50_decision.get('candidate_h3200', '')} / {f50_decision.get('candidate_h4800', '')}",
        f"- retention ratios h1600/h800, h3200/h1600, h4800/h3200: {f50_decision.get('candidate_h1600_retention_ratio', '')} / {f50_decision.get('candidate_h3200_retention_ratio', '')} / {f50_decision.get('candidate_h4800_retention_ratio', '')}",
        f"- candidate early/continuous/h3200/h4800: {f50_decision.get('candidate_early_chain', '')} / {f50_decision.get('candidate_continuous', '')} / {f50_decision.get('candidate_productive_h3200', '')} / {f50_decision.get('candidate_productive_h4800', '')}",
        "",
        md_table(f50_summary, ["carrier", "v22_id", "rows", "blocked_rows", "h100", "h400", "h800", "h1600", "h3200", "h4800", "h1600_retention_ratio", "h3200_retention_ratio", "h4800_retention_ratio", "early_source_chain_group", "continuous_retention_group", "source_chain_blocker"], max_rows=20),
        "## F51-F54 Late-Phase Source Preservation / Terminal Recovery",
        "",
        f"- F51 decision: `{f51_decision.get('decision', '')}`; h3200/h4800: {f51_decision.get('candidate_h3200', '')} / {f51_decision.get('candidate_h4800', '')}",
        f"- F52 decision: `{f52_decision.get('decision', '')}`; h3200/h4800: {f52_decision.get('candidate_h3200', '')} / {f52_decision.get('candidate_h4800', '')}",
        f"- F53 decision: `{f53_decision.get('decision', '')}`; h3200/h4800: {f53_decision.get('candidate_h3200', '')} / {f53_decision.get('candidate_h4800', '')}",
        f"- F54 decision: `{f54_decision.get('decision', '')}`; h3200/h4800: {f54_decision.get('candidate_h3200', '')} / {f54_decision.get('candidate_h4800', '')}",
        "",
        md_table(
            f51_summary + f52_summary + f53_summary + f54_summary,
            ["carrier", "v22_id", "rows", "blocked_rows", "h100", "h400", "h800", "h1600", "h3200", "h4800", "h1600_retention_ratio", "h3200_retention_ratio", "h4800_retention_ratio", "early_source_chain_group", "continuous_retention_group", "source_chain_blocker"],
            max_rows=30,
        ),
        "## F55-F56 C4 Split-Fisher Source Observability Reset",
        "",
        f"- F55 decision: `{f55_decision.get('decision', '')}`; h800/h1600/h3200/h4800: {f55_decision.get('candidate_h800', '')} / {f55_decision.get('candidate_h1600', '')} / {f55_decision.get('candidate_h3200', '')} / {f55_decision.get('candidate_h4800', '')}",
        f"- F56 decision: `{f56_decision.get('decision', '')}`; h800/h1600/h3200/h4800: {f56_decision.get('candidate_h800', '')} / {f56_decision.get('candidate_h1600', '')} / {f56_decision.get('candidate_h3200', '')} / {f56_decision.get('candidate_h4800', '')}",
        "",
        md_table(
            f55_summary + f56_summary,
            ["carrier", "v22_id", "rows", "blocked_rows", "h100", "h400", "h800", "h1600", "h3200", "h4800", "h1600_retention_ratio", "h3200_retention_ratio", "h4800_retention_ratio", "early_source_chain_group", "continuous_retention_group", "source_chain_blocker"],
            max_rows=20,
        ),
        "## F57-F65 Source-Theory Reset",
        "",
        f"- F57 decision: `{f57_decision.get('decision', '')}`; h800/h1600/h3200/h4800: {f57_decision.get('candidate_h800', '')} / {f57_decision.get('candidate_h1600', '')} / {f57_decision.get('candidate_h3200', '')} / {f57_decision.get('candidate_h4800', '')}",
        f"- F58 decision: `{f58_decision.get('decision', '')}`; h800/h1600/h3200/h4800: {f58_decision.get('candidate_h800', '')} / {f58_decision.get('candidate_h1600', '')} / {f58_decision.get('candidate_h3200', '')} / {f58_decision.get('candidate_h4800', '')}",
        f"- F59 decision: `{f59_decision.get('decision', '')}`; h800/h1600/h3200/h4800: {f59_decision.get('candidate_h800', '')} / {f59_decision.get('candidate_h1600', '')} / {f59_decision.get('candidate_h3200', '')} / {f59_decision.get('candidate_h4800', '')}",
        f"- F60 decision: `{f60_decision.get('decision', '')}`; h800/h1600/h3200/h4800: {f60_decision.get('candidate_h800', '')} / {f60_decision.get('candidate_h1600', '')} / {f60_decision.get('candidate_h3200', '')} / {f60_decision.get('candidate_h4800', '')}",
        f"- F61 decision: `{f61_decision.get('decision', '')}`; h800/h1600/h3200/h4800: {f61_decision.get('candidate_h800', '')} / {f61_decision.get('candidate_h1600', '')} / {f61_decision.get('candidate_h3200', '')} / {f61_decision.get('candidate_h4800', '')}",
        f"- F62 decision: `{f62_decision.get('decision', '')}`; h800/h1600/h3200/h4800: {f62_decision.get('candidate_h800', '')} / {f62_decision.get('candidate_h1600', '')} / {f62_decision.get('candidate_h3200', '')} / {f62_decision.get('candidate_h4800', '')}",
        f"- F63 decision: `{f63_decision.get('decision', '')}`; h800/h1600/h3200/h4800: {f63_decision.get('candidate_h800', '')} / {f63_decision.get('candidate_h1600', '')} / {f63_decision.get('candidate_h3200', '')} / {f63_decision.get('candidate_h4800', '')}",
        f"- F64 decision: `{f64_decision.get('decision', '')}`; h800/h1600/h3200/h4800: {f64_decision.get('candidate_h800', '')} / {f64_decision.get('candidate_h1600', '')} / {f64_decision.get('candidate_h3200', '')} / {f64_decision.get('candidate_h4800', '')}",
        f"- F65 decision: `{f65_decision.get('decision', '')}`; h800/h1600/h3200/h4800: {f65_decision.get('candidate_h800', '')} / {f65_decision.get('candidate_h1600', '')} / {f65_decision.get('candidate_h3200', '')} / {f65_decision.get('candidate_h4800', '')}",
        "",
        md_table(
            f57_summary + f58_summary + f59_summary + f60_summary + f61_summary + f62_summary + f63_summary + f64_summary + f65_summary,
            ["carrier", "v22_id", "rows", "blocked_rows", "h100", "h400", "h800", "h1600", "h3200", "h4800", "h1600_retention_ratio", "h3200_retention_ratio", "h4800_retention_ratio", "early_source_chain_group", "continuous_retention_group", "source_chain_blocker"],
            max_rows=20,
        ),
        "## F66 KAN C3 Readout-Only Source Writer",
        "",
        f"- F66 decision: `{f66_decision.get('decision', '')}`; h800/h1600/h3200/h4800: {f66_decision.get('candidate_h800', '')} / {f66_decision.get('candidate_h1600', '')} / {f66_decision.get('candidate_h3200', '')} / {f66_decision.get('candidate_h4800', '')}",
        "",
        md_table(
            f66_summary,
            ["carrier", "v22_id", "rows", "blocked_rows", "h100", "h400", "h800", "h1600", "h3200", "h4800", "h1600_retention_ratio", "h3200_retention_ratio", "h4800_retention_ratio", "early_source_chain_group", "continuous_retention_group", "source_chain_blocker"],
            max_rows=20,
        ),
        "## F67 KAN C3 Low-Bank B3-Null Source Writer",
        "",
        f"- F67 decision: `{f67_decision.get('decision', '')}`; h800/h1600/h3200/h4800: {f67_decision.get('candidate_h800', '')} / {f67_decision.get('candidate_h1600', '')} / {f67_decision.get('candidate_h3200', '')} / {f67_decision.get('candidate_h4800', '')}",
        "",
        md_table(
            f67_summary,
            ["carrier", "v22_id", "rows", "blocked_rows", "h100", "h400", "h800", "h1600", "h3200", "h4800", "h1600_retention_ratio", "h3200_retention_ratio", "h4800_retention_ratio", "early_source_chain_group", "continuous_retention_group", "source_chain_blocker"],
            max_rows=20,
        ),
        "## F68 AdamW-Boundary Low-Bank B3-Null Source Writer",
        "",
        f"- F68 decision: `{f68_decision.get('decision', '')}`; h800/h1600/h3200/h4800: {f68_decision.get('candidate_h800', '')} / {f68_decision.get('candidate_h1600', '')} / {f68_decision.get('candidate_h3200', '')} / {f68_decision.get('candidate_h4800', '')}",
        "",
        md_table(
            f68_summary,
            ["carrier", "v22_id", "rows", "blocked_rows", "h100", "h400", "h800", "h1600", "h3200", "h4800", "h1600_retention_ratio", "h3200_retention_ratio", "h4800_retention_ratio", "early_source_chain_group", "continuous_retention_group", "source_chain_blocker"],
            max_rows=20,
        ),
        "## F69 AdamW Low-Bank Anchor Anti-Washout",
        "",
        f"- F69 decision: `{f69_decision.get('decision', '')}`; h800/h1600/h3200/h4800: {f69_decision.get('candidate_h800', '')} / {f69_decision.get('candidate_h1600', '')} / {f69_decision.get('candidate_h3200', '')} / {f69_decision.get('candidate_h4800', '')}",
        "",
        md_table(
            f69_summary,
            ["carrier", "v22_id", "rows", "blocked_rows", "h100", "h400", "h800", "h1600", "h3200", "h4800", "h1600_retention_ratio", "h3200_retention_ratio", "h4800_retention_ratio", "early_source_chain_group", "continuous_retention_group", "source_chain_blocker"],
            max_rows=20,
        ),
        "## F70 Terminal Positive Lookahead Floor",
        "",
        f"- F70 decision: `{f70_decision.get('decision', '')}`; h800/h1600/h3200/h4800: {f70_decision.get('candidate_h800', '')} / {f70_decision.get('candidate_h1600', '')} / {f70_decision.get('candidate_h3200', '')} / {f70_decision.get('candidate_h4800', '')}",
        "",
        md_table(
            f70_summary,
            ["carrier", "v22_id", "rows", "blocked_rows", "h100", "h400", "h800", "h1600", "h3200", "h4800", "h1600_retention_ratio", "h3200_retention_ratio", "h4800_retention_ratio", "early_source_chain_group", "continuous_retention_group", "source_chain_blocker"],
            max_rows=20,
        ),
        "## F71 Terminal H3200 Checkpoint Reentry",
        "",
        f"- F71 decision: `{f71_decision.get('decision', '')}`; h800/h1600/h3200/h4800: {f71_decision.get('candidate_h800', '')} / {f71_decision.get('candidate_h1600', '')} / {f71_decision.get('candidate_h3200', '')} / {f71_decision.get('candidate_h4800', '')}",
        "",
        md_table(
            f71_summary,
            ["carrier", "v22_id", "rows", "blocked_rows", "h100", "h400", "h800", "h1600", "h3200", "h4800", "h1600_retention_ratio", "h3200_retention_ratio", "h4800_retention_ratio", "early_source_chain_group", "continuous_retention_group", "source_chain_blocker"],
            max_rows=20,
        ),
        "## F72 Terminal Hard-Split Source",
        "",
        f"- F72 decision: `{f72_decision.get('decision', '')}`; h800/h1600/h3200/h4800: {f72_decision.get('candidate_h800', '')} / {f72_decision.get('candidate_h1600', '')} / {f72_decision.get('candidate_h3200', '')} / {f72_decision.get('candidate_h4800', '')}",
        "",
        md_table(
            f72_summary,
            ["carrier", "v22_id", "rows", "blocked_rows", "h100", "h400", "h800", "h1600", "h3200", "h4800", "h1600_retention_ratio", "h3200_retention_ratio", "h4800_retention_ratio", "early_source_chain_group", "continuous_retention_group", "source_chain_blocker"],
            max_rows=20,
        ),
        "## F73 Terminal AdamW Lookahead",
        "",
        f"- F73 decision: `{f73_decision.get('decision', '')}`; h800/h1600/h3200/h4800: {f73_decision.get('candidate_h800', '')} / {f73_decision.get('candidate_h1600', '')} / {f73_decision.get('candidate_h3200', '')} / {f73_decision.get('candidate_h4800', '')}",
        "",
        md_table(
            f73_summary,
            ["carrier", "v22_id", "rows", "blocked_rows", "h100", "h400", "h800", "h1600", "h3200", "h4800", "h1600_retention_ratio", "h3200_retention_ratio", "h4800_retention_ratio", "early_source_chain_group", "continuous_retention_group", "source_chain_blocker"],
            max_rows=20,
        ),
        "## F74 Terminal Optimizer Selector",
        "",
        f"- F74 decision: `{f74_decision.get('decision', '')}`; h800/h1600/h3200/h4800: {f74_decision.get('candidate_h800', '')} / {f74_decision.get('candidate_h1600', '')} / {f74_decision.get('candidate_h3200', '')} / {f74_decision.get('candidate_h4800', '')}",
        "",
        md_table(
            f74_summary,
            ["carrier", "v22_id", "rows", "blocked_rows", "h100", "h400", "h800", "h1600", "h3200", "h4800", "h1600_retention_ratio", "h3200_retention_ratio", "h4800_retention_ratio", "early_source_chain_group", "continuous_retention_group", "source_chain_blocker"],
            max_rows=20,
        ),
        "## F53/F70/F71/F72/F73/F74 Terminal Recovery Row-Level Localization",
        "",
        md_table(
            terminal_compare_rows,
            ["v22_id", "dataset", "seed", "source_h800", "source_h1600", "source_h3200", "source_h4800", "early_source_chain", "continuous_retention_chain", "source_chain_blocker"],
            max_rows=80,
        ),
        "## 修改记录",
        "",
        "- 新增 `dgkan/fu/source_chain.py`：实现 v22 early-source / continuous-retention / late-rebound 公式和单元测试。",
        "- 新增 `dgkan/profiling/efficiency_v22.py`：实现 v22 E1/S1/E2 efficiency gates。",
        "- 新增 v22 runner：S0.7 truth gate、efficiency wrapper、source-chain dynamics wrapper、MLP/KAN/target wrapper、finalizer。",
        "- 新增 `experiments/run_v22_source_observability_audit.py`：离线审计 train-only / audit-readback / forbidden source-readback predictors，结果只作为 C4 no-go 或 held-out selector 候选，不直接 promotion。",
        "- 新增 `M87-AdamWBoundaryThenMomentumSourceFU` 与 `MLP-F40-adamw-boundary-to-momentum-source`：前 400 step 使用 AdamW boundary 对齐 h100/h400，之后切换到 M2 momentum source，用来检验 early-phase mismatch 是否可修复。",
        "- 新增 `M88-AdamWBoundaryThenDualTimescaleSourceFU` 与 `MLP-F41-adamw-boundary-to-dual-timescale-source`：在 F40 打开 early chain 后加入 short/long slow-state retention gate，检验 h3200/h4800 washout 是否可修复。",
        "- 新增 `M89-AdamWBoundaryDualTimescaleSGDFloorFU` 与 `MLP-F42-adamw-boundary-dual-timescale-sgd-floor-source`：在 F41 short/long source-state gate 后保留 SGD floor，检验 h4800 断链是否来自后段优化落后 controls。",
        "- 新增 `M90-AdamWBoundaryDualTimescaleLateSGDFloorFU` 与 `MLP-F43-adamw-boundary-dual-timescale-late-sgd-floor-source`：保持 F41 到 h3200，再开启 SGD floor，检验 F41 h4800 失败是否来自 h3200 之后缺少 optimizer progress。",
        "- 新增 `M91-AdamWBoundaryDualTimescaleTinyLateSGDFloorFU` 与 `MLP-F44-adamw-boundary-dual-timescale-tiny-late-sgd-floor-source`：h3200 后只使用 0.1x train-gradient floor，检验 full SGD floor 过强导致的 source 破坏。",
        "- 新增 `M92-AdamWBoundaryDualTimescaleReinforcedTinyLateSGDFloorFU` 与 `MLP-F45-adamw-boundary-dual-timescale-reinforced-tiny-late-sgd-floor-source`：在 F44 的 tiny floor 上把 gated slow-source residual 提到 0.5x FU，检验 h3200 ratio 是否只是 source residual 强度不足。",
        "- 新增 `experiments/run_v22_f46_train_loss_selector_holdout.py`：用 F40-F43 phase-reset rows 拟合 train-loss selector 阈值，并在 F44/F45 held-out rows 上验证 selected subgroup 是否形成 continuous/h4800 source chain；该审计不直接 promotion，只生成下一轮 repair hypothesis。",
        "- 新增 `M93-TrainLossGatedDualTimescaleTinyLateFU` 与 `MLP-F47-trainloss-gated-dual-timescale-tiny-late-source`：把 F46 学到的 train-loss threshold 变成在线 gate；h800 后 gate 不通过时回退 SGD，gate 通过时才继续 F44-style tiny late retention。",
        "- 新增 `M94-TrainLossGatedDualTimescaleHoldFallbackFU` 与 `MLP-F48-trainloss-gated-dual-timescale-hold-fallback-source`：针对 F47 的 reject-row fallback washout，h800 后 train-loss gate 不通过时不再执行 SGD fallback，而是 hold 参数；gate 通过时保留 tiny late floor 与 slow-source residual。",
        "- 新增 `M95-TrainLossGatedDualTimescaleTinyLateFallbackFU` 与 `MLP-F49-trainloss-gated-dual-timescale-tiny-late-fallback-source`：针对 F48 的长期 hold 后 h4800 仍输 controls，reject rows 在 h3200 前 hold，h3200 后只执行 0.1x tiny train-gradient floor，不提交 slow-source residual。",
        "- 新增 `M96-TrainLossGatedDualTimescaleBoostedTinyLateFallbackFU` 与 `MLP-F50-trainloss-gated-dual-timescale-boosted-tiny-late-fallback-source`：针对 F49 h4800 仍略负，reject rows 在 h3200 后使用 0.2x tiny train-gradient floor；selected rows 的 source path 不变。",
        "- 新增 `M97-TrainLossLateHoldRecoveryFU` / `MLP-F51-trainloss-late-hold-recovery-source`：h3200 后默认 hold，仅当 train loss 相对 h800 gate 明显恶化时提交 0.05x micro floor，检验 h4800 失败是否来自后段扰动。",
        "- 新增 `M98-TrainLossLateLookaheadFloorFU` / `MLP-F52-trainloss-late-lookahead-floor-source`：h3200 后用 train split A/B 与 corrupt split 做 micro-floor precommit gate，只提交 train-only 即时有益且不明显帮助 corrupt 的恢复步。",
        "- 新增 `M99-TrainLossTerminalLookaheadFloorFU` / `MLP-F53-trainloss-terminal-lookahead-floor-source`：h3200-h4000 保持 source，最后 800 step 才允许 train-split lookahead micro-floor，分离 source 保存与末端 SGD catch-up。",
        "- 新增 `M100-TrainLossEarlyTerminalLookaheadFloorFU` / `MLP-F54-trainloss-early-terminal-lookahead-floor-source`：把 terminal catch-up 提前到 step 3600，检验 F53 near-miss 是否只是 terminal phase 太短。",
        "- 新增 v22 F55/F56 official specs：复用既有 `M37-SplitFisherAgreementSlowFU` 与 `M39-MomentumWarmSplitFisherFU`，把 C4 的 split-gradient / Fisher-normalized / corrupt-filter source observability reset 纳入 fresh h4800 matrix。",
        "- 新增 `M101-AdamWBoundaryDualTimescaleAntiWashoutFU` / `MLP-F57-adamw-boundary-dual-timescale-antiwashout-source`：复用 F40/F41 early source 起点，h1200 后用 train-only source-axis anti-washout projection 继续优化，检验长期 optimizer 是否反向冲掉 source。",
        "- 新增 `M102-AdamWBoundaryDualTimescaleSourceAnchorFU` / `MLP-F58-adamw-boundary-dual-timescale-source-anchor`：复用 F40/F41 early source 起点，h1200 后保持 optimizer progress 并在 train-only/corrupt gate 通过时提交 source-anchor residual，检验 retained target anchor 是否优于 terminal floor。",
        "- 新增 `M103-AdamWBoundaryDualTimescaleParamEMAReentryFU` / `MLP-F59-adamw-boundary-dual-timescale-param-ema-reentry`：复用 F40/F41 early source 起点，source warm 阶段维护参数 EMA，h3200 后只在 train-only/corrupt gate 通过时向 EMA 小幅回投，检验 schedule-free/fast-slow 权重轨迹是否能保留 terminal source。",
        "- 新增 `M104-AdamWBoundaryDualTimescaleReadoutChannelFU` / `MLP-F60-adamw-boundary-dual-timescale-readout-channel`：复用 F40/F41 early source 起点，h1200 后只把 short/long agreed source 投影到 readout/classifier channel，并通过 train A/B + corrupt gate 提交，检验 retained target 是否应由低容量 readout channel 承载。",
        "- 新增 `M105-AdamWBoundaryDualTimescaleHiddenMatrixChannelFU` / `MLP-F61-adamw-boundary-dual-timescale-hidden-matrix-channel`：复用 F40/F41 early source 起点，h1200 后只把 short/long agreed source 投影到 hidden matrix block，并通过 train A/B + corrupt gate 提交，检验 retained target 是否应由隐藏矩阵通道承载。",
        "- 新增 `M106-TrainLossTerminalProjectedLookaheadFloorFU` / `MLP-F62-trainloss-terminal-projected-lookahead-floor-source`：复用 F53 的 terminal lookahead micro-floor，但在提交 terminal gradient 前投掉与 accumulated source-axis 反向的分量，检验 cumulative optimizer projection 是否能保留 h3200 source 并完成 h4800 recovery。",
        "- 新增 `M107-TrainLossTerminalConsensusLookaheadFloorFU` / `MLP-F63-trainloss-terminal-consensus-lookahead-floor-source`：复用 F53 的 terminal recovery 时机，但 terminal floor 改为 train split A/B 共识梯度，并投掉 corrupt-label 方向后用 train/corrupt lookahead gate 提交，检验 C4 E2/E7 noise-reservoir separation 是否能把 F53 near-miss 闭合。",
        "- 新增 `M108-TrainLossTerminalSelectorLookaheadFloorFU` / `MLP-F64-trainloss-terminal-selector-lookahead-floor-source`：保持 F53 的 terminal recovery 时机与尺度不变，用 train split A/B + corrupt-label lookahead 在 raw floor、source-projected floor、cross-split consensus floor 三个合法方向中逐步选择，检验 C4 train-only terminal direction selector 能否闭合 h4800。",
        "- 新增 `MLP-F65-adamw-split-fisher-residual-source`：复用既有 `M38-AdamWSplitFisherAgreementResidualFU`，把 C4 split-Fisher estimator 与 AdamW base/residual 直接组合，检验 F55/F56 未覆盖的 AdamW+Fisher source-observability reset 是否能形成 h4800 retained chain。",
        "- 新增 F66 KAN C3 fresh h4800 复验入口：把既有 `KSW1-basis-estimate-readout-commit` 纳入 v22 KAN scope/default 与 S0.7 合同，并与 `KSW2-lowdegree-lowfreq-source-bank`、controls 同 run 验证 readout-only commit 是否优于 full low-bank writer。",
        "- 新增 F67 KAN C3 fresh h4800 复验入口：用既有 `F36-lowbank-loss-b3-null` / `F38-gain-gated-lowbank-loss-b3-null` 与 controls 检验低阶/低频 source bank 加 B3 reservoir-null 是否能把 KSW2 的 D-CHE late rebound 转成 early/continuous retained source。",
        "- 新增 `M109-AdamWBoundaryToGainGatedLowBankB3NullFU` / `F68-adamw-boundary-to-gated-lowbank-b3-null`：前 400 step 使用 AdamW boundary，之后只在 train split B2 gain 明显优于 B3 safety gain 时提交低阶/低频 B3-null source-channel residual，检验 F67 的 late rebound 是否能被 boundary-conditioned writer 转成 early chain。",
        "- 新增 `M110-AdamWBoundaryLowBankAnchorAntiWashoutFU` / `F69-adamw-boundary-lowbank-anchor-antiwashout`：在 F68 的 accepted low-bank source-state 上加入 AdamW gradient anti-washout projection，检验 D-FOU h800 early chain 是否只是被后段 optimizer 反向冲掉。",
        "- 新增 `M111-TrainLossTerminalPositiveLookaheadFloorFU` / `MLP-F70-trainloss-terminal-positive-lookahead-floor-source`：复用 F53 的 source-preserving terminal recovery，但 terminal floor gate 必须满足 train split A/B 正 gain，并把 corrupt gain allowance 从 0.50 收紧到 0.25，检验 F53 h4800 near-miss 是否来自 terminal gate 过宽。",
        "- 新增 `M112-TrainLossTerminalCheckpointReentryFU` / `MLP-F71-trainloss-terminal-h3200-checkpoint-reentry`：在 h3200 保存 source-bearing 参数检查点，terminal phase 先用 train split A/B + corrupt gate 尝试向 h3200 checkpoint 小幅回灌，若不通过再回退 F53 raw terminal lookahead floor，检验 h4800 断链是否只是 h3200 retained target 被终端漂移冲掉。",
        "- 新增 `M113-TrainLossTerminalHardSplitSourceFU` / `MLP-F72-trainloss-terminal-hard-split-source`：基于 F53/F70/F71 row-level localization，terminal phase 不再使用全样本平均 floor，而只从 train split A/B 高 CE hard examples 构造 split-consensus source，并用 corrupt-label hard split gate 过滤，检验 Fashion-MNIST/KMNIST terminal 失败是否来自 hard-example source channel 被平均 floor 稀释。",
        "- 新增 `M114-TrainLossTerminalAdamWLookaheadFU` / `MLP-F73-trainloss-terminal-adamw-lookahead`：基于 F53 近失败，terminal phase 不再提交 SGD-style raw floor，而用 train split A/B + corrupt gate 预览缩放 AdamW step，检验 Fashion-MNIST/KMNIST h4800 失败是否来自 terminal catch-up optimizer carrier 错配。",
        "- 新增 `M115-TrainLossTerminalOptimizerSelectorFU` / `MLP-F74-trainloss-terminal-optimizer-selector`：在 terminal phase 用 train split A/B + corrupt gate 在 F53 raw floor 与 F73 AdamW-lookahead carrier 间逐步选择，检验 AdamW carrier 是否只需被 train-only selector 局部启用。",
        "- 扩展 `experiments/run_v22_source_observability_audit.py`：新增 C4 E1 split-Fisher SNR、E6 split-Fisher coherence、E7 noise/reservoir separation 三个 train-stream diagnostic composite predictors。",
        "- 修复 source/control attribution key：`experiments/run_v21_01_source_retention.py` 的 matched control key 加入 `run_label`，避免 F40/F41/F42 等 fresh run 在 cumulative merge 时被其他 run 的同 seed/offset controls 污染。",
        "- 扩展 `experiments/run_v22_f40_phase_reset_summary.py`：可按 `run_prefix` / `candidate_prefix` / `artifact_prefix` 写出 F40 或 F41 独立 phase-reset matrix / summary / decision。",
        "- 训练 loop 复用 v21.01 已审计 source-retention runner；v22 wrapper 在独立 official dir 中重算 source-chain 判定，不复用 v21.01 旧结论。",
        "",
        "## 分析 / Insight / 结论",
        "",
        "- v22 的 promotion gate 仍以 grouped source-chain 为准；single-row positive、h1600-only signal、late rebound 或 smoke positive 都不能 promotion。",
        "- 若 source-chain evidence 中 `early_source_chain_group=0`，说明当前尝试连 h100/h400/h800 的连续早期 source 链都没有打开；按计划不能升级 full 或 independent confirmation。",
        "- C4/F46 显示 train-only loss 指标能解释部分 row-level retained source，但这不是 promotion：必须通过真实在线机制把 selected subgroup 的 row-level 信号转成 grouped continuous/h4800 evidence，并排除独立 offset/control contamination。",
        "- F40 若仍不能形成 early chain，说明问题不是简单的 AdamW early boundary/h100-h400 相位错配；若形成 early chain，则只能升级 full h3200/h4800 验证，不能直接 promotion。",
        "- F41 若在 F40 early chain 基础上仍不能通过 h3200/h4800，说明 dual-timescale slow-state gate 没能解决长期 retention washout，不能把 early-chain smoke 写成 retained source。",
        "- F42 若保留 SGD floor 后仍不能通过 h3200/h4800，说明后段优化停滞不是唯一 blocker；如果它保住 h4800，则还必须做 independent offsets 和 controls attribution，不能直接 promotion。",
        "- F43 若 h3200 前保留 F41、h3200 后才加 SGD floor 仍失败，则 blocker 更像 source/optimizer 两相动态不可简单拼接，而不是单纯 floor 时机问题。",
        "- F44 若 tiny late floor 仍失败，则可以关闭简单 floor-scale 修复；继续推进需要新的 train-only source observability，而不是继续调 floor 强度。",
        "- F45 若 reinforced source residual 仍失败，则关闭 phase-reset + dual-timescale + late-floor 这条修复线，避免继续做无审计价值的 scale sweep。",
        "- F46 若 held-out selected subgroup 有 continuous/h4800，说明下一步可以尝试 train-loss gated late-retention repair；如果只停留在 post-hoc selection，仍不能写 breakthrough。",
        "- F47 若不能把离线 selected subgroup 变成全 grouped continuous/h4800，说明 train-loss selector 主要是 failure taxonomy，而不是足够强的在线 source-retention mechanism。",
        "- F48 若 hold fallback 仍不能形成 grouped continuous/h4800，说明 F47 失败不只是 reject rows 被 SGD fallback 冲刷；train-loss selector 仍无法覆盖全 dataset/seed grouped source-retention dynamics。",
        "- F49 若 tiny late fallback 仍不能形成 grouped continuous/h4800，说明 rejected rows 需要的不只是 h3200 后少量 optimizer progress；当前 train-loss gate 不能把 high-loss rows 纳入 retained source chain。",
        "- F50 是本轮对 rejected-row late floor 的最后定向强度验证；若 h4800 仍不过线，应关闭 train-loss gate + late-floor 修复线，避免继续做无审计价值的 scale sweep。",
        "- F51 说明纯 late hold 略优于 F49 但仍 h4800 负；F52 说明全 late-phase train split lookahead micro-floor 反而退化。",
        "- F53 是当前最强 late-phase repair，h4800=-0.003963543309105767，仍未达到 `>=0.005`；F54 提前 terminal catch-up 退化，说明不是简单把 terminal window 加长即可闭合。",
        "- F55/F56 是按 C4 进行的 source observability reset：若仍不能超过 F53 并形成 h4800 retained chain，说明 split-Fisher / noise-reservoir separation 也不足以把 early source 保存到 terminal horizon。",
        "- F57/F58 是 F55/F56 后的 source-theory reset，不属于 terminal boundary 小扫；若仍不能形成 h4800 retained chain，说明当前 train-only source-axis anti-washout / anchor residual 仍不足以把 F40/F41 的 early source 变成 terminal retained source。",
        "- F59 是 schedule-free/fast-slow iterate 假设的参数轨迹回投测试；若仍不能形成 h4800 retained chain，说明仅把 early source 轨迹保存为参数 EMA 也不足以抵抗 terminal washout。",
        "- F60 是 representation/channel reset：若 readout-only source channel 仍不能形成 h4800 retained chain，说明当前 MLP retained target 不只是全参数 carrier 错，而是 source-target definition 本身仍不够。",
        "- F61 是 hidden-matrix channel reset：若 hidden matrix block 仍不能形成 h4800 retained chain，说明 matrix-block carrier 也不足以承接当前 source target，后续需要重新定义 train-only source observability/retained target，而不是继续小扫通道强度。",
        "- F62 是 source-preserving terminal recovery：若 terminal gradient 投影仍不能形成 h4800 retained chain，说明 F53 near-miss 不是简单的 terminal optimizer/source-axis 冲突，不能继续靠 terminal floor 小修写 breakthrough。",
        "- F63 是 C4 terminal consensus recovery：若 split-consensus/noise-reservoir terminal floor 仍不能形成 h4800 retained chain，说明当前 train-only cross-split estimator 也不足以把 h3200 retained source 转成 h4800 source。",
        "- F64 是 C4 terminal selector recovery：若 train-only lookahead 在 raw/projected/consensus 三个合法方向之间选择仍不能形成 h4800 retained chain，说明当前 terminal 方向选择器不足以修复 source-retention 断链，后续应转向新的 retained target/source observability theory，而不是继续围绕 terminal floor 做局部小扫。",
        "- F65 是非 terminal 的 C4 AdamW split-Fisher residual reset：若它仍不能形成 h4800 retained chain，说明把 split-Fisher source estimator 直接接到 AdamW base/residual 也不足以修复 source observability/retention 断链。",
        "- F66 是 C3 representation/channel reset：若 KAN readout-only commit 仍不能形成 early/continuous/h4800 retained source，说明当前 KAN low-bank/readout channel 也没有承载 v22 retained source，不应把 KAN writer smoke 或 efficiency pass 写成 functional breakthrough。",
        "- F67 是 C3 reservoir-null reset：若低阶/低频 source bank + B3 null 仍不能形成 early/continuous/h4800 retained source，说明 KSW2 的 D-CHE h4800 positive 仍只是 delayed migration/late rebound，而不是 source-channel writer breakthrough。",
        "- F68 是 C3 boundary-conditioned source-channel writer reset：若 AdamW boundary + gain-gated low-bank B3-null 仍不能形成 early/continuous/h4800 retained source，说明当前 low-bank source-channel target 与 optimizer boundary 的组合也不能把 late migration 转成 retained source。",
        "- F69 是 F68 后的 source-state anti-washout 验证：若 accepted low-bank source-state + AdamW anti-source projection 仍不能闭合 h1600/h3200/h4800，说明当前 blocker 不只是后段 AdamW 梯度反向冲刷。",
        "- F70 是 F53 near-miss 后的 terminal gate 收紧验证：若 positive-only terminal lookahead 仍不能让 h4800 达到 `>=0.005`，说明 F53 失败不是单纯由 terminal gate 允许微负 train-split gain 导致。",
        "- F71 是 h3200 retained target 参数检查点回灌验证：若 checkpoint reentry 仍不能让 h4800 达到 `>=0.005`，说明 blocker 不只是 h3200 后参数漂移遗失 source-bearing target；当前需要重新定义 train-only source observability 或 retained target，而不是继续围绕 terminal 回灌做局部小扫。",
        "- F72 是困难样本 source theory reset：若 hard-split terminal source 仍不能让 h4800 达到 `>=0.005`，说明 Fashion-MNIST/KMNIST 的 terminal blocker 不只是全样本 floor 稀释 hard-example source channel；下一步需要更上游的 train-only source observability theory，而不是继续在 terminal phase 调方向或尺度。",
        "- F73 是 terminal optimizer-carrier reset：若 AdamW-lookahead terminal catch-up 仍不能让 h4800 达到 `>=0.005`，说明 F53 近失败不是简单因为 SGD-style terminal floor 输给 AdamW/SGD controls；当前 h4800 blocker 更可能在 source observability/retained target definition，而不是 terminal optimizer carrier。",
        "- F74 是 terminal carrier selector reset：若 raw-vs-AdamW train-only selector 仍不能让 h4800 达到 `>=0.005`，说明即使允许合法 online carrier selection，当前 terminal phase 也无法把 h3200 retained source 变成 h4800 retained source；应停止 terminal selector/carrier 局部修复线。",
        "- 因此当前不能把 F53 near-miss 或 F55/F56 的局部 positive 写成 breakthrough；下一步若继续，不应继续小扫 terminal boundary，而应转向新的 source observability / retained target theory。",
        "- 若 efficiency S1 有 rows 但 functional source-chain 未闭合，主 blocker 是 functional source retention，而不是 kernel officialization。",
        "- 当前 `promotion_allowed=0` 表示真实证据链未闭合，不是预算性放弃；后续只能从新的 train-only early-source observability 或 target-to-retention theory 继续。",
    ]
    write_text(V22_RECAP_DOC, "\n".join(text) + "\n")


def main() -> None:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir)
    init_docs()
    append_exec(out_dir, f"{PYTHON} experiments/run_v22_merge_finalize.py", status="started")
    decision = summarize(out_dir)
    render_recap(out_dir, decision)
    build_packet(out_dir)
    for name in ["v22_code_review_packet.zip", "v22_results_bundle.zip"]:
        path = out_dir / name
        if path.exists():
            append_text(V22_RECAP_DOC, f"\n- {name}: size={path.stat().st_size} sha256={sha256_file(path)}\n")
    append_exec(out_dir, f"{PYTHON} experiments/run_v22_merge_finalize.py", status="completed", note=f"route={decision.get('route')} promotion_allowed={decision.get('promotion_allowed')}")


if __name__ == "__main__":
    main()
