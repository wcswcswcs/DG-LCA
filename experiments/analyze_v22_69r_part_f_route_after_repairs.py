#!/usr/bin/env python3
"""Summarize v22.69R Part F target-free repair attempts into a route artifact."""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import experiments.run_v22_69r_mlp_winner_projected_quotient_kan_carrier_audit as r69  # noqa: E402


OUT = ROOT / "results/v22_69R"
CHUNK_ROOT = ROOT / "results/v22_66/chunks"
MIXED_ARCH = "DGKAN_MIXED_FOU4LIN_HAT4XLIN"
SUMMARIES = [
    (
        "baseline_fixmetrics",
        OUT / "v22_69R_part_f_target_free_mixedbank_summary.json",
    ),
    (
        "source_witness_task_visible_126",
        OUT / "v22_69R_part_f_target_free_mixedbank_v22_69R_part_f_126_estimator_repair_st30_summary.json",
    ),
    (
        "fixed_mlpwinner_residual_126",
        OUT / "v22_69R_part_f_target_free_mixedbank_v22_69R_part_f_126_mlpwinner_residual_st30_summary.json",
    ),
    (
        "fixed_mlpwinner_residual_active_metrics_st30",
        OUT / "v22_69R_part_f_target_free_mixedbank_v22_69R_part_f_active_metrics_mlpwinner_residual_st30_summary.json",
    ),
    (
        "fixed_mlpwinner_residual_active_metrics_st100",
        OUT / "v22_69R_part_f_target_free_mixedbank_v22_69R_part_f_active_metrics_mlpwinner_residual_st100_summary.json",
    ),
    (
        "gradcoh_source_witness_active_metrics_st30",
        OUT / "v22_69R_part_f_target_free_mixedbank_v22_69R_part_f_active_metrics_gradcoh_consensus_st30_summary.json",
    ),
    (
        "ooc_poet_pion_bankbudget_st30",
        OUT / "v22_69R_part_f_target_free_mixedbank_v22_69R_part_f_ooc_poet_pion_bankbudget_st30_summary.json",
    ),
    (
        "warm50_residual_old_full_coordinate_st100",
        OUT / "v22_69R_part_f_target_free_mixedbank_v22_69R_part_f_warm50_residual_active_metrics_st100_summary.json",
    ),
    (
        "taskvisible_warm50_bankbudget_st100",
        OUT / "v22_69R_part_f_target_free_mixedbank_v22_69R_part_f_taskvisible_warm50_bankbudget_st100_summary.json",
    ),
    (
        "warm50_residual_active_delta_st100",
        OUT / "v22_69R_part_f_target_free_mixedbank_v22_69R_part_f_warm50_residual_active_delta_metrics_st100_summary.json",
    ),
    (
        "warm50_actgrad_cvar_active_delta_st100",
        OUT / "v22_69R_part_f_target_free_mixedbank_v22_69R_part_f_warm50_actgrad_debt_active_delta_st100_summary.json",
    ),
    (
        "warm80_residual_active_delta_st100",
        OUT / "v22_69R_part_f_target_free_mixedbank_v22_69R_part_f_warm80_residual_active_delta_st100_summary.json",
    ),
    (
        "warm50_loweta_tail_active_delta_st100",
        OUT / "v22_69R_part_f_target_free_mixedbank_v22_69R_part_f_warm50_loweta_tail_active_delta_st100_summary.json",
    ),
    (
        "debtbudget50_active_delta_st100",
        OUT / "v22_69R_part_f_target_free_mixedbank_v22_69R_part_f_debtbudget50_active_delta_st100_summary.json",
    ),
    (
        "debtbudget80_active_delta_st100",
        OUT / "v22_69R_part_f_target_free_mixedbank_v22_69R_part_f_debtbudget80_active_delta_st100_summary.json",
    ),
]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def compact_summary(label: str, path: Path) -> dict[str, Any]:
    s = read_json(path)
    return {
        "label": label,
        "path": str(path.relative_to(ROOT)),
        "run_label": s.get("run_label", ""),
        "completed_rows": s.get("completed_rows", 0),
        "total_rows_collected": s.get("total_rows_collected", 0),
        "part_f_exploration_gate_pass": s.get("part_f_exploration_gate_pass", 0),
        "official_candidate_gate_pass": s.get("official_candidate_gate_pass", 0),
        "final_route": s.get("final_route", ""),
        "failure_components": s.get("failure_components", []),
        "KAN_improves_own_rows": s.get("KAN_improves_own_rows", 0),
        "KAN_beats_MLP_matched_rows": s.get("KAN_beats_MLP_matched_rows", 0),
        "KAN_beats_best_KAN_control_rows": s.get("KAN_beats_best_KAN_control_rows", 0),
        "KAN_beats_same_generator_controls_rows": s.get("KAN_beats_same_generator_controls_rows", 0),
        "TrueKANGain_plus_BothGain_rows": s.get("TrueKANGain_plus_BothGain_rows", 0),
        "no_debt_rows": s.get("no_debt_rows", 0),
        "readout_visible_CVaR25_ge_025_rows": s.get("readout_visible_CVaR25_ge_025_rows", 0),
        "control_contrastive_margin_p10_positive_rows": s.get("control_contrastive_margin_p10_positive_rows", 0),
        "basis_Gram_condition_pass_rows": s.get("basis_Gram_condition_pass_rows", 0),
        "functional_spectrum_drift_le_threshold_rows": s.get("functional_spectrum_drift_le_threshold_rows", 0),
        "mean_Delta_NLL_vs_KAN_own_strong_optimizer": s.get("mean_Delta_NLL_vs_KAN_own_strong_optimizer", ""),
        "mean_Delta_NLL_vs_MLP_matched_coordinate": s.get("mean_Delta_NLL_vs_MLP_matched_coordinate", ""),
        "mean_Delta_NLL_vs_best_KAN_control": s.get("mean_Delta_NLL_vs_best_KAN_control", ""),
        "readout_CVaR25_p10": s.get("readout_CVaR25_p10", ""),
        "control_margin_p10": s.get("control_margin_p10", ""),
        "candidate_methods": s.get("candidate_methods", []),
        "control_methods": s.get("control_methods", []),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(row.get(key), ensure_ascii=False) if isinstance(row.get(key), (list, dict)) else row.get(key, "") for key in keys})


def as_int(row: dict[str, Any], key: str) -> int:
    value = row.get(key, 0)
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def flt(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def finite_sum(row: dict[str, Any], keys: list[str]) -> float:
    values = [flt(row.get(key)) for key in keys]
    if not all(math.isfinite(v) for v in values):
        return float("nan")
    return sum(values)


def best_run_debt_decomposition(run_label: str) -> dict[str, Any]:
    """Recompute ECE+Brier+tail_q95 debt against best own reference from raw chunks."""
    if not run_label:
        return {"available": 0, "reason": "empty_run_label"}
    files = sorted(CHUNK_ROOT.glob(f"{run_label}_{MIXED_ARCH}_*_st100.csv"))
    if not files:
        files = sorted(CHUNK_ROOT.glob(f"{run_label}_{MIXED_ARCH}_*_st30.csv"))
    rows: list[dict[str, Any]] = []
    for path in files:
        with path.open(newline="", encoding="utf-8") as f:
            rows.extend(csv.DictReader(f))
    refs: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    candidates: list[dict[str, Any]] = []
    for row in rows:
        if row.get("run_status") != "completed":
            continue
        if row.get("method_family") == "reference":
            refs[(str(row.get("dataset")), str(row.get("seed")))].append(row)
        elif row.get("method_family") == "candidate":
            candidates.append(row)

    dataset_counts: Counter[str] = Counter()
    dataset_failures: Counter[str] = Counter()
    method_failures: Counter[str] = Counter()
    component_failures: Counter[str] = Counter()
    top_gaps: list[dict[str, Any]] = []
    pass_count = 0
    fail_count = 0
    for row in candidates:
        key = (str(row.get("dataset")), str(row.get("seed")))
        ref_rows = refs.get(key, [])
        ref_debts = [(finite_sum(ref, ["ECE", "Brier", "tail_loss_q95"]), ref) for ref in ref_rows]
        ref_debts = [(debt, ref) for debt, ref in ref_debts if math.isfinite(debt)]
        if not ref_debts:
            continue
        ref_debt, ref = min(ref_debts, key=lambda item: item[0])
        candidate_debt = finite_sum(row, ["ECE", "Brier", "tail_loss_q95"])
        if not math.isfinite(candidate_debt):
            continue
        dataset = str(row.get("dataset"))
        dataset_counts[dataset] += 1
        if candidate_debt <= ref_debt + 1.0e-9:
            pass_count += 1
            continue

        fail_count += 1
        dataset_failures[dataset] += 1
        method_failures[str(row.get("method"))] += 1
        gaps = {
            "ECE": flt(row.get("ECE")) - flt(ref.get("ECE")),
            "Brier": flt(row.get("Brier")) - flt(ref.get("Brier")),
            "tail_q95": flt(row.get("tail_loss_q95")) - flt(ref.get("tail_loss_q95")),
        }
        positive_components = [name for name, gap in gaps.items() if math.isfinite(gap) and gap > 1.0e-9]
        component_failures.update(positive_components or ["sum_only"])
        top_gaps.append(
            {
                "dataset": dataset,
                "seed": str(row.get("seed")),
                "method": str(row.get("method")),
                "debt_gap": candidate_debt - ref_debt,
                "ece_gap": gaps["ECE"],
                "brier_gap": gaps["Brier"],
                "tail_q95_gap": gaps["tail_q95"],
                "candidate_debt": candidate_debt,
                "best_reference_debt": ref_debt,
                "best_reference_method": str(ref.get("method")),
            }
        )
    top_gaps.sort(key=lambda item: item["debt_gap"], reverse=True)
    tail_positive = sum(1 for item in top_gaps if item["tail_q95_gap"] > 1.0e-9)
    tail_gaps = sorted(item["tail_q95_gap"] for item in top_gaps if math.isfinite(item["tail_q95_gap"]))
    median_tail_gap = tail_gaps[len(tail_gaps) // 2] if tail_gaps else ""
    return {
        "available": 1,
        "run_label": run_label,
        "input_files": len(files),
        "raw_rows": len(rows),
        "candidate_rows_audited": len(candidates),
        "reference_rows_audited": sum(len(v) for v in refs.values()),
        "no_debt_pass_rows": pass_count,
        "no_debt_fail_rows": fail_count,
        "dataset_counts": dict(dataset_counts),
        "dataset_failures": dict(dataset_failures),
        "method_failures": dict(method_failures),
        "component_positive_gap_counts_among_failures": dict(component_failures),
        "tail_q95_positive_gap_failures": tail_positive,
        "median_tail_q95_gap_among_failures": median_tail_gap,
        "top_debt_gaps": top_gaps[:8],
        "debt_definition": "ECE+Brier+tail_loss_q95 <= best own-reference debt for the same dataset/seed.",
    }


def main() -> int:
    rows = []
    missing = []
    for label, path in SUMMARIES:
        if not path.exists():
            missing.append(str(path.relative_to(ROOT)))
            continue
        rows.append(compact_summary(label, path))
    csv_path = OUT / "v22_69R_part_f_after_repairs_route_summary.csv"
    route_path = OUT / "v22_69R_part_h_route_decision_after_part_f_repairs.json"
    write_csv(csv_path, rows)

    best_control = max(
        rows,
        key=lambda r: (
            int(r.get("KAN_beats_best_KAN_control_rows") or 0),
            int(r.get("KAN_beats_same_generator_controls_rows") or 0),
            int(r.get("control_contrastive_margin_p10_positive_rows") or 0),
            int(r.get("basis_Gram_condition_pass_rows") or 0),
            int("active_metrics" in str(r.get("label", ""))),
        ),
        default={},
    )
    full_rows = [r for r in rows if as_int(r, "completed_rows") >= 30]
    best_gate_closure = min(
        full_rows,
        key=lambda r: (
            len(r.get("failure_components") or []),
            -as_int(r, "KAN_improves_own_rows"),
            -as_int(r, "no_debt_rows"),
            -as_int(r, "KAN_beats_MLP_matched_rows"),
            -as_int(r, "KAN_beats_best_KAN_control_rows"),
            -as_int(r, "readout_visible_CVaR25_ge_025_rows"),
            str(r.get("label", "")),
        ),
        default={},
    )
    common_blockers = {
        "KAN_improves_own_rows": [r.get("KAN_improves_own_rows") for r in rows],
        "readout_visible_CVaR25_ge_025_rows": [r.get("readout_visible_CVaR25_ge_025_rows") for r in rows],
        "basis_Gram_condition_pass_rows": [r.get("basis_Gram_condition_pass_rows") for r in rows],
        "KAN_beats_MLP_matched_rows": [r.get("KAN_beats_MLP_matched_rows") for r in rows],
        "no_debt_rows": [r.get("no_debt_rows") for r in rows],
        "failure_components_by_attempt": {r.get("label", ""): r.get("failure_components", []) for r in rows},
    }
    best_debt_decomposition = best_run_debt_decomposition(str(best_gate_closure.get("run_label", "")))
    route = {
        "generated_at_sg": r69.now_sg(),
        "final_route": "R4-KANDiagnosticLiftOpened_TargetFreeFullLoopFailed",
        "route_reason": (
            "Part D diagnostic mixed-bank capacity opened, but all target-free Part F official-eligible "
            "full-loop attempts failed the exploration gate. The active-delta diagnostic repair corrected "
            "Part F readout/basis measurement to use the learned atlas displacement instead of the dormant "
            "full coordinate set, which removed the previous readout/basis blocker. The follow-up "
            "signal/debt-ratio budget repair added a train-only tail-aligned coordinate budget plus a "
            "same-budget random control. The best latest attempt, debtbudget50_active_delta_st100, passed "
            "matched-MLP 21/30, best-control 30/30, same-generator-control 30/30, readout 22/30, basis "
            "30/30, margin 29/30, spectrum 30/30, and ControlExplained 0%, but still failed own-optimizer "
            "13/30 (threshold 16/30) and no-debt 13/30 (threshold 22/30). This improves over the prior "
            "warm50 residual best (own 11/30, no-debt 10/30) but does not close the exploration gate."
        ),
        "part_d_basis": "DGKAN_FOU4_LIN+DGKAN_HAT4_XLIN target-augmented visible diagnostic Part D passed vt025/vt030 fraction gates.",
        "part_f_summary_rows": rows,
        "common_blockers": common_blockers,
        "best_control_resistance_attempt": best_control,
        "best_part_f_gate_closure_attempt": best_gate_closure,
        "best_part_f_gate_closure_debt_decomposition": best_debt_decomposition,
        "promotion_allowed": 0,
        "architecture_claim_ready": 0,
        "official_candidate_gate_pass": 0,
        "missing_inputs": missing,
        "audit_caveats": [
            "The Part F runner is target-free: MLP_target_used_in_runtime=0 and loss_total_is_task_loss_only rows were recorded in each summary.",
            "Some over-POET rows reported POET layer count 0 because mixed-bank readout dimensions were not divisible by the external POET block size; these rows are kept as measured, not discarded.",
            "Part F control_margin capacity diagnostics can contain very large clipped values from ill-conditioned least-squares projection; route does not rely on those large values because margin count did not pass and readout/basis/own/MLP gates failed independently.",
            "The active-metrics reruns replace that clipped least-squares diagnostic with SVD subspace projection and active atlas coordinate design; older summaries are retained for audit history only.",
            "The active-delta reruns further change the official Part F readout/basis diagnostic to weight active atlas coordinates by the learned KAN update delta while preserving full-atlas coordinate metrics under atlas_coordinate_* fields.",
        ],
        "artifacts": {
            "summary_csv": str(csv_path.relative_to(ROOT)),
            "route_json": str(route_path.relative_to(ROOT)),
        },
    }
    r69.write_json(route_path, route)
    r69.append_exec(
        "H_after_part_f_repairs_route",
        r69.command_text([sys.executable, r69.rel(Path(__file__))]),
        "done",
        gpu="cpu",
        files=f"{r69.rel(csv_path)}; {r69.rel(route_path)}",
        note=json.dumps({"final_route": route["final_route"], "summaries": [r["label"] for r in rows], "missing": missing}, ensure_ascii=False),
    )
    r69.append_recap(
        "Part H after Part F repairs route decision",
        [
            f"final_route={route['final_route']}; architecture_claim_ready=0; official_candidate_gate_pass=0.",
            "Baseline fixmetrics Part F stayed R4: improves_own=0/30, beats_MLP=0/30, beats_best_control=12/30, beats_same_generator=14/30, no_debt=3/30, readout=0/30, margin=22/30, basis_cond=0/30.",
            "12.6 source/witness task-visible repair stayed R4: improves_own=0/30, beats_MLP=2/30, beats_best_control=7/30, same-estimator controls=7/30, no_debt=4/30, readout=0/30, margin=12/30, basis_cond=0/30.",
            "Fixed MLP-winner residual repair gave the main partial positive: beats_best_KAN_control=30/30 and beats_same_generator_controls=30/30; however improves_own=0/30, beats_MLP=0/30, no_debt=2/30, readout=0/30, margin=20/30, basis_cond=0/30, so gate still fails.",
            "Active-metrics correction: Part F capacity/readout/basis diagnostics now use active atlas coordinate design plus SVD subspace projection; rerun st30 fixed residual has control=30/30, margin=30/30, basis_cond=20/30, but still improves_own=0/30, beats_MLP=0/30, no_debt=2/30, readout=2/30.",
            "Horizon-undertraining audit: st100 fixed residual did not rescue the gate; improves_own=0/30, beats_MLP=0/30, beats_best_control=10/30, no_debt=6/30, readout=2/30, basis_cond=20/30.",
            "Gradcoh source/witness consensus audit: small local gains appeared but were far below threshold; improves_own=1/30, beats_MLP=4/30, beats_best_control=5/30, no_debt=7/30, readout=0/30, basis_cond=11/30, ControlExplained_pct=83.33.",
            "OOC over-POET/Pion bankbudget audit: matched-MLP wins improved to 10/30 and margin stayed 30/30, but improves_own=0/30, beats_best_control=6/30, beats_same_generator=6/30, no_debt=5/30, readout=1/30, basis_cond=17/30, ControlExplained_pct=80.0; method-level split showed the Wine gains did not transfer to MNIST.",
            "Warm50 residual old full-coordinate diagnostic: improves_own=11/30, beats_MLP=22/30, control=30/30, no_debt=10/30, readout=0/30, basis_cond=10/30; this isolated a diagnostic visibility problem, not a control-resistance problem.",
            "Active-delta diagnostic repair result: warm50_residual_active_delta_st100 is the best gate-closure attempt so far; improves_own=11/30, beats_MLP=22/30, control=30/30, same-controls=30/30, TrueKANGain+BothGain=11/30, no_debt=10/30, readout=22/30, basis_cond=30/30, margin=28/30, spectrum=29/30, ControlExplained_pct=0.0. It fails only KAN_improves_own and no_debt.",
            "Task-visible warm50 bankbudget repair failed solidly: improves_own=3/30, beats_MLP=4/30, control=7/30, no_debt=7/30, readout=0/30, basis_cond=0/30, mean_Delta_NLL_vs_own=5.458282620754411.",
            "Actgrad/cvar debt repair improved no_debt only to 12/30 and lowered readout to 19/30; final blockers were improves_own, no_debt, readout_visible.",
            "Warm80 residual repair rejected the longer-warmup hypothesis: improves_own=7/30, no_debt=8/30, control=27/30, readout=23/30, basis_cond=30/30.",
            "Low-eta warm50 tail repair improved own wins to 13/30, but still failed improves_own, no_debt=11/30, and readout=21/30; offline combination analysis across already-run eta001/eta002/eta005/blend65 rows still could not satisfy no_debt (best observed combined no_debt=12/30).",
            "Signal/debt-ratio budget repair: added train-only `debtbudget50` coordinate scaling and `same_debtbudget_generator_random` control, then ran `v22_69R_part_f_debtbudget50_active_delta_st100`; result improves own/no-debt but still fails gate: improves_own=13/30, beats_MLP=21/30, control=30/30, same-controls=30/30, TrueKANGain+BothGain=13/30, no_debt=13/30, readout=22/30, basis_cond=30/30, margin=29/30, spectrum=30/30, ControlExplained_pct=0.0.",
            "Stronger signal/debt-ratio budget repair did not help: `v22_69R_part_f_debtbudget80_active_delta_st100` kept improves_own=13/30 but reduced no_debt to 12/30 and readout to 21/30; final blockers were improves_own, no_debt, readout_visible.",
            f"Best-run debt decomposition from raw chunks: no_debt_pass={best_debt_decomposition.get('no_debt_pass_rows')}, no_debt_fail={best_debt_decomposition.get('no_debt_fail_rows')}, dataset_failures={best_debt_decomposition.get('dataset_failures')}, component_positive_gap_counts={best_debt_decomposition.get('component_positive_gap_counts_among_failures')}, tail_q95_positive_gap_failures={best_debt_decomposition.get('tail_q95_positive_gap_failures')}, median_tail_q95_gap_among_failures={best_debt_decomposition.get('median_tail_q95_gap_among_failures')}.",
            "Analysis: active-delta metrics show the learned carrier can be readout-visible and control-resistant, so the remaining failure is not the old dormant-coordinate/basis measurement. The failure concentrates in self-optimizer competitiveness and tail-loss debt, especially MNIST rows.",
            "Insight: current blocker is the target-free residual estimator's debt-safe utility, not diagnostic carrier capacity and not random-control explanation. The best run is near the exploration gate structurally, but still short of the plan thresholds, so no architecture claim is justified.",
            "Implementation changes audited: added configurable Part F method lists and run-label-specific artifacts; fixed Part F capacity diagnostics to read MetricCompatibleLowRankLinear.effective_weight; switched Part F diagnostics from raw feature design/lstsq to active atlas coordinate design/SVD projection; changed official Part F readout/basis diagnostics to active-delta-weighted atlas coordinates while preserving atlas_coordinate_* legacy metrics; added fixed mixed-bank trainable base hook for DGKAN_FOU4_LIN+DGKAN_HAT4_XLIN; added task-visible eta002/eta005 warm50 aliases; generalized warmup parsing to warmN and added warm80 aliases; added eta001/eta002 warm50 residual aliases; added `debtbudget50` signal/debt-ratio coordinate budget and `same_debtbudget_generator_random` same-budget control.",
            f"Evidence: `{r69.rel(csv_path)}`, `{r69.rel(route_path)}`.",
        ],
    )
    print(json.dumps(route, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
