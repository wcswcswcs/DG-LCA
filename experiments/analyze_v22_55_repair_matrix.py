#!/usr/bin/env python3
"""Summarize v22.55 diagnostic repair matrices without overwriting official outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


WGO_CANDIDATES = {
    "wgo_diag_snr",
    "wgo_block_lowrank_r4",
    "wgo_safety_diag",
    "wgo_residual_deleak_diag",
    "wgo_residual_safety_diag",
    "wgo_poet",
}
WGO_CONTROLS = {
    "same_gate_distribution_random",
    "same_gate_entropy_random",
    "same_block_norm_random_gate",
    "shuffled_witness_gate",
    "frozen_gate_from_previous_seed",
    "source_only_gate",
    "witness_only_gate",
    "inverse_class_count_gate",
    "hard_loss_gate",
    "loss_rank_gate",
    "random_label_gate",
    "same_compute_noop_gate",
    "margin_shuffle_gate",
}
REWEIGHTING_CONTROLS = {
    "inverse_class_count_gate",
    "hard_loss_gate",
    "loss_rank_gate",
    "margin_shuffle_gate",
    "source_only_gate",
    "witness_only_gate",
}


def fval(value: Any, default: float | None = None) -> float | None:
    try:
        if value in ("", None):
            return default
        out = float(value)
        if math.isnan(out):
            return default
        return out
    except (TypeError, ValueError):
        return default


def iflag(value: Any) -> int:
    return int(str(value).strip().lower() in {"1", "true", "yes"})


def mean(values: list[float | None]) -> float | None:
    clean = [v for v in values if v is not None and not math.isnan(v)]
    return sum(clean) / len(clean) if clean else None


def read_rows(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[Path] = set()
    for path in sorted(paths):
        path = path.resolve()
        if path in seen:
            continue
        seen.add(path)
        with path.open(newline="", encoding="utf-8") as fh:
            rows.extend(csv.DictReader(fh))
    return rows


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def delta_vs(ref_by_key: dict[tuple[str, str, str], dict[str, str]], row: dict[str, str], method: str) -> float | str:
    ref = ref_by_key.get((method, row.get("dataset", ""), str(row.get("seed", ""))))
    final_nll = fval(row.get("final_NLL"))
    ref_nll = fval(ref.get("final_NLL")) if ref else None
    return (final_nll - ref_nll) if final_nll is not None and ref_nll is not None else ""


def classify(candidate_pairwise: list[dict[str, Any]], route: dict[str, Any]) -> str:
    if not candidate_pairwise:
        return "R1-WGOUnitOnly"
    if route["standard_loop_pass_candidate_rows"] != len(candidate_pairwise):
        return "R0-TrainingLoopBoundaryFailed"
    if route["mlp_exploration_pass"]:
        return "R6-MLPWGOExplorationOpened"
    gate_means = [fval(r.get("gate_mean")) for r in candidate_pairwise]
    gate_stds = [fval(r.get("gate_std")) for r in candidate_pairwise]
    if gate_means and (mean(gate_means) or 0.0) >= 0.95 and (mean(gate_stds) or 1.0) <= 0.03:
        return "R2-WGOIdentityCollapse_NoEffectiveFU"
    if any(
        (fval(r.get("gate_class_count_correlation_abs"), 0.0) or 0.0) > 0.80
        or (fval(r.get("gate_loss_correlation_abs"), 0.0) or 0.0) > 0.80
        for r in candidate_pairwise
    ):
        return "R3-ReweightingExplained_NoFU"
    if route["beats_strongest_rows"] == 0 and route["beats_POET_or_external_OET_rows"] == 0:
        return "R5-FUWeakOptimizerPatchOnly"
    if route["beats_best_control_rows"] < 6:
        return "R4-RandomGateOrAuxiliaryExplained_NoFU"
    if route["no_debt_rows"] < 7:
        return "SafetyDebtBlocksWGO"
    return "ExternalOETDominates_CurrentFUInsufficient"


def summarize(rows: list[dict[str, str]], reference_rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    rows = [r for r in rows if r.get("run_status") == "completed"]
    wgo_rows = [r for r in rows if r.get("phase") == "WGO"]
    ref_by_key = {(r.get("method", ""), r.get("dataset", ""), str(r.get("seed", ""))): r for r in reference_rows}
    strongest_by_key: dict[tuple[str, str], dict[str, str]] = {}
    for row in reference_rows:
        key = (row.get("dataset", ""), str(row.get("seed", "")))
        current = strongest_by_key.get(key)
        if current is None or (fval(row.get("final_NLL"), float("inf")) or float("inf")) < (
            fval(current.get("final_NLL"), float("inf")) or float("inf")
        ):
            strongest_by_key[key] = row

    pairwise: list[dict[str, Any]] = []
    for row in wgo_rows:
        ref_method = row.get("reference_method", "")
        key = (row.get("dataset", ""), str(row.get("seed", "")))
        ref = ref_by_key.get((ref_method, row.get("dataset", ""), str(row.get("seed", ""))))
        poet = ref_by_key.get(("poet_official", row.get("dataset", ""), str(row.get("seed", ""))))
        strongest = strongest_by_key.get(key)
        controls = [
            r
            for r in wgo_rows
            if r.get("dataset") == row.get("dataset")
            and str(r.get("seed")) == str(row.get("seed"))
            and r.get("reference_method") == ref_method
            and r.get("method") in WGO_CONTROLS
        ]
        reweight_controls = [r for r in controls if r.get("method") in REWEIGHTING_CONTROLS]
        best_control = min(controls, key=lambda r: fval(r.get("final_NLL"), float("inf")) or float("inf"), default=None)
        best_reweight = min(reweight_controls, key=lambda r: fval(r.get("final_NLL"), float("inf")) or float("inf"), default=None)
        final_nll = fval(row.get("final_NLL"))
        ref_nll = fval(ref.get("final_NLL")) if ref else None
        poet_nll = fval(poet.get("final_NLL")) if poet else None
        strongest_nll = fval(strongest.get("final_NLL")) if strongest else None
        control_nll = fval(best_control.get("final_NLL")) if best_control else None
        reweight_nll = fval(best_reweight.get("final_NLL")) if best_reweight else None
        pairwise.append(
            {
                "dataset": row.get("dataset", ""),
                "seed": row.get("seed", ""),
                "method": row.get("method", ""),
                "control_kind": row.get("control_kind", ""),
                "reference_method": ref_method,
                "reference_final_NLL": ref_nll,
                "poet_final_NLL": poet_nll,
                "strongest_method": strongest.get("method", "") if strongest else "",
                "strongest_final_NLL": strongest_nll,
                "final_NLL": final_nll,
                "Delta_NLL_vs_AdamW": delta_vs(ref_by_key, row, "adamw"),
                "Delta_NLL_vs_Cautious": delta_vs(ref_by_key, row, "cautious_adamw"),
                "Delta_NLL_vs_ScheduleFree": delta_vs(ref_by_key, row, "schedule_free_adamw_local"),
                "Delta_NLL_vs_POET": (final_nll - poet_nll) if final_nll is not None and poet_nll is not None else "",
                "Delta_NLL_vs_Pion": delta_vs(ref_by_key, row, "pion_oet_sphere_official"),
                "Delta_NLL_vs_strongest": (final_nll - strongest_nll) if final_nll is not None and strongest_nll is not None else "",
                "beats_reference": int(final_nll is not None and ref_nll is not None and final_nll < ref_nll),
                "beats_POET_or_external_OET": int(final_nll is not None and poet_nll is not None and final_nll < poet_nll),
                "beats_strongest": int(final_nll is not None and strongest_nll is not None and final_nll < strongest_nll),
                "best_control_method": best_control.get("method", "") if best_control else "",
                "best_control_final_NLL": control_nll,
                "beats_best_control": int(final_nll is not None and control_nll is not None and final_nll < control_nll),
                "beats_all_controls": int(
                    final_nll is not None
                    and bool(controls)
                    and all(final_nll < (fval(c.get("final_NLL"), float("inf")) or float("inf")) for c in controls)
                ),
                "best_reweighting_control_method": best_reweight.get("method", "") if best_reweight else "",
                "best_reweighting_control_final_NLL": reweight_nll,
                "beats_all_reweighting_controls": int(
                    final_nll is not None
                    and bool(reweight_controls)
                    and all(final_nll < (fval(c.get("final_NLL"), float("inf")) or float("inf")) for c in reweight_controls)
                ),
                "available_control_count": len(controls),
                "available_reweighting_control_count": len(reweight_controls),
                "available_controls": ",".join(sorted({c.get("method", "") for c in controls})),
                "available_reweighting_controls": ",".join(sorted({c.get("method", "") for c in reweight_controls})),
                "no_ECE_Brier_tail_debt": row.get("no_ECE_Brier_tail_debt", ""),
                "AUC_loss_time_improvement": int(
                    (fval(row.get("AUC_loss_time")) or float("inf")) < (fval(ref.get("AUC_loss_time")) or float("inf"))
                )
                if ref
                else 0,
                "controller_overhead_ratio": row.get("controller_overhead_ratio", ""),
                "gate_class_count_correlation_abs": abs(fval(row.get("gate_class_count_correlation"), 0.0) or 0.0),
                "gate_loss_correlation_abs": abs(fval(row.get("gate_loss_correlation"), 0.0) or 0.0),
                "gate_margin_correlation_abs": abs(fval(row.get("gate_margin_correlation"), 0.0) or 0.0),
                "standard_loop_hard_gate_pass": row.get("standard_loop_hard_gate_pass", ""),
                "gate_mean": row.get("gate_mean", ""),
                "gate_std": row.get("gate_std", ""),
                "cohort_mode": row.get("cohort_mode", ""),
                "wgo_stats_interval": row.get("wgo_stats_interval", ""),
                "wgo_param_scope": row.get("wgo_param_scope", ""),
                "wgo_stats_backend": row.get("wgo_stats_backend", ""),
                "wgo_parameter_count": row.get("wgo_parameter_count", ""),
                "deleak_projection_fraction": row.get("deleak_projection_fraction", ""),
                "deleak_nuisance_std": row.get("deleak_nuisance_std", ""),
                "diagnostic_only": row.get("diagnostic_only", ""),
            }
        )

    method_summary: list[dict[str, Any]] = []
    for method in sorted({row.get("method", "") for row in rows}):
        group = [row for row in rows if row.get("method") == method]
        method_summary.append(
            {
                "method": method,
                "phase": group[0].get("phase", "") if group else "",
                "rows": len(group),
                "completed_rows": sum(int(r.get("run_status") == "completed") for r in group),
                "mean_final_NLL": mean([fval(r.get("final_NLL")) for r in group]),
                "mean_accuracy": mean([fval(r.get("final_accuracy")) for r in group]),
                "no_debt_rows": sum(iflag(r.get("no_ECE_Brier_tail_debt")) for r in group),
                "standard_loop_pass_rows": sum(iflag(r.get("standard_loop_hard_gate_pass")) for r in group),
                "mean_gate_mean": mean([fval(r.get("gate_mean")) for r in group]),
                "mean_gate_std": mean([fval(r.get("gate_std")) for r in group]),
                "mean_overhead_ratio": mean([fval(r.get("controller_overhead_ratio")) for r in group]),
                "mean_gate_class_count_correlation_abs": mean(
                    [abs(fval(r.get("gate_class_count_correlation"), 0.0) or 0.0) for r in group]
                ),
                "mean_gate_loss_correlation_abs": mean(
                    [abs(fval(r.get("gate_loss_correlation"), 0.0) or 0.0) for r in group]
                ),
                "mean_gate_margin_correlation_abs": mean(
                    [abs(fval(r.get("gate_margin_correlation"), 0.0) or 0.0) for r in group]
                ),
                "mean_deleak_projection_fraction": mean([fval(r.get("deleak_projection_fraction")) for r in group]),
                "mean_deleak_nuisance_std": mean([fval(r.get("deleak_nuisance_std")) for r in group]),
            }
        )

    candidate_pairwise = [row for row in pairwise if row.get("method") in WGO_CANDIDATES]
    route = {
        "reference_rows": len(reference_rows),
        "repair_wgo_rows": len(wgo_rows),
        "candidate_wgo_rows": len(candidate_pairwise),
        "tier0_candidate_rows": len([r for r in candidate_pairwise if r.get("dataset") in {"MNIST", "FashionMNIST", "KMNIST"}]),
        "standard_loop_pass_candidate_rows": sum(iflag(r.get("standard_loop_hard_gate_pass")) for r in candidate_pairwise),
        "beats_strongest_rows": sum(iflag(r.get("beats_strongest")) for r in candidate_pairwise),
        "beats_POET_or_external_OET_rows": sum(iflag(r.get("beats_POET_or_external_OET")) for r in candidate_pairwise),
        "beats_best_control_rows": sum(iflag(r.get("beats_best_control")) for r in candidate_pairwise),
        "beats_all_controls_rows": sum(iflag(r.get("beats_all_controls")) for r in candidate_pairwise),
        "beats_all_reweighting_controls_rows": sum(iflag(r.get("beats_all_reweighting_controls")) for r in candidate_pairwise),
        "no_debt_rows": sum(iflag(r.get("no_ECE_Brier_tail_debt")) for r in candidate_pairwise),
        "AUC_loss_time_improvement_rows": sum(iflag(r.get("AUC_loss_time_improvement")) for r in candidate_pairwise),
        "overhead_le_035_rows": sum(
            int((fval(r.get("controller_overhead_ratio"), float("inf")) or float("inf")) <= 0.35)
            for r in candidate_pairwise
        ),
        "gate_class_corr_le_070_rows": sum(
            int((fval(r.get("gate_class_count_correlation_abs"), float("inf")) or float("inf")) <= 0.70)
            for r in candidate_pairwise
        ),
        "gate_loss_corr_le_070_rows": sum(
            int((fval(r.get("gate_loss_correlation_abs"), float("inf")) or float("inf")) <= 0.70)
            for r in candidate_pairwise
        ),
        "gate_margin_corr_le_070_rows": sum(
            int((fval(r.get("gate_margin_correlation_abs"), float("inf")) or float("inf")) <= 0.70)
            for r in candidate_pairwise
        ),
        "candidate_methods": sorted({r.get("method", "") for r in candidate_pairwise}),
        "control_methods": sorted({r.get("method", "") for r in rows if r.get("method") in WGO_CONTROLS}),
        "mlp_exploration_pass": 0,
        "final_route": "",
    }
    route["mlp_exploration_pass"] = int(
        route["tier0_candidate_rows"] >= 9
        and route["beats_strongest_rows"] >= 5
        and route["beats_POET_or_external_OET_rows"] >= 5
        and route["beats_best_control_rows"] >= 6
        and route["beats_all_reweighting_controls_rows"] >= 6
        and route["no_debt_rows"] >= 7
        and route["AUC_loss_time_improvement_rows"] >= 5
        and route["overhead_le_035_rows"] >= 8
        and route["gate_class_corr_le_070_rows"] >= 8
        and route["gate_loss_corr_le_070_rows"] >= 8
        and route["standard_loop_pass_candidate_rows"] == len(candidate_pairwise)
    )
    route["final_route"] = classify(candidate_pairwise, route)

    per_method: dict[str, dict[str, Any]] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidate_pairwise:
        grouped[str(row.get("method", ""))].append(row)
    for method, group in sorted(grouped.items()):
        per_method[method] = {
            "rows": len(group),
            "beats_strongest_rows": sum(iflag(r.get("beats_strongest")) for r in group),
            "beats_best_control_rows": sum(iflag(r.get("beats_best_control")) for r in group),
            "beats_all_controls_rows": sum(iflag(r.get("beats_all_controls")) for r in group),
            "beats_all_reweighting_controls_rows": sum(iflag(r.get("beats_all_reweighting_controls")) for r in group),
            "no_debt_rows": sum(iflag(r.get("no_ECE_Brier_tail_debt")) for r in group),
            "AUC_loss_time_improvement_rows": sum(iflag(r.get("AUC_loss_time_improvement")) for r in group),
            "overhead_le_035_rows": sum(
                int((fval(r.get("controller_overhead_ratio"), float("inf")) or float("inf")) <= 0.35) for r in group
            ),
            "gate_class_corr_le_070_rows": sum(
                int((fval(r.get("gate_class_count_correlation_abs"), float("inf")) or float("inf")) <= 0.70)
                for r in group
            ),
            "gate_loss_corr_le_070_rows": sum(
                int((fval(r.get("gate_loss_correlation_abs"), float("inf")) or float("inf")) <= 0.70) for r in group
            ),
            "mean_Delta_NLL_vs_strongest": mean([fval(r.get("Delta_NLL_vs_strongest")) for r in group]),
            "mean_overhead_ratio": mean([fval(r.get("controller_overhead_ratio")) for r in group]),
            "mean_gate_class_count_correlation_abs": mean([fval(r.get("gate_class_count_correlation_abs")) for r in group]),
            "mean_gate_loss_correlation_abs": mean([fval(r.get("gate_loss_correlation_abs")) for r in group]),
            "mean_deleak_projection_fraction": mean([fval(r.get("deleak_projection_fraction")) for r in group]),
            "mean_deleak_nuisance_std": mean([fval(r.get("deleak_nuisance_std")) for r in group]),
        }
    route["per_candidate_method"] = per_method
    return method_summary, pairwise, route


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk-dir", default="results/v22_55/chunks")
    parser.add_argument(
        "--repair-prefix",
        action="append",
        default=[],
        help="Chunk label prefix without leading v22_55_; can be passed multiple times.",
    )
    parser.add_argument("--reference-prefix", default="v22_55_reference")
    parser.add_argument("--out-prefix", default="results/v22_55/v22_55_repair_label_loss_stratified")
    args = parser.parse_args()

    chunk_dir = Path(args.chunk_dir)
    repair_prefixes = args.repair_prefix or [
        "v22_55_repair_label_loss_stratified",
        "v22_55_repair_label_loss_stratified_extra_controls",
    ]
    repair_paths: list[Path] = []
    for prefix in repair_prefixes:
        repair_paths.extend(chunk_dir.glob(f"v22_55_{prefix}_*_summary.csv"))
    reference_paths = list(chunk_dir.glob(f"v22_55_{args.reference_prefix}_*_summary.csv"))

    rows = read_rows(repair_paths)
    reference_rows = [r for r in read_rows(reference_paths) if r.get("run_status") == "completed"]
    method_summary, pairwise, route = summarize(rows, reference_rows)

    out_prefix = Path(args.out_prefix)
    write_rows(out_prefix.with_name(out_prefix.name + "_method_summary.csv"), method_summary)
    write_rows(out_prefix.with_name(out_prefix.name + "_pairwise.csv"), pairwise)
    out_prefix.with_name(out_prefix.name + "_summary.json").write_text(
        json.dumps(route, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(route, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
