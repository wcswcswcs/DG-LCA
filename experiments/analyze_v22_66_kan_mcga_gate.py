#!/usr/bin/env python3
"""Aggregate v22.66 KAN+MCGA rows by run label and architecture."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


REFERENCE_METHODS = {"adamw", "cautious_adamw", "schedule_free_adamw_local"}


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        out = float(text)
    except ValueError:
        return None
    return out if math.isfinite(out) else None


def mean(values: Iterable[float | None]) -> float | None:
    kept = [v for v in values if v is not None]
    return None if not kept else sum(kept) / len(kept)


def pct(count: int, total: int) -> float:
    return 0.0 if total <= 0 else 100.0 * count / total


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def debt(row: dict[str, str]) -> float | None:
    vals = [
        safe_float(row.get("ECE")),
        safe_float(row.get("Brier")),
        safe_float(row.get("tail_loss_q95")),
        safe_float(row.get("tail_loss_q99")),
    ]
    if any(v is None for v in vals):
        return None
    return sum(float(v) for v in vals)


def classify(own: bool, mlp: bool, controls: bool) -> str:
    if own and mlp and controls:
        return "TrueKANGain"
    if own and controls:
        return "BothGain"
    if own:
        return "KANInternalValueOnly"
    if not controls:
        return "ControlExplained"
    if not mlp:
        return "MLPStructuredSupportStronger"
    if mlp and not own:
        return "MLPDegradationDriven"
    return "NoGain"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks-dir", default="results/v22_66/chunks")
    parser.add_argument("--out-dir", default="results/v22_66")
    parser.add_argument("--label", required=True)
    parser.add_argument("--candidate-architecture", default="DGKAN_DFOU")
    parser.add_argument("--candidate-method", default="mcga_over_poet_fsclip_eta025_residual_rank4")
    parser.add_argument(
        "--control-methods",
        default="same_functional_spectrum_random_coordinate,same_generator_descent_energy_random,same_C_skew_spectrum_random",
    )
    parser.add_argument("--mlp-architecture", default="MLP")
    parser.add_argument("--artifact-suffix", default="")
    parser.add_argument("--spectrum-threshold", type=float, default=0.50)
    args = parser.parse_args()

    rows: list[dict[str, str]] = []
    fieldnames: list[str] = []
    for path in sorted(Path(args.chunks_dir).glob("*.csv")):
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("run_label") != args.label:
                    continue
                row["chunk_path"] = str(path)
                rows.append(row)
                for name in list(reader.fieldnames or []) + ["chunk_path"]:
                    if name not in fieldnames:
                        fieldnames.append(name)

    control_methods = {m.strip() for m in str(args.control_methods).split(",") if m.strip()}
    by_cond: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_cond[(row.get("dataset", ""), row.get("seed", ""))].append(row)

    detail: list[dict[str, Any]] = []
    for cond, cond_rows in sorted(by_cond.items()):
        dataset, seed = cond
        cand = [
            r
            for r in cond_rows
            if r.get("architecture_key") == args.candidate_architecture
            and r.get("method") == args.candidate_method
            and r.get("run_status") == "completed"
        ]
        if not cand:
            continue
        row = cand[0]
        cand_nll = safe_float(row.get("held_NLL"))
        own_refs = [
            r
            for r in cond_rows
            if r.get("architecture_key") == args.candidate_architecture
            and r.get("method") in REFERENCE_METHODS
            and r.get("run_status") == "completed"
        ]
        mlp_rows = [
            r
            for r in cond_rows
            if r.get("architecture_key") == args.mlp_architecture
            and r.get("method") == args.candidate_method
            and r.get("run_status") == "completed"
        ]
        control_rows = [
            r
            for r in cond_rows
            if r.get("architecture_key") == args.candidate_architecture
            and r.get("method") in control_methods
            and r.get("run_status") == "completed"
        ]
        own_best = min((safe_float(r.get("held_NLL")) for r in own_refs), default=None)
        mlp_best = min((safe_float(r.get("held_NLL")) for r in mlp_rows), default=None)
        control_best = min((safe_float(r.get("held_NLL")) for r in control_rows), default=None)
        own_debt = min((debt(r) for r in own_refs if debt(r) is not None), default=None)
        cand_debt = debt(row)
        own_gain = bool(cand_nll is not None and own_best is not None and cand_nll < own_best)
        beats_mlp = bool(cand_nll is not None and mlp_best is not None and cand_nll < mlp_best)
        beats_controls = bool(cand_nll is not None and control_best is not None and cand_nll < control_best)
        no_debt = bool(cand_debt is not None and own_debt is not None and cand_debt <= own_debt + 1.0e-9)
        overhead = safe_float(row.get("controller_overhead_ratio"))
        drift = safe_float(row.get("active_Gram_drift_mean"))
        spectrum = safe_float(row.get("functional_spectrum_drift_mean"))
        gen_frac = safe_float(row.get("generator_descent_fraction"))
        detail.append(
            {
                "dataset": dataset,
                "seed": seed,
                "candidate_held_NLL": cand_nll,
                "own_best_NLL": own_best,
                "mlp_matched_NLL": mlp_best,
                "best_control_NLL": control_best,
                "delta_vs_own": cand_nll - own_best if cand_nll is not None and own_best is not None else "",
                "delta_vs_mlp": cand_nll - mlp_best if cand_nll is not None and mlp_best is not None else "",
                "delta_vs_best_control": cand_nll - control_best if cand_nll is not None and control_best is not None else "",
                "KAN_improves_own": int(own_gain),
                "KAN_beats_MLP_matched": int(beats_mlp),
                "KAN_beats_best_KAN_control": int(beats_controls),
                "TrueKANGain_class": classify(own_gain, beats_mlp, beats_controls),
                "no_ECE_Brier_tail_debt": int(no_debt),
                "candidate_debt": cand_debt,
                "own_best_debt": own_debt,
                "controller_overhead_ratio": overhead,
                "overhead_le_035": int(overhead is not None and overhead <= 0.35),
                "active_Gram_drift_mean": drift,
                "active_Gram_drift_le_005": int(drift is not None and drift <= 0.05),
                "functional_spectrum_drift_mean": spectrum,
                "functional_spectrum_drift_le_threshold": int(
                    spectrum is not None and spectrum <= float(args.spectrum_threshold)
                ),
                "generator_descent_fraction": gen_frac,
                "generator_descent_fraction_positive": int(gen_frac is not None and gen_frac > 1.0e-4),
                "standard_loop_runtime_trace_pass": int(safe_float(row.get("standard_loop_runtime_trace_pass")) == 1.0),
                "kan_readout_linearization_max_abs_error": row.get("kan_readout_linearization_max_abs_error", ""),
                "chunk_path": row.get("chunk_path", ""),
            }
        )

    class_counts = Counter(str(r.get("TrueKANGain_class", "")) for r in detail)
    total = len(detail)
    true_rows = class_counts.get("TrueKANGain", 0) + class_counts.get("BothGain", 0)
    control_rows = class_counts.get("ControlExplained", 0)
    mlp_degradation_rows = class_counts.get("MLPDegradationDriven", 0)
    summary = {
        "label": args.label,
        "candidate": f"{args.candidate_architecture}:{args.candidate_method}",
        "matrix_rows": len(rows),
        "candidate_rows": total,
        "conditions": len({(r.get("dataset"), r.get("seed")) for r in detail}),
        "datasets": ",".join(sorted({str(r.get("dataset")) for r in detail})),
        "seeds": ",".join(sorted({str(r.get("seed")) for r in detail}, key=lambda x: int(float(x)))),
        "KAN_improves_own_rows": sum(int(r["KAN_improves_own"]) for r in detail),
        "KAN_beats_MLP_matched_rows": sum(int(r["KAN_beats_MLP_matched"]) for r in detail),
        "KAN_beats_best_KAN_control_rows": sum(int(r["KAN_beats_best_KAN_control"]) for r in detail),
        "TrueKANGain_plus_BothGain_rows": true_rows,
        "TrueKANGain_plus_BothGain_pct": pct(true_rows, total),
        "ControlExplained_rows": control_rows,
        "ControlExplained_pct": pct(control_rows, total),
        "MLPDegradationDriven_rows": mlp_degradation_rows,
        "MLPDegradationDriven_pct": pct(mlp_degradation_rows, total),
        "no_debt_rows": sum(int(r["no_ECE_Brier_tail_debt"]) for r in detail),
        "overhead_le_035_rows": sum(int(r["overhead_le_035"]) for r in detail),
        "active_Gram_drift_le_005_rows": sum(int(r["active_Gram_drift_le_005"]) for r in detail),
        "functional_spectrum_drift_le_threshold_rows": sum(
            int(r["functional_spectrum_drift_le_threshold"]) for r in detail
        ),
        "generator_descent_fraction_positive_rows": sum(
            int(r["generator_descent_fraction_positive"]) for r in detail
        ),
        "standard_loop_pass_rows": sum(int(r["standard_loop_runtime_trace_pass"]) for r in detail),
        "mean_candidate_held_NLL": mean(safe_float(r.get("candidate_held_NLL")) for r in detail),
        "mean_delta_vs_own": mean(safe_float(r.get("delta_vs_own")) for r in detail),
        "mean_delta_vs_mlp": mean(safe_float(r.get("delta_vs_mlp")) for r in detail),
        "mean_delta_vs_best_control": mean(safe_float(r.get("delta_vs_best_control")) for r in detail),
        "max_kan_readout_linearization_abs_error": max(
            [safe_float(r.get("kan_readout_linearization_max_abs_error")) or 0.0 for r in detail],
            default=0.0,
        ),
        "TrueKANGain_class_counts": dict(class_counts),
    }
    summary["kan_mcga_gate_pass"] = int(
        total >= 15
        and summary["conditions"] >= 15
        and summary["KAN_improves_own_rows"] >= 10
        and summary["KAN_beats_MLP_matched_rows"] >= 9
        and summary["KAN_beats_best_KAN_control_rows"] >= 10
        and summary["TrueKANGain_plus_BothGain_pct"] >= 40.0
        and summary["ControlExplained_pct"] <= 30.0
        and summary["MLPDegradationDriven_pct"] <= 20.0
        and summary["no_debt_rows"] >= 12
        and summary["overhead_le_035_rows"] >= 12
        and summary["active_Gram_drift_le_005_rows"] >= 12
        and summary["functional_spectrum_drift_le_threshold_rows"] >= 12
        and summary["generator_descent_fraction_positive_rows"] >= 12
        and summary["standard_loop_pass_rows"] == total
    )

    artifact = args.label
    if args.artifact_suffix:
        artifact = f"{artifact}_{args.artifact_suffix}"
    out_dir = Path(args.out_dir)
    matrix_path = out_dir / f"{artifact}_kan_mcga_matrix.csv"
    detail_path = out_dir / f"{artifact}_kan_mcga_candidate_detail.csv"
    summary_json = out_dir / f"{artifact}_kan_mcga_summary.json"
    summary_csv = out_dir / f"{artifact}_kan_mcga_summary.csv"
    write_csv(matrix_path, rows, fieldnames)
    detail_fields = list(detail[0].keys()) if detail else ["status"]
    write_csv(detail_path, detail or [{"status": "no_candidate_rows"}], detail_fields)
    summary["matrix_copy"] = str(matrix_path)
    summary["candidate_detail"] = str(detail_path)
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    flat = dict(summary)
    flat["TrueKANGain_class_counts"] = json.dumps(summary["TrueKANGain_class_counts"], sort_keys=True)
    write_csv(summary_csv, [flat], list(flat.keys()))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
