#!/usr/bin/env python3
"""Aggregate the v22.66 post-MLP KAN S6 carrier audit."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


def safe_float(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def safe_int_flag(value: object) -> int:
    number = safe_float(value)
    return int(number == 1.0)


def mean(values: Iterable[float | None]) -> float | None:
    kept = [v for v in values if v is not None]
    if not kept:
        return None
    return sum(kept) / len(kept)


def count_if(rows: list[dict[str, str]], predicate) -> int:
    return sum(1 for row in rows if predicate(row))


def pct(count: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return 100.0 * count / total


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def safe_suffix(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in value).strip("_")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="results/v22_44R/v22_43_metric_support_full_loop_matrix.csv")
    parser.add_argument("--label", default="v22_66_kan_s6_official_like")
    parser.add_argument("--out-dir", default="results/v22_66")
    parser.add_argument("--candidate-architecture-key", default="DGKAN_DCHE")
    parser.add_argument("--candidate-variant", default="KAN-D-CHE-BasisGram")
    parser.add_argument("--candidate-control-mode", default="none")
    parser.add_argument(
        "--artifact-suffix",
        default="",
        help="Optional suffix for candidate-specific output files when one label has multiple candidates.",
    )
    args = parser.parse_args()

    source = Path(args.source)
    out_dir = Path(args.out_dir)
    artifact_label = args.label
    if args.artifact_suffix:
        artifact_label = f"{args.label}_{safe_suffix(args.artifact_suffix)}"
    with source.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = [row for row in reader if row.get("run_label", "").startswith(args.label)]
        fieldnames = list(reader.fieldnames or [])

    matrix_path = out_dir / f"{artifact_label}_matrix.csv"
    write_csv(matrix_path, rows, fieldnames)

    candidate_rows = [
        row
        for row in rows
        if row.get("architecture_key") == args.candidate_architecture_key
        and row.get("variant") == args.candidate_variant
        and row.get("control_mode") == args.candidate_control_mode
    ]
    candidate_rows.sort(key=lambda r: (r.get("dataset", ""), int(safe_float(r.get("seed")) or 0)))

    detail_fields = [
        "run_label",
        "dataset",
        "seed",
        "final_NLL",
        "held_NLL",
        "final_accuracy",
        "NLL_improvement_vs_own_strong_optimizer",
        "KAN_NLL_delta_vs_MLP_matched_support_FU",
        "beats_same_basis_Gram_random",
        "beats_same_basis_Gram_signflip",
        "beats_both_same_basis_controls",
        "TrueKANGain_class",
        "no_ECE_Brier_tail_debt",
        "controller_overhead_ratio",
        "basis_energy_fraction",
        "readout_leakage_fraction",
        "basis_Gram_condition",
    ]
    detail_rows: list[dict[str, object]] = []
    for row in candidate_rows:
        random_flag = safe_int_flag(row.get("beats_same_basis_Gram_random"))
        signflip_flag = safe_int_flag(row.get("beats_same_basis_Gram_signflip"))
        detail_rows.append(
            {
                "run_label": row.get("run_label", ""),
                "dataset": row.get("dataset", ""),
                "seed": row.get("seed", ""),
                "final_NLL": row.get("final_NLL", ""),
                "held_NLL": row.get("held_NLL", ""),
                "final_accuracy": row.get("final_accuracy", ""),
                "NLL_improvement_vs_own_strong_optimizer": row.get(
                    "NLL_improvement_vs_own_strong_optimizer", ""
                ),
                "KAN_NLL_delta_vs_MLP_matched_support_FU": row.get(
                    "KAN_NLL_delta_vs_MLP_matched_support_FU", ""
                ),
                "beats_same_basis_Gram_random": random_flag,
                "beats_same_basis_Gram_signflip": signflip_flag,
                "beats_both_same_basis_controls": int(random_flag == 1 and signflip_flag == 1),
                "TrueKANGain_class": row.get("TrueKANGain_class", ""),
                "no_ECE_Brier_tail_debt": safe_int_flag(row.get("no_ECE_Brier_tail_debt")),
                "controller_overhead_ratio": row.get("controller_overhead_ratio", ""),
                "basis_energy_fraction": row.get("basis_energy_fraction", ""),
                "readout_leakage_fraction": row.get("readout_leakage_fraction", ""),
                "basis_Gram_condition": row.get("basis_Gram_condition", ""),
            }
        )
    detail_path = out_dir / f"{artifact_label}_candidate_detail.csv"
    write_csv(detail_path, detail_rows, detail_fields)

    conditions = {(row.get("dataset", ""), row.get("seed", "")) for row in candidate_rows}
    class_counts = Counter(row.get("TrueKANGain_class", "") for row in candidate_rows)
    own_rows = count_if(
        candidate_rows, lambda row: (safe_float(row.get("NLL_improvement_vs_own_strong_optimizer")) or 0.0) > 0.0
    )
    mlp_rows = count_if(
        candidate_rows, lambda row: (safe_float(row.get("KAN_NLL_delta_vs_MLP_matched_support_FU")) or 0.0) < 0.0
    )
    random_rows = count_if(candidate_rows, lambda row: safe_int_flag(row.get("beats_same_basis_Gram_random")) == 1)
    signflip_rows = count_if(candidate_rows, lambda row: safe_int_flag(row.get("beats_same_basis_Gram_signflip")) == 1)
    both_control_rows = count_if(
        candidate_rows,
        lambda row: safe_int_flag(row.get("beats_same_basis_Gram_random")) == 1
        and safe_int_flag(row.get("beats_same_basis_Gram_signflip")) == 1,
    )
    true_gain_rows = sum(class_counts.get(name, 0) for name in ("TrueKANGain", "BothGain"))
    control_explained_rows = class_counts.get("ControlExplained", 0)
    mlp_degradation_rows = class_counts.get("MLPDegradationDriven", 0)
    no_debt_rows = count_if(candidate_rows, lambda row: safe_int_flag(row.get("no_ECE_Brier_tail_debt")) == 1)
    overhead_rows = count_if(
        candidate_rows,
        lambda row: (safe_float(row.get("controller_overhead_ratio")) is not None)
        and safe_float(row.get("controller_overhead_ratio")) <= 0.35,
    )
    basis_rows = count_if(
        candidate_rows,
        lambda row: (safe_float(row.get("basis_energy_fraction")) is not None)
        and safe_float(row.get("basis_energy_fraction")) >= 0.5,
    )
    readout_rows = count_if(
        candidate_rows,
        lambda row: (safe_float(row.get("readout_leakage_fraction")) is not None)
        and safe_float(row.get("readout_leakage_fraction")) <= 0.3,
    )

    by_dataset: dict[str, dict[str, object]] = {}
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in candidate_rows:
        grouped[row.get("dataset", "")].append(row)
    for dataset, ds_rows in sorted(grouped.items()):
        by_dataset[dataset] = {
            "rows": len(ds_rows),
            "KAN_improves_own_rows": count_if(
                ds_rows, lambda row: (safe_float(row.get("NLL_improvement_vs_own_strong_optimizer")) or 0.0) > 0.0
            ),
            "KAN_beats_MLP_matched_rows": count_if(
                ds_rows, lambda row: (safe_float(row.get("KAN_NLL_delta_vs_MLP_matched_support_FU")) or 0.0) < 0.0
            ),
            "KAN_beats_both_same_basis_controls_rows": count_if(
                ds_rows,
                lambda row: safe_int_flag(row.get("beats_same_basis_Gram_random")) == 1
                and safe_int_flag(row.get("beats_same_basis_Gram_signflip")) == 1,
            ),
            "TrueKANGain_plus_BothGain_rows": count_if(
                ds_rows, lambda row: row.get("TrueKANGain_class", "") in {"TrueKANGain", "BothGain"}
            ),
            "no_debt_rows": count_if(ds_rows, lambda row: safe_int_flag(row.get("no_ECE_Brier_tail_debt")) == 1),
            "overhead_le_035_rows": count_if(
                ds_rows,
                lambda row: (safe_float(row.get("controller_overhead_ratio")) is not None)
                and safe_float(row.get("controller_overhead_ratio")) <= 0.35,
            ),
            "mean_controller_overhead_ratio": mean(
                safe_float(row.get("controller_overhead_ratio")) for row in ds_rows
            ),
            "mean_NLL_improvement_vs_own": mean(
                safe_float(row.get("NLL_improvement_vs_own_strong_optimizer")) for row in ds_rows
            ),
            "mean_KAN_delta_vs_MLP_matched": mean(
                safe_float(row.get("KAN_NLL_delta_vs_MLP_matched_support_FU")) for row in ds_rows
            ),
            "mean_held_NLL": mean(safe_float(row.get("held_NLL")) for row in ds_rows),
        }

    total = len(candidate_rows)
    gate_pass = int(
        total >= 15
        and len(conditions) >= 15
        and own_rows >= 10
        and mlp_rows >= 9
        and both_control_rows >= 10
        and pct(true_gain_rows, total) >= 40.0
        and pct(control_explained_rows, total) <= 30.0
        and pct(mlp_degradation_rows, total) <= 20.0
        and no_debt_rows >= 12
        and overhead_rows >= 12
    )

    summary = {
        "source_merged_matrix": str(source),
        "matrix_copy": str(matrix_path),
        "candidate_detail": str(detail_path),
        "label": args.label,
        "artifact_label": artifact_label,
        "matrix_rows": len(rows),
        "candidate": (
            f"{args.candidate_architecture_key}:{args.candidate_variant}:{args.candidate_control_mode}"
        ),
        "candidate_rows": total,
        "conditions": len(conditions),
        "datasets": ",".join(sorted({row.get("dataset", "") for row in candidate_rows})),
        "seeds": ",".join(sorted({row.get("seed", "") for row in candidate_rows}, key=lambda x: int(safe_float(x) or 0))),
        "KAN_improves_own_rows": own_rows,
        "KAN_beats_MLP_matched_rows": mlp_rows,
        "KAN_beats_same_basis_random_rows": random_rows,
        "KAN_beats_same_basis_signflip_rows": signflip_rows,
        "KAN_beats_both_same_basis_controls_rows": both_control_rows,
        "TrueKANGain_plus_BothGain_rows": true_gain_rows,
        "TrueKANGain_plus_BothGain_pct": pct(true_gain_rows, total),
        "ControlExplained_rows": control_explained_rows,
        "ControlExplained_pct": pct(control_explained_rows, total),
        "MLPDegradationDriven_rows": mlp_degradation_rows,
        "MLPDegradationDriven_pct": pct(mlp_degradation_rows, total),
        "no_debt_rows": no_debt_rows,
        "overhead_le_035_rows": overhead_rows,
        "basis_energy_ge_05_rows": basis_rows,
        "readout_leakage_le_03_rows": readout_rows,
        "TrueKANGain_class_counts": dict(class_counts),
        "dataset_breakdown": by_dataset,
        "kan_gate_pass": gate_pass,
    }

    json_path = out_dir / f"{artifact_label}_summary.json"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    summary_csv_path = out_dir / f"{artifact_label}_summary.csv"
    flat_summary = dict(summary)
    flat_summary["TrueKANGain_class_counts"] = json.dumps(summary["TrueKANGain_class_counts"], sort_keys=True)
    flat_summary["dataset_breakdown"] = json.dumps(summary["dataset_breakdown"], sort_keys=True)
    write_csv(summary_csv_path, [flat_summary], list(flat_summary.keys()))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
