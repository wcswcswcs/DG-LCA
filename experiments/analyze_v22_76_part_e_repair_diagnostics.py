#!/usr/bin/env python3
"""Summarize v22.76 Part E repair attempts from raw microprobe rows."""

from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "results/v22_76"


def fval(row: dict[str, str], key: str) -> float:
    try:
        value = float(row.get(key, ""))
        return value if math.isfinite(value) else 0.0
    except Exception:
        return 0.0


def quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round((len(ordered) - 1) * q)))]


def main() -> None:
    rows_path = OUT_ROOT / "v22_76_part_e_control_residualized_edge_preflight.csv"
    family_path = OUT_ROOT / "v22_76_part_e_family_summaries.csv"
    rows = list(csv.DictReader(rows_path.open(newline="", encoding="utf-8")))
    family_rows = list(csv.DictReader(family_path.open(newline="", encoding="utf-8")))

    attempts = []
    for attempt in sorted({row["attempt_label"] for row in rows}):
        attempt_rows = [row for row in rows if row["attempt_label"] == attempt]
        attempts.append(
            {
                "attempt_label": attempt,
                "rows": len(attempt_rows),
                "candidate_NLL_improve_rows": sum(fval(row, "candidate_delta_NLL_guard") < 0.0 for row in attempt_rows),
                "candidate_Brier_nonpositive_rows": sum(fval(row, "candidate_delta_Brier_guard") <= 0.0 for row in attempt_rows),
                "Brier_UCB_nonpositive_rows": sum(fval(row, "Brier_UCB") <= 0.0 for row in attempt_rows),
                "all_debt_UCB_nonpositive_rows": sum(fval(row, "all_debt_UCB_max") <= 0.0 for row in attempt_rows),
                "same_domain_candidate_better_rows": sum(fval(row, "same_domain_control_gap_preflight") < 0.0 for row in attempt_rows),
                "control_margin_positive_rows": sum(fval(row, "control_contrastive_margin_p10") > 0.0 for row in attempt_rows),
                "sign_calibration_applied_rows": sum(fval(row, "sign_calibration_applied") > 0.5 for row in attempt_rows),
                "median_candidate_delta_NLL_guard": quantile([fval(row, "candidate_delta_NLL_guard") for row in attempt_rows], 0.5),
                "median_candidate_delta_Brier_guard": quantile([fval(row, "candidate_delta_Brier_guard") for row in attempt_rows], 0.5),
                "median_same_domain_control_gap_preflight": quantile([fval(row, "same_domain_control_gap_preflight") for row in attempt_rows], 0.5),
                "median_control_contrastive_margin_p10": quantile([fval(row, "control_contrastive_margin_p10") for row in attempt_rows], 0.5),
                "median_control_residual_energy_fraction": quantile([fval(row, "control_residual_energy_fraction") for row in attempt_rows], 0.5),
                "median_downstream_sensitivity_norm": quantile([fval(row, "downstream_sensitivity_norm") for row in attempt_rows], 0.5),
            }
        )

    blocking_counts: dict[str, int] = {}
    for row in family_rows:
        key = row.get("blocking_metric", "")
        if key:
            blocking_counts[key] = blocking_counts.get(key, 0) + 1

    obj = {
        "generated_at_sg": time.strftime("%Y-%m-%d %H:%M:%S %z", time.localtime()),
        "source_rows": str(rows_path.relative_to(ROOT)),
        "source_family_summaries": str(family_path.relative_to(ROOT)),
        "total_microprobe_rows": len(rows),
        "family_summary_rows": len(family_rows),
        "attempt_count": len(attempts),
        "dominant_blocking_metric_counts": blocking_counts,
        "attempt_diagnostics": attempts,
        "non_fabrication_note": "Computed directly from v22.76 Part E raw CSV after downstream-sensitivity repair rerun.",
    }
    out = OUT_ROOT / "v22_76_part_e_repair_diagnostics_summary.json"
    out.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(out.relative_to(ROOT))


if __name__ == "__main__":
    main()
