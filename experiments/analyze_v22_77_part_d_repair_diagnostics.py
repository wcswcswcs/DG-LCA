#!/usr/bin/env python3
"""Summarize v22.77 Part D repair attempts."""

from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "results/v22_77"


def fval(row: dict[str, str], key: str) -> float:
    try:
        value = float(row.get(key, ""))
        return value if math.isfinite(value) else 0.0
    except Exception:
        return 0.0


def quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round((len(ordered) - 1) * q)))]


def main() -> None:
    rows = list(csv.DictReader((OUT_ROOT / "v22_77_part_d_conditional_edge_preflight.csv").open(newline="", encoding="utf-8")))
    family = list(csv.DictReader((OUT_ROOT / "v22_77_part_d_family_summaries.csv").open(newline="", encoding="utf-8")))
    attempts = []
    for attempt in sorted({row["attempt_label"] for row in rows}):
        rs = [row for row in rows if row["attempt_label"] == attempt]
        fs = [row for row in family if row["attempt_label"] == attempt]
        attempts.append(
            {
                "attempt_label": attempt,
                "microprobe_rows": len(rs),
                "family_rows": len(fs),
                "passed_family_count": sum(int(row.get("family_gate_pass", "0")) for row in fs),
                "candidate_NLL_improve_rows": sum(fval(row, "candidate_delta_NLL_guard") < 0.0 for row in rs),
                "candidate_Brier_nonpositive_rows": sum(fval(row, "candidate_delta_Brier_guard") <= 0.0 for row in rs),
                "all_debt_UCB_nonpositive_rows": sum(fval(row, "all_debt_UCB_max") <= 0.0 for row in rs),
                "control_margin_positive_rows": sum(fval(row, "control_contrastive_margin_p10") > 0.0 for row in rs),
                "same_domain_candidate_better_rows": sum(fval(row, "same_domain_control_gap_guard") > 0.0 for row in rs),
                "coherence_LCB_positive_rows": sum(fval(row, "source_witness_conditional_coherence_LCB") > 0.0 for row in rs),
                "median_conditional_residual_fraction": quantile([fval(row, "conditional_residual_fraction") for row in rs], 0.5),
                "median_domain_nuisance_fraction": quantile([fval(row, "domain_nuisance_fraction") for row in rs], 0.5),
                "median_control_margin_p10": quantile([fval(row, "control_contrastive_margin_p10") for row in rs], 0.5),
                "median_candidate_delta_NLL_guard": quantile([fval(row, "candidate_delta_NLL_guard") for row in rs], 0.5),
                "median_candidate_delta_Brier_guard": quantile([fval(row, "candidate_delta_Brier_guard") for row in rs], 0.5),
                "family_dominant_blockers": {key: sum(1 for row in fs if row.get("dominant_blocker") == key) for key in sorted({row.get("dominant_blocker", "") for row in fs})},
            }
        )
    obj = {
        "generated_at_sg": time.strftime("%Y-%m-%d %H:%M:%S %z", time.localtime()),
        "source_rows": "results/v22_77/v22_77_part_d_conditional_edge_preflight.csv",
        "source_family_summaries": "results/v22_77/v22_77_part_d_family_summaries.csv",
        "total_microprobe_rows": len(rows),
        "family_summary_rows": len(family),
        "attempt_count": len(attempts),
        "attempt_diagnostics": attempts,
        "non_fabrication_note": "Computed directly from v22.77 Part D raw CSV and family summaries.",
    }
    out = OUT_ROOT / "v22_77_part_d_repair_diagnostics_summary.json"
    out.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(out.relative_to(ROOT))


if __name__ == "__main__":
    main()
