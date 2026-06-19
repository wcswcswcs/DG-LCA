#!/usr/bin/env python3
"""Derive v22.33 Part E control-win diagnostics from repair branch artifacts."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "results/v22_33"


def finite_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value in {"", None}:
            return default
        out = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(out):
        return default
    return out


def int_flag(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def mean_float(values: list[Any]) -> str:
    vals = [finite_float(v) for v in values]
    vals_f = [float(v) for v in vals if v is not None]
    if not vals_f:
        return ""
    return sum(vals_f) / len(vals_f)


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def control_level(row: dict[str, Any]) -> str:
    level = str(row.get("control_level", "")).strip()
    if level:
        return level
    name = str(row.get("selector_name", ""))
    for candidate in ["L1", "L2", "L3", "L4", "L5"]:
        if name.startswith(candidate):
            return candidate
    return ""


def derive_for_source(source_name: str, rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in rows:
        key = (str(row.get("dataset", "")), str(row.get("seed", "")), str(row.get("branch_H", "")))
        grouped.setdefault(key, []).append(row)

    out: list[dict[str, Any]] = []
    for (dataset, seed, horizon), group in grouped.items():
        controls = {control_level(r): r for r in group if int_flag(r.get("is_control_branch"))}
        real_rows = [
            r
            for r in group
            if not int_flag(r.get("is_control_branch"))
            and not str(r.get("selector_name", "")).startswith("L_cohort_influence")
        ]
        for real in real_rows:
            real_nll = finite_float(real.get("NLL_delta_vs_base"))
            real_ece = finite_float(real.get("ECE_delta_vs_base"))
            real_tail = finite_float(real.get("tail_q99_delta"))
            real_margin_mean = finite_float(real.get("margin_mean_delta_vs_base"))
            real_margin_q10 = finite_float(real.get("margin_q10_delta_vs_base"))
            real_margin_q01 = finite_float(real.get("margin_q01_delta_vs_base"))
            control_nll = {lvl: finite_float(row.get("NLL_delta_vs_base")) for lvl, row in controls.items()}
            control_ece = {lvl: finite_float(row.get("ECE_delta_vs_base")) for lvl, row in controls.items()}
            control_tail = {lvl: finite_float(row.get("tail_q99_delta")) for lvl, row in controls.items()}
            control_margin_mean = {lvl: finite_float(row.get("margin_mean_delta_vs_base")) for lvl, row in controls.items()}
            control_margin_q10 = {lvl: finite_float(row.get("margin_q10_delta_vs_base")) for lvl, row in controls.items()}
            finite_controls = [(lvl, val) for lvl, val in control_nll.items() if val is not None]
            best_level, best_control = min(finite_controls, key=lambda item: item[1]) if finite_controls else ("", None)
            loses_to = [
                lvl
                for lvl, val in control_nll.items()
                if real_nll is not None and val is not None and real_nll >= val
            ]
            beats = [
                lvl
                for lvl, val in control_nll.items()
                if real_nll is not None and val is not None and real_nll < val
            ]
            best_tail = control_tail.get(best_level)
            best_ece = control_ece.get(best_level)
            best_margin_mean = control_margin_mean.get(best_level)
            best_margin_q10 = control_margin_q10.get(best_level)
            tail_debt = real_tail is not None and real_tail > 0.05
            ece_debt = real_ece is not None and real_ece > 0.01
            margin_debt = real_margin_q10 is not None and real_margin_q10 < 0.0
            if "L1" in loses_to:
                diagnosis = "loses_to_L1_no_task_information_or_random_explains"
            elif {"L1", "L2"}.issubset(set(beats)) and ({"L3", "L4", "L5"} & set(loses_to)):
                diagnosis = "beats_low_controls_but_loses_geometry_direction_not_causal"
            elif real_nll is not None and real_nll < 0 and best_control is not None and real_nll >= best_control:
                diagnosis = "real_improves_but_best_control_explains"
            elif tail_debt or ece_debt:
                diagnosis = "safety_debt_primary"
            elif not loses_to and real_nll is not None and real_nll < 0:
                diagnosis = "row_beats_controls_but_not_gate_stable"
            else:
                diagnosis = "task_harm_or_no_stable_signal"
            out.append(
                {
                    "source_family": source_name,
                    "dataset": dataset,
                    "seed": seed,
                    "branch_H": horizon,
                    "selector_name": real.get("selector_name", ""),
                    "real_NLL_delta": real_nll,
                    "real_ECE_delta": real_ece,
                    "real_tail_q99_delta": real_tail,
                    "best_control_level": best_level,
                    "best_control_NLL_delta": best_control,
                    "real_minus_best_control_NLL": "" if real_nll is None or best_control is None else real_nll - best_control,
                    "best_control_ECE_delta": best_ece,
                    "real_minus_best_control_ECE": "" if real_ece is None or best_ece is None else real_ece - best_ece,
                    "best_control_tail_q99_delta": best_tail,
                    "real_minus_best_control_tail_q99": "" if real_tail is None or best_tail is None else real_tail - best_tail,
                    "real_margin_mean_delta": real_margin_mean,
                    "real_margin_q10_delta": real_margin_q10,
                    "real_margin_q01_delta": real_margin_q01,
                    "best_control_margin_mean_delta": best_margin_mean,
                    "real_minus_best_control_margin_mean": "" if real_margin_mean is None or best_margin_mean is None else real_margin_mean - best_margin_mean,
                    "best_control_margin_q10_delta": best_margin_q10,
                    "real_minus_best_control_margin_q10": "" if real_margin_q10 is None or best_margin_q10 is None else real_margin_q10 - best_margin_q10,
                    "L1_NLL_delta": control_nll.get("L1"),
                    "L2_NLL_delta": control_nll.get("L2"),
                    "L3_NLL_delta": control_nll.get("L3"),
                    "L4_NLL_delta": control_nll.get("L4"),
                    "L5_NLL_delta": control_nll.get("L5"),
                    "loses_to_levels": ",".join(loses_to),
                    "beats_levels": ",".join(beats),
                    "real_improves": int(real_nll is not None and real_nll < 0),
                    "beats_all_available_controls": int(not loses_to and bool(finite_controls)),
                    "tail_debt_gt_0p05": int(tail_debt),
                    "ECE_debt_gt_0p01": int(ece_debt),
                    "margin_q10_debt_lt_0": int(margin_debt),
                    "control_win_cause_label": diagnosis,
                    "source_artifact": real.get("source_artifact", ""),
                }
            )
    return out


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for source in sorted({str(r.get("source_family", "")) for r in rows}):
        src = [r for r in rows if r.get("source_family") == source]
        labels = sorted({str(r.get("control_win_cause_label", "")) for r in src})
        row = {
            "source_family": source,
            "rows": len(src),
            "real_improves_rows": sum(int_flag(r.get("real_improves")) for r in src),
            "beats_all_available_controls_rows": sum(int_flag(r.get("beats_all_available_controls")) for r in src),
            "loses_to_L1_rows": sum("L1" in str(r.get("loses_to_levels", "")).split(",") for r in src),
            "loses_to_L3_rows": sum("L3" in str(r.get("loses_to_levels", "")).split(",") for r in src),
            "loses_to_L4_rows": sum("L4" in str(r.get("loses_to_levels", "")).split(",") for r in src),
            "loses_to_L5_rows": sum("L5" in str(r.get("loses_to_levels", "")).split(",") for r in src),
            "tail_debt_gt_0p05_rows": sum(int_flag(r.get("tail_debt_gt_0p05")) for r in src),
            "ECE_debt_gt_0p01_rows": sum(int_flag(r.get("ECE_debt_gt_0p01")) for r in src),
            "margin_q10_debt_lt_0_rows": sum(int_flag(r.get("margin_q10_debt_lt_0")) for r in src),
            "mean_real_margin_q10_delta": mean_float([r.get("real_margin_q10_delta") for r in src]),
            "diagnosis_labels": ";".join(f"{label}:{sum(1 for r in src if r.get('control_win_cause_label') == label)}" for label in labels),
            "promotion_allowed": 0,
            "diagnostic_note": "Part E/F diagnostic only; does not alter final route or promotion gates.",
        }
        out.append(row)
    return out


def main() -> None:
    c1c6_rows = read_rows(OUT_ROOT / "v22_33_T1_C1C6_repair_branch_matrix.csv")
    c7_rows = read_rows(OUT_ROOT / "v22_33_T1_C7_curvature_repair_branch_matrix.csv")
    rows = derive_for_source("C1C6_repair", c1c6_rows) + derive_for_source("C7_curvature_safe", c7_rows)
    summary = summarize(rows)
    write_rows(OUT_ROOT / "v22_33_T1_repair_control_win_decomposition.csv", rows)
    write_rows(OUT_ROOT / "v22_33_T1_repair_control_win_summary.csv", summary)
    print({"rows": len(rows), "summary_rows": len(summary)})


if __name__ == "__main__":
    main()
