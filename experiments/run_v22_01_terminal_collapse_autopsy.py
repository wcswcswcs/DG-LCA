#!/usr/bin/env python3
"""v22.01 F0/F1 source-chain reaggregation and terminal-collapse autopsy."""

from __future__ import annotations

import argparse
from collections import defaultdict
import math
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dgkan.fu.debt_accounting import debt_peak_final  # noqa: E402
from dgkan.fu.source_chain import RETENTION_EPS, classify_source_chain, source_chain_row  # noqa: E402
from experiments.run_v22_01_common import (  # noqa: E402
    HORIZONS,
    PYTHON,
    V2200_OFFICIAL,
    append_exec,
    ensure_out,
    finite_float,
    int_flag,
    mean,
    read_rows,
    simple_svg,
    write_json,
    write_rows,
)


CONTROL_IDS = ("CTRL-", "RandomMatched", "NoOpMatched", "StableRandom", "random-matched", "stable-random")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None)
    p.add_argument("--source-dir", default=str(V2200_OFFICIAL))
    return p


def key(row: dict[str, Any]) -> tuple[str, str, str, str, str]:
    return (
        str(row.get("carrier", "")),
        str(row.get("basis_repair_variant", row.get("variant", ""))),
        str(row.get("v21_id", row.get("v22_id", ""))),
        str(row.get("dataset", "")),
        str(row.get("seed", "")),
    )


def is_control(row: dict[str, Any]) -> int:
    ident = str(row.get("v21_id", row.get("v22_id", "")))
    return int(any(token in ident for token in CONTROL_IDS))


def horizon_mean(rows: list[dict[str, Any]], step: int) -> float:
    return mean(rows, f"source_h{step}")


def finite_steps(row: dict[str, Any], prefix: str) -> list[float]:
    vals = []
    for h in HORIZONS:
        v = finite_float(row.get(f"{prefix}{h}"))
        if math.isfinite(v):
            vals.append(v)
    return vals


def debt_fields(row: dict[str, Any], metric: str) -> dict[str, Any]:
    vals = finite_steps(row, f"{metric}_h")
    baseline = vals[0] if vals else 0.0
    out = debt_peak_final(vals, baseline=baseline)
    return {
        f"{metric}_debt_peak": out["debt_peak"],
        f"{metric}_debt_final": out["debt_final"],
        f"{metric}_debt_recovery": out["debt_recovery"],
    }


def reaggregate(source_dir: Path, out_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    raw = read_rows(source_dir / "v21_01_source_retention_matrix.csv")
    old_by_key = {key(r): r for r in read_rows(source_dir / "v22_source_chain_matrix.csv")}
    enriched = []
    diff = []
    for row in raw:
        item = dict(row)
        item["v22_id"] = item.get("v21_id", "")
        item["control_equivalent"] = is_control(item)
        item.update(source_chain_row(item, prefix="source_h"))
        old = old_by_key.get(key(item), {})
        diff.append(
            {
                "carrier": item.get("carrier", ""),
                "v22_id": item.get("v22_id", ""),
                "dataset": item.get("dataset", ""),
                "seed": item.get("seed", ""),
                "old_early_chain": old.get("early_source_chain", ""),
                "new_early_chain": item.get("early_source_chain", ""),
                "old_continuous": old.get("continuous_retention_chain", ""),
                "new_continuous": item.get("continuous_retention_chain", ""),
                "old_h3200": old.get("source_h3200", ""),
                "new_h3200": item.get("source_h3200", ""),
                "old_h4800": old.get("source_h4800", ""),
                "new_h4800": item.get("source_h4800", ""),
                "control_equivalent": item.get("control_equivalent", ""),
                "changed_reason": "source_eps_or_control_equivalent" if str(old.get("early_source_chain", "")) != str(item.get("early_source_chain", "")) else "",
            }
        )
        enriched.append(item)
    write_rows(out_dir / "v22_01_reaggregated_v22_source_chain.csv", enriched)
    write_rows(out_dir / "v22_01_reaggregated_candidate_diff.csv", diff)

    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in enriched:
        groups[(str(row.get("carrier", "")), str(row.get("basis_repair_variant", "")), str(row.get("v22_id", "")))].append(row)
    summary = []
    for (carrier, variant, v22_id), rows in sorted(groups.items()):
        h = {f"h{step}": horizon_mean(rows, step) for step in (100, 400, 800, 1600, 2400, 3200, 4800, 6400)}
        control_eq = int(any(is_control(r) for r in rows))
        decision = classify_source_chain(h["h100"], h["h400"], h["h800"], h["h1600"], h["h3200"], h["h4800"], control_equivalent=control_eq)
        summary.append(
            {
                "carrier": carrier,
                "variant": variant,
                "v22_id": v22_id,
                "rows": len(rows),
                **h,
                "control_equivalent_group": control_eq,
                "early_source_chain_group": decision.early_source_chain,
                "continuous_retention_group": decision.continuous_retention_chain,
                "continuous_h4800_group": decision.continuous_h4800_chain,
                "productive_h3200_group": decision.continuous_retention_chain,
                "productive_h4800_group": decision.continuous_h4800_chain,
                "late_rebound_group": decision.late_rebound,
                "terminal_collapse_group": decision.terminal_collapse,
                "h1600_retention_ratio": decision.h1600_retention_ratio,
                "h3200_retention_ratio": decision.h3200_retention_ratio,
                "h4800_retention_ratio": decision.h4800_retention_ratio,
                "row_early_chain_count": sum(int_flag(r.get("early_source_chain")) for r in rows),
                "row_h4800_positive_count": sum(finite_float(r.get("source_h4800")) >= RETENTION_EPS for r in rows),
                "source_chain_blocker": decision.blocker,
            }
        )
    write_rows(out_dir / "v22_01_reaggregated_v22_source_chain_summary.csv", summary)
    candidates = [r for r in summary if str(r.get("v22_id", "")).startswith(("F", "KSW", "MLP"))]
    route = {
        "source_chain_rows": len(enriched),
        "source_chain_groups": len(summary),
        "candidate_groups": len(candidates),
        "candidate_early_chain": sum(int_flag(r.get("early_source_chain_group")) for r in candidates),
        "candidate_continuous_h3200": sum(int_flag(r.get("continuous_retention_group")) for r in candidates),
        "candidate_h4800": sum(int_flag(r.get("productive_h4800_group")) for r in candidates),
        "candidate_h6400": sum(finite_float(r.get("h6400")) >= RETENTION_EPS and int_flag(r.get("productive_h4800_group")) for r in candidates),
        "late_rebound_groups": sum(int_flag(r.get("late_rebound_group")) for r in summary),
        "terminal_collapse_groups": sum(int_flag(r.get("terminal_collapse_group")) for r in summary),
        "control_equivalent_groups": sum(int_flag(r.get("control_equivalent_group")) for r in summary),
    }
    if route["candidate_continuous_h3200"] == 0:
        route["decision"] = "R0-SourceChainRuleInflatedPreviousRoute"
    elif route["candidate_h4800"] == 0:
        route["decision"] = "TerminalCollapseAutopsyRequired"
    else:
        route["decision"] = "H4800CandidateRequiresIndependentConfirmation"
    write_json(out_dir / "v22_01_reaggregated_v22_route.json", route)
    write_rows(out_dir / "v22_01_reaggregated_v22_route.csv", [route])
    return enriched, summary, route


def classify_terminal(row: dict[str, Any], stable_fraction: float, candidate_fraction: float) -> str:
    source_drop = finite_float(row.get("source_derivative_h3200_to_h4800"))
    debt_spike = max(
        finite_float(row.get("CEp99_debt_peak"), 0.0),
        finite_float(row.get("ECE_debt_peak"), 0.0),
        finite_float(row.get("Brier_debt_peak"), 0.0),
        finite_float(row.get("LineC_channel_loss_debt_peak"), 0.0),
    )
    act = finite_float(row.get("ActuationR2_h800"))
    linec = finite_float(row.get("LineC_channel_loss_h4800"))
    if stable_fraction >= candidate_fraction * 0.5 and stable_fraction > 0.0:
        return "ControlDrift"
    if finite_float(row.get("dataset_seed_heterogeneity_score"), 0.0) >= 0.95 and int(row.get("h4800_positive_rows", 0) or 0) > 0:
        return "DatasetHeterogeneity"
    if math.isfinite(source_drop) and source_drop < -0.2 and debt_spike > 0.0:
        return "DebtCollapse"
    if math.isfinite(act) and act >= 0.5 and math.isfinite(linec) and linec > 0.5:
        return "ReservoirTarget"
    return "UnknownTerminalCollapse"


def autopsy(enriched: list[dict[str, Any]], summary: list[dict[str, Any]], out_dir: Path) -> list[dict[str, Any]]:
    terminal_ids = {str(r.get("v22_id")) for r in summary if int_flag(r.get("terminal_collapse_group")) and not int_flag(r.get("control_equivalent_group"))}
    stable_rows = [r for r in enriched if "StableRandom" in str(r.get("v22_id", "")) or "RandomMatched" in str(r.get("v22_id", ""))]
    stable_fraction = sum(finite_float(r.get("source_h4800")) >= RETENTION_EPS for r in stable_rows) / max(1, len(stable_rows))
    cand_rows = [r for r in enriched if str(r.get("v22_id")) in terminal_ids]
    candidate_fraction = sum(finite_float(r.get("source_h4800")) >= RETENTION_EPS for r in cand_rows) / max(1, len(cand_rows))
    rows = []
    for group in summary:
        if str(group.get("v22_id")) not in terminal_ids:
            continue
        members = [r for r in enriched if str(r.get("v22_id")) == str(group.get("v22_id")) and str(r.get("carrier")) == str(group.get("carrier")) and str(r.get("basis_repair_variant")) == str(group.get("variant"))]
        pos = [r for r in members if finite_float(r.get("source_h4800")) >= RETENTION_EPS]
        by_dataset: dict[str, int] = defaultdict(int)
        by_seed: dict[str, int] = defaultdict(int)
        for r in pos:
            by_dataset[str(r.get("dataset", ""))] += 1
            by_seed[str(r.get("seed", ""))] += 1
        hetero = max(max(by_dataset.values(), default=0), max(by_seed.values(), default=0)) / max(1, len(pos))
        row = {
            **group,
            "source_derivative_h800_to_h1600": finite_float(group.get("h1600")) - finite_float(group.get("h800")),
            "source_derivative_h1600_to_h2400": finite_float(group.get("h2400")) - finite_float(group.get("h1600")),
            "source_derivative_h2400_to_h3200": finite_float(group.get("h3200")) - finite_float(group.get("h2400")),
            "source_derivative_h3200_to_h4800": finite_float(group.get("h4800")) - finite_float(group.get("h3200")),
            "retention_h3200_over_h800": max(0.0, finite_float(group.get("h3200"))) / max(1.0e-12, finite_float(group.get("h800"))),
            "retention_h4800_over_h3200": max(0.0, finite_float(group.get("h4800"))) / max(1.0e-12, finite_float(group.get("h3200"))),
            "optimizer_cumulative_projection_on_source": "",
            "optimizer_projection_status": "not_measured_in_v22_raw_artifacts",
            "h4800_positive_rows": len(pos),
            "dataset_seed_heterogeneity_score": hetero,
            "stable_random_h4800_positive_fraction": stable_fraction,
            "candidate_h4800_positive_fraction": candidate_fraction,
            "control_equivalent_fraction": stable_fraction,
        }
        for metric in ["CEp99", "NLL", "ECE", "Brier", "LineC_channel_loss", "LineC_fast_loss"]:
            vals = []
            for h in (100, 400, 800, 1600, 2400, 3200, 4800):
                vals.append(mean(members, f"{metric}_h{h}"))
            debt = debt_peak_final(vals, baseline=vals[0] if vals else 0.0)
            row[f"{metric}_debt_peak"] = debt["debt_peak"]
            row[f"{metric}_debt_final"] = debt["debt_final"]
            row[f"{metric}_debt_recovery"] = debt["debt_recovery"]
        row["terminal_collapse_class"] = classify_terminal(row, stable_fraction, candidate_fraction)
        rows.append(row)
    write_rows(out_dir / "v22_01_terminal_collapse_autopsy.csv", rows)
    class_rows = []
    for cls in sorted({str(r.get("terminal_collapse_class")) for r in rows}):
        cr = [r for r in rows if r.get("terminal_collapse_class") == cls]
        class_rows.append({"class": cls, "groups": len(cr), "mean_h3200": mean(cr, "h3200"), "mean_h4800": mean(cr, "h4800")})
    write_rows(out_dir / "v22_01_terminal_collapse_taxonomy.csv", class_rows)
    simple_svg(out_dir / "figures" / "source_trajectory_by_candidate.svg", "Source trajectory by candidate", rows, "h4800")
    simple_svg(out_dir / "figures" / "terminal_collapse_waterfall.svg", "Terminal collapse waterfall", rows, "source_derivative_h3200_to_h4800")
    simple_svg(out_dir / "figures" / "debt_recovery_heatmap.svg", "Debt recovery heatmap", rows, "CEp99_debt_recovery")
    return rows


def main() -> None:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir)
    source_dir = Path(args.source_dir)
    enriched, summary, route = reaggregate(source_dir, out_dir)
    autopsy_rows = autopsy(enriched, summary, out_dir)
    append_exec(out_dir, f"{PYTHON} experiments/run_v22_01_terminal_collapse_autopsy.py --source-dir {source_dir}", status="completed", note=f"reaggregated_groups={len(summary)} terminal_groups={len(autopsy_rows)} decision={route.get('decision')}")


if __name__ == "__main__":
    main()
