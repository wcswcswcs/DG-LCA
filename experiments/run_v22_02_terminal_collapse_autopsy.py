#!/usr/bin/env python3
"""v22.02 source-chain reaggregation and terminal-collapse autopsy."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dgkan.fu.debt_accounting import debt_peak_final  # noqa: E402
from dgkan.fu.source_chain import SOURCE_EPS  # noqa: E402
from dgkan.fu.terminal_collapse import classify_terminal_collapse, classify_v22_02_source_chain  # noqa: E402
from experiments.run_v22_02_common import (  # noqa: E402
    PYTHON,
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


CONTROL_PREFIXES = ("CTRL-",)
HORIZONS = (100, 400, 800, 1600, 2400, 3200, 4000, 4800, 6400)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None)
    p.add_argument("--source-dir", required=True)
    p.add_argument("--tag", default="fresh_terminal_pool")
    return p


def is_control_id(v22_id: str) -> int:
    return int(str(v22_id).startswith(CONTROL_PREFIXES) or "RandomMatched" in str(v22_id) or "stable-random" in str(v22_id))


def group_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("carrier", "")),
        str(row.get("basis_repair_variant", row.get("variant", ""))),
        str(row.get("v21_id", row.get("v22_id", ""))),
    )


def source_derivative(row: dict[str, Any], a: int, b: int) -> float | str:
    av = finite_float(row.get(f"h{a}"))
    bv = finite_float(row.get(f"h{b}"))
    return bv - av if av == av and bv == bv else ""


def debt_growth(members: list[dict[str, Any]], metric: str) -> tuple[Any, Any, Any]:
    vals = [mean(members, f"{metric}_h{h}") for h in (100, 400, 800, 1600, 2400, 3200, 4000, 4800)]
    vals = [v for v in vals if v == v]
    debt = debt_peak_final(vals, baseline=vals[0] if vals else 0.0)
    h3200 = mean(members, f"{metric}_h3200")
    h4800 = mean(members, f"{metric}_h4800")
    if h3200 == h3200 and h3200 != 0.0 and h4800 == h4800:
        growth = (h4800 - h3200) / abs(h3200)
    else:
        growth = ""
    return debt, h3200, growth


def enrich_rows(raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in raw:
        item = dict(row)
        item["v22_id"] = item.get("v21_id", item.get("v22_id", ""))
        item["control_equivalent"] = is_control_id(str(item.get("v22_id", "")))
        decision = classify_v22_02_source_chain(
            item.get("source_h100"),
            item.get("source_h400"),
            item.get("source_h800"),
            item.get("source_h1600"),
            item.get("source_h2400"),
            item.get("source_h3200"),
            item.get("source_h4800"),
            control_equivalent=item.get("control_equivalent", 0),
        )
        item["v22_02_early_source_chain"] = decision.early_source_chain
        item["v22_02_continuous_h3200_chain"] = decision.continuous_h3200_chain
        item["v22_02_productive_h4800_chain"] = decision.productive_h4800_chain
        item["v22_02_terminal_collapse"] = decision.terminal_collapse
        item["v22_02_h1600_retention_ratio"] = decision.h1600_retention_ratio
        item["v22_02_h3200_retention_ratio"] = decision.h3200_retention_ratio
        item["v22_02_h4800_retention_ratio"] = decision.h4800_retention_ratio
        item["v22_02_source_chain_blocker"] = decision.blocker
        item["h4000_recorded"] = int(str(item.get("source_h4000", "")) not in {"", "nan", "None"})
        out.append(item)
    return out


def summarize(enriched: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in enriched:
        groups[group_key(row)].append(row)
    summary = []
    for (carrier, variant, v22_id), members in sorted(groups.items()):
        h = {f"h{step}": mean(members, f"source_h{step}") for step in HORIZONS}
        control = is_control_id(v22_id)
        decision = classify_v22_02_source_chain(h["h100"], h["h400"], h["h800"], h["h1600"], h["h2400"], h["h3200"], h["h4800"], control_equivalent=control)
        pos = [r for r in members if finite_float(r.get("source_h4800")) >= SOURCE_EPS]
        by_dataset: dict[str, int] = defaultdict(int)
        by_seed: dict[str, int] = defaultdict(int)
        for row in pos:
            by_dataset[str(row.get("dataset", ""))] += 1
            by_seed[str(row.get("seed", ""))] += 1
        hetero = max(max(by_dataset.values(), default=0), max(by_seed.values(), default=0)) / max(1, len(pos))
        item = {
            "carrier": carrier,
            "variant": variant,
            "v22_id": v22_id,
            "rows": len(members),
            **h,
            "control_equivalent_group": control,
            "early_source_chain_group": decision.early_source_chain,
            "continuous_h3200_group": decision.continuous_h3200_chain,
            "productive_h4800_group": decision.productive_h4800_chain,
            "terminal_collapse_group": decision.terminal_collapse,
            "h1600_retention_ratio": decision.h1600_retention_ratio,
            "h3200_retention_ratio": decision.h3200_retention_ratio,
            "h4800_retention_ratio": decision.h4800_retention_ratio,
            "row_h4800_positive_count": len(pos),
            "h4000_recorded_rows": sum(int_flag(r.get("h4000_recorded")) for r in members),
            "dataset_seed_heterogeneity_score": hetero,
            "source_derivative_h3200_to_h4000": source_derivative({**h}, 3200, 4000),
            "source_derivative_h4000_to_h4800": source_derivative({**h}, 4000, 4800),
            "source_derivative_h3200_to_h4800": source_derivative({**h}, 3200, 4800),
            "source_chain_blocker": decision.blocker,
        }
        for metric, out_name in [("CEp99", "tail"), ("LineC_channel_loss", "LineC_channel"), ("ECE", "ECE"), ("Brier", "Brier")]:
            debt, _h3200, growth = debt_growth(members, metric)
            item[f"{out_name}_debt_peak"] = debt["debt_peak"]
            item[f"{out_name}_debt_final"] = debt["debt_final"]
            item[f"{out_name}_debt_recovery"] = debt["debt_recovery"]
            item[f"{out_name}_debt_growth_h3200_to_h4800"] = growth
        for metric in [
            "hidden_source_energy",
            "readout_source_energy",
            "bias_source_energy",
            "other_source_energy",
            "hidden_source_fraction",
            "readout_source_fraction",
            "bias_source_fraction",
            "matrix_block_alignment",
            "parameter_alignment",
            "source_reconstruction_error",
            "source_subspace_rank",
        ]:
            item[metric] = mean(members, metric)
            for step in (800, 1600, 2400, 3200, 4000, 4800):
                item[f"{metric}_h{step}"] = mean(members, f"{metric}_h{step}")
        singular_rows = [str(r.get("source_singular_values_h4800") or r.get("source_singular_values") or "") for r in members]
        item["source_singular_values"] = next((v for v in singular_rows if v), "")
        opt_state = [str(r.get("optimizer_state_source_energy_h4800") or r.get("optimizer_state_source_energy") or "") for r in members]
        item["optimizer_state_source_energy"] = next((v for v in opt_state if v), "")
        item["stable_random_h4800_source_rate"] = 0.0
        item["source_channel_projection_decay_h3200_h4800"] = mean(members, "source_channel_projection_h4800") - mean(members, "source_channel_projection_h3200")
        item["B2_transfer_gain_h800"] = mean(members, "B2_transfer_gain_h800")
        item["cumulative_optimizer_projection_on_source_h3200_to_h4800"] = ""
        item["optimizer_projection_status"] = "not_measured_by_current_runner"
        item["terminal_collapse_class"] = classify_terminal_collapse(item) if decision.terminal_collapse else ""
        summary.append(item)
    candidates = [r for r in summary if not int_flag(r.get("control_equivalent_group"))]
    route = {
        "source_chain_rows": len(enriched),
        "source_chain_groups": len(summary),
        "candidate_groups": len(candidates),
        "candidate_early_chain": sum(int_flag(r.get("early_source_chain_group")) for r in candidates),
        "candidate_continuous_h3200": sum(int_flag(r.get("continuous_h3200_group")) for r in candidates),
        "candidate_h4800": sum(int_flag(r.get("productive_h4800_group")) for r in candidates),
        "terminal_collapse_groups": sum(int_flag(r.get("terminal_collapse_group")) for r in candidates),
        "h4000_missing_group_count": sum(int_flag(r.get("h4000_recorded_rows")) == 0 for r in candidates),
        "promotion_allowed": 0,
    }
    if route["candidate_h4800"]:
        route["decision"] = "H4800CandidateNeedsIndependentConfirmation"
    elif route["candidate_continuous_h3200"]:
        route["decision"] = "TerminalCollapseClassifiedNoH4800"
    else:
        route["decision"] = "NoContinuousH3200UnderV2202Rule"
    return summary, route


def main() -> None:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir)
    source_dir = Path(args.source_dir)
    raw = read_rows(source_dir / "v21_01_source_retention_matrix.csv")
    enriched = enrich_rows(raw)
    summary, route = summarize(enriched)
    write_rows(out_dir / "v22_02_source_retention_matrix.csv", enriched)
    write_rows(out_dir / "v22_02_source_chain_summary.csv", summary)
    write_rows(out_dir / "v22_02_terminal_collapse_autopsy.csv", [r for r in summary if int_flag(r.get("terminal_collapse_group"))])
    classes = []
    for cls in sorted({str(r.get("terminal_collapse_class")) for r in summary if str(r.get("terminal_collapse_class"))}):
        rows = [r for r in summary if str(r.get("terminal_collapse_class")) == cls]
        classes.append({"class": cls, "groups": len(rows), "mean_h3200": mean(rows, "h3200"), "mean_h4800": mean(rows, "h4800")})
    write_rows(out_dir / "v22_02_terminal_collapse_taxonomy.csv", classes)
    write_json(out_dir / "v22_02_terminal_collapse_route.json", route)
    write_rows(out_dir / "v22_02_terminal_collapse_route.csv", [route])
    simple_svg(out_dir / "figures" / "fig_source_chain_horizons_by_carrier.svg", "v22.02 source chain horizons", summary, "h4800")
    simple_svg(out_dir / "figures" / "fig_terminal_collapse_waterfall_top12.svg", "terminal collapse waterfall", [r for r in summary if int_flag(r.get("terminal_collapse_group"))], "source_derivative_h3200_to_h4800")
    simple_svg(out_dir / "figures" / "fig_terminal_collapse_taxonomy_heatmap.svg", "terminal collapse taxonomy", classes, "groups")
    simple_svg(out_dir / "figures" / "fig_debt_transition_h3200_h4800.svg", "debt transition h3200 h4800", summary, "tail_debt_growth_h3200_to_h4800")
    out_arg = f" --out-dir {out_dir}" if args.out_dir else ""
    append_exec(
        out_dir,
        f"{PYTHON} experiments/run_v22_02_terminal_collapse_autopsy.py --source-dir {source_dir}{out_arg} --tag {args.tag}",
        status="completed",
        note=f"rows={len(enriched)} groups={len(summary)} decision={route.get('decision')}",
    )


if __name__ == "__main__":
    main()
