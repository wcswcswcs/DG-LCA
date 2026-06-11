#!/usr/bin/env python3
"""v22.03 C0 terminal-erosion autopsy from measured fresh artifacts."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dgkan.fu.terminal_retention import classify_terminal_erosion, classify_v22_03_source_chain  # noqa: E402
from experiments.run_v22_03_common import (  # noqa: E402
    PYTHON,
    V2202_OFFICIAL,
    append_exec,
    ensure_out,
    finite_float,
    int_flag,
    read_rows,
    simple_svg,
    write_json,
    write_rows,
)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None)
    p.add_argument("--baseline-audit", default=str(V2202_OFFICIAL / "v22_02_f106_f144_oldshape_full_audit.csv"))
    return p


def main() -> None:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir)
    rows = read_rows(args.baseline_audit)
    autopsy: list[dict[str, Any]] = []
    for row in rows:
        if str(row.get("v22_id", "")).startswith("CTRL"):
            continue
        decision = classify_v22_03_source_chain(
            row.get("h100"),
            row.get("h400"),
            row.get("h800"),
            row.get("h1600"),
            row.get("h2400"),
            row.get("h3200"),
            row.get("h4000"),
            row.get("h4800"),
            "",
            row_h4800_positive_count=row.get("row_h4800_positive_count", ""),
        )
        if not decision.continuous_h3200_chain:
            continue
        item = dict(row)
        item.update(
            {
                "v22_03_early_source_chain": decision.early_source_chain,
                "v22_03_continuous_h3200_chain": decision.continuous_h3200_chain,
                "v22_03_productive_h4800_chain": decision.productive_h4800_chain,
                "v22_03_terminal_erosion": decision.terminal_erosion,
                "v22_03_terminal_collapse": decision.terminal_collapse,
                "v22_03_blocker": decision.blocker,
                "terminal_erosion_class": classify_terminal_erosion(row),
                "source_derivative_h3200_h4000": (
                    (finite_float(row.get("h4000")) - finite_float(row.get("h3200"))) / 800.0
                    if finite_float(row.get("h4000")) == finite_float(row.get("h4000"))
                    and finite_float(row.get("h3200")) == finite_float(row.get("h3200"))
                    else ""
                ),
                "source_derivative_h4000_h4800": (
                    (finite_float(row.get("h4800")) - finite_float(row.get("h4000"))) / 800.0
                    if finite_float(row.get("h4800")) == finite_float(row.get("h4800"))
                    and finite_float(row.get("h4000")) == finite_float(row.get("h4000"))
                    else ""
                ),
            }
        )
        autopsy.append(item)
    class_counts = Counter(str(r.get("terminal_erosion_class", "")) for r in autopsy)
    class_rows = [
        {"terminal_erosion_class": name, "groups": count, "fraction": count / len(autopsy) if autopsy else ""}
        for name, count in sorted(class_counts.items())
    ]
    non_unknown = sum(count for name, count in class_counts.items() if name not in {"MeasuredTerminalErosionUnresolved", "UnknownTerminalCollapse", ""})
    route = {
        "baseline_audit": str(args.baseline_audit),
        "continuous_h3200_groups": len(autopsy),
        "productive_h4800_groups": sum(int_flag(r.get("v22_03_productive_h4800_chain")) for r in autopsy),
        "terminal_erosion_groups": sum(int_flag(r.get("v22_03_terminal_erosion")) for r in autopsy),
        "terminal_collapse_groups": sum(int_flag(r.get("v22_03_terminal_collapse")) for r in autopsy),
        "non_unknown_classified_groups": non_unknown,
        "non_unknown_classified_fraction": non_unknown / len(autopsy) if autopsy else "",
        "decision": "TerminalErosionExplained" if autopsy and non_unknown / len(autopsy) >= 0.80 else "TerminalErosionPartiallyExplained",
    }
    write_rows(out_dir / "v22_03_terminal_erosion_autopsy.csv", autopsy)
    write_rows(out_dir / "v22_03_terminal_erosion_class_summary.csv", class_rows)
    write_rows(out_dir / "v22_03_terminal_erosion_route.csv", [route])
    write_json(out_dir / "v22_03_terminal_erosion_route.json", route)
    simple_svg(out_dir / "figures/v22_03_terminal_erosion_h4800_ratio.svg", "v22.03 terminal erosion h4800 ratio", autopsy, "h4800_retention_ratio")
    append_exec(out_dir, f"{PYTHON} experiments/run_v22_03_terminal_erosion_autopsy.py --baseline-audit {args.baseline_audit}", status="completed", note=f"groups={len(autopsy)} decision={route['decision']}")


if __name__ == "__main__":
    main()

