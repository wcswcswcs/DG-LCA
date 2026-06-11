#!/usr/bin/env python3
"""v22.04 terminal-erosion taxonomy aggregation."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dgkan.fu.terminal_erosion import classify_terminal_erosion_v22_04  # noqa: E402
from experiments.run_v22_04_common import PYTHON, V2203_OFFICIAL, append_exec, ensure_out, finite_float, int_flag, read_rows, simple_svg, write_json, write_rows  # noqa: E402


V2203_AUDITS = [
    "v22_03_f145_f147_source_preserve_audit.csv",
    "v22_03_f148_f150_diffeomorphic_target_audit.csv",
    "v22_03_f151_f153_terminal_preserve_repair_audit.csv",
    "v22_03_f154_f156_terminal_debt_lownds_dualmem_audit.csv",
    "v22_03_f157_f159_signal_estimator_audit.csv",
    "v22_03_f160_f162_post_h4000_source_floor_audit.csv",
    "v22_03_f163_f165_raw_guard_post_h4000_floor_audit.csv",
    "v22_03_f166_f168_anti_erosion_audit.csv",
    "v22_03_f169_f171_h3200_anchor_transport_audit.csv",
    "v22_03_f172_f174_progress_carry_audit.csv",
    "v22_03_f175_f177_gentle_progress_audit.csv",
    "v22_03_f178_f180_control_relative_catchup_audit.csv",
    "v22_03_f181_f183_trajectory_adaptive_audit.csv",
    "v22_03_f184_f186_minimal_transport_audit.csv",
    "v22_03_f187_f189_accept_memory_audit.csv",
]

PRE_WIRING_AUDITS = {
    "v22_04_d1a_lambda_strength_audit.csv",
    "v22_04_d1a_lambda_strength_wired_audit.csv",
    "v22_04_d1a_lambda_strength_wired2_audit.csv",
}


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None)
    p.add_argument("--include-v2203", type=int, default=1)
    return p


def _candidate(row: dict[str, Any]) -> bool:
    return str(row.get("mechanism", "")).startswith("M") and not str(row.get("v22_id", "")).startswith("CTRL")


def main() -> None:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir)
    rows: list[dict[str, Any]] = []
    if args.include_v2203:
        for name in V2203_AUDITS:
            for row in read_rows(V2203_OFFICIAL / name):
                if _candidate(row):
                    row = dict(row)
                    row["autopsy_source"] = str(V2203_OFFICIAL / name)
                    rows.append(row)
    for path in sorted(out_dir.glob("v22_04_*_audit.csv")):
        if path.name == "v22_04_terminal_erosion_autopsy.csv":
            continue
        if path.name in PRE_WIRING_AUDITS:
            continue
        for row in read_rows(path):
            if _candidate(row):
                row = dict(row)
                row["autopsy_source"] = str(path)
                rows.append(row)
    out = []
    for row in rows:
        audit = classify_terminal_erosion_v22_04(row)
        out.append(
            {
                "autopsy_source": row.get("autopsy_source", ""),
                "v22_id": row.get("v22_id", ""),
                "plan_line": row.get("plan_line", ""),
                "h1600": row.get("h1600", ""),
                "h2400": row.get("h2400", ""),
                "h3200": row.get("h3200", ""),
                "h4000": row.get("h4000", ""),
                "h4800": row.get("h4800", ""),
                "h4800_retention_ratio": row.get("h4800_retention_ratio", ""),
                "row_h4800_positive_count": row.get("row_h4800_positive_count", ""),
                **audit.to_row(),
            }
        )
    counts = Counter(str(r.get("terminal_erosion_class", "")) for r in out)
    total = len(out)
    explained = sum(int_flag(r.get("terminal_erosion_explained")) for r in out)
    summary = [
        {"terminal_erosion_class": cls, "groups": n, "fraction": (n / total if total else 0.0)}
        for cls, n in sorted(counts.items())
    ]
    route = {
        "autopsy_groups": total,
        "explained_groups": explained,
        "explained_fraction": explained / total if total else 0.0,
        "decision": "TerminalErosionTaxonomyCoveragePass" if total and explained / total >= 0.70 else "TerminalErosionTaxonomyCoverageBlocked",
        "blocker": "" if total and explained / total >= 0.70 else "coverage_below_70pct_or_no_rows",
    }
    write_rows(out_dir / "v22_04_terminal_erosion_autopsy.csv", out)
    write_rows(out_dir / "v22_04_terminal_erosion_class_summary.csv", summary)
    write_rows(out_dir / "v22_04_terminal_erosion_route.csv", [route])
    write_json(out_dir / "v22_04_terminal_erosion_route.json", route)
    simple_svg(out_dir / "figures" / "v22_04_terminal_erosion_h4800.svg", "v22.04 terminal erosion autopsy", out, "h4800")
    append_exec(
        out_dir,
        f"{PYTHON} experiments/run_v22_04_terminal_erosion_autopsy.py --out-dir {out_dir} --include-v2203 {args.include_v2203}",
        status="completed",
        note=f"groups={total} explained_fraction={route['explained_fraction']}",
    )


if __name__ == "__main__":
    main()
