#!/usr/bin/env python3
"""Aggregate v22.04 D1 source-preserving FU continuations."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_v22_03_source_preservation import summarize as summarize_v2203  # noqa: E402
from experiments.run_v22_04_common import (  # noqa: E402
    PYTHON,
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
    p.add_argument("--source-dir", required=True)
    p.add_argument("--tag", default="d1_source_preserving_fu")
    p.add_argument("--label", default="D1 source-preserving FU oldshape full")
    return p


def _candidate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in rows if str(r.get("mechanism", "")).startswith("M") and not str(r.get("v22_id", "")).startswith("CTRL")]


def _line(v22_id: str) -> str:
    if "D1a" in v22_id:
        return "D1a-source-preserving-late-projection"
    if "D1b" in v22_id:
        return "D1b-Nora-row-orthogonal"
    if "D1c" in v22_id:
        return "D1c-low-NDS-matrix-block"
    if "D1d" in v22_id:
        return "D1d-dual-memory-source-state"
    if "D1e" in v22_id:
        return "D1e-debt-aware-terminal-FU"
    if "D1f" in v22_id:
        return "D1f-info-volume-preserving-FU"
    return ""


def _rename_row_keys(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        item = dict(row)
        item["plan_line"] = _line(str(item.get("v22_id", "")))
        for old, new in [
            ("v22_03_early_source_chain", "v22_04_early_source_chain"),
            ("v22_03_continuous_h3200_chain", "v22_04_continuous_h3200_chain"),
            ("v22_03_productive_h4800_chain", "v22_04_productive_h4800_chain"),
            ("v22_03_terminal_erosion", "v22_04_terminal_erosion"),
            ("v22_03_source_chain_blocker", "v22_04_source_chain_blocker"),
        ]:
            if old in item:
                item[new] = item.pop(old)
        out.append(item)
    return out


def main() -> None:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir)
    source_dir = Path(args.source_dir)
    audit, dataset_localization, row_localization = summarize_v2203(source_dir, args.label)
    for row in audit:
        row["plan_line"] = _line(str(row.get("v22_id", "")))
        row["v22_04_productive_gate"] = "mean_h4800>=0.005;R4800_over_3200>=0.50;row_positive>=7/9"
    dataset_localization = _rename_row_keys(dataset_localization)
    row_localization = _rename_row_keys(row_localization)
    tag = str(args.tag).strip() or "d1_source_preserving_fu"
    write_rows(out_dir / f"v22_04_{tag}_audit.csv", audit)
    write_rows(out_dir / f"v22_04_{tag}_dataset_localization.csv", dataset_localization)
    write_rows(out_dir / f"v22_04_{tag}_row_localization.csv", row_localization)
    candidates = _candidate_rows(audit)
    best = sorted(candidates, key=lambda r: finite_float(r.get("h4800_retention_ratio"), -1.0), reverse=True)[0] if candidates else {}
    route = {
        "continuation": args.label,
        "source_dir": str(source_dir),
        "source_chain_rows": len(read_rows(source_dir / "v21_01_source_retention_matrix.csv")),
        "candidate_groups": len(candidates),
        "candidate_early_chain": sum(int_flag(r.get("early_source_chain_group")) for r in candidates),
        "candidate_continuous_h3200": sum(int_flag(r.get("continuous_h3200_group")) for r in candidates),
        "candidate_h4800": sum(int_flag(r.get("productive_h4800_group")) for r in candidates),
        "terminal_erosion_groups": sum(int_flag(r.get("terminal_erosion_group")) for r in candidates),
        "best_v22_id": best.get("v22_id", ""),
        "best_plan_line": best.get("plan_line", ""),
        "best_h3200": best.get("h3200", ""),
        "best_h4000": best.get("h4000", ""),
        "best_h4800": best.get("h4800", ""),
        "best_h4800_retention_ratio": best.get("h4800_retention_ratio", ""),
        "promotion_allowed": int(any(int_flag(r.get("productive_h4800_group")) for r in candidates)),
        "decision": "HasProductiveH4800Candidate" if any(int_flag(r.get("productive_h4800_group")) for r in candidates) else "NoH4800Candidate",
    }
    write_rows(out_dir / f"v22_04_{tag}_route.csv", [route])
    write_json(out_dir / f"v22_04_{tag}_route.json", route)
    simple_svg(out_dir / "figures" / f"v22_04_{tag}_h4800.svg", f"v22.04 {tag} h4800", candidates, "h4800")
    append_exec(
        out_dir,
        f"{PYTHON} experiments/run_v22_04_source_preserving_fu.py --source-dir {source_dir} --tag {tag} --label '{args.label}' --out-dir {out_dir}",
        status="completed",
        note=f"candidate_groups={route['candidate_groups']} productive_h4800={route['candidate_h4800']} best={route['best_v22_id']}",
    )


if __name__ == "__main__":
    main()
