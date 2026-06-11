#!/usr/bin/env python3
"""v22.04 D2 KAN source-channel mapping aggregation."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dgkan.fu.terminal_retention import classify_v22_03_source_chain  # noqa: E402
from experiments.run_v22_04_common import PYTHON, V2203_OFFICIAL, append_exec, ensure_out, finite_float, int_flag, mean, read_rows, simple_svg, write_json, write_rows  # noqa: E402


HORIZONS = [100, 400, 800, 1600, 2400, 3200, 4000, 4800]


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None)
    p.add_argument("--source-dir", default="")
    return p


def _summarize_fresh(source_dir: Path) -> list[dict[str, Any]]:
    rows = read_rows(source_dir / "v21_01_source_retention_matrix.csv")
    buckets: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[(str(row.get("carrier", "")), str(row.get("basis_repair_variant", "")), str(row.get("v21_id", "")))].append(row)
    out = []
    for (carrier, variant, v21_id), group in sorted(buckets.items()):
        item: dict[str, Any] = {"source": str(source_dir), "carrier": carrier, "variant": variant, "v22_id": v21_id, "rows": len(group)}
        for step in HORIZONS:
            item[f"h{step}"] = mean(group, f"source_h{step}")
        row_pos = sum(1 for row in group if finite_float(row.get("source_h4800")) >= 0.005)
        dec = classify_v22_03_source_chain(
            item.get("h100"),
            item.get("h400"),
            item.get("h800"),
            item.get("h1600"),
            item.get("h2400"),
            item.get("h3200"),
            item.get("h4000"),
            item.get("h4800"),
            control_equivalent=int(str(v21_id).startswith("CTRL")),
            row_h4800_positive_count=row_pos,
        )
        item.update(
            {
                "h4800_positive_rows": row_pos,
                "h4800_retention_ratio": dec.h4800_retention_ratio,
                "early_source_chain_group": dec.early_source_chain,
                "continuous_h3200_group": dec.continuous_h3200_chain,
                "productive_h4800_group": dec.productive_h4800_chain,
                "v22_04_source_mapping_decision": "KANMappedProductiveSource" if dec.productive_h4800_chain else "KANSourceChannelMismatch",
                "blocker": dec.blocker,
                "fresh_source_retention_run": 1,
            }
        )
        out.append(item)
    return out


def _summarize_readback() -> list[dict[str, Any]]:
    rows = read_rows(V2203_OFFICIAL / "v22_03_kan_source_channel_writer_matrix.csv")
    out = []
    for row in rows:
        item = dict(row)
        item["source"] = str(V2203_OFFICIAL / "v22_03_kan_source_channel_writer_matrix.csv")
        item["v22_04_source_mapping_decision"] = "KANMappedProductiveSource" if int_flag(row.get("KAN_FU_S3_v22_02")) else "KANSourceChannelMismatch"
        item["fresh_source_retention_run"] = 0
        out.append(item)
    return out


def main() -> None:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir)
    source_dir = Path(args.source_dir) if args.source_dir else Path("")
    fresh = bool(args.source_dir) and (source_dir / "v21_01_source_retention_matrix.csv").exists()
    rows = _summarize_fresh(source_dir) if fresh else _summarize_readback()
    candidates = [r for r in rows if str(r.get("v22_id", "")).startswith("KSW") or "RAT" in str(r.get("v22_id", "")) or "RBF" in str(r.get("v22_id", ""))]
    route = {
        "source_dir": str(source_dir) if fresh else str(V2203_OFFICIAL),
        "fresh_source_retention_run": int(fresh),
        "groups": len(rows),
        "candidate_groups": len(candidates),
        "mapped_productive_groups": sum(1 for r in candidates if str(r.get("v22_04_source_mapping_decision")) == "KANMappedProductiveSource"),
        "decision": "KANMappedProductiveSource" if any(str(r.get("v22_04_source_mapping_decision")) == "KANMappedProductiveSource" for r in candidates) else "KANSourceChannelMismatch",
        "blocker": "" if any(str(r.get("v22_04_source_mapping_decision")) == "KANMappedProductiveSource" for r in candidates) else "KAN_source_channel_mapping_missing",
    }
    write_rows(out_dir / "v22_04_kan_source_mapping_matrix.csv", rows)
    write_rows(out_dir / "v22_04_kan_source_mapping_route.csv", [route])
    write_json(out_dir / "v22_04_kan_source_mapping_decision.json", route)
    simple_svg(out_dir / "figures" / "v22_04_kan_source_mapping_h4800.svg", "v22.04 KAN source mapping h4800", candidates, "h4800")
    append_exec(
        out_dir,
        f"{PYTHON} experiments/run_v22_04_kan_source_mapping.py --out-dir {out_dir}" + (f" --source-dir {source_dir}" if fresh else ""),
        status="completed",
        note=f"fresh={int(fresh)} candidate_groups={route['candidate_groups']} mapped={route['mapped_productive_groups']}",
    )


if __name__ == "__main__":
    main()
