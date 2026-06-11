#!/usr/bin/env python3
"""v22.05 KAN source-channel mapping readback."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_v22_05_common import PYTHON, V2204_OFFICIAL, append_exec, ensure_out, finite_float, int_flag, read_rows, simple_svg, write_json, write_rows  # noqa: E402


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None)
    p.add_argument("--source-dir", default=str(V2204_OFFICIAL))
    return p


def main() -> None:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir)
    source_dir = Path(args.source_dir)
    rows = read_rows(source_dir / "v22_04_kan_source_mapping_matrix.csv")
    out = []
    for row in rows:
        h100 = finite_float(row.get("h100"), finite_float(row.get("source_h100_mean"), float("nan")))
        h400 = finite_float(row.get("h400"), finite_float(row.get("source_h400_mean"), float("nan")))
        h800 = finite_float(row.get("h800"), finite_float(row.get("source_h800_mean"), float("nan")))
        h3200 = finite_float(row.get("h3200"), finite_float(row.get("source_h3200_mean"), float("nan")))
        h4800 = finite_float(row.get("h4800"), finite_float(row.get("source_h4800_mean"), float("nan")))
        r4800 = finite_float(row.get("h4800_retention_ratio"), finite_float(row.get("retention_h4800_over_h3200"), float("nan")))
        opened = int(all(v == v and v >= 0.005 for v in [h100, h400, h800, h3200]))
        productive = int(opened and h4800 == h4800 and h4800 >= 0.005 and r4800 == r4800 and r4800 >= 0.50)
        out.append(
            {
                **row,
                "KAN_h100_h400_h800_h3200_chain_open": opened,
                "KAN_productive_terminal_source": productive,
                "v22_05_source_mapping_decision": "KANProductiveTerminalSource" if productive else ("KANSourceChannelOpened" if opened else "KANSourceChannelMismatch"),
                "v22_05_blocker": "" if opened else "KAN_source_channel_mapping",
            }
        )
    write_rows(out_dir / "v22_05_kan_source_mapping_matrix.csv", out)
    route = {
        "source_rows": len(out),
        "opened_rows": sum(int_flag(r.get("KAN_h100_h400_h800_h3200_chain_open")) for r in out),
        "productive_rows": sum(int_flag(r.get("KAN_productive_terminal_source")) for r in out),
        "decision": "KANProductiveTerminalSource"
        if any(int_flag(r.get("KAN_productive_terminal_source")) for r in out)
        else ("KANSourceChannelOpened" if any(int_flag(r.get("KAN_h100_h400_h800_h3200_chain_open")) for r in out) else "KANSourceChannelMismatch"),
        "source_v2204": str(source_dir),
    }
    write_json(out_dir / "v22_05_kan_source_mapping_route.json", route)
    simple_svg(out_dir / "figures/KAN_source_channel_mapping_heatmap.svg", "v22.05 KAN source channel mapping", out, "h4800")
    append_exec(
        out_dir,
        f"{PYTHON} experiments/run_v22_05_kan_source_mapping.py --source-dir {source_dir} --out-dir {out_dir}",
        status="completed",
        note=f"rows={len(out)} decision={route['decision']}",
    )


if __name__ == "__main__":
    main()
