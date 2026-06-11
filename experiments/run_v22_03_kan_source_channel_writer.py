#!/usr/bin/env python3
"""v22.03 KAN source-channel writer readback."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_v22_03_common import PYTHON, V2202_OFFICIAL, append_exec, ensure_out, int_flag, read_json, read_rows, write_json, write_rows  # noqa: E402


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None)
    p.add_argument("--source-dir", default=str(V2202_OFFICIAL))
    return p


def main() -> None:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir)
    source_dir = Path(args.source_dir)
    matrix = read_rows(source_dir / "v22_02_kan_source_bank_matrix.csv")
    decision = read_json(source_dir / "v22_02_kan_source_writer_decision.json")
    out_rows = []
    for row in matrix:
        out_rows.append(
            {
                **row,
                "v22_03_source_channel_writer_decision": "KANSourceWriterS3Pass" if int_flag(row.get("KAN_FU_S3_v22_02")) else "KANSourceChannelMismatch",
            }
        )
    write_rows(out_dir / "v22_03_kan_source_channel_writer_matrix.csv", out_rows)
    write_json(out_dir / "v22_03_kan_source_channel_writer_decision.json", {"source": str(source_dir), **decision})
    append_exec(out_dir, f"{PYTHON} experiments/run_v22_03_kan_source_channel_writer.py --source-dir {source_dir}", status="completed", note=f"rows={len(out_rows)} readback only")


if __name__ == "__main__":
    main()

