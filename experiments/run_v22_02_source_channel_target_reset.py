#!/usr/bin/env python3
"""v22.02 source-channel target reset readback."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_v22_02_common import (  # noqa: E402
    PYTHON,
    V2201_OFFICIAL,
    append_exec,
    ensure_out,
    finite_float,
    int_flag,
    read_rows,
    sha256_file,
    simple_svg,
    write_json,
    write_rows,
)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None)
    p.add_argument("--source-dir", default=str(V2201_OFFICIAL))
    return p


def target_decision(row: dict[str, Any]) -> str:
    act = finite_float(row.get("ActuationR2"))
    transfer = finite_float(row.get("B2_transfer_gain"))
    random_transfer = finite_float(row.get("random_target_B2_gain"), 0.0)
    h4800 = finite_float(row.get("h4800_retention_rate"), 0.0)
    if act >= 0.20 and transfer > random_transfer + 0.05 and h4800 <= 0.0:
        return "TargetObservableNoRetention"
    if act < 0.20:
        return "ActuationSolverLowRank"
    if h4800 > 0.0:
        return "TargetCandidateNeedsFreshRerun"
    return "TargetControlEquivalentOrWeak"


def main() -> None:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir)
    source_dir = Path(args.source_dir)
    source_path = source_dir / "v22_01_function_space_target_reset.csv"
    rows = []
    for row in read_rows(source_path):
        item = dict(row)
        item["v22_02_source_artifact"] = str(source_path)
        item["v22_02_source_artifact_sha256"] = sha256_file(source_path) if source_path.exists() else ""
        item["target_reset_decision"] = target_decision(item)
        item["Target_S2"] = int(finite_float(item.get("ActuationR2")) >= 0.20 and finite_float(item.get("B2_transfer_gain")) > finite_float(item.get("random_target_B2_gain"), 0.0) + 0.05)
        item["Target_S3"] = int_flag(item.get("h3200_retention_rate")) and int_flag(item.get("Target_S2"))
        item["Target_S4"] = int(finite_float(item.get("h4800_retention_rate"), 0.0) > 0.0 and int_flag(item.get("Target_S2")))
        rows.append(item)
    write_rows(out_dir / "v22_02_source_channel_target_matrix.csv", rows)
    high_act_no_retention = sum(1 for r in rows if str(r.get("target_reset_decision")) == "TargetObservableNoRetention")
    h4800 = sum(int_flag(r.get("Target_S4")) for r in rows)
    decision = {
        "decision": "TargetObservableNoRetention" if high_act_no_retention and not h4800 else ("TargetH4800CandidateNeedsFreshRerun" if h4800 else "TargetResetNoSignal"),
        "rows": len(rows),
        "high_actuation_no_retention_rows": high_act_no_retention,
        "target_s2_rows": sum(int_flag(r.get("Target_S2")) for r in rows),
        "target_s4_rows": h4800,
        "promotion_allowed": 0,
    }
    write_json(out_dir / "v22_02_source_channel_target_decision.json", decision)
    simple_svg(out_dir / "figures" / "fig_target_actuation_vs_retention.svg", "target actuation vs retention", rows, "ActuationR2")
    append_exec(out_dir, f"{PYTHON} experiments/run_v22_02_source_channel_target_reset.py --source-dir {source_dir}", status="completed", note=f"rows={len(rows)} decision={decision['decision']} source_artifact={source_path}")


if __name__ == "__main__":
    main()
