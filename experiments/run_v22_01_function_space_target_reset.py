#!/usr/bin/env python3
"""v22.01 F4 function-space target reset audit."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dgkan.fu.source_chain import RETENTION_EPS  # noqa: E402
from experiments.run_v22_01_common import (  # noqa: E402
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


TARGET_IDS = ("F3-", "F10-", "F25-", "F30-", "F33-", "F35-", "F37-", "F39-")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None)
    return p


def target_family(v22_id: str) -> str:
    if "loss-cotangent" in v22_id:
        return "T1-loss-cotangent"
    if "cross-split-consensus" in v22_id or "b1-consensus" in v22_id:
        return "T2-cross-split-consensus"
    if "lowbank" in v22_id or "lowbank-loss" in v22_id:
        return "T3-low-degree-low-frequency"
    if "random-matched" in v22_id:
        return "T9-random-matched"
    if "stable-random" in v22_id:
        return "T-control-stable-random"
    return "T-other"


def main() -> None:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir)
    matrix = read_rows(out_dir / "v22_01_reaggregated_v22_source_chain.csv")
    summary = read_rows(out_dir / "v22_01_reaggregated_v22_source_chain_summary.csv")
    target_summary = [r for r in summary if str(r.get("v22_id", "")).startswith(TARGET_IDS) and str(r.get("carrier")) in {"D-CHE", "D-FOU"}]
    rows = []
    random_gain = {}
    for carrier in {"D-CHE", "D-FOU"}:
        rand_members = [m for m in matrix if str(m.get("carrier")) == carrier and "random-matched" in str(m.get("v22_id", m.get("v21_id", "")))]
        random_gain[carrier] = mean(rand_members, "B2_transfer_gain_h800")
    for row in target_summary:
        members = [m for m in matrix if str(m.get("carrier")) == str(row.get("carrier")) and str(m.get("v22_id", m.get("v21_id", ""))) == str(row.get("v22_id"))]
        early_rate = sum(int_flag(m.get("early_source_chain")) for m in members) / max(1, len(members))
        h3200_rate = sum(finite_float(m.get("source_h3200")) >= RETENTION_EPS for m in members) / max(1, len(members))
        h4800_rate = sum(finite_float(m.get("source_h4800")) >= RETENTION_EPS for m in members) / max(1, len(members))
        b2 = mean(members, "B2_transfer_gain_h800")
        b3 = mean(members, "B3_safety_gain_h800")
        rand = random_gain.get(str(row.get("carrier")), float("nan"))
        s2 = int(finite_float(mean(members, "ActuationR2_h800")) >= 0.50 and b2 > rand + 0.005 and b3 >= -0.005 and early_rate > 0.0)
        rows.append(
            {
                **row,
                "target_family": target_family(str(row.get("v22_id", ""))),
                "ActuationR2": mean(members, "ActuationR2_h800"),
                "projection_residual_norm": mean(members, "projection_residual_norm"),
                "B2_transfer_gain": b2,
                "B3_train_safety_gain": b3,
                "random_target_B2_gain": rand,
                "signflip_B2_gain": "",
                "corrupt_B2_gain": mean(members, "source_state_corrupt_gain_h800"),
                "early_source_chain_rate": early_rate,
                "h3200_retention_rate": h3200_rate,
                "h4800_retention_rate": h4800_rate,
                "debt_recovery_rate": "",
                "Target_S2": s2,
                "Target_S3": int(s2 and int_flag(row.get("productive_h4800_group"))),
            }
        )
    write_rows(out_dir / "v22_01_function_space_target_reset.csv", rows)
    s2 = sum(int_flag(r.get("Target_S2")) for r in rows)
    s3 = sum(int_flag(r.get("Target_S3")) for r in rows)
    decision = {
        "decision": "Target-S3" if s3 else ("Target-S2OnlyNoRetention" if s2 else "TargetObservabilityNoRetention"),
        "target_s2_rows": s2,
        "target_s3_rows": s3,
        "best_actuation_r2": max([finite_float(r.get("ActuationR2"), -999.0) for r in rows] or [-999.0]),
        "best_h4800": max([finite_float(r.get("h4800"), -999.0) for r in rows] or [-999.0]),
    }
    write_json(out_dir / "v22_01_function_space_target_reset_decision.json", decision)
    simple_svg(out_dir / "figures" / "target_contrast_actuation_vs_retention.svg", "Actuation vs retention", rows, "h4800")
    append_exec(out_dir, f"{PYTHON} experiments/run_v22_01_function_space_target_reset.py", status="completed", note=f"target_rows={len(rows)} decision={decision['decision']}")


if __name__ == "__main__":
    main()
