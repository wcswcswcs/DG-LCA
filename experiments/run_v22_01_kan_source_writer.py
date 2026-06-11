#!/usr/bin/env python3
"""v22.01 F5 KAN source-channel writer audit."""

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


WRITER_IDS = ("KSW", "F36-", "F38-", "F66", "F67", "F68", "F69")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None)
    return p


def writer_family(carrier: str, v22_id: str) -> str:
    if v22_id.startswith("KSW1") or "readout-commit" in v22_id:
        return f"{carrier}-F5a-readout-commit"
    if v22_id.startswith("KSW2") or "lowdegree" in v22_id or "lowbank" in v22_id:
        return f"{carrier}-F5b-low-degree-low-frequency-bank"
    if "adamw-boundary" in v22_id:
        return f"{carrier}-F5f-source-conserving-slow-state"
    return f"{carrier}-F5-other"


def main() -> None:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir)
    matrix = read_rows(out_dir / "v22_01_reaggregated_v22_source_chain.csv")
    summary = read_rows(out_dir / "v22_01_reaggregated_v22_source_chain_summary.csv")
    eff = read_rows(out_dir / "v22_01_efficiency_officialization_summary.csv")
    eff_s1 = {str(r.get("carrier")): int_flag(r.get("S1_pass_rows")) for r in eff}
    writers = [r for r in summary if str(r.get("carrier")) in {"D-CHE", "D-FOU"} and str(r.get("v22_id", "")).startswith(WRITER_IDS)]
    mlp_rows = [r for r in summary if str(r.get("carrier")) == "MLP"]
    mlp_best_h4800 = max([finite_float(r.get("h4800"), -999.0) for r in mlp_rows if str(r.get("v22_id", "")).startswith("MLP-")] or [-999.0])
    rows = []
    for row in writers:
        members = [m for m in matrix if str(m.get("carrier")) == str(row.get("carrier")) and str(m.get("v22_id", m.get("v21_id", ""))) == str(row.get("v22_id"))]
        h3200_pos = sum(finite_float(m.get("source_h3200")) >= RETENTION_EPS for m in members)
        h4800_pos = sum(finite_float(m.get("source_h4800")) >= RETENTION_EPS for m in members)
        h4800_ret = finite_float(row.get("h4800")) / max(1.0e-12, finite_float(row.get("h3200"))) if finite_float(row.get("h3200")) > 0 else 0.0
        kan_specific_delta = finite_float(row.get("h4800")) - mlp_best_h4800 if mlp_best_h4800 > -900 else ""
        s2 = int(int_flag(row.get("continuous_retention_group")) and h3200_pos >= 4)
        s3 = int(s2 and int_flag(row.get("productive_h4800_group")) and h4800_ret >= 0.50 and h4800_pos >= 5 and finite_float(kan_specific_delta, -999.0) >= 0.005 and eff_s1.get(str(row.get("carrier")), 0) > 0)
        rows.append(
            {
                **row,
                "writer_family": writer_family(str(row.get("carrier")), str(row.get("v22_id", ""))),
                "basis_source_estimate_norm": mean(members, "source_bank_feature_norm_h800"),
                "readout_commit_norm": mean(members, "source_channel_projection_h800"),
                "basis_commit_norm": mean(members, "operator_parameter_norm"),
                "low_degree_energy": mean(members, "low_degree_source_energy_h800"),
                "high_degree_energy": mean(members, "high_degree_reservoir_energy_h800"),
                "low_frequency_energy": mean(members, "low_frequency_source_energy_h800"),
                "high_frequency_energy": mean(members, "high_frequency_reservoir_energy_h800"),
                "source_bank_drift": mean(members, "source_bank_feature_norm_h4800") - mean(members, "source_bank_feature_norm_h800"),
                "readout_source_projection": mean(members, "source_channel_projection_h4800"),
                "basis_source_projection": mean(members, "low_degree_source_energy_h4800"),
                "h3200_positive_rows": h3200_pos,
                "h4800_positive_rows": h4800_pos,
                "retention_h4800_over_h3200": h4800_ret,
                "KAN_specific_delta_vs_MLP_same_mechanism": kan_specific_delta,
                "same_run_efficiency_S1_rows": eff_s1.get(str(row.get("carrier")), 0),
                "KAN_FU_S2": s2,
                "KAN_FU_S3": s3,
            }
        )
    write_rows(out_dir / "v22_01_kan_source_writer_summary.csv", rows)
    s2 = sum(int_flag(r.get("KAN_FU_S2")) for r in rows)
    s3 = sum(int_flag(r.get("KAN_FU_S3")) for r in rows)
    decision = {
        "decision": "KAN-FU-S3" if s3 else ("KAN-FU-S2OnlyNoH4800" if s2 else "KANWriterNoEarlyContinuousSource"),
        "kan_s2_rows": s2,
        "kan_s3_rows": s3,
        "best_h3200": max([finite_float(r.get("h3200"), -999.0) for r in rows] or [-999.0]),
        "best_h4800": max([finite_float(r.get("h4800"), -999.0) for r in rows] or [-999.0]),
        "dche_s1_rows": eff_s1.get("D-CHE", 0),
        "dfou_s1_rows": eff_s1.get("D-FOU", 0),
    }
    write_json(out_dir / "v22_01_kan_source_writer_decision.json", decision)
    simple_svg(out_dir / "figures" / "kan_source_channel_map.svg", "KAN source channel map", rows, "readout_source_projection")
    append_exec(out_dir, f"{PYTHON} experiments/run_v22_01_kan_source_writer.py", status="completed", note=f"kan_rows={len(rows)} decision={decision['decision']}")


if __name__ == "__main__":
    main()
