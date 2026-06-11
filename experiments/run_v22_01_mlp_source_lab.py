#!/usr/bin/env python3
"""v22.01 F2/F3 MLP source-lab and precommit-selector audit."""

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
    V2200_OFFICIAL,
    append_exec,
    ensure_out,
    finite_float,
    int_flag,
    mean,
    read_json,
    read_rows,
    simple_svg,
    write_json,
    write_rows,
)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None)
    p.add_argument("--source-dir", default=str(V2200_OFFICIAL))
    return p


def family(v22_id: str) -> str:
    if v22_id.startswith("MLP-F1"):
        return "F2-M1-strong-source-replay"
    if v22_id.startswith("MLP-F2"):
        return "F2-M2-weak-stable-replay"
    if any(v22_id.startswith(prefix) for prefix in ["MLP-F53", "MLP-F63", "MLP-F70", "MLP-F71", "MLP-F72", "MLP-F73", "MLP-F74"]):
        return "F2-M3-terminal-family-replay"
    if any(v22_id.startswith(prefix) for prefix in ["MLP-F40", "MLP-F41", "MLP-F44", "MLP-F45"]):
        return "F2-M4-slow-anchor-boundary-family"
    if any(v22_id.startswith(prefix) for prefix in ["MLP-F57", "MLP-F58", "MLP-F59"]):
        return "F2-M5-fast-slow-memory-source-writer"
    if any(v22_id.startswith(prefix) for prefix in ["MLP-F60", "MLP-F61"]):
        return "F2-M7-matrix-readout-channel-writer"
    if any(v22_id.startswith(prefix) for prefix in ["CTRL-", "MLP-CTRL"]):
        return "F2-M10-control"
    return "F2-other"


def main() -> None:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir)
    matrix = read_rows(out_dir / "v22_01_reaggregated_v22_source_chain.csv")
    if not matrix:
        matrix = read_rows(Path(args.source_dir) / "v22_source_chain_matrix.csv")
    summary = read_rows(out_dir / "v22_01_reaggregated_v22_source_chain_summary.csv")
    mlp = [r for r in summary if str(r.get("carrier")) == "MLP"]
    control_h4800 = max([finite_float(r.get("h4800"), -999.0) for r in mlp if "CTRL" in str(r.get("v22_id", ""))] or [-999.0])
    rows = []
    for row in mlp:
        members = [m for m in matrix if str(m.get("carrier")) == "MLP" and str(m.get("v22_id", m.get("v21_id", ""))) == str(row.get("v22_id"))]
        h4800_pos = sum(finite_float(m.get("source_h4800")) >= RETENTION_EPS for m in members)
        h6400_pos = sum(finite_float(m.get("source_h6400")) >= RETENTION_EPS for m in members)
        h4800_ret = finite_float(row.get("h4800")) / max(1.0e-12, finite_float(row.get("h3200"))) if finite_float(row.get("h3200")) > 0 else 0.0
        h6400_ret = finite_float(row.get("h6400")) / max(1.0e-12, finite_float(row.get("h4800"))) if finite_float(row.get("h4800")) > 0 else 0.0
        s3 = int(
            int_flag(row.get("continuous_h4800_group"))
            and h4800_ret >= 0.50
            and h4800_pos >= 5
            and control_h4800 < RETENTION_EPS
        )
        s4 = int(s3 and finite_float(row.get("h6400")) >= RETENTION_EPS and h6400_ret >= 0.50 and h6400_pos >= 5)
        rows.append(
            {
                **row,
                "f2_family": family(str(row.get("v22_id", ""))),
                "h4800_positive_rows": h4800_pos,
                "h6400_positive_rows": h6400_pos,
                "retention_h4800_over_h3200": h4800_ret,
                "retention_h6400_over_h4800": h6400_ret,
                "stable_random_and_random_matched_controls_fail": int(control_h4800 < RETENTION_EPS),
                "hidden_matrix_source_projection": mean(members, "source_channel_projection_h3200"),
                "readout_source_projection": mean(members, "source_channel_projection_h4800"),
                "source_projection_decay_h3200_h4800": mean(members, "source_channel_projection_h4800") - mean(members, "source_channel_projection_h3200"),
                "fast_state_norm": "",
                "slow_state_norm": mean(members, "source_state_norm_h3200"),
                "fast_slow_cosine": mean(members, "source_state_current_cos_h3200"),
                "old_gradient_memory_cosine": "",
                "matrix_block_update_norm": mean(members, "block_update_norm"),
                "source_anchor_norm": mean(members, "source_state_norm_h4800"),
                "source_anchor_drift": mean(members, "source_state_norm_h4800") - mean(members, "source_state_norm_h3200"),
                "MLP_FU_S3": s3,
                "MLP_FU_S4": s4,
            }
        )
    write_rows(out_dir / "v22_01_mlp_source_lab_summary.csv", rows)
    s3_rows = [r for r in rows if int_flag(r.get("MLP_FU_S3"))]
    decision = {
        "decision": "MLP-FU-S3" if s3_rows else "MLPStillTerminalCollapse",
        "s3_rows": len(s3_rows),
        "s4_rows": sum(int_flag(r.get("MLP_FU_S4")) for r in rows),
        "best_h3200": max([finite_float(r.get("h3200"), -999.0) for r in rows] or [-999.0]),
        "best_h4800": max([finite_float(r.get("h4800"), -999.0) for r in rows] or [-999.0]),
        "control_h4800_best": control_h4800,
    }
    write_json(out_dir / "v22_01_mlp_source_lab_decision.json", decision)
    obs = read_rows(Path(args.source_dir) / "v22_source_observability_predictor_scan.csv")
    f46 = read_json(Path(args.source_dir) / "v22_f46_train_loss_selector_decision.json")
    selector_rows = []
    for row in obs:
        item = dict(row)
        item["precommit_selector_pass"] = 0
        item["retrospective_only_flag"] = int(str(row.get("kind")) == "train_only" and int_flag(row.get("pass")))
        item["selector_decision"] = "RetrospectiveOnly" if item["retrospective_only_flag"] else "NoPrecommitPass"
        selector_rows.append(item)
    write_rows(out_dir / "v22_01_precommit_selector_audit.csv", selector_rows)
    write_json(
        out_dir / "v22_01_precommit_selector_decision.json",
        {
            "decision": "RetrospectiveOnlyNoPrecommitSelector",
            "retrospective_f46_decision": f46.get("decision", ""),
            "passing_retrospective_train_only_predictors": sum(int_flag(r.get("retrospective_only_flag")) for r in selector_rows),
            "precommit_selector_pass": 0,
        },
    )
    simple_svg(out_dir / "figures" / "selector_roc_pr.svg", "Selector ROC/PR audit", selector_rows, "AUC_continuous_retention")
    simple_svg(out_dir / "figures" / "carrier_source_channel_map.svg", "MLP source channel map", rows, "readout_source_projection")
    append_exec(out_dir, f"{PYTHON} experiments/run_v22_01_mlp_source_lab.py --source-dir {args.source_dir}", status="completed", note=f"mlp_rows={len(rows)} decision={decision['decision']}")


if __name__ == "__main__":
    main()
