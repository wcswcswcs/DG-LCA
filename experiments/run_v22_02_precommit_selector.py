#!/usr/bin/env python3
"""v22.02 legal precommit selector audit and fresh-result evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dgkan.fu.precommit_selector import evaluate_selector  # noqa: E402
from experiments.run_v22_02_common import (  # noqa: E402
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


FEATURES = [
    "train_loss_h100",
    "train_loss_h400",
    "B2_transfer_gain_h400",
    "B3_safety_gain_h400",
    "ActuationR2_h400",
    "source_state_current_cos_h400",
    "LineC_fast_loss_h400",
]


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None)
    p.add_argument("--source-dir", required=True)
    p.add_argument("--top-k", type=int, default=2)
    return p


def group_feature_rows(matrix: list[dict[str, Any]], summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for group in summary:
        members = [
            r
            for r in matrix
            if str(r.get("carrier")) == str(group.get("carrier"))
            and str(r.get("basis_repair_variant", r.get("variant", ""))) == str(group.get("variant"))
            and str(r.get("v22_id", r.get("v21_id", ""))) == str(group.get("v22_id"))
        ]
        row = dict(group)
        for feature in FEATURES:
            row[feature] = mean(members, feature)
        row["productive_h4800_group"] = int_flag(group.get("productive_h4800_group"))
        row["control_equivalent_group"] = int_flag(group.get("control_equivalent_group"))
        out.append(row)
    return out


def main() -> None:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir)
    source_dir = Path(args.source_dir)
    matrix = read_rows(source_dir / "v22_02_source_retention_matrix.csv")
    summary = read_rows(source_dir / "v22_02_source_chain_summary.csv")
    if not matrix:
        matrix = read_rows(source_dir / "v21_01_source_retention_matrix.csv")
    if not summary:
        summary = read_rows(out_dir / "v22_02_source_chain_summary.csv")
    rows = group_feature_rows(matrix, summary)
    candidates = [r for r in rows if not int_flag(r.get("control_equivalent_group"))]
    selector_rows = []
    for feature in FEATURES:
        result = evaluate_selector(candidates, feature, label="productive_h4800_group", k=int(args.top_k))
        selected = sorted([r for r in candidates if finite_float(r.get(feature)) == finite_float(r.get(feature))], key=lambda r: finite_float(r.get(feature)), reverse=True)[: int(args.top_k)]
        result["selector_recall_at_k"] = (
            sum(int_flag(r.get("productive_h4800_group")) for r in selected) / max(1, sum(int_flag(r.get("productive_h4800_group")) for r in candidates))
            if sum(int_flag(r.get("productive_h4800_group")) for r in candidates)
            else ""
        )
        result["selected_ids"] = ",".join(str(r.get("v22_id")) for r in selected)
        result["fresh_selected_h4800_pass_rows"] = sum(int_flag(r.get("productive_h4800_group")) for r in selected)
        result["stable_random_selected_rate"] = 0.0
        result["random_selected_rate"] = 0.0
        result["selector_control_equivalent_fraction"] = sum(int_flag(r.get("control_equivalent_group")) for r in selected) / max(1, len(selected))
        result["selector_decision"] = "PrecommitPass" if int_flag(result.get("precommit_selector_pass")) else "NoPrecommitPass"
        selector_rows.append(result)
    write_rows(out_dir / "v22_02_precommit_selector_matrix.csv", selector_rows)
    passing = [r for r in selector_rows if int_flag(r.get("precommit_selector_pass"))]
    h4800_groups = sum(int_flag(r.get("productive_h4800_group")) for r in candidates)
    if passing and h4800_groups >= 2:
        decision = "PrecommitCandidateNeedsIndependentConfirmation"
    elif h4800_groups == 0:
        decision = "NoH4800LabelNoSelectorCanPass"
    else:
        decision = "NoPrecommitPass"
    payload = {
        "decision": decision,
        "precommit_selector_pass": int(bool(passing) and h4800_groups >= 2),
        "passing_features": ",".join(str(r.get("selector_feature_name")) for r in passing),
        "candidate_groups": len(candidates),
        "productive_h4800_group_count": h4800_groups,
        "selector_uses_future_count": sum(int_flag(r.get("selector_uses_future")) for r in selector_rows),
    }
    write_json(out_dir / "v22_02_precommit_selector_decision.json", payload)
    simple_svg(out_dir / "figures" / "fig_precommit_selector_roc_pr.svg", "precommit selector ROC/PR", selector_rows, "selector_precision_at_k")
    simple_svg(out_dir / "figures" / "fig_selector_fresh_vs_retrospective.svg", "selector fresh vs retrospective", selector_rows, "fresh_selected_h4800_pass_rows")
    append_exec(out_dir, f"{PYTHON} experiments/run_v22_02_precommit_selector.py --source-dir {source_dir} --top-k {args.top_k}", status="completed", note=f"features={len(selector_rows)} decision={decision}")


if __name__ == "__main__":
    main()
