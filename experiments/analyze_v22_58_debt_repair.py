#!/usr/bin/env python3
"""Debt-state repair report for DG-KAN v22.58 Part H."""

from __future__ import annotations

import argparse
import json
import math
import shlex
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_v22_58_state_coupled_functional_geometry_fu import (  # noqa: E402
    OUT_ROOT,
    append_exec,
    append_recap,
    collect_chunk_rows,
    fval,
    iflag,
    md_table,
    mean,
    write_json,
    write_rows,
)


DEBT_VARIANTS = {
    "scfg-state-transition-tail-aware",
    "scfg-state-transition-ece-aware",
    "scfg-state-transition-margin-aware",
    "scfg-state-transition-combined-debt",
}
MATCHED_CONTROLS = {
    "same_debt_metric_random_credit",
    "same_debt_metric_self_only",
    "same_debt_metric_hard_loss_control",
    "same_debt_metric_loss_rank_control",
}


def _proxy_sign_match(row: dict[str, Any]) -> int:
    proxy = fval(row.get("train_only_debt_proxy"), 0.0) or 0.0
    held = max(
        fval(row.get("held_ECE_delta"), 0.0) or 0.0,
        fval(row.get("held_Brier_delta"), 0.0) or 0.0,
        fval(row.get("held_tail_q95_delta"), 0.0) or 0.0,
    )
    return int((proxy > 0.0 and held > 0.0) or (proxy <= 0.0 and held <= 0.0))


def _primary_debt(row: dict[str, Any]) -> str:
    parts = {
        "ECE": fval(row.get("held_ECE_delta"), 0.0) or 0.0,
        "Brier": fval(row.get("held_Brier_delta"), 0.0) or 0.0,
        "tail_q95": fval(row.get("held_tail_q95_delta"), 0.0) or 0.0,
        "tail_q99": fval(row.get("held_tail_q99_delta"), 0.0) or 0.0,
    }
    return max(parts, key=parts.get)


def build_report(prefix: str) -> dict[str, Any]:
    rows = collect_chunk_rows([prefix])
    by_key_method = {
        (r.get("dataset", ""), str(r.get("seed", "")), r.get("method", "")): r
        for r in rows
        if r.get("run_status") == "completed"
    }
    method_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    for method in sorted({r.get("method", "") for r in rows}):
        group = [r for r in rows if r.get("method") == method]
        if not group:
            continue
        method_rows.append(
            {
                "method": method,
                "rows": len(group),
                "completed_rows": sum(int(r.get("run_status") == "completed") for r in group),
                "no_debt_rows": sum(iflag(r.get("no_ECE_Brier_tail_debt")) for r in group),
                "overhead_le_035_rows": sum(int((fval(r.get("controller_overhead_ratio"), float("inf")) or float("inf")) <= 0.35) for r in group),
                "proxy_to_held_debt_sign_match_rate": mean(_proxy_sign_match(r) for r in group),
                "mean_final_NLL": mean(fval(r.get("final_NLL")) for r in group),
                "mean_held_ECE_delta": mean(fval(r.get("held_ECE_delta")) for r in group),
                "mean_held_Brier_delta": mean(fval(r.get("held_Brier_delta")) for r in group),
                "mean_held_tail_q95_delta": mean(fval(r.get("held_tail_q95_delta")) for r in group),
                "mean_held_tail_q99_delta": mean(fval(r.get("held_tail_q99_delta")) for r in group),
                "primary_debt_mode": max(
                    {name: sum(int(_primary_debt(r) == name) for r in group) for name in ["ECE", "Brier", "tail_q95", "tail_q99"]},
                    key=lambda name: sum(int(_primary_debt(r) == name) for r in group),
                ),
            }
        )
    for row in rows:
        method = row.get("method", "")
        if method not in DEBT_VARIANTS or row.get("run_status") != "completed":
            continue
        dataset = row.get("dataset", "")
        seed = str(row.get("seed", ""))
        final_nll = fval(row.get("final_NLL"))
        control_nlls = {}
        beats = {}
        for control in sorted(MATCHED_CONTROLS):
            control_row = by_key_method.get((dataset, seed, control))
            control_nll = fval(control_row.get("final_NLL")) if control_row else None
            control_nlls[control] = control_nll
            beats[control] = int(final_nll is not None and control_nll is not None and final_nll < control_nll)
        pair_rows.append(
            {
                "dataset": dataset,
                "seed": seed,
                "method": method,
                "final_NLL": final_nll,
                "no_debt": iflag(row.get("no_ECE_Brier_tail_debt")),
                "overhead_le_035": int((fval(row.get("controller_overhead_ratio"), float("inf")) or float("inf")) <= 0.35),
                "proxy_to_held_debt_sign_match": _proxy_sign_match(row),
                "primary_debt_component": _primary_debt(row),
                "beats_same_debt_metric_random_credit": beats["same_debt_metric_random_credit"],
                "beats_same_debt_metric_self_only": beats["same_debt_metric_self_only"],
                "beats_same_debt_metric_hard_loss_control": beats["same_debt_metric_hard_loss_control"],
                "beats_same_debt_metric_loss_rank_control": beats["same_debt_metric_loss_rank_control"],
                "beats_all_matched_debt_controls": int(all(beats.values())),
                "degenerated_to_reweighting": int(not beats["same_debt_metric_hard_loss_control"] or not beats["same_debt_metric_loss_rank_control"]),
                "matched_random_credit_NLL": control_nlls["same_debt_metric_random_credit"],
                "matched_self_only_NLL": control_nlls["same_debt_metric_self_only"],
                "matched_hard_loss_NLL": control_nlls["same_debt_metric_hard_loss_control"],
                "matched_loss_rank_NLL": control_nlls["same_debt_metric_loss_rank_control"],
            }
        )
    gate_rows = []
    for method in sorted(DEBT_VARIANTS):
        group = [r for r in pair_rows if r.get("method") == method]
        n = len(group)
        gate_rows.append(
            {
                "method": method,
                "rows": n,
                "no_debt_rows": sum(int(r.get("no_debt", 0)) for r in group),
                "proxy_to_held_debt_sign_match_rate": mean(fval(r.get("proxy_to_held_debt_sign_match")) for r in group),
                "beats_all_matched_debt_controls_rows": sum(int(r.get("beats_all_matched_debt_controls", 0)) for r in group),
                "overhead_le_035_rows": sum(int(r.get("overhead_le_035", 0)) for r in group),
                "degenerated_to_reweighting_rows": sum(int(r.get("degenerated_to_reweighting", 0)) for r in group),
                "part_h_gate_pass": int(
                    n >= 15
                    and sum(int(r.get("no_debt", 0)) for r in group) >= 12
                    and (mean(fval(r.get("proxy_to_held_debt_sign_match")) for r in group) or 0.0) >= 0.70
                    and sum(int(r.get("beats_all_matched_debt_controls", 0)) for r in group) >= 10
                ),
            }
        )
    label = prefix.replace("/", "_")
    method_path = OUT_ROOT / f"{label}_debt_method_report.csv"
    pair_path = OUT_ROOT / f"{label}_debt_pairwise.csv"
    gate_path = OUT_ROOT / f"{label}_debt_gate.csv"
    json_path = OUT_ROOT / f"{label}_debt_report.json"
    write_rows(method_path, method_rows)
    write_rows(pair_path, pair_rows)
    write_rows(gate_path, gate_rows)
    summary = {
        "prefix": prefix,
        "rows": len(rows),
        "candidate_rows": len(pair_rows),
        "part_h_gate_pass_rows": sum(int(r.get("part_h_gate_pass", 0)) for r in gate_rows),
        "candidate_no_debt_rows": sum(int(r.get("no_debt", 0)) for r in pair_rows),
        "candidate_overhead_le_035_rows": sum(int(r.get("overhead_le_035", 0)) for r in pair_rows),
        "candidate_proxy_to_held_debt_sign_match_rate": mean(fval(r.get("proxy_to_held_debt_sign_match")) for r in pair_rows),
        "candidate_beats_all_matched_debt_controls_rows": sum(int(r.get("beats_all_matched_debt_controls", 0)) for r in pair_rows),
        "candidate_degenerated_to_reweighting_rows": sum(int(r.get("degenerated_to_reweighting", 0)) for r in pair_rows),
        "part_h_pass": int(any(int(r.get("part_h_gate_pass", 0)) for r in gate_rows)),
    }
    write_json(json_path, summary)
    return {
        "summary": summary,
        "method_rows": method_rows,
        "pair_rows": pair_rows,
        "gate_rows": gate_rows,
        "method_path": method_path,
        "pair_path": pair_path,
        "gate_path": gate_path,
        "json_path": json_path,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", default="v22_58_debt_repair_plan_h")
    args = parser.parse_args()
    report = build_report(args.prefix)
    command = " ".join(shlex.quote(x) for x in [sys.executable, __file__, "--prefix", args.prefix])
    append_exec(
        command,
        task_id=f"{args.prefix}_debt_specific_report",
        status="pass",
        files=(
            f"{report['method_path'].relative_to(ROOT)}; "
            f"{report['pair_path'].relative_to(ROOT)}; "
            f"{report['gate_path'].relative_to(ROOT)}; "
            f"{report['json_path'].relative_to(ROOT)}"
        ),
        note=json.dumps(report["summary"], ensure_ascii=False, sort_keys=True),
    )
    append_recap(
        f"Part H debt-specific repair audit {args.prefix}",
        "Debt repair summary:\n\n```json\n"
        + json.dumps(report["summary"], ensure_ascii=False, indent=2, sort_keys=True)
        + "\n```\n\nPer-method gate:\n\n"
        + md_table(
            report["gate_rows"],
            [
                "method",
                "rows",
                "part_h_gate_pass",
                "no_debt_rows",
                "proxy_to_held_debt_sign_match_rate",
                "beats_all_matched_debt_controls_rows",
                "overhead_le_035_rows",
                "degenerated_to_reweighting_rows",
            ],
            20,
        )
        + "\n\nMethod debt metrics:\n\n"
        + md_table(
            report["method_rows"],
            [
                "method",
                "rows",
                "no_debt_rows",
                "overhead_le_035_rows",
                "proxy_to_held_debt_sign_match_rate",
                "mean_final_NLL",
                "mean_held_ECE_delta",
                "mean_held_Brier_delta",
                "mean_held_tail_q95_delta",
                "primary_debt_mode",
            ],
            20,
        ),
    )
    print(json.dumps(report["summary"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
