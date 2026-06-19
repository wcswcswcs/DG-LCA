#!/usr/bin/env python3
"""Audit benefit labels after matched-control projection.

This repair follows the v22.18 plan path for the blocker "controls also
positive": compare every real action against the strongest matched control in
the same dataset/seed/horizon cell.  The output is diagnostic evidence only; it
does not relax the official B/C gates.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import shlex
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_v22_18_common import OUT_ROOT, append_exec, ensure_out, finite_float, read_rows, write_rows  # noqa: E402


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--labels-csv", required=True)
    p.add_argument("--output-suffix", default="v22_18_control_projected")
    return p


def _path(raw: str) -> Path:
    p = Path(raw)
    return p if p.is_absolute() else ROOT / p


def _out(name: str, suffix: str) -> Path:
    clean = suffix.strip().strip("_")
    path = OUT_ROOT / name
    if clean:
        return path.with_name(f"{path.stem}_{clean}{path.suffix}")
    return path


def _is_control(row: dict[str, Any]) -> bool:
    return int(float(row.get("action_is_control") or 0)) == 1


def _cell(row: dict[str, Any]) -> tuple[str, str, str]:
    return (str(row.get("dataset", "")), str(row.get("seed", "")), str(row.get("horizon", "")))


def _best_control_stats(rows: list[dict[str, str]]) -> dict[str, Any]:
    controls = [r for r in rows if _is_control(r)]
    if not controls:
        return {
            "matched_control_rows": 0,
            "best_control_utility": "",
            "best_control_delta_NLL": "",
            "best_control_delta_AUC_loss_time": "",
            "best_control_delta_source_loss": "",
            "best_control_action_id_by_utility": "",
        }
    best_utility_row = max(controls, key=lambda r: finite_float(r.get("benefit_utility"), -999.0))
    best_nll_row = min(controls, key=lambda r: finite_float(r.get("delta_NLL"), 999.0))
    best_auc_row = min(controls, key=lambda r: finite_float(r.get("delta_AUC_loss_time"), 999.0))
    best_source_row = max(controls, key=lambda r: finite_float(r.get("delta_source_loss"), -999.0))
    return {
        "matched_control_rows": len(controls),
        "best_control_utility": finite_float(best_utility_row.get("benefit_utility"), -999.0),
        "best_control_delta_NLL": finite_float(best_nll_row.get("delta_NLL"), 999.0),
        "best_control_delta_AUC_loss_time": finite_float(best_auc_row.get("delta_AUC_loss_time"), 999.0),
        "best_control_delta_source_loss": finite_float(best_source_row.get("delta_source_loss"), -999.0),
        "best_control_action_id_by_utility": best_utility_row.get("action_id", ""),
    }


def _rate(rows: list[dict[str, Any]], field: str) -> float:
    return sum(int(float(r.get(field) or 0)) for r in rows) / max(1, len(rows))


def main() -> None:
    args = parser().parse_args()
    ensure_out()
    rows = read_rows(_path(args.labels_csv))
    by_cell: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_cell[_cell(row)].append(row)
    cell_stats = {cell: _best_control_stats(cell_rows) for cell, cell_rows in by_cell.items()}

    projected_rows: list[dict[str, Any]] = []
    for row in rows:
        stats = cell_stats.get(_cell(row), {})
        matched_controls = int(stats.get("matched_control_rows") or 0)
        is_control = _is_control(row)
        utility = finite_float(row.get("benefit_utility"))
        nll = finite_float(row.get("delta_NLL"))
        auc = finite_float(row.get("delta_AUC_loss_time"))
        source = finite_float(row.get("delta_source_loss"))
        best_utility = finite_float(stats.get("best_control_utility"), -999.0)
        best_nll = finite_float(stats.get("best_control_delta_NLL"), 999.0)
        best_auc = finite_float(stats.get("best_control_delta_AUC_loss_time"), 999.0)
        best_source = finite_float(stats.get("best_control_delta_source_loss"), -999.0)
        real_with_controls = int((not is_control) and matched_controls > 0)
        beats_utility = int(real_with_controls and utility > best_utility)
        beats_nll = int(real_with_controls and nll < best_nll)
        beats_source = int(real_with_controls and source > best_source)
        beats_nll_source = int(beats_nll and beats_source)
        beats_nll_auc_source = int(beats_nll and auc < best_auc and beats_source)
        projected_rows.append(
            {
                **row,
                "matched_control_rows": matched_controls,
                "best_control_utility": stats.get("best_control_utility", ""),
                "best_control_delta_NLL": stats.get("best_control_delta_NLL", ""),
                "best_control_delta_AUC_loss_time": stats.get("best_control_delta_AUC_loss_time", ""),
                "best_control_delta_source_loss": stats.get("best_control_delta_source_loss", ""),
                "best_control_action_id_by_utility": stats.get("best_control_action_id_by_utility", ""),
                "projected_benefit_utility_vs_best_control": utility - best_utility if matched_controls else "",
                "projected_delta_NLL_vs_best_control": nll - best_nll if matched_controls else "",
                "projected_delta_AUC_vs_best_control": auc - best_auc if matched_controls else "",
                "projected_delta_source_vs_best_control": source - best_source if matched_controls else "",
                "beats_best_control_utility": beats_utility,
                "beats_best_control_NLL": beats_nll,
                "beats_best_control_source": beats_source,
                "beats_best_control_NLL_and_source": beats_nll_source,
                "beats_best_control_NLL_AUC_and_source": beats_nll_auc_source,
                "control_projected_diagnostic_only": 1,
            }
        )

    real_rows = [r for r in projected_rows if not _is_control(r)]
    control_rows = [r for r in projected_rows if _is_control(r)]
    groups = sorted({(r.get("dataset", ""), r.get("seed", ""), r.get("horizon", "")) for r in projected_rows})
    summary = [
        {
            "source": "v22_18_control_projected_label_audit",
            "input_labels_csv": str(_path(args.labels_csv).relative_to(ROOT)),
            "label_rows": len(projected_rows),
            "real_rows": len(real_rows),
            "control_rows": len(control_rows),
            "matched_dataset_seed_horizon_groups": len(groups),
            "real_rate_beats_best_control_utility": _rate(real_rows, "beats_best_control_utility"),
            "real_rate_beats_best_control_NLL": _rate(real_rows, "beats_best_control_NLL"),
            "real_rate_beats_best_control_source": _rate(real_rows, "beats_best_control_source"),
            "real_rate_beats_best_control_NLL_and_source": _rate(real_rows, "beats_best_control_NLL_and_source"),
            "real_rate_beats_best_control_NLL_AUC_and_source": _rate(real_rows, "beats_best_control_NLL_AUC_and_source"),
            "strict_control_projected_B_signal_pass": int(
                0.05 <= _rate(real_rows, "beats_best_control_NLL_and_source") <= 0.40
                and _rate(real_rows, "beats_best_control_NLL_and_source") >= 0.10
            ),
            "control_projected_diagnostic_only": 1,
            "B_official_preparation_pass": 0,
            "blocker": (
                "real actions rarely beat matched controls on both NLL and source_loss"
                if _rate(real_rows, "beats_best_control_NLL_and_source") < 0.05
                else "control-projected diagnostic only; not promoted without runtime policy evidence"
            ),
        }
    ]
    action_summary: list[dict[str, Any]] = []
    for action_id in sorted({str(r.get("action_id", "")) for r in real_rows}):
        action_rows = [r for r in real_rows if str(r.get("action_id", "")) == action_id]
        action_summary.append(
            {
                "action_id": action_id,
                "real_rows": len(action_rows),
                "rate_beats_best_control_utility": _rate(action_rows, "beats_best_control_utility"),
                "rate_beats_best_control_NLL": _rate(action_rows, "beats_best_control_NLL"),
                "rate_beats_best_control_source": _rate(action_rows, "beats_best_control_source"),
                "rate_beats_best_control_NLL_and_source": _rate(action_rows, "beats_best_control_NLL_and_source"),
                "rate_beats_best_control_NLL_AUC_and_source": _rate(action_rows, "beats_best_control_NLL_AUC_and_source"),
            }
        )

    labels_out = _out("v22_18_control_projected_labels.csv", args.output_suffix)
    summary_out = _out("v22_18_control_projected_label_audit.csv", args.output_suffix)
    action_out = _out("v22_18_control_projected_action_audit.csv", args.output_suffix)
    write_rows(labels_out, projected_rows)
    write_rows(summary_out, summary)
    write_rows(action_out, action_summary)
    cmd = " ".join(shlex.quote(x) for x in [sys.executable, *sys.argv])
    append_exec(
        cmd,
        task_id="B-control-projected-label-audit",
        status="pass" if summary[0]["strict_control_projected_B_signal_pass"] else "partial",
        exit_code=0,
        files=f"{labels_out.relative_to(ROOT)}, {summary_out.relative_to(ROOT)}, {action_out.relative_to(ROOT)}",
        note=(
            "diagnostic_only=1; "
            f"utility_rate={summary[0]['real_rate_beats_best_control_utility']:.6g}; "
            f"nll_source_rate={summary[0]['real_rate_beats_best_control_NLL_and_source']:.6g}; "
            f"nll_auc_source_rate={summary[0]['real_rate_beats_best_control_NLL_AUC_and_source']:.6g}; "
            f"B_official=0; blocker={summary[0]['blocker']}"
        ),
    )


if __name__ == "__main__":
    main()
