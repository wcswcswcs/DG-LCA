#!/usr/bin/env python3
"""Materialize strict benefit labels as an offline diagnostic label bank."""

from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import sys
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_v22_18_common import OUT_ROOT, append_exec, ensure_out, finite_float, read_rows, write_rows  # noqa: E402


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--labels-csv", required=True)
    p.add_argument(
        "--label-name",
        choices=[
            "nll_and_source_positive",
            "nll_and_auc_positive",
            "nll_auc_source_positive",
            "nll_margin_source_positive",
        ],
        default="nll_auc_source_positive",
    )
    p.add_argument("--output-suffix", default="")
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


def _nll(row: dict[str, Any]) -> float:
    return finite_float(row.get("delta_NLL"))


def _auc(row: dict[str, Any]) -> float:
    return finite_float(row.get("delta_AUC_loss_time"))


def _source(row: dict[str, Any]) -> float:
    return finite_float(row.get("delta_source_loss"))


def _defs() -> dict[str, tuple[Callable[[dict[str, Any]], bool], str]]:
    return {
        "nll_and_source_positive": (lambda r: _nll(r) < 0.0 and _source(r) > 0.0, "delta_NLL<0 and delta_source_loss>0"),
        "nll_and_auc_positive": (lambda r: _nll(r) < 0.0 and _auc(r) < 0.0, "delta_NLL<0 and delta_AUC_loss_time<0"),
        "nll_auc_source_positive": (
            lambda r: _nll(r) < 0.0 and _auc(r) < 0.0 and _source(r) > 0.0,
            "delta_NLL<0 and delta_AUC_loss_time<0 and delta_source_loss>0",
        ),
        "nll_margin_source_positive": (lambda r: _nll(r) < -0.001 and _source(r) > 0.0, "delta_NLL<-0.001 and delta_source_loss>0"),
    }


def main() -> None:
    args = parser().parse_args()
    ensure_out()
    rows = read_rows(_path(args.labels_csv))
    fn, definition = _defs()[args.label_name]
    strict_rows: list[dict[str, Any]] = []
    for row in rows:
        strict_positive = int(fn(row))
        strict_rows.append(
            {
                **row,
                "original_benefit_positive": row.get("benefit_positive", ""),
                "benefit_positive": strict_positive,
                "strict_label_name": args.label_name,
                "strict_label_definition": definition,
                "strict_label_diagnostic_only": 1,
                "label_source": f"{row.get('label_source', '')}+strict_offline_filter",
            }
        )
    positives = [r for r in strict_rows if int(r.get("benefit_positive") or 0)]
    real = [r for r in strict_rows if not _is_control(r)]
    control = [r for r in strict_rows if _is_control(r)]
    real_pos = [r for r in real if int(r.get("benefit_positive") or 0)]
    control_pos = [r for r in control if int(r.get("benefit_positive") or 0)]
    real_rate = len(real_pos) / max(1, len(real))
    control_rate = len(control_pos) / max(1, len(control))
    horizons = sorted({int(r.get("horizon") or 0) for r in strict_rows})
    summary = [
        {
            "source": "v22_18_strict_label_bank",
            "input_labels_csv": str(_path(args.labels_csv).relative_to(ROOT)),
            "strict_label_name": args.label_name,
            "strict_label_definition": definition,
            "label_rows": len(strict_rows),
            "positive_rows": len(positives),
            "real_rows": len(real),
            "control_rows": len(control),
            "real_positive_rate": real_rate,
            "control_positive_rate": control_rate,
            "real_minus_control_rate": real_rate - control_rate,
            "horizons_present": ",".join(str(h) for h in horizons),
            "true_horizon_counterfactual": int(all(int(r.get("true_horizon_counterfactual") or 0) for r in strict_rows)) if strict_rows else 0,
            "per_step_runtime_branch_counterfactual": 0,
            "strict_label_diagnostic_only": 1,
            "B_strict_preparation_signal_pass": int(
                0.05 <= real_rate <= 0.40
                and real_rate - control_rate >= 0.10
                and {50, 100, 200}.issubset(set(horizons))
            ),
            "B_official_preparation_pass": 0,
            "official_blocker": "strict offline label bank is not per-step runtime branch evidence",
        }
    ]
    label_out = _out("v22_18_strict_action_benefit_labels.csv", args.output_suffix)
    summary_out = _out("v22_18_strict_label_bank_summary.csv", args.output_suffix)
    write_rows(label_out, strict_rows)
    write_rows(summary_out, summary)
    cmd = " ".join(shlex.quote(x) for x in [sys.executable, *sys.argv])
    append_exec(
        cmd,
        task_id="B-strict-label-bank",
        status="pass" if summary[0]["B_strict_preparation_signal_pass"] else "partial",
        exit_code=0,
        files=f"{label_out.relative_to(ROOT)}, {summary_out.relative_to(ROOT)}",
        note=(
            f"strict_label={args.label_name}; positives={len(positives)}; "
            f"real_rate={real_rate:.6g}; control_rate={control_rate:.6g}; "
            f"strict_signal_pass={summary[0]['B_strict_preparation_signal_pass']}; diagnostic_only=1"
        ),
    )


if __name__ == "__main__":
    main()
