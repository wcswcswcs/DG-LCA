#!/usr/bin/env python3
"""Audit stricter benefit label definitions against controls."""

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
    p.add_argument("--output-suffix", default="v22_18_branch_counterfactual")
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


def _metric_defs() -> list[tuple[str, Callable[[dict[str, Any]], bool], str]]:
    return [
        ("benefit_positive_existing", lambda r: int(float(r.get("benefit_positive") or 0)) == 1, "existing label: NLL<-0.01 or NLL<=0 and AUC<0"),
        ("true_improvement_existing", lambda r: int(float(r.get("true_improvement_positive") or 0)) == 1, "existing strict label: NLL<-0.01 and AUC<0"),
        ("nll_and_source_positive", lambda r: _nll(r) < 0.0 and _source(r) > 0.0, "delta_NLL<0 and delta_source_loss>0"),
        ("nll_and_auc_positive", lambda r: _nll(r) < 0.0 and _auc(r) < 0.0, "delta_NLL<0 and delta_AUC_loss_time<0"),
        ("nll_auc_source_positive", lambda r: _nll(r) < 0.0 and _auc(r) < 0.0 and _source(r) > 0.0, "delta_NLL<0 and delta_AUC_loss_time<0 and delta_source_loss>0"),
        ("nll_margin_source_positive", lambda r: _nll(r) < -0.001 and _source(r) > 0.0, "delta_NLL<-0.001 and delta_source_loss>0"),
    ]


def main() -> None:
    args = parser().parse_args()
    ensure_out()
    rows = read_rows(_path(args.labels_csv))
    out_rows: list[dict[str, Any]] = []
    for label_name, fn, definition in _metric_defs():
        pos = [r for r in rows if fn(r)]
        real = [r for r in rows if not _is_control(r)]
        control = [r for r in rows if _is_control(r)]
        real_pos = [r for r in real if fn(r)]
        control_pos = [r for r in control if fn(r)]
        real_rate = len(real_pos) / max(1, len(real))
        control_rate = len(control_pos) / max(1, len(control))
        out_rows.append(
            {
                "label_name": label_name,
                "definition": definition,
                "label_rows": len(rows),
                "positive_rows": len(pos),
                "real_rows": len(real),
                "real_positive_rows": len(real_pos),
                "control_rows": len(control),
                "control_positive_rows": len(control_pos),
                "real_positive_rate": real_rate,
                "control_positive_rate": control_rate,
                "real_minus_control_rate": real_rate - control_rate,
                "B_official_gap_pass": int(0.05 <= real_rate <= 0.40 and real_rate - control_rate >= 0.10),
                "blocker": "" if 0.05 <= real_rate <= 0.40 and real_rate - control_rate >= 0.10 else "controls_not_lower_than_real_or_real_rate_out_of_range",
            }
        )
    out_path = _out("v22_18_branch_label_strictness_audit.csv", args.output_suffix)
    write_rows(out_path, out_rows)
    cmd = " ".join(shlex.quote(x) for x in [sys.executable, *sys.argv])
    best = max(out_rows, key=lambda r: float(r["real_minus_control_rate"]), default={})
    append_exec(
        cmd,
        task_id="B-label-strictness-audit",
        status="pass" if any(int(r["B_official_gap_pass"]) for r in out_rows) else "partial",
        exit_code=0,
        files=str(out_path.relative_to(ROOT)),
        note=(
            f"definitions={len(out_rows)}; best_gap={best.get('real_minus_control_rate', '')}; "
            f"best_label={best.get('label_name', '')}; any_gap_pass={int(any(int(r['B_official_gap_pass']) for r in out_rows))}"
        ),
    )


if __name__ == "__main__":
    main()
