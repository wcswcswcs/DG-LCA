#!/usr/bin/env python3
"""Build true-H branch counterfactual labels from separate horizon runs.

The input runs must share dataset, seed, initialization, model list, and data
subsets, while differing only in horizon/steps.  This is still an init-branch
counterfactual rather than a per-step runtime branch, so official status is
reported conservatively.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dgkan.fu.source_action_bank import model_to_action_family  # noqa: E402
from dgkan.fu.treatment_effect_labels import treatment_effect_label  # noqa: E402
from experiments.run_v22_18_common import OUT_ROOT, append_exec, ensure_out, finite_float, read_rows, write_rows  # noqa: E402


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--horizon-csv",
        action="append",
        required=True,
        help="HORIZON:CSV_PATH. Repeat for H50/H100/H200.",
    )
    p.add_argument("--features-csv", default="")
    p.add_argument("--controls-csv", default="")
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


def _base_model(model: str) -> str:
    if model.startswith("DGMLP"):
        return "DGMLP"
    if model.startswith("DGKAN_DCHE") or model.startswith("DGKAN_DFOU"):
        base = model.replace("_FU_NATIVE_JVP_REFRESH", "").replace("_FU_SPLIT_VIRTUAL", "")
        if "_FU" in base:
            return base.split("_FU", 1)[0]
        return base.replace("_NATIVE_ADAMW", "")
    return model


def _parse_horizon(raw: str) -> tuple[int, Path]:
    if ":" not in raw:
        raise ValueError(f"--horizon-csv must be HORIZON:PATH, got {raw!r}")
    horizon, path = raw.split(":", 1)
    return int(horizon), _path(path)


def _sort_key(row: dict[str, Any]) -> tuple[str, str, str, int]:
    return (str(row.get("dataset", "")), str(row.get("seed", "")), str(row.get("model_name", "")), int(row.get("horizon") or 0))


def main() -> None:
    args = parser().parse_args()
    ensure_out()
    horizon_rows: dict[int, list[dict[str, str]]] = {}
    for spec in args.horizon_csv:
        horizon, path = _parse_horizon(spec)
        rows = read_rows(path)
        for row in rows:
            row["_horizon_source_csv"] = str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path)
        horizon_rows[horizon] = rows

    label_rows: list[dict[str, Any]] = []
    branch_matrix: list[dict[str, Any]] = []
    for horizon, rows in sorted(horizon_rows.items()):
        by_key = {(r.get("dataset", ""), str(r.get("seed", "")), r.get("model_name", "")): r for r in rows}
        for row in rows:
            model = row.get("model_name", "")
            base_name = _base_model(model)
            base = by_key.get((row.get("dataset", ""), str(row.get("seed", "")), base_name))
            if not base or model == base.get("model_name"):
                continue
            action = model_to_action_family(model, row.get("source_candidate_type", ""), row.get("controller_policy", ""))
            label = treatment_effect_label(row, base, horizon=horizon)
            label["label_source"] = "true_init_branch_counterfactual"
            label["true_horizon_counterfactual"] = 1
            label_row = {
                "dataset": row.get("dataset", ""),
                "seed": row.get("seed", ""),
                "model_name": model,
                "base_model_name": base.get("model_name", ""),
                "action_id": action.action_id,
                "action_name": action.name,
                "action_is_control": action.is_control,
                **label,
                "horizon_source_csv": row.get("_horizon_source_csv", ""),
            }
            label_rows.append(label_row)
            branch_matrix.append(
                {
                    **label_row,
                    "final_test_loss_NLL": row.get("final_test_loss_NLL", ""),
                    "base_final_test_loss_NLL": base.get("final_test_loss_NLL", ""),
                    "AUC_loss_time": row.get("AUC_loss_time", ""),
                    "base_AUC_loss_time": base.get("AUC_loss_time", ""),
                    "controller_overhead_ratio": row.get("controller_overhead_ratio", ""),
                    "train_size": row.get("train_size", ""),
                    "test_size": row.get("test_size", ""),
                    "steps": row.get("steps", ""),
                }
            )

    real_labels = [r for r in label_rows if not int(r.get("action_is_control") or 0)]
    control_labels = [r for r in label_rows if int(r.get("action_is_control") or 0)]
    positives = [r for r in label_rows if int(r.get("benefit_positive") or 0)]
    real_positive_rate = len([r for r in real_labels if int(r.get("benefit_positive") or 0)]) / max(1, len(real_labels))
    control_positive_rate = len([r for r in control_labels if int(r.get("benefit_positive") or 0)]) / max(1, len(control_labels))
    horizons_present = sorted({int(r.get("horizon") or 0) for r in label_rows})
    action_families = sorted({str(r.get("action_id", "")) for r in label_rows})
    if not label_rows:
        blocker = "no branch labels"
    elif control_positive_rate >= real_positive_rate:
        blocker = "true branch labels generated, but controls positive rate is not lower than real action rate"
    elif not {50, 100, 200}.issubset(set(horizons_present)):
        blocker = "missing required H50/H100/H200 branch labels"
    else:
        blocker = "init-branch counterfactual only; not per-step runtime branch labels"
    summary = [
        {
            "source": "v22_18_true_init_branch_counterfactual",
            "final_rows": sum(len(v) for v in horizon_rows.values()),
            "runtime_source_rows": len(read_rows(_path(args.features_csv))) if args.features_csv else 0,
            "neutral_control_rows": len(read_rows(_path(args.controls_csv))) if args.controls_csv else 0,
            "action_families_present": len(action_families),
            "action_families_with_nonzero_accepts": len(action_families),
            "benefit_label_rows": len(label_rows),
            "positive_benefit_rows": len(positives),
            "real_positive_benefit_rate": real_positive_rate,
            "control_positive_benefit_rate": control_positive_rate,
            "horizons_present": ",".join(str(h) for h in horizons_present),
            "true_horizon_counterfactual": 1,
            "per_step_runtime_branch_counterfactual": 0,
            "B_exploration_pass": int(
                len(action_families) >= 4
                and len(positives) > 0
                and len(positives) < len(label_rows)
                and bool(control_labels)
            ),
            "B_official_preparation_pass": int(
                0.05 <= real_positive_rate <= 0.40
                and real_positive_rate - control_positive_rate >= 0.10
                and {50, 100, 200}.issubset(set(horizons_present))
            ),
            "official_blocker": blocker,
        }
    ]

    label_rows = sorted(label_rows, key=_sort_key)
    branch_matrix = sorted(branch_matrix, key=_sort_key)
    write_rows(_out("v22_18_action_benefit_labels_H50_H100_H200.csv", args.output_suffix), label_rows)
    write_rows(_out("v22_18_branch_counterfactual_matrix.csv", args.output_suffix), branch_matrix)
    write_rows(_out("v22_18_source_action_bank_summary.csv", args.output_suffix), summary)
    cmd = " ".join(shlex.quote(x) for x in [sys.executable, *sys.argv])
    append_exec(
        cmd,
        task_id="B-true-branch-label-readback",
        status="pass" if summary[0]["B_exploration_pass"] else "partial",
        exit_code=0,
        files=(
            f"{_out('v22_18_action_benefit_labels_H50_H100_H200.csv', args.output_suffix).relative_to(ROOT)}, "
            f"{_out('v22_18_branch_counterfactual_matrix.csv', args.output_suffix).relative_to(ROOT)}, "
            f"{_out('v22_18_source_action_bank_summary.csv', args.output_suffix).relative_to(ROOT)}"
        ),
        note=(
            f"labels={len(label_rows)}; positives={len(positives)}; "
            f"real_rate={real_positive_rate:.6g}; control_rate={control_positive_rate:.6g}; "
            f"B_official={summary[0]['B_official_preparation_pass']}; blocker={blocker}"
        ),
    )


if __name__ == "__main__":
    main()
