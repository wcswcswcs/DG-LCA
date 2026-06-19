#!/usr/bin/env python3
"""Build v22.18 source/action bank and benefit labels from real run artifacts."""

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

from dgkan.fu.source_action_bank import ACTION_FAMILIES, RUNTIME_FEATURES, model_to_action_family, runtime_feature_row  # noqa: E402
from dgkan.fu.treatment_effect_labels import treatment_effect_label  # noqa: E402
from experiments.run_v22_18_common import (  # noqa: E402
    OUT_ROOT,
    append_exec,
    ensure_out,
    finite_float,
    read_rows,
    write_rows,
)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--bridge-csv", default="")
    p.add_argument("--source-csv", default="")
    p.add_argument("--neutral-control-csv", default="")
    p.add_argument("--kan-native-csv", default="")
    p.add_argument("--horizons", default="50,100,200")
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


def _base_model(model: str) -> str:
    if model.startswith("DGMLP") or model.startswith("MLP+"):
        return "DGMLP"
    if model.startswith("DGKAN_DCHE") or model.startswith("DGKAN_DFOU"):
        base = model.replace("_FU_NATIVE_JVP_REFRESH", "").replace("_FU_SPLIT_VIRTUAL", "")
        if "_FU" in base:
            return base.split("_FU", 1)[0]
        return base.replace("_NATIVE_ADAMW", "")
    return model


def _read_optional(raw: str) -> list[dict[str, str]]:
    if not raw:
        return []
    return read_rows(_path(raw))


def _group_mean(rows: list[dict[str, Any]], fields: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for field in fields:
        vals = [finite_float(r.get(field)) for r in rows if r.get(field) not in {"", None}]
        out[field] = sum(vals) / max(1, len(vals)) if vals else 0.0
    return out


def main() -> None:
    args = parser().parse_args()
    ensure_out()
    final_rows = _read_optional(args.bridge_csv)
    native_rows = _read_optional(args.kan_native_csv)
    if native_rows:
        final_rows.extend(native_rows)
    source_rows = _read_optional(args.source_csv)
    neutral_rows = _read_optional(args.neutral_control_csv)
    horizons = [int(x.strip()) for x in args.horizons.split(",") if x.strip()]

    source_by_key: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in source_rows:
        source_by_key[(row.get("dataset", ""), str(row.get("seed", "")), row.get("model_name", ""))].append(row)

    final_by_key = {(r.get("dataset", ""), str(r.get("seed", "")), r.get("model_name", "")): r for r in final_rows}
    bank_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []
    controls_rows: list[dict[str, Any]] = []

    for row in final_rows:
        model = row.get("model_name", "")
        action = model_to_action_family(model, row.get("source_candidate_type", ""), row.get("controller_policy", ""))
        key = (row.get("dataset", ""), str(row.get("seed", "")), model)
        src = source_by_key.get(key, [])
        mean_features = _group_mean(src, RUNTIME_FEATURES)
        if not src:
            mean_features = runtime_feature_row(row, action)
        # action_family_numeric is an action identity feature, not a source-row
        # measurement.  Source matrices usually do not contain it, so preserve
        # the candidate action id instead of averaging missing values to zero.
        mean_features["action_family_numeric"] = runtime_feature_row(row, action)["action_family_numeric"]
        bank = {
            "dataset": row.get("dataset", ""),
            "seed": row.get("seed", ""),
            "model_name": model,
            "action_id": action.action_id,
            "action_name": action.name,
            "action_is_control": action.is_control,
            "controller_policy": row.get("controller_policy", ""),
            "source_candidate_type": row.get("source_candidate_type", ""),
            "accepted_candidates": row.get("gate_accept_count", ""),
            "rejected_candidates": row.get("gate_reject_count", ""),
            "intervention_count": row.get("intervention_count", ""),
            "runtime_feature_rows": len(src),
            "final_test_loss_NLL": row.get("final_test_loss_NLL", ""),
            "final_test_accuracy": row.get("final_test_accuracy", ""),
            "AUC_loss_time": row.get("AUC_loss_time", ""),
            "controller_overhead_ratio": row.get("controller_overhead_ratio", ""),
            "label_runtime_firewall": "runtime rows exclude final/test/AUC labels; labels are joined offline only",
        }
        bank_rows.append(bank)
        feature_row = {
            "dataset": row.get("dataset", ""),
            "seed": row.get("seed", ""),
            "model_name": model,
            "action_id": action.action_id,
            "action_name": action.name,
            "action_is_control": action.is_control,
            **mean_features,
            "analysis_only_future_label_used_for_runtime_direction": 0,
        }
        feature_rows.append(feature_row)
        if action.is_control:
            controls_rows.append({**bank, "control_source": "final_model_variant"})
        base = final_by_key.get((row.get("dataset", ""), str(row.get("seed", "")), _base_model(model)))
        if base and model != base.get("model_name"):
            for horizon in horizons:
                label = treatment_effect_label(row, base, horizon=horizon)
                label_rows.append(
                    {
                        "dataset": row.get("dataset", ""),
                        "seed": row.get("seed", ""),
                        "model_name": model,
                        "base_model_name": base.get("model_name", ""),
                        "action_id": action.action_id,
                        "action_name": action.name,
                        "action_is_control": action.is_control,
                        **label,
                    }
                )

    for row in neutral_rows:
        action = model_to_action_family(row.get("model_name", ""), row.get("source_candidate_type", ""), row.get("controller_policy", ""))
        controls_rows.append(
            {
                "dataset": row.get("dataset", ""),
                "seed": row.get("seed", ""),
                "model_name": row.get("model_name", ""),
                "source_id": row.get("source_id", ""),
                "action_id": action.action_id,
                "action_name": action.name,
                "action_is_control": 1,
                "neutral_control_source_type": row.get("neutral_control_source_type", ""),
                "source_loss_t": row.get("source_loss_t", ""),
                "U_task_usefulness": row.get("U_task_usefulness", ""),
                "control_source": "neutral_source_row",
            }
        )

    firewall_rows = []
    for feature in RUNTIME_FEATURES:
        firewall_rows.append({"field": feature, "table": "v22_18_action_features_runtime_only.csv", "uses_future_test_or_validation": 0})
    for field in ["final_test_loss_NLL", "AUC_loss_time", "benefit_utility", "benefit_positive", "delta_NLL", "delta_AUC_loss_time"]:
        firewall_rows.append({"field": field, "table": "v22_18_action_benefit_labels_H50_H100_H200.csv", "uses_future_test_or_validation": 1})

    action_counts: dict[str, int] = defaultdict(int)
    accepted_nonzero: dict[str, int] = defaultdict(int)
    for row in bank_rows:
        action_counts[str(row["action_id"])] += 1
        if finite_float(row.get("accepted_candidates")) > 0 or finite_float(row.get("intervention_count")) > 0 or row["action_id"] == "A0":
            accepted_nonzero[str(row["action_id"])] += 1
    positives = [r for r in label_rows if int(r.get("benefit_positive") or 0)]
    real_labels = [r for r in label_rows if not int(r.get("action_is_control") or 0)]
    control_labels = [r for r in label_rows if int(r.get("action_is_control") or 0)]
    real_positive_rate = len([r for r in real_labels if int(r.get("benefit_positive") or 0)]) / max(1, len(real_labels))
    control_positive_rate = len([r for r in control_labels if int(r.get("benefit_positive") or 0)]) / max(1, len(control_labels))
    summary = [
        {
            "source": "v22_18_source_action_bank",
            "final_rows": len(final_rows),
            "runtime_source_rows": len(source_rows),
            "neutral_control_rows": len(neutral_rows),
            "action_families_present": len(action_counts),
            "action_families_with_nonzero_accepts": len(accepted_nonzero),
            "benefit_label_rows": len(label_rows),
            "positive_benefit_rows": len(positives),
            "real_positive_benefit_rate": real_positive_rate,
            "control_positive_benefit_rate": control_positive_rate,
            "B_exploration_pass": int(
                len(accepted_nonzero) >= 4
                and len(positives) > 0
                and len(positives) < len(label_rows)
                and bool(control_labels)
            ),
            "B_official_preparation_pass": int(
                0.05 <= real_positive_rate <= 0.40
                and real_positive_rate - control_positive_rate >= 0.10
                and {50, 100, 200}.issubset({int(r.get("horizon") or 0) for r in label_rows})
            ),
            "official_blocker": "labels are final-run delta proxy, not true H-branch counterfactuals" if label_rows else "no labels",
        }
    ]

    write_rows(_out("v22_18_source_action_bank.csv", args.output_suffix), bank_rows)
    write_rows(_out("v22_18_action_benefit_labels_H50_H100_H200.csv", args.output_suffix), label_rows)
    write_rows(_out("v22_18_action_features_runtime_only.csv", args.output_suffix), feature_rows)
    write_rows(_out("v22_18_action_controls_matrix.csv", args.output_suffix), controls_rows)
    write_rows(_out("v22_18_action_label_leakage_firewall.csv", args.output_suffix), firewall_rows)
    write_rows(_out("v22_18_source_action_bank_summary.csv", args.output_suffix), summary)
    write_rows(OUT_ROOT / "v22_18_action_family_dictionary.csv", [family.__dict__ for family in ACTION_FAMILIES])

    cmd = " ".join(shlex.quote(x) for x in [sys.executable, *sys.argv])
    append_exec(
        cmd,
        task_id="B-source-action-bank",
        status="pass" if summary[0]["B_exploration_pass"] else "partial",
        exit_code=0,
        files=(
            f"{_out('v22_18_source_action_bank.csv', args.output_suffix).relative_to(ROOT)}, "
            f"{_out('v22_18_action_benefit_labels_H50_H100_H200.csv', args.output_suffix).relative_to(ROOT)}, "
            f"{_out('v22_18_action_features_runtime_only.csv', args.output_suffix).relative_to(ROOT)}, "
            f"{_out('v22_18_action_controls_matrix.csv', args.output_suffix).relative_to(ROOT)}"
        ),
        note=f"labels={len(label_rows)}; positives={len(positives)}; {summary[0]['official_blocker']}",
    )


if __name__ == "__main__":
    main()
